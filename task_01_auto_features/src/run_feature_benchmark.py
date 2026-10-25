"""Profile and benchmark fixed feature values; never train or evaluate targets."""
from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
from pathlib import Path
import pstats
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

if __package__:
    from .pilot_features import FeatureEngine
    from .pilot_storage import ensure_run_writable
    from .run_pilot import ResourceMonitor, load_mapping, sha256, write_json
    from .run_retrieval import load_development_events, load_frozen_queries
else:
    from pilot_features import FeatureEngine
    from pilot_storage import ensure_run_writable
    from run_pilot import ResourceMonitor, load_mapping, sha256, write_json
    from run_retrieval import load_development_events, load_frozen_queries

TASK = Path(__file__).resolve().parents[1]


def profile_table(profiler: cProfile.Profile) -> list[dict]:
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        rows.append({"file": filename, "line": line, "function": function, "primitive_calls": primitive,
                     "calls": calls, "own_seconds": own, "cumulative_seconds": cumulative})
    return sorted(rows, key=lambda row: (-row["cumulative_seconds"], row["file"], row["line"]))


def feature_build(engine_class, events: dict, queries: list[dict], artists: dict, albums: dict,
                  config: dict, output: Path, monitor, profile: bool = False,
                  progress: dict | None = None) -> dict:
    """Materialize identical feature rows incrementally without targets entering features."""
    progress = progress if progress is not None else {}
    engine = engine_class(artists, albums, config["short_window_raw"], config["half_life_raw"])
    if len(engine.feature_names) != config["feature_count"]:
        raise ValueError("Unexpected feature schema")
    rows = sum(len(q["candidate_items"]) for q in queries)
    if not rows:
        raise ValueError("A feature build needs at least one candidate")
    structural = hashlib.sha256()
    with (output / "feature_inputs.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        offset = 0
        for q in queries:
            inputs = {"query_id": q["query_id"], "user_id": q["user_id"], "timestamp": q["timestamp"],
                      "offset": offset, "size": len(q["candidate_items"]), "candidate_items": q["candidate_items"]}
            line = json.dumps(inputs, separators=(",", ":")) + "\n"
            structural.update(line.encode("utf-8"))
            stream.write(line)
            offset += inputs["size"]
    write_json(output / "feature_schema.json", {"feature_names": engine.feature_names, "feature_groups": engine.feature_groups})
    matrix = np.lib.format.open_memmap(output / "features.npy", mode="w+", dtype=np.float32,
                                       shape=(rows, len(engine.feature_names)))
    timers = {"observe_seconds": 0.0, "transform_seconds": 0.0, "materialize_seconds": 0.0}
    profiler = cProfile.Profile() if profile else None
    started = time.perf_counter()
    next_log = started + 15
    index, offset = 0, 0
    if profiler:
        profiler.enable()
    try:
        for position, query in enumerate(queries):
            tick = time.perf_counter()
            while index < len(events["timestamp"]) and int(events["timestamp"][index]) < query["timestamp"]:
                engine.observe({"uid": int(events["uid"][index]), "item_id": int(events["item_id"][index]),
                                "timestamp": int(events["timestamp"][index]),
                                "event_type": events["event_names"][int(events["event_code"][index])]})
                index += 1
                if index % 5000 == 0:
                    monitor.check()
            timers["observe_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
            values = engine.transform(query["user_id"], query["candidate_items"], query["timestamp"])
            timers["transform_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
            if values.shape != (len(query["candidate_items"]), len(engine.feature_names)) or values.dtype != np.float32:
                raise ValueError("Feature output shape/dtype mismatch")
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite feature value")
            matrix[offset:offset + len(values)] = values
            offset += len(values)
            timers["materialize_seconds"] += time.perf_counter() - tick
            progress.update(queries_completed=position + 1, rows_written=offset, events_observed=index)
            if position % 20 == 0:
                monitor.check()
            if time.perf_counter() >= next_log:
                print(f"FEATURE BUILD {engine_class.__name__}: {position+1}/{len(queries)} queries, {index:,} past events", flush=True)
                next_log = time.perf_counter() + 15
        tick = time.perf_counter()
        matrix.flush()
        timers["materialize_seconds"] += time.perf_counter() - tick
    finally:
        # A failed run remains explicitly partial, but dirty mapped pages are preserved.
        try:
            matrix.flush()
        except OSError:
            pass
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(output / "profile.pstats"))
            write_json(output / "profile_summary.json", profile_table(profiler))
    replay_seconds = time.perf_counter() - started
    values_digest = hashlib.sha256(memoryview(matrix).cast("B")).hexdigest()
    result = {"shape": list(matrix.shape), "dtype": str(matrix.dtype), "matrix_values_bytes": matrix.nbytes,
              "matrix_file_bytes": (output / "features.npy").stat().st_size,
              "matrix_values_sha256": values_digest, "matrix_file_sha256": sha256(output / "features.npy"),
              "feature_inputs_sha256": structural.hexdigest(), "feature_schema_sha256": sha256(output / "feature_schema.json"),
              "queries": len(queries), "past_events_observed": index, "profiled": profile,
              "replay_seconds": round(replay_seconds, 6),
              "phase_timers": {name: round(value, 6) for name, value in timers.items()}}
    del engine, matrix
    gc.collect()
    monitor.check()
    return result


def verify_builds(left: Path, right: Path, output: Path) -> dict:
    """Compare every float32 cell plus ordering/schema, using mmap chunks."""
    if output.exists():
        raise FileExistsError("Equivalence evidence must use a fresh filename")
    lstat = json.loads((left / "metrics.json").read_text(encoding="utf-8"))
    rstat = json.loads((right / "metrics.json").read_text(encoding="utf-8"))
    if lstat["status"] != "complete" or rstat["status"] != "complete":
        raise ValueError("Only completed builds can be compared")
    for field in ("config_sha256", "input_sha256"):
        if lstat[field] != rstat[field]:
            raise ValueError(f"Builds have different {field}")
    for name in lstat.get("source_code_sha256", {}).keys() & rstat.get("source_code_sha256", {}).keys():
        if lstat["source_code_sha256"][name] != rstat["source_code_sha256"][name]:
            raise ValueError(f"Shared benchmark source changed between builds: {name}")
    a, b = np.load(left / "features.npy", mmap_mode="r"), np.load(right / "features.npy", mmap_mode="r")
    same = a.shape == b.shape and a.dtype == b.dtype == np.float32
    checked = 0
    if same:
        for start in range(0, len(a), 2048):
            if not np.array_equal(a[start:start+2048], b[start:start+2048]):
                same = False
                break
            checked += a[start:start+2048].size
    actual_values = [hashlib.sha256(memoryview(matrix).cast("B")).hexdigest() for matrix in (a, b)]
    actual_files = [sha256(folder / "features.npy") for folder in (left, right)]
    for index, (folder, saved) in enumerate(((left, lstat), (right, rstat))):
        if actual_values[index] != saved["build"]["matrix_values_sha256"] or actual_files[index] != saved["build"]["matrix_file_sha256"]:
            raise ValueError(f"Saved matrix digest changed: {folder}")
    structure_match = (left / "feature_inputs.jsonl").read_bytes() == (right / "feature_inputs.jsonl").read_bytes()
    schema_match = (left / "feature_schema.json").read_bytes() == (right / "feature_schema.json").read_bytes()
    result = {"status": "pass" if same and structure_match and schema_match and len(set(actual_values)) == 1 and len(set(actual_files)) == 1 else "fail",
              "left": str(left), "right": str(right), "shape": list(a.shape), "dtype": str(a.dtype),
              "cells_compared": checked, "numpy_array_equal_all_cells": same,
              "query_candidate_order_identical": structure_match, "feature_schema_identical": schema_match,
              "matrix_values_sha256": actual_values, "matrix_file_sha256": actual_files}
    write_json(output, result)
    if result["status"] != "pass":
        raise AssertionError("Feature equivalence failed; scale stage is forbidden")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["baseline", "optimized"], default="baseline")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=TASK / "configs/p003_features.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["test_evaluation"] or config["model_training"]:
        raise ValueError("No test evaluation or training in P003")
    output = args.output.resolve()
    ensure_run_writable(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new subrun/output directory")
    output.mkdir(parents=True)
    engine_class = FeatureEngine
    if args.engine == "optimized":
        if __package__:
            from .optimized_features import OptimizedFeatureEngine
        else:
            from optimized_features import OptimizedFeatureEngine
        engine_class = OptimizedFeatureEngine
    monitor = ResourceMonitor(config["resource_limits"])
    monitor.start()
    started = time.perf_counter()
    sources = ["run_feature_benchmark.py", "pilot_features.py", "run_retrieval.py", "run_pilot.py", "pilot_storage.py"]
    if args.engine == "optimized":
        sources.append("optimized_features.py")
    status = {"run_id": output.name, "status": "running", "engine": args.engine, "profiled": args.profile,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
              "config_sha256": sha256(args.config), "protocol_sha256": sha256(TASK / "configs/P003_PROTOCOL.md"),
              "source_code_sha256": {name: sha256(TASK / "src" / name) for name in sources},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TASK, text=True).strip(),
              "command": sys.argv, "model_training": False, "final_test_evaluated": False, "build_progress": {}}
    write_json(output / "run_status.json", status)
    try:
        monitor.check()
        original = json.loads((TASK / "report/pilot_001_snapshot/metrics.json").read_text(encoding="utf-8"))
        if sha256(TASK / "src/pilot_features.py") != original["source_code_sha256"]["pilot_features.py"]:
            raise ValueError("Baseline feature source differs from frozen P001")
        if config["boundaries_raw"] != original["boundaries_raw"]:
            raise ValueError("P003 time boundaries differ from P001")
        manifest = json.loads((TASK / "report/pilot_001_snapshot/checkpoint_manifest.json").read_text(encoding="utf-8"))
        queries_path = TASK / config["queries_path"]
        digest = sha256(queries_path)
        if digest != manifest["archived_files"][queries_path.relative_to(TASK.parent).as_posix()]["sha256"]:
            raise ValueError("Frozen queries changed")
        status["input_sha256"] = {config["queries_path"]: digest}
        data_dir = TASK / config["data_dir"]
        for name, expected in original["prepared_data_sha256"].items():
            actual = sha256(data_dir / name)
            if actual != expected:
                raise ValueError(f"Prepared P001 file changed: {name}")
            status["input_sha256"][f"{config['data_dir']}/{name}"] = actual
        write_json(output / "run_status.json", status)
        tick = time.perf_counter()
        queries = load_frozen_queries(queries_path, config)
        events = load_development_events(data_dir / "events.parquet", config["boundaries_raw"]["validation_end"])
        artists = load_mapping(data_dir / "artist_mapping.parquet", "artist_id")
        albums = load_mapping(data_dir / "album_mapping.parquet", "album_id")
        status["load_seconds"] = round(time.perf_counter() - tick, 6)
        status["development_rows_loaded"] = len(events["timestamp"])
        print(f"P003 {args.engine} profile={args.profile}: {len(queries)} fixed queries, development events {len(events['timestamp']):,}", flush=True)
        status["build"] = feature_build(engine_class, events, queries, artists, albums, config,
                                         output, monitor, args.profile, status["build_progress"])
        monitor.check()
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        status["resources"] = monitor.finish()
        if status["resources"]["limit_exceeded"]:
            status.update(status="failed", error=status["resources"]["limit_exceeded"])
        write_json(output / "run_status.json", status)
        write_json(output / "metrics.json", status)
    if status["status"] != "complete":
        raise MemoryError(status["error"])
    print(f"P003 BUILD COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
