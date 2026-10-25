"""Guarded 256-user feature build, permitted only after P003 exact equivalence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__:
    from .run_feature_benchmark import feature_build
    from .run_pilot import CandidateGenerator, ResourceMonitor, load_mapping, sha256, write_json
    from .run_retrieval import load_development_events, QUOTAS
    from .pilot_data import _manifest_hashes
    from .scale_data import prepare_scale, EVENT_SOURCE, ARTIST_SOURCE, ALBUM_SOURCE
else:
    from run_feature_benchmark import feature_build
    from run_pilot import CandidateGenerator, ResourceMonitor, load_mapping, sha256, write_json
    from run_retrieval import load_development_events, QUOTAS
    from pilot_data import _manifest_hashes
    from scale_data import prepare_scale, EVENT_SOURCE, ARTIST_SOURCE, ALBUM_SOURCE

TASK = Path(__file__).resolve().parents[1]


def select_fixed_queries(events: dict, config: dict) -> tuple[list[dict], dict]:
    """Sample unambiguous listens using frozen boundaries, without reading reserved times."""
    boundaries = config["boundaries_raw"]
    listens = np.flatnonzero(events["event_code"] == events["event_names"].index("listen"))
    per_user = defaultdict(list)
    ambiguous = Counter()
    current, matches, first = None, 0, None
    def finish_group():
        if current is None:
            return
        uid, timestamp = current
        if not boundaries["train_query_start"] <= timestamp < boundaries["validation_end"]:
            return
        split = "train" if timestamp < boundaries["train_end"] else "validation"
        if matches != 1:
            ambiguous[split] += 1
        else:
            per_user[(uid, split)].append(first)
    for index in listens:
        index = int(index)
        key = (int(events["uid"][index]), int(events["timestamp"][index]))
        if key != current:
            finish_group()
            current, matches, first = key, 1, index
        else:
            matches += 1
    finish_group()
    rng = np.random.default_rng(config["seed"])
    selected, available = [], Counter()
    for (uid, split), indices in sorted(per_user.items()):
        available[split] += len(indices)
        limit = config[f"{split}_queries_per_user"]
        for index in rng.choice(indices, size=min(limit, len(indices)), replace=False):
            selected.append({"user_id": uid, "timestamp": int(events["timestamp"][index]),
                             "target_item": int(events["item_id"][index]), "split": split})
    selected.sort(key=lambda q: (q["timestamp"], q["user_id"], q["target_item"]))
    for index, query in enumerate(selected):
        query["query_id"] = index
    return selected, {"sampled": dict(Counter(q["split"] for q in selected)),
                      "eligible": dict(available), "excluded_ambiguous_listen_groups": dict(ambiguous),
                      "boundaries_raw": boundaries}


def seal_scale_candidates(events: dict, queries: list[dict], artists: dict, albums: dict,
                          config: dict, output: Path, monitor) -> dict:
    # Same operational algorithm as P002 repeat_heavy; omit unused depth200 diagnostics.
    generator = CandidateGenerator(artists, albums, {"max_candidates": config["max_candidates"],
                                                    "candidate_quotas": QUOTAS["repeat_heavy"]})
    index = 0
    sizes = []
    tick = time.perf_counter()
    next_log = tick + 15
    with (output / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for position, query in enumerate(queries):
            while index < len(events["timestamp"]) and int(events["timestamp"][index]) < query["timestamp"]:
                generator.observe({"uid": int(events["uid"][index]), "item_id": int(events["item_id"][index]),
                                   "timestamp": int(events["timestamp"][index]),
                                   "event_type": events["event_names"][int(events["event_code"][index])]})
                index += 1
                if index % 5000 == 0:
                    monitor.check()
            candidates = generator.candidates(query["user_id"], query["timestamp"])
            if len(set(candidates)) != len(candidates) or any(item not in generator.catalog for item in candidates):
                raise AssertionError("Invalid scale candidates")
            query["candidate_items"] = candidates
            sizes.append(len(candidates))
            # Labels remain in the provenance file only, after the candidates have been fixed.
            stream.write(json.dumps(query, separators=(",", ":")) + "\n")
            if position % 20 == 0:
                monitor.check()
            if time.perf_counter() >= next_log:
                print(f"SCALE CANDIDATES {position+1}/{len(queries)} queries, {index:,} past events", flush=True)
                next_log = time.perf_counter() + 15
    return {"strategy": "repeat_heavy", "implementation": "P001 CandidateGenerator with fixed P002 repeat-heavy quotas; same profile5/prefix100",
            "quotas": QUOTAS["repeat_heavy"], "elapsed_seconds": round(time.perf_counter() - tick, 6),
            "past_events_observed": index, "candidate_rows": sum(sizes), "minimum_candidates": min(sizes),
            "maximum_candidates": max(sizes), "queries_sha256": sha256(output / "queries.jsonl")}


def validate_equivalence_gate(path: Path, config_sha: str) -> dict:
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate["status"] != "pass" or not gate["numpy_array_equal_all_cells"] or not gate["query_candidate_order_identical"] or not gate["feature_schema_identical"]:
        raise ValueError("Scale requires successful full exact equivalence")
    if gate["cells_compared"] != 184800 * 92 or gate["shape"] != [184800, 92] or gate["dtype"] != "float32":
        raise ValueError("Scale gate did not compare all expected cells")
    original = json.loads((TASK / "report/pilot_001_snapshot/metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((TASK / "report/pilot_001_snapshot/checkpoint_manifest.json").read_text(encoding="utf-8"))
    expected_inputs = {f"artifacts/pilot_001/data/{name}": digest
                       for name, digest in original["prepared_data_sha256"].items()}
    query_name = "artifacts/pilot_001/queries.jsonl"
    expected_inputs[query_name] = manifest["archived_files"][f"task_01_auto_features/{query_name}"]["sha256"]
    expected_sources = {"run_feature_benchmark.py", "pilot_features.py", "run_retrieval.py", "run_pilot.py", "pilot_storage.py"}
    structural_hashes, schema_hashes = [], []
    evidence = []
    for index, side in enumerate(("left", "right")):
        folder = Path(gate[side])
        status = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        if status["status"] != "complete" or status["config_sha256"] != config_sha:
            raise ValueError("Equivalence gate uses different config or incomplete build")
        if status["engine"] != ("baseline" if side == "left" else "optimized") or status["profiled"]:
            raise ValueError("Scale requires fresh unprofiled baseline and optimized builds")
        source_keys = expected_sources | ({"optimized_features.py"} if side == "right" else set())
        if set(status["source_code_sha256"]) != source_keys or status["input_sha256"] != expected_inputs:
            raise ValueError("Equivalence lacks expected frozen sources/inputs")
        if status["source_code_sha256"]["pilot_features.py"] != original["source_code_sha256"]["pilot_features.py"]:
            raise ValueError("Baseline feature source differs from frozen P001")
        if status["protocol_sha256"] != sha256(TASK / "configs/P003_PROTOCOL.md"):
            raise ValueError("Equivalence protocol changed")
        for name, expected in status["source_code_sha256"].items():
            if sha256(TASK / "src" / name) != expected:
                raise ValueError(f"Equivalence runtime source changed: {name}")
        if sha256(folder / "features.npy") != gate["matrix_file_sha256"][index]:
            raise ValueError("Equivalence matrix changed after verification")
        for filename, field, hashes in (("feature_inputs.jsonl", "feature_inputs_sha256", structural_hashes),
                                        ("feature_schema.json", "feature_schema_sha256", schema_hashes)):
            digest = sha256(folder / filename)
            if digest != status["build"][field]:
                raise ValueError(f"Equivalence input/schema changed: {filename}")
            hashes.append(digest)
        evidence.append({"engine": status["engine"], "source_code_sha256": status["source_code_sha256"],
                         "input_sha256": status["input_sha256"], "feature_inputs_sha256": structural_hashes[-1],
                         "feature_schema_sha256": schema_hashes[-1]})
    if len(set(structural_hashes)) != 1 or len(set(schema_hashes)) != 1:
        raise ValueError("Equivalence structure/schema differ")
    return {"path": str(path), "sha256": sha256(path), "cells_compared": gate["cells_compared"], "status": "pass",
            "verified_provenance": evidence}


def main() -> None:
    if __package__:
        from .optimized_features import OptimizedFeatureEngine
    else:
        from optimized_features import OptimizedFeatureEngine
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TASK / "artifacts/p003_scale256")
    parser.add_argument("--config", type=Path, default=TASK / "configs/p003_features.json")
    parser.add_argument("--gate", type=Path, default=TASK / "artifacts/p003_equivalence.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["test_evaluation"] or config["model_training"] or config["scale_users"] != 256 or config["scale_candidate_strategy"] != "repeat_heavy":
        raise ValueError("Scale configuration differs from fixed P003")
    gate = validate_equivalence_gate(args.gate, sha256(args.config))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Scale output must be a fresh directory")
    output.mkdir(parents=True)
    monitor = ResourceMonitor(config["resource_limits"])
    monitor.start()
    started = time.perf_counter()
    sources = ["run_feature_scale.py", "scale_data.py", "run_feature_benchmark.py", "optimized_features.py",
               "pilot_features.py", "pilot_data.py", "run_pilot.py", "run_retrieval.py"]
    status = {"run_id": "p003_scale256", "status": "running", "config": config,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256(args.config),
              "protocol_sha256": sha256(TASK / "configs/P003_PROTOCOL.md"), "equivalence_gate": gate,
              "source_code_sha256": {name: sha256(TASK / "src" / name) for name in sources},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TASK, text=True).strip(),
              "command": sys.argv, "final_test_evaluated": False, "model_training": False, "build_progress": {}}
    write_json(output / "run_status.json", status)
    try:
        monitor.check()
        expected = _manifest_hashes()
        status["source_data_sha256"] = {}
        for source, relative in ((EVENT_SOURCE, "yambda/flat/50m/multi_event.parquet"),
                                  (ARTIST_SOURCE, "yambda/artist_item_mapping.parquet"),
                                  (ALBUM_SOURCE, "yambda/album_item_mapping.parquet")):
            digest = sha256(source)
            if digest != expected[relative]:
                raise ValueError(f"Original dataset changed: {source.name}")
            status["source_data_sha256"][relative] = digest
            monitor.check()
        write_json(output / "run_status.json", status)
        tick = time.perf_counter()
        metadata = prepare_scale(output / "data", config["scale_users"], config["seed"],
                                 config["boundaries_raw"]["validation_end"], monitor=monitor)
        status["data_preparation_seconds"] = round(time.perf_counter() - tick, 6)
        if metadata["development"]["chronology"]["violations"]:
            raise ValueError("Development chronology violations")
        original_metadata = json.loads((TASK / "report/pilot_001_snapshot/data/metadata.json").read_text(encoding="utf-8"))
        status["cohort"] = {"selected_users": metadata["selected_uid_count"], "seed": config["seed"],
                            "overlap_with_P001_users": len(set(metadata["selected_users"]) & set(original_metadata["selected_users"])),
                            "full_history_rows_stored": metadata["full_history_retention"]["stored_row_count_arrow"],
                            "development_rows": metadata["development"]["rows"],
                            "development_unique_items": metadata["development"]["unique_items"]}
        status["prepared_input_sha256"] = {name: sha256(output / "data" / name) for name in
                                           ("events_full.parquet", "events.parquet", "artist_mapping.parquet", "album_mapping.parquet", "metadata.json")}
        write_json(output / "run_status.json", status)
        tick = time.perf_counter()
        events = load_development_events(output / "data/events.parquet", config["boundaries_raw"]["validation_end"])
        artists = load_mapping(output / "data/artist_mapping.parquet", "artist_id")
        albums = load_mapping(output / "data/album_mapping.parquet", "album_id")
        queries, query_info = select_fixed_queries(events, config)
        if not queries:
            raise ValueError("No eligible scale queries")
        status["query_selection"] = query_info
        status["load_and_query_selection_seconds"] = round(time.perf_counter() - tick, 6)
        status["candidates"] = seal_scale_candidates(events, queries, artists, albums, config, output, monitor)
        gc.collect()
        monitor.check()
        write_json(output / "run_status.json", status)
        status["build"] = feature_build(OptimizedFeatureEngine, events, queries, artists, albums, config,
                                         output, monitor, progress=status["build_progress"])
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
    print(f"P003 SCALE COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
