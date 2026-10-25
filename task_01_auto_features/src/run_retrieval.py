"""Run the frozen P002 retrieval experiment on saved P001 development queries."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if __package__:
    from .run_pilot import ResourceMonitor, load_mapping, sha256, write_json
    from .pilot_storage import ensure_run_writable
else:
    from run_pilot import ResourceMonitor, load_mapping, sha256, write_json
    from pilot_storage import ensure_run_writable

TASK = Path(__file__).resolve().parents[1]
SOURCES = ("popular", "repeat", "artist", "album")
STRATEGIES = ("baseline", "repeat_heavy", "affinity", "rank_fusion")
QUOTAS = {
    "baseline": dict(zip(SOURCES, (25, 25, 35, 15))),
    "repeat_heavy": dict(zip(SOURCES, (10, 60, 20, 10))),
    "affinity": dict(zip(SOURCES, (10, 20, 50, 20))),
}


def load_development_events(path: Path, validation_end: int) -> dict:
    # Filter in Arrow before Python conversion, including mixed-timestamp row groups.
    names_to_read = ["uid", "item_id", "timestamp", "event_type"]
    source = pq.ParquetFile(path)
    schema = pa.schema([source.schema_arrow.field(name) for name in names_to_read])
    # Use core Arrow compute: this Windows runtime lacks the optional dataset DLL.
    filtered = [batch.filter(pc.less(batch.column("timestamp"), validation_end))
                for batch in source.iter_batches(batch_size=65536, columns=names_to_read, use_threads=False)]
    table = pa.Table.from_batches(filtered, schema=schema)
    columns = table.to_pydict()
    names = sorted(set(columns["event_type"]))
    codes = {name: index for index, name in enumerate(names)}
    arrays = {
        "uid": np.asarray(columns["uid"], dtype=np.uint32),
        "item_id": np.asarray(columns["item_id"], dtype=np.uint32),
        "timestamp": np.asarray(columns["timestamp"], dtype=np.uint32),
        "event_code": np.asarray([codes[name] for name in columns["event_type"]], dtype=np.uint8),
    }
    if len(arrays["timestamp"]) and int(arrays["timestamp"].max()) >= validation_end:
        raise AssertionError("Reserved events crossed the Arrow development filter")
    order = np.lexsort((arrays["item_id"], arrays["uid"], arrays["timestamp"]))
    return {**{name: values[order] for name, values in arrays.items()}, "event_names": names}


def load_frozen_queries(path: Path, config: dict) -> list[dict]:
    queries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    bounds = config["boundaries_raw"]
    if dict(Counter(q["split"] for q in queries)) != config["expected_queries"]:
        raise ValueError("Query counts differ from frozen P001")
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    if queries != sorted(queries, key=lambda q: (q["timestamp"], q["user_id"], q["target_item"])):
        raise ValueError("Queries must keep P001 ordering")
    for q in queries:
        lo, hi = ((bounds["train_query_start"], bounds["train_end"]) if q["split"] == "train"
                  else (bounds["train_end"], bounds["validation_end"]))
        if not lo <= q["timestamp"] < hi:
            raise ValueError("Query outside its frozen development interval")
    return queries


def average_by_user(rows: list[dict], values: list[float]) -> float | None:
    if not rows:
        return None
    grouped = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        grouped[row["user_id"]].append(value)
    return float(np.mean([np.mean(v) for v in grouped.values()]))


def summarize(rows: list[dict]) -> dict:
    hits = sum(row["hit"] for row in rows)
    summary = {
        "queries": len(rows), "users": len({row["user_id"] for row in rows}),
        "hits": hits, "misses": len(rows) - hits,
        "candidate_recall_query_micro": hits / len(rows) if rows else None,
        "candidate_recall_user_macro": average_by_user(rows, [int(row["hit"]) for row in rows]),
        "mean_candidates": float(np.mean([row["candidate_count"] for row in rows])) if rows else None,
    }
    summary["source_pool_recall"] = {
        str(depth): {source: {
            "hits": sum(row[f"pool_hits_{depth}"][source] for row in rows),
            "query_micro": float(np.mean([row[f"pool_hits_{depth}"][source] for row in rows])) if rows else None,
            "user_macro": average_by_user(rows, [int(row[f"pool_hits_{depth}"][source]) for row in rows]),
        } for source in SOURCES} for depth in (100, 200)
    }
    summary["pool_union_200"] = {
        "hits": sum(row["pool_union_hit"] for row in rows),
        "query_micro": float(np.mean([row["pool_union_hit"] for row in rows])) if rows else None,
        "user_macro": average_by_user(rows, [int(row["pool_union_hit"]) for row in rows]),
    }
    attribution_names = sorted({name for row in rows for name in row["selected_by_source"]})
    summary["actual_selection"] = {name: {
        "mean_candidates": float(np.mean([row["selected_by_source"].get(name, 0) for row in rows])),
        "target_hits": sum(row["target_selection_source"] == name for row in rows),
    } for name in attribution_names}
    summary["mean_pairwise_pool_overlap"] = {
        f"{left}|{right}": float(np.mean([row["pairwise_overlap"][f"{left}|{right}"] for row in rows]))
        for index, left in enumerate(SOURCES) for right in SOURCES[index+1:]
    } if rows else {}
    return summary


def evaluate_candidate_result(query: dict, result) -> dict:
    # This function is downstream of candidate generation. No target enters the generator.
    target = query["target_item"]
    pools = result.source_pools
    candidates = tuple(result.candidate_items)
    selections = dict(result.selected_by_source)
    if result.fallback_contributions:
        selections["popular_fallback"] = result.fallback_items
    return {
        "query_id": query["query_id"], "user_id": query["user_id"],
        "target_is_repeat": query["target_is_repeat"], "target_known_before": query["target_known_before"],
        "hit": target in candidates, "candidate_count": len(candidates),
        **{f"pool_hits_{depth}": {s: target in pools[s][:depth] for s in SOURCES} for depth in (100, 200)},
        "pool_union_hit": any(target in pools[s] for s in SOURCES),
        "selected_by_source": {name: len(items) for name, items in selections.items()},
        "target_selection_source": next((name for name, items in selections.items() if target in items), None),
        "pairwise_overlap": {
            f"{left}|{right}": len(set(pools[left]) & set(pools[right]))
            for index, left in enumerate(SOURCES) for right in SOURCES[index+1:]
        },
    }


def validate_config(config: dict) -> None:
    if config.get("test_evaluation") or config.get("model_training"):
        raise ValueError("P002 permits neither test evaluation nor model training")
    frozen = (config["max_candidates"], config["source_depth"], config["baseline_profile_depth"],
              config["affinity_profile_depth"], config["rrf_constant"])
    if frozen != (100, 200, 5, 32, 60) or tuple(config["strategies"]) != STRATEGIES or config["quotas"] != QUOTAS:
        raise ValueError("P002 strategies differ from the preregistered protocol")


def main() -> None:
    if __package__:
        from .retrieval_candidates import RetrievalCandidateGenerator
    else:
        from retrieval_candidates import RetrievalCandidateGenerator
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TASK / "configs/retrieval_002.json")
    parser.add_argument("--output", type=Path, default=TASK / "artifacts/retrieval_002")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    output = args.output.resolve()
    ensure_run_writable(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory; partial attempts are preserved")
    output.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(config["resource_limits"])
    monitor.start()
    started = time.perf_counter()
    status = {"run_id": config["run_id"], "status": "running", "config": config,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256(args.config),
              "source_code_sha256": {name: sha256(TASK / "src" / name) for name in
                                      ("run_retrieval.py", "retrieval_candidates.py", "pilot_storage.py", "run_pilot.py")},
              "protocol_sha256": sha256(TASK / "configs/P002_PROTOCOL.md"),
              "final_test_evaluated": False, "model_training": False}
    status["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TASK, text=True).strip()
    write_json(output / "run_status.json", status)
    try:
        monitor.check()
        manifest_path = TASK / config["baseline_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen = json.loads((TASK / config["baseline_metrics"]).read_text(encoding="utf-8"))
        if config["boundaries_raw"] != frozen["boundaries_raw"]:
            raise ValueError("Boundaries differ from P001")
        queries_path = TASK / config["queries_path"]
        query_digest = sha256(queries_path)
        query_key = queries_path.relative_to(TASK.parent).as_posix()
        if query_digest != manifest["archived_files"][query_key]["sha256"]:
            raise ValueError("P001 query file changed")
        status["input_sha256"] = {config["queries_path"]: query_digest,
                                  config["baseline_manifest"]: sha256(manifest_path),
                                  config["baseline_metrics"]: sha256(TASK / config["baseline_metrics"])}
        data_dir = TASK / config["data_dir"]
        for name, expected in frozen["prepared_data_sha256"].items():
            actual = sha256(data_dir / name)
            if actual != expected:
                raise ValueError(f"P001 prepared data changed: {name}")
            status["input_sha256"][f"{config['data_dir']}/{name}"] = actual
        status["input_validation_seconds"] = round(time.perf_counter() - started, 3)
        write_json(output / "run_status.json", status)
        stage_started = time.perf_counter()
        queries = load_frozen_queries(queries_path, config)
        events = load_development_events(data_dir / "events.parquet", config["boundaries_raw"]["validation_end"])
        status["filtered_load_seconds"] = round(time.perf_counter() - stage_started, 3)
        stage_started = time.perf_counter()
        generator = RetrievalCandidateGenerator(load_mapping(data_dir / "artist_mapping.parquet", "artist_id"),
                                                load_mapping(data_dir / "album_mapping.parquet", "album_id"),
                                                max_candidates=config["max_candidates"],
                                                candidate_quotas=config["quotas"]["baseline"],
                                                source_depth=config["source_depth"],
                                                baseline_profile_depth=config["baseline_profile_depth"],
                                                affinity_profile_depth=config["affinity_profile_depth"])
        status["mapping_load_seconds"] = round(time.perf_counter() - stage_started, 3)
        status["development_rows_loaded"] = len(events["timestamp"])
        print(f"P002 LOADED {len(events['timestamp']):,} development events; {len(queries)} frozen queries; reserved period filtered in Arrow", flush=True)
        stage_started = time.perf_counter()
        rows = {strategy: [] for strategy in STRATEGIES}
        index, compared = 0, 0
        generation_seconds = Counter()
        next_log = time.perf_counter() + 15
        with (output / "candidate_records.jsonl").open("w", encoding="utf-8") as stream:
            for position, query in enumerate(queries):
                timestamp = query["timestamp"]
                while index < len(events["timestamp"]) and int(events["timestamp"][index]) < timestamp:
                    generator.observe({"uid": int(events["uid"][index]), "item_id": int(events["item_id"][index]),
                                       "timestamp": int(events["timestamp"][index]),
                                       "event_type": events["event_names"][int(events["event_code"][index])]})
                    index += 1
                    if index % 5000 == 0:
                        monitor.check()
                sealed = {}
                for strategy in STRATEGIES:
                    tick = time.perf_counter()
                    result = generator.candidates(query["user_id"], timestamp, strategy=strategy)
                    generation_seconds[strategy] += time.perf_counter() - tick
                    if len(result.candidate_items) > config["max_candidates"] or len(set(result.candidate_items)) != len(result.candidate_items):
                        raise AssertionError("Invalid candidate count or duplicate item")
                    if any(item not in generator.catalog for pool in result.source_pools.values() for item in pool):
                        raise AssertionError("Source pool contains a future catalog item")
                    sealed[strategy] = result
                # Only after all strategies finish may target-derived query fields be inspected.
                if sealed["baseline"].candidate_items != query["candidate_items"]:
                    raise AssertionError(f"Baseline differs from frozen P001 query {query['query_id']}")
                if query["target_is_repeat"] != (query["target_item"] in generator.last_listen.get(query["user_id"], {})):
                    raise AssertionError("Frozen repeat stratum differs from past-only replay")
                if query["target_known_before"] != (query["target_item"] in generator.catalog):
                    raise AssertionError("Frozen warm/cold stratum differs from past-only replay")
                compared += 1
                for strategy, result in sealed.items():
                    public_query = {name: query[name] for name in ("query_id", "user_id", "timestamp", "split",
                                                                  "target_item", "target_is_repeat", "target_known_before")}
                    record = {"strategy": strategy, "query": public_query, "candidates": asdict(result)}
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    if query["split"] == "validation":
                        rows[strategy].append(evaluate_candidate_result(query, result))
                if position % 20 == 0:
                    monitor.check()
                if time.perf_counter() >= next_log:
                    print(f"RETRIEVAL queries {position+1}/{len(queries)}, past events {index:,}", flush=True)
                    next_log = time.perf_counter() + 15
        status["baseline_exact_queries_checked"] = compared
        status["target_strata_prefix_queries_checked"] = compared
        status["replay_and_record_seconds"] = round(time.perf_counter() - stage_started, 3)
        stage_started = time.perf_counter()
        status["past_events_observed"] = index
        status["generation_seconds_shared_pool_order"] = {key: round(value, 3) for key, value in generation_seconds.items()}
        status["generation_timing_note"] = "Sequential fixed order baseline/repeat/affinity/RRF; shared pool cache costs are not independent strategy benchmarks."
        status["strategies"] = {}
        for strategy, evaluation in rows.items():
            status["strategies"][strategy] = {
                "validation": summarize(evaluation),
                "strata": {name: summarize([row for row in evaluation if bool(row[field]) == flag])
                           for name, field, flag in (("repeat", "target_is_repeat", True), ("new", "target_is_repeat", False),
                                                    ("warm", "target_known_before", True), ("cold", "target_known_before", False))},
            }
        with (output / "evaluation_rows.jsonl").open("w", encoding="utf-8") as stream:
            for strategy, evaluation in rows.items():
                for row in evaluation:
                    stream.write(json.dumps({"strategy": strategy, **row}, separators=(",", ":")) + "\n")
        status["best_validation_strategy"] = max(STRATEGIES, key=lambda s: status["strategies"][s]["validation"]["candidate_recall_user_macro"])
        status["output_sha256"] = {name: sha256(output / name) for name in ("candidate_records.jsonl", "evaluation_rows.jsonl")}
        status["output_bytes"] = {name: (output / name).stat().st_size for name in status["output_sha256"]}
        status["summary_and_hash_seconds"] = round(time.perf_counter() - stage_started, 3)
        monitor.check()
        status["status"] = "complete"
    except Exception as error:
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        status["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        status["resources"] = monitor.finish()
        if status["resources"]["limit_exceeded"] and status["status"] == "complete":
            status["status"] = "failed"
            status["error"] = status["resources"]["limit_exceeded"]
        write_json(output / "run_status.json", status)
        write_json(output / "metrics.json", status)
        gc.collect()
    if status["status"] != "complete":
        raise MemoryError(status.get("error", "P002 did not complete"))
    print(f"P002 COMPLETE: {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
