"""Prepare frozen P008 sampled-negative training blocks (no fitting/scoring)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

if __package__:
    from .auto_features import AutoFeatureEngine, generate_specs
    from .auto_search import offsets_sizes
    from .run_pilot import ResourceMonitor, load_mapping
    from .run_retrieval import load_development_events
    from .sampled_negatives import build_query_samples
else:
    from auto_features import AutoFeatureEngine, generate_specs
    from auto_search import offsets_sizes
    from run_pilot import ResourceMonitor, load_mapping
    from run_retrieval import load_development_events
    from sampled_negatives import build_query_samples

TASK = Path(__file__).resolve().parents[1]
CONFIG_PATH = TASK / "configs/p008_sampled.json"
PROTOCOL_PATH = TASK / "configs/P008_PROTOCOL.md"
P006_MANIFEST_PATH = TASK / "configs/p006_freeze_manifest.json"
P007_SNAPSHOT = TASK / "report/p007_snapshot"
P007_SNAPSHOT_MANIFEST = P007_SNAPSHOT / "checkpoint_manifest.json"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_queries(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    queries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bounds = config["boundaries_raw"]
    if not queries or len({int(row["query_id"]) for row in queries}) != len(queries):
        raise ValueError("development query source must be non-empty with unique query IDs")
    key = lambda row: (int(row["timestamp"]), int(row["user_id"]), int(row["target_item"]))
    if queries != sorted(queries, key=key):
        raise ValueError("frozen development queries are not in canonical chronological order")
    for row in queries:
        split = row.get("split")
        timestamp = int(row["timestamp"])
        if split == "train":
            valid = bounds["train_query_start"] <= timestamp < bounds["train_end"]
        elif split == "validation":
            valid = bounds["train_end"] <= timestamp < bounds["validation_end"]
        else:
            valid = False
        if not valid:
            raise ValueError(f"query {row.get('query_id')} is outside frozen P008 development splits")
        candidates = row.get("candidate_items")
        if not isinstance(candidates, list) or len(candidates) != int(config["max_candidates"]):
            raise ValueError("P008 original candidate blocks must retain frozen K100 width")
        if len(set(map(int, candidates))) != len(candidates):
            raise ValueError("frozen K100 candidate block contains duplicate IDs")
    return queries


def _item_first_seen(events: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    items = np.asarray(events["item_id"], dtype=np.int64)
    times = np.asarray(events["timestamp"], dtype=np.int64)
    if items.ndim != 1 or times.ndim != 1 or len(items) != len(times):
        raise ValueError("development item/timestamp event columns are not aligned")
    item_ids, inverse = np.unique(items, return_inverse=True)
    first_seen = np.full(len(item_ids), np.iinfo(np.int64).max, dtype=np.int64)
    if len(items):
        np.minimum.at(first_seen, inverse, times)
    return item_ids.astype("<i8", copy=False), first_seen.astype("<i8", copy=False)


def _catalog_hash(catalog: list[int] | np.ndarray) -> str:
    values = np.asarray(sorted(map(int, catalog)), dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _manifest_files(manifest: Mapping[str, Any], description: str) -> dict[str, str]:
    files = manifest.get("files")
    # P006 uses {"files": {...}}; the frozen P007 snapshot checkpoint is a
    # direct relative-path -> SHA256 mapping.
    if files is None and all(isinstance(value, str) for value in manifest.values()):
        files = manifest
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{description} must contain a non-empty files hash ledger")
    return {str(name).replace("\\", "/"): str(digest) for name, digest in files.items()}


def _check_pins(root: Path, ledger: Mapping[str, str], required: list[str], description: str) -> dict[str, str]:
    checked: dict[str, str] = {}
    for relative in required:
        normalized = relative.replace("\\", "/")
        expected = ledger.get(normalized)
        if not expected:
            raise ValueError(f"{description} omits required pinned file: {normalized}")
        path = root / Path(normalized)
        if not path.is_file():
            raise FileNotFoundError(f"{description} pinned file missing: {normalized}")
        actual = _sha256(path)
        if actual != expected:
            raise AssertionError(f"{description} hash mismatch: {normalized}")
        checked[normalized] = actual
    return checked


def _verify_feature_inputs(queries: list[dict[str, Any]], inputs_path: Path,
                           offsets: np.ndarray, sizes: np.ndarray) -> None:
    rows = [json.loads(line) for line in inputs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(queries):
        raise ValueError("frozen feature_inputs query count differs from query source")
    for position, (query, row) in enumerate(zip(queries, rows)):
        expected = {
            "query_id": int(query["query_id"]), "user_id": int(query["user_id"]),
            "timestamp": int(query["timestamp"]), "offset": int(offsets[position]),
            "size": int(sizes[position]), "candidate_items": list(map(int, query["candidate_items"])),
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise AssertionError(f"frozen feature_inputs structural mismatch at query {query['query_id']}")


def _verify_feature_schema(schema_path: Path, feature_names: list[str]) -> None:
    schema = _read_json(schema_path)
    names = schema.get("feature_names")
    if not isinstance(names, list) or names[:66] != feature_names:
        raise AssertionError("frozen feature schema first 66 names differ from P008 A1 specs")


def _load_teacher(model_path: Path):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def prepare(config: Mapping[str, Any], cohort: Mapping[str, Any], output: str | Path) -> dict[str, Any]:
    """Create one cohort's immutable sampled-negative preparation checkpoint.

    The only feature materialization is the original K100 exact gate and the
    66-column sampled train blocks. This function never fits models or scores
    validation queries.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"P008 prepare output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    cohort_id = str(cohort["id"])
    started = time.perf_counter()
    monitor = ResourceMonitor(config["resource_limits"])
    monitor.start()

    status: dict[str, Any] = {
        "experiment": "P008", "stage": "prepare", "status": "running", "cohort": cohort_id,
        "config_sha256": _sha256(CONFIG_PATH), "protocol_sha256": _sha256(PROTOCOL_PATH),
        "source_sha256": {}, "inputs_sha256": {}, "outputs_sha256": {},
        "teacher_model_file": None, "teacher_model_sha256": None, "masks": {},
        "query_counts": {"train": 0, "hit": 0, "warm": 0, "cold": 0},
        "exact_gate": {"identical_float32_bytes": False, "cells": 0, "queries": 0},
        "progress": {"train_queries_completed": 0}, "final_test_evaluated": False,
    }
    uniform_rows = hybrid_rows = None
    try:
        source_names = ["prepare_sampled_training.py", "sampled_negatives.py", "auto_features.py",
                        "pilot_features.py", "auto_search.py", "run_pilot.py", "run_retrieval.py"]
        status["source_sha256"] = {name: _sha256(TASK / "src" / name) for name in source_names}
        p006_manifest = _read_json(P006_MANIFEST_PATH)
        p006_ledger = _manifest_files(p006_manifest, "P006 freeze manifest")
        # P008's own implementation is new, but the causal engine, frozen
        # query/matrix contract and parquet inputs must remain P006-pinned.
        core_sources = ["src/auto_features.py", "src/pilot_features.py", "src/auto_search.py",
                        "src/run_pilot.py", "src/run_retrieval.py"]
        status["p006_manifest_sha256"] = _sha256(P006_MANIFEST_PATH)
        status["p006_pinned_sha256"] = {}
        p007_root = TASK / config["p007_root"]
        p007_run = p007_root / cohort_id
        p007_status_path = p007_run / "run_status.json"
        selection_path = p007_run / "selection_frozen.json"
        p007_status = _read_json(p007_status_path)
        selection = _read_json(selection_path)
        if p007_status.get("status") != "complete":
            raise ValueError("P008 teacher requires a complete P007 development fit")
        final_by_label = {entry["label"]: entry for entry in p007_status.get("final_models", [])}
        if "M_UIDC" not in final_by_label:
            raise ValueError("P007 checkpoint is missing the frozen M_UIDC teacher")
        teacher_entry = final_by_label["M_UIDC"]
        teacher_path = p007_run / teacher_entry["model_file"]
        teacher_sha = _sha256(teacher_path)
        if teacher_sha != teacher_entry.get("model_sha256"):
            raise ValueError("P007 M_UIDC teacher model hash mismatch")
        masks = selection.get("final_masks", {})
        uidc_columns = list(map(int, masks.get("M_UIDC", [])))
        a1_columns = list(map(int, masks.get("Original_A1", [])))
        if uidc_columns != list(range(66)):
            raise ValueError("P008 UIDC mask must be the first 66 frozen A1 columns")
        if not a1_columns or len(a1_columns) != len(set(a1_columns)) or any(c < 0 or c >= 66 for c in a1_columns):
            raise ValueError("P008 Original_A1 mask must be a non-empty subset of the first 66 columns")
        status["teacher_model_file"] = str(teacher_path.relative_to(TASK)).replace("\\", "/")
        status["teacher_model_sha256"] = teacher_sha
        status["masks"] = {"UIDC": uidc_columns, "A1": a1_columns}

        query_path = TASK / cohort["development_queries"]
        history_path = TASK / cohort["history"]
        dev_run = TASK / cohort["development_run"]
        feature_path = dev_run / "features.npy"
        feature_inputs_path = dev_run / "feature_inputs.jsonl"
        schema_path = dev_run / "feature_schema.json"
        mapping_dir = TASK / cohort["mapping_dir"]
        artist_path = mapping_dir / "artist_mapping.parquet"
        album_path = mapping_dir / "album_mapping.parquet"
        inputs = {"queries": query_path, "history": history_path, "development_features": feature_path,
                  "feature_inputs": feature_inputs_path, "feature_schema": schema_path,
                  "artist_mapping": artist_path, "album_mapping": album_path,
                  "p007_run_status": p007_status_path, "p007_selection_frozen": selection_path,
                  "teacher_model": teacher_path}
        p006_paths = [query_path, history_path, feature_path, feature_inputs_path, schema_path,
                      artist_path, album_path]
        p006_relative = [str(path.relative_to(TASK)).replace("\\", "/") for path in p006_paths]
        status["p006_pinned_sha256"].update(_check_pins(TASK, p006_ledger,
                                                        p006_relative + core_sources,
                                                        "P006 freeze manifest"))

        # P007's local run files must still be the exact inputs preserved in
        # the immutable development snapshot; the snapshot ledger itself
        # authenticates those copies before the live files are compared.
        p007_ledger = _manifest_files(_read_json(P007_SNAPSHOT_MANIFEST), "P007 snapshot manifest")
        snapshot_status_rel = f"{cohort_id}/status_K100.json"
        snapshot_selection_rel = f"{cohort_id}/selection_frozen.json"
        p007_expected = _check_pins(P007_SNAPSHOT, p007_ledger,
                                    [snapshot_status_rel, snapshot_selection_rel],
                                    "P007 snapshot manifest")
        if _sha256(p007_status_path) != p007_expected[snapshot_status_rel]:
            raise AssertionError("P007 live run status differs from pinned K100 snapshot")
        if _sha256(selection_path) != p007_expected[snapshot_selection_rel]:
            raise AssertionError("P007 live frozen selection differs from pinned snapshot")
        status["p007_snapshot_manifest_sha256"] = _sha256(P007_SNAPSHOT_MANIFEST)
        status["p007_snapshot_pinned_sha256"] = p007_expected
        status["inputs_sha256"] = {str(path.relative_to(TASK)).replace("\\", "/"): _sha256(path)
                                   for path in inputs.values()}

        queries = _load_queries(query_path, config)
        train_queries = [query for query in queries if query["split"] == "train"]
        status["query_counts"]["train"] = len(train_queries)
        matrix = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[1] < 66 or matrix.dtype != np.float32:
            raise ValueError("frozen development feature matrix must be float32 with at least 66 columns")
        all_offsets, all_sizes = offsets_sizes(queries)
        _verify_feature_inputs(queries, feature_inputs_path, all_offsets, all_sizes)
        events = load_development_events(history_path, int(config["boundaries_raw"]["train_end"]))
        if len(events["timestamp"]) and int(np.max(events["timestamp"])) >= int(config["boundaries_raw"]["train_end"]):
            raise AssertionError("training event loader returned rows at/after train_end")
        item_ids, first_seen = _item_first_seen(events)
        inventory_path = output / "item_first_seen.npz"
        np.savez_compressed(inventory_path, item_ids=item_ids, first_seen=first_seen)
        first_by_item = {int(item): int(ts) for item, ts in zip(item_ids, first_seen)}

        artist_map = load_mapping(artist_path, "artist_id")
        album_map = load_mapping(album_path, "album_id")
        specs = generate_specs(config["windows_raw"], config["short_window_raw"])
        _verify_feature_schema(schema_path, [spec["feature_id"] for spec in specs[:66]])
        engine = AutoFeatureEngine(artist_map, album_map, config["short_window_raw"],
                                   config["half_life_raw"], specs=specs[:66])
        teacher = _load_teacher(teacher_path)

        # Eligibility is derivable from the saved causal first-seen inventory;
        # this pass sizes bounded output arrays before the replay pass.
        hit_ids: list[int] = []
        warm_ids: list[int] = []
        for query in train_queries:
            target = int(query["target_item"])
            warm = target in first_by_item and first_by_item[target] < int(query["timestamp"])
            old_hit = target in set(map(int, query["candidate_items"]))
            if old_hit and not warm:
                raise AssertionError("original K100 hit is absent from causal C(t)")
            if old_hit:
                hit_ids.append(int(query["query_id"]))
            if warm:
                warm_ids.append(int(query["query_id"]))
        if not set(hit_ids).issubset(warm_ids):
            raise AssertionError("P008 hit queries must be a subset of warm queries")
        status["query_counts"].update({"hit": len(hit_ids), "warm": len(warm_ids),
                                        "cold": len(train_queries) - len(warm_ids)})
        if not hit_ids or not warm_ids:
            raise ValueError("P008 cohort must contain both hit and warm train queries")
        uniform_path, hybrid_path = output / "uniform.npy", output / "hybrid.npy"
        uniform_rows = np.lib.format.open_memmap(uniform_path, mode="w+", dtype=np.float32,
                                                  shape=(len(hit_ids) * 100, 66))
        hybrid_rows = np.lib.format.open_memmap(hybrid_path, mode="w+", dtype=np.float32,
                                                 shape=(len(warm_ids) * 100, 66))

        offsets = {int(query["query_id"]): int(all_offsets[position])
                   for position, query in enumerate(queries)}
        sizes = {int(query["query_id"]): int(all_sizes[position])
                 for position, query in enumerate(queries)}
        query_position = {int(query["query_id"]): position for position, query in enumerate(queries)}
        next_uniform = 0
        next_hybrid = 0
        event_pointer = 0
        records_path = output / "records.jsonl"
        with records_path.open("x", encoding="utf-8", newline="\n") as records_stream:
            for index, query in enumerate(train_queries):
                qid = int(query["query_id"])
                uid, timestamp, target = int(query["user_id"]), int(query["timestamp"]), int(query["target_item"])
                while event_pointer < len(events["timestamp"]) and int(events["timestamp"][event_pointer]) < timestamp:
                    engine.observe({"uid": int(events["uid"][event_pointer]),
                                    "item_id": int(events["item_id"][event_pointer]),
                                    "timestamp": int(events["timestamp"][event_pointer]),
                                    "event_type": events["event_names"][int(events["event_code"][event_pointer])]})
                    event_pointer += 1
                    if event_pointer % 5000 == 0:
                        monitor.check()
                catalog = sorted(map(int, engine._items.keys()))
                warm = target in engine._items
                old_hit = target in set(map(int, query["candidate_items"]))
                if warm != (first_by_item.get(target, np.iinfo(np.int64).max) < timestamp):
                    raise AssertionError("live engine warm eligibility differs from saved first-seen inventory")
                if not warm and old_hit:
                    raise AssertionError("original candidate hit cannot be cold")
                catalog_digest = _catalog_hash(catalog)

                original_features = engine.transform(uid, list(map(int, query["candidate_items"])), timestamp)
                matrix_position = query_position[qid]
                row_offset, row_size = offsets[qid], sizes[qid]
                if row_size != len(query["candidate_items"]):
                    raise AssertionError(f"query {qid} frozen matrix candidate size mismatch")
                expected = np.asarray(matrix[row_offset:row_offset + row_size, :66])
                if original_features.dtype != np.float32 or original_features.tobytes() != expected.tobytes():
                    mismatch = np.argwhere(original_features != expected)
                    raise AssertionError(f"P008 exact UIDC66 gate failed at query {qid}: {mismatch[:5].tolist()}")
                status["exact_gate"]["cells"] += int(original_features.size)
                status["exact_gate"]["queries"] += 1

                sample_info: dict[str, Any] | None = None
                uniform_row_start: int | None = None
                hybrid_row_start: int | None = None
                if warm:
                    def teacher_score(items: list[int]) -> np.ndarray:
                        values = engine.transform(uid, items, timestamp)
                        scores = np.asarray(teacher.predict_proba(values, thread_count=2)[:, 1], dtype=np.float64)
                        return scores

                    sample_info = build_query_samples(catalog, query["candidate_items"], target,
                                                      teacher_score, cohort_id, qid, config["sampling"])
                    if old_hit:
                        uniform_features = engine.transform(uid, sample_info["uniform_items"], timestamp)
                        uniform_row_start = next_uniform * 100
                        uniform_rows[uniform_row_start:uniform_row_start + 100] = uniform_features
                        next_uniform += 1
                    hybrid_features = engine.transform(uid, sample_info["hybrid_items"], timestamp)
                    hybrid_row_start = next_hybrid * 100
                    hybrid_rows[hybrid_row_start:hybrid_row_start + 100] = hybrid_features
                    next_hybrid += 1

                record = {
                    "cohort": cohort_id, "query_id": qid, "user_id": uid, "timestamp": timestamp,
                    "target_item": target, "split": "train", "old_hit": bool(old_hit),
                    "warm": bool(warm), "cold": not bool(warm), "catalog_size": len(catalog),
                    "catalog_sha256": catalog_digest, "uniform_row_start": uniform_row_start,
                    "hybrid_row_start": hybrid_row_start,
                }
                if sample_info is None:
                    record.update({"uniform_draw_order": [], "uniform_items": [], "hybrid_items": [],
                                   "hard_pool_items": [], "hard_pool_scores": [], "hard_items": [],
                                   "hybrid_uniform_draw_order": [], "hard_pool_draw_order": [],
                                   "seed_uniform99": None, "seed_hard_pool256": None,
                                   "sampling_status": "cold_skip"})
                else:
                    record.update(sample_info)
                    record["sampling_status"] = "sampled"
                records_stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                status["progress"]["train_queries_completed"] = index + 1
                if (index + 1) % 25 == 0:
                    monitor.check()
                    _write_json(output / "run_status.json", status)
                if time.perf_counter() - float(status["progress"].get("last_report_seconds", 0.0)) >= 15.0:
                    elapsed = time.perf_counter() - started
                    status["progress"]["last_report_seconds"] = elapsed
                    print(f"P008 prepare {cohort_id}: {index + 1}/{len(train_queries)} train queries, "
                          f"{elapsed:.1f}s", flush=True)

        if next_uniform != len(hit_ids) or next_hybrid != len(warm_ids):
            raise AssertionError("sample output block counts differ from frozen eligibility pass")
        uniform_rows.flush()
        hybrid_rows.flush()
        del uniform_rows, hybrid_rows
        uniform_rows = hybrid_rows = None
        status["exact_gate"].update({"identical_float32_bytes": True,
                                     "queries": status["exact_gate"]["queries"]})
        if status["exact_gate"]["queries"] != len(train_queries):
            raise AssertionError("exact UIDC66 gate did not cover every train query")
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if uniform_rows is not None:
            try:
                uniform_rows.flush()
            except Exception:
                pass
            del uniform_rows
        if hybrid_rows is not None:
            try:
                hybrid_rows.flush()
            except Exception:
                pass
            del hybrid_rows
        status["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        status["resources"] = monitor.finish()
        if status["resources"].get("limit_exceeded"):
            status.update(status="failed", error=status["resources"]["limit_exceeded"])
        status["outputs_sha256"] = {
            str(path.relative_to(output)).replace("\\", "/"): _sha256(path)
            for path in output.rglob("*") if path.is_file() and path.name != "run_status.json"
        }
        _write_json(output / "run_status.json", status)
    if status.get("status") != "complete":
        raise MemoryError(status.get("error", "P008 preparation did not complete"))
    return status
