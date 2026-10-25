"""Independent checks for P008 sampled training preparation and scoring outputs.

This module deliberately reimplements the public sampling contract instead of
importing the training sampler. It reads prepared numeric artifacts and query
metadata only; it never opens event payloads.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .expanded_metrics import summarize_ranks
except ImportError:
    from expanded_metrics import summarize_ranks

UNIFORM_NEGATIVES = 99
HYBRID_UNIFORM = 50
HYBRID_HARD = 49
HARD_POOL_UNIFORM = 256
ARMS = ("Baseline", "Uniform", "HybridHit", "HybridWarm")
MASKS = ("UIDC", "A1")
LABELS = tuple(f"{arm}_{mask}" for arm in ARMS for mask in MASKS) + ("Popularity",)
TASK = Path(__file__).resolve().parents[1]


def derive_sampling_seed(seed: int, cohort_id: str, query_id: int, stream: str) -> int:
    """P008 RNG seed: big-endian integer from the first eight SHA256 bytes."""
    if stream not in {"uniform99", "hard_pool256"}:
        raise ValueError(f"unknown P008 sampling stream: {stream}")
    framed = f"{int(seed)}|{cohort_id}|{int(query_id)}|{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(framed).digest()[:8], "big", signed=False)


def _unique_sorted(values: Iterable[int], name: str) -> list[int]:
    result = [int(value) for value in values]
    if len(result) != len(set(result)):
        raise AssertionError(f"{name} contains duplicate item IDs")
    return sorted(result)


def catalog_before(item_ids: Sequence[int], first_seen: Sequence[int], timestamp: int) -> list[int]:
    """Return C(t), using only the saved item first-seen inventory."""
    ids = np.asarray(item_ids, dtype=np.int64)
    seen = np.asarray(first_seen, dtype=np.int64)
    if ids.ndim != 1 or seen.shape != ids.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("item first-seen inventory has invalid shape or duplicate IDs")
    return ids[seen < int(timestamp)].tolist()


def catalog_sha256(catalog_ids: Sequence[int]) -> str:
    """P008 catalog digest: sorted unique IDs as little-endian signed int64."""
    ids = _unique_sorted(catalog_ids, "catalog")
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes(order="C")).hexdigest()


def sample_uniform_draw(seed: int, cohort_id: str, query_id: int,
                        catalog_ids: Sequence[int], target_item: int,
                        count: int = UNIFORM_NEGATIVES) -> list[int]:
    """Reproduce the uniform stream, retaining RNG draw order."""
    available = _unique_sorted(catalog_ids, "catalog")
    if int(target_item) not in available:
        raise AssertionError("uniform sampling target is absent from C(t)")
    available.remove(int(target_item))
    if len(available) < count:
        raise AssertionError(f"C(t) has only {len(available)} eligible uniform negatives; need {count}")
    rng = np.random.default_rng(derive_sampling_seed(seed, cohort_id, query_id, "uniform99"))
    return np.asarray(rng.choice(np.asarray(available, dtype=np.int64), size=count, replace=False),
                      dtype=np.int64).tolist()


def sample_hard_pool_uniform(seed: int, cohort_id: str, query_id: int,
                             catalog_ids: Sequence[int], target_item: int,
                             uniform_first50: Sequence[int],
                             count: int = HARD_POOL_UNIFORM) -> list[int]:
    """Draw the additional mining pool after excluding target and uniform50."""
    available = _unique_sorted(catalog_ids, "catalog")
    excluded = {int(target_item), *[int(item) for item in uniform_first50]}
    if len(excluded) != 1 + len(uniform_first50):
        raise AssertionError("uniform50 contains duplicate/target item")
    pool = [item for item in available if item not in excluded]
    draw_size = min(count, len(pool))
    if not draw_size:
        return []
    rng = np.random.default_rng(derive_sampling_seed(seed, cohort_id, query_id, "hard_pool256"))
    return np.asarray(rng.choice(np.asarray(pool, dtype=np.int64), size=draw_size, replace=False),
                      dtype=np.int64).tolist()


def choose_teacher_hard_items(hard_pool_items: Sequence[int], scores: Sequence[float],
                              count: int = HYBRID_HARD) -> list[int]:
    """Pick teacher hard negatives by descending score, then ascending item ID."""
    items = [int(item) for item in hard_pool_items]
    values = np.asarray(scores, dtype=np.float64)
    if len(items) != len(values) or len(items) != len(set(items)):
        raise AssertionError("teacher pool item/score lengths differ or IDs duplicate")
    if not np.isfinite(values).all():
        raise AssertionError("teacher pool contains a non-finite score")
    if len(items) < count:
        raise AssertionError(f"teacher pool has {len(items)} items; need {count} hard negatives")
    order = sorted(range(len(items)), key=lambda index: (-float(values[index]), items[index]))
    return [items[index] for index in order[:count]]


def expected_sampled_blocks(seed: int, cohort_id: str, query_id: int, target_item: int,
                            catalog_ids: Sequence[int], original_candidates: Sequence[int],
                            hard_pool_scores: Mapping[int, float]) -> dict[str, list[int]]:
    """Rebuild all P008 sampled blocks from inventory and recorded teacher scores."""
    draw = sample_uniform_draw(seed, cohort_id, query_id, catalog_ids, target_item)
    uniform50 = draw[:HYBRID_UNIFORM]
    extra_pool_draw = sample_hard_pool_uniform(seed, cohort_id, query_id, catalog_ids,
                                               target_item, uniform50)
    pool_ids = sorted(({int(item) for item in original_candidates} | set(extra_pool_draw)) -
                      {int(target_item), *uniform50})
    missing = set(pool_ids) - set(map(int, hard_pool_scores))
    extra_scores = set(map(int, hard_pool_scores)) - set(pool_ids)
    if missing or extra_scores:
        raise AssertionError(f"teacher score pool IDs differ (missing={len(missing)}, extra={len(extra_scores)})")
    scores = [float(hard_pool_scores[item]) for item in pool_ids]
    hard = choose_teacher_hard_items(pool_ids, scores)
    uniform = sorted({int(target_item), *draw})
    hybrid = sorted({int(target_item), *uniform50, *hard})
    if len(uniform) != 100 or len(hybrid) != 100 or set(uniform50) & set(hard):
        raise AssertionError("sampled training block size/overlap invariant failed")
    return {"uniform_draw_order": draw, "hybrid_uniform_draw_order": uniform50,
            "uniform_items": uniform, "hard_pool_draw_order": extra_pool_draw,
            "hard_pool_items": pool_ids,
            "hard_items": hard, "hybrid_items": hybrid}


def validate_prepare_record(record: Mapping[str, Any], query: Mapping[str, Any],
                            catalog_ids: Sequence[int], expected_uniform: Mapping[str, Any] | None,
                            expected_hybrid: Mapping[str, Any] | None,
                            seed: int = 20261103) -> None:
    """Check eligibility, causality and exact saved sampling for a train query."""
    for key in ("query_id", "user_id", "timestamp", "target_item"):
        if int(record[key]) != int(query[key]):
            raise AssertionError(f"prepare {key} differs from frozen query")
    old_hit = int(query["target_item"]) in set(map(int, query["candidate_items"]))
    warm = int(query["target_item"]) in set(map(int, catalog_ids))
    if bool(record["old_hit"]) != old_hit or bool(record["warm"]) != warm:
        raise AssertionError("prepare old_hit/warm eligibility differs from source query/catalog")
    if "cold" in record and bool(record["cold"]) == warm:
        raise AssertionError("prepare cold flag differs from warm eligibility")
    if old_hit and not warm:
        raise AssertionError("every old shortlist hit must also be warm")
    if int(record["catalog_size"]) != len(catalog_ids):
        raise AssertionError("prepare catalog size differs from C(t)")
    if record.get("catalog_sha256") != catalog_sha256(catalog_ids):
        raise AssertionError("prepare catalog hash differs from C(t)")
    if not warm:
        if record.get("uniform_row_start") is not None or record.get("hybrid_row_start") is not None:
            raise AssertionError("cold query must not have a sampled feature matrix row")
        empty_fields = ("uniform_draw_order", "uniform_items", "hybrid_items", "hard_pool_items",
                        "hard_pool_scores", "hard_items", "hybrid_uniform_draw_order", "hard_pool_draw_order")
        if any(record.get(key) for key in empty_fields):
            raise AssertionError("cold query must be skipped by sampled arms")
        if record.get("sampling_status") != "cold_skip" or record.get("seed_uniform99") is not None or record.get("seed_hard_pool256") is not None:
            raise AssertionError("cold query must use the explicit cold_skip sampling status")
        return
    if record.get("sampling_status") != "sampled":
        raise AssertionError("warm query is missing sampled status")
    if expected_uniform is None or expected_hybrid is None:
        raise AssertionError("warm query lacks expected sampled blocks")
    if old_hit and record.get("uniform_row_start") is None:
        raise AssertionError("old shortlist hit lacks uniform feature rows")
    if not old_hit and record.get("uniform_row_start") is not None:
        raise AssertionError("non-hit query must not have uniform feature rows")
    if record.get("hybrid_row_start") is None:
        raise AssertionError("warm query lacks hybrid feature rows")
    for key in ("uniform_items", "uniform_draw_order"):
        if list(record[key]) != expected_uniform[key]:
            raise AssertionError(f"prepare uniform {key} differs from independent reconstruction")
    for key in ("hybrid_items", "hard_pool_items", "hard_items", "hybrid_uniform_draw_order", "hard_pool_draw_order"):
        if list(record[key]) != expected_hybrid[key]:
            raise AssertionError(f"prepare hybrid {key} differs from independent reconstruction")
    if (int(record.get("seed_uniform99", -1)) != derive_sampling_seed(seed, str(record["cohort"]),
            int(record["query_id"]), "uniform99") or
            int(record.get("seed_hard_pool256", -1)) != derive_sampling_seed(seed, str(record["cohort"]),
            int(record["query_id"]), "hard_pool256")):
        raise AssertionError("prepare stored seed does not match independently derived RNG seed")
    recorded_scores = np.asarray(record.get("hard_pool_scores", []), dtype=np.float64)
    pool_items = list(map(int, record["hard_pool_items"]))
    if len(recorded_scores) != len(pool_items) or not np.isfinite(recorded_scores).all():
        raise AssertionError("prepare hard pool scores do not align with item IDs")
    chosen = choose_teacher_hard_items(pool_items, recorded_scores)
    if list(map(int, record["hard_items"])) != chosen:
        raise AssertionError("prepare hard items do not follow recorded teacher score/item-ID order")


def verify_sampled_training(root: Path, task_root: Path, config_path: Path | None = None,
                            output: Path | None = None) -> dict[str, Any]:
    """Verify completed P008 prepare/fit/gate/full outputs without event access."""
    root, task_root = Path(root), Path(task_root)
    config_path = Path(config_path) if config_path is not None else task_root / "configs/p008_sampled.json"
    config = _read_json(config_path)
    protocol_path = task_root / "configs/P008_PROTOCOL.md"
    config_sha, protocol_sha = _sha256(config_path), _sha256(protocol_path)
    output = Path(output) if output is not None else root / "verification_independent.json"
    if not output.is_absolute():
        output = root / output
    if output.exists():
        raise FileExistsError(f"verification output already exists: {output}")
    cohorts_config = config.get("cohorts", [])
    if len(cohorts_config) != 3:
        raise AssertionError("P008 requires the three frozen development cohorts")
    methods = set(config.get("methods", []))
    if methods != set(LABELS):
        raise AssertionError("P008 configured labels differ from the fixed nine-label protocol")
    sampling = config.get("sampling", {})
    if (int(sampling.get("uniform_negatives", -1)) != UNIFORM_NEGATIVES or
            int(sampling.get("hybrid_uniform", -1)) != HYBRID_UNIFORM or
            int(sampling.get("hybrid_hard", -1)) != HYBRID_HARD or
            int(sampling.get("hard_pool_uniform", -1)) != HARD_POOL_UNIFORM or
            int(config.get("feature_count", -1)) != 66 or int(config.get("max_candidates", -1)) != 100):
        raise AssertionError("P008 fixed feature/sampling dimensions differ from the protocol")
    all_fits_path = root / "all_fits_frozen.json"
    if not all_fits_path.is_file():
        raise AssertionError("P008 all-fits freeze is missing")
    all_fits_sha = _sha256(all_fits_path)
    all_fits = _read_json(all_fits_path)
    if (all_fits.get("experiment") != "P008" or all_fits.get("frozen_before_evaluation") is not True or
            all_fits.get("config_sha256") != config_sha or all_fits.get("protocol_sha256") != protocol_sha or
            set(all_fits.get("cohort_fit_status_sha256", {})) != {str(c["id"]) for c in cohorts_config}):
        raise AssertionError("P008 all-fits freeze metadata/coverage differs")
    for relative, expected in all_fits.get("source_sha256", {}).items():
        source = task_root / (relative if "/" in relative else f"src/{relative}")
        if not source.is_file() or _sha256(source) != expected:
            raise AssertionError(f"P008 all-fits source hash mismatch: {relative}")
    for cohort in cohorts_config:
        fit_status_path = root / str(cohort["id"]) / "fit/run_status.json"
        if (not fit_status_path.is_file() or
                _sha256(fit_status_path) != all_fits["cohort_fit_status_sha256"].get(str(cohort["id"]))):
            raise AssertionError(f"{cohort['id']} fit status differs from the all-fits freeze")

    verified = []
    p007_root = task_root / config.get("p007_root", "artifacts/p007_expanded_search")
    for cohort in cohorts_config:
        cohort_id = str(cohort["id"])
        folder = root / cohort_id
        prepare_dir, fit_dir, gate_dir, full_dir = (folder / name for name in ("prepare", "fit", "gate", "full"))
        prepare_status = _read_json(prepare_dir / "run_status.json")
        fit_status = _read_json(fit_dir / "run_status.json")
        gate_status = _read_json(gate_dir / "run_status.json")
        full_status = _read_json(full_dir / "run_status.json")
        for status, stage in ((prepare_status, "prepare"), (fit_status, "fit"),
                              (gate_status, "gate"), (full_status, "full")):
            _check_stage_status(status, cohort_id, stage, config_sha, protocol_sha, task_root)
        if gate_status.get("all_fits_frozen_sha256") != all_fits_sha or full_status.get("all_fits_frozen_sha256") != all_fits_sha:
            raise AssertionError(f"{cohort_id} scoring stage does not reference the all-fits freeze")
        _check_output_hashes(prepare_dir, prepare_status)
        _check_output_hashes(full_dir, full_status)
        if fit_status.get("prepare_status_sha256") != _sha256(prepare_dir / "run_status.json"):
            raise AssertionError(f"{cohort_id} fit does not reference the verified prepare status")
        if full_status.get("fit_status_sha256") != _sha256(fit_dir / "run_status.json"):
            raise AssertionError(f"{cohort_id} full stage does not reference the verified fit status")
        if full_status.get("reference_parity_exact") is not True:
            raise AssertionError(f"{cohort_id} P007 reference parity is not declared exact")

        queries_path = task_root / cohort["development_queries"]
        queries_all = _read_jsonl(queries_path)
        train_queries = [query for query in queries_all if query.get("split") == "train"]
        validation_queries = [query for query in queries_all if query.get("split") == "validation"]
        if not train_queries or not validation_queries:
            raise AssertionError(f"{cohort_id} train or validation queries are missing")
        if any(len(query.get("candidate_items", [])) != 100 for query in train_queries + validation_queries):
            raise AssertionError(f"{cohort_id} source query candidate blocks are not fixed K=100")
        records = _read_jsonl(prepare_dir / "records.jsonl")
        if len(records) != len(train_queries):
            raise AssertionError(f"{cohort_id} prepare records do not cover every original train query")
        inventory_path = prepare_dir / "item_first_seen.npz"
        with np.load(inventory_path, allow_pickle=False) as archive:
            if not {"item_ids", "first_seen"}.issubset(archive.files):
                raise AssertionError(f"{cohort_id} item first-seen inventory schema invalid")
            item_ids = np.asarray(archive["item_ids"], dtype=np.int64)
            first_seen = np.asarray(archive["first_seen"], dtype=np.int64)
        if item_ids.ndim != 1 or first_seen.shape != item_ids.shape or np.any(np.diff(item_ids) <= 0):
            raise AssertionError(f"{cohort_id} item first-seen inventory must be sorted unique IDs")

        uniform_path, hybrid_path = prepare_dir / "uniform.npy", prepare_dir / "hybrid.npy"
        uniform = np.load(uniform_path, mmap_mode="r", allow_pickle=False)
        hybrid = np.load(hybrid_path, mmap_mode="r", allow_pickle=False)
        if uniform.dtype != np.float32 or hybrid.dtype != np.float32 or uniform.ndim != 2 or hybrid.ndim != 2:
            raise AssertionError(f"{cohort_id} sampled feature matrices must be 2D float32")
        if uniform.shape[1] != 66 or hybrid.shape[1] != 66 or not np.isfinite(uniform).all() or not np.isfinite(hybrid).all():
            raise AssertionError(f"{cohort_id} sampled feature matrix shape/value invalid")

        p007_dir = p007_root / cohort_id
        p007_status = _read_json(p007_dir / "run_status.json")
        if p007_status.get("status") != "complete":
            raise AssertionError(f"{cohort_id} P007 source run is incomplete")
        p007_frozen = _read_json(p007_dir / "selection_frozen.json")
        old_masks = p007_frozen.get("final_masks", {})
        masks = {"UIDC": old_masks.get("M_UIDC"), "A1": old_masks.get("Original_A1")}
        prep_masks = prepare_status.get("masks")
        if prep_masks != masks or any(not value or max(value) >= 66 for value in masks.values()):
            raise AssertionError(f"{cohort_id} prepared P008 masks differ from fixed P007 masks")
        p007_teacher = next((model for model in p007_status.get("final_models", [])
                             if model.get("label") == "M_UIDC"), None)
        teacher_relative = prepare_status.get("teacher_model_file")
        teacher_path = task_root / str(teacher_relative)
        if (p007_teacher is None or not teacher_path.is_file() or
                prepare_status.get("teacher_model_sha256") != p007_teacher.get("model_sha256") or
                _sha256(teacher_path) != p007_teacher.get("model_sha256") or
                teacher_path != p007_dir / p007_teacher["model_file"]):
            raise AssertionError(f"{cohort_id} P008 teacher is not the pinned P007 M_UIDC model")

        expected_u_cursor = expected_h_cursor = 0
        counts = {"train": len(train_queries), "warm": 0, "hit": 0, "cold": 0}
        for query, record in zip(train_queries, records, strict=True):
            if record.get("cohort") != cohort_id or record.get("split") != "train":
                raise AssertionError(f"{cohort_id}/{query['query_id']} prepare cohort/split metadata differs")
            catalog_ids = catalog_before(item_ids, first_seen, int(query["timestamp"]))
            if any(int(item) not in set(catalog_ids) for item in query["candidate_items"]):
                raise AssertionError(f"{cohort_id}/{query['query_id']} old candidate outside C(t)")
            warm = int(query["target_item"]) in set(catalog_ids)
            old_hit = int(query["target_item"]) in set(map(int, query["candidate_items"]))
            expected_u = expected_h = None
            if warm:
                counts["warm"] += 1
                if old_hit:
                    counts["hit"] += 1
                uniform_draw = sample_uniform_draw(int(config["sampling"]["seed"]), cohort_id,
                                                   int(query["query_id"]), catalog_ids, int(query["target_item"]))
                first50 = uniform_draw[:HYBRID_UNIFORM]
                pool_draw = sample_hard_pool_uniform(int(config["sampling"]["seed"]), cohort_id,
                    int(query["query_id"]), catalog_ids, int(query["target_item"]), first50,
                    int(config["sampling"]["hard_pool_uniform"]))
                expected_pool = sorted((set(map(int, query["candidate_items"])) | set(pool_draw)) -
                                       {int(query["target_item"]), *first50})
                saved_pool = list(map(int, record.get("hard_pool_items", [])))
                if saved_pool != expected_pool:
                    raise AssertionError(f"{cohort_id}/{query['query_id']} hard-pool IDs differ from independent draw")
                recorded_scores = record.get("hard_pool_scores", [])
                if len(recorded_scores) != len(saved_pool):
                    raise AssertionError(f"{cohort_id}/{query['query_id']} teacher scores do not align with hard pool")
                score_map = {item: float(score) for item, score in zip(saved_pool, recorded_scores, strict=True)}
                expected = expected_sampled_blocks(int(config["sampling"]["seed"]), cohort_id,
                    int(query["query_id"]), int(query["target_item"]), catalog_ids,
                    query["candidate_items"], score_map)
                expected_u = expected
                expected_h = expected
            else:
                counts["cold"] += 1
            validate_prepare_record(record, query, catalog_ids, expected_u, expected_h,
                                    int(config["sampling"]["seed"]))
            if warm:
                if old_hit:
                    if int(record["uniform_row_start"]) != expected_u_cursor:
                        raise AssertionError(f"{cohort_id} uniform row offsets are not contiguous")
                    expected_u_cursor += 100
                if int(record["hybrid_row_start"]) != expected_h_cursor:
                    raise AssertionError(f"{cohort_id} hybrid row offsets are not contiguous")
                expected_h_cursor += 100
        if uniform.shape != (expected_u_cursor, 66) or hybrid.shape != (expected_h_cursor, 66):
            raise AssertionError(f"{cohort_id} sampled matrix rows do not match eligible queries")
        del uniform, hybrid
        saved_counts = prepare_status.get("query_counts", {})
        for key, value in counts.items():
            if int(saved_counts.get(key, -1)) != value:
                raise AssertionError(f"{cohort_id} prepare query_counts.{key} differs")
        expected_train_cells = sum(len(query["candidate_items"]) for query in train_queries) * 66
        prepare_gate = prepare_status.get("exact_gate", {})
        if (prepare_gate.get("identical_float32_bytes") is not True or
                int(prepare_gate.get("cells", -1)) != expected_train_cells or
                int(prepare_gate.get("queries", -1)) != len(train_queries)):
            raise AssertionError(f"{cohort_id} prepare exact original-feature gate invalid")
        if gate_status.get("fit_status_sha256") != _sha256(fit_dir / "run_status.json"):
            raise AssertionError(f"{cohort_id} exact gate does not reference the verified fits")
        validation_cells = sum(len(query["candidate_items"]) for query in validation_queries) * 66
        for stage_name, status in (("gate", gate_status), ("full", full_status)):
            exact = status.get("exact_gate", {})
            if (exact.get("identical_float32_bytes") is not True or int(exact.get("cells", -1)) != validation_cells or
                    int(exact.get("queries", -1)) != len(validation_queries) or
                    not isinstance(exact.get("values_sha256"), str) or len(exact["values_sha256"]) != 64):
                raise AssertionError(f"{cohort_id} {stage_name} exact validation feature gate invalid")

        training_frozen = _read_json(fit_dir / "training_frozen.json")
        if fit_status.get("training_frozen_sha256") != _sha256(fit_dir / "training_frozen.json"):
            raise AssertionError(f"{cohort_id} frozen training specification hash mismatch")
        expected_arm_ids = {
            "Uniform": [int(record["query_id"]) for record in records if record["old_hit"]],
            "HybridHit": [int(record["query_id"]) for record in records if record["old_hit"]],
            "HybridWarm": [int(record["query_id"]) for record in records if record["warm"]],
        }
        if training_frozen.get("arms") != expected_arm_ids or training_frozen.get("masks") != masks:
            raise AssertionError(f"{cohort_id} frozen fit query IDs/masks differ from prepare eligibility")
        models = fit_status.get("models", [])
        model_by_label = {str(model.get("label")): model for model in models}
        expected_model_labels = {f"{arm}_{mask}" for arm in ARMS for mask in MASKS if arm != "Baseline"}
        expected_model_labels |= {"Baseline_UIDC", "Baseline_A1"}
        if len(models) != 8 or len(model_by_label) != len(models) or set(model_by_label) != expected_model_labels:
            raise AssertionError(f"{cohort_id} fit model labels differ from protocol")
        p007_by_label = {model["label"]: model for model in p007_status.get("final_models", [])}
        for label, old_label in (("Baseline_UIDC", "M_UIDC"), ("Baseline_A1", "Original_A1")):
            model = model_by_label[label]
            prior = p007_by_label[old_label]
            if (model.get("model_origin") != "p007" or model.get("columns") != masks[label.split("_")[1]] or
                    model.get("model_sha256") != prior.get("model_sha256") or
                    model.get("model_file") != (p007_dir / prior["model_file"]).relative_to(task_root).as_posix()):
                raise AssertionError(f"{cohort_id}/{label} is not the exact pinned P007 model")
        for label in expected_model_labels - {"Baseline_UIDC", "Baseline_A1"}:
            model = model_by_label[label]
            arm, mask_name = label.rsplit("_", 1)
            arm_ids = expected_arm_ids[arm]
            rows = 100 * len(arm_ids)
            if (model.get("model_origin") != "p008" or model.get("columns") != masks[mask_name] or
                    model.get("training_query_ids") != arm_ids or
                    model.get("training") != {"queries": len(arm_ids), "rows": rows, "positives": len(arm_ids)}):
                raise AssertionError(f"{cohort_id}/{label} training IDs/counts differ from sample records")
        for label, old_label in (("Baseline_UIDC", "M_UIDC"), ("Baseline_A1", "Original_A1")):
            if model_by_label[label].get("training") != p007_by_label[old_label].get("training"):
                raise AssertionError(f"{cohort_id}/{label} baseline training counts differ from P007")
        for model in models:
            model_path = (task_root / model["model_file"] if model["model_origin"] == "p007"
                          else fit_dir / model["model_file"])
            if not model_path.is_file() or _sha256(model_path) != model.get("model_sha256"):
                raise AssertionError(f"{cohort_id}/{model['label']} model hash mismatch")

        qids = [int(query["query_id"]) for query in validation_queries]
        labels_full = list(LABELS)
        p007_map = {"Baseline_UIDC": "M_UIDC", "Baseline_A1": "Original_A1", "Popularity": "Popularity"}
        p007_final = {model["label"]: model for model in p007_status.get("final_models", [])}
        rank100_by_label: dict[str, list[dict[str, Any]]] = {}
        metrics100 = _read_json(full_dir / "metrics100.json")
        metrics_full = _read_json(full_dir / "metrics_full.json")
        if set(metrics100) != set(labels_full) or set(metrics_full) != set(labels_full):
            raise AssertionError(f"{cohort_id} saved metrics label set differs")
        for label in labels_full:
            base_path = full_dir / "ranks100" / f"{label}.jsonl"
            rows100 = _read_jsonl(base_path)
            scores_by_query = _load_prediction(full_dir / "predictions" / f"{label}.npz", qids, validation_queries)
            if len(rows100) != len(validation_queries):
                raise AssertionError(f"{cohort_id}/{label} K100 row count differs")
            reconstructed = [_rank_row(query, scores, cohort_id) for query, scores in zip(validation_queries, scores_by_query, strict=True)]
            for index, (saved, actual) in enumerate(zip(rows100, reconstructed, strict=True)):
                if saved != actual:
                    raise AssertionError(f"{cohort_id}/{label}/{qids[index]} K100 prediction rank differs")
            if label in p007_map:
                old = p007_map[label]
                reference_rows = _read_jsonl(p007_dir / "ranks100" / f"{old}.jsonl")
                if rows100 != reference_rows:
                    raise AssertionError(f"{cohort_id}/{label} K100 rows differ from P007 {old}")
                if label in p007_final:
                    prior_model = p007_final[label]
                    prior_scores = _load_prediction(p007_dir / prior_model["prediction_file"], qids, validation_queries)
                    if any(not np.array_equal(left, right) for left, right in zip(scores_by_query, prior_scores, strict=True)):
                        raise AssertionError(f"{cohort_id}/{label} K100 scores differ from P007")
            _compare_metrics(metrics100[label], summarize_ranks(rows100), f"{cohort_id}/{label} metrics100")
            rank100_by_label[label] = rows100

        if set(full_status.get("outputs_sha256", {})) != {
            p.relative_to(full_dir).as_posix() for p in full_dir.rglob("*")
            if p.is_file() and p.name != "run_status.json"
        }:
            raise AssertionError(f"{cohort_id} full output hash ledger does not cover all outputs")
        for label in labels_full:
            rows = _read_jsonl(full_dir / "ranksfull" / f"{label}.jsonl")
            _check_full_rows(rows, validation_queries, rank100_by_label[label], cohort_id, label)
            if label in p007_map:
                old = p007_map[label]
                reference_rows = _read_jsonl(p007_dir / "full_catalog" / f"{old}.jsonl")
                if rows != reference_rows:
                    raise AssertionError(f"{cohort_id}/{label} full rows differ from P007 {old}")
            _compare_metrics(metrics_full[label], summarize_ranks(rows), f"{cohort_id}/{label} full metrics")
        verified.append({"cohort": cohort_id, "status": "verified", "train_queries": len(train_queries),
                         "warm_queries": counts["warm"], "hit_queries": counts["hit"],
                         "cold_queries": counts["cold"], "validation_queries": len(validation_queries),
                         "models": len(models), "rank_labels": len(labels_full),
                         "exact_gate_cells": validation_cells})

    receipt = {"status": "verified", "experiment": "P008", "cohort_count": len(verified),
               "cohorts": verified, "all_fits_frozen_sha256": all_fits_sha,
               "sampling_rng": "independent SHA256 first-eight-byte big-endian seed framing; NumPy default_rng",
               "teacher_hard_evidence": "Recorded P007 teacher scores are checked for ID alignment and deterministic score-descending/item-ID-ascending selection; teacher inference is not independently rerun.",
               "data_access": "No event payloads read; verifier uses query metadata, item first-seen inventory, numeric prepared matrices, prediction archives, and saved ranks."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=TASK / "artifacts/p008_sampled_training")
    parser.add_argument("--task-root", type=Path, default=TASK)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else args.task_root / args.root
    output = args.output
    if output is not None and not output.is_absolute():
        output = root / output
    result = verify_sampled_training(root, args.task_root, args.config, output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compare_metrics(saved: Mapping[str, Any], actual: Mapping[str, Any], label: str) -> None:
    for block in ("overall", "strata"):
        left_block, right_block = saved.get(block, {}), actual.get(block, {})
        groups = ["overall"] if block == "overall" else sorted(set(left_block) | set(right_block))
        for group in groups:
            left = left_block if block == "overall" else left_block.get(group, {})
            right = right_block if block == "overall" else right_block.get(group, {})
            prefix = label if block == "overall" else f"{label}.{group}"
            for key in ("query_count", "user_count", "target_known_before_count", "repeat_count"):
                if left.get(key) != right.get(key):
                    raise AssertionError(f"{prefix}.{key} differs")
            for key in ("candidate_recall",):
                left_value, right_value = left.get(key), right.get(key)
                if left_value is None or right_value is None:
                    if left_value != right_value:
                        raise AssertionError(f"{prefix}.{key} differs")
                elif not math.isclose(float(left_value), float(right_value), rel_tol=0.0, abs_tol=1e-12):
                    raise AssertionError(f"{prefix}.{key} differs")
            for average in ("query_micro", "user_macro"):
                if set(left.get(average, {})) != set(right.get(average, {})):
                    raise AssertionError(f"{prefix}.{average} metric names differ")
                for metric, value in left.get(average, {}).items():
                    if not math.isclose(float(value), float(right[average][metric]), rel_tol=0.0, abs_tol=1e-12):
                        raise AssertionError(f"{prefix}.{average}.{metric} differs")
    if saved.get("counts") != actual.get("counts"):
        raise AssertionError(f"{label}.counts differ")


def _rank_row(query: Mapping[str, Any], scores: Sequence[float], cohort_id: str) -> dict[str, Any]:
    items = [int(item) for item in query["candidate_items"]]
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(items),) or not np.isfinite(values).all():
        raise AssertionError(f"{cohort_id}/{query['query_id']} prediction score shape/value invalid")
    order = sorted(range(len(items)), key=lambda index: (-float(values[index]), items[index]))
    target = int(query["target_item"])
    target_index = next((index for index, item in enumerate(items) if item == target), None)
    rank = None if target_index is None else next(pos + 1 for pos, index in enumerate(order) if index == target_index)
    row = {key: query[key] for key in ("query_id", "user_id", "timestamp", "target_is_repeat", "target_known_before")}
    row.update(cohort=cohort_id, candidate_count=len(items), target_rank=rank,
               target_score=None if target_index is None else float(values[target_index]),
               top_items=[{"item_id": items[index], "score": float(values[index])} for index in order[:20]])
    return row


def _check_row_metadata(row: Mapping[str, Any], query: Mapping[str, Any], cohort_id: str,
                        candidate_count: int) -> None:
    for key in ("query_id", "user_id", "timestamp", "target_is_repeat", "target_known_before"):
        if row.get(key) != query.get(key):
            raise AssertionError(f"{cohort_id}/{query['query_id']} rank metadata {key} differs")
    if row.get("cohort") != cohort_id or int(row.get("candidate_count", -1)) != candidate_count:
        raise AssertionError(f"{cohort_id}/{query['query_id']} cohort/candidate count differs")


def _load_prediction(path: Path, qids: list[int], queries: list[dict[str, Any]]) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"query_ids", "offsets", "scores"}
        if not required.issubset(archive.files):
            raise AssertionError(f"prediction archive missing keys: {sorted(required - set(archive.files))}")
        saved_qids = np.asarray(archive["query_ids"], dtype=np.int64).tolist()
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        scores = np.asarray(archive["scores"], dtype=np.float64)
    if saved_qids != qids or offsets.shape != (len(qids) + 1,) or offsets[0] != 0 or offsets[-1] != len(scores):
        raise AssertionError("prediction query IDs or offsets differ from validation queries")
    if np.any(np.diff(offsets) < 0) or not np.isfinite(scores).all():
        raise AssertionError("prediction scores or offsets invalid")
    output = []
    for index, query in enumerate(queries):
        values = scores[offsets[index]:offsets[index + 1]]
        if len(values) != len(query["candidate_items"]):
            raise AssertionError(f"prediction candidate count differs for query {qids[index]}")
        output.append(values)
    return output


def _check_stage_status(status: Mapping[str, Any], cohort_id: str, stage: str,
                        config_sha: str, protocol_sha: str, task_root: Path) -> None:
    if (status.get("experiment") != "P008" or status.get("cohort") != cohort_id or
            status.get("stage") != stage or status.get("status") != "complete" or
            status.get("final_test_evaluated") is not False):
        raise AssertionError(f"{cohort_id}/{stage} is not a complete development-only P008 stage")
    if status.get("config_sha256") != config_sha or status.get("protocol_sha256") != protocol_sha:
        raise AssertionError(f"{cohort_id}/{stage} config/protocol hash mismatch")
    ledger = status.get("source_sha256", {})
    if not isinstance(ledger, dict) or not ledger:
        raise AssertionError(f"{cohort_id}/{stage} source hash ledger missing")
    for relative, expected in ledger.items():
        path = task_root / (relative if "/" in relative else f"src/{relative}")
        if not path.is_file() or _sha256(path) != expected:
            raise AssertionError(f"{cohort_id}/{stage} source hash mismatch: {relative}")


def _check_output_hashes(folder: Path, status: Mapping[str, Any], field: str = "outputs_sha256") -> None:
    ledger = status.get(field, {})
    if not isinstance(ledger, dict) or not ledger:
        raise AssertionError(f"{folder.name} output hash ledger missing")
    for relative, expected in ledger.items():
        path = folder / relative
        if not path.is_file() or _sha256(path) != expected:
            raise AssertionError(f"{folder.name} output hash mismatch: {relative}")


def _check_full_rows(rows: list[dict[str, Any]], queries: list[dict[str, Any]],
                     base_rows: list[dict[str, Any]], cohort_id: str, label: str) -> None:
    if len(rows) != len(queries) or len(base_rows) != len(queries):
        raise AssertionError(f"{cohort_id}/{label} rank row counts differ")
    for index, (row, query, base) in enumerate(zip(rows, queries, base_rows, strict=True)):
        _check_row_metadata(row, query, cohort_id, int(row.get("candidate_count", -1)))
        candidate_count = int(row["candidate_count"])
        if candidate_count < len(query["candidate_items"]) or candidate_count < 100:
            raise AssertionError(f"{cohort_id}/{label}/{index} full catalog is smaller than K100")
        top = row.get("top_items")
        if not isinstance(top, list) or len(top) != min(candidate_count, 20):
            raise AssertionError(f"{cohort_id}/{label}/{index} invalid top-20 evidence")
        pairs = [(float(item["score"]), int(item["item_id"])) for item in top]
        if any(pairs[j][0] < pairs[j + 1][0] or
               (pairs[j][0] == pairs[j + 1][0] and pairs[j][1] >= pairs[j + 1][1])
               for j in range(len(pairs) - 1)):
            raise AssertionError(f"{cohort_id}/{label}/{index} top-items ordering differs")
        rank = row.get("target_rank")
        target_known = bool(query["target_known_before"])
        if (rank is None) == target_known:
            raise AssertionError(f"{cohort_id}/{label}/{index} target presence differs from warm flag")
        if rank is not None and not 1 <= int(rank) <= candidate_count:
            raise AssertionError(f"{cohort_id}/{label}/{index} target rank outside catalog")
        if base.get("target_rank") is not None:
            if rank is None or int(rank) < int(base["target_rank"]):
                raise AssertionError(f"{cohort_id}/{label}/{index} full rank improves over K100")
            if row.get("target_score") != base.get("target_score"):
                raise AssertionError(f"{cohort_id}/{label}/{index} full target score differs from K100")
        target_id = int(query["target_item"])
        found = [position for position, item in enumerate(top) if int(item["item_id"]) == target_id]
        if rank is not None and int(rank) <= 20:
            if found != [int(rank) - 1] or float(top[found[0]]["score"]) != float(row["target_score"]):
                raise AssertionError(f"{cohort_id}/{label}/{index} target top-item evidence differs")
        elif found:
            raise AssertionError(f"{cohort_id}/{label}/{index} target beyond top20 appears in evidence")


if __name__ == "__main__":
    main()

