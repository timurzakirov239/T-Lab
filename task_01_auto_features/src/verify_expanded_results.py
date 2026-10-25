"""Independent verification for P007 K=100 and full-catalog results."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .expanded_metrics import summarize_ranks
else:
    from expanded_metrics import summarize_ranks

TASK = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = TASK / "artifacts/p007_expanded_search"
TOLERANCE = 1e-12
CONTROL_ORIGINAL_LABEL = {"M_I": "M_I", "M_UIDC": "M_UIDC",
                          "Original_A1": "Auto_A1", "Original_A2": "Auto_A2"}
SEARCH_SOURCE_REQUIRED = {"run_expanded_search.py", "expanded_search.py", "expanded_metrics.py",
                         "auto_features.py", "pilot_features.py", "auto_search.py", "run_auto_features.py"}


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                rows.append(row)
    return rows


def _close(a: Any, b: Any, label: str, tol: float = TOLERANCE) -> None:
    if a is None or b is None:
        if a != b:
            raise AssertionError(f"{label}: expected {b!r}, got {a!r}")
    elif not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{label}: expected {b!r}, got {a!r}")


def _compare_metrics(saved: dict[str, Any], actual: dict[str, Any], label: str) -> None:
    for block in ("overall", "strata"):
        saved_block = saved.get(block, {})
        actual_block = actual.get(block, {})
        groups = ["overall"] if block == "overall" else sorted(set(saved_block) | set(actual_block))
        for group in groups:
            left = saved_block if block == "overall" else saved_block.get(group, {})
            right = actual_block if block == "overall" else actual_block.get(group, {})
            prefix = label if block == "overall" else f"{label}.{group}"
            for key in ("query_count", "user_count", "target_known_before_count", "repeat_count"):
                if key in left and left.get(key) != right.get(key):
                    raise AssertionError(f"{prefix}.{key}: expected {left.get(key)!r}, got {right.get(key)!r}")
            if "candidate_recall" in left:
                _close(left.get("candidate_recall"), right.get("candidate_recall"), f"{prefix}.candidate_recall")
            for average in ("query_micro", "user_macro"):
                expected = left.get(average, {})
                observed = right.get(average, {})
                if set(expected) != set(observed):
                    raise AssertionError(f"{prefix}.{average} metric names differ")
                for metric, value in expected.items():
                    _close(value, observed[metric], f"{prefix}.{average}.{metric}")
    if saved.get("counts"):
        for key, value in saved["counts"].items():
            if actual.get("counts", {}).get(key) != value:
                raise AssertionError(f"{label}.counts.{key}: expected {value!r}, got {actual.get('counts', {}).get(key)!r}")


def _prediction_scores(run: Path, result: dict[str, Any], expected_qids: list[int], queries: list[dict[str, Any]]) -> list[np.ndarray]:
    relative = result.get("prediction_file")
    if not relative:
        raise ValueError(f"{result.get('label')} lacks prediction_file")
    path = Path(str(relative))
    if not path.is_absolute():
        path = run / path
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = result.get("prediction_sha256")
    if expected_hash and sha256(path) != expected_hash:
        raise AssertionError(f"prediction hash mismatch: {result.get('label')}")
    with np.load(path, allow_pickle=False) as archive:
        id_key = next((key for key in ("query_ids", "query_id") if key in archive), None)
        off_key = next((key for key in ("offsets", "candidate_offsets") if key in archive), None)
        score_key = next((key for key in ("scores", "predictions", "probabilities") if key in archive), None)
        if not id_key or not off_key or not score_key:
            raise ValueError(f"prediction archive schema invalid: {path}")
        qids = np.asarray(archive[id_key], dtype=np.int64).tolist()
        offsets = np.asarray(archive[off_key], dtype=np.int64)
        flat = np.asarray(archive[score_key], dtype=np.float64)
    if qids != expected_qids or offsets.ndim != 1 or len(offsets) != len(qids) + 1 or offsets[0] != 0 or offsets[-1] != len(flat):
        raise AssertionError(f"prediction query IDs/offsets differ from validation queries: {result.get('label')}")
    if np.any(np.diff(offsets) < 0) or not np.isfinite(flat).all():
        raise ValueError(f"prediction scores/offsets invalid: {result.get('label')}")
    output = []
    for index, query in enumerate(queries):
        if offsets[index + 1] - offsets[index] != len(query["candidate_items"]):
            raise AssertionError(f"candidate offset mismatch for {result.get('label')}/{query['query_id']}")
        output.append(flat[offsets[index]:offsets[index + 1]])
    return output


def _rank_row(query: dict[str, Any], scores: np.ndarray, cohort: str) -> dict[str, Any]:
    items = [int(item) for item in query["candidate_items"]]
    if len(scores) != len(items) or not np.isfinite(scores).all():
        raise ValueError("score/candidate mismatch or non-finite score")
    ordered = sorted(range(len(items)), key=lambda i: (-float(scores[i]), items[i]))
    target_index = next((i for i, item in enumerate(items) if item == int(query["target_item"])), None)
    rank = None if target_index is None else next(position + 1 for position, index in enumerate(ordered) if index == target_index)
    return {"cohort": cohort, "query_id": int(query["query_id"]), "user_id": int(query["user_id"]),
            "timestamp": int(query["timestamp"]), "candidate_count": len(items), "target_rank": rank,
            "target_score": None if target_index is None else float(scores[target_index]),
            "target_is_repeat": bool(query["target_is_repeat"]),
            "target_known_before": bool(query["target_known_before"]),
            "top_items": [{"item_id": items[index], "score": float(scores[index])} for index in ordered[:20]]}


def _check_rank_rows(rows: list[dict[str, Any]], queries: list[dict[str, Any]], cohort: str,
                     label: str) -> None:
    if len(rows) != len(queries):
        raise AssertionError(f"{cohort}/{label} rank row count mismatch: {len(rows)} != {len(queries)}")
    for i, (row, query) in enumerate(zip(rows, queries, strict=True)):
        expected = (int(query["query_id"]), int(query["user_id"]), int(query["timestamp"]),
                    len(query["candidate_items"]), bool(query["target_is_repeat"]), bool(query["target_known_before"]))
        actual = (int(row["query_id"]), int(row["user_id"]), int(row["timestamp"]), int(row["candidate_count"]),
                  bool(row["target_is_repeat"]), bool(row["target_known_before"]))
        if actual != expected or row.get("cohort") != cohort:
            raise AssertionError(f"{cohort}/{label} rank metadata mismatch at row {i}")


def _source_hash_check(status: dict[str, Any], task_root: Path, field: str, cohort: str,
                       required: set[str] | None = None) -> None:
    ledger = status.get(field, {})
    if required is not None and not required.issubset(ledger):
        raise AssertionError(f"{cohort} source hash ledger lacks mandatory files: {sorted(required - set(ledger))}")
    for relative, expected in ledger.items():
        if required is not None and relative not in required:
            continue
        path = task_root / "src" / relative
        if path.is_file() and sha256(path) != expected:
            raise AssertionError(f"{cohort} source code drift: {relative}")


def _old_metric(status_path: Path, label: str) -> dict[str, Any]:
    status = _json(status_path)
    entry = next((row for row in status.get("final_models", []) if row.get("label") == label), None)
    if entry is None:
        raise AssertionError(f"old development status lacks {label}: {status_path}")
    metric = entry.get("metrics", {})
    return metric.get("by_split", {}).get("validation", metric)


def verify_expanded(root: Path = DEFAULT_ROOT, output: Path | None = None,
                    task_root: Path = TASK) -> dict[str, Any]:
    root, task_root = Path(root), Path(task_root)
    if output is None:
        output = root / "verification.json"
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    if output.exists():
        raise FileExistsError(f"verification output already exists: {output}")

    config = _json(task_root / "configs/p007_expanded.json")
    cohort_specs = config.get("cohorts", [])
    if len(cohort_specs) != 3:
        raise AssertionError("P007 requires three frozen development cohorts")
    verified_cohorts = []
    for cohort in cohort_specs:
        cohort_id = str(cohort["id"])
        search = root / cohort_id
        search_status = _json(search / "run_status.json")
        if search_status.get("status") != "complete" or search_status.get("cohort") != cohort_id:
            raise AssertionError(f"P007 search is not complete for {cohort_id}")
        if search_status.get("final_test_evaluated") is not False or search_status.get("outer_evaluation_started") is True:
            raise AssertionError(f"P007 search has unexpected evaluation flags for {cohort_id}")
        _source_hash_check(search_status, task_root, "source_sha256", cohort_id, SEARCH_SOURCE_REQUIRED)
        frozen = _json(search / "selection_frozen.json")
        if frozen.get("experiment") != "P007" or frozen.get("cohort") != cohort_id:
            raise AssertionError(f"P007 frozen selection mismatch for {cohort_id}")
        frozen_sha = sha256(search / "selection_frozen.json")
        if search_status.get("selection_frozen_sha256") != frozen_sha:
            raise AssertionError(f"P007 frozen selection hash mismatch for {cohort_id}")
        if frozen.get("outer_evaluation_started") is not False or frozen.get("full_catalog_started") is not False:
            raise AssertionError(f"selection freeze records outer/full evaluation as started: {cohort_id}")
        queries_all = [json.loads(line) for line in
                       (task_root / cohort["development_queries"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        queries = [query for query in queries_all if query.get("split") == "validation"]
        if not queries:
            raise AssertionError(f"no validation queries for {cohort_id}")
        qids = [int(query["query_id"]) for query in queries]
        expected_gate_cells = sum(len(query["candidate_items"]) for query in queries) * 206
        final_models = search_status.get("final_models", [])
        final_by_label = {str(row.get("label")): row for row in final_models}
        if len(final_models) != len(final_by_label):
            raise AssertionError(f"duplicate P007 final model labels in {cohort_id}")
        expected_k_labels = set(final_by_label) | {"Popularity"}
        if set(final_by_label) != set(config.get("methods", [])):
            raise AssertionError(f"P007 search labels differ from frozen methods in {cohort_id}")
        masks = frozen.get("final_masks", {})
        if set(masks) != set(final_by_label):
            raise AssertionError(f"frozen mask labels differ from model labels in {cohort_id}")

        metrics100 = _json(search / "metrics100.json")
        if set(metrics100) != expected_k_labels:
            raise AssertionError(f"metrics100 labels differ from results in {cohort_id}")
        per_label_rows: dict[str, list[dict[str, Any]]] = {}
        per_label_metrics: dict[str, dict[str, Any]] = {}
        for label in expected_k_labels:
            rank_path = search / "ranks100" / f"{label}.jsonl"
            rows = _jsonl(rank_path)
            _check_rank_rows(rows, queries, cohort_id, label)
            if label == "Popularity":
                # Popularity scores are stored with each frozen development query.
                scores = [np.asarray(query["popularity_scores"], dtype=np.float64) for query in queries]
            else:
                result = final_by_label[label]
                scores = _prediction_scores(search, result, qids, queries)
            reconstructed = [_rank_row(query, score, cohort_id) for query, score in zip(queries, scores, strict=True)]
            for index, (saved, actual) in enumerate(zip(rows, reconstructed, strict=True)):
                for field in ("target_rank", "candidate_count", "user_id", "timestamp", "target_is_repeat", "target_known_before"):
                    if saved.get(field) != actual[field]:
                        raise AssertionError(f"{cohort_id}/{label}/{qids[index]} rank field {field} differs")
                _close(saved.get("target_score"), actual["target_score"], f"{cohort_id}/{label}/{qids[index]} target_score")
                if saved.get("top_items") != actual["top_items"]:
                    raise AssertionError(f"{cohort_id}/{label}/{qids[index]} top_items differ from score ranking")
            actual_metrics = summarize_ranks(rows)
            _compare_metrics(metrics100[label], actual_metrics, f"{cohort_id}/{label} metrics100")
            if label in CONTROL_ORIGINAL_LABEL:
                original_label = CONTROL_ORIGINAL_LABEL[label]
                original = _old_metric(task_root / cohort["development_run"] / "metrics.json", original_label)
                _compare_metrics(original, actual_metrics, f"{cohort_id}/{label} original validation metrics")
            per_label_rows[label] = rows
            per_label_metrics[label] = actual_metrics

        full_path = search / "full_catalog" / "run_status.json"
        full = _json(full_path)
        gate_name = str(full.get("gate_directory", "catalog_gate"))
        if Path(gate_name).name != gate_name or not gate_name.startswith("catalog_gate"):
            raise AssertionError(f"invalid successful gate directory in full-catalog status: {cohort_id}")
        gate_path = search / gate_name / "run_status.json"
        gate = _json(gate_path)
        if gate.get("status") != "complete" or gate.get("stage") != "gate":
            raise AssertionError(f"P007 catalog exact gate did not complete: {cohort_id}")
        if gate.get("exact_gate", {}).get("identical_float32_bytes") is not True:
            raise AssertionError(f"P007 catalog exact gate failed: {cohort_id}")
        if (int(gate.get("exact_gate", {}).get("cells", -1)) != expected_gate_cells or
                int(gate.get("exact_gate", {}).get("queries", -1)) != len(queries)):
            raise AssertionError(f"P007 catalog gate summary differs from K=100 feature rows: {cohort_id}")
        if full.get("status") != "complete" or full.get("stage") != "full_catalog" or full.get("cohort") != cohort_id:
            raise AssertionError(f"P007 full catalog run did not complete: {cohort_id}")
        if full.get("final_test_evaluated") is not False or full.get("selection_frozen_sha256") != frozen_sha:
            raise AssertionError(f"P007 full catalog flags/frozen mask mismatch: {cohort_id}")
        if full.get("all_k100_scores_exact") is not True or full.get("exact_gate", {}).get("identical_float32_bytes") is not True:
            raise AssertionError(f"P007 full catalog exact K=100 score gate missing: {cohort_id}")
        if (int(full.get("exact_gate", {}).get("cells", -1)) != expected_gate_cells or
                int(full.get("exact_gate", {}).get("queries", -1)) != len(queries)):
            raise AssertionError(f"P007 full-catalog feature gate summary mismatch: {cohort_id}")
        _source_hash_check(full, task_root, "source_sha256", cohort_id,
                           {"expanded_vector_features.py", "expanded_catalog.py", "run_expanded_catalog.py", "expanded_metrics.py"})
        metrics_full = _json(search / "full_catalog/metrics.json")
        full_hash_ledger = full.get("output_sha256", {})
        full_labels = set(final_by_label) | {"Popularity"}
        if set(full_hash_ledger) != {f"{label}.jsonl" for label in full_labels}:
            raise AssertionError(f"full-catalog output hash ledger labels mismatch in {cohort_id}")
        if set(metrics_full) != full_labels:
            raise AssertionError(f"full-catalog metrics labels mismatch in {cohort_id}")
        for label in full_labels:
            relative = f"{label}.jsonl"
            if full_hash_ledger.get(relative) != sha256(search / "full_catalog" / relative):
                raise AssertionError(f"full rank output hash mismatch: {cohort_id}/{relative}")
            full_rows = _jsonl(search / "full_catalog" / relative)
            if len(full_rows) != len(queries):
                raise AssertionError(f"full-catalog row count mismatch: {cohort_id}/{label}")
            base_rows = per_label_rows[label]
            if len(base_rows) != len(full_rows):
                raise AssertionError(f"K=100/full rows differ: {cohort_id}/{label}")
            for i, (base, row) in enumerate(zip(base_rows, full_rows, strict=True)):
                for field in ("cohort", "query_id", "user_id", "timestamp", "target_is_repeat", "target_known_before"):
                    if row.get(field) != base.get(field):
                        raise AssertionError(f"full/K=100 row metadata differs: {cohort_id}/{label}/{i}/{field}")
                if int(row["candidate_count"]) < int(base["candidate_count"]):
                    raise AssertionError(f"full catalog smaller than K=100: {cohort_id}/{label}/{i}")
                if int(row["candidate_count"]) < int(config.get("max_candidates", 100)):
                    raise AssertionError(f"full catalog has fewer than 100 items: {cohort_id}/{label}/{i}")
                top = row.get("top_items")
                if not isinstance(top, list) or len(top) != min(20, int(row["candidate_count"])):
                    raise AssertionError(f"invalid full-catalog top-20 list: {cohort_id}/{label}/{i}")
                top_pairs = [(float(item["score"]), int(item["item_id"])) for item in top]
                if any(top_pairs[j][0] < top_pairs[j + 1][0] or
                       (top_pairs[j][0] == top_pairs[j + 1][0] and top_pairs[j][1] >= top_pairs[j + 1][1])
                       for j in range(len(top_pairs) - 1)):
                    raise AssertionError(f"full top items are not score-desc/item-ID-asc: {cohort_id}/{label}/{i}")
                target_rank = row.get("target_rank")
                if base["target_rank"] is not None:
                    if target_rank is None or int(target_rank) < int(base["target_rank"]):
                        raise AssertionError(f"full-catalog target rank improved or target disappeared: {cohort_id}/{label}/{i}")
                    _close(row.get("target_score"), base.get("target_score"),
                           f"{cohort_id}/{label}/{i} K=100/full target score")
                if target_rank is not None and int(target_rank) <= 20:
                    if not any(int(item["item_id"]) == int(queries[i]["target_item"]) for item in top):
                        raise AssertionError(f"target in top-20 rank but missing from top_items: {cohort_id}/{label}/{i}")
            actual_full_metrics = summarize_ranks(full_rows)
            _compare_metrics(metrics_full[label], actual_full_metrics, f"{cohort_id}/{label} full metrics")
        verified_cohorts.append({"cohort": cohort_id, "queries": len(queries),
                                 "search_models": len(final_models), "rank100_groups": len(per_label_rows),
                                 "full_catalog_groups": len(full_labels),
                                 "exact_gate_cells": gate.get("exact_gate", {}).get("cells"),
                                 "status": "verified"})

    report = {"status": "verified", "experiment": "P007", "cohort_count": len(verified_cohorts),
              "cohorts": verified_cohorts, "tolerance": TOLERANCE,
              "independent_metric_implementation": "expanded_metrics.summarize_ranks; no evaluator or event scan"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else TASK / args.root
    output = args.output
    if output is not None and not output.is_absolute():
        output = TASK / output
    result = verify_expanded(root, output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
