from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from task_01_auto_features.src.expanded_metrics import summarize_ranks
from task_01_auto_features.src.verify_expanded_results import verify_expanded


CONTROLS = ["M_I", "M_UIDC", "Original_A1", "Original_A2"]
ORIGINAL_LABELS = {"M_I": "M_I", "M_UIDC": "M_UIDC", "Original_A1": "Auto_A1", "Original_A2": "Auto_A2"}
COHORTS = ["p001", "p005_20261026", "p005_20261027"]
SEARCH_REQUIRED = {"run_expanded_search.py", "expanded_search.py", "expanded_metrics.py",
                   "auto_features.py", "pilot_features.py", "auto_search.py", "run_auto_features.py"}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_queries(cohort: str) -> list[dict]:
    base = int(COHORTS.index(cohort) * 100)
    return [
        {"query_id": base, "user_id": base + 1, "timestamp": 1000,
         "split": "validation", "target_item": 50, "candidate_items": [50, 2, 3],
         "target_is_repeat": True, "target_known_before": True,
         "popularity_scores": [2.0, 2.0, 0.0]},
        {"query_id": base + 1, "user_id": base + 2, "timestamp": 1001,
         "split": "validation", "target_item": 999, "candidate_items": [4, 5],
         "target_is_repeat": False, "target_known_before": False,
         "popularity_scores": [1.0, 0.0]},
    ]


def k100_rows(cohort: str, label: str, queries: list[dict]) -> list[dict]:
    scores = [[2.0, 2.0, 0.0], [1.0, 0.0]]
    rows = []
    for q, score in zip(queries, scores):
        items = q["candidate_items"]
        order = sorted(range(len(items)), key=lambda i: (-score[i], items[i]))
        idx = items.index(q["target_item"]) if q["target_item"] in items else None
        rank = next((pos + 1 for pos, i in enumerate(order) if i == idx), None)
        rows.append({"cohort": cohort, "query_id": q["query_id"], "user_id": q["user_id"],
                     "timestamp": q["timestamp"], "candidate_count": len(items), "target_rank": rank,
                     "target_score": None if idx is None else score[idx],
                     "target_is_repeat": q["target_is_repeat"], "target_known_before": q["target_known_before"],
                     "top_items": [{"item_id": items[i], "score": score[i]} for i in order[:20]]})
    return rows


def full_rows(cohort: str, label: str, base_rows: list[dict]) -> list[dict]:
    rows = []
    for i, base in enumerate(base_rows):
        n = 100
        # Keep target score exactly equal to K=100 while adding high-scoring
        # lower-ID catalog items so the full rank cannot improve.
        target = 50 if i == 0 else 999
        rank = (5 if i == 0 else None)
        values = [3.0] * 4 + [1.0] * (n - 4)
        if rank is not None:
            values[target - 1] = base["target_score"]
            target_score = base["target_score"]
            # Equal-score ties are resolved by item ID; item IDs before the target
            # are smaller, matching the declared one-based rank.
        else:
            target_score = None
        # Ensure stable unique catalog IDs and score/item tie order.
        top_order = sorted(range(n), key=lambda j: (-values[j], j + 1))[:20]
        rows.append({"cohort": cohort, "query_id": base["query_id"], "user_id": base["user_id"],
                     "timestamp": base["timestamp"], "candidate_count": n,
                     "target_rank": rank, "target_score": target_score,
                     "target_is_repeat": base["target_is_repeat"],
                     "target_known_before": base["target_known_before"],
                     "top_items": [{"item_id": j + 1, "score": values[j]} for j in top_order]})
    return rows


class VerifyExpandedResultsTests(unittest.TestCase):
    def _fixture(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name) / "artifacts/p007_expanded_search"
        task = Path(temp.name) / "task_01_auto_features"
        config = {"cohorts": [{"id": cohort, "development_run": f"artifacts/dev_{cohort}",
                               "development_queries": f"artifacts/queries_{cohort}.jsonl"} for cohort in COHORTS],
                  "metrics_k": [10, 20], "methods": CONTROLS, "max_candidates": 100}
        write_json(task / "configs/p007_expanded.json", config)
        source_names = {"run_expanded_search.py", "expanded_search.py", "expanded_metrics.py",
                        "auto_features.py", "pilot_features.py", "auto_search.py", "run_auto_features.py",
                        "expanded_vector_features.py", "expanded_catalog.py", "run_expanded_catalog.py"}
        source_hashes = {}
        frozen_hashes = {}
        for name in source_names:
            source_path = task / "src" / name
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(f"synthetic {name}\n", encoding="utf-8")
            source_hashes[name] = digest(source_path)
        for ci, cohort in enumerate(COHORTS):
            qroot = task / f"artifacts/queries_{cohort}.jsonl"
            queries = validation_queries(cohort)
            qroot.parent.mkdir(parents=True, exist_ok=True)
            qroot.write_text("".join(json.dumps(q) + "\n" for q in queries), encoding="utf-8")
            dev_run = task / f"artifacts/dev_{cohort}"
            dev_run.mkdir(parents=True, exist_ok=True)
            with (dev_run / "feature_inputs.jsonl").open("w", encoding="utf-8") as stream:
                stream.write("".join(json.dumps({key: q[key] for key in
                    ("query_id", "user_id", "timestamp", "candidate_items")}) + "\n" for q in queries))
            feature_names = ["i_listen_all"]
            write_json(dev_run / "feature_schema.json", {"feature_names": feature_names})
            feature_values = np.concatenate([np.asarray(q["popularity_scores"], dtype=np.float32)
                                             for q in queries]).reshape(-1, 1)
            np.save(dev_run / "features.npy", feature_values)
            for name in ("features.npy", "feature_inputs.jsonl", "feature_schema.json"):
                frozen_hashes[f"artifacts/dev_{cohort}/{name}"] = digest(dev_run / name)
            search = root / cohort
            (search / "ranks100").mkdir(parents=True)
            (search / "predictions").mkdir()
            masks = {label: [i] for i, label in enumerate(CONTROLS)}
            frozen = {"experiment": "P007", "cohort": cohort, "final_masks": masks,
                      "outer_evaluation_started": False, "full_catalog_started": False}
            write_json(search / "selection_frozen.json", frozen)
            frozen_sha = digest(search / "selection_frozen.json")
            final_models = []
            rows_by_label = {}
            for label in CONTROLS:
                scores = [np.array([2.0, 2.0, 0.0]), np.array([1.0, 0.0])]
                pred_path = search / "predictions" / f"{label}.npz"
                np.savez_compressed(pred_path, query_ids=np.asarray([q["query_id"] for q in queries]),
                                    offsets=np.asarray([0, 3, 5]), scores=np.concatenate(scores))
                rows = k100_rows(cohort, label, queries)
                rows_by_label[label] = rows
                with (search / "ranks100" / f"{label}.jsonl").open("w", encoding="utf-8") as stream:
                    stream.write("".join(json.dumps(row) + "\n" for row in rows))
                final_models.append({"label": label, "stage": "outer_validation", "columns": masks[label],
                                     "prediction_file": f"predictions/{label}.npz", "prediction_sha256": digest(pred_path),
                                     "metrics": summarize_ranks(rows)})
            pop_rows = k100_rows(cohort, "Popularity", queries)
            # Popularity is computed from each query's saved popularity scores.
            with (search / "ranks100/Popularity.jsonl").open("w", encoding="utf-8") as stream:
                stream.write("".join(json.dumps(row) + "\n" for row in pop_rows))
            metrics100 = {label: summarize_ranks(rows) for label, rows in {**rows_by_label, "Popularity": pop_rows}.items()}
            write_json(search / "metrics100.json", metrics100)
            status = {"experiment": "P007", "cohort": cohort, "status": "complete",
                      "config_sha256": "cfg", "protocol_sha256": "proto", "source_sha256": {},
                      "final_test_evaluated": False, "selection_frozen_sha256": frozen_sha,
                      "final_models": final_models}
            status["source_sha256"] = {name: source_hashes[name] for name in SEARCH_REQUIRED}
            write_json(search / "run_status.json", status)

            full_dir = search / "full_catalog"
            full_dir.mkdir()
            full_metrics = {}
            output_hashes = {}
            for label in [*CONTROLS, "Popularity"]:
                rows = full_rows(cohort, label, rows_by_label[label] if label in rows_by_label else pop_rows)
                path = full_dir / f"{label}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                output_hashes[path.name] = digest(path)
                full_metrics[label] = summarize_ranks(rows)
            write_json(full_dir / "metrics.json", full_metrics)
            write_json(full_dir / "run_status.json", {"experiment": "P007", "cohort": cohort,
                      "stage": "full_catalog", "status": "complete", "final_test_evaluated": False,
                      "selection_frozen_sha256": frozen_sha, "all_k100_scores_exact": True,
                      "gate_directory": "catalog_gate_v2",
                      "exact_gate": {"identical_float32_bytes": True, "cells": 5 * 206, "queries": 2},
                      "source_sha256": {name: source_hashes[name] for name in
                          ("expanded_vector_features.py", "expanded_catalog.py", "run_expanded_catalog.py", "expanded_metrics.py")},
                      "output_sha256": output_hashes})
            write_json(search / "catalog_gate/run_status.json", {"experiment": "P007", "cohort": cohort,
                      "stage": "gate", "status": "failed", "error": "synthetic initial gate failure"})
            write_json(search / "catalog_gate_v2/run_status.json", {"experiment": "P007", "cohort": cohort,
                      "stage": "gate", "status": "complete", "exact_gate": {
                          "identical_float32_bytes": True, "cells": 5 * 206, "queries": 2}})
            # Create the frozen original-control metric sources consumed by verifier.
            old_status = {"final_models": []}
            for label in CONTROLS:
                current = summarize_ranks(rows_by_label[label])["overall"]
                historical = {"query_count": current["query_count"], "user_count": current["user_count"],
                              "candidate_recall": current["candidate_recall"],
                              "target_known_before_count": current["target_known_before_count"],
                              "target_known_before_fraction": current["target_known_before_count"] / current["query_count"],
                              "repeat_count": current["repeat_count"],
                              "repeat_fraction": current["repeat_count"] / current["query_count"],
                              "query_micro": current["query_micro"], "user_macro": current["user_macro"]}
                old_status["final_models"].append({"label": ORIGINAL_LABELS[label],
                    "metrics": {"by_split": {"validation": historical}}})
            write_json(task / f"{dev_run}/metrics.json", old_status)
        write_json(task / "configs/p006_freeze_manifest.json", {"files": frozen_hashes})
        return temp, root

    def test_verifies_three_cohorts_and_all_rank_metrics(self):
        temp, root = self._fixture()
        try:
            result = verify_expanded(root, task_root=Path(temp.name) / "task_01_auto_features")
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["cohort_count"], 3)
            self.assertTrue(all(row["status"] == "verified" for row in result["cohorts"]))
            self.assertTrue((root / "verification.json").is_file())
        finally:
            temp.cleanup()

    def test_missing_k100_rows_are_rejected(self):
        temp, root = self._fixture()
        try:
            path = root / "p001/ranks100/M_I.jsonl"
            path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
            with self.assertRaises(AssertionError):
                verify_expanded(root, task_root=Path(temp.name) / "task_01_auto_features")
        finally:
            temp.cleanup()

    def test_historical_metric_schema_is_compared_and_mismatch_rejected(self):
        temp, root = self._fixture()
        try:
            task = Path(temp.name) / "task_01_auto_features"
            status_path = task / "artifacts/dev_p001/metrics.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            metric = status["final_models"][2]["metrics"]["by_split"]["validation"]
            metric["candidate_recall"] = 0.75
            write_json(status_path, status)
            with self.assertRaisesRegex(AssertionError, "candidate_recall"):
                verify_expanded(root, task_root=task)
        finally:
            temp.cleanup()

    def test_popularity_uses_frozen_matrix_when_query_scores_are_absent(self):
        temp, root = self._fixture()
        try:
            task = Path(temp.name) / "task_01_auto_features"
            queries_path = task / "artifacts/queries_p001.jsonl"
            queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines()]
            for query in queries:
                query.pop("popularity_scores")
            queries_path.write_text("".join(json.dumps(q) + "\n" for q in queries), encoding="utf-8")
            # Query files are not the frozen-input boundary for this synthetic case;
            # the matrix and structural feature_inputs remain pinned and authoritative.
            result = verify_expanded(root, output=Path(temp.name) / "report.json", task_root=task)
            self.assertEqual(result["status"], "verified")
            with self.assertRaisesRegex(AssertionError, "target_rank differs"):
                path = root / "p001/ranks100/Popularity.jsonl"
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                rows[0]["target_rank"] = 1
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                verify_expanded(root, output=Path(temp.name) / "report2.json", task_root=task)
        finally:
            temp.cleanup()

    def test_full_catalog_rank_better_than_subset_is_rejected(self):
        temp, root = self._fixture()
        try:
            path = root / "p001/full_catalog/M_I.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["target_rank"] = 1
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            status_path = root / "p001/full_catalog/run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["output_sha256"][path.name] = digest(path)
            write_json(status_path, status)
            with self.assertRaises(AssertionError):
                verify_expanded(root, task_root=Path(temp.name) / "task_01_auto_features")
        finally:
            temp.cleanup()

    def test_full_catalog_tie_order_is_checked_even_with_updated_hash(self):
        temp, root = self._fixture()
        try:
            path = root / "p001/full_catalog/M_I.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["top_items"][0], rows[0]["top_items"][1] = rows[0]["top_items"][1], rows[0]["top_items"][0]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            status_path = root / "p001/full_catalog/run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["output_sha256"][path.name] = digest(path)
            write_json(status_path, status)
            with self.assertRaises(AssertionError):
                verify_expanded(root, task_root=Path(temp.name) / "task_01_auto_features")
        finally:
            temp.cleanup()

    def test_target_rank_over_20_cannot_still_appear_in_top_items(self):
        temp, root = self._fixture()
        try:
            cohort = "p001"
            path = root / cohort / "full_catalog/M_I.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["target_rank"] = 21
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            metrics_path = root / cohort / "full_catalog/metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["M_I"] = summarize_ranks(rows)
            write_json(metrics_path, metrics)
            status_path = root / cohort / "full_catalog/run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["output_sha256"][path.name] = digest(path)
            write_json(status_path, status)
            with self.assertRaisesRegex(AssertionError, "outside top-20 appears"):
                verify_expanded(root, task_root=Path(temp.name) / "task_01_auto_features")
        finally:
            temp.cleanup()

    def test_full_rank_file_hash_is_enforced(self):
        temp, root = self._fixture()
        try:
            path = root / "p001/full_catalog/M_I.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(AssertionError):
                verify_expanded(root, task_root=Path(temp.name) / "task_01_auto_features")
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
