from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from task_01_auto_features.src.verify_sampled_training import (
    catalog_before,
    catalog_sha256,
    choose_teacher_hard_items,
    derive_sampling_seed,
    expected_sampled_blocks,
    sample_hard_pool_uniform,
    sample_uniform_draw,
    validate_prepare_record,
    _check_full_rows,
    _rank_row,
    _sha256,
    verify_sampled_training,
)
from task_01_auto_features.src.expanded_metrics import summarize_ranks


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_prediction(path: Path, qid: int, scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, query_ids=np.asarray([qid]), offsets=np.asarray([0, len(scores)]), scores=scores)


def _hash_files(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): _sha256(path) for path in folder.rglob("*")
            if path.is_file() and path.name != "run_status.json"}


def _make_full_fixture(root: Path) -> tuple[Path, Path]:
    task = root / "task_01_auto_features"
    artifacts = task / "artifacts"
    (task / "configs").mkdir(parents=True)
    (task / "src").mkdir()
    config = {"p007_root": "artifacts/p007_expanded_search", "seed": 20261025,
              "feature_count": 66, "max_candidates": 100,
              "sampling": {"seed": 20261103, "uniform_negatives": 99, "hybrid_uniform": 50,
                           "hybrid_hard": 49, "hard_pool_uniform": 256},
              "methods": ["Baseline_UIDC", "Baseline_A1", "Uniform_UIDC", "Uniform_A1",
                          "HybridHit_UIDC", "HybridHit_A1", "HybridWarm_UIDC", "HybridWarm_A1", "Popularity"],
              "cohorts": [{"id": f"p00{i}", "development_run": "artifacts/dev",
                           "development_queries": f"artifacts/queries_p00{i}.jsonl"} for i in range(1, 4)]}
    config_path = task / "configs/p008_sampled.json"
    protocol_path = task / "configs/P008_PROTOCOL.md"
    _write_json(config_path, config)
    protocol_path.write_text("synthetic protocol", encoding="utf-8")
    (task / "src/fake.py").write_text("pinned", encoding="utf-8")
    fake_source_hash = _sha256(task / "src/fake.py")
    config_sha, protocol_sha = _sha256(config_path), _sha256(protocol_path)
    output_root = artifacts / "p008_sampled_training"
    p007_root = artifacts / "p007_expanded_search"
    masks = {"UIDC": list(range(66)), "A1": [0, 1]}
    item_ids = np.arange(400, dtype=np.int64)
    first_seen = np.zeros(400, dtype=np.int64)
    train_q = {"query_id": 1, "user_id": 7, "timestamp": 100, "split": "train",
               "target_item": 50, "candidate_items": list(range(100)),
               "target_is_repeat": True, "target_known_before": True}
    valid_q = {"query_id": 2, "user_id": 7, "timestamp": 200, "split": "validation",
               "target_item": 50, "candidate_items": list(range(100)),
               "target_is_repeat": True, "target_known_before": True}
    candidate_scores = np.arange(100, dtype=np.float64)
    p007_scores_by_old = {"M_UIDC": candidate_scores, "Original_A1": candidate_scores,
                          "Popularity": candidate_scores}
    p007_model_info: dict[str, dict] = {}
    fit_status_paths = {}

    for ci in range(3):
        cohort_id = f"p00{ci + 1}"
        expected_rows = {old: [_rank_row(valid_q, scores, cohort_id)]
                         for old, scores in p007_scores_by_old.items()}
        query_path = task / f"artifacts/queries_{cohort_id}.jsonl"
        _write_jsonl(query_path, [train_q, valid_q])
        p007_dir = p007_root / cohort_id
        p007_dir.mkdir(parents=True, exist_ok=True)
        _write_json(p007_dir / "selection_frozen.json", {"final_masks": {"M_UIDC": masks["UIDC"],
                    "Original_A1": masks["A1"]}})
        final_models = []
        for old in ("M_UIDC", "Original_A1"):
            model_file = p007_dir / "models" / f"{old}.cbm"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            model_file.write_bytes((cohort_id + old).encode())
            prediction_file = p007_dir / "predictions" / f"{old}.npz"
            _write_prediction(prediction_file, valid_q["query_id"], candidate_scores)
            p007_model_info[(cohort_id, old)] = {"model_file": f"models/{old}.cbm",
                "model_sha256": _sha256(model_file), "prediction_file": f"predictions/{old}.npz",
                "training": {"queries": 1, "rows": 100, "positives": 1}}
            final_models.append({"label": old, **p007_model_info[(cohort_id, old)]})
        for folder_name, filename in (("ranks100", "M_UIDC"), ("ranks100", "Original_A1"),
                                      ("ranks100", "Popularity"), ("full_catalog", "M_UIDC"),
                                      ("full_catalog", "Original_A1"), ("full_catalog", "Popularity")):
            old = filename
            _write_jsonl(p007_dir / folder_name / f"{filename}.jsonl", expected_rows[old])
        _write_json(p007_dir / "run_status.json", {"status": "complete", "cohort": cohort_id,
                    "final_models": final_models})

        folder = output_root / cohort_id
        prepare_dir, fit_dir, gate_dir, full_dir = (folder / name for name in ("prepare", "fit", "gate", "full"))
        prepare_dir.mkdir(parents=True)
        catalog = item_ids.tolist()
        uniform_draw = sample_uniform_draw(20261103, cohort_id, train_q["query_id"], catalog, train_q["target_item"])
        first50 = uniform_draw[:50]
        extra_pool = sample_hard_pool_uniform(20261103, cohort_id, train_q["query_id"], catalog,
                                               train_q["target_item"], first50)
        pool = sorted((set(train_q["candidate_items"]) | set(extra_pool)) - {train_q["target_item"], *first50})
        sample = expected_sampled_blocks(20261103, cohort_id, train_q["query_id"], train_q["target_item"],
                                         catalog, train_q["candidate_items"], {item: float(item) for item in pool})
        record = {"cohort": cohort_id, "query_id": train_q["query_id"], "user_id": train_q["user_id"],
                  "timestamp": train_q["timestamp"], "target_item": train_q["target_item"], "split": "train",
                  "old_hit": True, "warm": True, "cold": False, "catalog_size": len(catalog),
                  "catalog_sha256": catalog_sha256(catalog), "uniform_row_start": 0, "hybrid_row_start": 0,
                  "sampling_status": "sampled", "seed_uniform99": derive_sampling_seed(20261103, cohort_id, 1, "uniform99"),
                  "seed_hard_pool256": derive_sampling_seed(20261103, cohort_id, 1, "hard_pool256"),
                  "hard_pool_scores": [float(item) for item in sample["hard_pool_items"]], **sample}
        _write_jsonl(prepare_dir / "records.jsonl", [record])
        np.savez_compressed(prepare_dir / "item_first_seen.npz", item_ids=item_ids, first_seen=first_seen)
        np.save(prepare_dir / "uniform.npy", np.zeros((100, 66), dtype=np.float32))
        np.save(prepare_dir / "hybrid.npy", np.zeros((100, 66), dtype=np.float32))
        prepare_status = {"experiment": "P008", "cohort": cohort_id, "stage": "prepare", "status": "complete",
            "final_test_evaluated": False, "config_sha256": config_sha, "protocol_sha256": protocol_sha,
            "source_sha256": {"fake.py": fake_source_hash}, "outputs_sha256": _hash_files(prepare_dir),
            "masks": masks, "teacher_model_file": f"artifacts/p007_expanded_search/{cohort_id}/models/M_UIDC.cbm",
            "teacher_model_sha256": p007_model_info[(cohort_id, "M_UIDC")]["model_sha256"],
            "query_counts": {"train": 1, "hit": 1, "warm": 1, "cold": 0},
            "exact_gate": {"identical_float32_bytes": True, "cells": 100 * 66, "queries": 1}}
        _write_json(prepare_dir / "run_status.json", prepare_status)

        fit_dir.mkdir()
        (fit_dir / "models").mkdir()
        training_frozen = {"arms": {"Uniform": [1], "HybridHit": [1], "HybridWarm": [1]}, "masks": masks}
        _write_json(fit_dir / "training_frozen.json", training_frozen)
        models = []
        for label, old in (("Baseline_UIDC", "M_UIDC"), ("Baseline_A1", "Original_A1")):
            info = p007_model_info[(cohort_id, old)]
            models.append({"label": label, "arm": "Baseline", "columns": masks["UIDC" if old == "M_UIDC" else "A1"],
                           "model_origin": "p007", "model_file": f"artifacts/p007_expanded_search/{cohort_id}/{info['model_file']}",
                           "model_sha256": info["model_sha256"], "training": info["training"]})
        new_labels = ["Uniform_UIDC", "Uniform_A1", "HybridHit_UIDC", "HybridHit_A1",
                      "HybridWarm_UIDC", "HybridWarm_A1"]
        for label in new_labels:
            model_path = fit_dir / "models" / f"{label}.cbm"
            model_path.write_bytes(label.encode())
            arm, mask = label.rsplit("_", 1)
            models.append({"label": label, "arm": arm, "columns": masks[mask], "model_origin": "p008",
                           "model_file": f"models/{label}.cbm", "model_sha256": _sha256(model_path),
                           "training_query_ids": [1], "training": {"queries": 1, "rows": 100, "positives": 1}})
        fit_status = {"experiment": "P008", "cohort": cohort_id, "stage": "fit", "status": "complete",
            "final_test_evaluated": False, "config_sha256": config_sha, "protocol_sha256": protocol_sha,
            "source_sha256": {"fake.py": fake_source_hash}, "prepare_status_sha256": _sha256(prepare_dir / "run_status.json"),
            "training_frozen_sha256": _sha256(fit_dir / "training_frozen.json"), "models": models}
        _write_json(fit_dir / "run_status.json", fit_status)
        fit_status_paths[cohort_id] = _sha256(fit_dir / "run_status.json")

        gate_dir.mkdir()
        _write_json(gate_dir / "run_status.json", {"experiment": "P008", "cohort": cohort_id,
            "stage": "gate", "status": "complete", "final_test_evaluated": False,
            "config_sha256": config_sha, "protocol_sha256": protocol_sha, "source_sha256": {"fake.py": fake_source_hash},
            "fit_status_sha256": _sha256(fit_dir / "run_status.json"),
            "exact_gate": {"identical_float32_bytes": True, "cells": 100 * 66, "queries": 1,
                           "values_sha256": "a" * 64}})

        full_dir.mkdir()
        (full_dir / "ranks100").mkdir()
        (full_dir / "ranksfull").mkdir()
        (full_dir / "predictions").mkdir()
        labels = config["methods"]
        for label in labels:
            scores = candidate_scores
            row = _rank_row(valid_q, scores, cohort_id)
            _write_jsonl(full_dir / "ranks100" / f"{label}.jsonl", [row])
            _write_jsonl(full_dir / "ranksfull" / f"{label}.jsonl", [row])
            _write_prediction(full_dir / "predictions" / f"{label}.npz", valid_q["query_id"], scores)
        _write_json(full_dir / "metrics100.json", {label: summarize_ranks([_rank_row(valid_q, candidate_scores, cohort_id)])
                    for label in labels})
        _write_json(full_dir / "metrics_full.json", {label: summarize_ranks([_rank_row(valid_q, candidate_scores, cohort_id)])
                    for label in labels})
        full_status = {"experiment": "P008", "cohort": cohort_id, "stage": "full", "status": "complete",
            "final_test_evaluated": False, "config_sha256": config_sha, "protocol_sha256": protocol_sha,
            "source_sha256": {"fake.py": fake_source_hash},
            "fit_status_sha256": _sha256(fit_dir / "run_status.json"), "reference_parity_exact": True,
            "exact_gate": {"identical_float32_bytes": True, "cells": 100 * 66, "queries": 1,
                           "values_sha256": "a" * 64}}
        _write_json(full_dir / "run_status.json", full_status)
        full_status["outputs_sha256"] = _hash_files(full_dir)
        _write_json(full_dir / "run_status.json", full_status)

    all_fits = {"experiment": "P008", "frozen_before_evaluation": True, "config_sha256": config_sha,
                "protocol_sha256": protocol_sha, "source_sha256": {"fake.py": fake_source_hash},
                "cohort_fit_status_sha256": fit_status_paths}
    _write_json(output_root / "all_fits_frozen.json", all_fits)
    all_fits_sha = _sha256(output_root / "all_fits_frozen.json")
    for cohort in config["cohorts"]:
        folder = output_root / cohort["id"]
        for stage in ("gate", "full"):
            path = folder / stage / "run_status.json"
            status = json.loads(path.read_text(encoding="utf-8"))
            status["all_fits_frozen_sha256"] = all_fits_sha
            if stage == "full":
                status["outputs_sha256"] = _hash_files(folder / stage)
            _write_json(path, status)
    return output_root, task


class VerifySampledTrainingTests(unittest.TestCase):
    def test_full_synthetic_p008_verification_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, task = _make_full_fixture(Path(tmp))
            receipt_path = root / "verification_independent.json"
            receipt = verify_sampled_training(root, task, output=receipt_path)
            self.assertEqual(receipt["status"], "verified")
            self.assertEqual(receipt["cohort_count"], 3)
            self.assertTrue(receipt_path.is_file())

    def test_seed_framing_is_exact(self):
        framed = b"20261103|p005_20261026|123|uniform99"
        expected = int.from_bytes(hashlib.sha256(framed).digest()[:8], "big")
        self.assertEqual(derive_sampling_seed(20261103, "p005_20261026", 123, "uniform99"), expected)
        self.assertEqual(derive_sampling_seed(20261103, "p005_20261026", 123, "uniform99"), expected)
        self.assertNotEqual(expected, derive_sampling_seed(20261103, "p005_20261026", 123, "hard_pool256"))

    def test_uniform99_draw_is_reproducible_and_excludes_target(self):
        catalog = list(range(400))
        first = sample_uniform_draw(20261103, "p001", 20, catalog, 177)
        second = sample_uniform_draw(20261103, "p001", 20, catalog, 177)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 99)
        self.assertEqual(len(set(first)), 99)
        self.assertNotIn(177, first)
        self.assertEqual(set(first), set(sample_uniform_draw(20261103, "p001", 20, catalog, 177)))

    def test_hard_pool_excludes_target_and_first_fifty_uniforms(self):
        catalog = list(range(500))
        uniform = sample_uniform_draw(20261103, "p001", 20, catalog, 177)[:50]
        hard_draw = sample_hard_pool_uniform(20261103, "p001", 20, catalog, 177, uniform)
        self.assertEqual(len(hard_draw), 256)
        self.assertEqual(len(set(hard_draw)), len(hard_draw))
        self.assertNotIn(177, hard_draw)
        self.assertFalse(set(uniform) & set(hard_draw))

    def test_teacher_hard_selection_uses_score_then_item_id(self):
        selected = choose_teacher_hard_items([9, 2, 5], [1.0, 2.0, 2.0], count=2)
        self.assertEqual(selected, [2, 5])

    def test_catalog_uses_strict_first_seen_and_canonical_hash(self):
        ids = np.asarray([2, 5, 8], dtype=np.int64)
        first_seen = np.asarray([10, 11, 9], dtype=np.int64)
        self.assertEqual(catalog_before(ids, first_seen, 11), [2, 8])
        expected = hashlib.sha256(np.asarray([2, 8], dtype="<i8").tobytes()).hexdigest()
        self.assertEqual(catalog_sha256([8, 2]), expected)

    def test_prepare_validation_enforces_hit_subset_warm_and_sampling(self):
        catalog = list(range(400))
        q = {"query_id": 123, "user_id": 7, "timestamp": 10, "target_item": 177,
             "candidate_items": [177, 1, 2]}
        expected_u = {"uniform_items": sorted({177, *sample_uniform_draw(20261103, "c", 123, catalog, 177)}),
                      "uniform_draw_order": sample_uniform_draw(20261103, "c", 123, catalog, 177)}
        # Use distinct deterministic teacher scores over the reconstructed candidate pool.
        uniform50 = expected_u["uniform_draw_order"][:50]
        extra = sample_hard_pool_uniform(20261103, "c", 123, catalog, 177, uniform50)
        pool_items = sorted((set(q["candidate_items"]) | set(extra)) - {177, *uniform50})
        score_map = {item: float(item) for item in pool_items}
        expected_h = expected_sampled_blocks(20261103, "c", 123, 177, catalog,
                                             q["candidate_items"], score_map)
        record = {"query_id": 123, "user_id": 7, "timestamp": 10, "target_item": 177,
                  "cohort": "c", "old_hit": True, "warm": True, "cold": False, "sampling_status": "sampled",
                  "seed_uniform99": derive_sampling_seed(20261103, "c", 123, "uniform99"),
                  "seed_hard_pool256": derive_sampling_seed(20261103, "c", 123, "hard_pool256"),
                  "catalog_size": len(catalog),
                  "catalog_sha256": catalog_sha256(catalog), "uniform_row_start": 0,
                  "hybrid_row_start": 0, "hard_pool_scores": [score_map[item] for item in expected_h["hard_pool_items"]],
                  **expected_u, **expected_h}
        validate_prepare_record(record, q, catalog, expected_u, expected_h)
        record["old_hit"] = False
        with self.assertRaisesRegex(AssertionError, "old_hit/warm"):
            validate_prepare_record(record, q, catalog, expected_u, expected_h)

    def test_cold_query_has_no_matrix_rows(self):
        q = {"query_id": 1, "user_id": 4, "timestamp": 30, "target_item": 90,
             "candidate_items": [3, 4]}
        catalog = [1, 2, 3]
        record = {"query_id": 1, "user_id": 4, "timestamp": 30, "target_item": 90,
                  "old_hit": False, "warm": False, "cold": True, "catalog_size": 3,
                  "catalog_sha256": catalog_sha256(catalog), "uniform_row_start": None,
                  "hybrid_row_start": None, "uniform_items": [], "hybrid_items": [],
                  "sampling_status": "cold_skip", "seed_uniform99": None, "seed_hard_pool256": None}
        validate_prepare_record(record, q, catalog, None, None)
        record["hybrid_row_start"] = 0
        with self.assertRaisesRegex(AssertionError, "cold query"):
            validate_prepare_record(record, q, catalog, None, None)

    def test_full_rank_evidence_rejects_wrong_top_item_order(self):
        query = {"query_id": 1, "user_id": 3, "timestamp": 20,
                 "target_is_repeat": False, "target_known_before": True,
                 "target_item": 99, "candidate_items": list(range(100))}
        top = [{"item_id": item, "score": float(100 - item)} for item in range(20)]
        row = {"cohort": "c", "query_id": 1, "user_id": 3, "timestamp": 20,
               "target_is_repeat": False, "target_known_before": True,
               "candidate_count": 100, "target_rank": 1, "target_score": 1.0,
               "top_items": top}
        # The top list is score-descending but inconsistent with item IDs and target rank.
        row["top_items"][0], row["top_items"][1] = row["top_items"][1], row["top_items"][0]
        with self.assertRaises(AssertionError):
            _check_full_rows([row], [query], [row], "c", "M")


if __name__ == "__main__":
    unittest.main()
