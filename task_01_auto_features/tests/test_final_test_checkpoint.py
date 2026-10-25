import tempfile
import unittest
from pathlib import Path

import numpy as np

from task_01_auto_features.src.verify_auto_checkpoint import recompute
from task_01_auto_features.src.verify_final_test import (
    _nested_close,
    _validate_output_ledger,
    sha256,
    training_counts,
    validate_frozen_masks,
    validate_test_queries,
)


class FinalTestCheckpointTests(unittest.TestCase):
    def test_query_protocol_rejects_pre_boundary_wrong_split_and_duplicate_uid_time(self):
        base = {"query_id": 0, "user_id": 7, "timestamp": 100, "target_item": 10,
                "split": "validation", "evaluation_period": "test",
                "candidate_items": [10, 11], "popularity_scores": [1.0, 0.0]}
        counts = validate_test_queries([base], [7], start=100, per_user_limit=2)
        self.assertEqual(counts["query_count"], 1)
        with self.assertRaisesRegex(ValueError, "precedes"):
            validate_test_queries([dict(base, timestamp=99)], [7], start=100)
        with self.assertRaisesRegex(ValueError, "legacy validation/test"):
            validate_test_queries([dict(base, split="train")], [7], start=100)
        with self.assertRaisesRegex(ValueError, "duplicate UID/timestamp"):
            validate_test_queries([base, dict(base, query_id=1, target_item=12)], [7], start=100)

    def test_query_protocol_keeps_missing_retrieval_targets_and_limits_user_quota(self):
        rows = [{"query_id": i, "user_id": 7, "timestamp": 100 + i, "target_item": 99,
                 "split": "validation", "evaluation_period": "test",
                 "candidate_items": [10, 11], "popularity_scores": [1.0, 0.0]}
                for i in range(3)]
        with self.assertRaisesRegex(ValueError, "per-user query limit"):
            validate_test_queries(rows, [7], start=100, per_user_limit=2)
        # An absent target is intentionally not a protocol error: it stays a zero in ranking metrics.
        metric = recompute([dict(rows[0], target_is_repeat=False, target_known_before=False)], [[0.5, 0.5]])
        self.assertEqual(metric["counts"]["targets_missing_from_candidates"], 1)
        self.assertEqual(metric["overall"]["query_count"], 1)
        self.assertEqual(metric["overall"]["query_micro"]["ndcg@20"], 0.0)

    def test_score_ties_use_item_id_ascending(self):
        query = {"query_id": 0, "user_id": 1, "target_item": 9, "candidate_items": [9, 8],
                 "target_is_repeat": False, "target_known_before": True}
        result = recompute([query], [np.asarray([0.4, 0.4])], ks=(1, 2))
        self.assertEqual(result["overall"]["query_micro"]["recall@1"], 0.0)
        self.assertEqual(result["overall"]["query_micro"]["ndcg@2"], 1.0 / np.log2(3.0))

    def test_frozen_masks_must_match_development_columns_exactly(self):
        expected = {"M_I": [0, 1], "M_UIDC": [0, 1, 2], "Auto_A1": [4], "Auto_A2": [5], "Union": [0, 4]}
        actual = {key: list(value) for key, value in expected.items()}
        self.assertEqual(validate_frozen_masks(expected, actual, expected), expected)
        changed = dict(actual, Auto_A2=[6])
        with self.assertRaisesRegex(AssertionError, "development-frozen"):
            validate_frozen_masks(expected, changed, expected)
        duplicate_mask = dict(actual, Auto_A1=[1, 1])
        with self.assertRaisesRegex(ValueError, "invalid frozen feature columns"):
            validate_frozen_masks(expected, duplicate_mask, duplicate_mask)

    def test_training_counts_include_both_development_splits_but_not_test(self):
        queries = [
            {"split": "train", "timestamp": 10, "target_item": 1, "candidate_items": [1, 2]},
            {"split": "validation", "timestamp": 20, "target_item": 9, "candidate_items": [1, 2]},
        ]
        counts = training_counts(queries, test_start=30)
        self.assertEqual(counts["queries_requested"], 2)
        self.assertEqual(counts["queries_with_target"], 1)
        self.assertEqual(counts["queries_missing_target"], 1)
        self.assertEqual(counts["candidate_rows_examined"], 4)
        self.assertEqual(counts["candidate_rows_used"], 2)
        with self.assertRaisesRegex(ValueError, "test boundary"):
            training_counts([dict(queries[0], timestamp=30)], test_start=30)

    def test_child_hash_ledger_detects_modified_output_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "predictions" / "scores.npz"
            artifact.parent.mkdir()
            artifact.write_bytes(b"first")
            # P006's Windows runner serializes relative Path values with backslashes.
            status = {"output_sha256": {"predictions\\scores.npz": sha256(artifact)}}
            self.assertEqual(_validate_output_ledger(root, status)["predictions\\scores.npz"], sha256(artifact))
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(AssertionError, "SHA-256 mismatch"):
                _validate_output_ledger(root, status)

    def test_bootstrap_json_compare_tolerates_only_float_roundoff(self):
        expected = {"p001": {"ndcg20": {"mean": 0.01234567890123456, "users": 64}}}
        actual = {"p001": {"ndcg20": {"mean": 0.01234567890123457, "users": 64}}}
        _nested_close(expected, actual, tolerance=1e-12)
        with self.assertRaisesRegex(AssertionError, "beyond tolerance"):
            _nested_close(expected, {"p001": {"ndcg20": {"mean": 0.01235, "users": 64}}}, tolerance=1e-12)


if __name__ == "__main__":
    unittest.main()
