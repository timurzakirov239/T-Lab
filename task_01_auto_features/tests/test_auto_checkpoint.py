import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from task_01_auto_features.src.verify_auto_checkpoint import recompute, sha256


class IndependentCheckpointTests(unittest.TestCase):
    def test_independent_metrics_tie_break_rank_three_and_missing_target(self):
        queries = [
            {"query_id": 1, "user_id": 10, "candidate_items": [30, 10, 20], "target_item": 10,
             "target_is_repeat": True, "target_known_before": True},
            {"query_id": 2, "user_id": 10, "candidate_items": [1, 2, 3], "target_item": 3,
             "target_is_repeat": False, "target_known_before": True},
            {"query_id": 3, "user_id": 20, "candidate_items": [7, 8, 9], "target_item": 99,
             "target_is_repeat": False, "target_known_before": False},
        ]
        # Query 1 ties score; item_id=10 wins score tie, so target ranks 1st.
        # Query 2 target ranks 3rd (rank 2); query 3 is missing and remains a zero.
        scores = [[1.0, 1.0, 0.0], [0.9, 0.8, 0.7], [0.5, 0.4, 0.3]]
        result = recompute(queries, scores, ks=(1, 2, 3))
        self.assertEqual(result["overall"]["query_count"], 3)
        self.assertEqual(result["overall"]["user_count"], 2)
        self.assertAlmostEqual(result["overall"]["query_micro"]["recall@1"], 1 / 3)
        self.assertAlmostEqual(result["overall"]["query_micro"]["recall@2"], 1 / 3)
        self.assertAlmostEqual(result["overall"]["query_micro"]["recall@3"], 2 / 3)
        expected_ndcg3 = (1 + 1 / (np.log2(4))) / 3
        self.assertAlmostEqual(result["overall"]["query_micro"]["ndcg@3"], expected_ndcg3)
        self.assertEqual(result["counts"]["targets_in_candidates"], 2)
        self.assertEqual(result["counts"]["targets_missing_from_candidates"], 1)
        self.assertEqual(result["strata"]["repeat"]["query_count"], 1)
        self.assertEqual(result["strata"]["new"]["query_count"], 2)
        self.assertEqual(result["strata"]["warm"]["query_count"], 2)
        self.assertEqual(result["strata"]["cold"]["candidate_recall"], 0.0)

    def test_hash_helper_matches_saved_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tiny.bin"
            path.write_bytes(b"verifier fixture")
            import hashlib
            self.assertEqual(sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
