import unittest

from task_01_auto_features.src.verify_robustness_checkpoint import (
    _validate_split,
    pick_inner_winner,
    recompute,
    validate_frozen_winner,
)


class RobustnessCheckpointTests(unittest.TestCase):
    def test_recompute_is_analytical_and_keeps_missing_target_in_denominator(self):
        queries = [
            {"query_id": 0, "user_id": 7, "candidate_items": [11, 10, 12], "target_item": 10,
             "target_is_repeat": True, "target_known_before": True},
            {"query_id": 1, "user_id": 7, "candidate_items": [20, 21], "target_item": 99,
             "target_is_repeat": False, "target_known_before": False},
            {"query_id": 2, "user_id": 9, "candidate_items": [30, 31], "target_item": 31,
             "target_is_repeat": False, "target_known_before": True},
        ]
        # Query 2 is a score tie: item_id ascending puts 30 ahead of target 31.
        result = recompute(queries, [[0.5, 0.8, 0.2], [0.9, 0.1], [0.4, 0.4]], ks=(1, 2))
        overall = result["overall"]
        self.assertEqual(result["counts"]["targets_in_candidates"], 2)
        self.assertEqual(overall["query_count"], 3)
        self.assertEqual(overall["user_count"], 2)
        self.assertAlmostEqual(overall["query_micro"]["recall@1"], 1.0 / 3.0)
        self.assertAlmostEqual(overall["query_micro"]["ndcg@2"], (1.0 + 0.0 + 1.0 / 1.584962500721156) / 3.0)
        self.assertAlmostEqual(overall["user_macro"]["recall@1"], 0.25)
        self.assertEqual(result["strata"]["repeat"]["query_count"], 1)
        self.assertEqual(result["strata"]["cold"]["query_count"], 1)

    def test_inner_winner_uses_score_then_width_then_feature_ids(self):
        rows = [
            {"label": "wide", "inner_score": 0.5, "columns": [0, 1, 2], "spec_ids": ["a", "b", "c"]},
            {"label": "tie_z", "inner_score": 0.6, "columns": [0, 1], "spec_ids": ["z", "x"]},
            {"label": "tie_a", "inner_score": 0.6, "columns": [0, 1], "spec_ids": ["a", "y"]},
            {"label": "narrow", "inner_score": 0.6, "columns": [4], "spec_ids": ["q"]},
        ]
        winner = pick_inner_winner(rows)
        self.assertEqual(winner["label"], "narrow")
        same_width = pick_inner_winner(rows[1:3])
        self.assertEqual(same_width["label"], "tie_a")

    def test_frozen_winner_rejects_mismatch(self):
        actual = {"label": "inner_A1_top8", "inner_score": 0.25, "columns": [1, 2], "spec_ids": ["a", "b"]}
        validate_frozen_winner(dict(actual), actual)
        wrong = dict(actual, spec_ids=["a", "c"])
        with self.assertRaisesRegex(AssertionError, "feature IDs"):
            validate_frozen_winner(wrong, actual)

    def test_variable_size_split_alignment_and_equal_time_boundary(self):
        queries = [
            {"query_id": 0, "user_id": 1, "timestamp": 1, "split": "train", "candidate_items": [1]},
            {"query_id": 1, "user_id": 1, "timestamp": 2, "split": "train", "candidate_items": [2]},
            {"query_id": 2, "user_id": 2, "timestamp": 3, "split": "train", "candidate_items": [3]},
            {"query_id": 3, "user_id": 2, "timestamp": 4, "split": "validation", "candidate_items": [4]},
        ]
        split = {"cutoff": 3, "inner_fit": [0, 1], "inner_selection": [2], "external_validation": [3]}
        actual = _validate_split(queries, split)
        self.assertEqual(actual["query_counts"], {"inner_fit": 2, "inner_selection": 1, "external_validation": 1})
        self.assertEqual(actual["external_validation"], [3])

        bad = dict(split, inner_fit=[0, 2], inner_selection=[1])
        with self.assertRaisesRegex(ValueError, "timestamp boundary"):
            _validate_split(queries, bad)


if __name__ == "__main__":
    unittest.main()
