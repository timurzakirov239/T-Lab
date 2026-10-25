from __future__ import annotations

import math
import unittest

from task_01_auto_features.src.expanded_metrics import paired_user_delta, summarize_ranks


def row(qid, user, rank, count=100, repeat=False, warm=False, cohort="c1"):
    return {"cohort": cohort, "query_id": qid, "user_id": user, "target_rank": rank,
            "candidate_count": count, "target_is_repeat": repeat, "target_known_before": warm}


class ExpandedMetricsTests(unittest.TestCase):
    def test_summary_metrics_overall_and_strata_keep_misses_in_denominators(self):
        rows = [row(0, 1, 1, repeat=True, warm=True), row(1, 1, 11, repeat=False, warm=True),
                row(2, 2, None, count=0, repeat=False, warm=False)]
        result = summarize_ranks(rows)
        overall = result["overall"]
        self.assertEqual(overall["query_count"], 3)
        self.assertEqual(overall["user_count"], 2)
        self.assertAlmostEqual(overall["candidate_recall"], 2 / 3)
        self.assertEqual(overall["repeat_count"], 1)
        self.assertEqual(overall["target_known_before_count"], 2)
        self.assertAlmostEqual(overall["query_micro"]["recall@10"], 1 / 3)
        self.assertAlmostEqual(overall["query_micro"]["ndcg@20"],
                               (1.0 + 1.0 / math.log2(12)) / 3)
        self.assertAlmostEqual(overall["user_macro"]["recall@10"], 0.25)
        self.assertEqual(result["strata"]["repeat"]["query_count"], 1)
        self.assertEqual(result["strata"]["new"]["query_count"], 2)
        self.assertEqual(result["strata"]["warm"]["query_count"], 2)
        self.assertEqual(result["strata"]["cold"]["query_count"], 1)
        self.assertEqual(result["counts"]["targets_missing_from_candidates"], 1)

    def test_empty_and_custom_cutoffs(self):
        self.assertIsNone(summarize_ranks([])["overall"]["candidate_recall"])
        result = summarize_ranks([row(0, 1, 3)], ks=(1, 3))
        self.assertEqual(result["overall"]["query_micro"]["recall@1"], 0.0)
        self.assertEqual(result["overall"]["query_micro"]["recall@3"], 1.0)

    def test_paired_delta_is_user_macro_and_bootstrap_reproducible(self):
        a = [row(0, 1, 10), row(1, 1, None), row(2, 2, 3), row(3, 3, 30)]
        b = [row(0, 1, 5), row(1, 1, 10), row(2, 2, 2), row(3, 3, 1)]
        first = paired_user_delta(a, b, seed=9, resamples=500)
        second = paired_user_delta(a, b, seed=9, resamples=500)
        self.assertEqual(first, second)
        self.assertEqual(first["paired_queries"], 4)
        self.assertEqual(first["paired_users"], 3)
        self.assertEqual(first["direction"], "mode_b_minus_mode_a")
        ndcg = first["bootstrap"]["ndcg20"]
        expected_user_deltas = [
            (1 / math.log2(6) + 1 / math.log2(11)) / 2 - (1 / math.log2(11)) / 2,
            1 / math.log2(3) - 1 / math.log2(4),
            1 / math.log2(2) - 0.0,
        ]
        self.assertAlmostEqual(ndcg["mean"], sum(expected_user_deltas) / 3)
        self.assertEqual(ndcg["users"], 3)
        self.assertTrue(ndcg["ci_lower"] <= ndcg["ci_upper"])
        self.assertIn("descriptive", ndcg["interpretation"])

    def test_pairing_rejects_different_keys_user_or_strata_and_catalog_size_by_default(self):
        a = [row(0, 1, 2)]
        with self.assertRaises(ValueError):
            paired_user_delta(a, [row(1, 1, 2)])
        with self.assertRaises(ValueError):
            paired_user_delta(a, [row(0, 2, 2)])
        with self.assertRaises(ValueError):
            paired_user_delta(a, [row(0, 1, 2, repeat=True)])
        with self.assertRaises(ValueError):
            paired_user_delta(a, [row(0, 1, 2, count=200)])

    def test_catalog_change_can_be_allowed_without_ignoring_shared_metadata(self):
        a = [row(0, 1, 12, count=100)]
        full = [row(0, 1, 33, count=500)]
        result = paired_user_delta(a, full, allow_catalog_change=True, resamples=50)
        self.assertFalse(result["candidate_count_equality_required"])
        with self.assertRaises(ValueError):
            paired_user_delta(a, [row(0, 1, 33, count=500, warm=True)], allow_catalog_change=True)

    def test_rank_schema_validation(self):
        with self.assertRaises(ValueError):
            summarize_ranks([row(0, 1, 4, count=3)])
        with self.assertRaises(ValueError):
            summarize_ranks([row(0, 1, 1), row(0, 2, 1)])
        with self.assertRaises(TypeError):
            summarize_ranks([row(0, 1, 1, repeat=1)])
        with self.assertRaises(ValueError):
            paired_user_delta([row(0, 1, 1)], [row(0, 1, 1)], resamples=0)


if __name__ == "__main__":
    unittest.main()
