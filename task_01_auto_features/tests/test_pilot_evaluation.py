import math
import unittest

from task_01_auto_features.src.pilot_evaluation import evaluate_ranking


def query(query_id, user_id, target, candidates, *, split="validation", repeat=False, warm=True):
    return {
        "query_id": query_id,
        "user_id": user_id,
        "timestamp": query_id,
        "target_item": target,
        "candidate_items": candidates,
        "split": split,
        "target_is_repeat": repeat,
        "target_known_before": warm,
    }


class PilotEvaluationTests(unittest.TestCase):
    def test_missing_target_keeps_denominator_and_warm_cold_counts(self):
        queries = [
            query(1, 10, 7, [1, 2], warm=True),
            query(2, 11, 8, [8, 9], warm=False, repeat=True),
        ]
        result = evaluate_ranking(queries, [[0.9, 0.1], [0.5, 0.4]], ks=(1,))
        self.assertEqual(result["counts"]["queries"], 2)
        self.assertEqual(result["counts"]["targets_missing_from_candidates"], 1)
        self.assertEqual(result["overall"]["candidate_recall"], 0.5)
        self.assertEqual(result["overall"]["query_micro"]["recall@1"], 0.5)
        self.assertEqual(result["strata"]["warm"]["query_count"], 1)
        self.assertEqual(result["strata"]["cold"]["query_count"], 1)
        self.assertEqual(result["strata"]["repeat"]["query_count"], 1)

    def test_tie_break_is_item_id_not_candidate_input_order(self):
        first = query(1, 1, 2, [9, 2, 1])
        second = query(1, 1, 2, [1, 9, 2])
        a = evaluate_ranking([first], [[0.5, 0.5, 0.5]], ks=(1, 2))
        b = evaluate_ranking([second], [[0.5, 0.5, 0.5]], ks=(1, 2))
        self.assertEqual(a, b)
        self.assertEqual(a["overall"]["query_micro"]["recall@1"], 0.0)
        self.assertEqual(a["overall"]["query_micro"]["recall@2"], 1.0)

    def test_user_macro_is_user_first_not_query_micro(self):
        queries = [
            query(1, 1, 1, [1]),
            query(2, 1, 2, [3]),
            query(3, 2, 4, [4]),
        ]
        result = evaluate_ranking(queries, [[1.0], [1.0], [1.0]], ks=(1,))
        self.assertAlmostEqual(result["overall"]["query_micro"]["recall@1"], 2 / 3)
        self.assertAlmostEqual(result["overall"]["user_macro"]["recall@1"], 0.75)

    def test_ndcg_single_target_analytical_rank_three(self):
        q = query(1, 1, 3, [1, 2, 3])
        result = evaluate_ranking([q], [[0.9, 0.8, 0.7]], ks=(2, 3))
        self.assertEqual(result["overall"]["query_micro"]["ndcg@2"], 0.0)
        self.assertAlmostEqual(result["overall"]["query_micro"]["ndcg@3"], 1 / math.log2(4))

    def test_k_larger_than_candidate_count_is_supported(self):
        q = query(1, 1, 2, [1, 2])
        result = evaluate_ranking([q], [[0.5, 0.4]], ks=(10, 20))
        self.assertEqual(result["overall"]["query_micro"]["recall@20"], 1.0)
        self.assertAlmostEqual(result["overall"]["query_micro"]["ndcg@10"], 1 / math.log2(3))

    def test_duplicate_candidate_ids_rejected(self):
        q = query(1, 1, 2, [1, 1, 2])
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            evaluate_ranking([q], [[0.1, 0.2, 0.3]])

    def test_nan_scores_rejected(self):
        q = query(1, 1, 2, [1, 2])
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_ranking([q], [[0.1, float("nan")]])

    def test_score_length_mismatch_rejected(self):
        q = query(1, 1, 2, [1, 2])
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            evaluate_ranking([q], [[0.1]])


if __name__ == "__main__":
    unittest.main()
