import hashlib
import unittest

import numpy as np

from task_01_auto_features.src.sampled_negatives import build_query_samples, seed_for, uniform_draw


class SampledNegativeTests(unittest.TestCase):
    def test_seed_encoding_is_stable_and_stream_separated(self):
        encoded = b"20261103|p005_20261026|139|uniform99"
        expected = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)
        self.assertEqual(seed_for("p005_20261026", 139, "uniform99"), expected)
        self.assertEqual(seed_for("p005_20261026", 139, "uniform99"),
                         seed_for("p005_20261026", 139, "uniform99"))
        self.assertNotEqual(seed_for("p005_20261026", 139, "uniform99"),
                            seed_for("p005_20261026", 139, "hard_pool256"))

    def test_uniform_draw_excludes_target_and_exclusions_and_preserves_order(self):
        catalog = np.arange(20, dtype=np.int64)
        first = uniform_draw(catalog, target=3, count=7,
                             rng=np.random.default_rng(seed_for("p001", 4, "unit", 12)), exclude=[1, 2])
        second = uniform_draw(catalog, target=3, count=7,
                              rng=np.random.default_rng(seed_for("p001", 4, "unit", 12)), exclude=[1, 2])
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertFalse({1, 2, 3} & set(first))
        self.assertEqual(len(uniform_draw(catalog, 3, 0)), 0)
        with self.assertRaisesRegex(ValueError, "only"):
            uniform_draw([3, 4, 5], target=3, count=3, seed=1)
        with self.assertRaisesRegex(ValueError, "unique"):
            uniform_draw([1, 1, 2], target=3, count=1, seed=1)

    def test_build_blocks_are_sorted_unique_and_hard_pool_excludes_target_and_uniform50(self):
        catalog = list(range(400))
        target = 177
        candidates = list(range(100))
        score_calls = []

        def teacher(items):
            score_calls.append(items.copy())
            # Constant scores exercise item-ID ascending tie breaking.
            return np.zeros(len(items), dtype=np.float64)

        result = build_query_samples(catalog, candidates, target, teacher, "p001", 80,
                                     {"seed": 20261103, "uniform_negatives": 99,
                                      "hybrid_uniform": 50, "hybrid_hard": 49,
                                      "hard_pool_uniform": 256})
        self.assertEqual(result["uniform_items"], sorted(result["uniform_items"]))
        self.assertEqual(result["hybrid_items"], sorted(result["hybrid_items"]))
        self.assertEqual(len(result["uniform_items"]), 100)
        self.assertEqual(len(result["hybrid_items"]), 100)
        self.assertEqual(result["uniform_items"].count(target), 1)
        self.assertEqual(result["hybrid_items"].count(target), 1)
        self.assertEqual(len(result["uniform_draw_order"]), 99)
        self.assertEqual(result["hybrid_uniform_draw_order"], result["uniform_draw_order"][:50])
        self.assertEqual(score_calls, [result["hard_pool_items"]])
        self.assertEqual(result["hard_pool_items"], sorted(result["hard_pool_items"]))
        self.assertEqual(len(result["hard_pool_items"]), len(result["hard_pool_scores"]))
        self.assertEqual(result["hard_items"], sorted(result["hard_items"])[:49])
        self.assertEqual(result["hard_items"], result["hard_pool_items"][:49])
        self.assertFalse(set(result["hard_pool_items"]) & {target, *result["hybrid_uniform_draw_order"]})
        self.assertFalse(set(result["uniform_draw_order"]) & {target})
        self.assertFalse(set(result["hybrid_uniform_draw_order"]) & set(result["hard_items"]))

    def test_build_samples_rejects_insufficient_or_invalid_teacher_pools(self):
        config = {"seed": 5, "uniform_negatives": 99, "hybrid_uniform": 50,
                  "hybrid_hard": 49, "hard_pool_uniform": 256}
        with self.assertRaisesRegex(ValueError, "only"):
            build_query_samples(range(80), [1, 2], 3, lambda ids: np.zeros(len(ids)), "c", 1, config)
        with self.assertRaisesRegex(ValueError, "one score"):
            build_query_samples(range(300), list(range(100)), 50, lambda ids: [0.0], "c", 1, config)
        with self.assertRaisesRegex(ValueError, "finite"):
            build_query_samples(range(300), list(range(100)), 50,
                                lambda ids: np.full(len(ids), np.nan), "c", 1, config)


if __name__ == "__main__":
    unittest.main()
