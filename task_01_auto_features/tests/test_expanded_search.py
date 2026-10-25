import unittest

import numpy as np

from task_01_auto_features.src.expanded_search import (
    forward_group_proposals,
    importance_proposals,
    query_group_permutation,
    query_group_permutation_importance,
    random_proposals,
    stability_importance,
)


class ExpandedSearchTests(unittest.TestCase):
    def test_importance_masks_are_unique_ranked_and_tie_break_by_column(self):
        columns = list(range(12))
        scores = [0.0] * 12
        groups = ["U"] * 3 + ["I"] * 3 + ["D"] * 3 + ["C"] * 3
        masks = importance_proposals(columns, scores, groups, sizes=(2, 4, 8), balanced_size=4)
        self.assertEqual(masks["full"], columns)
        self.assertEqual(masks["top2"], [0, 1])
        self.assertEqual(masks["top4"], [0, 1, 2, 3])
        self.assertEqual(masks["balanced4"], [0, 3, 6, 9])
        self.assertEqual(len({tuple(mask) for mask in masks.values()}), len(masks))

    def test_importance_zero_budget_and_validation(self):
        self.assertEqual(random_proposals(range(10), n_proposals=0), {})
        with self.assertRaisesRegex(ValueError, "positive"):
            importance_proposals([0, 1], [0.0, 0.0], ["U", "I"], sizes=(0,))
        with self.assertRaisesRegex(ValueError, "finite"):
            importance_proposals([0], [float("nan")], ["U"])

    def test_random_search_is_reproducible_unique_and_respects_size_budget(self):
        columns = list(range(206))
        first = random_proposals(columns, n_proposals=6, seed=2027)
        second = random_proposals(columns, n_proposals=6, seed=2027)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual([len(mask) for mask in first.values()], [8, 16, 32, 64, 128, 8])
        self.assertEqual(len({tuple(mask) for mask in first.values()}), len(first))
        self.assertTrue(all(mask == sorted(mask) for mask in first.values()))
        tiny = random_proposals([0, 1], sizes=(1,), n_proposals=8, seed=1)
        self.assertLessEqual(len(tiny), 2)

    def test_forward_group_proposals_only_make_one_group_extensions(self):
        group_columns = {"U": [0, 1], "I": [2], "D": [3, 4], "C": [5]}
        initial = forward_group_proposals(group_columns)
        self.assertEqual(list(initial), ["add_U", "add_I", "add_D", "add_C"])
        self.assertEqual(initial["add_D"], [3, 4])
        next_step = forward_group_proposals(group_columns, selected_groups=("U", "I"))
        self.assertEqual(list(next_step), ["add_D", "add_C"])
        self.assertEqual(next_step["add_D"], [0, 1, 2, 3, 4])

    def test_query_group_permutation_is_seeded_and_preserves_blocks_and_rank(self):
        offsets = [0, 2, 4, 7, 9, 11, 14]
        first = query_group_permutation(offsets, seed=7)
        self.assertTrue(np.array_equal(first, query_group_permutation(offsets, seed=7)))
        self.assertEqual(sorted(first.tolist()), list(range(offsets[-1])))
        # A 3-row query is only exchanged with another whole 3-row block.
        self.assertIn(tuple(first[4:7]), {(4, 5, 6), (11, 12, 13)})
        # Every 2-row block is copied from one complete 2-row block, same rank order.
        source_starts = [0, 2, 4, 7, 9, 11]
        for destination_start in (0, 2, 7, 9):
            source_pair = first[destination_start:destination_start + 2]
            self.assertEqual(int(source_pair[1] - source_pair[0]), 1)
            self.assertIn(int(source_pair[0]), source_starts)

    def test_permutation_importance_changes_only_requested_feature_column(self):
        matrix = np.arange(12, dtype=np.float32).reshape(6, 2)
        offsets = [0, 2, 4, 6]
        changed = query_group_permutation_importance(matrix, offsets, 1, seed=3)
        np.testing.assert_array_equal(changed[:, 0], matrix[:, 0])
        self.assertEqual(changed.dtype, matrix.dtype)
        self.assertEqual(sorted(changed[:, 1].tolist()), sorted(matrix[:, 1].tolist()))
        with self.assertRaisesRegex(ValueError, "cover"):
            query_group_permutation_importance(matrix, [0, 2, 4], 1)

    def test_stability_importance_normalizes_two_fits_and_penalizes_dispersion(self):
        stable = stability_importance([8.0, 2.0, 0.0], [4.0, 1.0, 5.0], [0, 1, 2], dispersion_penalty=1.0)
        self.assertGreater(stable[0], stable[1])
        self.assertGreater(stable[1], stable[2])
        self.assertEqual(stability_importance([0, 0], [0, 0], [0, 1]), {0: 0.0, 1: 0.0})
        unpenalized = stability_importance([1, 0], [0, 1], [0, 1], dispersion_penalty=0)
        penalized = stability_importance([1, 0], [0, 1], [0, 1], dispersion_penalty=1)
        self.assertTrue(all(penalized[index] < unpenalized[index] for index in (0, 1)))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            stability_importance([1], [1], [0], dispersion_penalty=-1)


if __name__ == "__main__":
    unittest.main()
