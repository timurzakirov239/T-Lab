import unittest
from unittest.mock import patch

import numpy as np

from task_01_auto_features.src import auto_search
from task_01_auto_features.src.auto_search import (
    eligible_training_rows,
    offsets_sizes,
    pick_best,
    prune_columns,
    select_trial_columns,
    temporal_split,
)


class AutoSearchTests(unittest.TestCase):
    def test_temporal_split_uses_train_only_and_never_splits_equal_timestamps(self):
        train_times = [1, 2, 2, 2, 4, 5, 6, 7, 8, 9]
        queries = [{"split": "train", "timestamp": t} for t in train_times]
        queries += [{"split": "validation", "timestamp": 10, "target_item": 3}]
        # A reserved query intentionally has no timestamp/label: split must not inspect it.
        queries += [{"split": "test"}]
        parts = temporal_split(queries)
        expected_cutoff = int(np.quantile(np.asarray(train_times), .70, method="lower"))
        self.assertEqual(parts["cutoff"], expected_cutoff)
        fit = parts["inner_fit"].tolist()
        selection = parts["inner_selection"].tolist()
        self.assertTrue(fit and selection)
        self.assertEqual(parts["external_validation"].tolist(), [10])
        fit_times = [queries[i]["timestamp"] for i in fit]
        selection_times = [queries[i]["timestamp"] for i in selection]
        self.assertLess(max(fit_times), expected_cutoff)
        self.assertGreaterEqual(min(selection_times), expected_cutoff)
        self.assertEqual(len(set(fit_times) & set(selection_times)), 0)

    def test_eligible_rows_keep_complete_candidate_blocks_and_report_misses(self):
        queries = [
            {"split": "train", "candidate_items": [11, 12, 13], "target_item": 12},
            {"split": "train", "candidate_items": [21, 22], "target_item": 99},
            {"split": "train", "candidate_items": [31, 32, 33], "target_item": 33},
        ]
        offsets, sizes = offsets_sizes(queries)
        self.assertEqual(offsets.tolist(), [0, 3, 5])
        self.assertEqual(sizes.tolist(), [3, 2, 3])
        rows, y, stats = eligible_training_rows(queries, [0, 1], offsets, sizes)
        self.assertEqual(rows.tolist(), [0, 1, 2])
        self.assertEqual(y.tolist(), [0, 1, 0])
        self.assertEqual(stats["queries_requested"], 2)
        self.assertEqual(stats["queries_with_target"], 1)
        self.assertEqual(stats["queries_missing_target"], 1)
        self.assertEqual(stats["candidate_rows_examined"], 5)
        self.assertEqual(stats["candidate_rows_used"], 3)
        self.assertEqual(stats["positive_rows"], 1)
        with self.assertRaisesRegex(ValueError, "complete query-major"):
            eligible_training_rows(queries, [0], [0, 2, 5], [3, 2, 3])
        queries[2]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "train-split"):
            eligible_training_rows(queries, [2], offsets, sizes)

    def test_pruning_uses_inner_fit_rows_only_and_preserves_original_representative(self):
        matrix = np.asarray([
            [1, 1, 7, 1],
            [2, 2, 7, 2],
            [3, 3, 7, 3],
            [5, 999, 8, 5],  # outer/selection rows cannot affect pruning
        ], dtype=np.float32)
        kept, removed = prune_columns(matrix, [0, 1, 2], [3, 2, 1, 0])
        self.assertEqual(kept, [0])
        self.assertEqual(removed, [
            {"column": 1, "reason": "duplicate", "representative": 0},
            {"column": 2, "reason": "constant", "representative": None},
            {"column": 3, "reason": "duplicate", "representative": 0},
        ])
        changed = matrix.copy()
        changed[3] = [-10, 55, -300, 99]
        self.assertEqual(prune_columns(changed, [0, 1, 2], [3, 2, 1, 0]), (kept, removed))

    def test_pruning_verifies_hash_collisions_with_exact_array_comparison(self):
        matrix = np.asarray([[1, 1, 5], [2, 2, 4], [3, 3, 3]], dtype=np.float64)
        with patch.object(auto_search, "_column_digest", return_value=b"same-hash"):
            kept, removed = prune_columns(matrix, [0, 1, 2], [0, 1, 2])
        self.assertEqual(kept, [0, 2])
        self.assertEqual(removed, [{"column": 1, "reason": "duplicate", "representative": 0}])

    def test_trial_proposals_are_bounded_unique_ranked_balanced_and_canonical(self):
        specs = []
        for col in range(70):
            group = "U" if col < 40 else ("I", "D", "C")[(col - 40) % 3]
            specs.append({"feature_id": f"f{col:03d}", "group": group})
        importances = {col: float(100 - col) for col in range(70)}
        importances[1] = importances[0]  # tie must resolve by spec_id
        proposals = select_trial_columns(list(reversed(range(70))), specs, importances)
        self.assertEqual(list(proposals), ["full", "top8", "top16", "top32", "top64", "balanced32"])
        self.assertEqual(len(proposals), 6)
        self.assertEqual(proposals["top8"], list(range(8)))
        self.assertEqual(proposals["full"], list(range(70)))
        for columns in proposals.values():
            self.assertEqual(columns, sorted(columns))
            self.assertLessEqual(len(columns), 70)
        self.assertEqual(len(proposals["balanced32"]), 32)
        self.assertTrue({0, 40, 41, 42}.issubset(proposals["balanced32"]))
        self.assertNotEqual(proposals["balanced32"], proposals["top32"])

    def test_pick_best_uses_inner_score_then_size_then_spec_ids_and_rejects_outer_fields(self):
        trials = [
            {"label": "bigger", "inner_score": .8, "columns": [1, 2, 3], "spec_ids": ["b"]},
            {"label": "z", "inner_score": .8, "columns": [1], "spec_ids": ["z"]},
            {"label": "a", "inner_score": .8, "columns": [2], "spec_ids": ["a"]},
            {"label": "higher", "inner_score": .9, "columns": list(range(20)), "spec_ids": ["zz"]},
        ]
        self.assertEqual(pick_best(trials)["label"], "higher")
        tie = pick_best(trials[:3])
        self.assertEqual(tie["label"], "a")
        with self.assertRaisesRegex(ValueError, "disallowed fields"):
            pick_best([{"inner_score": .9, "columns": [0], "validation_score": .99}])


if __name__ == "__main__":
    unittest.main()
