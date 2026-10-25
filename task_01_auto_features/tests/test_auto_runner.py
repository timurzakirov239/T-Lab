import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from task_01_auto_features.src.auto_features import generate_specs
from task_01_auto_features.src.run_auto_features import candidate_rows, union_columns, final_masks, validate_frozen_config, TASK


class AutoRunnerTests(unittest.TestCase):
    def test_frozen_configuration_rejects_budget_or_protocol_drift(self):
        validate_frozen_config(TASK / "configs/p004_auto.json", TASK / "configs/P004_PROTOCOL.md")
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / "config.json"
            original = json.loads((TASK / "configs/p004_auto.json").read_text())
            original["maximum_inner_fits_per_family"] = 99
            config.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                validate_frozen_config(config, TASK / "configs/P004_PROTOCOL.md")
            config.write_bytes((TASK / "configs/p004_auto.json").read_bytes())
            protocol = Path(d) / "protocol.md"
            protocol.write_text("changed")
            with self.assertRaises(ValueError):
                validate_frozen_config(config, protocol)

    def test_union_keeps_manual_and_deduplicates_only_inner_fit(self):
        matrix = np.array([[1, 1, 1, 3, 3], [2, 2, 2, 4, 4], [9, 8, 7, 6, 5]], dtype=np.float32)
        manual = [0, 1]
        selected, removed = union_columns(matrix, np.array([0, 1]), manual, [2, 3, 4])
        self.assertEqual(selected, [0, 1, 3])
        self.assertEqual([row["representative"] for row in removed], [0, 3])
        matrix[2] *= -99
        self.assertEqual(union_columns(matrix, np.array([0, 1]), manual, [2, 3, 4]), (selected, removed))

    def test_final_masks_include_nine_fixed_manuals_and_no_post_selection_search(self):
        specs = generate_specs()
        manual = list(range(66))
        union = [*manual, 150, 180]
        masks = final_masks(specs, [10, 24], [66, 150], union)
        self.assertEqual(len(masks), 16)
        self.assertEqual(masks["M_UIDC"], manual)
        self.assertEqual(len(masks["M_UID_Cartist"]), 49)
        self.assertEqual(len(masks["M_UID_Calbum"]), 49)
        self.assertEqual(masks["Auto_A1"], [10, 24])
        for group in "UIDC":
            self.assertEqual(masks[f"Union_without_{group}"], [i for i in union if specs[i]["group"] != group])

    def test_candidate_rows_cover_whole_selected_query_blocks_in_order(self):
        offsets = np.array([0, 2, 5])
        sizes = np.array([2, 3, 1])
        np.testing.assert_array_equal(candidate_rows([0, 2], offsets, sizes), [0, 1, 5])
        self.assertEqual(candidate_rows([], offsets, sizes).shape, (0,))


if __name__ == "__main__":
    unittest.main()
