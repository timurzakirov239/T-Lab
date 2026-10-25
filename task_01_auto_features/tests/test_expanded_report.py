from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import report_expanded_search


def row(cohort, query_id, user_id, mode, rank, *, repeat=False, warm=False, candidate_count=100):
    return {"cohort": cohort, "query_id": query_id, "user_id": user_id,
            "timestamp": 100 + query_id, "mode": mode, "model": "M_UIDC",
            "target_rank": rank, "candidate_count": candidate_count,
            "target_is_repeat": repeat, "target_known_before": warm}


def bridge_rows():
    # query_id=0 is deliberately reused across cohorts. Pairing must include cohort.
    return [
        row("cohort_a", 0, 10, "K100", None, repeat=True, warm=True),
        row("cohort_a", 0, 10, "Full", 10, repeat=True, warm=True, candidate_count=1000),
        row("cohort_b", 0, 20, "K100", None, repeat=False, warm=False),
        row("cohort_b", 0, 20, "Full", 25, repeat=False, warm=False, candidate_count=900),
        row("cohort_a", 1, 11, "K100", 5, repeat=True, warm=True),
        row("cohort_a", 1, 11, "Full", 6, repeat=True, warm=True, candidate_count=1000),
        row("cohort_b", 1, 21, "K100", 10, repeat=False, warm=True),
        row("cohort_b", 1, 21, "Full", None, repeat=False, warm=True, candidate_count=900),
    ]


class CatalogBridgeTests(unittest.TestCase):
    def _bridge(self, rows):
        with patch.object(report_expanded_search, "LABELS", ["M_UIDC"]):
            return report_expanded_search.catalog_bridge(rows)["M_UIDC"]

    def test_pairs_same_query_id_separately_within_each_cohort(self):
        result = self._bridge(bridge_rows())
        self.assertEqual(result["paired_full_minus_k100"]["paired_queries"], 4)
        self.assertEqual(result["paired_full_minus_k100"]["paired_users"], 4)
        # Two id=0 pairs remain separate: one recovered at rank 10, one at rank 25.
        overall = result["target_transitions"]["overall"]
        self.assertEqual(overall["queries"], 4)
        self.assertEqual(overall["targets_recovered"], 2)
        self.assertEqual(overall["recovered_targets_in_top20"], 1)

    def test_recovery_top20_loss_and_retention_counts(self):
        groups = self._bridge(bridge_rows())["target_transitions"]
        overall = groups["overall"]
        self.assertEqual(overall["recovered_targets_in_top20"], 1)
        self.assertEqual(overall["top20_hits_retained"], 1)
        self.assertEqual(overall["top20_hits_lost"], 1)
        self.assertEqual(overall["top20_hits_gained"], 1)
        self.assertEqual(overall["targets_absent_from_full"], 1)

    def test_repeat_new_stratum_counts(self):
        groups = self._bridge(bridge_rows())["target_transitions"]
        self.assertEqual(groups["repeat"]["queries"], 2)
        self.assertEqual(groups["repeat"]["targets_recovered"], 1)
        self.assertEqual(groups["repeat"]["recovered_targets_in_top20"], 1)
        self.assertEqual(groups["repeat"]["top20_hits_retained"], 1)
        self.assertEqual(groups["repeat"]["top20_hits_lost"], 0)
        self.assertEqual(groups["new"]["queries"], 2)
        self.assertEqual(groups["new"]["targets_recovered"], 1)
        self.assertEqual(groups["new"]["recovered_targets_in_top20"], 0)
        self.assertEqual(groups["new"]["top20_hits_lost"], 1)
        self.assertEqual(groups["new"]["targets_absent_from_full"], 1)

    def test_transition_counts_are_independent_of_input_order(self):
        rows = bridge_rows()
        forward = self._bridge(rows)["target_transitions"]
        reverse = self._bridge(list(reversed(rows)))["target_transitions"]
        self.assertEqual(forward, reverse)

    def test_pairing_rejects_missing_query_and_metadata_mismatch(self):
        rows = bridge_rows()
        mismatch = copy.deepcopy(rows)
        mismatch.remove(next(r for r in mismatch if r["cohort"] == "cohort_b" and r["mode"] == "Full" and r["query_id"] == 1))
        with self.assertRaises(ValueError):
            self._bridge(mismatch)

        mismatch = bridge_rows()
        next(r for r in mismatch if r["cohort"] == "cohort_b" and r["mode"] == "Full" and r["query_id"] == 1)["user_id"] = 999
        with self.assertRaises(ValueError):
            self._bridge(mismatch)

        mismatch = bridge_rows()
        next(r for r in mismatch if r["cohort"] == "cohort_a" and r["mode"] == "Full" and r["query_id"] == 0)["target_is_repeat"] = False
        with self.assertRaises(ValueError):
            self._bridge(mismatch)


if __name__ == "__main__":
    unittest.main()
