from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import report_sampled_training as report


def synthetic_rows():
    rows = []
    for cohort_index, cohort in enumerate(report.COHORTS):
        user = 1000 + cohort_index
        query_id = 7  # Deliberately reused across cohorts; pairing must include cohort.
        for mode, catalog_size in (("K100", 100), ("Full", 1000 + cohort_index)):
            for label in report.LABELS:
                rank = 20
                if label == "Baseline_UIDC":
                    rank = 12 if mode == "Full" else 10
                elif label == "HybridHit_UIDC":
                    rank = 8 if mode == "Full" else 9
                rows.append({
                    "cohort": cohort, "mode": mode, "model": label,
                    "query_id": query_id, "user_id": user, "timestamp": 100 + cohort_index,
                    "candidate_count": catalog_size, "target_rank": rank,
                    "target_is_repeat": bool(cohort_index % 2),
                    "target_known_before": True,
                })
    return rows


class SampledReportTests(unittest.TestCase):
    def test_contrast_plan_has_expected_29_distinct_paired_comparisons(self):
        contrasts = report.contrast_specs()
        self.assertEqual(len(contrasts), 29)
        self.assertIn("Full:HybridHit_UIDC-minus-Baseline_UIDC", contrasts)
        self.assertIn("K100:HybridWarm_A1-minus-HybridHit_A1", contrasts)
        self.assertIn("Full-minus-K100:Popularity", contrasts)

    def test_analysis_contains_18_pooled_and_54_cohort_metric_groups(self):
        analysis = report.analyze_rows(synthetic_rows(), {
            "bootstrap": {"seed": 20261104, "resamples": 20}
        })
        self.assertEqual(sum(len(models) for models in analysis["metrics"].values()), 18)
        self.assertEqual(sum(len(models) for cohort in analysis["cohort_metrics"].values()
                             for models in cohort.values()), 54)
        self.assertEqual(len(analysis["paired_contrasts"]), 29)
        primary = analysis["paired_contrasts"]["Full:HybridHit_UIDC-minus-Baseline_UIDC"]["pooled"]
        self.assertEqual(primary["paired_queries"], 3)
        self.assertEqual(primary["paired_users"], 3)
        self.assertGreater(primary["bootstrap"]["ndcg20"]["mean"], 0)
        self.assertEqual(analysis["scope"]["queries"], 3)
        self.assertEqual(analysis["scope"]["users"], 3)


if __name__ == "__main__":
    unittest.main()
