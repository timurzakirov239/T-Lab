import copy
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from task_01_auto_features.src.retrieval_candidates import RetrievalCandidateGenerator
from task_01_auto_features.src.run_retrieval import (
    evaluate_candidate_result, load_development_events, load_frozen_queries, summarize, validate_config,
)


TASK = Path(__file__).resolve().parents[1]


class RetrievalRunnerTests(unittest.TestCase):
    def test_arrow_filters_reserved_events_from_mixed_row_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.parquet"
            pq.write_table(pa.table({"uid": [1, 1, 2, 2], "item_id": [10, 20, 30, 40],
                                     "timestamp": [1, 99, 3, 100],
                                     "event_type": ["listen", "test-sentinel", "like", "test-sentinel"]}),
                           path, row_group_size=4)
            loaded = load_development_events(path, 99)
            self.assertEqual(loaded["timestamp"].tolist(), [1, 3])
            self.assertEqual(loaded["item_id"].tolist(), [10, 30])
            self.assertNotIn("test-sentinel", loaded["event_names"])

    def test_selection_attribution_includes_popularity_fallback_and_keeps_misses(self):
        generator = RetrievalCandidateGenerator({}, {})
        for item in range(1, 201):
            generator.observe({"uid": 1, "item_id": item, "timestamp": item, "event_type": "listen"})
        result = generator.candidates(1, 201)
        self.assertTrue(result.fallback_items)
        query = {"query_id": 1, "user_id": 1, "target_item": result.fallback_items[0],
                 "target_is_repeat": True, "target_known_before": True}
        hit = evaluate_candidate_result(query, result)
        self.assertEqual(hit["target_selection_source"], "popular_fallback")
        self.assertEqual(sum(hit["selected_by_source"].values()), len(result.candidate_items))
        missing = evaluate_candidate_result({**query, "query_id": 2, "target_item": 999,
                                              "target_is_repeat": False, "target_known_before": False}, result)
        summary = summarize([hit, missing])
        self.assertEqual((summary["queries"], summary["hits"], summary["misses"]), (2, 1, 1))
        self.assertEqual(summary["candidate_recall_query_micro"], .5)
        self.assertEqual(summary["actual_selection"]["popular_fallback"]["target_hits"], 1)

    def test_user_macro_averages_users_before_queries_and_empty_strata(self):
        generator = RetrievalCandidateGenerator({}, {})
        generator.observe({"uid": 1, "item_id": 1, "timestamp": 1, "event_type": "listen"})
        result = generator.candidates(1, 2)
        def row(user, target, query_id):
            return evaluate_candidate_result({"query_id": query_id, "user_id": user, "target_item": target,
                                               "target_is_repeat": False, "target_known_before": target == 1}, result)
        summary = summarize([row(1, 1, 0), row(1, 1, 1), row(1, 1, 2), row(2, 9, 3)])
        self.assertEqual(summary["candidate_recall_query_micro"], .75)
        self.assertEqual(summary["candidate_recall_user_macro"], .5)
        self.assertEqual(summarize([])["candidate_recall_user_macro"], None)

    def test_config_cannot_enable_test_or_change_frozen_search(self):
        original = json.loads((TASK / "configs/retrieval_002.json").read_text(encoding="utf-8"))
        validate_config(original)
        for name, value in (("test_evaluation", True), ("model_training", True),
                            ("rrf_constant", 1), ("max_candidates", 101)):
            changed = copy.deepcopy(original)
            changed[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_config(changed)

    def test_query_loader_preserves_frozen_ids_and_rejects_reserved_query(self):
        query = {"query_id": 7, "user_id": 1, "timestamp": 11, "target_item": 1,
                 "split": "validation", "candidate_items": [1]}
        config = {"expected_queries": {"validation": 1},
                  "boundaries_raw": {"train_query_start": 1, "train_end": 10, "validation_end": 20}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text(json.dumps(query) + "\n", encoding="utf-8")
            self.assertEqual(load_frozen_queries(path, config), [query])
            path.write_text(json.dumps({**query, "timestamp": 20}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_frozen_queries(path, config)


if __name__ == "__main__":
    unittest.main()
