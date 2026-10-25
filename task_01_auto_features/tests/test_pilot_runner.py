import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from task_01_auto_features.src.run_pilot import (
    CandidateGenerator,
    replay_features,
    select_queries,
    train_and_score,
)


class NoOpMonitor:
    def check(self):
        pass


class PilotRunnerTests(unittest.TestCase):
    def test_query_selection_respects_global_splits_and_excludes_uid_time_ties(self):
        timestamps = list(range(20)) + [8]
        events = {
            "uid": np.asarray([1] * len(timestamps), dtype=np.uint32),
            "item_id": np.arange(100, 100 + len(timestamps), dtype=np.uint32),
            "timestamp": np.asarray(timestamps, dtype=np.uint32),
            "event_code": np.zeros(len(timestamps), dtype=np.uint8),
            "event_names": ["listen"],
        }
        # Keep this synthetic feed in the stable event ordering expected by the runner.
        order = np.lexsort((events["item_id"], events["uid"], events["timestamp"]))
        for key in ("uid", "item_id", "timestamp", "event_code"):
            events[key] = events[key][order]
        selected, bounds, counts = select_queries(
            events,
            {"seed": 41, "train_queries_per_user": 100, "validation_queries_per_user": 100},
        )
        self.assertGreaterEqual(counts["ambiguous_uid_timestamp_listen_groups_excluded"].get("train", 0), 1)
        self.assertGreaterEqual(counts["ambiguous_uid_timestamp_listen_groups_excluded"].get("validation", 0), 0)
        self.assertFalse(any(q["timestamp"] == 8 for q in selected))
        self.assertTrue(all(bounds["train_query_start"] <= q["timestamp"] < bounds["validation_end"] for q in selected))
        self.assertTrue(all(q["timestamp"] < bounds["train_end"] for q in selected if q["split"] == "train"))
        self.assertTrue(all(bounds["train_end"] <= q["timestamp"] < bounds["validation_end"] for q in selected if q["split"] == "validation"))
        self.assertNotIn("test", {q["split"] for q in selected})

    def test_candidate_generator_uses_only_observed_catalog_and_item_id_ties(self):
        generator = CandidateGenerator(
            {},
            {},
            {"max_candidates": 4, "candidate_quotas": {"popular": 2, "repeat": 1, "artist": 1, "album": 1}},
        )
        generator.observe({"uid": 1, "item_id": 9, "timestamp": 1, "event_type": "listen"})
        generator.observe({"uid": 1, "item_id": 2, "timestamp": 1, "event_type": "listen"})
        candidates = generator.candidates(1, 2)
        self.assertEqual(candidates, [2, 9])
        self.assertNotIn(77, candidates)  # Hypothetical future target is never an input.
        with self.assertRaisesRegex(ValueError, "strictly before"):
            generator.candidates(1, 1)

    def test_replay_excludes_target_timestamp_from_history_and_candidates(self):
        events = {
            "uid": np.asarray([1, 1], dtype=np.uint32),
            "item_id": np.asarray([10, 20], dtype=np.uint32),
            "timestamp": np.asarray([1, 2], dtype=np.uint32),
            "event_code": np.asarray([0, 0], dtype=np.uint8),
            "event_names": ["listen"],
        }
        queries = [{"query_id": 0, "user_id": 1, "timestamp": 2, "target_item": 20, "split": "validation"}]
        config = {
            "max_candidates": 10,
            "candidate_quotas": {"popular": 3, "repeat": 2, "artist": 2, "album": 2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            matrix, observed, offsets, sizes, details = replay_features(
                events, queries, {}, {}, config, 1, 1, NoOpMonitor(), Path(tmp)
            )
            self.assertEqual(observed[0]["candidate_items"], [10])
            self.assertNotIn(20, observed[0]["candidate_items"])
            self.assertFalse(observed[0]["target_known_before"])
            self.assertEqual(details["events_observed"], 1)
            self.assertEqual((int(offsets[0]), int(sizes[0])), (0, 1))
            self.assertTrue(np.isfinite(matrix).all())

    def test_train_uses_only_retrieved_positive_queries_and_validation_keeps_misses(self):
        class FakeClassifier:
            fitted_y = None

            def __init__(self, **kwargs):
                pass

            def fit(self, x, y):
                FakeClassifier.fitted_y = np.asarray(y).copy()

            def predict_proba(self, x):
                return np.tile(np.asarray([[0.5, 0.5]]), (len(x), 1))

            def save_model(self, path):
                Path(path).write_text("synthetic", encoding="utf-8")

        fake_catboost = types.SimpleNamespace(CatBoostClassifier=FakeClassifier)
        previous = sys.modules.get("catboost")
        sys.modules["catboost"] = fake_catboost
        try:
            queries = [
                {"query_id": 0, "user_id": 1, "timestamp": 1, "target_item": 1, "candidate_items": [1, 2], "split": "train", "target_is_repeat": False, "target_known_before": True},
                {"query_id": 1, "user_id": 1, "timestamp": 2, "target_item": 9, "candidate_items": [2, 3], "split": "train", "target_is_repeat": False, "target_known_before": False},
                {"query_id": 2, "user_id": 2, "timestamp": 3, "target_item": 8, "candidate_items": [4, 5], "split": "validation", "target_is_repeat": False, "target_known_before": False},
                {"query_id": 3, "user_id": 2, "timestamp": 4, "target_item": 6, "candidate_items": [6, 7], "split": "validation", "target_is_repeat": True, "target_known_before": True},
            ]
            matrix = np.arange(8, dtype=np.float32).reshape(8, 1)
            offsets = np.asarray([0, 2, 4, 6])
            sizes = np.asarray([2, 2, 2, 2])
            feature_info = {"feature_names": ["i_count"], "feature_groups": {"i_count": "I"}, "short_window_raw": 1}
            config = {"model": {}, "seed": 1, "metrics_k": [1, 20]}
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                (out / "models").mkdir()
                (out / "predictions").mkdir()
                (out / "results").mkdir()
                result = train_and_score("synthetic", {"I"}, matrix, queries, offsets, sizes, feature_info, config, out, NoOpMonitor())
                self.assertEqual(len(FakeClassifier.fitted_y), 2)
                self.assertEqual(int(FakeClassifier.fitted_y.sum()), 1)
                self.assertEqual(result["train_queries_missing_target"], 1)
                self.assertEqual(result["metrics"]["overall"]["query_count"], 2)
                self.assertEqual(result["metrics"]["overall"]["candidate_recall"], 0.5)
                saved = json.loads((out / "results" / "synthetic.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["metrics"]["overall"]["query_count"], 2)
        finally:
            if previous is None:
                sys.modules.pop("catboost", None)
            else:
                sys.modules["catboost"] = previous


if __name__ == "__main__":
    unittest.main()
