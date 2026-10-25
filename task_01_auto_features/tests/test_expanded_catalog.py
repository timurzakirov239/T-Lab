from __future__ import annotations

import unittest

import numpy as np

from task_01_auto_features.src.auto_features import AutoFeatureEngine
from task_01_auto_features.src.expanded_catalog import evaluate_full_catalog, transform_catalog
from task_01_auto_features.src.pilot_features import FeatureEngine


class ExpandedCatalogTests(unittest.TestCase):
    @staticmethod
    def _engine(engine_type=FeatureEngine):
        engine = engine_type({}, {}, 10, 10) if engine_type is FeatureEngine else engine_type(
            {}, {}, 10, 10, specs=[
                {"feature_id": "item_events", "family": "A1", "group": "I", "path": "i",
                 "op": "count", "signal": "events", "window": "all", "reducer": None,
                 "provenance": "synthetic"},
                {"feature_id": "user_item_events", "family": "A1", "group": "D", "path": "d",
                 "op": "count", "signal": "events", "window": "all", "reducer": None,
                 "provenance": "synthetic"},
            ])
        for event in (
            {"uid": 1, "item_id": 8, "timestamp": 1, "event_type": "listen"},
            {"uid": 2, "item_id": 5, "timestamp": 2, "event_type": "like"},
            {"uid": 1, "item_id": 3, "timestamp": 3, "event_type": "dislike"},
            {"uid": 1, "item_id": 8, "timestamp": 5, "event_type": "listen"},
        ):
            engine.observe(event)
        return engine

    def test_full_catalog_scores_match_brute_force_and_bounds_chunks(self):
        engine = self._engine()
        chunks = []

        def scorer(matrix):
            chunks.append(matrix.shape[0])
            return matrix[:, 0] + matrix[:, 1] * 2.0

        result = evaluate_full_catalog(engine, 1, 6, 8, {"model": scorer}, chunk_size=2, top_k=20)
        catalog = sorted(engine._items)
        brute = engine.transform(1, catalog, 6)[:, 0] + engine.transform(1, catalog, 6)[:, 1] * 2.0
        order = sorted(range(len(catalog)), key=lambda i: (-float(brute[i]), catalog[i]))
        self.assertEqual(result["catalog_items"], catalog)
        self.assertEqual(result["catalog_size"], len(catalog))
        self.assertEqual(max(chunks), 2)
        self.assertEqual(result["models"]["model"]["target_score"], float(brute[catalog.index(8)]))
        self.assertEqual(result["models"]["model"]["target_rank"], order.index(catalog.index(8)) + 1)
        self.assertEqual([row["item_id"] for row in result["models"]["model"]["top_items"]],
                         [catalog[i] for i in order])

    def test_ties_use_item_id_ascending_and_missing_target_is_not_injected(self):
        engine = self._engine()
        result = evaluate_full_catalog(engine, 1, 6, 99, {"tie": lambda matrix: np.ones(len(matrix))},
                                       chunk_size=1, top_k=2)
        self.assertFalse(result["target_present"])
        self.assertIsNone(result["models"]["tie"]["target_score"])
        self.assertIsNone(result["models"]["tie"]["target_rank"])
        self.assertEqual(result["models"]["tie"]["top_items"], [
            {"item_id": 3, "score": 1.0}, {"item_id": 5, "score": 1.0}])
        self.assertNotIn(99, result["catalog_items"])

    def test_observed_equal_timestamp_or_future_events_are_rejected_by_engine(self):
        engine = self._engine()
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            evaluate_full_catalog(engine, 1, 5, 8, {"m": lambda matrix: np.zeros(len(matrix))})

    def test_multiple_scorers_share_feature_chunks_and_columns_can_differ(self):
        engine = self._engine(AutoFeatureEngine)
        calls = []
        original = engine.transform

        def tracked(user_id, item_ids, timestamp):
            calls.append(len(item_ids))
            return original(user_id, item_ids, timestamp)

        engine.transform = tracked
        result = evaluate_full_catalog(
            engine, 2, 6, 8,
            {"item": lambda matrix: matrix[:, 0], "pair": lambda matrix: matrix[:, 0]},
            columns={"item": [0], "pair": [1]}, chunk_size=2,
            transform_fn=lambda _engine, uid, item_ids, timestamp: tracked(uid, item_ids, timestamp))
        self.assertEqual(calls, [2, 1])
        self.assertEqual(result["catalog_size"], 3)
        self.assertNotEqual(result["models"]["item"]["target_score"], result["models"]["pair"]["target_score"])

    def test_memoized_auto_transform_is_float32_exact_and_reuses_path_key_work(self):
        engine = self._engine(AutoFeatureEngine)
        catalog = sorted(engine._items)
        expected = engine.transform(1, catalog, 6)
        original = engine._one_value
        calls = []

        def tracked(*args, **kwargs):
            calls.append((args[0]["path"], args[2], args[3]))
            return original(*args, **kwargs)

        engine._one_value = tracked
        actual = transform_catalog(engine, 1, catalog, 6)
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(calls), len(set(calls)))

    def test_bad_chunk_score_shape_and_nonfinite_values_fail(self):
        engine = self._engine()
        with self.assertRaises(ValueError):
            evaluate_full_catalog(engine, 1, 6, 8, {"bad": lambda matrix: np.zeros((len(matrix), 2))})
        with self.assertRaises(ValueError):
            evaluate_full_catalog(engine, 1, 6, 8, {"bad": lambda matrix: np.full(len(matrix), np.nan)})
        with self.assertRaises(ValueError):
            evaluate_full_catalog(engine, 1, 6, 8, {"ok": lambda matrix: np.zeros(len(matrix))}, chunk_size=0)


if __name__ == "__main__":
    unittest.main()
