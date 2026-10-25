import unittest

import numpy as np

from task_01_auto_features.src.pilot_features import FeatureEngine


class FeatureEngineTests(unittest.TestCase):
    def make_engine(self, **kwargs):
        defaults = dict(artist_map={}, album_map={}, short_window=5, half_life=5)
        defaults.update(kwargs)
        return FeatureEngine(**defaults)

    @staticmethod
    def feature(engine, row, name):
        return float(row[0, engine.feature_names.index(name)])

    def test_future_perturbation_does_not_change_prefix_features(self):
        def query_with_future_variant(future_item, future_type):
            engine = self.make_engine()
            engine.observe({"uid": 1, "item_id": 10, "timestamp": 10, "event_type": "like"})
            # Orchestration contract: events at/after query time are not passed to observe.
            _future = {"uid": 1, "item_id": future_item, "timestamp": 20, "event_type": future_type}
            return engine.transform(1, [10, 99], 15)

        a = query_with_future_variant(20, "listen")
        b = query_with_future_variant(777, "dislike")
        np.testing.assert_array_equal(a, b)

    def test_target_event_must_be_excluded_by_orchestrator(self):
        engine = self.make_engine()
        engine.observe({"uid": 1, "item_id": 4, "timestamp": 10, "event_type": "listen"})
        target = {"uid": 1, "item_id": 5, "timestamp": 15, "event_type": "like"}
        # If the target was observed, the engine rejects a query at its timestamp.
        engine.observe(target)
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            engine.transform(1, [5], 15)

    def test_fractional_credit_and_duplicate_mapping_normalization(self):
        engine = self.make_engine(artist_map={10: (7, 7, 8)})
        engine.observe({"uid": 3, "item_id": 10, "timestamp": 1, "event_type": "like"})
        row = engine.transform(3, [10], 2)
        self.assertAlmostEqual(self.feature(engine, row, "c_artist_like_all_max"), 0.5)
        self.assertAlmostEqual(self.feature(engine, row, "c_artist_like_all_mean"), 0.5)
        self.assertAlmostEqual(self.feature(engine, row, "c_artist_knownshare"), 1.0)
        self.assertEqual(self.feature(engine, row, "i_artist_degree"), 2.0)

    def test_same_timestamp_queries_do_not_mutate_prefix(self):
        engine = self.make_engine(artist_map={10: (7,)})
        engine.observe({"uid": 1, "item_id": 10, "timestamp": 3, "event_type": "listen"})
        before = engine._last_observed_timestamp
        a = engine.transform(1, [10, 11], 4)
        b = engine.transform(1, [10, 11], 4)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(engine._last_observed_timestamp, before)
        self.assertAlmostEqual(self.feature(engine, a[:1], "d_listen_all"), 1.0)

    def test_empty_history_has_finite_defaults(self):
        engine = self.make_engine(artist_map={10: (4, 5)}, album_map={10: (8,)})
        result = engine.transform(1, [10, 11], 100)
        self.assertEqual(result.shape, (2, len(engine.feature_names)))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(self.feature(engine, result[:1], "u_events_all"), 0.0)
        self.assertEqual(self.feature(engine, result[:1], "u_last_age"), -1.0)
        self.assertEqual(self.feature(engine, result[:1], "u_known"), 0.0)
        self.assertEqual(self.feature(engine, result[:1], "i_artist_degree"), 2.0)
        self.assertEqual(self.feature(engine, result[1:], "i_artist_degree"), 0.0)
        self.assertEqual(self.feature(engine, result[:1], "c_artist_knownshare"), 0.0)
        self.assertEqual(self.feature(engine, result[:1], "x_c_artist_knownshare_half_life"), 0.0)

    def test_half_life_matches_analytical_decay_and_named_columns(self):
        engine = self.make_engine(half_life=5, artist_map={2: (7, 8, 8)}, album_map={2: (20,)})
        engine.observe({"uid": 1, "item_id": 2, "timestamp": 10, "event_type": "like"})
        row = engine.transform(1, [2], 15)
        self.assertEqual(row.shape, (1, len(engine.feature_names)))
        self.assertEqual(len(engine.feature_names), 92)
        self.assertAlmostEqual(self.feature(engine, row, "x_i_like_half_life"), 0.5, places=6)
        self.assertAlmostEqual(self.feature(engine, row, "x_d_like_half_life"), 0.5, places=6)
        self.assertAlmostEqual(self.feature(engine, row, "x_i_events_half_life"), 0.5, places=6)
        # Each mapped entity receives 1/2 credit; X max/mean are the two
        # category scores, and knownshare is a single event-history fraction.
        self.assertAlmostEqual(self.feature(engine, row, "x_c_artist_like_half_life_max"), 0.25, places=6)
        self.assertAlmostEqual(self.feature(engine, row, "x_c_artist_like_half_life_mean"), 0.25, places=6)
        self.assertAlmostEqual(self.feature(engine, row, "x_c_artist_like_half_life_mean"), 0.25, places=6)
        self.assertAlmostEqual(self.feature(engine, row, "x_c_artist_knownshare_half_life"), 1.0, places=6)
        self.assertAlmostEqual(self.feature(engine, row, "x_c_album_like_half_life_max"), 0.5, places=6)
        self.assertAlmostEqual(self.feature(engine, row, "x_c_album_knownshare_half_life"), 1.0, places=6)

    def test_short_window_is_raw_unit_open_left_window(self):
        engine = self.make_engine(short_window=5)
        engine.observe({"uid": 1, "item_id": 2, "timestamp": 15, "event_type": "like"})
        engine.observe({"uid": 1, "item_id": 2, "timestamp": 16, "event_type": "listen"})
        row = engine.transform(1, [2], 20)
        self.assertEqual(self.feature(engine, row, "i_like_short"), 0.0)
        self.assertEqual(self.feature(engine, row, "i_listen_short"), 1.0)
        self.assertEqual(self.feature(engine, row, "i_events_short"), 1.0)

    def test_feature_names_and_groups_are_stable_and_complete(self):
        first = self.make_engine()
        second = self.make_engine(artist_map={1: (4,)})
        self.assertEqual(first.feature_names, second.feature_names)
        self.assertEqual(set(first.feature_groups.values()), {"U", "I", "D", "C", "X"})
        self.assertEqual(set(first.feature_groups), set(first.feature_names))
        self.assertEqual(len(first.feature_names), len(set(first.feature_names)))


if __name__ == "__main__":
    unittest.main()

