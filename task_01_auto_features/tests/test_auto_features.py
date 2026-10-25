import unittest
import random

import numpy as np

from task_01_auto_features.src.auto_features import AutoFeatureEngine, generate_specs
from task_01_auto_features.src.pilot_features import FeatureEngine


class AutoFeatureEngineTests(unittest.TestCase):
    @staticmethod
    def index(engine, name):
        return engine.feature_names.index(name)

    @staticmethod
    def observe_both(engines, event):
        for engine in engines:
            engine.observe(dict(event))

    def test_a1_reproduces_all_66_baseline_uidc_columns_exactly(self):
        artist_map = {10: (7, 8, 8), 11: (7,), 12: (9,)}
        album_map = {10: (100,), 11: (100, 101), 12: (102,)}
        baseline = FeatureEngine(artist_map, album_map, short_window=5, half_life=7)
        specs = generate_specs(windows=(3, 5, 9), baseline_window=5)
        automatic = AutoFeatureEngine(artist_map, album_map, 5, 7, specs=specs[:66])
        events = (
            {"uid": 1, "item_id": 10, "timestamp": 1, "event_type": "listen"},
            {"uid": 1, "item_id": 11, "timestamp": 2, "event_type": "like"},
            {"uid": 2, "item_id": 10, "timestamp": 3, "event_type": "unlike"},
            {"uid": 1, "item_id": 12, "timestamp": 5, "event_type": "dislike"},
        )
        for event in events:
            self.observe_both((baseline, automatic), event)

        expected = baseline.transform(1, [10, 11, 12, 999], 7)
        actual = automatic.transform(1, [10, 11, 12, 999], 7)
        baseline_names = [name for name in baseline.feature_names if not name.startswith("x_")]
        self.assertEqual(len(automatic.feature_names), 66)
        self.assertEqual(automatic.feature_names, baseline_names)
        np.testing.assert_array_equal(actual, expected[:, [baseline.feature_names.index(n) for n in baseline_names]])
        self.assertEqual(actual.dtype, np.dtype(np.float32))

    def test_a1_randomized_multi_mapping_matches_with_241_degree(self):
        rng = random.Random(24017)
        artist_map = {10: tuple(range(1000, 1241))}
        album_map = {10: tuple(range(500, 545))}
        for item_id in range(11, 31):
            artist_ids = [rng.randrange(980, 1270) for _ in range(rng.randrange(1, 8))]
            album_ids = [rng.randrange(450, 570) for _ in range(rng.randrange(1, 6))]
            # Include duplicate links: both engines normalize to unique entities.
            artist_map[item_id] = tuple(artist_ids + artist_ids[:1])
            album_map[item_id] = tuple(album_ids + album_ids[:1])

        baseline = FeatureEngine(artist_map, album_map, short_window=13, half_life=17)
        specs = generate_specs(windows=(7, 13, 21), baseline_window=13)
        automatic = AutoFeatureEngine(artist_map, album_map, 13, 17, specs=specs[:66])
        events = [
            {
                "uid": rng.randrange(1, 5),
                "item_id": rng.randrange(10, 31),
                "timestamp": rng.randrange(0, 70),
                "event_type": rng.choice(("listen", "like", "dislike", "unlike", "undislike")),
            }
            for _ in range(160)
        ]
        events = [event for _, event in sorted(enumerate(events), key=lambda pair: (pair[1]["timestamp"], pair[0]))]
        for event in events:
            self.observe_both((baseline, automatic), event)

        candidates = [10, 11, 13, 19, 30, 999]
        expected = baseline.transform(2, candidates, 75)
        actual = automatic.transform(2, candidates, 75)
        baseline_names = [name for name in baseline.feature_names if not name.startswith("x_")]
        self.assertEqual(float(actual[0, self.index(automatic, "i_artist_degree")]), 241.0)
        np.testing.assert_array_equal(actual, expected[:, [baseline.feature_names.index(n) for n in baseline_names]])

    def test_invalid_specs_fail_early(self):
        invalid = generate_specs()[:1]
        invalid[0]["group"] = "C"  # path u is constrained to group U
        with self.assertRaisesRegex(ValueError, "path/group mismatch"):
            AutoFeatureEngine({}, {}, 670390, 670390, specs=invalid)

        invalid = generate_specs()[:1]
        invalid[0]["reducer"] = "max"  # a direct U expression cannot aggregate entity values
        with self.assertRaisesRegex(ValueError, "invalid 'count' expression"):
            AutoFeatureEngine({}, {}, 670390, 670390, specs=invalid)

    def test_default_grammar_is_deterministic_206_unique_columns(self):
        first = generate_specs()
        second = generate_specs()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 206)
        self.assertEqual(len({spec["feature_id"] for spec in first}), 206)
        self.assertEqual(sum(spec["family"] == "A1" for spec in first), 66)
        self.assertEqual(sum(spec["family"] == "A2" for spec in first), 140)
        self.assertEqual({spec["group"] for spec in first}, {"U", "I", "D", "C"})

    def test_fractional_mapping_credit_and_missing_relation_share_defaults(self):
        artist_map = {10: (7, 7, 8), 11: (9,)}
        specs = generate_specs()
        engine = AutoFeatureEngine(artist_map, {}, 670390, 670390, specs=specs)
        engine.observe({"uid": 3, "item_id": 10, "timestamp": 1, "event_type": "like"})
        rows = engine.transform(3, [10, 11, 12], 2)

        def value(row, feature_id):
            return float(rows[row, self.index(engine, feature_id)])

        self.assertAlmostEqual(value(0, "c_artist_like_all_max"), 0.5)
        self.assertAlmostEqual(value(0, "c_artist_like_all_mean"), 0.5)
        self.assertAlmostEqual(value(0, "c_artist_like_share_all_max"), 2.0 / 3.0)
        self.assertAlmostEqual(value(1, "c_artist_like_share_all_mean"), 0.5)
        self.assertEqual(value(2, "c_artist_like_share_all_max"), 0.0)
        self.assertEqual(value(0, "i_artist_degree"), 2.0)
        self.assertEqual(value(1, "i_artist_degree"), 1.0)
        self.assertEqual(value(2, "i_artist_degree"), 0.0)

    def test_share_smoothing_and_trend_window_boundaries_are_analytic(self):
        engine = AutoFeatureEngine({}, {}, 5, 5, specs=generate_specs(windows=(3, 5, 9), baseline_window=5))
        for timestamp, event_type in ((11, "listen"), (15, "listen"), (16, "listen"), (19, "unlike")):
            engine.observe({"uid": 1, "item_id": 10, "timestamp": timestamp, "event_type": event_type})
        row = engine.transform(1, [10], 20)

        def value(name):
            return float(row[0, self.index(engine, name)])

        self.assertAlmostEqual(value("i_listen_share_all"), 3.5 / 5.0)
        self.assertAlmostEqual(value("i_listen_share_w5"), (1.0 + 0.5) / (2.0 + 1.0))
        self.assertAlmostEqual(value("i_events_trend_w5"), 0.0)
        self.assertAlmostEqual(value("i_listen_trend_w5"), -1.0 / 3.0)

    def test_future_perturbations_are_absent_and_current_time_observation_is_rejected(self):
        specs = generate_specs()
        engines = [
            AutoFeatureEngine({}, {}, 670390, 670390, specs=specs),
            AutoFeatureEngine({}, {}, 670390, 670390, specs=specs),
        ]
        self.observe_both(engines, {"uid": 1, "item_id": 10, "timestamp": 1, "event_type": "listen"})
        before = [engine.transform(1, [10], 5) for engine in engines]
        np.testing.assert_array_equal(before[0], before[1])

        engines[0].observe({"uid": 1, "item_id": 20, "timestamp": 5, "event_type": "like"})
        engines[1].observe({"uid": 1, "item_id": 99, "timestamp": 5, "event_type": "dislike"})
        for engine in engines:
            with self.assertRaisesRegex(ValueError, "strictly earlier"):
                engine.transform(1, [10], 5)


if __name__ == "__main__":
    unittest.main()
