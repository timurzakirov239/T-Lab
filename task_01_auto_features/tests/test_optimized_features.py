import random
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from task_01_auto_features.src.optimized_features import OptimizedFeatureEngine
from task_01_auto_features.src.pilot_features import FeatureEngine, _Stats


class OptimizedFeatureEngineTests(unittest.TestCase):
    @staticmethod
    def create_pair(artist_map=None, album_map=None, short_window=5, half_life=7):
        artist_map = artist_map or {}
        album_map = album_map or {}
        args = (artist_map, album_map)
        kwargs = {"short_window": short_window, "half_life": half_life}
        return FeatureEngine(*args, **kwargs), OptimizedFeatureEngine(*args, **kwargs)

    @staticmethod
    def observe_pair(baseline, optimized, event):
        baseline.observe(event)
        optimized.observe(event)

    def assert_identical(self, baseline, optimized, user_id, item_ids, timestamp):
        self.assertEqual(optimized.feature_names, baseline.feature_names)
        self.assertEqual(optimized.feature_groups, baseline.feature_groups)
        expected = baseline.transform(user_id, item_ids, timestamp)
        actual = optimized.transform(user_id, item_ids, timestamp)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_randomized_multi_event_prefixes_match_bitwise(self):
        rng = random.Random(731)
        artist_map = {
            1: (10, 10, 11),       # duplicate edge is normalized by base init
            2: (10,),
            4: tuple(range(20, 35)),
            6: (12, 13),
            8: (12,),
            11: (50, 51, 52),
        }
        album_map = {
            1: (100,),
            2: (100, 100, 101),
            4: tuple(range(200, 225)),
            5: (102,),
            8: (101, 102),
            12: (300,),
        }
        baseline, optimized = self.create_pair(artist_map, album_map, short_window=6, half_life=9)

        event_types = ("listen", "like", "dislike", "unlike", "undislike")
        events = []
        for _ in range(240):
            events.append({
                "uid": rng.randint(1, 5),
                "item_id": rng.randint(1, 14),
                "timestamp": rng.randint(0, 40),
                "event_type": rng.choice(event_types),
            })
        # Preserve stable arbitrary order within a timestamp; the engines must treat
        # equal-time observations identically and queries only see full earlier groups.
        events = [event for _, event in sorted(enumerate(events), key=lambda pair: (pair[1]["timestamp"], pair[0]))]
        query_times = (0, 5, 13, 22, 31, 45)
        candidate_items = list(range(1, 16))
        cursor = 0
        for query_time in query_times:
            while cursor < len(events) and events[cursor]["timestamp"] < query_time:
                self.observe_pair(baseline, optimized, events[cursor])
                cursor += 1
            for uid in range(1, 6):
                self.assert_identical(baseline, optimized, uid, candidate_items, query_time)
                # Query-local caches must not make repeat calls stateful.
                self.assert_identical(baseline, optimized, uid, candidate_items, query_time)
        self.assertEqual(baseline._last_observed_timestamp, optimized._last_observed_timestamp)

    def test_counts_keep_open_left_window_boundary_and_feature_order(self):
        baseline, optimized = self.create_pair(
            artist_map={1: (7, 8, 8), 2: (7,)}, album_map={1: (20, 21), 2: (20,)},
            short_window=5, half_life=10,
        )
        for event in (
            {"uid": 3, "item_id": 1, "timestamp": 5, "event_type": "like"},
            {"uid": 3, "item_id": 1, "timestamp": 6, "event_type": "listen"},
            {"uid": 3, "item_id": 2, "timestamp": 6, "event_type": "unlike"},
        ):
            self.observe_pair(baseline, optimized, event)
        self.assert_identical(baseline, optimized, 3, [1, 2, 1, 999], 10)
        result = optimized.transform(3, [1], 10)
        self.assertEqual(result[0, optimized.feature_names.index("i_like_short")], 0.0)
        self.assertEqual(result[0, optimized.feature_names.index("i_listen_short")], 1.0)
        # Feature names remain grouped and ordered exactly as the frozen engine.
        self.assertEqual(optimized.feature_names[0], "u_events_all")
        self.assertEqual(optimized.feature_names[10], "i_events_all")
        self.assertEqual(optimized.feature_names[22], "d_events_all")
        self.assertEqual(optimized.feature_names[32], "c_artist_events_all_max")

    def test_empty_history_empty_candidates_and_missing_mappings_match(self):
        baseline, optimized = self.create_pair(artist_map={4: (90, 91)}, album_map={4: (500,)})
        self.assert_identical(baseline, optimized, 11, [4, 999], 100)
        self.assert_identical(baseline, optimized, 11, [], 100)
        empty = optimized.transform(11, [], 100)
        self.assertEqual(empty.shape, (0, len(optimized.feature_names)))
        self.assertEqual(empty.dtype, np.dtype(np.float32))

    def test_causal_query_validation_and_later_observation_are_preserved(self):
        baseline, optimized = self.create_pair(artist_map={1: (4, 5)}, album_map={1: (9,)})
        event = {"uid": 2, "item_id": 1, "timestamp": 10, "event_type": "listen"}
        self.observe_pair(baseline, optimized, event)
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            baseline.transform(2, [1], 10)
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            optimized.transform(2, [1], 10)
        self.assert_identical(baseline, optimized, 2, [1, 2], 11)
        target_time_event = {"uid": 2, "item_id": 2, "timestamp": 11, "event_type": "like"}
        self.observe_pair(baseline, optimized, target_time_event)
        self.assert_identical(baseline, optimized, 2, [1, 2], 12)


    def test_query_reuse_limits_user_and_shared_entity_count_lookups(self):
        baseline, optimized = self.create_pair(
            artist_map={1: (7,), 2: (7,)}, album_map={1: (9,), 2: (9,)},
        )
        for event in (
            {"uid": 1, "item_id": 1, "timestamp": 1, "event_type": "listen"},
            {"uid": 1, "item_id": 2, "timestamp": 2, "event_type": "like"},
        ):
            self.observe_pair(baseline, optimized, event)

        original_count = _Stats.count
        counted_stats = []

        def count_spy(stats, signal, timestamp, short_window):
            counted_stats.append(stats)
            return original_count(stats, signal, timestamp, short_window)

        with patch.object(_Stats, "count", count_spy):
            expected = baseline.transform(1, [1, 2], 3)
            counted_stats.clear()
            actual = optimized.transform(1, [1, 2], 3)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.tobytes(), expected.tobytes())

        calls_by_stats = Counter(id(stats) for stats in counted_stats)
        # One query-level U payload and one per distinct shared artist/album.
        self.assertEqual(calls_by_stats[id(optimized._users[1])], len(("events", "listen", "like", "dislike")))
        self.assertEqual(calls_by_stats[id(optimized._user_artists[(1, 7)])], 4)
        self.assertEqual(calls_by_stats[id(optimized._user_albums[(1, 9)])], 4)
        self.assertEqual(len(counted_stats), 28)

    def test_decay_knownshare_keeps_strict_positive_underflow_behavior(self):
        baseline, optimized = self.create_pair(artist_map={1: (4,)}, half_life=1)
        self.observe_pair(baseline, optimized, {"uid": 1, "item_id": 1, "timestamp": 0, "event_type": "listen"})
        self.assert_identical(baseline, optimized, 1, [1], 2000)
        row = optimized.transform(1, [1], 2000)
        index = optimized.feature_names.index("x_c_artist_knownshare_half_life")
        self.assertEqual(row[0, index], 0.0)


if __name__ == "__main__":
    unittest.main()
