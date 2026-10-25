import unittest

import numpy as np

from task_01_auto_features.src.auto_features import AutoFeatureEngine, generate_specs
from task_01_auto_features.src.expanded_vector_features import FullCatalogFeatures


def _fixture():
    events = [
        (1, 10, 1, "listen"), (2, 11, 2, "like"), (1, 11, 3, "dislike"),
        (1, 12, 3, "unlike"), (2, 10, 4, "undislike"), (1, 10, 8, "listen"),
        (2, 12, 9, "like"), (1, 13, 11, "listen"), (1, 11, 11, "like"),
        (2, 13, 12, "dislike"), (1, 14, 18, "undislike"), (1, 10, 20, "listen"),
    ]
    arrays = {key: np.asarray([row[index] for row in events])
              for key, index in (("uid", 0), ("item_id", 1), ("timestamp", 2), ("event_type", 3))}
    artist_map = {10: (100, 101, 100), 11: (100,), 12: (102,), 13: (), 14: (101, 103)}
    album_map = {10: (200,), 11: (200, 201), 12: (201, 202), 13: (203,), 14: ()}
    return events, arrays, artist_map, album_map


def _engine(artist_map, album_map):
    return AutoFeatureEngine(artist_map, album_map, short_window=6, half_life=6,
                             specs=generate_specs(windows=(2, 6, 12), baseline_window=6))


class ExpandedVectorFeatureTests(unittest.TestCase):
    def test_exact_float32_match_across_ties_windows_unknowns_and_many_to_many(self):
        rows, arrays, artist_map, album_map = _fixture()
        reference = _engine(artist_map, album_map)
        vector_engine = _engine(artist_map, album_map)
        vector = FullCatalogFeatures(vector_engine, arrays)
        queries = [(1, [10, 11, 12, 13, 14, 99], 4), (2, [10, 11, 13, 99], 8),
                   (1, [10, 11, 12, 14, 99], 11), (2, [10, 12, 13, 14], 18),
                   (1, [10, 11, 12, 13, 14, 99], 20), (2, [10, 13, 99], 30)]
        cursor = 0
        for uid, candidates, timestamp in queries:
            while cursor < len(rows) and rows[cursor][2] < timestamp:
                event_uid, item, time, event_type = rows[cursor]
                reference.observe({"uid": event_uid, "item_id": item, "timestamp": time, "event_type": event_type})
                vector_engine.observe({"uid": event_uid, "item_id": item, "timestamp": time, "event_type": event_type})
                cursor += 1
            expected = reference.transform(uid, candidates, timestamp)
            actual = vector.transform(uid, candidates, timestamp)
            self.assertEqual(expected.shape, (len(candidates), 206))
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, expected)

    def test_candidate_order_and_duplicates_are_preserved(self):
        _rows, arrays, artists, albums = _fixture()
        engine = _engine(artists, albums)
        vector = FullCatalogFeatures(engine, arrays)
        candidates = [12, 10, 12, 99, 11]
        actual = vector.transform(7, candidates, 1)
        self.assertEqual(actual.shape, (5, 206))
        np.testing.assert_array_equal(actual[0], actual[2])
        self.assertEqual(actual[3, engine.feature_names.index("i_artist_degree")], 0.0)
        self.assertEqual(actual[3, engine.feature_names.index("i_album_degree")], 0.0)

    def test_causal_observation_and_nondecreasing_query_guard(self):
        rows, arrays, artists, albums = _fixture()
        engine = _engine(artists, albums)
        vector = FullCatalogFeatures(engine, arrays)
        engine.observe({"uid": 1, "item_id": 10, "timestamp": 1, "event_type": "listen"})
        before = vector.transform(1, [10, 11], 2)
        self.assertEqual(before[0, engine.feature_names.index("i_listen_all")], 1.0)
        self.assertEqual(before[1, engine.feature_names.index("i_listen_all")], 0.0)
        same_time_other_user = vector.transform(2, [10, 11], 2)
        self.assertEqual(same_time_other_user.shape, (2, 206))
        same_user_chunk = vector.transform(1, [10], 2)
        np.testing.assert_array_equal(same_user_chunk, before[:1])
        reverse_engine = _engine(artists, albums)
        reverse = FullCatalogFeatures(reverse_engine, arrays)
        reverse.transform(1, [10], 5)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            reverse.transform(1, [10], 4)

    def test_equal_time_chunks_match_full_reference_transform(self):
        rows, arrays, artists, albums = _fixture()
        reference = _engine(artists, albums)
        vector_engine = _engine(artists, albums)
        vector = FullCatalogFeatures(vector_engine, arrays)
        for event_uid, item, time, event_type in rows:
            if time < 12:
                event = {"uid": event_uid, "item_id": item, "timestamp": time, "event_type": event_type}
                reference.observe(event)
                vector_engine.observe(event)
        candidates = [10, 11, 12, 13, 14, 99]
        expected_uid1 = reference.transform(1, candidates, 12)
        expected_uid2 = reference.transform(2, candidates[4:], 12)
        actual = np.concatenate([vector.transform(1, candidates[:2], 12),
                                 vector.transform(1, candidates[2:4], 12),
                                 vector.transform(2, candidates[4:], 12)], axis=0)
        expected = np.concatenate([expected_uid1[:4], expected_uid2], axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_event_schema_and_causal_engine_requirements(self):
        _rows, arrays, artists, albums = _fixture()
        engine = _engine(artists, albums)
        engine.observe({"uid": 1, "item_id": 10, "timestamp": 1, "event_type": "listen"})
        with self.assertRaisesRegex(ValueError, "event_type"):
            FullCatalogFeatures(engine, {key: value for key, value in arrays.items() if key != "event_type"})
        vector = FullCatalogFeatures(engine, arrays)
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            vector.transform(1, [10], 1)


if __name__ == "__main__":
    unittest.main()
