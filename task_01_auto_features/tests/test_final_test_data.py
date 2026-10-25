from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from task_01_auto_features.src import final_test_data
from task_01_auto_features.src.final_test_data import load_published_history, select_test_queries


class _FailingMonitor:
    def __init__(self):
        self.checks = 0

    def check(self):
        self.checks += 1
        raise MemoryError("synthetic monitor stop")


class FinalTestDataTests(unittest.TestCase):
    def test_loader_reads_only_four_columns_keeps_full_history_and_sorts_across_batches(self):
        table = pa.table({
            "uid": [2, 1, 1, 2], "item_id": [9, 5, 4, 8],
            "timestamp": [50, 50, 50, 49], "event_type": ["zzz", "listen", "like", "listen"],
            "future_payload": ["DO_NOT_DECODE"] * 4,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.parquet"
            pq.write_table(table, path, row_group_size=2)
            real_parquet_file = pq.ParquetFile
            requested_columns = []

            class _SpyParquetFile:
                def __init__(self, source):
                    self._source = real_parquet_file(source)
                    self.schema_arrow = self._source.schema_arrow

                def iter_batches(self, **kwargs):
                    requested_columns.append(kwargs["columns"])
                    return self._source.iter_batches(**kwargs)

            with patch.object(final_test_data.pq, "ParquetFile", _SpyParquetFile), \
                    patch.object(final_test_data, "_BATCH_SIZE", 2):
                events = load_published_history(path)

        self.assertTrue(requested_columns)
        self.assertTrue(all(columns == ["uid", "item_id", "timestamp", "event_type"] for columns in requested_columns))
        self.assertEqual(events["event_names"], ["like", "listen", "zzz"])
        self.assertEqual(events["timestamp"].tolist(), [49, 50, 50, 50])
        self.assertEqual(events["uid"].tolist(), [2, 1, 1, 2])
        self.assertEqual(events["item_id"].tolist(), [8, 4, 5, 9])
        self.assertEqual(events["event_code"].dtype, np.dtype(np.uint8))
        self.assertTrue(all(events[name].dtype == np.dtype(np.uint32) for name in ("uid", "item_id", "timestamp")))
        self.assertNotIn("DO_NOT_DECODE", events["event_names"])

    def test_loader_calls_monitor_for_each_batch_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.parquet"
            pq.write_table(pa.table({"uid": [1, 2, 3], "item_id": [10, 20, 30],
                                     "timestamp": [1, 2, 3], "event_type": ["listen"] * 3}), path)
            monitor = _FailingMonitor()
            with patch.object(final_test_data, "_BATCH_SIZE", 1), self.assertRaisesRegex(MemoryError, "synthetic"):
                load_published_history(path, monitor=monitor)
        self.assertEqual(monitor.checks, 1)

    @staticmethod
    def _events():
        # uid 1 has two eligible groups and one ambiguous group with 2 listens.
        # uid 2 has one eligible group; uid 3 has only pre-test history.
        rows = [
            (1, 11, 5, "listen"), (1, 12, 10, "listen"), (1, 13, 10, "listen"),
            (2, 20, 10, "listen"), (1, 99, 10, "like"), (1, 14, 20, "listen"),
            (1, 15, 15, "listen"), (3, 30, 4, "listen"),
        ]
        rows.sort(key=lambda row: (row[2], row[0], row[1]))
        names = sorted({row[3] for row in rows})
        codes = {name: index for index, name in enumerate(names)}
        return {"uid": np.asarray([row[0] for row in rows], dtype=np.uint32),
                "item_id": np.asarray([row[1] for row in rows], dtype=np.uint32),
                "timestamp": np.asarray([row[2] for row in rows], dtype=np.uint32),
                "event_code": np.asarray([codes[row[3]] for row in rows], dtype=np.uint8),
                "event_names": names}

    def test_query_selection_is_seeded_per_uid_includes_all_eligible_users_and_preserves_history(self):
        events = self._events()
        before = {key: value.copy() for key, value in events.items() if isinstance(value, np.ndarray)}
        first, info = select_test_queries(events, seed=17, start=10, limit=1)
        second, second_info = select_test_queries(events, seed=17, start=10, limit=1)
        self.assertEqual(first, second)
        self.assertEqual(info, second_info)
        self.assertEqual({query["user_id"] for query in first}, {1, 2})
        self.assertEqual([query["query_id"] for query in first], list(range(len(first))))
        self.assertEqual(first, sorted(first, key=lambda q: (q["timestamp"], q["user_id"], q["target_item"])))
        self.assertTrue(all(q["split"] == "validation" and q["evaluation_period"] == "test" for q in first))
        self.assertTrue(all("candidate_items" not in query for query in first))
        self.assertEqual(info["eligible_queries_before_sampling"], 3)
        self.assertEqual(info["sampled_queries"], 2)
        self.assertEqual(info["eligible_users"], 2)
        self.assertEqual(info["sampled_users"], 2)
        self.assertEqual(info["users_without_eligible_queries"], 1)
        self.assertEqual(info["ambiguous_uid_timestamp_listen_groups_excluded"], 1)
        for key, value in before.items():
            np.testing.assert_array_equal(events[key], value)

    def test_query_limit_caps_each_user_without_filtering_users_or_candidate_misses(self):
        events = self._events()
        queries, info = select_test_queries(events, seed=3, start=10, limit=20)
        self.assertEqual(len(queries), 3)
        self.assertEqual(info["sampled_users"], 2)
        self.assertEqual(sum(query["user_id"] == 1 for query in queries), 2)
        self.assertEqual(sum(query["user_id"] == 2 for query in queries), 1)
        # Target values are kept as labels; selection has no candidate list with which to filter misses.
        self.assertEqual({query["target_item"] for query in queries}, {14, 15, 20})

    def test_query_selection_validation(self):
        with self.assertRaises(ValueError):
            select_test_queries(self._events(), seed=1, limit=-1)
        with self.assertRaises(ValueError):
            select_test_queries(self._events(), seed=1, start=-1)
        with self.assertRaises(TypeError):
            select_test_queries(self._events(), seed=True)


if __name__ == "__main__":
    unittest.main()
