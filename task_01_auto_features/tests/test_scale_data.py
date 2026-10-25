import random
import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TASK = Path(__file__).resolve().parents[1]
from task_01_auto_features.src.scale_data import _prepare


def write_fixture(root: Path, rows, artist_links=None, album_links=None):
    event_path = root / "source_events.parquet"
    table = pa.table({
        "uid": pa.array([r[0] for r in rows], type=pa.uint32()),
        "timestamp": pa.array([r[1] for r in rows], type=pa.uint32()),
        "item_id": pa.array([r[2] for r in rows], type=pa.uint32()),
        "is_organic": pa.array([1] * len(rows), type=pa.uint8()),
        "played_ratio_pct": pa.array([60 if r[1] < 9 else 65000 for r in rows], type=pa.uint16()),
        "track_length_seconds": pa.array([180] * len(rows), type=pa.uint32()),
        "event_type": ["listen" if r[1] < 9 else "POST_CUTOFF_SENTINEL" for r in rows],
    })
    pq.write_table(table, event_path)
    artist_path = root / "artist.parquet"
    apairs = artist_links if artist_links is not None else [(r[2], 5) for r in rows if r[1] < 9]
    pq.write_table(pa.table({"item_id": pa.array([x[0] for x in apairs], type=pa.uint32()),
                             "artist_id": pa.array([x[1] for x in apairs], type=pa.uint32())}), artist_path)
    album_path = root / "album.parquet"
    bpairs = album_links if album_links is not None else [(r[2], 8) for r in rows if r[1] < 9]
    pq.write_table(pa.table({"item_id": pa.array([x[0] for x in bpairs], type=pa.uint32()),
                             "album_id": pa.array([x[1] for x in bpairs], type=pa.uint32())}), album_path)
    return event_path, artist_path, album_path


class ScaleDataTests(unittest.TestCase):
    def prepare(self, root, rows, users, seed, cutoff=9, batch_size=2, artist_links=None, album_links=None, monitor=None):
        event, artist, album = write_fixture(root, rows, artist_links, album_links)
        return _prepare(root / "out", users, seed, cutoff, event, artist, album, {}, batch_size, monitor)

    def test_uniform_uid_selection_keeps_all_history_and_hides_postcutoff_from_dev_stats(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [(uid, t, uid * 10 + t) for uid in (10, 20, 30, 40) for t in (1, 11)]
            result = self.prepare(root, rows, 2, 77)
            expected = sorted(random.Random(77).sample([10, 20, 30, 40], 2))
            self.assertEqual(result["selected_users"], expected)
            full = pq.read_table(root / "out" / "events_full.parquet")
            dev = pq.read_table(root / "out" / "events.parquet")
            self.assertEqual(full.num_rows, 4)
            self.assertEqual(set(full["uid"].to_pylist()), set(expected))
            self.assertTrue(any(x == "POST_CUTOFF_SENTINEL" for x in full["event_type"].to_pylist()))
            self.assertEqual(dev.num_rows, 2)
            self.assertTrue(all(t < 9 for t in dev["timestamp"].to_pylist()))
            self.assertNotIn("POST_CUTOFF_SENTINEL", result["development"]["event_type_counts"])
            self.assertFalse(result["full_history_retention"]["post_cutoff_payload_interpreted"])

    def test_development_ties_and_order_are_checked_across_batch_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [(1, t, i + 1) for i, t in enumerate((1, 2, 2, 3, 9))]
            result = self.prepare(root, rows, 1, 1, batch_size=2)
            chronology = result["development"]["chronology"]
            self.assertEqual(result["development"]["rows"], 4)
            self.assertEqual(chronology["violations"], 0)
            self.assertEqual(chronology["equal_uid_timestamp_extra_rows"], 1)
            self.assertEqual(chronology["equal_uid_timestamp_groups"], 1)
            self.assertTrue(chronology["checked_across_batch_boundaries"])

    def test_empty_mapping_outputs_keep_schema_and_zero_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [(1, 1, 101), (1, 10, 102)]
            result = self.prepare(root, rows, 1, 1, cutoff=9,
                                  artist_links=[(999, 2)], album_links=[(999, 3)])
            for name, entity in (("artist", "artist_id"), ("album", "album_id")):
                path = root / "out" / f"{name}_mapping.parquet"
                self.assertEqual(pq.ParquetFile(path).metadata.num_rows, 0)
                self.assertEqual(pq.read_schema(path).names, ["item_id", entity])
                self.assertEqual(result["mappings_snapshot"][name]["selected_item_coverage_pct"], 0.0)

    def test_monitor_checked_at_batches_and_stats_chunks(self):
        class Monitor:
            def __init__(self): self.calls = 0
            def check(self): self.calls += 1
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            monitor = Monitor()
            rows = [(1, i, i) for i in range(1, 41)]
            self.prepare(root, rows, 1, 1, cutoff=41, batch_size=20, monitor=monitor)
            self.assertGreaterEqual(monitor.calls, 5)  # UID batch, event batches, and bounded stats chunks.

    def test_monitor_failure_closes_partial_writers_and_marks_run_failed(self):
        class Monitor:
            def __init__(self): self.calls = 0
            def check(self):
                self.calls += 1
                if self.calls == 5: raise RuntimeError("synthetic resource limit")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [(1, t, t) for t in (1, 2, 3, 4, 5)]
            event, artist, album = write_fixture(root, rows)
            output = root / "out"
            with self.assertRaisesRegex(RuntimeError, "synthetic resource limit"):
                _prepare(output, 1, 1, 9, event, artist, album, {}, 2, Monitor())
            self.assertTrue((output / "FAILED.json").is_file())
            self.assertEqual(json.loads((output / "FAILED.json").read_text())["status"], "failed")
            self.assertEqual(pq.ParquetFile(output / "events_full.parquet").metadata.num_rows, 2)
            self.assertEqual(pq.ParquetFile(output / "events.parquet").metadata.num_rows, 2)
            with self.assertRaises(FileExistsError):
                _prepare(output, 1, 1, 9, event, artist, album, {}, 2)

    def test_uid_pass_monitor_failure_also_marks_run_failed(self):
        class Monitor:
            def check(self): raise RuntimeError("stop during uid pass")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            event, artist, album = write_fixture(root, [(1, 1, 1), (1, 2, 2)])
            output = root / "out"
            with self.assertRaisesRegex(RuntimeError, "stop during uid pass"):
                _prepare(output, 1, 1, 9, event, artist, album, {}, 2, Monitor())
            self.assertEqual(json.loads((output / "FAILED.json").read_text())["status"], "failed")
            self.assertEqual(list(output.iterdir()), [output / "FAILED.json"])
            with self.assertRaises(FileExistsError):
                _prepare(output, 1, 1, 9, event, artist, album, {}, 2)


if __name__ == "__main__": unittest.main()
