import json
import random
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from task_01_auto_features.src.robustness_data import prepare_cohort, prepare_memberships


def make_sources(root: Path):
    events = root / "events.parquet"
    rows = []
    # Unequal user activity; membership sampling must remain uniform over these six UIDs.
    for uid, n in ((10, 20), (20, 1), (30, 2), (40, 4), (50, 3), (60, 8)):
        for i in range(n):
            rows.append((uid, i + 1, uid * 100 + i, "listen"))
    # A post-cutoff payload sentinel for UID 10 must stay in full storage only.
    rows.extend([(10, 100, 99101, "POST_CUTOFF_SENTINEL"), (10, 100, 99102, "POST_CUTOFF_SENTINEL")])
    table = pa.table({
        "uid": pa.array([r[0] for r in rows], type=pa.uint32()),
        "timestamp": pa.array([r[1] for r in rows], type=pa.uint32()),
        "item_id": pa.array([r[2] for r in rows], type=pa.uint32()),
        "is_organic": pa.array([1] * len(rows), type=pa.uint8()),
        "played_ratio_pct": pa.array([65000 if r[1] >= 100 else 50 for r in rows], type=pa.uint16()),
        "track_length_seconds": pa.array([180] * len(rows), type=pa.uint32()),
        "event_type": [r[3] for r in rows],
    })
    pq.write_table(table, events, row_group_size=3)
    artist = root / "artist.parquet"
    pq.write_table(pa.table({"item_id": pa.array([uid * 100 for uid in (10, 20, 30, 40, 50, 60)], type=pa.uint32()),
                             "artist_id": pa.array([1, 1, 2, 2, 3, 3], type=pa.uint32())}), artist)
    album = root / "album.parquet"
    pq.write_table(pa.table({"item_id": pa.array([uid * 100 for uid in (10, 20, 30, 40, 50, 60)], type=pa.uint32()),
                             "album_id": pa.array([11, 11, 12, 12, 13, 13], type=pa.uint32())}), album)
    return events, artist, album


class RobustnessDataTests(unittest.TestCase):
    def test_membership_sampling_is_seeded_uniform_and_sequentially_disjoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events, _, _ = make_sources(root)
            output = root / "memberships"
            result = prepare_memberships(output, seeds=(17, 18), users=2, exclude_users=[10],
                                         event_source=events, batch_size=4)
            population = [10, 20, 30, 40, 50, 60]
            first_pool = [x for x in population if x != 10]
            expected_first = sorted(random.Random(17).sample(first_pool, 2))
            second_pool = [x for x in first_pool if x not in expected_first]
            expected_second = sorted(random.Random(18).sample(second_pool, 2))
            self.assertEqual(result["cohorts"][0]["selected_users"], expected_first)
            self.assertEqual(result["cohorts"][1]["selected_users"], expected_second)
            self.assertFalse(set(expected_first) & set(expected_second))
            self.assertTrue(result["disjoint_cohorts"])
            self.assertEqual(result["source_unique_uid_count"], 6)
            self.assertEqual(result["excluded_users"], [10])
            self.assertTrue((output / "memberships.json").is_file())
            with self.assertRaises(FileExistsError):
                prepare_memberships(output, seeds=(17,), users=1, event_source=events)

    def test_full_cohort_history_retained_but_cutoff_payload_excluded_from_dev(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events, artist, album = make_sources(root)
            output = root / "cohort"
            result = prepare_cohort(output, [10], seed=1, cutoff=50, event_source=events,
                                    artist_source=artist, album_source=album, batch_size=3)
            full = pq.read_table(output / "events_full.parquet")
            dev = pq.read_table(output / "events.parquet")
            self.assertEqual(full.num_rows, 22)
            self.assertTrue(any(x == "POST_CUTOFF_SENTINEL" for x in full["event_type"].to_pylist()))
            self.assertEqual(dev.num_rows, 20)
            self.assertTrue(all(ts < 50 for ts in dev["timestamp"].to_pylist()))
            self.assertNotIn("POST_CUTOFF_SENTINEL", result["development"]["event_type_counts"])
            self.assertEqual(result["full_history_retention"]["stored_full_rows_arrow_count"], 22)
            self.assertEqual(result["development"]["chronology"]["equal_uid_timestamp_extra_rows"], 0)

    def test_development_timestamp_ties_across_batches_are_counted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events, artist, album = make_sources(root)
            # Use a compact dedicated fixture with an equal-timestamp group spanning batches.
            rows = [(1, 1, 101), (1, 2, 102), (1, 2, 103), (1, 3, 104), (1, 9, 105)]
            tie_source = root / "ties.parquet"
            pq.write_table(pa.table({"uid": pa.array([r[0] for r in rows], type=pa.uint32()),
                                    "timestamp": pa.array([r[1] for r in rows], type=pa.uint32()),
                                    "item_id": pa.array([r[2] for r in rows], type=pa.uint32()),
                                    "is_organic": pa.array([1] * len(rows), type=pa.uint8()),
                                    "played_ratio_pct": pa.array([50] * len(rows), type=pa.uint16()),
                                    "track_length_seconds": pa.array([180] * len(rows), type=pa.uint32()),
                                    "event_type": ["listen"] * len(rows)}), tie_source, row_group_size=2)
            artist_tie = root / "artist_tie.parquet"
            album_tie = root / "album_tie.parquet"
            for path, key in ((artist_tie, "artist_id"), (album_tie, "album_id")):
                pq.write_table(pa.table({"item_id": pa.array([101, 102, 103, 104], type=pa.uint32()),
                                         key: pa.array([1, 1, 2, 2], type=pa.uint32())}), path)
            result = prepare_cohort(root / "tie_out", [1], 1, 9, tie_source, artist_tie, album_tie, batch_size=2)
            stats = result["development"]["chronology"]
            self.assertEqual(result["development"]["rows"], 4)
            self.assertEqual(stats["violations"], 0)
            self.assertEqual(stats["equal_uid_timestamp_extra_rows"], 1)
            self.assertEqual(stats["equal_uid_timestamp_groups"], 1)
            self.assertTrue(stats["checked_across_batch_boundaries"])

    def test_monitor_failure_closes_writers_and_failed_output_cannot_be_reused(self):
        class Monitor:
            def __init__(self): self.calls = 0
            def check(self):
                self.calls += 1
                if self.calls == 5: raise RuntimeError("synthetic stop")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events, artist, album = make_sources(root)
            output = root / "failed_cohort"
            with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
                prepare_cohort(output, [10], 1, 50, events, artist, album, monitor=Monitor(), batch_size=3)
            self.assertEqual(json.loads((output / "FAILED.json").read_text())["status"], "failed")
            self.assertTrue(pq.ParquetFile(output / "events_full.parquet").metadata.num_rows > 0)
            self.assertTrue(pq.ParquetFile(output / "events.parquet").metadata.num_rows > 0)
            with self.assertRaises(FileExistsError):
                prepare_cohort(output, [10], 1, 50, events, artist, album)


if __name__ == "__main__":
    unittest.main()
