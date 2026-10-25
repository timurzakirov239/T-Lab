import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "src"))
from pilot_data import _prepare


class PilotDataTests(unittest.TestCase):
    def fixtures(self, root: Path, unsorted=False):
        events = root / "events.parquet"
        ts = [1, 2, 2, 3]
        if unsorted: ts = [1, 3, 2, 4]
        table = pa.table({
            "uid": pa.array([10, 10, 10, 10], type=pa.uint32()),
            "timestamp": pa.array(ts, type=pa.uint32()),
            "item_id": pa.array([100, 100, 200, 200], type=pa.uint32()),
            "is_organic": pa.array([1, 1, 0, 1], type=pa.uint8()),
            "played_ratio_pct": pa.array([60, 80, None, 100], type=pa.uint16()),
            "track_length_seconds": pa.array([180, 180, None, 240], type=pa.uint32()),
            "event_type": ["listen", "listen", "like", "listen"],
        })
        pq.write_table(table, events)
        artist = root / "artist.parquet"
        pq.write_table(pa.table({"item_id": pa.array([100, 100, 100, 999], type=pa.uint32()),
                                 "artist_id": pa.array([4, 4, 5, 6], type=pa.uint32())}), artist)
        album = root / "album.parquet"
        pq.write_table(pa.table({"item_id": pa.array([100, 200, 200, 999], type=pa.uint32()),
                                 "album_id": pa.array([7, 8, 8, 9], type=pa.uint32())}), album)
        return events, artist, album

    def test_keeps_complete_user_history_and_counts_cross_batch_tie(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events, artist, album = self.fixtures(root)
            out = root / "out"
            result = _prepare(out, 1, 20261025, events, artist, album, {}, batch_size=2)
            saved = pq.read_table(out / "events.parquet")
            self.assertEqual(saved.num_rows, 4)
            self.assertEqual(set(saved["uid"].to_pylist()), {10})
            self.assertEqual(result["rows"], 4)
            self.assertEqual(result["chronology"]["violations_in_selected_rows"], 0)
            self.assertEqual(result["chronology"]["same_uid_timestamp_extra_rows"], 1)
            self.assertEqual(result["chronology"]["same_uid_timestamp_groups"], 1)
            self.assertEqual(result["mappings"]["artist"]["unique_links"], 2)
            self.assertEqual(result["mappings"]["artist"]["duplicate_links_removed"], 1)
            self.assertEqual(result["mappings"]["artist"]["items_with_multiple_entities"], 1)
            self.assertTrue((out / "metadata.json").is_file())

    def test_reports_timestamp_order_violation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events, artist, album = self.fixtures(root, unsorted=True)
            result = _prepare(root / "out", 1, 1, events, artist, album, {}, batch_size=2)
            self.assertEqual(result["chronology"]["violations_in_selected_rows"], 1)

    def test_writes_valid_empty_mapping_when_no_items_are_covered(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events, artist, album = self.fixtures(root)
            for path, entity in ((artist, "artist_id"), (album, "album_id")):
                pq.write_table(pa.table({"item_id": pa.array([999], type=pa.uint32()),
                                         entity: pa.array([1], type=pa.uint32())}), path)
            out = root / "out"
            result = _prepare(out, 1, 1, events, artist, album, {}, batch_size=2)
            for name in ("artist", "album"):
                mapping_path = out / f"{name}_mapping.parquet"
                self.assertEqual(pq.ParquetFile(mapping_path).metadata.num_rows, 0)
                self.assertEqual(result["mappings"][name]["unique_links"], 0)
                self.assertEqual(result["mappings"][name]["selected_item_coverage_pct"], 0.0)
                self.assertEqual(pq.read_schema(mapping_path).names, ["item_id", "artist_id" if name == "artist" else "album_id"])


if __name__ == "__main__":
    unittest.main()
