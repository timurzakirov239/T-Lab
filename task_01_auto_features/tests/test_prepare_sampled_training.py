import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from task_01_auto_features.src import prepare_sampled_training as prep
from task_01_auto_features.src.auto_features import AutoFeatureEngine, generate_specs


class _Monitor:
    def __init__(self, limits):
        self.limits = limits

    def start(self):
        pass

    def check(self):
        pass

    def finish(self):
        return {"peak_process_rss_mb": 1.0, "minimum_available_system_mb": 1000.0,
                "limits": self.limits, "limit_exceeded": None}


class _Teacher:
    def predict_proba(self, matrix, thread_count=None):
        score = np.asarray(matrix[:, 0], dtype=np.float64)
        return np.column_stack((1.0 - score, score))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class PrepareSampledTrainingTests(unittest.TestCase):
    def _fixture(self, root: Path):
        task = root / "task_01_auto_features"
        for folder in ("configs", "src", "artifacts/dev", "artifacts/p007/p001/models",
                       "data/mappings"):
            (task / folder).mkdir(parents=True, exist_ok=True)
        config_path = task / "configs/p008_sampled.json"
        protocol_path = task / "configs/P008_PROTOCOL.md"
        config_path.write_text("synthetic p008 config", encoding="utf-8")
        protocol_path.write_text("synthetic protocol", encoding="utf-8")
        for name in ("prepare_sampled_training.py", "sampled_negatives.py", "auto_features.py",
                     "pilot_features.py", "auto_search.py", "run_pilot.py", "run_retrieval.py"):
            (task / "src" / name).write_text(name, encoding="utf-8")

        events_list = [(900, item, 1, "listen") for item in range(181)] + [(900, 200, 10, "like")]
        events_list.sort(key=lambda row: (row[2], row[0], row[1]))
        event_names = ["like", "listen"]
        event_code = {name: position for position, name in enumerate(event_names)}
        events = {"uid": np.asarray([row[0] for row in events_list], dtype=np.uint32),
                  "item_id": np.asarray([row[1] for row in events_list], dtype=np.uint32),
                  "timestamp": np.asarray([row[2] for row in events_list], dtype=np.uint32),
                  "event_code": np.asarray([event_code[row[3]] for row in events_list], dtype=np.uint8),
                  "event_names": event_names}
        artist_map = {item: (item % 20, (item + 1) % 20) for item in range(202)}
        album_map = {item: (100 + item % 25,) for item in range(202)}

        candidates = list(range(100))
        queries = [
            {"query_id": 1, "user_id": 1, "timestamp": 10, "target_item": 5,
             "split": "train", "candidate_items": candidates,
             "target_is_repeat": False, "target_known_before": True},
            {"query_id": 2, "user_id": 2, "timestamp": 12, "target_item": 105,
             "split": "train", "candidate_items": candidates,
             "target_is_repeat": False, "target_known_before": True},
            {"query_id": 3, "user_id": 3, "timestamp": 15, "target_item": 300,
             "split": "train", "candidate_items": candidates,
             "target_is_repeat": False, "target_known_before": False},
            {"query_id": 4, "user_id": 1, "timestamp": 30, "target_item": 6,
             "split": "validation", "candidate_items": candidates,
             "target_is_repeat": False, "target_known_before": True},
        ]
        queries_path = task / "artifacts/dev/queries.jsonl"
        queries_path.write_text("\n".join(json.dumps(row) for row in queries) + "\n", encoding="utf-8")
        engine = AutoFeatureEngine(artist_map, album_map, 4, 4, specs=generate_specs()[:66])
        rows = []
        pointer = 0
        for query in queries:
            while pointer < len(events_list) and events_list[pointer][2] < query["timestamp"]:
                uid, item, timestamp, event_type = events_list[pointer]
                engine.observe({"uid": uid, "item_id": item, "timestamp": timestamp, "event_type": event_type})
                pointer += 1
            rows.append(engine.transform(query["user_id"], query["candidate_items"], query["timestamp"]))
        features_path = task / "artifacts/dev/features.npy"
        np.save(features_path, np.concatenate(rows, axis=0))
        offsets = list(range(0, len(queries) * 100, 100))
        inputs_path = task / "artifacts/dev/feature_inputs.jsonl"
        inputs_path.write_text("".join(json.dumps({"query_id": row["query_id"], "user_id": row["user_id"],
                                                     "timestamp": row["timestamp"], "offset": offsets[i],
                                                     "size": 100, "candidate_items": row["candidate_items"]}) + "\n"
                                       for i, row in enumerate(queries)), encoding="utf-8")
        _write_json(task / "artifacts/dev/feature_schema.json",
                    {"feature_names": engine.feature_names, "feature_groups": engine.feature_groups})

        history = task / "data/history.parquet"
        artist_file = task / "data/mappings/artist_mapping.parquet"
        album_file = task / "data/mappings/album_mapping.parquet"
        for path in (history, artist_file, album_file):
            path.write_bytes(b"synthetic")
        teacher_model = task / "artifacts/p007/p001/models/M_UIDC.cbm"
        teacher_model.write_bytes(b"teacher")
        teacher_sha = hashlib.sha256(teacher_model.read_bytes()).hexdigest()
        _write_json(task / "artifacts/p007/p001/run_status.json", {
            "status": "complete", "final_models": [{"label": "M_UIDC", "model_file": "models/M_UIDC.cbm",
                                                        "model_sha256": teacher_sha}]})
        _write_json(task / "artifacts/p007/p001/selection_frozen.json", {
            "final_masks": {"M_UIDC": list(range(66)), "Original_A1": [0, 2, 4, 6]}})
        p007_snapshot = task / "report/p007_snapshot"
        (p007_snapshot / "p001").mkdir(parents=True, exist_ok=True)
        for source, target in ((task / "artifacts/p007/p001/run_status.json", p007_snapshot / "p001/status_K100.json"),
                               (task / "artifacts/p007/p001/selection_frozen.json", p007_snapshot / "p001/selection_frozen.json")):
            target.write_bytes(source.read_bytes())
        snapshot_manifest = {f"p001/{name}": hashlib.sha256((p007_snapshot / "p001" / name).read_bytes()).hexdigest()
                             for name in ("status_K100.json", "selection_frozen.json")}
        _write_json(p007_snapshot / "checkpoint_manifest.json", snapshot_manifest)
        config = {"boundaries_raw": {"train_query_start": 0, "train_end": 20, "validation_end": 40},
                  "windows_raw": [2, 4, 8], "short_window_raw": 4, "half_life_raw": 4,
                  "max_candidates": 100, "resource_limits": {"runner_peak_rss_mb": 1100,
                                                                   "minimum_available_mb": 150},
                  "sampling": {"seed": 20261103, "uniform_negatives": 99,
                               "hybrid_uniform": 50, "hybrid_hard": 49, "hard_pool_uniform": 256},
                  "p007_root": "artifacts/p007"}
        cohort = {"id": "p001", "development_run": "artifacts/dev", "development_queries": "artifacts/dev/queries.jsonl",
                  "history": "data/history.parquet", "mapping_dir": "data/mappings"}
        p006_paths = ["artifacts/dev/queries.jsonl", "data/history.parquet", "artifacts/dev/features.npy",
                      "artifacts/dev/feature_inputs.jsonl", "artifacts/dev/feature_schema.json",
                      "data/mappings/artist_mapping.parquet", "data/mappings/album_mapping.parquet",
                      "src/auto_features.py", "src/pilot_features.py", "src/auto_search.py",
                      "src/run_pilot.py", "src/run_retrieval.py"]
        _write_json(task / "configs/p006_freeze_manifest.json", {
            "files": {relative: hashlib.sha256((task / relative).read_bytes()).hexdigest()
                      for relative in p006_paths}})
        return task, config, cohort, events, artist_map, album_map, teacher_model

    def test_prepare_builds_all_train_records_and_exact_hit_warm_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            bad_root = Path(temp) / "bad"
            task, config, cohort, events, artist_map, album_map, _teacher_model = self._fixture(bad_root)
            output = task / "artifacts/p008_sampled_training/p001/prepare"
            with patch.object(prep, "TASK", task), patch.object(prep, "CONFIG_PATH", task / "configs/p008_sampled.json"), \
                 patch.object(prep, "PROTOCOL_PATH", task / "configs/P008_PROTOCOL.md"), \
                 patch.object(prep, "P006_MANIFEST_PATH", task / "configs/p006_freeze_manifest.json"), \
                 patch.object(prep, "P007_SNAPSHOT", task / "report/p007_snapshot"), \
                 patch.object(prep, "P007_SNAPSHOT_MANIFEST", task / "report/p007_snapshot/checkpoint_manifest.json"), \
                 patch.object(prep, "ResourceMonitor", _Monitor), \
                 patch.object(prep, "load_development_events", return_value=events), \
                 patch.object(prep, "load_mapping", side_effect=lambda path, _name:
                              artist_map if "artist" in str(path) else album_map), \
                 patch.object(prep, "_load_teacher", return_value=_Teacher()):
                status = prep.prepare(config, cohort, output)
                self.assertEqual(status["status"], "complete")
                self.assertEqual(status["query_counts"], {"train": 3, "hit": 1, "warm": 2, "cold": 1})
                self.assertTrue(status["exact_gate"]["identical_float32_bytes"])
                self.assertEqual(status["exact_gate"]["queries"], 3)
                uniform = np.load(output / "uniform.npy", allow_pickle=False)
                hybrid = np.load(output / "hybrid.npy", allow_pickle=False)
                self.assertEqual(uniform.shape, (100, 66))
                self.assertEqual(hybrid.shape, (200, 66))
                records = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 3)
                self.assertTrue(records[0]["old_hit"] and records[0]["warm"])
                self.assertFalse(records[1]["old_hit"])
                self.assertTrue(records[1]["warm"])
                self.assertTrue(records[2]["cold"])
                self.assertIsNone(records[1]["uniform_row_start"])
                self.assertEqual(records[0]["hybrid_row_start"], 0)
                self.assertEqual(records[1]["hybrid_row_start"], 100)
                self.assertIsNone(records[2]["hybrid_row_start"])
                # Item 200 first appears exactly at q1's timestamp, so strict
                # timestamp<t keeps it out of q1's catalog and includes it q2.
                self.assertEqual(records[0]["catalog_size"], 181)
                self.assertEqual(records[1]["catalog_size"], 182)
                with np.load(output / "item_first_seen.npz", allow_pickle=False) as inventory:
                    position = np.searchsorted(inventory["item_ids"], 200)
                    self.assertEqual(int(inventory["item_ids"][position]), 200)
                    self.assertEqual(int(inventory["first_seen"][position]), 10)
                self.assertIn("uniform.npy", status["outputs_sha256"])
                self.assertIn("teacher_model_sha256", status)
                with self.assertRaises(FileExistsError):
                    prep.prepare(config, cohort, output)

            # The original inputs are read-only and must still match their
            # P006 pin before any sampled blocks are materialized.
            task, config, cohort, events, artist_map, album_map, _teacher_model = self._fixture(Path(temp))
            (task / "artifacts/dev/feature_schema.json").write_text('{"feature_names": ["tampered"]}', encoding="utf-8")
            bad_output = task / "artifacts/p008_sampled_training/p001/prepare_bad_pin"
            with patch.object(prep, "TASK", task), patch.object(prep, "CONFIG_PATH", task / "configs/p008_sampled.json"), \
                 patch.object(prep, "PROTOCOL_PATH", task / "configs/P008_PROTOCOL.md"), \
                 patch.object(prep, "P006_MANIFEST_PATH", task / "configs/p006_freeze_manifest.json"), \
                 patch.object(prep, "P007_SNAPSHOT", task / "report/p007_snapshot"), \
                 patch.object(prep, "P007_SNAPSHOT_MANIFEST", task / "report/p007_snapshot/checkpoint_manifest.json"), \
                 patch.object(prep, "ResourceMonitor", _Monitor), \
                 patch.object(prep, "load_development_events", return_value=events), \
                 patch.object(prep, "load_mapping", side_effect=lambda path, _name:
                              artist_map if "artist" in str(path) else album_map), \
                 patch.object(prep, "_load_teacher", return_value=_Teacher()):
                with self.assertRaisesRegex(AssertionError, "P006 freeze manifest hash mismatch"):
                    prep.prepare(config, cohort, bad_output)


if __name__ == "__main__":
    unittest.main()
