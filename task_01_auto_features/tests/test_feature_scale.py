import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from task_01_auto_features.src.retrieval_candidates import RetrievalCandidateGenerator
from task_01_auto_features.src.run_feature_scale import select_fixed_queries, seal_scale_candidates, validate_equivalence_gate
from task_01_auto_features.src.run_pilot import sha256


class Monitor:
    def check(self):
        pass


class FeatureScaleTests(unittest.TestCase):
    def gate_fixture(self, root):
        sources = {"run_feature_benchmark.py", "pilot_features.py", "run_retrieval.py", "run_pilot.py", "pilot_storage.py"}
        (root / "src").mkdir()
        for name in sources | {"optimized_features.py"}:
            (root / "src" / name).write_text(name)
        protocol = root / "configs/P003_PROTOCOL.md"
        protocol.parent.mkdir()
        protocol.write_text("protocol")
        snapshot = root / "report/pilot_001_snapshot"
        snapshot.mkdir(parents=True)
        original = {"prepared_data_sha256": {"events.parquet": "events"},
                    "source_code_sha256": {"pilot_features.py": sha256(root / "src/pilot_features.py")}}
        (snapshot / "metrics.json").write_text(json.dumps(original))
        (snapshot / "checkpoint_manifest.json").write_text(json.dumps({"archived_files": {
            "task_01_auto_features/artifacts/pilot_001/queries.jsonl": {"sha256": "queries"}}}))
        inputs = {"artifacts/pilot_001/data/events.parquet": "events", "artifacts/pilot_001/queries.jsonl": "queries"}
        gate = {"status": "pass", "numpy_array_equal_all_cells": True, "query_candidate_order_identical": True,
                "feature_schema_identical": True, "cells_compared": 184800 * 92, "shape": [184800, 92],
                "dtype": "float32", "matrix_file_sha256": []}
        for side, engine in (("left", "baseline"), ("right", "optimized")):
            folder = root / side
            folder.mkdir()
            build = {}
            for filename, field in (("features.npy", "matrix_file_sha256"), ("feature_inputs.jsonl", "feature_inputs_sha256"),
                                     ("feature_schema.json", "feature_schema_sha256")):
                (folder / filename).write_text(filename)
                build[field] = sha256(folder / filename)
            keys = sources | ({"optimized_features.py"} if side == "right" else set())
            status = {"status": "complete", "engine": engine, "profiled": False, "config_sha256": "config",
                      "protocol_sha256": sha256(protocol), "input_sha256": inputs, "build": build,
                      "source_code_sha256": {name: sha256(root / "src" / name) for name in keys}}
            (folder / "metrics.json").write_text(json.dumps(status))
            gate[side] = str(folder)
            gate["matrix_file_sha256"].append(build["matrix_file_sha256"])
        path = root / "gate.json"
        path.write_text(json.dumps(gate))
        return path

    def test_scale_gate_binds_source_structure_schema_and_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = self.gate_fixture(root)
            with patch("task_01_auto_features.src.run_feature_scale.TASK", root):
                result = validate_equivalence_gate(path, "config")
                self.assertEqual(result["status"], "pass")
                self.assertEqual(len(result["verified_provenance"]), 2)
                with self.assertRaises(ValueError):
                    validate_equivalence_gate(path, "changed-config")
                for changed in (root / "src/pilot_features.py", root / "right/feature_inputs.jsonl", root / "right/feature_schema.json"):
                    previous = changed.read_bytes()
                    changed.write_text("changed")
                    with self.assertRaises(ValueError):
                        validate_equivalence_gate(path, "config")
                    changed.write_bytes(previous)
                metrics = root / "right/metrics.json"
                saved = json.loads(metrics.read_text())
                saved["input_sha256"]["artifacts/pilot_001/queries.jsonl"] = "wrong"
                metrics.write_text(json.dumps(saved))
                with self.assertRaises(ValueError):
                    validate_equivalence_gate(path, "config")

    def config(self):
        return {"seed": 17, "boundaries_raw": {"train_query_start": 2, "train_end": 5, "validation_end": 9},
                "train_queries_per_user": 30, "validation_queries_per_user": 10, "max_candidates": 100}

    def events(self, rows):
        rows = sorted(rows, key=lambda r: (r[2], r[0], r[1]))
        return {"uid": np.array([r[0] for r in rows]), "item_id": np.array([r[1] for r in rows]),
                "timestamp": np.array([r[2] for r in rows]), "event_code": np.array([r[3] for r in rows]),
                "event_names": ["listen", "like"]}

    def test_fixed_boundaries_ties_and_reproducible_query_sampling(self):
        events = self.events([(1, 1, 1, 0), (1, 2, 2, 0), (1, 3, 2, 1),
                              (1, 4, 3, 0), (1, 5, 3, 0), (2, 6, 5, 0), (2, 7, 8, 0)])
        queries, info = select_fixed_queries(events, self.config())
        self.assertEqual([(q["timestamp"], q["split"]) for q in queries],
                         [(2, "train"), (5, "validation"), (8, "validation")])
        self.assertEqual(info["excluded_ambiguous_listen_groups"], {"train": 1})
        self.assertEqual(select_fixed_queries(events, self.config()), (queries, info))

    def test_scale_candidates_exact_p002_repeat_heavy_and_label_independent(self):
        artists = {i: (i % 7,) for i in range(1, 181)}
        albums = {i: (i % 11,) for i in range(1, 181)}
        events = self.events([(i % 3 + 1, i, i, 0) for i in range(1, 181)] +
                             [(1, 179, 181, 0), (2, 180, 181, 1)])
        queries = [{"query_id": i, "user_id": uid, "timestamp": t, "target_item": 999, "split": "train"}
                   for i, (uid, t) in enumerate(((1, 100), (2, 100), (1, 181), (1, 182)))]
        generator = RetrievalCandidateGenerator(artists, albums)
        expected, index = [], 0
        for q in queries:
            while index < len(events["timestamp"]) and events["timestamp"][index] < q["timestamp"]:
                generator.observe({"uid": int(events["uid"][index]), "item_id": int(events["item_id"][index]),
                                   "timestamp": int(events["timestamp"][index]),
                                   "event_type": events["event_names"][int(events["event_code"][index])]})
                index += 1
            expected.append(generator.candidates(q["user_id"], q["timestamp"], "repeat_heavy").candidate_items)
        with tempfile.TemporaryDirectory() as d:
            for label, target in (("a", 999), ("b", 1)):
                output = Path(d) / label
                output.mkdir()
                actual_queries = copy.deepcopy(queries)
                for q in actual_queries:
                    q["target_item"] = target
                seal_scale_candidates(events, actual_queries, artists, albums, self.config(), output, Monitor())
                actual = [json.loads(line)["candidate_items"] for line in (output / "queries.jsonl").read_text().splitlines()]
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
