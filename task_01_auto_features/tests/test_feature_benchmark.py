import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from task_01_auto_features.src.pilot_features import FeatureEngine
from task_01_auto_features.src.run_feature_benchmark import feature_build, verify_builds


class NoOpMonitor:
    def check(self):
        pass


class FeatureBenchmarkTests(unittest.TestCase):
    def inputs(self):
        events = {"uid": np.asarray([1, 1], dtype=np.uint32), "item_id": np.asarray([10, 99], dtype=np.uint32),
                  "timestamp": np.asarray([1, 4], dtype=np.uint32), "event_code": np.asarray([0, 0], dtype=np.uint8),
                  "event_names": ["listen"]}
        queries = [{"query_id": 7, "user_id": 1, "timestamp": 4, "candidate_items": [10, 99], "target_item": 99,
                    "target_is_repeat": False, "target_known_before": False}]
        config = {"short_window_raw": 5, "half_life_raw": 5, "feature_count": 92}
        return events, queries, config

    def build(self, directory):
        directory.mkdir()
        events, queries, config = self.inputs()
        progress = {}
        result = feature_build(FeatureEngine, events, queries, {}, {}, config, directory, NoOpMonitor(), progress=progress)
        status = {"status": "complete", "config_sha256": "same-config", "input_sha256": {"same": "same"}, "build": result}
        (directory / "metrics.json").write_text(json.dumps(status), encoding="utf-8")
        return result, progress

    def test_incremental_matrix_preserves_order_and_excludes_current_target_event(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "build"
            result, progress = self.build(output)
            matrix = np.load(output / "features.npy", mmap_mode="r")
            expected = FeatureEngine({}, {}, 5, 5)
            expected.observe({"uid": 1, "item_id": 10, "timestamp": 1, "event_type": "listen"})
            np.testing.assert_array_equal(matrix, expected.transform(1, [10, 99], 4))
            self.assertEqual(progress, {"queries_completed": 1, "rows_written": 2, "events_observed": 1})
            self.assertEqual(result["matrix_values_bytes"], 2 * 92 * 4)
            structural = (output / "feature_inputs.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("target", structural)
            self.assertEqual(json.loads(structural)["candidate_items"], [10, 99])
            del matrix

    def test_full_equivalence_pass_and_changed_float_rejects_scale_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.build(base / "a")
            self.build(base / "b")
            result = verify_builds(base / "a", base / "b", base / "pass.json")
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["cells_compared"], 184)
            with self.assertRaises(FileExistsError):
                verify_builds(base / "a", base / "b", base / "pass.json")
            matrix = np.load(base / "b/features.npy", mmap_mode="r+")
            matrix[0, 0] += 1
            matrix.flush()
            raw_digest = hashlib.sha256(memoryview(matrix).cast("B")).hexdigest()
            del matrix
            status_path = base / "b/metrics.json"
            status = json.loads(status_path.read_text())
            status["build"]["matrix_values_sha256"] = raw_digest
            status["build"]["matrix_file_sha256"] = hashlib.sha256((base / "b/features.npy").read_bytes()).hexdigest()
            status_path.write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaises(AssertionError):
                verify_builds(base / "a", base / "b", base / "fail.json")
            self.assertEqual(json.loads((base / "fail.json").read_text())["status"], "fail")


if __name__ == "__main__":
    unittest.main()
