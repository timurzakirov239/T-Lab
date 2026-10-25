import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from task_01_auto_features.src.pilot_data import _prepare
from task_01_auto_features.src.pilot_storage import ensure_run_writable
from task_01_auto_features.src.run_pilot import main


class PilotStorageTests(unittest.TestCase):
    def test_new_output_is_allowed_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-run"
            ensure_run_writable(output)
            self.assertFalse(output.exists())

    def test_frozen_or_unreadable_status_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            status = output / "run_status.json"
            for content in ('{"status":"complete"}', 'broken-json', '[]'):
                with self.subTest(content=content):
                    status.write_text(content, encoding="utf-8")
                    with self.assertRaises(FileExistsError):
                        ensure_run_writable(output)
                    self.assertEqual(status.read_text(encoding="utf-8"), content)
            status.write_text('{"status":"running"}', encoding="utf-8")
            (output / "FROZEN.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_run_writable(output)

    def test_preparation_cannot_delete_completed_run_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "data"
            output.mkdir()
            (output.parent / "run_status.json").write_text('{"status":"complete"}', encoding="utf-8")
            sentinel = output / "events.parquet"
            sentinel.write_bytes(b"original evidence")
            missing = output / "nonexistent-source.parquet"
            with self.assertRaises(FileExistsError):
                _prepare(output, 64, 1, missing, missing, missing, {})
            self.assertEqual(sentinel.read_bytes(), b"original evidence")

    def test_runner_refuses_before_mutating_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "completed"
            output.mkdir()
            original = b'{"status":"complete","sentinel":123}'
            status = output / "run_status.json"
            status.write_bytes(original)
            config = base / "config.json"
            config.write_text(json.dumps({"test_evaluation": False}), encoding="utf-8")
            with patch.object(sys, "argv", ["run_pilot.py", "--output", str(output), "--config", str(config)]):
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(status.read_bytes(), original)
            self.assertEqual([p.name for p in output.iterdir()], ["run_status.json"])


if __name__ == "__main__":
    unittest.main()
