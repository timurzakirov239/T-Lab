"""Check the tracked P001 evidence and, when present, its local full backup."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "task_01_auto_features/report/pilot_001_snapshot"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads((SNAPSHOT / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    checked = 0
    for relative, expected in manifest["tracked_evidence"].items():
        path = SNAPSHOT / relative
        if digest(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Frozen evidence mismatch: {relative}")
        checked += 1
    run = json.loads((SNAPSHOT / "metrics.json").read_text(encoding="utf-8"))
    assert run["status"] == "complete" and not run["final_test_evaluated"]
    assert run["config"]["seed"] == 20261025 and run["unique_users"] == 64
    backup = ROOT / manifest["local_backup"]["path"]
    if backup.exists():
        if digest(backup) != manifest["local_backup"]["sha256"]:
            raise ValueError("Full backup checksum mismatch")
        with zipfile.ZipFile(backup) as archive:
            for relative, expected in manifest["archived_files"].items():
                payload = archive.read(relative)
                if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                    raise ValueError(f"Archived file mismatch: {relative}")
            for name, expected in run["source_code_sha256"].items():
                path = f"task_01_auto_features/src/{name}"
                if manifest["archived_files"][path]["sha256"] != expected:
                    raise ValueError(f"Original source mismatch: {name}")
        backup_status = "verified (all archived payloads and original source hashes)"
    else:
        backup_status = "not present locally; expected after a fresh Git clone"
    print(json.dumps({"status": "PASS", "tracked_evidence_files": checked,
                      "local_full_backup": backup_status}, ensure_ascii=False))


if __name__ == "__main__":
    main()
