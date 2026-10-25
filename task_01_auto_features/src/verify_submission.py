"""Run the portable P001, P006, P007, and P008 snapshot checks.

This entry point invokes the existing verifiers as child processes. Checks run
from the repository root except P001, which is staged with only its compact
snapshot to keep the optional raw-input archive out of the verification path.
It does not launch any preparation, fitting, or scoring run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parent


def _run_verifier(name: str, arguments: list[str] | None = None,
                  stage_root: Path | None = None) -> dict[str, Any]:
    path = (stage_root / "task_01_auto_features/src" / name) if stage_root else TASK / "src" / name
    command = [sys.executable, str(path), *(arguments or [])]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=stage_root or REPO, env=environment,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(
            f"{name} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-6000:]}\nstderr:\n{completed.stderr[-6000:]}"
        )
    output = completed.stdout.strip()
    try:
        result = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{name} did not print one JSON result: {output[-3000:]}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"{name} result is not a JSON object")
    return {"verifier": name, "status": "PASS", "result": result,
            "stderr": completed.stderr.strip() or None}


def _verify_p007_compact_snapshot() -> dict[str, Any]:
    """Check P007 snapshot file hashes and its saved independent receipt."""
    snapshot = TASK / "report/p007_snapshot"
    manifest_path = snapshot / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("P007 checkpoint manifest must be a non-empty relative-path hash map")
    checked = 0
    for relative, expected in manifest.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe P007 snapshot manifest path: {relative}")
        path = snapshot / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"P007 snapshot evidence missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise AssertionError(f"P007 snapshot evidence hash mismatch: {relative}")
        checked += 1
    independent = json.loads((snapshot / "verification_independent.json").read_text(encoding="utf-8"))
    if independent.get("status") not in {"PASS", "verified"} or independent.get("experiment") != "P007":
        raise AssertionError("P007 saved independent verification receipt is not PASS")
    return {"verifier": "P007 compact snapshot receipt", "status": "PASS",
            "result": {"tracked_files": checked, "independent_status": independent["status"],
                       "cohorts": independent.get("cohort_count")}}


def _run_p001_snapshot_verifier() -> dict[str, Any]:
    """Run P001 against an isolated compact snapshot, excluding the local ZIP.

    The legacy verifier also checks the optional frozen-input archive. The
    temporary staging area intentionally has no such archive, so no raw
    archived inputs are read during this portable check.
    """
    with tempfile.TemporaryDirectory(prefix="p001_snapshot_check_") as temporary:
        stage = Path(temporary)
        task = stage / "task_01_auto_features"
        (task / "src").mkdir(parents=True)
        shutil.copy2(TASK / "src/verify_pilot_checkpoint.py", task / "src/verify_pilot_checkpoint.py")
        shutil.copytree(TASK / "report/pilot_001_snapshot", task / "report/pilot_001_snapshot")
        return _run_verifier("verify_pilot_checkpoint.py", stage_root=stage)


def verify_all() -> dict[str, Any]:
    checks = [
        _run_p001_snapshot_verifier(),
        _run_verifier("verify_final_snapshot.py"),
        _run_verifier("report_expanded_search.py", ["--verify"]),
        _run_verifier("report_sampled_training.py", ["--verify"]),
    ]
    return {
        "status": "PASS",
        "scope": "P001, P006, P007, and P008 compact snapshots",
        "data_access": "Snapshot evidence only; no P006 runner, model fitting, or scoring is launched.",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="optional receipt path; existing files are never overwritten")
    args = parser.parse_args()
    receipt = verify_all()
    serialized = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
        return 0
    output = args.output if args.output.is_absolute() else REPO / args.output
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"submission verification receipt already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    print(json.dumps({"status": "PASS", "receipt": str(output), "checks": len(receipt["checks"])},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
