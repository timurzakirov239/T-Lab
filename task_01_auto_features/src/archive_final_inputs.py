"""Archive the exact P006 frozen inputs without decoding event payloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

if __package__:
    from .verify_auto_checkpoint import sha256
else:
    from verify_auto_checkpoint import sha256

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parent
MANIFEST = TASK / "configs/p006_freeze_manifest.json"
LOCK = TASK / "requirements-pilot.lock.txt"
ARCHIVE = TASK / "artifacts/p006_frozen_inputs.zip"
REPORT = TASK / "report/p006_frozen_inputs_archive.json"
FORBIDDEN_PATH_PARTS = {
    "artifacts/p006_final_test/", "artifacts/p006_test_opened.json",
    "report/p006_snapshot/",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def build_archive() -> dict:
    if ARCHIVE.exists():
        raise FileExistsError(f"Refusing to overwrite frozen-input archive: {ARCHIVE}")
    if REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite archive report: {REPORT}")
    if not MANIFEST.is_file() or not LOCK.is_file():
        raise FileNotFoundError("P006 source manifest or requirements lock is missing")

    manifest = _json(MANIFEST)
    files = manifest.get("files")
    if manifest.get("experiment") != "P006" or not isinstance(files, dict) or len(files) != 72:
        raise ValueError("Expected the frozen P006 manifest with exactly 72 file hashes")

    entries: list[tuple[str, Path]] = []
    verified_hashes: dict[str, str] = {}
    byte_sizes: dict[str, int] = {}
    forbidden = []
    task_resolved = TASK.resolve()
    for relative, expected_hash in files.items():
        source = (TASK / Path(relative)).resolve()
        if not source.is_relative_to(task_resolved):
            raise ValueError(f"Manifest path escapes the task directory: {relative}")
        if not source.is_file():
            raise FileNotFoundError(f"Frozen source is missing: {relative}")
        archive_name = f"task_01_auto_features/{Path(relative).as_posix()}"
        if any(part in archive_name for part in FORBIDDEN_PATH_PARTS) or archive_name.endswith(".test_opened.json"):
            forbidden.append(archive_name)
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise AssertionError(f"Frozen source SHA-256 mismatch: {relative}")
        verified_hashes[archive_name] = actual_hash
        byte_sizes[archive_name] = source.stat().st_size
        entries.append((archive_name, source))
    if forbidden:
        raise ValueError(f"Freeze manifest unexpectedly contains P006 test outputs/markers: {forbidden}")

    manifest_name = "task_01_auto_features/configs/p006_freeze_manifest.json"
    lock_name = "task_01_auto_features/requirements-pilot.lock.txt"
    additions = ((manifest_name, MANIFEST), (lock_name, LOCK))
    known = {name for name, _ in entries}
    for archive_name, source in additions:
        if archive_name in known:
            raise ValueError(f"Archive addition duplicates a frozen source path: {archive_name}")
        entries.append((archive_name, source))
        known.add(archive_name)
        byte_sizes[archive_name] = source.stat().st_size
        verified_hashes[archive_name] = sha256(source)
    manifest_sha = sha256(MANIFEST)

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with zipfile.ZipFile(ARCHIVE, mode="x", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as bundle:
            created = True
            for archive_name, source in sorted(entries, key=lambda entry: entry[0]):
                bundle.write(source, arcname=archive_name, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=1)
    except Exception:
        if created and ARCHIVE.exists():
            ARCHIVE.unlink()
        raise

    try:
        with zipfile.ZipFile(ARCHIVE, mode="r") as bundle:
            if bundle.testzip() is not None:
                raise AssertionError("ZIP CRC check failed")
            actual_members = set(bundle.namelist())
            if actual_members != known or len(bundle.namelist()) != len(known):
                raise AssertionError("ZIP members differ from the exact frozen-input inventory")
            uncompressed_bytes = sum(info.file_size for info in bundle.infolist())
    except Exception:
        ARCHIVE.unlink(missing_ok=True)
        raise

    report = {
        "status": "verified",
        "experiment": "P006",
        "archive_path": ARCHIVE.relative_to(TASK).as_posix(),
        "archive_repo_relative_path": _repo_relative(ARCHIVE),
        "archive_sha256": sha256(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_member_count": len(known),
        "manifest_file_count": len(files),
        "uncompressed_bytes": uncompressed_bytes,
        "manifest_sha256": manifest_sha,
        "requirements_lock_path": _repo_relative(LOCK),
        "compression": "ZIP_DEFLATED",
        "compression_level": 1,
        "zip_crc_check": "passed",
        "frozen_manifest_sha256_checks": "72/72 passed",
        "event_payload_decoded": False,
        "test_outputs_or_reservation_markers_included": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(build_archive(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
