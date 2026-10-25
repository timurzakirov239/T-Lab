"""Download the public KION data and Yambda-50M with pinned provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile


TASK = Path(__file__).resolve().parents[1]
USER_AGENT = "T-Lab-research-data-download/1.0"


def get_json(url: str):
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as response:
        return json.load(response)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hashes(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1()
    git_blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    return sha256.hexdigest(), git_blob.hexdigest()


def download(spec: dict) -> dict:
    target = TASK / "data" / "raw" / spec["local_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    expected_size = spec["expected_size"]
    if not target.exists():
        for attempt in range(1, 6):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == expected_size:
                    break
                if offset > expected_size:
                    raise ValueError(f"Partial file exceeds expected size: {partial}")
                headers = {"User-Agent": USER_AGENT}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with urlopen(Request(spec["url"], headers=headers), timeout=90) as response:
                    if offset and response.status != 206:
                        offset = 0
                    if response.status == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected Content-Range")
                    downloaded = offset
                    last_log = time.monotonic()
                    with partial.open("ab" if offset else "wb") as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                            downloaded += len(chunk)
                            if time.monotonic() - last_log >= 10:
                                print(f"PROGRESS {spec['local_path']}: {downloaded:,}/{expected_size:,} bytes", flush=True)
                                last_log = time.monotonic()
                if partial.stat().st_size != expected_size:
                    raise ValueError(f"Size mismatch for {spec['local_path']}")
                break
            except (OSError, HTTPError, ValueError) as error:
                # Do not print redirect URLs, which may contain temporary signed tokens.
                print(f"RETRY {attempt}/5 {spec['local_path']}: {type(error).__name__}", flush=True)
                if attempt == 5:
                    raise RuntimeError(f"Download failed: {spec['local_path']}") from None
                time.sleep(min(2 ** attempt, 15))
        check_path = partial
    else:
        check_path = target
    if check_path.stat().st_size != expected_size:
        raise ValueError(f"File size mismatch: {check_path}")
    sha256, git_blob = file_hashes(check_path)
    if spec.get("expected_sha256") and sha256 != spec["expected_sha256"]:
        raise ValueError(f"SHA256 mismatch: {check_path}")
    if spec.get("expected_git_blob_sha1") and git_blob != spec["expected_git_blob_sha1"]:
        raise ValueError(f"Git blob SHA1 mismatch: {check_path}")
    if check_path == partial:
        partial.replace(target)
    print(f"VERIFIED {spec['local_path']}: {expected_size:,} bytes", flush=True)
    return {**spec, "bytes": expected_size, "sha256": sha256, "verified_at_utc": datetime.now(timezone.utc).isoformat()}


def extract_kion_zip():
    root = TASK / "data" / "raw" / "kion"
    root_resolved = root.resolve()
    extracted = []
    with zipfile.ZipFile(root / "data_original.zip") as archive:
        for entry in archive.infolist():
            destination = (root / entry.filename).resolve()
            if not destination.is_relative_to(root_resolved):
                raise ValueError("Archive contains a path outside the dataset directory")
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"ZIP CRC check failed: {corrupt}")
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith("data_original/"):
                continue
            destination = root / entry.filename
            if destination.exists():
                if destination.stat().st_size != entry.file_size:
                    raise ValueError(f"Existing extracted file size mismatch: {destination}")
                with destination.open("rb") as local, archive.open(entry) as original:
                    while True:
                        local_chunk, original_chunk = local.read(4 * 1024 * 1024), original.read(4 * 1024 * 1024)
                        if local_chunk != original_chunk:
                            raise ValueError(f"Existing extracted file content mismatch: {destination}")
                        if not local_chunk:
                            break
            else:
                archive.extract(entry, root)
            sha256, _ = file_hashes(destination)
            extracted.append({"path": destination.relative_to(TASK).as_posix(), "bytes": destination.stat().st_size, "sha256": sha256, "zip_crc_verified": True})
    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    manifest_path = TASK / "data" / "download_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    github_revision = previous.get("kion_revision") or get_json("https://api.github.com/repos/irsafilo/KION_DATASET/commits/main")["sha"]
    hf_revision = previous.get("yambda_revision") or get_json("https://huggingface.co/api/datasets/yandex/yambda")["sha"]
    github_tree = get_json(f"https://api.github.com/repos/irsafilo/KION_DATASET/git/trees/{github_revision}?recursive=1")
    hf_tree = get_json(f"https://huggingface.co/api/datasets/yandex/yambda/tree/{hf_revision}?recursive=true&expand=false")
    github_files = {entry["path"]: entry for entry in github_tree["tree"] if entry["type"] == "blob"}
    hf_files = {entry["path"]: entry for entry in hf_tree if entry["type"] == "file"}
    specs = []
    for remote_path in ["README.md", "data_original.zip"]:
        entry = github_files[remote_path]
        local_path = "kion/" + ("SOURCE_README.md" if remote_path == "README.md" else remote_path)
        specs.append({"dataset": "kion", "remote_path": remote_path, "local_path": local_path, "url": f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{github_revision}/{remote_path}", "expected_size": entry["size"], "expected_git_blob_sha1": entry["sha"]})
    for remote_path in ["README.md", "LICENSE", "flat/50m/multi_event.parquet", "artist_item_mapping.parquet", "album_item_mapping.parquet"]:
        entry = hf_files[remote_path]
        local_path = "yambda/" + ("SOURCE_README.md" if remote_path == "README.md" else remote_path)
        spec = {"dataset": "yambda", "remote_path": remote_path, "local_path": local_path, "url": f"https://huggingface.co/datasets/yandex/yambda/resolve/{hf_revision}/{remote_path}", "expected_size": entry["size"]}
        if entry.get("lfs"):
            spec["expected_sha256"] = entry["lfs"]["oid"]
        else:
            spec["expected_git_blob_sha1"] = entry["oid"]
        specs.append(spec)
    manifest = {"status": "downloading", "started_at_utc": datetime.now(timezone.utc).isoformat(), "kion_revision": github_revision, "yambda_revision": hf_revision, "selection": "KION public interactions and original Russian metadata; Yambda flat/50m unified events and artist/album mappings. No local event sampling.", "files": [], "planned_files": specs}
    write_json(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as executor:
        futures = [executor.submit(download, spec) for spec in specs]
        for future in as_completed(futures):
            manifest["files"].append(future.result())
            write_json(manifest_path, manifest)
    manifest["extracted_files"] = extract_kion_zip()
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print(f"COMPLETE manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
