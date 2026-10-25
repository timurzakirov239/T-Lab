"""Verify tracked P002 evidence and recompute recall from its compact hit table."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
SNAPSHOT = TASK / "report/p002_snapshot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((SNAPSHOT / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["tracked_evidence"].items():
        assert sha256(SNAPSHOT / relative) == expected["sha256"], relative
        assert (SNAPSHOT / relative).stat().st_size == expected["bytes"], relative
    metrics = json.loads((SNAPSHOT / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "complete" and not metrics["final_test_evaluated"] and not metrics["model_training"]
    with (SNAPSHOT / "validation_hits.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({row["query_id"] for row in rows}) == 499
    assert len({row["user_id"] for row in rows}) == 52
    def verify(group: list[dict], expected: dict, strategy: str) -> None:
        hits = [int(row[f"{strategy}_hit"]) for row in group]
        assert all(hit in (0, 1) for hit in hits)
        by_user = defaultdict(list)
        for row, hit in zip(group, hits, strict=True):
            by_user[row["user_id"]].append(hit)
        macro = sum(sum(v)/len(v) for v in by_user.values()) / len(by_user)
        assert len(group) == expected["queries"] and len(by_user) == expected["users"]
        assert sum(hits) == expected["hits"] and len(group)-sum(hits) == expected["misses"]
        assert math.isclose(sum(hits)/len(group), expected["candidate_recall_query_micro"], abs_tol=1e-12)
        assert math.isclose(macro, expected["candidate_recall_user_macro"], abs_tol=1e-12)
    for strategy, values in metrics["strategies"].items():
        verify(rows, values["validation"], strategy)
        for name, field, value in (("repeat", "target_is_repeat", "True"), ("new", "target_is_repeat", "False"),
                                   ("warm", "target_known_before", "True"), ("cold", "target_known_before", "False")):
            verify([row for row in rows if row[field] == value], values["strata"][name], strategy)
    checked_outputs = 0
    for name, expected in metrics["output_sha256"].items():
        path = TASK / "artifacts/retrieval_002" / name
        if path.exists():
            assert sha256(path) == expected, name
            checked_outputs += 1
    print(json.dumps({"status": "PASS", "tracked_files": len(manifest["tracked_evidence"]),
                      "strategies_and_strata_recomputed": "4 x 5", "local_raw_output_hashes": checked_outputs}))


if __name__ == "__main__":
    main()
