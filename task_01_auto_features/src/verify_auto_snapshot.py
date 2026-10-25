"""Verify compact P004 evidence and independently recompute per-user rank metrics."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

if __package__:
    from .verify_auto_checkpoint import recompute, sha256
else:
    from verify_auto_checkpoint import recompute, sha256

TASK = Path(__file__).resolve().parents[1]


def main():
    snapshot = TASK / "report/p004_snapshot"
    manifest = json.loads((snapshot / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["tracked_evidence"].items():
        path = snapshot / name
        assert sha256(path) == expected["sha256"] and path.stat().st_size == expected["bytes"], name
    groups = defaultdict(list)
    with (snapshot / "query_ranks.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            groups[row["model"]].append(row)
    for model, rows in groups.items():
        saved = json.loads((snapshot / "results" / f"{model}.json").read_text(encoding="utf-8"))
        assert len({row["query_id"] for row in rows}) == len(rows)
        queries, scores = [], []
        for row in rows:
            rank = int(row["target_rank"]) if row["target_rank"] else None
            assert rank is None or 1 <= rank <= 100
            queries.append({"query_id": int(row["query_id"]), "user_id": int(row["user_id"]),
                            "target_item": rank - 1 if rank is not None else 999,
                            "candidate_items": list(range(100)), "target_is_repeat": row["target_is_repeat"] == "True",
                            "target_known_before": row["target_known_before"] == "True"})
            scores.append(list(range(100, 0, -1)))
        actual = recompute(queries, scores)
        for key, values in [("overall", actual["overall"]), *actual["strata"].items()]:
            expected = saved["metrics"]["overall"] if key == "overall" else saved["metrics"]["strata"][key]
            assert values["query_count"] == expected["query_count"] and values["user_count"] == expected["user_count"]
            for average in ("query_micro", "user_macro"):
                for metric, value in values[average].items():
                    assert abs(value - expected[average][metric]) < 1e-12, (model, key, average, metric)
    status = json.loads((snapshot / "metrics.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete" and not status["final_test_evaluated"] and status["fits_actual"] <= 28
    gate = json.loads((snapshot / "a1_equivalence.json").read_text(encoding="utf-8"))
    assert gate["status"] == "pass" and gate["cells_compared"] == 12_196_800
    print(json.dumps({"status": "PASS", "tracked_files": len(manifest["tracked_evidence"]),
                      "ranking_results_recomputed": len(groups), "rank_rows": sum(map(len, groups.values())),
                      "final_test_evaluated": False}))


if __name__ == "__main__":
    main()
