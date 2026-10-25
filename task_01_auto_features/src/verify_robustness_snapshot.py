"""Verify a compact P005 snapshot without loading source data or predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from .verify_auto_checkpoint import _compare_metrics, recompute, sha256
else:
    from verify_auto_checkpoint import _compare_metrics, recompute, sha256

TASK = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = TASK / "report/p005_snapshot"
TOLERANCE = 1e-12


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _rank_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Use the independent metric implementation with synthetic scores derived from ranks.

    P005 reports recall/NDCG only at 10 and 20. A known rank above 20 is
    therefore represented by rank 21; blank ranks represent a missing target.
    """
    queries, scores = [], []
    seen: set[int] = set()
    for row in rows:
        qid = int(row["query_id"])
        if qid in seen:
            raise ValueError(f"duplicate query_id {qid} in a model/stage rank group")
        seen.add(qid)
        raw_rank = row.get("target_rank", "").strip()
        rank = int(raw_rank) if raw_rank else None
        if rank is not None and rank < 1:
            raise ValueError(f"target rank must be one-based and positive, got {rank}")
        represented_rank = min(rank, 21) if rank is not None else None
        length = represented_rank if represented_rank is not None else 21
        candidates = list(range(length))
        query = {"query_id": qid, "user_id": int(row["user_id"]),
                 "target_item": represented_rank - 1 if represented_rank is not None else length + 1,
                 "candidate_items": candidates,
                 "target_is_repeat": row["target_is_repeat"].lower() == "true",
                 "target_known_before": row["target_known_before"].lower() == "true"}
        queries.append(query)
        scores.append([float(length - i) for i in range(length)])
    return recompute(queries, scores)


def _stored_metrics(snapshot_child: Path, model: str, stage: str) -> dict[str, Any]:
    if stage == "inner":
        rows = _read_json(snapshot_child / "inner_results.json")
        entry = next((row for row in rows if str(row.get("label")) == model), None)
        if entry is None:
            raise FileNotFoundError(f"inner result for {model} is missing")
        metrics = entry.get("metrics", entry.get("inner_metrics", {}))
        if "metrics" in metrics and isinstance(metrics["metrics"], dict):
            metrics = metrics["metrics"]
        return metrics
    path = snapshot_child / "results" / f"{model}.json"
    result = _read_json(path)
    metrics = result.get("metrics", {})
    if result.get("reused_from"):
        target = snapshot_child / "results" / f"{result['reused_from']}.json"
        if target.is_file():
            metrics = _read_json(target).get("metrics", metrics)
    return metrics


def _expected_groups(snapshot_child: Path) -> set[tuple[str, str]]:
    inner = _read_json(snapshot_child / "inner_results.json")
    metrics = _read_json(snapshot_child / "metrics.json")
    status = _read_json(snapshot_child / "run_status.json")
    final = metrics.get("final_models", status.get("final_models", []))
    expected: set[tuple[str, str]] = set()
    for entry in list(inner) + list(final):
        if entry.get("status") == "skipped_empty_group_ablation":
            continue
        label = entry.get("label")
        prediction = entry.get("prediction_file", entry.get("prediction_path", entry.get("predictions_path")))
        if not label and not prediction:
            continue
        if not label or not prediction:
            raise ValueError("saved result group must have both label and prediction path")
        stage = str(entry.get("stage", "inner" if entry in inner else "final"))
        if stage not in {"inner", "final"}:
            raise ValueError(f"unexpected saved result stage {stage!r}")
        key = (str(label), stage)
        if key in expected:
            raise ValueError(f"duplicate saved model/stage group: {key}")
        expected.add(key)
    return expected


def _validate_memberships(snapshot: Path, index: dict[str, Any], parent: dict[str, Any]) -> None:
    membership = _read_json(snapshot / "membership" / "memberships.json")
    if membership.get("status") != "frozen" or membership.get("disjoint_cohorts") is not True:
        raise AssertionError("saved memberships are not frozen and marked disjoint")
    excluded = set(map(int, membership.get("excluded_users", [])))
    listed = {int(row["seed"]): row for row in membership.get("cohorts", [])}
    indexed = {int(row["seed"]): row for row in index.get("cohorts", [])}
    summarized = {int(row["seed"]): row for row in parent.get("cohorts", [])}
    if (len(listed) != len(membership.get("cohorts", [])) or
            len(indexed) != len(index.get("cohorts", [])) or
            len(summarized) != len(parent.get("cohorts", [])) or
            set(listed) != set(indexed) or set(listed) != set(summarized)):
        raise AssertionError("snapshot, parent, and frozen membership cohort sets differ")
    prior: set[int] = set()
    for seed, member in listed.items():
        users = list(map(int, member.get("selected_users", [])))
        if users != sorted(set(users)) or len(users) != int(membership.get("users_per_cohort", len(users))):
            raise AssertionError(f"invalid selected users for cohort seed {seed}")
        if set(users) & excluded or set(users) & prior:
            raise AssertionError("selected users overlap exclusions or another cohort")
        prior.update(users)
        digest = hashlib.sha256(json.dumps(users, separators=(",", ":")).encode("ascii")).hexdigest()
        if member.get("selected_users_sha256") != digest:
            raise AssertionError(f"membership hash mismatch for cohort seed {seed}")
        if summarized[seed].get("status") != indexed[seed].get("status"):
            raise AssertionError(f"parent/snapshot status mismatch for cohort seed {seed}")
    if index.get("final_test_evaluated") is not False or parent.get("final_test_evaluated") is not False:
        raise AssertionError("P005 snapshot records final-test evaluation")
    verification = _read_json(snapshot / "verification_independent.json")
    if verification.get("status") != "verified" or verification.get("parent_status") != parent.get("status"):
        raise AssertionError("saved parent verification is missing or inconsistent")
    verification_rows = verification.get("cohorts", [])
    verified = {int(row["seed"]): row.get("status") for row in verification_rows}
    state_map = {"complete": "verified", "failed": "failed_checkpoint_verified"}
    if (len(verified) != len(verification_rows) or len(verified) != len(indexed) or
            verified != {seed: state_map.get(row.get("status")) for seed, row in indexed.items()}):
        raise AssertionError("parent independent verification does not cover snapshot cohort statuses")


def verify_snapshot(snapshot: Path = DEFAULT_SNAPSHOT, tolerance: float = TOLERANCE) -> dict[str, Any]:
    snapshot = Path(snapshot)
    manifest = _read_json(snapshot / "checkpoint_manifest.json")
    evidence = manifest.get("tracked_evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("checkpoint manifest has no tracked_evidence")
    for relative, expected in evidence.items():
        path = snapshot / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest evidence missing: {relative}")
        if path.stat().st_size != int(expected["bytes"]) or sha256(path) != expected["sha256"]:
            raise AssertionError(f"manifest evidence hash/size mismatch: {relative}")

    index = _read_json(snapshot / "snapshot_summary.json")
    parent = _read_json(snapshot / "summary.json")
    if index.get("experiment") != "P005" or index.get("parent_status") not in {"complete", "complete_with_failures"}:
        raise AssertionError("snapshot summary does not describe a completed P005 parent")
    if parent.get("status") != index.get("parent_status"):
        raise AssertionError("saved parent summary status differs from snapshot index")
    _validate_memberships(snapshot, index, parent)
    checks, failed = [], []
    for cohort in index.get("cohorts", []):
        child = snapshot / str(cohort["run_dir"])
        if cohort.get("status") == "failed":
            if cohort.get("rank_rows") is not None:
                raise AssertionError("failed child unexpectedly has compact rank rows")
            child_status = _read_json(child / "run_status.json")
            child_metrics = _read_json(child / "metrics.json")
            child_verification = _read_json(child / "verification_independent.json")
            if child_status.get("status") != "failed" or child_metrics.get("status") != "failed":
                raise AssertionError(f"failed snapshot cohort has inconsistent status: {cohort['run_dir']}")
            if child_verification.get("status") != "failed_checkpoint_verified":
                raise AssertionError(f"failed child lacks verified failure report: {cohort['run_dir']}")
            if child_status.get("final_test_evaluated") or child_metrics.get("final_test_evaluated"):
                raise AssertionError(f"failed cohort {cohort['run_dir']} records final-test evaluation")
            failed.append(str(cohort["run_dir"]))
            continue
        if cohort.get("status") != "complete":
            raise AssertionError(f"unknown child snapshot status: {cohort.get('status')!r}")
        child_status = _read_json(child / "run_status.json")
        child_metrics = _read_json(child / "metrics.json")
        child_verification = _read_json(child / "verification_independent.json")
        if child_status.get("status") != "complete" or child_metrics.get("status") != "complete":
            raise AssertionError(f"complete snapshot cohort has incomplete saved status: {cohort['run_dir']}")
        if child_verification.get("status") != "verified":
            raise AssertionError(f"complete child lacks verified independent report: {cohort['run_dir']}")
        if child_status.get("final_test_evaluated") or child_metrics.get("final_test_evaluated"):
            raise AssertionError(f"cohort {cohort['run_dir']} records final-test evaluation")
        groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        with (child / "query_ranks.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"model", "stage", "query_id", "user_id", "target_is_repeat", "target_known_before", "target_rank"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"rank CSV has missing columns in {cohort['run_dir']}")
            for row in reader:
                if row["stage"] not in {"inner", "final"}:
                    raise ValueError(f"unexpected rank stage {row['stage']!r}")
                groups[(row["model"], row["stage"])].append(row)
        if sum(len(rows) for rows in groups.values()) != int(cohort.get("rank_rows", -1)):
            raise AssertionError(f"rank row count differs from snapshot index for {cohort['run_dir']}")
        if set(groups) != _expected_groups(child):
            missing = sorted(_expected_groups(child) - set(groups))
            extra = sorted(set(groups) - _expected_groups(child))
            raise AssertionError(f"rank CSV model/stage groups differ from results for {cohort['run_dir']}; missing={missing}, extra={extra}")
        for (model, stage), rows in sorted(groups.items()):
            actual = _rank_metrics(rows)
            expected = _stored_metrics(child, model, stage)
            errors = _compare_metrics(expected, actual, tolerance)
            if errors:
                raise AssertionError(f"{cohort['run_dir']} {model}/{stage}: {'; '.join(errors[:8])}")
            checks.append({"run_dir": cohort["run_dir"], "model": model, "stage": stage,
                           "query_count": len(rows), "status": "verified"})
    return {"status": "PASS", "tracked_files": len(evidence), "ranking_groups_recomputed": len(checks),
            "rank_rows": sum(item["query_count"] for item in checks), "failed_children_skipped": failed,
            "tolerance": tolerance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    snapshot = args.snapshot if args.snapshot.is_absolute() else TASK / args.snapshot
    print(json.dumps(verify_snapshot(snapshot), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
