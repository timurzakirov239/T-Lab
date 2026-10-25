"""Build compact P006 final-test evidence after independent verification and analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .verify_auto_checkpoint import sha256
else:
    from verify_auto_checkpoint import sha256

TASK = Path(__file__).resolve().parents[1]
DEFAULT_RUN = TASK / "artifacts/p006_final_test"
DEFAULT_SNAPSHOT = TASK / "report/p006_snapshot"
EXPECTED_LABELS = ("Popularity", "M_I", "M_UIDC", "Auto_A1", "Auto_A2", "Union")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_status(value: dict[str, Any]) -> bool:
    return value.get("status") in {"verified", "verified_with_failures"}


def _copy(source: Path, dest: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def _rank(query: dict[str, Any], scores: np.ndarray) -> int | None:
    candidates = [int(item) for item in query["candidate_items"]]
    if len(candidates) != len(scores):
        raise ValueError(f"prediction/candidate count mismatch for query {query.get('query_id')}")
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite prediction score at query {query.get('query_id')}")
    ordered = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), candidates[i]))
    try:
        return next(position + 1 for position, index in enumerate(ordered)
                    if candidates[index] == int(query["target_item"]))
    except StopIteration:
        return None


def _load_prediction(path: Path, expected_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        id_key = next((name for name in ("query_ids", "query_id") if name in archive), None)
        offset_key = next((name for name in ("offsets", "candidate_offsets") if name in archive), None)
        score_key = next((name for name in ("scores", "predictions", "probabilities") if name in archive), None)
        if not id_key or not offset_key or not score_key:
            raise ValueError(f"prediction archive lacks query IDs, offsets, or scores: {path}")
        query_ids = np.asarray(archive[id_key], dtype=np.int64)
        offsets = np.asarray(archive[offset_key], dtype=np.int64)
        scores = np.asarray(archive[score_key], dtype=np.float64)
    if query_ids.tolist() != expected_ids:
        raise AssertionError(f"prediction query IDs differ from saved query order: {path}")
    if offsets.ndim != 1 or len(offsets) != len(query_ids) + 1 or not len(offsets) or offsets[0] != 0:
        raise ValueError(f"invalid prediction offsets: {path}")
    if offsets[-1] != len(scores) or np.any(np.diff(offsets) < 0):
        raise ValueError(f"prediction offsets do not cover score vector: {path}")
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite prediction scores: {path}")
    return offsets, scores


def _snapshot_child(run: Path, dest: Path, entry: dict[str, Any], child_verification: dict[str, Any]) -> dict[str, Any]:
    cohort_id = str(entry["id"])
    child = run / str(entry.get("run_dir", cohort_id))
    dest.mkdir(parents=True)
    record = {"id": cohort_id, "run_dir": cohort_id, "status": entry["status"],
              "final_test_evaluated": bool(child_verification.get("final_test_evaluated", False)),
              "rank_rows": 0, "rank_groups": []}
    for name in ("metrics.json", "run_status.json", "verification_independent.json"):
        target = dest / name
        if name == "verification_independent.json":
            target.write_text(json.dumps(child_verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        elif (child / name).is_file():
            _copy(child / name, target)
    for name in ("masks_frozen.json", "specs.json", "a1_equivalence.json", "feature_schema.json"):
        if (child / name).is_file():
            _copy(child / name, dest / name)
    for relative in (Path("baseline/feature_schema.json"),):
        if (child / relative).is_file():
            _copy(child / relative, dest / relative)
    for relative in (Path("FAILED.json"), Path("data/FAILED.json"), Path("data/metadata.json")):
        if (child / relative).is_file():
            _copy(child / relative, dest / relative)
    log = run / f"{cohort_id}.log"
    if log.is_file():
        _copy(log, dest / log.name)

    if entry["status"] != "complete":
        # Keep partial status/diagnostics, but do not emit result JSON or ranks
        # that could be mistaken for completed test scores.
        record["rank_groups"] = []
        record["rank_rows"] = None
        record["score_policy"] = "failed cohort; no score reconstruction or result sidecars copied"
        return record

    for required in ("metrics.json", "run_status.json", "masks_frozen.json", "specs.json",
                     "a1_equivalence.json", "feature_schema.json"):
        if not (child / required).is_file():
            raise FileNotFoundError(f"completed P006 child lacks {required}: {cohort_id}")
    status = _json(child / "run_status.json")
    metrics = _json(child / "metrics.json")
    if status != metrics or status.get("status") != "complete" or not status.get("final_test_evaluated"):
        raise ValueError(f"P006 child is not a completed, evaluated test cohort: {cohort_id}")
    queries = [json.loads(line) for line in (child / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [int(query["query_id"]) for query in queries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate query IDs in {cohort_id}")
    results_by_label = {}
    for result_path in sorted((child / "results").glob("*.json")):
        result = _json(result_path)
        label = str(result.get("label", ""))
        if not label or label in results_by_label:
            raise ValueError(f"missing/duplicate result label in {result_path}")
        results_by_label[label] = (result, result_path)
        _copy(result_path, dest / "results" / result_path.name)
    final_entries = status.get("final_models", [])
    status_by_label = {str(result.get("label")): result for result in final_entries}
    if len(status_by_label) != len(final_entries) or set(status_by_label) != set(EXPECTED_LABELS):
        raise ValueError(f"completed cohort {cohort_id} must contain exactly six P006 models")
    if set(results_by_label) != set(EXPECTED_LABELS):
        raise ValueError(f"completed cohort {cohort_id} lacks result sidecars for all six models")

    rank_path = dest / "query_ranks.csv"
    fields = ["model", "stage", "query_id", "user_id", "candidate_count", "target_is_repeat",
              "target_known_before", "target_rank"]
    group_names = []
    rank_rows = 0
    with rank_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in EXPECTED_LABELS:
            result, source_path = results_by_label[label]
            status_entry = status_by_label[label]
            if result != status_entry:
                raise AssertionError(f"{cohort_id}/{label} result sidecar differs from run status entry")
            if result.get("stage") != "final_test" or result.get("evaluation_period") != "test":
                raise AssertionError(f"{cohort_id}/{label} is not explicitly a final test result")
            if result.get("prediction_sha256") != sha256(child / result["prediction_file"]):
                raise AssertionError(f"{cohort_id}/{label} prediction SHA mismatch before compaction")
            prediction = child / result["prediction_file"]
            offsets, scores = _load_prediction(prediction, ids)
            for index, query in enumerate(queries):
                raw = scores[offsets[index]:offsets[index + 1]]
                rank = _rank(query, raw)
                writer.writerow({"model": label, "stage": "final_test", "query_id": int(query["query_id"]),
                                 "user_id": int(query["user_id"]), "candidate_count": len(query["candidate_items"]),
                                 "target_is_repeat": bool(query["target_is_repeat"]),
                                 "target_known_before": bool(query["target_known_before"]),
                                 "target_rank": rank if rank is not None else ""})
                rank_rows += 1
            group_names.append(label)
    record.update(rank_rows=rank_rows, rank_groups=group_names)
    return record


def snapshot_checkpoint(run: Path = DEFAULT_RUN, snapshot: Path = DEFAULT_SNAPSHOT,
                        task_root: Path = TASK) -> dict[str, Any]:
    run, snapshot, task_root = Path(run), Path(snapshot), Path(task_root)
    if snapshot.exists():
        raise FileExistsError(f"P006 snapshot path already exists: {snapshot}")
    summary = _json(run / "summary.json")
    if summary.get("status") not in {"complete", "complete_with_failures"}:
        raise ValueError("P006 parent must be complete or complete_with_failures before snapshot")
    verification_path = run / "verification_independent.json"
    analysis_path = run / "analysis.json"
    if not verification_path.is_file() or not analysis_path.is_file():
        raise FileNotFoundError("P006 snapshot requires parent independent verification and analysis.json")
    verification = _json(verification_path)
    analysis = _json(analysis_path)
    if not _verified_status(verification):
        raise ValueError("P006 parent independent verification is not verified")
    if verification.get("final_test_evaluated") != summary.get("final_test_evaluated"):
        raise ValueError("P006 parent verifier and summary disagree on test evaluation")
    if analysis.get("status") != summary.get("status") or analysis.get("final_test_evaluated") != summary.get("final_test_evaluated"):
        raise ValueError("P006 analysis status/evaluation flags disagree with parent summary")

    config_path = task_root / "configs/p006_final_test.json"
    manifest_path = task_root / "configs/p006_freeze_manifest.json"
    protocol_path = task_root / "configs/P006_PROTOCOL.md"
    config = _json(config_path)
    configured = config.get("cohorts", [])
    entries = summary.get("cohorts", [])
    verified_rows = verification.get("cohorts", [])
    if len(configured) != 3 or len(entries) != 3 or len(verified_rows) != 3:
        raise ValueError("P006 snapshot requires exactly three configured, summarized, verified cohorts")
    ids = [str(row["id"]) for row in configured]
    if ids != [str(row.get("id")) for row in entries] or len(set(ids)) != 3:
        raise ValueError("P006 cohort IDs/order differ among config and parent summary")
    snapshot.mkdir(parents=True)

    top = ((config_path, snapshot / "configs/p006_final_test.json"),
           (manifest_path, snapshot / "configs/p006_freeze_manifest.json"),
           (protocol_path, snapshot / "configs/P006_PROTOCOL.md"),
           (run / "summary.json", snapshot / "summary.json"),
           (verification_path, snapshot / "verification_independent.json"),
           (analysis_path, snapshot / "analysis.json"))
    for source, dest in top:
        _copy(source, dest)
    for cohort in configured:
        membership_source = task_root / cohort["membership_metadata"]
        _copy(membership_source, snapshot / "memberships" / f"{cohort['id']}.json")
    for source_name, dest_name in (("metrics.csv", "metrics.csv"),
                                   ("paired_users.csv", "paired_users.csv"),
                                   ("paired_queries.csv", "paired_queries.csv")):
        source = run / source_name
        if source.is_file():
            _copy(source, snapshot / dest_name)
    for review_name in ("P006_REVIEW.md", "P006_RESULT_REVIEW.md"):
        review = task_root / "research" / review_name
        if review.is_file():
            _copy(review, snapshot / review_name)

    reservation = summary.get("reservation")
    if not isinstance(reservation, dict):
        raise ValueError("P006 parent summary lacks reservation marker")
    marker = task_root / "artifacts/p006_test_opened.json"
    if not marker.is_file() or _json(marker) != reservation:
        raise ValueError("P006 parent reservation marker differs from summary")
    _copy(marker, snapshot / "reservation_markers/p006_test_opened.json")
    child_records = []
    for cohort, entry, report in zip(configured, entries, verified_rows):
        cohort_id = str(cohort["id"])
        if report.get("id") != cohort_id or report.get("status") != entry.get("status"):
            raise ValueError(f"P006 parent verification does not cover cohort {cohort_id}")
        marker = run / f"{cohort_id}.test_opened.json"
        if marker.is_file():
            _copy(marker, snapshot / "reservation_markers" / marker.name)
        elif entry.get("status") == "complete":
            raise FileNotFoundError(f"completed P006 cohort has no reservation marker: {cohort_id}")
        child_report = {"id": cohort_id, "status": report.get("status"),
                        "final_test_evaluated": report.get("final_test_evaluated", False),
                        "query_protocol": report.get("query_protocol"), "models": report.get("models", []),
                        "partial_outputs_not_scored": report.get("partial_outputs_not_scored", True),
                        "verification_source": "parent verification_independent.json"}
        child_record = _snapshot_child(run, snapshot / cohort_id, entry, child_report)
        child_records.append(child_record)

    index = {"experiment": "P006", "parent_status": summary["status"], "final_test_evaluated": summary.get("final_test_evaluated"),
             "cohorts": child_records, "expected_models_per_complete_cohort": list(EXPECTED_LABELS),
             "rank_group_count": sum(len(row.get("rank_groups", [])) for row in child_records)}
    (snapshot / "snapshot_summary.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (snapshot / "README.md").write_text(
        "# P006 final-test compact snapshot\n\n"
        "Origin: the one-time parent run `artifacts/p006_final_test`, using the frozen config and source/input "
        "hash inventory in `configs/`. The global and per-cohort reservation markers are preserved under "
        "`reservation_markers/`.\n\n"
        "This snapshot preserves the one-time P006 reservation, frozen configuration and source manifest, "
        "parent/child verification evidence, frozen masks, compact result JSON, analysis, and per-query ranks.\n\n"
        "Ranks were derived from the verified test predictions using score descending and item ID ascending "
        "for ties. A blank target rank means the retrieval candidate set missed the target; those queries remain "
        "in all metric denominators. `validation` in query metadata is the evaluator-compatible split label; "
        "`evaluation_period=test` marks the reserved period.\n\n"
        "Raw histories, mappings, feature matrices, trained model files, and prediction NPZ archives are not copied. "
        "Their hashes remain in source/run status evidence. This is a compact audit copy after the single test "
        "evaluation, not a rerun or a new estimate. P001/P005 cohort results remain conditional on these fixed cohorts; "
        "the paired-user bootstrap is descriptive and does not estimate model/cohort uncertainty. Failed cohorts "
        "retain diagnostics and do not contribute score claims or rank rows.\n\n"
        "Verify this isolated copy with `python task_01_auto_features/src/verify_final_snapshot.py`.\n",
        encoding="utf-8")

    evidence = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            evidence[path.relative_to(snapshot).as_posix()] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (snapshot / "checkpoint_manifest.json").write_text(json.dumps(
        {"experiment": "P006", "tracked_evidence": evidence,
         "source_child_outputs_sha256": {row["id"]: _json(snapshot / row["id"] / "run_status.json").get("output_sha256", {})
                                          for row in child_records if row["status"] == "complete"}},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    result = snapshot_checkpoint(args.run, args.snapshot)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
