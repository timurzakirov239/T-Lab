"""Create compact, hash-manifested P005 evidence after child verification.

Large source tables, feature matrices, candidate matrices, models, and prediction
archives are intentionally excluded. Their source hashes remain in run metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

TASK = Path(__file__).resolve().parents[1]
DEFAULT_RUN = TASK / "artifacts/p005_robustness"
DEFAULT_SNAPSHOT = TASK / "report/p005_snapshot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verified(report: dict) -> bool:
    status = str(report.get("status", report.get("verification", ""))).lower()
    return status in {"verified", "pass", "passed", "complete"} or report.get("overall_status") == "PASS"


def _npz_predictions(path: Path) -> tuple[list[int], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        qkey = next((x for x in ("query_ids", "query_id") if x in data), None)
        okey = next((x for x in ("offsets", "candidate_offsets") if x in data), None)
        skey = next((x for x in ("scores", "predictions", "probabilities") if x in data), None)
        if not qkey or not okey or not skey:
            raise ValueError(f"Prediction archive lacks query_ids/offsets/scores: {path}")
        qids = np.asarray(data[qkey], dtype=np.int64)
        offsets = np.asarray(data[okey], dtype=np.int64)
        scores = np.asarray(data[skey], dtype=np.float64)
    if offsets.ndim != 1 or len(offsets) != len(qids) + 1 or offsets[0] != 0 or offsets[-1] != len(scores):
        raise ValueError(f"Invalid prediction archive offsets: {path}")
    return qids.tolist(), offsets, scores


def _target_rank(query: dict, scores: np.ndarray) -> int | None:
    candidates = query["candidate_items"]
    if len(candidates) != len(scores):
        raise ValueError(f"Prediction/candidate length mismatch: query={query.get('query_id')}")
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), int(candidates[i])))
    try:
        return order.index(candidates.index(query["target_item"])) + 1
    except ValueError:
        return None


def _copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def _snapshot_child(run: Path, snapshot_child: Path, cohort_entry: dict) -> dict:
    child_name = str(cohort_entry["run_dir"])
    child = run / child_name
    metrics_path = child / "metrics.json"
    status_path = child / "run_status.json"
    if not metrics_path.exists() or not status_path.exists():
        raise FileNotFoundError(f"Missing status/metrics for {child_name}")
    metrics, status = _read_json(metrics_path), _read_json(status_path)
    snapshot_child.mkdir(parents=True)
    compact = ["metrics.json", "run_status.json"]
    if metrics.get("status") == "complete" and status.get("status") == "complete":
        verification_path = next((child / name for name in ("verification_independent.json", "verification.json") if (child / name).exists()), None)
        if verification_path is None:
            raise ValueError(f"Completed cohort {child_name} has no independent verification evidence")
        verification = _read_json(verification_path)
        if not _verified(verification):
            raise ValueError(f"Completed cohort {child_name} is not independently verified")
        compact += ["specs.json", "split.json", "pruning.json", "selection_frozen.json", "inner_results.json",
                    "a1_equivalence.json", "verification_independent.json"]
        # The candidate feature schema is a compact, 206-column definition needed
        # to interpret the frozen outputs; baseline schema remains for A1 parity.
        compact.append("feature_schema.json")
        for name in compact:
            if name not in ("verification_independent.json",) and not (child / name).is_file():
                raise FileNotFoundError(child / name)
        for name in compact:
            src = verification_path if name == "verification_independent.json" else child / name
            _copy_file(src, snapshot_child / name)
        # Keep the baseline schema used for A1/manual equivalence, and data metadata only.
        for relative in (Path("baseline/feature_schema.json"), Path("data/metadata.json")):
            if (child / relative).is_file(): _copy_file(child / relative, snapshot_child / relative)
        results_src = child / "results"
        for result_path in sorted(results_src.glob("*.json")):
            _copy_file(result_path, snapshot_child / "results" / result_path.name)
        queries = [json.loads(line) for line in (child / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {int(q["query_id"]): q for q in queries}
        results = []
        inner = _read_json(child / "inner_results.json")
        results.extend(inner)
        results.extend(metrics.get("final_models", status.get("final_models", [])))
        rank_path = snapshot_child / "query_ranks.csv"
        fields = ["model", "stage", "query_id", "user_id", "split", "target_is_repeat", "target_known_before", "target_rank"]
        with rank_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for result in results:
                label = result.get("label")
                prediction_file = result.get("prediction_file")
                if not label or not prediction_file: continue
                pred_path = child / prediction_file
                if not pred_path.exists(): raise FileNotFoundError(pred_path)
                qids, offsets, scores = _npz_predictions(pred_path)
                for i, qid in enumerate(qids):
                    query = by_id[int(qid)]
                    rank = _target_rank(query, scores[offsets[i]:offsets[i + 1]])
                    writer.writerow({"model": label, "stage": result.get("stage", "final"), "query_id": int(qid),
                                     "user_id": int(query["user_id"]), "split": query["split"],
                                     "target_is_repeat": bool(query["target_is_repeat"]),
                                     "target_known_before": bool(query["target_known_before"]), "target_rank": rank or ""})
        return {"seed": metrics.get("query_sampling_seed", status.get("query_sampling_seed", cohort_entry.get("seed"))),
                "run_dir": child_name, "status": "complete", "prediction_outputs_ranked": len(results),
                "rank_rows": sum(1 for _ in rank_path.open(encoding="utf-8")) - 1,
                "source_run_output_sha256": status.get("output_sha256", {})}

    # Failed/partial child: preserve status and available diagnostics only; never infer scores.
    for name in compact:
        if (child / name).is_file(): _copy_file(child / name, snapshot_child / name)
    verification_path = next((child / name for name in ("verification_independent.json", "verification.json")
                              if (child / name).is_file()), None)
    if verification_path is None:
        raise ValueError(f"Failed cohort {child_name} has no independent verification evidence")
    verification = _read_json(verification_path)
    if verification.get("status") != "failed_checkpoint_verified":
        raise ValueError(f"Failed cohort {child_name} lacks a verified failure report")
    _copy_file(verification_path, snapshot_child / "verification_independent.json")
    for name in ("specs.json", "split.json", "pruning.json", "selection_frozen.json", "inner_results.json", "a1_equivalence.json"):
        if (child / name).is_file(): _copy_file(child / name, snapshot_child / name)
    for relative in (Path("data/metadata.json"), Path("data/FAILED.json")):
        if (child / relative).is_file(): _copy_file(child / relative, snapshot_child / relative)
    log = run / f"{child_name}.log"
    if log.is_file(): _copy_file(log, snapshot_child / log.name)
    failed = child / "FAILED.json"
    if failed.is_file(): _copy_file(failed, snapshot_child / "FAILED.json")
    for result_path in sorted((child / "results").glob("*.json")) if (child / "results").exists() else []:
        _copy_file(result_path, snapshot_child / "results" / result_path.name)
    return {"seed": cohort_entry.get("seed"), "run_dir": child_name, "status": "failed",
            "rank_reconstruction": "skipped for failed child", "source_run_output_sha256": status.get("output_sha256", {})}


def snapshot_checkpoint(run: Path = DEFAULT_RUN, snapshot: Path = DEFAULT_SNAPSHOT) -> dict:
    run, snapshot = Path(run), Path(snapshot)
    if snapshot.exists(): raise FileExistsError(f"Snapshot path already exists: {snapshot}")
    summary = _read_json(run / "summary.json")
    memberships = _read_json(run / "membership" / "memberships.json")
    if summary.get("status") not in {"complete", "complete_with_failures"}:
        raise ValueError("P005 parent must be complete or complete_with_failures before snapshot")
    top_files = [(run / "summary.json", snapshot / "summary.json"),
                 (run / "membership" / "memberships.json", snapshot / "membership" / "memberships.json"),
                 (TASK / "configs/p005_robustness.json", snapshot / "p005_robustness.json"),
                 (TASK / "configs/P005_PROTOCOL.md", snapshot / "P005_PROTOCOL.md")]
    for review in (TASK / "research/P005_REVIEW.md", TASK / "research/P005_RESULT_REVIEW.md"):
        if review.is_file(): top_files.append((review, snapshot / review.name))
    parent_verification = run / "verification_independent.json"
    if not parent_verification.is_file():
        raise FileNotFoundError("P005 parent must have verification_independent.json before snapshot")
    parent_report = _read_json(parent_verification)
    if not _verified(parent_report) or parent_report.get("status") != "verified":
        raise ValueError("P005 parent independent verification must have status 'verified'")
    cohorts = summary.get("cohorts", [])
    verified_cohorts = parent_report.get("cohorts", [])
    if len(verified_cohorts) != len(cohorts):
        raise ValueError("P005 parent verification cohort count differs from summary")
    expected_vstates = {"complete": "verified", "failed": "failed_checkpoint_verified"}
    for entry, report_entry in zip(cohorts, verified_cohorts):
        if (int(entry["seed"]) != int(report_entry.get("seed", -1)) or
                expected_vstates.get(entry.get("status")) != report_entry.get("status")):
            raise ValueError(f"P005 parent verification does not cover cohort {entry.get('run_dir')}")
    top_files.append((parent_verification, snapshot / "verification_independent.json"))
    snapshot.mkdir(parents=True)
    for src, dest in top_files: _copy_file(src, dest)
    child_records = []
    for entry in summary.get("cohorts", []):
        child_records.append(_snapshot_child(run, snapshot / entry["run_dir"], entry))
    index = {"experiment": "P005", "parent_status": summary["status"], "cohorts": child_records,
             "source_data_sha256": summary.get("source_data_sha256", {}),
             "config_sha256": summary.get("config_sha256"), "protocol_sha256": summary.get("protocol_sha256"),
             "final_test_evaluated": summary.get("final_test_evaluated", False)}
    (snapshot / "snapshot_summary.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    included_reviews = [name for name in ("P005_REVIEW.md", "P005_RESULT_REVIEW.md") if (snapshot / name).is_file()]
    review_note = ", ".join(f"[{name}]({name})" for name in included_reviews) if included_reviews else "reviews отсутствуют"
    (snapshot / "README.md").write_text(
        "# Компактная контрольная точка P005\n\n"
        "Снимок содержит замороженные конфигурацию, membership, родительскую и дочерние независимые проверки, "
        "метрики и компактные per-query ranks для завершённых когорт.\n"
        "Неуспешные когорты сохранены вместе с доступной диагностикой; их ranking checks пропускаются.\n\n"
        f"Reviews: {review_note}.\n\n"
        "Исходные таблицы, матрицы признаков, модели и архивы predictions не копируются. Их хеши и статусы "
        "остаются в сохранённых manifests/status evidence. Snapshot verifier работает только по компактным rank CSV.\n\n"
        "Проверка: `python task_01_auto_features/src/verify_robustness_snapshot.py`\n",
        encoding="utf-8")
    evidence = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_file(): evidence[path.relative_to(snapshot).as_posix()] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {"experiment": "P005", "tracked_evidence": evidence,
                "source_run_outputs_sha256": {item["run_dir"]: item["source_run_output_sha256"] for item in child_records}}
    (snapshot / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    result = snapshot_checkpoint(args.run, args.snapshot)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
