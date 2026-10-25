"""Portable verifier for a compact P006 final-test snapshot.

Only snapshot files are read. The verifier never opens source histories,
feature matrices, trained models, or prediction archives.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .verify_auto_checkpoint import _compare_metrics, recompute
else:
    from verify_auto_checkpoint import _compare_metrics, recompute

TASK = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = TASK / "report/p006_snapshot"
TOLERANCE = 1e-12
EXPECTED_LABELS = ("Popularity", "M_I", "M_UIDC", "Auto_A1", "Auto_A2", "Union")
PAIRED_LABELS = ("Auto_A1", "Auto_A2", "Union")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _safe_evidence_path(snapshot: Path, relative: str) -> Path:
    path = (snapshot / relative).resolve()
    if not path.is_relative_to(snapshot.resolve()):
        raise ValueError(f"manifest path escapes snapshot: {relative}")
    return path


def _bool(value: str, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{field} must be True or False, got {value!r}")
    return normalized == "true"


def _rank_metrics(rows: list[dict[str, str]], ks: tuple[int, ...]) -> dict[str, Any]:
    """Recompute metrics using ranks and candidate counts, without prediction files."""
    queries: list[dict[str, Any]] = []
    scores: list[list[float]] = []
    seen: set[int] = set()
    for row in rows:
        qid = int(row["query_id"])
        if qid in seen:
            raise ValueError(f"duplicate query ID {qid} in a model/cohort group")
        seen.add(qid)
        count = int(row["candidate_count"])
        if count < 0:
            raise ValueError("candidate_count must be non-negative")
        raw_rank = row.get("target_rank", "").strip()
        rank = int(raw_rank) if raw_rank else None
        if rank is not None and not 1 <= rank <= count:
            raise ValueError(f"one-based target rank {rank} is outside candidate_count={count}")
        synthetic_count = max(count, rank or 0)
        target = rank - 1 if rank is not None else synthetic_count + 1
        queries.append({"query_id": qid, "user_id": int(row["user_id"]),
                        "target_item": target, "candidate_items": list(range(synthetic_count)),
                        "target_is_repeat": _bool(row["target_is_repeat"], "target_is_repeat"),
                        "target_known_before": _bool(row["target_known_before"], "target_known_before")})
        scores.append([float(synthetic_count - i) for i in range(synthetic_count)])
    return recompute(queries, scores, ks=ks)


def _queries_from_rank_rows(rows: list[dict[str, str]], *, cohort: str,
                            start_qid: int = 0) -> tuple[list[dict[str, Any]], list[list[float]]]:
    queries, scores = [], []
    for offset, row in enumerate(rows):
        count = int(row["candidate_count"])
        raw_rank = row.get("target_rank", "").strip()
        rank = int(raw_rank) if raw_rank else None
        if rank is not None and not 1 <= rank <= count:
            raise ValueError(f"rank outside candidate_count for {cohort}/{row['query_id']}")
        n = max(count, rank or 0)
        queries.append({"query_id": start_qid + offset, "user_id": int(row["user_id"]),
                        "target_item": rank - 1 if rank is not None else n + 1,
                        "candidate_items": list(range(n)),
                        "target_is_repeat": _bool(row["target_is_repeat"], "target_is_repeat"),
                        "target_known_before": _bool(row["target_known_before"], "target_known_before")})
        scores.append([float(n - i) for i in range(n)])
    return queries, scores


def _validate_csv_groups(child: Path, expected_labels: set[str],
                         query_rows: list[dict[str, str]], child_id: str) -> dict[str, list[dict[str, str]]]:
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in query_rows:
        if row["model"] not in expected_labels or row["stage"] != "final_test":
            raise AssertionError(f"unexpected P006 rank group {row['model']}/{row['stage']} in {child_id}")
        key = (row["model"], int(row["query_id"]))
        if key in seen:
            raise AssertionError(f"duplicate model/query rank row: {key}")
        seen.add(key)
        by_model[row["model"]].append(row)
    if set(by_model) != expected_labels:
        raise AssertionError(f"rank model groups mismatch in {child_id}: {sorted(by_model)}")
    model_qids = {model: [int(row["query_id"]) for row in rows] for model, rows in by_model.items()}
    reference = model_qids["Popularity"]
    if any(qids != reference for qids in model_qids.values()):
        raise AssertionError(f"models do not cover identical query IDs in {child_id}")
    query_meta: dict[int, tuple[int, int, str, str]] = {}
    for row in by_model["Popularity"]:
        query_meta[int(row["query_id"])] = (int(row["user_id"]), int(row["candidate_count"]),
                                                row["target_is_repeat"], row["target_known_before"])
    for model, rows in by_model.items():
        for row in rows:
            qid = int(row["query_id"])
            if query_meta[qid] != (int(row["user_id"]), int(row["candidate_count"]),
                                   row["target_is_repeat"], row["target_known_before"]):
                raise AssertionError(f"query metadata differs across model ranks: {child_id}/{model}/{qid}")
    if len(query_rows) != len(expected_labels) * len(reference):
        raise AssertionError(f"expected six rank rows per query in {child_id}")
    return by_model


def _expected_bootstrap(deltas: list[float], config: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires nonempty finite deltas")
    resamples = int(config["resamples"])
    seed = int(config["seed"])
    confidence = float(config.get("confidence", .95))
    if resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha], method="linear")
    return {"users": len(values), "mean": float(values.mean()), "ci_lower": float(lower),
            "ci_upper": float(upper), "confidence": confidence, "resamples": resamples, "seed": seed,
            "quantile_method": "linear", "positive_users": int(np.sum(values > 0)),
            "negative_users": int(np.sum(values < 0)), "equal_users": int(np.sum(values == 0))}


def _compare_number(actual: Any, expected: Any, label: str) -> None:
    if actual is None or expected is None or not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=TOLERANCE):
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def verify_snapshot(snapshot: Path = DEFAULT_SNAPSHOT) -> dict[str, Any]:
    snapshot = Path(snapshot).resolve()
    manifest = _json(snapshot / "checkpoint_manifest.json")
    evidence = manifest.get("tracked_evidence")
    if manifest.get("experiment") != "P006" or not isinstance(evidence, dict) or not evidence:
        raise ValueError("invalid P006 checkpoint manifest")
    for relative, expected in evidence.items():
        path = _safe_evidence_path(snapshot, relative)
        if not path.is_file():
            raise FileNotFoundError(f"tracked snapshot evidence missing: {relative}")
        if path.stat().st_size != int(expected["bytes"]) or sha256(path) != expected["sha256"]:
            raise AssertionError(f"snapshot evidence SHA/size mismatch: {relative}")
    actual_evidence = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*")
                       if path.is_file() and path.name != "checkpoint_manifest.json"}
    if actual_evidence != set(evidence):
        raise AssertionError("checkpoint manifest does not cover every snapshot file")
    forbidden = {".parquet", ".npz", ".npy", ".cbm"}
    for path in snapshot.rglob("*"):
        if path.is_file() and (path.suffix.lower() in forbidden or path.name == "queries.jsonl"):
            raise AssertionError(f"large or query source artifact was copied into compact snapshot: {path.relative_to(snapshot)}")

    config = _json(snapshot / "configs/p006_final_test.json")
    source_manifest = _json(snapshot / "configs/p006_freeze_manifest.json")
    summary = _json(snapshot / "summary.json")
    index = _json(snapshot / "snapshot_summary.json")
    verification = _json(snapshot / "verification_independent.json")
    analysis = _json(snapshot / "analysis.json")
    if index.get("experiment") != "P006" or summary.get("status") not in {"complete", "complete_with_failures"}:
        raise AssertionError("snapshot does not describe a completed P006 parent")
    if summary.get("status") != index.get("parent_status") or analysis.get("status") != summary.get("status"):
        raise AssertionError("summary/index/analysis parent statuses differ")
    if not config.get("test_evaluation"):
        raise AssertionError("copied P006 configuration does not mark the reserved evaluation")
    if summary.get("final_test_evaluated") is not index.get("final_test_evaluated") or \
            analysis.get("final_test_evaluated") is not summary.get("final_test_evaluated") or \
            verification.get("final_test_evaluated") is not summary.get("final_test_evaluated"):
        raise AssertionError("P006 parent test-evaluation flags disagree")
    if verification.get("status") not in {"verified", "verified_with_failures"}:
        raise AssertionError("parent independent verification is not successful")
    cohorts = config.get("cohorts", [])
    summary_cohorts = summary.get("cohorts", [])
    index_cohorts = index.get("cohorts", [])
    verify_cohorts = verification.get("cohorts", [])
    if len(cohorts) != 3 or any(len(rows) != 3 for rows in (summary_cohorts, index_cohorts, verify_cohorts)):
        raise AssertionError("P006 snapshot must preserve exactly three cohorts across all ledgers")
    ids = [str(row["id"]) for row in cohorts]
    if len(set(ids)) != 3 or ids != [str(row.get("id")) for row in summary_cohorts] or \
            ids != [str(row.get("id")) for row in verify_cohorts] or ids != [str(row.get("id")) for row in index_cohorts]:
        raise AssertionError("cohort IDs/order differ across config, summary, verification, and index")
    if tuple(config.get("model_labels", [])) != EXPECTED_LABELS[1:]:
        raise AssertionError("frozen P006 model labels differ from expected five models")

    global_marker = _json(snapshot / "reservation_markers/p006_test_opened.json")
    if global_marker != summary.get("reservation"):
        raise AssertionError("copied global reservation marker differs from parent summary")
    config_sha = sha256(snapshot / "configs/p006_final_test.json")
    manifest_sha = sha256(snapshot / "configs/p006_freeze_manifest.json")
    protocol_sha = sha256(snapshot / "configs/P006_PROTOCOL.md")
    for key, digest in (("config_sha256", config_sha), ("manifest_sha256", manifest_sha),
                        ("protocol_sha256", protocol_sha)):
        if global_marker.get(key) != digest or summary.get(key) != digest:
            raise AssertionError(f"frozen P006 {key} differs from copied marker/configuration")
    if not isinstance(source_manifest.get("files"), dict) or not source_manifest["files"]:
        raise ValueError("frozen P006 source manifest is empty")
    source_outputs = manifest.get("source_child_outputs_sha256", {})
    if not isinstance(source_outputs, dict):
        raise ValueError("checkpoint manifest lacks source child output hash ledgers")

    membership_by_cohort: dict[str, set[int]] = {}
    all_members: set[int] = set()
    for cohort in cohorts:
        cohort_id = str(cohort["id"])
        member_doc = _json(snapshot / "memberships" / f"{cohort_id}.json")
        users = [int(uid) for uid in member_doc.get("selected_users", [])]
        if len(users) != 64 or len(set(users)) != 64 or all_members.intersection(users):
            raise AssertionError(f"P006 membership must contain 64 disjoint users: {cohort_id}")
        all_members.update(users)
        membership_by_cohort[cohort_id] = set(users)

    expected_labels = set(EXPECTED_LABELS)
    cohort_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    cohort_ranks: dict[str, dict[str, list[dict[str, str]]]] = {}
    cohort_queries: dict[str, list[dict[str, str]]] = {}
    failed = []
    group_count = 0
    for cohort, parent_row, verify_row, index_row in zip(cohorts, summary_cohorts, verify_cohorts, index_cohorts):
        cohort_id = str(cohort["id"])
        status_name = str(parent_row.get("status"))
        if status_name not in {"complete", "failed"} or status_name != index_row.get("status") or status_name != verify_row.get("status"):
            raise AssertionError(f"P006 cohort status differs among ledgers: {cohort_id}")
        marker_path = snapshot / "reservation_markers" / f"{cohort_id}.test_opened.json"
        if status_name == "complete" and not marker_path.is_file():
            raise FileNotFoundError(f"completed cohort marker missing: {cohort_id}")
        if marker_path.is_file():
            marker = _json(marker_path)
            if marker.get("cohort") != cohort_id or marker.get("manifest_sha256") != manifest_sha:
                raise AssertionError(f"cohort reservation marker mismatch: {cohort_id}")
            parent_output = str(global_marker.get("output", "")).replace("\\", "/").rstrip("/")
            child_output = str(marker.get("output", "")).replace("\\", "/").rstrip("/")
            if child_output != f"{parent_output}/{cohort_id}":
                raise AssertionError(f"cohort marker points outside the reserved P006 parent: {cohort_id}")
        child = snapshot / cohort_id
        child_verification = _json(child / "verification_independent.json")
        if child_verification.get("id") != cohort_id or child_verification.get("status") != status_name:
            raise AssertionError(f"child verification report mismatch: {cohort_id}")
        if child_verification != {"id": cohort_id, "status": verify_row.get("status"),
                                  "final_test_evaluated": verify_row.get("final_test_evaluated", False),
                                  "query_protocol": verify_row.get("query_protocol"),
                                  "models": verify_row.get("models", []),
                                  "partial_outputs_not_scored": verify_row.get("partial_outputs_not_scored", True),
                                  "verification_source": "parent verification_independent.json"}:
            raise AssertionError(f"child verification excerpt differs from parent report: {cohort_id}")

        status_path, metrics_path = child / "run_status.json", child / "metrics.json"
        if status_path.is_file() != metrics_path.is_file():
            raise AssertionError(f"child status and metrics must be preserved together: {cohort_id}")
        if status_path.is_file():
            status = _json(status_path)
            metrics_doc = _json(metrics_path)
            if status != metrics_doc or status.get("status") != status_name:
                raise AssertionError(f"child status/metrics mismatch: {cohort_id}")
            if not status.get("test_period_opened"):
                raise AssertionError(f"test-open flag missing in child status: {cohort_id}")
            if (status.get("config") != config or status.get("cohort") != cohort or
                    status.get("config_sha256") != config_sha or
                    status.get("protocol_sha256") != protocol_sha or
                    status.get("manifest_sha256") != manifest_sha or
                    status.get("source_and_input_sha256") != source_manifest.get("files")):
                raise AssertionError(f"frozen source/config ledger mismatch: {cohort_id}")
            if bool(status.get("final_test_evaluated")) != bool(index_row.get("final_test_evaluated")):
                raise AssertionError(f"child test-evaluated flag mismatch: {cohort_id}")
            if status_name == "complete" and source_outputs.get(cohort_id) != status.get("output_sha256", {}):
                raise AssertionError(f"omitted child output hashes differ from original status ledger: {cohort_id}")
        if status_name == "failed":
            if index_row.get("rank_rows") is not None or index_row.get("rank_groups"):
                raise AssertionError(f"failed cohort must have no compact ranks: {cohort_id}")
            if (child / "results").exists() or (child / "query_ranks.csv").exists():
                raise AssertionError(f"failed cohort snapshot contains score sidecars/ranks: {cohort_id}")
            failed.append(cohort_id)
            continue

        if not status_path.is_file():
            raise FileNotFoundError(f"complete child status missing: {cohort_id}")
        status = _json(status_path)
        if not status.get("final_test_evaluated") or not child_verification.get("final_test_evaluated"):
            raise AssertionError(f"complete cohort has not evaluated frozen test: {cohort_id}")
        if status.get("status") != "complete" or status.get("final_test_evaluated") is not True:
            raise AssertionError(f"invalid complete child state: {cohort_id}")
        masks_doc = _json(child / "masks_frozen.json")
        masks = masks_doc.get("final_masks")
        if not isinstance(masks, dict) or set(masks) != set(EXPECTED_LABELS[1:]):
            raise AssertionError(f"frozen P006 masks are incomplete: {cohort_id}")
        if masks_doc.get("test_scores_started") is not False:
            raise AssertionError(f"mask freeze does not record pre-score lock: {cohort_id}")
        if status.get("masks_frozen_sha256") != sha256(child / "masks_frozen.json"):
            raise AssertionError(f"frozen mask digest mismatch: {cohort_id}")
        development_selection = str(Path(cohort["development_run"]) / "selection_frozen.json").replace("\\", "/")
        if source_manifest["files"].get(development_selection) != masks_doc.get("development_selection_sha256"):
            raise AssertionError(f"development mask source hash differs from freeze manifest: {cohort_id}")
        specs = _json(child / "specs.json")
        if not isinstance(specs, list) or len(specs) != 206:
            raise AssertionError(f"P006 requires the frozen 206-expression spec list: {cohort_id}")
        a1_gate = _json(child / "a1_equivalence.json")
        rows_built = int(status.get("build", {}).get("shape", [0])[0])
        if (a1_gate.get("status") != "pass" or a1_gate.get("identical_float32_bytes") is not True or
                int(a1_gate.get("cells_compared", -1)) != rows_built * 66):
            raise AssertionError(f"A1/manual parity gate is incomplete: {cohort_id}")
        schema = _json(child / "feature_schema.json")
        if len(schema.get("feature_names", [])) != 206:
            raise AssertionError(f"P006 test feature schema is not 206 columns: {cohort_id}")
        baseline_schema = _json(child / "baseline/feature_schema.json")
        if len(baseline_schema.get("feature_names", [])) != 92:
            raise AssertionError(f"P006 baseline feature schema is not 92 columns: {cohort_id}")

        final_models = status.get("final_models", [])
        final_by_label = {str(row.get("label")): row for row in final_models}
        if len(final_models) != 6 or set(final_by_label) != expected_labels:
            raise AssertionError(f"completed child must have all six final model records: {cohort_id}")
        result_paths = sorted((child / "results").glob("*.json"))
        result_by_label = {str(_json(path).get("label")): _json(path) for path in result_paths}
        if len(result_paths) != 6 or set(result_by_label) != expected_labels:
            raise AssertionError(f"completed child must have six result JSONs: {cohort_id}")
        for label in EXPECTED_LABELS:
            result = result_by_label[label]
            if result != final_by_label[label]:
                raise AssertionError(f"{cohort_id}/{label} result sidecar differs from run status")
            if result.get("stage") != "final_test" or result.get("evaluation_period") != "test":
                raise AssertionError(f"{cohort_id}/{label} result lacks final-test flags")
            columns = [int(value) for value in result.get("columns", [])]
            if int(result.get("feature_count", -1)) != len(columns):
                raise AssertionError(f"{cohort_id}/{label} feature count differs from columns")
            if label == "Popularity":
                if columns or int(result["feature_count"]) != 0:
                    raise AssertionError("Popularity must be candidate popularity only")
            elif columns != [int(value) for value in masks[label]]:
                raise AssertionError(f"{cohort_id}/{label} result differs from frozen final mask")
            if result.get("reused_from"):
                source_label = str(result["reused_from"])
                if source_label not in final_by_label or source_label == label:
                    raise AssertionError(f"invalid alias target for {cohort_id}/{label}")
                source = final_by_label[source_label]
                for field in ("prediction_file", "prediction_sha256", "columns", "model_file", "model_sha256"):
                    if result.get(field) != source.get(field):
                        raise AssertionError(f"{cohort_id}/{label} alias differs from {source_label} in {field}")

        rank_rows = _csv(child / "query_ranks.csv")
        by_model = _validate_csv_groups(child, expected_labels, rank_rows, cohort_id)
        if (len(rank_rows) != int(index_row.get("rank_rows", -1)) or
                set(index_row.get("rank_groups", [])) != expected_labels):
            raise AssertionError(f"snapshot rank row/group count mismatch: {cohort_id}")
        if len(rank_rows) != len(by_model["Popularity"]) * len(EXPECTED_LABELS):
            raise AssertionError(f"rank rows do not cover all six models for every query: {cohort_id}")
        membership = membership_by_cohort[cohort_id]
        qids = [int(row["query_id"]) for row in by_model["Popularity"]]
        if qids != list(range(len(qids))):
            raise AssertionError(f"test query IDs must be sequential from zero: {cohort_id}")
        if any(int(row["user_id"]) not in membership for row in by_model["Popularity"]):
            raise AssertionError(f"query user outside frozen membership: {cohort_id}")
        if any(int(row["candidate_count"]) > int(config["max_candidates"]) for row in rank_rows):
            raise AssertionError(f"candidate count exceeds P006 frozen maximum: {cohort_id}")
        expected_metrics = {}
        for label, rows in by_model.items():
            actual = _rank_metrics(rows, tuple(map(int, config["metrics_k"])))
            expected = result_by_label[label].get("metrics", {})
            errors = _compare_metrics(expected, actual, TOLERANCE)
            if errors:
                raise AssertionError(f"{cohort_id}/{label} compact rank metrics mismatch: {'; '.join(errors[:8])}")
            expected_metrics[label] = actual
        cohort_metrics[cohort_id] = expected_metrics
        cohort_ranks[cohort_id] = by_model
        cohort_queries[cohort_id] = by_model["Popularity"]
        group_count += len(by_model)

    if group_count != len(EXPECTED_LABELS) * sum(row["status"] == "complete" for row in index_cohorts):
        raise AssertionError("P006 compact snapshot does not contain exactly six groups per complete cohort")
    if group_count != int(index.get("rank_group_count", -1)):
        raise AssertionError("snapshot summary rank_group_count differs from rank CSVs")
    if set(source_outputs) != set(cohort_metrics):
        raise AssertionError("source child output hash ledgers differ from complete cohorts")

    _verify_analysis(snapshot, analysis, config, cohort_metrics, cohort_ranks, cohort_queries,
                     [str(row["id"]) for row in cohorts if row["id"] in cohort_metrics])
    return {"status": "PASS", "tracked_files": len(evidence), "cohort_count": len(cohorts),
            "complete_cohorts": len(cohort_metrics), "failed_cohorts": failed,
            "ranking_groups_recomputed": group_count,
            "rank_rows": sum(len(rows) for group in cohort_ranks.values() for rows in group.values()),
            "pooled_metrics_recomputed": bool(analysis.get("pooled_metrics")),
            "primary_bootstrap_recomputed": bool(analysis.get("primary_paired_user_bootstrap")),
            "tolerance": TOLERANCE}


def _verify_analysis(snapshot: Path, analysis: dict[str, Any], config: dict[str, Any],
                     cohort_metrics: dict[str, dict[str, dict[str, Any]]],
                     cohort_ranks: dict[str, dict[str, list[dict[str, str]]]],
                     cohort_queries: dict[str, list[dict[str, str]]], complete_ids: list[str]) -> None:
    labels = list(EXPECTED_LABELS)
    pooled_queries = {label: [] for label in labels}
    pooled_scores = {label: [] for label in labels}
    expected_metric_rows = []
    expected_user_rows = []
    expected_query_rows = []
    user_delta_by_context: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pooled_user_deltas: dict[str, dict[int, float]] = defaultdict(dict)
    for cohort_id in complete_ids:
        queries = cohort_queries[cohort_id]
        per_model = cohort_ranks[cohort_id]
        per_query_metrics: dict[str, dict[int, tuple[int | None, float, float]]] = {}
        for label in labels:
            rows = per_model[label]
            result_metrics = cohort_metrics[cohort_id][label]
            per_query_metrics[label] = {}
            for row in rows:
                qid = int(row["query_id"])
                raw_rank = row.get("target_rank", "").strip()
                rank = int(raw_rank) if raw_rank else None
                hit = rank is not None and rank <= 20
                ndcg = 1.0 / math.log2(rank + 1) if hit else 0.0
                per_query_metrics[label][qid] = (rank, ndcg, float(hit))
            result_path = snapshot / cohort_id / "results" / f"{label}.json"
            result = _json(result_path)
            expected_metric_rows.append({"cohort": cohort_id, "model": label,
                                         "features": result["feature_count"],
                                         "users": result_metrics["overall"]["user_count"],
                                         "queries": result_metrics["overall"]["query_count"],
                                         "ndcg20": result_metrics["overall"]["user_macro"]["ndcg@20"],
                                         "recall20": result_metrics["overall"]["user_macro"]["recall@20"]})
            synthetic, synthetic_scores = _queries_from_rank_rows(rows, cohort=cohort_id)
            pooled_queries[label].extend(synthetic)
            pooled_scores[label].extend(synthetic_scores)
        for label in PAIRED_LABELS:
            by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for query in queries:
                qid = int(query["query_id"])
                base_rank, base_ndcg, base_recall = per_query_metrics["M_UIDC"][qid]
                rank, ndcg, recall = per_query_metrics[label][qid]
                row = {"cohort": cohort_id, "model": label, "query_id": qid,
                       "user_id": int(query["user_id"]), "manual_rank": base_rank if base_rank is not None else "",
                       "auto_rank": rank if rank is not None else "", "ndcg20_delta": ndcg - base_ndcg,
                       "recall20_delta": recall - base_recall}
                expected_query_rows.append(row)
                by_user[int(query["user_id"])].append(row)
            for user_id, rows in sorted(by_user.items()):
                ndcg_delta = math.fsum(float(row["ndcg20_delta"]) for row in rows) / len(rows)
                recall_delta = math.fsum(float(row["recall20_delta"]) for row in rows) / len(rows)
                expected_user_rows.append({"cohort": cohort_id, "model": label, "user_id": user_id,
                                           "queries": len(rows), "ndcg20_delta": ndcg_delta,
                                           "recall20_delta": recall_delta})
        # Primary comparison deltas are paired per user and average the user's queries first.
        base_by_user: dict[int, list[tuple[float, float]]] = defaultdict(list)
        auto_by_user: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for query in queries:
            qid, uid = int(query["query_id"]), int(query["user_id"])
            _, base_ndcg, base_recall = per_query_metrics["M_UIDC"][qid]
            _, auto_ndcg, auto_recall = per_query_metrics["Auto_A2"][qid]
            base_by_user[uid].append((base_ndcg, base_recall))
            auto_by_user[uid].append((auto_ndcg, auto_recall))
        for uid in sorted(set(base_by_user) & set(auto_by_user)):
            user_delta_by_context[cohort_id]["ndcg20"].append(
                math.fsum(x[0] for x in auto_by_user[uid]) / len(auto_by_user[uid]) -
                math.fsum(x[0] for x in base_by_user[uid]) / len(base_by_user[uid]))
            user_delta_by_context[cohort_id]["recall20"].append(
                math.fsum(x[1] for x in auto_by_user[uid]) / len(auto_by_user[uid]) -
                math.fsum(x[1] for x in base_by_user[uid]) / len(base_by_user[uid]))
            pooled_user_deltas["ndcg20"][uid] = user_delta_by_context[cohort_id]["ndcg20"][-1]
            pooled_user_deltas["recall20"][uid] = user_delta_by_context[cohort_id]["recall20"][-1]

    pooled = {}
    if expected_metric_rows:
        for label in labels:
            pooled_queries[label] = [dict(row, query_id=i) for i, row in enumerate(pooled_queries[label])]
            pooled[label] = recompute(pooled_queries[label], pooled_scores[label],
                                      ks=tuple(map(int, config["metrics_k"])))
            overall = pooled[label]["overall"]
            expected_metric_rows.append({"cohort": "pooled", "model": label, "features": "cohort-specific",
                                         "users": overall["user_count"], "queries": overall["query_count"],
                                         "ndcg20": overall["user_macro"]["ndcg@20"],
                                         "recall20": overall["user_macro"]["recall@20"]})
        user_delta_by_context["pooled"]["ndcg20"] = [pooled_user_deltas["ndcg20"][uid]
                                                       for uid in sorted(pooled_user_deltas["ndcg20"])]
        user_delta_by_context["pooled"]["recall20"] = [pooled_user_deltas["recall20"][uid]
                                                        for uid in sorted(pooled_user_deltas["recall20"])]
    stored_pooled = analysis.get("pooled_metrics", {})
    if set(stored_pooled) != set(pooled):
        raise AssertionError("analysis pooled metric labels differ from recomputed labels")
    for label, actual in pooled.items():
        errors = _compare_metrics(stored_pooled[label], actual, TOLERANCE)
        if errors:
            raise AssertionError(f"analysis pooled {label} metrics differ: {'; '.join(errors[:8])}")

    intervals = analysis.get("primary_paired_user_bootstrap", {})
    contexts = list(complete_ids) + (["pooled"] if pooled else [])
    if set(intervals) != set(contexts):
        raise AssertionError("primary bootstrap contexts differ from complete cohort set")
    bootstrap_config = config["bootstrap"]
    for context in contexts:
        if context not in intervals:
            raise AssertionError(f"missing primary bootstrap for {context}")
        for metric in ("ndcg20", "recall20"):
            expected = _expected_bootstrap(user_delta_by_context[context][metric], bootstrap_config)
            observed = intervals[context].get(metric, {})
            if set(observed) != set(expected):
                raise AssertionError(f"bootstrap fields differ for {context}/{metric}")
            for key, value in expected.items():
                if isinstance(value, float):
                    _compare_number(observed.get(key), value, f"bootstrap {context}/{metric}/{key}")
                elif observed.get(key) != value:
                    raise AssertionError(f"bootstrap {context}/{metric}/{key}: expected {value!r}, got {observed.get(key)!r}")

    expected_csv = {"metrics.csv": expected_metric_rows,
                    "paired_users.csv": expected_user_rows,
                    "paired_queries.csv": expected_query_rows}
    for filename, expected_rows in expected_csv.items():
        path = snapshot / filename
        if not expected_rows:
            if path.exists():
                raise AssertionError(f"unexpected empty analysis CSV: {filename}")
            continue
        if not path.is_file():
            raise FileNotFoundError(f"analysis CSV missing: {filename}")
        observed_rows = _csv(path)
        if len(observed_rows) != len(expected_rows):
            raise AssertionError(f"analysis CSV row count mismatch: {filename}")
        for row_number, (observed, expected) in enumerate(zip(observed_rows, expected_rows), 1):
            if set(observed) != set(expected):
                raise AssertionError(f"analysis CSV columns mismatch: {filename}")
            for field, value in expected.items():
                actual = observed[field]
                label = f"{filename} row {row_number} {field}"
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    _compare_number(actual, value, label)
                elif actual != str(value):
                    raise AssertionError(f"{label}: expected {value!r}, got {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    snapshot = args.snapshot if args.snapshot.is_absolute() else TASK / args.snapshot
    print(json.dumps(verify_snapshot(snapshot), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
