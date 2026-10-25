"""Independent verifier for the frozen P006 final-test checkpoint.

The verifier reads only the saved test queries, result JSON and prediction
vectors. It never rebuilds features or trains models. Source/history inputs are
checked by streaming SHA-256 against the precommitted freeze manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .verify_auto_checkpoint import _compare_metrics, _load_npz, recompute
except ImportError:  # direct script invocation
    from verify_auto_checkpoint import _compare_metrics, _load_npz, recompute

TASK = Path(__file__).resolve().parents[1]
DEFAULT_RUN = Path("artifacts/p006_final_test")
CONFIG_PATH = TASK / "configs/p006_final_test.json"
PROTOCOL_PATH = TASK / "configs/P006_PROTOCOL.md"
MANIFEST_PATH = TASK / "configs/p006_freeze_manifest.json"
LOCK_PATH = TASK / "artifacts/p006_test_opened.json"
TOLERANCE = 1e-12
EXPECTED_LABELS = ("Popularity", "M_I", "M_UIDC", "Auto_A1", "Auto_A2", "Union")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row {line_number} is not an object: {path}")
                rows.append(row)
    return rows


def validate_test_queries(queries: list[dict[str, Any]], members: Iterable[int],
                          start: int, per_user_limit: int = 10,
                          max_candidates: int = 100) -> dict[str, int]:
    """Validate causal protocol metadata without requiring target retrieval."""
    member_set = {int(uid) for uid in members}
    ids: set[int] = set()
    uid_times: set[tuple[int, int]] = set()
    counts: Counter[int] = Counter()
    previous: tuple[int, int, int] | None = None
    for row in queries:
        qid = int(row["query_id"])
        uid = int(row["user_id"])
        timestamp = int(row["timestamp"])
        target = int(row["target_item"])
        if qid in ids:
            raise ValueError("test query IDs are not unique")
        ids.add(qid)
        if uid not in member_set:
            raise ValueError(f"test query user {uid} is outside the frozen cohort")
        if timestamp < int(start):
            raise ValueError(f"test query {qid} precedes the frozen test boundary")
        if row.get("split") != "validation" or row.get("evaluation_period") != "test":
            raise ValueError(f"test query {qid} lacks legacy validation/test period labels")
        key = (uid, timestamp)
        if key in uid_times:
            raise ValueError("test queries contain duplicate UID/timestamp groups")
        uid_times.add(key)
        order = (timestamp, uid, target)
        if previous is not None and order < previous:
            raise ValueError("test queries are not in chronological deterministic order")
        previous = order
        counts[uid] += 1
        if counts[uid] > per_user_limit:
            raise ValueError(f"user {uid} exceeds the frozen per-user query limit")
        candidates = row.get("candidate_items")
        if not isinstance(candidates, list) or len(candidates) > max_candidates:
            raise ValueError(f"test query {qid} has an invalid candidate list")
        candidate_ids = [int(item) for item in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"test query {qid} contains duplicate candidates")
        scores = row.get("popularity_scores")
        if not isinstance(scores, list) or len(scores) != len(candidate_ids):
            raise ValueError(f"test query {qid} popularity score alignment is invalid")
        if not np.isfinite(np.asarray(scores, dtype=np.float64)).all():
            raise ValueError(f"test query {qid} has non-finite popularity scores")
    return {"query_count": len(queries), "sampled_users": len(counts),
            "cohort_members": len(member_set), "users_without_queries": len(member_set - set(counts)),
            "maximum_queries_per_user": max(counts.values(), default=0)}


def training_counts(queries: list[dict[str, Any]], test_start: int) -> dict[str, int | float]:
    """Reconstruct full development train+validation eligible candidate blocks."""
    requested = with_target = missing = candidate_rows_seen = candidate_rows_used = positives = 0
    allowed = {"train", "validation", "val"}
    for row in queries:
        if row.get("split") not in allowed:
            raise ValueError("development queries contain a non-development split")
        if int(row["timestamp"]) >= int(test_start):
            raise ValueError("development query reaches or crosses the frozen test boundary")
        requested += 1
        candidates = [int(item) for item in row["candidate_items"]]
        if len(candidates) != len(set(candidates)):
            raise ValueError("development query has duplicate candidates")
        candidate_rows_seen += len(candidates)
        matches = sum(item == int(row["target_item"]) for item in candidates)
        if matches == 0:
            missing += 1
            continue
        if matches != 1:
            raise ValueError("development target appears more than once in a candidate block")
        with_target += 1
        candidate_rows_used += len(candidates)
        positives += 1
    return {"queries_requested": requested, "queries_with_target": with_target,
            "queries_missing_target": missing,
            "target_coverage": with_target / requested if requested else 0.0,
            "candidate_rows_examined": candidate_rows_seen,
            "candidate_rows_used": candidate_rows_used, "positive_rows": positives}


def validate_frozen_masks(labels: Iterable[str], saved_masks: dict[str, Any],
                          dev_masks: dict[str, Any]) -> dict[str, list[int]]:
    normalized: dict[str, list[int]] = {}
    for label in labels:
        if label not in saved_masks or label not in dev_masks:
            raise ValueError(f"frozen development mask is missing {label}")
        columns = [int(value) for value in saved_masks[label]]
        if columns != [int(value) for value in dev_masks[label]]:
            raise AssertionError(f"{label} final mask differs from development-frozen selection")
        if label == "Popularity":
            if columns:
                raise AssertionError("Popularity must not have trained feature columns")
        elif not columns or len(columns) != len(set(columns)) or min(columns) < 0 or max(columns) >= 206:
            raise ValueError(f"{label} contains invalid frozen feature columns")
        normalized[label] = columns
    return normalized


def _compare_training(expected: dict[str, Any], actual: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        got = actual.get(key)
        if key == "target_coverage":
            if got is None or not math.isclose(float(got), float(value), rel_tol=0.0, abs_tol=TOLERANCE):
                raise AssertionError(f"{label} training.{key}: expected {value}, got {got}")
        elif got != value:
            raise AssertionError(f"{label} training.{key}: expected {value}, got {got}")


def _validate_source_manifest(manifest: dict[str, Any], run_statuses: list[dict[str, Any]]) -> dict[str, str]:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("freeze manifest must contain a non-empty files hash ledger")
    verified: dict[str, str] = {}
    for relative, expected in files.items():
        path = TASK / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"frozen source/input missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"frozen source/input SHA-256 mismatch: {relative}")
        verified[str(relative)] = actual
    manifest_sha = sha256(MANIFEST_PATH)
    for status in run_statuses:
        ledger = status.get("source_and_input_sha256")
        if ledger != files:
            raise AssertionError("child source/input hash ledger differs from precommitted manifest")
        if status.get("manifest_sha256") != manifest_sha:
            raise AssertionError("child manifest digest differs from current frozen manifest")
    return verified


def _required_manifest_paths(config: dict[str, Any]) -> set[str]:
    required = {"configs/p006_final_test.json", "configs/P006_PROTOCOL.md",
                "src/run_final_test.py", "src/final_test_data.py", "src/verify_final_test.py",
                "src/final_test_analysis.py"}
    for cohort in config["cohorts"]:
        required.update({str(cohort["history"]), str(cohort["development_queries"]),
                         str(cohort["membership_metadata"]),
                         str(Path(cohort["mapping_dir"]) / "artist_mapping.parquet"),
                         str(Path(cohort["mapping_dir"]) / "album_mapping.parquet"),
                         str(Path(cohort["development_run"]) / "features.npy"),
                         str(Path(cohort["development_run"]) / "selection_frozen.json"),
                         str(Path(cohort["development_run"]) / "metrics.json"),
                         str(Path(cohort["snapshot"]) / "metrics.json")})
    return {item.replace("\\", "/") for item in required}


def _validate_output_ledger(child: Path, status: dict[str, Any]) -> dict[str, str]:
    ledger = status.get("output_sha256")
    if not isinstance(ledger, dict):
        raise ValueError(f"missing child output hash ledger: {child}")
    actual_files = {str(path.relative_to(child)).replace("\\", "/")
                    for path in child.rglob("*") if path.is_file()
                    and path.name not in {"run_status.json", "metrics.json"}}
    ledger_names = {str(relative).replace("\\", "/") for relative in ledger}
    if len(ledger_names) != len(ledger):
        raise AssertionError(f"child output hash ledger has duplicate normalized paths: {child}")
    if actual_files != ledger_names:
        raise AssertionError(f"child output files differ from hash ledger: {child}")
    for relative, expected in ledger.items():
        path = child / Path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise AssertionError(f"child output SHA-256 mismatch: {relative}")
    return dict(ledger)


def _scores_for_result(child: Path, result: dict[str, Any], queries: list[dict[str, Any]]) -> list[np.ndarray]:
    relative = result.get("prediction_file")
    if not relative:
        raise ValueError(f"{result.get('label')} result has no prediction_file")
    prediction_path = Path(relative)
    if not prediction_path.is_absolute():
        prediction_path = child / prediction_path
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if result.get("prediction_sha256") != sha256(prediction_path):
        raise AssertionError(f"prediction hash mismatch for {result.get('label')}")
    query_ids = [int(row["query_id"]) for row in queries]
    saved_ids, offsets, flat = _load_npz(prediction_path)
    if saved_ids != query_ids:
        raise AssertionError(f"{result.get('label')} prediction query IDs differ from frozen test query order")
    scores = []
    for index, row in enumerate(queries):
        if offsets[index + 1] - offsets[index] != len(row["candidate_items"]):
            raise AssertionError(f"{result.get('label')} candidate offsets mismatch at query {query_ids[index]}")
        scores.append(flat[offsets[index]:offsets[index + 1]])
    return scores


def _per_user_contributions(queries: list[dict[str, Any]], scores: list[np.ndarray], k: int = 20) -> dict[int, dict[str, float]]:
    by_user: dict[int, list[dict[str, float]]] = defaultdict(list)
    for query, raw in zip(queries, scores):
        pairs = sorted(zip(np.asarray(raw, dtype=np.float64).tolist(), map(int, query["candidate_items"])),
                       key=lambda pair: (-pair[0], pair[1]))
        rank = next((i for i, (_score, item) in enumerate(pairs) if item == int(query["target_item"])), None)
        hit = rank is not None and rank < k
        by_user[int(query["user_id"])].append({"ndcg20": 1.0 / math.log2(rank + 2) if hit else 0.0,
                                                "recall20": float(hit)})
    return {uid: {metric: math.fsum(row[metric] for row in values) / len(values)
                  for metric in ("ndcg20", "recall20")}
            for uid, values in by_user.items()}


def _per_user_paired_deltas(queries: list[dict[str, Any]], auto_scores: list[np.ndarray],
                            manual_scores: list[np.ndarray], k: int = 20,
                            zero_tolerance: float = TOLERANCE) -> dict[int, dict[str, float]]:
    """Average per-query paired metric deltas, matching the published analysis order."""
    if not (len(queries) == len(auto_scores) == len(manual_scores)):
        raise ValueError("query and paired prediction lengths must match")
    values: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"ndcg20": [], "recall20": []})
    for query, auto_raw, manual_raw in zip(queries, auto_scores, manual_scores):
        def contribution(raw: np.ndarray) -> dict[str, float]:
            pairs = sorted(zip(np.asarray(raw, dtype=np.float64).tolist(), map(int, query["candidate_items"])),
                           key=lambda pair: (-pair[0], pair[1]))
            rank = next((i for i, (_score, item) in enumerate(pairs)
                         if item == int(query["target_item"])), None)
            hit = rank is not None and rank < k
            return {"ndcg20": 1.0 / math.log2(rank + 2) if hit else 0.0,
                    "recall20": float(hit)}
        auto, manual = contribution(auto_raw), contribution(manual_raw)
        uid = int(query["user_id"])
        for metric in ("ndcg20", "recall20"):
            values[uid][metric].append(auto[metric] - manual[metric])
    deltas = {}
    for uid, metrics in values.items():
        deltas[uid] = {}
        for metric, rows in metrics.items():
            mean_delta = float(np.mean(np.asarray(rows, dtype=np.float64)))
            deltas[uid][metric] = 0.0 if abs(mean_delta) <= zero_tolerance else mean_delta
    return deltas


def _paired_user_bootstrap(cohorts: list[tuple[str, dict[int, dict[str, float]]]],
                           resamples: int, seed: int, confidence: float) -> dict[str, Any]:
    """Reproduce the specified paired whole-user percentile bootstrap independently."""
    contexts: dict[str, list[dict[str, Any]]] = {}
    for name, delta_by_user in cohorts:
        rows = []
        for user in sorted(delta_by_user):
            rows.append({"user_id": user, **{f"{metric}_delta": delta_by_user[user][metric]
                                             for metric in ("ndcg20", "recall20")}})
        if rows:
            contexts[name] = rows
    all_users: dict[int, dict[str, Any]] = {}
    for _name, rows in contexts.items():
        for row in rows:
            if row["user_id"] in all_users:
                raise AssertionError("cohort user IDs overlap in pooled paired analysis")
            all_users[row["user_id"]] = row
    if all_users:
        contexts["pooled"] = [all_users[user] for user in sorted(all_users)]
    alpha = (1.0 - confidence) / 2.0
    output: dict[str, Any] = {}
    for context, rows in contexts.items():
        for metric in ("ndcg20", "recall20"):
            deltas = np.asarray([row[f"{metric}_delta"] for row in rows], dtype=np.float64)
            if not len(deltas) or not np.isfinite(deltas).all():
                raise ValueError("paired bootstrap requires finite non-empty user deltas")
            rng = np.random.default_rng(seed)
            draws = deltas[rng.integers(0, len(deltas), size=(resamples, len(deltas)))].mean(axis=1)
            lo, hi = np.quantile(draws, [alpha, 1.0 - alpha], method="linear")
            output.setdefault(context, {})[metric] = {
                "users": len(deltas), "mean": float(deltas.mean()), "ci_lower": float(lo),
                "ci_upper": float(hi), "confidence": confidence, "resamples": resamples,
                "seed": seed, "quantile_method": "linear",
                "positive_users": int(np.sum(deltas > 0)), "negative_users": int(np.sum(deltas < 0)),
                "equal_users": int(np.sum(deltas == 0))}
    return output


def _nested_close(expected: Any, actual: Any, tolerance: float = TOLERANCE, path: str = "root") -> None:
    """Compare JSON structures exactly except for float roundoff within tolerance."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise AssertionError(f"{path} keys differ")
        for key in expected:
            _nested_close(expected[key], actual[key], tolerance, f"{path}.{key}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise AssertionError(f"{path} list lengths differ")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _nested_close(left, right, tolerance, f"{path}[{index}]")
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            raise AssertionError(f"{path} differs: {expected!r} != {actual!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=tolerance):
            raise AssertionError(f"{path} differs beyond tolerance: {expected!r} != {actual!r}")
        return
    if expected != actual:
        raise AssertionError(f"{path} differs: {expected!r} != {actual!r}")


def verify_parent(run: str | Path, output: str | Path) -> dict[str, Any]:
    """Verify the P006 parent summary and every complete/partial child result.

    Failed cohorts remain failed in the report. Their incomplete matrices are
    never accepted as model scores; only fully materialized prediction sets
    listed in ``final_models`` are eligible for independent recomputation.
    """
    run = Path(run)
    if not run.is_absolute():
        run = TASK / run
    run = run.resolve()
    output = Path(output)
    if not output.is_absolute():
        output = TASK / output
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Verifier output must be new: {output}")

    config = _json(CONFIG_PATH)
    protocol_sha = sha256(PROTOCOL_PATH)
    config_sha = sha256(CONFIG_PATH)
    manifest = _json(MANIFEST_PATH)
    manifest_sha = sha256(MANIFEST_PATH)
    committed_manifest = subprocess.check_output(
        ["git", "show", "HEAD:task_01_auto_features/configs/p006_freeze_manifest.json"], cwd=TASK)
    if committed_manifest != MANIFEST_PATH.read_bytes():
        raise AssertionError("freeze manifest is not committed unchanged at the current HEAD")
    lock = _json(LOCK_PATH)
    summary = _json(run / "summary.json")
    reservation = summary.get("reservation")
    if not isinstance(reservation, dict) or reservation != lock:
        raise AssertionError("parent summary reservation differs from one-time test-opened marker")
    if Path(str(lock.get("output", ""))).resolve() != run:
        raise AssertionError("one-time marker output path differs from requested run")
    for key, actual in (("config_sha256", config_sha), ("protocol_sha256", protocol_sha),
                        ("manifest_sha256", manifest_sha)):
        if lock.get(key) != actual or summary.get(key) != actual:
            raise AssertionError(f"P006 reservation/summary {key} differs from frozen input")

    cohorts = config.get("cohorts", [])
    cohort_ids = [str(item["id"]) for item in cohorts]
    if len(cohort_ids) != 3 or len(set(cohort_ids)) != 3:
        raise ValueError("P006 config must define exactly three distinct cohorts")
    manifest_names = {str(name).replace("\\", "/") for name in manifest.get("files", {})}
    missing_pins = _required_manifest_paths(config) - manifest_names
    if missing_pins:
        raise AssertionError(f"freeze manifest omits required P006 source/input files: {sorted(missing_pins)}")
    summary_rows = summary.get("cohorts", [])
    if [str(row.get("id")) for row in summary_rows] != cohort_ids:
        raise AssertionError("parent summary cohort order/IDs differ from frozen config")

    memberships: dict[str, list[int]] = {}
    seen: set[int] = set()
    for cohort in cohorts:
        metadata = _json(TASK / cohort["membership_metadata"])
        users = [int(uid) for uid in metadata.get("selected_users", [])]
        if len(users) != 64 or len(set(users)) != 64 or seen.intersection(users):
            raise AssertionError(f"frozen cohort membership is not 64-way disjoint: {cohort['id']}")
        seen.update(users)
        memberships[str(cohort["id"])] = users

    child_statuses: list[dict[str, Any]] = []
    child_reports: list[dict[str, Any]] = []
    source_ledger: dict[str, str] | None = None
    pooled_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pooled_scores: dict[str, list[np.ndarray]] = defaultdict(list)
    bootstrap_inputs: list[tuple[str, dict[int, dict[str, float]]]] = []
    any_test_evaluated = False
    any_failure = False
    for config_cohort, parent_row in zip(cohorts, summary_rows):
        cohort_id = str(config_cohort["id"])
        if parent_row.get("run_dir") != cohort_id:
            raise AssertionError(f"parent run directory differs for {cohort_id}")
        child = run / cohort_id
        status_path = child / "run_status.json"
        metrics_path = child / "metrics.json"
        if not status_path.is_file() or not metrics_path.is_file():
            if parent_row.get("status") == "failed":
                any_failure = True
                child_reports.append({"id": cohort_id, "status": "failed", "verification": "no complete child ledger; no scores accepted"})
                continue
            raise FileNotFoundError(f"completed cohort lacks status/metrics: {cohort_id}")
        status = _json(status_path)
        metrics = _json(metrics_path)
        if status != metrics:
            raise AssertionError(f"{cohort_id} metrics.json differs from final run_status.json")
        child_statuses.append(status)
        if source_ledger is None:
            source_ledger = status.get("source_and_input_sha256")
        elif source_ledger != status.get("source_and_input_sha256"):
            raise AssertionError("cohorts do not share the same frozen source/input manifest")
        if status.get("config_sha256") != config_sha or status.get("protocol_sha256") != protocol_sha:
            raise AssertionError(f"{cohort_id} frozen config/protocol digest mismatch")
        if status.get("cohort", {}).get("id") != cohort_id or status.get("cohort") != config_cohort:
            raise AssertionError(f"{cohort_id} status cohort differs from P006 config")
        if status.get("config") != config:
            raise AssertionError(f"{cohort_id} child config differs from frozen P006 config")
        if not status.get("test_period_opened"):
            raise AssertionError(f"{cohort_id} does not record that the test period was opened")
        if status.get("status") not in {"complete", "failed"}:
            raise AssertionError(f"{cohort_id} has invalid terminal status")
        if parent_row.get("status") != status.get("status"):
            raise AssertionError(f"parent/child status mismatch for {cohort_id}")
        any_test_evaluated |= bool(status.get("final_test_evaluated"))
        if status.get("status") == "failed":
            any_failure = True

        output_hashes = _validate_output_ledger(child, status)
        queries: list[dict[str, Any]] = []
        queries_path = child / "queries.jsonl"
        if queries_path.is_file():
            queries = _jsonl(queries_path)
            query_check = validate_test_queries(queries, memberships[cohort_id], config["test_start_raw"],
                                                config["test_queries_per_user"], config["max_candidates"])
            candidate_ledger = status.get("candidates", {})
            if candidate_ledger.get("queries_sha256") != sha256(queries_path):
                raise AssertionError(f"{cohort_id} sealed query hash differs from run status")
            if candidate_ledger.get("candidate_rows") != sum(len(row["candidate_items"]) for row in queries):
                raise AssertionError(f"{cohort_id} candidate row count differs from sealed queries")
        else:
            query_check = {"query_count": 0, "status": "not_materialized"}
            if status.get("final_test_evaluated"):
                raise AssertionError(f"{cohort_id} marked evaluation before query output exists")

        expected_labels = set(EXPECTED_LABELS)
        finals = status.get("final_models", [])
        final_by_label = {str(entry.get("label")): entry for entry in finals}
        if len(final_by_label) != len(finals) or not set(final_by_label).issubset(expected_labels):
            raise AssertionError(f"{cohort_id} final model labels contain duplicates or unknown values")
        attempts = status.get("attempts", [])
        if len(attempts) > int(config["maximum_total_fits"]):
            raise AssertionError(f"{cohort_id} exceeded final fit budget")
        if any(attempt.get("stage") != "final_test_fit" for attempt in attempts):
            raise AssertionError(f"{cohort_id} contains an inner/development fit attempt")
        if any(attempt.get("stage") != "final_test" for attempt in finals):
            raise AssertionError(f"{cohort_id} final model has a non-final stage")
        if status.get("fits_actual", len(attempts)) != len(attempts):
            raise AssertionError(f"{cohort_id} fit count differs from attempts ledger")
        if not status.get("final_test_evaluated") and finals:
            raise AssertionError(f"{cohort_id} exposes model results before evaluation was marked")

        mask_path = child / "masks_frozen.json"
        dev_selection_path = TASK / config_cohort["development_run"] / "selection_frozen.json"
        masks: dict[str, list[int]] = {}
        if mask_path.is_file():
            masks_doc = _json(mask_path)
            dev_selection = _json(dev_selection_path)
            masks = validate_frozen_masks(config["model_labels"], masks_doc.get("final_masks", {}),
                                          dev_selection.get("final_masks", {}))
            if masks_doc.get("development_selection_sha256") != sha256(dev_selection_path):
                raise AssertionError(f"{cohort_id} frozen mask source digest mismatch")
            if status.get("masks_frozen_sha256") != sha256(mask_path):
                raise AssertionError(f"{cohort_id} run status mask digest mismatch")
        elif finals:
            raise AssertionError(f"{cohort_id} has results without a frozen mask record")

        dev_queries_path = TASK / config_cohort["development_queries"]
        dev_queries = _jsonl(dev_queries_path)
        expected_training = training_counts(dev_queries, config["test_start_raw"])
        attempted_labels = [str(attempt.get("label")) for attempt in attempts]
        if len(attempted_labels) != len(set(attempted_labels)):
            raise AssertionError(f"{cohort_id} attempt ledger contains duplicate actual fits")
        for attempt in attempts:
            label = str(attempt.get("label"))
            if label not in masks or attempt.get("columns") != masks[label]:
                raise AssertionError(f"{cohort_id}/{label} attempted columns differ from frozen mask")
            _compare_training(expected_training, attempt.get("training", {}), f"{cohort_id}/{label} attempt")
        result_checks = []
        per_label_scores: dict[str, list[np.ndarray]] = {}
        for label, result in final_by_label.items():
            if result.get("stage") != "final_test" or result.get("evaluation_period") != "test":
                raise AssertionError(f"{cohort_id}/{label} is not explicitly a final test result")
            if int(result.get("feature_count", -1)) != len(result.get("columns", [])):
                raise AssertionError(f"{cohort_id}/{label} feature_count/columns mismatch")
            if label == "Popularity":
                if result.get("columns") != [] or result.get("feature_count") != 0:
                    raise AssertionError("Popularity result must be retrieval-only")
            else:
                if result.get("columns") != masks.get(label):
                    raise AssertionError(f"{cohort_id}/{label} columns differ from frozen feature mask")
                _compare_training(expected_training, result.get("training", {}), f"{cohort_id}/{label}")
                model_file = result.get("model_file")
                if not result.get("reused_from"):
                    if not model_file:
                        raise AssertionError(f"{cohort_id}/{label} trained result lacks a saved model")
                    model_path = child / model_file
                    if not model_path.is_file() or sha256(model_path) != result.get("model_sha256"):
                        raise AssertionError(f"{cohort_id}/{label} model hash mismatch")
            if result.get("reused_from"):
                source_label = str(result["reused_from"])
                if source_label not in final_by_label or source_label == label:
                    raise AssertionError(f"{cohort_id}/{label} has invalid alias target")
                source = final_by_label[source_label]
                if result.get("prediction_file") != source.get("prediction_file"):
                    raise AssertionError(f"{cohort_id}/{label} alias prediction path differs from target")
                if result.get("columns") != source.get("columns"):
                    raise AssertionError(f"{cohort_id}/{label} aliases a different feature mask")
            if not queries:
                raise AssertionError(f"{cohort_id}/{label} results exist but test queries are absent")
            scores = _scores_for_result(child, result, queries)
            if label == "Popularity":
                for index, (score, query) in enumerate(zip(scores, queries)):
                    if not np.array_equal(np.asarray(score, dtype=np.float64),
                                          np.asarray(query["popularity_scores"], dtype=np.float64)):
                        raise AssertionError(f"{cohort_id}/Popularity predictions differ from sealed retrieval scores at query {index}")
            actual = recompute(queries, scores, ks=tuple(config["metrics_k"]))
            errors = _compare_metrics(result.get("metrics"), actual, TOLERANCE)
            if errors:
                raise AssertionError(f"{cohort_id}/{label} independently recomputed metrics differ: {'; '.join(errors[:8])}")
            per_label_scores[label] = scores
            if status.get("status") == "complete":
                pooled_metrics[label].extend(queries)
                pooled_scores[label].extend(scores)
            result_checks.append({"label": label, "stage": result["stage"], "evaluation_period": "test",
                                 "queries": len(queries), "independent_metrics": "match",
                                 "prediction_sha256": result.get("prediction_sha256"),
                                 "alias_of": result.get("reused_from")})
        if status.get("status") == "complete" and set(final_by_label) != expected_labels:
            raise AssertionError(f"{cohort_id} complete child does not contain all six model results")
        if status.get("status") == "complete" and len(queries) == 0:
            raise AssertionError(f"{cohort_id} complete child has no test queries")
        if status.get("status") == "complete" and "Auto_A2" in per_label_scores and "M_UIDC" in per_label_scores:
            bootstrap_inputs.append((cohort_id, _per_user_paired_deltas(
                queries, per_label_scores["Auto_A2"], per_label_scores["M_UIDC"])))
        child_reports.append({"id": cohort_id, "status": status["status"],
                              "final_test_evaluated": bool(status.get("final_test_evaluated")),
                              "query_protocol": query_check, "verified_output_sha256": output_hashes,
                              "models": result_checks,
                              "partial_outputs_not_scored": status["status"] == "failed"})

    if bool(summary.get("final_test_evaluated")) != any_test_evaluated:
        raise AssertionError("parent final_test_evaluated flag differs from child states")
    if summary.get("status") not in {"complete", "complete_with_failures"}:
        raise AssertionError("P006 parent has not reached a terminal complete/complete_with_failures state")
    if (summary.get("status") == "complete_with_failures") != any_failure:
        raise AssertionError("parent complete/failure status does not preserve child failure states")
    verified_sources = _validate_source_manifest(manifest, child_statuses)
    bootstrap = _paired_user_bootstrap(bootstrap_inputs, int(config["bootstrap"]["resamples"]),
                                       int(config["bootstrap"]["seed"]), float(config["bootstrap"]["confidence"]))
    pooled_ranking = {label: recompute(rows, pooled_scores[label], ks=tuple(config["metrics_k"]))
                      for label, rows in pooled_metrics.items()}
    analysis_path = run / "analysis.json"
    analysis_crosscheck: dict[str, Any] = {"status": "not_available"}
    if analysis_path.is_file():
        analysis = _json(analysis_path)
        analysis_pooled = analysis.get("pooled_metrics", {})
        if set(analysis_pooled) != set(pooled_ranking):
            raise AssertionError("analysis pooled model labels differ from independently scored complete cohorts")
        for label, actual in pooled_ranking.items():
            errors = _compare_metrics(analysis_pooled[label], actual, TOLERANCE)
            if errors:
                raise AssertionError(f"pooled {label} metrics differ from independent recomputation: {'; '.join(errors[:8])}")
        _nested_close(analysis.get("primary_paired_user_bootstrap"), bootstrap,
                      TOLERANCE, "analysis.primary_paired_user_bootstrap")
        analysis_crosscheck = {"status": "match", "pooled_models": sorted(pooled_ranking),
                               "bootstrap_contexts": sorted(bootstrap)}
    elif summary.get("status") == "complete":
        raise FileNotFoundError("analysis.json must be generated before independent final-test verification")
    report = {"status": "verified_with_failures" if any_failure else "verified",
              "run": str(run), "config_sha256": config_sha, "protocol_sha256": protocol_sha,
              "manifest_sha256": manifest_sha, "verified_source_and_input_sha256": verified_sources,
              "cohort_count": len(cohort_ids), "disjoint_membership_users": len(seen),
              "final_test_evaluated": any_test_evaluated, "cohorts": child_reports,
              "pooled_user_macro_metrics": pooled_ranking,
              "primary_paired_user_bootstrap": bootstrap,
              "analysis_crosscheck": analysis_crosscheck,
              "independent_metric_implementation": "verify_auto_checkpoint.recompute; no feature building or model training",
              "tolerance": TOLERANCE}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else TASK / args.run
    output = args.output or run / "verification_independent.json"
    result = verify_parent(run, output)
    print(f"Verified {result['cohort_count']} P006 cohorts; status={result['status']}; report={output}")


if __name__ == "__main__":
    main()
