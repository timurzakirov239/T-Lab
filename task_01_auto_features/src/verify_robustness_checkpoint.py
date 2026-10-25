"""Independent integrity and ranking verifier for P005 cohort checkpoints.

The verifier consumes frozen cohort queries, split/manifests, result JSON and
saved NPZ predictions. It never loads event tables, rebuilds features, or trains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__:
    from .verify_auto_checkpoint import (
        _compare_metrics,
        _load_json,
        _load_npz,
        _prediction_scores,
        recompute,
        sha256,
    )
else:
    from verify_auto_checkpoint import (
        _compare_metrics,
        _load_json,
        _load_npz,
        _prediction_scores,
        recompute,
        sha256,
    )


TASK = Path(__file__).resolve().parents[1]
DEFAULT_RUN = Path("artifacts/p005_robustness")
TOLERANCE = 1e-12


def _read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _task_path(task_root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else task_root / path


def _hash_ledger(ledger: Mapping[str, str], task_root: Path, label: str) -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative, expected in ledger.items():
        path = _task_path(task_root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"{label} file missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"{label} hash mismatch: {relative}")
        verified[str(relative)] = actual
    return verified


def _load_specs(run: Path) -> list[dict[str, Any]]:
    value = _read_json(run / "specs.json")
    if not isinstance(value, list):
        raise ValueError("specs.json must contain a list")
    ids = [str(item.get("feature_id", item.get("id", ""))) for item in value]
    if len(ids) != len(set(ids)) or any(not feature_id for feature_id in ids):
        raise ValueError("specs.json feature IDs must be nonempty and unique")
    return value


def _available_for_family(pruning: Mapping[str, Any], family: str) -> list[int]:
    record = pruning.get(family)
    if not isinstance(record, Mapping) or "available" not in record:
        raise ValueError(f"pruning.json lacks available columns for {family}")
    columns = [int(value) for value in record["available"]]
    if columns != sorted(set(columns)):
        raise ValueError(f"pruning columns for {family} are not unique canonical order")
    return columns


def _winner_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    score = float(row.get("inner_score", row.get("score", float("nan"))))
    if not math.isfinite(score):
        raise ValueError("inner score must be finite")
    ids = row.get("spec_ids", row.get("feature_ids", []))
    label = str(row.get("label", ""))
    return (-score, len(row.get("columns", [])), tuple(sorted(map(str, ids))), label)


def pick_inner_winner(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Pick a frozen winner using only inner score, width, feature IDs, label."""
    if not rows:
        raise ValueError("cannot pick a winner from an empty family")
    return min(rows, key=_winner_key)


def validate_frozen_winner(recorded: Mapping[str, Any], actual: Mapping[str, Any], *, tol: float = TOLERANCE) -> None:
    """Raise when a frozen family winner differs from reconstructed inner results."""
    if list(recorded.get("columns", [])) != list(actual.get("columns", [])):
        raise AssertionError("frozen winner columns differ from reconstructed inner winner")
    recorded_ids = list(recorded.get("spec_ids", recorded.get("feature_ids", [])))
    actual_ids = list(actual.get("spec_ids", actual.get("feature_ids", [])))
    if recorded_ids != actual_ids:
        raise AssertionError("frozen winner feature IDs differ from reconstructed inner winner")
    left = float(recorded.get("inner_score", recorded.get("score", float("nan"))))
    right = float(actual.get("inner_score", actual.get("score", float("nan"))))
    if not (math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=0.0, abs_tol=tol)):
        raise AssertionError("frozen winner score differs from reconstructed inner winner")
    if recorded.get("label") and actual.get("label") and recorded["label"] != actual["label"]:
        raise AssertionError("frozen winner label differs from reconstructed inner winner")


def _position_ids(positions: Sequence[int], queries: Sequence[Mapping[str, Any]], label: str) -> list[int]:
    output = []
    for position in positions:
        index = int(position)
        if index < 0 or index >= len(queries):
            raise IndexError(f"{label} query position outside local query JSONL")
        output.append(int(queries[index]["query_id"]))
    if len(output) != len(set(output)):
        raise ValueError(f"{label} positions contain duplicate query IDs")
    return output


def _validate_split(queries: Sequence[Mapping[str, Any]], split: Mapping[str, Any]) -> dict[str, Any]:
    """Validate variable-size local train/validation queries and temporal split."""
    query_ids = [int(query["query_id"]) for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("cohort queries contain duplicate query_id")
    if query_ids != list(range(len(queries))):
        raise ValueError("cohort query IDs must be contiguous in local JSONL order")
    seen_user_times: set[tuple[int, int]] = set()
    for query in queries:
        if "candidate_items" not in query:
            raise ValueError("each cohort query must have candidate_items")
        candidates = [int(item) for item in query["candidate_items"]]
        if len(candidates) != len(set(candidates)):
            raise ValueError(f"query {query['query_id']} has duplicate candidate IDs")
        key = (int(query["user_id"]), int(query["timestamp"]))
        if key in seen_user_times:
            raise ValueError("more than one query for a user/timestamp group")
        seen_user_times.add(key)
    fit = _position_ids(split.get("inner_fit", []), queries, "inner_fit")
    selection = _position_ids(split.get("inner_selection", []), queries, "inner_selection")
    external = _position_ids(split.get("external_validation", []), queries, "external_validation")
    if not fit or not selection or not external:
        raise ValueError("each split partition must be nonempty")
    parts = [set(fit), set(selection), set(external)]
    if any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("query partitions overlap")
    if set.union(*parts) != set(query_ids):
        raise ValueError("query partitions do not cover local query JSONL")
    by_id = {int(query["query_id"]): query for query in queries}
    for query_id in fit + selection:
        if by_id[query_id].get("split") != "train":
            raise ValueError("inner query position does not reference train split")
    for query_id in external:
        if by_id[query_id].get("split") not in ("validation", "val"):
            raise ValueError("external query position does not reference validation split")
    cutoff = int(split["cutoff"])
    fit_times = [int(by_id[qid]["timestamp"]) for qid in fit]
    selection_times = [int(by_id[qid]["timestamp"]) for qid in selection]
    external_times = [int(by_id[qid]["timestamp"]) for qid in external]
    if max(fit_times) >= cutoff or min(selection_times) < cutoff:
        raise ValueError("inner timestamp boundary does not separate fit and selection")
    if max(fit_times) >= min(selection_times):
        raise ValueError("inner fit timestamps must precede inner selection")
    if max(selection_times) >= min(external_times):
        raise ValueError("inner selection timestamps must precede external validation")
    if set(fit_times) & set(selection_times):
        raise ValueError("equal timestamps were split across fit and inner selection")
    return {"inner_fit": fit, "inner_selection": selection, "external_validation": external,
            "cutoff": cutoff, "query_counts": {"inner_fit": len(fit), "inner_selection": len(selection),
                                                "external_validation": len(external)}}


def _validate_membership(run: Path, task_root: Path, status: Mapping[str, Any], summary: Mapping[str, Any] | None) -> dict[str, Any]:
    parent = run.parent
    membership_path = parent / "membership" / "memberships.json"
    if not membership_path.is_file():
        raise FileNotFoundError(membership_path)
    membership = _read_json(membership_path)
    seed = int(status.get("query_sampling_seed", status.get("config", {}).get("seed", -1)))
    cohort = next((row for row in membership.get("cohorts", []) if int(row["seed"]) == seed), None)
    if cohort is None:
        raise AssertionError(f"membership file lacks cohort seed {seed}")
    selected = [int(x) for x in cohort.get("selected_users", [])]
    if len(selected) != 64 or len(selected) != len(set(selected)):
        raise AssertionError("P005 membership must contain 64 unique selected users")
    stored_hash = cohort.get("selected_users_sha256")
    digest = hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode("ascii")).hexdigest()
    if stored_hash and stored_hash != digest:
        raise AssertionError("membership selected-user list hash mismatch")
    excluded = set(map(int, membership.get("excluded_users", [])))
    if excluded.intersection(selected):
        raise AssertionError("cohort membership intersects excluded P001/P004 users")
    other_members = [set(map(int, row.get("selected_users", []))) for row in membership.get("cohorts", [])
                     if int(row["seed"]) != seed]
    if any(set(selected) & other for other in other_members):
        raise AssertionError("P005 cohort memberships are not disjoint")
    metadata_path = run / "data" / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = _read_json(metadata_path)
    if metadata.get("status") != "complete" or [int(x) for x in metadata.get("selected_users", [])] != selected:
        raise AssertionError("child data metadata selected users differ from frozen membership")
    if int(metadata.get("seed", -1)) != seed:
        raise AssertionError("child data metadata seed differs from frozen membership")
    if summary:
        expected_membership_hash = summary.get("memberships_sha256")
        if expected_membership_hash and sha256(membership_path) != expected_membership_hash:
            raise AssertionError("membership SHA differs from parent summary")
    return {"seed": seed, "selected_users": len(selected), "selected_users_sha256": digest,
            "excluded_users": len(excluded), "disjoint_other_cohorts": True,
            "metadata_file": str(metadata_path)}


def _validate_attempt_budget(status: Mapping[str, Any], config: Mapping[str, Any], *, complete: bool) -> dict[str, int]:
    attempts = status.get("attempts", [])
    inner = [row for row in attempts if row.get("stage") == "inner"]
    final = [row for row in attempts if row.get("stage") == "final"]
    max_total = int(config.get("maximum_total_fits", 28))
    max_family = int(config.get("maximum_inner_fits_per_family", 6))
    if len(attempts) > max_total:
        raise AssertionError(f"fit budget exceeded: {len(attempts)} > {max_total}")
    family_counts = {family: sum(str(row.get("label", "")).startswith(f"inner_{family}_") for row in inner)
                     for family in ("A1", "A2")}
    if any(value > max_family for value in family_counts.values()):
        raise AssertionError("inner fit family budget exceeded")
    if complete and any(row.get("status") != "complete" for row in attempts):
        raise AssertionError("complete cohort contains a failed or unfinished fit attempt")
    actual = status.get("fits_actual")
    if actual is not None and int(actual) != len(attempts):
        raise AssertionError("fits_actual differs from recorded attempts")
    return {"inner": len(inner), "final_unique_fits": len(final), "total": len(attempts),
            "A1_inner": family_counts["A1"], "A2_inner": family_counts["A2"],
            "maximum_total": max_total, "maximum_per_family": max_family}


def _validate_winners(run: Path, specs: Sequence[Mapping[str, Any]], *, tol: float = TOLERANCE) -> dict[str, Any]:
    inner = _read_json(run / "inner_results.json")
    if not isinstance(inner, list) or not inner:
        raise ValueError("inner_results.json must contain completed inner trials")
    frozen = _read_json(run / "selection_frozen.json")
    reconstructed: dict[str, Mapping[str, Any]] = {}
    for family in ("A1", "A2"):
        rows = []
        for row in inner:
            if str(row.get("label", "")).startswith(f"inner_{family}_") and row.get("status", "complete") == "complete":
                score = row.get("metrics", {}).get("overall", {}).get("user_macro", {}).get("ndcg@20")
                if score is None:
                    raise ValueError(f"{row.get('label')} lacks inner user-macro NDCG@20")
                rows.append({"label": row["label"], "columns": row["columns"], "spec_ids": row["spec_ids"],
                             "inner_score": float(score)})
        if not rows:
            raise ValueError(f"no complete inner trials for {family}")
        winner = pick_inner_winner(rows)
        recorded = frozen.get("winners", {}).get(family)
        if not recorded:
            raise AssertionError(f"selection_frozen.json lacks winner for {family}")
        validate_frozen_winner(recorded, winner, tol=tol)
        reconstructed[family] = winner
    best = pick_inner_winner(list(reconstructed.values()))
    chosen = frozen.get("chosen_auto_by_inner", {})
    validate_frozen_winner(chosen, best, tol=tol)
    expected_specs_hash = sha256(run / "specs.json")
    if frozen.get("specs_sha256") != expected_specs_hash:
        raise AssertionError("frozen selection spec hash mismatch")
    if frozen.get("external_predictions_started") is True or frozen.get("external_evaluation_started") is True:
        raise AssertionError("selection freeze records external evaluation as already started")
    return {"A1": {"label": reconstructed["A1"]["label"], "inner_score": reconstructed["A1"]["inner_score"]},
            "A2": {"label": reconstructed["A2"]["label"], "inner_score": reconstructed["A2"]["inner_score"]},
            "chosen_auto": best["label"], "specs_sha256": expected_specs_hash}


def _validate_final_masks(run: Path, specs: Sequence[Mapping[str, Any]], status: Mapping[str, Any]) -> dict[str, Any]:
    frozen = _read_json(run / "selection_frozen.json")
    masks = frozen.get("final_masks")
    if not isinstance(masks, Mapping):
        raise ValueError("selection_frozen.json lacks final_masks")
    family_columns = {
        family: [i for i, spec in enumerate(specs) if family == "A2" or spec.get("family") == "A1"]
        for family in ("A1", "A2")
    }
    manual = [i for i, spec in enumerate(specs) if spec.get("family") == "A1"]
    if len(manual) != 66:
        raise AssertionError("manual reference in P005 specs does not contain 66 A1 features")
    for family in ("A1", "A2"):
        available = _available_for_family(_read_json(run / "pruning.json"), family)
        if any(column not in family_columns[family] for column in available):
            raise AssertionError(f"pruning {family} contains a column outside its DSL pool")
        winner_columns = frozen["winners"][family].get("columns", [])
        if any(int(column) not in available for column in winner_columns):
            raise AssertionError(f"frozen {family} winner contains a pruned/unavailable column")
        if list(masks.get(f"Auto_{family}", [])) != list(winner_columns):
            raise AssertionError(f"Auto_{family} final mask differs from frozen winner")
    manual_expected = {
        "M_I": [i for i in manual if specs[i]["group"] == "I"],
        "M_UI": [i for i in manual if specs[i]["group"] in "UI"],
        "M_UID": [i for i in manual if specs[i]["group"] in "UID"],
        "M_UIC": [i for i in manual if specs[i]["group"] in "UIC"],
        "M_UIDC": list(manual),
        "M_IDC": [i for i in manual if specs[i]["group"] in "IDC"],
        "M_UDC": [i for i in manual if specs[i]["group"] in "UDC"],
        "M_UID_Cartist": [i for i in manual if specs[i]["group"] != "C" or specs[i]["path"] == "c_artist"],
        "M_UID_Calbum": [i for i in manual if specs[i]["group"] != "C" or specs[i]["path"] == "c_album"],
    }
    for label, expected in manual_expected.items():
        if list(masks.get(label, [])) != expected:
            raise AssertionError(f"manual final mask {label} differs from frozen U/I/D/C schema")
    union = [int(x) for x in masks.get("Union", [])]
    if not set(manual).issubset(union):
        raise AssertionError("Union does not preserve all manual A1 columns")
    if any(int(column) not in family_columns["A2"] for column in union):
        raise AssertionError("Union contains a column outside the 206-feature DSL")
    chosen_auto = frozen.get("chosen_auto_by_inner", {}).get("columns", [])
    recorded_removed = frozen.get("union_removed", [])
    removed_columns = {int(row["column"]): row for row in recorded_removed}
    expected_union = set(manual) | {int(column) for column in chosen_auto}
    if set(union) - expected_union:
        raise AssertionError("Union contains columns not present in manual or chosen automatic set")
    for column in expected_union - set(union):
        removed = removed_columns.get(column)
        if not removed or removed.get("reason") not in {"already_manual", "duplicate_on_inner_fit"}:
            raise AssertionError(f"Union omitted automatic feature {column} without recorded de-dup reason")
        if int(removed.get("representative", -1)) not in set(union):
            raise AssertionError(f"Union duplicate {column} points to a missing representative")
    for group in "UIDC":
        expected = [column for column in union if specs[column]["group"] != group]
        if list(masks.get(f"Union_without_{group}", [])) != expected:
            raise AssertionError(f"Union_without_{group} mask is inconsistent")
    final_by_label = {str(row.get("label")): row for row in status.get("final_models", [])}
    if len(final_by_label) != len(status.get("final_models", [])):
        raise AssertionError("duplicate final variant labels")
    for label, columns in masks.items():
        entry = final_by_label.get(label)
        if entry is None:
            raise AssertionError(f"final mask {label} has no corresponding final_models entry")
        if entry.get("status") == "skipped_empty_group_ablation":
            if columns:
                raise AssertionError(f"nonempty final mask {label} was marked skipped")
        elif list(entry.get("columns", [])) != list(columns):
            raise AssertionError(f"final model columns disagree with frozen mask {label}")
        reused = entry.get("reused_from")
        if reused:
            target = final_by_label.get(str(reused))
            if target is None or list(target.get("columns", [])) != list(columns):
                raise AssertionError(f"final alias {label} does not match its reused mask target")
            for field in ("prediction_file", "prediction_sha256", "model_file", "model_sha256"):
                if entry.get(field) != target.get(field):
                    raise AssertionError(f"final alias {label} {field} differs from reused target")
    return {"final_variants": len(masks), "unique_manual_columns": len(manual),
            "union_columns": len(union), "aliases": sum(bool(row.get("reused_from")) for row in final_by_label.values())}


def _verify_predictions(run: Path, queries: Sequence[Mapping[str, Any]], split_ids: Mapping[str, Any],
                        status: Mapping[str, Any], *, tol: float = TOLERANCE) -> list[dict[str, Any]]:
    all_by_id = {int(query["query_id"]): query for query in queries}
    inner_entries = _read_json(run / "inner_results.json")
    final_entries = status.get("final_models", [])
    entries: list[Mapping[str, Any]] = list(inner_entries) + list(final_entries)
    checks = []
    for entry in entries:
        label = str(entry.get("label", "unnamed"))
        if entry.get("status") == "skipped_empty_group_ablation":
            continue
        pred_ref = entry.get("prediction_file", entry.get("prediction_path", entry.get("predictions_path")))
        if pred_ref is None:
            raise ValueError(f"{label} lacks prediction file path")
        prediction_path = Path(pred_ref)
        if not prediction_path.is_absolute():
            prediction_path = run / prediction_path
        if entry.get("reused_from"):
            reused = str(entry["reused_from"])
            target = next((row for row in final_entries if row.get("label") == reused), None)
            if target is None:
                raise AssertionError(f"{label} aliases missing final model {reused}")
            target_path = Path(target.get("prediction_file", ""))
            if not target_path.is_absolute():
                target_path = run / target_path
            if prediction_path != target_path:
                raise AssertionError(f"{label} prediction path differs from alias target")
            # A reused model carries its source metrics and prediction digest.
            if entry.get("prediction_sha256") != target.get("prediction_sha256"):
                raise AssertionError(f"{label} prediction hash differs from alias target")
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        pred_ids, _, _ = _load_npz(prediction_path)
        if pred_ids == list(split_ids["external_validation"]):
            expected_ids = list(split_ids["external_validation"])
            subset = "external_validation"
        elif pred_ids == list(split_ids["inner_selection"]):
            expected_ids = list(split_ids["inner_selection"])
            subset = "inner_selection"
        else:
            raise AssertionError(f"{label} prediction IDs match neither frozen split")
        scores = _prediction_scores(prediction_path, expected_ids, all_by_id)
        actual = recompute([all_by_id[qid] for qid in expected_ids], scores)
        stored = entry.get("metrics", entry.get("inner_metrics", {}))
        if entry.get("stage") == "inner" and "metrics" in stored:
            stored = stored["metrics"]
        if entry.get("reused_from"):
            source = next(row for row in final_entries if row.get("label") == entry["reused_from"])
            stored = source.get("metrics", {})
        errors = _compare_metrics(stored, actual, tol)
        if errors:
            raise AssertionError(f"{label} saved metrics mismatch: {'; '.join(errors[:8])}")
        expected_hash = entry.get("prediction_sha256")
        actual_hash = sha256(prediction_path)
        if expected_hash and actual_hash != expected_hash:
            raise AssertionError(f"{label} prediction SHA mismatch")
        checks.append({"name": label, "subset": subset, "query_count": len(expected_ids),
                       "prediction_sha256": actual_hash, "status": "verified"})
    return checks


def verify_cohort(root: Path, output: Path, task_root: Path = TASK) -> dict[str, Any]:
    """Verify one completed or failed P005 child without training or event scans."""
    run, output, task_root = Path(root), Path(output), Path(task_root)
    if output.exists():
        raise FileExistsError(f"independent verification output must be new: {output}")
    status_path = run / "run_status.json"
    status = _read_json(status_path)
    state = status.get("status")
    if state not in {"complete", "failed"}:
        raise ValueError(f"cohort status must be complete or failed, got {state!r}")
    config_path = _task_path(task_root, status["config_path"])
    protocol_path = _task_path(task_root, status["protocol_path"])
    for name, path, expected in (("config", config_path, status.get("config_sha256")),
                                 ("protocol", protocol_path, status.get("protocol_sha256"))):
        if not expected or sha256(path) != expected:
            raise AssertionError(f"P005 {name} SHA does not match child run")
    sources = _hash_ledger({str(Path("src") / name): digest for name, digest in status.get("source_code_sha256", {}).items()},
                           task_root, "source")
    inputs = _hash_ledger(status.get("input_sha256", {}), task_root, "input")
    resources = status.get("resources")
    if not isinstance(resources, Mapping):
        raise AssertionError("cohort status lacks final resource record")
    limits = resources.get("limits", status.get("config", {}).get("resource_limits", {}))
    peak = resources.get("peak_process_rss_mb")
    available = resources.get("minimum_available_system_mb")
    if state == "failed":
        if not status.get("error"):
            raise AssertionError("failed cohort status lacks error reason")
        budget = _validate_attempt_budget(status, status.get("config", {}), complete=False)
        report = {"status": "failed_checkpoint_verified", "run": str(run), "error": status["error"],
                  "resources": {"peak_process_rss_mb": peak, "minimum_available_system_mb": available,
                                "limits": limits}, "fit_budget": budget,
                  "verified_source_hashes": sources, "verified_input_hashes": inputs,
                  "metrics_recomputed": False, "failure_policy": "score outputs skipped for failed child"}
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report

    if status.get("final_test_evaluated"):
        raise AssertionError("P005 child must not evaluate final test")
    if resources.get("limit_exceeded"):
        raise AssertionError("complete child records a resource limit exceedance")
    if peak is not None and float(peak) > float(limits.get("runner_peak_rss_mb", math.inf)):
        raise AssertionError("complete child peak RSS exceeds the configured limit")
    if available is not None and float(available) < float(limits.get("minimum_available_mb", 0)):
        raise AssertionError("complete child violated minimum system memory reserve")
    budget = _validate_attempt_budget(status, status.get("config", {}), complete=True)
    membership_path = run.parent / "membership" / "memberships.json"
    parent_summary_path = run.parent / "summary.json"
    summary = _read_json(parent_summary_path) if parent_summary_path.exists() else None
    membership = _validate_membership(run, task_root, status, summary)
    queries = []
    with (run / "queries.jsonl").open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                query = json.loads(line)
                if not isinstance(query, dict):
                    raise ValueError(f"query line {line_no} is not an object")
                queries.append(query)
    split = _read_json(run / "split.json")
    split_info = _validate_split(queries, split)
    reported_counts = status.get("query_counts")
    if reported_counts:
        for key, actual_count in split_info["query_counts"].items():
            if int(reported_counts.get(key, -1)) != actual_count:
                raise AssertionError(f"run query count mismatch for {key}")
    metadata = _read_json(run / "data" / "metadata.json")
    boundaries = status.get("config", {}).get("boundaries_raw", {})
    expected_cutoff = int(boundaries.get("validation_end", metadata.get("cutoff_timestamp_raw", -1)))
    if int(metadata.get("cutoff_timestamp_raw", -2)) != expected_cutoff:
        raise AssertionError("cohort data cutoff differs from P005 frozen boundary")
    specs = _load_specs(run)
    if len(specs) != 206:
        raise AssertionError(f"P005 DSL must contain 206 specs, found {len(specs)}")
    if status.get("specs_sha256") != sha256(run / "specs.json"):
        raise AssertionError("specs hash mismatch")
    a1 = status.get("a1_manual_equivalence", {})
    build_shape = status.get("build", {}).get("shape", [])
    if len(build_shape) != 2 or a1.get("status") != "pass" or int(a1.get("cells_compared", -1)) != int(build_shape[0]) * 66:
        raise AssertionError("A1/manual byte-equivalence gate missing or incomplete")
    if not a1.get("identical_float32_bytes"):
        raise AssertionError("A1/manual byte-equivalence did not pass")
    frozen_hash = status.get("selection_frozen_sha256")
    if not frozen_hash or sha256(run / "selection_frozen.json") != frozen_hash:
        raise AssertionError("frozen selection hash mismatch")
    winners = _validate_winners(run, specs)
    masks = _validate_final_masks(run, specs, status)
    checks = _verify_predictions(run, queries, split_info, status)
    if status.get("output_sha256"):
        outputs = _hash_ledger(status["output_sha256"], run, "output")
    else:
        raise AssertionError("complete child has no output hash ledger")
    report = {"status": "verified", "run": str(run), "seed": membership["seed"],
              "query_counts": split_info["query_counts"], "cutoff": split_info["cutoff"],
              "membership": {k: v for k, v in membership.items() if k != "metadata_file"},
              "fit_budget": budget, "winners": winners, "final_masks": masks,
              "checks": checks, "verified_prediction_count": len(checks),
              "verified_output_sha256": outputs, "verified_input_sha256": inputs,
              "verified_source_sha256": sources,
              "resources": {"peak_process_rss_mb": peak, "minimum_available_system_mb": available,
                            "limits": limits},
              "independent_metric_implementation": "verify_auto_checkpoint.recompute; no project evaluator imported"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def verify_parent(run: Path, output: Path, task_root: Path = TASK) -> dict[str, Any]:
    """Verify membership and every completed/failed child in a finished P005 run."""
    run, output, task_root = Path(run), Path(output), Path(task_root)
    if output.exists():
        raise FileExistsError(f"independent verification output must be new: {output}")
    summary = _read_json(run / "summary.json")
    if summary.get("status") not in {"complete", "complete_with_failures", "failed"}:
        raise ValueError("P005 parent summary is still running or incomplete")
    config_path = _task_path(task_root, summary["config_path"])
    protocol_path = _task_path(task_root, summary["protocol_path"])
    if sha256(config_path) != summary.get("config_sha256") or sha256(protocol_path) != summary.get("protocol_sha256"):
        raise AssertionError("parent P005 config/protocol hash mismatch")
    membership = run / "membership" / "memberships.json"
    if not membership.is_file() or sha256(membership) != summary.get("memberships_sha256"):
        raise AssertionError("parent membership hash mismatch")
    memberships = _read_json(membership)
    if memberships.get("status") != "frozen" or not memberships.get("disjoint_cohorts"):
        raise AssertionError("P005 membership record is not frozen and disjoint")
    excluded = set(map(int, memberships.get("excluded_users", [])))
    cohort_sets = []
    for row in memberships.get("cohorts", []):
        users = set(map(int, row.get("selected_users", [])))
        if len(users) != 64 or users & excluded:
            raise AssertionError("invalid cohort size or excluded-user overlap")
        if any(users & previous for previous in cohort_sets):
            raise AssertionError("P005 cohort memberships overlap")
        cohort_sets.append(users)
        actual = hashlib.sha256(json.dumps(sorted(users), separators=(",", ":")).encode("ascii")).hexdigest()
        if row.get("selected_users_sha256") != actual:
            raise AssertionError("cohort user membership hash mismatch")
    cohort_reports = []
    for row in summary.get("cohorts", []):
        child = run / row["run_dir"]
        state = str(row.get("status"))
        child_status = _read_json(child / "run_status.json") if (child / "run_status.json").exists() else {}
        if state != child_status.get("status"):
            raise AssertionError(f"parent/child status disagreement for {row['run_dir']}")
        report_path = child / "verification_independent.json"
        report = verify_cohort(child, report_path, task_root)
        cohort_reports.append({"seed": int(row["seed"]), "status": report["status"],
                               "verification_file": str(report_path), "checks": report.get("verified_prediction_count", 0)})
    report = {"status": "verified", "run": str(run), "parent_status": summary["status"],
              "cohort_count": len(cohort_reports), "excluded_users": len(excluded),
              "disjoint_cohort_user_counts": [len(users) for users in cohort_sets],
              "membership_sha256": sha256(membership), "cohorts": cohort_reports}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cohort-only", action="store_true")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else TASK / args.run
    output = args.output or (run / "verification_independent.json")
    if not output.is_absolute():
        output = TASK / output
    if args.cohort_only:
        result = verify_cohort(run, output)
    else:
        result = verify_parent(run, output)
    print(f"P005 verification {result['status']}: {output}")


if __name__ == "__main__":
    main()
