"""Independent ranking-metric verifier for a completed P004 checkpoint.

This module reads query JSONL, split/result JSON, and prediction NPZ files only.
It does not import the project evaluator, feature code, model package, or event
tables. It ranks ties by score DESC then item_id ASC and keeps missing targets
as zeroes in the full query denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RUN = Path("artifacts/p004_auto64")
DEFAULT_QUERIES = Path("artifacts/pilot_001/queries.jsonl")
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queries(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"query line {line_no} is not a JSON object")
                rows.append(row)
    ids = [q["query_id"] for q in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("query JSONL has duplicate query_id")
    return rows


def _metric_block(rows: list[tuple[dict[str, Any], int | None]], ks: tuple[int, ...]) -> dict[str, Any]:
    if not rows:
        return {"query_count": 0, "user_count": 0, "candidate_recall": None,
                "target_known_before_count": 0, "repeat_count": 0,
                "query_micro": {}, "user_macro": {}}
    by_user: dict[int, list[dict[str, float]]] = defaultdict(list)
    per_query: list[dict[str, float]] = []
    for query, rank in rows:
        one = {}
        for k in ks:
            hit = rank is not None and rank < k
            one[f"recall@{k}"] = float(hit)
            one[f"ndcg@{k}"] = (1.0 / math.log2(rank + 2)) if hit else 0.0
        per_query.append(one)
        by_user[int(query["user_id"])].append(one)
    names = [f"{kind}@{k}" for k in ks for kind in ("recall", "ndcg")]
    micro = {name: math.fsum(row[name] for row in per_query) / len(per_query) for name in names}
    macro = {
        name: math.fsum(math.fsum(row[name] for row in user_rows) / len(user_rows)
                        for user_rows in by_user.values()) / len(by_user)
        for name in names
    }
    candidate_hits = sum(rank is not None for _, rank in rows)
    return {"query_count": len(rows), "user_count": len(by_user),
            "candidate_recall": candidate_hits / len(rows),
            "target_known_before_count": sum(bool(q.get("target_known_before")) for q, _ in rows),
            "repeat_count": sum(bool(q.get("target_is_repeat")) for q, _ in rows),
            "query_micro": micro, "user_macro": macro}


def recompute(queries: list[dict[str, Any]], scores: list[list[float]] | list[np.ndarray],
              ks: tuple[int, ...] = (10, 20)) -> dict[str, Any]:
    """Independently recompute overall and repeat/new, warm/cold ranking metrics."""
    if len(queries) != len(scores):
        raise ValueError("queries and scores must have equal lengths")
    ranked_rows = []
    hits = 0
    for i, (query, raw) in enumerate(zip(queries, scores)):
        candidates = query.get("candidate_items")
        if candidates is None:
            raise ValueError(f"query at index {i} lacks candidate_items")
        vector = np.asarray(raw, dtype=np.float64)
        if vector.ndim != 1 or len(vector) != len(candidates):
            raise ValueError(f"prediction length mismatch at query index {i}")
        if not np.isfinite(vector).all():
            raise ValueError(f"non-finite score at query index {i}")
        pairs = sorted(zip(vector.tolist(), map(int, candidates)), key=lambda pair: (-pair[0], pair[1]))
        rank = next((position for position, (_, item) in enumerate(pairs)
                     if item == int(query["target_item"])), None)
        hits += rank is not None
        ranked_rows.append((query, rank))
    strata = {
        "repeat": _metric_block([x for x in ranked_rows if bool(x[0]["target_is_repeat"])], ks),
        "new": _metric_block([x for x in ranked_rows if not bool(x[0]["target_is_repeat"])], ks),
        "warm": _metric_block([x for x in ranked_rows if bool(x[0]["target_known_before"])], ks),
        "cold": _metric_block([x for x in ranked_rows if not bool(x[0]["target_known_before"])], ks),
    }
    return {"overall": _metric_block(ranked_rows, ks), "strata": strata,
            "counts": {"queries": len(queries), "users": len({int(q["user_id"]) for q in queries}),
                       "targets_in_candidates": hits, "targets_missing_from_candidates": len(queries) - hits}}


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_npz(path: Path) -> tuple[list[int], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        query_key = next((k for k in ("query_ids", "query_id") if k in data), None)
        offsets_key = next((k for k in ("offsets", "candidate_offsets") if k in data), None)
        score_key = next((k for k in ("scores", "predictions", "probabilities") if k in data), None)
        if not query_key or not offsets_key or not score_key:
            raise ValueError(f"NPZ must contain query_ids, offsets, scores: {path}")
        query_ids = np.asarray(data[query_key], dtype=np.int64)
        offsets = np.asarray(data[offsets_key], dtype=np.int64)
        scores = np.asarray(data[score_key], dtype=np.float64)
    if offsets.ndim != 1 or len(offsets) != len(query_ids) + 1 or offsets[0] != 0:
        raise ValueError(f"invalid prediction offsets: {path}")
    if np.any(np.diff(offsets) < 0) or offsets[-1] != len(scores):
        raise ValueError(f"prediction offsets do not cover scores: {path}")
    return query_ids.tolist(), offsets, scores


def _prediction_scores(path: Path, query_ids: list[int], all_queries: dict[int, dict]) -> list[np.ndarray]:
    pred_ids, offsets, flat = _load_npz(path)
    if pred_ids != query_ids:
        raise ValueError(f"prediction query IDs do not match expected split: {path}")
    results = []
    for i, qid in enumerate(pred_ids):
        candidate_count = len(all_queries[qid]["candidate_items"])
        if offsets[i + 1] - offsets[i] != candidate_count:
            raise ValueError(f"candidate offsets mismatch for query {qid}: {path}")
        results.append(flat[offsets[i]:offsets[i + 1]])
    return results


def _compare_metrics(expected: dict, actual: dict, tol: float = TOLERANCE) -> list[str]:
    errors = []
    if not isinstance(expected, dict) or "overall" not in expected or "strata" not in expected:
        return ["stored metrics lack overall/strata ranking summaries"]
    for section in ("overall", "strata"):
        exp_section = expected.get(section, {})
        act_section = actual.get(section, {})
        labels = ["overall"] if section == "overall" else sorted(set(exp_section) | set(act_section))
        for label in labels:
            e = exp_section if section == "overall" else exp_section.get(label, {})
            a = act_section if section == "overall" else act_section.get(label, {})
            prefix = section if section == "overall" else f"{section}.{label}"
            for count_key in ("query_count", "user_count", "target_known_before_count", "repeat_count"):
                if count_key in e and e.get(count_key) != a.get(count_key):
                    errors.append(f"{prefix}.{count_key}: expected {e.get(count_key)!r}, got {a.get(count_key)!r}")
            if "candidate_recall" in e:
                ev, av = e.get("candidate_recall"), a.get("candidate_recall")
                if ev is None or av is None:
                    if ev != av: errors.append(f"{prefix}.candidate_recall: expected {ev!r}, got {av!r}")
                elif not math.isclose(float(ev), float(av), rel_tol=0.0, abs_tol=tol):
                    errors.append(f"{prefix}.candidate_recall: expected {ev:.16g}, got {av:.16g}")
            for avg_type in ("query_micro", "user_macro"):
                e_avg, a_avg = e.get(avg_type, {}), a.get(avg_type, {})
                for metric in sorted(set(e_avg) | set(a_avg)):
                    ev, av = e_avg.get(metric), a_avg.get(metric)
                    if ev is None or av is None:
                        if ev != av: errors.append(f"{prefix}.{avg_type}.{metric}: expected {ev!r}, got {av!r}")
                    elif not math.isclose(float(ev), float(av), rel_tol=0.0, abs_tol=tol):
                        errors.append(f"{prefix}.{avg_type}.{metric}: expected {ev:.16g}, got {av:.16g}")
    return errors


def _resolve_alias(entry: dict[str, Any], run: Path) -> tuple[Path, dict[str, Any]]:
    current = entry
    seen = set()
    while current.get("reused_from"):
        ref = str(current["reused_from"])
        if ref in seen: raise ValueError("prediction alias cycle")
        seen.add(ref)
        candidate = Path(ref)
        if not candidate.is_absolute(): candidate = run / candidate
        if candidate.is_file():
            canonical = dict(current)
            canonical["predictions_path"] = str(candidate)
            return candidate, canonical
        result_path = run / "results" / (ref if ref.endswith(".json") else f"{ref}.json")
        current = _load_json(result_path)
    path = current.get("predictions_path", current.get("prediction_path", current.get("prediction_file")))
    if path is None:
        raise ValueError(f"result entry {current.get('model', current.get('trial_id', 'unknown'))} lacks predictions_path")
    pred_path = Path(path)
    if not pred_path.is_absolute(): pred_path = run / pred_path
    return pred_path, current


def verify_run(run: Path, query_path: Path, output: Path, tolerance: float = TOLERANCE) -> dict[str, Any]:
    run, query_path, output = Path(run), Path(query_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Verifier output must be a new file: {output}")
    status_path = run / "run_status.json"
    status = _load_json(status_path)
    if status.get("status") != "complete":
        raise ValueError("P004 run_status must be complete")
    split_path = run / "split.json"
    split = _load_json(split_path)
    all_rows = load_queries(query_path)
    all_by_id = {int(q["query_id"]): q for q in all_rows}
    expected_external_ids = [int(q["query_id"]) for q in all_rows if q.get("split") == "validation"]
    if len(expected_external_ids) != 499:
        raise ValueError(f"Expected 499 original P001 validation queries, found {len(expected_external_ids)}")
    if len(all_rows) != 1848:
        raise ValueError(f"Expected 1848 original P001 queries, found {len(all_rows)}")
    split_ids = split.get("query_ids", split)
    def to_query_ids(key: str, fallback: str) -> list[int]:
        values = [int(x) for x in split_ids.get(key, split_ids.get(fallback, []))]
        # P004 stores positions into the canonical P001 JSONL ordering.
        if all(0 <= x < len(all_rows) for x in values):
            return [int(all_rows[x]["query_id"]) for x in values]
        return values
    inner_fit_ids = to_query_ids("inner_fit", "inner_fit_query_ids")
    inner_selection_ids = to_query_ids("inner_selection", "inner_selection_query_ids")
    external_ids = to_query_ids("external_validation", "external_validation_query_ids")
    if not inner_fit_ids or not inner_selection_ids:
        raise ValueError("split.json must contain nonempty inner_fit and inner_selection query IDs")
    if external_ids != expected_external_ids:
        raise ValueError("split.json external_validation IDs differ from original P001 validation IDs")
    if set(inner_fit_ids) & set(inner_selection_ids) or set(inner_fit_ids) & set(external_ids) or set(inner_selection_ids) & set(external_ids):
        raise ValueError("query split IDs overlap")

    inner_results_path = run / "inner_results.json"
    inner_entries = json.loads(inner_results_path.read_text(encoding="utf-8")) if inner_results_path.exists() else []
    final_entries = status.get("final_models", [])
    entries = list(inner_entries) + list(final_entries)
    if not entries:
        results_dir = run / "results"
        entries = [_load_json(path) for path in sorted(results_dir.glob("*.json")) if path.name != "selection_frozen.json"]
    checks = []
    outputs_hashes: dict[str, str] = {}
    final_by_label = {str(entry.get("label")): entry for entry in final_entries}
    for entry in entries:
        name = str(entry.get("model", entry.get("trial_id", entry.get("label", "unnamed"))))
        pred_path, result = _resolve_alias(entry, run)
        if entry in final_entries and entry.get("reused_from"):
            target_label = str(entry["reused_from"])
            target = final_by_label.get(target_label)
            if target is None:
                raise AssertionError(f"{name}: reused_from target {target_label!r} is absent")
            declared = Path(str(entry.get("prediction_file", "")))
            target_declared = Path(str(target.get("prediction_file", "")))
            if declared != target_declared:
                raise AssertionError(f"{name}: cached prediction path differs from reused_from target")
        result_file = entry.get("result_path", entry.get("result_json"))
        if result_file:
            result_path = Path(result_file)
            if not result_path.is_absolute(): result_path = run / result_path
            result = _load_json(result_path)
        elif entry.get("stage") == "final":
            sidecar = run / "results" / f"{name}.json"
            if sidecar.exists(): result = _load_json(sidecar)
        if not pred_path.is_file(): raise FileNotFoundError(pred_path)
        pred_ids, _, _ = _load_npz(pred_path)
        if pred_ids == external_ids:
            expected_ids = external_ids
            subset_type = "external_validation"
            stored_metrics = result.get("metrics", result.get("external_metrics", {}))
            if "metrics" in stored_metrics and isinstance(stored_metrics["metrics"], dict): stored_metrics = stored_metrics["metrics"]
            # Run metrics may wrap the ranking block in a validation key.
            if "validation" in stored_metrics and "overall" not in stored_metrics: stored_metrics = stored_metrics["validation"]
        elif pred_ids == inner_selection_ids:
            expected_ids = inner_selection_ids
            subset_type = "inner_selection"
            stored_metrics = result.get("inner_metrics", result.get("metrics", {}))
        else:
            raise ValueError(f"{name}: predictions query IDs match neither inner selection nor external validation")
        selected_queries = [all_by_id[qid] for qid in expected_ids]
        scores = _prediction_scores(pred_path, expected_ids, all_by_id)
        actual = recompute(selected_queries, scores)
        errors = _compare_metrics(stored_metrics, actual, tolerance)
        if errors:
            raise AssertionError(f"{name}: metric mismatch: {'; '.join(errors[:10])}")
        outputs_hashes[str(pred_path.relative_to(run)) if pred_path.is_relative_to(run) else str(pred_path)] = sha256(pred_path)
        checks.append({"name": name, "subset": subset_type, "query_count": len(expected_ids), "prediction_path": str(pred_path),
                       "metrics_match_tolerance": tolerance, "status": "verified"})

    # Frozen inner winner is independently reconstructed from saved inner result summaries.
    frozen_path = run / "selection_frozen.json"
    if not frozen_path.exists(): frozen_path = run / "results" / "selection_frozen.json"
    frozen = _load_json(frozen_path)
    trials = inner_entries or frozen.get("trials", frozen.get("inner_trials", []))
    if trials:
        for family in ("A1", "A2"):
            family_trials = [t for t in trials if family in str(t.get("label", t.get("trial_id", "")))
                             and t.get("status", "complete") == "complete"]
            if not family_trials: continue
            def trial_score(t):
                metrics = t.get("metrics", {})
                score = metrics.get("overall", {}).get("user_macro", {}).get("ndcg@20")
                if score is None: score = t.get("inner_user_macro_ndcg@20")
                return float(score)
            winner = min(family_trials, key=lambda t: (-trial_score(t), len(t.get("spec_ids", t.get("feature_ids", []))),
                                                       tuple(sorted(t.get("spec_ids", t.get("feature_ids", []))))))
            recorded = frozen.get("winners", {}).get(family, {})
            winner_ids = sorted(winner.get("spec_ids", winner.get("feature_ids", [])))
            if recorded and sorted(recorded.get("spec_ids", recorded.get("feature_ids", []))) != winner_ids:
                raise AssertionError(f"frozen {family} winner does not match inner scores/tie-break")
            if recorded and not math.isclose(float(recorded.get("inner_score", trial_score(winner))), trial_score(winner), rel_tol=0.0, abs_tol=tolerance):
                raise AssertionError(f"frozen {family} score differs from best inner result")
    # Verify every saved output digest declared by run_status, including feature bytes by streaming hash only.
    declared_outputs = status.get("output_sha256", {})
    outputs_verified = {}
    for relative, expected_hash in declared_outputs.items():
        output_path = run / relative
        if not output_path.is_file():
            raise FileNotFoundError(f"declared output missing: {relative}")
        actual_hash = sha256(output_path)
        if actual_hash != expected_hash:
            raise AssertionError(f"output hash mismatch: {relative}")
        outputs_verified[relative] = actual_hash
    inputs_verified = {}
    task_root = Path(__file__).resolve().parents[1]
    for relative, expected_hash in status.get("input_sha256", {}).items():
        input_path = task_root / relative
        if not input_path.is_file():
            raise FileNotFoundError(f"declared input missing: {relative}")
        actual_hash = sha256(input_path)
        if actual_hash != expected_hash:
            raise AssertionError(f"input hash mismatch: {relative}")
        inputs_verified[relative] = actual_hash
    if sha256(query_path) != status.get("input_sha256", {}).get("artifacts/pilot_001/queries.jsonl", sha256(query_path)):
        raise AssertionError("original query JSONL hash differs from run input hash")
    config_path = task_root / "configs" / "p004_auto.json"
    protocol_path = task_root / "configs" / "P004_PROTOCOL.md"
    for label, path, expected in (("config", config_path, status.get("config_sha256")),
                                  ("protocol", protocol_path, status.get("protocol_sha256"))):
        if expected:
            actual_hash = sha256(path)
            if actual_hash != expected:
                raise AssertionError(f"{label} source hash differs from the run manifest")
            inputs_verified[str(path.relative_to(task_root))] = actual_hash
    for name, expected_hash in status.get("source_code_sha256", {}).items():
        code_path = task_root / "src" / name
        if not code_path.is_file(): raise FileNotFoundError(code_path)
        actual_hash = sha256(code_path)
        if actual_hash != expected_hash:
            raise AssertionError(f"source code hash differs from run manifest: {name}")
        inputs_verified[str(code_path.relative_to(task_root))] = actual_hash
    # Verify frozen best-family choice by the two independently reconstructed family winners.
    if "A1" in frozen.get("winners", {}) and "A2" in frozen.get("winners", {}):
        best_family = min((frozen["winners"][family] for family in ("A1", "A2")),
                          key=lambda x: (-float(x["inner_score"]), len(x.get("spec_ids", [])), tuple(sorted(x.get("spec_ids", [])))))
        chosen = frozen.get("chosen_auto_by_inner", {})
        if chosen and list(chosen.get("spec_ids", [])) != list(best_family.get("spec_ids", [])):
            raise AssertionError("chosen automatic family does not match best inner family score")
    for path in (status_path, split_path, query_path, frozen_path):
        outputs_hashes[str(path)] = sha256(path)
    report = {"status": "verified", "run": str(run), "query_source": str(query_path), "queries": len(all_rows),
              "expected_external_validation_queries": len(expected_external_ids), "inner_fit_queries": len(inner_fit_ids),
              "inner_selection_queries": len(inner_selection_ids), "checks": checks, "verified_prediction_sha256": outputs_hashes,
              "verified_declared_output_sha256": outputs_verified, "verified_input_sha256": inputs_verified,
              "tolerance": tolerance, "independent_metric_implementation": "verify_auto_checkpoint.recompute; does not import pilot_evaluation"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else Path(__file__).resolve().parents[1] / args.run
    queries = args.queries if args.queries.is_absolute() else Path(__file__).resolve().parents[1] / args.queries
    output = args.output or run / "verification_independent.json"
    if not output.is_absolute(): output = Path(__file__).resolve().parents[1] / output
    report = verify_run(run, queries, output)
    print(f"Verified {len(report['checks'])} prediction outputs; report: {output}")


if __name__ == "__main__": main()
