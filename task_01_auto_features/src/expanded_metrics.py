"""Independent metric summaries and paired-user bootstrap from saved ranks."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

_STRATA = ("repeat", "new", "warm", "cold")


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _normalize_rows(rows: Iterable[Mapping[str, Any]], *, require_query_id: bool = False) -> list[dict[str, Any]]:
    normalized = []
    seen = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"row {index} must be a mapping")
        required = {"user_id", "target_rank", "candidate_count", "target_is_repeat", "target_known_before"}
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"row {index} is missing fields: {sorted(missing)}")
        user = _integer(raw["user_id"], f"row {index} user_id")
        count = _integer(raw["candidate_count"], f"row {index} candidate_count")
        rank_raw = raw["target_rank"]
        rank = None if rank_raw is None else _integer(rank_raw, f"row {index} target_rank", minimum=1)
        if rank is not None and rank > count:
            raise ValueError(f"row {index} target_rank exceeds candidate_count")
        repeat = _boolean(raw["target_is_repeat"], f"row {index} target_is_repeat")
        warm = _boolean(raw["target_known_before"], f"row {index} target_known_before")
        key = raw.get("cohort", None)
        if "query_id" in raw:
            query_id = _integer(raw["query_id"], f"row {index} query_id")
            key = (str(key), query_id) if key is not None else query_id
        else:
            if require_query_id:
                raise ValueError(f"paired row {index} must contain query_id")
            # summarize_ranks allows anonymous rows, but paired rows need stable query IDs.
            key = (str(key), index) if key is not None else index
        if key in seen:
            raise ValueError(f"duplicate query key: {key!r}")
        seen.add(key)
        normalized.append({"key": key, "user_id": user, "target_rank": rank,
                           "candidate_count": count, "target_is_repeat": repeat,
                           "target_known_before": warm})
    return normalized


def _empty_block() -> dict[str, Any]:
    return {"query_count": 0, "user_count": 0, "candidate_recall": None,
            "target_known_before_count": 0, "repeat_count": 0,
            "query_micro": {}, "user_macro": {}}


def _metric_block(rows: Sequence[Mapping[str, Any]], ks: tuple[int, ...]) -> dict[str, Any]:
    if not rows:
        return _empty_block()
    by_user: dict[int, list[dict[str, float]]] = defaultdict(list)
    per_query: list[dict[str, float]] = []
    for row in rows:
        rank = row["target_rank"]
        metrics: dict[str, float] = {}
        for k in ks:
            hit = rank is not None and rank <= k
            metrics[f"recall@{k}"] = float(hit)
            metrics[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
        per_query.append(metrics)
        by_user[int(row["user_id"])].append(metrics)
    names = [f"{metric}@{k}" for k in ks for metric in ("recall", "ndcg")]
    query_micro = {name: math.fsum(row[name] for row in per_query) / len(per_query) for name in names}
    user_macro = {
        name: math.fsum(math.fsum(row[name] for row in user_rows) / len(user_rows)
                        for user_rows in by_user.values()) / len(by_user)
        for name in names
    }
    return {"query_count": len(rows), "user_count": len(by_user),
            "candidate_recall": sum(row["target_rank"] is not None for row in rows) / len(rows),
            "target_known_before_count": sum(row["target_known_before"] for row in rows),
            "repeat_count": sum(row["target_is_repeat"] for row in rows),
            "query_micro": query_micro, "user_macro": user_macro}


def summarize_ranks(rows: Iterable[Mapping[str, Any]], ks: tuple[int, ...] = (10, 20)) -> dict[str, Any]:
    """Compute ranking metrics directly from one-based target ranks.

    A ``None`` target rank is a miss and remains zero in query and user
    denominators. Candidate recall measures whether the target occurs anywhere
    in the candidate catalog, independent of the selected ranking cutoffs.
    """
    if not isinstance(ks, tuple) or not ks:
        raise ValueError("ks must be a nonempty tuple of positive integers")
    normalized_ks = tuple(_integer(k, "k", minimum=1) for k in ks)
    if len(set(normalized_ks)) != len(normalized_ks):
        raise ValueError("ks must contain unique cutoffs")
    normalized = _normalize_rows(rows)
    strata_rows = {
        "repeat": [row for row in normalized if row["target_is_repeat"]],
        "new": [row for row in normalized if not row["target_is_repeat"]],
        "warm": [row for row in normalized if row["target_known_before"]],
        "cold": [row for row in normalized if not row["target_known_before"]],
    }
    return {"overall": _metric_block(normalized, normalized_ks),
            "strata": {name: _metric_block(strata_rows[name], normalized_ks) for name in _STRATA},
            "counts": {"queries": len(normalized), "users": len({row["user_id"] for row in normalized}),
                       "targets_in_candidates": sum(row["target_rank"] is not None for row in normalized),
                       "targets_missing_from_candidates": sum(row["target_rank"] is None for row in normalized)}}


def _bootstrap(deltas: Sequence[float], *, seed: int, resamples: int,
               confidence: float) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires nonempty finite per-user deltas")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha], method="linear")
    return {"users": len(values), "mean": float(math.fsum(values.tolist()) / len(values)),
            "ci_lower": float(lower), "ci_upper": float(upper), "confidence": confidence,
            "resamples": resamples, "seed": seed, "quantile_method": "linear",
            "positive_users": int(np.sum(values > 0)), "negative_users": int(np.sum(values < 0)),
            "equal_users": int(np.sum(values == 0)),
            "interpretation": "descriptive conditional paired-user bootstrap; no refitting or cohort uncertainty"}


def paired_user_delta(rows_a: Iterable[Mapping[str, Any]], rows_b: Iterable[Mapping[str, Any]],
                      seed: int = 20261102, resamples: int = 2000, *,
                      allow_catalog_change: bool = False, confidence: float = 0.95) -> dict[str, Any]:
    """Compare two rank modes on identical queries, then bootstrap users.

    Rows pair by ``(cohort, query_id)`` when cohort is present, otherwise by
    ``query_id``. Each pair must preserve user and target strata. Candidate counts
    must also match by default; set ``allow_catalog_change=True`` when comparing
    full-catalog ranks to a smaller retrieval catalog. The target rank itself may
    differ or be missing in either mode.
    """
    seed = _integer(seed, "seed")
    resamples = _integer(resamples, "resamples", minimum=1)
    if not isinstance(allow_catalog_change, bool):
        raise TypeError("allow_catalog_change must be boolean")
    if isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, (int, float, np.number)):
        raise TypeError("confidence must be numeric")
    confidence = float(confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    a = _normalize_rows(rows_a, require_query_id=True)
    b = _normalize_rows(rows_b, require_query_id=True)
    by_key_a = {row["key"]: row for row in a}
    by_key_b = {row["key"]: row for row in b}
    if set(by_key_a) != set(by_key_b):
        raise ValueError("paired rank modes must contain exactly the same cohort/query keys")
    per_user: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"ndcg20": [], "recall20": []})
    for key in by_key_a:
        left, right = by_key_a[key], by_key_b[key]
        for field in ("user_id", "target_is_repeat", "target_known_before"):
            if left[field] != right[field]:
                raise ValueError(f"paired query {key!r} differs in {field}")
        if not allow_catalog_change and left["candidate_count"] != right["candidate_count"]:
            raise ValueError(f"paired query {key!r} differs in candidate_count")
        user_id = left["user_id"]
        for field, cutoff in (("ndcg20", 20), ("recall20", 20)):
            rank_a, rank_b = left["target_rank"], right["target_rank"]
            if field == "ndcg20":
                score_a = 1.0 / math.log2(rank_a + 1) if rank_a is not None and rank_a <= cutoff else 0.0
                score_b = 1.0 / math.log2(rank_b + 1) if rank_b is not None and rank_b <= cutoff else 0.0
            else:
                score_a = float(rank_a is not None and rank_a <= cutoff)
                score_b = float(rank_b is not None and rank_b <= cutoff)
            per_user[user_id][field].append(score_b - score_a)
    output: dict[str, Any] = {"paired_queries": len(by_key_a), "paired_users": len(per_user),
                              "direction": "mode_b_minus_mode_a",
                              "candidate_count_equality_required": not allow_catalog_change,
                              "bootstrap": {}}
    for metric in ("ndcg20", "recall20"):
        deltas = [math.fsum(values[metric]) / len(values[metric]) for values in per_user.values()]
        output["bootstrap"][metric] = _bootstrap(deltas, seed=seed, resamples=resamples,
                                                 confidence=confidence)
    return output
