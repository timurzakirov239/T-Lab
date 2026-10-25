"""Ranking metrics for the small, reserved-free pilot evaluation.

Targets are never added to candidate lists here. A target omitted by retrieval
is a miss in ranking metrics and in candidate_recall.
"""

from __future__ import annotations

import math
from collections import defaultdict
from numbers import Integral
from typing import Any

import numpy as np


def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _validate(queries: list[dict], scores_by_query: list[np.ndarray | list], ks: tuple[int, ...]):
    if not isinstance(queries, list) or not isinstance(scores_by_query, list):
        raise TypeError("queries and scores_by_query must be lists")
    if len(queries) != len(scores_by_query):
        raise ValueError("queries and scores_by_query must have equal lengths")
    if not isinstance(ks, tuple) or not ks:
        raise ValueError("ks must be a non-empty tuple of positive integers")
    if any(not _is_int(k) or k <= 0 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("ks must contain unique positive integers")

    seen_query_ids = set()
    validated = []
    required = {
        "query_id", "user_id", "timestamp", "target_item", "candidate_items",
        "split", "target_is_repeat", "target_known_before",
    }
    for index, (query, raw_scores) in enumerate(zip(queries, scores_by_query)):
        if not isinstance(query, dict):
            raise TypeError(f"query at index {index} must be a dict")
        missing = required.difference(query)
        if missing:
            raise ValueError(f"query at index {index} is missing keys: {sorted(missing)}")
        for name in ("query_id", "user_id", "timestamp", "target_item"):
            if not _is_int(query[name]):
                raise TypeError(f"{name} at query index {index} must be an integer")
        if query["query_id"] in seen_query_ids:
            raise ValueError(f"duplicate query_id: {query['query_id']}")
        seen_query_ids.add(query["query_id"])
        if query["split"] not in ("train", "validation"):
            raise ValueError(f"unsupported split at query index {index}: {query['split']!r}")
        for name in ("target_is_repeat", "target_known_before"):
            if not isinstance(query[name], (bool, np.bool_)):
                raise TypeError(f"{name} at query index {index} must be boolean")
        candidates = query["candidate_items"]
        if not isinstance(candidates, list):
            raise TypeError(f"candidate_items at query index {index} must be a list")
        if any(not _is_int(item) for item in candidates):
            raise TypeError(f"candidate item IDs at query index {index} must be integers")
        if len(set(candidates)) != len(candidates):
            raise ValueError(f"duplicate candidate item ID at query index {index}")

        scores = np.asarray(raw_scores)
        if scores.ndim != 1:
            raise ValueError(f"scores at query index {index} must be one-dimensional")
        if len(scores) != len(candidates):
            raise ValueError(f"score/candidate length mismatch at query index {index}")
        if not np.issubdtype(scores.dtype, np.number):
            raise TypeError(f"scores at query index {index} must be numeric")
        if not np.isfinite(scores).all():
            raise ValueError(f"scores at query index {index} must be finite (no NaN/inf)")

        # Candidate input order never resolves ties: item ID ascending does.
        order = sorted(range(len(candidates)), key=lambda j: (-float(scores[j]), candidates[j]))
        ranked_items = [candidates[j] for j in order]
        try:
            target_rank = ranked_items.index(query["target_item"])
        except ValueError:
            target_rank = None
        validated.append((query, target_rank))
    return validated


def _summarize(rows: list[tuple[dict, int | None]], ks: tuple[int, ...]) -> dict:
    if not rows:
        return {
            "query_count": 0,
            "user_count": 0,
            "target_known_before_count": 0,
            "target_known_before_fraction": None,
            "repeat_count": 0,
            "repeat_fraction": None,
            "candidate_recall": None,
            "query_micro": {f"{metric}@{k}": None for k in ks for metric in ("recall", "ndcg")},
            "user_macro": {f"{metric}@{k}": None for k in ks for metric in ("recall", "ndcg")},
        }

    per_query: list[dict[str, float]] = []
    by_user: dict[int, list[dict[str, float]]] = defaultdict(list)
    candidate_hits = 0
    known_count = 0
    repeat_count = 0
    for query, rank in rows:
        candidate_hits += rank is not None
        known_count += bool(query["target_known_before"])
        repeat_count += bool(query["target_is_repeat"])
        values: dict[str, float] = {}
        for k in ks:
            hit = rank is not None and rank < k
            values[f"recall@{k}"] = float(hit)
            values[f"ndcg@{k}"] = 1.0 / math.log2(rank + 2) if hit else 0.0
        per_query.append(values)
        by_user[query["user_id"]].append(values)

    metric_names = [f"{metric}@{k}" for k in ks for metric in ("recall", "ndcg")]
    query_micro = {name: float(np.mean([row[name] for row in per_query])) for name in metric_names}
    user_macro = {
        name: float(np.mean([
            np.mean([row[name] for row in user_rows]) for user_rows in by_user.values()
        ]))
        for name in metric_names
    }
    return {
        "query_count": len(rows),
        "user_count": len(by_user),
        "target_known_before_count": int(known_count),
        "target_known_before_fraction": float(known_count / len(rows)),
        "repeat_count": int(repeat_count),
        "repeat_fraction": float(repeat_count / len(rows)),
        "candidate_recall": float(candidate_hits / len(rows)),
        "query_micro": query_micro,
        "user_macro": user_macro,
    }


def evaluate_ranking(
    queries: list[dict],
    scores_by_query: list[np.ndarray | list],
    ks: tuple[int, ...] = (10, 20),
) -> dict:
    """Evaluate one next-listen target per query without changing candidates.

    Missing targets count as zero for Recall/NDCG and remain in every applicable
    denominator. Query-micro averages all queries equally; user-macro first
    averages a user's queries, then averages users equally. Candidate recall is
    the fraction of targets present in the supplied candidate lists.
    """
    rows = _validate(queries, scores_by_query, ks)
    overall = _summarize(rows, ks)

    by_split = {
        split: _summarize([row for row in rows if row[0]["split"] == split], ks)
        for split in ("train", "validation")
    }
    strata = {
        "repeat": _summarize([row for row in rows if row[0]["target_is_repeat"]], ks),
        "new": _summarize([row for row in rows if not row[0]["target_is_repeat"]], ks),
        "warm": _summarize([row for row in rows if row[0]["target_known_before"]], ks),
        "cold": _summarize([row for row in rows if not row[0]["target_known_before"]], ks),
    }
    return {
        "overall": overall,
        "by_split": by_split,
        "strata": strata,
        "counts": {
            "queries": len(rows),
            "users": overall["user_count"],
            "target_known_before": overall["target_known_before_count"],
            "target_not_known_before": len(rows) - overall["target_known_before_count"],
            "repeat_targets": overall["repeat_count"],
            "new_targets": len(rows) - overall["repeat_count"],
            "targets_in_candidates": int(round((overall["candidate_recall"] or 0.0) * len(rows))),
            "targets_missing_from_candidates": len(rows) - int(round((overall["candidate_recall"] or 0.0) * len(rows))),
        },
    }
