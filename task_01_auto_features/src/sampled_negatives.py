"""Deterministic query-local negative sampling for exploratory P008.

The module contains no model fitting, feature preparation or validation
scoring. Sampling order and seed derivation are explicit for independent
reconstruction by the P008 verifier.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def seed_for(cohort_id: str, query_id: int, stream: str, seed: int = 20261103) -> int:
    """Derive a stable PCG64 seed from SHA-256.

    Hash input is UTF-8 ``f"{seed}|{cohort_id}|{query_id}|{stream}"``; the
    first eight digest bytes are interpreted as an unsigned big-endian int.
    """
    if not isinstance(cohort_id, str) or not cohort_id:
        raise ValueError("cohort_id must be a non-empty string")
    if isinstance(query_id, (bool, np.bool_)) or not isinstance(query_id, (int, np.integer)):
        raise TypeError("query_id must be an integer")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a non-empty string")
    payload = f"{int(seed)}|{cohort_id}|{int(query_id)}|{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def uniform_draw(
    catalog: Sequence[int] | np.ndarray,
    target: int,
    count: int,
    rng: np.random.Generator | None = None,
    *,
    seed: int | None = None,
    exclude: Sequence[int] = (),
) -> list[int]:
    """Draw distinct catalog IDs uniformly without replacement.

    The returned list preserves NumPy's draw order. Callers sort only when
    materializing model rows. The target and optional exclusions are removed
    before sampling; insufficient eligible catalog size is an error.
    """
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if rng is not None and seed is not None:
        raise ValueError("supply either rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(seed)
    values = np.asarray(catalog, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("catalog must be one-dimensional")
    if values.size != np.unique(values).size:
        raise ValueError("catalog IDs must be unique")
    if values.size and np.any(values < 0):
        raise ValueError("catalog IDs must be non-negative")
    excluded = {int(target), *(int(item) for item in exclude)}
    eligible = values[~np.isin(values, np.fromiter(excluded, dtype=np.int64))]
    if int(count) > len(eligible):
        raise ValueError(f"requested {count} negatives from only {len(eligible)} eligible catalog items")
    if int(count) == 0:
        return []
    return list(map(int, rng.choice(eligible, size=int(count), replace=False).tolist()))


def build_query_samples(
    catalog: Sequence[int] | np.ndarray,
    original_candidates: Sequence[int],
    target: int,
    teacher_score_fn: Callable[[list[int]], Sequence[float] | np.ndarray],
    cohort_id: str,
    query_id: int,
    sampling_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the uniform and hybrid 100-item query blocks and mining evidence.

    Hybrid hard-pool sampling is performed from the catalog after excluding
    the target and the first 50 uniform draws. Its 256-item (or smaller)
    uniform draw is unioned with the old K100, then target/uniform overlaps are
    removed before sorted teacher scoring. The return data preserve draw order
    separately from score-rank order.
    """
    target = int(target)
    catalog_ids = np.asarray(catalog, dtype=np.int64)
    if catalog_ids.ndim != 1 or catalog_ids.size != np.unique(catalog_ids).size:
        raise ValueError("catalog must be a one-dimensional array of unique IDs")
    catalog_set = set(map(int, catalog_ids.tolist()))
    if target not in catalog_set:
        raise ValueError("build_query_samples requires a warm target in C(t)")
    candidates = list(map(int, original_candidates))
    if len(candidates) != len(set(candidates)):
        raise ValueError("original candidate IDs must be unique")
    if not set(candidates).issubset(catalog_set):
        raise ValueError("original K100 candidates must belong to C(t)")

    seed = int(sampling_config["seed"])
    uniform_count = int(sampling_config["uniform_negatives"])
    hybrid_uniform_count = int(sampling_config["hybrid_uniform"])
    hybrid_hard_count = int(sampling_config["hybrid_hard"])
    pool_draw_count = int(sampling_config["hard_pool_uniform"])
    if uniform_count != 99 or hybrid_uniform_count != 50 or hybrid_hard_count != 49:
        raise ValueError("P008 requires exactly 99 uniform negatives and a 50+49 hybrid block")
    if pool_draw_count <= 0:
        raise ValueError("hard_pool_uniform must be positive")
    uniform_rng = np.random.default_rng(seed_for(cohort_id, query_id, "uniform99", seed))
    uniform_draw_order = uniform_draw(catalog_ids, target, uniform_count, uniform_rng)
    uniform_items = sorted([target, *uniform_draw_order])
    uniform_hybrid = uniform_draw_order[:hybrid_uniform_count]

    pool_rng = np.random.default_rng(seed_for(cohort_id, query_id, "hard_pool256", seed))
    pool_count = min(pool_draw_count, len(catalog_set) - 1 - len(uniform_hybrid))
    pool_draw_order = uniform_draw(catalog_ids, target, pool_count, pool_rng, exclude=uniform_hybrid)
    hard_pool_set = (set(candidates) | set(pool_draw_order)) - {target} - set(uniform_hybrid)
    hard_pool_items = sorted(hard_pool_set)
    if len(hard_pool_items) < hybrid_hard_count:
        raise ValueError(f"hard pool has only {len(hard_pool_items)} items; {hybrid_hard_count} are required")
    hard_pool_scores = np.asarray(teacher_score_fn(hard_pool_items), dtype=np.float64)
    if hard_pool_scores.ndim != 1 or len(hard_pool_scores) != len(hard_pool_items):
        raise ValueError("teacher_score_fn must return one score per sorted hard-pool item")
    if not np.isfinite(hard_pool_scores).all():
        raise ValueError("teacher scores must all be finite")
    hard_items = [item for _score, item in sorted(
        zip(hard_pool_scores.tolist(), hard_pool_items), key=lambda pair: (-pair[0], pair[1]))[:hybrid_hard_count]]
    if len(set(hard_items)) != hybrid_hard_count or set(hard_items) & set(uniform_hybrid):
        raise AssertionError("hybrid uniform and hard negatives must be distinct")
    hybrid_items = sorted([target, *uniform_hybrid, *hard_items])
    if len(uniform_items) != 100 or len(hybrid_items) != 100:
        raise AssertionError("P008 model blocks must contain 100 unique items")
    return {
        "uniform_items": uniform_items,
        "hybrid_items": hybrid_items,
        "hard_pool_items": hard_pool_items,
        "hard_pool_scores": hard_pool_scores.tolist(),
        "hard_items": hard_items,
        "uniform_draw_order": uniform_draw_order,
        "hybrid_uniform_draw_order": uniform_hybrid,
        "hard_pool_draw_order": pool_draw_order,
        "seed_uniform99": seed_for(cohort_id, query_id, "uniform99", seed),
        "seed_hard_pool256": seed_for(cohort_id, query_id, "hard_pool256", seed),
    }
