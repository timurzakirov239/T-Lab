"""Pure deterministic proposal utilities for bounded P007 feature search.

These functions only construct candidate column masks or importance rankings.
They do not read experiment data, fit models, evaluate outer splits, or select a
winner. All column indices refer to the caller's fixed 206-column matrix.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_GROUP_ORDER = ("U", "I", "D", "C")


def _available(columns: Sequence[int]) -> list[int]:
    values = [int(column) for column in columns]
    if len(values) != len(set(values)):
        raise ValueError("available columns must be unique")
    if any(column < 0 or column >= 206 for column in values):
        raise ValueError("feature columns must be in [0, 205]")
    return sorted(values)


def _score(importances: Sequence[float] | Mapping[int, float], column: int) -> float:
    try:
        value = float(importances[column])
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"missing importance for column {column}") from exc
    if not math.isfinite(value):
        raise ValueError(f"importance for column {column} must be finite")
    return value


def _group(groups: Sequence[str] | Mapping[int, str], column: int) -> str:
    try:
        value = groups[column]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"missing group label for column {column}") from exc
    value = str(value)
    if not value:
        raise ValueError(f"empty group label for column {column}")
    return value


def _unique_proposals(proposals: Sequence[tuple[str, Sequence[int]]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    seen: set[tuple[int, ...]] = set()
    for name, raw in proposals:
        columns = sorted(set(map(int, raw)))
        key = tuple(columns)
        if key in seen:
            continue
        seen.add(key)
        result[str(name)] = columns
    return result


def importance_proposals(
    available_columns: Sequence[int],
    importances: Sequence[float] | Mapping[int, float],
    groups: Sequence[str] | Mapping[int, str],
    sizes: Sequence[int] = (8, 16, 32, 64),
    balanced_size: int = 32,
    group_order: Sequence[str] = DEFAULT_GROUP_ORDER,
) -> dict[str, list[int]]:
    """Return full, top-k and group-balanced masks using deterministic ranking.

    Rank is descending importance, then ascending original column index. The
    balanced mask first includes the strongest column from each represented
    feature group (in ``group_order``), then fills remaining slots by the same
    global ranking. Duplicate masks are omitted while preserving proposal
    order.
    """
    available = _available(available_columns)
    if not available:
        raise ValueError("at least one available feature column is required")
    limits = [int(size) for size in sizes]
    if any(size <= 0 for size in limits) or len(set(limits)) != len(limits):
        raise ValueError("sizes must be distinct positive integers")
    balanced_size = int(balanced_size)
    if balanced_size <= 0:
        raise ValueError("balanced_size must be positive")
    scores = {column: _score(importances, column) for column in available}
    labels = {column: _group(groups, column) for column in available}
    ranked = sorted(available, key=lambda column: (-scores[column], column))
    proposed: list[tuple[str, Sequence[int]]] = [("full", available)]
    proposed.extend((f"top{size}", ranked[:size]) for size in limits)

    represented = set(labels.values())
    order = [str(group) for group in group_order]
    order.extend(sorted(represented - set(order)))
    picks = [next(column for column in ranked if labels[column] == group)
             for group in order if group in represented]
    balanced = picks[:balanced_size]
    included = set(balanced)
    for column in ranked:
        if len(balanced) >= min(balanced_size, len(available)):
            break
        if column not in included:
            balanced.append(column)
            included.add(column)
    proposed.append((f"balanced{len(balanced)}", balanced))
    return _unique_proposals(proposed)


def random_proposals(
    available_columns: Sequence[int],
    sizes: Sequence[int] = (8, 16, 32, 64, 128),
    n_proposals: int = 6,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Draw reproducible unique masks, cycling requested sizes as needed.

    Returned masks are sorted original column indices. If the finite search
    space cannot provide the requested number of unique masks, generation
    stops early after a deterministic attempt budget and returns what exists.
    """
    available = _available(available_columns)
    if not available:
        raise ValueError("at least one available feature column is required")
    if n_proposals < 0:
        raise ValueError("n_proposals cannot be negative")
    if n_proposals == 0:
        return {}
    limits = [int(size) for size in sizes]
    if not limits or any(size <= 0 or size > len(available) for size in limits):
        raise ValueError("every requested size must be between 1 and available column count")
    rng = np.random.default_rng(int(seed))
    result: dict[str, list[int]] = {}
    seen: set[tuple[int, ...]] = set()
    attempts = max(100, n_proposals * 100)
    for attempt in range(attempts):
        if len(result) >= n_proposals:
            break
        size = limits[len(result) % len(limits)]
        mask = tuple(sorted(map(int, rng.choice(available, size=size, replace=False))))
        if mask in seen:
            continue
        seen.add(mask)
        result[f"random_{len(result) + 1:02d}_n{size}"] = list(mask)
    return result


def forward_group_proposals(
    group_columns: Mapping[str, Sequence[int]],
    selected_groups: Sequence[str] = (),
    group_order: Sequence[str] = DEFAULT_GROUP_ORDER,
) -> dict[str, list[int]]:
    """Propose each one-group extension of a caller-selected group set.

    The caller evaluates proposals on inner data and decides which group to
    add. This helper deliberately does not score or rank the alternatives.
    """
    normalized: dict[str, list[int]] = {}
    for group, columns in group_columns.items():
        name = str(group)
        if not name:
            raise ValueError("group names must be non-empty")
        values = _available(columns)
        if not values:
            raise ValueError(f"group {name} has no feature columns")
        normalized[name] = values
    selected = set(map(str, selected_groups))
    if not selected.issubset(normalized):
        raise ValueError("selected_groups contains an unknown group")
    all_columns = [column for values in normalized.values() for column in values]
    if len(all_columns) != len(set(all_columns)):
        raise ValueError("group column sets must be disjoint")
    order = [str(group) for group in group_order]
    order.extend(sorted(set(normalized) - set(order)))
    proposals = []
    base = set().union(*(set(normalized[group]) for group in selected)) if selected else set()
    for group in order:
        if group in normalized and group not in selected:
            proposals.append((f"add_{group}", sorted(base | set(normalized[group]))))
    return _unique_proposals(proposals)


def query_group_permutation(offsets: Sequence[int], seed: int = 0) -> np.ndarray:
    """Return row-source indices for whole-query block permutation.

    Candidate blocks are permuted only among queries with the same candidate
    count. Within each block candidate rank/order is preserved. Singleton size
    strata remain unchanged; the result is always a permutation of all flat
    candidate-row indices.
    """
    starts = np.asarray(offsets, dtype=np.int64)
    if starts.ndim != 1 or len(starts) == 0 or starts[0] != 0:
        raise ValueError("offsets must be a 1-D array starting at zero")
    if np.any(np.diff(starts) < 0):
        raise ValueError("offsets must be nondecreasing")
    sizes = np.diff(starts)
    if np.any(sizes <= 0):
        raise ValueError("query groups must contain at least one row")
    rng = np.random.default_rng(int(seed))
    strata: dict[int, list[int]] = defaultdict(list)
    for index, size in enumerate(sizes):
        strata[int(size)].append(index)
    source = np.arange(int(starts[-1]), dtype=np.int64)
    for indices in strata.values():
        if len(indices) < 2:
            continue
        shuffled = rng.permutation(indices)
        for destination_query, source_query in zip(indices, shuffled):
            destination_start = int(starts[destination_query])
            source_start = int(starts[int(source_query)])
            size = int(sizes[destination_query])
            source[destination_start:destination_start + size] = np.arange(source_start, source_start + size)
    return source


def stability_importance(
    first_importances: Sequence[float] | Mapping[int, float],
    second_importances: Sequence[float] | Mapping[int, float],
    available_columns: Sequence[int] | None = None,
    dispersion_penalty: float = 1.0,
) -> dict[int, float]:
    """Combine two temporal-fit importances as normalized mean minus dispersion.

    Each fit is normalized by its sum of absolute importance, making the
    scales comparable. All-zero fits normalize to zero. The returned score is
    ``mean(normalized importance) - penalty * population_std`` and may be
    negative for unstable columns.
    """
    if not math.isfinite(float(dispersion_penalty)) or dispersion_penalty < 0:
        raise ValueError("dispersion_penalty must be a finite non-negative number")
    if available_columns is None:
        if isinstance(first_importances, Mapping) and isinstance(second_importances, Mapping):
            columns = sorted(set(map(int, first_importances)) | set(map(int, second_importances)))
        else:
            columns = list(range(min(len(first_importances), len(second_importances))))
    else:
        columns = _available(available_columns)
    if not columns:
        raise ValueError("at least one available feature column is required")
    first = np.asarray([_score(first_importances, column) for column in columns], dtype=np.float64)
    second = np.asarray([_score(second_importances, column) for column in columns], dtype=np.float64)
    first_scale = float(np.abs(first).sum())
    second_scale = float(np.abs(second).sum())
    first = first / first_scale if first_scale else np.zeros_like(first)
    second = second / second_scale if second_scale else np.zeros_like(second)
    values = np.vstack((first, second))
    means = values.mean(axis=0)
    dispersion = values.std(axis=0, ddof=0)
    scores = means - float(dispersion_penalty) * dispersion
    return {column: float(score) for column, score in zip(columns, scores)}


def query_group_permutation_importance(
    matrix: np.ndarray,
    offsets: Sequence[int],
    column: int,
    seed: int = 0,
) -> np.ndarray:
    """Return a copied matrix with one column permuted by whole query groups.

    This convenience function only changes feature values. Evaluation and
    metric comparison remain the caller's responsibility on inner data.
    """
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if column < 0 or column >= values.shape[1]:
        raise IndexError("column is out of range")
    permutation = query_group_permutation(offsets, seed)
    if len(permutation) != values.shape[0]:
        raise ValueError("offsets must cover all matrix rows")
    result = values.copy()
    result[:, column] = values[permutation, column]
    return result
