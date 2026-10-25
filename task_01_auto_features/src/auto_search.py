"""Small deterministic helpers for bounded automatic feature search.

All positions refer to an immutable ordered query sequence. Candidate rows are
flattened query-major; feature matrices use the same row order. These helpers
never accept outer-validation scores for feature selection.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PRUNE_CHUNK_SIZE = 65_536
_SPLIT_TRAIN = "train"
_SPLIT_VALIDATION = "validation"
_GROUP_ORDER = ("U", "I", "D", "C", "X")


def _split_name(query: Mapping[str, Any]) -> str:
    value = query.get("split")
    if value == "val":
        return _SPLIT_VALIDATION
    return value


def temporal_split(queries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Split train queries into earlier fit/later selection; return outer validation positions.

    Input records need only ``timestamp`` and ``split``. Train cutoff is
    ``np.quantile(train_timestamps, .70, method="lower")``. Queries at the
    cutoff all go to inner selection so tied timestamps are never divided.
    """
    train_positions = [i for i, q in enumerate(queries) if _split_name(q) == _SPLIT_TRAIN]
    external_validation = [i for i, q in enumerate(queries) if _split_name(q) == _SPLIT_VALIDATION]
    if not train_positions:
        raise ValueError("temporal_split requires at least one train query")
    if not external_validation:
        raise ValueError("temporal_split requires external validation queries")
    try:
        train_timestamps = np.asarray([int(queries[i]["timestamp"]) for i in train_positions], dtype=np.int64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("every train query must have an integer timestamp") from exc
    cutoff = int(np.quantile(train_timestamps, .70, method="lower"))
    inner_fit = [i for i in train_positions if int(queries[i]["timestamp"]) < cutoff]
    inner_selection = [i for i in train_positions if int(queries[i]["timestamp"]) >= cutoff]
    if not inner_fit or not inner_selection:
        raise ValueError("70th-percentile cutoff must leave nonempty inner-fit and inner-selection sets")
    return {"cutoff": cutoff, "inner_fit": np.asarray(inner_fit, dtype=np.int64),
            "inner_selection": np.asarray(inner_selection, dtype=np.int64),
            "external_validation": np.asarray(external_validation, dtype=np.int64)}


def _candidate_ids(query: Mapping[str, Any]) -> list[int]:
    if "candidate_items" in query:
        raw = query["candidate_items"]
    elif "candidate_ids" in query:
        raw = query["candidate_ids"]
    elif "candidates" in query:
        raw = query["candidates"]
    else:
        raise ValueError("each query must contain candidate_items")
    ids = []
    for candidate in raw:
        if isinstance(candidate, Mapping):
            if "item_id" not in candidate:
                raise ValueError("candidate records must contain item_id")
            candidate = candidate["item_id"]
        ids.append(int(candidate))
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique within each query")
    return ids


def offsets_sizes(queries: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened candidate-row offsets and sizes, in query order."""
    offsets = np.empty(len(queries), dtype=np.int64)
    sizes = np.empty(len(queries), dtype=np.int64)
    offset = 0
    for i, query in enumerate(queries):
        size = len(_candidate_ids(query))
        offsets[i] = offset
        sizes[i] = size
        offset += size
    return offsets, sizes


def eligible_training_rows(queries: Sequence[Mapping[str, Any]], positions: Sequence[int],
                           offsets: Sequence[int], sizes: Sequence[int]) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Select complete candidate blocks only for queries whose target is present.

    Returns flattened candidate row indices, one binary target per returned row,
    and selection coverage counts. The helper uses targets only to determine
    training eligibility/labels; callers must keep these labels out of features.
    Missed targets remain in the reported query denominator, not as fabricated
    negatives. Every retained query contributes its entire candidate block.
    """
    if len(offsets) != len(queries) or len(sizes) != len(queries):
        raise ValueError("offsets and sizes must have one entry per query")
    expected_offsets, expected_sizes = offsets_sizes(queries)
    if not np.array_equal(np.asarray(offsets, dtype=np.int64), expected_offsets) or not np.array_equal(
            np.asarray(sizes, dtype=np.int64), expected_sizes):
        raise ValueError("offsets/sizes must describe the complete query-major candidate flattening")
    selected = [int(x) for x in positions]
    if any(i < 0 or i >= len(queries) for i in selected):
        raise IndexError("query position is out of range")
    if any(_split_name(queries[i]) != _SPLIT_TRAIN for i in selected):
        raise ValueError("eligible_training_rows accepts train-split positions only")
    row_indices: list[int] = []
    labels: list[int] = []
    hits = misses = candidate_rows_seen = 0
    for query_pos in selected:
        query = queries[query_pos]
        ids = _candidate_ids(query)
        offset, size = int(offsets[query_pos]), int(sizes[query_pos])
        if size != len(ids) or offset < 0:
            raise ValueError(f"candidate offset/size mismatch at query position {query_pos}")
        candidate_rows_seen += size
        if "target_item" in query:
            target = int(query["target_item"])
        elif "target_item_id" in query:
            target = int(query["target_item_id"])
        else:
            raise ValueError("each training query must contain target_item or target_item_id")
        positive_matches = sum(item == target for item in ids)
        if positive_matches == 0:
            misses += 1
            continue
        if positive_matches != 1:
            raise ValueError("an eligible query must have exactly one candidate matching its target")
        hits += 1
        row_indices.extend(range(offset, offset + size))
        labels.extend(int(item == target) for item in ids)
    query_count = len(selected)
    stats: dict[str, int | float] = {
        "queries_requested": query_count,
        "queries_with_target": hits,
        "queries_missing_target": misses,
        "target_coverage": hits / query_count if query_count else 0.0,
        "candidate_rows_examined": candidate_rows_seen,
        "candidate_rows_used": len(row_indices),
        "positive_rows": sum(labels),
    }
    return np.asarray(row_indices, dtype=np.int64), np.asarray(labels, dtype=np.uint8), stats


def _is_nan_value(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _column_digest(matrix: np.ndarray, column: int, fit_rows: np.ndarray,
                   chunk_size: int = PRUNE_CHUNK_SIZE) -> bytes:
    digest = hashlib.sha256()
    digest.update(matrix.dtype.str.encode("ascii"))
    digest.update(len(fit_rows).to_bytes(8, "little"))
    for start in range(0, len(fit_rows), chunk_size):
        values = matrix[fit_rows[start:start + chunk_size], column]
        if values.dtype.kind == "f":
            values = values.copy()
            values[np.isnan(values)] = np.nan  # canonicalize NaN payload bits for bucketing
            values[values == 0] = 0  # +0 and -0 compare equal
        digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.digest()


def _columns_equal(matrix: np.ndarray, left: int, right: int, fit_rows: np.ndarray,
                   chunk_size: int = PRUNE_CHUNK_SIZE) -> bool:
    for start in range(0, len(fit_rows), chunk_size):
        rows = fit_rows[start:start + chunk_size]
        if not np.array_equal(matrix[rows, left], matrix[rows, right], equal_nan=True):
            return False
    return True


def prune_columns(matrix: np.ndarray, fit_row_indices: Sequence[int],
                   candidate_columns: Sequence[int]) -> tuple[list[int], list[dict[str, int | str | None]]]:
    """Remove fit-row constants and exact duplicate columns, preserving first column index.

    Only one candidate column and bounded row chunks are read at a time. SHA256
    narrows duplicate checks; `array_equal` verifies each same-hash candidate.
    Rows outside `fit_row_indices` never affect pruning.
    """
    values = np.asarray(matrix)
    if values.ndim != 2 or values.dtype.kind not in "biuf":
        raise ValueError("matrix must be a 2-D numeric array")
    fit_rows = np.asarray(fit_row_indices, dtype=np.int64)
    if fit_rows.ndim != 1 or fit_rows.size == 0:
        raise ValueError("fit_row_indices must be a nonempty 1-D sequence")
    if np.any(fit_rows < 0) or np.any(fit_rows >= values.shape[0]):
        raise IndexError("fit row index is out of range")
    columns = sorted(set(int(c) for c in candidate_columns))
    if any(c < 0 or c >= values.shape[1] for c in columns):
        raise IndexError("candidate column index is out of range")

    kept: list[int] = []
    removed: list[dict[str, int | str | None]] = []
    digest_buckets: dict[bytes, list[int]] = {}
    for column in columns:
        first_value = None
        has_first = False
        constant = True
        for start in range(0, len(fit_rows), PRUNE_CHUNK_SIZE):
            rows = fit_rows[start:start + PRUNE_CHUNK_SIZE]
            chunk = values[rows, column]
            if not has_first:
                first_value = chunk[0].item() if hasattr(chunk[0], "item") else chunk[0]
                has_first = True
            equal = chunk == first_value
            if values.dtype.kind == "f" and _is_nan_value(first_value):
                equal = np.isnan(chunk)
            elif values.dtype.kind == "f":
                equal = equal | np.isnan(chunk) & np.isnan(first_value)
            if not bool(np.all(equal)):
                constant = False
                break
            if not constant:
                break
        if constant:
            removed.append({"column": column, "reason": "constant", "representative": None})
            continue
        digest = _column_digest(values, column, fit_rows)
        duplicates = digest_buckets.get(digest, [])
        representative = next((earlier for earlier in duplicates
                               if _columns_equal(values, column, earlier, fit_rows)), None)
        if representative is not None:
            removed.append({"column": column, "reason": "duplicate", "representative": representative})
        else:
            kept.append(column)
            digest_buckets.setdefault(digest, []).append(column)
    return kept, removed


def _spec_for_column(specs: Sequence[Mapping[str, Any]] | Mapping[int, Mapping[str, Any]], column: int) -> Mapping[str, Any]:
    if isinstance(specs, Mapping):
        spec = specs.get(column)
    elif column < len(specs):
        spec = specs[column]
    else:
        spec = None
    if spec is None:
        raise ValueError(f"missing feature spec for column {column}")
    return spec


def _spec_id(spec: Mapping[str, Any], column: int) -> str:
    value = spec.get("feature_id", spec.get("spec_id", spec.get("id")))
    if value is None:
        raise ValueError(f"feature spec for column {column} must have feature_id")
    return str(value)


def select_trial_columns(available_cols: Sequence[int], specs: Sequence[Mapping[str, Any]] | Mapping[int, Mapping[str, Any]],
                         importances: Sequence[float] | Mapping[int, float]) -> dict[str, list[int]]:
    """Create up to six unique subsets from inner-fit importances only.

    `specs` is indexed by original matrix column, or maps column index to a
    record with canonical `feature_id` and `group`. `importances` likewise indexes original
    columns. Returned lists are in canonical original-column order.
    """
    available = sorted(set(int(c) for c in available_cols))

    def importance(column: int) -> float:
        try:
            raw = importances[column]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"missing full-inner-fit importance for column {column}") from exc
        score = float(raw)
        if not math.isfinite(score):
            raise ValueError(f"importance for column {column} must be finite")
        return score

    records = {c: _spec_for_column(specs, c) for c in available}
    ids = {c: _spec_id(records[c], c) for c in available}
    if len(set(ids.values())) != len(ids):
        raise ValueError("spec_id values must be unique")
    ranked = sorted(available, key=lambda c: (-importance(c), ids[c]))
    subsets: list[tuple[str, list[int]]] = [("full", sorted(available))]
    for limit in (8, 16, 32, 64):
        subsets.append((f"top{limit}", sorted(ranked[:limit])))

    group_best: list[int] = []
    groups = {c: str(records[c].get("group", "")) for c in available}
    present = set(groups.values())
    ordered_groups = [g for g in _GROUP_ORDER if g in present]
    ordered_groups.extend(sorted(present - set(ordered_groups)))
    for group in ordered_groups:
        best = next((c for c in ranked if groups[c] == group), None)
        if best is not None:
            group_best.append(best)
    balanced = list(group_best[:32])
    balanced_set = set(balanced)
    for column in ranked:
        if len(balanced) >= 32:
            break
        if column not in balanced_set:
            balanced.append(column)
            balanced_set.add(column)
    subsets.append(("balanced32", sorted(balanced)))

    proposals: dict[str, list[int]] = {}
    seen: set[tuple[int, ...]] = set()
    for label, columns in subsets:
        signature = tuple(columns)
        if signature in seen:
            continue
        seen.add(signature)
        proposals[label] = columns
    return proposals


def pick_best(trial_results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Pick by inner score, then fewer columns, spec-id tuple, and label.

    Accepted keys are deliberately limited to inner-fit/selection trial data;
    external-validation/test scores are not accepted or consulted.
    """
    allowed = {"inner_score", "columns", "spec_ids", "label", "trial_id"}
    if not trial_results:
        raise ValueError("at least one trial result is required")
    candidates = []
    for result in trial_results:
        unexpected = set(result) - allowed
        if unexpected:
            raise ValueError(f"trial result contains disallowed fields: {sorted(unexpected)}")
        if "inner_score" not in result or "columns" not in result:
            raise ValueError("each trial result needs inner_score and columns")
        score = float(result["inner_score"])
        if not math.isfinite(score):
            raise ValueError("inner_score must be finite")
        columns = tuple(int(c) for c in result["columns"])
        spec_ids = tuple(sorted(str(x) for x in result.get("spec_ids", ())))
        label = str(result.get("label", result.get("trial_id", "")))
        candidates.append(((-score, len(columns), spec_ids, label), result))
    return min(candidates, key=lambda pair: pair[0])[1]
