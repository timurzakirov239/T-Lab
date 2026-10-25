"""Causal, bounded-memory scoring over every item observed before a query."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def _normalize_columns(columns: Sequence[int] | Mapping[str, Sequence[int]] | None,
                       label: str, width: int) -> np.ndarray | None:
    if columns is None:
        return None
    raw = columns.get(label) if isinstance(columns, Mapping) else columns
    if raw is None:
        raise ValueError(f"columns mapping has no entry for scorer {label!r}")
    values = [int(value) for value in raw]
    if len(values) != len(set(values)):
        raise ValueError(f"columns for {label!r} contain duplicates")
    if any(value < 0 or value >= width for value in values):
        raise ValueError(f"columns for {label!r} are outside feature width {width}")
    return np.asarray(values, dtype=np.int64)


def _score_vector(scorer: Callable[[np.ndarray], Any], matrix: np.ndarray,
                  label: str) -> np.ndarray:
    raw = np.asarray(scorer(matrix), dtype=np.float64)
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw[:, 0]
    if raw.ndim != 1 or len(raw) != len(matrix):
        raise ValueError(f"scorer {label!r} must return one score per catalog item")
    if not np.isfinite(raw).all():
        raise ValueError(f"scorer {label!r} returned non-finite scores")
    return raw


def transform_catalog(engine: Any, user_id: int, item_ids: Sequence[int], timestamp: int) -> np.ndarray:
    """Transform catalog items with exact-value memoization for AutoFeatureEngine.

    The shared 206-column grammar repeats user, user-item, and user-entity
    calculations across expressions. This path memoizes each validated
    expression value by its underlying item/user/entity state key while sharing
    AutoFeatureEngine's existing count cache. It returns the same float32 values
    and order as ``engine.transform``. Other supported feature engines use their
    native transform unchanged.
    """
    specs = getattr(engine, "specs", None)
    if specs is None or not callable(getattr(engine, "_one_value", None)):
        return engine.transform(int(user_id), list(map(int, item_ids)), int(timestamp))
    engine._assert_query_time(int(timestamp))
    uid = int(user_id)
    items = list(map(int, item_ids))
    count_cache: dict[tuple[int, str, str | int], float] = {}
    value_cache: dict[tuple[int, str, Any], float] = {}
    rows: list[list[float]] = []
    for item_id in items:
        artist_ids = engine.artist_map.get(item_id, ())
        album_ids = engine.album_map.get(item_id, ())
        row = []
        for spec_index, spec in enumerate(specs):
            path = spec["path"]
            if path == "u":
                state_key: Any = uid
            elif path == "i":
                state_key = item_id
            elif path == "d":
                state_key = (uid, item_id)
            elif path == "c_artist":
                state_key = (uid, artist_ids)
            elif path == "c_album":
                state_key = (uid, album_ids)
            elif path == "i_artist":
                state_key = artist_ids
            elif path == "i_album":
                state_key = album_ids
            else:
                raise ValueError(f"unexpected feature path {path!r}")
            key = (spec_index, path, state_key)
            value = value_cache.get(key)
            if value is None:
                value = engine._one_value(spec, uid, item_id, int(timestamp), artist_ids, album_ids, count_cache)
                value_cache[key] = value
            row.append(value)
        rows.append(row)
    if not rows:
        return np.empty((0, len(engine.feature_names)), dtype=np.float32)
    result = np.asarray(rows, dtype=np.float32)
    if result.shape[1] != len(engine.feature_names):
        raise RuntimeError("internal automatic feature order mismatch")
    return result


def evaluate_full_catalog(
    engine: Any,
    user_id: int,
    timestamp: int,
    target_item: int,
    scorers: Mapping[str, Callable[[np.ndarray], Any]],
    *,
    columns: Sequence[int] | Mapping[str, Sequence[int]] | None = None,
    chunk_size: int = 512,
    top_k: int = 20,
    transform_fn: Callable[[Any, int, Sequence[int], int], np.ndarray] | None = None,
    monitor: Any = None,
) -> dict[str, Any]:
    """Score all catalog items represented in the engine's causal prefix.

    The catalog is the sorted set of keys in ``engine._items``. The engine's
    ``transform`` implementation enforces ``last_observed_timestamp < timestamp``;
    therefore a future-only item cannot enter this candidate set. Feature rows
    are materialized and scored one bounded chunk at a time. When several model
    scorers are supplied, each feature chunk is shared across them.

    ``columns`` may be one shared sequence or a per-scorer mapping. A scorer
    receives the corresponding selected feature columns, or the complete matrix
    when columns is ``None``. Ties are sorted by score descending then item ID
    ascending. An unavailable target is retained as ``target_rank=None`` and
    ``target_score=None``; no target is inserted into the catalog.

    Returns a dictionary with ``catalog_size``, sorted ``catalog_items``,
    ``target_present``, and a ``models`` mapping. Each model record contains
    ``target_score``, one-based ``target_rank``, and up to ``top_k`` records of
    ``{"item_id", "score"}``.
    """
    if isinstance(user_id, (bool, np.bool_)) or not isinstance(user_id, (int, np.integer)):
        raise TypeError("user_id must be an integer")
    if isinstance(timestamp, (bool, np.bool_)) or not isinstance(timestamp, (int, np.integer)):
        raise TypeError("timestamp must be an integer")
    if isinstance(target_item, (bool, np.bool_)) or not isinstance(target_item, (int, np.integer)):
        raise TypeError("target_item must be an integer")
    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if not isinstance(scorers, Mapping) or not scorers:
        raise ValueError("scorers must be a non-empty label-to-callable mapping")
    if any(not isinstance(label, str) or not label or not callable(scorer)
           for label, scorer in scorers.items()):
        raise TypeError("scorers must map nonempty string labels to callable scorers")

    item_stats = getattr(engine, "_items", None)
    transform = getattr(engine, "transform", None)
    if not isinstance(item_stats, Mapping) or not callable(transform):
        raise TypeError("engine must expose an item-state mapping `_items` and callable transform")
    catalog = np.asarray(sorted(map(int, item_stats.keys())), dtype=np.int64)
    feature_width = len(getattr(engine, "feature_names", []))
    selected_columns = {label: _normalize_columns(columns, label, feature_width)
                        for label in scorers}
    buffers: dict[str, list[np.ndarray]] = {label: [] for label in scorers}

    for start in range(0, len(catalog), int(chunk_size)):
        item_chunk = catalog[start:start + int(chunk_size)].tolist()
        features = np.asarray(transform_fn(engine, int(user_id), item_chunk, int(timestamp))
                              if transform_fn is not None else transform_catalog(engine, int(user_id), item_chunk, int(timestamp)))
        if features.ndim != 2 or features.shape[0] != len(item_chunk):
            raise ValueError("engine.transform must return a 2D row per catalog item")
        if features.dtype != np.float32:
            raise TypeError("engine.transform must preserve float32 feature values")
        if feature_width and features.shape[1] != feature_width:
            raise ValueError("engine.transform feature width differs from feature_names")
        if not np.isfinite(features).all():
            raise ValueError("engine.transform returned non-finite feature values")
        for label, scorer in scorers.items():
            col = selected_columns[label]
            model_features = features if col is None else features[:, col]
            buffers[label].append(_score_vector(scorer, model_features, label))
        if monitor is not None:
            check = getattr(monitor, "check", None)
            if not callable(check):
                raise TypeError("monitor must expose a callable check() method")
            check()

    models: dict[str, Any] = {}
    for label in scorers:
        values = np.concatenate(buffers[label]) if buffers[label] else np.empty(0, dtype=np.float64)
        ordered = sorted(range(len(catalog)), key=lambda index: (-float(values[index]), int(catalog[index])))
        rank = None
        target_score = None
        target_position = int(np.searchsorted(catalog, int(target_item)))
        if target_position < len(catalog) and int(catalog[target_position]) == int(target_item):
            target_score = float(values[target_position])
            rank = next(position + 1 for position, index in enumerate(ordered) if index == target_position)
        models[label] = {
            "target_score": target_score,
            "target_rank": rank,
            "top_items": [{"item_id": int(catalog[index]), "score": float(values[index])}
                          for index in ordered[:int(top_k)]],
        }
    return {"catalog_size": int(len(catalog)), "catalog_items": catalog.tolist(),
            "target_present": int(target_item) in set(map(int, catalog)), "models": models}
