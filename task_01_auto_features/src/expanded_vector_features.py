"""Vectorized 206-column transforms for large fixed candidate catalogs.

The caller advances ``engine`` causally with ``observe`` exactly as for
``AutoFeatureEngine.transform``. This adapter uses the full development event
arrays only as a timestamp-indexed acceleration structure; each query reads a
strict prefix and never materializes future values into returned features.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    from .pilot_features import EVENT_SIGNAL, SIGNALS
except ImportError:  # direct src/ use
    from pilot_features import EVENT_SIGNAL, SIGNALS

_SIGNAL_INDEX = {name: index for index, name in enumerate(SIGNALS)}
_EVENT_TYPES = frozenset(("listen", "like", "dislike", "unlike", "undislike"))


def _event_arrays(events: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ("uid", "item_id", "timestamp")
    if any(name not in events for name in required):
        raise ValueError("events need uid, item_id and timestamp columns")
    uid = np.asarray(events["uid"], dtype=np.int64)
    item = np.asarray(events["item_id"], dtype=np.int64)
    times = np.asarray(events["timestamp"], dtype=np.int64)
    if any(values.ndim != 1 for values in (uid, item, times)) or not (len(uid) == len(item) == len(times)):
        raise ValueError("event columns must be 1-D arrays of equal length")
    if "event_type" in events:
        raw_types = np.asarray(events["event_type"])
        names = [str(value) for value in raw_types.tolist()]
    elif "event_code" in events and "event_names" in events:
        codes = np.asarray(events["event_code"], dtype=np.int64)
        names_by_code = list(events["event_names"])
        if codes.ndim != 1 or len(codes) != len(uid) or (len(codes) and (codes.min() < 0 or codes.max() >= len(names_by_code))):
            raise ValueError("event_code must align with event rows and event_names")
        names = [str(names_by_code[int(code)]) for code in codes]
    else:
        raise ValueError("events need event_type or event_code plus event_names")
    if len(names) != len(uid) or any(name not in _EVENT_TYPES for name in names):
        raise ValueError("event types must align and use supported Yambda names")
    if len(times) and (np.any(times < 0) or np.any(uid < 0) or np.any(item < 0)):
        raise ValueError("event IDs and timestamps must be nonnegative")
    signal = np.fromiter((_SIGNAL_INDEX.get(name, -1) for name in names), dtype=np.int8, count=len(names))
    order = np.argsort(times, kind="stable")
    return uid[order], item[order], times[order], signal[order]


class FullCatalogFeatures:
    """Drop-in transform adapter for ``AutoFeatureEngine`` over large pools.

    ``events`` must be the complete allowed development history for the
    cohort, sorted internally once. The causal ``engine`` remains externally
    advanced. Query timestamps must be nondecreasing. Equal-time requests are
    valid when the external engine has observed no event at that timestamp.
    """

    def __init__(self, engine: Any, events: Mapping[str, Any]) -> None:
        self.engine = engine
        self.uid, self.item, self.timestamp, self.signal = _event_arrays(events)
        item_ids = set(map(int, self.item.tolist()))
        item_ids.update(map(int, engine.artist_map.keys()))
        item_ids.update(map(int, engine.album_map.keys()))
        self.item_ids = np.asarray(sorted(item_ids), dtype=np.int64)
        self.item_to_dense = {int(item_id): index for index, item_id in enumerate(self.item_ids)}
        self.event_dense = np.fromiter((self.item_to_dense[int(item_id)] for item_id in self.item),
                                       dtype=np.int32, count=len(self.item))
        self._cursor = 0
        self._all_counts = np.zeros((len(self.item_ids), len(SIGNALS)), dtype=np.float64)
        self._last_seen = np.full(len(self.item_ids), -1, dtype=np.int64)
        widths = {int(engine.short_window)}
        for spec in engine.specs:
            window = spec.get("window")
            if isinstance(window, int):
                widths.add(window)
                if spec.get("op") == "trend":
                    widths.add(2 * window)
        self._widths = tuple(sorted(widths))
        self._window_counts = {width: np.zeros_like(self._all_counts) for width in self._widths}
        self._window_start = {width: 0 for width in self._widths}
        self._window_end = {width: 0 for width in self._widths}
        self._user_positions: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._last_query_timestamp: int | None = None
        self._last_count_timestamp: int | None = None
        self._active_query_key: tuple[int, int] | None = None
        self._active_pair_all: np.ndarray | None = None
        self._active_pair_recent: dict[int, np.ndarray] = {}
        self._active_user_values: dict[int, float] = {}
        self._active_relation_values: dict[int, np.ndarray] = {}
        self._mapping_edges = {
            "artist": self._edges(engine.artist_map),
            "album": self._edges(engine.album_map),
        }
        self._mapping_degree = {
            relation: np.bincount(edges[0], minlength=len(self.item_ids)).astype(np.float64)
            for relation, edges in self._mapping_edges.items()
        }
        self._mapping_groups = {}
        for relation, (edge_rows, _edge_entities, _unique, _inverse) in self._mapping_edges.items():
            if len(edge_rows):
                starts = np.r_[0, np.flatnonzero(edge_rows[1:] != edge_rows[:-1]) + 1]
                items = edge_rows[starts]
                ends = np.r_[starts[1:], len(edge_rows)]
                degrees = ends - starts
            else:
                starts = ends = items = degrees = np.empty(0, dtype=np.int64)
            self._mapping_groups[relation] = (starts, ends, items, degrees)

    def _edges(self, mapping: Mapping[int, Sequence[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows: list[int] = []
        entities: list[int] = []
        for item_id in self.item_ids:
            dense = self.item_to_dense[int(item_id)]
            for entity in mapping.get(int(item_id), ()):
                rows.append(dense)
                entities.append(int(entity))
        if not rows:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
        unique, inverse = np.unique(np.asarray(entities, dtype=np.int64), return_inverse=True)
        return (np.asarray(rows, dtype=np.int32), np.asarray(entities, dtype=np.int64),
                unique, inverse.astype(np.int32, copy=False))

    def _apply_event_range(self, target: np.ndarray, start: int, end: int, sign: float) -> None:
        if end <= start:
            return
        dense = self.event_dense[start:end]
        signals = self.signal[start:end]
        target[:, 0] += sign * np.bincount(dense, minlength=len(self.item_ids))
        for signal_index in range(1, len(SIGNALS)):
            selected = dense[signals == signal_index]
            if selected.size:
                target[:, signal_index] += sign * np.bincount(selected, minlength=len(self.item_ids))

    def _advance_item_counts(self, timestamp: int) -> None:
        end = int(np.searchsorted(self.timestamp, timestamp, side="left"))
        if end < self._cursor:
            raise ValueError("query timestamps must be nondecreasing over the event history")
        if end > self._cursor:
            self._apply_event_range(self._all_counts, self._cursor, end, 1.0)
            np.maximum.at(self._last_seen, self.event_dense[self._cursor:end], self.timestamp[self._cursor:end])
        for width in self._widths:
            window = self._window_counts[width]
            old_end = self._window_end[width]
            if end > old_end:
                self._apply_event_range(window, old_end, end, 1.0)
            cutoff = timestamp - width
            remove_end = min(end, int(np.searchsorted(self.timestamp, cutoff, side="right")))
            start = self._window_start[width]
            if remove_end > start:
                self._apply_event_range(window, start, remove_end, -1.0)
            self._window_start[width] = remove_end
            self._window_end[width] = end
        self._cursor = end

    def _user_events(self, user_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._user_positions.get(user_id)
        if cached is not None:
            return cached
        positions = np.flatnonzero(self.uid == user_id)
        cached = (self.timestamp[positions], self.event_dense[positions], self.signal[positions])
        self._user_positions[user_id] = cached
        return cached

    def _pair_counts(self, user_id: int, timestamp: int, width: int | None = None) -> np.ndarray:
        times, dense, signals = self._user_events(user_id)
        end = int(np.searchsorted(times, timestamp, side="left"))
        if width is None:
            start = 0
        else:
            start = int(np.searchsorted(times, timestamp - width, side="right"))
        rows = dense[start:end]
        kinds = signals[start:end]
        counts = np.zeros((len(self.item_ids), len(SIGNALS)), dtype=np.float64)
        if len(rows):
            counts[:, 0] = np.bincount(rows, minlength=len(self.item_ids))
            for signal_index in range(1, len(SIGNALS)):
                chosen = rows[kinds == signal_index]
                if len(chosen):
                    counts[:, signal_index] = np.bincount(chosen, minlength=len(self.item_ids))
        return counts

    def _entity_values(self, relation: str, user_id: int, timestamp: int,
                       stats_cache: dict[tuple[int, str, int | str], float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
        edge_rows, _edge_entities, unique_entities, inverse = self._mapping_edges[relation]
        store = self.engine._user_artists if relation == "artist" else self.engine._user_albums
        relation_columns = [column for column, spec in enumerate(self.engine.specs)
                            if spec["path"] == f"c_{relation}"]
        values = np.zeros((len(unique_entities), len(relation_columns)), dtype=np.float64)
        for local_column, column in enumerate(relation_columns):
            if self.engine.specs[column]["op"] == "share":
                values[:, local_column] = 0.5
        for entity_position, entity in enumerate(unique_entities):
            stats = store.get((user_id, int(entity)))
            if stats is None:
                continue
            for local_column, column in enumerate(relation_columns):
                spec = self.engine.specs[column]
                if spec["op"] == "knownshare":
                    values[entity_position, local_column] = float(stats.has_activity())
                elif spec["op"] == "count":
                    values[entity_position, local_column] = self.engine._count(
                        stats, spec["signal"], spec["window"], timestamp, stats_cache)
                elif spec["op"] == "share":
                    numerator = self.engine._count(stats, spec["signal"], spec["window"], timestamp, stats_cache)
                    denominator = self.engine._count(stats, "events", spec["window"], timestamp, stats_cache)
                    values[entity_position, local_column] = (numerator + 0.5) / (denominator + 1.0)
                elif spec["op"] == "trend":
                    width = int(spec["window"])
                    current = self.engine._count(stats, spec["signal"], width, timestamp, stats_cache)
                    two = self.engine._count(stats, spec["signal"], 2 * width, timestamp, stats_cache)
                    previous = max(0.0, two - current)
                    values[entity_position, local_column] = (current - previous) / (previous + 1.0)
                else:
                    raise ValueError(f"unsupported entity operator: {spec['op']}")
        return edge_rows, unique_entities, inverse, relation_columns, values

    def _build_relation_columns(self, relation: str, user_id: int, timestamp: int,
                                stats_cache: dict[tuple[int, str, int | str], float]) -> dict[int, np.ndarray]:
        edge_rows, _unique, inverse, columns, entity_values = self._entity_values(
            relation, user_id, timestamp, stats_cache)
        reduced_columns: dict[int, np.ndarray] = {}
        if not len(edge_rows):
            return {column: np.zeros(len(self.item_ids), dtype=np.float64) for column in columns}
        starts, ends, items, degrees = self._mapping_groups[relation]
        for local_column, column in enumerate(columns):
            reducer = self.engine.specs[column]["reducer"]
            edge_values = entity_values[inverse, local_column]
            if reducer == "max":
                reduced = np.full(len(self.item_ids), -np.inf, dtype=np.float64)
                reduced[items] = np.maximum.reduceat(edge_values, starts)
                reduced[reduced == -np.inf] = 0.0
            else:
                reduced = np.zeros(len(self.item_ids), dtype=np.float64)
                single = degrees == 1
                reduced[items[single]] = edge_values[starts[single]]
                multi = degrees > 1
                minima = np.minimum.reduceat(edge_values, starts)
                maxima = np.maximum.reduceat(edge_values, starts)
                constant = multi & (minima == maxima)
                exact_constants = constant & ((minima == 0.0) | (minima == 0.5) | (minima == 1.0))
                reduced[items[exact_constants]] = minima[exact_constants]
                repeated = constant & ~exact_constants
                for raw_degree in np.unique(degrees[repeated]):
                    degree = int(raw_degree)
                    selected = repeated & (degrees == degree)
                    distinct_values, inverse_values = np.unique(minima[selected], return_inverse=True)
                    means = np.asarray([sum(float(value) for _ in range(degree)) / degree
                                         for value in distinct_values], dtype=np.float64)
                    reduced[items[selected]] = means[inverse_values]
                # Python 3.12's float sum uses compensated accumulation. Match
                # the reference reducer exactly for heterogeneous many-to-many
                # mappings, where cancellation can change a float32 ULP. Most
                # catalog groups are constant 0/.5 defaults and use the fast path.
                for group_index in np.flatnonzero(multi & ~constant):
                    start, end = int(starts[group_index]), int(ends[group_index])
                    reduced[int(items[group_index])] = sum(
                        float(value) for value in edge_values[start:end]) / int(degrees[group_index])
            reduced_columns[column] = reduced
        return reduced_columns

    def _prepare_query(self, user_id: int, timestamp: int) -> None:
        key = (user_id, timestamp)
        if self._active_query_key == key:
            return
        if self._last_count_timestamp != timestamp:
            self._advance_item_counts(timestamp)
            self._last_count_timestamp = timestamp
        self._active_query_key = key
        self._active_pair_all = self._pair_counts(user_id, timestamp)
        self._active_pair_recent = {width: self._pair_counts(user_id, timestamp, width)
                                    for width in self._widths}
        self._active_user_values = {}
        stats_cache: dict[tuple[int, str, int | str], float] = {}
        for column, spec in enumerate(self.engine.specs):
            if spec["path"] == "u":
                self._active_user_values[column] = self.engine._one_value(
                    spec, user_id, 0, timestamp, (), (), stats_cache)
        self._active_relation_values = {}
        for relation in ("artist", "album"):
            self._active_relation_values.update(
                self._build_relation_columns(relation, user_id, timestamp, stats_cache))

    def transform(self, uid: int, item_ids: Sequence[int], timestamp: int) -> np.ndarray:
        timestamp = int(timestamp)
        self.engine._assert_query_time(timestamp)
        if self._last_query_timestamp is not None and timestamp < self._last_query_timestamp:
            raise ValueError("query timestamps must be nondecreasing")
        user_id = int(uid)
        requested_items = np.asarray([int(item) for item in item_ids], dtype=np.int64)
        output = np.zeros((len(requested_items), len(self.engine.specs)), dtype=np.float64)
        if not len(requested_items):
            self._last_query_timestamp = timestamp
            return output.astype(np.float32)
        requested_dense = np.fromiter((self.item_to_dense.get(int(item), -1) for item in requested_items),
                                      dtype=np.int64, count=len(requested_items))
        valid = requested_dense >= 0
        self._prepare_query(user_id, timestamp)

        # Item and pair paths are vectorized over the complete requested pool.
        for column, spec in enumerate(self.engine.specs):
            path, op = spec["path"], spec["op"]
            if path == "u":
                output[:, column] = self._active_user_values[column]
                continue
            if path in ("i_artist", "i_album"):
                relation = "artist" if path == "i_artist" else "album"
                degree = self._mapping_degree[relation]
                output[:, column] = np.where(valid, degree[np.maximum(requested_dense, 0)], 0.0)
                continue
            if path.startswith("c_"):
                values = self._active_relation_values[column]
                output[:, column] = np.where(valid, values[np.maximum(requested_dense, 0)], 0.0)
                continue
            if path not in ("i", "d"):
                continue
            source = self._all_counts if path == "i" else self._active_pair_all
            recent_source = self._window_counts if path == "i" else self._active_pair_recent
            if op == "age":
                if path == "i":
                    age = np.full(len(self.item_ids), -1.0, dtype=np.float64)
                    known_items = self._last_seen >= 0
                    age[known_items] = timestamp - self._last_seen[known_items]
                else:
                    times, dense, _signals = self._user_events(user_id)
                    end = int(np.searchsorted(times, timestamp, side="left"))
                    age = np.full(len(self.item_ids), -1.0, dtype=np.float64)
                    np.maximum.at(age, dense[:end], times[:end])
                    known_items = age >= 0
                    age[known_items] = timestamp - age[known_items]
                output[:, column] = np.where(valid, age[np.maximum(requested_dense, 0)], -1.0)
            elif op == "known":
                counts = source[:, 0]
                output[:, column] = np.where(valid, (counts[np.maximum(requested_dense, 0)] > 0).astype(float), 0.0)
            else:
                signal_index = _SIGNAL_INDEX[spec["signal"]]
                window = spec["window"]
                def counts_for(signal_name: str, raw_window: str | int) -> np.ndarray:
                    if raw_window == "all":
                        return source[:, _SIGNAL_INDEX[signal_name]]
                    width = self.engine.short_window if raw_window == "short" else int(raw_window)
                    return recent_source[width][:, _SIGNAL_INDEX[signal_name]]
                if op == "count":
                    values = counts_for(spec["signal"], window)
                elif op == "share":
                    numerator = counts_for(spec["signal"], window)
                    denominator = counts_for("events", window)
                    values = (numerator + 0.5) / (denominator + 1.0)
                elif op == "trend":
                    width = int(window)
                    current = counts_for(spec["signal"], width)
                    recent_two = counts_for(spec["signal"], 2 * width)
                    previous = np.maximum(0.0, recent_two - current)
                    values = (current - previous) / (previous + 1.0)
                else:
                    raise ValueError(f"unsupported direct feature operation: {op}")
                default = 0.5 if op == "share" else 0.0
                output[:, column] = np.where(valid, values[np.maximum(requested_dense, 0)], default)

        self._last_query_timestamp = timestamp
        result = output.astype(np.float32)
        if result.shape[1] != len(self.engine.feature_names):
            raise RuntimeError("internal expanded feature width mismatch")
        return result
