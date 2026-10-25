"""Causal sparse feature extraction for a chronologically observed Yambda prefix."""
from __future__ import annotations

from array import array
from bisect import bisect_right
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


SIGNALS = ("events", "listen", "like", "dislike")
EVENT_SIGNAL = {"listen": 1, "like": 2, "dislike": 3}
VALID_EVENT_TYPES = frozenset(("listen", "like", "dislike", "unlike", "undislike"))


class _Series:
    """Timestamp-sorted sparse weighted prefix sums for one signal."""

    __slots__ = ("timestamps", "prefix")

    def __init__(self) -> None:
        self.timestamps = array("Q")
        self.prefix = array("d")

    def append(self, timestamp: int, weight: float) -> None:
        self.timestamps.append(timestamp)
        previous = self.prefix[-1] if self.prefix else 0.0
        self.prefix.append(previous + weight)

    def total(self) -> float:
        return self.prefix[-1] if self.prefix else 0.0

    def recent(self, timestamp: int, window: int) -> float:
        if not self.timestamps:
            return 0.0
        # Window is (timestamp-window, timestamp); the caller only observes < timestamp.
        cutoff = timestamp - window
        index = bisect_right(self.timestamps, cutoff)
        return self.total() - (self.prefix[index - 1] if index else 0.0)


class _Stats:
    """Sparse counts and exponentially decayed counts for one entity key."""

    __slots__ = ("series", "decayed", "decay_time", "last_timestamp")

    def __init__(self) -> None:
        self.series: dict[int, _Series] = {}
        self.decayed = [0.0, 0.0, 0.0, 0.0]
        self.decay_time: int | None = None
        self.last_timestamp: int | None = None

    def add(self, timestamp: int, signal: int, weight: float, half_life: int) -> None:
        s = self.series.get(signal)
        if s is None:
            s = self.series[signal] = _Series()
        s.append(timestamp, weight)

        if self.decay_time is not None:
            delta = timestamp - self.decay_time
            if delta < 0:
                raise ValueError("events must be observed in nondecreasing timestamp order")
            if delta:
                factor = math.exp2(-delta / half_life)
                for i in range(len(self.decayed)):
                    self.decayed[i] *= factor
        self.decayed[signal] += weight
        self.decay_time = timestamp
        self.last_timestamp = timestamp

    def count(self, signal: int, timestamp: int, short_window: int) -> tuple[float, float]:
        s = self.series.get(signal)
        if s is None:
            return 0.0, 0.0
        return s.total(), s.recent(timestamp, short_window)

    def recency(self, timestamp: int) -> tuple[float, float]:
        if self.last_timestamp is None:
            return -1.0, 0.0
        return float(timestamp - self.last_timestamp), 1.0

    def decay(self, signal: int, timestamp: int, half_life: int) -> float:
        if self.decay_time is None:
            return 0.0
        delta = timestamp - self.decay_time
        if delta < 0:
            raise ValueError("query timestamp must be later than every observed event")
        return self.decayed[signal] * math.exp2(-delta / half_life)

    def has_activity(self) -> bool:
        s = self.series.get(0)
        return bool(s and s.timestamps)


class FeatureEngine:
    """Build causal U/I/D/C plus fixed half-life X features from an observed prefix.

    The caller controls the history and query construction. Every observed event must
    have timestamp strictly less than any query that uses it. Timestamps and both
    window parameters are compared in their raw source units; no calendar conversion
    or interpretation of those units is performed here.
    """

    def __init__(
        self,
        artist_map: dict[int, tuple[int, ...]],
        album_map: dict[int, tuple[int, ...]],
        short_window: int,
        half_life: int,
    ) -> None:
        if not isinstance(short_window, int) or short_window <= 0:
            raise ValueError("short_window must be a positive integer in raw timestamp units")
        if not isinstance(half_life, int) or half_life <= 0:
            raise ValueError("half_life must be a positive integer in raw timestamp units")
        self.short_window = short_window
        self.half_life = half_life
        # Mappings are treated as fixed snapshots. Deduplicate and sort to make
        # many-to-many fractional credit deterministic and immune to repeated edges.
        self.artist_map = self._normalize_map(artist_map)
        self.album_map = self._normalize_map(album_map)

        self._users: dict[int, _Stats] = {}
        self._items: dict[int, _Stats] = {}
        self._pairs: dict[tuple[int, int], _Stats] = {}
        self._user_artists: dict[tuple[int, int], _Stats] = {}
        self._user_albums: dict[tuple[int, int], _Stats] = {}
        self._last_observed_timestamp: int | None = None

        self.feature_names: list[str] = []
        self.feature_groups: dict[str, str] = {}
        self._names_by_group: dict[str, list[str]] = {g: [] for g in ("U", "I", "D", "C", "X")}
        self._define_features()

    @staticmethod
    def _normalize_map(mapping: Mapping[int, Sequence[int]]) -> dict[int, tuple[int, ...]]:
        normalized: dict[int, tuple[int, ...]] = {}
        for item, entities in mapping.items():
            normalized[int(item)] = tuple(sorted({int(entity) for entity in entities}))
        return normalized

    def _add_name(self, name: str, group: str) -> None:
        self.feature_names.append(name)
        self.feature_groups[name] = group
        self._names_by_group[group].append(name)

    def _define_features(self) -> None:
        # Names and order are API: U, then I, D, C, X; definitions below use same order.
        for prefix, group in (("u", "U"), ("i", "I"), ("d", "D")):
            for window in ("all", "short"):
                for signal in SIGNALS:
                    self._add_name(f"{prefix}_{signal}_{window}", group)
            self._add_name(f"{prefix}_last_age", group)
            self._add_name(f"{prefix}_known", group)
            if prefix == "i":
                self._add_name("i_artist_degree", group)
                self._add_name("i_album_degree", group)

        for relation in ("artist", "album"):
            for window in ("all", "short"):
                for signal in SIGNALS:
                    for aggregate in ("max", "mean"):
                        self._add_name(f"c_{relation}_{signal}_{window}_{aggregate}", "C")
            self._add_name(f"c_{relation}_knownshare", "C")

        for prefix in ("i", "d"):
            for signal in SIGNALS:
                self._add_name(f"x_{prefix}_{signal}_half_life", "X")
        for relation in ("artist", "album"):
            for signal in SIGNALS:
                for aggregate in ("max", "mean"):
                    self._add_name(f"x_c_{relation}_{signal}_half_life_{aggregate}", "X")
            self._add_name(f"x_c_{relation}_knownshare_half_life", "X")

    @staticmethod
    def _get_stats(store: dict[Any, _Stats], key: Any) -> _Stats:
        stats = store.get(key)
        if stats is None:
            stats = store[key] = _Stats()
        return stats

    def observe(self, event: dict[str, Any]) -> None:
        """Append one event to the causal prefix; input timestamps must be nondecreasing."""
        try:
            user_id = int(event["uid"])
            item_id = int(event["item_id"])
            timestamp = int(event["timestamp"])
            event_type = str(event["event_type"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("event must contain integer uid/item_id/timestamp and event_type") from exc
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"unsupported event_type: {event_type!r}")
        if timestamp < 0:
            raise ValueError("timestamp must be nonnegative")
        if self._last_observed_timestamp is not None and timestamp < self._last_observed_timestamp:
            raise ValueError("events must be observed in nondecreasing timestamp order")

        # Every event contributes to activity. Only listen/like/dislike receive separate
        # response counts; unlike/undislike remain represented by activity and recency.
        signals = (0, EVENT_SIGNAL[event_type]) if event_type in EVENT_SIGNAL else (0,)
        weight = 1.0
        self._get_stats(self._users, user_id)
        self._get_stats(self._items, item_id)
        self._get_stats(self._pairs, (user_id, item_id))
        targets = [
            (self._users, user_id, weight),
            (self._items, item_id, weight),
            (self._pairs, (user_id, item_id), weight),
        ]
        artists = self.artist_map.get(item_id, ())
        albums = self.album_map.get(item_id, ())
        for entity_id in artists:
            targets.append((self._user_artists, (user_id, entity_id), 1.0 / len(artists)))
        for entity_id in albums:
            targets.append((self._user_albums, (user_id, entity_id), 1.0 / len(albums)))
        for store, key, contribution in targets:
            stats = self._get_stats(store, key)
            for signal in signals:
                stats.add(timestamp, signal, contribution, self.half_life)
        self._last_observed_timestamp = timestamp

    def _assert_query_time(self, timestamp: int) -> None:
        if not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError("timestamp must be a nonnegative integer")
        if self._last_observed_timestamp is not None and self._last_observed_timestamp >= timestamp:
            raise ValueError("all observed events must be strictly earlier than query timestamp")

    def _append_counts(self, output: list[float], stats: _Stats | None, timestamp: int) -> None:
        for window in ("all", "short"):
            for signal in range(len(SIGNALS)):
                all_count, short_count = (0.0, 0.0) if stats is None else stats.count(signal, timestamp, self.short_window)
                output.append(all_count if window == "all" else short_count)
        age, known = (-1.0, 0.0) if stats is None else stats.recency(timestamp)
        output.extend((age, known))

    def _append_entity_features(
        self,
        output: list[float],
        store: dict[tuple[int, int], _Stats],
        user_id: int,
        entity_ids: tuple[int, ...],
        timestamp: int,
    ) -> None:
        related = [store.get((user_id, entity_id)) for entity_id in entity_ids]
        for window in ("all", "short"):
            for signal in range(len(SIGNALS)):
                values = []
                for stats in related:
                    all_count, short_count = (0.0, 0.0) if stats is None else stats.count(signal, timestamp, self.short_window)
                    values.append(all_count if window == "all" else short_count)
                output.append(max(values, default=0.0))
                output.append(sum(values) / len(values) if values else 0.0)
        known = sum(bool(stats and stats.has_activity()) for stats in related)
        output.append(known / len(entity_ids) if entity_ids else 0.0)

    def _append_decay_aggregate(
        self,
        output: list[float],
        store: dict[tuple[int, int], _Stats],
        user_id: int,
        entity_ids: tuple[int, ...],
        signal: int,
        timestamp: int,
    ) -> None:
        values = [
            stats.decay(signal, timestamp, self.half_life) if stats else 0.0
            for entity_id in entity_ids
            for stats in (store.get((user_id, entity_id)),)
        ]
        output.append(max(values, default=0.0))
        output.append(sum(values) / len(values) if values else 0.0)

    def _append_decay_knownshare(
        self,
        output: list[float],
        store: dict[tuple[int, int], _Stats],
        user_id: int,
        entity_ids: tuple[int, ...],
        timestamp: int,
    ) -> None:
        # Shared knownness is based on any historical event, not one flag per signal.
        known = sum(
            (stats := store.get((user_id, entity_id))) is not None
            and stats.decay(0, timestamp, self.half_life) > 0.0
            for entity_id in entity_ids
        )
        output.append(known / len(entity_ids) if entity_ids else 0.0)

    def transform(self, user_id: int, item_ids: list[int], timestamp: int) -> np.ndarray:
        """Return float32 features [len(item_ids), n_features] without changing history."""
        self._assert_query_time(timestamp)
        user_id = int(user_id)
        rows: list[list[float]] = []
        user_stats = self._users.get(user_id)
        for raw_item_id in item_ids:
            item_id = int(raw_item_id)
            item_stats = self._items.get(item_id)
            pair_stats = self._pairs.get((user_id, item_id))
            artists = self.artist_map.get(item_id, ())
            albums = self.album_map.get(item_id, ())
            row: list[float] = []

            self._append_counts(row, user_stats, timestamp)
            self._append_counts(row, item_stats, timestamp)
            row.extend((float(len(artists)), float(len(albums))))
            self._append_counts(row, pair_stats, timestamp)

            self._append_entity_features(row, self._user_artists, user_id, artists, timestamp)
            self._append_entity_features(row, self._user_albums, user_id, albums, timestamp)

            for stats in (item_stats, pair_stats):
                for signal in range(len(SIGNALS)):
                    row.append(0.0 if stats is None else stats.decay(signal, timestamp, self.half_life))
            for signal in range(len(SIGNALS)):
                self._append_decay_aggregate(row, self._user_artists, user_id, artists, signal, timestamp)
            self._append_decay_knownshare(row, self._user_artists, user_id, artists, timestamp)
            for signal in range(len(SIGNALS)):
                self._append_decay_aggregate(row, self._user_albums, user_id, albums, signal, timestamp)
            self._append_decay_knownshare(row, self._user_albums, user_id, albums, timestamp)
            rows.append(row)

        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        result = np.asarray(rows, dtype=np.float32)
        if result.shape[1] != len(self.feature_names):
            raise RuntimeError("internal feature order mismatch")
        return result
