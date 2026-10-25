"""Small, deterministic DSL for causal automated U/I/D/C feature expressions.

The generated A1 family mirrors the non-decayed P001 U/I/D/C control. A2 adds
only bounded raw-window counts, smoothed response shares and two-window trends.
All state and event semantics come from :mod:`pilot_features`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

if __package__:
    from .pilot_features import SIGNALS, FeatureEngine
else:  # Support direct use when src/ is placed on sys.path.
    from pilot_features import SIGNALS, FeatureEngine


_SIGNAL_INDEX = {signal: index for index, signal in enumerate(SIGNALS)}
_RESPONSES = ("listen", "like", "dislike")
_DIRECT_PATHS = (("u", "U"), ("i", "I"), ("d", "D"))
_RELATIONS = (("artist", "C"), ("album", "C"))
_OPS = frozenset(("count", "age", "known", "degree", "knownshare", "share", "trend"))


def _spec(
    feature_id: str,
    group: str,
    family: str,
    op: str,
    path: str,
    signal: str | None,
    window: str | int | None,
    reducer: str | None,
    provenance: str,
) -> dict[str, Any]:
    return {
        "feature_id": feature_id,
        "group": group,
        "family": family,
        "op": op,
        "path": path,
        "signal": signal,
        "window": window,
        "reducer": reducer,
        "provenance": provenance,
    }


def generate_specs(
    windows: Sequence[int] = (201117, 670390, 2011171),
    baseline_window: int = 670390,
) -> list[dict[str, Any]]:
    """Generate A1 (66 manual-equivalent) and bounded A2 (206 default) specs.

    ``windows`` are raw timestamp widths. ``baseline_window`` must be one of
    them. The baseline ``short`` columns retain their legacy names and are
    interpreted using the engine's configured ``short_window``; numeric A2
    windows remain explicit in their IDs and provenance.
    """
    if any(not isinstance(window, int) or isinstance(window, bool) for window in windows):
        raise ValueError("windows must contain integer raw timestamp widths")
    normalized = tuple(windows)
    if not normalized or any(window <= 0 for window in normalized):
        raise ValueError("windows must contain positive raw timestamp widths")
    if len(set(normalized)) != len(normalized):
        raise ValueError("windows must be unique")
    if not isinstance(baseline_window, int) or baseline_window <= 0:
        raise ValueError("baseline_window must be a positive integer")
    if baseline_window not in normalized:
        raise ValueError("baseline_window must be included in windows")

    specs: list[dict[str, Any]] = []

    # A1: preserve the exact legacy feature names and order, excluding X.
    for path, group in _DIRECT_PATHS:
        for window in ("all", "short"):
            for signal in SIGNALS:
                specs.append(_spec(
                    f"{path}_{signal}_{window}", group, "A1", "count", path,
                    signal, window, None, "P001 U/I/D count grammar",
                ))
        specs.append(_spec(f"{path}_last_age", group, "A1", "age", path, None, None, None,
                           "P001 last-event age"))
        specs.append(_spec(f"{path}_known", group, "A1", "known", path, None, None, None,
                           "P001 any-event-known indicator"))
        if path == "i":
            specs.append(_spec("i_artist_degree", group, "A1", "degree", "i_artist", None,
                               None, None, "unique fixed-snapshot artist mapping degree"))
            specs.append(_spec("i_album_degree", group, "A1", "degree", "i_album", None,
                               None, None, "unique fixed-snapshot album mapping degree"))

    for relation, group in _RELATIONS:
        path = f"c_{relation}"
        for window in ("all", "short"):
            for signal in SIGNALS:
                for reducer in ("max", "mean"):
                    specs.append(_spec(
                        f"{path}_{signal}_{window}_{reducer}", group, "A1", "count", path,
                        signal, window, reducer, "P001 fractional entity count then reducer",
                    ))
        specs.append(_spec(f"{path}_knownshare", group, "A1", "knownshare", path, None,
                           None, None, "P001 mapped-entity historical-known fraction"))

    # A2 count expressions: U/I/D, plus C entity max/mean, for each extra window.
    extra_windows = tuple(window for window in normalized if window != baseline_window)
    for path, group in _DIRECT_PATHS:
        for window in extra_windows:
            for signal in SIGNALS:
                specs.append(_spec(
                    f"{path}_{signal}_w{window}", group, "A2", "count", path, signal,
                    window, None, "causal count in explicit raw window",
                ))
    for relation, group in _RELATIONS:
        path = f"c_{relation}"
        for window in extra_windows:
            for signal in SIGNALS:
                for reducer in ("max", "mean"):
                    specs.append(_spec(
                        f"{path}_{signal}_w{window}_{reducer}", group, "A2", "count", path,
                        signal, window, reducer, "fractional entity window count then reducer",
                    ))

    # Smoothed event-response shares; C ratios are formed per entity first.
    for path, group in _DIRECT_PATHS:
        for window, suffix in (("all", "all"), (baseline_window, f"w{baseline_window}")):
            for signal in _RESPONSES:
                specs.append(_spec(
                    f"{path}_{signal}_share_{suffix}", group, "A2", "share", path, signal,
                    window, None, "(response_count + 0.5) / (event_count + 1)",
                ))
    for relation, group in _RELATIONS:
        path = f"c_{relation}"
        for window, suffix in (("all", "all"), (baseline_window, f"w{baseline_window}")):
            for signal in _RESPONSES:
                for reducer in ("max", "mean"):
                    specs.append(_spec(
                        f"{path}_{signal}_share_{suffix}_{reducer}", group, "A2", "share",
                        path, signal, window, reducer,
                        "per-entity smoothed share then max/mean",
                    ))

    # Compare the current and immediately preceding equal-width raw windows.
    for path, group in _DIRECT_PATHS:
        for window in normalized:
            for signal in ("events", "listen"):
                specs.append(_spec(
                    f"{path}_{signal}_trend_w{window}", group, "A2", "trend", path,
                    signal, window, None, "(current - previous) / (previous + 1)",
                ))
    for relation, group in _RELATIONS:
        path = f"c_{relation}"
        for window in normalized:
            for signal in ("events", "listen"):
                for reducer in ("max", "mean"):
                    specs.append(_spec(
                        f"{path}_{signal}_trend_w{window}_{reducer}", group, "A2", "trend",
                        path, signal, window, reducer,
                        "per-entity trend ratio then max/mean",
                    ))

    return specs


class AutoFeatureEngine(FeatureEngine):
    """Materialize generated feature specs over inherited causal sparse state."""

    def __init__(
        self,
        artist_map: Mapping[int, Sequence[int]],
        album_map: Mapping[int, Sequence[int]],
        short_window: int,
        half_life: int,
        *,
        specs: Sequence[Mapping[str, Any]],
    ) -> None:
        super().__init__(dict(artist_map), dict(album_map), short_window, half_life)
        if not specs:
            raise ValueError("specs must not be empty")
        self.specs = tuple(dict(spec) for spec in specs)
        ids: set[str] = set()
        expected_groups = {
            "u": "U", "i": "I", "d": "D", "c_artist": "C", "c_album": "C",
            "i_artist": "I", "i_album": "I",
        }
        for spec in self.specs:
            required = {"feature_id", "group", "family", "op", "path", "signal", "window", "reducer", "provenance"}
            missing = required.difference(spec)
            if missing:
                raise ValueError(f"feature spec missing fields: {sorted(missing)}")
            feature_id = spec["feature_id"]
            if not isinstance(feature_id, str) or not feature_id or feature_id in ids:
                raise ValueError(f"feature IDs must be nonempty and unique: {feature_id!r}")
            ids.add(feature_id)
            if spec["group"] not in {"U", "I", "D", "C"}:
                raise ValueError(f"invalid feature group: {spec['group']!r}")
            if spec["family"] not in {"A1", "A2"}:
                raise ValueError(f"invalid feature family: {spec['family']!r}")
            if spec["op"] not in _OPS:
                raise ValueError(f"invalid feature operation: {spec['op']!r}")
            if spec["signal"] is not None and spec["signal"] not in SIGNALS:
                raise ValueError(f"invalid feature signal: {spec['signal']!r}")
            path = spec["path"]
            op = spec["op"]
            signal = spec["signal"]
            window = spec["window"]
            reducer = spec["reducer"]
            if path not in expected_groups or spec["group"] != expected_groups[path]:
                raise ValueError(f"feature path/group mismatch: {path!r}/{spec['group']!r}")
            if reducer not in (None, "max", "mean"):
                raise ValueError(f"invalid feature reducer: {reducer!r}")
            if not isinstance(spec["provenance"], str) or not spec["provenance"]:
                raise ValueError("feature provenance must be a nonempty string")

            direct = path in ("u", "i", "d")
            relation = path in ("c_artist", "c_album")
            if op == "degree":
                valid = path in ("i_artist", "i_album") and signal is None and window is None and reducer is None
            elif op in ("age", "known"):
                valid = direct and signal is None and window is None and reducer is None
            elif op == "knownshare":
                valid = relation and signal is None and window is None and reducer is None
            elif op == "count":
                valid_window = window in ("all", "short") or (
                    isinstance(window, int) and not isinstance(window, bool) and window > 0
                )
                valid = (direct or relation) and signal in SIGNALS and valid_window
                valid = valid and (reducer in ("max", "mean") if relation else reducer is None)
            elif op == "share":
                valid_window = window == "all" or (
                    isinstance(window, int) and not isinstance(window, bool) and window > 0
                )
                valid = (direct or relation) and signal in _RESPONSES and valid_window
                valid = valid and (reducer in ("max", "mean") if relation else reducer is None)
            else:  # trend
                valid = (direct or relation) and signal in ("events", "listen")
                valid = valid and isinstance(window, int) and not isinstance(window, bool) and window > 0
                valid = valid and (reducer in ("max", "mean") if relation else reducer is None)
            if not valid:
                raise ValueError(f"invalid {op!r} expression for feature {feature_id!r}")

        self.feature_names = [str(spec["feature_id"]) for spec in self.specs]
        self.feature_groups = {str(spec["feature_id"]): str(spec["group"]) for spec in self.specs}
        self._names_by_group = {group: [] for group in ("U", "I", "D", "C")}
        for spec in self.specs:
            self._names_by_group[spec["group"]].append(spec["feature_id"])

    @staticmethod
    def _store_and_key(
        path: str,
        user_id: int,
        item_id: int,
        artist_ids: tuple[int, ...],
        album_ids: tuple[int, ...],
        engine: "AutoFeatureEngine",
    ) -> tuple[dict[Any, Any], Any] | tuple[dict[Any, Any], tuple[int, ...]]:
        if path == "u":
            return engine._users, user_id
        if path == "i":
            return engine._items, item_id
        if path == "d":
            return engine._pairs, (user_id, item_id)
        if path == "c_artist":
            return engine._user_artists, tuple((user_id, entity) for entity in artist_ids)
        if path == "c_album":
            return engine._user_albums, tuple((user_id, entity) for entity in album_ids)
        if path == "i_artist":
            return {}, artist_ids
        if path == "i_album":
            return {}, album_ids
        raise ValueError(f"invalid feature path: {path!r}")

    def _count(
        self,
        stats: Any,
        signal: str,
        window: str | int,
        timestamp: int,
        cache: dict[tuple[int, str, str | int], float],
    ) -> float:
        cache_window: str | int = "all" if window == "all" else (self.short_window if window == "short" else int(window))
        if stats is None:
            return 0.0
        key = (id(stats), signal, cache_window)
        value = cache.get(key)
        if value is None:
            all_count, recent_count = stats.count(_SIGNAL_INDEX[signal], timestamp, self.short_window if window == "short" else int(cache_window) if cache_window != "all" else self.short_window)
            value = all_count if window == "all" else recent_count
            cache[key] = value
        return value

    @staticmethod
    def _reduce(values: Sequence[float], reducer: str | None) -> float:
        if not values:
            return 0.0
        if reducer == "max":
            return max(values)
        if reducer == "mean":
            return sum(values) / len(values)
        if reducer is None:
            return values[0]
        raise ValueError(f"invalid reducer: {reducer!r}")

    def _one_value(
        self,
        spec: Mapping[str, Any],
        user_id: int,
        item_id: int,
        timestamp: int,
        artist_ids: tuple[int, ...],
        album_ids: tuple[int, ...],
        cache: dict[tuple[int, str, str | int], float],
    ) -> float:
        path = spec["path"]
        store, key_or_keys = self._store_and_key(path, user_id, item_id, artist_ids, album_ids, self)
        if path in ("i_artist", "i_album"):
            return float(len(key_or_keys))
        is_relation = path in ("c_artist", "c_album")
        if is_relation:
            stats_list = [store.get(key) for key in key_or_keys]
        else:
            stats = store.get(key_or_keys)
            stats_list = [stats]
        op = spec["op"]
        signal = spec["signal"]

        if op == "age":
            age, _known = (-1.0, 0.0) if stats_list[0] is None else stats_list[0].recency(timestamp)
            return float(age)
        if op == "known":
            return float(stats_list[0] is not None and stats_list[0].has_activity())
        if op == "knownshare":
            entities = key_or_keys
            if not entities:
                return 0.0
            return sum(bool(stats and stats.has_activity()) for stats in stats_list) / len(entities)
        if op == "degree":
            raise ValueError("degree specs must use i_artist or i_album path")

        window = spec["window"]
        reducer = spec["reducer"]
        if op == "count":
            values = [self._count(entity_stats, signal, window, timestamp, cache) for entity_stats in stats_list]
            return self._reduce(values, reducer)
        if op == "share":
            values = []
            for entity_stats in stats_list:
                numerator = self._count(entity_stats, signal, window, timestamp, cache)
                denominator = self._count(entity_stats, "events", window, timestamp, cache)
                values.append((numerator + 0.5) / (denominator + 1.0))
            return self._reduce(values, reducer)
        if op == "trend":
            width = int(window)
            values = []
            for entity_stats in stats_list:
                current = self._count(entity_stats, signal, width, timestamp, cache)
                recent_two = self._count(entity_stats, signal, 2 * width, timestamp, cache)
                previous = max(0.0, recent_two - current)
                values.append((current - previous) / (previous + 1.0))
            return self._reduce(values, reducer)
        raise ValueError(f"unsupported feature operation: {op!r}")

    def transform(self, user_id: int, item_ids: list[int], timestamp: int) -> np.ndarray:
        """Return generated float32 columns for an unchanged causal prefix."""
        self._assert_query_time(timestamp)
        user_id = int(user_id)
        rows: list[list[float]] = []
        cache: dict[tuple[int, str, str | int], float] = {}
        for raw_item_id in item_ids:
            item_id = int(raw_item_id)
            artist_ids = self.artist_map.get(item_id, ())
            album_ids = self.album_map.get(item_id, ())
            rows.append([
                self._one_value(spec, user_id, item_id, timestamp, artist_ids, album_ids, cache)
                for spec in self.specs
            ])
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        result = np.asarray(rows, dtype=np.float32)
        if result.shape[1] != len(self.feature_names):
            raise RuntimeError("internal automatic feature order mismatch")
        return result

