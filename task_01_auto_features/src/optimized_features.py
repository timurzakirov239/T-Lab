"""Value-preserving query-local optimizations for the causal pilot feature engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

if __package__:
    from .pilot_features import FeatureEngine, SIGNALS, _Stats
else:
    from pilot_features import FeatureEngine, SIGNALS, _Stats


@dataclass(frozen=True)
class _EntityPayload:
    """One transform-local snapshot of a user-entity's causal values."""

    counts: tuple[tuple[float, float], ...]  # per signal: (all, short)
    decays: tuple[float, ...]
    has_activity: bool
    decay_known: bool


class OptimizedFeatureEngine(FeatureEngine):
    """FeatureEngine-compatible subclass with query-scoped redundant-work reuse.

    The causal state, mapping normalization, feature contract and arithmetic helpers
    are inherited from FeatureEngine. No cache survives a transform call.
    """

    def _user_values(self, stats: _Stats | None, timestamp: int) -> list[float]:
        if stats is None:
            counts = ((0.0, 0.0),) * len(SIGNALS)
            age, known = -1.0, 0.0
        else:
            # One Stats.count returns both values. Preserve output layout: all signals,
            # then short-window signals, followed by recency and known.
            counts = tuple(stats.count(signal, timestamp, self.short_window) for signal in range(len(SIGNALS)))
            age, known = stats.recency(timestamp)
        return [pair[0] for pair in counts] + [pair[1] for pair in counts] + [age, known]

    def _count_values(self, output: list[float], stats: _Stats | None, timestamp: int) -> None:
        if stats is None:
            counts = ((0.0, 0.0),) * len(SIGNALS)
            age, known = -1.0, 0.0
        else:
            counts = tuple(stats.count(signal, timestamp, self.short_window) for signal in range(len(SIGNALS)))
            age, known = stats.recency(timestamp)
        output.extend(pair[0] for pair in counts)
        output.extend(pair[1] for pair in counts)
        output.extend((age, known))

    def _entity_payload(
        self,
        store: dict[tuple[int, int], _Stats],
        user_id: int,
        entity_id: int,
        timestamp: int,
        cache: dict[int, _EntityPayload],
    ) -> _EntityPayload:
        payload = cache.get(entity_id)
        if payload is None:
            stats = store.get((user_id, entity_id))
            if stats is None:
                payload = _EntityPayload(
                    counts=((0.0, 0.0),) * len(SIGNALS),
                    decays=(0.0,) * len(SIGNALS),
                    has_activity=False,
                    decay_known=False,
                )
            else:
                counts = tuple(stats.count(signal, timestamp, self.short_window) for signal in range(len(SIGNALS)))
                decays = tuple(stats.decay(signal, timestamp, self.half_life) for signal in range(len(SIGNALS)))
                payload = _EntityPayload(
                    counts=counts,
                    decays=decays,
                    has_activity=stats.has_activity(),
                    # Preserve X knownshare's strict comparison, including underflow.
                    decay_known=decays[0] > 0.0,
                )
            cache[entity_id] = payload
        return payload

    @staticmethod
    def _append_relation_counts(
        output: list[float],
        entity_ids: tuple[int, ...],
        payloads: dict[int, _EntityPayload],
    ) -> None:
        related = [payloads[entity_id] for entity_id in entity_ids]
        # Match base order exactly: all signal aggregates, then short aggregates.
        for window_index in (0, 1):
            for signal in range(len(SIGNALS)):
                values = [payload.counts[signal][window_index] for payload in related]
                output.append(max(values, default=0.0))
                output.append(sum(values) / len(values) if values else 0.0)
        known = sum(payload.has_activity for payload in related)
        output.append(known / len(entity_ids) if entity_ids else 0.0)

    @staticmethod
    def _append_relation_decays(
        output: list[float],
        entity_ids: tuple[int, ...],
        payloads: dict[int, _EntityPayload],
    ) -> None:
        related = [payloads[entity_id] for entity_id in entity_ids]
        for signal in range(len(SIGNALS)):
            values = [payload.decays[signal] for payload in related]
            output.append(max(values, default=0.0))
            output.append(sum(values) / len(values) if values else 0.0)
        known = sum(payload.decay_known for payload in related)
        output.append(known / len(entity_ids) if entity_ids else 0.0)

    def transform(self, user_id: int, item_ids: list[int], timestamp: int) -> np.ndarray:
        """Return the baseline float32 matrix without mutating observed prefix state."""
        self._assert_query_time(timestamp)
        user_id = int(user_id)
        user_values = self._user_values(self._users.get(user_id), timestamp)
        # These caches exist only inside this query; they cannot leak values between
        # users, timestamps, or state updates.
        artist_payloads: dict[int, _EntityPayload] = {}
        album_payloads: dict[int, _EntityPayload] = {}
        rows: list[list[float]] = []

        for raw_item_id in item_ids:
            item_id = int(raw_item_id)
            item_stats = self._items.get(item_id)
            pair_stats = self._pairs.get((user_id, item_id))
            artists = self.artist_map.get(item_id, ())
            albums = self.album_map.get(item_id, ())
            row: list[float] = []

            row.extend(user_values)
            self._count_values(row, item_stats, timestamp)
            row.extend((float(len(artists)), float(len(albums))))
            self._count_values(row, pair_stats, timestamp)

            for entity_id in artists:
                self._entity_payload(self._user_artists, user_id, entity_id, timestamp, artist_payloads)
            self._append_relation_counts(row, artists, artist_payloads)
            for entity_id in albums:
                self._entity_payload(self._user_albums, user_id, entity_id, timestamp, album_payloads)
            self._append_relation_counts(row, albums, album_payloads)

            for stats in (item_stats, pair_stats):
                for signal in range(len(SIGNALS)):
                    row.append(0.0 if stats is None else stats.decay(signal, timestamp, self.half_life))
            self._append_relation_decays(row, artists, artist_payloads)
            self._append_relation_decays(row, albums, album_payloads)
            rows.append(row)

        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        result = np.asarray(rows, dtype=np.float32)
        if result.shape[1] != len(self.feature_names):
            raise RuntimeError("internal feature order mismatch")
        return result
