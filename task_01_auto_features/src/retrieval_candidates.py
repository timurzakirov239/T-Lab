"""Deterministic causal retrieval candidates and source diagnostics for P002."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import math
from typing import Any, Mapping, Sequence


SOURCE_ORDER = ("popular", "repeat", "artist", "album")
BASELINE_QUOTAS = {"popular": 25, "repeat": 25, "artist": 35, "album": 15}
REPEAT_HEAVY_QUOTAS = {"popular": 10, "repeat": 60, "artist": 20, "album": 10}
AFFINITY_QUOTAS = {"popular": 10, "repeat": 20, "artist": 50, "album": 20}
STRATEGIES = ("baseline", "repeat_heavy", "affinity", "rank_fusion")


@dataclass(frozen=True)
class CandidateResult:
    """Candidates plus ranked source provenance and before-quota diagnostics."""

    candidate_items: list[int]
    source_pools: dict[str, list[int]]
    provenance: dict[int, list[str]]
    quota_contributions: dict[str, int]
    selected_by_source: dict[str, list[int]]
    selection_sources: dict[int, str | None]
    fallback_items: list[int]
    fallback_contributions: dict[str, int]
    overlaps: dict[str, int]
    rrf_scores: dict[int, float]


class RetrievalCandidateGenerator:
    """Online past-only retrieval with fixed quota variants and RRF.

    The event contract matches P001 CandidateGenerator: every observed object enters
    the cohort catalog; only listens affect popularity, recency and user profiles.
    Candidate queries must have timestamp strictly greater than the last observed
    event timestamp. Equal-timestamp events may all be observed before a later query,
    but no query can see only an arbitrary subset of that equal-timestamp group.
    """

    def __init__(
        self,
        artist_map: Mapping[int, Sequence[int]],
        album_map: Mapping[int, Sequence[int]],
        max_candidates: int = 100,
        candidate_quotas: Mapping[str, int] | None = None,
        source_depth: int = 200,
        baseline_profile_depth: int = 5,
        affinity_profile_depth: int = 32,
    ) -> None:
        if max_candidates <= 0 or source_depth <= 0:
            raise ValueError("max_candidates and source_depth must be positive")
        if baseline_profile_depth <= 0 or affinity_profile_depth <= 0:
            raise ValueError("profile depths must be positive")
        self.max_candidates = int(max_candidates)
        self.source_depth = int(source_depth)
        self.baseline_profile_depth = int(baseline_profile_depth)
        self.affinity_profile_depth = int(affinity_profile_depth)
        self.candidate_quotas = dict(candidate_quotas or BASELINE_QUOTAS)
        if set(self.candidate_quotas) != set(SOURCE_ORDER):
            raise ValueError(f"candidate_quotas must have keys {SOURCE_ORDER}")
        if any(int(v) < 0 for v in self.candidate_quotas.values()):
            raise ValueError("candidate quota values must be nonnegative")

        self.artists = self._normalize_map(artist_map)
        self.albums = self._normalize_map(album_map)
        self.counts: Counter[int] = Counter()
        self.last_listen: dict[int, dict[int, int]] = defaultdict(dict)
        self.user_artists: dict[int, Counter[int]] = defaultdict(Counter)
        self.user_albums: dict[int, Counter[int]] = defaultdict(Counter)
        self.artist_items: dict[int, set[int]] = defaultdict(set)
        self.album_items: dict[int, set[int]] = defaultdict(set)
        for item, entities in self.artists.items():
            for entity in entities:
                self.artist_items[entity].add(item)
        for item, entities in self.albums.items():
            for entity in entities:
                self.album_items[entity].add(item)
        self.catalog: set[int] = set()
        self.previous_t: int | None = None
        self._pool_cache_key: tuple[int, int] | None = None
        self._pool_cache: dict[str, dict[str, list[int]]] = {}

    @staticmethod
    def _normalize_map(mapping: Mapping[int, Sequence[int]]) -> dict[int, tuple[int, ...]]:
        return {int(item): tuple(sorted({int(entity) for entity in entities})) for item, entities in mapping.items()}

    @staticmethod
    def top(counter: Mapping[int, float], limit: int) -> list[int]:
        # Identical deterministic score-descending, key-ascending ordering as P001.
        return [int(key) for key, _ in heapq.nsmallest(limit, counter.items(), key=lambda kv: (-kv[1], kv[0]))]

    def observe(self, event: Mapping[str, Any]) -> None:
        """Observe one logged event; unlike/undislike affect catalog only, as in P001."""
        try:
            uid = int(event["uid"])
            item = int(event["item_id"])
            timestamp = int(event["timestamp"])
            event_type = str(event["event_type"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("event must contain uid, item_id, timestamp, event_type") from exc
        if timestamp < 0:
            raise ValueError("timestamp must be nonnegative")
        if self.previous_t is not None and timestamp < self.previous_t:
            raise ValueError("events must be observed in nondecreasing timestamp order")
        self.catalog.add(item)
        if event_type == "listen":
            self.counts[item] += 1
            self.last_listen[uid][item] = timestamp
            for entities, profile in (
                (self.artists.get(item, ()), self.user_artists[uid]),
                (self.albums.get(item, ()), self.user_albums[uid]),
            ):
                if entities:
                    credit = 1.0 / len(entities)
                    for entity in entities:
                        profile[entity] += credit
        self.previous_t = timestamp
        # Even a same-timestamp observation invalidates query-pool cache state.
        self._pool_cache_key = None
        self._pool_cache = {}

    def _related(
        self,
        profile: Mapping[int, float],
        entity_items: Mapping[int, set[int]],
        limit: int,
        profile_depth: int,
    ) -> list[int]:
        scores: defaultdict[int, float] = defaultdict(float)
        for entity in self.top(profile, profile_depth):
            for item in entity_items.get(entity, ()):
                if item in self.catalog and self.counts[item] > 0:
                    scores[item] += float(profile[entity]) * math.log1p(self.counts[item])
        return self.top(scores, limit)

    def _source_pool_variants(self, uid: int, timestamp: int) -> dict[str, dict[str, list[int]]]:
        key = (int(uid), int(timestamp))
        if self._pool_cache_key == key:
            return self._pool_cache
        popular = self.top(self.counts, self.source_depth)
        repeat = self.top(self.last_listen.get(int(uid), {}), self.source_depth)
        baseline_artist = self._related(
            self.user_artists.get(int(uid), {}), self.artist_items, self.source_depth, self.baseline_profile_depth
        )
        baseline_album = self._related(
            self.user_albums.get(int(uid), {}), self.album_items, self.source_depth, self.baseline_profile_depth
        )
        affinity_artist = self._related(
            self.user_artists.get(int(uid), {}), self.artist_items, self.source_depth, self.affinity_profile_depth
        )
        affinity_album = self._related(
            self.user_albums.get(int(uid), {}), self.album_items, self.source_depth, self.affinity_profile_depth
        )
        baseline = {"popular": popular, "repeat": repeat, "artist": baseline_artist, "album": baseline_album}
        affinity = {"popular": popular, "repeat": repeat, "artist": affinity_artist, "album": affinity_album}
        self._pool_cache = {"baseline": baseline, "affinity": affinity}
        self._pool_cache_key = key
        return self._pool_cache

    def source_pools(self, uid: int, timestamp: int, strategy: str = "baseline") -> dict[str, list[int]]:
        """Return depth-limited ranked pools before quotas are applied.

        Pools are cached per (uid, timestamp); baseline/repeat/RRF share top-5
        profile pools while affinity uses the documented top-32 profile variant.
        """
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
        if not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError("timestamp must be a nonnegative integer")
        if self.previous_t is not None and self.previous_t >= timestamp:
            raise ValueError("Candidate history must be strictly before query")
        pool_variant = "affinity" if strategy == "affinity" else "baseline"
        cached = self._source_pool_variants(uid, timestamp)[pool_variant]
        return {source: list(cached[source]) for source in SOURCE_ORDER}

    @staticmethod
    def _diagnostics(source_pools: Mapping[str, Sequence[int]]) -> tuple[dict[int, list[str]], dict[str, int]]:
        source_by_item: defaultdict[int, list[str]] = defaultdict(list)
        for source in SOURCE_ORDER:
            for item in source_pools.get(source, ()):
                source_by_item[int(item)].append(source)
        provenance = {item: list(sources) for item, sources in source_by_item.items()}
        overlaps: dict[str, int] = {
            "union_items": len(source_by_item),
            "multi_source_items": sum(len(sources) > 1 for sources in source_by_item.values()),
        }
        for i, left in enumerate(SOURCE_ORDER):
            left_items = set(source_pools.get(left, ()))
            for right in SOURCE_ORDER[i + 1:]:
                overlaps[f"{left}&{right}"] = len(left_items.intersection(source_pools.get(right, ())))
        return provenance, overlaps

    @staticmethod
    def _rrf(source_pools: Mapping[str, Sequence[int]], c: int = 60) -> dict[int, float]:
        if c < 0:
            raise ValueError("RRF c must be nonnegative")
        scores: defaultdict[int, float] = defaultdict(float)
        for source in SOURCE_ORDER:
            for rank, item in enumerate(source_pools.get(source, ()), start=1):
                scores[int(item)] += 1.0 / (c + rank)
        return dict(scores)

    def candidates(self, uid: int, timestamp: int, strategy: str = "baseline") -> CandidateResult:
        """Retrieve K causal seen objects; target labels are never accepted by this API."""
        if not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError("timestamp must be a nonnegative integer")
        if self.previous_t is not None and self.previous_t >= timestamp:
            raise ValueError("Candidate history must be strictly before query")
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

        pools = self.source_pools(uid, timestamp, strategy=strategy)
        provenance, overlaps = self._diagnostics(pools)
        rrf_scores: dict[int, float] = {}
        quota_contributions = {} if strategy == "rank_fusion" else {source: 0 for source in SOURCE_ORDER}
        selected_by_source = {} if strategy == "rank_fusion" else {source: [] for source in SOURCE_ORDER}
        selection_sources: dict[int, str | None] = {}
        fallback_items: list[int] = []
        fallback_contributions = {} if strategy == "rank_fusion" else {"popular_fallback": 0}
        selected: list[int] = []
        seen: set[int] = set()

        if strategy == "rank_fusion":
            rrf_scores = self._rrf(pools, c=60)
            selected = sorted(rrf_scores, key=lambda item: (-rrf_scores[item], item))[: self.max_candidates]
            selection_sources = {item: None for item in selected}
        else:
            quotas = {
                "baseline": self.candidate_quotas,
                "repeat_heavy": REPEAT_HEAVY_QUOTAS,
                "affinity": AFFINITY_QUOTAS,
            }[strategy]
            # P001 creates each source pool with max_candidates items. The depth-200
            # diagnostic pools therefore only contribute their first K operationally.
            for source in SOURCE_ORDER:
                added = 0
                for item in pools[source][: self.max_candidates]:
                    if item not in seen and item in self.catalog:
                        selected.append(item)
                        seen.add(item)
                        selected_by_source[source].append(item)
                        selection_sources[item] = source
                        added += 1
                        if added >= quotas[source] or len(selected) >= self.max_candidates:
                            break
                quota_contributions[source] = added
                if len(selected) >= self.max_candidates:
                    break
            popular_operational = pools["popular"][: self.max_candidates]
            for item in popular_operational:
                if len(selected) >= self.max_candidates:
                    break
                if item not in seen:
                    selected.append(item)
                    seen.add(item)
                    fallback_items.append(item)
                    selection_sources[item] = "popular_fallback"
                    fallback_contributions["popular_fallback"] += 1

        if any(item not in self.catalog for item in selected):
            raise RuntimeError("internal error: retrieval returned an unobserved catalog item")
        return CandidateResult(
            candidate_items=selected,
            source_pools=pools,
            provenance=provenance,
            quota_contributions=quota_contributions,
            selected_by_source=selected_by_source,
            selection_sources=selection_sources,
            fallback_items=fallback_items,
            fallback_contributions=fallback_contributions,
            overlaps=overlaps,
            rrf_scores=rrf_scores,
        )
