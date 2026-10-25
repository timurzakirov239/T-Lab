# P002 — retrieval candidate diagnostics

Status: API and synthetic tests are prepared; no real retrieval experiment has been run and no strategy has been selected from real outcomes. The generator is intended to replay the same P001 cohort and exact train/validation query IDs with the same causal event prefix.

## API for the runner

Import `RetrievalCandidateGenerator` and `CandidateResult` from `src/retrieval_candidates.py`.

```python
generator = RetrievalCandidateGenerator(
    artist_map,
    album_map,
    max_candidates=100,
    candidate_quotas={"popular": 25, "repeat": 25, "artist": 35, "album": 15},
    source_depth=200,
    baseline_profile_depth=5,
    affinity_profile_depth=32,
)

generator.observe(event)  # each event is strictly before a query when replayed
result = generator.candidates(user_id, query_timestamp, strategy="baseline")
```

The runner should feed events in nondecreasing raw timestamp order and call `candidates` only after replaying every event with `timestamp < query_timestamp`. If multiple events have equal timestamps, observe the complete equal-time group before querying a later timestamp. The generator rejects a query at or before its last observed timestamp. The API accepts no target or label argument and never inserts a target. It never draws from unseen catalog items.

`CandidateResult` fields:

- `candidate_items`: ordered unique shortlist, at most K=100.
- `source_pools`: ranked `popular`, `repeat`, `artist`, and `album` pools before quota processing, each up to depth 200.
- `provenance`: for every item in the union of those pools, a list of source names containing it, in fixed order popular, repeat, artist, album.
- `quota_contributions`: counts of unique items actually added during each source's quota turn. For rank fusion this is `{}`.
- `selected_by_source`: item ID lists credited to each quota turn. For rank fusion this is `{}` because no source owns a fused item uniquely.
- `selection_sources`: selected item ID to first source that admitted it, or `popular_fallback`; for RRF every selected item maps to `None`.
- `fallback_items` and `fallback_contributions`: actual item list and count added by popular fill after quota turns; for RRF both are empty.
- `overlaps`: union size, number of items occurring in multiple pools, and pairwise pool overlap counts at depth 200.
- `rrf_scores`: RRF score for each item in the source-pool union; empty for quota strategies.

Pool lists returned to the runner are copies, so annotation cannot mutate the generator cache. Pools are cached per `(user_id, query_timestamp)`: baseline, repeat-heavy and RRF share the same top-5-profile pools; affinity uses the top-32-profile variant. Any subsequent `observe`, including another event at the same timestamp, invalidates that cache. The generator has no dependence on later queries and can be replayed to regenerate an earlier prefix.

## Strategies

| Strategy key | Quotas in source order | Related-profile depth | Candidate construction |
| --- | --- | ---: | --- |
| `baseline` | popular 25, repeat 25, artist 35, album 15 | Top 5 artist and top 5 album profile entities, matching `CandidateGenerator` in `src/run_pilot.py` | Consume only the first K=100 entries from each source pool during quota filling; then fill any remaining capacity from the first K popular items. This preserves P001 semantics despite exposing depth-200 pools for diagnostics. |
| `repeat_heavy` | popular 10, repeat 60, artist 20, album 10 | Top 5 | Same ordered unique quota fill and popular fallback, with repeat-heavy quotas. |
| `affinity` | popular 10, repeat 20, artist 50, album 20 | Top 32 artist and top 32 album profile entities | Same quota fill and fallback. P001 code actually uses top 5 profiles; top 32 is an explicit P002 variant, not a claim about P001. |
| `rank_fusion` | None | Top 5 | Compute reciprocal rank fusion over the four complete depth-200 pools with `score(item) = Σ_source 1/(60 + rank_source(item))`, ranks starting at 1. Return the top K by score descending, tie by item_id ascending. No quota pass and no popularity fallback. |

All strategies observe the same history. The source definitions intentionally retain P001 behavior:

- Popular: listen count descending, then item ID ascending.
- Repeat: most recent listen timestamp descending, then item ID ascending.
- Artist/album: select top user profile entities by fractional listen count (ties by entity ID), then score each related seen item by `profile_count * log1p(item_listen_count)`; sort by score descending, then item ID ascending.
- Catalog: any observed event adds the item. Only `listen` changes popularity, repeat history and artist/album profiles. `like`, `dislike`, `unlike` and `undislike` affect catalog membership only, matching P001.
- Mappings are deduplicated and treated as fixed snapshots, consistent with P001 preparation.

## Diagnostic interpretation

`source_pools` and `provenance` describe pool membership before quotas; `quota_contributions` describes what survived deduplication in the ordered quota step; `fallback_contributions` makes popular fill separately visible. These measurements allow the runner to distinguish pool overlap from quota saturation. For RRF, the full source membership and scores remain available, while quota and fallback contributions are empty because those steps do not apply.

Candidate generation must use exactly the query ID set and per-query history prefix used by P001. Do not recompute or sample queries in this module. Compare candidate recall and downstream ranking on the same request IDs. This is a fixed retrieval diagnostic; no strategy selection or parameter tuning is based on real results in this implementation package.

## Synthetic validation

`tests/test_retrieval_candidates.py` checks exact list equivalence against the existing P001 `CandidateGenerator` at K=100; prefix causality and rejection of equal-time/earlier-time queries; deterministic ties, deduplication, quotas, provenance and overlaps; top-5 versus top-32 profile pools; cache invalidation after observation; non-listen catalog-only behavior; no target interface/unseen items; and a hand-computed RRF example with tie ordering.
