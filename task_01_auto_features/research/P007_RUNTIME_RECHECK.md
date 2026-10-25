# P007 runtime recheck — vector exact gate and full-catalog path

## Scope

Read-only review of the current vector transform, full-catalog scorer and P001 runner after the compensated-mean and constant-value fast paths. No model training, reserved-period payload, or incomplete-cohort outputs were inspected. The only run evidence reviewed was the completed P001 `catalog_gate_v3` and `full_catalog` status records. This is a bounded code/runtime review, not a universal correctness proof.

Current source SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `src/expanded_vector_features.py` | `8e9390d21c6132a41c28ca6c0e4c2e06ea8068601c252edba8afaeb607bb2e7b` |
| `src/expanded_catalog.py` | `7bf33e8bdd2033f98e40601845e49b5bbb73c914d7b27269b6a562aec4ccb008` |
| `src/run_expanded_catalog.py` | `a50da6bdcbbe81401db4731a4f4307f280abaeed1d341c4d61ea5a6ca9494e7e` |

These replace the earlier hashes in `research/P007_CODE_REVIEW.md` for the reviewed runtime implementation.

## Numeric parity and causal boundary

P001 `catalog_gate_v3` reports `complete`, with exact float32 byte equality for **10,279,400 cells across 499 validation queries**. The value digest is `0381df423fd60531c2a1c7fffd351d2f739fa158dd829584fccff78d4d99cae9`. The gate compares against the frozen P004 feature matrix and does not establish correctness for arbitrary inputs or feature grammars.

The earlier P001 gate discrepancy was in many-to-many C trend means: columns 195/197 for one item combined three nearly cancelling album values. NumPy reduction produced a different float32 ULP from the reference. The current heterogeneous multi-edge path accumulates Python floats with built-in `sum` in normalized mapping order, matching the reference reducer's accumulation behavior. The direct constant fast path covers only exactly 0, 0.5, and 1, whose repeated mean is exactly that value at practical mapping degrees. Other constant values use the built-in repeated sum; heterogeneous values retain ordered built-in summation. The synthetic regression captures the cancellation case and its float32 bit patterns.

The adapter indexes only development events supplied by the caller. It advances event-count prefixes with `timestamp < query_timestamp`; window cutoffs match the source series rule by removing timestamps `<= timestamp - width`. `engine._assert_query_time` remains the causal guard, and the adapter allows nondecreasing query times, including distinct users and chunks at the same timestamp. No same-time event is included in features. Mapping degrees come from the fixed mapping snapshot; they are not derived from future interaction counts. The full-catalog scorer takes candidates from `engine._items`, the externally observed prefix, rather than from the adapter's larger indexing universe.

## Chunking and memory

`run_expanded_catalog.py` transforms catalog chunks of 2,048 rows. The returned feature block is therefore at most 2,048 × 206 float32 values (about 1.6 MiB), plus temporary intermediates. For a repeated `(uid, timestamp)` call, the adapter reuses pair counts, user values and per-column artist/album reductions; a new query key replaces the active user/query cache. It does not retain a full `catalog_size × 206` feature matrix.

The cache is still proportional to the dense item universe: it retains item-wide all-time/recent count arrays, pair-count arrays for the active query, and one float64 reduced vector per C feature column. Entity-value arrays are temporary while those vectors are built. The scorer also retains score vectors for the active catalog and sorts them to rank items. Thus chunking bounds the feature output block, but total memory still grows with catalog size, mapping edges, and scorer count.

P001 measured peak process RSS of **650.74 MiB** for the completed exact gate and **665.99 MiB** for full-catalog scoring, with no configured resource-limit exceedance. The gate took 364.64 s. Full scoring took 662.23 s across 499 queries and wrote 19,098,715 catalog rows. Observed catalog size ranged from 35,580 to 41,240 items (mean 38,273.98). These measurements are specific to P001's local environment and cohort.

## Full-catalog score checks and limits

The P001 full run records `complete`, `all_k100_scores_exact: true`, and the same 499-query exact feature gate digest. The runner recomputed scores for the original K100 candidates and compared them with the saved predictions before full scoring; it also checked that chunked transforms matched the K100 feature rows. Catalog ranking uses score descending and item-ID ascending for ties. It retains a target miss as `target_rank: null`; it does not insert the target into the catalog.

The evidence reviewed here supports exact parity for this frozen P001 development matrix, configured 206-spec grammar, and these completed P001 queries. It does not certify another cohort, another source/mapping snapshot, all possible C mapping degrees or values, different dtypes/platform arithmetic, or all failure paths. P005 cohort runs remain outside this recheck's evidence scope. Full-catalog rows are development-validation outputs and should not be described as independent test results.
