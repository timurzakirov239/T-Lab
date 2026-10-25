# P006 final-test compact snapshot

Origin: the one-time parent run `artifacts/p006_final_test`, using the frozen config and source/input hash inventory in `configs/`. The global and per-cohort reservation markers are preserved under `reservation_markers/`.

This snapshot preserves the one-time P006 reservation, frozen configuration and source manifest, parent/child verification evidence, frozen masks, compact result JSON, analysis, and per-query ranks.

Ranks were derived from the verified test predictions using score descending and item ID ascending for ties. A blank target rank means the retrieval candidate set missed the target; those queries remain in all metric denominators. `validation` in query metadata is the evaluator-compatible split label; `evaluation_period=test` marks the reserved period.

Raw histories, mappings, feature matrices, trained model files, and prediction NPZ archives are not copied. Their hashes remain in source/run status evidence. This is a compact audit copy after the single test evaluation, not a rerun or a new estimate. P001/P005 cohort results remain conditional on these fixed cohorts; the paired-user bootstrap is descriptive and does not estimate model/cohort uncertainty. Failed cohorts retain diagnostics and do not contribute score claims or rank rows.

Verify this isolated copy with `python task_01_auto_features/src/verify_final_snapshot.py`.
