# P008 — compact rank checkpoint

Three existing disjoint development cohorts; all 1531 outer-validation queries and nine labels in K100 and Full modes. P008 models use fixed UIDC66 or frozen cohort-specific P007 A1 masks. Full reuses the same trained model and scores the prior-observed cohort catalog. No P006 test replay.

The CSV contains one-based target ranks and denominators. Training records, sampled feature matrices, teacher pool payloads, models and event histories are omitted. The copied stage statuses and training_frozen files retain hashes, eligibility, model provenance and cost. The shared all_fits_frozen.json ledger and gate/full status hashes document that all 18 new fits were frozen before any validation scoring. Portable verification recomputes pooled/cohort rank metrics and all paired comparisons with NumPy and expanded_metrics only.

Run `python task_01_auto_features/src/report_sampled_training.py --verify` with NumPy. Human source verification remains pending.
