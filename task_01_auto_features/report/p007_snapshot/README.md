# P007 — compact rank checkpoint

Three disjoint 64-user development cohorts, all 1531 original validation queries, K100 vs complete prior-observed cohort catalog. No P006 test replay. Models trained on identical original conditional K100 rows; full evaluation is a fixed-model catalog bridge.

The CSV contains one-based target ranks for nine methods and two catalog modes. Missing ranks remain zero in metric denominators. Raw histories, feature matrices, predictions and models are omitted. Local independent verification checked original prediction ranks, top20 score evidence and catalog-superset invariants; portable verification checks hashes and recomputes pooled/cohort metrics, target transitions and descriptive bootstrap.

Run `python task_01_auto_features/src/report_expanded_search.py --verify` with NumPy only. Human source verification remains pending.
