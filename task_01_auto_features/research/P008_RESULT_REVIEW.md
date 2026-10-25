# P008 — read-only review of completed results

Date: 2026-10-25
Review type: read-only consistency and interpretation review; this is not human source verification.

## Scope and integrity checks

Compared the generated report, `report/p008_snapshot/analysis.json`, the independent verifier receipt and the frozen P008 protocol. The report's main metric table matches the analysis values to its six-decimal display precision. The primary contrast label and direction match the predeclared protocol (`Full:HybridHit_UIDC − Baseline_UIDC`).

- Independent verification: `status=verified` for all three cohorts; 8 models and 9 rank labels per cohort; all 18 new fits referenced by the common all-fits freeze before validation evaluation.
- Portable snapshot verification: `PASS`, 29 tracked files, 27,558 rank rows, 18 pooled metric groups, 54 cohort metric groups and all 29 paired contrasts recomputed.
- Scope: 1,531 previously used development validation queries / 158 users. P006 test was not evaluated. Human source review remains pending.

## Primary result

Full user-macro NDCG@20 was 0.069154 for `Baseline_UIDC` and 0.068288 for `HybridHit_UIDC`. The predeclared paired difference was **−0.000867**, descriptive paired-user 95% interval **[−0.005067, +0.003618]** over 158 users. All three cohort-specific intervals also include zero (P001 +0.001553 [−0.004691, +0.008225]; Seed26 −0.001590 [−0.009311, +0.006441]; Seed27 −0.002517 [−0.010029, +0.005420]). This does not establish an improvement or equivalence.

Full Recall@20 is 0.159177 for the baseline and 0.152215 for HybridHit. The report appropriately keeps NDCG@20 as primary and does not promote the secondary measure into a replacement endpoint.

## Secondary sampling and warm-query contrasts

- Uniform negative sampling is below the corresponding P007 baseline in the pooled Full result: `Uniform_UIDC − Baseline_UIDC` is −0.008292 [−0.017176, −0.000156]; `Uniform_A1 − Baseline_A1` is −0.015539 [−0.024791, −0.007055]. These are descriptive comparisons over the reused development validation set.
- Teacher-hard sampling is above Uniform in the pooled Full comparison: `HybridHit_UIDC − Uniform_UIDC` is +0.007425 [−0.000766, +0.016387], while `HybridHit_A1 − Uniform_A1` is +0.014446 [+0.005840, +0.024197]. The UIDC interval includes zero. The A1 cohort estimates vary; its pooled interval is not evidence of a general or independent winner. Against the baseline, HybridHit's primary UIDC difference remains near zero and uncertain.
- HybridWarm expands training from the hit-query counts (349 / 330 / 336) to cohort-warm counts (1,153 / 1,282 / 1,209). Its pooled Full NDCG@20 is 0.009811 for UIDC and 0.010174 for A1, versus 0.068288 / 0.071635 for HybridHit. The paired Full differences are −0.058477 [−0.072445, −0.045012] for `HybridWarm_UIDC − HybridHit_UIDC`, and −0.061460 [−0.075807, −0.047818] for A1; all three cohort intervals for each contrast are below zero. The same large descriptive drop appears in K100.
- Across cohorts, the training-query sets total 1,015 hit queries (910 repeats, 105 new-to-user) and 3,644 cohort-warm queries (2,525 repeats, 1,119 new-to-user). In Full validation, HybridWarm UIDC repeat user-macro Recall@20 is 0.018580 versus 0.256436 for Baseline UIDC; the matching NDCG@20 values are 0.008981 versus 0.109937. This is a notable observed tradeoff after broadening the training set; the metrics alone do not explain its mechanism.
- Stratified metrics help qualify the warm result: HybridWarm has higher Full NDCG@20 on new-to-user targets than Baseline UIDC (0.012332 vs 0.002356), but much lower values on repeat targets (0.008981 vs 0.109937) and on cohort-warm targets (0.012348 vs 0.080257). These descriptive strata overlap conceptually: “new” is user-level, while “warm” is cohort-catalog-level. The results do not identify why the overall warm-arm metric falls, and should not be presented as a causal explanation.

The result report is careful to describe these as exploratory, to distinguish cohort-warm from user-repeat status, and not to name a winner. At the time of this first review, the secondary HybridWarm tradeoff was easier to find in the linked analysis JSON than in the report narrative. No numerical or direction mismatch was found in the inspected primary and secondary values.

At first review, two narrative details in the generated report needed correction before it could be treated as a finalized account: it said the P007 A1 masks were chosen after P007, while the masks themselves were frozen in P004/P005 and selected as a secondary benchmark after P007; and it said target transitions were saved in paired analysis, although the analysis contains paired metric deltas/counts rather than transition counts. The training table's “Fit counts” label also described training-query counts per arm, not a number of fits. These were report wording issues, not mismatches in the primary numerical result or the inspected paired contrasts.

## Follow-up: report wording review closed

Re-read the updated generated report. The three wording issues above are corrected: it distinguishes when A1 masks were frozen from their later use as a secondary benchmark, labels the arm counts as training queries, and describes Full−K100 paired differences without claiming that transition counts are stored in `analysis.json`. The report now presents all four pooled Full sampling/warm contrasts with their intervals and adds repeat/new-to-user metrics plus cautious interpretation of the HybridWarm result and possible explanations. The earlier “stratum table” reference above means stratified data in `analysis.json`; the generated report's new table reports repeat and new-to-user strata, while cohort-warm strata remain available in the analysis JSON. The updated report continues to make no independent winner claim. No new numerical discrepancy was found in the text reviewed.

## Limits that remain

The validation queries and P007 A1 masks were previously examined; these are not an independent holdout. The paired bootstrap is conditional on these users and the fixed models and does not include model-refit or cohort-sampling uncertainty; there is no multiplicity adjustment. Teacher-hard items were selected from saved in-sample P007 teacher scores, which the independent verifier checks for alignment and ordering but does not independently regenerate by rerunning teacher inference. Unmarked sampled items are not known negative feedback. Cohort-wide catalog availability is not a claim about the full service catalog or historical mapping availability. Human review of primary sources remains pending.

## Reviewed artifacts

- [Generated P008 report](../report/P008_SAMPLED_TRAINING.md)
- [Portable paired analysis](../report/p008_snapshot/analysis.json)
- [Independent verification receipt](../report/p008_snapshot/verification_independent.json)
- [P008 protocol](../configs/P008_PROTOCOL.md)
- [Report reproducibility instructions](../report/P008_REPRODUCIBILITY.md)
