# passk_cont_v1 — pre-registration (committed 2026-09-25 before any new generation)

Frozen before dispatch; any deviation is recorded in `gates/` and the report as a deviation. Interfaces:
`00_contract.md`. Plan: `/home/mabdel03/.claude/plans/i-want-to-run-sharded-swan.md`.

## Questions
- E1: at B = 16A, is one 16A call better than several adequately deep independent calls (8×2A, 4×4A, 2×8A) or
  many shallow ones (16×A)? Also at B = 8A.
- E2: do C48 depth-response measurements predict fresh-draw (V48) coverage, plurality accuracy and rankings?
  Does answer-class structure add over shallow correctness (H2)?
- E5: do observable five-sample features plus a mechanism-informed model (P2) predict allocation outcomes on
  unseen T200 problems better than benchmark/configuration (P0) and an observable-feature logistic (P1)?

## Populations and estimands
- T200 mixture (primary): weights aime 0.10, hmmt 0.10, omni_math_hard 0.50, gpqa_diamond 0.30 applied to
  per-family means. Also the equal four-family macro and per-family results. L96 and V48: equal macro.
- T200 is 20/20/100/60 as frozen in `manifests/splits.json`; it is never expanded, and failed or unknown items are
  never replaced.

## Primary family (Holm, family-wise α = 0.05, two tests; E3 omitted so its test is not retained)
1. **Mixed vs deep.** Locked mixed allocation M* ∈ {8x2, 4x4, 2x8} vs `deep_1x16`, delivered (plurality)
   accuracy on T200 at B = 16A, two-sided, T200 mixture estimand. M* = argmax over the three of L96 tie-averaged
   plurality accuracy under the mixture weights (applied to L96 family means); ties → fewer calls. M* is written to
   `sealed/e1_selection.json` before any T200 label is released.
2. **P2 vs P1.** Item-level mean Brier score of delivered correctness over the five B = 16A configurations
   (16x1, 8x2, 4x4, 2x8, 1x16); per item averaged over its known configurations; items with fewer than 3 known
   configurations dropped and reported; test statistic = mixture-weighted mean of item-level (Brier_P2 − Brier_P1),
   two-sided.

Intervals and p-values: benchmark-stratified paired problem bootstrap, 10,000 replicates, seed 20260925; every
draw, pool and configuration of a sampled problem moves together; studentized paired contrast from
`_vendor/ard_stats.paired_contrast`. Effect sizes with 95% intervals. Everything else is secondary/exploratory.
Non-significance is not equivalence.

## Readout and missingness
- Plurality: classes by exact vote key then mutual complete-link Math-Verify (5 s per pair, timeout = no merge),
  built within each pool in frozen replicate order; GPQA option identity. Ties: seeded uniform over tied modal
  classes, key sha256(`passk_cont_v1/tie`, 20260925, pool_id, class_id). Also the analytic tie-averaged accuracy.
- Correctness tri-state: invalid output = false; grader timeout = unknown. Contrasts use items where both compared
  cells are known, with best/worst-case bounds; an operational analysis counts failed delivery as failure.
- Absent pool members (failed generation after 3 attempts) follow readout rule 07-R12; an all-absent pool is a
  failure in the operational analysis and missing in the primary analysis.

## E2 (C48 → V48; secondary, sealed)
Models: Beta(½,½) per item-depth; pooled glm (benchmark × categorical depth + item intercept, ridge by
problem-grouped 5-fold CV); Dirichlet answer-class model with unseen residual (diffuse and concentrated
treatments); ablations (shallow-only, depth without class competition, full, unstructured ridge). Predictions for
all V48 items × the 5 + 4 configurations, rankings and crossover signs for k = 2, 4, 8, 16 are sealed before V48
labels. Metrics: population-average error, Brier/log score, predictive-interval coverage, paired ranking accuracy,
crossover residuals. Illustrative items: largest predicted |crossover| per benchmark.

## E5 (dev = L96 → T200)
- Training rows (pools from A slots 0–15 / deep prefixes): 1×A; 2×A, 1×2A; 4×A, 2×2A, 1×4A; 16x1, 8x2, 2x8, 1x16;
  primitive 1×8A; C48 only: 2×16A, 4×8A, 8×4A. **Withheld from all fitting/tuning:** aggregate targets
  (n, a/A) ∈ {(4,4), (8,1), (4,2), (2,4)}. Diagnostics: L96 slots 20–24; T200 `t200_diag`.
- Features: CONC = {sorted class frequencies, plurality share, top-two gap, number of classes, entropy}; invalid
  fraction; answer length; ANSWER finish reason; latency z-scored within provider × budget × collection (dropped if
  grouped CV shows no gain); benchmark; provider (P1–P3 only).
- Predictors P0 (benchmark + budget + compositional configuration: log n, log a/A, R, B), P1 (+features +
  provider), P3 (P1 + pairwise products and squares, ridge), P2 (glm mixed model → simulated G with one latent item
  effect per simulation, × η from a second glm on covered multi-call dev pools; η = 1 for singletons), P1−conc,
  P2−conc. Hyperparameters by problem-grouped 5-fold CV on L96 only.
- Primary metric: mean Brier of delivered correctness over the five 16A configurations. Secondary: coverage Brier,
  log loss, calibration, MAE, paired ranking accuracy, residuals by benchmark, withheld cells (4×4A; 8×A, 4×2A,
  2×4A) reported separately, ΔBrier for concentration, a 21A per-item configuration-selection policy readout.

## Seal
One `sealed/SEAL_predictions.json` covers V48 and T200 predictions, models, preprocessing, hyperparameters, seeds,
`e1_selection.json`, the code commit, strace logs and static scan. Released only when all mechanical checks pass;
post-release refits go to `posthoc/` and are exploratory.
