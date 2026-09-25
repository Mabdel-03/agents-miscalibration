# WP-M handoff — passk_cont_v1 E2 + E5 models (branch `passk-wpm`)

Worktree `/orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_wpM`, based on `e781c34` (day-0 contract). All code is
numpy/scipy/pandas only; no new dependency, no change to `contracts.py`, `GenerationSpec`, templates or
`runtime/scheduler.py`, and no edit outside the WP-M modules listed in contract §4 (plus tests).

## Done

| priority | module | status |
|---|---|---|
| P1 | `passk/models/glm.py` | done, tested. `fit_glm(X, y, groups, *, item_intercept, ridge, gh_nodes=20, sample_weight, bounds, sigma_fixed, laplace)` → `GLMFit` (`predict_proba(X, item_effect=None)` marginal or conditional, `sample_item_effects`, `posterior_item_effects` (grid posterior for training groups), `sample_params` (Laplace)); `grouped_folds`, `cv_ridge(..., folds=5)` grouped by problem, salted sha256 fold order. |
| P2 | `passk/models/response.py`, `answers.py`, `e2.py`, `inputs.py` | done, tested. Entry points `e2.fit_and_predict(run_root)` → `sealed/v48/`, `e2.score(run_root)` → `scores/e2/`. |
| P3 | `passk/features.py`, `passk/models/predictors.py`, `e5.py` | done, tested. Entry points `e5.fit_and_predict(run_root)` → `sealed/t200/`, `e5.score(run_root)` → `scores/e5/`. |
| dev shim | `tests/unit/passk/devshim.py` | **DEV SHIM, tests only.** Builds draws-shaped frames from the read-only legacy root (candidates.parquet + shards/*/parsed). Kept out of `passk/models` so the seal static scan stays clean. |

Tests: `tests/unit/passk/{test_glm,test_e2,test_e5,test_legacy_integration}.py`, 28 passed (about 25 s). The legacy
integration tests skip when the legacy root is absent.

### E2 outputs (`sealed/v48/`, write-once: refuses to overwrite)
- `predictions.parquet`: `item_id, benchmark, config_id, n, a, model, pred_coverage, pred_plurality, pi_lo, pi_hi`
  (90% posterior band of the plurality probability), `cov_pi_lo, cov_pi_hi, in_menu, budget, pred_rank`. Rows cover
  the 9 menu configs plus the crossover arms `2x1,4x1,1x2,1x4` (with `in_menu=False`).
- Models: `beta_full_{diffuse,concentrated}`, `glm_full_{diffuse,concentrated}`, `shallow_{diffuse,concentrated}`
  (ablation 1), `beta_nocomp` (ablation 2), `ridge` (ablation 4, the unstructured predictor). Ablation 3 is
  `beta_full_*`.
- `crossover.parquet` (k = 2, 4, 8, 16: `pred_deep` = P(1×kA correct), `pred_wide_cov` = P(k×A covered), `pred_diff`,
  `pred_sign`, `p_sign_pos`, band), `population.parquet` (macro and per-benchmark 95% predictive intervals, including
  Bernoulli noise, for plurality and coverage), `illustrative.parquet` (frozen rule: largest |pred_diff| per
  benchmark, `beta_full_diffuse`), `illustrative_curves.parquet` (p_x(a), h_x(a)/a, modal margin under both
  treatments), `treatment_gap.parquet` (diffuse vs concentrated; gap > 0.10 flagged as material), `models/` (pooled
  glm incl. the categorical-vs-monotone dev CV table, ridge), `seeds.json`, `manifest.json` (sha256 of every file,
  class-source accounting).
- The scorer computes Brier and log score, population-average error, population interval coverage, paired ranking
  accuracy (within budget groups, ties = 1/2), and crossover residuals with sign agreement (failures included).

### E5 outputs (`sealed/t200/`, write-once)
- `predictions.parquet`: `item_id, benchmark, config_id, predictor ∈ {P0,P1,P2,P3,P1-conc,P2-conc}, p_delivered,
  p_coverage` for every T200 item × 9 configs. Items with no diagnostics get dev-mean features and `imputed=True` in
  `features.parquet`.
- `features.parquet`, `models/<predictor>.json` (+ `latency.json`, which records whether grouped CV kept latency),
  `seeds.json`, `manifest.json`.
- Guards (`predictors.guard_training_frame`): raises on (n,a) ∈ {(4,4),(8,1),(4,2),(2,4)}, on any
  `v48_bank`/`t200_bank` row or member, and on any `t200_diag` label column or row with labels.
  `fit_and_predict` also refuses when held-out partitions carry labels. `config_rank(frame) == 3` is asserted before
  every fit.
- The scorer computes the primary metric (item-level mean Brier over known 16A configs, ≥ 3 known, mixture and macro
  weighted), coverage Brier, log loss, ECE with the calibration table, MAE, paired ranking, withheld 4x4 and 8A cells,
  concentration contrasts (P1-conc − P1, P2-conc − P2), residuals by benchmark, and the 21A select-per-item policy
  against each fixed 16A config (never against the realized ceiling). `item_brier.parquet` is the per-item input that
  WP-A's Holm/bootstrap test (P2 vs P1) needs.

### E5 validation on the real legacy L96 data (dev shim; 4 disjoint folds, each predicting 24 held-out items from 72)

Only the legacy draws exist here: 16 A slots and one deep draw per depth, so the training configs are
1x1, 2x1, 4x1, 16x1, 1x2, 1x4, 1x8, 1x16. The observable 16A configs are 16x1 and 1x16, so the primary metric uses
`min_known=2`. Classes come from the exact-key fallback. 8x1 is withheld from training and is scored here.

| predictor | primary Brier (mixture) | macro | coverage Brier 16A | log loss 16A | ECE 16A | Brier 8x1 (withheld) |
|---|---|---|---|---|---|---|
| P0 | 0.189 | 0.187 | 0.189 | 0.567 | 0.106 | 0.193 |
| P1 | 0.161 | 0.151 | 0.158 | 0.476 | 0.106 | 0.113 |
| P1-conc | 0.180 | 0.166 | 0.182 | 0.523 | 0.082 | 0.145 |
| P2 | 0.159 | 0.141 | 0.148 | 0.459 | 0.068 | 0.107 |
| P2-conc | 0.190 | 0.171 | 0.181 | 0.535 | 0.080 | 0.153 |
| P3 | 0.157 | 0.138 | 0.143 | 0.438 | 0.101 | 0.101 |

Mixture ΔBrier: P2−P1 −0.002, P1-conc−P1 +0.019, P2-conc−P2 +0.031, P1−P0 −0.029. Concentration features carry most
of the gain. The 21A policy picks 1x16 for nearly every item (0.750 mixture, the same as fixed 1x16; P2 0.725).
Per-config Brier: 16x1 P2 0.117 / P3 0.111 / P0 0.206; 1x16 about 0.165 for all.

Production-size E2 (S=800, inner=48) on the legacy pseudo-C48 takes about 51 s. E5 takes about 7 s per fit on
legacy-size data.

## Remaining, in priority order
1. **Wire into WP-A's `passk-analyze`**: stage `e2-fit` → `e2.fit_and_predict(run_root)`, `e5-fit` →
   `e5.fit_and_predict(run_root)`, `score` → `e2.score(run_root)` and `e5.score(run_root)`. Thin CLIs also exist:
   `python -m agent_allocation.passk.models.e2 fit|score --run-root R` (the same for e5).
2. **Agree the classes schema with WP-A.** WP-M reads optional `analysis_inputs/classes.parquet` with columns
   `scope_id, candidate_id, class_id` (null = no vote). Scope ids:
   - one per pool, as `pool_id`;
   - one per diagnostic pool, as the `diag5` `pool_id`, falling back to `diag5:<item>`;
   - one per C48 item over its fitting draws, as **`e2fit:<item_id>`**.

   A scope that is missing, or that lacks any member, falls back to exact vote keys. The manifest `classes` field
   counts the fallbacks, and it should be 0 in production.
3. **Readout alignment.** The observed pool outcomes use the tie-averaged plurality, which is the expectation of the
   seeded rule. If the PREREG requires the seeded delivered answer for the primary Brier, pass WP-A's readout as
   `observed=` (columns `item_id, config_id, coverage, delivered`) to `e5.score` / `e2.score`.
4. Bootstrap and Holm for P2 vs P1 belong to WP-A `stats.py`, fed from `scores/e5/item_brier.parquet`.
5. Possible refinements, not required:
   - a random depth slope in P2;
   - E2 ridge training rows use features built from the same fitting draws as their targets (in-sample, as the
     "unstructured" ablation allows);
   - add P1/P3 parameter intervals.

## Known issues and notes
- `pools.parquet` from WP-A: L96 pools are filtered to the training configs, so withheld configs are physically
  absent. `view == "current"` is used when present. `(item, config)` duplicates across L96/C48 panels are dropped. V48
  scoring fills crossover arms that are missing from WP-A pools by the same prefix rule.
- Class correctness: a class counts as correct when most of its known member labels are correct. Classes whose
  members are all unknown are left out of the E2 class model. An observed pool with a tied class whose labels are all
  unknown gives NaN.
- E2 class shares do not depend on depth within an item. Only the correct mass p_a varies with depth.
- A posterior mean can fall outside a skewed 5–95% band. `pi_lo`/`pi_hi` are quantiles, not mean ± k·sd.
- The dev-shim cache defaults to `/tmp/passk_wpm_devshim_draws.parquet`; set `PASSK_DEVSHIM_CACHE=<dir>` to move it.

## Exact next commands
```bash
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_wpM/agent_allocation
export UV_CACHE_DIR=/orcd/data/tpoggio/001/mabdel03/.cache/uv HF_HOME=/orcd/data/tpoggio/001/mabdel03/.cache/huggingface \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MPLCONFIGDIR=/orcd/data/tpoggio/001/mabdel03/.cache/matplotlib \
  PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pytest -q tests/unit/passk            # 28 tests, about 25 s
# after WP-A extract has written analysis_inputs/ and splits.json exists (pre-seal):
.venv/bin/python -m agent_allocation.passk.models.e2 fit --run-root /orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1
.venv/bin/python -m agent_allocation.passk.models.e5 fit --run-root /orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1
# after the V48 / T200 releases:
.venv/bin/python -m agent_allocation.passk.models.e2 score --run-root /orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1
.venv/bin/python -m agent_allocation.passk.models.e5 score --run-root /orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1
```
