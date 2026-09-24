# dynorch_v1 report (`openrouter_dynorch_v1`)

Mirrors `docs/forced_v2_report/`: `analysis.py` writes `numbers.json`, `tables/` and `figures/`; `render.py`
fills every `{{path|fmt}}` placeholder of `report.tmpl.tex` from `numbers.json` (an unresolved or null placeholder
fails the build); `build.sh` runs analysis, render, lint and `latexmk`. The committed copy of the built report goes
to the main checkout under `docs/dynorch_v1/report/` at Gate 4 (plan section 7).

## Inputs (all read-only)

| input | path |
|---|---|
| dynorch run root | `RUN` (default `agents_scaling_results/dynorch_v1/artifacts`): `tables/{dynorch_runs,dynorch_calls,policy_outcomes_dynorch,policy_costs_dynorch}.parquet` from `dynorch-tables`, `manifests/dynorch_plan_<stage>.jsonl` |
| sealed ladder | `LADDER` (default `forced_budget_v2/artifacts/merged`): `tables/{policy_outcomes,policy_costs,sample_bank_items}.parquet`, `../manifests/panels.json` |
| config | `configs/dynorch_v1.json` (`statistics` block: primary family, Holm over 4, seed 20260923) |
| sealed ladder numbers | `docs/forced_v2_report/numbers.json` (the best non-deep configuration per rung is fixed from it) |

Nothing is written into the sealed run root; the ladder and harness rows are concatenated in memory
(`dynorch/tables.merged_outcomes`, `core.Frame`).

## Build

```bash
./build.sh                 # analysis (~ minutes at 10,000 replicates) -> render -> lint -> compile -> log check
./build.sh --no-analysis   # re-render and compile only
RUN=/path/to/run LADDER=/path/to/forced_budget_v2/artifacts/merged ./build.sh
```

TeX caches go to `dynorch_scratch/texmf-*` (never `/home`).

| file | role |
|---|---|
| `core.py` | `Frame` (subclass of `forced_v2_report/core.py::Frame`): ladder + harness columns, `families = LADDER_FAMILIES + ["harness"]`, `SEED = 20260923`, ONE `build_population` per contrast on `main_96`, paired per-item oracle pass@k from `sample_bank_items` |
| `primary.py` | `dynorch_primary_analysis`: anchor `H0/16 - deep/16` first, then the four frozen contrasts through `analysis/contrasts.evaluate_contrast(n_primary=4)` and `_vendor/ard_stats.holm` |
| `analysis.py` | every number, table and figure (`run_analysis(...)` is the callable used by the tests) |
| `figures.py` | `d1_design` .. `d9_repeat_agreement` |
| `render.py` | placeholder rendering (`render_text` raises on any unresolved placeholder) |

## numbers.json (top-level keys)

`study, n_items, n_items_by_bench, n_cells_planned, n_runs_core, n_runs_all_attempts, n_reruns, n_runs_by_arm_rung,
n_cells_by_state, completeness, partial_cells, stages_present, spend, ledger, harness_versions, proxy_versions,
ladder_provider_assignment, provider_assignment, same_provider_share, harness_prompt_tokens (S_h), acc (every arm x
rung and every comparator: macro + per benchmark, lo/hi), ladder_check (sealed point estimates reproduced), oracle,
best_non_deep_at_rung, best_team_at_rung, primary (anchor + 4, est/lo/hi in points, p, p_holm, reading),
primary_notes, descriptive (vs_deep, vs_sample, vs_private, vs_best_non_deep, vs_best_team, vs_H0, vs_oracle),
slopes, realized (B10: nominal cap beside mean/median realized, grace, cap-bound, terminal, realized_share),
frontier_f2, frontier_usd, frontier, f1_bias_answer_tokens, diagnostics, spawn_split (exploratory),
provider_sensitivity, repeat_agreement, refit, headline`.

Tables: `headline.tex, primary.tex, primary_contrasts.csv, rung_table.csv, descriptive.csv, realized.tex,
diagnostics.tex, provider_sensitivity.tex|csv, cost.tex, frontier.csv, repeat_agreement.csv, forecast_refit.csv`.

## Conventions

- Every number in the prose is a placeholder; `report.tex` is generated.
- Harness rows use the sealed `policy_outcomes` / `policy_costs` schemas with `team_size = total_cycles = 1`,
  `reasoning_budget = C`, so `C = B*R*A` and `rung = C // 1024` hold in the sealed `core.py` code path.
- The ladder's CIs here are drawn per contrast (a different `seed_label` from the sealed joint population): point
  estimates are identical to the sealed report, intervals differ slightly, and the report says so.
- Pending cells (no terminal record yet) carry `y = NaN`, `delivered_state = "pending"`; the completeness per cell
  is reported and cells under 0.95 are listed in `partial_cells`.
