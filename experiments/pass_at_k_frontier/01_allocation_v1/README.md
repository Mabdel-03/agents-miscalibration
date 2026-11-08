# Study 1: `openrouter_agent_allocation_v1`

**Question.** At matched compute `C = B * R * A` (agents x rounds x reasoning allowance), which way of spending the budget
pays?

**Run.** 2026-09-17, collection 11:35-13:01 EDT. `google/gemini-2.5-flash-lite` on 400 items (100 each from MATH-500,
MMLU-Pro, BBH logic, MBPP+). 95 configurations, 25-sample independent bank per item. 67,514 generations, 101,088 checkpoint
rows, all with known outcomes. $47.64 of a $135 ceiling. Audit: 29 pass, 0 fail, 1 skip.

**Read this first: the budgets were soft.** `A` was a cap the model could leave unspent. `deep/16` was allowed 16,384
reasoning tokens and spent a median of 1,187 (mean 2,774). Every depth conclusion below is limited by that, and
[study 3](../03_forced_budget_v2/) reverses it.

## Findings

Macro accuracy over the four benchmarks unless stated.

| Finding | Value | Source |
|---|---|---|
| Peer exposure hurts (primary): `peer/4/4 - private_team/4/4` | -2.5 pts [-4.6, -0.8], Holm p = 0.027 | `tables/primary_contrasts.csv` |
| Communicate-then-private gain (primary) | -0.5 pts, Holm p = 0.57 | same |
| Allocation shape at matched compute | 0 of 70 per-benchmark comparisons against sampling survive multiplicity correction | `report/report.pdf` |
| Sampling plateaus: `sample/1`, `sample/16`, `sample/25` | 0.835, 0.8725, 0.875 | `tables/policy_outcomes.parquet` |
| Oracle pass@25 against vote@25 | 0.9375 against 0.875 | `tables/pass_at_k_vs_readout.csv` |
| Peer penalty at 5 agents, 4 rounds (three-benchmark macro) | -5.0 pts | `report/numbers.json` (`peer_cost`) |

The report's reading is that selection, not generation, was the binding constraint in this regime. Treat that as specific
to soft budgets on these benchmarks.

## Caveats

- [`FINDINGS.md`](../../../agent_allocation/docs/FINDINGS.md) is required reading before quoting a contrast. MMLU-Pro's zero
  intervals are fragile, and `cp/0 - peer/4/1` is a tautology.
- `FINDINGS.md` gives main-panel oracle pass@25 as 0.948. That is the `high` panel's value; the main panel's is 0.9375.
- `cp/*` and `pooled/4` ran on only 192 of the 400 items, so their macro is on a different cohort.
- BBH logic is at ceiling. Report the three-benchmark macro too.
- MBPP+ accuracy uses a public-test selector, not a vote.
- Two finalization deviations are logged in `run_decisions.jsonl`.

## Files

| Path | Contents |
|---|---|
| `report/report.pdf` | Iso-compute report (25 pp), the allocation analysis; `numbers.json` holds its figures |
| `run_report/` | `RUN_REPORT.md`, `AUDIT.md`, `COMPLETENESS.md`, `COST_RECONCILIATION.md`, `PILOT_REPORT.md` |
| `tables/` | Sealed tables, with `SCHEMA.md`. Four CSVs over 10 MB exist as Parquet only |
| `figures/` | Run report figures |
| `gates/`, `amendments.jsonl`, `run_decisions.jsonl` | Gate records, amendments and decisions, including superseded pilots |
| `manifests/panels.json` | Item ids per panel |

Spec: [`OpenRouter_Agent_Scaling_Master_Spec.md`](../../../expermt_agent_specs/Pass@K-Frontier/OpenRouter_Agent_Scaling_Master_Spec.md).
Raw transcripts are not on GitHub, see [`RAW_DATA.md`](../RAW_DATA.md).
