# Pass@K frontier investigation

**Start here.** This folder holds the results of five linked studies run from 2026-09-17 to 2026-09-26. The code that
produced them is in [`agent_allocation/`](../../agent_allocation/).

**The question.** A fixed inference budget `C` can buy more agents, more rounds of revision, more independent samples
read out by a vote, or a longer chain of reasoning inside one call. Which purchase is worth making, and how close can a
real readout get to oracle pass@k, the accuracy a perfect answer picker would achieve over `k` samples?

**The answer so far.** Once a reasoning budget is really spent, one deep call beats every way of splitting the same
budget. Neither off-the-shelf harnesses, nor statistical predictors, nor LLM selectors reliably beat the simple rule of
always making the deep call.

## How the studies connect

| # | Folder | Study id | Run (EDT) | Why it was run | Result |
|---|---|---|---|---|---|
| 1 | [`01_allocation_v1`](01_allocation_v1/) | `openrouter_agent_allocation_v1` | Sep 17 | First comparison of allocations at matched compute | Allocation shape barely matters and selection looks like the bottleneck, **but** the budgets were soft: `deep/16` was allowed 16,384 reasoning tokens and spent a median of 1,187 |
| 2 | [`02_forced_deep_pilot`](02_forced_deep_pilot/) | `forced_deep_pilot_v1` | Sep 18 | Find a way to make a call spend its whole budget | `logit_bias` on the end-of-reasoning token forces exact budgets, on providers that honour it |
| 3 | [`03_forced_budget_v2`](03_forced_budget_v2/) | `openrouter_forced_budget_v2` | Sep 18 | Rerun study 1 with every budget forced | Reverses study 1: `deep/16` delivers 0.792 against 0.417 for `sample/16` and 0.580 for oracle pass@16 |
| 4 | [`04_dynorch_v1`](04_dynorch_v1/) | `openrouter_dynorch_v1` | Sep 23-24 | Would a harness that decides when to delegate do better? | No harness delegated in the core runs; both lose to `deep/16` by 12.5-14.6 pts and beat `sample/16` by 22.9-25.0 pts |
| 5 | [`05_passk_cont_v1`](05_passk_cont_v1/) | `openrouter_passk_cont_v1` | Sep 25-26 | Confirm on fresh problems and try to predict allocation per problem | Confirmed on 200 new problems (`1x16` 0.705 against locked `4x4` 0.580, Holm p = 0.0008). No selector beats always-`1x16` |

Studies 3, 4 and 5 share one model and protocol, and studies 4 and 5 reuse study 3's forced draws. Study 1 used a
different model and different benchmarks, so **do not compare its accuracies directly with the others**.

| # | Model | Benchmarks | Scale | Spend |
|---|---|---|---|---|
| 1 | `google/gemini-2.5-flash-lite` (replication panel on `gemini-2.5-flash`) | MATH-500, MMLU-Pro, BBH logic (4 subtasks), MBPP+ | 400 main items, 67,514 generations | $47.64 |
| 2 | Qwen3.5-9B, Qwen3.5-397B | Mixed dev items | 32 items per model | about $3.6 |
| 3 | Qwen3.8-27B, A = 1,024 tokens | AIME, HMMT, Omni-MATH (difficulty 7+), GPQA-Diamond | 96 items, 13,344 main generations | $70.00 |
| 4 | Qwen3.8-27B under OpenClaude and Codex CLI | Same 96 items | 3,648 core and extension runs | $59.86 |
| 5 | Qwen3.8-27B | Same benchmarks; L96, C48, V48 and new T200 panels | 13,932 draws (2,976 reused from study 3) | $131.56 |

## Where to look

| You want to know | Read |
|---|---|
| The whole story of one study | That study's `README.md`, then its `report/report.pdf` |
| The exact table and filter behind a headline number | [`tools/check_headlines.py`](tools/check_headlines.py) |
| Methods, statistics and every result of the last study | [`05_passk_cont_v1/results_package/README.md`](05_passk_cont_v1/results_package/README.md) and its `DATA_DICTIONARY.md` |
| Per-item accuracy of any configuration | `0N_*/tables/policy_outcomes.parquet` (studies 1, 3); `04_dynorch_v1/tables/policy_outcomes_dynorch.parquet`; `05_passk_cont_v1/results_package/csv/03_pools/` |
| Cost of any configuration | `tables/policy_costs.parquet`, `tables/cost_summary.csv` |
| One row per generation | `tables/candidates.parquet` (1, 3); `04_dynorch_v1/tables/dynorch_runs.parquet`; `05_passk_cont_v1/results_package/csv/02_draws/draws_all.csv` |
| Pre-registered tests | `tables/primary_contrasts.csv` (1, 3); `04_dynorch_v1/report/numbers.json` (`primary`); `05_passk_cont_v1/results_package/csv/04_e1/primary_tests.csv` |
| Columns of any table | `tables/SCHEMA.md`, or `DATA_DICTIONARY` in studies 3 and 5 |
| Gate decisions and deviations | `gates/`, `amendments.jsonl`, `run_decisions.jsonl`, and [`agent_allocation/docs/`](../../agent_allocation/docs/) |
| What the study was supposed to do | [Pass@K-Frontier spec](../../expermt_agent_specs/Pass@K-Frontier/OpenRouter_Agent_Scaling_Master_Spec.md), [Pass@K-Frontier-Cont spec](../../expermt_agent_specs/Pass@K-Frontier-Cont/remaining_experiments_spec.md) |
| Prompts and model transcripts | [`RAW_DATA.md`](RAW_DATA.md): they are **not on GitHub**, only on the cluster |

## Loading the tables

```bash
pip install pandas pyarrow
```

```python
import pandas as pd

df = pd.read_parquet("03_forced_budget_v2/tables/policy_outcomes.parquet")
main = df[(df["view"] == "current") & (df["partition_id"] == "main")]   # always apply both filters

def macro(name):
    # Macro accuracy: mean within each benchmark, then an equal-weight mean across benchmarks.
    return main[main["name"] == name].groupby("benchmark")["y"].mean().mean()

macro("deep/16")    # 0.7917
macro("sample/16")  # 0.4167
```

Configuration names in the `name` column:

| Name | Meaning |
|---|---|
| `deep/m` | one call with `m*A` reasoning tokens |
| `sample/k` | plurality vote over `k` independent calls at `A` |
| `private_single/k` | a revision chain: one agent updating its own answer, `k` generations in total |
| `private_team/n/r` | `n` agents, each revising privately for `r` rounds |
| `peer/n/r` | `n` agents seeing each other's work for `r` rounds |
| `cp/h` | `n = 4` agents communicate once, then make `h` private updates |
| `pooled/4` | four draws read by one synthesizer call |

Study 5 writes allocations as `nxa`: `4x4` means 4 independent calls at 4A each.

## Gotchas that have caused wrong numbers

**All tables**

1. In studies 1 and 3, filter to `view == "current"` and the partition you mean. Each configuration also has an `archive`
   row, so unfiltered counts are double. Study 4's table has only `current`.
2. Compute is `C = team_size * total_cycles * reasoning_budget`, and the name does not tell you it: `total_cycles` is
   rounds + 1, so `peer/4/4` is 20A, not 16A. Do not derive it from `nominal_generations` either: `deep/m` is one generation
   carrying `m*A`, and `pooled/4` is 4 x 2 by that formula but 5 nominal generations, so it does not fit the ladder and is
   excluded from it.
3. `y` is 1 for correct, 0 for wrong and NaN when the outcome is unknown. Check `delivered_state`.
4. `*_usd` columns in `policy_costs` and `candidates` are decimal **strings**. `cost_summary` has floats and no `view`
   column; it already reflects `current`.
5. `oracle_pass_at_k` is combinatorial over the 25-sample bank and is not paired per item with delivered accuracy. For a
   same-pool capture ratio use `prefix_coverage`.
6. The tables hold 100 and 101 names (studies 1 and 3) where the reports count 95 and 96 configurations. The difference
   matches the five `cp/*` and `pooled/4` names. In study 1 those five ran on only 192 of the 400 items, so their macro is
   on a different cohort from every other configuration.

**Study 1**

7. Its budgets were caps the model was free to leave unspent, so its `C` is nominal. It does not show that depth is
   useless. Studies 3 to 5 supersede it on that point.
8. BBH logic is at ceiling (0.998 mean across main-panel configurations) and carries almost no signal. Report a
   three-benchmark macro beside the four-benchmark one.
9. Read [`FINDINGS.md`](../../agent_allocation/docs/FINDINGS.md) before quoting a contrast. MMLU-Pro's zero-width
   intervals are a degenerate bootstrap, not a precise null, and `cp/0 - peer/4/1` is 0 by definition. **One error in that
   file:** it gives main-panel oracle pass@25 as 0.948. That is the `high` panel's value; the main panel's is 0.9375.

**Study 3**

10. Its run report is stamped **INTERIM** for two documented reasons, and nothing further was collected afterwards. 80
    checkpoint rows lack a known outcome (49 `current` plus 31 `archive`). All 49 `current` rows are on `hmmt/2025/18` and
    `omni_math_hard/473`, from Math-Verify timeouts. And the ledger's conservative charge ($72.26, including $2.19 charged
    at full reservation) exceeds the provider's figure ($70.33).
11. 79.2 and 78.5 are the same `deep/16` statistic on the 96-item and 94-item cohorts. Study 5's E0 reconciles them.

**Study 4**

12. A harness run granted `C` often stops naturally: realized share of the cap at `k = 16` is about 0.6, where `deep/k`
    is forced to 1.0. Compare per realized token as well as per cap.
13. In `dynorch_runs`, keep `status == "done"`, which leaves exactly one row per cell in every partition. Retries add
    `transport_failed` rows (16 in main). Do not filter on `generation`: it marks 8 re-executed main cells and every canary
    row, and superseded executions are already out of the table.
14. Extensions E3 (OpenClaw) and E4 (RLM) were **not run**. Nothing here measures them.

**Study 5**

15. T200 headlines use **mixture** weights (AIME .10, HMMT .10, Omni-MATH .50, GPQA .30), not the equal-weight macro used
    elsewhere. The two can change a conclusion, so state which one you use.
16. E8a's prompt was written with T200 outcomes already known; this is disclosed in its pre-registration. The E8a, LEX and
    E8b extensions came after the core results.
17. `05_passk_cont_v1/results_package/MANIFEST.csv` describes the full original package. For this public copy use the
    top-level `MANIFEST.csv` and `REDACTIONS.csv`. The package README also still mentions the withheld example prompts and
    cites cluster paths (`R/`, `W/`, `MEM`), defined under its "Path conventions".

## What was withheld from this public copy

This repository is public, so this folder contains **no benchmark question text and no answer-key files**. GPQA-Diamond's
maintainers in particular ask that it stay out of public corpora.

Two limits on that statement:

- The results record each model's answer and whether it was correct (for example `csv/02_draws/draws_all.csv`), so the right
  answer can be inferred for many items, including GPQA option letters. A letter means little without its question.
- Outside this folder, a few test fixtures under `agent_allocation/tests/` contain examples from open datasets (AIME,
  Omni-MATH, MATH, MMLU-Pro, MBPP+), including some real study 4 transcripts. None are from GPQA.

What was removed:

- **Files:** eight rendered example prompts are left out of `results_package/prompts/`.
- **Cells:** 477 table cells where a model's output quoted its question now read
  `[withheld-benchmark-text sha256:<16 hex>]`. The hash is of the original cell, so joins still work. No rows or numeric
  values changed.
- **Whole categories:** raw transcripts, per-call request logs and item manifests are not here at all.

[`REDACTIONS.csv`](REDACTIONS.csv) lists all of it. The originals remain on the cluster, as described in
[`RAW_DATA.md`](RAW_DATA.md).

Items are still identified by `item_id` (for example `aime/2026/3`).

## Code

| Study | Code in `agent_allocation/` | Contract and runbook | Commit at study end |
|---|---|---|---|
| 1 | `src/agent_allocation/` | `docs/00_contract.md`, `docs/RUNBOOK.md` | `033a899` |
| 2 | `tools/forced_deep_pilot.py` | none | see `git log` for that file |
| 3 | `shards.py`, `calibrate.py`, `analysis/merge.py` | `configs/forced27b.json` | `dc7eb24` |
| 4 | `src/agent_allocation/dynorch/` | `docs/dynorch/` | `4b9348f` |
| 5 | `src/agent_allocation/passk/` | `docs/passk_cont/` | `bc9690a` |

All of these commits are ancestors of `main`, so `git log <sha>` works. Study 5's sealed fits cite `c8b9b7b`, its last
commit before launch, which is also in this history.

**Do not re-run paid stages.** Commands marked `--allow-paid` spend real money through OpenRouter. Analyses can be redone
from the tables here without any API call. The Slurm scripts and default paths assume the MIT ORCD cluster where this ran.

## Verifying your copy

```bash
python experiments/pass_at_k_frontier/tools/verify_manifest.py   # every file against MANIFEST.csv
python experiments/pass_at_k_frontier/tools/check_headlines.py   # headline numbers recomputed from the tables
```

`tools/build_archive.py` is the script that built this folder from the run roots; it only runs on the cluster.
