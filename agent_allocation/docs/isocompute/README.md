# Iso-compute allocation report

Re-keys the sealed `openrouter_agent_allocation_v1` tables by compute budget
`C = B * R * A` instead of by protocol family, and asks which way of spending a
fixed `C` pays: more agents, more rounds, a longer reasoning allowance, or more
independent samples.

## Build

```bash
./build.sh                 # analysis -> figures -> lint -> compile -> log check
./build.sh --self-check    # accounting checks only, no figures
RUN_DIR=/path/to/artifacts/openrouter_agent_allocation_v1 ./build.sh
```

`build.sh` uses the run's own venv (`../../.venv/bin/python`) and TinyTeX on
`/home/mabdel03/.local/bin`. Output is `report.pdf`.

## Layout

| file | role |
|---|---|
| `isocore.py` | loading, the `C = B*R*A` accounting, the self-check, palettes |
| `isoframe.py` | the master frame: paired item-clustered bootstrap over every checkpoint |
| `isostyle.py` | shared figure style |
| `fig_*.py` | one module per figure group |
| `isocompute.py` | driver: checks, 11 figures, `numbers.json`, `tables/*.tex` |
| `check_numbers.py` | asserts every number in the prose traces to `numbers.json` |
| `report.tex` | the report (abstract, methods, results, interpretation) |

## What the build verifies

1. Derived `C` equals the sealed `requested_reasoning_tokens` on every panel, and
   `B*R` equals the sealed generation count except for `deep` (one call by construction).
2. Recomputed accuracies match the published run report: `sample/1` 0.835,
   `private_team/5/4` 0.8875, `peer/4/4` 0.855, `deep/8` 0.885, oracle pass@25 0.9375.
3. `private_team/n/0` = `peer/n/0` = `sample/n` on all 400 items, for every `n`.
4. The two frozen contrasts are quoted from `primary_contrasts.parquet`, not recomputed.
5. No em dashes; every quoted number traces to `numbers.json`; no undefined refs
   or overfull boxes.

Nothing in the sealed run is modified. The analysis reads the sealed tables only.

## Forced-budget runs (openrouter_forced_budget_v2)

`RUN_DIR=<run_root>/merged ./build.sh` detects a forced run (its `tables/effective_depth_summary.parquet`, written by
`allocation analyze` / `allocation merge-shards` when `budget_semantics` is `forced_exact`) and runs
`forced_ladder.py` instead: the `C = B*R*A` self-check, realized compute per checkpoint (must equal `C`), and
**effective depth** per checkpoint (reasoning tokens before loop onset: first 3,000-char window, step 500, whose zlib
ratio is below 0.1, as a fraction of the reasoning x its budget; summed over the checkpoint's own generations).
The deep ladder reaches `deep/25` there. Outputs: `tables/forced_ladder.*`, `tables/effective_depth.*`,
`numbers_forced.json`.
