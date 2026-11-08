# Forced-budget allocation report (`openrouter_forced_budget_v2`)

`report.pdf` is the deliverable (23 pages, 19 figures): the v1 iso-compute comparison of agents, rounds,
sampling and single-call depth, rerun with every generation forced to spend exactly its reasoning budget
(Qwen3.8-27B; AIME 2025+2026, HMMT Feb 2025+2026, Omni-MATH >= 7, GPQA-Diamond; 96 items; A = 1024).

## Headline

- Forcing works: all 13,344 main-run generations realized exactly their budget.
- Once depth is real, it dominates. deep/16 delivers 0.792 macro against 0.417 for sample/16 at equal
  realized compute (+37.5 points, 95% CI +28.1 to +46.9), and exceeds oracle pass@16 of the sampling
  bank by 21.2 points. deep/4 beats sample/25 by 18.3 points. Every point on the $-frontier is a deep call.
- Carried by competition math; on GPQA-Diamond depth and serial revision tie (a selection regime, as in v1).
- Serial revision is the second lever; peer exposure now helps slightly (it hurt in v1).
- Deep calls loop more as they get longer (28% at 16A), but looping calls are the ones that had already
  solved the problem (accuracy 0.96 vs 0.73 at 16A).

## Generation versus selection (gap.py)

- At every budget carrying a deep call, only the deep call delivers more than pass@k of 1024-token samples.
- Structured multi-call allocations generate above pass@k (revision chains +6.0 pts coverage) but capture only
  0.34-0.61 of their own headroom; no simulated verifier (AUC 0.5-1.0) over samples, chains or teams reaches depth
  (perfect selection over 16 samples is 23.0 pts below deep/16). GPQA is the exception.
- 37/96 items have no correct sample in 25; deep/16 solves 23. Low vote agreement flags them.
- Slopes per doubling: deep 11.0, pass@k 5.7, revision 4.8, vote 2.0. Deep portfolios add ~1 pt coverage.
- Section 5 costs a width-at-depth follow-up (~$37).

## Flat CSVs for analysis (`export_flat.py`)

`./../../.venv/bin/python export_flat.py` writes, from the merged sealed tables:

| file | grain | rows x cols |
|---|---|---|
| `data/forced_budget_v2_item_config.csv` | item x configuration x view | 20,218 x 52 |
| `data/forced_budget_v2_generations.csv` | one API generation node | 14,031 x 35 |
| `data/DATA_DICTIONARY.csv` | column meanings | |

The item x configuration file is the central one: design coordinates (family, B, R, A, C), delivered outcome,
pool coverage and selection gap, tokens and cost, effective depth, and each item's own reference numbers from
the 25-sample bank (difficulty, pass@k / prefix coverage / vote readout at that configuration's own k = C/A).
Filter `partition_id == "main"` and `view == "current"` (9,696 rows) for the study panel as reported; `view ==
"archive"` gives the same configurations scored over every candidate they generated.

## Build

```bash
./build.sh                 # analysis (numbers.json, tables, figures) -> render -> lint -> compile -> log check
./build.sh --no-analysis   # re-render and compile only
```

`report.tmpl.tex` is the source. Every number in the prose is a `{{path|fmt}}` placeholder filled from
`numbers.json` by `render.py`; an unresolved placeholder fails the build, so no quoted number is typed by hand.
`report.tex` is generated.

| file | role |
|---|---|
| `core.py` | loading, `C = B*R*A`, the paired item-clustered bootstrap frame (run's own `analysis.stats`) |
| `analysis.py` | every number, table and figure, from the merged sealed tables |
| `figures.py` | the ten figures |
| `gap.py` | generation/selection decomposition, verifier sweep, strata, vote calibration, portfolios, dynamics, slopes |
| `gfig.py` | the nine generation/selection figures |
| `render.py` | placeholder rendering |

Inputs (read only): `agents_scaling_results/forced_budget_v2/artifacts/merged/tables/` (merge of the
Mancer and Parasail shards), `artifacts/calibration.json`, and the v1 sealed tables for Figure 1.
