# Forced-budget allocation report (`openrouter_forced_budget_v2`)

`report.pdf` is the deliverable (12 pages, 10 figures): the v1 iso-compute comparison of agents, rounds,
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
| `render.py` | placeholder rendering |

Inputs (read only): `agents_scaling_results/forced_budget_v2/artifacts/merged/tables/` (merge of the
Mancer and Parasail shards), `artifacts/calibration.json`, and the v1 sealed tables for Figure 1.
