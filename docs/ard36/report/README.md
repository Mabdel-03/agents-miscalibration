# ARD36 report

`ard36_report.pdf` — the full-length characterization of multi-agent round dynamics, over the
**repaired** tables (`tables_v2`). 12 pages, 8 figures.

The frozen v1 tables are quoted in exactly one place (the before/after table in §2) and nowhere
else. Every other number comes from `tables_v2`, because a defect in v1 inverted one of the
study's three pre-registered contrasts — see `../05_stage0_repair.md`.

## Rebuild

```bash
# figures, straight from tables_v2 — the report never quotes a number the figures cannot reproduce
python docs/ard36/report/make_figures.py /orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36
cp <run_root>/report/figs/*.pdf docs/ard36/report/figs/

cd docs/ard36/report && tectonic ard36_report.tex
```

`tectonic` is the working engine on this cluster; the local `pdflatex` has no format file
(`mktexfmt` is broken). `tectonic` fetches packages on demand and needs network access on first
run. `make_figures.py` needs `pyarrow` + `matplotlib` (present in `asys_env`).

## Where the content comes from

| Source | What it carries |
|---|---|
| `<run_root>/tables_v2/` | the repaired tables every figure reads |
| `../04_data_quality.md` | the Q1–Q30 data-quality register |
| `../05_stage0_repair.md` | the ten repairs, each with its falsifiable verification |
| `<run_root>/analysis_v2/TRENDS.md` | the full ranked trend table (2,295 lines) |
| `<run_root>/analysis_v2/ENUMERATION.md` | the frozen manifest: 1,441 cells, 692 estimable |
| `<run_root>/analysis_v2/COVERAGE.md` | which enumerated cells were and were not computed |
| `<run_root>/analysis_v2/PAPER_SPINE.md` | the ICLR argument, objections stated at full strength |

## Caveats the report states, repeated here so they are not lost

- The headline rests on **13 discordant items**, exactly at the minimum for a paired exact test.
  It reproduces three independent ways, but it is one contrast on one model family.
- Identification comes from randomized protocol assignment, **not** from the shared root baseline.
  Round 0 is bit-identical across all seven arms, so the pre-registered "root→r3 change" is
  algebraically the r3 *level* contrast. The shared root buys variance reduction only.
- **35 of 39** adversarially verified claims were *weakened*, 4 refuted, 0 confirmed as stated.
- E6 neural capture is descriptive only — a minority-vs-bloc probe has MDE₈₀(AUC) ≈ 0.73 on this
  panel and was deliberately not attempted.
- The E2c challenge cohort was **not run**: 27 eligible items under the repaired grading rule,
  below the testability floor.
