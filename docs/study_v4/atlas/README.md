# Scaffolding Atlas

`atlas.pdf` — how agent-system configuration reshapes model behaviour, and what changes inside
the model. Exploratory, uncorrected findings from the `study_v4` run.

## Build

```bash
bash build.sh          # prepends TinyTeX to PATH, then pdflatex -> bibtex -> pdflatex x2
```

The conda `pdflatex` that appears first on `PATH` has an empty `texmf` tree and fails on every
package; `build.sh` puts `/orcd/home/002/mabdel03/.TinyTeX/bin/x86_64-linux` in front. There is no
`biber` on this machine, so the bibliography is `natbib` + `bibtex`.

## Regenerating the numbers

Every figure and table reads only `data/atlas_data.json`, which is produced by the scripts below
and carries the source, sample size and generating script for each key. Nothing in `atlas.tex` is
transcribed from an intermediate report.

```bash
PY=/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python
cd scripts
$PY s1_design.py      # design coverage, axis levels, nominal-vs-realised cell collapse
$PY s2_behaviour.py   # configuration -> behaviour: policies, difficulty, compute, scale, calibration
$PY s3_dynamics.py    # within-episode dynamics from the item files (peer rounds, hub cycles)
$PY s4_geometry.py    # representational geometry, recomputed from the raw fp16 vectors  (~12 min)
$PY s5_probes.py      # linear decodability probes                                        (~10 min)
$PY merge.py          # fragments -> data/atlas_data.json
$PY s6_figures.py     # figures/*.pdf
$PY s7_tables.py      # tables/*.tex
$PY s8_novelty.py     # tables/t_novelty.tex from data/novelty.json
$PY check_numbers.py  # asserts every prose number traces to atlas_data.json
```

`s4`/`s5` read the activation cache at `/orcd/data/tpoggio/001/mabdel03/.cache/mech_cache/`
(per-anchor `fp16` memmaps, PCA-256 reductions and `index.parquet`). If that cache is gone it can
be rebuilt from `<run_root>/neural/native/*.npz`.

## Layout

| path | contents |
|---|---|
| `atlas.tex`, `refs.bib` | source; 130 deduplicated references |
| `figures/` | 10 vector figures, matplotlib |
| `tables/` | 9 generated booktabs tables |
| `data/atlas_data.json` | 92 provenance-carrying keys — the single source of truth |
| `data/novelty.json` | per-claim literature verdicts after adversarial adjudication |
| `scripts/` | the recomputation pipeline above |

## Caveats that travel with every number

Uncorrected and exploratory: the confirmatory analysis of this run rejected nothing. One model
family, one sampling recipe, two benchmarks, a self-judged exam metric with no audit artifact, and
correlational mechanism only. Appendix B of the PDF lists the known data defects, including the
defective `outcome_agreement` column, module D's missing code half and the ~72%-complete
activation capture.
