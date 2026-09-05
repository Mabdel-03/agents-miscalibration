# BigCodeBench evaluator container (verified 2026-09-05 18:20 EDT)

- SIF: `/orcd/data/tpoggio/001/mabdel03/containers/bcb-evaluate-v0.2.4.sif` (9.1 GB, pulled from docker://bigcodebench/bigcodebench-evaluate:v0.2.4 on the login node; sha256 to be recorded in FROZEN.yaml by profile/freeze).
- Inside: Python 3.10.16, numpy 1.21.2, pandas 2.0.3, scikit-learn 1.3.1, scipy, matplotlib, seaborn; `import bigcodebench` works (0.2.4).
- Working exec (6 s cold start; run ONE exec per cell with an in-container loop, not one per candidate):
  `apptainer exec --containall --cleanenv --no-home --net --network none --bind <workdir>:/work --pwd /work <SIF> python3 /work/bcb_container_driver.py /work/jobs.jsonl /work/results.jsonl`
- `python3 -m unittest -q solution` on a canonical solution + its test returned `OK`.
- Gotcha: with `--no-home`/`--containall`, fontconfig prints "No writable cache directories" and matplotlib may be slow/fail on font cache → the driver must `os.environ.update(MPLBACKEND="Agg", MPLCONFIGDIR="/work/.mpl", XDG_CACHE_HOME="/work/.cache", HOME="/work/.home", TMPDIR=<per-job dir>)` and create those dirs before running each candidate subprocess.
- Set `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` (already exported in slurm/common.sh) — never let apptainer write to $HOME.
