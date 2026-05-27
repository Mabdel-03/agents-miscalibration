# Operations runbook

Everything needed to run the harness on the MIT Engaging / ORCD SLURM cluster, plus the
gotchas already hit and fixed (so they aren't re-hit).

## Cluster facts

- **Scheduler:** SLURM. `MaxArraySize=25000`, **`MaxSubmitJobs=500`** (per association —
  this is why the sweep is submitted in chunks, see below).
- **GPU partitions used:** `pi_tpoggio` (8× A100-80GB, 1 node `node3807`, 7-day) for
  servers; `ou_bcs_normal`/`ou_bcs_low` (A100/H100, 1-day) for a 2nd server set;
  `mit_preemptable` (CPU-OK, 2-day, preemptible) for cell workers. **Never `pi_manoli`.**
- **Storage:** weights + results on scratch (`/orcd/scratch/orcd/012/mabdel03`, 254 TB
  free). The project dir (`/orcd/data/tpoggio/001`) has little space — code only.

## Environments

Two conda envs (see [env/SETUP.md](../env/SETUP.md)); created at explicit prefixes under
`/home/mabdel03/conda_envs/` (which is NOT in the cluster's default `envs_dirs`, so they
must be activated by **full prefix**, not name — `slurm/common.sh` does this).

| env | prefix | purpose |
|-----|--------|---------|
| `serve_env` | `/home/mabdel03/conda_envs/serve_env` | vLLM 0.21 (+ matched torch/CUDA, + `libstdcxx-ng`) |
| `asys_env` | `/home/mabdel03/conda_envs/asys_env` | the harness/analysis package (`pip install -e ".[embeddings,dev]"`) |

Run tests with: `PYTHONPATH=src /home/mabdel03/conda_envs/asys_env/bin/python -m pytest tests/ -q`

## Hugging Face auth

GPQA (`Idavidrein/gpqa`) is **gated**; Qwen3 and the other datasets are not. The token is
read from a **private, untracked file** so it never lands in git:

```bash
mkdir -p ~/.config/agents_scaling && chmod 700 ~/.config/agents_scaling
read -rs HFTOK && printf '%s' "$HFTOK" > ~/.config/agents_scaling/hf_token \
  && chmod 600 ~/.config/agents_scaling/hf_token && unset HFTOK    # paste token at the silent prompt
```

`slurm/common.sh` loads it into `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` for every job. Also
click "Agree and access" on the gated dataset page once for your HF account.

## Running the full sweep (the real thing)

The full grid is `configs/full_sweep.yaml` (~11,520 cells). It runs as: **one vLLM server
per model size** (long-lived) + **chunked cell arrays** (resumable). The cell list lives in
`<run>/cells.json`; arrays index into it by global task id.

```bash
RUN=full_sweep_v1
export ASYS_RESULTS_ROOT=/orcd/scratch/orcd/012/mabdel03/agents_scaling_results
export HF_HOME=/orcd/scratch/orcd/012/mabdel03/.cache/huggingface
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling
PY=/home/mabdel03/conda_envs/asys_env/bin/python

# 1. servers — primary set on pi_tpoggio (7-day)
PYTHONPATH=src $PY slurm/launch_sweep.py --serve-only --config configs/full_sweep.yaml \
  --run-id $RUN --serve-partition pi_tpoggio --serve-gpu-type a100 --serve-time 7-00:00:00

#    (optional) 2nd endpoints for a couple of sizes on ou_bcs to spread cell load:
for sz in 4B 8B; do
  PYTHONPATH=src $PY -m agents_scaling.serving.launch_server --model-size $sz \
    --run-root $ASYS_RESULTS_ROOT/$RUN --partition ou_bcs_normal --gpu-type a100 --time 1-00:00:00
done

# 2. wait until every size has a registered endpoint (servers download+load+compile)
PYTHONPATH=src $PY - <<'EOF'
from agents_scaling.serving import registry as R
import os
run=os.path.join(os.environ["ASYS_RESULTS_ROOT"],"full_sweep_v1")
for s in ["0.6B","1.7B","4B","8B","14B","32B"]:
    print(s, [e.base_url for e in R.list_servers(run,s)] or "PENDING")
EOF

# 3. submit the cell list as chained chunks of <=480 (under MaxSubmitJobs=500)
PYTHONPATH=src $PY slurm/launch_chunked.py --config configs/full_sweep.yaml --run-id $RUN \
  --chunk-size 480 --throttle 480 --cell-partition mit_preemptable --cell-time 2-00:00:00
```

`launch_chunked.py` writes `<run>/chunk_jobs.json` and submits ~24 array jobs, each gated
on the previous (`--dependency=afterany`) so only one chunk (≤480 tasks) is ever live.

> **Why chunks, not one big array?** A single `0-11519%N` array counts all 11,520 elements
> against `MaxSubmitJobs=500` and is rejected. Chained chunks of ≤480 keep live jobs under
> the cap while still covering the whole grid.

## Monitoring

```bash
squeue -u $USER -o "%.10i %.18j %.2t %.10M %.20R"          # jobs
ls $ASYS_RESULTS_ROOT/$RUN/cells | wc -l                    # cells touched
find $ASYS_RESULTS_ROOT/$RUN/cells -name meta.json | wc -l  # cells COMPLETED (of 11520)
tail -f $ASYS_RESULTS_ROOT/$RUN/logs/cell_*_*.out           # a worker's log
```

Progress + partial aggregation any time (reads only completed cells):
```bash
PYTHONPATH=src $PY scripts/aggregate_results.py --run-id $RUN --out analysis/$RUN.parquet
```

## Resuming after kills / preemption

The runner is **idempotent**:
- a cell with `meta.json` is skipped entirely;
- a partial `results.jsonl` resumes from the first unanswered `qid`.

So to resume, just **re-submit** the chunked arrays (same command as step 3) — completed
work is not redone. Preemption on `mit_preemptable` is therefore safe.

## Server keepalive (long runs)

Servers have walltime limits (pi_tpoggio 7-day, ou_bcs 1-day). For a multi-week run they
will be killed and must be relaunched, or cells will block on `wait_for_server`. Re-run the
step-1 launch commands for any size whose endpoint is gone:
```bash
# which sizes currently have NO live endpoint?
PYTHONPATH=src $PY - <<'EOF'
from agents_scaling.serving import registry as R; import os
run=os.path.join(os.environ["ASYS_RESULTS_ROOT"],"full_sweep_v1")
print([s for s in ["0.6B","1.7B","4B","8B","14B","32B"] if not R.list_servers(run,s)])
EOF
```
(Stale registry files from a dead server are harmless — `wait_for_server` will get a
connection error and the cell retries via `backoff`; relaunching republishes a fresh
endpoint. Consider a cron/loop that relaunches missing sizes.)

## Cost / wall-clock projection

Measured on the reasoning pilot: a 1.7B **decentralized + thinking** cell runs ~32 s/question
(6 agent-turns × two-call). Extrapolated, the full 11,520-cell grid is on the order of
**~60k A100-hours ≈ weeks–months** even well-parallelized; 32B + unlimited-thinking cells
dominate. This is an accepted, deliberately-large run; the chunked+resumable design is what
makes it survivable. To get first results faster, point the same machinery at a trimmed
config (fewer seeds / prompt levels / benchmarks).

## Gotchas already fixed (don't re-hit)

| Symptom | Cause | Fix (in repo) |
|---|---|---|
| `vllm: error: unrecognized arguments: --disable-log-requests` | vLLM ≥0.21 removed the flag (logging off by default) | dropped from `serve_qwen.sbatch.tmpl` |
| `mamba activate serve_env` → "prefix does not exist" | envs live outside the cluster `envs_dirs` | activate by full prefix (`$ASYS_SERVE_ENV`) in `common.sh` |
| `GLIBCXX_3.4.26 not found` loading flashinfer `sampling.so` | system libstdc++ too old | `libstdcxx-ng` in `serve_env` + `LD_LIBRARY_PATH` prepend in `common.sh` |
| `Invalid HF URI 'hf://datasets/truthful_qa…'` | newer `huggingface_hub` rejects bare repo id | use `truthfulqa/truthful_qa` |
| MATH `FileNotFoundError … hendrycks/competition_math` | HF disabled dataset scripts | use `HuggingFaceH4/MATH-500` (parquet) |
| thinking on but `reasoning_content` empty | vLLM 0.21 qwen3 parser names it `reasoning` | client checks `reasoning`/`reasoning_content`/`model_dump()` |
| cell array `sbatch` returns non-zero | 11,520-task array exceeds `MaxSubmitJobs=500` | `launch_chunked.py` (chained ≤480-task chunks) |
| 32B server fails at engine init: "KV cache … larger than available" | 32B weights leave <8 GiB KV on one A100-80GB at 32K ctx | 32B `max_model_len=16384` in `models.py` (or raise `--gpu-memory-utilization` / use tp=2) |
