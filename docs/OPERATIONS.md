# Legacy and development operations

> **Not the schema-5 production procedure.** The chunk arrays, editable environments,
> scratch defaults, `meta.json` completion shortcut, and legacy keepalive commands below
> describe the retired v1 system or local development. Do not use them to start or resume
> an authoritative sweep. Use
> [SCHEMA5_RECOVERY_RUNBOOK.md](SCHEMA5_RECOVERY_RUNBOOK.md) for the homogeneous schema-5
> run, whose control state is `.dispatcher-schema5-v1` and whose only accepted code and
> environments come from the sealed release bundle.

Everything needed to run the harness on the MIT Engaging / ORCD SLURM cluster, plus the
gotchas already hit and fixed (so they aren't re-hit).

## Cluster facts

- **Scheduler:** SLURM. `MaxArraySize=25000`, **`MaxSubmitJobs=500`** (per association —
  this is why the sweep is submitted in chunks, see below).
- **GPU partitions used:** `pi_tpoggio` (8× A100-80GB, 1 node `node3807`, 7-day) for
  servers; `ou_bcs_normal`/`ou_bcs_low` (A100/H100, 1-day) for additional servers;
  `mit_preemptable` (CPU-OK, 2-day, preemptible) for cell workers. **Never `pi_manoli`.**
- **`pi_tpoggio` GPU QOS cap.** Despite 8 physical A100s, a group QOS limits how many GPUs
  you can hold there at once; extra server jobs sit `PD` with reason **`QOSGrpGRES`**.
  Practically this caps the primary server set at ~3–4 sizes on pi_tpoggio — **put the rest
  on `ou_bcs_normal`** (where 4B/8B/32B run fine). Spreading sizes across clusters is the
  intended design (the registry is multi-endpoint), not just a speedup.
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

## Retired v1 full-sweep procedure (forensics only)

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

# 3. dispatch the cell list as chunks via the DRIVE loop (run under nohup; it's long-lived)
nohup env PYTHONPATH=src $PY slurm/launch_chunked.py --config configs/full_sweep.yaml \
  --run-id $RUN --chunk-size 400 --throttle 36 --submit-cap 440 \
  --cell-partition mit_preemptable --cell-time 2-00:00:00 > $ASYS_RESULTS_ROOT/$RUN/driver.log 2>&1 &
# NOTE: --throttle 36 (~6 cells/server x 6 servers), NOT 400 — see the throttle note below.
```

`launch_chunked.py` renders 29 chunk sbatches (400 cells each) and **drives** them: submit
one chunk, wait until the submitted-job count leaves headroom under `--submit-cap`, submit
the next. It writes `<run>/chunk_jobs.json` and skips chunks whose cells already have
`meta.json` (resumable — a killed driver just re-runs the same command).

> **Throttle must match server capacity, NOT the submit cap.** `--throttle` is how many
> cell tasks run concurrently against the (few) vLLM servers — each cell fans out to
> `n_agents` agents plus a forced-answer probe, so throttle 400 against 6 servers means
> ~1000+ concurrent requests. That swamps the servers: requests queue behind
> unlimited-thinking generations and blow the client timeout, producing an
> `APITimeoutError`/`ReadTimeout` storm (observed: ~197 cells failed with 0 rows). Budget
> roughly **~6 concurrent cells per server endpoint** → **throttle ≈ 36 for 6 servers**.
> vLLM continuous batching keeps them busy at that level; the client also tolerates bursts
> (600s timeout, 8 retries). Raise throttle only if you add server endpoints. (The
> resumable runner meant the storm cost nothing permanent — relaunching at throttle 36
> resumed the partial cells.)
>
> **Why a driver, not one big array or dependency chain?**
> - A single `0-11519%N` array counts all 11,520 elements against the **association**
>   `MaxSubmitJobs=500` → rejected.
> - Dependency-chaining many chunks still fails: `mit_preemptable`'s **QOS**
>   `QOSMaxSubmitJobPerUserLimit=448` counts *every submitted array task* (PD or R), so
>   chunk1's 400 tasks + chunk0's 400 = 800 > 448 → the 2nd chunk's `sbatch` is rejected
>   even with a dependency.
> - So chunks are serialized by the drive loop, keeping submitted tasks ≤ `--submit-cap`.
>   `_my_submitted_count()` uses `squeue -r` (array tasks expanded) to measure it.

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

## Scaling the server fleet (more GPUs = faster)

Throughput is GPU-bound: with 1 server/size (6 GPUs) the sweep runs ~15 cells/hr. Add
**replicas** per size so cells round-robin across more endpoints (registry/client/runner are
already multi-endpoint). Replicas use distinct ports (`_port_for(size, replica)` offsets by
replica index) so they don't collide even co-located.

> **Interleave cells when scaling concurrency.** Cells are sorted by `cell_id` (clustered by
> model size), so a chunk is single-size unless interleaved — at high throttle that sends all
> running cells to one size's server while the rest idle (observed: 240 cells stuck on one
> 0.6B server, 0 rows). `launch_chunked.py` interleaves `cells.json` by default; pass
> `--reshuffle` to regenerate it (only when no array is mid-flight against the old ordering —
> resume is by `cell_id` so completed work is safe regardless).

Bring the fleet up to a target with the one-shot, idempotent **`scale_up.py`** (tops up to
the desired count; re-runnable):
```bash
SPEC="0.6B:1:pi_tpoggio:7-00:00:00,1.7B:2:ou_bcs_low:1-00:00:00,4B:3:ou_bcs_low:1-00:00:00,\
8B:4:ou_bcs_low:1-00:00:00,14B:5:ou_bcs_low:1-00:00:00,32B:6:ou_bcs_low:1-00:00:00"
PYTHONPATH=src $PY slurm/scale_up.py --run-id $RUN --spec "$SPEC" --dry-run   # preview
PYTHONPATH=src $PY slurm/scale_up.py --run-id $RUN --spec "$SPEC"             # launch the missing replicas
```
Spec = `size:count:partition:time` per size (count = TOTAL desired endpoints for that size).
Then **restart keepalive with the SAME spec** so it maintains the larger fleet, and **raise
the cell throttle** to ~12×endpoints (see below).

> **Weight replicas to the SLOW models.** Work is dominated by big models (32B ≈ 40%, 14B+32B
> ≈ 62% of total), so to minimize wall-clock give 32B/14B the most replicas (e.g. 6/5) and the
> small models 1–2. Small cells clear fast but are not the long pole.
>
> **Respect per-user GPU QOS caps:** `ou_bcs_low` = 32 A100/H100 (biggest), `ou_bcs_normal` =
> 8+8, `pi_tpoggio` ≈ 3–4, `mit_preemptable` = 4, `mit_normal_gpu` = 2. Concentrate replicas
> on `ou_bcs_low`. Over-cap just leaves jobs PD (harmless). Note some nodes (e.g. node3904)
> are shared across `ou_bcs_low`/`mit_preemptable`/`pi_manoli` partitions — submitting to
> `ou_bcs_low` is correct even if the node also appears under pi_manoli; `squeue %P` shows the
> partition the job actually runs under.

## Server keepalive (long runs) — automated

Servers have walltime limits (pi_tpoggio 7-day, ou_bcs 1-day). Over a multi-week run they
get killed; without relaunching, cells block on `wait_for_server`. **`slurm/keepalive.py`**
handles this automatically — run it under nohup for the duration of the sweep:

```bash
# DURABLE: run the driver + keepalive as SLURM JOBS, not login-node nohup.
# On this cluster, login-node user processes (even with setsid+disown / PPID=1) get
# reaped when sessions end. Twice we lost ~280 and ~100 cells of progress when both
# loops died this way. The fix: submit them as SLURM batch jobs (CPU-only, tiny) so
# they run on compute nodes independent of any login session.
PYTHONPATH=src $PY slurm/launch_loops.py --run-id $RUN --spec "$SPEC"
# This submits two --requeue jobs (loop_driver + loop_keepalive) on mit_preemptable.
# --requeue makes SLURM auto-restart them on preemption/walltime; the loops are
# resumable, so a restart is safe. Find them via:
squeue -u $USER -h -o "%i %j %t %R" | grep asys-loop
```

Each pass: probe a real HTTP `/health` on every endpoint (registry presence ≠ liveness);
prune stale registry files for dead endpoints; and for each size, if the serve-job count
(R/PD/CF — authoritative, since live endpoints are a subset of running jobs) is below the
target, relaunch enough replicas on free replica-ids to reach the desired count. Stateless/
idempotent — kill and restart freely. Check current state any time:
```bash
PYTHONPATH=src $PY slurm/keepalive.py --run-id $RUN --spec "$SPEC" --once   # one pass, then exit
```

> A dead endpoint with a stale registry file is also tolerated at the cell level
> (`backoff` retries the connection error), but keepalive removes the stale file and
> restores a live server so cells don't waste retries / eventually fail.

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
