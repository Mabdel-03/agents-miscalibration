# agent2

SUMMARY: Operations plan for running agent_design_v4 (32B flagship + 4B/8B/14B dense panel) on ORCD by Mon 09:00 EDT. Verified scheduler facts change the brief in two places: pi_tpoggio's QOS has a group cap of 6 A100 / 750 GB / 144 CPU (so only 3 × 32B-long TP2 fit there and no 14B), and the user belongs to ou_bcs_low (tier 10, preemptable, 1-day, per-user cap 32 A100 + 32 H100, four idle 8×A100 nodes right now), which the brief omits. Recommended fleet: 3 × 32B-long TP2 + 14B-long + co-served 4B/8B pair on pi_manoli (idle, no caps, 7-day), 3 × 32B-long on pi_tpoggio, 2 × 32B-long + 2 × 14B + 8B + 4B on ou_bcs_normal A100s (8-GPU cap), plus opportunistic 32B-long replicas on ou_bcs_low (6) and H100 (2). Cell workers run as 1-CPU/4G array tasks primarily on pi_manoli's 128 spare CPUs (non-preemptable, requires lowering the legacy 240 GB/server memory reservation), overflow on mit_preemptable. Compute model at 3,500 tok/gen and B4 binding at the 64-call cap gives tier 1 (N_main=400) ≈ 416M generation tokens + ~150k judge calls ≈ 13 h on the mid fleet (8 core + 4 opportunistic TP2 servers), tier 2 ≈ 5–6 h, dense panel 8–12 h on its own lanes; N_main=400 is recommended with a prospectively declared hash-prefix fallback to 300, and M committed at 100 items (150 as nested extension). Timeline: fleet up Sat 17:00, code freeze Sat 22:30, pilot to Sun 00:30, tier 1 dispatch Sun 00:30, tier 2 append Sun 08:00, tier-1 seal + judge/eval wave Sun 14:00, tier 3 Sun 18:00, hard stop Mon 06:00, aggregation to 09:00. Storage ≈ 50–80 GB and ~25k inodes on data; HOME must receive nothing (852k/1000k files).

# Operations / throughput / timeline plan for agent_design_v4 on ORCD (Sat 2026-09-05 15:40 EDT → Mon 09:00)

All cluster numbers below are either from the brief or were verified by me now with read-only commands (`sacctmgr show qos`, `scontrol show partition/node`, `squeue`, `sinfo`, `pip show`, `quota -s`, `df`). Where I disagree with the brief I say so explicitly (section 0).

## 0. Verified facts that change the brief

| Fact | Command | Consequence |
|---|---|---|
| `pi_tpoggio` QOS **GrpTRES = cpu=144, gres/gpu:a100=6, mem=750G** (group cap) | `sacctmgr show qos pi_tpoggio -P format=Name,GrpTRES` | Max **3 × `32B-long` TP2** there (6 GPUs, 3×240 GB = 720 GB ≤ 750 GB, 48 CPUs). The brief's `14B-long:1:pi_tpoggio` would sit PD forever with reason `QOSGrpGRES` (docs/OPERATIONS.md already documents this symptom). "7 usable" is physical, not schedulable. node3807 currently has 4 GPUs held by others (3 by a `mit_preemptable` job we preempt, 1 by an `ou_bcs_low` job we do not). |
| User is in group `orcd_rg_par_ou_bcs_low`; partition `ou_bcs_low`: PriorityTier=10, PreemptMode=REQUEUE, MaxTime 1-00:00:00, QOS MaxTRESPU `gres/gpu:a100=32, h100=32`, MaxSubmitPU=256 | `id -Gn`, `scontrol show partition ou_bcs_low`, `sacctmgr show qos ou_bcs_low` | Large opportunistic pool the brief omits. Right now node3810/3906/3910/3911 are IDLE (31 free A100s). Servers there can be preempted by any tier ≥ 25 job (mit_normal 25, ou_bcs_normal 50, pi_* 100); keepalive relaunches; cells must tolerate endpoint death. Note node3904 (pi_manoli) and node3807 (pi_tpoggio) are also members of ou_bcs_low, so launch pi_* servers first. |
| `ou_bcs_normal`: tier 50, PreemptMode=OFF, 1-day, QOS `a100=8, h100=8, cpu=448`, MaxSubmit 256; H100 nodes (node1802/1803/2702/2703/2802, 4×H100 each) have ≤2 free GPUs and 11 pending single-H100 jobs ahead | `scontrol show node`, `squeue -p ou_bcs_normal -t PD` | A100 entries start promptly (idle nodes); H100 entries are low-odds; treat as bonus. |
| `mit_normal` QOS **MaxTRESPU cpu=96, mem=386G** | `sacctmgr show qos mit_normal` | At most 96 concurrent 1-CPU cell tasks there; not enough for ~112 concurrent cells. |
| `mit_preemptable` QOS: cpu=1024, gres/gpu=4, mem=4T, MaxSubmit 448, tier 1, REQUEUE | same | Fine for overflow/eval; cells there get preempted (resumable). |
| `pi_manoli` = node3904: 192 CPU, 1,031 GB RAM, 8×A100, **IDLE**, QOS `pi_manoli` has no limits, tier 100, PreemptMode=OFF, 7-day | `scontrol show node node3904` | Best home for the 32B core, the dense lanes, the control loops **and the cell workers** (128 CPUs left after servers), provided server host-memory reservations are lowered (legacy renderer reserves `tp×120G` = 240 GB per TP2 server → 960 GB on this node would starve cells). |
| Association `mit_general` MaxSubmit=**500** (all jobs + expanded array tasks, all partitions) | `sacctmgr show assoc user=$USER` | Chunked arrays with ≤ 100 tasks; keep total queued ≤ 460. |
| Legacy renderer: `cpus = max(8, tp*8)`, `mem = f"{tp*120}G"` hardcoded in `src/agents_scaling/serving/launch_server.py:518-519`; `--gpu-memory-utilization 0.90` hardcoded in `slurm/serve_qwen.sbatch.tmpl:156`; no `VLLM_CACHE_ROOT`/`TORCHINDUCTOR_CACHE_DIR` export and `--export=NONE` → vLLM compile cache lands in `$HOME/.cache/vllm` | read | HOME is at **852k / 1000k files** (`quota -s`); 20 vLLM servers each writing an inductor cache is a real inode risk → template patch C (below) before the first launch. |
| `keepalive.tick()` counts in-flight serve jobs per (profile, partition) only (`_serve_jobs_in_flight(size, partition)`, keepalive.py:206-237, 3546) | read | Two spec entries for the same profile on the same partition with different GPU types (a100 vs h100) fight over one count. H100 entries need patch B or manual launch. |
| `rlms`, `jsonschema`, `rfc8785` are **not** installed in `asys_env` (`pip show` empty); `openai 2.38.0`, `datasets 4.8.5`, `transformers 5.9.0`, `pyarrow 24`, `backoff 2.2.1` are | `asys_env/bin/pip show …` | Any pip install must happen on the login node before freeze (compute nodes: HF offline env). |
| BigCodeBench is **not** in the HF cache; HLE shards (16 parquet, 320 MB) + `hle_verified.parquet` are in the session scratchpad under **/tmp on the login node** | `ls $HF_HOME/hub`, `ls scratchpad/hle_shards` | Compute nodes cannot see `/tmp/claude-…`; copy data to `<run_root>/data/` before any job; download BCB v0.1.4 parquet (public) on the login node. |
| `apptainer 1.4.5` at `/usr/bin/apptainer`; no `APPTAINER_*` env set; `/orcd/data/tpoggio/001/mabdel03/containers` does not exist | `which apptainer`, `env` | Create cache/tmp/containers dirs on data; pull on the login node. |
| Data FS: 15 T free, inodes 97 M / 31 B used | `df -h`, `df -i` | Storage is a non-issue on data; only HOME matters. |
| Qwen3 configs (cached snapshots match `configs/model_contracts.v1.json` revisions): 4B/8B: 36 layers, 8 KV heads, head_dim 128; 14B: 40 layers; 32B: 64 layers; all `max_position_embeddings=40960`; weights on disk 7.5/16/28/62 GiB | `config.json` in HF cache | KV math in §1.3. |
| vLLM 0.21.0 has `--kv-cache-memory-bytes` (arg_utils.py:1086) and `--gpu-memory-utilization`; the worker asserts `init free memory ≥ requested` at startup (gpu_worker.py:434, 652-680) | read | Co-serving is feasible only with sequential startup (large model first) or explicit `--kv-cache-memory-bytes`. |

Spec citations used throughout: §3.6 (seeds/aliases), §4.2 (policies, 64-call cap, DEC `N*(1+r)≤64`, CEN ≤8 cycles), §4.4 (JUDGE_BEST ≤1 call/candidate, 1,024 cap), §5.5 (R_bank=10, four cells), §6.2 (matrix), §6.5 (B0/B4, reserve-debit), §6.7-6.8 (aliases, physical union), §9.3 (bootstrap), §10.2-10.6 (profiling, ledgers, retries ≤2, preemption, isolation), §11.2 (package layout).

---

## 1. Fleet plan

### 1.1 Targets (all `32B-long` = TP2 @ 40,960 ctx; dense `<size>-long` = TP1 @ 40,960)

| Partition (tier / preempt / walltime) | Entries | GPUs | Rationale |
|---|---|---|---|
| `pi_manoli` (100 / OFF / 7-day, no QOS caps, idle) | `32B-long:3`, `14B-long:1`, **pair job** `8B-long`+`4B-long` co-served (not keepalive-managed) | 6+1+1 = 8 | Non-preemptable core; also hosts loops + cell workers on its 128 spare CPUs. |
| `pi_tpoggio` (100 / OFF / 7-day, group cap 6 GPU/750G/144 CPU) | `32B-long:3` | 6 | Exactly the group cap. Our tier-100 jobs preempt sarahpan's 3-GPU `mit_preemptable` job on node3807 (Slurm `PreemptType=preempt/partition_prio`, `preempt_youngest_first`). |
| `ou_bcs_normal` (50 / OFF / 1-day, cap 8 A100 + 8 H100) | `32B-long:2:a100`, `14B-long:2:a100`, `8B-long:1:a100`, `4B-long:1:a100`; bonus `32B-long:2:…:h100` | 4+2+1+1 = 8 A100; 4 H100 | Non-preemptable, but 24 h clocks → keepalive relaunches; launch at T0+5h so the first day covers Sat 22:00→Sun 22:00. |
| `ou_bcs_low` (10 / REQUEUE / 1-day, cap 32 A100) | `32B-long:6` | 12 | Opportunistic; four 8×A100 nodes idle now. Preemptable → cells retry on another endpoint. |

32B-long endpoints: **core 8** (3+3+2) ≈ 7,200 gen tok/s; **mid 12** (+4 of the 6 low replicas) ≈ 10,800; **max 16** (+6 low +2 H100 @ ~1,300 tok/s each, assumption) ≈ 15,200. Dense: 14B 3 endpoints (2,955 tok/s), 8B 1 dedicated + 1 co-served (≈1,163+650 = 1,813), 4B 1 dedicated + 1 co-served (≈1,241+650 = 1,891). Co-served rates are an assumption (decode is bandwidth-bound; two processes time-slice) — verify in the pilot via `/metrics`.

### 1.2 Keepalive `--spec` (legacy mode; `parse_spec` at keepalive.py:172 accepts a trailing `:gpu_type`)

```
SPEC="32B-long:3:pi_manoli:7-00:00:00,14B-long:1:pi_manoli:7-00:00:00,\
32B-long:3:pi_tpoggio:7-00:00:00,\
32B-long:2:ou_bcs_normal:1-00:00:00,14B-long:2:ou_bcs_normal:1-00:00:00,\
8B-long:1:ou_bcs_normal:1-00:00:00,4B-long:1:ou_bcs_normal:1-00:00:00,\
32B-long:6:ou_bcs_low:1-00:00:00"
# Only after patch B (gpu-type scoping in _serve_jobs_in_flight):
#   ,32B-long:2:ou_bcs_normal:1-00:00:00:h100
```
Without patch B, launch H100 servers manually once (`launch_server … --partition ou_bcs_normal --gpu-type h100 --replica 20/21`) and **raise** the a100 entry to `32B-long:4:ou_bcs_normal` so keepalive's partition count (which includes the H100 jobs) still launches 2 A100 replicas — brittle; prefer patch B (≈10 lines: parse `squeue -o "%j|%t|%P|%b"` and match `gres/gpu:<type>` when `Target.gpu_type` is set).

Replica indices are assigned globally per profile by `tick()` (keepalive.py:3552-3569) so ports never collide across partitions (`_port_for(profile, replica)` = 8000 + crc32 % 1000 + replica).

### 1.3 Co-serving 4B + 8B on one A100-80GB (template variant `slurm/serve_qwen_pair.sbatch.tmpl`)

KV bytes/token (bf16) = 2 × layers × kv_heads × head_dim × 2 B:

| Model | KV/token | KV per 40,960-token seq | bf16 weights (GiB) | Standalone @0.90 of 79.2 GiB: KV ≈ | full-length seqs / seqs @6k tokens |
|---|---|---|---|---|---|
| Qwen3-4B | 144 KiB | 5.6 GiB | 7.5 | 71.3−7.5−3 ≈ 60 GiB (437k tok) | 10 / 72 |
| Qwen3-8B | 144 KiB | 5.6 GiB | 15.3 | ≈ 53 GiB (385k tok) | 9 / 64 |
| Qwen3-14B | 160 KiB | 6.25 GiB | 27.6 | ≈ 40 GiB (262k tok) | 6 / 43 |
| Qwen3-32B TP2 | 256 KiB (128/GPU) | 10 GiB | 61.1 (30.6/GPU) | ≈ 36 GiB/GPU → 72 GiB (294k tok) | 7 / 49 |

(32B TP1 would leave <8 GiB KV, the documented OOM — never use the `32B` profile.)

Pair on one GPU: **8B at `--gpu-memory-utilization 0.60`** (47.5 GiB: 15.3 weights + ~2.5 activation + ~0.6 CUDA graphs + ~0.5 non-torch → KV ≈ 28.5 GiB ≈ 207k tokens ≈ 5 full-length / 34 @6k) then **4B at `0.28`** (22.2 GiB: 7.5 + 2 + 0.5 + 0.5 → KV ≈ 11.7 GiB ≈ 85k tokens ≈ 2 full-length / 14 @6k). Sum 0.88 + two CUDA contexts (~1.2 GiB) leaves ~8 GiB slack. Startup order matters because of the free-memory assertion: start 8B, wait for `/health` + `/v1/models` (the `--register` role's `wait_until_ready`, healthcheck.py:35), then start 4B (sees ≈30.7 GiB free ≥ 22.2 requested). Add `--max-num-seqs 48` (8B) / `32` (4B).

What the variant needs beyond `serve_qwen.sbatch.tmpl`: `#SBATCH --gres=gpu:a100:1 --cpus-per-task=16 --mem=128G --time=7-00:00:00 --job-name=asys-srv-<pool>-pair-8B-4B`; two `vllm serve` blocks (each with its own `--revision/--tokenizer/--served-model-name/--tensor-parallel-size 1/--max-model-len 40960/--max-logprobs 20/--reasoning-parser qwen3/--port`), two ports (`_port_for("8B-long", r)`, `_port_for("4B-long", r)` — distinct via crc32), two `launch_server --register --model-size 8B --profile 8B-long …` / `--model-size 4B --profile 4B-long …` calls (registry dirs `servers/8B-long/`, `servers/4B-long/`, so `lookup_server(run_root, "8B-long")` works unchanged), and `wait -n; kill $PID_A $PID_B` so a single crash kills both (clean death for relaunch). Render with a new `render_pair_sbatch()` in a new module `src/agents_scaling/serving/launch_pair.py` (do not touch the 1,000-line `launch_server.py`); keepalive does not manage it — relaunch by hand if it dies (7-day walltime, non-preemptable, so this is a rare event).

Is it worth it? It frees exactly one GPU (half a 32B server). With ou_bcs_low available it is optional, but the user asked for it; use it on pi_manoli GPU 8 only. ou_bcs_normal dense endpoints stay single-model and keepalive-managed.

### 1.4 Required small patches before launch (ops-critical, ≤ 1 h total)

- **A. Server host-memory override** (`launch_server.py:518-519`): `cpus = max(8, tp*8)`; `mem = f"{tp * int(os.environ.get('ASYS_SERVE_MEM_PER_GPU_GB','120'))}G"`. Run with `ASYS_SERVE_MEM_PER_GPU_GB=64` → 128 GB per TP2 server (vLLM loads safetensors via mmap; CPU swap 4 GiB/rank) → pi_manoli servers reserve 3×128+2×64 = 512 GB, leaving ~519 GB for 128 cells × 4 GB. Keepalive passes env through (`--export=ALL` in `launch_loops.py`).
- **B. keepalive gpu-type scoping** (optional, for H100 entries) as above.
- **C. Serve template cache exports** (`serve_qwen.sbatch.tmpl` after line 62): `export XDG_CACHE_HOME=/orcd/data/tpoggio/001/mabdel03/.cache VLLM_CACHE_ROOT=…/.cache/vllm TORCHINDUCTOR_CACHE_DIR=…/.cache/torchinductor TRITON_CACHE_DIR=…/.cache/triton`; add `--max-num-seqs 64` and a `{GPU_MEM_UTIL}` placeholder defaulting to `0.90`. keepalive's `_spooled_job_provenance` only checks presence of specific markers (keepalive.py:420-507), so extra lines are safe. Do this **first** (5 min) so the T0 servers do not write to HOME.
- **D. Cell template** (`slurm/run_cell_array.sbatch.tmpl`): re-add `{MEM}`, `{TIME}`, `{CPUS}`, `{LANE}` placeholders (`launch_chunked._render` already substitutes `MEM`/`TIME` but the template hardcodes `--mem=4G --time=12:00:00`); job-name `asys-study-{RUN_ID}-{LANE}`; exec `python -m agents_scaling.study.run_one --run-id … --cells-file … --index $SLURM_ARRAY_TASK_ID`. Keep `--signal=B:USR1@1200` + `exec` so Python receives SIGUSR1.
- **E. Pair template** (§1.3).
- **F. New loop launcher** `slurm/launch_study_loops.py` (copy of `launch_loops.py`): renders `loop_keepalive.sbatch.tmpl` unchanged (needs `--export=ALL,ASYS_ALLOW_LEGACY_CONTROL=1`) and a new `loop_study_driver.sbatch.tmpl` that execs `python -m agents_scaling.study.dispatch --run-id … --lanes lanes.json`. Submit both on `--partition pi_manoli --time 7-00:00:00` (no preemption; still keep the `afterany` self-resubmit line, never `--requeue`).

### 1.5 Launch commands, in order

```bash
export ASYS_RESULTS_ROOT=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results
export HF_HOME=/orcd/data/tpoggio/001/mabdel03/.cache/huggingface
export ASYS_SERVE_MEM_PER_GPU_GB=64
RUN=study_v4_run1; RUN_ROOT=$ASYS_RESULTS_ROOT/$RUN
PY=/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling

# 0. init run root (T0+10 min, after patch C)
mkdir -p $RUN_ROOT/{servers,logs,cells,requests,data,eval,seals} /orcd/data/tpoggio/001/mabdel03/{containers,.cache/{vllm,torchinductor,triton,apptainer/tmp}}
cp -r /tmp/claude-225593/-orcd-data-tpoggio-001-mabdel03-agents-scaling/ba72bd31-8ff0-45c2-b3b0-7ae1ad469566/scratchpad/{hle_shards,hle_verified.parquet} $RUN_ROOT/data/

# 1. core servers (T0+15 min): pi_manoli first (so ou_bcs_low jobs never land on node3904), then pi_tpoggio
for r in 0 1 2; do PYTHONPATH=src $PY -m agents_scaling.serving.launch_server --model-size 32B --profile 32B-long \
  --run-root $RUN_ROOT --partition pi_manoli --gpu-type a100 --time 7-00:00:00 --replica $r; done
PYTHONPATH=src $PY -m agents_scaling.serving.launch_server --model-size 14B --profile 14B-long \
  --run-root $RUN_ROOT --partition pi_manoli --gpu-type a100 --time 7-00:00:00 --replica 0
for r in 3 4 5; do PYTHONPATH=src $PY -m agents_scaling.serving.launch_server --model-size 32B --profile 32B-long \
  --run-root $RUN_ROOT --partition pi_tpoggio --gpu-type a100 --time 7-00:00:00 --replica $r; done
# pair job once launch_pair.py exists (T0+2h): 
PYTHONPATH=src $PY -m agents_scaling.serving.launch_pair --run-root $RUN_ROOT --partition pi_manoli --time 7-00:00:00 --primary 8B-long --secondary 4B-long --util 0.60 0.28

# 2. verify registry (T0+35 min; 32B TP2 loads in ~8-15 min from data)
squeue -u $USER -o "%.10i %.34j %.13P %.2t %.9M %.11l %.16R"
ls $RUN_ROOT/servers/32B-long/ $RUN_ROOT/servers/14B-long/
PYTHONPATH=src $PY -c "from agents_scaling.serving import registry as R; import sys
for p in ['32B-long','14B-long','8B-long','4B-long']: print(p, [e.base_url for e in R.list_live_servers(sys.argv[1], p)])" $RUN_ROOT
curl -s http://<node>:<port>/v1/models | head -c 300

# 3. loops (T0+5h, once the dispatcher exists): keepalive + study driver as pi_manoli Slurm jobs
PYTHONPATH=src $PY slurm/launch_study_loops.py --run-id $RUN --spec "$SPEC" \
  --partition pi_manoli --time 7-00:00:00 --cell-partition pi_manoli --cell-time 1-00:00:00 --cell-mem 4G \
  --chunk-size 100 --submit-cap 380 --qos-limit 460
squeue -u $USER -h -o "%i %j %t %R" | grep asys-loop     # exactly ONE of each
```
Keepalive's first tick then tops up ou_bcs_normal / ou_bcs_low to the spec (the pi_* entries are already satisfied, so it launches nothing there). To retire the fleet on Monday: `scancel -n asys-loop-keepalive` first, then `scancel -u $USER --name 'asys-srv-*'`.

---

## 2. Compute-demand table

Assumptions (stated): mean generated tokens per solver call 3,500 (HLE ≈ 4,500 under the 8,192 cap, BCB ≈ 2,500; pessimistic column 4,500); `32B-long` A100 TP2 = 900 tok/s, H100 TP2 = 1,300 (assumption), 14B 985, 8B 1,163, 4B 1,241 (July p90), co-served ≈ 650 each; judge/selector calls (32B thinking off, §4.4 1,024 cap) ≈ 300 generated + ~2,500 prefill tokens, prefill ≈ 8,000 tok/s per TP2 server (compute-bound estimate); realized B4 binds at the 64-call cap for stateless/sequential methods (brief). Per-item physical calls after exact aliases (§3.6, §6.7):

| Module (N=5, B4) | Calls/item | New physical gens/item | Sequential depth |
|---|---|---|---|
| F: 4 framing cells × R_bank 10 (§5.5) | 40 | 40 | 1 |
| S_FRESH (00 bank to 64; first 10 alias F00) | 64 | 54 | 1 (stateless; serve at concurrency 8, record mode per §5.4) |
| IND_VOTE (11; first 10 alias F11) | 64 | 54 | 1 |
| S_HISTORY (root aliases F00 draw 0; 63 revisions) | 64 | 63 | **64** |
| DEC (5 roots native wording + 8 rounds × 5) | 45 | 45 | 9 |
| CEN_FLAT (1 + 8 cycles × (4 workers + 1 hub)) | ≤ 41 (expected ~25) | ≤ 41 | ≤ 17 |
| **Tier-1 solver total** | | **297 worst / ~281 expected** | |
| JUDGE_BEST (§4.4, ≤1/candidate): F 40 + S_FRESH 64 + IND 64 + S_HISTORY 64 + DEC 45 (5 latest primary + 40 archive diagnostic) + CEN 1 | 278 | 278 short calls | 1 |
| HLE correctness judge (exact-match HLE items only: 1,229/1,600 = 77%) | 278 per such item | 278 short calls | 1 |

| Tier | Items | Solver gens | Gen tokens (3,500 / 4,500) | Judge calls (gen tok; prefill tok) | Hours core 8 TP2 (7,200 tok/s) | mid 12 (10,800) | max 16 (15,200) |
|---|---|---|---|---|---|---|---|
| **Tier 1** F + A(5 methods) | 400 (200 HLE / 200 BCB) | 118.8k worst (112k exp.) | 416M / 535M | JUDGE_BEST 111k + HLE judge 43k = 154k (46M; 385M) | 16.0 + 3.3 = **19.3** (24 @4,500) | 10.7 + 2.2 = **12.9** (16.0) | 7.6 + 1.6 = **9.2** |
| Tier 2 32B: D (25/item ×100), N-panel (DEC 9+18+27+63, CEN 1+17+25+64, IND aliases the 00 bank ⇒ 224/item ×100), IND_PRIVATE_REVISION + DEC_ONE_ROUND (~10/item ×100), E (5 new episodes × 213/item × 30), C forecasts (1,600 × 300 tok) | 100 / 100 / 100 / 30 / 400 | 2.5k + 22.4k + 1k + 32k ≈ 58k | 202M / 260M | ~60k (18M; 150M) | 7.8 + 1.4 = **9.2** | 5.2 + 0.9 = **6.1** | 3.7 + 0.7 = 4.4 |
| M dense panel (IND 64 + DEC 45 + CEN 41 = 150/item; no F banks for dense) | 150 (100) | 22.5k (15k) per model | 79M (52M) per model | judged on 32B lane: 45k (small) | 14B @2,955 tok/s: **7.4 h** (4.9); 8B @1,813: **12.1 h** (8.0); 4B @1,891: **11.6 h** (7.7) — own lanes, parallel to the above | | |
| Tier 3: M-cross 8B (N=3,9 DEC+CEN = 179/item ×100) | 100 | 17.9k on 8B | 63M | | 8B lane after M: **9.6 h** | | |
| Tier 3: M-cross 32B N=3,9 on the same 100-prefix ⇒ aliases N-panel; A-budget B1/B2 ⇒ deterministic prefix replay of B4 trajectories (§6.7 valid because no remaining-budget input is model-visible) + ≤2 forced-finalization calls per item-method; CEN_RLM dev 20 items ≤64 | 100 / 100 / 20 | ~0 / ≤1k / ≤1.3k | ≤ 8M | | < 0.5 h | | |
| **32B lane total** | | ~180k gens + ~215k judge calls | ~630M gen + ~540M prefill | | **≈ 30 h** | **≈ 19.5 h** | ≈ 14 h |

Reading: the 30 h window (Sun 00:30 → Mon 06:30) holds everything on the **mid** fleet with ~8 h margin; on the core-only fleet tier 3 (32B) and part of tier 2 must be cut. The brief's "tier 1 ≈ 14 h on 8 servers" understates by ~5 h because judge/selector prefill (≈540M tokens) is not free.

**N_main recommendation: 400**, with this rule declared in the freeze manifest (§6.2 "pre-freeze downscope must be consistent"): items are ranked by salted hash within each sub-stratum (HLE-Gold, HLE-Revision, BCB); N_main = 400 is the top 100 + 100 + 200; dispatch is in rank blocks of 50 (12–13 Gold + 12–13 Revision + 25 BCB), so at any time the completed set is a balanced prefix. Analysis uses the largest n ∈ {300, 400} whose tier-1 modules are complete on the first n ranked items by Mon 06:00; if even 300 is incomplete, report the largest completed 50-block prefix. N_main = 600 is infeasible (tier 1 alone ≈ 19 h on the mid fleet). Decision points: Sun 08:00 and Sun 12:00 (§5).

---

## 3. Cell sizing, throttles, partitions, dispatch order

A cell = (module, method, checkpoint, N, B, framing, episode_rep, item block) run by one CPU array task; two knobs per cell: `items` and `parallel_items` (independent item chains in a ThreadPool), each item using its policy's own request concurrency. Per-sequence decode on a loaded 32B TP2 server ≈ 25–35 tok/s → one 3,500-token call ≈ 2–2.5 min.

| Lane / cell type | items × parallel_items | in-flight requests per cell | est. cell wall (32B) | cells (tier 1, N=400) |
|---|---|---|---|---|
| F (one framing × block) | 20 × 2 | 20 | ~1 h | 80 |
| S_FRESH / IND | 12 × 1, concurrency 8 | 8 | 7 waves × 2.5 min × 12 ≈ 3.5 h | 34 + 34 |
| S_HISTORY | 8 × 4 | 4 | 63 × 2.5 min ≈ 2.6 h per item → ~5.5 h | 50 |
| DEC | 12 × 2 | 10 | 9 steps ≈ 23 min/item → 2.3 h | 34 |
| CEN_FLAT | 8 × 2 | ≤ 10 (often 2) | ≤ 40 min/item → ≤ 2.7 h | 50 |
| judge_hle / judge_best (32B, thinking off) | 200 candidates × 8 | 8 | ~20 min | ~700 total |
| dense M (14B/8B/4B) | IND 12×1 (c=8), DEC 12×2, CEN 8×2 | 8–10 | 2–5 h | ~40 per model |
| eval_bcb (CPU, apptainer) | 200 candidates × 1 | 0 | 200 × ~5 s ≈ 17 min | ~260 (tier 1) |

Walltime: pi_manoli cells `--time 1-00:00:00` (no 12 h cap there); mit_preemptable overflow `1-00:00:00`; mit_normal only if used `12:00:00`. Memory 4G (judge/eval 8G). Per-call journaling (`items/<source_id>.partial.jsonl` via `io.append_jsonl`) so a killed S_HISTORY/DEC/CEN item resumes from its last completed call with identical seeds; SIGUSR1 handler finishes the in-flight call, flushes, exits 0 without `meta.json`.

**Throttle**: target ≤ 48 in-flight sequences per 32B endpoint (KV table §1.3) → **8 cells per live 32B endpoint** (mid fleet: 96), 8 per dense endpoint (14B: 24, 8B: 16, 4B: 16). The driver recomputes `throttle = 8 × len(list_live_servers(profile))` every poll and applies it live with `scontrol update JobId=<array> ArrayTaskThrottle=N` (no resubmission needed). Judge lanes: 4 per endpoint, interleaved.

**launch_chunked-style parameters** for the new `study.dispatch` (copy `_drive` from `slurm/launch_chunked.py:205-266`, keep its two gates + top-up logic): `--chunk-size 100 --submit-cap 380 --qos-limit 460` (association MaxSubmit 500 minus ~25 servers, 2 loops, eval arrays), `--poll-s 120`, per-lane arrays named `asys-study-<run_id>-<lane>` (32B, 14B, 8B, 4B, judge, eval), `depends_on` per cell (an A cell for block b lists the four F cells of block b; the driver only submits cells whose dependencies have `meta.json`, and the run_one worker double-checks the F bank's `request_id` index before generating, falling back to generating under the same request identity with create-if-absent semantics so there is exactly one canonical output per `request_id`, §10.4 "duplicate submissions attach to the same committed request").

**Partition for cells**: primary **`pi_manoli`** (`--partition=pi_manoli --cpus-per-task=1 --mem=4G`): 128 CPUs / ~519 GB free after the servers (with patch A), no QOS caps, tier 100 → never preempted, 7-day. This avoids both the mit_preemptable preemptions and mit_normal's 96-CPU cap. The whole steady state (96 32B-lane + 24 dense + 8 judge cells ≈ 128) fits. Overflow (eval_bcb arrays, extra judge lanes) on `mit_preemptable` (cpu=1024, resumable per shard). Never mit_normal for the 32B lane.

**Dispatch ORDER** (cells.json order; each lane consumed in order):
1. Tier 1, block b = 0..7 (50 ranked items each, balanced): `F00, F11, F01, F10` → `S_HISTORY` (the long pole goes first so it never becomes the tail that starves the fleet) → `DEC` → `CEN_FLAT` → `IND` → `S_FRESH` → judge cells for block b (after the block's pools seal) → next block. F precedes A because A aliases F (S_FRESH/IND first 10, S_HISTORY root, D's 9 roots).
2. Dense lanes start simultaneously on their own endpoints: M blocks of 25 items (first 100 ranked items; blocks 5–6 = extension to 150), order IND → DEC → CEN per block.
3. Tier 2 (appended Sun 08:00; the driver re-reads cells.json each poll, append-only): D (100-prefix), N-panel (100-prefix, N=9 DEC cells first), IND_PRIVATE_REVISION/DEC_ONE_ROUND, E (30-prefix in blocks of 10: IND, DEC, CEN, then S_HISTORY), C forecasts after each item's FINAL_HANDOFF seal.
4. Tier 3 (Sun 18:00): M-cross 8B on the 8B lane; A-budget replay (CPU-only + finalization calls); CEN_RLM dev.

Because every module's item order is the same salted rank, partial completion at any cutoff yields nested balanced panels (brief §"Proposed scope").

---

## 4. Evaluation pipeline ops

**BigCodeBench evaluator image** (spec §3.2/§10.6: official tests, fresh sandbox, evaluator-only identity):
```bash
export APPTAINER_CACHEDIR=/orcd/data/tpoggio/001/mabdel03/.cache/apptainer
export APPTAINER_TMPDIR=/orcd/data/tpoggio/001/mabdel03/.cache/apptainer/tmp
apptainer pull /orcd/data/tpoggio/001/mabdel03/containers/bcb-evaluate-v0.2.4.sif docker://bigcodebench/bigcodebench-evaluate:v0.2.4
```
Run on the **login node** (compute nodes are offline for HF; assume the same for Docker Hub) during the implementation window (T0+1h); 9.27 GB of layers → expect 15–40 min; disk: ~9.3 GB layer cache + ~10 GB tmp + ~4–6 GB SIF → reserve 30 GB on data. Verify inside: `apptainer exec --containall $SIF python3 -c "import sys,numpy,pandas,matplotlib;print(sys.version)"` (image is py3.10 with the 73 pins in `scratchpad/bcb_req_eval.txt`), and once on a compute node (user namespaces).

**Per-candidate unittest as CPU array tasks** (`study/eval_bcb.py`, lane `eval`, `mit_preemptable`, 96 tasks × 1 CPU × 8G × 6 h, resumable per shard of 200 candidates): one `apptainer exec --containall --cleanenv --no-home --network none --bind $SHARD_DIR:/work --pwd /work $SIF python3 /work/eval_shard.py` per shard; inside, for each sealed candidate write `solution.py` (= `final_answer`) + the item's hidden `test` into a fresh tmpdir and run `python3 -m unittest` in a fresh subprocess with `timeout 120`, `RLIMIT_AS 8 GiB`, `RLIMIT_STACK 10 MiB`, `TMPDIR` per candidate, `MPLBACKEND=Agg` (mirrors `bigcodebench.eval.untrusted_check` statuses pass/fail/timeout; flat 120 s instead of the gt-time-limit factor — document). Output one JSONL per shard under `<run_root>/eval/bcb/shard_<k>.jsonl` (candidate_id, status, stdout/err tails, wall). Throughput: tier 1 ≈ 200 BCB items × ~265 candidates = 53k runs × ~5 s (pandas/matplotlib imports dominate) = 74 CPU-h → **~50 min on 96 tasks**; tier 2 adds ~30k. Labels join only after the item's selections are sealed (`<run_root>/seals/<block>.json`, §3.3/§10.3).

**HLE judge** (`study/judge_hle.py`, 32B lane, `chat_template_kwargs={"enable_thinking": false}`, `max_tokens 1024`, temperature 0): official cais/hle judge prompt (question, response = `final_answer`, correct_answer) → structured fields `extracted_final_answer / reasoning / correct: yes|no / confidence`; anything not a clean `yes` is scored incorrect and flagged `ambiguous` (§3.3 "never auto-score an ambiguous judge response as correct"). MC items (371/1,600) scored by letter exact match (no judge). Volume ≈ 43k (tier 1) + 15k (tier 2) calls ≈ 3 h fleet time interleaved.

**JUDGE_BEST** (`study/judge_best.py`, same lane): pointwise `prompts/judge_best.txt`, 1,024 cap, thinking off, one call per valid candidate, blind HMAC tie (`metrics_reference.judge_best`), sealed selection before any label join.

**Audit sample** (§3.3): 120 judged candidates, stratified 10 per (module ∈ {F, S_FRESH, S_HISTORY, IND, DEC, CEN} × judge label ∈ {yes, no}), arm labels stripped, exported Mon 06:30 as `<run_root>/eval/hle_audit_sample.jsonl` for the user's blinded review; report disagreement rate and whether differential error could move any claimed contrast by > 2 pp.

---

## 5. Wall-clock timeline (EDT) with go/no-go gates

| When | Milestone | Go / no-go | Monitoring |
|---|---|---|---|
| Sat 16:30 (T0) | Approval; patch C (serve template cache exports, 5 min); init run root; copy HLE shards + download BCB parquet on the login node; start apptainer pull in background | — | `ls $RUN_ROOT/data` |
| 16:45 | Launch 3 × 32B-long + 14B-long on pi_manoli, 3 × 32B-long on pi_tpoggio | pi_tpoggio jobs must show R (they preempt the mit_preemptable GPU job); if PD with `QOSGrpGRES`, another group member holds GPUs → drop to 2 there | `squeue -u $USER -o "%.10i %.34j %.13P %.2t %.9M %.11l %.18R"` |
| 17:15 | Registry check | ≥ 1 `32B-long` entry live (`list_live_servers`), `curl …/v1/models` OK; else read `$RUN_ROOT/logs/serve_32B-long_*.out` (HF offline path, KV OOM) | `ls $RUN_ROOT/servers/32B-long/` |
| 16:45–22:30 | Implementation (parallel packages: data/manifest, client/request store, policies/broker, selectors, dispatch/run_one, judge/eval, pair template, patches A/B/D/F) with fake-server unit tests | 19:00 smoke: one real F00 request round-trip with token ids, reasoning split, strict JSON parse; measured per-seq tok/s | `pytest tests/study -q` |
| 21:30 | Launch loops (keepalive with full SPEC → ou_bcs_normal / ou_bcs_low fill; study driver idle until cells.json exists) | ou_bcs_normal A100 servers R within 20 min (idle nodes); ou_bcs_low is bonus | `tail -n 40 $RUN_ROOT/logs/loop_keepalive_*.out` |
| 22:30–00:30 | Dev pilot on 20 dev items (S_HISTORY on 4): all methods, judge_hle, judge_best, eval_bcb on real image; measure tokens/gen, JSON-valid rate, co-served tok/s, B0 profile (§6.5, outcome-blind) | Valid-JSON ≥ 85 %, no budget overshoot, eval_bcb runs in container, per-endpoint in-flight ≤ 48 with no client timeouts; fix prompts/parsers only on dev | `/metrics`: `vllm:num_requests_running|waiting`, `vllm:generation_tokens_total` deltas |
| Sun 00:30 | **Freeze** (`<run_root>/freeze.json`: sha256 of prompts, configs, code tree, B0, N_main rule) → **tier 1 dispatch** (cells.json blocks 0–7 + dense M blocks 0–3) | — | `squeue -r -h -n asys-study-$RUN-32B \| wc -l` |
| 04:00 | Early-throughput check | ≥ 10 live 32B endpoints (else stay on core numbers), gens/h ≥ 8k on the 32B lane (mid target ≈ 11k), zero `Errno 122`, waiting-queue per endpoint < 100; items with all tier-1 modules complete should be ≥ 80 by 06:00 | `wc -l $RUN_ROOT/requests/*/requests.jsonl`; `find $RUN_ROOT/cells -name 'item_*.json' -newermt '-1 hour' \| wc -l` |
| 08:00 | Tier-2 append (D, N, PR/DEC1, E) + first judge/eval wave for sealed blocks 0–2 | If tier-1 cells < 30 % complete: declare fallback N_main=300 now and cut E to 20 items | driver log `loop_study_driver_*.out` |
| 12:00 | Mid-run gate | tier-1 ≥ 75 % (mid fleet) → keep 400; 55–75 % → N_main=300 fallback, keep tier 2; < 55 % → 300 and drop E/M-cross | same |
| 14:00 | Tier-1 blocks sealed (`seal.py --kind candidates,pools,selections`), judge_hle/judge_best/eval_bcb wave 2 | — | `ls $RUN_ROOT/seals` |
| 18:00 | Tier-3 gate | only if tier 2 ≥ 60 % and ≥ 10 live endpoints: M-cross 8B, A-budget replay, CEN_RLM dev | |
| 22:00 | ou_bcs_normal/low 24 h expiry → keepalive relaunch (10-min tick); cells retry onto other endpoints | check registry prunes/relaunches | keepalive log |
| Mon 02:00 | Tier-2 seal + eval wave 3 | — | |
| **06:00** | Hard stop for generation: `scancel -n asys-study-$RUN-32B` (and dense lanes); final seal; last judge/eval wave (~1.5 h) | — | |
| 07:30–08:45 | Aggregation (`study analyze`: VOTE/JUDGE_BEST accuracy, pass@K via `metrics_reference.pass_at_k`, calibration, 20,000-resample source-cluster bootstrap §9.3; families R/G/M p=1) | — | |
| 09:00 | Report; audit sample handed to the user | | |

**Known failure modes (docs/OPERATIONS.md, docs/RESULTS.md) and how each is prevented here**

| Failure mode | Prevention |
|---|---|
| Queue starvation: two drivers counting each other's `asys-cells` tasks | one driver per run; exact job names `asys-study-<run_id>-<lane>` counted with `squeue -r -h -o %j` |
| Driver deadlock waiting for a whole chunk of headroom while a slow tail (S_HISTORY) drains | `_drive` top-up logic (`topup_min=80`), chunk 100, S_HISTORY dispatched first per block, live `ArrayTaskThrottle` updates |
| Timeout storm (throttle 400 vs 6 servers; 120 s client timeout) | throttle = 8 cells/endpoint with per-cell in-flight caps; new client `timeout=3600`, SDK `max_retries=0`, ≤ 2 technical retries (§10.4) with the same seed, `endpoint_instance_id` recorded per attempt; vLLM `--max-num-seqs 64` |
| Cells clustered on one endpoint (cells sorted by id) | endpoint = `lookup_server(run_root, profile, shard=array_index, live_only=True)` re-resolved every 8 calls; lanes are per profile |
| Serverless cells burning walltime | `wait_for_server(timeout_s=300)` → exit 3 with no `meta.json`; driver submits a lane's chunk only when `len(list_live_servers(profile)) ≥ 1` |
| Driver/keepalive killed with the login session | both are Slurm jobs on pi_manoli (7-day) via `launch_study_loops.py` |
| Loop chain duplicated after manual relaunch (1→3→7 loops) | never `sbatch` the loop scripts by hand and never `--requeue`; relaunch only after `scancel -n asys-loop-keepalive -n asys-loop-study-driver` and an empty `squeue -n …` |
| Scratch / HOME EDQUOT (results, HF `.lock` files, vLLM compile cache) | everything on data: `ASYS_RESULTS_ROOT`, `HF_HOME`, `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `APPTAINER_CACHEDIR/TMPDIR`, `PIP_CACHE_DIR`; tell: `grep -l "Errno 122\|Disk quota" $RUN_ROOT/logs/*.out`, `quota -s`, `df -i /orcd/data/tpoggio/001` |
| 32B KV-cache OOM at TP1 | only the `32B-long` (TP2) profile is ever launched |
| pi_tpoggio `QOSGrpGRES` pending | never more than 3 TP2 there; dense models never there |
| Servers landing on the wrong node (ou_bcs_low placing on node3904) | pi_* launched first; keepalive then fills the rest |
| ou_bcs 1-day expiry / ou_bcs_low preemption mid-call | keepalive relaunch; client retries the same request on another live endpoint; per-call journaling means at most one call is redone |
| Login-node scratchpad invisible to compute nodes | data copied to `<run_root>/data/` at T0 |
| Association MaxSubmit=500 | chunk 100, `--qos-limit 460`, servers ≈ 25 |

---

## 6. Storage and inode estimate (run root on data)

Request record (full prompt text ≤ 16k tokens for DEC revisions, reasoning + content ≈ 14k chars, input ids ≈ 12 KB, output ids ≈ 21 KB, sampled-token logprobs ≈ 35 KB, metadata ≈ 2 KB) ≈ 95–160 KB, mean ~110 KB, stored as per-cell append JSONL (`requests/<cell_id>/requests.jsonl`, `io.append_jsonl`) plus a `request_id → (cell, offset)` index per lane.

| Class | Count | Size |
|---|---|---|
| Solver generations (32B ≈ 200k, dense ≈ 67k, M-cross ≈ 18k) | ~285k | ~31 GB (≤ 45 GB at 4,500 tok/gen) |
| Judge / selector calls | ~215k × 12 KB | ~2.6 GB |
| Item results (`items/<source_id>.json` incl. packets, selections, ledgers) | ~12k × 60 KB | ~0.7 GB |
| eval_bcb outputs (per-shard JSONL) | ~85k runs × 2 KB | ~0.2 GB |
| Cell/driver/server logs | ~1,500 + 25 | ~2 GB |
| Apptainer SIF + layer cache + tmp | | ~15–25 GB (cache deletable) |
| **Total** | | **≈ 52–80 GB** of 15 T free |

Inodes ≈ 1,300 cell dirs × 4 + 12k item files + 1.5k logs + 700 eval shards + ~200 seals/manifests ≈ **25k** on data (1 % of 31 B used). HOME receives nothing (verify `quota -s` stays at 852k files).

---

## Critical Files for Implementation
- /orcd/data/tpoggio/001/mabdel03/agents_scaling/slurm/keepalive.py (legacy `--spec` fleet loop: `parse_spec`, `_serve_jobs_in_flight`, `tick`; patch B)
- /orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/serving/launch_server.py (`render_sbatch`/`submit`, `_port_for`, hardcoded cpus/mem at lines 518-519; patch A; model for the new `launch_pair.py`)
- /orcd/data/tpoggio/001/mabdel03/agents_scaling/slurm/serve_qwen.sbatch.tmpl (patch C; base for `serve_qwen_pair.sbatch.tmpl`)
- /orcd/data/tpoggio/001/mabdel03/agents_scaling/slurm/launch_chunked.py (`_drive` gates/top-up to copy into `src/agents_scaling/study/dispatch.py`) and /orcd/data/tpoggio/001/mabdel03/agents_scaling/slurm/run_cell_array.sbatch.tmpl (patch D)
- /orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/serving/registry.py (`list_live_servers`, `lookup_server(run_root, profile, shard, live_only=True)`, `wait_for_server`) and /orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/experiment/io.py (`atomic_write_text`, `append_jsonl`, `run_dir`)

## Disagreements
- pi_tpoggio is not '7 usable': its QOS carries a group cap GrpTRES=cpu=144,gres/gpu:a100=6,mem=750G (verified with sacctmgr). Only 3 x 32B-long TP2 servers can run there and the brief's '14B-long:1:pi_tpoggio' entry would pend forever with QOSGrpGRES. Move the 14B endpoint to pi_manoli/ou_bcs_normal.
- The brief omits ou_bcs_low, which the user can submit to (group orcd_rg_par_ou_bcs_low): tier 10, preemptable (REQUEUE), 1-day, per-user cap 32 A100 + 32 H100, and four 8xA100 nodes are idle right now. It is the largest opportunistic pool and should carry up to 6 extra 32B-long replicas; the plan's 'mid' fleet assumes 4 of them.
- The brief's cell-worker partitions (mit_preemptable / mit_normal) are worse than pi_manoli CPU slots: mit_normal caps at 96 CPUs per user, mit_preemptable preempts; pi_manoli has 128 spare CPUs after servers, no caps, no preemption, 7-day walltime. This requires lowering the legacy per-server host-memory reservation (240 GB per TP2 server would consume 960 GB of node3904's 1,031 GB).
- The brief's tier-1 estimate (~14 h on 8 servers) omits judge/selector work: ~154k JUDGE_BEST + HLE-judge calls with ~385M prefill tokens add ~3 h of fleet time; tier 1 is ~19 h on the 8-server core and ~13 h only with the opportunistic replicas. N_main=400 remains recommended but the 300-prefix fallback must be declared at freeze and decided at the Sun 08:00/12:00 gates.
- The brief's dense-panel estimate (6-12 h on 1-2 GPUs each, 150 items) is optimistic: 22.5k gens per model at 3,500 tok/gen is 79M tokens; 8B and 4B need ~12 h even with two endpoints each (one co-served). Commit M at 100 items (nested prefix) and extend to 150 only if the lanes finish early.
- The brief says keepalive spec entries can mix H100 and A100 on ou_bcs_normal; keepalive counts in-flight serve jobs per (profile, partition) ignoring GPU type, so two 32B-long entries on ou_bcs_normal would fight over one count. Either patch _serve_jobs_in_flight to scope by gres type or launch H100 servers manually and compensate the A100 target.
- The brief's compute model treats judge calls as negligible and does not schedule the apptainer pull, the BCB parquet download, or copying the HLE shards out of the login-node /tmp scratchpad; all three must happen on the login node during the implementation window.
- The brief's 'reuse verbatim' serve template writes vLLM/inductor caches to $HOME (852k/1000k files); a one-line cache-export patch must precede the first server launch.

## Open questions
- Approve pi_manoli as the primary partition for CPU-only cell workers (up to ~128 x 1 CPU/4 GB tasks on node3904 alongside the servers) and for the two control-loop jobs? docs/OPERATIONS.md's old rule was 'never pi_manoli'; the brief now allows it.
- Approve using ou_bcs_low (preemptable, 1-day, cap 32 A100) for up to 6 opportunistic 32B-long replicas, accepting occasional server preemption handled by keepalive + client retries?
- Approve lowering the per-server host memory reservation to 64 GB per GPU (128 GB per TP2 server) via a small env-driven override in launch_server.py, which is what makes cell workers fit on pi_manoli?
- Commit the dense checkpoint panel (M) at 100 items with 150 as a nested extension, rather than 150 from the start?
- Repeated-episodes module E: keep all four methods including S_HISTORY (the 64-call sequential chain makes E ~32k gens, ~27% of tier 1) or restrict E to IND/DEC/CEN_FLAT?
- Launch the pi_manoli/pi_tpoggio servers at T0+15 min (before the new package exists) so they are warm for the 19:00 smoke test and 22:30 pilot? They would idle ~5 h on 7-day partitions at no scheduling cost.
- Is the HLE correctness judge allowed to be the same Qwen3-32B checkpoint (thinking off) as the solver, as the brief states, given spec 3.3 asks for a frozen isolated judge? (No other judge-capable model is cached.)
- Is anyone else in the tpoggio group expected to use pi_tpoggio GPUs this weekend? The 6-GPU cap is shared across the group; if so the core drops from 8 to 6-7 TP2 servers and the N_main=300 fallback becomes likely.

## Risks
- Opportunistic capacity does not materialize (ou_bcs_low nodes get taken, H100 never scheduled, ou_bcs_normal pends behind the 23 pending jobs) leaving only the 8-server core: tier 1 takes ~19 h and tier 2/3 are squeezed. → Dispatch order guarantees balanced hash-prefix completion; the freeze manifest declares the N_main=300 fallback and the Sun 08:00/12:00 gates; keepalive keeps trying every 10 min; E and tier 3 are the first modules to drop.
- Real tokens/gen is 4,500+ instead of 3,500 (HLE thinking hits the 8,192 cap often), inflating every estimate by ~30%. → Measure in the 22:30 pilot and recompute the table before freeze; if >4,200, freeze N_main=300 with 400 as extension instead of the reverse.
- Co-served 4B+8B pair fails at startup (vLLM free-memory assertion) or halves throughput more than assumed. → Start the larger model first, use --kv-cache-memory-bytes if utilization fractions fail, validate in the pilot via /metrics; fallback is 8B-long alone on that GPU with 4B served only on ou_bcs_normal (or one more ou_bcs_low GPU).
- vLLM compile/HF caches or per-request files land in $HOME and hit the 1,000k-file quota, killing servers or cells mid-run. → Template patch C before the first launch; all cache env vars on data; monitor quota -s hourly; request records stored as per-cell JSONL rather than one file per request.
- Cell concurrency overwhelms endpoints (waiting queue grows, latencies exceed the client timeout) or, conversely, endpoints idle because the driver throttle lags the live fleet. → Throttle = 8 cells per live endpoint with per-cell in-flight caps (<=48 seqs/endpoint), client timeout 3,600 s, driver updates ArrayTaskThrottle live from list_live_servers, pilot verifies num_requests_waiting stays < 100.
- Judge/eval waves start too late (labels needed for pass@K, calibration, and all accuracy tables) and Monday aggregation lacks correctness for late blocks. → Seal and judge per 50-item block as soon as tier-1 pools complete (first wave Sun 08:00), reserve the last 1.5 h before 07:30 for the final wave, and report the largest fully-evaluated prefix.
- Our pi_tpoggio submissions must preempt another user's 3-GPU mit_preemptable job; if the remaining GPUs are held by non-preemptable ou_bcs_low/normal jobs or another group member uses pi_tpoggio, fewer than 3 servers start. → Check squeue -w node3807 at T0; if only 2 servers fit, add one more ou_bcs_low replica and record the smaller core in the plan; QOSGrpGRES pending is harmless.
- BigCodeBench evaluator image pull fails or is slow on the login node (9.27 GB), or the image's Python path differs from expectation. → Start the pull at T0 in the background with cache/tmp on data; verify with apptainer exec on both login and compute nodes during the pilot; fallback is a py3.10 conda env with the 73 pins from bcb_req_eval.txt (slower to build, ~30 min).
- Manual relaunch of loops or a second driver duplicates the self-resubmit chain or double-launches servers (observed 1->3->7 loop duplication). → Only launch_study_loops.py launches loops; before any relaunch scancel both loop names and confirm squeue is empty; never pass --requeue; keepalive counts PD jobs so duplicates are not launched.
- Association MaxSubmit=500 rejects array chunks once servers (~25) + loops + eval arrays accumulate. → chunk-size 100, --qos-limit 460, driver's absolute gate waits for real headroom; eval arrays limited to 96 tasks.
## Rolling evaluation waves (added Sat 23:45 EDT)

Seals are keyed by the generate manifest's sha256 and are append-only, so a manifest has ONE
seal and re-sealing adds newly completed items.  Eval-kind cell ids carried only `x<seal8>`
so rolling waves under one seal would have collided on the cell directory; `cells.py` now
accepts `--wave <tag>` (cell id `…x<seal8>.w<tag>.s000`) and `--items-file` (restricts the
select/eval tiers to listed sealed items).  `agents_scaling.study.waves` plans a wave = items
whose EVERY generate cell of the manifest is complete and sealed, minus items of earlier waves
(`<run_root>/waves/<seal8>_w<tag>.json`).  Judging an item exactly once, after all its pools
exist, keeps aggregate's seal-order guard satisfied for every selection record of the item.

Per wave (scripts under `slurm/`):
1. `slurm/study_wave_start.sh cells_1_32B.json 1 <wave> [min_items] [judge_throttle]` —
   seal pools → plan wave → `cells --tier 1-select … --wave` → dispatch JUDGE_BEST on the
   judge fleet (`--server-run-id study_v4_judge`, whole-manifest chunk).
2. when every JUDGE_BEST cell of the wave has `meta.json`:
   `slurm/study_wave_finish.sh cells_1_32B.json 1 <wave>` — seal selections (VOTE for all
   pools; JUDGE_BEST where scored) → `cells --tier 1-eval --lane 32B|eval … --wave` →
   dispatch JUDGE_HLE (judge fleet) and EVAL_BCB (eval lane, 2 CPU / 8 G).
3. `aggregate` any time (lenient mode counts `join_refused`; it must be 0 for waved items).

The loop driver (`slurm/study_loop_driver.sbatch.study_v4`) now iterates its per-lane drivers
every 15 min inside one job (the drivers exit as soon as every chunk is dispatched, which made
the afterany chain spin) and chains tier 2 (`cells_2_32B.json`, throttle 30) behind tier 1 on
the 32B lane so the fleet never idles.
