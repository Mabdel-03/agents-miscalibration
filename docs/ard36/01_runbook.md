# ARD36 runbook — the exact sequence, gate by gate

Companion to `00_contract.md`. Every command is written out because the 36-hour clock is
running and study_v4 lost 30–90 minutes to each remembered-wrong invocation.

```bash
REPO=/orcd/data/tpoggio/001/mabdel03/agents_scaling
PY=/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python
RUN=ard36
RESULTS=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results
```

## G1 — implementation gate (target ~h8)

Passes when WP0+WP1+WP3 and the manifest builder are in and the offline dry run is green.

```bash
cd $REPO && PYTHONPATH=src $PY -m pytest tests/ard tests/study -q      # both suites
PYTHONPATH=src $PY scripts/ard36_dry_run.py                            # every cell kind, 4 items, no GPU
PYTHONPATH=src $PY scripts/ard36_preflight.py --json-out $RESULTS/$RUN/preflight_h0.json
```

Then build and freeze the manifests, and only then launch the fleet:

```bash
PYTHONPATH=src $PY -m agents_scaling.ard.items_build --out $RESULTS/$RUN/data   # evaluator-only
PYTHONPATH=src $PY -m agents_scaling.ard.cells --run-id $RUN --module roots --lane 8B
PYTHONPATH=src $PY -m agents_scaling.ard.cells --run-id $RUN --module roots --lane 32B
# FROZEN.yaml is write-once; a second freeze means a new run root, never a silent edit
PYTHONPATH=src $PY -m agents_scaling.ard.freeze --run-id $RUN
```

Fleet (legacy path; the schema-5 contract path rejects non-a100 / non-1-day fleets):

```bash
cd $REPO && python slurm/study_loops.py --allow-legacy-driver --keepalive-only \
  --run-id $RUN --interval 180 \
  --spec "ard36-32B:2:pi_manoli:2-00:00:00,ard36-8B:3:pi_manoli:2-00:00:00,\
ard36-32B:1:pi_tpoggio:2-00:00:00,ard36-8B:2:pi_tpoggio:2-00:00:00,\
ard36-32B:1:ou_bcs_normal:1-00:00:00,ard36-8B:3:ou_bcs_normal:20:00:00,\
ard36-8B:2:ou_bcs_normal:1-00:00:00,ard36-8B:6:ou_bcs_low:1-00:00:00"
```

Submit the PI probe jobs first if the PI partitions are contended, and record time-to-allocation.
If the PI core has not registered by fleet-launch + 60 min, freeze the tier on whatever is
registered and apply §16.4 steps 1–2; keep the PI requests queued.

Watch registration, never assume it:

```bash
squeue -u $USER -n asys-srv-$RUN-* -o "%i %P %T %b %N %M"
PYTHONPATH=src $PY -c "from agents_scaling.serving import registry as r; \
print([(e.serving_profile, e.host, e.port) for e in r.list_live_servers('$RESULTS/$RUN')])"
```

## Launch gate (~h10) — spec §16.3 fields, all measured, none assumed

```bash
PYTHONPATH=src $PY -m agents_scaling.ard.gates launch --run-id $RUN --json-out $RESULTS/$RUN/gate_h2.json
```

Must report: registered health-checked replicas per profile×partition; BF16 confirmed per replica;
root parse validity ≥95%; effective tok/s **stratified by call type** with mean and p95 latency and
lengths; peak GPU memory and KV-cache utilisation; HTTP error + retry + ban rate (fail above 1%);
MATH grader items/s with its timeout and ambiguous rates over a 500-answer pilot; replay
sequences/s at p95 context. Then the projection:

```bash
$PY <(unzip -p expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Bundle.zip \
        Agents_Round_Dynamics_36h/budget_check.py) \
   --config <reduced run_plan.yaml copy> \
   --small-tps-per-replica <measured> --large-tps-per-replica <measured> --replay-hours <measured>
```

Freeze the tier on the result. Reduced core is the default: **47,424 8B + 8,256 32B**.

## Dispatch priority (one global order; the lane loops enforce it)

**P0** 8B ROOTS train+val → **P1** PEER + PRIVATE train/val → **P2** FORK t=1,2 + grading of
P0–P2 + features + fit/validate → **freeze `controller/<sha>`** → **P3** everything else,
longest chains first. Earliest safe freeze: PEER/PRIVATE train/val and FORK coverage 100% of
*decisions* and their grading closed.

```bash
python slurm/ard36_launch_chunked.py --allow-legacy-admission --run-id $RUN \
  --cells-file $RESULTS/$RUN/cells_roots_8B.json --lane 8B \
  --chunk-size 100 --throttle 48 --submit-cap 380 --qos-limit 420 \
  --cell-partition mit_preemptable --cell-time 1-00:00:00 --cell-mem 4G --cpus 1
python slurm/ard36_topup_lane.py --run-id $RUN --cells-file <manifest> --lane 8B --throttle 40
```

Deadline-bearing lanes (PEER/FORK train-val, GATE, challenge controller arms, capture) must not
use ou_bcs_low endpoints — those replicas lived 12–17 minutes under preemption in study_v4.

## Coverage gates

```bash
PYTHONPATH=src $PY -m agents_scaling.ard.gates coverage --run-id $RUN --hour 18   # train/val ≥99%, test ≥40%
PYTHONPATH=src $PY -m agents_scaling.ard.gates coverage --run-id $RUN --hour 24   # test ≥90%, else §16.4 steps 4–5
```

## Close-out (h30–36)

```bash
PYTHONPATH=src $PY -m agents_scaling.ard.compact  --run-id $RUN     # 7 spec + 12 descriptive tables
PYTHONPATH=src $PY -m agents_scaling.ard.analysis --run-id $RUN --write-predictions-first
PYTHONPATH=src $PY -m agents_scaling.ard.analysis --run-id $RUN --join-labels --bootstrap 20000
PYTHONPATH=src $PY -m agents_scaling.ard.report   --run-id $RUN     # results_summary.md + hardware_report.md
```

Teardown, in this order:

```bash
scancel -n asys-loop-keepalive-$RUN      # the keepalive FIRST, or it relaunches what you cancel
scancel -n asys-srv-$RUN-*
PYTHONPATH=src $PY -c "from agents_scaling.serving import registry as r; \
print(r.list_live_servers('$RESULTS/$RUN'))"    # must be empty
```

## Traps carried from study_v4 (each cost 30–90 minutes)

- Never hand-`sbatch` a loop script; the `afterany` self-resubmit is the only durability mechanism
  and combining it with `--requeue` duplicated the chain.
- Never run two waves of one manifest concurrently — the second seal postdates the first wave's
  evaluation start and the join refuses those rows.
- `submit()` refuses to overwrite an existing server render; move it aside or let keepalive retire it.
- A preempted server job is gone, not requeued (`--no-requeue`); recovery is keepalive interval plus
  model load. Cached endpoint lists go stale, hence the 120 s ban on a just-failed endpoint.
- `pkill -f <pattern>` from a shell whose own command line contains the pattern kills the shell.
- Compile caches must stay on node-local `/tmp`; `$HOME` is at 92% of its file quota and TP>1 ranks
  racing on one inductor dir gave `Stale file handle` on NFS.
- Write nothing to `/orcd/scratch` — it is inode-exhausted at 100%.
