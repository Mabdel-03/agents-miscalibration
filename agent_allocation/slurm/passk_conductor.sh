#!/bin/bash
# passk_cont_v1 conductor: Gate 0 -> Gate 1 (canary + smoke) -> main per shard (+ grade --follow per shard, Gate 2
# checkpoints, Gate 3 stop rules) -> analysis chain (extract ... seal) -> Gate 4 -> release -> score -> report.
# Run it through slurm/passk_conductor.sbatch (pi_lhtsai, 2 days, --mail-type=FAIL,END). Every step that spends or
# grades is its own sbatch job of slurm/passk_cont.sbatch; the conductor evaluates the gates between them, writes
# conductor.state + STATUS.md at every transition, and stops (exit != 0 -> Slurm mails the owner) on any failure.
#
# Resume: START_AT=gate0|gate1|main|analysis|gate4 (default gate0). A stopped chain is safe: held-out labels stay
# fenced until sealed/SEAL_predictions.json verifies.
# Exit codes of the steps: 0 advance, 3 stop (gate failed / stopped early), 2 blocked, 1 crashed.
set -uo pipefail
CODE=${CODE:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_cont/agent_allocation}
STUDY=${STUDY:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1}
CFG=${CFG:-configs/passk_cont_v1.json}
START_AT=${START_AT:-gate0}
POLL=${POLL:-300}
SHARDS=(mancer parasail)
MAIN_PARTITIONS=${MAIN_PARTITIONS:-c48_extra+e1_fill+t200_diag+v48_bank+t200_bank}
ANALYSIS_PRE=(extract pairs e1-lock e2-fit e5-fit seal)
ANALYSIS_POST=(release score report)
RUN="$STUDY/artifacts"
STATE="$STUDY/conductor.state"
STATUS="$STUDY/STATUS.md"
LOCK="$STUDY/conductor.lock"
SB="$CODE/slurm/passk_cont.sbatch"
cd "$CODE" || exit 2
source slurm/env.sh
A=.venv/bin/allocation
PY=.venv/bin/python
mkdir -p "$STUDY/logs" "$STUDY/gates"

# ---- single conductor: flock + refuse a duplicate Slurm job ---------------------------------------------------------
exec 9>"$LOCK"
flock -n 9 || { echo "another conductor holds $LOCK"; exit 1; }
others=$(squeue -h -u "$USER" -n passk-conductor -o %i 2>/dev/null | grep -v "^${SLURM_JOB_ID:-none}$" || true)
[ -n "$others" ] && { echo "refusing: another passk-conductor job is queued/running: $others"; exit 1; }

say(){ echo "[conductor $(date -Is)] $*"; }
note(){  # $1 state token, $2 free-text detail
  echo "$1" > "$STATE"
  {
    echo "# passk_cont_v1 STATUS"
    echo
    echo "- updated: $(date -Is)"
    echo "- conductor job: ${SLURM_JOB_ID:-interactive} on $(hostname)"
    echo "- state: \`$1\`"
    echo "- detail: ${2:-}"
    echo "- code: $CODE @ $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null)"
    echo "- resume: \`sbatch --export=ALL,START_AT=<gate0|gate1|main|analysis|gate4> slurm/passk_conductor.sbatch\`"
    echo
    echo "## gates"
    for f in "$STUDY"/gates/*.json; do
      [ -f "$f" ] || continue
      printf -- "- %s: %s\n" "$(basename "$f")" "$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('passed'))" "$f" 2>/dev/null)"
    done
    echo
    echo "## jobs"
    squeue -h -u "$USER" -o "%i %j %T %M" 2>/dev/null | grep passk || echo "(none)"
  } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"
  say "state=$1 ${2:-}"
}
stop(){ note "stopped:$1" "$2"; say "STOP: $2"; exit "${3:-3}"; }
drain_all(){ local s; for s in "${SHARDS[@]}"; do mkdir -p "$RUN/shards/$s/control"; touch "$RUN/shards/$s/control/DRAIN"; done; }
stop_main(){ drain_all; stop "$@"; }   # during generation a stop drains BOTH shards (nobody would watch the other)

submit(){  # $1 verb, $2 shard (or ""), extra NAME=VALUE pairs (no commas in values) -> echoes job id
  local verb=$1 shard=$2; shift 2
  local exports="ALL,VERB=$verb,CFG=$CFG,STUDY=$STUDY,CODE=$CODE"
  [ -n "$shard" ] && exports="$exports,SHARD=$shard"
  for kv in "$@"; do exports="$exports,$kv"; done
  sbatch --parsable --job-name="passk-$verb${shard:+-$shard}" --export="$exports" "$SB" 2>>"$STUDY/logs/conductor_sbatch.err" | cut -d';' -f1
}
job_state(){ sacct -j "$1" -n -X -o State 2>/dev/null | head -1 | tr -d ' '; }
job_rc(){ sacct -j "$1" -n -X -o ExitCode 2>/dev/null | head -1 | tr -d ' ' | cut -d: -f1; }
job_done(){ case "$(job_state "$1")" in COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED) return 0;; *) return 1;; esac; }
wait_jobs(){  # wait until every job id given has ended
  local j
  for j in "$@"; do
    while ! job_done "$j"; do sleep 60; done
    say "job $j ended $(job_state "$j") rc=$(job_rc "$j")"
  done
}
checkpoint(){  # $1 stage, $2 shard, extra args -> rc
  local stage=$1 shard=$2; shift 2
  $PY -m agent_allocation.passk.checkpoints "$stage" --config "$CFG" --study-root "$STUDY" --shard "$shard" "$@"
}
gate0_endpoint_class(){  # transient | semantic | passed | absent (the forced endpoint screen of a shard's canary)
  $PY - "$RUN/shards/$1/gates/gate0_endpoint.json" <<'PY'
import json, re, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("absent"); raise SystemExit
if d.get("passed"):
    print("passed"); raise SystemExit
detail = str(d.get("detail", ""))
failing = re.findall(r"GateCriterion\(name='([^']+)', observed='(.*?)', threshold=.*?passed=False", detail)
semantic = [n for n, obs in failing if '"status": "ok"' in obs]
print("semantic" if semantic or not failing else "transient")
PY
}

# ====================================================================================================================
# Gate 0 ($0): suite + mocked e2e, node lists, parity, E0 subset, slices, forecast, /key re-read
# ====================================================================================================================
if [ "$START_AT" = gate0 ]; then
  note gate0 "offline Gate 0 + /key"
  ok=""
  for attempt in 1 2 3; do
    $PY -m agent_allocation.passk.gates gate0 --config "$CFG" --study-root "$STUDY" --check-key --run-tests --run-e0
    rc=$?
    [ $rc -eq 0 ] && { ok=1; break; }
    [ $rc -eq 3 ] && stop gate0 "Gate 0 failed (gates/gate0.json)"
    say "Gate 0 crashed rc=$rc (transient?); retry $attempt/3 in 10 min"; sleep 600
  done
  [ -n "$ok" ] || stop gate0 "Gate 0 did not complete after 3 attempts" 1
  START_AT=gate1
fi

# ====================================================================================================================
# Gate 1: canary per shard (both submitted together), canary --evaluate-only for both, then the smoke
# ====================================================================================================================
if [ "$START_AT" = gate1 ]; then
  note gate1-canary "canary on both shards"
  declare -A CJ
  for s in "${SHARDS[@]}"; do CJ[$s]=$(submit canary "$s"); [ -n "${CJ[$s]}" ] || stop gate1 "canary $s not submitted" 1; done
  wait_jobs "${CJ[@]}"
  for s in "${SHARDS[@]}"; do
    rc=$(job_rc "${CJ[$s]}")
    tries=1
    while [ "$rc" != 0 ] && [ "$rc" != 3 ]; do
      cls=$(gate0_endpoint_class "$s")
      [ "$cls" = semantic ] && stop gate1 "forced endpoint screen failed semantically on $s (banana/think-close/exactness)"
      [ $tries -ge 3 ] && stop gate1 "canary $s failed $tries times (last rc=$rc, class=$cls)"
      say "canary $s rc=$rc class=$cls: transient, retry in 10 min"; sleep 600
      j=$(submit canary "$s"); wait_jobs "$j"; rc=$(job_rc "$j"); tries=$((tries + 1))
    done
  done
  note gate1-canary-eval "canary --evaluate-only on both shards (union key reconciliation)"
  for s in "${SHARDS[@]}"; do
    MAX=$($PY -c "import sys; from agent_allocation.config import load_config; from agent_allocation.shards import shard_config; print(shard_config(load_config(sys.argv[1]), sys.argv[2])['budget']['credit_ceiling_usd'])" "$CFG" "$s")
    $A canary --config "$CFG" --run-dir "$RUN" --shard "$s" --allow-paid --max-usd "$MAX" --evaluate-only
    rc=$?; [ $rc -eq 0 ] || stop gate1 "canary evaluate-only $s rc=$rc (gates/gate1_canary.json of the shard)"
  done
  note gate1-smoke "passk-run smoke on both shards"
  declare -A SJ
  for s in "${SHARDS[@]}"; do SJ[$s]=$(submit smoke "$s"); done
  wait_jobs "${SJ[@]}"
  for s in "${SHARDS[@]}"; do [ "$(job_rc "${SJ[$s]}")" = 0 ] || stop gate1 "smoke $s rc=$(job_rc "${SJ[$s]}")"; done
  declare -A GJ
  for s in "${SHARDS[@]}"; do GJ[$s]=$(submit grade "$s"); done
  wait_jobs "${GJ[@]}"
  for s in "${SHARDS[@]}"; do
    checkpoint gate1-smoke "$s" --fetch-metadata; rc=$?
    [ $rc -eq 0 ] || stop gate1 "smoke gate $s rc=$rc (gates/gate1_smoke_$s.json)"
  done
  START_AT=main
fi

# ====================================================================================================================
# Main: passk-run per shard + grade --follow per shard; Gate 2 / Gate 3 every POLL; analysis chain once stage 1 is done
# ====================================================================================================================
stage1_done(){  # every stage-1 node of both shards parsed, and every non-fenced candidate graded
  $PY - "$STUDY" <<'PY'
import sys
from pathlib import Path
from agent_allocation.passk.checkpoints import node_rows, _shard_paths
study = Path(sys.argv[1])
for shard in ("mancer", "parasail"):
    paths = _shard_paths(study, shard)
    rows = node_rows(paths, ["c48_extra", "e1_fill", "t200_diag"])
    if not rows or not all(r["parsed"] for r in rows):
        print("no"); raise SystemExit
    if any(r.get("has_candidate") and r["partition"] != "t200_diag" and not paths.grade(r["node_id"]).exists()
           for r in rows):
        print("no"); raise SystemExit
print("yes")
PY
}
run_analysis(){  # $@ stages; each its own job, sequential
  local st j
  for st in "$@"; do
    note "analysis:$st" "passk-analyze --stage $st"
    j=$(submit analyze "" "STAGE=$st"); [ -n "$j" ] || stop "analysis:$st" "not submitted" 1
    wait_jobs "$j"
    [ "$(job_rc "$j")" = 0 ] || stop "analysis:$st" "passk-analyze --stage $st rc=$(job_rc "$j")"
  done
}

if [ "$START_AT" = main ]; then
  note main "passk-run $MAIN_PARTITIONS on both shards"
  declare -A RJ FJ RESUB
  for s in "${SHARDS[@]}"; do
    RJ[$s]=$(submit run "$s" "PARTITIONS=$MAIN_PARTITIONS"); RESUB[$s]=0
    FJ[$s]=$(submit grade-follow "$s")
    [ -n "${RJ[$s]}" ] && [ -n "${FJ[$s]}" ] || stop_main main "run/grade-follow $s not submitted" 1
  done
  analysis_started=""
  polls=0
  while true; do
    sleep "$POLL"
    polls=$((polls + 1))
    for s in "${SHARDS[@]}"; do
      if [ $((polls * POLL)) -ge 1800 ] && [ $(( (polls * POLL) % 1800 )) -lt "$POLL" ]; then   # every ~30 min
        checkpoint sample "$s" >/dev/null; rc=$?
        [ $rc -eq 3 ] && stop_main main "permaslug/provider sample failed on $s (gates/gate3_sample_$s.json)"
      fi
      checkpoint gate3 "$s" >/dev/null; rc=$?
      [ $rc -eq 3 ] && stop_main main "Gate 3 stop rule on $s (gates/gate3_$s.json; control/DRAIN touched)"
      checkpoint gate2 "$s" >/dev/null; rc=$?
      if [ $rc -eq 3 ]; then
        touch "$RUN/shards/$s/control/DRAIN"
        stop_main main "Gate 2 checkpoint failed on $s (gates/gate2_$s.json; control/DRAIN touched)"
      fi
    done
    if [ -z "$analysis_started" ] && [ "$(stage1_done)" = yes ]; then
      analysis_started=1
      note main "stage 1 complete on both shards: analysis chain (pre-seal) in the background of generation"
      ( run_analysis "${ANALYSIS_PRE[@]}" ) &
      ANALYSIS_PID=$!
    fi
    all_done=1
    for s in "${SHARDS[@]}"; do
      j=${RJ[$s]}
      job_done "$j" || { all_done=""; continue; }
      rc=$(job_rc "$j")
      [ "$rc" = 0 ] && continue
      # stopped early: resubmit only when no gate stopped the shard (a Slurm time-limit drain), at most 3 times
      if [ -f "$STUDY/gates/gate3_$s.json" ] && ! $PY -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['passed'] else 1)" "$STUDY/gates/gate3_$s.json"; then
        stop_main main "run $s stopped by Gate 3"
      fi
      [ "${RESUB[$s]}" -ge 3 ] && stop_main main "run $s stopped early 3 times (rc=$rc)"
      rm -f "$RUN/shards/$s/control/DRAIN"
      RESUB[$s]=$((RESUB[$s] + 1)); RJ[$s]=$(submit run "$s" "PARTITIONS=$MAIN_PARTITIONS"); all_done=""
      say "resubmitted run $s (#${RESUB[$s]}) as ${RJ[$s]}"
    done
    note main "polling; runs: ${RJ[*]}; grade-follow: ${FJ[*]}"
    [ -n "$all_done" ] && break
  done
  if [ -n "${ANALYSIS_PID:-}" ]; then
    wait "$ANALYSIS_PID" || stop analysis "pre-seal analysis chain failed (see STATUS.md)"
  else
    run_analysis "${ANALYSIS_PRE[@]}"
  fi
  START_AT=gate4
fi
[ "$START_AT" = analysis ] && { run_analysis "${ANALYSIS_PRE[@]}"; START_AT=gate4; }

# ====================================================================================================================
# Gate 4 per shard -> release (the grade --follow jobs grade the held-out within one poll of the seal) -> score/report
# ====================================================================================================================
if [ "$START_AT" = gate4 ]; then
  for s in "${SHARDS[@]}"; do
    note gate4 "completion checks on $s"
    checkpoint gate4 "$s"; rc=$?
    [ $rc -eq 0 ] || stop gate4 "Gate 4 failed on $s (gates/gate4_$s.json)"
  done
  run_analysis "${ANALYSIS_POST[@]}"
  for s in "${SHARDS[@]}"; do touch "$RUN/shards/$s/control/STOP"; done      # ends the grade --follow jobs
  note done "chain complete: report built"
  say "CONDUCTOR FINISHED"
fi
exit 0
