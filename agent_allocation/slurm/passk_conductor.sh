#!/bin/bash
# passk_cont_v1 conductor: Gate 0 -> Gate 1 (canary + smoke) -> main per shard (+ grade --follow per shard, Gate 2
# checkpoints, Gate 3 stop rules) -> analysis chain (extract ... seal) -> Gate 4 -> release -> score -> report.
# Run it through slurm/passk_conductor.sbatch (pi_lhtsai, 2 days, --mail-type=FAIL,END). Every step that spends or
# grades is its own sbatch job of slurm/passk_cont.sbatch; the conductor evaluates the gates between them, writes
# conductor.state + STATUS.md at every transition, and stops (exit != 0 -> Slurm mails the owner) on any failure.
#
# Resume: START_AT=gate0|gate1|smoke|main|<any analysis step token>|gate4 (default gate0). Analysis step tokens (the order
# they run in): extract pairs extract2 e1-lock e2-fit e5-fit seal | gate4 | release-v48 pairs-v48 extract-v48
# score-v48 report-v48 release-t200 pairs-t200 extract-t200 score-t200 report-t200 ("analysis" = extract). A stopped
# chain is safe: held-out labels stay fenced until sealed/SEAL_predictions.json verifies.
# Mail: export OWNER_MAIL before sbatch; the conductor sets its own job's MailUser (FAIL,END) from it.
# Exit codes of the steps: 0 advance, 3 stop (gate failed / stopped early), 2 blocked, 1 crashed.
set -uo pipefail
CODE=${CODE:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_cont/agent_allocation}
STUDY=${STUDY:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1}
CFG=${CFG:-configs/passk_cont_v1.json}
START_AT=${START_AT:-gate0}
POLL=${POLL:-300}
SHARDS=(mancer parasail)
MAIN_PARTITIONS=${MAIN_PARTITIONS:-c48_extra+e1_fill+t200_diag+v48_bank+t200_bank}
WAIT_MAX=${WAIT_MAX:-50400}          # bound on any one step job (14 h): a stuck/unknown job stops + mails
SUBMIT_RETRY_S=${SUBMIT_RETRY_S:-60}
LEASE_WAIT=${LEASE_WAIT:-960}        # dispatcher lease_stale_seconds (900) + margin before a resubmission after a kill
# analysis steps: token|stage|FLAGS|RELEASE. The fits run in the bwrap fence (--sandbox, Gate P); the second extract
# rebuilds classes.parquet from the filled pair cache; each held-out split is released as its own job, then its
# pairs are enumerated and classes re-extracted before scoring.
ANALYSIS_PRE=("extract|extract||" "pairs|pairs||" "extract2|extract||" "e1-lock|e1-lock||"
              "e2-fit|e2-fit|--sandbox|" "e5-fit|e5-fit|--sandbox|" "seal|seal||")
ANALYSIS_POST=()
for split in v48 t200; do
  ANALYSIS_POST+=("release-$split|release||$split" "pairs-$split|pairs||" "extract-$split|extract||"
                  "score-$split|score||" "report-$split|report||")
done
FAILED_MARK="$STUDY/analysis.failed"
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
if [ -n "${OWNER_MAIL:-}" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  scontrol update JobId="$SLURM_JOB_ID" MailUser="$OWNER_MAIL" MailType=FAIL,END >/dev/null 2>&1 \
    || echo "[conductor] WARNING: could not set MailUser from OWNER_MAIL"
fi

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
    echo "- mail: ${OWNER_MAIL:-UNSET (export OWNER_MAIL before sbatch)}"
    echo "- resume: \`sbatch --export=ALL,START_AT=<gate0|gate1|main|step token|gate4> slurm/passk_conductor.sbatch\`"
    if [ -f "$FAILED_MARK" ]; then echo "- **ANALYSIS FAILED**: $(cat "$FAILED_MARK")"; fi
    echo
    echo "## gates"
    for f in "$STUDY"/gates/*.json; do
      [ -f "$f" ] || continue
      printf -- "- %s: %s\n" "$(basename "$f")" "$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('passed'))" "$f" 2>/dev/null)"
    done
    echo
    echo "## jobs"
    squeue -h -u "$USER" -o "%i %j %T %M" 2>/dev/null | grep passk || echo "(none)"
  } > "$STATUS.tmp.$BASHPID" && mv "$STATUS.tmp.$BASHPID" "$STATUS"
  say "state=$1 ${2:-}"
}
stop(){ note "stopped:$1" "$2"; say "STOP: $2"; exit "${3:-3}"; }
drain_all(){ local s; for s in "${SHARDS[@]}"; do mkdir -p "$RUN/shards/$s/control"; touch "$RUN/shards/$s/control/DRAIN"; done; }
stop_main(){ drain_all; stop "$@"; }   # during generation a stop drains BOTH shards (nobody would watch the other)

submit(){  # $1 verb, $2 shard (or ""), extra NAME=VALUE pairs (no commas in values) -> echoes job id ("" = failed)
  local verb=$1 shard=$2; shift 2
  local exports="ALL,VERB=$verb,CFG=$CFG,STUDY=$STUDY,CODE=$CODE" out try
  [ -n "$shard" ] && exports="$exports,SHARD=$shard"
  for kv in "$@"; do exports="$exports,$kv"; done
  for try in 1 2 3; do            # an sbatch hiccup (slurmctld timeout, submit limit) is retried, never waited on
    out=$(sbatch --parsable --job-name="passk-$verb${shard:+-$shard}" --export="$exports" "$SB" \
          2>>"$STUDY/logs/conductor_sbatch.err" | cut -d';' -f1)
    [[ "$out" =~ ^[0-9]+$ ]] && { echo "$out"; return 0; }
    [ $try -lt 3 ] && sleep "$SUBMIT_RETRY_S"
  done
  echo ""
}
job_line(){  # "STATE|EXITCODE" of exactly this job id (parsable: no 10-char truncation of OUT_OF_MEMORY etc.)
  [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "INVALID|"; return; }
  sacct -j "$1" -n -X -P -o JobID,State,ExitCode 2>/dev/null | awk -F'|' -v j="$1" '$1==j {print $2"|"$3; exit}'
}
job_state(){ local l; l=$(job_line "$1"); l=${l%%|*}; echo "${l%% *}"; }   # "CANCELLED by 123" -> CANCELLED
job_rc(){  # the script's exit code for COMPLETED/FAILED; 99 for TIMEOUT/CANCELLED/NODE_FAIL/OOM/... (never "0" by accident)
  local l; l=$(job_line "$1")
  case "${l%%|*}" in
    COMPLETED|FAILED) l=${l#*|}; echo "${l%%:*}" ;;
    *) echo 99 ;;
  esac
}
job_done(){  # ended = any state other than the live ones; an id sacct does not know (yet) is not done
  local st; st=$(job_state "$1")
  case "$st" in
    INVALID) return 0 ;;
    ""|PENDING|RUNNING|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|SUSPENDED|CONFIGURING|COMPLETING|RESIZING|SIGNALING|STAGE_OUT)
      return 1 ;;
    *) return 0 ;;
  esac
}
wait_jobs(){  # wait until every job id given has ended; bounded (WAIT_MAX s from the call): returns 1 on timeout
  local j t0; t0=$(date +%s)
  for j in "$@"; do
    [[ "$j" =~ ^[0-9]+$ ]] || { say "wait_jobs: invalid job id '$j'"; return 1; }
    while ! job_done "$j"; do
      [ $(( $(date +%s) - t0 )) -ge "$WAIT_MAX" ] && { say "wait_jobs: $j still $(job_state "$j") after ${WAIT_MAX}s"; return 1; }
      sleep "${WAIT_POLL:-60}"
    done
    say "job $j ended $(job_state "$j") rc=$(job_rc "$j")"
  done
}
clear_sentinels(){  # benign leftovers (a previous stop's DRAIN, the final STOP) must not end a fresh run at once
  local s
  for s in "${SHARDS[@]}"; do rm -f "$RUN/shards/$s/control/DRAIN" "$RUN/shards/$s/control/STOP"; done
}
ensure_grade_follow(){  # $1 shard: a grade --follow job must exist (the graders hold single_grader_lock: harmless dup)
  local s=$1 j
  squeue -h -u "$USER" -n "passk-grade-follow-$s" -o %i 2>/dev/null | grep -q . && return 0
  rm -f "$RUN/shards/$s/control/STOP"
  j=$(submit grade-follow "$s"); [ -n "$j" ] || return 1
  say "grade-follow $s submitted as $j"; echo "$j" > "$STUDY/logs/grade_follow_$s.jobid"
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
  if [ ! -f "$RUN/shards/assignment.json" ]; then
    note shard-plan "materializing the provider shards and adopting the forced_v2 endpoint locks"
    $A shard-plan --config "$CFG" --run-dir "$RUN" --write || stop shard-plan "shard-plan --write failed" 1
  fi
  for s in "${SHARDS[@]}"; do
    $A passk-run --config "$CFG" --run-dir "$RUN" --shard "$s" --partitions smoke --adopt-legacy-lock \
      || stop shard-plan "legacy lock adoption failed on $s" 1
  done
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
  clear_sentinels
  note gate1-canary "canary on both shards"
  declare -A CJ
  for s in "${SHARDS[@]}"; do CJ[$s]=$(submit canary "$s"); [ -n "${CJ[$s]}" ] || stop gate1 "canary $s not submitted" 1; done
  wait_jobs "${CJ[@]}" || stop gate1 "canary jobs did not end within ${WAIT_MAX}s" 1
  for s in "${SHARDS[@]}"; do
    rc=$(job_rc "${CJ[$s]}")
    tries=1
    while [ "$rc" != 0 ] && [ "$rc" != 3 ]; do
      cls=$(gate0_endpoint_class "$s")
      [ "$cls" = semantic ] && stop gate1 "forced endpoint screen failed semantically on $s (banana/think-close/exactness)"
      [ $tries -ge 3 ] && stop gate1 "canary $s failed $tries times (last rc=$rc, class=$cls)"
      say "canary $s rc=$rc class=$cls: transient, retry in 10 min"; sleep 600
      j=$(submit canary "$s"); [ -n "$j" ] || stop gate1 "canary $s resubmission failed" 1
      wait_jobs "$j" || stop gate1 "canary $s did not end within ${WAIT_MAX}s" 1
      rc=$(job_rc "$j"); tries=$((tries + 1))
    done
  done
  note gate1-canary-eval "canary --evaluate-only on both shards (union key reconciliation)"
  for s in "${SHARDS[@]}"; do
    MAX=$($PY -c "import sys; from agent_allocation.config import load_config; from agent_allocation.shards import shard_config; print(shard_config(load_config(sys.argv[1]), sys.argv[2])['budget']['credit_ceiling_usd'])" "$CFG" "$s")
    $A canary --config "$CFG" --run-dir "$RUN" --shard "$s" --allow-paid --max-usd "$MAX" --evaluate-only
    rc=$?; [ $rc -eq 0 ] || stop gate1 "canary evaluate-only $s rc=$rc (gates/gate1_canary.json of the shard)"
  done
  START_AT=smoke
fi
# START_AT=smoke: the canaries passed (and were evaluated) already; resume at the smoke without re-paying them
if [ "$START_AT" = smoke ]; then
  clear_sentinels
  note gate1-smoke "passk-run smoke on both shards"
  declare -A SJ
  for s in "${SHARDS[@]}"; do SJ[$s]=$(submit smoke "$s"); [ -n "${SJ[$s]}" ] || stop gate1 "smoke $s not submitted" 1; done
  wait_jobs "${SJ[@]}" || stop gate1 "smoke jobs did not end within ${WAIT_MAX}s" 1
  for s in "${SHARDS[@]}"; do [ "$(job_rc "${SJ[$s]}")" = 0 ] || stop gate1 "smoke $s rc=$(job_rc "${SJ[$s]}")"; done
  declare -A GJ
  for s in "${SHARDS[@]}"; do GJ[$s]=$(submit grade "$s"); [ -n "${GJ[$s]}" ] || stop gate1 "grade $s not submitted" 1; done
  wait_jobs "${GJ[@]}" || stop gate1 "grade jobs did not end within ${WAIT_MAX}s" 1
  for s in "${SHARDS[@]}"; do [ "$(job_rc "${GJ[$s]}")" = 0 ] || stop gate1 "smoke grade $s rc=$(job_rc "${GJ[$s]}")"; done
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
step_tokens(){ local x; for x in "$@"; do echo "${x%%|*}"; done; }
run_analysis(){  # $1 first step token ("" = from the start), then the step specs; each its own job, sequential
  local from=$1 spec tok st flags rel j started="" rc; shift
  [ -z "$from" ] && started=1
  for spec in "$@"; do
    IFS='|' read -r tok st flags rel <<< "$spec"
    [ -z "$started" ] && [ "$tok" = "$from" ] && started=1
    [ -n "$started" ] || continue
    local sk skip=""
    for sk in ${SKIP_SPLITS:-}; do [[ "$tok" == *-"$sk" ]] && skip=1; done
    [ -n "$skip" ] && { say "skip $tok: Gate 4 failed for its split"; continue; }
    note "analysis:$tok" "passk-analyze --stage $st ${flags} ${rel:+--release $rel}"
    local ex=("STAGE=$st")
    [ -n "$flags" ] && ex+=("FLAGS=$flags")
    [ -n "$rel" ] && ex+=("RELEASE=$rel")
    j=$(submit analyze "" "${ex[@]}")
    if [ -z "$j" ]; then echo "$tok (not submitted)" > "$FAILED_MARK"; stop "analysis:$tok" "not submitted" 1; fi
    if ! wait_jobs "$j"; then echo "$tok (job $j did not end)" > "$FAILED_MARK"; stop "analysis:$tok" "job $j did not end" 1; fi
    rc=$(job_rc "$j")
    if [ "$rc" != 0 ]; then
      echo "$tok job=$j rc=$rc state=$(job_state "$j")" > "$FAILED_MARK"
      stop "analysis:$tok" "passk-analyze --stage $st rc=$rc (resume: START_AT=$tok)"
    fi
  done
  [ -n "$started" ] || stop "analysis" "unknown step token '$from'" 2
}

if [ "$START_AT" = main ]; then
  squeue -h -u "$USER" -o %j 2>/dev/null | grep -qx "passk-run-.*" && stop main "a passk run job is already queued" 1
  clear_sentinels
  rm -f "$FAILED_MARK"
  note main "passk-run $MAIN_PARTITIONS on both shards"
  declare -A RJ FJ RESUB RESUB_AFTER
  for s in "${SHARDS[@]}"; do
    RJ[$s]=$(submit run "$s" "PARTITIONS=$MAIN_PARTITIONS"); RESUB[$s]=0; RESUB_AFTER[$s]=0
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
    for s in "${SHARDS[@]}"; do    # grade --follow ends at its 12 h limit: resubmit (graders are single-locked)
      if job_done "${FJ[$s]}"; then
        say "grade-follow $s (${FJ[$s]}) ended $(job_state "${FJ[$s]}"): resubmitting"
        rm -f "$RUN/shards/$s/control/STOP"
        j=$(submit grade-follow "$s"); [ -n "$j" ] || stop_main main "grade-follow $s resubmission failed" 1
        FJ[$s]=$j
      fi
    done
    if [ -n "${ANALYSIS_PID:-}" ] && ! kill -0 "$ANALYSIS_PID" 2>/dev/null; then
      wait "$ANALYSIS_PID"; arc=$?
      ANALYSIS_PID=""; analysis_rc=$arc
      [ "$arc" = 0 ] || stop_main "analysis:$(cut -d' ' -f1 "$FAILED_MARK" 2>/dev/null)" \
        "pre-seal analysis chain failed during generation: $(cat "$FAILED_MARK" 2>/dev/null) (rc=$arc)"
    fi
    if [ -z "$analysis_started" ] && [ "$(stage1_done)" = yes ]; then
      analysis_started=1
      note main "stage 1 complete on both shards: analysis chain (pre-seal) in the background of generation"
      ( run_analysis "" "${ANALYSIS_PRE[@]}" ) &
      ANALYSIS_PID=$!
    fi
    all_done=1
    for s in "${SHARDS[@]}"; do
      j=${RJ[$s]}
      job_done "$j" || { all_done=""; continue; }
      rc=$(job_rc "$j")
      [ "$rc" = 0 ] && continue
      all_done=""
      # stopped early: resubmit only when no gate stopped the shard (a Slurm time-limit drain)
      if [ -f "$STUDY/gates/gate3_$s.json" ] && ! $PY -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['passed'] else 1)" "$STUDY/gates/gate3_$s.json"; then
        stop_main main "run $s stopped by Gate 3"
      fi
      [ -f "$RUN/shards/$s/control/STOP" ] && stop_main main "run $s ended with control/STOP present (owner hard stop)"
      now=$(date +%s)
      if [ "${RESUB_AFTER[$s]}" = 0 ]; then
        st=$(job_state "$j")
        # a killed dispatcher (TIMEOUT/OOM/NODE_FAIL/CANCELLED) leaves its claim: wait out the lease, not a resubmit
        case "$st" in COMPLETED|FAILED) RESUB_AFTER[$s]=$now ;; *) RESUB_AFTER[$s]=$((now + LEASE_WAIT)) ;; esac
        # a run that worked >= 30 min before ending (the 12 h time-limit drain) is benign: not counted
        el=$(sacct -j "$j" -n -X -P -o JobID,ElapsedRaw 2>/dev/null | awk -F'|' -v j="$j" '$1==j {print $2; exit}')
        [ "${el:-0}" -lt 1800 ] && RESUB[$s]=$((RESUB[$s] + 1))
        say "run $s ($j) ended $st rc=$rc elapsed=${el:-?}s; counted resubmits=${RESUB[$s]}"
      fi
      [ "${RESUB[$s]}" -gt 3 ] && stop_main main "run $s failed fast 3+ times (last rc=$rc; see run_decisions.jsonl)"
      [ "$now" -lt "${RESUB_AFTER[$s]}" ] && continue
      rm -f "$RUN/shards/$s/control/DRAIN"
      RJ[$s]=$(submit run "$s" "PARTITIONS=$MAIN_PARTITIONS"); RESUB_AFTER[$s]=0
      [ -n "${RJ[$s]}" ] || stop_main main "run $s resubmission failed" 1
      say "resubmitted run $s as ${RJ[$s]}"
    done
    note main "polling; runs: ${RJ[*]}; grade-follow: ${FJ[*]}"
    [ -n "$all_done" ] && break
  done
  if [ -n "${ANALYSIS_PID:-}" ]; then
    wait "$ANALYSIS_PID" || stop "analysis:$(cut -d' ' -f1 "$FAILED_MARK" 2>/dev/null)" \
      "pre-seal analysis chain failed: $(cat "$FAILED_MARK" 2>/dev/null)"
  elif [ -z "${analysis_rc:-}" ]; then
    run_analysis "" "${ANALYSIS_PRE[@]}"
  fi
  START_AT=gate4
fi
[ "$START_AT" = analysis ] && START_AT=extract
if step_tokens "${ANALYSIS_PRE[@]}" | grep -qx "$START_AT"; then
  rm -f "$FAILED_MARK"
  for s in "${SHARDS[@]}"; do ensure_grade_follow "$s" || stop "analysis" "grade-follow $s not submitted" 1; done
  run_analysis "$START_AT" "${ANALYSIS_PRE[@]}"; START_AT=gate4
fi

# ====================================================================================================================
# Gate 4 per shard -> release (the grade --follow jobs grade the held-out within one poll of the seal) -> score/report
# ====================================================================================================================
if [ "$START_AT" = gate4 ]; then
  for s in "${SHARDS[@]}"; do ensure_grade_follow "$s" || stop gate4 "grade-follow $s not submitted" 1; done
  # Gate 4 per held-out split: V48 is released once v48_bank passes on both shards, T200 once t200_bank does
  SKIP_SPLITS=""
  for split in v48 t200; do
    for s in "${SHARDS[@]}"; do
      note gate4 "completion checks on $s for $split"
      checkpoint gate4 "$s" --split "$split"; rc=$?
      if [ $rc -ne 0 ]; then SKIP_SPLITS="$SKIP_SPLITS $split"; say "Gate 4 failed on $s for $split (rc=$rc)"; break; fi
    done
  done
  [ "$(echo $SKIP_SPLITS)" = "v48 t200" ] && stop gate4 "Gate 4 failed for both splits (gates/gate4_<shard>_<split>.json)"
  # the seal exists (ANALYSIS_PRE ends with `seal`): the grade --follow jobs now grade the held-out partitions; wait
  # (max ~4 h) until every candidate of both shards is graded before releasing labels
  for i in $(seq 1 48); do
    pending=""
    for s in "${SHARDS[@]}"; do checkpoint graded "$s" >/dev/null || pending="$pending $s"; done
    [ -z "$pending" ] && break
    note gate4 "waiting for held-out grading on:$pending ($i/48)"; sleep 300
  done
  [ -n "$pending" ] && stop gate4 "held-out grading incomplete on:$pending (is grade --follow running? is the seal valid?)"
  START_AT=post
fi
[ "$START_AT" = post ] && START_AT=$(step_tokens "${ANALYSIS_POST[@]}" | head -1)
if step_tokens "${ANALYSIS_POST[@]}" | grep -qx "$START_AT"; then
  rm -f "$FAILED_MARK"
  run_analysis "$START_AT" "${ANALYSIS_POST[@]}"
  for s in "${SHARDS[@]}"; do touch "$RUN/shards/$s/control/STOP"; done      # ends the grade --follow jobs
  [ -n "${SKIP_SPLITS:-}" ] && stop gate4 "report built for the released split only; Gate 4 failed for:$SKIP_SPLITS \
(gates/gate4_<shard>_<split>.json; resume with START_AT=release-<split> after an owner decision)"
  note done "chain complete: report built"
  say "CONDUCTOR FINISHED"
  exit 0
fi
case "$START_AT" in
  gate0|gate1|smoke|main|gate4) exit 0 ;;
  *) stop start "unknown START_AT=$START_AT" 2 ;;
esac
