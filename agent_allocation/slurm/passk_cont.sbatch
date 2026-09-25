#!/bin/bash
# passk_cont_v1: one step of the study (CPU only). Submitted by slurm/passk_conductor.sh (or by hand):
#   sbatch --export=ALL,VERB=<verb>,SHARD=<id>[,PARTITIONS=a+b,STAGE=x,MAX_USD=y] slurm/passk_cont.sbatch
# --export values NEVER contain commas: partitions are joined with '+' and split by passk-run itself.
#
# VERB
#   shard-plan    allocation shard-plan --write, then adopt each legacy forced_v2 shard endpoint lock  (no SHARD)
#   gate0         python -m agent_allocation.passk.gates gate0 --check-key --run-tests --run-e0      (no SHARD, $0)
#   canary        allocation canary --shard SHARD --allow-paid (Gate 1 live deep exactness; <= $0.50 per shard)
#   canary-eval   allocation canary --shard SHARD --evaluate-only (the union key reconciliation)
#   smoke         passk-run --partitions smoke (12 D32 items x 1A across both shards)
#   run           passk-run --partitions ${PARTITIONS:-c48_extra+e1_fill+t200_diag+v48_bank+t200_bank}
#   grade         allocation grade --resume (one pass; the held-out fence applies)
#   grade-follow  allocation grade --resume --follow (fenced every pass; ends when control/STOP exists)
#   checkpoint    python -m agent_allocation.passk.checkpoints ${STAGE} --shard SHARD (Gate 1 smoke / 2 / 3 / 4)
#   analyze       allocation passk-analyze --stage ${STAGE} (WP-A)                                  (no SHARD)
#
# Paid verbs (canary, smoke, run) always pass --account-auto-reload to passk-run (the account auto-refills; the key
# limit, --max-usd and the shard slice still bind). SIGTERM (Slurm --signal, scancel --signal=TERM --batch) drains:
# the trap touches control/DRAIN, the dispatcher stops admitting and finishes in-flight work, and the script waits
# for it (the python runs in the background and is waited on twice; no exec, so the trap can run).
#SBATCH --job-name=passk-cont
#SBATCH --partition=pi_lhtsai,mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1/logs/%x_%j.out
set -uo pipefail
CODE=${CODE:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_cont/agent_allocation}
cd "$CODE" || exit 2
source slurm/env.sh
VERB=${VERB:?VERB required}
CFG=${CFG:-configs/passk_cont_v1.json}
STUDY=${STUDY:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1}
RUN="$STUDY/artifacts"
SHARD=${SHARD:-}
A=.venv/bin/allocation
PY=.venv/bin/python
SHARD_DIR="$RUN/shards/${SHARD:-_none_}"
SHARD_CFG="$SHARD_DIR/state/shard_config.json"
mkdir -p "$STUDY/logs"
echo "[passk] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} verb=$VERB shard=${SHARD:-} code=$CODE head=$(git -C "$CODE" rev-parse --short HEAD 2>/dev/null)"

need_shard(){
  [ -n "$SHARD" ] || { echo "VERB=$VERB needs SHARD"; exit 2; }
  [ -d "$SHARD_DIR" ] || { echo "no shard dir $SHARD_DIR: run VERB=shard-plan first"; exit 2; }
}
shard_max_usd(){  # the shard's own credit slice (the CLI refuses more)
  $PY -c "
import sys
from agent_allocation.config import load_config
from agent_allocation.shards import shard_config
print(shard_config(load_config(sys.argv[1]), sys.argv[2])['budget']['credit_ceiling_usd'])" "$CFG" "$SHARD"
}

CHILD=""
on_term(){
  echo "[passk] $(date -Is) SIGTERM: draining ${SHARD_DIR}"
  if [ -n "$SHARD" ] && [ -d "$SHARD_DIR" ]; then mkdir -p "$SHARD_DIR/control"; touch "$SHARD_DIR/control/DRAIN"; fi
}
trap on_term TERM

run_waited(){  # run "$@" in the background; wait (twice if a signal interrupted the first wait); return its code
  "$@" &
  CHILD=$!
  wait "$CHILD"; local rc=$?
  if kill -0 "$CHILD" 2>/dev/null; then
    wait "$CHILD"; rc=$?
  fi
  echo "[passk] $(date -Is) verb=$VERB exit=$rc"
  return $rc
}

common=(--config "$CFG" --run-dir "$RUN")
case "$VERB" in
  shard-plan)
    run_waited $A shard-plan "${common[@]}" --write || exit $?
    for s in mancer parasail; do
      run_waited $A passk-run "${common[@]}" --shard "$s" --partitions smoke --adopt-legacy-lock || exit $?
    done ;;
  gate0)
    run_waited $PY -m agent_allocation.passk.gates gate0 --config "$CFG" --study-root "$STUDY" \
      --check-key --run-tests --run-e0 ;;
  canary)
    need_shard; MAX_USD=${MAX_USD:-$(shard_max_usd)}
    run_waited $A canary "${common[@]}" --shard "$SHARD" --allow-paid --max-usd "$MAX_USD" ;;
  canary-eval)
    need_shard; MAX_USD=${MAX_USD:-$(shard_max_usd)}
    run_waited $A canary "${common[@]}" --shard "$SHARD" --allow-paid --max-usd "$MAX_USD" --evaluate-only ;;
  smoke)
    need_shard; MAX_USD=${MAX_USD:-$(shard_max_usd)}
    run_waited $A passk-run "${common[@]}" --shard "$SHARD" --partitions smoke --allow-paid --max-usd "$MAX_USD" \
      --account-auto-reload --resume ;;
  run)
    need_shard; MAX_USD=${MAX_USD:-$(shard_max_usd)}
    PARTITIONS=${PARTITIONS:-c48_extra+e1_fill+t200_diag+v48_bank+t200_bank}
    run_waited $A passk-run "${common[@]}" --shard "$SHARD" --partitions "$PARTITIONS" --allow-paid \
      --max-usd "$MAX_USD" --account-auto-reload --resume ;;
  grade)
    need_shard
    run_waited $A grade --run-dir "$SHARD_DIR" --resume --workers "${WORKERS:-4}" --config "$SHARD_CFG" ;;
  grade-follow)
    need_shard
    run_waited $A grade --run-dir "$SHARD_DIR" --resume --follow --workers "${WORKERS:-4}" --config "$SHARD_CFG" \
      --poll-seconds "${POLL_SECONDS:-60}" ;;
  checkpoint)
    need_shard; : "${STAGE:?STAGE required (gate1-smoke|gate2|gate3|gate4)}"
    run_waited $PY -m agent_allocation.passk.checkpoints "$STAGE" --config "$CFG" --study-root "$STUDY" \
      --shard "$SHARD" ;;
  analyze)
    : "${STAGE:?STAGE required}"
    run_waited $A passk-analyze --config "$CFG" --run-root "$STUDY" --stage "$STAGE" ;;
  *) echo "unknown VERB=$VERB"; exit 2 ;;
esac
