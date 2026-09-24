#!/bin/bash
# Submit one slurm/dynorch_v1.sbatch job and print its expected start (plan §9 Gate 3: `squeue --start` checked).
#   slurm/launch_dynorch_v1.sh [VERB] [--stage S] [--dry-run] [--test-only]
# VERB: gate0 | canary | pilot | run | finalize (default run). Env passthrough: CFG, RUN, CODE, MAX_USD, RUN_FLAGS.
# --test-only asks Slurm when the job would start without submitting it (sbatch --test-only).
set -euo pipefail
CODE=${CODE:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_dynorch_v1/agent_allocation}
cd "$CODE"
VERB=run; DRY=0; TEST=0; STAGE=${STAGE:-main}
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --test-only) TEST=1 ;;
    --stage) STAGE="$2"; shift ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) VERB="$1" ;;
  esac
  shift
done
case "$VERB" in gate0|canary|pilot|run|finalize) ;; *) echo "unknown VERB=$VERB"; exit 2 ;; esac
CFG=${CFG:-configs/dynorch_v1.json}
RUN=${RUN:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/dynorch_v1/artifacts}
mkdir -p "$(dirname "$RUN")/logs"
# `--export` splits on commas, so `RUN_FLAGS="--arms H1,H3"` would arrive as `--arms H1` plus a bogus variable
# `H3` and the stage would silently run fewer arms (a restriction never adds cells, so nothing complains).
# Review minor 19: refuse commas, and keep the caller's environment (ALL) so PATH/conda reach the job.
case "${RUN_FLAGS:-}" in
  *,*) echo "RUN_FLAGS must not contain a comma (sbatch --export splits on it): ${RUN_FLAGS}" >&2
       echo "       use spaces: RUN_FLAGS=\"--arms H1 --arms H3\"" >&2; exit 2 ;;
esac
EXPORTS="ALL,VERB=$VERB,STAGE=$STAGE,CFG=$CFG,RUN=$RUN,CODE=$CODE${MAX_USD:+,MAX_USD=$MAX_USD}${RUN_FLAGS:+,RUN_FLAGS=$RUN_FLAGS}"
CMD=(sbatch --parsable --job-name="dyn-$VERB-$STAGE" --export="$EXPORTS" slurm/dynorch_v1.sbatch)
if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
if [ "$TEST" = 1 ]; then sbatch --test-only --job-name="dyn-$VERB-$STAGE" --export="$EXPORTS" slurm/dynorch_v1.sbatch; exit $?; fi
JID=$("${CMD[@]}")
echo "[launch] $(date -Is) verb=$VERB stage=$STAGE job=$JID log=$(dirname "$RUN")/logs/dyn-$VERB-${STAGE}_$JID.out"
sleep 2
squeue --start -j "$JID" || true
echo "[launch] if START_TIME is hours away, cancel (scancel $JID) and resubmit on the other partition; the runbook has the stop rules"
