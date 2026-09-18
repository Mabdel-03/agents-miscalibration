#!/bin/bash
# Submit one slurm/forced_v2_shard.sbatch job per provider shard of the config, all concurrently.
#   slurm/launch_forced_v2_shards.sh [VERB] [--dry-run] [--shard ID ...]
# VERB defaults to run. With VERB=finalize a dependent merge job (merge-shards) is submitted after every shard's
# finalize job; with any other verb the shards simply run side by side. Env: CFG, RUN, CODE, MAX_USD (per shard).
set -euo pipefail
CODE=${CODE:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_forced_v2/agent_allocation}
cd "$CODE"
VERB=run; DRY=0; ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --shard) ONLY+=("$2"); shift ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) VERB="$1" ;;
  esac
  shift
done
CFG=${CFG:-configs/forced27b.json}
RUN=${RUN:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/artifacts}
mkdir -p /orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/logs
if [ ${#ONLY[@]} -gt 0 ]; then SHARDS=("${ONLY[@]}"); else
  mapfile -t SHARDS < <(.venv/bin/python -c "
import sys
from agent_allocation.config import load_config
from agent_allocation.shards import shard_ids
print('\n'.join(shard_ids(load_config(sys.argv[1]))))" "$CFG")
fi
JOBS=()
for SHARD in "${SHARDS[@]}"; do
  EXPORTS="SHARD=$SHARD,VERB=$VERB,CFG=$CFG,RUN=$RUN,CODE=$CODE${MAX_USD:+,MAX_USD=$MAX_USD}"
  CMD=(sbatch --parsable --job-name="fv2-$VERB-$SHARD" --export="$EXPORTS" slurm/forced_v2_shard.sbatch)
  if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; continue; fi
  JID=$("${CMD[@]}")
  JOBS+=("$JID")
  echo "[launch] $(date -Is) shard=$SHARD verb=$VERB job=$JID"
done
if [ "$VERB" = finalize ] && [ "$DRY" = 0 ] && [ ${#JOBS[@]} -gt 0 ]; then
  DEP=$(IFS=:; echo "afterany:${JOBS[*]}")
  MJ=$(sbatch --parsable --job-name=fv2-merge --dependency="$DEP" --partition=mit_normal,pi_lhtsai \
       --cpus-per-task=8 --mem=48G --time=04:00:00 \
       --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/logs/merge_%j.out \
       --wrap="cd $CODE && source slurm/env.sh && .venv/bin/allocation merge-shards --run-dir $RUN --config $CFG")
  echo "[launch] merge job=$MJ after $DEP"
fi
