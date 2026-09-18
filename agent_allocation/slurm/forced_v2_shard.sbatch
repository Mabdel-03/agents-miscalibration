#!/bin/bash
# One provider shard of forced_budget_v2 (CPU only). Submitted once per shard by slurm/launch_forced_v2_shards.sh:
#   sbatch --export=SHARD=<id>,VERB=run[,CFG=...,RUN=...,MAX_USD=...,APPROVAL=...] slurm/forced_v2_shard.sbatch
# VERB: canary | calibrate | pilot | freeze | run | finalize (stop + grade + seal + analyze of this shard's run dir).
# Every shard runs in <RUN>/shards/<SHARD>/ under its derived config (endpoint tag set, budgets = share of the cap);
# the ledger journal, raw responses and the state backup live on /orcd/data, so a resubmit on another node resumes.
#SBATCH --job-name=fv2-shard
#SBATCH --partition=mit_normal,pi_lhtsai
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/logs/shard_%x_%j.out
set -euo pipefail
CODE=${CODE:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_forced_v2/agent_allocation}
cd "$CODE"
source slurm/env.sh
: "${SHARD:?SHARD required (a shards[].id of the config)}"
VERB=${VERB:-run}
CFG=${CFG:-configs/forced27b.json}
RUN=${RUN:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/artifacts}
A=.venv/bin/allocation
SHARD_DIR="$RUN/shards/$SHARD"
SHARD_CFG="$SHARD_DIR/state/shard_config.json"
# Default --max-usd: the shard's own credit ceiling (share/sum(share) of the study cap; the CLI refuses more).
MAX_USD=${MAX_USD:-$(.venv/bin/python -c "
import sys
from agent_allocation.config import load_config
from agent_allocation.shards import shard_config
print(shard_config(load_config(sys.argv[1]), sys.argv[2])['budget']['credit_ceiling_usd'])" "$CFG" "$SHARD")}
echo "[shard] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} shard=$SHARD verb=$VERB max_usd=$MAX_USD run=$RUN"
[ -d "$SHARD_DIR" ] || { echo "no shard run dir $SHARD_DIR: run 'allocation shard-plan --config $CFG --run-dir $RUN --write' first"; exit 2; }
common=(--config "$CFG" --run-dir "$RUN" --shard "$SHARD")
case "$VERB" in
  canary)    exec $A canary "${common[@]}" --allow-paid --max-usd "$MAX_USD" ;;
  calibrate) exec $A calibrate "${common[@]}" --allow-paid --max-usd "$MAX_USD" ;;
  pilot)     exec $A pilot "${common[@]}" --allow-paid --max-usd "$MAX_USD" ;;
  freeze)    exec $A freeze "${common[@]}" ;;
  run)       exec $A run "${common[@]}" --allow-paid --max-usd "$MAX_USD" --resume ;;
  finalize)
    set +e
    $A stop --run-dir "$SHARD_DIR"
    $A grade --run-dir "$SHARD_DIR" --resume --workers 4 --config "$SHARD_CFG"; echo "grade rc=$?"
    $A seal "${common[@]}"; echo "seal rc=$?"
    $A analyze --run-dir "$SHARD_DIR" --config "$SHARD_CFG"; echo "analyze rc=$?"
    ;;
  *) echo "unknown VERB=$VERB"; exit 2 ;;
esac
