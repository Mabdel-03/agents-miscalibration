#!/bin/bash
# dynorch_v1: the metering proxy and the harness runner share ONE job (both on 127.0.0.1). Plan §9 Gate 3 header.
#   sbatch --export=VERB=gate0|canary|pilot|run|finalize[,CFG=...,RUN=...,MAX_USD=...,RUN_FLAGS=...,STAGE=...] slurm/dynorch_v1.sbatch
# VERB: gate0 (offline, $0) | canary | pilot | run (--stage main, --resume) | finalize (reconcile -> tables -> gate 4).
# RUN_FLAGS passes through to dynorch-run (e.g. "--account-auto-reload --concurrency 12" or "--arms H1 --rungs 16").
# STAGE overrides the stage of VERB=run/finalize (main by default; ext_e0..ext_e3 for the extensions).
# The runner traps SIGTERM (--signal=B:TERM@900): stops admitting, waits <= 800 s for in-flight settles, exits 0;
# resubmit the same job with --resume to continue.
#SBATCH --job-name=dyn-v1
#SBATCH --partition=pi_lhtsai,mit_normal
#SBATCH --account=mit_general
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/dynorch_v1/logs/%x_%j.out
set -euo pipefail
CODE=${CODE:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_dynorch_v1/agent_allocation}
cd "$CODE"
source slurm/env.sh
# Every harness state dir, temp dir and npm cache stays off /home (995k of 1,000k inodes).
export DYNORCH_SCRATCH=${DYNORCH_SCRATCH:-/orcd/data/tpoggio/001/mabdel03/dynorch_scratch}
export TMPDIR=$DYNORCH_SCRATCH/tmp
export npm_config_cache=$DYNORCH_SCRATCH/npm-cache
export NODE_OPTIONS=--max-old-space-size=1024
export PATH="$DYNORCH_SCRATCH/npm-prefix/bin:$PATH"          # codex 0.156.1 (tools/install_codex.sh)
# openclaude 0.13.0 lives in the consortium conda env and is NOT on a bare job PATH (review minor 19/major 8).
export OPENCLAUDE_BIN_DIR=${OPENCLAUDE_BIN_DIR:-/orcd/home/002/mabdel03/conda_envs/consortium/bin}
[ -x "$OPENCLAUDE_BIN_DIR/openclaude" ] && export PATH="$OPENCLAUDE_BIN_DIR:$PATH"
# nothing is created under /scratch or /tmp: the local state dir follows the run's scratch root
export ALLOCATION_LOCAL_STATE=${ALLOCATION_LOCAL_STATE:-$DYNORCH_SCRATCH/local_state}
mkdir -p "$TMPDIR" "$npm_config_cache" "$ALLOCATION_LOCAL_STATE"
VERB=${VERB:-run}
CFG=${CFG:-configs/dynorch_v1.json}
RUN=${RUN:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/dynorch_v1/artifacts}
STAGE=${STAGE:-main}
mkdir -p "$(dirname "$RUN")/logs"
A=.venv/bin/allocation
# Default ceilings: the stage allowances of the config for canary/pilot, the credit ceiling otherwise. The CLI
# refuses anything above the preset ceiling (config.PRESET_CEILINGS_USD) without a recorded --approval.
budget() { .venv/bin/python -c "import json,sys;print(json.load(open(sys.argv[1]))['budget'][sys.argv[2]])" "$CFG" "$1"; }
case "$VERB" in
  canary) MAX_USD=${MAX_USD:-$(budget canary_allowance_usd)} ;;
  pilot)  MAX_USD=${MAX_USD:-$(budget pilot_allowance_usd)} ;;
  *)      MAX_USD=${MAX_USD:-$(budget credit_ceiling_usd)} ;;
esac
echo "[dynorch] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} verb=$VERB stage=$STAGE max_usd=$MAX_USD run=$RUN"
echo "[dynorch] codex=$(command -v codex || echo MISSING) openclaude=$(command -v openclaude || echo MISSING) tmpdir=$TMPDIR"
common=(--config "$CFG" --run-dir "$RUN")
case "$VERB" in
  gate0)    exec $A dynorch-gates "${common[@]}" --gate 0 ;;
  # every VERB resumes: re-running a stage without --resume is refused when records exist (review major 4/15)
  canary)   exec $A dynorch-run "${common[@]}" --stage canary --allow-paid --max-usd "$MAX_USD" --resume ${RUN_FLAGS:-} ;;
  pilot)    exec $A dynorch-run "${common[@]}" --stage pilot  --allow-paid --max-usd "$MAX_USD" --resume ${RUN_FLAGS:-} ;;
  run)      exec $A dynorch-run "${common[@]}" --stage "$STAGE" --allow-paid --max-usd "$MAX_USD" --resume ${RUN_FLAGS:-} ;;
  finalize)
    set +e
    $A dynorch-reconcile "${common[@]}" --stage "$STAGE" --phase after; echo "reconcile rc=$?"
    $A dynorch-tables "${common[@]}"; echo "tables rc=$?"
    $A dynorch-gates "${common[@]}" --gate 4 --stage "$STAGE"; echo "gate4 rc=$?"
    ;;
  *) echo "unknown VERB=$VERB"; exit 2 ;;
esac
