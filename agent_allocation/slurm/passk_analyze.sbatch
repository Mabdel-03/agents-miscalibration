#!/bin/bash
# One passk-analyze stage (WP-A; $0, CPU only). The conductor submits one job per stage:
#   sbatch --export=STAGE=pairs slurm/passk_analyze.sbatch                     (P0 legacy pair cache, 64 CPUs)
#   sbatch --cpus-per-task=4 --export=STAGE=extract slurm/passk_analyze.sbatch
#   sbatch --cpus-per-task=4 --export=STAGE=release,RELEASE=v48 slurm/passk_analyze.sbatch
# Stages: extract pairs e1-lock e2-fit e5-fit seal release score report e0. FLAGS is split on '+' (no commas in
# --export): FLAGS=--strace for the fit stages (Gate P needs their open logs).
#SBATCH --job-name=passk-analyze
#SBATCH --partition=mit_normal,pi_lhtsai
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1/logs/analyze_%x_%j.out
set -euo pipefail
CODE=${CODE:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_cont/agent_allocation}
cd "$CODE"
source slurm/env.sh
: "${STAGE:?STAGE required}"
RUN_ROOT=${RUN_ROOT:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1}
CFG=${CFG:-configs/passk_cont_v1.json}
args=(--config "$CFG" --run-root "$RUN_ROOT" --stage "$STAGE" --workers "${SLURM_CPUS_PER_TASK:-1}")
[ -n "${RELEASE:-}" ] && args+=(--release "$RELEASE")
[ -n "${FLAGS:-}" ] && IFS='+' read -r -a extra <<< "$FLAGS" && args+=("${extra[@]}")
echo "[passk-analyze] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} stage=$STAGE run_root=$RUN_ROOT"
.venv/bin/allocation passk-analyze "${args[@]}"
