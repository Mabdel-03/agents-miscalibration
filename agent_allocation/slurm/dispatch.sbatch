#!/bin/bash
# The single paid dispatcher. Usage (from agent_allocation/):
#   sbatch slurm/dispatch.sbatch run --config configs/balanced150.json --allow-paid --max-usd 135 --resume
# Pinned to one pi_lhtsai node (no preemption, 2-day limit) so state.sqlite on its local /scratch survives resubmits;
# the ledger journal, raw responses and DB backups live on /orcd/data regardless.
#SBATCH --job-name=alloc-dispatch
#SBATCH --partition=pi_lhtsai
#SBATCH --nodelist=node4004
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/openrouter_agent_allocation_v1/logs/dispatch_%j.out
#SBATCH --export=NONE
set -euo pipefail
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling_allocation_v1/agent_allocation
source slurm/env.sh
echo "[dispatch] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} args: $*"
exec .venv/bin/allocation "$@"
