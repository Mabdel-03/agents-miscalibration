#!/bin/bash
# Offline grader: public checks for code first, then private grades. Never blocks generation; resubmittable.
#   sbatch slurm/grade.sbatch --run-dir artifacts/openrouter_agent_allocation_v1 --resume --follow --workers 8
#SBATCH --job-name=alloc-grade
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/openrouter_agent_allocation_v1/logs/grade_%j.out
#SBATCH --export=NONE
set -euo pipefail
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling_allocation_v1/agent_allocation
source slurm/env.sh
echo "[grade] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} args: $*"
exec .venv/bin/allocation grade "$@"
