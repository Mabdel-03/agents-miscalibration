#!/bin/bash
# Owner-approved Gate 2 acceptance -> freeze -> launch dispatcher, follower grader, and finalizer (survives the session).
#   sbatch --export=APPROVAL="<owner quote + timestamp>" slurm/accept_freeze_launch.sbatch
#SBATCH --job-name=alloc-launch
#SBATCH --partition=pi_lhtsai
#SBATCH --nodelist=node4004
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/openrouter_agent_allocation_v1/logs/launch_%j.out
set -euo pipefail
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling_allocation_v1/agent_allocation
source slurm/env.sh
A=.venv/bin/allocation; CFG=configs/balanced150.json; RUN=artifacts/openrouter_agent_allocation_v1
echo "[launch] $(date -Is) approval: ${APPROVAL:?APPROVAL required}"
$A pilot --config $CFG --allow-paid --max-usd 5 --phase owner-accept --approval "$APPROVAL"
$A freeze --config $CFG | tail -20
DJ=$(sbatch --parsable slurm/dispatch.sbatch run --config $CFG --allow-paid --max-usd 135 --resume)
GJ=$(sbatch --parsable slurm/grade.sbatch --run-dir $RUN --resume --follow --workers 8 --config $CFG)
FJ=$(sbatch --parsable --dependency=afterany:$DJ --export=GRADER_JOB=$GJ slurm/finalize.sbatch)
echo "[launch] $(date -Is) dispatch=$DJ grade=$GJ finalize=$FJ"
