#!/bin/bash
# Gate 5 finalization, chained after the dispatcher so results appear without an interactive session:
#   sbatch --dependency=afterany:<DISPATCH_JOB> --export=GRADER_JOB=<GRADE_JOB> slurm/finalize.sbatch
# 1) ask the follower grader to stop (control/STOP) and wait for it to exit (graceful; its POSIX lock is released);
# 2) final cached grading pass (public checks before private grades); 3) seal every partition's selections before any
# grade join; 4) analyze; 5) audit. A STOP sentinel stays behind on purpose: resuming collection needs `stop --clear`.
#SBATCH --job-name=alloc-finalize
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/openrouter_agent_allocation_v1/logs/finalize_%j.out
set -uo pipefail
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling_allocation_v1/agent_allocation
source slurm/env.sh
RUN=artifacts/openrouter_agent_allocation_v1
CFG=configs/balanced150.json
step() { echo "[finalize] $(date -Is) $*"; }
step "host=$(hostname) job=${SLURM_JOB_ID:-none} grader_job=${GRADER_JOB:-none}"
.venv/bin/allocation stop --run-dir "$RUN"
if [ -n "${GRADER_JOB:-}" ]; then
  while squeue -h -j "$GRADER_JOB" 2>/dev/null | grep -q .; do sleep 30; done
  step "follower grader $GRADER_JOB has exited"
fi
step "final grading pass"; .venv/bin/allocation grade --run-dir "$RUN" --resume --workers 8 --config "$CFG"; echo "grade rc=$?"
step "seal";    .venv/bin/allocation seal --config "$CFG" --run-dir "$RUN"; echo "seal rc=$?"
step "analyze"; .venv/bin/allocation analyze --run-dir "$RUN" --config "$CFG"; echo "analyze rc=$?"
step "audit";   .venv/bin/allocation audit --run-dir "$RUN" --config "$CFG"; echo "audit rc=$?"
step "done"
