#!/bin/bash
#SBATCH --job-name=passk-conductor
#SBATCH --partition=pi_lhtsai
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL,END
# Mail recipient: never hard-coded. Export OWNER_MAIL and submit with
#   sbatch --mail-user="$OWNER_MAIL" --export=ALL slurm/passk_conductor.sbatch
# (the conductor also sets its own job's MailUser from OWNER_MAIL via scontrol, and STATUS.md shows it).
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1/logs/conductor_%j.out
# Runs the passk_cont_v1 gate chain (slurm/passk_conductor.sh) independently of any interactive session. It submits
# every paid / grading / analysis step as its own job of slurm/passk_cont.sbatch, evaluates the gates between them and
# exits non-zero on any stop (Slurm mails FAIL). START_AT=<gate0|gate1|main|analysis|gate4> via --export to resume.
# Gate 0 runs the passk tests here, hence 4 CPUs.
set -uo pipefail
CODE=${CODE:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_cont/agent_allocation}
bash "$CODE/slurm/passk_conductor.sh"
