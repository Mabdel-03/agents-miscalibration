#!/bin/bash
#SBATCH --job-name=asys-neural-test
#SBATCH --partition=pi_manoli
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/neural/_smoke/smoke_%j.out
#SBATCH --no-requeue
#SBATCH --export=NONE

# N1 GPU verification (Qwen3-8B): mirrors the environment block of slurm/study_neural.sbatch.tmpl.
set -euo pipefail
umask 027
REPO=/orcd/data/tpoggio/001/mabdel03/agents_scaling
PYTHON=/orcd/data/tpoggio/001/mabdel03/envs/neural_env/bin/python
OUT_DIR=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/neural/_smoke

export PATH=/usr/bin:/bin
export LANG=C.UTF-8 LC_ALL=C.UTF-8
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
export HF_HOME=/orcd/data/tpoggio/001/mabdel03/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false DO_NOT_TRACK=1

ASYS_LOCAL_CACHE="/tmp/asys_neural_${SLURM_JOB_ID}"
mkdir -p "$ASYS_LOCAL_CACHE"/{xdg,triton,torchinductor,tmp}
trap 'rm -rf "$ASYS_LOCAL_CACHE"' EXIT
export XDG_CACHE_HOME="$ASYS_LOCAL_CACHE/xdg"
export TRITON_CACHE_DIR="$ASYS_LOCAL_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$ASYS_LOCAL_CACHE/torchinductor"
export TMPDIR="$ASYS_LOCAL_CACHE/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[smoke] job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
cd "$REPO"
"$PYTHON" -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
"$PYTHON" tests/study_neural/gpu_smoke.py --size 8B --out "$OUT_DIR/smoke_${SLURM_JOB_ID}.json" --max-batch-tokens 16384
echo "[smoke] end=$(date -Is) rc=$?"
