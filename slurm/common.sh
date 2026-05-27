#!/bin/bash
# Sourced by every sbatch job. Sets modules, conda, and the scratch HF cache.
# Keep this idempotent and side-effect-free beyond exports + module loads.

module load miniforge/25.11.0-0 2>/dev/null || true
module load cuda/12.9.1 2>/dev/null || true

# Weights + results on scratch (project dir has little free space).
export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/012/mabdel03/.cache/huggingface}"
export ASYS_RESULTS_ROOT="${ASYS_RESULTS_ROOT:-/orcd/scratch/orcd/012/mabdel03/agents_scaling_results}"
mkdir -p "$HF_HOME" "$ASYS_RESULTS_ROOT"

# vLLM: avoid usage stats phone-home; sensible defaults.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

# Reproducibility / quieter logs.
export TOKENIZERS_PARALLELISM=false
