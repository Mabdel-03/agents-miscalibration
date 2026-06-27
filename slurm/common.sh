#!/bin/bash
# Sourced by every sbatch job. Sets modules, conda, and the scratch HF cache.
# Keep this idempotent and side-effect-free beyond exports + module loads.

module load miniforge/25.11.0-0 2>/dev/null || true
module load cuda/12.9.1 2>/dev/null || true

# Conda env prefixes (created under /home/mabdel03/conda_envs, which is NOT in the default
# envs_dirs, so activate by full prefix rather than by name).
export ASYS_SERVE_ENV="${ASYS_SERVE_ENV:-/home/mabdel03/conda_envs/serve_env}"
export ASYS_HARNESS_ENV="${ASYS_HARNESS_ENV:-/home/mabdel03/conda_envs/asys_env}"

# The system libstdc++ (GLIBCXX up to 3.4.25) is too old for flashinfer's compiled
# kernels (need GLIBCXX_3.4.26+). Prepend the serve_env's newer libstdc++ so vLLM/
# flashinfer load it instead of /lib64/libstdc++.so.6.
export LD_LIBRARY_PATH="$ASYS_SERVE_ENV/lib:${LD_LIBRARY_PATH:-}"

# Weights stay on scratch (large regenerable cache). Results live on the tpoggio DATA
# filesystem (8T+ free): the per-user SCRATCH quota hit EDQUOT and halted the run, so
# outputs moved to data where the group quota has ample headroom and writes never block.
export HF_HOME="${HF_HOME:-/orcd/scratch/orcd/012/mabdel03/.cache/huggingface}"
export ASYS_RESULTS_ROOT="${ASYS_RESULTS_ROOT:-/orcd/data/tpoggio/001/mabdel03/agents_scaling_results}"
mkdir -p "$HF_HOME" "$ASYS_RESULTS_ROOT"

# HF auth for gated repos (GPQA, Llama). Token is read from a private, untracked file
# (~/.config/agents_scaling/hf_token) so it never lands in git. Create it with:
#   mkdir -p ~/.config/agents_scaling && chmod 700 ~/.config/agents_scaling
#   printf 'hf_xxxxx' > ~/.config/agents_scaling/hf_token && chmod 600 ~/.config/agents_scaling/hf_token
_ASYS_HF_TOKEN_FILE="${ASYS_HF_TOKEN_FILE:-$HOME/.config/agents_scaling/hf_token}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$_ASYS_HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$_ASYS_HF_TOKEN_FILE")"
fi
# huggingface_hub also reads HUGGING_FACE_HUB_TOKEN; keep both in sync.
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# vLLM: avoid usage stats phone-home; sensible defaults.
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

# Reproducibility / quieter logs.
export TOKENIZERS_PARALLELISM=false
