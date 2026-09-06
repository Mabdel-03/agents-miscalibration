#!/bin/bash
# Sourced by every sbatch job. Sets modules, conda, and the scratch HF cache.
# Keep this idempotent and side-effect-free beyond exports + module loads.

module load miniforge/25.11.0-0 2>/dev/null || true
module load cuda/12.9.1 2>/dev/null || true

# Conda env prefixes (created under /home/mabdel03/conda_envs, which is NOT in the default
# envs_dirs, so activate by full prefix rather than by name).
export ASYS_SERVE_ENV="${ASYS_SERVE_ENV:-/orcd/home/002/mabdel03/conda_envs/serve_env}"
export ASYS_HARNESS_ENV="${ASYS_HARNESS_ENV:-/orcd/home/002/mabdel03/conda_envs/asys_env}"

# The system libstdc++ (GLIBCXX up to 3.4.25) is too old for flashinfer's compiled
# kernels (need GLIBCXX_3.4.26+). Prepend the serve_env's newer libstdc++ so vLLM/
# flashinfer load it instead of /lib64/libstdc++.so.6.
export LD_LIBRARY_PATH="$ASYS_SERVE_ENV/lib:${LD_LIBRARY_PATH:-}"

# BOTH the HF cache (weights + datasets) AND results live on the tpoggio DATA filesystem
# (8T+ free, group quota). The per-user SCRATCH quota repeatedly hit EDQUOT: first it blocked
# results writes (moved to data 06-23), then it RECURRED and blocked cells writing HF dataset
# .lock files because HF_HOME was still on scratch (moved to data 06-28). Keeping everything
# on data means no run component depends on the contended scratch quota.
export HF_HOME="${HF_HOME:-/orcd/data/tpoggio/001/mabdel03/.cache/huggingface}"
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

# study-v4: legacy server renders need an explicit model contract, otherwise the
# --register role resolves an empty path to the repo root and the job dies.
export ASYS_MODEL_CONTRACT="${ASYS_MODEL_CONTRACT:-/orcd/data/tpoggio/001/mabdel03/agents_scaling/configs/model_contracts.v1.json}"
# study-v4: keep pip/apptainer caches off $HOME (file quota ~85%).
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/orcd/data/tpoggio/001/mabdel03/.cache/pip}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/orcd/data/tpoggio/001/mabdel03/.cache/apptainer}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/orcd/data/tpoggio/001/mabdel03/.cache/apptainer/tmp}"

# study-v4: compute nodes lack /usr/bin/apptainer; use the site module binary (works without `module load`).
export ASYS_APPTAINER_BIN="${ASYS_APPTAINER_BIN:-/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer}"
