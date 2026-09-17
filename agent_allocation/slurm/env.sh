# Sourced by every allocation job and interactive shell. Keeps every cache off /home (95% of its file quota)
# and pins uv to the harness interpreter. Never export the OpenRouter key here.
export ALLOCATION_ROOT=/orcd/data/tpoggio/001/mabdel03/agents_scaling_allocation_v1/agent_allocation
export UV_CACHE_DIR=/orcd/data/tpoggio/001/mabdel03/.cache/uv
export UV_PYTHON=/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python3.11
export UV_PYTHON_DOWNLOADS=never
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
export MPLCONFIGDIR=/orcd/data/tpoggio/001/mabdel03/.cache/matplotlib
export HF_HOME=/orcd/data/tpoggio/001/mabdel03/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
