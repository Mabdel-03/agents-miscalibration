# Environment setup

Two conda environments, by design (the vLLM serving stack pins a newer torch/CUDA than
the harness/analysis code needs, and we don't want a single env where a vLLM upgrade
breaks pandas/datasets):

| env | purpose | runs where |
|-----|---------|-----------|
| `serve_env` | runs `vllm serve` (owns its torch + CUDA) | GPU nodes (serving sbatch jobs) |
| `asys_env` | the `agents_scaling` harness + analysis (HTTP clients, datasets, plotting) | CPU is fine; runs the sweep cells + notebooks |

Agents are thin HTTP clients to the vLLM server, so the harness env needs **no GPU torch**.

## 0. Shared: where weights and results live

Use the group data filesystem for Hugging Face state and results. The per-user scratch
quota repeatedly returned `EDQUOT`, including on cache lock files, so production must not
silently fall back to scratch:

```bash
export HF_HOME=/orcd/data/tpoggio/001/mabdel03/.cache/huggingface
export ASYS_RESULTS_ROOT=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results
mkdir -p "$HF_HOME" "$ASYS_RESULTS_ROOT"
```

The legacy `slurm/common.sh` has the same development defaults. Schema-5 production does
not source it: the immutable control pins export both paths explicitly.

## 1. serve_env (vLLM)

```bash
module load miniforge/25.11.0-0
module load cuda/12.9.1
mamba create -y -n serve_env python=3.11
mamba activate serve_env
# Let vLLM pull its own matched torch/CUDA wheels:
pip install -r env/serve_env.txt
# sanity: the native reasoning-budget contract is version-pinned
python -c "import vllm; assert vllm.__version__ == '0.21.0', vllm.__version__; print('vllm', vllm.__version__)"
```

## 2. asys_env (harness + analysis)

```bash
module load miniforge/25.11.0-0
mamba create -y -n asys_env python=3.11
mamba activate asys_env
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling
pip install -e ".[embeddings,dev]"      # installs deps from pyproject + this package
# sanity
python -c "import agents_scaling, openai, datasets; print('asys ok')"
pytest -q                               # unit tests should pass without a GPU
```

> An existing `consortium` conda env has torch 2.3.1 / transformers 4.44.2 — **too old**
> for current vLLM. Do not reuse it for `serve_env`. It *could* host `asys_env`'s deps,
> but a clean env avoids surprises.

## 3. asys_analysis (notebooks + statistics)

A third env for the post-sweep analysis notebooks (`analysis/notebooks/`). Kept separate
from `asys_env` so the notebook stack (jupyterlab, statsmodels, seaborn) never collides
with the harness pins. Prefix install under `/orcd/home/002/mabdel03/conda_envs/`:

```bash
module load miniforge/25.11.0-0
mamba create -y -p /orcd/home/002/mabdel03/conda_envs/asys_analysis python=3.11
mamba activate /orcd/home/002/mabdel03/conda_envs/asys_analysis
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling
pip install -e ".[dev]"                  # agents_scaling editable + core deps (no embeddings/torch)
pip install -r env/analysis_nb_env.txt   # jupyterlab, ipykernel, statsmodels, seaborn, ...
python -m ipykernel install --user --name asys_analysis \
  --display-name "asys_analysis (py3.11, agents_scaling)"
# sanity
python -c "import agents_scaling, statsmodels, seaborn, jupyterlab; print('asys_analysis ok')"
```

The kernel then appears in any Jupyter front-end as `asys_analysis (py3.11, agents_scaling)`.
All analysis caches and figures go under `analysis/` on the **data** filesystem — never
`/orcd/scratch` (recurring EDQUOT quota failures killed jobs writing there).

## 4. Gated weights (Llama only)

The Qwen2.5 ladder is ungated. If you run the optional Llama-3.1 robustness check:

```bash
huggingface-cli login   # token with access to meta-llama repos
```
