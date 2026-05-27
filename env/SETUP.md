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

Always point HuggingFace at scratch (the project dir has little free space):

```bash
export HF_HOME=/orcd/scratch/orcd/012/mabdel03/.cache/huggingface
mkdir -p "$HF_HOME"
```

`env/common.sh` (sourced by the sbatch templates) exports this for every job.

## 1. serve_env (vLLM)

```bash
module load miniforge/25.11.0-0
module load cuda/12.9.1
mamba create -y -n serve_env python=3.11
mamba activate serve_env
# Let vLLM pull its own matched torch/CUDA wheels:
pip install -r env/serve_env.txt
# sanity
python -c "import vllm; print('vllm', vllm.__version__)"
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

## 3. Gated weights (Llama only)

The Qwen2.5 ladder is ungated. If you run the optional Llama-3.1 robustness check:

```bash
huggingface-cli login   # token with access to meta-llama repos
```
