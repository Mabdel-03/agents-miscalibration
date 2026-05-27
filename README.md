# agents_scaling

Follow-up to **Kim et al. 2025, "Towards a Science of Scaling Agent Systems"** (arXiv:2512.08296).

Kim et al. study multi-agent topologies using closed-API models and therefore cannot
inspect logits or measure **calibration**. This project reproduces their topology
framework with **open-weight models** (logits accessible via vLLM) and measures how three
scaling axes affect (a) performance, (b) efficiency, and (c) **miscalibration**:

1. **Model capacity** — parameter count (Qwen2.5-Instruct ladder, 0.5B → 72B).
2. **Context sharing between agents** — `ARTIFACT_ONLY` → `+INTERMEDIATE` → `+COT`.
3. **System-prompt complexity** — token count + a prompt-quality score (L0 → L3).

**Headline question:** does multi-agent coordination amplify or correct per-agent
miscalibration? (system-level ECE vs. mean per-agent ECE, across each axis).

## Topologies

`single_agent` (baseline), `independent` (majority vote), `decentralized` (all-to-all
debate), `centralized` (orchestrator + sub-agents). See `src/agents_scaling/agents/topologies/`.

## Layout

```
src/agents_scaling/   # the harness (pip install -e .)
  serving/            # vLLM server launch + LogprobClient
  agents/             # base agent, message_builder (axis 2), topologies
  prompts/            # prompt ladder (axis 3) + quality scoring
  benchmarks/         # GPQA / MMLU-Pro / MATH / TruthfulQA loaders + grading
  calibration/        # confidence signals + ECE/MCE/Brier
  efficiency/         # Kim coordination metrics (Ec, Ae, O%, c, R)
  experiment/         # runner, result schema, sweep generation
configs/              # pilot.yaml, full_sweep.yaml, models.yaml, prompts/
slurm/                # sbatch templates + sweep launcher
analysis/             # notebooks + scaling-law fitting
tests/                # unit tests
```

## Setup

See [env/SETUP.md](env/SETUP.md). Two conda envs: a `serve_env` (vLLM, owns its torch/CUDA)
and a lighter harness/analysis env (this package's deps).

## Quickstart

```bash
# 1. launch a vLLM server for one model size on SLURM
python slurm/launch_sweep.py --serve-only --models 1.5B

# 2. run the pilot once a server is up
python scripts/run_pilot.py --config configs/pilot.yaml

# 3. aggregate
python scripts/aggregate_results.py --run-id <run_id>
```

Model weights and results live on scratch
(`HF_HOME=/orcd/scratch/orcd/012/mabdel03/.cache/huggingface`), **not** the project dir.
