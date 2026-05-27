# agents_scaling

**Scaling multi-agent LLM systems: performance, efficiency, and miscalibration.**

A research harness that follows up on **Kim et al. 2025, "Towards a Science of Scaling
Agent Systems"** ([arXiv:2512.08296](https://arxiv.org/abs/2512.08296)). Kim et al. study
multi-agent topologies using closed-API models and therefore cannot inspect logits or
measure **calibration**. This project reproduces their topology framework with
**open-weight models** (Qwen3, logits accessible via vLLM) and measures how four scaling
axes affect (a) task **performance**, (b) **efficiency**, and (c) **miscalibration**.

> **Headline question:** does multi-agent coordination *amplify* or *correct* per-agent
> miscalibration? — operationalized as **system-level ECE vs. mean per-agent ECE**, across
> each scaling axis.

---

## The four scaling axes

| # | Axis | What varies | Knob in code |
|---|------|-------------|--------------|
| 1 | **Model capacity** | parameter count | Qwen3 ladder `0.6B…32B` ([models.py](src/agents_scaling/models.py)) |
| 2 | **Context sharing** | what an agent sees of peers: `artifact_only` → `+intermediate` → `+cot` | [`message_builder.py`](src/agents_scaling/agents/message_builder.py) |
| 3 | **System-prompt complexity** | prompt elaborateness `L0…L3` (+ token count & quality score) | [`prompts/`](src/agents_scaling/prompts/) |
| 4 | **Reasoning capacity** | thinking effort `off / 512 / 2048 / 8192 / unlimited` thinking-token budget | [`config.ReasoningLevel`](src/agents_scaling/config.py) + [`client.py`](src/agents_scaling/serving/client.py) |

Each axis is an independent, unit-tested knob; see [docs/AXES.md](docs/AXES.md) for the
full design of each and how it is *measured* (not just set).

## Topologies (from Kim et al.)

`single_agent` (baseline P_SA / T_SAS / E_SAS), `independent` (majority vote),
`decentralized` (all-to-all debate), `centralized` (orchestrator + sub-agents). See
[src/agents_scaling/agents/topologies/](src/agents_scaling/agents/topologies/).

## Benchmarks

Static QA with ground truth, chosen so calibration (ECE) is well-defined:
**GPQA-Diamond**, **MMLU-Pro**, **MATH-500**, **TruthfulQA-MC1**. See
[`benchmarks/loaders.py`](src/agents_scaling/benchmarks/loaders.py). (GPQA is gated on
HuggingFace — see [docs/OPERATIONS.md](docs/OPERATIONS.md#hugging-face-auth).)

## Metrics

- **Performance** — task accuracy / pass-rate.
- **Efficiency** — Kim's coordination metrics vs. the matched single-agent baseline:
  `Ec` (coordination efficiency), `Ae` (error amplification), `O%` (turn overhead), `c`
  (message density), `R` (redundancy). `T`=turns, `E`=error-rate (confirmed against the
  paper). See [`efficiency/metrics.py`](src/agents_scaling/efficiency/metrics.py).
- **Miscalibration** — ECE / MCE / Brier (Guo et al. 2017) from four confidence signals
  (option-logprob, verbalized, self-consistency, semantic-entropy), computed **per-agent
  AND system-level**. See [`calibration/`](src/agents_scaling/calibration/).

---

## Repository layout

```
src/agents_scaling/        # the harness (pip install -e .)
  config.py                # ExperimentCell + the axis enums (the single config schema)
  models.py                # Qwen3 model registry (size -> hf_id, tp_size, ...)
  serving/                 # vLLM server launch + multi-endpoint registry + LogprobClient
  agents/                  # base agent, message_builder (axis 2), 4 topologies, aggregate
  prompts/                 # prompt ladder (axis 3) + token-count + quality scoring
  benchmarks/              # GPQA / MMLU-Pro / MATH-500 / TruthfulQA loaders, grading
  calibration/             # confidence signals + ECE/MCE/Brier + per-agent vs system roll-up
  efficiency/              # Kim coordination metrics (Ec, Ae, O%, c, R)
  experiment/              # ExperimentCell runner, sweep generation, result schema, analyze
configs/                   # pilot.yaml, pilot_reasoning.yaml, full_sweep.yaml, prompts/
slurm/                     # sbatch templates + launchers (serve, sweep, chunked) + common.sh
scripts/                   # run_pilot.py, aggregate_results.py
analysis/                  # fitting.py (scaling-law fits) + notebooks
tests/                     # unit tests (pytest)
docs/                      # ARCHITECTURE, AXES, OPERATIONS (runbook), DATA_SCHEMA, RESULTS
```

Full module-by-module map: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart

```bash
# 0. one-time: create the two conda envs (see docs/OPERATIONS.md and env/SETUP.md)
#    - serve_env  : vLLM (owns its torch/CUDA)
#    - asys_env   : this package's harness/analysis deps (pip install -e ".[embeddings,dev]")

# 1. run the unit tests (no GPU needed)
PYTHONPATH=src python -m pytest tests/ -q

# 2. launch vLLM servers for a config's model sizes on SLURM
python slurm/launch_sweep.py --serve-only --config configs/pilot_reasoning.yaml --run-id mypilot

# 3. once servers register, run a pilot
python scripts/run_pilot.py --config configs/pilot_reasoning.yaml --run-id mypilot

# 4. aggregate -> tidy table
python scripts/aggregate_results.py --run-id mypilot --out analysis/mypilot.parquet
```

For the **full multi-cluster sweep** (servers + chunked cell arrays), see the runbook:
[docs/OPERATIONS.md](docs/OPERATIONS.md).

## Storage & environment

Model weights and all results live on **scratch**, never the small project dir:
`HF_HOME=/orcd/scratch/orcd/012/mabdel03/.cache/huggingface`,
`ASYS_RESULTS_ROOT=/orcd/scratch/orcd/012/mabdel03/agents_scaling_results`.
[`slurm/common.sh`](slurm/common.sh) exports these (and the HF token) for every job.

## Status

Harness built and validated on real A100s (reasoning two-call ECE confirmed; reasoning
tokens scale off→unlimited). The full 4-axis sweep (`configs/full_sweep.yaml`, ~11,520
cells) runs across `pi_tpoggio` + `ou_bcs` via the resumable chunked launcher. See
[docs/RESULTS.md](docs/RESULTS.md) for pilot findings and run status.

## License / citation

Research code. If you build on it, please cite Kim et al. 2025 (arXiv:2512.08296) and the
calibration methodology (Guo et al. 2017, arXiv:1706.04599).
