# agents_scaling

**Scaling multi-agent LLM systems: performance, efficiency, and miscalibration.**

This repository is a research harness for asking whether multi-agent coordination
amplifies or corrects the miscalibration of the agents inside the system. It follows the
topology framing of Kim et al., "Towards a Science of Scaling Agent Systems"
([arXiv:2512.08296](https://arxiv.org/abs/2512.08296)), but uses open-weight Qwen3 models
served with vLLM so logits, option probabilities, reasoning traces, and token costs are
observable.

Start with [docs/EXPERIMENT.md](docs/EXPERIMENT.md) for the research questions,
experimental design, method tables, and diagrams.

## Headline Question

Does multi-agent coordination amplify or correct per-agent miscalibration?

The primary comparison is:

```text
Delta ECE = system-level ECE - mean per-agent ECE
```

Positive Delta ECE means the system's final confidence is less calibrated than its
component agents; negative Delta ECE means coordination improved calibration.

## What Varies

| Axis | Values | Where implemented |
|---|---|---|
| Model capacity | Qwen3 dense `0.6B` through `32B` | [`models.py`](src/agents_scaling/models.py) |
| Context sharing | `artifact_only`, `plus_intermediate`, `plus_cot` | [`message_builder.py`](src/agents_scaling/agents/message_builder.py) |
| Prompt complexity | Prompt levels `L0..L3` | [`configs/prompts/`](configs/prompts/) |
| Reasoning capacity | `off`, `b512`, `b2048`, `b8192`, `unlimited` | [`config.py`](src/agents_scaling/config.py), [`client.py`](src/agents_scaling/serving/client.py) |

Implemented topologies are `single_agent`, `independent`, `decentralized`, and
`centralized`. Kim et al.'s Hybrid topology is not implemented in this harness.

## What Is Measured

- **Performance:** task accuracy/pass rate.
- **Efficiency:** Kim-style coordination metrics relative to matched single-agent
  baselines: `Ec`, `Ae`, `O%`, `c`, and `R`.
- **Calibration:** ECE/MCE/Brier for per-agent confidence and multiple system-confidence
  definitions, including final-producer confidence.

Benchmarks are GPQA-Diamond, MMLU-Pro, MATH-500, and TruthfulQA-MC1. MCQ benchmarks support
direct option-logprob ECE; MATH is numeric/free-form and should be interpreted with the
available non-MCQ confidence signals.

## Documentation Map

- [docs/EXPERIMENT.md](docs/EXPERIMENT.md): research-reader dossier with objectives,
  methods, diagrams, and interpretation guide.
- [docs/AXES.md](docs/AXES.md): detailed scaling-axis reference.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): code and data-flow map.
- [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md): exact `results.jsonl`, `meta.json`, and
  parquet fields.
- [docs/OPERATIONS.md](docs/OPERATIONS.md): SLURM, environments, launch/resume, and
  monitoring runbook.
- [docs/RESULTS.md](docs/RESULTS.md): validation milestones, pilot results, and live run
  status.
- [analysis/README.md](analysis/README.md): aggregation, example scripts, and planned
  notebooks.

## Repository Layout

```text
src/agents_scaling/        # harness package
  config.py                # ExperimentCell and axis enums
  models.py                # Qwen3 model registry
  serving/                 # vLLM launch, registry, logprob client
  agents/                  # agents, aggregation, message builder, topologies
  prompts/                 # prompt loading and prompt-quality scoring
  benchmarks/              # normalized benchmark loaders and grading
  calibration/             # confidence signals and ECE/MCE/Brier
  efficiency/              # coordination metrics
  experiment/              # runner, sweep generation, aggregation schema
configs/                   # pilot and full-sweep YAML specs
docs/                      # research dossier, architecture, operations, schema, results
analysis/                  # fitting utilities and runnable examples
scripts/                   # pilot and aggregation entrypoints
slurm/                     # server/cell launchers and sbatch templates
tests/                     # unit tests
```

## Quickstart

```bash
# 1. run tests
PYTHONPATH=src python -m pytest tests/ -q

# 2. inspect the planned full sweep
PYTHONPATH=src python analysis/examples/inspect_sweep.py --config configs/full_sweep.yaml

# 3. launch servers for a pilot config
python slurm/launch_sweep.py --serve-only --config configs/pilot_reasoning.yaml --run-id mypilot

# 4. run the pilot after servers register
python scripts/run_pilot.py --config configs/pilot_reasoning.yaml --run-id mypilot

# 5. aggregate completed cells
python scripts/aggregate_results.py --run-id mypilot --out analysis/mypilot.parquet
PYTHONPATH=src python analysis/examples/summarize_run.py --parquet analysis/mypilot.parquet
```

The full multi-cluster sweep is intentionally large and resumable. The authoritative
schema-5 recovery and production path is
[docs/SCHEMA5_RECOVERY_RUNBOOK.md](docs/SCHEMA5_RECOVERY_RUNBOOK.md). The older
[docs/OPERATIONS.md](docs/OPERATIONS.md) procedures are retained only for development and
legacy-run forensics.

## Storage And Environment

Production paths are immutable pins, not shell defaults. For the current schema-5 release
they are derived from `RELEASE_COMPLETE.json` and `.dispatcher-schema5-v1/control.json`.
The current development defaults use the data filesystem because the scratch quota
previously returned `EDQUOT` during model/dataset lock writes:

```text
HF_HOME=/orcd/data/tpoggio/001/mabdel03/.cache/huggingface
ASYS_RESULTS_ROOT=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results
```

See [env/SETUP.md](env/SETUP.md) for the two-environment setup: one vLLM serving env and
one lighter harness/analysis env.

## Citation

If you build on this code, cite Kim et al. for the agent-systems scaling framework and
Guo et al. 2017 for calibration methodology:

- Kim et al., "Towards a Science of Scaling Agent Systems", arXiv:2512.08296.
- Guo et al., "On Calibration of Modern Neural Networks", arXiv:1706.04599.
