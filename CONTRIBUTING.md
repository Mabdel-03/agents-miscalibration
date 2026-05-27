# Contributing / development

## Setup

```bash
# harness/analysis env (no GPU needed for dev + tests)
pip install -e ".[embeddings,dev]"
PYTHONPATH=src python -m pytest tests/ -q
```

For serving/inference work you also need the `serve_env` (vLLM) and a GPU — see
[docs/OPERATIONS.md](docs/OPERATIONS.md).

## Tests (run on every change)

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

- `test_calibration_metrics.py` — ECE/MCE/Brier on synthetic data with known properties.
- `test_message_builder.py` — Axis-2 context levels are strictly nested + monotonic.
- `test_grading.py` — MCQ letter extraction + numeric boxed-answer matching.
- `test_sweep_cardinality.py` — sweep collapsing, SAS-baseline enforcement, dedup, 4-axis.
- `test_reasoning_level.py` — `ReasoningLevel` rank/budget mapping, cell_id, round-trip.

The tests run **without a GPU**. For wiring confidence without a server, exercise the
topologies/runner with a fake client that returns canned `ChatResult`/`OptionScores`
(see the dry-run snippets referenced in commit history).

## Invariants to preserve

1. **`ExperimentCell` is the single config schema.** New knobs go here, are coerced in
   `__post_init__`, appear in `cell_id`, and round-trip through `to_dict`/`from_dict`.
2. **One axis = one knob in one place.** Don't scatter an axis across modules. Follow the
   existing trace (see [docs/AXES.md](docs/AXES.md) and the reasoning-axis commit as the
   template for adding a 5th axis).
3. **Calibration stays on the validated path.** Confidence for ECE comes from the
   forced-answer `score_options` probe (option-logprobs), not from free-form generation —
   especially under non-greedy sampling (thinking mode).
4. **Resumability.** Anything that writes results must be append-only and skippable by
   `qid`/`meta.json`, so SLURM kills/preemption never corrupt or duplicate data.
5. **Servers are discovered, never hardcoded.** Use the registry (`wait_for_server`); a
   size may have multiple endpoints.

## Adding a new axis (recipe)

Mirror `prompt_complexity_level` / `reasoning_level` end-to-end:
`config.py` (field + enum + cell_id + coercion) → `sweep.py` (product + any collapsing +
baseline keying) → consume at runtime (`runner.py`/`base_agent.py`/`client.py`) →
`result_schema.py` + `analyze.py` (column + measured attribute) → `configs/*.yaml` →
tests (`test_sweep_cardinality.py` + an enum-mapping test).

## Adding a benchmark

Add a loader to `benchmarks/loaders.py` returning normalized `Question`s, register it in
`LOADERS`, and confirm grading in `benchmarks/grading.py`. Prefer parquet-based HF datasets
(dataset scripts are disabled on the Hub).

## Commit / push conventions

- Work on a feature branch (the harness lives on `harness-scaffold`); don't commit to the
  default branch directly.
- Keep `results/`, `*.parquet`, `*.egg-info/`, `.pytest_cache/`, and SLURM logs out of git
  (see `.gitignore`). Results live on scratch.
- Co-author trailer: `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`.
