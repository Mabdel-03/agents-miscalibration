# Architecture

This document maps the harness module-by-module and traces the end-to-end data flow of a
single experiment cell, from config to result to analysis.

## Design principles

1. **Each scaling axis is an isolated, testable knob.** Capacity lives in the model
   registry/serving; context-sharing in one function (`build_peer_context`); prompt
   complexity in `prompts/`; reasoning in the client + `ReasoningLevel`. Adding the 4th
   axis (reasoning) touched the same files in the same way as the 3rd, by design.
2. **The `ExperimentCell` is the single source of truth.** One frozen dataclass fully
   describes a run; it serializes to/from JSON so a SLURM array task reconstructs it by
   index, and every result row carries the exact config that produced it.
3. **Calibration is first-class.** Confidence is captured at generation time (logprobs)
   and rolled up both per-agent and at the system level under multiple definitions, so
   the headline conclusion is robust to how "system confidence" is defined.
4. **Everything is resumable and file-based.** Servers publish endpoints to a filesystem
   registry; cells append JSONL and skip already-done work; no external services.

## The execution model

```
            ┌─────────────────────────── SLURM ───────────────────────────┐
            │                                                              │
  vLLM servers (GPU)                         cell workers (CPU, HTTP clients)
  one per model size,                        chunked arrays, throttled,
  across pi_tpoggio + ou_bcs                 round-robin across endpoints
            │                                                              │
            ▼                                                              ▼
  servers/<size>/<host>_<port>.json  ◀──── registry (shared scratch FS) ────▶  wait_for_server()
            │                                                              │
            └──────────────► OpenAI-compatible /v1 endpoint ◀──────────────┘
                                       │
                              LogprobClient (always logprobs;
                              enable_thinking + thinking_budget)
```

- **Servers** are long-lived SLURM jobs (`slurm/serve_qwen.sbatch.tmpl`). Each runs
  `vllm serve <hf_id> --reasoning-parser qwen3 ...`, then registers its `node:port` to the
  registry once healthy. A size may have **multiple** endpoints (different clusters).
- **Cell workers** are CPU-only array tasks. Each loads one `ExperimentCell` from
  `cells.json` by `$SLURM_ARRAY_TASK_ID`, discovers a server endpoint (round-robined by
  its index), runs the topology over the benchmark, and appends results.
- Communication is plain HTTP to the vLLM OpenAI-compatible API. The shared scratch
  filesystem is the only coordination layer.

## Module map

### Core config — `config.py`
- `ExperimentCell` (frozen dataclass): the four axes (`model_size`,
  `context_share_level`, `prompt_complexity_level`, `reasoning_level`) + topology,
  benchmark, and fixed knobs (`n_agents`, `rounds`, `n_samples`, `temperature`, `seed`).
  `cell_id` is a stable, collision-free string; `to_dict`/`from_dict` round-trip via YAML
  strings; `config_hash` for provenance.
- Axis enums: `ContextShareLevel` (rank 0–2), `ReasoningLevel` (rank 0–4 with
  `enable_thinking` + `thinking_budget` properties), `Topology` (with `is_multi_agent` /
  `shares_context`).

### Model registry — `models.py`
- `ModelSpec(size, hf_id, param_count, tp_size, max_model_len, supports_reasoning)`.
- `QWEN3_LADDER`: `0.6B / 1.7B / 4B / 8B / 14B / 32B` (Apache-2.0, ungated, unified
  hybrid-thinking). The dense profiles through 14B use one-GPU 32,768-token standard
  profiles plus selective one-GPU 40,960-token long profiles; 32B uses a one-GPU
  16,384-token standard profile plus a TP=2, 40,960-token long profile.
  Optional `QWEN3_MOE` (30B-A3B).
- `param_count` (billions) is the capacity regressor in the scaling-law fit.

### Serving — `serving/`
- `client.py` — **`LogprobClient`**, the calibration linchpin.
  - `chat()` uses one native vLLM 0.21 chat generation for every reasoning rung and
    normally requests `logprobs`+`top_logprobs`. Finite rungs pass
    `thinking_token_budget`; OFF and UNLIMITED omit it. The request asks vLLM for exact
    prompt/completion token IDs, checks them against the locally rendered chat template,
    requires one generated Qwen think-start at completion index zero and uses the first
    generated think-end as the parser boundary. Later think-end IDs are retained as
    literal answer content; exact first-span/parser-local-decode agreement is required.
    A thinking prompt is rejected locally only if it ends in unmatched think-start state.
    The client accepts only `finish_reason=stop` with terminal `<|im_end|>`. Protocol version/hash,
    seed, profile/context limit, exact token counts, delimiter indexes, and token-ID
    hashes are persisted with every new agent result.
  - `score_options()` — the **forced-answer probe**: completions endpoint, prompt ending
    `"Answer: "`, `max_tokens=1`, reads the option-letter logprob distribution and
    renormalizes over the options. This is the clean confidence signal for ECE.
- `registry.py` — file-based, **multi-endpoint**. Each server writes
  `servers/<size>/<host>_<port>.json`; `lookup_server(shard=…)` round-robins; back-compat
  reads a legacy single `servers/<size>.json`.
- `healthcheck.py` — polls `/health` then `/v1/models` until the model is loaded.
- `launch_server.py` — renders `serve_qwen.sbatch.tmpl` for a size and `sbatch`es it; the
  `--register` role runs inside the job to healthcheck + publish the endpoint.

### Agents & topologies — `agents/`
- `base_agent.py` — `Agent` (client + system prompt + reasoning level) and `AgentOutput`
  (answer, raw text, CoT, intermediate results, option logprobs, verbalized conf, token
  counts, reasoning text/tokens). `answer()` obtains one complete native chat generation
  for every rung, then runs the independent `score_options` probe for the ECE
  distribution. Calibration is therefore decoupled from non-greedy thinking sampling.
  Coordinated outputs also carry an exact hash,
  token count, per-peer block counts, and explicit truncation-marker count for the peer
  context they consumed.
- `message_builder.py` — the Axis-2 linchpin and protocol-v2 peer renderer. Eligible
  fields follow the context
  level; each CoT field is capped at 4,000 characters and each complete peer block is
  capped at 4,000 exact serving-profile tokens with a hash-covered truncation marker.
- `topologies/` — `single_agent`, `independent`, `decentralized` (all-to-all debate over
  rounds), `centralized` (orchestrator + sub-agents). All multi-agent topologies route
  inter-agent messages through `build_peer_context`. `build_topology(...)` is the factory.
- `aggregate.py` — `majority_vote` + `system_confidences` (multiple system-confidence
  definitions: vote fraction, mean agreeing logprob/verbal, orchestrator logprob).

### Benchmarks — `benchmarks/`
- `schema.py` — uniform `Question{qid, benchmark, prompt_stem, options, answer_key,
  answer_type}` (`MCQ` | `NUMERIC`).
- `loaders.py` — normalizes GPQA / MMLU-Pro / MATH-500 / TruthfulQA into `Question`s
  (deterministic option shuffling per seed).
- `formatting.py` — renders MCQ prompts + the `score_prompt` ending in `"Answer: "`.
- `grading.py` — letter extraction (MCQ) and boxed-answer numeric matching (MATH).

### Calibration — `calibration/`
- `signals.py` — four confidence signals → p(chosen)∈[0,1].
- `semantic_entropy.py` — tiered meaning clustering for free-form answers.
- `metrics.py` — `compute_calibration` → ECE (15 equal-width bins, Guo 2017), MCE, Brier,
  reliability bins.

### Efficiency — `efficiency/`
- `metrics.py` — `coordination_metrics(...)` computes Kim's `Ec, Ae, O%, c` vs. the
  matched single-agent baseline (turn-based; token variants logged alongside).
- `embeddings.py` — redundancy `R` (mean pairwise cosine similarity of agent outputs).

### Experiment orchestration — `experiment/`
- `result_schema.py` — `QuestionResult` (one JSONL row/question) + `CellMeta` (per-cell
  config, prompt token count, prompt quality, mean reasoning tokens, git commit).
- `runner.py` — `run_cell()`: builds agents (threading all four axes), runs the topology
  over the benchmark, writes JSONL + meta. **Resumable**: skips a cell with `meta.json`,
  and skips already-answered qids in a partial `results.jsonl`.
- `run_one.py` — SLURM array entrypoint: runs `cells[$SLURM_ARRAY_TASK_ID]`, passing the
  index as the round-robin shard.
- `sweep.py` — `generate_cells(spec)`: cross-product of the axes with **validity
  collapsing** (single-agent/independent collapse context-share; reasoning is per-agent so
  it is never collapsed) and **per-`(size, benchmark, seed, reasoning_level)` SAS-baseline
  enforcement**. Deterministic, deduplicated by `cell_id`.
- `analyze.py` — walks `cells/*/{results.jsonl,meta.json}` → tidy per-cell records:
  accuracy, per-agent & system ECE (per confidence definition), Kim efficiency vs. the
  matched baseline, mean reasoning tokens.

### SLURM — `slurm/`
- `common.sh` — sourced by every job: modules, conda env prefixes, `HF_HOME`,
  `LD_LIBRARY_PATH` (newer libstdc++ for flashinfer), and HF token loading.
- `serve_qwen.sbatch.tmpl` / `run_cell_array.sbatch.tmpl` — job templates.
- `launch_server.py` (module) — launch one server.
- `launch_sweep.py` — launch all servers for a config + (optionally) one cell array.
- `launch_chunked.py` — submit the cell list as dependency-chained array **chunks** of
  ≤480 (works around `MaxSubmitJobs=500`); resumable.

## End-to-end data flow (one cell)

```
configs/*.yaml
   │  sweep.generate_cells()            (cross-product + collapse + SAS baselines)
   ▼
cells.json  ──(SLURM array index)──►  run_one.py  ──►  runner.run_cell(cell, shard)
                                                          │
                          registry.wait_for_server(size, shard)  ──►  LogprobClient
                                                          │
                          build_topology(cell.topology, agents, ctx, rounds)
                              agents carry cell.reasoning_level (two-call ECE)
                              and the L{prompt} system prompt
                                                          │
                          topo.run(question)  →  TopologyResult(final, per_agent, turns…)
                                                          │
                          QuestionResult  ──►  results.jsonl   (one row per question)
                          CellMeta        ──►  meta.json       (config + measured attrs)
   ▼
analyze.aggregate_run(run_id)  ──►  tidy per-cell metrics  ──►  analysis/*.parquet
   ▼
analysis/fitting.py            ──►  scaling-law fits, reliability diagrams, headline ΔECE
```

See [DATA_SCHEMA.md](DATA_SCHEMA.md) for the exact JSONL/meta fields.
