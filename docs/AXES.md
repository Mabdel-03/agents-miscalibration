# The four scaling axes

Each axis is a single, isolated knob recorded on `ExperimentCell` and varied in the sweep
cross-product. Crucially, for each axis we separate the **knob we set** from the
**attributes we measure**, so we can decouple correlated quantities (e.g. "more prompt
tokens" from "higher prompt quality", or "thinking on" from "how much it actually
thought").

---

## Axis 1 — Model capacity

**Knob:** `model_size` ∈ Qwen3 ladder. **Measured:** `param_count` (billions), the
regressor in the scaling-law fit.

One model family ([Qwen3 unified dense](https://huggingface.co/Qwen/Qwen3-8B)) so that
parameter count is the *only* thing varying — same architecture, tokenizer, and training
recipe across the ladder. Qwen3 was chosen over Qwen2.5 specifically because its unified
checkpoints support the reasoning toggle (Axis 4) on the *same weights*.

| size | hf_id | params (B) | tp |
|------|-------|-----------|----|
| 0.6B | Qwen/Qwen3-0.6B | 0.6 | 1 |
| 1.7B | Qwen/Qwen3-1.7B | 1.7 | 1 |
| 4B   | Qwen/Qwen3-4B   | 4.0 | 1 |
| 8B   | Qwen/Qwen3-8B   | 8.2 | 1 |
| 14B  | Qwen/Qwen3-14B  | 14.8 | 1 |
| 32B  | Qwen/Qwen3-32B  | 32.8 | 1 |

Defined in [`models.py`](../src/agents_scaling/models.py). All Apache-2.0, ungated, 32K
context, bf16 fits on a single A100-80GB.

---

## Axis 2 — Context sharing between agents

**Knob:** `context_share_level` ∈ `{ARTIFACT_ONLY, PLUS_INTERMEDIATE, PLUS_COT}` (ranks
0/1/2). **Measured:** shared-token counts (monotone in level by construction).

This is the axis the open-weight choice most enables: closed APIs often hide intermediate
work and chain-of-thought. The levels are **strictly nested** — each higher level exposes
everything the lower one does, plus more:

- `ARTIFACT_ONLY` — a peer's final answer (and stated confidence) only.
- `PLUS_INTERMEDIATE` — + the peer's intermediate result / scratchpad / sub-conclusions.
- `PLUS_COT` — + the peer's raw chain-of-thought (or native thinking trace).

The **single** place this is applied is
[`message_builder.build_peer_context(peers, level)`](../src/agents_scaling/agents/message_builder.py).
Every topology that passes inter-agent messages routes through it, so the axis is applied
uniformly and is unit-tested (`tests/test_message_builder.py`) including the monotonicity
property. `single_agent` and `independent` never share peer context, so the sweep collapses
this axis to a canonical value for them (no duplicate cells).

PLUS_COT context is truncated (`_COT_TRUNCATE_CHARS`) to guard against runaway context.

---

## Axis 3 — System-prompt complexity

**Knob:** `prompt_complexity_level` ∈ `{0,1,2,3}`. **Measured:** prompt token count + a
prompt-quality score (recorded in `meta.json`), decoupling "more tokens" from "better
prompt".

The ladder lives in [`configs/prompts/level{0..3}.txt`](../configs/prompts/):

- **L0** — minimal ("You are a helpful assistant. Answer the question.").
- **L1** — + expert role + answer-format hint (the "standard" level; SAS baselines use it).
- **L2** — + an explicit step-by-step strategy and elimination guidance.
- **L3** — + full strategy, calibration instruction, output format, and a worked example.

[`prompts/system_prompts.py`](../src/agents_scaling/prompts/system_prompts.py) loads the
ladder and computes `token_count(level, tokenizer)`.
[`prompts/prompt_quality.py`](../src/agents_scaling/prompts/prompt_quality.py) scores each
prompt two ways, both logged: a deterministic **heuristic** (readability, instruction
density, presence of role/format/examples/constraints) and an optional **LLM-judge** rubric
(0–100). Because length and quality are *measured attributes*, the analysis can ask whether
performance/ECE track tokens, quality, or both.

---

## Axis 4 — Reasoning capacity

**Knob:** `reasoning_level` ∈ `{OFF, B512, B2048, B8192, UNLIMITED}` (ranks 0–4).
**Measured:** mean reasoning ("thinking") tokens per question.

The open-weight analog of the closed-API `reasoning_effort` knob Kim et al. could not study
mechanistically. Implemented on Qwen3's unified hybrid-thinking models:

| rung | `enable_thinking` | `thinking_token_budget` |
|------|-------------------|-------------------------|
| OFF | false | — (plain instruct, no `<think>`) |
| B512 | true | 512 |
| B2048 | true | 2048 |
| B8192 | true | 8192 |
| UNLIMITED | true | none (bounded only by `max_tokens`) |

Set via [`config.ReasoningLevel`](../src/agents_scaling/config.py) →
[`LogprobClient.chat()`](../src/agents_scaling/serving/client.py), which passes
`extra_body={"chat_template_kwargs": {"enable_thinking": …}, "thinking_token_budget": …}`
and Qwen3's recommended sampling (thinking: T=0.6/top_p=0.95/top_k=20; non-thinking:
T=0.7/top_p=0.8/top_k=20). Served with `vllm serve … --reasoning-parser qwen3` (vLLM
≥0.9; our env has 0.21).

### Two-call ECE protocol (important)

Qwen3 thinking mode is documented **not** to use greedy decoding, which would contaminate
a logprob-based calibration measurement. So when thinking is on,
[`Agent.answer()`](../src/agents_scaling/agents/base_agent.py) makes **two calls**:

1. a free **thinking call** — captures the reasoning trace + the answer text, and
2. a **forced-answer probe** (`score_options`, completions endpoint, no thinking) — yields
   the clean option-letter logprob distribution used for ECE.

This decouples *reasoning depth* (Axis 4) from the *confidence measurement*, keeping
calibration on the exact MCQ-logprob path validated in the non-reasoning pilot. The OFF
rung is a single call (the original behavior).

> **Implementation note discovered live:** vLLM 0.21's `qwen3` parser surfaces the thinking
> trace in a message field named **`reasoning`** (not `reasoning_content`, and not as a
> typed OpenAI attribute). The client checks both names plus the raw `model_dump()`.

### Cost note

The reasoning axis multiplies the sweep ~5× and the budget/unlimited rungs are
substantially slower (long traces; the two-call protocol adds a call). `analyze.py` records
`mean_reasoning_tokens` per cell so cost vs. benefit is quantifiable. See
[OPERATIONS.md](OPERATIONS.md) for the wall-clock projection.

---

## How the axes combine in the sweep

`sweep.generate_cells()` takes the cross-product of all four axes × topology × benchmark ×
seed, then:

- **collapses** axis values a topology ignores (context-share for `single_agent` /
  `independent`), deduping by `cell_id`;
- **does not collapse** reasoning (it is per-agent; every topology can think);
- **enforces a single-agent baseline** for every `(model_size, benchmark, seed,
  reasoning_level)` — efficiency ratios (`Ec/Ae/O%`) are reasoning-conditioned, so the
  baseline is too.

See `tests/test_sweep_cardinality.py` and `tests/test_reasoning_level.py` for the invariants.
