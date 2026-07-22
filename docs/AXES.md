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

Defined in [`models.py`](../src/agents_scaling/models.py). All are Apache-2.0 and ungated.
The 0.6B–14B standard profiles use 32,768 tokens on one A100-80GB; selective one-GPU
`-long` profiles use the native 40,960-token limit for dense, high-agent peer contexts.
The standard 32B profile uses 16,384 tokens on one GPU; coordinated 32B cells sharing
intermediate results or CoT use the TP=2 `32B-long` profile at 40,960 tokens.

---

## Axis 2 — Context sharing between agents

**Knob:** `context_share_level` ∈ `{ARTIFACT_ONLY, PLUS_INTERMEDIATE, PLUS_COT}` (ranks
0/1/2). **Measured:** exact shared-context tokens and per-block truncation markers.

This is the axis the open-weight choice most enables: closed APIs often hide intermediate
work and chain-of-thought. The eligible fields are nested — each higher level makes the
lower-level fields plus one additional field available before the registered safety cap:

- `ARTIFACT_ONLY` — a peer's final answer (and stated confidence) only.
- `PLUS_INTERMEDIATE` — + the peer's intermediate result / scratchpad / sub-conclusions.
- `PLUS_COT` — + the peer's raw chain-of-thought (or native thinking trace).

The **single** place this is applied is
[`message_builder.build_peer_context(peers, level)`](../src/agents_scaling/agents/message_builder.py).
Every topology that passes inter-agent messages routes through it, so the axis is applied
uniformly and is unit-tested (`tests/test_message_builder.py`). `single_agent` and
`independent` never share peer context, so the sweep collapses
this axis to a canonical value for them (no duplicate cells).

The protocol retains a 4,000-character cap on each CoT field and imposes a second,
tokenizer-aware cap of 4,000 tokens on each fully rendered peer block. The latter covers
the answer, confidence, intermediate result, CoT, and templates together. When it binds,
the builder appends an explicit marker containing the original block's hash and token
count; the consuming `AgentOutput` records exact context/block token counts, hash, and
marker count. Consequently, fields are nested by eligibility, but a cap-bound higher
level need not be a literal text superset of the lower rendered block.

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

| rung | `enable_thinking` | native thinking budget |
|------|-------------------|------------------------|
| OFF | false | — (plain instruct, no `<think>`) |
| B512 | true | 512 generated tokens |
| B2048 | true | 2048 generated tokens |
| B8192 | true | 8192 generated tokens |
| UNLIMITED | true | — (no native budget; 8192-token reasoning allowance) |

Set via [`config.ReasoningLevel`](../src/agents_scaling/config.py) →
[`LogprobClient.chat()`](../src/agents_scaling/serving/client.py). Every rung uses one
native vLLM 0.21 chat request. B512/B2048/B8192 pass the native
`thinking_token_budget`; OFF and UNLIMITED omit it. The request uses a deterministic
topology-derived seed, asks vLLM to return exact token IDs, and is preflighted against the
served context. A valid thinking result has exactly one generated `<think>` token at
output index zero and exactly one later `</think>` token; locally decoded reasoning and
answer text must equal vLLM's Qwen3 parser fields. The overall output allowance is 4096
answer tokens plus the finite budget, or 4096+8192 for UNLIMITED. Only a stopped response
with terminal `<|im_end|>` is accepted. The protocol specification and token-ID hashes
are stored with every result. Sampling remains Qwen3's recommendation (thinking:
T=0.6/top_p=0.95/top_k=20; non-thinking: T=0.7/top_p=0.8/top_k=20). Servers are pinned to
vLLM 0.21.0 with the Qwen3 reasoning parser enabled.

### Generation + ECE probe protocol (important)

Qwen3 thinking mode is documented **not** to use greedy decoding, which would contaminate
a logprob-based calibration measurement. So when thinking is on,
[`Agent.answer()`](../src/agents_scaling/agents/base_agent.py) separates generation from
confidence measurement:

1. a free **answer generation** — one native, token-audited chat request for every rung —
   captures the reasoning trace + answer, and
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
