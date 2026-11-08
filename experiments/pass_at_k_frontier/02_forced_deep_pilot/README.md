# Study 2: `forced_deep_pilot_v1`

**Question.** Study 1's deep calls left most of their budget unspent. Can a single call be made to spend exactly `k*A`
reasoning tokens from one prompt, with no re-prompting?

**Run.** 2026-09-18. A small method pilot with no formal analysis, about $3.6 in total. Script:
[`agent_allocation/tools/forced_deep_pilot.py`](../../../agent_allocation/tools/forced_deep_pilot.py).

## Mechanism

Send a raw `/completions` request on the chat template ending in `<think>`, with `logit_bias` banning the end-of-reasoning
and end-of-text tokens, and `max_tokens = k*A`. Then re-send the prompt plus the reasoning and a closing `</think>` to get
the answer. The model sees one prompt and no injected text.

## Results

| Model and provider | Forced calls that hit budget | Accuracy at k = 1, 4, 16 | Natural accuracy | Cost |
|---|---|---|---|---|
| Qwen3.5-9B on Together | 96/96 | 0.781, 0.844, 0.906 | 0.688 | $0.34 |
| Qwen3.5-397B on Parasail | 96/96 | 0.906, 0.906, 0.938 | 0.938 | $3.29 |

- **Listing `logit_bias` does not mean a provider honours it.** DeepInfra lists it and ignored it, so the Qwen3-32B attempt
  there has no report. Test every new endpoint with a large bias on an arbitrary word before trusting it.
- **Forcing far past natural length degrades into looping.** On 397B at `k = 16`, 8 of 32 calls looped, with onset around
  7,000 tokens. Report effective depth beside nominal depth.
- 397B was at ceiling on these items, so a real depth test needed harder benchmarks. That is why study 3 changed them.

## Files

`qwen3.5-9b-together/REPORT.md` and `qwen3.5-397b-parasail/REPORT.md` hold the per-arm tables. The per-call `pilot.jsonl`
files contain prompts, so they are not on GitHub, see [`RAW_DATA.md`](../RAW_DATA.md).
