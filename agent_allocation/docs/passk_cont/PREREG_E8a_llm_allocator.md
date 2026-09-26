# passk_cont_v1 — E8a addendum: an LLM allocator as a comparison (pre-registered 2026-09-25, before any allocator call)

Owner request (2026-09-25, after the core study completed): add an LLM that selects the allocation, as a comparison
to the fixed rules and the statistical selectors. This is a reduced spec §13 (E8) extension: a single-decision
supervisor that chooses one action from the frozen B = 16A menu. It is not an orchestrator harness (that was
dynorch_v1) and not a candidate verifier (E4).

**Disclosure.** The T200 outcomes of all five allocations were already scored and seen by the analyst when this
addendum was written. The allocator prompt below is written once from the menu definition, is not tuned, and is not
revised after any allocator output (L96 or T200) is seen. No allocator output exists at commit time.

## Arms
- **A0 (problem only):** the allocator sees the problem and the menu.
- **A1 (problem + diagnostics):** additionally the final answers of the item's five independent A-level diagnostic
  draws (L96: legacy slots 20–24; T200: `t200_diag` r0–4), answers only, in replicate order; an invalid diagnostic
  is shown as `(no valid final answer)`. No correctness information. This matches P1's information set.
- Allocator model/protocol: the study's own (Qwen3.8-27B, same provider pin as the item, forced THINK exactly
  1,024 tokens + ANSWER headroom 4,096, T 1.0 / top_p 0.95 / top_k 20, no seed, no cache), through the unchanged
  `ForcedClient.generate`. Prompt text: the `SYSTEM` / `USER_*` constants in `src/agent_allocation/passk/e8a.py` (frozen at this commit; their sha256 is recorded with every run).
- Menu: `1x16, 2x8, 4x4, 8x2, 16x1` (n × a/A, plurality over attempts, seeded uniform ties). Presentation order is a
  seeded per-item permutation, salt `passk_cont_v1/e8a/order|seed=20260925|<arm>|<item_id>`.
- Parse: exactly one `<final>…</final>` in the ANSWER text; its content lowercased, spaces removed, `×` → `x`, must
  equal a menu label. Otherwise invalid → one outcome-blind retry (a new call) → still invalid → **fallback
  `1x16`** (primary) and, as a sensitivity, delivered = 0.

## Scoring (no new solver generation)
The chosen configuration's realized delivered correctness is read from the sealed/released readouts
(`analysis_inputs/readouts.parquet`, mode `passk`, seeded tie rule, `S_known`; unknown stays unknown). Panels:
**L96 (development, descriptive)** and **T200 (test)**.

Budget labels (logical cost): A0 = 16A + 1A allocator reasoning (+ its prompt/answer tokens) = a 17A policy;
A1 = 16A + 5A diagnostics + 1A = a 22A policy. Always-deep is a 16A policy. Costs are reported beside accuracy.

## Tests (separate family from the core primary family)
Holm over two, family-wise α = 0.05, T200 mixture estimand (0.10/0.10/0.50/0.30), benchmark-stratified paired
item bootstrap, 10,000 replicates, seed 20260925, two-sided:
1. A0 − always-`1x16` delivered accuracy.
2. A1 − always-`1x16` delivered accuracy.

Secondary (descriptive, intervals only): A1 − A0; each arm − the P1 per-item selection policy (`scores/e5`); each arm −
uniform-random choice (per-item mean over the five configurations); each arm − the realized per-item best
configuration (a clairvoyant ceiling, diagnostic only); choice distribution by benchmark; invalid rate; macro
estimand; the invalid = 0 sensitivity; L96 results.

## Spend
Extension cap **$8** (inside the study's $150; ~$96 spent). About 600 allocator calls at ~$0.004–0.005.
