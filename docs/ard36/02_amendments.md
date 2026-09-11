# ARD36 amendment register

Every deviation from `Agents_Round_Dynamics_36h_Spec.md`, each naming the claim it makes
unsupported. Frozen into `<run_root>/freeze/amendments.json` before dispatch; mid-run additions
are timestamped and justified, never silent. This is the study_v4 discipline that made its final
report auditable.

## A. Scope and tier

| id | deviation | spec clause | claim it limits |
|---|---|---|---|
| S1 | The 36-hour clock starts at implementation rather than at fleet allocation (user decision, 2026-09-10). §16.4 steps 1–3 are therefore applied **prospectively at T0**, before any test outcome exists: HUB5 dropped, E2d 32→16 items, challenge cap 96→48, receiver 64→32 programs, selector audit reduced to the root and GATE-final pools, extensions off. Reduced core 47,424 8B + 8,256 32B decisions. | §16.4, §16.5 line 666 | No HUB5 result. Challenge and receiver intervals are wider than the spec's power sketch. The full core is restored only if the launch-gate projection fits by hour 30. |
| S2 | Extensions 5–8 of the plan (team size N=3/9, cost-matched long single-agent baseline, heterogeneous teams, neural-augmented GATE) are not run. | §17 items 4, 5, 7, 9 | **No claim of superiority over optimised single-agent inference is made**, because §17 item 9's cost-matched baseline was not run. No post-interaction team-size effect. No cross-scale team claim. |
| S3 | The native BigCodeBench bridge, a second model family and activation patching are not run. | §5.5, §17 items 6, 8; §14.4 | No code-domain generality. Results are one model family, one sampling recipe, two task families. |
| S4 | The calibrated-confidence communication ablation is not run. | §6.4 line 262 | No evidence on whether transmitting peer confidence changes the dynamics; confidence is stripped from every base protocol. |
| S5 | Reported attribution / cited source ids are not collected: the frozen report schema has no citation field. | §10 line 355 | Influence is never inferred from answer matches. Only the randomised packet manipulations support causal language. |

## B. Measurement

| id | deviation | spec clause | claim it limits |
|---|---|---|---|
| M1 | ECE, MCE, reliability panels and the system-minus-agent miscalibration delta are reported although the specification never asks for them (the string ECE does not occur in its 880 lines). Added at the user's request; it was this repository's original headline quantity. | beyond §10, §15.2 | These are labelled exploratory additions, not spec-required outcomes, and carry the matched-n rule because valid-confidence counts vary by round. |
| M2 | 20,000 bootstrap resamples rather than the bundle's `bootstrap_resamples: 10000`. | bundle `run_plan.yaml` | None; more resamples, and it matches the freshness gate the report assembler already enforces. |
| M3 | The 256-token final allowance is enforced at parse time (the first 256 post-`</think>` tokens, with `final_overrun` recorded) rather than by a server-side cap. | §6.2 | Distributionally equivalent under autoregressive sampling; the surplus is wasted compute only, bounded by `max_tokens`. Recorded because it is a mechanism difference. |
| M4 | The noninferiority margin for "does not hurt correct majorities" is pre-registered at **δ = 8 points** on the constructed C0=4 challenge cohort, not on the natural C0∈{4,5} subgroup. | §15.3 line 575 | At the planned SE of 0.029 a 3-point margin could only be cleared by a superiority result. The natural subgroup is reported as estimation-only. |

## C. Data and grading

| id | deviation | spec clause | claim it limits |
|---|---|---|---|
| D1 | The "existing validated symbolic evaluator" does not exist in this repository. MATH grading uses **lark 1.3.1** driving `sympy.parsing.latex.parse_latex(backend='lark')` with a per-item timeout, pinned into the grader hash. | §5.1 line 186 | None, given D2: validated pre-freeze on all 500 pinned gold answers at parse rate 1.000, self-equivalence 1.000, and 500/500 agreement with each item's own worked solution. |
| D2 | Five MATH answer shapes are handled by hand-rolled rewrites outside the lark grammar: `\pi` (absent from sympy's greek grammar), leading-dot decimals, `x \in [-2,7]`, `pmatrix` with `&` separators, and a unit suffix after a fraction. Shapes still unsupported (`\sigma`, `\iota`, `\Pi`, other missing commands) are reported as unparsed, never guessed. | §5.1 line 171 | Ambiguous parses stay explicit as their own evaluator status and are reported separately, as the spec requires. |
| D3 | MMLU-Pro is restricted to the 9,981 ten-option rows, and 239 duplicate question strings are dropped (the plan estimated 231; the difference is stripping whitespace before de-duplicating). | §5.1 line 173 | The cohort is not a uniform sample of MMLU-Pro. Reported as a ten-option subset, never as a benchmark evaluation. |
| D4 | **Prior exposure:** earlier runs of this harness already used MMLU-Pro and MATH-500. | §18, study_v4 `prior_exposure_manifest.json` | No first-exposure claim. Contamination relative to prior harness runs is not controlled. |

## D. Implementation

| id | deviation | spec clause | claim it limits |
|---|---|---|---|
| I1 | The split budget is one chat request using vLLM's native `thinking_token_budget`, not two requests with an exact-token-prefix continuation. Verified: forcing is a logits boost, so the boundary token is genuinely sampled, `boundary_injected == (thinking_tokens == budget)` is exact, exact token ids are preserved trivially, and ≥256 final tokens always remain after a cap hit. | §6.2 line 232 | None. This is the "existing tested backend mechanism" the clause asks for. The two-phase path remains the documented fallback if the smoke fails. |
| I2 | `configs/model_contracts.v1.json` is unedited: it is sha256-pinned, its loader rejects any key set other than the ordered dense ladder, and both ARD36 profiles already resolve through it by model size to the pinned revisions. | §6.1 | None. |
| I3 | `study/types.py` `NAMESPACES`/`PURPOSES` are unextended; the ARD namespaces live in `ard/contract.py`. The whitelist is enforced in `study/policies/base.py::ctx.seed()`, which the ARD episode loop does not use. | §18 cache identity | None. |
| I4 | The no-gold firewall is absolute rather than per-function: no runtime import closure may reach `agents_scaling.benchmarks` at all. `ard/graders/_boxed.py` vendors the one pure extractor this cost, with a parity test against the original. | §18 line 693 | Strengthens the guarantee. The claim is code-level discipline plus a runtime scan, never filesystem isolation, since the HF dataset cache is world-readable. |
| I5 | `study/stats.py`, `study/selection/vote.py::vote()` and `study/packets.py`'s packet compiler are vendored or wrapped in `ard/` rather than imported, because each is welded to a frozen vocabulary that ARD36 changes. A regression test asserts the vendored statistics reproduce the original bit-for-bit on an (`hle`,`bcb`) fixture. | n/a (engineering) | None; the study package stays frozen and its 541 tests stay green. |
| I6 | Pre-freeze schema repair. The output contract gains one line: *"If your reasoning is cut short, still reply with only these three tags: commit to your best current answer instead of continuing to reason in the report."* The pilot (128 root draws, 8B) gave 83.6% valid reports against the 95% gate; all 21 truncations came from draws whose thinking hit the 1,024-token cap, while reports that closed reasoning naturally had a median of 77 tokens. Cut off mid-reasoning, the model continues reasoning inside the report and overruns the 256-token allowance. Applied uniformly to every item and every protocol; the pilot data generated before it are superseded and regenerated. | §6.3 lines 250-252 ("repair the schema/cap **before freezing the run**"; "Do not change the cap selectively on difficult test items") | None: the repair is the remedy the clause prescribes, and the allowances themselves are unchanged at 1,024 + 256. The cap-hit rate stays a reported outcome rather than something engineered away. |

## E. Mid-run additions

None yet. Each entry added after dispatch carries a timestamp, the gate that prompted it, and the
claim it limits.
