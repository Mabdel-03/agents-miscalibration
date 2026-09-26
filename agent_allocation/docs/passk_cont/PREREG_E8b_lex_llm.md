# passk_cont_v1 — E8b addendum: LLM selectors told the lexicographic objective (pre-registered 2026-09-25, before any E8b call)

Owner request (2026-09-25): compare statistical and LLM selectors under the owner's lexicographic criteria
(`PREREG_LEX_front.md`: accuracy → correctness stability p² + (1 − p)² → parallel wall-clock latency; ties within
0.02). E8a's allocators were asked for accuracy only over 5 options, which tilts that comparison against the LLM.
E8b gives the LLM the same objective and the same menu as the statistical selectors (owner-approved, ≈ $9).

**Disclosure.** All T200 and C48 outcomes were already seen by the analyst. The prompt (constants in
`src/agent_allocation/passk/e8b.py`, sha256 recorded per call) is written once and is not revised after any E8b
output is seen.

## Arms and protocol
- **B0**: the problem, the menu and the objective. **B1**: additionally the five diagnostic answers (L96 slots
  20–24 for C48 items; `t200_diag` for T200), answers only, as in E8a.
- Menu M15 (every n×a ≤ 16A): 1×1, 2×1, 4×1, 8×1, 16×1, 1×2, 2×2, 4×2, 8×2, 1×4, 2×4, 4×4, 1×8, 2×8, 1×16, each shown
  with its typical parallel wall-clock (C48 population mean, rounded to 10 s; population-level information that
  the statistical selectors' latency table also uses).
- Objective text: (1) highest probability the delivered answer is correct, options within 2 percentage points
  count as equally accurate; (2) among those, correctness most stable across repeated runs (probability near 0 or
  1); (3) among those, the fastest.
- Same model and forced protocol as E8a (THINK exactly 1,024 + ANSWER 4,096, provider pin, no seed, no cache).
  **K = 5 independent decisions per problem and arm**, option order reshuffled per decision (seeded salt
  `passk_cont_v1/e8b/order|seed=20260925|<arm>|<item>|<sample>`). Invalid → one retry → still invalid → `1x16`.
- Panels: C48 (48) and T200 (200). ≈ 2,480 decisions; cap $14.

## Scoring (identical to the statistical selectors in `passk/selcmp.py`)
- **C48 (bank-scored):** each decision's option is scored with the item's bank estimates (p, latency); the system's
  per-item accuracy is the mean over its 5 decisions (the selector's own randomness included); consistency per
  item = p_sys² + (1 − p_sys)²; latency = mean over decisions of (option wall-clock + the decision call's latency,
  + the diagnostics' parallel wall-clock for B1).
- **T200 (realized):** each decision's option is scored with the realized delivered correctness and realized
  wall-clock of that option's pool (+ the same overheads), averaged over the 5 decisions.
- Population lexicographic ranking against the statistical selectors, fixed options and E8a (accuracy within 0.02
  of the best → consistency within 0.02 → fastest), macro (C48) and mixture/macro (T200), with bootstrap intervals.
  Also reported: choice distributions and choice stability (share of the modal choice across the 5 decisions).

Descriptive comparison; no confirmatory test.
