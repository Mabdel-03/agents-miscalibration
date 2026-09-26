# passk_cont_v1 — LEX addendum: a lexicographic accuracy → consistency → latency front (pre-registered 2026-09-25)

Owner objective (2026-09-25): choose the allocation that maximizes **(1) accuracy**, then among equally accurate
options **(2) consistency**, then among equally accurate and consistent options **(3) latency** (fastest). This
guards against the degenerate "always go deep" argument: a shallower, faster allocation wins whenever it is as
accurate and as consistent. Owner definitions:

- **Accuracy** of allocation *c* on problem *x*: p_x(c) = probability that one run of *c* delivers a correct answer
  (plurality over its *n* attempts, seeded uniform ties, i.e. the tie-averaged value; invalid = wrong).
- **Consistency** = correctness stability: the probability two independent runs of *c* on *x* are both right or both
  wrong, C_x(c) = p² + (1 − p)².
- **Latency** = parallel wall-clock: the *n* attempts run concurrently; latency = the slowest attempt's end-to-end
  time (THINK + ANSWER). Ties in latency → fewer generated tokens (n·a).
- **Ties**: an option is tied on accuracy if its accuracy is within **δ = 0.02** of the best option's; the same
  δ applies to consistency among the accuracy-tied options.
- **Menu**: every n×a with n·a ≤ 25A, n, a ∈ {1, 2, 4, 8, 16, 25}: 1×1, 1×2, 1×4, 1×8, 1×16, 1×25, 2×1, 2×2, 2×4,
  2×8, 4×1, 4×2, 4×4, 8×1, 8×2, 16×1, 25×1 (17 options).

## Data
- **New generation (owner-approved, ≈ $26):** partition `c48_25a` = 8 forced 25A draws (DEEP, r1–8) per C48 item
  (384 calls), exactly the study protocol (forced THINK 25,600 + ANSWER 4,096, provider pin, no seed, no cache),
  run through the unchanged `ForcedClient.generate` by `passk/lex.py` with a $32 cap; graded with the unchanged
  forced_v2 grader (Math-Verify 5 s, tri-state).
- **Per-problem banks (C48, 48 items, 12/benchmark):** the C48 ∪ V48 draws of each item, pooled per allowance:
  A: legacy slots 0–24 + V48 r0–15 (41); 2A: 16; 4A: 12; 8A: 10; 16A: 9; 25A: legacy r0 + `c48_25a` (9). All draws
  are independent forced draws of the same protocol (E0: identical THINK bodies; exchangeability is assumed and the
  collection era of each draw is recorded).
- **Estimation:** p̂_x(c) = expected delivered correctness of *n* draws sampled without replacement from the item's
  bank at allowance *a* (exact enumeration when C(m, n) ≤ 5,000, else 5,000 seeded random subsets, seed 20260925);
  answer classes per subset by exact vote key then the gold-free Math-Verify pair cache (mutual complete-link),
  GPQA by option identity; unknown grades are excluded from the item's bank (reported). Latency: the expected max of
  *n* call latencies drawn without replacement from the item's measured latencies at allowance *a*.

## Readouts
1. **Per-problem lexicographic winner** on C48 (in-sample), its distribution by benchmark and difficulty, and how
   often it is not the deepest/most expensive option.
2. **Split-half check** (guards against selecting on noise): per item and allowance, draws are split into two halves
   by a seeded hash; winners are chosen on half 1 and scored on half 2 (and vice versa); the realized accuracy,
   consistency and latency of the chosen options are reported against always-1×16 and always-1×25.
3. **Population-level ordering** of the 17 fixed options by the same lexicographic rule (mean accuracy, mean
   consistency, mean latency).
4. **Policy readout on T200 (exploratory; T200 labels already released):** the lexicographic rule applied per item
   to predicted per-configuration accuracy (the sealed P-model predictions, ≤16A menu available there), scored on the
   realized T200 outcomes and latencies against always-1×16.

Descriptive: no confirmatory hypothesis test. Uncertainty by the benchmark-stratified item bootstrap (10,000
replicates, seed 20260925).
