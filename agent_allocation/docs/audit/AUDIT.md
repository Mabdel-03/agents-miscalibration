# AUDIT: openrouter_agent_allocation_v1, independent QA (agent 08)

Status as of **M1** (2026-09-17 03:20 EDT, branch `alloc-wp08`). No production module other than the lead's WP0
files is merged into this worktree yet. Every production check below is written against the frozen signatures and
skips until its module lands.

**Release recommendation: DO NOT DISPATCH.** Gate 0 cannot pass: 0 of 20 section 15 checks have run against
production code, and 11 contract/config/path defects are open (D1–D11, `docs/requests/08.md`). Pause rather than
ship a fast invalid result.

Command: `source slurm/env.sh && uv run --frozen --no-sync pytest -q tests/integration`
Result at M1: 58 passed, 38 skipped (all "waiting on module"), 11 xfailed (strict; each is an open defect).

## Section 15 checks

| # | Check | Reference level (runs now) | Production (module it waits on) |
|---|---|---|---|
| 1 | Same-prompt draws are distinct opportunities; shared roots not regenerated | PASSED | not yet runnable: `protocols.grid`, `prompts.render`, `api.request`, `api.client` |
| 2 | Roots record cannot overwrite private/peer decisions under one episode label | PASSED (ARD36 collision recreated) | not yet runnable: `protocols.grid`, `storage.state`, `storage.raw` |
| 3 | 125 / 123 / 57 per item; CP +12, POOLED +1 | PASSED (3 independent counts agree) | not yet runnable: `protocols.grid` (node-by-node id comparison with the reference builder) |
| 4 | PRIVATE_SINGLE t=1 and team r=0 reference the roots; 25-bank is no team archive | PASSED | not yet runnable: `protocols.grid` |
| 5 | PEER 5x4 has 25 ancestors; no future/same-round visibility | PASSED | not yet runnable: `protocols.policy_cost`, `prompts.render` |
| 6 | PRIVATE_TEAM reuses across n; PEER does not | PASSED | not yet runnable: `protocols.grid` |
| 7 | Usage charged once in collection, once per policy occurrence; no reasoning double count | PASSED | not yet runnable: `protocols.policy_cost` |
| 8 | Reservations never exceed ceiling; timeout/restart/late bill/429 retain money | PASSED (checker vs adversarial streams) | not yet runnable: `storage.journal`, `runtime.ledger`; **blocked by D2** for the restart split |
| 9 | Fixed population across checkpoints; unknown/unrun not dropped | PASSED | not yet runnable: `metrics` (pass@k exactness), `analysis` (population, M2) |
| 10 | Malformed output not retried; retries bounded and metered | PASSED | not yet runnable: `api.client`, `extraction.final`; **open D1** (ERROR_ENVELOPE) |
| 11 | Selector cannot see private fields; canaries absent from requests/raw/logs | PASSED | not yet runnable: `selection.readout`, `prompts.render`, `api.request`, `api.client`, `storage.raw` |
| 12 | MBPP public selection isolated; sandbox contains attacks | PASSED (public/private split of fixtures) | not yet runnable: `grading.code.public` (`sandbox` marker) |
| 13 | Answer classes gold-blind; no one-direction MATH merges | PASSED | not yet runnable: `selection.readout` |
| 14 | Unknown stays unknown; invalid vs wrong distinguishable | PASSED; **open D3** | not yet runnable: `grading.text.grade` |
| 15 | Missing reasoning usage is null; cache HIT blocks partition | PASSED | not yet runnable: `api.client` (+ dispatcher partition block, M2) |
| 16 | Larger budget not reported as more realized reasoning | PASSED | not yet runnable: `analysis`, `reporting.cli` |
| 17 | Selectors have real rows; all-null is a blocker, not 0% | PASSED | not yet runnable: `selection.seal`, `selection.readout` |
| 18 | Synthesis may exceed coverage; candidate-only may not | PASSED | not yet runnable: `metrics` |
| 19 | Peer order and root ids independent of completion time | PASSED | not yet runnable: `protocols.grid`, `prompts.render` |
| 20 | Resume idempotent; raw artifacts and failures kept | PASSED (planner) | not yet runnable: `storage.state`, `storage.raw` (+ dispatcher resume, M2) |

## Independent recomputations (tests/integration/reference_impl.py)

- **DAG from the `GridSpec` coordinate table**, written without reading agent 03's code. It reproduces all five of
  the lead's pinned golden node ids and matches `tools/reference_dag.py` node for node (names, parents, budgets,
  checkpoint members, ancestor counts) for main, high, replication and extension.
- **Exact pass@k** with `Fraction`, checked against brute-force subset enumeration for n ≤ 7.
- **Ledger replay checker**: ceiling at every reservation; no release after SENT; one terminal event per
  attempt; CHARGE_MAX not below the reservation.
- **Tri-state transitions** and fixed-denominator bounds; **complete-link MATH classes** (two-way equivalence
  only); **delivered ≤ coverage** limited to candidate-only families; **gz/SQLite-aware canary scanner**.
- Pending (M2): `independent_recompute.py` (`recompute_primary_contrasts`, `recompute_item`) from raw per-item
  files, and one full item per benchmark traced from raw request through policy cost.

## Adversarial fixtures (tests/fixtures/adversarial/, byte-checked against the builder)

ARD36 episode-label collision; 25 prompt-identical roots plus repeated canary draws; duplicate, stale and
out-of-order completion notifications; all-null "selected" selector rows plus legitimate abstentions; tri-state
grades including outage/timeout/inconsistent rows; 12 malformed finals; a ~3 MB peer-packet growth recipe; canary
tokens planted only in private records (MATH gold/solution, MMLU rationale, BBH target note, MBPP private
tests/reference code/plus inputs) and a split fake `sk-or-v1-` key that is never committed as a literal.

## Unresolved blockers

D1–D11 in `docs/requests/08.md`. The ones that endanger money or validity: **D1** (ERROR_ENVELOPE),
**D2** (unresolved exposure lost on replay), **D5** (shared SQLite across run dirs), **D6** (shared write-once
locks), **D8** (sealing CP/POOLED after a MAIN grade join).
