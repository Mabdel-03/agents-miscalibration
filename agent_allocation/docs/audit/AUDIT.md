# AUDIT: openrouter_agent_allocation_v1, independent QA (agent 08)

Status at **M2** (2026-09-17, branch `alloc-wp08`, merged `allocation-v1` = contract v2 plus every agent's modules).
Against the merged production modules the integration suite has **no unexpected failures**. Remaining defects are
tracked as xfails with owner IDs (table below) and in `docs/requests/08.md`.

## Release recommendation

**Do not authorize paid dispatch yet (Gate 1 canary stays closed). The offline pilot plumbing is otherwise sound.**

Fix first (both small, both in the paid-dispatch path):
1. **W02-3**: circuit breakers open late. After a cache HIT, provider mismatch, 402 or 401 response, the dispatcher
   still sent 4–5 more requests at concurrency 1 before stopping. At production concurrency (128, 250 ms group commit)
   that window is much larger. A provider mismatch or cache HIT in that window means billed, contaminated draws.
2. **W02-4**: the $2 unresolved-exposure pause is not re-checked just before sending. Attempts reserved before an
   ambiguous timeout was processed are still sent afterwards (race, 2 of 3 runs).

Also before Gate 2: make `tests/unit` green on the merged branch (it had other agents' contract-v2 breakages at
03:40; re-check), and fix W05-1/W06-2, or code transition and invalid-rate claims (15.14) are not supportable.
Nothing below suggests a scientific-validity leak: gold, private tests, reference code and the key never reached
prompts, bodies or artifacts. Selections are sealed before joins, and the independent recompute agrees with the
analysis module.

Command: `source slurm/env.sh && uv run --frozen --no-sync pytest -q tests/integration`
(about 3–4 min on 8 shared CPUs. The two e2e runs and the L-1 check take most of it; `-k "not e2e"` takes about 40 s.)

## Section 15 acceptance checks

Status: PASSED = production module exercised and passing; XFAIL = production defect confirmed (ID); REF = only the
reference level runs.

| # | Check | Status | Evidence (tests/integration/...) |
|---|---|---|---|
| 1 | Same-prompt draws are distinct network opportunities; shared roots not regenerated | PASSED | `test_acceptance_s15.py::test_s15_01_*` (25 identical bodies, 25 requests, r=0 pools reference roots); e2e exactly-once dispatch |
| 2 | Roots record cannot overwrite private/peer decisions under one episode label | PASSED | ARD36 collision fixture; StateStore keeps colliding-label nodes; extension reuses the 125 core ids |
| 3 | 125 / 123 / 57 per item; CP +12, POOLED +1 | PASSED | production graph equals the independent reference **node for node** for main/high/replication/extension |
| 4 | PRIVATE_SINGLE t=1 and team r=0 reference the roots; 25-bank is no team archive | PASSED | `test_s15_04_*` (both views) |
| 5 | PEER 5x4 = 25 ancestors; no future/same-round visibility | PASSED | `policy_ancestors` equals the reference closure for all 200 checkpoints; rendered prompts contain exactly the parents' markers |
| 6 | PRIVATE_TEAM reuses across n; PEER does not | PASSED | `test_s15_06_*` |
| 7 | Usage charged once in collection, once per policy occurrence; no reasoning double count | PASSED | `policy_cost_rows` vs reference; dispatcher settles exactly Σ usage (happy path, e2e) |
| 8 | Reservations never exceed ceiling; timeout/restart/late bill/429 retain money | PASSED, **W02-4 XFAIL** | tiny-ceiling contention, restart replay, late settlement, dead-process sent-but-unknown, 429 burst; pause race |
| 9 | Fixed population; unknown/unrun not dropped | PASSED | exact `pass_at_k`; independent `fixed_population` check on e2e; unknown grades make contrasts `None` with bounds |
| 10 | Malformed output not retried; retries bounded and metered | PASSED, W06-1 noted | client never retries; malformed final / empty length stop = 1 request; persistent 429 = exactly 3 attempts, each reserved |
| 11 | Selector cannot see private fields; canaries absent from requests/raw/logs/reports | PASSED | selector signature; 1,104 rendered prompts/bodies; key only in the Authorization header; e2e run-dir and SQLite scan |
| 12 | MBPP public selection isolated; attacks contained | PASSED (sandbox present on node4004) | env-key read, host-fixture read, network, fork bomb, infinite loop, stdout flood |
| 13 | Answer classes gold-blind; no one-direction MATH merge | PASSED | one-way equivalence cannot merge; tie-break independent of mapping order |
| 14 | Unknown stays unknown; invalid vs wrong distinguishable | PASSED, **W05-1/W06-2 XFAIL** | tri-state codec refuses contradictions; evaluator-outage injection yields EVALUATOR_ERROR; prose in a Python final is VALID/OK-error |
| 15 | Missing reasoning usage is null; cache HIT blocks the partition | PASSED, **W02-3 XFAIL** | client returns `reasoning_tokens=None`, CACHE_HIT; the circuit opens, but late |
| 16 | Larger budget not reported as more realized reasoning | REF | reference `binding_cap` classifier; production table schema not frozen (skip) |
| 17 | Selectors have real rows; all-null is a blocker | PASSED | seal refuses empty/null-selected sets; readout abstains correctly; independent `selector_rows_present` |
| 18 | Synthesis may exceed coverage; candidate-only may not | PASSED at run level, REF for metrics tables | e2e: POOLED correct over four wrong roots, independent `delivered_le_coverage` and agent 07 audit both pass |
| 19 | Peer order and root ids independent of completion time | PASSED | reversed parents render identically; blind permutation applied |
| 20 | Resume idempotent; raw artifacts and failures kept | PASSED, **L-1 XFAIL** | crash after raw write / SQLite failure / e2e resume: zero re-dispatch, raw/parsed/selections byte-identical, verdicts identical; grades re-computed and rewritten |

## Failure injection (tests/integration/test_failure_injection.py, real 01+02 with agent 08's stub transport)

Passed: exactly-once dispatch plus exact metering; all four profiles through the dispatcher (125/123/57/138, parents
before children, synchronous rounds, deep budgets unclipped); 429 burst with Retry-After ≥ 1 s; persistent 429 bounded
at 3 and not stalling other items; 503, connect failure and 200 error envelope retried as new reserved attempts; an
ambiguous timeout never released; malformed final and empty length stop not redrawn; crash after raw write before
commit with no re-dispatch on resume; SQLite commit failure with resume re-requesting exactly the uncommitted-and-unserved
nodes; stale lease never runs a node concurrently; a dead process's sent-but-unknown attempt reconciled conservatively
with a new attempt id; tiny ceiling under 64-way contention never overbooks and resume completes without re-requesting;
`max_usd` below the ledger ceiling respected; duplicate/stale completion notifications idempotent; double settle never
double-charges.
Confirmed defects: W02-3 (circuit breakers, 4 params, strict xfail) and W02-4 (pause race, non-strict xfail).

## Independent recomputations

- `tests/integration/reference_impl.py`: a DAG builder from the `GridSpec` coordinate table, written before seeing
  agent 03's code. It reproduces the lead's five golden ids and matches production **node for node**; its only gap
  was the `deep/1` (= root[0]) checkpoint, added per spec 5.5. Also exact pass@k (brute force verified), tri-state
  transitions, complete-link classes, the ledger replay checker, and the gz/SQLite canary scanner.
- `tests/integration/independent_recompute.py` (R07-3 hook): contrasts and 16 checks from raw per-item files only.
  - **Synthetic run with hand-computed answers:** `primary_i` = 7/16 and `primary_ii` = 13/16 exactly, with BBH
    subtask weighting verified (item weighting would give 5/12 and 5/6). **Agent 07's `audit_run` agrees with the
    hook on both contrasts.**
  - **Mocked e2e run through `live.run_offline`:** `primary_i` = −1 and `primary_ii` = +1 as designed; all independent
    checks pass; agent 07's audit exits 0 with the hook compared.
  - **One full item per benchmark** traced raw → parsed → selection → grade → policy cost by `recompute_item`
    (125/138 nodes, every raw file re-hashed and its content compared).
- Scratch validation: every production test was first run against spec-faithful scratch shims and 21 deliberate
  mutants (all caught) before any agent's module existed. The shims were not committed.

## Adversarial fixtures (tests/fixtures/adversarial/, byte-checked against `adversarial.py`)

ARD36 episode-label collision; 25 prompt-identical roots plus repeated canary draws; duplicate, stale and
out-of-order completion notifications; all-null "selected" rows plus legitimate abstentions; tri-state grades,
including five contradictory raw rows the strict codec must refuse; 12 malformed finals; a ~3 MB peer-packet growth
recipe; canaries planted only in private records (MATH gold/solution, MMLU rationale, BBH target note, MBPP private
tests/reference code/plus inputs); and a split fake `sk-or-v1-` key that is never committed as a literal.

## Open findings

| ID | Owner | Severity | Summary |
|---|---|---|---|
| W02-3 | 02 | high (paid path) | circuit opens only after raw write + ledger fsync (`runtime/scheduler.py:984-1011`) |
| W02-4 | 02 | medium | unresolved-exposure pause not re-checked before send (`runtime/scheduler.py:905`) |
| W02-5 | 02 | low | journal writes RECONCILE "unresolved:*" instead of the v2 `UNRESOLVED` kind |
| W02-6 | 02 | low | release after SENT journals no reason; journal-only audits need wire evidence |
| W05-1 / W06-2 | 05, 06 | medium (claims) | prose in a Python final is VALID and public-checks OK/'error', not INVALID_CANDIDATE |
| W06-1 | 06 | low | illegal option letters extract VALID (correctly rejected downstream; invalid rates undercount) |
| W07-2 | 07 | low | `reporting/audit.py:32-41` loads the hook outside the try/except and without registering it in `sys.modules`; a hook import error crashes the whole audit (worked around in the hook) |
| L-1 | lead (`live.py`) / 04-05 | low | resume re-grades and rewrites every current grade file; grade cache (spec 10.4) not consulted |
| D1–D11 | lead | resolved in v2 | see `docs/requests/08.md` |

Not covered offline (outside QA scope or needs live/cached data): real dataset loaders against the HF cache
(`needs_hf_cache`), Gate 1 live endpoint and numeric-budget behavior, provider-side billing reconciliation, and
wall-clock throughput. The live canary must verify these under the lead's authorization.
