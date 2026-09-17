# AUDIT: openrouter_agent_allocation_v1, independent QA (agent 08)

Status at **M2** (2026-09-17 04:20 EDT). Branch `alloc-wp08` has `allocation-v1` @ 38a3820 merged: contract v2 and
every agent's current modules, including `live.run_offline`.

**Integration suite:** `source slurm/env.sh && uv run --frozen --no-sync pytest -q tests/integration`
Result: **147 passed, 2 skipped, 3 xfailed, 0 unexpected failures**, in 3.5 min on 8 shared CPUs.
- The 2 skips are 15.16/15.18 table-level checks waiting on a frozen metrics-table schema.
- The 3 xfails are open defects L-1, L-2 and W05-1/W06-2.
- For a fast run, use `-k "not e2e"` (about 40 s).

**Unit suite** on the merged branch: 760 passed, 2 failed. Both failures are W07-3: agent 07's fixture records a
placeholder raw sha256, which the independent hook correctly rejects.

## Release recommendation

**Offline Gate 0: pass, conditional on two small fixes. Paid dispatch (Gate 1 canary) may be authorized after W07-3
is fixed so `tests/unit` is green.**

The dispatch-path defects found during M2 are fixed upstream and now guarded by passing tests:
- **W02-3:** circuit breakers now open at classification.
- **W02-4:** the unresolved-exposure pause no longer races.

No leak was found: gold, private tests, reference code and the key never reached prompts, request bodies, raw files,
logs, reports or SQLite. Selections are sealed before any grade join. On both the synthetic and the mocked end-to-end
run, the independent recompute agrees exactly with agent 07's analysis.

Must hold before main-study claims:
- **W05-1/W06-2** (prose in a Python final counted as a program). Without the fix, code transition and invalid-rate
  statements (15.14) are not supportable.
- **L-1** (resume re-grades everything; the spec 10.4 grade cache is unused). This is a cost/time issue for MBPP+,
  not validity.
- **L-2** (stubbed code grader not labelled). Offline runs only.

Not covered offline:
- real dataset loaders (`needs_hf_cache`)
- live endpoint and numeric-budget behavior
- provider-side billing reconciliation
- wall-clock throughput

These are Gate 1/2 items for the lead's authorized canary and pilot.

## Section 15 acceptance checks

| # | Check | Status | Evidence (tests/integration/...) |
|---|---|---|---|
| 1 | Same-prompt draws are distinct network opportunities; shared roots not regenerated | PASSED | `test_s15_01_*`: 25 identical bodies → 25 requests; r=0 pools reference roots; e2e exactly-once dispatch |
| 2 | Roots record cannot overwrite private/peer decisions under one episode label | PASSED | ARD36 collision fixture; StateStore keeps colliding-label nodes; the extension grid reuses the 125 core ids |
| 3 | 125 / 123 / 57 per item; CP +12, POOLED +1 | PASSED | production graph equals the independent reference **node for node** (main/high/replication/extension); dispatcher runs each profile once |
| 4 | PRIVATE_SINGLE t=1 and team r=0 reference the roots; 25-bank is no team archive | PASSED | `test_s15_04_*` (current and archive views) |
| 5 | PEER 5x4 = 25 ancestors; no future/same-round visibility | PASSED | `policy_ancestors` equals the reference closure on all 200 checkpoints; rendered prompts contain exactly the parents' markers |
| 6 | PRIVATE_TEAM reuses across n; PEER does not | PASSED | `test_s15_06_*` |
| 7 | Usage charged once in collection, once per policy occurrence; no reasoning double count | PASSED | `policy_cost_rows` vs reference; dispatcher settles exactly Σ usage (happy path, e2e) |
| 8 | Reservations never exceed ceiling; timeout/restart/late bill/429 retain money | PASSED | 64-way tiny-ceiling contention; restart replay refuses to release; late settlement; dead-process sent-but-unknown; 429 burst; unresolved pause (W02-4 fixed) |
| 9 | Fixed population; unknown/unrun not dropped | PASSED | exact `pass_at_k`; e2e PEER outage keeps all 250 nodes planned and agent 07 agrees with the hook; unknown grades make contrasts `None` with bounds |
| 10 | Malformed output not retried; retries bounded and metered | PASSED (W06-1 noted) | client never retries; malformed final / empty length stop = 1 request; persistent 429 = exactly 3 reserved attempts |
| 11 | Selector cannot see private fields; canaries absent everywhere | PASSED | selector signature; 1,104 rendered prompts/bodies; key only in the Authorization header; e2e scan of run dir and SQLite |
| 12 | MBPP public selection isolated; attacks contained | PASSED (sandbox present on node4004) | env-key read, host-fixture read, network, fork bomb, infinite loop, stdout flood; a missing sandbox → INFRASTRUCTURE_ERROR and the selector pauses |
| 13 | Answer classes gold-blind; no one-direction MATH merge | PASSED | one-way equivalence never merges; tie-break independent of mapping order; **vote classes (06) partition finals exactly as the grader (04) does** |
| 14 | Unknown stays unknown; invalid vs wrong distinguishable | PASSED for text; **XFAIL W05-1/W06-2 for code** | strict codec refuses contradictory grades; evaluator-outage injection → EVALUATOR_ERROR; prose Python final → VALID/OK-error |
| 15 | Missing reasoning usage is null; cache HIT blocks the partition | PASSED | client `reasoning_tokens=None`, CACHE_HIT; no request after a breaker response (W02-3 fixed) |
| 16 | Larger budget not reported as more realized reasoning | REF (skip at table level) | reference `binding_cap`; e2e deep calls realize 300 tokens at every budget; no frozen table schema to assert on |
| 17 | Selectors have real rows; all-null is a blocker | PASSED | seal refuses empty/null-selected sets; readout abstains correctly; independent `selector_rows_present` |
| 18 | Synthesis may exceed coverage; candidate-only may not | PASSED at run level; REF for metrics tables | e2e: POOLED correct over four wrong roots; the independent check and agent 07's audit both pass |
| 19 | Peer order and root ids independent of completion time | PASSED | reversed parents render identically; blind permutation applied |
| 20 | Resume idempotent; raw artifacts and failures kept | PASSED (**XFAIL L-1** for grade-file rewrites) | crash after raw write, SQLite failure, e2e crash and plain resume: zero re-dispatch; raw/parsed/selections byte-identical; verdicts identical |

## Failure injection (real 01+02 with agent 08's stub transport; tests/integration/test_failure_injection.py)

All passing:
- exactly-once dispatch and exact metering
- the four profiles (125/123/57/138) with parents before children, synchronous rounds and unclipped deep budgets
- a 429 burst honoring Retry-After ≥ 1 s
- persistent 429 bounded at 3 attempts without stalling other items
- 503, connect failure and a 200 error envelope retried as new reserved attempts
- an ambiguous timeout never released
- the unresolved-exposure pause holds admissions until reconciliation
- all four circuit breakers stop dispatch immediately
- a malformed final and an empty length stop are never redrawn
- crash after raw write before commit: no re-dispatch
- SQLite commit failure: resume requests exactly the unserved nodes
- a stale lease never runs a node concurrently
- a dead process's sent-but-unknown attempt is charged conservatively under a new attempt id
- a tiny ceiling under 64-way contention never overbooks, and resume completes without re-requesting
- `max_usd` below the ledger ceiling is respected
- duplicate and stale completion notifications are idempotent
- a double settle never double-charges
- sandbox outage and public-check outage pause the code selector

## Independent recomputations

- `reference_impl.py`: a DAG built only from the `GridSpec` coordinate table, before any agent code existed.
  - It reproduces the lead's golden ids and equals production node for node. Its only gap was the spec-5.5 `deep/1`
    checkpoint, now added.
  - Also: exact pass@k (brute-force verified), tri-state transitions, complete-link classes, the ledger replay checker
    (release-after-send needs wire evidence), and the gz/SQLite canary scanner.
- `independent_recompute.py` (R07-3 hook, loaded by `allocation audit`): contrasts and 16 checks from raw per-item
  files only.
  - **Synthetic run with hand-computed answers:** `primary_i` = 7/16 and `primary_ii` = 13/16 exactly, with BBH
    subtask weighting verified (item weighting would give 5/12 and 5/6). **Agent 07's analysis agrees.**
  - **Mocked e2e run via `live.run_offline`:** −1 and +1 as designed; every independent check passes; the audit exits
    0 with the comparison included, including under a PEER outage.
  - **Tamper tests:** seal bytes, raw/parsed drift, evaluator outage (bounds), missing selector rows, leaked keys,
    unsettled successes and invalid-graded-as-wrong are all detected.
  - **One full item per benchmark** traced raw → parsed → selection → grade → policy cost by `recompute_item`.
- Before the modules existed, every production test ran against scratch shims plus 21 deliberate mutants (all caught);
  the vote/grade agreement test was mutation-checked as well (2/2 caught).

## Adversarial fixtures (tests/fixtures/adversarial/, byte-checked against `adversarial.py`)

- ARD36 episode-label collision
- 25 prompt-identical roots plus repeated canary draws
- duplicate, stale and out-of-order completion notifications
- all-null "selected" rows plus legitimate abstentions
- tri-state grades, including five contradictory raw rows the strict codec must refuse
- 12 malformed finals
- a ~3 MB peer-packet growth recipe
- canaries planted only in private records: MATH gold/solution, MMLU rationale, BBH target note, MBPP private
  tests/reference code/plus inputs
- a split fake `sk-or-v1-` key, never committed as a literal

## Open findings (details: docs/requests/08.md)

| ID | Owner | Severity | Summary | Test |
|---|---|---|---|---|
| W05-1 / W06-2 | 05, 06 | medium (claims) | prose in a Python final is VALID and public-checks OK/'error', not INVALID_CANDIDATE | strict xfail |
| L-1 | lead `live.py` (+04/05 cache) | low (cost) | resume re-grades and rewrites every grade file; grade cache unused | strict xfail |
| L-2 | lead / 07 | low | stubbed code grader not labelled in reports | strict xfail |
| W07-3 | 07 | low (blocks unit green) | unit fixture writes a placeholder raw sha256; the hook correctly fails it | tests/unit (2 failures) |
| W07-2 | 07 | low | hook loaded outside the audit's error handling and without a `sys.modules` entry | worked around |
| W06-1 | 06 | low | illegal option letters extract VALID (rejected downstream; invalid rates undercount) | tolerated with downstream check |
| W02-5 / W02-6 | 02 | low | journal writes RECONCILE "unresolved:*" instead of the v2 kind; release after SENT carries no reason | noted |
| W02-3, W02-4 | 02 | fixed | circuit timing; unresolved pause race | now passing tests |
| D1–D11 | lead | resolved in contract v2 | see requests | passing tests |
