# Adversarial review: node identity and reuse exactness (passk-cont @ 3c0a530)

Lens: banks.py, run.py (--adopt-legacy-lock, drift check), gates.py think_parity, shards.py override, Gate 0 / passk-run jobs sha freezing.

## [major] Drift check refuses all paid dispatch on any endpoint-listing change, including cosmetic ones, and there is no recovery path that keeps the frozen node ids

**Location:** `src/agent_allocation/passk/run.py:183-194` (calls `live.verify_endpoint_drift`, `src/agent_allocation/live.py:127-142`). The comparison is `live.lock_hash != lock.lock_hash`, and `contracts.py:271-276` builds `EndpointLock.lock_hash` from every non-volatile field: provider_name, the three price **strings**, context_length, max_completion_tokens and the whole `supported_parameters` list.

**Problem:** The adopted forced_v2 locks (fa8b9ec9 mancer, 6050d9c4 parasail) were fetched 2026-09-18. Every paid `passk-run` repeats the check: the smoke run, the first main run, and each of the up to 3 conductor resubmissions after a 12 h time-limit drain. Any difference makes the run refuse. That includes changes that do not affect the forced protocol:
- a parameter added to or dropped from `supported_parameters`, even one never used;
- a change to `max_completion_tokens` or `context_length` that still clears 16A + 4,096 headroom;
- the same decimal price written differently.

The node ids and the Gate 0 jobs sha256 values embed this hash, so the only way forward is a fresh lock. A fresh lock changes all 10,572 ids. `gates gate0` then refuses (rc 3) until someone passes `--overwrite`, and after that every node that already finished sits under an old id. Those old ids are missing from the new jobs files, so the nodes would be dispatched again, which pays twice, and `extract` would never see the orphaned results. No document describes this procedure. In the conductor, drift in the middle of the run looks like this: passk-run returns 2, the job reads COMPLETED rc=2, the conductor resubmits 3 times ($0 each), and then it runs `stop_main "run <s> stopped early 3 times (rc=2)"`. The stop message never says "drift". The refusal also records only the string `"live endpoint semantics differ from the lock"`. No field diff is recorded and the live snapshot is not saved (`verify_endpoint_drift` discards `_raw`), so the owner cannot tell whether the change matters.

**Evidence** (snippet against the legacy parasail snapshot, fake `list_endpoints`):
```
unchanged                                   -> ok
adds min_p to supported_parameters          -> REFUSE
drops unused 'tools'                        -> REFUSE
max_completion_tokens 235929->262144        -> REFUSE
prompt price '0.00000024'->'0.000000240'    -> REFUSE   (Decimal-equal)
status/uptime change                        -> ok
```
There has been no drift so far. The dynorch_v1 snapshot from 2026-09-24T04:59Z (`dynorch_v1/artifacts/locks/snapshots/endpoints_qwen_qwen3.8-27b_20260924T045920Z.json`) selects to the same lock hashes (fa8b9ec9 and 6050d9c4). The risk is a multi-day run meeting a single listing edit.

**Fix:** Split the check into two parts:
- **Hard fields (refuse):** `provider_tag`, `provider_name` (the pin), the prices compared as `Decimal`, `FORCED_REQUIRED_ENDPOINT_PARAMETERS ⊆ supported_parameters`, `max_completion_tokens >= 16*A` and `>= headroom`, and `context_length` large enough for the longest prompt plus 16A plus headroom.
- **Other fields (record, keep dispatching):** the rest of `supported_parameters`, a larger cap, and so on. Record these as `record_decision(..., decision="drift_noted", diff=...)`.

The adopted lock stays the identity of the nodes. On refusal, save the live snapshot and print a field diff. Also run the same check read-only inside Gate 0 when `--check-key` is on (it already goes online for `/key`), so drift is caught before the canaries spend money. Add a RUNBOOK entry for the owner that says what to do on drift.

(How likely the drift finding bites: the plan budgets about 2 h of generation, so what matters most is (a) drift between the last identical snapshot, 2026-09-24T04:59Z, and launch, found only after both canaries and the Gate 0 forced probes have spent money, because `run_canary` reuses the adopted lock and never checks drift; and (b) any resubmission after a stop. The chance is low, but the result is a hard stall that no one can recover from within the rules.)

## [minor] A failed transient GET /endpoints in the drift probe crashes passk-run; the smoke run then stops the conductor

**Location:** `passk/run.py:186-191` calls `OpenRouterClient.list_endpoints`, which goes through `api/client.py:184-204` and makes one attempt with no retry. Any transport error or 5xx raises `OpenRouterAccountError`. `cmd_passk_run` (`run.py:285-291`) catches only `PasskRunRefused`, so the process exits 1 with a traceback.

**Problem:** In Gate 1 the conductor stops as soon as the smoke job returns rc != 0 (`passk_conductor.sh:162`). By then the canaries have already been paid for. One 502 on a $0 metadata GET therefore halts the chain, and an operator has to resume it with `START_AT=gate1`. That resume re-runs both canaries. The canary nodes are already terminal, but each canary re-sends its paid Gate 0 endpoint probes. In main, each such blip uses up one of the 3 resubmissions (`passk_conductor.sh:245`).

**Fix:** Retry the drift probe with backoff, for example 5 tries over about 5 minutes. Only `EndpointBlocked` and a real field diff should count as drift. If the probe still fails after the retries, exit with a distinct code the conductor can tell apart from drift, and have the conductor retry that code at Gate 1 instead of stopping. For resume, give the conductor a `START_AT=smoke` entry point that skips the canaries.

## [minor] A Gate 0 re-run after spending money can replace a passed record with a failed one, and every paid passk-run then refuses

**Location:** `passk/gates.py:167-176`. `write_gate0` refuses only when the jobs shas change; it writes over a passed record with any result, including a failed one. `gates.py:230-240`: `key_check` requires `sum(credit ceilings) = 150 <= limit_remaining - 3` and ignores money already settled. `run.py:136-146`: `require_gate0` refuses unless the record says `passed` and `complete`.

**Problem:** STATUS.md advertises `START_AT=gate0` as a resume point (`passk_conductor.sh:49`). Suppose it is used after spend S, for example after a code fix, to re-run the tests on the new commit. The key check now needs `limit_remaining >= 153` even though only `150 - S` more can be spent. On a key whose limit is close to the cap, Gate 0 fails and `gate0.json` becomes `passed: false`. From then on every paid `passk-run` refuses "no passed and complete Gate 0 record", at Gate 1 and in main, and the conductor stops or burns its 3 resubmissions. The only way out is to edit the record by hand. The same happens if any other criterion fails transiently on re-run, such as `--run-tests`.

**Fix:** Make `write_gate0` refuse to replace a `passed: true` record with a failed one unless `--overwrite` is given. Keep failed re-runs in `gates/gate0_attempts/<ts>.json` instead. Compute the key check against the money still to be spent: `sum(ceilings) - sum(ledger_settled) <= limit_remaining - 3`.

## [minor] `reserve_override` falls back silently to a hash assignment, and Gate 0 never checks that the materialized T200 split equals `splits.json:assignment_t200`

**Location:** `shards.py:290-311`. `assignment_t200` is used only when `items <= set(explicit)`; otherwise the function returns without a message and uses `shard_assignment(with_shares(cfg, {1:1}))`. `gates.py:243-325` has no criterion comparing `shards/assignment.json` or `shards/<s>/manifests/shard_assignment.json` against `splits.json`.

**Problem:** On today's artifacts the verbatim path is taken. Verified in memory with `root_assignment` on the real run root: `panel_overrides.reserve.source == "splits.json:assignment_t200"`, all 200 T200 items match, and L96 ∪ D32 reproduces the legacy mapping. The risk is what happens if `panels.json` reserve and `splits.json` ever drift apart. `shard-plan --write` would then silently produce a different 1:1 split. The node counts would be unchanged, so the `planned_counts` check still passes. Gate 0 would freeze the wrong assignment. E5 imputes the provider feature from `splits.json:assignment_t200` (`models/e5.py:160`, `models/inputs.py:250`), so its provider covariate would disagree with the provider that actually served the item.

**Fix:** In the passk path, when `splits.json` exists but does not cover the reserve, raise `ShardError` instead of falling back. Add a Gate 0 criterion `assignment_t200_verbatim`: for every T200 item, the shard whose `shard_assignment.json` holds it equals `assignment_t200[item]`. Add a second criterion that L96 ∪ D32 equals the legacy `shards/assignment.json`.

**Addendum to the drift-probe finding above (same stall, second cause):** Each `passk-run` builds a new `Runtime`. Its `Runtime.run` calls `forced_gate0()` (`live.py:311-312, 322-336`), which sends four single-shot paid probes (`api/forced.py:~730-840`); any non-OK transport status fails the criterion. A single 429 or 5xx on a probe raises `RunRefused`. That is not `PasskRunRefused`, so `cmd_passk_run` crashes with rc 1. The canary step gets 3 transient retries from the conductor (`passk_conductor.sh:144-150`); the smoke step and the first main step get none (`:162`). Fix: catch `RunRefused` and endpoint-probe `OpenRouterAccountError` in `cmd_passk_run`, classify them as transient or semantic the same way `gate0_endpoint_class` does, and return a distinct rc that the conductor retries at Gate 1.

## [minor] Nothing asserts that the shard lock the new node ids carry equals the forced_v2 lock; `--adopt-legacy-lock` silently does nothing when a lock already exists

**Location:** `passk/run.py:85-86`. `adopt_legacy_lock` returns `{"adopted": false, "reason": "lock exists"}` with exit 0, and the conductor carries on (`passk_conductor.sh:116-119`). Gate 0 (`passk/gates.py:273-325`) has no criterion for lock equality. `think_parity` cannot catch a difference, because the THINK body carries only `provider.only` and no lock field. The E0 gate takes its pin from the legacy locks (`e0_gate.py:60-63`). Gate 1 and the checkpoints take their pin from the new locks (`checkpoints.py:_pin`). Nothing compares the two.

**Problem:** `HANDOFF_wpG.md` §Remaining 3 lists "fresh locks are preferred instead, run Gate 0 after the canaries" as an allowed path, and anyone who runs `VERB=canary` before the shard-plan step gets a fresh lock too. On that path, suppose OpenRouter changed the endpoint after 2026-09-18: a new `provider_name` (another Mancer backend), a new price, or a new cap. Then the new ids carry a different lock hash and the pin guards check the new name. The drift check compares live against the fresh lock and passes. Every gate passes, and the new draws get pooled with the legacy r0 and A-slot draws (E1 L96 pools, C48 fits) even though they came from a differently described endpoint. The contract's reuse exactness then holds only by assumption. With adoption, as the conductor does it, the drift check forces live == legacy, which is safe. The gap is that the other path is accepted without any record.

**Evidence:** Rebuilding the new specs with the legacy `run_id` substituted reproduces **every** legacy ROOT and DEEP node id: 3,192 of 3,192 across main, pilot, calibration and canary on both shards, run in memory with the real passk manifests and the legacy locks. So today's adopted-lock plan is exactly legacy-identical in every identity field except `run_id` and `partition_id`: the lock hash, the decoding hash, `input_manifest_hash`, the headroom and `prompt_version`. That identity depends on the adoption having actually happened, and nothing checks that it did.

**Fix:** Add Gate 0 criterion `lock_equals_legacy`: for each shard, `read_endpoint_locks(shard)[model].lock_hash == legacy shards/<s> lock_hash` (fa8b9ec9…, 6050d9c4…). Have `adopt_legacy_lock` return non-zero when an existing lock's hash differs from the legacy one, unless the owner passes an explicit flag. The flag's use should be recorded in `gates/`.

**Addendum to the Gate 0 re-run finding (same function, opposite direction):** `write_gate0` (`gates.py:170-173`) freezes whatever `jobs_sha256` the previous record held, **even when that record failed**. `evaluate_gate0` skips any shard that has no lock (`gates.py:279-281`), so its `jobs_sha256` covers only the shards that do. Take a Gate 0 run before `--adopt-legacy-lock` has run on both shards, for example RUNBOOK §3 run by hand, or a first adoption that failed on one shard. It writes a record whose sha map lacks that shard. After the lock is adopted, every later Gate 0 refuses with rc 3 ("recorded jobs sha256 differ … refusing to overwrite"), and the conductor stops at `gate0`. Only `--overwrite` gets past it, which the RUNBOOK reserves for owner decisions. Fix: freeze shas only from a record with `passed: true`, and compare shard by shard and partition by partition, so that adding a shard or partition that was missing is not treated as a change.

## [minor] Nothing checks that reused legacy draws and new draws are exchangeable; the only guards compare names (permaslug and provider pin)

**Location:** The design is in contract §3 (legacy deep/m = r0 of the m·A bank; legacy A slots 0–15 form the L96/C48 16×A pools). The pools are built in `pools.py:47-52, 113` (L96/C48 prefixes r0…r(n−1) mix `legacy_main` with `e1_fill` and `c48_extra`). The gates are `checkpoints.py` (gate1-smoke, sample, gate2, gate4) and `e0_gate.py`. None of them compares outcome distributions between legacy and new draws.

**Problem:** Every multi-call L96 pool in the E1 lock set mixes eras: 8x2 is legacy r0 plus new r1–7, 4x4 is legacy r0 plus new r1–3, and 2x8 is legacy r0 plus new r1. By contrast, 16x1 and 1x16 on L96 are entirely legacy. The C48 fitting scope for E2 uses legacy A slots 0–15 and mixes deep replicates across eras, and it is validated against V48, which is entirely new. Suppose Mancer or Parasail changed their serving stack after 2026-09-18 while the permaslug `…-20260814` and the `provider_name` stayed the same (a kernel, quantization or vLLM update, for example). Then the E1 lock (the share of new draws differs across 8x2, 4x4 and 2x8), the L96 menu contrasts and the E2 validation would all absorb an era effect that nothing detects. The T200 primary test uses only new draws and is unaffected.

**Evidence:** The identity fields match exactly: every legacy ROOT/DEEP node id reproduces (see the lock finding above), and THINK bodies match at root_slot > 0 up to 16A (120/120 checked in memory: DEEP 16A r7, 8A r5, 4A r6, ROOT r100 and r15). The protocol is therefore identical. What remains unchecked is whether the provider's behaviour is also the same.

**Fix:** Pre-register two diagnostics that cost nothing to run and put both in the report:
1. **Before the lock** (L96 labels are released): for each L96 item, pair the legacy 2A r0 draw with the mean of the new 2A r1–7 draws. Report the paired difference in correctness, and the KS difference in valid-final rate, ANSWER length and THINK-to-ANSWER finish reasons, per shard. Flag the result if the 95% interval excludes 0 by more than a preset margin.
2. **After the V48 release:** on the same 48 items, compare legacy A slots 0–15 with `v48_bank` A r0–15. That is 16 vs 16 draws per item, a strong era test. Also report the E2 V48 validation after removing the estimated era offset.

These checks do not need to stop the run. The report has to state whether the reused draws can be treated as exchangeable with the new ones.

## Verified OK (no defect), for the record
- **Replicate map** (`banks.py:56-67`) equals contract §3 exactly. Read back from nodes built in memory with the real manifests: smoke {1A:[100]}; e1_fill {2A:r1–7, 4A:r1–3, 8A:r1}; c48_extra {4A:r4–7, 8A:r2–7, 16A:r1–7}; v48_bank/t200_bank {A:r0–15, 2A:r0–7, 4A:r0–3, 8A:r0–1, 16A:r0}; t200_diag {A:r0–4}. The counts are 12/1056/816/1488/6200/1000. The pool prefixes in `pools.py` are all covered with no (allowance, replicate) collision.
- **Ids:** 10,572 planned ids, all unique, and 0 collide with the 14,031 legacy ids. No T200 item appears in any legacy partition.
- **Legacy identity:** with `run_id` substituted, new specs reproduce 3,192 of 3,192 legacy ROOT/DEEP node ids. So the lock hash, decoding hash, input_manifest_hash, headroom and prompt_version are all equal to forced_v2's.
- **THINK parity:** rendering (`prompts/render.py:389-391`, `TEMPLATE_FOR_KIND`) and `build_forced_think_body` (`api/request.py:205-222`) never read `root_slot` or `draw_id`. 120/120 DEEP and ROOT bodies at root_slot > 0 and 4A–16A equal legacy apart from `max_tokens`.
- **T200 assignment:** `root_assignment` on the real run root takes `splits.json:assignment_t200` verbatim, 200/200 (10/10, 10/10, 50/50, 30/30). L96 ∪ D32 reproduces the legacy mapping for all 128 items. 4 of the 15 legacy-reserve items now in T200 move shards, which is harmless because legacy never ran them.
- **Gate 0 vs passk-run shas:** both come from `banks.jobs_text_sha`, which is textually identical to `live.write_jobs`, through the same `shard_plan` with the same shard `amendments.jsonl`. Only the passk pilot writes that file, and passk never runs the pilot. `test_gate0.py::test_passk_run_plan_matches_gate0_and_refuses_a_changed_list` passes. `tests/unit/passk/{test_banks,test_gate0,test_think_parity,test_shard_override}.py` give 21 passed.
- **Lock drift today:** the dynorch_v1 snapshot from 2026-09-24T04:59Z selects to exactly fa8b9ec9 (mancer/fp8) and 6050d9c4 (parasail/fp8).
