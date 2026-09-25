# WP-G handoff (branch `passk-wpg`, worktree `agents_scaling_passk_wpG`)

Generation + ops for passk_cont_v1. All P1–P8 items are built and committed; tests are offline (fake transport and
read-only legacy files only). Operator commands: `agent_allocation/docs/passk_cont/RUNBOOK.md`.

## Done

| item | where | tests |
|---|---|---|
| P1 node banks: six partitions, ROOT@A / DEEP≥2A, `root_slot = draw_id = r`, descending budget; stage-1 waves then seeded stage-2 item-waves (`passk_cont_v1/wave\|seed=20260925`) | `passk/banks.py` | `tests/unit/passk/test_banks.py`: counts 12/1056/816/1488/6200/1000, replicate map, no legacy id, waves |
| P2 `allocation passk-run` (registered in `cli.py`): full frozen list per partition, sha asserted against `gates/gate0.json`, ceiling `min(active_ceiling(max_usd, settled, snap, account_auto_reload), shard credit slice)`, drift check before paid dispatch, never grades, `--plan-only`, `--rehearsal`, `--adopt-legacy-lock` | `passk/run.py` | `test_gate0.py`, e2e |
| P3 fenced grading: `blocked_node_ids`, `verify_seal`, `grade_scope`; `grading/worker.py` threads `skip` + `only` (fail closed: only ids listed in the shard's jobs files) through `scan_candidates`/`grade_pass`, recomputed every `--follow` pass; legacy configs unchanged | `passk/grade_scope.py`, `grading/worker.py` | `test_grade_scope.py`, e2e |
| P4 T200 1:1 reserve override (explicit `splits.json:assignment_t200` wins; else `passk.t200_shard_shares`); dev/main keep seed 20260917 and 3:2 | `shards.py` (`with_shares`, `reserve_override`) | `test_shard_override.py`: legacy L96/D32 assignment reproduced |
| P5 mocked e2e: 16 items, two shards, all partitions; order (first admission per wave DEEP or ROOT slot 0; all stage-1 before stage-2); SIGKILLed rehearsal subprocess + resume with no re-send and no duplicate 200 bodies; fence until a valid seal; in-run gates | `tests/integration/test_passk_wpg_e2e.py` (~60–90 s) | |
| P6 THINK-body parity: new ROOT@A and DEEP@2A bodies == legacy raw `request_body.think` bar `max_tokens`, same `provider.only` — **128/128 L96+D32 pass** | `passk/gates.py::think_parity` | `test_think_parity.py` (+ negative controls) |
| P7 E0 subset — **passes on the legacy root in ~15–40 s**: 28,030 unique THINK/ANSWER ids (a natural calibration call's single id counted once), body-sha collisions beyond by-design banks = exactly omni_math_hard/3907 and hmmt/2026/24 (private r18/19), 14,031 parsed on the pin, 1,054 metadata records on `…-20260814` + pin, forced_budget_check exact (21 rows) | `passk/e0_gate.py` | `test_e0_gate.py` |
| P7 forecast (plan unit costs ×1.02 + probes): Mancer ≈ $56.4, Parasail ≈ $40.8 | `passk/forecast.py` | `test_gate0.py` (≤ $115) |
| Gate 0 evaluator + record (`python -m agent_allocation.passk.gates gate0 --check-key --run-tests --run-e0`) | `passk/gates.py` | `test_gate0.py` |
| P8 ops: `slurm/passk_cont.sbatch` (VERBs shard-plan, gate0, canary, canary-eval, smoke, run, grade, grade-follow, checkpoint, analyze; `--signal=B:TERM@900` + trap → `control/DRAIN`, python waited twice, no exec; `+`-joined PARTITIONS; `--account-auto-reload` on paid verbs), `slurm/passk_conductor.sh` + `.sbatch` (flock, squeue duplicate refusal, STATUS.md + conductor.state each transition, `--mail-type=FAIL,END`), in-run gates | `slurm/`, `passk/checkpoints.py` | e2e (gates), `test_checkpoints.py`; shell: `bash -n` only |

## Validated against the lead's REAL manifests (scratch copy, 14:58 EDT; nothing written to the run root)

Copied `passk_cont_v1/artifacts/{manifests,locks}` + `gates/gate_s.json` (no private/) to the scratchpad, then:
`shard-plan --write` (reserve 10/10/50/30 per shard, legacy dev/main counts), `--adopt-legacy-lock` (lock hashes
fa8b9ec9… mancer, 6050d9c4… parasail = forced_v2's), `passk-run --plan-only` (mancer 22 waves / parasail 19) and
`python -m agent_allocation.passk.gates gate0 --run-e0`: **every offline criterion passed** — counts
12/1056/816/1488/6200/1000, 10,572 unique ids with 0 legacy collisions, prompt/decoding equal, THINK parity 128/128
with the lead's manifests, E0 subset, slices, forecast Mancer $57.30 / Parasail $40.32 (total $97.62 incl. probes).
Only `key` and `tests` stay unevaluated offline (the conductor passes `--check-key --run-tests`). A CLI rehearsal of
`passk-run --partitions smoke` on that scratch shard stops at `provider_mismatch` because the fake transport answers
as "Mancer", not the real lock's "Mancer 2" — the pin guard working, not a defect.

## Remaining (priority order)

1. **Merge `passk-wpg` into `passk-cont`** and rerun: `tests/unit/passk`, `tests/integration/test_passk_wpg_e2e.py`,
   `tests/unit/test_shards.py`, `tests/unit/code/test_worker.py`, `tests/integration/test_forced_wp3_mocked.py`.
   Merge touch points: `cli.py` (WP-A adds `passk-analyze` on the next line), `tests/unit/passk/__init__.py` (empty).
2. **Against the lead's real `splits.json`** (Gate S): `passk-run --plan-only` per shard must show the contract counts;
   then `python -m agent_allocation.passk.gates gate0 ...` (RUNBOOK §3). My tests use a §5.1 fixture
   (`tests/unit/passk/wpg_fixtures.py`); if the real schema differs (panel keys, `assignment_t200`), adjust
   `banks.load_splits` / `shards.reserve_override`.
3. **Lead decision — legacy lock adoption.** Node ids embed the endpoint lock hash, so the frozen lists (and Gate 0's
   jobs shas) need the lock before any paid call. I copy each forced_v2 shard lock (`passk-run --adopt-legacy-lock`,
   done by VERB=shard-plan and by the conductor). The canary then reuses it (no refetch) and `passk-run` runs
   `verify_endpoint_drift` before paid dispatch: a changed live endpoint (e.g. price) refuses to dispatch rather than
   silently changing ids. If fresh locks are preferred instead, run Gate 0 after the canaries.
4. **Adversarial review lenses** (Workflow 2): the ceiling arithmetic in `run.run_partitions`; the fence
   (`grade_scope` + worker `only`); conductor stop paths (untested live — only `bash -n`); Gate 2/3 thresholds in
   `checkpoints.py`.
5. Not implemented (noted for the reviewer/lead): Gate 1 smoke `/key` reconciliation (the canary's
   `--evaluate-only` covers the canary window only); Gate 3 "unresolved exposure > $2 stops after 30 min" (the
   dispatcher pauses admission by itself; no stop); Gate 4 union key reconciliation and pair-cache completeness
   (WP-A); Gate 4 duplicate-THINK check is implemented but reads every raw (legacy + new; a few minutes).
6. Overlap with WP-A: contract §5.6 names `labels.blocked_node_ids/verify_seal`; mine live in
   `passk/grade_scope.py` (the grader imports them). WP-A's `extract.py`/`score.py` already import
   `grade_scope.verify_seal` when present; the acceptance rule is aligned (parses, non-empty `code_commit`, non-empty
   `artifacts`, every sha matches, paths relative to the study root). WP-A's `labels.blocked_node_ids` should
   delegate to `grade_scope.blocked_node_ids` after the merge (one definition).
7. Conductor: only dry-run tested with stub Slurm (`tests/integration/test_passk_conductor_dryrun.py`); the
   transient-retry branches (10-min sleeps) and run resubmission are not exercised.

## Known issues

- Suite at 15:01 EDT: `pytest tests/unit tests/integration/test_forced_wp3_mocked.py
  tests/integration/test_passk_wpg_e2e.py` -> **1005 passed, 1 failed**; the failure
  (`tests/unit/runtime/test_status_control.py::test_status_json_is_written_live_and_at_the_end`, and in an earlier
  run `test_aimd_waves.py::test_retries_may_use_the_reserve_above_the_soft_ceiling`) is a timing flake in untouched
  scheduler tests under load: both pass alone, with and without WP-G's changes.
- `PROBE_USD_PER_RUN = $0.05` per passk-run invocation is an estimate for the forced Gate-0 probes.
- `checkpoints gate2/gate3` reread every jobs/parsed file per poll (NFS); fine at 5-min polls, slow if polled faster.
- The conductor's analysis stages call `allocation passk-analyze --stage X` (WP-A); until that verb exists the chain
  stops at `analysis:extract` with exit 2 (safe: the fence holds).

## Exact next commands

```bash
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_cont && git merge --no-ff passk-wpg
cd agent_allocation && source slurm/env.sh
.venv/bin/python -m pytest -q tests/unit/passk tests/integration/test_passk_wpg_e2e.py tests/unit/test_shards.py
STUDY=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1; RUN=$STUDY/artifacts; CFG=configs/passk_cont_v1.json
.venv/bin/allocation shard-plan --config $CFG --run-dir $RUN --write
for s in mancer parasail; do .venv/bin/allocation passk-run --config $CFG --run-dir $RUN --shard $s --partitions smoke --adopt-legacy-lock; done
for s in mancer parasail; do .venv/bin/allocation passk-run --config $CFG --run-dir $RUN --shard $s --plan-only --partitions smoke+c48_extra+e1_fill+t200_diag+v48_bank+t200_bank; done
.venv/bin/python -m agent_allocation.passk.gates gate0 --config $CFG --study-root $STUDY --check-key --run-tests --run-e0
sbatch slurm/passk_conductor.sbatch          # after review + Gate 0 green
```
