# dynorch_v1 — adversarial review, fix pass (2026-09-24)

Source: `agents_scaling_results/dynorch_v1/review_issues.json` (3 blockers, 14 majors, 9 minors, two reviewers).
Every issue is answered below: severity, file, verdict (`fixed` / `rejected`), the commit, and for a rejection or a
deviation from the reviewer's suggested fix, the one-line reason. **Where a suggested fix conflicted with the plan
(`~/.claude/plans/we-ran-the-plans-lovely-meteor.md`) or the contract, the PLAN WON** — those lines say so.
Interface-level consequences are recorded in `docs/dynorch/00_contract.md` §11 (§10.22–§10.27).
Every fix landed with a regression test that fails without it (named in the commit).

| # | severity | file | verdict | commit | note |
|---|---|---|---|---|---|
| 1 | blocker | `dynorch/plan.py` | fixed | `745f2bf` | run_key, RunStore/journal/work paths and the runner's cell key all carry the partition (contract §10.22); canary and pilot no longer share four cells' identity. |
| 2 | blocker | `dynorch/runner.py` | fixed | `343738d` | the watchdog counts `idle_seconds` from the last SERVED reply and never fires while `in_flight > 0`, so a slow TERMINAL answer is not killed mid-call (plan B4 verbatim). |
| 3 | blocker | `dynorch/core.py` | fixed | `4662476` | B9 cache keyed `(conversation, prompt sha)` and filled only by undelivered calls; two sub-agents with identical prompts get two generations (contract §10.23). |
| 4 | major | `dynorch/cli.py` | fixed | `4fe8a10` | Gate 4's grader re-run compares against `gates.PARSE_VALID` (the codec value, contract §10.9) and logs how many rows it re-graded. |
| 5 | major | `dynorch/runner.py` | fixed | `ba37752` | the store is always consulted; a finished cell is skipped, an unfinished one needs `--resume` (RunRefused); every sbatch VERB now passes `--resume`. |
| 6 | major | `dynorch/httpd.py` | fixed | `cb2fd46` | parked calls are shielded and awaited (`wait_for_run` / `wait_idle`, `close()` drains first) and `Upstream.complete` charges the maximum if a cancellation escapes after `mark_sent`. |
| 7 | major | `dynorch/harness/codex.py` | fixed | `0ceb5b9` | the `timeout` wrapper sits at `wall + sigkill_after + 60`; a 124/137 exit is classified `timeout`. Reviewer's "drop the wrapper" option rejected: Gate 0 pins that it kills the whole tree. |
| 8 | major | `dynorch/runner.py` | fixed | `ba37752` | the drain stops admissions and only terminates the harnesses still running after `drain_wait_seconds`; the wait is shielded so a timeout no longer cancels the runs. |
| 9 | major | `dynorch/runner.py` | fixed | `ba37752` | `_attempt` records an exception after `open_run` as a failed run and `one()` catches the rest, so one bad cell cannot take down the job; the sbatch adds the openclaude bin dir to PATH. |
| 10 | major | `dynorch/core.py` | fixed | `1b6d6e0` | cap-bound is `length` at/above the grant; any other non-`stop` finish is `forced_think_short` → settle, journal, `upstream_failed` (contract §10.25). |
| 11 | major | `dynorch/runner.py` | fixed | `ba37752` | the rerun decision reads only `ReplyLog.last_event`; the `record.error` branch is gone, so a recovered sub-agent failure no longer discards a delivered answer. |
| 12 | major | `dynorch/runner.py` | fixed | `ba37752` | an upstream 402/401 drains the stage and stays rerun-eligible on `--resume`. Deviation: it does not consume one of the two reruns AT ALL (the frozen table bounds reruns of runs the model served; B6 calls this infrastructure, not a model failure). |
| 13 | major | `dynorch/tables.py` | fixed | `4fe8a10` | `read_run_records` reads `runs/` and `runs_superseded/`; the test helper writes the real RunStore layout and the kill/resume e2e checks `retry_cost_usd`. |
| 14 | major | `dynorch/plan.py` | fixed | `4fe8a10` | the stage manifest is write-once: a subset leaves it untouched, a foreign spec raises. |
| 15 | major | `dynorch/core.py` | fixed | `4662476` | one rule for B7: the delivered answer is the last root reply the wire layer handed to the client; the runner awaits the run's in-flight call before `close_run` (`ba37752`). |
| 16 | major | `dynorch/cli.py` | fixed | `ba37752`, `607486d` | the store-always-consulted rule in `Runner.run_specs` plus defence in depth: `open_run` refuses a cell that already has a terminal record or journaled calls, and a `LedgerError` (a duplicate attempt id) is answered 500 `internal_error` with the token grant RELEASED — it used to fall into the generic BaseException branch and charge a call that never left the process. |
| 17 | major | `dynorch/core.py` | fixed (part) | `1b6d6e0` | every settled grant is journaled (THINK phase, cost, generation id, CallRecord). **Rejected** the second half — returning the THINK reply as `mid_call_unfunded`: B5 freezes `forced_close_kind ∈ {none, mid_call, visible_continuation, terminal}` and the frozen rerun rule already covers a ledger error, so the plan wins. |
| 18 | minor | `dynorch/runner.py` | fixed | `ba37752` | the `≤ 2 reruns` cap now bounds interrupted records too; beyond it the record is closed `failed` with y = NaN. |
| 19 | minor | `dynorch/doctor.py` | fixed | `6915f0f` | version probes redirect HOME, CLAUDE_CONFIG_DIR, CODEX_HOME, TMPDIR and the XDG dirs into scratch. |
| 20 | minor | `slurm/launch_dynorch_v1.sh` | fixed | `6915f0f` | a comma in `RUN_FLAGS` is refused, `--export=ALL`, the sbatch puts openclaude on PATH and `ALLOCATION_LOCAL_STATE` on scratch; the runbook documents the direct-sbatch log dir. |
| 21 | minor | `dynorch/runner.py` | fixed | `ba37752` | `_supersede` saves the mutated record before archiving, so the superseded copy and the attempts journal agree. |
| 22 | minor | `dynorch/cli.py` | fixed | `6915f0f` | Gate 1's `$HOME` check is a quota inode delta recorded in the stage marker; "not scanned" is its own notify criterion instead of a failed one (the walk stays as the fallback). |
| 23 | minor | `dynorch/runner.py` | fixed | `ba37752` | a stage refuses to start while `control/STOP` / `control/DRAIN` exists (the reviewer's other option); the runbook says to remove it. |
| 24 | minor | `dynorch/cli.py` | fixed | `6915f0f` | extension admission: the previous extension must be sealed and `remaining ≥ 1.25 × the block forecast`; `ALLOCATION_LOCAL_STATE` moved off `/scratch`. |
| 25 | minor | `dynorch/core.py` | fixed | `1b6d6e0` | the E0 ceiling-close exception path settles `think + cont_max` (B2's full amount). |
| 26 | minor | `dynorch/core.py` | fixed | `343738d`, `1b6d6e0` | the TERMINAL call renders the re-injected messages and its reply is returned verbatim (B4 offers no tools, so no tool-call parse and no `finish: tool_calls` after exhaustion). |

(The numbering follows `review_issues.json` order, blockers first; the file column is the reviewer's, normalised to
the package path.)

## Decisions the reviewers left open

* **Delivered answer (B7).** The reviewers disagreed on the source. The plan says "the visible content of the last
  root-conversation assistant message with non-empty content *as served by the proxy*" and the runner's docstring
  calls the wire layer authoritative. Resolution (conservative, never credit an answer the harness did not get):
  the core and the wire layer share ONE signal — a reply counts only once `httpd` reports it delivered; a reply
  produced after the harness died is charged, journaled and reported, but is never the delivered answer.
* **A disconnected reply served to a later identical request.** Kept servable for repeated retries of the same
  disconnected request (B9's own words, and it never spends more). The scientific bug — a DELIVERED reply reused
  for a second conversation — is what the disconnect condition removes.
* **Infrastructure stops and the rerun cap.** See issue 12 above: an upstream 402/401 never consumes a rerun.
* **Nothing was rejected outright.** The only rejected sub-proposal is issue 17's `mid_call_unfunded` (plan wins).
| blocker | dynorch/core.py (B3 grace) | fixed | `b5043a2` | canary 2026-09-24: grace continuation offered tools; 3/8 harness answers lost; plan B3 "no tool schemas" enforced; E0 ceiling_close keeps tools |

## Second fix pass (2026-09-24; `rereview_findings.md`, findings 2-13; blocker 1 was fixed by the lead in `b5043a2`)

Same rules: one line per finding, a regression test that fails before the fix, interface changes in contract §12
(§10.29 onward); where a suggested fix conflicted with the plan, the plan won and the line says so.

| # | severity | file | verdict | commit | note |
|---|---|---|---|---|---|
| 1-smoke | blocker follow-up | `tests/integration/test_dynorch_{openclaude,codex}_smoke.py` | fixed | `c3a7898` | `*/forced_close_then_402` pinned the pre-§10.28 defect; split into `forced_close_text` (grace XML reaches the CLI as text, nothing spawned, 402 probe) and `subagent_402_then_terminal` (natural spawn leaves < 64, sub-agent 402, root TERMINAL); every Gate 0 criterion kept. |
| 7 | major | `dynorch/reconcile.py` | fixed | `861d337` | (a) `/generation` `native_tokens_completion`, not the normalised `tokens_completion`; (b) `/key` re-read every 30 s until two reads agree (<= 300 s) before a dollar mismatch (contract §10.29). |
| 2 | major | `dynorch/httpd.py`, `dynorch/core.py` | fixed | `4623b7f` | an EOF watcher sets `client_gone` at once; the core caches a gone client's reply before the FIFO lock is released, so the queued identical retry is served it (contract §10.30). The reviewer's "wait for an in-flight flagged call" is realised by the FIFO lock itself. |
| 10 | minor | `dynorch/httpd.py` | fixed | `4623b7f` | delivery requires `client_gone` clear before the write and one loop turn after the drain (contract §10.30). |
| 3 | major | `dynorch/runner.py`, `dynorch/core.py` | fixed | `2c0079b` | the core's `infrastructure_stop` flag is cleared by any later served reply; the runner rewrites a run to `transport_failed` only while it is set; a recovered run keeps its answer, the stage still drains (contract §10.31). |
| 4 | major | `dynorch/runner.py` | fixed | `e376350` | `ledger_refused` -> `rerun_reason = "ledger_stop"`: no in-job rerun, the stage drains, `--resume` re-executes without consuming a rerun; plus an admission check (`ledger_block_reason`) before every cell (contract §10.32). Both reviewer options taken. The frozen rule's "ledger error -> rerun" is kept (the rerun happens on `--resume`). |
| 8 | minor | `dynorch/upstream.py`, `dynorch/core.py` | fixed | `e376350` | a pre-send refusal on transport attempt >= 2 carries the earlier phases/$; the grant is settled at 0 tokens and the call journaled; the PRE_SEND comment corrected (contract §10.32). |
| 9 | minor | `dynorch/core.py` | fixed | `e376350` | `_terminal` handles `internal_error` like `ledger_refused` (call number back, error recorded, grace unused). |
| 5 | major | `dynorch/runner.py`, `dynorch/cli.py` | fixed | `d9283ea` | drain wait 540 s inside a 660 s drain deadline that bounds every settle wait; cancelled attempts closed `killed_by_drain`; a drained stage's `/key` settle wait cut to 60 s; worst case < 900 s pinned by a test (contract §10.33). 540 s is within the plan's "wait <= 800 s". |
| 6 | major | `dynorch/core.py`, `dynorch/runner.py`, `dynorch/journal.py` | fixed | `d9283ea` | `core.harness_exited` on every exit path; queued/late calls refused `run_closed` with no grant. Deviation: a late update after close is not dropped but written only where it cannot clobber another attempt (`RunStore.save_if_owned`), because the late call was paid and B1 counts every served call (contract §10.33). |
| 13 | minor | `dynorch/runner.py` | fixed | `d9283ea` | `_close_failed` awaits `_await_settled` first. |
| 11 | minor | `dynorch/plan.py`, `dynorch/runner.py` | fixed | `98fb1f9` | `write_plan(registered=)`: the full `plan_stage` list is pre-registered on the first launch whatever filter runs; incoming specs must be a subset (contract §10.34). |
| 12 | minor | `dynorch/tables.py` | fixed | `98fb1f9` | `run_key_of(..., partition=)` delegates to `plan.run_key_for`. |
| supersede | lead request | `dynorch/supersede.py`, `dynorch/cli.py`, `slurm/*` | added | `14ddce6` | `dynorch-run --supersede "<reason>"`: moves the defective execution aside, records the decision, re-executes every planned cell under a new generation; never touches the ledger journal, locks or other stages (contract §10.35). |
| 4 (cap) | major follow-up | `dynorch/runner.py` | fixed | `ef9cf03` | the rerun cap counts MODEL attempts: `_reruns_used = attempt - 1 - stop attempts` (earlier `infrastructure_stop` / `ledger_stop` records of the cell); the attempt number still advances so run keys never repeat. Makes review issue 12's "never consumes a rerun" true (it was read off the attempt number). |
| major | dynorch/reconcile.py (key settle) | fixed | (this commit) | re-run canary 2026-09-24: two-equal-reads rule declared a false mismatch during a billing pause; now converge-on-the-ledger with a confirming read (contract §10.36) |
