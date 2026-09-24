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
| 16 | major | `dynorch/cli.py` | fixed | `ba37752`, `<sha16>` | the store-always-consulted rule in `Runner.run_specs` plus defence in depth: `open_run` refuses a cell that already has a terminal record or journaled calls, and a `LedgerError` (a duplicate attempt id) is answered 500 `internal_error` with the token grant RELEASED — it used to fall into the generic BaseException branch and charge a call that never left the process. |
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
