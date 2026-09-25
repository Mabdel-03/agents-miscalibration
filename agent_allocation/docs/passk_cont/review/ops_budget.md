# ops_budget review (money, ledger, autonomous ops) of passk-cont 3c0a530

## [blocker] Conductor's `release` stage is submitted without `--release`, so the chain can never reach score/report

- Location: `slurm/passk_conductor.sh:20,280` (`ANALYSIS_POST=(release score report)`, `run_analysis`), `slurm/passk_conductor.sh:66-72` (`submit analyze "" "STAGE=$st"`), `slurm/passk_cont.sbatch:111-113` (`analyze` verb runs `passk-analyze --config --run-root --stage "$STAGE"` with no RELEASE/FLAGS support), `src/agent_allocation/passk/analyze.py:156-157`.
- Problem: `passk-analyze --stage release` calls `parser.error("--stage release needs --release v48|t200")` when `--release` is absent. The conductor never passes it, and `passk_cont.sbatch` has no way to pass it (unlike `passk_analyze.sbatch`, which reads `RELEASE`). The release job exits 2, `run_analysis` calls `stop "analysis:release"`, and the chain dies after Gate 4 and after up to 4 h of held-out grading wait. A resume with `START_AT=gate4` hits the same failure, so the report is never built unattended. The plan also calls for V48 and T200 to be released separately (V48 once `v48_bank` passes Gate 4, T200 once `t200_bank` passes). The conductor has no per-split release at all.
- Evidence: `allocation passk-analyze --config configs/passk_cont_v1.json --run-root <scratch> --stage release` prints `error: --stage release needs --release v48|t200` and exits 2. `tests/integration/test_passk_conductor_dryrun.py` stubs `sbatch` and only checks the stage names, so it cannot catch this.
- Fix: in `passk_cont.sbatch`, make the `analyze` verb append `--release` for each `+`-split `RELEASE` value, and append `FLAGS` split on `+`. In the conductor, submit `submit analyze "" STAGE=release RELEASE=v48+t200`, or two release jobs (v48, then t200), each after its partition's Gate 4 completion. Add a dry-run assertion that the submitted export contains `RELEASE=`.

## [blocker] Fit stages are submitted without `--strace`, so `seal` always refuses and held-out labels are never released

- Location: `slurm/passk_conductor.sh:19,193-200` (`ANALYSIS_PRE=(extract pairs e1-lock e2-fit e5-fit seal)`, submitted with only `STAGE=`), `slurm/passk_cont.sbatch:111-113`, `src/agent_allocation/passk/analyze.py:173-188`, `src/agent_allocation/passk/seal.py:152-157`.
- Problem: `run_seal` needs `logs/strace/e2-fit.log` and `logs/strace/e5-fit.log` ("missing (run the fit stage under strace)" goes into `problems`, and it returns 1). Those logs are written only when `passk-analyze --stage e2-fit|e5-fit --strace` runs. The conductor submits `e2-fit` and `e5-fit` through `passk_cont.sbatch`, which never adds `--strace` and cannot forward FLAGS. So `seal` exits 1, the background pre-seal chain stops at `analysis:seal`, and the parent only notices after all generation ends (`wait $ANALYSIS_PID || stop analysis`). Nothing is ever sealed, and grade-follow never grades V48/T200. The chain stops without reaching Gate 4. The plan's order also has a second `extract` after `pairs` (extract -> pairs -> extract -> e1-lock). The conductor runs `pairs` straight into `e1-lock`.
- Evidence: `seal.py:153-157` checks for the logs. `analyze.py:174-188` writes them only under `--strace`. `passk_conductor.sh:197` passes `STAGE=$st` only. The dry-run test asserts the stage list but not the flags.
- Fix: have `passk_cont.sbatch analyze` honour `FLAGS` (split on `+`, as `passk_analyze.sbatch:23` does). Have the conductor submit `STAGE=e2-fit FLAGS=--strace` and `STAGE=e5-fit FLAGS=--strace`. Put the second `extract` into `ANALYSIS_PRE` if extract must pick up the pair cache (confirm with WP-A). Better still, submit analysis stages through `slurm/passk_analyze.sbatch`, which already has `RELEASE`, `FLAGS` and `--workers`. See the pairs CPU finding below.

## [major] The 30-min permaslug sample treats "/generation not available yet" as drift: it drains both shards and stops the run

- Location: `src/agent_allocation/passk/checkpoints.py:348-357` (`sample_check`: `passed = bool(rows) and ...`), `:399-410` (a failure touches `control/DRAIN` and returns 3), `src/agent_allocation/live.py:427-451` (`fetch_metadata` silently drops ids that stay 404), `src/agent_allocation/api/client.py:235-236` (404 returns None), `slurm/passk_conductor.sh:217-220` (rc 3 calls `stop_main`, which drains both shards).
- Problem: the sample reads `/generation` for the 5 most recently modified parsed results, with 6 attempts 20 s apart. If none of the 5 is available within about 2 min, `fetch_metadata` returns `{}`, `rows` is empty and `passed` is False. The checkpoint then touches DRAIN and exits 3, and the conductor drains both shards and exits. The results are the freshest ones, possibly seconds old, so `/generation` indexing lag is the likely cause, not provider drift. This is a benign event that ends the run and needs a manual resume. The resume also re-runs a leftover-DRAIN cycle (see the DRAIN/STOP finding). A 5xx/network exception, by contrast, gives rc 1 and is ignored, so only the harmless "not yet indexed" case stops the run. The plan's stop condition is "a permaslug change", not "no record".
- Evidence: `sample_check(cfg, paths, {})` gives `rows=[]`, then `passed=False`, then `code=3` plus `drain_sentinel.touch()` (lines 356, 407-410). The dry-run test stubs `sample` to exit 4, so this path is never exercised.
- Fix: return 4 ("not due / inconclusive") when no record was fetched, and fail only when a fetched record carries a different permaslug or provider. Better, sample generations that are at least 5 min old (skip parsed files newer than about 300 s), and require some minimum number of fetched records before judging.

## [major] grade --follow jobs are never monitored or resubmitted; their 12 h limit and SIGTERM trap can drain generation and stall held-out grading

- Location: `slurm/passk_cont.sbatch:26-27,57-62` (`--time=12:00:00`, `--signal=B:TERM@900`, and `on_term` touches `$SHARD_DIR/control/DRAIN` for every verb that has a SHARD, including `grade-follow`), `slurm/passk_conductor.sh:203-252` (`FJ[$s]` submitted once and never checked), `:260-279` (the `START_AT=analysis` and `START_AT=gate4` paths submit no grade-follow but wait up to 4 h for `checkpoint graded`).
- Problem:
  1. A grade-follow job lives from main start until the post-seal held-out grading ends: generation, then the pre-seal analysis chain (fits and seal), then Gate 4 (the duplicate-THINK scan over about 38k raws), then held-out grading. At 11h45 Slurm sends TERM. The trap then touches the shard's `control/DRAIN`. If generation is still running, that drains the shard's run, and one of the 3 resubmits is used up. The grader exits and nobody restarts it, so `stage1_done` can go false for good, Gate 4's `dev_grades` fails, and after the seal the held-out candidates are never graded. The conductor waits 48 x 5 min and then stops ("is grade --follow running?").
  2. The resume paths `START_AT=analysis` and `START_AT=gate4`, which the two blockers above make certain, never submit grade-follow. If the original jobs have ended (12 h limit, or a resume the next morning), Gate 4 held-out grading cannot finish.
  3. A plain `scancel` of a grade-follow job (SIGTERM reaches the batch shell) also drains that shard's generation.
- Fix: only touch DRAIN in `on_term` for the paid verbs (`canary|smoke|run`). In the main loop, check `job_done ${FJ[$s]}` each poll and resubmit it (the grader's `single_grader_lock` makes a duplicate harmless). At the start of `START_AT=analysis|gate4`, submit a grade-follow per shard when `squeue` shows none, and remove a stale `control/STOP` first. Give grade-follow the pi_lhtsai 2-day limit (`--time` override in `submit`).

## [major] Stops are silent: no `--mail-user`, and failures in the background analysis chain are masked until generation ends

- Location: `slurm/passk_conductor.sbatch:8` (`--mail-type=FAIL,END` with no `--mail-user`), `slurm/passk_conductor.sh:229-234,250,253-254` (background `run_analysis` subshell; the parent's `note main "polling..."` overwrites `conductor.state` and STATUS.md every poll).
- Problem:
  1. Earlier studies in this repo set `--mail-user=mabdel03@mit.edu` explicitly (`docs/SCHEMA5_V12_RECOVERY_RUNBOOK.md:2212`). Without it, Slurm mails the bare local user, which may never reach the owner. The plan makes "a gate failure stops the run and notifies the owner" a condition of autonomy. No mail-test log exists under `passk_cont_v1/logs/` (only the analyze, bwrap and strace probes).
  2. When a pre-seal stage fails in the background subshell (which will happen: see the `--strace` blocker), the subshell writes `stopped:analysis:<stage>`. The parent overwrites it with `main` at the next poll (`note` at line 250), keeps polling until generation ends, and only then exits with the generic "pre-seal analysis chain failed". The failing stage name is lost from STATUS.md, and nobody learns of it for about 1-2 h. Both writers also share `$STATUS.tmp`, so they race on the tmp file and the `mv`.
- Fix: add `#SBATCH --mail-user=mabdel03@mit.edu` (and send one $0 test job with the same header). In the parent loop, check `kill -0 $ANALYSIS_PID` each poll. If the chain has exited non-zero, `wait` it and `stop` at once, keeping the failing stage (have the subshell write it to `$STUDY/analysis.failed`). Use a per-PID tmp name in `note`.

## [blocker] The chain never re-extracts after `pairs` and never fills pairs for the released splits: the seal is built on degraded classes, and `score` crashes

- Location: `slurm/passk_conductor.sh:19-20` (`ANALYSIS_PRE=(extract pairs e1-lock e2-fit e5-fit seal)`, `ANALYSIS_POST=(release score report)`), `src/agent_allocation/passk/extract.py:388-411` (`_derived_tables`: pools needing an uncached pair are left out of `classes.parquet` and `readouts.parquet`, listed only in `manifest.classes_pending_pools`), `src/agent_allocation/passk/models/inputs.py:157-166` (`ClassSource.classes` falls back to `exact_classes` without a word when a scope has no rows), `src/agent_allocation/passk/readout.py:119-137` and `src/agent_allocation/selection/plurality.py` (the `MissingPair` from `cache.equivalence` propagates), `src/agent_allocation/passk/score.py:103-109`.
- Problem:
  1. **Pre-seal.** The first `extract` builds `pools/classes/readouts.parquet` while the pair cache holds only the legacy (P0) pairs. Every pool or scope containing a new `e1_fill`, `c48_extra` or `t200_diag` answer text with an uncached pair is dropped from `classes.parquet` and from the E5 dev training table (`readouts.parquet`). `pairs` then fills the cache, but nothing rebuilds the derived tables, so `e2-fit` and `e5-fit` read the stale ones.
     - The E2 Dirichlet and the E5 features/targets then use exact-vote-key classes instead of Math-Verify-merged ones (`ClassSource` counts them in `n_fallback`, which is not a failure).
     - The dev pool-outcome rows go missing in an answer-dependent way: pools with diverse answers are the ones dropped.
     - `seal` then freezes those predictions write-once. The plan's order is extract -> pairs -> extract -> e1-lock.
  2. **Post-release.** `release` re-extracts with V48/T200 but no `pairs` stage runs for the new held-out answer texts. The plan's pair-cache follow job and Gate 4's "pair cache complete" do not exist. So `score` hits `cache.equivalence(...)` on uncached pairs, `MissingPair` propagates out of `readout_pools` (there is no catch there, only in `class_rows`), and `score` exits 1. The chain cannot reach `report`.
- Evidence: `extract.py:395-406` computes `missing` and filters `ready = pools[~pools.pool_id.isin(missing)]`. `inputs.py:164-166` has the silent fallback. `readout.py:134-136` has no `try`. `plurality.py` docstring: "exceptions from the callable propagate (fail loudly)". `pairs.py:201` raises `MissingPair`.
- Fix:
  - `ANALYSIS_PRE=(extract pairs extract e1-lock e2-fit e5-fit seal)`.
  - `ANALYSIS_POST=(release pairs release score report)`, with RELEASE passed as in the release blocker, or `extract --release` again after `pairs`.
  - Make `e2-fit`/`e5-fit` (or `seal`) refuse when `analysis_inputs/manifest.json:classes_pending_pools` is non-empty.
  - Make `score` refuse (exit 2) on missing pairs with a clear message rather than crash.
  - Optionally run a `pairs` follow job during generation, as the plan intended, so the tail is short.

## [major] Conductor wait loops can hang for the whole 2-day allocation, or read another job's result, on an sbatch hiccup or an OOM

- Location: `slurm/passk_conductor.sh:73` (`job_state`: `sacct -j "$1" -n -X -o State | head -1`), `:80` (`job_done` pattern), `:84` (`while ! job_done "$j"; do sleep 60; done`, no timeout), `:149,160,164,247` (the canary retry, smoke, grade and run resubmission store `$(submit ...)` without checking it is non-empty).
- Problem:
  1. **Empty job id.** `submit` returns an empty string when `sbatch` fails (a slurmctld timeout, or a QOS/assoc submit limit; stderr goes to `conductor_sbatch.err`). `sacct -j ""` does not fail: it lists the user's jobs of the day. Verified here: `sacct -j "" -n -X -o State | head -1` printed `RUNNING`. So `wait_jobs ""` follows some unrelated job. If that job is RUNNING (for example the conductor itself or an interactive allocation), the loop spins until the conductor's 2-day limit. For a resubmitted `run`, the shard is never resumed and no mail goes out until then. If that job is COMPLETED 0, the step reads as passed.
  2. **Truncated state.** sacct's default `State` column is 10 characters wide (`CANCELLED by N` prints as `CANCELLED+`, seen above). `OUT_OF_MEMORY` therefore prints as `OUT_OF_ME+`, which the `job_done` pattern never matches, so an OOM-killed job leaves the conductor waiting forever. The analysis stages run through `passk_cont.sbatch` with `--mem=16G`, and the `run`, grade and pairs jobs have the same limit. `BOOT_FAIL` and `DEADLINE` are not matched either.
- Evidence: the `sacct -j ""` output above. `job_done "OUT_OF_ME+"` returns 1 when run in bash.
- Fix: in `job_state`/`job_rc` use `sacct -j "$1" -n -X -P -o State,ExitCode` (parsable, no truncation) and refuse an empty `$1`. Treat any state other than PENDING/RUNNING/REQUEUED/SUSPENDED/CONFIGURING/COMPLETING as done. Check `[ -n "$j" ]` after every `submit` and retry sbatch 3 times, 60 s apart, before `stop`. Put an upper bound on `wait_jobs` (for example 14 h for a step), after which it stops and mails.

## [major] Benign causes use up the 3 run resubmissions: a leftover DRAIN/STOP and a stale dispatcher claim

- Location: `slurm/passk_conductor.sh:62-64` (`stop_main` touches DRAIN on both shards and never clears it), `:203-209` (`START_AT=main` submits runs without removing DRAIN or STOP), `:241-248` (resubmission removes only DRAIN, 5-min cadence, at most 3 times), `src/agent_allocation/runtime/scheduler.py:617-619` (DRAIN or STOP present at start ends the run with `control_sentinel_present`, rc 3 after the paid forced Gate-0 probes), `:740-748` + `runtime/owner.py:40-41` (a claim from another host is never "dead", so a SIGKILLed dispatcher's claim blocks a new one for `lease_stale_seconds`=900 s).
- Problem:
  1. **Leftover DRAIN.** Every documented resume after a main-phase stop (`START_AT=main`) starts the runs with DRAIN still present. Both exit at once (probes paid), and one resubmission is burnt per shard before the conductor removes DRAIN.
  2. **Leftover STOP** (the RUNBOOK hard stop, or the conductor's own final STOP). It is never removed. The run exits rc 3 at each of the 3 resubmissions, the conductor then stops with "stopped early 3 times", and the new grade-follow exits after one pass.
  3. **Stale dispatcher claim.** A run killed without a clean exit leaves `dispatcher_owner`/heartbeat behind: a TIMEOUT kill when the drain outlasts the 900 s signal lead (16A p99 latency is 850 s and `read_timeout` is 900 s), an OOM, a NODE_FAIL, or a plain `scancel`. The replacement starts on another node within about 5-10 min and raises `DispatchRefused`. That exits rc 1 and is resubmitted again 5 min later. Two or three refusals exhaust `RESUB` and stop the run for a transient condition.
- Evidence: the code paths above. The dry-run test never exercises resubmission (HANDOFF_wpG "Remaining 7").
- Fix: at the top of `START_AT=main` (and `gate1`), `rm -f` DRAIN and STOP on both shards after checking that no passk run job is queued. Before a resubmission, wait until `now - dispatcher_heartbeat > lease_stale_seconds` (or pass a takeover flag once `sacct` confirms the old job ended). Count only non-benign endings (rc 3 with a gate or budget reason) toward the limit of 3. Log the run's `stop_reason` from `run_decisions.jsonl` in STATUS.md so the owner can tell budget exhaustion from a drain.

## [minor] After the canary, nothing checks /key: the $150 owner cap rests on the shard ledgers alone, while the key allows about $218

- Location: `src/agent_allocation/passk/run.py:204-207` (the /key snapshot is read once at run start), `src/agent_allocation/passk/checkpoints.py:292-320` (Gate 4 has no union key reconciliation), `:105-126` (Gate 1 smoke has no `/key` reconciliation), HANDOFF_wpG "Remaining 5".
- Problem: the key now has limit 400 and about $218.68 remaining, well above the $150 hard cap. The plan assumed about $155, so the key itself was the backstop. The only thing keeping spend to 87 + 63 is each shard's own ledger ceiling. The canary window is the last point where ledger and `/key` are reconciled. Gate 2 compares ledger to forecast, never ledger to key. Gate 4 does not do the "key reconciles across both shards as a union" check at all. A ledger undercount (unreconciled ERROR_ENVELOPE/ambiguous exchanges, abandoned exchanges after a SIGKILL, or probes outside the ledger) would stay invisible until the owner looks at the account. Gate 3's "unresolved exposure > $2 stops after 30 min" is not implemented either (the dispatcher only pauses).
- Fix: have Gate 2 (each evaluation) and Gate 4 compute `key_usage_now - min(canary_before key usage over shards)` against the sum of shard ledger settled since their `canary_before`. Stop (drain both shards) if the key delta exceeds the ledger by more than max($0.50, 2%). Alternatively, ask the owner to lower the key limit so that `limit_remaining` is about $155 again, which restores the hard backstop at no cost.

## [minor] The conductor script is read live from the worktree that will be edited and committed during the run

- Location: `slurm/passk_conductor.sbatch:15-16` (`bash "$CODE/slurm/passk_conductor.sh"`), and the plan's Workflow 3 ("code freeze during generation"; `gates/analysis_review.json` must equal HEAD at seal time).
- Problem: bash reads a script incrementally, by byte offset, while it runs. Slurm snapshots only the `.sbatch` wrapper, not the `.sh` it calls. The analysis blockers above can only be fixed by editing `passk_conductor.sh`/`passk_cont.sbatch` and committing (to move HEAD for the seal) while the conductor is still polling. Such a commit changes the bytes under the running bash, and the result is undefined: skipped or garbled commands, possibly right at `stop_main` or resubmission.
- Fix: in the `.sbatch`, copy the script to `$TMPDIR` (or `$STUDY/logs/conductor_<jobid>.sh`) and `exec bash` the copy, recording the commit it came from. The rule then becomes: a code change takes effect only through a stop and `START_AT=` resume.

## [minor] The canary verb does not pass `--account-auto-reload`, although the sbatch header says every paid verb does

- Location: `slurm/passk_cont.sbatch:18-19` (the comment), `:85-87` (`canary` has no flag, and `allocation canary` has none: `live.py:366-370`), `live.py:593` (`ceiling = min(active_ceiling(max_usd, settled_before, before), settled_before + 0.50)`, where `account_auto_reload` defaults to False).
- Problem: the canary ceiling is also capped by `account_balance - $1`. Legacy snapshots show the auto-reloading account at about $24-59 (total_credits minus total_usage). If the balance sits near the reload threshold when the canary starts, the canary's $0.50 window can fall below what it needs. It then stops with budget exhausted, Gate 1 fails, and the conductor stops. This is a low-probability stall, and the header comment is inaccurate.
- Fix: thread `account_auto_reload=True` through `run_canary`'s `active_ceiling` (for example a `--account-auto-reload` flag on `canary`), and pass it in `passk_cont.sbatch`.

## [minor] Gate 1 smoke never checks the dev grader, and the conductor ignores the grade job's exit code

- Location: `slurm/passk_conductor.sh:163-165` (`submit grade`, `wait_jobs`, no rc check), `src/agent_allocation/passk/checkpoints.py:105-126` (`gate1_smoke` criteria: terminal, exact, cache, pin, permaslug; no grades).
- Problem: the plan's Gate 1 runs the §15.5 smoke "through the new verb and the dev grader". A broken grader (text grader blocked, missing `private/` gold, a fence misconfiguration that blocks dev ids) passes Gate 1 unseen. It surfaces only later: `stage1_done` never becomes true, the analysis waits for the end of generation, and then `e1-lock` exits 2 ("not graded yet") and the chain stops after all the money has been spent.
- Fix: add a `smoke_graded` criterion (every smoke candidate has a grade and `grade_state != missing`) and require `job_rc(GJ[s]) == 0`.

## [minor] A transient canary retry can make the union key reconciliation fail for good

- Location: `slurm/passk_conductor.sh:144-150` (a canary retry after a 10-min sleep), `src/agent_allocation/live.py:572-583` (each `run_canary` writes a fresh `canary_before` snapshot and `canary_before.json`), `:640-666` (`shard_key_window`: `key_delta = max(afters) - min(befores)`; settled counts each shard only since its latest `settled_before`).
- Problem: when a shard's canary is retried, its own `before` moves forward, but the sibling's earlier `before` still opens the union window. So the key delta includes the failed first attempt's spend (probes, plus any canary calls already sent), while the ledger delta excludes it (it is before the new `settled_before`). With tolerance max($0.01, 5% of about $0.75), a first attempt that got as far as sending canary calls fails `canary --evaluate-only` on every re-evaluation. The conductor stops at gate1.
- Fix: keep the first `canary_before` per shard (write-once) across retries, or reconcile from the earliest before snapshot of each shard.

## [minor] A few minutes of provider 5xx outage permanently fails nodes and trips Gate 4's 99% rule; the conductor has no way to recover

- Location: `src/agent_allocation/runtime/scheduler.py:95-96,1354-1363` (backoff 1 s, then 2 s; `max_transport_attempts`=3; then `mark_exhausted`, terminal), `src/agent_allocation/passk/checkpoints.py:301-303` (≥ 99% succeeded per partition), `slurm/passk_conductor.sh:245` (3 resubmissions, 5 min apart).
- Problem: 5xx and connect failures are retryable, but a node uses up its 3 sends within about 5 s plus latency. Every node admitted during a provider outage becomes a terminal failure, and failures are never replaced. For `t200_bank`, 62 failed nodes per partition (1%) are enough to fail Gate 4, and the chain stops before release with the money spent. forced_v2 had zero terminal failures, so this only bites in an outage, but a 48-96-wide in-flight window fills 62 quickly. The run also keeps going, because the circuit breaker opens only on auth, payment or mismatch.
- Fix (conductor side, no harness change): have Gate 3 count terminal `attempts_exhausted` nodes per trailing 10 min and drain the shard (without stopping) when they exceed about 10, then resubmit after a 15-min backoff. Gate 4 should list exhausted nodes separately so the owner can decide. Do not replace them.

## [major] No resume path recovers from a seal refusal, the most likely stop point: the fit stages are write-once and there is no `START_AT=seal`

- Location: `slurm/passk_conductor.sh:8,260` (resume menu `gate0|gate1|main|analysis|gate4`; `analysis` re-runs the whole `ANALYSIS_PRE`), `src/agent_allocation/passk/models/e2.py:135-137` and `e5.py:149-151` (`FileExistsError` when `sealed/{v48,t200}/predictions.parquet` exists; `analyze._stage_models` never passes `overwrite`), `src/agent_allocation/passk/seal.py:126-143` (seal needs `gates/analysis_review.json` with `commit == HEAD` and a clean tree; right now `passk_cont_v1/gates/` holds only `gate_s.json`).
- Problem:
  1. `seal` refuses whenever any of these holds:
     - the strace logs are absent (always, today: see the `--strace` blocker);
     - `analysis_review.json` is missing (Workflow 3 must finish before stage 1 completes, about launch+45 min);
     - HEAD has moved since the review (any fix commit, including one to the conductor);
     - the tree is dirty.
  2. After such a refusal, `START_AT=analysis` re-runs `e2-fit`, which raises `FileExistsError` because the predictions from the first pass exist, so it stops again. `START_AT=gate4` skips the seal entirely, grade-follow never grades V48/T200, and after the 4 h `graded` wait it stops.
  3. The only way forward is hand work: run `seal` alone, or delete `sealed/v48|t200` and re-fit under strace. Nothing in the RUNBOOK or STATUS.md says so.
  4. Deleting and re-fitting sealed artifacts by hand after the tranche has progressed is exactly the kind of step the fence is meant to make auditable.
- Fix:
  - Add `START_AT=seal` (run `seal` and the stages after it).
  - Make `run_analysis` skip a fit stage whose `sealed/<split>/predictions.parquet` already exists and whose strace log exists. If the log is missing, refuse with a message naming the exact recovery command.
  - Record in STATUS.md which seal precondition failed. Have `seal` print the problems to stdout too, since they currently go only to the job's stderr.
  - Write `gates/analysis_review.json` (the Workflow 3 commit) before launching the conductor, and freeze the worktree from then on. Code fixes afterwards should go to a separate checkout so HEAD stays equal to the reviewed commit.

## [minor] The 1A unit cost is the calibration figure, not the forced_v2 main mean: this leaves Gate 2 only about 12% headroom on Parasail

- Location: `src/agent_allocation/passk/forecast.py:18-23` (`1: 0.00381` Mancer, `1: 0.00335` Parasail), `src/agent_allocation/passk/checkpoints.py:159-166` (a stop when the mean is more than 25% off the plan for any budget with ≥ 10 calls).
- Problem: the forced_v2 `main` results (read-only sqlite, `usage.actual_cost_usd`) average $0.00413 per 1A node on Mancer and $0.00369 on Parasail. Reweighted to the T200 mixture (0.1/0.1/0.5/0.3) they average $0.00410 and $0.00373: 1.076x and 1.113x the plan. The plan's 1A figures match the calibration and canary partitions ($0.00376/$0.00331). 2A-16A match main exactly. So the upper band for the ~5,000 1A calls (`v48_bank`, `t200_bank`, `t200_diag`) sits only about 14% (Mancer) and 12% (Parasail) above the realistic mean. On the new items, 50% of which are Omni (legacy Parasail Omni 1A: $0.00398), slightly longer ANSWERs would trip a Gate 2 stop on a benign cost mix. The forecast also under-counts 1A by about $1.7 in total.
- Fix: before Gate 0, set `UNIT_COST_USD[...][1]` to the main means ($0.00413 / $0.00369). The forecast becomes about $99.3, still ≤ $115 and within the soft slices. Alternatively, apply the ±25% check to the per-benchmark mixture of the nodes actually settled.

## Checked, no defect found (for the lead)

- **Ceiling arithmetic.** `run.py:204-207`: `min(active_ceiling(87|63, settled, snap, auto_reload=True), slice)`. Both slices together (150) are ≤ `limit_remaining` (218.68) − 3. Each shard's ceiling is its own slice. The derived shard configs give 87/82 and 63/60, Mancer start 48/rps 6, Parasail max 40/rps 8.
- **Gate 2/3 thresholds against forced_v2 history** (read-only replay of the legacy sqlite and raws):
  - valid-final is about 0.99 per benchmark at 1A;
  - `usage.reasoning_tokens == budget` on 8,342 + 5,481 of 8,342 + 5,481 accepted nodes;
  - Gate 3 would not have fired: 0 closed attempts and 4 stream cuts on Mancer, none on Parasail;
  - 2A-16A unit costs match main to within 0.2%;
  - the `/generation` record shape (`model` plus `provider_responses[].model_permaslug`) matches `sample_check`/`gate1_smoke`.
- **Drain handling.** A drain or leftover sentinel ends `passk-run` with `stop_reason` drain or `control_sentinel_present`, so rc 3 and resubmission (not a false "done"). `run_waited` recovers the child's rc after the TERM trap.
- **Slurm limits.** pi_lhtsai MaxTime is 2-00:00:00, mit_normal is 12:00:00, and the user is in `orcd_rg_par_pi_lhtsai`. The 12 h step limit fits both partitions.
- `tests/integration/test_passk_conductor_dryrun.py`: 4 passed. It stubs sbatch/sacct and never checks exported flags (RELEASE/FLAGS), resubmission, the sample path or resume.
