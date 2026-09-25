# FIX_STATUS: launch_ops fixer

Files owned: `slurm/passk_conductor.sh`, `slurm/passk_conductor.sbatch`, `slurm/passk_cont.sbatch`,
`src/agent_allocation/passk/checkpoints.py`, `src/agent_allocation/passk/run.py`, `docs/passk_cont/RUNBOOK.md`,
`tests/integration/test_passk_conductor_dryrun.py`, `tests/unit/passk/test_checkpoints.py`,
`tests/unit/passk/test_run_drift.py` (new).

## Fix pass 1 status (2026-09-25 16:00 EDT)

### Fixed (each its own commit on `passk-cont`)

| Finding (review file) | Commit | What changed |
|---|---|---|
| [major] permaslug sample treats "/generation not available yet" as drift (ops_budget) | c01f5d9 | `sample_check` fails only on a positive permaslug/provider mismatch. Absent/None/empty records are `unavailable`, retried by the next sample (carried in `gates/gate3_sample_<s>.json`), rc 4 (inconclusive, no DRAIN). Generations younger than 300 s are not sampled. |
| [major] drift check refuses on any listing change (identity_reuse) | a5749d0 | `passk/run.py::classify_endpoint_drift`. Changes that refuse: model, provider tag/name (the tag carries the quantization), a dropped `logit_bias/top_k/temperature/top_p/max_tokens`, a price rise above +10%, a smaller context/completion cap. Other changes are recorded as `drift_noted` and dispatch continues on the adopted lock (node ids frozen). The live listing is saved to `locks/snapshots/drift_probe_*.json`. |
| [minor] transient GET /endpoints crashes passk-run (identity_reuse) | a5749d0 | The probe is retried 5 times, 60 s apart, then refused as "probe unavailable", which is kept distinct from drift. |
| [blocker] conductor never passes `--strace`/`--release` (integration_spec, fence, ops_budget) | ed3b4de | `passk_cont.sbatch analyze` forwards `RELEASE` and `FLAGS` (each split on `+`). The fits run with `FLAGS=--sandbox` (the bwrap fence). `--strace` is gone. Each release is its own job (`RELEASE=v48`, then `RELEASE=t200`). |
| [blocker] no re-extract after `pairs`; no pairs/extract after release (integration_spec, ops_budget) | ed3b4de | Chain: `extract, pairs, extract, e1-lock, e2-fit --sandbox, e5-fit --sandbox, seal`, then per split `release <split>, pairs, extract, score, report`. |
| [major] no START_AT for seal/post; fits are write-once (integration_spec, ops_budget, fence) | ed3b4de, 31cd905 | START_AT accepts every step token (`extract pairs extract2 e1-lock e2-fit e5-fit seal gate4 post release-v48 ... report-t200`), plus `smoke`. An unknown token gives rc 2. A stop names its resume token. |
| [major] background analysis failures masked until generation ends; no `--mail-user` (ops_budget, integration_spec) | ed3b4de | The subshell writes `$STUDY/analysis.failed`. STATUS.md shows **ANALYSIS FAILED**. The main loop checks the chain every poll and calls `stop_main` at once, which drains both shards. MailUser is set from `OWNER_MAIL` via `scontrol update` (FAIL,END) and documented in RUNBOOK (`sbatch --mail-user="$OWNER_MAIL"`). No address is hard-coded. The STATUS tmp file is per PID. |
| [major] wait loops hang or read another job's result (ops_budget) | ed3b4de | `sacct -P` for exactly this job id. Empty or invalid ids are refused. Any state other than a live one counts as ended (OOM, NODE_FAIL, `CANCELLED by N`, BOOT_FAIL, DEADLINE). sbatch is retried 3 times. Every `wait_jobs` is bounded by `WAIT_MAX` (14 h), and every submit is checked. |
| [major] benign causes use up the 3 resubmits (ops_budget) | ed3b4de | DRAIN/STOP are cleared at the start of `gate1`/`smoke`/`main`, which refuses if a passk-run job is already queued. After a killed dispatcher (TIMEOUT/OOM/NODE_FAIL/CANCELLED) the conductor waits `LEASE_WAIT` (960 s) before resubmitting. Only runs that end within 30 min count toward the limit. A STOP present when a run ends is treated as an owner hard stop. |
| [major] grade --follow never monitored; its TERM drains generation (ops_budget) | ed3b4de | Grade-follow jobs are checked every poll and resubmitted when they end. `ensure_grade_follow` runs on analysis/gate4 resumes. `on_term` touches DRAIN only for `canary|smoke|run`. |
| [major] conductor dry-run test freezes the broken stage list (integration_spec) | ed3b4de and later | The test asserts the new stage list, `FLAGS=--sandbox` on both fits, no strace, and one release job per split. New tests cover resume at `seal`/`score-t200`/`smoke`, the background failure stop with DRAIN, an unknown START_AT, and a Gate 4 failure on one split. |
| [minor] Gate 4 couples the two splits (integration_spec) | e48d9f4 | `checkpoints gate4 --split v48|t200` writes `splits.{v48,t200}` to `gates/gate4_<s>_<split>.json`. The conductor releases, scores and reports each passing split and skips a failing one. It stops (and mails) after the report, naming the withheld split. |
| [minor] conductor script read live from the edited worktree (ops_budget) | c6b0cd4 | The sbatch copies the script to `logs/conductor_<jobid>.sh`, records HEAD, and execs the copy. |
| [minor] Gate 1 ignores the grade rc and the dev grader (ops_budget) | c6b0cd4, b84aa1c | The smoke `grade` job must exit 0. `gate1_smoke` has a `smoke_graded` criterion. |
| [minor] canary header says it passes `--account-auto-reload` (ops_budget) | b84aa1c | Only the comment is corrected. The flag needs `live.py`, which is outside my files. |

### Not fixed (outside my files or out of time)

- `score()` gates e2/e5 on the released splits (the known failing `test_release_then_score_primary_family`; WP-M/analysis files). The per-split release chain depends on it: a V48-only `score` must not crash.
- Gate 4 `pairs_complete` criterion (it has to be evaluated after the release extract; analysis side).
- `report` stage renders no E2/E5 results and a stale E0 (integration_spec minor; report code).
- Seal without predictions / seal without `"splits"` (fence.md; seal.py / grade_scope.py).
- Union key reconciliation after a canary retry, and the `canary --account-auto-reload` flag (`live.py`).
- /key re-read after the canary (ops_budget minor).
- Gate 0 check that the live endpoint matches the adopted lock before the canaries (identity_reuse): the classifier exists (`passk.run.classify_endpoint_drift`), but Gate 0 (`passk/gates.py`) does not call it yet.
- A provider 5xx outage permanently fails nodes and trips the 99% rule in Gate 4 (ops_budget minor): no recovery path.
- grade-follow `--time` beyond 12 h: not done, because the partition list includes `mit_normal`. It is resubmitted instead.

### Tests

`tests/unit/passk` + `tests/integration/test_passk_wpg_e2e.py` + `tests/integration/test_passk_conductor_dryrun.py`:
147 passed at 15:54 (before the last three commits), and the touched files' tests passed after each commit.
The final full run result is recorded below.

Final run (16:03 EDT, HEAD 31cd905 plus this file): **155 passed**. That covers `tests/unit/passk`,
`tests/integration/test_passk_wpg_e2e.py` and `tests/integration/test_passk_conductor_dryrun.py`, which has 10 tests.
