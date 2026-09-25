# Fence / leakage review of passk-cont @ 3c0a530

Reviewer lens: held-out fence and leakage.

## [blocker] Conductor never runs the fit stages under strace, so Gate P can never seal (and the fits are then write-once)

- Location: `slurm/passk_conductor.sh:193-200` (`run_analysis` submits `analyze` with only `STAGE=$st`),
  `slurm/passk_cont.sbatch:111-113` (VERB=analyze runs `passk-analyze --config --run-root --stage "$STAGE"`, no FLAGS),
  `passk/analyze.py:173-188` (strace only with `--strace`), `passk/seal.py:153-157` (missing log => refused).
- Problem: `--strace` is only reachable through `slurm/passk_analyze.sbatch` (FLAGS), which the conductor does not use.
  The autonomous chain runs `e2-fit`/`e5-fit` untraced, writes the write-once `sealed/v48/*`, `sealed/t200/*`, then
  `seal` refuses (`logs/strace/e2-fit.log missing`) -> rc 1 -> `stop analysis:seal`. Recovery needs a human, and a
  re-run of the fit under `--strace` hits the write-once refusal (see next finding) -- whose clean log then satisfies
  the seal anyway.
- Evidence: `grep -n FLAGS\|strace slurm/*.sh slurm/*.sbatch` -> only `passk_analyze.sbatch`; the conductor dry-run
  test (`tests/integration/test_passk_conductor_dryrun.py:101-102`) asserts the stage names only, never the flag.
- Fix: in `passk_cont.sbatch` VERB=analyze append `--strace` when `STAGE` is `e2-fit|e5-fit` (or honour `FLAGS`
  split on '+'), and have `run_analysis` pass `FLAGS=--strace` for the fit stages; extend the dry-run test to assert it.

## [major] Conductor `release` stage is submitted without `--release`, so the post-seal chain always stops

- Location: `slurm/passk_conductor.sh:20,280` (`ANALYSIS_POST=(release score report)`), `slurm/passk_cont.sbatch:111-113`,
  `passk/analyze.py:156-157` (`parser.error("--stage release needs --release v48|t200")` -> exit 2).
- Problem: no RELEASE/--release is ever passed on the conductor path, so `analysis:release` exits 2 and the chain stops
  before score/report every time (fail-closed, but the autonomous run never finishes; and plan says V48 and T200 are
  released separately, each after its own Gate 4).
- Fix: pass `RELEASE=v48+t200` (or one job per split) through the sbatch into `--release v48 --release t200`; make the
  dry-run test assert the flag.

## [major] strace is blocked on the compute nodes, and the seal silently accepts the audit-hook fallback, which cannot see pyarrow reads

- Location: `passk/analyze.py:174-187` (fallback to `_audited_models` on "Operation not permitted"),
  `passk/analyze.py:121-134` (audit hook logs only the `open` event), `passk/seal.py:92-115` (`ok = opened > 0 and bad == 0`,
  `mode` recorded but never checked), `passk/seal.py:158-162`.
- Problem: the probe job settled it: `logs/strace_probe_23919235.out` on node4003 shows
  `strace: ptrace(PTRACE_TRACEME, ...): Operation not permitted` (same on the login node). So every conductor fit
  will run under the audit hook. That hook does not see C-level opens. I tested this with the project venv:
  `pq.read_table(p)`, `pyarrow.dataset.dataset(p).to_table()`, `pd.read_parquet(p, filesystem=LocalFileSystem())` and
  `os.listdir(dir)` on a file under `artifacts/shards/.../grades/` are **NOT SEEN**. Only `pd.read_parquet(str)` and
  `os.open` are seen. Python's own module imports always produce opens, so `opened > 0` holds and the seal writes
  `ok: true` with `mode: audithook`. The Gate P "strace shows no forbidden opens" check is therefore a check of
  Python-level opens only, and nothing downstream refuses because of that.
- Fix: in `run_seal`, refuse `mode == "audithook"` unless an explicit, operator-written waiver
  (`gates/strace_waiver.json` with reason + commit) exists, and copy the waiver into the seal. Also widen the hook:
  record the `os.listdir`/`os.scandir`/`glob.glob` events, and wrap `pyarrow.fs.LocalFileSystem.open_input_file`/
  `open_input_stream` and `pyarrow.parquet.read_table` so they log paths. Better still (the bwrap probe exists), run the
  fit stages under `bwrap` with only `analysis_inputs/`, `sealed/`, `artifacts/manifests/splits.json` and the code/venv
  bound, so a forbidden read fails with ENOENT instead of depending on a log.

## [major] Gate P's open-log check is not tied to the run that produced the sealed artifacts (trivially passed by a failing re-run)

- Location: `passk/seal.py:153-162` (only log existence, `opens > 0`, no forbidden path),
  `passk/analyze.py:181-184` (the traced child's return code is returned but not recorded),
  `passk/models/e5.py:145-151` and `models/e2.py:131-137` (inputs are loaded **before** the write-once
  `FileExistsError`).
- Problem: the seal never checks that the logged run exited 0 or that it wrote the files under `sealed/v48|t200`. A
  concrete path follows from the blocker above. The conductor fits run untraced, possibly reading anything, and write
  `sealed/*`. The seal then refuses. The operator re-runs `--stage e5-fit --strace`: it opens `analysis_inputs/*`
  (opens > 0, none forbidden) and raises `FileExistsError`. The log is clean, so the next `--stage seal` passes and
  seals artifacts that no traced run produced. The same happens after a crash midway through `_write` (e2 writes
  `predictions.parquet` first, non-atomically): the next run refuses, and the partial set is sealed.
- Fix: have the traced/audited stage write `logs/strace/<stage>.done.json` with `{rc, started_at, finished_at,
  outputs: {path: sha256}}` after a successful fit, and have the seal require `rc == 0`. Every `sealed/v48|t200` artifact's
  sha must equal the one recorded there, and the log must contain the O_WRONLY/O_CREAT open of each output. Also do
  the write-once check before loading inputs, and write each prediction set atomically into a temp dir that is then
  renamed.

## [major] The seal need not contain any prediction, and it releases BOTH splits at once

- Location: `passk/seal.py:174-179` (only requires *some* file under `sealed/` other than the inputs snapshot;
  `sealed/e1_selection.json` from `e1-lock` satisfies it), `passk/seal.py:183-185` (no `splits` key written),
  `passk/grade_scope.py:75-80` (no `splits` key => every held-out split released), `passk/labels.py:73-77` (any verifying
  seal => nothing blocked).
- Problem: if `e2-fit` or `e5-fit` dies after its log is written, or never writes (for example `NotReady` inside
  the model), `seal` still writes `SEAL_predictions.json` with only `e1_selection.json` (+ whatever exists). Every
  `grade --follow` then grades `v48_bank`, `t200_bank` and `t200_diag` within one poll. `extract --release` works too, and
  a later `e5-fit` (before the release re-extract adds labels) writes `sealed/t200/predictions.parquet`. That file is
  not in the seal's `artifacts`, and `score.py:125` reads it without checking the seal's hash for it
  (`labels.join_labels` / `join_sealed`, the hash-checked join, is dead code: nothing calls it or
  `write_held_out_predictions`).
- Fix: `run_seal` must require the complete expected set (`sealed/e1_selection.json`, `sealed/v48/{predictions,
  crossover,population}.parquet`, `sealed/t200/{predictions,features}.parquet`, each model manifest). Write
  `"splits": [...]` for only the splits whose predictions are present. In `score` (and `e2/e5.score`), load predictions
  only through `labels.join_labels` / a check that `seal["artifacts"][rel] == sha256(file)`.

## [major] The Gate P static scan passes the most natural ways of reading fenced data

- Location: `passk/seal.py:34-36` (`_FORBIDDEN_TEXT` needs `"shards/"` / `"grades/"` **with** a trailing slash in one
  literal), `passk/seal.py:70-86`.
- Problem: pathlib segment joins (`Path(r) / "artifacts" / "shards" / s / "grades"`), `importlib.import_module("...lab" +
  "els")`, `getattr(mod, name)` and every forbidden module not on the list all pass. The list is `passk.splits`
  (reads `artifacts/private` golds for Gate S), `passk.checkpoints`, `passk.score`, `passk.e0`, `grading.*` and
  `passk.banks` (`study_root`, jobs manifests).
- Evidence: I wrote a scratch tree with `models/leak.py` doing
  `Path(run_root)/"artifacts"/"shards"/"mancer"/"grades"` + `.rglob("*.json")` and
  `importlib.import_module("agent_allocation.passk." + "lab" + "els")`. `seal.static_scan(scratch)` returned
  `{'findings': {}, 'ok': True}`. With the audit-hook fallback (strace blocked on compute nodes, see above), neither
  `rglob` (listdir/scandir) nor pyarrow reads are logged, so neither mechanical check would catch this module.
- Fix: flag string constants equal to `"shards"`, `"private"`, `"grades"`, `"parsed"`, `"raw"` (not only the
  slash forms), and flag any use of `importlib`, `__import__`, `getattr` on a module, `open`/`os.*` with a
  non-`analysis_inputs`/`sealed` path. Better: allow-list imports (numpy, scipy, pandas, stdlib math/json/hashlib/
  dataclasses/pathlib, `agent_allocation.passk.models.*`, `agent_allocation.selection.{keys,plurality}`,
  `agent_allocation.contracts`) and fail on anything else. Also scan the modules those import (`selection/*`,
  `contracts.py`).

## [major] Re-running `--stage seal` permanently un-verifies the existing seal (re-fences everything), and the pre-seal chain cannot be resumed

- Location: `passk/seal.py:169-173` (rewrites `sealed/inputs_manifest_at_seal.json` **before** the write-once check at
  `:186-189`); `passk/models/e2.py:136-137`, `models/e5.py:150-151` (`FileExistsError` on re-run);
  `slurm/passk_conductor.sh:19,254-260` (`START_AT=analysis` and a `main` resume re-run all of
  `ANALYSIS_PRE = extract pairs e1-lock e2-fit e5-fit seal`).
- Problem: (a) the snapshot is one of the seal's `artifacts`. A second `seal` run after any `extract` (every extract
  writes a new `created_at`, and a release extract changes `releases`/`labelled_partitions`) overwrites it, then
  refuses because the seal exists. From then on `verify_seal` is None for good: `grade --follow` stops grading the held-out,
  `extract` drops the sticky releases, `score` raises PermissionError, and the seal cannot be rewritten (write-once).
  (b) Today any resume through `ANALYSIS_PRE` after the fits ran stops at `analysis:e2-fit`
  (`FileExistsError`, rc 1), so the documented `START_AT=analysis` resume cannot get past a partial chain. The
  obvious fix for (b), idempotent fits, would then reach `seal` and trigger (a).
- Evidence (scratch root, `run_seal(require_git=False)`): first seal rc 0 and `grade_scope.verify_seal` ok. After
  rewriting `analysis_inputs/manifest.json` as a release does: second seal rc 1 ("exists; a seal is written once"),
  and `verifies after re-run: False`.
- Fix: at the top of `run_seal`, if `SEAL_predictions.json` exists, return 0 when it verifies (1 otherwise) without
  writing anything. Write the snapshot to a temp name and move it into place only together with the seal. Make
  `e2-fit`/`e5-fit` return 0 (no-op) when their complete sealed output set already exists and matches their
  `.done.json`. Have the conductor skip `ANALYSIS_PRE` stages already done when a verifying seal exists.

## [minor] Two divergent fence implementations survive the merge; the contract's `labels.py` one is not the one in use

- Location: `passk/grade_scope.py:47-138` (used by `grading/worker.py:512-516`, `extract.py:288-294`,
  `score.py:32-40`, `checkpoints.py:294,325`) versus `passk/labels.py:50-86` (used only by the dead
  `labels.join_labels` and by tests).
- Problem: contract §5.6 and the plan ("`passk.labels.blocked_node_ids`") name `labels.py`. The two versions differ.
  `grade_scope` honours `seal["splits"]` and `cfg.passk.heldout`, finds the study root through
  `banks.artifacts_root`, and drops malformed jobs lines. `labels` releases every split on any verifying seal,
  hard-codes the partitions, uses `resolve().parents[2]`, and raises on a malformed line. They agree today only
  because `seal.py` never writes `splits`. A test or reviewer checking `labels.blocked_node_ids` is not checking the
  grader.
- Fix: make `labels.verify_seal`/`blocked_node_ids` thin re-exports of `grade_scope` (or the reverse), delete the
  other body, and add one test that the worker, extract and score all resolve to the same function object.

## [minor] The grader's fence fails open when no passk config is found

- Location: `grading/worker.py:501-516` (`_scope_config` -> None or a config without `passk` -> `(frozenset(), None)`:
  grade everything).
- Problem: a hand-run `allocation grade --run-dir <passk shard>` (the RUNBOOK tells operators to run verbs by hand) with
  a wrong `--config`, or on a shard whose `state/shard_config.json` is missing, grades `v48_bank`/`t200_bank`/`t200_diag`
  before the seal. The conductor path passes `--config $SHARD_CFG`, so this is an operator hazard, not a conductor bug.
- Fix: in `cmd_grade`, if the run dir's `manifests/jobs/` contains any of `v48_bank|t200_bank|t200_diag.jsonl` and the
  resolved config has no `passk` block, refuse (`EXIT_BLOCKED`) instead of grading unrestricted.

## [major] The conductor seals E2/E5 predictions built on classes computed before the new pairs exist (no second extract)

- Location: `slurm/passk_conductor.sh:19` (`ANALYSIS_PRE=(extract pairs e1-lock e2-fit e5-fit seal)`), and the dry-run test
  that freezes this order (`tests/integration/test_passk_conductor_dryrun.py:101-102`). Also
  `passk/extract.py:396-421` (`class_rows` puts pools with a missing pair in `missing`, which get no class rows), and
  `passk/models/inputs.py:160-167` (a scope with no rows silently falls back to exact vote keys, `n_fallback += 1`).
  Nothing in `models/e2.py`/`e5.py` checks `manifest.json["classes_pending_pools"]`.
- Problem: WP-A's handoff (item 4) requires `extract -> pairs -> extract` so that `classes.parquet` has no pending
  pools. The conductor runs a single extract **before** the pair fill. The P0 cache covers legacy answers only, and
  the new `e1_fill`/`c48_extra`/`t200_diag` MATH answers need new pairs. So every scope that contains them has
  no rows in `classes.parquet`, and the E2 (`e2fit:<item>`) and E5 (`diag5` T200 features, dev pool outcomes) fits
  quietly use exact-key classes, without the Math-Verify merges. `readouts.parquet`/`dev_training_readouts.parquet`
  also leave those pools out. The outputs are write-once and then sealed, so the preregistered predictions are fixed
  on the wrong class definition. The only trace is `scopes_exact_fallback` in the model manifests (WP-M:
  "should be 0 in production").
- Fix: make `ANALYSIS_PRE=(extract pairs extract e1-lock e2-fit e5-fit seal)` and update the dry-run test. In
  `e2/e5.fit_and_predict`, refuse (exit 2) when `analysis_inputs/manifest.json["classes_pending_pools"] > 0`, and assert
  `cls.n_fallback == 0` before writing `sealed/*`.

## [major] The post-release chain never fills the held-out pairs, so `score` raises `MissingPair`

- Location: `slurm/passk_conductor.sh:20` (`ANALYSIS_POST=(release score report)`), `passk/score.py:106-109`
  (`readout_pools(..., cache=PairCache(...))` over the L96/V48/T200 pools), `passk/pairs.py:195-202` (`lookup` raises
  `MissingPair` on an uncached pair), `passk/analyze.py:83-99` (pairs are enumerated from `analysis_inputs/draws.parquet`,
  which holds no bank rows before the release).
- Problem: the `v48_bank`/`t200_bank`/`t200_diag` MATH answers only enter `draws.parquet` at the release extract. No
  `pairs` stage runs after that, so the first V48/T200 pool with two distinct MATH answer texts raises in `score`: rc 1,
  `stop analysis:score`. This comes on top of the missing `--release` finding above, so the chain stops twice.
- Fix: make `ANALYSIS_POST=(release pairs extract score report)`. The second extract keeps the sticky release, and
  its `classes_pending_pools` must be 0. `score` should refuse with exit 2 and a clear message while
  `classes_pending_pools > 0`, not fail on a traceback.

## [minor] The open-log check compares raw, unnormalised path strings (`..`, relative paths and symlinks pass)

- Location: `passk/seal.py:104-113`.
- Problem: the check is a `startswith` against the resolved forbidden roots on the path exactly as logged. The
  path is not normalised, relative paths are not resolved against the process cwd (the sbatch `cd`s to `$CODE`),
  and symlinks are not followed, although strace logs the path as given. This matters less than the audit-hook
  finding, but it means the strace mode cannot be trusted either.
- Evidence: I fed a three-line log to `check_strace`. Only the literal `.../artifacts/shards/...` open was flagged.
  `$R/analysis_inputs/../artifacts/shards/mancer/grades/ab/x.json` and
  `../../agents_scaling_results/passk_cont_v1/artifacts/private/aime.jsonl` gave
  `forbidden_opens: 1` of 3 opens.
- Fix: for each logged path, compute `os.path.realpath(os.path.join(cwd_of_trace, p))` (record the cwd in the log
  header) and test that against the forbidden roots. Better, also invert the check into an allow-list: only
  `analysis_inputs/`, `sealed/`, `catalog/`, `artifacts/manifests/splits.json`, the code tree, the venv, and
  `/usr`/`/lib*`/`/proc`/`/sys`/`/etc`/`/dev`.

### Evidence addendum for "The seal need not contain any prediction"
Scratch root with only `sealed/v48/predictions.parquet` (no `sealed/t200/*`, no `e1_selection.json`), two clean logs,
`run_seal(require_git=False)` -> `{"seal": "written", "n_artifacts": 2}` and `grade_scope.verify_seal` returns the seal.
With no `splits` key in it, `grade_scope._scope` unblocks `v48_bank`, `t200_bank` and `t200_diag` on the next poll.

## [minor] Labels are never frozen: `grade --follow` keeps re-grading timeouts after the E1 lock, after the seal and after the release extract

- Location: `grading/worker.py:16,48,229,243` (a Math-Verify timeout is `EVALUATOR_ERROR`, which is not in
  `TERMINAL_GRADE = {GRADED, INVALID_OUTPUT}`, so every pass redoes it); `slurm/passk_conductor.sh:280-281` (`control/STOP`,
  which ends the follow jobs, is touched only after `release score report`).
- Problem: an `e1_fill`/`c48_extra` unknown can turn correct or wrong after `e1-lock` and after the sealed fits read
  `analysis_inputs/`. A held-out unknown can flip between the release extract and `score`, or between `score` and a
  later re-extract. The E1 selection, the sealed training data and the scored primary can therefore rest on different
  label snapshots, and a rerun of `score` is not reproducible. The legacy side is frozen (read-only root), so this
  asymmetry is specific to new draws. The plan accepts retries, but not after the lock/seal.
- Fix: have the conductor touch `control/STOP` (and wait for the follow jobs to exit) once `checkpoint graded`
  passes and before `release`. Record in `extract`'s manifest a sha over each labelled partition's grade files, and
  in `e1_selection.json` the dev grade digest it used. After that, retry timeouts only through an explicit, logged
  regrade verb.

### Addendum to the audit-hook finding (fallback trigger)
- `analyze.py:182` falls back whenever the traced run's stderr (strace's **and the fit's own** stderr) contains
  `"Operation not permitted"` or `PTRACE_TRACEME`. A fit that fails with an EPERM error message on a node where strace
  works is silently re-run in-process under the weaker hook. That run overwrites `logs/strace/<stage>.log`, including
  any forbidden open the real strace had recorded. Fix: probe ptrace separately first
  (`strace -f -e trace=none -o /dev/null true`), decide the mode from that probe's exit code, and never fall back after
  a traced run has started.
- If a node lacks `/usr/bin/strace`, `subprocess.run` raises `FileNotFoundError`. The stage then dies with rc 1 and no
  log, and the conductor stops. The probe found strace on node4003, but the partition list
  (`pi_lhtsai,mit_normal`) may land elsewhere. Handle `FileNotFoundError` the same way as the probe failure.

## [minor] `verify_seal` ignores the seal's own gate facts, so any hand-written JSON that hashes one file lifts the fence

- Location: `passk/grade_scope.py:47-68` (and `labels.py:50-65`): the checks are "parses", "non-empty `code_commit`",
  "non-empty `artifacts`" and "hashes match". Relative or **absolute** artifact paths are accepted (`:61-62`).
- Problem: the recorded `static_scan.ok`, `strace_logs[*].ok`/`mode`, `reviewed_commit == code_commit`, and the
  presence of the PREREG prediction files are never re-checked at the point where the fence lifts. A seal written by
  any other tool, by a test fixture pointed at the real root, or by hand (`{"code_commit":"x","artifacts":
  {"/etc/hostname": "<sha>"}}`) releases every split on the next grader poll.
- Fix: in the single `verify_seal`, also require `reviewed_commit == code_commit`, `static_scan.ok`, every
  `strace_logs[*].ok` with `mode == "strace"` (or a waiver hash), the required prediction files among `artifacts`, and
  only relative paths under `sealed/`.

## Checked and found sound (no finding)
- Pre-seal extract never opens `v48_bank`/`t200_bank` jobs/parsed/raw/grades, and never reads `t200_diag` grades.
  `plan_partitions` puts only `smoke,e1_fill,c48_extra,t200_diag` in `partitions`, and the label columns of the unlabelled
  passk rows are forced to None (`extract.py:348-365`). A release requires `grade_scope.verify_seal`.
- Grader: `(skip, only)` come from one read of the shard's jobs manifests on every pass. Ids not listed are never graded
  (fail closed). The conductor passes `--config $SHARD_DIR/state/shard_config.json`, which keeps the `passk` block
  (`shards.shard_config` deep-copies the base). Legacy nodes are not in any passk jobs file, so the grader cannot touch
  them, and nothing in `slurm/passk_*` grades or writes the forced_v2 root.
- No T200 item appears in any forced_v2 jobs file (canary, calibration, calibration_natural, main, pilot), so legacy
  labels cannot leak T200 correctness. The passk canary and smoke use D32 dev items (`live.canary_graphs`).
- Fit-stage imports (`agent_allocation.cli`, `passk.models.e2/e5`) open nothing under the HF cache, shards, private or
  the legacy root (audit-hook check of 940 import-time opens). The fits read `analysis_inputs/*` plus
  `artifacts/manifests/splits.json`. That file is gold-free (panels, strata by year, legacy C48 bank counts), but it
  is outside the plan's "analysis_inputs/ and sealed/ only".
- Withheld guard: `fit_models` asserts rank 3 and runs `guard_training_frame` on the pool frame before
  `choose_latency`. `P2Predictor.fit` guards the draw frame (partitions) and the pool frame. Eta is fit on a subset of the
  guarded pool frame, and `RidgePredictor.fit` guards. `TRAIN_CONFIGS`/`C48_ONLY_CONFIGS` have no overlap with
  WITHHELD (module assert). `e5.fit_and_predict` refuses when held-out partitions carry labels. V48 pools are built only
  from `v48_bank` (`pools.py:50-51`).
- Fence unit tests pass: `test_grade_scope.py`, `test_seal.py`, `test_extract_fence.py` -> 17 passed, 1 skipped (the only
  real-strace test is skipped here, and would be skipped on the compute nodes too).

### Evidence addendum (tests encode the gaps)
- `tests/unit/passk/test_seal.py:14-40` (`_prepare` + `test_seal_written_and_verifies`) seals a root whose only sealed
  artifact is `sealed/e1_selection.json`, with no V48 or T200 predictions, and asserts that the result verifies. The
  suite therefore locks in the "seal need not contain any prediction" behaviour.
- All seal tests hand-write the `logs/strace/*.log` lines. The only real-trace test (`:65-77`) skips when ptrace is
  blocked, which is the case on the login node and on node4003. No test covers `_audited_models` or `--strace`
  end-to-end, and none covers a fit that reads through pyarrow.

### Evidence addendum for "Two divergent fence implementations"
I built a scratch study with jobs for `v48_bank`/`t200_bank`/`t200_diag`/`e1_fill` and a verifying seal carrying
`"splits": ["v48"]` (a V48-only release, which `grade_scope` supports).
`grade_scope.blocked_node_ids` -> `['d1', 't1']` (T200 stays fenced). `labels.blocked_node_ids` -> `[]` (T200
released). The contract-named function releases T200 labels that the grader keeps fenced.

