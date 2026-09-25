# FIX_STATUS -- fixer `analysis_chain`

## Fix pass 1 status (16:05 EDT, 2026-09-25)

Branch `passk-cont`. Commits in order: ddf82de, 919b795, 44c876f, bceaf08, 451ed6a, 42dbdda, 6cdd2df.

### Fixed
- **fence.md [blocker] strace never passed / [major] strace blocked + audit-hook accepted** (ddf82de).
  `passk-analyze --stage e2-fit|e5-fit|seal --sandbox` re-execs the same command inside bwrap:
  `--unshare-all --die-with-parent --tmpfs /tmp --proc /proc --dev /dev`, ro-binds `/usr /lib /lib64 /bin /etc`, the
  consortium env, the code worktree, `analysis_inputs/`, `gates/`, **and `artifacts/manifests/splits.json`** (one
  file; the fits read the panels from it, and it is gold-free), plus a rw bind of `sealed/`. Nothing else from
  /orcd/data is mounted. The outer process writes `sealed/runs/<stage>.json` with the exact argv, its sha256, rc, the
  commit, the dirty flag, the host and the sha256 of every file the run wrote. Every attempt is also logged to
  `logs/sandbox/<stage>.<ts>.json`. The audit-hook fallback is deleted. `--strace` is now a deprecated alias of
  `--sandbox`. Real bwrap works on this node (the tests run it rather than skip).
- **fence.md [major] open-log check not tied to the producing run** (ddf82de). The seal requires a verifying sandbox
  record per fit stage: rc 0, bwrap argv hash matches, every mount is on the allow-list (a wider mount is refused),
  record commit == HEAD, not dirty, and each recorded output's sha still matches. Every PREREG fit file must be among
  that stage's recorded outputs.
- **fence.md / integration_spec [major] the seal need not contain any prediction** (ddf82de, bceaf08). `seal.REQUIRED`
  lists: `sealed/e1_selection.json`, `v48/{predictions,crossover,population}.parquet`, `v48/{manifest,seeds}.json`,
  a non-empty `v48/models/`, `t200/{predictions,features}.parquet`, `t200/{manifest,seeds}.json` and a non-empty
  `t200/models/`. The T200 predictions must hold all six predictor sets. The seal writes `"splits"`.
- **fence.md [major] re-running seal un-verifies it; the pre-seal chain cannot resume** (ddf82de). A second `seal`
  returns 0 if the existing seal verifies and 1 if not, and writes nothing. The inputs snapshot is moved into place
  only together with the seal. A fit stage with a verifying sandbox record is a no-op returning 0; a tampered output
  makes it return 1. `e1-lock` was already idempotent.
- **fence.md [major] static scan passes natural reads** (ddf82de). The scan now also flags: imports of `grading.*`,
  `passk.{splits,checkpoints,score,e0,banks,labels,grade_scope,extract}`, `shards` and `importlib`; `__import__`; the
  bare segments `shards`/`private`/`grades`/`parsed`; and `open`/`read_*`/`rglob`/`listdir`/`ParquetFile`/... on
  paths built from `artifacts`/`catalog`/`logs`/absolute literals and not `analysis_inputs`/`sealed`. The real modules
  scan clean. The reviewer's `leak.py` example is now flagged.
- **fence.md [minor] two divergent fence implementations** (919b795). `labels.verify_seal` and
  `labels.blocked_node_ids` are the `grade_scope` function objects; a test asserts identity. `grade_scope.blocked_node_ids`
  accepts a shard dir, and `cfg=None` defaults to fenced. Documented: a verifying seal lifts **grading** for both
  splits, while **analysis** release stays per split.
- **fence.md [minor] verify_seal accepts any file** (42dbdda, partial). Artifact keys must be relative, under
  `sealed/`, with no `..`.
- **integration_spec [major] failing test_score.py::test_release_then_score_primary_family** (44c876f). `score()`
  scores only splits that are released (manifest `releases`) **and** whose predictions are in the verifying seal. The
  T200 primary family and `e5.score` run only for t200; `e2.score` runs only for v48. The test passes.
- **integration_spec [blocker] brier_items KeyError 'benchmark'** (44c876f). The join takes only
  `item_id, config_id, predictor, p_delivered`. The test now uses WP-M's schema (with `benchmark`).
- **integration_spec [blocker] score MissingPair + seeded/tie-averaged mixing** (44c876f, bceaf08). `score()` raises
  `ScoreBlocked` (stage exit 2) while `classes_pending_pools > 0` or when nothing is released and sealed. In the
  production path, `e2.score`/`e5.score` refuse pending classes, a material exact-key fallback, or any scored pool
  missing from `readouts.parquet`.
- **integration_spec / fence [blocker/major] stale classes before fits** (bceaf08, 6cdd2df). In production
  (`fit_and_predict(run_root)`), the fit does the write-once check before reading any input. It refuses while
  `classes_pending_pools > 0`, and refuses to write if any *material* scope (>= 2 distinct exact vote keys) fell back
  to exact keys. All-absent and single-answer pools are benign.
- **integration_spec [major] no committed test crosses the WP-A -> WP-M seam** (6cdd2df, 451ed6a, partial).
  `test_pipeline_on_real_legacy` runs the strict real `e2.fit_and_predict` on WP-A's real extract, then seal, then the
  V48 release, then the real `e2.score`. It skips if no pair cache exists; it runs here.
  `test_sandbox_stage.py` runs a real bwrap `--stage e5-fit --sandbox`: the model runs inside and reaches its final
  check, and the argv log passes the allow-list.
- **fence.md [minor] raw-path open-log check**: moot, because strace is removed.

### Not fixed
- **Seam test gap.** There is no single synthetic test that goes extract -> e1-lock -> real e2-fit -> real e5-fit ->
  seal -> release -> score. The E5 half of the seal still uses sealkit stubs, and no fixture exists that satisfies a
  strict real E5 fit (it needs `classes.parquet` with T200 diag scopes).
- **verify_seal.** It still does not re-check `static_scan.ok`, `reviewed_commit == code_commit` or the sandbox
  records at fence-lift time. Only `run_seal` enforces those.
- **Static scan.** There is no import allow-list, and the modules the models import (`selection/*`, `contracts.py`)
  are not scanned.
- **Other minors not handled.**
  - Labels frozen after lock/seal (grader/conductor side).
  - The grader fails open without a passk cfg (`worker.py`, not my file).
  - An all-absent pool counts as delivered 0 (WP-M) vs missing (WP-A).
  - T200 menu contrasts are macro only, and the E1 lock compares different item sets.
  - Pending held-out rows are not refused.
  - The report stage lacks E2/E5.
- **Conductor side (launch_ops).**
  - Pass `FLAGS=--sandbox` for `e2-fit`/`e5-fit`. `seal` can run unsandboxed; with `--sandbox`, the commit is
    measured outside and passed in, because `.git` is not mounted inside.
  - Chain order: `extract pairs extract e1-lock e2-fit e5-fit seal`, then `release pairs extract score report`.
  - `score` now exits 2 while classes are pending.

### Tests
The passk fence/seal/score/entrypoint/legacy-seam/sandbox tests pass (see the commits). The full `tests/unit` run was
started at 15:58 on a tree that was still changing; its result is recorded below if it finished.

### Test results (16:08 EDT)
- `pytest tests/unit/passk` at 7263463: **144 passed**, 0 failed, 0 skipped. The real-bwrap tests ran and were not
  skipped.
- `pytest tests/unit tests/integration/test_passk_wpg_e2e.py` (started 15:58, while this tree and the other fixer's
  tree were still changing): **1114 passed, 1 failed**. The failure was
  `tests/unit/runtime/test_status_control.py::test_status_json_is_written_live_and_at_the_end`, a timing assert
  (`asyncio.sleep(0.15)`) run under the load of two concurrent suites. Run alone it passes (3/3). None of my files
  are involved.
