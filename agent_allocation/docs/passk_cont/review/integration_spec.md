# passk_cont_v1 adversarial review: integration seams and spec compliance

Reviewer lens: WP-A/WP-M/WP-G seams, conductor stage wiring, PREREG compliance. Merged branch `passk-cont` @ 3c0a530.
Each finding was checked against the code. Where a cheap check was possible, it was run in this session.

## [blocker] The conductor's analysis jobs can never pass `--release` or `--strace`, so the chain stops at `seal` and `release` is unreachable

- **Location:** `slurm/passk_conductor.sh:19-20,193-200` (`run_analysis` → `submit analyze "" "STAGE=$st"`);
  `slurm/passk_cont.sbatch:111-113` (`analyze)` runs `passk-analyze --config --run-root --stage "$STAGE"` and
  nothing else); `src/agent_allocation/passk/analyze.py:154-157,173-188`; `src/agent_allocation/passk/seal.py:153-157`.
- **Problem:** `passk_cont.sbatch` is the script the conductor submits, and its `analyze` verb forwards only
  `STAGE`. It has no `RELEASE` or `FLAGS` handling, which exists only in the separate `slurm/passk_analyze.sbatch`
  that the conductor never uses.
  1. `e2-fit` and `e5-fit` run without `--strace`, so `logs/strace/{e2-fit,e5-fit}.log` is never written.
     `run_seal` then records "`… missing (run the fit stage under strace)`" and returns 1, and the conductor stops
     at `analysis:seal`. Gate P can never pass through the conductor.
  2. `ANALYSIS_POST=(release score report)` submits `--stage release` with no `--release`, which hits
     `parser.error("--stage release needs --release v48|t200")` → exit 2 → the conductor stops at
     `analysis:release`. Even with a `--release`, one stage cannot release both V48 and T200.
  3. Recovery is not a simple resume. `e2.fit_and_predict` and `e5.fit_and_predict` refuse to run when
     `sealed/{v48,t200}/predictions.parquet` exists (`e2.py:136-137`, `e5.py:150-151`). A re-run with `--strace`,
     or `START_AT=analysis`, therefore dies with `FileExistsError` until someone deletes the sealed outputs by
     hand. `START_AT` has no `seal` or `post` entry point.
- **Evidence:** `grep -n "strace\|RELEASE\|FLAGS" slurm/passk_conductor.sh slurm/passk_cont.sbatch` finds nothing.
  `analyze.py:156-157` shows the `parser.error` path.
- **Fix:**
  - Make `passk_cont.sbatch analyze` forward `RELEASE` (split on `+` into repeated `--release`) and `FLAGS` (split
    on `+`), plus `--workers "$SLURM_CPUS_PER_TASK"`.
  - Have the conductor submit `e2-fit`/`e5-fit` with `FLAGS=--strace`.
  - Replace `ANALYSIS_POST` with the WP-A order:
    `release RELEASE=v48+t200` → `pairs` → `extract` → `score` → `report`. Also see the next findings.
  - Add `START_AT=seal|post` entry points.
  - Make the fit stages idempotent. When outputs exist and no seal exists yet, either skip or allow an explicit
    `--overwrite-unsealed`. Never overwrite after `SEAL_predictions.json` exists.

## [blocker] The conductor order `extract → pairs → e1-lock → e2-fit` seals E2/E5 predictions built on stale classes (silent exact-key fallback, not the PREREG Math-Verify classes)

- **Location:**
  - `slurm/passk_conductor.sh:19`: `ANALYSIS_PRE=(extract pairs e1-lock e2-fit e5-fit seal)`, with no second `extract`.
  - `src/agent_allocation/passk/extract.py:390-426`: `classes.parquet` is written at extract time. Pools that need
    an uncached pair are skipped and only counted as `classes_pending_pools`.
  - `src/agent_allocation/passk/models/inputs.py:158-168`: a scope with no rows silently falls back to
    `exact_classes`.
  - `src/agent_allocation/passk/seal.py`: no check on `classes_pending_pools` or `scopes_exact_fallback`.
- **Problem:** No pair-cache follow job runs during generation. The conductor never submits one, although the plan
  calls for it. So at the first `extract`, every scope that contains new-run MATH draws with distinct exact keys is
  pending:
  - the E2 `c48fit` scopes, which include `e1_fill` and `c48_extra`;
  - the T200 `diag5` scopes (`t200_diag`);
  - the L96 E5 training pools 8x2/2x8/2x2. `catalog/pair_verdicts.parquet` appeared on the real run root at 15:27 (the P0 legacy fill), so legacy-only pools are probably covered. Every scope that mixes in new-run draws is not. Single-member pools (1xk) need no pairs.
  
  `pairs` fills the cache, but `classes.parquet` is never rebuilt. `e2-fit` and `e5-fit` then read the stale table,
  and `ClassSource` falls back to exact vote keys for exactly those scopes. That removes the Math-Verify merge the
  PREREG requires ("exact vote key then mutual complete-link Math-Verify"). The E2 answer-class model, the E5 CONC
  features, and every E5 training target on those pools are then sealed on the wrong class definition, and
  PREREG allows no refit after the seal. The model manifest still reports
  `classes_source = "analysis_inputs/classes.parquet"`. Only `classes.scopes_exact_fallback` reveals the problem,
  and nothing gates on it.
- **Evidence (run here, scratch run root):** one T200 item's `t200_diag` answers are `5, 5.0, 6, 5, 7`.
  - First extract: `classes_pending_pools = 1`.
  - After `pairs` (stub evaluator 5≡5.0), with no re-extract: `ClassSource.stats() = {'source':
    'analysis_inputs/classes.parquet', 'scopes_from_table': 0, 'scopes_exact_fallback': 1}` → **4 classes**.
  - After a second extract: `scopes_exact_fallback: 0` → **3 classes**.
  - Script: `scratchpad/stale_classes.py` in this session.
  - Side defect found by the same script: when `classes.parquet` has zero rows (every scope pending), it is
    written without columns. `ClassSource._index` then takes the member-level branch and raises
    `KeyError: 'scope_id'` (`inputs.py:150`).
- **Fix:**
  - Change the chain to `extract → pairs → extract → e1-lock → e2-fit → e5-fit → seal`.
  - Make `e2-fit`/`e5-fit` refuse (exit 2) when `analysis_inputs/manifest.json:classes_pending_pools > 0`, or when
    `ClassSource.n_fallback > 0` for any scope that has MATH voters.
  - Make `seal` refuse when either `sealed/{v48,t200}/manifest.json` records `scopes_exact_fallback > 0`.
  - Write `classes.parquet` with its schema even when it is empty.

## [blocker] After release, `score` raises `MissingPair` (no `pairs` + re-`extract` in the chain), and WP-M scorers silently mix seeded and tie-averaged outcomes on partial `readouts.parquet`

- **Location:**
  - `slurm/passk_conductor.sh:20`: `ANALYSIS_POST=(release score report)`.
  - `src/agent_allocation/passk/score.py:107-109`: `readout_pools(..., cache=cache)` raises on any uncached pair.
  - `src/agent_allocation/passk/extract.py:408-426`: `readouts.parquet` holds only the pools whose pairs are
    cached.
  - `src/agent_allocation/passk/models/inputs.py:214-226`: a pool missing from `readouts` silently uses the
    tie-averaged value, with exact-key fallback classes.
- **Problem:** The held-out `v48_bank`/`t200_bank` answers are first read at release. Their pairs are never in the
  cache, because `pairs` only enumerates what `analysis_inputs/draws.parquet` held before the release. The
  conductor goes straight from `release` to `score`.
  - WP-A's `score()` raises `MissingPair` on the first MATH pool that has two distinct answers, and the chain
    stops.
  - If `score()` were made tolerant, WP-M's `e2.score`/`e5.score` would still compute the E2/E5 metrics, and the
    P2-vs-P1 inputs (`scores/e5/item_brier.parquet`), on a mixture:
    - seeded `delivered_correct` for pools whose pairs happened to be cached;
    - tie-averaged plurality on exact-key classes for the rest.
- **Evidence (run here):** `fakeroot.build_t200(n_per_bench=4)` → seal → `extract(release=["t200"])` gives
  `classes_pending_pools = 105`. `readouts.parquet` holds 151 T200 pools. `score()` then raises
  `MissingPair "aime/t0: no cached verdict for ('6', '7')"`. The existing `test_score.py` only passes that point
  because the test fills the pair cache itself, a step the conductor does not take.
- **Fix:**
  - Post-release chain: `release (v48+t200)` → `pairs` → `extract` (sticky releases) → `score` → `report`.
  - `score()` should refuse (exit 2) while `manifest.json:classes_pending_pools > 0`.
  - `observed_pool_table` should raise when a scored pool is missing from `readouts`, instead of falling back.

## [major] Known failing test `test_score.py::test_release_then_score_primary_family`: `score()` runs the E2 scorer unconditionally

- **Location:** `src/agent_allocation/passk/score.py:140-147`; `src/agent_allocation/passk/models/e2.py:356-359`.
- **Root cause:** After the merge, `importlib.import_module("agent_allocation.passk.models.e2")` succeeds and has
  `score`. `score()` therefore calls `e2.score(run_root)`, which reads `sealed/v48/predictions.parquet`. The WP-A
  test (written before WP-M existed) seals only `t200/predictions.parquet` and `e1_selection.json`, so the read
  raises `FileNotFoundError`.
- **Production impact:** `score()` also:
  - runs `e5.score` even when T200 is not released;
  - runs `e2.score` even when V48 is not released;
  - computes `mixed_vs_deep` on T200 unconditionally, which fails with a `KeyError` in the pivot when only V48 is
    released.
  
  So the plan's staged releases ("V48 once `v48_bank` passes Gate 4, T200 once `t200_bank` passes Gate 4") cannot
  be scored separately. A V48-only release crashes `score`. If V48 fails Gate 4, then T200 cannot be scored
  either, because `e2.score` runs on a V48 panel with no rows.
- **Evidence:** full unit suite here: **1 failed, 1090 passed, 1 skipped** (the skip is `test_seal.py:76`, ptrace
  blocked). The only failure is this test. `tests/integration/test_passk_*.py`: 5 passed.
- **Fix:**
  - Gate each part on `analysis_inputs/manifest.json:releases` and on the sealed inputs existing:
    - the primary family and `e5.score` only when `"t200"` is released;
    - `e2.score` only when `"v48"` is released and `sealed/v48/predictions.parquet` is sealed.
  - Update the test to either seal a V48 stub or assert that E2 scoring is skipped when V48 is not released.

## [major] `seal` does not check that the PREREG seal contents exist; a partial `sealed/` directory seals and lifts the fence for both splits

- **Location:** `src/agent_allocation/passk/seal.py:174-179` (the only content check is "nothing under sealed/ to
  seal"); `src/agent_allocation/passk/grade_scope.py:75-80` (a seal without `"splits"` releases every held-out
  split).
- **Problem:** PREREG §Seal requires one seal to cover all of the following:
  - the V48 predictions;
  - the T200 predictions from all six predictor sets for all 200 items;
  - models and seeds;
  - `e1_selection.json`.
  
  `run_seal` hashes whatever happens to be under `sealed/` and accepts any non-empty set. So `seal` still writes
  a verifying `SEAL_predictions.json` in these cases:
  - `e5-fit` crashed after `e2-fit` (or was skipped by hand);
  - the conductor is resumed out of order;
  - only `e1_selection.json` exists.
  
  The seal has no `"splits"` key, so `grade --follow` then grades both V48 and T200, and a `--release t200` becomes
  legal. After that, P2-vs-P1 (primary test 2) can never be pre-registered, and the fence cannot be restored
  without deleting the seal. Nor is coverage checked: `e2.fit_and_predict` silently drops a C48 item that has no
  fitting draws (`e2.py:145`, `items = [i for i in c48 if i in bench]`), and nothing asserts 48 V48 items × 13
  configs × 8 models, or 200 × 9 × 6 T200 rows.
- **Evidence:** `test_score.py` itself seals only `e1_selection.json` + `t200/predictions.parquet`, and
  `verify_seal` accepts that. `run_seal` has no list of required paths.
- **Fix:** In `run_seal`, require each of the following, or refuse:
  - `sealed/e1_selection.json`;
  - `sealed/v48/{predictions,crossover,population}.parquet` + `manifest.json`;
  - `sealed/t200/{predictions,features}.parquet` + `manifest.json`;
  - a row-count check: V48 `item_id` set == `splits.json` V48; T200 `item_id` set == T200 (200); predictor set ==
    {P0,P1,P2,P3,P1-conc,P2-conc}; the 9 menu configs present for every item.
  
  Also record `"splits": ["v48","t200"]` explicitly.

## [major] The pre-seal chain cannot resume, and its one owner precondition (`gates/analysis_review.json`) is not in the RUNBOOK or Gate 0

- **Location:**
  - `slurm/passk_conductor.sh:229-260`: `START_AT` only offers `analysis`, which re-runs `extract pairs e1-lock
    e2-fit e5-fit seal` from the top.
  - `src/agent_allocation/passk/models/e2.py:136-137` and `e5.py:150-151`: the fits are write-once and raise
    `FileExistsError`.
  - `src/agent_allocation/passk/seal.py:136-141`: `analysis_review.json` is required.
  - `docs/passk_cont/RUNBOOK.md`: no mention of `analysis_review.json`. It does not exist in the run root today
    (`gates/` holds only `gate_s.json`).
- **Problem:** The seal job runs in the background at about launch+45 min. At that point it needs:
  - `gates/analysis_review.json` with `commit == HEAD`;
  - a clean tree;
  - two strace logs.
  
  If any of these is missing, `seal` exits 1 and the conductor stops, while generation continues and later stops
  as well. Resuming with `START_AT=analysis` re-runs `e2-fit`, which raises `FileExistsError` because
  `sealed/v48/predictions.parquet` exists. No entry point resumes at `seal`. Every recovery therefore means
  deleting sealed outputs by hand and refitting. Nothing gets leaked, but it wastes the autonomous window and
  invites error-prone manual deletes in `sealed/`. Any code fix committed after the review also changes HEAD, and
  the review file then has to be rewritten, which is not documented either.
- **Fix:**
  - Add `analysis_review.json` (commit == HEAD) as a Gate 0 precondition, or as a check at the start of the
    analysis chain that exits 2 before any fit.
  - Document it in RUNBOOK §4.
  - Make each pre-seal stage skip itself when its outputs exist and verify:
    - `e1-lock`: already the case;
    - `e2-fit`/`e5-fit`: skip if `sealed/{v48,t200}/manifest.json` exists and its `sha256` map matches, and the
      strace log exists.
  - Add `START_AT=seal` and `START_AT=post`.

## [minor] WP-M's E2/E5 scorers count an all-absent pool as delivered = 0, while PREREG and WP-A treat it as missing in the primary analysis; two "primary Brier" numbers will disagree

- **Location:** `src/agent_allocation/passk/readout.py:279-280` (`delivered_correct = False` when nothing is
  delivered); `src/agent_allocation/passk/models/inputs.py:210-226` (`seeded` is taken as is, and absent members
  are forced to 0); `src/agent_allocation/passk/metrics.py:83-85` (WP-A sets `S_known` NaN for all-absent pools,
  per PREREG).
- **Problem:** A 1x16 (or 1x8, 2x8) pool whose only members failed after 3 attempts is all-absent. Gate 4 allows up
  to 1% failures, so a few T200/V48 deep pools will be all-absent.
  - WP-A's `scored/primary.json` (P2 vs P1) drops that cell, as PREREG says ("missing in the primary analysis").
  - WP-M's `scores/e5/summary.parquet:primary_brier_mixture` and `item_brier.parquet`, and the E2 plurality
    Brier, score the cell as a failure.
  
  The WP-M handoff calls `item_brier.parquet` the input to the Holm test. The report reads WP-A's `primary.json`,
  so the two artifacts carry different "primary" values for the same metric. `n_known` can also differ, which
  changes which items pass the ≥ 3-configurations rule.
- **Fix:** In `observed_pool_table`, set `plurality`/`delivered` to NaN for pools where every member is absent
  (the WP-A readouts already carry `n_absent` and `n`). Keep an `operational` column that holds 0. Name WP-A's
  `primary.json` the single authoritative primary, and label WP-M's number "operational".

## [minor] T200 menu contrasts are macro-weighted only, and the E1 lock compares configs on different item sets

- **Location:** `src/agent_allocation/passk/stats.py:210-235` (`menu_contrasts(..., weights=MACRO)` for every
  panel, called from `score.py:115-117`); `src/agent_allocation/passk/analyze.py:66-74`.
- **Problem:**
  1. PREREG makes the T200 mixture the primary estimand and macro secondary. For T200, the secondary menu
     contrasts (every 16A/8A config vs 1x16/1x8, with simultaneous intervals) are computed only under macro. No
     mixture version is written for T200.
  2. `e1_lock` averages `S_tie` per family with NaNs skipped. `S_tie` is NaN when the tie bounds differ (an
     unknown grade among tied first members) or the pool is all-absent. So 8x2, 4x4 and 2x8 can be averaged over
     different L96 item sets, and M* can flip on missingness rather than accuracy. PREREG does not define the
     item set. `e1_selection.json` records `n_unknown_tie` but not which items were excluded.
- **Fix:**
  - For T200, add `menu_contrasts(..., weights=MIXTURE)` next to the macro version.
  - In `e1_lock`, restrict to items with a known `S_tie` in all three candidates (and report the excluded items),
    or record the choice as a PREREG clarification before the lock runs.

## [major] No committed test crosses the WP-A → WP-M seam, and the conductor dry-run test freezes the broken stage list

- **Location:** `tests/integration/test_passk_conductor_dryrun.py:101-102` asserts
  `stages == ["extract", "pairs", "e1-lock", "e2-fit", "e5-fit", "seal", "release", "score", "report"]` (sbatch is
  stubbed, so neither the missing `--strace`/`--release` nor the missing re-extract can fail it).
  `tests/unit/passk/test_run_root_entrypoints.py` runs `e2/e5.fit_and_predict` + `score` on a hand-written
  `draws.parquet` with **no** `pools.parquet`/`classes.parquet`/`readouts.parquet`, so the `ClassSource` aliases
  (`c48fit`→`e2fit:`, `diag5`), the WP-A class-row schema and `readouts.delivered_correct` are never exercised
  together. The WP-M handoff's cross-worktree check ran from an uncommitted scratch script (`wpa_build.py`).
  `test_integration_legacy.py` (WP-A) stops before the model stages.
- **Problem:** The two blockers above (stale classes → silent exact fallback; post-release MissingPair / partial
  readouts) and the `test_score.py` failure all live on this seam, and nothing in the suite would catch a
  regression there. The conductor dry-run test is the only guard on stage order, and it asserts the wrong order.
- **Fix:** Add one integration test on a `fakeroot`-style run root: `extract` → `pairs` → `extract` → `e1-lock` →
  `e2.fit_and_predict` / `e5.fit_and_predict` (assert `classes.scopes_exact_fallback == 0` in both manifests) →
  `run_seal(require_git=False)` with stub strace logs → `release v48+t200` → `pairs` → `extract` → `score` (assert
  every scored pool took `plurality_source == "readouts_seeded"`). Update the dry-run assertion to the corrected
  chain including the `FLAGS`/`RELEASE` exports.

## [minor] The conductor's `report` stage renders no E2/E5 results and a stale E0

- **Location:** `docs/passk_cont_report/analysis.py:30-60,100-120` (it reads `e0/audit.json`,
  `sealed/e1_selection.json`, `scored/primary.json`, `scored/menus.csv` and `analysis_inputs/draws.parquet`, but
  nothing from `scores/e2/` or `scores/e5/`); `slurm/passk_conductor.sh` (no `e0` stage).
- **Problem:** The PREREG's secondary outputs never reach the report the chain builds:
  - E2: population error, Brier/log score, interval coverage, ranking accuracy, crossover residuals;
  - E5: per-predictor Brier, calibration, withheld cells, ΔBrier for concentration, the 21A policy.
  
  The F2/F6 slots are still comments. `e0/` was last produced from the scratch legacy pair cache (WP-A handoff),
  and the conductor never re-runs `--stage e0` after the real cache is filled. The "report built" state therefore
  does not mean the study's deliverable is complete.
- **Fix:** Have `analysis.py` read `scores/e2/{overall,by_config,crossover}.parquet` and
  `scores/e5/{summary,contrasts,policy}.parquet` into `numbers.json` with T-tables. Add `e0` after the pre-seal
  `pairs` in the chain, or document it as a manual step in RUNBOOK §4.

### Evidence addendum for the first blocker and the seal-completeness finding (run here, scratch run root)
- `run_seal(root, require_git=False)` with no `logs/strace/*.log` gives rc **1**, which is what the conductor
  chain hits.
- With two one-line fake strace logs and **only** `sealed/e1_selection.json` under `sealed/`, `run_seal` gives
  rc **0**: `{"seal": "written", "n_artifacts": 1}`. `grade_scope.verify_seal(root)` then returns True, so both
  held-out splits would be released with no V48/T200 predictions sealed.
- `cmd_passk_analyze(["--run-root", R, "--stage", "release"])` → `SystemExit(2)` ("--stage release needs
  --release v48|t200").

## [minor] `score` and the WP-M scorers do not refuse pending (unsettled) held-out rows; pending counts as absent, i.e. a failure

- **Location:** `src/agent_allocation/passk/extract.py:173,177` (a pending row starts `absent=True`);
  `src/agent_allocation/passk/readout.py:263` (`present` excludes it);
  `src/agent_allocation/passk/models/inputs.py:210-213` (absent → label 0); `score.py` has no pending check
  (the WP-A handoff says so).
- **Problem:** Only `e1-lock` refuses pending rows. Suppose `score` runs after a Gate 3 stop or a truncated run
  (e.g. an owner resumes with `START_AT=gate4` after a drain). A `v48_bank`/`t200_bank` node that was never
  settled is then scored as an absent member:
  - WP-A excludes it from voting and counts it in `n_absent`, and an all-pending pool becomes missing;
  - WP-M sets its coverage label to 0.
  
  So unsent calls silently become failures in the E2/E5 scores. PREREG's missingness rule covers "failed
  generation after 3 attempts", not "never ran".
- **Fix:** `score()` (and `e2.score`/`e5.score`) should exit 2 while any released-partition row has
  `pending=True`, or report pending separately and never fold it into absent.

## [minor] Gate 4 omits "pair cache complete", and the conductor makes each release depend on the other split's Gate 4

- **Location:** `src/agent_allocation/passk/checkpoints.py:292-320` checks completion ≥ 99%, dev grades and
  unique THINK texts. It has no pair-cache criterion and no union key reconciliation.
  `slurm/passk_conductor.sh:265-280`: any shard's Gate 4 failure stops the chain before any release.
- **Problem:**
  1. The plan's Gate 4 includes "pair cache complete". That criterion is exactly what would have caught the
     missing post-release `pairs` step (the post-release `MissingPair` blocker above). Without it, nothing
     between Gate 4 and `score` checks that classes are complete.
  2. The plan says "V48 is released once `v48_bank` passes Gate 4, and T200 once `t200_bank` passes Gate 4". The
     conductor instead requires every partition on both shards to pass. A 98.9% `t200_bank` on one shard also
     withholds V48 (E2), and the reverse holds too.
- **Fix:**
  - Add a `pairs_complete` criterion. It passes when the post-release extract manifest has
    `classes_pending_pools == 0`, so it must be evaluated after the release extract (or run as a pre-`score`
    check).
  - Evaluate Gate 4 per split, and release/score each split independently. This needs `score()` to gate by
    release, as in the `test_score.py` finding above.

## [minor] A failed background pre-seal chain is hidden in STATUS.md until generation ends

- **Location:** `slurm/passk_conductor.sh:229-257`. `( run_analysis … ) &` records `stopped:analysis:<stage>` in
  `conductor.state`/`STATUS.md`. The main loop then overwrites it with `note main "polling; …"` every `POLL`
  seconds. The failure only shows up at `wait "$ANALYSIS_PID"` after both runs finish, when Slurm mails FAIL.
- **Problem:** A seal-time stop happens at about launch+45 min. It includes the strace, review-file and
  stale-class failures described above. It stays invisible for the rest of generation (1–2 h), which is the
  window in which the owner could have fixed it.
- **Fix:** On failure, have the subshell write a sticky marker (for example `$STUDY/ANALYSIS_FAILED` holding the
  stage and rc). Have `note` append that marker to `STATUS.md` while it exists. Optionally send a mail via
  `sbatch --wrap true --mail-type=END` right away.

## [blocker] Primary test 2 (P2 vs P1) crashes on WP-M's real `sealed/t200/predictions.parquet`: `brier_items` raises `KeyError: 'benchmark'`

- **Location:** `src/agent_allocation/passk/score.py:64-70` (`brier_items`); the schema comes from
  `src/agent_allocation/passk/models/e5.py:135-137` (`predict_all` writes
  `item_id, benchmark, config_id, predictor, p_delivered, p_coverage`).
- **Problem:** `brier_items` merges `predictions` (which has a `benchmark` column) with
  `obs = metrics[[item_id, benchmark, config_id, S_known]]` on `["item_id", "config_id"]`. Both frames have
  `benchmark`, so pandas suffixes it to `benchmark_x`/`benchmark_y`. The next
  `groupby(["item_id", "benchmark", "predictor"])` then raises `KeyError: 'benchmark'`.
  - `score()` crashes before `primary.json` is written, so neither primary test is reported and Holm never runs.
    This happens after the labels have been released.
  - `test_score.py` misses it because its hand-made predictions lack the `benchmark` column. That is the
    contract §5.7 minimal schema, but WP-M writes more.
- **Evidence (run here):** `brier_items(pred, metrics)` with `pred` built exactly like `e5.predict_all` raised
  `KeyError 'benchmark'`. The same call with `pred.drop(columns=["benchmark"])` returned the expected (4, 4) item
  frame.
- **Fix:** In `brier_items`, merge on `["item_id", "config_id"]` after selecting
  `predictions[["item_id", "config_id", "predictor", "p_delivered"]]`, or merge on
  `["item_id", "benchmark", "config_id"]`. Make `test_score.py` load predictions written by
  `e5.fit_and_predict` (or at least include `benchmark`).

## Test runs (this session, merged tree @ 3c0a530)
- `pytest tests/unit`: **1 failed, 1090 passed, 1 skipped** (282 s).
  - The failure is `tests/unit/passk/test_score.py::test_release_then_score_primary_family`; see the
    `test_score.py` finding above.
  - The skip is `tests/unit/passk/test_seal.py:76`: ptrace is not permitted on this node.
- `pytest tests/integration/test_passk_*.py`: **5 passed** (30 s).
- `seal.static_scan()` on the merged tree: ok, 11 files, no findings.
- Imports of `cli`, `passk.analyze`, `models.e2/e5` and `features` open nothing under the HF cache, the legacy
  root, `shards/` or `private/` (audit hook).

## Not reviewed (out of lens or out of time)
- The E2/E5 model internals: the glm/GH quadrature, the Beta/Dirichlet simulators, and P2 η out-of-fold
  correctness.
- WP-G's generation, ceilings and ledger, Gates 1–3 thresholds, and `splits.py`.
- E0 replay.
- Live strace on a compute node (ptrace is blocked here).
- The LaTeX build.
