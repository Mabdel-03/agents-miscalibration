# WP-A handoff: passk_cont_v1 analysis core + E0 (branch `passk-wpa`)

Worktree `/orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_wpA`, package `agent_allocation/src/agent_allocation/passk/`.
Tests: `cd agent_allocation && .venv/bin/python -m pytest -q tests/unit/passk` (40 pass, 1 skipped where ptrace is
blocked; includes `test_integration_legacy.py`: real L96 + synthetic e1_fill/v48 through extract, E1 lock, seal
refusing an injected fenced read, and release). The rest of the unit suite (`tests/unit --ignore=tests/unit/passk`)
is green: 971 passed. Without a filled pair cache the five cache-dependent real-data tests skip (35 pass, 6 skip);
point `PASSK_PAIR_CACHE` at a cache (e.g. the copy named below) to run them (~25 s).

## Done (all committed)

| priority | module | what |
|---|---|---|
| P1 | `passk/pairs.py` | Append-only gold-free pair cache `catalog/pair_verdicts.parquet` (contract §5.2 columns). `PairCache.lookup` (order-invariant, raises `MissingPair`), `.equivalence(item)`, `.fill(pairs, workers)` (spawn ProcessPool, one pair per task, `math_equivalent` on the worker main thread so SIGALRM 5 s works in-process; chunked atomic flush under flock; a stored None is final and never recomputed). `enumerate_pairs` (only texts whose exact keys differ, MATH only), `math_votes_from_frame`. |
| P2 | `passk/extract.py` | The single reader. Legacy L96 (main root@A slots 0-24 + deep 2/4/8/16/25A) and D32 (pilot root@A), plus new-run `artifacts/shards/<shard>/manifests/jobs/<partition>.jsonl`. Pre-seal allow-list smoke/e1_fill/c48_extra/t200_diag (t200_diag without `correct`/`grade_state`); `v48_bank`/`t200_bank` never opened pre-seal; `release=["v48"|"t200"]` needs a verifying seal (else `PermissionError`). Writes `analysis_inputs/{draws,pools,classes,readouts,dev_training_readouts}.parquet` + `manifest.json` (sha256 of each, partitions read, allow-list, code commit, `classes_pending_pools`). `load_draws` verifies the sha. Extra columns beyond §5.3: `pending`, `graded`, `absent`, `draw_role`, raw path/sha, THINK body sha (without `max_tokens`), THINK text sha, THINK/ANSWER served model, THINK provider, answer failures, `provider_only`, spec hashes. `think_latency_s` comes from the raw record. |
| P3 | `passk/pools.py` | All §5.4 config ids incl. training rows and `diag5`, prefix rule, B16/B8 menus, `WITHHELD`, `DEV_TRAINING`, `LEGACY_CONFIGS` (25x1, 1x25 for E0), `c48_fit_scopes` (one class scope per C48 item over fitting draws only). |
| P3 | `passk/readout.py` | Classes via `build_classes` in frozen member order with cache-served equivalence; `mode="passk"` tie = min sha256(json(["passk_cont_v1/tie", 20260925, pool_id, class_id])); `mode="legacy"` = `class_tie_key(20260917, item, class)`; single-member and absent rules (07-R12); analytic tie-averaged (lo, hi); `readout_pools`, `class_rows`. |
| P3 | `passk/metrics.py` | H/S/S_tie bounds, entropy, top-wrong mass, `decomposition` (exact three-term), finite-bank pass@k (raises k > m), `mixed_coverage` (exact G_x), `bank_counts` (m_A = 20), `item_mixed_coverage`, macro/mixture/micro `config_summary`; all-absent pool = missing in primary, failure in `S_operational`. |
| P3 | `passk/stats.py` | Mixture .1/.1/.5/.3 and macro through `_vendor/ard_stats` (`Resampler`, `paired_contrast`, `mean_ci`), `holm2` (n_primary = 2), 10,000 replicates, seed 20260925. |
| P3 test | real legacy | deep/16 macro **0.7917 (96) / 0.7853 (94)**; sample/16 legacy tie **0.4167 (96)**; vote@25 **0.4271 (96) / 0.4330 (94)**. Legacy tie replay: **2,976 / 2,976** sealed sample+deep checkpoints match (159 drift-eligible pools also match). |
| P4 | `passk/labels.py`, `_vendor/seal_join.py` | `verify_seal`, `blocked_node_ids(paths, cfg)` (held-out ids of a shard while no verifying seal), `join_labels`, `write_held_out_predictions` (refuses label-like columns). Vendored from nextdyn `analysis.py` with a PROVENANCE row. |
| P4 | `passk/analyze.py` + `cli.py` | Verb `passk-analyze` registered. Stages: extract, release, pairs, e1-lock (argmax 8x2/4x4/2x8 of L96 mixture tie-averaged S; ties to fewer calls; write-once `sealed/e1_selection.json`; exit 2 until every E1 node settled and graded), e2-fit/e5-fit (lazy `passk.models.e2/e5.fit_and_predict(run_root)`; `--strace` wraps the stage), seal, score, report, e0. Exit 0 ok / 1 failed / 2 blocked. |
| P4 | `passk/seal.py` | Gate P: HEAD == `gates/analysis_review.json["commit"]` + clean tree; AST scan (docstrings exempt) of `passk/models`, `features.py`, `pools.py`, `readout.py`; strace-log check of `logs/strace/{e2-fit,e5-fit}.log`; `SEAL_predictions.json` (write-once) hashes every file under `sealed/` into `artifacts` (what `verify_seal` checks) and records `analysis_inputs/` hashes in `inputs_at_seal` + a snapshot `sealed/inputs_manifest_at_seal.json` -- so a release re-extract (which rewrites `analysis_inputs/`) cannot un-verify the seal. An injected fenced open fails the seal (test). |
| P4+ | `passk/score.py` | Post-release: `scored/menus.csv`, `menu_contrasts.csv` (every menu config vs 1x16 / 1x8 per panel, Bonferroni-simultaneous intervals), `pool_metrics.parquet`, `records/` (§15.2), `primary.json` = locked M* vs 1x16 on T200 (mixture, paired, both-known items, best/worst bounds) + P2 vs P1 item mean Brier over the five 16A configs (items with < 3 known dropped and listed) + Holm-2; then WP-M `models.e2/e5.score(run_root)` if present. |
| P5 | `passk/e0/{replay,run}.py` | `e0/`: audit.json, correct_count_trajectories.csv, provenance.parquet (THINK + ANSWER row per node), candidate_catalog.parquet (policy memberships from `manifests/policies/*.jsonl`), exclusions.csv (96/94/93 reproduce: 94 drops hmmt/2025/18 + omni_math_hard/473 with bank-unknown grades; 93 also hmmt/2026/9, main-partition timeouts), legacy_metrics.csv (96/94/93 × macro/mixture/micro), legacy_replay.parquet + legacy_replay_drift.csv, reuse_manifest.json, legacy physical/logical cost ledgers, three decomposition PNGs. |
| P6 | `passk/costs.py`, `passk/records.py` | Physical (each request once) and logical (every member charged, nominal C = n·R·a·A) ledgers; `records/run_manifest.json` + candidates/pools/selections parquet. |
| P6 | `docs/passk_cont_report/` | `render.py` (copied), `build.sh`, `analysis.py` (numbers.json, F1 frontier, F3 copies, T1/T2/T3 tables), `report.tmpl.tex` with F1/F3/T1/T2/T3 slots (F2/F6 left as a comment for WP-M). Placeholders fail loudly until their inputs exist. |
| ops | `slurm/passk_analyze.sbatch` | One stage per job (`STAGE=...`, `RELEASE=...`, `FLAGS=--strace` split on `+`), 64 CPUs default. **Not submitted.** |

## Produced in the real run root (this session, $0)

- `passk_cont_v1/e0/*` -- the full E0 output set (audit.json: cohorts 96/94/93, replay 2,976/2,976 match,
  deep/16 .7917/.7853, vote@25 .4271/.4330, sample/16 .4167; model pinned; generation ids unique; exact THINK
  budgets). It was computed with the scratch legacy pair cache, copied for provenance to
  `passk_cont_v1/catalog_scratch_wpa/pair_verdicts_legacy_4cpu.parquet` (audit.json `pair_cache_path` names it).
  Re-run `--stage e0` after the real P0 cache exists; it overwrites `e0/`.
- Nothing was written to `catalog/`, `analysis_inputs/`, `sealed/`, `gates/` or `artifacts/`.
- Validated (in scratch) that extract with the lead's real `manifests/splits.json` builds 48 C48 pools per config and
  48 `c48fit` scopes with zero pending pairs.

## Remaining (priority order)

1. **P0 pair cache on the real run root (not run).** Submit the 64-CPU job (command below). I filled a scratch copy
   (15,158 legacy L96+D32 pairs, 173 True / 14,057 False / 928 None, 9 min on 4 shared CPUs) at
   `/tmp/claude-225593/-orcd-data-tpoggio-001-mabdel03-agents-scaling/dcd715b7-a48a-49b5-8d50-e93e177bacf5/scratchpad/rr/catalog/pair_verdicts.parquet`
   but deliberately did **not** copy it to the run root: it was computed on a loaded node, where more 5 s timeouts
   (final Nones) are likely. (The scratch cache still reproduced every legacy selection.)
2. **Strace is blocked on this interactive node** (`PTRACE_TRACEME: Operation not permitted`). If compute nodes also
   block ptrace, `--strace` falls back to a Python audit hook (log header `# mode=audithook`; the seal records `mode`).
   The audit hook misses C-level opens (pyarrow parquet reads), so it is weaker. Check on a compute node with
   `sbatch --wrap 'strace -f -e trace=openat -o /dev/null true'` before Gate P; decide whether audithook is acceptable.
3. **Merge with WP-G**: `cli.py` (both add a verb to `VERBS`; trivial conflict). `passk/grade_scope.py` does not exist
   here: `extract.plan_partitions` and `score._seal` try `passk.grade_scope.verify_seal` and fall back to
   `passk.labels.verify_seal` (contract §5.6 assigns both functions to `labels.py`; WP-G's grader can import
   `passk.labels.blocked_node_ids(paths, cfg)` directly). Reconcile to one implementation at merge.
4. **Conductor order** (not encoded anywhere yet): `extract` → `pairs` → `extract` (so `classes.parquet` has no
   `classes_pending_pools`) → `e1-lock` → WP-M `e2-fit --strace` / `e5-fit --strace` → `seal` → per release:
   `release --release v48|t200` → `pairs` → `extract --release ...` (re-run keeps the release) → `score` → `report`.
   Releases are sticky: a later plain `extract` keeps every release recorded in the previous manifest while the seal
   still verifies (and restores the fence if it stops verifying).
5. `splits.json` (lead) drives `panel_roles` (C48/V48/T200 membership). Without it C48 pools and fit scopes are empty;
   T200/V48 pools fall back to partition membership. Re-run extract once `manifests/splits.json` exists.
6. `score` covers the primary family, menus (L96/V48/T200, passk tie rule) with simultaneous intervals, and finite-bank
   coverage; E2/E5 secondary scoring is WP-M's `score(run_root)` hook. WP-M fits must drop or wait on `pending` rows
   (`analysis_inputs/draws.parquet`), exactly as `e1-lock` does.
7. E0 not done: the `/generation` backfill of 500 ids (network; out of scope here) and the request-body collision
   check (WP-G `e0_gate.py`). THINK-text duplicates are audited (0 of 2,976).
8. Report: F2/F6 figures (WP-M outputs), prose beyond the skeleton, `latexmk` not tested here.

## Known issues

- `extract` reads parsed/grades for every planned node; a node without a parsed file is `pending=True` and treated as
  absent by pools/readout. Only `e1-lock` refuses pending/ungraded rows; `score` does not check Gate 4 itself.
- The pairs stage reads `analysis_inputs/draws.parquet` when present, else the legacy root directly
  (`legacy_draws(..., with_labels=False)`); it never reads fenced partitions (their pairs are filled after release).
- `pool_id`/tie key serialization is my choice (JSON arrays); documented in the module docstrings, frozen by tests.
- Cohort rules in `e0.run.cohorts` are data-driven (94 = no unknown grade in the root/deep bank; 93 = no unknown grade
  anywhere in legacy main) and reproduce the published 96/94/93 sets.

## Exact next commands

```bash
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_wpA/agent_allocation   # or the merged passk-cont checkout
export UV_CACHE_DIR=/orcd/data/tpoggio/001/mabdel03/.cache/uv HF_HOME=/orcd/data/tpoggio/001/mabdel03/.cache/huggingface \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MPLCONFIGDIR=/orcd/data/tpoggio/001/mabdel03/.cache/matplotlib \
  PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pytest -q tests/unit/passk
# P0 legacy pair cache on the real run root (64 CPUs; CODE defaults to the passk_cont worktree -- override if needed):
mkdir -p /orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1/logs
sbatch --export=ALL,STAGE=pairs,CODE=$PWD slurm/passk_analyze.sbatch
# then (cheap, local or sbatch): extract with the filled cache, and E0
.venv/bin/allocation passk-analyze --stage extract
.venv/bin/allocation passk-analyze --stage e0
# real-data plurality tests against the real cache:
.venv/bin/python -m pytest -q tests/unit/passk/test_legacy_headlines.py
```
