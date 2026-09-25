# HANDOFF: passk_cont_v1 splits and Gate S (lead-owned `passk/splits.py`)

Branch `passk-cont`. Code: `agent_allocation/src/agent_allocation/passk/splits.py`. Tests:
`agent_allocation/tests/unit/passk/test_splits.py` (16 tests, all passing, about 90 s: real-data tests read the legacy
run root and write only under pytest's tmp dir).

## Done
- **Gate S PASSED** (13/13 checks): `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1/gates/gate_s.json`.
  - `splits.json` sha256 is `c6384ba1d4c3ff8e0aeeae4fcfed7c8904fd020cc37c19ae170e9a84a6528d3c`.
  - A rebuild in a scratch root and a rebuild in the tests give the same bytes.
- **Frozen artifacts** under `.../passk_cont_v1/artifacts/`. All are write-once: an identical rerun does nothing, and
  a run that would change a file refuses.
  - `manifests/splits.json` (contract §5.1 schema plus `panels_by_benchmark`, `assignment_t200_counts`,
    `duplicate_exclusions`, `legacy_seed`).
  - `manifests/duplicate_groups.json` and `manifests/reserve_pool.json`.
  - `manifests/panels.json` in legacy format:
    - dev = D32 and main = L96, both in legacy order;
    - reserve = T200 in frozen rank order;
    - seed 20260917; procedure `passk_cont_v1`.
  - `manifests/waves.json` = `[]`.
  - `manifests/exclusions.json` and `locks/datasets.json`, both byte copies of the legacy files.
  - `manifests/public/*.jsonl`, and `private/*.jsonl` with mode 0600 in a 0700 directory:
    - rows are dev, then main, then T200;
    - the 256 L96/D32 rows are byte-identical copies of the legacy rows;
    - the T200 rows come from the adapter path. All 320 legacy rows, public and private including GPQA, regenerate
      byte-equal through that path.
- **Composition:**
  - T200 is 20/20/100/60.
  - X80 is 8/8/40/24 and lies inside T200. It is stratified with the same legacy stratum rule and its own salt
    `...|x80-v1`.
  - The reserve pool holds aime 8, hmmt 11, omni 355 and gpqa 104.
  - Pools before T200: aime 28, hmmt 31, omni 455 after deduplication, gpqa 164.
  - Legacy reserve items that entered T200: aime 5, hmmt 6, omni 2, gpqa 2 (listed in
    `t200_strata[*].legacy_reserve_in_t200`).
- **Duplicate registry:**
  - The scan covers the 820 eligible items: exact normalized sha, plus char-5 or word-3 Jaccard ≥ 0.3.
  - It flags 56 pairs:
    - The 13 reviewed Omni pairs. All are flagged by the scan.
    - 43 other pairs, which I read. They share templates (functional equations, or GPQA shared stems) but are
      different problems, so they are not duplicates.
  - It excludes 14 items: 3611 (partner of main 336), 162 (partner of dev 3962), 308 and 3566 (conflicting golds),
    and one member of each of the 10 remaining pairs. In each pair the lowest source row is kept.
  - Gold comparison: whitespace-free text first, then Math-Verify in both directions. Only "not equal in either
    direction" counts as conflicting.
    - 328/3724 (`\{n \ge 13\}` vs `n \geq 13`) is `one_way` and is kept as a duplicate.
    - 308/3566 is the only conflict.
  - No gold text is written to manifests.
- **C48:** 94 bank-known items. `bank_correct` from `sample_bank_items` agrees with `candidates.parquet`
  (main / root / 1024; count correct of 25) on all 94. Allocation table (selected/size, bins 0 / 1-5 / 6-15 / 16-25):
  - aime 4/12 3/3 1/1 4/8
  - hmmt 6/16 3/3 0/0 3/4
  - omni 4/9 4/7 3/6 1/1
  - gpqa 0/0 2/2 4/6 6/16

  Inclusion probabilities are in `splits.json` `c48.inclusion_prob`. V48 = C48.
- **T200 assignment:** 1:1 per benchmark (10/10, 10/10, 50/50, 30/30) in `splits.json` `assignment_t200`.
- **Legacy assignment is reproduced:** unchanged `shards.root_assignment` on the new `panels.json` (seed 20260917,
  shares 3:2) gives exactly the legacy mapping for all 128 L96 ∪ D32 items, and the main/dev counts are equal.
  `allocation shard-plan` (preview, no `--write`) runs cleanly on the new root.
- **PREREG check:** `PREREG.md` at e781c34 equals HEAD and the working tree, and e781c34 is an ancestor of HEAD.
- **Loader for other WPs:** `passk.splits.load_splits(run_root)` returns `splits.json` only when Gate S passed and
  the file's sha256 still equals the one Gate S recorded.

## Remaining (priority order)
1. **WP-G's T200 1:1 override must use `splits.json` `assignment_t200` verbatim.**
   - Unchanged `root_assignment` gives the reserve panel 3:2: mancer 12/12/60/36, parasail 8/8/40/24. That is the
     number `shard-plan` prints today.
   - A plain 1:1 `shard_assignment` (seed 20260917, block cut) would also differ from the frozen alternating rule
     (salt `...|t200-shard`).
   - Gate S records `t200_1to1_equals_root_assignment_reserve: false` for the current code.
2. Run `allocation shard-plan --write` on the root only after the WP-G override is merged. It writes
   `shards/assignment.json` and the shard run directories.
3. Optional: add `splits.json`'s sha to the Gate 0 record ("Gate S passed and splits.json unchanged"). That check
   can be `load_splits()` itself.

## Known issues / notes
- `omni_math_hard/3974` is a legacy reserve item and is now in T200. Its gold is
  `\boxed{f(x)=cx},\text{<Chinese "where">} c \in ...`. It passes the frozen eligibility rules, because the non-ASCII
  `\text` escapes `gold_prose`. It may grade oddly. It was left in place because the eligibility rules are frozen;
  flag it in the report if it grades as unknown.
- The legacy `shards/*/state/state.sqlite-shm` files have mtime 2026-09-25 07:16. That predates this build, and
  splits.py never opens SQLite. A WAL-mode SQLite opened with `?mode=ro` can still touch `-shm`, so E0 readers should
  use `?mode=ro&immutable=1`.
- `gate_s.json` is write-once. A rerun compares every field except `code_commit` and `evaluated_at`. To regenerate
  after a real change, delete the run-root files by hand and rerun; nothing downstream reads them yet.

## Exact commands
```bash
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_cont/agent_allocation
export UV_CACHE_DIR=/orcd/data/tpoggio/001/mabdel03/.cache/uv HF_HOME=/orcd/data/tpoggio/001/mabdel03/.cache/huggingface \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MPLCONFIGDIR=/orcd/data/tpoggio/001/mabdel03/.cache/matplotlib \
  PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m pytest -q tests/unit/passk/test_splits.py          # 16 passed
.venv/bin/python -m agent_allocation.passk.splits --config configs/passk_cont_v1.json --dry-run   # no writes
.venv/bin/python -m agent_allocation.passk.splits --config configs/passk_cont_v1.json             # idempotent re-verify
```
