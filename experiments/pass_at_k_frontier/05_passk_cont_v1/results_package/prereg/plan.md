# passk_cont_v1 — fast-path extension of forced_budget_v2 (E0 + E1 + E2 + E5)

## Context

`expermt_agent_specs/Pass@K-Frontier-Cont/remaining_experiments_spec.md` (2026-09-25) asks for experiments that
turn the forced_budget_v2 result (Qwen3.8-27B via OpenRouter; every THINK forced to exactly its allowance, A = 1,024;
deep/16 0.792 vs sample/16 vote 0.417 on 96 items) into **predictive allocation principles**. Do measured
depth-response curves predict mixed depth×width allocations on fresh draws (E2, C48→V48) and on unseen problems
(E5, T200)? Is one deep call better than several adequately deep calls (E1)? E0 audits the legacy logs every claim
rests on. The owner wants the fastest, most parallel execution possible. This plan was adversarially reviewed
before approval: four critics raised findings, a verifier checked each one, and 29 were confirmed and folded in.

## Owner decisions (2026-09-25)

- Scope: **E0 + E1 + E2 (C48/V48) + E5.** Out of scope: E3, E4 LLM selectors, E6 natural stopping (the free
  legacy cost-ledger reconstruction stays), E7, E8, and the optional E2 development expansion.
- Spend: **$150 hard cap**, $142 soft. The owner **will raise the OpenRouter key limit to ≥ $340**, so that
  `limit_remaining` ≥ $155.
- Autonomy: **auto-advance on the frozen gates below.** A gate failure stops the run and notifies the owner.

## Verified facts that shape the design

- **Legacy banks are exact and complete.** Each of the 96 L96 items has 25 forced-A roots (slots 0–24) and one
  DEEP draw at each of 2/4/8/16/25A. ROOT and DEEP THINK bodies are byte-identical except `max_tokens`. No seed was
  ever sent. All 28,085 generation ids are unique, and no THINK text repeats. The served model is
  `qwen/qwen3.8-27b` on every call. The permaslug is `…-20260814` on all 1,054 `/generation` records that exist.
  Every accepted THINK hit its exact budget.
- **The headline numbers reconcile.** 79.2 is deep/16 macro over 96 items. 78.5 is the same over the 94
  bank-known items. 43.3, 60.3 and 58.0 are also 94-item macros. The 94 cohort drops hmmt/2025/18 and
  omni_math_hard/473, whose 82 unknowns are all 5 s Math-Verify timeouts. The 93 cohort also drops hmmt/2026/9.
- **D32** has 7–8 forced-A draws per item. It has no depth banks, so it serves only as the smoke panel.
- **Split pools.** After removing main and dev items, these remain unused: AIME 28, HMMT 31, Omni≥7 469 (455
  after deduplication), GPQA 164. Each count includes 8 reserve items that were never generated. The Omni
  duplicate pairs 3611≡main 336 and 162≡dev 3962 are excluded. The pair 308/3566 has conflicting golds and both
  items are dropped. For 10 other eligible pairs, one representative is kept.
- **Harness** (forced-v2@dc7eb24). `node_id` is the sha256 of every `GenerationSpec` field, so a field must
  never be added. A replica with the same coordinates aliases an existing node and is never sent.
  `live.standalone_spec` and `live.nodes_only_graph` build nodes outside the grid. The fast path needs no new
  templates, so `prompt_version` stays `prompts-v1-e998b556294c`. It needs no legacy parents, so there is no
  adopt step. `scheduler.critical_class` does **not** rank by budget: DEEP and ROOT slot 0 are both class 0.
  The forced_v2 dispatcher has no SIGTERM handler. `grade --resume` redoes timeouts.
- **Cost and throughput** (measured, forced_v2 main). Cost per call, Mancer / Parasail:
  1A $0.00381 / $0.00335, 2A $0.00659 / $0.00572, 4A $0.01203 / $0.01056, 8A $0.02307 / $0.02033,
  16A $0.04519 / $0.03989. Mancer sustained 3.1–3.6k tok/s at 66 mean in flight; 429s produced a 48–96 sawtooth
  and it degraded late. Parasail sustained 2.3–2.8k tok/s with a 40 in-flight cap. Node latency at 16A is
  p50 303 s and p99 850 s. Only Mancer/fp8 and Parasail/fp8 are verified to honour `logit_bias` for this model.
- **Session.** This session runs in Slurm job 23681055, which **ends 16:25 EDT today**. Because the review ran
  long, the build will straddle that time. Every paid, grading and analysis step is an sbatch job driven by a
  conductor job. The owner resumes in a fresh session: see Owner actions.

## Design

### Panels and splits (lead writes `passk/splits.py`; outputs are write-once with sha256 → Gate S)

| panel | items | role |
|---|---|---|
| L96 | 96 legacy main | E0; E1 legacy completion; E5 development training |
| C48 | 48 ⊂ bank-known L96, 12/benchmark | E2 fitting |
| V48 | same 48 items, new draw ids | E2 fresh-draw validation (evaluation only) |
| T200 | 20 AIME / 20 HMMT / 100 Omni / 60 GPQA, unused, reserve items eligible | E1 primary test; E5 prospective test |
| D32 | 32 legacy dev | Gate 1 smoke only |
| X80 | 8/8/40/24 ⊂ T200 (frozen now, unused) | future E6/E7 tranche; `reserve_pool.json` = leftovers |

- **C48:**
  - Strata are the shallow bins 0 / 1–5 / 6–15 / 16–25 correct out of 25. The target is 3 per cell.
  - A cell with ≤ 3 items is taken whole.
  - The remaining quota goes to cells with spare capacity, by largest remainder of spare-proportional shares.
  - Within a cell, items are ranked by sha256(`openrouter_passk_cont_v1|C48|<benchmark>|seed=20260925|`+item).
  - The per-item inclusion probability is recorded. The 4×4 allocation table is printed in `splits.json`.
- **T200:**
  - Pool = eligible − (main ∪ dev) − duplicate partners of used items − conflicting-gold groups. One
    representative (lowest row index) is kept per duplicate group.
  - Strata follow forced_v2 `text_manifest._STRATUM_RULE`, apportioned by largest remainder.
  - Ranking uses the explicit salt `openrouter_passk_cont_v1|<benchmark>|seed=20260925|t200-v1`, never the
    default `contracts.STUDY_ID`.
  - The duplicate registry (normalized hash plus shingle Jaccard ≥ 0.3, 13 Omni pairs reviewed) is frozen in
    `manifests/duplicate_groups.json`.
- **Provider pinning and assignment:**
  - Config keeps forced27b's `seed: 20260917` and 3:2 shares, so the unchanged `allocation shard-plan`
    reproduces the legacy assignment. 20260925 is used only for passk salts, tie keys and the bootstrap.
  - Root `manifests/panels.json` is written in legacy format: dev = D32, main = L96 (legacy order),
    reserve = T200 (frozen rank order), `waves.json` = `[]`.
  - T200 items get their own seeded, benchmark-balanced **1:1** Mancer:Parasail assignment. This is a per-panel
    share override in `shards.root_assignment`, and it balances token load against the legacy 58:38 pin.
  - Gate S asserts that the new assignment restricted to L96 ∪ D32 equals the legacy mapping.
- **Draw roles (frozen):**
  - L96/C48 A slots 0–15 are the 16×A pool and the E2 fitting draws. Slots 0–19 form the finite-bank A bank
    (m_A = 20). Slots 20–24 are diagnostics and enter only `features.py`.
  - Replicate indices are global per (item, budget) across partitions:
    - L96: the legacy draw is r0; `e1_fill` adds 2A r1–7, 4A r1–3 and 8A r1.
    - C48: `c48_extra` adds 4A r4–7, 8A r2–7 and 16A r1–7.
    - V48 and T200 banks start at r0 in their own partitions; `t200_diag` is r0–4.

### New generation (forced THINK = a exactly, ANSWER 4,096 headroom, T 1.0 / top_p 0.95 / top_k 20, no seed, no cache)

| partition | per item | items | calls | allowance | Mancer $ | Parasail $ |
|---|---|---:|---:|---:|---:|---:|
| `e1_fill` | 7@2A, 3@4A, 1@8A | L96 | 1,056 | 3,264A | 5.8 | 3.4 |
| `c48_extra` | 4@4A, 6@8A, 7@16A | C48 | 816 | 8,448A | ~13 | ~9 |
| `v48_bank` | 16@A, 8@2A, 4@4A, 2@8A, 1@16A | V48 | 1,488 | 3,840A | ~7 | ~4.5 |
| `t200_bank` | same 31-call bank | T200 | 6,200 | 16,000A | ~25.5 | ~22.5 |
| `t200_diag` | 5@A | T200 | 1,000 | 1,000A | ~1.9 | ~1.7 |
| **total** | | | **10,560** | **32,552A (33.3M tokens)** | **≈ 56.4** | **≈ 40.8** |

- The forecast is **≈ $99 including probes and 2% retries.** Budget slices set explicitly: Mancer credit
  $87 / soft $82, Parasail $63 / $60 (sum 150/142). forced27b's slices are never inherited: its $36.5 Parasail
  slice would truncate T200.
- If the key raise is missing at launch, the slices shrink to E = min($150, `limit_remaining` − $3), split
  58:42.
- Nodes are kind ROOT at A and DEEP at ≥ 2A (DEEP renders `root.txt` exactly like ROOT), with `root_slot =
  draw_id = replicate`. The run_id is `openrouter_passk_cont_v1`, with one partition per table row.
- `passk-run` always writes the full frozen node list per partition and asserts that its sha equals the list
  recorded at Gate 0.
- **Ordering (no scheduler change):**
  - Each item's nodes are emitted in descending budget. Combined with DEEP = class 0 and FIFO, this sends
    16A→8A→4A→2A first.
  - Stage 1 waves: `c48_extra`, `e1_fill`, `t200_diag`.
  - Stage 2: `v48_bank` and `t200_bank` interleaved in **seeded item-waves**. A wave carries all bank nodes
    of its items, so every arm of an item runs together in time, and a truncation removes whole items, never
    single arms.
  - wave_items 16, max_active_waves 6. Expected tail: about one 16A latency.
- **Scheduler:** Mancer concurrency_max 96, start 48, rps start 6. Parasail max 40, rps 8, default start 16.
  AIMD unchanged; no concurrency is tuned during the run.

### Answer classes, pools and readouts (analysis-side; the harness family taxonomy is untouched)

- **Pair-verdict cache** (`catalog/pair_verdicts.parquet`):
  - Append-only and gold-free. One row per unordered pair of normalized answer texts per item: verdict
    True/False/None, Math-Verify version, `timeout_s` 5, and elapsed time. A stored None is final and never
    recomputed.
  - Filled in a process pool of up to the allocated CPUs, parallel per pair. The forced_v2 seal spent 73% of
    its 60 min on one item, so per-item parallelism is not enough.
  - Stage P0 (legacy L96 pairs) runs as soon as the module passes its tests. A follow job fills new pairs
    during generation.
- **Classes are built per scope** with `selection.plurality.build_classes`: exact vote key, then mutual
  complete-link Math-Verify, in frozen member order (replicate index, never completion order). GPQA uses option
  identity. Scopes:
  - each evaluated pool, over its own members;
  - each 5-draw diagnostic pool (L96 slots 20–24; `t200_diag`), over those five draws only;
  - for the E2 answer model, one scope per C48 item over its fitting draws only.

  Held-out candidates never enter a development scope. The E0 legacy replay keeps the legacy per-partition path.
- **Plurality (new-run policy):**
  - The modal class wins.
  - Ties go to a seeded uniform draw keyed by sha256(`passk_cont_v1/tie`, 20260925, pool_id, class_id).
  - The delivered answer is the first member of the winning class.
  - Also reported: the analytic tie-averaged accuracy.
  - The legacy `class_tie_key` mode is used only to reproduce historical numbers.
- **Per configuration:** coverage H, plurality accuracy S, reliability S/H, answer entropy, top-wrong-class mass,
  finite-bank pass@k where the bank exceeds the pool (never k > m), per-problem mixed coverage
  `G_x = 1 − Π_a C(m_a−c_a,k_a)/C(m_a,k_a)`, and full usage (THINK/ANSWER tokens in/out, $, latency).
- **Menus:** at B=16A, 16×A, 8×2A, 4×4A, 2×8A and 1×16A (prefixes r0…); at B=8A, the correlated prefixes 8×A,
  4×2A, 2×4A and 1×8A. There is no DP: nonuniform allocations are not evaluated in this tranche.
- **Missingness (PREREG):** correctness is tri-state; invalid = false; a timeout = unknown. Absent members
  follow readout rule 07-R12. Primary contrasts use items where both cells are known, with best/worst bounds
  reported. An operational analysis counts failed delivery as failure. There is no separate regrade; the
  grader's `--resume` retries timeouts, and the grade record keeps them versioned.

### Held-out fence (outputs and gold, not just grades)

- Fenced partitions are `v48_bank` and `t200_bank`, plus `t200_diag` correctness. The fence lifts only when
  `sealed/SEAL_predictions.json` exists and verifies. `t200_diag` parsed and raw outputs are released for
  features.
- **Deferred grading, no vault.** The per-shard `grade --follow` process adds every fenced node id
  (`passk.labels.blocked_node_ids`, from `manifests/jobs/<partition>.jsonl`) to its skip set on every pass. Once
  the seal exists, the same follow process grades those nodes within one poll, taking about 1–4 min.
- **One reader.** Only `passk-analyze --stage extract` opens `shards/*/{raw,parsed,grades}`, `private/` and the
  legacy run root:
  - It resolves node → partition from the jobs manifests.
  - It keeps an allow-list: legacy, `e1_fill`, `c48_extra`, `smoke`, and `t200_diag` parsed/raw only.
  - It writes `analysis_inputs/*.parquet` with a sha256 and the list of partitions it read.
  - Every fit stage reads only `analysis_inputs/` and `sealed/`.
  - The E5 training-frame guard keys on partition/draw_id, not on item. C48 items remain training data.
    Any row from `v48_bank` or `t200_bank`, and any `t200_diag` correctness column, raises.
- **Mechanical seal checks** (the conductor evaluates them; no agent is needed after the session ends). All
  must hold:
  - HEAD equals the commit recorded in `gates/analysis_review.json`, and the tree is clean.
  - An AST/import scan of `passk/models`, `features.py`, `pools.py` and `readout.py` finds no reference to
    `labels` release functions, `private`, `shards/` or the HF cache.
  - The fit stages ran under `strace -f -e trace=open,openat,openat2`, and the log shows no opens under
    `shards/`, `private/`, the HF cache or the legacy run root.
  - Every artifact hash is written into the seal.

  The agent review happens once, at code freeze, before launch (see Orchestration).

### E0 — audit ($0; read-only on forced_v2)

- **Gate subset (offline, blocks launch; built inside WP-G as `passk/e0_gate.py`):**
  - no identity collisions: generation ids are unique, and no request-body sha collides across distinct nodes
    except the two known benign private-chain pairs;
  - `observed_model == qwen/qwen3.8-27b` and provider == the pin on every legacy result (from `parsed/`, not
    `candidates.parquet`);
  - all 1,054 existing `/generation` records carry permaslug `…-20260814` with provider == the pin;
  - zero split overlap;
  - `forced_budget_check` is exact on all rows.
- **Rest of E0 (WP-A, after launch):**
  - `audit.json`;
  - `provenance.parquet`, one row per HTTP call: purpose, request hash, generation id, provider, served model,
    status, tokens, cost, latency, raw path and sha;
  - `candidate_catalog.parquet`, with logical memberships from `manifests/policies/*.jsonl`;
  - `exclusions.csv` for the 96/94/93 cohorts;
  - `legacy_metrics.csv`, with every headline under the 96-macro, 94-macro and micro conventions from one
    library;
  - a legacy tie replay, required to match 100% on pools whose replay reports no unknown equivalence (drift
    pools are listed);
  - `reuse_manifest.json`, keyed on model + permaslug, provider tag, item, the THINK body sha without
    `max_tokens`, template hashes, decoding hash, allowance, headroom, stopping mode, parser/grader versions
    and split role, not on lock hash;
  - three decomposition plots.

  A non-blocking `/generation` backfill of 500 legacy main ids runs after launch. It writes only under
  `passk_cont_v1/e0/` and records fetch failures as `unavailable`.

### E1 — mixed independent depth and width

- **L96 legacy completion:** all five 16A and four 8A allocations, labelled descriptive.
- **Lock:** as soon as `e1_fill` is graded, pick the argmax over {8x2, 4x4, 2x8} of L96 tie-averaged plurality
  accuracy under the mixture weights (0.1/0.1/0.5/0.3, applied to L96 family means); ties go to fewer calls.
  Macro is reported alongside. The result is written to `sealed/e1_selection.json`, and the E5 workstream
  cannot read it until the seal.
- **Prospective test (T200):** the locked allocation vs `deep_1x16`, plurality accuracy, two-sided. The full
  menus on L96, V48 and T200 are secondary, with simultaneous intervals.

### E2 — response curves (C48 fit → V48 validation; numpy/scipy only)

All models share one kernel, `passk/models/glm.py`: penalized logistic with an optional Gauss–Hermite item
intercept and 5-fold problem-grouped CV.

1. **Beta(½,½)** per item-depth cell. Coverage is E[(1−p)^k] = B(α, β+k)/B(α, β), with the same p reused for
   repeated calls at one depth; mixed allocations use products per posterior draw.
2. **Pooled response model** (glm): benchmark × categorical depth plus an item intercept. The monotone variant
   is compared on development data only.
3. **Answer-distribution model:** a Dirichlet over the C48 per-item classes plus an explicit unseen residual,
   with two treatments (diffuse; concentrated wrong). Pools are simulated under the frozen tie rule. If the two
   treatments disagree, that is reported as uncertainty.
4. **Ablations:** shallow-only; depth without class competition; full (these three are flags on 1 and 3); an
   unstructured ridge on the same inputs.

- **Predictions** for every V48 item: coverage and plurality correctness for the 5 + 4 configurations, the
  ranking, and the crossover sign(h(kA) − k·h(A)) for k = 2, 4, 8, 16. That sign is positive exactly when
  P(1×kA correct) > P(k×A covered). The residual is the observed difference minus the predicted one.
- **Metrics:** population-average error, Brier and log score, predictive-interval coverage (Beta and glm
  posterior draws), paired ranking accuracy, and crossover residuals, including failures.
- **Illustrative items** are picked by a frozen rule: the largest predicted |crossover| per benchmark.

### E5 — prospective prediction on T200

- **Training data (development = L96):** draw-level correctness at every depth (legacy + `e1_fill` +
  `c48_extra`). Pool-outcome rows, all built from A slots 0–15 or deep prefixes:
  - B=A: 1×A
  - B=2A: 2×A, 1×2A
  - B=4A: 4×A, 2×2A, 1×4A
  - B=16A: 16×A, 8×2A, 2×8A, 1×16A
  - primitive 1×8A
  - C48 only, B=32A: 2×16A, 4×8A, 8×4A

  **Withheld from all fitting and tuning** (the guard keys on (n, a/A); singletons are allowed): (4,4), (8,1),
  (4,2), (2,4). B=8A therefore becomes interpolation in log budget. A unit test asserts
  rank([1, log n, log a/A]) = 3 on every training frame. The development pool table is materialized with the
  withheld configurations physically absent.
- **Features** (label-free, from the 5-draw scopes):
  - concentration group CONC: sorted class frequencies, plurality share, top-two gap, number of classes,
    entropy;
  - invalid fraction, answer length, ANSWER finish reason;
  - latency, z-scored within provider × budget × collection, and dropped if grouped CV shows no gain;
  - benchmark, provider.
- **Predictors:**
  - **P0:** benchmark + budget + configuration (compositional: log n, log a/A, R, B).
  - **P1:** P0 + features + provider.
  - **P3:** ridge on P1 inputs plus pairwise products and squares.
  - **P2 (minimum executable):**
    - the glm mixed model on draw correctness (categorical depth, benchmark, features, depth×feature, item
      intercept);
    - per simulation, one latent item effect shared across all calls and depths → predicted G;
    - a second glm for η on covered multi-call development pools, using out-of-fold response predictions
      (η = 1 for singletons);
    - output P(delivered) = G·η.
  - **P1−conc and P2−conc:** the same models without CONC, same folds and seeds.
- **Seal (joint with E2):** everything below is hashed into one `SEAL_predictions.json`, and no held-out label
  is released before it:
  - predictions for all 200 items (including those with invalid diagnostics) from all six predictor sets;
  - the V48 predictions;
  - models, preprocessing, CV-chosen hyperparameters and seeds;
  - `e1_selection.json`;
  - the code commit and strace log.

  Post-release refits go to `posthoc/`, labelled exploratory.
- **Evaluation:**
  - Primary: mean Brier of delivered correctness over the five 16A configurations.
  - Also: coverage Brier, log loss, calibration, MAE, and paired ranking accuracy.
  - Residuals are reported by benchmark.
  - Withheld cells are reported separately: 4×4A, and the three compositional 8A cells.
  - Secondary readouts: observable concentration (ΔBrier P1−conc vs P1, and P2−conc vs P2), and a 21A
    "select a configuration per item" policy readout. Regret is never measured against the realized-run
    ceiling.

### Statistics (`docs/passk_cont/PREREG.md`, committed on day 0)

- **Estimands:** the T200 mixture (0.10 AIME / 0.10 HMMT / 0.50 Omni / 0.30 GPQA) is primary; the equal
  four-family macro and per-family results are also reported. L96 and V48 use the macro.
- **Primary family (Holm, α = 0.05, two tests, as the spec requires when E3 is omitted):**
  1. the locked mixed allocation vs deep_1x16 on T200;
  2. P2 vs P1 item-level mean Brier difference over the five 16A configurations. Each item averages over its
     known configurations; items with fewer than 3 known configurations are dropped and reported.
- **Bootstrap:** benchmark-stratified paired problem bootstrap, 10,000 replicates, seed 20260925. All draws and
  configurations of an item move together. Effect sizes come with 95% intervals.
- T200 is never expanded, failed generations are never replaced, and non-significance is not claimed as
  equivalence.

## Implementation — branch `passk-cont` off `forced-v2@dc7eb24`

- **Locations:** worktree `/orcd/data/tpoggio/001/mabdel03/agents_scaling_passk_cont`; study id
  `openrouter_passk_cont_v1`; run root `agents_scaling_results/passk_cont_v1/` (`artifacts/shards/{mancer,
  parasail}`, `analysis_inputs/`, `catalog/`, `sealed/`, `e0/`, `logs/`, `STATUS.md`, `conductor.state`).
- **New code** goes in `agent_allocation/src/agent_allocation/passk/`. Nothing is written into the forced_v2 or
  dynorch run roots: they are opened read-only, and SQLite with `?mode=ro`. The live `nextdyn` checkout is never
  switched.
- **Each worktree's venv:** `uv venv --python /orcd/home/002/mabdel03/conda_envs/consortium/bin/python3 &&
  uv sync --frozen --offline`. The consortium env itself is never modified.

**Day 0 (lead, ~45 min):**
- `configs/passk_cont_v1.json`: a copy of forced27b with the changes above (budget slices, scheduler, seeds;
  the `panels` block unchanged).
- `config.py`: add `PRESET_CEILINGS_USD["passk_cont_v1"] = 150` and `PRESET_STUDY_IDS`.
- `docs/passk_cont/00_contract.md`:
  - the frozen parameter table;
  - partitions and replicate map;
  - schemas for splits, pair cache, pools, `analysis_inputs`, sealed artifacts, and the §15.2 records
    (`run_manifest.json`, candidates / pools / selections parquet);
  - the fence rules;
  - module ownership.
- `PREREG.md`.
- The memory file `passk-cont-run.md`, plus its MEMORY.md pointer.
- A $0 Slurm mail test job.
- Commit, fan out, then the lead writes `passk/splits.py` in parallel (≤ 75 min) → Gate S.

**Work packages** (parallel worktrees `agents_scaling_passk_wp{G,A,M}`):

| WP | scope | needed by |
|---|---|---|
| **G** generation + ops | `passk/banks.py`; the `passk-run` verb (modelled on `calibrate.collect`, with explicit per-shard ceilings via `live.active_ceiling(..., account_auto_reload=True)`); the grader `blocked_node_ids` skip (a one-line include filter threaded through `grading/worker.py::scan_candidates`); `passk/e0_gate.py`; the T200 1:1 share override in `shards.root_assignment`; THINK-body parity; forecast over node lists; `slurm/passk_cont.sbatch` (`trap 'touch control/DRAIN' TERM`, no `exec`, no commas in `--export`); `conductor.sh` (flock, duplicate-job refusal, STATUS.md, `--mail-type=FAIL,END`); a mocked e2e on the fake transport for every partition, with SIGKILL + resume, a fence test, and an item-wave ordering test | **paid launch** |
| **A** analysis core + rest of E0 | `extract.py` (the single reader), `labels.py` (seal/join guard vendored from `agents_scaling/src/agents_scaling/nextdyn/analysis.py:110,146` into `_vendor/` with a PROVENANCE row), `pairs.py` (the pair cache), `pools.py`, `readout.py` (new and legacy tie modes, tie-averaged), `metrics.py` (H, S, reliability, entropy, top-wrong, pass@k, G_x, decomposition with bounds), `stats.py` (mixture and macro weights through `_vendor/ard_stats.paired_contrast`, Holm-2), `costs.py` (physical and logical ledgers), the §15.2 record emitters, the E0 outputs, and a report reusing the `docs/forced_v2_report/` skeleton (`render.py` placeholders, `build.sh`) | dev data lands (~launch + 45 min) |
| **M** models | `glm.py` first (merged before anything builds on it); then two sub-agents in the same worktree: E2 (Beta, Dirichlet, ablations, crossover, the V48 writer and scorer) and E5 (`features.py`, P0–P3 ± conc, P2, η, the withheld-target guard, evaluation, policy readout) | joint seal (~launch + 1:30) |

- **Reuse** (forced_v2 worktree):
  - `live.py`: `standalone_spec`, `nodes_only_graph`, `write_jobs`, `active_ceiling`, `record_decision`
  - `calibrate.py::collect`; `runtime/scheduler.py` (unchanged); `api/forced.py` (incl. `validate_request`)
  - `grading/worker.py`, `grading/text/*`; `selection/{plurality,keys,readout}.py`
  - `metrics/{passk,outcomes}.py`; `analysis/{stats,contrasts,costs,itemmetrics,effective_depth}.py`
  - `_vendor/ard_stats.py`; `benchmarks/text_manifest.py` (`rank_key`, `largest_remainder`,
    `_stratified_panels`); `benchmarks/*`; `shards.py`
  - `docs/forced_v2_report/{core,gap}.py`
  - dynorch `conductor.sh` and `slurm/dynorch_v1.sbatch` patterns
- **Orchestration (ultracode):**
  - Workflow 1 builds G, A and M in isolated worktrees, with G as top priority.
  - Workflow 2 runs an adversarial review of G plus the contract before any paid call. Lenses: identity and
    reuse exactness; fence and leakage; budget, ledger and conductor; spec compliance. Fixes follow, then
    Gate 0.
  - Workflow 3 reviews A and M at code freeze during generation. The reviewed commit goes into
    `gates/analysis_review.json`, which the mechanical seal checks.

## Gates (frozen; the conductor records each in `gates/`; a failure drains and stops, and mails the owner)

- **Gate S ($0):**
  - split manifests are write-once with sha256;
  - T200 ∩ (main ∪ dev ∪ duplicate partners) = ∅;
  - the composition is 20/20/100/60;
  - C48 ⊂ bank-known L96, with inclusion probabilities;
  - the legacy assignment is reproduced for L96 ∪ D32;
  - PREREG is committed.
- **Gate 0 ($0):**
  - the suite is green and the mocked e2e passes (all partitions, kill/resume without duplicate dispatch,
    fence test, ordering test);
  - #unique node ids == #planned requests, and none equals a legacy id;
  - `prompt_version` and the decoding hash equal forced_v2's;
  - **for every L96/C48/V48/D32 item, the new THINK body from `ForcedClient.validate_request` equals the legacy
    `request_body.think`, apart from `max_tokens`, with the same `provider.only`;**
  - `e0_gate.py` passes;
  - the per-shard slices match the config;
  - forecast ≤ $115;
  - the conductor re-reads `/key` and requires the sum of shard credit ceilings ≤ `limit_remaining` − $3;
  - Gate S passed and `splits.json` is unchanged;
  - nothing is written under /home.
- **Gate 1 — canary + smoke, per shard (≤ $0.60, ~10 min):**
  - The legacy `allocation canary` runs per shard: 4 dev items × 512…16,384, giving live deep exactness.
  - `passk-run --partitions smoke` runs 12 D32 items × 1A: the spec §15.5 smoke through the new verb and the
    dev grader.
  - Pass criteria: exact budgets 100%; permaslug `…-20260814` and provider == the pin on the new
    `/generation` records; zero cache hits; `/key` reconciles.
  - The conductor submits both shards together, and a canary exit code 3 from the first shard is accepted. It
    then runs `canary --evaluate-only` for both.
  - A transient Gate 0 endpoint refusal is retried up to 3×, 10 min apart. A semantic failure (the banana
    bias test, or a permaslug change) stops the run immediately.
- **Gate 2 — in-run checkpoint (no separate pilot):**
  - Evaluated per shard at ≥ 150 settled nodes or $4, and again every 500 nodes.
  - Exact budgets 100%.
  - Valid-final (parse VALID): pooled ≥ 0.90, and ≥ 0.85 per benchmark when n ≥ 20.
  - Unit cost within ±25% of the forecast for every budget with ≥ 10 calls.
  - Zero provider drift.
  - Refit forecast × 1.02 ≤ that shard's soft slice.
- **Gate 3 — run stop rules (frozen):**
  - Any accepted node that fails `forced_budget_check` stops that shard immediately.
  - ≥ 3 `forced_think_closed` attempts, or `forced_think_short` with finish = stop, in any 30 min window
    (meaning the bias is ignored) stops the shard.
  - Stream cuts (`forced_think_short` with finish None) stop the shard only if they exceed 2% of the trailing
    500 attempts, number ≥ 10, and span ≥ 5 min. Otherwise they are left to the 3-attempt retry.
  - 429/5xx never stop the run by themselves.
  - Unresolved exposure > $2 pauses admission, and stops only after 30 min.
  - The permaslug and provider are sampled every 30 min.
  - The run never raises ceilings, switches providers or edits prompts.
- **Gate 4 — completion:**
  - ≥ 99% of planned nodes succeeded per partition; failures are reported, never replaced;
  - zero duplicate THINK texts across legacy ∪ new;
  - the key reconciles across both shards as a union;
  - dev grades complete; pair cache complete.
- **Gate P — joint prediction seal:** the mechanical checks in the fence section. Then V48 is released once
  `v48_bank` passes Gate 4, and T200 once `t200_bank` passes Gate 4.

## Conductor DAG and timeline (T0 = approval, ≈ 14:00 EDT)

- **T0+0:45** fan-out.
- **T0+2:00** Gate S; pair-cache P0 on legacy (sbatch, 64 CPUs).
- **T0+3:00** WP-G merged.
- **T0+3:00–5:00** review + fixes → Gate 0. **16:25: this session ends** (all work committed; the memory file
  is current).
- **~T0+5:15** conductor job: Gate 1 → main per shard (+ per-shard `grade --follow`, + a pair-cache follow job).
- **launch+0:45** dev partitions and `t200_diag` done → `extract` → E1 lock → E2 + E5 fits → Gate P seal
  (~launch+1:30).
- **launch+~2:00** all generation done → Gate 4 → release V48, then T200 → `score` → `report` → copy to the main
  checkout `docs/passk_cont/report/`.

The conductor (pi_lhtsai, 2 days) owns every step after Gate 0, and each analysis stage is its own sbatch job:
`passk-analyze --stage extract|pairs|e1-lock|e2-fit|e5-fit|seal|release|score|report`. A stop writes STATUS.md
and mails the owner, and a new session resumes from `conductor.state`. Seals are not time-critical: labels stay
fenced until the seal exists, so a stopped chain carries no leakage risk. Expected end-to-end is about T0+8 to
9 h (late evening). The build is the long pole; generation takes about 1.5–2 h.

## Owner actions

1. Raise the OpenRouter key limit to ≥ $340, so that `limit_remaining` ≥ $155, before the conductor's Gate 0
   (≈ T0+5 h).
2. Before or at 16:25 EDT, open a new Claude Code session on a fresh allocation (e.g. a 2-day interactive job
   on pi_lhtsai or pi_tpoggio) in this repo and say "resume passk_cont". The memory file and STATUS.md carry
   the state.

## Verification

- **Unit:**
  - node-id uniqueness and no legacy aliasing; replicate map; THINK-body parity;
  - split determinism; duplicate-registry exclusions;
  - the pair cache is order-invariant and never recomputes a stored None;
  - per-scope class builds never include held-out members in development scopes;
  - the tie rule is seeded per pool; the legacy mode reproduces sealed selections on no-unknown pools;
  - the decomposition identity is exact on legacy pools; Beta coverage matches Monte Carlo;
  - the withheld-target and partition guards raise; rank([1, log n, log a/A]) = 3;
  - the label loader refuses unsealed or changed seals; nominal C = n·R·a;
  - the logical cost charges every member; the physical cost counts each request once.
- **Integration ($0):**
  - mocked e2e over 16 items × all partitions with kill/resume;
  - the analysis pipeline end-to-end on real L96 data, with synthetic held-out splits and an injected fenced
    read that must fail the seal.
- **Live:**
  - every new node exact; permaslug sampled; ledger vs `/key` within tolerance;
  - E0 reproduces 79.2 / 78.5 / 43.3 / 60.3 / 58.0;
  - the report builds with no unresolved placeholder;
  - STATUS.md and the memory file are current.

## Critical files

- Spec; prior plans `ok-then-do-that-robust-elephant.md` (forced_v2) and `we-ran-the-plans-lovely-meteor.md`
  (dynorch).
- forced_v2 worktree `agent_allocation/`:
  - `contracts.py`, `config.py`, `live.py`, `calibrate.py`, `shards.py`
  - `runtime/scheduler.py`, `api/forced.py`, `grading/worker.py`
  - `selection/`, `metrics/`, `analysis/`, `benchmarks/text_manifest.py`
  - `configs/forced27b.json`, `slurm/forced_v2_shard.sbatch`, `docs/forced_v2_report/`
- Legacy run root `agents_scaling_results/forced_budget_v2/artifacts/`:
  - `shards/{mancer,parasail}/{raw,parsed,grades,state,manifests/policies}`
  - `merged/tables/`, `manifests/`, `shards/assignment.json`, `locks/datasets.json`
- dynorch: `agents_scaling_results/dynorch_v1/conductor.sh`, `agents_scaling_dynorch_v1/agent_allocation/slurm/`;
  vendor source `agents_scaling/src/agents_scaling/nextdyn/{analysis,stats}.py`.
