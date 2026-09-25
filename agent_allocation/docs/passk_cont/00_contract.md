# passk_cont_v1 — build contract (frozen 2026-09-25, day 0)

Study `openrouter_passk_cont_v1`, preset `passk_cont_v1`, config `configs/passk_cont_v1.json`, branch `passk-cont`
off `forced-v2@dc7eb24`. Approved plan: `/home/mabdel03/.claude/plans/i-want-to-run-sharded-swan.md` (read it; this
file freezes the interfaces the work packages build against). Spec:
`expermt_agent_specs/Pass@K-Frontier-Cont/remaining_experiments_spec.md` (fast path E0+E1+E2+E5 only).

## 1. Frozen parameters

| parameter | value |
|---|---|
| model / permaslug | `qwen/qwen3.8-27b` / `qwen/qwen3.8-27b-20260814` (Gate 1 stops on any other) |
| providers | `mancer/fp8`, `parasail/fp8` only (shards `mancer`, `parasail`) |
| A | 1,024; allowances a ∈ {1,2,4,8,16}·A |
| protocol | forced THINK (logit_bias −100 on 248069/248046/248044, max_tokens = a, finish=length, exact) + ANSWER (headroom 4,096); T 1.0, top_p 0.95, top_k 20; no seed; `X-OpenRouter-Cache: false` |
| prompt_version | `prompts-v1-e998b556294c` — **no template file may be added or changed** |
| decoding_config_hash | `a641b002e3cb…` (identical to forced_v2; asserted in tests) |
| run_id | `openrouter_passk_cont_v1` (from `study_id`) |
| cfg.seed | 20260917 (legacy; keeps shard-plan reproducing the legacy assignment) |
| passk_seed | 20260925 (salts, tie key, bootstrap) |
| budget | credit 150 / soft 142; shards mancer 87/82, parasail 63/60; `--account-auto-reload` always |
| scheduler | forced27b values; mancer start 48 / rps start 6 / max 96; parasail max 40 / rps 8; wave_items 16; max_active_waves 6 |
| grader | forced_v2 grader unchanged (Math-Verify 0.9.0, 5 s, tri-state) |
| bootstrap | 10,000 replicates, benchmark-stratified, item-clustered, seed 20260925 |
| mixture weights | aime .10, hmmt .10, omni_math_hard .50, gpqa_diamond .30 |

## 2. Run root layout (`/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1/`)

```
artifacts/                     run root for the harness (manifests/, private/, locks/, shards/<id>/…)
artifacts/manifests/splits.json            write-once; sha256 in gates/gate_s.json
artifacts/manifests/duplicate_groups.json  write-once
artifacts/manifests/reserve_pool.json      items left after T200 (future tranches)
artifacts/shards/{mancer,parasail}/        per-shard run dirs made by `allocation shard-plan`
analysis_inputs/               ONLY written by `passk-analyze --stage extract`; the only data fit stages read
catalog/pair_verdicts.parquet  append-only gold-free Math-Verify pair cache
sealed/                        e1_selection.json, v48/, t200/, SEAL_predictions.json
e0/                            E0 outputs
gates/                         gate records (gate_s, gate0, gate1_<shard>, gate2_<shard>, gate4, gate_p, analysis_review)
logs/  STATUS.md  conductor.state  conductor.lock
```
Legacy run root `…/forced_budget_v2/artifacts` is **read-only** (SQLite opened with `?mode=ro`); nothing is ever
written there or into the dynorch run root.

## 3. Panels, partitions, replicate map

Panels (written by `passk/splits.py`, lead-owned): L96 (legacy main 96), C48 ⊂ bank-known L96 (12/benchmark),
V48 (= C48 items), T200 (20/20/100/60 unused; legacy reserve items eligible), D32 (legacy dev), X80 ⊂ T200
(8/8/40/24; frozen, unused). Root `manifests/panels.json` in legacy format: dev = D32, main = L96 (legacy order),
reserve = T200 (frozen rank order), `waves.json` = `[]`.

| partition | kind(s) | items | replicate indices (`root_slot = draw_id = r`) |
|---|---|---|---|
| `smoke` | ROOT @A | 12 D32 (3/benchmark, first 3 of each benchmark's dev list) | r = 100 (never collides with legacy slots 0–24) |
| `e1_fill` | DEEP @2A r1–7, @4A r1–3, @8A r1 | L96 | legacy draw is r0 |
| `c48_extra` | DEEP @4A r4–7, @8A r2–7, @16A r1–7 | C48 | |
| `v48_bank` | ROOT @A r0–15; DEEP @2A r0–7, @4A r0–3, @8A r0–1, @16A r0 | V48 | |
| `t200_bank` | same 31-node bank | T200 | |
| `t200_diag` | ROOT @A r0–4 | T200 | |

Nodes are built with `live.standalone_spec(cfg=…, partition=…, task=…, lock=…, kind=…, budget=a, root_slot=r,
draw_id=r, amendments=…)`; each item's nodes are emitted in descending budget. Because `run_id` differs from
forced_v2, no new node id can equal a legacy one (asserted anyway). Held-out = `v48_bank`, `t200_bank`,
`t200_diag` (correctness only; `t200_diag` parsed/raw are released for features).

Legacy draw roles (L96/C48): A slots 0–15 = 16×A pool + E2 fitting; slots 0–19 = finite-bank A bank (m_A = 20);
slots 20–24 = diagnostics (enter only `features.py`). Legacy deep/m draw = replicate r0 of the m·A bank.

## 4. Module ownership (`src/agent_allocation/passk/`)

| module | owner | purpose |
|---|---|---|
| `splits.py` | lead | C48/T200/X80 selection, duplicate registry, panels.json, T200 manifests |
| `banks.py`, `run.py` (verb `passk-run`), `e0_gate.py`, `forecast.py`, `gates.py`, `grade_scope.py` | WP-G | generation, gates, fenced grading |
| `slurm/passk_cont.sbatch`, `slurm/passk_conductor.sh` | WP-G | ops |
| `extract.py`, `labels.py`, `pairs.py`, `pools.py`, `readout.py`, `metrics.py`, `stats.py`, `costs.py`, `records.py`, `e0/` , verb `passk-analyze` | WP-A | analysis core, E0 |
| `models/glm.py`, `models/response.py`, `models/answers.py`, `models/e2.py`, `features.py`, `models/predictors.py`, `models/e5.py` | WP-M | E2/E5 models |
| `docs/passk_cont_report/` | WP-A | report |

Shared-file edits outside `passk/` allowed only as listed: `grading/worker.py` (blocked-id skip, WP-G),
`shards.py` (per-panel share override, WP-G), `cli.py` (verb registration: WP-G adds `passk-run`, WP-A adds
`passk-analyze`), `config.py` (lead, done). No change to `contracts.py`, `GenerationSpec`, templates,
`runtime/scheduler.py`, or anything that alters node ids.

## 5. Interfaces (all paths relative to the run root unless absolute)

### 5.1 `manifests/splits.json`
```json
{"version": "passk-splits-v1", "passk_seed": 20260925,
 "panels": {"L96": [...], "C48": [...], "V48": [...], "T200": [...], "D32": [...], "X80": [...]},
 "c48": {"strata": {"<benchmark>": {"<bin>": {"size": n, "selected": k}}}, "inclusion_prob": {"<item>": p}},
 "t200_strata": {...}, "assignment_t200": {"<item>": "mancer|parasail"},
 "salts": {...}, "sha256_inputs": {...}}
```
Item ids use the legacy form (`aime/2025/7`, `omni_math_hard/473`, `gpqa_diamond/rec…`).

### 5.2 `catalog/pair_verdicts.parquet` (WP-A `pairs.py`)
Columns: `item_id, text_a, text_b` (normalized `VoteKey.text`, a < b lexicographically), `pair_sha`,
`verdict` (bool|null), `math_verify_version`, `timeout_s` (5.0), `elapsed_s`, `computed_at`. Append-only;
stored nulls are final. API: `lookup(item_id, a, b) -> bool|None` (raises `MissingPair`), `fill(pairs, workers)`.

### 5.3 `analysis_inputs/` (WP-A `extract.py`; the single reader)
- `draws.parquet` — one row per candidate: `candidate_id` (node_id), `source` (`legacy|passk`), `partition`,
  `panel_roles` (list), `item_id`, `benchmark`, `shard`, `kind`, `allowance_a` (1,2,4,8,16,25), `replicate`,
  `parse_status`, `vote_text`, `vote_exact_key`, `option`, `answer_len_chars`, `answer_finish_reason`,
  `think_latency_s`, `latency_s`, `think_in, think_out, answer_in, answer_out, cost_usd`, `generation_id_think`,
  `generation_id_answer`, `observed_model`, `observed_provider`, and — **only for released partitions** —
  `correct` (bool|null), `grade_state`.
- `manifest.json` — partitions read, allow-list, sha256 of every output, code commit.
Allow-list before the seal: legacy (L96 root/deep, D32 root @A), `smoke`, `e1_fill`, `c48_extra`, `t200_diag`
(without `correct`/`grade_state`). After `SEAL_predictions.json` verifies, `--release v48|t200` adds those.

### 5.4 Pools (WP-A `pools.py`) — `pools.parquet`
`pool_id` (sha256 of item + config + view + member ids), `item_id`, `panel`, `config_id`
(`16x1, 8x2, 4x4, 2x8, 1x16, 8x1, 4x2, 2x4, 1x8, 1x1, 2x1, 1x2, 4x1, 2x2, 1x4, 2x16, 4x8, 8x4, diag5`),
`n, a, R=1, B=n·a`, `member_ids` (frozen replicate order), `view` (`current`), `candidate_order_seed`.
Prefix rule: n×a uses replicates r0…r(n−1) at allowance a (A-level uses legacy slots 0…n−1 on L96/C48).

### 5.5 Readout (WP-A `readout.py`)
`plurality(pool, classes, mode="passk"|"legacy") -> {delivered_id, tied_class_ids, winner_class, support,
tie_averaged_correct(labels)}`; passk tie key = sha256(`passk_cont_v1/tie`, 20260925, pool_id, class_id).
Classes via `selection.plurality.build_classes` in frozen member order with equivalence served by the pair cache.

### 5.6 Labels and fence (WP-A `labels.py`)
`blocked_node_ids(paths, cfg) -> frozenset[str]` (held-out node ids from `manifests/jobs/<partition>.jsonl`
whose split has no verified seal) — used by the grader (WP-G) and extract. `verify_seal(run_root) -> dict|None`.
`SEAL_predictions.json`: `{code_commit, reviewed_commit, artifacts: {path: sha256}, strace_logs: {...},
static_scan: {...}, created_at}`.

### 5.7 Models (WP-M) — inputs are `analysis_inputs/*` + `pools.parquet` + classes; outputs under `sealed/`
- `models/glm.py`: `fit_glm(X, y, groups, *, item_intercept: bool, ridge: float, gh_nodes=20) -> GLMFit`,
  `cv_ridge(…, folds=5, grouped by problem)`, `GLMFit.predict_proba(X, item_effect=None)`,
  `GLMFit.sample_item_effects(n, rng)`.
- E2: `sealed/v48/predictions.parquet` (`item_id, config_id, model, pred_coverage, pred_plurality,
  pi_lo, pi_hi`), `sealed/v48/crossover.parquet`, `sealed/v48/models/*`.
- E5: `sealed/t200/predictions.parquet` (`item_id, config_id, predictor ∈ {P0,P1,P2,P3,P1-conc,P2-conc},
  p_delivered, p_coverage`), `sealed/t200/models/*`, `sealed/t200/features.parquet`.
- Withheld (never in any fit/tune frame): (n, a/A) ∈ {(4,4), (8,1), (4,2), (2,4)}. Guard raises on them and on any
  row from `v48_bank`/`t200_bank` or any `t200_diag` label column.

## 6. Verbs
- `allocation passk-run --config C --run-dir <shard dir> --partitions p1,p2 [--allow-paid --max-usd X
  --account-auto-reload --resume] [--plan-only]` (WP-G). Never grades inline; grading is `allocation grade` with the
  blocked-id skip.
- `allocation passk-analyze --config C --run-root R --stage extract|pairs|e1-lock|e2-fit|e5-fit|seal|release|score|report|e0`
  (WP-A; model stages call WP-M functions).
- Slurm `--export` values never contain commas: partitions are passed as `PARTITIONS=e1_fill+c48_extra` and split
  on `+` inside the sbatch.

## 7. Rules for every work package
- Tests: `.venv/bin/python -m pytest -q` from `agent_allocation/` (never touch the consortium conda env).
- No network, no key, no paid call in any test. The fake transport (`api/fake.py`) is the only upstream in tests.
- Read-only access to the legacy run root; tests that read real legacy data must be skippable when absent.
- Commit small and often on your WP branch with clear messages; end commit messages with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
