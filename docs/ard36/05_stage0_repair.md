# ARD36 Stage 0 — the repair register

What changed between the frozen `tables/` (v1) and the repaired `tables_v2/`, why, and what it did
to the published numbers. One entry per repair, each with the verification that proves it.

**v1 is untouched and stays that way.** `provenance/v1_tables.sha256` hashes all 19 v1 artifacts,
`provenance/v1_git_head.txt` records `30dd53b7aabd2e5c72a451cb3d0aef389261c6c9`. Re-hash to prove it.

**Baseline:** `PYTHONPATH=src pytest tests/ard tests/study -q` was 1,336 passed / 1 skipped /
17 subtests before any change, and is **1,465 passed / 1 skipped / 17 subtests** after, with no
regressions. The +129 are new tests shipped with the repair.

## The headline: a pre-registered primary changed its sign and now rejects

| contrast | v1 | v2 |
|---|---|---|
| GATE − PEER round-3 selected accuracy | −0.00833, p=0.465, Holm 0.931, reject=False | −0.00893, p=0.396, Holm 0.396, reject=False |
| GATE − RANDOM_PROTECT round-3 selected accuracy | −0.02584, p=0.056, Holm 0.168, reject=False | −0.02257, p=0.035, Holm 0.071, reject=False |
| **PEER − PRIVATE root→r3 coverage change** | **+0.0125, p=0.535, Holm 0.931, reject=False** | **−0.04583, p=0.0080, Holm 0.0240, reject=TRUE** |

Cause: **R1** below. `analysis.coverage_change_arm` accumulated into `per_episode[episode_id][round]`
and overwrote. A ROOTS cell defaults its protocol to PRIVATE and shares an `episode_id` with the
real PRIVATE episode of the same item, so the 16-draw bank's round-0 coverage replaced the
five-agent team's round-0 coverage in the PRIVATE arm — inflating PRIVATE's baseline until PRIVATE
appeared to *lose* coverage over the episode.

The isolation is clean and is the reason to believe it: **PEER's change is byte-identical in v1 and
v2** (−0.02083), because no ROOTS cell aliases a PEER `episode_id`. Only PRIVATE moved, −0.03333 →
**+0.02500**.

Reproduced three independent ways:
1. the repaired estimator: −0.04583;
2. a direct trajectory computation: PEER P(C_t≥1) 80.00→77.92% (−2.08 pp) vs PRIVATE 80.00→82.50%
   (+2.50 pp), difference −4.58 pp;
3. a root-paired exact McNemar on identical round-0 roots: **12 vs 1 discordant, p = 0.0034**.

**Consequence.** The study was reported as "all three pre-registered contrasts null", and the
pivot to a descriptive characterization rested on that. It is not true. One primary rejects, in the
direction that peer visibility destroys correct answers relative to private self-revision.

## Repairs

### R1 — cell-kind keying and the ROOTS/FORK de-collision (M2) **[changes numbers]**
`episode_id` is not unique: 472 ids are one EPISODE + one ROOTS row, 8 ids (the D-11 items) carry
five rows, and **144 ids are two FORK cells** branching off parent rounds 1 and 2 that differ only
in `cell_id` — so `cell_kind` alone cannot separate them and the key had to be `cell_id`.

- `cell_id` + `cell_kind` added to `episode_rows`, `round_summary_rows`, `transition_rows`;
  `cell_id`, `cell_kind` and `item_id` added to `decision_rows`.
- ROOTS skipped in `round_summary_rows` (−504 rows), `selection_rows` (−7,560), `cost_summary_rows`
  (−504). `transition_rows` needed no skip — a ROOTS episode has a single round.
- `analysis.coverage_change_arm` re-keyed to `(cell_id, episode_id)`.

*Verified:* `(cell_id, episode_id, round)` unique on 12,832/12,832 rows; `n_valid + n_unparsed`
histogram is `{1: 1280 (receiver), 5: 11552}` with the 16s gone; natural PRIVATE `round_summary`
is `{0:480, 1:480, 2:480, 3:480}` (v1 was 984 at round 0).

### R2 — tri-state correctness, at the real locus (M1) **[changes numbers]**
The coercion is **not** in `compact._is_correct`, which was already tri-state. It is
`graders.GradeResult.to_row` emitting `is_correct = (status == STATUS_CORRECT)`, so all 5,047
ungradeable rows (5,046 `ambiguous` + 1 `timeout`) are written `False` and `labels.is_correct` is
never null — making `_is_correct`'s `None` branch unreachable. **The repair originally specified
(editing `_is_correct` to return `None` on a null) would have been a no-op diff.**

`_is_correct` now reads `evaluator_status`. `round_summary` gained `n_graded`, `n_ungraded`,
`team_fully_graded`, `team_class_complete` so that every `n_correct`-conditioned statistic can
condition on a fully-graded team instead of silently assuming one.

*Verified:* `C_{t+1} − C_t = WC − CW` holds on **6,771 fully-graded round pairs with 0 violations**,
and `accounting_residual` is non-zero on 0 rows. Of 11,552 team rounds, 9,529 are fully graded and
9,894 are class-complete — both independently reproduced by `ard/panels.py`.

### R3 — GPU-second attribution, and the cubic it was hiding (B-1) **[changes numbers]**
`resources.request_resources_from_decisions` read `latency_s`; `compact.decision_rows` writes
`latency`. Every request therefore contributed zero, and `mean_concurrency` was 1.0 on every row.

Fixing it exposed a latent performance defect: `_mean_concurrency` rescanned every peer inside each
of O(n) sub-intervals for each of n requests — **cubic per replica**, which only ever terminated
because the zero latencies made every span degenerate. On the largest replica (4,757 requests) it
would not have finished. Replaced with a sweep-line that builds the in-flight step function once
per replica and answers each request from an interpolated prefix integral.

*Verified:* the sweep-line reproduces the original brute force on 5,824 requests across 300
synthetic replicas (including degenerate zero-length spans) to a **maximum absolute difference of
1.4 × 10⁻¹⁴**; runtime on the real 67,104 requests is **0.5 s**. Recovered accounting:
**268.75 request-hours → 25.41 attributed GPU-hours**, against 27.15 h of replica busy time, at
mean concurrency 23.19 (max 112.91).

### R4 — fork re-metering (M4/Q13) **[changes numbers]**
All 1,440 `cohort == 'fork'` rows carried `thinking_tokens = final_tokens = prompt_tokens =
cached_prefill_tokens = latency = 0` while their FLOPs were non-zero. `ard/remeter.py` replays the
raw record through `client.decision_from_record`, including the two-phase continuation merge —
**196 of the 1,440 needed it**.

*Verified, and this is the decisive check:* replayed `flops_total` is **byte-identical to the frozen
column on 1,440/1,440**, which proves the replayed record is the right one. The falsifiable gate —
re-metering *known-good* rows — passed: 200 non-fork decisions → 263 compared rows, **0 mismatches**
across all nine fields with literal byte-equality (no float tolerance on latency), stratified to
cover both the single-phase and two-phase paths.

Two losses the register did not previously name: `endpoint_id` is `""` and `answer_logprob` is null
on all 1,440. Both are recoverable and are now returned.

### R5 — the transition change rule (M3/Q7) **[changes numbers]**
`changed_class` conflates a genuine switch with an answer entering or leaving validity. Split into
`switched_class` / `lost_validity` / `gained_validity`, plus `alternate_correct_switch`.

*Verified:* of 3,952 `changed_class` rows, **1,581 (40.0%) are validity flips** and 2,371 are real
switches — so a "switch rate" read off `changed_class` overstates switching by two thirds.
`alternate_correct_switch` is **exactly 1** row in the whole study, and that one is a symbol-case
artifact (`2x + 3` → `Y = 2X + 3`): report it as *not estimable (1 event / 1 item)*, never as a
phenomenon.

### R6 — pass@k on a consistent convention **[changes numbers]**
After R2 the team panel stopped scoring ungradeable answers as wrong, but `bank_summary` still did
(`bool(_is_correct(...))` maps `None` → `False`). The two halves of the communication-vs-sampling
comparison were on **different conventions**, biasing `pass@k` downward against the teams it is
compared to.

`pass@k` is now complete-case over graded draws, with `pass_at_k_upper` — a Manski upper bound
counting every ungradeable draw correct — published beside it, plus `n_graded`, `n_ungraded`,
`plurality_gradeable` and `oracle_coverage_upper`.

*Second selection effect caught here:* complete-case `pass@k` over all 240 items silently shrinks
the item set as k grows (236 items at k=1, **171** at k=16), because a bank needs k graded draws.
Any k-vs-k comparison must use the **fixed 171-item set** with `n_ungraded == 0`.

On that fixed set: pass@1 83.04, vote@16 84.80, pass@2 85.67, pass@3 86.71, pass@5 87.80,
pass@16 91.23; team round-3 output 84.80 (FRESH) to 87.06 (RANDOM_PROTECT). Every arm lands between
pass@2 and pass@5 — 20 generations buying what 3–5 oracle-verified samples buy. Note FRESH at round
3 equals vote@16 **exactly** (84.80%).

### R7 — the minority frame (Q1/Q2) **[changes numbers]**
v1 selected teams with `len(members) == N_AGENTS`. That did not exclude the bank; it excluded the
*collision*, dropping all 2,400 natural PRIVATE round-0 rows — the arm PRIVATE is measured against.
Now keyed on `(cell_id, episode_id, round)` with ROOTS and RECEIVER excluded by kind.

*Verified:* PRIVATE round-0 is **2,720 rows / 544 episodes** (2,400 rescued + 320 that were never
dropped), matching PEER's 2,720 exactly, and `excluded_non_team_groups` is now `{}` — every group is
exactly five members. Frame size 55,360 → 57,760, matching `panels.panel_p0` independently.

### R8 — receiver assay graded for the first time **[new capability]**
Recorded as `unexecuted`; 1,280 decisions had **no label row at all** (absence, not null). They
cannot go through the GRADE lane: `graders.grade()` raises for any family but math500/mmlu_pro, and
receiver rows are stamped `family='math500'`. Graded instead with `graders/trace.py`, the
independent mod-97 evaluator, into a **separate** `receiver_outcomes.parquet` so main-panel
denominators are untouched.

Result: **947 correct / 333 incorrect**, accuracy 0.7398, 0 ungradeable. Budget dominates
everything — the control arm is 0.500 at T1024 vs 0.836 at T4096. Answer-logit margin recovered for
1,260/1,280 (k=20 confirmed present exactly on receiver records).

*Caveats that must travel with it:* 31 of 64 probes are degenerate and **all 31 are
degenerate-correct** (a ceiling, not a floor), so the honest denominator is the 33 non-degenerate
probes; and `packet_tokens == 0` with `seen_message_ids` empty on every row, so the
MINORITY/MAJORITY labels come from `packet_policy` and **cannot be verified against packet content**.

### R9 — calibration **[changes numbers]**
`build_panel` clusters on `item_id` read off decisions rows — and v1 `decisions.parquet` had no
`item_id` column, so every row's item was `None` and all rows collapsed into one pseudo-cluster.
Every "within-item" number in v1 was a pooled number. R1 added a real `item_id`, which fixes it
with no code change. Also wired the discrete-signal dispatch so `ece_discrete` actually fires on
the ordinal confidence grid.

*The correction is large and it removes a signal:* self-reported confidence within-item AUC
**0.7555 → 0.5723**; `answer_logprob` **0.7588 → 0.6250**. Verbal confidence barely ranks correct
above incorrect inside an item — the v1 figure was between-item difficulty. All three *system*
signals' within-item AUC is now **undefined** on every cell (a team row is one candidate per item,
so a mixed pair cannot exist at that grain); v1's 0.7294 was pure artifact.

There are no v1 calibration numbers to diff against: `ard_stats.json` was written 112 s *before*
`calibration_panel.parquet` existed, so every calibration entry in v1 is
`{"estimate": null, "status": "unavailable"}`.

### R10 — the null selectors: diagnosed, not repaired **[capability + limits]**
Three of five selectors emit `selected_correct` entirely null — 120,024 of 200,040 v1 rows (60%).
Three *different* causes, none a simple wiring bug:

- `llm_audit`: **zero SELECTOR_AUDIT cells ever ran** (0 of 934 cell directories). Legitimately
  blocked on generation — and on a wiring gap besides, since `compact` never reads episode events.
- `CURRENT` and `dynamics_only`: blocked on a model fit that never happened. The one frozen
  controller carries only `p_peer` and `p_private` heads with `meta.n_current_rows = 0`;
  `predict("CURRENT", ...)` raises. No caller attaches features, and `features.parquet` is the
  `.empty` sentinel. **Running every cell the spec names would still leave two of the three null.**

A head was refit offline for diagnosis and is **quarantined** in `round_selection_static.parquet`
with `selector_source="refit_offline_v1"`: it was fitted after the test labels existed, which is a
provenance problem, not a repair, and it must not be reported as the sealed spec §12.4 CURRENT.

*Consequence for the record:* any "0 out of N" statistic computed on these selectors is an
artifact of an empty column, not a measurement. An earlier sweep's "a lone correct agent reaches
the team answer 0/186 times" is one such cell.

## Defects found during the repair that no prior register contains

1. **`final_overrun` has two definitions in frozen v1**, depending on which lane wrote the row. On
   a two-phase decision the harness prefills `"\n\n<final>"` (4 tokens) into phase 2's prompt; the
   WP2/episode lane counts the whole post-`</think>` span including those 4, the receiver and
   selector lanes count only generated tokens. Measured: natural 20,718/20,718 and high_budget
   490/490 follow the span rule; receiver 796/796 follows the generated rule.
2. **`receiver.margin_from_record` is broken, not merely unretained.** It probes
   `record.logprobs_content` / `record.top_logprobs`, neither of which exists; the data is in
   `final_top_logprobs`. And the shapes are incompatible, so a rename would not fix it — the entry
   shape has to be built. `tests/ard/test_integration_margin.py` pins the parsing and the width but
   never the lookup, so the gap it claims to close is still open.
3. **System calibration signals group a ROOTS bank as if it were a team.** 449 graded groups have
   more than five members and every one is protocol `NONE` / phase `root`, so every `NONE` row of
   `system_vote_fraction` measures bank draw-agreement, not team agreement.
4. **Three plurality implementations coexist** — `compact._plurality` (via
   `metrics_reference.plurality_vote`), `selectors` (via `ard.vote.plurality`), and
   `compact._confidence_weighted` — with different tie-breaks.
5. **`decision_id` is not a key** (52,768 distinct in 67,104 rows), and `aliased` does **not**
   identify the duplicates: 1,695 duplicated rows would survive a dedupe keyed on it.

## Corrections to the repair plan's own numbers

The plan was itself audited; these of its figures did not survive contact with the data.

| plan said | actually |
|---|---|
| the M1 fix is in `compact._is_correct` | it is in `graders.GradeResult.to_row`; the named edit was a no-op |
| M1 is the consequential defect | M1 cannot flip Holm (smallest p 0.0559); **M2 flips a primary's sign** |
| ambiguity splits 3,353 / 2,016 | not an arithmetic error but a **unit mismatch** — 3,353/2,016 are `minority_frame` rows, 3,166/1,881 are distinct decisions; both are right |
| 219 correct→correct class changes | 160 flagged, 159 of them validity flips, **1** real |
| 2,400 free t=0 fork pairs | **2,720** (544 episode pairs × 5 slots) |
| round-0 replication inflates calibration n by ×4.9 | within-cell factor is **1.0108**; the ×2.878 and ×5.893 are row-population facts, not estimator-sample facts |
| confidence: 95% of mass at ≥0.98 | mass at ≥0.98 is **52.3%**; the four atoms holding 95.1% are {0.95, 1.0, 0.99, 0.98}, modal atom **0.95 at 42.9%** |
| `ece_discrete` must be written | it already exists in `analysis/nb_lib/calib.py`; it just never fired |
| parity panel is 3 decisions, extend to 20 | already 20 via `parity_panel_items(per_cell=2)` |
| `resolve_decision_anchors` is untested | tested against the real pinned Qwen3-8B tokenizer; ran clean on 50 real records |
| E2c is 40 items | **27** under tri-state; 9–11 of the 13 lost have *zero* graded-incorrect roots, so no C0=1 team exists for them |
| E2c costs ~0.25 h of fleet | **~6.6 replica-hours** at 910 decisions/h/replica |
| E6 panel is 144 items | no 144-item selector exists; `build_module("E6")` passes all 384 → **2.67×** the stated budget |

## How to reproduce

```bash
cd /orcd/data/tpoggio/001/mabdel03/agents_scaling
export PYTHONPATH=src
R=/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36

python -m agents_scaling.ard.compact   --run-root $R --out-dir tables_v2
python -m agents_scaling.ard.minority  --tables  $R/tables_v2
python -m agents_scaling.ard.analysis  --run-root $R --tables-dir tables_v2
python -m agents_scaling.ard.results   --run-root $R --tables-dir tables_v2 --prefix v2_
python -m agents_scaling.ard.keys      --tables  $R/tables_v2
python -m agents_scaling.ard.panels    --tables  $R/tables_v2 --check
```

`compact` overwrites in place — **do not `rm -rf tables_v2`**, it also holds
`receiver_outcomes.parquet`, `round_selection_static.parquet`, `selector_availability.parquet`
and `controller_refit/`, which no recompaction regenerates.
