# ARD36 consolidated data-quality register (Q1–Q30, B-1)

The two analysis sweeps that ran over the frozen v1 tables catalogued 30 systematic
data-quality findings and gave them the ids **Q1–Q30**; a thirty-first, **B-1**, came from the
blocker series. Stage 0 of the Phase 2 repair plan is keyed to those ids — every repair step and
every acceptance test names them — but **until this file was written no definition of them
existed anywhere on disk**. `git grep -lE '\bQ(2[0-9]|30)\b'` over 40 revisions returns nothing;
neither `docs/ard36/` nor the run root nor `analysis/` contains them. The plan document *uses*
the ids — "Q17–Q20, Q27, Q30", "de-duplicate Q29" — without ever saying what any of them is, so
it cannot serve as the definition either. An auditor working from the repo alone called the
taxonomy unresolvable and was right to: a blocking repair stage whose
acceptance tests reference identifiers no one outside one session can resolve cannot be verified
by anyone else, which is the same failure mode that made the sweeps retractable in the first
place.

This file is that resolution. It is deliberately a register of *findings*, not a narrative: one
row per id, with the mechanism it belongs to, whether it is repairable, the table-and-column
evidence, and its status.

**This is a different series from `03_defects.md`.** That file carries **D-00 … D-11**:
engineering defects found *while the run was executing*, written as the run went. The Q series is
*analysis*-side and was raised *after* the run, by agents reading the frozen tables. The two
overlap in exactly one place and it is worth naming: D-11's root cause (a ROOTS cell has no
protocol, so `wp2_episode_executor` defaults `cell.protocol or K.Protocol.PRIVATE` and the
16-draw bank shares an `episode_id` with the real PRIVATE round 0) is the same mechanism as
**M2**, and therefore the same underlying fault as Q1/Q2/Q3/Q15. D-11 found it from the bank
side; the sweeps found it from four other sides without knowing it was one fault. Nothing else
in the D series and the Q series is the same item. Do not cross-reference the two by number.

## Provenance, and a caveat that must travel with every number below

The register was recovered from this session's workflow records, not from a document. Primary
sources, in the order they were consulted:

| what | where |
|---|---|
| The 30-row register itself (`# DATA-QUALITY FINDINGS RAISED BY THE SWEEPS`, columns `# / Defect / Evidence and magnitude / Consequence as stated / Raised by`) | session transcript `0ae3b5a5-1ac5-4be7-8266-6c792a94f773.jsonl`, line 6282 — a task notification carrying agent `a8cae264323af665f`'s return. Its `tasks/*.output` file has been reaped; the transcript is the only copy. |
| The `M1–M9` defect-to-mechanism map (`## 0. Defect-to-mechanism map (30 defects collapse to 9 mechanisms)`) | same transcript, line 6362 — the repair/estimator-design agent's return. |
| Adjudication that both of the above are primary sources and that the ids are complete and gap-free | audit workflow `wf_f09ab21d-11b`, journal line 35, agent `a495e3b3916d2e6d1`, verdict CONFIRMED. |
| The sweep verdict tallies | workflow journals `wf_4cd6c827-dfa` (sweep 1) and `wf_d58bbdc9-773` (sweep 2). |
| Stage 0 step numbering and the 24-repairable / 6-policy split | `~/.claude/plans/expermt-agent-specs-agents-round-dynamic-woolly-naur.md`. |

### The provenance caveat (verbatim — do not paraphrase it away)

Two analysis sweeps ran: 134 agents, 110 adversarial verifications, 2 syntheses. Their verdict
tallies are **sweep 1: 52 weakened / 8 refuted / 0 confirmed** and **sweep 2: 21 weakened /
25 refuted / 4 confirmed**. Both re-counted directly from the journals for this file and both
reproduce exactly. Those tallies are the reason Stage 0 exists — and they are *not* neutral
measurements of a retraction rate. The adjudicator who established this wrote:

> CAVEAT THE PLAN SHOULD CARRY (does not change the count): "0 confirmed" is partly instructed,
> not a neutral audit result. The verifier prompt (script lines 234-239) says "Try to REFUTE it
> ... Default to 'weakened' or 'refuted' when the evidence is thin." A tally produced under an
> instructed refute-by-default prior is weak support for the plan's inference at line 18-19
> ("A third sweep that ignored them would reproduce that retraction rate") and for the Risk-table
> row "A third sweep repeats the retraction rate". The numbers are real; their interpretation as
> a retraction *rate* is inflated by the prompt.

and the form the plan adopted:

> Both sweeps instructed verifiers to "try to refute" and to default to weakened/refuted on thin
> evidence, so these are **upper bounds on the retraction rate, not neutral estimates of it**.

The instructions themselves, quoted verbatim from the workflow scripts:

> `ard36-minority-analytics-wf_4cd6c827-dfa.js:234-239` — "Independently RE-DERIVE this from the
> parquet tables. Do not trust their query; write your own. **Try to REFUTE it**: check the
> denominator, check whether ungradeable rows were excluded, check whether the cell sizes support
> the claim, check whether a confound (item difficulty, different item sets across checkpoints,
> selection on n_correct) explains it. If the numbers are right but the claim overstates them,
> return 'weakened' with a corrected claim. **Default to 'weakened' or 'refuted' when the
> evidence is thin.**"

> `ard36-minority-dynamics-wf_d58bbdc9-773.js:248-255` — "Independently RE-DERIVE it. Write your
> own query; do not trust theirs. **Attack it** in this order: (1) IS IT COMPOSITION? … **Default
> to 'weakened' when thin and 'refuted' when the balanced-panel version reverses or vanishes.**"

Both scripts' `VERDICT_SCHEMA` enum is `['confirmed','weakened','refuted','unverifiable']`, and
sweep 1's Verify phase is titled "adversarial re-derivation of every claim". The consequence for
this register: a Q id records **a defect that was found**, which is a positive finding and
survives the prior. It does **not** record that the claim it damaged was actually false. Anywhere
a Q id is used to justify discarding a v1 number, the justification is the defect's own evidence
column below, never the sweep tally.

## How to read the register

**Class.** `REPAIRABLE DEFECT` — a fault in the data or the code that produced it, fixed by a
Stage 0 step writing `tables_v2/`. `DESIGN/POLICY FACT` — a property of the study as designed, or
of the signal itself; it is not fixable and enters as an estimator rule or a declared limit. The
plan's split is **24 repairable + 6 policy (Q17–Q20, Q27, Q30)**; see the Q29 resolution below
for the one id where that count is contestable.

**Evidence.** Every cell names a table, a column and a count. Cells marked **✓** were re-derived
against the frozen v1 tables while writing this file (`tables/*.parquet`, read-only, column reads
only, no new compute) and reproduced to the digit. Cells marked **(sweep)** are quoted from the
register as raised and were *not* re-derived here — treat them as claims with a source, not as
verified facts.

**Status.** No repair has been executed: `tables_v2/` does not exist as of this writing
(`git rev-parse HEAD` = `30dd53b7aabd2e5c72a451cb3d0aef389261c6c9`). Status names the Stage 0
step or estimator rule that owns the id.

## Mechanisms — the 30 findings collapse to 9

Transcribed from the source map; the "class" column is the source's own.

| id | mechanism | member findings | where it lives | class |
|---|---|---|---|---|
| **M1** | ambiguous coerced to wrong | Q4 Q5 Q6 Q16 Q22 | `graders.GradeResult.to_row`, `compact._is_correct`, `compact.label_rows`, `analysis.within_item_rows` | changes published numbers |
| **M2** | shared-root aliasing without a team/cell key | Q1 Q2 Q3 Q15 Q29a | `runner` ROOTS/FORK protocol default; `compact.episode_rows` / `round_summary_rows` / `selection_rows`; `minority.build_frame` team filter | changes published numbers |
| **M3** | validity vs class-switch vs grader-class conflation | Q7 Q9 Q10 Q11 Q21 | `compact.transition_rows` L747, `round_summary_rows` L484, `minority.build_frame` L177-187 | changes published numbers |
| **M4** | metering / instrumentation gaps | Q8 Q13 B-1 E3-margin | `resources.request_resources_from_decisions`, fork executor rows, `compact` ignoring `events` | mixed: B-1/Q13 change cost and fork numbers; Q8/E3-margin add capability |
| **M5** | fork provenance | Q12 Q14 | `compact.fork_rows` (`parent_decision_ids` from a field forks never set); `frames.build_paired_rows` already has the right join | adds capability |
| **M6** | evidence never surfaced | Q23 Q24 Q25 Q26 Q28 | no receiver table; bank axis unjoined; `round_selection`, `agent_id`, `leader_id`, `evidence_text` unused | adds capability |
| **M7** | design asymmetries | Q27 Q29b | not repairable; enter as strata/covariates | policy |
| **M8** | dependence | Q30 | every sweep p-value computed with iid SEs | policy |
| **M9** | degenerate / censored signals | Q17 Q18 Q19 Q20 | structural; policy on what may be estimated | policy + limits |

Arithmetic: 31 Q slots across nine mechanisms, 30 distinct Q ids (Q29 appears twice — resolved
below), plus two non-Q ids in M4. Union is exactly Q1…Q30 with no gaps, no Q31, and no invented
ids.

### Q29 is listed under both M2 and M7 — resolution

The double listing is **deliberate at source**, not a transcription slip: the map writes
"Q29(hb r0)" under M2 and plain "Q29" under M7. Q29 as raised is a *compound* entry — its title
is "Item-set and coverage asymmetries that are design facts rather than defects", but one of the
things bundled into it is an actual keying defect. Leaving it as one id makes the Class column
unanswerable, so this register **splits it and uses the split ids everywhere**:

- **Q29a** (M2, REPAIRABLE DEFECT) — the high-budget cohort's round 0 runs at allowance 1024
  while its rounds 1–3 run at 4096, and `minority_frame.reasoning_allowance` faithfully reports
  1024 for those rows. An analyst selecting the budget arm with `reasoning_allowance == 4096`
  therefore silently drops the cohort's entire round 0, and the arm cannot be identified from
  `minority_frame` alone — it needs `episodes.final_allowance`. Repaired by carrying both
  `episode_allowance` and `decision_allowance` (Stage 0 step 9).
- **Q29b** (M7, DESIGN/POLICY FACT) — the item-set and coverage asymmetries proper: 32B runs a
  frozen 96-item subset, GATE and RANDOM_PROTECT run 240 of 384, `difficulty` exists only for
  math500. Not repairable; enters as strata (estimator rule E10) and as declared limits.

Counting consequence, stated plainly: the plan's headline **24 repairable + 6 policy** holds only
if Q29 is counted once, as repairable. Under the split there are 30 ids but **31 register rows** —
24 repairable (including Q29a) and 7 policy rows (including Q29b). The plan's 6-policy list
(Q17–Q20, Q27, Q30) is correct as a list of ids that are *wholly* policy. Both statements are
true; they count different things. Stage 0's step-to-id mapping is unaffected.

## The register

| id | finding | M | class | evidence — table · column · count | status |
|---|---|---|---|---|---|
| **Q1** | Round-0 rows are replicated across protocol arms from a shared root bank, so any round-0 cross-protocol comparison is a set compared with itself | M2 | REPAIRABLE | `minority_frame` 55,360 rows / **46,080 distinct `decision_id`**; round 0 = **11,680 rows from 2,400 distinct** decisions (480 root teams × 5 agents), duplication ×4.87 ✓. Replicas agree on `correct`/`n_correct`/`plurality_class`/`confidence` (sweep: 0 of 2,400 vary). Also exploitable: it makes round 0 an exactly matched counterfactual across arms | OPEN — Stage 0 steps 1, 3, 9 (`cell_kind` key; emit `is_alias` / `physical_team_id` rather than dropping) |
| **Q2** | `minority_frame` is missing **all** round-0 rows for the 480 natural PRIVATE episodes (2,400 agent-rows) | M2 | REPAIRABLE | `minority_frame`: 480 PRIVATE/natural episodes present, **0 of them with a `round==0` row** ✓. The same episodes have round-0 rows in `decisions`, `round_summary` and a `round_from=0` row in `transitions`. Root cause is `minority.py:84-90`'s `len(members) == N_AGENTS` filter firing on the 16+5 = 21-member collision, **not** a protocol filter | OPEN — Stage 0 steps 1, 9 (`minority_frame_v2`; target PRIVATE r0 = 2,720 rows / 544 episodes) |
| **Q3** | `round_summary` carries spurious round-0 rows from the 16-draw ROOTS bank, reusing PRIVATE `episode_id`s verbatim | M2 | REPAIRABLE | `round_summary`: **504 rows with `n_valid + n_unparsed == 16`, every one `protocol='PRIVATE'`, `cohort='natural'`, `round=0`** ✓; PRIVATE round 0 is 984 natural rows = 504 bank + 480 real teams ✓; 480 duplicate `(episode_id, round)` groups ✓. Clean team rows are the 11,552 with sum 5; the 1,280 with sum 1 are receiver probes ✓. Consequence (sweep): `c_dist`'s C0 axis runs to 16 for PRIVATE only, giving `P(C3=0 \| C0=1) = 0.667` where the truth is 1/6 | OPEN — Stage 0 step 1 (skip ROOTS in `round_summary_rows`, −504; assert no row has `n_valid + n_unparsed == 16`) |
| **Q4** | Half of raw `minority_correct` rows come from partially-graded teams | M1 | REPAIRABLE | `minority_frame`, 1,048 MC rows by `n_graded`: **{1: 69, 2: 213, 3: 104, 4: 138, 5: 524}** — 524 (50.0%) have `n_graded < 5` ✓; **69 of 366 `lone_correct` rows have `n_graded == 1`** (the agent is the only graded member) ✓. Requiring `n_graded == 5` costs 1,048 → 524 rows, 394 → 222 episodes, 106 → 67 items (sweep) | OPEN — Stage 0 step 2 (tri-state) + step 14 panel **P2 class-complete**, required for any `n_correct`-conditioned statistic |
| **Q5** | Ungradeable rows split cleanly by family, with different mechanisms — mmlu_pro has no parsed class at all, math500 has a class the grader could not resolve | M1 | REPAIRABLE | `labels.evaluator_status` = correct 37,198 / incorrect 9,115 / **ambiguous 5,046 / timeout 1**, and `is_correct` has **0 nulls** ✓. **All 3,353 mmlu_pro ungraded `minority_frame` rows are `valid=False` with null `answer_class`; all 2,016 math500 ones are `valid=True` with a parsed class** ✓. Consequence: in mmlu_pro teams `n_correct` is already the true count | OPEN — Stage 0 step 2 (`team_class_complete`), step 12 (relaxed grader + Manski bounds) |
| **Q6** | `transitions.parquet` coerces ungradeable decisions to `correct_from` / `correct_to` = `False` | M1 | REPAIRABLE | `transitions` 42,240 rows; `correct_from` and `correct_to` are **non-nullable bool, 0 nulls** ✓. 4,112 rows affected, manufacturing **377 phantom correct→wrong events** (sweep). Locus is `graders.GradeResult.to_row` (`graders/__init__.py:76`) emitting `is_correct = (status == STATUS_CORRECT)`; `compact._is_correct`'s `None` branch never fires | OPEN — Stage 0 steps 2, 5 (nullable correctness; the 377 phantom transitions must disappear) |
| **Q7** | `transitions.changed_class` conflates an answer switch with a validity change | M3 | REPAIRABLE | `transitions`: `changed_class == True` on 3,952 rows, of which **valid→invalid 742, invalid→valid 839, genuine valid→valid switch 2,371**; invalid→invalid (2,152) coded `False` ✓. **Overcount 1,581 of 3,952 = 40.0%** ✓ | OPEN — Stage 0 step 5 (split into `switched_class` / `lost_validity` / `gained_validity`) |
| **Q8** | `packet_tokens` is 0.0 on every row of every protocol, including PEER; `seen_message_ids` is empty for the whole receiver arm | M4 | REPAIRABLE (capability) | `decisions.packet_tokens`: **min 0, max 0, nonzero 0 of 67,104** ✓. `decisions.seen_message_ids` empty on **all 1,280 receiver rows** ✓ (it is populated elsewhere — 26,880 rows overall — so the gap is receiver-specific, which the source states and which is worth not overstating) | OPEN — Stage 0 step 11 (`packet_tokens_marginal` by re-rendering and tokenizing the packet-free prompt; r1 only, since r2–3 have no stored packet-free counterpart) |
| **Q9** | `coverage_current` is a pure re-encoding of `n_correct > 0` and carries no information; `coverage_archive` is monotone by construction and accrues too little | M3 | REPAIRABLE | `coverage_current` is 1.000 on all strict MC rows and 0.000 on all `n_correct == 0` rows (sweep); shipped `coverage_archive` disagrees with a recomputed cummax on **30 of 11,264 rows (0.27%), always too low** (sweep). Root cause: `compact.py:484` adds a class to the archive only `if ok and cls`, and 454 graded-correct decisions carry a null runtime `answer_class` | OPEN — Stage 0 step 7 (`coverage_archive_cummax`) |
| **Q10** | Rows that are graded but carry no parsed answer class (`valid == False`, `answer_class` null, `own_class_share == 0`) | M3 | REPAIRABLE | `minority_frame`: **472 gradeable rows with null `answer_class`, of which 352 were graded CORRECT** ✓; **52 `minority_correct` rows have `own_class_share == 0` with null `answer_class`** ✓; `own_class_share` is 0 — not null — on all **3,825** invalid rows ✓. A systematic grader-vs-canonicaliser disagreement, and the source of Q9's archive shortfall | OPEN — Stage 0 steps 5, 7 |
| **Q11** | `minority_correct` ≠ (`correct` AND `own_class_share ≤ 2`) — a team can hold two distinct canonical forms of the same correct answer | M3 | REPAIRABLE | `minority_frame`: **395 rows are `correct` with `own_class_share ≤ 2` yet `minority_correct` is False** ✓. `against_plurality` is True only for correct agents, so it cannot construct a wrong-minority state (sweep). 219 correct→correct steps changed answer class (sweep; **the audit re-derived this as 160 flagged, of which 159 are validity flips and exactly 1 is a genuine both-valid switch — see Contradictions**) | OPEN — Stage 0 step 5 (`alternate_correct_switch`, to be reported as "not estimable (1 event / 1 item)") |
| **Q12** | The 144 fork-cohort episodes exist only at rounds 2–3 and have **no rows at all** in `transitions.parquet` | M5 | REPAIRABLE (capability) | `transitions.cohort` = {natural 40,320, high_budget 1,920} — **fork absent entirely** ✓. `minority_frame`: 1,440 fork rows, 144 fork episodes, rounds **{2: 720, 3: 720}** only ✓. Forks branch off parent rounds 1 and 2, never round 0, so the most minority-dense exposure is outside the experiment (sweep) | OPEN — Stage 0 steps 5, 13 (fork pair rows via `shared_prefix_hash → parent_state_id`; `fork_pairs.parquet`) |
| **Q13** | All 1,440 fork decisions carry `thinking_tokens`, `final_tokens`, `prompt_tokens` and `latency` of exactly 0 — a metering gap, not a degenerate arm | M4 | REPAIRABLE | `decisions`: **1,440 fork rows, all four fields 0 on all 1,440** ✓, while **`flops_total` is non-zero on all 1,440** ✓ — and the FLOP oracle is documented as a pure function of token counts, so one of the two values is wrong. Root cause: `runner._ref_from_wp2_row` reads metering from the WP2 row and gets zeros when the fork executor did not attach the record | OPEN — Stage 0 step 6a (`ard/remeter.py`, replay through `client.decision_from_record`; gate on reproducing v1 exactly for 5,000 non-fork decisions) |
| **Q14** | Fork provenance columns are broken | M5 | REPAIRABLE (capability) | `episodes.parent_ref` is **one constant state hash shared by all 288 fork episode records** ✓; `forks.parent_decision_ids` is **`[None]` on all 1,440 rows** ✓. Linkage is recoverable but only by inference — `(shared_prefix_hash → parent_state_id, focal_agent)` matches 1,440/1,440 (sweep) | OPEN — Stage 0 step 13 |
| **Q15** | `episodes.parquet` has 648 duplicate `episode_id`s | M2 | REPAIRABLE | `episodes`: **4,888 rows, 4,240 distinct ids, excess 648** ✓ = 504 ROOTS + 144 FORK second registrations ✓ (`PRIVATE/natural` holds 984 episode records where only 480 are real teams; `PRIVATE/fork` holds 288 for 144 episodes) ✓. Silently fans out any merge | OPEN — Stage 0 steps 1, 3 (`cell_kind` on `episode_rows`; assert `(cell_kind, episode_id)` unique) |
| **Q16** | The shipped `capitulated` column folds ungradeable-next rows into `False`; `a_correction` / `d_destruction` and `correct_from`/`correct_to` likewise code ambiguous and timeout as wrong | M1 | REPAIRABLE | 357 ungradeable-next rows counted as "not capitulated" (sweep); 46% of the relevant `a_correction` cells contain at least one such agent (sweep); 19 of 47 round-3 "erasures" are rounds in which not one of five agents was gradeable (sweep). Under-pinned by Q6's verified non-nullable bools ✓ | OPEN — Stage 0 step 2 (all 30 correctness-derived `round_summary` columns) |
| **Q17** | `confidence` is missing on 3.49% of rows and the missingness is strongly non-random and group-dependent | M9 | DESIGN/POLICY FACT | `minority_frame.confidence`: **1,930 nulls of 55,360 = 3.486%** ✓. Missingness by group (sweep): minority_incorrect 26.0%, wrong_plurality 9.0%, minority_correct 4.6%, majority_correct 0.63%; **11.60% among cap-hit rows vs 0.018–0.02% among non-cap**; 98.88% among `parse_status='truncated_after_final'`. `answer_logprob` missing at only 2.6% overall but **missing for the entire 1,440-row fork cohort** | POLICY — estimator rule **E7** (missing-is-missing, rate printed, IPW sensitivity; missingness depends on `boundary_injected`). Never repaired |
| **Q18** | `confidence` is a near-degenerate, censored grid | M9 | DESIGN/POLICY FACT | `minority_frame.confidence`: **23 distinct values; 95.44% of mass ≥ 0.95; the four modal atoms carry 95.38%** ✓; mean 0.9669, sd 0.062 (sweep). 88% of persister steps show exactly zero change; 66.7% of within-cell pairs tie outright; round 0 uses a coarser vocabulary than later rounds (sweep) | POLICY — estimator rule **E8** (exact-atom ECE, tie rate printed, **no confidence effect may be a headline**). Stage 0 step 15 wires the discrete ECE dispatch but does not change the signal |
| **Q19** | `thinking_tokens` is hard-censored at the reasoning allowance, and the frame's `cap_hit` is exactly the indicator `thinking_tokens == allowance` | M9 | DESIGN/POLICY FACT | `minority_frame`: **`cap_hit == (thinking_tokens == reasoning_allowance)` on 53,920 / 53,920 non-fork rows, zero mismatches** ✓ — the identity is exact and holds at 4096 as well as 1024, which is stronger than the source's "12,644/12,644 at 1024". **The column is mislabelled**: `minority_frame.cap_hit` equals `decisions.boundary_injected` on 55,360/55,360 rows and `decisions.cap_hit` on only 40,668/55,360 ✓. Panel-wide: `boundary_injected` 22,200 rows vs `cap_hit` (`finish_reason == "length"`) 2,658 ✓ | POLICY — estimator rule **E8** (`thinking_tokens` only as `P(reached allowance)` or within the 4096 arm). Stage 0 step 9 splits the misnamed column into `boundary_injected` / `cap_hit_response`, which is a naming repair, not a signal repair |
| **Q20** | `answer_length` is overwhelmingly determined by `(item_id, answer_class)`, not by the agent, so it is not an agent observable | M9 | DESIGN/POLICY FACT | Re-derived panel-wide on `minority_frame` (55,341 non-null rows, 1,000 groups): **R² = 0.7205** ✓. The source's 98.85–99.99% is real but scoped to 4,870 mixed-cell rows and to math500 alone — see Contradictions. Either way the leave-one-episode-out `(item, answer_class)` mean reproduces the whole within-cell effect while the residual is null (AUC 0.527) (sweep) | POLICY — estimator rule **E8**; excluded from any "minority-correct agents are identifiable" headline |
| **Q21** | `class_entropy` carries no information beyond `n_unique_classes` plus the split shape; `own_class_share` ≡ `n_correct` for correct agents | M3 | REPAIRABLE | `minority_frame.class_entropy` takes **13 distinct values panel-wide**; the source's 7-value ladder {0, 0.500, 0.673, 0.950, 1.055, 1.332, 1.609} is exactly the full-5-member-team subset, the other six values coming from teams with `n_valid < 5` ✓ (scope qualification added here). Within `n_unique_classes == 2` entropy has effectively two values (0.500 = 4-1, 0.673 = 3-2). `own_class_share` **is** `n_correct` for a correct agent with a parsed answer in 99.01% of rows (sweep) | OPEN — Stage 0 step 5; and a standing rule that entropy adds nothing beyond the split, so its within-cell "signal" (AUC 0.250) is label-derived and unusable |
| **Q22** | Ten purpose-built `round_summary` columns were never used by the sweep, and two of them are artefactual where they were checked | M1 | REPAIRABLE | Unused: `persistence_same_agent`, `persistence_without_original_holder`, `dissemination_new_agents`, `wrong_minority_persists`, `novel_correct_on_c0_zero`, `unanimous_wrong_escape`, `first_correct_loss_round`, `recovery_round`, `survives_to_round3`, `new_correct_class_after_loss` — all ten present in the schema ✓. `persistence_without_original_holder` is artefactual for PRIVATE natural and fork episodes because `root_holders` is set only at round 0, which those episodes lack in team form (Q2) — inflating PRIVATE to 130/129 True at rounds 2/3 (sweep). `new_correct_class_after_loss` is True on 17 episode-rounds in the whole study (sweep) | OPEN — Stage 0 step 2 (they are correctness-derived, so they move under tri-state); the artefact itself is fixed by step 1 |
| **Q23** | An entire randomised arm — 1,280 receiver episodes — exists in `episodes.parquet`, never appears in `minority_frame`, and was never analysed | M6 | REPAIRABLE (capability) | `episodes`: **1,280 rows at `protocol='NONE', cohort='receiver'`** ✓, absent from `minority_frame` ✓, **ungraded — all 1,280 have no label row** ✓ (the only other unlabelled decisions are 128 root-phase rows, also `protocol='NONE'`) ✓. `ard_stats.json` records `/secondary/receiver/status = unexecuted`, `n_contrasts = 0`. 31 of 64 probes are degenerate (sweep). The only place in the run where social framing is manipulated with content held fixed | OPEN — Stage 0 step 10 (grade via `graders/trace.py`, **not** the GRADE lane, which raises `ValueError("unknown grader family")`; write a separate file so main-panel denominators are untouched) |
| **Q24** | `bank_summary.parquet` — an exogenous 16-draw difficulty axis, independent of the panel — had zero queries | M6 | REPAIRABLE (capability) | `bank_summary`: **480 rows** carrying `pass_at_{1,2,3,5,9,10,16}`, `oracle_coverage`, `selection_gap`, `correct_plurality_status` (distribution over rows: strict_majority 341, none 79, below_plurality 43, unique_plurality 12, tied_plurality 5) ✓. The sweeps' item bins were called "partly self-referential" by three verdicts while the non-circular axis sat unused; the circularity is worth about 2.9× (sweep) | OPEN — Stage 0 steps 2 (graded-only pass@k + Manski upper bound; 918 of 7,808 draws are ungradeable and currently scored as failures) and 9 (join the bank axis into `minority_frame_v2`) |
| **Q25** | `round_selection.parquet` (200,040 rows) had zero queries — nobody asked what the team actually answered; every finding proxied team outcome with `n_correct` | M6 | REPAIRABLE (capability) | `round_selection`: **200,040 rows, 5 selectors × 40,008** ✓; **`selected_correct` null on 129,272 rows**, of which **120,024 are the three selectors that are 100% null — `CURRENT`, `dynamics_only`, `llm_audit`** ✓ (`plurality` and `confidence_weighted` contribute 4,624 nulls each). Joined to fully-graded team states, a lone correct agent reaches the team output 0 of 186 times conf-weighted and 2 of 186 under plurality (sweep) | OPEN — Stage 0 step 4 (populate or mark structurally unavailable and **remove those contrasts from the decision-matched family**, re-declaring the family size in code) |
| **Q26** | `agent_id` was never used — the study is about agents and no analysis used the agent index, the hub seat, or `episodes.leader_id` | M6 | REPAIRABLE (capability) | `minority_frame.agent_id` and `episodes.leader_id` are both present and unread ✓. The sweep's seat effect (MC rate by `agent_id` 1.17 / 0.93 / 1.31 / 1.20 / 1.35%, χ² p = 0.043, n = 10,583 each) **did not reproduce at panel scope**: panel-wide the rates are 1.99 / 1.67 / 2.06 / 1.89 / 1.86% on 11,072 rows each and the ordering differs. The n = 10,583 panel is not defined in the source, so the discrepancy is a scope gap, not a refutation — **the seat-effect numbers must be re-derived on a declared panel before use** | OPEN — Stage 0 step 14 (declare the panel), then Stage 2 category 1. The hub result (0 MC rows in 973 hub rows vs 18 in 3,892 spoke rows, Fisher p = 0.034) is likewise panel-dependent and was independently re-derived by the audit as **5/1,071 vs 2.61%** — see Contradictions |
| **Q27** | The signature dimension's headline marker is a budget artefact, and budget was declared "not measurable" | M7 | DESIGN/POLICY FACT | Receiver arm, randomised 640/640 within item: truncation **7.66% @1024 vs 0.47% @4096**; `final_tokens` 93.2 vs 75.0; cap_hit 7.5% vs 0.5% (sweep). Main panel, 32 matched items × {PEER, PRIVATE}, within item×checkpoint×protocol: truncation 1.49% vs 0.41%, Wilcoxon p = 0.033 (sweep). The "single largest behavioural discriminator" is 10–16× determined by a knob never entered as a covariate | POLICY — estimator rule **E10** (stratify, never pool, across budget). Not repairable |
| **Q28** | Four further tables and one column had zero queries | M6 | REPAIRABLE (capability) | All present and unread ✓: `calibration_panel` **300 rows**, `calibration_system_minus_agent` **180 rows**, `drift` **441 rows** (`a_of_m`, `d_of_m`, `predicted_delta_m`, `realized_delta_m`, `residual`), `cost_accuracy` **59 rows** with its `flop_matched` / `decision_matched` arms, and `decisions.evidence_text` — **the text agents actually exchanged was never read**, so every statement about conformity, copying and argument content is inferred from class transitions alone | OPEN — Stage 0 step 15 (calibration; note `calibration_panel`'s own clustering bug — it clusters on an `item_id` read off `decisions`, which has no such column, so every "within-item" figure in it is pooled) and Stage 2 category 6 |
| **Q29a** | High-budget round 0 runs at allowance 1024 while its rounds 1–3 run at 4096, so the budget arm cannot be identified from `minority_frame` alone | M2 | REPAIRABLE | `minority_frame`, `cohort='high_budget'`: **32 items, 2,560 rows, `(reasoning_allowance, round)` = {(1024, 0): 640, (4096, 1): 640, (4096, 2): 640, (4096, 3): 640}** ✓ — allowance is perfectly confounded with round inside the cohort. **No round-0 row anywhere in the panel has `reasoning_allowance == 4096`** ✓. Of the cohort's 56 MC rows, 22 sit at round 0 / 1024 ✓. Identification needs `episodes.final_allowance` | OPEN — Stage 0 step 9 (carry both `episode_allowance` and `decision_allowance`) |
| **Q29b** | Item-set and coverage asymmetries that are design facts: the arms do not run the same items | M7 | DESIGN/POLICY FACT | `minority_frame`: **32B runs 96 items, all a strict subset of 8B's 384** ✓; **GATE and RANDOM_PROTECT run 240 items each, a strict subset of the other five arms' 384** ✓; `difficulty` is populated (levels 1–5) for all 27,680 math500 rows and is the **empty string on all 27,680 mmlu_pro rows** ✓, so difficulty is collinear with family. Consequence: "common item set" restrictions do almost no work because GATE/RP items are nested, and absolute rates are never comparable across checkpoints | POLICY — estimator rule **E10**; Stage 0 step 14 declares panels P3–P5 and P8 so the asymmetry is explicit rather than latent. Not repairable |
| **Q30** | Non-independence that row-level n's hide: rows within an item are not independent, and protocol arms of the same item share round 0 | M8 | DESIGN/POLICY FACT | `minority_frame`: **1,048 MC rows rest on 394 episodes and 106 items** ✓; at 32B, **135 MC rows on 15 items**, and in the 4096 arm **34 rows on 6 items** ✓ — both below any ≥10-item floor. The 543 minority cells come from 83 items; the sweep-2 churn DiD bootstrap has 21 clusters (sweep). Row-level n overstates precision by roughly an order of magnitude wherever an item-clustered interval is not quoted | POLICY — estimator rules **E2** (two-way item × episode clustering; **measure** the inflation rather than asserting ~10×) and **E6** (minimum cell). Not repairable; this is the single id that most constrains what Stage 2 may claim |
| **B-1** | GPU-second attribution is identically zero because the writer and the reader disagree on a key name | M4 | REPAIRABLE | `compact.decision_rows` writes `"latency"` (compact.py:305, `"latency": decision.latency_s`) while `resources.request_resources_from_decisions` reads `"latency_s"` (resources.py:316-317, `latency_s=float(row.get("latency_s") or 0.0)`) — confirmed in source ✓. `decisions.latency` is populated on **65,664 of 67,104 rows** (the 1,440 zeros are Q13's fork rows) ✓, so the data is there and only the attribution is lost — 268.747 request-hours. `retries` / `errors` are never emitted by `decision_rows` either | OPEN — Stage 0 step 6b |

Arithmetic check on the table above: **32 rows** — 25 `REPAIRABLE` (17 data/number repairs +
8 capability additions, B-1 among them) and 7 `DESIGN/POLICY FACT`. Dropping B-1, which is not a
Q id, leaves **24 repairable Q ids** (Q1–Q16, Q21–Q26, Q28, Q29a) and **7 policy rows** on 6
wholly-policy Q ids (Q17–Q20, Q27, Q30) plus Q29b. That reconciles with the plan's
"24 repairable + 6 policy" exactly, and the only difference is the Q29 split.

### One further id in the mechanism map that is not a defect

**E3-margin** appears as the fourth member of M4. It is not a Q id and not a defect: **E3** is the
receiver experiment (`contract.py:265`, `RECEIVER = "receiver"  # E3 synthetic program traces`)
and "E3-margin" is the *capability* of recovering the answer-logit margin for that arm from
`response.top_logprobs`, which is retained at inner k = 20 on receiver records and only k = 1 on
natural and fork records. It is delivered by Stage 0 step 10 alongside Q23. Recorded here so the
map's member list resolves completely; it is not counted in the 30.

## Findings the audit added that the Q series does not contain

These are real, were found *after* the sweeps, and are **not** Q ids. Do not retro-fit them into
the Q numbering — give them their own ids if they need one.

- **Three of five selectors emit `selected_correct` entirely null** (120,024 of 200,040 rows).
  Filed above under Q25's evidence because it is the same table, but the sweeps raised Q25 as
  "nobody queried this table", not as "this table is 60% null". The null selectors are the
  generator of the "0/N" artifacts and silently void 2 of the 6 decision-matched contrasts.
- **`calibration_panel.build_panel` clusters on an `item_id` read off `decisions` rows, and
  `decisions.parquet` has no `item_id` column** — every item is `None`, all rows collapse to one
  pseudo-cluster, so every "within-item" statistic in `calibration_panel.parquet` is pooled.
- **No v1 calibration numbers exist at all**: `ard_stats.json` (11:40:04) predates
  `calibration_panel.parquet` (11:41:56), so every calibration entry is `status: unavailable`.
  There is no v1 baseline to diff a calibration repair against.
- **`compact.SCHEMAS` is a strict whitelist** and `_project` (compact.py:1058-1064) raises on any
  undeclared column, so the ~12 columns Stage 0 adds will abort recompaction on the first one
  unless the schemas are extended first.
- **FORK episodes also default to `protocol=PRIVATE`**, not just ROOTS: `round_summary` PRIVATE
  r2 and r3 each carry 144 excess fork rows (verified here: PRIVATE by round = {0: 1,048,
  1: 544, 2: 688, 3: 688}; 688 − 144 = 544 ✓). A round-0-only gate misses this.
- **Fork rows have zero token counts but non-zero FLOPs** while the FLOP oracle is documented as a
  pure function of token counts — one of the two v1 values is wrong (Q13's evidence column).

## Where this register disagrees with its own sources

Recorded rather than smoothed over, because a register that hides its disagreements is worth
less than one that shows them.

1. **Q5's family split, 3,353 / 2,016 vs 3,166 / 1,881.** The plan flags this as an arithmetic
   error ("v1's pair sums to 5,369 > its own 5,047"). It is not an error, it is a **unit
   mismatch**, and both figures are right at their own unit: 3,353 mmlu_pro + 2,016 math500 =
   5,369 **`minority_frame` rows**, and 3,166 + 1,881 = 5,047 **distinct labelled-ungradeable
   `decision_id`s** — both re-derived here ✓. The gap is Q1's round-0 replication. Quote the
   5,047 pair for anything decision-level and the 5,369 pair for anything row-level, and say
   which.
2. **Q19's `cap_hit`.** The register writes "cap_hit ≡ thinking_tokens == 1024". The identity is
   exact and more general than that — it holds against `reasoning_allowance` on all 53,920
   non-fork rows ✓ — but the column is **mislabelled**: `minority_frame.cap_hit` is
   `decisions.boundary_injected` (22,200 rows panel-wide), not `decisions.cap_hit`
   (`finish_reason == "length"`, 2,658 rows). Anything reasoning about "hitting the cap" must say
   which of the two events it means.
3. **Q20's `answer_length` determination.** The register says 98.85–99.99%; the audit's
   re-derivation says 70.5%; this file measures **72.05%** panel-wide on `(item_id,
   answer_class)` ✓. The three are not in conflict — 98.85% is a 4,870-row mixed-cell scope and
   99.92% is math500 alone — but **the headline generalisation does not hold panel-wide**. Use
   ~70% and name the panel. The conclusion (not an agent observable) is unchanged either way.
4. **Q11's "219 correct→correct class changes".** The audit re-derived this as **160** flagged,
   of which 159 are validity flips, leaving **exactly 1** genuine both-valid switch — and that
   one is a symbol-case artefact (`2x + 3` → `Y = 2X + 3`). Report it as "not estimable
   (1 event / 1 item)". Not re-derived here.
5. **Q21's entropy ladder.** The 7-value ladder is the full-5-member-team subset; panel-wide the
   column takes 13 values ✓, the extra six coming from teams with `n_valid < 5`. Scope the claim.
6. **Q26's seat effect did not reproduce at panel scope** (see its row). The source's panel
   (n = 10,583 per agent) is undefined; the panel-wide rates differ in both magnitude and
   ordering. The *capability* claim — `agent_id` and `leader_id` were never used — stands ✓; the
   *finding* must be recomputed on a declared panel.
7. **Q1's duplication factor.** The register gives ×4.88 on a strict panel; the audit gives
   4.867× in `minority_frame` v1 rising to 5.867× in v2, with per-decision multiplicity 3/5/7/9
   rather than a constant. This file measures 11,680 / 2,400 = **×4.867** ✓ on the full frame.
   The calibration-panel inflation is 2.878× panel-wide, **not** 4.9×.

## What was and was not verified for this file

**Verified** (✓ in the table): every count re-derived by reading columns from
`/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36/tables/*.parquet` with
`pyarrow.parquet`, read-only, no new generation, at `git rev-parse HEAD =
30dd53b7aabd2e5c72a451cb3d0aef389261c6c9`. Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q10, Q11, Q12, Q13,
Q14, Q15, Q17, Q18, Q19, Q20, Q21, Q23, Q24, Q25, Q28, Q29a, Q29b, Q30 and B-1 each have at
least one figure re-derived to the digit; the two sweep tallies were re-counted from the workflow
journals and match exactly (52/8/0 and 21/25/4 over 60 and 50 verifier returns).

**Not verified here, and marked (sweep) in the table**: Q9's 30-of-11,264 archive disagreement;
Q16's 357 folded rows and 46% figure; Q22's artefact magnitudes; Q26's and Q27's effect sizes;
Q23's and Q25's downstream estimates; Q24's minority-concentration rates. These are quoted with
their source and must be re-derived on `tables_v2` before any of them is published.

**Not recoverable from any source**: nothing. All 30 Q ids and B-1 resolved to substantive
content; there is no `UNRECOVERED` row. Two source cells were empty in the original register —
Q15's evidence cell and Q28's consequence cell were both em-dashes — and both are filled above
from measurements made for this file rather than from the source.

**Deliberately not asserted**: that the defects below constitute a "50% retraction rate", or that
a third sweep would reproduce one. The sweeps' verdicts were produced under an instructed
refute-by-default prior, so the tallies bound that rate from above and nothing here estimates it.
