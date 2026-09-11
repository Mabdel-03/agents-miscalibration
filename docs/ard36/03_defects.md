# ARD36 known defects and open items

Updated at every gate, per the plan's verification section. study_v4's defect list was
reconstructed afterwards from the atlas appendix; this one is written as the run goes, so a
reader of the final report can tell what was known at the time from what was found later.

## Open

### D-01 — parity panel must exclude prefilled positions for cap-hit draws
**Severity: medium; affects a diagnostic, not the captured vectors.**

A cap-hit draw is now generated in two phases (amendment I6), and its stored
`completion_token_ids` are the stitched sequence: thinking, `</think>`, the prefilled
`\n\n<final>`, then the phase-2 report. Replay of that sequence is *correct* — a transformer's
hidden state at position *i* depends only on tokens 0…*i*, and phase 2's forward pass saw
exactly this prefix — so the anchor-to-hidden-state mapping and every captured vector are
sound.

What is not sound is the fidelity statistic. `replay.fidelity` scores argmax agreement against
the *sampled* token at each position, but the prefilled `<final>` tokens were injected, not
sampled. Scoring them will depress argmax agreement on exactly the cap-hit draws and could
trip the hour-0–2 replay parity gate for a reason that is not a fidelity problem.

*Fix:* exclude the prefill span (its length is on the record's `continuation` block) from the
argmax-agreement denominator, and report agreement separately for naturally-closed and
continued draws. Owner: whoever runs the E6 capture, before the parity panel is read.

### D-02 — GPU-second attribution omits phase-2 prefill
**Severity: low; affects one cost column, not FLOPs.**

`ard/resources.py` derives attributed GPU-seconds from the decision row's token counts, so a
continued draw's phase-2 prefix prefill and phase 1's discarded post-boundary span fall outside
it. Logical FLOPs are correct end to end (both phases are summed on the record). This affects
only the wall-clock attribution column, which is documented as an approximation.

**Correction (2026-09-10).** An earlier revision of this entry claimed `physical_requests` was
"now correct too". It was not. `DecisionRecord` carried the two ids, but the joinable
`DecisionRef` is built by `runner._ref_from_wp2_row` from WP2's §18 row, which has a single
`request_hash` and no place for the second request — so the field defaulted to an empty tuple on
*every* decision and `cost_summary` charged one request per continued draw. Caught on the
two-phase pilot, where all 128 decisions carried an empty tuple despite 67 recorded
continuations. Fixed by consulting the record in the adapter (`runner._physical_ids`), pinned by
`tests/ard/test_two_phase.py::test_the_wp2_adapter_carries_both_physical_requests`, and verified
on the re-scored pilot: 61 decisions at one physical request, 67 at two, matching the 67
continuation events exactly.

**Recurrence (2026-09-10).** The same omission was then found at the two *other* sites that
build a `DecisionRef` field-by-field from a record: `receiver.py` (E3) and `selectors.py`
(the selector audit). The receiver pilot cell recorded 4 continuation events and charged 4
single requests. The field defaults to an empty tuple, so forgetting it at a new site is
silent -- it costs no test and no error, only an under-reported run. All three sites now
route through `types.physical_request_ids_of`, and
`test_every_decision_ref_construction_site_copies_the_physical_requests` walks the AST of
all three modules and fails on any `DecisionRef(...)` built without the field, so a fourth
site cannot reintroduce it. Re-run receiver pilot: 4 decisions, 4 cap hits, all at two
physical requests.

## Closed

### D-00 — the forced reasoning boundary produced no report *(closed 2026-09-10)*
The single-call split budget lost the report on 27% of cap-hit draws: vLLM forces `</think>`
but the model continues its derivation rather than switching to the report format, spending the
whole 256-token allowance on reasoning. Overall validity 0.859 against a 0.95 gate, concentrated
on hard items (MMLU-Pro 43/64 cap hits versus MATH-500 19/64). A prompt-wording repair was tried
and failed (+2 points). Closed by the two-phase continuation with a prefilled `<final>` opening;
recorded as amendment I6. Found by the pilot, before the write-once freeze.

## Caveats carried into the report rather than fixed

- For cap-hit draws the report *format* is induced by the prefilled opening tag, so
  report-format compliance is not a model property on those draws (amendment I6).
- Summed FLOPs count the re-sent phase-2 prefix even when it was a prefix-cache hit: the
  analytic oracle is a pure function of token counts, so the figure is an upper bound.
- `cached_prefill_tokens` on a merged record is phase 1's, not the sum; phase 2's sits in the
  `continuation` block. Summing would drive `prefill_tokens = prompt_tokens − cached` to zero.

### D-03 — a round's five calls were issued one at a time
**Severity: high for the schedule; none for results.**

`KIND_TABLE` declares `max_inflight=N_AGENTS` for every episode kind, but `episode.execute_updates`
looped over the slots and blocked on each `generate` in turn, and `run_roots` did the same over the
16 bank draws. `max_inflight` was declared and never read. Measured on the pilot: peak concurrency
**1 per cell**, each agent starting within 20 ms of the previous one finishing, so a cell's wall
clock was the *sum* over its decisions rather than the max over its agents. Thirteen 8B replicas
served an aggregate 234 tok/s across ~3.4 average concurrent requests — a fleet essentially idle.

Fixed by `episode.dispatch_batch`: every prompt in a round is rendered from the immutable pre-round
snapshot first, then the batch is dispatched through a thread pool bounded by `max_inflight`, and
results are collected in slot order. This is result-identical rather than merely faster, and for a
reason the loop already documented: the batch is exchangeable by construction (no member can observe
another's round-*t* output), request identity is content-addressed, and the engine seed is
HMAC-derived from the seed key — so a record depends on neither when it was issued nor which replica
served it. A refused slot (`ContextTooLong`) is returned rather than raised, so one over-long prompt
cannot cancel the other four decisions of its round.

The inference layer was already built for this: `EndpointPool`, `VllmChatClient` and the continuation
path all carry locks, `EndpointAffinity` uses `threading.local()`, and the cell runner has always run
items in parallel threads. Pinned by
`test_wp2_episode.py::test_a_round_goes_out_as_one_concurrent_batch_without_changing_its_results`,
which asserts peak in-flight equals `N_AGENTS`, that concurrent and serial dispatch agree element for
element, and that a refusal in one slot leaves the rest of the round intact.

### D-04 — two manifests could be given the same cell id
**Severity: high; caught before it corrupted anything.**

A cell id *is* a directory under `<run_root>/cells/`, but shard numbers restart at `s000` in
every `ArdBuilder`, and the id carried no split. So building a module's `--split train` and
`--split test` manifests as separate CLI invocations produced
`ROOTS.natural.none.8B.T1024.s000` twice, with different item lists and different cohorts.
The second run would have written its items beside the first's and overwritten `meta.json`,
leaving one directory that claims one manifest's cohort while holding two — and nothing
downstream could tell, because every item file in it is individually well formed. Coverage
would then be counted against the wrong split, which is the one thing the hour-18 and hour-24
gates exist to measure.

Found by inspecting the live run after dispatching the test-split roots: the 8B ROOTS
directories were 12 train + 6 validation (complete, `s000`–`s017`) and 12 empty test cells
(`s018`–`s029`) — the test cells that would have collided had not yet written. **No directory
was ever conflated** (verified over every cell in the run).

Two fixes, because either alone leaves the hazard:
1. The split is now part of the cell id (`ROOTS.natural.none.8B.T1024.test.s000`), and the
   shard counter is keyed per split so numbering is stable however a manifest is cut.
2. `ard/runner.py` refuses to run a cell whose directory holds item files the cell does not
   declare, whatever the cause. Refusing is cheap and reversible — requests are
   content-addressed, so a cell re-dispatched under a correct id re-derives its items from the
   store with no new generation. Silently merging two cohorts is neither.

Pinned by `test_wp4_cells.py::test_two_manifests_of_the_same_module_cannot_share_a_cell_directory`
and `test_wp4_runner.py::test_a_cell_refuses_a_directory_that_holds_another_manifests_items`.

### D-05 — the frozen manifest wrote the config hash where the freeze gate does not read it
**Severity: blocking; the whole run stopped for 17 minutes.**

`ard/freeze.py` recorded the study-config hash nested, as `config: {sha256: ...}`. The freeze
gate every generate cell calls before its first request — `study.config.StudyConfig.frozen` —
reads **top-level** `config_sha256`. It therefore read `None`, compared it against the real
hash, and raised "stale or edited manifest" for every cell. `FROZEN.yaml` was written at
03:06 UTC and **95 cells suspended** against it; not one cell ever passed its gate. The 33
cells that completed after 03:06 had started before the file existed, when the gate was a
no-op, so nothing ran under a manifest it had actually verified.

The failure mode is the one this run keeps meeting: two modules agreeing on a fact and
disagreeing on where it lives. The writer was ARD's, the reader is the frozen study's, and
nothing tied them together.

Fixed by writing `config_sha256` at the top level (the nested block stays, it carries the
path and the raw config), and by refusing to *write* a FROZEN.yaml without it — a freeze that
cannot be verified should fail once, loudly, at freeze time, not once per cell for the rest
of the run.

`FROZEN.yaml` is write-once, so the unusable file was preserved as
`FROZEN.yaml.d05-superseded` and re-frozen. The two differ in exactly three top-level keys:
`config_sha256` (`None` → the real hash), `frozen_at`, and `code_version` (the commit carrying
the fix). `tier`, `contract` and `items` are identical — no scientific content changed, and
the tier decision of amendment S6 was taken from the launch gate, not from this file. All 95
suspensions were verified to share this single cause before they were cleared.

### D-06 — the fork stage had a dependency mechanism at both ends and no step in the middle
**Severity: blocking for E4 (forks, and therefore the controller); found before it was needed.**

`runner.resolve_hard_dependencies` refuses a FORK or SUFFIX cell until `state:<sha>` resolves to
`<run_root>/states/<sha>/REGISTER.json`, and `runner.freeze_register` is exactly the writer that
satisfies that reader. Nothing ever called it outside tests. `states.py` has a `StateRegister`
that writes a *different* layout (`states/<file>`, no sha directory), so the two halves could
never have met: `build_e4_forks(state_sha=...)` had no way to obtain a sha, and every fork cell
would have exited `EXIT_DEPS_UNMET` forever.

Same shape as D-04 and D-05 — two modules agreeing on a fact and disagreeing on where it lives.

`ard/seal.py` is the missing step: it reads the per-round states each completed episode already
records, seals exactly the parent rounds a fork branches from, and returns the sha a manifest
pins. Two rules travel with it, because a fork frame is controller training data:

* **Only completed cells seal.** A cell without `meta.json` is still generating, and sealing its
  states would freeze a cohort about to grow — the sha would then name a set no later run could
  reproduce. Skipped cells are counted in the census rather than passed over silently.
* **An empty seal is refused.** A dependency that resolves but contains no parents is worse than
  an unmet one: the cell would run, generate nothing, and report success.

The register is content-addressed over its own contents, so re-sealing the same states returns
the same sha, adding a cell produces a *new* register rather than mutating the frozen one, and a
tampered register is reported as `tampered` rather than met. A `state_id` that appears twice with
different content raises, since a content address disagreeing with itself is a harness defect.

Verified on the live run before use: 16 completed PEER cells → 128 episodes → 256 sealed states
across rounds 1 and 2, exactly 2 per episode.

### D-07 — the controller had no path from generated data to a fitted model
**Severity: blocking for E4, the study's primary estimand; found before it was needed.**

`features.features_from_state` turns a WP2 `RoundState` into the frozen v1 vector.
`controller.fit_controller` consumes `PairedFrame`s. Neither reads the run root, `compact.py`
emits `"features": []`, and nothing assembled a frame from the episodes and forks on disk — so
the chain from generated data to a frozen controller, and therefore to GATE, did not exist.
Third instance of the run's recurring shape: both ends built, no middle.

`ard/frames.py` is the middle. It reconstructs the round states each episode already records
(only four state attributes are read by the feature builder, so the reconstruction is faithful
rather than guessed), builds the frozen vector on each, and pairs the two branches spec §12.1
asks for: the t=0 pair from PRIVATE's and PEER's round-1 children off the shared root state,
and the t=1,2 pairs from the FORK cells against the main PEER trajectory.

Two rules it enforces, both about not manufacturing training data:

* **A pair is only a pair when both children share a `parent_state_id`**, which is a content
  address — so a mismatched join cannot quietly become a training row.
* **An ungraded decision is an unobserved outcome, never a zero.** The evaluator's `ambiguous`
  and `timeout` statuses do not become labels either. Scoring them 0 would teach the controller
  that every ungraded branch failed, and ungraded correlates with *hard*, not with *wrong*.

Test rows are refused rather than filtered, so a leak raises instead of shrinking the frame.

Verified on the live run before use: 16 PEER and 18 PRIVATE completed cells reconstruct 1,920
paired rows carrying the full 31-column frozen schema, and **all 640 round-0 states join both
branches** — the "t=0 pairs for free" claim of spec §12.1, confirmed on real data rather than
assumed. The 1,280 round-1 and round-2 states are peer-only, which is exactly the gap the FORK
cells fill.


### D-08 — the frame builder read the wrong label key (self-inflicted, caught on first real use)
**Severity: would have trained the controller on nothing; fixed before any fit.**

`frames.load_labels` read `row["status"]`. The evaluator writes `evaluator_status` and
`is_correct`. Against the live grades that returned **zero labels from 11,040 graded rows** —
so every paired row would have carried `y_private = y_peer = None`, the fitter would have seen
no observed outcome, and the controller would have been fit on an empty frame or refused
outright.

Written into D-07's own module, one commit after documenting this exact defect class. Worth
recording as such: knowing the shape of a bug is not the same as not writing it.

The fix also avoids the trap next to it. `is_correct` alone is **not** the label: the grader
sets it `False` for `ambiguous` and `timeout` as well as `incorrect`, so reading it directly
would score every ungradeable answer as wrong. On this run that is 1,149 of 11,040 decisions
(10.4%), concentrated on answers the symbolic comparison could not parse rather than on answers
the model got wrong — precisely the bias D-07's docstring warns about. `load_labels` now gates
on `evaluator_status` against the grader's own `STATUS_CORRECT`/`STATUS_INCORRECT` constants,
imported rather than spelled out, and leaves everything else unobserved.

Verified on the live grades: 9,891 labels (7,726 correct, 2,165 incorrect), 1,149 ambiguous or
timed-out correctly excluded.

### D-09 — the frame builder took a manifest as a branch (self-inflicted, caught by reading the numbers)
**Severity: would have fit the controller on counterfactuals against themselves; fixed before any fit.**

`frames.build_paired_rows` selected the peer branch by *manifest* rather than by episode
protocol. The P1 manifest holds PEER and PRIVATE cells together — a manifest is a dispatch
unit, not a branch — so passing it as `peer_cells` made every PRIVATE state a decision point
and paired it with its **own** PRIVATE child as the "peer branch": a counterfactual against
itself. The private branch was contaminated symmetrically.

What makes this worth recording is how it presented. The dry run reported 3,600 paired rows,
2,439 with both branches observed, and paired rows at all three decision rounds — every
coverage check green, before the FORK cells had produced a single completed cell. That is the
tell: rounds 1 and 2 *cannot* have paired rows until forks land, because a PRIVATE trajectory's
round-1 and round-2 children hang off PRIVATE's own parent states, not off PEER's. The numbers
looked healthy and were meaningless.

Branch selection is now by `episode.protocol`, so the same manifest can be passed for both
arguments — which is how P1 is actually shaped — and a PRIVATE state is never a decision point.
`_children_by_parent` takes an explicit protocol filter; fork cells are unfiltered, being all
forks by construction.

Pinned by `test_frames.py::test_a_mixed_manifest_does_not_pair_private_states_with_their_own_children`,
which feeds one mixed manifest and asserts exactly `N_AGENTS` decision points on the shared root
state, with the two branches resolving to genuinely different decisions.

### D-10 — a FORK cell ran an ordinary PRIVATE episode instead of a counterfactual branch
**Severity: E4's training data was wrong; caught before the controller was fit.**

`ard/forks.py` implements the spec §19 contract in full — `fork_private(previous_peer_state,
focal_agent)` issues exactly one private revision of one agent from a sealed parent, and
`free_root_pair` gives the t=0 pair. **The runner never called any of it.** `episode_worker`
dispatched FORK cells to the same `wp2_episode_executor` as EPISODE cells, which has a branch
for ROOTS and otherwise runs `run_episode` — so each FORK cell generated a fresh three-round
PRIVATE trajectory from its own roots: 20 decisions per item instead of 5, parented on its own
states instead of the sealed PEER ones.

Measured on the completed cells before the fix: 2,160 distinct fork parent states, **zero**
overlap with the 288 sealed states. The forks pair with nothing, which is exactly why
`ard/fit.py` kept refusing with "no paired rows at round(s) [1, 2]" even after all 36 fork
cells had completed and been graded. The refusal check earned its place here — without it the
controller would have been fit on the t=0 pairs alone and reported as if it covered all three
decision rounds.

Fourth instance of this run's dominant shape: both ends built, no middle.

The executor now branches on `ArdCellKind.FORK` and calls `fork_private` per focal agent. Two
supporting changes:

* `states.rebuild_round_states` puts a written episode back into a WP2 `RoundState`, with the
  canonical classifier **injected** — a FORK cell runs on the generation path, where importing
  the grader would breach the no-gold firewall. `frames.py` now delegates to it rather than
  keeping a second copy.
* The parent is read back from the PEER cells named in `options['parent_cells']` and **checked
  against the sealed register** before any request is issued. Without that check the seal is
  decorative: a PEER cell regenerated after sealing would hand the fork a different parent than
  the manifest pinned, and nothing would notice.

Pinned by `test_fork_executor.py`: a fork cell produces exactly `N_AGENTS` decisions, all
parented on the sealed PEER state; a missing parent suspends the cell rather than yielding an
empty one; and a parent whose decision ids have moved since sealing is refused by name.
