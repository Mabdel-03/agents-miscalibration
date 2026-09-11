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
