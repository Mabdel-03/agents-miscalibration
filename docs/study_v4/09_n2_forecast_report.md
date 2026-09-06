# N2 — FINAL_HANDOFF_REPORT compiler and shadow-forecast cell (spec §8.7; brief 09 R1 step 1)

Package `src/agents_scaling/study/forecast/` (`report.py`, `manifest.py`, `shadow.py`, `run.py`); tests `tests/study_neural/test_n2_*.py`; Slurm `slurm/study_neural_forecast.sbatch.tmpl`. Nothing here touches the frozen generation pipeline; inputs are sealed artifacts and the public export only (`data/protected` is never read).

## Inputs and refusal rules
- Item file `<run_root>/cells/<cell_id>/items/<source_id>.json` (`EpisodeResult`), the seal register `<run_root>/seals/<sha>/SELECTIONS.json`, `data/public/tasks.jsonl`.
- The report's **selected candidate** is the sealed `VOTE` selection over the method's primary pool: `archive` (IND_VOTE), `latest_slots` (DEC), `native` (CEN_FLAT). No sealed selection for `(cell_id, source_id)` → `ProtocolError` (no report before the seal, P0-3). A sealed winner that is not a valid complete candidate is a harness defect (`ProtocolError`).
- `selected_candidate_id == null` (no valid complete candidate) → the exact §3.5 sentinel bytes + `{"scope":"PERSONAL_FINAL","value":null,"missing":true}`.

## Frozen report layout (bytes the reader sees; inserted verbatim between `=== REPORT (observable state) ===` / `=== END REPORT ===` by `prompts.render.render_forecast`)
```
=== TASK ===
<exact task text>                       ← task span; task_only_anchor = its end (byte offset)
=== END TASK ===
--- item ---                            <compact JSON: ids, domain, format, method, N, B, checkpoint, status>
--- selected_candidate ---              <candidate JSON | sentinel>
--- selected_personal_confidence ---    {"scope":"PERSONAL_FINAL","value":q,"missing":false}
--- decision_rule ---                   ← evidence span starts (8,192-token cap applies from here)
--- vote_metadata ---                   selector, pool_kind, pool_size, planned/valid counts, winning_count, tied_classes, all_singleton, grouping_mode(+counts), method counters (optional_draws | rounds_completed/r_max | delegation_cycles/hub_calls/realized_participation/hub_action_errors)
--- budget_metadata ---                 B, B_flops, spent, slack, calls_admitted, calls_by_role, context failures, invocations, complete candidates, stop_reason
--- nonselected_candidates: k of n ---  ("none" when the pool has no other member)
[packet i] <exact §4.3 packet bytes>    ← evidence span ends at the end of the text
```
- Non-selected artifacts = the sealed pool minus the winner, in `blind_order_key(study_seed, "REPORT_PACKETS", source_id, candidate_id)` order (blind to validity/confidence/correctness); at most 4; invalid members render as the typed unavailable packet; each compiled by `study.packets.compile_packet` at the 2,048/1,024 caps.
- Cap: the complete rendered evidence span (one contiguous byte range, tokenized as a whole with the reader tokenizer) ≤ 8,192. Zero-packet render must fit (else `ProtocolError`); packets admitted in order while the whole span fits; the first packet that does not fit whole is re-compiled under the largest packet cap that fits (binary search over the cap; the packet compiler's deterministic excerpt clipper) or dropped if even its skeleton does not fit; nothing later is admitted. The final count is asserted after admission (BPE non-monotonicity guard).
- `report_id = sha256(JCS(["final_handoff_report", seal, selection_id, cell_id, source_id, sha256(text)]))`.

## Trusted manifest and request
`{"scope":["PERSONAL_FINAL","TEAM_SELECTED"],"mask":{"q_personal":"required","q_team_now":"required","q_child_contract":null,"q_recover":null,"q_preserve":null},"observer_role":"COMMON_REPORT_READER","information_set":SELF_ONLY|PEER_EXPOSED|HUB_STATE,"selected_pool_id":<sealed pool_id>,"checkpoint_id":"FINAL_HANDOFF_REPORT","operation_id":"NONE","remaining_allowance":<ledger slack>,"allowance_unit":"logical_flops"}`. One user message (`forecast.txt`), `FORECAST_DECODING` (thinking off, temperature 0, 256 tokens, `guided_json=False`), seed key `(source_id, split, model_cell, episode_rep, 0, "forecast", 0, "forecast")`, role `forecast`.

## Anchors for the capture stage (`forecast/reports/<item>.<method>.json`)
Byte offsets into the message content: `task_only_anchor` (end of the task span) and `state_anchor` (end of `=== END REPORT ===`, i.e. the end of the complete compiled report; the last prefill token is recorded separately as `last_prompt_token`). Token indices come from the tokenizer offset mapping over the rendered chat text, verified to reproduce the exact `prompt_token_ids`; the anchor token is the last token that starts before the anchor (`straddles=true` when it also runs past it, e.g. `?\n`). Whitespace test tokenizers use a verified `\S+` fallback.

## Outputs
- `forecast/reports/<item>.<method>.json` — render: messages, manifest, `prompt_token_ids`, `byte_anchors` + `anchor_tokens` and their flat mirrors `task_only_anchor_byte` / `state_anchor_byte` / `task_only_anchor_token` / `state_anchor_token` (the N1 capture interface; N1 reads both shapes and asserts byte/token agreement), `report_sha256` (= `report.text_sha256`), `request_id`, checkpoint pins (`checkpoint.model_revision` is checked by the capture), the `Report` dict.
- `forecast/<item>.<method>.json` — outcome: raw content, parse status (`ok | TRUNCATED | EMPTY | NOT_JSON | DUPLICATE_KEY | NONFINITE | TRAILING_TEXT | SCHEMA | REQUIRED_NULL`), parsed values, cost, §8.13 `ConfidenceRow` projection. Invalid forecasts are recorded as such; the development marginal prior + missingness flag are applied in analysis, never here.
- `forecast/errors/<item>.<method>.json`, `forecast/events.s<shard>.jsonl`.

## CLI
`--panel-per-domain N` derives the items from the public export with the shared rule of `neural/panel.py` (`PublicTask.rank < N` per superdomain on `main`, domain-major rank order) — the same panel N1's native capture and N3's confirmation stage use — so no ops-written items file can drift from it.
`python -m agents_scaling.study.forecast.run --run-id study_v4 --seal <sha> --methods IND_VOTE,DEC,CEN_FLAT (--items-file <panel items> | --panel-per-domain 150) --shard K --num-shards N [--report-only] [--server-run-id …] [--checkpoint 32B --N 5 --B 4 --module A --episode-rep 0] [--parallel 4] [--wait-s 300]`. Exit 0 done / 2 no endpoint / 3 infra failures / 4 protocol error(s) recorded. Cell id of the forecast requests: `forecast.<method>.<ckpt>.x<seal[:8]>`.

## Decisions recorded (not in the brief)
- The blind order key includes the candidate id (`…, source_id, candidate_id`): the brief's `blind_order_key(study_seed, "REPORT_PACKETS", source_id)` alone would be constant across the pool.
- CEN_FLAT reports carry no packets unless the native final is invalid (then the one unavailable packet); worker subtask results are not selectable candidates (§3.5) and are summarized through the hub counters only.
- The evidence count is the tokenization of the contiguous evidence substring, not the sum of fragment counts; the full prompt is additionally checked against the 32,768-token envelope.
