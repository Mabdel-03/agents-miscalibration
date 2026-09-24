# WP-A (core) — interface requests and notes for WP-B / WP-C / WP-D / lead

Everything below is additive to `docs/dynorch/00_contract.md`; nothing in the frozen signatures changed.

## R-A1 `ChatRequest.subagent: bool = False` (WP-B `responses_format.py`, `openai_format.py`)

Codex V1 sub-agents are full-history forks: their system text (`instructions`) and first user text are the
root's, so the §1 conversation rule (system sha / first user text) cannot tell a fork from the root. Plan §6 says
sub-agent activity is identified "by the `x-openai-subagent` request header at the proxy". `records.ChatRequest`
gained a defaulted field `subagent: bool = False`; `ProxyCore._classify` treats `subagent=True` as **solver**
regardless of the system text. WP-B: set it from the `x-openai-subagent` header on the Responses route (and leave
it `False` on the chat route). Without it a Codex fork classifies as `root` (same system text), which would let a
sub-agent take the root-only grace.

## R-A2 Shield `core.chat` from client disconnects (WP-B `httpd.py`)

B9: "if a client disconnects after `mark_sent`, the upstream call is finished and settled, its reply cached by
rendered-prompt sha256, and an identical re-request is served from the cache". The core caches only replies it
finished. The server must therefore run `await asyncio.shield(core.chat(run_key, request))` (or keep the task alive
independently of the connection) and never cancel the core task when the socket closes. If a cancellation does
reach the core mid-upstream, the grant is charged whole and the run lock is released (tested), but the reply is
lost and nothing is cached.

## R-A3 Classification is by system-text prefix (WP-B adapters, WP-E smoke)

`solver` = `request.subagent` or the system text **starts with** `subagent_dynorch.txt` (OpenClaude appends env
details to a custom agent's prompt); `root` = starts with the run's root system text, or the first non-solver
conversation of the run (Codex appends its multi-agent instructions to `model_instructions_file`); everything
else is `other`. Adapters must pass the exact files' contents as `root_system_text` / `subagent_system_text` to
`open_run`. Gate 0's `other_share == 0` check for OpenClaude relies on the solver prompt being a prefix of what
the CLI sends — verify in the smoke.

## R-A4 `forced_close_kind = "ceiling_close"` (WP-C tables, lead)

The plan gives no rule for an E0 root call bound by its **ceiling** (`max(A, C/4)`) rather than the cap: closing
it with the grace would spend the run's single grace on the first call and make the capped variant pointless;
returning an empty reply would end the harness turn. WP-A closes such a call with the same ANSWER continuation,
`max_tokens = min(G, C − spent)`, **charged inside C** (no grace), recorded as `CallRecord.forced_close_kind =
"ceiling_close"` (not in the contract's enum; `RunRecord.forced_close_kind` keeps the B5 set
`{none, mid_call, visible_continuation, terminal}` and is unaffected). Core arms never produce it. Lead: confirm or
amend before E0.

## R-A5 Reply content is edge-trimmed (WP-B formats, WP-C)

`ChatReply.content` (and the delivered `final_message`) has leading/trailing whitespace stripped — the pinned
template `|trim`s every content on re-render, so the model never sees a difference; it keeps the `\n\n` the
inline-style servers emit after `</think>` out of the harness's view. Inner whitespace is untouched.

## R-A6 Records the runner must finalise (WP-B `runner.py`, `grade.py`)

`close_run(run_key, status=...)` fills `final_message`/`final_source`/`finished_at`/`wall_seconds` and persists.
The runner then sets `cli_result`, `cli_result_mismatch`, `harness_exit_code`, `harness_reported`, `parse_status`,
`final_answer`, `grade_status`, `correct`, `grade_detail`, `rerun_reason` and re-saves through `RunStore.save`.
`RunRecord.error` starts with `infrastructure_stop:` after an upstream 402/401/403 (B6 `stop_notify`);
`tool_surface_violation: <names>` after a 400; `upstream_failed: <status>` after an ambiguous outcome (rerun rule:
"last reply was a transport error"). Refused calls (402s) are counted in `refused_calls`, not in `calls_detail`.

## R-A7 Attempt ids and resume (WP-B `runner.py`, WP-D `cli.py`)

Ledger attempt ids are `journal.attempt_id(run_key, generation, call_no, phase, n)` =
`f"{run_key}:g{gen}:{call_no}:{phase}:{n}"` (`n` = transport retry 1..3). On `--resume` construct the
`CollectionLedger` over the same journal, call `await journal.resolve_inherited(ledger, store)` **before** moving
any raw files (`CallJournal.supersede`) or archiving records (`RunStore.archive_previous`), then open new attempts
with `generation + 1`.

## R-A8 `api/fake.py` additions available to WP-B/WP-E

`Fault.NO_USAGE` (200 without a usage object), `FakeReply.think_tokens` (per-reply natural think length),
`fake_tool_call_text(name, arguments, prefix=...)` (the XML block the template teaches), and a cap-consistent
natural close (`finish_reason="length"` when the visible answer is cut after `</think>`). Long
`FakeReply.delay_seconds` already worked (a delay beyond the client's read timeout raises `ReadTimeout` and is
billed).
