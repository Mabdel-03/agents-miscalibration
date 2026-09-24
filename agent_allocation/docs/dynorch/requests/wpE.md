# Interface requests from WP-E (wire probe) — evidence in docs/dynorch/harness_wire_observations.md

Each item cites the capture that proves it. WP-E edits none of the files below.

## To the lead (contract / config / plan text)

1. `00_contract.md` §1 "solver = system text sha equals `subagent_system_sha`": FALSE for both harnesses. OpenClaude
   prepends `"You are OpenClaude, built on the Claude Agent SDK.\n\n"` and appends `"\n\nNotes:…<env>…You are powered by
   the model …"` (captures/openclaude/b, seq 2); Codex children carry the identical `instructions` as the root
   (captures/codex/b_ns, seq 3). Proposed rule: solver iff (`x-openai-subagent` header present) or (system text
   contains `subagent_dynorch.txt.rstrip("\n")`); root system text == `OPENCLAUDE_PREFIX + system_dynorch.txt.rstrip("\n")`.
2. Plan §9 Gate 0 "`messages[0]` byte-identical to SYSTEM.md": amend to the prefixed form above (captures/openclaude/a).
3. Plan §6 Codex `config.toml`: add `goals = false` under `[features]`, plus `[tools.experimental_request_user_input]
   enabled = false` and `[tools.update_plan] enabled = true` — without them the offered tools include
   `request_user_input`, `get_goal`, `create_goal`, `update_goal` and exclude `update_plan` (captures/codex/a vs
   captures/codex/x_tools_struct).
4. Plan §6 OpenClaude env: add `CLAUDE_CODE_PROVIDER_PROFILE_ENV_APPLIED=1` (silences the "ignoring saved provider
   profile" warning; captures/openclaude/x_profile_env). `agentModels`/`agentRouting` in `.openclaude.json` are optional.
5. `configs/dynorch_v1.json` / `locks/harness_tools.json` allowlists: H1 `{"Agent"}`; H3 `{"update_plan","spawn_agent",
   "send_input","wait_agent","resume_agent","close_agent"}` (namespace `multi_agent_v1` on the last five).

## To WP-A (core)

6. `ChatRequest` needs the sub-agent hint from the wire layer (e.g. `subagent_header: str | None`) so `classify` can
   mark Codex children as `solver`, not `other` (`other_share == 0` is a Gate 0 criterion).
7. Tool-surface check must treat a missing `tools` key as an empty list (OpenClaude solver requests have no `tools`
   key; captures/openclaude/b seq 2) and must not flag `temperature`/`parallel_tool_calls`/`reasoning`/`include`/
   `prompt_cache_key`/`client_metadata` (record them in `ignored_body_keys`).
8. Reasoning cache: OpenClaude never re-sends `reasoning_content` (captures/openclaude/e) — key 1 (echoed tool-call ids)
   is available and exact; Codex re-sends `reasoning` items with `content[].text` and `encrypted_content: null`
   (captures/codex/h) — treat as `explicit`.
9. Conversation keys must not hash the raw first user turn: OpenClaude prepends a dated `<system-reminder>`; Codex
   sends `<environment_context>` (cwd, date, timezone) as a separate user item before the task.

## To WP-B (wire + adapters)

10. `responses_sse`: emit `namespace` on `function_call` items for namespaced tools — mandatory (captures/codex/b_nons:
    "unsupported call: spawn_agent" without it).
11. `responses_sse`: never emit `response.output_item.done` before `response.completed` is guaranteed; Codex keeps a
    partial item in history across its retries (captures/codex/d).
12. `openclaude.py::parse_result`: success iff `is_error == false`; `subtype` stays `"success"` on a 402
    (captures/openclaude/c). `codex.py::parse_result`: `-o` file or last `agent_message`; `turn.failed` ⇒ failure; the
    `item.completed {type:"error", "Model metadata … not found"}` event is normal.
13. Codex per-run `CODEX_HOME` gets ≈60 skill files + 4 sqlite files and the first start takes ≈18 s (captures/codex/a).

## Gate 0 smoke findings (WP-E, 2026-09-23, both real CLIs through the REAL proxy over the fake upstream, $0)

Evidence: `tests/integration/test_dynorch_openclaude_smoke.py`, `tests/integration/test_dynorch_codex_smoke.py`
(cases recorded in `<run_root>/gates/gate0_smoke.json`), `docs/dynorch/GATE0_REPORT.md`.

## To the lead (contract amendments proposed as §10.19 – §10.21)

14. **§10.19 — a refusal decided before the headers carries its real HTTP status (fixed in `httpd.py`).** The
    streaming path sent `200` + `text/event-stream` headers before calling the core, so a budget 402 (decided in
    microseconds, no upstream call) reached OpenClaude as an error frame inside a 200 stream, which it retried once
    per conversation (`CLAUDE_CODE_MAX_RETRIES=1`: four 402s in 70 ms for a solver + root pair) — the probe's
    "no retry, no further request" (captures/openclaude/c) was for an HTTP 402 status. `ProxyServer` now waits
    `refusal_grace_seconds` (default `min(2 s, keepalive_seconds)`) for an outright refusal and answers it with its
    status + JSON body; errors after the headers stay in-stream. B9's "headers at once" becomes "headers within the
    refusal grace" (both harnesses' header timeouts are minutes). Pinned by
    `test_httpd.py::test_streaming_refusal_before_headers_carries_the_real_http_status` and the OpenClaude smoke
    case `forced_close_then_402` (ends 0.1 s after the 402, exactly one further request — the root's, after the
    solver's 402 came back as its tool result — `is_error: true`, exit 1, no rerun).
15. **§10.20 — Codex 0.156.1 retries an HTTP 402.** Codex reports `unexpected status 402 Payment Required` and
    re-sends the identical request `stream_max_retries` (2) times per conversation (`Reconnecting... 1/2`, `2/2`,
    ~200 ms / ~400 ms apart), then `turn.failed`, exit 1, no `-o` file. Each retry is refused in-process (no
    upstream call, no cost; `refused_calls` counts them). The frozen table's "402 non-retryable by both harnesses"
    holds for the budget (nothing is ever served after the grace) but not for Codex's transport-level retry loop;
    the smoke criterion for Codex is therefore: only 402s after the first 402, at most `2 × (stream_max_retries+1)`
    of them (root + child), `turn.failed` within 10 s, zero upstream calls. No config key disables the loop
    (`stream_max_retries = 0` would also remove the wanted retries on real transient failures). Recorded in
    `codex/forced_close_then_402` (0.7 s from the first 402 to exit).
16. **§10.21 — Codex leaves `state/codex/tmp/arg0/codex-arg0*/{apply_patch,applypatch,codex-execve-wrapper,
    codex-linux-sandbox}` symlinks to its own binary behind on a failed exit** (turn.failed, 402, timeout; a clean
    exit removes them; `codex --version` then warns "failed to clean up stale arg0 temp dirs"). Symlinks (169 bytes,
    4 inodes per failed run under scratch) — harmless, but any key scan of the scratch tree must skip symlinks and
    binaries: the binary carries a `sk-or-OPENAI...` substring the key regex matches (`doctor.key_shaped_strings`
    already skips symlinks and files > 4 MB; `cli.key_hits_in_run_tree` scans the run root only).

## To WP-D (gates / doctor)

17. `dynorch-doctor` blocks on `state_paths_outside_home` when `npm_config_cache` is unset in the calling shell
    (`env.npm_config_cache: null` is treated as bad); the runbook's shell header exports it, so the check passes
    there. Consider distinguishing "unset" (a warning: Codex is already installed) from "under /home" (blocking).
18. The `$HOME` check of Gate 1 (`cli.new_files_under_home`) will see the same foreign writers the smoke saw on
    this node: cursor-server's Codex app-server (`CODEX_HOME=$HOME/.codex`, writes `logs_2.sqlite*` and
    `models_cache.json` continuously), the Claude Code sessions (`~/.claude.json`), a root-owned `orcd/.quota`, and
    `.humming/tmp/lock/launcher.lock`. The smoke attributes writes to live foreign harness instances from
    `/proc/<pid>/environ` and classifies pre-existing-file churn separately from new files
    (`dynorch_smoke_support.new_under_home`); Gate 1 could reuse that rule instead of a static exclude list.
