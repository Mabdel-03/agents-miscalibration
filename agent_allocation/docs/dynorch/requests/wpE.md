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
