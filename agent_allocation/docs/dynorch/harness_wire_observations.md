# dynorch_v1 — harness wire observations (WP-E early probe, $0, real CLIs vs a throwaway stub)

Contract: `docs/dynorch/00_contract.md` §6 (wire layer), §1 (conversation classes), plan §6 (harness invocations) and
§9 Gate 0 (real-CLI smoke list). Everything below marked VERIFIED was captured on 2026-09-23 by running the real CLIs
against `tools/dynorch_wire_probe.py` (a stdlib HTTP stub on 127.0.0.1 that replays scripted replies and journals every
request). Raw captures: `tests/fixtures/dynorch/captures/<harness>/<case>/{requests.jsonl, stdout, stderr, argv.json,
env.json, summary.json}` (bodies trimmed above 200 KB; bearer headers redacted; no OpenRouter key was ever read).
Reproduce with `cd agent_allocation && source slurm/env.sh && uv run --frozen --no-sync python tools/dynorch_wire_probe.py
run openclaude b` (cases a–e) / `run codex b_ns` (cases a, b_ns, b_nons, c–h). Host: OpenClaude 0.13.0
(`/orcd/home/002/mabdel03/conda_envs/consortium/bin/openclaude`, node v25.7.0), Codex CLI 0.156.1
(`/orcd/data/tpoggio/001/mabdel03/dynorch_scratch/npm-prefix/bin/codex`), kernel 4.18, no root.

Nothing was written under `/home` by any run (`summary.json: home_writes_since_marker == []` in every capture; the check
is `find /home/mabdel03 -maxdepth 4 -newer <marker>` with this session's own `~/.claude/*` dirs pruned).

---

## 1. OpenClaude 0.13.0 (H1)

Invocation used = plan §6 verbatim (`argv.json`): `openclaude -p --output-format json --no-session-persistence
--provider openai --model qwen/qwen3.8-27b --system-prompt <system_dynorch.txt, trailing newline stripped> --tools Agent
--allowedTools Agent --disallowedTools Bash Edit MultiEdit Write Read NotebookEdit Glob Grep WebSearch WebFetch TodoWrite
TaskCreate TaskUpdate KillShell BashOutput SendMessage TaskStop ToolSearchTool Skill --permission-mode bypassPermissions
--agents '{"solver": {...,"prompt": <subagent_dynorch.txt>, "tools": []}}' --setting-sources "" --strict-mcp-config
--disable-slash-commands`, task on stdin, env = plan §6 (`env.json`). Every flag was accepted: `--setting-sources ""`
is fine (no error, no settings loaded), `--disable-slash-commands` exists ("Disable all skills").

### 1.1 Results per case

| case | script | requests (role) | exit | stdout `type/subtype/is_error` | wall |
|---|---|---|---|---|---|
| a | text `<final>7</final>` | 1 (root) | 0 | result / success / false | 1.3 s |
| b | Agent(solver) → child text → text | 3 (root, child, root) | 0 | result / success / false, `num_turns: 2` | 1.2 s |
| c | Agent(solver) → child → **402** | 3 (root, child, root-402), **0 requests after the 402** | **1** | result / **success** / **is_error: true**, `result: "API Error: 402 {\"error\":{\"code\":\"budget_exhausted\",…}}"`, `stop_reason: "stop_sequence"` | 1.2 s |
| d | text after a **330 s** streaming delay with 21 keepalive comments | 1 | 0 | result / success / false | 332 s |
| e | as b with `reasoning_content` on every reply | 3 | 0 | result / success / false | 1.7 s |
| x_minimal | as a, `.openclaude.json` without `agentModels`/`agentRouting` | 1 | 0 | identical to a | |
| x_profile_env | as b plus env `CLAUDE_CODE_PROVIDER_PROFILE_ENV_APPLIED=1` | 3 | 0 | identical to b, provider warning gone | |

### 1.2 VERIFIED wire facts (quoted from `captures/openclaude/*/requests.jsonl`)

- **Route and headers.** Only `POST /r/<key>/v1/chat/completions` is ever called; **`GET /models` is never called**
  (`models_get_called: false` in every case). Headers: `user-agent: claude-cli/99.0.0 (undefined, sdk-cli)`,
  `authorization: Bearer <OPENAI_API_KEY>` (equals the path key: `bearer_matches_path_key: true`), `accept: */*`,
  `accept-encoding: gzip, deflate`, `content-type: application/json`, `connection: keep-alive`. No `x-*` harness headers.
- **Body top-level keys (root):** exactly `["max_tokens", "messages", "model", "stream", "tools"]`; `stream: true`,
  `max_tokens: 32000` (= `CLAUDE_CODE_MAX_OUTPUT_TOKENS`), `model: "qwen/qwen3.8-27b"`, **no `stream_options`**, no
  `temperature`, no `tool_choice`, no `reasoning_effort`, no `store`. The **solver child** body keys are
  `["max_tokens", "messages", "model", "stream", "temperature"]` with `temperature: 1` and **no `tools` key at all**
  (`tools: []` in `--agents` really means no tools, although the Agent description advertises "solver … (Tools: All tools)").
- **`tools[]` (root) is exactly `[{"type":"function","function":{"name":"Agent",…}}]`** — `--tools "Agent"` without
  `--bare` yields exactly `{Agent}`; no `Read`/`Write`/`TodoWrite`/`SendMessage` etc. appear. Agent `parameters`:
  ```json
  {"type":"object","properties":{"description":{"type":"string",…},"prompt":{"type":"string",…},
   "subagent_type":{"type":"string",…},"model":{"type":"string","enum":["sonnet","opus","haiku"],…},
   "isolation":{"type":"string","enum":["worktree"],…}},"required":["description","prompt","subagent_type"],
   "additionalProperties":false}
  ```
  The Agent `description` is 6,331 chars and lists the built-in agent roster: `general-purpose … (Tools: *)`,
  `statusline-setup … (Tools: Read, Edit)`, `Explore`, `Plan`, `verification` and our `solver … (Tools: All tools)`,
  plus Claude-Code usage notes and examples (`claude-code-guide`, hooks, status line). It is paid as input on every
  root call (S_h covariate) and it invites `general-purpose` unless the system prompt forbids it.
- **`messages[0]` is NOT byte-identical to `system_dynorch.txt`.** VERIFIED:
  `messages[0].content == "You are OpenClaude, built on the Claude Agent SDK.\n\n" + system_dynorch.txt.rstrip("\n")`.
  The prefix comes from `src/constants/system.ts::getCLISyspromptPrefix` (cli.mjs ≈1913932: non-interactive → `AGENT_SDK_PREFIX
  = "You are ${PRODUCT_DISPLAY_NAME}, built on the Claude Agent SDK."`) and is prepended unconditionally in the
  system-prompt assembly (cli.mjs ≈18335631: `asSystemPrompt([getAttributionHeader(…), getCLISyspromptPrefix({…}),
  ...systemPrompt, …])`); the only `skipSystemPromptPrefix` switch is on the Anthropic-direct path. There is no flag or
  env var that removes it. The sha256 of `messages[0]` is stable across runs (same for a, b, c, d, e).
- **`messages[1]` (user) is NOT the task text alone.** VERIFIED: `user == "<system-reminder>\nAs you answer the user's
  questions, you can use the following context:\n# currentDate\nToday's date is 2026-09-23.\n\n      IMPORTANT: this
  context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant
  to your task.\n</system-reminder>\n\n\n\n" + task.txt` (308-char prefix, the task verbatim incl. its trailing newline).
  The date changes daily, so the root conversation key must not hash the raw first user text.
- **Solver child conversation.** `messages[0].content` = the same OpenClaude prefix + `subagent_dynorch.txt.rstrip("\n")`
  + `"\n\nNotes:\n- Agent threads always have their cwd reset between bash calls, as a result please only use absolute
  file paths.\n- In your final response, share file paths … \n- For clear communication with the user the assistant MUST
  avoid using emojis.\n- Do not use a colon before tool calls. …\n\nHere is useful information about the environment you
  are running in:\n<env>\nWorking directory: <ws>\nIs directory a git repo: No\nPlatform: linux\nShell: unknown\nOS
  Version: Linux 4.18.0-…\n</env>\nYou are powered by the model qwen/qwen3.8-27b."` (full text in
  `captures/openclaude/b/requests.jsonl` seq 2). Its sha therefore differs per working directory: **the contract §1 rule
  "system sha equals `subagent_system_sha`" cannot classify the solver; classify by containment of the subagent prompt
  text (or by the sha of the text between the OpenClaude prefix and `"\n\nNotes:"`)**. `messages[1]` = the same
  `<system-reminder>` block + the `prompt` argument of the Agent call, verbatim.
- **Tool-call round trip.** The stub's `tool_calls[0].id` (`call_1c3491764440`) is echoed back verbatim as
  `messages[2].tool_calls[0].id` and as `messages[3].tool_call_id`; `arguments` are re-sent as the same JSON string;
  the assistant message is `{"role":"assistant","content":"","tool_calls":[…]}` (**`content: ""`, never null**). The
  tool result is `{"role":"tool","tool_call_id":"call_…","content":"<rationale>\nStub reply.\n</rationale>\n<final>child
  says 7</final>\n\nagentId: a8613a253fd812708 (use SendMessage with to: 'a8613a253fd812708' to continue this
  agent)\n<usage>total_tokens: 120\ntool_uses: 0\nduration_ms: 30</usage>"}` — the harness appends an `agentId` line and a
  `<usage>` block (with the stub's usage numbers) to the child's text.
- **`reasoning_content` is accepted and NOT re-sent** (case e: `assistant_messages_resent[0].keys == ["content","role",
  "tool_calls"]`, `has_reasoning_content: false`). Reasoning re-injection must come from the proxy cache keyed on the
  echoed tool-call id (contract §4 key 1) — that key is available: ids are echoed exactly.
- **Streaming.** The stub's SSE was consumed correctly: role chunk, `reasoning_content` delta, `tool_calls` deltas
  (`index`, `id`, `type`, `function.name`, then `function.arguments`), `finish_reason`, a usage chunk with `choices: []`,
  `[DONE]`. The stdout `usage` reflects the usage chunk (`input_tokens: 100, output_tokens: 20` per call; `modelUsage`
  sums root+child: `inputTokens: 300, outputTokens: 60` in case b because the child call is counted too).
- **330 s delay (case d):** headers-then-keepalive-comments every 15 s for 330 s produced exactly one request, no retry,
  `subtype: success`, `duration_api_ms ≈ 330 s`. `API_TIMEOUT_MS=3600000` + keepalives are sufficient; nothing in
  stderr about timeouts.
- **HTTP 402 (case c):** the body `{"error":{"code":"budget_exhausted","message":"…","type":"insufficient_quota"}}` ends the
  run immediately: no retry, no further request, process exit **1** within ~1 s, stdout still a single `type: "result"`
  JSON with `subtype: "success"` but `is_error: true` and `result` = the error text. Note `subtype` is NOT
  `error_*`; the adapter must key on `is_error`, not `subtype`.
- **stdout JSON shape** (case a): keys `["duration_api_ms","duration_ms","fast_mode_state","is_error","modelUsage",
  "num_turns","permission_denials","result","session_id","stop_reason","subtype","total_cost_usd","type","usage","uuid"]`;
  `result` = last assistant text; `total_cost_usd: 0.001` (b: 0.003) — computed from an unknown-model price table, so
  **ignore**; no `budget_usd` string anywhere in stdout or in any request (`budget_usd_text_*: false`).
- **stderr** (every run): `[context] Warning: model "qwen/qwen3.8-27b" not in integration model metadata — using
  conservative 128k default…` ×7, plus `Warning: ignoring saved provider profile. Codex auth is required for codexplan.
  Set CODEX_API_KEY, choose Codex OAuth in /provider or put auth.json at $HOME/.codex/auth.json.` — the latter comes
  from `buildStartupEnvFromProfile` (no persisted profile → tries a Codex profile → validation fails → env kept). It is
  harmless but **`CLAUDE_CODE_PROVIDER_PROFILE_ENV_APPLIED=1` silences it** (case x_profile_env, behaviour otherwise
  identical). The `agentModels`/`agentRouting` keys in `.openclaude.json` are not needed (case x_minimal identical).
- **Files created** (all under `CLAUDE_CONFIG_DIR`, none elsewhere): `cfg/settings.json` (`{"skipDangerousModePermissionPrompt":
  true}`), `cfg/backups/.openclaude.json.backup.<ts>`, and despite `--no-session-persistence`
  `cfg/projects/<ws-slug>/<session-uuid>/subagents/agent-<id>.meta.json` whenever an Agent call happens. Nothing in the
  workspace, nothing in `$HOME` outside `cfg/`, nothing under `/home`.

### 1.3 Minimal compliant reply sequence (chat/completions, `stream: true`)

```
data: {"id":…,"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
data: {…"choices":[{"index":0,"delta":{"reasoning_content":"…"},"finish_reason":null}]}          # optional
data: {…"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"Agent","arguments":""}}]},"finish_reason":null}]}
data: {…"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"description\":…}"}}]},"finish_reason":null}]}
    — or, for text —  data: {…"choices":[{"index":0,"delta":{"content":"<rationale>…</rationale>\n<final>7</final>"},"finish_reason":null}]}
data: {…"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"|"stop"}]}
data: {…"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"completion_tokens_details":{"reasoning_tokens":5}}}
data: [DONE]
```
Keepalive comment lines `: dynorch keepalive\n\n` before the first data chunk are ignored by the parser (case d). The
non-streaming JSON form was exercised only by curl (self-test), never by OpenClaude, which always streams.

### 1.4 Checklist of adapter / proxy changes vs plan §6 (OpenClaude)

1. **Contract §1 / §9 Gate 0 wording "messages[0] byte-identical to SYSTEM.md" is false as stated.** Replace with:
   `messages[0].content == OPENCLAUDE_PREFIX + system_text.rstrip("\n")` where `OPENCLAUDE_PREFIX = "You are OpenClaude,
   built on the Claude Agent SDK.\n\n"`; the Gate 0 check should assert that equality and record the prefix in
   `locks/harness_prompts.json`. The proxy's root classification must use the sha of the observed first system text of
   the run (root = first conversation) — that already works — but `other_share == 0` needs the solver rule below.
2. **Solver classification by containment, not equality** (§1): solver iff the OpenClaude prefix is followed by
   `subagent_dynorch.txt.rstrip("\n")` (the harness appends `\n\nNotes:…<env>…You are powered by the model …`). Interface
   request for WP-A: expose `subagent_system_text` and classify by `text.startswith(prefix + subagent_text)` or by a
   normalised sha (strip prefix, cut at `"\n\nNotes:"`). Otherwise every solver call lands in `other` and Gate 0 fails.
3. **First user text is not the task text**: a `<system-reminder>` block with today's date precedes it. Conversation
   keys (§1) must therefore not hash the raw user turn; the B8 budget sentence is still delivered verbatim (the task is
   appended unchanged). The solver's user turn is the spawn `prompt` after the same reminder block.
4. Env: add `CLAUDE_CODE_PROVIDER_PROFILE_ENV_APPLIED=1` to the plan §6 env list (silences the profile warning; no
   behaviour change). `agentModels`/`agentRouting` in `.openclaude.json` are optional; keep `hasCompletedOnboarding`
   and `bypassPermissionsModeAccepted`.
5. Adapter `parse_result`: `cli_result` = `result` of the single stdout JSON; success iff `is_error == false`
   (`subtype` stays `"success"` on a 402). Record `num_turns`, `usage`, `modelUsage`, `stop_reason`; ignore
   `total_cost_usd`.
6. Proxy tool-surface allowlist for H1 = `{"Agent"}` exactly (verified); the solver's requests carry no `tools` key, so
   the check must treat a missing `tools` as an empty list. The solver request carries `temperature: 1` — an
   `ignored_body_keys` entry, not a violation.
7. Reasoning re-injection for H1 relies solely on the proxy cache (ids echoed exactly; `reasoning_content` never re-sent).
8. Tool-result text seen by the root = child text + `"\n\nagentId: … (use SendMessage …)\n<usage>…</usage>"`; the
   `<usage>` numbers are the proxy's own usage figures (from the usage chunk), so the child's `completion_tokens` leak into
   the root's prompt — harmless, but `SendMessage` is in `--disallowedTools`, so a model that follows that hint gets a
   permission denial (logged in `permission_denials`).
9. The Agent description advertises `general-purpose (Tools: *)`, `Explore`, `Plan`, `verification`, `statusline-setup`:
   the system prompt must say `subagent_type` must be `solver`; the proxy flags any other `subagent_type` value in a
   parsed Agent call (spawn arguments are visible in `tool_calls[].function.arguments`).
10. Runner: `cfg/projects/…/subagents/*.meta.json` is written per spawn even with `--no-session-persistence`; keep
    `CLAUDE_CONFIG_DIR` under the per-run work dir (already the plan) and count it in the per-run cleanup.

---

## 2. Codex CLI 0.156.1 (H3)

Invocation = plan §6 verbatim: `CODEX_HOME=<probe>/home` holding the plan's `config.toml` (`captures/codex/<case>/config.toml`),
`timeout --kill-after=30s <wall>s codex exec --json --ephemeral --strict-config --skip-git-repo-check -C <ws> -s read-only
-o <final.txt> -` with the task on stdin; env only `PATH HOME CODEX_HOME TMPDIR LANG TERM OPENAI_API_KEY` (`env_key`).

### 2.1 Results per case

| case | script | requests (role) | exit | events.jsonl | `-o` file | wall |
|---|---|---|---|---|---|---|
| a | text | 1 (root) | 0 | thread.started, item.completed/error, turn.started, item.completed/agent_message, turn.completed | final text | 18 s first run (writes system skills), <1 s after |
| b_ns | spawn_agent **with** `namespace:"multi_agent_v1"` → wait_agent → text | 4: root, root, **child (`x-openai-subagent: collab_spawn`)**, root | 0 | … item.started/completed **collab_tool_call** ×2, agent_message, turn.completed (usage 300/60/15 = root+child summed) | final text | 0.8 s |
| b_nons | same **without** `namespace` | 3 (root ×3, **no child**) | 0 | agent_message, turn.completed; stderr `ERROR codex_core::tools::router: error=unsupported call: spawn_agent` | final text | 0.6 s |
| c | function_call `exec_command` | 2 (root) | 0 | no `command_execution` item; next request carries `function_call_output: "unsupported call: exec_command"`; leak file not created | final text | 0.5 s |
| d | stream without `response.completed` | 3 (root, 2 retries) | **1** | 3× `error: "Reconnecting... n/2 (stream disconnected before completion: stream closed before response.completed)"`, then **turn.failed** | none | 1.0 s |
| e | `response.failed` `rate_limit_exceeded`, then text | 2 | 0 | `error: "Reconnecting... 1/2 (rate limit exceeded: stub rate_limit_exceeded)"`, agent_message, turn.completed | final text | 0.8 s |
| f | text after **660 s** delay with 43 keepalive comments | 1 (root) | 0 | thread.started, item.completed/error, turn.started, item.completed/agent_message, turn.completed (usage 100/20/5) | final text | 660.4 s |
| g | `--strict-config` + unknown key | 0 | **1** | stderr: `Error loading config.toml:\n<path>:11:1: unknown configuration field \`this_key_does_not_exist\`` | none | 0.1 s |
| h | reasoning item + wait_agent(targets: []) → text | 2 | 0 | usage 200/40/10 | final text | |
| x_goals_off | a + `-c features.goals=false` | | 0 | goal tools gone | | |
| x_tools_struct | a + `[tools.experimental_request_user_input] enabled=false`, `[tools.update_plan] enabled=true`, `features.goals=false` | | 0 | tools exactly `update_plan` + `multi_agent_v1` | | |
| x_rui_off / x_update_plan / x_tools_table | `tools.experimental_request_user_input=false` (bool) | 0 | 1 | `invalid type: boolean \`false\`, expected struct ExperimentalRequestUserInput` (same for `tools.update_plan`) | | |
| x_skills_table | `[skills] enabled=false` | 0 | 1 | `unknown configuration field \`skills.enabled\`` | | |
| x_skills_off | `features.skip_host_skill_discovery=true` | | 0 | skills block still present | | |

### 2.2 VERIFIED request shape (`captures/codex/a/requests.jsonl`, first request)

- Route `POST /r/<key>/v1/responses` only; **no `GET /models`**. Headers: `accept: text/event-stream`, `authorization:
  Bearer <OPENAI_API_KEY>` (= path key), `originator: codex_exec`, `user-agent: codex_exec/0.156.1 (Rocky Linux 8.10.0;
  x86_64) dumb (codex_exec; 0.156.1)`, `session-id`, `thread-id`, `x-client-request-id`, `x-codex-window-id`,
  `x-codex-beta-features: remote_compaction_v2`, `x-codex-turn-metadata: {"installation_id":…,"session_id":…,
  "thread_id":…,"agent_name":"/root","turn_id":…,"request_kind":"turn","thread_source":"user","turn_trigger":"exec",
  "sandbox":"seccomp","sandbox_mode":"read-only",…,"model":"qwen/qwen3.8-27b"}`. **No gzip request compression** was
  used (`content-encoding` absent; the stub would have decompressed it).
- Body top-level keys: `["client_metadata","include","input","instructions","model","parallel_tool_calls",
  "prompt_cache_key","reasoning","store","stream","tool_choice","tools"]`. Values: `stream: true`, `store: false`,
  `include: ["reasoning.encrypted_content"]`, **`reasoning: {}`** (empty object with `model_reasoning_summary = "none"`
  and no `model_reasoning_effort`), `tool_choice: "auto"`, **`parallel_tool_calls: true`** (the design note said false),
  `prompt_cache_key: <session uuid>`, **no `text`**, no `max_output_tokens`, no `stream_options`, no `service_tier`;
  `client_metadata` = the same fields as the `x-codex-*` headers.
- **`instructions` == `system_dynorch.txt` with the trailing newline stripped** (1216 chars; the file is 1217 bytes;
  `instructions == file.rstrip("\n")` VERIFIED). Codex does NOT append its multi-agent instructions to `instructions`
  when `model_instructions_file` is set — the collab guidance is only in the tool descriptions.
- `input` (root, first call) = 3 items: (1) `message/developer` with two `input_text` parts — a `<skills_instructions>`
  block listing the five bundled system skills (imagegen, openai-docs, plugin-creator, skill-creator, skill-installer;
  Codex copies ≈60 files into `$CODEX_HOME/skills/.system/` at first start) and
  `"<permissions instructions>\nFilesystem sandboxing defines which files can be read or written. \`sandbox_mode\` is
  \`read-only\`: The sandbox only permits reading files. Network access is restricted.\nApproval policy is currently
  never. Do not provide the \`sandbox_permissions\` for any reason, commands will be rejected.\n</permissions
  instructions>"`; (2) `message/user` `<environment_context>\n  <cwd>…/ws</cwd>\n  <shell>bash</shell>\n
  <current_date>2026-09-23</current_date>\n  <timezone>America/New_York</timezone>\n  <filesystem>…</filesystem>\n
  </environment_context>`; (3) `message/user` = **task.txt verbatim** (incl. trailing newline). Again the first user
  turn is not the task text; the date and cwd vary per run.
- **`tools[]` with the plan's config**: `[{"type":"function","name":"request_user_input",…}, {"type":"namespace",
  "name":"multi_agent_v1","description":"Tools for spawning and managing sub-agents.","tools":[close_agent,
  resume_agent, send_input, spawn_agent, wait_agent]}, {"type":"function","name":"get_goal"}, {…"create_goal"},
  {…"update_goal"}]`. **Absent (VERIFIED):** `exec_command`, `write_stdin`, `shell`, `apply_patch`, `web_search`,
  `view_image`, `update_plan`, `sleep`, `current_time`, `get_context_remaining`, any MCP tool. **Present but not in the
  plan:** `request_user_input` ("This tool is only available in Plan mode") and the three goal tools (feature `goals`,
  stable, default true). With `[features] goals = false` plus `[tools.experimental_request_user_input] enabled = false`
  and `[tools.update_plan] enabled = true` the list becomes exactly `update_plan` + the `multi_agent_v1` namespace
  (case x_tools_struct) — the plan's intended surface. Every function has `strict: false`.
- `spawn_agent` parameters: `{fork_context: boolean ("True forks the current thread history into the new agent; false
  or omitted starts with only the initial prompt."), items: [...], message: string, model: string, reasoning_effort:
  string}`, no `required`; its description begins with a list of "Available model overrides" (`gpt-6-astra`, `gpt-6-sol`,
  `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, …) that the model could pass as `model` — the proxy must ignore any
  `model` field of a spawn (the child request still carries `model: "qwen/qwen3.8-27b"` because the provider is
  inherited; VERIFIED in case b_ns). `wait_agent`: `{targets: string[] (required), timeout_ms: number (default 30000,
  min 10000, max 3600000)}`.

### 2.3 VERIFIED spawn round trip (case b_ns)

1. Root reply `function_call {name:"spawn_agent", namespace:"multi_agent_v1", call_id:"call_15fa…", arguments:"{\"message\":…}"}`.
   Codex echoes it back on the next request as `{"type":"function_call","id":"fc_…","name":"spawn_agent",
   "namespace":"multi_agent_v1","arguments":"…","call_id":"call_15fa…"}` followed by
   `{"type":"function_call_output","id":"fco_…","call_id":"call_15fa…","output":"{\"agent_id\":\"01a0d07d-913b-…\",
   \"nickname\":\"Franklin\"}"}`. **Without `namespace` the call is rejected** (`output: "unsupported call: spawn_agent"`,
   stderr `ERROR codex_core::tools::router`) and no child is created (case b_nons). The proxy MUST set `namespace` on
   function_call items whose tool came from a namespace (contract §6 already says so; this is the evidence).
2. The child request arrives at the same base URL with `x-openai-subagent: collab_spawn`, `x-codex-parent-thread-id`,
   `thread-id` ≠ root, same `session-id`, turn metadata `"thread_source":"subagent","subagent_kind":"thread_spawn"`,
   **the same `instructions`** (same sha as the root → the contract §1 solver-by-system-sha rule cannot work for Codex;
   classify by the `x-openai-subagent` header), `input` = [developer (skills + permissions), user environment_context,
   user = the spawn `message` verbatim] — **not a full-history fork** (`fork_context` omitted). Its `tools` = the root's
   minus the `multi_agent_v1` namespace (depth 1 enforced; `request_user_input` and goal tools still present under the
   plan config).
3. Root reply `function_call wait_agent {targets:[agent_id]}` (namespace set) → next root request carries
   `function_call_output: "{\"status\":{\"<agent_id>\":{\"completed\":\"<child text>\"}},\"timed_out\":false}"` **and an
   injected user message** `"<subagent_notification>\n{\"agent_path\":\"<agent_id>\",\"status\":{\"completed\":\"<child
   text>\"}}\n</subagent_notification>"`. Both are inputs the proxy renders as `tool` / `user` turns.
4. events.jsonl shows `item.started/item.completed` with `item.type == "collab_tool_call"` for spawn and wait, then
   `agent_message`, then `turn.completed` whose `usage` is the **sum over root and child calls** as reported by the
   stub's `response.completed.usage` (300 input / 60 output / 15 reasoning for 3 calls of 100/20/5): usage in
   `turn.completed` is derived entirely from our `usage` objects (`input_tokens`, `cached_input_tokens: 0` from
   `input_tokens_details.cached_tokens`, `cache_write_input_tokens: 0`, `output_tokens`, `reasoning_output_tokens` from
   `output_tokens_details.reasoning_tokens`).

### 2.4 VERIFIED protocol robustness

- **Missing `response.completed` (case d):** `stream_max_retries = 2` → 3 requests total; each retry's `input`
  contains the output items already received (`message:assistant` items accumulate: 0, 1, 2 on requests 1, 2, 3), i.e.
  **an `output_item.done` emitted before the failure is treated as history**. Then `turn.failed {"error":{"message":
  "stream disconnected before completion: stream closed before response.completed"}}`, exit 1, no `-o` file. The proxy
  must never emit output items before it is sure it will emit `response.completed` (buffer, then send items + completed).
- **`response.failed` with `rate_limit_exceeded` (case e):** one `error: "Reconnecting... 1/2 (rate limit exceeded: …)"`
  event, an immediate retry with an identical request, success. So `rate_limit_exceeded` is the right retryable code
  (contract §6) and a duplicate-served cache hit is what the retry should get.
- **Unknown tool `exec_command` (case c):** no process is spawned (no `command_execution` item, no sandbox invocation,
  leak file absent), the model gets `function_call_output: "unsupported call: exec_command"`. The same for any name not
  in the tool list. strace could not be used on this host (`PTRACE_TRACEME: Operation not permitted`), so "no process"
  rests on the absence of `command_execution` items, the router error and the missing leak file.
- **Reasoning items (case h):** a `{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":…}]}`
  output item is accepted and **echoed back verbatim on the next request** as
  `{"type":"reasoning","id":"rs_…","summary":[],"content":[{"type":"reasoning_text","text":"STUB-REASONING-ONE: …"}],
  "encrypted_content":null}` placed before the `function_call` item. With `show_raw_agent_reasoning = false` no
  reasoning item appears in events.jsonl. So for H3 the harness itself supplies `reasoning_content`
  ("explicit" re-injection, contract §4) — the proxy cache is only a fallback.
- **Exec-mode limits:** `wait_agent` with `targets: []` → `output: "agent ids must be non-empty"`. `request_user_input`
  is advertised but exec mode rejects it (`"request_user_input is not supported in exec mode"` string in the binary).
- **`--strict-config`:** an unknown key is a hard error before any request (case g). Typos in the generated config
  cannot silently pass. Bool-valued `tools.*` keys are rejected (they are structs).
- **stderr:** empty in every successful run (the "Unknown model … fallback metadata" notice is an `item.completed`
  `{"type":"error","message":"Model metadata for \`qwen/qwen3.8-27b\` not found. Defaulting to fallback metadata; this
  can degrade performance and cause issues."}` event on stdout, not a stderr warning). No bwrap/bubblewrap message ever
  appeared (`bwrap_mentioned_in_stderr: false`): with `shell_tool = false` the sandbox is never launched, so the bundled
  bwrap was **not** exercised by these runs (the design note's kernel-4.18 question stays open but moot for H3).
- **Files written by Codex** (all under `CODEX_HOME`): `config.toml` (ours), `.sandbox_migration`, `installation_id`,
  `goals_1.sqlite`, `logs_2.sqlite`, `memories_1.sqlite`, `queue_1.sqlite`, `skills/.system/**` (≈60 files, once per
  fresh `CODEX_HOME`; the first run took 18 s because of it, later runs < 1 s) and `final.txt` where `-o` points.
  Nothing under `/home`, nothing in the workspace. `codex --version` prints `WARNING: failed to clean up stale arg0 temp
  dirs: Directory not empty` when `TMPDIR` holds leftovers — harmless.

### 2.5 660 s delay (case f)

VERIFIED (`captures/codex/f/summary.json`): the stub sent the response headers at once, then `: dynorch keepalive\n\n`
comments every 15 s for 660 s (43 comments), then `response.created` → `output_item.done` → `response.completed`.
Result: exactly **one** request, no reconnect, no `error` event, `turn.completed` with our usage, `-o` file written,
exit 0, wall 660.4 s. `stream_idle_timeout_ms = 1800000` plus keepalive comments is sufficient; whether comment lines
alone reset Codex's idle timer or the 1800 s budget simply had not elapsed is not separable at 660 s (the default idle
timeout is 300 s, so the comments did reset it or the header timer does not apply — either way the plan's setting works).

### 2.6 Minimal compliant reply sequence (Responses SSE)

```
event: response.created
data: {"type":"response.created","response":{"id":"resp_…","object":"response","status":"in_progress","output":[]}}

event: response.output_item.done            # optional reasoning item (echoed back next turn)
data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_…","summary":[],"content":[{"type":"reasoning_text","text":"…"}]}}

event: response.output_item.done            # one of:
data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_…","call_id":"call_…","name":"spawn_agent","arguments":"{\"message\":\"…\"}","status":"completed","namespace":"multi_agent_v1"}}
data: {"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_…","role":"assistant","status":"completed","content":[{"type":"output_text","text":"…","annotations":[]}]}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_…","object":"response","status":"completed","output":[…items…],"usage":{"input_tokens":100,"input_tokens_details":{"cached_tokens":0},"output_tokens":20,"output_tokens_details":{"reasoning_tokens":5},"total_tokens":120}}}
```
Error: `event: response.failed` / `data: {"type":"response.failed","response":{"id":…,"status":"failed","error":{"code":
"rate_limit_exceeded","message":"…"}}}` (retried up to `stream_max_retries`). No `response.output_item.added`,
`response.output_text.delta` or `response.in_progress` events are needed (none were sent; Codex was happy).

### 2.7 Checklist of adapter / proxy / config changes vs plan §6 (Codex)

1. **config.toml additions** (plan §6 block): `[features] goals = false`; `[tools.experimental_request_user_input]
   enabled = false`; `[tools.update_plan] enabled = true`. With these the offered tools are exactly `update_plan` +
   `multi_agent_v1::{spawn_agent, send_input, wait_agent, resume_agent, close_agent}` (the plan's stated surface). The
   H3 allowlist in `locks/harness_tools.json` must list those six names; `request_user_input`/`get_goal`/`create_goal`/
   `update_goal` would otherwise trip the fail-closed tool-surface check (or, worse, be silently allowed).
2. **Skills block cannot be removed by config** (`[skills]` table unknown; `skill_search=false` and
   `skip_host_skill_discovery=true` leave it). Budget ≈ 700 input tokens per call for `<skills_instructions>` +
   `<permissions instructions>` + `<environment_context>` in the S_h covariate; the model has no tool to read a
   SKILL.md so it is inert.
3. **Classification (§1):** root vs child by `x-openai-subagent` (value `collab_spawn`), not by instructions sha (identical
   for both). Interface request for WP-A/WP-B: the `ChatRequest` needs a `subagent_header: str | None` (or the wire
   layer passes `conversation_class_hint`), and `other_share` must not count Codex children as `other`.
4. **`namespace` on emitted `function_call` items is mandatory** for namespaced tools (the contract §6 text already
   requires it; b_nons is the failure mode). `ToolSpec.namespace` must survive flattening and be written back.
5. **Never emit `response.output_item.done` before the reply is final**: a partial item followed by a dropped stream is
   kept in Codex's history and re-sent on the retry. The proxy buffers the whole upstream reply (it does anyway, B2) and
   emits items + `response.completed` back-to-back; the `duplicate_served` cache must key on the request as re-sent
   (identical bytes on retry, VERIFIED in cases d/e).
6. **Reasoning:** Codex re-sends `reasoning` items with `content[].text`; map them to `Message.reasoning_content`
   (contract §6 says so) and count them as `reinjection: "explicit"`. `encrypted_content: null` must be tolerated.
7. **`parallel_tool_calls: true`** and `reasoning: {}` are sent; both go to `ignored_keys`. `prompt_cache_key` is the
   session uuid (constant per run) — usable as a secondary run fingerprint, not needed.
8. The injected `<subagent_notification>` user message after a completed child and the `function_call_output` JSON
   (`{"agent_id":…,"nickname":…}`, `{"status":{id:{"completed":…}},"timed_out":false}`) are harness-authored text the
   proxy renders as-is; the delivered-answer rule (B7, last root assistant message) is unaffected, but `final_message`
   extraction must skip `reasoning`/`function_call` items and take the last `message` item the proxy served to the root.
9. Adapter `parse_result`: `cli_result` = `-o` file (written only on success) or the last `item.completed` with
   `item.type == "agent_message"`; failure iff a `turn.failed` event or exit ≠ 0; the `item.completed {type:"error",
   "Model metadata … not found"}` event is normal noise. `turn.completed.usage` is the sum over the whole thread tree of
   our own usage numbers (cross-check only).
10. Spawn arguments may carry `model` / `reasoning_effort` overrides from the advertised list (`gpt-6-astra`, …): the
    child still requests `qwen/qwen3.8-27b` (provider inheritance), so the proxy ignores `model` on the wire and records
    the requested override in the spawn call record.
11. Runner: pre-create `CODEX_HOME` (Codex requires it to exist) and expect ≈60 skill files + 4 sqlite files per run
    dir; the first start per fresh home costs ≈18 s (skills copy), so the wall-clock budget `600 + 120·k` is safe but
    the per-run directory count is ~70 files (inode note for scratch, not `/home`).
12. Codex `x-codex-turn-metadata` carries `sandbox_mode`, `thread_source`, `parent_thread_id`: journal it (redaction not
    needed; contains no secrets) — it is the cheapest way to attribute a call to a thread for `dynorch_calls.parquet`.

---

## 3. Cross-cutting facts for the proxy (both harnesses)

- Both harnesses put the bearer equal to the path key on every request (root and child) — the contract's 401 rule is safe.
- Neither harness calls `GET /models`; keep the route (cheap) but do not depend on it.
- Neither harness sends `stream_options`; the chat path must always append the usage chunk (contract §6 already says so).
- Both harnesses prepend harness text to the task turn (OpenClaude `<system-reminder>` with the date; Codex
  `<environment_context>` with cwd/date/timezone as a separate user item). The B8 budget sentence arrives verbatim in both.
- Both harnesses echo proxy-minted call ids exactly (`call_<12hex>` worked; Codex additionally mints its own `fc_`/`fco_`
  item ids).
- 15 s keepalive comments through a 330 s (OpenClaude) and a 660 s (Codex) stall each produced exactly one request and
  a successful run (`subtype: success` / `turn.completed`); the B9 keepalive design holds for both harnesses.
- Neither run wrote under `/home`; every state file stayed under the per-run `HOME`/`CLAUDE_CONFIG_DIR`/`CODEX_HOME`.

## 4. Not done / caveats

- No paid call, no network beyond 127.0.0.1, no proxy code: this probe used a stub, not `dynorch/httpd.py`; the WP-B
  smoke tests must re-run the same cases through the real proxy + `api/fake.py`.
- strace-based "no process spawned" evidence is unavailable on this host (ptrace denied); the Codex exec_command case
  relies on the router error, the absent `command_execution` item and the absent leak file.
- The bundled bwrap was not exercised (nothing to sandbox with `shell_tool = false`); the kernel-4.18 bwrap question is
  moot for H3 as configured.
- OpenClaude `--tools ""` (E2 spawn-disabled control) and the rung-25 fake run were not probed here (Gate 0 items owned
  by WP-B's smoke tests).
- No unit tests accompany the probe: WP-E owns no `tests/unit/dynorch` file (contract §9); the captures are the
  fixtures the WP-B `test_openai_format` / `test_responses_format` / `test_adapters` tests should assert against.
