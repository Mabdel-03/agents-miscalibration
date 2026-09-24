# dynorch_v1 — integration contract (frozen on day 0; only the lead edits this file)

Study `openrouter_dynorch_v1`; branch `dynorch-v1` off `forced-v2@8b6c5f4`; package `src/agent_allocation/dynorch/`;
config `configs/dynorch_v1.json` (the frozen parameter table lives there under `dynorch` and `budget`; this file
explains the interfaces the work packages code against). Plan: `~/.claude/plans/we-ran-the-plans-lovely-meteor.md`.
Run root `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/dynorch_v1/artifacts`; every harness state dir,
workspace, temp dir and npm prefix lives under `/orcd/data/tpoggio/001/mabdel03/dynorch_scratch/` (never `/home`).

Rules that apply to every work package: never print or copy the OpenRouter key (`apikey.load_api_key()` only, headers
through `apikey.redact_headers`); nothing new is written under `/home`; tests run with
`source slurm/env.sh && uv run --frozen --no-sync pytest tests/unit/dynorch -q`; no paid call outside `--allow-paid`
verbs; the sealed forced_v2 run root is read-only; every new module has a module docstring naming its WP and the
contract section it implements; do not edit files owned by another WP (list in §9) — write an interface request in
`docs/dynorch/requests/<wp>.md` instead.

## 1. Vocabulary

- **run** = one (arm, item, rung, repeat, attempt): one harness process (or one plain call for H0) under one `run_key`.
- **arm** ∈ core {H1 openclaude, H3 codex, H0 null} (extensions H1c/H3c capped, H1n/H3n spawn-disabled, H2 openclaw).
- **rung k** ∈ {1,2,4,8,16,25}; **A** = 1024; **C** = k·A = the run's cap on completion tokens.
- **conversation** = a distinct conversation seen by the proxy under one run_key (keyed by the system text and the
  wire-level thread markers, NEVER by the raw first user turn: both harnesses inject a dated block before the task).
  The first non-subagent conversation is the **root**; a conversation is **solver** iff the request carried a
  sub-agent marker (Codex `x-openai-subagent` header → `ChatRequest.subagent`) or its system text contains
  `prompts/subagent_dynorch.txt` after the harness's own prefix (OpenClaude); anything else is **other**.
  (Amended by §10.1 — system-sha equality classifies neither harness.)
- **grant** = `max_tokens` given to one upstream call = `C − spent` at lock acquisition (E0 capped arms: root calls
  `min(C − spent, max(A, C/4))`).
- **grace** = the ≤ 1,024-token continuation outside C, root conversation only, at most one per run
  (`forced_close_kind ∈ {none, mid_call, visible_continuation, terminal}`).

## 2. Internal message model (WP-A owns; WP-B converts to/from wire formats)

```python
# dynorch/records.py
@dataclass(frozen=True)
class ToolSpec:            # OpenAI function tool, already flattened (namespaces removed)
    name: str; description: str; parameters: dict[str, Any]; namespace: str | None = None
@dataclass(frozen=True)
class ToolCall:
    id: str; name: str; arguments: dict[str, Any] | str      # dict when JSON-parseable, else the raw string
@dataclass(frozen=True)
class Message:             # roles: system | user | assistant | tool
    role: str; content: str; reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = (); tool_call_id: str | None = None; name: str | None = None
@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[Message, ...]; tools: tuple[ToolSpec, ...]
    requested_max_tokens: int | None; requested_reasoning_effort: str | None; stream: bool
    ignored_keys: tuple[str, ...]; wire: str                  # "chat" | "responses"
@dataclass(frozen=True)
class ChatReply:
    content: str; reasoning_content: str; tool_calls: tuple[ToolCall, ...]
    finish_reason: str                                         # "stop" | "tool_calls" ONLY
    usage: ReplyUsage; call: CallRecord; headers: dict[str, str]   # X-Dynorch-* headers
@dataclass(frozen=True)
class ReplyUsage:
    prompt_tokens: int | None; completion_tokens: int | None; reasoning_tokens: int | None
    cached_prompt_tokens: int | None
```
`ProxyError(status: int, code: str, message: str, retryable: bool)` is the only exception the wire layer converts:
codes `unknown_run` (404), `bad_bearer` (401), `run_closed` (409), `budget_exhausted` (402), `call_limit` (402),
`input_limit` (402), `tool_surface_violation` (400), `bad_request` (400), `ledger_refused` (503), `upstream_failed`
(502, non-retryable for the harness; the run becomes `transport_failed`).

## 3. ProxyCore (WP-A owns `dynorch/core.py`)

```python
class ProxyCore:
    def __init__(self, settings: DynorchSettings, upstream: Upstream, journal: CallJournal, store: RunStore) -> None
    def open_run(self, spec: RunSpec, *, root_system_text: str, subagent_system_text: str | None) -> None
    def close_run(self, run_key: str, *, status: str) -> RunRecord
    async def chat(self, run_key: str, request: ChatRequest) -> ChatReply      # raises ProxyError
    def budget_view(self, run_key: str) -> dict[str, int]                        # cap/spent/remaining/calls
```
`chat` does, in order: authenticate run (open, not closed) → classify conversation → fail-closed tool-surface check
(any `ToolSpec.name` outside `spec.allowed_tools` → `tool_surface_violation`) → per-run FIFO lock → grant → if
`remaining < terminal_threshold` and root and no grace yet → TERMINAL path; else render (Jinja2, §4) → `$` reserve
(`CollectionLedger.reserve`, attempt id `f"{run_key}:g{spec.generation}:{call_no}:{phase}:1"`) → `mark_sent` →
`Upstream.complete` → settle → split reasoning/visible → forced close (root only, once) → parse tool calls →
reasoning cache insert → journal → `RunRecord` update (persisted every call) → reply. Sub-agent calls that hit
their grant return truncated visible text with `finish_reason="stop"` and `forced_close_kind="truncated_subagent"`.
Duplicate protection: if the client disconnects after `mark_sent`, the upstream call is completed and settled and
the reply cached by `rendered_prompt_sha256`; an identical re-request is served from the cache (`duplicate_served`).
After a grace or after `remaining < 64` on a non-root conversation every call raises `budget_exhausted`.

## 4. Rendering and parsing (WP-A owns `dynorch/template.py`, `dynorch/toolcalls.py`)

- `render_chat(messages: Sequence[Message], *, tools: Sequence[ToolSpec], reasoning_effort: str,
  enable_thinking: bool = True, template: str = "qwen3.8-27b") -> str` — Jinja2 3.1.6
  `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)` over the pinned template
  (`api/forced.py::template_path`, sha verified by `template_sha256`), with HF-compatible globals/filters exactly as
  `transformers.utils.chat_template_utils._compile_jinja_template`: `tojson(x, ensure_ascii=False, indent=None,
  separators=None, sort_keys=False)` (plain `json.dumps`, NO HTML escaping), `raise_exception`, `strftime_now`.
  Tools are passed as `[{"type":"function","function":{name,description,parameters}}]`; `ToolCall.arguments` must be a
  dict when rendered (a raw string is re-emitted verbatim inside the assistant content instead of a `<tool_call>`
  block). Parity requirements (tests in WP-A): `render_chat([system,user])` == `api/forced.py::render_chat_prompt`
  byte-for-byte on the golden cases and all 96 main prompts; on tools / tool_calls / tool-response / empty-reasoning
  fixtures the output equals `transformers.utils.chat_template_utils.render_jinja_template(...)` run under
  `/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python3.11` (fixture goldens generated once by
  `tests/unit/dynorch/gen_oracle_goldens.py` and committed; the test asserts equality and that `<` never
  appears).
- `parse_tool_calls(text: str, tools: Sequence[ToolSpec], *, call_no: int) -> ToolParse` with
  `ToolParse(content: str, calls: tuple[ToolCall, ...], status: "none"|"ok"|"failed", reason: str | None)`.
  Grammar over the visible text: `<tool_call>` WS `<function=NAME>` WS (`<parameter=P>` "\n" VALUE "\n</parameter>" WS)*
  `</function>` WS `</tool_call>`; matched spans must be contiguous up to trailing whitespace; parameter typing from
  `parameters.properties[P].type` (string/absent → raw; else `json.loads`, fallback raw + note). Any grammar
  violation → `status="failed"`, whole text returned as content. Ids: `f"call_{call_no}_{i}"`.
- Reasoning cache: key 1 = the proxy-minted tool-call ids present in the assistant message (`tool_calls[].id` /
  echoed `tool_call_id`s), key 2 = sha256 of normalised content (strip a leading re-sent reasoning block and the
  literal `[Tool execution interrupted by user]`); explicit `reasoning_content` from the harness wins; a miss renders
  an empty think block and increments `reinjection_misses`.

## 5. Budget, upstream, journal (WP-A owns `budget.py`, `upstream.py`, `journal.py`, `records.py`, `config.py`)

```python
class TokenBudget:      # per run; asyncio.Lock per run_key enforces one in-flight upstream call
    def __init__(self, cap: int, *, terminal_threshold: int, grace: int, max_calls: int, max_prompt_tokens: int)
    async def acquire(self) -> "Grant"          # FIFO; records queue_wait_seconds
    def settle(self, grant: "Grant", completion_tokens: int, prompt_tokens: int) -> None   # never releases a sent grant
class Upstream:
    def __init__(self, client: ForcedClient, locks: Mapping[str, EndpointLock], sampling: Mapping[str, Any],
                 in_flight: Mapping[str, int], rps: Mapping[str, int], ledger: CollectionLedger) -> None
    async def complete(self, *, shard_id: str, prompt: str, max_tokens: int, attempt_id: str,
                       coordinates: Mapping[str, Any]) -> UpstreamResult
```
`UpstreamResult(reasoning: str, text: str, closed: bool, finish_reason: str | None, usage: Usage | None,
generation_id: str | None, cache_status: str | None, latency_seconds: float, transport_status: TransportStatus,
http_status: int | None, raw: dict)`. Body builder: `api/request.py::build_capped_body(prompt, lock, sampling, *,
max_tokens: int) -> dict` (new public function in WP-A: `_forced_common` keys minus `logit_bias` + `_forced_tail`;
must pass `api/fake.py::completions_contract_violations`). Ambiguous outcomes (`AMBIGUOUS_TIMEOUT`, 504/524) settle
the full grant in tokens, `charge_max` in dollars, and raise `upstream_failed`.
Journal: `<run_root>/raw/<arm>/<rung>/<repeat>/<item_slug>/a<attempt>/call_<nnnn>.json.gz` via `storage/_atomic.write_once`,
payload `{"layout":"dynorch/v1", run_key, call_no, phases:[{phase, request_body, response_status, response_headers,
response_body_text, latency_seconds, transport_status, generation_id}], harness_request, rendered_prompt_sha256,
rendered_chars, reply, tool_parse, budget:{granted, realized, remaining_after}}` with headers/text redacted.
`RunStore`: `<run_root>/runs/<arm>/<rung>/<repeat>/<item_slug>.json` (`replace_atomic`), `scan()`, `is_terminal`.

```python
@dataclass(frozen=True)
class RunSpec:
    run_key: str; arm: str; harness: str; item_id: str; benchmark: str; stratum: str; shard_id: str; endpoint_tag: str
    rung: int; cap_tokens: int; repeat: int; attempt: int; generation: int; partition: str      # partition: canary|pilot|main|ext_e0…
    allowed_tools: tuple[str, ...]; spawn_tools: tuple[str, ...]; root_call_ceiling: int | None
    sampling_hash: str; template_sha256: str; prompt_version: str; proxy_version: str; harness_version: str
@dataclass
class CallRecord:
    call_no: int; conversation_key: str; conversation_class: str      # root|solver|other
    requested_max_tokens: int | None; granted_max_tokens: int; queue_wait_seconds: float
    prompt_tokens: int | None; completion_tokens: int | None; reasoning_tokens: int | None; cached_prompt_tokens: int | None
    reasoning_chars: int; visible_chars: int; forced_close_kind: str; grace_tokens: int
    finish_reason_upstream: str | None; finish_reason_returned: str; tool_calls: int; spawn_calls: int
    tool_parse: str; reinjection: str          # hit|miss|explicit|n/a
    reasoning_effort_requested: str | None; cost_usd: str; generation_ids: tuple[str, ...]
    latency_seconds: float; transport_status: str; returned_status: int; stream: bool; wire: str
    cache_status: str | None; duplicate_served: bool; ignored_body_keys: tuple[str, ...]
@dataclass
class RunRecord:
    spec: RunSpec; status: str        # planned|running|done|failed|timeout|transport_failed|refused_before_start
    calls: int; conversations: int; spawns: int; calls_by_class: dict[str, int]
    completion_total: int; reasoning_total: int | None; prompt_total: int; grace_tokens: int; overdraft_tokens: int
    forced_close_kind: str; exhausted: bool; refused_calls: int; guard_triggered: bool
    tool_parse_failures: int; reinjection_hits: int; reinjection_misses: int; duplicates_served: int
    final_message: str | None; final_source: str      # proxy_root|none
    cli_result: str | None; cli_result_mismatch: bool | None
    parse_status: str | None; final_answer: str | None; grade_status: str | None; correct: bool | None; grade_detail: str | None
    wall_seconds: float; harness_exit_code: int | None; harness_reported: dict[str, Any]
    cost_usd: str; started_at: str; finished_at: str | None; error: str | None; rerun_reason: str | None
    calls_detail: list[CallRecord]
```

## 6. Wire layer and runner (WP-B owns)

- `dynorch/httpd.py`: `asyncio.start_server` + `h11`; routes under `/r/<run_key>/v1/`: `POST chat/completions`,
  `POST responses`, `GET models`, `GET budget`; `GET /healthz`. Bearer must equal the path run_key. Streaming
  replies send headers immediately and an SSE comment `: dynorch keepalive\n\n` every `keepalive_seconds`.
- `dynorch/openai_format.py`: `parse_chat_body(body: dict) -> ChatRequest`; `chat_reply_json(reply) -> dict`;
  `chat_reply_sse(reply) -> Iterable[bytes]` (chunks: role, reasoning_content, content, tool_calls per call with
  `index`, finish, a usage chunk ALWAYS, `[DONE]`); `error_json(err: ProxyError) -> (status, dict)`.
- `dynorch/responses_format.py`: `parse_responses_body(body) -> ChatRequest` (instructions → system; `message` items
  with `input_text`/`output_text`; `function_call` → assistant `ToolCall`; `function_call_output` → tool message with
  `tool_call_id = call_id`; `reasoning` items → `reasoning_content` when present; `tools[]` flattened, namespace kept on
  `ToolSpec.namespace`; unsupported tool types → `bad_request`); `responses_sse(reply, *, response_id: str) ->
  Iterable[bytes]` emitting `response.created`, `response.output_item.done` per item (`reasoning` with
  `summary: []` and `content:[{type:"reasoning_text",text}]`, `message` with `output_text`, `function_call` with
  `call_id`, `name`, `arguments` as a JSON string, `namespace` when the tool had one), `response.completed` with a
  string `id` and `usage {input_tokens, output_tokens, total_tokens, input_tokens_details:{cached_tokens},
  output_tokens_details:{reasoning_tokens}}`; errors as `response.failed {response:{error:{code,message}}}` with
  codes `rate_limit_exceeded` (retryable upstream), `context_length_exceeded`, else `server_error`.
  Parity test: the same scripted conversation through chat and Responses produces identical `CallRecord`s.
- `dynorch/plan.py`: `plan_runs(cfg, *, stage: str, arms: Sequence[str], rungs, repeats, items) -> list[RunSpec]`;
  `run_key = sha256(f"{study_id}|{partition}|{arm}|{item_id}|{rung}|{repeat}|{attempt}")[:24]` (§10.22); order: seeded item interleave,
  rungs in `cell_rung_order` inside each item.
- `dynorch/runner.py`: `Runner(core, adapters, settings).run_stage(stage, *, resume)`: subprocess pool, per-run work
  dir `<scratch>/work/<arm>/<rung>/<repeat>/<item_slug>/a<attempt>/{home,state,ws,tmp}`, env scrub (only
  `OPENAI_BASE_URL`, `OPENAI_API_KEY=<run_key>` plus harness-specific vars; every `OPENROUTER_*`/`ANTHROPIC_*`/other
  `OPENAI_*` removed), `launch.json`, wall timeout `base + per_rung·k`, SIGTERM→SIGKILL on the process group,
  Slurm drain trap (SIGTERM: stop admitting, wait ≤ 800 s, exit 0), rerun rule (§1 of the plan's frozen table),
  then `grade.finalize_run(record, task, private)`.
- `dynorch/grade.py`: `candidate_from_text(text: str, task: PublicTask, *, node_id: str) -> Candidate` (rules of
  `extraction/final.py::extract_candidate` for text kinds via `_vendor/extraction_tags.strict_single_pair`; parity
  test builds a minimal `GenerationResult`/`GenerationSpec` and compares) and `grade_text` from
  `grading/text/grade.py`.
- `dynorch/harness/base.py`:
```python
@dataclass(frozen=True) class LaunchPlan: argv: tuple[str, ...]; env: dict[str, str]; cwd: Path; stdin: bytes | None
@dataclass class HarnessResult: exit_code: int | None; timed_out: bool; cli_result: str | None; reported: dict[str, Any]
class HarnessAdapter(Protocol):
    name: str
    def version(self) -> str
    def prepare(self, work: Path, spec: RunSpec, proxy_base_url: str, system_text: str, subagent_text: str, task_text: str) -> None
    def build_command(self, work: Path, spec: RunSpec, proxy_base_url: str) -> LaunchPlan
    def parse_result(self, stdout: bytes, stderr: bytes, work: Path) -> HarnessResult
```
Adapters: `null.py` (H0: one non-streaming chat request in-process), `fake.py` (+ `tests/dynorch/fake_harness_cli.py`),
`openclaude.py`, `codex.py`, `openclaw.py` (fallback). Flags/env verbatim from the plan §6.

## 7. Tables, forecast, report (WP-C owns `dynorch/tables.py`, `dynorch/forecast.py`, `docs/dynorch_report/`)

- `dynorch_runs.parquet` (one row per RunRecord incl. attempts), `dynorch_calls.parquet`.
- `policy_outcomes_dynorch.parquet` with EXACTLY the sealed 36 columns and dtypes of
  `forced_budget_v2/artifacts/merged/tables/policy_outcomes.parquet`: `partition_id, policy_id, item_id, benchmark,
  subtask, stratum, wave, panels, name, view, family, team_size, total_cycles, peer_rounds, reasoning_budget,
  selector_name, selector_version, n_pool, n_required, n_required_terminal, n_required_no_output, n_pool_valid,
  n_pool_correct, n_pool_unknown, coverage, input_coverage, selection_status, selection_batch, selector_concrete,
  selected_candidate_id, selected_in_pool, sealed, delivered_state, y, shard_id, endpoint_tag`; values:
  `partition_id="main"` (extensions `ext_e0`…), `policy_id=f"main/{item_id}/{name}/current"`, `name=f"{arm}/{rung}"`
  (repeat 2: `f"{arm}/{rung}/r2"`), `view="current"`, `family="harness"`, `team_size=1`, `total_cycles=1`,
  `peer_rounds=0`, `reasoning_budget=cap`, `selector_name="last_message"`, `selector_version="dynorch-v1"`,
  `selector_concrete="LAST_ANSWER"`, `n_pool=1`, `y ∈ {1.0, 0.0, NaN}`, `delivered_state ∈ {delivered, op_no_output,
  unknown, pending}`; one row per planned cell from `plan_runs` BEFORE the stage starts, finalised in place.
- `policy_costs_dynorch.parquet` with the sealed 39 columns: `policy_id, partition_id, item_id, name, view, family,
  nominal_generations, nominal_generations_expected, n_required_terminal, requested_reasoning_tokens,
  requested_completion_cap_tokens, prompt_tokens, completion_tokens, reasoning_tokens, cached_prompt_tokens,
  realized_tokens, success_cost_usd, retry_cost_usd, total_charged_usd, standardized_uncached_cost_usd,
  public_check_cpu_seconds, selection_cpu_seconds, private_grading_cpu_seconds, total_latency_seconds,
  critical_path_latency_seconds, answer_tokens, upstream_inference_cost_usd, critical_path_generations,
  results_missing, transport_ok_nodes, pool_size, benchmark, subtask, stratum, y, delivered_state, cost_source,
  shard_id, endpoint_tag` (`completion_tokens = completion_total + grace`, `realized_tokens = completion_tokens`,
  `requested_reasoning_tokens = cap`, `standardized_uncached_cost_usd` via `analysis/costs.standardized_cost`,
  `cost_source="dynorch_ledger"`).
- `merged_outcomes(base_merged_dir, dynorch_run_dir) -> pd.DataFrame` (in-memory concat; sealed tables untouched);
  `docs/dynorch_report/core.py::Frame` subclasses `docs/forced_v2_report/core.py::Frame`, `families = LADDER_FAMILIES +
  ["harness"]`, `SEED = 20260923`, one `build_population` per contrast; regression test: `acc("deep/16")` = 0.7917 and
  `acc("sample/16")` = 0.4167 with harness columns present.
- `forecast.py`: reproduces the plan §8 table from the config and, given `dynorch_runs.parquet`, the Gate 2 re-fit
  (+25 % contingency).

## 8. Ops (WP-D owns)

`dynorch/cli.py` verbs registered in `agent_allocation/cli.py::VERBS`: `dynorch-plan`, `dynorch-run` (`--stage
canary|pilot|main|ext_e0|…`, `--allow-paid`, `--max-usd`, `--resume`, `--account-auto-reload`, `--arms`, `--rungs`),
`dynorch-tables`, `dynorch-gates --gate 0|1|2|4`, `dynorch-reconcile`, `dynorch-doctor`; `dynorch/gates.py` (criteria
of plan §9, `GateResult` records via `gates.write_gate`), `dynorch/reconcile.py` (`/key`, `/credits`, `/generation`
sample), `dynorch/endpoint_rule.py` (plan §4), `slurm/dynorch_v1.sbatch` (`--partition=pi_lhtsai,mit_normal
--time=12:00:00 --signal=B:TERM@900 --cpus-per-task=8 --mem=32G`, VERBs gate0|canary|pilot|run|finalize),
`slurm/launch_dynorch_v1.sh`, `tools/install_codex.sh`, `docs/dynorch/RUNBOOK.md`, doctor checks (`/home` files <
990k, key mode 0600, no `sk-or-` in the tree, no state path under `/home`).

## 9. Ownership (files a WP may create/edit)

- **WP-A**: `dynorch/{__init__,config,records,template,toolcalls,budget,upstream,core,journal}.py`,
  `api/request.py` (add `build_capped_body` only), `api/fake.py` (additive: tool-call content replies, `Fault.NO_USAGE`,
  long `delay_seconds` support if missing), `tests/unit/dynorch/{__init__,conftest,test_template,test_toolcalls,
  test_budget,test_core_flow,test_journal_resume,test_upstream,gen_oracle_goldens}.py`, `tests/fixtures/dynorch/`.
- **WP-B**: `dynorch/{openai_format,responses_format,httpd,plan,runner,grade}.py`, `dynorch/harness/*`,
  `tests/dynorch/fake_harness_cli.py`, `tests/unit/dynorch/{test_openai_format,test_responses_format,test_httpd,
  test_plan,test_grade_parity,test_adapters,test_env_scrub}.py`, `tests/integration/test_dynorch_*.py`.
- **WP-C**: `dynorch/{tables,forecast}.py`, `docs/dynorch_report/*`, `tests/unit/dynorch/{test_tables,test_forecast}.py`.
- **WP-D**: `dynorch/{cli,gates,reconcile,endpoint_rule,doctor}.py`, `agent_allocation/cli.py` (VERBS only),
  `slurm/dynorch_v1.sbatch`, `slurm/launch_dynorch_v1.sh`, `tools/install_codex.sh`, `docs/dynorch/RUNBOOK.md`,
  `tests/unit/dynorch/{test_gates,test_cli,test_endpoint_rule}.py`.
- **Lead only**: this file, `configs/dynorch_v1.json`, `prompts/system_dynorch.txt`, `prompts/subagent_dynorch.txt`,
  `config.py` preset registration, `pyproject.toml`/`uv.lock`.
Until WP-A lands, WP-B codes against `ProxyCore` through a stub implementing §3's signatures
(`tests/unit/dynorch/stubs.py`, WP-B owned, deleted at merge).


## 10. Integration amendments (lead; evidence in docs/dynorch/harness_wire_observations.md and tests/fixtures/dynorch/captures/)

WP-E ran both real CLIs against a throwaway stub on 2026-09-23 ($0). These facts override the plan text where they
disagree; each is VERIFIED by a committed capture.

**10.1 Conversation classes** (§1 amended above). OpenClaude prepends `"You are OpenClaude, built on the Claude Agent
SDK.\n\n"` to every system prompt unconditionally (no flag removes it) and appends `"\n\nNotes:…<env>…You are powered
by the model …"` to a sub-agent's prompt; Codex sub-agents carry the root's `instructions` verbatim and are identified
only by the `x-openai-subagent: collab_spawn` header. `ChatRequest.subagent: bool` (WP-A, additive) is set by the wire
layer from that header. Both harnesses put a dated block (`<system-reminder>` / `<environment_context>`) before the
task text, so conversation keys must not hash the raw first user turn.

**10.2 Gate 0 criterion "messages[0] byte-identical to SYSTEM.md" is false as written.** The check is
`messages[0].content == OPENCLAUDE_PREFIX + system_text.rstrip("\n")`; the prefix is recorded in
`locks/harness_prompts.json`. Codex's `instructions == system_text.rstrip("\n")` exactly (it does NOT append its
multi-agent guidance there; that lives in the tool descriptions).

**10.3 Tool surfaces (frozen in `configs/dynorch_v1.json`).** H1 offers exactly `{Agent}` with the plan's flags (no
`--bare`); the solver's requests carry **no `tools` key at all** (a missing key is an empty list, not a violation) and
a `temperature: 1` field (an `ignored_body_keys` entry). H3 with the plan's config also offers `request_user_input`
and three goal tools and lacks `update_plan`; adding `features.goals = false`,
`[tools.experimental_request_user_input] enabled = false` and `[tools.update_plan] enabled = true` yields exactly
`update_plan` + the `multi_agent_v1` namespace. Bool-valued `tools.*` keys are rejected by `--strict-config`.

**10.4 Codex wire specifics.** `function_call` items for namespaced tools MUST carry `"namespace": "multi_agent_v1"`
or the call is rejected as `unsupported call`. An `output_item.done` emitted before a failed stream is kept in the
harness's history, so the proxy buffers and emits items only when `response.completed` is guaranteed.
`rate_limit_exceeded` is retried (a duplicate-served cache hit is the right answer); a missing `response.completed`
costs 2 retries then `turn.failed`. Codex echoes reasoning items back verbatim (explicit re-injection); OpenClaude
never re-sends `reasoning_content`, so its re-injection relies on the proxy cache keyed by the echoed tool-call id
(ids are echoed exactly). Codex sends `parallel_tool_calls: true`, `reasoning: {}`, `store: false`,
`include: ["reasoning.encrypted_content"]`, no `text`, no `max_output_tokens`.

**10.5 Adapter result parsing.** OpenClaude's stdout keeps `subtype: "success"` even on a 402: success is
`is_error == false`. `total_cost_usd` comes from an unknown-model price table and is ignored. Codex: the `-o` file or
the last `agent_message` item; `turn.failed` is a failure; the "Model metadata … not found" error item is normal noise.

**10.6 Environment.** Add `CLAUDE_CODE_PROVIDER_PROFILE_ENV_APPLIED=1` to the OpenClaude env (silences a spurious
Codex-profile warning). `agentModels`/`agentRouting` in `.openclaude.json` are optional; keep `hasCompletedOnboarding`
and `bypassPermissionsModeAccepted`. Codex writes ≈60 skill files and 4 sqlite files into `CODEX_HOME` on first start
(≈18 s); OpenClaude writes `cfg/projects/<ws>/<session>/subagents/*.meta.json` per spawn even with
`--no-session-persistence`. Both stay inside the per-run work dir; nothing under `/home`.

**10.7 Known, accepted input overheads.** Codex's `<skills_instructions>` developer block (≈700 tokens) cannot be
removed by config and is inert (no file tools). OpenClaude's Agent tool description (6,331 chars) lists built-in
agent types, so the system prompt must require `subagent_type: "solver"`; the proxy flags any other value seen in a
parsed Agent call. Both are `S_h` covariates, paid as input, outside C.

## 10 (continued). Integration amendments at the merge (lead, 2026-09-23; evidence: the merged test suite)

The five branches merged without textual conflicts; the seams below were found by running the integrated code
(`tests/unit/dynorch`, `tests/integration/test_dynorch_e2e_fake_harness.py`, the rehearsal launch in
`tests/unit/dynorch/test_launch_stage.py`). Where the plan text, a request file and the code disagreed, the contract
and the plan won; each item names the resolution and the test that pins it.

**10.8 `ProxyError` signature.** WP-A shipped `ProxyError(code, message, *, status=None, retryable=None, detail=None)`;
the contract (§2) and WP-B's wire layer use `ProxyError(status, code, message, retryable)`. Resolution: the contract
signature, positional `status` first; `ProxyError.of(code, message)` raises at the code's frozen status for the core;
codes `not_found` (404, an unknown route) and `internal_error` (500, a proxy bug answered loudly) are added to the
frozen map for the wire layer; `retryable` defaults to the code's frozen value when omitted. Evidence:
`test_responses_format.py::test_responses_failed_codes`, `test_httpd.py::test_bearer_must_match_run_key`.

**10.9 Parse / grade vocabulary.** `RunRecord.parse_status` and `grade_status` carry the codec VALUES the runner writes
(`contracts.ParseStatus.VALID.value == "valid"`, `GradeStatus.EVALUATOR_ERROR.value == "evaluator_error"`, ...), the
sealed ladder's own spelling. WP-C's tables and WP-D's gates compared against the enum NAMES (`"VALID"`, `"error"`);
both now import the values (`tables.PARSE_VALID`, `gates.PARSE_VALID` / `gates.GRADE_EVALUATOR_ERROR`). Evidence:
`test_dynorch_e2e_fake_harness.py` (`parse_status == "valid"` on real records), `test_tables.py::test_outcome_of`.

**10.10 Conversation classification is by containment (§1, §10.1 made precise).** `solver` iff `ChatRequest.subagent`
or the sub-agent prompt (`subagent_dynorch.txt.rstrip("\n")`) is CONTAINED in the system text (OpenClaude prepends its
line and appends `Notes:...<env>`); `root` for the run's first non-solver conversation and for any later conversation
whose system text is that one's or contains the root prompt; else `other`. Conversation keys hash the system text and
the first user message with OpenClaude's `<system-reminder>...</system-reminder>` / Codex's
`<environment_context>...</environment_context>` block removed (a Codex child's key falls back to the first non-empty
user item after the environment block). Evidence: `test_core_flow.py` classification tests; the e2e's
`calls_by_class == {"root": 2, "solver": 1}` on a scripted delegation.

**10.11 `ProxyCore.budget_view()` carries `exhausted` (0/1) and `ProxyCore.proxy_version`.** R-B3 / R-B6 granted:
`exhausted` is 1 once the run's grace was used (every later call is a 402 whatever `remaining` says); the runner's
watchdog honours it. `plan_runs` stamps `proxy_version` from `core.proxy_version` (= `config.PROXY_VERSION`,
`dynorch-proxy-1.0.0`); `open_run` refuses a spec stamped with another version.

**10.12 `forced_close_kind = "ceiling_close"` (R-A4) is CONFIRMED for E0.** An E0 root call bound by its ceiling
`max(A, C/4)` rather than the cap is closed with the ANSWER continuation charged INSIDE C (no grace),
`CallRecord.forced_close_kind = "ceiling_close"`; `RunRecord.forced_close_kind` keeps the B5 set. Core arms never
produce it. WP-C's tables carry the value through unchanged (no enum check on `forced_close_kind`).

**10.13 The canary's k=16 ceiling probe** runs on the canary items (`dynorch.canary.items`: aime, gpqa_diamond dev
item 0), not the first two benchmarks in `BENCHMARKS` order: `plan.plan_stage` and `cli.enumerate_cells` are now
cell-for-cell identical for every stage (`test_launch_stage.py::test_plan_stage_and_enumerate_cells_agree_cell_for_cell`,
10 / 144 / 1,728 / 384 / 1,152 / 384 / 192 cells).

**10.14 `runner.launch_stage` and `runner.build_stack`** (requests/wpD.md item 1) exist with the requested signature.
`build_stack(cfg, run_root, *, client: ForcedClient, locks, ceiling_usd, active_ceiling_usd)` is the ONE place the
stack is assembled (settings -> `LedgerJournal`/`CollectionLedger` -> `resolve_inherited` FIRST (R-A7) ->
`CallJournal`/`RunStore` -> `Upstream` -> `ProxyCore`); the e2e and `dynorch-run` both use it. `launch_stage` reads
`locks/openrouter_endpoints.json` and applies `locks/provider_map.json` to every item's shard (plan §4: after a switch
every item runs on the passing shard; `forced_v2_shard` stays the covariate), binds `httpd` on `127.0.0.1:0`, calls
`on_bound(host, port)` before the first harness, refreshes the active ceiling from `/key` every 5 minutes (a 402 from
the account endpoints drains the stage and is reported as `upstream_402`), drains on SIGTERM and on
`control/STOP` / `control/DRAIN`, and returns `{runs, executed, skipped, reruns, drained, done, failed, timeout,
transport_failed, refused_before_start, upstream_402, settled_usd, in_flight_usd, unresolved_usd, inherited}`.
`cli.make_client(cfg)` returns a `ForcedClient` (the fake transport in rehearsal) so one client serves the endpoint
rule, the proxy's upstream and the reconciliation. Evidence: `test_launch_stage.py::test_dynorch_run_canary_h0_in_rehearsal_runs_the_real_stack`
(the verb, in rehearsal: endpoint rule frozen, proxy announced, two H0 runs through the real proxy, tables built,
reconciliation `clean`).

**10.15 `tables.build_tables(run_root, cfg, *, stage=None)`** reads EVERY pre-registered manifest
(`manifests/dynorch_plan_<stage>.jsonl`) when `stage` is None, cells keyed by `(partition, arm, item, rung, repeat)`
(a cell planned in two manifests raises); `dynorch-tables [--stage S]` and the post-stage build in `dynorch-run` call
it that way (WP-D's assumed `build_tables(cfg, paths)` does not exist). WP-D's Gate 1/2/4 read `dynorch_runs.partition`
(= the stage) and `policy_outcomes_dynorch.partition_id` (= the stage). A tables failure after a stage is recorded in
`status.json` (`tables: "error: ..."`) and the reconciliation still runs; `dynorch-tables` re-builds.

**10.16 `forecast.gate2_refit(cfg, runs, *, spent_usd)`** (requests/wpD.md item 5) returns
`{core_forecast_usd, spent_usd, soft_ceiling_usd, room_usd, cuts_applied, fits, contingency_share, per_cell}`: per
(arm, rung) the pilot-measured re-fit where the pilot ran that cell, else the §8 ceiling model; cuts C1-C3 applied in
order only while the forecast exceeds `soft_admission_ceiling_usd - spent`; C4 is never applied.

**10.17 Test doubles.** The pre-merge records shim and `StubRunStore` are deleted; `tests/unit/dynorch/stubs.py`
keeps `StubProxyCore` only as the wire layer's scripted double for `test_httpd` (delays, error frames, disconnects).
Everything that touches the budget, the ledger, rendering or the delivered answer is tested on the real `ProxyCore`
over `api/fake.py`: `test_core_flow`, the chat/Responses parity test and the fake-harness e2e (`-m slow` selects the
330 s / 660 s cases).

**10.18 Not amended, recorded for Gate 0.** Real-CLI smoke tests (plan §9 Gate 0 lists) against the real proxy +
fake upstream are NOT in the tree; WP-E's captures were taken against a throwaway stub. `gates/gate0_smoke.json`
(`{passed, failures}`) is therefore absent and `dynorch-gates --gate 0` returns stop_notify until the smoke is run.

**10.19 B9 "headers at once" → "headers within `refusal_grace_seconds`" (WP-E item 14, lead ACCEPTED).** The streaming
path waits up to `refusal_grace_seconds = min(2 s, keepalive_seconds)` before committing the response headers, so a
refusal decided before any upstream work (budget 402, tool_surface 400, bad bearer 401) carries its real HTTP status
instead of a 200 followed by an in-stream error frame. Evidence: OpenClaude retried an in-stream 402 but ends the run
immediately on a real 402. Keepalive cadence and the "one upstream call per request" rule are unchanged, and the
330 s / 660 s stall cases still pass for both harnesses.

**10.20 Codex retries an HTTP 402 twice per conversation (WP-E item 15, lead ACCEPTED as observed behaviour).**
Codex 0.156.1 reads 402 as "unexpected status", re-sends the identical request `stream_max_retries` (2) times and then
reports `turn.failed`, all in-process in under a second with no upstream call and no cost. Consequences, pre-registered
here: the plan §9 Gate 0/Gate 1 phrase "≤ 1 further request after the first 402" does NOT apply to H3 — the pinned
criterion for Codex is "every request after the first 402 is itself refused with 402, no upstream call, no cost, and
the run ends within the wall timeout"; and `RunRecord.refused_calls` counts up to 3 per exhausted Codex conversation.
`refused_calls` is a diagnostic, never a compute or cost quantity, so no budget or accounting rule changes.

**10.21 Codex leaves `arg0` symlinks on a failed exit (WP-E item 16, lead ACCEPTED).** A failed Codex run leaves about
four symlinks to its own binary under `state/codex/tmp/arg0/`. They live in the per-run scratch work dir, cost four
inodes each on `/orcd/data` (not `/home`), and are harmless. Any scan of the scratch tree (the doctor's key-shaped
string scan, a Gate scan) must not follow symlinks and must not read the linked binary.

## 11. Review fix pass (2026-09-24; adversarial review of the merged code, 3 blockers / 14 majors / 9 minors)

Two reviewers rejected the merged code for paid use. Every finding is answered in `docs/dynorch/REVIEW_LOG.md`
(one line per issue with its commit). The items below CHANGE an interface or a frozen rule of §1–§10 and are
therefore recorded here; everything else was a bug fixed behind an unchanged interface.

**§10.22 The partition is part of a run's identity (blocker).** `run_key = sha256(f"{study_id}|{partition}|{arm}|
{item_id}|{rung}|{repeat}|{attempt}")[:24]` (§6 amended), and the partition is a path element of
`RunStore.path` / `superseded_path` (`runs/<partition>/<arm>/…`), `CallJournal.run_dir` (`raw/<partition>/<arm>/…`)
and the runner's work dir (`work/<partition>/<arm>/…`); the runner's cell key is
`(partition, arm, item_id, rung, repeat)`. Without it the canary's four k=16 ceiling-probe cells and the pilot's
cells for the same items shared a run_key, a record file, a raw directory and the ledger attempt ids.

**§10.23 B9 duplicate protection is a RETRY cache (blocker).** The reply cache is keyed
`(conversation_key, rendered_prompt_sha256)` and holds ONLY calls the wire layer could not deliver. `ProxyCore`
gains `mark_delivered(run_key, call_no)` / `mark_undelivered(run_key, call_no)`; `httpd` calls exactly one of them
for every reply (after the write, or when it parks a call whose client went away). Two sub-agents with identical
prompts are two conversations and each gets its own generation. B7's delivered answer is the last root reply the
wire layer actually handed to the client (`close_run` and `ReplyLog.delivered` use the same signal).

**§10.24 The B4 watchdog clock and the parked calls (blocker + major).** `TokenBudget.view()` /
`ProxyCore.budget_view()` carry `in_flight`, `replies_served` and `idle_seconds`; the runner's exhaustion watchdog
never fires while a call of the run is in flight and counts from the last SERVED reply. `ProxyServer` gains
`wait_for_run(run_key, timeout)` and `wait_idle(timeout)`; `close()` drains the parked (shielded) calls before
cancelling anything, and `Upstream.complete` charges the maximum when a cancellation escapes after `mark_sent`.
The runner awaits the run's in-flight call before `close_run`.

**§10.25 Cap-bound is `length` at the grant (major).** B3's forced close applies when
`finish_reason == "length"` and `completion_tokens >= granted max_tokens`. Any other non-`stop` finish, or `length`
below the grant, is forced_v2's `forced_think_short`: the reported tokens are settled, the call is journaled and
`upstream_failed` is raised (the rerun rule decides). Every settled grant is journaled, whatever ended it.

**§10.26 Stage re-execution and the pre-registration (majors).** `Runner.run_specs` always consults the store: a
terminal cell is skipped; an unfinished or rerun-eligible one requires `--resume` (else `RunRefused`). The plan
manifest `manifests/dynorch_plan_<stage>.jsonl` is write-once per stage (a subset leaves it untouched; a spec it
does not pre-register raises). `tables.read_run_records` reads `runs/` AND `runs_superseded/`, so every paid rerun
is a row and `retry_cost_usd` is real. The drain stops admissions and terminates only the harnesses still running
after `drain_wait_seconds`. An upstream 402/401 drains the stage and never consumes one of the two reruns.

**§10.27 Codex's wall (major).** The adapter's `timeout` wrapper sits at `wall + sigkill_after +
timeout_margin_seconds` (default 60), so the runner's `600 + 120·k` timer is authoritative; a `timeout`-wrapped
exit (124, or 137 after its kill-after) is classified `timeout`, not `done`.

**10.28 The B3 grace carries no tool schemas and is returned as text (lead, 2026-09-24; canary finding).** Until
`1791e63` the mid-call and visible continuations reused the THINK prompt, tool definitions included, and parsed the
grace for tool calls. In the first paid canary three of eight harness runs spent their one grace on a tool call
(OpenClaude spawning a verifier, Codex calling `update_plan`), had no budget left to run it, and delivered no answer —
a scored zero for a mechanical reason, at exactly the budgets that matter. The plan's B3 already said "no tool schemas"
and "finish_reason: stop"; the code now matches it: the `mid_call` / `visible_continuation` continuation re-renders the
same messages with `tools=()` and its text is returned as content with `finish_reason = "stop"`, never as a tool call.
`ceiling_close` (E0) is not a final answer — the harness continues — so it keeps its tools and may delegate. The two
Gate 0 smoke cases `*/forced_close_then_402` pinned the old behaviour (they asserted a spawn emitted in the
continuation) and are rewritten to the rule; the canary's ten cells were run under the defect and are re-executed.

## 12. Second fix pass (2026-09-24; re-review of `afe3093..1791e63`, findings 2-13 in `rereview_findings.md`)

Answered line by line in `docs/dynorch/REVIEW_LOG.md` (second table). The items below change an interface or a rule.

**10.29 Reconciliation compares native tokens and waits for `/key` to settle (finding 7; needed before the pilot).**
(a) The proxy journals the provider's own `usage.completion_tokens`; `/generation` calls that
`native_tokens_completion`, while its `tokens_completion` is OpenRouter's normalised re-count and is not comparable
(canary: 4,323 native vs 2,388 normalised on one call at an identical cost). `reconcile.compare_call` sums
`native_tokens_completion` over the call's generation ids; a metadata document without it gives `tokens_match = None`
and a named problem, never a comparison on the normalised field. (b) `/key` usage lags the settle by tens of seconds
(canary: $0.1139 read the second the stage ended; exactly $0.253105 = the ledger within ~45 s). The after-snapshot is
taken by `reconcile.settled_after_snapshot`: one read; if the key delta matches the ledger delta within the stage
tolerance it is used, otherwise `/key` is re-read every 30 s until two consecutive reads agree (the key has settled)
or 300 s of waiting have passed. The report carries `key_settle = {polls, waited_s, matched, settled, ...}`; a
mismatch on an unsettled key says so in `problems`. Tolerances ($0.02 canary/pilot; $0.50 or 2 % later) are unchanged.

**10.30 A client that goes away is noticed at once; "delivered" requires a live client (findings 2 and 10).**
While a completion is being produced, `httpd` keeps a read pending on the connection (`_watch_eof`); EOF or a reset
sets the request's `client_gone` event immediately (not at the next 15 s keepalive write). `ProxyCore.chat(run_key,
request, *, client_gone=None)` (additive keyword; `ProxyCore.accepts_client_gone = True`, cores without it are called
as before): a reply produced while `client_gone` is set is marked undelivered and cached for B9 SYNCHRONOUSLY, with
no `await` between the settle that releases the run's FIFO lock and the return, so an identical retry already
queued on that lock is served the same reply (`duplicate_served`) instead of minting a second grant. A gone client
parks its call at once (streaming and non-streaming); the call is still finished, settled and journaled. A reply
counts as delivered only if `client_gone` was clear before the write and still clear one loop turn after the drain
(the first write after a FIN succeeds, so a successful write alone is not proof); otherwise it is undelivered (B9
cache, never B7's delivered answer). A client that half-closes its side while waiting is treated as gone.

**10.31 An account refusal decides a run only while it is the run's last upstream outcome (finding 3).**
`ProxyCore.budget_view()` carries `infrastructure_stop` (0/1): set when an upstream 402/401 (`UpstreamFailed.
infrastructure_stop`) ends a call, cleared by any later served reply (fresh or duplicate). The runner makes a run
`transport_failed` / `rerun_reason = "infrastructure_stop"` only when the flag is still set at close; a run that
recovered (a later reply, e.g. the root's answer after a sub-agent's 402) keeps the status the frozen rerun rule
gives it and its delivered answer, while `record.error` keeps the `infrastructure_stop: ...` text and the stage still
drains (Gate 3 stop rule; the reconciliation reports the 402). `_rerun_pending` / `_attempts_left` read only
`rerun_reason`, never the error text.

**10.32 A ledger refusal pauses admissions; pre-send refusals keep what was already paid (finding 4, minors 8-9).**
A run whose last proxy reply was `503 ledger_refused` is closed `transport_failed` with
`rerun_reason = "ledger_stop"` (new; in `RERUN_REASONS` and, with `infrastructure_stop`, in `runner.STOP_REASONS`):
the frozen rule keeps it rerun-eligible, but it is NOT rerun in-job, the stage drains, and `_rerun_pending` /
`_attempts_left` let `--resume` re-execute it without the `<= 2 reruns` cap; the cap counts MODEL attempts
(`Runner._reruns_used = attempt - 1 - earlier stop attempts of the cell`), so a stop never consumes a rerun although
the attempt number (and the run key) advances. `Runner(..., admission_check=)` (additive)
is consulted before each cell starts; `launch_stage` wires `runner.ledger_block_reason(ledger)` (`"unresolved_exposure"`
when the unresolved exposure is over `max_unresolved_spend_usd`, `"ceiling"` when nothing fits under the active
ceiling), and a reason drains the stage (plan §5 "pauses admissions"). `Upstream.complete` attaches the transport
attempts already made to a pre-send `ledger_refused` / `internal_error` (`prior_phases`, `prior_cost_usd`,
`prior_charged_max`; `upstream.with_prior`): on attempt 1 they are empty and the token grant is released as before;
otherwise the grant is settled at 0 tokens, keeping its call number, and the call is journaled with those phases and
dollars (`record.cost_usd` agrees with the ledger). `_terminal` treats `internal_error` like `ledger_refused` (call
number given back, error recorded, grace unused).

**10.33 The drain fits Slurm's 900 s; an exited harness is served nothing more (findings 5, 6, minor 13).**
`RunnerSettings.drain_wait_seconds = 540` (was 800; the plan's "wait <= 800 s" is an upper bound) and
`drain_total_seconds = 660`: `request_drain` fixes a deadline; in-flight harnesses get `drain_wait_seconds` to finish,
the survivors are terminated, every settle wait (`_await_settled`) is bounded by the deadline, and an attempt still
running at the deadline is cancelled and closed `failed` / `rerun_reason = "killed_by_drain"` (never left `running`).
After a drained stage `dynorch-run` bounds the `/key` settle wait at `cli.DRAINED_KEY_SETTLE_MAX_S = 60`; worst case
660 + 60 (server close) + 60 + 30 (`/generation`) s < 900 s. `ProxyCore.harness_exited(run_key)` (new; the runner calls
it the moment the harness process exits, on every path): a call still queued on the run's lock, or any later call, is
refused `409 run_closed` without minting a grant; a call already in flight is finished, settled and journaled. After
`close_run` the core persists a late update only through `RunStore.save_if_owned` (new): the cell's file while it holds
this run_key, else this attempt's `runs_superseded/` copy, else nowhere (the call journal keeps it) -- never over the
next attempt's record. `ReplyLog.last_event` ignores `run_closed` refusals (nobody was listening). `Runner._close_failed`
awaits the settle like the normal path.

**10.34 The stage manifest always pre-registers the FULL stage; one run_key formula (minors 11, 12).**
`plan.write_plan(specs, run_root, stage, *, registered=None)`: with `registered` (the stage's unfiltered `plan_stage`
list) a first launch writes the full cell set and the `specs` being run must be a subset of it; `Runner.run_specs(...,
registered=)`, `Runner.run_stage` and `launch_stage` always pass it, so a filtered first launch (`--arms H0`) no longer
shrinks the pre-registration (plan §1 "a policy_outcomes row exists for every planned cell before the run starts").
`tables.run_key_of(..., *, partition)` delegates to `plan.run_key_for` (the §10.22 formula).

**10.35 Superseding a stage execution (lead request; needed to re-execute the canary under §10.28).**
`dynorch-run ... --stage S --supersede "<reason>"` (with `--resume` as every VERB passes; `SUPERSEDE_REASON` in the
sbatch / launcher) calls `dynorch.supersede.supersede_stage(paths, S, reason=, scratch_root=)` before the launch. It
refuses (nothing moved) when the reason is empty, the stage has nothing to supersede, a record of S is `running`, or
a ledger attempt of S (`partition_id == S`) is still open. Otherwise it picks the new execution generation
`G = 1 + max(every generation in S's records, in S's ledger attempt ids, and the previous G)`, moves `runs/S`,
`runs_superseded/S`, `raw/S`, `state/{reconcile_S_before.json, reconciliation_S.json, stage_S_started}` and
`gates/*_S.json` to `superseded_executions/S/g<old>_<utc>/` beside `SUPERSEDED.json` (layout `dynorch/supersede/v1`:
reason, generations, run keys, ledger attempts and dollars, what moved), moves `<scratch>/work/S` to
`<scratch>/work_superseded/S/g<old>_<utc>/`, writes `state/stage_generations.json` and records
`live.record_decision(kind="dynorch_supersede", ...)`. `launch_stage` / `Runner.run_stage` stamp attempt-1 specs with
`supersede.execution_generation(run_root, S)` (`apply_generation`): the cells keep their run keys, record paths and
pre-registration, and every ledger attempt id `{run_key}:g{G}:...` is new. Never touched: the ledger journal (read
only), `locks/`, the stage manifest, any other stage. `tables.read_run_records` does not read
`superseded_executions/`, so a superseded execution is no row; its spend stays in the ledger.

**10.36 The `/key` dollar verdict waits for the key to converge on the ledger (lead, 2026-09-24; supersedes the §10.29
settle rule).** Both canaries showed `/key` usage lagging the ledger and posting in bursts: the re-run canary's key
read $0.152905 of the ledger's $0.2454825, stayed flat for more than 30 s, then reached it to the micro-dollar 13 s after
the recorded read. §10.29's "two reads 30 s apart agree" therefore declared a false mismatch, which would halt the
conductor after every stage. The ledger delta is the expected value, so `settled_after_snapshot` now keeps re-reading
while the key is below it (a pause is not settlement), confirms a match or an overshoot with one more read, and reports
`converged` (clean), `over` (the provider billed more than was metered: a real discrepancy) or `did_not_converge` (still
below after `KEY_SETTLE_MAX_S`, a mismatch). Unit tests never wait on the wall clock (an autouse fixture swaps the
module-level `_sleep`); the tests that pin the timing pass their own `sleep`.

**10.37 Owner waivers of the delegation criteria (lead, 2026-09-24; owner decision after Gate 1).** The re-run canary
showed neither harness delegating (8/8 harness runs were one model call, no sub-agents, no tool calls), which trips the
pre-registered stop-and-ask (`ceiling_probe_spawns`, 0/4) and leaves `codex_responses_tool_loop` with nothing live to
verify. The owner chose to run the core as planned: non-delegation is the finding. `dynorch-gates --owner-waive
<criterion> --waiver-text "<owner's words>"` stores a STANDING waiver in `state/owner_waivers.json` and a
`kind=owner_waiver` line in `run_decisions.jsonl`; every later evaluation of that gate applies it. Waivable criteria
are fixed in code (`gates.WAIVABLE`): gate 1 `ceiling_probe_spawns`, `codex_responses_tool_loop`; gate 2
`H1_spawn_share_k16`, `H3_spawn_share_k16`. Nothing else can be waived — money, accounting, grants, provider pinning,
answers, forecast and hygiene criteria stay binding — a `stop` decision is never waived, a partial waiver leaves the
gate stopped, and a waived criterion stays recorded as FAILED (its threshold is prefixed `WAIVED by owner`).
The standalone `dynorch-reconcile` command now runs its HTTP client inside one event loop (it used to crash on close
with "Event loop is closed" after writing a correct report).

**10.38 An upstream failure latches the run; the attempt is an infrastructure rerun (lead, 2026-09-24; main-stage
finding).** Parasail (and Mancer) sometimes break a generation mid-stream: an error envelope (`choices[0].error`
`{code: 502, "Stream interrupted: ..."}`, or `finish_reason: "error"` without an error object) or a read error. B2
handles that call correctly -- full grant settled in tokens, the maximum charged in dollars, `upstream_failed` (502)
-- but both harnesses RETRY a 502 (OpenClaude's `withRetry` with `CLAUDE_CODE_MAX_RETRIES=1`, Codex's
`request_max_retries` / `stream_max_retries`). The retry found `remaining < 64` and the root was served the B4
TERMINAL call (thinking disabled, "reply now with only your final answer"), so the run ended `done` with a valid,
no-thinking answer scored as a model outcome (main: H1/16 omni_math_hard/3907 and /1257, H1/25 aime/2026/23; H0
never retries, so its interrupted runs were correctly `transport_failed` -- the defect biased only the harness
arms, against the frozen rule "infrastructure failure is not a model failure"). The rule, pre-registered here:
(a) **The latch (core).** The first `UpstreamFailed` of a run -- ambiguous outcome, error envelope, 5xx after the
upstream retries, `forced_think_short`, an upstream 402/401, on a root or a sub-agent call, in the THINK, ANSWER or
TERMINAL phase -- latches the run. Every later call of the run (a harness retry, a call queued on the run's lock, a
root continuation after a failed sub-agent) is refused `502 upstream_failed` under the run's lock BEFORE any grant:
no reservation, no upstream call, no cost; `refused_calls` counts it. `budget_view()["upstream_latched"]` and
`ProxyCore.upstream_latch(run_key)` (call number, class, transport status, detail) expose it. A cancellation or a
pre-send ledger refusal is not an `UpstreamFailed` and does not latch.
(b) **The runner.** A latched run is `transport_failed` with `rerun_reason = "upstream_latched"` whatever the
harness did afterwards (retried into the latch, exited, timed out, was drained); the latch is kept in
`harness_reported["upstream_latch"]` and the core's error text in `error`. An account refusal keeps the §10.31
`infrastructure_stop` path (it drains the stage) -- with the latch it is always the run's last upstream outcome, so
§10.31's "a run that recovered keeps its answer" no longer arises (that answer was the B4 TERMINAL). `upstream_latched`
and `infra_invalidated` are INFRASTRUCTURE reruns (`runner.INFRA_RERUN_REASONS`): like §10.32's stops they consume no
model rerun (`_reruns_used` excludes them) and the attempt number and run_key advance, but unlike a stop they are
re-executed in-job, bounded separately by `RunnerSettings.max_infra_reruns = 3` (a provider that fails every call
then leaves the cell `transport_failed`, y = NaN). The rerun is a new run_key (new ledger attempt ids, a new journal
directory), a fresh `TokenBudget` and, on `--resume`, generation + 1. The latched attempt's dollars stay charged
(`retry_cost_usd`); its tokens stay journaled; it is never the cell's scored attempt (`grade.is_infra_failure` clears
its parse/grade fields, `tables.outcome_of` gives it y = NaN / `unknown`, and the rerun is the cell's current record).
H0's interrupted runs are the same `upstream_latched` reruns (the arms are treated alike).
(c) **Invalidation of records completed under the defect.** `allocation dynorch-invalidate --config CFG --run-dir RUN
--stage S --reason "<text>" [--dry-run]` (`dynorch/invalidate.py`) selects every CURRENT record of `S` not
`transport_failed` whose attempt had an upstream failure (journaled `error.code == "upstream_failed"`, a 502 in
`calls_detail`, or an `upstream_failed:` / `infrastructure_stop:` error), refuses while the stage's job may run (a
`running` record, an open ledger attempt of `S`, `state/status.json` running `S` -- `--ignore-status-file` drops only
this last check, for a job Slurm killed outright; `--dry-run` reports them and moves nothing), copies each original verbatim to `superseded_executions/S/infra_invalidation_<utc>/originals/`,
writes it rewritten (`transport_failed` / `infra_invalidated`, scored fields cleared, the originals under
`harness_reported["infra_invalidation"]`) to its `runs_superseded/` attempt path, removes it from `runs/`, writes
`INVALIDATION.json` (layout `dynorch/infra_invalidation/v1`) and records `live.record_decision(kind=
"infra_invalidation", cells=[...], reason=...)`. `Runner._existing_by_cell` takes a cell's highest superseded attempt
when `runs/` holds none, so the next `dynorch-run --stage S --resume` re-executes exactly those cells as attempt + 1 /
generation + 1. Never touched: the ledger journal, `locks/`, the plan manifest, `raw/`, other stages. Plan §9 Gate 3
("a post-freeze fix that affects scoring forces re-execution of every affected cell") is satisfied by running it after
the main job ends and resuming. Evidence: `test_core_flow.py` (latch cases), `test_runner_rules.py`,
`test_dynorch_e2e_fake_harness.py` (interruption + 5xx-retrying fake harness, H0, bounded infra reruns, kill/resume,
invalidate → resume on the real journaled records in `tests/fixtures/dynorch/infra_invalidation/`),
`test_invalidate.py`, and the real-CLI smoke cases `openclaude/interruption_retry_latched` and
`codex/interruption_retry_latched` (both CLIs retry the 502; every retry is refused with no upstream call).

**10.39 The `/key` verdict checks the key against the ledger's BOUNDS (lead, 2026-09-24; main-stage finding).** The
ledger records two things: `settled` (the provider's own reported cost of every completed call) and `charged_max`
(the full reservation of ambiguous calls — B2 — such as provider-interrupted streams, which the provider may bill at
nothing). True spend therefore lies in [settled, settled + charged_max]. The main stage settled $26.277 and charged
$0.691 at maximum; the key settled at $26.301 (eleven identical reads over five minutes), i.e. those interrupted calls
really cost about $0.02. §10.36 compared the key only with the UPPER figure and declared a mismatch that would also
have failed Gate 4. The verdict is now: clean iff settled − tol ≤ key delta ≤ settled + charged_max + tol. A key below
the settled floor (the ledger over-counted billed calls) or above the upper bound (the provider billed more than was
metered) is still a mismatch. The settle loop converges on the range (two consecutive in-range reads that are stable
or at the upper bound). With `charged_max = 0` the range is a single value and the §10.36 behaviour is unchanged. The
report adds `ledger_settled_delta_usd`, `ledger_charged_max_delta_usd`, `key_minus_settled_usd` and
`key_within_ledger_bounds`.
