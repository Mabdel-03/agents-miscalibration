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
  `run_key = sha256(f"{study_id}|{arm}|{item_id}|{rung}|{repeat}|{attempt}")[:24]`; order: seeded item interleave,
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
