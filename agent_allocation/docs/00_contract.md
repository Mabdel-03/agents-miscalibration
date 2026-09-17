# 00 — Integration contract for `openrouter_agent_allocation_v1`

Lead-owned. Read with `MASTER_SPEC.md`, your brief in `agents/0N_*.md`, `configs/balanced150.json`, and
`src/agent_allocation/contracts.py`. Where this file is more specific than your brief, this file wins; where it
conflicts with `MASTER_SPEC.md`, stop and tell the lead.

## 0. Rules

- **R1 — Ownership.** Edit only the paths your brief assigns (table in §1). Lead-only: `pyproject.toml`, `uv.lock`,
  `contracts.py`, `paths.py`, `config.py`, `config_cli.py`, `apikey.py`, `cli.py`, `manifest.py`, `plan.py`,
  `live.py`, `gates.py`, `freeze.py`, `doctor.py`, `slurm/`, this file. Need a change there (a field, a dependency,
  a path)? Write it in `docs/requests/<agent>.md` on your branch and carry on against a local shim.
- **R2 — No money.** No API key, no network call to openrouter.ai, ever, from any agent. Everything runs on the shared
  fake transport (`agent_allocation.api.fake`). Tests that would need the network are marked `live` and never run.
- **R3 — One codec.** Contract records cross module or process boundaries only via `contracts.dump_record` /
  `contracts.load_record`. If a file needs extra keys, wrap: `{"<key>": ..., "record": dump_record(x)}`.
- **R4 — One path registry.** Resolve every artifact location through `paths.RunPaths`. Never join run-dir paths by hand.
- **R5 — Stable identity.** `GenerationSpec.node_id` is the node identity; `candidate_id == node_id`. The golden
  fixtures in `tests/unit/test_node_identity_golden.py` pin the coordinates in `GridSpec`'s docstring. Prompt hashes
  are audit data, never cache keys.
- **R6 — Standalone package.** No imports from `agents_scaling.*` or `learnagents.*` at runtime. Reuse by copying into
  `src/agent_allocation/_vendor/<module>.py` and adding a row to `src/agent_allocation/_vendor/PROVENANCE.json`
  (`{"file", "origin", "origin_commit", "origin_sha256", "functions", "changes"}`). Keep copies minimal.
- **R7 — Environment.** From `agent_allocation/`: `source slurm/env.sh && uv sync --frozen` once, then
  `uv run --frozen --no-sync pytest tests/unit/<yours>`. Never write caches under `/home`.
- **R8 — Commit early.** Commit on your branch at every green increment. **Milestone M1 (≤45 min after start):** a
  walking-skeleton implementation of *every* interface you own in §2 — simple but correct on the happy path, no
  `NotImplementedError`, with unit tests — committed. The lead merges all M1 branches and runs the end-to-end
  skeleton test; then deepen. **M2:** full brief. Before each commit: `uv run --frozen --no-sync pytest tests/unit`
  green in your worktree.
- **R9 — Fail loudly.** Unknown stays unknown (`None`), never `0`/`False`. A declared knob must be read
  (config keys you consume are listed in your module docstring). Empty selector output is a blocker, not zero accuracy.

## 1. Ownership

| Agent | Owns | Delivers to |
|---|---|---|
| 01 OpenRouter | `src/agent_allocation/api/`, `tests/unit/api/` | 02, lead (`live.py`), everyone (`api/fake.py`) |
| 02 Scheduler/ledger | `src/agent_allocation/runtime/`, `src/agent_allocation/storage/`, `tests/unit/runtime/`, `tests/unit/storage/` | lead (`live.py`) |
| 03 Protocols | `src/agent_allocation/protocols/`, `src/agent_allocation/prompts/`, `tests/unit/protocols/` | 02, 06, 07, lead |
| 04 Text benchmarks | `src/agent_allocation/benchmarks/{__init__,math500,mmlu_pro,bbh_logic4,text_manifest}.py`, `src/agent_allocation/grading/text/`, `tests/unit/text/` | lead (`manifest.py`), 05 worker, 06 |
| 05 Code benchmark | `src/agent_allocation/benchmarks/mbpp_plus.py`, `src/agent_allocation/grading/code/`, `src/agent_allocation/grading/worker.py`, `src/agent_allocation/sandbox/`, `tests/unit/code/` | lead, 06 |
| 06 Selection/metrics | `src/agent_allocation/extraction/`, `src/agent_allocation/selection/`, `src/agent_allocation/metrics/`, `tests/unit/selection/`, `tests/unit/metrics/` | 07, lead |
| 07 Analysis/reports | `src/agent_allocation/analysis/`, `src/agent_allocation/reporting/`, `tests/unit/analysis/` | lead |
| 08 QA | `tests/integration/`, `tests/fixtures/adversarial/`, `docs/audit/` | lead |

`src/agent_allocation/_vendor/` is shared: add files, never edit another agent's vendored file.
`src/agent_allocation/grading/__init__.py` belongs to 05.

## 2. Frozen interfaces

Signatures are frozen; internals are yours. Types come from `agent_allocation.contracts` unless noted.

### 01 — `agent_allocation.api`
```python
# api/request.py
REQUEST_HEADERS: Mapping[str, str]   # includes "X-OpenRouter-Cache": "false", Content-Type, X-Title
def build_request_body(spec: GenerationSpec, messages: Sequence[Mapping[str, str]], lock: EndpointLock,
                       sampling: Mapping[str, Any]) -> dict[str, Any]
    # {"model", "messages", "temperature", "top_p", "max_tokens": spec.max_completion_tokens,
    #  "reasoning": {"max_tokens": spec.reasoning_budget, "exclude": False},
    #  "provider": {"only": [lock.provider_tag], "allow_fallbacks": False, "require_parameters": True},
    #  "stream": False}  -- no seed, no reasoning.effort, no tools/plugins/transforms
# api/client.py
class OpenRouterClient:
    def __init__(self, *, api_key: str | None, base_url: str = "https://openrouter.ai/api/v1",
                 transport: httpx.AsyncBaseTransport | None = None, connect_timeout: float = 15.0,
                 read_timeout: float = 600.0) -> None      # one persistent httpx.AsyncClient, retries=0
    async def generate(self, spec: GenerationSpec, messages: Sequence[Mapping[str, str]], *, attempt_id: str,
                       lock: EndpointLock, sampling: Mapping[str, Any]) -> GenerationOutcome
        # never retries, never touches the ledger or disk; classifies TransportStatus/CompletionStatus;
        # cache HIT -> CACHE_HIT; served provider/model outside lock -> PROVIDER_MISMATCH
    async def key_info(self) -> Mapping[str, Any]            # /api/v1/key data WITHOUT "label"
    async def credits(self) -> Mapping[str, Any]             # /api/v1/credits data
    async def generation_metadata(self, generation_id: str) -> Mapping[str, Any] | None   # None if not ready
    async def list_endpoints(self, model: str) -> tuple[bytes, Mapping[str, Any]]           # raw bytes + parsed
    async def aclose(self) -> None
# api/endpoints.py
class EndpointBlocked(RuntimeError): ...
def select_standard_endpoint(snapshot: Mapping[str, Any], model: str, *, preferred_tag: str = "google-ai-studio",
                             snapshot_path: str = "", snapshot_sha256: str = "", fetched_at: str = "") -> EndpointLock
    # rejects tags containing "/flex", "/priority", ":free", preview/experimental models; never invents a tag
# api/fake.py  (M1 priority: everyone's tests use it)
@dataclass class FakeScript: ...   # per-(item_id, kind, root_slot, update_round) canned content, usage, headers,
                                   # injected faults: 429+Retry-After, 402, 401, 5xx, 200 error envelope,
                                   # cache HIT header, provider mismatch, empty length, read timeout, delay
def fake_openrouter_transport(script: FakeScript | None = None) -> httpx.MockTransport
def fake_endpoint_lock(model: str = "google/gemini-2.5-flash-lite") -> EndpointLock
```

### 02 — `agent_allocation.storage`, `agent_allocation.runtime`
```python
# storage/raw.py
def write_raw_response(paths: RunPaths, attempt_id: str, outcome: GenerationOutcome) -> tuple[str, str]
    # atomic (tmp + fsync + rename), gzip JSON {attempt_id, node_id, request_body, response_status,
    # response_headers, response_body_text}; returns (run-dir-relative path, sha256 of the file bytes)
# storage/journal.py
class LedgerJournal:
    def __init__(self, path: Path, *, group_commit_ms: int = 250) -> None
    async def append(self, event: LedgerEvent) -> None       # returns only after fsync
    def replay(self) -> Iterator[LedgerEvent]
# storage/state.py
class StateStore:                                            # SQLite WAL on paths.state_db_local, one writer
    def __init__(self, paths: RunPaths) -> None
    def register_nodes(self, specs: Iterable[GenerationSpec], *, wave_index: int | None) -> None
    def terminal_node_ids(self) -> set[str]
    def record_attempt(self, attempt: Attempt) -> None
    def record_result(self, result: GenerationResult, candidate: Candidate) -> None   # after raw write
    def backup(self) -> None                                 # online backup to paths.state_db_backup
    @classmethod
    def recover(cls, paths: RunPaths) -> "StateStore"        # restore backup, replay journal, re-ingest parsed/raw
# runtime/ledger.py
@dataclass(frozen=True) class LedgerSnapshot: settled: Decimal; in_flight: Decimal; unresolved: Decimal; ceiling: Decimal
class CollectionLedger:
    def __init__(self, journal: LedgerJournal, *, ceiling_usd: Decimal, max_unresolved_usd: Decimal) -> None
    async def reserve(self, *, attempt_id: str, node_id: str, partition_id: str, amount_usd: Decimal) -> bool
    async def mark_sent(self, attempt_id: str) -> None
    async def settle(self, attempt_id: str, cost_usd: Decimal) -> None
    async def release_unsent(self, attempt_id: str) -> None
    async def charge_max(self, attempt_id: str) -> None
    def set_active_ceiling(self, ceiling_usd: Decimal) -> None  # min(study cap, key remaining, account balance)
    def snapshot(self) -> LedgerSnapshot
# runtime/estimate.py
def reservation_usd(spec: GenerationSpec, rendered_chars: int, lock: EndpointLock, *, multiplier: Decimal) -> Decimal
    # conservative: input tokens <= ceil(chars / 2) + 64; output = spec.max_completion_tokens at the output price
# runtime/scheduler.py
RenderFn = Callable[[GenerationSpec, Sequence[tuple[GenerationSpec, Candidate | None]]], list[dict[str, str]]]
ExtractFn = Callable[[GenerationResult, GenerationSpec], Candidate]
@dataclass(frozen=True) class AdmissionPlan: waves: tuple[tuple[ItemGraph, ...], ...]   # lead builds it
@dataclass(frozen=True) class RunSummary: ...   # counts, spend, measured peak/mean in-flight, stop reason
class Dispatcher:
    def __init__(self, *, config: Mapping[str, Any], paths: RunPaths, store: StateStore, ledger: CollectionLedger,
                 client: OpenRouterClient, lock: EndpointLock, render: RenderFn, extract: ExtractFn) -> None
    async def run(self, plan: AdmissionPlan, *, allow_paid: bool, max_usd: Decimal) -> RunSummary
# runtime/status.py:  def cmd_status(argv: list[str]) -> int
# runtime/control.py: def cmd_stop(argv: list[str]) -> int      # writes paths.stop_sentinel / drain_sentinel
```

### 03 — `agent_allocation.protocols`, `agent_allocation.prompts`
```python
# protocols/grid.py
def build_item_graph(grid: GridSpec, *, run_id: str, partition_id: str, task: PublicTask, model: str,
                     base_budget: int, headroom: int, endpoint_lock_hash: str, decoding_config_hash: str,
                     prompt_version: str) -> ItemGraph
    # nodes topologically ordered; checkpoints named as in tools/reference_dag.py, both CURRENT and ARCHIVE views
# protocols/policy_cost.py
def policy_ancestors(graph: ItemGraph, checkpoint: PolicyCheckpoint) -> tuple[str, ...]
def policy_cost_rows(graph: ItemGraph, results: Mapping[str, GenerationResult]) -> list[dict[str, Any]]
# prompts/render.py
def prompt_version() -> str                       # "prompts-v1-" + first 12 hex of the template-set hash
def prompt_hashes() -> dict[str, str]             # per template file
def output_requirement(task: PublicTask) -> str   # from prompts/output_requirements.json (bbh_choice/bbh_boolean)
def render_messages(spec: GenerationSpec, task: PublicTask,
                    parents: Sequence[tuple[GenerationSpec, Candidate | None]], *, seed: int) -> list[dict[str, str]]
    # system.txt + root/private/peer/pooled; latest visible solution only; blind peer order from a stable hash of
    # (seed, item, n, round, receiver); invalid-prior marker; "[No prior candidate response is available.]";
    # symmetric rationale trimming that never cuts <final>, recording bytes removed
```

### 04 — text benchmarks
```python
# benchmarks/math500.py, mmlu_pro.py, bbh_logic4.py (each)
def load_inventory(cfg: Mapping[str, Any], paths: RunPaths) -> list[InventoryItem]
# benchmarks/text_manifest.py
def select_panels(inventory: Sequence[InventoryItem], *, benchmark: str, seed: int, n_dev: int, n_main: int,
                  n_reserve: int) -> PanelDraw      # PanelDraw ordering rules in contracts.PanelDraw
# grading/text/grade.py
def grade_text(candidate: Candidate, private: PrivateEvalSpec) -> GradeResult
# grading/text/math_equiv.py   (gold-blind helper exported to 06; bounded time, runs in worker processes)
def math_equivalent(a: str, b: str, *, timeout_s: float = 5.0) -> bool | None   # None = unknown/timeout
```

### 05 — code benchmark
```python
# benchmarks/mbpp_plus.py
def load_inventory(cfg: Mapping[str, Any], paths: RunPaths) -> list[InventoryItem]
def select_panels(inventory: Sequence[InventoryItem], *, seed: int, n_dev: int, n_main: int, n_reserve: int) -> PanelDraw
# grading/code/public.py
def run_public_checks(candidate: Candidate, task: PublicTask) -> PublicCheckResult
# grading/code/private.py
def grade_code(candidate: Candidate, private: PrivateEvalSpec, paths: RunPaths) -> GradeResult
# grading/worker.py
def cmd_grade(argv: list[str]) -> int   # --run-dir --resume [--follow] [--workers N]; public checks first
                                        # (code), then private grades (grade_text / grade_code); cache by
                                        # (revision, item, code sha, kind, evaluator version, env digest)
```

### 06 — extraction, selection, metrics
```python
# extraction/final.py
def extract_candidate(result: GenerationResult, spec: GenerationSpec, task: PublicTask) -> Candidate
# selection/readout.py
def select_checkpoint(checkpoint: PolicyCheckpoint, task: PublicTask, candidates: Mapping[str, Candidate],
                      public_checks: Mapping[str, PublicCheckResult], *, seed: int,
                      math_equivalent: Callable[[str, str], bool | None] | None = None) -> SelectionResult
# selection/seal.py
def seal_selections(paths: RunPaths, partition: str, results: Sequence[SelectionResult]) -> str   # sha256
# metrics/ : pass_at_k(n, c, k) exact; coverage; selection gap; archive coverage; transitions; per-item tables
```

### 07 — analysis, reporting
```python
# reporting/cli.py:   def cmd_analyze(argv: list[str]) -> int     # --run-dir
# reporting/audit.py: def cmd_audit(argv: list[str]) -> int       # §12 invariants + 08's independent recompute
```

### Lead
`manifest.py` (`prepare-data`: inventories → `PanelDraw`s → waves with nested panels first → manifests + write-once
dataset lock), `plan.py` (dry-run counts), `live.py` (canary/pilot/run wiring 01+02+03+06), `gates.py`,
`freeze.py`, `doctor.py`.

## 3. On-disk formats (all under `RunPaths`)

| Path | Content |
|---|---|
| `manifests/public/{benchmark}.jsonl` | `PublicTask` per line |
| `private/{benchmark}.jsonl` (0700) | `PrivateEvalSpec` per line |
| `manifests/panels.json` | `{benchmark: PanelDraw}` |
| `manifests/waves.json` | `[Wave]` |
| `manifests/jobs/{partition}.jsonl` | `{"node_id", "record": GenerationSpec}` (readers verify `node_id`) |
| `manifests/policies/{partition}.jsonl` | `PolicyCheckpoint` |
| `raw/{shard}/{attempt}.json.gz` | see `write_raw_response` |
| `parsed/{shard}/{node_id}.json` | `{"result": GenerationResult, "candidate": Candidate}` |
| `public_checks/{shard}/{candidate_id}.json` | `PublicCheckResult` |
| `grades/{shard}/{candidate_id}.json` (0700) | `GradeResult` |
| `selections/{partition}.jsonl` + `.seal.json` | `SelectionResult`; seal `{sha256, count, sealed_at}` written before any grade join |
| `state/ledger_journal.jsonl` | `LedgerEvent` per line, fsync'd |
| `gates/{gate}_{stage}.json` | `GateResult` |
| `amendments.jsonl` | `{id, recorded_at, rule, scope, before, after, reason}` |

## 4. Facts already established (do not rediscover)

- Pins: MATH-500 `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`; MMLU-Pro `b189ec765aa7ed75c8acfea42df31fdae71f97be`
  (both cached: `$HF_HOME/hub/datasets--<org>--<name>/snapshots/<sha>/`); BBH commit
  `9ee07bd481feebf959a6b59d61ea57bdcf30964d` (MIT; fetch the four files once into `paths.bbh_sources_dir` with sha256).
- OpenRouter (live 2026-09-17): Flash-Lite standard endpoint tag `google-ai-studio` ($0.10/$0.40 per M); variants
  `google-ai-studio/flex`, `/priority`, `google-vertex`, `google-vertex/eu`. Flash: `google-ai-studio` ($0.30/$2.50).
  Base slug does not match tier variants. Cache header `X-OpenRouter-Cache: false`; status `X-OpenRouter-Cache-Status`.
  Usage is always returned (`usage.cost`, `completion_tokens_details.reasoning_tokens`,
  `prompt_tokens_details.cached_tokens`, `cost_details.upstream_inference_cost`). Reasoning that exhausts
  `max_tokens` returns `finish_reason:"length"` with empty content and is billed. `/generation?id=` has
  `provider_name`, `service_tier`, `native_tokens_reasoning`, `response_cache_source_id`; it lags, so `None` = not ready.
- Existing reusable code (copy per R6; origin commit `e4b4037` of `/orcd/data/tpoggio/001/mabdel03/agents_scaling`):
  exact `pass_at_k`, `plurality_vote`, `evaluate_sealed_selection` in `src/agents_scaling/study/metrics_reference.py`;
  class tie-break `src/agents_scaling/nextdyn/selectors.py:335`; transitions `src/agents_scaling/ard/reference_metrics.py:57`,
  `src/agents_scaling/nextdyn/outcomes.py:350-491`; cluster bootstrap/Holm `src/agents_scaling/ard/stats.py` (pass
  explicit domains/weights; NOT `study/stats.py`); `<final>` extraction `src/agents_scaling/ard/report.py:147-173`;
  one-fence strip `src/agents_scaling/study/parse/candidate.py:127`; MC keys `src/agents_scaling/nextdyn/graders/mc.py`,
  `src/agents_scaling/study/selection/normalize.py`; stratified allocation `src/agents_scaling/ard/items_build.py:229-289`;
  blind order/HMAC `src/agents_scaling/study/identity.py`; prompt lock `src/agents_scaling/nextdyn/prompts.py`;
  packet clipper `src/agents_scaling/study/packets.py:200` (add a never-clip-final guard).
- MBPP+ (agent 05): `/orcd/data/tpoggio/001/mabdel03/agent_methods_bench` (commit `2af2879`):
  `learnagents/runtime/sandbox.py` + `sandbox_worker.py` (bwrap `--unshare-all` + seccomp + prlimit; stdlib worker),
  `learnagents/data/evalplus_scorer.py` (EvalPlus `26d6d00b` comparisons from hash-pinned upstream source in
  `data/upstream/evalplus/26d6d00b…`; expected outputs stay in the trusted process), certification
  `manifests/evalplus_reference_certification_v4.json` (542/542). MBPP+ frozen tasks `data/frozen/mbpp_plus_v1/`
  (378); oracles: 377 in `data/derived/evalplus_oracles_v1/mbpp_plus` + 1 in `data/derived/evalplus_oracles_v4/mbpp_plus`
  (resolve via the v4 certification paths/hashes; Mbpp/404 needs `decode_evalplus_record`). Pass
  `python_prefix=/orcd/home/002/mabdel03/conda_envs/asys_env` explicitly (venv symlinks break inside bwrap).
  Every MBPP+ prompt has exactly one `assert`: public checks parse it from the prompt, **never** from the `assertion`
  field (it holds private base tests); pre-import `math` and `sys`. Config limits (512 MiB, 64 pids, 64 KiB stdout,
  3 s public case, 60 s suite, ≤8 workers) differ from the certified 4 GiB profile: re-run all references at the
  config limits first and report conflicts to the lead (one recorded amendment, never silent). Stdout overflow must
  truncate, not fail; set an explicit stack limit; sandbox outage ≠ wrong answer (`EVALUATOR_ERROR`).
- Seccomp in that sandbox denies thread creation: count EPERM failures and report them as a limitation.
- `os.cpu_count()` over-reports on Slurm; use `len(os.sched_getaffinity(0))`.
- pandas is 3.x (string dtype): never test `dtype == object`.

## 5. Grids and the pilot

`config.grids_from_config` returns `main`, `main_extension`, `high`, `replication`, `pilot_r1`, `pilot_r2`.
Counts (acceptance): main 125, high 123, replication 57, main_extension 138 per item; pilot_r1 12, pilot_r2 15.
Pilot (lead): 32 dev items × `pilot_r1`, 16 of them `pilot_r2`, plus 4 DEEP 16A calls and 16 A=4,096 roots.

## 6. Contract v2 (2026-09-17 03:45 EDT) — resolves docs/requests; merge `allocation-v1` into your branch

**contracts.py**
- `ERROR_ENVELOPE` is in `RETRYABLE_TRANSPORT`; its reservation is kept exactly like `AMBIGUOUS_TIMEOUT` (settle from
  generation metadata or `charge_max`). (D1, R01-1)
- `LedgerEventKind.UNRESOLVED` (amount 0). Replay rule: every SENT reservation without SETTLE/CHARGE_MAX/RELEASE_UNSENT
  counts as unresolved and feeds the $2 pause. (D2)
- `GradeResult.__post_init__`: GRADED ⇒ `semantic_correct` is a bool; EVALUATOR_ERROR/NOT_EVALUATED ⇒ None;
  INVALID_OUTPUT ⇒ `semantic_correct=None` and `operational_success=False`. (D3)
- `PublicCheckResult.pass_fraction` is None when `cases_total == 0`; the checker reports INFRASTRUCTURE_ERROR then. (D4)
- `PrivateEvalSpec.allowed_choices: tuple[str, ...] = ()` — agent 04 fills it for option/boolean items. (R04-1)
- `SelectionResult.status: SelectionStatus` (typed). (D10)
- `load_record(cls, data, *, strict=True)`: every field must be present. Regenerate fixtures rather than using
  `strict=False` (reserved for hand-written fixtures). (D11)
- `Generator` Protocol now matches `OpenRouterClient.generate`. (R01-2)

**paths.py**
- Locks are run-scoped: `<run_dir>/locks/` (spec §14 tree). Node-local state dir = `<base>/<user>/agent_allocation/<name>-<sha12 of resolved run dir>`. (D5, D6)
- Selections are append-only batches: `selections(partition, batch)` / `selection_seal(partition, batch)`; seal =
  `{sha256: sha256 of the batch .jsonl file bytes, count, sealed_at, previous_seal_sha256}`; a sealed batch is never
  rewritten and later batches hold only unsealed checkpoints. (D8, R07-1)
- Grades and public checks keep one current file; a different grader/checker version first moves the old file to
  `superseded_grade(cid, old_version)` / `superseded_public_check(cid, old_version)`. (D9)
- New: `grade_cache_dir` (0700), `metrics_table(name)`, `render_reports`, `context_limit_events`, `provider_usage`,
  `generation_metadata`, `run_decisions`. (R05-1, R06-3, R03-1/2, R07-2)

**config.py** validates deep budgets against `requested_reasoning_max`, `reservation_multiplier >= 1`,
`max_unresolved_spend_usd <= 2`, lock-file basenames, and calls `protocols.grid.check_config_constants`. (D7, R03-3)

**Decisions**
- R02-1: the raw artifact's additive `result` key is part of the format. R02-2: sentinels persist until `stop --clear`.
- R03-1 context-limit policy: when `render` raises `prompts.render.ContextLimitEvent`, the dispatcher (02) makes the
  receiver terminal with no API call and no reservation, candidate None (children render the absent marker). The lead's
  render wrapper logs `context_limit_events` and `render_reports`.
- R03-5 / R06-1: `PolicyCheckpoint.selector_name` stays the declared config family (`plurality` | `public_tests_only`,
  version `v1`); `SelectionResult.selector_name` is 06's concrete selector. 07's audit checks the concrete name is legal
  for the declared family via `selection.readout.selector_for`.
- R06-4: the lead passes `math_equivalent` for MATH checkpoints only.
- R07-2: provider usage lines `{fetched_at, stage, endpoint: key|credits, data}` in `paths.provider_usage` (lead).
- R07-3: 08 provides `tests/integration/independent_recompute.py::independent_recompute(run_dir) -> {"contrasts": {...},
  "checks": [...]}` with contrast names `primary_i_peer_vs_private_team_n4_r4`, `primary_ii_cp_minus_private_team_gain_h3`.
- R07-4: `locks/config.json` contains `"config": {...}`. R07-5: wave labels are `main|extension|high|replication`.
- R01-3: served model must match exactly; the canary confirms; any alias becomes an explicit recorded lock amendment.
- Q1 (08): lead provides `live.run_offline(paths, config, items, transport, *, extension_item_ids=(), code_grader=None)`
  (writes manifests from `items`, generate → extract → grade → select → seal → analyze → audit; rerun = resume; a
  stubbed code grader is labelled in reports). Q2: `Dispatcher` keeps injectable store/client/ledger; 02 defines
  `RunSummary.stop_reason` from a fixed set of constants.
