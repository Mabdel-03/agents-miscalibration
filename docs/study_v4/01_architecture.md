# agent1

SUMMARY: Build a new self-contained package src/agents_scaling/study/ (flat variant of spec §11.2) with 5 parallel work packages behind a shared types.py/identity.py contract written first by the lead: (1) data adapters + salted-hash splits + byte-exact prompt rendering from the handoff templates, (2) a thin OpenAI-SDK vLLM 0.21 client with registry-based endpoint failover and a content-addressed on-disk request store (<run_root>/requests/<id[:2]>/<id>.json, O_EXCL first-writer-wins) that makes every spec alias (S_FRESH↔F00, IND↔F11, D roots↔F00, IND_PRIVATE_REVISION↔DEC roots) automatic because request_id = SHA256(JCS([...input_hash, decoding_hash, semantic_seed, caps...])) per §3.6, (3) strict candidate/coordinator/subtask parsers, the 2,048-token packet compiler and VOTE normalizers (MC letter, exact-answer normalization, AST-canonical code fallback per amendment S1), (4) an analytic Qwen3 FLOP oracle F(L,T)=C_lin·S+C_attn·S(S+1)/2 (S=L+T; C_lin/C_attn per checkpoint given numerically) plus a reserve-before-launch/debit-actual episode ledger and the five policies S_FRESH/S_HISTORY/IND_VOTE/DEC/CEN_FLAT, and (5) evaluators (HLE MC exact, HLE judge with the official cais/hle prompt on Qwen3-32B thinking-off, BCB unittest inside the bigcodebench-evaluate apptainer image), a deterministic tiered cell-manifest generator, a resumable per-item ThreadPool runner, a run_one entrypoint, and two small forks of launch_chunked.py/run_cell_array.sbatch.tmpl (the originals cannot index study cells because _chunk_complete calls ExperimentCell.from_dict). Everything is unit-tested against a fake OpenAI-compatible HTTP server registered in the real registry directory layout; integration is F-cell → aliased A-cell → seal → judge/eval → aggregate on the fake server before the 20-item dev pilot fixes B0.

# Implementation plan: `src/agents_scaling/study/` (agent_design_v4 on ORCD)

All cluster/data facts are from the design brief; spec citations are to `/orcd/data/tpoggio/001/mabdel03/agents_scaling/expermt_agent_specs/agents_scaling_local_future_state_reversal_experiment_spec_v4_0.md` (§ numbers). Handoff dir = `/tmp/claude-225593/-orcd-data-tpoggio-001-mabdel03-agents-scaling/ba72bd31-8ff0-45c2-b3b0-7ae1ad469566/scratchpad/agent_design_v4/`. Repo = `/orcd/data/tpoggio/001/mabdel03/agents_scaling`. Results root = `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results` (`agents_scaling.config.DEFAULT_RESULTS_ROOT`, also `$ASYS_RESULTS_ROOT` in `slurm/common.sh`).

Verified environment facts that shape the design:
- `asys_env` (`/orcd/home/002/mabdel03/conda_envs/asys_env`): openai 2.38.0, datasets 4.8.5, transformers 5.9.0, pyarrow 24, pandas 3, numpy 2.4, scipy 1.17, pyyaml, backoff 2.2.1, pytest 9. **`jsonschema` and `rlms` are NOT installed** → all wire-format validation is hand-written strict code (the spec demands stricter-than-JSON-Schema checks anyway, §3.5/§11.4). `apptainer` is at `/usr/bin/apptainer`.
- Qwen3 configs in the HF cache (`/orcd/data/tpoggio/001/mabdel03/.cache/huggingface/hub/models--Qwen--Qwen3-{4B,8B,14B,32B}/snapshots/<rev>/config.json`), snapshot revisions: 4B `1cfa9a72…`, 8B `b968826d…`, 14B `40c06982…`, 32B `9216db57…` (these are the `model_revision`/`tokenizer_revision` pins). Dims listed in §4 of this plan.
- HLE-Verified full set is in scratchpad `hle_shards/*.parquet` (15 shards; `hle_verified.parquet` there is only a 134-row sample). Columns: `id, Verified_Classes, category, raw_subject, problem_is_valid…, question, answer, json` where `json` is a string-encoded dict with `image, image_preview, answer_type ∈ {multipleChoice, exactMatch}, rationale, canary, …`. Neither HLE-Verified nor BigCodeBench is in the HF cache → a one-time login-node export step is required (both public, no token).
- Registry layout: `<run_root>/servers/<profile>/<name>.json` containing `ServerEntry` fields; `registry.list_live_servers(run_root, "32B-long")` filters by `entry_matches_profile` (needs `serving_profile`, `served_model_name="32B"`, `max_model_len=40960`, `tp_size=2`) and probes `/health` for entries without a Slurm job id → a test fixture can register a fake server by writing that JSON.
- Old `slurm/launch_chunked.py` cannot drive study cells unmodified: `_chunk_complete` calls `ExperimentCell.from_dict(cells[idx]).cell_id`, and `slurm/run_cell_array.sbatch.tmpl` hardcodes `python -m agents_scaling.experiment.run_one` with fixed `--mem=4G --time=12:00:00` (no `{MEM}`/`{TIME}` placeholders even though `_render` substitutes them). Two small forks are needed (see §5).

---

## 1. Package layout, modules, public interfaces

Flat variant of §11.2 (guide, not mandate). Every module is importable without a GPU or network.

```
src/agents_scaling/study/
  __init__.py
  types.py                 # WP0: shared dataclasses/enums (frozen first)
  identity.py              # WP0: JCS, semantic_seed, request_id, engine seed
  config.py                # WP0: StudyConfig loader + constants (caps, decoding, pins)
  metrics_reference.py     # WP0: vendored verbatim from handoff (pass_at_k, plurality_vote, judge_best)
  data/{hle.py, bcb.py, splits.py, export.py}        # WP1
  prompts/templates/*.txt, prompts/framing_clauses.json  # WP1 (copied byte-exact from handoff)
  prompts/render.py, inference/tokens.py             # WP1
  inference/{client.py, store.py}                    # WP2
  parse/{candidate.py, coordinator.py}, packets.py   # WP3
  selection/{normalize.py, vote.py, judge_best.py}   # WP3 (judge_best cell logic: WP5)
  resources/{oracle.py, broker.py, profile.py}       # WP4
  policies/{base.py, s_fresh.py, s_history.py, ind_vote.py, dec.py, cen_flat.py}  # WP4
  evaluation/{hle.py, bcb.py, bcb_container_driver.py}  # WP5
  cells.py, runner.py, run_one.py, seal.py, aggregate.py # WP5
configs/study_v4.yaml                                 # WP0 (frozen study manifest)
slurm/study_cell_array.sbatch.tmpl, slurm/study_launch_chunked.py, slurm/study_loops.py, slurm/study_loop_driver.sbatch.tmpl  # WP5
tests/study/{fake_vllm.py, conftest.py, test_*.py}   # each WP
```

### 1.1 `types.py` (WP0 — the contract everyone codes against)
```python
class Domain(str, Enum): HLE="hle"; BCB="bcb"
class Framing(str, Enum): F00="00"; F01="01"; F10="10"; F11="11"; NATIVE="nat"
class Method(str, Enum): BANK, S_FRESH, S_HISTORY, IND_VOTE, DEC, CEN_FLAT, IND_PRIVATE_REVISION, DEC_ONE_ROUND, DEGREE
class CellKind(str, Enum): GENERATE, JUDGE_BEST, JUDGE_HLE, EVAL_BCB, FORECAST

@dataclass(frozen=True) class PublicTask:      # task-public view only (§3.3, §10.6)
    source_id: str          # "hle:<id>" | "bcb:<task_id>"
    domain: Domain; split: str  # "dev"|"main"|"reserve"
    task_text: str          # HLE question (revised top-level) | BCB instruct_prompt
    answer_format: str      # "multipleChoice"|"exactMatch"|"code"
    stratum: str            # "Gold"|"Revision"|"bcb"
    rank: int               # salted-hash rank inside its split (item order)
    task_tokens: int        # under the flagship tokenizer (must be <= 4096, §4.3)

@dataclass(frozen=True) class Checkpoint: size: str; hf_id: str; model_revision: str; tokenizer_revision: str; profile: str; tp_size: int; served_model_name: str
@dataclass(frozen=True) class Decoding: temperature: float; top_p: float; top_k: int; min_p: float; presence_penalty: float; repetition_penalty: float; max_tokens: int; enable_thinking: bool
SOLVER_DECODING = Decoding(0.6, 0.95, 20, 0.0, 0.0, 1.0, 8192, True)     # §10.1 dense recipe, §6.4 cap
JUDGE_DECODING  = Decoding(0.0, 1.0, 1, 0.0, 0.0, 1.0, 1024, False)      # §4.4/§6.4 selector cap, thinking off

@dataclass(frozen=True) class SeedKey:          # §3.6 order is the JCS array order
    source_id: str; split: str; model_cell: str; episode_rep: int; actor_slot: int; purpose: str; step_slot: int; namespace: str
@dataclass(frozen=True) class RequestSpec:
    messages: tuple[dict, ...]; decoding: Decoding; checkpoint: Checkpoint; seed_key: SeedKey; role: str  # role for cost/ledger accounting only
    @property request_id -> str; semantic_seed -> bytes; engine_seed -> int; input_hash -> str; decoding_hash -> str
@dataclass class RequestRecord: ...             # §2.1 below (on-disk JSON)
@dataclass(frozen=True) class Candidate: approach: str; evidence: tuple[Evidence,...]; alternatives_considered: tuple[str,...]; failure_checks: tuple[str,...]; final_answer: str; confidence: float
@dataclass(frozen=True) class CandidateRecord: candidate_id: str; request_id: str; slot: int; stage: str; valid: bool; failure_code: str|None; candidate: Candidate|None; candidate_sha256: str; raw_content_sha256: str
@dataclass(frozen=True) class Packet: packet_id: str; sender_slot: int; candidate_sha256: str; fields: dict; spans: list; truncated: dict[str,bool]; final_partial: bool; recipient_tokens: int; serialized: str; unavailable: bool
@dataclass(frozen=True) class CellSpec: cell_id: str; kind: CellKind; module: str; method: Method; checkpoint: str; N: int; B: int; framing: Framing; episode_rep: int; split: str; items: tuple[str,...]; depends_on: tuple[str,...]; max_inflight: int; degree: int|None
@dataclass class EpisodeResult: ...             # §2.2 below
class InfraFailure(RuntimeError)                # exogenous fault after retries (§10.4) -> item INCOMPLETE
class ProtocolError(RuntimeError)               # harness defect -> cell suspended
```

### 1.2 `identity.py` (WP0)
- `jcs(obj) -> bytes`: RFC 8785 for the subset used (str/int/bool/null/list/dict; `json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False).encode()`); raises on float (floats never enter identity inputs; decoding floats are serialized as fixed strings `"0.6"` inside `decoding_hash`). Spec §3.5/§3.6.
- `semantic_seed(study_seed: bytes, key: SeedKey) -> bytes`: `hmac.new(study_seed, jcs([key.source_id, key.split, key.model_cell, key.episode_rep, key.actor_slot, key.purpose, key.step_slot, key.namespace]), sha256).digest()[:16]` (§3.6).
- `engine_seed(seed16) -> int`: `int.from_bytes(seed16[:8],"big") & (2**63-1)` (vLLM `seed` is a signed 64-bit int); collision check = `cells.py` asserts engine-seed uniqueness within every cell manifest and `store.publish` asserts that an existing record with the same engine seed has the same request_id (documented, §3.6).
- `request_id(spec) -> str`: `sha256(jcs([study_id, model_revision, tokenizer_revision, engine_digest, input_hash, decoding_hash, semantic_seed_hex, local_caps, hook_hash])).hexdigest()` with `input_hash = sha256(jcs({"messages": messages, "chat_template_kwargs": {"enable_thinking": …}}))`, `decoding_hash = sha256(jcs(decoding_as_strings))`, `local_caps = {"max_tokens": max_tokens}`, `hook_hash = "none"`, `engine_digest = "vllm==0.21.0;dtype=bf16;tp=<tp>;reasoning_parser=qwen3;profile=<profile>"` (§3.6, §6.7). Budget label is NOT in the identity (§6.7).
- `blind_order_key(study_seed, namespace, *parts) -> bytes`: HMAC for peer/candidate display order and D-module root permutation (§4.1, §4.6).

### 1.3 `config.py` (WP0) — `configs/study_v4.yaml`
```yaml
study_id: agent_design_v4_orcd
study_seed_hex: <64 hex, generated once with secrets.token_hex(32) and committed>
split_salt_hex: <64 hex>
checkpoints: {32B: {hf_id: Qwen/Qwen3-32B, model_revision: 9216db5781bf21249d130ec9da846c4624c16137, profile: 32B-long, tp_size: 2}, 14B: {…40c069824f…, profile: 14B-long, tp: 1}, 8B: {…b968826d9c…, 8B-long}, 4B: {…1cfa9a7208…, 4B-long}}
flagship: 32B          # amendment F1
judge_checkpoint: 32B  # thinking off
caps: {task_tokens: 4096, prompt_tokens: 32768, solver_out: 8192, selector_out: 1024, own_candidate_tokens: 8192, packet_tokens: 2048, packet_final_tokens: 1024, solver_calls: 64, selector_calls: 64, dec_max_rounds: 8, cen_max_cycles: 8}
items: {dev: {hle: 30, bcb: 30}, main: {hle: 200, bcb: 200}, panels: {N: 100, D: 100, E: 30, M: 150, B: 100}}
budget: {B0_flops: null, root_prompt_max_tokens: 6144}   # B0 filled by resources/profile.py after pilot; frozen copy written to <run_root>/FROZEN.yaml
```
`load_config(path) -> StudyConfig`; `StudyConfig.frozen(run_root)` refuses to run generate cells if `<run_root>/FROZEN.yaml` is absent or its sha256 differs (§9.1 freeze, §11.2 "reject stale hashes").

### 1.4 `data/` (WP1)
- `hle.py`: `load_hle_rows(shards_dir | hf) -> list[dict]`; `hle_eligible(row) -> tuple[bool, str]` with reasons: `Verified_Classes` not in {`Gold subset`, `Revision subset`} (exact strings verified for Gold; confirm Revision/Uncertain literals on the shards at implementation), nested `json.image` non-empty or `json.image_preview` not None → `image_dependent`, duplicate `id` → `duplicate`, `task_tokens > 4096` → `envelope` (§3.2, §4.3). Public view uses top-level `question`/`answer` (brief: revised authoritative). `answer_format = json.answer_type`. `PublicTask` + `ProtectedLabel{source_id, answer, answer_type, rationale}`.
- `bcb.py`: `load_bcb(revision="b74c0d0bf70d2c0bc459be537895cca163007f1a", split="v0.1.4")` via `datasets` on the login node; public = `{task_id, instruct_prompt, entry_point, libs}`; protected = `{test, canonical_solution, code_prompt}` (§3.2: official tests and canonical_solution are evaluator-only).
- `splits.py`: `rank_key(salt, source_id) = HMAC-SHA256(salt, b"rank:"+source_id)`; per domain (and per HLE stratum) sort by rank; `assign_splits(tasks, cfg) -> dict[source_id, (split, rank)]`: dev = first 15 Gold + 15 Revision + 30 BCB; main = next 100 Gold + 100 Revision + 200 BCB; rest = reserve (10% reserve retained, §3.3). Nested panels are prefixes of the main rank order per domain (`panel_items(main, n_per_domain)`), so any cutoff is a balanced prefix (§6.2 "balanced source-item-hashed subset").
- `export.py` CLI: `python -m agents_scaling.study.data.export --run-id study_v4 --config configs/study_v4.yaml --hle-shards <dir>` → writes `<run_root>/data/public/tasks.jsonl` (PublicTask rows), `<run_root>/data/protected/hle_labels.jsonl`, `<run_root>/data/protected/bcb_tests.jsonl` (mode 0600, directory 0700), `<run_root>/data/splits.json`, `<run_root>/data/exclusions.jsonl`, and `<run_root>/data/DATA_SHA256.json`. Generate/judge_best cells import only `load_public_tasks(run_root)`; `evaluation/*` is the only importer of `load_protected_*` (§10.6 firewall; test asserts by grep that no module outside `evaluation/` references `protected`).

### 1.5 `prompts/` (WP1)
- Templates copied byte-exact from the handoff (`independent_root.txt`, `focal_revision.txt`, `central_hub.txt`, `central_worker.txt`, `common_consumer.txt`, `judge_best.txt`, `forecast.txt`) plus `framing_clauses.json`; `prompts/hle_judge.txt` = the official cais/hle judge prompt (fields `extracted_final_answer / reasoning / correct: yes|no / confidence`), `prompts/dec_root.txt` and `prompts/dec_revision_contract.txt`/`cen_*_contract.txt` = the truthful native clauses required by §4.1 (DEC members told their terminal answer enters the vote and they can exchange messages; hub/worker responsibilities). `PROMPT_HASHES.json` records sha256 of every template (§5.5 "freeze literally").
- `render.py`:
  - `render_root(task: PublicTask, framing: Framing, native_clause: str = "") -> list[dict]`: fills `{{TEAM_CLAUSE_OR_EMPTY}}` with `TEAM_1`/`""`, `{{VOTE_AWARE_CLAUSE…}}` with `VOTE_1 + "\n" + (HLE_CHOICE_SELECTOR | CODE_SELECTOR)` or `""`, `{{EXACT_PUBLIC_TASK}}`, `{{EXACT_CANDIDATE_JSON_SCHEMA}}` (the §3.5 one-line schema). Single `user` message, no system prompt (recorded in `chat_template_hash`). Placement is literal; empty clauses leave their lines empty (part of the intervention, §5.5).
  - `render_dec_revision(task, own: Packet|CandidateRecord|sentinel, peer_packets: list[Packet], N: int, round: int)`, `render_s_history(task, previous: Candidate|sentinel)`, `render_hub(task, N_total, N_workers, prior_plan, returned: list[SubtaskResult], cycle)`, `render_worker(task, assignment, forwarded)`, `render_focal_revision(task, own_root, peer_packets)` (D module), `render_common_consumer(task, saved_state)`, `render_judge_best(task, candidate: Candidate)`, `render_hle_judge(question, response_final_answer, correct_answer)`, `render_forecast(...)`. All state fields are rendered as the ordered canonical array `[role_contract, task, task_only_anchor, own_saved_candidate, allowed_peer_or_hub_packets, allowed_public_observations, state_anchor, output_contract]` with fixed neutral delimiters (§4.3); saved objects are wrapped as "fallible task data".
  - `envelope_check(messages, tokenizer, cap=32768) -> int` raises `ContextFailure` (→ typed context-failure record, §4.3) if rendered prompt tokens exceed the cap.
- `inference/tokens.py`: `load_tokenizer(ckpt) -> PreTrainedTokenizerFast` (local files only, `HF_HUB_OFFLINE=1`), `count_tokens(text)`, `render_chat(messages, enable_thinking) -> list[int]` via `apply_chat_template(add_generation_prompt=True, enable_thinking=…)`; `chat_template_hash`. Used for packet budgeting, envelope checks and prompt-token-id equality with the server's `prompt_token_ids` (§4.3 "preflight checks actual token IDs").

### 1.6 `inference/client.py` (WP2)
```python
class EndpointPool:
    def __init__(self, server_run_root: Path, profile: str, shard: int, refresh_s=300)
    def endpoints(self) -> list[ServerEntry]      # registry.list_live_servers(server_run_root, profile) cached refresh_s
    def pick(self) -> ServerEntry                 # endpoints[(shard + self.rotation) % n]
    def report_failure(self, entry)                # 2 consecutive failures -> rotation += 1, force refresh (rate-limited 60 s)
    def wait(self, timeout_s=300)                  # registry.wait_for_server(server_run_root, profile, shard=shard)

class VllmChatClient:
    def __init__(self, pool: EndpointPool, tokenizer, request_timeout_s=1800, max_exogenous_retries=2)
    def generate(self, spec: RequestSpec) -> RequestRecord   # one HTTP call (+retries), no store access
```
`generate` builds exactly what the old `LogprobClient` sends (verified at `client.py:1081-1170`): `openai.OpenAI(base_url=entry.base_url, api_key="EMPTY").chat.completions.create(model=ckpt.served_model_name, messages=…, temperature=…, max_tokens=…, seed=engine_seed, extra_body={"chat_template_kwargs": {"enable_thinking": …}, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0, "return_token_ids": True})`. Response handling: `message.content`; reasoning read from `message.reasoning` first, falling back to `message.reasoning_content` (both observed by the old client on vLLM 0.21; record `reasoning_field_name`); `choice.token_ids` and `response.prompt_token_ids` (vLLM 0.21 `return_token_ids`); `usage.prompt_tokens/completion_tokens`; `finish_reason`. Assert `prompt_token_ids == render_chat(messages)` (else `ProtocolError`, §4.3). Retry policy (§10.4): only `APIConnectionError/APITimeoutError/InternalServerError/RateLimit` are exogenous, max 2 retries, each retry re-picks an endpoint; anything else or exhaustion → `InfraFailure`. Model EOS/`length` truncation are completed outcomes, never retried.

### 1.7 `inference/store.py` (WP2) — see §2 for semantics
```python
class RequestStore:
    def __init__(self, root: Path)                      # <run_root>/requests
    def path(self, request_id) -> Path                   # root/<id[:2]>/<id>.json
    def get(self, request_id) -> RequestRecord | None    # strict JSON, verifies content sha
    def publish(self, record) -> tuple[RequestRecord, bool]  # (committed_record, was_new); O_EXCL first-writer-wins
    def get_or_generate(self, spec, client, cell_id) -> tuple[RequestRecord, bool]  # aliased=True when found
```

### 1.8 `parse/` and `packets.py` (WP3)
- `parse_candidate(content: str, finish_reason: str) -> CandidateRecord-parts`: if `finish_reason=="length"` → invalid `TRUNCATED`; strip exactly one enclosing ``` fence + surrounding whitespace; `json.loads(text, object_pairs_hook=_reject_dupes, parse_constant=_reject_nonfinite)`; must be a dict with exactly the six keys; types/bounds per `schemas/candidate.schema.json` (approach ≤16,384 code points, ≤32 evidence/alternatives/checks, strings ≤8,192, final_answer ≤131,072, uncertainty ∈ {low,medium,high,unknown}, confidence a non-bool finite number in [0,1]); trailing text → invalid; JSON found only in the reasoning channel → invalid `AMBIGUOUS_CHANNEL` (no salvage, §3.5). Failure codes: `TRUNCATED, NOT_JSON, DUPLICATE_KEY, NONFINITE, SCHEMA, AMBIGUOUS_CHANNEL, EMPTY`. `SENTINEL = {"status":"unavailable","failure_code":"NO_VALID_PARENT"}` (§3.5).
- `parse_coordinator_action(content, N_total) -> Final|Delegate|ActionError`: per `schemas/coordinator_action.schema.json` plus cross-record rules: 1 ≤ len(assignments) ≤ N_total−1, unique `worker_slot` in `[1, N_total−1]`, `source_handles ⊆ allowed handles` (`{"task"}` + forwarded result ids); typed errors `TOO_MANY, DUP_SLOT, BAD_SLOT, HIDDEN_HANDLE, INVALID` (§4.2). `parse_subtask_result(content) -> SubtaskResult|failure` per `schemas/subtask_result.schema.json`; never convertible to a Candidate (type system enforces).
- `compile_packet(cand: CandidateRecord|None, tokenizer, sender_slot, cap=2048, final_cap=1024) -> Packet` (§4.3): priority `final_answer → evidence(claim,support in order) → failure_checks → alternatives → approach → confidence`; each field an exact code-point-prefix excerpt with `[start,end)` spans; final display capped at 1,024 recipient tokens and flagged `partial`; remaining space allocated in priority order; wrapper = `{"sender_slot","candidate_sha256","fields":{…},"truncated":{…},"final_partial"}`; the **serialized** packet (with escapes) is re-tokenized and shrunk (binary search on the prefix length of the last admitted field) until ≤2,048; invalid/missing source → `{"status":"unavailable"}` packet. Unicode-prefix rule: cut on code points, never inside a surrogate pair; record `bytes_sha256`.

### 1.9 `selection/` (WP3)
- `normalize.py`: `hle_mc_key(final_answer) -> str` (regex `^\s*\(?([A-Za-z])\)?[.:)]?\s*$` or leading `Answer:`/`(X)` forms → upper letter; else fall back to `hle_exact_key`), `hle_exact_key(s)`: NFKC → strip → strip one `$…$`/`\boxed{…}`/quotes → lowercase → collapse whitespace → drop trailing `.` → numeric canonicalization via `Decimal` when the whole string parses (drop `+`, thousands separators, trailing zeros) → else the string; frozen on dev (§5.3). `code_key(final_answer) -> (key, mode)`: strip one enclosing fence (dev-frozen rule for code tasks, recorded in the protocol manifest), `ast.parse` → `sha256(ast.dump(tree, include_attributes=False))` with mode `ast`; `SyntaxError` → `sha256(source)` mode `exact_source` (amendment S1; §4.4/§5.3 fallback).
- `vote.py`: `vote(pool: list[CandidateRecord], source_id, study_seed, domain) -> SelectionRecord` wrapping vendored `metrics_reference.plurality_vote` with `PublicCandidate(candidate_id, vote_key, valid)`, `tie_seed = HMAC(study_seed, b"VOTE_TIE")`; records `winning_count, tied_classes, all_singleton, grouping_mode`, `NO_VALID_CANDIDATE` when none valid (§4.4/§5.3).
- `judge_best.py`: `judge_best_select(scores: dict[cid,float], pool, …)` wrapping `metrics_reference.judge_best`; `parse_judge_best(content) -> float|None` (strict JSON with `quality_score` finite in [0,1] + three brief strings; invalid → frozen low score 0.0 + `failure=True`).

### 1.10 `resources/` (WP4) — formulas in §4 of this plan
```python
@dataclass(frozen=True) class DenseArch: n_layers, d_model, n_heads, n_kv_heads, head_dim, d_ff, vocab, tied_embeddings
QWEN3_ARCH = {"4B": DenseArch(36,2560,32,8,128,9728,151936,True), "8B": DenseArch(36,4096,32,8,128,12288,151936,False), "14B": DenseArch(40,5120,40,8,128,17408,151936,False), "32B": DenseArch(64,5120,64,8,128,25600,151936,False)}
class FlopOracle:
    def __init__(self, arch); c_lin: int; c_attn: int
    def prefill(self, L) -> int; def decode(self, ctx) -> int; def call(self, L, T) -> int
    def reservation(self, prompt_tokens, max_tokens) -> int      # known-prompt prefill + full-cap decode (§6.5 step 1)
    def debit(self, prompt_tokens, completion_tokens) -> int      # actual
    def convention(self) -> dict                                  # published counting convention + config hash (§10.2)
class EpisodeLedger:                                              # one per episode, methods are atomic (threading.Lock)
    def __init__(self, B: int, oracle, final_reserve: int)
    def try_reserve(self, group: list[int], keep_final=True) -> Reservation | None   # all-or-nothing for a symmetric group
    def debit(self, res: Reservation, actual: list[int])          # releases unused headroom
    @property remaining, spent, peak_reserved, events            # ResourceEvent list (§11.4)
    def stop(self, reason: Literal["budget","call_cap","round_cap","cycle_cap","completed","context_failure"])
```
`profile.py` CLI: `python -m agents_scaling.study.resources.profile --config … --run-root …` computes `B0 = max_m reservation(minimal full-cap path of m at N=5)` using `root_prompt_max_tokens=6144` (task ≤4,096 + wrapper) and writes `FROZEN.yaml` with `B0_flops`, the per-checkpoint tables (prefill/decode work at L∈{1k,6k,16k,32k}, T∈{1k,8192}) and the N=9/B1 feasibility check for 8B/32B (§6.5, §10.2). Also emits `--from-pilot` telemetry (realized tokens/call by role) for scheduling only.

### 1.11 `policies/` (WP4)
```python
class EpisodeContext:   # built by runner per (cell, item)
    task: PublicTask; cell: CellSpec; cfg; ckpt: Checkpoint; store: RequestStore; client: VllmChatClient; tokenizer; oracle; ledger: EpisodeLedger; pool: ThreadPoolExecutor
    def generate_group(self, specs: list[RequestSpec], role: str) -> list[RequestRecord]   # reserve -> get_or_generate in parallel -> debit; returns in request order
    def seed(self, actor_slot, purpose, step_slot, namespace) -> SeedKey
class Policy(Protocol):  def run(self, ctx) -> EpisodeResult;  id: Method
```
Frozen behaviour (all: 64 solver calls, §6.2; stop reasons recorded, §6.5):
- `S_FRESH` (§4.2): draws k=0,1,2,… with `SeedKey(actor_slot=0, purpose="root", step_slot=k, namespace="stateless_bank")`, framing `00`; each draw reserved singly; VOTE over all valid draws. Draws 0..9 alias the F00 bank automatically.
- `S_HISTORY`: draw 0 = neutral root (same key as S_FRESH draw 0 → aliases F00[0]); draw j≥1 = `render_s_history(task, prev_valid_candidate or SENTINEL)` with `SeedKey(actor_slot=0, purpose="revise", step_slot=j, namespace="S_HISTORY")`; archive all valid candidates; VOTE over archive.
- `IND_VOTE(N, framing)`: roots = draws 0..N−1 with the stateless key (framing `11` in module A → aliases F11[0..4]; `00` in the N panel → aliases F00); optional fresh draws k=N,N+1,… in blind round-robin over slots (slot assignment is metadata only; seed key still `actor_slot=0, step_slot=k` → draws up to 9 alias the bank too, §6.2 "alias its first ten requests"); each optional draw is one indivisible reservation; VOTE over all completed candidates.
- `DEC(N)`: roots = `render_root(task, NATIVE, dec_clause(N))` with `SeedKey(actor_slot=0, purpose="root", step_slot=s, namespace="stateless_bank")` (different prompt bytes → distinct ids from F; identical to `IND_PRIVATE_REVISION` roots → alias); rounds r=1..8 while `N*(1+r) ≤ 64` and `ledger.try_reserve([reservation(prompt_r_s, 8192) for s])` succeeds for the full round (no partial rounds, §4.2); each member s gets `render_dec_revision(task, own_prev, packets of the other N−1 previous-round candidates in `blind_order_key` order)` with `SeedKey(actor_slot=s, purpose="revise", step_slot=r, namespace="DEC")`; invalid own → SENTINEL; VOTE over the N latest slot outputs (invalid retained as failed opportunities). `DEC_ONE_ROUND` = same with `max_rounds=1`; `IND_PRIVATE_REVISION` = DEC roots + one revision with empty peer list, VOTE over the 5 terminal outputs (§4.6).
- `CEN_FLAT(N)`: hub cycle c=0..7: reserve `hub_call + (N−1)*worker_full_cap + final_hub_call` before launching a delegation (§6.5 step 1 "mandatory work those outputs can require"); `render_hub` → `parse_coordinator_action`; `final` → candidate; `delegate` → workers `render_worker` with `SeedKey(actor_slot=slot, purpose="worker", step_slot=c, namespace="CEN_FLAT")` in parallel, results returned in planned order; typed errors count as a used hub call and the next hub prompt carries the error; after cycle 8 or when a delegation would not fit, issue the reserved final "no-more-work" hub call (§4.2). Output = coordinator's own candidate (invalid → episode incorrect). Hub seed `SeedKey(actor_slot=0, purpose="hub", step_slot=c, namespace="CEN_FLAT")`.
- `DEGREE(d)` (module D, §4.6): 9 roots = F00 draws 0..8 (aliases); focal = root at position 0 of `blind_order_key(study_seed,"D_ROOT_PERM",source_id)` permutation; peers = next d in that order; one `render_focal_revision` (`purpose="revise", step_slot=d, namespace="DEGREE"`), four `render_common_consumer` children (`actor_slot=j, purpose="consumer", step_slot=d`); no VOTE; records focal + 4 child candidates.
- `EpisodeResult` records everything in §2.2 including `counters` (§3.4: assigned roster, unique actors used, resets, peak live contexts, calls by role, candidates).

### 1.12 `evaluation/` (WP5; the only code allowed to read `data/protected`)
- `hle.py`: `score_mc(candidate.final_answer, gold_letter) -> bool` via `hle_mc_key`; `hle_judge_request(question, final_answer, gold) -> RequestSpec` (JUDGE_DECODING on the judge checkpoint, `SeedKey(purpose="hle_judge", namespace="judge")`); `parse_hle_judge(content) -> "yes"|"no"|"ambiguous"` (regex on `correct:` line, case-insensitive; anything else ambiguous → scored incorrect, flagged for audit, §3.3 "never auto-score ambiguous as correct"). Candidates are deduplicated by `sha256(final_answer)` per item before judging (deterministic judge, greedy).
- `bcb.py`: `BcbEvaluator(image_sif: Path, timeout_s=120, mem_mb=4096)`; `evaluate_many(items: list[(cid, code, test_src)]) -> dict[cid, {"status": pass|fail|timeout|error, "stdout_tail","stderr_tail","elapsed"}]` runs ONE `apptainer exec --contain --containall --net --network none --no-home --bind <tmpdir>:/work <sif> python /work/bcb_container_driver.py /work/jobs.jsonl /work/results.jsonl`; `bcb_container_driver.py` (copied into tmpdir) loops over jobs, writes `solution.py = code + "\n\n" + test`, runs `subprocess.run([sys.executable,"-m","unittest","-q","solution"], timeout=…, preexec_fn=rlimits)` in a fresh subdir; pass = returncode 0 and "OK" in stderr. Unique code strings evaluated once per item (result copied to duplicates). Image pulled once on the login node: `apptainer pull /orcd/data/tpoggio/001/mabdel03/containers/bigcodebench-evaluate-v0.2.4.sif docker://bigcodebench/bigcodebench-evaluate:v0.2.4`; cells never pull.

### 1.13 `cells.py`, `runner.py`, `run_one.py`, `seal.py`, `aggregate.py` (WP5)
- `cells.py`: `build_cells(cfg, tasks, tier) -> list[CellSpec]`; deterministic; `cell_id = f"{module}.{method}.{ckpt}.N{N}.B{B}.F{framing}.e{rep}.s{shard:03d}"`; items per cell: F 20, A/N/E/D/M/B 10, JUDGE_HLE/EVAL_BCB/JUDGE_BEST 25; items ordered by split rank; shards interleave domains so every shard has HLE and BCB. Tiers: `pilot` (dev 20 items: all five methods N=5 B4 + 4 banks + judges/evals), `1`, `1-eval`, `2`, `2-eval`, `3`. CLI `python -m agents_scaling.study.cells --run-id study_v4 --tier 1 --out cells_tier1.json` writes `<run_root>/cells_tier1.json` + `.sha256` and refuses to overwrite a differing frozen manifest. Also asserts engine-seed uniqueness (§3.6 collision check).
- `runner.py`: `run_cell(cell, run_root, server_run_root, shard, stop_event) -> Path` (details §3).
- `run_one.py` argparse: `--run-id` (required), `--cells-file` (required, absolute or relative to run_root), `--index` (required), `--server-run-id` (default = run-id), `--store-root` (default `<run_root>/requests`), `--max-inflight` (override), `--dry-run`. Compatible with the array template `exec python -m agents_scaling.study.run_one --run-id … --cells-file … --index $SLURM_ARRAY_TASK_ID`.
- `seal.py`: `python -m agents_scaling.study.seal --run-id study_v4 --cells-file cells_tier1.json` → `<run_root>/seals/<manifest_sha>/{candidates.jsonl, pools.jsonl, selections.jsonl, SEAL.json}`; F prefixes 1,2,3,5,9,10 per framing get VOTE selections here (§5.5, §6.2); evaluation cells refuse to run unless `SEAL.json` covers their inputs (§3.6 "seal pool definitions and selected IDs before correctness joins").
- `aggregate.py`: joins item files + eval files → `<run_root>/tables/*.parquet` (per item×cell×selector rows: native_final_correct, vote_correct, judge_best_correct, candidate_mean, oracle_coverage, selection_gap, pass@K for K∈{1,2,3,5,10} via `metrics_reference.pass_at_k`, slack, stop_reason, calls_by_role, flops) + `analysis/nb_lib/boot.py::cluster_boot_mean/paired_accuracy_test` and `calib.py::brier_decomposition` for the first-pass report (verified functions exist at `analysis/nb_lib/boot.py:137,186` and `calib.py:120`).

---

## 2. F-bank aliasing and the on-disk layout

### 2.1 Request store: `<run_root>/requests/<request_id[:2]>/<request_id>.json`
Content-addressed by `request_id` (§3.6). Record:
```json
{"schema_version":1,"request_id":"…","identity":{"study_id":"…","model_revision":"9216db57…","tokenizer_revision":"9216db57…","engine_digest":"vllm==0.21.0;dtype=bf16;tp=2;reasoning_parser=qwen3;profile=32B-long","input_hash":"…","decoding_hash":"…","semantic_seed_hex":"32hex","local_caps":{"max_tokens":8192},"hook_hash":"none"},
 "seed_key":["hle:668825f8…","main","Qwen3-32B@9216db57","0",0,"root",3,"stateless_bank"],"engine_seed":123,
 "model":{"hf_id":"Qwen/Qwen3-32B","served_model_name":"32B","profile":"32B-long","tp_size":2},
 "messages":[{"role":"user","content":"…"}],"chat_template_kwargs":{"enable_thinking":true},"chat_template_hash":"…","sampling":{"temperature":"0.6","top_p":"0.95","top_k":20,"min_p":"0.0","presence_penalty":"0.0","repetition_penalty":"1.0","max_tokens":8192},
 "prompt_token_ids":[…],"prompt_tokens":1532,
 "response":{"content":"…","reasoning":"…","reasoning_field_name":"reasoning","token_ids":[…],"completion_tokens":4210,"reasoning_tokens":3800,"finish_reason":"stop"},
 "flops":{"prefill":…, "decode":…, "total":…},
 "timing":{"submitted_at":…, "completed_at":…, "latency_s":…},"endpoint":{"host":"node3904","port":8001,"slurm_job_id":"…"},"attempts":1,
 "producer":{"cell_id":"F.BANK.32B.N1.B0.F00.e0.s003","run_id":"study_v4","slurm_job_id":"…"},
 "content_sha256":"sha256 of JCS(record without this field)"}
```
`publish` writes a same-directory temp file (`io.atomic_write_text` semantics: fsync) then `os.link(tmp, final)`; `FileExistsError` → the caller's freshly generated output is discarded (`alias_race` logged in the cell's `events.jsonl`) and the committed record is returned, so exactly one committed request exists per id (§10.4 "duplicate submissions attach to the same committed request"). Reads verify `content_sha256`; a corrupt file raises `ProtocolError` (never silently regenerated).

### 2.2 Cell directory: `<run_root>/cells/<cell_id>/`
```
items/<source_id>.json    one EpisodeResult (or bank) per item, atomic publish (io.write_json)
incomplete/<source_id>.json  InfraFailure records (deleted when the item later completes)
events.jsonl              append-only progress (io.append_jsonl): started/finished/aliased/failover/alias_race
meta.json                 written LAST, only when every item file exists: {cell_id, n_items, n_aliased_requests, n_generated_requests, flops_total, wall_s, code_version: io.git_commit(), config_sha256, frozen_sha256, host, slurm_job_id}
```
Item file (generate kind):
```json
{"schema_version":1,"cell":{…CellSpec…},"source_id":"…","domain":"hle","split":"main","status":"complete",
 "episode":{"episode_id":"sha256(cell_id,source_id)","calls":[{"request_id":"…","role":"root|revise|hub|worker|consumer","actor_slot":0,"step":0,"aliased":true,"prompt_tokens":1532,"completion_tokens":4210,"finish_reason":"stop","parse_status":"valid|TRUNCATED|…"}],
  "candidates":[{"candidate_id":"sha256(request_id)","request_id":"…","slot":0,"stage":"root|round3|final","valid":true,"failure_code":null,"candidate":{…§3.5 object…},"candidate_sha256":"…","vote_key":"D","grouping_mode":"mc_letter|exact_norm|ast|exact_source"}],
  "packets":[{"packet_id":"…","sender_slot":1,"recipient_slot":0,"round":1,"candidate_sha256":"…","recipient_tokens":2011,"final_partial":false,"truncated":{"evidence":true},"bytes_sha256":"…"}],
  "ledger":{"B_flops":…, "spent":…, "peak_reserved":…, "slack":…, "calls_admitted":31,"stop_reason":"budget","events":[{"t":…, "op":"reserve|debit|release","amount":…, "remaining":…, "owner":"round2"}]},
  "counters":{"assigned_roster":5,"unique_actors_used":5,"resets":0,"peak_live_contexts":5,"calls_by_role":{"root":5,"revise":25},"tokens_by_role_channel":{…},"complete_candidates":30},
  "selection":{"selector_id":"VOTE","pool_kind":"latest_slots|archive|native","pool_candidate_ids":[…],"selected_candidate_id":"…","winning_count":3,"tied_classes":1,"all_singleton":false,"no_valid_candidate":false,"grouping_mode":"exact_norm"},
  "native_final":{"candidate_id":"…","final_answer":"…","confidence":0.7}},
 "timing":{…},"worker":{"slurm_job_id":"…","host":"…"},"code_version":"…"}
```
Bank cells (module F) store `"bank":[10 CandidateRecords in draw order]` instead of `episode`; prefix selections live in the seal.

### 2.3 Why aliasing is automatic
`request_id` depends only on (prompt bytes, decoding, caps, model/tokenizer/engine pins, semantic seed). Semantic seed depends only on the `SeedKey`. The frozen SeedKey table makes intended aliases coincide (all with `split="main"`, `model_cell="Qwen3-32B@9216db57"`, `episode_rep=0`):

| Producer | prompt | SeedKey (actor_slot, purpose, step_slot, namespace) | Aliases |
|---|---|---|---|
| F bank cell framing f, draw k=0..9 | `render_root(task,f)` | (0, root, k, stateless_bank) | canonical |
| S_FRESH draw k | `render_root(task,00)` | (0, root, k, stateless_bank) | = F00[k] for k≤9; k≥10 new (§5.4, §6.2) |
| S_HISTORY draw 0 | `render_root(task,00)` | (0, root, 0, stateless_bank) | = F00[0] |
| IND_VOTE(N=5, 11) root s / optional draw k | `render_root(task,11)` | (0, root, k, stateless_bank) | = F11[k], k≤9 (§5.4 "first-five bank", §6.2 "first ten") |
| IND_VOTE(N, 00) in N panel | `render_root(task,00)` | (0, root, k, stateless_bank) | = F00[k] |
| DEC(N) root s / IND_PRIVATE_REVISION root s / DEC_ONE_ROUND root s | `render_root(task,NATIVE,dec_clause(N))` | (0, root, s, stateless_bank) | mutual alias among the three (§4.6); never F |
| DEGREE d, root i=0..8 | `render_root(task,00)` | (0, root, i, stateless_bank) | = F00[i] (§4.6 "must match exact request IDs") |
| E module episode_rep=e≥1 | same as above | episode_rep=e | fresh; e=0 = module A record (§5.6) |
| M panel (ckpt c) | same prompts | model_cell="Qwen3-{c}@rev" | fresh per checkpoint |

The cell code never "looks up the F bank"; it calls `store.get_or_generate(spec)`. If F cells ran first the record exists (`aliased=True`, no GPU, no ledger debit of physical work but the *logical* cost is still debited to the episode ledger, §6.5/§6.7 "logical uncached accounting"). If an A cell runs first, it generates and F later aliases it. Order only affects wall time. Dispatch F cells first anyway (tier-1 manifest lists F cells before A cells; the forked driver keeps manifest order).

A policy can't peek: `get_or_generate` is only called after `ledger.try_reserve` succeeded, so a bank hit never changes the admission decision (§6.7 "reveal a bank result only after its reservation is admitted").

---

## 3. Concurrency and resumability of a cell

`run_cell(cell, run_root, server_run_root, shard, stop_event)`:
1. `cfg = StudyConfig.frozen(run_root)` (generate cells require `FROZEN.yaml` with B0; F bank cells only need the study manifest — B is not applicable to fixed-opportunity banks).
2. `tasks = load_public_tasks(run_root)` filtered to `cell.items`; `todo = [t for t in cell.items if not items/<t>.json valid]`; if empty → write `meta.json` and return (idempotent re-run, driver-safe).
3. `pool = EndpointPool(server_run_root, ckpt.profile, shard)`; `pool.wait(300)` (uses `registry.wait_for_server`, which raises `TimeoutError` → exit code 2 so the chunk driver simply re-queues the cell later, as the old harness does).
4. `executor = ThreadPoolExecutor(max_workers=cell.max_inflight)`; `max_inflight` = 10 for F banks (all draws of one item in parallel), `N` for method cells (a symmetric round/batch runs in parallel), 8 for judge cells, 1 for `EVAL_BCB` (CPU-bound inside apptainer, the sbatch asks for 2 CPUs).
5. Items run **sequentially in rank order**; inside an item the policy calls `ctx.generate_group(specs)` which reserves once, maps `store.get_or_generate` over the executor, waits for all, debits actual, returns in order. Only one ledger per item and all reservations happen in the policy thread, so there is no cross-thread ledger race; a `threading.Lock` still guards the ledger for safety (§10.2 atomic reserve).
6. After the policy returns, `io.write_json(items/<sid>.json, result)` (same-directory temp + fsync + `os.replace`, verified in `experiment/io.py:92-118`) — the atomic per-item publish. Request records were already committed one by one, so a crash mid-item loses only harness bookkeeping; on restart the same prompts hash to the same ids and every already-generated call is a store hit (adaptive policies replay deterministically because their prompts are pure functions of committed parent records, §6.7).
7. Failure classes: `InfraFailure` → write `incomplete/<sid>.json`, continue with the next item; `ContextFailure` inside a policy → typed context-failure record for that opportunity, episode continues under the failure rule (§4.3); `ProtocolError` (budget overshoot, prompt-token mismatch, corrupt store) → stop the cell immediately with exit 4 and write `SUSPENDED.json` (§6.5 "suspend the affected cell").
8. `SIGUSR1` (`--signal=B:USR1@1200` in the template) sets `stop_event`; the loop finishes the in-flight group and the current item, then exits 0 **without** `meta.json` → the driver re-queues; nothing is duplicated.
9. Endpoint failover: `VllmChatClient.generate` picks `pool.pick()` per attempt; a connection/timeout/5xx failure calls `pool.report_failure(entry)`; after 2 consecutive failures on one endpoint the pool rotates and re-runs `list_live_servers` (which uses Slurm allocation authority for job-backed entries and `/health` probing otherwise, `registry.py:1722-1789`). Retries are capped at 2 per request (§10.4). Per-request timeout 1,800 s (8,192 tokens at loaded-server speeds).
10. `meta.json` is written only when `len(todo_remaining)==0 and incomplete/ is empty`; the driver's `_chunk_complete` checks `cells/<cell_id>/meta.json` exactly as today.

Throughput knobs (brief: 6–11 concurrent cells per endpoint): tier 1 with 8 `32B-long` servers → `--throttle 80` on the array; each cell holds ≤10 streams → ≤800 concurrent sequences fleet-wide (vLLM default `max_num_seqs=256` per server).

---

## 4. FLOP oracle (analytic, dense Qwen3, multiply-add = 2 FLOPs; §6.5, §10.2)

Per checkpoint constants from `config.json` (`d`=hidden_size, `d_ff`=intermediate_size, `n`=num_hidden_layers, `H`=num_attention_heads, `Hkv`=num_key_value_heads, `hd`=head_dim=128, `V`=151,936):

- `C_lin = n·[ 2·d·(H·hd + 2·Hkv·hd)  (q,k,v proj) + 2·(H·hd)·d (o proj) + 6·d·d_ff (SwiGLU gate/up/down) ] + 2·d·V (vocab projection)` — FLOPs per token independent of position; embedding lookup counted as 0 (published convention). Q/K norms and RMSNorms are O(d) and omitted (documented).
- `C_attn = n · 4 · H · hd` — FLOPs per (query token, attended key) for QKᵀ and PV.
- Token at 1-indexed position p costs `C_lin + C_attn·p`. Hence `F_prefill(L) = C_lin·L + C_attn·L(L+1)/2`, `F_decode(ctx) = C_lin + C_attn·ctx`, and `F(L,T) = F_prefill(L) + Σ_{t=1..T} F_decode(L+t) = C_lin·S + C_attn·S(S+1)/2` with `S = L+T`.

| ckpt | n | d | H | Hkv | d_ff | C_lin (GFLOP/token) | C_attn (FLOP/token/key) |
|---|---|---|---|---|---|---|---|
| 4B | 36 | 2560 | 32 | 8 | 9728 | 8.04 | 589,824 |
| 8B | 36 | 4096 | 32 | 8 | 12288 | 15.14 | 589,824 |
| 14B | 40 | 5120 | 40 | 8 | 17408 | 27.98 | 819,200 |
| 32B | 64 | 5120 | 64 | 8 | 25600 | 63.97 | 2,097,152 |

(32B check: 63.97 GFLOP/token ≈ 2P with P=32.8B — consistent with the `2P×tokens` planning approximation, §6.5.)

B0 (pre-freeze, §6.5): minimal full-cap paths at N=5 on 32B with `L_root_max = 6,144` (task envelope 4,096 + wrapper/schema reserve 2,048) and `T = 8,192`: one root call = `F(6144, 8192) ≈ 0.917 + 0.216 = 1.13 PFLOP`; IND_VOTE and DEC need 5 roots = **5.66 PFLOP = B0**; S_FRESH/S_HISTORY/CEN_FLAT need 1.13; VOTE is deterministic (0). So B1=5.66, B2=11.3, **B4=22.6**, B8=45.3 PFLOP (to be re-derived by `profile.py` from the frozen constants; these are hand checks). Runtime reservations use the *known* rendered prompt length (§6.5 step 1), e.g. a realized DEC N=5 revise call (≈10k prompt) reserves ≈1.5 PFLOP so a full round reserves ≈7.5 PFLOP and B4 admits ≈6 rounds (~35 calls); a realized 32B root (1.5k prompt, 3.5k output) debits ≈0.35 PFLOP so IND/S_FRESH at B4 admit ≈60 draws before the 64-call cap. The N=9/B1 feasibility check (§6.5) for DEC/IND is 9 roots = 10.2 PFLOP > B1 → **N=9 at B1 is infeasible under this B0 definition**; the N panel is B4-only (10.2 ≤ 22.6, and one N=9 revision round ≈19 PFLOP fits at most once). Report in the pilot and freeze before tier 2; do not raise B0 after outcomes.

---

## 5. Reuse from the repo (verified by reading)

| Need | Reuse | Notes |
|---|---|---|
| Durable writes | `src/agents_scaling/experiment/io.py`: `atomic_write_text`, `write_json`, `append_jsonl`, `results_root`, `run_dir`, `git_commit` | `run_dir` creates the dir; `io.write_json` uses `sort_keys=True, indent=2` — fine for item/meta files; the request store uses its own compact writer + `os.link` |
| Endpoint discovery | `serving/registry.py`: `list_live_servers(run_root, profile)`, `lookup_server(run_root, profile, shard=…, live_only=True)`, `wait_for_server(run_root, profile, shard, timeout_s=300)`, `ServerEntry.base_url` | first positional arg is the registry key = profile name (`32B-long`); `entry_matches_profile` requires the profile fields in the JSON |
| Profiles | `serving/profiles.py`: `get_serving_profile("32B-long")` → `hf_id, tp_size=2, max_model_len=40960, served_model_name="32B"`; `4B-long/8B-long/14B-long` tp=1 | `served_model_name` is what goes in `model=` of the chat call |
| Server launch | `python -m agents_scaling.serving.launch_server --model-size 32B --profile 32B-long --run-root <run_root> --partition pi_manoli --gpu-type a100 --time 7-00:00:00 --replica N` (`launch_server.main`, `submit(...)`); template `slurm/serve_qwen.sbatch.tmpl` (bf16 default, `--reasoning-parser qwen3`, `--gpu-memory-utilization 0.90`, `--max-logprobs 20`) | unchanged; add `VLLM_CACHE_ROOT/TORCHINDUCTOR_CACHE_DIR/XDG_CACHE_HOME` exports via the environment when launching (brief: HOME quota) — check the template's `--export=NONE` and pass through `hf-home`-style if needed |
| Fleet hold | `slurm/keepalive.py --allow-legacy-fleet --run-id study_v4 --spec "32B-long:4:pi_manoli:7-00:00:00,32B-long:3:pi_tpoggio:7-00:00:00,14B-long:1:pi_tpoggio:7-00:00:00,…" --interval 600` as a self-resubmitting loop job | `slurm/launch_loops.py` submits BOTH the legacy driver and keepalive; the driver would crash on a study config → write `slurm/study_loops.py` (~60 lines, fork of `_render_submit`) that renders `loop_keepalive.sbatch.tmpl` verbatim and the new `study_loop_driver.sbatch.tmpl` |
| Cell dispatch | fork `slurm/launch_chunked.py` → `slurm/study_launch_chunked.py`: keep `_drive`, `_my_submitted_count`, `_my_total_count` verbatim; replace `load_sweep`/`ExperimentCell` with `--cells-file` (pre-built by `study.cells`) and `_chunk_complete` reading `cells[idx]["cell_id"]`; render `slurm/study_cell_array.sbatch.tmpl` (copy of `run_cell_array.sbatch.tmpl` with `--mem={MEM}`, `--time={TIME}`, `--cpus-per-task={CPUS}`, `--job-name=asys-cells-{RUN_ID}` and `exec python -m agents_scaling.study.run_one --run-id "{RUN_ID}" --server-run-id "{SERVER_RUN_ID}" --cells-file "{CELLS_FILE}" --index "$SLURM_ARRAY_TASK_ID"`) | the array still sources `slurm/common.sh` (sets `HF_HOME`, `ASYS_RESULTS_ROOT`, activates `asys_env`) |
| Reference metrics | handoff `metrics_reference.py` vendored as `study/metrics_reference.py` (+ its `tests/test_metrics_reference.py`) | `plurality_vote`, `judge_best`, `pass_at_k`, `evaluate_sealed_selection` |
| Analysis | `analysis/nb_lib/boot.py` (`cluster_boot_mean`, `paired_accuracy_test`, `ci`), `calib.py` (`brier_decomposition`, `cox_recalibration`, `ece_*`) | first-pass report only; the 20,000-resample six-family machinery (§9) is post-deadline |
| vLLM call shape | copy the `extra_body` construction from `serving/client.py:1081-1090,1162-1171` and the `reasoning`/`reasoning_content` dual read at `client.py:770-815` | do NOT import `LogprobClient` (rejects `max_tokens != 4096+budget`) |

Exact commands (in order):
```bash
# T0+0h  fleet (login node)
python slurm/study_loops.py --run-id study_v4 --spec "32B-long:4:pi_manoli:7-00:00:00,32B-long:3:pi_tpoggio:7-00:00:00,14B-long:1:pi_tpoggio:7-00:00:00" --keepalive-only
apptainer pull /orcd/data/tpoggio/001/mabdel03/containers/bigcodebench-evaluate-v0.2.4.sif docker://bigcodebench/bigcodebench-evaluate:v0.2.4
# T0+1h  data export (login node, network)
python -m agents_scaling.study.data.export --run-id study_v4 --config configs/study_v4.yaml --hle-shards <scratchpad>/hle_shards
# T0+7h  pilot (dev split, 20 items)
python -m agents_scaling.study.cells --run-id study_v4 --tier pilot --out cells_pilot.json
python slurm/study_launch_chunked.py --allow-legacy-admission --run-id study_v4 --cells-file cells_pilot.json --chunk-size 60 --throttle 40 --submit-cap 400 --qos-limit 448 --cell-partition mit_preemptable --cell-time 12:00:00 --cell-mem 4G --cpus 1
python -m agents_scaling.study.resources.profile --run-id study_v4 --config configs/study_v4.yaml --from-pilot cells_pilot.json   # writes <run_root>/FROZEN.yaml (B0)
# T0+9h  tier 1
python -m agents_scaling.study.cells --run-id study_v4 --tier 1 --out cells_tier1.json     # F cells first, then A
python slurm/study_launch_chunked.py … --cells-file cells_tier1.json --chunk-size 400 --throttle 80
# after tier-1 gen completes
python -m agents_scaling.study.seal --run-id study_v4 --cells-file cells_tier1.json
python -m agents_scaling.study.cells --run-id study_v4 --tier 1-eval --out cells_tier1_eval.json   # JUDGE_HLE, EVAL_BCB (cpus 2), JUDGE_BEST
python slurm/study_launch_chunked.py … --cells-file cells_tier1_eval.json --throttle 60
python -m agents_scaling.study.aggregate --run-id study_v4
```
One run root `study_v4` holds the fleet registry, the shared request store and every tier's cells (aliases across tiers require the single store).

---

## 6. Work packages (5 coding agents + 1 lead), interfaces, tests, hours

**WP0 — Lead, T+0:00→T+0:45 (blocking).** `types.py`, `identity.py`, `config.py` + `configs/study_v4.yaml`, vendored `metrics_reference.py`, template copies + `PROMPT_HASHES.json`, `tests/study/conftest.py` (tmp run_root, fake tokenizer stub that counts whitespace tokens for speed but exposes the real API, registry-writing helper), the `RequestRecord`/`EpisodeResult` JSON examples above as fixtures. Everyone rebases on this before writing code. Tests: JCS byte examples, seed/request_id golden values, config freeze rejection.

**WP1 — Data + prompts (agent 1, 4.5 h).** `data/*`, `prompts/render.py`, `inference/tokens.py`. Tests: eligibility fixture rows (image row, Uncertain row, duplicate id, over-envelope row, Revision vs Gold strata), balanced nested prefixes, byte-exact rendering of the four framing cells against golden files (clause placement and empty lines), envelope check raises, protected-import firewall grep test, export CLI on fixtures.

**WP2 — Inference + store + fake server (agent 2, 4.5 h).** `inference/client.py`, `inference/store.py`, `tests/study/fake_vllm.py` (a `ThreadingHTTPServer` on 127.0.0.1 implementing `POST /v1/chat/completions` (deterministic content keyed by `sha256(JCS(messages)+seed)`: candidate JSON with `final_answer` drawn from a small set so votes are non-degenerate, optional invalid outputs, `reasoning` when `enable_thinking`, `prompt_token_ids`/`token_ids`/`usage`/`finish_reason`), `GET /health`, `GET /v1/models`; fault injection `fail_next(n)` / `hang(seconds)`); `register_fake(run_root, profile, port)` writes `servers/<profile>/127.0.0.1_<port>.json` with the profile fields. Tests: round trip incl. reasoning field name, prompt-token-id mismatch → `ProtocolError`, two endpoints with one killed → failover + `report_failure` rotation, retry cap → `InfraFailure`, `publish` O_EXCL race (two threads same id → one `was_new`), corrupt record → `ProtocolError`, `get_or_generate` alias hit makes zero HTTP calls.

**WP3 — Contracts (agent 3, 4.5 h).** `parse/candidate.py`, `parse/coordinator.py`, `packets.py`, `selection/normalize.py`, `selection/vote.py`, `selection/judge_best.py`. Tests: 25 strict-parse cases (fence, double fence, duplicate keys, NaN/Infinity, trailing text, extra key, bool confidence, >32 evidence, uncertainty enum, `length` finish), sentinel bytes exact, coordinator cross-record errors, packet cap under adversarial escapes/emoji/CJK (serialized ≤2,048 by the tokenizer stub; final excerpt ≤1,024 + `partial`), priority allocation order, unavailable packet, MC key extraction table, exact-answer normalization table (numbers, `\boxed{}`, `$..$`, case, trailing period), AST grouping equal for whitespace/comment variants and unequal for identifier changes, `exact_source` mode on SyntaxError, plurality multiplicity and blind ties (reuse `tests/test_metrics_reference.py`).

**WP4 — Resources + policies (agent 4, 5 h; depends on WP2/WP3 interfaces only — code against `types.py` and a stub `EpisodeContext` until hour 3).** `resources/oracle.py`, `resources/broker.py`, `resources/profile.py`, `policies/*`. Tests: oracle equals hand formula for (L,T)∈{(1,1),(100,10),(6144,8192)} on all four archs; reservation no-fit vs fit boundary; atomic group reserve under 5 threads never overshoots; debit releases headroom; DEC full-round stop at exact budget boundary, `N*(1+r)≤64`, invalid own → sentinel bytes, packets from previous round only (no current-round leakage: assert prompt bytes of round r contain only round r−1 hashes); CEN slot validation and reserved final call after a delegation at cycle 8; IND blind round-robin and indivisible optional draw; S_HISTORY retains only the previous candidate; end-to-end each policy on the fake server with `B` small enough to stop on budget and large enough to stop on the call cap (stop reasons asserted); no policy imports `evaluation` or `data.load_protected_*`.

**WP5 — Evaluation + cells + runner + Slurm (agent 5, 5 h; depends on all, uses fakes early).** `evaluation/*`, `cells.py`, `runner.py`, `run_one.py`, `seal.py`, `aggregate.py`, the two Slurm forks + `study_loops.py`. Tests: manifest determinism (same sha twice), tier nesting (tier-2 items ⊂ tier-1 items per domain), engine-seed uniqueness, F-before-A ordering; runner skip-done/resume after simulated kill mid-item (request store hits, no extra HTTP calls), `incomplete/` handling and exit codes, SIGUSR1 stop without `meta.json`; HLE judge parse (yes/no/ambiguous → ambiguous scored incorrect); MC exact scoring; BCB evaluator with `--container none` test mode running the driver in-process on a canned BCB task (pass/fail/timeout) and a shell shim asserting the exact `apptainer exec` argv; seal refuses to evaluate unsealed pools; aggregate produces pass@K matching `metrics_reference`.

**Integration (lead, T+5:00→T+7:00).** Merge order WP1→WP2→WP3 at T+3:00 (interfaces frozen), WP4 at T+5:00, WP5 at T+5:30. End-to-end on the fake server in `tests/study/test_e2e.py`: export fixtures → `cells --tier pilot` → run F cells → run A cells (assert `n_aliased_requests == 10·(items)` for S_FRESH and `≥5·items` for IND_VOTE) → seal → judge/eval cells → aggregate; then the real dev pilot (20 items) against the live fleet, `profile.py` → `FROZEN.yaml`, and tier-1 manifest.

Interfaces to agree on before coding (in WP0): `RequestSpec/RequestRecord` fields and JSON, `SeedKey` table, `CandidateRecord/Packet`, `EpisodeContext.generate_group` signature, `CellSpec` and `cell_id` format, item-file JSON, registry JSON needed by `entry_matches_profile`, exit codes (0 done/stopped, 2 no server, 3 incomplete items, 4 suspended).

---

## 7. Compute sanity (unchanged from brief, refined by aliasing)
Tier 1 on 400 main items: F = 16,000 gens; A ≈ S_FRESH ~50 new + S_HISTORY ~60 + IND ~50 new + DEC ~35 + CEN ~20 ≈ 215 gens/item → 86,000 gens; ≈ 102k gens × 3,500 tok ≈ 357M tokens; at 8×900 tok/s ≈ 13.8 h. HLE judge ≈ 200 items × ~190 unique answers ≈ 38k short thinking-off calls (≈1–2 h fleet-wide); BCB eval ≈ 38k unittest runs ≈ 40 CPU-h across 2-CPU cells.

### Critical Files for Implementation
- `/orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/study/types.py` (new; the shared contract) together with `identity.py` and `inference/store.py` (aliasing semantics)
- `/orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/study/policies/dec.py` and `resources/broker.py` (new; the budget/round logic that determines what B4 actually admits)
- `/orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/study/runner.py` + `run_one.py` (new; concurrency/resume/failover, Slurm entry)
- `/orcd/data/tpoggio/001/mabdel03/agents_scaling/src/agents_scaling/serving/registry.py` (reused: `list_live_servers`, `lookup_server`, `wait_for_server`, `ServerEntry`)
- `/orcd/data/tpoggio/001/mabdel03/agents_scaling/slurm/launch_chunked.py` and `/orcd/data/tpoggio/001/mabdel03/agents_scaling/slurm/run_cell_array.sbatch.tmpl` (forked to `study_launch_chunked.py` / `study_cell_array.sbatch.tmpl`; `experiment/io.py` reused as-is)

## Disagreements
- Aliasing scope: the brief says 'IND_VOTE N=5 first 5 roots = F11 bank' and 'S_FRESH first 10 = F00'. Under the stateless seed rule (§4.2: actor_slot=0, step_slot=draw ordinal) and §6.2 ('alias its first ten requests only when prompts, caps, seeds ... match exactly'), IND_VOTE's optional fresh draws 5..9 also alias F11[5..9], and IND in the N panel (00 wording) aliases F00 — this is automatic under content-addressed request_ids and saves ~5 gens/item; I designed the SeedKey table so these coincide rather than limiting aliasing to the first 5.
- 'B4 will typically bind at the 64-call cap': true for S_FRESH/IND (≈60 realized draws), but with the spec's reserve-before-launch rule (§6.5) DEC N=5 reserves a full round (≈5 × 1.5 PFLOP at realized prompt lengths) and will stop on *budget* after ~6 rounds (~35 calls), and DEC/IND at N=9 are infeasible at B1 (9 roots × 1.13 PFLOP > B0=5.66) and admit at most one revision round at B4. The pilot must report this; it is a property of the frozen policy, not a defect. Tier-3 'A-budget B1' cells at N=9 should therefore not be planned.
- launch_chunked.py and run_cell_array.sbatch.tmpl cannot be reused verbatim: `_chunk_complete` calls `ExperimentCell.from_dict(...).cell_id` (study cells are not ExperimentCells) and the template hardcodes `agents_scaling.experiment.run_one` and has no `{MEM}/{TIME}` placeholders. Plan forks two files (~300 lines total) instead of editing the frozen legacy driver; likewise `launch_loops.py` submits the legacy driver too, so a `study_loops.py` keepalive-only launcher is needed.
- The brief's 'runs items sequentially with per-item concurrency = N slots' is kept for method cells, but F bank cells should run all 10 draws of an item in parallel and judge cells 8-wide; a single `max_inflight` per CellSpec covers both.
- The scratchpad `hle_verified.parquet` is a 134-row Gold-only sample, not the 2,500-row set; the adapter must read `hle_shards/*.parquet` (or re-download the pinned revision on the login node).
- `jsonschema` is not installed in asys_env (and `rlms` is not either), so schema files from the handoff are used as documentation and test fixtures only; all wire validation is hand-written strict code, which the spec requires anyway (§3.5, §11.4).

## Open questions
- Confirm one run root `study_v4` for fleet registry + shared request store + all tiers (needed so cross-tier aliases resolve), with the keepalive `--run-id study_v4`.
- Judge model placement: Qwen3-32B thinking-off HLE/JUDGE_BEST calls on the same `32B-long` fleet (contends with generation) — or dedicate the `14B-long:1` server from the fleet spec to something else? What is the 14B-long replica in the brief's spec intended for (M panel)?
- Dev-frozen rules to approve before the pilot: (a) strip one enclosing ``` fence inside `final_answer` for code tasks before AST/eval; (b) MC letter regex + exact-answer normalization table; (c) HLE judge greedy decoding, 1,024-token cap, `[response]` = final_answer only; (d) BCB timeout 120 s / 4 GB rlimit per candidate.
- Root prompt maximum for B0 (`root_prompt_max_tokens=6144` = 4,096 task envelope + 2,048 wrapper): acceptable, or should B0 be resolved from the actual maximum rendered root prompt over the frozen main set (smaller B0, tighter budgets)?
- Apptainer image location `/orcd/data/tpoggio/001/mabdel03/containers/bigcodebench-evaluate-v0.2.4.sif` and whether the login node can pull 9.27 GB from Docker Hub now (compute nodes should never pull).
- Is a single-user `study_seed`/`split_salt` generated now and committed in `configs/study_v4.yaml` acceptable as the frozen randomness (§3.6), given no separate operator will hold it?

## Risks
- vLLM 0.21 response field names (`reasoning` vs `reasoning_content`, `prompt_token_ids`, `token_ids`) differ from what the fake server emits, so unit tests pass but the pilot fails on the first call. → WP2 runs one real smoke request against a live 32B-long endpoint as soon as the fleet is up (T0+1h) and encodes the observed shape into the fake server; the client accepts both reasoning aliases exactly like the old client.
- Per-request `seed` on vLLM does not give bit-identical replays across batch compositions, so a re-generated aliased request (e.g. after a corrupt store record) would differ. → Never regenerate a committed request; a corrupt record is a ProtocolError that suspends the cell. Aliasing is by stored bytes, not by re-sampling; record engine_seed and document that reproducibility is at the record level.
- Full-cap reservation makes DEC/CEN budgets bind earlier than the brief's mental model, changing expected call counts and wall time. → profile.py prints the admitted-call schedule per method from pilot telemetry before tier-1 dispatch; B0 rule is frozen before outcomes and reported with slack/stop reasons (§6.5).
- Packet compilation requires the real Qwen tokenizer on every CPU cell (transformers 5.9 load ~2-5 s, plus repeated tokenization inside the shrink loop). → Load once per cell process; cache token counts per excerpt; binary search on excerpt length bounds the number of tokenizations to ~12 per field.
- BCB apptainer evaluation is slow or the image's Python cannot import a candidate's library (image pins 73 py3.10 packages). → One `apptainer exec` per cell with an in-container driver loop (no per-candidate container start), dedupe identical code, 2 CPUs per eval cell, test the image on the dev BCB items during the pilot; missing-library failures are candidate failures under the frozen rule.
- Shared request store hot directories (400k files) on the group filesystem slow `os.link`/stat. → 256-way fan-out by id prefix, no directory listings in the hot path (get = stat one path), meta counts kept in cell files not in the store.
- Six-hour implementation window is tight; WP4 (policies) is the largest and depends on WP2/WP3 interfaces. → WP0 freezes `types.py` first; WP4 codes against a stub EpisodeContext and the fake server from hour 1; S_FRESH/IND_VOTE are implemented first (needed for tier 1 F/A aliasing tests), DEC and CEN_FLAT second, DEGREE/IND_PRIVATE_REVISION last (tier 2).
- Fleet capacity: `ou_bcs_normal` 1-day walltime servers die mid-tier and cells time out waiting for endpoints. → EndpointPool refreshes from the registry every 5 min and on failure; `wait_for_server` 300 s fail-fast + driver re-queue is already the proven pattern; keepalive re-submits replicas.