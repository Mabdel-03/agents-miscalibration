# ARD36 WP0 contract (frozen)

Study: **Agents Round Dynamics 36h** — "Who Should Listen? Predicting and Controlling
Correctness Dynamics in Multi-Agent Reasoning".

- Specification: `expermt_agent_specs/AgentsRoundV1/Agents_Round_Dynamics_36h_Spec.md` (880 lines, md5 `9c261a49…`)
- Execution plan: `~/.claude/plans/expermt-agent-specs-agents-round-dynamic-woolly-naur.md`
- Frozen constants: **`src/agents_scaling/ard/contract.py`** — import it, never re-declare its values.

Branch `ard36`. Run root `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36`.

## 0. Rules that bind every work package

1. **`src/agents_scaling/study/` is frozen.** Import from it. The only permitted edits are the
   additive ones in §4.2 of the plan, and they belong to **WP1 alone**. The existing 541
   `tests/study` tests must stay green.
2. **Gold never enters a runtime module.** `ard/items_build.py` is the sole importer of
   `agents_scaling.benchmarks.loaders` / `datasets`. Runtime code reads the frozen public
   parquet through `ard/items.py`. The firewall test is transitive over the import closure.
3. **Anything sent to vLLM enters the request identity**, or two different requests silently
   share one cache entry. Budgets go into `Decoding`; floats are forbidden by JCS.
4. **Every observable is emitted at the round grain** for every protocol and cohort. See plan §5.8.
5. New code lives under `src/agents_scaling/ard/` and `tests/ard/`. Do not edit another WP's files.

## 1. File ownership (disjoint; no two packages touch the same file)

| WP | Owns | Depends on |
|---|---|---|
| WP1 inference | `ard/client.py`, `ard/decoding.py`; the additive edits to `study/types.py`, `serving/profiles.py`, `slurm/serve_qwen.sbatch.tmpl`, `serving/launch_server.py`, `configs/model_contracts.v1.json`, `configs/ard36.yaml`; `tests/study/fake_vllm.py` extensions | contract |
| WP2 protocols | `ard/report.py`, `ard/packets.py`, `ard/vote.py`, `ard/episode.py`, `ard/protocols.py`, `ard/states.py`, `ard/forks.py`, `ard/suffix.py` | WP1 interface |
| WP3 data | `ard/items_build.py`, `ard/items.py`, `ard/graders/` | contract |
| WP4 dispatch | `ard/types.py`, `ard/cells.py`, `ard/kinds.py`, `ard/runner.py`, `ard/run_one.py`, `ard/compact.py`, `ard/resources.py`; `slurm/ard36_*.{py,sh,tmpl}` | WP1, WP2, WP3 interfaces |
| WP5 controller | `ard/features.py`, `ard/controller.py`, `ard/selectors.py`, `ard/receiver.py` | WP2 interface |
| WP6 neural | `ard/neural/` | WP2 interface |
| WP7 analysis | `ard/legacy_e0.py`, `ard/stats.py`, `ard/analysis.py`, `ard/gates.py`, `ard/reference_metrics.py` | contract |

## 2. Frozen interfaces

Code against these signatures. Where a dependency is not yet built, stub it.

```python
# ---- WP1: one logical decision == one chat request ------------------------
@dataclass(frozen=True)
class DecisionRequest:
    messages: tuple[dict, ...]      # rendered chat messages
    checkpoint: str                 # "8B" | "32B"
    thinking_tokens: int            # 1024 | 4096
    seed_key: SeedKey               # study.types.SeedKey, namespaces per contract
    purpose: str                    # contract.PURPOSE_*

@dataclass(frozen=True)
class DecisionRecord:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    thinking_tokens: int            # ids strictly between <think> and the first </think>
    final_tokens: int               # ids after </think>
    boundary_injected: bool         # thinking_tokens == budget (sampler forced </think>)
    cap_hit: bool
    final_overrun: bool             # more than FINAL_TOKENS generated after </think>
    final_text: str                 # the post-</think> span, first FINAL_TOKENS tokens
    answer_logprob: float | None    # mean logprob over the final span
    prompt_tokens: int
    cached_prefill_tokens: int      # usage.prompt_tokens_details.cached_tokens
    finish_reason: str
    latency_s: float
    endpoint_id: str
    flops: dict                     # {"prefill":int,"decode":int,"total":int,"oracle":str}

def generate_decision(req: DecisionRequest, *, store, pool, cost_fn) -> DecisionRecord: ...
```

WP1 must attach `oracle_factory=FlopOracle.from_checkpoint` and `make_cost_fn` exactly as
`study/runner.py:254-261` does. Without it `flops` is the `None` placeholder and the run
produces no FLOP number at all.

```python
# ---- WP2: report, packet, episode ----------------------------------------
@dataclass(frozen=True)
class Report:
    final_answer: str | None
    evidence: str | None
    confidence: float | None        # Optional: missing is a missing feature, never 0.5
    parse_status: ParseStatus
    content_sha256: str

def parse_report(final_text: str, finish_reason: str,
                 completion_ids: tuple[int, ...]) -> Report: ...
    # reads ONLY the post-</think> span; never searches the thinking span for an answer

def packet_plan(state: RoundState, protocol: Protocol, *, round_: int,
                protected_slot: int | None = None) -> dict[int, tuple[Packet, ...]]: ...
    # PRIVATE/protected slot -> (); PEER -> the other four previous-round reports;
    # BALANCED -> one per canonical class (+ one typed "unparsed peer" entry);
    # HUB5 -> leader reads four, then workers read the current leader report.

def run_episode(item, protocol, checkpoint, *, rounds=3, ...) -> Episode: ...
    # asserts: reads only round t-1 unless HUB5; exactly N*rounds decisions even when
    # unanimous; archive coverage never falls.
```

```python
# ---- WP3: items and grading (evaluator namespace only) --------------------
def build_items(out_dir, *, split_salt, study_seed) -> None: ...   # public parquet + protected gold
def load_ard_tasks(run_root) -> dict[str, PublicTask]: ...         # runtime reader, public only
def canonical_class(family: Family, answer: str) -> str | None: ...  # gold-independent
def grade(family: Family, answer: str, item_id: str) -> GradeResult: ...
    # GradeResult.status in {"correct","incorrect","ambiguous","timeout"}
```

```python
# ---- WP5: controller ------------------------------------------------------
FEATURES_V1: tuple[str, ...]        # <= 20 names, frozen, none derived from gold
def features_from_state(state, agent_slot, *, round_) -> dict[str, float]: ...
def fit_controller(train_frame, val_frame) -> FrozenController: ...   # FrozenG-style hash
def choose_protected(controller, state, *, round_, episode_id) -> int: ...  # argmax u, HMAC tie
```

## 3. Facts already verified (do not re-derive)

- vLLM 0.21 forces `</think>` by a logits boost once the thinking count reaches the budget,
  so `boundary_injected == (thinking_tokens == budget)` is exact. `--reasoning-parser qwen3`
  alone creates the required `reasoning_config`. Prefix caching is on by default, block 16.
- Both pinned snapshots share one tokenizer blob: `<think>`=151667, `</think>`=151668.
  Qwen3-8B has 36 layers, so capture blocks are (8, 17, 26).
- `study/selection/vote.py::vote()` re-keys every record through the frozen `normalize.vote_key`
  dispatch and raises on a caller-supplied key. ARD36 therefore votes through `ard/vote.py`
  calling `metrics_reference.plurality_vote` with its own canonical classes.
- `study/packets.py::compile_candidate_packet` cannot carry an ARD report: `confidence` is in
  the frozen field priority and `sender_slot` must be a non-negative int. Wrap the byte
  primitives instead.
- `study/runner.py` unconditionally calls `load_public_tasks`, which rejects any split outside
  dev/main/reserve. `ard/runner.py` composes `CellRunner` with an injected task loader.
- `study/stats.py` is hard-wired to `DOMAINS=("hle","bcb")` and returns `p=1.0` silently for
  unknown domain keys. `ard/stats.py` vendors it with `domains`/`weights`/`n_primary` as arguments.
- `depends_on` is advisory today (`runner.py:285-287` logs and proceeds). Hard gating for
  `controller:` / `state:` prefixes is new plumbing across five files.

## 4. Definition of done, per work package

Every WP lands with: its modules, its `tests/ard/` group passing against the fake server, and no
regression in `tests/study`. The offline dry run (`scripts/ard36_dry_run.py`, WP4) must drive its
cell kind end to end on four synthetic items before the WP is considered complete.

Canonical test command:

```bash
PYTHONPATH=src /orcd/home/002/mabdel03/conda_envs/asys_env/bin/python -m pytest tests/ard tests/study -q
```
