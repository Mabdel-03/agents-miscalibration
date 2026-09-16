# nextdyn WP0 contract (frozen)

Study: **task_conditioned_dynamics_v1** — "Task-conditioned multi-agent dynamics" (spec v1.0,
2026-09-15).

- Specification: `expermt_agent_specs/taskCondAgentDynV1/task_conditioned_agent_dynamics_experiment_spec_v1_0.md`
- Approved plan: `~/.claude/plans/please-flesh-out-the-velvet-brooks.md` (the plan wins over
  every design note where they disagree)
- Parametric contract: **`src/agents_scaling/nextdyn/contract.py`** — the *shapes* are frozen
  here; every *value* comes from `configs/nextdyn/study.yaml` through `config.load_study()`.
  No experimental number is a module constant anywhere in the package.

Branch `nextdyn` (off `ard36`). Run root
`$ASYS_RESULTS_ROOT/task_conditioned_dynamics_v1`
(= `/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/task_conditioned_dynamics_v1`);
never the repo, never `/orcd/scratch` (the compiler refuses any configured scratch path).

## 0. Rules that bind every work package (plan §1.1; pinned by `tests/nextdyn/test_guards.py`)

| Rule | ARD36 defect prevented | Mechanism |
|---|---|---|
| **R1** Every fact lives in exactly one module. Table columns + keys are the row dataclasses in `tables.py` (schemas derived from `dataclasses.fields`); token contract in `contract.TokenContract`; output shape in `schema.PublicOutput`; frozen manifest in `freeze.FrozenManifest` (same class writes and verifies); dependency facts in `deps.py`; DAG in `plan.py`. | D-05, D-08, `latency` vs `latency_s` | consumers import the dataclass; dict-key access to item files forbidden by AST test; `test_every_row_dataclass_is_the_only_declaration_of_its_table_columns` |
| **R2** Every declared knob is consumed, checked **statically**: `config.py` compiles YAML into frozen dataclasses (unknown keys rejected) and declares `CONSUMERS: {leaf key path → consuming module or slurm template}`; `test_every_config_leaf_has_a_declared_consumer_that_reads_it` walks the package AST for attribute reads; template-consumed keys are checked by a render test. No runtime read-recording. | D-03 `max_inflight` | AST walk |
| **R3** Every producer has a consumer test in the same change: `test_pipeline_dry_run.py` drives every cell kind on 4 synthetic items through the fake vLLM, materializes, asserts keys, runs `analyze` in test mode. A WP is not done until its kind runs there. | "both ends, no middle" | `TableSchema.producer/consumers` importable names |
| **R4** `cell_kind` is the first component of every id. | D-11 | `ids.episode_id` / `ids.bank_draw_id` refuse the other kind |
| **R5** Nothing sent to vLLM is outside the request identity; nothing about the condition is inside the seed; no template names a condition or describes the graph. | contract rule 3; D4 | `test_request_cache_invalidates_on_model_prompt_seed_or_parent_change`, `test_no_template_contains_a_protocol_label`, byte-equality tests of aliasing pairs |
| **R6** Rows carrying physical ids are built at one site: `telemetry.decision_row()` is the only `DecisionRow` constructor. | D-02 recurrence | AST test |
| **R7** Gold never enters the generation import closure: `grading`, `construct`, `items_build`, `datasets`, `inventory`, `e0`, `analysis` are WRITER_SIDE; runtime reads only `items.py` / `bundles.py`. | contract rule 2 | import-closure walk (`tests/ard/test_wp4_runner.py:298` pattern) |

Further binding rules (plan D2, D3, D16, §1.11):

1. **`src/agents_scaling/study/` and `src/agents_scaling/ard/` are never edited.** nextdyn imports
   study-agnostic mechanisms and re-implements everything that carries ARD36 vocabulary. The
   exact set of `agents_scaling.ard` names nextdyn may import is `nextdyn.ARD_IMPORTS`
   (machine-checked by `test_ard_imports_are_exactly_the_allowlist`); vendored functions each
   carry a parity test.
2. **Bank draws are `decisions` rows** (`phase="bank"`, `round=0`, `agent_slot=draw_index`,
   `episode_id=bank_draw_id`, `root_team_id=""`); episode round 0 is stored as reference rows
   (`phase="root_ref"`, `is_reference=True`, `generation_uid` = the draw's uid, `reuse_of` = the
   draw's `decision_id`, resolved inside `decisions`). `banks` is a keyed view (plan D14).
3. **The generating model is in every key.** `CompositeKey.receiver_model_hash` (the spec-17.2
   name) is `ids.model_hash(<the model that generates this decision>)` for every kind — bank,
   episode, receiver, audit — and an empty value is refused. `decisions` has no second model
   column; `banks.model_hash` and the derived keys are views of it (R1).
4. **One dependency vocabulary** (`deps.py`): `cell:<cell_id>`, `bundle:<family>/<bucket>`,
   `register:<kind>/<sha>` with register kinds `pool | packets | panel`. Nothing else is a
   dependency; a bare id is an error, never advice (plan D16).
5. **Pools are sub-run-roots**: `$RUN/pools/{deadline,low,audit}/servers/` (the study registry
   hard-codes `<root>/servers`), pool port offsets from `compute.yaml`.
6. **Grades key on `generation_uid`** `(generation_uid, grader_hash)`; no boolean grade column
   exists anywhere; `grading.tri_state(row)` is the only reader.
7. **Private rows never land under `cells/`.** Rows of a `__private__` table
   (`grades_private`, `root_teams_private`) travel in `ItemFile.private_rows` and the runner
   writes them to `<run_root>/protected/cells/<cell_id>/items/<item>.json` (0600 in 0700,
   `tables.protected_item_path`); the public item file never carries them (R7, plan §2.5).
8. **Every verb obtains its config through `config.load_study(path, check=True)`** (the default):
   it runs `compile_checks` against `compute.yaml`, `analysis.yaml` and `protocols.yaml` beside
   `study.yaml` and raises `ContractViolation` listing every violation. Only a test that builds a
   violating config on purpose passes `check=False`. A non-null `locks.*_lock_hash` must equal
   sha256 of the lock file beside `study.yaml`; the `.sha256` sidecars the lock writer adds are a
   convenience for humans and never substitute for that pin.
9. **One fact, one declaration across the YAMLs** (`test_no_fact_is_declared_twice_across_the_configs`
   lists the forbidden pairs): the E3a / capture / E2a / natural-protection panel sizes live in
   the experiment and instrumentation blocks, the core protocol set in `protocol.protocols`, the
   E2b grid in `experiments.E2b` (instantiated by `protocols.yaml` through
   `config.E2B_ENDORSEMENT_SUPPORT`, cross-checked by the compiler), Holm's `n_primary` /
   alpha and the power-planning alpha are computed from `study.analysis` (never re-typed in
   `analysis.yaml`), and the capture partition is `compute.capture.partition` only.
10. New code lives under `src/agents_scaling/nextdyn/`, `tests/nextdyn/`, `configs/nextdyn/`,
    `slurm/nextdyn_*`, `scripts/nextdyn_*`. Do not edit another WP's files. Downstream WPs may
    **stub** the interfaces in §2 but must not change their signatures.

## 1. File ownership (plan §5; disjoint; parallel after WP0)

| WP | Owns | Depends on |
|---|---|---|
| WP0 | `errors, contract, config, locks, ids, tables, kinds, deps, cli, __init__`; config skeletons with null lock fields; `compute.yaml`; this document | — (first) |
| WP1 | `decoding, adapters, inference, continuation, telemetry`; fake-server fixtures; serving profiles `nd1-8B`/`nd1-32B` (already added to `serving/profiles.py`; crc32 port bases 8884/8083, `nextdyn-32B` would have collided with `0.6B`) + sampler injection | WP0 |
| WP2 | `graphs, protocols, prompts, templates/, schema, packets, states, episode` | WP0, WP1 signatures |
| WP3 | `families, datasets, items, items_build, inventory, graders/, grading, banks, construct, bundles`; `datasets.lock.json` | WP0 |
| WP4 | `cells, plan, runner, run_one, dispatch, fleet, status, materialize, freeze, gates, __main__` (verb wiring only; the dispatcher itself is WP0's `cli.py`); `slurm/nextdyn_*`; `scripts/nextdyn_*` | WP0–WP3, WP5 interfaces |
| WP5 | `mod97, checker, receiver, natural_packets, selectors, audit` | WP1, WP2 |
| WP6 | `capture` (JSON-packet anchors; `ANSWER_END` pre-confidence eligibility; `study.neural` engine) | WP2, WP4 |
| WP7 | `outcomes, stats, calib, panels, analysis, power, figures, release, e0/*`; `analysis.yaml`, `figure_manifest.yaml`, `legacy_sources.yaml` | WP0 |

`analysis.yaml`, `protocols.yaml`, `compositions.yaml` and `legacy_sources.yaml` ship from WP0
as complete skeletons: every leaf of all six YAMLs has a declared consumer in `config.CONSUMERS`
(pending until that module lands, then held to reading it). WP2 owns `protocols.yaml`'s values,
WP3 `compositions.yaml`'s, WP7 `analysis.yaml`'s and `legacy_sources.yaml`'s from here on.
`study.yaml` also carries `pilot:` (spec 19.1 subset sizes and the `validate --smoke` sizes)
and `gates:` (spec 18.6 targets); `plan` and `gates` read them and hard-code nothing.

## 2. Frozen interfaces

Code against these signatures. Where a dependency is not yet built, stub it. Everything below
is reconciled with the plan; where an engine note said something else (a `records.py`, runtime
read-recording, `register:teams`), the plan's version is the one written here.

### 2.0 WP0 — identity, contract, tables, kinds, dependencies (implemented)

```python
# contract.py — parametric shapes; values only from config
@dataclass(frozen=True)
class TokenContract:            # thinking_cap, final_cap, served_window, reserve, packet_cap,
    ...                         # rationale_cap, claims_max, claim_cap   (all ints, no defaults)
    @property
    def max_new_tokens(self) -> int      # 1 + thinking_cap + 1 + final_cap (3074 main; amendment A1)
    def budget_hash(self) -> str         # sha256(JCS(asdict)) — CompositeKey.budget_hash

@dataclass(frozen=True)
class Sampling:                 # temperature, top_p, min_p, repetition_penalty, presence_penalty,
    ...                         # frequency_penalty as identity.float_str strings; top_k int
    @classmethod
    def from_floats(cls, **values) -> "Sampling"

@dataclass(frozen=True) class BankSpec:   difficulty_draws, donor_initial, donor_max, natural_root_draws
@dataclass(frozen=True) class TeamShape:  n_agents, revision_rounds, score_rounds: tuple[int, ...]
@dataclass(frozen=True) class ModelCell:  name, hf_id, revision, tokenizer_revision, adapter, profile, tp, served_model_name,
                                          lane ("8B" | "32B": the generation lane, a scale class independent of served_model_name),
                                          total_output_tokens: int | None   # non-null iff adapter == total_output_envelope (spec 5.6);
                                                                            # the envelope adapter's cap, in the contract sha
@dataclass(frozen=True) class SelectionSpec: primary, pool_scope, tie_break, secondary_deployed, analysis_only,
                                             audit_model, audit_thinking_tokens, audit_final_tokens, audit_failure_fallback,
                                             synthesis_allowed, gold_access, archive_audit_enabled
@dataclass(frozen=True) class FailurePolicy: transient_retries, retry_same_request_and_seed, model_output_repair_calls,
                                             wrong_answer_retry, context_overflow, unresolved_semantics, null_item_ids,
                                             duplicate_logical_keys, hidden_label_leakage
@dataclass(frozen=True)
class StudyContract:            # study_id, study_seed_hex, split_salt_hex, contract_version, families,
    ...                         # models: Mapping[str, ModelCell], sampling, tokens, receiver_tokens: Mapping[str, TokenContract],
                                # audit_tokens, banks, team, selection, failure_policy
    def sha256(self) -> str
    def model(self, name) -> ModelCell

SEED_NAMESPACE_PREFIX = "ND1"
PHASE_BANK="bank"  PHASE_ROOT_REF="root_ref"  PHASE_REVISION="revision"
PHASE_RECEIVER_PLAN="receiver_plan"  PHASE_RECEIVER_DECISION="receiver_decision"  PHASE_AUDIT="audit"
ORDER_PACKET  ORDER_PSEUDONYM  ORDER_PROTECT  ORDER_OPTION_PERM  ORDER_WSTAR  ORDER_DONOR_ORDER
ORDER_SLOT  ORDER_RING  ORDER_DISPATCH  ORDER_PASSK  VOTE_TIE          # blind-order namespaces (strings)
ADAPTER_NATIVE_THINKING="native_thinking"  ADAPTER_TOTAL_OUTPUT_ENVELOPE="total_output_envelope"
CONTRACT_FIELD_CONSUMERS: Mapping[(dataclass name, field), module]   # R2 for the contract itself; guard-checked, pending until landed
```

`models.*.thinking_tokens` in `study.yaml` is the spec-17.4 trap field: its only admitted value
is `null` (on the envelope adapter a value is the spec's error; on a native cell it would be a
per-cell override nothing implements in v1 — the one native cap is `budgets.thinking_tokens_main`).

```python
# ids.py — identity scheme (spec 17.2; plan §1.3)
class CellKind(str, Enum): DIFFICULTY_BANK, DONOR_BANK, NATURAL_ROOT, EPISODE, RECEIVER, SELECTION_AUDIT,
                           GRADE, CONSTRUCT, CAPTURE, E0, MATERIALIZE          # values are the lowercase names
BANK_KINDS; EPISODE_KINDS = {EPISODE, RECEIVER}; GENERATION_KINDS = BANK_KINDS | EPISODE_KINDS | {SELECTION_AUDIT}

@dataclass(frozen=True)
class CompositeKey:      # exactly the 12 spec-17.2 fields, in this order
    study_id, cell_kind: CellKind, source_item_id, item_variant_id, root_team_id, protocol_hash,
    receiver_model_hash, budget_hash, replicate_id: int, round: int, agent_slot: int, phase: str
    # __post_init__: empty source_item_id -> IdentityError; bank kind with root_team_id != "" -> IdentityError;
    # empty receiver_model_hash -> IdentityError (every kind); a non-generation kind -> IdentityError
    def as_array(self) -> list;  def decision_id(self) -> str        # "d:" + sha256(JCS(array))[:40]

def episode_id(kind, root_team_id, protocol_hash, model_hash, budget_hash, replicate) -> str
    # "<kind>|<team12>|<cond12>|<model16>|<budget8>|rep<k>" ("episode|…" / "receiver|…"); IdentityError unless kind in EPISODE_KINDS.
    # RECEIVER episodes pass the fixed-packet panel id as root_team_id: E2a "packets:<register sha>", E2b the mod97
    # problem_id.  <team12> is the first 12 characters after an optional "<tag>:" prefix; a shorter token (an E2b
    # problem id) is sha256-hashed to 12, so any token of any length works and no two collide by truncation.
def bank_draw_id(kind, source_item_id, item_variant_id, model_hash, budget_hash, draw) -> str
    # "<kind>|<item>|<variant>|<model16>|<budget8>|k<draw>"; IdentityError unless kind in BANK_KINDS
def audit_episode_id(pool_id, model_hash, budget_hash) -> str       # "selection_audit|<pool12>|<model16>|<budget8>"
def root_team_id(origin_kind, source_item_id, item_variant_id, member_generation_uids, focal_slot, rule) -> str
    # "team:" + sha256(JCS([...]))[:32]; refuses empty or duplicated members
def seed_key_for(key: CompositeKey, *, split_role) -> study.types.SeedKey
    # EVERY component pinned (plan D4 + the two SeedKey slots D4 does not list):
    #   source_id=key.source_item_id, split=split_role (item_splits.split_role of the item),
    #   model_cell=key.receiver_model_hash, episode_rep=key.replicate_id, actor_slot=key.agent_slot,
    #   purpose=key.phase, step_slot=key.round, namespace=f"ND1/{cell_kind}/{item_variant_id}/{root_team_id}"
    # NO condition, NO budget: protocol_hash and budget_hash never enter the seed.
def generation_uid(physical_request_ids: Sequence[str]) -> str     # "g:" + sha256(JCS(list)), UNTRUNCATED (plan §1.3), phase order
def content_hash(final_bytes: bytes) -> str                         # "c:" + sha256
def condition_hash(identity_object: Mapping) -> str                 # sha256(JCS(obj)); refuses label keys
def model_hash(model: ModelCell) -> str                             # sha256(JCS([hf_id, revision, tokenizer_revision, adapter]))[:16]
def blind(study_seed: bytes, namespace: str, *parts) -> bytes       # = study.identity.blind_order_key
```

**Per-kind key layout** — the value of every `CompositeKey` field for each kind that writes
`decisions` (`study_id`, `source_item_id`, `item_variant_id` are always the item's; `M` is the
generating model, `model_hash(M)` its 16-hex hash):

| kind | `root_team_id` | `protocol_hash` | `receiver_model_hash` | `budget_hash` | `replicate_id` | `round` | `agent_slot` | `phase` | `episode_id` |
|---|---|---|---|---|---|---|---|---|---|
| DIFFICULTY_BANK / DONOR_BANK / NATURAL_ROOT | `""` | `""` | `model_hash(M)` | `contract.tokens.budget_hash()` | `0` | `0` | draw index (0..63; extension draws 32..63) | `bank` | `bank_draw_id(kind, item, variant, model_hash(M), budget_hash, draw)` |
| EPISODE | `ids.root_team_id(...)` of the team | `VisibilitySpec.condition_hash` | `model_hash(M)` (the episode's model; 32B replay of an 8B team is a different key) | the episode's `TokenContract.budget_hash()` | replicate k | `0` (`root_ref`) or `1..R` (`revision`) | slot | `root_ref` / `revision` | `episode_id(EPISODE, team, cond, model_hash(M), budget_hash, k)` |
| RECEIVER | panel id: E2a `"packets:<register sha>"`, E2b the mod97 `problem_id` | `ReceiverSpec.condition_hash` | `model_hash(M)` (the receiver model) | **episode-level** `contract.receiver_tokens[f"{B}/one_call"].budget_hash()` for BOTH phases (the per-phase `B/2` contracts enter only the request identity through the decoding) | replicate k | `0` | `0` (one receiver) | `receiver_plan` / `receiver_decision` | `episode_id(RECEIVER, panel id, cond, model_hash(M), budget_hash, k)` — one id per E2b episode |
| SELECTION_AUDIT | `pool_id` of the sealed pool | the audited episode's `protocol_hash` | `model_hash(M)` (the audit model) | `contract.audit_tokens.budget_hash()` | `0` | the pool's round | candidate index in the sealed pool | `audit` | `audit_episode_id(pool_id, model_hash(M), budget_hash)` |

```python
# tables.py — the single declaration of every table (spec 17.3; plan §1.4)
# One frozen row dataclass per table; primary-key fields carry field(metadata={"pk": True});
# unique constraints are the class attribute ``__unique__``; ``__private__`` marks protected tables.
PlanRow, SourceItemRow, ItemVariantRow, ItemSplitRow, BankRow, RootTeamRow, RootTeamPrivateRow, RequestRow,
DecisionRow, TransmissionRow, SelectionRow, ToolEventRow, GradePrivateRow, ActivationRow, MetricRow
@dataclass(frozen=True) class Column: name, type, nullable
@dataclass(frozen=True) class TableSchema: name, row_type, columns, primary_key, unique, producer, consumers, private
TABLES: Mapping[str, TableSchema];  PRIMARY_KEYS: Mapping[str, tuple[str, ...]];  DERIVED_PRIMARY_KEYS;  PRIVATE_TABLES
# DecisionRow: decision_id + the 12 key fields (receiver_model_hash = the generating model; no other model column) + lineage
# (episode_id, cell_id, display_label, generation_uid, content_hash, request_hashes_json, aliased, reuse_of, is_reference,
# first_decision_id, bundle_export_hash, parent_decision_ids_json, seen_transmission_ids_json) + public output + generation + metering
def schema_of(row_type) -> TableSchema
def project(table: str, row: Mapping) -> dict          # strict whitelist; undeclared column -> MaterializeError
def row_from_dict(row_type, data) -> row               # refuses unknown keys
def write_table(rows: Sequence[Row], path: Path) -> None   # declared column order; ``.empty`` sentinel (study/aggregate.py:257 semantics, vendored)
def read_table(tables_dir, name) -> list[dict]
def check_keys(tables_dir, *, strict=True) -> list[KeyViolation]      # ard/keys.py:188-249 logic, vendored
def assert_referential(tables_dir, *, strict=True) -> list[str]        # plan §1.4 / D14 materialize-time assertions

class ItemStatus(str, Enum): COMPLETE, CONTEXT_FAILURE, FAILED
@dataclass(frozen=True)
class ItemFile:            # schema_version, cell_id, cell_kind, source_item_id, status, bundle_export_hash,
    ...                    # rows: Mapping[table, tuple[Row, ...]]          -- PUBLIC tables only
                           # private_rows: Mapping[table, tuple[Row, ...]]  -- __private__ tables only (default {})
    has_private_rows: bool
    def to_json(self) -> str;  def to_private_json(self) -> str
    @classmethod def from_json(cls, text, private_text=None) -> "ItemFile"          # unknown keys refused; a private table under rows refused
    def write(self, path, *, protected_path=None) -> None    # public file at path; private rows REQUIRE protected_path (0600 in 0700, never under cells/)
    @classmethod def read(cls, path, *, protected_path=None) -> "ItemFile"
def item_path(run_root, cell_id, source_item_id) -> Path             # <run_root>/cells/<cell_id>/items/<item>.json
def protected_item_path(run_root, cell_id, source_item_id) -> Path   # <run_root>/protected/cells/<cell_id>/items/<item>.json
```

Worker contract (every cell kind, all WPs): `worker(runner, source_item_id) -> ItemFile`; the runner
writes the returned `ItemFile` atomically (tmp + fsync + rename, `meta.json` last):
`item.write(tables.item_path(run_root, cell_id, item), protected_path=tables.protected_item_path(run_root, cell_id, item))`.
GRADE and CONSTRUCT workers return their `grades_private` / `root_teams_private` rows in
`private_rows` (never in `rows`); the public item file still exists for every item so `cell:`
dependencies resolve uniformly. Workers never write item files themselves.
`telemetry.decision_row()` is the only `DecisionRow` constructor.

```python
# kinds.py — the one table a dispatcher asks about a cell kind (plan §3.3)
class Lane(str, Enum): SMALL="8B"; LARGE="32B"; AUDIT="audit"; CPU="cpu"; CAPTURE="capture"
class Pool(str, Enum): DEADLINE="deadline"; LOW="low"; AUDIT="audit"
@dataclass(frozen=True)
class KindSpec: kind, worker ("module:callable"), needs_endpoint, lane (Lane | None: None = follows the model's lane),
                deadline_bearing, dep_prefixes, doc                            # no numbers here (all in compute.yaml)
KIND_FIELD_CONSUMERS: Mapping[field, module]   # kind/worker -> runner, needs_endpoint -> run_one, lane/deadline_bearing -> dispatch,
                                               # dep_prefixes -> cells, doc -> plan (--dry-run listing); guard-checked, tests never count
KIND_TABLE: Mapping[CellKind, KindSpec]     # covers every CellKind (import-time assertion)
DEADLINE_BEARING: frozenset[CellKind];  PRIORITY: tuple[CellKind, ...]  # dispatch order, lower index first
CELL_KIND_COMPAT: Mapping[str, tuple[CellKind, str | None]]   # spec names -> (kind, phase)
def spec_for(kind) -> KindSpec;  def priority_rank(kind) -> int
def lane_for(kind, model: ModelCell | None) -> Lane          # generation kinds: Lane(model.lane) (8B | 32B), so the Llama entry compiles
def pool_for(kind, model: ModelCell | None, compute: ComputeConfig) -> Pool | None
@dataclass(frozen=True) class CellShape: items_per_cell, parallel_items, max_inflight, cpus, mem, time, partition
def items_per_cell(base, *, reference: ReferenceShape, n, revisions, thinking_cap) -> int
    # clip(floor(base * (ref_n*ref_r)/(n*revisions) * ref_thinking/thinking_cap), 1, base)
def cell_shape(kind, *, compute: ComputeConfig, model=None, thinking_cap=None, team=None, grade_resources=None) -> CellShape
    # CAPTURE's partition is compute.capture.partition (cells.kinds.capture.partition is null, R1)
```

```python
# deps.py — dependency vocabulary (plan D16; ard/runner.py:123-226 + ard/seal.py rules vendored)
CELL_PREFIX="cell:"  BUNDLE_PREFIX="bundle:"  REGISTER_PREFIX="register:"  REGISTER_KINDS=("pool","packets","panel")
def parse_dependency(dep) -> ParsedDependency(kind, value)          # unknown prefix / bare id -> DependencyUnmet
def cell_dependency(cell_id) / bundle_dependency(family, bucket) / register_dependency(kind, sha) -> str
def register_path(run_root, kind, sha) -> Path                       # <run_root>/registers/<kind>/<sha>/REGISTER.json
def write_register(run_root, kind, payload: Mapping) -> str          # payload["entries"] non-empty; content-addressed over JCS(body); tmp+fsync+rename;
                                                                     # a float in the payload -> ManifestError (registers hold ids, hashes, ints)
def verify_register(run_root, kind, sha) -> RegisterStatus(status ∈ met|tampered|missing, detail)
def bundle_path(run_root, family, bucket) -> Path                    # <run_root>/plan/root_teams/<family>/<bucket>.json (+ .sha256 sidecar)
def write_bundle(run_root, family, bucket, payload: Mapping) -> str  # sidecar sha; refuses a differing overwrite
def verify_bundle(run_root, family, bucket) -> RegisterStatus
def dependency_status(run_root, dep) -> DependencyStatus(dep, kind, path, status ∈ met|unmet|tampered, detail)
    # cell: meta.json present AND directory ownership (meta.cell_id == dir, n_items == item files, every item file claims the cell)
def unmet(run_root, deps) -> list[DependencyStatus]
```

```python
# config.py / locks.py
def load_study(path, *, mode=None, check=True) -> StudyConfig   # strict YAML; unknown keys -> ConfigError naming the path;
                                                               # check=True runs compile_checks against the sibling YAMLs and raises ContractViolation
def load_compute(path) -> ComputeConfig;  def load_analysis(path, *, profile) -> AnalysisConfig
def compile_checks(cfg, *, compute=None, analysis=None, protocols: Path | None = None) -> list[ContractViolation]
def lock_hash_checks(cfg) -> list[ContractViolation]        # non-null locks.*_lock_hash == sha256(lock file); inventory hashes == datasets lock fields
def e2b_grid_checks(cfg, protocols_path) -> list[ContractViolation]   # experiments.E2b grid == enabled two-phase receivers (E2B_ENDORSEMENT_SUPPORT)
E2B_ENDORSEMENT_SUPPORT = {"correct_minority": "1:4", "correct_majority": "4:1", "hidden": "hidden"}
CONSUMERS: Mapping[str, str]                                # every leaf key path of all six YAMLs -> consuming module / slurm template
StudyConfig.pilot: PilotBlock   # natural_items_per_family, constructed_paired_items_per_family, directional_items_per_family,
                                # natural_receiver_items, natural_receiver_tasks, synthetic_receiver_problems, adapter_check_problems,
                                # capture_items_per_family, smoke_solver_calls_per_model, smoke_receiver_episodes      (spec 19.1; read by plan / gates)
StudyConfig.gates: GatesBlock   # max_unresolved_infra_rate, max_format_failure_rate, max_arm_differential_pp          (spec 18.6; read by gates)
AnalysisProfile has no holm block and PowerBlock no alpha: stats computes n_primary = len(study.analysis.primary_hypotheses),
Holm alpha = study.analysis.familywise_alpha, and power uses familywise_alpha / n_primary (R1).
def lock_paths(config_path) -> LockPaths                    # models/datasets/prompts lock files beside study.yaml
def load_models_lock / write_models_lock / verify_models_lock(lock, hf_home) / lock_models(cfg, compute, hf_home)
def load_datasets_lock / write_datasets_lock;  def load_prompts_lock / write_prompts_lock / lock_prompts(templates_dir, prefill_text)
def lock_inputs_main(argv) -> cli.Report                    # models + prompts here; datasets via nextdyn.datasets.lock_datasets or blocked
```

```python
# cli.py — one flag vocabulary, one summary line
@dataclass(frozen=True)
class Report: completed=0; failed=0; skipped=0; blocked=0; reused=0; detail=""
    def line(self) -> str        # "completed=.. failed=.. skipped=.. blocked=.. reused=.."
VERBS: Mapping[str, str]         # verb -> "module:callable", resolved lazily; missing module -> Report(blocked=1), exit 2
def common_parser(prog) -> argparse.ArgumentParser        # --config (default configs/nextdyn/study.yaml), --run-root, --manifest
def resolve_run_root(args) -> Path                        # $ASYS_RESULTS_ROOT/<study id> unless --run-root
def main(argv) -> int                                     # exit 1 on failed>0, 2 on blocked>0 (status always 0)
```

### 2.1 WP1 — inference (`decoding`, `adapters`, `inference`, `continuation`, `telemetry`)

```python
class ModelAdapter(Protocol):
    name: str                                                   # contract.ADAPTER_*
    def decoding(self, sampling: Sampling, tokens: TokenContract, *, purpose: str) -> study.types.Decoding
    def request_kwargs(self, decoding: Decoding) -> dict        # native: extra_body.thinking_token_budget; envelope: {}
    def split(self, record, tokenizer) -> SplitCompletion       # (native_ids, delimiter_positions, final_ids, boundary_injected, natural_stop)
    def continuation_supported(self) -> bool
    def assert_compatible(self, model: ModelCell, cfg_block) -> None   # native budget on an envelope adapter -> ContractViolation
ADAPTERS: Mapping[str, ModelAdapter]

CONTINUATION_PREFILL_TEXT = '\n\n{\n  "reasoning": "'        # decoding.py; bytes hashed into prompts.lock.json (D5)
def decoding_for(adapter, sampling, tokens, *, purpose) -> Decoding      # asserts max_tokens == 1+T+1+F (native)
def continuation_decoding(tokens) -> Decoding                             # max_tokens == final_cap, thinking 0, enable_thinking False

@dataclass(frozen=True)
class DecisionRequest: messages: tuple[dict, ...]; model: ModelCell; adapter: str; tokens: TokenContract
                       seed_key: SeedKey; purpose: str; top_logprobs: int; key: CompositeKey
@dataclass(frozen=True)
class DecisionRecord:
    request_ids: tuple[str, ...]; physical_request_ids: tuple[str, ...]; generation_uid: str
    prompt_token_ids: tuple[int, ...]; completion_token_ids: tuple[int, ...]
    native_reasoning_text: str; final_text: str; thinking_tokens: int; final_tokens: int
    boundary_injected: bool; natural_stop: bool; cap_hit: bool; final_overrun: bool
    answer_logprob: float | None; final_top_logprobs: Any; prompt_tokens: int; cached_prefill_tokens: int
    finish_reason: str; latency_s: float; endpoint_id: str; replica_id: str; server_batch_id: str; arrival_ts: float
    gpu_type: str; partition: str; host: str; flops: dict; continuation: dict | None; discarded_span_tokens: int
    aliased: bool; uncertain_execution: bool; attempts: tuple[AttemptInfo, ...]
Generate = Callable[[DecisionRequest], DecisionRecord]
class NextdynChatClient(VllmChatClient): ...                  # max_exogenous_retries = cfg.failure_policy.transient_retries; never sends guided fields
class NextdynEndpointPool(EndpointPool): allowed_partitions: frozenset[str] | None
@dataclass class DispatchContext: store; pool_for; cost_fn; tokenizer_for; contract: StudyContract; cell_id: str; run_id: str
def preflight(req, ctx) -> None                               # serving.context.preflight_chat_context at served_window; ContextTooLong is typed, never retried
def generate_decision(req, *, ctx, on_event=None) -> DecisionRecord   # store get_or_generate -> adapter split -> continuation iff boundary_injected -> merged record
def request_spec(req, ctx) -> study.types.RequestSpec
@dataclass(frozen=True)
class CellContext:            # what a DecisionRow needs that neither the record nor the key carries
    cell_id: str; episode_id: str; display_label: str | None; bundle_export_hash: str | None
    # episode_id: EPISODE/RECEIVER -> ids.episode_id(...); a bank kind -> ids.bank_draw_id(...); SELECTION_AUDIT -> ids.audit_episode_id(...)
@dataclass(frozen=True)
class ProducerStamp:          # {cell_id, run_id, slurm_job_id, host, endpoint_id, gpu_type, partition} -> RequestRow.producer_json
    cell_id: str; run_id: str; slurm_job_id: str | None; host: str; endpoint_id: str; gpu_type: str | None; partition: str | None
def decision_row(record: DecisionRecord, key: CompositeKey, out: PublicOutput, *, cell: CellContext,
                 parents: tuple[str, ...], seen_transmissions: tuple[str, ...], aliased: bool, reuse_of: str | None,
                 is_reference: bool = False, native_reasoning_ref: str | None = None) -> DecisionRow   # THE only constructor (R6)
    # receiver_model_hash, episode_id, cell_id, display_label, bundle_export_hash, seen_transmission_ids_json come from key / cell;
    # canonical_class is None here (grades are writer-side); content_hash = ids.content_hash(record.final_text bytes)
def request_rows(record: DecisionRecord, key: CompositeKey, *, cell_id: str, request_specs: Sequence[study.types.RequestSpec],
                 producer: ProducerStamp) -> tuple[RequestRow, ...]      # one per physical request, in phase order;
    # decoding_hash / seed_hex / tokenizer_hash / engine_digest / model_hash / prompt_token_hash / completion_token_hash are derived
    # from request_specs[i] (RequestSpec.decoding, .seed_key, .checkpoint) and record.prompt_token_ids / completion_token_ids;
    # first_decision_id is None at write time (materialize fills it per generation_uid)
```

<!-- WP1 correction to §2.1 (inference).  Two of the signatures above cannot be implemented as
written; everything else in §2.1 is unchanged, and every name below is an ADDITION that leaves
the listed signatures intact.  Reason for each is given inline. -->

**§2.1 corrections and additions (WP1).**

1. `continuation_decoding(tokens, sampling)` — **signature corrected.** As listed
   (`continuation_decoding(tokens)`) it is unimplementable: a `Decoding` needs the spec-5.5
   sampler recipe, that recipe is a `study.yaml` value, and no experimental number may be a
   module constant in this package (contract preamble), so it cannot be defaulted. The recipe
   is passed explicitly. Callers use `decoding.decoding_for_contract(...)` /
   `continuation.ContinuationContract`, which read it from `StudyContract.sampling`.
2. `ModelAdapter.total_output_tokens: int | None` — **protocol member added.** §2.1 requires
   `decoding_for` to assert `max_tokens == total_output_tokens` for the envelope adapter, and
   `decoding_for(adapter, sampling, tokens, *, purpose)` receives only the adapter, so the
   envelope cap has to be reachable through it. `None` on the native contract, whose caps are
   `TokenContract`'s and live in one place only (R1). `adapters.adapter_for(model)` binds it
   from `ModelCell.total_output_tokens`.
3. Additions that no listed signature changes (WP1 module placement is WP1's to choose):
   `adapters.BlockedAdapter` (a `NextdynError`; §2.1 names it without a home) and
   `inference.ContextTooLong` (likewise, carrying the `serving.context.ContextPreflight`);
   `inference.ReplicaStamp` / `JobInfo` / `NextdynEndpointPool.replica_stamp` (the registry
   entry names a Slurm job, not its partition or GPU type, and `RequestRow` declares both);
   `inference.request_specs_for(req, ctx, record)` (rebuilds a continued decision's phase-2
   spec so `request_rows`' frozen signature can be satisfied without keeping one alive across
   the two calls); `inference.server_batch_id` + `ARRIVAL_BUCKET_S` (vendored from
   `ard/resources.py:177-186`, which is not on the `ARD_IMPORTS` allowlist, with a parity test);
   `continuation.ContinuationContract` (every ARD36 constant of the vendored
   `ard/client.py:978-1358` as a parameter — this is what makes the parity test possible);
   `telemetry.Metering` / `RequestResource` / `ReplicaRef` / `ConservationRow` and the
   `INFRA_*` / `LOGPROB_NORM_*` / `TERMINATION_REASONS` vocabularies (`decisions` column values
   are facts and `telemetry` is that table's producer, R1).
4. `DecisionRecord` field semantics fixed by WP1 where §2.1 lists a name but not a meaning, and
   `telemetry.decision_row` depends on it: `final_tokens` is the length of the **delivered**
   final span after the `final_cap` truncation, `answer_logprob` is the mean over exactly those
   tokens (so `answer_logprob_sum == mean * answer_token_count` holds by construction, R1), and
   `final_overrun` says the raw span was longer. `request_ids` are the specs that were asked
   for and `physical_request_ids` the ids the store holds; the merge asserts they agree.

### 2.2 WP2 — protocol engine (`graphs`, `protocols`, `prompts`, `schema`, `packets`, `states`, `episode`)

```python
Graph = tuple[tuple[bool, ...], ...]                          # G[j][i]: i receives j's previous packet; no self edges
GRAPH_BUILDERS: Mapping[str, Callable[..., Graph]]            # complete, empty, focal_channels, ring_fixed, protect_slot
@dataclass(frozen=True) class GraphRef: name: str; params: Mapping[str, int | bool | str]
@dataclass(frozen=True) class GraphSchedule: kind: Literal["fixed", "per_round"]; ref: GraphRef

class ConditionSpec(Protocol):                                # plan D7 — conditions are data
    display_label: str; legacy_alias: str | None; equivalent_to: str | None
    def identity_object(self) -> dict                         # labels EXCLUDED
    @property
    def condition_hash(self) -> str                           # = ids.condition_hash(identity_object()); CompositeKey.protocol_hash
@dataclass(frozen=True)
class VisibilitySpec:   own_draft: bool; graph: GraphSchedule; packet_content: PacketContent; filler: Filler; template: Template
                        schedule: Literal["synchronous"]; display_label; legacy_alias=None; equivalent_to=None
@dataclass(frozen=True)
class ReceiverSpec:     packet_content: PacketContent; support: Literal["1:4", "4:1", "hidden"]; checker: bool
                        draft_role: Literal["correct", "wrong", "none"]; phases: Literal[1, 2]; templates: tuple[str, ...]
                        display_label; legacy_alias=None; equivalent_to=None
def load_registry(path=configs/nextdyn/protocols.yaml) -> Registry(episodes: Mapping[str, VisibilitySpec], receivers: Mapping[str, ReceiverSpec])
@dataclass(frozen=True) class TransmissionIntent: sender_slot; source_round; source_decision_id; content; filler: bool; display_position
PacketPlan = Mapping[int, tuple[TransmissionIntent, ...]]
def packet_plan(state: RoundState, spec: VisibilitySpec, *, round_: int, study_seed: bytes) -> PacketPlan   # ONE function
def packet_order(study_seed, pairing_id, round_, recipient, senders) -> tuple[int, ...]              # vendored ard/protocols.py:241

class ParseStatus(str, Enum): OK, OK_ORDER_VIOLATION, OK_TRUNCATED_FIELDS, NO_JSON, INVALID_JSON, MISSING_FINAL,
                              PREMATURE_REASONING_EOS, NO_THINK_OPEN, CONTEXT_FAILURE
@dataclass(frozen=True)
class PublicOutput: reasoning: str | None; shareable_claims: tuple[str, ...]; final_answer: str | None; confidence: float | None
                    field_order: tuple[str, ...]; order_violation: bool; parse_status: ParseStatus; raw_text: str; content_hash: str
                    over_limits: Mapping[str, bool]; final_answer_span: tuple[int, int] | None; confidence_span: tuple[int, int] | None
                    confidence_emitted_before_answer: bool
def parse_public_output(final_text, finish_reason, completion_ids, counter) -> PublicOutput
def split_native_reasoning(record: DecisionRecord, tokenizer) -> tuple[str, str]

FIELD_PRIORITY = ("final_answer", "reasoning", "shareable_claims")
@dataclass(frozen=True)
class TransmittedPacket: packet_id; pseudonym; fields; truncated: Mapping[str, bool]; token_count; counter_tokenizer_hash
                         serialized; packet_hash; original_content_hash; content: PacketContent; filler: bool
def render_packet(out: PublicOutput, *, pseudonym, counter, counter_tokenizer_hash, cap, content) -> TransmittedPacket
def unparsed_packet(pseudonym, counter) -> TransmittedPacket;  def inert_filler(n_tokens, seed, counter) -> TransmittedPacket
def check_packet_bytes(serialized: str) -> None;  def worst_case_prompt_tokens(contract, family, n, receiver_model) -> int
def transmissions_for(plan, packets, receiver_decision_id, *, destination_round) -> tuple[TransmissionRow, ...]

@dataclass(frozen=True) class AgentState: slot; pseudonym; latest: DecisionRef; root: DecisionRef; history
@dataclass(frozen=True) class RoundState: pairing_id; round_; agents: tuple[AgentState, ...]; classify; graph: Graph | None
@dataclass(frozen=True)
class EpisodeSpec: cell_kind: CellKind; source_item: SourceItem; roots: RootBundle; visibility: VisibilitySpec; model: ModelCell
                   tokens: TokenContract; replicate: int; revisions: int; n: int; split_role: str
    @property pairing_id -> str;  @property episode_id -> str      # ids.episode_id(...)
def run_episode(spec: EpisodeSpec, generate: Generate, *, tokenizer, study_seed: bytes, on_event=None) -> Episode
def request_for(spec, snapshot, slot, r, plan, packets) -> DecisionRequest
```

<!-- WP2 correction to §2.2 (protocol engine).  Every item below is an ADDITION that the
     frozen signature could not express; no frozen name, positional parameter or return type
     changed.  Reason column says why the frozen form was unimplementable. -->

**§2.2 corrections (WP2).**

| Frozen form | As implemented | Why |
|---|---|---|
| `VisibilitySpec{… display_label; legacy_alias=None; equivalent_to=None}` and `ReceiverSpec{…}` | both gain `enabled: bool = True` | `protocols.yaml` ships an `enabled` leaf for every condition and `config.CONSUMERS` declares `plan` as its consumer; a registry that dropped the key could not load the shipped file. It is a planning knob and is **excluded from `identity_object()`**, so no condition hash changes. |
| `parse_public_output(final_text, finish_reason, completion_ids, counter)` | `…, *, limits: contract.TokenContract` | the spec-6.1 limits (256-token rationale, 4 claims, 64-token claim) are *values* from `study.yaml`, and `contract.CONTRACT_FIELD_CONSUMERS` already names `schema` as the consumer of `rationale_cap` / `claims_max` / `claim_cap`. Hard-coding them in the parser would put experimental numbers in code. |
| `unparsed_packet(pseudonym, counter)` | `…, *, counter_tokenizer_hash, content=PacketContent.FULL, original_content_hash=None` | `TransmittedPacket.counter_tokenizer_hash` is non-null by construction; a packet that cannot say which tokenizer counted it is exactly the cross-model replay defect the field exists to prevent. |
| `inert_filler(n_tokens, seed, counter)` | `…, *, pseudonym, counter_tokenizer_hash` | same, plus `TransmittedPacket.pseudonym`: a filler occupies a *packet slot*, so it must look like one. |
| `transmissions_for(plan, packets, receiver_decision_id, *, destination_round)` | `…, receiver_slot, episode_id, cell_kind, protocol_hash, receiver_token_counts=None` | `tables.TransmissionRow` declares `receiver_slot`, `episode_id`, `cell_kind`, `protocol_hash` and `transmitted_tokens_receiver` as columns and the frozen parameters supply none of them. `plan` stays the whole `PacketPlan`; `receiver_slot` selects the recipient the one `receiver_decision_id` belongs to. |
| `rebuild_round_states(item_file, classify)` | `…, *, study_seed: bytes, episode_id=None` | pseudonyms are drawn from `blind(study_seed, ORDER_PSEUDONYM, pairing_id, slot)`, so a state cannot be rebuilt without the seed; `episode_id` disambiguates an item file holding more than one episode. |
| `EpisodeSpec{cell_kind, source_item, roots, visibility, model, tokens, replicate, revisions, n, split_role}` | gains `study_id: str`, `root_round: int`, `top_logprobs: int`, `family=None` | `study_id` is `CompositeKey`'s first field; `root_round` is `study.protocol.root_round`, whose declared consumer in `config.CONSUMERS` **is** `episode`; `top_logprobs` is a field of `inference.DecisionRequest`; `family` supplies the spec-6.2 `{task_specific_answer_instruction}` that the root template (and therefore the FRESH arm) renders. None of them enters a condition hash. |
| `run_episode(spec, generate, *, tokenizer, study_seed, on_event=None)` | `…, classify: Classifier \| None = None` | the gold-independent canonicalizer used for unanimity and coverage bookkeeping; when omitted the raw final answer is its own class, which keeps the loop free of family knowledge (R7). |

**Two interface facts §2 did not state, settled by WP2 and needed by WP1/WP3/WP4:**

1. *The typed context refusal is detected structurally, not by class.* `ard.client.ContextTooLong`
   is **not** on the `ARD_IMPORTS` allowlist and `nextdyn.inference` may not be importable while
   WP1 is in flight, so `episode.is_context_refusal(exc)` treats an exception as a build-time
   context refusal when it is a `serving.context.ContextCapacityError` **or** when it carries a
   `parse_status` whose value is `schema.ParseStatus.CONTEXT_FAILURE`. `inference.ContextTooLong`
   satisfies the second clause by carrying that attribute (ARD36's does the same).
2. *A root reference carries the bank draw's record.* `telemetry.decision_row` requires a
   `DecisionRecord`, and a `root_ref` row is a reference to a bank draw: `run_episode` reads the
   draw's record from the bundle slot (`RootSlot.record`, optional), so the reference row is built
   from the same physical sample the bank wrote. A **context-failure** decision has no record at
   all; until `telemetry.decision_row` admits `record=None`, `episode.episode_worker` refuses such
   a row with a typed `ManifestError` rather than inventing one.


### 2.3 WP3 — data (`families`, `datasets`, `items`, `items_build`, `inventory`, `graders/`, `grading`, `banks`, `construct`, `bundles`)

```python
@dataclass(frozen=True)
class FamilySpec: name; loader; renderer; answer_instruction; canonicalizer; grader; grade_resources: GradeResources
                  # GradeResources(cpus, mem, time, items_per_cell, container: "none" | "apptainer", sif: str | None)
FAMILIES: Mapping[str, FamilySpec]
def lock_datasets(cfg, hf_home) -> DatasetsLock                        # the lock-inputs half WP0 calls when importable
def build_items(run_root, cfg, *, split_salt, study_seed) -> None      # WRITER_SIDE
def load_items(run_root) -> Mapping[str, SourceItem]                   # runtime public reader + firewall
class BankRole(str, Enum): DIFFICULTY, DONOR, NATURAL_ROOT             # -> CellKind bank kinds
def bank_key(role, item, variant, model: ModelCell, tokens: TokenContract, draw) -> CompositeKey
    # root_team_id="", protocol_hash="", receiver_model_hash=ids.model_hash(model), budget_hash=tokens.budget_hash(),
    # replicate_id=0, round=0, agent_slot=draw, phase=PHASE_BANK
def bank_worker(runner, source_item_id) -> ItemFile
def difficulty(rows, grades) -> Difficulty
def grade_decision(family, out: PublicOutput, gold) -> GradePrivateRow;  def tri_state(row) -> Literal["C", "W", "U"]
def grade_worker(runner, source_item_id) -> ItemFile
@dataclass(frozen=True) class Composition: name; n; c0; wrong_pattern            # from compositions.yaml
def build_team(item, donors, comp, wstar, study_seed, *, shared) -> RootTeam;  def natural_team(item, draws, study_seed) -> RootTeam
def seal_root_teams(...) -> ...;  def export_root_bundle(team, drafts) -> Mapping    # label-free; written through deps.write_bundle
def construct_worker(runner, source_item_id) -> ItemFile
@dataclass(frozen=True) class RootBundle: root_team_id; source_item_id; item_variant_id; n; slots: tuple[RootSlot, ...]; export_hash; focal_slot
def load(run_root, family, bucket) -> RootBundle;  def assert_label_free(bundle_json) -> None
```

### 2.4 WP5 — receiver assays and selectors (`mod97`, `checker`, `receiver`, `natural_packets`, `selectors`, `audit`)

```python
def generate(seed: bytes, depth: int, index: int) -> Problem;  def correct_trace(p) -> Trace;  def wrong_trace(p, seed) -> Trace
def render_problem(p) -> str;  def render_trace(t) -> str;  def local_step_valid(p, row) -> bool
class CheckStepTool:
    def __init__(self, packets: Mapping[str, Trace], problem: Problem, *, enabled: bool, max_checks: int): ...
    def check_step(self, packet_id, step_id) -> CheckResponse            # status ∈ valid|invalid|disabled|bad_request
    def serve(self, requests) -> tuple[tuple[CheckResponse, ...], tuple[ToolEventRow, ...]]   # beyond max_checks -> over_budget events
def parse_plan_output(text) -> tuple[CheckRequest, ...]
def run_receiver_episode(cond: ReceiverSpec, packets: FixedPacketPair, generate, tool, *, study_seed) -> ReceiverEpisode
def receiver_worker(runner, source_item_id) -> ItemFile
def fix_natural_pair(item, donor_grades, *, study_seed) -> FixedPacketPair | None   # through packets.render_packet

@dataclass(frozen=True) class SealedPool: source_item_id; episode_id; cell_kind; round_; scope; candidates; protocol_hash
    @property pool_id -> str                                             # content-addressed; register:pool/<sha>
    # audit keys: root_team_id=pool_id, protocol_hash=pool.protocol_hash, receiver_model_hash=model_hash(audit model),
    # budget_hash=contract.audit_tokens.budget_hash(), round=pool.round_, agent_slot=candidate index, phase=PHASE_AUDIT
class Selector(Protocol):
    name: str; version: str; deployable: bool; needs_cells: bool
    def select(self, pool: SealedPool, ctx: SelectionContext) -> Selection
    def audit_requests(self, pool: SealedPool, ctx) -> Mapping[str, DecisionRequest]   # generative selectors only
    def parse_score(self, record: DecisionRecord) -> Score | Failure
SELECTORS: Mapping[str, Selector]
def tie_break(source_item_id, classes, selection_seed) -> str            # min sha256(JCS([item, class, seed]))
def audit_worker(runner, source_item_id) -> ItemFile
```

### 2.5 WP4 — cells, plan, runner, dispatch, fleet, materialize, freeze, gates, status

```python
@dataclass(frozen=True)
class CellSpec: cell_id; kind: CellKind; lane: Lane; pool: Pool | None; model: str | None; family: str; tokens_hash: str
                condition: ConditionRef | None; items: tuple[str, ...]; replicate: int; split_role: str
                depends_on: tuple[str, ...]; options: Mapping[str, Any]; reuse_of: str | None
def cell_id_for(kind, *, family, model, tokens_hash8, condition_hash12, split_role, replicate, shard, extra=()) -> str   # kind first
def write_manifest(path, cells) -> str;  def load_manifest(path) -> list[CellSpec]
def build_dag(cfg, inventory, *, stage) -> Plan;  def workload(plan) -> Workload
class NextdynCellRunner(study.runner.CellRunner): ...     # _freeze_gate -> freeze.gate; _hard_dependencies -> deps.dependency_status
def resolve_worker(kind) -> Callable[[NextdynCellRunner, str], ItemFile]
def run_cell(cell, run_root, *, server_run_root, pool, stop_event, cfg, max_inflight=None, dry_run=False) -> CellOutcome
EXIT_DEPS_UNMET = 5
def materialize(run_root, *, out_dir="tables", strict=True) -> Report   # tables.<ts>.tmp -> check_keys + assert_referential -> MANIFEST.json -> rename tables_vN
@dataclass(frozen=True)
class FrozenManifest: contract_sha256; config_sha256; locks; manifests; sample_size_lock; analysis_lock_hash; code_version; frozen_at
    def write(self, path) -> None;  @classmethod def load(cls, path);  def verify(self, cfg) -> None   # same class both ways (R1)
def smoke(...) / launch_gate(...) / pilot_gate(...) / coverage_gate(...) / release_check(run_root) -> GateReport
def capture_worker(runner, source_item_id) -> ItemFile                 # WP6
```

Every `main(argv)` behind a verb returns `cli.Report`; the dispatcher prints one summary line.

## 3. Facts already verified (do not re-derive)

- vLLM 0.21.0 wire shape: `choices[0].message.reasoning`, top-level `prompt_token_ids`,
  per-choice `token_ids`; `boundary_injected == (thinking_tokens == budget)` exactly.
  `guided_json` is accepted but **not enforced** on 0.21, so no guided field is ever sent
  (`final_phase_constrained_decoding: false`).
- Both pinned snapshots share one tokenizer blob (`<think>`=151667, `</think>`=151668).
  Qwen3-8B has 36 layers (capture blocks 8/17/26), Qwen3-32B 64 (15/31/47). No Llama weights
  are cached (gated): the family-transfer entry compiles, is `enabled: false`, and its adapter
  raises `BlockedAdapter` at request time.
- `study/packets.clip_fields` supports `kind="list"`; `preflight_chat_context(tokenizer,
  messages, *, requested_output_tokens, served_context, profile_name, enable_thinking,
  reserve_tokens)`; `SeedKey.namespace` is an unvalidated string (root team goes there).
- `registry.servers_dir` is hard-coded to `<run_root>/servers`, hence pools are sub-run-roots.
- `SeedKey.model_cell` is `CompositeKey.receiver_model_hash` (16 hex) and `SeedKey.split` is the
  item's `item_splits.split_role`; no producer chooses these strings.
- `tp` enters `engine_digest`, so `nd1-32B` is TP2 on every GPU type; `max_num_seqs` is
  outside request identity and varies per GPU type (`compute.yaml`).
- HF blobs are symlinks whose target basename is the content sha256 for safetensors shards
  (verified on both snapshots); `chat_template` lives inside `tokenizer_config.json` (no
  `chat_template.jinja` in either snapshot).

## 4. Definition of done, per work package

Every WP lands with: its modules, its `tests/nextdyn/` group green against the fake server,
no regression in `tests/ard` and `tests/study`, and its cell kinds driven by
`tests/nextdyn/test_pipeline_dry_run.py` on four synthetic items (R3). Spec §18 names are used
verbatim; `tests/nextdyn/test_spec18_index.py` lists all 74 and reports which are not yet
collected.

Canonical test command:

```bash
PYTHONPATH=src /orcd/home/002/mabdel03/conda_envs/asys_env/bin/python -m pytest tests/nextdyn tests/ard tests/study -q
```
