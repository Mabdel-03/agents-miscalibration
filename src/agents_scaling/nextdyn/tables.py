"""The single declaration of every table: columns, primary keys, unique constraints,
producers and consumers (spec 17.3; plan §1.4, D14, D15; rule R1).

ARD36 asserted **no** primary key anywhere (``ard/keys.py`` docstring: 14,336 excess
``decisions`` rows, 648 excess ``episodes`` rows, every analysis join silently multiplied).
Here a table *is* its row dataclass: the columns are ``dataclasses.fields``, the primary key
is the fields marked ``metadata={"pk": True}``, the unique constraints are the class
attribute ``__unique__``, and :func:`check_keys` / :func:`assert_referential` are run by
``materialize`` before a tables directory is published.  Nothing else in the package may
declare a column list (``test_every_row_dataclass_is_the_only_declaration_of_its_table_columns``).

Conventions
* ``*_json`` columns hold a JSON-encoded string (lists of ids, mappings); scalar columns hold
  scalars; ``tuple[...]`` columns become parquet list columns.
* Bank draws are ``decisions`` rows (``phase="bank"``, ``episode_id = bank_draw_id``); ``banks``
  is the keyed view materialize derives from them (D14).
* Episode round 0 rows are references (``phase="root_ref"``, ``is_reference=True``,
  ``reuse_of`` = the draw's decision id, which must resolve inside ``decisions``).
* ``grades_private`` keys on ``(generation_uid, grader_hash)`` and carries no boolean column
  (D15); ``semantic_correct`` is ``C``/``W``/``U``.
* Rows of a ``__private__`` table (``grades_private``, ``root_teams_private``) travel in
  ``ItemFile.private_rows`` and are written by the runner under
  ``<run_root>/protected/cells/<cell_id>/items/`` (0600 in 0700), never under
  ``cells/<cell_id>/items/`` (R7 / plan §2.5: the generation lane reads the public cell path).
* ``decisions.receiver_model_hash`` is the generating model of the row for every kind
  (``ids.CompositeKey``); there is no second model column on ``decisions`` (R1).
* Downstream WPs may **append** columns to a row dataclass here (with their WP tag in a
  comment); they may not move a declaration elsewhere, rename or drop a column.

Vendored: ``ard/keys.py:188-249`` (duplicate-group reporting), ``study/aggregate.py:257``
(``_write_parquet`` with the ``.empty`` sentinel).
"""

from __future__ import annotations

import dataclasses
import json
import os
import types
import typing
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final, Hashable, Mapping, Sequence

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import deps
from agents_scaling.nextdyn.errors import ManifestError, MaterializeError
from agents_scaling.nextdyn.ids import BANK_KINDS, CellKind, coerce_kind

SCHEMA_VERSION: Final[int] = 1
EMPTY_SUFFIX: Final[str] = ".empty"
PROTECTED_DIR: Final[str] = "protected"
ITEMS_DIR: Final[str] = "items"
PROTECTED_FILE_MODE: Final[int] = 0o600
PROTECTED_DIR_MODE: Final[int] = 0o700

_PK = {"pk": True}


def pk() -> Any:
    """A primary-key column."""
    return field(metadata=_PK)


# --------------------------------------------------------------------------- row dataclasses


@dataclass(frozen=True)
class PlanRow:
    """One planned cell and its terminal census (plan §1.4 ``plan``)."""

    __table__: ClassVar[str] = "plan"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    cell_id: str = pk()
    cell_kind: str
    lane: str
    pool: str | None
    experiment: str
    manifest_hash: str
    priority_rank: int
    n_items: int
    depends_on_json: str
    status: str
    terminal_census_json: str | None
    reuse_of: str | None


PLAN_STATUSES: Final[tuple[str, ...]] = (
    "planned", "running", "complete", "failed", "blocked", "skipped_quota_unfilled", "reused",
)


@dataclass(frozen=True)
class SourceItemRow:
    """Public item row (spec 17.3 ``source_items``; gold lives in ``protected/gold``)."""

    __table__: ClassVar[str] = "source_items"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = (("normalized_text_hash",),)

    source_item_id: str = pk()
    family: str
    dataset_id: str
    dataset_revision: str
    source_split: str
    original_id: str
    subject: str | None
    level: str | None
    answer_format: str
    n_options: int | None
    option_map_json: str | None
    normalized_text_hash: str
    variant_parent_id: str | None
    prior_use_status: str
    prior_use_sources_json: str
    near_duplicate_of: str | None
    sampling_weight: float
    frame_size: int | None
    frame_rule: str
    split_role: str
    grader_hash: str
    gold_location: str


PRIOR_USE_STATUSES: Final[tuple[str, ...]] = ("used_verified", "used_inferred", "unknown", "reserved_unused")


@dataclass(frozen=True)
class ItemVariantRow:
    __table__: ClassVar[str] = "item_variants"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    item_variant_id: str = pk()
    source_item_id: str
    variant_kind: str
    option_permutation_json: str | None
    rendered_task_hash: str
    prompt_hash: str


@dataclass(frozen=True)
class ItemSplitRow:
    """Split role and sub-panel membership, written before any outcome (plan §2.2)."""

    __table__: ClassVar[str] = "item_splits"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    source_item_id: str = pk()
    family: str
    rank_hex: str
    split_role: str
    pilot_stage: str
    natural_rank: int | None
    constructed_scan_rank: int | None
    sub_second_root: bool
    sub_bridge: bool
    sub_fragmented: bool
    sub_e3a: bool
    sub_e3a_fill: bool
    sub_capture: bool
    sub_capture_natural: bool
    sub_e2a: bool
    sub_natprot: bool
    assigned_at_manifest_hash: str


@dataclass(frozen=True)
class BankRow:
    """Keyed view of the bank draws in ``decisions`` (D14)."""

    __table__: ClassVar[str] = "banks"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("generation_uid",),
        ("cell_kind", "source_item_id", "item_variant_id", "model_hash", "budget_hash", "draw_index"),
    )

    bank_draw_id: str = pk()
    decision_id: str
    generation_uid: str
    cell_kind: str
    bank_role: str
    source_item_id: str
    item_variant_id: str
    model_hash: str
    budget_hash: str
    draw_index: int
    extension_stage: int
    request_hash: str
    content_hash: str | None
    parse_status: str
    validity: bool
    cell_id: str


@dataclass(frozen=True)
class RootTeamRow:
    """Public root-team row: opaque tag, ordered member uids, focal slot, export hash."""

    __table__: ClassVar[str] = "root_teams"
    # WP4 correction (plan §2.4, §1.3): the unique constraint is ``construction_hash``, not
    # ``export_hash``.  A *bucket* is one exported file and a paired bucket holds BOTH members
    # of the declared pair (``construct.seal_root_teams(run_root, family, bucket, teams, ...)``
    # -> ``plan/root_teams/<family>/<bucket>.json``, which ``bundles.load_bucket`` reads as a
    # tuple of teams), so the two teams of one pair share an ``export_hash`` by construction and
    # the original constraint could never hold on a real run.  ``construction_hash`` is the
    # per-team content address (rule version, composition, n, item, variant, ordered members,
    # focal slot) and carries the invariant that was meant: two *different* teams are never the
    # same construction.  Appended, never moved or dropped (R1).
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = (("construction_hash",),)

    root_team_id: str = pk()
    team_tag: str
    family: str
    source_item_id: str
    item_variant_id: str
    cohort: str
    ordered_generation_uids_json: str
    ordered_decision_ids_json: str
    n_independent_generations: int
    focal_slot: int | None
    construction_rule_version: str
    construction_hash: str
    export_hash: str
    shared_donor_uids_json: str
    bucket: str


@dataclass(frozen=True)
class RootTeamPrivateRow:
    """Protected: composition, C0, w*, slot states, eligibility (never exported)."""

    __table__: ClassVar[str] = "root_teams_private"
    __private__: ClassVar[bool] = True
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    root_team_id: str = pk()
    team_tag: str
    composition: str
    c0: int
    w_star_class: str | None
    correct_class: str | None
    slot_states_json: str
    eligibility_json: str
    extension_stage_used: int


@dataclass(frozen=True)
class RequestRow:
    """One physical request (spec 17.3 ``requests``); two per continued draw."""

    __table__: ClassVar[str] = "requests"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    request_hash: str = pk()
    generation_uid: str
    phase_index: int
    first_decision_id: str | None
    prompt_token_hash: str
    completion_token_hash: str
    decoding_hash: str
    seed_hex: str
    model_hash: str
    tokenizer_hash: str
    engine_digest: str
    adapter: str
    endpoint_id: str
    replica_id: str | None
    host: str | None
    gpu_type: str | None
    partition: str | None
    server_batch_id: str | None
    arrival_ts: float
    finish_ts: float
    attempts_json: str
    uncertain_execution: bool
    aliased: bool
    prompt_tokens: int
    completion_tokens: int
    cached_prefill_tokens: int
    latency_s: float
    cell_id: str
    producer_json: str


@dataclass(frozen=True)
class DecisionRow:
    """One intended observation (spec 17.2/17.3).  Built ONLY by ``telemetry.decision_row`` (R6)."""

    __table__: ClassVar[str] = "decisions"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("cell_kind", "episode_id", "round", "agent_slot", "phase"),
    )

    decision_id: str = pk()
    # -- the twelve composite-key fields, in spec order -------------------------------
    study_id: str
    cell_kind: str
    source_item_id: str
    item_variant_id: str
    root_team_id: str
    protocol_hash: str
    receiver_model_hash: str
    budget_hash: str
    replicate_id: int
    round: int
    agent_slot: int
    phase: str
    # -- lineage ----------------------------------------------------------------------
    episode_id: str
    cell_id: str
    display_label: str | None
    generation_uid: str
    content_hash: str | None
    request_hashes_json: str
    aliased: bool
    reuse_of: str | None
    is_reference: bool
    first_decision_id: str | None
    bundle_export_hash: str | None
    parent_decision_ids_json: str
    seen_transmission_ids_json: str
    # -- public output ----------------------------------------------------------------
    native_reasoning_ref: str | None
    public_reasoning: str | None
    shareable_claims_json: str | None
    final_answer_raw: str | None
    canonical_class: str | None
    confidence: float | None
    answer_logprob_sum: float | None
    answer_logprob_mean: float | None
    answer_token_count: int | None
    logprob_norm_scheme: str | None
    parse_status: str
    #: Did the escape salvage rule have to run on this draw (schema.repair_escapes, A5)?
    #: Counted so analysis can show the repairs are not concentrated in one *arm*; they are
    #: expected to concentrate in one *family*, which is the reason the rule exists.
    escape_repaired: bool
    field_order_violation: bool
    validity: bool
    delivery_success: bool
    infrastructure_status: str
    termination_reason: str
    # -- generation ------------------------------------------------------------------
    boundary_injected: bool
    continuation_used: bool
    discarded_span_tokens: int
    thinking_tokens: int
    final_tokens: int
    prompt_tokens: int
    cached_prefill_tokens: int
    token_offsets_json: str | None
    # -- metering ---------------------------------------------------------------------
    endpoint_id: str | None
    latency_s: float | None
    flops_prefill: int | None
    flops_decode: int | None
    flops_total: int | None
    linear_compute_proxy: int | None


@dataclass(frozen=True)
class TransmissionRow:
    """One delivered (or filler) packet slot of one receiving decision."""

    __table__: ClassVar[str] = "transmissions"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("receiver_decision_id", "sender_slot", "source_round"),
    )

    transmission_id: str = pk()
    episode_id: str
    cell_kind: str
    protocol_hash: str
    source_round: int
    destination_round: int
    sender_slot: int
    receiver_slot: int
    receiver_decision_id: str
    source_decision_id: str | None
    packet_hash: str
    original_packet_hash: str | None
    transmitted_tokens: int
    counter_tokenizer_hash: str
    transmitted_tokens_receiver: int | None
    original_tokens: int | None
    truncated_fields_json: str
    filler_flag: bool
    filler_tokens: int | None
    packet_content_mode: str
    declared_edge: str
    display_position: int
    # WP5 append (spec 9.2 "Log token length and error position"): where the injected error of
    # an E2b wrong trace sits along the answer chain, relative in [0, 1).  The *token length*
    # half of that sentence is already ``transmitted_tokens`` above, so only the position
    # needed a home.  Null on every episode packet slot and on a correct E2b packet; appended
    # last with a default so ``packets.transmissions_for`` keeps its call shape (R1: this is
    # the one declaration of the column).
    error_position_rel: float | None = None


@dataclass(frozen=True)
class SelectionRow:
    """One selector applied to one sealed pool; grades joined only afterwards."""

    __table__: ClassVar[str] = "selections"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("selector", "selector_version", "pool_id", "pool_scope"),
    )

    selection_id: str = pk()
    episode_id: str
    cell_kind: str
    source_item_id: str
    round: int
    selector: str
    selector_version: str
    pool_id: str
    pool_scope: str
    candidate_decision_ids_json: str
    valid_count: int
    selected_decision_id: str | None
    selected_class: str | None
    tied_class_ids_json: str
    tie_broken: bool
    fallback_status: str
    audit_cell_id: str | None
    cost_tokens: int
    cost_flops: int | None


FALLBACK_STATUSES: Final[tuple[str, ...]] = (
    "none", "audit_failed_fallback_plurality", "unexecuted", "no_valid_candidate",
)


@dataclass(frozen=True)
class ToolEventRow:
    """One ``check_step`` request of an E2b receiver (spec 9.3)."""

    __table__: ClassVar[str] = "tool_events"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = (("decision_id", "request_index"),)

    tool_event_id: str = pk()
    decision_id: str
    episode_id: str
    request_index: int
    packet_id: str | None
    step_id: int | None
    access_condition: str
    status: str
    elapsed_s: float
    duplicate: bool
    request_json: str
    response_json: str


TOOL_STATUSES: Final[tuple[str, ...]] = ("valid", "invalid", "disabled", "bad_request", "over_budget")


@dataclass(frozen=True)
class GradePrivateRow:
    """Protected tri-state grade per physical sample (D15).  No boolean column exists."""

    __table__: ClassVar[str] = "grades_private"
    __private__: ClassVar[bool] = True
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    generation_uid: str = pk()
    grader_hash: str = pk()
    decision_ids_json: str
    source_item_id: str
    family: str
    semantic_correct: str
    grading_status: str
    u_reason: str | None
    extracted: str | None
    canonical_class: str | None
    gold_ref_hash: str
    audit_status: str
    grader_version: str


SEMANTIC_STATES: Final[tuple[str, ...]] = ("C", "W", "U")


@dataclass(frozen=True)
class ActivationRow:
    __table__: ClassVar[str] = "activations_manifest"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    decision_id: str = pk()
    layer: int = pk()
    anchor: str = pk()
    model_hash: str
    token_offset: int | None
    dtype: str | None
    shape_json: str | None
    storage_hash: str | None
    storage_path: str | None
    replay_parity: str
    missing_reason: str | None
    pre_confidence_eligible: bool | None


@dataclass(frozen=True)
class MetricRow:
    """One estimate of the analysis (spec 17.3 ``metrics``)."""

    __table__: ClassVar[str] = "metrics"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("estimand_id", "estimand_version", "population", "model_scope", "task_scope", "horizon"),
    )

    metric_id: str = pk()
    estimand_id: str
    estimand_version: str
    population: str
    model_scope: str
    task_scope: str
    horizon: str
    weighting: str
    denominator: str
    missingness_convention: str
    estimate: float | None
    ci_lo: float | None
    ci_hi: float | None
    p_value: float | None
    multiplicity_family: str
    n_items: int | None
    n_clusters: int | None
    artifact_sha256: str


ROW_TYPES: Final[tuple[type, ...]] = (
    PlanRow, SourceItemRow, ItemVariantRow, ItemSplitRow, BankRow, RootTeamRow, RootTeamPrivateRow,
    RequestRow, DecisionRow, TransmissionRow, SelectionRow, ToolEventRow, GradePrivateRow,
    ActivationRow, MetricRow,
)

#: Derived tables (plan §1.4 "derived: ... at analyze"): keys declared now, columns declared
#: when WP7 appends their row dataclasses to this module.
DERIVED_PRIMARY_KEYS: Final[Mapping[str, tuple[str, ...]]] = {
    "episodes": ("cell_kind", "episode_id"),
    "team_rounds": ("cell_kind", "episode_id", "round"),
    "agent_rounds": ("cell_kind", "episode_id", "round", "agent_slot"),
    "transitions": ("cell_kind", "episode_id", "round", "agent_slot"),
    "difficulty": ("source_item_id", "model_hash", "budget_hash"),
    "bank_summary": ("source_item_id", "model_hash", "budget_hash", "bank_role"),
    "selections_graded": ("selection_id",),
    "efficiency_frontier": ("protocol_hash", "budget_level", "family"),
    "tool_usage_panel": ("cell_kind", "episode_id"),
    "e2a_e1_link": ("source_item_id", "model_hash"),
}


# --------------------------------------------------------------------------- schemas

_NULLABLE_MARKERS = ("| None",)


@dataclass(frozen=True)
class Column:
    name: str
    type: str          # "str" | "int" | "float" | "bool" | "list[str]" | "list[int]"
    nullable: bool


@dataclass(frozen=True)
class TableSchema:
    name: str
    row_type: type
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...]
    unique: tuple[tuple[str, ...], ...]
    producer: str
    consumers: tuple[str, ...]
    private: bool = False

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


def _type_hints(row_type: type) -> dict[str, Any]:
    return typing.get_type_hints(row_type)


def _column_type(hint: Any) -> tuple[str, bool]:
    """``(type name, nullable)`` of one field annotation."""
    nullable = False
    origin = typing.get_origin(hint)
    if origin is types.UnionType or origin is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        nullable = len(args) != len(typing.get_args(hint))
        if len(args) != 1:
            raise ManifestError(f"unsupported column union {hint!r}")
        hint = args[0]
        origin = typing.get_origin(hint)
    if origin is tuple:
        inner = typing.get_args(hint)[0]
        return f"list[{inner.__name__}]", nullable
    if hint in (str, int, float, bool):
        return hint.__name__, nullable
    raise ManifestError(f"unsupported column type {hint!r}")


def schema_of(row_type: type) -> TableSchema:
    """The :class:`TableSchema` a row dataclass declares (columns = ``dataclasses.fields``)."""
    if not dataclasses.is_dataclass(row_type):
        raise ManifestError(f"{row_type!r} is not a row dataclass")
    name = getattr(row_type, "__table__", None)
    if not name:
        raise ManifestError(f"{row_type.__name__} declares no __table__")
    hints = _type_hints(row_type)
    columns = []
    primary = []
    for f in dataclasses.fields(row_type):
        type_name, nullable = _column_type(hints[f.name])
        if f.metadata.get("pk"):
            primary.append(f.name)
            if nullable:
                raise ManifestError(f"{name}.{f.name} is a primary-key column and cannot be nullable")
        columns.append(Column(f.name, type_name, nullable))
    if not primary:
        raise ManifestError(f"{name} declares no primary key")
    unique = tuple(tuple(u) for u in getattr(row_type, "__unique__", ()))
    names = {c.name for c in columns}
    for constraint in unique:
        missing = [c for c in constraint if c not in names]
        if missing:
            raise ManifestError(f"{name}: unique constraint {constraint} names unknown columns {missing}")
    producer, consumers = _PRODUCERS[name]
    return TableSchema(name=name, row_type=row_type, columns=tuple(columns), primary_key=tuple(primary),
                       unique=unique, producer=producer, consumers=tuple(consumers),
                       private=bool(getattr(row_type, "__private__", False)))


_P = "agents_scaling.nextdyn."
#: table -> (producer module, consumer modules).  Importable dotted names (R3); a consumer
#: that does not exist yet is "pending" in the tests, never silently absent.
_PRODUCERS: Final[Mapping[str, tuple[str, tuple[str, ...]]]] = {
    "plan": (_P + "plan", (_P + "materialize", _P + "status", _P + "release")),
    "source_items": (_P + "items_build", (_P + "items", _P + "inventory", _P + "analysis")),
    "item_variants": (_P + "items_build", (_P + "items", _P + "analysis")),
    "item_splits": (_P + "inventory", (_P + "plan", _P + "analysis")),
    "banks": (_P + "materialize", (_P + "banks", _P + "construct", _P + "analysis")),
    "root_teams": (_P + "construct", (_P + "materialize", _P + "analysis")),
    "root_teams_private": (_P + "construct", (_P + "analysis",)),
    "requests": (_P + "telemetry", (_P + "materialize", _P + "analysis")),
    "decisions": (_P + "telemetry", (_P + "materialize", _P + "outcomes", _P + "audit", _P + "capture", _P + "analysis")),
    "transmissions": (_P + "packets", (_P + "materialize", _P + "outcomes")),
    "selections": (_P + "selectors", (_P + "materialize", _P + "analysis")),
    "tool_events": (_P + "checker", (_P + "materialize", _P + "analysis")),
    "grades_private": (_P + "grading", (_P + "construct", _P + "analysis")),
    "activations_manifest": (_P + "capture", (_P + "analysis",)),
    "metrics": (_P + "analysis", (_P + "release", _P + "figures")),
}

TABLES: Final[Mapping[str, TableSchema]] = {t.__table__: schema_of(t) for t in ROW_TYPES}
if set(TABLES) != set(_PRODUCERS):  # pragma: no cover - import-time contract
    raise RuntimeError("every table needs exactly one producer entry")
PRIMARY_KEYS: Final[Mapping[str, tuple[str, ...]]] = {n: s.primary_key for n, s in TABLES.items()}
PRIVATE_TABLES: Final[frozenset[str]] = frozenset(n for n, s in TABLES.items() if s.private)


def schema(name: str) -> TableSchema:
    try:
        return TABLES[name]
    except KeyError as exc:
        raise MaterializeError(f"unknown table {name!r}; declared: {sorted(TABLES)}") from exc


# WP5 append (R3: every producer is declared, none implied).  A table has exactly one *owning*
# producer -- the module that declares the row's shape and writes the common case -- and may
# have a second writer for a case the first cannot express.  ``transmissions`` is the one such
# table: ``packets.transmissions_for`` writes the episode packet slots from a resolved
# ``PacketPlan``, while ``receiver.transmission_rows`` writes the *assigned* receiver slots,
# which have no plan, no sender decision, a ``declared_edge`` of "assigned" and the E2b
# ``error_position_rel``.  Declared here rather than left implied, because an undeclared
# second writer is how two row constructors drift apart (ARD36 D-05).
CO_PRODUCERS: Final[Mapping[str, tuple[str, ...]]] = {
    "transmissions": (_P + "receiver",),
}
if set(CO_PRODUCERS) - set(_PRODUCERS):                     # pragma: no cover - import-time
    raise RuntimeError("a co-producer names a table that does not exist")


def producers(name: str) -> tuple[str, ...]:
    """Every module that writes rows of ``name``: the owning producer first (R3)."""
    return (schema(name).producer, *CO_PRODUCERS.get(name, ()))


# --------------------------------------------------------------------------- rows <-> dicts


def _check_scalar(table: str, column: Column, value: Any) -> Any:
    if value is None:
        if not column.nullable:
            raise MaterializeError(f"{table}.{column.name} is not nullable")
        return None
    if column.type.startswith("list["):
        if not isinstance(value, (list, tuple)):
            raise MaterializeError(f"{table}.{column.name} expects a list, got {type(value).__name__}")
        return tuple(value)
    expected = {"str": str, "int": int, "float": (int, float), "bool": bool}[column.type]
    if isinstance(value, bool) and column.type != "bool":
        raise MaterializeError(f"{table}.{column.name} expects {column.type}, got a bool")
    if not isinstance(value, expected):
        raise MaterializeError(f"{table}.{column.name} expects {column.type}, got {type(value).__name__}")
    return float(value) if column.type == "float" else value


def project(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Strict whitelist projection (``ard/compact.py:_project`` pattern, made fatal).

    Undeclared columns raise; declared columns absent from ``row`` are ``None`` when nullable
    and raise otherwise; primary-key columns must be present and non-null.
    """
    spec = schema(table)
    unknown = sorted(set(row) - set(spec.column_names))
    if unknown:
        raise MaterializeError(f"{table}: undeclared column(s) {unknown}; declare them on "
                               f"{spec.row_type.__name__} or drop them")
    out: dict[str, Any] = {}
    for column in spec.columns:
        out[column.name] = _check_scalar(table, column, row.get(column.name))
    return out


def row_from_dict(row_type: type, data: Mapping[str, Any]):
    """A row instance from a mapping; unknown keys and missing keys are both errors."""
    spec = schema_of(row_type)
    unknown = sorted(set(data) - set(spec.column_names))
    if unknown:
        raise MaterializeError(f"{spec.name}: unknown key(s) {unknown}")
    missing = sorted(set(spec.column_names) - set(data))
    if missing:
        raise MaterializeError(f"{spec.name}: missing key(s) {missing}")
    return row_type(**project(spec.name, data))


def row_to_dict(row: Any) -> dict[str, Any]:
    if not dataclasses.is_dataclass(row) or type(row) not in ROW_TYPES:
        raise MaterializeError(f"{type(row).__name__} is not a declared row type")
    return {f.name: (list(v) if isinstance(v, tuple) else v)
            for f, v in ((f, getattr(row, f.name)) for f in dataclasses.fields(row))}


# --------------------------------------------------------------------------- item files


class ItemStatus(str, Enum):
    COMPLETE = "complete"
    CONTEXT_FAILURE = "context_failure"
    FAILED = "failed"


def item_path(run_root: str | os.PathLike, cell_id: str, source_item_id: str) -> Path:
    """``<run_root>/cells/<cell_id>/items/<source_item_id>.json`` (the public item file)."""
    return deps.cell_dir(run_root, cell_id) / ITEMS_DIR / f"{source_item_id}.json"


def protected_item_path(run_root: str | os.PathLike, cell_id: str, source_item_id: str) -> Path:
    """``<run_root>/protected/cells/<cell_id>/items/<source_item_id>.json``: where the private
    rows of one item go (gold-derived grades, compositions).  Never under ``cells/``."""
    return Path(run_root) / PROTECTED_DIR / deps.CELLS_DIR / cell_id / ITEMS_DIR / f"{source_item_id}.json"


def _check_rows(label: str, rows: Mapping[str, Any], *, private: bool) -> dict[str, tuple[Any, ...]]:
    out: dict[str, tuple[Any, ...]] = {}
    for table, items in dict(rows).items():
        if table not in TABLES:
            raise ManifestError(f"item file {label} name an undeclared table {table!r}; declared: {sorted(TABLES)}")
        spec = schema(table)
        if spec.private != private:
            where = "private_rows" if spec.private else "rows"
            raise ManifestError(f"item file {label} hold the {'protected' if spec.private else 'public'} table "
                                f"{table!r}; it belongs in ItemFile.{where} (R7: gold-derived rows never "
                                "land under cells/)")
        for item in items:
            if type(item) is not spec.row_type:
                raise ManifestError(f"item file {label}[{table!r}] holds a {type(item).__name__}, "
                                    f"expected {spec.row_type.__name__}")
        out[table] = tuple(items)
    return out


@dataclass(frozen=True)
class ItemFile:
    """The rows one worker produced for one item, in two channels.

    ``rows`` (public tables) is written by the runner to
    ``cells/<cell_id>/items/<source_item_id>.json``; ``private_rows`` (``__private__`` tables:
    ``grades_private``, ``root_teams_private``) is written to
    ``protected/cells/<cell_id>/items/<source_item_id>.json`` with mode 0600 in a 0700
    directory.  A private table under ``rows`` (or a public one under ``private_rows``) is
    refused, so a GRADE worker cannot put gold-derived grades on the public cell path that
    ``deps.dependency_status`` opens and the generation lane reads (R7, plan §2.5).

    The runner writes both files atomically after the worker returns; ``materialize`` reads
    them back through :meth:`read` and projects every row through the table schema.  Unknown
    tables and unknown keys are refused in both directions (R1: no dict-key access).
    """

    schema_version: int
    cell_id: str
    cell_kind: CellKind
    source_item_id: str
    status: ItemStatus
    bundle_export_hash: str | None
    rows: Mapping[str, tuple[Any, ...]]
    private_rows: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)

    _TOP_KEYS: ClassVar[tuple[str, ...]] = (
        "schema_version", "cell_id", "cell_kind", "source_item_id", "status", "bundle_export_hash", "rows",
    )
    _PRIVATE_TOP_KEYS: ClassVar[tuple[str, ...]] = (
        "schema_version", "cell_id", "cell_kind", "source_item_id", "private_rows",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_kind", coerce_kind(self.cell_kind))
        object.__setattr__(self, "status", ItemStatus(self.status))
        if int(self.schema_version) != SCHEMA_VERSION:
            raise ManifestError(f"item file schema_version {self.schema_version} != {SCHEMA_VERSION}")
        if not self.cell_id or not self.source_item_id:
            raise ManifestError("item file needs cell_id and source_item_id")
        object.__setattr__(self, "rows", _check_rows("rows", self.rows, private=False))
        object.__setattr__(self, "private_rows", _check_rows("private_rows", self.private_rows, private=True))

    @property
    def has_private_rows(self) -> bool:
        return any(self.private_rows.values())

    def to_dict(self) -> dict[str, Any]:
        """The public file: never carries a private row."""
        return {
            "schema_version": int(self.schema_version),
            "cell_id": self.cell_id,
            "cell_kind": self.cell_kind.value,
            "source_item_id": self.source_item_id,
            "status": self.status.value,
            "bundle_export_hash": self.bundle_export_hash,
            "rows": {table: [row_to_dict(r) for r in items] for table, items in self.rows.items()},
        }

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "cell_id": self.cell_id,
            "cell_kind": self.cell_kind.value,
            "source_item_id": self.source_item_id,
            "private_rows": {table: [row_to_dict(r) for r in items] for table, items in self.private_rows.items()},
        }

    @staticmethod
    def _dumps(data: Mapping[str, Any]) -> str:
        return json.dumps(dict(data), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"

    def to_json(self) -> str:
        return self._dumps(self.to_dict())

    def to_private_json(self) -> str:
        return self._dumps(self.to_private_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], private: Mapping[str, Any] | None = None) -> "ItemFile":
        if not isinstance(data, Mapping):
            raise ManifestError("item file is not a JSON object")
        unknown = sorted(set(data) - set(cls._TOP_KEYS))
        if unknown:
            raise ManifestError(f"item file carries unknown key(s) {unknown}")
        missing = sorted(set(cls._TOP_KEYS) - set(data))
        if missing:
            raise ManifestError(f"item file lacks key(s) {missing}")
        raw_rows = data["rows"]
        if not isinstance(raw_rows, Mapping):
            raise ManifestError("item file 'rows' must be an object keyed by table name")
        rows = {table: tuple(row_from_dict(schema(table).row_type, r) for r in items)
                for table, items in raw_rows.items()}
        private_rows: dict[str, tuple[Any, ...]] = {}
        if private is not None:
            if not isinstance(private, Mapping):
                raise ManifestError("protected item file is not a JSON object")
            unknown = sorted(set(private) - set(cls._PRIVATE_TOP_KEYS))
            if unknown:
                raise ManifestError(f"protected item file carries unknown key(s) {unknown}")
            missing = sorted(set(cls._PRIVATE_TOP_KEYS) - set(private))
            if missing:
                raise ManifestError(f"protected item file lacks key(s) {missing}")
            for name in ("schema_version", "cell_id", "cell_kind", "source_item_id"):
                if private[name] != data[name]:
                    raise ManifestError(f"protected item file {name}={private[name]!r} does not match the "
                                        f"public file ({data[name]!r})")
            raw_private = private["private_rows"]
            if not isinstance(raw_private, Mapping):
                raise ManifestError("protected item file 'private_rows' must be an object keyed by table name")
            private_rows = {table: tuple(row_from_dict(schema(table).row_type, r) for r in items)
                            for table, items in raw_private.items()}
        return cls(schema_version=data["schema_version"], cell_id=data["cell_id"],
                   cell_kind=data["cell_kind"], source_item_id=data["source_item_id"],
                   status=data["status"], bundle_export_hash=data["bundle_export_hash"], rows=rows,
                   private_rows=private_rows)

    @classmethod
    def from_json(cls, text: str, private_text: str | None = None) -> "ItemFile":
        try:
            data = json.loads(text)
            private = None if private_text is None else json.loads(private_text)
        except ValueError as exc:
            raise ManifestError(f"item file is not valid JSON ({exc})") from exc
        return cls.from_dict(data, private)

    def write(self, path: str | os.PathLike, *, protected_path: str | os.PathLike | None = None) -> None:
        """Write the public file at ``path``; private rows go to ``protected_path`` (mode 0600,
        directory 0700), which is required whenever there are any.  ``protected_path`` may
        never sit under a ``cells/`` directory."""
        path = Path(path)
        if self.has_private_rows:
            if protected_path is None:
                raise ManifestError(f"{self.cell_id}/{self.source_item_id}: private rows "
                                    f"{sorted(t for t, r in self.private_rows.items() if r)} need a protected_path "
                                    "(tables.protected_item_path); they never land under cells/")
            protected = Path(protected_path)
            if deps.CELLS_DIR in protected.parts and PROTECTED_DIR not in protected.parts:
                raise ManifestError(f"protected_path {protected} is under cells/ and not under {PROTECTED_DIR}/")
            if protected.resolve() == path.resolve():
                raise ManifestError("protected_path must differ from the public item path")
            protected.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(protected.parent, PROTECTED_DIR_MODE)
            deps.atomic_write_bytes(protected, self.to_private_json().encode("utf-8"))
            os.chmod(protected, PROTECTED_FILE_MODE)
        deps.atomic_write_bytes(path, self.to_json().encode("utf-8"))

    @classmethod
    def read(cls, path: str | os.PathLike, *, protected_path: str | os.PathLike | None = None) -> "ItemFile":
        private_text = None
        if protected_path is not None and Path(protected_path).exists():
            private_text = Path(protected_path).read_text(encoding="utf-8")
        return cls.from_json(Path(path).read_text(encoding="utf-8"), private_text)


# --------------------------------------------------------------------------- parquet io


def write_parquet_rows(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """``study/aggregate.py:257 _write_parquet`` with the column order fixed by the schema.

    No rows -> a ``<name>.empty`` sentinel and no parquet file (a vacuously unique table,
    distinguishable from a missing one).  Otherwise tmp + ``os.replace``.
    """
    path = Path(path)
    if not rows:
        deps.atomic_write_bytes(path.with_suffix(EMPTY_SUFFIX), b"no rows\n")
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([{k: row.get(k) for k in columns} for row in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def write_table(rows: Sequence[Any], path: str | os.PathLike, *, table: str | None = None) -> None:
    """Write row dataclasses of one table.  Every row must be that table's row type."""
    path = Path(path)
    name = table or (path.name[: -len(path.suffix)] if path.suffix else path.name)
    spec = schema(name)
    dicts = []
    for row in rows:
        if type(row) is not spec.row_type:
            raise MaterializeError(f"{name}: row of type {type(row).__name__}, expected {spec.row_type.__name__}")
        dicts.append(project(name, row_to_dict(row)))
    write_parquet_rows(path, [{k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()} for d in dicts],
                       spec.column_names)


def read_table(tables_dir: str | os.PathLike, name: str, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Rows of ``<tables_dir>/<name>.parquet`` as dicts; ``[]`` for the ``.empty`` sentinel."""
    root = Path(tables_dir)
    path = root / f"{name}.parquet"
    if not path.exists():
        if (root / f"{name}{EMPTY_SUFFIX}").exists():
            return []
        raise MaterializeError(f"{path} does not exist and no {EMPTY_SUFFIX} sentinel is present")
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=list(columns) if columns else None).to_pylist()


# --------------------------------------------------------------------------- key assertions
# Vendored from ard/keys.py:75-82 (records) and :188-249 (check_table / check_keys): the same
# duplicate-group counting and worst-first ordering, keyed by TABLES instead of a hand list, and
# extended to the declared unique constraints.

#: ``transmissions.declared_edge`` of a receiver assay's assigned packet (``receiver.py``
#: declares it; spelled here rather than imported because ``tables`` sits below every producer
#: in the import order and importing one back would make the row dataclasses unimportable).
#: ``tests/nextdyn/test_materialize.py`` holds the two spellings together.
DECLARED_EDGE_ASSIGNED: Final[str] = "assigned"

EXAMPLE_LIMIT: Final[int] = 5
STATUS_OK: Final[str] = "ok"
STATUS_DUPLICATE: Final[str] = "duplicate_keys"
STATUS_UNVERIFIABLE: Final[str] = "unverifiable"


@dataclass(frozen=True)
class DuplicateKey:
    values: tuple[str, ...]
    n_rows: int


@dataclass(frozen=True)
class KeyCheck:
    table: str
    key: tuple[str, ...]
    constraint: str          # "primary_key" | "unique"
    n_rows: int
    n_distinct: int
    n_duplicate_groups: int
    n_excess_rows: int
    examples: tuple[DuplicateKey, ...]
    status: str
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


@dataclass(frozen=True)
class KeyViolation:
    table: str
    key: tuple[str, ...]
    constraint: str
    n_rows: int
    n_distinct: int
    n_duplicate_groups: int
    examples: tuple[DuplicateKey, ...]
    note: str = ""


def _hashable(value: Any) -> Hashable:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _check_one(root: Path, table: str, key: Sequence[str], constraint: str) -> KeyCheck:
    declared = tuple(key)
    path = root / f"{table}.parquet"
    if not path.exists():
        if (root / f"{table}{EMPTY_SUFFIX}").exists():
            return KeyCheck(table, declared, constraint, 0, 0, 0, 0, (), STATUS_OK,
                            "empty sentinel: the table has no rows")
        return KeyCheck(table, declared, constraint, 0, 0, 0, 0, (), STATUS_UNVERIFIABLE,
                        f"{path} does not exist")
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = [c for c in declared if c not in available]
    if missing:
        return KeyCheck(table, (), constraint, int(pq.ParquetFile(path).metadata.num_rows), 0, 0, 0, (),
                        STATUS_UNVERIFIABLE, f"missing key column(s) {missing}")
    arrow = pq.read_table(path, columns=list(declared))
    values = [arrow.column(name).to_pylist() for name in declared]
    counts = Counter(tuple(_hashable(v) for v in row) for row in zip(*values))
    n_rows = sum(counts.values())
    duplicates = {vals: n for vals, n in counts.items() if n > 1}
    worst = sorted(duplicates.items(), key=lambda kv: (-kv[1], tuple(str(v) for v in kv[0])))
    examples = tuple(DuplicateKey(tuple(str(v) for v in vals), n) for vals, n in worst[:EXAMPLE_LIMIT])
    return KeyCheck(table, declared, constraint, n_rows, len(counts), len(duplicates),
                    n_rows - len(counts), examples, STATUS_OK if not duplicates else STATUS_DUPLICATE)


def run_checks(tables_dir: str | os.PathLike, *, tables: Mapping[str, TableSchema] = TABLES) -> list[KeyCheck]:
    """Every primary key and unique constraint of every declared table, clean ones included."""
    root = Path(tables_dir)
    out: list[KeyCheck] = []
    for name, spec in tables.items():
        out.append(_check_one(root, name, spec.primary_key, "primary_key"))
        for constraint in spec.unique:
            out.append(_check_one(root, name, constraint, "unique"))
    return out


def check_keys(tables_dir: str | os.PathLike, *, strict: bool = True) -> list[KeyViolation]:
    """Violations, worst first.  A measured duplicate is a violation under either setting;
    ``strict`` decides whether an absent table or key column also is."""
    checks = run_checks(tables_dir)
    violations = [
        KeyViolation(c.table, c.key, c.constraint, c.n_rows, c.n_distinct, c.n_duplicate_groups, c.examples, c.note)
        for c in checks
        if c.status == STATUS_DUPLICATE or (strict and c.status == STATUS_UNVERIFIABLE)
    ]
    violations.sort(key=lambda v: (-v.n_duplicate_groups, v.table, v.constraint))
    return violations


def assert_keys(tables_dir: str | os.PathLike, *, strict: bool = True) -> None:
    violations = check_keys(tables_dir, strict=strict)
    if violations:
        lines = [f"{v.table} {v.constraint}{v.key}: {v.n_duplicate_groups} duplicate groups "
                 f"(e.g. {[e.values for e in v.examples[:2]]}) {v.note}" for v in violations]
        raise MaterializeError("primary-key assertion failed:\n  " + "\n  ".join(lines))


# --------------------------------------------------------------------------- referential assertions


def referential_violations(tables_dir: str | os.PathLike) -> list[str]:
    """Plan §1.4 / D14 materialize-time assertions, as human-readable violations.

    1. ``decisions.reuse_of`` resolves inside ``decisions``; reference rows carry it, bank
       rows never do.
    2. A difficulty-bank draw is never a root-team member (spec 5.3).
    3. ``transmissions.source_round == destination_round - 1`` (spec 6.4 synchronous updates).
    4. Every ``episode_id`` maps to exactly one ``cell_kind`` (R4).
    5. Bank rows sit at ``phase == "bank"`` with an empty ``root_team_id``.
    6. Every decision carries the generating model in ``receiver_model_hash`` (spec 17.2).
    """
    root = Path(tables_dir)
    out: list[str] = []
    decisions = read_table(root, "decisions", ["decision_id", "cell_kind", "episode_id", "phase", "reuse_of",
                                                "is_reference", "generation_uid", "root_team_id",
                                                "receiver_model_hash"])
    ids = {d["decision_id"] for d in decisions}
    kinds_by_episode: dict[str, set[str]] = {}
    difficulty_uids: set[str] = set()
    for d in decisions:
        kinds_by_episode.setdefault(d["episode_id"], set()).add(d["cell_kind"])
        kind = d["cell_kind"]
        if not d["receiver_model_hash"]:
            out.append(f"decisions {d['decision_id']}: receiver_model_hash is empty (the generating model is "
                       "part of every key, spec 17.2)")
        if kind == CellKind.DIFFICULTY_BANK.value:
            difficulty_uids.add(d["generation_uid"])
        if kind in {k.value for k in BANK_KINDS}:
            if d["phase"] != K.PHASE_BANK or d["root_team_id"]:
                out.append(f"decisions {d['decision_id']}: a {kind} row must carry phase={K.PHASE_BANK!r} "
                           f"and root_team_id='' (got {d['phase']!r}, {d['root_team_id']!r})")
            if d["reuse_of"] is not None:
                out.append(f"decisions {d['decision_id']}: a bank draw never reuses another decision")
        if d["phase"] == K.PHASE_ROOT_REF and (not d["is_reference"] or d["reuse_of"] is None):
            out.append(f"decisions {d['decision_id']}: a root_ref row must be a reference with reuse_of set")
        if d["reuse_of"] is not None and d["reuse_of"] not in ids:
            out.append(f"decisions {d['decision_id']}: reuse_of {d['reuse_of']} does not resolve in decisions")
    for episode, kinds in sorted(kinds_by_episode.items()):
        if len(kinds) != 1:
            out.append(f"episode_id {episode} maps to {sorted(kinds)} (must be exactly one cell_kind)")
    for team in read_table(root, "root_teams", ["root_team_id", "ordered_generation_uids_json"]):
        try:
            members = json.loads(team["ordered_generation_uids_json"] or "[]")
        except ValueError:
            out.append(f"root_teams {team['root_team_id']}: ordered_generation_uids_json is not JSON")
            continue
        leaked = sorted(set(members) & difficulty_uids)
        if leaked:
            out.append(f"root_teams {team['root_team_id']}: difficulty-bank draws {leaked[:3]} are members "
                       "(spec 5.3: the difficulty bank is never a root)")
        if len(set(members)) != len(members):
            out.append(f"root_teams {team['root_team_id']}: a generation record appears twice")
    # WP4 correction (spec 6.4 vs spec 9.1): the "previous round only" rule is about the
    # *episode* packet slots a resolved graph produced.  A receiver assay's packets are
    # ASSIGNED -- two frozen candidates the harness shows, not another agent's previous output
    # -- and its episode has a single round, so ``source_round == destination_round == 0`` by
    # construction and the original form of this check could never hold once E2a/E2b ran.  The
    # check is scoped by ``declared_edge`` (``packets.DECLARED_EDGE_GRAPH`` / ``_FILLER`` are
    # the graph-produced slots, ``receiver.DECLARED_EDGE_ASSIGNED`` the assay's) and the
    # assigned slots get their own assertion: they never claim to come from an earlier round.
    # Appended, never dropped (R1).
    for row in read_table(root, "transmissions",
                          ["transmission_id", "source_round", "destination_round", "declared_edge"]):
        if row["declared_edge"] == DECLARED_EDGE_ASSIGNED:
            if row["source_round"] != row["destination_round"]:
                out.append(f"transmissions {row['transmission_id']}: an assigned receiver packet "
                           f"carries source_round {row['source_round']} != destination_round "
                           f"{row['destination_round']} (it is shown, not received from a round)")
            continue
        if row["source_round"] != row["destination_round"] - 1:
            out.append(f"transmissions {row['transmission_id']}: source_round "
                       f"{row['source_round']} != destination_round "
                       f"{row['destination_round']} - 1")
    return out


def assert_referential(tables_dir: str | os.PathLike, *, strict: bool = True) -> list[str]:
    violations = referential_violations(tables_dir)
    if violations and strict:
        raise MaterializeError("referential assertion failed:\n  " + "\n  ".join(violations))
    return violations


__all__ = [
    "ActivationRow", "BankRow", "CO_PRODUCERS", "Column", "DERIVED_PRIMARY_KEYS", "DecisionRow",
    "DuplicateKey",
    "EMPTY_SUFFIX", "EXAMPLE_LIMIT", "FALLBACK_STATUSES", "GradePrivateRow", "ITEMS_DIR", "ItemFile",
    "ItemSplitRow", "ItemStatus", "ItemVariantRow", "KeyCheck", "KeyViolation", "MetricRow",
    "PLAN_STATUSES", "PRIMARY_KEYS", "PRIOR_USE_STATUSES", "PRIVATE_TABLES", "PROTECTED_DIR",
    "PROTECTED_DIR_MODE", "PROTECTED_FILE_MODE", "PlanRow",
    "ROW_TYPES", "RequestRow", "RootTeamPrivateRow", "RootTeamRow", "SCHEMA_VERSION",
    "SEMANTIC_STATES", "STATUS_DUPLICATE", "STATUS_OK", "STATUS_UNVERIFIABLE", "SelectionRow",
    "SourceItemRow", "TABLES", "TOOL_STATUSES", "TableSchema", "ToolEventRow", "TransmissionRow",
    "assert_keys", "assert_referential", "check_keys", "item_path", "pk", "project",
    "producers", "protected_item_path", "read_table", "referential_violations", "row_from_dict",
    "row_to_dict",
    "run_checks", "schema", "schema_of", "write_parquet_rows", "write_table",
]


# ============================================================================================
# WP7 (plan §4.1, §4.2, §4.3) -- the derived analysis tables.
#
# ``DERIVED_PRIMARY_KEYS`` above declares the *keys* of the derived tables; this appended
# section is the single declaration of their *columns* (R1), exactly as the module docstring
# anticipates ("columns declared when WP7 appends their row dataclasses to this module").
#
# They are deliberately NOT in ``ROW_TYPES``/``TABLES``: those are the tables ``materialize``
# publishes and ``check_keys`` demands on disk, and a derived table that has not been computed
# yet would otherwise make every materialize run report an unverifiable key.  The derived set
# gets its own registry (:data:`DERIVED_TABLES`), its own key check
# (:func:`check_derived_keys`) and its own writer (:func:`write_derived_table`), all built from
# the same ``dataclasses.fields`` machinery, so there is still exactly one place a derived
# column is named.
#
# Only the derived tables WP7 *produces* are declared here.  ``episodes``, ``difficulty`` and
# ``bank_summary`` are produced by ``materialize``/``banks`` and are declared by those WPs.
# ============================================================================================


@dataclass(frozen=True)
class TeamRoundRow:
    """One ``(episode, round)`` team state (plan §4.1; spec 4.2-4.6, 8.5, 10.2).

    ``c_count + w_count + u_count == n_agents`` is asserted by ``outcomes.team_round``
    (spec 18.3 ``test_c_w_u_counts_sum_to_n``).  Coverage is carried as a lo/hi pair
    everywhere: ``*_lo`` counts only resolved ``C``, ``*_hi`` credits every ``U`` as possibly
    correct, so the pair brackets every feasible labelling of the unresolved slots.
    """

    __table__: ClassVar[str] = "team_rounds"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    cell_kind: str = pk()
    episode_id: str = pk()
    round: int = pk()
    source_item_id: str
    item_variant_id: str
    family: str
    experiment: str
    cohort: str
    composition: str | None
    protocol_hash: str
    display_label: str | None
    model_hash: str
    budget_hash: str
    replicate_id: int
    n_agents: int
    c_count: int
    w_count: int
    u_count: int
    n_valid: int
    fully_graded: bool
    class_hist_json: str
    slot_classes_json: str
    # -- coverage and margin (spec 4.2) -----------------------------------------------
    m_lo: float
    m_hi: float
    q_lo: float
    q_hi: float
    h_lo: float
    h_hi: float
    # -- winning rules (spec 4.5) -----------------------------------------------------
    plurality_class: str | None
    plurality_status: str
    plurality_correct: float | None
    strict_majority_correct: bool
    unanimous_correct: bool
    wrong_unanimity: bool
    # -- concentration ----------------------------------------------------------------
    max_class_share: float | None
    class_entropy_norm: float | None
    wrong_class_max_share: float | None
    wrong_collision_prob: float | None
    # -- minority definitions (spec 4.4) ----------------------------------------------
    arithmetic_correct_minority: bool
    outnumbered_correct: bool
    all_singleton: bool
    # -- focal / construction (spec 10.2) ---------------------------------------------
    w_star_class: str | None
    c0: int | None
    focal_slot: int | None
    focal_class: str | None
    focal_retained_root_class: bool | None
    focal_state: str | None
    nonfocal_init_wrong_now_c: float | None
    nonfocal_init_correct_now_wstar: float | None
    n_nonfocal_init_wrong: int
    n_nonfocal_init_correct: int
    nonfocal_unknown: int
    # -- hazards and preservation, both framings, never merged (plan §4.1) ------------
    baseline_c0_positive: bool
    continuous_preservation_lo: bool | None
    terminal_preservation_lo: bool | None
    at_risk_hazard: bool
    hazard_event: bool | None
    at_risk_discovery: bool
    discovery_event: bool | None
    recovered_lo: bool
    # -- delivered accuracy and the failure decomposition (filled once selections are
    #    graded; null before the label join, which is sealed after the predictions) ----
    a_lo: float | None
    fail_no_history: float | None
    fail_lost: float | None
    fail_selection: float | None


@dataclass(frozen=True)
class AgentRoundRow:
    """One agent's state in one round (plan §4.1 derived ``agent_rounds``)."""

    __table__: ClassVar[str] = "agent_rounds"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    cell_kind: str = pk()
    episode_id: str = pk()
    round: int = pk()
    agent_slot: int = pk()
    decision_id: str
    generation_uid: str
    source_item_id: str
    item_variant_id: str
    family: str
    experiment: str
    cohort: str
    composition: str | None
    protocol_hash: str
    display_label: str | None
    model_hash: str
    budget_hash: str
    replicate_id: int
    state: str
    cls: str | None
    valid: bool
    u_reason: str | None
    confidence: float | None
    answer_logprob_mean: float | None
    retained_root_class: bool
    is_focal: bool
    received_slots_json: str


@dataclass(frozen=True)
class TransitionRow:
    """One agent's move from ``round`` to ``round + 1`` (plan §4.1 derived ``transitions``).

    ``round`` is the *from* round, which is what ``DERIVED_PRIMARY_KEYS['transitions']``
    declares; ``round_to`` is carried beside it so a reader never has to assume ``+1``.
    """

    __table__: ClassVar[str] = "transitions"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    cell_kind: str = pk()
    episode_id: str = pk()
    round: int = pk()
    agent_slot: int = pk()
    round_to: int
    source_item_id: str
    state_from: str
    state_to: str
    kind: str
    cls_from: str | None
    cls_to: str | None
    switched_class: bool
    validity_change: bool
    adopted_wstar: bool
    adopted_visible_class: bool
    copied_exact: bool
    ngram_overlap_max: float | None
    placebo_overlap_max: float | None
    confidence_from: float | None
    confidence_to: float | None


@dataclass(frozen=True)
class SelectionGradedRow:
    """A sealed selection joined to its grade (spec 15.1; plan §4.5 seal-then-join)."""

    __table__: ClassVar[str] = "selections_graded"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    selection_id: str = pk()
    cell_kind: str
    episode_id: str
    source_item_id: str
    round: int
    selector: str
    selector_version: str
    pool_id: str
    pool_scope: str
    selected_decision_id: str | None
    selected_class: str | None
    selected_state: str | None
    selected_correct: float | None
    oracle_lo: float
    oracle_hi: float
    selection_gap_lo: float | None
    fallback_status: str
    n_candidates: int
    valid_count: int


@dataclass(frozen=True)
class EfficiencyFrontierRow:
    """One (protocol, budget level, family) point of the spec-16.1 frontier (plan §4.3)."""

    __table__: ClassVar[str] = "efficiency_frontier"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    protocol_hash: str = pk()
    budget_level: float = pk()
    family: str = pk()
    display_label: str | None
    truncation_rule: str
    truncation_round: int
    reference_cost: float
    allowance: float
    measured_cost: float
    unused_allowance: float
    delivered_accuracy_lo: float | None
    delivered_accuracy_hi: float | None
    n_items: int


@dataclass(frozen=True)
class ToolUsagePanelRow:
    """Per-E2b-episode ``check_step`` descriptives (spec 9.6; a post-treatment descriptive)."""

    __table__: ClassVar[str] = "tool_usage_panel"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    cell_kind: str = pk()
    episode_id: str = pk()
    source_item_id: str
    protocol_hash: str
    model_hash: str
    n_checks: int
    n_over_budget: int
    n_duplicate: int
    step_distribution_json: str
    detected_invalid: int
    adhered_after_detection: bool | None
    confidence: float | None
    validity: bool


@dataclass(frozen=True)
class E2aE1LinkRow:
    """Per-item E2a endorsement effect beside the E1 switch rates (spec 9.6, cross-fitted)."""

    __table__: ClassVar[str] = "e2a_e1_link"
    __unique__: ClassVar[tuple[tuple[str, ...], ...]] = ()

    source_item_id: str = pk()
    model_hash: str = pk()
    family: str
    fold: int
    e2a_endorsement_effect: float | None
    e1_w_to_c_rate: float | None
    e1_c_to_w_rate: float | None
    n_e2a_episodes: int
    n_e1_transitions: int


DERIVED_ROW_TYPES: Final[tuple[type, ...]] = (
    TeamRoundRow, AgentRoundRow, TransitionRow, SelectionGradedRow, EfficiencyFrontierRow,
    ToolUsagePanelRow, E2aE1LinkRow,
)

#: derived table -> (producer module, consumer modules); same contract as ``_PRODUCERS`` (R3).
_DERIVED_PRODUCERS: Final[Mapping[str, tuple[str, tuple[str, ...]]]] = {
    "team_rounds": (_P + "outcomes", (_P + "analysis", _P + "panels", _P + "figures")),
    "agent_rounds": (_P + "outcomes", (_P + "analysis", _P + "calib")),
    "transitions": (_P + "outcomes", (_P + "analysis", _P + "figures")),
    "selections_graded": (_P + "analysis", (_P + "figures", _P + "release")),
    "efficiency_frontier": (_P + "analysis", (_P + "figures",)),
    "tool_usage_panel": (_P + "analysis", (_P + "figures",)),
    "e2a_e1_link": (_P + "analysis", (_P + "figures",)),
}


def derived_schema_of(row_type: type) -> TableSchema:
    """:class:`TableSchema` of a *derived* row dataclass (columns = ``dataclasses.fields``)."""
    if not dataclasses.is_dataclass(row_type):
        raise ManifestError(f"{row_type!r} is not a row dataclass")
    name = getattr(row_type, "__table__", None)
    if name not in _DERIVED_PRODUCERS:
        raise ManifestError(f"{row_type!r} declares no derived table; declared: {sorted(_DERIVED_PRODUCERS)}")
    hints = _type_hints(row_type)
    columns: list[Column] = []
    primary: list[str] = []
    for f in dataclasses.fields(row_type):
        type_name, nullable = _column_type(hints[f.name])
        if f.metadata.get("pk"):
            primary.append(f.name)
            if nullable:
                raise ManifestError(f"{name}.{f.name} is a primary-key column and cannot be nullable")
        columns.append(Column(f.name, type_name, nullable))
    if not primary:
        raise ManifestError(f"{name} declares no primary key")
    producer, consumers = _DERIVED_PRODUCERS[name]
    return TableSchema(name=name, row_type=row_type, columns=tuple(columns), primary_key=tuple(primary),
                       unique=tuple(tuple(u) for u in getattr(row_type, "__unique__", ())),
                       producer=producer, consumers=tuple(consumers), private=False)


DERIVED_TABLES: Final[Mapping[str, TableSchema]] = {
    t.__table__: derived_schema_of(t) for t in DERIVED_ROW_TYPES
}
for _name, _spec in DERIVED_TABLES.items():  # pragma: no cover - import-time contract
    _declared = DERIVED_PRIMARY_KEYS.get(_name)
    if _declared is not None and tuple(_declared) != _spec.primary_key:
        raise RuntimeError(f"derived table {_name}: row dataclass key {_spec.primary_key} != "
                           f"DERIVED_PRIMARY_KEYS {tuple(_declared)}")
del _name, _spec, _declared


def derived_schema(name: str) -> TableSchema:
    try:
        return DERIVED_TABLES[name]
    except KeyError as exc:
        raise MaterializeError(f"unknown derived table {name!r}; declared: {sorted(DERIVED_TABLES)}") from exc


def derived_row_to_dict(row: Any) -> dict[str, Any]:
    if type(row) not in DERIVED_ROW_TYPES:
        raise MaterializeError(f"{type(row).__name__} is not a declared derived row type")
    return {f.name: (list(v) if isinstance(v, tuple) else v)
            for f, v in ((f, getattr(row, f.name)) for f in dataclasses.fields(row))}


def derived_project(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Strict whitelist projection for a derived table (same rule as :func:`project`)."""
    spec = derived_schema(table)
    unknown = sorted(set(row) - set(spec.column_names))
    if unknown:
        raise MaterializeError(f"{table}: undeclared column(s) {unknown}; declare them on "
                               f"{spec.row_type.__name__} or drop them")
    return {c.name: _check_scalar(table, c, row.get(c.name)) for c in spec.columns}


def write_derived_table(rows: Sequence[Any], path: str | os.PathLike, *, table: str | None = None) -> None:
    """Write derived row dataclasses in the declared column order (``.empty`` when no rows)."""
    path = Path(path)
    name = table or (path.name[: -len(path.suffix)] if path.suffix else path.name)
    spec = derived_schema(name)
    dicts = []
    for row in rows:
        if type(row) is not spec.row_type:
            raise MaterializeError(f"{name}: row of type {type(row).__name__}, expected {spec.row_type.__name__}")
        dicts.append(derived_project(name, derived_row_to_dict(row)))
    write_parquet_rows(path, [{k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
                              for d in dicts], spec.column_names)


def check_derived_keys(derived_dir: str | os.PathLike, *, strict: bool = True) -> list[KeyViolation]:
    """:func:`check_keys` over the derived registry instead of the published one."""
    checks = run_checks(derived_dir, tables=DERIVED_TABLES)
    violations = [
        KeyViolation(c.table, c.key, c.constraint, c.n_rows, c.n_distinct, c.n_duplicate_groups, c.examples, c.note)
        for c in checks
        if c.status == STATUS_DUPLICATE or (strict and c.status == STATUS_UNVERIFIABLE)
    ]
    violations.sort(key=lambda v: (-v.n_duplicate_groups, v.table, v.constraint))
    return violations


__all__ += [
    "AgentRoundRow", "DERIVED_ROW_TYPES", "DERIVED_TABLES", "E2aE1LinkRow",
    "EfficiencyFrontierRow", "SelectionGradedRow", "TeamRoundRow", "ToolUsagePanelRow",
    "TransitionRow", "check_derived_keys", "derived_project", "derived_row_to_dict",
    "derived_schema", "derived_schema_of", "write_derived_table",
]
