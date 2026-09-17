"""Frozen shared contracts for openrouter_agent_allocation_v1.

Seeded from the handoff pack's ``shared/contracts.py`` (sha256 f131577b…) and completed by the lead.
Only the lead edits this file; agents propose changes. ``tests/unit/test_contracts_frozen.py`` pins its hash.

Rules that every module relies on (see ``docs/00_contract.md`` for the full table):

* A generation node is one fresh API sampling opportunity, identified by ``GenerationSpec.node_id``.
  ``candidate_id == node_id``: one accepted result per semantic node.
* ``node_id`` hashes every ``GenerationSpec`` field. The three hash fields must therefore be *stable*:
  ``endpoint_lock_hash`` is ``EndpointLock.lock_hash`` (semantic fields only, never fetch timestamps or
  uptime); ``input_manifest_hash`` is the hash of *this item's* ``PublicTask`` record
  (``public_task_hash``), not of a whole manifest file; ``decoding_config_hash`` comes from
  ``config.decoding_config_hash``. Lock files are write-once; resume reads them and never re-fetches.
* Records cross module and process boundaries only through ``dump_record`` / ``load_record``.
  Never hand-roll a dict for a contract type: writer/reader key drift is this project's recurring defect.
* Selectors never receive ``PrivateEvalSpec`` or ``GradeResult``.

Contract v2 (2026-09-17 03:40 EDT) resolves docs/requests D1-D4, D10, D11, R01-1, R01-2, R04-1; see docs/00_contract.md §6.
"""
from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping, Protocol, Sequence

STUDY_ID = "openrouter_agent_allocation_v1"
BENCHMARKS: tuple[str, ...] = ("math500", "mmlu_pro", "bbh_logic4", "mbpp_plus")
BBH_SUBTASKS: tuple[str, ...] = (
    "logical_deduction_seven_objects",
    "tracking_shuffled_objects_seven_objects",
    "boolean_expressions",
    "temporal_sequences",
)
WAVE_ITEMS_PER_BENCHMARK = 4


# ---------------------------------------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """The one canonical JSON encoding used for every content hash in the study."""
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(obj: Any) -> str:
    return sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------------------
# Enumerations (seed values are unchanged)
# ---------------------------------------------------------------------------------------------------------

class OutputKind(str, Enum):
    MATH = "math"
    OPTION = "option"
    BOOLEAN = "boolean"
    PYTHON = "python"


class GenerationKind(str, Enum):
    ROOT = "root"
    PRIVATE = "private"
    PEER = "peer"
    DEEP = "deep"
    CP = "communicate_then_private"
    POOLED = "pooled"


class ParseStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing_final"
    MALFORMED = "malformed_final"
    EMPTY = "empty_content"
    REFUSAL = "refusal"


class GradeStatus(str, Enum):
    GRADED = "graded"
    INVALID_OUTPUT = "invalid_output"
    EVALUATOR_ERROR = "evaluator_error"
    NOT_EVALUATED = "not_evaluated"


class Partition(str, Enum):
    """Values for ``GenerationSpec.partition_id``. CP/POOLED nodes live in MAIN (kind distinguishes them)."""
    CANARY = "canary"
    PILOT = "pilot"
    MAIN = "main"
    HIGH = "high"
    REPLICATION = "replication"


class AttemptState(str, Enum):
    PENDING = "pending"
    LEASED = "leased"
    SENT = "sent"
    RESPONSE_PERSISTED = "response_persisted"
    TERMINAL = "terminal"


class TransportStatus(str, Enum):
    OK = "ok"                                # HTTP 200 carrying a completion (content may still be empty)
    ERROR_ENVELOPE = "error_envelope"        # HTTP 2xx without one usable completion: retryable, charge unknown
    RATE_LIMITED = "rate_limited"            # 429: retryable, honor Retry-After, halve allowances
    AUTH_FAILED = "auth_failed"              # 401/403: open the circuit
    PAYMENT_REQUIRED = "payment_required"    # 402: open the circuit
    BAD_REQUEST = "bad_request"              # 400/404/422 (e.g. unsupported parameter): blocker, no retry
    UPSTREAM_ERROR = "upstream_error"        # 5xx / provider failure: retryable
    CONNECT_FAILED = "connect_failed"        # provably never sent: retryable, reservation may be released
    AMBIGUOUS_TIMEOUT = "ambiguous_timeout"  # may have been billed: reservation retained until reconciled
    CACHE_HIT = "cache_hit"                  # response-cache HIT on an independent draw: open the circuit
    PROVIDER_MISMATCH = "provider_mismatch"  # served outside the endpoint lock: open the circuit


RETRYABLE_TRANSPORT = frozenset({
    TransportStatus.ERROR_ENVELOPE,          # v2: provider error inside HTTP 200; reservation kept until reconciled
    TransportStatus.RATE_LIMITED,
    TransportStatus.UPSTREAM_ERROR,
    TransportStatus.CONNECT_FAILED,
    TransportStatus.AMBIGUOUS_TIMEOUT,
})
CIRCUIT_BREAKING_TRANSPORT = frozenset({
    TransportStatus.AUTH_FAILED,
    TransportStatus.PAYMENT_REQUIRED,
    TransportStatus.CACHE_HIT,
    TransportStatus.PROVIDER_MISMATCH,
    TransportStatus.BAD_REQUEST,
})


class CompletionStatus(str, Enum):
    """Model outcome of a transport-OK response. Every value is terminal: never redrawn."""
    COMPLETED = "completed"  # finish_reason stop, non-empty visible content
    LENGTH = "length"        # finish_reason length (visible content may be empty or partial)
    EMPTY = "empty"          # non-length finish with empty visible content
    BLOCKED = "blocked"      # safety / recitation / content filter
    OTHER = "other"


class EvalKind(str, Enum):
    MATH_VERIFY = "math_verify"
    OPTION_EXACT = "option_exact"
    TASK_EXACT = "task_exact"
    EVALPLUS_FULL = "evalplus_full"


class CheckStatus(str, Enum):
    OK = "ok"
    INVALID_CANDIDATE = "invalid_candidate"          # no parseable program: all cases count as failed
    INFRASTRUCTURE_ERROR = "infrastructure_error"    # checker broke: the selector must pause, not guess


class SelectionStatus(str, Enum):
    SELECTED = "selected"
    ABSTAINED_ALL_INVALID = "abstained_all_invalid"
    PAUSED_CHECK_ERROR = "paused_public_check_error"
    SINGLE_CANDIDATE = "single_candidate"            # last-answer readouts: PRIVATE_SINGLE, DEEP, POOLED, n=1


class LedgerEventKind(str, Enum):
    RESERVE = "reserve"
    SENT = "sent"
    SETTLE = "settle"
    RELEASE_UNSENT = "release_unsent"   # only for attempts that provably never left the process
    CHARGE_MAX = "charge_max"           # conservative settlement of an unresolved attempt
    UNRESOLVED = "unresolved"           # v2 (amount 0): attempt may have been billed; counts toward the $2 pause.
                                        # On replay, every SENT reservation without SETTLE/CHARGE_MAX/RELEASE_UNSENT
                                        # is unresolved, whether or not this event was written before a crash.
    RECONCILE = "reconcile"             # provider-side reconciliation note (no reservation change)


class CheckpointView(str, Enum):
    CURRENT = "current"
    ARCHIVE = "archive"


# ---------------------------------------------------------------------------------------------------------
# Tasks and private evaluation
# ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PublicTask:
    item_id: str
    benchmark: str
    source_revision: str
    prompt: str
    output_kind: OutputKind
    public_examples: tuple[str, ...] = ()
    allowed_choices: tuple[str, ...] = ()
    subtask: str | None = None


def public_task_hash(task: PublicTask) -> str:
    """Per-item ``GenerationSpec.input_manifest_hash``: the request depends on this record only."""
    return canonical_hash(task)


@dataclass(frozen=True)
class PrivateEvalSpec:
    item_id: str
    benchmark: str
    source_revision: str
    eval_kind: EvalKind
    gold_answer: str | None            # MATH answer, option letter, "True"/"False"; None for code
    private_record_ref: str | None     # run-dir-relative path of the restricted record / oracle (code)
    private_hash: str                  # canonical hash of the full private record (gold + tests)
    subtask: str | None = None
    allowed_choices: tuple[str, ...] = ()   # v2: legal final identities for option/boolean items; () for MATH/code


@dataclass(frozen=True)
class InventoryItem:
    public: PublicTask
    private: PrivateEvalSpec
    strata: tuple[tuple[str, str], ...] = ()      # e.g. (("level", "3"), ("type", "Algebra"))
    eligible: bool = True
    exclusion_reason: str | None = None           # pre-generation engineering reason only


@dataclass(frozen=True)
class PanelDraw:
    """Per-benchmark seeded selection. Waves take ``main_item_ids`` in order, four at a time.

    For ``bbh_logic4`` every block ``main_item_ids[4w:4w+4]`` holds exactly one item per subtask, in
    ``BBH_SUBTASKS`` order, and ``dev_item_ids`` holds two per subtask. Nested panels are wave prefixes,
    so the order must be a seeded permutation of the stratified sample (never outcome-dependent).
    """
    benchmark: str
    dev_item_ids: tuple[str, ...]
    main_item_ids: tuple[str, ...]
    reserve_item_ids: tuple[str, ...]
    selection_seed: int
    procedure: str


@dataclass(frozen=True)
class Wave:
    index: int                       # 0-based admission order
    item_ids: tuple[str, ...]        # 16 items: four per benchmark, benchmarks in BENCHMARKS order
    panels: tuple[str, ...]          # nested panels this wave belongs to, e.g. ("main", "extension", "high")


# ---------------------------------------------------------------------------------------------------------
# Model/endpoint contract and generation identity
# ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EndpointLock:
    model: str
    provider_tag: str                          # exact endpoint tag from metadata, e.g. "google-ai-studio"
    provider_name: str                         # e.g. "Google AI Studio"
    service_tier: str                          # "standard"
    input_price_per_token: str                 # decimal strings from the endpoints snapshot
    output_price_per_token: str
    internal_reasoning_price_per_token: str
    context_length: int
    max_completion_tokens: int
    supported_parameters: tuple[str, ...]
    snapshot_path: str = ""                    # raw endpoints response saved beside the lock
    snapshot_sha256: str = ""
    fetched_at: str = ""

    @property
    def lock_hash(self) -> str:
        volatile = {"snapshot_path", "snapshot_sha256", "fetched_at"}
        semantic = {k: v for k, v in asdict(self).items() if k not in volatile}
        semantic["supported_parameters"] = sorted(semantic["supported_parameters"])
        return canonical_hash(semantic)


@dataclass(frozen=True)
class GenerationSpec:
    run_id: str
    partition_id: str
    item_id: str
    benchmark: str
    kind: GenerationKind
    model: str
    endpoint_lock_hash: str
    input_manifest_hash: str
    decoding_config_hash: str
    prompt_version: str
    reasoning_budget: int
    completion_headroom: int
    root_slot: int
    team_size: int
    update_round: int
    draw_id: int
    repetition_id: int = 0
    parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.reasoning_budget <= 0 or self.completion_headroom <= 0:
            raise ValueError("Reasoning budget and completion headroom must be positive.")
        if min(self.root_slot, self.team_size, self.update_round, self.draw_id, self.repetition_id) < 0:
            raise ValueError("Indices must be nonnegative.")
        if len(self.parent_ids) != len(set(self.parent_ids)):
            raise ValueError("Duplicate parent reference.")

    @property
    def node_id(self) -> str:
        # Seed encoding kept byte-for-byte: json of asdict with sorted keys (enum values are str subclasses).
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    @property
    def max_completion_tokens(self) -> int:
        return self.reasoning_budget + self.completion_headroom


@dataclass(frozen=True)
class GridSpec:
    """Parameterized protocol grid built per item by agent 03.

    Coordinates (golden fixtures in ``tests/unit/test_node_identity_golden.py``):
      ROOT    root_slot=j  team_size=0 update_round=0 parents=()
      PRIVATE root_slot=j  team_size=0 update_round=r parents=(previous candidate of slot j,)
      PEER    root_slot=receiver j, team_size=n, update_round=r, parents=all n round r-1 nodes in slot order
      DEEP    root_slot=0  team_size=0 update_round=0 reasoning_budget=A*m parents=()
      CP      root_slot=j  team_size=4 update_round=h parents=(peer n=4 r=1 slot j, or previous CP of slot j,)
      POOLED  root_slot=0  team_size=4 update_round=0 parents=(root 0..3 in slot order)
    ``draw_id`` is 0 except for deliberate repeats of otherwise identical coordinates (canary only).
    """
    name: str
    bank_size: int                      # independent roots root[0..bank_size-1]
    private_single_cycles: int          # total cycles including root[0]
    private_team_rounds: int            # private updates per slot for slots 1..max(team_sizes)-1
    team_sizes: tuple[int, ...]
    peer_rounds: int
    deep_multipliers: tuple[int, ...]
    extension: bool = False             # CP (from PEER n=4 r=1, three private steps) + POOLED (roots 0..3)


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None           # missing is None, never 0
    cached_prompt_tokens: int | None
    actual_cost_usd: Decimal | None
    raw_usage: Mapping[str, Any] = field(default_factory=dict)
    upstream_inference_cost_usd: Decimal | None = None


@dataclass(frozen=True)
class Attempt:
    attempt_id: str                        # f"{node_id}:{attempt_no}"
    node_id: str
    attempt_no: int                        # 1..max_transport_attempts
    state: AttemptState
    reservation_usd: Decimal
    created_at: str                        # ISO-8601 UTC
    sent_at: str | None = None
    completed_at: str | None = None
    transport_status: TransportStatus | None = None
    http_status: int | None = None
    generation_id: str | None = None
    settled_usd: Decimal | None = None
    retry_after_seconds: float | None = None
    error_detail: str | None = None        # redacted


@dataclass(frozen=True)
class GenerationResult:
    node_id: str
    attempt_id: str
    transport_status: TransportStatus
    completion_status: CompletionStatus | None
    generation_id: str | None
    observed_model: str | None
    observed_provider: str | None
    finish_reason: str | None
    native_finish_reason: str | None
    visible_content: str | None            # message.content exactly as returned
    reasoning_returned: bool               # reasoning text/details present (kept in raw only, never relayed)
    usage: Usage | None
    cache_status: str | None               # X-OpenRouter-Cache-Status value
    raw_response_path: str | None          # run-dir-relative
    raw_response_sha256: str | None
    latency_seconds: float | None
    http_status: int | None
    retry_after_seconds: float | None = None
    error_detail: str | None = None        # redacted


@dataclass(frozen=True)
class GenerationOutcome:
    """01 -> 02 handoff: the typed result plus exactly what 02 must persist before committing a terminal record.

    ``result.raw_response_path`` / ``raw_response_sha256`` are None here; 02 fills them after the atomic write.
    """
    result: GenerationResult
    request_body: Mapping[str, Any]        # exactly what was sent; never contains credentials
    response_body: bytes
    response_headers: Mapping[str, str]    # lowercase keys, redacted


@dataclass(frozen=True)
class LedgerEvent:
    seq: int
    kind: LedgerEventKind
    amount_usd: Decimal
    recorded_at: str
    attempt_id: str | None = None
    node_id: str | None = None
    partition_id: str | None = None
    detail: str | None = None


# ---------------------------------------------------------------------------------------------------------
# Candidates, checkpoints, selection, grading
# ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    candidate_id: str                      # == node_id
    node_id: str
    item_id: str
    visible_content: str
    final_answer: str | None
    rationale: str | None
    parse_status: ParseStatus
    finish_reason: str | None
    output_kind: OutputKind


@dataclass(frozen=True)
class PolicyCheckpoint:
    policy_id: str                         # f"{partition_id}/{item_id}/{name}/{view}" with name as below
    item_id: str
    family: str                            # sample | private_single | private_team | peer | deep | cp | pooled
    team_size: int
    total_cycles: int
    peer_rounds: int
    candidate_ids: tuple[str, ...]         # the declared pool (current view) or archive (archive view)
    required_node_ids: tuple[str, ...]     # every ancestor needed to deploy from scratch, deduplicated
    selector_name: str
    selector_version: str
    partition_id: str = ""
    name: str = ""                         # reference-DAG name: sample/k, private_single/t, private_team/n/r,
                                           # peer/n/r, deep/m, cp/h, pooled/4
    view: CheckpointView = CheckpointView.CURRENT
    reasoning_budget: int = 0


@dataclass(frozen=True)
class PublicCheckResult:
    candidate_id: str
    item_id: str
    code_sha256: str | None
    status: CheckStatus
    cases_total: int
    cases_passed: int
    case_outcomes: tuple[str, ...]         # "pass" | "fail" | "timeout" | "error", official order
    checker_version: str
    public_inputs_sha256: str
    environment_digest: str
    cpu_seconds: float

    @property
    def pass_fraction(self) -> float | None:
        """None when no public case exists: the checker must report INFRASTRUCTURE_ERROR and the selector pauses."""
        return self.cases_passed / self.cases_total if self.cases_total else None


@dataclass(frozen=True)
class SelectionResult:
    policy_id: str
    selected_candidate_id: str | None
    selector_name: str
    selector_version: str
    pool_hash: str
    public_evidence_hash: str | None
    cpu_seconds: float
    status: SelectionStatus                # v2: typed (was str)


@dataclass(frozen=True)
class GradeResult:
    candidate_id: str
    status: GradeStatus
    semantic_correct: bool | None          # tri-state: None = unknown
    operational_success: bool | None
    grader_version: str
    private_eval_hash: str
    detail_code: str | None = None
    environment_digest: str | None = None
    cpu_seconds: float | None = None

    def __post_init__(self) -> None:
        # v2: the tri-state rule is enforced by the type (ARD36's grading defect cannot be constructed).
        if self.status == GradeStatus.GRADED and self.semantic_correct is None:
            raise ValueError("GRADED requires semantic_correct True/False")
        if self.status in (GradeStatus.EVALUATOR_ERROR, GradeStatus.NOT_EVALUATED) and self.semantic_correct is not None:
            raise ValueError(f"{self.status.value} requires semantic_correct=None (unknown)")
        if self.status == GradeStatus.INVALID_OUTPUT and (self.semantic_correct is not None
                                                          or self.operational_success is not False):
            raise ValueError("INVALID_OUTPUT requires semantic_correct=None and operational_success=False")


@dataclass(frozen=True)
class GateCriterion:
    name: str
    observed: str
    threshold: str
    passed: bool
    evidence: str                          # run-dir-relative artifact the value was computed from


@dataclass(frozen=True)
class GateResult:
    gate: str                              # gate0 .. gate5
    stage: str
    passed: bool
    decision: str                          # advance | stop_notify | amend_headroom_once
    criteria: tuple[GateCriterion, ...]
    evaluated_at: str


@dataclass(frozen=True)
class ItemGraph:
    item_id: str
    partition_id: str
    grid: GridSpec
    nodes: tuple[GenerationSpec, ...]      # topologically ordered
    checkpoints: tuple[PolicyCheckpoint, ...]


class Generator(Protocol):
    async def generate(self, spec: GenerationSpec, messages: Sequence[Mapping[str, str]], *, attempt_id: str,
                       lock: EndpointLock, sampling: Mapping[str, Any]) -> "GenerationOutcome": ...


class Selector(Protocol):
    def select(self, task: PublicTask, candidates: Sequence[Candidate],
               policy_id: str) -> SelectionResult: ...


# ---------------------------------------------------------------------------------------------------------
# The one record codec
# ---------------------------------------------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def dump_record(obj: Any) -> dict[str, Any]:
    if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)):
        raise TypeError(f"dump_record expects a contract dataclass, got {type(obj).__name__}")
    out = to_jsonable(obj)
    out["__record__"] = type(obj).__name__
    return out


_HINT_CACHE: dict[type, dict[str, Any]] = {}
import contextvars as _contextvars
_NONSTRICT: "_contextvars.ContextVar[bool]" = _contextvars.ContextVar("allocation_nonstrict_load", default=False)


def _hints(cls: type) -> dict[str, Any]:
    if cls not in _HINT_CACHE:
        _HINT_CACHE[cls] = typing.get_type_hints(cls)
    return _HINT_CACHE[cls]


def _coerce(tp: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _coerce(non_none[0], value)
        return value
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v) for v in value)
        return tuple(_coerce(a, v) for a, v in zip(args, value))
    if origin in (list, Sequence):
        return [_coerce(args[0], v) for v in value] if args else list(value)
    if origin in (dict, Mapping) or tp is Mapping:
        return dict(value)
    if isinstance(tp, type):
        if issubclass(tp, Enum):
            return tp(value)
        if tp is Decimal:
            return Decimal(str(value))
        if dataclasses.is_dataclass(tp):
            return load_record(tp, value, strict=False) if _NONSTRICT.get() else load_record(tp, value)
        if tp is float and isinstance(value, int):
            return float(value)
    return value


def load_record(cls: type, data: Mapping[str, Any], *, strict: bool = True) -> Any:
    """v2: strict by default -- every field must be present (a missing field never silently takes its default,
    which would change e.g. a checkpoint's view or a node's identity). ``strict=False`` is for hand-written fixtures."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a contract dataclass")
    tag = data.get("__record__")
    if tag is not None and tag != cls.__name__:
        raise ValueError(f"record tagged {tag!r} cannot load as {cls.__name__}")
    hints = _hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names - {"__record__"}
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown fields {sorted(unknown)}")
    if strict:
        missing = names - set(data)
        if missing:
            raise ValueError(f"{cls.__name__}: missing fields {sorted(missing)} (strict load)")
    token = _NONSTRICT.set(not strict)
    try:
        kwargs = {name: _coerce(hints[name], data[name]) for name in names if name in data}
    finally:
        _NONSTRICT.reset(token)
    return cls(**kwargs)
