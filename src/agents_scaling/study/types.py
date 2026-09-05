"""Shared contract for the agent_design_v4 study package (WP0).

Every other work package codes against these enums, frozen dataclasses, constants and
exception classes.  Nothing here touches a GPU, the network or protected data.

Spec citations: §3.3 (task-public view), §3.4 (counters), §3.5 (candidate schema, the
failure sentinel), §3.6 (seed key order, request identity), §4.2 (policies), §4.3 (packet
fields, envelopes), §6.4 (caps), §10.4 (failure classes).  Architecture:
docs/study_v4/01_architecture.md §1.1, §2.1–2.2.  Corrections: 04_critic_corrections.md
P0-4 (``parallel_items``), P0-7 (``lane``), §2 table (seed purposes and namespaces are
frozen here), §4 item 7 (``ContextFailure`` is a used opportunity).

Serialization convention: ``to_dict()`` yields JSON-safe values only (enums → value,
tuples → lists, nested dataclasses → dicts); ``from_dict()`` restores enums, nested
dataclasses and list-typed fields (as tuples).  Values *inside* free-form ``dict`` fields
keep JSON-native types (lists stay lists) so that a round trip is an identity.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Union

from agents_scaling.study import identity

# --------------------------------------------------------------------------- enums


class Domain(str, Enum):
    """Superdomain of a source task (§3.1)."""

    HLE = "hle"
    BCB = "bcb"


class Framing(str, Enum):
    """§5.5 factorial framing cell ``<TEAM_FRAME><VOTE_AWARE>``; ``NATIVE`` = a policy's own
    truthful role wording (main-tier DEC root, P0-1)."""

    F00 = "00"
    F01 = "01"
    F10 = "10"
    F11 = "11"
    NATIVE = "nat"

    @property
    def team_frame(self) -> int:
        if self is Framing.NATIVE:
            raise ValueError("NATIVE framing has no TEAM_FRAME level")
        return int(self.value[0])

    @property
    def vote_aware(self) -> int:
        if self is Framing.NATIVE:
            raise ValueError("NATIVE framing has no VOTE_AWARE level")
        return int(self.value[1])


class Method(str, Enum):
    """Generation policies (§4.2) plus the controls of §4.6 and the F banks (§5.5)."""

    BANK = "BANK"
    S_FRESH = "S_FRESH"
    S_HISTORY = "S_HISTORY"
    IND_VOTE = "IND_VOTE"
    DEC = "DEC"
    CEN_FLAT = "CEN_FLAT"
    IND_PRIVATE_REVISION = "IND_PRIVATE_REVISION"
    DEC_ONE_ROUND = "DEC_ONE_ROUND"
    DEGREE = "DEGREE"


class CellKind(str, Enum):
    """What a cell does; only ``GENERATE`` needs the frozen study manifest with B0."""

    GENERATE = "GENERATE"
    JUDGE_BEST = "JUDGE_BEST"
    JUDGE_HLE = "JUDGE_HLE"
    EVAL_BCB = "EVAL_BCB"
    FORECAST = "FORECAST"


# --------------------------------------------------------------------------- frozen constants

#: Semantic-seed ``purpose`` strings (§3.6; critic §2 table: arbitrary but frozen once here).
PURPOSE_ROOT = "root"
PURPOSE_REVISE = "revise"
PURPOSE_HUB = "hub"
PURPOSE_WORKER = "worker"
PURPOSE_CONSUMER = "consumer"
PURPOSE_HLE_JUDGE = "hle_judge"
PURPOSE_JUDGE_BEST = "judge_best"
PURPOSE_FORECAST = "forecast"
PURPOSES: tuple[str, ...] = (
    PURPOSE_ROOT,
    PURPOSE_REVISE,
    PURPOSE_HUB,
    PURPOSE_WORKER,
    PURPOSE_CONSUMER,
    PURPOSE_HLE_JUDGE,
    PURPOSE_JUDGE_BEST,
    PURPOSE_FORECAST,
)

#: Semantic-seed ``namespace`` strings.  Stateless roots of every policy share
#: ``NS_STATELESS_BANK`` (actor_slot=0, step_slot=draw ordinal) so that the §6.2 aliases
#: coincide automatically (architecture §2.3, P0-1).
NS_STATELESS_BANK = "stateless_bank"
NS_S_HISTORY = "S_HISTORY"
NS_DEC = "DEC"
NS_CEN_FLAT = "CEN_FLAT"
NS_DEGREE = "DEGREE"
NS_JUDGE = "judge"
NS_FORECAST = "forecast"
NAMESPACES: tuple[str, ...] = (
    NS_STATELESS_BANK,
    NS_S_HISTORY,
    NS_DEC,
    NS_CEN_FLAT,
    NS_DEGREE,
    NS_JUDGE,
    NS_FORECAST,
)

#: §3.6 JCS array order of a seed key.
SEED_KEY_ORDER: tuple[str, ...] = (
    "source_id",
    "split",
    "model_cell",
    "episode_rep",
    "actor_slot",
    "purpose",
    "step_slot",
    "namespace",
)

#: Exit codes of ``run_one`` (architecture §3 steps 3, 7, 8).
EXIT_DONE = 0
EXIT_NO_SERVER = 2
EXIT_INCOMPLETE = 3
EXIT_SUSPENDED = 4

#: Static prompt envelope (§4.3, §6.4).  The yaml ``caps`` block must agree (config.py).
TASK_TOKENS_CAP = 4096
PROMPT_TOKENS_CAP = 32768
SOLVER_OUT_CAP = 8192
SELECTOR_OUT_CAP = 1024
FORECAST_OUT_CAP = 256

#: Exact §3.5 failure sentinel handed to a revision whose parent is invalid.  It is always
#: rendered from ``SENTINEL_JSON`` (insertion order), never through ``identity.jcs``.
SENTINEL: dict[str, str] = {"status": "unavailable", "failure_code": "NO_VALID_PARENT"}
SENTINEL_JSON = '{"status":"unavailable","failure_code":"NO_VALID_PARENT"}'
assert json.dumps(SENTINEL, separators=(",", ":")) == SENTINEL_JSON

#: §4.3 packet compiler field priority (final-answer display first).
PACKET_FIELD_PRIORITY: tuple[str, ...] = (
    "final_answer",
    "evidence",
    "failure_checks",
    "alternatives_considered",
    "approach",
    "confidence",
)

#: §3.5 candidate safety limits (parser bounds; they never raise the 8,192 token cap).
CANDIDATE_LIMITS: Mapping[str, int] = {
    "approach": 16384,
    "list_items": 32,
    "list_string": 8192,
    "final_answer": 131072,
}
UNCERTAINTY_LEVELS: tuple[str, ...] = ("low", "medium", "high", "unknown")
SUBTASK_STATUSES: tuple[str, ...] = ("complete", "partial", "failed")


# --------------------------------------------------------------------------- exceptions


class InfraFailure(RuntimeError):
    """Exogenous fault after the paired technical retries (§10.4) → item INCOMPLETE."""


class ProtocolError(RuntimeError):
    """Harness defect (overshoot, corrupt store, stale freeze) → cell SUSPENDED (exit 4)."""


class ContextFailure(RuntimeError):
    """A rendered prompt cannot fit the frozen envelope (§4.3).  A *used* opportunity:
    the episode continues under the normal failure rule (Y=0, DEC slot gets the sentinel)."""


# --------------------------------------------------------------------------- serialization


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses/enums/tuples into JSON-native values."""
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{type(value).__name__} is not JSON-serializable")


def _tuplize(value: Any) -> Any:
    """Lists → tuples recursively (dict contents are left JSON-native)."""
    if isinstance(value, list):
        return tuple(_tuplize(item) for item in value)
    return value


class _Record:
    """Mixin giving dataclasses ``to_dict``/``from_dict`` driven by class-level tables."""

    _enum_fields: ClassVar[Mapping[str, type[Enum]]] = {}
    _nested: ClassVar[Mapping[str, type]] = {}
    _nested_seq: ClassVar[Mapping[str, type]] = {}

    def to_dict(self) -> dict[str, Any]:
        return {f.name: to_jsonable(getattr(self, f.name)) for f in dataclasses.fields(self)}

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)

    @classmethod
    def _decode_field(cls, name: str, value: Any) -> Any:
        if name in cls._enum_fields:
            return cls._enum_fields[name](value)
        if name in cls._nested:
            return None if value is None else cls._nested[name].from_dict(value)
        if name in cls._nested_seq:
            if value is None:
                return None
            return tuple(cls._nested_seq[name].from_dict(item) for item in value)
        return _tuplize(value)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name in data:
                kwargs[f.name] = cls._decode_field(f.name, data[f.name])
            elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise KeyError(f"{cls.__name__}.from_dict: missing field {f.name!r}")
        return cls(**kwargs)


# --------------------------------------------------------------------------- task / model


@dataclass(frozen=True)
class PublicTask(_Record):
    """Task-public view only (§3.3, §10.6; amendment P2).  Never carries answers or tests."""

    source_id: str  # "hle:<id>" | "bcb:<task_id>"
    domain: Domain
    split: str  # "dev" | "main" | "reserve"
    task_text: str  # HLE question (revised top-level) | BCB instruct_prompt
    answer_format: str  # "multipleChoice" | "exactMatch" | "code"
    stratum: str  # "Gold" | "Revision" | "bcb"
    rank: int  # salted-hash rank inside its split (item order)
    task_tokens: int  # under the flagship tokenizer; must be <= TASK_TOKENS_CAP (§4.3)
    entry_point: str | None = None  # BCB only
    category: str | None = None

    _enum_fields = {"domain": Domain}


@dataclass(frozen=True)
class Checkpoint(_Record):
    """One pinned dense Qwen3 checkpoint and its serving layout (§6.4, amendment P4)."""

    size: str
    hf_id: str
    model_revision: str
    tokenizer_revision: str
    profile: str
    tp_size: int
    served_model_name: str

    @property
    def model_cell(self) -> str:
        """Seed-key ``model_cell`` component, e.g. ``Qwen3-32B@9216db57`` (architecture §2.1)."""
        return f"Qwen3-{self.size}@{self.model_revision[:8]}"

    @property
    def engine_digest(self) -> str:
        return identity.engine_digest(self)


@dataclass(frozen=True)
class Decoding(_Record):
    """Sampling contract of one request (§10.1 recipe, §6.4 caps, amendment F2/E2)."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    max_tokens: int
    enable_thinking: bool
    guided_json: bool = False  # amendment E2: judges/forecast only, decided at pilot (P2-1)

    def as_strings(self) -> dict[str, Any]:
        """Identity form: floats as fixed strings, ints/bools native (architecture §1.2)."""
        return {
            "temperature": identity.float_str(self.temperature),
            "top_p": identity.float_str(self.top_p),
            "top_k": int(self.top_k),
            "min_p": identity.float_str(self.min_p),
            "presence_penalty": identity.float_str(self.presence_penalty),
            "repetition_penalty": identity.float_str(self.repetition_penalty),
            "max_tokens": int(self.max_tokens),
            "enable_thinking": bool(self.enable_thinking),
            "guided_json": bool(self.guided_json),
        }

    @property
    def chat_template_kwargs(self) -> dict[str, bool]:
        return {"enable_thinking": bool(self.enable_thinking)}

    @property
    def hash(self) -> str:
        return identity.decoding_hash(self)


#: §10.1 dense Qwen3 thinking recipe with the §6.4 output cap — every solver role.
SOLVER_DECODING = Decoding(0.6, 0.95, 20, 0.0, 0.0, 1.0, SOLVER_OUT_CAP, True)
#: §4.4/§6.4 selector cap, thinking off, temperature 0 (JUDGE_BEST and the HLE judge).
JUDGE_DECODING = Decoding(0.0, 1.0, 1, 0.0, 0.0, 1.0, SELECTOR_OUT_CAP, False)
#: §8.7 shadow forecast (amendment C1): 256 tokens, thinking off, temperature 0.
FORECAST_DECODING = Decoding(0.0, 1.0, 1, 0.0, 0.0, 1.0, FORECAST_OUT_CAP, False)


# --------------------------------------------------------------------------- identity objects


@dataclass(frozen=True)
class SeedKey(_Record):
    """§3.6 semantic-seed key; field order *is* the JCS array order."""

    source_id: str
    split: str
    model_cell: str
    episode_rep: int
    actor_slot: int
    purpose: str
    step_slot: int
    namespace: str

    def as_array(self) -> list[Any]:
        return [getattr(self, name) for name in SEED_KEY_ORDER]

    def semantic_seed(self, study_seed: bytes) -> bytes:
        return identity.semantic_seed(study_seed, self)


@dataclass(frozen=True)
class RequestSpec(_Record):
    """Everything that determines one request's identity (§3.6) plus ledger metadata.

    ``role`` is for cost/ledger accounting only and is *not* part of the identity;
    ``study_id``/``study_seed_hex`` are carried so the identity properties are pure.
    """

    messages: tuple[Mapping[str, Any], ...]
    decoding: Decoding
    checkpoint: Checkpoint
    seed_key: SeedKey
    role: str
    study_id: str
    study_seed_hex: str

    _nested = {"decoding": Decoding, "checkpoint": Checkpoint, "seed_key": SeedKey}

    @property
    def chat_template_kwargs(self) -> dict[str, bool]:
        return self.decoding.chat_template_kwargs

    @property
    def local_caps(self) -> dict[str, int]:
        """Frozen local caps entering the identity (audit §2.0 item 4)."""
        return {"max_input": PROMPT_TOKENS_CAP, "max_tokens": int(self.decoding.max_tokens)}

    @property
    def input_hash(self) -> str:
        return identity.input_hash(self.messages, self.chat_template_kwargs)

    @property
    def decoding_hash(self) -> str:
        return identity.decoding_hash(self.decoding)

    @property
    def semantic_seed(self) -> bytes:
        return identity.semantic_seed(bytes.fromhex(self.study_seed_hex), self.seed_key)

    @property
    def semantic_seed_hex(self) -> str:
        return self.semantic_seed.hex()

    @property
    def engine_seed(self) -> int:
        return identity.engine_seed(self.semantic_seed)

    @property
    def engine_digest(self) -> str:
        return identity.engine_digest(self.checkpoint)

    def identity_fields(self) -> dict[str, Any]:
        """The nine §3.6 components, in request_id order (stored as ``RequestRecord.identity``)."""
        return {
            "study_id": self.study_id,
            "model_revision": self.checkpoint.model_revision,
            "tokenizer_revision": self.checkpoint.tokenizer_revision,
            "engine_digest": self.engine_digest,
            "input_hash": self.input_hash,
            "decoding_hash": self.decoding_hash,
            "semantic_seed_hex": self.semantic_seed_hex,
            "local_caps": self.local_caps,
            "hook_hash": identity.HOOK_HASH_NONE,
        }

    @property
    def request_id(self) -> str:
        return identity.request_id(**self.identity_fields())


@dataclass
class RequestRecord(_Record):
    """On-disk committed request (architecture §2.1), content-addressed by ``request_id``.

    Nested blocks are plain dicts (their exact shape is pinned by WP2's live smoke test);
    ``content_sha256`` is the integrity hash of every other field.
    """

    schema_version: int
    request_id: str
    identity: dict[str, Any]
    seed_key: SeedKey
    engine_seed: int
    model: dict[str, Any]
    messages: tuple[Mapping[str, Any], ...]
    chat_template_kwargs: dict[str, Any]
    chat_template_hash: str
    sampling: dict[str, Any]
    prompt_token_ids: tuple[int, ...]
    prompt_tokens: int
    response: dict[str, Any]
    flops: dict[str, Any]
    timing: dict[str, Any]
    endpoint: dict[str, Any]
    attempts: int
    producer: dict[str, Any]
    content_sha256: str | None = None

    _nested = {"seed_key": SeedKey}

    def compute_content_sha256(self) -> str:
        return identity.content_sha256(self.to_dict())

    def with_content_sha256(self) -> "RequestRecord":
        return dataclasses.replace(self, content_sha256=self.compute_content_sha256())

    def verify(self) -> None:
        """Raise ``ProtocolError`` on a corrupt record (never silently regenerate)."""
        if self.content_sha256 != self.compute_content_sha256():
            raise ProtocolError(f"request record {self.request_id} failed its content hash")


# --------------------------------------------------------------------------- candidates


@dataclass(frozen=True)
class Evidence(_Record):
    claim: str
    support: str
    uncertainty: str  # one of UNCERTAINTY_LEVELS


@dataclass(frozen=True)
class Candidate(_Record):
    """The exact six-key §3.5 complete-candidate object (field order = wire order)."""

    approach: str
    evidence: tuple[Evidence, ...]
    alternatives_considered: tuple[str, ...]
    failure_checks: tuple[str, ...]
    final_answer: str
    confidence: float

    _nested_seq = {"evidence": Evidence}


@dataclass(frozen=True)
class CandidateRecord(_Record):
    """One sampling opportunity's outcome (architecture §2.2 ``candidates[]``)."""

    candidate_id: str  # sha256(request_id)
    request_id: str
    slot: int
    stage: str  # "root" | "round<k>" | "final" | "revise" | ...
    valid: bool
    failure_code: str | None
    candidate: Candidate | None
    candidate_sha256: str
    raw_content_sha256: str
    vote_key: str | None = None
    grouping_mode: str | None = None  # mc_letter | exact_norm | ast | exact_source

    _nested = {"candidate": Candidate}


@dataclass(frozen=True)
class Packet(_Record):
    """§4.3 bounded message packet (never a candidate).  ``serialized`` is the exact bytes sent."""

    packet_id: str
    sender_slot: int
    candidate_sha256: str
    fields: dict[str, Any]
    spans: tuple[Any, ...]
    truncated: dict[str, bool]
    final_partial: bool
    recipient_tokens: int
    serialized: str
    unavailable: bool


@dataclass(frozen=True)
class SubtaskResult(_Record):
    """Worker return object (handoff ``subtask_result.schema.json``); never a full-task candidate."""

    subtask_id: str
    contract: str
    status: str  # one of SUBTASK_STATUSES
    result: str
    assumptions: tuple[str, ...]
    evidence_handles: tuple[str, ...]
    confidence: float | None


@dataclass(frozen=True)
class Assignment(_Record):
    """One hub delegation (handoff ``coordinator_action.schema.json``, §4.2 CEN_FLAT)."""

    worker_slot: int
    subtask_id: str
    question: str
    source_handles: tuple[str, ...]
    required_output_type: str
    return_contract: str


@dataclass(frozen=True)
class FinalAction(_Record):
    """``{"action":"final","candidate":{...}}``."""

    candidate: Candidate

    action: ClassVar[str] = "final"
    _nested = {"candidate": Candidate}

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "candidate": self.candidate.to_dict()}


@dataclass(frozen=True)
class DelegateAction(_Record):
    """``{"action":"delegate","assignments":[...]}`` (1..N-1 unique worker slots)."""

    assignments: tuple[Assignment, ...]

    action: ClassVar[str] = "delegate"
    _nested_seq = {"assignments": Assignment}

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "assignments": [a.to_dict() for a in self.assignments]}


CoordinatorAction = Union[FinalAction, DelegateAction]


def coordinator_action_from_dict(data: Mapping[str, Any]) -> CoordinatorAction:
    """Dispatch on ``action``; the strict validator (WP3) runs before this is called."""
    action = data.get("action")
    if action == FinalAction.action:
        return FinalAction.from_dict(data)
    if action == DelegateAction.action:
        return DelegateAction.from_dict(data)
    raise ValueError(f"unknown coordinator action {action!r}")


# --------------------------------------------------------------------------- cells / episodes


@dataclass(frozen=True)
class CellSpec(_Record):
    """One unit of Slurm work (architecture §1.1 + P0-4 ``parallel_items`` + P0-7 ``lane``).

    ``B`` is the budget multiplier (1/2/4/8; 0 = not applicable, e.g. F banks).
    ``degree`` is set only for ``Method.DEGREE`` cells.
    """

    cell_id: str
    kind: CellKind
    module: str  # "F" | "A" | "N" | "D" | "E" | "M" | "B" | "C" | "eval"
    method: Method
    checkpoint: str  # size key into StudyConfig.checkpoints
    N: int
    B: int
    framing: Framing
    episode_rep: int
    split: str
    items: tuple[str, ...]
    depends_on: tuple[str, ...]
    max_inflight: int
    parallel_items: int
    lane: str  # "32B" | "14B" | "8B" | "4B" | "eval"
    degree: int | None = None

    _enum_fields = {"kind": CellKind, "method": Method, "framing": Framing}


@dataclass
class EpisodeResult(_Record):
    """One item file ``cells/<cell_id>/items/<source_id>.json`` (architecture §2.2).

    Generate cells fill ``episode`` (calls, candidates, packets, ledger, counters,
    selection, native_final); F bank cells fill ``bank`` instead.  The object is dict-like
    (``result["episode"]``) so aggregate code can treat files and objects alike.
    """

    schema_version: int
    cell: CellSpec
    source_id: str
    domain: Domain
    split: str
    status: str  # "complete" | "incomplete" | "context_failure" | "suspended"
    episode: dict[str, Any] | None
    bank: tuple[CandidateRecord, ...] | None
    timing: dict[str, Any]
    worker: dict[str, Any]
    code_version: str | None = None

    _enum_fields = {"domain": Domain}
    _nested = {"cell": CellSpec}
    _nested_seq = {"bank": CandidateRecord}

    # dict-like view -------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and any(f.name == key for f in dataclasses.fields(self))

    def __iter__(self) -> Iterator[str]:
        return iter(f.name for f in dataclasses.fields(self))

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self)]

    def items(self) -> list[tuple[str, Any]]:
        return list(self.to_dict().items())


__all__ = [name for name in globals() if not name.startswith("_")]
