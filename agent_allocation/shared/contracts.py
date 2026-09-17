"""Proposed shared interface seed. No networking or experiment execution.

The implementation lead owns changes to this file. All identifiers are immutable.
The public task type deliberately has no gold-answer/private-test fields.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Mapping, Protocol, Sequence

class OutputKind(str, Enum):
    MATH = 'math'
    OPTION = 'option'
    BOOLEAN = 'boolean'
    PYTHON = 'python'

class GenerationKind(str, Enum):
    ROOT = 'root'
    PRIVATE = 'private'
    PEER = 'peer'
    DEEP = 'deep'
    CP = 'communicate_then_private'
    POOLED = 'pooled'

class ParseStatus(str, Enum):
    VALID = 'valid'
    MISSING = 'missing_final'
    MALFORMED = 'malformed_final'
    EMPTY = 'empty_content'
    REFUSAL = 'refusal'

class GradeStatus(str, Enum):
    GRADED = 'graded'
    INVALID_OUTPUT = 'invalid_output'
    EVALUATOR_ERROR = 'evaluator_error'
    NOT_EVALUATED = 'not_evaluated'

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
            raise ValueError('Reasoning budget and completion headroom must be positive.')
        if min(self.root_slot, self.team_size, self.update_round, self.draw_id, self.repetition_id) < 0:
            raise ValueError('Indices must be nonnegative.')
        if len(self.parent_ids) != len(set(self.parent_ids)):
            raise ValueError('Duplicate parent reference.')

    @property
    def node_id(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()
        return sha256(encoded).hexdigest()

    @property
    def max_completion_tokens(self) -> int:
        return self.reasoning_budget + self.completion_headroom

@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cached_prompt_tokens: int | None
    actual_cost_usd: Decimal | None
    raw_usage: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Candidate:
    candidate_id: str
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
    policy_id: str
    item_id: str
    family: str
    team_size: int
    total_cycles: int
    peer_rounds: int
    candidate_ids: tuple[str, ...]
    required_node_ids: tuple[str, ...]
    selector_name: str
    selector_version: str

@dataclass(frozen=True)
class SelectionResult:
    policy_id: str
    selected_candidate_id: str | None
    selector_name: str
    selector_version: str
    pool_hash: str
    public_evidence_hash: str | None
    cpu_seconds: float
    status: str

@dataclass(frozen=True)
class GradeResult:
    candidate_id: str
    status: GradeStatus
    semantic_correct: bool | None
    operational_success: bool | None
    grader_version: str
    private_eval_hash: str
    detail_code: str | None = None

class Generator(Protocol):
    async def generate(self, spec: GenerationSpec, messages: Sequence[Mapping[str, str]]) -> Any: ...

class Selector(Protocol):
    def select(self, task: PublicTask, candidates: Sequence[Candidate],
               policy_id: str) -> SelectionResult: ...
