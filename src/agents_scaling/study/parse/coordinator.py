"""CEN_FLAT wire contracts: ``coordinator_action`` and ``subtask_result`` (WP3).

Spec §4.2 (CEN_FLAT: "bounded delegation to unique available worker slots with explicit
questions, allowed sources and checkable return contracts; or a final object containing
the complete candidate"), §3.5 (subtask outputs "use a separate schema ... Those confidence
targets are never substituted for original-task confidence"), §4.3 (packets and results
are fallible data).  Handoff schemas ``coordinator_action.schema.json`` (both branches,
``$comment``: "Cross-record validator enforces unique worker slots <= N_total-1 and
allowed handles") and ``subtask_result.schema.json`` (``$comment``: "never count this
object toward original-task pass@K").  Architecture §1.8; audit fixture T7 (typed errors;
the hub call is still debited by the policy, no worker is launched).

Type discipline: a :class:`~agents_scaling.study.types.SubtaskResult` has no conversion
to a :class:`~agents_scaling.study.types.Candidate` anywhere in the package — a worker
return can never enter a candidate pool.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Union

from agents_scaling.study import identity
from agents_scaling.study.parse.candidate import (
    FINISH_LENGTH,
    FINISH_STOP,
    SchemaError,
    StrictJSONError,
    check_exact_keys,
    check_str,
    check_str_list,
    check_unit_number,
    jcs_with_numbers,
    load_strict_json,
    strip_one_fence,
    validate_candidate_object,
)
from agents_scaling.study.types import (
    SUBTASK_STATUSES,
    Assignment,
    DelegateAction,
    FinalAction,
    SubtaskResult,
)

# --------------------------------------------------------------------------- frozen bounds (handoff schemas)

#: Handoff ``coordinator_action.schema.json`` bounds.
MAX_ASSIGNMENTS = 8
MAX_WORKER_SLOT = 8
SUBTASK_ID_MAX = 128
QUESTION_MAX = 8192
SOURCE_HANDLES_MAX_ITEMS = 32
HANDLE_MAX = 8192
REQUIRED_OUTPUT_TYPE_MAX = 256
RETURN_CONTRACT_MAX = 8192
#: Handoff ``subtask_result.schema.json`` bounds.
CONTRACT_MAX = 8192
RESULT_MAX = 32768
LIST_MAX_ITEMS = 32
LIST_STRING_MAX = 8192

ASSIGNMENT_KEYS: tuple[str, ...] = (
    "worker_slot",
    "subtask_id",
    "question",
    "source_handles",
    "required_output_type",
    "return_contract",
)
SUBTASK_RESULT_KEYS: tuple[str, ...] = (
    "subtask_id",
    "contract",
    "status",
    "result",
    "assumptions",
    "evidence_handles",
    "confidence",
)

#: The handle every worker may always cite (the original task); results are forwarded by
#: id (architecture §1.8 "{"task"} + forwarded result ids").
TASK_HANDLE = "task"

#: Typed coordinator failures (architecture §1.8 + ``DUP_SUBTASK``/``TRUNCATED``).
ACTION_ERROR_CODES: tuple[str, ...] = (
    "TRUNCATED",
    "INVALID",  # EMPTY / NOT_JSON / DUPLICATE_KEY / NONFINITE / TRAILING_TEXT / SCHEMA (see ``reason``)
    "TOO_MANY",
    "DUP_SLOT",
    "BAD_SLOT",
    "HIDDEN_HANDLE",
    "DUP_SUBTASK",
)
SUBTASK_ERROR_CODES: tuple[str, ...] = (
    "TRUNCATED",
    "EMPTY",
    "NOT_JSON",
    "DUPLICATE_KEY",
    "NONFINITE",
    "TRAILING_TEXT",
    "SCHEMA",
)


# --------------------------------------------------------------------------- error records


@dataclass(frozen=True)
class ActionError:
    """A hub output that is not a legal coordinator action (a *used* hub call, T7).

    ``code`` ∈ ``ACTION_ERROR_CODES``; ``reason`` carries the JSON-level sub-code for
    ``INVALID``; ``raw_sha256`` hashes the exact content.
    """

    code: str
    detail: str
    raw_sha256: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.code not in ACTION_ERROR_CODES:
            raise ValueError(f"unknown action error code {self.code!r}")


@dataclass(frozen=True)
class SubtaskError:
    """A worker output that is not a legal ``subtask_result`` (the hub receives an unavailable packet)."""

    code: str
    detail: str
    raw_sha256: str

    def __post_init__(self) -> None:
        if self.code not in SUBTASK_ERROR_CODES:
            raise ValueError(f"unknown subtask error code {self.code!r}")


CoordinatorParse = Union[FinalAction, DelegateAction, ActionError]
SubtaskParse = Union[SubtaskResult, SubtaskError]


# --------------------------------------------------------------------------- helpers


def _check_finish(finish_reason: str) -> None:
    if finish_reason not in (FINISH_STOP, FINISH_LENGTH):
        raise ValueError(f"finish_reason {finish_reason!r} is not a parser outcome (stop|length)")


def _check_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{name} must be an integer, got {type(value).__name__}")
    return value


def validate_assignment(obj: Any, index: int) -> Assignment:
    """Per-record schema of one assignment (no cross-record rules here)."""
    name = f"assignments[{index}]"
    body = check_exact_keys(obj, name, ASSIGNMENT_KEYS)
    slot = _check_int(body["worker_slot"], f"{name}.worker_slot")
    if not 1 <= slot <= MAX_WORKER_SLOT:
        raise SchemaError(f"{name}.worker_slot={slot} outside [1,{MAX_WORKER_SLOT}]")
    return Assignment(
        worker_slot=slot,
        subtask_id=check_str(body["subtask_id"], f"{name}.subtask_id", SUBTASK_ID_MAX),
        question=check_str(body["question"], f"{name}.question", QUESTION_MAX),
        source_handles=check_str_list(
            body["source_handles"], f"{name}.source_handles", SOURCE_HANDLES_MAX_ITEMS, HANDLE_MAX
        ),
        required_output_type=check_str(
            body["required_output_type"], f"{name}.required_output_type", REQUIRED_OUTPUT_TYPE_MAX
        ),
        return_contract=check_str(body["return_contract"], f"{name}.return_contract", RETURN_CONTRACT_MAX),
    )


def _cross_record(assignments: tuple[Assignment, ...], N_total: int, allowed: frozenset[str]) -> tuple[str, str] | None:
    """Return ``(code, detail)`` for the first violated §4.2 cross-record rule, else ``None``."""
    n_workers = N_total - 1
    if len(assignments) > n_workers:
        return "TOO_MANY", f"{len(assignments)} assignments > N_total-1={n_workers}"
    seen_slots: set[int] = set()
    seen_ids: set[str] = set()
    for i, a in enumerate(assignments):
        if not 1 <= a.worker_slot <= n_workers:
            return "BAD_SLOT", f"assignments[{i}].worker_slot={a.worker_slot} outside [1,{n_workers}]"
        if a.worker_slot in seen_slots:
            return "DUP_SLOT", f"assignments[{i}] reuses worker_slot {a.worker_slot}"
        seen_slots.add(a.worker_slot)
        if a.subtask_id in seen_ids:
            return "DUP_SUBTASK", f"assignments[{i}] reuses subtask_id {a.subtask_id!r}"
        seen_ids.add(a.subtask_id)
        hidden = [h for h in a.source_handles if h not in allowed]
        if hidden:
            return "HIDDEN_HANDLE", f"assignments[{i}] cites handles outside the allowed set: {hidden[:4]}"
    return None


# --------------------------------------------------------------------------- public parsers


def parse_coordinator_action(
    content: str | None,
    N_total: int,
    allowed_handles: Iterable[str] = (TASK_HANDLE,),
    finish_reason: str = FINISH_STOP,
) -> CoordinatorParse:
    """Parse a hub output into ``FinalAction`` | ``DelegateAction`` | ``ActionError`` (§4.2).

    Per-record rules follow ``coordinator_action.schema.json`` (exact keys per branch, the
    final candidate through :func:`validate_candidate_object`, 1..8 assignments with the
    frozen string bounds).  Cross-record rules (in this order, first violation wins):
    ``TOO_MANY`` (more than ``N_total-1`` assignments, so any delegation at N_total=1),
    ``BAD_SLOT`` (``worker_slot`` outside ``[1, N_total-1]``), ``DUP_SLOT``, ``DUP_SUBTASK``,
    ``HIDDEN_HANDLE`` (``source_handles`` ⊄ ``allowed_handles``; ``"task"`` plus the ids of
    results the hub actually holds).  JSON-level failures are ``INVALID`` with ``reason`` =
    the strict-JSON/schema sub-code; ``finish_reason == "length"`` is ``TRUNCATED``.
    """
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise TypeError("content must be a str or None")
    if isinstance(N_total, bool) or not isinstance(N_total, int) or N_total < 1:
        raise ValueError(f"N_total must be an int >= 1, got {N_total!r}")
    _check_finish(finish_reason)
    allowed = frozenset(allowed_handles)
    if not all(isinstance(h, str) for h in allowed):
        raise TypeError("allowed_handles must be strings")
    raw_sha256 = identity.sha256_hex(content)
    if finish_reason == FINISH_LENGTH:
        return ActionError("TRUNCATED", "finish_reason=length", raw_sha256)
    try:
        value = load_strict_json(strip_one_fence(content))
    except StrictJSONError as exc:
        return ActionError("INVALID", exc.detail, raw_sha256, reason=exc.code)
    try:
        if not isinstance(value, Mapping):
            raise SchemaError(f"coordinator_action must be an object, got {type(value).__name__}")
        action = value.get("action")
        if action == FinalAction.action:
            body = check_exact_keys(value, "final action", ("action", "candidate"))
            return FinalAction(candidate=validate_candidate_object(body["candidate"]))
        if action == DelegateAction.action:
            body = check_exact_keys(value, "delegate action", ("action", "assignments"))
            raw = body["assignments"]
            if not isinstance(raw, list):
                raise SchemaError(f"assignments must be an array, got {type(raw).__name__}")
            if not 1 <= len(raw) <= MAX_ASSIGNMENTS:
                raise SchemaError(f"assignments has {len(raw)} items, allowed 1..{MAX_ASSIGNMENTS}")
            assignments = tuple(validate_assignment(item, i) for i, item in enumerate(raw))
        else:
            raise SchemaError(f"action must be 'final' or 'delegate', got {action!r}")
    except SchemaError as exc:
        return ActionError("INVALID", exc.detail, raw_sha256, reason="SCHEMA")
    violation = _cross_record(assignments, N_total, allowed)
    if violation is not None:
        return ActionError(violation[0], violation[1], raw_sha256)
    return DelegateAction(assignments=assignments)


def validate_subtask_result_object(obj: Any) -> SubtaskResult:
    """Validate a decoded value against ``subtask_result.schema.json``; raise :class:`SchemaError`."""
    body = check_exact_keys(obj, "subtask_result", SUBTASK_RESULT_KEYS)
    status = body["status"]
    if not isinstance(status, str) or status not in SUBTASK_STATUSES:
        raise SchemaError(f"status={status!r} not in {SUBTASK_STATUSES}")
    confidence = body["confidence"]
    return SubtaskResult(
        subtask_id=check_str(body["subtask_id"], "subtask_id", SUBTASK_ID_MAX),
        contract=check_str(body["contract"], "contract", CONTRACT_MAX),
        status=status,
        result=check_str(body["result"], "result", RESULT_MAX),
        assumptions=check_str_list(body["assumptions"], "assumptions", LIST_MAX_ITEMS, LIST_STRING_MAX),
        evidence_handles=check_str_list(
            body["evidence_handles"], "evidence_handles", LIST_MAX_ITEMS, LIST_STRING_MAX
        ),
        confidence=None if confidence is None else check_unit_number(confidence, "confidence"),
    )


def parse_subtask_result(content: str | None, finish_reason: str = FINISH_STOP) -> SubtaskParse:
    """Parse a worker output into a ``SubtaskResult`` or a typed ``SubtaskError``.

    Same salvage rule as candidates (strip one fence, strict JSON); schema per the handoff:
    exact seven keys, ``status`` ∈ complete/partial/failed, ``result`` ≤ 32,768 code points,
    ``confidence`` a finite number in [0,1] or ``null`` (about the *subtask* contract, §3.5).
    The returned object is never a candidate and there is deliberately no conversion.
    """
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise TypeError("content must be a str or None")
    _check_finish(finish_reason)
    raw_sha256 = identity.sha256_hex(content)
    if finish_reason == FINISH_LENGTH:
        return SubtaskError("TRUNCATED", "finish_reason=length", raw_sha256)
    try:
        value = load_strict_json(strip_one_fence(content))
        return validate_subtask_result_object(value)
    except StrictJSONError as exc:  # SchemaError is a StrictJSONError with code SCHEMA
        return SubtaskError(exc.code, exc.detail, raw_sha256)


def subtask_result_sha256(result: SubtaskResult) -> str:
    """RFC 8785 content hash of a validated subtask result (its ``Packet.candidate_sha256`` slot)."""
    if not isinstance(result, SubtaskResult):
        raise TypeError("subtask_result_sha256 expects a SubtaskResult")
    return identity.sha256_hex(jcs_with_numbers(result.to_dict()))


__all__ = [
    "ACTION_ERROR_CODES",
    "ActionError",
    "CoordinatorParse",
    "MAX_ASSIGNMENTS",
    "SUBTASK_ERROR_CODES",
    "SubtaskError",
    "SubtaskParse",
    "TASK_HANDLE",
    "parse_coordinator_action",
    "parse_subtask_result",
    "subtask_result_sha256",
    "validate_assignment",
    "validate_subtask_result_object",
]
