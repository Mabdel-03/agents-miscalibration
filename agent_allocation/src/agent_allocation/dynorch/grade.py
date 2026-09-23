"""WP-B: delivered answer -> ``Candidate`` -> grade, and the run-level outcome fields (contract §6; plan §2 B7).

The delivered answer of a run is the visible content of the last root-conversation assistant message with
non-empty content *as served by the proxy* (``httpd.ReplyLog.delivered``); the harness's own CLI result is only a
logged cross-check (``cli_result_mismatch``). ``candidate_from_text`` applies exactly the text-kind rules of
``extraction/final.py::extract_candidate`` (rules 4-7: whitespace-only -> EMPTY; exactly one byte-exact
``<final>``/``</final>`` pair with non-blank inner text -> VALID; neither tag -> MISSING; anything else ->
MALFORMED; ``candidate_id == node_id``), without the transport/blocked rules that need a ``GenerationResult`` (a
served proxy reply is transport-OK by construction) -- the parity test builds a real ``GenerationResult`` and
compares. Python items are not text kinds and raise. Grading is ``grading/text/grade.py::grade_text`` verbatim.

``finalize_run`` fills ``parse_status``, ``final_answer``, ``grade_status``, ``correct`` and ``grade_detail`` on the
``RunRecord`` in place and returns it; ``outcome_y``/``delivered_state`` derive the sealed-table values the WP-C
tables carry: VALID and graded -> ``delivered`` with ``y in {1.0, 0.0}``; anything not VALID -> ``op_no_output``
with ``y = 0.0`` (exactly the ladder's invalid-candidate rule); an evaluator error -> ``unknown`` / NaN; a run whose
final attempt is still an infrastructure failure (``transport_failed``, ``refused_before_start``, a ``failed``
attempt with a ``rerun_reason``) -> ``unknown`` / NaN (frozen table: all-infra -> y = NaN), never 0.
"""
from __future__ import annotations

import math
from typing import Any

from agent_allocation._vendor.extraction_tags import TAG_EMPTY, TAG_MISSING, TAG_OK, strict_single_pair
from agent_allocation.contracts import (
    Candidate, GradeResult, GradeStatus, OutputKind, ParseStatus, PrivateEvalSpec, PublicTask,
)
from agent_allocation.dynorch.records import RunRecord
from agent_allocation.extraction.final import FINAL_CLOSE, FINAL_OPEN, RATIONALE_CLOSE, RATIONALE_OPEN
from agent_allocation.grading.text.grade import grade_text

__all__ = [
    "INFRA_STATUSES", "TEXT_KINDS", "candidate_from_text", "delivered_state", "finalize_run", "grade_candidate",
    "is_infra_failure", "outcome_y",
]

TEXT_KINDS: frozenset[OutputKind] = frozenset({OutputKind.MATH, OutputKind.OPTION, OutputKind.BOOLEAN})
INFRA_STATUSES: frozenset[str] = frozenset({"transport_failed", "refused_before_start"})


def candidate_from_text(text: str | None, task: PublicTask, *, node_id: str, finish_reason: str | None = None,
                        ) -> Candidate:
    """``extract_candidate``'s text-kind rules over a bare string (module docstring)."""
    if not isinstance(task, PublicTask):
        raise TypeError(f"candidate_from_text expects PublicTask, got {type(task).__name__}")
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("candidate_from_text needs a non-empty node_id")
    if task.output_kind not in TEXT_KINDS:
        raise ValueError(f"{task.item_id}: output kind {task.output_kind.value!r} is not a text kind")
    if text is not None and not isinstance(text, str):
        raise TypeError(f"text must be str or None, got {type(text).__name__}")
    visible = text if isinstance(text, str) else ""

    def build(status: ParseStatus, final_answer: str | None = None, rationale: str | None = None) -> Candidate:
        return Candidate(candidate_id=node_id, node_id=node_id, item_id=task.item_id, visible_content=visible,
                         final_answer=final_answer, rationale=rationale, parse_status=status,
                         finish_reason=finish_reason, output_kind=task.output_kind)

    if not visible.strip():
        return build(ParseStatus.EMPTY)
    rationale_status, rationale = strict_single_pair(visible, RATIONALE_OPEN, RATIONALE_CLOSE)
    rationale = rationale if rationale_status == TAG_OK else None
    status, inner = strict_single_pair(visible, FINAL_OPEN, FINAL_CLOSE)
    if status == TAG_MISSING:
        return build(ParseStatus.MISSING, rationale=rationale)
    if status == TAG_EMPTY or status != TAG_OK or inner is None or not inner.strip():
        return build(ParseStatus.MALFORMED, rationale=rationale)
    return build(ParseStatus.VALID, final_answer=inner, rationale=rationale)


def grade_candidate(candidate: Candidate, private: PrivateEvalSpec, *, timeout_s: float = 5.0) -> GradeResult:
    """``grade_text`` with the plan's 5 s Math-Verify timeout."""
    return grade_text(candidate, private, timeout_s=timeout_s)


def is_infra_failure(record: RunRecord) -> bool:
    """Whether the record's (final) attempt ended in infrastructure, not model, failure."""
    if record.status in INFRA_STATUSES:
        return True
    return record.status == "failed" and record.rerun_reason is not None


def finalize_run(record: RunRecord, task: PublicTask, private: PrivateEvalSpec) -> RunRecord:
    """Fill the parse/grade fields of a closed run in place (module docstring) and return it."""
    if task.item_id != record.spec.item_id:
        raise ValueError(f"task {task.item_id} does not belong to run {record.spec.run_key} ({record.spec.item_id})")
    if private.item_id != record.spec.item_id:
        raise ValueError(f"private spec {private.item_id} does not belong to run {record.spec.run_key}")
    if record.status == "running" or record.status == "planned":
        raise ValueError(f"run {record.spec.run_key} is still {record.status}; close it before finalizing")
    if is_infra_failure(record):
        record.parse_status = None
        record.final_answer = None
        record.grade_status = None
        record.correct = None
        record.grade_detail = None
        return record
    candidate = candidate_from_text(record.final_message, task, node_id=record.spec.run_key)
    record.parse_status = candidate.parse_status.value
    record.final_answer = candidate.final_answer
    graded = grade_candidate(candidate, private)
    record.grade_status = graded.status.value
    record.grade_detail = graded.detail_code
    if graded.status is GradeStatus.GRADED:
        record.correct = bool(graded.semantic_correct)
    elif graded.status is GradeStatus.INVALID_OUTPUT:
        record.correct = False
    else:
        record.correct = None
    return record


def delivered_state(record: RunRecord) -> str:
    """``pending`` (not finalised) | ``delivered`` | ``op_no_output`` | ``unknown`` (sealed-table vocabulary)."""
    if record.status in ("planned", "running"):
        return "pending"
    if is_infra_failure(record):
        return "unknown"
    if record.grade_status is None:
        return "pending"
    if record.parse_status != ParseStatus.VALID.value:
        return "op_no_output"
    if record.grade_status == GradeStatus.GRADED.value:
        return "delivered"
    if record.grade_status == GradeStatus.INVALID_OUTPUT.value:
        return "op_no_output"
    return "unknown"


def outcome_y(record: RunRecord) -> float:
    """``1.0`` correct, ``0.0`` wrong or not delivered, ``NaN`` unknown (evaluator error / infra / pending)."""
    state = delivered_state(record)
    if state == "delivered":
        return 1.0 if record.correct else 0.0
    if state == "op_no_output":
        return 0.0
    return math.nan


def outcome_row(record: RunRecord) -> dict[str, Any]:
    """The outcome-facing fields WP-C's tables read, in one place."""
    return {"delivered_state": delivered_state(record), "y": outcome_y(record), "parse_status": record.parse_status,
            "grade_status": record.grade_status, "correct": record.correct, "final_answer": record.final_answer}
