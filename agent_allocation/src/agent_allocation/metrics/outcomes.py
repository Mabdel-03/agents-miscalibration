"""Tri-state candidate outcomes and delivered outcomes with fixed-denominator bounds (MASTER_SPEC §12, §15.14).

Candidate states (``CandidateState``):

* ``C`` -- graded ``semantic_correct is True``.
* ``W`` -- graded ``semantic_correct is False`` on a validly parsed candidate: a real wrong answer.
* ``I`` -- no valid answer: parse status other than ``VALID`` (whatever the grader did), or grader
  ``INVALID_OUTPUT``. Operationally unsuccessful, known not correct, and never an answer-class switch.
* ``U`` -- unknown: no grade yet, ``EVALUATOR_ERROR``, ``NOT_EVALUATED``, or ``semantic_correct is None``.
* ``M`` -- the candidate itself is absent (node not terminal or never admitted). Unknown for every bound; kept
  distinct so administrative gaps are never read as model errors.

Contradictions (an invalid candidate graded correct, ``INVALID_OUTPUT`` with ``semantic_correct=True``, a grade for a
different candidate) raise.

Delivered states (``DeliveredState``) for a sealed selection: ``correct``, ``wrong``, ``invalid_output`` (the delivered
candidate has no valid answer), ``unknown`` (its grade is unknown or it is absent), ``abstained`` (all-invalid pool),
``paused`` (public-check infrastructure error), ``missing_selection`` (no sealed row: a blocker, never zero).
Bounds: ``lo`` counts only known-correct, ``hi`` credits every unknown as possibly correct, so ``[lo, hi]`` brackets every
feasible labelling on a fixed denominator.

Config keys consumed: none.
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable

from agent_allocation.contracts import Candidate, GradeResult, GradeStatus, ParseStatus, SelectionResult, SelectionStatus

__all__ = [
    "CandidateState", "DeliveredState", "candidate_state", "coverage_bounds", "delivered_bounds", "delivered_correct",
    "delivered_state",
]


class CandidateState(str, Enum):
    CORRECT = "C"
    WRONG = "W"
    INVALID = "I"
    UNKNOWN = "U"
    MISSING = "M"


class DeliveredState(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    INVALID_OUTPUT = "invalid_output"
    UNKNOWN = "unknown"
    ABSTAINED = "abstained"
    PAUSED = "paused"
    MISSING_SELECTION = "missing_selection"


_KNOWN_NOT_CORRECT = frozenset({CandidateState.WRONG, CandidateState.INVALID})
_POSSIBLY_CORRECT = frozenset({CandidateState.CORRECT, CandidateState.UNKNOWN, CandidateState.MISSING})


def candidate_state(candidate: Candidate | None, grade: GradeResult | None) -> CandidateState:
    if candidate is None:
        if grade is not None:
            raise ValueError(f"grade for {grade.candidate_id} has no candidate record")
        return CandidateState.MISSING
    if grade is not None and grade.candidate_id != candidate.candidate_id:
        raise ValueError(f"grade {grade.candidate_id} joined to candidate {candidate.candidate_id}")
    invalid_parse = candidate.parse_status is not ParseStatus.VALID
    if grade is not None and grade.semantic_correct not in (True, False, None):
        raise TypeError(f"{candidate.candidate_id}: semantic_correct must be bool | None")
    if grade is not None and grade.semantic_correct is True and (invalid_parse
                                                                 or grade.status is not GradeStatus.GRADED):
        raise ValueError(f"{candidate.candidate_id}: correct grade on an invalid or ungraded candidate "
                         f"(parse {candidate.parse_status.value}, grade {grade.status.value})")
    if invalid_parse or (grade is not None and grade.status is GradeStatus.INVALID_OUTPUT):
        return CandidateState.INVALID
    if grade is None or grade.status is not GradeStatus.GRADED or grade.semantic_correct is None:
        return CandidateState.UNKNOWN
    return CandidateState.CORRECT if grade.semantic_correct else CandidateState.WRONG


def coverage_bounds(states: Iterable[CandidateState]) -> tuple[int, int]:
    """``(lo, hi)`` of "at least one candidate is correct"."""
    states = list(states)
    lo = int(any(s is CandidateState.CORRECT for s in states))
    hi = int(any(s in _POSSIBLY_CORRECT for s in states))
    return lo, hi


def delivered_state(selection: SelectionResult | None, selected: CandidateState | None) -> DeliveredState:
    if selection is None:
        return DeliveredState.MISSING_SELECTION
    status = SelectionStatus(selection.status)
    if status is SelectionStatus.ABSTAINED_ALL_INVALID:
        return DeliveredState.ABSTAINED
    if status is SelectionStatus.PAUSED_CHECK_ERROR:
        return DeliveredState.PAUSED
    if selected is None:
        raise ValueError(f"{selection.policy_id}: status {status.value} needs the selected candidate's state")
    return {
        CandidateState.CORRECT: DeliveredState.CORRECT,
        CandidateState.WRONG: DeliveredState.WRONG,
        CandidateState.INVALID: DeliveredState.INVALID_OUTPUT,
        CandidateState.UNKNOWN: DeliveredState.UNKNOWN,
        CandidateState.MISSING: DeliveredState.UNKNOWN,
    }[selected]


def delivered_bounds(state: DeliveredState) -> tuple[int, int]:
    if state is DeliveredState.CORRECT:
        return 1, 1
    if state in (DeliveredState.WRONG, DeliveredState.INVALID_OUTPUT, DeliveredState.ABSTAINED):
        return 0, 0
    return 0, 1


def delivered_correct(state: DeliveredState) -> bool | None:
    """Operational tri-state: abstention and invalid output are ``False``; unknown/paused/missing are ``None``."""
    lo, hi = delivered_bounds(state)
    return None if lo != hi else bool(lo)


def is_known_not_correct(state: CandidateState) -> bool:
    return state in _KNOWN_NOT_CORRECT
