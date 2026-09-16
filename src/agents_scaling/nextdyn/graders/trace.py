"""E2b receiver-trace grading: a re-export of ``ard.graders.trace`` (spec 9.5; plan §2.5).

The mod-97 chain *evaluator* is imported unchanged (``ARD_IMPORTS`` allowlists
``agents_scaling.ard.graders.trace``); WP5's ``nextdyn.mod97`` **generator** is a separate
implementation that must never import it, and ``test_mod97_evaluator_independent_of_generator``
is the AST test that keeps the two apart.  ARD36's own defect list notes that
``ard/graders.grade()`` never dispatched this family; here it is reached through
``FamilySpec.grader`` like every other family.

Writer side (R7).
"""

from __future__ import annotations

from typing import Final, Mapping

from agents_scaling.ard.graders import trace as _trace
from agents_scaling.nextdyn.errors import ConfigError
from agents_scaling.nextdyn.graders import (
    STATUS_AMBIGUOUS,
    STATUS_CORRECT,
    STATUS_INCORRECT,
    STATUS_TIMEOUT,
    GradeTarget,
    Verdict,
)

#: The evaluator surface other modules re-use by name (never re-implemented here).
parse_program = _trace.parse_program
evaluate_program = _trace.evaluate_program
check_trace = _trace.check_trace
extract_residue = _trace.extract_residue
RECEIVER_MODULUS: Final[int] = _trace.RECEIVER_MODULUS

_STATUS_MAP: Final[Mapping[str, str]] = {
    _trace.STATUS_CORRECT: STATUS_CORRECT,
    _trace.STATUS_INCORRECT: STATUS_INCORRECT,
    _trace.STATUS_AMBIGUOUS: STATUS_AMBIGUOUS,
    _trace.STATUS_TIMEOUT: STATUS_TIMEOUT,
}


def canonical_class(answer: str | None, target: GradeTarget) -> str | None:
    """The residue the answer states, or ``None`` (gold-independent)."""
    del target
    return _trace.canonical_class(answer)


def grade(answer: str | None, gold: Mapping[str, object], target: GradeTarget) -> Verdict:
    """Grade against ``gold["residue"]``, the privately computed final residue."""
    reference = gold.get("residue")
    if not isinstance(reference, int) or isinstance(reference, bool):
        raise ConfigError(f"{target.source_item_id}: trace gold needs an integer 'residue'")
    verdict = _trace.grade(answer, reference)
    value = extract_residue(answer)
    return Verdict(status=_STATUS_MAP[verdict.status], detail=verdict.detail,
                   extracted=None if value is None else str(value),
                   canonical_class=canonical_class(answer, target))


__all__ = ["RECEIVER_MODULUS", "canonical_class", "check_trace", "evaluate_program",
           "extract_residue", "grade", "parse_program"]
