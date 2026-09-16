"""MATH grading: a thin wrapper over ``ard.graders.math_equiv`` (spec 5.1; plan §2.5).

The sympy/lark equivalence rule is the one piece of ARD36 grading nextdyn keeps *verbatim*
rather than vendoring: nothing about it carries ARD36 vocabulary and re-implementing it
would give two answers to "is ``\\frac12`` the same as ``0.5``".  ``ARD_IMPORTS`` allowlists
``agents_scaling.ard.graders.math_equiv`` for exactly this.

What this module adds is the uniform family interface -- ``canonical_class(answer, target)``
and ``grade(answer, gold, target)`` returning a :class:`~agents_scaling.nextdyn.graders.Verdict`
-- and the mapping of the ARD statuses onto nextdyn's five (a ``timeout`` stays a
``timeout``: it is unresolved, never ``incorrect``).

Writer side (R7): importing it pulls ``agents_scaling.ard.graders``, which defines ARD36's
gold store.
"""

from __future__ import annotations

from typing import Final, Mapping

from agents_scaling.ard.graders import math_equiv
from agents_scaling.nextdyn.errors import ConfigError
from agents_scaling.nextdyn.graders import (
    STATUS_AMBIGUOUS,
    STATUS_CORRECT,
    STATUS_INCORRECT,
    STATUS_TIMEOUT,
    GradeTarget,
    Verdict,
)

#: The grading deadline of one MATH answer, in seconds (``math_equiv``'s own default; the
#: GRADE cell walltime comes from ``FamilySpec.grade_resources``).
DEFAULT_TIMEOUT_S: Final[float] = math_equiv.DEFAULT_TIMEOUT_S

_STATUS_MAP: Final[Mapping[str, str]] = {
    math_equiv.STATUS_CORRECT: STATUS_CORRECT,
    math_equiv.STATUS_INCORRECT: STATUS_INCORRECT,
    math_equiv.STATUS_AMBIGUOUS: STATUS_AMBIGUOUS,
    math_equiv.STATUS_TIMEOUT: STATUS_TIMEOUT,
}


def canonical_class(answer: str | None, target: GradeTarget) -> str | None:
    """The sympy canonical form of ``answer`` (gold-independent), or ``None`` when empty."""
    del target  # a MATH class depends on the answer alone: no options, no permutation
    return math_equiv.canonical_class(answer)


def grade(answer: str | None, gold: Mapping[str, object], target: GradeTarget,
          *, timeout_s: float = DEFAULT_TIMEOUT_S) -> Verdict:
    """Grade against ``gold["answer"]`` (the boxed reference of the source row)."""
    reference = gold.get("answer")
    if not isinstance(reference, str) or not reference.strip():
        raise ConfigError(f"{target.source_item_id}: MATH gold needs a non-empty 'answer'")
    verdict = math_equiv.compare(answer, reference, timeout_s=timeout_s)
    return Verdict(status=_STATUS_MAP[verdict.status], detail=verdict.detail,
                   extracted=verdict.candidate_normalised or None,
                   canonical_class=canonical_class(answer, target))


__all__ = ["DEFAULT_TIMEOUT_S", "canonical_class", "grade", "math_equiv"]
