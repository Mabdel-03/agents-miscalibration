"""Grader vocabulary: the four statuses, the verdict shape and the per-decision target.

ARD36's grader facade held the statuses, the family dispatch **and** the gold store in one
module, so anything that wanted to *canonicalize* an answer (voting, packet construction)
imported the gold reader as well.  Here the package root holds only pure vocabulary --
importing it reaches nothing but the standard library -- and the three graders under it are
pure functions of ``(answer, gold, target)``:

=====================  =========================================================
module                 rule
=====================  =========================================================
``graders.mc``         parametric multiple choice (4-26 options), runtime-safe
``graders.math``       wraps ``ard.graders.math_equiv`` (sympy/lark)
``graders.trace``      re-exports ``ard.graders.trace`` (the E2b evaluator)
``graders.families``   family dispatch, ``grader_hash`` and the ``GoldStore``
=====================  =========================================================

Only ``graders.families`` (and, through their ARD imports, ``graders.math`` /
``graders.trace``) may be reached from the writer side: they carry or import a gold store.
``graders.mc`` is deliberately gold-free -- gold arrives as an argument -- so the renderer in
``families.py`` can share its one parametric alphabet without dragging gold into the
generation import closure (R7, ``tests/nextdyn/test_leakage.py``).

Spec 4.4, 4.6, 5.1, 6.5; plan §2.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from agents_scaling.nextdyn.errors import ConfigError

STATUS_CORRECT: Final[str] = "correct"
STATUS_INCORRECT: Final[str] = "incorrect"
STATUS_AMBIGUOUS: Final[str] = "ambiguous"
STATUS_TIMEOUT: Final[str] = "timeout"
STATUS_ERROR: Final[str] = "error"
#: Every status a grader may return.  ``ambiguous``/``timeout``/``error`` are *never* silently
#: turned into ``incorrect``: they become ``U`` at the writer (spec 4.6, plan §2.5).
STATUSES: Final[tuple[str, ...]] = (
    STATUS_CORRECT, STATUS_INCORRECT, STATUS_AMBIGUOUS, STATUS_TIMEOUT, STATUS_ERROR,
)
#: Statuses that resolve the sample; the rest are unresolved.
RESOLVED_STATUSES: Final[tuple[str, ...]] = (STATUS_CORRECT, STATUS_INCORRECT)

#: Bumped when the grading *rules* change independently of the module bytes.
GRADER_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True)
class GradeTarget:
    """What a grader needs to know about the item variant it is grading.

    ``permutation[displayed position] = canonical option index`` (the frozen per-variant
    option permutation) and ``options`` are the **displayed** option texts, so a grader can
    resolve a non-letter answer that exactly matches one displayed option and then map it
    back to the permutation-invariant canonical option id.  Both are empty for an
    ``exactMatch`` family.
    """

    family: str
    source_item_id: str
    item_variant_id: str
    answer_format: str
    permutation: tuple[int, ...] = ()
    options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "permutation", tuple(int(p) for p in self.permutation))
        object.__setattr__(self, "options", tuple(str(o) for o in self.options))
        if self.permutation and sorted(self.permutation) != list(range(len(self.permutation))):
            raise ConfigError(f"{self.item_variant_id}: option permutation {self.permutation} is "
                              "not a permutation of 0..n-1")
        if self.options and len(self.options) != len(self.permutation):
            raise ConfigError(f"{self.item_variant_id}: {len(self.options)} displayed options but "
                              f"{len(self.permutation)} permutation entries")


@dataclass(frozen=True)
class Verdict:
    """One grader's answer about one sample.  ``canonical_class`` is gold-independent."""

    status: str
    detail: str
    extracted: str | None
    canonical_class: str | None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ConfigError(f"unknown grader status {self.status!r}; expected one of {STATUSES}")

    @property
    def resolved(self) -> bool:
        return self.status in RESOLVED_STATUSES


__all__ = [
    "GRADER_SCHEMA_VERSION", "GradeTarget", "RESOLVED_STATUSES", "STATUSES", "STATUS_AMBIGUOUS",
    "STATUS_CORRECT", "STATUS_ERROR", "STATUS_INCORRECT", "STATUS_TIMEOUT", "Verdict",
]
