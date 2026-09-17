"""Frozen, gold-blind answer identities for option and boolean finals (``text-normalize-v1``).

These are the only parsers the text grader applies to ``Candidate.final_answer`` for MMLU-Pro and BBH. They
never see a reference answer, so agent 06 may import them to build voting classes that agree with grading.

* Option identity: after removing outer whitespace, the final must be exactly ``X`` or ``(X)`` with ``X`` an
  uppercase letter that is one of the item's ``allowed_choices``. Nothing else is accepted: no lowercase, no
  trailing period, no ``Answer:`` prefix, no search through rationale text.
* Boolean identity: after removing outer characters in :data:`BOOLEAN_OUTER_CHARACTERS` (whitespace and
  ``. , ; : ! ? " ' ` * _ ( ) [ ] { } $``), the remaining text compared case-insensitively must be exactly
  ``true`` or ``false``; it maps to ``"True"`` / ``"False"``.
"""
from __future__ import annotations

import re
from typing import Sequence

NORMALIZER_VERSION = "text-normalize-v1"
BOOLEAN_OUTER_CHARACTERS = " \t\n\r\f\v.,;:!?\"'`*_()[]{}$"
_OPTION = re.compile(r"([A-Z])|\(([A-Z])\)")


def option_identity(final_answer: str | None, allowed_choices: Sequence[str]) -> str | None:
    """The legal option letter named by ``final_answer``, or ``None`` when it is not exactly ``X`` / ``(X)``."""
    if final_answer is None:
        return None
    match = _OPTION.fullmatch(final_answer.strip())
    if match is None:
        return None
    letter = match.group(1) or match.group(2)
    return letter if letter in tuple(allowed_choices) else None


def boolean_identity(final_answer: str | None) -> str | None:
    """``"True"`` / ``"False"`` for an explicit boolean final, else ``None``."""
    if final_answer is None:
        return None
    text = final_answer.strip().strip(BOOLEAN_OUTER_CHARACTERS).casefold()
    if text == "true":
        return "True"
    if text == "false":
        return "False"
    return None
