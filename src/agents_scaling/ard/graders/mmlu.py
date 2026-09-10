"""MMLU-Pro grader: exact permuted option id (WP3).

Spec §5.1: "MMLU grouping: exact option ID after a fixed per-item option permutation.
All agents/protocols see the same permutation for that item.  No option key is exposed."
The public item carries the *permuted* option texts and the permutation; the permuted key
lives only in the protected namespace (``ard/items_build.py``), so this module receives
the key from the evaluator-side gold store and never reads a dataset.

Extraction reuses the frozen, outcome-blind
:func:`agents_scaling.study.selection.normalize.hle_mc_key`, whose accepted forms already
cover ``A``-``J`` — the full alphabet of a ten-option MMLU-Pro item
(:data:`agents_scaling.ard.contract.MMLU_REQUIRED_OPTIONS`).

Statuses follow the WP0 contract.  A non-letter final that exactly matches one displayed
option text resolves to that option's letter; a non-letter final that matches nothing is
``ambiguous`` (explicit), never a silent ``incorrect``.  An empty answer is ``incorrect``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from agents_scaling.ard.contract import MC_LETTERS, MMLU_REQUIRED_OPTIONS
from agents_scaling.study.selection.normalize import hle_mc_key

STATUS_CORRECT = "correct"
STATUS_INCORRECT = "incorrect"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_TIMEOUT = "timeout"
STATUSES: tuple[str, ...] = (STATUS_CORRECT, STATUS_INCORRECT, STATUS_AMBIGUOUS, STATUS_TIMEOUT)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"^[\s\(\[\{]*|[\s\)\]\}\.,;:]*$")


@dataclass(frozen=True)
class MmluVerdict:
    status: str
    detail: str
    chosen_letter: str | None
    key_letter: str


def letter_for_index(position: int) -> str:
    """The display letter of the ``position``-th (0-based) *displayed* option."""
    if not 0 <= position < MMLU_REQUIRED_OPTIONS:
        raise ValueError(f"option position {position} outside 0..{MMLU_REQUIRED_OPTIONS - 1}")
    return MC_LETTERS[position]


def normalise_option_text(text: str) -> str:
    """NFKC, strip wrapping brackets/punctuation, collapse whitespace, casefold."""
    out = unicodedata.normalize("NFKC", str(text))
    out = _WS.sub(" ", out).strip()
    out = _PUNCT.sub("", out)
    return out.casefold()


def canonical_class(answer: str | None) -> str | None:
    """Gold-independent voting class: the chosen option letter, or ``None``.

    Never consults the key or the option texts, so two agents that pick the same option
    share a class regardless of correctness (spec §5.3 outcome-blind grouping).
    """
    if answer is None or not str(answer).strip():
        return None
    letter = hle_mc_key(str(answer))
    if letter is not None and letter in MC_LETTERS[:MMLU_REQUIRED_OPTIONS]:
        return letter
    return None


def grade(
    answer: str | None,
    key_letter: str,
    *,
    options: Sequence[str] | None = None,
) -> MmluVerdict:
    """Grade ``answer`` against the permuted ``key_letter`` (evaluator only)."""
    if key_letter not in MC_LETTERS[:MMLU_REQUIRED_OPTIONS]:
        raise ValueError(f"key_letter {key_letter!r} is not one of {MC_LETTERS[:MMLU_REQUIRED_OPTIONS]}")
    if answer is None or not str(answer).strip():
        return MmluVerdict(STATUS_INCORRECT, "empty answer", None, key_letter)
    letter = canonical_class(answer)
    if letter is None and options:
        target = normalise_option_text(answer)
        matches = [i for i, option in enumerate(options) if normalise_option_text(option) == target]
        if len(matches) == 1:
            letter = letter_for_index(matches[0])
    if letter is None:
        return MmluVerdict(
            STATUS_AMBIGUOUS,
            "final answer is neither an A-J option id nor an exact option text",
            None,
            key_letter,
        )
    if letter == key_letter:
        return MmluVerdict(STATUS_CORRECT, "exact permuted option id", letter, key_letter)
    return MmluVerdict(STATUS_INCORRECT, "different option id", letter, key_letter)


__all__ = [
    "MmluVerdict",
    "STATUSES",
    "STATUS_AMBIGUOUS",
    "STATUS_CORRECT",
    "STATUS_INCORRECT",
    "STATUS_TIMEOUT",
    "canonical_class",
    "grade",
    "letter_for_index",
    "normalise_option_text",
]
