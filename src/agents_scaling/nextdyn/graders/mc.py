"""Parametric multiple choice: one alphabet, one key rule, one canonical class (spec 5.1, 6.5).

ARD36 froze ten options: ``MC_LETTERS[:10]``, ``MMLU_REQUIRED_OPTIONS`` and a regex
``[A-Ja-j]`` in three different modules, and its canonical class was the *displayed letter*,
so the same option under a different permutation was a different voting class and the
tie-break hash moved with the permutation.  Both are fixed here:

* the alphabet is :func:`mc_letters` (``n`` options, ``n <= 26``), so TruthfulQA MC1's 4-13
  option items and MMLU-Pro's 10-option items share one rule;
* the canonical class is :func:`canonical_option_id` -- the **canonical option index**, not
  the displayed letter -- so it is permutation-invariant, two agents that chose the same
  option share a class under different permutations, and
  ``tie_break(source_item_id, canonical_class, seed)`` is invariant too
  (``test_option_shuffle_preserves_canonical_answer_identity``).

This module is **gold-free**: the reference option arrives as an argument, nothing is read
from disk, and its import closure is the standard library.  It is therefore the one grader a
runtime module (and ``families.Renderer``) may import (R7).

Vendored from ``study/selection/normalize.py`` (``_pre``, ``_strip_one_wrapper``,
``hle_mc_key``'s accepted forms) and ``ard/graders/mmlu.py:53-104`` (the exact-option-text
fallback and the explicit ``ambiguous``), generalized from ten options to ``n``.
``tests/nextdyn/test_graders.py::test_mc_key_matches_hle_mc_key_at_ten_options`` is the
parity test the vendoring discipline requires.
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Mapping, Sequence

from agents_scaling.nextdyn.errors import ConfigError
from agents_scaling.nextdyn.graders import (
    STATUS_AMBIGUOUS,
    STATUS_CORRECT,
    STATUS_INCORRECT,
    GradeTarget,
    Verdict,
)

#: The largest alphabet a multiple-choice item may use.
MAX_OPTIONS: Final[int] = 26
ALPHABET: Final[str] = string.ascii_uppercase

#: ``canonical_class`` prefix, so a MC class can never collide with a MATH class string.
CLASS_PREFIX: Final[str] = "opt:"

# -- vendored from study/selection/normalize.py (byte-for-byte behaviour, parity-tested) ----
_QUOTES: Final[str] = "\"'`“”‘’«»"
_ANSWER_PREFIX = re.compile(
    r"^(?:(?:the|my|final|correct)\s+)*(?:answer|option|choice)(?:\s+is)?\s*(?:[:=]\s*)?",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")
_BOXED = re.compile(r"^\\boxed\{(.*)\}$", re.DOTALL)
_DOLLAR = re.compile(r"^\$\$?(.*?)\$\$?$", re.DOTALL)
_PUNCT = re.compile(r"^[\s\(\[\{]*|[\s\)\]\}\.,;:]*$")


def _strip_one_wrapper(text: str) -> str:
    """Strip at most one wrapper of each kind: ``$..$`` then ``\\boxed{..}`` then quotes."""
    match = _DOLLAR.match(text)
    if match and len(text) >= 2:
        text = match.group(1).strip()
    match = _BOXED.match(text)
    if match:
        text = match.group(1).strip()
    if len(text) >= 2 and text[0] in _QUOTES and text[-1] in _QUOTES:
        text = text[1:-1].strip()
    return text


def _pre(text: str) -> str:
    """NFKC -> strip -> drop one leading answer prefix -> strip one wrapper -> strip."""
    if not isinstance(text, str):
        raise ConfigError("a multiple-choice answer must be a string")
    out = unicodedata.normalize("NFKC", text).strip()
    out = _ANSWER_PREFIX.sub("", out, count=1).strip()
    return _strip_one_wrapper(out).strip()


# --------------------------------------------------------------------------- the alphabet


def mc_letters(n_options: int) -> str:
    """``"A".."<n-th letter>"``: the displayed alphabet of an ``n``-option item."""
    if isinstance(n_options, bool) or not isinstance(n_options, int) or not 1 <= n_options <= MAX_OPTIONS:
        raise ConfigError(f"a multiple-choice item needs 1..{MAX_OPTIONS} options, got {n_options!r}")
    return ALPHABET[:n_options]


@lru_cache(maxsize=MAX_OPTIONS + 1)
def _patterns(n_options: int) -> tuple[re.Pattern[str], re.Pattern[str]]:
    letters = mc_letters(n_options)
    body = f"([{letters}{letters.lower()}])"
    return (re.compile(r"^\(?" + body + r"\)?[.:)]?$"),
            re.compile(r"^\*\*\(?" + body + r"\)?[.:)]?\*\*$"))


def mc_key(answer: str | None, n_options: int) -> str | None:
    """The unambiguous displayed letter of ``answer``, or ``None``.

    Accepted forms (after :func:`_pre`): ``B``, ``(B)``, ``B.``, ``B)``, ``B:``, ``**B**`` and
    the ``Answer: B`` / ``The answer is (B)`` / ``Option B`` prefixes, case-insensitive, one
    trailing ``.`` tolerated -- exactly ``hle_mc_key``'s set, with the alphabet parametric.
    """
    if answer is None:
        return None
    text = _pre(str(answer))
    if text.endswith("."):
        text = text[:-1].strip()
    text = _strip_one_wrapper(text)
    for pattern in _patterns(int(n_options)):
        match = pattern.match(text)
        if match:
            return match.group(1).upper()
    return None


def normalise_option_text(text: str) -> str:
    """NFKC, collapse whitespace, strip wrapping brackets/punctuation, casefold."""
    out = unicodedata.normalize("NFKC", str(text))
    out = _WS.sub(" ", out).strip()
    out = _PUNCT.sub("", out)
    return out.casefold()


# --------------------------------------------------------------------------- canonical ids


def canonical_option_id(letter: str, permutation: Sequence[int]) -> int:
    """The **canonical** option index behind a displayed ``letter``.

    ``permutation[displayed position] = canonical option index``.  This value is the
    canonical class: it does not move when the item is re-permuted, so the same option keeps
    one identity across variants (spec 6.5 "gold-independent canonicalizer"; spec 18.3
    ``test_option_shuffle_preserves_canonical_answer_identity``).
    """
    letters = mc_letters(len(permutation))
    position = letters.find(str(letter).upper())
    if position < 0:
        raise ConfigError(f"letter {letter!r} is outside the {len(permutation)}-option alphabet {letters}")
    return int(permutation[position])


def displayed_letter(canonical_index: int, permutation: Sequence[int]) -> str:
    """The displayed letter of a canonical option index (the inverse of
    :func:`canonical_option_id`); this is the *permuted key* and lives only in the protected
    store (spec 5.1: "no option key is exposed")."""
    order = list(int(p) for p in permutation)
    try:
        position = order.index(int(canonical_index))
    except ValueError as exc:
        raise ConfigError(f"canonical option {canonical_index} is outside the permutation") from exc
    return mc_letters(len(order))[position]


def class_of(canonical_index: int | None) -> str | None:
    return None if canonical_index is None else f"{CLASS_PREFIX}{int(canonical_index)}"


def class_index(canonical_class: str) -> int:
    if not str(canonical_class).startswith(CLASS_PREFIX):
        raise ConfigError(f"{canonical_class!r} is not a multiple-choice canonical class")
    return int(str(canonical_class)[len(CLASS_PREFIX):])


# --------------------------------------------------------------------------- grading


@dataclass(frozen=True)
class McVerdict:
    """The multiple-choice verdict before it is widened to a :class:`Verdict`."""

    status: str
    detail: str
    chosen_letter: str | None
    chosen_canonical: int | None


def resolve_letter(answer: str | None, displayed_options: Sequence[str]) -> tuple[str | None, str]:
    """``(letter, detail)``: the option id of ``answer``, with the exact-option-text fallback.

    A non-letter answer that matches exactly one displayed option text resolves to that
    option; one that matches none (or several) stays ``None`` and the caller reports
    ``ambiguous`` -- never a silent ``incorrect`` (spec 4.6).
    """
    n_options = len(displayed_options)
    letter = mc_key(answer, n_options)
    if letter is not None:
        return letter, "option id"
    if answer is None or not str(answer).strip():
        return None, "empty answer"
    target = normalise_option_text(answer)
    matches = [i for i, option in enumerate(displayed_options)
               if normalise_option_text(option) == target]
    if len(matches) == 1:
        return mc_letters(n_options)[matches[0]], "exact option text"
    if len(matches) > 1:
        return None, "answer text matches several displayed options"
    return None, "final answer is neither an option id nor an exact option text"


def grade_mc(answer: str | None, displayed_options: Sequence[str], permutation: Sequence[int],
             gold_canonical_index: int) -> McVerdict:
    """Grade ``answer`` against the **canonical** gold option index."""
    if len(displayed_options) != len(permutation):
        raise ConfigError(f"{len(displayed_options)} displayed options but {len(permutation)} "
                          "permutation entries")
    if int(gold_canonical_index) not in set(int(p) for p in permutation):
        raise ConfigError(f"gold option {gold_canonical_index} is outside the item's options")
    if answer is None or not str(answer).strip():
        return McVerdict(STATUS_INCORRECT, "empty answer", None, None)
    letter, detail = resolve_letter(answer, displayed_options)
    if letter is None:
        return McVerdict(STATUS_AMBIGUOUS, detail, None, None)
    chosen = canonical_option_id(letter, permutation)
    if chosen == int(gold_canonical_index):
        return McVerdict(STATUS_CORRECT, detail, letter, chosen)
    return McVerdict(STATUS_INCORRECT, f"different option ({detail})", letter, chosen)


# --------------------------------------------------------------------------- family interface


def canonical_class(answer: str | None, target: GradeTarget) -> str | None:
    """Gold-independent canonical class: ``"opt:<canonical option index>"`` or ``None``."""
    if answer is None or not str(answer).strip():
        return None
    letter, _ = resolve_letter(answer, target.options)
    if letter is None:
        return None
    return class_of(canonical_option_id(letter, target.permutation))


def grade(answer: str | None, gold: Mapping[str, object], target: GradeTarget) -> Verdict:
    """The uniform family-grader entry point (``FamilySpec.grader``).

    ``gold["canonical_index"]`` is the canonical option index of the reference answer; the
    displayed key letter is never stored in the public table and never passed here.
    """
    index = gold.get("canonical_index")
    if not isinstance(index, int) or isinstance(index, bool):
        raise ConfigError(f"{target.source_item_id}: multiple-choice gold needs an integer "
                          "'canonical_index'")
    verdict = grade_mc(answer, target.options, target.permutation, index)
    return Verdict(status=verdict.status, detail=verdict.detail,
                   extracted=verdict.chosen_letter,
                   canonical_class=class_of(verdict.chosen_canonical))


__all__ = [
    "ALPHABET", "CLASS_PREFIX", "MAX_OPTIONS", "McVerdict", "canonical_class",
    "canonical_option_id", "class_index", "class_of", "displayed_letter", "grade", "grade_mc",
    "mc_key", "mc_letters", "normalise_option_text", "resolve_letter",
]
