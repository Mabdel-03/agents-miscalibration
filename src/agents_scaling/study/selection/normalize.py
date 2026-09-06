"""Outcome-blind vote keys: MC letters, exact-answer normalization, code identity (WP3).

Spec §5.3: "For multiple-choice answers, VOTE is plurality over the normalized choice
labels ... For short exact-answer tasks, voting is permitted only through an
outcome-blind, task-valid normalization rule fixed on development data.  Do not merge
answers by reference correctness."  §4.4: "use Python AST canonicalization that strips
source positions/comments while preserving identifiers, literals, ordering and executable
semantics; if parsing is unsupported, use exact source identity.  Name the grouping mode
in every record".  Amendment S1 (AST_CANONICAL / EXACT_SOURCE / all-singleton reporting),
amendment E3' + correction P1-5 (``strip_one_fence`` is the one frozen code-string rule
shared with ``evaluation/bcb.py``), correction P2-2 (``hle_mc_key`` only for
``multipleChoice`` and the five non-letter MC answers fall back to ``hle_exact_key``).
Architecture §1.9.

Every function here is pure, deterministic and never sees a reference answer.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from agents_scaling.study import identity
from agents_scaling.study.parse.candidate import strip_one_fence

#: Public grouping modes (types.GROUPING_MODES).
MODE_MC = "mc_letter"
MODE_EXACT = "exact_norm"
MODE_AST = "ast"
MODE_EXACT_SOURCE = "exact_source"
MODE_EMPTY = "empty"

ANSWER_FORMATS: tuple[str, ...] = ("multipleChoice", "exactMatch", "code")

# --------------------------------------------------------------------------- shared pre-treatment

_QUOTES = "\"'`“”‘’«»"
_ANSWER_PREFIX = re.compile(
    r"^(?:(?:the|my|final|correct)\s+)*(?:answer|option|choice)(?:\s+is)?\s*(?:[:=]\s*)?",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")
_BOXED = re.compile(r"^\\boxed\{(.*)\}$", re.DOTALL)
_DOLLAR = re.compile(r"^\$\$?(.*?)\$\$?$", re.DOTALL)
_FRAC = re.compile(r"^\\[dt]?frac\{\s*([+-]?\d+)\s*\}\{\s*(\d+)\s*\}$")
_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_FRACTION = re.compile(r"^([+-]?\d+)\s*/\s*(\d+)$")
_MC_LETTER = r"([A-Ja-j])"
_MC_WHOLE = re.compile(r"^\(?" + _MC_LETTER + r"\)?[.:)]?$")
_MC_BOLD = re.compile(r"^\*\*\(?" + _MC_LETTER + r"\)?[.:)]?\*\*$")


def _strip_one_wrapper(text: str) -> str:
    """Strip at most one wrapper *of each kind*, in the order ``$..$``/``$$..$$`` →
    ``\\boxed{..}`` → a matching pair of quotes (so ``$\\boxed{x}$`` → ``x`` but
    ``$$x$$`` nested twice, ``\\boxed{\\boxed{x}}`` or ``""x""`` keep their inner wrapper)."""
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
    """NFKC → strip → drop one leading answer prefix → strip one wrapper → strip."""
    if not isinstance(text, str):
        raise TypeError("final_answer must be str")
    out = unicodedata.normalize("NFKC", text).strip()
    out = _ANSWER_PREFIX.sub("", out, count=1).strip()
    out = _strip_one_wrapper(out)
    return out.strip()


# --------------------------------------------------------------------------- MC letters


def hle_mc_key(final_answer: str) -> str | None:
    """Unambiguous single choice letter A–J of an MC answer, or ``None`` (§5.3, P2-2).

    Accepted forms (after :func:`_pre`): ``B``, ``(B)``, ``B.``, ``B)``, ``B:``, ``**B**``,
    and the prefixes ``Answer: B`` / ``The answer is (B)`` / ``Option B`` (case-insensitive,
    one trailing ``.`` tolerated).  Anything else — free text, several letters, a non-letter
    answer — returns ``None`` and the caller falls back to :func:`hle_exact_key`.  Only to
    be used when ``answer_format == "multipleChoice"``.
    """
    text = _pre(final_answer)
    if text.endswith("."):
        text = text[:-1].strip()
    text = _strip_one_wrapper(text)
    for pattern in (_MC_WHOLE, _MC_BOLD):
        match = pattern.match(text)
        if match:
            return match.group(1).upper()
    return None


# --------------------------------------------------------------------------- exact answers


def _canonical_number(text: str) -> str | None:
    """Decimal/Fraction canonical form when the *whole* string is a number, else ``None``."""
    compact = text.replace(" ", "")
    match = _FRACTION.match(compact)
    if match:
        numerator, denominator = int(match.group(1)), int(match.group(2))
        if denominator == 0:
            return None
        value = Fraction(numerator, denominator)
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    match = _FRAC.match(compact)
    if match:
        return _canonical_number(f"{match.group(1)}/{match.group(2)}")
    if not _NUMBER.match(compact) or not any(ch.isdigit() for ch in compact):
        return None
    try:
        number = Decimal(compact.replace(",", "").lstrip("+"))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number == number.to_integral_value():
        return str(int(number))
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def hle_exact_key(final_answer: str) -> str:
    """Frozen exact-answer normalization (§5.3; fixed on dev, outcome-blind).

    NFKC → strip → drop one leading ``the answer is``/``answer:`` prefix → strip one
    ``$..$``/``\\boxed{..}``/quote wrapper → casefold → collapse whitespace → drop one
    trailing ``.`` → numeric canonicalization when the whole string is a number (``+``
    dropped, thousands separators removed, trailing zeros removed, ``a/b`` and
    ``\\frac{a}{b}`` reduced) → otherwise the string itself.  Empty input → ``""``.
    """
    text = _pre(final_answer).casefold()
    text = _WS.sub(" ", text).strip()
    if text.endswith("."):
        text = text[:-1].strip()
    number = _canonical_number(text)
    return number if number is not None else text


# --------------------------------------------------------------------------- code


def code_key(final_answer: str) -> tuple[str | None, str]:
    """AST-canonical or exact-source identity of a program (§4.4, S1, E3').

    ``program = strip_one_fence(final_answer)``; empty → ``(None, "empty")``;
    ``ast.parse`` succeeds → ``(sha256(ast.dump(tree, include_attributes=False)), "ast")``
    (positions and comments dropped; identifiers, literals, docstrings and order kept);
    any parse failure (``SyntaxError``, ``ValueError`` for null bytes, recursion/memory
    limits) → ``(sha256(program), "exact_source")``.
    """
    if not isinstance(final_answer, str):
        raise TypeError("final_answer must be str")
    program = strip_one_fence(final_answer)
    if program == "":
        return None, MODE_EMPTY
    try:
        tree = ast.parse(program)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return identity.sha256_hex(program), MODE_EXACT_SOURCE
    return identity.sha256_hex(ast.dump(tree, include_attributes=False)), MODE_AST


# --------------------------------------------------------------------------- dispatch


def vote_key(answer_format: str, final_answer: str) -> tuple[str | None, str]:
    """``(key, grouping_mode)`` for a public candidate under the task's answer format.

    ``multipleChoice`` → ``mc_letter`` when :func:`hle_mc_key` finds a letter, else
    ``exact_norm`` (P2-2); ``exactMatch`` → ``exact_norm``; ``code`` → :func:`code_key`.
    An empty key (empty answer) is returned as ``None`` — such a candidate is structurally
    valid but casts no vote (``selection.vote`` records it as unkeyed).
    """
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f"unknown answer_format {answer_format!r}; expected one of {ANSWER_FORMATS}")
    if answer_format == "code":
        return code_key(final_answer)
    if answer_format == "multipleChoice":
        letter = hle_mc_key(final_answer)
        if letter is not None:
            return letter, MODE_MC
    key = hle_exact_key(final_answer)
    return (key if key != "" else None), MODE_EXACT


__all__ = [
    "ANSWER_FORMATS",
    "MODE_AST",
    "MODE_EMPTY",
    "MODE_EXACT",
    "MODE_EXACT_SOURCE",
    "MODE_MC",
    "code_key",
    "hle_exact_key",
    "hle_mc_key",
    "vote_key",
]
