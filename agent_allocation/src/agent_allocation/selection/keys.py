"""Gold-blind vote keys: option letters, booleans and exact-normalized mathematical answers (MASTER_SPEC §11.1).

Every function here reads only the candidate's own final answer and the public task. Nothing here sees a gold
answer, a grade or another candidate.

* OPTION (MMLU-Pro, BBH choice subtasks): the option letter, accepted in the vendored ``mc_key`` forms (``B``,
  ``(B)``, ``B.``, ``**B**``, ``Answer: B``...), and only when it is one of ``PublicTask.allowed_choices``. An
  illegal or unreadable letter has no key: it is not a vote (MMLU options keep their original order, so the letter
  *is* the option identity).
* BOOLEAN (BBH boolean_expressions): ``True``/``False`` after the same pre-treatment, case-insensitive.
* MATH: ``math_text`` is what an equivalence check reads -- NFKC, outer whitespace, the content of the *last*
  ``\\boxed{...}`` when one is present, otherwise one outer ``$``/``$$``/``\\(..\\)``/``\\[..\\]`` wrapper removed.
  ``math_exact_key`` is the exact-equality key built from it: ``\\dfrac``/``\\tfrac`` -> ``\\frac``, sizing
  ``\\left``/``\\right`` dropped, LaTeX spacing commands and ``~`` treated as spaces, ``\\%`` -> ``%``, whitespace
  removed except one space between a control word and a following letter (``\\pi r`` stays two tokens), one
  trailing period dropped. Anything subtler is left to the injected, mutual, complete-link equivalence check.
* PYTHON has no vote key: code is never voted on (public-example selection, ``selection.code``).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from agent_allocation._vendor.extraction_tags import extract_boxed
from agent_allocation._vendor.selection_keys import mc_letter_key, pre, strip_one_wrapper
from agent_allocation.contracts import Candidate, OutputKind, ParseStatus, PublicTask

__all__ = [
    "KEYS_VERSION", "VoteKey", "boolean_key", "math_exact_key", "math_text", "option_key", "option_letters",
    "vote_key",
]

KEYS_VERSION = "06-keys-v1"

_CHOICE = re.compile(r"^\(?([A-Za-z])\)?$")
_MATH_WRAPPERS: tuple[tuple[str, str], ...] = (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$"))
_DFRAC = re.compile(r"\\[dt]frac(?![A-Za-z])")
_SIZING = re.compile(r"\\(?:left|right)(?![A-Za-z])")
_SPACING = re.compile(r"\\[,;:! ]")
_CONTROL_WORD_END = re.compile(r"\\[A-Za-z]+$")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class VoteKey:
    """``exact`` groups candidates by identity; ``text`` is what a mathematical equivalence check reads."""
    exact: str
    text: str


def option_letters(task: PublicTask) -> frozenset[str]:
    """The legal option letters of an OPTION task, from ``allowed_choices`` (``"A"`` or ``"(A)"`` forms)."""
    if not task.allowed_choices:
        raise ValueError(f"{task.item_id}: an OPTION task needs PublicTask.allowed_choices to define legal letters")
    letters: set[str] = set()
    for choice in task.allowed_choices:
        match = _CHOICE.match(str(choice).strip())
        if match is None:
            raise ValueError(f"{task.item_id}: allowed choice {choice!r} is not an option letter")
        letters.add(match.group(1).upper())
    return frozenset(letters)


def option_key(final_answer: str, letters: frozenset[str]) -> str | None:
    return mc_letter_key(final_answer, letters)


def boolean_key(final_answer: str) -> str | None:
    text = pre(final_answer).casefold()
    if text.endswith("."):
        text = text[:-1].strip()
    text = strip_one_wrapper(text).strip()
    return {"true": "True", "false": "False"}.get(text)


def _strip_math_wrapper(text: str) -> str:
    for left, right in _MATH_WRAPPERS:
        if len(text) >= len(left) + len(right) and text.startswith(left) and text.endswith(right):
            inner = text[len(left):len(text) - len(right)]
            if left in inner or right in inner:
                continue            # "$a$ and $b$" is two formulas, not one wrapper
            return inner.strip()
    return text


def math_text(final_answer: str) -> str | None:
    text = unicodedata.normalize("NFKC", final_answer).strip()
    boxed = extract_boxed(text) if "\\boxed" in text else None
    text = boxed if boxed is not None else _strip_math_wrapper(text)
    text = text.strip()
    return text or None


def _squeeze_whitespace(text: str) -> str:
    out: list[str] = []
    pos = 0
    for match in _WS.finditer(text):
        out.append(text[pos:match.start()])
        following = text[match.end():match.end() + 1]
        if following.isalpha() and _CONTROL_WORD_END.search(text[:match.start()]):
            out.append(" ")
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def math_exact_key(text: str) -> str | None:
    key = _DFRAC.sub(r"\\frac", text)
    key = _SIZING.sub("", key)
    key = _SPACING.sub(" ", key).replace("~", " ").replace("\\%", "%")
    key = _squeeze_whitespace(key.strip())
    if key.endswith(".") and len(key) > 1:
        key = key[:-1]
    return key or None


def vote_key(candidate: Candidate, task: PublicTask, *, letters: frozenset[str] | None = None) -> VoteKey | None:
    """The candidate's vote key, or ``None`` when it casts no vote (invalid parse, illegal or unreadable answer)."""
    if candidate.parse_status is not ParseStatus.VALID or candidate.final_answer is None:
        return None
    kind = task.output_kind
    if kind is OutputKind.OPTION:
        letter = option_key(candidate.final_answer, letters if letters is not None else option_letters(task))
        return None if letter is None else VoteKey(exact=letter, text=letter)
    if kind is OutputKind.BOOLEAN:
        value = boolean_key(candidate.final_answer)
        return None if value is None else VoteKey(exact=value, text=value)
    if kind is OutputKind.MATH:
        text = math_text(candidate.final_answer)
        exact = None if text is None else math_exact_key(text)
        return None if text is None or exact is None else VoteKey(exact=exact, text=text)
    raise ValueError(f"{task.item_id}: output kind {kind.value!r} has no vote key (code is selected, not voted)")
