"""Gold-blind vote keys: option letters, booleans and exact-normalized mathematical answers (MASTER_SPEC §2.2, §11.1).

Every function here reads only the candidate's own final answer and the public task. Nothing here sees a gold
answer, a grade or another candidate.

* OPTION (MMLU-Pro, BBH choice subtasks) and BOOLEAN (BBH boolean_expressions): the vote identity *is* the grader's
  frozen identity, imported from ``grading.text.normalize`` (``option_identity`` / ``boolean_identity``, lead
  decision on 04-R2): one normalization for voting and grading. An option final is a vote only when, after outer
  whitespace, it is exactly ``X`` or ``(X)`` for a letter in ``PublicTask.allowed_choices``; ``B.``, ``**B**`` or
  ``Answer: B`` are not votes and cannot win plurality. A boolean final is a vote only when it is an explicit
  ``true``/``false`` under that module's outer-character rule. That module is pure string handling with no gold.
  MMLU options keep their original order, so the letter is the option identity.
* MATH: ``math_text`` is what an equivalence check reads -- outer whitespace, one enclosing ``**...**`` pair
  removed, then the content of the ``\\boxed{...}`` when the final carries boxed expressions, otherwise one outer
  ``$``/``$$``/``\\(..\\)``/``\\[..\\]`` wrapper removed. A final whose top-level boxed expressions are not all the
  same answer (``\\boxed{2} or \\boxed{3}``) is a hedge: it has no key and casts no vote (the grader marks hedged
  answers wrong, so no class may be represented by one); repeating one answer (``\\boxed{3} ... \\boxed{3}``) is not
  a hedge. ``math_exact_key`` is the exact-equality key built from the text with *value-preserving* rewrites only:
  ``\\dfrac``/``\\tfrac`` -> ``\\frac``, sizing ``\\left``/``\\right`` dropped, LaTeX spacing commands
  (``\\, \\; \\: \\! \\ ``) read as spaces, a whitespace run removed unless it separates two alphanumeric characters
  (``1 2``, ``x y`` and ``\\pi r`` keep one space), one trailing period dropped. No Unicode normalization (``x²`` is
  not ``x2``). Everything else -- ``0.5`` vs ``\\frac{1}{2}``, ``50\\%`` vs ``0.5`` -- is left to the injected mutual,
  complete-link equivalence check.
* PYTHON has no vote key: code is never voted on (public-example selection, ``selection.code``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from agent_allocation.contracts import Candidate, OutputKind, ParseStatus, PublicTask
from agent_allocation.grading.text.normalize import NORMALIZER_VERSION, boolean_identity, option_identity

__all__ = [
    "KEYS_VERSION", "TEXT_NORMALIZER_VERSION", "VoteKey", "boolean_key", "boxed_contents", "math_exact_key",
    "math_text", "option_key", "option_letters", "vote_key",
]

KEYS_VERSION = "06-keys-v4"
#: The grader normalizer whose identities are the option/boolean vote keys.
TEXT_NORMALIZER_VERSION = NORMALIZER_VERSION

_CHOICE = re.compile(r"^\(?([A-Za-z])\)?$")
_MATH_WRAPPERS: tuple[tuple[str, str], ...] = (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$"))
_BOXED_OPEN = re.compile(r"\\boxed\s*\{")
_WHOLE_BOX = re.compile(r"^\\boxed\s*\{")
_DFRAC = re.compile(r"\\[dt]frac(?![A-Za-z])")
_SIZING = re.compile(r"\\(?:left|right)(?![A-Za-z])")
_SPACING = re.compile(r"\\[,;:! ]")
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


def option_key(final_answer: str | None, letters: frozenset[str]) -> str | None:
    """``grading.text.normalize.option_identity`` over the item's legal letters."""
    return option_identity(final_answer, tuple(sorted(letters)))


def boolean_key(final_answer: str | None) -> str | None:
    """``grading.text.normalize.boolean_identity``."""
    return boolean_identity(final_answer)


def _balanced(text: str, open_brace: int) -> int | None:
    """Index just past the brace that closes ``text[open_brace]``, or ``None`` when it never closes."""
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _unwrap_box(content: str) -> str:
    """``\\boxed{\\boxed{x}}`` content -> ``x``: a box that is the whole content is the same expression."""
    while True:
        stripped = content.strip()
        match = _WHOLE_BOX.match(stripped)
        end = None if match is None else _balanced(stripped, match.end() - 1)
        if match is None or end != len(stripped):
            return stripped
        content = stripped[match.end():end - 1]


def boxed_contents(text: str) -> list[str] | None:
    """Contents of the top-level ``\\boxed{...}`` expressions in order; ``None`` when a box never closes."""
    contents: list[str] = []
    position = 0
    while True:
        match = _BOXED_OPEN.search(text, position)
        if match is None:
            return contents
        end = _balanced(text, match.end() - 1)
        if end is None:
            return None
        contents.append(_unwrap_box(text[match.end():end - 1]))
        position = end


def _strip_math_wrapper(text: str) -> str:
    for left, right in _MATH_WRAPPERS:
        if len(text) >= len(left) + len(right) and text.startswith(left) and text.endswith(right):
            inner = text[len(left):len(text) - len(right)]
            if left in inner or right in inner:
                continue            # "$a$ and $b$" is two formulas, not one wrapper
            return inner.strip()
    return text


def math_text(final_answer: str) -> str | None:
    """The answer expression an equivalence check reads, or ``None`` (blank, unclosed box, or a boxed hedge)."""
    text = final_answer.strip()
    if len(text) > 4 and text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()
    if "\\boxed" in text:
        boxes = boxed_contents(text)
        if boxes is None:
            return None
        if boxes:
            distinct = {math_exact_key(box) for box in boxes}
            if len(distinct) != 1 or None in distinct:
                return None         # several different boxed answers (a hedge) or an empty box
            text = boxes[-1]
        else:
            text = _strip_math_wrapper(text)
    else:
        text = _strip_math_wrapper(text)
    text = text.strip()
    return text or None


def _squeeze_whitespace(text: str) -> str:
    """Drop each whitespace run unless it separates two alphanumeric characters (then keep one space)."""
    out: list[str] = []
    pos = 0
    for match in _WS.finditer(text):
        out.append(text[pos:match.start()])
        before = text[match.start() - 1:match.start()]
        after = text[match.end():match.end() + 1]
        if before.isalnum() and after.isalnum():
            out.append(" ")
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def math_exact_key(text: str) -> str | None:
    key = _DFRAC.sub(r"\\frac", text)
    key = _SIZING.sub("", key)
    key = _SPACING.sub(" ", key)
    key = _squeeze_whitespace(key.strip())
    if key.endswith(".") and len(key) > 1:
        key = key[:-1]
    return key or None


def vote_key(candidate: Candidate, task: PublicTask, *, letters: frozenset[str] | None = None) -> VoteKey | None:
    """The candidate's vote key, or ``None`` when it casts no vote (invalid parse, illegal, unreadable or hedged)."""
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
