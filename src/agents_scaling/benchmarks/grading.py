"""Answer extraction + correctness checking.

MCQ: pull the chosen option letter from free-form text (handles "The answer is B",
"(B)", "B.", a bare "B", etc.). Numeric: extract the last ``\\boxed{...}`` and compare to
the gold value after light normalization.
"""

from __future__ import annotations

import re

from agents_scaling.benchmarks.schema import AnswerType, Question

_LETTER_PATTERNS = [
    re.compile(r"\banswer\s*(?:is|:)?\s*\(?([A-J])\)?\b", re.IGNORECASE),
    re.compile(r"\boption\s*\(?([A-J])\)?\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-J])\)?[\.\):]", re.MULTILINE),
    re.compile(r"\(([A-J])\)"),
]


def extract_letter(text: str, valid_letters: list[str]) -> str | None:
    """Best-effort extraction of an MCQ option letter from free-form text."""
    valid = set(valid_letters)
    # Prefer the most explicit patterns first.
    for pat in _LETTER_PATTERNS:
        for m in pat.finditer(text):
            cand = m.group(1).upper()
            if cand in valid:
                return cand
    # Last resort: a lone capital letter token anywhere.
    for m in re.finditer(r"\b([A-J])\b", text):
        if m.group(1) in valid:
            return m.group(1)
    return None


def extract_boxed(text: str) -> str | None:
    """Return the content of the last ``\\boxed{...}`` (brace-balanced)."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


def _normalize_numeric(s: str) -> str:
    s = s.strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\!", "").replace(r"\,", "").replace(" ", "")
    s = s.replace("\\$", "").replace("$", "").replace("%", "")
    s = s.rstrip(".")
    # strip a wrapping \text{...}
    m = re.fullmatch(r"\\text\{(.*)\}", s)
    if m:
        s = m.group(1)
    return s


def grade(q: Question, answer_text: str) -> bool:
    """Whether ``answer_text`` is correct for question ``q``."""
    if q.answer_type == AnswerType.MCQ:
        chosen = extract_letter(answer_text, q.option_letters)
        return chosen == q.answer_key
    # NUMERIC
    boxed = extract_boxed(answer_text)
    cand = boxed if boxed is not None else answer_text
    return _normalize_numeric(cand) == _normalize_numeric(q.answer_key)


def extract_answer(q: Question, answer_text: str) -> str | None:
    """The agent's chosen answer in canonical form (letter or normalized value)."""
    if q.answer_type == AnswerType.MCQ:
        return extract_letter(answer_text, q.option_letters)
    boxed = extract_boxed(answer_text)
    return _normalize_numeric(boxed) if boxed is not None else None
