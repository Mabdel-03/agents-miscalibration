"""Gold-free answer pre-treatment and multiple-choice letter keys (vendored for agent 06; see PROVENANCE.json).

Origins at ``agents_scaling@e4b4037``:

* ``pre`` / ``strip_one_wrapper`` -- ``src/agents_scaling/study/selection/normalize.py`` (``_pre``,
  ``_strip_one_wrapper``), byte-for-byte behaviour.
* ``mc_letter_key`` -- ``src/agents_scaling/nextdyn/graders/mc.py`` (``mc_key`` / ``_patterns``). Change: the legal
  alphabet is an explicit letter set taken from ``PublicTask.allowed_choices`` instead of ``n_options``, and a letter
  outside that set yields ``None``.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Collection

_QUOTES = "\"'`“”‘’«»"
_ANSWER_PREFIX = re.compile(
    r"^(?:(?:the|my|final|correct)\s+)*(?:answer|option|choice)(?:\s+is)?\s*(?:[:=]\s*)?",
    re.IGNORECASE,
)
_BOXED = re.compile(r"^\\boxed\{(.*)\}$", re.DOTALL)
_DOLLAR = re.compile(r"^\$\$?(.*?)\$\$?$", re.DOTALL)
_MC_WHOLE = re.compile(r"^\(?([A-Za-z])\)?[.:)]?$")
_MC_BOLD = re.compile(r"^\*\*\(?([A-Za-z])\)?[.:)]?\*\*$")


def strip_one_wrapper(text: str) -> str:
    """Strip at most one wrapper of each kind, in the order ``$..$``/``$$..$$`` -> ``\\boxed{..}`` -> quotes."""
    match = _DOLLAR.match(text)
    if match and len(text) >= 2:
        text = match.group(1).strip()
    match = _BOXED.match(text)
    if match:
        text = match.group(1).strip()
    if len(text) >= 2 and text[0] in _QUOTES and text[-1] in _QUOTES:
        text = text[1:-1].strip()
    return text


def pre(text: str) -> str:
    """NFKC -> strip -> drop one leading answer prefix -> strip one wrapper -> strip."""
    if not isinstance(text, str):
        raise TypeError("final_answer must be str")
    out = unicodedata.normalize("NFKC", text).strip()
    out = _ANSWER_PREFIX.sub("", out, count=1).strip()
    out = strip_one_wrapper(out)
    return out.strip()


def mc_letter_key(answer: str, letters: Collection[str]) -> str | None:
    """The unambiguous option letter of ``answer`` when it is one of ``letters`` (uppercase), else ``None``.

    Accepted forms (after :func:`pre`): ``B``, ``(B)``, ``B.``, ``B)``, ``B:``, ``**B**`` and the ``Answer: B`` /
    ``The answer is (B)`` / ``Option B`` prefixes, case-insensitive, one trailing ``.`` tolerated.
    """
    text = pre(answer)
    if text.endswith("."):
        text = text[:-1].strip()
    text = strip_one_wrapper(text)
    for pattern in (_MC_WHOLE, _MC_BOLD):
        match = pattern.match(text)
        if match:
            letter = match.group(1).upper()
            return letter if letter in letters else None
    return None
