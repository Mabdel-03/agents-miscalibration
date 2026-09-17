"""Strict tag extraction, one-fence strip and last ``\\boxed{}`` (vendored for agent 06; see PROVENANCE.json).

Origins at ``agents_scaling@e4b4037``:

* ``strict_single_pair`` -- ``src/agents_scaling/ard/report.py:147-173`` (``_extract``): text only, no token ids.
  Change: an empty inner element is reported as ``TAG_EMPTY`` with ``content=None`` (the origin returned ``""``).
* ``strip_one_fence`` -- ``src/agents_scaling/study/parse/candidate.py:127`` with the fence regexes at ``:72-73``.
  Change: the opening fence's info string must be in ``allowed_info`` (case-insensitive) when that is given.
* ``extract_boxed`` -- ``src/agents_scaling/ard/graders/_boxed.py`` verbatim.
"""
from __future__ import annotations

import re
from typing import Collection

TAG_OK = "ok"
TAG_MISSING = "missing"
TAG_MALFORMED = "malformed"
TAG_EMPTY = "empty"

_OPEN_FENCE = re.compile(r"^```[ \t]*([A-Za-z0-9_+.#-]*)[ \t]*\r?\n")
_CLOSE_FENCE = re.compile(r"[ \t]*```[ \t]*$")


def strict_single_pair(text: str, open_tag: str, close_tag: str) -> tuple[str, str | None]:
    """Exactly one ``open_tag`` followed by exactly one ``close_tag``; anything else is not OK.

    Returns ``(status, content)``: ``TAG_MISSING`` when neither delimiter occurs, ``TAG_MALFORMED`` for a duplicate,
    nested, stray, reversed or unclosed delimiter, ``TAG_EMPTY`` for a whitespace-only inner text, and ``TAG_OK`` with
    the inner text stripped of outer whitespace.
    """
    opens = text.count(open_tag)
    closes = text.count(close_tag)
    if opens == 0 and closes == 0:
        return TAG_MISSING, None
    if opens != 1 or closes != 1:
        return TAG_MALFORMED, None
    start = text.index(open_tag)
    stop = text.index(close_tag)
    if stop < start:
        return TAG_MALFORMED, None
    content = text[start + len(open_tag):stop]
    if not content.strip():
        return TAG_EMPTY, None
    return TAG_OK, content.strip()


def strip_one_fence(text: str, allowed_info: Collection[str] | None = None) -> str:
    """Strip surrounding whitespace and at most one enclosing Markdown fence.

    The fence is removed only when an opening fence line starts the text and a closing ````` ``` ````` ends it; the
    inner text is stripped once more. A second fence inside is left in place.
    """
    if not isinstance(text, str):
        raise TypeError("strip_one_fence expects str")
    stripped = text.strip()
    opening = _OPEN_FENCE.match(stripped)
    if opening is None:
        return stripped
    if allowed_info is not None and opening.group(1).lower() not in {a.lower() for a in allowed_info}:
        return stripped
    rest = stripped[opening.end():]
    closing = _CLOSE_FENCE.search(rest)
    if closing is None:
        return stripped
    return rest[: closing.start()].strip()


def extract_boxed(text: str) -> str | None:
    """Return the content of the last ``\\boxed{...}``, brace-balanced, or ``None``."""
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
                return text[i + 1: j].strip()
    return None
