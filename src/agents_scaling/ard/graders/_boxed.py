"""Brace-balanced ``\\boxed{...}`` extraction, vendored for the firewall.

This is a verbatim behavioural copy of
``agents_scaling.benchmarks.grading.extract_boxed``.  It is vendored rather than
imported because ``canonical_class`` is runtime-reachable (``ard/vote.py`` and the
BALANCED packet builder call it), and the ARD36 firewall rule is absolute: the import
closure of a runtime module never reaches ``agents_scaling.benchmarks``.

The rule is kept absolute on purpose.  ``benchmarks.grading`` happens to be pure string
handling today, but a per-function whitelist would have to be re-adjudicated on every
future edit to that module, whereas "no benchmarks import in a runtime closure" is
mechanically checkable and cannot rot.

A parity test (``tests/ard/test_wp3_graders.py``) asserts this function and the
benchmarks original agree on a shared corpus, so the copy cannot silently drift.
"""

from __future__ import annotations

__all__ = ["extract_boxed"]


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
                return text[i + 1 : j].strip()
    return None
