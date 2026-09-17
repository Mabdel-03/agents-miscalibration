"""Pinned Math-Verify evaluation: gold grading (one direction, documented order) and the gold-blind symmetric
equivalence helper exported to agent 06 for MATH voting classes.

Parse rule ``math-parse-v3`` (applied identically to gold answers and candidate finals; gold-blind):

1. strip outer whitespace; remove one enclosing markdown bold pair ``**...**``; drop one trailing ``.``;
2. if the text carries a ``\\boxed`` expression, an unescaped ``$``, or starts with ``\\(`` / ``\\[``, parse it as
   written first (Math-Verify extracts the delimited math);
3. otherwise, or if that yields nothing, parse ``\\boxed{<text>}``, so a bare expression such as ``2\\sqrt{2}`` or
   ``x^2 + 1`` is read whole (parsing bare text as written extracts fragments like ``2`` or ``1``).

Parsing uses ``math_verify.parse`` with its default extraction targets (LaTeX, then plain expression),
``fallback_mode="no_fallback"`` and ``raise_on_error=True``: only successfully parsed mathematical objects count, so
malformed LaTeX (``\\frac{1}{``) is *unparseable* instead of being compared as a raw string (v2 used Math-Verify's
default ``first_match`` string fallback; on the pinned MATH-500 golds the two modes give identical verdicts on all
self-checks and on 11,175 gold pairs). Prose finals without delimiters ("The answer is 5") violate the output
requirement and are not rescued: they parse as a product of symbols and grade as not equal.

Comparison: ``math_verify.verify(gold, prediction)`` exactly in that argument order (Math-Verify documents that
``verify`` is not symmetric), ``raise_on_error=True``, default float rounding/precision/strictness.

Unknowns: a failure of ``parse``/``verify`` themselves (exception or timeout) is unknown, never a mismatch. Known
limitation of the pinned evaluator: inside ``verify`` each comparison strategy (numeric, symbolic, set, relational)
catches its own sympy errors and moves on, returning False when no strategy proves equality. Those internal errors
are part of Math-Verify's documented design and cannot be told apart from a genuine inequality without patching the
evaluator, so they surface as ``math_verify_not_equal``.

Time bound: one SIGALRM wall-clock limit (``_limits.wall_clock_limit``) covers every parse and comparison of a
call. Math-Verify's own per-operation alarms are disabled so they cannot re-arm or cancel that limit. On the main
thread of a (worker) process the work runs in-process. Off the main thread, where SIGALRM is unavailable, the call
is delegated to a persistent isolated child process (``_isolated``) that applies the same limit on its main thread
and is hard-killed if it overruns ``timeout_s + grace``.

No evaluator state is cached across calls: ``math_verify.parser.parse`` / ``math_verify.grader.verify`` are looked
up on every call (only a one-time warm-up flag persists), so results never depend on call or test order.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import Any

from agent_allocation.grading.text._limits import (
    EvaluatorTimeout,
    EvaluatorUnavailable,
    signal_bounds_available,
    wall_clock_limit,
)

MATH_PARSE_VERSION = "math-parse-v3"
DEFAULT_TIMEOUT_S = 5.0
_DISTRIBUTIONS = ("math-verify", "latex2sympy2_extended", "sympy", "antlr4-python3-runtime")
_WARMED = False


@lru_cache(maxsize=1)
def _dependency_versions() -> tuple[tuple[str, str], ...]:
    versions = {"python": sys.version.split()[0]}
    for name in _DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"
    return tuple(sorted(versions.items()))


def dependency_versions() -> dict[str, str]:
    """Interpreter and evaluator distribution versions (read once per process; they cannot change in-process)."""
    return dict(_dependency_versions())


@lru_cache(maxsize=1)
def environment_digest() -> str:
    return hashlib.sha256(json.dumps(dependency_versions(), sort_keys=True).encode()).hexdigest()


def _math_verify() -> tuple[Any, Any]:
    """The current ``parse`` / ``verify`` callables; imports and warms Math-Verify once, outside any alarm window.

    Called before a wall-clock limit is armed, so a short limit never interrupts a module import or the lazy
    construction of the ANTLR parser. The callables are looked up afresh on every call and never cached.
    """
    global _WARMED
    import math_verify.grader as mv_grader
    import math_verify.parser as mv_parser

    # We pass timeout None (our single SIGALRM limit replaces theirs); silence their one-time warning about it.
    mv_parser.TIMEOUT_WARNING_SHOWN = True
    mv_grader.TIMEOUT_WARNING_SHOWN = True
    if not _WARMED:
        warm = mv_parser.parse("\\boxed{\\frac{1}{2}}", fallback_mode="no_fallback", parsing_timeout=None,
                               raise_on_error=True)
        mv_grader.verify(warm, warm, timeout_seconds=None, raise_on_error=True)
        _WARMED = True
    return mv_parser.parse, mv_grader.verify


def _check_timeout(timeout_s: float) -> float:
    value = float(timeout_s)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"timeout_s must be a finite positive number, got {timeout_s!r}")
    return value


_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def _documents(answer: str) -> list[str]:
    text = answer.strip()
    if len(text) > 4 and text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()
    if text.endswith(".") and len(text) > 1:
        text = text[:-1].rstrip()
    if not text:
        return []
    documents = []
    if "\\boxed" in text or _UNESCAPED_DOLLAR.search(text) or text.startswith(("\\(", "\\[")):
        documents.append(text)
    documents.append("\\boxed{" + text + "}")
    return documents


def parse_answer(answer: str) -> list[Any]:
    """Math-Verify parse of one answer under ``math-parse-v3``; must run inside a wall-clock limit.

    Returns only parsed mathematical objects; ``[]`` means unparseable.
    """
    parse, _ = _math_verify()
    for document in _documents(answer):
        parsed = parse(document, fallback_mode="no_fallback", parsing_timeout=None, raise_on_error=True)
        objects = [value for value in parsed or () if not isinstance(value, str)]
        if objects:
            return objects
    return []


def _verify(gold: list[Any], prediction: list[Any]) -> bool:
    _, verify = _math_verify()
    return bool(verify(gold, prediction, timeout_seconds=None, raise_on_error=True))


@dataclass(frozen=True)
class MathVerdict:
    """``correct`` is tri-state; ``code`` names the outcome for ``GradeResult.detail_code``."""
    correct: bool | None
    code: str


def verify_gold(gold_answer: str, candidate_answer: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> MathVerdict:
    """Grade ``candidate_answer`` against ``gold_answer`` with ``verify(gold, prediction)``.

    Codes: ``math_verify_equal`` / ``math_verify_not_equal`` (known), ``candidate_unparseable`` (no mathematical
    object could be parsed from the candidate final: an output problem, not a wrong answer), ``gold_unparseable``,
    ``timeout``, ``evaluator_exception:<Type>``, ``evaluator_unavailable`` (unknown). Raises ``ValueError`` only for
    an invalid ``timeout_s``.
    """
    timeout_s = _check_timeout(timeout_s)
    if not signal_bounds_available():
        from agent_allocation.grading.text._isolated import shared_evaluator
        try:
            verdict = shared_evaluator().call("verify_gold", (gold_answer, candidate_answer),
                                              {"timeout_s": timeout_s}, timeout_s)
        except EvaluatorTimeout:
            return MathVerdict(None, "timeout")
        except Exception:  # noqa: BLE001 - EvaluatorUnavailable or any start/transport failure: an outage
            return MathVerdict(None, "evaluator_unavailable")
        return verdict if isinstance(verdict, MathVerdict) else MathVerdict(None, "evaluator_unavailable")
    try:
        _math_verify()
    except Exception as exc:  # noqa: BLE001 - an unimportable evaluator is an outage, not a wrong answer
        return MathVerdict(None, f"evaluator_exception:{type(exc).__name__}")
    try:
        with wall_clock_limit(timeout_s):
            gold = parse_answer(gold_answer)
            if not gold:
                return MathVerdict(None, "gold_unparseable")
            prediction = parse_answer(candidate_answer)
            if not prediction:
                return MathVerdict(None, "candidate_unparseable")
            equal = _verify(gold, prediction)
    except EvaluatorTimeout:
        return MathVerdict(None, "timeout")
    except Exception as exc:  # noqa: BLE001 - every evaluator failure is an explicit unknown
        return MathVerdict(None, f"evaluator_exception:{type(exc).__name__}")
    return MathVerdict(True, "math_verify_equal") if equal else MathVerdict(False, "math_verify_not_equal")


def math_equivalent(a: str, b: str, *, timeout_s: float = 5.0) -> bool | None:
    """Gold-blind symmetric equivalence for voting classes.

    ``True`` only when ``verify(parse(a), parse(b))`` and ``verify(parse(b), parse(a))`` both hold; ``False``
    when both answers parse and at least one direction fails; ``None`` when either answer does not parse, on
    timeout, or on any evaluator exception or outage. Off the main thread it runs in the isolated child process.
    Raises ``ValueError`` only for an invalid ``timeout_s``.
    """
    timeout_s = _check_timeout(timeout_s)
    if not signal_bounds_available():
        from agent_allocation.grading.text._isolated import shared_evaluator
        try:
            value = shared_evaluator().call("math_equivalent", (a, b), {"timeout_s": timeout_s}, timeout_s)
        except BaseException as exc:  # noqa: BLE001 - timeout, outage or start failure: unknown
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return None
        return value if value is None or isinstance(value, bool) else None
    try:
        _math_verify()
    except Exception:  # noqa: BLE001
        return None
    try:
        with wall_clock_limit(timeout_s):
            parsed_a = parse_answer(a)
            parsed_b = parse_answer(b)
            if not parsed_a or not parsed_b:
                return None
            if not _verify(parsed_a, parsed_b):
                return False
            return _verify(parsed_b, parsed_a)
    except EvaluatorTimeout:
        return None
    except Exception:  # noqa: BLE001 - unknown, never a merge or a split
        return None
