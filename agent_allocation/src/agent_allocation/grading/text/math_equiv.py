"""Pinned Math-Verify evaluation: gold grading (one direction, documented order) and the gold-blind symmetric
equivalence helper exported to agent 06 for MATH voting classes.

Parse rule ``math-parse-v1`` (applied identically to gold answers and candidate finals):
strip outer whitespace; if the text already carries a ``\\boxed`` expression or starts with ``$``, ``\\(`` or
``\\[``, parse it as written first; otherwise (or if that yields nothing) parse ``\\boxed{<text>}``. Parsing uses
``math_verify.parse`` with its default extraction targets (LaTeX, then plain expression), ``raise_on_error=True``.

Comparison: ``math_verify.verify(gold, prediction)`` exactly in that argument order (Math-Verify documents that
``verify`` is not symmetric), ``raise_on_error=True``, default float rounding/precision/strictness.

Time bound: one SIGALRM wall-clock limit (``_limits.wall_clock_limit``) covers every parse and comparison of a
call. Math-Verify's own per-operation alarms are disabled so they cannot re-arm or cancel that limit. Calls must
therefore run on the main thread of a (worker) process; elsewhere they raise ``EvaluatorUnavailable`` rather
than running unbounded. Any exception or timeout is reported as unknown, never as a mismatch.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import Any

from agent_allocation.grading.text._limits import EvaluatorTimeout, EvaluatorUnavailable, wall_clock_limit

MATH_PARSE_VERSION = "math-parse-v1"
DEFAULT_TIMEOUT_S = 5.0
_DISTRIBUTIONS = ("math-verify", "latex2sympy2_extended", "sympy", "antlr4-python3-runtime")


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
    """Interpreter and evaluator distribution versions (read once per process)."""
    return dict(_dependency_versions())


@lru_cache(maxsize=1)
def environment_digest() -> str:
    return hashlib.sha256(json.dumps(dependency_versions(), sort_keys=True).encode()).hexdigest()


def _math_verify() -> tuple[Any, Any]:
    import math_verify.grader as mv_grader
    import math_verify.parser as mv_parser

    # We pass timeout None (our single SIGALRM limit replaces theirs); silence their one-time warning about it.
    mv_parser.TIMEOUT_WARNING_SHOWN = True
    mv_grader.TIMEOUT_WARNING_SHOWN = True
    return mv_parser.parse, mv_grader.verify


def _documents(answer: str) -> list[str]:
    text = answer.strip()
    if not text:
        return []
    documents = []
    if "\\boxed" in text or text.startswith(("$", "\\(", "\\[")):
        documents.append(text)
    documents.append("\\boxed{" + text + "}")
    return documents


def parse_answer(answer: str) -> list[Any]:
    """Math-Verify parse of one answer under ``math-parse-v1``; must run inside a wall-clock limit."""
    parse, _ = _math_verify()
    for document in _documents(answer):
        parsed = parse(document, parsing_timeout=None, raise_on_error=True)
        if parsed:
            return list(parsed)
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

    Codes: ``math_verify_equal`` / ``math_verify_not_equal`` (known), ``candidate_unparseable`` (no expression
    could be extracted from the candidate final: an output problem, not a wrong answer), ``gold_unparseable``,
    ``timeout``, ``evaluator_exception:<Type>`` (unknown).
    """
    try:
        with wall_clock_limit(timeout_s):
            gold = parse_answer(gold_answer)
            if not gold:
                return MathVerdict(None, "gold_unparseable")
            prediction = parse_answer(candidate_answer)
            if not prediction:
                return MathVerdict(None, "candidate_unparseable")
            equal = _verify(gold, prediction)
    except EvaluatorUnavailable:
        raise
    except EvaluatorTimeout:
        return MathVerdict(None, "timeout")
    except Exception as exc:  # noqa: BLE001 - every evaluator failure is an explicit unknown
        return MathVerdict(None, f"evaluator_exception:{type(exc).__name__}")
    return MathVerdict(True, "math_verify_equal") if equal else MathVerdict(False, "math_verify_not_equal")


def math_equivalent(a: str, b: str, *, timeout_s: float = 5.0) -> bool | None:
    """Gold-blind symmetric equivalence for voting classes.

    ``True`` only when ``verify(parse(a), parse(b))`` and ``verify(parse(b), parse(a))`` both hold; ``False``
    when both answers parse and at least one direction fails; ``None`` when either answer does not parse, on
    timeout, or on any evaluator exception. Raises ``EvaluatorUnavailable`` off the main thread.
    """
    try:
        with wall_clock_limit(timeout_s):
            parsed_a = parse_answer(a)
            parsed_b = parse_answer(b)
            if not parsed_a or not parsed_b:
                return None
            if not _verify(parsed_a, parsed_b):
                return False
            return _verify(parsed_b, parsed_a)
    except EvaluatorUnavailable:
        raise
    except EvaluatorTimeout:
        return None
    except Exception:  # noqa: BLE001 - unknown, never a merge or a split
        return None
