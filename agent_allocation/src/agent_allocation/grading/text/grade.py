"""Private grading of text candidates (MATH-500, MMLU-Pro, BBH-logic4). Runs only in grader processes.

Status rules (tri-state, ``text-grade-v1``):

* ``Candidate.parse_status`` other than ``VALID``, or an empty final: ``INVALID_OUTPUT``,
  ``semantic_correct=None``, ``operational_success=False``. Not a semantic answer; downstream counts it as not
  delivered-correct.
* A valid final that names no legal answer (MMLU/BBH choice: not exactly ``X``/``(X)`` for a listed option;
  boolean: not an explicit True/False; MATH: no mathematical object parses, e.g. malformed LaTeX):
  ``INVALID_OUTPUT`` as above, with a ``detail_code`` saying why.
* A gradable final: ``GRADED`` with ``semantic_correct`` True/False and ``operational_success=True``.
  MATH uses ``math_equiv.verify_gold`` (Math-Verify ``verify(gold, prediction)``); options and booleans compare
  the ``normalize`` identity with the gold exactly.
* Evaluator exception, timeout, outage, or unusable gold: ``EVALUATOR_ERROR``, ``semantic_correct=None`` (never
  False), ``operational_success=True`` (the model delivered a well-formed final; the failure is ours). Errors that
  Math-Verify itself absorbs inside one comparison strategy are part of its pinned design and read as not equal
  (see ``math_equiv``).
* MATH grading is SIGALRM-bounded in-process on a main thread and runs in an isolated, killable child process
  otherwise (see ``math_equiv``).

Wiring errors (candidate and spec for different items, non-text eval kinds, output-kind mismatch) raise.

Legal answers come from ``PrivateEvalSpec.allowed_choices`` (contract v2): the item's option letters, or
``("True", "False")`` for boolean items (an empty tuple on a boolean item means that implied pair). An option spec
without letters, a boolean spec with a different set, or a gold outside the legal set is unusable:
``EVALUATOR_ERROR`` rather than a guess.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache

from agent_allocation.contracts import (
    Candidate,
    EvalKind,
    GradeResult,
    GradeStatus,
    OutputKind,
    ParseStatus,
    PrivateEvalSpec,
)
from agent_allocation.grading.text import math_equiv
from agent_allocation.grading.text.normalize import NORMALIZER_VERSION, boolean_identity, option_identity

GRADE_RULES_VERSION = "text-grade-v1"
BOOLEAN_CHOICES = ("True", "False")
BOOLEAN_SUBTASK = "boolean_expressions"
_TEXT_KINDS = {EvalKind.MATH_VERIFY: OutputKind.MATH, EvalKind.OPTION_EXACT: OutputKind.OPTION}
_LETTER = re.compile(r"[A-Z]")


@lru_cache(maxsize=1)
def grader_version() -> str:
    versions = math_equiv.dependency_versions()
    return (f"{GRADE_RULES_VERSION}+{NORMALIZER_VERSION}+{math_equiv.MATH_PARSE_VERSION}"
            f"+math-verify=={versions['math-verify']}")


def _expected_output_kind(private: PrivateEvalSpec) -> OutputKind:
    if private.eval_kind is EvalKind.TASK_EXACT:
        return OutputKind.BOOLEAN if private.subtask == BOOLEAN_SUBTASK else OutputKind.OPTION
    if private.eval_kind in _TEXT_KINDS:
        return _TEXT_KINDS[private.eval_kind]
    raise ValueError(f"{private.item_id}: grade_text does not grade eval kind {private.eval_kind.value!r}")


def grade_text(candidate: Candidate, private: PrivateEvalSpec, *,
               timeout_s: float = math_equiv.DEFAULT_TIMEOUT_S) -> GradeResult:
    started = time.process_time()
    if candidate.item_id != private.item_id:
        raise ValueError(f"candidate {candidate.candidate_id} is for {candidate.item_id}, spec for {private.item_id}")
    expected_kind = _expected_output_kind(private)
    if candidate.output_kind is not expected_kind:
        raise ValueError(f"{candidate.candidate_id}: output kind {candidate.output_kind.value} != "
                         f"{expected_kind.value} for {private.item_id}")

    def result(status: GradeStatus, semantic: bool | None, operational: bool | None, detail: str) -> GradeResult:
        return GradeResult(candidate_id=candidate.candidate_id, status=status, semantic_correct=semantic,
                           operational_success=operational, grader_version=grader_version(),
                           private_eval_hash=private.private_hash, detail_code=detail,
                           environment_digest=math_equiv.environment_digest(),
                           cpu_seconds=round(time.process_time() - started, 6))

    if candidate.parse_status is not ParseStatus.VALID:
        return result(GradeStatus.INVALID_OUTPUT, None, False, f"parse_status:{candidate.parse_status.value}")
    final = candidate.final_answer
    if final is None or not final.strip():
        return result(GradeStatus.INVALID_OUTPUT, None, False, "empty_final_answer")
    gold = private.gold_answer
    if gold is None or not gold.strip():
        return result(GradeStatus.EVALUATOR_ERROR, None, True, "gold_missing")

    if expected_kind is OutputKind.MATH:
        verdict = math_equiv.verify_gold(gold, final, timeout_s=timeout_s)
        if verdict.correct is not None:
            return result(GradeStatus.GRADED, verdict.correct, True, verdict.code)
        if verdict.code == "candidate_unparseable":
            return result(GradeStatus.INVALID_OUTPUT, None, False, verdict.code)
        return result(GradeStatus.EVALUATOR_ERROR, None, True, verdict.code)

    legal = tuple(private.allowed_choices)
    if expected_kind is OutputKind.BOOLEAN:
        if legal and legal != BOOLEAN_CHOICES:               # () is implied True/False; anything else conflicts
            return result(GradeStatus.EVALUATOR_ERROR, None, True, "allowed_choices_not_boolean")
        if gold not in BOOLEAN_CHOICES:
            return result(GradeStatus.EVALUATOR_ERROR, None, True, "gold_not_boolean")
        answer = boolean_identity(final)
        if answer is None:
            return result(GradeStatus.INVALID_OUTPUT, None, False, "not_an_explicit_boolean")
        return result(GradeStatus.GRADED, answer == gold, True, "boolean_exact")

    if not legal or any(not _LETTER.fullmatch(choice) for choice in legal):
        return result(GradeStatus.EVALUATOR_ERROR, None, True, "allowed_choices_missing_or_malformed")
    if gold not in legal:
        return result(GradeStatus.EVALUATOR_ERROR, None, True, "gold_not_a_legal_option")
    answer = option_identity(final, legal)
    if answer is None:
        return result(GradeStatus.INVALID_OUTPUT, None, False, "not_a_legal_option")
    return result(GradeStatus.GRADED, answer == gold, True, "option_exact")
