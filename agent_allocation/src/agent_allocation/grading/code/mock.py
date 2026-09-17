"""Typed, sandbox-free stand-ins for the code checker and grader, for downstream tests (06, 07, 08, lead offline runs).

Same signatures and record types as ``public.run_public_checks`` / ``private.grade_code``; every record carries a
``mock-`` version string so reports can label stubbed code results. Behaviour is deterministic and gold-blind from the
program text: a program containing ``MOCK_PUBLIC_FAIL`` fails its public cases, ``MOCK_TIMEOUT`` times out, and one
containing ``MOCK_WRONG`` is privately wrong; ``MOCK_OUTAGE`` yields INFRASTRUCTURE_ERROR / EVALUATOR_ERROR. Invalid
candidates follow the real rules (INVALID_CANDIDATE, INVALID_OUTPUT with semantic_correct None).
"""
from __future__ import annotations

from agent_allocation.contracts import (Candidate, CheckStatus, GradeResult, GradeStatus, ParseStatus, PrivateEvalSpec,
                                        PublicCheckResult, PublicTask)
from agent_allocation.grading.code.public import code_sha256, public_inputs_sha256
from agent_allocation.paths import RunPaths

MOCK_CHECKER_VERSION = "mock-public-assert-v1"
MOCK_GRADER_VERSION = "mock-evalplus-v1"
MOCK_ENVIRONMENT = "mock-environment"


def _program(candidate: Candidate) -> str | None:
    text = candidate.final_answer
    return text if candidate.parse_status is ParseStatus.VALID and text and text.strip() else None


def mock_public_check(candidate: Candidate, task: PublicTask) -> PublicCheckResult:
    program = _program(candidate)
    n = len(task.public_examples)
    if n == 0:
        status, outcomes = CheckStatus.INFRASTRUCTURE_ERROR, ()
    elif program is None:
        status, outcomes = CheckStatus.INVALID_CANDIDATE, ("fail",) * n
    elif "MOCK_OUTAGE" in program:
        status, outcomes = CheckStatus.INFRASTRUCTURE_ERROR, ("error",) * n
    elif "MOCK_TIMEOUT" in program:
        status, outcomes = CheckStatus.OK, ("timeout",) * n
    elif "MOCK_PUBLIC_FAIL" in program:
        status, outcomes = CheckStatus.OK, ("fail",) * n
    else:
        status, outcomes = CheckStatus.OK, ("pass",) * n
    return PublicCheckResult(candidate_id=candidate.candidate_id, item_id=task.item_id,
                             code_sha256=code_sha256(program) if program else None, status=status, cases_total=n,
                             cases_passed=sum(o == "pass" for o in outcomes), case_outcomes=tuple(outcomes),
                             checker_version=MOCK_CHECKER_VERSION, public_inputs_sha256=public_inputs_sha256(task),
                             environment_digest=MOCK_ENVIRONMENT, cpu_seconds=0.0)


def mock_grade_code(candidate: Candidate, private: PrivateEvalSpec, paths: RunPaths | None = None) -> GradeResult:
    program = _program(candidate)
    if program is None:
        status, correct, operational, detail = GradeStatus.INVALID_OUTPUT, None, False, "no_program"
    elif "MOCK_OUTAGE" in program:
        status, correct, operational, detail = GradeStatus.EVALUATOR_ERROR, None, True, "mock_outage"
    elif "MOCK_WRONG" in program or "MOCK_TIMEOUT" in program:
        status, correct, operational, detail = GradeStatus.GRADED, False, True, "plus_wrong_output"
    else:
        status, correct, operational, detail = GradeStatus.GRADED, True, True, "pass"
    return GradeResult(candidate_id=candidate.candidate_id, status=status, semantic_correct=correct,
                       operational_success=operational, grader_version=MOCK_GRADER_VERSION,
                       private_eval_hash=private.private_hash, detail_code=detail,
                       environment_digest=MOCK_ENVIRONMENT, cpu_seconds=0.0)
