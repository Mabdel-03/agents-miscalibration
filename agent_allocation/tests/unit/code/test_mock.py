"""The sandbox-free mock checker/grader produce valid, clearly labelled contract records."""
from __future__ import annotations

from agent_allocation.contracts import (CheckStatus, EvalKind, GradeStatus, OutputKind, ParseStatus, PrivateEvalSpec,
                                        PublicCheckResult, GradeResult, PublicTask, dump_record, load_record)
from agent_allocation.grading.code.mock import mock_grade_code, mock_public_check

from .conftest import make_candidate

TASK = PublicTask(item_id="mbpp_plus/Mbpp_2", benchmark="mbpp_plus", source_revision="r", prompt="p",
                  output_kind=OutputKind.PYTHON, public_examples=("assert f(1) == 2",))
PRIVATE = PrivateEvalSpec(item_id="mbpp_plus/Mbpp_2", benchmark="mbpp_plus", source_revision="r",
                          eval_kind=EvalKind.EVALPLUS_FULL, gold_answer=None, private_record_ref="x", private_hash="h")


def test_mock_records_round_trip_and_follow_the_real_status_rules():
    cases = {"def f(x): return x + 1": (CheckStatus.OK, 1, GradeStatus.GRADED, True),
             "def f(x): return 2  # MOCK_WRONG": (CheckStatus.OK, 1, GradeStatus.GRADED, False),
             "# MOCK_PUBLIC_FAIL MOCK_WRONG": (CheckStatus.OK, 0, GradeStatus.GRADED, False),
             "# MOCK_OUTAGE": (CheckStatus.INFRASTRUCTURE_ERROR, 0, GradeStatus.EVALUATOR_ERROR, None)}
    for program, (check_status, passed, grade_status, correct) in cases.items():
        candidate = make_candidate(TASK.item_id, program)
        check, grade = mock_public_check(candidate, TASK), mock_grade_code(candidate, PRIVATE)
        assert (check.status, check.cases_passed, grade.status, grade.semantic_correct) == \
            (check_status, passed, grade_status, correct)
        assert load_record(PublicCheckResult, dump_record(check)) == check
        assert load_record(GradeResult, dump_record(grade)) == grade
        assert check.checker_version.startswith("mock-") and grade.grader_version.startswith("mock-")
    invalid = make_candidate(TASK.item_id, None, status=ParseStatus.MISSING)
    assert mock_public_check(invalid, TASK).status is CheckStatus.INVALID_CANDIDATE
    assert mock_grade_code(invalid, PRIVATE).status is GradeStatus.INVALID_OUTPUT
