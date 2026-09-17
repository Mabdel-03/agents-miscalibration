"""Private full EvalPlus grading: reference passes, mutants fail, invalid output is not an evaluator error."""
from __future__ import annotations

import pytest

from agent_allocation.contracts import GradeStatus, ParseStatus
from agent_allocation.grading.code.private import GRADER_VERSION, grade_code

from .conftest import make_candidate, reference_program

pytestmark = [pytest.mark.sandbox, pytest.mark.needs_sources]


def test_reference_program_is_graded_correct(small_inventory, run_paths):
    chosen, plan = small_inventory
    item = chosen["Mbpp/3"]
    grade = grade_code(make_candidate(item.public.item_id, reference_program(plan, item.public.item_id)),
                       item.private, run_paths)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is True
    assert grade.operational_success is True and grade.grader_version == GRADER_VERSION
    assert grade.private_eval_hash == item.private.private_hash and grade.environment_digest


def test_off_by_one_mutant_fails_private_tests(small_inventory, run_paths):
    chosen, _ = small_inventory
    item = chosen["Mbpp/3"]
    mutant = ("import math\ndef is_not_prime(n):\n    if n < 2:\n        return True\n"
              "    for i in range(2, int(math.sqrt(n))):\n        if n % i == 0:\n            return True\n"
              "    return False\n")
    grade = grade_code(make_candidate(item.public.item_id, mutant), item.private, run_paths)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is False


def test_invalid_output_is_operational_failure_not_outage(small_inventory, run_paths):
    chosen, _ = small_inventory
    item = chosen["Mbpp/3"]
    grade = grade_code(make_candidate(item.public.item_id, None, status=ParseStatus.MALFORMED), item.private,
                       run_paths)
    assert grade.status is GradeStatus.INVALID_OUTPUT
    assert grade.semantic_correct is None and grade.operational_success is False
