"""Public-example checks: the prompt's assert only, same checks for every candidate, typed outcomes."""
from __future__ import annotations

import pytest

from agent_allocation.contracts import CheckStatus, ParseStatus
from agent_allocation.grading.code.public import CHECKER_VERSION, run_public_checks

from .conftest import make_candidate, reference_program

pytestmark = [pytest.mark.sandbox, pytest.mark.needs_sources]


def test_reference_passes_its_prompt_assert(small_inventory):
    chosen, plan = small_inventory
    task = chosen["Mbpp/2"].public
    result = run_public_checks(make_candidate(task.item_id, reference_program(plan, task.item_id)), task)
    assert result.status is CheckStatus.OK
    assert (result.cases_total, result.cases_passed, result.case_outcomes) == (1, 1, ("pass",))
    assert result.checker_version == CHECKER_VERSION and result.code_sha256 and result.environment_digest


def test_wrong_program_fails_and_missing_entry_point_errors(small_inventory):
    chosen, _ = small_inventory
    task = chosen["Mbpp/3"].public          # assert is_not_prime(2) == False
    wrong = run_public_checks(make_candidate(task.item_id, "def is_not_prime(n):\n    return True\n"), task)
    assert wrong.status is CheckStatus.OK and wrong.case_outcomes == ("fail",) and wrong.pass_fraction == 0.0
    missing = run_public_checks(make_candidate(task.item_id, "def other(n):\n    return 1\n"), task)
    assert missing.status is CheckStatus.OK and missing.case_outcomes == ("error",)


def test_invalid_candidate_counts_all_cases_failed(small_inventory):
    chosen, _ = small_inventory
    task = chosen["Mbpp/3"].public
    result = run_public_checks(make_candidate(task.item_id, None, status=ParseStatus.MISSING), task)
    assert result.status is CheckStatus.INVALID_CANDIDATE
    assert result.cases_passed == 0 and result.case_outcomes == ("fail",) and result.code_sha256 is None
