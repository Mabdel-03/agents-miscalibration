"""Evaluator outage vs verdict: injected sandbox faults retry once after a canary, then become unknown."""
from __future__ import annotations

import os
import signal
import time

import pytest

from agent_allocation.contracts import CheckStatus, GradeStatus
from agent_allocation.grading.code import private as private_mod
from agent_allocation.grading.code import public as public_mod
from agent_allocation.sandbox import evalplus_scorer
from agent_allocation.sandbox.sandbox import STARTUP_FAILED, Sandbox, SandboxError

from .conftest import GOOD_PROGRAM, install_synthetic_task, make_candidate

pytestmark = pytest.mark.sandbox


def _flaky(failures: int):
    """A Sandbox whose first ``failures`` constructions fail like a bwrap crash (None = always)."""

    class FlakySandbox(Sandbox):
        remaining = failures
        started = 0

        def __init__(self, *args, **kwargs):
            cls = type(self)
            cls.started += 1
            if cls.remaining is None or cls.remaining > 0:
                if cls.remaining is not None:
                    cls.remaining -= 1
                raise SandboxError("injected: bwrap: setting up uid map: Permission denied", kind=STARTUP_FAILED,
                                   phase="startup", returncode=1)
            super().__init__(*args, **kwargs)

    return FlakySandbox


def _install(monkeypatch, cls):
    monkeypatch.setattr(public_mod, "Sandbox", cls)
    monkeypatch.setattr(evalplus_scorer, "Sandbox", cls)


def test_public_startup_fault_is_retried_once_after_canary(run_paths, monkeypatch):
    public, _ = install_synthetic_task(run_paths)
    flaky = _flaky(1)
    _install(monkeypatch, flaky)
    result = public_mod.run_public_checks(make_candidate(public.item_id, GOOD_PROGRAM), public)
    assert result.status is CheckStatus.OK and result.case_outcomes == ("pass",)
    assert flaky.started == 3                         # failed run, canary, retry


def test_persistent_public_fault_is_infrastructure_error(run_paths, monkeypatch):
    public, _ = install_synthetic_task(run_paths)
    _install(monkeypatch, _flaky(None))
    result = public_mod.run_public_checks(make_candidate(public.item_id, GOOD_PROGRAM), public)
    assert result.status is CheckStatus.INFRASTRUCTURE_ERROR
    assert result.cases_passed == 0 and result.case_outcomes == ("error",)


def test_persistent_private_fault_is_evaluator_error_never_wrong(run_paths, monkeypatch):
    _, private = install_synthetic_task(run_paths)
    _install(monkeypatch, _flaky(None))
    grade = private_mod.grade_code(make_candidate(private.item_id, GOOD_PROGRAM), private, run_paths)
    assert grade.status is GradeStatus.EVALUATOR_ERROR and grade.semantic_correct is None
    assert grade.detail_code == "sandbox_outage_canary_failed"


def test_private_fault_after_healthy_canary_is_retried_to_a_verdict(run_paths, monkeypatch):
    _, private = install_synthetic_task(run_paths)
    flaky = _flaky(1)
    _install(monkeypatch, flaky)
    grade = private_mod.grade_code(make_candidate(private.item_id, GOOD_PROGRAM), private, run_paths)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is True


def test_sandbox_killed_mid_candidate_once_is_retried_after_canary(run_paths, monkeypatch):
    _, private = install_synthetic_task(run_paths)

    class KilledOnce(Sandbox):
        kills = 1

        def invoke(self, value, entry_point="solve", *, phase="candidate"):
            if entry_point == "_invoke_native" and type(self).kills > 0:
                type(self).kills -= 1
                os.killpg(self.process.pid, signal.SIGKILL)          # an external kill, e.g. an OOM killer
                time.sleep(0.2)
            return super().invoke(value, entry_point, phase=phase)

    _install(monkeypatch, KilledOnce)
    canaries = []
    real = private_mod._canary_ok
    monkeypatch.setattr(private_mod, "_canary_ok", lambda p: canaries.append(1) or real(p))
    grade = private_mod.grade_code(make_candidate(private.item_id, GOOD_PROGRAM), private, run_paths)
    assert grade.status is GradeStatus.GRADED and grade.semantic_correct is True and canaries == [1]


def test_trusted_bootstrap_failure_is_an_outage(run_paths, monkeypatch):
    _, private = install_synthetic_task(run_paths)

    class BrokenBootstrap(Sandbox):
        def execute(self, code, *, phase="candidate"):
            if phase == "trusted":
                return {"ok": False, "error": "injected", "_resource_usage": {"sandbox_cpu_seconds": 0.0}}
            return super().execute(code, phase=phase)

    _install(monkeypatch, BrokenBootstrap)
    grade = private_mod.grade_code(make_candidate(private.item_id, GOOD_PROGRAM), private, run_paths)
    assert grade.status is GradeStatus.EVALUATOR_ERROR and grade.semantic_correct is None


def test_missing_or_drifted_private_record_is_evaluator_error(run_paths):
    import dataclasses
    _, private = install_synthetic_task(run_paths)
    drifted = dataclasses.replace(private, private_hash="0" * 64)
    grade = private_mod.grade_code(make_candidate(private.item_id, GOOD_PROGRAM), drifted, run_paths)
    assert grade.status is GradeStatus.EVALUATOR_ERROR and grade.detail_code == "private_record_unavailable"
    missing = dataclasses.replace(private, private_record_ref="private/evalplus_oracles/Mbpp_999")
    grade = private_mod.grade_code(make_candidate(private.item_id, GOOD_PROGRAM), missing, run_paths)
    assert grade.status is GradeStatus.EVALUATOR_ERROR and grade.semantic_correct is None


def test_private_record_ref_cannot_escape_the_oracle_directory(run_paths):
    import dataclasses
    _, private = install_synthetic_task(run_paths)
    escaping = dataclasses.replace(private, private_record_ref="../../etc")
    grade = private_mod.grade_code(make_candidate(private.item_id, GOOD_PROGRAM), escaping, run_paths)
    assert grade.status is GradeStatus.EVALUATOR_ERROR


def test_task_without_public_case_is_infrastructure_error(run_paths):
    public, _ = install_synthetic_task(run_paths, examples=())
    result = public_mod.run_public_checks(make_candidate(public.item_id, GOOD_PROGRAM), public)
    assert result.status is CheckStatus.INFRASTRUCTURE_ERROR
    assert result.cases_total == 0 and result.pass_fraction is None
