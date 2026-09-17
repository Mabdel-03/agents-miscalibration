"""Gold-blind symmetric MATH equivalence: mutual verification, asymmetry, unknowns and bounds."""
from __future__ import annotations

import threading
import time

import pytest

from agent_allocation.grading.text import math_equiv
from agent_allocation.grading.text._isolated import IsolatedEvaluator
from agent_allocation.grading.text._limits import EvaluatorTimeout, EvaluatorUnavailable, wall_clock_limit
from agent_allocation.grading.text.math_equiv import math_equivalent


@pytest.mark.parametrize("a, b", [
    ("\\frac{1}{2}", "0.5"), ("\\boxed{3}", "3"), ("$2\\sqrt{2}$", "\\sqrt{8}"), ("x=5", "5"),
    ("\\{1,2\\}", "\\{2,1\\}"),
])
def test_mutually_verified_answers_are_equivalent(a, b):
    assert math_equivalent(a, b) is True
    assert math_equivalent(b, a) is True


@pytest.mark.parametrize("a, b", [("3", "4"), ("\\frac{1}{2}", "\\frac{1}{3}"), ("\\text{Evelyn}", "\\text{Navin}")])
def test_different_answers_are_not_equivalent(a, b):
    assert math_equivalent(a, b) is False


def test_one_direction_only_is_not_a_merge():
    parse, verify = math_equiv.parse_answer, math_equiv._verify
    from agent_allocation.grading.text._limits import wall_clock_limit
    with wall_clock_limit(10):
        interval, relation = parse("(1,2)"), parse("1<x<2")
        assert verify(relation, interval) is True       # Math-Verify's documented asymmetry
        assert verify(interval, relation) is False
    assert math_equivalent("(1,2)", "1<x<2") is False
    assert math_equivalent("1<x<2", "(1,2)") is False


def test_unparseable_is_unknown():
    assert math_equivalent("", "3") is None
    assert math_equivalent("3", "   ") is None


def test_exceptions_and_timeouts_are_unknown(monkeypatch):
    def broken(answer):
        raise ValueError("bad latex")
    monkeypatch.setattr(math_equiv, "parse_answer", broken)
    assert math_equivalent("1", "1") is None
    monkeypatch.undo()

    def spin(gold, prediction):
        while True:
            pass
    monkeypatch.setattr(math_equiv, "_verify", spin)
    assert math_equivalent("1", "1", timeout_s=0.2) is None


def test_wall_clock_limit_refuses_off_main_thread():
    errors = []

    def run():
        try:
            with wall_clock_limit(1.0):
                pass
        except EvaluatorUnavailable as exc:
            errors.append(exc)
    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    assert len(errors) == 1


def test_off_main_thread_calls_run_bounded_in_the_isolated_child():
    results = {}

    def run():
        results["equal"] = math_equivalent("\\frac{1}{2}", "0.5")
        results["different"] = math_equivalent("3", "4")
        results["verdict"] = math_equiv.verify_gold("1<x<2", "(1,2)")
    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=240)
    assert results == {"equal": True, "different": False,
                       "verdict": math_equiv.MathVerdict(True, "math_verify_equal")}


def test_isolated_child_stalled_in_c_code_is_killed_and_replaced():
    evaluator = IsolatedEvaluator(grace_s=0.5)
    try:
        started = time.monotonic()
        with pytest.raises(EvaluatorTimeout):
            evaluator.call("_stall", (60.0,), {}, timeout_s=0.5)
        assert time.monotonic() - started < 30
        assert evaluator.call("math_equivalent", ("2", "2.0"), {"timeout_s": 5.0}, timeout_s=5.0) is True
    finally:
        evaluator.close()
