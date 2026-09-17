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


# ---------------------------------------------------------------------------------------------------------
# Regressions: order independence, unparseable finals, signal state, isolation failures, fork safety
# ---------------------------------------------------------------------------------------------------------

def test_outage_injected_at_math_verify_is_unknown_even_after_warm_calls(monkeypatch):
    import math_verify.grader
    import math_verify.parser
    assert math_equivalent("1", "1.0") is True                  # warm every lazy path first (test-order proof)
    assert math_equiv.verify_gold("4", "4").correct is True

    def outage(*args, **kwargs):
        raise RuntimeError("injected evaluator outage")
    monkeypatch.setattr(math_verify.parser, "parse", outage)
    monkeypatch.setattr(math_verify.grader, "verify", outage)
    assert math_equiv.verify_gold("4", "4") == math_equiv.MathVerdict(None, "evaluator_exception:RuntimeError")
    assert math_equivalent("4", "4") is None
    monkeypatch.undo()
    assert math_equiv.verify_gold("4", "4").correct is True


@pytest.mark.parametrize("a, b", [("\\frac{1}{", "0.5"), ("???", "???"), ("\\frac{1}{", "\\frac{1}{."),
                                  ("(1,2", "(1,2")])
def test_malformed_math_is_unparseable_not_a_string_match(a, b):
    assert math_equivalent(a, b) is None


def test_invalid_timeout_raises_before_touching_signal_state():
    import signal
    handler_before = signal.getsignal(signal.SIGALRM)
    for bad in (float("nan"), 0, -1, float("inf")):
        with pytest.raises(ValueError):
            math_equivalent("1", "1", timeout_s=bad)
        with pytest.raises(ValueError):
            with wall_clock_limit(bad):
                pass
    assert signal.getsignal(signal.SIGALRM) is handler_before
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_limit_restores_the_callers_handler_and_rearms_its_timer():
    import signal
    fired = []

    def outer(signum, frame):
        fired.append(signum)
    previous = signal.signal(signal.SIGALRM, outer)
    try:
        signal.setitimer(signal.ITIMER_REAL, 30.0)
        with pytest.raises(EvaluatorTimeout):
            with wall_clock_limit(0.05):
                while True:
                    pass
        assert signal.getsignal(signal.SIGALRM) is outer
        remaining = signal.getitimer(signal.ITIMER_REAL)[0]
        assert 25.0 < remaining <= 30.0
        signal.setitimer(signal.ITIMER_REAL, 0.02)                   # an outer alarm that expires inside the limit
        with pytest.raises(EvaluatorTimeout):
            with wall_clock_limit(5.0):
                while True:
                    pass
        deadline = time.monotonic() + 2.0
        while not fired and time.monotonic() < deadline:             # re-armed: the caller's handler still runs
            time.sleep(0.01)
        assert fired
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_isolated_start_failure_is_unknown_not_raised(monkeypatch):
    from agent_allocation.grading.text import _isolated

    def broken_context(method):
        raise OSError("EAGAIN: cannot fork")
    monkeypatch.setattr(_isolated, "_SHARED", None)
    monkeypatch.setattr(_isolated.multiprocessing, "get_context", broken_context)
    results = {}

    def run():
        results["equivalent"] = math_equivalent("1", "1")
        results["verdict"] = math_equiv.verify_gold("1", "1")
    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=60)
    assert results == {"equivalent": None, "verdict": math_equiv.MathVerdict(None, "evaluator_unavailable")}


def test_forked_child_never_inherits_the_shared_evaluator():
    import os
    from agent_allocation.grading.text import _isolated
    parent_instance = _isolated.shared_evaluator()
    pid = os.fork()
    if pid == 0:                                                     # child: must see a fresh, unlocked state
        ok = _isolated._SHARED is None and _isolated.shared_evaluator() is not parent_instance
        os._exit(0 if ok else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert _isolated.shared_evaluator() is parent_instance
