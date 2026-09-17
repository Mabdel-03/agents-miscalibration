"""Wall-clock bound for symbolic work, via SIGALRM on the main thread of a (worker) process.

``EvaluatorTimeout`` derives from ``BaseException`` so that broad ``except Exception`` handlers inside sympy,
latex2sympy2 or Math-Verify cannot swallow it; callers turn it into an explicit unknown. A previously armed
``ITIMER_REAL`` (for example pytest-timeout's) is re-armed with its remaining time on exit.
"""
from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from typing import Iterator


class EvaluatorTimeout(BaseException):
    """The bounded evaluation ran out of wall-clock time."""


class EvaluatorUnavailable(RuntimeError):
    """SIGALRM bounds are unavailable here (not the main thread of the main interpreter)."""


def signal_bounds_available() -> bool:
    return threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer")


def _raise_timeout(signum: int, frame: object) -> None:
    raise EvaluatorTimeout("bounded evaluation exceeded its wall-clock limit")


@contextmanager
def wall_clock_limit(seconds: float) -> Iterator[None]:
    if not seconds or seconds <= 0:
        raise ValueError("a positive time limit is required")
    if not signal_bounds_available():
        raise EvaluatorUnavailable("SIGALRM time limits need the main thread; run graders in worker processes")
    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    previous_delay, previous_interval = signal.setitimer(signal.ITIMER_REAL, 0)
    started = time.monotonic()
    delay = float(seconds) if previous_delay <= 0 else min(float(seconds), previous_delay)
    signal.setitimer(signal.ITIMER_REAL, delay)
    try:
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except EvaluatorTimeout:                 # fired after the body finished: the body's result stands
            signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0:
            remaining = previous_delay - (time.monotonic() - started)
            signal.setitimer(signal.ITIMER_REAL, max(remaining, 1e-3), previous_interval)
