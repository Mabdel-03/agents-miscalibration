"""Wall-clock bound for symbolic work, via SIGALRM on the main thread of a (worker) process.

``EvaluatorTimeout`` derives from ``BaseException`` so that broad ``except Exception`` handlers inside sympy,
latex2sympy2 or Math-Verify cannot swallow it; callers turn it into an explicit unknown.

State discipline: the limit validates its argument before touching any signal state; the previous ``SIGALRM``
handler and ``ITIMER_REAL`` timer are read first and restored in a ``finally`` that covers every step after the
first change, including an alarm that fires between installing the handler and arming the timer. A caller's timer
(for example a watchdog) is re-armed with its remaining time; if it expired while the limit was active, it is
re-armed to fire almost immediately so its own handler still runs.
"""
from __future__ import annotations

import math
import signal
import threading
import time
from contextlib import contextmanager
from typing import Iterator


class EvaluatorTimeout(BaseException):
    """The bounded evaluation ran out of wall-clock time."""


class EvaluatorUnavailable(RuntimeError):
    """A bounded evaluation cannot run here: no SIGALRM on this thread, or the isolated evaluator is down."""


_REARM_EXPIRED_S = 1e-3


def signal_bounds_available() -> bool:
    return threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer")


def _raise_timeout(signum: int, frame: object) -> None:
    raise EvaluatorTimeout("bounded evaluation exceeded its wall-clock limit")


def _restore(previous_handler: object, previous_delay: float, previous_interval: float, started: float) -> None:
    try:
        signal.setitimer(signal.ITIMER_REAL, 0)
    except EvaluatorTimeout:                     # our alarm fired at the very end: the body already finished
        signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous_handler if previous_handler is not None else signal.SIG_DFL)
    if previous_delay > 0:
        remaining = previous_delay - (time.monotonic() - started)
        signal.setitimer(signal.ITIMER_REAL, max(remaining, _REARM_EXPIRED_S), previous_interval)


@contextmanager
def wall_clock_limit(seconds: float) -> Iterator[None]:
    limit = float(seconds)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(f"a finite positive time limit is required, got {seconds!r}")
    if not signal_bounds_available():
        raise EvaluatorUnavailable("SIGALRM time limits need the main thread; run graders in worker processes")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    try:
        signal.signal(signal.SIGALRM, _raise_timeout)
        delay = limit if previous_delay <= 0 else min(limit, previous_delay)
        signal.setitimer(signal.ITIMER_REAL, delay)
        yield
    finally:
        _restore(previous_handler, previous_delay, previous_interval, started)
