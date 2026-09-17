"""A persistent spawn-context child process that runs bounded Math-Verify work on its own main thread.

Used automatically when a caller is not on the main thread (SIGALRM is unavailable there). The child applies the
same SIGALRM limit; the parent additionally waits at most ``timeout_s + grace_s`` for the reply and hard-kills the
child when it overruns, which also covers stalls inside C code that a signal cannot interrupt. Calls are serialized
per parent process; a killed or dead child is restarted on the next call.
"""
from __future__ import annotations

import multiprocessing
import signal
import threading
import time
from typing import Any

from agent_allocation.grading.text._limits import EvaluatorTimeout, EvaluatorUnavailable

DEFAULT_GRACE_S = 10.0
DEFAULT_STARTUP_S = 180.0


def _stall(seconds: float) -> str:
    """Test hook: an evaluation that ignores SIGALRM, like a stall inside C code."""
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    time.sleep(seconds)
    return "stalled"


def _serve(conn: Any) -> None:
    from agent_allocation.grading.text import math_equiv

    functions = {"verify_gold": math_equiv.verify_gold, "math_equivalent": math_equiv.math_equivalent,
                 "_stall": _stall}
    conn.send(("ready", None))
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message is None:
            return
        name, args, kwargs = message
        try:
            conn.send(("ok", functions[name](*args, **kwargs)))
        except BaseException as exc:  # noqa: BLE001 - reported to the parent as unavailability
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class IsolatedEvaluator:
    def __init__(self, *, grace_s: float = DEFAULT_GRACE_S, startup_s: float = DEFAULT_STARTUP_S) -> None:
        self.grace_s = grace_s
        self.startup_s = startup_s
        self._lock = threading.Lock()
        self._process: Any = None
        self._conn: Any = None

    def _kill(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.kill()
        if self._process is not None:
            self._process.join(timeout=5)
        if self._conn is not None:
            self._conn.close()
        self._process = None
        self._conn = None

    def _start(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_serve, args=(child,), daemon=True, name="agent04-isolated-evaluator")
        process.start()
        child.close()
        self._process, self._conn = process, parent
        if not parent.poll(self.startup_s):
            self._kill()
            raise EvaluatorUnavailable("isolated evaluator did not start in time")
        try:
            status, _ = parent.recv()
        except (EOFError, OSError) as exc:
            self._kill()
            raise EvaluatorUnavailable(f"isolated evaluator failed to start: {exc}") from exc
        if status != "ready":
            self._kill()
            raise EvaluatorUnavailable("isolated evaluator sent an unexpected handshake")

    def call(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any], timeout_s: float) -> Any:
        with self._lock:
            if self._process is None or not self._process.is_alive():
                self._kill()
                self._start()
            try:
                self._conn.send((name, args, kwargs))
                if not self._conn.poll(timeout_s + self.grace_s):
                    self._kill()
                    raise EvaluatorTimeout("isolated evaluation overran its limit; child killed")
                status, value = self._conn.recv()
            except (EOFError, OSError) as exc:
                self._kill()
                raise EvaluatorUnavailable(f"isolated evaluator died: {exc}") from exc
            if status != "ok":
                raise EvaluatorUnavailable(f"isolated evaluator error: {value}")
            return value

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.send(None)
                except (OSError, BrokenPipeError):
                    pass
            self._kill()


_SHARED: IsolatedEvaluator | None = None
_SHARED_LOCK = threading.Lock()


def shared_evaluator() -> IsolatedEvaluator:
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = IsolatedEvaluator()
        return _SHARED
