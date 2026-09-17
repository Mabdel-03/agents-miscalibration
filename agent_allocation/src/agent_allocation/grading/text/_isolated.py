"""A persistent spawn-context child process that runs bounded Math-Verify work on its own main thread.

Used automatically when a caller is not on the main thread (SIGALRM is unavailable there). The child imports and
warms Math-Verify *before* its ready handshake, so import time counts against ``startup_s``, not a call. Each call
applies the same SIGALRM limit in the child; the parent additionally waits at most ``timeout_s + grace_s`` for the
reply and hard-kills the child when it overruns, which also covers stalls inside C code that a signal cannot
interrupt. Calls are serialized per parent process; a killed or dead child is restarted on the next call.

Every failure to start, reach or hear from the child (including ``process.start()`` errors such as a daemonic
parent or EAGAIN) is raised as ``EvaluatorUnavailable``; an overrun as ``EvaluatorTimeout``. A forked child process
never inherits the evaluator: ``os.register_at_fork`` resets the shared instance and its lock there.
"""
from __future__ import annotations

import multiprocessing
import os
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
    try:
        from agent_allocation.grading.text import math_equiv

        math_equiv._math_verify()                    # import + warm-up inside the startup window
        functions = {"verify_gold": math_equiv.verify_gold, "math_equivalent": math_equiv.math_equivalent,
                     "_stall": _stall}
    except BaseException as exc:  # noqa: BLE001 - reported through the handshake
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
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
        process, conn = self._process, self._conn
        self._process = None
        self._conn = None
        try:
            if process is not None and process.is_alive():
                process.kill()
            if process is not None:
                process.join(timeout=5)
        except Exception:  # noqa: BLE001 - best effort; the object is dropped either way
            pass
        try:
            if conn is not None:
                conn.close()
        except Exception:  # noqa: BLE001
            pass

    def _start(self) -> None:
        try:
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            process = context.Process(target=_serve, args=(child,), daemon=True, name="agent04-isolated-evaluator")
            process.start()
            child.close()
        except Exception as exc:  # noqa: BLE001 - e.g. daemonic parent, EAGAIN, EMFILE
            raise EvaluatorUnavailable(f"isolated evaluator could not start: {type(exc).__name__}: {exc}") from exc
        self._process, self._conn = process, parent
        try:
            if not parent.poll(self.startup_s):
                raise EvaluatorUnavailable("isolated evaluator did not start in time")
            status, detail = parent.recv()
        except EvaluatorUnavailable:
            self._kill()
            raise
        except Exception as exc:  # noqa: BLE001
            self._kill()
            raise EvaluatorUnavailable(f"isolated evaluator failed to start: {type(exc).__name__}: {exc}") from exc
        if status != "ready":
            self._kill()
            raise EvaluatorUnavailable(f"isolated evaluator failed to start: {detail}")

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
            except EvaluatorTimeout:
                raise
            except Exception as exc:  # noqa: BLE001 - broken pipe, EOF, unpicklable reply...
                self._kill()
                raise EvaluatorUnavailable(f"isolated evaluator died: {type(exc).__name__}: {exc}") from exc
            if status != "ok":
                raise EvaluatorUnavailable(f"isolated evaluator error: {value}")
            return value

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.send(None)
                except Exception:  # noqa: BLE001
                    pass
            self._kill()


_SHARED: IsolatedEvaluator | None = None
_SHARED_LOCK = threading.Lock()


def shared_evaluator() -> IsolatedEvaluator:
    """The per-process evaluator (created lazily)."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = IsolatedEvaluator()
        return _SHARED


def _reset_after_fork() -> None:
    """In a forked child: drop the parent's evaluator without touching its process, and replace the locks."""
    global _SHARED, _SHARED_LOCK
    inherited = _SHARED
    _SHARED = None
    _SHARED_LOCK = threading.Lock()
    if inherited is not None:
        inherited._lock = threading.Lock()
        try:
            if inherited._conn is not None:
                inherited._conn.close()          # closes only this child's copy of the parent's pipe end
        except Exception:  # noqa: BLE001
            pass
        inherited._conn = None
        inherited._process = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
