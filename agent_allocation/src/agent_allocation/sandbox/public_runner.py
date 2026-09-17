"""Trusted in-sandbox helpers for public-example checks (new code, not vendored).

``PUBLIC_TRANSPORT`` is the source of the definitions below; it is executed inside a fresh sandbox (trusted phase)
before any candidate code. The candidate program then runs in its own module dict pre-populated with ``math`` and
``sys`` (no ``__name__``, as in EvalPlus ``exec(code, {})``), and each public ``assert`` is executed in that dict
under an interval timer. Only the public assert text (already shown in the prompt) enters the process.
"""
from __future__ import annotations

import inspect


class _PublicTimeout(BaseException):
    pass


def _public_timer(seconds):
    import signal

    def expired(*_):
        raise _PublicTimeout()

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return previous


def _public_stop(previous):
    import signal
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


def _load_public(value):
    import math
    import sys
    module = {"math": math, "sys": sys}
    _public_state["module"] = module
    previous = _public_timer(value["timeout"])
    try:
        exec(compile(value["source"], "<candidate>", "exec"), module)
    except _PublicTimeout:
        return {"outcome": "timeout"}
    except BaseException as exc:
        return {"outcome": "error", "error_type": type(exc).__name__}
    finally:
        _public_stop(previous)
    return {"outcome": "loaded"}


def _run_public_case(value):
    module = _public_state["module"]
    previous = _public_timer(value["timeout"])
    try:
        exec(compile(value["assert"], "<public-example>", "exec"), module)
    except _PublicTimeout:
        return {"outcome": "timeout"}
    except AssertionError:
        return {"outcome": "fail"}
    except BaseException as exc:
        return {"outcome": "error", "error_type": type(exc).__name__}
    finally:
        _public_stop(previous)
    return {"outcome": "pass"}


PUBLIC_TRANSPORT = "_public_state={}\n" + "\n".join(
    inspect.getsource(f) for f in [_PublicTimeout, _public_timer, _public_stop, _load_public, _run_public_case])
