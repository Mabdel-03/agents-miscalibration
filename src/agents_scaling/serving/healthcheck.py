"""Poll a vLLM server until it can serve requests.

vLLM exposes ``/health`` (200 once the engine is up) and ``/v1/models`` (lists the
loaded model). We wait on both: ``/health`` for liveness, ``/v1/models`` to confirm the
model finished loading (large models take minutes to load weights).
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request


def _get(url: str, timeout: float) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (trusted localhost)
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return 0, b""


def wait_until_ready(host: str, port: int, timeout_s: float = 1800.0, poll_s: float = 5.0) -> None:
    """Block until ``/health`` is 200 AND ``/v1/models`` lists a model. Raises on timeout."""
    base = f"http://{host}:{port}"
    deadline = time.time() + timeout_s
    last = "no attempt"
    while time.time() < deadline:
        hstatus, _ = _get(f"{base}/health", timeout=poll_s)
        if hstatus == 200:
            mstatus, body = _get(f"{base}/v1/models", timeout=poll_s)
            if mstatus == 200 and b'"id"' in body:
                return
            last = f"/health=200 but /v1/models={mstatus}"
        else:
            last = f"/health={hstatus}"
        time.sleep(poll_s)
    raise TimeoutError(f"vLLM at {base} not ready within {timeout_s}s (last: {last})")
