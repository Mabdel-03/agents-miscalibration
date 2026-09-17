"""Trusted RPC bootstrap, mounted alone inside the candidate namespace.

No grading labels or private paths may be passed to this process. Results are
untrusted values and are graded by a separate parent process.

Vendored from ``learnagents/runtime/sandbox_worker.py`` (agent_methods_bench@2af2879, sha256 d052fbf4…).
Changes: the RPC channel moves off fds 0/1 (candidate writes to raw fd 1 cannot corrupt the protocol and stdin
reads see EOF, like EvalPlus ``swallow_io``); captured stdout/stderr is bounded by ``output_limit`` and
truncated instead of growing without bound; results are written with ``surrogatepass``.
"""
import contextlib
import io
import json
import os
import sys
import traceback

_rpc_in_fd = os.dup(0)
_rpc_out_fd = os.dup(1)
_null_fd = os.open(os.devnull, os.O_RDWR)
os.dup2(_null_fd, 0)
os.dup2(_null_fd, 1)
os.close(_null_fd)
rpc_in = os.fdopen(_rpc_in_fd, "r", encoding="utf-8", errors="surrogatepass")
rpc_out = os.fdopen(_rpc_out_fd, "w", encoding="utf-8", errors="surrogatepass")


def emit(value):
    rpc_out.write(json.dumps(value, ensure_ascii=False) + "\n")
    rpc_out.flush()


def rpc(kind, payload):
    emit({"type": "rpc", "kind": kind, "payload": payload})
    reply = json.loads(rpc_in.readline())
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error", "Broker denied request"))
    return reply.get("result")


class Finalized(BaseException):
    def __init__(self, value):
        self.value = value


def finalize(value):
    raise Finalized(value)


def call_tool(name, **kwargs):
    return rpc("tool", {"name": name, "arguments": kwargs})


def lm_call(prompt, context_refs=None, **kwargs):
    return rpc("lm", {"prompt": prompt, "context_refs": context_refs or [], **kwargs})


class BoundedCapture(io.TextIOBase):
    """Keeps the first ``limit`` characters written; later text is counted and discarded."""

    def __init__(self, limit):
        self.limit = limit
        self.parts = []
        self.kept = 0
        self.dropped = 0

    def writable(self):
        return True

    def write(self, text):
        if not isinstance(text, str):
            raise TypeError("write() argument must be str, not " + type(text).__name__)
        room = self.limit - self.kept
        if room > 0:
            piece = text[:room]
            self.parts.append(piece)
            self.kept += len(piece)
        self.dropped += max(0, len(text) - max(room, 0))
        return len(text)

    def getvalue(self):
        data = "".join(self.parts).encode("utf-8", "surrogatepass")
        return data[:self.limit].decode("utf-8", "ignore"), self.dropped > 0 or len(data) > self.limit


output_limit = 64 * 1024
namespace = {"__name__": "__agent__", "finalize": finalize, "call_tool": call_tool}
for line in rpc_in:
    try:
        request = json.loads(line)
        if request["op"] == "init":
            namespace["public_input"] = request["public"]
            namespace["documents"] = request["public"].get("documents", [])
            output_limit = int(request.get("output_limit", output_limit))
            if request.get("allow_children"):
                namespace["lm_call"] = lm_call
            emit({"type": "result", "ok": True, "stdout": ""})
            continue
        if request["op"] == "close":
            break
        output = BoundedCapture(output_limit)
        value = None
        finalized = False
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            if request["op"] == "exec":
                try:
                    exec(compile(request["code"], "<agent>", "exec"), namespace)
                except Finalized as final:
                    value, finalized = final.value, True
            elif request["op"] == "invoke":
                # The candidate has only the single current input, never expected answers.
                value = namespace[request.get("entry_point", "solve")](request["input"])
            else:
                raise ValueError("Unknown sandbox operation")
        captured, truncated = output.getvalue()
        emit({"type": "result", "ok": True, "stdout": captured, "stdout_truncated": truncated,
              "value": value, "finalized": finalized})
    except BaseException as exc:
        try:
            error = type(exc).__name__ + ": " + str(exc)[:2000]
        except BaseException:
            error = type(exc).__name__
        emit({"type": "result", "ok": False, "error": error, "error_type": type(exc).__name__,
              "stdout": "", "traceback": traceback.format_exc(limit=3)[-4000:]})
