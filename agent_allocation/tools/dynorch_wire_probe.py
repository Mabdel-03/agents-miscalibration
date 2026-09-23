#!/usr/bin/env python3
"""dynorch_v1 WP-E: early wire probe (throwaway stub server + real-CLI case runner, $0).

Contract reference: `docs/dynorch/00_contract.md` §6 (wire layer: `POST chat/completions`, `POST responses`,
`GET models`, SSE keepalives) and the plan §6 (harness invocations) / §9 Gate 0 (real-CLI smoke list). This
tool is NOT the proxy: it is a minimal stdlib HTTP server that replays a scripted reply sequence so the exact
wire behaviour of OpenClaude 0.13.0 and Codex CLI 0.156.1 can be captured before `dynorch/httpd.py` exists.

Routes (all on 127.0.0.1, any run key in the path):
  POST /r/<key>/v1/chat/completions   OpenAI chat body; SSE when ``stream`` is true, else JSON
  POST /r/<key>/v1/responses          Codex Responses body; always SSE (response.created -> output_item.done*
                                      -> response.completed with usage)
  GET  /r/<key>/v1/models             one model
Everything else is logged and answered 404 so unexpected traffic is visible, never silently served.

Replies are chosen per conversation role (``root`` or ``child``; a child is a request carrying the
``x-openai-subagent`` header or whose system text is the sub-agent prompt) by the number of earlier calls of that
role. Optional faults: HTTP 402 on a given call, a delayed streaming reply with keepalive comments, a Responses
stream with no ``response.completed``, a ``response.failed`` event, ``reasoning_content`` on every reply.

Every request is journaled to ``requests.jsonl`` (method, path, headers with the bearer redacted, full JSON body
trimmed above 200 KB, the reply summary). No network call leaves 127.0.0.1; no OpenRouter key is read.

Usage:
  uv run python tools/dynorch_wire_probe.py serve [--port P] [--harness openclaude|codex] [--case NAME]
  uv run python tools/dynorch_wire_probe.py run openclaude a|b|c|d|e [--delay S] [--wall S]
  uv run python tools/dynorch_wire_probe.py run codex a|b_ns|b_nons|c|d|e|f|g|h [--delay S] [--wall S] [--cli-config k=v] [--config-extra TOML] [--label NAME]
  uv run python tools/dynorch_wire_probe.py summarize openclaude|codex   # rebuild summary.json from captures
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PKG = Path(__file__).resolve().parents[1]
PROMPTS = PKG / "prompts"
SCRATCH = Path("/orcd/data/tpoggio/001/mabdel03/dynorch_scratch")
PROBE_ROOT = SCRATCH / "probe"
CAPTURES = PKG / "tests" / "fixtures" / "dynorch" / "captures"
MODEL = "qwen/qwen3.8-27b"
BODY_TRIM = 200_000
CODEX_BIN = SCRATCH / "npm-prefix" / "bin" / "codex"
OPENCLAUDE_BIN = "openclaude"
CONSORTIUM_BIN = "/orcd/home/002/mabdel03/conda_envs/consortium/bin"
KEEPALIVE_SECONDS = 15.0
CHILD_MARKER = "You are a sub-agent solving one delegated task"
ROUTE = re.compile(r"^/r/(?P<key>[^/]+)/v1/(?P<route>chat/completions|responses|models)$")
REDACTED_HEADERS = frozenset({"authorization", "cookie", "set-cookie", "x-api-key", "proxy-authorization"})
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{4,}")
_KEY = re.compile("sk-" + "or-" + r"(?:v\d+-)?[A-Za-z0-9_\-]{8,}")  # split so a doctor grep never matches this file

OPENCLAUDE_DISALLOWED = ["Bash", "Edit", "MultiEdit", "Write", "Read", "NotebookEdit", "Glob", "Grep", "WebSearch",
                        "WebFetch", "TodoWrite", "TaskCreate", "TaskUpdate", "KillShell", "BashOutput", "SendMessage",
                        "TaskStop", "ToolSearchTool", "Skill"]

BUDGET_SENTENCE = ("COMPUTE BUDGET: you have a total budget of {C} completion tokens for this task, counted across "
                   "every model call you make, including your own reasoning, any tool calls, and every sub-agent you "
                   "spawn (each sub-agent runs the same model and draws from this same budget). When the budget is "
                   "exhausted your current output is closed and you must answer immediately, so reserve enough for "
                   "the final answer. Input tokens are not counted.")
TOY_TASK = "Compute the sum of the first three odd primes. Give the integer."
TOY_REQUIREMENT = ("Put exactly one mathematical answer inside <final>. A boxed mathematical expression is allowed. "
                   "Do not put the derivation in <final>.")


# ----------------------------------------------------------------------------------------------- redaction / io

def redact_text(text: str) -> str:
    """Redact any bearer token or OpenRouter-shaped key in journaled text (contract: every journaled byte)."""
    return _BEARER.sub("Bearer [REDACTED]", _KEY.sub("[REDACTED_KEY]", text))


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k.lower(): ("[REDACTED]" if k.lower() in REDACTED_HEADERS else redact_text(v)) for k, v in headers.items()}


def trim_body(raw: bytes) -> Any:
    """Parsed JSON body, or a raw prefix when it is not JSON or exceeds the 200 KB trim limit."""
    text = raw.decode("utf-8", errors="replace")
    if len(raw) > BODY_TRIM:
        return {"_truncated": True, "_bytes": len(raw), "_prefix": redact_text(text[:BODY_TRIM])}
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": redact_text(text)}


def system_text() -> str:
    """The root system prompt exactly as `--system-prompt "$(cat …)"` passes it (trailing newlines stripped)."""
    return (PROMPTS / "system_dynorch.txt").read_text(encoding="utf-8").rstrip("\n")


def subagent_text() -> str:
    return (PROMPTS / "subagent_dynorch.txt").read_text(encoding="utf-8").rstrip("\n")


def task_text(cap: int = 4096) -> str:
    root = (PROMPTS / "root.txt").read_text(encoding="utf-8")
    body = root.replace("{task_prompt}", TOY_TASK).replace("{output_requirement}", TOY_REQUIREMENT).rstrip("\n")
    return body + "\n\n" + BUDGET_SENTENCE.format(C=cap) + "\n"


# ----------------------------------------------------------------------------------------------- reply scripts

@dataclass
class Reply:
    """One scripted model reply. ``kind``: text | tool_call | http_error | missing_completed | failed_event."""
    kind: str
    text: str = ""
    reasoning: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    namespace: str | None = None
    status: int = 200
    error_code: str | None = None
    delay: float = 0.0
    usage: tuple[int, int, int] = (100, 20, 5)  # prompt, completion, reasoning

    def summary(self) -> dict[str, Any]:
        return {"kind": self.kind, "status": self.status, "tool_name": self.tool_name, "namespace": self.namespace,
                "delay": self.delay, "error_code": self.error_code, "reasoning": self.reasoning is not None,
                "text": self.text[:200]}


def _final(text: str, reasoning: str | None = None, **kw: Any) -> Reply:
    return Reply("text", text=f"<rationale>\nStub reply.\n</rationale>\n<final>{text}</final>", reasoning=reasoning, **kw)


def openclaude_script(case: str, delay: float) -> dict[str, list[Reply]]:
    """Scripted sequences for the OpenClaude cases (a)–(e) of the WP-E brief."""
    spawn = Reply("tool_call", tool_name="Agent",
                  tool_args={"description": "solve the toy task", "subagent_type": "solver",
                             "prompt": f"{TOY_TASK}\n{TOY_REQUIREMENT}"})
    child = _final("child says 7")
    if case == "a":
        return {"root": [_final("7")], "child": [child]}
    if case == "b":
        return {"root": [spawn, _final("7")], "child": [child]}
    if case == "c":
        return {"root": [spawn, Reply("http_error", status=402, error_code="budget_exhausted")], "child": [child]}
    if case == "d":
        return {"root": [_final("7", delay=delay)], "child": [child]}
    if case == "e":
        r1 = copy.deepcopy(spawn)
        r1.reasoning = "STUB-REASONING-ONE: the root is thinking before it delegates."
        r2 = _final("7", reasoning="STUB-REASONING-TWO: the root is thinking before it answers.")
        return {"root": [r1, r2], "child": [_final("child says 7", reasoning="STUB-REASONING-CHILD")]}
    raise ValueError(f"unknown openclaude case {case!r}")


def codex_script(case: str, delay: float) -> dict[str, list[Reply]]:
    """Scripted sequences for the Codex cases (a)–(g) of the WP-E brief."""
    ns = "multi_agent_v1"
    spawn_args = {"message": f"{TOY_TASK}\n{TOY_REQUIREMENT}"}
    child = _final("child says 7")
    if case == "a":
        return {"root": [_final("7")], "child": [child]}
    if case in {"b_ns", "b_nons"}:
        space = ns if case == "b_ns" else None
        return {"root": [Reply("tool_call", tool_name="spawn_agent", tool_args=spawn_args, namespace=space),
                         Reply("tool_call", tool_name="wait_agent", tool_args={"targets": "$spawned_ids"},
                               namespace=space),
                         _final("7")],
                "child": [child]}
    if case == "c":
        return {"root": [Reply("tool_call", tool_name="exec_command", tool_args={"cmd": "touch /tmp/dynorch_probe_leak"}),
                         _final("7")], "child": [child]}
    if case == "d":
        return {"root": [Reply("missing_completed", text="<final>7</final>")], "child": [child]}
    if case == "e":
        return {"root": [Reply("failed_event", error_code="rate_limit_exceeded"), _final("7")], "child": [child]}
    if case == "f":
        return {"root": [_final("7", delay=delay)], "child": [child]}
    if case == "g":
        return {"root": [_final("7")], "child": [child]}
    if case == "h":  # reasoning item on a tool-call turn: is it echoed back on the next request?
        return {"root": [Reply("tool_call", tool_name="wait_agent", tool_args={"targets": []}, namespace=ns,
                               reasoning="STUB-REASONING-ONE: thinking before the tool call."),
                         _final("7", reasoning="STUB-REASONING-TWO: thinking before the answer.")],
                "child": [child]}
    raise ValueError(f"unknown codex case {case!r}")


# ----------------------------------------------------------------------------------------------- stub server

class Probe:
    """Shared state of the stub: the script, the per-role counters and the JSONL journal."""

    def __init__(self, script: dict[str, list[Reply]], journal: Path) -> None:
        self.script = script
        self.journal = journal
        self.lock = threading.Lock()
        self.seq = 0
        self.counts: dict[str, int] = {"root": 0, "child": 0}
        self.records: list[dict[str, Any]] = []
        self.system_sha = _sha(system_text())
        self.child_sha = _sha(subagent_text())

    def next_reply(self, role: str) -> tuple[int, Reply]:
        with self.lock:
            n = self.counts[role]
            self.counts[role] = n + 1
            seq = self.script[role]
            return n, seq[min(n, len(seq) - 1)]

    def record(self, rec: dict[str, Any]) -> None:
        with self.lock:
            self.seq += 1
            rec = {"seq": self.seq, **rec}
            self.records.append(rec)
            with self.journal.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_role(headers: dict[str, str], body: Any, route: str) -> tuple[str, dict[str, Any]]:
    """root | child plus the evidence used (header, system sha)."""
    lower = {k.lower(): v for k, v in headers.items()}
    info: dict[str, Any] = {"x_openai_subagent": lower.get("x-openai-subagent")}
    system = None
    if isinstance(body, dict):
        if route == "chat/completions":
            msgs = body.get("messages") or []
            if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
                system = msgs[0].get("content")
                if isinstance(system, list):
                    system = "".join(part.get("text", "") for part in system if isinstance(part, dict))
        elif route == "responses":
            system = body.get("instructions")
    if isinstance(system, str):
        info["system_sha256"] = _sha(system)
        info["system_chars"] = len(system)
    if lower.get("x-openai-subagent") is not None:
        return "child", info
    if isinstance(system, str) and CHILD_MARKER in system:
        return "child", info
    return "root", info


def extract_spawned_ids(body: Any) -> list[str]:
    """Agent ids Codex reported in earlier ``function_call_output`` items (used by the wait_agent step)."""
    ids: list[str] = []
    if not isinstance(body, dict):
        return ids
    for item in body.get("input") or []:
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        out = item.get("output")
        text = out if isinstance(out, str) else json.dumps(out)
        for match in re.finditer(r'"(?:agent_id|thread_id|id)"\s*:\s*"([^"]+)"', text):
            ids.append(match.group(1))
    return ids


def _usage_chat(reply: Reply) -> dict[str, Any]:
    p, c, r = reply.usage
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
            "completion_tokens_details": {"reasoning_tokens": r}}


def _usage_responses(reply: Reply) -> dict[str, Any]:
    p, c, r = reply.usage
    return {"input_tokens": p, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": c,
            "output_tokens_details": {"reasoning_tokens": r}, "total_tokens": p + c}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "StubServer"

    def log_message(self, fmt: str, *args: Any) -> None:  # silence the default stderr log
        return

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    # -- plumbing

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.headers.get("Content-Encoding", "").lower() == "gzip":
            import gzip
            raw = gzip.decompress(raw)
        return raw

    def _send(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

    def _start_sse(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()

    def _sse_wait(self, delay: float) -> int:
        """Keep the stream alive with comments for ``delay`` seconds; returns the number of comments sent."""
        sent = 0
        deadline = time.monotonic() + delay
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return sent
            time.sleep(min(KEEPALIVE_SECONDS, remaining))
            if time.monotonic() < deadline:
                self.wfile.write(b": dynorch keepalive\n\n")
                self.wfile.flush()
                sent += 1

    def _event(self, obj: dict[str, Any], event_name: str | None = None) -> None:
        line = b""
        if event_name:
            line += f"event: {event_name}\n".encode()
        line += b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n"
        self.wfile.write(line)
        self.wfile.flush()

    # -- routing

    def _handle(self, method: str) -> None:
        probe = self.server.probe
        started = time.monotonic()
        raw = self._read_body()
        headers = {k: v for k, v in self.headers.items()}
        body = trim_body(raw)
        rec: dict[str, Any] = {"ts": time.time(), "method": method, "path": self.path,
                               "headers": redact_headers(headers), "body_bytes": len(raw), "body": body}
        match = ROUTE.match(self.path)
        if not match:
            rec["reply"] = {"kind": "not_found", "status": 404}
            probe.record(rec)
            self._send(404, "application/json", b'{"error":{"code":"not_found"}}')
            return
        key, route = match.group("key"), match.group("route")
        auth = headers.get("Authorization") or headers.get("authorization") or ""
        rec["run_key"] = key
        rec["bearer_matches_path_key"] = auth == f"Bearer {key}"
        if route == "models":
            if method != "GET":
                rec["reply"] = {"kind": "method_not_allowed", "status": 405}
                probe.record(rec)
                self._send(405, "application/json", b'{"error":{"code":"method_not_allowed"}}')
                return
            payload = json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "dynorch"}]}).encode()
            rec["reply"] = {"kind": "models", "status": 200}
            probe.record(rec)
            self._send(200, "application/json", payload)
            return
        if method != "POST":
            rec["reply"] = {"kind": "method_not_allowed", "status": 405}
            probe.record(rec)
            self._send(405, "application/json", b'{"error":{"code":"method_not_allowed"}}')
            return
        role, evidence = classify_role(headers, body, route)
        n, reply = probe.next_reply(role)
        rec.update({"role": role, "call_index_in_role": n, "classification": evidence, "reply": reply.summary()})
        stream = bool(body.get("stream")) if isinstance(body, dict) else False
        rec["stream_requested"] = stream
        try:
            if reply.kind == "http_error":
                payload = json.dumps({"error": {"code": reply.error_code, "message": f"stub {reply.error_code}",
                                                "type": "insufficient_quota"}}).encode()
                self._send(reply.status, "application/json", payload)
            elif route == "chat/completions":
                rec["reply"]["keepalives"] = self._chat(body, reply, stream)
            else:
                rec["reply"]["keepalives"] = self._responses(body, reply)
        except (BrokenPipeError, ConnectionResetError) as exc:
            rec["reply"]["client_disconnected"] = repr(exc)
        rec["duration_s"] = round(time.monotonic() - started, 3)
        probe.record(rec)

    # -- chat/completions

    def _chat(self, body: Any, reply: Reply, stream: bool) -> int:
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        tool_calls = None
        finish = "stop"
        if reply.kind == "tool_call":
            tool_calls = [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                           "function": {"name": reply.tool_name, "arguments": json.dumps(reply.tool_args)}}]
            finish = "tool_calls"
        keepalives = 0
        if not stream:
            if reply.delay:
                time.sleep(reply.delay)
            message: dict[str, Any] = {"role": "assistant", "content": reply.text if not tool_calls else ""}
            if reply.reasoning is not None:
                message["reasoning_content"] = reply.reasoning
            if tool_calls:
                message["tool_calls"] = tool_calls
            payload = {"id": cid, "object": "chat.completion", "created": created, "model": MODEL,
                       "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                       "usage": _usage_chat(reply)}
            self._send(200, "application/json", json.dumps(payload, ensure_ascii=False).encode())
            return 0
        self._start_sse()
        if reply.delay:
            keepalives = self._sse_wait(reply.delay)

        def chunk(delta: dict[str, Any], finish_reason: str | None = None, usage: dict[str, Any] | None = None,
                  choices: bool = True) -> None:
            obj: dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                                   "choices": ([{"index": 0, "delta": delta, "finish_reason": finish_reason}] if choices else [])}
            if usage is not None:
                obj["usage"] = usage
            self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        chunk({"role": "assistant", "content": ""})
        if reply.reasoning is not None:
            chunk({"reasoning_content": reply.reasoning})
        if tool_calls:
            for i, call in enumerate(tool_calls):
                chunk({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                       "function": {"name": call["function"]["name"], "arguments": ""}}]})
                chunk({"tool_calls": [{"index": i, "function": {"arguments": call["function"]["arguments"]}}]})
        else:
            chunk({"content": reply.text})
        chunk({}, finish_reason=finish)
        chunk({}, usage=_usage_chat(reply), choices=False)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        return keepalives

    # -- responses

    def _responses(self, body: Any, reply: Reply) -> int:
        rid = f"resp_{uuid.uuid4().hex[:16]}"
        self._start_sse()
        keepalives = 0
        if reply.delay:
            keepalives = self._sse_wait(reply.delay)
        self._event({"type": "response.created", "response": {"id": rid, "object": "response", "status": "in_progress",
                                                               "output": []}}, "response.created")
        if reply.kind == "failed_event":
            self._event({"type": "response.failed", "response": {"id": rid, "object": "response", "status": "failed",
                                                                  "error": {"code": reply.error_code,
                                                                            "message": f"stub {reply.error_code}"}}},
                        "response.failed")
            return keepalives
        items: list[dict[str, Any]] = []
        if reply.reasoning is not None:
            items.append({"type": "reasoning", "id": f"rs_{uuid.uuid4().hex[:12]}", "summary": [],
                          "content": [{"type": "reasoning_text", "text": reply.reasoning}]})
        if reply.kind == "tool_call":
            args = dict(reply.tool_args)
            if args.get("targets") == "$spawned_ids":
                args["targets"] = extract_spawned_ids(body)
            item = {"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:12]}", "call_id": f"call_{uuid.uuid4().hex[:12]}",
                    "name": reply.tool_name, "arguments": json.dumps(args), "status": "completed"}
            if reply.namespace:
                item["namespace"] = reply.namespace
            items.append(item)
        else:
            items.append({"type": "message", "id": f"msg_{uuid.uuid4().hex[:12]}", "role": "assistant", "status": "completed",
                          "content": [{"type": "output_text", "text": reply.text, "annotations": []}]})
        for i, item in enumerate(items):
            self._event({"type": "response.output_item.done", "output_index": i, "item": item}, "response.output_item.done")
        if reply.kind == "missing_completed":
            return keepalives
        self._event({"type": "response.completed", "response": {"id": rid, "object": "response", "status": "completed",
                                                                 "output": items, "usage": _usage_responses(reply)}},
                    "response.completed")
        return keepalives


class StubServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, probe: Probe, port: int) -> None:
        super().__init__(("127.0.0.1", port), Handler)
        self.probe = probe


def start_server(script: dict[str, list[Reply]], journal: Path, port: int = 0) -> tuple[StubServer, threading.Thread]:
    probe = Probe(script, journal)
    server = StubServer(probe, port)
    thread = threading.Thread(target=server.serve_forever, name="dynorch-stub", daemon=True)
    thread.start()
    return server, thread


# ----------------------------------------------------------------------------------------------- case runner

def _fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _home_marker() -> Path:
    marker = PROBE_ROOT / "home_marker"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(time.time()))
    return marker


def _home_writes_since(marker: Path) -> list[str]:
    """Files under the real /home directory created or modified after the marker (this session's own dirs excluded)."""
    real_home = Path("/home/mabdel03")
    prune = [".claude/projects", ".claude/session-env", ".claude/file-history", ".claude/history.jsonl", ".claude/backups",
             ".claude/ide", ".claude/plans", ".claude/debug", ".claude/statsig", ".claude/todos", ".claude/shell-snapshots",
             ".claude/tasks", ".claude/cache"]
    cmd = ["find", str(real_home), "-xdev", "-maxdepth", "4"]
    for p in prune:
        cmd += ["-path", str(real_home / p), "-prune", "-o"]
    cmd += ["-newer", str(marker), "-print"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600).stdout
    except subprocess.TimeoutExpired:
        return ["<find timed out>"]
    return [line for line in out.splitlines() if line and line != str(real_home)]


def _write_capture(out_dir: Path, *, argv: list[str], env: dict[str, str], stdout: bytes, stderr: bytes,
                   exit_code: int | None, timed_out: bool, wall: float, server: StubServer, marker: Path,
                   extra: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stdout").write_bytes(stdout[:BODY_TRIM * 4])
    (out_dir / "stderr").write_bytes(stderr[:BODY_TRIM * 4])
    (out_dir / "argv.json").write_text(json.dumps([redact_text(a) for a in argv], indent=1, ensure_ascii=False))
    env_view = {k: ("[REDACTED]" if k.endswith("_KEY") or k.endswith("TOKEN") else v) for k, v in sorted(env.items())}
    (out_dir / "env.json").write_text(json.dumps(env_view, indent=1))
    summary = {"exit_code": exit_code, "timed_out": timed_out, "wall_seconds": round(wall, 2),
               "requests": len(server.probe.records), "requests_by_role": dict(server.probe.counts),
               "routes": sorted({r["path"].rsplit("/v1/", 1)[-1] for r in server.probe.records if "/v1/" in r["path"]}),
               "unmatched_paths": [r["path"] for r in server.probe.records if r["reply"]["kind"] == "not_found"],
               "home_writes_since_marker": _home_writes_since(marker), **extra}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def _run(argv: list[str], env: dict[str, str], cwd: Path, stdin: bytes, wall: float) -> tuple[bytes, bytes, int | None, bool, float]:
    started = time.monotonic()
    proc = subprocess.Popen(argv, env=env, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        out, err = proc.communicate(stdin, timeout=wall)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, 15)
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)
            out, err = proc.communicate()
    return out, err, proc.returncode, timed_out, time.monotonic() - started


def run_openclaude(case: str, *, delay: float, wall: float, port: int, minimal_config: bool = False,
                   label: str | None = None, extra_env: tuple[str, ...] = ()) -> dict[str, Any]:
    """Plan §6 OpenClaude invocation against the stub; captures under tests/fixtures/dynorch/captures/openclaude/<case>.

    ``minimal_config`` drops the plan's ``agentModels``/``agentRouting`` keys from ``.openclaude.json`` and
    ``extra_env`` adds ``KEY=VALUE`` pairs to the harness environment (both are experiments, recorded in the doc).
    """
    label = label or case
    key = f"probe-oc-{label}"
    probe_dir = _fresh(PROBE_ROOT / "openclaude" / label)
    home, cfg, ws, tmp = probe_dir / "home", probe_dir / "home" / "cfg", probe_dir / "ws", probe_dir / "tmp"
    for d in (home, cfg, ws, tmp):
        d.mkdir(parents=True, exist_ok=True)
    out_dir = _fresh(CAPTURES / "openclaude" / label)
    server, _ = start_server(openclaude_script(case, delay), out_dir / "requests.jsonl", port)
    base_url = f"http://127.0.0.1:{server.server_address[1]}/r/{key}/v1"
    config: dict[str, Any] = {"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True, "theme": "dark"}
    if not minimal_config:
        config["agentModels"] = {MODEL: {"base_url": base_url, "api_key": key}}
        config["agentRouting"] = {"default": MODEL, "solver": MODEL, "general-purpose": MODEL}
    (cfg / ".openclaude.json").write_text(json.dumps(config, indent=1))
    agents = {"solver": {"description": "Independent solver. Give it the FULL problem statement and a precise sub-task; "
                                        "it starts with no context.",
                         "prompt": subagent_text(), "tools": []}}
    (probe_dir / "agents.json").write_text(json.dumps(agents))
    task = task_text()
    (probe_dir / "task.txt").write_text(task)
    argv = [OPENCLAUDE_BIN, "-p", "--output-format", "json", "--no-session-persistence", "--provider", "openai",
            "--model", MODEL, "--system-prompt", system_text(), "--tools", "Agent", "--allowedTools", "Agent",
            "--disallowedTools", *OPENCLAUDE_DISALLOWED, "--permission-mode", "bypassPermissions",
            "--agents", json.dumps(agents), "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands"]
    env = {"PATH": f"{CONSORTIUM_BIN}:/usr/bin:/bin", "HOME": str(home), "CLAUDE_CONFIG_DIR": str(cfg),
           "TMPDIR": str(tmp), "LANG": "C.UTF-8", "TERM": "dumb",
           "CLAUDE_CODE_USE_OPENAI": "1", "OPENAI_BASE_URL": base_url, "OPENAI_API_KEY": key, "OPENAI_MODEL": MODEL,
           "CLAUDE_CODE_SUBAGENT_MODEL": MODEL, "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
           "CLAUDE_CODE_AGENT_LIST_IN_MESSAGES": "0", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
           "DISABLE_TELEMETRY": "1", "DISABLE_AUTOUPDATER": "1", "DISABLE_ERROR_REPORTING": "1", "DISABLE_COMPACT": "1",
           "CLAUDE_CODE_MAX_RETRIES": "1", "API_TIMEOUT_MS": "3600000", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000"}
    for pair in extra_env:
        name, _, value = pair.partition("=")
        env[name] = value
    marker = _home_marker()
    out, err, code, timed_out, elapsed = _run(argv, env, ws, task.encode(), wall)
    server.shutdown()
    extra = openclaude_facts(server.probe.records, out, err, system_text(), probe_dir)
    summary = _write_capture(out_dir, argv=argv, env=env, stdout=out, stderr=err, exit_code=code, timed_out=timed_out,
                             wall=elapsed, server=server, marker=marker, extra=extra)
    return summary


def openclaude_facts(records: list[dict[str, Any]], stdout: bytes, stderr: bytes, system: str, probe_dir: Path) -> dict[str, Any]:
    """The facts the brief asks for, computed from the journal and stdout (documented in harness_wire_observations.md)."""
    chat = [r for r in records if r["path"].endswith("/chat/completions") and isinstance(r.get("body"), dict)]
    facts: dict[str, Any] = {"models_get_called": any(r["path"].endswith("/v1/models") for r in records)}
    if chat:
        first = chat[0]["body"]
        facts["first_request"] = {
            "tools": [t.get("function", {}).get("name") for t in first.get("tools") or []],
            "stream": first.get("stream"), "max_tokens": first.get("max_tokens"),
            "max_completion_tokens": first.get("max_completion_tokens"),
            "stream_options": first.get("stream_options"), "model": first.get("model"),
            "top_level_keys": sorted(first.keys()),
            "messages0_role": (first.get("messages") or [{}])[0].get("role"),
            "system_byte_identical": (first.get("messages") or [{}])[0].get("content") == system,
            "system_byte_identical_with_trailing_newline": (first.get("messages") or [{}])[0].get("content") == system + "\n",
            "n_messages": len(first.get("messages") or []),
        }
    resend: list[dict[str, Any]] = []
    for r in chat:
        for m in r["body"].get("messages") or []:
            if m.get("role") == "assistant":
                resend.append({"seq": r["seq"], "role_of_request": r.get("role"), "has_reasoning_content": "reasoning_content" in m,
                               "keys": sorted(m.keys()), "tool_call_ids": [c.get("id") for c in m.get("tool_calls") or []]})
    facts["assistant_messages_resent"] = resend
    facts["tool_messages"] = [{"seq": r["seq"], "tool_call_id": m.get("tool_call_id"), "content_prefix": str(m.get("content"))[:120]}
                              for r in chat for m in r["body"].get("messages") or [] if m.get("role") == "tool"]
    facts["tool_calls_issued"] = [{"seq": r["seq"], "role": r.get("role"), "kind": r["reply"]["kind"], "tool": r["reply"].get("tool_name")}
                                  for r in records if "reply" in r and r["reply"].get("kind") in {"tool_call", "http_error"}]
    facts["child_requests"] = [{"seq": r["seq"], "tools": [t.get("function", {}).get("name") for t in r["body"].get("tools") or []],
                                "system_sha256": r.get("classification", {}).get("system_sha256"),
                                "system_prefix": str((r["body"].get("messages") or [{}])[0].get("content"))[:160],
                                "n_messages": len(r["body"].get("messages") or [])}
                               for r in chat if r.get("role") == "child"]
    facts["budget_usd_text_in_requests"] = any("budget_usd" in json.dumps(r.get("body")) for r in records)
    facts["budget_usd_text_in_stdout"] = b"budget_usd" in stdout
    text = stdout.decode("utf-8", errors="replace")
    result: Any = None
    try:
        result = json.loads(text) if text.strip() else None
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            try:
                result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if isinstance(result, dict):
        facts["stdout_json"] = {k: result.get(k) for k in ("type", "subtype", "is_error", "result", "num_turns", "usage",
                                                            "total_cost_usd", "stop_reason", "duration_ms", "duration_api_ms",
                                                            "modelUsage", "permission_denials")}
        facts["stdout_json_keys"] = sorted(result.keys())
    else:
        facts["stdout_json"] = None
    facts["stderr_prefix"] = redact_text(stderr.decode("utf-8", errors="replace")[:2000])
    facts["probe_dir_files"] = sorted(str(p.relative_to(probe_dir)) for p in probe_dir.rglob("*") if p.is_file())
    return facts


CODEX_CONFIG = """model = "{model}"
model_provider = "meter"
model_context_window = 131072
model_reasoning_summary = "none"
model_instructions_file = "{instructions}"
show_raw_agent_reasoning = false
approval_policy = "never"
sandbox_mode = "read-only"
web_search = "disabled"
check_for_update_on_startup = false
{extra}
[model_providers.meter]
name = "Local metering proxy"
base_url = "{base_url}"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
request_max_retries = 2
stream_max_retries = 2
stream_idle_timeout_ms = 1800000

[features]
shell_tool = false
unified_exec = false
view_image = false
code_mode = false
apps = false
plugins = false
hooks = false
image_generation = false
skill_search = false
multi_agent = true
multi_agent_v2 = false

[agents]
enabled = true
max_concurrent_threads_per_session = 4
max_depth = 1

[shell_environment_policy]
inherit = "none"
"""


def run_codex(case: str, *, delay: float, wall: float, port: int, strace: bool = False,
              config_extra: str = "", cli_config: tuple[str, ...] = (), label: str | None = None) -> dict[str, Any]:
    """Plan §6 Codex invocation (config.toml + `codex exec --json …`) against the stub; captures under captures/codex/<case>.

    ``config_extra`` appends top-level TOML lines and ``cli_config`` adds ``-c key=value`` overrides (both used for the
    config experiments recorded in the observations doc); ``label`` names the capture directory when it differs from
    the script case.
    """
    label = label or case
    key = f"probe-cx-{label}"
    probe_dir = _fresh(PROBE_ROOT / "codex" / label)
    home, ws, tmp = probe_dir / "home", probe_dir / "ws", probe_dir / "tmp"
    for d in (home, ws, tmp):
        d.mkdir(parents=True, exist_ok=True)
    out_dir = _fresh(CAPTURES / "codex" / label)
    server, _ = start_server(codex_script(case, delay), out_dir / "requests.jsonl", port)
    base_url = f"http://127.0.0.1:{server.server_address[1]}/r/{key}/v1"
    extra = ('this_key_does_not_exist = 1\n' if case == "g" else "") + config_extra
    (home / "config.toml").write_text(CODEX_CONFIG.format(model=MODEL, instructions=PROMPTS / "system_dynorch.txt",
                                                          base_url=base_url, extra=extra))
    task = task_text()
    (probe_dir / "task.txt").write_text(task)
    final = probe_dir / "final.txt"
    argv = ["timeout", "--kill-after=30s", f"{int(wall)}s"]
    if strace:
        argv += ["strace", "-f", "-o", str(probe_dir / "strace.txt"), "-e", "trace=execve,clone,clone3,vfork,fork"]
    argv += [str(CODEX_BIN), "exec", "--json", "--ephemeral", "--strict-config", "--skip-git-repo-check", "-C", str(ws),
             "-s", "read-only", "-o", str(final)]
    for override in cli_config:
        argv += ["-c", override]
    argv.append("-")
    env = {"PATH": f"{CONSORTIUM_BIN}:/usr/bin:/bin", "HOME": str(home), "CODEX_HOME": str(home), "TMPDIR": str(tmp),
           "LANG": "C.UTF-8", "TERM": "dumb", "OPENAI_API_KEY": key}
    marker = _home_marker()
    out, err, code, timed_out, elapsed = _run(argv, env, ws, task.encode(), wall + 45)
    server.shutdown()
    facts = codex_facts(server.probe.records, out, err, final, probe_dir)
    summary = _write_capture(out_dir, argv=argv, env=env, stdout=out, stderr=err, exit_code=code, timed_out=timed_out,
                             wall=elapsed, server=server, marker=marker, extra=facts)
    (out_dir / "config.toml").write_text((home / "config.toml").read_text())
    return summary


def codex_facts(records: list[dict[str, Any]], stdout: bytes, stderr: bytes, final: Path, probe_dir: Path) -> dict[str, Any]:
    """The Codex facts the brief asks for (tool list, request shape, events, -o file, warnings, /home writes)."""
    reqs = [r for r in records if r["path"].endswith("/v1/responses") and isinstance(r.get("body"), dict)]
    facts: dict[str, Any] = {"models_get_called": any(r["path"].endswith("/v1/models") for r in records)}

    def tool_names(tools: list[Any]) -> list[str]:
        names: list[str] = []
        for t in tools or []:
            if t.get("type") == "namespace":
                names += [f"{t.get('name')}::{s.get('name')}" for s in t.get("tools") or []]
            else:
                names.append(f"{t.get('type')}:{t.get('name')}")
        return names

    if reqs:
        first = reqs[0]["body"]
        facts["first_request"] = {
            "top_level_keys": sorted(first.keys()), "model": first.get("model"), "stream": first.get("stream"),
            "store": first.get("store"), "include": first.get("include"), "reasoning": first.get("reasoning"),
            "text": first.get("text"), "tool_choice": first.get("tool_choice"),
            "parallel_tool_calls": first.get("parallel_tool_calls"), "prompt_cache_key_present": "prompt_cache_key" in first,
            "stream_options": first.get("stream_options"), "max_output_tokens": first.get("max_output_tokens"),
            "instructions_chars": len(first.get("instructions") or ""),
            "instructions_sha256": _sha(first.get("instructions") or ""),
            "instructions_byte_identical_to_file": (first.get("instructions") or "") == (PROMPTS / "system_dynorch.txt").read_text(),
            "instructions_prefix": (first.get("instructions") or "")[:200],
            "instructions_suffix": (first.get("instructions") or "")[-400:],
            "tools": tool_names(first.get("tools") or []),
            "input_item_types": [f"{i.get('type')}:{i.get('role', '')}" for i in first.get("input") or []],
            "n_input": len(first.get("input") or []),
        }
        forbidden = {"exec_command", "shell", "apply_patch", "web_search", "view_image", "write_stdin", "local_shell"}
        present = {n.split(":")[-1] for n in facts["first_request"]["tools"]}
        facts["forbidden_tools_present"] = sorted(present & forbidden)
        facts["multi_agent_v1_present"] = any(n.startswith("multi_agent_v1::") for n in facts["first_request"]["tools"])
    facts["requests"] = [{"seq": r["seq"], "role": r.get("role"), "x_openai_subagent": r.get("classification", {}).get("x_openai_subagent"),
                          "n_input": len(r["body"].get("input") or []),
                          "input_item_types": [f"{i.get('type')}:{i.get('role', i.get('name', ''))}" for i in r["body"].get("input") or []],
                          "tools": tool_names(r["body"].get("tools") or []), "reply": r["reply"],
                          "headers_of_interest": {k: v for k, v in r["headers"].items()
                                                  if k in {"x-openai-subagent", "accept", "content-type", "user-agent", "session_id",
                                                           "x-client-request-id", "originator", "accept-encoding", "content-encoding",
                                                           "conversation_id", "x-codex-turn-id", "x-codex-beta-features"}}}
                         for r in reqs]
    facts["function_call_outputs"] = [
        {"seq": r["seq"], "call_id": i.get("call_id"), "output": (i.get("output") if isinstance(i.get("output"), str) else json.dumps(i.get("output")))[:600]}
        for r in reqs for i in r["body"].get("input") or [] if i.get("type") == "function_call_output"]
    facts["function_calls_echoed"] = [
        {"seq": r["seq"], "call_id": i.get("call_id"), "name": i.get("name"), "namespace": i.get("namespace"), "keys": sorted(i.keys())}
        for r in reqs for i in r["body"].get("input") or [] if i.get("type") == "function_call"]
    events: list[dict[str, Any]] = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"_raw": line[:300]})
    facts["event_types"] = [e.get("type", "_raw") + ("/" + str(e.get("item", {}).get("type")) if e.get("item") else "") for e in events]
    facts["turn_completed_usage"] = [e.get("usage") for e in events if e.get("type") == "turn.completed"]
    facts["turn_failed"] = [e.get("error") for e in events if e.get("type") == "turn.failed"]
    facts["error_events"] = [e for e in events if e.get("type") == "error"]
    facts["items"] = [e.get("item") for e in events if e.get("type") == "item.completed"]
    facts["final_file"] = final.read_text() if final.exists() else None
    err = redact_text(stderr.decode("utf-8", errors="replace"))
    facts["stderr_lines"] = [line for line in err.splitlines() if line.strip()][:60]
    facts["bwrap_mentioned_in_stderr"] = "bwrap" in err or "bubblewrap" in err
    facts["unknown_model_warning"] = "Unknown model" in err
    facts["probe_dir_files"] = sorted(str(p.relative_to(probe_dir)) for p in probe_dir.rglob("*") if p.is_file())
    strace_file = probe_dir / "strace.txt"
    if strace_file.exists():
        execs = [line for line in strace_file.read_text(errors="replace").splitlines() if "execve(" in line]
        facts["strace_execve"] = [redact_text(line)[:200] for line in execs][:40]
    return facts


def summarize(harness: str) -> None:
    root = CAPTURES / harness
    for case in sorted(p.name for p in root.iterdir() if p.is_dir()):
        summary = json.loads((root / case / "summary.json").read_text())
        print(f"{harness}/{case}: exit={summary['exit_code']} timed_out={summary['timed_out']} wall={summary['wall_seconds']} "
              f"requests={summary['requests']} roles={summary['requests_by_role']} routes={summary['routes']} "
              f"home_writes={len(summary['home_writes_since_marker'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run the stub server in the foreground")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--harness", choices=["openclaude", "codex"], default="openclaude")
    serve.add_argument("--case", default="b")
    serve.add_argument("--delay", type=float, default=0.0)
    serve.add_argument("--journal", type=Path, default=PROBE_ROOT / "serve_requests.jsonl")
    run = sub.add_parser("run", help="run one real-CLI case against the stub and capture everything")
    run.add_argument("harness", choices=["openclaude", "codex"])
    run.add_argument("case")
    run.add_argument("--delay", type=float, default=330.0)
    run.add_argument("--wall", type=float, default=600.0)
    run.add_argument("--port", type=int, default=0)
    run.add_argument("--strace", action="store_true")
    run.add_argument("--cli-config", action="append", default=[], help="codex only: extra `-c key=value` override")
    run.add_argument("--config-extra", default="", help="codex only: extra top-level TOML lines")
    run.add_argument("--label", default=None, help="capture directory name when it differs from the case")
    run.add_argument("--minimal-config", action="store_true", help="openclaude only: no agentModels/agentRouting")
    run.add_argument("--extra-env", action="append", default=[], help="openclaude only: extra KEY=VALUE for the harness")
    summ = sub.add_parser("summarize")
    summ.add_argument("harness", choices=["openclaude", "codex"])
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        script = openclaude_script(args.case, args.delay) if args.harness == "openclaude" else codex_script(args.case, args.delay)
        server, thread = start_server(script, args.journal, args.port)
        print(f"stub listening on http://127.0.0.1:{server.server_address[1]}/r/<key>/v1  journal={args.journal}", flush=True)
        try:
            thread.join()
        except KeyboardInterrupt:
            server.shutdown()
        return 0
    if args.cmd == "summarize":
        summarize(args.harness)
        return 0
    if args.harness == "openclaude":
        summary = run_openclaude(args.case, delay=args.delay, wall=args.wall, port=args.port,
                                 minimal_config=args.minimal_config, label=args.label, extra_env=tuple(args.extra_env))
    else:
        summary = run_codex(args.case, delay=args.delay, wall=args.wall, port=args.port, strace=args.strace,
                            config_extra=args.config_extra.replace("\\n", "\n"), cli_config=tuple(args.cli_config),
                            label=args.label)
    print(json.dumps({k: summary[k] for k in ("exit_code", "timed_out", "wall_seconds", "requests", "requests_by_role", "routes",
                                              "unmatched_paths", "home_writes_since_marker")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
