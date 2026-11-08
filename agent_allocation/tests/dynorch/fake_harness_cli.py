#!/usr/bin/env python3
"""WP-B: a stdlib-only fake harness that talks to the dynorch proxy the way OpenClaude does (plan §6 fake harness).

Protocol (chat completions over ``OPENAI_BASE_URL`` with ``Authorization: Bearer $OPENAI_API_KEY``):

1. root conversation = ``[system (SYSTEM file), user (stdin)]`` streamed (``stream: true``), offering one
   ``spawn_subagent(task)`` tool (or none when ``--tools`` is empty); the SSE reader ignores comment lines
   (``: dynorch keepalive``) and accumulates ``delta.reasoning_content`` / ``delta.content`` /
   ``delta.tool_calls[index]`` until ``data: [DONE]``;
2. every returned tool call is executed by opening a *second* conversation ``[system (SOLVER file), user (task)]``
   with no tools, whose visible content becomes the ``tool`` message (``tool_call_id`` echoed back); with
   ``--tolerate-tool-errors`` a failed child call becomes a tool error message and the root keeps going (what a
   real orchestrator does with a fallible sub-agent);
3. the assistant turn is re-sent WITHOUT ``reasoning_content`` (OpenClaude does not re-send it either, so the
   proxy's reasoning re-injection is exercised) and the loop continues until a reply has no tool calls or
   ``--max-turns`` is hit;
4. stdout gets one OpenClaude-shaped line ``{"type":"result","subtype":"success","result":<last content>,
   "num_turns":N,"usage":{...}}``; an HTTP error (402 after exhaustion, 400, 5xx) or an in-stream error frame ends
   the run with ``subtype: "error_during_execution"``, ``is_error: true`` and exit code 1 -- and no retry, like
   OpenClaude on a 402;
5. ``--retry-5xx N`` re-sends a request answered with a 5xx (HTTP status or in-stream error frame) up to N times,
   the way OpenClaude's ``withRetry`` does with ``CLAUDE_CODE_MAX_RETRIES=1`` and Codex with
   ``request_max_retries`` (contract §10.38: that retry is what used to reach the B4 TERMINAL call after an
   upstream failure had settled the run's whole grant).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
from typing import Any
from urllib.parse import urlsplit

SPAWN_TOOL = {
    "type": "function",
    "function": {
        "name": "spawn_subagent",
        "description": "Delegate a self-contained sub-task to a fresh solver that shares your token budget.",
        "parameters": {"type": "object", "properties": {"task": {"type": "string", "description": "the full task"}},
                       "required": ["task"]},
    },
}


class HarnessError(RuntimeError):
    def __init__(self, status: int, payload: Any) -> None:
        super().__init__(f"HTTP {status}: {payload}")
        self.status = status
        self.payload = payload


class Client:
    def __init__(self, base_url: str, key: str, model: str, *, timeout: float) -> None:
        parts = urlsplit(base_url)
        if parts.scheme != "http":
            raise SystemExit(f"only http base urls are supported: {base_url}")
        self.host, self.port = parts.hostname, parts.port or 80
        self.prefix = parts.path.rstrip("/")
        self.key, self.model, self.timeout = key, model, timeout
        self.usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
        self.requests = 0
        self.keepalives = 0
        self.retry_5xx = 0
        self.retries = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, stream: bool) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                return self._chat_once(messages, tools, stream=stream)
            except HarnessError as exc:
                if exc.status >= 500 and attempt < self.retry_5xx:
                    attempt += 1
                    self.retries += 1
                    continue
                raise

    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def _chat_once(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, stream: bool
                   ) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages, "stream": stream, "max_tokens": 32000, "temperature": 1.0}
        if tools:
            body["tools"] = tools
        data = json.dumps(body).encode("utf-8")
        conn = self._conn()
        self.requests += 1
        conn.request("POST", f"{self.prefix}/chat/completions", body=data,
                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                              "Accept": "text/event-stream" if stream else "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raw = response.read()
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = raw.decode("utf-8", "replace")
            raise HarnessError(response.status, payload)
        if not stream:
            payload = json.loads(response.read())
            message = payload["choices"][0]["message"]
            self._add_usage(payload.get("usage"))
            return {"content": message.get("content") or "", "reasoning_content": message.get("reasoning_content") or "",
                    "tool_calls": message.get("tool_calls") or [], "finish_reason": payload["choices"][0]["finish_reason"]}
        return self._read_sse(response)

    def _add_usage(self, usage: Any) -> None:
        if isinstance(usage, dict):
            for key in self.usage_total:
                self.usage_total[key] += int(usage.get(key) or 0)

    def _read_sse(self, response: http.client.HTTPResponse) -> dict[str, Any]:
        content, reasoning, finish = [], [], None
        calls: dict[int, dict[str, Any]] = {}
        keepalives = 0
        while True:
            line = response.readline()
            if not line:
                raise HarnessError(0, "stream closed before [DONE]")
            line = line.rstrip(b"\r\n")
            if not line:
                continue
            if line.startswith(b":"):
                keepalives += 1
                continue
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                raise HarnessError(int(chunk["error"].get("status") or 0), chunk)
            self._add_usage(chunk.get("usage"))
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])
                for call in delta.get("tool_calls") or []:
                    slot = calls.setdefault(int(call["index"]), {"id": "", "name": "", "arguments": ""})
                    if call.get("id"):
                        slot["id"] = call["id"]
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    slot["arguments"] += fn.get("arguments") or ""
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        tool_calls = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                      for _, c in sorted(calls.items())]
        self.keepalives += keepalives
        return {"content": "".join(content), "reasoning_content": "".join(reasoning), "tool_calls": tool_calls,
                "finish_reason": finish, "keepalives": keepalives}


def run(args: argparse.Namespace) -> int:
    base_url = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")
    if not base_url or not key:
        print(json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True,
                          "result": "OPENAI_BASE_URL / OPENAI_API_KEY missing"}))
        return 2
    system_text = open(args.system_file, encoding="utf-8").read()
    solver_text = open(args.solver_file, encoding="utf-8").read()
    task_text = sys.stdin.read()
    tools = [SPAWN_TOOL] if "spawn_subagent" in [t for t in args.tools.split(",") if t] else []
    client = Client(base_url, key, args.model, timeout=args.timeout)
    client.retry_5xx = max(0, int(args.retry_5xx))
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_text}, {"role": "user", "content": task_text}]
    turns, spawns, last_content = 0, 0, ""
    started = time.time()
    try:
        while turns < args.max_turns:
            reply = client.chat(messages, tools, stream=not args.no_stream)
            turns += 1
            if reply["content"]:
                last_content = reply["content"]
            assistant: dict[str, Any] = {"role": "assistant", "content": reply["content"] or None}
            if reply["tool_calls"]:
                assistant["tool_calls"] = reply["tool_calls"]
            messages.append(assistant)
            if not reply["tool_calls"]:
                break
            for call in reply["tool_calls"]:
                if call["function"]["name"] != "spawn_subagent":
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": f"Error: unknown tool {call['function']['name']}"})
                    continue
                try:
                    task = json.loads(call["function"]["arguments"]).get("task", "")
                except ValueError:
                    task = call["function"]["arguments"]
                spawns += 1
                try:
                    child = client.chat([{"role": "system", "content": solver_text},
                                         {"role": "user", "content": str(task)}], [], stream=not args.no_stream)
                except HarnessError as exc:
                    if not args.tolerate_tool_errors:
                        raise
                    # a real orchestrator treats a failed sub-agent as fallible data and keeps going
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": f"Error: the sub-agent failed (HTTP {exc.status})"})
                    continue
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": child["content"] or ""})
    except HarnessError as exc:
        print(json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True,
                          "result": f"API error {exc.status}: {json.dumps(exc.payload)[:500]}", "num_turns": turns,
                          "spawns": spawns, "usage": client.usage_total, "requests": client.requests,
                          "keepalives": client.keepalives, "retries": client.retries,
                          "duration_ms": int((time.time() - started) * 1000)}), flush=True)
        return 1
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": last_content,
                      "num_turns": turns, "spawns": spawns, "usage": client.usage_total, "requests": client.requests,
                      "retries": client.retries,
                      "keepalives": client.keepalives, "duration_ms": int((time.time() - started) * 1000)}), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-file", required=True)
    parser.add_argument("--solver-file", required=True)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--tools", default="spawn_subagent", help="comma-separated tool names to offer ('' = none)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="socket timeout per read (seconds)")
    parser.add_argument("--no-stream", action="store_true", help="use non-streaming requests")
    parser.add_argument("--tolerate-tool-errors", action="store_true",
                        help="a failed sub-agent call becomes a tool error message instead of ending the run")
    parser.add_argument("--retry-5xx", type=int, default=0,
                        help="re-send a request answered with a 5xx up to N times (OpenClaude/Codex retry)")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
