"""WP-B: OpenAI chat-completions wire format <-> the internal message model (contract §2, §6; plan §5).

``parse_chat_body`` turns the JSON body OpenClaude / the fake harness POST to ``/r/<run_key>/v1/chat/completions``
into a ``ChatRequest``; ``chat_reply_json`` / ``chat_reply_sse`` render a ``ChatReply`` as the non-streaming object or
the exact streaming chunk sequence; ``error_json`` converts a ``ProxyError``. Rules enforced here:

* the body keys the proxy acts on are ``messages``, ``tools``, ``max_tokens`` / ``max_completion_tokens``,
  ``reasoning_effort`` / ``reasoning.effort``, ``stream`` and ``n``; every other key (``model``, ``temperature``,
  ``top_p``, ``stop``, ``stream_options``, ``store``, ...) is accepted and recorded in ``ChatRequest.ignored_keys``
  (the study's sampling block is authoritative, plan §2 B2; the harness's ``max_tokens`` is recorded and ignored);
* ``n`` other than 1 -> ``bad_request`` (400); a role outside {system, developer, user, assistant, tool}, a content
  part other than text, a non-function tool, a tool message without ``tool_call_id``, a non-bool ``stream`` and a
  non-object body all raise ``bad_request`` -- an unexpected shape never degrades silently;
* ``tool_calls[].function.arguments`` is parsed to a dict when it is a JSON object string (or already an object);
  anything else keeps the raw string (contract §2 ``ToolCall.arguments``);
* the streamed chunk sequence is: role chunk, ``reasoning_content`` delta (when non-empty), ``content`` delta (when
  non-empty), one ``tool_calls`` delta per call carrying ``index``/``id``/``name``/``arguments``, the finish chunk,
  a usage chunk ALWAYS (whatever ``stream_options`` said: the fake harness, OpenClaude on a loopback URL and
  OpenClaw all tolerate it and the runner cross-checks usage from it), then ``data: [DONE]``;
* ``finish_reason`` is whatever the core returned -- the core guarantees {stop, tool_calls} (plan §2 B3).
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterable, Mapping

from agent_allocation.dynorch.records import (
    ChatReply, ChatRequest, Message, ProxyError, ReplyUsage, ToolCall, ToolSpec,
)

__all__ = [
    "CONSUMED_CHAT_KEYS", "ROLES", "chat_reply_json", "chat_reply_sse", "error_json", "error_sse", "parse_chat_body",
    "parse_tool_arguments", "tool_arguments_json", "usage_json",
]

#: Body keys the proxy acts on; everything else is recorded as ignored (contract §2 ``ignored_keys``).
CONSUMED_CHAT_KEYS: frozenset[str] = frozenset({
    "messages", "tools", "max_tokens", "max_completion_tokens", "reasoning_effort", "reasoning", "stream", "n",
})
ROLES: frozenset[str] = frozenset({"system", "developer", "user", "assistant", "tool"})
_TEXT_PART_TYPES: frozenset[str] = frozenset({"text", "input_text", "output_text"})
DONE_FRAME = b"data: [DONE]\n\n"


def _bad(message: str) -> ProxyError:
    return ProxyError(400, "bad_request", message, False)


def _expect(value: Any, kind: type | tuple[type, ...], what: str) -> Any:
    if not isinstance(value, kind) or isinstance(value, bool) and kind is int:
        raise _bad(f"{what} must be {getattr(kind, '__name__', kind)}, got {type(value).__name__}")
    return value


def parse_tool_arguments(raw: Any) -> dict[str, Any] | str:
    """Contract §2: a dict when the arguments are a JSON object (string or already parsed), else the raw string."""
    if isinstance(raw, dict):
        return dict(raw)
    if raw is None:
        return {}
    if not isinstance(raw, str):
        raise _bad(f"tool call arguments must be a JSON string or object, got {type(raw).__name__}")
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    return dict(parsed) if isinstance(parsed, dict) else raw


def tool_arguments_json(arguments: dict[str, Any] | str) -> str:
    """The wire form of ``ToolCall.arguments``: a JSON object string, or the raw string verbatim."""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def _content_text(content: Any, what: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            part = _expect(part, dict, f"{what} content part")
            kind = part.get("type")
            if kind not in _TEXT_PART_TYPES:
                raise _bad(f"{what}: unsupported content part type {kind!r} (text only)")
            parts.append(_expect(part.get("text", ""), str, f"{what} content part text"))
        return "".join(parts)
    raise _bad(f"{what} content must be a string or a list of text parts, got {type(content).__name__}")


def _parse_message(index: int, raw: Any) -> Message:
    raw = _expect(raw, dict, f"messages[{index}]")
    role = raw.get("role")
    if role not in ROLES:
        raise _bad(f"messages[{index}]: unsupported role {role!r}")
    what = f"messages[{index}] ({role})"
    content = _content_text(raw.get("content"), what)
    name = raw.get("name")
    if name is not None:
        name = _expect(name, str, f"{what} name")
    if role == "developer":
        role = "system"
    if role == "tool":
        call_id = raw.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise _bad(f"{what}: tool messages need a non-empty tool_call_id")
        return Message(role="tool", content=content, tool_call_id=call_id, name=name)
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning: str | None = None
    if role == "assistant":
        raw_calls = raw.get("tool_calls")
        if raw_calls is not None:
            calls: list[ToolCall] = []
            for j, call in enumerate(_expect(raw_calls, list, f"{what} tool_calls")):
                call = _expect(call, dict, f"{what} tool_calls[{j}]")
                if call.get("type", "function") != "function":
                    raise _bad(f"{what} tool_calls[{j}]: unsupported type {call.get('type')!r}")
                fn = _expect(call.get("function"), dict, f"{what} tool_calls[{j}].function")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise _bad(f"{what} tool_calls[{j}]: missing id")
                fn_name = fn.get("name")
                if not isinstance(fn_name, str) or not fn_name:
                    raise _bad(f"{what} tool_calls[{j}]: missing function.name")
                calls.append(ToolCall(id=call_id, name=fn_name, arguments=parse_tool_arguments(fn.get("arguments"))))
            tool_calls = tuple(calls)
        raw_reasoning = raw.get("reasoning_content")
        if raw_reasoning is not None:
            reasoning = _expect(raw_reasoning, str, f"{what} reasoning_content")
    elif "tool_calls" in raw and raw["tool_calls"]:
        raise _bad(f"{what}: only assistant messages may carry tool_calls")
    return Message(role=role, content=content, reasoning_content=reasoning, tool_calls=tool_calls, name=name)


def _parse_tool(index: int, raw: Any) -> ToolSpec:
    raw = _expect(raw, dict, f"tools[{index}]")
    if raw.get("type") != "function":
        raise _bad(f"tools[{index}]: unsupported tool type {raw.get('type')!r} (function only)")
    fn = _expect(raw.get("function"), dict, f"tools[{index}].function")
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        raise _bad(f"tools[{index}]: missing function.name")
    description = fn.get("description", "")
    if description is None:
        description = ""
    description = _expect(description, str, f"tools[{index}].function.description")
    parameters = fn.get("parameters")
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
    parameters = _expect(parameters, dict, f"tools[{index}].function.parameters")
    return ToolSpec(name=name, description=description, parameters=dict(parameters), namespace=None)


def parse_chat_body(body: Mapping[str, Any]) -> ChatRequest:
    """OpenAI chat body -> ``ChatRequest`` (contract §6); every unexpected shape raises ``bad_request``."""
    body = _expect(body, dict, "request body")
    n = body.get("n", 1)
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n != 1):
        raise _bad(f"n must be 1, got {n!r}")
    raw_messages = _expect(body.get("messages"), list, "messages")
    if not raw_messages:
        raise _bad("messages must not be empty")
    messages = tuple(_parse_message(i, m) for i, m in enumerate(raw_messages))
    raw_tools = body.get("tools")
    tools = tuple(_parse_tool(i, t) for i, t in enumerate(_expect(raw_tools, list, "tools"))) if raw_tools else ()
    stream = body.get("stream", False)
    if stream is None:
        stream = False
    if not isinstance(stream, bool):
        raise _bad(f"stream must be a boolean, got {stream!r}")
    requested: int | None = None
    for key in ("max_tokens", "max_completion_tokens"):      # the newer key wins when both are sent
        value = body.get(key)
        if value is not None:
            requested = _expect(value, int, key)
    effort: str | None = None
    if body.get("reasoning_effort") is not None:
        effort = _expect(body["reasoning_effort"], str, "reasoning_effort")
    elif isinstance(body.get("reasoning"), dict) and body["reasoning"].get("effort") is not None:
        effort = _expect(body["reasoning"]["effort"], str, "reasoning.effort")
    ignored = tuple(sorted(k for k in body if k not in CONSUMED_CHAT_KEYS))
    return ChatRequest(messages=messages, tools=tools, requested_max_tokens=requested,
                       requested_reasoning_effort=effort, stream=stream, ignored_keys=ignored, wire="chat")


# ---------------------------------------------------------------------------------------------------------
# Replies
# ---------------------------------------------------------------------------------------------------------

def usage_json(usage: ReplyUsage) -> dict[str, Any]:
    """OpenAI ``usage`` object: prompt/completion/total plus the details blocks both harnesses read."""
    prompt = usage.prompt_tokens or 0
    completion = usage.completion_tokens or 0
    out: dict[str, Any] = {
        "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens or 0},
        "prompt_tokens_details": {"cached_tokens": usage.cached_prompt_tokens or 0},
    }
    return out


def _tool_call_json(index: int, call: ToolCall, *, with_index: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"id": call.id, "type": "function",
                           "function": {"name": call.name, "arguments": tool_arguments_json(call.arguments)}}
    if with_index:
        out = {"index": index, **out}
    return out


def _dynorch_extension(reply: ChatReply) -> dict[str, Any]:
    call = reply.call
    return {"call_no": call.call_no, "conversation_class": call.conversation_class,
            "granted_max_tokens": call.granted_max_tokens, "forced_close_kind": call.forced_close_kind,
            "grace_tokens": call.grace_tokens, "tool_parse": call.tool_parse, "reinjection": call.reinjection,
            "duplicate_served": call.duplicate_served,
            "headers": {k.lower(): v for k, v in reply.headers.items()}}


def chat_reply_json(reply: ChatReply, *, model: str = "qwen/qwen3.8-27b", response_id: str | None = None,
                    created: int | None = None) -> dict[str, Any]:
    """The non-streaming ``chat.completion`` object (contract §6), plus a ``dynorch`` diagnostics extension."""
    message: dict[str, Any] = {"role": "assistant", "content": reply.content,
                               "reasoning_content": reply.reasoning_content}
    if reply.tool_calls:
        message["tool_calls"] = [_tool_call_json(i, c, with_index=False) for i, c in enumerate(reply.tool_calls)]
    return {
        "id": response_id or _response_id(reply),
        "object": "chat.completion",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": reply.finish_reason, "logprobs": None}],
        "usage": usage_json(reply.usage),
        "dynorch": _dynorch_extension(reply),
    }


def _response_id(reply: ChatReply) -> str:
    return f"chatcmpl-dynorch-{reply.call.conversation_key[:12]}-{reply.call.call_no}"


def _frame(payload: Mapping[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n"


def chat_reply_sse(reply: ChatReply, *, model: str = "qwen/qwen3.8-27b", response_id: str | None = None,
                   created: int | None = None) -> Iterable[bytes]:
    """The streaming chunk sequence (module docstring): role, reasoning, content, tool calls, finish, usage, DONE."""
    rid = response_id or _response_id(reply)
    stamp = created if created is not None else int(time.time())

    def chunk(delta: Mapping[str, Any] | None, finish: str | None = None, *, usage: Mapping[str, Any] | None = None,
              choices: bool = True) -> bytes:
        payload: dict[str, Any] = {"id": rid, "object": "chat.completion.chunk", "created": stamp, "model": model}
        if choices:
            payload["choices"] = [{"index": 0, "delta": dict(delta or {}), "finish_reason": finish, "logprobs": None}]
        else:
            payload["choices"] = []
        payload["usage"] = dict(usage) if usage is not None else None
        return _frame(payload)

    yield chunk({"role": "assistant", "content": ""})
    if reply.reasoning_content:
        yield chunk({"reasoning_content": reply.reasoning_content})
    if reply.content:
        yield chunk({"content": reply.content})
    for i, call in enumerate(reply.tool_calls):
        yield chunk({"tool_calls": [_tool_call_json(i, call, with_index=True)]})
    yield chunk({}, reply.finish_reason)
    yield chunk(None, usage=usage_json(reply.usage), choices=False)
    yield DONE_FRAME


def error_json(err: ProxyError) -> tuple[int, dict[str, Any]]:
    """``(http status, OpenAI-shaped error body)``; ``code`` carries the contract §2 code string verbatim."""
    return err.status, {"error": {"message": err.message, "type": err.code, "code": err.code, "param": None,
                                  "retryable": bool(err.retryable), "status": err.status}}


def error_sse(err: ProxyError) -> Iterable[bytes]:
    """An in-stream error (headers already sent): one error frame, then ``[DONE]``. OpenClaude throws on it."""
    _, body = error_json(err)
    yield _frame(body)
    yield DONE_FRAME
