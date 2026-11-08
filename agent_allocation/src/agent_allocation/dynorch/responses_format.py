"""WP-B: OpenAI Responses API (Codex CLI 0.156.1) <-> the internal message model (contract §2, §6; plan §5).

Codex only speaks the Responses API over SSE (``wire_api = "chat"`` is a hard error since Feb 2026; Codex deep dive
§1, §6 and verifier C1/C6). ``parse_responses_body`` translates its request into the same ``ChatRequest`` the chat
route produces, so one scripted conversation yields identical ``CallRecord``s on both wires (parity test):

* ``instructions`` -> the system message; ``input`` items: ``message`` (roles ``user`` / ``assistant`` /
  ``developer`` -> system, content ``input_text`` / ``output_text``), ``function_call`` -> an assistant ``ToolCall``
  (``id = call_id``; attached to the immediately preceding assistant message of the same turn, else a new one),
  ``function_call_output`` -> a ``tool`` message with ``tool_call_id = call_id`` (``output`` string or text parts),
  ``reasoning`` -> ``reasoning_content`` of the next assistant message (``content[].reasoning_text`` joined; a
  summary-only item carries no text and yields ``None`` so the core's reasoning cache decides);
* ``tools``: ``{"type":"function", name, description, parameters, strict?}`` and ``{"type":"namespace", name,
  tools:[...]}`` groups, flattened with the group name kept on ``ToolSpec.namespace`` (Codex's ``multi_agent_v1``
  collab tools arrive this way); any other tool type -> ``bad_request`` (loud 400, never a silent drop);
* every other body key (``model``, ``store``, ``include``, ``prompt_cache_key``, ``text``, ``tool_choice``,
  ``parallel_tool_calls``, ``client_metadata``, ...) is recorded in ``ignored_keys``; ``max_output_tokens`` and
  ``reasoning.effort`` are recorded as the requested values (ignored by the core, plan §2 B2).

``responses_sse`` emits the minimum viable stream Codex's parser (``codex-api/src/sse/responses.rs``) consumes:
``response.created`` -> one ``response.output_item.done`` per item (``reasoning`` with the REQUIRED ``summary: []``
and ``content: [{type: "reasoning_text", text}]``; ``message`` with ``output_text``; ``function_call`` with
``call_id`` / ``name`` / ``arguments`` as a JSON string / ``namespace`` when the tool had one) ->
``response.completed`` with a string ``id`` and ``usage {input_tokens, output_tokens, total_tokens,
input_tokens_details.cached_tokens, output_tokens_details.reasoning_tokens}``. Errors after the headers went out
become ``response.failed {response: {error: {code, message}}}`` with ``rate_limit_exceeded`` for a retryable
upstream refusal (Codex retries), ``context_length_exceeded`` for the input guard, else ``server_error``.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from agent_allocation.dynorch.openai_format import (
    ROLES, parse_tool_arguments, tool_arguments_json,
)
from agent_allocation.dynorch.records import (
    ChatReply, ChatRequest, Message, ProxyError, ReplyUsage, ToolCall, ToolSpec,
)

__all__ = [
    "CONSUMED_RESPONSES_KEYS", "output_items", "parse_responses_body", "responses_error_code", "responses_failed_sse",
    "responses_json", "responses_sse", "responses_usage",
]

CONSUMED_RESPONSES_KEYS: frozenset[str] = frozenset({
    "instructions", "input", "tools", "stream", "reasoning", "max_output_tokens",
})
_INPUT_TEXT_TYPES = frozenset({"input_text", "output_text", "text"})
_MESSAGE_ROLES = frozenset({"user", "assistant", "developer", "system"})


def _bad(message: str) -> ProxyError:
    return ProxyError(400, "bad_request", message, False)


def _expect(value: Any, kind: type | tuple[type, ...], what: str) -> Any:
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is int):
        raise _bad(f"{what} must be {getattr(kind, '__name__', kind)}, got {type(value).__name__}")
    return value


def _text_of(content: Any, what: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            part = _expect(part, dict, f"{what} content part")
            kind = part.get("type")
            if kind not in _INPUT_TEXT_TYPES:
                raise _bad(f"{what}: unsupported content part type {kind!r} (text only)")
            parts.append(_expect(part.get("text", ""), str, f"{what} content part text"))
        return "".join(parts)
    raise _bad(f"{what} content must be a string or a list of text parts, got {type(content).__name__}")


def _parse_tool(index: int, raw: Any, namespace: str | None) -> ToolSpec:
    what = f"tools[{index}]" if namespace is None else f"tools[{index}] (namespace {namespace})"
    raw = _expect(raw, dict, what)
    kind = raw.get("type", "function")
    if kind != "function":
        raise _bad(f"{what}: unsupported tool type {kind!r}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise _bad(f"{what}: missing name")
    description = raw.get("description") or ""
    description = _expect(description, str, f"{what}.description")
    parameters = raw.get("parameters")
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
    parameters = _expect(parameters, dict, f"{what}.parameters")
    return ToolSpec(name=name, description=description, parameters=dict(parameters), namespace=namespace)


def _parse_tools(raw_tools: Any) -> tuple[ToolSpec, ...]:
    if raw_tools is None:
        return ()
    out: list[ToolSpec] = []
    for i, raw in enumerate(_expect(raw_tools, list, "tools")):
        raw = _expect(raw, dict, f"tools[{i}]")
        kind = raw.get("type")
        if kind == "namespace":
            ns = raw.get("name")
            if not isinstance(ns, str) or not ns:
                raise _bad(f"tools[{i}]: namespace without a name")
            for j, inner in enumerate(_expect(raw.get("tools"), list, f"tools[{i}].tools")):
                out.append(_parse_tool(j, inner, ns))
        elif kind == "function":
            out.append(_parse_tool(i, raw, None))
        else:
            raise _bad(f"tools[{i}]: unsupported tool type {kind!r} (function / namespace only)")
    return tuple(out)


class _Builder:
    """Folds Responses input items into chat messages, keeping a turn's reasoning + calls on one assistant message."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.pending_reasoning: str | None = None
        self.open_assistant: bool = False      # the last message is an assistant message of the current turn

    def add(self, message: Message) -> None:
        self.messages.append(message)
        self.open_assistant = message.role == "assistant"

    def system_text(self, text: str, what: str) -> None:
        """A ``developer`` / ``system`` item: Codex sends one leading developer message (skills + permissions,
        contract §10.7) after ``instructions``; the pinned template allows a single system message at the head of the
        conversation, so leading system texts are joined into it (blank-line separated) and one arriving after the
        conversation started is a ``bad_request`` (never a 500 at render time)."""
        if self.messages and all(m.role == "system" for m in self.messages):
            head = self.messages[0]
            joined = "\n\n".join(part for part in (head.content, text) if part)
            self.messages = [Message(role="system", content=joined)] + self.messages[1:]
            return
        if not self.messages:
            self.add(Message(role="system", content=text))
            return
        raise _bad(f"{what}: a developer/system message after the conversation started cannot be rendered")

    def assistant_text(self, text: str) -> None:
        self.add(Message(role="assistant", content=text, reasoning_content=self._take_reasoning()))

    def function_call(self, call: ToolCall) -> None:
        if self.open_assistant and not self.messages[-1].tool_calls and self.pending_reasoning is None:
            last = self.messages[-1]
            self.messages[-1] = Message(role="assistant", content=last.content, reasoning_content=last.reasoning_content,
                                        tool_calls=(call,), name=last.name)
            return
        if self.open_assistant and self.messages[-1].tool_calls and self.pending_reasoning is None:
            last = self.messages[-1]
            self.messages[-1] = Message(role="assistant", content=last.content, reasoning_content=last.reasoning_content,
                                        tool_calls=last.tool_calls + (call,), name=last.name)
            return
        self.add(Message(role="assistant", content="", reasoning_content=self._take_reasoning(), tool_calls=(call,)))

    def _take_reasoning(self) -> str | None:
        reasoning, self.pending_reasoning = self.pending_reasoning, None
        return reasoning


def parse_responses_body(body: Mapping[str, Any], *, subagent: bool = False) -> ChatRequest:
    """Codex's ``ResponsesApiRequest`` -> ``ChatRequest`` (contract §6); unexpected shapes raise ``bad_request``.

    ``subagent`` is True when the request carried the ``x-openai-subagent`` header (Codex children; contract §10.1):
    a Codex fork shares the root's ``instructions``, so the header is the only solver marker.
    """
    body = _expect(body, dict, "request body")
    builder = _Builder()
    instructions = body.get("instructions")
    if instructions is not None:
        builder.add(Message(role="system", content=_expect(instructions, str, "instructions")))
    raw_input = body.get("input")
    if isinstance(raw_input, str):
        raw_input = [{"type": "message", "role": "user", "content": raw_input}]
    if raw_input is None:
        raise _bad("input is required")
    for i, item in enumerate(_expect(raw_input, list, "input")):
        item = _expect(item, dict, f"input[{i}]")
        kind = item.get("type", "message")
        what = f"input[{i}] ({kind})"
        if kind == "message":
            role = item.get("role")
            if role not in _MESSAGE_ROLES:
                raise _bad(f"{what}: unsupported role {role!r}")
            text = _text_of(item.get("content"), what)
            if role == "assistant":
                builder.assistant_text(text)
            elif role in ("developer", "system"):
                builder.system_text(text, what)
            else:
                builder.add(Message(role="user", content=text))
        elif kind == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if not isinstance(call_id, str) or not call_id:
                raise _bad(f"{what}: missing call_id")
            if not isinstance(name, str) or not name:
                raise _bad(f"{what}: missing name")
            builder.function_call(ToolCall(id=call_id, name=name, arguments=parse_tool_arguments(item.get("arguments"))))
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise _bad(f"{what}: missing call_id")
            builder.add(Message(role="tool", content=_text_of(item.get("output"), what), tool_call_id=call_id))
        elif kind == "reasoning":
            texts: list[str] = []
            content = item.get("content")
            if content is not None:
                for part in _expect(content, list, f"{what}.content"):
                    part = _expect(part, dict, f"{what}.content part")
                    if part.get("type") != "reasoning_text":
                        raise _bad(f"{what}: unsupported reasoning content type {part.get('type')!r}")
                    texts.append(_expect(part.get("text", ""), str, f"{what}.content text"))
            builder.pending_reasoning = "".join(texts) if texts else None
        else:
            raise _bad(f"{what}: unsupported input item type {kind!r}")
    if not builder.messages:
        raise _bad("input produced no messages")
    stream = body.get("stream", False)
    if stream is None:
        stream = False
    if not isinstance(stream, bool):
        raise _bad(f"stream must be a boolean, got {stream!r}")
    requested = body.get("max_output_tokens")
    if requested is not None:
        requested = _expect(requested, int, "max_output_tokens")
    effort: str | None = None
    reasoning = body.get("reasoning")
    if reasoning is not None:
        reasoning = _expect(reasoning, dict, "reasoning")
        if reasoning.get("effort") is not None:
            effort = _expect(reasoning["effort"], str, "reasoning.effort")
    ignored = tuple(sorted(k for k in body if k not in CONSUMED_RESPONSES_KEYS))
    return ChatRequest(messages=tuple(builder.messages), tools=_parse_tools(body.get("tools")),
                       requested_max_tokens=requested, requested_reasoning_effort=effort, stream=stream,
                       ignored_keys=ignored, wire="responses", subagent=bool(subagent))


# ---------------------------------------------------------------------------------------------------------
# Replies
# ---------------------------------------------------------------------------------------------------------

def responses_usage(usage: ReplyUsage) -> dict[str, Any]:
    prompt = usage.prompt_tokens or 0
    completion = usage.completion_tokens or 0
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": prompt + completion,
            "input_tokens_details": {"cached_tokens": usage.cached_prompt_tokens or 0},
            "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens or 0}}


def output_items(reply: ChatReply, *, response_id: str, namespaces: Mapping[str, str | None] | None = None,
                 ) -> list[dict[str, Any]]:
    """The ordered ``output[]`` items: reasoning (if any text), message (if any text), one function_call per call."""
    items: list[dict[str, Any]] = []
    if reply.reasoning_content:
        items.append({"type": "reasoning", "id": f"rs_{response_id}", "summary": [],
                      "content": [{"type": "reasoning_text", "text": reply.reasoning_content}]})
    if reply.content:
        items.append({"type": "message", "id": f"msg_{response_id}", "role": "assistant", "status": "completed",
                      "content": [{"type": "output_text", "text": reply.content, "annotations": []}]})
    for i, call in enumerate(reply.tool_calls):
        item: dict[str, Any] = {"type": "function_call", "id": f"fc_{response_id}_{i}", "call_id": call.id,
                                "name": call.name, "arguments": tool_arguments_json(call.arguments),
                                "status": "completed"}
        ns = (namespaces or {}).get(call.name)
        if ns:
            item["namespace"] = ns
        items.append(item)
    return items


def _event(kind: str, payload: Mapping[str, Any], seq: int) -> bytes:
    body = {"type": kind, "sequence_number": seq, **payload}
    return (f"event: {kind}\n".encode("utf-8") + b"data: "
            + json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n\n")


def _response_object(response_id: str, *, status: str, model: str, output: list[dict[str, Any]] | None = None,
                     usage: Mapping[str, Any] | None = None) -> dict[str, Any]:
    obj: dict[str, Any] = {"id": response_id, "object": "response", "status": status, "model": model,
                           "output": list(output or [])}
    if usage is not None:
        obj["usage"] = dict(usage)
    return obj


def responses_sse(reply: ChatReply, *, response_id: str, model: str = "qwen/qwen3.8-27b",
                  namespaces: Mapping[str, str | None] | None = None) -> Iterable[bytes]:
    """``response.created`` -> ``response.output_item.done`` per item -> ``response.completed`` (module docstring)."""
    if not isinstance(response_id, str) or not response_id:
        raise ValueError("responses_sse needs a non-empty string response_id (Codex requires a string id)")
    items = output_items(reply, response_id=response_id, namespaces=namespaces)
    seq = 0
    yield _event("response.created", {"response": _response_object(response_id, status="in_progress", model=model)}, seq)
    for index, item in enumerate(items):
        seq += 1
        yield _event("response.output_item.done", {"output_index": index, "item": item}, seq)
    seq += 1
    completed = _response_object(response_id, status="completed", model=model, output=items,
                                 usage=responses_usage(reply.usage))
    completed["dynorch"] = {"call_no": reply.call.call_no, "conversation_class": reply.call.conversation_class,
                            "forced_close_kind": reply.call.forced_close_kind, "grace_tokens": reply.call.grace_tokens,
                            "tool_parse": reply.call.tool_parse, "headers": {k.lower(): v for k, v in reply.headers.items()}}
    yield _event("response.completed", {"response": completed}, seq)


def responses_json(reply: ChatReply, *, response_id: str, model: str = "qwen/qwen3.8-27b",
                   namespaces: Mapping[str, str | None] | None = None) -> dict[str, Any]:
    """The non-streaming ``response`` object (Codex never asks for it; kept for tools that do)."""
    items = output_items(reply, response_id=response_id, namespaces=namespaces)
    return _response_object(response_id, status="completed", model=model, output=items,
                            usage=responses_usage(reply.usage))


def responses_error_code(err: ProxyError) -> str:
    """Contract §6: ``rate_limit_exceeded`` (retryable upstream), ``context_length_exceeded`` (input guard), else
    ``server_error``."""
    if err.retryable:
        return "rate_limit_exceeded"
    if err.code == "input_limit":
        return "context_length_exceeded"
    return "server_error"


def responses_failed_sse(err: ProxyError, *, response_id: str, model: str = "qwen/qwen3.8-27b") -> Iterable[bytes]:
    """``response.failed`` for an error raised after the SSE headers went out."""
    failed = _response_object(response_id, status="failed", model=model)
    failed["error"] = {"code": responses_error_code(err), "message": f"dynorch {err.code} ({err.status}): {err.message}"}
    yield _event("response.failed", {"response": failed}, 0)
