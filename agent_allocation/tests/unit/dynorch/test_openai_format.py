"""WP-B: chat-completions parsing and rendering (contract §6, plan §5 chat route)."""
from __future__ import annotations

import json

import pytest

from tests.unit.dynorch import stubs  # noqa: F401  installs the records shim before any dynorch import
from agent_allocation.dynorch import openai_format as of
from agent_allocation.dynorch.records import (
    CallRecord, ChatReply, ProxyError, ReplyUsage, ToolCall,
)

SYSTEM = "Solve the supplied task accurately."
USER = "TASK\nWhat is 6 * 7?"


def _call(**over) -> CallRecord:
    base = dict(call_no=1, conversation_key="a" * 64, conversation_class="root", requested_max_tokens=32000,
                granted_max_tokens=4096, queue_wait_seconds=0.0, prompt_tokens=500, completion_tokens=120,
                reasoning_tokens=100, cached_prompt_tokens=0, reasoning_chars=40, visible_chars=20,
                forced_close_kind="none", grace_tokens=0, finish_reason_upstream="stop", finish_reason_returned="stop",
                tool_calls=0, spawn_calls=0, tool_parse="none", reinjection="n/a", reasoning_effort_requested=None,
                cost_usd="0.001", generation_ids=("gen-1",), latency_seconds=1.5, transport_status="ok",
                returned_status=200, stream=True, wire="chat", cache_status="MISS", duplicate_served=False,
                ignored_body_keys=("model",))
    base.update(over)
    return CallRecord(**base)


def _reply(content: str = "<final>42</final>", reasoning: str = "think", tool_calls=(), finish: str = "stop") -> ChatReply:
    return ChatReply(content=content, reasoning_content=reasoning, tool_calls=tuple(tool_calls), finish_reason=finish,
                     usage=ReplyUsage(prompt_tokens=500, completion_tokens=120, reasoning_tokens=100,
                                      cached_prompt_tokens=0),
                     call=_call(tool_calls=len(tool_calls), finish_reason_returned=finish),
                     headers={"X-Dynorch-Call-No": "1", "X-Dynorch-Remaining-Tokens": "3976"})


def _body(**over):
    body = {"model": "qwen/qwen3.8-27b", "messages": [{"role": "system", "content": SYSTEM},
                                                       {"role": "user", "content": USER}],
            "stream": True, "max_tokens": 32000, "temperature": 1.0}
    body.update(over)
    return body


# -- parse ------------------------------------------------------------------------------------------------

def test_parse_basic_body_records_ignored_keys_and_requested_values():
    req = of.parse_chat_body(_body(stream_options={"include_usage": True}, store=False))
    assert [m.role for m in req.messages] == ["system", "user"]
    assert req.messages[0].content == SYSTEM and req.messages[1].content == USER
    assert req.stream is True and req.wire == "chat"
    assert req.requested_max_tokens == 32000 and req.requested_reasoning_effort is None
    assert req.ignored_keys == ("model", "store", "stream_options", "temperature")
    assert req.tools == ()


def test_parse_tools_tool_calls_and_tool_role():
    tools = [{"type": "function", "function": {"name": "Agent", "description": "spawn",
                                               "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}}}}]
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": USER}]},
        {"role": "assistant", "content": None, "reasoning_content": "let me delegate",
         "tool_calls": [{"id": "call_1_0", "type": "function",
                         "function": {"name": "Agent", "arguments": "{\"prompt\": \"solve\"}"}},
                        {"id": "call_1_1", "type": "function", "function": {"name": "Agent", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "call_1_0", "content": "<final>7</final>"},
        {"role": "developer", "content": "note"},
    ]
    req = of.parse_chat_body(_body(messages=messages, tools=tools, reasoning_effort="high", max_completion_tokens=100))
    assert req.tools[0].name == "Agent" and req.tools[0].namespace is None
    assert req.tools[0].parameters["properties"]["prompt"]["type"] == "string"
    assistant = req.messages[2]
    assert assistant.content == "" and assistant.reasoning_content == "let me delegate"
    assert assistant.tool_calls[0] == ToolCall(id="call_1_0", name="Agent", arguments={"prompt": "solve"})
    assert assistant.tool_calls[1].arguments == "not json"        # raw string kept (contract §2)
    tool = req.messages[3]
    assert tool.role == "tool" and tool.tool_call_id == "call_1_0" and tool.content == "<final>7</final>"
    assert req.messages[4].role == "system"                        # developer -> system
    assert req.requested_reasoning_effort == "high" and req.requested_max_tokens == 100


def test_parse_nested_reasoning_effort_object():
    req = of.parse_chat_body(_body(reasoning={"effort": "low"}))
    assert req.requested_reasoning_effort == "low"


def test_parse_arguments_already_an_object_and_non_object_json():
    assert of.parse_tool_arguments({"a": 1}) == {"a": 1}
    assert of.parse_tool_arguments("[1, 2]") == "[1, 2]"
    assert of.parse_tool_arguments(None) == {}
    assert of.tool_arguments_json({"a": "é"}) == '{"a": "é"}'
    assert of.tool_arguments_json("raw") == "raw"


@pytest.mark.parametrize("body, needle", [
    (_body(n=2), "n must be 1"),
    (_body(n=0), "n must be 1"),
    (_body(messages=[]), "must not be empty"),
    (_body(messages=[{"role": "function", "content": "x"}]), "unsupported role"),
    (_body(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]), "content part type"),
    (_body(messages=[{"role": "tool", "content": "x"}]), "tool_call_id"),
    (_body(tools=[{"type": "web_search"}]), "unsupported tool type"),
    (_body(tools=[{"type": "function", "function": {"description": "no name"}}]), "function.name"),
    (_body(stream="yes"), "stream must be a boolean"),
    (_body(max_tokens="lots"), "max_tokens must be int"),
    (_body(messages=[{"role": "user", "content": "x", "tool_calls": [{"id": "c", "function": {"name": "f"}}]}]),
     "only assistant messages"),
    ([1, 2, 3], "request body must be dict"),
])
def test_parse_rejects_unexpected_shapes_with_bad_request(body, needle):
    with pytest.raises(ProxyError) as info:
        of.parse_chat_body(body)
    assert info.value.status == 400 and info.value.code == "bad_request"
    assert needle in info.value.message


def test_parse_n_equal_one_and_null_content_are_fine():
    req = of.parse_chat_body(_body(n=1, messages=[{"role": "system", "content": SYSTEM},
                                                  {"role": "assistant", "content": None}]))
    assert req.messages[1].content == ""


# -- JSON reply --------------------------------------------------------------------------------------------

def test_chat_reply_json_shape():
    reply = _reply(tool_calls=[ToolCall(id="call_2_0", name="Agent", arguments={"prompt": "go"})], finish="tool_calls")
    payload = of.chat_reply_json(reply, model="qwen/qwen3.8-27b", response_id="chatcmpl-x", created=7)
    assert payload["object"] == "chat.completion" and payload["id"] == "chatcmpl-x" and payload["created"] == 7
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "<final>42</final>"
    assert choice["message"]["reasoning_content"] == "think"
    call = choice["message"]["tool_calls"][0]
    assert call == {"id": "call_2_0", "type": "function", "function": {"name": "Agent", "arguments": '{"prompt": "go"}'}}
    assert payload["usage"] == {"prompt_tokens": 500, "completion_tokens": 120, "total_tokens": 620,
                                "completion_tokens_details": {"reasoning_tokens": 100},
                                "prompt_tokens_details": {"cached_tokens": 0}}
    assert payload["dynorch"]["call_no"] == 1 and payload["dynorch"]["headers"]["x-dynorch-call-no"] == "1"
    json.dumps(payload)


def test_chat_reply_json_without_tool_calls_omits_the_key():
    payload = of.chat_reply_json(_reply())
    assert "tool_calls" not in payload["choices"][0]["message"]
    assert payload["id"].startswith("chatcmpl-dynorch-")


# -- SSE ---------------------------------------------------------------------------------------------------

def _frames(reply: ChatReply) -> list:
    out = []
    for raw in of.chat_reply_sse(reply, model="m", response_id="rid", created=1):
        assert raw.startswith(b"data: ") and raw.endswith(b"\n\n")
        payload = raw[len(b"data: "):-2]
        out.append(None if payload == b"[DONE]" else json.loads(payload))
    return out


def test_sse_exact_chunk_sequence_with_tool_calls():
    calls = [ToolCall(id="call_3_0", name="Agent", arguments={"prompt": "a"}),
             ToolCall(id="call_3_1", name="Agent", arguments="raw")]
    frames = _frames(_reply(content="visible", reasoning="hmm", tool_calls=calls, finish="tool_calls"))
    deltas = [f["choices"][0]["delta"] if f and f["choices"] else f for f in frames]
    assert deltas[0] == {"role": "assistant", "content": ""}
    assert deltas[1] == {"reasoning_content": "hmm"}
    assert deltas[2] == {"content": "visible"}
    assert deltas[3] == {"tool_calls": [{"index": 0, "id": "call_3_0", "type": "function",
                                         "function": {"name": "Agent", "arguments": '{"prompt": "a"}'}}]}
    assert deltas[4] == {"tool_calls": [{"index": 1, "id": "call_3_1", "type": "function",
                                         "function": {"name": "Agent", "arguments": "raw"}}]}
    assert deltas[5] == {} and frames[5]["choices"][0]["finish_reason"] == "tool_calls"
    usage_chunk = frames[6]
    assert usage_chunk["choices"] == [] and usage_chunk["usage"]["completion_tokens"] == 120
    assert frames[7] is None                                       # data: [DONE]
    assert len(frames) == 8
    for f in frames[:5]:
        assert f["choices"][0]["finish_reason"] is None
    assert all(f["object"] == "chat.completion.chunk" and f["id"] == "rid" for f in frames if f)


def test_sse_usage_chunk_is_always_present_even_for_plain_stop():
    frames = _frames(_reply(content="<final>1</final>", reasoning="", finish="stop"))
    # role, content, finish, usage, DONE (no reasoning chunk when reasoning is empty)
    assert len(frames) == 5
    assert frames[1]["choices"][0]["delta"] == {"content": "<final>1</final>"}
    assert frames[2]["choices"][0]["finish_reason"] == "stop"
    assert frames[3]["choices"] == [] and frames[3]["usage"]["total_tokens"] == 620
    assert frames[4] is None


def test_sse_tool_call_only_turn_has_no_content_chunk():
    frames = _frames(_reply(content="", reasoning="r", tool_calls=[ToolCall(id="c", name="Agent", arguments={})],
                            finish="tool_calls"))
    kinds = [tuple(f["choices"][0]["delta"]) if f and f["choices"] else "usage" if f else "done" for f in frames]
    assert kinds == [("role", "content"), ("reasoning_content",), ("tool_calls",), (), "usage", "done"]


# -- errors ------------------------------------------------------------------------------------------------

def test_error_json_carries_status_and_code():
    status, body = of.error_json(ProxyError(402, "budget_exhausted", "cap reached", False))
    assert status == 402
    assert body["error"]["code"] == "budget_exhausted" and body["error"]["type"] == "budget_exhausted"
    assert body["error"]["message"] == "cap reached" and body["error"]["retryable"] is False


def test_error_sse_frames_then_done():
    frames = list(of.error_sse(ProxyError(503, "ledger_refused", "no money", True)))
    assert len(frames) == 2 and frames[-1] == b"data: [DONE]\n\n"
    assert json.loads(frames[0][6:])["error"]["code"] == "ledger_refused"
