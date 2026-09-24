"""WP-B: Responses API translation, SSE event order, usage, failures, and chat/Responses parity (contract §6)."""
from __future__ import annotations

import dataclasses
import json

import pytest

from agent_allocation.api.fake import FakeReply, FakeScript, fake_tool_call_text
from agent_allocation.dynorch import openai_format as of
from agent_allocation.dynorch import responses_format as rf
from agent_allocation.dynorch.records import ChatReply, ProxyError, ReplyUsage, RunSpec, ToolCall, ToolSpec
from tests.unit.dynorch.test_openai_format import _call
from tests.unit.dynorch.conftest import make_spec

SYSTEM = "You are the orchestrator."
USER = "TASK: prove it"
COLLAB_NAMESPACE = {
    "type": "namespace", "name": "multi_agent_v1", "description": "Tools for spawning and managing sub-agents.",
    "tools": [
        {"type": "function", "name": "spawn_agent", "description": "Spawn a sub-agent", "strict": False,
         "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}},
        {"type": "function", "name": "wait_agent", "description": "Wait", "parameters": {"type": "object", "properties": {}}},
    ],
}
UPDATE_PLAN = {"type": "function", "name": "update_plan", "description": "plan", "strict": False,
               "parameters": {"type": "object", "properties": {"plan": {"type": "array"}}}}


def _body(**over):
    body = {"model": "qwen/qwen3.8-27b", "instructions": SYSTEM,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": USER}]}],
            "tools": [COLLAB_NAMESPACE, UPDATE_PLAN], "tool_choice": "auto", "parallel_tool_calls": False,
            "reasoning": {"effort": "high"}, "store": False, "stream": True,
            "include": ["reasoning.encrypted_content"], "prompt_cache_key": "uuid", "text": {"verbosity": "low"},
            "client_metadata": {}}
    body.update(over)
    return body


# -- parse ------------------------------------------------------------------------------------------------

def test_parse_instructions_message_and_namespace_flattening():
    req = rf.parse_responses_body(_body())
    assert req.wire == "responses" and req.stream is True
    assert [m.role for m in req.messages] == ["system", "user"]
    assert req.messages[0].content == SYSTEM and req.messages[1].content == USER
    assert [t.name for t in req.tools] == ["spawn_agent", "wait_agent", "update_plan"]
    assert [t.namespace for t in req.tools] == ["multi_agent_v1", "multi_agent_v1", None]
    assert req.tools[0].parameters["properties"]["message"]["type"] == "string"
    assert req.requested_reasoning_effort == "high" and req.requested_max_tokens is None
    assert req.ignored_keys == ("client_metadata", "include", "model", "parallel_tool_calls", "prompt_cache_key",
                                "store", "text", "tool_choice")


def test_parse_history_items_map_to_chat_messages():
    history = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev note"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": USER}]},
        {"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": "I delegate"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Delegating."}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1_0", "name": "spawn_agent",
         "arguments": "{\"message\": \"solve\"}", "namespace": "multi_agent_v1"},
        {"type": "function_call_output", "call_id": "call_1_0", "output": "{\"agent_id\":\"a1\"}"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "summary only"}]},
        {"type": "function_call", "call_id": "call_2_0", "name": "wait_agent", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_2_0",
         "output": [{"type": "input_text", "text": "<final>3</final>"}]},
    ]
    req = rf.parse_responses_body(_body(input=history, max_output_tokens=2048))
    roles = [m.role for m in req.messages]
    assert roles == ["system", "system", "user", "assistant", "tool", "assistant", "tool"]
    assert req.messages[1].content == "dev note"
    first = req.messages[3]
    assert first.content == "Delegating." and first.reasoning_content == "I delegate"
    assert first.tool_calls == (ToolCall(id="call_1_0", name="spawn_agent", arguments={"message": "solve"}),)
    assert req.messages[4].tool_call_id == "call_1_0" and req.messages[4].content == '{"agent_id":"a1"}'
    second = req.messages[5]
    assert second.content == "" and second.reasoning_content is None      # summary-only reasoning carries no text
    assert second.tool_calls == (ToolCall(id="call_2_0", name="wait_agent", arguments={}),)
    assert req.messages[6].content == "<final>3</final>" and req.messages[6].tool_call_id == "call_2_0"
    assert req.requested_max_tokens == 2048


def test_parse_two_function_calls_in_one_turn_share_one_assistant_message():
    history = [
        {"type": "message", "role": "user", "content": USER},
        {"type": "function_call", "call_id": "c1", "name": "spawn_agent", "arguments": "{}"},
        {"type": "function_call", "call_id": "c2", "name": "spawn_agent", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "x"},
        {"type": "function_call_output", "call_id": "c2", "output": "y"},
    ]
    req = rf.parse_responses_body(_body(instructions=None, input=history))
    assert [m.role for m in req.messages] == ["user", "assistant", "tool", "tool"]
    assert [c.id for c in req.messages[1].tool_calls] == ["c1", "c2"]


def test_parse_string_input_becomes_user_message():
    req = rf.parse_responses_body(_body(input="hello"))
    assert req.messages[-1].role == "user" and req.messages[-1].content == "hello"


@pytest.mark.parametrize("body, needle", [
    (_body(tools=[{"type": "web_search"}]), "unsupported tool type"),
    (_body(tools=[{"type": "local_shell"}]), "unsupported tool type"),
    (_body(tools=[{"type": "namespace", "tools": []}]), "namespace without a name"),
    (_body(input=[{"type": "local_shell_call", "action": {}}]), "unsupported input item type"),
    (_body(input=[{"type": "message", "role": "tool", "content": "x"}]), "unsupported role"),
    (_body(input=[{"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]),
     "content part type"),
    (_body(input=[{"type": "function_call", "name": "f", "arguments": "{}"}]), "missing call_id"),
    (_body(input=[{"type": "function_call_output", "output": "x"}]), "missing call_id"),
    (_body(input=None), "input is required"),
    (_body(stream=1), "stream must be a boolean"),
])
def test_parse_rejects_unsupported_shapes(body, needle):
    with pytest.raises(ProxyError) as info:
        rf.parse_responses_body(body)
    assert info.value.status == 400 and info.value.code == "bad_request" and needle in info.value.message


# -- SSE ---------------------------------------------------------------------------------------------------

def _reply(content: str, reasoning: str, tool_calls=(), finish="stop") -> ChatReply:
    return ChatReply(content=content, reasoning_content=reasoning, tool_calls=tuple(tool_calls), finish_reason=finish,
                     usage=ReplyUsage(prompt_tokens=700, completion_tokens=250, reasoning_tokens=200,
                                      cached_prompt_tokens=64),
                     call=_call(tool_calls=len(tool_calls), finish_reason_returned=finish, wire="responses"),
                     headers={"X-Dynorch-Call-No": "1"})


def _events(raw_frames):
    out = []
    for raw in raw_frames:
        lines = raw.decode("utf-8").split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: ") and raw.endswith(b"\n\n")
        data = json.loads(lines[1][6:])
        assert data["type"] == lines[0][7:]
        out.append(data)
    return out


def test_responses_sse_event_order_items_and_usage():
    calls = [ToolCall(id="call_5_0", name="spawn_agent", arguments={"message": "go"}),
             ToolCall(id="call_5_1", name="update_plan", arguments={"plan": []})]
    events = _events(rf.responses_sse(_reply("Working.", "deep thought", calls, "tool_calls"), response_id="resp_1",
                                      namespaces={"spawn_agent": "multi_agent_v1", "update_plan": None}))
    assert [e["type"] for e in events] == ["response.created", "response.output_item.done",
                                           "response.output_item.done", "response.output_item.done",
                                           "response.output_item.done", "response.completed"]
    assert events[0]["response"]["id"] == "resp_1" and events[0]["response"]["status"] == "in_progress"
    reasoning = events[1]["item"]
    assert reasoning["type"] == "reasoning" and reasoning["summary"] == []
    assert reasoning["content"] == [{"type": "reasoning_text", "text": "deep thought"}]
    message = events[2]["item"]
    assert message["type"] == "message" and message["role"] == "assistant"
    assert message["content"][0] == {"type": "output_text", "text": "Working.", "annotations": []}
    spawn = events[3]["item"]
    assert spawn["type"] == "function_call" and spawn["call_id"] == "call_5_0" and spawn["name"] == "spawn_agent"
    assert spawn["arguments"] == '{"message": "go"}' and spawn["namespace"] == "multi_agent_v1"
    plan = events[4]["item"]
    assert plan["call_id"] == "call_5_1" and "namespace" not in plan and plan["arguments"] == '{"plan": []}'
    assert [e["output_index"] for e in events[1:5]] == [0, 1, 2, 3]
    completed = events[5]["response"]
    assert isinstance(completed["id"], str) and completed["id"] == "resp_1" and completed["status"] == "completed"
    assert completed["usage"] == {"input_tokens": 700, "output_tokens": 250, "total_tokens": 950,
                                  "input_tokens_details": {"cached_tokens": 64},
                                  "output_tokens_details": {"reasoning_tokens": 200}}
    assert [item["type"] for item in completed["output"]] == ["reasoning", "message", "function_call", "function_call"]
    assert [e["sequence_number"] for e in events] == list(range(6))


def test_responses_sse_skips_empty_reasoning_and_message_items():
    events = _events(rf.responses_sse(_reply("<final>9</final>", "", ()), response_id="r2"))
    assert [e["type"] for e in events] == ["response.created", "response.output_item.done", "response.completed"]
    assert events[1]["item"]["type"] == "message"


def test_responses_sse_requires_string_id():
    with pytest.raises(ValueError):
        list(rf.responses_sse(_reply("x", "y"), response_id=""))


@pytest.mark.parametrize("err, code", [
    (ProxyError(503, "ledger_refused", "pause", True), "rate_limit_exceeded"),
    (ProxyError(402, "input_limit", "600k prompt tokens", False), "context_length_exceeded"),
    (ProxyError(402, "budget_exhausted", "cap", False), "server_error"),
    (ProxyError(502, "upstream_failed", "ambiguous", False), "server_error"),
])
def test_responses_failed_codes(err, code):
    assert rf.responses_error_code(err) == code
    events = _events(rf.responses_failed_sse(err, response_id="r3"))
    assert events[0]["type"] == "response.failed"
    assert events[0]["response"]["status"] == "failed" and events[0]["response"]["error"]["code"] == code
    assert err.code in events[0]["response"]["error"]["message"]


def test_responses_json_object():
    payload = rf.responses_json(_reply("hi", "r"), response_id="r4")
    assert payload["object"] == "response" and payload["status"] == "completed"
    assert [i["type"] for i in payload["output"]] == ["reasoning", "message"]


# -- parity: one scripted conversation through both wires yields identical CallRecords ---------------------
# Gate 0 (plan §9): "Responses <-> chat parity of CallRecords" -- against the REAL ProxyCore over api/fake.py.

def _spec(run_key: str, *, repeat: int = 1) -> RunSpec:
    """Two runs of the same cell (repeats 1 and 2, so their journals do not collide) driven over the two wires."""
    return make_spec(run_key=run_key, arm="H3", harness="codex", rung=4, item_id="aime/2025/1", repeat=repeat,
                     allowed_tools=("spawn_agent", "wait_agent", "update_plan"), spawn_tools=("spawn_agent",))


def _parity_script() -> FakeScript:
    """Call 1 (root): reasoning + one spawn_agent tool call; call 2 (root, after the tool result): the answer."""
    script = FakeScript()
    script.set_reply("aime/2025/1", reply=[
        FakeReply(content=fake_tool_call_text("spawn_agent", {"message": "solve"}), think_tokens=300),
        FakeReply(content="<final>17</final>", think_tokens=150),
    ])
    return script


async def _drive_chat(core, run_key: str) -> tuple[list, list]:
    tools = [{"type": "function", "function": {"name": "spawn_agent", "description": "Spawn a sub-agent",
                                               "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}}},
             {"type": "function", "function": {"name": "wait_agent", "description": "Wait",
                                               "parameters": {"type": "object", "properties": {}}}},
             {"type": "function", "function": {"name": "update_plan", "description": "plan",
                                               "parameters": {"type": "object", "properties": {"plan": {"type": "array"}}}}}]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    req1 = of.parse_chat_body({"model": "m", "messages": messages, "tools": tools, "stream": True,
                               "max_tokens": 32000, "reasoning_effort": "high"})
    r1 = await core.chat(run_key, req1)
    messages.append({"role": "assistant", "content": r1.content, "reasoning_content": r1.reasoning_content,
                     "tool_calls": [{"id": c.id, "type": "function",
                                     "function": {"name": c.name, "arguments": of.tool_arguments_json(c.arguments)}}
                                    for c in r1.tool_calls]})
    messages.append({"role": "tool", "tool_call_id": r1.tool_calls[0].id, "content": "<final>17</final>"})
    req2 = of.parse_chat_body({"model": "m", "messages": messages, "tools": tools, "stream": True,
                               "max_tokens": 32000, "reasoning_effort": "high"})
    r2 = await core.chat(run_key, req2)
    return [r1.call, r2.call], [req1, req2]


async def _drive_responses(core, run_key: str) -> tuple[list, list]:
    body = _body(reasoning={"effort": "high"}, max_output_tokens=32000)
    req1 = rf.parse_responses_body(body)
    r1 = await core.chat(run_key, req1)
    items = rf.output_items(r1, response_id="resp_a", namespaces={"spawn_agent": "multi_agent_v1"})
    history = list(body["input"]) + items + [
        {"type": "function_call_output", "call_id": r1.tool_calls[0].id, "output": "<final>17</final>"}]
    req2 = rf.parse_responses_body(_body(input=history, reasoning={"effort": "high"}, max_output_tokens=32000))
    r2 = await core.chat(run_key, req2)
    return [r1.call, r2.call], [req1, req2]


async def test_chat_and_responses_parity_produces_identical_call_records(make_env):
    env = make_env(_parity_script())
    env.open(_spec("k" * 24), root_system=SYSTEM, subagent_system="solver")
    env.open(_spec("r" * 24, repeat=2), root_system=SYSTEM, subagent_system="solver")
    chat_calls, chat_reqs = await _drive_chat(env.core, "k" * 24)
    resp_calls, resp_reqs = await _drive_responses(env.core, "r" * 24)
    assert len(chat_calls) == len(resp_calls) == 2
    # Only the wire tag, the per-wire ignored keys, wall-clock noise (queue wait, latency) and the fake's
    # generation ids may differ; tokens, grants, forced close, grace, cost, parse and re-injection are identical.
    masked = ("wire", "ignored_body_keys", "generation_ids", "queue_wait_seconds", "latency_seconds")
    for a, b in zip(chat_calls, resp_calls):
        da, db = dataclasses.asdict(a), dataclasses.asdict(b)
        for key in masked:
            da.pop(key), db.pop(key)
        assert da == db
    assert chat_calls[0].wire == "chat" and resp_calls[0].wire == "responses"
    assert chat_calls[0].tool_calls == 1 and chat_calls[0].spawn_calls == 1 and chat_calls[0].finish_reason_returned == "tool_calls"
    assert chat_calls[1].reinjection == resp_calls[1].reinjection == "explicit"
    assert chat_calls[1].completion_tokens == resp_calls[1].completion_tokens > 0
    # The messages the core saw were the same on both wires (conversation key + reasoning re-sent) and so were
    # the rendered prompts (identical prompt tokens on both calls).
    chat_req, resp_req = chat_reqs[1], resp_reqs[1]
    assert [m.role for m in chat_req.messages] == [m.role for m in resp_req.messages]
    assert chat_req.messages[2].tool_calls == resp_req.messages[2].tool_calls
    # the fake seeds its thinking text per node, so the two runs' reasoning differ in wording but not in length
    # (reasoning_chars / reasoning_tokens are part of the identical CallRecords above); both wires re-sent it
    assert chat_req.messages[2].reasoning_content and resp_req.messages[2].reasoning_content
    assert [t.name for t in chat_req.tools] == [t.name for t in resp_req.tools]
    assert [c.prompt_tokens for c in chat_calls] == [c.prompt_tokens for c in resp_calls]
    think = env.think_calls(phase="think")
    assert len(think) == 4
    assert chat_req.messages[2].reasoning_content in think[1].body["prompt"]
    assert resp_req.messages[2].reasoning_content in think[3].body["prompt"]
