"""WP-B: the h11 server over loopback with ``httpx.AsyncClient`` (contract §6 routes, auth, streaming keepalive)."""
from __future__ import annotations

import asyncio
import gzip
import json
import time

import httpx
import pytest

from tests.unit.dynorch import stubs
from tests.unit.dynorch.stubs import Scripted, StubProxyCore
from agent_allocation.dynorch.httpd import KEEPALIVE_COMMENT, ProxyServer, route
from agent_allocation.dynorch.records import ProxyError, RunSpec

RUN = "abc123def456abc123def456"
SYSTEM = "You are the orchestrator."


def _spec(run_key: str = RUN, cap: int = 4096) -> RunSpec:
    return RunSpec(run_key=run_key, arm="H1", harness="openclaude", item_id="aime/2025/1", benchmark="aime",
                   stratum="aime", shard_id="parasail", endpoint_tag="parasail/fp8", rung=cap // 1024, cap_tokens=cap,
                   repeat=1, attempt=1, generation=1, partition="main", allowed_tools=("Agent",), spawn_tools=("Agent",),
                   root_call_ceiling=None, sampling_hash="s", template_sha256="t", prompt_version="dynorch-v1",
                   proxy_version="p", harness_version="h")


def _chat_body(stream: bool = False, **over):
    body = {"model": "qwen/qwen3.8-27b", "messages": [{"role": "system", "content": SYSTEM},
                                                       {"role": "user", "content": "TASK"}], "stream": stream,
            "max_tokens": 32000}
    body.update(over)
    return body


@pytest.fixture
async def server():
    core = StubProxyCore(lambda run, req, n: Scripted(reasoning="r", content="<final>5</final>", completion_tokens=50))
    core.open_run(_spec(), root_system_text=SYSTEM, subagent_system_text="solver")
    srv = ProxyServer(core, keepalive_seconds=0.2)
    await srv.start()
    try:
        yield srv
    finally:
        await srv.close()


def _headers(run: str = RUN) -> dict:
    return {"Authorization": f"Bearer {run}"}


def test_route_parsing():
    assert route("/healthz") == ("healthz", None, None)
    assert route(f"/r/{RUN}/v1/chat/completions?x=1") == ("run", RUN, "chat/completions")
    assert route(f"/r/{RUN}/v1/responses") == ("run", RUN, "responses")
    assert route(f"/r/{RUN}/v1/models") == ("run", RUN, "models")
    assert route(f"/r/{RUN}/v1/budget") == ("run", RUN, "budget")
    assert route("/v1/chat/completions") == ("unknown", None, None)
    assert route(f"/r/{RUN}/v1/embeddings") == ("unknown", None, None)


async def test_healthz_models_budget_and_keepalive(server: ProxyServer):
    async with httpx.AsyncClient(base_url=server.base_url) as client:
        health = await client.get("/healthz")
        assert health.status_code == 200 and health.json()["ok"] is True and health.json()["runs_open"] == 1
        models = await client.get(f"/r/{RUN}/v1/models", headers=_headers())
        assert models.json() == {"object": "list", "data": [{"id": "qwen/qwen3.8-27b", "object": "model",
                                                             "owned_by": "dynorch"}]}
        budget = await client.get(f"/r/{RUN}/v1/budget", headers=_headers())
        assert budget.json() == {"cap": 4096, "spent": 0, "remaining": 4096, "calls": 0}
        # keep-alive: the same connection served all three requests
        assert server.requests_served == 3
        first = await client.post(f"/r/{RUN}/v1/chat/completions", headers=_headers(), json=_chat_body())
        second = await client.post(f"/r/{RUN}/v1/chat/completions", headers=_headers(), json=_chat_body())
        assert first.status_code == 200 and second.status_code == 200
        assert first.json()["choices"][0]["message"]["content"] == "<final>5</final>"
        assert first.headers["x-dynorch-call-no"] == "1" and second.headers["x-dynorch-call-no"] == "2"
        assert first.headers["content-type"] == "application/json"
        assert (await client.get(f"/r/{RUN}/v1/budget", headers=_headers())).json()["spent"] == 100
    assert len(server.reply_log.served[RUN]) == 2


async def test_bearer_must_match_run_key(server: ProxyServer):
    async with httpx.AsyncClient(base_url=server.base_url) as client:
        wrong = await client.post(f"/r/{RUN}/v1/chat/completions", headers=_headers("other"), json=_chat_body())
        assert wrong.status_code == 401 and wrong.json()["error"]["code"] == "bad_bearer"
        missing = await client.get(f"/r/{RUN}/v1/models")
        assert missing.status_code == 401
        unknown_run = await client.post("/r/nosuchrun/v1/chat/completions", headers=_headers("nosuchrun"),
                                        json=_chat_body())
        assert unknown_run.status_code == 404 and unknown_run.json()["error"]["code"] == "unknown_run"
        nowhere = await client.get("/v1/models", headers=_headers())
        assert nowhere.status_code == 404 and nowhere.json()["error"]["code"] == "not_found"
        bad_json = await client.post(f"/r/{RUN}/v1/chat/completions", headers=_headers(), content=b"{not json")
        assert bad_json.status_code == 400 and bad_json.json()["error"]["code"] == "bad_request"
        bad_n = await client.post(f"/r/{RUN}/v1/chat/completions", headers=_headers(), json=_chat_body(n=3))
        assert bad_n.status_code == 400
        get_chat = await client.get(f"/r/{RUN}/v1/chat/completions", headers=_headers())
        assert get_chat.status_code == 405


async def test_tool_surface_violation_is_a_400(server: ProxyServer):
    tools = [{"type": "function", "function": {"name": "Bash", "parameters": {"type": "object"}}}]
    async with httpx.AsyncClient(base_url=server.base_url) as client:
        r = await client.post(f"/r/{RUN}/v1/chat/completions", headers=_headers(), json=_chat_body(tools=tools))
        assert r.status_code == 400 and r.json()["error"]["code"] == "tool_surface_violation"


async def test_chunked_and_gzip_request_bodies(server: ProxyServer):
    payload = json.dumps(_chat_body()).encode()

    async def chunks():
        for i in range(0, len(payload), 37):
            yield payload[i:i + 37]

    async with httpx.AsyncClient(base_url=server.base_url) as client:
        chunked = await client.post(f"/r/{RUN}/v1/chat/completions", headers=_headers(), content=chunks())
        assert chunked.status_code == 200 and chunked.json()["choices"][0]["finish_reason"] == "stop"
        gz = await client.post(f"/r/{RUN}/v1/chat/completions",
                               headers={**_headers(), "Content-Encoding": "gzip", "Content-Type": "application/json"},
                               content=gzip.compress(payload))
        assert gz.status_code == 200 and gz.json()["usage"]["completion_tokens"] == 50


async def test_streaming_sends_headers_at_once_and_keepalives_while_core_is_pending():
    core = StubProxyCore(lambda run, req, n: Scripted(reasoning="r", content="<final>5</final>", completion_tokens=50,
                                                      delay_seconds=0.7))
    core.open_run(_spec(), root_system_text=SYSTEM, subagent_system_text=None)
    async with ProxyServer(core, keepalive_seconds=0.2) as srv:
        async with httpx.AsyncClient(base_url=srv.base_url, timeout=10) as client:
            started = time.monotonic()
            async with client.stream("POST", f"/r/{RUN}/v1/chat/completions", headers=_headers(),
                                     json=_chat_body(stream=True)) as response:
                headers_at = time.monotonic() - started
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"
                assert headers_at < 0.4, "headers must be sent before the core call finishes"
                raw = b"".join([chunk async for chunk in response.aiter_raw()])
    keepalives = raw.count(KEEPALIVE_COMMENT)
    assert keepalives >= 2, raw
    body = raw.replace(KEEPALIVE_COMMENT, b"")
    assert body.startswith(b"data: {") and body.endswith(b"data: [DONE]\n\n")
    frames = [json.loads(f[6:]) for f in body.split(b"\n\n") if f and f != b"data: [DONE]"]
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
    assert frames[-1]["choices"] == [] and frames[-1]["usage"]["completion_tokens"] == 50


async def test_streaming_error_after_headers_is_an_error_frame():
    core = StubProxyCore(lambda run, req, n: Scripted(error=ProxyError(402, "budget_exhausted", "cap", False)))
    core.open_run(_spec(), root_system_text=SYSTEM, subagent_system_text=None)
    async with ProxyServer(core, keepalive_seconds=5) as srv:
        async with httpx.AsyncClient(base_url=srv.base_url) as client:
            r = await client.post(f"/r/{RUN}/v1/chat/completions", headers=_headers(), json=_chat_body(stream=True))
            assert r.status_code == 200
            frames = [f for f in r.content.split(b"\n\n") if f]
            assert json.loads(frames[0][6:])["error"]["code"] == "budget_exhausted"
            assert frames[1] == b"data: [DONE]"
            plain = await client.post(f"/r/{RUN}/v1/chat/completions", headers=_headers(), json=_chat_body())
            assert plain.status_code == 402 and plain.json()["error"]["code"] == "budget_exhausted"


async def test_responses_route_streams_codex_events_and_failures():
    def script(run, req, n):
        if n == 1:
            return Scripted(reasoning="r", content="", tool_calls=[("Agent", {"prompt": "p"})], completion_tokens=40)
        return Scripted(error=ProxyError(503, "ledger_refused", "paused", True))

    core = StubProxyCore(script)
    core.open_run(_spec(), root_system_text=SYSTEM, subagent_system_text=None)
    body = {"model": "m", "instructions": SYSTEM, "input": [{"type": "message", "role": "user", "content": "TASK"}],
            "tools": [{"type": "namespace", "name": "ns", "tools": [
                {"type": "function", "name": "Agent", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}}}]}],
            "stream": True, "store": False}
    async with ProxyServer(core, keepalive_seconds=5) as srv:
        async with httpx.AsyncClient(base_url=srv.base_url) as client:
            r = await client.post(f"/r/{RUN}/v1/responses", headers=_headers(), json=body)
            assert r.status_code == 200 and r.headers["content-type"] == "text/event-stream"
            events = [json.loads(f.split(b"\n")[1][6:]) for f in r.content.split(b"\n\n") if f]
            assert [e["type"] for e in events] == ["response.created", "response.output_item.done",
                                                   "response.output_item.done", "response.completed"]
            assert events[2]["item"]["type"] == "function_call" and events[2]["item"]["namespace"] == "ns"
            assert events[3]["response"]["usage"]["output_tokens"] == 40
            failed = await client.post(f"/r/{RUN}/v1/responses", headers=_headers(), json=body)
            events = [json.loads(f.split(b"\n")[1][6:]) for f in failed.content.split(b"\n\n") if f]
            assert events[0]["type"] == "response.failed"
            assert events[0]["response"]["error"]["code"] == "rate_limit_exceeded"
            non_stream = await client.post(f"/r/{RUN}/v1/responses", headers=_headers(), json={**body, "stream": False})
            assert non_stream.status_code == 503


async def test_client_disconnect_does_not_cancel_the_core_call():
    finished = asyncio.Event()

    async def script(run, req, n):
        await asyncio.sleep(0.5)
        finished.set()
        return Scripted(content="<final>1</final>", completion_tokens=10)

    core = StubProxyCore(script)
    core.open_run(_spec(), root_system_text=SYSTEM, subagent_system_text=None)
    async with ProxyServer(core, keepalive_seconds=0.1) as srv:
        reader, writer = await asyncio.open_connection("127.0.0.1", srv.port)
        payload = json.dumps(_chat_body(stream=True)).encode()
        writer.write((f"POST /r/{RUN}/v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {RUN}\r\n"
                      f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n").encode() + payload)
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
        assert head.startswith(b"HTTP/1.1 200")
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(finished.wait(), 5)
        await asyncio.sleep(0.1)
        assert core.runs[RUN].record.calls == 1 and core.runs[RUN].record.completion_total == 10


async def test_internal_error_is_a_loud_500(server: ProxyServer):
    def boom(run, req, n):
        raise RuntimeError("proxy bug")

    server.core.script = boom
    async with httpx.AsyncClient(base_url=server.base_url) as client:
        r = await client.post(f"/r/{RUN}/v1/chat/completions", headers=_headers(), json=_chat_body())
        assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
