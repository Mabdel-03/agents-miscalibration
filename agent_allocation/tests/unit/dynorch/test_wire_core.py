"""The REAL ``ProxyCore`` behind the REAL ``httpd.ProxyServer`` over loopback (second fix pass, 2026-09-24).

The wire layer's stub core cannot show races between the two: when a client goes away, when a reply counts as
delivered, and what an identical retry of a call still in flight is served. These tests drive raw sockets / httpx at
a real server over ``make_env``'s core (fake upstream, $0, loopback only)."""
from __future__ import annotations

import asyncio
import json

import httpx

from agent_allocation.api.fake import FakeReply, FakeScript
from agent_allocation.dynorch.httpd import ProxyServer

from .conftest import ROOT_SYSTEM, make_spec

FINAL = "<rationale>r</rationale>\n<final>42</final>"


def chat_body(*, stream: bool = True, task: str = "TASK\nWhat is 6 * 7?\n") -> dict:
    return {"model": "qwen/qwen3.8-27b", "stream": stream, "max_tokens": 32000,
            "messages": [{"role": "system", "content": ROOT_SYSTEM}, {"role": "user", "content": task}]}


def raw_post(run_key: str, body: dict) -> bytes:
    payload = json.dumps(body).encode()
    return (f"POST /r/{run_key}/v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {run_key}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n\r\n").encode() + payload


async def test_identical_retry_of_a_disconnected_in_flight_call_is_served_its_reply(make_env):
    """Review finding 2 (B9 regression): client 1 streams, disconnects while the upstream call is running, and its
    identical retry arrives 0.3 s later. The retry used to wait on the run's FIFO lock, wake first when the call
    settled, miss the cache (filled only later by the keeper task) and mint a SECOND grant: two THINK calls, the
    budget charged twice. Now the wire layer sees the EOF at once, the core caches the reply synchronously before
    the lock is released, and the retry is served the same reply with no new upstream call."""
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=200, delay_seconds=1.5)
    env = make_env(script)
    spec = env.open(make_spec(rung=4))
    async with ProxyServer(env.core, keepalive_seconds=5, refusal_grace_seconds=0.2) as srv:
        reader, writer = await asyncio.open_connection("127.0.0.1", srv.port)
        writer.write(raw_post(spec.run_key, chat_body()))
        await writer.drain()
        head = await reader.readuntil(b"\r\n\r\n")
        assert head.startswith(b"HTTP/1.1 200")
        writer.close()                                      # client 1 goes away mid-call
        await writer.wait_closed()
        await asyncio.sleep(0.3)
        async with httpx.AsyncClient(base_url=srv.base_url, timeout=30) as client:
            retry = await client.post(f"/r/{spec.run_key}/v1/chat/completions", json=chat_body(),
                                      headers={"Authorization": f"Bearer {spec.run_key}"})
        await srv.wait_idle(10)
    assert retry.status_code == 200 and "<final>42</final>" in retry.text
    record = env.core.record_of(spec.run_key)
    assert len(env.think_calls()) == 1, "the retry minted a second grant"
    assert record.calls == 1 and record.duplicates_served == 1
    assert record.completion_total == record.calls_detail[0].completion_tokens
    assert srv.reply_log.delivered(spec.run_key)[0] is not None      # the retry's copy was delivered (B7)


async def test_a_reply_written_after_the_client_closed_its_side_is_not_delivered(make_env):
    """Review finding 10: "delivered" meant write + drain succeeded, and the first write after the peer's FIN
    succeeds, so a late answer to a client that had already gone counted as delivered (B7) depending on timing.
    The EOF watcher is consulted before and after the write: a client that closed its side before the reply was
    written never received it -- undelivered, cached for a retry, never the delivered answer."""
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=200, delay_seconds=0.6)
    env = make_env(script)
    spec = env.open(make_spec(rung=4))
    async with ProxyServer(env.core, keepalive_seconds=5, refusal_grace_seconds=0.1) as srv:
        reader, writer = await asyncio.open_connection("127.0.0.1", srv.port)
        writer.write(raw_post(spec.run_key, chat_body()))
        await writer.drain()
        assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200")
        writer.write_eof()                                  # FIN: the harness is gone; the socket still accepts writes
        await asyncio.wait_for(reader.read(), 10)           # whatever the server still sends, until it closes
        writer.close()
        await srv.wait_idle(10)
    assert len(env.think_calls()) == 1
    assert [e.delivered for e in srv.reply_log.served[spec.run_key]] == [False]
    assert srv.reply_log.delivered(spec.run_key) == (None, "none")
    record = env.core.close_run(spec.run_key, status="done")
    assert record.final_message is None and record.final_source == "none"
