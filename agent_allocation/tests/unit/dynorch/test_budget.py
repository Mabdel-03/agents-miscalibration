"""Contract §5 / plan B2-B6: grants equal the remaining cap, concurrent calls are serialised and never exceed C,
no 402 is issued while remaining >= the terminal threshold, exactly one grace per run, the E0 ceiling, the cost
guards, and no code path releases a sent grant."""
from __future__ import annotations

import asyncio

import pytest

from agent_allocation.api.fake import FakeReply, FakeScript
from agent_allocation.dynorch.budget import BudgetError, TokenBudget
from agent_allocation.dynorch.records import Message, ProxyError

from .conftest import SUBAGENT_SYSTEM, make_request, make_spec, root_messages


def budget(cap: int = 4096) -> TokenBudget:
    return TokenBudget(cap, terminal_threshold=64, grace=1024, max_calls=24, max_prompt_tokens=600_000)


# ---- TokenBudget alone --------------------------------------------------------------------------------------------

async def test_grant_is_cap_minus_spent_and_shrinks():
    b = budget(4096)
    g1 = await b.acquire()
    assert g1.max_tokens == 4096 and g1.call_no == 1 and b.locked
    b.mark_sent(g1)
    b.settle(g1, 1000, 500)
    assert not b.locked and b.spent == 1000 and b.prompt_total == 500 and b.remaining == 3096
    g2 = await b.acquire()
    assert g2.max_tokens == 3096 and g2.call_no == 2
    b.mark_sent(g2)
    b.settle(g2, 3096, 0)
    assert b.remaining == 0 and b.exhausted_for(True) and b.exhausted_for(False)


async def test_concurrent_acquires_are_fifo_and_never_exceed_the_cap():
    b = budget(2048)
    order: list[int] = []
    granted: list[int] = []

    async def worker(i: int) -> None:
        g = await b.acquire(root=False)
        order.append(i)
        granted.append(g.max_tokens)
        assert g.max_tokens == b.cap - b.spent           # granted == remaining at lock acquisition
        b.mark_sent(g)
        await asyncio.sleep(0)
        b.settle(g, min(600, g.max_tokens), 10)

    async def start(i: int) -> None:
        await asyncio.sleep(i * 0.001)
        await worker(i)

    tasks = [asyncio.create_task(start(i)) for i in range(3)]
    await asyncio.gather(*tasks)
    assert order == [0, 1, 2] and granted == [2048, 1448, 848]
    assert b.spent == 1800 <= b.cap and b.peak_granted <= b.cap


async def test_sent_grant_is_never_released_and_settle_full_charges_the_grant():
    b = budget(1024)
    g = await b.acquire()
    b.mark_sent(g)
    with pytest.raises(BudgetError, match="never released"):
        b.release_unsent(g)
    b.settle_full(g)
    assert b.spent == 1024 and b.remaining == 0
    g_unsent = None
    b2 = budget(1024)
    g_unsent = await b2.acquire()
    b2.release_unsent(g_unsent)                       # never sent: allowed, call count restored
    assert b2.calls == 0 and b2.spent == 0 and not b2.locked


async def test_exactly_one_grace_per_run():
    b = budget(1024)
    b.record_grace("mid_call", 300)
    assert b.grace_used and b.closed and b.grace_tokens == 300 and b.grace_kind == "mid_call"
    with pytest.raises(BudgetError, match="second grace"):
        b.record_grace("terminal", 10)
    with pytest.raises(BudgetError):
        budget().record_grace("ceiling_close", 1)


async def test_e0_root_ceiling_binds_root_grants_only():
    b = budget(16384)
    g = await b.acquire(root=True, root_ceiling=4096)
    assert g.max_tokens == 4096 and g.ceiling_bound
    b.mark_sent(g)
    b.settle(g, 4096, 0)
    g2 = await b.acquire(root=False, root_ceiling=4096)
    assert g2.max_tokens == 16384 - 4096 and not g2.ceiling_bound
    b.mark_sent(g2)
    b.settle(g2, 10000, 0)
    g3 = await b.acquire(root=True, root_ceiling=4096)
    assert g3.max_tokens == 2288 and not g3.ceiling_bound   # the cap, not the ceiling, binds now


async def test_guards_and_overdraft():
    b = TokenBudget(4096, terminal_threshold=64, grace=1024, max_calls=2, max_prompt_tokens=1000)
    assert b.guard() is None
    g = await b.acquire()
    b.mark_sent(g)
    b.settle(g, 10, 600)
    g = await b.acquire()
    b.mark_sent(g)
    b.settle(g, 10, 600)
    assert b.guard() == "call_limit"
    b2 = TokenBudget(4096, terminal_threshold=64, grace=1024, max_calls=24, max_prompt_tokens=1000)
    g = await b2.acquire()
    b2.mark_sent(g)
    b2.settle(g, 4000, 1200)
    assert b2.guard() == "input_limit"
    b2.record_grace("terminal", 500)
    assert b2.overdraft_tokens == 4000 + 500 - 4096
    assert b2.view()["grace_tokens"] == 500


def test_budget_construction_is_validated():
    with pytest.raises(BudgetError):
        TokenBudget(0, terminal_threshold=64, grace=1024, max_calls=24, max_prompt_tokens=10)
    with pytest.raises(BudgetError):
        TokenBudget(1024, terminal_threshold=2048, grace=1024, max_calls=24, max_prompt_tokens=10)


async def test_grant_protocol_errors():
    b = budget()
    with pytest.raises(BudgetError, match="without the run lock"):
        b.grant(root=True, queue_wait_seconds=0.0)
    g = await b.acquire()
    with pytest.raises(BudgetError, match="already active"):
        b.grant(root=True, queue_wait_seconds=0.0)
    with pytest.raises(BudgetError, match="unlock while a grant is active"):
        b.unlock()
    b.mark_sent(g)
    b.settle(g, 1, 1)
    with pytest.raises(BudgetError, match="already settled"):
        b.settle(g, 1, 1)


# ---- through the core against the fake -------------------------------------------------------------------------

async def test_concurrent_fake_calls_never_exceed_c_no_402_above_threshold_one_grace(make_env):
    """Plan §9 Gate 0: N concurrent calls under one run -> serialised grants that never add up beyond C, a 402 only
    at ``remaining < 64`` (never while ``spent < C - 63``), and exactly one grace (the root's terminal call)."""
    script = FakeScript(natural_think_tokens=500)
    script.default = FakeReply(content="<final>1</final>", think_tokens=500)
    env = make_env(script)
    spec = env.open(make_spec(rung=2))               # C = 2048; each solver call realises ~505 tokens
    core = env.core
    outcomes: list[tuple[str, int | None]] = []

    async def solver(i: int) -> None:
        await asyncio.sleep(i * 0.001)
        req = make_request([Message("system", SUBAGENT_SYSTEM), Message("user", f"sub-task {i}")], tools=())
        try:
            reply = await core.chat(spec.run_key, req)
            outcomes.append(("ok", reply.call.granted_max_tokens))
        except ProxyError as exc:
            outcomes.append((exc.code, None))

    await asyncio.gather(*(solver(i) for i in range(8)))
    view = core.budget_view(spec.run_key)
    assert view["spent"] <= 2048
    served = [g for status, g in outcomes if status == "ok"]
    refused = [status for status, _ in outcomes if status != "ok"]
    assert len(served) == 4 and refused == ["budget_exhausted"] * 4
    record = core.record_of(spec.run_key)
    for call in record.calls_detail:
        assert call.granted_max_tokens == 2048 - sum(c.completion_tokens for c in record.calls_detail
                                                     if c.call_no < call.call_no)
        assert call.completion_tokens <= call.granted_max_tokens
    assert view["remaining"] < 64 and record.refused_calls == 4 and record.grace_tokens == 0
    # the root now gets its one terminal grace, then 402 like everyone else
    reply = await core.chat(spec.run_key, make_request(root_messages()))
    assert reply.call.forced_close_kind == "terminal" and reply.call.grace_tokens > 0 and reply.finish_reason == "stop"
    with pytest.raises(ProxyError) as exc:
        await core.chat(spec.run_key, make_request(root_messages(task="again")))
    assert exc.value.code == "budget_exhausted" and exc.value.status == 402
    record = core.record_of(spec.run_key)
    assert record.completion_total <= 2048 and record.grace_tokens <= 1024
    assert sum(1 for c in record.calls_detail if c.forced_close_kind in ("mid_call", "visible_continuation",
                                                                        "terminal")) == 1
    assert env.script.violations == []


async def test_no_code_path_releases_a_sent_grant(make_env):
    """A grant whose call may have left the process is charged whole when the outcome is ambiguous."""
    from agent_allocation.api.fake import Fault

    script = FakeScript()
    script.default = FakeReply(fault=Fault.READ_TIMEOUT)
    env = make_env(script)
    spec = env.open(make_spec(rung=1))
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(root_messages()))
    assert exc.value.code == "upstream_failed"
    view = env.core.budget_view(spec.run_key)
    assert view["spent"] == 1024 and view["remaining"] == 0         # the full grant, conservatively
    snapshot = env.ledger.snapshot()
    assert snapshot.in_flight == 0 and snapshot.unresolved == 0 and snapshot.settled > 0   # charge_max, not release
