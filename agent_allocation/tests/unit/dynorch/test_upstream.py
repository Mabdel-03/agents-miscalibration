"""Contract §5 upstream: the capped body (provider pin in every body, no logit_bias, inside the fake's contract),
retry classes (429 / 5xx / connect retried at most 3 attempts with sibling attempt ids and $0 settlements),
ambiguous outcomes (timeouts, gateway 504, missing usage, cache HIT twice) -> charge_max + upstream_failed, refusals
(402 upstream = infrastructure stop), and a ledger refusal before anything is sent."""
from __future__ import annotations

import asyncio

import json
from decimal import Decimal

import pytest

from agent_allocation.api.fake import FakeReply, FakeScript, Fault, completions_contract_violations, \
    fake_forced_endpoint_lock
from agent_allocation.api.request import RequestContractError, build_capped_body
from agent_allocation.contracts import LedgerEventKind, TransportStatus
from agent_allocation.dynorch.journal import attempt_id
from agent_allocation.dynorch.records import ProxyError
from agent_allocation.dynorch.upstream import RateLimiter, UpstreamFailed, sibling_attempt_id
from agent_allocation.runtime.ledger import journal_attempts

from .conftest import load_cfg

PROMPT = "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n"
COORDS = {"run_key": "rk1", "partition": "main", "phase": "think", "call_no": 1, "item_id": "aime/2025/1"}


def aid(n: int = 1) -> str:
    return attempt_id("rk1", 0, 1, "think", n)


# ---- the body ----------------------------------------------------------------------------------------------------

def test_capped_body_is_the_forced_contract_without_logit_bias():
    cfg = load_cfg()
    lock = fake_forced_endpoint_lock(tag="mancer/fp8")
    body = build_capped_body(PROMPT, lock, cfg["sampling"], max_tokens=4096)
    assert body == {"model": lock.model, "prompt": PROMPT, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                    "max_tokens": 4096,
                    "provider": {"only": ["mancer/fp8"], "allow_fallbacks": False, "require_parameters": True},
                    "usage": {"include": True}, "stream": False}
    assert "logit_bias" not in body and "reasoning" not in body
    assert completions_contract_violations(body) == []


@pytest.mark.parametrize("max_tokens", [0, -1, True, 2.0, 70000])
def test_capped_body_refuses_bad_caps(max_tokens):
    with pytest.raises(RequestContractError):
        build_capped_body(PROMPT, fake_forced_endpoint_lock(), load_cfg()["sampling"], max_tokens=max_tokens)


def test_capped_body_needs_the_endpoint_parameters_and_a_prompt():
    from dataclasses import replace
    lock = fake_forced_endpoint_lock()
    no_top_k = replace(lock, supported_parameters=tuple(p for p in lock.supported_parameters if p != "top_k"))
    with pytest.raises(RequestContractError, match="top_k"):
        build_capped_body(PROMPT, no_top_k, load_cfg()["sampling"], max_tokens=10)
    with pytest.raises(RequestContractError):
        build_capped_body("", lock, load_cfg()["sampling"], max_tokens=10)


def test_sibling_attempt_ids():
    assert sibling_attempt_id("rk:g0:3:think:1", 2) == "rk:g0:3:think:2"
    with pytest.raises(ValueError):
        sibling_attempt_id("nonsense", 2)


# ---- one exchange ------------------------------------------------------------------------------------------------

async def test_ok_call_settles_the_reported_cost_and_pins_the_provider(make_env):
    env = make_env(FakeScript(natural_think_tokens=100))
    result = await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=500, attempt_id=aid(),
                                         coordinates=COORDS)
    assert result.transport_status is TransportStatus.OK and result.closed and result.finish_reason == "stop"
    assert result.usage.completion_tokens <= 500 and result.attempts == (aid(),)
    assert Decimal(result.cost_usd) == result.usage.actual_cost_usd == env.ledger.snapshot().settled
    assert env.ledger.snapshot().in_flight == 0 and env.ledger.snapshot().unresolved == 0
    call = env.think_calls()[0]
    assert call.body["provider"] == {"only": ["parasail/fp8"], "allow_fallbacks": False, "require_parameters": True}
    assert "logit_bias" not in call.body and call.body["max_tokens"] == 500 and call.headers["x-openrouter-cache"] == "false"
    assert call.coordinates["attempt_id"] == aid() and call.coordinates["phase"] == "think"
    phase = result.phases[0]
    assert phase["phase"] == "think" and phase["attempt_id"] == aid() and phase["charged_max"] is False
    assert phase["cost_usd"] == result.cost_usd and phase["response_status"] == 200
    assert env.script.violations == []


async def test_mancer_shard_pins_mancer(make_env):
    env = make_env(FakeScript(natural_think_tokens=50))
    await env.upstream.complete(shard_id="mancer", prompt=PROMPT, max_tokens=200, attempt_id=aid(), coordinates=COORDS)
    assert env.think_calls()[0].body["provider"]["only"] == ["mancer/fp8"]
    with pytest.raises(ProxyError, match="unknown shard"):
        await env.upstream.complete(shard_id="deepinfra", prompt=PROMPT, max_tokens=200, attempt_id=aid(2),
                                    coordinates=COORDS)


@pytest.mark.parametrize("fault", [Fault.RATE_LIMITED, Fault.UPSTREAM_ERROR, Fault.CONNECT_ERROR,
                                   Fault.CONNECT_TIMEOUT])
async def test_retryable_transport_failures_are_retried_with_sibling_ids(make_env, fault):
    script = FakeScript(natural_think_tokens=50)
    script.queue = [FakeReply(fault=fault), FakeReply(fault=fault), FakeReply()]
    env = make_env(script)
    result = await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=200, attempt_id=aid(),
                                         coordinates=COORDS)
    assert result.attempts == (aid(1), aid(2), aid(3)) and env.upstream.retries == 2
    assert [s for s in env.sleeps if s >= 1.0] == [2.0, 4.0]          # the backoff sleeps (rps waits are < 1 s)
    attempts = journal_attempts(env.ledger_journal)
    assert attempts[aid(1)].resolution is LedgerEventKind.SETTLE and attempts[aid(1)].resolved_usd == 0
    assert attempts[aid(2)].resolution is LedgerEventKind.SETTLE and attempts[aid(2)].resolved_usd == 0
    assert attempts[aid(3)].resolution is LedgerEventKind.SETTLE and attempts[aid(3)].resolved_usd > 0
    assert Decimal(result.cost_usd) == attempts[aid(3)].resolved_usd
    assert [p["transport_status"] for p in result.phases][-1] == "ok"
    assert every_body_pinned(env)


def every_body_pinned(env) -> bool:
    calls = [c for c in env.think_calls() if c.body is not None]
    return bool(calls) and all(c.body["provider"]["only"] == ["parasail/fp8"] and "logit_bias" not in c.body
                               for c in calls)


async def test_three_retryable_failures_exhaust_the_attempts_without_charge_max(make_env):
    script = FakeScript()
    script.default = FakeReply(fault=Fault.RATE_LIMITED)
    env = make_env(script)
    with pytest.raises(UpstreamFailed) as exc:
        await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=200, attempt_id=aid(),
                                    coordinates=COORDS)
    assert exc.value.code == "upstream_failed" and exc.value.status == 502 and not exc.value.retryable
    assert exc.value.transport_status is TransportStatus.RATE_LIMITED and not exc.value.charged_max
    assert exc.value.cost_usd == 0 and len(exc.value.phases) == 3
    assert env.ledger.snapshot().settled == 0 and env.ledger.snapshot().in_flight == 0


@pytest.mark.parametrize("reply", [
    FakeReply(fault=Fault.READ_TIMEOUT),
    FakeReply(fault=Fault.REMOTE_DISCONNECT),
    FakeReply(fault=Fault.UPSTREAM_ERROR, status_code=504),
    FakeReply(fault=Fault.NO_USAGE),
    FakeReply(fault=Fault.ERROR_ENVELOPE),
    FakeReply(fault=Fault.MALFORMED_JSON),
])
async def test_ambiguous_outcomes_charge_max_and_fail_without_retry(make_env, reply):
    script = FakeScript(natural_think_tokens=50)
    script.default = reply
    env = make_env(script)
    with pytest.raises(UpstreamFailed) as exc:
        await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=200, attempt_id=aid(),
                                    coordinates=COORDS)
    assert exc.value.charged_max and len(exc.value.phases) == 1 and env.upstream.retries == 0
    reserved = env.upstream.reservation_usd(PROMPT, 200, env.locks["parasail"])
    assert exc.value.cost_usd == reserved
    attempts = journal_attempts(env.ledger_journal)
    assert attempts[aid()].resolution is LedgerEventKind.CHARGE_MAX and attempts[aid()].resolved_usd == reserved
    snapshot = env.ledger.snapshot()
    assert snapshot.settled == reserved and snapshot.in_flight == 0 and snapshot.unresolved == 0


async def test_cache_hit_is_retried_once_then_fails(make_env):
    script = FakeScript(natural_think_tokens=50)
    script.default = FakeReply(fault=Fault.CACHE_HIT)
    env = make_env(script)
    with pytest.raises(UpstreamFailed) as exc:
        await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=200, attempt_id=aid(),
                                    coordinates=COORDS)
    assert exc.value.transport_status is TransportStatus.CACHE_HIT and len(exc.value.phases) == 2
    assert exc.value.charged_max
    script2 = FakeScript(natural_think_tokens=50)
    script2.queue = [FakeReply(fault=Fault.CACHE_HIT), FakeReply()]
    env2 = make_env(script2)
    result = await env2.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=200, attempt_id=aid(),
                                          coordinates=COORDS)
    assert result.attempts == (aid(1), aid(2)) and result.phases[0]["charged_max"] is True


@pytest.mark.parametrize("fault,stop", [(Fault.PAYMENT_REQUIRED, True), (Fault.AUTH_FAILED, True),
                                        (Fault.FORBIDDEN, True), (Fault.BAD_REQUEST, False)])
async def test_refusals_settle_zero_and_flag_infrastructure_stops(make_env, fault, stop):
    script = FakeScript()
    script.default = FakeReply(fault=fault)
    env = make_env(script)
    with pytest.raises(UpstreamFailed) as exc:
        await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=200, attempt_id=aid(),
                                    coordinates=COORDS)
    assert exc.value.infrastructure_stop is stop and exc.value.cost_usd == 0 and not exc.value.charged_max
    assert len(exc.value.phases) == 1 and env.upstream.retries == 0
    assert env.ledger.snapshot().settled == 0


async def test_provider_mismatch_fails_but_settles_the_known_cost(make_env):
    script = FakeScript(natural_think_tokens=50)
    script.default = FakeReply(fault=Fault.PROVIDER_MISMATCH)
    env = make_env(script)
    with pytest.raises(UpstreamFailed) as exc:
        await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=200, attempt_id=aid(),
                                    coordinates=COORDS)
    assert exc.value.transport_status is TransportStatus.PROVIDER_MISMATCH and not exc.value.charged_max
    assert exc.value.cost_usd > 0


async def test_ledger_refusal_happens_before_anything_is_sent(make_env):
    env = make_env(FakeScript(), ceiling_usd="0.0000001")
    with pytest.raises(ProxyError) as exc:
        await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=200, attempt_id=aid(),
                                    coordinates=COORDS)
    assert exc.value.code == "ledger_refused" and exc.value.status == 503 and exc.value.retryable
    assert env.think_calls() == [] and env.ledger.snapshot().in_flight == 0


async def test_reservation_is_conservative_and_covers_the_grant(make_env):
    env = make_env(FakeScript(natural_think_tokens=100))
    lock = env.locks["parasail"]
    amount = env.upstream.reservation_usd(PROMPT, 4096, lock)
    from agent_allocation.runtime.estimate import input_token_bound
    expected = (Decimal(input_token_bound(len(PROMPT))) * Decimal(lock.input_price_per_token)
                + Decimal(4096) * Decimal(lock.output_price_per_token)) * Decimal("1.1")
    assert amount == expected
    result = await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=4096, attempt_id=aid(),
                                         coordinates=COORDS)
    assert Decimal(result.cost_usd) <= amount and env.ledger.overruns == []


async def test_rate_limiter_spaces_sends():
    clock = [0.0]
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    limiter = RateLimiter(4, clock=lambda: clock[0], sleep=sleep)
    for _ in range(3):
        await limiter.wait()
    assert sleeps == [0.25, 0.25]
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_upstream_construction_is_validated(make_env):
    from agent_allocation.dynorch.upstream import Upstream
    env = make_env(FakeScript())
    with pytest.raises(ValueError):
        Upstream(env.client, env.locks, env.cfg["sampling"], {"parasail": 1}, {"parasail": 1, "mancer": 1}, env.ledger)
    with pytest.raises(ValueError):
        Upstream(env.client, env.locks, env.cfg["sampling"], env.settings.provider_in_flight,
                 env.settings.provider_rps, env.ledger, reservation_multiplier=Decimal("0.5"))


# ---- the fake's capped natural prompt (what the proxy relies on) ------------------------------------------------------

async def test_fake_capped_natural_prompt_cuts_unclosed_and_answers_the_continuation(make_env):
    """``api/fake.py`` on a capped natural prompt: ``max_tokens`` below the natural think length -> ``length``,
    unclosed, exactly ``max_tokens`` tokens; the ANSWER continuation (prompt ending ``\\n</think>\\n\\n``) -> the
    scripted content with ``stop``; a cap between the close and the end of the answer -> a consistent ``length``."""
    script = FakeScript()
    script.default = FakeReply(content="<final>9</final>", think_tokens=400)
    env = make_env(script)
    cut = await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=100, attempt_id=aid(1),
                                      coordinates=COORDS)
    assert cut.finish_reason == "length" and not cut.closed and cut.text == "" and cut.usage.completion_tokens == 100
    assert "</think>" not in cut.reasoning and len(cut.reasoning) == 400
    answer = await env.upstream.complete(shard_id="parasail", prompt=PROMPT + cut.reasoning + "\n</think>\n\n",
                                         max_tokens=1024, attempt_id=aid(2), coordinates={**COORDS, "phase": "answer"})
    assert answer.finish_reason == "stop" and answer.text == "<final>9</final>" and answer.closed
    assert answer.reasoning == ""
    tight = await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=402, attempt_id=aid(3),
                                        coordinates=COORDS)
    assert tight.closed and tight.finish_reason == "length" and tight.usage.completion_tokens == 402
    assert tight.text == "\n\n" + "<final>9</final>"[:8]
    script.default = FakeReply(fault=Fault.NO_USAGE)
    with pytest.raises(UpstreamFailed) as exc:
        await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=50, attempt_id=aid(4),
                                    coordinates=COORDS)
    body = json.loads(exc.value.phases[0]["response_body_text"])
    assert "usage" not in body and body["choices"][0]["text"] and exc.value.charged_max


async def test_a_delay_beyond_the_read_timeout_is_ambiguous(make_env):
    from agent_allocation.api import fake
    from agent_allocation.api.forced import ForcedClient, ForcedSettings

    script = FakeScript(natural_think_tokens=50)
    script.default = FakeReply(delay_seconds=0.5)
    env = make_env(script)
    await env.client.aclose()
    forced_cfg = dict(env.cfg)
    forced_cfg["provider"] = {**env.cfg["provider"], "endpoint_tag": "parasail/fp8"}
    env.client = ForcedClient(settings=ForcedSettings.from_config(forced_cfg), api_key=None,
                              transport=fake.fake_openrouter_transport(script), read_timeout=0.05)
    env.upstream.client = env.client
    with pytest.raises(UpstreamFailed) as exc:
        await env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=200, attempt_id=aid(),
                                    coordinates=COORDS)
    assert exc.value.transport_status is TransportStatus.AMBIGUOUS_TIMEOUT and exc.value.charged_max
    assert env.script.billed_usd > 0 and env.script.calls[-1].outcome == "read_timeout"


async def test_a_cancellation_after_mark_sent_never_leaves_an_open_reservation(make_env):
    """Review major 5: a parked call that is cancelled after ``mark_sent`` (the server going away, a stage
    teardown) used to leave a SENT attempt the ledger could never resolve -- ``open_reservations_usd != 0``, a
    Gate stop for nothing, and a ``charge_max`` on the next resume. Unknown money is charged here instead."""
    script = FakeScript()
    script.default = FakeReply(delay_seconds=5.0, content="<final>1</final>")
    env = make_env(script)
    call = asyncio.ensure_future(env.upstream.complete(shard_id="parasail", prompt=PROMPT, max_tokens=200,
                                                       attempt_id=aid(), coordinates=COORDS))
    while env.script.in_flight == 0:
        await asyncio.sleep(0.01)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    snapshot = env.ledger.snapshot()
    assert snapshot.in_flight == 0 and snapshot.unresolved == 0 and snapshot.settled > 0
    assert env.ledger.state_of(aid()) == "charge_max"
