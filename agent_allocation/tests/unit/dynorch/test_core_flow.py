"""Contract §3 ``ProxyCore`` flow against the fake upstream: natural close, forced close (B3), visible continuation,
terminal path (B4), sub-agent truncation, 402 after exhaustion, tool-surface 400, ledger 503, duplicates (B9),
re-injection, the finish-reason set, usage accounting, classification, the E0 ceiling, cost guards (B6), the
delivered answer (B7) and the frozen config."""
from __future__ import annotations

import copy
import json
from decimal import Decimal

import pytest

from agent_allocation.api.fake import FakeReply, FakeScript, Fault, fake_tool_call_text
from agent_allocation.api.forced import answer_prompt
from agent_allocation.dynorch.config import PROXY_VERSION, ConfigError, DynorchSettings
from agent_allocation.dynorch.core import ProxyCore
from agent_allocation.dynorch.records import (
    FINISH_REASONS_RETURNED, CallRecord, ChatReply, Message, ProxyError, ReplyUsage, ToolCall,
)
from agent_allocation.dynorch.template import THINK_DISABLED_SUFFIX

from .conftest import (
    AGENT_TOOL, Env, H0_SYSTEM, ROOT_SYSTEM, SPAWN_TOOL, SUBAGENT_SYSTEM, load_cfg, make_request, make_spec,
    root_messages,
)

CLOSE = "\n</think>\n\n"
FINAL = "<rationale>r</rationale>\n<final>42</final>"


def solver_request(task: str = "sub-task", **kw):
    return make_request([Message("system", SUBAGENT_SYSTEM), Message("user", task)], tools=(), **kw)


# ---- natural close and forced close (B3) ---------------------------------------------------------------------------

async def test_natural_close_makes_no_answer_call(make_env):
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=300)
    env = make_env(script)
    spec = env.open(make_spec(rung=4))
    reply = await env.core.chat(spec.run_key, make_request(root_messages()))
    assert reply.finish_reason == "stop" and reply.content == FINAL and reply.tool_calls == ()
    assert reply.call.forced_close_kind == "none" and reply.call.grace_tokens == 0
    assert reply.call.granted_max_tokens == 4096 and reply.call.completion_tokens == reply.usage.completion_tokens
    assert len(env.think_calls()) == 1 and env.think_calls()[0].body["max_tokens"] == 4096
    assert reply.headers["X-Dynorch-Forced-Close"] == "none" and reply.headers["X-Dynorch-Conversation"] == "root"
    assert reply.headers["X-Dynorch-Remaining-Tokens"] == str(4096 - reply.usage.completion_tokens)
    assert reply.call.requested_max_tokens == 32000 and reply.call.reinjection == "n/a"
    assert reply.reasoning_content and "</think>" not in reply.reasoning_content
    assert env.core.record_of(spec.run_key).calls == 1 and env.store.load(spec).calls == 1


async def test_forced_close_builds_exactly_the_forced_v2_answer_prompt(make_env):
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=5000)
    env = make_env(script)
    spec = env.open(make_spec(rung=1))
    reply = await env.core.chat(spec.run_key, make_request(root_messages()))
    think, answer = env.think_calls(phase="think")[0], env.think_calls(phase="answer")[0]
    assert think.body["max_tokens"] == 1024
    assert answer.body["prompt"] == answer_prompt(think.body["prompt"], reply.reasoning_content, CLOSE)
    assert answer.body["max_tokens"] == 1024 and "logit_bias" not in answer.body
    assert answer.body["prompt"].endswith(CLOSE) and "# Tools" in think.body["prompt"]
    assert reply.finish_reason == "stop" and reply.content == FINAL
    assert reply.call.forced_close_kind == "mid_call" and reply.headers["X-Dynorch-Forced-Close"] == "mid_call"
    assert reply.call.finish_reason_upstream == "length"
    assert len(env.think_calls()) == 2
    record = env.core.record_of(spec.run_key)
    assert record.forced_close_kind == "mid_call" and record.exhausted and record.completion_total == 1024
    assert record.grace_tokens == reply.call.grace_tokens > 0 and record.overdraft_tokens == record.grace_tokens
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(root_messages(task="again")))
    assert exc.value.code == "budget_exhausted" and exc.value.status == 402 and not exc.value.retryable


async def test_forced_close_pair_counts_think_and_answer_once(make_env):
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=5000)
    env = make_env(script)
    spec = env.open(make_spec(rung=1))
    reply = await env.core.chat(spec.run_key, make_request(root_messages()))
    think, answer = env.think_calls(phase="think")[0], env.think_calls(phase="answer")[0]
    think_usage = json.loads(env.script.generations[think.generation_id]["tokens_completion"] and "0")
    t = env.script.generations[think.generation_id]["tokens_completion"]
    a = env.script.generations[answer.generation_id]["tokens_completion"]
    assert t == 1024 and a > 0
    assert reply.usage.completion_tokens == t + a
    assert reply.call.completion_tokens == t and reply.call.grace_tokens == a
    record = env.core.record_of(spec.run_key)
    assert record.completion_total == t and record.grace_tokens == a and record.calls == 1
    assert record.completion_total + record.grace_tokens == reply.usage.completion_tokens
    assert reply.usage.prompt_tokens == (env.script.generations[think.generation_id]["tokens_prompt"]
                                         + env.script.generations[answer.generation_id]["tokens_prompt"])
    assert Decimal(record.cost_usd) == Decimal(reply.call.cost_usd) == env.ledger.snapshot().settled
    assert reply.call.generation_ids == (think.generation_id, answer.generation_id)


async def test_visible_continuation_when_the_cap_binds_after_the_close(make_env):
    script = FakeScript()
    script.default = FakeReply(content="X" * 500, think_tokens=1000)     # 1000 + 125 > 1024: cut after </think>
    env = make_env(script)
    spec = env.open(make_spec(rung=1))
    reply = await env.core.chat(spec.run_key, make_request(root_messages()))
    think, cont = env.think_calls(phase="think")[0], env.think_calls(phase="answer")[0]
    assert reply.call.forced_close_kind == "visible_continuation" and reply.call.finish_reason_upstream == "length"
    head = answer_prompt(think.body["prompt"], reply.reasoning_content, CLOSE)
    assert cont.body["prompt"].startswith(head) and cont.body["prompt"].endswith("X" * 96)
    assert cont.body["max_tokens"] == 1024
    assert reply.content == "X" * 96 + "X" * 500 and reply.finish_reason == "stop"
    record = env.core.record_of(spec.run_key)
    assert record.completion_total == 1024 and record.grace_tokens == reply.call.grace_tokens > 0


async def test_forced_close_only_once_per_run_root_only(make_env):
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=5000)
    env = make_env(script)
    spec = env.open(make_spec(rung=2))
    # a sub-agent that hits its grant is truncated, never forced-closed
    sub = await env.core.chat(spec.run_key, solver_request())
    assert sub.call.forced_close_kind == "truncated_subagent" and sub.finish_reason == "stop"
    assert sub.content == "" and sub.call.grace_tokens == 0 and sub.call.completion_tokens == 2048
    assert len(env.think_calls(phase="answer")) == 0 and env.core.budget_view(spec.run_key)["remaining"] == 0
    # the root, at a boundary with remaining < 64, gets the terminal call (its single grace)
    root = await env.core.chat(spec.run_key, make_request(root_messages()))
    assert root.call.forced_close_kind == "terminal" and root.finish_reason == "stop"
    for req in (make_request(root_messages(task="more")), solver_request("later")):
        with pytest.raises(ProxyError) as exc:
            await env.core.chat(spec.run_key, req)
        assert exc.value.code == "budget_exhausted"
    record = env.core.record_of(spec.run_key)
    assert record.forced_close_kind == "terminal" and record.refused_calls == 2 and record.exhausted


# ---- terminal path (B4) --------------------------------------------------------------------------------------------

async def test_terminal_path_renders_the_verbatim_sentence_without_thinking_or_tools(make_env):
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=1000)
    env = make_env(script)
    spec = env.open(make_spec(rung=1))
    first = await env.core.chat(spec.run_key, make_request(root_messages()))
    assert first.call.forced_close_kind == "none" and env.core.budget_view(spec.run_key)["remaining"] < 64
    history = root_messages() + [Message("assistant", first.content, reasoning_content=first.reasoning_content),
                                 Message("user", "Please double-check.")]
    reply = await env.core.chat(spec.run_key, make_request(history))
    call = env.think_calls(phase="terminal")[0]
    prompt = call.body["prompt"]
    assert prompt.endswith("<|im_start|>user\n" + env.settings.terminal_sentence + "<|im_end|>\n"
                           + THINK_DISABLED_SUFFIX)
    assert "# Tools" not in prompt and "<tools>" not in prompt and call.body["max_tokens"] == 1024
    assert "Please double-check." in prompt and first.reasoning_content in prompt
    assert reply.finish_reason == "stop" and reply.content == FINAL and reply.reasoning_content == ""
    assert reply.call.forced_close_kind == "terminal" and reply.call.completion_tokens == 0
    assert reply.call.grace_tokens == reply.usage.completion_tokens > 0
    record = env.core.record_of(spec.run_key)
    assert record.exhausted and record.grace_tokens == reply.call.grace_tokens and record.calls == 2
    assert record.completion_total == first.usage.completion_tokens      # the terminal call is outside C


async def test_the_terminal_prompt_carries_reinjected_reasoning_and_returns_no_tool_calls(make_env):
    """Review minor 25: the TERMINAL call renders the RE-INJECTED messages (OpenClaude never re-sends
    ``reasoning_content``, so dropping the cache here empties every prior think block although the hits were
    already counted), and its reply is the final answer verbatim -- B4 offers no tools, so a tool-call-shaped
    block must not be parsed away from the delivered content."""
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=1000)
    tool_text = fake_tool_call_text("Agent", {"prompt": "go", "subagent_type": "solver"}, prefix="<final>8</final>")
    script.hook = lambda call: (FakeReply(content=tool_text)
                                if call.coordinates.get("phase") == "terminal" else None)
    env = make_env(script)
    spec = env.open(make_spec(rung=1))
    first = await env.core.chat(spec.run_key, make_request(root_messages()))
    assert first.reasoning_content
    history = root_messages() + [Message("assistant", first.content),       # no reasoning_content, like OpenClaude
                                 Message("user", "Please double-check.")]
    reply = await env.core.chat(spec.run_key, make_request(history))
    prompt = env.think_calls(phase="terminal")[0].body["prompt"]
    assert first.reasoning_content in prompt, "the terminal prompt lost the cached reasoning of the prior turn"
    assert reply.call.reinjection == "hit" and reply.call.forced_close_kind == "terminal"
    assert reply.finish_reason == "stop" and reply.tool_calls == () and reply.call.tool_calls == 0
    assert reply.content == tool_text.strip() and reply.call.tool_parse == "none"


async def test_terminal_path_is_root_only_a_solver_at_the_boundary_gets_402(make_env):
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=1000)
    env = make_env(script)
    spec = env.open(make_spec(rung=1))
    await env.core.chat(spec.run_key, make_request(root_messages()))
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, solver_request())
    assert exc.value.code == "budget_exhausted" and len(env.think_calls(phase="terminal")) == 0
    assert not env.core.record_of(spec.run_key).exhausted or env.core.budget_view(spec.run_key)["grace_tokens"] == 0
    # (an identical root re-request would be a B9 duplicate served from the cache; a new turn reaches the boundary)
    reply = await env.core.chat(spec.run_key, make_request(root_messages(task="TASK\nnext turn\n")))
    assert reply.call.forced_close_kind == "terminal"


# ---- refusals ------------------------------------------------------------------------------------------------------

async def test_tool_surface_violation_is_a_400_and_flags_the_run(make_env):
    env = make_env(FakeScript())
    spec = env.open(make_spec(rung=4))
    from agent_allocation.dynorch.records import ToolSpec
    bash = ToolSpec("Bash", "run", {"type": "object", "properties": {"cmd": {"type": "string"}}})
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(root_messages(), tools=(AGENT_TOOL, bash)))
    assert exc.value.code == "tool_surface_violation" and exc.value.status == 400
    assert env.think_calls() == [] and env.store.load(spec).error == "tool_surface_violation: Bash"
    h0 = env.open(make_spec(rung=4, arm="H0", harness="null", allowed_tools=(), spawn_tools=(), item_id="aime/2025/9"))
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(h0.run_key, make_request(root_messages(system=H0_SYSTEM), tools=(AGENT_TOOL,)))
    assert exc.value.code == "tool_surface_violation"
    reply = await env.core.chat(h0.run_key, make_request(root_messages(system=H0_SYSTEM), tools=()))
    assert reply.finish_reason == "stop" and reply.call.tool_parse == "none"


async def test_ledger_refusal_is_a_503_and_releases_the_unsent_grant(make_env):
    env = make_env(FakeScript(), ceiling_usd="0.000001")
    spec = env.open(make_spec(rung=4))
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(root_messages()))
    assert exc.value.code == "ledger_refused" and exc.value.status == 503 and exc.value.retryable
    view = env.core.budget_view(spec.run_key)
    assert {k: view[k] for k in ("cap", "spent", "remaining", "calls", "grace_tokens", "prompt_total", "exhausted")} \
        == {"cap": 4096, "spent": 0, "remaining": 4096, "calls": 0, "grace_tokens": 0, "prompt_total": 0,
            "exhausted": 0}
    assert view["in_flight"] == 0 and view["replies_served"] == 0      # B4 clock: nothing was served
    assert env.think_calls() == [] and not env.core.record_of(spec.run_key).exhausted


async def test_system_message_after_the_head_is_a_400_not_a_render_500(make_env):
    """The pinned template refuses a system message anywhere but first; the core answers bad_request before rendering
    (the chat route maps ``developer`` to ``system`` at any position)."""
    env = make_env(FakeScript())
    spec = env.open(make_spec(rung=4))
    messages = root_messages() + [Message("assistant", "ok"), Message("system", "late note"), Message("user", "go")]
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(messages))
    assert exc.value.code == "bad_request" and exc.value.status == 400
    assert env.think_calls() == [] and env.core.budget_view(spec.run_key)["calls"] == 0


async def test_unknown_and_closed_runs(make_env):
    env = make_env(FakeScript())
    with pytest.raises(ProxyError) as exc:
        await env.core.chat("nope", make_request(root_messages()))
    assert exc.value.code == "unknown_run" and exc.value.status == 404
    spec = env.open(make_spec(rung=1))
    with pytest.raises(ValueError):
        env.core.open_run(spec, root_system_text=ROOT_SYSTEM, subagent_system_text=None)
    record = env.core.close_run(spec.run_key, status="timeout")
    assert record.status == "timeout" and record.final_source == "none" and record.finished_at
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(root_messages()))
    assert exc.value.code == "run_closed" and exc.value.status == 409
    with pytest.raises(ValueError):
        env.core.close_run(spec.run_key, status="running")


async def test_open_run_validates_the_spec(make_env):
    env = make_env(FakeScript())
    with pytest.raises(ValueError, match="proxy_version"):
        from dataclasses import replace
        env.core.open_run(replace(make_spec(), proxy_version="old"), root_system_text=ROOT_SYSTEM,
                          subagent_system_text=None)
    with pytest.raises(ValueError, match="cap_tokens"):
        from dataclasses import replace
        env.core.open_run(replace(make_spec(rung=4), cap_tokens=4000), root_system_text=ROOT_SYSTEM,
                          subagent_system_text=None)
    with pytest.raises(ValueError, match="shard"):
        env.core.open_run(make_spec(shard_id="deepinfra"), root_system_text=ROOT_SYSTEM, subagent_system_text=None)


async def test_upstream_failure_settles_the_full_grant_and_marks_the_run(make_env):
    script = FakeScript()
    script.default = FakeReply(fault=Fault.READ_TIMEOUT)
    env = make_env(script)
    spec = env.open(make_spec(rung=2))
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(root_messages()))
    assert exc.value.code == "upstream_failed" and exc.value.status == 502
    record = env.core.record_of(spec.run_key)
    assert record.completion_total == 2048 and record.calls == 1 and record.error.startswith("upstream_failed")
    assert record.calls_detail[0].transport_status == "ambiguous_timeout" and record.calls_detail[0].returned_status == 502
    stored = env.journal.read(spec, 1)
    assert stored["reply"] is None and stored["error"]["code"] == "upstream_failed" and stored["error"]["charged_max"]
    assert env.core.budget_view(spec.run_key)["remaining"] == 0


async def test_upstream_402_is_an_infrastructure_stop_not_an_exhaustion(make_env):
    script = FakeScript()
    script.default = FakeReply(fault=Fault.PAYMENT_REQUIRED)
    env = make_env(script)
    spec = env.open(make_spec(rung=2))
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(root_messages()))
    assert exc.value.code == "upstream_failed"
    record = env.core.record_of(spec.run_key)
    assert record.error.startswith("infrastructure_stop") and not record.exhausted and record.refused_calls == 0


# ---- provider-side cuts are transport outcomes, not the cap binding (B3, review major 9) -----------------------

async def test_a_provider_cut_is_a_transport_failure_not_a_forced_close(make_env):
    """finish_reason null / error / content_filter with most of the grant unspent is forced_v2's
    ``forced_think_short``: it must NOT spend the run's one grace and 402 the rest of an unspent budget."""
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=100, finish_reason=None)
    env = make_env(script)
    spec = env.open(make_spec(rung=4))
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(root_messages()))
    assert exc.value.code == "upstream_failed" and exc.value.status == 502
    record = env.core.record_of(spec.run_key)
    assert record.forced_close_kind == "none" and record.grace_tokens == 0 and not record.exhausted
    assert len(env.think_calls(phase="answer")) == 0             # no ANSWER continuation was made
    assert record.calls == 1 and record.completion_total == record.calls_detail[0].completion_tokens > 0
    view = env.core.budget_view(spec.run_key)
    assert view["remaining"] == 4096 - record.completion_total and view["grace_tokens"] == 0
    # the budget is intact: the next call is served, not refused
    script.default = FakeReply(content=FINAL, think_tokens=100)
    again = await env.core.chat(spec.run_key, make_request(root_messages(task="TASK\nsecond\n")))
    assert again.finish_reason == "stop" and again.call.granted_max_tokens == view["remaining"]


async def test_length_below_the_grant_is_a_transport_failure_on_root_and_on_a_solver(make_env):
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=100, finish_reason="length")
    env = make_env(script)
    spec = env.open(make_spec(rung=4))
    with pytest.raises(ProxyError) as root_exc:
        await env.core.chat(spec.run_key, make_request(root_messages()))
    assert root_exc.value.code == "upstream_failed"
    assert env.core.record_of(spec.run_key).grace_tokens == 0
    with pytest.raises(ProxyError) as solver_exc:
        await env.core.chat(spec.run_key, solver_request())
    assert solver_exc.value.code == "upstream_failed"
    assert all(c.forced_close_kind == "none" for c in env.core.record_of(spec.run_key).calls_detail)


async def test_length_at_the_grant_is_still_the_cap_binding(make_env):
    """The narrowing must not turn a real cap-bound call into a transport failure."""
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=99999)
    env = make_env(script)
    spec = env.open(make_spec(rung=1))
    reply = await env.core.chat(spec.run_key, make_request(root_messages()))
    assert reply.call.forced_close_kind == "mid_call" and reply.call.finish_reason_upstream == "length"
    assert env.core.record_of(spec.run_key).completion_total == 1024


# ---- every settled grant is journaled (review major 16) --------------------------------------------------------

async def test_a_ledger_refusal_between_think_and_answer_still_journals_the_paid_think(make_env):
    """The THINK exchange was paid and settled; if the $ ledger refuses the ANSWER continuation the call must
    still appear in the journal (its generation id reconciles against /generation) and in ``calls_detail``."""
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=5000)
    env = make_env(script)
    spec = env.open(make_spec(rung=1))
    original = env.ledger.reserve

    async def gated(*, attempt_id: str, **kw):
        if ":answer:" in attempt_id:
            env.ledger.last_refusal = "ceiling reached between THINK and ANSWER"
            return False
        return await original(attempt_id=attempt_id, **kw)

    env.ledger.reserve = gated
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(root_messages()))
    assert exc.value.code == "ledger_refused"
    journaled = env.journal.scan(spec)
    assert len(journaled) == 1 and journaled[0]["call_no"] == 1
    phases = journaled[0]["phases"]
    assert [p["phase"] for p in phases] == ["think"] and phases[0]["generation_id"]
    assert journaled[0]["error"]["code"] == "ledger_refused"
    record = env.core.record_of(spec.run_key)
    assert record.calls == 1 and len(record.calls_detail) == 1
    assert record.calls_detail[0].completion_tokens == record.completion_total == 1024
    assert record.grace_tokens == 1024                # conservative: the grace is booked as spent
    assert Decimal(record.cost_usd) > 0


async def test_an_e0_ceiling_continuation_that_fails_settles_the_full_amount_inside_c(make_env):
    """B2 minor 24: an ambiguous ANSWER outcome on an E0 ceiling close settles think + cont_max inside C (the
    clamp to the ceiling-bound grant lost up to G tokens that may have been generated)."""
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=99999)
    script.hook = lambda call: (FakeReply(fault=Fault.READ_TIMEOUT)
                                if call.coordinates.get("phase") == "answer" else None)
    env = make_env(script)
    spec = env.open(make_spec(rung=16, arm="H1c", root_call_ceiling=4096))
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(root_messages()))
    assert exc.value.code == "upstream_failed"
    view = env.core.budget_view(spec.run_key)
    assert view["spent"] == 4096 + 1024 and view["grace_tokens"] == 0    # inside C, no grace on a ceiling close


# ---- duplicates (B9) ------------------------------------------------------------------------------------------------

async def test_a_disconnected_reply_is_cached_and_its_identical_retry_is_served_from_it(make_env):
    """B9 (contract §3): the cache holds ONLY the reply of a call whose client went away after ``mark_sent``; an
    identical re-request of that same conversation is then served from it with no new upstream call."""
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=200)
    env = make_env(script)
    spec = env.open(make_spec(rung=4))
    first = await env.core.chat(spec.run_key, make_request(root_messages()))
    env.core.mark_undelivered(spec.run_key, first.call.call_no)      # the wire layer could not write it
    again = await env.core.chat(spec.run_key, make_request(root_messages(), stream=True))
    assert len(env.think_calls()) == 1
    assert again.content == first.content and again.tool_calls == first.tool_calls
    assert again.reasoning_content == first.reasoning_content and again.usage == first.usage
    assert again.call.duplicate_served and again.call.granted_max_tokens == 0 and again.call.stream is True
    assert again.headers["X-Dynorch-Duplicate"] == "1"
    record = env.core.record_of(spec.run_key)
    assert record.duplicates_served == 1 and record.calls == 1 and len(record.calls_detail) == 2
    assert record.completion_total == first.usage.completion_tokens         # counted once
    assert env.core.budget_view(spec.run_key)["calls"] == 1
    # a different prompt is not a duplicate
    other = await env.core.chat(spec.run_key, make_request(root_messages(task="TASK\nother\n")))
    assert not other.call.duplicate_served and len(env.think_calls()) == 2


async def test_a_delivered_reply_is_never_reused_for_a_later_identical_request(make_env):
    """Review blocker 3: without the disconnect condition any second request with the same rendered prompt was
    served the first answer. A reply the client received is a completed exchange; asking again is a new call."""
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=200)
    env = make_env(script)
    spec = env.open(make_spec(rung=16))
    first = await env.core.chat(spec.run_key, make_request(root_messages()))
    again = await env.core.chat(spec.run_key, make_request(root_messages()))
    assert len(env.think_calls()) == 2 and not again.call.duplicate_served
    assert again.call.granted_max_tokens == spec.cap_tokens - first.usage.completion_tokens
    record = env.core.record_of(spec.run_key)
    assert record.calls == 2 and record.duplicates_served == 0
    assert record.completion_total == first.usage.completion_tokens + again.usage.completion_tokens


async def test_two_sub_agents_with_identical_prompts_each_get_their_own_generation(make_env):
    """Review blocker 3 (scientific correctness): best-of-n delegation spawns several solvers with the SAME task
    text; their rendered prompts are byte-identical. Each conversation must draw its own sample -- one cached
    answer for all of them would delete the diversity the study measures and under-count C."""
    script = FakeScript()
    script.default = FakeReply(content="<final>1</final>", think_tokens=200)
    env = make_env(script)
    spec = env.open(make_spec(rung=16))
    first = await env.core.chat(spec.run_key, solver_request("solve the whole problem"))
    second = await env.core.chat(spec.run_key, solver_request("solve the whole problem"))
    assert len(env.think_calls()) == 2, "two sub-agents were served one sample"
    assert not first.call.duplicate_served and not second.call.duplicate_served
    assert second.call.granted_max_tokens == spec.cap_tokens - first.usage.completion_tokens > 0
    record = env.core.record_of(spec.run_key)
    assert record.calls_by_class["solver"] == 2 and record.duplicates_served == 0
    assert record.completion_total == first.usage.completion_tokens + second.usage.completion_tokens


async def test_a_cached_disconnected_reply_is_not_served_to_another_conversation(make_env):
    """The cache key carries the conversation: a solver whose reply was never delivered must not answer the root
    (or another solver) even when the rendered prompts coincide."""
    script = FakeScript()
    script.default = FakeReply(content="<final>1</final>", think_tokens=200)
    env = make_env(script)
    spec = env.open(make_spec(rung=16))
    first = await env.core.chat(spec.run_key, solver_request("same text"))
    env.core.mark_undelivered(spec.run_key, first.call.call_no)
    retry = await env.core.chat(spec.run_key, solver_request("same text"))
    assert retry.call.duplicate_served and len(env.think_calls()) == 1
    # the same prompt text under the ROOT system prompt is a different conversation: a fresh generation
    root_same = await env.core.chat(spec.run_key, make_request(root_messages(task="same text"), tools=()))
    assert not root_same.call.duplicate_served and len(env.think_calls()) == 2


# ---- re-injection (§4) ------------------------------------------------------------------------------------------------

async def test_reinjection_hit_by_tool_call_id_miss_and_explicit(make_env):
    script = FakeScript()
    script.default = FakeReply(content=fake_tool_call_text("Agent", {"prompt": "go", "subagent_type": "solver"},
                                                           prefix="Delegating."), think_tokens=120)
    env = make_env(script)
    spec = env.open(make_spec(rung=8))
    first = await env.core.chat(spec.run_key, make_request(root_messages()))
    assert first.finish_reason == "tool_calls" and first.tool_calls[0].id == "call_1_0"
    assert first.call.spawn_calls == 1 and env.core.record_of(spec.run_key).spawns == 1
    reasoning = first.reasoning_content
    # (1) OpenClaude style: content coalesced/edited, no reasoning_content, the proxy-minted id echoed -> hit
    script.default = FakeReply(content=FINAL, think_tokens=100)
    history = root_messages() + [Message("assistant", "[Tool execution interrupted by user]Delegating.",
                                         tool_calls=first.tool_calls),
                                 Message("tool", "<final>3</final>", tool_call_id="call_1_0")]
    hit = await env.core.chat(spec.run_key, make_request(history))
    assert hit.call.reinjection == "hit"
    assert ("<think>\n" + reasoning + "\n</think>\n\n[Tool execution interrupted by user]Delegating."
            in env.think_calls()[1].body["prompt"])          # content is rendered verbatim, only the key is normalised
    # (2) content key: ids dropped but the served content re-sent (with a re-sent reasoning block stripped) -> hit
    history2 = root_messages() + [Message("assistant", "<think>\nstale\n</think>\n\nDelegating."),
                                  Message("user", "next")]
    hit2 = await env.core.chat(spec.run_key, make_request(history2))
    assert hit2.call.reinjection == "hit" and reasoning in env.think_calls()[2].body["prompt"]
    # (3) a rewritten assistant turn the proxy never produced -> miss, empty think block
    history3 = root_messages() + [Message("assistant", "Something the harness invented."), Message("user", "next")]
    miss = await env.core.chat(spec.run_key, make_request(history3))
    assert miss.call.reinjection == "miss"
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\nSomething the harness invented." in env.think_calls()[3].body["prompt"]
    # (4) explicit reasoning from the harness wins over the cache
    history4 = root_messages() + [Message("assistant", "Delegating.", reasoning_content="HARNESS REASONING",
                                          tool_calls=first.tool_calls),
                                  Message("tool", "<final>3</final>", tool_call_id="call_1_0")]
    explicit = await env.core.chat(spec.run_key, make_request(history4))
    assert explicit.call.reinjection == "explicit"
    prompt = env.think_calls()[4].body["prompt"]
    assert "<think>\nHARNESS REASONING\n</think>" in prompt and reasoning not in prompt
    record = env.core.record_of(spec.run_key)
    assert record.reinjection_hits == 2 and record.reinjection_misses == 1
    # a synthetic empty assistant turn is neither a hit nor a miss
    history5 = root_messages() + [Message("assistant", "[Tool execution interrupted by user]"), Message("user", "n")]
    synthetic = await env.core.chat(spec.run_key, make_request(history5))
    assert synthetic.call.reinjection == "n/a" and env.core.record_of(spec.run_key).reinjection_misses == 1


# ---- finish reasons and tool parsing ---------------------------------------------------------------------------------

async def test_finish_reason_set_is_exactly_stop_and_tool_calls(make_env):
    script = FakeScript()
    env = make_env(script)
    spec = env.open(make_spec(rung=4))
    seen = set()
    for reply_spec, msgs in [
        (FakeReply(content=FINAL, think_tokens=100), root_messages()),
        (FakeReply(content=fake_tool_call_text("Agent", {"prompt": "p", "subagent_type": "solver"}), think_tokens=100),
         root_messages(task="T2")),
        (FakeReply(content=FINAL, think_tokens=9000), root_messages(task="T3")),       # forced close
    ]:
        script.default = reply_spec
        seen.add((await env.core.chat(spec.run_key, make_request(msgs))).finish_reason)
    assert seen == {"stop", "tool_calls"} == set(FINISH_REASONS_RETURNED) == set(env.settings.finish_reasons_returned)
    call = env.core.record_of(spec.run_key).calls_detail[0]
    with pytest.raises(ValueError):
        ChatReply(content="", reasoning_content="", tool_calls=(), finish_reason="length",
                  usage=ReplyUsage(None, None, None, None), call=call, headers={})


async def test_malformed_tool_call_is_returned_as_content_and_counted(make_env):
    script = FakeScript()
    script.default = FakeReply(content="<tool_call>\n<function=Agent>\n<parameter=prompt>\nunclosed", think_tokens=100)
    env = make_env(script)
    spec = env.open(make_spec(rung=4))
    reply = await env.core.chat(spec.run_key, make_request(root_messages()))
    assert reply.finish_reason == "stop" and reply.tool_calls == () and reply.call.tool_parse == "failed"
    assert reply.content.startswith("<tool_call>") and reply.headers["X-Dynorch-Tool-Parse"] == "failed"
    assert env.core.record_of(spec.run_key).tool_parse_failures == 1


async def test_parallel_calls_and_spawn_counting_on_root_only(make_env):
    script = FakeScript()
    xml = fake_tool_call_text("Agent", {"prompt": "a", "subagent_type": "solver"}) + "\n" + \
        fake_tool_call_text("Agent", {"prompt": "b", "subagent_type": "solver"})
    script.default = FakeReply(content=xml, think_tokens=100)
    env = make_env(script)
    spec = env.open(make_spec(rung=8))
    root = await env.core.chat(spec.run_key, make_request(root_messages()))
    assert [c.id for c in root.tool_calls] == ["call_1_0", "call_1_1"] and root.call.spawn_calls == 2
    sub = await env.core.chat(spec.run_key, make_request([Message("system", SUBAGENT_SYSTEM), Message("user", "a")],
                                                         tools=(AGENT_TOOL,)))
    assert sub.call.conversation_class == "solver" and sub.call.spawn_calls == 0 and sub.finish_reason == "tool_calls"
    record = env.core.record_of(spec.run_key)
    assert record.spawns == 2 and record.calls_by_class == {"root": 1, "solver": 1, "other": 0}
    assert record.conversations == 2


# ---- classification (B1) ---------------------------------------------------------------------------------------------

async def test_conversation_classes_root_solver_other_and_codex_style(make_env):
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=50)
    env = make_env(script)
    spec = env.open(make_spec(rung=8, arm="H3", harness="codex", allowed_tools=("spawn_agent",),
                              spawn_tools=("spawn_agent",)))
    codex_system = ROOT_SYSTEM + "\n\n# Multi-agent\nYou can spawn agents."
    root = await env.core.chat(spec.run_key, make_request(root_messages(system=codex_system), tools=(SPAWN_TOOL,)))
    assert root.call.conversation_class == "root"
    fork = await env.core.chat(spec.run_key, make_request(root_messages(system=codex_system) + [
        Message("user", "child task")], tools=(SPAWN_TOOL,), subagent=True))
    assert fork.call.conversation_class == "solver"                  # x-openai-subagent -> solver
    solver = await env.core.chat(spec.run_key, make_request([Message("system", SUBAGENT_SYSTEM + "\nenv details"),
                                                             Message("user", "x")], tools=()))
    assert solver.call.conversation_class == "solver"
    other = await env.core.chat(spec.run_key, make_request([Message("system", "Some hidden harness prompt"),
                                                            Message("user", "summarise")], tools=()))
    assert other.call.conversation_class == "other"
    record = env.core.record_of(spec.run_key)
    assert record.calls_by_class == {"root": 1, "solver": 2, "other": 1} and record.conversations == 4
    # a first conversation whose system text is unknown is still the root (Codex appends its own text)
    spec2 = env.open(make_spec(rung=4, item_id="aime/2025/7"))
    first = await env.core.chat(spec2.run_key, make_request([Message("system", "totally different"),
                                                             Message("user", "q")], tools=()))
    assert first.call.conversation_class == "root"
    with pytest.raises(ProxyError, match="no user message"):
        await env.core.chat(spec2.run_key, make_request([Message("system", "s")], tools=()))


# ---- delivered answer (B7) and persistence ------------------------------------------------------------------------------

async def test_close_run_delivers_the_last_non_empty_root_content(make_env):
    script = FakeScript()
    env = make_env(script)
    spec = env.open(make_spec(rung=8))
    script.default = FakeReply(content="<final>1</final>", think_tokens=50)
    await env.core.chat(spec.run_key, make_request(root_messages()))
    script.default = FakeReply(content=fake_tool_call_text("Agent", {"prompt": "p", "subagent_type": "solver"}),
                               think_tokens=50)
    await env.core.chat(spec.run_key, make_request(root_messages(task="T2")))            # tool-call-only turn
    script.default = FakeReply(content="<final>solver</final>", think_tokens=50)
    await env.core.chat(spec.run_key, solver_request())                                   # sub-agent text never used
    record = env.core.close_run(spec.run_key, status="done")
    assert record.final_message == "<final>1</final>" and record.final_source == "proxy_root"
    assert record.status == "done" and record.wall_seconds >= 0 and env.store.load(spec) == record
    assert record.calls == 3 and len(record.calls_detail) == 3 and record.reasoning_total is not None


async def test_record_is_persisted_after_every_call_and_headers_are_complete(make_env):
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=50)
    env = make_env(script)
    spec = env.open(make_spec(rung=8))
    assert env.store.load(spec).status == "running"
    for n in range(1, 4):
        reply = await env.core.chat(spec.run_key, make_request(root_messages(task=f"T{n}")))
        stored = env.store.load(spec)
        assert stored.calls == n and stored.completion_total == env.core.budget_view(spec.run_key)["spent"]
        assert set(reply.headers) >= {"X-Dynorch-Cap-Tokens", "X-Dynorch-Spent-Tokens", "X-Dynorch-Remaining-Tokens",
                                      "X-Dynorch-Call-No", "X-Dynorch-Forced-Close", "X-Dynorch-Grace-Tokens",
                                      "X-Dynorch-Tool-Parse", "X-Dynorch-Conversation", "X-Dynorch-Proxy-Version"}
        assert reply.headers["X-Dynorch-Call-No"] == str(n) and reply.headers["X-Dynorch-Proxy-Version"] == PROXY_VERSION
        assert env.journal.path(spec, n).exists()


# ---- E0 ceiling and cost guards (B6) -----------------------------------------------------------------------------------

async def test_e0_root_ceiling_closes_inside_c_without_grace(make_env):
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=5000)
    env = make_env(script)
    spec = env.open(make_spec(rung=4, arm="H1c", root_call_ceiling=1024))
    reply = await env.core.chat(spec.run_key, make_request(root_messages()))
    think, cont = env.think_calls(phase="think")[0], env.think_calls(phase="answer")[0]
    assert think.body["max_tokens"] == 1024 and cont.body["max_tokens"] == 1024
    assert reply.call.forced_close_kind == "ceiling_close" and reply.call.grace_tokens == 0
    view = env.core.budget_view(spec.run_key)
    assert view["spent"] == reply.call.completion_tokens == 1024 + reply.usage.completion_tokens - 1024
    assert view["grace_tokens"] == 0 and not env.core.record_of(spec.run_key).exhausted
    # the sub-agent gets the remaining budget, uncapped by the ceiling
    sub = await env.core.chat(spec.run_key, solver_request())
    assert sub.call.granted_max_tokens == 4096 - view["spent"]


async def test_cost_guards_trigger_the_terminal_path_for_root_and_402_otherwise(tmp_path):
    cfg = load_cfg()
    cfg["dynorch"]["max_calls_per_run"] = 2
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=50)
    env = Env(tmp_path / "guard", cfg, script)
    spec = env.open(make_spec(rung=8))
    await env.core.chat(spec.run_key, make_request(root_messages()))
    await env.core.chat(spec.run_key, solver_request())
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, solver_request("second"))
    assert exc.value.code == "call_limit" and exc.value.status == 402
    reply = await env.core.chat(spec.run_key, make_request(root_messages(task="T2")))
    assert reply.call.forced_close_kind == "terminal"
    record = env.core.record_of(spec.run_key)
    assert record.guard_triggered and record.exhausted and record.refused_calls == 1
    await env.aclose()
    cfg2 = load_cfg()
    cfg2["dynorch"]["max_prompt_tokens_per_run"] = 100
    env2 = Env(tmp_path / "guard2", cfg2, script)
    spec2 = env2.open(make_spec(rung=8))
    await env2.core.chat(spec2.run_key, make_request(root_messages()))
    with pytest.raises(ProxyError) as exc:
        await env2.core.chat(spec2.run_key, solver_request())
    assert exc.value.code == "input_limit"
    await env2.aclose()


# ---- the frozen config ---------------------------------------------------------------------------------------------------

def test_frozen_settings_load_from_the_config():
    settings = DynorchSettings.from_config(load_cfg())
    assert settings.rungs == (1, 2, 4, 8, 16, 25) and settings.a_tokens == 1024 and settings.grace_tokens == 1024
    assert settings.terminal_threshold_tokens == 64 and settings.max_calls_per_run == 24
    assert settings.max_prompt_tokens_per_run == 600_000 and settings.finish_reasons_returned == ("stop", "tool_calls")
    assert settings.wall_timeout_seconds(25) == 3600 and settings.cap_tokens(16) == 16384
    assert settings.arms["H1"].root_call_ceiling(cap_tokens=16384, a_tokens=1024) is None
    assert settings.arms["H1c"].root_call_ceiling(cap_tokens=16384, a_tokens=1024) == 4096
    assert settings.arms["H3c"].root_call_ceiling(cap_tokens=4096, a_tokens=1024) == 1024
    assert settings.arms["H1n"].spawning_disabled and settings.arms["H2"].fallback_for == "H3"
    assert settings.credit_ceiling_usd == Decimal("95") and settings.reservation_multiplier == Decimal("1.1")
    assert settings.provider_in_flight == {"parasail": 40, "mancer": 48} and settings.provider_rps == {"parasail": 8, "mancer": 12}
    assert "{C}" not in settings.budget_sentence_for(4096) and "4096 completion tokens" in settings.budget_sentence_for(4096)
    assert settings.proxy_version == PROXY_VERSION
    with pytest.raises(ConfigError):
        settings.cap_tokens(3)


@pytest.mark.parametrize("mutate,message", [
    (lambda c: c["dynorch"].__setitem__("finish_reasons_returned", ["stop", "length"]), "finish_reasons"),
    (lambda c: c["dynorch"].__setitem__("grace_root_only", False), "root-only"),
    (lambda c: c["dynorch"].__setitem__("max_grace_events_per_run", 2), "one grace"),
    (lambda c: c["dynorch"].__setitem__("per_run_serialization", False), "in flight"),
    (lambda c: c["dynorch"].__setitem__("harness_max_tokens_ignored", False), "max_tokens"),
    (lambda c: c["dynorch"].__setitem__("rungs", [1, 2, 4, 8, 16]), "rungs"),
    (lambda c: c["dynorch"]["arms"]["H1"].__setitem__("root_call_ceiling", "max(A, C/4)"), "no per-call ceiling"),
    (lambda c: c["dynorch"].__setitem__("terminal_threshold_tokens", 2048), "terminal_threshold"),
    (lambda c: c["budget"].__setitem__("reservation_multiplier", 0.9), "reservation_multiplier"),
    (lambda c: c["dynorch"].pop("budget_sentence"), "budget_sentence"),
    (lambda c: c["dynorch"].__setitem__("cell_rung_order", [16, 4]), "permutation"),
    (lambda c: c.pop("dynorch"), "dynorch"),
    (lambda c: c["dynorch"]["extension_arms"]["H1c"].__setitem__("root_call_ceiling", "C/2"), "root_call_ceiling"),
])
def test_config_outside_the_frozen_table_is_refused(mutate, message):
    cfg = copy.deepcopy(load_cfg())
    mutate(cfg)
    with pytest.raises(ConfigError, match=message):
        DynorchSettings.from_config(cfg)


# ---- cancellation never releases a sent grant --------------------------------------------------------------------------

async def test_a_cancelled_call_charges_its_whole_grant(make_env):
    """A client disconnect that reaches the core mid-upstream (the wire layer should shield it) is treated as
    ambiguous: the grant is charged whole and the lock is released for the next call."""
    import asyncio

    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=100, delay_seconds=5.0)
    env = make_env(script)
    spec = env.open(make_spec(rung=2))
    task = asyncio.create_task(env.core.chat(spec.run_key, make_request(root_messages())))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    view = env.core.budget_view(spec.run_key)
    assert view["spent"] == 2048 and view["remaining"] == 0 and view["calls"] == 1
    assert not env.core._runs[spec.run_key].budget.locked
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, solver_request())
    assert exc.value.code == "budget_exhausted"


# ---- open_run and the ledger's own errors (review majors 4/15) ------------------------------------------------

async def test_open_run_refuses_a_cell_that_already_has_a_terminal_record_or_journaled_calls(make_env):
    """A stage re-executed as attempt 1 / generation 1 used to overwrite the paid record in place and then
    conflict in the write-once journal AFTER the next call had been paid. The core refuses both shapes."""
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=200)
    env = make_env(script)
    spec = env.open(make_spec(rung=4))
    await env.core.chat(spec.run_key, make_request(root_messages()))
    env.core.close_run(spec.run_key, status="done")
    fresh = ProxyCore(env.settings, env.upstream, env.journal, env.store)
    with pytest.raises(ValueError, match="terminal record"):
        fresh.open_run(spec, root_system_text=ROOT_SYSTEM, subagent_system_text=SUBAGENT_SYSTEM)
    env.store.path(spec).unlink()                      # record gone, the paid call journal still there
    with pytest.raises(ValueError, match="journaled call"):
        fresh.open_run(spec, root_system_text=ROOT_SYSTEM, subagent_system_text=SUBAGENT_SYSTEM)


async def test_a_ledger_attempt_id_collision_is_a_500_and_charges_nothing(make_env):
    """``LedgerError`` (a duplicate attempt id) is decided before the body leaves the process: it must be answered
    loudly and release the token grant, not be swallowed by the generic BaseException branch that charges the
    whole grant for a call that was never sent (review major 15)."""
    script = FakeScript()
    script.default = FakeReply(content=FINAL, think_tokens=200)
    env = make_env(script)
    spec = env.open(make_spec(rung=4))
    original = env.ledger.reserve

    async def collide(*, attempt_id: str, **kw):
        from agent_allocation.runtime.ledger import LedgerError
        raise LedgerError(f"attempt {attempt_id} is already reserved")

    env.ledger.reserve = collide
    with pytest.raises(ProxyError) as exc:
        await env.core.chat(spec.run_key, make_request(root_messages()))
    assert exc.value.code == "internal_error" and exc.value.status == 500
    view = env.core.budget_view(spec.run_key)
    assert view["spent"] == 0 and view["calls"] == 0 and env.think_calls() == []
    env.ledger.reserve = original
    reply = await env.core.chat(spec.run_key, make_request(root_messages()))
    assert reply.call.granted_max_tokens == spec.cap_tokens        # the grant was released, not charged
