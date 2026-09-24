"""WP-E Gate 0 smoke, OpenClaude 0.13.0 (H1) through the REAL proxy stack over the fake upstream ($0; plan §9
Gate 0 "Real-CLI smoke ... OpenClaude"; contract §10.1-10.7, §10.18).

The real CLI is launched by the real ``OpenclaudeAdapter`` from the real ``Runner`` against the real
``ProxyCore`` (``tests/integration/dynorch_smoke_support.py``); only the upstream is scripted. Cases and the Gate 0
items each pins (``record_case`` writes them to ``$DYNORCH_GATE0_SMOKE_OUT`` for ``dynorch-gates --gate 0``):

* ``delegation_rung4`` -- one dev item at rung 4, three scripted ``Agent`` calls in a row: the first request's tool
  list is exactly ``{Agent}``; ``messages[0] == OPENCLAUDE_PREFIX + system_text.rstrip()`` (§10.2); every solver
  conversation carries no tools and is classified ``solver`` (§10.3, §10.10); ``other`` calls == 0; the 3-turn
  tool loop re-injects the cached reasoning (``reinjection_hit_rate >= 0.95``, §4 key 1: OpenClaude never re-sends
  ``reasoning_content``, contract §10.4); no ``budget_usd`` text anywhere; the delivered answer is the proxy's and
  the CLI result cross-checks; no key material; nothing under ``$HOME``;
* ``forced_close_text`` -- rung 1: the first root call thinks past the cap, the proxy closes it with the grace
  (``mid_call``, B3) rendered WITHOUT tools; the continuation writes spawn XML, which reaches OpenClaude as plain
  text with finish ``stop`` (contract §10.28): nothing is spawned, the CLI ends on that text (exit 0), it is the
  delivered answer, and a probe sent after the exit is refused with HTTP 402 and no upstream call;
* ``subagent_402_then_terminal`` -- rung 1, no grace involved: the root stops naturally with an ``Agent`` call that
  leaves ``remaining < 64``; the solver's call gets HTTP 402 (not retried), the one further request is the root's
  TERMINAL call (B4) and the run ends on its answer within 10 s of the 402; no rerun (frozen rerun rule);
* ``interruption_retry_latched`` -- contract §10.38: root call 1 gets the provider's "Stream interrupted" error
  envelope (an in-stream error after the headers); OpenClaude's own 5xx retry (``CLAUDE_CODE_MAX_RETRIES=1``) is
  refused by the latch with no upstream call and no TERMINAL; the attempt is ``transport_failed`` /
  ``upstream_latched`` and the in-job rerun (attempt 2, full budget) is the scored record;
* ``rung25`` -- one rung-25 run ends with ``subtype == success`` and a graded record;
* ``delayed_35s`` -- a 35 s upstream delay (two 15 s keepalive comments) produces exactly one upstream call and
  ``subtype == success``; ``delayed_330s`` / ``delayed_660s`` are the plan's stalls (``-m slow``).

Skipped when ``openclaude`` is not on PATH or the sealed base run root is unreadable.
"""
from __future__ import annotations

import json
import shutil
import socket
import time
from pathlib import Path
from typing import Any

import pytest

from agent_allocation.api.fake import FakeReply, Fault, fake_tool_call_text
from agent_allocation.dynorch.harness.openclaude import OpenclaudeAdapter
from tests.integration import dynorch_smoke_support as S
from tests.integration.dynorch_smoke_support import RunScript, SmokeStack

pytestmark = pytest.mark.skipif(shutil.which("openclaude") is None, reason="openclaude is not installed here")
_REAL_CONNECT = socket.socket.connect
ARM = "H1"


@pytest.fixture(autouse=True)
def _loopback_only(monkeypatch):
    """The integration conftest refuses every AF_INET connect; this stand IS a loopback server (the CLI connects
    from its own process), so allow 127.0.0.1 in-process and keep refusing anything else."""
    def guarded(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError(f"the smoke must not leave loopback (tried {address!r})")
        return _REAL_CONNECT(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture(scope="module")
def base():
    loaded = S.load_base()
    if loaded is None:
        pytest.skip("base run root not readable here")
    return loaded


def _adapters() -> dict[str, Any]:
    return {"openclaude": OpenclaudeAdapter(model=S.CFG["primary_model"])}


def spawn_text(task: str, *, prefix: str = "") -> str:
    """A scripted ``Agent`` call (the tool's required arguments; ``subagent_type`` must be ``solver``, §10.7)."""
    return fake_tool_call_text("Agent", {"description": "delegate", "prompt": task, "subagent_type": "solver"},
                               prefix=prefix)


FINAL = FakeReply(content="<rationale>Direct.</rationale>\n<final>42</final>", think_tokens=120)
CHILD = FakeReply(content="<rationale>Child.</rationale>\n<final>child says 42</final>", think_tokens=90)


# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.timeout(600)
async def test_openclaude_scripted_delegation_rung4(tmp_path: Path, base):
    marker = S.home_marker()
    item = S.dev_item(base, "aime")
    async with SmokeStack(tmp_path, adapters=_adapters()) as stack:
        spec = stack.specs(base, arm=ARM, rung=4, items=[item])[0]
        stack.scripts[spec.run_key] = RunScript(
            root=[FakeReply(content=spawn_text("part one"), think_tokens=150),
                  FakeReply(content=spawn_text("part two"), think_tokens=150),
                  FakeReply(content=spawn_text("part three"), think_tokens=150), FINAL],
            solver=[CHILD])
        records = await stack.run(base, [spec])
    record = records[spec.run_key]
    docs = stack.journaled(spec)
    requests = [d["harness_request"] for d in docs]
    classes = [d["call"]["conversation_class"] for d in docs]
    first = requests[0]
    solver_requests = [r for r, c in zip(requests, classes) if c == "solver"]
    result = stack.result_line(spec) or {}
    rate = S.reinjection_hit_rate(record)
    common = S.common_checks(stack, spec, record, marker)
    checks: dict[str, bool] = {
        "run_done_exit_0": record.status == "done" and record.harness_exit_code == 0 and record.rerun_reason is None,
        "first_request_tools_exactly_agent": [t["name"] for t in first["tools"]] == ["Agent"] and first["wire"] == "chat",
        "messages0_is_prefix_plus_system": first["messages"][0]["role"] == "system"
                                            and first["messages"][0]["content"] == S.OPENCLAUDE_PREFIX + S.SYSTEM_TEXT.rstrip("\n"),
        "budget_sentence_delivered": any("Input tokens are not counted." in m["content"] for m in first["messages"]
                                         if m["role"] == "user"),
        "three_spawns_three_solver_conversations": record.spawns == 3 and len(solver_requests) == 3
                                                    and record.conversations == 4 and record.calls == 7,
        "solver_conversations_have_no_tools": all(r["tools"] == [] for r in solver_requests)
                                              and all(r["messages"][0]["content"].startswith(S.OPENCLAUDE_PREFIX + S.SUBAGENT_TEXT.rstrip("\n"))
                                                      for r in solver_requests),
        "call_classes_root_solver_alternate": classes == ["root", "solver", "root", "solver", "root", "solver", "root"],
        "other_calls_zero": record.calls_by_class.get("other", 0) == 0,
        "reinjection_hit_rate_ge_0_95": rate is not None and rate >= 0.95 and record.reinjection_misses == 0,
        # the proxy-minted ids (call_<call_no>_0 on root calls 1, 3, 5) come back verbatim on the assistant turn
        # and its tool result -- the reasoning cache's key 1 (§4)
        "tool_call_ids_echoed_exactly": (
            [tc["id"] for m in requests[-1]["messages"] if m["role"] == "assistant" for tc in m["tool_calls"]]
            == ["call_1_0", "call_3_0", "call_5_0"]
            and [m["tool_call_id"] for m in requests[-1]["messages"] if m["role"] == "tool"]
            == ["call_1_0", "call_3_0", "call_5_0"]),
        "spawns_all_solver_type": all(d["tool_parse"].get("non_solver_spawns", []) == [] for d in docs),
        "delivered_from_proxy_and_cross_checked": record.final_source == "proxy_root" and record.parse_status == "valid"
                                                   and record.cli_result_mismatch is False,
        "cli_result_success": result.get("is_error") is False and result.get("subtype") == "success"
                              and result.get("num_turns") == 4,
        "finish_reasons_in_set": all(c.finish_reason_returned in ("stop", "tool_calls") for c in record.calls_detail),
        "grants_are_cap_minus_spent": _grants_exact(record, spec),
        **common[0],
    }
    failed = S.record_case("openclaude/delegation_rung4", checks, detail={**common[1],
        "calls": record.calls, "spawns": record.spawns, "reinjection_hits": record.reinjection_hits,
        "reinjection_misses": record.reinjection_misses, "harness_version": record.spec.harness_version,
        "classes": classes})
    assert not failed, (failed, result, [c.__dict__ for c in record.calls_detail][:3])


def _grants_exact(record, spec) -> bool:
    spent = 0
    for call in record.calls_detail:
        if call.duplicate_served or call.forced_close_kind == "terminal":
            continue
        if call.granted_max_tokens != spec.cap_tokens - spent:
            return False
        spent += call.completion_tokens
    return True


#: What a model about to delegate writes when its one grace comes: the tool-call XML of a spawn. Under plan B3 /
#: contract §10.28 the continuation carries no tool schemas and is returned as TEXT with finish "stop".
GRACE_SPAWN_XML = spawn_text("solve it all", prefix="I must delegate now.")
TERMINAL_FINAL = FakeReply(content="<rationale>Out of budget; answering.</rationale>\n<final>42</final>")


@pytest.mark.timeout(600)
async def test_openclaude_forced_close_is_text_and_ends_the_run(tmp_path: Path, base):
    """Plan B3 / contract §10.28 through the real CLI: the THINK hits the cap at rung 1, the grace continuation
    (rendered WITHOUT tools) writes tool-call XML, and OpenClaude receives it as plain text with finish "stop": it
    spawns nothing, ends on that text (exit 0) and that text is the delivered answer. Any further call is refused:
    a probe sent after the harness exited gets HTTP 402 with no upstream call."""
    marker = S.home_marker()
    item = S.dev_item(base, "gpqa_diamond")
    async with SmokeStack(tmp_path, adapters=_adapters(), sigterm_after_exhaustion=45.0, probe_after_exit=True) as stack:
        spec = stack.specs(base, arm=ARM, rung=1, items=[item])[0]
        cap = spec.cap_tokens
        stack.scripts[spec.run_key] = RunScript(root=[FakeReply(think_tokens=cap + 500)], answer=FakeReply(
            content=GRACE_SPAWN_XML), solver=[CHILD])
        records = await stack.run(base, [spec])
    record = records[spec.run_key]
    docs = stack.journaled(spec)
    result = stack.result_line(spec) or {}
    script = stack.scripts[spec.run_key]
    probe = stack.probes.get(spec.run_key)
    call = record.calls_detail[0] if record.calls_detail else None
    reply = docs[0]["reply"] if docs else {}
    common = S.common_checks(stack, spec, record, marker)
    checks: dict[str, bool] = {
        "one_served_call_forced_close_mid_call": record.calls == 1 and call is not None
                                                 and call.forced_close_kind == "mid_call"
                                                 and record.forced_close_kind == "mid_call",
        "grace_used_and_bounded": 0 < record.grace_tokens <= S.CFG["dynorch"]["grace_tokens"]
                                  and record.completion_total == cap and record.exhausted is True,
        # B3: the continuation is rendered with no tool schemas; the THINK kept the harness's tools
        "continuation_carries_no_tools": [c["prompt_has_tools"] for c in script.phase_calls("think")] == [True]
                                         and [c["prompt_has_tools"] for c in script.phase_calls("answer")] == [False],
        # contract §10.28: the grace is text with finish "stop", never a tool call
        "grace_returned_as_text_finish_stop": call is not None and call.finish_reason_returned == "stop"
                                              and call.tool_calls == 0 and reply.get("tool_calls") == []
                                              and "<tool_call>" in str(reply.get("content")),
        "nothing_spawned": record.spawns == 0 and record.calls_by_class.get("solver", 0) == 0
                           and script.think_calls("solver") == 0,
        "cli_ended_on_that_text": result.get("is_error") is False and result.get("subtype") == "success"
                                  and record.harness_exit_code == 0 and result.get("num_turns") == 1
                                  and "<tool_call>" in str(result.get("result", "")),
        "delivered_answer_is_the_grace_text": record.final_source == "proxy_root"
                                              and record.final_message is not None
                                              and "<tool_call>" in record.final_message
                                              and record.cli_result_mismatch is False,
        "further_call_refused_402_no_upstream": probe is not None and probe.status == 402
                                                and probe.code == "budget_exhausted"
                                                and probe.upstream_calls_after == probe.upstream_calls_before,
        "no_upstream_call_after_the_grace": script.think_calls() == 1 and len(script.calls) == 2,
        "no_rerun": record.status == "done" and record.rerun_reason is None and record.spec.attempt == 1,
        "not_killed_by_watchdog": record.harness_reported.get("sigterm_after_exhaustion") is False
                                  and record.harness_reported.get("timed_out") is False,
        "single_request_journaled": len(docs) == 1,
        "finish_reasons_in_set": all(c.finish_reason_returned in ("stop", "tool_calls") for c in record.calls_detail),
        **common[0],
    }
    failed = S.record_case("openclaude/forced_close_text", checks, detail={**common[1],
        "probe": None if probe is None else probe.__dict__, "result": str(result.get("result", ""))[:300],
        "upstream": script.calls, "events": [e.__dict__ for e in stack.reply_log.timed.get(spec.run_key, [])],
        "parse_status": record.parse_status})
    assert not failed, (failed, result, record.harness_reported)


@pytest.mark.timeout(600)
async def test_openclaude_subagent_402_then_terminal_ends_the_run(tmp_path: Path, base):
    """The 402 case without the grace (plan B4, Gate 0 "402 ends the run within 10 s with <= 1 further request"):
    the root's THINK stops naturally with an ``Agent`` call that leaves ``remaining < 64``; the solver's call is
    refused with HTTP 402 (a sub-agent never gets the TERMINAL), OpenClaude does not retry it, the root's one
    further request is its TERMINAL call (thinking off, no tools, ``max_tokens = G``) and the run ends on that
    answer. A probe after the exit gets 402 (the grace is spent)."""
    marker = S.home_marker()
    item = S.dev_item(base, "gpqa_diamond")
    async with SmokeStack(tmp_path, adapters=_adapters(), sigterm_after_exhaustion=45.0, probe_after_exit=True) as stack:
        spec = stack.specs(base, arm=ARM, rung=1, items=[item])[0]
        cap = spec.cap_tokens
        threshold = int(S.CFG["dynorch"]["terminal_threshold_tokens"])
        spent = cap - (threshold - 24)                      # remaining = threshold - 24 < threshold after call 1
        spawn = spawn_text("solve it all")
        # the fake's completion = THINK tokens + ceil(len(visible) / 4); the call closes on its own below the grant
        stack.scripts[spec.run_key] = RunScript(
            root=[FakeReply(content=spawn, think_tokens=spent - -(-len(spawn) // 4))], solver=[CHILD],
            answer=TERMINAL_FINAL)
        records = await stack.run(base, [spec])
        ended = time.monotonic()
    record = records[spec.run_key]
    docs = stack.journaled(spec)
    result = stack.result_line(spec) or {}
    script = stack.scripts[spec.run_key]
    probe = stack.probes.get(spec.run_key)
    horizon = probe.sent_at if probe is not None else float("inf")
    first_402 = stack.reply_log.first_error(spec.run_key, 402)
    after = [e for e in (stack.reply_log.events_after(spec.run_key, first_402.at) if first_402 else []) if e.at < horizon]
    classes = [d["call"]["conversation_class"] for d in docs]
    kinds = [c.forced_close_kind for c in record.calls_detail]
    common = S.common_checks(stack, spec, record, marker)
    checks: dict[str, bool] = {
        "root_think_stopped_naturally_with_a_spawn": bool(record.calls_detail)
                                                     and record.calls_detail[0].forced_close_kind == "none"
                                                     and record.calls_detail[0].finish_reason_returned == "tool_calls"
                                                     and record.calls_detail[0].completion_tokens == spent
                                                     and record.spawns == 1 and cap - spent < threshold,
        "solver_call_refused_402": first_402 is not None and record.refused_calls >= 1
                                   and script.think_calls("solver") == 0 and record.calls_by_class.get("solver", 0) == 0,
        "at_most_one_request_after_first_402": len(after) <= 1,
        "the_further_request_is_the_root_terminal": len(after) == 1 and after[0].kind == "reply"
                                                    and after[0].conversation_class == "root",
        "run_ended_within_10s_of_402": first_402 is not None and (ended - first_402.at) < 10.0,
        "terminal_served_once_no_tools": classes == ["root", "root"] and kinds == ["none", "terminal"]
                                         and record.forced_close_kind == "terminal"
                                         and 0 < record.grace_tokens <= S.CFG["dynorch"]["grace_tokens"]
                                         and [c["prompt_has_tools"] for c in script.phase_calls("terminal")] == [False]
                                         and [c["max_tokens"] for c in script.phase_calls("terminal")]
                                         == [S.CFG["dynorch"]["grace_tokens"]],
        "no_think_call_after_the_spawn": script.think_calls() == 1 and len(script.calls) == 2,
        "cli_success_on_the_terminal_answer": result.get("is_error") is False and result.get("subtype") == "success"
                                              and record.harness_exit_code == 0,
        "delivered_from_proxy_and_cross_checked": record.final_source == "proxy_root" and record.parse_status == "valid"
                                                   and record.cli_result_mismatch is False,
        "further_call_refused_402_no_upstream": probe is not None and probe.status == 402
                                                and probe.code == "budget_exhausted"
                                                and probe.upstream_calls_after == probe.upstream_calls_before,
        "no_rerun_after_402": record.status == "done" and record.rerun_reason is None and record.spec.attempt == 1,
        "not_killed_by_watchdog": record.harness_reported.get("sigterm_after_exhaustion") is False
                                  and record.harness_reported.get("timed_out") is False,
        "finish_reasons_in_set": all(c.finish_reason_returned in ("stop", "tool_calls") for c in record.calls_detail),
        **common[0],
    }
    failed = S.record_case("openclaude/subagent_402_then_terminal", checks, detail={**common[1],
        "seconds_from_402_to_end": None if first_402 is None else round(ended - first_402.at, 2),
        "events": [e.__dict__ for e in stack.reply_log.timed.get(spec.run_key, [])],
        "probe": None if probe is None else probe.__dict__, "upstream": script.calls, "classes": classes})
    assert not failed, (failed, result, record.harness_reported)


@pytest.mark.timeout(600)
async def test_openclaude_interruption_retry_is_refused_by_the_latch(tmp_path: Path, base):
    """Contract §10.38 through the real CLI (the live defect of 2026-09-24): call 1 is interrupted upstream after
    the response headers went out (an in-stream 502 error frame, the full grant settled). OpenClaude retries it;
    the retry is refused 502 by the latch before any grant -- no second upstream call, no TERMINAL call -- and the
    CLI ends in error. The run is ``transport_failed`` / ``upstream_latched`` and the runner's infrastructure rerun
    (attempt 2, a new run_key, the full budget, no model rerun consumed) is what is scored."""
    marker = S.home_marker()
    item = S.dev_item(base, "aime")
    async with SmokeStack(tmp_path, adapters=_adapters()) as stack:
        spec = stack.specs(base, arm=ARM, rung=4, items=[item])[0]
        grace = float(S.CFG["dynorch"]["proxy"].get("refusal_grace_seconds", 2.0))
        stack.scripts[spec.run_key] = RunScript(root=[FakeReply(fault=Fault.STREAM_INTERRUPTED,
                                                                delay_seconds=grace + 1.5), FINAL])
        records = await stack.run(base, [spec])
        first = stack.store.scan(include_superseded=True)[spec.run_key]
    final = next(iter(records.values()))
    script = stack.scripts[spec.run_key]
    events = stack.reply_log.timed.get(spec.run_key, [])
    errors = [e for e in events if e.kind == "error"]
    first_result = None
    stdout = Path(first.harness_reported.get("work_dir", "")) / "stdout"
    if stdout.is_file():
        lines = [l for l in stdout.read_text("utf-8", "replace").splitlines() if l.strip().startswith("{")]
        for raw in reversed(lines):
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if obj.get("type") == "result":
                first_result = obj
                break
    common = S.common_checks(stack, final.spec, final, marker)
    checks: dict[str, bool] = {
        "call1_interrupted_full_grant_settled": first.calls == 1 and bool(first.calls_detail)
                                                and first.calls_detail[0].transport_status == "error_envelope"
                                                and first.calls_detail[0].returned_status == 502
                                                and first.completion_total == spec.cap_tokens,
        "cli_retried_the_502": first.refused_calls >= 1 and len(errors) >= 2
                               and all(e.status == 502 and e.code == "upstream_failed" for e in errors),
        "retry_refused_no_upstream_call": script.think_calls() == 1 and script.phase_calls("terminal") == []
                                          and len(script.calls) == 1 and not any(e.kind == "reply" for e in events),
        "no_terminal_no_grace": [c.forced_close_kind for c in first.calls_detail] == ["none"]
                                and first.grace_tokens == 0,
        "cli_ended_in_error": first_result is None or first_result.get("is_error") is True,
        "attempt1_transport_failed_latched": first.status == "transport_failed"
                                             and first.rerun_reason == "upstream_latched"
                                             and first.parse_status is None and first.correct is None,
        "rerun_full_budget_scored": final.spec.attempt == 2 and final.spec.run_key != spec.run_key
                                    and final.status == "done" and final.parse_status == "valid"
                                    and final.calls_detail[0].granted_max_tokens == spec.cap_tokens
                                    and final.harness_exit_code == 0,
        "no_model_rerun_consumed": stack.runner._reruns_used(final.spec) == 0,
        **common[0],
    }
    failed = S.record_case("openclaude/interruption_retry_latched", checks, detail={**common[1],
        "events": [e.__dict__ for e in events], "upstream": script.calls, "refused_calls": first.refused_calls,
        "result": None if first_result is None else str(first_result.get("result", ""))[:300]})
    assert not failed, (failed, first_result, [e.__dict__ for e in events])


@pytest.mark.timeout(600)
async def test_openclaude_rung25_ends_with_success(tmp_path: Path, base):
    marker = S.home_marker()
    item = S.dev_item(base, "hmmt")
    async with SmokeStack(tmp_path, adapters=_adapters()) as stack:
        spec = stack.specs(base, arm=ARM, rung=25, items=[item])[0]
        stack.scripts[spec.run_key] = RunScript(root=[FakeReply(think_tokens=3000)])
        records = await stack.run(base, [spec])
    record = records[spec.run_key]
    result = stack.result_line(spec) or {}
    common = S.common_checks(stack, spec, record, marker)
    checks = {
        "rung25_cap": spec.cap_tokens == 25 * 1024 and record.calls_detail[0].granted_max_tokens == 25 * 1024,
        "subtype_success": result.get("subtype") == "success" and result.get("is_error") is False,
        "run_done_graded": record.status == "done" and record.parse_status == "valid" and record.correct in (True, False)
                           and record.harness_exit_code == 0,
        "one_call_no_grace": record.calls == 1 and record.grace_tokens == 0 and record.forced_close_kind == "none",
        "wall_timeout_is_3600s": record.harness_reported.get("wall_timeout_seconds") == 3600,
        **common[0],
    }
    failed = S.record_case("openclaude/rung25", checks, detail={**common[1], "wall_seconds": record.wall_seconds})
    assert not failed, (failed, result)


async def _delay_case(tmp_path: Path, base, delay: float, name: str) -> None:
    marker = S.home_marker()
    item = S.dev_item(base, "aime")
    async with SmokeStack(tmp_path, adapters=_adapters()) as stack:
        spec = stack.specs(base, arm=ARM, rung=4, items=[item])[0]
        stack.scripts[spec.run_key] = RunScript(root=[FakeReply(delay_seconds=delay, think_tokens=200)])
        records = await stack.run(base, [spec])
    record = records[spec.run_key]
    result = stack.result_line(spec) or {}
    script = stack.scripts[spec.run_key]
    common = S.common_checks(stack, spec, record, marker)
    checks = {
        "exactly_one_upstream_call": script.think_calls() == 1 and len(script.calls) == 1 and record.calls == 1,
        "subtype_success": result.get("subtype") == "success" and result.get("is_error") is False,
        "waited_through_the_delay": record.wall_seconds >= delay and record.calls_detail[0].latency_seconds >= delay,
        "run_done_valid": record.status == "done" and record.parse_status == "valid" and record.harness_exit_code == 0
                          and record.cli_result_mismatch is False,
        "no_duplicate_served": record.duplicates_served == 0,
        **common[0],
    }
    failed = S.record_case(f"openclaude/{name}", checks, detail={**common[1], "delay": delay, "wall_seconds": record.wall_seconds})
    assert not failed, (failed, result)


@pytest.mark.timeout(600)
async def test_openclaude_delayed_reply_35s(tmp_path: Path, base):
    await _delay_case(tmp_path, base, 35.0, "delayed_35s")


@pytest.mark.slow
@pytest.mark.timeout(1500)
async def test_openclaude_delayed_reply_330s(tmp_path: Path, base):
    await _delay_case(tmp_path, base, 330.0, "delayed_330s")


@pytest.mark.slow
@pytest.mark.timeout(1500)
async def test_openclaude_delayed_reply_660s(tmp_path: Path, base):
    await _delay_case(tmp_path, base, 660.0, "delayed_660s")
