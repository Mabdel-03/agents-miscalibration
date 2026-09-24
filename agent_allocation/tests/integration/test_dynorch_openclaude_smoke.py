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
* ``forced_close_then_402`` -- rung 1: the first root call thinks past the cap, the proxy closes it with the grace
  (``mid_call``, B3) and the continuation is a spawn; the solver's call and the root's next call are refused with
  HTTP 402 (B4: after the grace every call is refused): the run ends within 10 s of the first 402 with at most one
  further request, ``is_error`` true, exit 1, no rerun (frozen rerun rule);
* ``rung25`` -- one rung-25 run ends with ``subtype == success`` and a graded record;
* ``delayed_35s`` -- a 35 s upstream delay (two 15 s keepalive comments) produces exactly one upstream call and
  ``subtype == success``; ``delayed_330s`` / ``delayed_660s`` are the plan's stalls (``-m slow``).

Skipped when ``openclaude`` is not on PATH or the sealed base run root is unreadable.
"""
from __future__ import annotations

import shutil
import socket
import time
from pathlib import Path
from typing import Any

import pytest

from agent_allocation.api.fake import FakeReply, fake_tool_call_text
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


@pytest.mark.timeout(600)
async def test_openclaude_forced_close_then_402_ends_the_run(tmp_path: Path, base):
    marker = S.home_marker()
    item = S.dev_item(base, "gpqa_diamond")
    async with SmokeStack(tmp_path, adapters=_adapters(), sigterm_after_exhaustion=45.0) as stack:
        spec = stack.specs(base, arm=ARM, rung=1, items=[item])[0]
        cap = spec.cap_tokens
        # root call 1 thinks past the cap -> length inside <think> -> the proxy's ANSWER continuation (grace)
        # returns a spawn; the solver's call and the root's next call meet a closed budget -> HTTP 402
        stack.scripts[spec.run_key] = RunScript(
            root=[FakeReply(think_tokens=cap + 500)],
            answer=FakeReply(content=spawn_text("solve it all", prefix="I must delegate now.")),
            solver=[CHILD])
        started = time.monotonic()
        records = await stack.run(base, [spec])
        ended = time.monotonic()
    record = records[spec.run_key]
    docs = stack.journaled(spec)
    result = stack.result_line(spec) or {}
    first_402 = stack.reply_log.first_error(spec.run_key, 402)
    after = stack.reply_log.events_after(spec.run_key, first_402.at) if first_402 else []
    common = S.common_checks(stack, spec, record, marker)
    checks: dict[str, bool] = {
        "one_served_call_forced_close_mid_call": record.calls == 1 and record.calls_detail[0].forced_close_kind == "mid_call"
                                                 and record.forced_close_kind == "mid_call",
        "grace_used_and_bounded": 0 < record.grace_tokens <= S.CFG["dynorch"]["grace_tokens"]
                                  and record.completion_total == cap and record.exhausted is True,
        "spawn_emitted_in_the_continuation": record.spawns == 1 and record.calls_detail[0].finish_reason_returned == "tool_calls",
        "402_served": first_402 is not None and record.refused_calls >= 1,
        "at_most_one_request_after_first_402": len(after) <= 1 and all(e.kind == "error" and e.status == 402 for e in after),
        "run_ended_within_10s_of_402": first_402 is not None and (ended - first_402.at) < 10.0,
        "harness_exit_1_is_error": record.harness_exit_code == 1 and result.get("is_error") is True
                                   and "402" in str(result.get("result", "")),
        "no_rerun_after_402": record.status == "done" and record.rerun_reason is None and record.spec.attempt == 1,
        "not_killed_by_watchdog": record.harness_reported.get("sigterm_after_exhaustion") is False
                                  and record.harness_reported.get("timed_out") is False,
        "no_upstream_call_after_the_grace": stack.scripts[spec.run_key].think_calls() == 1
                                            and len(stack.scripts[spec.run_key].calls) == 2,
        "single_request_before_402_journaled": len(docs) == 1,
        **common[0],
    }
    failed = S.record_case("openclaude/forced_close_then_402", checks, detail={**common[1],
        "seconds_from_402_to_end": None if first_402 is None else round(ended - first_402.at, 2),
        "events": [e.__dict__ for e in stack.reply_log.timed.get(spec.run_key, [])],
        "wall_seconds": record.wall_seconds, "started_to_end": round(ended - started, 2)})
    assert not failed, (failed, result, record.harness_reported)


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
