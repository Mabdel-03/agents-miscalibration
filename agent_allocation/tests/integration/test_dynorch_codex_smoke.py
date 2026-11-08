"""WP-E Gate 0 smoke, OpenAI Codex CLI 0.156.1 (H3) through the REAL proxy stack over the fake upstream ($0; plan §9
Gate 0 "Real-CLI smoke ... Codex"; contract §10.3, §10.4, §10.5, §10.18).

The real CLI is launched by the real ``CodexAdapter`` from the real ``Runner`` against the real ``ProxyCore``
(``tests/integration/dynorch_smoke_support.py``); only the upstream is scripted. Cases and the Gate 0 items each pins
(``record_case`` writes them to ``$DYNORCH_GATE0_SMOKE_OUT`` for ``dynorch-gates --gate 0``):

* ``install`` -- ``codex --version`` from the scratch prefix is ``codex-cli 0.156.1``; ``wire_api = "chat"`` is
  refused by ``--strict-config`` before any request (documented: Codex speaks only the Responses API);
* ``delegation_rung4`` -- the first request's tools are exactly ``update_plan`` + the ``multi_agent_v1`` namespace
  (§10.3) and contain none of ``exec_command`` / ``shell`` / ``apply_patch`` / ``web_search`` / ``view_image``;
  ``instructions == system_text.rstrip()`` (§10.2); a scripted ``spawn_agent`` produces a child request carrying
  ``x-openai-subagent`` at the same base URL (classified ``solver``, §10.10) whose tools contain no spawn tool; the
  root's ``wait_agent`` completes the loop; Codex re-sends reasoning items (``reinjection == "explicit"``, §10.4);
  no ``other`` calls; no ``budget_usd`` text; no key material; nothing under ``$HOME``;
* ``exec_command`` -- a scripted ``exec_command`` call (not offered) is answered ``unsupported call: exec_command``
  with no ``command_execution`` item and no process (the leak file the command would create is absent);
* ``missing_completed`` -- a stream cut after ``response.created`` is retried twice (``stream_max_retries = 2``)
  then ``turn.failed``; the proxy serves the identical retries from its cache (B9: one upstream call);
* ``timeout`` -- the adapter's ``timeout --kill-after`` ends the run when its wall elapses and the whole process
  group is gone afterwards; the upstream call it left behind is finished and settled (B2);
* ``forced_close_text`` -- rung 1: the cap binds inside ``<think>``; the grace continuation (no tool schemas) writes
  ``spawn_agent`` XML, which reaches Codex as a plain message with finish ``stop`` (contract §10.28): no function
  call, no child, ``turn.completed`` on that text, which is the delivered answer; a probe after the exit gets 402;
* ``subagent_402_then_terminal`` -- rung 1, no grace involved: the root stops naturally with ``spawn_agent`` leaving
  ``remaining < 64``; the child's request gets HTTP 402, which Codex re-sends at most ``stream_max_retries`` (2) times,
  each refused in-process with no upstream call or cost (contract §10.20); the root's next request is its one
  TERMINAL (B4) and the turn completes on it within 10 s of the first 402;
* ``interruption_retry_latched`` -- contract §10.38: root call 1 gets the provider's "Stream interrupted" error
  envelope after the headers went out; Codex's own retries of the failed turn are refused 502 by the latch with no
  upstream call and no TERMINAL; the attempt is ``transport_failed`` / ``upstream_latched`` and the in-job rerun
  (attempt 2, full budget) is the scored record;
* ``rung25`` -- one rung-25 run ends with ``turn.completed`` and a graded record;
* ``delayed_35s`` -- a 35 s upstream delay (two keepalive comments) produces exactly one upstream call and a
  completed turn; ``delayed_330s`` / ``delayed_660s`` are the plan's stalls (``-m slow``).

Skipped when the Codex binary is absent from the scratch prefix or the sealed base run root is unreadable.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import h11
import pytest

from agent_allocation.api.fake import FakeCall, FakeReply, Fault, fake_tool_call_text
from agent_allocation.dynorch.harness.base import harness_env
from agent_allocation.dynorch.harness.codex import CodexAdapter, config_toml
from agent_allocation.dynorch.httpd import ProxyServer
from tests.integration import dynorch_smoke_support as S
from tests.integration.dynorch_smoke_support import RunScript, SmokeStack

CODEX_BIN = Path(S.CFG["dynorch"]["codex_prefix"]) / "bin" / "codex"
pytestmark = pytest.mark.skipif(not CODEX_BIN.is_file(), reason="codex is not installed in the scratch prefix")
_REAL_CONNECT = socket.socket.connect
ARM = "H3"
NAMESPACE = "multi_agent_v1"
NAMESPACED = tuple(S.CFG["dynorch"]["namespaced_tools"][NAMESPACE])
FORBIDDEN_TOOLS = ("exec_command", "shell", "apply_patch", "web_search", "view_image", "write_stdin",
                   "request_user_input", "get_goal", "create_goal", "update_goal")
STREAM_MAX_RETRIES = 2          # config_toml: stream_max_retries = 2 (plan §6)
FINAL = FakeReply(content="<rationale>Direct.</rationale>\n<final>42</final>", think_tokens=120)
CHILD = FakeReply(content="<rationale>Child.</rationale>\n<final>child says 42</final>", think_tokens=90)


@pytest.fixture(autouse=True)
def _loopback_only(monkeypatch):
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


def _adapter(**over: Any) -> CodexAdapter:
    wall = S.CFG["dynorch"]["wall_timeout_seconds"]
    kw: dict[str, Any] = {"model": S.CFG["primary_model"], "prefix": str(S.CFG["dynorch"]["codex_prefix"]),
                          "wall_base": int(wall["base"]), "wall_per_rung": int(wall["per_rung"])}
    kw.update(over)
    return CodexAdapter(**kw)


def _adapters(**over: Any) -> dict[str, Any]:
    return {"codex": _adapter(**over)}


def spawn_text(message: str, *, prefix: str = "") -> str:
    return fake_tool_call_text("spawn_agent", {"message": message}, prefix=prefix)


def wait_text(call: FakeCall) -> FakeReply:
    """``wait_agent`` on the agent id Codex minted for the spawn (read from the rendered tool response)."""
    return FakeReply(content=fake_tool_call_text("wait_agent", {"targets": [S.agent_id_in_prompt(call)],
                                                                "timeout_ms": 60000}), think_tokens=80)


def _events(stack: SmokeStack, spec) -> list[dict[str, Any]]:
    return stack.stdout_json_lines(spec)


def _event_types(events: list[dict[str, Any]]) -> list[str]:
    return [str(e.get("type")) for e in events]


def _item_types(events: list[dict[str, Any]]) -> list[str]:
    return [str(e["item"].get("type")) for e in events if isinstance(e.get("item"), dict)]


def _tool_names(request: dict[str, Any]) -> list[str]:
    return [t["name"] for t in request["tools"]]


# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_codex_install_and_wire_api_chat_refused(tmp_path: Path):
    adapter = _adapter()
    version = adapter.version()
    work = tmp_path / "work"
    for sub in ("home", "state/codex", "ws", "tmp"):
        (work / sub).mkdir(parents=True)
    run_key = "0123456789abcdef01234567"
    proxy = f"http://127.0.0.1:9/r/{run_key}/v1"
    instructions = work / "SYSTEM.md"
    instructions.write_text(S.SYSTEM_TEXT, "utf-8")
    toml = config_toml(model=S.CFG["primary_model"], proxy_base_url=proxy, instructions_file=instructions)
    assert 'wire_api = "responses"' in toml
    (work / "state" / "codex" / "config.toml").write_text(toml.replace('wire_api = "responses"', 'wire_api = "chat"'), "utf-8")
    env = harness_env(os.environ, work=work, proxy_base_url=proxy, run_key=run_key,
                      extra={"CODEX_HOME": str(work / "state" / "codex")})
    proc = subprocess.run([str(CODEX_BIN), "exec", "--json", "--ephemeral", "--strict-config", "--skip-git-repo-check",
                           "-C", str(work / "ws"), "-s", "read-only", "-"], input=b"TASK\n", capture_output=True,
                          env=env, cwd=str(work / "ws"), timeout=120)
    stderr = proc.stderr.decode("utf-8", "replace")
    checks = {
        "codex_version_0_156_1": version == "codex-cli 0.156.1",
        "wire_api_chat_refused_before_any_request": proc.returncode != 0 and "wire_api" in stderr
                                                    and ("chat" in stderr.lower()),
        "no_event_stream_on_refusal": b'"type":"turn.started"' not in proc.stdout,
    }
    failed = S.record_case("codex/install", checks, detail={"version": version, "returncode": proc.returncode,
                                                            "stderr": stderr[:400]})
    assert not failed, (failed, version, proc.returncode, stderr[:800])


@pytest.mark.timeout(600)
async def test_codex_scripted_delegation_rung4(tmp_path: Path, base):
    marker = S.home_marker()
    item = S.dev_item(base, "aime")
    async with SmokeStack(tmp_path, adapters=_adapters()) as stack:
        spec = stack.specs(base, arm=ARM, rung=4, items=[item])[0]
        stack.scripts[spec.run_key] = RunScript(
            root=[FakeReply(content=spawn_text("Solve the whole problem and report the integer."), think_tokens=150),
                  wait_text, FINAL],
            solver=[CHILD])
        records = await stack.run(base, [spec])
    record = records[spec.run_key]
    docs = stack.journaled(spec)
    requests = [d["harness_request"] for d in docs]
    classes = [d["call"]["conversation_class"] for d in docs]
    first = requests[0]
    children = [r for r, c in zip(requests, classes) if c == "solver"]
    roots = [r for r, c in zip(requests, classes) if c == "root"]
    events = _events(stack, spec)
    collab = [e["item"] for e in events if isinstance(e.get("item"), dict) and e["item"].get("type") == "collab_tool_call"
              and e.get("type") == "item.completed"]
    first_names = _tool_names(first)
    common = S.common_checks(stack, spec, record, marker)
    checks: dict[str, bool] = {
        "run_done_exit_0": record.status == "done" and record.harness_exit_code == 0 and record.rerun_reason is None,
        "wire_is_responses": all(r["wire"] == "responses" for r in requests),
        "first_request_tools_exactly_update_plan_plus_namespace": sorted(first_names) == sorted(("update_plan",) + NAMESPACED)
                                                                   and all(t["namespace"] == NAMESPACE for t in first["tools"]
                                                                           if t["name"] != "update_plan")
                                                                   and next(t["namespace"] for t in first["tools"]
                                                                            if t["name"] == "update_plan") is None,
        "no_execution_or_web_tools": not (set(first_names) & set(FORBIDDEN_TOOLS)),
        "instructions_are_the_system_text": first["messages"][0]["role"] == "system"
                                            and first["messages"][0]["content"].startswith(S.SYSTEM_TEXT.rstrip("\n"))
                                            and "<skills_instructions>" in first["messages"][0]["content"],
        "budget_sentence_delivered": any("Input tokens are not counted." in m["content"] for m in first["messages"]
                                         if m["role"] == "user"),
        "child_request_marked_subagent_same_run": len(children) == 1 and children[0]["subagent"] is True
                                                   and record.calls_by_class.get("solver") == 1
                                                   and record.conversations == 2,
        "child_has_no_spawn_tool": bool(children) and not (set(_tool_names(children[0])) & set(NAMESPACED))
                                   and set(_tool_names(children[0])) <= set(spec.allowed_tools),
        "child_prompt_is_the_spawn_message": bool(children) and any(
            "Solve the whole problem and report the integer." in m["content"] for m in children[0]["messages"]
            if m["role"] == "user"),
        "root_loop_spawn_wait_final": len(roots) == 3 and record.spawns == 1 and record.calls == 4
                                      and [d["call"]["tool_calls"] for d in docs if d["call"]["conversation_class"] == "root"] == [1, 1, 0],
        "function_call_items_echoed_with_namespace": any(
            m["role"] == "assistant" and any(tc["name"] == "spawn_agent" for tc in m["tool_calls"])
            for m in roots[-1]["messages"]) and any(m["role"] == "tool" and "agent_id" in m["content"]
                                                     for m in roots[-1]["messages"]),
        "wait_agent_returned_child_text": any(m["role"] == "tool" and "child says 42" in m["content"]
                                              for m in roots[-1]["messages"]),
        "other_calls_zero": record.calls_by_class.get("other", 0) == 0,
        "reasoning_reinjected_explicitly": all(d["call"]["reinjection"] in ("explicit", "n/a") for d in docs)
                                           and any(d["call"]["reinjection"] == "explicit" for d in docs)
                                           and record.reinjection_misses == 0,
        "events_show_spawn_and_wait": [c.get("tool") for c in collab] == ["spawn_agent", "wait"]
                                      and all(c.get("status") == "completed" for c in collab),
        "turn_completed_no_failures": "turn.completed" in _event_types(events) and "turn.failed" not in _event_types(events)
                                      and record.harness_reported.get("failures") == [],
        "delivered_from_proxy_and_cross_checked": record.final_source == "proxy_root" and record.parse_status == "valid"
                                                   and record.cli_result_mismatch is False
                                                   and record.harness_reported.get("final_file_matches_last_message") is True,
        "finish_reasons_in_set": all(c.finish_reason_returned in ("stop", "tool_calls") for c in record.calls_detail),
        **common[0],
    }
    failed = S.record_case("codex/delegation_rung4", checks, detail={**common[1],
        "calls": record.calls, "classes": classes, "first_tools": first_names,
        "child_tools": _tool_names(children[0]) if children else None, "events": _event_types(events)[:12],
        "harness_version": record.spec.harness_version})
    assert not failed, (failed, classes, _event_types(events), record.harness_reported)


@pytest.mark.timeout(600)
async def test_codex_unknown_exec_command_spawns_nothing(tmp_path: Path, base):
    marker = S.home_marker()
    item = S.dev_item(base, "hmmt")
    leak = tmp_path / "LEAK_FROM_EXEC_COMMAND"
    async with SmokeStack(tmp_path, adapters=_adapters()) as stack:
        spec = stack.specs(base, arm=ARM, rung=4, items=[item])[0]
        stack.scripts[spec.run_key] = RunScript(
            root=[FakeReply(content=fake_tool_call_text("exec_command", {"cmd": f"touch {leak}"}), think_tokens=100),
                  FINAL])
        records = await stack.run(base, [spec])
    record = records[spec.run_key]
    docs = stack.journaled(spec)
    events = _events(stack, spec)
    second = docs[1]["harness_request"] if len(docs) > 1 else {"messages": []}
    outputs = [m["content"] for m in second["messages"] if m["role"] == "tool"]
    common = S.common_checks(stack, spec, record, marker)
    checks = {
        "run_done_exit_0": record.status == "done" and record.harness_exit_code == 0,
        "proxy_emitted_the_unknown_call": docs[0]["call"]["tool_calls"] == 1 and docs[0]["tool_parse"]["status"] == "ok"
                                          and "unknown_function:exec_command" in str(docs[0]["tool_parse"]["reason"]),
        "codex_answered_unsupported_call": outputs == ["unsupported call: exec_command"],
        "no_execution_item": "command_execution" not in _item_types(events)
                             and record.harness_reported.get("execution_items") == 0,
        "no_process_ran": not leak.exists(),
        "two_root_calls_no_spawn": record.calls == 2 and record.spawns == 0 and record.calls_by_class.get("root") == 2,
        "turn_completed": "turn.completed" in _event_types(events) and record.parse_status == "valid",
        **common[0],
    }
    failed = S.record_case("codex/exec_command", checks, detail={**common[1], "outputs": outputs, "items": _item_types(events)})
    assert not failed, (failed, outputs, _event_types(events))


class DroppingProxyServer(ProxyServer):
    """The real server, except that the first ``drops`` streaming replies are cut after their first frame
    (``response.created``): the wire fault the probe's case d scripted, now behind the real core."""

    def __init__(self, *args: Any, drops: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.drops_remaining = drops
        self.dropped = 0

    async def _write_frames(self, conn: h11.Connection, writer: asyncio.StreamWriter, frames: Iterable[bytes]) -> None:
        frames = list(frames)
        if self.drops_remaining <= 0:
            await super()._write_frames(conn, writer, frames)
            return
        self.drops_remaining -= 1
        self.dropped += 1
        writer.write(conn.send(h11.Data(data=frames[0])))
        await writer.drain()
        writer.close()
        raise ConnectionResetError("smoke: stream cut after the first frame")


@pytest.mark.timeout(600)
async def test_codex_missing_response_completed_retries_then_turn_failed(tmp_path: Path, base):
    marker = S.home_marker()
    item = S.dev_item(base, "omni_math_hard")
    factory = lambda core, **kw: DroppingProxyServer(core, drops=3, **kw)  # noqa: E731
    async with SmokeStack(tmp_path, adapters=_adapters(), server_factory=factory) as stack:
        spec = stack.specs(base, arm=ARM, rung=4, items=[item])[0]
        stack.scripts[spec.run_key] = RunScript(root=[FINAL])
        records = await stack.run(base, [spec])
        dropped = stack.server.dropped  # type: ignore[union-attr]
    record = records[spec.run_key]
    events = _events(stack, spec)
    errors = [str(e.get("message")) for e in events if e.get("type") == "error"]
    script = stack.scripts[spec.run_key]
    common = S.common_checks(stack, spec, record, marker)
    checks = {
        "three_requests_two_reconnects_then_turn_failed": dropped == 3 and len(record.calls_detail) == 3
                                                          and sum("Reconnecting" in m for m in errors) == 2
                                                          and "turn.failed" in _event_types(events),
        "retries_served_from_cache_one_upstream_call": record.calls == 1 and record.duplicates_served == 2
                                                       and script.think_calls() == 1
                                                       and [c.duplicate_served for c in record.calls_detail] == [False, True, True],
        "harness_exit_1_no_final_file": record.harness_exit_code == 1 and record.cli_result is None
                                        and record.harness_reported.get("final_file_present") is False,
        "every_error_is_a_stream_disconnect": bool(errors) and all("stream disconnected before completion" in m for m in errors)
                                              and "agent_message" not in _item_types(events),
        "no_rerun": record.rerun_reason is None and record.spec.attempt == 1 and record.status == "done",
        **common[0],
    }
    failed = S.record_case("codex/missing_completed", checks, detail={**common[1], "errors": errors, "events": _event_types(events)})
    assert not failed, (failed, errors, _event_types(events))


@pytest.mark.timeout(600)
async def test_codex_timeout_kills_the_whole_tree(tmp_path: Path, base):
    marker = S.home_marker()
    item = S.dev_item(base, "aime")
    # the adapter's ``timeout --kill-after=5s 17s`` fires first; the runner's own wall (90 s) never does
    async with SmokeStack(tmp_path, adapters=_adapters(wall_base=12, wall_per_rung=0, sigkill_after=5,
                                                       timeout_margin_seconds=0),
                          wall_base=90, wall_per_rung=0) as stack:
        spec = stack.specs(base, arm=ARM, rung=4, items=[item])[0]
        stack.scripts[spec.run_key] = RunScript(root=[FakeReply(delay_seconds=40.0, think_tokens=100)])
        task = asyncio.create_task(stack.run(base, [spec]))
        pgid: int | None = None
        while pgid is None:
            await asyncio.sleep(0.1)
            proc = stack.runner._in_flight.get(spec.run_key)  # type: ignore[union-attr]
            if proc is not None:
                pgid = proc.pid
            if task.done():
                break
        started = time.monotonic()
        records = await task
        ended = time.monotonic()
        group_alive_after_exit = S.process_group_alive(pgid) if pgid is not None else None
        if group_alive_after_exit:
            os.killpg(pgid, 9)          # never leave a tree behind in the test session
        await stack.settle_upstream()
        settled = stack.stack.ledger.snapshot()
    record = records[spec.run_key]
    launch = json.loads((stack.work_dir(spec) / "launch.json").read_text())
    common = S.common_checks(stack, spec, record, marker)
    checks = {
        "argv_is_timeout_wrapped": launch["argv"][:3] == ["timeout", "--kill-after=5s", "17s"],
        # the HARNESS lifetime is what the wrapper bounds (record.wall_seconds is taken when the process exits);
        # the attempt then waits for the upstream call the dead harness abandoned to settle (review major 5/14),
        # which the 40 s scripted delay makes visible here, and still ends far inside the runner's 90 s wall
        "timeout_ended_the_run": record.harness_exit_code in (124, 137, -9, -15)
                                 and 10 <= record.wall_seconds < 40 and (ended - started) < 90,
        # review major 6: a wall-killed Codex run is a TIMEOUT (it used to be recorded `done`, so Gate 2's
        # timeout_share <= 0.02 and the Gate 3 stop rule were blind for H3)
        "wall_kill_recorded_as_timeout": record.harness_reported.get("timed_out") is True
                                         and record.status == "timeout" and record.rerun_reason is None,
        "runner_wall_was_the_longer_one": record.harness_reported.get("wall_timeout_seconds") == 90,
        "whole_process_group_gone": group_alive_after_exit is False,
        "upstream_call_finished_and_settled": record.calls == 1 and settled.in_flight == 0 and settled.unresolved == 0
                                              and stack.scripts[spec.run_key].think_calls() == 1,
        **common[0],
    }
    failed = S.record_case("codex/timeout", checks, detail={**common[1], "exit_code": record.harness_exit_code,
                                                            "seconds": round(ended - started, 1),
                                                            "harness_seconds": round(record.wall_seconds, 1),
                                                            "pgid": pgid})
    assert not failed, (failed, record.harness_exit_code, record.harness_reported)


#: What a model about to delegate writes when its one grace comes (plan B3 / contract §10.28: returned as TEXT).
GRACE_SPAWN_XML = spawn_text("solve it all", prefix="I must delegate now.")
TERMINAL_FINAL = FakeReply(content="<rationale>Out of budget; answering.</rationale>\n<final>42</final>")


@pytest.mark.timeout(600)
async def test_codex_forced_close_is_text_and_ends_the_run(tmp_path: Path, base):
    """Plan B3 / contract §10.28 through the real CLI: the THINK hits the cap at rung 1, the grace continuation
    (rendered WITHOUT tools) writes ``spawn_agent`` XML, and Codex receives it as a plain ``message`` item with
    finish "stop": no ``function_call`` item, no ``collab_tool_call``, no child request; the turn completes on that
    text (exit 0, the ``-o`` file holds it) and it is the delivered answer. Any further call is refused: a probe
    sent after the harness exited gets HTTP 402 with no upstream call."""
    marker = S.home_marker()
    item = S.dev_item(base, "gpqa_diamond")
    async with SmokeStack(tmp_path, adapters=_adapters(), sigterm_after_exhaustion=45.0, probe_after_exit=True) as stack:
        spec = stack.specs(base, arm=ARM, rung=1, items=[item])[0]
        cap = spec.cap_tokens
        stack.scripts[spec.run_key] = RunScript(root=[FakeReply(think_tokens=cap + 500)],
                                                answer=FakeReply(content=GRACE_SPAWN_XML), solver=[CHILD])
        records = await stack.run(base, [spec])
    record = records[spec.run_key]
    docs = stack.journaled(spec)
    events = _events(stack, spec)
    script = stack.scripts[spec.run_key]
    probe = stack.probes.get(spec.run_key)
    call = record.calls_detail[0] if record.calls_detail else None
    reply = docs[0]["reply"] if docs else {}
    messages = [e["item"].get("text", "") for e in events if isinstance(e.get("item"), dict)
                and e["item"].get("type") == "agent_message" and e.get("type") == "item.completed"]
    common = S.common_checks(stack, spec, record, marker)
    checks: dict[str, bool] = {
        "one_served_call_forced_close_mid_call": record.calls == 1 and call is not None
                                                 and call.forced_close_kind == "mid_call"
                                                 and record.forced_close_kind == "mid_call",
        "grace_used_and_bounded": 0 < record.grace_tokens <= S.CFG["dynorch"]["grace_tokens"]
                                  and record.completion_total == cap and record.exhausted is True,
        "continuation_carries_no_tools": [c["prompt_has_tools"] for c in script.phase_calls("think")] == [True]
                                         and [c["prompt_has_tools"] for c in script.phase_calls("answer")] == [False],
        "grace_returned_as_text_finish_stop": call is not None and call.finish_reason_returned == "stop"
                                              and call.tool_calls == 0 and reply.get("tool_calls") == []
                                              and "<tool_call>" in str(reply.get("content")),
        "nothing_spawned": record.spawns == 0 and record.calls_by_class.get("solver", 0) == 0
                           and script.think_calls("solver") == 0
                           and "collab_tool_call" not in _item_types(events),
        "turn_completed_on_that_text": "turn.completed" in _event_types(events)
                                       and "turn.failed" not in _event_types(events)
                                       and record.harness_exit_code == 0 and len(messages) == 1
                                       and "<tool_call>" in messages[0],
        "delivered_answer_is_the_grace_text": record.final_source == "proxy_root"
                                              and record.final_message is not None
                                              and "<tool_call>" in record.final_message
                                              and record.cli_result_mismatch is False
                                              and record.harness_reported.get("final_file_present") is True,
        "further_call_refused_402_no_upstream": probe is not None and probe.status == 402
                                                and probe.code == "budget_exhausted"
                                                and probe.upstream_calls_after == probe.upstream_calls_before,
        "no_upstream_call_after_the_grace": script.think_calls() == 1 and len(script.calls) == 2,
        "no_rerun": record.status == "done" and record.rerun_reason is None and record.spec.attempt == 1,
        "not_killed_by_watchdog": record.harness_reported.get("sigterm_after_exhaustion") is False
                                  and record.harness_reported.get("timed_out") is False,
        "finish_reasons_in_set": all(c.finish_reason_returned in ("stop", "tool_calls") for c in record.calls_detail),
        **common[0],
    }
    failed = S.record_case("codex/forced_close_text", checks, detail={**common[1],
        "probe": None if probe is None else probe.__dict__, "messages": [m[:200] for m in messages],
        "upstream": script.calls, "harness_events": _event_types(events), "parse_status": record.parse_status})
    assert not failed, (failed, _event_types(events), record.harness_reported)


@pytest.mark.timeout(600)
async def test_codex_subagent_402_then_terminal_ends_the_run(tmp_path: Path, base):
    """The 402 case without the grace (plan B4; contract §10.20 for Codex): the root's THINK stops naturally with a
    ``spawn_agent`` call that leaves ``remaining < 64``; the child's request (``x-openai-subagent``) is refused with
    HTTP 402 and Codex re-sends it at most ``stream_max_retries`` (2) times, each refused in-process with no upstream
    call; the root's next request gets its one TERMINAL answer (B4) and the turn completes on it. Pinned: every
    request after the first 402 is a 402 or that one root TERMINAL; no THINK after the spawn; the run ends within
    10 s of the first 402; a probe after the exit gets 402. The TERMINAL is held 3 s upstream so the child's request
    reaches the proxy whichever of the two Codex sends first (they race for the run's FIFO lock)."""
    marker = S.home_marker()
    item = S.dev_item(base, "gpqa_diamond")
    async with SmokeStack(tmp_path, adapters=_adapters(), sigterm_after_exhaustion=45.0, probe_after_exit=True) as stack:
        spec = stack.specs(base, arm=ARM, rung=1, items=[item])[0]
        cap = spec.cap_tokens
        threshold = int(S.CFG["dynorch"]["terminal_threshold_tokens"])
        spent = cap - (threshold - 24)
        spawn = spawn_text("Solve the whole problem and report the answer.")
        stack.scripts[spec.run_key] = RunScript(
            root=[FakeReply(content=spawn, think_tokens=spent - -(-len(spawn) // 4)), FINAL], solver=[CHILD],
            answer=FakeReply(content=TERMINAL_FINAL.content, delay_seconds=3.0))
        records = await stack.run(base, [spec])
        ended = time.monotonic()
    record = records[spec.run_key]
    docs = stack.journaled(spec)
    events = _events(stack, spec)
    script = stack.scripts[spec.run_key]
    probe = stack.probes.get(spec.run_key)
    horizon = probe.sent_at if probe is not None else float("inf")
    timed = [e for e in stack.reply_log.timed.get(spec.run_key, []) if e.at < horizon]
    first_402 = next((e for e in timed if e.kind == "error" and e.status == 402), None)
    after = [e for e in timed if first_402 is not None and e.at > first_402.at]
    refusals = [e for e in timed if e.kind == "error"]
    classes = [d["call"]["conversation_class"] for d in docs]
    kinds = [c.forced_close_kind for c in record.calls_detail]
    common = S.common_checks(stack, spec, record, marker)
    checks: dict[str, bool] = {
        "root_think_stopped_naturally_with_a_spawn": bool(record.calls_detail)
                                                     and record.calls_detail[0].forced_close_kind == "none"
                                                     and record.calls_detail[0].finish_reason_returned == "tool_calls"
                                                     and record.calls_detail[0].completion_tokens == spent
                                                     and record.spawns == 1 and cap - spent < threshold,
        "child_request_refused_402": first_402 is not None and record.refused_calls >= 1
                                     and all(e.status == 402 and e.code == "budget_exhausted" for e in refusals)
                                     and script.think_calls("solver") == 0
                                     and record.calls_by_class.get("solver", 0) == 0,
        # contract §10.20: Codex re-sends an HTTP 402 stream_max_retries (2) times per conversation, in-process
        "at_most_stream_max_retries_per_conversation": len(refusals) <= STREAM_MAX_RETRIES + 1,
        "after_first_402_only_402s_or_the_root_terminal": all(
            (e.kind == "error" and e.status == 402) or (e.kind == "reply" and e.conversation_class == "root")
            for e in after) and sum(1 for e in after if e.kind == "reply") <= 1,
        "run_ended_within_10s_of_402": first_402 is not None and (ended - first_402.at) < 10.0,
        "terminal_served_once_no_tools": classes == ["root", "root"] and kinds == ["none", "terminal"]
                                         and record.forced_close_kind == "terminal"
                                         and 0 < record.grace_tokens <= S.CFG["dynorch"]["grace_tokens"]
                                         and [c["prompt_has_tools"] for c in script.phase_calls("terminal")] == [False],
        "no_think_call_after_the_spawn": script.think_calls() == 1 and len(script.calls) == 2,
        "turn_completed_on_the_terminal_answer": "turn.completed" in _event_types(events)
                                                 and record.harness_exit_code == 0
                                                 and record.harness_reported.get("final_file_present") is True,
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
    failed = S.record_case("codex/subagent_402_then_terminal", checks, detail={**common[1],
        "seconds_from_402_to_end": None if first_402 is None else round(ended - first_402.at, 2),
        "events": [e.__dict__ for e in stack.reply_log.timed.get(spec.run_key, [])],
        "probe": None if probe is None else probe.__dict__, "upstream": script.calls, "classes": classes,
        "harness_events": _event_types(events)})
    assert not failed, (failed, _event_types(events), record.harness_reported)


@pytest.mark.timeout(600)
async def test_codex_interruption_retry_is_refused_by_the_latch(tmp_path: Path, base):
    """Contract §10.38 through the real Codex CLI: call 1 is interrupted upstream after the headers went out (an
    in-stream ``response.failed``, the full grant settled). Codex re-sends the turn (``request_max_retries`` /
    ``stream_max_retries``); every re-send is refused 502 by the latch before any grant -- no second upstream call,
    no TERMINAL -- and the turn fails. The run is ``transport_failed`` / ``upstream_latched``; the runner's
    infrastructure rerun (attempt 2, full budget, no model rerun consumed) is the scored record."""
    marker = S.home_marker()
    item = S.dev_item(base, "aime")
    async with SmokeStack(tmp_path, adapters=_adapters()) as stack:
        spec = stack.specs(base, arm=ARM, rung=4, items=[item])[0]
        stack.scripts[spec.run_key] = RunScript(root=[FakeReply(fault=Fault.STREAM_INTERRUPTED, delay_seconds=3.5),
                                                      FINAL])
        records = await stack.run(base, [spec])
        first = stack.store.scan(include_superseded=True)[spec.run_key]
    final = next(iter(records.values()))
    script = stack.scripts[spec.run_key]
    events = stack.reply_log.timed.get(spec.run_key, [])
    errors = [e for e in events if e.kind == "error"]
    first_events = _events(stack, spec)
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
        "turn_failed": "turn.failed" in _event_types(first_events)
                       and "turn.completed" not in _event_types(first_events),
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
    failed = S.record_case("codex/interruption_retry_latched", checks, detail={**common[1],
        "events": [e.__dict__ for e in events], "upstream": script.calls, "refused_calls": first.refused_calls,
        "event_types": _event_types(first_events)})
    assert not failed, (failed, first.refused_calls, [e.__dict__ for e in events], _event_types(first_events))


@pytest.mark.timeout(600)
async def test_codex_rung25_ends_with_success(tmp_path: Path, base):
    marker = S.home_marker()
    item = S.dev_item(base, "hmmt")
    async with SmokeStack(tmp_path, adapters=_adapters()) as stack:
        spec = stack.specs(base, arm=ARM, rung=25, items=[item])[0]
        stack.scripts[spec.run_key] = RunScript(root=[FakeReply(think_tokens=3000)])
        records = await stack.run(base, [spec])
    record = records[spec.run_key]
    events = _events(stack, spec)
    common = S.common_checks(stack, spec, record, marker)
    checks = {
        "rung25_cap": spec.cap_tokens == 25 * 1024 and record.calls_detail[0].granted_max_tokens == 25 * 1024,
        "turn_completed": "turn.completed" in _event_types(events) and "turn.failed" not in _event_types(events),
        "run_done_graded": record.status == "done" and record.parse_status == "valid" and record.correct in (True, False)
                           and record.harness_exit_code == 0 and record.cli_result_mismatch is False,
        "one_call_no_grace": record.calls == 1 and record.grace_tokens == 0 and record.forced_close_kind == "none",
        # review major 6: the runner's wall (3,600 s at k=25) is authoritative; the `timeout` wrapper sits behind
        # it (wall + sigkill_after + margin) so a Codex run that hits the wall is recorded as a timeout, not done
        "wall_timeout_is_3600s_and_the_wrapper_sits_behind_it":
            record.harness_reported.get("wall_timeout_seconds") == 3600
            and json.loads((stack.work_dir(spec) / "launch.json").read_text())["argv"][2] == "3690s",
        **common[0],
    }
    failed = S.record_case("codex/rung25", checks, detail={**common[1], "wall_seconds": record.wall_seconds})
    assert not failed, (failed, _event_types(events))


async def _delay_case(tmp_path: Path, base, delay: float, name: str) -> None:
    marker = S.home_marker()
    item = S.dev_item(base, "aime")
    async with SmokeStack(tmp_path, adapters=_adapters()) as stack:
        spec = stack.specs(base, arm=ARM, rung=4, items=[item])[0]
        stack.scripts[spec.run_key] = RunScript(root=[FakeReply(delay_seconds=delay, think_tokens=200)])
        records = await stack.run(base, [spec])
    record = records[spec.run_key]
    events = _events(stack, spec)
    script = stack.scripts[spec.run_key]
    common = S.common_checks(stack, spec, record, marker)
    checks = {
        "exactly_one_upstream_call": script.think_calls() == 1 and len(script.calls) == 1 and record.calls == 1
                                     and record.duplicates_served == 0,
        "turn_completed_no_reconnect": "turn.completed" in _event_types(events)
                                       and not any(e.get("type") == "error" for e in events),
        "waited_through_the_delay": record.wall_seconds >= delay and record.calls_detail[0].latency_seconds >= delay,
        "run_done_valid": record.status == "done" and record.parse_status == "valid" and record.harness_exit_code == 0
                          and record.cli_result_mismatch is False,
        **common[0],
    }
    failed = S.record_case(f"codex/{name}", checks, detail={**common[1], "delay": delay, "wall_seconds": record.wall_seconds})
    assert not failed, (failed, _event_types(events))


@pytest.mark.timeout(600)
async def test_codex_delayed_reply_35s(tmp_path: Path, base):
    await _delay_case(tmp_path, base, 35.0, "delayed_35s")


@pytest.mark.slow
@pytest.mark.timeout(1500)
async def test_codex_delayed_reply_330s(tmp_path: Path, base):
    await _delay_case(tmp_path, base, 330.0, "delayed_330s")


@pytest.mark.slow
@pytest.mark.timeout(1500)
async def test_codex_delayed_reply_660s(tmp_path: Path, base):
    await _delay_case(tmp_path, base, 660.0, "delayed_660s")
