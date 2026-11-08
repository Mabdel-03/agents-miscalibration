"""WP-B's scripted stand-in for the WP-A core, kept after the merge ONLY as the wire layer's unit-test double
(``test_httpd`` needs a core whose replies it can delay, fail and script byte-for-byte; ``test_responses_format``'s
event-order tests likewise). It implements ProxyCore's §3 signatures and the budget semantics the wire layer depends
on (one in-flight call per run, grant ``C - spent``, the TERMINAL grace on the root, 402 afterwards). It never talks
to anything upstream; token counts come from the script.

Everything that touches the budget, the ledger, rendering or the delivered answer is tested against the REAL
``ProxyCore`` over ``api/fake.py`` (``tests/unit/dynorch/test_core_flow.py``, the chat/Responses parity test and
``tests/integration/test_dynorch_e2e_fake_harness.py``). The pre-merge records shim and ``StubRunStore`` were
deleted at the merge (contract §9).
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from agent_allocation.dynorch.records import (
    CallRecord, ChatReply, ChatRequest, ProxyError, ReplyUsage, RunRecord, RunSpec, ToolCall,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Scripted:
    """One scripted upstream outcome for ``StubProxyCore``.

    ``content``/``reasoning`` are the visible/reasoning texts; ``tool_calls`` is a sequence of ``(name, arguments)``;
    ``completion_tokens`` is what the fake "model" spent (capped at the grant by the stub, exactly as the real proxy
    caps ``max_tokens``); ``delay_seconds`` sleeps before replying (the keepalive tests); ``error`` raises that
    ``ProxyError`` instead of replying (after the delay).
    """
    content: str = ""
    reasoning: str = ""
    tool_calls: Sequence[tuple[str, dict[str, Any]]] = ()
    completion_tokens: int = 100
    prompt_tokens: int = 500
    reasoning_tokens: int | None = None
    delay_seconds: float = 0.0
    error: ProxyError | None = None
    finish_reason_upstream: str = "stop"


ScriptFn = Callable[["StubRun", ChatRequest, int], Scripted | Awaitable[Scripted]]


@dataclass
class StubRun:
    spec: RunSpec
    root_system_sha: str
    subagent_system_sha: str | None
    record: RunRecord
    root_text: str = ""
    subagent_text: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    conversations: dict[str, str] = field(default_factory=dict)      # conversation_key -> class
    closed: bool = False
    grace_used: bool = False
    served: list[ChatReply] = field(default_factory=list)

    @property
    def spent(self) -> int:
        return self.record.completion_total

    @property
    def remaining(self) -> int:
        return max(0, self.spec.cap_tokens - self.spent)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_run_record(spec: RunSpec) -> RunRecord:
    """A fresh ``RunRecord`` in the ``running`` state with every counter at zero (contract §5)."""
    return RunRecord(spec=spec, status="running", calls=0, conversations=0, spawns=0, calls_by_class={},
                     completion_total=0, reasoning_total=None, prompt_total=0, grace_tokens=0, overdraft_tokens=0,
                     forced_close_kind="none", exhausted=False, refused_calls=0, guard_triggered=False,
                     tool_parse_failures=0, reinjection_hits=0, reinjection_misses=0, duplicates_served=0,
                     final_message=None, final_source="none", cli_result=None, cli_result_mismatch=None,
                     parse_status=None, final_answer=None, grade_status=None, correct=None, grade_detail=None,
                     wall_seconds=0.0, harness_exit_code=None, harness_reported={}, cost_usd="0", started_at=_utc(),
                     finished_at=None, error=None, rerun_reason=None, calls_detail=[])


def default_script(run: StubRun, request: ChatRequest, call_no: int) -> Scripted:
    """A conversation shaped like the fake harness expects: the root delegates once at rung >= 4, then answers.

    The final answer is derived from the item id (``fake.default_final_answer``-style: an integer for math items,
    a letter for gpqa) so the e2e can grade against the base run root's golds without being right.
    """
    item = run.spec.item_id
    if item.startswith("gpqa_diamond/"):
        answer = "ABCD"[int(hashlib.sha256(item.encode()).hexdigest(), 16) % 4]
    else:
        answer = str(int(hashlib.sha256(item.encode()).hexdigest(), 16) % 1000)
    klass = run.conversations.get(conversation_key(request), "root")
    n_tool_msgs = sum(1 for m in request.messages if m.role == "tool")
    spawn = next((t.name for t in request.tools if t.name in run.spec.spawn_tools), None)
    if klass == "root" and spawn is not None and run.spec.rung >= 4 and n_tool_msgs == 0:
        return Scripted(reasoning="Let me delegate.", content="",
                        tool_calls=[(spawn, {"task": f"Solve: {item}", "description": "solve",
                                             "prompt": f"Solve: {item}", "subagent_type": "solver"})],
                        completion_tokens=min(300, run.remaining))
    text = f"<rationale>stub</rationale>\n<final>{answer}</final>"
    return Scripted(reasoning="thinking", content=text, completion_tokens=min(200, run.remaining))


def conversation_key(request: ChatRequest) -> str:
    """sha256 of (system text, first user text, sub-agent marker) -- the §1/§10.1 conversation identity (a Codex
    fork shares the root's texts and differs only by the marker; the real core keys on wire-level thread markers)."""
    system = next((m.content for m in request.messages if m.role == "system"), "")
    user = next((m.content for m in request.messages if m.role == "user"), "")
    return sha256_text(system + "\x1f" + user + ("\x1fsubagent" if request.subagent else ""))


class StubProxyCore:
    """Scripted stand-in for ``ProxyCore`` (contract §3 signatures)."""

    def __init__(self, script: ScriptFn | None = None, *, terminal_threshold: int = 64, grace: int = 1024) -> None:
        self.script: ScriptFn = script or default_script
        self.terminal_threshold = terminal_threshold
        self.grace = grace
        self.runs: dict[str, StubRun] = {}
        self.calls: list[tuple[str, ChatRequest]] = []

    # -- §3 --------------------------------------------------------------------------------------------
    def open_run(self, spec: RunSpec, *, root_system_text: str, subagent_system_text: str | None) -> None:
        if spec.run_key in self.runs:
            raise ValueError(f"run {spec.run_key} already open")
        self.runs[spec.run_key] = StubRun(spec=spec, root_system_sha=sha256_text(root_system_text),
                                          subagent_system_sha=(sha256_text(subagent_system_text)
                                                               if subagent_system_text is not None else None),
                                          record=new_run_record(spec), root_text=root_system_text,
                                          subagent_text=subagent_system_text)

    def close_run(self, run_key: str, *, status: str) -> RunRecord:
        run = self._run(run_key)
        run.closed = True
        run.record.status = status
        run.record.finished_at = _utc()
        return run.record

    def budget_view(self, run_key: str) -> dict[str, int]:
        run = self._run(run_key)
        return {"cap": run.spec.cap_tokens, "spent": run.spent, "remaining": run.remaining, "calls": run.record.calls}

    async def chat(self, run_key: str, request: ChatRequest) -> ChatReply:
        run = self.runs.get(run_key)
        if run is None:
            raise ProxyError(404, "unknown_run", f"no open run {run_key}")
        if run.closed:
            raise ProxyError(409, "run_closed", f"run {run_key} is closed")
        bad = [t.name for t in request.tools if t.name not in run.spec.allowed_tools]
        if bad:
            raise ProxyError(400, "tool_surface_violation", f"tools outside the allowlist: {bad}")
        self.calls.append((run_key, request))
        queued = time.monotonic()
        async with run.lock:
            wait = time.monotonic() - queued
            key = conversation_key(request)
            klass = run.conversations.get(key)
            if klass is None:
                klass = self._classify(run, request)
                run.conversations[key] = klass
                run.record.conversations += 1
            call_no = run.record.calls + 1
            terminal = False
            if run.remaining < self.terminal_threshold:
                if klass == "root" and not run.grace_used:
                    terminal = True
                else:
                    run.record.refused_calls += 1
                    run.record.exhausted = True
                    raise ProxyError(402, "budget_exhausted", "completion-token budget exhausted")
            granted = self.grace if terminal else run.remaining
            scripted = self.script(run, request, call_no)
            if asyncio.iscoroutine(scripted):
                scripted = await scripted
            if scripted.delay_seconds:
                await asyncio.sleep(scripted.delay_seconds)
            if scripted.error is not None:
                raise scripted.error
            spent = min(scripted.completion_tokens, granted)
            tool_calls = tuple(ToolCall(id=f"call_{call_no}_{i}", name=name, arguments=dict(args))
                               for i, (name, args) in enumerate(scripted.tool_calls))
            finish = "tool_calls" if tool_calls else "stop"
            forced_close = "terminal" if terminal else "none"
            if terminal:
                run.grace_used = True
                run.record.grace_tokens += spent
                run.record.forced_close_kind = "terminal"
                run.record.exhausted = True
            else:
                run.record.completion_total += spent
            run.record.calls = call_no
            run.record.prompt_total += scripted.prompt_tokens
            run.record.calls_by_class[klass] = run.record.calls_by_class.get(klass, 0) + 1
            spawn_calls = sum(1 for t in tool_calls if t.name in run.spec.spawn_tools) if klass == "root" else 0
            run.record.spawns += spawn_calls
            call = CallRecord(
                call_no=call_no, conversation_key=key, conversation_class=klass,
                requested_max_tokens=request.requested_max_tokens, granted_max_tokens=granted,
                queue_wait_seconds=round(wait, 6), prompt_tokens=scripted.prompt_tokens, completion_tokens=spent,
                reasoning_tokens=scripted.reasoning_tokens, cached_prompt_tokens=None,
                reasoning_chars=len(scripted.reasoning), visible_chars=len(scripted.content),
                forced_close_kind=forced_close, grace_tokens=spent if terminal else 0,
                finish_reason_upstream=scripted.finish_reason_upstream, finish_reason_returned=finish,
                tool_calls=len(tool_calls), spawn_calls=spawn_calls, tool_parse="ok" if tool_calls else "none",
                reinjection="n/a", reasoning_effort_requested=request.requested_reasoning_effort, cost_usd="0",
                generation_ids=(f"gen-{run_key[:6]}-{call_no}",), latency_seconds=scripted.delay_seconds,
                transport_status="ok", returned_status=200, stream=request.stream, wire=request.wire,
                cache_status="MISS", duplicate_served=False, ignored_body_keys=request.ignored_keys)
            run.record.calls_detail.append(call)
            usage = ReplyUsage(prompt_tokens=scripted.prompt_tokens, completion_tokens=spent,
                               reasoning_tokens=scripted.reasoning_tokens, cached_prompt_tokens=None)
            headers = {"X-Dynorch-Call-No": str(call_no), "X-Dynorch-Cap-Tokens": str(run.spec.cap_tokens),
                       "X-Dynorch-Spent-Tokens": str(run.spent), "X-Dynorch-Remaining-Tokens": str(run.remaining),
                       "X-Dynorch-Forced-Close": forced_close, "X-Dynorch-Tool-Parse": call.tool_parse}
            reply = ChatReply(content=scripted.content, reasoning_content=scripted.reasoning, tool_calls=tool_calls,
                              finish_reason=finish, usage=usage, call=call, headers=headers)
            run.served.append(reply)
            return reply

    # -- helpers -----------------------------------------------------------------------------------------
    def _run(self, run_key: str) -> StubRun:
        run = self.runs.get(run_key)
        if run is None:
            raise ProxyError(404, "unknown_run", f"no open run {run_key}")
        return run

    @staticmethod
    def _classify(run: StubRun, request: ChatRequest) -> str:
        """Contract §1 as amended by §10.1: a sub-agent marker or the solver text after the harness prefix -> solver;
        the first non-solver conversation -> root; a later conversation starting with the root text -> root; else other."""
        system = next((m.content for m in request.messages if m.role == "system"), "")
        if request.subagent or (run.subagent_text is not None and run.subagent_text.strip()
                                and run.subagent_text.rstrip("\n") in system):
            return "solver"
        if "root" not in run.conversations.values():
            return "root"
        if system.startswith(run.root_text.rstrip("\n")) or system.endswith(run.root_text.rstrip("\n")):
            return "root"
        return "other"
