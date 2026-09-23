"""WP-B test doubles for the WP-A core (contract ``docs/dynorch/00_contract.md`` §2, §3, §5; deleted at merge).

Two things live here:

1. **The records shim.** Until WP-A lands ``agent_allocation.dynorch.records``, importing this module installs a
   module of the same name into ``sys.modules`` holding the §2/§5 dataclasses *verbatim from the contract*. When
   the real module is importable the shim is not installed and the real records are used, so every WP-B module
   imports ``agent_allocation.dynorch.records`` directly and the merge is a no-op. Test modules therefore import
   ``tests.unit.dynorch.stubs`` **before** any ``agent_allocation.dynorch.*`` module.
2. **``StubProxyCore``** implementing ProxyCore's §3 signatures with scripted replies, and ``StubRunStore``
   implementing the §5 ``RunStore`` surface WP-B codes against (``save``/``scan``/``is_terminal``).

The stub core mirrors the budget semantics the wire layer and runner depend on (plan §2 B1, B2, B4, B7): one
in-flight call per run (FIFO lock), grant ``C - spent``, conversation classification by system-text sha256, a
TERMINAL grace on the root conversation when ``remaining < 64``, HTTP 402 ``budget_exhausted`` afterwards, and a
``CallRecord`` per served call. It never talks to anything upstream; token counts come from the script.
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
import time
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

import json

from agent_allocation.storage._atomic import replace_atomic

# ---------------------------------------------------------------------------------------------------------
# 1. Records shim (contract §2 + §5, verbatim)
# ---------------------------------------------------------------------------------------------------------


def _build_records_module() -> types.ModuleType:
    module = types.ModuleType("agent_allocation.dynorch.records")
    module.__doc__ = "WP-B shim of the WP-A records module (contract §2/§5). Installed only when the real one is absent."
    from dataclasses import dataclass as _dc, field as _field

    @_dc(frozen=True)
    class ToolSpec:
        name: str
        description: str
        parameters: dict[str, Any]
        namespace: str | None = None

    @_dc(frozen=True)
    class ToolCall:
        id: str
        name: str
        arguments: dict[str, Any] | str

    @_dc(frozen=True)
    class Message:
        role: str
        content: str
        reasoning_content: str | None = None
        tool_calls: tuple[ToolCall, ...] = ()
        tool_call_id: str | None = None
        name: str | None = None

    @_dc(frozen=True)
    class ChatRequest:
        messages: tuple[Message, ...]
        tools: tuple[ToolSpec, ...]
        requested_max_tokens: int | None
        requested_reasoning_effort: str | None
        stream: bool
        ignored_keys: tuple[str, ...]
        wire: str

    @_dc(frozen=True)
    class ReplyUsage:
        prompt_tokens: int | None
        completion_tokens: int | None
        reasoning_tokens: int | None
        cached_prompt_tokens: int | None

    @_dc(frozen=True)
    class RunSpec:
        run_key: str
        arm: str
        harness: str
        item_id: str
        benchmark: str
        stratum: str
        shard_id: str
        endpoint_tag: str
        rung: int
        cap_tokens: int
        repeat: int
        attempt: int
        generation: int
        partition: str
        allowed_tools: tuple[str, ...]
        spawn_tools: tuple[str, ...]
        root_call_ceiling: int | None
        sampling_hash: str
        template_sha256: str
        prompt_version: str
        proxy_version: str
        harness_version: str

    @_dc
    class CallRecord:
        call_no: int
        conversation_key: str
        conversation_class: str
        requested_max_tokens: int | None
        granted_max_tokens: int
        queue_wait_seconds: float
        prompt_tokens: int | None
        completion_tokens: int | None
        reasoning_tokens: int | None
        cached_prompt_tokens: int | None
        reasoning_chars: int
        visible_chars: int
        forced_close_kind: str
        grace_tokens: int
        finish_reason_upstream: str | None
        finish_reason_returned: str
        tool_calls: int
        spawn_calls: int
        tool_parse: str
        reinjection: str
        reasoning_effort_requested: str | None
        cost_usd: str
        generation_ids: tuple[str, ...]
        latency_seconds: float
        transport_status: str
        returned_status: int
        stream: bool
        wire: str
        cache_status: str | None
        duplicate_served: bool
        ignored_body_keys: tuple[str, ...]

    @_dc(frozen=True)
    class ChatReply:
        content: str
        reasoning_content: str
        tool_calls: tuple[ToolCall, ...]
        finish_reason: str
        usage: ReplyUsage
        call: CallRecord
        headers: dict[str, str]

    @_dc
    class RunRecord:
        spec: RunSpec
        status: str
        calls: int
        conversations: int
        spawns: int
        calls_by_class: dict[str, int]
        completion_total: int
        reasoning_total: int | None
        prompt_total: int
        grace_tokens: int
        overdraft_tokens: int
        forced_close_kind: str
        exhausted: bool
        refused_calls: int
        guard_triggered: bool
        tool_parse_failures: int
        reinjection_hits: int
        reinjection_misses: int
        duplicates_served: int
        final_message: str | None
        final_source: str
        cli_result: str | None
        cli_result_mismatch: bool | None
        parse_status: str | None
        final_answer: str | None
        grade_status: str | None
        correct: bool | None
        grade_detail: str | None
        wall_seconds: float
        harness_exit_code: int | None
        harness_reported: dict[str, Any]
        cost_usd: str
        started_at: str
        finished_at: str | None
        error: str | None
        rerun_reason: str | None
        calls_detail: list[CallRecord] = _field(default_factory=list)

    class ProxyError(Exception):
        """The only exception the wire layer converts (contract §2)."""

        def __init__(self, status: int, code: str, message: str, retryable: bool = False) -> None:
            super().__init__(f"{status} {code}: {message}")
            self.status = status
            self.code = code
            self.message = message
            self.retryable = retryable

    for obj in (ToolSpec, ToolCall, Message, ChatRequest, ReplyUsage, RunSpec, CallRecord, ChatReply, RunRecord,
                ProxyError):
        obj.__module__ = module.__name__
        setattr(module, obj.__name__, obj)
    module.__all__ = ["ToolSpec", "ToolCall", "Message", "ChatRequest", "ReplyUsage", "RunSpec", "CallRecord",
                      "ChatReply", "RunRecord", "ProxyError"]
    return module


def _install_shim() -> bool:
    """Install the shim unless the real records module imports. Returns True when the shim is in use."""
    name = "agent_allocation.dynorch.records"
    if name in sys.modules:
        return getattr(sys.modules[name], "_WPB_SHIM", False)
    try:
        import importlib
        importlib.import_module(name)
        return False
    except ModuleNotFoundError as exc:
        if exc.name not in (name, "agent_allocation.dynorch"):
            raise
    module = _build_records_module()
    module._WPB_SHIM = True  # type: ignore[attr-defined]
    sys.modules[name] = module
    return True


SHIM_IN_USE: bool = _install_shim()

from agent_allocation.dynorch.records import (  # noqa: E402  (after the shim is installed)
    CallRecord, ChatReply, ChatRequest, Message, ProxyError, ReplyUsage, RunRecord, RunSpec, ToolCall, ToolSpec,
)


def real_core_available() -> bool:
    """Whether WP-A's ``agent_allocation.dynorch.core.ProxyCore`` can be imported (the e2e switches on it)."""
    try:
        import importlib
        importlib.import_module("agent_allocation.dynorch.core")
        return True
    except ModuleNotFoundError:
        return False


# ---------------------------------------------------------------------------------------------------------
# 2. Scripted core
# ---------------------------------------------------------------------------------------------------------

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
    """sha256 of (system text, first user text) -- the contract §1 conversation identity."""
    system = next((m.content for m in request.messages if m.role == "system"), "")
    user = next((m.content for m in request.messages if m.role == "user"), "")
    return sha256_text(system + "\x1f" + user)


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
                                          record=new_run_record(spec))

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
        if not run.conversations:
            return "root"
        system = next((m.content for m in request.messages if m.role == "system"), "")
        sha = sha256_text(system)
        if sha == run.root_system_sha and "root" not in run.conversations.values():
            return "root"
        if run.subagent_system_sha is not None and sha == run.subagent_system_sha:
            return "solver"
        return "other"


# ---------------------------------------------------------------------------------------------------------
# 3. Run store
# ---------------------------------------------------------------------------------------------------------

TERMINAL_STATUSES = frozenset({"done", "failed", "timeout", "transport_failed", "refused_before_start"})


def record_to_json(record: RunRecord) -> dict[str, Any]:
    return asdict(record)


def record_from_json(data: Mapping[str, Any]) -> RunRecord:
    spec = RunSpec(**{**data["spec"], "allowed_tools": tuple(data["spec"]["allowed_tools"]),
                      "spawn_tools": tuple(data["spec"]["spawn_tools"])})
    calls = [CallRecord(**{**c, "generation_ids": tuple(c["generation_ids"]),
                           "ignored_body_keys": tuple(c["ignored_body_keys"])}) for c in data["calls_detail"]]
    body = {k: v for k, v in data.items() if k not in ("spec", "calls_detail")}
    return RunRecord(spec=spec, calls_detail=calls, **body)


class StubRunStore:
    """``<run_root>/runs/<arm>/<rung>/<repeat>/<item_slug>.json`` via ``replace_atomic`` (contract §5)."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = Path(run_root)

    def path_for(self, spec: RunSpec) -> Path:
        slug = spec.item_id.replace("/", "__")
        return self.run_root / "runs" / spec.arm / str(spec.rung) / str(spec.repeat) / f"{slug}.json"

    def save(self, record: RunRecord) -> None:
        payload = json.dumps(record_to_json(record), sort_keys=True, ensure_ascii=False, indent=1)
        replace_atomic(self.path_for(record.spec), payload.encode("utf-8"))

    def scan(self) -> dict[str, RunRecord]:
        out: dict[str, RunRecord] = {}
        root = self.run_root / "runs"
        if not root.is_dir():
            return out
        for path in sorted(root.rglob("*.json")):
            record = record_from_json(json.loads(path.read_text("utf-8")))
            out[record.spec.run_key] = record
        return out

    @staticmethod
    def is_terminal(record: RunRecord) -> bool:
        return record.status in TERMINAL_STATUSES
