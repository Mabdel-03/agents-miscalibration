"""WP-A, contract §3: ``ProxyCore`` -- the per-call flow of the metering proxy (plan §2 B1-B9, §5).

``chat(run_key, request)`` does, in order: authenticate the run (open, not closed) -> classify the conversation by
system text (root / solver / other; B1) -> fail-closed tool-surface check (any tool outside ``spec.allowed_tools`` ->
``tool_surface_violation``) -> the run's FIFO lock (one upstream call in flight per run; B2) -> reasoning
re-injection + render (§4) -> duplicate protection (an identical rendered prompt already answered under this run
is served from the cache with no upstream call; B9) -> the budget boundary: after the grace every call is
``budget_exhausted``; a cost guard (B6) or ``remaining < terminal_threshold`` (B4) gives the root conversation one
TERMINAL call (thinking disabled, the verbatim terminal sentence, ``max_tokens = G``, no tools) and every other
call HTTP 402 -> grant ``max_tokens = C - spent`` (E0: ``min(C - spent, root ceiling)``) -> ``Upstream.complete``
(``$`` reserve / mark_sent / settle inside) -> split reasoning / visible -> forced close (B3: root only, once:
``mid_call`` when the cap bound inside ``<think>``, ``visible_continuation`` when it bound after the close; the
ANSWER continuation is exactly ``api/forced.py::answer_prompt(prompt, reasoning, think_close_text)`` with
``max_tokens = G``; a sub-agent that hits its grant gets its truncated visible text with ``finish_reason="stop"``,
``truncated_subagent``; an E0 root call bound by its ceiling, not the cap, is closed with a continuation charged
inside C, ``ceiling_close``) -> tool-call parse (§4) -> reasoning-cache insert -> journal (``CallJournal``) ->
``RunRecord`` update persisted every call (``RunStore``) -> ``ChatReply`` with ``X-Dynorch-*`` headers and a
``finish_reason`` in exactly ``{stop, tool_calls}``.

Accounting: ``completion_total`` counts THINK completion tokens of every served call (inside C); the one grace
(``grace_tokens``) is outside C; a forced-close pair is counted once (THINK + ANSWER) both on the reply's usage and
on the run record. The delivered answer (B7) is the last root-conversation assistant message with non-empty content
served by the proxy (``close_run`` fills ``final_message`` / ``final_source``).
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, Sequence

from agent_allocation.api.forced import answer_prompt
from agent_allocation.api.normalize import answer_visible_text
from agent_allocation.contracts import Usage
from agent_allocation.dynorch.budget import Grant, TokenBudget
from agent_allocation.dynorch.config import PROXY_VERSION, DynorchSettings
from agent_allocation.dynorch.journal import JOURNAL_LAYOUT, CallJournal, RunStore, attempt_id
from agent_allocation.dynorch.records import (
    CallRecord, ChatReply, ChatRequest, Message, ProxyError, ReplyUsage, RunRecord, RunSpec, TERMINAL_RUN_STATUSES,
    ToolCall, UpstreamResult, new_run_record, to_json,
)
from agent_allocation.dynorch.template import (
    render_chat, rendered_sha256, requested_effort, system_sha256, template_effort,
)
from agent_allocation.dynorch.toolcalls import ToolParse, parse_tool_calls
from agent_allocation.dynorch.upstream import Upstream, UpstreamFailed

INTERRUPTED_LITERAL = "[Tool execution interrupted by user]"
_LEADING_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
PHASE_THINK, PHASE_ANSWER, PHASE_TERMINAL = "think", "answer", "terminal"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def normalise_content(content: str) -> str:
    """Reasoning-cache key 2 (§4): strip a leading re-sent reasoning block and the synthetic interruption text."""
    text = _LEADING_THINK.sub("", content, count=1)
    return text.replace(INTERRUPTED_LITERAL, "").strip()


def content_key(content: str) -> str | None:
    text = normalise_content(content)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None


class ReasoningCache:
    """§4: reasoning the proxy produced earlier in the run, keyed by the proxy-minted tool-call ids (key 1) and
    by normalised content (key 2)."""

    def __init__(self) -> None:
        self.by_id: dict[str, str] = {}
        self.by_content: dict[str, str] = {}

    def insert(self, reasoning: str, *, content: str, call_ids: Sequence[str]) -> None:
        for call_id in call_ids:
            self.by_id[call_id] = reasoning
        key = content_key(content)
        if key is not None:
            self.by_content[key] = reasoning

    def lookup(self, message: Message, following_tool_ids: Sequence[str]) -> str | None:
        for call_id in [c.id for c in message.tool_calls] + list(following_tool_ids):
            if call_id in self.by_id:
                return self.by_id[call_id]
        key = content_key(message.content)
        if key is not None and key in self.by_content:
            return self.by_content[key]
        return None


@dataclass
class _RunState:
    spec: RunSpec
    record: RunRecord
    budget: TokenBudget
    root_text: str
    subagent_text: str | None
    started_monotonic: float
    effective_root_sha: str | None = None
    conversations: dict[str, str] = field(default_factory=dict)
    cache: ReasoningCache = field(default_factory=ReasoningCache)
    replies: dict[str, ChatReply] = field(default_factory=dict)      # rendered_prompt_sha256 -> reply (B9)
    root_contents: list[str] = field(default_factory=list)
    closed: bool = False
    infrastructure_stop: bool = False


class ProxyCore:
    def __init__(self, settings: DynorchSettings, upstream: Upstream, journal: CallJournal, store: RunStore) -> None:
        self.settings = settings
        self.upstream = upstream
        self.journal = journal
        self.store = store
        self._runs: dict[str, _RunState] = {}

    # ---- runs ------------------------------------------------------------------------------------------------
    @property
    def open_runs(self) -> int:
        return sum(1 for s in self._runs.values() if not s.closed)

    def open_run(self, spec: RunSpec, *, root_system_text: str, subagent_system_text: str | None) -> None:
        """§3: register a run (its record becomes ``running`` and is persisted); refuses a second open, a spec
        stamped with another proxy version, or a cap that is not ``rung * A``."""
        if spec.run_key in self._runs:
            raise ValueError(f"run {spec.run_key} is already open")
        if spec.proxy_version != self.settings.proxy_version:
            raise ValueError(f"spec proxy_version {spec.proxy_version!r} != {self.settings.proxy_version!r}")
        if spec.cap_tokens != self.settings.cap_tokens(spec.rung):
            raise ValueError(f"cap_tokens {spec.cap_tokens} != rung {spec.rung} * A {self.settings.a_tokens}")
        if spec.shard_id not in self.upstream.locks:
            raise ValueError(f"spec shard {spec.shard_id!r} has no endpoint lock")
        if not isinstance(root_system_text, str) or not root_system_text.strip():
            raise ValueError("root_system_text must be a non-empty string")
        if subagent_system_text is not None and not isinstance(subagent_system_text, str):
            raise ValueError("subagent_system_text must be a string or None")
        budget = TokenBudget(spec.cap_tokens, terminal_threshold=self.settings.terminal_threshold_tokens,
                             grace=self.settings.grace_tokens, max_calls=self.settings.max_calls_per_run,
                             max_prompt_tokens=self.settings.max_prompt_tokens_per_run)
        self.store.archive_previous(spec)
        record = new_run_record(spec, status="running", started_at=utc_now())
        self.store.save(record)
        self._runs[spec.run_key] = _RunState(spec=spec, record=record, budget=budget, root_text=root_system_text,
                                             subagent_text=subagent_system_text or None,
                                             started_monotonic=time.monotonic())

    def close_run(self, run_key: str, *, status: str) -> RunRecord:
        """§3 / B7: finalise the record (delivered answer = last non-empty root assistant content served by the
        proxy), persist it and refuse further calls (``run_closed``)."""
        state = self._runs.get(run_key)
        if state is None:
            raise ProxyError("unknown_run", f"no run {run_key}")
        if status not in TERMINAL_RUN_STATUSES:
            raise ValueError(f"close_run needs a terminal status, got {status!r}")
        record = state.record
        record.status = status
        record.final_message = state.root_contents[-1] if state.root_contents else None
        record.final_source = "proxy_root" if record.final_message is not None else "none"
        record.finished_at = utc_now()
        record.wall_seconds = time.monotonic() - state.started_monotonic
        self._refresh(state)
        state.closed = True
        self.store.save(record)
        return record

    def record_of(self, run_key: str) -> RunRecord:
        return self._state(run_key, allow_closed=True).record

    def budget_view(self, run_key: str) -> dict[str, int]:
        """§3: cap / spent / remaining / calls (+ grace and prompt totals) for diagnostics."""
        return self._state(run_key, allow_closed=True).budget.view()

    def _state(self, run_key: str, *, allow_closed: bool = False) -> _RunState:
        state = self._runs.get(run_key)
        if state is None:
            raise ProxyError("unknown_run", f"no run {run_key}")
        if state.closed and not allow_closed:
            raise ProxyError("run_closed", f"run {run_key} is closed ({state.record.status})")
        return state

    # ---- classification (B1) ---------------------------------------------------------------------------------
    @staticmethod
    def _system_and_first_user(request: ChatRequest) -> tuple[str, str]:
        if not request.messages:
            raise ProxyError("bad_request", "messages must not be empty")
        system_text = request.messages[0].content if request.messages[0].role == "system" else ""
        for message in request.messages:
            if message.role == "user":
                return system_text, message.content
        raise ProxyError("bad_request", "no user message in the conversation")

    def _classify(self, state: _RunState, request: ChatRequest) -> tuple[str, str]:
        """``(conversation_key, class)``: solver when the harness flagged a sub-agent or the system text is the
        sub-agent prompt (possibly with harness-appended text); root when it is the run's root system text (or the
        first conversation's system text -- Codex appends its own instructions); anything else is ``other``."""
        system_text, first_user = self._system_and_first_user(request)
        sha = system_sha256(system_text)
        sub = state.subagent_text
        if request.subagent or (sub is not None and system_text.startswith(sub)):
            cls = "solver"
        elif system_text.startswith(state.root_text):
            cls = "root"
        elif state.effective_root_sha is None:
            cls = "root"
        elif sha == state.effective_root_sha:
            cls = "root"
        else:
            cls = "other"
        if cls == "root" and state.effective_root_sha is None:
            state.effective_root_sha = sha
        suffix = "\x1fsubagent" if (cls == "solver" and request.subagent) else ""
        key = hashlib.sha256((system_text + "\x1f" + first_user + suffix).encode("utf-8")).hexdigest()
        state.conversations.setdefault(key, cls)
        return key, cls

    def _check_tool_surface(self, state: _RunState, request: ChatRequest) -> None:
        allowed = set(state.spec.allowed_tools)
        outside = sorted({t.name for t in request.tools} - allowed)
        if outside:
            state.record.error = f"tool_surface_violation: {','.join(outside)}"
            self.store.save(state.record)
            raise ProxyError("tool_surface_violation",
                             f"tools outside the run's allowlist {sorted(allowed)}: {outside}")

    # ---- reasoning re-injection (§4) --------------------------------------------------------------------------
    def _reinject(self, state: _RunState, messages: Sequence[Message]) -> tuple[list[Message], str, int, int]:
        out: list[Message] = []
        explicit = hits = misses = 0
        for index, message in enumerate(messages):
            if message.role != "assistant":
                out.append(message)
                continue
            if isinstance(message.reasoning_content, str) and message.reasoning_content.strip():
                explicit += 1
                out.append(message)
                continue
            following: list[str] = []
            for later in messages[index + 1:]:
                if later.role != "tool":
                    break
                if later.tool_call_id:
                    following.append(later.tool_call_id)
            found = state.cache.lookup(message, following)
            if found is not None:
                hits += 1
                out.append(replace(message, reasoning_content=found))
                continue
            if message.tool_calls or content_key(message.content) is not None:
                misses += 1                     # a proxy-shaped turn whose reasoning is lost: empty think block
            out.append(replace(message, reasoning_content=""))
        if misses:
            status = "miss"
        elif hits:
            status = "hit"
        elif explicit:
            status = "explicit"
        else:
            status = "n/a"
        return out, status, hits, misses

    # ---- the call ---------------------------------------------------------------------------------------------
    async def chat(self, run_key: str, request: ChatRequest) -> ChatReply:
        state = self._state(run_key)
        if not isinstance(request, ChatRequest):
            raise ProxyError("bad_request", "request must be a ChatRequest")
        conv_key, conv_class = self._classify(state, request)
        self._check_tool_surface(state, request)
        root = conv_class == "root"
        budget = state.budget
        waited = await budget.lock()
        grant: Grant | None = None
        try:
            messages, reinjection, hits, misses = self._reinject(state, request.messages)
            effort = template_effort(request.requested_reasoning_effort, configured=self.settings.reasoning_effort)
            prompt = render_chat(messages, tools=request.tools, reasoning_effort=effort)
            prompt_sha = rendered_sha256(prompt)
            cached = state.replies.get(prompt_sha)
            if cached is not None:
                return self._serve_duplicate(state, cached, conv_key, conv_class, request, waited)
            state.record.reinjection_hits += hits
            state.record.reinjection_misses += misses
            if budget.closed:
                self._refuse(state, "budget_exhausted", "the run's grace was used; no further calls are accepted")
            guard = budget.guard()
            if guard is not None or budget.exhausted_for(root):
                if root and not budget.grace_used:
                    return await self._terminal(state, request, conv_key, conv_class, waited, reinjection,
                                                guard_triggered=guard is not None)
                self._refuse(state, guard or "budget_exhausted",
                             f"completion-token budget exhausted (cap={budget.cap}, spent={budget.spent})"
                             if guard is None else f"cost guard {guard} reached", guard=guard)
            grant = budget.grant(root=root, queue_wait_seconds=waited, root_ceiling=state.spec.root_call_ceiling)
            return await self._served(state, request, messages, prompt, prompt_sha, grant, conv_key, conv_class,
                                      reinjection)
        finally:
            if grant is not None and not grant.settled:
                # a cancellation or an unexpected error after the grant was minted: the call may have left the
                # process, so the grant is charged whole (a provably unsent ledger refusal settles itself above)
                budget.mark_sent(grant)
                budget.settle_full(grant)
            if grant is None or not budget.locked or budget._active is None and budget.locked:
                budget.unlock()

    def _refuse(self, state: _RunState, code: str, message: str, *, guard: str | None = None) -> None:
        record = state.record
        record.refused_calls += 1
        if code == "budget_exhausted":
            record.exhausted = True
        if guard is not None:
            record.guard_triggered = True
        self.store.save(record)
        raise ProxyError(code, message)

    def _serve_duplicate(self, state: _RunState, cached: ChatReply, conv_key: str, conv_class: str,
                         request: ChatRequest, waited: float) -> ChatReply:
        """B9: an identical re-request is served from the cache with no upstream call."""
        original = cached.call
        call = replace(original, conversation_key=conv_key, conversation_class=conv_class,
                       requested_max_tokens=request.requested_max_tokens, queue_wait_seconds=waited,
                       granted_max_tokens=0, prompt_tokens=None, completion_tokens=None, reasoning_tokens=None,
                       cached_prompt_tokens=None, grace_tokens=0, cost_usd="0", generation_ids=(),
                       latency_seconds=0.0, transport_status="cache", stream=request.stream, wire=request.wire,
                       cache_status=None, duplicate_served=True, ignored_body_keys=tuple(request.ignored_keys))
        state.record.duplicates_served += 1
        state.record.calls_detail.append(call)
        self._refresh(state)
        self.store.save(state.record)
        headers = {**cached.headers, "X-Dynorch-Duplicate": "1"}
        return replace(cached, call=call, headers=headers)

    def _coordinates(self, state: _RunState, call_no: int, phase: str, conv_class: str) -> dict[str, Any]:
        spec = state.spec
        return {"run_key": spec.run_key, "node_id": spec.run_key, "partition": spec.partition, "item_id": spec.item_id,
                "benchmark": spec.benchmark, "arm": spec.arm, "harness": spec.harness, "rung": spec.rung,
                "call_no": call_no, "phase": phase, "conversation_class": conv_class}

    async def _upstream(self, state: _RunState, *, prompt: str, max_tokens: int, call_no: int, phase: str,
                        conv_class: str) -> UpstreamResult:
        aid = attempt_id(state.spec.run_key, state.spec.generation, call_no, phase)
        return await self.upstream.complete(shard_id=state.spec.shard_id, prompt=prompt, max_tokens=max_tokens,
                                            attempt_id=aid,
                                            coordinates=self._coordinates(state, call_no, phase, conv_class))

    async def _served(self, state: _RunState, request: ChatRequest, messages: list[Message], prompt: str,
                      prompt_sha: str, grant: Grant, conv_key: str, conv_class: str, reinjection: str) -> ChatReply:
        budget = state.budget
        spec = state.spec
        root = conv_class == "root"
        call_no = grant.call_no
        phases: list[dict[str, Any]] = []
        cost = Decimal("0")
        try:
            think = await self.upstream.complete(
                shard_id=spec.shard_id, prompt=prompt, max_tokens=grant.max_tokens,
                attempt_id=attempt_id(spec.run_key, spec.generation, call_no, PHASE_THINK),
                coordinates=self._coordinates(state, call_no, PHASE_THINK, conv_class))
        except UpstreamFailed as exc:
            budget.mark_sent(grant)
            budget.settle_full(grant)
            self._journal_failure(state, request, prompt_sha, len(prompt), grant, conv_key, conv_class,
                                  reinjection, exc.phases, exc)
            raise
        except ProxyError as exc:
            if exc.code == "ledger_refused":
                budget.release_unsent(grant)          # provably nothing left the process
                state.record.error = f"ledger_refused: {exc.detail}"
                self.store.save(state.record)
            raise
        except BaseException:
            budget.mark_sent(grant)                   # cancelled mid-flight: the call may have been sent
            budget.settle_full(grant)
            raise
        budget.mark_sent(grant)
        phases.extend(think.phases)
        cost += Decimal(think.cost_usd)
        usage = think.usage
        assert usage is not None and usage.completion_tokens is not None
        think_tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens or 0
        reasoning, visible, closed = think.reasoning, think.text, think.closed
        upstream_finish = think.finish_reason
        cut = (upstream_finish or "").strip().lower() != "stop"     # cap / ceiling / provider cut
        kind = "none"
        grace_tokens = 0
        extra_inside_c = 0
        answer_usage: Usage | None = None
        generation_ids = [g for g in (think.generation_id,) if g]
        if cut and root:
            remaining_after = max(0, budget.cap - grant.spent_before - think_tokens)
            if grant.ceiling_bound and remaining_after >= 1:
                kind = "ceiling_close"
                cont_max = min(self.settings.grace_tokens, remaining_after)
            elif not budget.grace_used:
                kind = "mid_call" if not closed else "visible_continuation"
                cont_max = self.settings.grace_tokens
            else:
                kind, cont_max = "truncated_subagent", 0
            if cont_max > 0:
                cont_prompt = answer_prompt(prompt, reasoning, self.settings.think_close_text)
                if closed:
                    cont_prompt += visible
                try:
                    answer = await self._upstream(state, prompt=cont_prompt, max_tokens=cont_max, call_no=call_no,
                                                  phase=PHASE_ANSWER, conv_class=conv_class)
                except BaseException as exc:
                    # the continuation may have been sent: charge it whole (inside C for a ceiling close, as the
                    # spent grace otherwise) before re-raising; an UpstreamFailed is also journaled
                    if kind == "ceiling_close":
                        budget.settle(grant, min(grant.max_tokens, think_tokens + cont_max), prompt_tokens)
                    else:
                        budget.settle(grant, think_tokens, prompt_tokens)
                        budget.record_grace(kind, cont_max)      # conservative: the grace is spent
                    if isinstance(exc, UpstreamFailed):
                        phases.extend(exc.phases)
                        cost += exc.cost_usd
                        self._journal_failure(state, request, prompt_sha, len(prompt), grant, conv_key,
                                              conv_class, reinjection, tuple(phases), exc, cost=cost)
                    raise
                phases.extend(answer.phases)
                cost += Decimal(answer.cost_usd)
                answer_usage = answer.usage
                assert answer_usage is not None and answer_usage.completion_tokens is not None
                answer_text = answer_visible_text(answer.reasoning, answer.text)
                visible = (visible + answer_text) if closed else answer_text
                closed = True
                if answer.generation_id:
                    generation_ids.append(answer.generation_id)
                prompt_tokens += answer_usage.prompt_tokens or 0
                if kind == "ceiling_close":
                    extra_inside_c = answer_usage.completion_tokens
                else:
                    grace_tokens = answer_usage.completion_tokens
        elif cut and not root:
            kind = "truncated_subagent"
        if not closed:
            visible = ""                                     # thinking never closed: nothing visible was produced
        # ---- settle tokens: THINK (+ a ceiling continuation) inside C, the grace outside ----
        budget.settle(grant, think_tokens + extra_inside_c, prompt_tokens)
        if grace_tokens or kind in ("mid_call", "visible_continuation"):
            budget.record_grace(kind, grace_tokens)
        # ---- parse, cache, reply ----
        parse = (parse_tool_calls(visible, request.tools, call_no=call_no) if request.tools
                 else ToolParse(content=visible.strip(), calls=(), status="none"))
        finish = "tool_calls" if parse.calls else "stop"
        state.cache.insert(reasoning, content=parse.content, call_ids=[c.id for c in parse.calls])
        spawn_calls = sum(1 for c in parse.calls if c.name in spec.spawn_tools) if root else 0
        if root and parse.content.strip():
            state.root_contents.append(parse.content)
        reasoning_tokens = usage.reasoning_tokens
        cached_prompt = usage.cached_prompt_tokens
        if answer_usage is not None:
            if reasoning_tokens is not None and answer_usage.reasoning_tokens is not None:
                reasoning_tokens += answer_usage.reasoning_tokens
            if cached_prompt is not None and answer_usage.cached_prompt_tokens is not None:
                cached_prompt += answer_usage.cached_prompt_tokens
        completion_reported = think_tokens + extra_inside_c + grace_tokens
        record = CallRecord(
            call_no=call_no, conversation_key=conv_key, conversation_class=conv_class,
            requested_max_tokens=request.requested_max_tokens, granted_max_tokens=grant.max_tokens,
            queue_wait_seconds=grant.queue_wait_seconds, prompt_tokens=prompt_tokens,
            completion_tokens=think_tokens + extra_inside_c, reasoning_tokens=reasoning_tokens,
            cached_prompt_tokens=cached_prompt, reasoning_chars=len(reasoning), visible_chars=len(visible),
            forced_close_kind=kind, grace_tokens=grace_tokens, finish_reason_upstream=upstream_finish,
            finish_reason_returned=finish, tool_calls=len(parse.calls), spawn_calls=spawn_calls,
            tool_parse=parse.status, reinjection=reinjection,
            reasoning_effort_requested=requested_effort(request.requested_reasoning_effort), cost_usd=str(cost),
            generation_ids=tuple(generation_ids), latency_seconds=sum(p["latency_seconds"] for p in phases),
            transport_status=think.transport_status.value, returned_status=200, stream=request.stream,
            wire=request.wire, cache_status=think.cache_status, duplicate_served=False,
            ignored_body_keys=tuple(request.ignored_keys))
        reply = ChatReply(
            content=parse.content, reasoning_content=reasoning, tool_calls=parse.calls, finish_reason=finish,
            usage=ReplyUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_reported,
                             reasoning_tokens=reasoning_tokens, cached_prompt_tokens=cached_prompt),
            call=record, headers=self._headers(state, record))
        self._journal(state, request, prompt_sha, len(prompt), phases, reply, parse, grant, record)
        self._account(state, record, parse, spawn_calls, cost, kind)
        state.replies[prompt_sha] = reply
        self.store.save(state.record)
        return reply

    async def _terminal(self, state: _RunState, request: ChatRequest, conv_key: str, conv_class: str, waited: float,
                        reinjection: str, *, guard_triggered: bool) -> ChatReply:
        """B4: the one TERMINAL call of the root conversation (the harness's messages plus the verbatim terminal
        user turn, thinking disabled, ``max_tokens = G``, no tools); the whole reply is grace (outside C)."""
        budget = state.budget
        spec = state.spec
        messages = list(request.messages) + [Message(role="user", content=self.settings.terminal_sentence)]
        prompt = render_chat(messages, tools=(), reasoning_effort=self.settings.reasoning_effort,
                             enable_thinking=False)
        prompt_sha = rendered_sha256(prompt)
        call_no = budget.calls + 1
        budget.calls = call_no
        grant = Grant(call_no=call_no, max_tokens=self.settings.grace_tokens, spent_before=budget.spent,
                      queue_wait_seconds=waited, root=True, sent=True, settled=True)
        if guard_triggered:
            state.record.guard_triggered = True
        try:
            answer = await self._upstream(state, prompt=prompt, max_tokens=self.settings.grace_tokens,
                                          call_no=call_no, phase=PHASE_TERMINAL, conv_class=conv_class)
        except UpstreamFailed as exc:
            budget.record_grace("terminal", self.settings.grace_tokens)   # conservative: the grace is spent
            self._journal_failure(state, request, prompt_sha, len(prompt), grant, conv_key, conv_class,
                                  reinjection, exc.phases, exc)
            raise
        except ProxyError as exc:
            if exc.code == "ledger_refused":
                budget.calls -= 1
                state.record.error = f"ledger_refused: {exc.detail}"
                self.store.save(state.record)
            raise
        except BaseException:
            budget.record_grace("terminal", self.settings.grace_tokens)   # cancelled mid-flight: grace spent
            raise
        usage = answer.usage
        assert usage is not None and usage.completion_tokens is not None
        tokens = usage.completion_tokens
        prompt_tokens = usage.prompt_tokens or 0
        budget.prompt_total += prompt_tokens
        budget.record_grace("terminal", tokens)
        visible = answer_visible_text(answer.reasoning, answer.text)
        parse = (parse_tool_calls(visible, request.tools, call_no=call_no) if request.tools
                 else ToolParse(content=visible.strip(), calls=(), status="none"))
        finish = "tool_calls" if parse.calls else "stop"
        state.cache.insert(answer.reasoning, content=parse.content, call_ids=[c.id for c in parse.calls])
        if parse.content.strip():
            state.root_contents.append(parse.content)
        cost = Decimal(answer.cost_usd)
        record = CallRecord(
            call_no=call_no, conversation_key=conv_key, conversation_class=conv_class,
            requested_max_tokens=request.requested_max_tokens, granted_max_tokens=self.settings.grace_tokens,
            queue_wait_seconds=waited, prompt_tokens=prompt_tokens, completion_tokens=0,
            reasoning_tokens=usage.reasoning_tokens, cached_prompt_tokens=usage.cached_prompt_tokens,
            reasoning_chars=len(answer.reasoning), visible_chars=len(visible), forced_close_kind="terminal",
            grace_tokens=tokens, finish_reason_upstream=answer.finish_reason, finish_reason_returned=finish,
            tool_calls=len(parse.calls), spawn_calls=0, tool_parse=parse.status, reinjection=reinjection,
            reasoning_effort_requested=requested_effort(request.requested_reasoning_effort), cost_usd=str(cost),
            generation_ids=tuple(g for g in (answer.generation_id,) if g), latency_seconds=answer.latency_seconds,
            transport_status=answer.transport_status.value, returned_status=200, stream=request.stream,
            wire=request.wire, cache_status=answer.cache_status, duplicate_served=False,
            ignored_body_keys=tuple(request.ignored_keys))
        reply = ChatReply(
            content=parse.content, reasoning_content="", tool_calls=parse.calls, finish_reason=finish,
            usage=ReplyUsage(prompt_tokens=prompt_tokens, completion_tokens=tokens,
                             reasoning_tokens=usage.reasoning_tokens, cached_prompt_tokens=usage.cached_prompt_tokens),
            call=record, headers=self._headers(state, record))
        self._journal(state, request, prompt_sha, len(prompt), list(answer.phases), reply, parse, grant, record)
        self._account(state, record, parse, 0, cost, "terminal")
        state.record.exhausted = True
        state.replies[prompt_sha] = reply
        self.store.save(state.record)
        return reply

    # ---- bookkeeping ------------------------------------------------------------------------------------------
    def _headers(self, state: _RunState, record: CallRecord) -> dict[str, str]:
        budget = state.budget
        return {
            "X-Dynorch-Cap-Tokens": str(budget.cap),
            "X-Dynorch-Spent-Tokens": str(budget.spent),
            "X-Dynorch-Remaining-Tokens": str(budget.remaining),
            "X-Dynorch-Call-No": str(record.call_no),
            "X-Dynorch-Forced-Close": record.forced_close_kind,
            "X-Dynorch-Grace-Tokens": str(record.grace_tokens),
            "X-Dynorch-Tool-Parse": record.tool_parse,
            "X-Dynorch-Conversation": record.conversation_class,
            "X-Dynorch-Proxy-Version": PROXY_VERSION,
        }

    def _account(self, state: _RunState, record: CallRecord, parse: ToolParse, spawn_calls: int, cost: Decimal,
                 kind: str) -> None:
        run = state.record
        run.calls += 1
        run.calls_by_class[record.conversation_class] = run.calls_by_class.get(record.conversation_class, 0) + 1
        run.spawns += spawn_calls
        if parse.status == "failed":
            run.tool_parse_failures += 1
        if kind != "none" and kind != "truncated_subagent" and kind != "ceiling_close":
            run.forced_close_kind = kind
        run.cost_usd = str(Decimal(run.cost_usd) + cost)
        run.calls_detail.append(record)
        self._refresh(state)

    def _refresh(self, state: _RunState) -> None:
        run, budget = state.record, state.budget
        run.completion_total = budget.spent
        run.prompt_total = budget.prompt_total
        run.grace_tokens = budget.grace_tokens
        run.overdraft_tokens = budget.overdraft_tokens
        run.conversations = len(state.conversations)
        reasoning = [c.reasoning_tokens for c in run.calls_detail if not c.duplicate_served]
        run.reasoning_total = None if any(r is None for r in reasoning) else sum(reasoning)
        if budget.grace_used:
            run.exhausted = True

    def _journal(self, state: _RunState, request: ChatRequest, prompt_sha: str, rendered_chars: int,
                 phases: Sequence[Mapping[str, Any]], reply: ChatReply, parse: ToolParse, grant: Grant,
                 record: CallRecord) -> None:
        payload = {
            "layout": JOURNAL_LAYOUT, "run_key": state.spec.run_key, "call_no": record.call_no,
            "phases": list(phases), "harness_request": to_json(request), "rendered_prompt_sha256": prompt_sha,
            "rendered_chars": rendered_chars,
            "reply": {"content": reply.content, "reasoning_content": reply.reasoning_content,
                      "tool_calls": [to_json(c) for c in reply.tool_calls], "finish_reason": reply.finish_reason,
                      "usage": to_json(reply.usage), "headers": dict(reply.headers)},
            "tool_parse": {"status": parse.status, "reason": parse.reason, "calls": len(parse.calls)},
            "budget": {"granted": grant.max_tokens, "realized": record.completion_tokens + record.grace_tokens,
                       "remaining_after": state.budget.remaining},
            "call": to_json(record), "proxy_version": PROXY_VERSION,
        }
        self.journal.write(state.spec, record.call_no, payload)

    def _journal_failure(self, state: _RunState, request: ChatRequest, prompt_sha: str, rendered_chars: int,
                         grant: Grant, conv_key: str, conv_class: str, reinjection: str,
                         phases: Sequence[Mapping[str, Any]], exc: UpstreamFailed, *,
                         cost: Decimal | None = None) -> None:
        total = exc.cost_usd if cost is None else cost
        record = CallRecord(
            call_no=grant.call_no, conversation_key=conv_key, conversation_class=conv_class,
            requested_max_tokens=request.requested_max_tokens, granted_max_tokens=grant.max_tokens,
            queue_wait_seconds=grant.queue_wait_seconds, prompt_tokens=None,
            completion_tokens=grant.realized_completion if grant.realized_completion is not None else grant.max_tokens,
            reasoning_tokens=None, cached_prompt_tokens=None, reasoning_chars=0, visible_chars=0,
            forced_close_kind="none", grace_tokens=0, finish_reason_upstream=None, finish_reason_returned="stop",
            tool_calls=0, spawn_calls=0, tool_parse="none", reinjection=reinjection,
            reasoning_effort_requested=requested_effort(request.requested_reasoning_effort), cost_usd=str(total),
            generation_ids=tuple(p["generation_id"] for p in phases if p.get("generation_id")),
            latency_seconds=sum(p["latency_seconds"] for p in phases), transport_status=exc.transport_status.value,
            returned_status=exc.status, stream=request.stream, wire=request.wire, cache_status=None,
            duplicate_served=False, ignored_body_keys=tuple(request.ignored_keys))
        payload = {
            "layout": JOURNAL_LAYOUT, "run_key": state.spec.run_key, "call_no": grant.call_no,
            "phases": list(phases), "harness_request": to_json(request), "rendered_prompt_sha256": prompt_sha,
            "rendered_chars": rendered_chars, "reply": None,
            "tool_parse": {"status": "none", "reason": None, "calls": 0},
            "budget": {"granted": grant.max_tokens, "realized": record.completion_tokens,
                       "remaining_after": state.budget.remaining},
            "call": to_json(record), "error": {"code": exc.code, "message": exc.message, "detail": exc.detail,
                                               "charged_max": exc.charged_max},
            "proxy_version": PROXY_VERSION,
        }
        self.journal.write(state.spec, grant.call_no, payload)
        run = state.record
        run.calls += 1
        run.calls_by_class[conv_class] = run.calls_by_class.get(conv_class, 0) + 1
        run.cost_usd = str(Decimal(run.cost_usd) + total)
        run.calls_detail.append(record)
        run.error = f"upstream_failed: {exc.detail}"
        if exc.infrastructure_stop:
            state.infrastructure_stop = True
            run.error = f"infrastructure_stop: {exc.detail}"
        self._refresh(state)
        self.store.save(run)
