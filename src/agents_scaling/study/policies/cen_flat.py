"""``CEN_FLAT``: one coordinator, up to N−1 workers, at most eight delegation cycles (WP4).

Spec §4.2 (``CEN_FLAT`` row and the two coordinator paragraphs: strict ``coordinator_action``
— ``final`` with a complete candidate or ``delegate`` with 1..N−1 assignments to distinct
worker slots with task-public handles; more assignments than slots, duplicate slots,
hidden references and invalid objects are typed errors; workers see the task, their
assignment and any prior result the hub forwarded, return a typed subtask object, never a
candidate; a completed batch returns in planned order; the next hub call sees its previous
valid plan, the returned objects and the task; "A bounded no-more-work instruction
consumes a reserved final hub call if the last action was delegation"; the coordinator's
own candidate is the episode output), §4.1 (workers are told their results inform the
coordinator), §4.4 (native CEN finalization is synthesis, no VOTE), §6.5 (step 1: reserve
"all mandatory work those outputs can require, while retaining the final-answer reserve"
→ hub + (N−1) worker envelopes + the reserved final hub call before every hub call that
may delegate), §4.3 / amendment B2 (Table E: subtask results ≤ 4,096 recipient tokens,
prior plan ≤ 8,192, forwarded results ≤ 8,192 total) and amendment B2b (the hub's returned
results block ≤ ``caps.hub_returned_results_tokens`` in total: a k-assignment cycle clips
every returned result to ``min(subtask_result_tokens, block // k)``, so no hub prompt can
exceed 32,768 tokens at any N ≤ 9 — review P0-A, critic P1-1).  Architecture §1.11; audit
A-6 and fixtures T6/T7/T10; corrections P1-3 (N=1 → ``N_WORKERS=0``, hub-only) and §4 item 2.

Typed action errors consume the hub call and its cycle (T7).  The next hub call re-renders
the last valid state (plan + returned results) together with the typed error object of the
invalid action as the ``last_action_error`` field (audit A-6, review P1-B); the object is
clipped under ``caps.hub_action_error_tokens`` by the frozen code-point prefix rule and is
dropped again after the next valid action.  Every error is also recorded in the episode
(``hub_action_errors``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from agents_scaling.study import identity
from agents_scaling.study.inference.tokens import count_tokens
from agents_scaling.study.packets import largest_prefix, make_counter
from agents_scaling.study.policies.base import (
    INVALID_CANDIDATE_SHA256,
    AdmissionRefused,
    CallResult,
    EpisodeContext,
    action_error,
    subtask_error,
)
from agents_scaling.study.prompts import render as R
from agents_scaling.study.resources.broker import StopReason
from agents_scaling.study.resources.envelopes import TableE, measure_wrappers
from agents_scaling.study.types import (
    NS_CEN_FLAT,
    PURPOSE_HUB,
    PURPOSE_WORKER,
    SOLVER_DECODING,
    SOLVER_OUT_CAP,
    CandidateRecord,
    DelegateAction,
    EpisodeResult,
    FinalAction,
    Method,
    Packet,
    ProtocolError,
    SubtaskResult,
)

TASK_HANDLE = "task"
#: Code points of parser detail kept in the episode record and offered to the re-prompt clipper.
ACTION_ERROR_DETAIL_CHARS = 500
#: Tokens of the ``last_action_error`` field wrapper (label line, newlines, tokenizer boundary
#: effects) reserved out of ``caps.hub_action_error_tokens`` before the object is clipped.
ACTION_ERROR_FIELD_OVERHEAD = 64


def bounded_action_error(cycle: int, code: str, detail: str, count: Callable[[str], int], cap: int) -> dict[str, Any]:
    """The typed error object a hub is re-prompted with (A-6), its ``detail`` cut by the frozen
    code-point prefix rule so the rendered compact JSON is ≤ ``cap − ACTION_ERROR_FIELD_OVERHEAD``
    tokens under ``count`` (the field then fits the Table E allowance; amendment B2b).

    Raises :class:`ProtocolError` when even an empty detail does not fit (a caps defect).
    """
    budget = int(cap) - ACTION_ERROR_FIELD_OVERHEAD
    if budget < 1:
        raise ProtocolError(f"hub_action_error_tokens={cap} leaves no room for an error object")
    if not isinstance(detail, str):
        raise TypeError("detail must be a str")

    def serialized(prefix: str) -> str:
        return R.action_error_text({"cycle": cycle, "code": code, "detail": prefix})

    if count(serialized("")) > budget:
        raise ProtocolError(f"action error {code!r} does not fit {budget} tokens even without detail")
    end = largest_prefix(detail, budget, lambda prefix: count(serialized(prefix)))
    return {"cycle": cycle, "code": code, "detail": detail[:end]}


@dataclass
class CenFlatPolicy:
    id: ClassVar[Method] = Method.CEN_FLAT

    # ---- helpers -------------------------------------------------------------------------
    @staticmethod
    def _final_record(ctx: EpisodeContext, hub: CallResult, action: Any) -> CandidateRecord:
        request_id = hub.spec.request_id
        raw = identity.sha256_hex(hub.content or "")
        if isinstance(action, FinalAction):
            return CandidateRecord(
                candidate_id=identity.sha256_hex(request_id),
                request_id=request_id,
                slot=0,
                stage="final",
                valid=True,
                failure_code=None,
                candidate=action.candidate,
                candidate_sha256=ctx.c.candidate_sha256(action.candidate),
                raw_content_sha256=raw,
            )
        error = action_error(action)
        code = error[0] if error else "NOT_FINAL"
        return CandidateRecord(
            candidate_id=identity.sha256_hex(request_id),
            request_id=request_id,
            slot=0,
            stage="final",
            valid=False,
            failure_code=code,
            candidate=None,
            candidate_sha256=INVALID_CANDIDATE_SHA256,
            raw_content_sha256=raw,
        )

    def _forwarded(self, ctx: EpisodeContext, assignment, by_handle: dict[str, Packet]) -> list[Packet]:
        """Prior results the hub forwards to a worker, in returned order, clipped to the
        Table E total (later ones become typed unavailable packets, recorded)."""
        cap = int(ctx.cfg.caps.forwarded_results_tokens)
        wanted = [h for h in assignment.source_handles if h != TASK_HANDLE]
        out: list[Packet] = []
        used = 0
        for handle in wanted:
            packet = by_handle[handle]  # validated ⊆ allowed handles by the parser
            if used + packet.recipient_tokens > cap:
                out.append(ctx.subtask_packet(None, packet.sender_slot, recipient_slot=assignment.worker_slot, cycle=-1))
                continue
            used += packet.recipient_tokens
            out.append(packet)
        return out

    @staticmethod
    def returned_result_cap(caps: Any, k: int) -> int:
        """Per-result recipient-token cap for a ``k``-assignment cycle (amendment B2b):
        ``min(subtask_result_tokens, hub_returned_results_tokens // k)``."""
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive int, got {k!r}")
        cap = min(int(caps.subtask_result_tokens), int(caps.hub_returned_results_tokens) // k)
        if cap < 1:
            raise ProtocolError(f"hub_returned_results_tokens={caps.hub_returned_results_tokens} cannot hold {k} returned results")
        return cap

    # ---- episode -------------------------------------------------------------------------
    def run(self, ctx: EpisodeContext) -> EpisodeResult:
        N = ctx.N
        if N < 1:
            raise ProtocolError("CEN_FLAT needs N >= 1")
        W = N - 1
        caps = ctx.cfg.caps
        max_cycles = int(caps.cen_max_cycles)
        oracle = ctx.oracle
        block_cap = int(caps.hub_returned_results_tokens)
        error_cap = int(caps.hub_action_error_tokens)
        counter = make_counter(ctx.tokenizer)
        # Runtime Table E with this item's actual task tokens (§6.5 "known prompt").
        task_tokens = count_tokens(ctx.task.task_text, ctx.tokenizer)
        table = TableE(caps, measure_wrappers(ctx.tokenizer, ctx.task, Ns=(N,)), task_tokens=task_tokens)
        final_env = table.hub_prompt_max(N, cycle0=False)
        ctx.ledger.set_final_reserve(oracle.reservation(final_env, SOLVER_OUT_CAP), 1)

        prior_plan: DelegateAction | None = None
        returned: list[Packet] = []
        by_handle: dict[str, Packet] = {}
        allowed: list[str] = [TASK_HANDLE]
        errors: list[dict[str, Any]] = []
        last_error: dict[str, Any] | None = None
        cycles: list[dict[str, Any]] = []
        worker_slots_used: set[int] = set()
        final_record: CandidateRecord | None = None
        stop: StopReason | None = None
        hub_calls = 0
        error_reprompts = 0
        for c in range(max_cycles):
            render_cycle = 0 if prior_plan is None else c
            worker_env = table.worker_prompt_max(cycle0=(c == 0))
            messages = R.render_hub(ctx.task, N, W, prior_plan, returned, render_cycle, action_error=last_error)
            hub_spec = ctx.spec(messages, SOLVER_DECODING, ctx.seed(0, PURPOSE_HUB, c, NS_CEN_FLAT), "hub")
            try:
                [hub] = ctx.generate_group(
                    [hub_spec],
                    "hub",
                    owner=f"cycle{c}.hub",
                    extra_reserve=[oracle.reservation(worker_env, SOLVER_OUT_CAP)] * W,
                    actor_slots=[0],
                    steps=[c],
                )
            except AdmissionRefused as exc:
                stop = exc.reason
                break
            hub_calls += 1
            error_reprompts += int(last_error is not None)
            action = ctx.parse_action(hub, N, allowed)
            error = action_error(action)
            if error is not None:
                detail = error[1][:ACTION_ERROR_DETAIL_CHARS]
                errors.append({"cycle": c, "code": error[0], "detail": detail, "request_id": hub.spec.request_id})
                last_error = bounded_action_error(c, error[0], detail, counter, error_cap)
                cycles.append({"cycle": c, "action": "error", "code": error[0]})
                continue
            last_error = None
            if isinstance(action, FinalAction):
                final_record = self._final_record(ctx, hub, action)
                cycles.append({"cycle": c, "action": "final"})
                stop = StopReason.VOLUNTARY_FINISH
                break
            assert isinstance(action, DelegateAction)
            assignments = list(action.assignments)
            per_result_cap = self.returned_result_cap(caps, len(assignments))
            worker_specs = []
            for a in assignments:
                forwarded = self._forwarded(ctx, a, by_handle)
                worker_specs.append(
                    ctx.spec(R.render_worker(ctx.task, a, forwarded), SOLVER_DECODING, ctx.seed(a.worker_slot, PURPOSE_WORKER, c, NS_CEN_FLAT), "worker")
                )
            try:
                workers = ctx.generate_group(
                    worker_specs,
                    "worker",
                    owner=f"cycle{c}.workers",
                    actor_slots=[a.worker_slot for a in assignments],
                    steps=[c] * len(assignments),
                )
            except AdmissionRefused as exc:
                raise ProtocolError(
                    f"cycle {c}: {len(assignments)} planned workers do not fit although {W} envelopes were reserved ({exc.reason.value})"
                ) from exc
            returned = []
            statuses: list[str] = []
            for a, w in zip(assignments, workers):
                worker_slots_used.add(a.worker_slot)
                parsed = ctx.parse_subtask(w)
                failure = subtask_error(parsed)
                result: SubtaskResult | None = None if failure is not None else parsed
                packet = ctx.subtask_packet(result, a.worker_slot, recipient_slot=0, cycle=c, cap=per_result_cap)
                returned.append(packet)
                by_handle[a.subtask_id] = packet
                if a.subtask_id not in allowed:
                    allowed.append(a.subtask_id)
                statuses.append(failure[0] if failure is not None else result.status)
            returned_tokens = sum(p.recipient_tokens for p in returned)
            if returned_tokens > block_cap:
                raise ProtocolError(f"cycle {c}: returned-results block has {returned_tokens} tokens > {block_cap} (amendment B2b)")
            prior_plan = action
            cycles.append(
                {
                    "cycle": c,
                    "action": "delegate",
                    "workers": [a.worker_slot for a in assignments],
                    "statuses": statuses,
                    "returned_result_cap": per_result_cap,
                    "returned_tokens": returned_tokens,
                }
            )
        final_failure: str | None = None
        if final_record is None:
            if stop is None:
                stop = StopReason.CYCLE_CAP
            render_cycle = 0 if prior_plan is None else max_cycles
            messages = R.render_hub(ctx.task, N, W, prior_plan, returned, render_cycle, final=True, action_error=last_error)
            spec = ctx.spec(messages, SOLVER_DECODING, ctx.seed(0, PURPOSE_HUB, max_cycles, NS_CEN_FLAT), "hub")
            ctx.ledger.release_final()
            try:
                [fin] = ctx.generate_group([spec], "hub", owner="final", keep_final=False, actor_slots=[0], steps=[max_cycles])
            except AdmissionRefused as exc:
                raise ProtocolError(f"the reserved final hub call does not fit ({exc.reason.value}): broken final reserve") from exc
            hub_calls += 1
            error_reprompts += int(last_error is not None)
            if fin.record is None:
                final_failure = fin.context_failure
                stop = StopReason.CONTEXT_FAILURE
            action = ctx.parse_action(fin, N, allowed)
            final_record = self._final_record(ctx, fin, action)
            error = action_error(action)
            if error is not None:
                errors.append({"cycle": "final", "code": error[0], "detail": error[1][:ACTION_ERROR_DETAIL_CHARS], "request_id": fin.spec.request_id})
            cycles.append({"cycle": "final", "action": "final" if final_record.valid else "error"})
        return ctx.finish(
            candidates=[final_record],
            selection=None,
            native_final=final_record,
            assigned_roster=N,
            stop_reason=stop,
            counters_extra={
                "coordinators": 1,
                "workers_assigned": W,
                "realized_participation": len(worker_slots_used),
                "worker_slots_used": sorted(worker_slots_used),
                "delegation_cycles": sum(1 for cy in cycles if cy["action"] == "delegate"),
                "hub_calls": hub_calls,
                "hub_action_errors": len(errors),
                "hub_error_reprompts": error_reprompts,
            },
            episode_extra={
                "cycles": cycles,
                "hub_action_errors": errors,
                "final_context_failure": final_failure,
                "envelopes": table.to_dict(Ns=(N,)),
            },
        )


__all__ = ["ACTION_ERROR_DETAIL_CHARS", "ACTION_ERROR_FIELD_OVERHEAD", "CenFlatPolicy", "TASK_HANDLE", "bounded_action_error"]
