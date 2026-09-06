"""``IND_VOTE``: N independent roots, then blind round-robin fresh draws (WP4).

Spec §4.2 (``IND_VOTE`` row: N slots generate initial independent answers as one group;
optional work cycles through slots in blind round-robin order with a fresh context each
time; "initialize all N slots before optional attempts. Once initialized, an optional
fresh attempt is one indivisible allocation"; stateless keys ``actor_slot=0``,
``step_slot=draw ordinal`` — the assigned slot is metadata), §5.4/§6.2 (the ``11`` cell's
draws 0..9 alias F11; the N-panel ``00`` cells alias F00 then the S_FRESH-00 sequence),
§6.7 / audit A-3 (optional draws pre-generated in batches, admission from earlier debits).
Architecture §1.11, §2.3; correction P0-1 (framing comes from the cell manifest).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from agents_scaling.study.policies.base import AdmissionRefused, EpisodeContext
from agents_scaling.study.prompts import render as R
from agents_scaling.study.resources.broker import StopReason
from agents_scaling.study.types import (
    NS_STATELESS_BANK,
    PURPOSE_ROOT,
    SOLVER_DECODING,
    EpisodeResult,
    Framing,
    Method,
    ProtocolError,
)


@dataclass
class IndVotePolicy:
    batch_size: int | None = None

    id: ClassVar[Method] = Method.IND_VOTE

    def run(self, ctx: EpisodeContext) -> EpisodeResult:
        framing = Framing(ctx.cell.framing)
        if framing is Framing.NATIVE:
            raise ProtocolError("IND_VOTE roots use a factorial framing cell (11 in module A, 00 in the N panel), never NATIVE")
        N = ctx.N
        if N < 1:
            raise ProtocolError("IND_VOTE needs N >= 1")
        ctx.ledger.set_final_reserve(0, 0)
        cap = int(ctx.cfg.caps.solver_calls)
        if N > cap:
            raise ProtocolError(f"N={N} exceeds the {cap}-call cap")
        rendered = R.render_root(ctx.task, framing)
        specs = [ctx.spec(rendered, SOLVER_DECODING, ctx.seed(0, PURPOSE_ROOT, k, NS_STATELESS_BANK), "root") for k in range(cap)]
        slots = [k % N for k in range(cap)]
        try:
            roots = ctx.generate_group(specs[:N], "root", owner="roots", actor_slots=slots[:N], steps=list(range(N)))
        except AdmissionRefused as exc:
            raise ProtocolError(
                f"B={ctx.ledger.B} cannot admit the N={N} initialization ({exc.reason.value}): configuration error (§6.5)"
            ) from exc
        optional, refusal = ctx.generate_prefix(
            specs[N:], "root", owner="draw", actor_slots=slots[N:], steps=list(range(N, cap)), batch_size=self.batch_size
        )
        stop = refusal if refusal is not None else StopReason.CALL_CAP
        candidates = [ctx.candidate_record(r, slot=r.actor_slot, stage="root") for r in roots + optional]
        selection, candidates = ctx.vote(candidates, "archive")
        selected = {c.candidate_id: c for c in candidates}.get(selection.get("selected_candidate_id"))
        return ctx.finish(
            candidates=candidates,
            selection=selection,
            native_final=selected,
            assigned_roster=N,
            stop_reason=stop,
            counters_extra={"optional_draws": len(optional)},
        )


__all__ = ["IndVotePolicy"]
