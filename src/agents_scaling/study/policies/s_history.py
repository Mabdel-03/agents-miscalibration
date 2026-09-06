"""``S_HISTORY``: one solver revising from the task plus its previous saved candidate (WP4).

Spec §4.2 (``S_HISTORY`` row: one neutral root; successive calls revise from the task plus
the *previous valid saved candidate or its failure sentinel*; earlier hidden reasoning is
not retained; "retain only the immediately preceding candidate in the next model context;
retain every complete candidate in the selector archive"; VOTE over the archive), §3.5
(sentinel bytes), §4.3 (Table E: an own candidate above 8,192 recipient tokens →
context failure for that opportunity), §6.5 (each call one reservation; history prefill
is charged).  Architecture §1.11 (draw 0 = the S_FRESH draw 0 key → aliases F00[0]).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from agents_scaling.study.policies.base import AdmissionRefused, EpisodeContext
from agents_scaling.study.prompts import render as R
from agents_scaling.study.resources.broker import StopReason
from agents_scaling.study.types import (
    NS_S_HISTORY,
    NS_STATELESS_BANK,
    PURPOSE_REVISE,
    PURPOSE_ROOT,
    SOLVER_DECODING,
    ContextFailure,
    EpisodeResult,
    Framing,
    Method,
    ProtocolError,
)


@dataclass
class SHistoryPolicy:
    id: ClassVar[Method] = Method.S_HISTORY

    def run(self, ctx: EpisodeContext) -> EpisodeResult:
        if Framing(ctx.cell.framing) is not Framing.F00:
            raise ProtocolError(f"S_HISTORY requires the neutral 00 root, cell {ctx.cell.cell_id} has {ctx.cell.framing.value}")
        ctx.ledger.set_final_reserve(0, 0)
        cap = int(ctx.cfg.caps.solver_calls)
        root_spec = ctx.spec(R.render_root(ctx.task, Framing.F00), SOLVER_DECODING, ctx.seed(0, PURPOSE_ROOT, 0, NS_STATELESS_BANK), "root")
        try:
            [root] = ctx.generate_group([root_spec], "root", owner="draw[0]", actor_slots=[0], steps=[0])
        except AdmissionRefused as exc:
            raise ProtocolError(f"B={ctx.ledger.B} cannot admit the root draw ({exc.reason.value}): configuration error (§6.5)") from exc
        archive = [ctx.candidate_record(root, slot=0, stage="root")]
        stop: StopReason | None = None
        for j in range(1, cap):
            previous = archive[-1]  # the immediately preceding candidate (sentinel when invalid)
            forced: str | None = None
            try:
                ctx.check_own_state(previous)
            except ContextFailure as exc:
                forced = str(exc)
            spec = ctx.spec(
                R.render_s_history(ctx.task, previous), SOLVER_DECODING, ctx.seed(0, PURPOSE_REVISE, j, NS_S_HISTORY), "revise"
            )
            try:
                [result] = ctx.generate_group([spec], "revise", owner=f"revise[{j}]", actor_slots=[0], steps=[j], forced_failures=[forced])
            except AdmissionRefused as exc:
                stop = exc.reason
                break
            archive.append(ctx.candidate_record(result, slot=0, stage="revise"))
        if stop is None:
            stop = StopReason.CALL_CAP
        selection, archive = ctx.vote(archive, "archive")
        selected = {c.candidate_id: c for c in archive}.get(selection.get("selected_candidate_id"))
        return ctx.finish(candidates=archive, selection=selection, native_final=selected, assigned_roster=1, stop_reason=stop)


__all__ = ["SHistoryPolicy"]
