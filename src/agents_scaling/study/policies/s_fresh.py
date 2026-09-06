"""``S_FRESH``: one logical solver slot restarted fresh on the task, draw after draw (WP4).

Spec §4.2 (``S_FRESH`` row: neutral ``00`` root, no previous answers or feedback; VOTE over
all admitted valid complete attempts; stateless banks use ``actor_slot=0`` and
``step_slot=draw ordinal``), §5.4/§6.2 (draws 0..9 alias the F00 bank by request id),
§6.5 (each draw one indivisible reservation; stop when the next draw does not fit or the
64-call cap is reached), §6.7 / audit A-1 (physical pre-generation in batches with the
admitted prefix computed afterwards from earlier debits only).  Architecture §1.11, §2.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from agents_scaling.study.policies.base import EpisodeContext
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
class SFreshPolicy:
    """Draws ``k = 0, 1, …`` with ``SeedKey(0, "root", k, "stateless_bank")``, framing ``00``."""

    id: ClassVar[Method] = Method.S_FRESH
    batch_size: int | None = None

    def run(self, ctx: EpisodeContext) -> EpisodeResult:
        if Framing(ctx.cell.framing) is not Framing.F00:
            raise ProtocolError(f"S_FRESH requires the neutral 00 root, cell {ctx.cell.cell_id} has {ctx.cell.framing.value}")
        ctx.ledger.set_final_reserve(0, 0)  # VOTE is deterministic (§4.4)
        cap = int(ctx.cfg.caps.solver_calls)
        rendered = R.render_root(ctx.task, Framing.F00)
        specs = [
            ctx.spec(rendered, SOLVER_DECODING, ctx.seed(0, PURPOSE_ROOT, k, NS_STATELESS_BANK), "root")
            for k in range(cap)
        ]
        results, refusal = ctx.generate_prefix(
            specs, "root", owner="draw", actor_slots=[0] * cap, steps=list(range(cap)), batch_size=self.batch_size
        )
        if not results:
            raise ProtocolError(
                f"B={ctx.ledger.B} cannot admit a single root draw ({refusal}): a budget that cannot start the "
                "mandatory protocol is a configuration error (§6.5)"
            )
        stop = refusal if refusal is not None else StopReason.CALL_CAP
        candidates = [ctx.candidate_record(r, slot=0, stage="root") for r in results]
        selection, candidates = ctx.vote(candidates, "archive")
        selected = {c.candidate_id: c for c in candidates}.get(selection.get("selected_candidate_id"))
        return ctx.finish(
            candidates=candidates,
            selection=selection,
            native_final=selected,
            assigned_roster=1,
            stop_reason=stop,
        )


__all__ = ["SFreshPolicy"]
