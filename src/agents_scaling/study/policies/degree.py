"""``DEGREE``: nine fixed F00 roots, blind nested peer sets, one focal revision, four
common-consumer children (WP4; module D).

Spec §4.6 ("nine fixed roots and blind nested peer sets d=0,1,2,4,8. Slot 0 is the focal
root; root ordering is randomized before assignment, never chosen using
confidence/validity/correctness ... one focal revision and four common-consumer attempts
for each degree ... roots shared with the five-root transport bank must match exact
request IDs; a nine-root acquisition is not presumed free"), §4.5 (the focal revision
"reconsider its own solution using whatever fallible information was provided"; four
fresh consumers receive only the task and the exact revised focal state), §4.3 (packets;
own candidate ≤ 8,192 tokens else a context failure).  Architecture §1.11 (roots = F00
draws 0..8 by the stateless key → aliases; permutation ``blind_order_key(study_seed,
"D_ROOT_PERM", source_id, ·)``; ``SeedKey(0,"revise",d,"DEGREE")`` and
``SeedKey(j,"consumer",d,"DEGREE")``); audit D-1/D-2; amendment D1 (correctness /
continuation outcomes only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from agents_scaling.study.identity import blind_order_key
from agents_scaling.study.policies.base import AdmissionRefused, EpisodeContext
from agents_scaling.study.prompts import render as R
from agents_scaling.study.resources.broker import StopReason
from agents_scaling.study.types import (
    NS_DEGREE,
    NS_STATELESS_BANK,
    PURPOSE_CONSUMER,
    PURPOSE_REVISE,
    PURPOSE_ROOT,
    SOLVER_DECODING,
    ContextFailure,
    EpisodeResult,
    Framing,
    Method,
    ProtocolError,
)

N_ROOTS = 9
DEGREES: tuple[int, ...] = (0, 1, 2, 4, 8)
N_CONSUMERS = 4
ROOT_PERM_NAMESPACE = "D_ROOT_PERM"


def root_permutation(study_seed: bytes, source_id: str) -> list[int]:
    """Blind order of the nine root draw ordinals for one item (position 0 = focal)."""
    return sorted(range(N_ROOTS), key=lambda i: blind_order_key(study_seed, ROOT_PERM_NAMESPACE, source_id, i))


@dataclass
class DegreePolicy:
    id: ClassVar[Method] = Method.DEGREE

    def run(self, ctx: EpisodeContext) -> EpisodeResult:
        d = ctx.cell.degree
        if d not in DEGREES:
            raise ProtocolError(f"DEGREE cell {ctx.cell.cell_id} has degree {d!r}; allowed {DEGREES}")
        if Framing(ctx.cell.framing) is not Framing.F00:
            raise ProtocolError("DEGREE roots are the neutral 00 bank (F00 draws 0..8)")
        ctx.ledger.set_final_reserve(0, 0)
        rendered = R.render_root(ctx.task, Framing.F00)
        root_specs = [ctx.spec(rendered, SOLVER_DECODING, ctx.seed(0, PURPOSE_ROOT, i, NS_STATELESS_BANK), "root") for i in range(N_ROOTS)]
        try:
            roots = ctx.generate_group(root_specs, "root", owner="roots", actor_slots=list(range(N_ROOTS)), steps=list(range(N_ROOTS)))
        except AdmissionRefused as exc:
            raise ProtocolError(f"B={ctx.ledger.B} cannot admit the nine roots ({exc.reason.value}): configuration error (§6.5)") from exc
        root_records = [ctx.candidate_record(r, slot=i, stage="root") for i, r in enumerate(roots)]
        perm = root_permutation(ctx.study_seed, ctx.task.source_id)
        focal_index = perm[0]
        peer_indices = perm[1 : 1 + d]
        focal_root = root_records[focal_index]
        forced: str | None = None
        try:
            ctx.check_own_state(focal_root)
        except ContextFailure as exc:
            forced = str(exc)
        packets = [ctx.packet_for(root_records[i], position, recipient_slot=0, round_=d) for position, i in enumerate(peer_indices, 1)]
        focal_spec = ctx.spec(R.render_focal_revision(ctx.task, focal_root, packets), SOLVER_DECODING, ctx.seed(0, PURPOSE_REVISE, d, NS_DEGREE), "revise")
        try:
            [focal] = ctx.generate_group([focal_spec], "revise", owner=f"focal_d{d}", actor_slots=[0], steps=[d], forced_failures=[forced])
        except AdmissionRefused as exc:
            raise ProtocolError(f"B={ctx.ledger.B} cannot admit the focal revision ({exc.reason.value}): configuration error (§6.5)") from exc
        revised = ctx.candidate_record(focal, slot=0, stage="focal_revision")
        forced_child: str | None = None
        try:
            ctx.check_own_state(revised)
        except ContextFailure as exc:
            forced_child = str(exc)
        consumer_rendered = R.render_common_consumer(ctx.task, revised)
        child_specs = [
            ctx.spec(consumer_rendered, SOLVER_DECODING, ctx.seed(j, PURPOSE_CONSUMER, d, NS_DEGREE), "consumer") for j in range(1, N_CONSUMERS + 1)
        ]
        try:
            children = ctx.generate_group(
                child_specs,
                "consumer",
                owner=f"consumers_d{d}",
                actor_slots=list(range(1, N_CONSUMERS + 1)),
                steps=[d] * N_CONSUMERS,
                forced_failures=[forced_child] * N_CONSUMERS,
            )
        except AdmissionRefused as exc:
            raise ProtocolError(f"B={ctx.ledger.B} cannot admit the four consumers ({exc.reason.value}): configuration error (§6.5)") from exc
        child_records = [ctx.candidate_record(ch, slot=j, stage="consumer") for j, ch in enumerate(children, 1)]
        extra: dict[str, Any] = {
            "degree": d,
            "root_permutation": perm,
            "focal_root_index": focal_index,
            "peer_root_indices": peer_indices,
            "focal_candidate_id": revised.candidate_id,
            "child_candidate_ids": [c.candidate_id for c in child_records],
        }
        return ctx.finish(
            candidates=root_records + [revised] + child_records,
            selection=None,
            native_final=revised,
            assigned_roster=1 + d,
            stop_reason=StopReason.COMPLETED,
            counters_extra={"degree": d, "roots_acquired": N_ROOTS, "consumers": N_CONSUMERS},
            episode_extra=extra,
        )


__all__ = ["DEGREES", "DegreePolicy", "N_CONSUMERS", "N_ROOTS", "ROOT_PERM_NAMESPACE", "root_permutation"]
