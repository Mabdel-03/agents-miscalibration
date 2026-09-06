"""``DEC`` (and the ``DEC_ONE_ROUND`` / ``IND_PRIVATE_REVISION`` controls) (WP4).

Spec §4.2 (``DEC`` row: N independent roots; each round revises every member from the
task, its own previous candidate and packets of the other N−1 members' *immediately
preceding* candidates; no current-round answer visible before the barrier; VOTE over the
N latest slot outputs with invalid latest outputs retained as failed opportunities; at
most eight revision rounds; "A full next round requires N calls and its eventual decision
reserve; if either allowance is unavailable, stop at the last completed round"; "No
member is selected for a last partial round"; the compiler validates ``N·(1+r) ≤ 64``,
so N=9 admits at most six rounds; at N=1 peers are empty), §4.1 (truthful disclosure;
peer display order by immutable hashes), §4.3 (packets; own candidate ≤ 8,192 tokens else
a context failure), §4.6 (``IND_PRIVATE_REVISION``: the same N=5 roots, one private
revision with no peers, terminal N-slot VOTE; ``DEC_ONE_ROUND``), §6.5 (each round one
all-or-nothing group reservation), §6.7 (peer order from ``blind_order_key``).
Architecture §1.11; corrections P0-1 (main-tier DEC roots carry the truthful clause →
``render_dec_root``; N-panel / one-round-control roots are ``00`` → alias F00), P1-3
(N=1 never mentions peers), §4 item 3 (previous-round-only packets) and item 7 (a
context-failed member gets the sentinel next round).

Which root wording a cell uses is decided by its manifest ``framing`` (``NATIVE`` →
``render_dec_root``; ``00`` → ``render_root(task, 00)``) so that E/M/B replications of the
main tier share the main-tier bytes; the module letter is only cross-checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from agents_scaling.study.identity import blind_order_key
from agents_scaling.study.policies.base import AdmissionRefused, EpisodeContext
from agents_scaling.study.prompts import render as R
from agents_scaling.study.resources.broker import StopReason
from agents_scaling.study.types import (
    NS_DEC,
    NS_STATELESS_BANK,
    PURPOSE_REVISE,
    PURPOSE_ROOT,
    SOLVER_DECODING,
    CandidateRecord,
    ContextFailure,
    EpisodeResult,
    Framing,
    Method,
    ProtocolError,
)

PEER_ORDER_NAMESPACE = "PEER_ORDER"


def dec_round_cap(N: int, call_cap: int, dec_max_rounds: int) -> tuple[int, str]:
    """``r_max = min(dec_max_rounds, call_cap // N − 1)`` and which bound is binding (§4.2)."""
    if N < 1:
        raise ValueError("N must be >= 1")
    by_calls = call_cap // N - 1
    if by_calls < 0:
        raise ProtocolError(f"N={N} does not even fit its roots under the {call_cap}-call cap")
    if by_calls < dec_max_rounds:
        return by_calls, "call_cap"
    return dec_max_rounds, "dec_max_rounds"


def peer_order(study_seed: bytes, source_id: str, member: int, round_: int, peers: list[int]) -> list[int]:
    """Blind, correctness-independent display order of ``peers`` for (member, round) (§4.1, §6.7)."""
    return sorted(peers, key=lambda p: blind_order_key(study_seed, PEER_ORDER_NAMESPACE, source_id, member, round_, p))


@dataclass
class DecPolicy:
    """``max_rounds`` caps the revision rounds below the compiler bound (the one-round
    controls); ``private=True`` renders every revision without peers and without the DEC
    contract (``IND_PRIVATE_REVISION``, §4.6)."""

    max_rounds: int | None = None
    private: bool = False

    id: ClassVar[Method] = Method.DEC

    def _roots(self, ctx: EpisodeContext) -> R.Rendered:
        framing = Framing(ctx.cell.framing)
        N = ctx.N
        if framing is Framing.NATIVE:
            if N < 2:
                raise ProtocolError("a NATIVE (truthful) DEC root needs N >= 2; N=1 cells use the 00 root (P1-3)")
            if ctx.cell.module == "N":
                raise ProtocolError("N-panel DEC cells use the neutral 00 root (P0-1)")
            return R.render_dec_root(ctx.task, N)
        if framing is Framing.F00:
            if ctx.cell.module == "A" and N >= 2 and self.id is Method.DEC:
                raise ProtocolError("main-tier (module A) DEC roots carry the truthful clause: framing must be NATIVE (P0-1)")
            return R.render_root(ctx.task, Framing.F00)
        raise ProtocolError(f"DEC roots are NATIVE or 00, never {framing.value}")

    def run(self, ctx: EpisodeContext) -> EpisodeResult:
        N = ctx.N
        caps = ctx.cfg.caps
        r_max, binding = dec_round_cap(N, int(caps.solver_calls), int(caps.dec_max_rounds))
        if self.max_rounds is not None:
            if self.max_rounds < 0:
                raise ValueError("max_rounds must be >= 0")
            if self.max_rounds < r_max:
                r_max, binding = self.max_rounds, "max_rounds"
        ctx.ledger.set_final_reserve(0, 0)  # VOTE is deterministic
        rendered = self._roots(ctx)
        root_specs = [ctx.spec(rendered, SOLVER_DECODING, ctx.seed(0, PURPOSE_ROOT, s, NS_STATELESS_BANK), "root") for s in range(N)]
        try:
            roots = ctx.generate_group(root_specs, "root", owner="roots", actor_slots=list(range(N)), steps=[0] * N)
        except AdmissionRefused as exc:
            raise ProtocolError(
                f"B={ctx.ledger.B} cannot admit the N={N} roots ({exc.reason.value}): configuration error (§6.5)"
            ) from exc
        latest: list[CandidateRecord] = [ctx.candidate_record(r, slot=s, stage="root") for s, r in enumerate(roots)]
        archive: list[CandidateRecord] = list(latest)
        rounds: list[dict[str, Any]] = []
        stop: StopReason | None = None
        for r in range(1, r_max + 1):
            specs = []
            forced: list[str | None] = []
            packets_meta: list[list[str]] = []
            for s in range(N):
                own = latest[s]
                reason: str | None = None
                try:
                    ctx.check_own_state(own)
                except ContextFailure as exc:
                    reason = str(exc)
                if self.private:
                    packets: list[Any] = []
                    messages = R.render_focal_revision(ctx.task, own, packets)
                else:
                    order = peer_order(ctx.study_seed, ctx.task.source_id, s, r, [p for p in range(N) if p != s])
                    packets = [ctx.packet_for(latest[p], p, recipient_slot=s, round_=r) for p in order]
                    messages = R.render_dec_revision(ctx.task, own, packets, N, r)
                packets_meta.append([p.packet_id for p in packets])
                specs.append(ctx.spec(messages, SOLVER_DECODING, ctx.seed(s, PURPOSE_REVISE, r, NS_DEC), "revise"))
                forced.append(reason)
            try:
                results = ctx.generate_group(
                    specs, "revise", owner=f"round{r}", actor_slots=list(range(N)), steps=[r] * N, forced_failures=forced
                )
            except AdmissionRefused as exc:
                stop = exc.reason
                break
            latest = [ctx.candidate_record(res, slot=s, stage=f"round{r}") for s, res in enumerate(results)]
            archive.extend(latest)
            rounds.append({"round": r, "packets": packets_meta, "context_failures": [f for f in forced if f is not None]})
        if stop is None:
            # Every permitted round ran: report the bound that fixed r_max — the 64-call cap
            # when it, not the round cap, was binding (§6.5 reporting; review P1-C).
            stop = StopReason.CALL_CAP if binding == "call_cap" else StopReason.ROUND_CAP
        selection, latest_keyed = ctx.vote(latest, "latest_slots")
        keyed = {c.candidate_id: c for c in latest_keyed}
        archive = [keyed.get(c.candidate_id, c) for c in archive]
        selected = keyed.get(selection.get("selected_candidate_id"))
        return ctx.finish(
            candidates=archive,
            selection=selection,
            native_final=selected,
            assigned_roster=N,
            stop_reason=stop,
            counters_extra={"rounds_completed": len(rounds), "r_max": r_max, "r_max_binding": binding},
            episode_extra={"rounds": rounds, "latest_candidate_ids": [c.candidate_id for c in latest]},
        )


@dataclass
class DecOneRoundPolicy(DecPolicy):
    """``DEC_ONE_ROUND``: the DEC protocol with ``max_rounds=1`` (§4.6, N panel)."""

    max_rounds: int | None = 1
    private: bool = False

    id: ClassVar[Method] = Method.DEC_ONE_ROUND


@dataclass
class IndPrivateRevisionPolicy(DecPolicy):
    """``IND_PRIVATE_REVISION``: the same roots, one private revision each (no peers, no
    DEC contract — the neutral ``focal_revision`` instruction), terminal N-slot VOTE (§4.6)."""

    max_rounds: int | None = 1
    private: bool = True

    id: ClassVar[Method] = Method.IND_PRIVATE_REVISION


__all__ = ["DecOneRoundPolicy", "DecPolicy", "IndPrivateRevisionPolicy", "PEER_ORDER_NAMESPACE", "dec_round_cap", "peer_order"]
