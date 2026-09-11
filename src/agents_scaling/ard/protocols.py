"""ARD36 update rules: prompts, packet plans and the per-protocol schedules (WP2).

Spec §7 ("Protocols: what happens at every round") is implemented literally:

===============  ===============================================================
``FRESH``        five new independent task-only solutions each round
``PRIVATE``      task + own root + own latest report; the peer packet is empty
``PEER``         + the other four agents' previous-round reports, synchronously
``BALANCED``     one report per canonical class, multiplicity hidden, blind rep
``HUB5``         one leader reads four, then four workers read the leader
``GATE``/…       one protected agent gets PRIVATE's empty packet, four get PEER
===============  ===============================================================

Spec §6.4 fixes the memory of every iterative protocol — "task, the agent's original root
report, its latest report, and the specified current peer packet.  If root and latest
report are identical, include one copy.  Do not include private thinking traces" — and
this module owns the single renderer that produces it.  One renderer is not a convenience:
it is what makes spec §12.3's intervention exact ("the protected agent receives an empty
packet, byte-identical construction to PRIVATE") and what makes the E4 free ``t=0``
counterfactual pair *free* (plan §5.4: PRIVATE's round-1 prompt must be byte-identical to
the fork construction, so the request aliases in the store instead of being regenerated).

Ordering decisions are blind by construction (spec §7: "Randomize packet order with a
recorded schedule independent of correctness, shared across paired compatible arms.  Never
sort messages by gold correctness or fit a test-outcome-dependent order").  Every one of
them — packet order, BALANCED representative, HUB5 leader, RANDOM_PROTECT choice and GATE
argmax ties — is an ``identity.blind_order_key`` HMAC over the study seed and public
coordinates, and every one of them is keyed by the **pairing id** (item × checkpoint ×
cohort × repeat) rather than the protocol-qualified episode id, so paired arms such as
GATE and PEER produce byte-identical packets where the spec says they must.
"""

from __future__ import annotations

from typing import Callable, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard.packets import (
    ArdPacket,
    report_packet,
    unavailable_packet,
    unparsed_peer_packet,
)
from agents_scaling.ard.report import Report
from agents_scaling.ard.states import ReportRef, RoundState
from agents_scaling.study import identity
from agents_scaling.study.types import ProtocolError, SeedKey

__all__ = [
    "OUTPUT_CONTRACT",
    "PacketPlan",
    "confidence_protected_slot",
    "hub_leader",
    "hub_worker_plan",
    "packet_order",
    "packet_plan",
    "protected_slot_for",
    "random_protected_slot",
    "render_own_report",
    "render_root_prompt",
    "render_update_prompt",
    "root_seed_key",
    "update_seed_key",
]

#: One packet plan: recipient true slot -> the packets it will see, in delivery order.
PacketPlan = dict[int, tuple[ArdPacket, ...]]


# --------------------------------------------------------------------------- prompts

#: Spec §6.3, verbatim in the prompt.  The literal ``<final>`` is also the marker the
#: offline fake server routes on, so tests exercise the real prompt bytes.
OUTPUT_CONTRACT = (
    "Output contract: reply with exactly these three tags and nothing else.\n"
    f"{K.FINAL_OPEN}canonical answer or option letter{K.FINAL_CLOSE}\n"
    f"{K.EVIDENCE_OPEN}one or two concise, checkable reasons, at most "
    f"{K.EVIDENCE_MAX_TOKENS} tokens{K.EVIDENCE_CLOSE}\n"
    f"{K.CONFIDENCE_OPEN}probability from 0 to 1{K.CONFIDENCE_CLOSE}\n"
    "The answer precedes the confidence. The whole report is at most "
    f"{K.FINAL_TOKENS} tokens.\n"
    # Pre-freeze schema repair (spec §6.3).  The pilot showed the report allowance is
    # never the binding constraint when reasoning closes naturally (median 77 tokens),
    # but every truncated report came from a draw whose thinking hit the cap: cut off
    # mid-reasoning, the model carries on reasoning inside the report and overruns.  This
    # line addresses that case and nothing else, and applies to every item uniformly --
    # the spec forbids changing the cap selectively on difficult items.
    "If your reasoning is cut short, still reply with only these three tags: commit to "
    "your best current answer instead of continuing to reason in the report."
)

_ROOT_INSTRUCTION = "Solve the task independently and report in the required format."
_UPDATE_INSTRUCTION = (
    "Reconsider the task using only the material below and report in the required format. "
    "You may keep or change your answer."
)
_NO_PACKETS = "(no messages this round)"


def _approx_counter(text: str) -> int:
    """Fallback recipient-token counter: ~4 characters per token, rounded up.

    Used only when a :class:`RoundState` carries no tokenizer counter (unit tests of the
    plan logic).  Production always passes ``packets.make_counter(tokenizer)``: the frozen
    prompt budget of ``contract.PROMPT_TOKENS_CAP`` is stated in real tokens.
    """
    return (len(text) + 3) // 4


def counter_of(state: RoundState) -> Callable[[str], int]:
    return state.count if state.count is not None else _approx_counter


def render_own_report(report: Report) -> str:
    """An agent's own report as the prompt shows it back to it (spec §6.4 "preserve exact").

    Rendered in the frozen three-tag form so the memory an agent reads is the schema it
    must write.  A missing confidence stays missing — no 0.5 is invented (spec §6.3) — and
    an invalid report is shown as the typed marker rather than as a fabricated answer.
    """
    if not report.valid:
        return "(your previous report could not be parsed)"
    parts = [f"{K.FINAL_OPEN}{report.final_answer or ''}{K.FINAL_CLOSE}"]
    if report.evidence:
        parts.append(f"{K.EVIDENCE_OPEN}{report.evidence}{K.EVIDENCE_CLOSE}")
    if report.confidence is not None:
        parts.append(f"{K.CONFIDENCE_OPEN}{report.confidence}{K.CONFIDENCE_CLOSE}")
    return "".join(parts)


def render_root_prompt(task_text: str) -> tuple[dict[str, str], ...]:
    """The neutral root prompt — byte-identical across every protocol (plan §5.1).

    No slot name, no protocol name, no team size: independence lives in the seed, which is
    part of the cache key (spec §18), so all five root draws and every protocol share one
    prompt and one set of store records.
    """
    return (
        {
            "role": "user",
            "content": f"{_ROOT_INSTRUCTION}\n\nTask: {task_text}\n{OUTPUT_CONTRACT}",
        },
    )


def render_update_prompt(
    task_text: str,
    root: Report,
    latest: Report,
    packets: Sequence[ArdPacket] = (),
) -> tuple[dict[str, str], ...]:
    """The one update prompt of every iterative protocol (spec §6.4).

    ``task + own root + own latest + the specified packet``, with the latest report
    omitted when it is identical to the root ("include one copy"), and no protocol name
    anywhere.  An empty ``packets`` renders the ``(no messages this round)`` block, which
    is what PRIVATE, a protected GATE agent and an E4 private fork all produce — the same
    bytes, so the store aliases them into one record instead of paying for three.
    """
    blocks = [
        _UPDATE_INSTRUCTION,
        "",
        f"Task: {task_text}",
        OUTPUT_CONTRACT,
        "",
        "Your original report:",
        render_own_report(root),
    ]
    if root.content_sha256 != latest.content_sha256:
        blocks += ["", "Your latest report:", render_own_report(latest)]
    body = "\n".join(packet.serialized for packet in packets) if packets else _NO_PACKETS
    blocks += ["", "Messages from other agents:", body]
    return ({"role": "user", "content": "\n".join(blocks)},)


# --------------------------------------------------------------------------- seed keys


def root_seed_key(
    *, source_id: str, split: str, model_cell: str, draw: int, episode_rep: int = 0
) -> SeedKey:
    """``(actor_slot=0, purpose='root', step_slot=draw, ns='stateless_bank')`` (plan §5.1).

    The same key for every protocol, so the five shared root draws are one store record
    across the whole study and a protocol never pays to regenerate them.
    """
    return SeedKey(
        source_id=str(source_id),
        split=str(split),
        model_cell=str(model_cell),
        episode_rep=int(episode_rep),
        actor_slot=0,
        purpose=K.PURPOSE_ROOT,
        step_slot=int(draw),
        namespace=K.NS_STATELESS_BANK,
    )


def update_seed_key(
    *,
    source_id: str,
    split: str,
    model_cell: str,
    protocol: K.Protocol,
    agent_slot: int,
    round_: int,
    episode_rep: int = 0,
    purpose: str | None = None,
    namespace: str | None = None,
) -> SeedKey:
    """The seed key of one protocol update (plan §5.1).

    FRESH is the exception the bank layout demands: its round-``t`` draws are *roots* in
    the ``stateless_bank`` namespace at ordinal ``contract.fresh_draw_index(t, j)``, so the
    cumulative FRESH pool is draws 0-4 plus its own fifteen and never collides with the
    auxiliary E1 bank.
    """
    if protocol is K.Protocol.FRESH:
        return root_seed_key(
            source_id=source_id,
            split=split,
            model_cell=model_cell,
            draw=K.fresh_draw_index(int(round_), int(agent_slot)),
            episode_rep=episode_rep,
        )
    return SeedKey(
        source_id=str(source_id),
        split=str(split),
        model_cell=str(model_cell),
        episode_rep=int(episode_rep),
        actor_slot=int(agent_slot),
        purpose=str(purpose or K.PURPOSE_REVISE),
        step_slot=int(round_),
        namespace=str(namespace or K.PROTOCOL_NAMESPACE[protocol]),
    )


# --------------------------------------------------------------------------- schedules


def packet_order(
    state: RoundState, *, round_: int, recipient: int, senders: Sequence[int | str]
) -> tuple[int | str, ...]:
    """The recorded, correctness-blind delivery order of one recipient's packets (spec §7).

    Keyed by ``(pairing_id, round, recipient, sender)`` — deliberately *not* by protocol —
    so GATE and its paired PEER episode deliver the same senders in the same order and
    their packets can be compared byte-for-byte (spec §12.3).
    """
    return tuple(
        sorted(
            senders,
            key=lambda sender: identity.blind_order_key(
                state.study_seed,
                K.ORDER_PACKET,
                str(state.pairing_id),
                int(round_),
                int(recipient),
                str(sender),
            ),
        )
    )


def hub_leader(state: RoundState) -> int:
    """The HUB5 leader, drawn at episode initialization and fixed for the episode (spec §7)."""
    return min(
        range(K.N_AGENTS),
        key=lambda slot: identity.blind_order_key(
            state.study_seed, K.ORDER_HUB_LEADER, str(state.pairing_id), int(slot)
        ),
    )


def random_protected_slot(state: RoundState, *, round_: int) -> int:
    """RANDOM_PROTECT's uniformly selected agent, from the recorded schedule (spec §12.3)."""
    return min(
        range(K.N_AGENTS),
        key=lambda slot: identity.blind_order_key(
            state.study_seed, K.ORDER_PROTECT, str(state.pairing_id), int(round_), int(slot)
        ),
    )


def confidence_protected_slot(state: RoundState, *, round_: int) -> int:
    """CONF_PROTECT (spec §17 extension 3): protect the most confident agent.

    The frozen heuristic analogue of the learned controller — the agent whose own report
    is most confident is the one a confidence rule would predict benefits most from private
    continuation.  Missing confidences are *not* imputed (spec §6.3): an agent without one
    is ordered last, and every tie (including "nobody reported one") falls through to the
    same blind ``ORDER_PROTECT`` schedule RANDOM_PROTECT uses, so the two arms share one
    tie-break and differ only in the signal.
    """
    def key(slot: int) -> tuple:
        confidence = state.latest(slot).report.confidence
        present = confidence is not None
        return (
            0 if present else 1,
            -float(confidence) if present else 0.0,
            identity.blind_order_key(
                state.study_seed, K.ORDER_PROTECT, str(state.pairing_id), int(round_), int(slot)
            ),
        )

    return min(range(K.N_AGENTS), key=key)


def protected_slot_for(
    state: RoundState, protocol: K.Protocol, *, round_: int, controller_choice: int | None = None
) -> int:
    """Resolve the protected agent of a protection protocol (spec §12.3).

    GATE's choice is the frozen controller's argmax and must be supplied by WP5; the two
    controls derive theirs from the recorded schedules here.
    """
    if protocol is K.Protocol.RANDOM_PROTECT:
        return random_protected_slot(state, round_=round_)
    if protocol is K.Protocol.CONF_PROTECT:
        return confidence_protected_slot(state, round_=round_)
    if protocol is K.Protocol.GATE:
        if controller_choice is None:
            raise ProtocolError(
                "GATE's protected agent comes from the frozen controller (spec §12.3); "
                "pass controller_choice=choose_protected(...)"
            )
        return int(controller_choice)
    raise ProtocolError(f"{protocol} does not protect an agent")


# --------------------------------------------------------------------------- packet plan


def _peer_packet(
    state: RoundState, sender: int, *, round_: int, count: Callable[[str], int]
) -> ArdPacket:
    ref = state.latest(sender)
    if ref.round_ != round_ - 1:
        raise ProtocolError(
            f"round {round_} packet would carry a round-{ref.round_} report from slot "
            f"{sender}; a synchronous protocol reads round {round_ - 1} only (spec §19)"
        )
    if not ref.report.valid:
        # PEER transmits what the peer actually produced; an unparseable report is a typed
        # unavailable entry, never a fabricated answer.
        return unavailable_packet(count=count, source_round=round_ - 1)
    return report_packet(
        ref.report,
        pseudonym=state.pseudonyms[sender],
        count=count,
        source_round=round_ - 1,
        sender_slot=sender,
    )


def _peer_plan(
    state: RoundState, *, round_: int, count: Callable[[str], int]
) -> PacketPlan:
    plan: PacketPlan = {}
    for recipient in range(K.N_AGENTS):
        senders = [s for s in range(K.N_AGENTS) if s != recipient]
        order = packet_order(state, round_=round_, recipient=recipient, senders=senders)
        plan[recipient] = tuple(
            _peer_packet(state, int(s), round_=round_, count=count) for s in order
        )
    return plan


def _balanced_plan(
    state: RoundState, *, round_: int, count: Callable[[str], int]
) -> tuple[PacketPlan, dict[int, int]]:
    """One packet per canonical class + one typed unparsed entry (spec §7, plan §5.2).

    Returns the plan and, per recipient, how many peer reports the unparsed entry stands
    for.  That count is recorded on the episode — never in the packet bytes: multiplicity
    is exactly what BALANCED hides.
    """
    plan: PacketPlan = {}
    collapsed: dict[int, int] = {}
    for recipient in range(K.N_AGENTS):
        senders = [s for s in range(K.N_AGENTS) if s != recipient]
        classes: dict[str, list[int]] = {}
        unparsed: list[int] = []
        for sender in senders:
            ref = state.latest(sender)
            if ref.round_ != round_ - 1:
                raise ProtocolError(
                    f"round {round_} BALANCED packet would carry a round-{ref.round_} report"
                )
            name = state.canonical_class(ref.report)
            if name:
                classes.setdefault(name, []).append(sender)
            else:
                unparsed.append(sender)
        # Representative per class: the frozen HMAC schedule over (episode, round, class).
        # Never confidence, never logprob, never arrival order (plan §5.2).
        chosen: dict[str, int] = {}
        for name, members in classes.items():
            chosen[name] = min(
                members,
                key=lambda slot: identity.blind_order_key(
                    state.study_seed,
                    K.ORDER_BALANCED_REP,
                    str(state.pairing_id),
                    int(round_),
                    str(name),
                    int(slot),
                ),
            )
        tokens: list[int | str] = [chosen[name] for name in sorted(chosen)]
        if unparsed:
            tokens.append("unparsed")
        order = packet_order(state, round_=round_, recipient=recipient, senders=tokens)
        packets: list[ArdPacket] = []
        for token in order:
            if token == "unparsed":
                packets.append(
                    unparsed_peer_packet(
                        count=count, source_round=round_ - 1, collapsed=len(unparsed)
                    )
                )
            else:
                sender = int(token)
                packets.append(
                    report_packet(
                        state.latest(sender).report,
                        pseudonym=state.pseudonyms[sender],
                        count=count,
                        source_round=round_ - 1,
                        sender_slot=sender,
                    )
                )
        plan[recipient] = tuple(packets)
        collapsed[recipient] = len(unparsed)
    return plan, collapsed


def packet_plan(
    state: RoundState,
    protocol: K.Protocol | None = None,
    *,
    round_: int,
    protected_slot: int | None = None,
) -> PacketPlan:
    """The round-``round_`` packet plan built from the pre-round state (contract §2 WP2).

    * ``FRESH`` / ``PRIVATE`` → every slot gets ``()``.  (FRESH ignores the memory too; the
      episode loop renders it a root prompt at ``contract.fresh_draw_index``.)
    * ``PEER`` → the other four previous-round reports, in the recorded blind order.
    * ``BALANCED`` → one report per canonical class with a blind representative, plus one
      typed unparsed-peer entry when any peer report is invalid or unkeyed.
    * ``HUB5`` → the leader alone gets the other four previous reports; the workers get
      ``()`` here and their real packet from :func:`hub_worker_plan` *after* the leader has
      spoken, because a HUB5 worker reads the **current** leader report (spec §7: "Do not
      disguise this asynchronous information structure").
    * ``GATE`` / ``RANDOM_PROTECT`` / ``CONF_PROTECT`` → the protected slot gets ``()``,
      built identically to PRIVATE; the other four get the standard PEER packet over all
      four peers *including* the protected agent's round-``t-1`` report, in the same
      recorded order as the paired PEER episode (spec §12.3).

    Every packet carries ``source_round == round_ - 1`` and the builders raise
    ``ProtocolError`` if a report from any other round reaches them — the spec §19
    assertion "no request read same-round peers unless HUB5", enforced at construction.
    """
    proto = protocol if protocol is not None else state.protocol
    proto = K.Protocol(proto)
    if int(round_) < 1 or int(round_) > K.UPDATE_ROUNDS:
        raise ValueError(f"update rounds are 1..{K.UPDATE_ROUNDS}, got {round_}")
    if state.round_ != int(round_) - 1:
        raise ProtocolError(
            f"packet_plan for round {round_} needs the round-{int(round_) - 1} state, "
            f"got a round-{state.round_} state"
        )
    count = counter_of(state)

    if proto in (K.Protocol.FRESH, K.Protocol.PRIVATE):
        return {slot: () for slot in range(K.N_AGENTS)}
    if proto is K.Protocol.PEER:
        return _peer_plan(state, round_=round_, count=count)
    if proto is K.Protocol.BALANCED:
        return _balanced_plan(state, round_=round_, count=count)[0]
    if proto is K.Protocol.HUB5:
        leader = state.leader if state.leader is not None else hub_leader(state)
        senders = [s for s in range(K.N_AGENTS) if s != leader]
        order = packet_order(state, round_=round_, recipient=leader, senders=senders)
        plan: PacketPlan = {slot: () for slot in range(K.N_AGENTS)}
        plan[leader] = tuple(_peer_packet(state, int(s), round_=round_, count=count) for s in order)
        return plan
    if proto in K.PROTECTION_PROTOCOLS:
        if protected_slot is None:
            raise ProtocolError(f"{proto.value} needs a protected_slot (spec §12.3)")
        protected = int(protected_slot)
        if not 0 <= protected < K.N_AGENTS:
            raise ValueError(f"protected_slot must be in 0..{K.N_AGENTS - 1}")
        plan = _peer_plan(state, round_=round_, count=count)
        plan[protected] = ()
        return plan
    raise ProtocolError(f"no packet plan for protocol {proto!r}")


def balanced_collapsed_counts(state: RoundState, *, round_: int) -> dict[int, int]:
    """Per recipient, how many peer reports BALANCED's unparsed entry stands for (plan §5.2)."""
    return _balanced_plan(state, round_=round_, count=counter_of(state))[1]


def hub_worker_plan(
    state: RoundState, leader_report: ReportRef, *, round_: int
) -> PacketPlan:
    """The four HUB5 workers' packets: the **current** leader report (spec §7).

    This is the one place in ARD36 where a round-``t`` request reads a round-``t`` report,
    and it is recorded as such: the packet's ``source_round`` is ``round_``, the episode
    marks the phase ``worker``, and the leader's decision is ordered before the workers'.
    """
    leader = state.leader if state.leader is not None else hub_leader(state)
    if leader_report.round_ != int(round_):
        raise ProtocolError(
            f"HUB5 workers read the current leader report (round {round_}), got round "
            f"{leader_report.round_}"
        )
    count = counter_of(state)
    if leader_report.report.valid:
        packet = report_packet(
            leader_report.report,
            pseudonym=state.pseudonyms[leader],
            count=count,
            source_round=int(round_),
            sender_slot=leader,
        )
    else:
        packet = unavailable_packet(count=count, source_round=int(round_))
    return {slot: ((packet,) if slot != leader else ()) for slot in range(K.N_AGENTS)}
