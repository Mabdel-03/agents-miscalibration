"""ARD36 message packets: the study byte primitives under the ARD field priority (WP2).

Spec §6.4: "Peer confidence numbers are stripped in the base protocols.  They remain
measurable in logs."  Spec §7: "For every peer packet use stable pseudonymous agent IDs."
Spec §5.2 (plan): the packet carries one whole 256-token report, so the deterministic
priority clipper of ``study/packets.py`` is a *safety net* here, not a routine operation.

Why this module exists at all (contract §3, verified): ``study.packets`` cannot carry an
ARD report.  Its frozen ``PACKET_FIELD_PRIORITY`` contains ``confidence`` — which ARD36
must strip from every base-protocol packet — and ``study.types.Packet.sender_slot`` must
be a non-negative int that is hashed into the packet id, which would put the *true* agent
slot into the wire identity.  ARD36 therefore reuses the frozen byte primitives
(:func:`~agents_scaling.study.packets.clip_fields`,
:func:`~agents_scaling.study.packets.largest_prefix`,
:func:`~agents_scaling.study.packets.safe_prefix_end`, ``serialize_wrapper``) and wraps
them in its own :class:`ArdPacket`, whose sender identity is a per-episode **pseudonym
string**.

Three packet kinds, all deterministic bytes:

* a **report** packet — ``{"from": "agent-k", "final_answer": …, "evidence": …,
  "truncated": {…}}``;
* the typed **unparsed peer** entry — one entry collapsing every invalid or unkeyed peer
  report of a BALANCED round (plan §5.2); it carries no pseudonym, because naming one of
  the collapsed senders would leak which peers failed, and no count, because the count is
  recorded per episode and must not be visible in the bytes;
* the typed **unavailable** entry — the frozen ``{"status":"unavailable"}`` bytes of
  ``study.types.PACKET_UNAVAILABLE_JSON``, for a slot with no source report at all.

Nothing in a packet says how many peers hold the answer (BALANCED hides multiplicity),
what an agent's true slot is, or what anyone's confidence was.  :func:`check_packet_bytes`
is the frozen sanitizer that asserts it, and it runs on every packet this module builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard.report import Report
from agents_scaling.study import identity
from agents_scaling.study.packets import (  # frozen byte primitives (contract §3)
    _FieldSpec as FieldSpec,
    clip_fields,
    largest_prefix,
    make_counter,
    safe_prefix_end,
    serialize_wrapper,
)
from agents_scaling.study.types import PACKET_UNAVAILABLE_JSON, ProtocolError

__all__ = [
    "ARD_FIELD_PRIORITY",
    "ArdPacket",
    "FORBIDDEN_PACKET_SUBSTRINGS",
    "PACKET_KIND_REPORT",
    "PACKET_KIND_UNAVAILABLE",
    "PACKET_KIND_UNPARSED",
    "PSEUDONYM_PREFIX",
    "UNPARSED_PEER_JSON",
    "check_packet_bytes",
    "clip_fields",
    "largest_prefix",
    "make_counter",
    "packet_cap",
    "pseudonym_map",
    "report_packet",
    "safe_prefix_end",
    "serialize_wrapper",
    "unavailable_packet",
    "unparsed_peer_packet",
]

#: ARD36 field priority (plan §4.3).  ``confidence`` is deliberately absent: it is carried
#: outside the packet, in the private decision log, and stripped from every base protocol.
ARD_FIELD_PRIORITY: tuple[str, ...] = ("final_answer", "evidence")

PACKET_KIND_REPORT = "report"
PACKET_KIND_UNPARSED = "unparsed_peer"
PACKET_KIND_UNAVAILABLE = "unavailable"

#: Exact bytes of the typed unparsed-peer entry (no pseudonym, no count, no class).
UNPARSED_PEER_JSON: str = serialize_wrapper({"status": PACKET_KIND_UNPARSED})

PSEUDONYM_PREFIX = "agent-"

#: The frozen sanitizer's denylist over the serialized packet (spec §6.4, plan §5.2).
FORBIDDEN_PACKET_SUBSTRINGS: tuple[str, ...] = (
    "confidence",
    "sender_slot",
    "agent_slot",
    "multiplicity",
    "class_count",
    "n_peers",
)


def packet_cap() -> int:
    """Recipient-token cap of one packet: one whole report (``contract.REPORT_TOKENS_CAP``).

    The frozen prompt budget (``contract.PROMPT_TOKENS_CAP``) allocates exactly
    ``(N_AGENTS - 1) * REPORT_TOKENS_CAP`` to peer packets, so the cap that keeps that
    arithmetic true is the same constant.  A report that fits the spec §6.3 256-token
    report allowance fits here with room for the JSON wrapper; the clipper only bites on a
    pathological report and is asserted never to bite in the tests.
    """
    return int(K.REPORT_TOKENS_CAP)


# --------------------------------------------------------------------------- pseudonyms


def pseudonym_map(study_seed: bytes, pairing_id: str, n: int = K.N_AGENTS) -> tuple[str, ...]:
    """Per-episode pseudonyms ``agent-0…agent-{n-1}``, indexed by *true* slot (spec §7).

    "For every peer packet use stable pseudonymous agent IDs" — re-drawn per episode and
    stable within it (plan §5.2).  The permutation comes from
    ``identity.blind_order_key(study_seed, contract.ORDER_PSEUDONYM, pairing_id, slot)``,
    so it is independent of correctness, confidence and arrival order.

    ``pairing_id`` is the *paired-arm* episode identity (item × checkpoint × cohort ×
    repeat), not the protocol-qualified ``episode_id``: GATE and its paired PEER episode
    must hand out the same pseudonyms or their packets could not be byte-identical
    (spec §12.3, plan §5.2).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    order = sorted(
        range(n),
        key=lambda slot: identity.blind_order_key(
            study_seed, K.ORDER_PSEUDONYM, str(pairing_id), int(slot)
        ),
    )
    names: list[str] = [""] * n
    for position, slot in enumerate(order):
        names[slot] = f"{PSEUDONYM_PREFIX}{position}"
    return tuple(names)


# --------------------------------------------------------------------------- the packet


@dataclass(frozen=True)
class ArdPacket:
    """One bounded ARD36 message packet.  ``serialized`` is the exact bytes the peer sees."""

    packet_id: str
    kind: str
    pseudonym: str | None
    content_sha256: str
    fields: Mapping[str, Any]
    truncated: Mapping[str, bool]
    spans: tuple[Any, ...]
    recipient_tokens: int
    serialized: str
    #: Round whose report this packet carries.  The episode loop asserts round ``t``
    #: packets only ever carry ``t-1`` reports unless the protocol is HUB5 (spec §19).
    source_round: int = -1
    #: True agent slot of the sender — **private log only**, never serialized.
    sender_slot: int | None = None

    @property
    def bytes_sha256(self) -> str:
        return identity.sha256_hex(self.serialized)

    @property
    def clipped(self) -> bool:
        return any(bool(v) for v in self.truncated.values())

    def to_row(self) -> dict[str, Any]:
        """Private log row (``decisions.parquet`` ``seen_message_ids`` provenance)."""
        return {
            "packet_id": self.packet_id,
            "kind": self.kind,
            "pseudonym": self.pseudonym,
            "sender_slot": self.sender_slot,
            "content_sha256": self.content_sha256,
            "bytes_sha256": self.bytes_sha256,
            "recipient_tokens": self.recipient_tokens,
            "source_round": self.source_round,
            "clipped": self.clipped,
        }


def check_packet_bytes(serialized: str) -> None:
    """The frozen sanitizer (spec §6.4): refuse bytes that leak stripped metadata.

    Applied to *every* protocol's packets, so PEER and BALANCED are sanitized identically
    and the BALANCED−PEER contrast is not confounded by a policy that only ran on one arm.
    Confidence wording the model itself embedded in ``<evidence>`` is preserved verbatim
    (spec §6.4 forbids erasing it); what is refused is a structural ``confidence`` key.
    """
    lowered = serialized.lower()
    for needle in FORBIDDEN_PACKET_SUBSTRINGS:
        if f'"{needle}"' in lowered:
            raise ProtocolError(
                f"ARD36 packet bytes carry the stripped field {needle!r}: {serialized[:120]}"
            )


def _packet_id(kind: str, pseudonym: str | None, content_sha256: str, serialized: str) -> str:
    return identity.sha256_hex(
        identity.jcs(["ard_packet", kind, pseudonym, content_sha256, serialized])
    )


def report_packet(
    report: Report,
    *,
    pseudonym: str,
    count: Callable[[str], int],
    source_round: int = -1,
    sender_slot: int | None = None,
    cap: int | None = None,
) -> ArdPacket:
    """One peer's report as a packet: ``final_answer`` then ``evidence``, no confidence.

    The two fields are exact source excerpts admitted in :data:`ARD_FIELD_PRIORITY` order
    by the frozen clipper, so if a pathological report ever exceeded the cap the answer
    survives and the evidence is the field that shrinks (spec §4.3 priority semantics).
    """
    if not isinstance(report, Report):
        raise TypeError("report_packet expects an ard.report.Report")
    if not pseudonym or not isinstance(pseudonym, str):
        raise ValueError("a packet sender is identified by a non-empty pseudonym string")
    specs = [
        FieldSpec("final_answer", "str", report.final_answer or ""),
        FieldSpec("evidence", "str", report.evidence or ""),
    ]

    def wrap(fields: dict[str, Any], truncated: dict[str, bool]) -> dict[str, Any]:
        return {"from": pseudonym, **fields, "truncated": dict(truncated)}

    limit = packet_cap() if cap is None else int(cap)
    fields, truncated, spans, serialized = clip_fields(specs, wrap, count, limit)
    check_packet_bytes(serialized)
    return ArdPacket(
        packet_id=_packet_id(PACKET_KIND_REPORT, pseudonym, report.content_sha256, serialized),
        kind=PACKET_KIND_REPORT,
        pseudonym=pseudonym,
        content_sha256=report.content_sha256,
        fields=dict(fields),
        truncated=dict(truncated),
        spans=tuple(spans),
        recipient_tokens=int(count(serialized)),
        serialized=serialized,
        source_round=int(source_round),
        sender_slot=None if sender_slot is None else int(sender_slot),
    )


def unparsed_peer_packet(
    *, count: Callable[[str], int], source_round: int = -1, collapsed: int = 0
) -> ArdPacket:
    """The single typed "unparsed peer" entry of a BALANCED round (plan §5.2).

    Every invalid or unkeyed peer report collapses into exactly one of these.  ``collapsed``
    (how many reports it stands for) is recorded on the packet object for the episode log
    and the BALANCED/PEER packet-count difference — and is deliberately **not** in
    ``serialized``: multiplicity is what BALANCED hides.
    """
    serialized = UNPARSED_PEER_JSON
    check_packet_bytes(serialized)
    return ArdPacket(
        packet_id=_packet_id(PACKET_KIND_UNPARSED, None, "", serialized),
        kind=PACKET_KIND_UNPARSED,
        pseudonym=None,
        content_sha256="",
        fields={"status": PACKET_KIND_UNPARSED},
        truncated={},
        spans=(),
        recipient_tokens=int(count(serialized)),
        serialized=serialized,
        source_round=int(source_round),
        sender_slot=None,
    )


def unavailable_packet(*, count: Callable[[str], int], source_round: int = -1) -> ArdPacket:
    """The frozen typed unavailable entry (``study.types.PACKET_UNAVAILABLE_JSON``)."""
    serialized = PACKET_UNAVAILABLE_JSON
    check_packet_bytes(serialized)
    return ArdPacket(
        packet_id=_packet_id(PACKET_KIND_UNAVAILABLE, None, "", serialized),
        kind=PACKET_KIND_UNAVAILABLE,
        pseudonym=None,
        content_sha256="",
        fields={"status": "unavailable"},
        truncated={},
        spans=(),
        recipient_tokens=int(count(serialized)),
        serialized=serialized,
        source_round=int(source_round),
        sender_slot=None,
    )


def render_packets(packets: Sequence[ArdPacket]) -> str:
    """The packet block as it appears in a prompt: one serialized packet per line.

    An empty packet tuple renders as the empty string, which is what makes a PRIVATE
    prompt and a protected GATE agent's prompt byte-identical (spec §12.3).
    """
    return "\n".join(packet.serialized for packet in packets)
