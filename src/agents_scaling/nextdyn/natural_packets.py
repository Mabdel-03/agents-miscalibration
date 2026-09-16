"""The E2a fixed-packet writer: one correct and one wrong donor per item (spec 9.1; plan §1.8).

**WRITER_SIDE (R7).**  This module reads grades -- that is its whole job -- so it is never in
the generation import closure.  It runs offline, before any receiver episode exists, and what
it produces is a *sealed register* of packet bytes that the RECEIVER cells then replay:
``register:packets/<sha>`` is the hard dependency (D16) that makes "the panel was frozen
before the assay ran" a checkable fact rather than a promise.

Two decisions here are the ones that matter for the assay's validity.

*The packets go through* :func:`packets.render_packet`.  A packet the receiver sees must be
the same **kind of object** as a packet an E1 agent sees: same 512-token cap under the same
priority clipping, counted with the same sender tokenizer, wrapped by the same pseudonym map,
checked by the same metadata denylist.  If E2a had its own renderer, an E2a endorsement effect
and an E1 transmission effect would be measured on two different artifacts and spec 9.7's link
between them would be meaningless.

*Validity is audited separately from correctness.*  "Natural rationale validity is not
guaranteed by final-answer correctness; record audited validity separately and keep the
unfiltered panel visible" (spec 9.1).  So :func:`validity_audit_row` records gold-free
observations of each chosen packet -- did it parse, does it carry a rationale, does its
wrapper carry an explicit correctness tag -- under an opaque ``audit_slot`` label, and the
panel is sealed **unfiltered**.  Nothing here drops an item because its rationale looked weak.

*What the register does and does not hide.*  The E2a arms are **defined** over the two roles:
``draft_role`` assigns the correct or the wrong packet as the working draft and the
endorsement counts are stated for one of them, so the cell builder has to know which packet is
which.  The register therefore states :data:`ROLE_KEY` explicitly on each packet entry --
saying so is better than encoding it in the packet order and calling the result blinded, which
is what an earlier version of this module did.  What the blinding buys is narrower and real:
the packet entries and the audit slots are emitted in **one blind order** drawn per item
(:func:`blind_order`), so neither an index nor a slot name is correctness, and an audit
exported without the role column (the audit rows carry only ``audit_slot`` and the packet
hash) cannot be re-joined to correctness through position.  The gold-free guarantee that is
enforced rather than asserted is :func:`_assert_label_free`: no grade, no canonical class and
no donor uid ever reaches the register.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Final, Mapping, Sequence

from agents_scaling.ard.packets import pseudonym_map
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import deps
from agents_scaling.nextdyn import ids
from agents_scaling.nextdyn import packets as PK
from agents_scaling.nextdyn import receiver as RX
from agents_scaling.nextdyn.errors import ContractViolation, LeakageError
from agents_scaling.nextdyn.grading import tri_state
from agents_scaling.nextdyn.protocols import PacketContent
from agents_scaling.study import identity

#: The tri-state labels this writer selects on (``grading.tri_state``); ``U`` is never a donor.
STATE_CORRECT: Final[str] = "C"
STATE_WRONG: Final[str] = "W"

#: Opaque slot names for the audit rows: the audit table names "slot 0 / slot 1" and a packet
#: hash, never "the correct one" (spec 9.1 "a blinded audit checks that packets are
#: interpretable").  Slot *i* is :func:`blind_order`'s *i*-th packet, so the slot name carries
#: no correctness even though the register's packet entries state their role.
AUDIT_SLOTS: Final[tuple[str, ...]] = ("audit_slot_0", "audit_slot_1")

#: The packet-entry key that states which member of the pair an entry is.  The E2a design is
#: defined over these two roles (``ReceiverSpec.draft_role``, the endorsement counts), so the
#: cell builder reads them here rather than inferring them from the entry order.
ROLE_KEY: Final[str] = "role"
ROLE_CORRECT: Final[str] = "correct"
ROLE_WRONG: Final[str] = "wrong"

#: Wrapper substrings that would be an explicit correctness tag inside the packet body.  The
#: audit *records* a hit; it never rewrites the model's own bytes (spec 6.1).
CORRECTNESS_TAGS: Final[tuple[str, ...]] = (
    "correct answer", "the right answer", "this is correct", "this is wrong",
    "incorrect answer", "verified correct", "ground truth",
)

REGISTER_KIND: Final[str] = deps.REGISTER_PACKETS


@dataclass(frozen=True)
class DonorDraw:
    """One graded donor-bank draw, as the writer sees it.

    ``output`` is the public parse (``schema.PublicOutput``) and ``grade_row`` the protected
    ``grades_private`` row; the label is read through :func:`grading.tri_state` and nowhere
    else, so a boolean grade column cannot creep back in (D15).
    """

    generation_uid: str
    decision_id: str
    draw_index: int
    output: Any
    grade_row: Any

    @property
    def state(self) -> str:
        return tri_state(self.grade_row)


@dataclass(frozen=True)
class ValidityAudit:
    """One blinded observation of one chosen packet (spec 9.1)."""

    audit_slot: str
    packet_hash: str
    token_count: int
    parsed: bool
    has_rationale: bool
    has_claims: bool
    truncated_fields: tuple[str, ...]
    explicit_correctness_tag: bool

    def as_entry(self) -> dict[str, Any]:
        """The register entry: ids, hashes and ints only (``deps.write_register`` refuses floats)."""
        return {
            "audit_slot": self.audit_slot,
            "packet_hash": self.packet_hash,
            "token_count": int(self.token_count),
            "parsed": bool(self.parsed),
            "has_rationale": bool(self.has_rationale),
            "has_claims": bool(self.has_claims),
            "truncated_fields": list(self.truncated_fields),
            "explicit_correctness_tag": bool(self.explicit_correctness_tag),
        }


def _first(draws: Sequence[DonorDraw], state: str, study_seed: bytes,
           source_item_id: str) -> DonorDraw | None:
    """The first gradeable donor of ``state`` in **blind donor order** (spec 8.2's order rule).

    Draw order would work too, but it correlates with the bank's own sampling schedule; the
    blind order is the study's standing answer to "which member of a class do we take?" and
    keeps the E2a panel free of that correlation.
    """
    eligible = [d for d in draws if d.state == state and d.output is not None and d.output.valid]
    if not eligible:
        return None
    return min(eligible, key=lambda d: (ids.blind(study_seed, K.ORDER_DONOR_ORDER, source_item_id,
                                                  d.generation_uid), d.generation_uid))


def fix_natural_pair(item: Any, donor_grades: Sequence[DonorDraw], *, study_seed: bytes,
                     counter: Callable[[str], int], counter_tokenizer_hash: str,
                     tokens: K.TokenContract,
                     panel_id: str = "") -> RX.FixedPacketPair | None:
    """Freeze one correct and one wrong donor packet for one item, or ``None``.

    ``None`` -- not an exception -- when the item's donor bank holds no gradeable correct or no
    gradeable wrong draw: an item that cannot supply the contrast is simply not in the E2a
    panel, and the recruitment count records it (spec 9.1 selects "without receiver outcomes").
    """
    correct = _first(donor_grades, STATE_CORRECT, study_seed, item.source_item_id)
    wrong = _first(donor_grades, STATE_WRONG, study_seed, item.source_item_id)
    if correct is None or wrong is None:
        return None
    labels = pseudonym_map(study_seed, _panel_pairing_id(item.source_item_id), 2)
    rendered = []
    for draw, pseudonym in ((correct, labels[0]), (wrong, labels[1])):
        packet = PK.render_packet(draw.output, pseudonym=pseudonym, counter=counter,
                                  counter_tokenizer_hash=counter_tokenizer_hash,
                                  cap=tokens.packet_cap, content=PacketContent.FULL)
        rendered.append(packet)
    pair = RX.FixedPacketPair(
        source_item_id=item.source_item_id,
        panel_id=panel_id or item.source_item_id,
        correct=rendered[0], wrong=rendered[1],
        correct_generation_uid=correct.generation_uid,
        wrong_generation_uid=wrong.generation_uid,
    )
    return pair


def _panel_pairing_id(source_item_id: str) -> str:
    """The pseudonym map's pairing id for an E2a item: the item, and nothing about the arm."""
    return f"e2a|{source_item_id}"


def validity_audit_row(packet: PK.TransmittedPacket, *, audit_slot: str) -> ValidityAudit:
    """One blinded validity observation.  Reads the packet bytes only -- never the grade."""
    if audit_slot not in AUDIT_SLOTS:
        raise ContractViolation(f"unknown audit slot {audit_slot!r}; declared: {AUDIT_SLOTS}")
    body = packet.serialized.lower()
    return ValidityAudit(
        audit_slot=audit_slot,
        packet_hash=packet.packet_hash,
        token_count=int(packet.token_count),
        parsed=PK.STATUS_KEY not in packet.fields,
        has_rationale=bool(str(packet.fields.get("reasoning", "")).strip()),
        has_claims=bool(packet.fields.get("shareable_claims")),
        truncated_fields=tuple(sorted(name for name, hit in packet.truncated.items() if hit)),
        explicit_correctness_tag=any(tag in body for tag in CORRECTNESS_TAGS),
    )


def blind_order(pair: RX.FixedPacketPair,
                study_seed: bytes) -> tuple[tuple[PK.TransmittedPacket, str], ...]:
    """The pair's two packets with their roles, in one blind order drawn per item.

    **One** draw feeds both the audit slots and the register's packet entries, so
    ``packets[i]`` is ``audit_slot_i`` and neither index is correctness.  Two independent
    draws, or a fixed correct-first packet list beside a blinded audit, would leave the packet
    hash as a join back to correctness -- which is the channel this order closes.
    """
    order = ((pair.correct, ROLE_CORRECT), (pair.wrong, ROLE_WRONG))
    key = ids.blind(study_seed, K.ORDER_PROTECT, pair.source_item_id, "e2a_audit_slot")
    return tuple(reversed(order)) if key and key[0] & 1 else order


def audit_rows(pair: RX.FixedPacketPair, study_seed: bytes) -> tuple[ValidityAudit, ...]:
    """Both audit rows of one pair, in :func:`blind_order`.

    The slot label carries no correctness: which of the two packets is ``audit_slot_0`` is
    drawn from the study seed, so an audit table read on its own -- slot, hash and the
    gold-free observations -- says nothing about which packet was the correct donor.
    """
    return tuple(validity_audit_row(packet, audit_slot=slot)
                 for (packet, _role), slot in zip(blind_order(pair, study_seed), AUDIT_SLOTS))


def panel_entry(pair: RX.FixedPacketPair, audits: Sequence[ValidityAudit], *,
                study_seed: bytes) -> dict[str, Any]:
    """One item's entry in the sealed packets register: bytes, hashes, counts, roles.

    The packets are emitted in :func:`blind_order` -- the same order the audit slots use -- and
    each states its :data:`ROLE_KEY`, because the cell builder that turns this entry back into
    a :class:`receiver.FixedPacketPair` needs the roles and reading them out of position would
    make the order a covert label.  No grade, no canonical class and no donor uid is here:
    that is what :func:`_assert_label_free` enforces.
    """
    ordered = blind_order(pair, study_seed)
    entry = {
        "source_item_id": pair.source_item_id,
        "packets": [
            {"pseudonym": p.pseudonym, "packet_id": p.packet_id, "packet_hash": p.packet_hash,
             "serialized": p.serialized, "token_count": int(p.token_count),
             "counter_tokenizer_hash": p.counter_tokenizer_hash,
             "original_content_hash": p.original_content_hash, ROLE_KEY: role}
            for p, role in ordered
        ],
        "validity_audit": [a.as_entry() for a in audits],
    }
    _assert_label_free(entry)
    return entry


_LABEL_WORDS: Final[tuple[str, ...]] = ("semantic_correct", "is_correct", "gold", "grade",
                                        "correct_uid", "wrong_uid", "canonical_class")


def _assert_label_free(entry: Mapping[str, Any]) -> None:
    """The sealed panel carries no *grade* (R7), which is a different claim from blindness.

    An E2a arm is defined over the correct/wrong roles, so the register states them (see the
    module docstring) and a reader of the register can tell the two packets apart -- that is
    the design, not a leak.  What may never appear is the writer's own grading: a
    ``semantic_correct``, a ``canonical_class``, a ``gold`` reference or a donor
    ``generation_uid`` would carry the grader's verdict on the *source item* into the
    generation lane, and that is what is refused here.
    """
    leaked = sorted(_label_keys(entry))
    if leaked:
        raise LeakageError(f"an E2a panel entry carries label key(s) {leaked}")


def _label_keys(node: Any) -> set[str]:
    """Label-ish **keys**, recursively.  Values are never scanned: a MATH rationale that
    happens to contain the word "gold" is the model's own bytes and is transmitted verbatim
    (spec 6.1), while a *key* named ``gold`` would be the writer leaking its own knowledge."""
    found: set[str] = set()
    if isinstance(node, Mapping):
        for key, value in node.items():
            found.update(word for word in _LABEL_WORDS if word in str(key))
            found |= _label_keys(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            found |= _label_keys(value)
    return found


def seal_panel(run_root: Any, pairs: Sequence[RX.FixedPacketPair], *, study_seed: bytes,
               family: str, model_hash: str, budget_hash: str) -> str:
    """Write the ``register:packets/<sha>`` register and return its sha (D16).

    Content-addressed over the entries, so two runs that fixed the same packets seal the same
    register and an E2a cell's dependency is satisfied by identity rather than by timing.  An
    empty panel is refused by ``deps.write_register``: a register with no entries would let a
    RECEIVER cell start against nothing.
    """
    entries = [panel_entry(pair, audit_rows(pair, study_seed), study_seed=study_seed)
               for pair in pairs]
    payload = {
        "family": str(family),
        "model_hash": str(model_hash),
        "budget_hash": str(budget_hash),
        "n_items": len(entries),
        deps.REGISTER_ENTRIES_FIELD: entries,
    }
    return deps.write_register(run_root, REGISTER_KIND, payload)


__all__ = [
    "AUDIT_SLOTS", "CORRECTNESS_TAGS", "DonorDraw", "REGISTER_KIND", "ROLE_CORRECT", "ROLE_KEY",
    "ROLE_WRONG", "STATE_CORRECT", "STATE_WRONG", "ValidityAudit", "audit_rows", "blind_order",
    "fix_natural_pair", "panel_entry", "seal_panel", "validity_audit_row",
]
