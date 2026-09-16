"""The transmitted packet: what one agent's previous draft looks like to somebody else.

Spec 6.1: a deterministic 512-token cap per agent, enforced by *token-aware field
truncation* with the final answer preserved first, then the public rationale, then the
claims in order; exactly which fields were truncated is marked, the original bytes and
hashes are retained externally, and no summariser is ever invoked.  Confidence is hidden in
both the own-draft and the peer rendering (spec 6.1, plan D6), and the only difference
between the two renderings is the source label.

Three facts about counting, each one a defect ARD36 had to learn:

* the cap is counted on the **serialized** wrapper, not on the excerpt, because the JSON
  escaping and the wrapper keys are tokens too (``study.packets.clip_fields`` re-tokenizes
  after every admission);
* the count uses the **sender's** tokenizer and the tokenizer's hash is stamped on the
  packet, so a cross-model replay transmits the stored bytes verbatim and never re-clips
  (the receiver's own count is recorded beside it, never substituted);
* a filler packet is length-matched in the *same* counter, so an ``_FILLED`` arm differs
  from its unfilled twin in content and not in how many packet slots a recipient sees.

``pseudonym_map`` is imported from ``ard.packets`` (already n-parametric, on the allowlist):
the pseudonym permutation is keyed on the ``pairing_id``, so paired arms of one team hand
out the same names and their packets stay byte-comparable.

Spec: 6.1, 6.3, 6.4, 8.6, 10.3, 13.2.  Plan: §1.6, D6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Final, Mapping, Sequence

from agents_scaling.ard.packets import pseudonym_map  # allowlisted: already n-parametric
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import prompts
from agents_scaling.nextdyn.errors import ContractViolation, LeakageError
from agents_scaling.nextdyn.protocols import PacketContent, TransmissionIntent
from agents_scaling.nextdyn.schema import PublicOutput
from agents_scaling.nextdyn.tables import TransmissionRow
from agents_scaling.study import identity
from agents_scaling.study.packets import _FieldSpec, clip_fields, serialize_wrapper

Counter = Callable[[str], int]

#: Spec 6.1 clip order: "Preserve the final answer first, then the public rationale, then
#: claims in order."  ``confidence`` is absent from the wrapper by design (spec 6.1).
FIELD_PRIORITY: Final[tuple[str, ...]] = ("final_answer", "reasoning", "shareable_claims")

FROM_KEY: Final[str] = "from"
TRUNCATED_KEY: Final[str] = "truncated"
STATUS_KEY: Final[str] = "status"
STATUS_UNPARSED: Final[str] = "unparsed"
FILLER_KEY: Final[str] = "notes"

#: The only keys a transmitted packet wrapper may carry.
ALLOWED_KEYS: Final[frozenset[str]] = frozenset(
    {FROM_KEY, TRUNCATED_KEY, STATUS_KEY, FILLER_KEY, *FIELD_PRIORITY})

#: Key names that would leak treatment or correctness into the packet *metadata*.  The
#: denylist is deliberately a check on wrapper keys and on the sender label -- never on the
#: model's own excerpt bytes, which are exact excerpts and are never rewritten (spec 6.1).
PACKET_METADATA_DENYLIST: Final[tuple[str, ...]] = (
    "confidence", "slot", "agent_slot", "correct", "incorrect", "valid", "score", "rank",
    "votes", "support", "majority", "minority", "protocol", "condition", "arm", "model",
)

PACKET_ID_PREFIX: Final[str] = "pk:"
PACKET_ID_CHARS: Final[int] = 24
DECLARED_EDGE_GRAPH: Final[str] = "graph"
DECLARED_EDGE_FILLER: Final[str] = "filler"


@dataclass(frozen=True)
class TransmittedPacket:
    """The bytes one recipient sees for one sender slot, plus the accounting behind them."""

    packet_id: str
    pseudonym: str
    fields: Mapping[str, Any]
    truncated: Mapping[str, bool]
    token_count: int
    counter_tokenizer_hash: str
    serialized: str
    packet_hash: str
    original_content_hash: str | None
    content: PacketContent
    filler: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", dict(self.fields))
        object.__setattr__(self, "truncated", {k: bool(v) for k, v in dict(self.truncated).items()})
        object.__setattr__(self, "content", PacketContent(self.content))
        if not self.counter_tokenizer_hash:
            raise ContractViolation("a transmitted packet must name the tokenizer that counted it")

    @property
    def any_truncated(self) -> bool:
        return any(self.truncated.values())


# --------------------------------------------------------------------------- byte checks


def check_packet_bytes(serialized: str) -> None:
    """Refuse packet *metadata* that would encode treatment, slot or correctness.

    What is checked: the wrapper's key set and the sender label.  What is deliberately not
    checked: the excerpt values, because they are the model's own bytes and the spec forbids
    silently rewriting or withholding them.  ``test_transmitted_packet_carries_no_confidence_
    and_no_slot`` and ``test_candidate_ids_and_slot_labels_do_not_encode_correctness`` pin it.
    """
    try:
        wrapper = json.loads(serialized)
    except ValueError as exc:
        raise LeakageError(f"a transmitted packet must be JSON: {exc}") from exc
    if not isinstance(wrapper, dict):
        raise LeakageError("a transmitted packet must be a JSON object")
    unknown = sorted(set(wrapper) - ALLOWED_KEYS)
    if unknown:
        raise LeakageError(f"transmitted packet carries undeclared key(s) {unknown}; "
                           f"allowed: {sorted(ALLOWED_KEYS)}")
    denied = sorted(k for k in wrapper if k.lower() in PACKET_METADATA_DENYLIST)
    if denied:
        raise LeakageError(f"transmitted packet carries denied key(s) {denied}")
    label = wrapper.get(FROM_KEY)
    if label is not None and (not isinstance(label, str) or any(ch.isdigit() for ch in label.split("-")[0])):
        raise LeakageError(f"packet source label {label!r} is not an opaque pseudonym")


# --------------------------------------------------------------------------- rendering


def _wrap(pseudonym: str) -> Callable[[dict[str, Any], dict[str, bool]], dict[str, Any]]:
    def wrap(fields: Mapping[str, Any], truncated: Mapping[str, bool]) -> dict[str, Any]:
        wrapper: dict[str, Any] = {FROM_KEY: pseudonym}
        wrapper.update({k: v for k, v in fields.items()})
        wrapper[TRUNCATED_KEY] = {k: bool(v) for k, v in truncated.items()}
        return wrapper

    return wrap


def _specs(out: PublicOutput, content: PacketContent) -> tuple[_FieldSpec, ...]:
    """Field specs in clip priority order; ``ANSWER_ONLY`` transmits the answer alone."""
    answer = _FieldSpec("final_answer", "str", out.final_answer or "")
    if content is PacketContent.ANSWER_ONLY:
        return (answer,)
    return (
        answer,
        _FieldSpec("reasoning", "str", out.reasoning or ""),
        _FieldSpec("shareable_claims", "list", list(out.shareable_claims)),
    )


def _packet_id(serialized: str) -> str:
    return PACKET_ID_PREFIX + identity.sha256_hex(serialized.encode("utf-8"))[:PACKET_ID_CHARS]


def render_packet(out: PublicOutput, *, pseudonym: str, counter: Counter,
                  counter_tokenizer_hash: str, cap: int, content: PacketContent) -> TransmittedPacket:
    """Clip one public output to ``cap`` tokens of serialized wrapper and stamp the counter.

    ``cap`` is ``TokenContract.packet_cap`` (512 in the main contract) and is passed in, not
    read here, so the same renderer serves the receiver assays at their own caps.  An output
    with no final answer is not a packet: it renders as the typed unparsed packet, because
    transmitting an empty answer would look like an answer.
    """
    content = PacketContent(content)
    if content is PacketContent.NONE:
        raise ContractViolation("a condition whose packet_content is 'none' transmits nothing; "
                                "render_packet is not called for it")
    if not out.valid:
        return unparsed_packet(pseudonym, counter, counter_tokenizer_hash=counter_tokenizer_hash,
                               content=content, original_content_hash=out.content_hash)
    fields, truncated, _spans, serialized = clip_fields(
        list(_specs(out, content)), _wrap(pseudonym), counter, int(cap))
    check_packet_bytes(serialized)
    return TransmittedPacket(
        packet_id=_packet_id(serialized), pseudonym=pseudonym, fields=fields, truncated=truncated,
        token_count=counter(serialized), counter_tokenizer_hash=counter_tokenizer_hash,
        serialized=serialized, packet_hash=identity.sha256_hex(serialized.encode("utf-8")),
        original_content_hash=out.content_hash, content=content, filler=False,
    )


def unparsed_packet(pseudonym: str, counter: Counter, *, counter_tokenizer_hash: str,
                    content: PacketContent = PacketContent.FULL,
                    original_content_hash: str | None = None) -> TransmittedPacket:
    """The typed body for a sender whose own output was not gradeable (spec 6.1).

    Task-free by construction: it carries the source label and a status, nothing else, so a
    failed slot cannot be read as an answer and cannot be told apart from another failed slot.
    """
    wrapper = {FROM_KEY: pseudonym, STATUS_KEY: STATUS_UNPARSED}
    serialized = serialize_wrapper(wrapper)
    check_packet_bytes(serialized)
    return TransmittedPacket(
        packet_id=_packet_id(serialized), pseudonym=pseudonym, fields={STATUS_KEY: STATUS_UNPARSED},
        truncated={}, token_count=counter(serialized), counter_tokenizer_hash=counter_tokenizer_hash,
        serialized=serialized, packet_hash=identity.sha256_hex(serialized.encode("utf-8")),
        original_content_hash=original_content_hash, content=PacketContent(content), filler=False,
    )


def inert_filler(n_tokens: int, seed: bytes, counter: Counter, *, pseudonym: str,
                 counter_tokenizer_hash: str) -> TransmittedPacket:
    """A length-matched packet built only from the frozen word list (spec 10.3).

    Words come from ``templates/filler_words.txt`` -- no digits, no bare option letter, no
    task vocabulary -- and are chosen by a deterministic walk over ``seed``, so the same
    missing slot yields the same filler on a re-run.  Words are appended while the serialized
    wrapper still fits ``n_tokens``, which is the *sender's* count of the packet it replaces,
    so the two arms differ in content and not in packet-slot count
    (``test_missing_packet_padding_has_no_task_content``).
    """
    if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 1:
        raise ContractViolation(f"filler length must be a positive token count, got {n_tokens!r}")
    words = prompts.filler_words()
    digest = identity.sha256_hex(bytes(seed)) if seed else identity.sha256_hex(b"")
    chosen: list[str] = []
    serialized = serialize_wrapper({FROM_KEY: pseudonym, FILLER_KEY: ""})
    index = 0
    while True:
        # deterministic, unbounded stream of word choices from the seed digest
        if index and index % len(digest) == 0:
            digest = identity.sha256_hex(digest.encode("utf-8"))
        word = words[int(digest[index % len(digest)], 16) % len(words)]
        trial = serialize_wrapper({FROM_KEY: pseudonym, FILLER_KEY: " ".join([*chosen, word])})
        if counter(trial) > n_tokens:
            break
        chosen.append(word)
        serialized = trial
        index += 1
        if len(chosen) > n_tokens:  # pragma: no cover - a counter that never grows
            break
    check_packet_bytes(serialized)
    return TransmittedPacket(
        packet_id=_packet_id(serialized), pseudonym=pseudonym, fields={FILLER_KEY: " ".join(chosen)},
        truncated={}, token_count=counter(serialized), counter_tokenizer_hash=counter_tokenizer_hash,
        serialized=serialized, packet_hash=identity.sha256_hex(serialized.encode("utf-8")),
        original_content_hash=None, content=PacketContent.FULL, filler=True,
    )


# --------------------------------------------------------------------------- preflight


def worst_case_prompt_tokens(contract: K.StudyContract, family: Any, n: int,
                             receiver_model: K.ModelCell) -> int:
    """Spec 18.5: the largest revision prompt a compared model can be asked to read.

    ``task_tokens`` (the family's longest rendered question) + the agent's own packet +
    ``n - 1`` peer packets + the template itself, all at ``TokenContract.packet_cap``.  It is
    computed per compared model because the count is per tokenizer; ``receiver_model`` names
    which one the caller means, and the caller supplies its counter through the family's
    ``max_question_tokens`` measurement.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ContractViolation(f"worst_case_prompt_tokens needs n >= 2, got {n!r}")
    contract.model(receiver_model.name)
    task_tokens = int(getattr(family, "max_question_tokens"))
    template_tokens = int(getattr(family, "template_tokens", 0) or 0)
    cap = contract.tokens.packet_cap
    return task_tokens + template_tokens + cap * n


# --------------------------------------------------------------------------- rows


def transmissions_for(plan: Mapping[int, Sequence[TransmissionIntent]],
                      packets: Mapping[int, TransmittedPacket], receiver_decision_id: str, *,
                      destination_round: int, receiver_slot: int, episode_id: str,
                      cell_kind: str, protocol_hash: str,
                      receiver_token_counts: Mapping[int, int] | None = None) -> tuple[TransmissionRow, ...]:
    """One ``transmissions`` row per packet slot of one receiving decision.

    Rows are emitted for filler slots too: a filler is a delivered slot with no source, and
    the analysis must be able to tell "no packet" from "an inert packet" without re-deriving
    the condition.  ``declared_edge`` records why the slot exists -- the resolved graph, or
    the length-matching rule.
    """
    intents = tuple(plan.get(receiver_slot, ()))
    rows: list[TransmissionRow] = []
    for intent in intents:
        packet = packets.get(intent.sender_slot)
        if packet is None:
            raise ContractViolation(
                f"no packet rendered for sender slot {intent.sender_slot} of receiver "
                f"{receiver_slot} at round {destination_round}")
        received = (receiver_token_counts or {}).get(intent.sender_slot)
        rows.append(TransmissionRow(
            transmission_id=_transmission_id(receiver_decision_id, intent),
            episode_id=episode_id,
            cell_kind=str(cell_kind),
            protocol_hash=protocol_hash,
            source_round=int(intent.source_round),
            destination_round=int(destination_round),
            sender_slot=int(intent.sender_slot),
            receiver_slot=int(receiver_slot),
            receiver_decision_id=receiver_decision_id,
            source_decision_id=intent.source_decision_id or None,
            packet_hash=packet.packet_hash,
            original_packet_hash=packet.original_content_hash,
            transmitted_tokens=int(packet.token_count),
            counter_tokenizer_hash=packet.counter_tokenizer_hash,
            transmitted_tokens_receiver=None if received is None else int(received),
            original_tokens=None,
            truncated_fields_json=json.dumps(dict(packet.truncated), sort_keys=True),
            filler_flag=bool(intent.filler),
            filler_tokens=int(packet.token_count) if intent.filler else None,
            packet_content_mode=PacketContent(intent.content).value,
            declared_edge=DECLARED_EDGE_FILLER if intent.filler else DECLARED_EDGE_GRAPH,
            display_position=int(intent.display_position),
        ))
    return tuple(rows)


def _transmission_id(receiver_decision_id: str, intent: TransmissionIntent) -> str:
    return "t:" + identity.sha256_hex(identity.jcs(
        [receiver_decision_id, int(intent.sender_slot), int(intent.source_round)]))[:32]


def transmission_ids(receiver_decision_id: str,
                     intents: Sequence[TransmissionIntent]) -> tuple[str, ...]:
    """The ids of the packet slots one decision saw (``decisions.seen_transmission_ids_json``).

    Built from the same function that stamps the rows, so a decision's ``seen`` list and the
    ``transmissions`` table cannot drift apart (R1).
    """
    return tuple(_transmission_id(receiver_decision_id, i) for i in intents)


def tokenizer_hash(model: K.ModelCell) -> str:
    """The identity of the tokenizer that counted a packet (``transmissions.counter_tokenizer_hash``).

    Packet token counts are per tokenizer, so a cross-model replay must be able to say which
    tokenizer produced the stored count instead of re-clipping the stored bytes (spec 6.1).
    It is the pinned tokenizer, not the weights: ``hf_id`` plus ``tokenizer_revision``.
    """
    if not model.tokenizer_revision:
        raise ContractViolation(f"model cell {model.name!r} has no pinned tokenizer_revision; "
                                "lock-inputs first")
    return identity.sha256_hex(identity.jcs([model.hf_id, model.tokenizer_revision]))[:16]


__all__ = [
    "ALLOWED_KEYS", "Counter", "DECLARED_EDGE_FILLER", "DECLARED_EDGE_GRAPH", "PACKET_METADATA_DENYLIST",
    "FIELD_PRIORITY", "FILLER_KEY", "FROM_KEY", "STATUS_KEY", "STATUS_UNPARSED",
    "TRUNCATED_KEY", "TransmittedPacket", "check_packet_bytes", "inert_filler", "pseudonym_map",
    "render_packet", "tokenizer_hash", "transmission_ids", "transmissions_for",
    "unparsed_packet", "worst_case_prompt_tokens",
]
