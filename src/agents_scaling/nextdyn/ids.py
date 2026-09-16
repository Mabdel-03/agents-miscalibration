"""The identity scheme (spec 17.2; plan §1.3, D4, D8, D14, D15).

Four identities, each naming a different thing:

=================  ======================================  =======================================
id                 names                                   built from
=================  ======================================  =======================================
``decision_id``    the intended observation                :class:`CompositeKey` (12 fields)
``request_hash``   exact bytes + decoding + seed + model   ``study.types.RequestSpec.request_id``
``generation_uid`` the physical sample                     the physical request ids of a draw
``content_hash``   the output text                         the post-``</think>`` bytes
=================  ======================================  =======================================

Two rules from the ARD36 defects are enforced here rather than by convention:

* **R4** ``cell_kind`` is the first component of every id.  :func:`episode_id` refuses bank
  kinds and :func:`bank_draw_id` refuses episode kinds, so a bank draw can never carry the id
  a PRIVATE round-0 row carries (defect D-11: one 16-draw bank shared the episode id and
  produced 14,336 duplicate decision keys).
* **R5 / D4** nothing about the *condition* is inside the seed.  :func:`seed_key_for` builds the
  ``study.types.SeedKey`` from item, variant, root team, replicate, round, slot, phase, cell
  kind and the generating model only; ``protocol_hash`` and ``budget_hash`` are deliberately
  absent so paired arms share common random numbers and byte-identical prompts alias in the
  store.
* **The model is in every key.**  ``receiver_model_hash`` (the spec-17.2 name) is
  :func:`model_hash` of *the model that generates the decision* for every kind -- bank draws,
  episode rows, receiver phases and audit draws alike; the spec key has no other model slot.
  A key with an empty ``receiver_model_hash`` is refused, because two models' draws of one
  item (or the 8B and 32B episodes of one team, E3c) would otherwise collide on
  ``decision_id``.

Per-kind key layout (``docs/nextdyn/00_contract.md`` §2.0 carries the same table)::

    kind             root_team_id          protocol_hash              budget_hash                 round  agent_slot  phase
    *_BANK           ""                    ""                         tokens.budget_hash()        0      draw index  bank
    EPISODE          ids.root_team_id(..)  VisibilitySpec.hash        episode TokenContract       0..R   slot        root_ref | revision
    RECEIVER         panel id              ReceiverSpec.hash          receiver_tokens[B/one_call] 0      0           receiver_plan | receiver_decision
    SELECTION_AUDIT  pool_id               audited episode's hash     audit_tokens.budget_hash()  pool r candidate   audit
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn.errors import IdentityError
from agents_scaling.study import identity
from agents_scaling.study.types import SeedKey


class CellKind(str, Enum):
    """Typed cell kinds (plan D8).  The value is the first component of every id."""

    DIFFICULTY_BANK = "difficulty_bank"
    DONOR_BANK = "donor_bank"
    NATURAL_ROOT = "natural_root"
    EPISODE = "episode"
    RECEIVER = "receiver"
    SELECTION_AUDIT = "selection_audit"
    GRADE = "grade"
    CONSTRUCT = "construct"
    CAPTURE = "capture"
    E0 = "e0"
    MATERIALIZE = "materialize"


BANK_KINDS: Final[frozenset[CellKind]] = frozenset({
    CellKind.DIFFICULTY_BANK, CellKind.DONOR_BANK, CellKind.NATURAL_ROOT,
})
EPISODE_KINDS: Final[frozenset[CellKind]] = frozenset({CellKind.EPISODE, CellKind.RECEIVER})
GENERATION_KINDS: Final[frozenset[CellKind]] = BANK_KINDS | EPISODE_KINDS | {CellKind.SELECTION_AUDIT}

_HEX_RE = re.compile(r"^[0-9a-f]+$")
_ID_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:@+\-]+$")


def coerce_kind(kind: CellKind | str) -> CellKind:
    try:
        return CellKind(kind)
    except ValueError as exc:
        raise IdentityError(f"unknown cell kind {kind!r}") from exc


def _token(name: str, value: str) -> str:
    """An id component: non-empty, no separator characters."""
    if not isinstance(value, str) or not value:
        raise IdentityError(f"{name} must be a non-empty string, got {value!r}")
    if "|" in value or "/" in value or not _ID_TOKEN_RE.match(value):
        raise IdentityError(f"{name} {value!r} contains a separator or an unsupported character")
    return value


def _short(value: str, n: int) -> str:
    """The first ``n`` characters after an optional ``<tag>:`` prefix (``team:``, ``c:``,
    ``packets:``); a body shorter than ``n`` (an E2b problem id such as ``mod97:d4:i003``)
    is hashed instead, so every token of any length yields exactly ``n`` characters and no
    two distinct tokens are silently truncated onto each other."""
    body = value.split(":", 1)[1] if ":" in value else value
    if len(body) >= n:
        return body[:n]
    return identity.sha256_hex(value.encode("utf-8"))[:n]


# --------------------------------------------------------------------------- composite key


@dataclass(frozen=True)
class CompositeKey:
    """Exactly the twelve spec-17.2 fields, in spec order.  ``decision_id`` hashes them."""

    study_id: str
    cell_kind: CellKind
    source_item_id: str
    item_variant_id: str
    root_team_id: str
    protocol_hash: str
    receiver_model_hash: str
    budget_hash: str
    replicate_id: int
    round: int
    agent_slot: int
    phase: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_kind", coerce_kind(self.cell_kind))
        if not isinstance(self.source_item_id, str) or not self.source_item_id:
            raise IdentityError("CompositeKey.source_item_id must never be null or empty "
                                "(spec 17.2; failure_policy.null_item_ids = fatal)")
        if not self.study_id:
            raise IdentityError("CompositeKey.study_id is empty")
        if not self.item_variant_id:
            raise IdentityError("CompositeKey.item_variant_id is empty (use the canonical variant)")
        if self.cell_kind in BANK_KINDS and self.root_team_id:
            raise IdentityError(
                f"a {self.cell_kind.value} draw is not a member of a root team: root_team_id "
                f"must be '' (got {self.root_team_id!r}); banks are the authority, episodes reference them")
        if self.cell_kind in BANK_KINDS and self.phase != K.PHASE_BANK:
            raise IdentityError(f"a {self.cell_kind.value} row carries phase {K.PHASE_BANK!r}, "
                                f"got {self.phase!r}")
        if self.cell_kind in BANK_KINDS and self.round != 0:
            raise IdentityError("bank draws live at round 0")
        if self.cell_kind not in GENERATION_KINDS:
            raise IdentityError(f"{self.cell_kind.value} cells generate no decisions and carry no "
                                "composite key (only generation kinds do)")
        if not isinstance(self.receiver_model_hash, str) or not self.receiver_model_hash:
            raise IdentityError(
                f"CompositeKey.receiver_model_hash is empty for a {self.cell_kind.value} row: it is "
                "ids.model_hash(<the model that generates this decision>) for EVERY kind (spec 17.2 "
                "has no other model slot; two models' draws of one item must not share a decision_id)")
        if self.phase not in K.PHASES:
            raise IdentityError(f"unknown phase {self.phase!r}; expected one of {K.PHASES}")
        for name in ("replicate_id", "round", "agent_slot"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise IdentityError(f"CompositeKey.{name} must be a non-negative int, got {value!r}")
        for name in ("root_team_id", "protocol_hash", "receiver_model_hash", "budget_hash"):
            if not isinstance(getattr(self, name), str):
                raise IdentityError(f"CompositeKey.{name} must be a string ('' when absent)")

    def as_array(self) -> list[Any]:
        return [getattr(self, f.name) for f in fields(self)]

    def decision_id(self) -> str:
        return "d:" + identity.sha256_hex(identity.jcs(self.as_array()))[:40]


FIELD_ORDER: Final[tuple[str, ...]] = tuple(f.name for f in fields(CompositeKey))


# --------------------------------------------------------------------------- id builders


def episode_id(kind: CellKind | str, root_team_id: str, protocol_hash: str, model_hash: str,
               budget_hash: str, replicate: int) -> str:
    """``<kind>|<team12>|<cond12>|<model16>|<budget8>|rep<k>`` for EPISODE / RECEIVER kinds
    (``episode|...`` and ``receiver|...``; the kind is the first component, R4).

    A RECEIVER episode has no root team; it passes its fixed-packet panel id (the E2a
    ``packets:<register sha>``, the E2b ``mod97`` problem id) in the ``root_team_id``
    position; a short panel id is hashed to twelve characters (see :func:`_short`).
    ``budget_hash`` is the *episode-level* contract (``receiver_tokens["<B>/one_call"]``)
    for both receiver phases, so one E2b episode has one id.
    """
    kind = coerce_kind(kind)
    if kind in BANK_KINDS:
        raise IdentityError(f"{kind.value} is a bank kind: bank draws never carry an episode id "
                            "(R4, defect D-11); use bank_draw_id")
    if kind not in EPISODE_KINDS:
        raise IdentityError(f"{kind.value} does not run episodes")
    _token("root_team_id", root_team_id)
    _token("protocol_hash", protocol_hash)
    _token("budget_hash", budget_hash)
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise IdentityError(f"replicate must be a non-negative int, got {replicate!r}")
    return (f"{kind.value}|{_short(root_team_id, 12)}|{_short(protocol_hash, 12)}|"
            f"{_token('model_hash', model_hash)}|{_short(budget_hash, 8)}|rep{replicate}")


def bank_draw_id(kind: CellKind | str, source_item_id: str, item_variant_id: str,
                 model_hash: str, budget_hash: str, draw: int) -> str:
    """``<kind>|<item>|<variant>|<model16>|<budget8>|k<draw>`` for the three bank kinds (D15)."""
    kind = coerce_kind(kind)
    if kind not in BANK_KINDS:
        raise IdentityError(f"{kind.value} is not a bank kind: an episode row never carries a "
                            "bank draw id (R4, defect D-11); use episode_id")
    if isinstance(draw, bool) or not isinstance(draw, int) or draw < 0:
        raise IdentityError(f"draw must be a non-negative int, got {draw!r}")
    return (f"{kind.value}|{_token('source_item_id', source_item_id)}|"
            f"{_token('item_variant_id', item_variant_id)}|{_token('model_hash', model_hash)}|"
            f"{_short(_token('budget_hash', budget_hash), 8)}|k{draw}")


def audit_episode_id(pool_id: str, model_hash: str, budget_hash: str) -> str:
    """``selection_audit|<pool12>|<model16>|<budget8>``: the episode id of the audit rows of
    one sealed pool.  Audit rows never reuse the audited episode's id (every ``episode_id``
    maps to exactly one ``cell_kind``, asserted at materialize)."""
    _token("pool_id", pool_id)
    return (f"{CellKind.SELECTION_AUDIT.value}|{_short(pool_id, 12)}|{_token('model_hash', model_hash)}|"
            f"{_short(_token('budget_hash', budget_hash), 8)}")


def root_team_id(origin_kind: CellKind | str, source_item_id: str, item_variant_id: str,
                 member_generation_uids: Sequence[str], focal_slot: int | None, rule: str) -> str:
    """``team:`` + sha256(JCS([origin, item, variant, ordered members, focal, rule]))[:32].

    Content-addressed over the *ordered* member generation uids, so a re-run reproduces the
    id and a different slot order is a different team.
    """
    origin = coerce_kind(origin_kind)
    members = tuple(str(m) for m in member_generation_uids)
    if not members:
        raise IdentityError("a root team needs at least one member generation uid")
    if any(not m for m in members):
        raise IdentityError("a root team member generation uid is empty")
    if len(set(members)) != len(members):
        raise IdentityError("a root team never holds one generation record twice "
                            "(spec 8.2: four copies of one record are not four draws)")
    if focal_slot is not None and (isinstance(focal_slot, bool) or not isinstance(focal_slot, int)
                                   or not 0 <= focal_slot < len(members)):
        raise IdentityError(f"focal_slot {focal_slot!r} is outside 0..{len(members) - 1}")
    payload = [origin.value, _token("source_item_id", source_item_id),
               _token("item_variant_id", item_variant_id), list(members), focal_slot, str(rule)]
    return "team:" + identity.sha256_hex(identity.jcs(payload))[:32]


def seed_key_for(key: CompositeKey, *, split_role: str) -> SeedKey:
    """The ``study.types.SeedKey`` of one decision: condition-free by construction (D4).

    Every component is pinned so two producers cannot pick different strings and silently
    break byte-identical aliasing::

        source_id   = key.source_item_id
        split       = split_role                 (item_splits.split_role of the item)
        model_cell  = key.receiver_model_hash    (the generating model, already in the key)
        episode_rep = key.replicate_id;  actor_slot = key.agent_slot
        purpose     = key.phase;         step_slot  = key.round
        namespace   = "ND1/<cell_kind>/<item_variant_id>/<root_team_id>"

    The namespace carries the cell kind (R4) and the root team (paired arms of one team
    share seeds; different teams do not).  ``protocol_hash`` and ``budget_hash`` are absent
    on purpose: the condition enters the request only through the prompt bytes, the budget
    only through the decoding.
    """
    if not isinstance(split_role, str) or not split_role:
        raise IdentityError("split_role is empty")
    namespace = f"{K.SEED_NAMESPACE_PREFIX}/{key.cell_kind.value}/{key.item_variant_id}/{key.root_team_id}"
    return SeedKey(
        source_id=key.source_item_id,
        split=split_role,
        model_cell=key.receiver_model_hash,
        episode_rep=int(key.replicate_id),
        actor_slot=int(key.agent_slot),
        purpose=key.phase,
        step_slot=int(key.round),
        namespace=namespace,
    )


def generation_uid(physical_request_ids: Sequence[str]) -> str:
    """``g:`` + sha256(JCS(ids)) over every physical request of one draw, in phase order.

    Untruncated (plan §1.3 truncates ``decision_id`` and ``root_team_id`` explicitly and
    this one deliberately not).  A continued draw (two physical calls) is ONE sample: both
    ids go in, one uid comes out.
    """
    ids = tuple(str(r) for r in physical_request_ids)
    if not ids or any(not r for r in ids):
        raise IdentityError("generation_uid needs at least one non-empty physical request id")
    if len(set(ids)) != len(ids):
        raise IdentityError("a physical request id appears twice in one draw")
    return "g:" + identity.sha256_hex(identity.jcs(list(ids)))


def content_hash(final_bytes: bytes) -> str:
    """``c:`` + sha256 of the post-``</think>`` bytes.  May coincide across samples."""
    if not isinstance(final_bytes, (bytes, bytearray)):
        raise IdentityError("content_hash takes bytes")
    return "c:" + identity.sha256_hex(bytes(final_bytes))


_LABEL_KEYS: Final[frozenset[str]] = frozenset({"display_label", "legacy_alias", "equivalent_to",
                                                "label", "name"})


def condition_hash(identity_object: Mapping[str, Any]) -> str:
    """sha256(JCS(identity object)) of a condition: symbolic, team-independent, label-free.

    Refuses an object that still carries a label key, so two conditions with equal content
    and different names hash equal (IN1_OUT1 == PEER) and aliasing is mechanical.
    """
    if not isinstance(identity_object, Mapping) or not identity_object:
        raise IdentityError("condition_hash needs a non-empty mapping")
    leaked = sorted(set(identity_object) & _LABEL_KEYS)
    if leaked:
        raise IdentityError(f"condition identity object carries label keys {leaked}; "
                            "labels never enter a condition hash")
    return identity.sha256_hex(identity.jcs(dict(identity_object)))


def model_hash(model: K.ModelCell) -> str:
    """sha256(JCS([hf_id, revision, tokenizer_revision, adapter]))[:16]; needs pinned revisions."""
    if not model.pinned:
        raise IdentityError(f"model cell {model.name!r} has no pinned revision; lock-inputs first")
    return identity.sha256_hex(identity.jcs(
        [model.hf_id, model.revision, model.tokenizer_revision, model.adapter]))[:16]


def blind(study_seed: bytes, namespace: str, *parts: Any) -> bytes:
    """Correctness-blind HMAC ordering key (= ``study.identity.blind_order_key``)."""
    if namespace not in K.ORDER_NAMESPACES:
        raise IdentityError(f"unknown blind-order namespace {namespace!r}; "
                            f"declare it in contract.ORDER_NAMESPACES")
    return identity.blind_order_key(study_seed, namespace, *parts)


__all__ = [
    "BANK_KINDS", "CellKind", "CompositeKey", "EPISODE_KINDS", "FIELD_ORDER", "GENERATION_KINDS",
    "audit_episode_id", "bank_draw_id", "blind", "coerce_kind", "condition_hash", "content_hash", "episode_id",
    "generation_uid", "model_hash", "root_team_id", "seed_key_for",
]
