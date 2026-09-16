"""The runtime root-bundle reader: the only root input the generation lane sees (plan §2.4,
§1.3; spec 8.2, 8.6; rule R7).

A root bundle is the frozen output of a CONSTRUCT cell: the ordered slots of one team, each
carrying its generation uid, its decision id and its public output, plus the opaque
``team_tag``, the focal slot and the export hash.  What it deliberately does **not** carry is
the composition, ``C0``, ``w*``, per-slot correctness, any grade, any gold: those live in
``root_teams_private`` under ``protected/`` and never leave the writer side.

Two mechanisms, not one:

* ``assert_label_free`` walks the payload -- keys *and* string values -- against
  :data:`LABEL_DENYLIST` and :data:`LABEL_TOKENS`, so a future writer that adds
  ``"composition": "C1_COHERENT"`` or a slot named ``"correct_0"`` is refused at the door
  (``test_solver_export_contains_no_gold_or_construction_correctness``,
  ``test_candidate_ids_and_slot_labels_do_not_encode_correctness``);
* the file is written through ``deps.write_bundle``, which refuses a differing overwrite and
  keeps a ``.sha256`` sidecar, so every protocol fork of one team reads the same bytes and
  records the same ``export_hash`` (``test_paired_protocols_have_byte_identical_roots``).

Slots are numbered ``0..n-1`` in the frozen slot permutation of the team; the numbering
carries no information about correctness, and a bundle never says which slot is the minority
one except through ``focal_slot``, which E3a needs to address a channel and which is chosen
blind before the roots are drawn (spec 10.4).
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import deps
from agents_scaling.nextdyn import schema
from agents_scaling.nextdyn.errors import ContractViolation, LeakageError

BUNDLE_SCHEMA_VERSION: Final[int] = 1

#: The bundle document's shape is declared by :class:`RootBundle` / :class:`RootSlot` below and
#: nowhere else: these tuples are *derived* from those dataclasses at import time.  Writing the
#: field names out again here is the ARD36 D-05 shape (one fact, two declarations, nothing
#: reconciling them), and it is also what R1's column-list guard refuses.  ``teams`` is the
#: document's container for the per-team records, which the payload carries instead of the
#: identity fields a loaded :class:`RootBundle` resolves.
def _document_fields(row_type: type, *, drop: tuple[str, ...] = (), add: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The JSON field names of one bundle record, derived from its dataclass."""
    names = tuple(f.name for f in dataclasses.fields(row_type) if f.name not in drop)
    return names + tuple(a for a in add if a not in names)


#: Keys a label-free export may never carry (construction state and grades).
LABEL_DENYLIST: Final[frozenset[str]] = frozenset({
    "composition", "compositions", "c0", "w_star", "w_star_class", "correct_class", "wrong_class",
    "slot_states", "slot_state", "eligibility", "semantic_correct", "grade", "grades",
    "is_correct", "correct", "wrong", "label", "labels", "gold", "gold_ref_hash", "answer_key",
    "canonical_class", "extension_stage", "bank_role", "draw_index", "cohort", "difficulty",
    "prior_use_status", "u_reason", "grading_status",
})
#: Substrings a *value* may never contain: a composition name or a tri-state token would
#: encode the construction just as well as a key would.
LABEL_TOKENS: Final[tuple[str, ...]] = (
    "c1_coherent", "c4_coherent", "c1_fragmented", "c2_of_5", "c0_unanimous", "c5_unanimous",
    "c1_of_3", "c1_of_9", "c3_of_9", "w_star", "wstar", "semantic_correct", "is_correct",
)
#: The one admitted value of ``study.failure_policy.hidden_label_leakage`` (spec 15.3).
HIDDEN_LABEL_LEAKAGE: Final[str] = "fatal"


def assert_leakage_policy(policy: Any) -> None:
    """A run whose policy tolerates a label leak may not load a bundle at all."""
    if policy.hidden_label_leakage != HIDDEN_LABEL_LEAKAGE:
        raise LeakageError(
            f"study.failure_policy.hidden_label_leakage is {policy.hidden_label_leakage!r}; the "
            f"root-bundle reader only runs under {HIDDEN_LABEL_LEAKAGE!r} (spec 15.3)")


# --------------------------------------------------------------------------- the runtime view


@dataclass(frozen=True)
class RootSlot:
    """One team member as the episode loop sees it (``episode._root_refs``)."""

    slot: int
    generation_uid: str
    decision_id: str
    content_hash: str | None
    output: schema.PublicOutput


@dataclass(frozen=True)
class RootBundle:
    """One team, label-free.  ``EpisodeSpec.roots`` is exactly this object."""

    root_team_id: str
    team_tag: str
    family: str
    bucket: str
    source_item_id: str
    item_variant_id: str
    n: int
    slots: tuple[RootSlot, ...]
    export_hash: str
    focal_slot: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(self.slots))
        if len(self.slots) != int(self.n):
            raise ContractViolation(f"{self.root_team_id}: {len(self.slots)} slots for n={self.n}")
        if [s.slot for s in self.slots] != list(range(int(self.n))):
            raise ContractViolation(f"{self.root_team_id}: slots must be 0..{int(self.n) - 1} in order")
        uids = [s.generation_uid for s in self.slots]
        if len(set(uids)) != len(uids):
            raise ContractViolation(f"{self.root_team_id}: one generation record appears twice; "
                                    "four copies of one draw are not four draws (spec 8.2)")
        if self.focal_slot is not None and not 0 <= int(self.focal_slot) < int(self.n):
            raise ContractViolation(f"{self.root_team_id}: focal_slot {self.focal_slot} outside the team")


# --------------------------------------------------------------------------- public output io


def output_to_json(out: schema.PublicOutput) -> dict[str, Any]:
    """Serialize a public output without naming its fields (R1: ``schema.py`` declares them)."""
    payload: dict[str, Any] = {}
    for field in dataclasses.fields(out):
        value = getattr(out, field.name)
        if isinstance(value, Enum):
            value = value.value
        elif isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, Mapping):
            value = dict(value)
        payload[field.name] = value
    return payload


def output_from_json(data: Mapping[str, Any]) -> schema.PublicOutput:
    names = {field.name for field in dataclasses.fields(schema.PublicOutput)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ContractViolation(f"root bundle output carries unknown key(s) {unknown}")
    missing = sorted(names - set(data))
    if missing:
        raise ContractViolation(f"root bundle output lacks key(s) {missing}")
    return schema.PublicOutput(**dict(data))


# --------------------------------------------------------------------------- the firewall


def _walk(node: Any, path: str, out: list[str]) -> None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            lowered = str(key).casefold()
            if lowered in LABEL_DENYLIST:
                out.append(f"{path}.{key} (key)")
            _walk(value, f"{path}.{key}", out)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _walk(value, f"{path}[{index}]", out)
    elif isinstance(node, str):
        lowered = node.casefold()
        for token in LABEL_TOKENS:
            if token in lowered:
                out.append(f"{path} (value contains {token!r})")


#: Derived (see ``_document_fields``): the payload's top level, one team record, one slot record.
@dataclass(frozen=True)
class BundleDocument:
    """The envelope of one bucket's bundle file: the construction facts shared by its teams.

    A dataclass rather than a tuple of names for the same reason the tables are dataclasses --
    it is the one declaration of the document's shape, and ``BUNDLE_TOP_FIELDS`` is read off it.
    """

    schema_version: int
    family: str
    bucket: str
    construction_rule_version: str
    teams: tuple[Mapping[str, Any], ...]


BUNDLE_TOP_FIELDS: Final[tuple[str, ...]] = _document_fields(BundleDocument)
BUNDLE_TEAM_FIELDS: Final[tuple[str, ...]] = _document_fields(
    RootBundle, drop=("family", "bucket", "export_hash", "slots"), add=("slots",))
BUNDLE_SLOT_FIELDS: Final[tuple[str, ...]] = _document_fields(RootSlot)


def assert_label_free(bundle_json: Mapping[str, Any]) -> None:
    """Refuse a payload that carries any construction label, grade or gold (spec 8.2, R7)."""
    leaks: list[str] = []
    _walk(bundle_json, "bundle", leaks)
    if leaks:
        raise LeakageError("root bundle carries construction labels: " + "; ".join(sorted(leaks)))


# --------------------------------------------------------------------------- reading


def _team_from_payload(payload: Mapping[str, Any], team: Mapping[str, Any],
                       export_hash: str) -> RootBundle:
    unknown = sorted(set(team) - set(BUNDLE_TEAM_FIELDS))
    if unknown:
        raise ContractViolation(f"root bundle team carries unknown key(s) {unknown}")
    slots = []
    for entry in team["slots"]:
        extra = sorted(set(entry) - set(BUNDLE_SLOT_FIELDS))
        if extra:
            raise ContractViolation(f"root bundle slot carries unknown key(s) {extra}")
        slots.append(RootSlot(slot=int(entry["slot"]), generation_uid=str(entry["generation_uid"]),
                              decision_id=str(entry["decision_id"]),
                              content_hash=entry["content_hash"],
                              output=output_from_json(entry["output"])))
    return RootBundle(root_team_id=str(team["root_team_id"]), team_tag=str(team["team_tag"]),
                      family=str(payload["family"]), bucket=str(payload["bucket"]),
                      source_item_id=str(team["source_item_id"]),
                      item_variant_id=str(team["item_variant_id"]), n=int(team["n"]),
                      slots=tuple(slots), export_hash=export_hash,
                      focal_slot=None if team["focal_slot"] is None else int(team["focal_slot"]))


def load_bucket(run_root: str | os.PathLike, family: str, bucket: str | int) -> tuple[RootBundle, ...]:
    """Every team of one bucket, after the sidecar check and the label-free check."""
    payload, export_hash = deps.read_bundle(run_root, family, bucket)
    missing = sorted(set(BUNDLE_TOP_FIELDS) - set(payload))
    unknown = sorted(set(payload) - set(BUNDLE_TOP_FIELDS))
    if missing or unknown:
        raise ContractViolation(f"root bundle {family}/{bucket}: missing {missing}, unknown {unknown}")
    if int(payload["schema_version"]) != BUNDLE_SCHEMA_VERSION:
        raise ContractViolation(f"root bundle schema_version {payload['schema_version']} != "
                                f"{BUNDLE_SCHEMA_VERSION}")
    assert_label_free(payload)
    return tuple(_team_from_payload(payload, team, export_hash) for team in payload["teams"])


def load(run_root: str | os.PathLike, family: str, bucket: str | int,
         root_team_id: str | None = None) -> RootBundle:
    """One team's bundle.  ``root_team_id`` selects inside a bucket that holds a pair; a
    bucket with exactly one team needs no selector (the optional third argument is the WP3
    note appended to contract §2.3: a paired bucket holds both members)."""
    teams = load_bucket(run_root, family, bucket)
    if root_team_id is None:
        if len(teams) != 1:
            raise ContractViolation(
                f"bucket {family}/{bucket} holds {len(teams)} teams "
                f"({[t.root_team_id for t in teams]}); name the one you mean")
        return teams[0]
    for team in teams:
        if team.root_team_id == root_team_id:
            return team
    raise ContractViolation(f"bucket {family}/{bucket} has no team {root_team_id!r}")


def bundle_dependency(family: str, bucket: str | int) -> str:
    """The ``bundle:<family>/<bucket>`` dependency string of this bucket (one vocabulary, D16)."""
    return deps.bundle_dependency(family, bucket)


__all__ = [
    "BUNDLE_SCHEMA_VERSION", "HIDDEN_LABEL_LEAKAGE", "LABEL_DENYLIST", "LABEL_TOKENS",
    "BundleDocument", "RootBundle", "RootSlot", "BUNDLE_SLOT_FIELDS", "BUNDLE_TEAM_FIELDS", "BUNDLE_TOP_FIELDS", "assert_label_free",
    "assert_leakage_policy", "bundle_dependency", "load", "load_bucket", "output_from_json",
    "output_to_json",
]
