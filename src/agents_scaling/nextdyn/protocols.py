"""Conditions are data, not code branches (plan D7, §1.5; spec 6.3, 8.4, 9.1, 10.1-10.4, 11.1).

ARD36 spelled its arms as members of a closed ``Protocol`` enum and branched on them in the
episode loop, so "add a directed channel" meant editing the loop and "are these two arms the
same treatment?" had no mechanical answer.  Here a condition is a frozen record loaded from
``configs/nextdyn/protocols.yaml`` with two concrete kinds -- :class:`VisibilitySpec` for an
episode arm and :class:`ReceiverSpec` for a receiver assay -- both satisfying the
:class:`ConditionSpec` protocol.  The composite key's ``protocol_hash`` is
``ids.condition_hash(identity_object())`` for both, and the identity object excludes every
label, so:

* two arms that *are* the same treatment hash equal and the planner can reuse one for the
  other (IN1_OUT1's resolved graph is ``complete``, so it hashes equal to PEER; spec 10.1
  explicit reuse).  ``protocols.yaml`` must then declare the equivalence, and
  :func:`load_registry` refuses an undeclared collision or a declared one that does not hold;
* renaming an arm changes nothing about its identity, and a label can never leak into a seed,
  a prompt or a key (rule R5).

:func:`packet_plan` is the single function that turns a condition plus a round state into
"who sees what": resolve the graph, take each recipient's senders, order them with the
correctness-blind :func:`packet_order`, and emit one :class:`TransmissionIntent` per slot.
Filler intents occupy the slots the graph removed whenever the condition asks for
length-matched padding, so a ``_FILLED`` arm and its unfilled twin differ in packet
*content* and not in how many packet slots a recipient sees.

``packet_order`` is vendored from ``ard/protocols.py:241`` (parity test:
``test_vendored_packet_order_matches_ard_under_the_same_namespace``).  It is keyed on the
``pairing_id`` and deliberately **not** on the condition, so paired arms deliver the senders
they have in common in the same order and their prompts can be compared byte for byte
(spec 6.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Protocol, Sequence, runtime_checkable

import yaml

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import graphs, ids
from agents_scaling.nextdyn.errors import ConfigError, ContractViolation
from agents_scaling.nextdyn.graphs import Graph, GraphRef, GraphSchedule
from agents_scaling.study import identity

DEFAULT_PROTOCOLS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "configs" / "nextdyn" / "protocols.yaml")

EPISODES_KEY: Final[str] = "episodes"
RECEIVERS_KEY: Final[str] = "receivers"

#: The one admitted ``schedule`` value; asynchronous arrangements are out of scope for v1
#: and the field is reserved so adding one is a config change (plan §1.10).
SCHEDULE_SYNCHRONOUS: Final[str] = "synchronous"


class PacketContent(str, Enum):
    """What a delivered packet carries (spec 6.1, 8.6)."""

    FULL = "full"
    ANSWER_ONLY = "answer_only"
    NONE = "none"           # the E2a/E2b draft-only controls transmit no packet at all


class Filler(str, Enum):
    """Whether a removed packet slot is padded with inert length-matched bytes (spec 10.3)."""

    NONE = "none"
    INERT_LENGTH_MATCHED = "inert_length_matched"


class Template(str, Enum):
    """The prompt template an arm renders (``prompts.TEMPLATE_NAMES``)."""

    ROOT = "root"
    REVISION = "revision"
    RECEIVER_PLAN = "receiver_plan"
    RECEIVER_DECISION = "receiver_decision"
    AUDIT = "audit"


class Support(str, Enum):
    """Spec 9.1 endorsement cue for the correct candidate."""

    CORRECT_MINORITY = "1:4"
    CORRECT_MAJORITY = "4:1"
    HIDDEN = "hidden"


class DraftRole(str, Enum):
    """Spec 9.1 assigned working draft."""

    CORRECT = "correct"
    WRONG = "wrong"
    NONE = "none"


@runtime_checkable
class ConditionSpec(Protocol):
    """What every condition, of either kind, must be able to say about itself (plan D7)."""

    display_label: str
    legacy_alias: str | None
    equivalent_to: str | None

    def identity_object(self) -> dict[str, Any]: ...

    @property
    def condition_hash(self) -> str: ...


def _label_free(obj: Mapping[str, Any], label: str) -> dict[str, Any]:
    """The identity object, with the three label fields proven absent before it is hashed."""
    leaked = sorted(k for k in obj if k in ("display_label", "legacy_alias", "equivalent_to"))
    if leaked:
        raise ContractViolation(f"{label}: identity object carries label key(s) {leaked}")
    return dict(obj)


@dataclass(frozen=True)
class VisibilitySpec:
    """One episode arm: what an agent sees at a revision round, and in what shape."""

    own_draft: bool
    graph: GraphSchedule
    packet_content: PacketContent
    filler: Filler
    template: Template
    schedule: Literal["synchronous"]
    display_label: str
    legacy_alias: str | None = None
    equivalent_to: str | None = None
    enabled: bool = True          # WP2: a planning knob (configs consumer: plan), never identity

    def __post_init__(self) -> None:
        object.__setattr__(self, "packet_content", PacketContent(self.packet_content))
        object.__setattr__(self, "filler", Filler(self.filler))
        object.__setattr__(self, "template", Template(self.template))
        if self.schedule != SCHEDULE_SYNCHRONOUS:
            raise ContractViolation(
                f"{self.display_label}: schedule {self.schedule!r} is reserved; the only admitted "
                f"value in v1 is {SCHEDULE_SYNCHRONOUS!r}")
        if self.template not in (Template.ROOT, Template.REVISION):
            raise ContractViolation(f"{self.display_label}: an episode arm renders the root or "
                                    f"the revision template, not {self.template.value!r}")
        if self.template is Template.ROOT and self.own_draft:
            raise ContractViolation(f"{self.display_label}: the root template shows no own draft")
        if self.filler is Filler.INERT_LENGTH_MATCHED and self.packet_content is PacketContent.NONE:
            raise ContractViolation(f"{self.display_label}: nothing to length-match")

    def identity_object(self) -> dict[str, Any]:
        """Symbolic, team-independent, label-free (the ``protocol_hash`` pre-image).

        The graph enters through :func:`graphs.canonical_ref`, so two spellings of the same
        graph -- ``focal_channels(incoming, outgoing)`` and ``complete`` -- are one condition.
        """
        ref = graphs.canonical_ref(self.graph.ref)
        schedule = GraphSchedule(kind=self.graph.kind, ref=ref) if ref is not self.graph.ref \
            else self.graph
        return _label_free({
            "kind": EPISODES_KEY,
            "own_draft": bool(self.own_draft),
            "graph": schedule.identity_object(),
            "packet_content": self.packet_content.value,
            "filler": self.filler.value,
            "template": self.template.value,
            "schedule": self.schedule,
        }, self.display_label)

    @property
    def condition_hash(self) -> str:
        return ids.condition_hash(self.identity_object())


@dataclass(frozen=True)
class ReceiverSpec:
    """One receiver assay cell (spec 9.1 E2a one-call, 9.4 E2b two-phase)."""

    packet_content: PacketContent
    support: Support
    checker: bool
    draft_role: DraftRole
    phases: Literal[1, 2]
    templates: tuple[str, ...]
    display_label: str
    legacy_alias: str | None = None
    equivalent_to: str | None = None
    enabled: bool = True          # WP2: a planning knob (configs consumer: plan), never identity

    def __post_init__(self) -> None:
        object.__setattr__(self, "packet_content", PacketContent(self.packet_content))
        object.__setattr__(self, "support", Support(self.support))
        object.__setattr__(self, "draft_role", DraftRole(self.draft_role))
        object.__setattr__(self, "templates", tuple(str(t) for t in self.templates))
        if int(self.phases) not in (1, 2):
            raise ContractViolation(f"{self.display_label}: phases must be 1 or 2")
        object.__setattr__(self, "phases", int(self.phases))
        for name in self.templates:
            Template(name)
        if len(self.templates) != self.phases:
            raise ContractViolation(
                f"{self.display_label}: {self.phases} phase(s) need {self.phases} template(s), "
                f"got {self.templates}")
        if self.checker and self.phases != 2:
            raise ContractViolation(f"{self.display_label}: the checker is a two-phase tool "
                                    "(spec 9.4); a one-call assay cannot serve it")

    def identity_object(self) -> dict[str, Any]:
        return _label_free({
            "kind": RECEIVERS_KEY,
            "packet_content": self.packet_content.value,
            "support": self.support.value,
            "checker": bool(self.checker),
            "draft_role": self.draft_role.value,
            "phases": int(self.phases),
            "templates": list(self.templates),
        }, self.display_label)

    @property
    def condition_hash(self) -> str:
        return ids.condition_hash(self.identity_object())


@dataclass(frozen=True)
class Registry:
    """Both condition maps, already checked for alias and equivalence consistency."""

    episodes: Mapping[str, VisibilitySpec]
    receivers: Mapping[str, ReceiverSpec]

    def __post_init__(self) -> None:
        object.__setattr__(self, "episodes", dict(self.episodes))
        object.__setattr__(self, "receivers", dict(self.receivers))

    def condition(self, name: str) -> ConditionSpec:
        if name in self.episodes:
            return self.episodes[name]
        if name in self.receivers:
            return self.receivers[name]
        raise ConfigError(f"unknown condition {name!r}; declared: "
                          f"{sorted([*self.episodes, *self.receivers])}")

    def by_hash(self) -> dict[str, tuple[str, ...]]:
        """condition hash -> every name that hashes to it (an equivalence class)."""
        out: dict[str, list[str]] = {}
        for name, spec in {**self.episodes, **self.receivers}.items():
            out.setdefault(spec.condition_hash, []).append(name)
        return {h: tuple(sorted(names)) for h, names in out.items()}


# --------------------------------------------------------------------------- loading


def _require(block: Mapping[str, Any], name: str, keys: Sequence[str]) -> dict[str, Any]:
    unknown = sorted(set(block) - set(keys))
    if unknown:
        raise ConfigError(f"protocols.yaml: {name} carries unknown key(s) {unknown}")
    missing = sorted(set(keys) - set(block))
    if missing:
        raise ConfigError(f"protocols.yaml: {name} lacks key(s) {missing}")
    return dict(block)


_EPISODE_KEYS = ("own_draft", "graph", "packet_content", "filler", "template", "schedule",
                 "display_label", "legacy_alias", "equivalent_to", "enabled")
_RECEIVER_KEYS = ("packet_content", "support", "checker", "draft_role", "phases", "templates",
                  "display_label", "legacy_alias", "equivalent_to", "enabled")
_GRAPH_KEYS = ("kind", "ref")
_REF_KEYS = ("name", "params")


def _graph_schedule(name: str, block: Mapping[str, Any]) -> GraphSchedule:
    block = _require(block, f"{name}.graph", _GRAPH_KEYS)
    ref_block = _require(dict(block["ref"]), f"{name}.graph.ref", _REF_KEYS)
    ref = GraphRef(name=str(ref_block["name"]), params=dict(ref_block["params"] or {}))
    return GraphSchedule(kind=str(block["kind"]), ref=ref)


#: The builder a history-free arm must name: FRESH sees no own draft and no peer packet, so
#: its graph is ``empty`` and not merely an unused one (spec 6.3).
ROOT_TEMPLATE_GRAPH: Final[str] = "empty"


def _visibility(name: str, block: Mapping[str, Any]) -> VisibilitySpec:
    block = _require(block, name, _EPISODE_KEYS)
    spec = VisibilitySpec(
        own_draft=bool(block["own_draft"]),
        graph=_graph_schedule(name, dict(block["graph"])),
        packet_content=block["packet_content"],
        filler=block["filler"],
        template=block["template"],
        schedule=str(block["schedule"]),
        display_label=str(block["display_label"]),
        legacy_alias=block["legacy_alias"],
        equivalent_to=block["equivalent_to"],
        enabled=bool(block["enabled"]),
    )
    ref = spec.graph.ref
    if spec.template is Template.ROOT and ref.name != ROOT_TEMPLATE_GRAPH:
        raise ConfigError(
            f"{name}: an arm that renders the root template sees no history, so its graph must "
            f"be {ROOT_TEMPLATE_GRAPH!r}, not {ref.name!r}")
    return spec


def _receiver(name: str, block: Mapping[str, Any]) -> ReceiverSpec:
    block = _require(block, name, _RECEIVER_KEYS)
    return ReceiverSpec(
        packet_content=block["packet_content"],
        support=block["support"],
        checker=bool(block["checker"]),
        draft_role=block["draft_role"],
        phases=int(block["phases"]),
        templates=tuple(block["templates"]),
        display_label=str(block["display_label"]),
        legacy_alias=block["legacy_alias"],
        equivalent_to=block["equivalent_to"],
        enabled=bool(block["enabled"]),
    )


def load_registry(path: str | Path = DEFAULT_PROTOCOLS_PATH) -> Registry:
    """Load and *check* ``protocols.yaml``: labels, aliases and declared equivalences.

    Checks, all of them refusals rather than warnings:

    1. every ``display_label`` equals its key (a condition has one name);
    2. no ``legacy_alias`` is used twice, or collides with a condition name, across both kinds
       (ARD36's ``R_PRIVATE``/``C_FULL``/``FRESH_N`` names must map one-to-one);
    3. every declared ``equivalent_to`` names an existing condition of the same kind and the
       two really do hash equal;
    4. every set of conditions that hashes equal is declared -- an undeclared collision means
       two "different" arms are one treatment and the planner would dispatch both.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} is not a mapping")
    unknown = sorted(set(data) - {EPISODES_KEY, RECEIVERS_KEY})
    if unknown:
        raise ConfigError(f"{path} carries unknown top-level key(s) {unknown}")
    episodes = {name: _visibility(name, dict(block))
                for name, block in dict(data.get(EPISODES_KEY) or {}).items()}
    receivers = {name: _receiver(name, dict(block))
                 for name, block in dict(data.get(RECEIVERS_KEY) or {}).items()}
    registry = Registry(episodes=episodes, receivers=receivers)
    _assert_labels(registry)
    _assert_equivalences(registry)
    return registry


def _assert_labels(registry: Registry) -> None:
    names = {**registry.episodes, **registry.receivers}
    for name, spec in names.items():
        if spec.display_label != name:
            raise ConfigError(f"condition {name!r} carries display_label "
                              f"{spec.display_label!r}; a condition has one name")
    seen: dict[str, str] = {}
    for name, spec in names.items():
        alias = spec.legacy_alias
        if alias is None:
            continue
        if alias in names:
            raise ConfigError(f"legacy_alias {alias!r} of {name!r} is also a condition name")
        if alias in seen:
            raise ConfigError(f"legacy_alias {alias!r} is claimed by {seen[alias]!r} and {name!r}")
        seen[alias] = name


def _assert_equivalences(registry: Registry) -> None:
    for kind, specs in ((EPISODES_KEY, registry.episodes), (RECEIVERS_KEY, registry.receivers)):
        for name, spec in specs.items():
            target = spec.equivalent_to
            if target is None:
                continue
            if target not in specs:
                raise ConfigError(f"{name}.equivalent_to names {target!r}, which is not an "
                                  f"{kind} condition")
            if specs[target].condition_hash != spec.condition_hash:
                raise ConfigError(
                    f"{name} declares equivalence to {target} but their condition hashes differ "
                    f"({spec.condition_hash[:12]} vs {specs[target].condition_hash[:12]}): the "
                    "declaration is false, or the identity objects have drifted")
    declared: dict[str, set[str]] = {}
    for name, spec in {**registry.episodes, **registry.receivers}.items():
        if spec.equivalent_to is not None:
            declared.setdefault(spec.condition_hash, set()).update({name, spec.equivalent_to})
    for condition_hash, names in registry.by_hash().items():
        if len(names) > 1 and set(names) - declared.get(condition_hash, set()):
            raise ConfigError(
                f"conditions {list(names)} share condition hash {condition_hash[:12]} but the "
                "equivalence is not declared: they are one treatment and the planner would "
                "dispatch both (spec 10.1 explicit reuse)")


def assert_agrees_with_study(protocol_block: Any, registry: Registry) -> None:
    """Cross-check ``study.yaml``'s protocol block against the registry (R1, one fact).

    ``study.protocol.own_previous_public_draft`` is the study-level statement that a revising
    agent sees its own previous public draft; every arm that renders the revision template
    must agree with it, and every arm named in ``study.protocol.protocols`` must exist here.
    """
    own = bool(protocol_block.own_previous_public_draft)
    for name, spec in registry.episodes.items():
        if spec.template is Template.REVISION and bool(spec.own_draft) != own:
            raise ContractViolation(
                f"{name}.own_draft={spec.own_draft} contradicts "
                f"study.protocol.own_previous_public_draft={own}")
    missing = [p for p in protocol_block.protocols if p not in registry.episodes]
    if missing:
        raise ContractViolation(f"study.protocol.protocols names unknown conditions {missing}")


# --------------------------------------------------------------------------- packet plan


@dataclass(frozen=True)
class TransmissionIntent:
    """One packet slot a recipient will see: who it is from, and from which round."""

    sender_slot: int
    source_round: int
    source_decision_id: str
    content: PacketContent
    filler: bool
    display_position: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", PacketContent(self.content))
        if self.filler and self.source_decision_id:
            raise ContractViolation("a filler slot has no source decision")


#: recipient slot -> its ordered intents (an empty tuple renders the literal ``NONE``).
PacketPlan = Mapping[int, tuple[TransmissionIntent, ...]]


def packet_order(study_seed: bytes, pairing_id: str, round_: int, recipient: int,
                 senders: Sequence[int]) -> tuple[int, ...]:
    """Vendored from ``ard/protocols.py:241``: the blind delivery order of one recipient.

    Keyed on ``(pairing_id, round, recipient, sender)`` and deliberately **not** on the
    condition, so paired arms deliver their common senders in the same order and their
    prompts stay byte-comparable (spec 6.4).  Vendored rather than imported because the ARD36
    original takes an ``ard`` ``RoundState`` and its own HMAC namespace; the parity test runs
    both under one namespace and asserts the same permutation.
    """
    return tuple(sorted(
        (int(s) for s in senders),
        key=lambda sender: identity.blind_order_key(
            study_seed, K.ORDER_PACKET, str(pairing_id), int(round_), int(recipient), str(sender)),
    ))


def packet_plan(state: Any, spec: VisibilitySpec, *, round_: int, study_seed: bytes) -> PacketPlan:
    """Who sees what at ``round_``: the one function that turns a condition into intents.

    ``state`` is the **immutable** pre-round snapshot (``states.RoundState``); every intent
    therefore points at a round ``round_ - 1`` decision, which ``episode`` asserts again
    before dispatch (spec 6.4: a faster agent's new output cannot enter another agent's
    same-round input).

    Display order is drawn over *all* other slots, not only the connected ones, so a filler
    slot lands where its missing packet would have landed and a ``_FILLED`` arm is
    position-for-position comparable with its unfilled twin.
    """
    n = int(state.n)
    graph: Graph | None = state.graph
    if graph is None:
        graph = graphs.resolve_graph(spec.graph, n=n, round_=int(round_), focal_slot=None,
                                     study_seed=study_seed, pairing_id=str(state.pairing_id))
    if len(graph) != n:
        raise ContractViolation(f"graph is {len(graph)}x{len(graph)} but the state holds {n} agents")
    source_round = int(round_) - 1
    if source_round != int(state.round_):
        raise ContractViolation(
            f"round {round_} must be planned from the round {source_round} snapshot, "
            f"got a round {state.round_} state (spec 6.4)")
    plan: dict[int, tuple[TransmissionIntent, ...]] = {}
    pad = spec.filler is Filler.INERT_LENGTH_MATCHED
    for recipient in range(n):
        others = [j for j in range(n) if j != recipient]
        connected = set(graphs.senders_for(graph, recipient))
        ordered = packet_order(study_seed, str(state.pairing_id), int(round_), recipient, others)
        intents: list[TransmissionIntent] = []
        for sender in ordered:
            is_edge = sender in connected
            if not is_edge and not pad:
                continue
            intents.append(TransmissionIntent(
                sender_slot=sender,
                source_round=source_round,
                source_decision_id=state.agents[sender].latest.decision_id if is_edge else "",
                content=spec.packet_content,
                filler=not is_edge,
                display_position=len(intents),
            ))
        plan[recipient] = tuple(intents)
    return plan


def assert_packet_rounds(intents: Sequence[TransmissionIntent], *, round_: int) -> None:
    """Vendored from ``ard/episode.py:489``, parametrically: every packet is from ``round-1``.

    Spec 6.4's synchronous rule, asserted at dispatch time rather than trusted: ARD36 needed
    the same assertion because its HUB5 arm was the single exception, and nextdyn has none.
    """
    allowed = int(round_) - 1
    for intent in intents:
        if int(intent.source_round) != allowed:
            raise ContractViolation(
                f"round {round_} received a round-{intent.source_round} packet from slot "
                f"{intent.sender_slot}; only round {allowed} packets may be read (spec 6.4)")


__all__ = [
    "ConditionSpec", "DEFAULT_PROTOCOLS_PATH", "DraftRole", "EPISODES_KEY", "Filler",
    "PacketContent", "PacketPlan", "RECEIVERS_KEY", "ReceiverSpec", "Registry",
    "SCHEDULE_SYNCHRONOUS", "Support", "Template", "TransmissionIntent",
    "assert_agrees_with_study", "assert_packet_rounds", "load_registry", "packet_order",
    "packet_plan",
]
