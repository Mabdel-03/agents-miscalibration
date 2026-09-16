"""The synchronous episode loop: one round is one immutable snapshot and one batch (spec 6.4).

Everything the loop must guarantee is asserted here rather than trusted, because each of
these was an ARD36 defect or a near miss:

* **Every prompt of a round is rendered before any of them is dispatched.**  The batch is
  then exchangeable by construction -- no agent can observe another agent's round-``r``
  output -- which is why running the ``n`` calls concurrently is result-identical and not
  merely faster (``ard.episode.dispatch_batch``, imported, ``max_workers=n``).
* **Every packet a round reads is from ``round - 1``** (``protocols.assert_packet_rounds``,
  vendored from ``ard/episode.py:489`` without ARD36's HUB5 exception, since nextdyn has no
  same-round arm).
* **A round advances only when all ``n`` slots are in.**  A build-time context refusal is a
  *typed context-failure decision* that spends the slot's planned opportunity -- never a
  retry, never a second sample (spec 15.3, ``failure_policy.context_overflow``).
* **Round 0 is a reference, not a generation.**  The roots come from a sealed, label-free
  bundle, so every protocol fork of one team reads the same bytes and
  ``test_paired_protocols_have_byte_identical_roots`` holds by construction rather than by
  luck (plan D14).
* **Exactly ``n * R`` revision decisions, no early stop.**  ``study.protocol.
  early_stop_on_consensus`` is false and unanimity does not end an episode: all agents update
  after unanimity (spec 8.4), which the loop makes structurally impossible to get wrong.
* **Historical coverage is nondecreasing** -- a state can lose a class from the current round
  but never from its history (spec 18.3).

The loop is n- and R-parametric end to end: ``n`` is read from the root bundle, ``R`` from the
episode spec, and no module constant encodes either.

Spec: 6.2, 6.3, 6.4, 8.1, 8.4, 11.2, 15.3.  Plan: §1.7, D4, D6, D14.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Final, Mapping

from agents_scaling.ard.episode import dispatch_batch  # allowlisted (plan §1.11)
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import graphs, ids, packets as PK, prompts, protocols as P
from agents_scaling.nextdyn.errors import ContractViolation, ManifestError
from agents_scaling.nextdyn.ids import CellKind, CompositeKey
from agents_scaling.nextdyn.packets import TransmittedPacket
from agents_scaling.nextdyn.protocols import PacketPlan, Template, TransmissionIntent, VisibilitySpec
from agents_scaling.nextdyn.schema import (
    ParseStatus,
    PublicOutput,
    context_failure_output,
    parse_public_output,
)
from agents_scaling.nextdyn.states import (
    Classifier,
    DecisionRef,
    RoundState,
    advance,
    initial_state,
    pairing_id_for,
    state_id,
)
from agents_scaling.serving.context import ContextCapacityError
from agents_scaling.study.packets import make_counter

#: ``generate(request) -> DecisionRecord`` (WP1 binds the store, the pool and the cost function).
Generate = Callable[[Any], Any]

#: The request fields ``inference.DecisionRequest`` declares (contract §2.1).  While WP1 is
#: pending, :func:`request_for` builds :class:`PendingDecisionRequest` with the same field
#: names so the loop is testable; once ``nextdyn.inference`` imports, the real class is used
#: and this bridge is never constructed (``decision_request_type``).
REQUEST_FIELDS: Final[tuple[str, ...]] = (
    "messages", "model", "adapter", "tokens", "seed_key", "purpose", "top_logprobs", "key")


@dataclass(frozen=True)
class PendingDecisionRequest:
    """WP1 bridge: the same eight fields ``inference.DecisionRequest`` declares."""

    messages: tuple[dict[str, str], ...]
    model: Any
    adapter: str
    tokens: K.TokenContract
    seed_key: Any
    purpose: str
    top_logprobs: int
    key: CompositeKey


def decision_request_type() -> type:
    """``inference.DecisionRequest`` when WP1 has landed, else the bridge above."""
    try:
        from agents_scaling.nextdyn.inference import DecisionRequest
    except ImportError:
        return PendingDecisionRequest
    return DecisionRequest


def is_context_refusal(exc: BaseException) -> bool:
    """Is this exception the typed build-time context refusal (spec 15.3)?

    Typed, not name-matched: either the serving layer's :class:`ContextCapacityError` (what
    ``serving.context.preflight_chat_context`` raises) or an exception that carries the
    ``CONTEXT_FAILURE`` parse status, which is how ``inference.ContextTooLong`` labels itself.
    A context refusal is never retried and never resampled.
    """
    if isinstance(exc, ContextCapacityError):
        return True
    status = getattr(exc, "parse_status", None)
    return getattr(status, "value", status) == ParseStatus.CONTEXT_FAILURE.value


# --------------------------------------------------------------------------- the spec


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode's frozen identity: item, roots, condition, model, budget, replicate."""

    cell_kind: CellKind
    source_item: Any                 # WP3 SourceItem: rendered_question, source_item_id, variant
    roots: Any                       # WP3 RootBundle: root_team_id, n, slots, export_hash, focal
    visibility: VisibilitySpec
    model: K.ModelCell
    tokens: K.TokenContract
    replicate: int
    revisions: int
    n: int
    split_role: str
    study_id: str                    # WP2 append: the composite key's first field
    root_round: int                  # WP2 append: study.protocol.root_round
    top_logprobs: int                # WP2 append: the request's logprob width
    family: Any = None               # WP3 FamilySpec (answer_instruction), FRESH / root renders

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_kind", ids.coerce_kind(self.cell_kind))
        if self.cell_kind is not CellKind.EPISODE:
            raise ContractViolation(f"run_episode drives {CellKind.EPISODE.value} cells, "
                                    f"not {self.cell_kind.value}")
        if int(self.n) != int(self.roots.n):
            raise ContractViolation(
                f"EpisodeSpec.n={self.n} but the root bundle holds {self.roots.n} slots")
        if int(self.revisions) < 1:
            raise ContractViolation("an episode runs at least one revision round")

    @property
    def model_hash(self) -> str:
        return ids.model_hash(self.model)

    @property
    def budget_hash(self) -> str:
        return self.tokens.budget_hash()

    @property
    def protocol_hash(self) -> str:
        return self.visibility.condition_hash

    @property
    def pairing_id(self) -> str:
        """Shared by every protocol fork of this team (no condition; spec 6.4)."""
        return pairing_id_for(self.source_item.source_item_id, self.roots.root_team_id,
                              self.model_hash, self.budget_hash, self.replicate)

    @property
    def episode_id(self) -> str:
        return ids.episode_id(self.cell_kind, self.roots.root_team_id, self.protocol_hash,
                              self.model_hash, self.budget_hash, self.replicate)

    def key_for(self, *, round_: int, agent_slot: int, phase: str) -> CompositeKey:
        return CompositeKey(
            study_id=self.study_id, cell_kind=self.cell_kind,
            source_item_id=self.source_item.source_item_id,
            item_variant_id=self.source_item.item_variant_id,
            root_team_id=self.roots.root_team_id, protocol_hash=self.protocol_hash,
            receiver_model_hash=self.model_hash, budget_hash=self.budget_hash,
            replicate_id=int(self.replicate), round=int(round_), agent_slot=int(agent_slot),
            phase=phase)


@dataclass(frozen=True)
class Decision:
    """One planned observation of one slot at one round, generated or referenced."""

    key: CompositeKey
    decision_id: str
    round_: int
    agent_slot: int
    phase: str
    output: PublicOutput
    record: Any
    messages: tuple[dict[str, str], ...]
    intents: tuple[TransmissionIntent, ...]
    parent_decision_ids: tuple[str, ...]
    parent_state_id: str
    is_reference: bool
    reuse_of: str | None
    generation_uid: str | None
    context_failure: bool

    @property
    def aliased(self) -> bool:
        return bool(getattr(self.record, "aliased", False))


@dataclass(frozen=True)
class Episode:
    """Everything one episode produced: states, decisions, packets and resolved graphs."""

    spec: EpisodeSpec
    episode_id: str
    states: tuple[RoundState, ...]
    decisions: tuple[Decision, ...]
    packets: Mapping[int, Mapping[int, TransmittedPacket]]   # round -> sender slot -> packet
    plans: Mapping[int, PacketPlan]                          # round -> plan
    graphs_by_round: Mapping[int, graphs.Graph]

    @property
    def references(self) -> tuple[Decision, ...]:
        return tuple(d for d in self.decisions if d.is_reference)

    @property
    def revisions(self) -> tuple[Decision, ...]:
        return tuple(d for d in self.decisions if not d.is_reference)


# --------------------------------------------------------------------------- rendering


def _root_refs(spec: EpisodeSpec) -> tuple[DecisionRef, ...]:
    """Round-0 references: a root is a bank draw, referenced, never regenerated (plan D14)."""
    refs = []
    for slot, root in enumerate(spec.roots.slots):
        key = spec.key_for(round_=spec.root_round, agent_slot=slot, phase=K.PHASE_ROOT_REF)
        refs.append(DecisionRef(decision_id=key.decision_id(), round_=int(spec.root_round),
                                agent_slot=slot, generation_uid=root.generation_uid,
                                output=root.output))
    return tuple(refs)


def render_round_packets(spec: EpisodeSpec, snapshot: RoundState, plan: PacketPlan, *,
                         counter: Callable[[str], int], counter_tokenizer_hash: str,
                         study_seed: bytes, round_: int) -> dict[int, TransmittedPacket]:
    """Render each sender's packet **once** per round, shared by every recipient (spec 6.1).

    Filler packets are length-matched against the very packet the graph removed, counted in
    the same tokenizer, so an ``_FILLED`` arm differs from its unfilled twin in bytes and not
    in packet-slot count.
    """
    content = spec.visibility.packet_content
    real: dict[int, TransmittedPacket] = {}
    for agent in snapshot.agents:
        out = agent.latest.output
        if out is None:
            raise ContractViolation(f"slot {agent.slot} has no public output at round "
                                    f"{snapshot.round_}; a round advances with all n slots")
        real[agent.slot] = PK.render_packet(
            out, pseudonym=agent.pseudonym, counter=counter,
            counter_tokenizer_hash=counter_tokenizer_hash, cap=spec.tokens.packet_cap,
            content=content)
    needs_filler = sorted({i.sender_slot for intents in plan.values() for i in intents if i.filler})
    out_packets = dict(real)
    for sender in needs_filler:
        seed = ids.blind(study_seed, K.ORDER_PACKET, snapshot.pairing_id, int(round_), int(sender))
        out_packets[sender] = PK.inert_filler(
            real[sender].token_count, seed, counter,
            pseudonym=snapshot.agents[sender].pseudonym,
            counter_tokenizer_hash=counter_tokenizer_hash)
    return out_packets


def messages_for(spec: EpisodeSpec, snapshot: RoundState, slot: int, plan: PacketPlan,
                 packets: Mapping[int, TransmittedPacket]) -> tuple[dict[str, str], ...]:
    """The chat messages of one slot's revision request, from the snapshot alone.

    FRESH renders the **root** template and sees neither its own nor any peer history
    (spec 6.3); every other arm renders the revision template, whose ``OTHER DRAFTS`` block
    is the only thing that differs between conditions -- which is why the isolated focal slot
    of IN0_OUT1 and the same slot of PRIVATE produce byte-identical prompts.
    """
    if spec.visibility.template is Template.ROOT:
        return prompts.render_root(spec.source_item, spec.family)
    own = packets[slot].serialized if spec.visibility.own_draft else None
    peers = [packets[intent.sender_slot].serialized for intent in plan.get(slot, ())]
    return prompts.render_revision(spec.source_item, own, peers)


def request_for(spec: EpisodeSpec, snapshot: RoundState, slot: int, round_: int,
                plan: PacketPlan, packets: Mapping[int, TransmittedPacket]) -> Any:
    """One slot's request for one round: messages, key, and a condition-free seed key (D4)."""
    key = spec.key_for(round_=round_, agent_slot=slot, phase=K.PHASE_REVISION)
    return decision_request_type()(
        messages=messages_for(spec, snapshot, slot, plan, packets),
        model=spec.model,
        adapter=spec.model.adapter,
        tokens=spec.tokens,
        seed_key=ids.seed_key_for(key, split_role=spec.split_role),
        purpose=K.PHASE_REVISION,
        top_logprobs=int(spec.top_logprobs),
        key=key,
    )


# --------------------------------------------------------------------------- the loop


def execute_updates(spec: EpisodeSpec, snapshot: RoundState, plan: PacketPlan,
                    packets: Mapping[int, TransmittedPacket], *, generate: Generate,
                    round_: int, counter: Callable[[str], int]) -> tuple[Decision, ...]:
    """Vendored parametrically from ``ard/episode.py:504``: one synchronous batch of updates.

    Render every prompt from the **immutable** snapshot first, assert each recipient's packet
    rounds, then dispatch the whole batch at ``max_workers = n``.  A context refusal is
    returned in place of a record so one refused slot does not cancel its round.
    """
    n = snapshot.n
    planned: list[tuple[int, Any]] = []
    for slot in range(n):
        P.assert_packet_rounds(plan.get(slot, ()), round_=round_)
        planned.append((slot, request_for(spec, snapshot, slot, round_, plan, packets)))
    outcomes = dispatch_batch([req for _, req in planned], _guarded(generate), max_workers=n)
    parent = state_id(snapshot)
    decisions: list[Decision] = []
    for (slot, request), outcome in zip(planned, outcomes):
        key = request.key
        refused = isinstance(outcome, BaseException)
        if refused:
            output, record, generation_uid = context_failure_output(), None, None
        else:
            record = outcome
            output = parse_public_output(record.final_text, record.finish_reason,
                                         record.completion_token_ids, counter,
                                         limits=spec.tokens)
            generation_uid = record.generation_uid
        decisions.append(Decision(
            key=key, decision_id=key.decision_id(), round_=int(round_), agent_slot=slot,
            phase=K.PHASE_REVISION, output=output, record=record, messages=tuple(request.messages),
            intents=tuple(plan.get(slot, ())), parent_state_id=parent,
            parent_decision_ids=_parents(snapshot, slot, plan),
            is_reference=False, reuse_of=None, generation_uid=generation_uid,
            context_failure=refused))
    return tuple(decisions)


def _guarded(generate: Generate) -> Generate:
    """Turn a typed context refusal into a returned value; every other error still raises."""

    def one(request: Any) -> Any:
        try:
            return generate(request)
        except BaseException as exc:  # noqa: BLE001 - re-raised unless it is the typed refusal
            if is_context_refusal(exc):
                return exc
            raise

    return one


def _parents(snapshot: RoundState, slot: int, plan: PacketPlan) -> tuple[str, ...]:
    """The decisions this one was built from: the agent's own previous draft and its senders."""
    own = snapshot.agents[slot].latest.decision_id
    peers = tuple(i.source_decision_id for i in plan.get(slot, ()) if i.source_decision_id)
    return (own, *peers)


def assert_team_shape(spec: EpisodeSpec, contract: K.StudyContract) -> None:
    """The episode's shape must be the study's declared team shape (R1, one fact).

    ``TeamShape.n_agents`` and ``TeamShape.revision_rounds`` are compiled from
    ``study.protocol`` and are the only declaration of N and R; an episode spec that
    disagrees with them is a planner defect, not a per-cell override (an E3b block that runs
    a different N declares it in its own block config, which recompiles the contract).
    """
    team = contract.team
    if int(spec.n) != int(team.n_agents):
        raise ContractViolation(
            f"episode runs {spec.n} agents but the contract's team shape declares "
            f"{team.n_agents} (study.protocol.n_agents)")
    if int(spec.revisions) != int(team.revision_rounds):
        raise ContractViolation(
            f"episode runs {spec.revisions} revision rounds but the contract declares "
            f"{team.revision_rounds} (study.protocol.revision_rounds)")


def run_episode(spec: EpisodeSpec, generate: Generate, *, tokenizer: Any, study_seed: bytes,
                classify: Classifier | None = None, on_event: Callable[..., None] | None = None
                ) -> Episode:
    """Run one episode: round 0 references, then ``R`` synchronous revision rounds.

    ``classify`` is the gold-independent canonicalizer used for coverage and unanimity
    bookkeeping; when it is omitted the raw final answer is its own class, which is enough for
    the structural assertions and keeps the loop free of any family knowledge (R7).
    """
    counter = make_counter(tokenizer)
    counter_hash = PK.tokenizer_hash(spec.model)
    classify = classify or (lambda answer: answer)
    refs = _root_refs(spec)
    state = initial_state(pairing_id=spec.pairing_id, roots=refs, classify=classify,
                          study_seed=study_seed)
    decisions: list[Decision] = [
        Decision(key=spec.key_for(round_=spec.root_round, agent_slot=ref.agent_slot,
                                  phase=K.PHASE_ROOT_REF),
                 decision_id=ref.decision_id, round_=int(spec.root_round),
                 agent_slot=ref.agent_slot, phase=K.PHASE_ROOT_REF, output=ref.output,
                 # a root is a *reference* to a bank draw: the draw's own record travels on
                 # the label-free bundle slot, so the reference row is built from the same
                 # physical sample the bank wrote (plan D14) and never from a new one.
                 record=getattr(spec.roots.slots[ref.agent_slot], "record", None),
                 messages=(), intents=(), parent_decision_ids=(),
                 parent_state_id="", is_reference=True,
                 reuse_of=spec.roots.slots[ref.agent_slot].decision_id,
                 generation_uid=ref.generation_uid, context_failure=False)
        for ref in refs]
    states = [state]
    all_packets: dict[int, Mapping[int, TransmittedPacket]] = {}
    plans: dict[int, PacketPlan] = {}
    resolved: dict[int, graphs.Graph] = {}
    coverage = len(state.historical_classes())

    for round_ in range(spec.root_round + 1, spec.root_round + int(spec.revisions) + 1):
        graph = graphs.resolve_graph(spec.visibility.graph, n=spec.n, round_=round_,
                                     focal_slot=spec.roots.focal_slot, study_seed=study_seed,
                                     pairing_id=spec.pairing_id)
        snapshot = dataclasses.replace(states[-1], graph=graph)
        plan = P.packet_plan(snapshot, spec.visibility, round_=round_, study_seed=study_seed)
        round_packets = render_round_packets(
            spec, snapshot, plan, counter=counter, counter_tokenizer_hash=counter_hash,
            study_seed=study_seed, round_=round_)
        round_decisions = execute_updates(spec, snapshot, plan, round_packets,
                                          generate=generate, round_=round_, counter=counter)
        refs_by_slot = {
            d.agent_slot: DecisionRef(decision_id=d.decision_id, round_=d.round_,
                                      agent_slot=d.agent_slot, generation_uid=d.generation_uid,
                                      output=d.output)
            for d in round_decisions}
        state = advance(snapshot, refs_by_slot, round_=round_)
        grown = len(state.historical_classes())
        if grown < coverage:
            raise ContractViolation(
                f"historical coverage fell from {coverage} to {grown} at round {round_}")
        coverage = grown
        states.append(state)
        decisions.extend(round_decisions)
        all_packets[round_] = round_packets
        plans[round_] = plan
        resolved[round_] = graph
        if on_event is not None:
            on_event(round_, state)

    expected = spec.n * int(spec.revisions)
    produced = sum(1 for d in decisions if not d.is_reference)
    if produced != expected:
        raise ContractViolation(
            f"episode produced {produced} revision decisions, expected n*R = {expected} "
            "(no early stop, no skipped slot)")
    if len(states) != int(spec.revisions) + 1:
        raise ContractViolation(
            f"{spec.revisions} revisions must leave {int(spec.revisions) + 1} observed states, "
            f"got {len(states)}")
    return Episode(spec=spec, episode_id=spec.episode_id, states=tuple(states),
                   decisions=tuple(decisions), packets=all_packets, plans=plans,
                   graphs_by_round=resolved)


# --------------------------------------------------------------------------- the cell worker


def episode_worker(runner: Any, source_item_id: str) -> Any:
    """The EPISODE cell worker (``kinds.KIND_TABLE``); the runner writes what it returns.

    The runner contract this worker reads (WP4 supplies it, WP2 states it):

    ``runner.cell_id``            the cell this item belongs to
    ``runner.study_seed``         the study seed bytes
    ``runner.generate``           ``Generate`` bound to the store, pool and cost function
    ``runner.episode_specs(item)``  the :class:`EpisodeSpec` objects planned for this item
    ``runner.tokenizer_for(model)`` the sender tokenizer of a model cell
    ``runner.classifier_for(spec)`` the gold-independent canonicalizer of the item's family

    Rows are built at their single sites: ``telemetry.decision_row`` for decisions (R6) and
    ``packets.transmissions_for`` for the delivered packet slots.
    """
    try:
        from agents_scaling.nextdyn import telemetry
    except ImportError as exc:                      # pragma: no cover - WP1 pending
        raise ManifestError(f"the EPISODE worker needs nextdyn.telemetry (WP1): {exc}") from exc
    from agents_scaling.nextdyn.tables import ItemFile, ItemStatus, SCHEMA_VERSION

    build = getattr(runner, "episode_specs", None)
    if build is None:                               # pragma: no cover - WP4 pending
        raise ManifestError("the cell runner does not expose episode_specs(source_item_id); "
                            "the EPISODE worker is driven by nextdyn.runner (WP4)")
    decision_rows: list[Any] = []
    transmission_rows: list[Any] = []
    status = ItemStatus.COMPLETE
    for spec in build(source_item_id):
        episode = run_episode(spec, runner.generate, tokenizer=runner.tokenizer_for(spec.model),
                              study_seed=runner.study_seed,
                              classify=runner.classifier_for(spec))
        cell = telemetry.CellContext(cell_id=runner.cell_id, episode_id=episode.episode_id,
                                     display_label=spec.visibility.display_label,
                                     bundle_export_hash=spec.roots.export_hash)
        for decision in episode.decisions:
            if decision.context_failure:
                status = ItemStatus.CONTEXT_FAILURE
            if decision.record is None:
                raise ManifestError(
                    f"decision {decision.decision_id} has no DecisionRecord: a root reference "
                    "needs the bank draw's record on its bundle slot, and a context-failure "
                    "row needs telemetry.decision_row to admit a recordless decision")
            seen = tuple(sorted(PK.transmission_ids(decision.decision_id, decision.intents)))
            decision_rows.append(telemetry.decision_row(
                decision.record, decision.key, decision.output, cell=cell,
                parents=decision.parent_decision_ids, seen_transmissions=seen,
                aliased=decision.aliased, reuse_of=decision.reuse_of,
                is_reference=decision.is_reference))
            if decision.is_reference:
                continue
            transmission_rows.extend(PK.transmissions_for(
                episode.plans[decision.round_], episode.packets[decision.round_],
                decision.decision_id, destination_round=decision.round_,
                receiver_slot=decision.agent_slot, episode_id=episode.episode_id,
                cell_kind=spec.cell_kind.value, protocol_hash=spec.protocol_hash))
    return ItemFile(schema_version=SCHEMA_VERSION, cell_id=runner.cell_id,
                    cell_kind=CellKind.EPISODE, source_item_id=source_item_id, status=status,
                    bundle_export_hash=None,
                    rows={"decisions": tuple(decision_rows),
                          "transmissions": tuple(transmission_rows)})


__all__ = [
    "Decision", "Episode", "EpisodeSpec", "Generate", "PendingDecisionRequest", "REQUEST_FIELDS",
    "decision_request_type", "episode_worker", "execute_updates", "is_context_refusal",
    "messages_for", "render_round_packets", "request_for", "run_episode",
]
