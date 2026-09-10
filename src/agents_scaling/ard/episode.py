"""ARD36 episode loop: three synchronous update rounds, no early stopping (WP2).

Spec §19 gives the loop as pseudocode and this module is its literal implementation::

    state = common_root_state
    for t in 1..3:
        packet_plan = build_packets(state, protocol)    # prior round only
        snapshot = immutable_copy(state)
        reports = execute_updates(snapshot, packet_plan)
        assert no_request_read_same_round_peers_unless_HUB5
        append_archive(reports)
        emit_metrics_inputs_without_reading_gold()
        state = advance(snapshot, reports)

Spec §7: "N=5 and rounds t=0,1,2,3 … Three update rounds are mandatory.  There is **no
early stopping on unanimity** in the measurement run; unanimous error is not an absorbing
state by assumption."  All five logical calls are issued in every round even when the team
already agrees — :class:`Episode` asserts ``N_AGENTS * rounds`` update decisions.

Three assertions run inside the loop, all of them gold-blind:

1. **A round-``t`` request reads only round ``t-1``** — every packet carries its
   ``source_round`` and the check is exact.  HUB5 is the single, declared exception: its
   four workers read the *current* leader report, the asynchrony spec §7 insists must be
   recorded rather than disguised.
2. **Exactly ``N × rounds`` decisions per episode**, unanimity or not.
3. **Archive coverage never falls** — the cumulative set of canonical classes ever
   produced is monotone by round.  Coverage here is the class multiset, not correctness:
   nothing in this module knows the gold answer.

Per-round selection (spec §7) is emitted over three declared pools — ``root``,
``current`` and ``archive`` — through :mod:`agents_scaling.ard.vote`, plus the mandatory
FRESH cumulative baseline at k = 5, 10, 15, 20.

Frozen interface: ``docs/ard36/00_contract.md`` §2 (WP2).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard import protocols as P
from agents_scaling.ard import vote as V
from agents_scaling.ard.client import ContextTooLong, DecisionRecord, DecisionRequest
from agents_scaling.ard.packets import ArdPacket, pseudonym_map
from agents_scaling.ard.report import Report, context_failure_report, parse_report
from agents_scaling.ard.states import AgentState, ReportRef, RoundState
from agents_scaling.study.types import ProtocolError

__all__ = [
    "Decision",
    "Episode",
    "EpisodeSpec",
    "PHASE_LEADER",
    "PHASE_ROOT",
    "PHASE_UPDATE",
    "PHASE_WORKER",
    "RoundResult",
    "execute_updates",
    "fresh_bank",
    "initial_state",
    "run_episode",
    "run_roots",
]

PHASE_ROOT = "root"
PHASE_UPDATE = "update"
PHASE_LEADER = "leader"
PHASE_WORKER = "worker"

#: ``generate(request) -> DecisionRecord`` — WP4 binds the store, pool and cost function.
Generate = Callable[[DecisionRequest], DecisionRecord]


# --------------------------------------------------------------------------- spec


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode's frozen identity (spec §18 ``episodes.parquet``).

    ``pairing_id`` is the identity **shared by paired arms** (item × checkpoint × cohort ×
    repeat, no protocol).  Every blind schedule — pseudonyms, packet order, BALANCED
    representative, HUB5 leader, protection choice — is keyed by it, so a GATE episode and
    its PEER twin randomize identically and their packets are comparable byte-for-byte
    (spec §12.3).  ``episode_id`` adds the protocol and is what the tables key on.
    """

    item_id: str
    source_id: str
    split: str
    family: str
    task_text: str
    protocol: K.Protocol
    checkpoint: str
    model_cell: str
    cohort: str = K.Cohort.NATURAL.value
    episode_rep: int = 0
    rounds: int = K.UPDATE_ROUNDS
    thinking_tokens: int = K.THINKING_TOKENS_LOW
    root_draws: tuple[int, ...] = K.TEAM_DRAWS

    def __post_init__(self) -> None:
        if len(self.root_draws) != K.N_AGENTS:
            raise ValueError(f"a team needs {K.N_AGENTS} root draws, got {len(self.root_draws)}")
        if not 1 <= int(self.rounds) <= K.UPDATE_ROUNDS:
            raise ValueError(f"rounds must be 1..{K.UPDATE_ROUNDS}, got {self.rounds}")

    @property
    def pairing_id(self) -> str:
        return f"{self.item_id}|{self.checkpoint}|{self.cohort}|e{int(self.episode_rep)}"

    @property
    def episode_id(self) -> str:
        return f"{self.pairing_id}|{K.Protocol(self.protocol).value}"

    def to_row(self, *, leader: int | None, root_decision_ids: Sequence[str]) -> dict[str, Any]:
        """The ``episodes.parquet`` row (spec §18)."""
        return {
            "episode_id": self.episode_id,
            "item_id": self.item_id,
            "cohort": self.cohort,
            "protocol": K.Protocol(self.protocol).value,
            "checkpoint": self.checkpoint,
            "checkpoint_revision": K.REVISIONS.get(self.checkpoint, ""),
            "root_candidate_ids": list(root_decision_ids),
            "root_draws": list(self.root_draws),
            "N": K.N_AGENTS,
            "horizon": int(self.rounds),
            "reasoning_allowance": int(self.thinking_tokens),
            "final_allowance": int(K.FINAL_TOKENS),
            "packet_policy": K.Protocol(self.protocol).value,
            "memory_policy": "task+root+latest+packet",
            "leader_id": leader,
            "episode_rep": int(self.episode_rep),
            "family": self.family,
            "split": self.split,
        }


# --------------------------------------------------------------------------- decisions


@dataclass(frozen=True)
class Decision:
    """One logical decision = one chat request (spec §18 ``decisions.parquet``)."""

    decision_id: str
    candidate_id: str
    episode_id: str
    agent_slot: int
    round_: int
    phase: str
    action: str
    parent_state_id: str
    packets: tuple[ArdPacket, ...]
    report: Report
    record: DecisionRecord | None
    messages: tuple[Mapping[str, str], ...]

    @property
    def seen_message_ids(self) -> tuple[str, ...]:
        return tuple(p.packet_id for p in self.packets)

    @property
    def message_order(self) -> tuple[str | None, ...]:
        return tuple(p.pseudonym for p in self.packets)

    @property
    def canonical_source_rounds(self) -> tuple[int, ...]:
        return tuple(p.source_round for p in self.packets)

    def to_row(self) -> dict[str, Any]:
        record = self.record
        row: dict[str, Any] = {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "episode_id": self.episode_id,
            "agent_id": self.agent_slot,
            "round": self.round_,
            "phase": self.phase,
            "action": self.action,
            "parent_state_id": self.parent_state_id,
            "seen_message_ids": list(self.seen_message_ids),
            "message_order": [p for p in self.message_order],
            **self.report.to_dict(),
        }
        if record is not None:
            row.update(
                {
                    "request_hash": record.request_id,
                    "thinking_tokens": record.thinking_tokens,
                    "final_tokens": record.final_tokens,
                    "prompt_tokens": record.prompt_tokens,
                    "cached_prefill_tokens": record.cached_prefill_tokens,
                    "cap_hit": record.cap_hit,
                    "boundary_injected": record.boundary_injected,
                    "answer_logprob": record.answer_logprob,
                    "latency": record.latency_s,
                    "endpoint_id": record.endpoint_id,
                    "flops": dict(record.flops),
                    "finish_reason": record.finish_reason,
                    "aliased": record.aliased,
                }
            )
        return row


@dataclass(frozen=True)
class RoundResult:
    """Everything one round produced, at the round grain of plan §5.8."""

    round_: int
    decisions: tuple[Decision, ...]
    state: RoundState
    selections: Mapping[str, V.ArdSelection]
    class_counts: Mapping[str, int]
    archive_classes: frozenset[str]
    protected_slot: int | None = None
    leader: int | None = None
    collapsed_unparsed: Mapping[int, int] = field(default_factory=dict)

    @property
    def coverage(self) -> int:
        """Current-round coverage: how many distinct canonical classes the team holds."""
        return len(self.class_counts)


@dataclass(frozen=True)
class Episode:
    """The finished episode: roots, three rounds, decisions and selections."""

    spec: EpisodeSpec
    roots: tuple[ReportRef, ...]
    rounds: tuple[RoundResult, ...]
    root_state: RoundState
    root_selections: Mapping[str, V.ArdSelection]
    leader: int | None = None
    pseudonyms: tuple[str, ...] = ()

    @property
    def decisions(self) -> tuple[Decision, ...]:
        return tuple(d for r in self.rounds for d in r.decisions)

    @property
    def final_state(self) -> RoundState:
        return self.rounds[-1].state if self.rounds else self.root_state

    def states(self) -> tuple[AgentState, ...]:
        out = list(self.root_state.agents)
        for result in self.rounds:
            out.extend(result.state.agents)
        return tuple(out)

    def to_row(self) -> dict[str, Any]:
        return self.spec.to_row(
            leader=self.leader, root_decision_ids=[ref.decision_id for ref in self.roots]
        )


# --------------------------------------------------------------------------- helpers


def _candidate_id(episode_id: str, round_: int, slot: int) -> str:
    """Pool candidate id: one *planned opportunity*, unique inside an episode (spec §3)."""
    return f"{episode_id}#r{int(round_)}a{int(slot)}"


def _vote_candidate(
    state: RoundState, ref: ReportRef, candidate_id: str
) -> V.VoteCandidate:
    return V.candidate_from_report(candidate_id, ref.report, state.classify)


def _selections(
    state: RoundState,
    *,
    source_id: str,
    root_refs: Sequence[ReportRef],
    archive: Sequence[tuple[int, ReportRef]],
    episode_id: str,
) -> dict[str, V.ArdSelection]:
    """Root, current and archive selections, reported separately (spec §7)."""
    root_pool = [
        _vote_candidate(state, ref, _candidate_id(episode_id, 0, ref.agent_slot)) for ref in root_refs
    ]
    current_pool = [
        _vote_candidate(state, state.latest(slot), _candidate_id(episode_id, state.round_, slot))
        for slot in range(K.N_AGENTS)
    ]
    archive_pool = [
        _vote_candidate(state, ref, _candidate_id(episode_id, round_, ref.agent_slot))
        for round_, ref in archive
    ]
    out: dict[str, V.ArdSelection] = {}
    for kind, pool in (("root", root_pool), ("current", current_pool), ("archive", archive_pool)):
        out[kind] = V.plurality(
            pool, source_id=source_id, study_seed=state.study_seed, pool_kind=kind
        )
        out[f"{kind}:confidence_weighted"] = V.confidence_weighted_plurality(
            pool, source_id=source_id, study_seed=state.study_seed, pool_kind=kind
        )
    return out


def _archive_pairs(state: RoundState) -> list[tuple[int, ReportRef]]:
    return [(round_, ref) for round_ in sorted(state.archive) for ref in state.archive[round_]]


# --------------------------------------------------------------------------- roots


def run_roots(
    spec: EpisodeSpec, *, generate: Generate, draws: Sequence[int] | None = None
) -> tuple[ReportRef, ...]:
    """Generate (or alias) the team's root reports — round 0 (spec §5.2, §7).

    The prompt is the neutral root prompt and the seed key is
    ``(actor_slot=0, purpose='root', step_slot=draw, ns='stateless_bank')``, so every
    protocol, every budget arm and every cohort that shares an item shares these records:
    the store returns them aliased and no GPU work is repeated (plan §5.1).
    """
    refs: list[ReportRef] = []
    for slot, draw in enumerate(draws if draws is not None else spec.root_draws):
        request = DecisionRequest(
            messages=P.render_root_prompt(spec.task_text),
            checkpoint=spec.checkpoint,
            thinking_tokens=K.ROOT_THINKING_TOKENS,
            seed_key=P.root_seed_key(
                source_id=spec.source_id,
                split=spec.split,
                model_cell=spec.model_cell,
                draw=int(draw),
                episode_rep=spec.episode_rep,
            ),
            purpose=K.PURPOSE_ROOT,
        )
        record = generate(request)
        report = parse_report(record.final_text, record.finish_reason, record.completion_token_ids)
        refs.append(ReportRef(decision_id=record.request_id, report=report, round_=0, agent_slot=slot))
    return tuple(refs)


def initial_state(
    spec: EpisodeSpec,
    roots: Sequence[ReportRef],
    *,
    study_seed: bytes,
    classify: Callable[[str | None], str | None],
    count: Callable[[str], int] | None = None,
    leader: int | None = None,
) -> RoundState:
    """The common neutral root team at ``t = 0`` (spec §7: shared by every protocol)."""
    pseudonyms = pseudonym_map(study_seed, spec.pairing_id)
    agents = tuple(
        AgentState(
            item_id=spec.item_id,
            checkpoint=spec.checkpoint,
            agent_slot=slot,
            round_=0,
            root=roots[slot],
            latest=roots[slot],
            episode_id=spec.episode_id,
            protocol=K.Protocol(spec.protocol).value,
            pairing_id=spec.pairing_id,
        )
        for slot in range(K.N_AGENTS)
    )
    return RoundState(
        episode_id=spec.episode_id,
        pairing_id=spec.pairing_id,
        item_id=spec.item_id,
        protocol=K.Protocol(spec.protocol),
        checkpoint=spec.checkpoint,
        round_=0,
        agents=agents,
        archive={0: tuple(roots)},
        pseudonyms=pseudonyms,
        study_seed=study_seed,
        classify=classify,
        count=count,
        leader=leader,
        task_text=spec.task_text,
        family=spec.family,
    )


# --------------------------------------------------------------------------- updates


def _request_for(
    snapshot: RoundState,
    spec: EpisodeSpec,
    slot: int,
    packets: Sequence[ArdPacket],
    *,
    round_: int,
    purpose: str,
) -> DecisionRequest:
    protocol = K.Protocol(spec.protocol)
    if protocol is K.Protocol.FRESH:
        # Spec §7: "Five new independent task-only solutions each round, no previous
        # answers or peer messages."  Same neutral prompt as a root; independence is in
        # the seed (draw 16 + 5(t-1) + j).
        messages = P.render_root_prompt(spec.task_text)
        purpose = K.PURPOSE_ROOT
    else:
        agent = snapshot.agents[slot]
        messages = P.render_update_prompt(
            spec.task_text, agent.root.report, agent.latest.report, packets
        )
    return DecisionRequest(
        messages=messages,
        checkpoint=spec.checkpoint,
        thinking_tokens=spec.thinking_tokens,
        seed_key=P.update_seed_key(
            source_id=spec.source_id,
            split=spec.split,
            model_cell=spec.model_cell,
            protocol=protocol,
            agent_slot=slot,
            round_=round_,
            episode_rep=spec.episode_rep,
            purpose=purpose,
        ),
        purpose=K.PURPOSE_ROOT if protocol is K.Protocol.FRESH else purpose,
    )


def _assert_packet_rounds(
    protocol: K.Protocol, packets: Sequence[ArdPacket], *, round_: int, phase: str
) -> None:
    """Spec §19: "assert no_request_read_same_round_peers_unless_HUB5"."""
    allowed = {int(round_) - 1}
    if protocol is K.Protocol.HUB5 and phase == PHASE_WORKER:
        allowed = {int(round_)}
    for packet in packets:
        if packet.source_round not in allowed:
            raise ProtocolError(
                f"{protocol.value} round {round_} ({phase}) received a round-"
                f"{packet.source_round} packet; allowed: {sorted(allowed)}"
            )


def execute_updates(
    snapshot: RoundState,
    plan: Mapping[int, tuple[ArdPacket, ...]],
    *,
    spec: EpisodeSpec,
    generate: Generate,
    round_: int,
    slots: Sequence[int] | None = None,
    phase: str = PHASE_UPDATE,
    purpose: str = K.PURPOSE_REVISE,
    action: str | None = None,
) -> tuple[Decision, ...]:
    """Run one synchronous batch of updates from the **immutable** pre-round snapshot.

    Every prompt is rendered from ``snapshot`` alone, so no agent can observe another
    agent's round-``t`` output: the five updates of a synchronous protocol are exchangeable
    and the loop is order-independent by construction, not by convention.  Results are
    returned; the caller advances the state only after the whole batch is in.

    A build-time context refusal (``ard.client.ContextTooLong``, spec §6.3
    ``context_failure``) does not abort the round: the slot still spent its planned
    opportunity, so a ``CONTEXT_FAILURE`` decision is recorded with no request and the
    episode's decision count stays exact (spec §3: "K counts the planned opportunities,
    including failed ones").
    """
    protocol = K.Protocol(spec.protocol)
    targets = list(range(K.N_AGENTS) if slots is None else slots)
    decisions: list[Decision] = []
    for slot in targets:
        packets = tuple(plan.get(slot, ()))
        _assert_packet_rounds(protocol, packets, round_=round_, phase=phase)
        request = _request_for(
            snapshot, spec, slot, packets, round_=round_, purpose=purpose
        )
        parent_state_id = snapshot.agents[slot].state_id
        try:
            record = generate(request)
        except ContextTooLong:
            report = context_failure_report()
            record = None
            decision_id = ""
        else:
            report = parse_report(
                record.final_text, record.finish_reason, record.completion_token_ids
            )
            decision_id = record.request_id
        decisions.append(
            Decision(
                decision_id=decision_id,
                candidate_id=_candidate_id(spec.episode_id, round_, slot),
                episode_id=spec.episode_id,
                agent_slot=slot,
                round_=int(round_),
                phase=phase,
                action=action or protocol.value,
                parent_state_id=parent_state_id,
                packets=packets,
                report=report,
                record=record,
                messages=tuple(dict(m) for m in request.messages),
            )
        )
    return tuple(decisions)


def _advance(
    snapshot: RoundState, decisions: Sequence[Decision], *, round_: int
) -> RoundState:
    """``state = advance(snapshot, reports)`` — the post-round state (spec §19)."""
    by_slot = {d.agent_slot: d for d in decisions}
    if set(by_slot) != set(range(K.N_AGENTS)):
        raise ProtocolError(
            f"round {round_} produced decisions for {sorted(by_slot)}, expected all "
            f"{K.N_AGENTS} slots (spec §7: five logical calls per round)"
        )
    refs = tuple(
        ReportRef(
            decision_id=by_slot[slot].decision_id,
            report=by_slot[slot].report,
            round_=int(round_),
            agent_slot=slot,
        )
        for slot in range(K.N_AGENTS)
    )
    agents = tuple(
        replace(snapshot.agents[slot], round_=int(round_), latest=refs[slot])
        for slot in range(K.N_AGENTS)
    )
    archive = {**{k: tuple(v) for k, v in snapshot.archive.items()}, int(round_): refs}
    return replace(snapshot, round_=int(round_), agents=agents, archive=archive)


# --------------------------------------------------------------------------- the loop


def run_episode(
    item: Any,
    protocol: K.Protocol,
    checkpoint: str,
    *,
    rounds: int = K.UPDATE_ROUNDS,
    generate: Generate,
    study_seed: bytes,
    model_cell: str,
    classify: Callable[[str | None], str | None],
    count: Callable[[str], int] | None = None,
    cohort: str = K.Cohort.NATURAL.value,
    episode_rep: int = 0,
    thinking_tokens: int = K.THINKING_TOKENS_LOW,
    root_draws: Sequence[int] = K.TEAM_DRAWS,
    roots: Sequence[ReportRef] | None = None,
    choose_protected: Callable[[RoundState, int], int] | None = None,
) -> Episode:
    """Run one episode end to end (contract §2 WP2; spec §7, §19).

    ``item`` is a public :class:`~agents_scaling.ard.items.ArdTask` (or anything carrying
    ``item_id``/``source_id``/``split``/``family``/``task_text``) — the public row only;
    no gold, no reference, no correctness reaches this function or anything it calls.

    ``roots`` may be passed by reference (the challenge cohort constructs teams from
    chosen root candidate ids, spec §5.3; E2d reuses the T=1,024 bank, plan §5.1); when it
    is ``None`` the five draws in ``root_draws`` are generated, which the store aliases.

    ``choose_protected(state, round_) -> slot`` supplies GATE's protected agent from the
    frozen controller (spec §12.3); RANDOM_PROTECT and CONF_PROTECT derive theirs from
    their own recorded schedules and ignore it.
    """
    proto = K.Protocol(protocol)
    spec = EpisodeSpec(
        item_id=str(item.item_id),
        source_id=str(item.source_id),
        split=str(item.split),
        family=str(getattr(item, "family", "")),
        task_text=str(item.task_text),
        protocol=proto,
        checkpoint=str(checkpoint),
        model_cell=str(model_cell),
        cohort=str(cohort),
        episode_rep=int(episode_rep),
        rounds=int(rounds),
        thinking_tokens=int(thinking_tokens),
        root_draws=tuple(int(d) for d in root_draws),
    )
    root_refs = (
        tuple(roots)
        if roots is not None
        else run_roots(spec, generate=generate)
    )
    if len(root_refs) != K.N_AGENTS:
        raise ProtocolError(f"a team needs {K.N_AGENTS} root reports, got {len(root_refs)}")

    state = initial_state(spec, root_refs, study_seed=study_seed, classify=classify, count=count)
    leader = P.hub_leader(state) if proto is K.Protocol.HUB5 else None
    if leader is not None:
        state = replace(state, leader=leader)

    root_selections = _selections(
        state,
        source_id=spec.source_id,
        root_refs=root_refs,
        archive=_archive_pairs(state),
        episode_id=spec.episode_id,
    )

    results: list[RoundResult] = []
    coverage = state.archive_classes()
    for round_ in range(1, int(rounds) + 1):
        snapshot = state  # frozen dataclass: an immutable pre-round copy by construction
        protected: int | None = None
        collapsed: Mapping[int, int] = {}
        if proto in K.PROTECTION_PROTOCOLS:
            protected = P.protected_slot_for(
                snapshot,
                proto,
                round_=round_,
                controller_choice=(
                    choose_protected(snapshot, round_) if choose_protected is not None else None
                ),
            )
        plan = P.packet_plan(snapshot, proto, round_=round_, protected_slot=protected)
        if proto is K.Protocol.BALANCED:
            collapsed = P.balanced_collapsed_counts(snapshot, round_=round_)

        if proto is K.Protocol.HUB5:
            assert leader is not None
            leader_decisions = execute_updates(
                snapshot,
                plan,
                spec=spec,
                generate=generate,
                round_=round_,
                slots=[leader],
                phase=PHASE_LEADER,
                purpose=K.PURPOSE_HUB,
            )
            leader_ref = ReportRef(
                decision_id=leader_decisions[0].decision_id,
                report=leader_decisions[0].report,
                round_=round_,
                agent_slot=leader,
            )
            worker_plan = P.hub_worker_plan(snapshot, leader_ref, round_=round_)
            worker_decisions = execute_updates(
                snapshot,
                worker_plan,
                spec=spec,
                generate=generate,
                round_=round_,
                slots=[s for s in range(K.N_AGENTS) if s != leader],
                phase=PHASE_WORKER,
            )
            decisions = leader_decisions + worker_decisions
        else:
            phase = PHASE_UPDATE
            decisions = execute_updates(
                snapshot, plan, spec=spec, generate=generate, round_=round_, phase=phase
            )

        if len(decisions) != K.N_AGENTS:
            raise ProtocolError(
                f"round {round_} made {len(decisions)} decisions; spec §7 fixes five "
                "logical answer-generating calls per update round"
            )
        state = _advance(snapshot, decisions, round_=round_)
        classes = V.class_counts(
            [
                _vote_candidate(state, state.latest(slot), _candidate_id(spec.episode_id, round_, slot))
                for slot in range(K.N_AGENTS)
            ]
        )
        new_coverage = state.archive_classes()
        if not coverage <= new_coverage:
            raise ProtocolError(
                f"archive coverage fell at round {round_}: lost {sorted(coverage - new_coverage)}"
            )
        coverage = new_coverage
        results.append(
            RoundResult(
                round_=round_,
                decisions=decisions,
                state=state,
                selections=_selections(
                    state,
                    source_id=spec.source_id,
                    root_refs=root_refs,
                    archive=_archive_pairs(state),
                    episode_id=spec.episode_id,
                ),
                class_counts=classes,
                archive_classes=frozenset(new_coverage),
                protected_slot=protected,
                leader=leader,
                collapsed_unparsed=dict(collapsed),
            )
        )

    total = sum(len(r.decisions) for r in results)
    if total != K.N_AGENTS * int(rounds):
        raise ProtocolError(
            f"episode made {total} update decisions, expected {K.N_AGENTS * int(rounds)} "
            "(spec §7: no early stopping on unanimity)"
        )
    return Episode(
        spec=spec,
        roots=tuple(root_refs),
        rounds=tuple(results),
        root_state=initial_state(
            spec, root_refs, study_seed=study_seed, classify=classify, count=count, leader=leader
        ),
        root_selections=root_selections,
        leader=leader,
        pseudonyms=state.pseudonyms,
    )


# --------------------------------------------------------------------------- FRESH bank


def fresh_bank(episode: Episode) -> list[V.VoteCandidate]:
    """The FRESH cumulative pool in **draw order**: roots 0-4 then each round's five.

    Spec §7's mandatory practical baseline votes over 5, 10, 15 and 20 independent
    candidates after rounds 0, 1, 2 and 3; the order here is the frozen draw order, which
    is what makes the prefix at ``k`` the candidates actually accumulated by that round.
    """
    if K.Protocol(episode.spec.protocol) is not K.Protocol.FRESH:
        raise ProtocolError("the cumulative prefix baseline is defined for FRESH (spec §7)")
    state = episode.final_state
    bank = [
        _vote_candidate(state, ref, _candidate_id(episode.spec.episode_id, 0, ref.agent_slot))
        for ref in episode.roots
    ]
    for result in episode.rounds:
        for slot in range(K.N_AGENTS):
            ref = result.state.latest(slot)
            bank.append(
                _vote_candidate(
                    state, ref, _candidate_id(episode.spec.episode_id, result.round_, slot)
                )
            )
    return bank
