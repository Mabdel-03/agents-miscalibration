"""The declared public state of a team, rebuilt each round from the previous one (spec 6.4).

"The active state is the immediately preceding public draft, not an ever-growing transcript.
Rebuild each round's model input from the declared state.  Do not rely on backend
conversational memory or carry undisclosed KV state between rounds."  These dataclasses are
that declared state, and they are n-parametric: nothing here knows that the study runs five
agents (:attr:`RoundState.n` is the length of the agent tuple, and the same objects drive the
N=3 and N=9 blocks unchanged).

``history`` keeps every round's reference so historical coverage -- the classes the team has
*ever* held -- can be computed without re-reading the archive, and ``root`` keeps round 0
separately because a root is a *reference* to a bank draw and not a generation of its own
(plan D14).  ``classify`` is the gold-independent canonicalizer; a state never holds a grade,
which is what keeps the selection and episode paths free of gold (R7).

:func:`rebuild_round_states` reconstructs the same objects from a finished item file, so the
analysis of an episode and the running of it share one notion of state (R1) rather than the
two divergent ones ARD36 ended up with.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Final, Mapping, Sequence

from agents_scaling.nextdyn import ids
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.nextdyn.graphs import Graph
from agents_scaling.nextdyn.packets import pseudonym_map
from agents_scaling.nextdyn.schema import PublicOutput
from agents_scaling.study import identity

#: ``final answer -> canonical class`` (gold-independent; ``None`` when unresolved).
Classifier = Callable[[str], "str | None"]

STATE_ID_PREFIX: Final[str] = "s:"
STATE_ID_CHARS: Final[int] = 32


def pairing_id_for(source_item_id: str, root_team_id: str, model_hash: str, budget_hash: str,
                   replicate: int) -> str:
    """The identity **shared by paired arms**: item, team, model, budget, replicate -- no condition.

    Every blind schedule (pseudonyms, packet order, ring permutation, protection slot) is
    keyed by this, so PRIVATE / PEER / FRESH of one team randomize identically and the
    packets they have in common are byte-comparable (spec 6.4).  The condition is absent on
    purpose; putting it in was ARD36's way of making paired arms incomparable.
    """
    return (f"{source_item_id}|{root_team_id}|{model_hash}|{budget_hash}|rep{int(replicate)}")


@dataclass(frozen=True)
class DecisionRef:
    """One slot's decision at one round: what the next round is allowed to look at."""

    decision_id: str
    round_: int
    agent_slot: int
    generation_uid: str | None
    output: PublicOutput | None

    @property
    def canonical_input(self) -> str | None:
        """The final answer a canonicalizer would be given (``None`` when not gradeable)."""
        return None if self.output is None else self.output.final_answer


@dataclass(frozen=True)
class AgentState:
    """One agent's public state: its pseudonym, its root, its latest draft, its history."""

    slot: int
    pseudonym: str
    latest: DecisionRef
    root: DecisionRef
    history: tuple[DecisionRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "history", tuple(self.history))
        if self.latest.agent_slot != self.slot or self.root.agent_slot != self.slot:
            raise ContractViolation(f"AgentState slot {self.slot} holds another slot's decision")

    def advanced(self, ref: DecisionRef) -> "AgentState":
        return replace(self, latest=ref, history=(*self.history, ref))


@dataclass(frozen=True)
class RoundState:
    """The immutable pre-round snapshot every prompt of the next round is rendered from."""

    pairing_id: str
    round_: int
    agents: tuple[AgentState, ...]
    classify: Classifier
    graph: Graph | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "agents", tuple(self.agents))
        if not self.agents:
            raise ContractViolation("a round state holds at least one agent")
        if tuple(a.slot for a in self.agents) != tuple(range(len(self.agents))):
            raise ContractViolation("agents must be in slot order 0..n-1")
        if self.graph is not None and len(self.graph) != len(self.agents):
            raise ContractViolation("the round graph must be n x n")

    @property
    def n(self) -> int:
        """The team size: read from the state, never from a module constant."""
        return len(self.agents)

    def classes(self) -> tuple[str | None, ...]:
        """Each slot's canonical class this round (``None`` = invalid or unresolved)."""
        out = []
        for agent in self.agents:
            answer = agent.latest.canonical_input
            out.append(None if answer is None else self.classify(answer))
        return tuple(out)

    def historical_classes(self) -> frozenset[str]:
        """Every class the team has ever held up to and including this round (spec 18.3)."""
        seen: set[str] = set()
        for agent in self.agents:
            for ref in (agent.root, *agent.history):
                answer = ref.canonical_input
                if answer is None:
                    continue
                label = self.classify(answer)
                if label is not None:
                    seen.add(label)
        return frozenset(seen)

    def unanimous(self) -> str | None:
        """The single class every slot holds, or ``None`` (spec 18.3 unanimity)."""
        classes = self.classes()
        if any(c is None for c in classes) or len(set(classes)) != 1:
            return None
        return classes[0]


def state_id(state: RoundState) -> str:
    """Content-addressed over the *public* state: pairing, round, and each slot's class.

    Correctness never enters (the classifier is gold-independent), so a state id can be
    logged and joined without being an outcome.
    """
    payload = [state.pairing_id, int(state.round_),
               [[agent.slot, agent.latest.decision_id, label]
                for agent, label in zip(state.agents, state.classes())]]
    return STATE_ID_PREFIX + identity.sha256_hex(identity.jcs(payload))[:STATE_ID_CHARS]


def initial_state(*, pairing_id: str, roots: Sequence[DecisionRef], classify: Classifier,
                  study_seed: bytes, graph: Graph | None = None) -> RoundState:
    """Round 0: every slot's latest draft *is* its root reference (plan D14)."""
    refs = tuple(roots)
    names = pseudonym_map(study_seed, pairing_id, len(refs))
    agents = tuple(
        AgentState(slot=i, pseudonym=names[i], latest=ref, root=ref, history=())
        for i, ref in enumerate(refs))
    return RoundState(pairing_id=pairing_id, round_=refs[0].round_ if refs else 0,
                      agents=agents, classify=classify, graph=graph)


def advance(snapshot: RoundState, refs: Mapping[int, DecisionRef], *, round_: int) -> RoundState:
    """The post-round state.  Every slot must be present: a round is all-or-nothing (spec 6.4).

    A context-refused slot still supplies a reference (with a typed, non-gradeable output), so
    "all n slots" is a statement about planned opportunities, not about successes.
    """
    missing = sorted(set(range(snapshot.n)) - set(refs))
    if missing:
        raise ContractViolation(
            f"round {round_} produced no decision for slot(s) {missing}; a synchronous round "
            f"advances only when all {snapshot.n} slots are in (spec 6.4)")
    agents = tuple(snapshot.agents[slot].advanced(refs[slot]) for slot in range(snapshot.n))
    return replace(snapshot, round_=int(round_), agents=agents, graph=None)


def rebuild_round_states(item_file: Any, classify: Classifier, *, study_seed: bytes,
                         episode_id: str | None = None) -> list[RoundState]:
    """Rebuild the per-round states of one episode from a finished item file.

    The running path and the analysis path then share one notion of state (R1).  Rows are
    read through the ``decisions`` row dataclass, never by dict key, and the ``pairing_id``
    is recomputed from the key columns rather than stored twice.
    """
    rows = [r for r in item_file.rows.get("decisions", ())
            if episode_id is None or r.episode_id == episode_id]
    if not rows:
        raise ContractViolation("the item file holds no decisions rows for this episode")
    episodes = {r.episode_id for r in rows}
    if len(episodes) != 1:
        raise ContractViolation(f"pass episode_id: the item file holds {sorted(episodes)}")
    first = rows[0]
    pairing_id = pairing_id_for(first.source_item_id, first.root_team_id,
                                first.receiver_model_hash, first.budget_hash, first.replicate_id)
    by_round: dict[int, dict[int, DecisionRef]] = {}
    for row in rows:
        by_round.setdefault(int(row.round), {})[int(row.agent_slot)] = DecisionRef(
            decision_id=row.decision_id, round_=int(row.round), agent_slot=int(row.agent_slot),
            generation_uid=row.generation_uid, output=_output_of(row))
    rounds = sorted(by_round)
    states = [initial_state(pairing_id=pairing_id,
                            roots=[by_round[rounds[0]][slot] for slot in sorted(by_round[rounds[0]])],
                            classify=classify, study_seed=study_seed)]
    for round_ in rounds[1:]:
        states.append(advance(states[-1], by_round[round_], round_=round_))
    return states


def _output_of(row: Any) -> PublicOutput | None:
    """The public output a rebuilt reference carries: the final answer and its status only.

    A rebuilt state is used for coverage and unanimity, so it needs the answer and whether the
    row was gradeable; the original bytes stay in the row (R1) and are not duplicated here.
    """
    import json

    from agents_scaling.nextdyn.schema import ParseStatus

    if row.final_answer_raw is None and row.parse_status is None:
        return None
    claims = tuple(json.loads(row.shareable_claims_json)) if row.shareable_claims_json else ()
    return PublicOutput(
        reasoning=row.public_reasoning, shareable_claims=claims, final_answer=row.final_answer_raw,
        confidence=row.confidence, field_order=(), order_violation=bool(row.field_order_violation),
        parse_status=ParseStatus(row.parse_status), raw_text="",
        content_hash=row.content_hash or ids.content_hash(b""), over_limits={},
        final_answer_span=None, confidence_span=None, confidence_emitted_before_answer=False,
    )


__all__ = [
    "AgentState", "Classifier", "DecisionRef", "RoundState", "STATE_ID_PREFIX", "advance",
    "initial_state", "pairing_id_for", "rebuild_round_states", "state_id",
]
