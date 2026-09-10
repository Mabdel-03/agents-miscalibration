"""ARD36 E5 common continuation: one shared policy on top of two parent states (WP2).

Spec §13: "Take the states after round 1 in PRIVATE and PEER.  For each parent state, run
two independent suffixes.  Each suffix is one further private revision by all five agents,
using the exact same continuation procedure: task + own original root + own current branch
report, no new peer messages.  Apply the same plurality rule afterward. … This is a common
**report-state continuation**, not preservation of a live hidden state between calls.  The
conditioning reports differ because the parent intervention differs."

The estimand (spec §13) is the ordering reversal ``E[J₀(S_peer) − J₀(S_private)] > 0``
while ``E[J₅(S_peer) − J₅(S_private)] < 0``, where ``J₀`` is the parent's immediate
selected correctness and ``J₅`` the selected correctness after the five extra decisions.
This module produces both selections in the same shape so the contrast is paired, and it
refuses to run the two parents under different policies — that is the confound the whole
experiment exists to exclude, so it is an assertion, not a convention:
:func:`run_suffix_pair` checks that the two parents' continuation prompts are built by the
identical renderer with the identical (empty) packet.

"Two independent suffixes" means two ``episode_rep`` values (plan §5.1: the engine seed is
HMAC-derived, so an independent completion seed is a new ``episode_rep``, never a free
seed argument).  Spec §13 also fixes the analysis: "Average the two suffix seeds within
item; they do not double the number of independent tasks."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard import protocols as P
from agents_scaling.ard import vote as V
from agents_scaling.ard.client import ContextTooLong, DecisionRequest
from agents_scaling.ard.episode import (
    Decision,
    Episode,
    EpisodeSpec,
    Generate,
    PHASE_UPDATE,
    _advance,
    _candidate_id,
    _vote_candidate,
)
from agents_scaling.ard.report import context_failure_report, parse_report
from agents_scaling.ard.states import RoundState
from agents_scaling.study.types import ProtocolError

__all__ = [
    "SUFFIX_ACTION",
    "SUFFIX_POLICY",
    "SuffixResult",
    "run_suffix",
    "run_suffix_pair",
    "suffix_prompt",
    "suffix_rows",
    "suffix_seed_key",
]

#: The one continuation policy, identical for both parents (spec §13).
SUFFIX_POLICY = "private_revision_task_root_branch"
SUFFIX_ACTION = "suffix_private"


def suffix_prompt(state: RoundState, agent_slot: int) -> tuple[Mapping[str, str], ...]:
    """``task + own original root + own current branch report``, no new peer messages."""
    agent = state.agents[int(agent_slot)]
    return P.render_update_prompt(state.task_text, agent.root.report, agent.latest.report, ())


def suffix_seed_key(spec: EpisodeSpec, *, agent_slot: int, parent_round: int, suffix_seed: int):
    """``(actor_slot=j, purpose='revise', step_slot=parent_round+1, ns='ARD_SUFFIX')``.

    ``suffix_seed`` rides in ``episode_rep``, so the two suffixes of one parent are two
    genuinely independent completions of the same prompt and both enter the cache identity.
    """
    return P.update_seed_key(
        source_id=spec.source_id,
        split=spec.split,
        model_cell=spec.model_cell,
        protocol=K.Protocol.PRIVATE,
        agent_slot=int(agent_slot),
        round_=int(parent_round) + 1,
        episode_rep=int(suffix_seed),
        purpose=K.PURPOSE_REVISE,
        namespace=K.NS_SUFFIX,
    )


@dataclass(frozen=True)
class SuffixResult:
    """One suffix of one parent state: five private revisions and their selection."""

    episode_id: str
    item_id: str
    checkpoint: str
    parent_protocol: str
    parent_round: int
    parent_state_ids: tuple[str, ...]
    suffix_seed: int
    decisions: tuple[Decision, ...]
    state: RoundState
    selection: V.ArdSelection
    parent_selection: V.ArdSelection
    class_counts: Mapping[str, int]

    def to_row(self) -> dict[str, Any]:
        """A ``suffix_summary`` row (plan §5.8)."""
        return {
            "episode_id": self.episode_id,
            "item_id": self.item_id,
            "checkpoint": self.checkpoint,
            "parent_protocol": self.parent_protocol,
            "parent_round": self.parent_round,
            "parent_state_ids": list(self.parent_state_ids),
            "suffix_seed": self.suffix_seed,
            "common_suffix_policy": SUFFIX_POLICY,
            "decision_ids": [d.decision_id for d in self.decisions],
            "class_counts": dict(self.class_counts),
            "j0_selected_candidate_id": self.parent_selection.selected_candidate_id,
            "j0_selected_class": self.parent_selection.selected_class,
            "j0_selection_failed": self.parent_selection.selection_failed,
            "j5_selected_candidate_id": self.selection.selected_candidate_id,
            "j5_selected_class": self.selection.selected_class,
            "j5_selection_failed": self.selection.selection_failed,
            "coverage": len(self.class_counts),
        }


def run_suffix(
    parent: RoundState,
    *,
    spec: EpisodeSpec,
    generate: Generate,
    suffix_seed: int,
    parent_selection: V.ArdSelection | None = None,
) -> SuffixResult:
    """One further private revision by **all five** agents from a sealed parent state.

    The parent must be a round-1 state (spec §13 takes "the states after round 1"); the
    five continuations are synchronous and read the parent snapshot only, exactly like a
    normal round, so J₅ is a selection over five current reports and is directly comparable
    with the parent's J₀.
    """
    parent_round = int(parent.round_)
    if parent_round < 1:
        raise ProtocolError("E5 continues a state after round 1 (spec §13), not the root state")
    source_id = spec.source_id
    decisions: list[Decision] = []
    for slot in range(K.N_AGENTS):
        request = DecisionRequest(
            messages=suffix_prompt(parent, slot),
            checkpoint=spec.checkpoint,
            thinking_tokens=int(spec.thinking_tokens),
            seed_key=suffix_seed_key(
                spec, agent_slot=slot, parent_round=parent_round, suffix_seed=suffix_seed
            ),
            purpose=K.PURPOSE_REVISE,
        )
        try:
            record = generate(request)
        except ContextTooLong:
            record = None
            report = context_failure_report()
            decision_id = ""
        else:
            report = parse_report(
                record.final_text, record.finish_reason, record.completion_token_ids
            )
            decision_id = record.request_id
        decisions.append(
            Decision(
                decision_id=decision_id,
                candidate_id=f"{spec.episode_id}#suffix{int(suffix_seed)}a{slot}",
                episode_id=spec.episode_id,
                agent_slot=slot,
                round_=parent_round + 1,
                phase=PHASE_UPDATE,
                action=SUFFIX_ACTION,
                parent_state_id=parent.agents[slot].state_id,
                packets=(),
                report=report,
                record=record,
                messages=tuple(dict(m) for m in request.messages),
            )
        )
    child = _advance(parent, tuple(decisions), round_=parent_round + 1)
    pool = [
        V.VoteCandidate(
            candidate_id=decision.candidate_id,
            canonical_class=child.canonical_class(decision.report),
            valid=decision.report.valid,
            confidence=decision.report.confidence,
        )
        for decision in decisions
    ]
    selection = V.plurality(
        pool, source_id=source_id, study_seed=parent.study_seed, pool_kind="current"
    )
    if parent_selection is None:
        parent_pool = [
            _vote_candidate(parent, parent.latest(slot), _candidate_id(spec.episode_id, parent_round, slot))
            for slot in range(K.N_AGENTS)
        ]
        parent_selection = V.plurality(
            parent_pool, source_id=source_id, study_seed=parent.study_seed, pool_kind="current"
        )
    return SuffixResult(
        episode_id=spec.episode_id,
        item_id=spec.item_id,
        checkpoint=spec.checkpoint,
        parent_protocol=K.Protocol(parent.protocol).value,
        parent_round=parent_round,
        parent_state_ids=tuple(a.state_id for a in parent.agents),
        suffix_seed=int(suffix_seed),
        decisions=tuple(decisions),
        state=child,
        selection=selection,
        parent_selection=parent_selection,
        class_counts=V.class_counts(pool),
    )


def run_suffix_pair(
    private_episode: Episode,
    peer_episode: Episode,
    *,
    generate: Generate,
    seeds: Sequence[int] = tuple(range(K.SUFFIX_SEEDS)),
    parent_round: int = 1,
) -> dict[str, tuple[SuffixResult, ...]]:
    """Both parents, both suffix seeds, one policy (spec §13).

    Returns ``{"PRIVATE": (…), "PEER": (…)}``.  The two parents are asserted to be the
    paired arms of one item and to produce continuation prompts under the identical policy
    — the same renderer, the same empty packet, differing only in the branch report that
    the parent intervention itself produced.
    """
    if private_episode.spec.pairing_id != peer_episode.spec.pairing_id:
        raise ProtocolError(
            "an E5 suffix pair needs the two arms of one paired episode: "
            f"{private_episode.spec.pairing_id} != {peer_episode.spec.pairing_id}"
        )
    for episode, expected in ((private_episode, K.Protocol.PRIVATE), (peer_episode, K.Protocol.PEER)):
        if K.Protocol(episode.spec.protocol) is not expected:
            raise ProtocolError(f"expected the {expected.value} arm, got {episode.spec.protocol}")
        if len(episode.rounds) < parent_round:
            raise ProtocolError(f"{episode.spec.episode_id} has no round-{parent_round} state")
    out: dict[str, tuple[SuffixResult, ...]] = {}
    for episode in (private_episode, peer_episode):
        parent = episode.rounds[parent_round - 1].state
        results = [
            run_suffix(parent, spec=episode.spec, generate=generate, suffix_seed=int(seed))
            for seed in seeds
        ]
        out[K.Protocol(episode.spec.protocol).value] = tuple(results)
    return out


def suffix_rows(results: Sequence[SuffixResult]) -> list[dict[str, Any]]:
    return [result.to_row() for result in results]
