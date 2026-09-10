"""ARD36 E4 private counterfactuals: branching a stored state (WP2).

Spec §12.1: "For every 8B training/validation PEER state at t=1 and t=2, fork each of the
five agents into one private next-step revision.  The peer next-step outcome already
exists in the main trajectory.  Other agents do not need regeneration for the one-step
individual target.  This adds 144×2×5 = 1,440 logical decisions.  **Root-state private
counterparts at t=0 already exist in PRIVATE and can be reused if byte-exact construction
matches.**"  Spec §19's counterfactual feature boundary: the features are frozen on the
*shared prefix*, before the intervention.

Two kinds of fork, and the difference is the whole point of this module:

* a **paid** fork at ``t ∈ contract.FORK_PARENT_ROUNDS`` — one new private revision of the
  focal agent from a sealed PEER state, in the ``ARD_FORK`` seed namespace;
* the **free** ``t=0`` pair — PRIVATE's round-1 child and PEER's round-1 child are two
  branches of the *identical* root state (same item, same root report, same latest = root,
  same empty packet), so both already exist and the fork costs nothing.  The byte-exactness
  claim behind that reuse is a test, not a hope: ``tests/ard/test_wp2_forks.py`` asserts
  PRIVATE's round-1 prompt equals :func:`fork_prompt` on the root state, byte for byte.

Spec §12.1 also fixes the interpretation: "Use one stochastic outcome per branch as noisy
supervision, not an exact individual causal effect."  Nothing here computes an effect; it
emits ``forks.parquet`` rows (spec §18: ``fork_id, shared_prefix_hash, focal_agent, action,
branch_seed, common_suffix_policy, suffix_seed, parent/child_decision_ids``) for WP5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agents_scaling.ard import contract as K
from agents_scaling.ard import protocols as P
from agents_scaling.ard.client import ContextTooLong, DecisionRecord, DecisionRequest
from agents_scaling.ard.episode import Decision, Episode, EpisodeSpec, Generate
from agents_scaling.ard.report import Report, context_failure_report, parse_report
from agents_scaling.ard.states import RoundState
from agents_scaling.study import identity
from agents_scaling.study.types import ProtocolError

__all__ = [
    "ACTION_PEER",
    "ACTION_PRIVATE",
    "Fork",
    "NO_SUFFIX_POLICY",
    "fork_id",
    "fork_prompt",
    "fork_private",
    "fork_seed_key",
    "free_root_pair",
    "fork_rows",
]

ACTION_PRIVATE = "private"
ACTION_PEER = "peer"
#: ``forks.parquet.common_suffix_policy`` for a one-step fork (E5 owns the suffix policy).
NO_SUFFIX_POLICY = "none"


def fork_prompt(state: RoundState, focal_agent: int) -> tuple[Mapping[str, str], ...]:
    """The private continuation prompt of ``focal_agent`` from ``state`` (spec §6.4).

    Exactly ``task + own root + own latest (one copy when identical) + empty packet`` —
    the same renderer PRIVATE uses, which is what makes the ``t=0`` branch free.
    """
    agent = state.agents[int(focal_agent)]
    return P.render_update_prompt(state.task_text, agent.root.report, agent.latest.report, ())


def fork_seed_key(
    spec: EpisodeSpec, *, focal_agent: int, parent_round: int, branch: int = 0
):
    """``(actor_slot=focal, purpose='revise', step_slot=parent_round+1, ns='ARD_FORK')``.

    The fork lives in its own namespace so a counterfactual branch can never alias into the
    main trajectory's record and be double-counted as an observed outcome.  ``branch`` is
    carried in ``episode_rep``: the repeated-seed fork subset (extension 2, plan §3) is a
    second ``episode_rep``, never a free seed argument (plan §5.1).
    """
    return P.update_seed_key(
        source_id=spec.source_id,
        split=spec.split,
        model_cell=spec.model_cell,
        protocol=K.Protocol.PRIVATE,  # the counterfactual *action* is a private revision
        agent_slot=int(focal_agent),
        round_=int(parent_round) + 1,
        episode_rep=int(branch),
        purpose=K.PURPOSE_REVISE,
        namespace=K.NS_FORK,
    )


def fork_id(parent_state_id: str, focal_agent: int, action: str, branch: int) -> str:
    return identity.sha256_hex(
        identity.jcs(["ard_fork", str(parent_state_id), int(focal_agent), str(action), int(branch)])
    )


@dataclass(frozen=True)
class Fork:
    """One counterfactual branch of a sealed state (spec §18 ``forks.parquet``)."""

    fork_id: str
    shared_prefix_hash: str          # the parent ``state:<sha>``
    focal_agent: int
    action: str
    branch: int
    branch_seed: int
    parent_decision_ids: tuple[str, ...]
    child_decision_ids: tuple[str, ...]
    parent_round: int
    episode_id: str
    item_id: str
    checkpoint: str
    parent_protocol: str
    report: Report | None = None
    record: DecisionRecord | None = None
    messages: tuple[Mapping[str, str], ...] = ()
    common_suffix_policy: str = NO_SUFFIX_POLICY
    suffix_seed: int | None = None
    reused: bool = False

    def to_row(self) -> dict[str, Any]:
        return {
            "fork_id": self.fork_id,
            "shared_prefix_hash": self.shared_prefix_hash,
            "focal_agent": self.focal_agent,
            "action": self.action,
            "branch": self.branch,
            "branch_seed": self.branch_seed,
            "common_suffix_policy": self.common_suffix_policy,
            "suffix_seed": self.suffix_seed,
            "parent_decision_ids": list(self.parent_decision_ids),
            "child_decision_ids": list(self.child_decision_ids),
            "parent_round": self.parent_round,
            "episode_id": self.episode_id,
            "item_id": self.item_id,
            "checkpoint": self.checkpoint,
            "parent_protocol": self.parent_protocol,
            "reused": self.reused,
            "parse_status": None if self.report is None else self.report.parse_status.value,
            "content_sha256": None if self.report is None else self.report.content_sha256,
        }


def _parent_decision_ids(state: RoundState, focal_agent: int) -> tuple[str, ...]:
    agent = state.agents[int(focal_agent)]
    ids = [agent.root.decision_id]
    if agent.latest.decision_id != agent.root.decision_id:
        ids.append(agent.latest.decision_id)
    return tuple(ids)


def fork_private(
    state: RoundState,
    focal_agent: int,
    *,
    spec: EpisodeSpec,
    generate: Generate,
    study_seed: bytes,
    branch: int = 0,
    thinking_tokens: int | None = None,
) -> Fork:
    """``fork_private(previous_peer_state, focal_agent) -> counterfactual report`` (spec §19).

    One extra private revision of ``focal_agent`` from the sealed parent state.  No other
    agent is regenerated: the peer next-step outcome for the same prefix already exists in
    the main trajectory, which is exactly the pairing that makes the label
    ``y_private − y_peer`` (plan §5.4).
    """
    parent_round = int(state.round_)
    if parent_round not in K.FORK_PARENT_ROUNDS and parent_round != 0:
        raise ProtocolError(
            f"E4 forks branch from rounds {K.FORK_PARENT_ROUNDS} (and the free t=0 pair), "
            f"got round {parent_round}"
        )
    seed_key = fork_seed_key(spec, focal_agent=focal_agent, parent_round=parent_round, branch=branch)
    request = DecisionRequest(
        messages=fork_prompt(state, focal_agent),
        checkpoint=spec.checkpoint,
        thinking_tokens=int(spec.thinking_tokens if thinking_tokens is None else thinking_tokens),
        seed_key=seed_key,
        purpose=K.PURPOSE_REVISE,
    )
    try:
        record = generate(request)
    except ContextTooLong:
        record = None
        report = context_failure_report()
        child_ids: tuple[str, ...] = ()
    else:
        report = parse_report(record.final_text, record.finish_reason, record.completion_token_ids)
        child_ids = (record.request_id,)
    parent_state_id = state.agents[int(focal_agent)].state_id
    return Fork(
        fork_id=fork_id(parent_state_id, focal_agent, ACTION_PRIVATE, branch),
        shared_prefix_hash=parent_state_id,
        focal_agent=int(focal_agent),
        action=ACTION_PRIVATE,
        branch=int(branch),
        branch_seed=identity.engine_seed(identity.semantic_seed(study_seed, seed_key)),
        parent_decision_ids=_parent_decision_ids(state, focal_agent),
        child_decision_ids=child_ids,
        parent_round=parent_round,
        episode_id=spec.episode_id,
        item_id=spec.item_id,
        checkpoint=spec.checkpoint,
        parent_protocol=K.Protocol(state.protocol).value,
        report=report,
        record=record,
        messages=tuple(dict(m) for m in request.messages),
    )


def _round_one_decision(episode: Episode, focal_agent: int) -> Decision:
    for decision in episode.rounds[0].decisions:
        if decision.agent_slot == int(focal_agent):
            return decision
    raise ProtocolError(f"episode {episode.spec.episode_id} has no round-1 decision for slot {focal_agent}")


def free_root_pair(
    private_episode: Episode, peer_episode: Episode, *, study_seed: bytes
) -> tuple[Fork, ...]:
    """The free ``t=0`` counterfactual pair (spec §12.1, plan §5.4).

    PRIVATE's round-1 child and PEER's round-1 child branch from the *identical* root
    state, so both branches already exist and no request is issued here.  The two episodes
    must be the same item, checkpoint and cohort — the pairing id — or they do not share a
    prefix and the pair would be a comparison across items.

    Every returned row has ``reused=True``: these are references into existing decisions,
    never new generations, and the ledger must not count them twice.
    """
    if private_episode.spec.pairing_id != peer_episode.spec.pairing_id:
        raise ProtocolError(
            "the free t=0 pair needs two arms of one paired episode: "
            f"{private_episode.spec.pairing_id} != {peer_episode.spec.pairing_id}"
        )
    if K.Protocol(private_episode.spec.protocol) is not K.Protocol.PRIVATE:
        raise ProtocolError("free_root_pair expects the PRIVATE arm first")
    if K.Protocol(peer_episode.spec.protocol) is not K.Protocol.PEER:
        raise ProtocolError("free_root_pair expects the PEER arm second")
    out: list[Fork] = []
    for focal in range(K.N_AGENTS):
        private_state_id = private_episode.root_state.agents[focal].state_id
        peer_state_id = peer_episode.root_state.agents[focal].state_id
        if private_state_id != peer_state_id:
            raise ProtocolError(
                f"the t=0 branches of slot {focal} do not share a root state: "
                f"{private_state_id} != {peer_state_id} (are the root draws aliased?)"
            )
        for episode, action in ((private_episode, ACTION_PRIVATE), (peer_episode, ACTION_PEER)):
            decision = _round_one_decision(episode, focal)
            seed_key = P.update_seed_key(
                source_id=episode.spec.source_id,
                split=episode.spec.split,
                model_cell=episode.spec.model_cell,
                protocol=K.Protocol(episode.spec.protocol),
                agent_slot=focal,
                round_=1,
                episode_rep=episode.spec.episode_rep,
            )
            out.append(
                Fork(
                    fork_id=fork_id(private_state_id, focal, action, 0),
                    shared_prefix_hash=private_state_id,
                    focal_agent=focal,
                    action=action,
                    branch=0,
                    branch_seed=identity.engine_seed(identity.semantic_seed(study_seed, seed_key)),
                    parent_decision_ids=_parent_decision_ids(episode.root_state, focal),
                    child_decision_ids=(decision.decision_id,) if decision.decision_id else (),
                    parent_round=0,
                    episode_id=episode.spec.episode_id,
                    item_id=episode.spec.item_id,
                    checkpoint=episode.spec.checkpoint,
                    parent_protocol=K.Protocol(episode.spec.protocol).value,
                    report=decision.report,
                    record=decision.record,
                    messages=decision.messages,
                    reused=True,
                )
            )
    return tuple(out)


def fork_rows(forks: Sequence[Fork]) -> list[dict[str, Any]]:
    """``forks.parquet`` rows (spec §18)."""
    return [fork.to_row() for fork in forks]
