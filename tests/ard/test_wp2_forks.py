"""ARD36 WP2: E4 private counterfactual branches (spec §12.1, §19; plan §5.4).

The load-bearing claim is spec §12.1's "Root-state private counterparts at t=0 already
exist in PRIVATE and can be reused **if byte-exact construction matches**".  This file is
that check: PRIVATE's round-1 prompt must equal the fork construction byte for byte, or the
720 free training pairs are not free and the plan's decision ledger is wrong.
"""

from __future__ import annotations

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import protocols as P
from agents_scaling.ard.forks import (
    ACTION_PEER,
    ACTION_PRIVATE,
    fork_private,
    fork_prompt,
    fork_rows,
    fork_seed_key,
    free_root_pair,
)
from agents_scaling.study.types import ProtocolError
from tests.ard import wp2_support as W


def test_private_round_one_prompt_is_byte_identical_to_the_fork_construction(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        private = h.run(K.Protocol.PRIVATE, rounds=1)
    for slot in range(K.N_AGENTS):
        decision = next(d for d in private.rounds[0].decisions if d.agent_slot == slot)
        built = fork_prompt(private.root_state, slot)
        assert tuple(dict(m) for m in built) == decision.messages
        assert "(no messages this round)" in built[0]["content"]


def test_the_free_t0_pair_costs_no_requests(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        private = h.run(K.Protocol.PRIVATE, rounds=1)
        peer = h.run(K.Protocol.PEER, rounds=1)
        before = h.server.chat_count
        forks = free_root_pair(private, peer, study_seed=h.study_seed)
        assert h.server.chat_count == before          # no generation at all

    assert len(forks) == 2 * K.N_AGENTS
    assert all(fork.reused for fork in forks)
    for slot in range(K.N_AGENTS):
        pair = [f for f in forks if f.focal_agent == slot]
        assert {f.action for f in pair} == {ACTION_PRIVATE, ACTION_PEER}
        # both branches hang off the identical shared prefix (the root state)
        assert len({f.shared_prefix_hash for f in pair}) == 1
        assert pair[0].shared_prefix_hash == private.root_state.agents[slot].state_id
        assert pair[0].parent_round == 0
        # ... and their children are the two arms' existing round-1 decisions
        children = {f.action: f.child_decision_ids for f in pair}
        assert children[ACTION_PRIVATE] != children[ACTION_PEER]
        assert children[ACTION_PRIVATE][0] == next(
            d.decision_id for d in private.rounds[0].decisions if d.agent_slot == slot
        )
    rows = fork_rows(forks)
    assert {"fork_id", "shared_prefix_hash", "focal_agent", "action", "branch_seed",
            "common_suffix_policy", "suffix_seed", "parent_decision_ids",
            "child_decision_ids"} <= set(rows[0])


def test_the_free_pair_refuses_mismatched_arms(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        private = h.run(K.Protocol.PRIVATE, rounds=1)
        peer = h.run(K.Protocol.PEER, rounds=1, item=W.StubTask(item_id="other", source_id="other"))
        with pytest.raises(ProtocolError):
            free_root_pair(private, peer, study_seed=h.study_seed)
        with pytest.raises(ProtocolError):
            free_root_pair(peer, private, study_seed=h.study_seed)


@pytest.mark.parametrize("parent_round", list(K.FORK_PARENT_ROUNDS))
def test_a_paid_fork_branches_one_agent_from_a_peer_state(tmp_path, parent_round) -> None:
    with W.harness(tmp_path) as h:
        peer = h.run(K.Protocol.PEER, rounds=max(K.FORK_PARENT_ROUNDS))
        parent = peer.rounds[parent_round - 1].state
        before = h.server.chat_count
        fork = fork_private(parent, 3, spec=peer.spec, generate=h.generate, study_seed=h.study_seed)
        assert h.server.chat_count == before + 1      # exactly one extra decision

    assert fork.action == ACTION_PRIVATE and fork.focal_agent == 3
    assert fork.parent_round == parent_round
    assert fork.shared_prefix_hash == parent.agents[3].state_id
    assert fork.parent_protocol == K.Protocol.PEER.value
    assert fork.report is not None and fork.child_decision_ids
    # the branch never aliases into the main trajectory: its own namespace, its own id
    assert fork.child_decision_ids[0] not in {d.decision_id for d in peer.decisions}
    key = fork_seed_key(peer.spec, focal_agent=3, parent_round=parent_round)
    assert key.namespace == K.NS_FORK and key.step_slot == parent_round + 1


def test_the_fork_prompt_is_the_private_continuation_of_the_parent_state(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        peer = h.run(K.Protocol.PEER, rounds=1)
    parent = peer.rounds[0].state
    for slot in range(K.N_AGENTS):
        agent = parent.agents[slot]
        assert fork_prompt(parent, slot) == P.render_update_prompt(
            parent.task_text, agent.root.report, agent.latest.report, ()
        )


def test_a_repeated_seed_branch_is_a_second_episode_rep(tmp_path) -> None:
    """Plan §5.1: an independent completion seed is a new ``episode_rep``, never a free seed."""
    with W.harness(tmp_path) as h:
        peer = h.run(K.Protocol.PEER, rounds=1)
        parent = peer.rounds[0].state
        first = fork_private(parent, 0, spec=peer.spec, generate=h.generate, study_seed=h.study_seed)
        second = fork_private(
            parent, 0, spec=peer.spec, generate=h.generate, study_seed=h.study_seed, branch=1
        )
    assert first.child_decision_ids != second.child_decision_ids
    assert first.branch_seed != second.branch_seed
    assert first.fork_id != second.fork_id
    assert fork_seed_key(peer.spec, focal_agent=0, parent_round=1, branch=1).episode_rep == 1


def test_a_fork_refuses_a_round_outside_the_registered_parents(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        peer = h.run(K.Protocol.PEER)
        with pytest.raises(ProtocolError):
            fork_private(
                peer.rounds[2].state, 0, spec=peer.spec, generate=h.generate, study_seed=h.study_seed
            )
