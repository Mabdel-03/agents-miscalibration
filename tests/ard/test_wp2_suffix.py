"""ARD36 WP2: E5 common continuation (spec §13).

"For each parent state, run two independent suffixes.  Each suffix is one further private
revision by all five agents, using the exact same continuation procedure … Apply the same
plurality rule afterward."  The identity that matters is the *policy*: if the two parents
were continued differently, the J₅ contrast would measure the continuation, not the parent.
"""

from __future__ import annotations

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import protocols as P
from agents_scaling.ard.suffix import (
    SUFFIX_POLICY,
    run_suffix,
    run_suffix_pair,
    suffix_prompt,
    suffix_rows,
    suffix_seed_key,
)
from agents_scaling.study.types import ProtocolError
from tests.ard import wp2_support as W


def test_a_suffix_is_five_private_revisions_from_a_round_one_state(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        peer = h.run(K.Protocol.PEER, rounds=1)
        parent = peer.rounds[0].state
        before = h.server.chat_count
        result = run_suffix(parent, spec=peer.spec, generate=h.generate, suffix_seed=0)
        assert h.server.chat_count == before + K.N_AGENTS
        prompts = h.prompts()

    assert len(result.decisions) == K.N_AGENTS
    assert result.parent_round == 1 and result.state.round_ == 2
    for decision in result.decisions:
        assert decision.packets == ()                      # no new peer messages (§13)
        content = decision.messages[0]["content"]
        assert content in prompts
        assert "(no messages this round)" in content
    assert result.selection.pool_kind == "current"
    assert result.parent_selection.planned_count == K.N_AGENTS


def test_both_parents_are_continued_under_one_policy(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        private = h.run(K.Protocol.PRIVATE, rounds=1)
        peer = h.run(K.Protocol.PEER, rounds=1)
        pair = run_suffix_pair(private, peer, generate=h.generate)

    assert sorted(pair) == [K.Protocol.PEER.value, K.Protocol.PRIVATE.value]
    for name, results in pair.items():
        assert len(results) == K.SUFFIX_SEEDS
        assert {r.suffix_seed for r in results} == set(range(K.SUFFIX_SEEDS))
        for result in results:
            assert result.parent_protocol == name
            for row in suffix_rows([result]):
                assert row["common_suffix_policy"] == SUFFIX_POLICY
                assert "j0_selected_class" in row and "j5_selected_class" in row

    # the two parents' prompts are built by the identical renderer with the identical
    # (empty) packet; only the branch report the parent intervention produced differs
    for episode in (private, peer):
        parent = episode.rounds[0].state
        for slot in range(K.N_AGENTS):
            agent = parent.agents[slot]
            assert suffix_prompt(parent, slot) == P.render_update_prompt(
                parent.task_text, agent.root.report, agent.latest.report, ()
            )


def test_the_two_suffix_seeds_are_two_independent_completions(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        peer = h.run(K.Protocol.PEER, rounds=1)
        parent = peer.rounds[0].state
        first = run_suffix(parent, spec=peer.spec, generate=h.generate, suffix_seed=0)
        second = run_suffix(parent, spec=peer.spec, generate=h.generate, suffix_seed=1)

    assert [d.messages for d in first.decisions] == [d.messages for d in second.decisions]
    assert {d.decision_id for d in first.decisions}.isdisjoint(
        {d.decision_id for d in second.decisions}
    )
    keys = [
        suffix_seed_key(peer.spec, agent_slot=0, parent_round=1, suffix_seed=s) for s in (0, 1)
    ]
    assert keys[0].namespace == K.NS_SUFFIX and keys[0].step_slot == 2
    assert keys[0].episode_rep == 0 and keys[1].episode_rep == 1


def test_a_suffix_never_aliases_into_the_main_trajectory(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        peer = h.run(K.Protocol.PEER, rounds=2)
        result = run_suffix(
            peer.rounds[0].state, spec=peer.spec, generate=h.generate, suffix_seed=0
        )
    assert {d.decision_id for d in result.decisions}.isdisjoint(
        {d.decision_id for d in peer.decisions}
    )


def test_a_suffix_pair_refuses_the_wrong_parents(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        private = h.run(K.Protocol.PRIVATE, rounds=1)
        peer = h.run(K.Protocol.PEER, rounds=1)
        with pytest.raises(ProtocolError):
            run_suffix_pair(peer, private, generate=h.generate)
        with pytest.raises(ProtocolError):
            run_suffix(private.root_state, spec=private.spec, generate=h.generate, suffix_seed=0)
