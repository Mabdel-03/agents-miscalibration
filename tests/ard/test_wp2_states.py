"""ARD36 WP2: the ``state:<sha>`` register (plan §5.3; ``study/selection/seal.py`` shape).

A state is ``(item, checkpoint, agent, round, own root report, own latest report)`` — spec
§6.4's deployed memory and nothing else — so two protocols that reach the same memory share
a state id.  The register is append-only and immutable, which is what makes ``state:`` a
usable hard dependency for FORK and SUFFIX cells.
"""

from __future__ import annotations

import dataclasses

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard.states import STATE_PREFIX, StateRegister, states_path
from agents_scaling.study.types import ProtocolError
from tests.ard import wp2_support as W


def make_state(answer: str = "42", *, protocol=K.Protocol.PEER, round_: int = 0):
    st = W.round_state([W.report(answer)] * K.N_AGENTS, protocol=protocol, round_=round_)
    return st.agents[0]


def test_state_id_is_content_addressed_and_protocol_blind() -> None:
    peer = make_state(protocol=K.Protocol.PEER)
    private = make_state(protocol=K.Protocol.PRIVATE)
    assert peer.state_id.startswith(STATE_PREFIX)
    assert peer.state_id == private.state_id          # the t=0 root state is shared (§12.1)
    assert peer.episode_id != private.episode_id      # provenance still differs
    assert make_state("7").state_id != peer.state_id


def test_a_state_at_round_zero_is_deduplicated() -> None:
    state = make_state()
    assert state.deduplicated is True
    changed = dataclasses.replace(state, latest=W.ref(W.report("7"), slot=0, round_=1), round_=1)
    assert changed.deduplicated is False
    assert changed.state_id != state.state_id


def test_the_register_is_append_only_and_immutable(tmp_path) -> None:
    register = StateRegister(states_path(tmp_path))
    peer = make_state(protocol=K.Protocol.PEER)
    private = make_state(protocol=K.Protocol.PRIVATE)

    assert register.merge([peer]) == (1, 0)
    assert register.merge([peer]) == (0, 1)                 # re-assertion is a no-op
    assert register.merge([private]) == (0, 1)              # same state, other arm
    assert len(register) == 1
    assert sorted(register.get(peer.state_id)["episodes"]) == sorted(
        {peer.episode_id, private.episode_id}
    )

    register.register["states"][peer.state_id]["root_content_sha256"] = "tampered"
    with pytest.raises(ProtocolError):
        register.merge([peer])


def test_the_register_round_trips_and_refuses_junk(tmp_path) -> None:
    path = states_path(tmp_path)
    register = StateRegister(path)
    register.merge([make_state("42"), make_state("7")])
    register.write()

    reloaded = StateRegister(path)
    assert reloaded.ids() == register.ids()
    assert len(reloaded) == 2
    assert reloaded.require(make_state("42").state_id)["item_id"] == "math500:test-0001"
    with pytest.raises(ProtocolError):
        reloaded.require("state:deadbeef")

    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ProtocolError):
        StateRegister(path)


def test_every_agent_of_an_episode_is_registered(tmp_path) -> None:
    with W.harness(tmp_path) as h:
        episode = h.run(K.Protocol.PRIVATE, rounds=1)
    register = StateRegister(states_path(tmp_path / "run"))
    register.merge(episode.states())
    # five agents x (root + one round), and the round-1 states are new
    assert len(register) == K.N_AGENTS * 2
