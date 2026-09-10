"""ARD36 WP2: the packet plan of every protocol (spec §7, §12.3; plan §5.2).

These are pure tests over hand-built :class:`~agents_scaling.ard.states.RoundState`s, so a
specified class multiset and a specified invalid peer can be asserted exactly.  The
episode-level claims ("asserted on the recorded request bodies") live in
``test_wp2_episode.py``.
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import protocols as P
from agents_scaling.ard.packets import PACKET_KIND_UNPARSED, UNPARSED_PEER_JSON
from agents_scaling.study.types import PACKET_UNAVAILABLE_JSON, ProtocolError
from tests.ard import wp2_support as W


def state(answers, **kwargs):
    return W.round_state([W.report(a) for a in answers], **kwargs)


# --------------------------------------------------------------------------- empty packets


@pytest.mark.parametrize("protocol", [K.Protocol.PRIVATE, K.Protocol.FRESH])
def test_private_and_fresh_have_empty_packets(protocol) -> None:
    plan = P.packet_plan(state(["42"] * 5, protocol=protocol), protocol, round_=1)
    assert plan == {slot: () for slot in range(K.N_AGENTS)}


# --------------------------------------------------------------------------- PEER


def test_peer_gives_the_other_four_previous_round_reports() -> None:
    st = state(["42", "42", "7", "7", "9"])
    plan = P.packet_plan(st, K.Protocol.PEER, round_=1)
    for recipient, packets in plan.items():
        assert len(packets) == K.N_AGENTS - 1
        assert all(p.source_round == 0 for p in packets)
        senders = {p.sender_slot for p in packets}
        assert senders == set(range(K.N_AGENTS)) - {recipient}
        assert st.pseudonyms[recipient] not in {p.pseudonym for p in packets}


def test_peer_order_is_the_recorded_blind_schedule() -> None:
    st = state(["42", "43", "44", "45", "46"])
    plan = P.packet_plan(st, K.Protocol.PEER, round_=1)
    for recipient, packets in plan.items():
        expected = P.packet_order(
            st, round_=1, recipient=recipient, senders=[s for s in range(K.N_AGENTS) if s != recipient]
        )
        assert tuple(p.sender_slot for p in packets) == expected
    # blind: the order does not follow slot order for every recipient
    assert any(
        tuple(p.sender_slot for p in plan[r]) != tuple(s for s in range(K.N_AGENTS) if s != r)
        for r in range(K.N_AGENTS)
    )


def test_an_invalid_peer_is_a_typed_unavailable_entry_not_an_invented_answer() -> None:
    st = W.round_state([W.report("42"), W.report(None), W.report("7"), W.report("7"), W.report("9")])
    packets = P.packet_plan(st, K.Protocol.PEER, round_=1)[0]
    unavailable = [p for p in packets if p.serialized == PACKET_UNAVAILABLE_JSON]
    assert len(unavailable) == 1
    assert len(packets) == K.N_AGENTS - 1        # the slot is still spent


def test_a_synchronous_protocol_refuses_a_stale_state() -> None:
    st = state(["42"] * 5)                        # a round-0 state
    with pytest.raises(ProtocolError):
        P.packet_plan(st, K.Protocol.PEER, round_=2)
    with pytest.raises(ValueError):
        P.packet_plan(st, K.Protocol.PEER, round_=0)


# --------------------------------------------------------------------------- BALANCED


def test_balanced_keeps_one_packet_per_canonical_class() -> None:
    st = state(["42", "42", "42", "7", "9"])
    plan = P.packet_plan(st, K.Protocol.BALANCED, round_=1)
    # recipient 4 sees peers {42, 42, 42, 7} -> two classes, two packets
    assert len(plan[4]) == 2
    answers = sorted(json.loads(p.serialized)["final_answer"] for p in plan[4])
    assert answers == ["42", "7"]
    # recipient 0 sees {42, 42, 7, 9} -> three classes
    assert len(plan[0]) == 3


def test_balanced_bytes_are_byte_identical_to_the_peer_packet_of_that_sender() -> None:
    """Multiplicity is hidden by *dropping* duplicates, never by annotating a count."""
    st = state(["42", "42", "42", "7", "9"])
    balanced = P.packet_plan(st, K.Protocol.BALANCED, round_=1)
    peer = P.packet_plan(st, K.Protocol.PEER, round_=1)
    for recipient, packets in balanced.items():
        peer_bytes = {p.serialized for p in peer[recipient]}
        for packet in packets:
            assert packet.serialized in peer_bytes
            body = json.loads(packet.serialized)
            assert set(body) == {"from", "final_answer", "evidence", "truncated"}
            assert "3" not in json.dumps(body.get("truncated"))


def test_balanced_representative_is_the_frozen_schedule_not_confidence_or_order() -> None:
    high = [W.report("42", confidence=0.99), W.report("42", confidence=0.01)]
    low = [W.report("42", confidence=0.01), W.report("42", confidence=0.99)]
    rest = [W.report("7"), W.report("9"), W.report("11")]
    a = P.packet_plan(W.round_state(high + rest), K.Protocol.BALANCED, round_=1)[4]
    b = P.packet_plan(W.round_state(low + rest), K.Protocol.BALANCED, round_=1)[4]
    rep_a = [p.sender_slot for p in a if json.loads(p.serialized)["final_answer"] == "42"]
    rep_b = [p.sender_slot for p in b if json.loads(p.serialized)["final_answer"] == "42"]
    assert rep_a == rep_b        # swapping the confidences did not move the representative


def test_balanced_collapses_invalid_peers_into_one_typed_entry() -> None:
    st = W.round_state(
        [W.report("42"), W.report(None), W.report(None), W.report("7"), W.report("9")]
    )
    plan = P.packet_plan(st, K.Protocol.BALANCED, round_=1)
    collapsed = P.balanced_collapsed_counts(st, round_=1)
    # recipient 0 sees peers {invalid, invalid, 7, 9}: two classes + ONE unparsed entry
    unparsed = [p for p in plan[0] if p.kind == PACKET_KIND_UNPARSED]
    assert len(unparsed) == 1
    assert unparsed[0].serialized == UNPARSED_PEER_JSON
    assert len(plan[0]) == 3
    assert collapsed[0] == 2                      # the count is recorded outside the bytes
    assert collapsed[3] == 2 and collapsed[1] == 1


def test_balanced_packet_count_is_a_function_of_the_class_multiset() -> None:
    unanimous = P.packet_plan(state(["42"] * 5), K.Protocol.BALANCED, round_=1)
    assert all(len(p) == 1 for p in unanimous.values())
    distinct = P.packet_plan(state(["1", "2", "3", "4", "5"]), K.Protocol.BALANCED, round_=1)
    assert all(len(p) == 4 for p in distinct.values())


# --------------------------------------------------------------------------- HUB5


def test_hub5_is_a_one_plus_four_update() -> None:
    st = state(["42", "43", "44", "45", "46"], protocol=K.Protocol.HUB5)
    leader = P.hub_leader(st)
    plan = P.packet_plan(st, K.Protocol.HUB5, round_=1)
    assert len(plan[leader]) == K.N_AGENTS - 1
    assert all(p.source_round == 0 for p in plan[leader])
    assert all(plan[s] == () for s in range(K.N_AGENTS) if s != leader)

    leader_ref = W.ref(W.report("99"), slot=leader, round_=1)
    workers = P.hub_worker_plan(st, leader_ref, round_=1)
    assert workers[leader] == ()
    for slot in range(K.N_AGENTS):
        if slot == leader:
            continue
        assert len(workers[slot]) == 1
        packet = workers[slot][0]
        assert packet.source_round == 1               # the *current* leader report
        assert packet.sender_slot == leader
        assert json.loads(packet.serialized)["final_answer"] == "99"


def test_hub5_leader_is_fixed_for_the_episode_and_blind() -> None:
    st = state(["42"] * 5, protocol=K.Protocol.HUB5)
    assert P.hub_leader(st) == P.hub_leader(st)
    leaders = {P.hub_leader(state(["42"] * 5, item_id=f"i{i}", protocol=K.Protocol.HUB5)) for i in range(40)}
    assert len(leaders) > 1
    with pytest.raises(ProtocolError):
        P.hub_worker_plan(st, W.ref(W.report("99"), slot=0, round_=0), round_=1)


# --------------------------------------------------------------------------- protection


def test_gate_gives_exactly_one_empty_packet_and_peer_packets_to_the_rest() -> None:
    st = state(["42", "43", "44", "45", "46"])
    protected = 2
    plan = P.packet_plan(st, K.Protocol.GATE, round_=1, protected_slot=protected)
    assert plan[protected] == ()
    assert sum(1 for packets in plan.values() if not packets) == 1
    for slot, packets in plan.items():
        if slot == protected:
            continue
        assert len(packets) == K.N_AGENTS - 1
        assert protected in {p.sender_slot for p in packets}   # j still transmits (§12.3)


def test_gate_packets_are_byte_identical_to_the_paired_peer_episode() -> None:
    """Spec §12.3: the intervention changes j's *incoming* information, nothing else."""
    reports = [W.report(a) for a in ("42", "43", "44", "45", "46")]
    peer_state = W.round_state(reports, protocol=K.Protocol.PEER)
    gate_state = W.round_state(reports, protocol=K.Protocol.GATE)
    peer = P.packet_plan(peer_state, K.Protocol.PEER, round_=1)
    gate = P.packet_plan(gate_state, K.Protocol.GATE, round_=1, protected_slot=1)
    for slot in range(K.N_AGENTS):
        if slot == 1:
            assert gate[slot] == ()
            continue
        assert [p.serialized for p in gate[slot]] == [p.serialized for p in peer[slot]]


@pytest.mark.parametrize("protocol", sorted(K.PROTECTION_PROTOCOLS, key=lambda p: p.value))
def test_every_protection_protocol_protects_exactly_one_agent(protocol) -> None:
    st = state(["42", "43", "44", "45", "46"], protocol=protocol)
    slot = P.protected_slot_for(st, protocol, round_=1, controller_choice=3)
    plan = P.packet_plan(st, protocol, round_=1, protected_slot=slot)
    assert sum(1 for packets in plan.values() if not packets) == 1


def test_gate_refuses_to_guess_its_protected_agent() -> None:
    st = state(["42"] * 5, protocol=K.Protocol.GATE)
    with pytest.raises(ProtocolError):
        P.protected_slot_for(st, K.Protocol.GATE, round_=1)
    with pytest.raises(ProtocolError):
        P.packet_plan(st, K.Protocol.GATE, round_=1)


def test_random_protection_is_a_recorded_uniform_schedule() -> None:
    st = state(["42"] * 5, protocol=K.Protocol.RANDOM_PROTECT)
    assert P.random_protected_slot(st, round_=1) == P.random_protected_slot(st, round_=1)
    picks = {
        P.random_protected_slot(
            state(["42"] * 5, item_id=f"i{i}", protocol=K.Protocol.RANDOM_PROTECT), round_=1
        )
        for i in range(60)
    }
    assert picks == set(range(K.N_AGENTS))


def test_confidence_protection_takes_the_most_confident_and_never_imputes() -> None:
    reports = [
        W.report("42", confidence=0.1),
        W.report("43", confidence=0.9),
        W.report("44", confidence=None),
        W.report("45", confidence=0.5),
        W.report("46", confidence=0.2),
    ]
    st = W.round_state(reports, protocol=K.Protocol.CONF_PROTECT)
    assert P.confidence_protected_slot(st, round_=1) == 1
    # nobody reports one -> the blind schedule decides, not slot 0 by accident
    blind = W.round_state(
        [W.report(str(i), confidence=None) for i in range(5)], protocol=K.Protocol.CONF_PROTECT
    )
    assert P.confidence_protected_slot(blind, round_=1) == P.random_protected_slot(blind, round_=1)


# --------------------------------------------------------------------------- prompts


def test_the_update_prompt_deduplicates_an_identical_root_and_latest() -> None:
    root = W.report("42")
    once = P.render_update_prompt("task", root, root, ())
    twice = P.render_update_prompt("task", root, W.report("7"), ())
    assert once[0]["content"].count(f"{K.FINAL_OPEN}42{K.FINAL_CLOSE}") == 1
    assert "Your latest report:" not in once[0]["content"]
    assert "Your latest report:" in twice[0]["content"]


def test_the_root_prompt_names_no_slot_and_no_protocol() -> None:
    content = P.render_root_prompt("what is 6*7")[0]["content"]
    for protocol in K.Protocol:
        assert protocol.value not in content
    assert "agent-" not in content


def test_an_empty_packet_renders_identically_for_private_and_a_protected_agent() -> None:
    root, latest = W.report("42"), W.report("7")
    assert P.render_update_prompt("t", root, latest, ()) == P.render_update_prompt("t", root, latest, [])
