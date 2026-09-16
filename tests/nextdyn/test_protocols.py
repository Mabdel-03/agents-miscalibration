"""Conditions as data: identity, declared equivalence, and the one packet-plan function."""

from __future__ import annotations

from dataclasses import replace

import pytest
import yaml

from agents_scaling.nextdyn import graphs as G
from agents_scaling.nextdyn import ids
from agents_scaling.nextdyn import protocols as P
from agents_scaling.nextdyn import states as S
from agents_scaling.nextdyn.errors import ConfigError, ContractViolation
from agents_scaling.nextdyn.schema import ParseStatus, PublicOutput

SEED = bytes.fromhex("22" * 32)
PAIRING = "item-1|team:abc|0123456789abcdef|budget8|rep0"


def _output(answer: str) -> PublicOutput:
    return PublicOutput(reasoning="r", shareable_claims=("c",), final_answer=answer,
                        confidence=None, field_order=("reasoning", "shareable_claims",
                                                      "final_answer"),
                        order_violation=False, parse_status=ParseStatus.OK, raw_text="{}",
                        content_hash=ids.content_hash(answer.encode()), over_limits={},
                        final_answer_span=None, confidence_span=None,
                        confidence_emitted_before_answer=False)


def _state(n: int = 5, round_: int = 0, graph=None) -> S.RoundState:
    refs = [S.DecisionRef(decision_id=f"d:{slot}", round_=round_, agent_slot=slot,
                          generation_uid=f"g:{slot}", output=_output(f"ans{slot}"))
            for slot in range(n)]
    state = S.initial_state(pairing_id=PAIRING, roots=refs, classify=lambda a: a,
                            study_seed=SEED, graph=graph)
    return replace(state, round_=round_)


def _registry() -> P.Registry:
    return P.load_registry()


# ---------------------------------------------------------------------------- identity


def test_a_condition_hash_excludes_every_label():
    spec = _registry().episodes["PEER"]
    obj = spec.identity_object()
    assert not {"display_label", "legacy_alias", "equivalent_to"} & set(obj)
    renamed = replace(spec, display_label="PEER", legacy_alias="SOMETHING_ELSE")
    assert renamed.condition_hash == spec.condition_hash


def test_in1_out1_hashes_equal_to_peer_because_its_resolved_graph_is_complete():
    registry = _registry()
    assert registry.episodes["IN1_OUT1"].condition_hash == registry.episodes["PEER"].condition_hash
    assert registry.episodes["IN1_OUT1"].equivalent_to == "PEER"
    for n in (3, 5, 9):
        assert G.resolve_graph(registry.episodes["IN1_OUT1"].graph, n=n, round_=1, focal_slot=1,
                               study_seed=SEED, pairing_id=PAIRING) == G.complete(n)


def test_arms_that_differ_in_content_or_filler_do_not_share_a_hash():
    registry = _registry()
    hashes = {name: spec.condition_hash for name, spec in registry.episodes.items()}
    assert hashes["PEER"] != hashes["PEER_ANSWER_ONLY"]
    assert hashes["IN0_OUT1"] != hashes["IN0_OUT1_FILLED"]
    assert hashes["PRIVATE"] != hashes["FRESH"]
    assert hashes["PRIVATE"] != hashes["IN0_OUT0"]


def test_episode_and_receiver_conditions_never_collide_across_the_two_kinds():
    registry = _registry()
    episode_hashes = {s.condition_hash for s in registry.episodes.values()}
    receiver_hashes = {s.condition_hash for s in registry.receivers.values()}
    assert not episode_hashes & receiver_hashes


def test_every_shipped_condition_carries_a_usable_protocol_hash():
    for spec in {**_registry().episodes, **_registry().receivers}.values():
        assert ids.condition_hash(spec.identity_object()) == spec.condition_hash
        assert len(spec.condition_hash) == 64


def test_the_registry_refuses_an_undeclared_equivalence(tmp_path):
    data = yaml.safe_load(P.DEFAULT_PROTOCOLS_PATH.read_text())
    data["episodes"]["IN1_OUT1"]["equivalent_to"] = None
    path = tmp_path / "protocols.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="equivalence is not declared"):
        P.load_registry(path)


def test_the_registry_refuses_a_false_equivalence(tmp_path):
    data = yaml.safe_load(P.DEFAULT_PROTOCOLS_PATH.read_text())
    data["episodes"]["IN0_OUT1"]["equivalent_to"] = "PEER"
    path = tmp_path / "protocols.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="declares equivalence"):
        P.load_registry(path)


def test_the_registry_refuses_a_duplicated_legacy_alias(tmp_path):
    data = yaml.safe_load(P.DEFAULT_PROTOCOLS_PATH.read_text())
    data["episodes"]["FRESH"]["legacy_alias"] = "C_FULL"
    path = tmp_path / "protocols.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="legacy_alias"):
        P.load_registry(path)


def test_the_registry_refuses_an_unknown_key_and_a_reserved_schedule(tmp_path):
    data = yaml.safe_load(P.DEFAULT_PROTOCOLS_PATH.read_text())
    data["episodes"]["PEER"]["colour"] = "blue"
    path = tmp_path / "protocols.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="unknown key"):
        P.load_registry(path)
    data = yaml.safe_load(P.DEFAULT_PROTOCOLS_PATH.read_text())
    data["episodes"]["PEER"]["schedule"] = "staggered"
    path2 = tmp_path / "p2.yaml"
    path2.write_text(yaml.safe_dump(data))
    with pytest.raises(ContractViolation, match="reserved"):
        P.load_registry(path2)


def test_the_shipped_registry_agrees_with_the_study_protocol_block(study_cfg):
    P.assert_agrees_with_study(study_cfg.protocol, _registry())


def test_the_legacy_ard36_names_map_one_to_one():
    registry = _registry()
    aliases = {spec.legacy_alias: name for name, spec in registry.episodes.items()
               if spec.legacy_alias}
    assert aliases == {"R_PRIVATE": "PRIVATE", "C_FULL": "PEER", "FRESH_N": "FRESH"}


# ---------------------------------------------------------------------------- packet plan


def test_private_has_no_peer_content():
    """PRIVATE's graph is empty, so every recipient's plan is empty and the prompt renders
    the literal NONE (spec 6.3)."""
    spec = _registry().episodes["PRIVATE"]
    state = _state(graph=G.empty(5))
    plan = P.packet_plan(state, spec, round_=1, study_seed=SEED)
    assert set(plan) == set(range(5))
    assert all(intents == () for intents in plan.values())


def test_peer_messages_use_only_previous_round():
    spec = _registry().episodes["PEER"]
    state = _state(round_=2, graph=G.complete(5))
    plan = P.packet_plan(state, spec, round_=3, study_seed=SEED)
    for intents in plan.values():
        assert [i.source_round for i in intents] == [2] * 4
        P.assert_packet_rounds(intents, round_=3)


def test_a_packet_from_the_wrong_round_is_refused_at_dispatch():
    intent = P.TransmissionIntent(sender_slot=1, source_round=0, source_decision_id="d:1",
                                  content=P.PacketContent.FULL, filler=False, display_position=0)
    with pytest.raises(ContractViolation, match="only round 2 packets"):
        P.assert_packet_rounds((intent,), round_=3)


def test_planning_a_round_from_the_wrong_snapshot_is_refused():
    spec = _registry().episodes["PEER"]
    state = _state(round_=0, graph=G.complete(5))
    with pytest.raises(ContractViolation, match="snapshot"):
        P.packet_plan(state, spec, round_=3, study_seed=SEED)


def test_the_display_order_is_shared_by_paired_arms_and_keyed_on_the_pairing():
    """Spec 6.4: paired arms deliver the senders they have in common in the same order."""
    registry = _registry()
    state = _state(graph=G.complete(5))
    peer = P.packet_plan(state, registry.episodes["PEER"], round_=1, study_seed=SEED)
    focal_state = _state(graph=G.resolve_graph(registry.episodes["IN0_OUT1"].graph, n=5, round_=1,
                                               focal_slot=2, study_seed=SEED, pairing_id=PAIRING))
    gated = P.packet_plan(focal_state, registry.episodes["IN0_OUT1"], round_=1, study_seed=SEED)
    for recipient in (0, 1, 3, 4):
        assert [i.sender_slot for i in peer[recipient]] == [i.sender_slot for i in gated[recipient]]
    assert gated[2] == ()


def test_filler_slots_occupy_the_positions_the_graph_removed():
    registry = _registry()
    plain = registry.episodes["IN0_OUT1"]
    filled = registry.episodes["IN0_OUT1_FILLED"]
    graph = G.resolve_graph(plain.graph, n=5, round_=1, focal_slot=2, study_seed=SEED,
                            pairing_id=PAIRING)
    state = _state(graph=graph)
    plan_plain = P.packet_plan(state, plain, round_=1, study_seed=SEED)
    plan_filled = P.packet_plan(state, filled, round_=1, study_seed=SEED)
    assert len(plan_filled[2]) == 4 and all(i.filler for i in plan_filled[2])
    assert plan_plain[2] == ()
    for recipient in (0, 1, 3, 4):
        assert plan_plain[recipient] == plan_filled[recipient]
    assert [i.display_position for i in plan_filled[2]] == [0, 1, 2, 3]


def test_a_filler_intent_never_names_a_source_decision():
    with pytest.raises(ContractViolation):
        P.TransmissionIntent(sender_slot=1, source_round=0, source_decision_id="d:1",
                             content=P.PacketContent.FULL, filler=True, display_position=0)


def test_the_plan_is_n_parametric():
    spec = _registry().episodes["PEER"]
    for n in (3, 5, 9):
        state = _state(n=n, graph=G.complete(n))
        plan = P.packet_plan(state, spec, round_=1, study_seed=SEED)
        assert set(plan) == set(range(n))
        assert all(len(v) == n - 1 for v in plan.values())


# ---------------------------------------------------------------------------- vendoring parity


def test_vendored_packet_order_matches_ard_under_the_same_namespace(monkeypatch):
    """Parity with ``ard/protocols.py:241``: same sort, same key shape, one namespace apart."""
    from agents_scaling.ard import contract as ARD_K
    from agents_scaling.ard import protocols as ARD_P
    from agents_scaling.nextdyn import contract as K

    monkeypatch.setattr(ARD_K, "ORDER_PACKET", K.ORDER_PACKET)

    class _State:
        study_seed = SEED
        pairing_id = PAIRING

    for round_ in range(1, 5):
        for recipient in range(5):
            senders = [j for j in range(5) if j != recipient]
            mine = P.packet_order(SEED, PAIRING, round_, recipient, senders)
            theirs = ARD_P.packet_order(_State(), round_=round_, recipient=recipient,
                                        senders=senders)
            assert mine == theirs
