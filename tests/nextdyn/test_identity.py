"""Identity scheme (spec 17.2, 18.1; plan §1.3, D4, R4, R5).

The four spec-18.1 names WP0 owns are used verbatim.  Everything here is synthetic: no model,
no store, only the id builders and ``study.types.RequestSpec.request_id``.  Every key carries
a 16-hex ``receiver_model_hash`` because the generating model is part of every key.
"""

from __future__ import annotations

import dataclasses

import pytest

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids
from agents_scaling.nextdyn.errors import IdentityError
from agents_scaling.study.types import Checkpoint, Decoding, RequestSpec, SeedKey

TEAM = "team:" + "ab" * 16
COND = "c" * 64
BUDGET = "b" * 64
MODEL16 = "1234567890abcdef"          # the 8B cell
MODEL16_OTHER = "fedcba0987654321"    # the 32B cell


def _key(**over):
    base = dict(study_id="task_conditioned_dynamics_v1", cell_kind=ids.CellKind.EPISODE,
                source_item_id="item01", item_variant_id="canonical", root_team_id=TEAM,
                protocol_hash=COND, receiver_model_hash=MODEL16, budget_hash=BUDGET, replicate_id=0,
                round=1, agent_slot=2, phase=K.PHASE_REVISION)
    base.update(over)
    return ids.CompositeKey(**base)


def _bank_key(**over):
    base = dict(cell_kind=ids.CellKind.DONOR_BANK, root_team_id="", protocol_hash="", round=0,
                agent_slot=7, phase=K.PHASE_BANK)
    base.update(over)
    return _key(**base)


# ---------------------------------------------------------------------------- 18.1 names


def test_source_item_id_never_null():
    with pytest.raises(IdentityError):
        _key(source_item_id="")
    with pytest.raises(IdentityError):
        _key(source_item_id=None)  # type: ignore[arg-type]
    assert _key().source_item_id == "item01"


def test_bank_role_cannot_alias_private_episode():
    """R4 / defect D-11: a bank draw and an episode row can never share an id."""
    with pytest.raises(IdentityError):
        ids.episode_id(ids.CellKind.DONOR_BANK, TEAM, COND, MODEL16, BUDGET, 0)
    with pytest.raises(IdentityError):
        ids.bank_draw_id(ids.CellKind.EPISODE, "item01", "canonical", MODEL16, BUDGET, 0)
    # a bank draw is never a member of a root team
    with pytest.raises(IdentityError):
        _bank_key(root_team_id=TEAM)
    # and a bank draw at the same (item, model, budget, slot) as a PRIVATE round-0 row differs
    bank = _bank_key(agent_slot=0)
    ref = _key(round=0, agent_slot=0, phase=K.PHASE_ROOT_REF)
    assert bank.receiver_model_hash == ref.receiver_model_hash == MODEL16
    assert bank.decision_id() != ref.decision_id()
    assert ids.bank_draw_id(ids.CellKind.DONOR_BANK, "item01", "canonical", MODEL16, BUDGET, 0) \
        .startswith(ids.CellKind.DONOR_BANK.value + "|")
    assert ids.episode_id(ids.CellKind.EPISODE, TEAM, COND, MODEL16, BUDGET, 0).startswith("episode|")


def test_same_content_does_not_imply_same_sample():
    """Equal output bytes may come from two independent draws: content_hash coincides, the
    generation uids (from distinct physical request ids) do not."""
    text = b'{"reasoning": "x", "shareable_claims": [], "final_answer": "42", "confidence": 0.5}'
    assert ids.content_hash(text) == ids.content_hash(bytes(text))
    assert ids.generation_uid(["req-a"]) != ids.generation_uid(["req-b"])


def test_request_cache_invalidates_on_model_prompt_seed_or_parent_change(study_cfg):
    """R5: every change of model, prompt bytes, seed key or parent changes ``request_id``;
    a change of *condition alone* (protocol_hash) does not (common random numbers, D4)."""
    contract = study_cfg.contract
    key = _key()
    seed_key = ids.seed_key_for(key, split_role="pilot")
    decoding = Decoding(0.6, 0.95, 20, 0.0, 0.0, 1.0, contract.tokens.max_new_tokens, True,
                        False, contract.tokens.thinking_cap, contract.tokens.final_cap)
    ckpt = Checkpoint(size="8B", hf_id="Qwen/Qwen3-8B", model_revision="b968826d" * 5,
                      tokenizer_revision="b968826d" * 5, profile="nd1-8B", tp_size=1,
                      served_model_name="8B")
    messages = ({"role": "system", "content": "solve"},
                {"role": "user", "content": "QUESTION\nq\nOTHER DRAFTS\n[A] parent=d:parent1"})

    def request(msgs=messages, ck=ckpt, sk=seed_key):
        return RequestSpec(messages=msgs, decoding=decoding, checkpoint=ck, seed_key=sk,
                           role="revision", study_id=contract.study_id,
                           study_seed_hex=contract.study_seed_hex).request_id

    base = request()
    assert request() == base

    other_model = dataclasses.replace(ckpt, model_revision="9216db57" * 5)
    assert request(ck=other_model) != base

    other_prompt = (messages[0], {"role": "user", "content": messages[1]["content"] + " "})
    assert request(msgs=other_prompt) != base

    other_parent = (messages[0], {"role": "user", "content": messages[1]["content"].replace("parent1", "parent2")})
    assert request(msgs=other_parent) != base

    other_round = ids.seed_key_for(_key(round=2), split_role="pilot")
    assert request(sk=other_round) != base
    other_team = ids.seed_key_for(_key(root_team_id="team:" + "cd" * 16), split_role="pilot")
    assert request(sk=other_team) != base

    # the condition is NOT in the seed: PRIVATE and PEER keys of one slot share the stream
    peer = ids.seed_key_for(_key(protocol_hash="d" * 64), split_role="pilot")
    assert peer == seed_key
    assert request(sk=peer) == base
    # nor is the budget
    assert ids.seed_key_for(_key(budget_hash="e" * 64), split_role="pilot") == seed_key


# ---------------------------------------------------------------------------- composite key


def test_composite_key_has_exactly_the_twelve_spec_fields_in_order():
    assert ids.FIELD_ORDER == (
        "study_id", "cell_kind", "source_item_id", "item_variant_id", "root_team_id", "protocol_hash",
        "receiver_model_hash", "budget_hash", "replicate_id", "round", "agent_slot", "phase")
    key = _key()
    assert key.as_array()[1] is ids.CellKind.EPISODE
    assert key.decision_id().startswith("d:") and len(key.decision_id()) == 42


def test_decision_id_changes_with_every_key_component():
    base = _key().decision_id()
    variants = [
        _key(source_item_id="item02"), _key(item_variant_id="option_perm_v0"), _key(root_team_id="team:" + "cd" * 16),
        _key(protocol_hash="d" * 64), _key(receiver_model_hash=MODEL16_OTHER), _key(budget_hash="e" * 64),
        _key(replicate_id=1), _key(round=2), _key(agent_slot=3), _key(phase=K.PHASE_ROOT_REF),
        _key(cell_kind=ids.CellKind.RECEIVER, phase=K.PHASE_RECEIVER_DECISION),
    ]
    seen = {base}
    for variant in variants:
        assert variant.decision_id() not in seen
        seen.add(variant.decision_id())


def test_two_models_draws_of_one_item_have_distinct_decision_ids():
    """``receiver_model_hash`` is the generating model for EVERY kind: the 8B and 32B donor
    draws of one item, and the 8B and 32B episodes of one team (E3c), never collide."""
    draw_8b = _bank_key(agent_slot=3, receiver_model_hash=MODEL16)
    draw_32b = _bank_key(agent_slot=3, receiver_model_hash=MODEL16_OTHER)
    assert draw_8b.decision_id() != draw_32b.decision_id()
    episode_8b = _key(receiver_model_hash=MODEL16)
    episode_32b = _key(receiver_model_hash=MODEL16_OTHER)
    assert episode_8b.decision_id() != episode_32b.decision_id()
    # and the seed streams differ too (the model is pinned into SeedKey.model_cell from the key)
    assert ids.seed_key_for(episode_8b, split_role="x") != ids.seed_key_for(episode_32b, split_role="x")
    # an empty model hash is refused for every kind that generates decisions
    for kind, extra in ((ids.CellKind.DONOR_BANK, dict(root_team_id="", protocol_hash="", round=0, phase=K.PHASE_BANK)),
                        (ids.CellKind.EPISODE, {}),
                        (ids.CellKind.RECEIVER, dict(phase=K.PHASE_RECEIVER_DECISION, round=0, agent_slot=0)),
                        (ids.CellKind.SELECTION_AUDIT, dict(phase=K.PHASE_AUDIT))):
        with pytest.raises(IdentityError, match="receiver_model_hash"):
            _key(cell_kind=kind, receiver_model_hash="", **extra)
    # kinds that generate nothing carry no key
    with pytest.raises(IdentityError):
        _key(cell_kind=ids.CellKind.GRADE)


def test_bank_keys_live_at_round_zero_in_the_bank_phase():
    with pytest.raises(IdentityError):
        _bank_key(round=1)
    with pytest.raises(IdentityError):
        _bank_key(phase=K.PHASE_REVISION)
    with pytest.raises(IdentityError):
        _key(phase="root")  # not a declared phase


def test_seed_namespace_carries_kind_variant_and_team_but_no_condition_or_budget():
    key = _key()
    seed = ids.seed_key_for(key, split_role="natural")
    assert isinstance(seed, SeedKey)
    assert seed.namespace == f"ND1/episode/canonical/{TEAM}"
    assert seed.source_id == "item01" and seed.step_slot == 1 and seed.actor_slot == 2
    assert seed.purpose == K.PHASE_REVISION and seed.episode_rep == 0
    assert seed.split == "natural" and seed.model_cell == MODEL16      # pinned from the key
    assert COND not in seed.namespace and BUDGET not in seed.namespace
    assert seed.namespace.startswith(K.SEED_NAMESPACE_PREFIX + "/")
    with pytest.raises(IdentityError):
        ids.seed_key_for(key, split_role="")


def test_generation_uid_covers_both_phases_of_a_continued_draw():
    one = ids.generation_uid(["req-1"])
    two = ids.generation_uid(["req-1", "req-2"])
    assert one != two and one.startswith("g:") and len(two) == 2 + 64     # untruncated (plan §1.3)
    with pytest.raises(IdentityError):
        ids.generation_uid([])
    with pytest.raises(IdentityError):
        ids.generation_uid(["req-1", "req-1"])


def test_root_team_id_is_content_addressed_over_ordered_members():
    members = ["g:" + "1" * 32, "g:" + "2" * 32, "g:" + "3" * 32]
    a = ids.root_team_id(ids.CellKind.DONOR_BANK, "item01", "canonical", members, 0, "v1")
    b = ids.root_team_id(ids.CellKind.DONOR_BANK, "item01", "canonical", members, 0, "v1")
    c = ids.root_team_id(ids.CellKind.DONOR_BANK, "item01", "canonical", members[::-1], 0, "v1")
    assert a == b != c and a.startswith("team:") and len(a) == 37
    with pytest.raises(IdentityError):
        ids.root_team_id(ids.CellKind.DONOR_BANK, "item01", "canonical", members + [members[0]], 0, "v1")
    with pytest.raises(IdentityError):
        ids.root_team_id(ids.CellKind.DONOR_BANK, "item01", "canonical", members, 5, "v1")


def test_condition_hash_refuses_labels_and_is_symbolic():
    obj = {"own_draft": True, "graph": {"kind": "fixed", "ref": {"name": "complete", "params": {}}},
           "packet_content": "full", "filler": "none", "template": "revision", "schedule": "synchronous"}
    assert ids.condition_hash(obj) == ids.condition_hash(dict(reversed(list(obj.items()))))
    with pytest.raises(IdentityError):
        ids.condition_hash({**obj, "display_label": "PEER"})
    with pytest.raises(IdentityError):
        ids.condition_hash({})


def test_model_hash_needs_pinned_revisions(study_cfg):
    contract = study_cfg.contract
    primary = ids.model_hash(contract.model("primary"))
    assert len(primary) == 16 and primary != ids.model_hash(contract.model("scale"))
    # same weights, different adapter -> different model hash (the companion is a different cell)
    assert primary != ids.model_hash(contract.model("qwen_nonthinking"))
    with pytest.raises(IdentityError):
        ids.model_hash(contract.model("family_transfer"))


def test_blind_order_keys_require_a_declared_namespace(study_cfg):
    seed = study_cfg.contract.study_seed
    a = ids.blind(seed, K.ORDER_PACKET, "pairing", 1, 0)
    assert a == ids.blind(seed, K.ORDER_PACKET, "pairing", 1, 0) != ids.blind(seed, K.ORDER_PSEUDONYM, "pairing", 1, 0)
    with pytest.raises(IdentityError):
        ids.blind(seed, "ARD_PACKET_ORDER", "pairing")


def test_episode_and_bank_ids_start_with_the_cell_kind():
    ep = ids.episode_id(ids.CellKind.RECEIVER, "packets:" + "f" * 40, COND, MODEL16, BUDGET, 1)
    assert ep.split("|")[0] == "receiver" and ep.endswith("|rep1")
    draw = ids.bank_draw_id(ids.CellKind.DIFFICULTY_BANK, "item01", "canonical", MODEL16, BUDGET, 15)
    assert draw.split("|") == ["difficulty_bank", "item01", "canonical", MODEL16, "b" * 8, "k15"]
    with pytest.raises(IdentityError):
        ids.episode_id(ids.CellKind.GRADE, TEAM, COND, MODEL16, BUDGET, 0)
    audit = ids.audit_episode_id("pool:" + "9" * 64, MODEL16_OTHER, BUDGET)
    assert audit.split("|") == ["selection_audit", "9" * 12, MODEL16_OTHER, "b" * 8]


def test_short_panel_ids_are_hashed_never_truncated_onto_each_other():
    """An E2b problem id is shorter than the twelve characters an episode id needs: it is
    hashed to twelve, so distinct short ids stay distinct and no producer has to pad."""
    a = ids.episode_id(ids.CellKind.RECEIVER, "mod97:d4:i003", COND, MODEL16, BUDGET, 0)
    b = ids.episode_id(ids.CellKind.RECEIVER, "mod97:d4:i004", COND, MODEL16, BUDGET, 0)
    c = ids.episode_id(ids.CellKind.RECEIVER, "p003", COND, MODEL16, BUDGET, 0)
    assert len({a, b, c}) == 3 and all(len(x.split("|")[1]) == 12 for x in (a, b, c))
    assert a == ids.episode_id(ids.CellKind.RECEIVER, "mod97:d4:i003", COND, MODEL16, BUDGET, 0)
    # a long token keeps its own prefix (the team12 rule of plan §1.3)
    assert ids.episode_id(ids.CellKind.EPISODE, TEAM, COND, MODEL16, BUDGET, 0).split("|")[1] == "ab" * 6
