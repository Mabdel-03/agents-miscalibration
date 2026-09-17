"""The declared public state: n-parametric, gold-free, and rebuildable (spec 6.4, 4.2-4.3)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from agents_scaling.nextdyn import ids, states as S
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.nextdyn.graphs import complete
from agents_scaling.nextdyn.schema import ParseStatus, PublicOutput
from agents_scaling.nextdyn.tables import SCHEMA_VERSION, ItemFile, ItemStatus

SEED = bytes.fromhex("44" * 32)
PAIRING = "item-1|team:abc|0123456789abcdef|budget8|rep0"


def out(answer: str | None, status: ParseStatus = ParseStatus.OK) -> PublicOutput:
    return PublicOutput(reasoning="r", shareable_claims=(), final_answer=answer, confidence=None,
                        field_order=(), order_violation=False, parse_status=status, raw_text="",
                        content_hash=ids.content_hash(str(answer).encode()), over_limits={},
                        final_answer_span=None, confidence_span=None,
                        confidence_emitted_before_answer=False)


def refs(answers, round_=0):
    return [S.DecisionRef(decision_id=f"d:{round_}:{slot}", round_=round_, agent_slot=slot,
                          generation_uid=f"g:{round_}:{slot}", output=out(answer))
            for slot, answer in enumerate(answers)]


def state(answers, **kw):
    return S.initial_state(pairing_id=PAIRING, roots=refs(answers), classify=lambda a: a,
                           study_seed=SEED, **kw)


def test_the_team_size_is_read_from_the_state_not_from_a_constant():
    for n in (3, 5, 9):
        assert state(["A"] * n).n == n


def test_pairing_id_carries_no_condition():
    a = S.pairing_id_for("item-1", "team:abc", "m16", "b8", 0)
    b = S.pairing_id_for("item-1", "team:abc", "m16", "b8", 1)
    assert a != b and "PEER" not in a and "PRIVATE" not in a


def test_pseudonyms_are_assigned_once_per_episode_and_are_stable():
    first, second = state(["A"] * 5), state(["B"] * 5)
    assert [a.pseudonym for a in first.agents] == [a.pseudonym for a in second.agents]
    assert len({a.pseudonym for a in first.agents}) == 5


def test_a_state_holds_no_grade_only_a_canonical_class():
    snapshot = state(["A", "A", "B", None, "A"])
    assert snapshot.classes() == ("A", "A", "B", None, "A")
    assert not hasattr(snapshot, "correct")


def test_unanimity_needs_every_slot_valid_and_equal():
    assert state(["A"] * 5).unanimous() == "A"
    assert state(["A", "A", "A", "A", "B"]).unanimous() is None
    assert state(["A", "A", "A", "A", None]).unanimous() is None


def test_four_revisions_produce_five_observed_states():
    """Spec 8.1: a root plus R revisions is R+1 observed states, counted on the state list."""
    current = state(["A"] * 5)
    observed = [current]
    for round_ in range(1, 5):
        current = S.advance(current, {slot: refs(["X"] * 5, round_=round_)[slot]
                                      for slot in range(5)}, round_=round_)
        observed.append(current)
    assert len(observed) == 5
    assert [s.round_ for s in observed] == [0, 1, 2, 3, 4]
    assert all(len(a.history) == 4 for a in observed[-1].agents)


def test_a_round_advances_only_when_every_slot_is_in():
    current = state(["A"] * 5)
    partial = {slot: refs(["X"] * 5, round_=1)[slot] for slot in range(4)}
    with pytest.raises(ContractViolation, match="all 5 slots"):
        S.advance(current, partial, round_=1)


def test_historical_coverage_is_nondecreasing_even_when_a_class_is_abandoned():
    current = state(["A", "B", "B", "B", "B"])
    assert current.historical_classes() == frozenset({"A", "B"})
    current = S.advance(current, {slot: refs(["B"] * 5, round_=1)[slot] for slot in range(5)},
                        round_=1)
    assert current.classes() == ("B",) * 5
    assert current.historical_classes() == frozenset({"A", "B"})


def test_state_id_is_content_addressed_over_the_public_state_only():
    a = state(["A", "A", "B", "B", "B"])
    b = state(["A", "A", "B", "B", "B"])
    c = state(["A", "A", "A", "B", "B"])
    assert S.state_id(a) == S.state_id(b) != S.state_id(c)
    assert S.state_id(a).startswith(S.STATE_ID_PREFIX)
    assert S.state_id(replace(a, graph=complete(5))) == S.state_id(a)


def test_agents_must_be_in_slot_order():
    good = state(["A"] * 3)
    with pytest.raises(ContractViolation, match="slot order"):
        replace(good, agents=tuple(reversed(good.agents)))


def _decision_rows(rows_module, n=3, rounds=2):
    """Rows built from the schema, so a new column does not make this a different fixture.

    It used to spell out every ``decisions`` column, and so did two other modules; each time
    ``tables.py`` gained one -- ``escape_repaired``, then ``trailing_chars`` and
    ``plan_parsed`` -- all three broke at once.  ``conftest.placeholder`` fills the rest by
    type, and this names only the fields the states test is actually about.
    """
    from agents_scaling.nextdyn import tables
    from tests.nextdyn.conftest import placeholder

    rows = []
    for round_ in range(rounds + 1):
        for slot in range(n):
            row = placeholder("decisions") | dict(
                decision_id=f"d:{round_}:{slot}", study_id="nd1", cell_kind="episode",
                source_item_id="item-1", item_variant_id="item-1.v0", root_team_id="team:abc",
                protocol_hash="ph", receiver_model_hash="0123456789abcdef",
                budget_hash="budget8", replicate_id=0, round=round_, agent_slot=slot,
                phase="root_ref" if round_ == 0 else "revision",
                episode_id="episode|x", cell_id="cell", display_label="PEER",
                generation_uid=f"g:{round_}:{slot}", content_hash="c:0",
                request_hashes_json="[]", aliased=False, is_reference=round_ == 0,
                parent_decision_ids_json="[]", seen_transmission_ids_json="[]",
                public_reasoning="r", shareable_claims_json=json.dumps([]),
                final_answer_raw="A" if slot == 0 else "B",
                parse_status="ok", validity=True, delivery_success=True,
                infrastructure_status="ok", termination_reason="stop",
                thinking_tokens=1, final_tokens=1, prompt_tokens=1)
            rows.append(tables.row_from_dict(tables.DecisionRow, row))
    return tuple(rows)


def test_rebuild_round_states_reconstructs_the_same_states_from_an_item_file():
    import agents_scaling.nextdyn.tables as T

    item = ItemFile(schema_version=SCHEMA_VERSION, cell_id="cell", cell_kind="episode",
                    source_item_id="item-1", status=ItemStatus.COMPLETE, bundle_export_hash=None,
                    rows={"decisions": _decision_rows(T, n=3, rounds=2)})
    rebuilt = S.rebuild_round_states(item, lambda a: a, study_seed=SEED)
    assert [s.round_ for s in rebuilt] == [0, 1, 2]
    assert [s.n for s in rebuilt] == [3, 3, 3]
    assert rebuilt[0].pairing_id == S.pairing_id_for("item-1", "team:abc", "0123456789abcdef",
                                                     "budget8", 0)
    assert rebuilt[-1].classes() == ("A", "B", "B")
    assert rebuilt[-1].historical_classes() == frozenset({"A", "B"})


def test_rebuilding_an_item_file_with_two_episodes_needs_the_episode_named():
    import agents_scaling.nextdyn.tables as T
    from dataclasses import replace as dc_replace

    rows = _decision_rows(T, n=2, rounds=1)
    mixed = (*rows, *(dc_replace(r, episode_id="episode|y", decision_id=r.decision_id + "b")
                      for r in rows))
    item = ItemFile(schema_version=SCHEMA_VERSION, cell_id="cell", cell_kind="episode",
                    source_item_id="item-1", status=ItemStatus.COMPLETE, bundle_export_hash=None,
                    rows={"decisions": mixed})
    with pytest.raises(ContractViolation, match="pass episode_id"):
        S.rebuild_round_states(item, lambda a: a, study_seed=SEED)
    assert len(S.rebuild_round_states(item, lambda a: a, study_seed=SEED,
                                      episode_id="episode|y")) == 2
