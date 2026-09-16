"""Spec 18.3 measurement guarantees for ``outcomes.py`` (plan §4.1).

Eight of the seventeen spec-18.3 names live here.  Each one pins a distinction ARD36 lost:
a three-state count that sums, an identity that holds on every consecutive pair, a switch of
answer that is not a change of validity, a unanimity that needs a shared class, coverage that
cannot go backwards in history, and bounds that bracket every feasible labelling of the
unresolved slots.
"""

from __future__ import annotations

import pytest

from agents_scaling.nextdyn import outcomes
from agents_scaling.nextdyn.errors import ContractViolation, NextdynError

C, W, U = outcomes.STATE_CORRECT, outcomes.STATE_WRONG, outcomes.STATE_UNRESOLVED


def state(slot: int, kind: str, cls: str | None = None, *, valid: bool | None = None,
          confidence: float | None = None) -> outcomes.AgentState:
    """One slot's state; ``cls`` defaults to a class consistent with the state."""
    if cls is None:
        cls = {C: "right", W: "wrong_a"}.get(kind)
    if valid is None:
        valid = kind in (C, W)
    return outcomes.AgentState(slot=slot, valid=valid, cls=cls, state=kind,
                               u_reason=None if kind != U else "parse_failed",
                               confidence=confidence, decision_id=f"d:{slot}{kind}")


def team(kinds, classes=None, *, focal_slot=None, w_star="wrong_a", root_states=None,
         root_classes=None, prev_h_lo=0.0, prev_h_hi=0.0, prev_q_lo=None):
    classes = classes or [None] * len(kinds)
    states = [state(i, k, c) for i, (k, c) in enumerate(zip(kinds, classes))]
    return outcomes.team_round(
        states, len(states), focal_slot, w_star,
        root_classes or {i: s.cls for i, s in enumerate(states)},
        root_states=root_states or {i: s.state for i, s in enumerate(states)},
        prev_h_lo=prev_h_lo, prev_h_hi=prev_h_hi, prev_q_lo=prev_q_lo)


# --------------------------------------------------------------------------- spec 18.3


def test_c_w_u_counts_sum_to_n():
    row = team([C, W, W, U, W])
    assert (row["c_count"], row["w_count"], row["u_count"]) == (1, 3, 1)
    assert row["c_count"] + row["w_count"] + row["u_count"] == row["n_agents"] == 5
    assert row["fully_graded"] is False
    with pytest.raises(ContractViolation):
        # a missing slot is an incomplete round, never a smaller team
        outcomes.team_round([state(0, C), state(1, W)], 5, None, None, {})


def test_three_state_transition_identity():
    before = [state(0, C), state(1, W), state(2, U), state(3, W), state(4, C)]
    after = [state(0, W), state(1, C), state(2, C), state(3, W), state(4, U)]
    prev = outcomes.team_round(before, 5, None, None, {})
    cur = outcomes.team_round(after, 5, None, None, {})
    rows = outcomes.transitions(before, after, {}, {})
    counts = outcomes.transition_counts(rows)
    assert counts["CW"] == 1 and counts["WC"] == 1 and counts["UC"] == 1 and counts["CU"] == 1
    assert outcomes.assert_three_state_identity(prev, cur, rows) == 0
    # a dropped transition row breaks the identity and is fatal, not a warning
    with pytest.raises(ContractViolation, match="three-state identity violated"):
        outcomes.assert_three_state_identity(prev, cur, [r for r in rows if r["kind"] != "WC"])


def test_validity_change_is_not_answer_switch():
    before = [state(0, C, "right"), state(1, C, "right")]
    after = [state(0, U), state(1, W, "wrong_b")]
    rows = {r["agent_slot"]: r for r in outcomes.transitions(before, after, {}, {})}
    # C -> U: the agent stopped delivering an answer.  That is a validity change and NOT a switch.
    assert rows[0]["validity_change"] is True
    assert rows[0]["switched_class"] is False
    assert rows[0]["kind"] == "CU"
    # C -> W with a different class: the agent switched its answer, and stayed valid.
    assert rows[1]["switched_class"] is True
    assert rows[1]["validity_change"] is False
    assert rows[1]["kind"] == "CW"


def test_wrong_unanimity_requires_one_shared_wrong_class():
    shared = team([W] * 5, ["wrong_a"] * 5)
    assert shared["wrong_unanimity"] is True

    scattered = team([W] * 5, [f"wrong_{i}" for i in range(5)])
    assert scattered["wrong_unanimity"] is False, "five different wrong answers are not a unanimity"
    assert scattered["all_singleton"] is True

    unresolved = team([U] * 5)
    assert unresolved["wrong_unanimity"] is False, "five unresolved slots are not a wrong unanimity"

    correct = team([C] * 5, ["right"] * 5)
    assert correct["wrong_unanimity"] is False and correct["unanimous_correct"] is True


def test_historical_coverage_is_nondecreasing():
    h_lo = h_hi = 0.0
    seen = []
    for kinds in ([W] * 5, [C] + [W] * 4, [W] * 5, [W] * 5):
        row = team(kinds, prev_h_lo=h_lo, prev_h_hi=h_hi)
        seen.append((row["q_lo"], row["h_lo"]))
        assert row["h_lo"] >= h_lo and row["h_hi"] >= h_hi
        h_lo, h_hi = row["h_lo"], row["h_hi"]
    assert [q for q, _ in seen] == [0.0, 1.0, 0.0, 0.0]
    assert [h for _, h in seen] == [0.0, 1.0, 1.0, 1.0], "history cannot be lost once it exists"


def test_active_coverage_is_bounded_by_historical_coverage():
    h_lo = h_hi = 0.0
    for kinds in ([C] + [W] * 4, [W] * 5, [U] + [W] * 4, [C, U, W, W, W]):
        row = team(kinds, prev_h_lo=h_lo, prev_h_hi=h_hi)
        assert row["q_lo"] <= row["h_lo"]
        assert row["q_hi"] <= row["h_hi"]
        assert row["q_lo"] <= row["q_hi"] and row["h_lo"] <= row["h_hi"]
        h_lo, h_hi = row["h_lo"], row["h_hi"]


def test_failure_decomposition_on_fully_graded_fixture():
    rows = []
    for kinds, selected in (([C, W, W, W, W], 1.0),      # had it, selected it
                            ([C, W, W, W, W], 0.0),      # had it, missed it
                            ([W] * 5, 0.0),              # lost it (history says it existed)
                            ([W] * 5, 0.0)):             # never had it
        row = team(kinds, prev_h_lo=1.0 if kinds == [W] * 5 else 0.0,
                   prev_h_hi=1.0 if kinds == [W] * 5 else 0.0)
        rows.append(dict(row, a_lo=selected))
    rows[-1]["h_lo"] = 0.0                                # this team never held a correct answer
    result = outcomes.failure_decomposition(rows)
    assert result["n"] == 4 and result["n_excluded"] == 0
    assert result["residual"] == pytest.approx(0.0, abs=1e-12)
    assert (1 - result["a"]) == pytest.approx(
        result["no_history"] + result["lost"] + result["selection"], abs=1e-12)
    assert result["no_history"] == pytest.approx(0.25)


def test_failure_decomposition_excludes_teams_that_are_not_fully_graded():
    graded = dict(team([C, W, W, W, W]), a_lo=1.0)
    ungraded = dict(team([C, U, W, W, W]), a_lo=0.0)
    result = outcomes.failure_decomposition([graded, ungraded])
    assert result["n"] == 1 and result["n_excluded"] == 1


def test_lower_upper_coverage_bounds_include_all_feasible_labels():
    """``[q_lo, q_hi]`` must contain the value of EVERY labelling of the unresolved slots."""
    import itertools

    kinds = [C, U, W, U, W]
    row = team(kinds)
    unresolved = [i for i, k in enumerate(kinds) if k == U]
    feasible_q, feasible_m = [], []
    for assignment in itertools.product([C, W], repeat=len(unresolved)):
        filled = list(kinds)
        for slot, value in zip(unresolved, assignment):
            filled[slot] = value
        complete = team(filled)
        feasible_q.append(complete["q_lo"])
        feasible_m.append(complete["m_lo"])
    assert row["q_lo"] <= min(feasible_q) and max(feasible_q) <= row["q_hi"]
    assert row["m_lo"] <= min(feasible_m) and max(feasible_m) <= row["m_hi"]


# --------------------------------------------------------------------------- the rest


def test_an_invalid_delivered_answer_can_never_be_graded_correct():
    with pytest.raises(ContractViolation, match="cannot be graded C"):
        outcomes.AgentState(slot=0, valid=False, cls=None, state=C, u_reason=None, confidence=None)


def test_an_ungraded_decision_is_unresolved_never_wrong():
    row = {"decision_id": "d:1", "validity": True, "canonical_class": "right",
           "confidence": 0.6, "agent_slot": 2, "generation_uid": "g:1", "source_item_id": "i1"}
    assert outcomes.agent_state(row).state == U
    assert outcomes.agent_state(row).u_reason == "not_graded"
    graded = outcomes.agent_state(row, {"semantic_correct": "C", "u_reason": None})
    assert graded.state == C and graded.confidence == pytest.approx(0.6)


def test_a_boolean_grade_column_is_refused():
    """``grading.tri_state`` is the only reader, and it refuses a boolean correctness column."""
    row = {"decision_id": "d:1", "validity": True, "canonical_class": "right",
           "confidence": None, "agent_slot": 0, "generation_uid": "g:1", "source_item_id": "i1"}
    with pytest.raises(NextdynError, match="(?i)(boolean|label-ish)"):
        outcomes.agent_state(row, {"semantic_correct": "C", "is_correct": True})


def test_plurality_status_separates_ties_from_unique_pluralities():
    assert team([C, C, C, W, W], ["right"] * 3 + ["wrong_a"] * 2)["plurality_status"] == "strict_majority"
    assert team([C, C, W, W, W], ["right"] * 2 + ["wrong_a"] * 3)["plurality_status"] == "strict_majority"
    tied = team([C, C, W, W, U], ["right", "right", "wrong_a", "wrong_a", None])
    assert tied["plurality_status"] == "tied" and tied["plurality_class"] is None
    unique = team([C, C, W, W, W], ["right", "right", "wrong_a", "wrong_b", "wrong_c"])
    assert unique["plurality_status"] == "unique_plurality" and unique["plurality_class"] == "right"
    nothing = team([U] * 5)
    assert nothing["plurality_status"] == "none" and nothing["plurality_correct"] is None


def test_an_unresolved_plurality_winner_is_unknown_not_zero():
    row = team([U, U, W, W, C], [None, None, "wrong_a", "wrong_b", "right"])
    assert row["plurality_correct"] is not None      # every winner resolved (all size 1)
    unresolved_winner = outcomes.team_round(
        [state(0, U, "mystery", valid=True), state(1, U, "mystery", valid=True),
         state(2, W, "wrong_a"), state(3, C, "right"), state(4, C, "right")],
        5, None, None, {})
    assert unresolved_winner["plurality_correct"] is None


def test_concentration_and_minority_columns_describe_the_class_histogram():
    row = team([C, W, W, W, W], ["right", "wrong_a", "wrong_a", "wrong_a", "wrong_b"])
    assert row["max_class_share"] == pytest.approx(3 / 5)
    assert row["wrong_class_max_share"] == pytest.approx(3 / 4)
    assert row["wrong_collision_prob"] == pytest.approx((3 / 4) ** 2 + (1 / 4) ** 2)
    assert row["arithmetic_correct_minority"] is True
    assert row["outnumbered_correct"] is True
    assert row["all_singleton"] is False
    assert 0.0 < row["class_entropy_norm"] < 1.0


def test_focal_columns_are_measured_against_the_root_not_the_previous_round():
    root_states = {0: C, 1: W, 2: W, 3: W, 4: W}
    root_classes = {0: "right", 1: "wrong_a", 2: "wrong_a", 3: "wrong_a", 4: "wrong_a"}
    states = [state(0, W, "wrong_a"), state(1, C, "right"), state(2, W, "wrong_a"),
              state(3, U), state(4, W, "wrong_a")]
    row = outcomes.team_round(states, 5, 0, "wrong_a", root_classes, root_states=root_states)
    assert row["focal_state"] == W and row["focal_retained_root_class"] is False
    assert row["n_nonfocal_init_wrong"] == 4
    assert row["nonfocal_init_wrong_now_c"] == pytest.approx(1 / 4)
    assert row["n_nonfocal_init_correct"] == 0
    assert row["nonfocal_init_correct_now_wstar"] is None
    assert row["nonfocal_unknown"] == 1


def test_adoption_of_the_shared_wrong_class_counts_only_observed_adoption():
    root_states = {0: C, 1: C, 2: C, 3: C, 4: W}
    root_classes = {i: "right" for i in range(4)} | {4: "wrong_a"}
    states = [state(0, W, "wrong_a"), state(1, U), state(2, C, "right"), state(3, C, "right"),
              state(4, W, "wrong_a")]
    row = outcomes.team_round(states, 5, 4, "wrong_a", root_classes, root_states=root_states)
    # one of the four initially correct non-focal slots now holds w*; the unknown one is NOT
    # counted as an adoption, it is counted as unknown
    assert row["nonfocal_init_correct_now_wstar"] == pytest.approx(1 / 4)
    assert row["nonfocal_unknown"] == 1


def test_copying_is_reported_beside_its_placebo():
    before = [state(0, W, "wrong_a"), state(1, C, "right")]
    after = [state(0, C, "right"), state(1, C, "right")]
    shared = " ".join(f"tok{i}" for i in range(20))
    rows = {r["agent_slot"]: r for r in outcomes.transitions(
        before, after, {0: ["right"]}, {1: shared}, w_star="wrong_a",
        cur_rationales={0: shared, 1: "unrelated text entirely different words here now"},
        received_slots_by_slot={0: [1], 1: [0]},
        placebo_rationales=["completely other words that nobody sent to this agent at all"])}
    assert rows[0]["copied_exact"] is True
    assert rows[0]["ngram_overlap_max"] == pytest.approx(1.0)
    assert rows[0]["placebo_overlap_max"] == pytest.approx(0.0)
    assert rows[0]["adopted_visible_class"] is True
    assert rows[1]["copied_exact"] is False


def test_alpha_beta_exclude_unresolved_pairs_and_report_the_exclusion():
    before = [state(0, C), state(1, W), state(2, U), state(3, W)]
    after = [state(0, C), state(1, C), state(2, C), state(3, W)]
    rows = outcomes.transitions(before, after, {}, {})
    result = outcomes.alpha_beta(rows)
    assert result["n_excluded_unresolved"] == 1
    assert result["alpha"] == pytest.approx(1 / 2) and result["beta"] == pytest.approx(0.0)


def test_the_state_alphabet_is_held_to_the_config():
    class Block:
        semantic_correctness_states = ("C", "W", "U")

    outcomes.assert_state_alphabet(Block())

    class Wrong:
        semantic_correctness_states = ("C", "W")

    with pytest.raises(ContractViolation):
        outcomes.assert_state_alphabet(Wrong())
