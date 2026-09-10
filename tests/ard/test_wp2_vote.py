"""ARD36 WP2: plurality selection over canonical classes (spec §7, §5.3, §3).

"Preserve duplicate votes … Resolve equal counts using a fixed seed-based order
independent of answer labels and protected correctness" (§5.3); "Invalid answers cannot
form a winning content class; if all outputs are invalid, selection fails" (§7); "K counts
the planned opportunities, including failed ones" (§4.4).
"""

from __future__ import annotations

import random

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard.vote import (
    ArdSelection,
    VoteCandidate,
    class_counts,
    confidence_weighted_plurality,
    fresh_prefix_votes,
    plurality,
    plurality_expected_correctness,
    prefix_vote,
    selection_failed,
    vote_tie_seed,
)
from agents_scaling.study.selection.vote import vote_tie_seed as study_vote_tie_seed

SEED = bytes.fromhex("be" * 32)


def pool(*specs: tuple[str, str | None]) -> list[VoteCandidate]:
    return [
        VoteCandidate(cid, cls, valid=cls is not None or cid.endswith("!"), confidence=None)
        for cid, cls in specs
    ]


def test_tie_seed_is_the_frozen_study_derivation() -> None:
    """Contract §3: ARD36 reuses the VOTE_TIE derivation, it does not re-invent one."""
    assert vote_tie_seed is study_vote_tie_seed
    assert vote_tie_seed(SEED) == study_vote_tie_seed(SEED)


def test_multiplicity_is_preserved() -> None:
    selection = plurality(
        pool(("a1", "A"), ("a2", "A"), ("a3", "A"), ("b1", "B"), ("b2", "B")),
        source_id="item-1",
        study_seed=SEED,
    )
    assert selection.selected_class == "A"
    assert selection.winning_count == 3 and selection.tied_classes == 1
    assert selection.class_counts == {"A": 3, "B": 2}
    assert selection.valid_count == 5 and selection.planned_count == 5


def test_invalid_candidates_stay_in_the_denominator() -> None:
    selection = plurality(
        pool(("a1", "A"), ("a2", "A"), ("x1", None), ("x2", None), ("x3", None)),
        source_id="item-1",
        study_seed=SEED,
    )
    assert selection.planned_count == 5      # K counts planned opportunities (§4.4)
    assert selection.valid_count == 2
    assert selection.unkeyed_count == 3
    assert selection.selected_class == "A"


def test_all_invalid_fails_the_selection() -> None:
    rows = pool(("x1", None), ("x2", None), ("x3", None), ("x4", None), ("x5", None))
    selection = plurality(rows, source_id="item-1", study_seed=SEED)
    assert selection.selection_failed is True
    assert selection.selected_candidate_id is None and selection.selected_class is None
    assert selection.planned_count == 5 and selection.valid_count == 0
    assert selection_failed(rows) is True


def test_tie_break_is_blind_to_input_order() -> None:
    rows = pool(("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B"), ("x", None))
    baseline = plurality(rows, source_id="item-7", study_seed=SEED)
    assert baseline.tied_classes == 2 and set(baseline.tied_winning_classes) == {"A", "B"}
    rng = random.Random(0)
    for _ in range(12):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        again = plurality(shuffled, source_id="item-7", study_seed=SEED)
        assert again.selected_candidate_id == baseline.selected_candidate_id
        assert again.selected_class == baseline.selected_class


def test_tie_break_is_not_first_in_list_and_is_uniform_over_classes() -> None:
    """A frozen *uniform* class-level draw: both tied classes win for some items."""
    winners = {
        plurality(
            pool(("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B")),
            source_id=f"item-{i}",
            study_seed=SEED,
        ).selected_class
        for i in range(40)
    }
    assert winners == {"A", "B"}


def test_tie_break_ignores_the_answer_labels() -> None:
    """Relabelling the *classes* (not the candidate ids) must not move the winner slot."""
    rows = pool(("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B"))
    original = plurality(rows, source_id="item-3", study_seed=SEED)
    relabelled = [
        VoteCandidate(c.candidate_id, {"A": "ZZZ", "B": "AAA"}[str(c.canonical_class)], True, None)
        for c in rows
    ]
    swapped = plurality(relabelled, source_id="item-3", study_seed=SEED)
    assert swapped.selected_candidate_id == original.selected_candidate_id


def test_duplicate_candidate_ids_are_refused() -> None:
    from agents_scaling.study.types import ProtocolError

    with pytest.raises(ProtocolError):
        plurality(pool(("a1", "A"), ("a1", "B")), source_id="item-1", study_seed=SEED)


# --------------------------------------------------------------------------- prefixes


def test_prefix_vote_follows_draw_order() -> None:
    bank = pool(*[(f"d{i}", "A" if i % 2 == 0 else "B") for i in range(20)])
    votes = fresh_prefix_votes(bank, source_id="item-1", study_seed=SEED)
    assert sorted(votes) == sorted(K.FRESH_PREFIX_K)
    for k, selection in votes.items():
        assert selection.prefix_k == k
        assert selection.planned_count == k
        assert selection.pool_candidate_ids == tuple(f"d{i}" for i in range(k))
    with pytest.raises(ValueError):
        prefix_vote(bank, 21, source_id="item-1", study_seed=SEED)


def test_prefix_votes_skip_ks_the_bank_does_not_have() -> None:
    bank = pool(*[(f"d{i}", "A") for i in range(10)])
    assert sorted(fresh_prefix_votes(bank, source_id="i", study_seed=SEED)) == [5, 10]


# --------------------------------------------------------------------------- weighted


def test_confidence_weighted_matches_plurality_when_weights_are_flat() -> None:
    rows = [
        VoteCandidate("a1", "A", True, 1.0),
        VoteCandidate("a2", "A", True, 1.0),
        VoteCandidate("b1", "B", True, 1.0),
        VoteCandidate("b2", "B", True, 1.0),
    ]
    flat = confidence_weighted_plurality(rows, source_id="item-9", study_seed=SEED)
    counted = plurality(rows, source_id="item-9", study_seed=SEED)
    assert flat.selected_candidate_id == counted.selected_candidate_id
    assert flat.selected_class == counted.selected_class


def test_confidence_weighted_can_overturn_a_count_plurality() -> None:
    rows = [
        VoteCandidate("a1", "A", True, 0.1),
        VoteCandidate("a2", "A", True, 0.1),
        VoteCandidate("a3", "A", True, 0.1),
        VoteCandidate("b1", "B", True, 0.99),
        VoteCandidate("b2", "B", True, 0.99),
    ]
    assert plurality(rows, source_id="i", study_seed=SEED).selected_class == "A"
    assert confidence_weighted_plurality(rows, source_id="i", study_seed=SEED).selected_class == "B"


def test_missing_confidence_is_never_imputed() -> None:
    rows = [
        VoteCandidate("a1", "A", True, None),
        VoteCandidate("a2", "A", True, None),
        VoteCandidate("b1", "B", True, 0.4),
    ]
    weighted = confidence_weighted_plurality(rows, source_id="i", study_seed=SEED)
    assert weighted.missing_confidence_count == 2
    assert weighted.selected_class == "B"     # 0.4 beats 0 + 0, never 0.5 + 0.5
    assert weighted.degenerate_weights is False


def test_degenerate_weights_fall_back_to_plurality() -> None:
    rows = [
        VoteCandidate("a1", "A", True, None),
        VoteCandidate("a2", "A", True, None),
        VoteCandidate("b1", "B", True, None),
    ]
    weighted = confidence_weighted_plurality(rows, source_id="i", study_seed=SEED)
    counted = plurality(rows, source_id="i", study_seed=SEED)
    assert weighted.degenerate_weights is True
    assert weighted.selected_candidate_id == counted.selected_candidate_id
    assert isinstance(weighted, ArdSelection) and weighted.selector == "confidence_weighted"


# --------------------------------------------------------------------------- analytic


def test_expected_correctness_under_tied_winners() -> None:
    rows = pool(("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B"))
    selection = plurality(rows, source_id="item-7", study_seed=SEED)
    assert selection.tied_classes == 2
    assert plurality_expected_correctness(selection, {"A": True, "B": False}) == 0.5
    assert plurality_expected_correctness(selection, {"A": True, "B": True}) == 1.0
    assert plurality_expected_correctness(selection, {"A": False, "B": False}) == 0.0
    # class_grade_inconsistent (plan §10): excluded, never guessed
    assert plurality_expected_correctness(selection, {"A": True}) is None


def test_expected_correctness_of_a_failed_selection_is_zero() -> None:
    rows = pool(("x1", None), ("x2", None))
    selection = plurality(rows, source_id="item-1", study_seed=SEED)
    assert plurality_expected_correctness(selection, {}) == 0.0


def test_candidate_from_report_never_classifies_an_invalid_report() -> None:
    from agents_scaling.ard.vote import candidate_from_report
    from tests.ard import wp2_support as W

    good = candidate_from_report("c1", W.report("42", confidence=0.6), W.upper_class)
    assert good.canonical_class == "42" and good.valid and good.confidence == 0.6
    bad = candidate_from_report("c2", W.report(None), W.upper_class)
    assert bad.canonical_class is None and not bad.valid and not bad.keyed
    # a valid report whose answer has no class is keyless but still a planned opportunity
    unkeyable = candidate_from_report("c3", W.report("42"), lambda a: None)
    assert unkeyable.valid and unkeyable.canonical_class is None and not unkeyable.keyed


def test_class_counts_ignore_invalid_rows() -> None:
    assert class_counts(pool(("a", "A"), ("b", "A"), ("x", None))) == {"A": 2}
