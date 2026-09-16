"""Parametric MC, the MATH wrapper and the family dispatch (spec 4.4, 4.6, 5.1, 6.5; plan §2.5).

The vendoring parity tests live here: ``mc_key`` must agree with the frozen
``study.selection.normalize.hle_mc_key`` at ten options (the alphabet ARD36 froze) and must
keep working at thirteen (legacy TruthfulQA MC1), and the canonical class must be the
permutation-invariant canonical option id rather than the displayed letter.
"""

from __future__ import annotations

import pytest

from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn import items_build as IB
from agents_scaling.nextdyn.errors import ConfigError, LeakageError
from agents_scaling.nextdyn.graders import STATUS_AMBIGUOUS, STATUS_CORRECT, STATUS_INCORRECT
from agents_scaling.nextdyn.graders import GradeTarget, Verdict
from agents_scaling.nextdyn.graders import families as GF
from agents_scaling.nextdyn.graders import math as GMATH
from agents_scaling.nextdyn.graders import mc as MC
from agents_scaling.study.selection.normalize import hle_mc_key

SEED = bytes.fromhex("00" * 32)

#: The accepted forms of the frozen MC key rule plus a few that must stay unkeyable.
CORPUS = ("B", "(B)", "B.", "B)", "B:", "**B**", "Answer: B", "The answer is (B)", "Option B",
          "b", "j", "J.", " (j) ", "$\\boxed{C}$", "K", "hello", "A B", "", "12", "**Answer: D**")


def target(n_options: int, permutation, options=None) -> GradeTarget:
    return GradeTarget(family=F.MMLU_PRO, source_item_id="item0001", item_variant_id="item0001.v0",
                       answer_format=F.ANSWER_FORMAT_MC, permutation=permutation,
                       options=options or tuple(f"option {i}" for i in range(n_options)))


def test_mc_key_matches_hle_mc_key_at_ten_options():
    """Vendoring parity: the parametric rule is the frozen rule at the frozen alphabet."""
    for text in CORPUS:
        assert MC.mc_key(text, 10) == hle_mc_key(text), text


def test_mc_letters_is_parametric_and_bounded():
    assert MC.mc_letters(4) == "ABCD"
    assert MC.mc_letters(13) == "ABCDEFGHIJKLM"
    assert MC.mc_key("M", 13) == "M" and MC.mc_key("M", 10) is None
    for bad in (0, -1, 27, True):
        with pytest.raises(ConfigError):
            MC.mc_letters(bad)


@pytest.mark.parametrize("n_options", [10, 13])
def test_option_shuffle_preserves_canonical_answer_identity(n_options):
    """Spec 18.3: the canonical class of an answer does not move when the options do."""
    item = "itemshuffle00001"
    canonical = [f"canonical option {index}" for index in range(n_options)]
    gold_index = 3
    classes = set()
    for variant in IB.PERMUTATION_VARIANTS:
        permutation = IB.option_permutation(item, SEED, n_options, variant)
        displayed = [canonical[index] for index in permutation]
        letter = MC.displayed_letter(gold_index, permutation)
        spot = target(n_options, permutation, tuple(displayed))
        assert MC.canonical_option_id(letter, permutation) == gold_index
        classes.add(MC.canonical_class(letter, spot))
        verdict = MC.grade_mc(letter, displayed, permutation, gold_index)
        assert verdict.status == STATUS_CORRECT and verdict.chosen_canonical == gold_index
        # the same *text* answer resolves to the same canonical class through the fallback
        assert MC.canonical_class(canonical[gold_index], spot) == MC.class_of(gold_index)
    assert classes == {MC.class_of(gold_index)}, "the canonical class moved with the permutation"


def test_a_wrong_option_is_incorrect_and_an_unkeyable_answer_is_ambiguous():
    permutation = (2, 0, 1, 3)
    displayed = ["alpha", "beta", "gamma", "delta"]
    assert MC.grade_mc("A", displayed, permutation, 2).status == STATUS_CORRECT
    assert MC.grade_mc("B", displayed, permutation, 2).status == STATUS_INCORRECT
    ambiguous = MC.grade_mc("I am not sure", displayed, permutation, 2)
    assert ambiguous.status == STATUS_AMBIGUOUS and ambiguous.chosen_letter is None
    assert MC.grade_mc("", displayed, permutation, 2).status == STATUS_INCORRECT


def test_an_answer_matching_several_option_texts_is_ambiguous_not_a_guess():
    permutation = (0, 1, 2, 3)
    displayed = ["same", "same", "other", "third"]
    verdict = MC.grade_mc("same", displayed, permutation, 0)
    assert verdict.status == STATUS_AMBIGUOUS


def test_canonical_class_is_gold_independent():
    """The canonicalizer never sees the reference: two calls with different gold agree."""
    permutation = (1, 0, 2)
    spot = target(3, permutation, ("x", "y", "z"))
    assert MC.canonical_class("A", spot) == MC.class_of(1)
    assert MC.class_index(MC.canonical_class("A", spot)) == 1


def test_grade_mc_refuses_gold_outside_the_item():
    with pytest.raises(ConfigError):
        MC.grade_mc("A", ["x", "y"], (0, 1), 7)


def test_the_math_wrapper_keeps_the_four_statuses_apart():
    spot = GradeTarget(family=F.MATH500, source_item_id="i", item_variant_id="i.v",
                       answer_format=F.ANSWER_FORMAT_EXACT)
    assert GMATH.grade("\\frac{1}{2}", {"answer": "0.5"}, spot).status == STATUS_CORRECT
    assert GMATH.grade("7", {"answer": "0.5"}, spot).status == STATUS_INCORRECT
    assert GMATH.grade("", {"answer": "0.5"}, spot).status == STATUS_INCORRECT
    assert GMATH.canonical_class("0.5", spot) == GMATH.canonical_class("1/2", spot)
    with pytest.raises(ConfigError):
        GMATH.grade("1", {}, spot)


def test_grader_hash_changes_with_the_rules_and_covers_every_family():
    hashes = GF.grader_hashes()
    assert set(hashes) == set(F.FAMILIES)
    assert len(set(hashes.values())) == len(hashes)
    assert all(len(value) == 64 for value in hashes.values())


def test_family_dispatch_grades_through_the_declared_grader():
    permutation = (0, 1, 2, 3)
    spot = target(4, permutation, ("a", "b", "c", "d"))
    verdict = GF.grade(F.MMLU_PRO, "C", {"canonical_index": 2}, spot)
    assert isinstance(verdict, Verdict) and verdict.status == STATUS_CORRECT
    assert GF.canonical_class(F.MMLU_PRO, "C", spot) == MC.class_of(2)


def test_bigcodebench_grading_is_refused_outside_the_e0_sandbox():
    spot = GradeTarget(family=F.BIGCODEBENCH, source_item_id="i", item_variant_id="i.v",
                       answer_format=F.ANSWER_FORMAT_EXACT)
    with pytest.raises(GF.GoldUnavailable):
        GF.grade(F.BIGCODEBENCH, "def f(): pass", {}, spot)
    assert GF.code_canonical_class("def f():\n    return 1\n", spot) is not None


def test_the_gold_store_is_0600_inside_0700_and_refuses_a_loosened_file(tmp_path):
    import os
    import stat

    records = [{"source_item_id": "item0001", "family": F.MATH500, "answer": "42"}]
    path = GF.write_gold(tmp_path, F.MATH500, records)
    assert stat.S_IMODE(path.stat().st_mode) == GF.PROTECTED_FILE_MODE
    assert stat.S_IMODE(path.parent.stat().st_mode) == GF.PROTECTED_DIR_MODE
    store = GF.GoldStore.load(tmp_path, [F.MATH500])
    assert store.reference(F.MATH500, "item0001")["answer"] == "42"
    assert len(store.ref_hash(F.MATH500, "item0001")) == 64
    with pytest.raises(GF.GoldUnavailable):
        store.reference(F.MATH500, "nope")
    os.chmod(path, 0o644)
    with pytest.raises(LeakageError):
        GF.GoldStore.load(tmp_path, [F.MATH500])
