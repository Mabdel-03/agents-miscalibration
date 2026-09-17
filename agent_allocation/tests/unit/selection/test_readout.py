"""selection/readout.py: plurality, math classes, code public selection, single-candidate readouts, API guards."""
from __future__ import annotations

import functools
import inspect
import itertools
import typing
from dataclasses import replace

import pytest

from agent_allocation.contracts import (
    CheckpointView, CheckStatus, EvalKind, OutputKind, ParseStatus, PrivateEvalSpec, SelectionStatus,
)
from agent_allocation.selection import readout
from agent_allocation.selection.plurality import class_tie_key
from agent_allocation.selection.readout import select_checkpoint, select_checkpoint_detail
from tests.unit.selection import factories as f

SEED = 20260917


def _select(t, cands, family="sample", name=None, checks=None, view=CheckpointView.CURRENT, ids=None, **kw):
    ids = tuple(ids if ids is not None else cands)
    cp = f.checkpoint(t, name or f"{family}/{len(ids)}", family, ids, view=view)
    return select_checkpoint_detail(cp, t, cands, checks or {}, seed=kw.pop("seed", SEED), **kw)


# ----------------------------------------------------------------------------------------- happy paths


def test_option_plurality_selects_the_largest_class_first_member():
    t = f.task(OutputKind.OPTION)
    cands = f.pool(t, ["B", "(C)", "C", " C\n", "B", "Answer: B", "b", "**B**", "B."])
    detail = _select(t, cands)
    assert detail.result.status is SelectionStatus.SELECTED
    assert detail.result.selected_candidate_id == "c1"
    assert detail.result.selector_name == "plurality_option"
    assert detail.winning_support == 3 and detail.n_votes == 5          # decorated letters are not option identities
    assert detail.result.public_evidence_hash is None
    assert detail.result.policy_id == "main/item-1/sample/9/current"


def test_illegal_or_unreadable_options_cast_no_vote():
    t = f.task(OutputKind.OPTION, choices=("A", "B", "C", "D"))
    cands = f.pool(t, ["E", "E", "banana", "D"])
    detail = _select(t, cands)
    assert detail.result.selected_candidate_id == "c3" and detail.n_votes == 1


def test_parenthesized_allowed_choices_are_accepted():
    t = f.task(OutputKind.OPTION, choices=("(A)", "(B)"))
    detail = _select(t, f.pool(t, ["(B)", "B", "A"]))
    assert detail.result.selected_candidate_id == "c0"


def test_boolean_plurality():
    t = f.task(OutputKind.BOOLEAN)
    detail = _select(t, f.pool(t, ["true", "False", "**FALSE.**", "maybe", "True or False"]))
    assert detail.result.selected_candidate_id == "c1" and detail.result.selector_name == "plurality_boolean"
    assert detail.n_votes == 3


def test_math_exact_normalization_groups_before_equivalence():
    t = f.task(OutputKind.MATH)
    cands = f.pool(t, ["\\boxed{\\dfrac{1}{2}}", "$\\frac {1}{2}$", "0.5", "\\left(1,2\\right)"])
    detail = _select(t, cands)                       # no equivalence callable: exact classes only
    assert detail.result.selector_name == "plurality_math_exact"
    assert detail.result.selected_candidate_id == "c0" and detail.winning_support == 2


def test_all_invalid_pool_abstains():
    t = f.task(OutputKind.MATH)
    cands = {"c0": f.candidate("c0", None, t=t), "c1": f.candidate("c1", None, t=t, status=ParseStatus.MALFORMED)}
    detail = _select(t, cands)
    assert detail.result.status == SelectionStatus.ABSTAINED_ALL_INVALID.value
    assert detail.result.selected_candidate_id is None and detail.n_votes == 0


@pytest.mark.parametrize("family", ["private_single", "deep", "pooled", "sample", "peer"])
def test_single_candidate_pools_deliver_the_candidate(family):
    t = f.task(OutputKind.MATH)
    cands = f.pool(t, ["7"])
    detail = _select(t, cands, family=family, name=f"{family}/1")
    assert detail.result.status == SelectionStatus.SINGLE_CANDIDATE.value
    assert detail.result.selected_candidate_id == "c0" and detail.result.selector_name == "last_answer"


def test_invalid_single_candidate_is_still_the_delivered_output():
    t = f.task(OutputKind.MATH)
    cands = {"c0": f.candidate("c0", None, t=t)}
    detail = _select(t, cands, family="deep", name="deep/2")
    assert detail.result.selected_candidate_id == "c0" and detail.n_votes == 0


def test_last_answer_family_with_a_multi_candidate_current_pool_raises():
    t = f.task(OutputKind.MATH)
    with pytest.raises(ValueError):
        _select(t, f.pool(t, ["1", "2"]), family="private_single", name="private_single/2")


def test_archive_view_is_labelled_and_votes_over_occurrences():
    t = f.task(OutputKind.MATH)
    cands = f.pool(t, ["1", "2", "2", "1", "1"])
    detail = _select(t, cands, family="private_single", name="private_single/5", view=CheckpointView.ARCHIVE)
    assert detail.result.selector_name == "archive_plurality_math_exact"
    assert detail.result.selected_candidate_id == "c0" and detail.result.policy_id.endswith("/archive")


def test_pool_hash_covers_ordered_ids_and_final_answers():
    t = f.task(OutputKind.MATH)
    cands = f.pool(t, ["1", "2"])
    a = _select(t, cands).result.pool_hash
    b = _select(t, cands, ids=("c1", "c0")).result.pool_hash
    changed = dict(cands, c1=f.candidate("c1", "3", t=t))
    c = _select(t, changed).result.pool_hash
    assert len({a, b, c}) == 3


# ----------------------------------------------------------------------------------------- math classes


def _fake_equivalence(pairs):
    """Symmetric closure of ``pairs`` is True; ("?", x) is unknown; everything else False."""
    def eq(a, b):
        if "?" in (a, b):
            return None
        return a == b or (a, b) in pairs or (b, a) in pairs
    return eq


def test_math_mutual_equivalence_merges_classes():
    t = f.task(OutputKind.MATH)
    cands = f.pool(t, ["2", "1/2", "\\frac{1}{2}", "0.5", "2"])
    eq = _fake_equivalence({("1/2", "\\frac{1}{2}"), ("1/2", "0.5"), ("\\frac{1}{2}", "0.5")})
    detail = _select(t, cands, math_equivalent=eq)
    assert detail.result.selector_name == "plurality_math_mutual_complete_link"
    assert detail.result.selected_candidate_id == "c1" and detail.winning_support == 3
    assert detail.exact_only_selected_candidate_id == "c0"
    winner = next(c for c in detail.classes if "c1" in c.member_ids)
    assert winner.class_id == "math:0.5" and winner.member_ids == ("c1", "c2", "c3")


def test_one_direction_equivalence_never_merges():
    t = f.task(OutputKind.MATH)
    cands = f.pool(t, ["a", "b", "b", "a", "a"])
    detail = _select(t, cands, math_equivalent=lambda x, y: (x, y) == ("b", "a") or x == y)
    assert {c.member_ids for c in detail.classes} == {("c0", "c3", "c4"), ("c1", "c2")}


def test_unknown_equivalence_never_merges():
    t = f.task(OutputKind.MATH)
    cands = f.pool(t, ["?", "x", "x"])
    detail = _select(t, cands, math_equivalent=_fake_equivalence(set()))
    assert len(detail.classes) == 2 and detail.equivalence_unknown >= 1


def test_complete_link_blocks_nontransitive_chains():
    t = f.task(OutputKind.MATH)
    cands = f.pool(t, ["a", "b", "c"])        # a~b, b~c, but not a~c
    detail = _select(t, cands, math_equivalent=_fake_equivalence({("a", "b"), ("b", "c")}))
    assert {c.member_ids for c in detail.classes} == {("c0", "c1"), ("c2",)}


def test_exact_keys_never_merge_what_equivalence_keeps_apart():
    t = f.task(OutputKind.MATH)
    never = lambda a, b: a == b or None                                    # noqa: E731 - unknown for any pair
    detail = _select(t, f.pool(t, ["1 2", "12", "7"]), math_equivalent=never)
    assert {c.member_ids for c in detail.classes} == {("c0",), ("c1",), ("c2",)}
    detail = _select(t, f.pool(t, ["x²", "x2", "x^2"]), math_equivalent=never)
    assert len(detail.classes) == 3
    twelve = _fake_equivalence({("12", "12.0")})
    detail = _select(t, f.pool(t, ["12", "1 2", "12.0"]), math_equivalent=twelve)
    assert {c.member_ids for c in detail.classes} == {("c0", "c2"), ("c1",)}


def test_complete_link_checks_every_distinct_text_of_a_class():
    t = f.task(OutputKind.MATH)
    # "\\frac{1}{2}" and "\\dfrac{1}{2}" share an exact key; "0.5" verifies against only one of the two texts
    eq = lambda a, b: a == b or ({a, b} == {"0.5", "\\frac{1}{2}"}) or None  # noqa: E731
    detail = _select(t, f.pool(t, ["\\frac{1}{2}", "\\dfrac{1}{2}", "0.5"]), math_equivalent=eq)
    assert {c.member_ids for c in detail.classes} == {("c0", "c1"), ("c2",)}


def test_equivalence_callable_must_return_tristate():
    t = f.task(OutputKind.MATH)
    with pytest.raises(TypeError):
        _select(t, f.pool(t, ["1", "2"]), math_equivalent=lambda a, b: 1)


# ----------------------------------------------------------------------------------------- ties


def test_class_ties_use_the_frozen_key_not_order_or_ids():
    t = f.task(OutputKind.OPTION)
    finals = ["A", "B", "C", "D"]
    expected_class = min(("option:" + x for x in finals), key=lambda c: (class_tie_key(SEED, t.item_id, c), c))
    winners = set()
    for perm in itertools.permutations(finals):
        cands = {f"id-{x}-{i}": f.candidate(f"id-{x}-{i}", x, t=t) for i, x in enumerate(perm)}
        detail = _select(t, cands)
        winners.add(cands[detail.result.selected_candidate_id].final_answer)
        assert detail.tied[0] == expected_class
    assert winners == {expected_class.split(":")[1]}


def test_tie_winner_depends_on_seed_and_item_only():
    t = f.task(OutputKind.BOOLEAN)
    seen = set()
    for seed in range(40):
        detail = _select(t, f.pool(t, ["True", "False"]), seed=seed)
        seen.add(detail.result.selected_candidate_id)
    assert seen == {"c0", "c1"}                      # the key is pseudo-random across seeds


# ----------------------------------------------------------------------------------------- code


def test_code_selects_max_public_fraction():
    t = f.task(OutputKind.PYTHON)
    cands = f.pool(t, ["def f(): return 1", "def f(): return 2", "def f(): return 3"])
    checks = {"c0": f.check("c0", t, 1), "c1": f.check("c1", t, 2), "c2": f.check("c2", t, 0)}
    detail = _select(t, cands, checks=checks)
    assert detail.result.selected_candidate_id == "c1"
    assert detail.result.selector_name == "public_example_selection"
    assert detail.result.public_evidence_hash and detail.best_public_fraction == "1/1"


def test_code_ties_follow_frozen_priority_and_ignore_mapping_order():
    t = f.task(OutputKind.PYTHON)
    cands = f.pool(t, ["p0", "p1", "p2"])
    checks = {cid: f.check(cid, t, 1) for cid in cands}
    first = _select(t, cands, checks=checks).result.selected_candidate_id
    reversed_checks = dict(reversed(list(checks.items())))
    reversed_cands = dict(reversed(list(cands.items())))
    assert _select(t, reversed_cands, checks=reversed_checks, ids=("c0", "c1", "c2")).result.selected_candidate_id \
        == first


def test_code_zero_public_cases_pause_rather_than_score_zero():
    t = f.task(OutputKind.PYTHON)
    cands = f.pool(t, ["p0", "p1"])
    checks = {"c0": f.check("c0", t, 0, total=0), "c1": f.check("c1", t, 0, total=0)}
    detail = _select(t, cands, checks=checks)
    assert detail.result.status is SelectionStatus.PAUSED_CHECK_ERROR and detail.result.selected_candidate_id is None


def test_declared_selector_family_must_admit_the_concrete_selector():
    t = f.task(OutputKind.MATH)
    cands = f.pool(t, ["1", "2"])
    cp = replace(f.checkpoint(t, "sample/2", "sample", tuple(cands)), selector_name="public_tests_only",
                 selector_version="v1")
    with pytest.raises(ValueError):
        select_checkpoint(cp, t, cands, {}, seed=SEED)
    ok = select_checkpoint(replace(cp, selector_name="plurality"), t, cands, {}, seed=SEED)
    assert readout.concrete_selector_is_legal("plurality", ok.selector_name)
    assert readout.concrete_selector_is_legal("plurality", "archive_" + ok.selector_name)
    assert not readout.concrete_selector_is_legal("public_tests_only", ok.selector_name)
    assert readout.concrete_selector_is_legal("public_tests_only", "last_answer")


def test_code_infrastructure_error_pauses():
    t = f.task(OutputKind.PYTHON)
    cands = f.pool(t, ["p0", "p1"])
    checks = {"c0": f.check("c0", t, 2), "c1": f.check("c1", t, 0, status=CheckStatus.INFRASTRUCTURE_ERROR)}
    detail = _select(t, cands, checks=checks)
    assert detail.result.status == SelectionStatus.PAUSED_CHECK_ERROR.value
    assert detail.result.selected_candidate_id is None


def test_code_without_runnable_program_abstains():
    t = f.task(OutputKind.PYTHON)
    cands = {"c0": f.candidate("c0", None, t=t), "c1": f.candidate("c1", "def (", t=t)}
    checks = {"c1": f.check("c1", t, 0, status=CheckStatus.INVALID_CANDIDATE)}
    detail = _select(t, cands, checks=checks)
    assert detail.result.status == SelectionStatus.ABSTAINED_ALL_INVALID.value


def test_code_valid_candidate_without_public_check_raises():
    t = f.task(OutputKind.PYTHON)
    cands = f.pool(t, ["p0", "p1"])
    with pytest.raises(ValueError):
        _select(t, cands, checks={"c0": f.check("c0", t, 1)})


# ----------------------------------------------------------------------------------------- private inputs refused


def _private(t):
    return PrivateEvalSpec(t.item_id, t.benchmark, "rev", EvalKind.MATH_VERIFY, "1/2", None, "hash")


def test_selector_refuses_private_eval_specs_and_grades_anywhere():
    t = f.task(OutputKind.MATH)
    cands = f.pool(t, ["1", "2"])
    cp = f.checkpoint(t, "sample/2", "sample", tuple(cands))
    with pytest.raises(TypeError):
        select_checkpoint(cp, _private(t), cands, {}, seed=SEED)
    with pytest.raises(TypeError):
        select_checkpoint(cp, t, {**cands, "g": f.grade("c0", True)}, {}, seed=SEED)
    with pytest.raises(TypeError):
        select_checkpoint(cp, t, cands, {"c0": f.grade("c0", True)}, seed=SEED)
    with pytest.raises(TypeError):
        select_checkpoint(cp, t, cands, {}, seed=SEED, math_equivalent=functools.partial(
            lambda a, b, gold: a == b, gold=_private(t)))
    grade = f.grade("c0", True)
    with pytest.raises(TypeError):
        select_checkpoint(cp, t, cands, {}, seed=SEED, math_equivalent=lambda a, b: grade.semantic_correct)


def test_selector_signature_names_no_private_types():
    hints = typing.get_type_hints(readout.select_checkpoint)
    text = " ".join(str(v) for v in hints.values())
    assert "PrivateEvalSpec" not in text and "GradeResult" not in text
    assert list(inspect.signature(readout.select_checkpoint).parameters) == [
        "checkpoint", "task", "candidates", "public_checks", "seed", "math_equivalent"]


def test_selection_package_imports_only_the_gold_blind_normalizer_from_grading():
    import ast
    import pathlib
    root = pathlib.Path(readout.__file__).parent
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else \
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            for name in names:
                if "grading" in name:
                    assert name == "agent_allocation.grading.text.normalize", (path, name)


def test_results_ignore_grades_entirely():
    """Same public inputs, opposite gold: the selection rows are identical."""
    t = f.task(OutputKind.OPTION)
    cands = f.pool(t, ["A", "B", "B", "A"])
    cp = f.checkpoint(t, "sample/4", "sample", tuple(cands))
    one = select_checkpoint(cp, t, cands, {}, seed=SEED)
    two = select_checkpoint(cp, t, cands, {}, seed=SEED)
    assert (one.selected_candidate_id, one.pool_hash, one.status) == (two.selected_candidate_id, two.pool_hash,
                                                                       two.status)
