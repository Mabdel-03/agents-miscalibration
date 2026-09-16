"""The tri-state writer and its single reader (spec 4.6, 15.3, 17.3; plan §2.5, D15)."""

from __future__ import annotations

import json

import pytest

from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn import grading as G
from agents_scaling.nextdyn import items as I
from agents_scaling.nextdyn import schema, tables
from agents_scaling.nextdyn.errors import LeakageError, MaterializeError
from agents_scaling.nextdyn.graders import GradeTarget
from agents_scaling.nextdyn.graders import families as GF
from tests.nextdyn.test_banks import ITEM, VARIANT, stub_item, tokens

MATH_GOLD = {"source_item_id": ITEM, "family": F.MATH500, "answer": "4", "solution": ""}


def target() -> GradeTarget:
    return GradeTarget(family=F.MATH500, source_item_id=ITEM, item_variant_id=VARIANT,
                       answer_format=F.ANSWER_FORMAT_EXACT)


def output(answer: str | None = "4", *, status: schema.ParseStatus = schema.ParseStatus.OK):
    text = json.dumps({"reasoning": "r", "shareable_claims": ["c"], "final_answer": answer,
                       "confidence": 0.5})
    parsed = schema.parse_public_output(text, "stop", (1, 2), lambda s: len(s.split()),
                                        limits=tokens())
    import dataclasses

    return dataclasses.replace(parsed, parse_status=status)


def decision(*, uid: str = "g:" + "0" * 64, delivery: bool = True, infra: str = "ok",
             parse: str = "ok", decision_id: str = "d:" + "0" * 40) -> tables.DecisionRow:
    row = {name: None for name in tables.schema("decisions").column_names}
    row.update({
        "decision_id": decision_id, "study_id": "s", "cell_kind": "donor_bank",
        "source_item_id": ITEM, "item_variant_id": VARIANT, "root_team_id": "",
        "protocol_hash": "", "receiver_model_hash": "m" * 16, "budget_hash": "b" * 64,
        "replicate_id": 0, "round": 0, "agent_slot": 0, "phase": "bank",
        "episode_id": "donor_bank|x", "cell_id": "donor_bank/c", "generation_uid": uid,
        "request_hashes_json": "[]", "aliased": False, "is_reference": False,
        "parent_decision_ids_json": "[]", "seen_transmission_ids_json": "[]",
        "parse_status": parse, "escape_repaired": False,
        "field_order_violation": False, "validity": True,
        "delivery_success": delivery, "infrastructure_status": infra,
        "termination_reason": "stop", "boundary_injected": False, "continuation_used": False,
        "discarded_span_tokens": 0, "thinking_tokens": 1, "final_tokens": 1, "prompt_tokens": 1,
        "cached_prefill_tokens": 0})
    return tables.row_from_dict(tables.DecisionRow, row)


# --------------------------------------------------------------------------- the tri-state


def test_a_correct_and_an_incorrect_answer_resolve_to_c_and_w():
    correct = G.grade_decision(F.MATH500, output("4"), MATH_GOLD, decision=decision(),
                               target=target())
    wrong = G.grade_decision(F.MATH500, output("5"), MATH_GOLD, decision=decision(),
                             target=target())
    assert (correct.semantic_correct, correct.grading_status) == (G.STATE_CORRECT, G.GRADING_RESOLVED)
    assert (wrong.semantic_correct, wrong.u_reason) == (G.STATE_WRONG, None)
    assert correct.gold_ref_hash == wrong.gold_ref_hash and len(correct.gold_ref_hash) == 64
    assert correct.grader_hash == GF.grader_hash(F.MATH500)
    assert json.loads(correct.decision_ids_json) == [decision().decision_id]


@pytest.mark.parametrize("kwargs,reason", [
    ({"infra": "endpoint_lost"}, G.U_INFRA_FAILED),
    ({"delivery": False}, G.U_NOT_DELIVERED),
])
def test_an_operational_failure_is_unknown_never_wrong(kwargs, reason):
    row = G.grade_decision(F.MATH500, output("4"), MATH_GOLD, decision=decision(**kwargs),
                           target=target())
    assert row.semantic_correct == G.STATE_UNKNOWN
    assert row.u_reason == reason and row.grading_status == G.GRADING_NOT_GRADED


@pytest.mark.parametrize("status,reason", [
    (schema.ParseStatus.NO_JSON, G.U_PARSE_FAILED),
    (schema.ParseStatus.INVALID_JSON, G.U_PARSE_FAILED),
    (schema.ParseStatus.CONTEXT_FAILURE, G.U_CONTEXT_REFUSED),
])
def test_an_unparseable_output_is_unknown_with_a_typed_reason(status, reason):
    row = G.grade_decision(F.MATH500, output("4", status=status), MATH_GOLD,
                           decision=decision(), target=target())
    assert (row.semantic_correct, row.u_reason) == (G.STATE_UNKNOWN, reason)


def test_an_empty_answer_is_unknown_not_wrong():
    row = G.grade_decision(F.MATH500, output(None), MATH_GOLD, decision=decision(),
                           target=target())
    assert (row.semantic_correct, row.u_reason) == (G.STATE_UNKNOWN, G.U_NO_ANSWER)


def test_an_ambiguous_grader_verdict_is_unknown():
    spot = GradeTarget(family=F.MMLU_PRO, source_item_id=ITEM, item_variant_id=VARIANT,
                       answer_format=F.ANSWER_FORMAT_MC, permutation=(0, 1, 2, 3),
                       options=("a", "b", "c", "d"))
    row = G.grade_decision(F.MMLU_PRO, output("maybe the third one"),
                           {"canonical_index": 2}, decision=decision(), target=spot)
    assert (row.semantic_correct, row.u_reason) == (G.STATE_UNKNOWN, G.U_GRADER_AMBIGUOUS)
    assert row.grading_status == G.GRADING_AMBIGUOUS


def test_a_grader_crash_is_unknown_and_never_a_wrong_answer(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("lark exploded")

    monkeypatch.setattr(GF, "grade", boom)
    row = G.grade_decision(F.MATH500, output("4"), MATH_GOLD, decision=decision(), target=target())
    assert (row.semantic_correct, row.u_reason) == (G.STATE_UNKNOWN, G.U_GRADER_ERROR)
    assert row.grading_status == G.GRADING_ERROR


def test_every_u_carries_a_typed_reason_and_no_resolved_grade_does():
    rows = [G.grade_decision(F.MATH500, output(answer), MATH_GOLD, decision=decision(),
                             target=target()) for answer in ("4", "5", None)]
    for row in rows:
        assert (row.u_reason is None) == (row.semantic_correct != G.STATE_UNKNOWN)
        assert row.u_reason is None or row.u_reason in G.U_REASONS
        assert row.audit_status == G.AUDIT_NONE and row.grader_version == G.GRADER_VERSION


# --------------------------------------------------------------------------- the one reader


def test_no_table_in_the_schema_carries_a_boolean_grade_column():
    for name, spec in tables.TABLES.items():
        leaked = set(spec.column_names) & G.LABELISH_COLUMNS
        assert not leaked, f"{name} carries {sorted(leaked)}"
    assert "semantic_correct" in tables.schema("grades_private").column_names


def test_tri_state_is_the_only_reader_and_refuses_a_label_ish_row():
    row = G.grade_decision(F.MATH500, output("4"), MATH_GOLD, decision=decision(), target=target())
    assert G.tri_state(row) == G.STATE_CORRECT
    assert G.tri_state(tables.row_to_dict(row)) == G.STATE_CORRECT
    with pytest.raises(LeakageError):
        G.tri_state(dict(tables.row_to_dict(row), is_correct=True))
    with pytest.raises(MaterializeError):
        G.tri_state({"semantic_correct": "yes"})


def test_tri_state_index_and_counts_key_on_the_generation_uid():
    rows = [G.grade_decision(F.MATH500, output(answer), MATH_GOLD,
                             decision=decision(uid=f"g:{index:064d}",
                                               decision_id=f"d:{index:040d}"), target=target())
            for index, answer in enumerate(("4", "5", None))]
    index = G.tri_state_index(rows)
    assert set(index.values()) == {G.STATE_CORRECT, G.STATE_WRONG, G.STATE_UNKNOWN}
    assert G.counts(rows) == {G.STATE_CORRECT: 1, G.STATE_WRONG: 1, G.STATE_UNKNOWN: 1}
    assert G.canonical_class_index(rows)[rows[0].generation_uid] is not None


def test_two_labels_for_one_sample_are_refused():
    import dataclasses

    row = G.grade_decision(F.MATH500, output("4"), MATH_GOLD, decision=decision(), target=target())
    other = dataclasses.replace(row, semantic_correct=G.STATE_WRONG)
    with pytest.raises(MaterializeError):
        G.tri_state_index([row, other])
    with pytest.raises(MaterializeError):
        G.merge_decision_ids([row, other])


def test_merging_collects_every_decision_that_reuses_one_sample():
    """D14: a root reference is another decision id for the same physical sample."""
    first = G.grade_decision(F.MATH500, output("4"), MATH_GOLD, decision=decision(), target=target())
    second = G.grade_decision(F.MATH500, output("4"), MATH_GOLD,
                              decision=decision(decision_id="d:" + "1" * 40), target=target())
    merged = G.merge_decision_ids([first, second])
    assert len(merged) == 1
    assert json.loads(merged[0].decision_ids_json) == sorted(
        json.loads(first.decision_ids_json) + json.loads(second.decision_ids_json))


def test_the_writer_refuses_a_policy_that_would_score_an_unresolved_sample_wrong(study_cfg):
    G.assert_failure_policy(study_cfg.failure_policy)
    import dataclasses

    loosened = dataclasses.replace(study_cfg.failure_policy, unresolved_semantics="incorrect")
    with pytest.raises(MaterializeError):
        G.assert_failure_policy(loosened)


# --------------------------------------------------------------------------- the worker


class StubGradeRunner:
    def __init__(self, tmp_path, study_cfg, rows):
        self.run_root = tmp_path
        self.cfg = study_cfg
        self.cell = type("Cell", (), {"cell_id": "grade/test/0000", "kind": "grade"})()
        self._rows = rows

    def decisions_for(self, source_item_id):
        assert source_item_id == ITEM
        return self._rows


def test_the_grade_worker_returns_private_rows_only(tmp_path, study_cfg, monkeypatch):
    GF.write_gold(tmp_path, F.MATH500, [MATH_GOLD])
    monkeypatch.setattr(I, "load_items", lambda run_root: {ITEM: stub_item()})
    rows = [(decision(uid=f"g:{i:064d}", decision_id=f"d:{i:040d}"), output(a))
            for i, a in enumerate(("4", "5"))]
    runner = StubGradeRunner(tmp_path, study_cfg, rows)
    item_file = G.grade_worker(runner, ITEM)
    assert item_file.rows == {}
    grades = item_file.private_rows["grades_private"]
    assert len(grades) == 2
    assert {G.tri_state(row) for row in grades} == {G.STATE_CORRECT, G.STATE_WRONG}
    assert item_file.has_private_rows
    assert tables.schema("grades_private").private
