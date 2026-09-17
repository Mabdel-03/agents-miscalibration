"""Atomic publish: a crash leaves no tables directory, and a duplicate key is fatal.

Plan §3.4, spec 17.3.  ARD36 wrote parquet files into the live directory as it went and ran
the key check afterwards as a warning, which is how 14,336 duplicate decision keys shipped.
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.nextdyn import deps, materialize as MAT, tables
from agents_scaling.nextdyn.errors import MaterializeError
from agents_scaling.nextdyn.ids import CellKind
from tests.nextdyn.conftest import placeholder

ITEM = "nditem0000000001"


def decision_row(**kwargs) -> tables.DecisionRow:
    row = placeholder("decisions") | dict(
        decision_id="d:" + "a" * 40, study_id="task_conditioned_dynamics_v1",
        cell_kind=CellKind.DIFFICULTY_BANK.value, source_item_id=ITEM,
        item_variant_id=ITEM + ".v0", root_team_id="", protocol_hash="",
        receiver_model_hash="m" * 16, budget_hash="b" * 8, replicate_id=0, round=0, agent_slot=0,
        phase="bank", episode_id="difficulty_bank|x",
        cell_id="difficulty_bank.MATH.primary.a.x.pilot.r0.s0",
        generation_uid="g:" + "c" * 64, content_hash="c:" + "d" * 64, parse_status="OK",
        aliased=False, is_reference=False)
    row.update(kwargs)
    return tables.row_from_dict(tables.DecisionRow, row)


def nothing_published(run_root) -> bool:
    """No versioned set, and no published pointer.

    The empty ``tables/`` directory the run-root skeleton creates is not a publication: what
    a reader keys on is the ``MANIFEST.json`` (and the parquet files) the publish renames into
    place last.
    """
    published = run_root / MAT.TABLES_DIR
    return (not list(run_root.glob(f"{MAT.VERSIONED_PREFIX}*"))
            and not (published / MAT.MANIFEST_FILE).exists()
            and not list(published.glob("*.parquet")))


def write_cell(run_root, cell_id: str, rows, *, private=None, item: str = ITEM) -> None:
    item_file = tables.ItemFile(
        schema_version=tables.SCHEMA_VERSION, cell_id=cell_id,
        cell_kind=CellKind.DIFFICULTY_BANK, source_item_id=item,
        status=tables.ItemStatus.COMPLETE, bundle_export_hash=None, rows=rows,
        private_rows=private or {})
    item_file.write(tables.item_path(run_root, cell_id, item),
                    protected_path=tables.protected_item_path(run_root, cell_id, item))
    directory = deps.cell_dir(run_root, cell_id)
    (directory / deps.META_FILE).write_text(json.dumps({"cell_id": cell_id, "n_items": 1}))


# --------------------------------------------------------------------------- collection


def test_only_complete_cells_are_published(tmp_run_root):
    """An in-flight cell's items are real, but its census is not final."""
    write_cell(tmp_run_root, "difficulty_bank.MATH.primary.a.x.pilot.r0.s0",
               {"decisions": (decision_row(),)})
    half = deps.cell_dir(tmp_run_root, "difficulty_bank.MATH.primary.a.x.pilot.r0.s1")
    (half / tables.ITEMS_DIR).mkdir(parents=True)
    (half / tables.ITEMS_DIR / f"{ITEM}.json").write_text("{}")
    collected = MAT.collect(tmp_run_root)
    assert collected.cells == 1 and collected.skipped == 1
    assert len(collected.public["decisions"]) == 1


def test_first_decision_id_records_the_reuse(tmp_run_root):
    """plan §1.3: reuse is recorded, never silent -- two rows, one generation."""
    first = decision_row(decision_id="d:" + "1" * 40)
    second = decision_row(decision_id="d:" + "2" * 40, aliased=True)
    filled = MAT.fill_first_decision_id([second, first])
    assert {row.first_decision_id for row in filled} == {"d:" + "1" * 40}
    # and a row with no generation (a context failure) is left alone
    orphan = decision_row(decision_id="d:" + "3" * 40, generation_uid="")
    assert MAT.fill_first_decision_id([orphan])[0].first_decision_id is None


def test_collapse_by_key_is_for_one_object_recorded_twice():
    rows = [decision_row(decision_id="d:" + "1" * 40), decision_row(decision_id="d:" + "1" * 40)]
    kept, collapsed = MAT.collapse_by_key(rows, "decision_id")
    assert len(kept) == 1 and collapsed == 1


# --------------------------------------------------------------------------- publication


def test_materialize_publishes_a_versioned_set_and_a_pointer(tmp_run_root):
    write_cell(tmp_run_root, "difficulty_bank.MATH.primary.a.x.pilot.r0.s0",
               {"decisions": (decision_row(),)})
    report = MAT.materialize(tmp_run_root, strict=False)
    assert report.version == 1
    assert (report.tables_dir / MAT.MANIFEST_FILE).exists()
    published = tmp_run_root / MAT.TABLES_DIR
    assert published.exists()
    assert len(tables.read_table(published, "decisions")) == 1
    manifest = json.loads((report.tables_dir / MAT.MANIFEST_FILE).read_text())
    assert manifest["cells"] == 1 and manifest["rows"]["decisions"] == 1
    # a second publish is a new version, and the pointer moves
    again = MAT.materialize(tmp_run_root, strict=False)
    assert again.version == 2
    assert (tmp_run_root / f"{MAT.VERSIONED_PREFIX}1").exists()


def test_private_rows_are_published_under_protected_only(tmp_run_root):
    grade = tables.row_from_dict(tables.GradePrivateRow, placeholder("grades_private") | {
        "generation_uid": "g:" + "c" * 64, "grader_hash": "h" * 64, "semantic_correct": "C",
        "grading_status": "resolved", "canonical_class": "expr:4"})
    write_cell(tmp_run_root, "grade.MATH.x.x.x.pilot.r0.s0", {"decisions": (decision_row(),)},
               private={"grades_private": (grade,)})
    report = MAT.materialize(tmp_run_root, strict=False)
    assert (report.protected_dir / "grades_private.parquet").exists()
    assert not (report.tables_dir / "grades_private.parquet").exists()
    assert len(tables.read_table(report.protected_dir, "grades_private")) == 1


def test_partial_materialize_leaves_no_tables_dir(tmp_run_root, monkeypatch):
    """A crash mid-publish must leave "not materialized", never "materialized badly".

    The whole set is built under ``tables.<ts>.tmp/`` and renamed into place only after every
    assertion has passed and ``MANIFEST.json`` is written, so a reader can always tell the two
    apart -- which ARD36 could not.
    """
    write_cell(tmp_run_root, "difficulty_bank.MATH.primary.a.x.pilot.r0.s0",
               {"decisions": (decision_row(),)})

    boom = RuntimeError("killed mid-write")

    def explode(*args, **kwargs):
        raise boom

    monkeypatch.setattr(MAT, "manifest_payload", explode)
    with pytest.raises(RuntimeError):
        MAT.materialize(tmp_run_root, strict=False)
    assert nothing_published(tmp_run_root)
    assert not list(tmp_run_root.glob(f"*{MAT.TMP_SUFFIX}"))   # and no half-written tmp set

    monkeypatch.undo()
    report = MAT.materialize(tmp_run_root, strict=False)      # and the next pass just works
    assert not nothing_published(tmp_run_root) and report.version == 1


def test_a_duplicate_decision_key_aborts_the_publish(tmp_run_root):
    """spec 15.3 ``duplicate_logical_keys: fatal`` -- it aborts, it does not warn."""
    same_key = decision_row(decision_id="d:" + "9" * 40)
    write_cell(tmp_run_root, "difficulty_bank.MATH.primary.a.x.pilot.r0.s0",
               {"decisions": (same_key, same_key)})
    with pytest.raises(MaterializeError, match="primary-key"):
        MAT.materialize(tmp_run_root, strict=True)
    assert nothing_published(tmp_run_root)


def test_the_referential_assertions_are_fatal_here(tmp_run_root):
    """A ``reuse_of`` that resolves nowhere is a broken lineage, not a missing nicety."""
    write_cell(tmp_run_root, "difficulty_bank.MATH.primary.a.x.pilot.r0.s0",
               {"decisions": (decision_row(reuse_of="d:" + "f" * 40),)})
    with pytest.raises(MaterializeError):
        MAT.materialize(tmp_run_root, strict=True)


def test_the_assigned_receiver_packet_is_exempt_from_the_previous_round_rule(tmp_run_root):
    """spec 9.1: a receiver's packets are shown, not received from an earlier round.

    ``tables.DECLARED_EDGE_ASSIGNED`` is the scope of that exemption and must be the same
    string ``receiver.py`` writes.
    """
    from agents_scaling.nextdyn import receiver as RX

    assert tables.DECLARED_EDGE_ASSIGNED == RX.DECLARED_EDGE_ASSIGNED


def test_key_checks_run_per_visibility(tmp_run_root):
    """The private tables are not in the public set, so the check runs over each directory."""
    write_cell(tmp_run_root, "difficulty_bank.MATH.primary.a.x.pilot.r0.s0",
               {"decisions": (decision_row(),)})
    report = MAT.materialize(tmp_run_root, strict=False)
    assert MAT.assert_published_keys(report.tables_dir, report.protected_dir, strict=False) == []
