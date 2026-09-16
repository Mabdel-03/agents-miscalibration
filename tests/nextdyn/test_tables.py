"""Tables: the row dataclasses are the schema; keys are asserted; item files round-trip.

``test_composite_decision_key_unique`` (spec 18.1) runs on a synthetic tables directory
written through ``tables.write_table``, exactly the way materialize will write one.
"""

from __future__ import annotations

import dataclasses
import json
import os
import stat

import pytest

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import tables as T
from agents_scaling.nextdyn.errors import ManifestError, MaterializeError
from agents_scaling.nextdyn.ids import CellKind

_STR_DEFAULTS = {"str": "x", "int": 0, "float": 0.0, "bool": False, "list[str]": (), "list[int]": ()}


def _row(row_type, **over):
    """A fully populated row: nullable columns None, others a typed placeholder."""
    spec = T.schema_of(row_type)
    values = {c.name: (None if c.nullable else _STR_DEFAULTS[c.type]) for c in spec.columns}
    values.update(over)
    return row_type(**values)


def _decision(decision_id, *, episode_id="episode|x", cell_kind="episode", round_=1, slot=0,
              phase=K.PHASE_REVISION, reuse_of=None, is_reference=False, generation_uid="g:1", root_team_id="team:x"):
    return _row(T.DecisionRow, decision_id=decision_id, episode_id=episode_id, cell_kind=cell_kind, round=round_,
                agent_slot=slot, phase=phase, reuse_of=reuse_of, is_reference=is_reference,
                generation_uid=generation_uid, root_team_id=root_team_id, study_id="s", source_item_id="i",
                item_variant_id="canonical", protocol_hash="p", receiver_model_hash="1234567890abcdef",
                budget_hash="b", replicate_id=0, cell_id="c", request_hashes_json="[]", aliased=False,
                parent_decision_ids_json="[]", seen_transmission_ids_json="[]", parse_status="ok",
                field_order_violation=False, validity=True, delivery_success=True, infrastructure_status="ok",
                termination_reason="stop", boundary_injected=False, continuation_used=False,
                discarded_span_tokens=0, thinking_tokens=1, final_tokens=1, prompt_tokens=1, cached_prefill_tokens=0)


def _write(tables_dir, **rows):
    for name, items in rows.items():
        T.write_table(items, tables_dir / f"{name}.parquet")
    return tables_dir


# ---------------------------------------------------------------------------- 18.1


def test_composite_decision_key_unique(tmp_path):
    clean = _write(tmp_path / "clean", decisions=[_decision("d:1"), _decision("d:2", slot=1)])
    assert [v for v in T.check_keys(clean, strict=False) if v.table == "decisions"] == []

    dup = _write(tmp_path / "dup", decisions=[_decision("d:1"), _decision("d:1", slot=1)])
    violations = T.check_keys(dup, strict=False)
    assert [(v.table, v.constraint, v.n_duplicate_groups) for v in violations] == [("decisions", "primary_key", 1)]
    assert violations[0].examples[0].values == ("d:1",)
    with pytest.raises(MaterializeError):
        T.assert_keys(dup, strict=False)

    # the ARD36 failure shape: two rows for one (cell_kind, episode, round, slot, phase) under different ids
    logical = _write(tmp_path / "logical", decisions=[_decision("d:1"), _decision("d:2")])
    violations = T.check_keys(logical, strict=False)
    assert [(v.table, v.constraint) for v in violations] == [("decisions", "unique")]
    assert violations[0].key == ("cell_kind", "episode_id", "round", "agent_slot", "phase")


# ---------------------------------------------------------------------------- schemas


def test_every_table_is_a_row_dataclass_with_a_primary_key_and_a_producer():
    assert len(T.TABLES) == len(T.ROW_TYPES)
    for name, spec in T.TABLES.items():
        assert spec.primary_key and set(spec.primary_key) <= set(spec.column_names)
        for constraint in spec.unique:
            assert set(constraint) <= set(spec.column_names)
        assert spec.producer.startswith("agents_scaling.nextdyn.") and spec.consumers
        assert T.PRIMARY_KEYS[name] == spec.primary_key
        assert dataclasses.is_dataclass(spec.row_type) and spec.row_type.__table__ == name


def test_plan_keys_match_the_plan_table(study_cfg):
    assert T.PRIMARY_KEYS["decisions"] == ("decision_id",)
    assert T.PRIMARY_KEYS["grades_private"] == ("generation_uid", "grader_hash")
    assert T.PRIMARY_KEYS["activations_manifest"] == ("decision_id", "layer", "anchor")
    assert T.PRIMARY_KEYS["banks"] == ("bank_draw_id",)
    assert T.TABLES["transmissions"].unique == (("receiver_decision_id", "sender_slot", "source_round"),)
    assert T.TABLES["selections"].unique == (("selector", "selector_version", "pool_id", "pool_scope"),)
    assert T.TABLES["tool_events"].unique == (("decision_id", "request_index"),)
    assert T.PRIVATE_TABLES == {"grades_private", "root_teams_private"}
    # the twelve composite-key fields are decision columns, in spec order
    cols = T.TABLES["decisions"].column_names
    assert cols[1:13] == ("study_id", "cell_kind", "source_item_id", "item_variant_id", "root_team_id",
                          "protocol_hash", "receiver_model_hash", "budget_hash", "replicate_id", "round",
                          "agent_slot", "phase")
    assert "semantic_correct" in T.TABLES["grades_private"].column_names
    assert not any(c.type == "bool" for c in T.TABLES["grades_private"].columns), "no boolean grade column (D15)"
    # the generating model lives in receiver_model_hash and nowhere else on decisions (R1, spec 17.2)
    assert "model_hash" not in cols


def test_project_is_a_strict_whitelist():
    row = dataclasses.asdict(_row(T.ItemVariantRow, item_variant_id="v", source_item_id="i"))
    assert T.project("item_variants", row)["item_variant_id"] == "v"
    with pytest.raises(MaterializeError):
        T.project("item_variants", {**row, "composition": "C1_COHERENT"})
    with pytest.raises(MaterializeError):
        T.project("item_variants", {**row, "item_variant_id": None})
    with pytest.raises(MaterializeError):
        T.project("item_variants", {**row, "source_item_id": 3})
    with pytest.raises(MaterializeError):
        T.project("no_such_table", row)


def test_row_from_dict_refuses_unknown_and_missing_keys():
    good = dataclasses.asdict(_row(T.ToolEventRow, tool_event_id="t", decision_id="d", request_index=0))
    assert T.row_from_dict(T.ToolEventRow, good).tool_event_id == "t"
    with pytest.raises(MaterializeError):
        T.row_from_dict(T.ToolEventRow, {**good, "extra": 1})
    with pytest.raises(MaterializeError):
        T.row_from_dict(T.ToolEventRow, {k: v for k, v in good.items() if k != "status"})


def test_write_table_refuses_a_row_of_another_table(tmp_path):
    with pytest.raises(MaterializeError):
        T.write_table([_row(T.ToolEventRow, tool_event_id="t")], tmp_path / "decisions.parquet")


# ---------------------------------------------------------------------------- item files


def test_item_file_round_trip_refuses_unknown_keys(tmp_path):
    item = T.ItemFile(schema_version=T.SCHEMA_VERSION, cell_id="episode.MATH.8B.s000", cell_kind=CellKind.EPISODE,
                      source_item_id="item01", status=T.ItemStatus.COMPLETE, bundle_export_hash="e" * 64,
                      rows={"decisions": (_decision("d:1"),), "transmissions": ()})
    path = tmp_path / "item01.json"
    item.write(path)
    back = T.ItemFile.read(path)
    assert back == item
    payload = json.loads(path.read_text())
    assert payload["cell_kind"] == "episode" and payload["rows"]["decisions"][0]["decision_id"] == "d:1"
    with pytest.raises(ManifestError):
        T.ItemFile.from_dict({**payload, "grades": []})
    with pytest.raises(ManifestError):
        T.ItemFile.from_dict({**payload, "rows": {**payload["rows"], "no_such_table": []}})
    with pytest.raises(MaterializeError):
        T.ItemFile.from_dict({**payload, "rows": {"decisions": [{**payload["rows"]["decisions"][0], "gold": 1}]}})
    with pytest.raises(ManifestError):
        T.ItemFile.from_dict({k: v for k, v in payload.items() if k != "status"})
    with pytest.raises(ManifestError):
        T.ItemFile(schema_version=1, cell_id="c", cell_kind="episode", source_item_id="i", status="complete",
                   bundle_export_hash=None, rows={"decisions": (_row(T.ToolEventRow, tool_event_id="t"),)})


def _grade(uid="g:1"):
    return _row(T.GradePrivateRow, generation_uid=uid, grader_hash="h", decision_ids_json="[]",
                source_item_id="i", family="MATH", semantic_correct="C", grading_status="graded",
                gold_ref_hash="r", audit_status="none", grader_version="v1")


def test_private_rows_never_land_under_cells(tmp_run_root):
    """R7 / plan §2.5: gold-derived rows travel in ``private_rows`` and are written under
    ``protected/`` (0600 in 0700), never in the public item file the generation lane and
    ``deps.dependency_status`` read."""
    item = T.ItemFile(schema_version=T.SCHEMA_VERSION, cell_id="grade.MATH.s000", cell_kind=CellKind.GRADE,
                      source_item_id="item01", status=T.ItemStatus.COMPLETE, bundle_export_hash=None,
                      rows={}, private_rows={"grades_private": (_grade(),)})
    assert item.has_private_rows
    public = T.item_path(tmp_run_root, "grade.MATH.s000", "item01")
    protected = T.protected_item_path(tmp_run_root, "grade.MATH.s000", "item01")
    assert "cells" in public.parts and str(protected).startswith(str(tmp_run_root / "protected"))
    with pytest.raises(ManifestError, match="protected_path"):
        item.write(public)                                       # private rows need the protected path
    with pytest.raises(ManifestError, match="under cells/"):
        item.write(public, protected_path=public.with_name("item01.private.json"))
    item.write(public, protected_path=protected)
    public_payload = json.loads(public.read_text())
    assert "grades_private" not in json.dumps(public_payload) and "private_rows" not in public_payload
    assert all("grades_private" not in f.read_text() for f in (tmp_run_root / "cells").rglob("*.json"))
    assert stat.S_IMODE(os.stat(protected).st_mode) == T.PROTECTED_FILE_MODE
    assert stat.S_IMODE(os.stat(protected.parent).st_mode) == T.PROTECTED_DIR_MODE
    assert json.loads(protected.read_text())["private_rows"]["grades_private"][0]["semantic_correct"] == "C"
    assert T.ItemFile.read(public) == dataclasses.replace(item, private_rows={})
    assert T.ItemFile.read(public, protected_path=protected) == item
    # a private table under the public channel, or a public one under private_rows, is refused
    with pytest.raises(ManifestError, match="private_rows"):
        T.ItemFile(schema_version=1, cell_id="c", cell_kind="grade", source_item_id="i", status="complete",
                   bundle_export_hash=None, rows={"grades_private": (_grade(),)})
    with pytest.raises(ManifestError, match="ItemFile.rows"):
        T.ItemFile(schema_version=1, cell_id="c", cell_kind="episode", source_item_id="i", status="complete",
                   bundle_export_hash=None, rows={}, private_rows={"decisions": (_decision("d:1"),)})
    with pytest.raises(ManifestError):
        T.ItemFile.from_dict(public_payload, {**json.loads(protected.read_text()), "cell_id": "other"})
    with pytest.raises(ManifestError):
        T.ItemFile.from_dict({**public_payload, "rows": {"grades_private": []}})


# ---------------------------------------------------------------------------- parquet io


def test_write_table_matches_the_study_write_parquet_semantics(tmp_path):
    """Vendored from study/aggregate.py:257: same rows, same values, ``.empty`` for no rows."""
    from agents_scaling.study.aggregate import _write_parquet

    rows = [_row(T.ItemVariantRow, item_variant_id=f"v{i}", source_item_id="i") for i in range(3)]
    T.write_table(rows, tmp_path / "item_variants.parquet")
    _write_parquet(tmp_path / "ref.parquet", [dataclasses.asdict(r) for r in rows])
    ours = T.read_table(tmp_path, "item_variants")
    import pyarrow.parquet as pq

    theirs = pq.read_table(tmp_path / "ref.parquet").to_pylist()
    assert ours == theirs
    assert tuple(pq.ParquetFile(tmp_path / "item_variants.parquet").schema_arrow.names) == \
        T.TABLES["item_variants"].column_names

    T.write_table([], tmp_path / "empty" / "metrics.parquet")
    _write_parquet(tmp_path / "empty" / "ref.parquet", [])
    assert (tmp_path / "empty" / "metrics.empty").exists() and (tmp_path / "empty" / "ref.empty").exists()
    assert T.read_table(tmp_path / "empty", "metrics") == []
    with pytest.raises(MaterializeError):
        T.read_table(tmp_path / "empty", "decisions")


def test_missing_table_is_unverifiable_under_strict_only(tmp_path):
    _write(tmp_path, metrics=[_row(T.MetricRow, metric_id="m")])
    strict = {(v.table, v.constraint) for v in T.check_keys(tmp_path, strict=True)}
    assert ("decisions", "primary_key") in strict and ("metrics", "primary_key") not in strict
    assert T.check_keys(tmp_path, strict=False) == []
    checks = {(c.table, c.constraint): c for c in T.run_checks(tmp_path)}
    assert checks[("metrics", "primary_key")].status == T.STATUS_OK
    assert checks[("decisions", "primary_key")].status == T.STATUS_UNVERIFIABLE


# ---------------------------------------------------------------------------- referential


def _team(root_team_id, members):
    return _row(T.RootTeamRow, root_team_id=root_team_id, ordered_generation_uids_json=json.dumps(members),
                ordered_decision_ids_json="[]", shared_donor_uids_json="[]", export_hash=root_team_id + "x")


def test_assert_referential_accepts_a_consistent_directory(tmp_path):
    bank = _decision("d:bank", episode_id="donor_bank|i|canonical|m|bbbbbbbb|k0", cell_kind="donor_bank",
                     round_=0, slot=0, phase=K.PHASE_BANK, generation_uid="g:bank", root_team_id="")
    ref = _decision("d:ref", round_=0, phase=K.PHASE_ROOT_REF, reuse_of="d:bank", is_reference=True,
                    generation_uid="g:bank")
    rev = _decision("d:rev", round_=1)
    tr = _row(T.TransmissionRow, transmission_id="t", source_round=0, destination_round=1,
              receiver_decision_id="d:rev", truncated_fields_json="[]", filler_flag=False)
    _write(tmp_path, decisions=[bank, ref, rev], transmissions=[tr], root_teams=[_team("team:x", ["g:bank"])])
    assert T.assert_referential(tmp_path) == []
    # a decision without its generating model is a violation (spec 17.2)
    _write(tmp_path / "nomodel", decisions=[dataclasses.replace(rev, receiver_model_hash="")],
           transmissions=[], root_teams=[])
    assert any("receiver_model_hash is empty" in v for v in T.assert_referential(tmp_path / "nomodel", strict=False))


def test_assert_referential_reports_every_declared_violation(tmp_path):
    difficulty = _decision("d:diff", episode_id="difficulty_bank|i|canonical|m|bbbbbbbb|k0",
                           cell_kind="difficulty_bank", round_=0, slot=0, phase=K.PHASE_BANK,
                           generation_uid="g:diff", root_team_id="")
    dangling = _decision("d:ref", round_=0, phase=K.PHASE_ROOT_REF, reuse_of="d:nowhere", is_reference=True)
    bad_bank = _decision("d:bank2", episode_id="donor_bank|i|canonical|m|bbbbbbbb|k1", cell_kind="donor_bank",
                         round_=0, slot=1, phase=K.PHASE_BANK, root_team_id="")
    mixed = _decision("d:mix", episode_id="episode|x", cell_kind="receiver", round_=1, slot=3,
                      phase=K.PHASE_RECEIVER_DECISION)
    tr = _row(T.TransmissionRow, transmission_id="t", source_round=1, destination_round=1,
              receiver_decision_id="d:mix", truncated_fields_json="[]", filler_flag=False)
    _write(tmp_path, decisions=[difficulty, dangling, bad_bank, mixed], transmissions=[tr],
           root_teams=[_team("team:x", ["g:diff"])])
    bad_bank_phase = dataclasses.replace(bad_bank, phase=K.PHASE_REVISION)
    T.write_table([difficulty, dangling, bad_bank_phase, mixed], tmp_path / "decisions.parquet")
    violations = T.assert_referential(tmp_path, strict=False)
    text = "\n".join(violations)
    assert "reuse_of d:nowhere does not resolve" in text
    assert "difficulty-bank draws" in text
    assert "source_round 1 != destination_round 1 - 1" in text
    assert "maps to ['episode', 'receiver']" in text
    assert "must carry phase='bank'" in text
    with pytest.raises(MaterializeError):
        T.assert_referential(tmp_path)


def test_derived_tables_have_declared_keys_only():
    assert set(T.DERIVED_PRIMARY_KEYS) >= {"episodes", "team_rounds", "agent_rounds", "transitions", "difficulty"}
    assert not (set(T.DERIVED_PRIMARY_KEYS) & set(T.TABLES))
