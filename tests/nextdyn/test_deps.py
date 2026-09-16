"""Dependency vocabulary: registers are content-addressed and never empty, bundles carry a
sidecar, cells are met only when complete and owned, and tampering is reported not met."""

from __future__ import annotations

import json

import pytest

from agents_scaling.nextdyn import deps, tables as T
from agents_scaling.nextdyn.errors import DependencyUnmet, ManifestError
from agents_scaling.nextdyn.ids import CellKind


def _item_file(cell_id: str, item: str) -> T.ItemFile:
    return T.ItemFile(schema_version=1, cell_id=cell_id, cell_kind=CellKind.GRADE, source_item_id=item,
                      status=T.ItemStatus.COMPLETE, bundle_export_hash=None, rows={})


def _complete_cell(run_root, cell_id: str, items=("i1", "i2")):
    cell = deps.cell_dir(run_root, cell_id)
    (cell / "items").mkdir(parents=True)
    for item in items:
        _item_file(cell_id, item).write(cell / "items" / f"{item}.json")
    (cell / "meta.json").write_text(json.dumps({"cell_id": cell_id, "n_items": len(items)}))
    return cell


# ---------------------------------------------------------------------------- vocabulary


def test_parse_dependency_rejects_unknown_prefixes_and_bare_ids():
    assert deps.parse_dependency("cell:grade.MATH.s000") == deps.ParsedDependency("cell", "grade.MATH.s000")
    assert deps.parse_dependency("bundle:MATH/b0003").parts == ("MATH", "b0003")
    assert deps.parse_dependency("register:pool/" + "a" * 64).parts == ("pool", "a" * 64)
    for bad in ("grade.MATH.s000", "state:" + "a" * 64, "controller:" + "a" * 64, "register:teams/" + "a" * 64,
                "register:pool/xyz", "bundle:MATH", "cell:../x"):
        with pytest.raises(DependencyUnmet):
            deps.parse_dependency(bad)
    assert deps.cell_dependency("x.y") == "cell:x.y"
    assert deps.bundle_dependency("MATH", 3) == "bundle:MATH/b0003"
    assert deps.register_dependency("packets", "A" * 40) == "register:packets/" + "a" * 40
    with pytest.raises(ManifestError):
        deps.register_dependency("teams", "a" * 40)


# ---------------------------------------------------------------------------- registers


def test_register_is_content_addressed_and_idempotent(tmp_run_root):
    payload = {"entries": ["pool:1", "pool:2"], "scope": "terminal_only"}
    sha = deps.write_register(tmp_run_root, "pool", payload)
    assert deps.write_register(tmp_run_root, "pool", payload) == sha
    assert deps.write_register(tmp_run_root, "pool", {"entries": ["pool:2", "pool:1"], "scope": "terminal_only"}) != sha
    assert deps.verify_register(tmp_run_root, "pool", sha).met
    body = deps.read_register(tmp_run_root, "pool", sha)
    assert body["kind"] == "pool" and body["register_sha256"] == sha and body["entries"] == ["pool:1", "pool:2"]
    status = deps.dependency_status(tmp_run_root, deps.register_dependency("pool", sha))
    assert status.met and status.kind == "register"


def test_an_empty_or_float_bearing_register_is_refused(tmp_run_root):
    with pytest.raises(ManifestError):
        deps.write_register(tmp_run_root, "packets", {"entries": []})
    with pytest.raises(ManifestError):
        deps.write_register(tmp_run_root, "packets", {"n": 3})
    with pytest.raises(ManifestError):
        deps.write_register(tmp_run_root, "packets", {})
    with pytest.raises(ManifestError):
        deps.write_register(tmp_run_root, "teams", {"entries": [1]})
    # a float (a confidence, a weight) is a typed refusal, never a TypeError out of identity.jcs
    with pytest.raises(ManifestError, match="canonical-JSON safe"):
        deps.write_register(tmp_run_root, "pool", {"entries": ["a"], "weight": 0.5})
    assert not any((tmp_run_root / "registers").rglob("*.json"))


def test_a_tampered_register_is_reported_not_met(tmp_run_root):
    sha = deps.write_register(tmp_run_root, "panel", {"entries": ["d:1"]})
    path = deps.register_path(tmp_run_root, "panel", sha)
    data = json.loads(path.read_text())
    data["entries"].append("d:2")
    path.write_text(json.dumps(data))
    status = deps.verify_register(tmp_run_root, "panel", sha)
    assert status.status == deps.STATUS_TAMPERED and "declared" in status.detail
    dep = deps.dependency_status(tmp_run_root, deps.register_dependency("panel", sha))
    assert dep.status == deps.STATUS_TAMPERED and not dep.met
    with pytest.raises(DependencyUnmet):
        deps.read_register(tmp_run_root, "panel", sha)
    # emptied after sealing -> tampered as well; missing -> unmet
    data["entries"] = []
    path.write_text(json.dumps(data))
    assert deps.verify_register(tmp_run_root, "panel", sha).status == deps.STATUS_TAMPERED
    assert deps.dependency_status(tmp_run_root, deps.register_dependency("panel", "f" * 64)).status == deps.STATUS_UNMET
    assert deps.verify_register(tmp_run_root, "panel", "f" * 64).status == deps.STATUS_MISSING


# ---------------------------------------------------------------------------- bundles


def test_bundle_sidecar_is_verified(tmp_run_root):
    payload = {"root_team_id": "team:" + "a" * 32, "slots": [{"slot": 0, "packet": "..."}]}
    sha = deps.write_bundle(tmp_run_root, "MATH", 7, payload)
    assert deps.write_bundle(tmp_run_root, "MATH", 7, payload) == sha
    assert deps.bundle_path(tmp_run_root, "MATH", 7).name == "b0007.json"
    assert deps.verify_bundle(tmp_run_root, "MATH", "b0007").met
    body, export_hash = deps.read_bundle(tmp_run_root, "MATH", 7)
    assert body == payload and export_hash == sha
    assert deps.dependency_status(tmp_run_root, "bundle:MATH/b0007").met
    with pytest.raises(ManifestError):
        deps.write_bundle(tmp_run_root, "MATH", 7, {**payload, "slots": []})
    deps.bundle_path(tmp_run_root, "MATH", 7).write_text("{}")
    assert deps.dependency_status(tmp_run_root, "bundle:MATH/b0007").status == deps.STATUS_TAMPERED
    assert deps.dependency_status(tmp_run_root, "bundle:MATH/b0008").status == deps.STATUS_UNMET
    deps.bundle_path(tmp_run_root, "MATH", 7).with_name("b0007.json.sha256").unlink()
    assert deps.verify_bundle(tmp_run_root, "MATH", 7).status == deps.STATUS_TAMPERED


# ---------------------------------------------------------------------------- cells


def test_cell_dependency_needs_meta_and_directory_ownership(tmp_run_root):
    dep = "cell:grade.MATH.8B.s000"
    assert deps.dependency_status(tmp_run_root, dep).status == deps.STATUS_UNMET
    cell = _complete_cell(tmp_run_root, "grade.MATH.8B.s000")
    assert deps.dependency_status(tmp_run_root, dep).met
    assert deps.unmet(tmp_run_root, [dep]) == []

    # an intruding item from another manifest (defect D-04) -> tampered, never met
    _item_file("grade.MATH.8B.s001", "i9").write(cell / "items" / "i9.json")
    status = deps.dependency_status(tmp_run_root, dep)
    assert status.status == deps.STATUS_TAMPERED and "n_items" in status.detail
    (cell / "meta.json").write_text(json.dumps({"cell_id": "grade.MATH.8B.s000", "n_items": 3}))
    status = deps.dependency_status(tmp_run_root, dep)
    assert status.status == deps.STATUS_TAMPERED and "belongs to cell" in status.detail

    # in-flight items or a suspension mean not complete
    (cell / "items" / "i9.json").unlink()
    (cell / "meta.json").write_text(json.dumps({"cell_id": "grade.MATH.8B.s000", "n_items": 2}))
    (cell / "incomplete").mkdir()
    (cell / "incomplete" / "i3.json").write_text("{}")
    assert deps.dependency_status(tmp_run_root, dep).status == deps.STATUS_UNMET
    (cell / "incomplete" / "i3.json").unlink()
    (cell / "SUSPENDED.json").write_text("{}")
    assert deps.dependency_status(tmp_run_root, dep).status == deps.STATUS_UNMET
    (cell / "SUSPENDED.json").unlink()
    assert deps.dependency_status(tmp_run_root, dep).met

    # meta.json claiming another cell id -> tampered
    (cell / "meta.json").write_text(json.dumps({"cell_id": "other", "n_items": 2}))
    assert deps.dependency_status(tmp_run_root, dep).status == deps.STATUS_TAMPERED


def test_unmet_keeps_manifest_order_and_includes_tampered(tmp_run_root):
    _complete_cell(tmp_run_root, "a")
    sha = deps.write_register(tmp_run_root, "pool", {"entries": ["x"]})
    deps.register_path(tmp_run_root, "pool", sha).write_text("not json")
    statuses = deps.unmet(tmp_run_root, ["cell:a", "register:pool/" + sha, "cell:b", "bundle:MATH/b0000"])
    assert [(s.dep, s.status) for s in statuses] == [
        ("register:pool/" + sha, deps.STATUS_TAMPERED), ("cell:b", deps.STATUS_UNMET),
        ("bundle:MATH/b0000", deps.STATUS_UNMET)]
