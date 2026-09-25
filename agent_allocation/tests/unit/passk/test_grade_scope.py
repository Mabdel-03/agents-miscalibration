"""WP-G P3: the held-out grading fence and seal verification."""
from __future__ import annotations

import json
from hashlib import sha256

from agent_allocation.config import load_config
from agent_allocation.grading.worker import _grade_scope, _scope_config
from agent_allocation.passk import grade_scope as gs
from agent_allocation.paths import RunPaths
from tests.forced_cfg import PROJECT


def _cfg():
    return load_config(PROJECT / "configs" / "passk_cont_v1.json")


def _shard(tmp_path):
    study = tmp_path / "passk_cont_v1"
    paths = RunPaths.for_run(study / "artifacts" / "shards" / "mancer", local_state_dir=tmp_path / "local")
    ids = {}
    for partition in ("smoke", "e1_fill", "c48_extra", "v48_bank", "t200_bank", "t200_diag"):
        ids[partition] = [sha256(f"{partition}{k}".encode()).hexdigest() for k in range(3)]
        target = paths.jobs(partition)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps({"node_id": i, "record": {}}) + "\n" for i in ids[partition]))
    return study, paths, ids


def _write_seal(study, *, splits=None, tamper=False):
    art = study / "sealed" / "t200" / "predictions.parquet"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"predictions")
    seal = {"code_commit": "x", "artifacts": {"sealed/t200/predictions.parquet": sha256(b"predictions").hexdigest()}}
    if splits is not None:
        seal["splits"] = splits
    (study / "sealed" / "SEAL_predictions.json").write_text(json.dumps(seal))
    if tamper:
        art.write_bytes(b"changed")


def test_blocked_until_a_verified_seal(tmp_path):
    study, paths, ids = _shard(tmp_path)
    cfg = _cfg()
    heldout = set(ids["v48_bank"] + ids["t200_bank"] + ids["t200_diag"])
    assert gs.verify_seal(study) is None
    assert gs.blocked_node_ids(paths, cfg) == heldout
    skip, only = gs.grade_scope(paths, cfg)
    assert skip == heldout
    assert only == set(ids["smoke"] + ids["e1_fill"] + ids["c48_extra"])
    _write_seal(study, tamper=True)
    assert gs.verify_seal(study) is None and gs.blocked_node_ids(paths, cfg) == heldout
    _write_seal(study)
    assert gs.verify_seal(paths) is not None and gs.verify_seal(study) is not None
    assert gs.blocked_node_ids(paths, cfg) == frozenset()
    _skip, only = gs.grade_scope(paths, cfg)
    assert only == {i for v in ids.values() for i in v}


def test_seal_may_release_one_split(tmp_path):
    study, paths, ids = _shard(tmp_path)
    _write_seal(study, splits=["v48"])
    assert gs.blocked_node_ids(paths, _cfg()) == set(ids["t200_bank"] + ids["t200_diag"])


def test_empty_or_bad_seal_does_not_verify(tmp_path):
    study, paths, _ids = _shard(tmp_path)
    (study / "sealed").mkdir(parents=True)
    (study / "sealed" / "SEAL_predictions.json").write_text(json.dumps({"artifacts": {}}))
    assert gs.verify_seal(study) is None
    (study / "sealed" / "SEAL_predictions.json").write_text("{not json")
    assert gs.verify_seal(study) is None
    (study / "sealed" / "SEAL_predictions.json").write_text(json.dumps({"artifacts": {"sealed/missing.json": "0"}}))
    assert gs.verify_seal(study) is None


def test_legacy_config_is_unrestricted(tmp_path):
    _study, paths, _ids = _shard(tmp_path)
    legacy = load_config(PROJECT / "configs" / "forced27b.json")
    assert gs.blocked_node_ids(paths, legacy) == frozenset()
    assert gs.grade_scope(paths, legacy) == (frozenset(), None)
    assert _grade_scope(paths, legacy) == (frozenset(), None)
    assert _grade_scope(paths, None) == (frozenset(), None)


def test_worker_scope_config_prefers_flag_then_shard_config(tmp_path):
    _study, paths, ids = _shard(tmp_path)
    assert _scope_config(paths, None) is None
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    (paths.state_dir / "shard_config.json").write_text(json.dumps(_cfg()))
    cfg = _scope_config(paths, None)
    skip, only = _grade_scope(paths, cfg)
    assert skip == set(ids["v48_bank"] + ids["t200_bank"] + ids["t200_diag"]) and only is not None
    assert _scope_config(paths, str(PROJECT / "configs" / "forced27b.json")).get("passk") is None
