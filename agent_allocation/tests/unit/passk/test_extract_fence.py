from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from agent_allocation.passk.extract import extract, load_draws
from agent_allocation.passk.labels import SealError, blocked_node_ids, join_labels, verify_seal

from . import fakeroot


def _seal(run_root, rel="sealed/v48/predictions.parquet"):
    target = run_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": ["aime/2026/1"], "p": [0.5]}).to_parquet(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (run_root / "sealed" / "SEAL_predictions.json").write_text(json.dumps(
        {"code_commit": "abc", "reviewed_commit": "abc", "artifacts": {rel: digest}}))
    return target


def test_pre_seal_allow_list_and_label_fence(tmp_path):
    ids = fakeroot.build(tmp_path)
    manifest = extract(tmp_path, include_legacy=False)
    draws = load_draws(tmp_path)
    assert set(draws["partition"]) == {"smoke", "e1_fill", "c48_extra", "t200_diag"}
    assert not draws["candidate_id"].isin(ids["v48_bank"] + ids["t200_bank"]).any()
    diag = draws[draws["partition"] == "t200_diag"]
    assert diag["correct"].isna().all() and diag["grade_state"].isna().all()
    assert diag["vote_text"].notna().all()                       # parsed outputs are released for features
    dev = draws[draws["partition"] == "e1_fill"].sort_values("replicate")
    assert list(dev["correct"]) == [True, True, True] and list(dev["vote_exact_key"]) == ["7", "7", "7"]
    assert manifest["labelled_partitions"] == ["smoke", "e1_fill", "c48_extra"]
    assert set(manifest["sha256"]) == {"draws.parquet"}


def test_release_requires_verifying_seal(tmp_path):
    fakeroot.build(tmp_path)
    with pytest.raises(PermissionError):
        extract(tmp_path, include_legacy=False, release=["v48"])
    target = _seal(tmp_path)
    assert verify_seal(tmp_path) is not None
    extract(tmp_path, include_legacy=False, release=["v48"])
    draws = load_draws(tmp_path)
    v48 = draws[draws["partition"] == "v48_bank"]
    assert len(v48) == 3 and v48["correct"].notna().all()
    assert draws.loc[draws["partition"] == "t200_diag", "correct"].isna().all()
    target.write_bytes(b"changed")                                # a changed artifact un-verifies the seal
    assert verify_seal(tmp_path) is None
    with pytest.raises(PermissionError):
        extract(tmp_path, include_legacy=False, release=["t200"])


def test_blocked_ids_lift_only_with_seal(tmp_path):
    ids = fakeroot.build(tmp_path)
    shard = tmp_path / "artifacts" / "shards" / "mancer"
    blocked = blocked_node_ids(shard)
    assert blocked == frozenset(ids["v48_bank"] + ids["t200_bank"] + ids["t200_diag"])
    _seal(tmp_path)
    assert blocked_node_ids(shard) == frozenset()


def test_join_labels_refuses_without_seal_or_on_change(tmp_path):
    fakeroot.build(tmp_path)
    labels = pd.DataFrame({"item_id": ["aime/2026/1"], "y": [1]})
    with pytest.raises(SealError):
        join_labels(tmp_path, "sealed/v48/predictions.parquet", labels)
    _seal(tmp_path)
    joined = join_labels(tmp_path, "sealed/v48/predictions.parquet", labels)
    assert list(joined["y"]) == [1]


def test_tampered_draws_detected(tmp_path):
    fakeroot.build(tmp_path)
    extract(tmp_path, include_legacy=False)
    path = tmp_path / "analysis_inputs" / "draws.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[:1].to_parquet(path)
    with pytest.raises(ValueError):
        load_draws(tmp_path)
