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
    assert set(manifest["sha256"]) == {"draws.parquet", "pools.parquet", "classes.parquet", "readouts.parquet",
                                       "dev_training_readouts.parquet"}


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
    extract(tmp_path, include_legacy=False)                       # the release is sticky while the seal verifies
    assert (load_draws(tmp_path)["partition"] == "v48_bank").sum() == 3
    target.write_bytes(b"changed")                                # a changed artifact un-verifies the seal
    assert verify_seal(tmp_path) is None
    with pytest.raises(PermissionError):
        extract(tmp_path, include_legacy=False, release=["t200"])
    extract(tmp_path, include_legacy=False)                       # the fence is restored
    assert not (load_draws(tmp_path)["partition"] == "v48_bank").any()


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


def test_held_out_predictions_refuse_label_columns(tmp_path):
    from agent_allocation.passk.labels import LabelJoinRefused, write_held_out_predictions
    with pytest.raises(LabelJoinRefused):
        write_held_out_predictions(tmp_path / "p.parquet", pd.DataFrame({"item_id": ["a"], "correct": [True]}))
    digest = write_held_out_predictions(tmp_path / "p.parquet", pd.DataFrame({"item_id": ["a"], "p": [0.5]}))
    assert digest == hashlib.sha256((tmp_path / "p.parquet").read_bytes()).hexdigest()


def test_development_scopes_never_hold_held_out_members(tmp_path):
    public = tmp_path / "artifacts" / "manifests" / "public"
    public.mkdir(parents=True)
    (public / "aime.jsonl").write_text(json.dumps(fakeroot.MATH_TASK) + "\n")
    splits = {"panels": {"L96": ["aime/2026/1"], "C48": ["aime/2026/1"], "V48": ["aime/2026/1"], "T200": []}}
    (tmp_path / "artifacts" / "manifests" / "splits.json").write_text(json.dumps(splits))
    shard = tmp_path / "artifacts" / "shards" / "mancer"
    task = fakeroot.MATH_TASK
    for r in range(8):
        fakeroot.write_node(shard, "e1_fill", task, 2048, r, "\\boxed{%d}" % (r % 2), r % 2 == 1)
        fakeroot.write_node(shard, "c48_extra", task, 8192, r, "\\boxed{1}", True)
        fakeroot.write_node(shard, "v48_bank", task, 2048, r, "\\boxed{1}", True)
    _seal(tmp_path)
    from agent_allocation.passk.pairs import PairCache, Pair
    PairCache(tmp_path / "catalog" / "pair_verdicts.parquet").fill([Pair("aime/2026/1", "0", "1")])
    extract(tmp_path, include_legacy=False, release=["v48"])
    pools = pd.read_parquet(tmp_path / "analysis_inputs" / "pools.parquet")
    draws = load_draws(tmp_path)
    held = set(draws.loc[draws.partition == "v48_bank", "candidate_id"])
    assert len(held) == 8
    dev = pools[pools.panel.isin(["L96", "C48"])]
    assert len(dev) and {"8x2", "c48fit", "2x8"} <= set(dev.config_id)
    for members in dev.member_ids:
        assert not set(members) & held
    v48 = pools[pools.panel == "V48"]
    assert set(v48.config_id) >= {"8x2"} and all(set(m) <= held for m in v48.member_ids)
