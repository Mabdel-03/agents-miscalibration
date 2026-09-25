"""WP-G: Gate 0 over a materialized fixture study (legacy L96/D32 + synthetic T200), adopted legacy locks, and the
passk-run guard against a changed frozen node list."""
from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from hashlib import sha256

import pytest

from agent_allocation.config import load_config
from agent_allocation.passk import banks, gates
from agent_allocation.passk.run import (
    PasskRunRefused, adopt_legacy_lock, assert_frozen, cmd_passk_run, plan_summary, require_gate0, shard_plan,
)
from agent_allocation.paths import RunPaths
from agent_allocation.shards import materialize_shard, root_assignment, shard_config
from tests.forced_cfg import PROJECT
from tests.unit.passk.wpg_fixtures import LEGACY_ROOT, fixture_splits, legacy_available, write_fixture_root

pytestmark = pytest.mark.skipif(not legacy_available(), reason="legacy run root not readable")


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gate0")
    base = load_config(PROJECT / "configs" / "passk_cont_v1.json")
    splits, tasks = fixture_splits(use_legacy=True)
    study = tmp / "passk_cont_v1"
    artifacts = study / "artifacts"
    write_fixture_root(artifacts, splits, tasks)
    root = RunPaths.for_run(artifacts, local_state_dir=tmp / "local_root")
    assignment = root_assignment(base, root)
    shards = {}
    for sid in ("mancer", "parasail"):
        materialize_shard(base, root, sid, assignment, local_state_dir=tmp / f"local_{sid}")
        cfg = shard_config(base, sid)
        paths = RunPaths.for_run(artifacts / "shards" / sid, local_state_dir=tmp / f"local_{sid}")
        out = adopt_legacy_lock(cfg, paths)
        assert out["adopted"] and adopt_legacy_lock(cfg, paths)["adopted"] is False
        shards[sid] = (cfg, paths)
    (study / "gates").mkdir(parents=True, exist_ok=True)
    (study / "gates" / "gate_s.json").write_text(json.dumps(
        {"passed": True, "splits_sha256": banks.splits_sha256(artifacts)}))
    return base, study, shards


def test_gate0_offline_criteria_pass(study):
    base, root, shards = study
    e0 = {"passed": True, "checks": {"forced_budget_check": {"passed": True}}}
    result = gates.evaluate_gate0(base, root, e0=e0, local_state_root=root.parent / "g0local")
    failing = {k: v for k, v in result["criteria"].items() if v.get("passed") is False}
    assert result["passed"], failing
    assert result["complete"] is False                       # key + tests not evaluated offline
    assert result["criteria"]["planned_counts"]["counts"] == banks.EXPECTED_COUNTS
    assert result["criteria"]["think_parity"]["checked"] == 128
    forecast = result["criteria"]["forecast"]
    assert Decimal(forecast["total_usd"]) <= Decimal("115")
    assert set(result["jobs_sha256"]) == {"mancer", "parasail"}
    target = gates.write_gate0(root, result)
    assert json.loads(target.read_text())["jobs_sha256"] == result["jobs_sha256"]
    # the key check: slices 87 + 63 = 150 need limit_remaining >= 153
    assert gates.key_check(base, {"key": {"limit_remaining": 218.68}})["passed"] is True
    assert gates.key_check(base, {"key": {"limit_remaining": 150}})["passed"] is False


def test_passk_run_plan_matches_gate0_and_refuses_a_changed_list(study, capsys):
    base, root, shards = study
    cfg, paths = shards["mancer"]
    if not gates.gate0_path(root).is_file():
        gates.write_gate0(root, gates.evaluate_gate0(base, root, e0={"passed": True},
                                                     local_state_root=root.parent / "g0local"))
    graphs = shard_plan(cfg, paths, banks.PARTITIONS)
    summary = plan_summary(cfg, paths, graphs)
    assert all(summary["gate0_match"].values())
    assert_frozen(cfg, paths, graphs)
    # a different headroom (any GenerationSpec change) changes the frozen list: refused before dispatch
    changed = dict(cfg)
    changed["reasoning"] = {**cfg["reasoning"], "text_completion_headroom": 4097}
    with pytest.raises(PasskRunRefused):
        assert_frozen(changed, paths, shard_plan(changed, paths, ["e1_fill"]))
    rc = cmd_passk_run(["--config", str(PROJECT / "configs" / "passk_cont_v1.json"), "--run-dir",
                        str(paths.run_dir), "--partitions", "smoke+e1_fill", "--plan-only"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["shard"] == "mancer" and set(out["nodes"]) == {"smoke", "e1_fill"}
    # --plan-only wrote nothing
    assert not paths.jobs("e1_fill").exists()


def test_legacy_lock_tag_mismatch_refused(study, tmp_path):
    base, _root, shards = study
    cfg, _paths = shards["mancer"]
    wrong = dict(cfg)
    wrong["provider"] = {**cfg["provider"], "endpoint_tag": "parasail/fp8"}
    fresh = RunPaths.for_run(tmp_path / "shards" / "mancer", local_state_dir=tmp_path / "l")
    with pytest.raises(PasskRunRefused):
        adopt_legacy_lock(wrong, fresh)


def test_paid_dispatch_requires_a_passed_gate0(study, tmp_path):
    base, root, shards = study
    cfg, paths = shards["parasail"]
    if not gates.gate0_path(root).is_file():
        gates.write_gate0(root, gates.evaluate_gate0(base, root, e0={"passed": True},
                                                     local_state_root=root.parent / "g0local"))
    record = json.loads(gates.gate0_path(root).read_text())
    try:
        with pytest.raises(PasskRunRefused):                  # offline record: key + tests not evaluated
            require_gate0(cfg, paths, ["smoke"])
        gates.gate0_path(root).write_text(json.dumps({**record, "complete": True}))
        assert require_gate0(cfg, paths, list(banks.PARTITIONS))["passed"]
        gates.gate0_path(root).write_text(json.dumps({**record, "complete": True, "passed": False}))
        with pytest.raises(PasskRunRefused):
            require_gate0(cfg, paths, ["smoke"])
        gates.gate0_path(root).write_text(json.dumps({**record, "complete": True,
                                                      "jobs_sha256": {"mancer": record["jobs_sha256"]["mancer"]}}))
        with pytest.raises(PasskRunRefused):
            require_gate0(cfg, paths, ["smoke"])
        with pytest.raises(PasskRunRefused):                  # the paid path checks before any network call
            asyncio.run(__import__("agent_allocation.passk.run", fromlist=["run_partitions"]).run_partitions(
                cfg, paths, ["smoke"], max_usd=Decimal("1"), allow_paid=True))
    finally:
        gates.gate0_path(root).write_text(json.dumps(record))
