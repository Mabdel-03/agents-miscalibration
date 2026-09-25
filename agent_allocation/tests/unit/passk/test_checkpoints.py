"""WP-G: checkpoint helpers that the e2e does not reach (sampling, attempt classification)."""
from __future__ import annotations

import json

from agent_allocation.api.fake import fake_forced_endpoint_lock
from agent_allocation.config import load_config
from agent_allocation.contracts import dump_record
from agent_allocation.passk import checkpoints as cp
from agent_allocation.paths import RunPaths
from agent_allocation.shards import shard_config
from tests.forced_cfg import PROJECT


def test_classify_detail():
    assert cp.classify_detail("forced_think_closed: </think> in THINK output (10/1024 tokens)") == "closed"
    assert cp.classify_detail("forced_think_short: 900/1024 tokens, finish_reason=stop (x)") == "closed"
    assert cp.classify_detail("forced_think_short: 900/1024 tokens, finish_reason=None (x)") == "stream_cut"
    assert cp.classify_detail("http 429") is None and cp.classify_detail(None) is None


def test_sample_check_permaslug_and_pin(tmp_path):
    cfg = shard_config(load_config(PROJECT / "configs" / "passk_cont_v1.json"), "mancer")
    paths = RunPaths.for_run(tmp_path / "shards" / "mancer", local_state_dir=tmp_path / "l")
    lock = fake_forced_endpoint_lock("qwen/qwen3.8-27b", tag="mancer/fp8")
    paths.endpoint_lock.parent.mkdir(parents=True)
    paths.endpoint_lock.write_text(json.dumps({"locks": {"qwen/qwen3.8-27b": dump_record(lock)}}))
    good = {"model": cp.EXPECTED_PERMASLUG, "provider_name": lock.provider_name,
            "provider_responses": [{"model_permaslug": cp.EXPECTED_PERMASLUG}]}
    assert cp.sample_check(cfg, paths, {"g1": good})["passed"]
    drifted = {**good, "provider_responses": [{"model_permaslug": "qwen/qwen3.8-27b-20261001"}]}
    assert not cp.sample_check(cfg, paths, {"g1": good, "g2": drifted})["passed"]
    assert not cp.sample_check(cfg, paths, {"g1": {**good, "provider_name": "Other"}})["passed"]
    empty = cp.sample_check(cfg, paths, {})
    assert not empty["passed"] and not empty["drift"] and not empty["conclusive"]


def test_sample_unavailable_generation_is_not_drift(tmp_path):
    """ops_budget review: '/generation not available yet' (404 -> absent, None, empty record) is retried, never drift."""
    cfg = shard_config(load_config(PROJECT / "configs" / "passk_cont_v1.json"), "mancer")
    paths = RunPaths.for_run(tmp_path / "shards" / "mancer", local_state_dir=tmp_path / "l")
    lock = fake_forced_endpoint_lock("qwen/qwen3.8-27b", tag="mancer/fp8")
    paths.endpoint_lock.parent.mkdir(parents=True)
    paths.endpoint_lock.write_text(json.dumps({"locks": {"qwen/qwen3.8-27b": dump_record(lock)}}))
    good = {"model": cp.EXPECTED_PERMASLUG, "provider_name": lock.provider_name,
            "provider_responses": [{"model_permaslug": cp.EXPECTED_PERMASLUG}]}
    none_yet = cp.sample_check(cfg, paths, {"g2": None, "g3": {}}, requested=["g1", "g2", "g3"])
    assert not none_yet["drift"] and not none_yet["conclusive"] and none_yet["unavailable"] == ["g1", "g2", "g3"]
    mixed = cp.sample_check(cfg, paths, {"g1": good}, requested=["g1", "g2"])
    assert mixed["passed"] and not mixed["drift"] and mixed["unavailable"] == ["g2"]
    other = cp.sample_check(cfg, paths, {"g1": {**good, "provider_name": "Other"}}, requested=["g1", "g2"])
    assert other["drift"] and not other["passed"]


def test_sample_cli_inconclusive_returns_4_and_retries_pending(tmp_path, monkeypatch):
    import agent_allocation.live as live
    study = tmp_path / "study"
    paths = cp._shard_paths(study, "mancer")
    lock = fake_forced_endpoint_lock("qwen/qwen3.8-27b", tag="mancer/fp8")
    paths.endpoint_lock.parent.mkdir(parents=True, exist_ok=True)
    paths.endpoint_lock.write_text(json.dumps({"locks": {"qwen/qwen3.8-27b": dump_record(lock)}}))
    parsed = paths.run_dir / "parsed" / "n1"
    parsed.mkdir(parents=True)
    import os
    for i in range(2):
        f = parsed / f"r{i}.json"
        f.write_text(json.dumps({"result": {"generation_id": f"gen-{i}"}}))
        os.utime(f, (1, 1))                                     # old enough to be indexed
    fresh = parsed / "fresh.json"
    fresh.write_text(json.dumps({"result": {"generation_id": "gen-fresh"}}))   # too young: not sampled
    seen = []

    async def fake_fetch(_paths, ids, **_kw):
        seen.append(list(ids))
        return {}

    monkeypatch.setattr(live, "fetch_metadata", fake_fetch)
    argv = ["sample", "--config", str(PROJECT / "configs" / "passk_cont_v1.json"), "--study-root", str(study),
            "--shard", "mancer"]
    assert cp.main(argv) == 4
    assert seen[-1] == ["gen-0", "gen-1"]
    assert not paths.drain_sentinel.exists()
    rec = json.loads((study / "gates" / "gate3_sample_mancer.json").read_text())
    assert rec["unavailable"] == ["gen-0", "gen-1"]
    good = {"model": cp.EXPECTED_PERMASLUG, "provider_name": lock.provider_name}

    async def fetch_one(_paths, ids, **_kw):
        seen.append(list(ids))
        return {"gen-0": good}

    monkeypatch.setattr(live, "fetch_metadata", fetch_one)
    assert cp.main(argv) == 0
    assert seen[-1][:2] == ["gen-0", "gen-1"]                  # the pending ids are retried first


def test_gate4_per_split(tmp_path, monkeypatch):
    """integration_spec review: V48 is released once v48_bank passes Gate 4, T200 once t200_bank passes."""
    cfg = shard_config(load_config(PROJECT / "configs" / "passk_cont_v1.json"), "mancer")
    paths = RunPaths.for_run(tmp_path / "shards" / "mancer", local_state_dir=tmp_path / "l")

    def rows(ok_t200: bool):
        out = []
        for part in cp.MAIN_PARTITIONS:
            for i in range(100):
                good = ok_t200 or part != "t200_bank" or i < 90
                out.append({"node_id": f"{part}-{i}", "partition": part, "parsed": True, "succeeded": good,
                            "has_candidate": False})
        return out

    monkeypatch.setattr(cp, "node_rows", lambda _paths, _parts: rows(False))
    g4 = cp.gate4(cfg, paths, check_duplicates=False)
    assert not g4["passed"] and g4["splits"]["v48"]["passed"] and not g4["splits"]["t200"]["passed"]
    monkeypatch.setattr(cp, "node_rows", lambda _paths, _parts: rows(True))
    g4 = cp.gate4(cfg, paths, check_duplicates=False)
    assert g4["passed"] and g4["splits"]["v48"]["passed"] and g4["splits"]["t200"]["passed"]


def test_gate1_smoke_requires_dev_grades(tmp_path, monkeypatch):
    """ops_budget review: Gate 1 runs the smoke through the dev grader, so an ungraded smoke candidate fails it."""
    cfg = shard_config(load_config(PROJECT / "configs" / "passk_cont_v1.json"), "mancer")
    paths = RunPaths.for_run(tmp_path / "shards" / "mancer", local_state_dir=tmp_path / "l")
    lock = fake_forced_endpoint_lock("qwen/qwen3.8-27b", tag="mancer/fp8")
    paths.endpoint_lock.parent.mkdir(parents=True)
    paths.endpoint_lock.write_text(json.dumps({"locks": {"qwen/qwen3.8-27b": dump_record(lock)}}))
    row = {"node_id": "smoke-1", "partition": "smoke", "parsed": True, "succeeded": True, "exact": True,
           "has_candidate": True, "model": cfg["primary_model"], "provider": lock.provider_name}
    monkeypatch.setattr(cp, "node_rows", lambda _paths, _parts: [row])
    assert not cp.gate1_smoke(cfg, paths)["criteria"]["smoke_graded"]["passed"]
    grade = paths.grade("smoke-1")
    grade.parent.mkdir(parents=True, exist_ok=True)
    grade.write_text("{}")
    assert cp.gate1_smoke(cfg, paths)["criteria"]["smoke_graded"]["passed"]
