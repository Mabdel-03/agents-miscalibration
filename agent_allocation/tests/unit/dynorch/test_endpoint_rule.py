"""WP-D: the provider pinning rule (plan §4) on pass / one-fails / both-fail snapshots."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agent_allocation.api.fake import fake_forced_endpoints_listing
from agent_allocation.config import load_config
from agent_allocation.dynorch import endpoint_rule as er

CFG_PATH = Path(__file__).resolve().parents[3] / "configs" / "dynorch_v1.json"
MODEL = "qwen/qwen3.8-27b"
ASSIGNMENT = {"aime/2025/1": "mancer", "aime/2025/2": "parasail", "gpqa_diamond/r1": "mancer", "hmmt/2025/3": "mancer"}


@pytest.fixture(scope="module")
def cfg():
    return load_config(CFG_PATH)


def snapshot(**patch):
    """The fake Qwen listing (parasail/fp8, mancer/fp8, ... all status 0, uptime 99.5) with per-tag overrides."""
    snap = fake_forced_endpoints_listing(MODEL)
    for endpoint in snap["data"]["endpoints"]:
        for key, value in patch.get(endpoint["tag"], {}).items():
            endpoint[key] = value
    return snap


def test_all_pass_keeps_the_forced_v2_shards(cfg):
    decision = er.decide(snapshot(), cfg)
    assert decision.action == er.KEEP
    assert decision.shard_to_tag == {"mancer": "mancer/fp8", "parasail": "parasail/fp8"}
    rows = er.provider_map(decision, ASSIGNMENT, er.shard_tags(cfg))
    assert rows["aime/2025/1"] == {"forced_v2_shard": "mancer", "shard_id": "mancer", "endpoint_tag": "mancer/fp8"}
    assert rows["aime/2025/2"]["endpoint_tag"] == "parasail/fp8"
    assert er.counts_by_tag(rows) == {"mancer/fp8": 3, "parasail/fp8": 1}


@pytest.mark.parametrize("bad", [{"status": -2}, {"uptime_last_30m": 93.7}, {"supported_parameters": ["max_tokens", "temperature", "top_p"]}])
def test_one_failing_shard_moves_every_item_to_the_other(cfg, bad):
    decision = er.decide(snapshot(**{"mancer/fp8": bad}), cfg)
    assert decision.action == er.SWITCH, decision
    assert decision.shard_to_tag == {"mancer": "parasail/fp8", "parasail": "parasail/fp8"}
    mancer = next(v for v in decision.verdicts if v.shard_id == "mancer")
    assert not mancer.passed and mancer.problems
    rows = er.provider_map(decision, ASSIGNMENT, er.shard_tags(cfg))
    assert {r["endpoint_tag"] for r in rows.values()} == {"parasail/fp8"}
    assert {r["shard_id"] for r in rows.values()} == {"parasail"}
    assert rows["aime/2025/1"]["forced_v2_shard"] == "mancer"      # the covariate survives the switch


def test_uptime_exactly_at_threshold_passes(cfg):
    decision = er.decide(snapshot(**{"mancer/fp8": {"uptime_last_30m": 95.0}}), cfg)
    assert decision.action == er.KEEP


def test_both_failing_stops(cfg):
    decision = er.decide(snapshot(**{"mancer/fp8": {"status": -2}, "parasail/fp8": {"uptime_last_30m": 10}}), cfg)
    assert decision.action == er.STOP and decision.shard_to_tag == {}
    with pytest.raises(er.EndpointRuleError):
        er.provider_map(decision, ASSIGNMENT, er.shard_tags(cfg))
    with pytest.raises(er.EndpointRuleError):
        er.lock_payload(decision, snapshot(), cfg)


def test_unlisted_tag_fails_and_malformed_snapshot_raises(cfg):
    snap = snapshot()
    snap["data"]["endpoints"] = [e for e in snap["data"]["endpoints"] if e["tag"] != "mancer/fp8"]
    decision = er.decide(snap, cfg)
    assert decision.action == er.SWITCH
    assert not next(v for v in decision.verdicts if v.shard_id == "mancer").listed
    with pytest.raises(er.EndpointRuleError):
        er.decide({"data": {"id": "other/model", "endpoints": []}}, cfg)
    with pytest.raises(er.EndpointRuleError):
        er.decide({"nope": 1}, cfg)


def test_lock_payload_uses_select_standard_endpoint(cfg):
    decision = er.decide(snapshot(), cfg)
    payload = er.lock_payload(decision, snapshot(), cfg, fetched_at="20260923T000000Z")
    assert payload["layout"] == er.LOCK_LAYOUT and set(payload["locks"]) == {"mancer", "parasail"}
    assert payload["locks"]["mancer"]["provider_tag"] == "mancer/fp8"
    assert payload["locks"]["mancer"]["__record__"] == "EndpointLock"
    assert payload["locks"]["parasail"]["internal_reasoning_price_per_token"] == "0"
    switched = er.decide(snapshot(**{"mancer/fp8": {"status": -1}}), cfg)
    payload = er.lock_payload(switched, snapshot(**{"mancer/fp8": {"status": -1}}), cfg)
    assert set(payload["locks"]) == {"parasail"}


def test_freeze_writes_once_and_refuses_a_different_decision(cfg, tmp_path):
    raw = json.dumps(snapshot()).encode()
    decision = er.freeze(tmp_path, snapshot(), raw, cfg, assignment=ASSIGNMENT, fetched_at="20260923T000000Z")
    assert decision.action == er.KEEP
    locks = er.read_locks(tmp_path)
    assert locks["mancer"].provider_tag == "mancer/fp8" and locks["parasail"].provider_name == "Parasail"
    assert er.read_decision(tmp_path)["action"] == er.KEEP
    items = json.loads((tmp_path / "locks" / "provider_map.json").read_text())["items"]
    assert items["aime/2025/2"]["endpoint_tag"] == "parasail/fp8"
    assert (tmp_path / "locks" / "snapshots").is_dir()
    # identical re-freeze is a no-op; a different decision is refused (never re-evaluated mid-run)
    assert er.freeze(tmp_path, snapshot(), raw, cfg, assignment=ASSIGNMENT, fetched_at="20260923T000001Z").action == er.KEEP
    degraded = snapshot(**{"mancer/fp8": {"status": -2}})
    with pytest.raises(er.EndpointRuleError):
        er.freeze(tmp_path, degraded, json.dumps(degraded).encode(), cfg, assignment=ASSIGNMENT, fetched_at="20260923T000002Z")


def test_freeze_on_stop_writes_only_the_decision(cfg, tmp_path):
    snap = snapshot(**{"mancer/fp8": {"status": -2}, "parasail/fp8": {"status": -2}})
    decision = er.freeze(tmp_path, snap, json.dumps(snap).encode(), cfg, assignment=ASSIGNMENT, fetched_at="x")
    assert decision.action == er.STOP
    assert (tmp_path / "locks" / "provider_decision.json").exists()
    assert not (tmp_path / "locks" / "openrouter_endpoints.json").exists()
    assert not (tmp_path / "locks" / "provider_map.json").exists()


def test_real_base_assignment_shape(cfg):
    base = Path(cfg["base_run_root"])
    if not (base / "shards" / "assignment.json").exists():
        pytest.skip("base run root not mounted")
    assignment = er.load_assignment(base)
    panels = json.loads((base / "manifests" / "panels.json").read_text())
    main = [i for panel in panels.values() for i in panel["main_item_ids"]]
    assert len(main) == 96 and set(main) <= set(assignment)
    assert sum(1 for i in main if assignment[i] == "mancer") == 58       # plan §4: mancer 58 / parasail 38
