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
    assert not cp.sample_check(cfg, paths, {})["passed"]
