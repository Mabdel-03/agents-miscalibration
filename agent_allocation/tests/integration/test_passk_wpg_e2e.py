"""WP-G P5: passk_cont_v1 mocked end to end on the fake transport (no network, no key, no paid call).

16 items across every partition (L96 4, of which C48 = V48 2; T200 8; D32 4), a two-shard plan through the real
``shard-plan`` materialization (T200 1:1 from splits.json), then per shard:

* parasail: ``smoke`` then stage 1 + stage 2 in-process through :func:`passk.run.run_partitions` with a client
  wrapper that records the pick order (``validate_request`` runs synchronously in admission order): within each
  wave the first admitted node is DEEP or ROOT slot 0, and every stage-1 first admission precedes every stage-2 one;
* mancer: ``allocation passk-run --rehearsal`` in a subprocess, SIGKILLed mid-run, then resumed: every node ends
  parsed, no node that was parsed before the resume is sent again, and no node has two HTTP-200 bodies;
* the held-out fence: ``grade`` passes (the exact scope ``cmd_grade`` computes per pass) grade smoke / e1_fill /
  c48_extra and never v48_bank / t200_*; once a valid ``sealed/SEAL_predictions.json`` exists they are graded.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import pytest

from agent_allocation import live
from agent_allocation.api.fake import fake_forced_endpoint_lock, fake_openrouter_transport
from agent_allocation.config import load_config, validate_config
from agent_allocation.contracts import (
    EvalKind, GenerationKind, PrivateEvalSpec, canonical_hash, dump_record,
)
from agent_allocation.grading.worker import _grade_scope, grade_pass
from agent_allocation.grading.code import profile as profile_mod
from agent_allocation.passk import banks
from agent_allocation.passk.run import run_partitions, shard_plan
from agent_allocation.paths import RunPaths
from agent_allocation.shards import materialize_shard, root_assignment, shard_config
from tests.forced_cfg import PROJECT
from tests.integration.dispatch_harness import fast_config
from tests.unit.passk.wpg_fixtures import fixture_splits, write_fixture_root

pytestmark = [pytest.mark.timeout(900)]
MODEL = "qwen/qwen3.8-27b"
MAIN_PARTITIONS = ["c48_extra", "e1_fill", "t200_diag", "v48_bank", "t200_bank"]


def _private(task) -> PrivateEvalSpec:
    gold = "A" if task.benchmark == "gpqa_diamond" else "42"
    kind = EvalKind.OPTION_EXACT if task.benchmark == "gpqa_diamond" else EvalKind.MATH_VERIFY
    record = {"item_id": task.item_id, "gold": gold}
    return PrivateEvalSpec(item_id=task.item_id, benchmark=task.benchmark, source_revision="fixture", eval_kind=kind,
                           gold_answer=gold, private_record_ref=None, private_hash=canonical_hash(record),
                           allowed_choices=task.allowed_choices)


class Recorder:
    """Forwards to the runtime's client; records node ids in admission (validate) and send (generate) order."""

    def __init__(self, inner):
        self.inner = inner
        self.validated: list[str] = []
        self.sent: list[str] = []

    def validate_request(self, spec, messages, lock, sampling):
        if spec.node_id not in self.validated:
            self.validated.append(spec.node_id)
        return self.inner.validate_request(spec, messages, lock, sampling)

    async def generate(self, spec, messages, **kwargs):
        self.sent.append(spec.node_id)
        return await self.inner.generate(spec, messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _study(tmp_path):
    base = fast_config(load_config(PROJECT / "configs" / "passk_cont_v1.json"))
    base.pop("scheduler_by_shard")          # live per-provider limits would override the fast test scheduler
    assert validate_config(base) == [], validate_config(base)
    splits, tasks = fixture_splits(use_legacy=False, t200_quota={"aime": 2, "hmmt": 2, "omni_math_hard": 2,
                                                                  "gpqa_diamond": 2},
                                   l96_per=1, d32_per=1, c48_per=1)
    # C48 = 2 items (aime, gpqa), V48 = C48; alternate T200 shards explicitly.
    splits["panels"]["C48"] = [i for i in splits["panels"]["L96"] if i.startswith(("aime/", "gpqa_diamond/"))]
    splits["panels"]["V48"] = list(splits["panels"]["C48"])
    items = {*splits["panels"]["L96"], *splits["panels"]["T200"], *splits["panels"]["D32"]}
    assert len(items) == 16
    privates = {i: _private(tasks[i]) for i in items}
    study = tmp_path / "passk_cont_v1"
    artifacts = study / "artifacts"
    write_fixture_root(artifacts, splits, tasks, privates)
    root = RunPaths.for_run(artifacts, local_state_dir=tmp_path / "local_root")
    assignment = root_assignment(base, root)
    assert assignment["panel_overrides"]["reserve"]["source"] == "splits.json:assignment_t200"
    shards = {}
    for shard in ("mancer", "parasail"):
        info = materialize_shard(base, root, shard, assignment, local_state_dir=tmp_path / f"local_{shard}")
        assert info["problems"] == []
        derived = shard_config(base, shard)
        paths = RunPaths.for_run(artifacts / "shards" / shard, local_state_dir=tmp_path / f"local_{shard}")
        lock = fake_forced_endpoint_lock(MODEL, tag=derived["provider"]["endpoint_tag"])
        paths.endpoint_lock.parent.mkdir(parents=True, exist_ok=True)
        paths.endpoint_lock.write_text(json.dumps({"locks": {MODEL: dump_record(lock)}}, sort_keys=True))
        shards[shard] = (derived, paths)
    return study, splits, shards


def _profile(cfg):
    return profile_mod.profile_from_config(cfg)


def _nodes(graphs):
    return {n.node_id: n for gs in graphs.values() for g in gs for n in g.nodes}


def _grade(cfg, paths):
    skip, only = _grade_scope(paths, cfg)
    return grade_pass(paths, _profile(cfg), resume=True, workers=0, skip=skip, only=only, log=lambda *a: None)


def test_passk_e2e_all_partitions_kill_resume_fence_and_order(tmp_path):
    study, splits, shards = _study(tmp_path)

    # ---- parasail: in-process, recorded order ----------------------------------------------------------------
    cfg, paths = shards["parasail"]
    recorder_box = {}

    def wrap(client):
        recorder_box["r"] = Recorder(client)
        return recorder_box["r"]

    smoke = asyncio.run(run_partitions(cfg, paths, ["smoke"], max_usd=Decimal("20"),
                                       transport=fake_openrouter_transport(), allow_paid=False))
    assert smoke["stopped_early"] is None, smoke
    out = asyncio.run(run_partitions(cfg, paths, MAIN_PARTITIONS, max_usd=Decimal("20"),
                                     transport=fake_openrouter_transport(), allow_paid=False, wrap_client=wrap))
    assert out["stopped_early"] is None, out
    graphs = shard_plan(cfg, paths, MAIN_PARTITIONS)
    nodes = _nodes(graphs)
    assert all(paths.parsed(n).exists() for n in nodes)
    rec = recorder_box["r"]
    assert set(rec.validated) == set(nodes)
    position = {n: k for k, n in enumerate(rec.validated)}
    waves = banks.plan_waves(graphs, int(cfg["scheduler"]["wave_items"]))
    assert len(waves) >= 2
    for wave in waves:
        ids = [n.node_id for g in wave for n in g.nodes]
        first = nodes[min(ids, key=position.__getitem__)]
        assert first.kind is GenerationKind.DEEP or (first.kind is GenerationKind.ROOT and first.root_slot == 0), first
    stage1 = [position[n] for n, s in nodes.items() if banks.stage_of(s.partition_id) == 1]
    stage2 = [position[n] for n, s in nodes.items() if banks.stage_of(s.partition_id) == 2]
    assert stage1 and stage2 and max(stage1) < min(stage2)
    # the frozen jobs files hold exactly the planned lists
    for partition, gs in graphs.items():
        assert banks.jobs_text_sha(gs) == sha256(paths.jobs(partition).read_bytes()).hexdigest()

    # ---- mancer: subprocess, SIGKILL, resume -------------------------------------------------------------------
    cfg_m, paths_m = shards["mancer"]
    shard_cfg_path = paths_m.state_dir / "shard_config.json"
    assert json.loads(shard_cfg_path.read_text())["shard"]["id"] == "mancer"
    env = {**os.environ, "ALLOCATION_LOCAL_STATE": str(paths_m.local_state_dir), "PYTHONNOUSERSITE": "1"}
    exe = str(Path(sys.executable).parent / "allocation")
    args = [exe, "passk-run", "--config", str(shard_cfg_path), "--run-dir", str(paths_m.run_dir),
            "--partitions", "+".join(["smoke", *MAIN_PARTITIONS]), "--rehearsal", "--max-usd", "20", "--resume"]
    old_env = dict(os.environ)
    os.environ.update(env)
    try:
        killed = live.run_subprocess_kill_test(args, paths_m, kill_after_parsed=40, timeout_s=600)
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    graphs_m = shard_plan(cfg_m, paths_m, ["smoke", *MAIN_PARTITIONS])
    nodes_m = _nodes(graphs_m)
    parsed_before = {n for n in nodes_m if paths_m.parsed(n).exists()}
    assert killed["killed"], killed                          # the kill landed mid-run
    assert 40 <= len(parsed_before) < len(nodes_m)
    resume_box = {}

    def wrap_m(client):
        resume_box["r"] = Recorder(client)
        return resume_box["r"]

    resumed = asyncio.run(run_partitions(cfg_m, paths_m, ["smoke", *MAIN_PARTITIONS], max_usd=Decimal("20"),
                                         transport=fake_openrouter_transport(), allow_paid=False, wrap_client=wrap_m))
    assert resumed["stopped_early"] is None, resumed
    assert all(paths_m.parsed(n).exists() for n in nodes_m)
    assert not set(resume_box["r"].sent) & parsed_before
    assert live.duplicate_successful_dispatches(paths_m, set(nodes_m)) == 0
    assert set(nodes_m).isdisjoint(nodes)                      # disjoint shards, disjoint ids

    # ---- fence: held-out never graded until a verified seal ------------------------------------------------------
    heldout = {"v48_bank", "t200_bank", "t200_diag"}
    for shard_cfg, shard_paths, shard_nodes in ((cfg, paths, {**nodes, **_nodes(shard_plan(cfg, paths, ["smoke"]))}),
                                                (cfg_m, paths_m, nodes_m)):
        for _ in range(2):                                     # two --follow style passes
            _grade(shard_cfg, shard_paths)
        for node_id, spec in shard_nodes.items():
            graded = shard_paths.grade(node_id).exists()
            if spec.partition_id in heldout:
                assert not graded, (spec.partition_id, node_id)
            else:
                assert graded or not json.loads(shard_paths.parsed(node_id).read_text()).get("candidate"), node_id
    art = study / "sealed" / "t200" / "predictions.parquet"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"sealed predictions")
    (study / "sealed" / "SEAL_predictions.json").write_text(json.dumps(
        {"code_commit": "test", "artifacts": {"sealed/t200/predictions.parquet": sha256(art.read_bytes()).hexdigest()}}))
    for shard_cfg, shard_paths, shard_nodes in ((cfg, paths, nodes), (cfg_m, paths_m, nodes_m)):
        _grade(shard_cfg, shard_paths)
        released = [n for n, s in shard_nodes.items() if s.partition_id in heldout
                    and json.loads(shard_paths.parsed(n).read_text()).get("candidate")]
        assert released and all(shard_paths.grade(n).exists() for n in released)

    # ---- the in-run gates over the same shard dirs ---------------------------------------------------------------
    from agent_allocation.passk import checkpoints as cp
    smoke_gate = cp.gate1_smoke(cfg, paths)
    assert smoke_gate["criteria"]["all_terminal"]["passed"], smoke_gate
    assert smoke_gate["criteria"]["exact_budgets"]["passed"], smoke_gate
    assert smoke_gate["criteria"]["zero_cache_hits"]["passed"]
    assert smoke_gate["complete"] is False                     # permaslug needs --fetch-metadata (live)
    g2 = cp.gate2(cfg_m, paths_m, force=True)
    assert g2["due"] and g2["criteria"]["exact_budgets"]["passed"] and g2["criteria"]["provider_drift"]["passed"], g2
    assert not cp.gate2(cfg_m, paths_m, last_checked=g2["settled_nodes"])["due"]
    g3 = cp.gate3(cfg_m, paths_m)
    assert g3["passed"] and not paths_m.drain_sentinel.exists(), g3
    g4 = cp.gate4(cfg_m, paths_m, check_duplicates=False)
    assert g4["criteria"]["completion"]["passed"] and g4["criteria"]["dev_grades"]["passed"], g4
    assert cp.graded(cfg_m, paths_m)["passed"] and cp.graded(cfg, paths)["passed"]   # the seal released everything
