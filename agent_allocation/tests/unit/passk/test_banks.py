"""WP-G P1: the frozen node banks of contract section 3."""
from __future__ import annotations

import json

import pytest

from agent_allocation.api.fake import fake_forced_endpoint_lock
from agent_allocation.config import decoding_config_hash, load_config
from agent_allocation.contracts import GenerationKind, load_record, EndpointLock
from agent_allocation.passk import banks
from agent_allocation.prompts.render import prompt_version
from tests.forced_cfg import PROJECT
from tests.unit.passk.wpg_fixtures import LEGACY_ROOT, fixture_splits, legacy_available

LEGACY_DECODING = "a641b002e3cb629e24c2488d7d171c8cfed86aeae98b0ecc63adc1f91893cc39"


def _cfg():
    return load_config(PROJECT / "configs" / "passk_cont_v1.json")


def _lock():
    source = LEGACY_ROOT / "shards" / "mancer" / "locks" / "openrouter_endpoints.json"
    if source.is_file():
        data = json.loads(source.read_text())
        return load_record(EndpointLock, data["locks"]["qwen/qwen3.8-27b"])
    return fake_forced_endpoint_lock("qwen/qwen3.8-27b", tag="mancer/fp8")


@pytest.fixture(scope="module")
def plan():
    splits, tasks = fixture_splits()
    graphs = banks.build_plan(_cfg(), splits, tasks, _lock(), {}, banks.PARTITIONS)
    return splits, graphs


def test_counts_match_the_contract(plan):
    _, graphs = plan
    assert banks.EXPECTED_COUNTS == {"smoke": 12, "e1_fill": 1056, "c48_extra": 816, "v48_bank": 1488,
                                     "t200_bank": 6200, "t200_diag": 1000}
    assert banks.node_counts(graphs) == banks.EXPECTED_COUNTS
    total = sum(banks.EXPECTED_COUNTS.values())
    assert banks.check_unique(graphs) == total == 10_560 + 12


def test_replicate_map_is_exactly_the_contract(plan):
    _, graphs = plan
    A = 1024
    bank = {16 * A: [0], 8 * A: [0, 1], 4 * A: list(range(4)), 2 * A: list(range(8)), A: list(range(16))}
    assert banks.replicate_map(graphs) == {
        "smoke": {A: [100]},
        "e1_fill": {2 * A: list(range(1, 8)), 4 * A: [1, 2, 3], 8 * A: [1]},
        "c48_extra": {4 * A: [4, 5, 6, 7], 8 * A: list(range(2, 8)), 16 * A: list(range(1, 8))},
        "v48_bank": bank, "t200_bank": bank, "t200_diag": {A: list(range(5))},
    }
    for gs in graphs.values():
        for g in gs:
            budgets = [n.reasoning_budget for n in g.nodes]
            assert budgets == sorted(budgets, reverse=True)
            for n in g.nodes:
                assert n.root_slot == n.draw_id
                assert n.kind is (GenerationKind.ROOT if n.reasoning_budget == A else GenerationKind.DEEP)
                assert (n.team_size, n.update_round, n.repetition_id, n.parent_ids) == (0, 0, 0, ())
                assert n.run_id == "openrouter_passk_cont_v1" and n.partition_id == g.partition_id
                assert n.prompt_version == prompt_version() == "prompts-v1-e998b556294c"
                assert n.decoding_config_hash == LEGACY_DECODING
                assert n.completion_headroom == 4096


def test_panels_per_partition(plan):
    splits, graphs = plan
    items = {p: [g.item_id for g in gs] for p, gs in graphs.items()}
    assert items["e1_fill"] == splits["panels"]["L96"]
    assert items["c48_extra"] == splits["panels"]["C48"] == items["v48_bank"]
    assert items["t200_bank"] == splits["panels"]["T200"] == items["t200_diag"]
    d32 = splits["panels"]["D32"]
    assert len(items["smoke"]) == 12
    for b in ("aime", "hmmt", "omni_math_hard", "gpqa_diamond"):
        assert [i for i in items["smoke"] if i.startswith(b + "/")] == [i for i in d32 if i.startswith(b + "/")][:3]


def test_decoding_hash_equals_forced_v2():
    assert decoding_config_hash(_cfg()) == LEGACY_DECODING
    legacy_cfg = load_config(PROJECT / "configs" / "forced27b.json")
    assert decoding_config_hash(legacy_cfg) == LEGACY_DECODING


@pytest.mark.skipif(not legacy_available(), reason="legacy run root not readable")
def test_no_new_node_id_equals_a_legacy_id(plan):
    _, graphs = plan
    legacy = banks.legacy_node_ids(LEGACY_ROOT)
    assert len(legacy) > 13_000
    new = {n.node_id for gs in graphs.values() for g in gs for n in g.nodes}
    assert not new & legacy


def test_waves_stage_order_and_item_integrity(plan):
    _, graphs = plan
    main = {p: gs for p, gs in graphs.items() if p != "smoke"}
    waves = banks.plan_waves(main, 16)
    assert all(1 <= len(w) <= 16 for w in waves)
    stages = [max(banks.stage_of(g.partition_id) for g in w) for w in waves]
    assert stages == sorted(stages) and set(stages) == {1, 2}
    # no wave mixes stage 1 and stage 2; stage-1 partitions in the order c48_extra, e1_fill, t200_diag
    for w in waves:
        assert len({banks.stage_of(g.partition_id) for g in w}) == 1
    firsts = []
    for w in waves:
        for g in w:
            if g.partition_id in banks.STAGE1 and g.partition_id not in firsts:
                firsts.append(g.partition_id)
    assert firsts == list(banks.STAGE1)
    # every (partition, item) graph exactly once; all its bank nodes in one graph
    keys = [(g.partition_id, g.item_id) for w in waves for g in w]
    assert len(keys) == len(set(keys)) == sum(len(gs) for gs in main.values())
    stage2 = [g for w in waves for g in w if banks.stage_of(g.partition_id) == 2]
    assert all(len(g.nodes) == 31 for g in stage2)
    # stage 2 is interleaved (both partitions appear within the first few waves) and seeded (deterministic)
    first_parts = {g.partition_id for g in stage2[:64]}
    assert first_parts == {"v48_bank", "t200_bank"}
    again = banks.plan_waves(main, 16)
    assert [[(g.partition_id, g.item_id) for g in w] for w in again] == [[(g.partition_id, g.item_id) for g in w]
                                                                          for w in waves]


def test_parse_partitions():
    assert banks.parse_partitions("e1_fill+c48_extra") == ["e1_fill", "c48_extra"]
    assert banks.parse_partitions(["smoke", "smoke+t200_diag"]) == ["smoke", "t200_diag"]
    with pytest.raises(banks.BankError):
        banks.parse_partitions("main")


def test_jobs_sha_matches_write_jobs(tmp_path, plan):
    from agent_allocation import live
    from agent_allocation.paths import RunPaths
    _, graphs = plan
    paths = RunPaths.for_run(tmp_path / "run", local_state_dir=tmp_path / "local")
    sha = live.write_jobs(paths, "e1_fill", graphs["e1_fill"])
    assert sha == banks.jobs_text_sha(graphs["e1_fill"])


def test_study_root_layout(tmp_path):
    from agent_allocation.paths import RunPaths
    shard = RunPaths.for_run(tmp_path / "passk" / "artifacts" / "shards" / "mancer", local_state_dir=tmp_path / "l")
    assert banks.artifacts_root(shard) == tmp_path / "passk" / "artifacts"
    assert banks.study_root(shard) == tmp_path / "passk"
    assert banks.splits_path(shard) == tmp_path / "passk" / "artifacts" / "manifests" / "splits.json"
