"""WP-3: provider shards -- deterministic, disjoint, balanced item assignment; derived configs; shard run dirs."""
from __future__ import annotations

import copy
import json
import random
from collections import Counter
from decimal import Decimal

import pytest

from agent_allocation.config import validate_config
from agent_allocation.contracts import BENCHMARKS, OutputKind, PanelDraw, PrivateEvalSpec, EvalKind, PublicTask, Wave, dump_record
from agent_allocation.paths import RunPaths
from agent_allocation.shards import (
    ShardError, base_config, filter_graph_waves, items_of, materialize_shard, resolve_shard, root_assignment,
    shard_assignment, shard_budget_table, shard_config, shard_filter,
)
from tests.forced_cfg import three_shard_config, forced_test_config


@pytest.fixture()
def cfg():
    return three_shard_config()


def _items(per_bench: int, benches=("aime", "hmmt", "omni_math_hard", "gpqa_diamond")) -> list[tuple[str, str]]:
    return [(f"{b}/{i:03d}", b) for b in benches for i in range(per_bench)]


def test_assignment_is_deterministic_and_order_independent(cfg):
    items = _items(24)
    first = shard_assignment(cfg, items)
    shuffled = list(items)
    random.Random(7).shuffle(shuffled)
    assert shard_assignment(cfg, shuffled) == first
    assert shard_assignment(copy.deepcopy(cfg), items) == first


def test_assignment_covers_every_item_disjointly(cfg):
    items = _items(24)
    out = shard_assignment(cfg, items)
    assert set(out) == {i for i, _ in items}                       # covers all, each exactly once (a dict)
    parts = [items_of(out, s) for s in ("parasail", "mancer", "dekallm")]
    assert sum(len(p) for p in parts) == len(items)
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])


def test_assignment_is_benchmark_balanced(cfg):
    out = shard_assignment(cfg, _items(24))
    per = Counter((item.split("/")[0], shard) for item, shard in out.items())
    assert set(per.values()) == {8}                                # 24 per benchmark / 3 shards
    uneven = shard_assignment(cfg, _items(8))                      # dev panel: 8 per benchmark
    per_bench = Counter((i.split("/")[0], s) for i, s in uneven.items())
    for bench in ("aime", "hmmt", "omni_math_hard", "gpqa_diamond"):
        counts = [per_bench[(bench, s)] for s in ("parasail", "mancer", "dekallm")]
        assert max(counts) - min(counts) <= 1 and sum(counts) == 8
    totals = Counter(uneven.values())
    assert max(totals.values()) - min(totals.values()) <= 1        # remainders balance across benchmarks


def test_assignment_follows_shares_and_seed(cfg):
    weighted = copy.deepcopy(cfg)
    weighted["shards"][0]["share"] = 2
    out = shard_assignment(weighted, _items(24))
    per = Counter(out.values())
    assert per == {"parasail": 48, "mancer": 24, "dekallm": 24}
    reseeded = copy.deepcopy(cfg)
    reseeded["seed"] = 1
    assert shard_assignment(reseeded, _items(24)) != shard_assignment(cfg, _items(24))


def test_assignment_accepts_ids_and_records(cfg):
    tasks = [PublicTask(item_id=i, benchmark=b, source_revision="r", prompt="p", output_kind=OutputKind.MATH)
             for i, b in _items(4)]
    assert shard_assignment(cfg, tasks) == shard_assignment(cfg, [i for i, _ in _items(4)])


def test_derived_shard_config(cfg):
    derived = shard_config(cfg, "mancer")
    assert derived["provider"]["endpoint_tag"] == "mancer/fp8"
    assert derived["budget"]["credit_ceiling_usd"] == 31.66
    assert derived["budget"]["soft_admission_ceiling_usd"] == 31.0
    assert derived["budget"]["pilot_allowance_usd"] == 3.0 and derived["budget"]["contingency_usd"] == 2.33
    assert derived["budget"]["max_unresolved_spend_usd"] == cfg["budget"]["max_unresolved_spend_usd"]
    assert derived["shard"]["id"] == "mancer" and base_config(derived) == cfg
    assert validate_config(derived) == []
    assert shard_config(derived, "dekallm") == shard_config(cfg, "dekallm")     # derivation is idempotent
    total = sum(Decimal(r["credit_ceiling_usd"]) for r in shard_budget_table(cfg))
    assert total <= Decimal("95")
    tampered = copy.deepcopy(derived)
    tampered["budget"]["credit_ceiling_usd"] = 95
    assert any("credit ceiling" in p for p in validate_config(tampered))
    tampered = copy.deepcopy(derived)
    tampered["provider"]["endpoint_tag"] = "parasail/fp8"
    assert any("endpoint_tag" in p for p in validate_config(tampered))
    with pytest.raises(ShardError):
        shard_config(cfg, "nope")


def test_resolve_shard_requires_shard_for_sharded_config(cfg, tmp_path):
    with pytest.raises(ShardError):
        resolve_shard(cfg, tmp_path, None)
    derived, paths = resolve_shard(cfg, tmp_path, "parasail")
    assert paths.run_dir == tmp_path / "shards" / "parasail"
    assert derived["provider"]["endpoint_tag"] == "parasail/fp8"
    again, same = resolve_shard(derived, paths.run_dir, None)     # a derived config runs in its own dir
    assert same.run_dir == paths.run_dir and again == derived


def _prepared_root(tmp_path, cfg) -> RunPaths:
    paths = RunPaths.for_run(tmp_path / "root", local_state_dir=tmp_path / "local")
    paths.ensure()
    draws = {}
    for b in BENCHMARKS:
        dev = tuple(f"{b}/dev{i}" for i in range(8))
        main = tuple(f"{b}/m{i}" for i in range(8))
        draws[b] = PanelDraw(b, dev, main, (), 1, "test")
        tasks = [PublicTask(item_id=i, benchmark=b, source_revision="r", prompt=f"q {i}", output_kind=OutputKind.MATH)
                 for i in dev + main]
        paths.public_manifest(b).parent.mkdir(parents=True, exist_ok=True)
        paths.public_manifest(b).write_text("".join(json.dumps(dump_record(t)) + "\n" for t in tasks))
        privates = [PrivateEvalSpec(i, b, "r", EvalKind.MATH_VERIFY, "1", None, "h") for i in dev + main]
        paths.private_manifest(b).write_text("".join(json.dumps(dump_record(p)) + "\n" for p in privates))
    paths.panels.write_text(json.dumps({b: dump_record(d) for b, d in draws.items()}))
    waves = [Wave(w, tuple(i for b in BENCHMARKS for i in draws[b].main_item_ids[w * 4:(w + 1) * 4]),
                  ("main", "extension")) for w in range(2)]
    paths.waves.write_text(json.dumps([dump_record(w) for w in waves]))
    paths.dataset_lock.write_text(json.dumps({"study_id": cfg["study_id"]}))
    return paths


def test_materialized_shards_partition_the_root(cfg, tmp_path):
    root = _prepared_root(tmp_path, cfg)
    assignment = root_assignment(cfg, root)
    assert assignment["counts"]["main"] and assignment["counts"]["dev"]
    seen_main: set[str] = set()
    seen_public: set[str] = set()
    for shard in ("parasail", "mancer", "dekallm"):
        info = materialize_shard(cfg, root, shard, assignment, local_state_dir=tmp_path / f"local_{shard}")
        assert info["problems"] == []
        paths = RunPaths.for_run(root.run_dir / "shards" / shard, local_state_dir=tmp_path / f"local_{shard}")
        waves = json.loads(paths.waves.read_text())
        items = {i for w in waves for i in w["item_ids"]}
        assert items == items_of(assignment["assignment"], shard) & {i for w in json.loads(root.waves.read_text())
                                                                      for i in w["item_ids"]}
        assert not items & seen_main
        seen_main |= items
        public = {json.loads(l)["item_id"] for b in BENCHMARKS for l in paths.public_manifest(b).read_text().splitlines()}
        private = {json.loads(l)["item_id"] for b in BENCHMARKS for l in paths.private_manifest(b).read_text().splitlines()}
        assert public == private == items_of(assignment["assignment"], shard)
        assert not public & seen_public
        seen_public |= public
        derived = json.loads((paths.state_dir / "shard_config.json").read_text())
        assert derived["provider"]["endpoint_tag"] == shard_config(cfg, shard)["provider"]["endpoint_tag"]
        assert shard_filter(derived, paths) == items_of(assignment["assignment"], shard)
        assert paths.dataset_lock.is_file()
    assert len(seen_main) == 4 * 8 and len(seen_public) == 4 * 16
    assert shard_filter(cfg, root) is None


def test_filter_graph_waves_drops_empty_waves():
    class G:
        def __init__(self, item_id):
            self.item_id = item_id
    waves = [[G("a"), G("b")], [G("c")]]
    assert [[g.item_id for g in w] for w in filter_graph_waves(waves, {"a"})] == [["a"]]
    assert len(filter_graph_waves(waves, None)) == 2


def test_live_config_shards_follow_the_screen():
    """configs/forced27b.json as launched: Mancer 3 : Parasail 2, Parasail capped at 40 in flight."""
    live_cfg = forced_test_config()
    assert [(s["id"], s["share"]) for s in live_cfg["shards"]] == [("mancer", 3), ("parasail", 2)]
    mancer, parasail = shard_config(live_cfg, "mancer"), shard_config(live_cfg, "parasail")
    # owner-set slices within the $95 cap (pilot forecasts: Mancer 56.89, Parasail 34.57 core+spent+reserve)
    assert mancer["budget"]["credit_ceiling_usd"] == 58.5 and parasail["budget"]["credit_ceiling_usd"] == 36.5
    assert mancer["budget"]["soft_admission_ceiling_usd"] == 57.5
    assert parasail["budget"]["soft_admission_ceiling_usd"] == 35.5
    assert parasail["scheduler"]["concurrency_max"] == 40 and parasail["scheduler"]["requests_per_second_max"] == 8
    assert mancer["scheduler"]["concurrency_max"] == live_cfg["scheduler"]["concurrency_max"]
    bad = copy.deepcopy(live_cfg)
    bad["scheduler_by_shard"] = {"parasail": {"not_a_key": 1}}
    with pytest.raises(ValueError):
        shard_config(bad, "parasail")


def test_shard_key_window_reconciles_the_union_of_concurrent_shards(tmp_path, monkeypatch):
    """Shards share one key: each shard's /key window contains its sibling's spend, so Gate 1 reconciles the union
    (earliest before .. latest after) against the sum of every shard's ledger, and waits while a sibling is open."""
    from agent_allocation import live
    live_cfg = forced_test_config()
    settled = {"mancer": Decimal("0.40"), "parasail": Decimal("0.35")}
    monkeypatch.setattr(live, "ledger_settled", lambda p: settled[p.run_dir.name])

    def write(sid, before, after, settled_before="0"):
        p = RunPaths.for_run(tmp_path / "shards" / sid, local_state_dir=tmp_path / f"local_{sid}")
        p.state_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"endpoint": "key", "stage": "canary_before", "data": {"usage": before}}]
        if after is not None:
            rows.append({"endpoint": "key", "stage": "canary_after", "data": {"usage": after}})
        p.provider_usage.write_text("".join(json.dumps(r) + "\n" for r in rows))
        (p.state_dir / "canary_before.json").write_text(json.dumps({"settled_before": settled_before}))
        return p

    mancer = write("mancer", 10.00, 10.76)
    write("parasail", 9.99, None)
    derived = shard_config(live_cfg, "mancer")
    assert live.shard_key_window(derived, mancer, "canary") == {"pending": ["parasail"]}
    write("parasail", 9.99, 10.70)
    union = live.shard_key_window(derived, mancer, "canary")
    assert union["key_delta"] == Decimal("10.76") - Decimal("9.99")
    assert union["ledger_settled"] == Decimal("0.75") and union["shards"] == ["mancer", "parasail"]
    assert live.shard_key_window(live_cfg, mancer, "canary") is None          # unsharded config: own window


def test_budget_by_shard_overrides_slices_within_the_cap():
    from agent_allocation.shards import validate_shard_budgets
    live_cfg = copy.deepcopy(forced_test_config())
    live_cfg["budget_by_shard"] = {"mancer": {"credit_ceiling_usd": 58.5, "soft_admission_ceiling_usd": 57.5},
                                   "parasail": {"credit_ceiling_usd": 36.5, "soft_admission_ceiling_usd": 35.5}}
    live_cfg["budget"]["soft_admission_ceiling_usd"] = 93
    assert validate_shard_budgets(live_cfg) == []
    for sid, credit in (("mancer", 58.5), ("parasail", 36.5)):
        derived = shard_config(live_cfg, sid)
        assert derived["budget"]["credit_ceiling_usd"] == credit
        assert validate_config(derived) == []
    live_cfg["budget_by_shard"]["mancer"]["credit_ceiling_usd"] = 60
    assert any("credit_ceiling_usd slices sum" in p for p in validate_shard_budgets(live_cfg))
    live_cfg["budget_by_shard"]["mancer"] = {"max_unresolved_spend_usd": 5}
    with pytest.raises(ValueError):
        shard_config(live_cfg, "mancer")
