"""WP-G P4: the reserve panel (T200) takes its own 1:1 share; dev/main keep seed 20260917 and 3:2 (legacy)."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from agent_allocation.config import load_config
from agent_allocation.paths import RunPaths
from agent_allocation.shards import root_assignment
from tests.forced_cfg import PROJECT
from tests.unit.passk.wpg_fixtures import LEGACY_ROOT, fixture_splits, legacy_available, write_fixture_root


def _root(tmp_path, *, explicit: bool, use_legacy: bool | None = None):
    splits, tasks = fixture_splits(use_legacy=use_legacy)
    if not explicit:
        splits.pop("assignment_t200")
    artifacts = tmp_path / "passk" / "artifacts"
    write_fixture_root(artifacts, splits, tasks)
    return splits, RunPaths.for_run(artifacts, local_state_dir=tmp_path / "local")


def _cfg(name="passk_cont_v1"):
    return load_config(PROJECT / "configs" / f"{name}.json")


@pytest.mark.skipif(not legacy_available(), reason="legacy run root not readable")
def test_legacy_assignment_reproduced_for_l96_and_d32(tmp_path):
    splits, root = _root(tmp_path, explicit=False, use_legacy=True)
    out = root_assignment(_cfg(), root)
    legacy = json.loads((LEGACY_ROOT / "shards" / "assignment.json").read_text())["assignment"]
    for item in [*splits["panels"]["L96"], *splits["panels"]["D32"]]:
        assert out["assignment"][item] == legacy[item], item
    assert out["seed"] == 20260917
    assert out["counts"]["main"] == json.loads((LEGACY_ROOT / "shards" / "assignment.json").read_text())["counts"]["main"]


def test_t200_is_one_to_one_per_benchmark(tmp_path):
    splits, root = _root(tmp_path, explicit=False)
    out = root_assignment(_cfg(), root)
    t200 = splits["panels"]["T200"]
    by = Counter((i.split("/", 1)[0], out["assignment"][i]) for i in t200)
    for b, n in {"aime": 20, "hmmt": 20, "omni_math_hard": 100, "gpqa_diamond": 60}.items():
        assert by[(b, "mancer")] == by[(b, "parasail")] == n // 2
    assert out["panel_overrides"]["reserve"]["source"] == "passk.t200_shard_shares"


def test_explicit_assignment_t200_wins(tmp_path):
    splits, root = _root(tmp_path, explicit=True)
    out = root_assignment(_cfg(), root)
    for item, shard in splits["assignment_t200"].items():
        assert out["assignment"][item] == shard
    assert out["panel_overrides"]["reserve"]["source"] == "splits.json:assignment_t200"


def test_legacy_config_unchanged(tmp_path):
    _splits, root = _root(tmp_path, explicit=True)
    cfg = _cfg("forced27b")
    out = root_assignment(cfg, root)
    assert "panel_overrides" not in out
    reserve = Counter(out["counts"]["reserve"]["mancer"].values())
    # 3:2 on the reserve panel (not 1:1)
    total_m = sum(out["counts"]["reserve"]["mancer"].values())
    total_p = sum(out["counts"]["reserve"]["parasail"].values())
    assert total_m > total_p and reserve
