"""Integration ($0): the analysis pipeline on real L96 legacy data plus synthetic new partitions.

Synthetic ``e1_fill`` replicates copy each item's legacy deep r0 answer (so no new pair is needed); a synthetic
``v48_bank`` stands in for the fenced bank. Checks: extract fences v48, the E1 lock runs and is write-once, the
seal refuses an injected fenced read and accepts a clean log, and the release then exposes v48 labels.
Skipped without the legacy root or a filled pair cache.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from agent_allocation.passk import seal as seal_mod
from agent_allocation.passk.analyze import e1_lock
from agent_allocation.passk.extract import RUN_ROOT, extract, load_draws
from agent_allocation.passk.labels import verify_seal

from . import fakeroot


@pytest.fixture(scope="module")
def cache_path():
    path = Path(os.environ.get("PASSK_PAIR_CACHE") or RUN_ROOT / "catalog" / "pair_verdicts.parquet")
    if not path.exists():
        pytest.skip("no filled pair cache")
    return path


def test_pipeline_on_real_legacy(tmp_path, legacy_root, cache_path):
    run_root = tmp_path
    art = run_root / "artifacts"
    shutil.copytree(legacy_root / "manifests" / "public", art / "manifests" / "public")
    (run_root / "catalog").mkdir()
    shutil.copy(cache_path, run_root / "catalog" / "pair_verdicts.parquet")
    tasks = {}
    for f in (art / "manifests" / "public").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                tasks[t["item_id"]] = t
    from agent_allocation.passk.extract import legacy_draws
    legacy = legacy_draws(legacy_root, read_raw=False)
    deep = legacy[(legacy.partition == "legacy_main") & (legacy.kind == "deep")]
    shard = art / "shards" / "mancer"
    c48 = sorted(deep.item_id.unique())[:8]
    for row in deep.itertuples():
        if row.allowance_a not in (2, 4, 8):
            continue
        extra = {2: range(1, 8), 4: range(1, 4), 8: range(1, 2)}[row.allowance_a]
        text = row.vote_text if isinstance(row.vote_text, str) else None
        correct = row.correct if isinstance(row.correct, bool) else None
        final = text if tasks[row.item_id]["output_kind"] == "option" else (
            None if text is None else "\\boxed{" + text + "}")
        for r in extra:
            fakeroot.write_node(shard, "e1_fill", tasks[row.item_id], row.allowance_a * 1024, r, final,
                                correct)
    for item in c48:
        for r in range(8):
            fakeroot.write_node(shard, "v48_bank", tasks[item], 2048, r, None, None)
    splits = {"panels": {"L96": sorted(legacy[legacy.partition == "legacy_main"].item_id.unique()), "C48": c48,
                         "V48": c48, "T200": []}}
    (art / "manifests" / "splits.json").write_text(json.dumps(splits))

    manifest = extract(run_root, legacy_root=legacy_root, read_raw=False)
    assert manifest["classes_pending_pools"] == 0
    draws = load_draws(run_root)
    assert not (draws.partition == "v48_bank").any()
    assert (draws.partition == "e1_fill").sum() == 96 * 11
    dev = pd.read_parquet(run_root / "analysis_inputs" / "dev_training_readouts.parquet")
    assert dev.pool_id.is_unique and not set(dev.config_id) & {"4x4", "8x1", "4x2", "2x4"}
    pools = pd.read_parquet(run_root / "analysis_inputs" / "pools.parquet")
    assert (pools.config_id == "c48fit").sum() == len(c48)

    payload = e1_lock(run_root)
    assert payload["selected"] in ("8x2", "4x4", "2x8")
    assert payload["table"]["1x16"]["macro"] == pytest.approx(0.7917, abs=5e-5)
    # 8x2 over copies of the deep r0 answer delivers deep/2's answer; same for 4x4 and 2x8
    legacy_deep = legacy[(legacy.kind == "deep")].groupby(["allowance_a", "benchmark"]).correct.mean()
    for config_id, a in (("8x2", 2), ("4x4", 4), ("2x8", 8)):
        assert payload["table"][config_id]["macro"] == pytest.approx(legacy_deep.loc[a].mean(), abs=1e-9)

    (run_root / "sealed" / "v48").mkdir(parents=True)
    pd.DataFrame({"item_id": c48, "config_id": "1x16", "model": "beta", "pred_coverage": 0.5,
                  "pred_plurality": 0.5, "pi_lo": 0.1, "pi_hi": 0.9}).to_parquet(
        run_root / "sealed" / "v48" / "predictions.parquet")
    logs = run_root / "logs" / "strace"
    logs.mkdir(parents=True)
    clean = f'1 openat(AT_FDCWD, "{run_root}/analysis_inputs/draws.parquet", O_RDONLY) = 3\n'
    leak = f'1 openat(AT_FDCWD, "{art}/shards/mancer/parsed/aa/x.json", O_RDONLY) = 4\n'
    (logs / "e2-fit.log").write_text(clean)
    (logs / "e5-fit.log").write_text(clean + leak)
    assert seal_mod.run_seal(run_root, require_git=False) == 1 and verify_seal(run_root) is None
    (logs / "e5-fit.log").write_text(clean)
    assert seal_mod.run_seal(run_root, require_git=False) == 0 and verify_seal(run_root) is not None

    extract(run_root, legacy_root=legacy_root, read_raw=False, release=["v48"])
    released = load_draws(run_root)
    assert (released.partition == "v48_bank").sum() == 8 * len(c48)
    assert verify_seal(run_root) is not None
