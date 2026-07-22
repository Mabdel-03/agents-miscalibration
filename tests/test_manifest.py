import hashlib
import json

import pytest

from agents_scaling.experiment.manifest import (
    freeze_manifest,
    load_manifest,
    manifested_cell_dirs,
)


def _cell(size="0.6B", seed=0):
    return {
        "model_size": size,
        "context_share_level": "artifact_only",
        "prompt_complexity_level": 0,
        "reasoning_level": "off",
        "topology": "single_agent",
        "benchmark": "gpqa",
        "n_agents": 1,
        "rounds": 1,
        "n_samples": 1,
        "temperature": 0.0,
        "n_questions": 2,
        "seed": seed,
    }


def test_manifest_freeze_detects_drift(tmp_path):
    path = tmp_path / "cells.json"
    path.write_text(json.dumps([_cell()]))
    frozen = freeze_manifest(tmp_path)
    assert frozen.exists()
    assert load_manifest(tmp_path).sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    path.write_text(json.dumps([_cell(seed=1)]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_manifest(tmp_path)


def test_manifest_dirs_exclude_stale_by_default(tmp_path):
    (tmp_path / "cells.json").write_text(json.dumps([_cell()]))
    snapshot = load_manifest(tmp_path)
    cells_root = tmp_path / "cells"
    (cells_root / snapshot.ids[0]).mkdir(parents=True)
    (cells_root / "legacy_cell").mkdir()

    paths, stale, missing = manifested_cell_dirs(tmp_path)
    assert [path.name for path in paths] == [snapshot.ids[0]]
    assert stale == {"legacy_cell"}
    assert not missing

    paths, _, _ = manifested_cell_dirs(tmp_path, include_unmanifested=True)
    assert {path.name for path in paths} == {snapshot.ids[0], "legacy_cell"}
