"""Exact-clone and immutable-lineage tests for schema-5 run initialization."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from agents_scaling.config import ExperimentCell
from agents_scaling.experiment.manifest import freeze_manifest
from agents_scaling.serving.model_contracts import load_model_contracts
from scripts import clone_schema5_manifests as clone


def _cell(seed: int = 0, *, n_agents: int = 1) -> ExperimentCell:
    return ExperimentCell.from_dict(
        {
            "model_size": "0.6B",
            "context_share_level": "artifact_only",
            "prompt_complexity_level": 0,
            "reasoning_level": "off",
            "topology": "single_agent" if n_agents == 1 else "independent",
            "benchmark": "gpqa",
            "n_agents": n_agents,
            "rounds": 1,
            "n_samples": 1,
            "temperature": 0.0,
            "n_questions": 1,
            "seed": seed,
        }
    )


def _source_run(tmp_path: Path, cells: list[ExperimentCell]) -> Path:
    root = tmp_path / "legacy_v1"
    root.mkdir(parents=True)
    (root / "cells.json").write_text(
        json.dumps([cell.to_dict() for cell in cells]), encoding="utf-8"
    )
    freeze_manifest(root)
    manifest_sha = hashlib.sha256((root / "cells.json").read_bytes()).hexdigest()
    benchmark = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha,
        "contracts": [],
    }
    benchmark_bytes = (json.dumps(benchmark, sort_keys=True) + "\n").encode()
    (root / "benchmark_contracts.v1.json").write_bytes(benchmark_bytes)
    (root / "benchmark_contracts.v1.sha256").write_text(
        f"{hashlib.sha256(benchmark_bytes).hexdigest()}  benchmark_contracts.v1.json\n",
        encoding="utf-8",
    )
    return root


def _pins() -> clone.ReleasePins:
    return clone.ReleasePins(
        release_id="sweep-recovery-schema5-v1",
        git_commit="a" * 40,
        source_tree_sha256="b" * 64,
        harness_sha256="c" * 64,
        serving_sha256="d" * 64,
    )


def test_dry_run_does_not_create_target(tmp_path):
    source = _source_run(tmp_path, [_cell()])
    target = tmp_path / "clean_schema5"

    report = clone.clone_run(
        source_root=source,
        target_root=target,
        expected_cell_count=1,
        model_contracts=load_model_contracts(),
        pins=_pins(),
    )

    assert report["status"] == "dry_run"
    assert report["would_import_result_rows"] == 0
    assert not target.exists()


def test_clone_is_byte_exact_empty_policy_bound_and_idempotent(tmp_path):
    source = _source_run(tmp_path, [_cell()])
    target = tmp_path / "clean_schema5"
    model_contracts = load_model_contracts()

    report = clone.clone_run(
        source_root=source,
        target_root=target,
        expected_cell_count=1,
        model_contracts=model_contracts,
        pins=_pins(),
        apply=True,
    )

    assert report["status"] == "initialized"
    for filename in clone.STATIC_CONTRACT_FILENAMES:
        assert (target / filename).read_bytes() == (source / filename).read_bytes()
        assert stat.S_IMODE((target / filename).stat().st_mode) == 0o444
        assert (target / filename).stat().st_ino != (source / filename).stat().st_ino
    assert list((target / "cells").iterdir()) == []
    lineage = json.loads((target / clone.LINEAGE_FILENAME).read_text())
    assert lineage["source_run_id"] == source.name
    assert lineage["target_run_id"] == target.name
    assert lineage["imported_result_rows"] == 0
    assert lineage["transformations"] == []
    policy_bytes = (target / clone.POLICY_FILENAME).read_bytes()
    policy = json.loads(policy_bytes)
    assert policy["required_artifact_schema_version"] == 5
    assert policy["accepted_model_contract_sha256"] == model_contracts.sha256
    assert policy["legacy_result_import_allowed"] is False
    assert tuple(policy["required_metadata_fields"]) == clone.REQUIRED_METADATA_FIELDS
    assert (target / clone.POLICY_CHECKSUM_FILENAME).read_text() == (
        f"{hashlib.sha256(policy_bytes).hexdigest()}  {clone.POLICY_FILENAME}\n"
    )
    marker = json.loads((target / clone.INITIALIZED_FILENAME).read_text())
    assert marker["cells_directory_empty_at_initialization"] is True
    assert marker["legacy_result_rows_imported"] == 0

    # Normal production progress does not invalidate immutable initialization.
    cell_dir = target / "cells" / "cell-a"
    cell_dir.mkdir()
    (cell_dir / "results.jsonl").write_text('{"schema_version":5}\n')
    again = clone.clone_run(
        source_root=source,
        target_root=target,
        expected_cell_count=1,
        model_contracts=model_contracts,
        pins=_pins(),
        apply=True,
    )
    assert again["status"] == "already_initialized"


def test_wrong_source_checksum_and_nonempty_uninitialized_target_fail_closed(tmp_path):
    source = _source_run(tmp_path, [_cell()])
    target = tmp_path / "clean_schema5"
    (source / "cells.json").write_bytes((source / "cells.json").read_bytes() + b"\n")

    with pytest.raises(clone.CloneError, match="checksum"):
        clone.clone_run(
            source_root=source,
            target_root=target,
            expected_cell_count=1,
            model_contracts=load_model_contracts(),
            pins=_pins(),
        )

    source = _source_run(tmp_path / "second", [_cell(seed=1)])
    target.mkdir()
    (target / "legacy.jsonl").write_bytes(b"must not import")
    with pytest.raises(clone.CloneError, match="initialized marker"):
        clone.clone_run(
            source_root=source,
            target_root=target,
            expected_cell_count=1,
            model_contracts=load_model_contracts(),
            pins=_pins(),
            apply=True,
        )
    assert (target / "legacy.jsonl").read_bytes() == b"must not import"


def test_fixed_clone_map_is_exact_22680_and_new_ids():
    assert sum(count for _source, _target, count in clone.RUN_CLONES) == 22_680
    assert [target for _source, target, _count in clone.RUN_CLONES] == [
        "full_sweep_schema5_v1",
        "full_sweep_agent_counts_schema5_v1",
        "full_sweep_agent_count_7_schema5_v1",
    ]
    assert len({source for source, _target, _count in clone.RUN_CLONES}) == 3
