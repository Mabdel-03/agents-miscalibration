"""Tests for deterministic marker-last effective-fleet materialization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest

from agents_scaling.serving.fleet_contract import load_fleet_contract
from agents_scaling.serving.model_contracts import load_model_contracts
from scripts import materialize_schema5_effective_fleet as materializer


SOURCE_REPO = Path(materializer.__file__).resolve().parents[1]


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    root = (tmp_path / "release").resolve()
    configs = root / "configs"
    scripts = root / "scripts"
    configs.mkdir(parents=True)
    scripts.mkdir()
    for relative in (
        "configs/schema5_fleet.v1.json",
        "configs/schema5_fleet.v1.sha256",
        "configs/model_contracts.v1.json",
        "configs/model_contracts.v1.sha256",
        "scripts/materialize_schema5_effective_fleet.py",
        "scripts/run_schema5_throughput_qualification.py",
        "slurm/dispatch_sweeps.py",
    ):
        source = SOURCE_REPO / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Schema5 Test",
        "-c",
        "user.email=schema5@example.invalid",
        "commit",
        "-qm",
        "fixture release",
    )
    _git(
        root,
        "-c",
        "user.name=Schema5 Test",
        "-c",
        "user.email=schema5@example.invalid",
        "tag",
        "-a",
        materializer.RELEASE_TAG,
        "-m",
        "fixture annotated release",
    )
    commit = _git(root, "rev-parse", "HEAD")
    tag_object = _git(
        root, "rev-parse", f"refs/tags/{materializer.RELEASE_TAG}"
    )
    for path in root.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            path.chmod(0o444)
    monkeypatch.setattr(materializer, "REPO", root)
    monkeypatch.setattr(
        materializer,
        "__file__",
        str(root / "scripts" / "materialize_schema5_effective_fleet.py"),
    )
    base = configs / materializer.BASE_FILENAME
    model = configs / materializer.MODEL_FILENAME
    return {
        "root": root,
        "commit": commit,
        "tag_object": tag_object,
        "base": base,
        "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "model": model,
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "output_root": (tmp_path / "effective-fleet").resolve(),
    }


def _arguments(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "release_worktree": fixture["root"],
        "release_git_commit": fixture["commit"],
        "release_tag_object": fixture["tag_object"],
        "base_fleet_contract": fixture["base"],
        "base_fleet_contract_sha256": fixture["base_sha256"],
        "model_contract": fixture["model"],
        "model_contract_sha256": fixture["model_sha256"],
        "output_root": fixture["output_root"],
    }


def test_dry_run_derives_exact_fixed_delta_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release(tmp_path, monkeypatch)
    report = materializer.materialize(
        **_arguments(fixture), apply=False
    )

    assert report["status"] == "dry_run"
    assert report["passed"] is True
    assert report["base_profile_replicas"] == dict(
        materializer.EXPECTED_COUNTS
    )
    assert report["fixed_additive_profile_delta"] == {
        "0.6B": 0,
        "1.7B": 0,
        "4B": 0,
        "8B": 0,
        "14B": 0,
        "32B": 0,
        "0.6B-long": 0,
        "1.7B-long": 0,
        "4B-long": 0,
        "8B-long": 0,
        "14B-long": 0,
        "32B-long": 0,
    }
    assert sum(report["fixed_additive_profile_delta"].values()) == 0
    assert report["effective_profile_replicas"] == dict(
        materializer.EXPECTED_EFFECTIVE_COUNTS
    )
    assert not Path(fixture["output_root"]).exists()
    assert not (
        Path(fixture["output_root"]).parent
        / f".{Path(fixture['output_root']).name}.materialize.lock"
    ).exists()


def test_apply_publishes_byte_identical_22_replica_24_gpu_contract_marker_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release(tmp_path, monkeypatch)
    report = materializer.materialize(
        **_arguments(fixture), apply=True
    )
    root = Path(fixture["output_root"])

    assert report["status"] == "complete"
    assert {path.name for path in root.iterdir()} == {
        materializer.INTENT_FILENAME,
        materializer.OUTPUT_FILENAME,
        materializer.CHECKSUM_FILENAME,
        materializer.COMPLETE_FILENAME,
    }
    for path in root.iterdir():
        metadata = path.stat(follow_symlinks=False)
        assert stat.S_ISREG(metadata.st_mode)
        assert stat.S_IMODE(metadata.st_mode) & 0o222 == 0
        assert metadata.st_nlink == 1
    output = root / materializer.OUTPUT_FILENAME
    sidecar = root / materializer.CHECKSUM_FILENAME
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert sidecar.read_text(encoding="ascii") == (
        f"{digest}  {output.name}\n"
    )
    models = load_model_contracts(
        fixture["model"], expected_sha256=fixture["model_sha256"]
    )
    fleet = load_fleet_contract(
        output,
        model_contracts=models,
        expected_sha256=digest,
        allow_capacity_layout=True,
    )
    assert output.read_bytes() == Path(fixture["base"]).read_bytes()
    assert digest == fixture["base_sha256"]
    assert len(fleet.replicas) == 22
    assert sum(row.gpus_per_replica for row in fleet.replicas) == 24
    assert {
        profile: len(rows)
        for profile, rows in fleet.by_profile.items()
    } == dict(materializer.EXPECTED_EFFECTIVE_COUNTS)
    marker = json.loads(
        (root / materializer.COMPLETE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert marker["protocol"] == materializer.PROTOCOL
    assert marker["passed"] is True
    assert marker["additive_logical_replicas"] == 0
    assert marker["additive_allocated_gpus"] == 0
    assert marker["effective_logical_replicas"] == 22
    assert marker["effective_allocated_gpus"] == 24
    assert (
        marker["runbook_inputs"]["effective_fleet_contract"]
        == marker["runbook_inputs"]["additive_overlay_contract"]
        == str(output)
    )
    assert (
        marker["runbook_inputs"]["effective_fleet_contract_sha256"]
        == marker["runbook_inputs"]["additive_overlay_contract_sha256"]
        == digest
    )
    assert marker["runbook_inputs"]["source_tree_sha256"] == (
        marker["source_tree_sha256"]
    )
    assert marker["runbook_inputs"]["dispatcher_source_sha256"] == (
        marker["dispatcher_source_sha256"]
    )
    assert (
        marker["runbook_inputs"][
            "qualification_runner_source_sha256"
        ]
        == marker["qualification_runner_source_sha256"]
    )


def test_replay_revalidates_without_rewriting_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release(tmp_path, monkeypatch)
    arguments = _arguments(fixture)
    materializer.materialize(**arguments, apply=True)
    root = Path(fixture["output_root"])
    before = {
        path.name: (
            path.stat().st_ino,
            path.stat().st_mtime_ns,
            path.read_bytes(),
        )
        for path in root.iterdir()
    }

    report = materializer.materialize(**arguments, apply=True)
    verified = materializer.verify(**arguments)

    assert report["status"] == "already_complete"
    assert verified["status"] == "complete"
    after = {
        path.name: (
            path.stat().st_ino,
            path.stat().st_mtime_ns,
            path.read_bytes(),
        )
        for path in root.iterdir()
    }
    assert after == before


def test_replay_recovers_interrupted_pending_sidecar_before_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release(tmp_path, monkeypatch)
    arguments = _arguments(fixture)
    transaction = materializer._prepare_transaction(**arguments)
    root = Path(fixture["output_root"])
    root.mkdir()
    intent_id = transaction["intent"]["intent_id"]
    materializer._publish_once(
        root / materializer.INTENT_FILENAME,
        transaction["intent_raw"],
        transaction_id=intent_id,
        description="test intent",
    )
    materializer._publish_once(
        root / materializer.OUTPUT_FILENAME,
        transaction["candidate_raw"],
        transaction_id=intent_id,
        description="test fleet",
    )
    pending = root / (
        f".{materializer.CHECKSUM_FILENAME}.{intent_id}.pending"
    )
    pending.write_bytes(b"interrupted")

    report = materializer.materialize(**arguments, apply=True)

    assert report["status"] == "complete"
    assert not pending.exists()
    assert (
        root / materializer.COMPLETE_FILENAME
    ).is_file()


def test_replay_adopts_exact_sealed_pending_before_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release(tmp_path, monkeypatch)
    arguments = _arguments(fixture)
    transaction = materializer._prepare_transaction(**arguments)
    root = Path(fixture["output_root"])
    root.mkdir()
    intent_id = transaction["intent"]["intent_id"]
    materializer._publish_once(
        root / materializer.INTENT_FILENAME,
        transaction["intent_raw"],
        transaction_id=intent_id,
        description="test intent",
    )
    pending = root / (
        f".{materializer.OUTPUT_FILENAME}.{intent_id}.pending"
    )
    pending.write_bytes(transaction["candidate_raw"])
    pending.chmod(0o444)

    report = materializer.materialize(**arguments, apply=True)

    assert report["status"] == "complete"
    assert not pending.exists()
    assert (
        root / materializer.OUTPUT_FILENAME
    ).read_bytes() == transaction["candidate_raw"]


def test_conflicting_partial_contract_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release(tmp_path, monkeypatch)
    arguments = _arguments(fixture)
    transaction = materializer._prepare_transaction(**arguments)
    root = Path(fixture["output_root"])
    root.mkdir()
    intent_id = transaction["intent"]["intent_id"]
    materializer._publish_once(
        root / materializer.INTENT_FILENAME,
        transaction["intent_raw"],
        transaction_id=intent_id,
        description="test intent",
    )
    conflict = root / materializer.OUTPUT_FILENAME
    conflict.write_bytes(b"{}\n")
    conflict.chmod(0o444)

    with pytest.raises(
        materializer.EffectiveFleetMaterializationError,
        match="conflicts with the frozen transaction",
    ):
        materializer.materialize(**arguments, apply=True)
    assert not (root / materializer.COMPLETE_FILENAME).exists()


def test_mutable_tagged_input_and_wrong_tag_object_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release(tmp_path, monkeypatch)
    arguments = _arguments(fixture)
    base = Path(fixture["base"])
    base.chmod(0o644)
    with pytest.raises(
        materializer.EffectiveFleetMaterializationError,
        match="mutable, linked, or changed",
    ):
        materializer.materialize(**arguments, apply=False)
    base.chmod(0o444)
    arguments["release_tag_object"] = "0" * 40
    with pytest.raises(
        materializer.EffectiveFleetMaterializationError,
        match="exact annotated tag",
    ):
        materializer.materialize(**arguments, apply=False)


def test_zero_delta_preserves_every_base_replica_and_profile_byte_for_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release(tmp_path, monkeypatch)
    materializer.materialize(
        **_arguments(fixture), apply=True
    )
    base = json.loads(
        Path(fixture["base"]).read_text(encoding="utf-8")
    )
    effective = json.loads(
        (
            Path(fixture["output_root"])
            / materializer.OUTPUT_FILENAME
        ).read_text(encoding="utf-8")
    )
    base_profiles = {
        row["serving_profile"]: row for row in base["profiles"]
    }
    effective_profiles = {
        row["serving_profile"]: row
        for row in effective["profiles"]
    }

    assert effective == base
    for profile, delta in materializer.FIXED_ADDITIVE_PROFILE_DELTA.items():
        assert delta == 0
        old = base_profiles[profile]["replicas"]
        new = effective_profiles[profile]["replicas"]
        assert new == old


def test_sibling_publication_lock_rejects_symlink_and_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release(tmp_path, monkeypatch)
    arguments = _arguments(fixture)
    output_root = Path(fixture["output_root"])
    lock = output_root.parent / (
        f".{output_root.name}.materialize.lock"
    )
    target = output_root.parent / "lock-target"
    target.write_bytes(b"lock")
    lock.symlink_to(target)
    with pytest.raises(
        materializer.EffectiveFleetMaterializationError,
        match="cannot safely open",
    ):
        materializer.materialize(**arguments, apply=True)
    lock.unlink()
    os.link(target, lock)
    with pytest.raises(
        materializer.EffectiveFleetMaterializationError,
        match="aliased, linked, or unsafe",
    ):
        materializer.materialize(**arguments, apply=True)


def test_directory_fsync_failure_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = materializer.os.fsync

    def fail(_descriptor: int) -> None:
        raise OSError("injected directory durability failure")

    monkeypatch.setattr(materializer.os, "fsync", fail)
    with pytest.raises(
        materializer.EffectiveFleetMaterializationError,
        match="cannot durably fsync publication directory",
    ):
        materializer._fsync_directory(tmp_path)
    monkeypatch.setattr(materializer.os, "fsync", original)


def test_sealed_read_rejects_post_close_metadata_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "sealed.json").resolve()
    target.write_bytes(b"{}\n")
    target.chmod(0o444)
    original_close = materializer.os.close
    mutated = False

    def close_then_mutate(descriptor: int) -> None:
        nonlocal mutated
        original_close(descriptor)
        if not mutated:
            mutated = True
            target.chmod(0o644)

    monkeypatch.setattr(materializer.os, "close", close_then_mutate)
    with pytest.raises(
        materializer.EffectiveFleetMaterializationError,
        match="mutable, linked, or changed while read",
    ):
        materializer._stable_sealed_bytes(
            target, description="post-close mutation fixture"
        )


def test_apply_revalidates_source_under_lock_before_complete_fast_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _release(tmp_path, monkeypatch)
    arguments = _arguments(fixture)
    materializer.materialize(**arguments, apply=True)
    original = materializer._open_publication_lock
    publisher = Path(materializer.__file__)

    def acquire_then_drift(path: Path) -> int:
        descriptor = original(path)
        publisher.chmod(0o644)
        return descriptor

    monkeypatch.setattr(
        materializer, "_open_publication_lock", acquire_then_drift
    )
    with pytest.raises(
        materializer.EffectiveFleetMaterializationError,
        match="mutable, linked, or changed",
    ):
        materializer.materialize(**arguments, apply=True)
