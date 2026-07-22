"""Tests for exact, copy-based schema-5 release materialization."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import materialize_schema5_release as materialize


def _run(*argv: str, cwd: Path) -> str:
    completed = subprocess.run(
        list(argv), cwd=cwd, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _tagged_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _run("git", "init", "-q", cwd=repository)
    _run("git", "config", "user.email", "materialize@example.invalid", cwd=repository)
    _run("git", "config", "user.name", "Materialize Test", cwd=repository)
    (repository / "source.txt").write_text("release\n", encoding="utf-8")
    (repository / ".gitignore").write_text("*.egg-info/\n", encoding="utf-8")
    _run("git", "add", "source.txt", ".gitignore", cwd=repository)
    _run("git", "commit", "-q", "-m", "release", cwd=repository)
    _run("git", "tag", materialize.REQUIRED_TAG, cwd=repository)
    return repository, _run("git", "rev-parse", "HEAD", cwd=repository)


def _source_prefix(tmp_path: Path, name: str) -> Path:
    prefix = tmp_path / name
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "conda-meta" / "history").write_text("created\n", encoding="utf-8")
    (prefix / "bin").mkdir()
    (prefix / "bin" / "python").write_text("python\n", encoding="utf-8")
    return prefix


def _fake_conda(tmp_path: Path) -> Path:
    conda = tmp_path / "conda"
    conda.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    conda.chmod(0o755)
    return conda


def _release_runtime_resources(worktree: Path) -> dict:
    source = Path(__file__).resolve().parents[1]
    (worktree / "configs" / "prompts").mkdir(parents=True, exist_ok=True)
    (worktree / "slurm").mkdir(parents=True, exist_ok=True)
    for name in (
        "model_contracts.v1.json",
        "model_contracts.v1.sha256",
        "schema5_fleet.v1.json",
        "schema5_fleet.v1.sha256",
    ):
        shutil.copy2(source / "configs" / name, worktree / "configs" / name)
    for level in range(4):
        shutil.copy2(
            source / "configs" / "prompts" / f"level{level}.txt",
            worktree / "configs" / "prompts" / f"level{level}.txt",
        )
    shutil.copy2(
        source / "slurm" / "serve_qwen.sbatch.tmpl",
        worktree / "slurm" / "serve_qwen.sbatch.tmpl",
    )
    model = worktree / "configs" / "model_contracts.v1.json"
    fleet = worktree / "configs" / "schema5_fleet.v1.json"
    return {
        "release_worktree": str(worktree.resolve()),
        "model_contract_path": str(model.resolve()),
        "model_contract_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "fleet_contract_path": str(fleet.resolve()),
        "fleet_contract_sha256": hashlib.sha256(fleet.read_bytes()).hexdigest(),
        "serving_template": str(
            (worktree / "slurm" / "serve_qwen.sbatch.tmpl").resolve()
        ),
        "prompt_sha256": [
            hashlib.sha256(
                (worktree / "configs" / "prompts" / f"level{level}.txt")
                .read_text(encoding="utf-8")
                .strip()
                .encode("utf-8")
            ).hexdigest()
            for level in range(4)
        ],
        "fleet_replica_count": 22,
    }


def test_materialization_dry_run_is_read_only_and_plans_copy_semantics(tmp_path):
    repository, commit = _tagged_repository(tmp_path)
    source_harness = _source_prefix(tmp_path, "source-harness")
    source_serving = _source_prefix(tmp_path, "source-serving")
    release_root = tmp_path / "release"

    report = materialize.materialize_release(
        output_root=release_root,
        source_repository=repository,
        release_worktree=release_root / "worktree",
        source_harness_prefix=source_harness,
        source_serving_prefix=source_serving,
        harness_prefix=release_root / "environments" / "harness",
        serving_prefix=release_root / "environments" / "serving",
        conda_executable=_fake_conda(tmp_path),
    )

    assert report["status"] == "dry_run"
    assert report["tag_commit"] == commit
    assert report["clone_contract"] == {
        "command": "conda create --yes --copy --prefix DEST --clone SOURCE",
        "CONDA_ALWAYS_COPY": "true",
        "shared_regular_inode_count": 0,
    }
    assert report["harness_install_contract"]["editable"] is False
    assert report["harness_install_contract"]["index_access"] is False
    assert not release_root.exists()


def test_materialization_rejects_dirty_or_non_tag_source_checkout(tmp_path):
    repository, _commit = _tagged_repository(tmp_path)
    source_harness = _source_prefix(tmp_path, "source-harness")
    source_serving = _source_prefix(tmp_path, "source-serving")
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    release_root = tmp_path / "release"

    with pytest.raises(materialize.MaterializationError, match="clean exact production tag"):
        materialize.materialize_release(
            output_root=release_root,
            source_repository=repository,
            release_worktree=release_root / "worktree",
            source_harness_prefix=source_harness,
            source_serving_prefix=source_serving,
            harness_prefix=release_root / "environments" / "harness",
            serving_prefix=release_root / "environments" / "serving",
            conda_executable=_fake_conda(tmp_path),
        )

    assert not release_root.exists()


def test_copy_verifier_rejects_shared_regular_inodes(tmp_path):
    source = tmp_path / "source"
    copied = tmp_path / "copied"
    linked = tmp_path / "linked"
    source.mkdir()
    copied.mkdir()
    linked.mkdir()
    (source / "payload").write_bytes(b"exact bytes")
    shutil.copy2(source / "payload", copied / "payload")
    os.link(source / "payload", linked / "payload")

    assert materialize.verify_independent_copy(source, copied)[
        "shared_regular_inode_count"
    ] == 0
    with pytest.raises(materialize.MaterializationError, match="shares 1 regular-file"):
        materialize.verify_independent_copy(source, linked)


def test_clone_stage_uses_copy_flag_and_is_idempotent(tmp_path, monkeypatch):
    source = _source_prefix(tmp_path, "source")
    destination = tmp_path / "destination"
    output = tmp_path / "state"
    output.mkdir()
    conda = _fake_conda(tmp_path)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        shutil.copytree(source, destination)
        return ""

    monkeypatch.setattr(materialize, "_run", fake_run)
    monkeypatch.setattr(
        materialize,
        "_clone_identity",
        lambda **kwargs: {
            "source_prefix": str(source),
            "destination_prefix": str(destination),
            "clone_mode": "conda_create_clone_copy",
            "conda_always_copy": True,
            "conda_explicit_sha256": "a" * 64,
            "pip_freeze_sha256": "b" * 64,
            "source_regular_file_count": 2,
            "destination_regular_file_count": 2,
            "shared_regular_inode_count": 0,
        },
    )

    first = materialize._materialize_clone(
        role="harness",
        source=source,
        destination=destination,
        output_root=output,
        conda_executable=conda,
    )
    second = materialize._materialize_clone(
        role="harness",
        source=source,
        destination=destination,
        output_root=output,
        conda_executable=conda,
    )

    assert first == second
    assert len(calls) == 1
    argv, call = calls[0]
    assert argv == [
        str(conda),
        "create",
        "--yes",
        "--copy",
        "--prefix",
        str(destination),
        "--clone",
        str(source),
    ]
    assert call["env"]["CONDA_ALWAYS_COPY"] == "true"


def test_harness_install_is_noneditable_no_deps_no_index_and_stage_bound(
    tmp_path, monkeypatch
):
    harness = tmp_path / "harness"
    (harness / "bin").mkdir(parents=True)
    (harness / "bin" / "python").write_text("python\n", encoding="utf-8")
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    output = tmp_path / "state"
    output.mkdir()
    git_identity = {
        "git_commit": "1" * 40,
        "git_tag": materialize.REQUIRED_TAG,
        "source_tree_sha256": "2" * 64,
    }
    binding = {
        "name": "agents_scaling",
        "version": "0.1.0",
        "direct_url": worktree.resolve().as_uri(),
        "source_path": str(worktree),
        "source_git_commit": "1" * 40,
        "source_tree_sha256": "2" * 64,
        "installed_distribution_sha256": "3" * 64,
        "installed_file_count": 10,
        "editable": False,
    }
    probe = {
        "prefix": str(harness),
        "module_path": str(harness / "lib" / "agents_scaling" / "__init__.py"),
        "version": "0.1.0",
        "direct_url": {"url": worktree.resolve().as_uri(), "dir_info": {}},
    }
    calls = []
    monkeypatch.setattr(
        materialize,
        "_run",
        lambda argv, **kwargs: calls.append((list(argv), kwargs)) or "",
    )
    monkeypatch.setattr(
        materialize.freeze,
        "_pip_lock_material",
        lambda *args, **kwargs: (["agents_scaling @ " + worktree.as_uri()], binding),
    )
    monkeypatch.setattr(materialize, "verify_harness_import", lambda *args: probe)
    monkeypatch.setattr(
        materialize,
        "_archive_release_build_evidence",
        lambda **kwargs: {
            "generated_paths": [],
            "archive_path": None,
            "archive_inventory_sha256": None,
        },
    )
    monkeypatch.setattr(
        materialize.freeze,
        "verify_clean_exact_tag",
        lambda path: dict(git_identity),
    )

    first = materialize._materialize_harness_package(
        harness_prefix=harness,
        release_worktree=worktree,
        output_root=output,
        git_identity=git_identity,
    )
    second = materialize._materialize_harness_package(
        harness_prefix=harness,
        release_worktree=worktree,
        output_root=output,
        git_identity=git_identity,
    )

    assert first == second
    assert len(calls) == 2
    install = calls[1][0]
    assert "--no-index" in install
    assert "--no-deps" in install
    assert "--no-build-isolation" in install
    assert "--no-compile" in install
    assert "--force-reinstall" in install
    assert install[-1] == str(worktree)
    assert first["package_binding"] == binding
    assert first["install_contract"]["editable"] is False


def test_isolated_import_must_resolve_inside_prefix_and_be_noneditable(
    tmp_path, monkeypatch
):
    prefix = tmp_path / "harness"
    worktree = tmp_path / "worktree"
    module = prefix / "lib" / "python3.11" / "site-packages" / "agents_scaling" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("\n", encoding="utf-8")
    worktree.mkdir()
    resources = _release_runtime_resources(worktree)
    payload = {
        "prefix": str(prefix),
        "module_path": str(module),
        "version": "0.1.0",
        "direct_url": {"url": worktree.resolve().as_uri(), "dir_info": {}},
        "runtime_resources": resources,
    }
    monkeypatch.setattr(
        materialize, "_run", lambda *args, **kwargs: json.dumps(payload)
    )

    assert materialize.verify_harness_import(prefix, worktree) == payload

    payload["runtime_resources"]["fleet_replica_count"] = 21
    with pytest.raises(materialize.MaterializationError, match="wrong serving fleet"):
        materialize.verify_harness_import(prefix, worktree)
    payload["runtime_resources"]["fleet_replica_count"] = 22

    payload["direct_url"]["dir_info"] = {"editable": True}
    with pytest.raises(materialize.MaterializationError, match="editable"):
        materialize.verify_harness_import(prefix, worktree)


def test_unstaged_partial_clone_is_never_adopted(tmp_path):
    source = _source_prefix(tmp_path, "source")
    destination = _source_prefix(tmp_path, "partial-destination")
    output = tmp_path / "state"
    output.mkdir()

    with pytest.raises(materialize.MaterializationError, match="move it to quarantine"):
        materialize._materialize_clone(
            role="serving",
            source=source,
            destination=destination,
            output_root=output,
            conda_executable=_fake_conda(tmp_path),
        )


def test_setuptools_byproduct_is_atomically_retained_and_worktree_recovers_clean(
    tmp_path,
):
    worktree, _commit = _tagged_repository(tmp_path)
    generated = worktree / "src" / "agents_scaling.egg-info"
    generated.mkdir(parents=True)
    (generated / "PKG-INFO").write_text("Name: agents_scaling\n", encoding="utf-8")
    output = tmp_path / "materialization"
    output.mkdir()

    evidence = materialize._archive_release_build_evidence(
        release_worktree=worktree, output_root=output
    )

    assert not generated.exists()
    archive = Path(evidence["archive_path"])
    assert archive.is_dir()
    assert (
        archive / "src" / "agents_scaling.egg-info" / "PKG-INFO"
    ).read_text(encoding="utf-8") == "Name: agents_scaling\n"
    materialize._verify_build_evidence(evidence, output_root=output)
    assert (
        materialize.freeze.verify_clean_exact_tag(worktree)["git_tag"]
        == materialize.REQUIRED_TAG
    )
