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


def test_capture_binding_preserves_separate_policy_identities(
    tmp_path, monkeypatch
):
    capture_root = (tmp_path / "capture").resolve()
    harness_seed = (capture_root / "seeds" / "harness").resolve()
    serving_seed = (capture_root / "seeds" / "serving").resolve()
    report = {
        "release_id": materialize.RELEASE_ID,
        "capture_id": "1" * 64,
        "capture_marker_sha256": "2" * 64,
        "seed_prefixes": {
            "harness": str(harness_seed),
            "serving": str(serving_seed),
        },
        "ownership_policy_path": "/sealed/ownership-policy.json",
        "ownership_policy_sha256": "3" * 64,
        "integrity_normalization_policy_path": (
            "/sealed/integrity-normalization-policy.json"
        ),
        "integrity_normalization_policy_sha256": "4" * 64,
        "reconciliation_incident_path": "/sealed/reconciliation-incident.json",
        "reconciliation_incident_sha256": "5" * 64,
        "recovered_record_path": "/sealed/recovered-setuptools-record.json",
        "recovered_record_sha256": "6" * 64,
        "stage_records": {"harness": {}, "serving": {}},
    }
    monkeypatch.setattr(
        materialize.capture, "verify_capture", lambda root: dict(report)
    )

    binding = materialize._verified_environment_capture_binding(
        capture_root=capture_root,
        harness_seed=harness_seed,
        serving_seed=serving_seed,
    )

    assert binding["ownership_policy_sha256"] == "3" * 64
    assert binding["integrity_normalization_policy_sha256"] == "4" * 64
    assert (
        binding["ownership_policy_path"]
        != binding["integrity_normalization_policy_path"]
    )

    invalid = dict(report)
    invalid.pop("integrity_normalization_policy_sha256")
    monkeypatch.setattr(
        materialize.capture, "verify_capture", lambda root: dict(invalid)
    )
    with pytest.raises(
        materialize.MaterializationError,
        match="does not bind the supplied normalized seeds",
    ):
        materialize._verified_environment_capture_binding(
            capture_root=capture_root,
            harness_seed=harness_seed,
            serving_seed=serving_seed,
        )


def test_command_environment_rejects_hostile_pip_and_conda_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = {
        "PIP_INDEX_URL": "https://attacker.invalid/simple",
        "PIP_TARGET": "/tmp/attacker-target",
        "PIP_PREFIX": "/tmp/attacker-prefix",
        "PIP_CONFIG_FILE": "/tmp/attacker-pip.conf",
        "CONDA_PREFIX": "/tmp/attacker-prefix",
        "CONDA_SOLVER": "classic",
        "CONDA_PLUGINS_AUTO_ACCEPT_TOS": "yes",
        "CONDARC": "/tmp/attacker-condarc",
        "LD_PRELOAD": "/tmp/attacker.so",
    }
    for name, value in hostile.items():
        monkeypatch.setenv(name, value)
    environment = materialize._command_environment()
    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert environment["PIP_NO_INDEX"] == "1"
    assert environment["PIP_NO_INPUT"] == "1"
    assert environment["CONDARC"] == os.devnull
    assert environment["CONDA_NO_PLUGINS"] == "true"
    assert environment["CONDA_OFFLINE"] == "true"
    assert "PIP_INDEX_URL" not in environment
    assert "PIP_TARGET" not in environment
    assert "PIP_PREFIX" not in environment
    assert "CONDA_PREFIX" not in environment
    assert "CONDA_SOLVER" not in environment
    assert "CONDA_PLUGINS_AUTO_ACCEPT_TOS" not in environment
    assert "LD_PRELOAD" not in environment


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
    _run(
        "git",
        "tag",
        "-a",
        materialize.REQUIRED_TAG,
        "-m",
        "schema-5 materialization fixture",
        cwd=repository,
    )
    return repository, _run("git", "rev-parse", "HEAD", cwd=repository)


def _source_prefix(tmp_path: Path, name: str) -> Path:
    prefix = tmp_path / name
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "conda-meta" / "history").write_text("created\n", encoding="utf-8")
    (prefix / "conda-meta" / "python-3.11-test.json").write_text(
        json.dumps(
            {
                "name": "python",
                "version": "3.11",
                "url": "https://conda.example.invalid/python-3.11-test.conda",
                "sha256": "1" * 64,
            }
        ),
        encoding="utf-8",
    )
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


def test_materialization_dry_run_is_read_only_and_plans_copy_semantics(
    tmp_path, monkeypatch
):
    repository, commit = _tagged_repository(tmp_path)
    capture_root = tmp_path / "capture"
    source_harness = _source_prefix(capture_root / "seeds", "harness")
    source_serving = _source_prefix(capture_root / "seeds", "serving")
    monkeypatch.setattr(
        materialize,
        "_verified_environment_capture_binding",
        lambda **kwargs: {
            "release_id": materialize.RELEASE_ID,
            "capture_id": "c" * 64,
        },
    )
    release_root = tmp_path / "release"

    report = materialize.materialize_release(
        output_root=release_root,
        environment_capture_root=capture_root,
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
        "command": (
            "conda create --yes --copy --offline --no-default-packages "
            "--prefix DEST --clone NORMALIZED_SEED"
        ),
        "CONDA_ALWAYS_COPY": "true",
        "CONDA_OFFLINE": "true",
        "CONDA_PIP_INTEROP_ENABLED": "false",
        "CONDA_ADD_PIP_AS_PYTHON_DEPENDENCY": "false",
        "release_local_package_cache": str(release_root / "conda-package-cache"),
        "shared_regular_inode_count": 0,
        "source_prefix_target_symlink_count": 0,
        "unresolvable_symlink_count": 0,
    }
    assert report["harness_install_contract"]["editable"] is False
    assert report["harness_install_contract"]["index_access"] is False
    assert not release_root.exists()


def test_materialization_rejects_dirty_or_non_tag_source_checkout(
    tmp_path, monkeypatch
):
    repository, _commit = _tagged_repository(tmp_path)
    capture_root = tmp_path / "capture"
    source_harness = _source_prefix(capture_root / "seeds", "harness")
    source_serving = _source_prefix(capture_root / "seeds", "serving")
    monkeypatch.setattr(
        materialize,
        "_verified_environment_capture_binding",
        lambda **kwargs: {
            "release_id": materialize.RELEASE_ID,
            "capture_id": "c" * 64,
        },
    )
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    release_root = tmp_path / "release"

    with pytest.raises(materialize.MaterializationError, match="clean exact production tag"):
        materialize.materialize_release(
            output_root=release_root,
            environment_capture_root=capture_root,
            source_repository=repository,
            release_worktree=release_root / "worktree",
            source_harness_prefix=source_harness,
            source_serving_prefix=source_serving,
            harness_prefix=release_root / "environments" / "harness",
            serving_prefix=release_root / "environments" / "serving",
            conda_executable=_fake_conda(tmp_path),
        )

    assert not release_root.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("clone_contract", "clone/install contract drifted"),
        ("source_tree_sha256", "worktree commit drifted"),
    ),
)
def test_verifier_rejects_rehashed_materialization_contract_substitution(
    tmp_path, monkeypatch, mutation, message
):
    root = tmp_path / "materialization"
    root.mkdir()
    paths = {
        "source_repository": str(tmp_path / "retired-source"),
        "environment_capture_root": str(tmp_path / "capture"),
        "release_worktree": str(tmp_path / "worktree"),
        "source_harness_prefix": str(tmp_path / "harness-seed"),
        "source_serving_prefix": str(tmp_path / "serving-seed"),
        "harness_prefix": str(tmp_path / "harness"),
        "serving_prefix": str(tmp_path / "serving"),
    }
    for key, value in paths.items():
        if key != "source_repository":
            Path(value).mkdir(parents=True)
    capture_binding = {
        "release_id": materialize.RELEASE_ID,
        "capture_id": "c" * 64,
    }
    git_identity = {
        "git_commit": "1" * 40,
        "git_tag": materialize.REQUIRED_TAG,
        "source_tree_sha256": "2" * 64,
    }
    conda_tool = {
        "path": str(tmp_path / "archived-conda"),
        "sha256": "3" * 64,
    }
    marker = {
        "schema_version": materialize.SCHEMA_VERSION,
        "release_id": materialize.RELEASE_ID,
        "git_tag": materialize.REQUIRED_TAG,
        "tag_commit": git_identity["git_commit"],
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "paths": paths,
        "output_root": str(root),
        "conda_creation_tool": conda_tool,
        "environment_capture": capture_binding,
        "clone_contract": {
            "command": (
                "conda create --yes --copy --offline --no-default-packages "
                "--prefix DEST --clone NORMALIZED_SEED"
            ),
            "CONDA_ALWAYS_COPY": "true",
            "CONDA_OFFLINE": "true",
            "CONDA_PIP_INTEROP_ENABLED": "false",
            "CONDA_ADD_PIP_AS_PYTHON_DEPENDENCY": "false",
            "release_local_package_cache": str(
                root / "conda-package-cache"
            ),
            "shared_regular_inode_count": 0,
            "source_prefix_target_symlink_count": 0,
            "unresolvable_symlink_count": 0,
        },
        "harness_install_contract": {
            "source": paths["release_worktree"],
            "editable": False,
            "dependencies_installed": False,
            "build_isolation": False,
            "bytecode_compiled": False,
            "index_access": False,
            "isolated_import_required": True,
        },
        "complete": True,
        "publication_protocol": "stage_records_fsync_marker_last",
        "stage_records": {},
        "serving_pip_check": {},
    }
    if mutation == "clone_contract":
        marker["clone_contract"]["CONDA_OFFLINE"] = "false"
    else:
        marker["source_tree_sha256"] = "f" * 64
    marker["materialization_id"] = materialize._sha256_bytes(
        materialize._json_bytes(marker)
    )
    marker_path = root / materialize.COMPLETE_MARKER
    marker_path.write_bytes(materialize._json_bytes(marker))
    marker_path.chmod(0o444)
    monkeypatch.setattr(
        materialize,
        "_verified_environment_capture_binding",
        lambda **kwargs: dict(capture_binding),
    )
    monkeypatch.setattr(
        materialize.freeze,
        "verify_clean_exact_tag",
        lambda path: dict(git_identity),
    )

    with pytest.raises(materialize.MaterializationError, match=message):
        materialize.verify_materialization(root)


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


def test_safe_destination_rejects_lexical_leaf_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(materialize.MaterializationError, match="symlinked destination"):
        materialize._safe_destination(alias, description="destination")


def test_clone_symlink_audit_records_internal_link_and_rejects_external(tmp_path):
    source_harness = _source_prefix(tmp_path, "source-harness")
    source_serving = _source_prefix(tmp_path, "source-serving")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "internal-target").write_text("inside\n", encoding="utf-8")
    (destination / "internal-link").symlink_to("internal-target")
    external = tmp_path / "external"
    external.mkdir()
    (external / "target").write_text("outside\n", encoding="utf-8")
    report = materialize.verify_clone_symlinks(
        destination,
        source_prefixes=(source_serving, source_harness),
    )

    assert report == {
        "symlink_audit_source_prefixes": sorted(
            (str(source_harness.resolve()), str(source_serving.resolve()))
        ),
        "destination_symlink_count": 1,
        "destination_internal_symlink_count": 1,
        "destination_external_symlink_count": 0,
        "source_prefix_target_symlink_count": 0,
        "unresolvable_symlink_count": 0,
    }

    (destination / "external-relative").symlink_to(
        os.path.relpath(external / "target", destination)
    )
    with pytest.raises(materialize.MaterializationError, match="unpinned external"):
        materialize.verify_clone_symlinks(
            destination,
            source_prefixes=(source_serving, source_harness),
        )


def test_clone_symlink_audit_rejects_transitive_source_prefix_target(tmp_path):
    source_harness = _source_prefix(tmp_path, "source-harness")
    source_serving = _source_prefix(tmp_path, "source-serving")
    destination = tmp_path / "destination"
    destination.mkdir()
    bridge = tmp_path / "bridge"
    bridge.symlink_to(source_serving / "bin" / "python")
    (destination / "python").symlink_to(bridge)

    with pytest.raises(
        materialize.MaterializationError,
        match="unpinned external|mutable source prefix",
    ):
        materialize.verify_clone_symlinks(
            destination,
            source_prefixes=(source_harness, source_serving),
        )


def test_clone_symlink_audit_rejects_external_hop_that_reenters_clone(tmp_path):
    source = _source_prefix(tmp_path, "source")
    destination = tmp_path / "destination"
    destination.mkdir()
    target = destination / "target"
    target.write_text("inside\n", encoding="utf-8")
    external_bridge = tmp_path / "external-bridge"
    external_bridge.symlink_to(target)
    (destination / "link").symlink_to(external_bridge)

    with pytest.raises(
        materialize.MaterializationError,
        match="unpinned external symlink dependency",
    ):
        materialize.verify_clone_symlinks(
            destination,
            source_prefixes=(source,),
        )


def test_clone_symlink_audit_rejects_source_owned_hop_that_resolves_outward(tmp_path):
    source = _source_prefix(tmp_path, "source")
    destination = tmp_path / "destination"
    destination.mkdir()
    external = tmp_path / "external"
    external.write_text("outside\n", encoding="utf-8")
    (source / "bridge").symlink_to(external)
    (destination / "link").symlink_to(source / "bridge")

    with pytest.raises(
        materialize.MaterializationError,
        match="dependency enters a mutable source prefix",
    ):
        materialize.verify_clone_symlinks(
            destination,
            source_prefixes=(source,),
        )


@pytest.mark.parametrize("target", ("missing-target", "loop"))
def test_clone_symlink_audit_rejects_unresolvable_link(tmp_path, target):
    source = _source_prefix(tmp_path, "source")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "loop").symlink_to(target)

    with pytest.raises(materialize.MaterializationError, match="unresolvable symlink"):
        materialize.verify_clone_symlinks(
            destination,
            source_prefixes=(source,),
        )


def test_clone_identity_records_symlink_independence_contract(tmp_path, monkeypatch):
    source_harness = _source_prefix(tmp_path, "source-harness")
    source_serving = _source_prefix(tmp_path, "source-serving")
    destination = tmp_path / "destination"
    (destination / "bin").mkdir(parents=True)
    (destination / "bin" / "python").write_text("python\n", encoding="utf-8")
    (destination / "bin" / "python-link").symlink_to("python")
    monkeypatch.setattr(
        materialize.freeze, "_conda_lock_from_records", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(materialize, "_raw_pip_freeze", lambda *args, **kwargs: [])

    report = materialize._clone_identity(
        source=source_harness,
        destination=destination,
        conda_executable=_fake_conda(tmp_path),
        source_prefixes=(source_serving, source_harness),
    )

    assert report["symlink_audit_source_prefixes"] == sorted(
        (str(source_harness.resolve()), str(source_serving.resolve()))
    )
    assert report["destination_symlink_count"] == 1
    assert report["destination_internal_symlink_count"] == 1
    assert report["destination_external_symlink_count"] == 0
    assert report["source_prefix_target_symlink_count"] == 0
    assert report["unresolvable_symlink_count"] == 0


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
            "clone_mode": "conda_create_clone_copy_offline_normalized_seed",
            "conda_always_copy": True,
            "conda_offline": True,
            "conda_pip_interop_enabled": False,
            "conda_explicit_sha256": "a" * 64,
            "pip_freeze_sha256": "b" * 64,
            "source_regular_file_count": 2,
            "destination_regular_file_count": 2,
            "shared_regular_inode_count": 0,
            "symlink_audit_source_prefixes": [str(source.resolve())],
            "destination_symlink_count": 0,
            "destination_internal_symlink_count": 0,
            "destination_external_symlink_count": 0,
            "source_prefix_target_symlink_count": 0,
            "unresolvable_symlink_count": 0,
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
        "--offline",
        "--no-default-packages",
        "--prefix",
        str(destination),
        "--clone",
        str(source),
    ]
    assert call["env"]["CONDA_ALWAYS_COPY"] == "true"
    assert call["env"]["CONDA_OFFLINE"] == "true"
    assert call["env"]["CONDA_PIP_INTEROP_ENABLED"] == "false"
    assert call["env"]["CONDA_PKGS_DIRS"] == str(output / "conda-package-cache")


@pytest.mark.parametrize("mutation", ("regular_file", "internal_symlink"))
def test_clone_stage_resume_rejects_live_identity_drift(tmp_path, monkeypatch, mutation):
    source = _source_prefix(tmp_path, "source")
    destination = tmp_path / "destination"
    shutil.copytree(source, destination)
    output = tmp_path / "state"
    output.mkdir()
    conda = _fake_conda(tmp_path)
    monkeypatch.setattr(
        materialize.freeze, "_conda_lock_from_records", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(materialize, "_raw_pip_freeze", lambda *args, **kwargs: [])
    identity = materialize._clone_identity(
        source=source,
        destination=destination,
        conda_executable=conda,
        source_prefixes=(source,),
    )
    materialize._write_stage(
        output,
        "serving_clone",
        materialize._stage_payload("serving_clone", identity),
    )
    if mutation == "regular_file":
        (destination / "new-file").write_text("drift\n", encoding="utf-8")
        message = "clone identity drifted"
    else:
        (destination / "target").write_text("target\n", encoding="utf-8")
        (destination / "link").symlink_to("target")
        message = "symlink audit drifted"

    with pytest.raises(materialize.MaterializationError, match=message):
        materialize._materialize_clone(
            role="serving",
            source=source,
            destination=destination,
            output_root=output,
            conda_executable=conda,
            source_prefixes=(source,),
        )


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
    build_evidence = {
        "generated_paths": [],
        "archive_path": None,
        "archive_inventory_sha256": None,
    }
    build_evidence_marker = {
        "filename": materialize.BUILD_EVIDENCE_COMPLETE_MARKER,
        "sha256": "7" * 64,
        "record_sha256": "8" * 64,
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
        "_verify_pip_check",
        lambda prefix: {
            "command": "python -I -m pip check",
            "stdout": "No broken requirements found.",
            "clean": True,
        },
    )
    monkeypatch.setattr(
        materialize,
        "_archive_release_build_evidence",
        lambda **kwargs: dict(build_evidence),
    )
    monkeypatch.setattr(
        materialize,
        "_load_completed_build_evidence",
        lambda **kwargs: (dict(build_evidence), dict(build_evidence_marker)),
    )
    monkeypatch.setattr(
        materialize.freeze,
        "verify_clean_exact_tag",
        lambda path: dict(git_identity),
    )
    post_install_identity = {
        "conda_explicit_sha256": "4" * 64,
        "pip_freeze_sha256": "5" * 64,
        "content_inventory_sha256": "6" * 64,
        "content_inventory_entry_count": 3,
        "content_inventory_file_count": 1,
        "content_inventory_symlink_count": 0,
        "symlink_audit_source_prefixes": [str(worktree.resolve())],
        "destination_symlink_count": 0,
        "destination_internal_symlink_count": 0,
        "destination_external_symlink_count": 0,
        "source_prefix_target_symlink_count": 0,
        "unresolvable_symlink_count": 0,
    }
    monkeypatch.setattr(
        materialize,
        "_post_install_identity",
        lambda **kwargs: (dict(post_install_identity), binding),
    )
    conda = _fake_conda(tmp_path)

    first = materialize._materialize_harness_package(
        harness_prefix=harness,
        release_worktree=worktree,
        output_root=output,
        git_identity=git_identity,
        conda_executable=conda,
        source_prefixes=(worktree,),
    )
    second = materialize._materialize_harness_package(
        harness_prefix=harness,
        release_worktree=worktree,
        output_root=output,
        git_identity=git_identity,
        conda_executable=conda,
        source_prefixes=(worktree,),
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
    loaded, marker = materialize._load_completed_build_evidence(
        release_worktree=worktree,
        output_root=output,
    )
    assert loaded == evidence
    assert marker["filename"] == materialize.BUILD_EVIDENCE_COMPLETE_MARKER
    assert len(marker["sha256"]) == 64
    assert (
        materialize.freeze.verify_clean_exact_tag(worktree)["git_tag"]
        == materialize.REQUIRED_TAG
    )


def test_completed_build_evidence_binds_exact_archive_root_and_file_set(
    tmp_path,
):
    output = tmp_path / "materialization"
    archive = output / "build_evidence"
    package_info = archive / "src" / "agents_scaling.egg-info" / "PKG-INFO"
    package_info.parent.mkdir(parents=True)
    package_info.write_text("Name: agents_scaling\n", encoding="utf-8")
    inventory = materialize.freeze.directory_inventory(archive)
    valid = {
        "generated_paths": ["src/agents_scaling.egg-info/PKG-INFO"],
        "archive_path": str(archive),
        "archive_inventory_sha256": inventory["inventory_sha256"],
    }

    materialize._verify_build_evidence(valid, output_root=output)

    wrong_paths = dict(valid)
    wrong_paths["generated_paths"] = [
        "src/agents_scaling.egg-info/NOT-ARCHIVED"
    ]
    with pytest.raises(
        materialize.MaterializationError,
        match="does not contain the exact generated paths",
    ):
        materialize._verify_build_evidence(wrong_paths, output_root=output)

    nested = archive / "src"
    nested_inventory = materialize.freeze.directory_inventory(nested)
    wrong_root = {
        "generated_paths": ["agents_scaling.egg-info/PKG-INFO"],
        "archive_path": str(nested),
        "archive_inventory_sha256": nested_inventory["inventory_sha256"],
    }
    with pytest.raises(
        materialize.MaterializationError,
        match="archive is invalid|unsafe or unexpected",
    ):
        materialize._verify_build_evidence(wrong_root, output_root=output)


def test_completed_build_evidence_rejects_residual_ignored_byproduct(tmp_path):
    worktree, _commit = _tagged_repository(tmp_path)
    output = tmp_path / "materialization"
    output.mkdir()
    materialize._publish_build_evidence_marker(
        release_worktree=worktree,
        output_root=output,
        evidence={
            "generated_paths": [],
            "archive_path": None,
            "archive_inventory_sha256": None,
        },
    )
    residual = worktree / "src" / "agents_scaling.egg-info" / "PKG-INFO"
    residual.parent.mkdir(parents=True)
    residual.write_text("Name: agents_scaling\n", encoding="utf-8")

    with pytest.raises(
        materialize.MaterializationError,
        match="left release-worktree drift",
    ):
        materialize._load_completed_build_evidence(
            release_worktree=worktree,
            output_root=output,
        )


def test_harness_package_retry_adopts_completed_build_evidence_without_reinstall(
    tmp_path,
    monkeypatch,
):
    worktree, _commit = _tagged_repository(tmp_path)
    git_identity = materialize.freeze.verify_clean_exact_tag(worktree)
    generated = worktree / "src" / "agents_scaling.egg-info"
    generated.mkdir(parents=True)
    (generated / "PKG-INFO").write_text(
        "Name: agents_scaling\n", encoding="utf-8"
    )
    harness = tmp_path / "harness"
    (harness / "bin").mkdir(parents=True)
    (harness / "bin" / "python").write_text("python\n", encoding="utf-8")
    output = tmp_path / "materialization"
    output.mkdir()
    binding = {
        "name": "agents_scaling",
        "version": "0.1.0",
        "editable": False,
    }
    post_install_identity = {
        "conda_explicit_sha256": "1" * 64,
        "pip_freeze_sha256": "2" * 64,
    }
    calls: list[list[str]] = []
    real_run = materialize._run

    def track_package_commands(argv, **kwargs):
        if str(argv[0]) == str(harness / "bin" / "python"):
            calls.append(list(argv))
            return ""
        return real_run(argv, **kwargs)

    monkeypatch.setattr(
        materialize,
        "_run",
        track_package_commands,
    )
    monkeypatch.setattr(
        materialize,
        "_post_install_identity",
        lambda **kwargs: (dict(post_install_identity), dict(binding)),
    )
    monkeypatch.setattr(
        materialize,
        "verify_harness_import",
        lambda *args: {"verified": True},
    )
    monkeypatch.setattr(
        materialize,
        "_verify_pip_check",
        lambda prefix: {
            "command": "python -I -m pip check",
            "stdout": "No broken requirements found.",
            "clean": True,
        },
    )
    real_write_stage = materialize._write_stage

    def crash_before_package_stage(output_root, stage, payload):
        assert stage == "harness_package"
        raise RuntimeError("simulated crash before package stage")

    monkeypatch.setattr(materialize, "_write_stage", crash_before_package_stage)
    with pytest.raises(RuntimeError, match="simulated crash"):
        materialize._materialize_harness_package(
            harness_prefix=harness,
            release_worktree=worktree,
            output_root=output,
            git_identity=git_identity,
            conda_executable=_fake_conda(tmp_path),
            source_prefixes=(worktree,),
        )

    assert len(calls) == 2
    assert (output / materialize.BUILD_EVIDENCE_COMPLETE_MARKER).is_file()
    assert not (output / materialize.STAGE_FILENAMES["harness_package"]).exists()

    monkeypatch.setattr(materialize, "_write_stage", real_write_stage)
    archived_package_info = (
        output
        / "build_evidence"
        / "src"
        / "agents_scaling.egg-info"
        / "PKG-INFO"
    )
    pristine_evidence = archived_package_info.read_bytes()
    archived_package_info.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(materialize.MaterializationError, match="archive drifted"):
        materialize._materialize_harness_package(
            harness_prefix=harness,
            release_worktree=worktree,
            output_root=output,
            git_identity=git_identity,
            conda_executable=_fake_conda(tmp_path),
            source_prefixes=(worktree,),
        )
    assert len(calls) == 2
    archived_package_info.write_bytes(pristine_evidence)

    recovered = materialize._materialize_harness_package(
        harness_prefix=harness,
        release_worktree=worktree,
        output_root=output,
        git_identity=git_identity,
        conda_executable=_fake_conda(tmp_path),
        source_prefixes=(worktree,),
    )

    assert len(calls) == 2
    assert recovered["build_evidence_marker"]["filename"] == (
        materialize.BUILD_EVIDENCE_COMPLETE_MARKER
    )
