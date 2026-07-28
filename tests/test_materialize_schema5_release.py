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


def test_production_cli_accepts_only_sealed_toolchain_root() -> None:
    parser = materialize._build_parser()
    materialize_parser = next(
        action.choices["materialize"]
        for action in parser._actions
        if hasattr(action, "choices") and action.choices
    )
    options = {
        option
        for action in materialize_parser._actions
        for option in action.option_strings
    }
    assert "--conda-toolchain-root" in options
    assert "--conda-executable" not in options
    assert "--expected-package-cache-seed-input-json" in options
    expected_action = next(
        action
        for action in materialize_parser._actions
        if "--expected-package-cache-seed-input-json"
        in action.option_strings
    )
    assert expected_action.required is True


def _toolchain_binding(value: str | Path) -> dict:
    executable = Path(value).resolve()
    payload = executable.read_bytes()
    root = executable
    return {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r7-offline-conda-toolchain-v1",
        "release_tag": materialize.REQUIRED_TAG,
        "chain_namespace": "schema5-v1.2-r7",
        "toolchain_root": str(root),
        "base_prefix": str(executable.parent),
        "completion_marker": {
            "path": str(root / "CONDA_TOOLCHAIN_COMPLETE.json"),
            "sha256": "1" * 64,
            "size": 1,
        },
        "marker_id": "2" * 64,
        "installer_contract": {
            "filename": "Miniforge3-Linux-x86_64.sh",
            "release": "Miniforge3-25.11.0-1",
            "sha256": "3" * 64,
            "conda_version": "25.11.0",
        },
        "intent_id": "4" * 64,
        "conda_executable": {
            "path": str(executable),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "mode": executable.stat().st_mode & 0o777,
            "link_count": executable.stat().st_nlink,
        },
        "runtime_identity_sha256": "5" * 64,
        "complete_prefix_inventory_sha256": "6" * 64,
        "read_only_probes": {"probe_count": 2},
        "binding_id": "7" * 64,
    }


@pytest.fixture(autouse=True)
def _mock_verified_conda_toolchain(monkeypatch):
    monkeypatch.setattr(
        materialize.conda_toolchain,
        "verified_conda_toolchain_binding",
        lambda root, exercise=True: _toolchain_binding(root),
    )


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
        "PYTHONWARNINGS": "error",
        "CONDA_PREFIX": "/tmp/attacker-prefix",
        "CONDA_SOLVER": "classic",
        "CONDA_PLUGINS_AUTO_ACCEPT_TOS": "yes",
        "CONDARC": "/tmp/attacker-condarc",
        "LD_PRELOAD": "/tmp/attacker.so",
        "PATH": "/tmp/attacker-bin",
        "GIT_DIR": "/tmp/attacker-git",
        "GIT_CONFIG_PARAMETERS": "'core.hooksPath=/tmp/attacker-hooks'",
        "GIT_REPLACE_REF_BASE": "refs/attacker",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": "/tmp/attacker-hooks",
        "SBATCH_PARTITION": "attacker",
        "SQUEUE_FORMAT": "attacker",
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
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert "PIP_INDEX_URL" not in environment
    assert "PIP_TARGET" not in environment
    assert "PIP_PREFIX" not in environment
    assert "PYTHONWARNINGS" not in environment
    assert "CONDA_PREFIX" not in environment
    assert "CONDA_SOLVER" not in environment
    assert "CONDA_PLUGINS_AUTO_ACCEPT_TOS" not in environment
    assert "LD_PRELOAD" not in environment
    assert "GIT_DIR" not in environment
    assert "GIT_CONFIG_PARAMETERS" not in environment
    assert "GIT_REPLACE_REF_BASE" not in environment
    assert "GIT_CONFIG_KEY_0" not in environment
    assert "GIT_CONFIG_VALUE_0" not in environment
    assert "SBATCH_PARTITION" not in environment
    assert "SQUEUE_FORMAT" not in environment


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
                "build": "test",
                "fn": "python-3.11-test.conda",
                "url": "https://conda.example.invalid/python-3.11-test.conda",
                "sha256": "1" * 64,
            }
        ),
        encoding="utf-8",
    )
    (prefix / "bin").mkdir()
    (prefix / "bin" / "python").write_text("python\n", encoding="utf-8")
    return prefix


def _source_package_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "source-package-cache"
    extracted = cache / "python-3.11-test"
    (extracted / "info").mkdir(parents=True)
    identity = {
        "name": "python",
        "version": "3.11",
        "build": "test",
    }
    (extracted / "info" / "index.json").write_text(
        json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8"
    )
    (extracted / "info" / "repodata_record.json").write_text(
        json.dumps(
            {
                **identity,
                "fn": "python-3.11-test.conda",
                "url": "https://conda.example.invalid/python-3.11-test.conda",
                "sha256": "1" * 64,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (extracted / "payload").write_text("python package\n", encoding="utf-8")
    (cache / "cache").mkdir()
    (cache / "cache" / "channel.json").write_text(
        '{"packages": {}}\n', encoding="utf-8"
    )
    (cache / "urls").write_text("", encoding="utf-8")
    (cache / "urls.txt").write_text(
        "https://conda.example.invalid/python-3.11-test.conda\n",
        encoding="utf-8",
    )
    return cache


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
    source_package_cache = _source_package_cache(tmp_path)
    expected_cache_input = (
        materialize.selected_package_cache_input_binding(
            source_package_cache=source_package_cache,
            harness_seed=source_harness,
            serving_seed=source_serving,
        )
    )

    report = materialize.materialize_release(
        output_root=release_root,
        environment_capture_root=capture_root,
        source_repository=repository,
        release_worktree=release_root / "worktree",
        source_harness_prefix=source_harness,
        source_serving_prefix=source_serving,
        source_package_cache=source_package_cache,
        harness_prefix=release_root / "environments" / "harness",
        serving_prefix=release_root / "environments" / "serving",
        conda_toolchain_root=_fake_conda(tmp_path),
        expected_package_cache_seed_input=expected_cache_input,
    )

    assert report["status"] == "dry_run"
    assert report["tag_commit"] == commit
    assert report["package_cache_seed_input"] == {
        "source_package_cache": str(
            (tmp_path / "source-package-cache").resolve()
        ),
        "inventory_sha256": report["package_cache_seed_input"][
            "inventory_sha256"
        ],
        "inventory_entry_count": 9,
        "inventory_file_count": 6,
        "inventory_total_file_bytes": report["package_cache_seed_input"][
            "inventory_total_file_bytes"
        ],
        "requirements_sha256": report["package_cache_seed_input"][
            "requirements_sha256"
        ],
        "required_package_count": 1,
        "archive_count": 0,
        "selected_top_level_entries": [
            "cache",
            "python-3.11-test",
            "urls",
            "urls.txt",
        ],
        "input_id": report["package_cache_seed_input"]["input_id"],
    }
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
        "immutable_package_cache_seed": str(
            release_root / "conda-package-cache-seed"
        ),
        "package_cache_seed_protocol": materialize.PACKAGE_CACHE_SEED_PROTOCOL,
        "shared_regular_inode_count": 0,
        "source_prefix_target_symlink_count": 0,
        "unresolvable_symlink_count": 0,
    }
    assert report["harness_install_contract"]["editable"] is False
    assert report["harness_install_contract"]["index_access"] is False
    assert not release_root.exists()


def test_materialization_rejects_cache_drift_from_expected_binding_before_output(
    tmp_path, monkeypatch
):
    repository, _commit = _tagged_repository(tmp_path)
    capture_root = tmp_path / "capture"
    source_harness = _source_prefix(capture_root / "seeds", "harness")
    source_serving = _source_prefix(capture_root / "seeds", "serving")
    source_package_cache = _source_package_cache(tmp_path)
    expected_cache_input = (
        materialize.selected_package_cache_input_binding(
            source_package_cache=source_package_cache,
            harness_seed=source_harness,
            serving_seed=source_serving,
        )
    )
    (source_package_cache / "urls.txt").write_text(
        (source_package_cache / "urls.txt").read_text(encoding="utf-8")
        + "https://conda.example.invalid/unselected.conda\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        materialize,
        "_verified_environment_capture_binding",
        lambda **kwargs: {
            "release_id": materialize.RELEASE_ID,
            "capture_id": "c" * 64,
        },
    )
    release_root = tmp_path / "release"

    with pytest.raises(
        materialize.MaterializationError,
        match="differs from the expected sealed binding",
    ):
        materialize.materialize_release(
            output_root=release_root,
            environment_capture_root=capture_root,
            source_repository=repository,
            release_worktree=release_root / "worktree",
            source_harness_prefix=source_harness,
            source_serving_prefix=source_serving,
            source_package_cache=source_package_cache,
            harness_prefix=release_root / "environments" / "harness",
            serving_prefix=release_root / "environments" / "serving",
            conda_toolchain_root=_fake_conda(tmp_path),
            expected_package_cache_seed_input=expected_cache_input,
        )

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
    source_package_cache = _source_package_cache(tmp_path)
    expected_cache_input = (
        materialize.selected_package_cache_input_binding(
            source_package_cache=source_package_cache,
            harness_seed=source_harness,
            serving_seed=source_serving,
        )
    )

    with pytest.raises(materialize.MaterializationError, match="clean exact production tag"):
        materialize.materialize_release(
            output_root=release_root,
            environment_capture_root=capture_root,
            source_repository=repository,
            release_worktree=release_root / "worktree",
            source_harness_prefix=source_harness,
            source_serving_prefix=source_serving,
            source_package_cache=source_package_cache,
            harness_prefix=release_root / "environments" / "harness",
            serving_prefix=release_root / "environments" / "serving",
            conda_toolchain_root=_fake_conda(tmp_path),
            expected_package_cache_seed_input=expected_cache_input,
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
        "source_package_cache": str(tmp_path / "source-package-cache"),
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
    conda_path = _fake_conda(tmp_path)
    toolchain_binding = _toolchain_binding(conda_path)
    conda_tool = {
        "path": str(conda_path.resolve()),
        "sha256": toolchain_binding["conda_executable"]["sha256"],
    }
    marker = {
        "schema_version": materialize.SCHEMA_VERSION,
        "release_id": materialize.RELEASE_ID,
        "git_tag": materialize.REQUIRED_TAG,
        "tag_commit": git_identity["git_commit"],
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "paths": paths,
        "output_root": str(root),
        "conda_toolchain": toolchain_binding,
        "conda_creation_tool": conda_tool,
        "environment_capture": capture_binding,
        "package_cache_seed_input": {
            "source_package_cache": paths["source_package_cache"],
            "inventory_sha256": "4" * 64,
            "inventory_entry_count": 1,
            "inventory_file_count": 1,
            "inventory_total_file_bytes": 1,
            "requirements_sha256": "5" * 64,
            "required_package_count": 1,
            "archive_count": 0,
            "selected_top_level_entries": [
                "cache",
                "python-3.11-test",
                "urls",
                "urls.txt",
            ],
        },
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
            "immutable_package_cache_seed": str(
                root / "conda-package-cache-seed"
            ),
            "package_cache_seed_protocol": (
                materialize.PACKAGE_CACHE_SEED_PROTOCOL
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


def test_package_cache_seed_is_marker_first_real_copy_and_replay_safe(
    tmp_path, monkeypatch
):
    harness = _source_prefix(tmp_path, "harness-seed")
    serving = _source_prefix(tmp_path, "serving-seed")
    source_cache = _source_package_cache(tmp_path)
    unrelated = source_cache / "unrelated-9.9-build"
    unrelated.mkdir()
    (unrelated / "payload").write_text("not selected\n", encoding="utf-8")
    output = tmp_path / "materialization"
    observed_marker_first = []
    original_copy = materialize._copy_cache_inventory_bound

    def observing_copy(source, destination, inventory):
        observed_marker_first.append(
            (output / materialize.PACKAGE_CACHE_SEED_INTENT).is_file()
        )
        return original_copy(source, destination, inventory)

    monkeypatch.setattr(
        materialize, "_copy_cache_inventory_bound", observing_copy
    )
    first = materialize._materialize_package_cache_seed(
        output_root=output,
        source_cache=source_cache,
        harness_seed=harness,
        serving_seed=serving,
    )
    second = materialize._materialize_package_cache_seed(
        output_root=output,
        source_cache=source_cache,
        harness_seed=harness,
        serving_seed=serving,
    )

    seed = output / materialize.PACKAGE_CACHE_SEED_DIRECTORY
    runtime = output / materialize.PACKAGE_CACHE_DIRECTORY
    assert first == second
    assert first["required_package_count"] == 1
    assert first["archive_count"] == 0
    assert observed_marker_first and all(observed_marker_first)
    assert not (seed / unrelated.name).exists()
    assert not (runtime / unrelated.name).exists()
    assert materialize.verify_independent_copy(source_cache, seed)[
        "shared_regular_inode_count"
    ] == 0
    assert materialize.verify_independent_copy(seed, runtime)[
        "shared_regular_inode_count"
    ] == 0
    materialize.capture._assert_read_only(seed)
    source_cache.rename(tmp_path / "retired-source-package-cache")
    assert materialize._verify_package_cache_seed(
        output, require_preclone_runtime_identity=True
    ) == first


def test_package_cache_seed_resumes_exact_partial_copy(tmp_path):
    harness = _source_prefix(tmp_path, "harness-seed")
    serving = _source_prefix(tmp_path, "serving-seed")
    source_cache = _source_package_cache(tmp_path)
    output = tmp_path / "materialization"
    seed = output / materialize.PACKAGE_CACHE_SEED_DIRECTORY
    plan = materialize._package_cache_seed_plan(
        source_cache=source_cache,
        harness_seed=harness,
        serving_seed=serving,
    )
    partial_rows = [
        row
        for row in plan["inventory"]["entries"]
        if Path(row["path"]).parts[0] in {"urls", "urls.txt"}
    ]
    partial = materialize._inventory_from_entries(partial_rows)
    materialize.capture.copy_inventory_bound(source_cache, seed, partial)

    report = materialize._materialize_package_cache_seed(
        output_root=output,
        source_cache=source_cache,
        harness_seed=harness,
        serving_seed=serving,
    )

    assert report["required_package_count"] == 1
    assert materialize._verify_package_cache_seed(
        output, require_preclone_runtime_identity=True
    ) == report


def test_package_cache_seed_resumes_after_copy_was_sealed_before_marker(
    tmp_path,
):
    harness = _source_prefix(tmp_path, "harness-seed")
    serving = _source_prefix(tmp_path, "serving-seed")
    source_cache = _source_package_cache(tmp_path)
    output = tmp_path / "materialization"
    seed = output / materialize.PACKAGE_CACHE_SEED_DIRECTORY
    plan = materialize._package_cache_seed_plan(
        source_cache=source_cache,
        harness_seed=harness,
        serving_seed=serving,
    )
    materialize.capture.copy_inventory_bound(
        source_cache, seed, plan["inventory"]
    )
    materialize.capture._seal_tree(seed)

    report = materialize._materialize_package_cache_seed(
        output_root=output,
        source_cache=source_cache,
        harness_seed=harness,
        serving_seed=serving,
    )

    assert report["sealed_read_only"] is True
    assert materialize._verify_package_cache_seed(
        output, require_preclone_runtime_identity=True
    ) == report


def test_package_cache_seed_recovers_sigkill_partial_file_temporary(tmp_path):
    harness = _source_prefix(tmp_path, "harness-seed")
    serving = _source_prefix(tmp_path, "serving-seed")
    source_cache = _source_package_cache(tmp_path)
    output = tmp_path / "materialization"
    seed = output / materialize.PACKAGE_CACHE_SEED_DIRECTORY
    partial = seed / ".urls.txt.schema5-cache-copying"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"truncated")

    report = materialize._materialize_package_cache_seed(
        output_root=output,
        source_cache=source_cache,
        harness_seed=harness,
        serving_seed=serving,
    )

    assert report["sealed_read_only"] is True
    assert not partial.exists()
    assert (
        seed / "urls.txt"
    ).read_bytes() == (source_cache / "urls.txt").read_bytes()


def test_package_cache_seed_rejects_hardlinked_partial_copy(tmp_path):
    harness = _source_prefix(tmp_path, "harness-seed")
    serving = _source_prefix(tmp_path, "serving-seed")
    source_cache = _source_package_cache(tmp_path)
    output = tmp_path / "materialization"
    seed = output / materialize.PACKAGE_CACHE_SEED_DIRECTORY
    shutil.copytree(source_cache, seed, copy_function=os.link)

    with pytest.raises(
        materialize.MaterializationError, match="shares .* regular-file inode"
    ):
        materialize._materialize_package_cache_seed(
            output_root=output,
            source_cache=source_cache,
            harness_seed=harness,
            serving_seed=serving,
        )


def test_package_cache_seed_requires_repository_metadata_and_exact_urls(tmp_path):
    harness = _source_prefix(tmp_path, "harness-seed")
    serving = _source_prefix(tmp_path, "serving-seed")
    source_cache = _source_package_cache(tmp_path)
    (source_cache / "urls.txt").write_text(
        "https://conda.example.invalid/wrong.conda\n", encoding="utf-8"
    )

    with pytest.raises(
        materialize.MaterializationError, match="cannot resolve exact selected URL"
    ):
        materialize._package_cache_seed_plan(
            source_cache=source_cache,
            harness_seed=harness,
            serving_seed=serving,
        )


def test_package_cache_seed_allows_unresolved_internal_link_but_rejects_escape(
    tmp_path,
):
    harness = _source_prefix(tmp_path, "harness-seed")
    serving = _source_prefix(tmp_path, "serving-seed")
    source_cache = _source_package_cache(tmp_path)
    package = source_cache / "python-3.11-test"
    (package / "lib").mkdir()
    (package / "lib" / "internal-missing").symlink_to(
        "provided-when-linked.so"
    )

    report = materialize._materialize_package_cache_seed(
        output_root=tmp_path / "accepted",
        source_cache=source_cache,
        harness_seed=harness,
        serving_seed=serving,
    )
    assert report["seed_symlink_audit"] == {
        "symlink_count": 1,
        "unresolved_internal_symlink_count": 1,
        "external_symlink_count": 0,
    }

    (package / "lib" / "escape").symlink_to("../../outside")
    with pytest.raises(
        materialize.MaterializationError,
        match="symlink escapes its selected package",
    ):
        materialize._materialize_package_cache_seed(
            output_root=tmp_path / "rejected",
            source_cache=source_cache,
            harness_seed=harness,
            serving_seed=serving,
        )


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
