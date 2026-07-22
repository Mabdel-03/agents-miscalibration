"""Immutable release/environment/fleet contract tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest

from scripts import freeze_schema5_release as freeze
from slurm import schema5_control


REPO = Path(__file__).resolve().parent.parent


def _run(*argv: str, cwd: Path) -> str:
    completed = subprocess.run(
        list(argv), cwd=cwd, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _tagged_worktree(tmp_path: Path) -> Path:
    worktree = tmp_path / "release-worktree"
    worktree.mkdir()
    _run("git", "init", "-q", cwd=worktree)
    _run("git", "config", "user.email", "release-test@example.invalid", cwd=worktree)
    _run("git", "config", "user.name", "Release Test", cwd=worktree)
    (worktree / "source.txt").write_bytes(b"immutable source\n")
    (worktree / "pyproject.toml").write_text(
        "[project]\nname = \"agents_scaling\"\nversion = \"0.1.0\"\n",
        encoding="utf-8",
    )
    package = worktree / "src" / "agents_scaling"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    configs = worktree / "configs"
    configs.mkdir()
    for filename in (
        "model_contracts.v1.json",
        "model_contracts.v1.sha256",
        "schema5_fleet.v1.json",
        "schema5_fleet.v1.sha256",
    ):
        shutil.copy2(REPO / "configs" / filename, configs / filename)
    _run("git", "add", "source.txt", "pyproject.toml", "src", "configs", cwd=worktree)
    _run("git", "commit", "-q", "-m", "release", cwd=worktree)
    _run("git", "tag", freeze.REQUIRED_GIT_TAG, cwd=worktree)
    return worktree


def _fake_environment(
    tmp_path: Path,
    name: str,
    *,
    serving: bool,
    release_worktree: Path | None = None,
) -> Path:
    prefix = tmp_path / name
    (prefix / "bin").mkdir(parents=True)
    (prefix / "conda-meta").mkdir()
    runtime = {
        "python_version": "3.11.13",
        "python_implementation": "CPython",
        "cuda_version": "12.8" if serving else None,
        "packages": {
            "torch": "2.8.0" if serving else None,
            "vllm": "0.21.0" if serving else None,
            "transformers": "4.57.1",
            "tokenizers": "0.22.1",
        },
    }
    pip_rows = ["pip==25.1", "tokenizers==0.22.1", "transformers==4.57.1"]
    import_payload: dict | None = None
    if release_worktree is not None:
        site_packages = prefix / "lib" / "python3.11" / "site-packages"
        package = site_packages / "agents_scaling"
        dist_info = site_packages / "agents_scaling-0.1.0.dist-info"
        package.mkdir(parents=True)
        dist_info.mkdir()
        (package / "__init__.py").write_text("\n", encoding="utf-8")
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: agents_scaling\nVersion: 0.1.0\n\n",
            encoding="utf-8",
        )
        direct_url = release_worktree.resolve().as_uri()
        (dist_info / "direct_url.json").write_text(
            json.dumps({"url": direct_url, "dir_info": {}}), encoding="utf-8"
        )
        (dist_info / "RECORD").write_text(
            "agents_scaling/__init__.py,,\n"
            "agents_scaling-0.1.0.dist-info/METADATA,,\n"
            "agents_scaling-0.1.0.dist-info/direct_url.json,,\n"
            "agents_scaling-0.1.0.dist-info/RECORD,,\n",
            encoding="utf-8",
        )
        pip_rows.insert(0, f"agents_scaling @ {direct_url}")
        import_payload = {
            "prefix": str(prefix),
            "module_path": str((package / "__init__.py").resolve()),
            "version": "0.1.0",
            "direct_url": {"url": direct_url, "dir_info": {}},
        }
    pip_printf = " ".join(repr(row) for row in pip_rows)
    import_json = json.dumps(import_payload, sort_keys=True)
    script = "\n".join(
        (
            "#!/bin/sh",
            'if [ "$1" = "-I" ] && [ "$2" = "-c" ]; then',
            f"  printf '%s\\n' '{import_json}'",
            'elif [ "$1" = "-c" ]; then',
            f"  printf '%s\\n' '{json.dumps(runtime, sort_keys=True)}'",
            'elif [ "$1" = "-m" ] && [ "$2" = "pip" ]; then',
            f"  printf '%s\\n' {pip_printf}",
            "else",
            "  exit 9",
            "fi",
            "",
        )
    )
    python = prefix / "bin" / "python"
    python.write_text(script, encoding="utf-8")
    python.chmod(0o755)
    (prefix / "conda-meta" / "history").write_text("created\n", encoding="utf-8")
    return prefix


def _fake_conda(tmp_path: Path) -> Path:
    conda = tmp_path / "conda"
    conda.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                "printf '%s\\n' '@EXPLICIT' \\",
                "  'https://conda.example.invalid/linux-64/python-3.11.13-h1.conda#0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef' \\",
                "  'https://conda.example.invalid/noarch/pip-25.1-pyhd8ed1ab_0.conda#abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789'",
                "",
            )
        ),
        encoding="utf-8",
    )
    conda.chmod(0o755)
    return conda


def _inputs(tmp_path: Path) -> dict:
    worktree = _tagged_worktree(tmp_path)
    return {
        "release_worktree": worktree,
        "harness_prefix": _fake_environment(
            tmp_path, "harness", serving=False, release_worktree=worktree
        ),
        "serving_prefix": _fake_environment(tmp_path, "serving", serving=True),
        "model_contract_path": worktree / "configs" / "model_contracts.v1.json",
        "fleet_contract_path": worktree / "configs" / "schema5_fleet.v1.json",
        "conda_executable": _fake_conda(tmp_path),
    }


def test_checked_in_fleet_is_exactly_22_replicas_and_24_gpus():
    models = freeze.load_model_contracts()
    fleet, digest = freeze.load_and_validate_fleet_contract(
        REPO / "configs" / "schema5_fleet.v1.json",
        model_contracts=models,
    )

    assert digest == hashlib.sha256(
        (REPO / "configs" / "schema5_fleet.v1.json").read_bytes()
    ).hexdigest()
    assert fleet["logical_replica_count"] == 22
    assert fleet["allocated_gpu_count"] == 24
    assert sum(len(profile["replicas"]) for profile in fleet["profiles"]) == 22
    assert sum(
        len(profile["replicas"]) * profile["gpus_per_replica"]
        for profile in fleet["profiles"]
    ) == 24
    long_32b = next(
        profile for profile in fleet["profiles"] if profile["serving_profile"] == "32B-long"
    )
    assert len(long_32b["replicas"]) == 2
    assert long_32b["tensor_parallel_size"] == 2
    assert long_32b["effective_context_limit"] == 40_960
    assert all(
        replica["pool_id"] == "schema5-v1"
        for profile in fleet["profiles"]
        for replica in profile["replicas"]
    )


def test_fleet_count_or_revision_drift_fails_even_with_new_checksum(tmp_path):
    source = REPO / "configs" / "schema5_fleet.v1.json"
    target = tmp_path / source.name
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["profiles"][0]["model_revision"] = "0" * 40
    target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    target.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {target.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(freeze.ReleaseFreezeError, match="model_revision"):
        freeze.load_and_validate_fleet_contract(
            target, model_contracts=freeze.load_model_contracts()
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("replica_id", "schema5-v1--invented--standard--r00", "replica ID drifted"),
        ("scheduler_job_name", "asys-s5-serve-invented", "scheduler job name drifted"),
        ("time_limit", "2-00:00:00", "replica placement drifted"),
        ("cpus_per_task", 99, "replica placement drifted"),
        ("memory", "999G", "replica placement drifted"),
    ),
)
def test_fleet_per_replica_identity_and_resources_are_not_redefinable(
    tmp_path, field, value, message
):
    source = REPO / "configs" / "schema5_fleet.v1.json"
    target = tmp_path / source.name
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["profiles"][0]["replicas"][0][field] = value
    target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    target.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {target.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(freeze.ReleaseFreezeError, match=message):
        freeze.load_and_validate_fleet_contract(
            target, model_contracts=freeze.load_model_contracts()
        )


def test_git_identity_uses_control_plane_tree_hash_and_requires_clean_exact_tag(tmp_path):
    worktree = _tagged_worktree(tmp_path)

    identity = freeze.verify_clean_exact_tag(worktree)

    assert identity["git_tag"] == freeze.REQUIRED_GIT_TAG
    assert identity["source_tree_sha256"] == schema5_control.sha256_tree(worktree)
    (worktree / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(freeze.ReleaseFreezeError, match="not clean"):
        freeze.verify_clean_exact_tag(worktree)


def test_dry_run_is_read_only_then_marker_last_bundle_is_idempotent(tmp_path):
    inputs = _inputs(tmp_path)
    output = tmp_path / "release-root" / "identity"

    dry_run = freeze.create_release_bundle(output, **inputs)

    assert dry_run["status"] == "dry_run"
    assert dry_run["source_tree_sha256"] == schema5_control.sha256_tree(
        inputs["release_worktree"]
    )
    assert not output.exists()

    created = freeze.create_release_bundle(output, apply=True, **inputs)

    assert created["status"] == "created"
    marker = output / freeze.COMPLETE_MARKER_FILENAME
    assert marker.is_file()
    assert stat.S_IMODE(marker.stat().st_mode) == 0o444
    identity = json.loads(
        (output / freeze.RELEASE_IDENTITY_FILENAME).read_text(encoding="utf-8")
    )
    for role, filename in (
        ("harness", freeze.HARNESS_MANIFEST_FILENAME),
        ("serving", freeze.SERVING_MANIFEST_FILENAME),
    ):
        manifest = output / filename
        record = identity["environments"][role]
        # This exact file hash is what schema5_control must pin; the full directory
        # inventory has its own independent hash inside that file.
        assert record["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        entries = payload["directory_inventory"]["entries"]
        assert [entry["path"] for entry in entries] == sorted(
            entry["path"] for entry in entries
        )
        assert payload["directory_inventory"]["inventory_sha256"] == record[
            "directory_inventory_sha256"
        ]
        assert payload["locks"]["conda_explicit"][0] == "@EXPLICIT"
        expected_pip = [
            "pip==25.1",
            "tokenizers==0.22.1",
            "transformers==4.57.1",
        ]
        if role == "harness":
            expected_pip.insert(
                0, f"agents_scaling @ {inputs['release_worktree'].resolve().as_uri()}"
            )
            binding = payload["release_package"]
            assert binding["editable"] is False
            assert binding["source_git_commit"] == identity["git"]["git_commit"]
            assert binding["source_tree_sha256"] == identity["git"][
                "source_tree_sha256"
            ]
            assert binding["installed_file_count"] == 4
            assert Path(binding["isolated_import_prefix"]) == inputs["harness_prefix"]
            assert Path(binding["isolated_import_path"]).is_relative_to(
                inputs["harness_prefix"]
            )
        else:
            assert payload["release_package"] is None
        assert payload["locks"]["pip_freeze_all"] == expected_pip
    pins = identity["control_pin_fragment"]
    assert pins["source_tree_sha256"] == schema5_control.sha256_tree(
        inputs["release_worktree"]
    )
    assert pins["harness_environment_sha256"] == identity["environments"][
        "harness"
    ]["manifest_sha256"]
    again = freeze.create_release_bundle(output, apply=True, **inputs)
    assert again["status"] == "already_complete"
    assert again["release_bundle_id"] == created["release_bundle_id"]


def test_live_environment_or_source_drift_is_rejected_after_publication(tmp_path):
    inputs = _inputs(tmp_path)
    output = tmp_path / "release-root"
    freeze.create_release_bundle(output, apply=True, **inputs)
    history = inputs["harness_prefix"] / "conda-meta" / "history"
    history.write_text("mutated\n", encoding="utf-8")

    with pytest.raises(freeze.ReleaseFreezeError, match="inventory drifted"):
        freeze.verify_release_bundle(output)


def test_optional_read_only_seal_captures_final_modes(tmp_path):
    inputs = _inputs(tmp_path)
    output = tmp_path / "release-root"

    freeze.create_release_bundle(
        output,
        apply=True,
        seal_worktree=True,
        seal_environments=True,
        **inputs,
    )

    for root in (
        inputs["release_worktree"],
        inputs["harness_prefix"],
        inputs["serving_prefix"],
    ):
        for directory, directory_names, file_names in os.walk(root):
            paths = [Path(directory), *(Path(directory) / name for name in directory_names + file_names)]
            for path in paths:
                if not path.is_symlink():
                    assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0
    assert freeze.verify_release_bundle(output)["status"] == "verified"


def test_local_or_editable_pip_requirement_fails_closed(tmp_path, monkeypatch):
    prefix = _fake_environment(tmp_path, "harness", serving=False)
    original_run = freeze._run

    def fake_run(argv, *, env=None):
        if tuple(argv[1:5]) == ("-m", "pip", "freeze", "--all"):
            return "package @ file:///tmp/build\n"
        return original_run(argv, env=env)

    monkeypatch.setattr(freeze, "_run", fake_run)
    with pytest.raises(freeze.ReleaseFreezeError, match="unowned direct"):
        freeze._pip_lock(prefix)


def _add_conda_owned_direct_distribution(
    prefix: Path, *, name: str = "packaging", version: str = "26.2"
) -> str:
    dist_info = prefix / "lib" / "python3.11" / "site-packages" / f"{name}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    metadata = dist_info / "METADATA"
    direct = dist_info / "direct_url.json"
    direct_url = f"file:///conda/build/{name}/work"
    metadata.write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n",
        encoding="utf-8",
    )
    direct.write_text(json.dumps({"url": direct_url, "dir_info": {}}), encoding="utf-8")
    relative_metadata = metadata.relative_to(prefix).as_posix()
    relative_direct = direct.relative_to(prefix).as_posix()
    paths = []
    for relative, path in (
        (relative_metadata, metadata),
        (relative_direct, direct),
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        paths.append(
            {
                "_path": relative,
                "path_type": "hardlink",
                "sha256": digest,
                "sha256_in_prefix": digest,
            }
        )
    record = {
        "name": name,
        "version": version,
        "url": f"https://conda.example.invalid/noarch/{name}-{version}-0.conda",
        "sha256": "a" * 64,
        "paths_data": {"paths_version": 1, "paths": paths},
    }
    (prefix / "conda-meta" / f"{name}-{version}-0.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    return direct_url


def test_conda_owned_builder_direct_url_is_normalized_to_exact_version(
    tmp_path, monkeypatch
):
    prefix = _fake_environment(tmp_path, "harness", serving=False)
    direct_url = _add_conda_owned_direct_distribution(prefix)
    original_run = freeze._run

    def fake_run(argv, *, env=None):
        if tuple(argv[1:5]) == ("-m", "pip", "freeze", "--all"):
            return f"packaging @ {direct_url}\npip==25.1\n"
        return original_run(argv, env=env)

    monkeypatch.setattr(freeze, "_run", fake_run)

    assert freeze._pip_lock(prefix) == ["packaging==26.2", "pip==25.1"]


def test_stale_conda_record_cannot_bless_a_modified_direct_distribution(
    tmp_path, monkeypatch
):
    prefix = _fake_environment(tmp_path, "harness", serving=False)
    direct_url = _add_conda_owned_direct_distribution(prefix)
    direct = next(prefix.glob("lib/python*/site-packages/packaging-*.dist-info/direct_url.json"))
    direct.write_text(json.dumps({"url": direct_url, "dir_info": {"changed": True}}))
    monkeypatch.setattr(
        freeze,
        "_run",
        lambda argv, **kwargs: f"packaging @ {direct_url}\n",
    )

    with pytest.raises(freeze.ReleaseFreezeError, match="drifted from package record"):
        freeze._pip_lock(prefix)


@pytest.mark.parametrize(
    "row",
    (
        "-e git+ssh://git@example.invalid/repo.git#egg=package",
        "package @ git+https://example.invalid/repo.git@deadbeef",
        "package>=1.0",
        "package @ https://example.invalid/package.whl#sha256=" + "a" * 64,
    ),
)
def test_editable_vcs_unpinned_and_non_conda_direct_requirements_fail_closed(
    tmp_path, monkeypatch, row
):
    prefix = _fake_environment(tmp_path, "harness", serving=False)
    monkeypatch.setattr(freeze, "_run", lambda argv, **kwargs: row + "\n")

    with pytest.raises(freeze.ReleaseFreezeError, match="non-exact requirements"):
        freeze._pip_lock(prefix)


def test_release_package_direct_url_must_be_noneditable_and_exact_source(tmp_path):
    worktree = _tagged_worktree(tmp_path)
    prefix = _fake_environment(
        tmp_path, "harness", serving=False, release_worktree=worktree
    )
    identity = freeze.verify_clean_exact_tag(worktree)
    direct = next(
        prefix.glob("lib/python*/site-packages/agents_scaling-*.dist-info/direct_url.json")
    )
    payload = json.loads(direct.read_text(encoding="utf-8"))
    payload["dir_info"]["editable"] = True
    direct.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(freeze.ReleaseFreezeError, match="editable"):
        freeze._pip_lock_material(
            prefix,
            release_worktree=worktree,
            git_identity=identity,
            require_release_package=True,
        )


def test_release_package_import_cannot_escape_harness_prefix(tmp_path, monkeypatch):
    worktree = _tagged_worktree(tmp_path)
    prefix = _fake_environment(
        tmp_path, "harness", serving=False, release_worktree=worktree
    )
    identity = freeze.verify_clean_exact_tag(worktree)
    original_run = freeze._run

    def fake_run(argv, *, env=None):
        if "-I" in argv:
            return json.dumps(
                {
                    "prefix": str(prefix),
                    "module_path": str(worktree / "src" / "agents_scaling" / "__init__.py"),
                    "version": "0.1.0",
                    "direct_url": {
                        "url": worktree.resolve().as_uri(),
                        "dir_info": {},
                    },
                }
            )
        return original_run(argv, env=env)

    monkeypatch.setattr(freeze, "_run", fake_run)

    with pytest.raises(freeze.ReleaseFreezeError, match="outside the frozen harness"):
        freeze._pip_lock_material(
            prefix,
            release_worktree=worktree,
            git_identity=identity,
            require_release_package=True,
        )
