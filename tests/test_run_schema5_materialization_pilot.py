"""End-to-end orchestration tests for the schema-5 materialization pilot."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import time

import pytest

from scripts import run_schema5_materialization_pilot as pilot
from scripts import publish_schema5_durable_git_release as durable_git
from scripts import seal_recovery_evidence as seal_evidence


REPO = Path(__file__).resolve().parent.parent
SETUPTOOLS_ARTIFACT_SHA256 = (
    "82088a6e4daa33329a30bc26dc19a98c7c1d3f05c0f73ce9845d4eab4924e9e1"
)


def _fake_toolchain_binding(value: str | Path) -> dict:
    root = Path(value).resolve()
    executable = root / "bin" / "conda"
    inventory_rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            inventory_rows.append(
                (
                    path.relative_to(root).as_posix(),
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    path.stat().st_size,
                )
            )
    inventory_sha256 = hashlib.sha256(
        pilot._canonical_bytes(inventory_rows)
    ).hexdigest()
    executable_bytes = executable.read_bytes()
    binding = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r5-offline-conda-toolchain-v1",
        "release_tag": pilot.REQUIRED_TAG,
        "chain_namespace": "schema5-v1.2-r5",
        "toolchain_root": str(root),
        "base_prefix": str(root),
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
            "sha256": hashlib.sha256(executable_bytes).hexdigest(),
            "size": len(executable_bytes),
            "mode": executable.stat().st_mode & 0o777,
            "link_count": executable.stat().st_nlink,
        },
        "runtime_identity_sha256": inventory_sha256,
        "complete_prefix_inventory_sha256": inventory_sha256,
        "read_only_probes": {"probe_count": 2},
    }
    binding["binding_id"] = hashlib.sha256(
        pilot._canonical_bytes(binding)
    ).hexdigest()
    return binding


@pytest.fixture(autouse=True)
def _mock_sealed_conda_toolchain(monkeypatch):
    monkeypatch.setattr(
        pilot.conda_toolchain,
        "verified_conda_toolchain_binding",
        lambda root, exercise=True: _fake_toolchain_binding(root),
    )


def test_all_fresh_pilot_wire_protocols_identify_r5() -> None:
    protocols = {
        pilot.PILOT_QUARANTINE_PROTOCOL,
        pilot.PILOT_QUARANTINE_INTENT_PROTOCOL,
        pilot.SCHEDULER_INTENT_PROTOCOL,
        pilot.SCHEDULER_ACTIVE_PROTOCOL,
        pilot.SCHEDULER_ACCEPTANCE_PROTOCOL,
        pilot.SUBMISSION_INTENT_PROTOCOL,
        pilot.SUBMISSION_ATTEMPT_PROTOCOL,
        pilot.SUBMISSION_RESULT_PROTOCOL,
        pilot.SUBMISSION_ABSENT_PROTOCOL,
        pilot.SUBMISSION_ACCEPTED_PROTOCOL,
    }

    assert all("schema5-v1.2-r5-" in protocol for protocol in protocols)
    assert all("schema5-v1.2-r3-" not in protocol for protocol in protocols)


def test_pilot_cli_accepts_only_sealed_toolchain_root() -> None:
    parser = pilot._build_parser()
    choices = next(
        action.choices
        for action in parser._actions
        if hasattr(action, "choices") and action.choices
    )
    for command in ("run", "render-sbatch", "conda-toolchain-binding"):
        options = {
            option
            for action in choices[command]._actions
            for option in action.option_strings
        }
        assert "--conda-toolchain-root" in options
        assert "--conda-executable" not in options
    assert "conda-runtime-identity" not in choices


def test_pilot_subprocess_environment_rejects_hostile_command_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = {
        "PATH": "/tmp/hostile-bin:/usr/bin",
        "BASH_ENV": "/tmp/hostile-bash-env",
        "LD_PRELOAD": "/tmp/hostile.so",
        "GIT_DIR": "/tmp/hostile-git-dir",
        "GIT_CONFIG_PARAMETERS": "'core.hooksPath=/tmp/hostile-hooks'",
        "GIT_REPLACE_REF_BASE": "refs/hostile",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": "/tmp/hostile-hooks",
        "SBATCH_PARTITION": "hostile",
        "SQUEUE_FORMAT": "hostile",
        "SACCT_FORMAT": "hostile",
        "SCONTROL_ALL": "hostile",
        "SLURM_CONF": "/tmp/hostile-slurm.conf",
        "BASH_FUNC_git%%": "() { false; }",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)

    environment = pilot._sanitized_process_environment()

    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert not (set(hostile) - {"PATH"}).intersection(environment)


def _run(*argv: str, cwd: Path) -> str:
    completed = subprocess.run(
        list(argv), cwd=cwd, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _tagged_checkout(tmp_path: Path, *, annotated: bool = True) -> tuple[Path, str]:
    checkout = tmp_path / "release-checkout"
    checkout.mkdir()
    _run("git", "init", "-q", cwd=checkout)
    _run("git", "config", "user.email", "pilot@example.invalid", cwd=checkout)
    _run("git", "config", "user.name", "Pilot Test", cwd=checkout)
    (checkout / "source.txt").write_text("immutable release\n", encoding="utf-8")
    scripts = checkout / "scripts"
    scripts.mkdir()
    shutil.copy2(
        Path(pilot.__file__).resolve(),
        scripts / "run_schema5_materialization_pilot.py",
    )
    configs = checkout / "configs"
    configs.mkdir()
    (configs / "model_contracts.v1.json").write_text("{}\n", encoding="utf-8")
    (configs / "schema5_fleet.v1.json").write_text("{}\n", encoding="utf-8")
    _run("git", "add", ".", cwd=checkout)
    _run("git", "commit", "-q", "-m", "release", cwd=checkout)
    if annotated:
        _run(
            "git",
            "tag",
            "-a",
            pilot.REQUIRED_TAG,
            "-m",
            "schema5 pilot release",
            cwd=checkout,
        )
    else:
        _run("git", "tag", pilot.REQUIRED_TAG, cwd=checkout)
    return checkout, _run("git", "rev-parse", "HEAD", cwd=checkout)


def _record_hash(payload: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")


def _conda_record(*, name: str, version: str, build: str, digest: str) -> dict:
    return {
        "name": name,
        "version": version,
        "build": build,
        "fn": f"{name}-{version}-{build}.conda",
        "sha256": digest,
        "url": f"https://conda.example.invalid/noarch/{name}-{version}-{build}.conda",
    }


def _live_prefix(tmp_path: Path, name: str, *, stale_record: bool) -> Path:
    prefix = tmp_path / name
    conda_meta = prefix / "conda-meta"
    site = prefix / "lib" / "python3.11" / "site-packages"
    distribution = site / "setuptools-81.0.0.dist-info"
    package = site / "setuptools"
    conda_meta.mkdir(parents=True)
    distribution.mkdir(parents=True)
    package.mkdir()
    (prefix / "bin").mkdir()
    python = prefix / "bin" / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    package_bytes = b"__version__ = '81.0.0'\n"
    metadata_bytes = (
        b"Metadata-Version: 2.1\nName: setuptools\nVersion: 81.0.0\n\n"
    )
    (package / "__init__.py").write_bytes(package_bytes)
    (distribution / "METADATA").write_bytes(metadata_bytes)
    (distribution / "RECORD").write_text(
        "setuptools/__init__.py,sha256="
        + _record_hash(package_bytes)
        + f",{len(package_bytes)}\n"
        "setuptools-81.0.0.dist-info/METADATA,sha256="
        + _record_hash(metadata_bytes)
        + f",{len(metadata_bytes)}\n"
        "setuptools-81.0.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    python_record = _conda_record(
        name="python", version="3.11.13", build="h1_0", digest="1" * 64
    )
    (conda_meta / "python-3.11.13-h1_0.json").write_text(
        json.dumps(python_record, sort_keys=True) + "\n", encoding="utf-8"
    )
    if stale_record:
        setuptools_record = _conda_record(
            name="setuptools",
            version="82.0.1",
            build="pyh332efcf_0",
            digest=SETUPTOOLS_ARTIFACT_SHA256,
        )
        (conda_meta / "setuptools-82.0.1-pyh332efcf_0.json").write_text(
            json.dumps(setuptools_record, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return prefix


def _evidence(
    tmp_path: Path, harness: Path, serving: Path
) -> tuple[Path, Path]:
    recovered = tmp_path / "setuptools-82.0.1-pyh332efcf_0.json"
    recovered.write_text(
        json.dumps(
            _conda_record(
                name="setuptools",
                version="82.0.1",
                build="pyh332efcf_0",
                digest=SETUPTOOLS_ARTIFACT_SHA256,
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    recovered.chmod(0o444)
    failed_log = tmp_path / "failed-materialization.log"
    failed_log.write_text("pip distribution view changed\n", encoding="utf-8")
    incident = tmp_path / "CONDA_RECONCILIATION_INCIDENT.json"
    seal_evidence.record_conda_reconciliation_incident(
        output=incident,
        harness_prefix=harness,
        serving_prefix=serving,
        recovered_harness_record=recovered,
        failed_materialization_log=failed_log,
        observed_at="2026-07-23T11:43:55-04:00",
        apply=True,
    )
    return incident, recovered


def _inputs(tmp_path: Path, *, annotated: bool = True) -> tuple[dict, Path]:
    checkout, commit = _tagged_checkout(tmp_path, annotated=annotated)
    tag_object = _run(
        "git", "rev-parse", f"refs/tags/{pilot.REQUIRED_TAG}", cwd=checkout
    )
    durable_root = tmp_path / "durable-release"
    bundle_root = durable_root / durable_git.BUNDLE_DIRECTORY
    bundle_root.mkdir(parents=True)
    bundle = bundle_root / durable_git.BUNDLE_NAME
    bundle.write_bytes(b"fixture Git bundle bytes\n")
    bundle.chmod(0o444)
    bundle_sha256 = hashlib.sha256(bundle.read_bytes()).hexdigest()
    checksum = bundle_root / durable_git.CHECKSUM_NAME
    checksum.write_text(
        f"{bundle_sha256}  {durable_git.BUNDLE_NAME}\n",
        encoding="ascii",
    )
    checksum.chmod(0o444)
    durable_marker = durable_root / durable_git.MARKER_NAME
    durable_value = {
        "schema_version": 1,
        "protocol": durable_git.PROTOCOL,
        "passed": True,
        "release_id": durable_git.RELEASE_ID,
        "release_tag": durable_git.RELEASE_TAG,
        "release_git_commit": commit,
        "release_tag_object": tag_object,
        "chain_namespace": durable_git.CHAIN_NAMESPACE,
        "clean_checkout": True,
        "annotated_tag": True,
        "remote_query_read_only": True,
        "remote": "durable",
        "remote_commit_ref": durable_git.DURABLE_COMMIT_REF,
        "remote_commit": commit,
        "remote_tag_object": tag_object,
        "remote_peeled_commit": commit,
        "remote_url_sha256": "1" * 64,
        "bundle_path": str(bundle),
        "bundle_sha256": bundle_sha256,
        "bundle_size": bundle.stat().st_size,
        "checksum_path": str(checksum),
        "checksum_sha256": hashlib.sha256(checksum.read_bytes()).hexdigest(),
        "published_at": "2026-07-25T00:00:00+00:00",
    }
    durable_value["marker_id"] = durable_git._self_hash(
        durable_value, "marker_id"
    )
    durable_marker.write_bytes(durable_git._canonical(durable_value))
    durable_marker.chmod(0o444)
    harness = _live_prefix(tmp_path, "live-harness", stale_record=False)
    serving = _live_prefix(tmp_path, "live-serving", stale_record=True)
    incident, recovered = _evidence(tmp_path, harness, serving)
    conda_called = tmp_path / "CONDA_WAS_CALLED"
    conda_base = tmp_path / "miniforge"
    (conda_base / "bin").mkdir(parents=True)
    (conda_base / "lib" / "python3.12" / "site-packages" / "conda").mkdir(
        parents=True
    )
    conda_python = conda_base / "bin" / "python"
    shutil.copy2(Path("/bin/sh").resolve(), conda_python)
    (
        conda_base
        / "lib"
        / "python3.12"
        / "site-packages"
        / "conda"
        / "__init__.py"
    ).write_text("__version__ = 'fixture'\n", encoding="utf-8")
    conda = conda_base / "bin" / "conda"
    conda.write_text(
        f"#!{conda_python}\n: > {conda_called}\nexit 97\n", encoding="utf-8"
    )
    conda.chmod(0o755)
    package_cache = tmp_path / "source-package-cache"
    extracted = package_cache / "python-3.11.13-h1_0"
    (extracted / "info").mkdir(parents=True)
    python_record = _conda_record(
        name="python", version="3.11.13", build="h1_0", digest="1" * 64
    )
    (extracted / "info" / "index.json").write_text(
        json.dumps(
            {
                key: python_record[key]
                for key in ("name", "version", "build")
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (extracted / "info" / "repodata_record.json").write_text(
        json.dumps(python_record, sort_keys=True) + "\n", encoding="utf-8"
    )
    (extracted / "payload").write_text("python package\n", encoding="utf-8")
    (package_cache / "cache").mkdir()
    (package_cache / "cache" / "channel.json").write_text(
        '{"packages": {}}\n', encoding="utf-8"
    )
    (package_cache / "urls").write_text("", encoding="utf-8")
    (package_cache / "urls.txt").write_text(
        python_record["url"] + "\n", encoding="utf-8"
    )
    return (
        {
            "pilot_root": tmp_path / "pilot",
            "release_checkout": checkout,
            "expected_tag": pilot.REQUIRED_TAG,
            "expected_commit": commit,
            "harness_source": harness,
            "serving_source": serving,
            "ownership_policy": (
                REPO / "configs" / "environment_ownership_policy.v1.json"
            ),
            "integrity_normalization_policy": (
                REPO
                / "configs"
                / "environment_integrity_normalization_policy.v1.json"
            ),
            "reconciliation_incident": incident,
            "recovered_setuptools_record": recovered,
            "conda_toolchain_root": conda_base,
            "source_package_cache": package_cache,
            "durable_git_release_marker": durable_marker,
        },
        conda_called,
    )


def _seal_tree(root: Path) -> None:
    entries: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        entries.extend(Path(directory) / name for name in directory_names)
        entries.extend(Path(directory) / name for name in file_names)
    for path in sorted(entries, key=lambda value: len(value.parts), reverse=True):
        if not path.is_symlink():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)


def _install_stage_doubles(
    monkeypatch,
    *,
    hardlink_environments: bool = False,
    mutate_live: Path | None = None,
) -> list[dict]:
    materialization_reports: dict[str, dict] = {}
    release_reports: dict[str, dict] = {}
    consumer_calls: list[dict] = []

    def fake_materialize_release(*, apply=False, **kwargs):
        output = Path(kwargs["output_root"])
        paths = {
            key: str(Path(kwargs[key]).resolve())
            for key in (
                "source_repository",
                "environment_capture_root",
                "release_worktree",
                "source_harness_prefix",
                "source_serving_prefix",
                "source_package_cache",
                "harness_prefix",
                "serving_prefix",
            )
        }
        toolchain_binding = _fake_toolchain_binding(
            kwargs["conda_toolchain_root"]
        )
        package_cache_seed_input = (
            pilot.materialize.selected_package_cache_input_binding(
                source_package_cache=kwargs["source_package_cache"],
                harness_seed=kwargs["source_harness_prefix"],
                serving_seed=kwargs["source_serving_prefix"],
            )
        )
        assert (
            kwargs["expected_package_cache_seed_input"]
            == package_cache_seed_input
        )
        report = {
            "release_id": pilot.RELEASE_ID,
            "materialization_id": "b" * 64,
            "tag_commit": _run(
                "git",
                "rev-parse",
                "HEAD",
                cwd=Path(kwargs["source_repository"]),
            ),
            "source_tree_sha256": "c" * 64,
            "paths": paths,
            "environment_capture": {"capture_id": "d" * 64},
            "conda_toolchain": toolchain_binding,
            "conda_creation_tool": {
                "path": toolchain_binding["conda_executable"]["path"],
                "sha256": toolchain_binding["conda_executable"]["sha256"],
            },
            "package_cache_seed_input": package_cache_seed_input,
            "conda_package_cache_sha256": "e" * 64,
            "conda_package_cache_seed_sha256": "9" * 64,
        }
        if not apply:
            return {**report, "status": "dry_run"}
        marker = output / "MATERIALIZATION_COMPLETE.json"
        if not marker.exists():
            output.mkdir(parents=True)
            shutil.copytree(
                kwargs["source_repository"],
                kwargs["release_worktree"],
                symlinks=True,
            )
            copy_function = os.link if hardlink_environments else shutil.copy2
            shutil.copytree(
                kwargs["source_harness_prefix"],
                kwargs["harness_prefix"],
                symlinks=True,
                copy_function=copy_function,
            )
            shutil.copytree(
                kwargs["source_serving_prefix"],
                kwargs["serving_prefix"],
                symlinks=True,
                copy_function=copy_function,
            )
            cache_seed = output / "conda-package-cache-seed"
            cache = output / "conda-package-cache"
            shutil.copytree(kwargs["source_package_cache"], cache_seed)
            shutil.copytree(cache_seed, cache)
            marker.write_text(
                json.dumps({"materialization_id": report["materialization_id"]}) + "\n",
                encoding="utf-8",
            )
            status = "created"
        else:
            status = "already_complete"
        materialization_reports[str(output.resolve())] = report
        return {**report, "status": status}

    def fake_verify_materialization(output_root):
        output = Path(output_root).resolve()
        report = materialization_reports.get(str(output))
        if report is None or not (output / "MATERIALIZATION_COMPLETE.json").is_file():
            raise pilot.materialize.MaterializationError(
                "fake materialization is incomplete"
            )
        return {**report, "status": "verified"}

    def fake_freeze(*, apply=False, **kwargs):
        assert "conda_executable" not in kwargs
        output = Path(kwargs["output_root"])
        materialization_report = materialization_reports[
            str(output.parent.resolve())
        ]
        report = {
            "release_id": pilot.RELEASE_ID,
            "release_bundle_id": "f" * 64,
            "git_commit": _run(
                "git",
                "rev-parse",
                "HEAD",
                cwd=Path(kwargs["release_worktree"]),
            ),
            "release_tag_object": _run(
                "git",
                "rev-parse",
                f"refs/tags/{pilot.REQUIRED_TAG}",
                cwd=Path(kwargs["release_worktree"]),
            ),
            "source_tree_sha256": "c" * 64,
            "materialization_id": "b" * 64,
            "harness_environment_manifest_sha256": "1" * 64,
            "serving_environment_manifest_sha256": "2" * 64,
            "conda_package_cache_seed_sha256": "9" * 64,
            "model_contract_sha256": "3" * 64,
            "fleet_contract_sha256": "4" * 64,
            "logical_replica_count": 22,
            "allocated_gpu_count": 24,
        }
        if not apply:
            return {
                **report,
                "status": "dry_run",
                "would_seal_worktree": True,
                "would_seal_environments": True,
                "would_seal_output_root": True,
            }
        marker = output / "RELEASE_COMPLETE.json"
        if marker.exists():
            existing_identity = json.loads(
                (output / pilot.freeze.RELEASE_IDENTITY_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            for role in ("harness", "serving"):
                report[f"{role}_environment_manifest_sha256"] = (
                    existing_identity["environments"][role]["manifest_sha256"]
                )
        if not marker.exists():
            output.mkdir()
            environment_records = {}
            for role, filename, prefix, inventory_sha in (
                (
                    "harness",
                    pilot.freeze.HARNESS_MANIFEST_FILENAME,
                    Path(kwargs["harness_prefix"]).resolve(),
                    "5" * 64,
                ),
                (
                    "serving",
                    pilot.freeze.SERVING_MANIFEST_FILENAME,
                    Path(kwargs["serving_prefix"]).resolve(),
                    "6" * 64,
                ),
            ):
                manifest_path = output / filename
                manifest_path.write_text(
                    json.dumps(
                        {
                            "prefix": str(prefix),
                            "sealed_read_only": True,
                            "directory_inventory": {
                                "inventory_sha256": inventory_sha,
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                manifest_path.chmod(0o444)
                manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                report[f"{role}_environment_manifest_sha256"] = manifest_sha
                environment_records[role] = {
                    "prefix": str(prefix),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_sha,
                    "directory_inventory_sha256": inventory_sha,
                }
            fragment = {
                "release_id": pilot.RELEASE_ID,
                "release_worktree": str(Path(kwargs["release_worktree"]).resolve()),
                "git_commit": report["git_commit"],
                "release_tag_object": report["release_tag_object"],
                "source_tree_sha256": report["source_tree_sha256"],
                "conda_package_cache_seed_sha256": report[
                    "conda_package_cache_seed_sha256"
                ],
                "transport_uncertainty_binding": (
                    pilot.control.scheduler_safety
                    .expected_transport_uncertainty_binding()
                ),
                "transport_uncertainty_binding_sha256": (
                    pilot.control.scheduler_safety
                    .transport_uncertainty_binding_sha256(
                        pilot.control.scheduler_safety
                        .expected_transport_uncertainty_binding()
                    )
                ),
                "model_contract_path": str(
                    Path(kwargs["model_contract_path"]).resolve()
                ),
                "model_contract_sha256": report["model_contract_sha256"],
                "fleet_contract_path": str(
                    Path(kwargs["fleet_contract_path"]).resolve()
                ),
                "fleet_contract_sha256": report["fleet_contract_sha256"],
                "harness_environment_prefix": str(
                    Path(kwargs["harness_prefix"]).resolve()
                ),
                "harness_environment_manifest_path": str(
                    output / pilot.freeze.HARNESS_MANIFEST_FILENAME
                ),
                "harness_environment_sha256": report[
                    "harness_environment_manifest_sha256"
                ],
                "serving_environment_prefix": str(
                    Path(kwargs["serving_prefix"]).resolve()
                ),
                "serving_environment_manifest_path": str(
                    output / pilot.freeze.SERVING_MANIFEST_FILENAME
                ),
                "serving_environment_sha256": report[
                    "serving_environment_manifest_sha256"
                ],
                "conda_toolchain": materialization_report[
                    "conda_toolchain"
                ],
                "package_cache_seed_input": materialization_report[
                    "package_cache_seed_input"
                ],
            }
            (output / pilot.freeze.RELEASE_IDENTITY_FILENAME).write_text(
                json.dumps(
                    {
                        "control_pin_fragment": fragment,
                        "environments": environment_records,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            marker.write_text(
                json.dumps({"release_bundle_id": report["release_bundle_id"]}) + "\n",
                encoding="utf-8",
            )
            marker.chmod(0o444)
            for root in (
                Path(kwargs["release_worktree"]),
                Path(kwargs["harness_prefix"]),
                Path(kwargs["serving_prefix"]),
                output.parent / "conda-package-cache-seed",
                output.parent / "conda-package-cache",
            ):
                _seal_tree(root)
            output.chmod(0o555)
            if mutate_live is not None:
                (mutate_live / "unexpected-pilot-mutation").write_text(
                    "drift\n", encoding="utf-8"
                )
            status = "created"
        else:
            status = "already_complete"
        release_reports[str(output.resolve())] = report
        return {
            **report,
            "status": status,
            "output_root_sealed_read_only": True,
        }

    def fake_verify_release(output_root):
        output = Path(output_root).resolve()
        report = release_reports.get(str(output))
        if report is None or not (output / "RELEASE_COMPLETE.json").is_file():
            raise pilot.freeze.ReleaseFreezeError("fake release is incomplete")
        return {**report, "status": "verified"}

    def fake_control_consumer(pins):
        consumer_calls.append(dict(pins))
        assert set(pins) == {
            *pilot.control._RELEASE_FRAGMENT_FIELDS,
            "release_bundle_root",
            "release_bundle_id",
        }

    monkeypatch.setattr(
        pilot.materialize, "materialize_release", fake_materialize_release
    )
    monkeypatch.setattr(
        pilot.materialize,
        "verify_materialization",
        fake_verify_materialization,
    )
    monkeypatch.setattr(pilot.freeze, "create_release_bundle", fake_freeze)
    monkeypatch.setattr(pilot.freeze, "verify_release_bundle", fake_verify_release)
    monkeypatch.setattr(
        pilot.control, "_validate_release_bundle", fake_control_consumer
    )
    return consumer_calls


def test_pilot_dry_run_is_read_only_and_requires_no_conda_query(tmp_path):
    inputs, conda_called = _inputs(tmp_path)

    report = pilot.run_materialization_pilot(**inputs)

    assert report["status"] == "dry_run"
    assert report["expected_commit"] == inputs["expected_commit"]
    assert report["git_identity"]["tag_object_type"] == "tag"
    assert report["capture_dry_run"]["copy_contract"]["hardlinks"] is False
    assert not inputs["pilot_root"].exists()
    assert not conda_called.exists()


def test_rendered_sbatch_is_exact_immutable_no_requeue_and_rerunnable(tmp_path):
    inputs, conda_called = _inputs(tmp_path)
    python = tmp_path / "pilot-python"
    python.write_text("#!/bin/sh\nexit 98\n", encoding="utf-8")
    python.chmod(0o755)
    sbatch = tmp_path / "jobs" / "schema5-pilot.sbatch"
    logs = tmp_path / "logs"
    render_inputs = {
        **inputs,
        "sbatch_path": sbatch,
        "log_dir": logs,
        "partition": "sched_test",
        "python_executable": python,
    }

    dry = pilot.render_materialization_pilot_sbatch(**render_inputs)
    assert dry["status"] == "dry_run"
    assert dry["time_limit"] == "11:30:00"
    assert dry["no_requeue"] is True
    assert not sbatch.exists()
    assert not logs.exists()

    created = pilot.render_materialization_pilot_sbatch(
        **render_inputs, apply=True
    )
    repeated = pilot.render_materialization_pilot_sbatch(
        **render_inputs, apply=True
    )

    assert created["status"] == "created"
    assert repeated["status"] == "already_complete"
    assert created["sbatch_sha256"] == repeated["sbatch_sha256"]
    text = sbatch.read_text(encoding="utf-8")
    assert "#SBATCH --no-requeue\n" in text
    assert "#SBATCH --export=NONE\n" in text
    assert "#SBATCH --export=ALL\n" not in text
    assert "#SBATCH --time=11:30:00\n" in text
    assert "#SBATCH --partition=sched_test\n" in text
    assert f"#SBATCH --output={logs}/schema5-materialization-pilot-%j.out\n" in text
    assert f"#SBATCH --error={logs}/schema5-materialization-pilot-%j.err\n" in text
    assert f"#SBATCH --chdir={inputs['release_checkout']}\n" in text
    assert "#SBATCH --comment=asys-s5-pilot:r5:" in text
    assert "export PATH=/usr/bin:/bin\nreadonly PATH\n" in text
    exact_script = (
        Path(inputs["release_checkout"])
        / "scripts"
        / "run_schema5_materialization_pilot.py"
    )
    assert str(exact_script) in text
    assert (
        f"--source-package-cache {inputs['source_package_cache']}" in text
    )
    assert (
        f"--conda-toolchain-root {inputs['conda_toolchain_root']}" in text
    )
    assert "--apply" in text
    assert sbatch.stat().st_mode & 0o222 == 0
    checksum = Path(str(sbatch) + ".sha256")
    assert checksum.read_text(encoding="utf-8") == (
        f"{created['sbatch_sha256']}  {sbatch.name}\n"
    )
    receipt = json.loads(
        Path(created["receipt_path"]).read_text(encoding="utf-8")
    )
    assert receipt["expected_commit"] == inputs["expected_commit"]
    assert receipt["git_identity"]["tag_object_type"] == "tag"
    assert receipt["slurm"]["no_requeue"] is True
    assert receipt["slurm"]["time_limit"] == "11:30:00"
    assert created["submit_command"] == ["sbatch", str(sbatch.resolve())]
    assert logs.is_dir()
    assert not inputs["pilot_root"].exists()
    assert not conda_called.exists()


SUBMISSION_TEST_USER = "schema5-pilot-test"


def test_submission_scheduler_since_uses_scheduler_local_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = 1_784_952_540.0  # 2026-07-25 04:09 UTC / 00:09 EDT.
    with monkeypatch.context() as timezone_context:
        timezone_context.setenv("TZ", "America/New_York")
        time.tzset()
        try:
            since = pilot._submission_scheduler_since(timestamp)
            commands = pilot._submission_query_argv(
                {
                    "scheduler_user": SUBMISSION_TEST_USER,
                    "scheduler_since": since,
                }
            )
        finally:
            timezone_context.undo()
            time.tzset()

    assert since == "2026-07-25T00:04:00"
    assert commands["sacct"][commands["sacct"].index("-S") + 1] == since


def _rendered_submission_fixture(
    tmp_path: Path,
) -> tuple[dict, Path, dict, str]:
    inputs, _conda_called = _inputs(tmp_path)
    python = tmp_path / "pilot-python"
    python.write_text("#!/bin/sh\nexit 98\n", encoding="utf-8")
    python.chmod(0o755)
    rendered = pilot.render_materialization_pilot_sbatch(
        **inputs,
        sbatch_path=tmp_path / "jobs" / "schema5-pilot.sbatch",
        log_dir=tmp_path / "logs",
        partition="sched_test",
        python_executable=python,
        apply=True,
    )
    receipt_path = Path(rendered["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return inputs, receipt_path, receipt, "13579"


def _submission_query_result(
    argv,
    *,
    receipt: dict,
    job_ids: tuple[str, ...] = (),
    memory: str = "8G",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    command = list(argv)
    if command[0] == "squeue":
        assert command[:3] == ["squeue", "-u", SUBMISSION_TEST_USER]
        assert command[3:] == [
            "-h",
            "-r",
            "-o",
            "%i|%j|%T|%k|%P|%l|%C|%m|%o",
        ]
        stdout = "".join(
            (
                f"{job_id}|{pilot.SBATCH_JOB_NAME}|RUNNING|"
                f"{receipt['slurm']['comment']}|"
                f"{receipt['slurm']['partition']}|"
                f"{receipt['slurm']['time_limit']}|"
                f"{receipt['slurm']['cpus_per_task']}|{memory}|"
                f"{receipt['sbatch_path']}\n"
            )
            for job_id in job_ids
        )
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout,
            "" if returncode == 0 else "squeue unavailable\n",
        )
    if command[0] == "sacct":
        assert command[:3] == ["sacct", "-u", SUBMISSION_TEST_USER]
        assert command[3:7] == ["-X", "-n", "-P", "-S"]
        assert command[-1].startswith("--format=JobIDRaw,JobName")
        stdout = "".join(
            (
                f"{job_id}|{pilot.SBATCH_JOB_NAME}|RUNNING|0:0|None|"
                f"{receipt['slurm']['comment']}|"
                f"{receipt['slurm']['partition']}|"
                f"{receipt['slurm']['time_limit']}|"
                f"{shlex.join(receipt['submit_command'])}\n"
            )
            for job_id in job_ids
        )
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout,
            "" if returncode == 0 else "sacct unavailable\n",
        )
    raise AssertionError(command)


def _visible_after_sbatch_runner(
    receipt: dict,
    *,
    scheduler_job_id: str,
    sbatch_process: subprocess.CompletedProcess[str] | BaseException | None = None,
    publish_job: bool = True,
    before_sbatch=None,
) -> tuple[object, list[list[str]], dict]:
    calls: list[list[str]] = []
    state = {"visible": False, "sbatch_calls": 0}
    if sbatch_process is None:
        sbatch_process = subprocess.CompletedProcess(
            receipt["submit_command"],
            0,
            f"Submitted batch job {scheduler_job_id}\n",
            "",
        )

    def runner(argv):
        command = list(argv)
        calls.append(command)
        if command[0] in {"squeue", "sacct"}:
            return _submission_query_result(
                command,
                receipt=receipt,
                job_ids=(scheduler_job_id,) if state["visible"] else (),
            )
        if command == receipt["submit_command"]:
            state["sbatch_calls"] += 1
            state["visible"] = publish_job
            if before_sbatch is not None:
                before_sbatch()
            if isinstance(sbatch_process, BaseException):
                raise sbatch_process
            return subprocess.CompletedProcess(
                command,
                sbatch_process.returncode,
                sbatch_process.stdout,
                sbatch_process.stderr,
            )
        raise AssertionError(command)

    return runner, calls, state


def _submit_fixture(
    inputs: dict,
    receipt_path: Path,
    runner,
) -> dict:
    return pilot.submit_materialization_pilot_sbatch(
        pilot_root=inputs["pilot_root"],
        sbatch_receipt=receipt_path,
        scheduler_user=SUBMISSION_TEST_USER,
        visibility_timeout=0,
        runner=runner,
        apply=True,
    )


def test_submission_dry_run_creates_no_artifacts_and_does_not_invoke_sbatch(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, receipt, job_id = _rendered_submission_fixture(
        tmp_path
    )
    runner, calls, _state = _visible_after_sbatch_runner(
        receipt, scheduler_job_id=job_id
    )

    report = pilot.submit_materialization_pilot_sbatch(
        pilot_root=inputs["pilot_root"],
        sbatch_receipt=receipt_path,
        scheduler_user=SUBMISSION_TEST_USER,
        visibility_timeout=0,
        runner=runner,
    )

    assert report["status"] == "dry_run"
    assert report["marker_first"] is True
    assert calls == []
    assert not Path(inputs["pilot_root"]).exists()


def test_submission_publishes_read_only_root_and_attempt_intents_before_sbatch(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, receipt, job_id = _rendered_submission_fixture(
        tmp_path
    )
    root = Path(inputs["pilot_root"])
    observed = {"marker_first": False}

    def assert_marker_first() -> None:
        intent = root / pilot.SUBMISSION_INTENT_FILENAME
        attempt = (
            root
            / pilot.SUBMISSION_DIRECTORY
            / pilot.SUBMISSION_ATTEMPTS_DIRECTORY
            / "a0001"
            / pilot.SUBMISSION_ATTEMPT_INTENT_FILENAME
        )
        result = attempt.parent / pilot.SUBMISSION_RESULT_FILENAME
        acceptance = root / pilot.SUBMISSION_ACCEPTED_FILENAME
        assert root.is_dir()
        assert intent.is_file()
        assert attempt.is_file()
        assert intent.stat().st_mode & 0o222 == 0
        assert attempt.stat().st_mode & 0o222 == 0
        assert not result.exists()
        assert not acceptance.exists()
        observed["marker_first"] = True

    runner, _calls, _state = _visible_after_sbatch_runner(
        receipt,
        scheduler_job_id=job_id,
        before_sbatch=assert_marker_first,
    )

    submitted = _submit_fixture(inputs, receipt_path, runner)

    assert submitted["status"] == "submitted"
    assert observed["marker_first"] is True


def test_submission_is_marker_last_immutable_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, receipt_path, receipt, job_id = _rendered_submission_fixture(
        tmp_path
    )
    root = Path(inputs["pilot_root"])
    runner, _calls, state = _visible_after_sbatch_runner(
        receipt, scheduler_job_id=job_id
    )
    writes: list[Path] = []
    real_write = pilot._atomic_write_once

    def record_write(path, payload):
        writes.append(Path(path))
        return real_write(path, payload)

    monkeypatch.setattr(pilot, "_atomic_write_once", record_write)
    submitted = _submit_fixture(inputs, receipt_path, runner)
    marker = root / pilot.SUBMISSION_ACCEPTED_FILENAME
    marker_bytes = marker.read_bytes()
    replayed = _submit_fixture(inputs, receipt_path, runner)

    attempt = (
        root
        / pilot.SUBMISSION_DIRECTORY
        / pilot.SUBMISSION_ATTEMPTS_DIRECTORY
        / "a0001"
    )
    immutable = (
        root / pilot.SUBMISSION_INTENT_FILENAME,
        attempt / pilot.SUBMISSION_ATTEMPT_INTENT_FILENAME,
        attempt / pilot.SUBMISSION_RESULT_FILENAME,
        marker,
    )
    assert submitted["status"] == "submitted"
    assert replayed["status"] == "already_submitted"
    assert submitted["acceptance_id"] == replayed["acceptance_id"]
    assert state["sbatch_calls"] == 1
    assert writes[-1] == marker
    assert marker.read_bytes() == marker_bytes
    assert all(path.stat().st_mode & 0o222 == 0 for path in immutable)


def test_submission_crash_after_scheduler_acceptance_is_adopted_exactly_once(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, receipt, job_id = _rendered_submission_fixture(
        tmp_path
    )
    root = Path(inputs["pilot_root"])
    runner, _calls, state = _visible_after_sbatch_runner(
        receipt,
        scheduler_job_id=job_id,
        sbatch_process=RuntimeError("lost sbatch stdout"),
    )

    with pytest.raises(RuntimeError, match="lost sbatch stdout"):
        _submit_fixture(inputs, receipt_path, runner)

    attempt = (
        root
        / pilot.SUBMISSION_DIRECTORY
        / pilot.SUBMISSION_ATTEMPTS_DIRECTORY
        / "a0001"
    )
    assert (attempt / pilot.SUBMISSION_ATTEMPT_INTENT_FILENAME).is_file()
    assert not (attempt / pilot.SUBMISSION_RESULT_FILENAME).exists()
    assert not (root / pilot.SUBMISSION_ACCEPTED_FILENAME).exists()

    adopted = _submit_fixture(inputs, receipt_path, runner)
    replayed = _submit_fixture(inputs, receipt_path, runner)

    assert adopted["status"] == "adopted"
    assert replayed["status"] == "already_submitted"
    assert adopted["job_id"] == job_id
    assert adopted["sbatch_result"] is None
    assert state["sbatch_calls"] == 1


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [
        (0, "accepted without a parseable job identifier\n", ""),
        (9, "", "sbatch transport returned nonzero\n"),
    ],
    ids=["malformed-success", "nonzero"],
)
def test_submission_adopts_exact_scheduler_job_despite_inconclusive_sbatch_result(
    tmp_path: Path,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    inputs, receipt_path, receipt, job_id = _rendered_submission_fixture(
        tmp_path
    )
    process = subprocess.CompletedProcess(
        receipt["submit_command"], returncode, stdout, stderr
    )
    runner, _calls, state = _visible_after_sbatch_runner(
        receipt,
        scheduler_job_id=job_id,
        sbatch_process=process,
    )

    accepted = _submit_fixture(inputs, receipt_path, runner)

    result = json.loads(Path(accepted["sbatch_result"]).read_text(encoding="utf-8"))
    assert accepted["status"] == "submitted"
    assert accepted["job_id"] == job_id
    assert accepted["scheduler_sources"] == ["sacct", "squeue"]
    assert result["returncode"] == returncode
    assert result["reported_job_id"] is None
    assert state["sbatch_calls"] == 1


def test_submission_rejects_sbatch_stdout_scheduler_job_id_mismatch_without_retry(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, receipt, scheduler_job_id = (
        _rendered_submission_fixture(tmp_path)
    )
    process = subprocess.CompletedProcess(
        receipt["submit_command"],
        0,
        "Submitted batch job 97531\n",
        "",
    )
    runner, _calls, state = _visible_after_sbatch_runner(
        receipt,
        scheduler_job_id=scheduler_job_id,
        sbatch_process=process,
    )

    for _ in range(2):
        with pytest.raises(
            pilot.MaterializationPilotError,
            match="sbatch stdout and scheduler truth disagree",
        ):
            _submit_fixture(inputs, receipt_path, runner)

    root = Path(inputs["pilot_root"])
    assert state["sbatch_calls"] == 1
    assert not (root / pilot.SUBMISSION_ACCEPTED_FILENAME).exists()


@pytest.mark.parametrize(
    ("scenario", "expected_error"),
    [
        ("squeue-incomplete", "complete squeue\\+sacct truth"),
        ("sacct-incomplete", "complete squeue\\+sacct truth"),
        ("multiple", "multiple scheduler jobs"),
    ],
)
def test_submission_rejects_incomplete_or_ambiguous_scheduler_truth_without_retry(
    tmp_path: Path,
    scenario: str,
    expected_error: str,
) -> None:
    inputs, receipt_path, receipt, job_id = _rendered_submission_fixture(
        tmp_path
    )
    calls: list[list[str]] = []

    def runner(argv):
        command = list(argv)
        calls.append(command)
        if command[0] == "sbatch":
            raise AssertionError("sbatch must not run without unique complete truth")
        return _submission_query_result(
            command,
            receipt=receipt,
            job_ids=(job_id, "24680") if scenario == "multiple" else (),
            returncode=(
                1
                if scenario == f"{command[0]}-incomplete"
                else 0
            ),
        )

    for _ in range(2):
        with pytest.raises(
            pilot.MaterializationPilotError,
            match=expected_error,
        ):
            _submit_fixture(inputs, receipt_path, runner)

    root = Path(inputs["pilot_root"])
    assert not any(command[0] == "sbatch" for command in calls)
    assert not (root / pilot.SUBMISSION_ACCEPTED_FILENAME).exists()
    assert not (
        root
        / pilot.SUBMISSION_DIRECTORY
        / pilot.SUBMISSION_ATTEMPTS_DIRECTORY
    ).exists()


def test_submission_rejects_scheduler_memory_not_equivalent_to_eight_gibibytes(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, receipt, job_id = _rendered_submission_fixture(
        tmp_path
    )
    calls: list[list[str]] = []

    def runner(argv):
        command = list(argv)
        calls.append(command)
        if command[0] == "sbatch":
            raise AssertionError("conflicting scheduler identity must not submit")
        return _submission_query_result(
            command,
            receipt=receipt,
            job_ids=(job_id,),
            memory="8000M",
        )

    with pytest.raises(
        pilot.MaterializationPilotError,
        match="scheduler identity conflict",
    ):
        _submit_fixture(inputs, receipt_path, runner)

    root = Path(inputs["pilot_root"])
    assert not any(command[0] == "sbatch" for command in calls)
    assert not (root / pilot.SUBMISSION_ACCEPTED_FILENAME).exists()


def test_successful_sbatch_without_scheduler_truth_fails_closed_and_never_resubmits(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, receipt, job_id = _rendered_submission_fixture(
        tmp_path
    )
    runner, _calls, state = _visible_after_sbatch_runner(
        receipt,
        scheduler_job_id=job_id,
        publish_job=False,
    )

    for _ in range(2):
        with pytest.raises(
            pilot.MaterializationPilotError,
            match="accepted pilot job.*absent|success.*absent",
        ):
            _submit_fixture(inputs, receipt_path, runner)

    root = Path(inputs["pilot_root"])
    attempt = (
        root
        / pilot.SUBMISSION_DIRECTORY
        / pilot.SUBMISSION_ATTEMPTS_DIRECTORY
        / "a0001"
    )
    assert state["sbatch_calls"] == 1
    assert (attempt / pilot.SUBMISSION_RESULT_FILENAME).is_file()
    assert not (attempt / pilot.SUBMISSION_ABSENT_FILENAME).exists()
    assert not (root / pilot.SUBMISSION_ACCEPTED_FILENAME).exists()
    assert not attempt.with_name("a0002").exists()


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        ("acceptance", "acceptance identity drifted"),
        ("query", "scheduler query evidence drifted"),
    ],
)
def test_submission_replay_rejects_tampered_acceptance_or_query_evidence(
    tmp_path: Path,
    tamper: str,
    expected_error: str,
) -> None:
    inputs, receipt_path, receipt, job_id = _rendered_submission_fixture(
        tmp_path
    )
    runner, calls, state = _visible_after_sbatch_runner(
        receipt, scheduler_job_id=job_id
    )
    _submit_fixture(inputs, receipt_path, runner)
    calls_before_replay = list(calls)
    marker = Path(inputs["pilot_root"]) / pilot.SUBMISSION_ACCEPTED_FILENAME
    accepted = json.loads(marker.read_text(encoding="utf-8"))
    if tamper == "acceptance":
        accepted["job_name"] = "tampered-pilot-name"
    else:
        accepted["scheduler_snapshot"]["squeue"]["stdout"] += "tampered\n"
    marker.chmod(0o644)
    marker.write_text(
        json.dumps(accepted, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    marker.chmod(0o444)

    with pytest.raises(
        pilot.MaterializationPilotError,
        match=expected_error,
    ):
        _submit_fixture(inputs, receipt_path, runner)

    assert state["sbatch_calls"] == 1
    assert calls == calls_before_replay


def test_real_two_prefix_orchestration_is_marker_last_idempotent_and_verifiable(
    tmp_path, monkeypatch
):
    inputs, conda_called = _inputs(tmp_path)
    consumer_calls = _install_stage_doubles(monkeypatch)
    writes: list[str] = []
    real_write = pilot._atomic_write_once

    def record_write(path, payload):
        writes.append(Path(path).name)
        return real_write(path, payload)

    monkeypatch.setattr(pilot, "_atomic_write_once", record_write)
    created = pilot.run_materialization_pilot(**inputs, apply=True)
    verified = pilot._verify_materialization_pilot_semantic(
        inputs["pilot_root"]
    )
    repeated = pilot.run_materialization_pilot(**inputs, apply=True)

    assert created["status"] == "created"
    assert verified["status"] == "verified"
    assert repeated["status"] == "already_complete"
    assert created["pilot_id"] == verified["pilot_id"] == repeated["pilot_id"]
    assert verified["conda_toolchain"] == _fake_toolchain_binding(
        inputs["conda_toolchain_root"]
    )
    assert len(verified["package_cache_seed_input"]["input_id"]) == 64
    assert (
        json.loads(
            (
                Path(inputs["pilot_root"]) / pilot.INTENT_MARKER
            ).read_text(encoding="utf-8")
        )["inputs"]["source_package_cache"]
        == str(Path(inputs["source_package_cache"]).resolve())
    )
    assert verified["source_package_cache"] == str(
        Path(inputs["source_package_cache"]).resolve()
    )
    assert verified["conda_package_cache_seed_sha256"] == "9" * 64
    assert verified["ownership_policy_sha256"] == hashlib.sha256(
        Path(inputs["ownership_policy"]).read_bytes()
    ).hexdigest()
    assert verified["integrity_normalization_policy_sha256"] == (
        hashlib.sha256(
            Path(inputs["integrity_normalization_policy"]).read_bytes()
        ).hexdigest()
    )
    incident_payload = json.loads(
        Path(inputs["reconciliation_incident"]).read_text(encoding="utf-8")
    )
    assert verified["reconciliation_incident"] == {
        "path": str(
            Path(inputs["pilot_root"])
            / "environment-capture"
            / "evidence"
            / "CONDA_RECONCILIATION_INCIDENT.json"
        ),
        "sha256": hashlib.sha256(
            Path(inputs["reconciliation_incident"]).read_bytes()
        ).hexdigest(),
        "incident_id": incident_payload["incident_id"],
        "harness_stale_conda_record_present": False,
        "serving_stale_conda_record_present": True,
    }
    assert writes[-1] == pilot.COMPLETE_MARKER
    assert not conda_called.exists()
    assert verified["setuptools_contract"] == {
        "runtime_version": "81.0.0",
        "distribution_count_per_prefix": 1,
        "setuptools_82_conda_record_count": 0,
        "setuptools_82_versioned_path_count": 0,
    }
    assert verified["shared_regular_inode_count"] == 0
    assert verified["sealed_verification_repeatable"] is True
    assert len(consumer_calls) >= 6
    marker = json.loads(
        (inputs["pilot_root"] / pilot.COMPLETE_MARKER).read_text(encoding="utf-8")
    )
    assert marker["live_source_inventories"]["harness"]["equal"] is True
    assert marker["live_source_inventories"]["serving"]["equal"] is True
    audit = json.loads(
        (inputs["pilot_root"] / pilot.AUDIT_FILENAME).read_text(encoding="utf-8")
    )
    assert all(
        row["setuptools_81_distribution_count"] == 1
        for row in audit["setuptools"].values()
    )
    assert all(
        row["shared_regular_inode_count"] == 0
        for row in audit["transaction_inode_edges"].values()
    )
    frozen_identity = json.loads(
        (
            Path(marker["layout"]["release_bundle"])
            / pilot.freeze.RELEASE_IDENTITY_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert consumer_calls[-1] == {
        **frozen_identity["control_pin_fragment"],
        "release_bundle_root": marker["layout"]["release_bundle"],
        "release_bundle_id": marker["stage_ids"]["release_bundle_id"],
    }
    assert audit["control_consumer"]["validator"] == (
        "schema5_control._validate_release_bundle"
    )


def test_underlying_conda_runtime_drift_is_detected_without_invoking_conda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, conda_called = _inputs(tmp_path)
    _install_stage_doubles(monkeypatch)
    created = pilot.run_materialization_pilot(**inputs, apply=True)
    entrypoint = Path(inputs["conda_toolchain_root"]) / "bin" / "conda"
    entrypoint_sha256 = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    module = (
        entrypoint.parent.parent
        / "lib"
        / "python3.12"
        / "site-packages"
        / "conda"
        / "__init__.py"
    )
    module.write_text("__version__ = 'drifted'\n", encoding="utf-8")
    assert hashlib.sha256(entrypoint.read_bytes()).hexdigest() == entrypoint_sha256
    with pytest.raises(
        pilot.MaterializationPilotError,
        match="toolchain binding drifted",
    ):
        pilot._verify_materialization_pilot_semantic(inputs["pilot_root"])
    with pytest.raises(
        pilot.MaterializationPilotError,
        match="toolchain binding drifted",
    ):
        pilot.run_materialization_pilot(**inputs, apply=True)
    assert not conda_called.exists()


def test_pilot_crash_before_final_audit_resumes_without_redrawing_stages(
    tmp_path, monkeypatch
):
    inputs, _conda_called = _inputs(tmp_path)
    _install_stage_doubles(monkeypatch)
    real_write = pilot._atomic_write_once
    crashed = False

    def crash_once(path, payload):
        nonlocal crashed
        if Path(path).name == pilot.AUDIT_FILENAME and not crashed:
            crashed = True
            raise RuntimeError("simulated pilot crash")
        return real_write(path, payload)

    monkeypatch.setattr(pilot, "_atomic_write_once", crash_once)
    with pytest.raises(RuntimeError, match="simulated pilot crash"):
        pilot.run_materialization_pilot(**inputs, apply=True)
    assert not (inputs["pilot_root"] / pilot.COMPLETE_MARKER).exists()

    monkeypatch.setattr(pilot, "_atomic_write_once", real_write)
    recovered = pilot.run_materialization_pilot(**inputs, apply=True)

    assert recovered["status"] == "created"
    assert pilot._verify_materialization_pilot_semantic(inputs["pilot_root"])[
        "pilot_id"
    ] == recovered["pilot_id"]


@pytest.mark.parametrize(
    "drift",
    ("package-cache", "toolchain", "paths"),
)
def test_pilot_resume_rejects_authentic_but_stale_materialization_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    inputs, _conda_called = _inputs(tmp_path)
    _install_stage_doubles(monkeypatch)
    real_materialize = pilot.materialize.materialize_release
    apply_attempts = 0

    def crash_first_apply(*, apply=False, **kwargs):
        nonlocal apply_attempts
        if apply:
            apply_attempts += 1
            if apply_attempts == 1:
                raise RuntimeError("simulated crash before materialization apply")
        return real_materialize(apply=apply, **kwargs)

    monkeypatch.setattr(
        pilot.materialize,
        "materialize_release",
        crash_first_apply,
    )
    with pytest.raises(
        RuntimeError,
        match="simulated crash before materialization apply",
    ):
        pilot.run_materialization_pilot(**inputs, apply=True)

    evidence_path = (
        Path(inputs["pilot_root"])
        / pilot.STAGE_EVIDENCE_FILENAMES[("materialization", "dry_run")]
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if drift == "package-cache":
        evidence["report"]["package_cache_seed_input"]["input_id"] = "0" * 64
    elif drift == "toolchain":
        evidence["report"]["conda_toolchain"]["toolchain_id"] = "0" * 64
    else:
        evidence["report"]["paths"]["source_package_cache"] = (
            str(tmp_path / "different-source-package-cache")
        )
    evidence = pilot._stage_evidence_payload(
        stage="materialization",
        phase="dry_run",
        report=evidence["report"],
    )
    evidence_path.chmod(0o644)
    evidence_path.write_bytes(pilot._json_bytes(evidence))

    with pytest.raises(
        pilot.MaterializationPilotError,
        match="dry/apply/verify immutable phase binding differs",
    ):
        pilot.run_materialization_pilot(**inputs, apply=True)

    # The resumed transaction must reject the stale preflight before its first
    # mutating materialization call.
    assert apply_attempts == 1
    assert not (
        Path(pilot._layout(inputs["pilot_root"])["materialization_root"])
        / pilot.materialize.COMPLETE_MARKER
    ).exists()


def test_pilot_withholds_completion_if_any_live_source_byte_changes(
    tmp_path, monkeypatch
):
    inputs, _conda_called = _inputs(tmp_path)
    _install_stage_doubles(
        monkeypatch,
        mutate_live=Path(inputs["harness_source"]),
    )

    with pytest.raises(
        pilot.MaterializationPilotError,
        match="live harness prefix changed",
    ):
        pilot.run_materialization_pilot(**inputs, apply=True)

    assert not (inputs["pilot_root"] / pilot.COMPLETE_MARKER).exists()


def test_pilot_rejects_lightweight_release_tag_before_creating_output(tmp_path):
    inputs, _conda_called = _inputs(tmp_path, annotated=False)

    with pytest.raises(
        pilot.MaterializationPilotError, match="not an annotated tag"
    ):
        pilot.run_materialization_pilot(**inputs)

    assert not inputs["pilot_root"].exists()


def test_pilot_rejects_shared_seed_clone_inodes_and_withholds_marker(
    tmp_path, monkeypatch
):
    inputs, _conda_called = _inputs(tmp_path)
    _install_stage_doubles(monkeypatch, hardlink_environments=True)

    with pytest.raises(
        pilot.MaterializationPilotError, match="shares .* regular-file inode"
    ):
        pilot.run_materialization_pilot(**inputs, apply=True)

    assert not (inputs["pilot_root"] / pilot.COMPLETE_MARKER).exists()


def test_pilot_verifier_rejects_evidence_tamper(tmp_path, monkeypatch):
    inputs, _conda_called = _inputs(tmp_path)
    _install_stage_doubles(monkeypatch)
    pilot.run_materialization_pilot(**inputs, apply=True)
    evidence = (
        inputs["pilot_root"]
        / pilot.STAGE_EVIDENCE_FILENAMES[("capture", "verify")]
    )
    evidence.chmod(0o644)
    evidence.write_bytes(evidence.read_bytes() + b" ")
    evidence.chmod(0o444)

    with pytest.raises(
        pilot.MaterializationPilotError, match="evidence inventory drifted"
    ):
        pilot._verify_materialization_pilot_semantic(inputs["pilot_root"])


def test_pilot_verifier_requires_release_bundle_root_to_remain_sealed(
    tmp_path, monkeypatch
):
    inputs, _conda_called = _inputs(tmp_path)
    _install_stage_doubles(monkeypatch)
    pilot.run_materialization_pilot(**inputs, apply=True)
    release_bundle = Path(pilot._layout(inputs["pilot_root"])["release_bundle"])
    release_bundle.chmod(0o755)

    with pytest.raises(
        pilot.MaterializationPilotError,
        match="verifier runtime is not exact and read-only",
    ):
        pilot._verify_materialization_pilot_semantic(inputs["pilot_root"])


def _pilot_quarantine_fixture(
    tmp_path: Path,
    *,
    state: str = "NODE_FAIL",
    exit_code: str = "0:0",
    reason: str = "NodeDown",
) -> tuple[dict, Path, str, object]:
    inputs, _conda_called = _inputs(tmp_path)
    python = tmp_path / "pilot-python"
    python.write_text("#!/bin/sh\nexit 98\n", encoding="utf-8")
    python.chmod(0o755)
    sbatch = tmp_path / "jobs" / "schema5-pilot.sbatch"
    rendered = pilot.render_materialization_pilot_sbatch(
        **inputs,
        sbatch_path=sbatch,
        log_dir=tmp_path / "logs",
        partition="sched_test",
        python_executable=python,
        apply=True,
    )
    pilot_root = Path(inputs["pilot_root"])
    receipt_path = Path(rendered["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    job_id = "24680"
    submission_runner, _submission_calls, submission_state = (
        _visible_after_sbatch_runner(
            receipt,
            scheduler_job_id=job_id,
        )
    )
    submitted = pilot.submit_materialization_pilot_sbatch(
        pilot_root=pilot_root,
        sbatch_receipt=receipt_path,
        scheduler_user=SUBMISSION_TEST_USER,
        visibility_timeout=0,
        runner=submission_runner,
        apply=True,
    )
    assert submitted["job_id"] == job_id
    assert submission_state["sbatch_calls"] == 1
    (pilot_root / pilot.INTENT_MARKER).write_text(
        '{"interrupted": true}\n', encoding="utf-8"
    )

    def runner(argv):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "sacct":
            row = (
                f"{job_id}|{pilot.SBATCH_JOB_NAME}|{state}|{exit_code}|"
                f"{reason}|(null)|sbatch {sbatch.resolve()}\n"
            )
            return subprocess.CompletedProcess(argv, 0, row, "")
        raise AssertionError(argv)

    return inputs, receipt_path, job_id, runner


def test_transient_pilot_quarantine_is_marker_first_sealed_and_idempotent(
    tmp_path: Path,
) -> None:
    inputs, receipt, job_id, runner = _pilot_quarantine_fixture(tmp_path)
    root = Path(inputs["pilot_root"])

    dry = pilot.quarantine_interrupted_pilot(
        pilot_root=root,
        sbatch_receipt=receipt,
        job_id=job_id,
        runner=runner,
    )
    assert dry["status"] == "dry_run"
    assert dry["scheduler"]["raw_state"] == "NODE_FAIL"
    assert root.is_dir()

    created = pilot.quarantine_interrupted_pilot(
        pilot_root=root,
        sbatch_receipt=receipt,
        job_id=job_id,
        runner=runner,
        apply=True,
    )
    repeated = pilot.quarantine_interrupted_pilot(
        pilot_root=root,
        sbatch_receipt=receipt,
        job_id=job_id,
        runner=runner,
        apply=True,
    )

    destination = Path(created["destination"])
    evidence = root.parent / "quarantine_evidence"
    assert created["status"] == "quarantined_and_sealed"
    assert repeated["status"] == "already_quarantined_and_sealed"
    assert not root.exists()
    assert destination.is_dir()
    assert not any(
        os.lstat(path).st_mode & 0o222
        for path in (destination, *destination.rglob("*"))
        if not path.is_symlink()
    )
    assert (evidence / f"partial-job-{job_id}.intent.json").is_file()
    assert (evidence / f"partial-job-{job_id}.complete.json").is_file()
    assert (evidence / f"partial-job-{job_id}.sealed.json").is_file()
    assert created["scheduler"] == repeated["scheduler"]


def test_pilot_quarantine_recovers_after_rename_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, receipt, job_id, runner = _pilot_quarantine_fixture(tmp_path)
    root = Path(inputs["pilot_root"])
    real_write = pilot._atomic_write_once
    crashed = False

    def crash_before_completion(path, payload):
        nonlocal crashed
        if Path(path).name == f"partial-job-{job_id}.complete.json" and not crashed:
            crashed = True
            raise RuntimeError("simulated quarantine crash")
        return real_write(path, payload)

    monkeypatch.setattr(pilot, "_atomic_write_once", crash_before_completion)
    with pytest.raises(RuntimeError, match="simulated quarantine crash"):
        pilot.quarantine_interrupted_pilot(
            pilot_root=root,
            sbatch_receipt=receipt,
            job_id=job_id,
            runner=runner,
            apply=True,
        )
    assert not root.exists()
    assert (
        root.parent
        / "quarantine"
        / f"{root.name}.partial-job-{job_id}"
    ).is_dir()

    monkeypatch.setattr(pilot, "_atomic_write_once", real_write)
    recovered = pilot.quarantine_interrupted_pilot(
        pilot_root=root,
        sbatch_receipt=receipt,
        job_id=job_id,
        runner=runner,
        apply=True,
    )
    assert recovered["status"] == "quarantined_and_sealed"


@pytest.mark.parametrize(
    ("state", "exit_code", "reason"),
    [
        ("FAILED", "2:0", "NonZeroExitCode"),
        ("OUT_OF_MEMORY", "0:9", "OutOfMemory"),
        ("TIMEOUT", "0:0", "TimeLimit"),
        ("CANCELLED", "0:15", "DependencyNeverSatisfied"),
    ],
)
def test_pilot_quarantine_rejects_nontransient_terminal_failures(
    tmp_path: Path,
    state: str,
    exit_code: str,
    reason: str,
) -> None:
    inputs, receipt, job_id, runner = _pilot_quarantine_fixture(
        tmp_path,
        state=state,
        exit_code=exit_code,
        reason=reason,
    )
    root = Path(inputs["pilot_root"])

    with pytest.raises(
        pilot.MaterializationPilotError,
        match="require a superseding release",
    ):
        pilot.quarantine_interrupted_pilot(
            pilot_root=root,
            sbatch_receipt=receipt,
            job_id=job_id,
            runner=runner,
            apply=True,
        )
    assert root.is_dir()
    assert not (root.parent / "quarantine").exists()


def test_pilot_quarantine_accepts_explicit_external_cancellation(
    tmp_path: Path,
) -> None:
    inputs, receipt, job_id, runner = _pilot_quarantine_fixture(
        tmp_path,
        state="CANCELLED by 225593",
        exit_code="0:15",
        reason="None",
    )

    report = pilot.quarantine_interrupted_pilot(
        pilot_root=inputs["pilot_root"],
        sbatch_receipt=receipt,
        job_id=job_id,
        runner=runner,
    )
    assert report["status"] == "dry_run"
    assert (
        report["scheduler"]["transient_classification"]
        == "external_cancellation"
    )


def _sealed_pilot_quarantine(
    tmp_path: Path,
    *,
    state: str = "NODE_FAIL",
    exit_code: str = "0:0",
    reason: str = "NodeDown",
) -> tuple[dict, Path, dict, Path]:
    inputs, receipt_path, fixture_job_id, runner = _pilot_quarantine_fixture(
        tmp_path,
        state=state,
        exit_code=exit_code,
        reason=reason,
    )
    job_id = fixture_job_id
    quarantined = pilot.quarantine_interrupted_pilot(
        pilot_root=inputs["pilot_root"],
        sbatch_receipt=receipt_path,
        job_id=job_id,
        runner=runner,
        apply=True,
    )
    assert quarantined["status"] == "quarantined_and_sealed"
    root = Path(inputs["pilot_root"])
    seal_path = (
        root.parent
        / "quarantine_evidence"
        / f"partial-job-{job_id}.sealed.json"
    )
    assert seal_path.is_file()
    return (
        inputs,
        receipt_path,
        json.loads(receipt_path.read_text(encoding="utf-8")),
        seal_path,
    )


def test_submission_retry_excludes_only_explicit_sealed_transient_job(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, receipt, seal_path = _sealed_pilot_quarantine(
        tmp_path
    )
    old_job_id = "24680"
    new_job_id = "24681"
    state = {"visible": False, "sbatch_calls": 0}

    def runner(argv):
        command = list(argv)
        if command[0] == "squeue":
            job_ids = (new_job_id,) if state["visible"] else ()
            return _submission_query_result(
                command, receipt=receipt, job_ids=job_ids
            )
        if command[0] == "sacct":
            rows = [
                (
                    f"{old_job_id}|{pilot.SBATCH_JOB_NAME}|NODE_FAIL|0:0|"
                    f"NodeDown|{receipt['slurm']['comment']}|"
                    f"{receipt['slurm']['partition']}|"
                    f"{receipt['slurm']['time_limit']}|"
                    f"{shlex.join(receipt['submit_command'])}"
                )
            ]
            if state["visible"]:
                rows.append(
                    f"{new_job_id}|{pilot.SBATCH_JOB_NAME}|RUNNING|0:0|None|"
                    f"{receipt['slurm']['comment']}|"
                    f"{receipt['slurm']['partition']}|"
                    f"{receipt['slurm']['time_limit']}|"
                    f"{shlex.join(receipt['submit_command'])}"
                )
            return subprocess.CompletedProcess(
                command, 0, "\n".join(rows) + "\n", ""
            )
        if command == receipt["submit_command"]:
            state["sbatch_calls"] += 1
            state["visible"] = True
            return subprocess.CompletedProcess(
                command, 0, f"Submitted batch job {new_job_id}\n", ""
            )
        raise AssertionError(command)

    submitted = pilot.submit_materialization_pilot_sbatch(
        pilot_root=inputs["pilot_root"],
        sbatch_receipt=receipt_path,
        scheduler_user=SUBMISSION_TEST_USER,
        prior_quarantine_seals=[seal_path],
        visibility_timeout=0,
        runner=runner,
        apply=True,
    )
    repeated = pilot.submit_materialization_pilot_sbatch(
        pilot_root=inputs["pilot_root"],
        sbatch_receipt=receipt_path,
        scheduler_user=SUBMISSION_TEST_USER,
        prior_quarantine_seals=[seal_path],
        visibility_timeout=0,
        runner=runner,
        apply=True,
    )

    assert submitted["status"] == "submitted"
    assert submitted["job_id"] == new_job_id
    assert repeated["status"] == "already_submitted"
    assert state["sbatch_calls"] == 1
    intent = json.loads(
        (
            Path(inputs["pilot_root"]) / pilot.SUBMISSION_INTENT_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert [row["job_id"] for row in intent["retired_quarantines"]] == [
        old_job_id
    ]
    assert (
        intent["retired_quarantines"][0]["quarantine_seal"]
        == str(seal_path.resolve())
    )
    with pytest.raises(
        pilot.MaterializationPilotError,
        match="submission intent identity drifted",
    ):
        pilot.submit_materialization_pilot_sbatch(
            pilot_root=inputs["pilot_root"],
            sbatch_receipt=receipt_path,
            scheduler_user=SUBMISSION_TEST_USER,
            visibility_timeout=0,
            runner=runner,
            apply=True,
        )


def test_submission_retry_refuses_unsealed_historical_match(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, receipt, _seal_path = _sealed_pilot_quarantine(
        tmp_path
    )
    old_job_id = "24680"
    sbatch_calls = 0

    def runner(argv):
        nonlocal sbatch_calls
        command = list(argv)
        if command[0] == "squeue":
            return _submission_query_result(command, receipt=receipt)
        if command[0] == "sacct":
            row = (
                f"{old_job_id}|{pilot.SBATCH_JOB_NAME}|NODE_FAIL|0:0|"
                f"NodeDown|{receipt['slurm']['comment']}|"
                f"{receipt['slurm']['partition']}|"
                f"{receipt['slurm']['time_limit']}|"
                f"{shlex.join(receipt['submit_command'])}\n"
            )
            return subprocess.CompletedProcess(command, 0, row, "")
        if command == receipt["submit_command"]:
            sbatch_calls += 1
            return subprocess.CompletedProcess(
                command, 0, "Submitted batch job 24681\n", ""
            )
        raise AssertionError(command)

    with pytest.raises(
        pilot.MaterializationPilotError,
        match="visible pilot job lacks a marker-first sbatch attempt",
    ):
        pilot.submit_materialization_pilot_sbatch(
            pilot_root=inputs["pilot_root"],
            sbatch_receipt=receipt_path,
            scheduler_user=SUBMISSION_TEST_USER,
            visibility_timeout=0,
            runner=runner,
            apply=True,
        )
    assert sbatch_calls == 0


def test_submission_retry_rejects_quarantine_without_accepted_submission(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, receipt, job_id = _rendered_submission_fixture(
        tmp_path
    )
    root = Path(inputs["pilot_root"])
    root.mkdir()
    (root / pilot.INTENT_MARKER).write_text(
        '{"interrupted": true}\n', encoding="utf-8"
    )

    def terminal_runner(argv):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "sacct":
            row = (
                f"{job_id}|{pilot.SBATCH_JOB_NAME}|NODE_FAIL|0:0|NodeDown|"
                f"(null)|{shlex.join(receipt['submit_command'])}\n"
            )
            return subprocess.CompletedProcess(argv, 0, row, "")
        raise AssertionError(argv)

    pilot.quarantine_interrupted_pilot(
        pilot_root=root,
        sbatch_receipt=receipt_path,
        job_id=job_id,
        runner=terminal_runner,
        apply=True,
    )
    seal_path = (
        root.parent
        / "quarantine_evidence"
        / f"partial-job-{job_id}.sealed.json"
    )
    with pytest.raises(
        pilot.MaterializationPilotError,
        match="missing or symlinked retired pilot accepted submission",
    ):
        pilot.submit_materialization_pilot_sbatch(
            pilot_root=root,
            sbatch_receipt=receipt_path,
            scheduler_user=SUBMISSION_TEST_USER,
            prior_quarantine_seals=[seal_path],
        )


def test_submission_retry_rejects_retired_scheduler_state_drift(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, receipt, seal_path = _sealed_pilot_quarantine(
        tmp_path
    )

    def runner(argv):
        command = list(argv)
        if command[0] == "squeue":
            return _submission_query_result(command, receipt=receipt)
        if command[0] == "sacct":
            row = (
                f"24680|{pilot.SBATCH_JOB_NAME}|COMPLETED|0:0|None|"
                f"{receipt['slurm']['comment']}|"
                f"{receipt['slurm']['partition']}|"
                f"{receipt['slurm']['time_limit']}|"
                f"{shlex.join(receipt['submit_command'])}\n"
            )
            return subprocess.CompletedProcess(command, 0, row, "")
        raise AssertionError(command)

    with pytest.raises(
        pilot.MaterializationPilotError,
        match="retired pilot job reappeared with scheduler identity drift",
    ):
        pilot.submit_materialization_pilot_sbatch(
            pilot_root=inputs["pilot_root"],
            sbatch_receipt=receipt_path,
            scheduler_user=SUBMISSION_TEST_USER,
            prior_quarantine_seals=[seal_path],
            visibility_timeout=0,
            runner=runner,
            apply=True,
        )


def test_submission_retry_lineage_is_sorted_and_rejects_tamper_or_wrong_receipt(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, _receipt, first_seal = _sealed_pilot_quarantine(
        tmp_path
    )
    root = Path(inputs["pilot_root"])
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    second_submission_runner, _calls, second_submission_state = (
        _visible_after_sbatch_runner(
            receipt_payload,
            scheduler_job_id="24682",
        )
    )
    second_submission = pilot.submit_materialization_pilot_sbatch(
        pilot_root=root,
        sbatch_receipt=receipt_path,
        scheduler_user=SUBMISSION_TEST_USER,
        prior_quarantine_seals=[first_seal],
        visibility_timeout=0,
        runner=second_submission_runner,
        apply=True,
    )
    assert second_submission["job_id"] == "24682"
    assert second_submission_state["sbatch_calls"] == 1
    (root / pilot.INTENT_MARKER).write_text(
        '{"interrupted": true}\n', encoding="utf-8"
    )

    def second_transient_runner(argv):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "sacct":
            row = (
                f"24682|{pilot.SBATCH_JOB_NAME}|PREEMPTED|0:0|Preempted|"
                f"(null)|sbatch {receipt_payload['sbatch_path']}\n"
            )
            return subprocess.CompletedProcess(argv, 0, row, "")
        raise AssertionError(argv)

    second_report = pilot.quarantine_interrupted_pilot(
        pilot_root=root,
        sbatch_receipt=receipt_path,
        job_id="24682",
        runner=second_transient_runner,
        apply=True,
    )
    assert second_report["status"] == "quarantined_and_sealed"
    second_seal = (
        root.parent
        / "quarantine_evidence"
        / "partial-job-24682.sealed.json"
    )

    plan = pilot.submit_materialization_pilot_sbatch(
        pilot_root=inputs["pilot_root"],
        sbatch_receipt=receipt_path,
        scheduler_user=SUBMISSION_TEST_USER,
        prior_quarantine_seals=[second_seal, first_seal],
    )
    assert [
        row["job_id"] for row in plan["retired_quarantines"]
    ] == ["24680", "24682"]
    with pytest.raises(
        pilot.MaterializationPilotError,
        match="complete ordered prefix",
    ):
        pilot.submit_materialization_pilot_sbatch(
            pilot_root=inputs["pilot_root"],
            sbatch_receipt=receipt_path,
            scheduler_user=SUBMISSION_TEST_USER,
            prior_quarantine_seals=[second_seal],
        )

    alternate_python = tmp_path / "alternate-python"
    alternate_python.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    alternate_python.chmod(0o755)
    alternate = pilot.render_materialization_pilot_sbatch(
        **inputs,
        sbatch_path=tmp_path / "jobs" / "alternate-pilot.sbatch",
        log_dir=tmp_path / "alternate-logs",
        partition="sched_test",
        python_executable=alternate_python,
        apply=True,
    )
    with pytest.raises(
        pilot.MaterializationPilotError,
        match="quarantine (completion|intent) identity drifted",
    ):
        pilot.submit_materialization_pilot_sbatch(
            pilot_root=inputs["pilot_root"],
            sbatch_receipt=alternate["receipt_path"],
            scheduler_user=SUBMISSION_TEST_USER,
            prior_quarantine_seals=[first_seal],
        )

    other_inputs = dict(inputs)
    other_inputs["pilot_root"] = tmp_path / "other-pilot-root"
    other = pilot.render_materialization_pilot_sbatch(
        **other_inputs,
        sbatch_path=tmp_path / "jobs" / "other-root-pilot.sbatch",
        log_dir=tmp_path / "other-root-logs",
        partition="sched_test",
        python_executable=alternate_python,
        apply=True,
    )
    with pytest.raises(
        pilot.MaterializationPilotError,
        match=(
            "quarantine seal identity drifted|"
            "missing or symlinked retired pilot quarantine tree"
        ),
    ):
        pilot.submit_materialization_pilot_sbatch(
            pilot_root=other_inputs["pilot_root"],
            sbatch_receipt=other["receipt_path"],
            scheduler_user=SUBMISSION_TEST_USER,
            prior_quarantine_seals=[first_seal],
        )

    first_tree = (
        root.parent / "quarantine" / f"{root.name}.partial-job-24680"
    )
    first_tree.chmod(0o755)
    with pytest.raises(
        pilot.MaterializationPilotError,
        match="seal verification failed",
    ):
        pilot.submit_materialization_pilot_sbatch(
            pilot_root=inputs["pilot_root"],
            sbatch_receipt=receipt_path,
            scheduler_user=SUBMISSION_TEST_USER,
            prior_quarantine_seals=[first_seal],
        )


def test_submission_retry_rejects_branched_quarantine_lineage(
    tmp_path: Path,
) -> None:
    inputs, receipt_path, receipt, first_seal = _sealed_pilot_quarantine(
        tmp_path
    )
    root = Path(inputs["pilot_root"])
    branch_runner, _calls, branch_state = _visible_after_sbatch_runner(
        receipt,
        scheduler_job_id="24682",
    )
    branch_submission = pilot.submit_materialization_pilot_sbatch(
        pilot_root=root,
        sbatch_receipt=receipt_path,
        scheduler_user=SUBMISSION_TEST_USER,
        visibility_timeout=0,
        runner=branch_runner,
        apply=True,
    )
    assert branch_submission["job_id"] == "24682"
    assert branch_state["sbatch_calls"] == 1
    (root / pilot.INTENT_MARKER).write_text(
        '{"interrupted": true}\n', encoding="utf-8"
    )

    def transient_runner(argv):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "sacct":
            row = (
                f"24682|{pilot.SBATCH_JOB_NAME}|PREEMPTED|0:0|Preempted|"
                f"(null)|sbatch {receipt['sbatch_path']}\n"
            )
            return subprocess.CompletedProcess(argv, 0, row, "")
        raise AssertionError(argv)

    report = pilot.quarantine_interrupted_pilot(
        pilot_root=root,
        sbatch_receipt=receipt_path,
        job_id="24682",
        runner=transient_runner,
        apply=True,
    )
    assert report["status"] == "quarantined_and_sealed"
    branch_seal = (
        root.parent
        / "quarantine_evidence"
        / "partial-job-24682.sealed.json"
    )

    with pytest.raises(
        pilot.MaterializationPilotError,
        match="complete ordered prefix",
    ):
        pilot.submit_materialization_pilot_sbatch(
            pilot_root=root,
            sbatch_receipt=receipt_path,
            scheduler_user=SUBMISSION_TEST_USER,
            prior_quarantine_seals=[first_seal, branch_seal],
        )


def _completed_scheduled_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_requeue: int = 0,
    transactional_submission: bool = True,
) -> tuple[dict, Path, str, object]:
    inputs, _conda_called = _inputs(tmp_path)
    _install_stage_doubles(monkeypatch)
    python = tmp_path / "pilot-python"
    python.write_text("#!/bin/sh\nexit 98\n", encoding="utf-8")
    python.chmod(0o755)
    sbatch = tmp_path / "jobs" / "schema5-pilot.sbatch"
    rendered = pilot.render_materialization_pilot_sbatch(
        **inputs,
        sbatch_path=sbatch,
        log_dir=tmp_path / "logs",
        partition="sched_test",
        python_executable=python,
        apply=True,
    )
    receipt_path = Path(rendered["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    job_id = "13579"
    if transactional_submission:
        submission_runner, _submission_calls, submission_state = (
            _visible_after_sbatch_runner(
                receipt,
                scheduler_job_id=job_id,
            )
        )
        submitted = pilot.submit_materialization_pilot_sbatch(
            pilot_root=inputs["pilot_root"],
            sbatch_receipt=receipt_path,
            scheduler_user=SUBMISSION_TEST_USER,
            visibility_timeout=0,
            runner=submission_runner,
            apply=True,
        )
        assert submitted["status"] == "submitted"
        assert submitted["scheduler_sources"] == ["sacct", "squeue"]
        assert submission_state["sbatch_calls"] == 1

    def active_runner(argv):
        if argv[:2] == ["squeue", "-h"]:
            row = (
                f"{job_id}|{pilot.SBATCH_JOB_NAME}|RUNNING|"
                f"{receipt['slurm']['comment']}|sched_test|11:30:00|1|8G\n"
            )
            return subprocess.CompletedProcess(argv, 0, row, "")
        if argv[:4] == ["scontrol", "show", "job", "-o"]:
            row = (
                f"JobId={job_id} JobName={pilot.SBATCH_JOB_NAME} "
                "JobState=RUNNING Partition=sched_test TimeLimit=11:30:00 "
                f"Requeue={active_requeue} Comment={receipt['slurm']['comment']} "
                f"Command={sbatch.resolve()}\n"
            )
            return subprocess.CompletedProcess(argv, 0, row, "")
        if argv[:3] == ["scontrol", "write", "batch_script"]:
            Path(argv[4]).write_bytes(sbatch.read_bytes())
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(argv)

    pilot.run_materialization_pilot(
        **inputs,
        sbatch_receipt=receipt_path,
        scheduler_job_id=job_id,
        scheduler_runner=active_runner,
        require_scheduler=True,
        apply=True,
    )

    def terminal_runner(argv):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "sacct":
            row = (
                f"{job_id}|{pilot.SBATCH_JOB_NAME}|COMPLETED|0:0|None|"
                f"(null)|sbatch {sbatch.resolve()}\n"
            )
            return subprocess.CompletedProcess(argv, 0, row, "")
        raise AssertionError(argv)

    return inputs, receipt_path, job_id, terminal_runner


def test_scheduler_acceptance_refuses_a_manual_unaccepted_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, receipt, job_id, terminal_runner = _completed_scheduled_pilot(
        tmp_path,
        monkeypatch,
        transactional_submission=False,
    )
    terminal_calls: list[list[str]] = []

    def recording_terminal_runner(argv):
        terminal_calls.append(list(argv))
        return terminal_runner(argv)

    with pytest.raises(
        pilot.MaterializationPilotError,
        match="pilot submission intent|pilot submission acceptance",
    ):
        pilot.accept_pilot_scheduler(
            pilot_root=inputs["pilot_root"],
            sbatch_receipt=receipt,
            job_id=job_id,
            runner=recording_terminal_runner,
            apply=True,
        )

    assert terminal_calls == []
    assert not (
        Path(inputs["pilot_root"]) / pilot.SCHEDULER_ACCEPTANCE_FILENAME
    ).exists()


def test_production_pilot_requires_marker_last_scheduler_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, receipt, job_id, terminal_runner = _completed_scheduled_pilot(
        tmp_path, monkeypatch
    )
    root = Path(inputs["pilot_root"])
    with pytest.raises(
        pilot.MaterializationPilotError,
        match="scheduler acceptance",
    ):
        pilot.verify_materialization_pilot(root)

    dry = pilot.accept_pilot_scheduler(
        pilot_root=root,
        sbatch_receipt=receipt,
        job_id=job_id,
        runner=terminal_runner,
    )
    accepted = pilot.accept_pilot_scheduler(
        pilot_root=root,
        sbatch_receipt=receipt,
        job_id=job_id,
        runner=terminal_runner,
        apply=True,
    )
    repeated = pilot.accept_pilot_scheduler(
        pilot_root=root,
        sbatch_receipt=receipt,
        job_id=job_id,
        runner=terminal_runner,
        apply=True,
    )
    verified = pilot.verify_materialization_pilot(root)

    assert dry["status"] == "dry_run"
    assert accepted["status"] == "accepted"
    assert repeated["status"] == "already_accepted"
    assert accepted["terminal"]["normalized_state"] == "COMPLETED"
    assert accepted["terminal"]["exit_code"] == "0:0"
    assert accepted["terminal"]["squeue_query"] == {
        "argv": ["squeue", "-h", "-j", job_id, "-o", "%i|%j|%T|%k"],
        "returncode": 0,
        "stdout": "",
        "stdout_sha256": hashlib.sha256(b"").hexdigest(),
        "stderr": "",
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }
    assert (
        f"{job_id}|{pilot.SBATCH_JOB_NAME}|COMPLETED|0:0|None|"
        in accepted["terminal"]["sacct_query"]["stdout"]
    )
    assert accepted["terminal"]["sacct_query"]["stdout_sha256"] == hashlib.sha256(
        accepted["terminal"]["sacct_query"]["stdout"].encode("utf-8")
    ).hexdigest()
    active_path = (
        root
        / pilot.SCHEDULER_ATTEMPTS_DIRECTORY
        / job_id
        / pilot.SCHEDULER_ACTIVE_FILENAME
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    assert active["squeue"]["query"]["argv"][0] == "squeue"
    assert active["scontrol"]["query"]["argv"][0] == "scontrol"
    assert active["spool_query"]["argv"][:4] == [
        "scontrol",
        "write",
        "batch_script",
        job_id,
    ]
    assert accepted["effective_requeue"] == 0
    assert verified["scheduler_acceptance"] == {
        "acceptance_id": accepted["acceptance_id"],
        "marker": str(root / pilot.SCHEDULER_ACCEPTANCE_FILENAME),
        "marker_sha256": hashlib.sha256(
            (root / pilot.SCHEDULER_ACCEPTANCE_FILENAME).read_bytes()
        ).hexdigest(),
        "job_id": job_id,
        "receipt": str(receipt.resolve()),
        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "receipt_id": accepted["sbatch_receipt_id"],
        "effective_requeue": 0,
        "spooled_script_sha256": accepted["spooled_script_sha256"],
        "terminal_state": "COMPLETED",
        "exit_code": "0:0",
        "reason": "None",
    }


def test_production_pilot_rejects_effective_requeue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        pilot.MaterializationPilotError,
        match="effective Slurm identity",
    ):
        _completed_scheduled_pilot(
            tmp_path, monkeypatch, active_requeue=1
        )


def test_scheduler_acceptance_rejects_nonzero_terminal_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, receipt, job_id, _terminal_runner = _completed_scheduled_pilot(
        tmp_path, monkeypatch
    )
    sbatch = Path(
        json.loads(receipt.read_text(encoding="utf-8"))["sbatch_path"]
    )

    def failed_runner(argv):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "sacct":
            row = (
                f"{job_id}|{pilot.SBATCH_JOB_NAME}|FAILED|2:0|"
                f"NonZeroExitCode|(null)|sbatch {sbatch}\n"
            )
            return subprocess.CompletedProcess(argv, 0, row, "")
        raise AssertionError(argv)

    with pytest.raises(
        pilot.MaterializationPilotError,
        match=r"COMPLETED\\|0:0",
    ):
        pilot.accept_pilot_scheduler(
            pilot_root=inputs["pilot_root"],
            sbatch_receipt=receipt,
            job_id=job_id,
            runner=failed_runner,
            apply=True,
        )
    assert not (
        Path(inputs["pilot_root"]) / pilot.SCHEDULER_ACCEPTANCE_FILENAME
    ).exists()


def test_scheduler_acceptance_reparses_raw_active_scheduler_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, receipt, job_id, terminal_runner = _completed_scheduled_pilot(
        tmp_path, monkeypatch
    )
    active_path = (
        Path(inputs["pilot_root"])
        / pilot.SCHEDULER_ATTEMPTS_DIRECTORY
        / job_id
        / pilot.SCHEDULER_ACTIVE_FILENAME
    )
    active = json.loads(active_path.read_text(encoding="utf-8"))
    query = active["squeue"]["query"]
    query["stdout"] = query["stdout"].replace("|RUNNING|", "|CONFIGURING|")
    query["stdout_sha256"] = hashlib.sha256(
        query["stdout"].encode("utf-8")
    ).hexdigest()
    identity = dict(active)
    identity.pop("active_id")
    active["active_id"] = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    active_path.chmod(0o644)
    active_path.write_text(
        json.dumps(active, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    active_path.chmod(0o444)

    with pytest.raises(
        pilot.MaterializationPilotError,
        match="active pilot scheduler evidence identity drifted",
    ):
        pilot.accept_pilot_scheduler(
            pilot_root=inputs["pilot_root"],
            sbatch_receipt=receipt,
            job_id=job_id,
            runner=terminal_runner,
            apply=True,
        )


def test_scheduler_acceptance_rejects_wrong_job_and_spooled_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, receipt, job_id, terminal_runner = _completed_scheduled_pilot(
        tmp_path, monkeypatch
    )
    root = Path(inputs["pilot_root"])
    with pytest.raises(
        pilot.MaterializationPilotError,
        match="transactionally accepted submission",
    ):
        pilot.accept_pilot_scheduler(
            pilot_root=root,
            sbatch_receipt=receipt,
            job_id="97531",
            runner=terminal_runner,
            apply=True,
        )

    spooled = (
        root
        / pilot.SCHEDULER_ATTEMPTS_DIRECTORY
        / job_id
        / pilot.SCHEDULER_SPOOLED_SCRIPT_FILENAME
    )
    spooled.chmod(0o644)
    spooled.write_bytes(spooled.read_bytes() + b"# tamper\n")
    spooled.chmod(0o444)
    with pytest.raises(
        pilot.MaterializationPilotError,
        match="spooled batch script",
    ):
        pilot.accept_pilot_scheduler(
            pilot_root=root,
            sbatch_receipt=receipt,
            job_id=job_id,
            runner=terminal_runner,
            apply=True,
        )


def test_scheduler_acceptance_recovers_before_marker_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, receipt, job_id, terminal_runner = _completed_scheduled_pilot(
        tmp_path, monkeypatch
    )
    real_write = pilot._atomic_write_once
    crashed = False

    def crash_once(path, payload):
        nonlocal crashed
        if Path(path).name == pilot.SCHEDULER_ACCEPTANCE_FILENAME and not crashed:
            crashed = True
            raise RuntimeError("scheduler acceptance crash")
        return real_write(path, payload)

    monkeypatch.setattr(pilot, "_atomic_write_once", crash_once)
    with pytest.raises(RuntimeError, match="scheduler acceptance crash"):
        pilot.accept_pilot_scheduler(
            pilot_root=inputs["pilot_root"],
            sbatch_receipt=receipt,
            job_id=job_id,
            runner=terminal_runner,
            apply=True,
        )
    monkeypatch.setattr(pilot, "_atomic_write_once", real_write)
    accepted = pilot.accept_pilot_scheduler(
        pilot_root=inputs["pilot_root"],
        sbatch_receipt=receipt,
        job_id=job_id,
        runner=terminal_runner,
        apply=True,
    )
    assert accepted["status"] == "accepted"


def test_pilot_publication_never_launders_writable_preimage(tmp_path):
    artifact = tmp_path / "PILOT_STAGE.json"
    artifact.write_bytes(b"exact bytes\n")
    artifact.chmod(0o644)

    with pytest.raises(
        pilot.MaterializationPilotError,
        match="conflicting immutable pilot artifact",
    ):
        pilot._atomic_write_once(artifact, b"exact bytes\n")

    assert artifact.stat().st_mode & 0o222
