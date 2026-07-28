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
from scripts import materialize_schema5_release as materialize
from slurm import schema5_control


REPO = Path(__file__).resolve().parent.parent
_REAL_VERIFIED_MATERIALIZATION_BINDING = freeze._verified_materialization_binding
_REAL_VERIFY_BOUND_MATERIALIZATION_EVIDENCE = (
    freeze._verify_bound_materialization_evidence
)


def _toolchain_binding(root: Path) -> dict:
    toolchain = root / "conda"
    binding = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r8-offline-conda-toolchain-v1",
        "release_tag": freeze.REQUIRED_GIT_TAG,
        "chain_namespace": "schema5-v1.2-r8",
        "toolchain_root": str(toolchain),
        "base_prefix": str(toolchain / "base"),
        "completion_marker": {
            "path": str(toolchain / "CONDA_TOOLCHAIN_COMPLETE.json"),
            "sha256": "1" * 64,
            "size": 1,
        },
        "marker_id": "2" * 64,
        "installer_contract": (
            schema5_control.conda_toolchain.PINNED_INSTALLER_CONTRACT.as_dict()
        ),
        "intent_id": "4" * 64,
        "conda_executable": {
            "path": str(toolchain / "base/bin/conda"),
            "sha256": "5" * 64,
            "size": 1,
            "mode": 0o555,
            "link_count": 1,
        },
        "runtime_identity_sha256": "6" * 64,
        "complete_prefix_inventory_sha256": "7" * 64,
        "read_only_probes": {"probe_count": 2},
    }
    binding["binding_id"] = hashlib.sha256(
        (
            json.dumps(
                binding,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return binding


def _package_cache_seed_input(root: Path) -> dict:
    binding = {
        "source_package_cache": str(root / "source-package-cache"),
        "inventory_sha256": "9" * 64,
        "inventory_entry_count": 1,
        "inventory_file_count": 1,
        "inventory_total_file_bytes": 1,
        "requirements_sha256": "a" * 64,
        "required_package_count": 1,
        "archive_count": 0,
        "selected_top_level_entries": ["cache", "urls", "urls.txt"],
    }
    binding["input_id"] = hashlib.sha256(
        (
            json.dumps(
                binding,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return binding


def test_operational_retry_tag_is_distinct_from_stable_release_id() -> None:
    assert freeze.RELEASE_ID == "sweep-recovery-schema5-v1.2"
    assert freeze.REQUIRED_GIT_TAG == "sweep-recovery-schema5-v1.2-r8"
    assert materialize.RELEASE_ID == freeze.RELEASE_ID
    assert materialize.REQUIRED_TAG == freeze.REQUIRED_GIT_TAG
    assert freeze.REQUIRED_GIT_TAG != freeze.RELEASE_ID


def test_probe_environment_rejects_hostile_git_and_scheduler_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = {
        "PATH": "/tmp/attacker-bin",
        "BASH_ENV": "/tmp/attacker-env",
        "LD_PRELOAD": "/tmp/attacker.so",
        "GIT_DIR": "/tmp/attacker-git",
        "GIT_CONFIG_PARAMETERS": "'core.hooksPath=/tmp/attacker-hooks'",
        "GIT_REPLACE_REF_BASE": "refs/attacker",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": "/tmp/attacker-hooks",
        "SBATCH_PARTITION": "attacker",
        "SACCT_FORMAT": "attacker",
        "CONDA_PREFIX": "/tmp/attacker-conda",
        "PIP_INDEX_URL": "https://attacker.invalid/simple",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    environment = freeze._python_probe_environment()
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert not (set(hostile) - {"PATH"}).intersection(environment)


@pytest.fixture(autouse=True)
def _stub_materialization_binding(monkeypatch):
    def binding(*, output_root, release_worktree, harness_prefix, serving_prefix):
        root = Path(output_root).parent
        mutable_source_root = Path(release_worktree).with_name(
            Path(release_worktree).name + ".mutable-source"
        )
        return {
            "schema_version": materialize.SCHEMA_VERSION,
            "release_id": freeze.RELEASE_ID,
            "root": str(root),
            "marker_path": str(root / freeze.MATERIALIZATION_COMPLETE_FILENAME),
            "marker_sha256": "d" * 64,
            "materialization_id": "e" * 64,
            "tag_commit": freeze.verify_clean_exact_tag(release_worktree)["git_commit"],
            "source_tree_sha256": freeze.verify_clean_exact_tag(release_worktree)[
                "source_tree_sha256"
            ],
            "paths": {
                "source_repository": str(mutable_source_root),
                "environment_capture_root": str(root / "environment-capture"),
                "release_worktree": str(release_worktree),
                "source_harness_prefix": str(harness_prefix) + ".source",
                "source_serving_prefix": str(serving_prefix) + ".source",
                "source_package_cache": str(root / "source-package-cache"),
                "harness_prefix": str(harness_prefix),
                "serving_prefix": str(serving_prefix),
            },
            "stage_records": {
                "worktree": {"record_sha256": "1" * 64},
                "harness_clone": {"record_sha256": "2" * 64},
                "serving_clone": {"record_sha256": "3" * 64},
                "package_cache": {"record_sha256": "4" * 64},
                "harness_package": {"record_sha256": "5" * 64},
            },
            "environment_capture": {
                "capture_id": "6" * 64,
                "capture_marker_sha256": "7" * 64,
                "seed_prefixes": {
                    "harness": str(harness_prefix) + ".seed",
                    "serving": str(serving_prefix) + ".seed",
                },
                "ownership_policy_path": str(root / "environment-capture/policy.json"),
                "ownership_policy_sha256": "8" * 64,
                "integrity_normalization_policy_path": str(
                    root / "environment-capture/integrity-policy.json"
                ),
                "integrity_normalization_policy_sha256": "f" * 64,
                "stage_records": {
                    "harness": {
                        "normalization_receipt_id": "9" * 64,
                        "normalized_content_inventory_sha256": "a" * 64,
                    },
                    "serving": {
                        "normalization_receipt_id": "b" * 64,
                        "normalized_content_inventory_sha256": "c" * 64,
                    },
                },
            },
            "conda_creation_tool": {
                "path": str(root / "creation-conda"),
                "sha256": "d" * 64,
            },
            "conda_toolchain": _toolchain_binding(root),
            "package_cache_seed_input": _package_cache_seed_input(root),
            "conda_package_cache_sha256": "e" * 64,
            "conda_package_cache_seed_sha256": "0" * 64,
        }

    monkeypatch.setattr(freeze, "_verified_materialization_binding", binding)
    monkeypatch.setattr(
        freeze,
        "_verify_bound_materialization_evidence",
        lambda evidence, **kwargs: dict(evidence),
    )
    monkeypatch.setattr(
        schema5_control.conda_toolchain,
        "verified_conda_toolchain_binding",
        lambda root, exercise=True: _toolchain_binding(
            Path(root).resolve().parent
        ),
    )


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
    fleet_path = configs / "schema5_fleet.v1.json"
    fleet_payload = json.loads(fleet_path.read_text(encoding="utf-8"))
    fleet_payload["release_id"] = freeze.RELEASE_ID
    fleet_path.write_text(
        json.dumps(fleet_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fleet_path.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(fleet_path.read_bytes()).hexdigest()}  "
        f"{fleet_path.name}\n",
        encoding="utf-8",
    )
    _run("git", "add", "source.txt", "pyproject.toml", "src", "configs", cwd=worktree)
    _run("git", "commit", "-q", "-m", "release", cwd=worktree)
    _run(
        "git",
        "tag",
        "-a",
        freeze.REQUIRED_GIT_TAG,
        "-m",
        "immutable production release",
        cwd=worktree,
    )
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
            '  case "$3" in',
            f"    *SCHEMA5_RELEASE_IMPORT_PROBE*) printf '%s\\n' '{import_json}' ;;",
            f"    *) printf '%s\\n' '{json.dumps(runtime, sort_keys=True)}' ;;",
            "  esac",
            'elif [ "$1" = "-I" ] && [ "$2" = "-m" ] && [ "$3" = "pip" ]; then',
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
    conda_records = (
        {
            "name": "python",
            "version": "3.11.13",
            "url": "https://conda.example.invalid/linux-64/python-3.11.13-h1.conda",
            "sha256": "0123456789abcdef" * 4,
        },
        {
            "name": "pip",
            "version": "25.1",
            "url": "https://conda.example.invalid/noarch/pip-25.1-pyhd8ed1ab_0.conda",
            "sha256": "abcdef0123456789" * 4,
        },
    )
    for record in conda_records:
        record_path = (
            prefix
            / "conda-meta"
            / f"{record['name']}-{record['version']}-test.json"
        )
        record_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    return prefix


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
    }


def _v12_materialization_evidence(root: Path) -> tuple[dict, dict, str]:
    environment_capture = {
        "capture_id": "6" * 64,
        "capture_marker_sha256": "7" * 64,
        "seed_prefixes": {
            "harness": str(root / "environment-capture/seeds/harness"),
            "serving": str(root / "environment-capture/seeds/serving"),
        },
        "ownership_policy_path": str(root / "environment-capture/policy.json"),
        "ownership_policy_sha256": "8" * 64,
        "integrity_normalization_policy_path": str(
            root / "environment-capture/integrity-policy.json"
        ),
        "integrity_normalization_policy_sha256": "f" * 64,
        "stage_records": {
            "harness": {
                "normalization_receipt_id": "9" * 64,
                "normalized_content_inventory_sha256": "a" * 64,
            },
            "serving": {
                "normalization_receipt_id": "b" * 64,
                "normalized_content_inventory_sha256": "c" * 64,
            },
        },
    }
    conda_creation_tool = {
        "path": str(root / "creation-conda"),
        "sha256": "d" * 64,
    }
    return environment_capture, conda_creation_tool, "e" * 64


@pytest.mark.parametrize(
    "field",
    (
        "capture_id",
        "capture_marker_sha256",
        "normalized_content_inventory_sha256",
        "ownership_policy_sha256",
        "integrity_normalization_policy_sha256",
        "normalization_receipt_id",
        "conda_toolchain_binding_id",
        "conda_creation_tool_sha256",
        "conda_package_cache_sha256",
        "conda_package_cache_seed_sha256",
    ),
)
def test_environment_manifest_rejects_self_consistent_upstream_substitution(
    tmp_path, monkeypatch, field
):
    inputs = _inputs(tmp_path)
    prefix = inputs["harness_prefix"]
    git_identity = freeze.verify_clean_exact_tag(inputs["release_worktree"])
    root = tmp_path / "materialization"
    capture_binding, conda_tool, cache_sha = _v12_materialization_evidence(
        root
    )
    binding = {
        "paths": {"harness_prefix": str(prefix)},
        "environment_capture": capture_binding,
        "conda_toolchain": _toolchain_binding(root),
        "conda_creation_tool": conda_tool,
        "conda_package_cache_sha256": cache_sha,
        "conda_package_cache_seed_sha256": "0" * 64,
    }
    inventory = {
        "inventory_sha256": "1" * 64,
        "entry_count": 3,
        "file_count": 2,
        "directory_count": 1,
        "symlink_count": 0,
        "total_file_bytes": 7,
        "entries": [],
    }
    release_package = {"name": "agents_scaling", "editable": False}
    payload = {
        "schema_version": freeze.ENVIRONMENT_SCHEMA_VERSION,
        "release_id": freeze.RELEASE_ID,
        "role": "harness",
        "prefix": str(prefix),
        "sealed_read_only": False,
        "offline_environment": dict(freeze.REQUIRED_OFFLINE_ENVIRONMENT),
        "conda_toolchain": _toolchain_binding(root),
        "conda_creation_tool": dict(conda_tool),
        "environment_seed": {
            "capture_id": capture_binding["capture_id"],
            "capture_marker_sha256": capture_binding[
                "capture_marker_sha256"
            ],
            "prefix": capture_binding["seed_prefixes"]["harness"],
            "normalized_content_inventory_sha256": capture_binding[
                "stage_records"
            ]["harness"]["normalized_content_inventory_sha256"],
        },
        "ownership_policy": {
            "path": capture_binding["ownership_policy_path"],
            "sha256": capture_binding["ownership_policy_sha256"],
        },
        "integrity_normalization_policy": {
            "path": capture_binding[
                "integrity_normalization_policy_path"
            ],
            "sha256": capture_binding[
                "integrity_normalization_policy_sha256"
            ],
        },
        "normalization_receipt": {
            "id": capture_binding["stage_records"]["harness"][
                "normalization_receipt_id"
            ]
        },
        "conda_package_cache_sha256": cache_sha,
        "conda_package_cache_seed_sha256": "0" * 64,
        "runtime": {},
        "locks": {"conda_explicit": [], "pip_freeze_all": []},
        "release_package": release_package,
        "installed_files": {
            "inventory_sha256": inventory["inventory_sha256"],
            "entry_count": inventory["entry_count"],
            "file_count": inventory["file_count"],
            "total_file_bytes": inventory["total_file_bytes"],
        },
        "directory_inventory": inventory,
    }

    def content_sha():
        return freeze._sha256_bytes(
            freeze._canonical_bytes(
                {
                    "runtime": payload["runtime"],
                    "locks": payload["locks"],
                    "release_package": payload["release_package"],
                    "environment_seed": payload["environment_seed"],
                    "ownership_policy": payload["ownership_policy"],
                    "integrity_normalization_policy": payload[
                        "integrity_normalization_policy"
                    ],
                    "normalization_receipt": payload[
                        "normalization_receipt"
                    ],
                    "conda_toolchain": payload["conda_toolchain"],
                    "conda_creation_tool": payload[
                        "conda_creation_tool"
                    ],
                    "conda_package_cache_sha256": payload[
                        "conda_package_cache_sha256"
                    ],
                    "conda_package_cache_seed_sha256": payload[
                        "conda_package_cache_seed_sha256"
                    ],
                    "inventory_sha256": inventory["inventory_sha256"],
                }
            )
        )

    payload["environment_content_sha256"] = content_sha()
    monkeypatch.setattr(freeze, "_conda_lock_from_records", lambda path: [])
    monkeypatch.setattr(
        freeze,
        "_pip_lock_material",
        lambda *args, **kwargs: ([], release_package),
    )
    monkeypatch.setattr(freeze, "_collect_runtime", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        freeze, "directory_inventory", lambda path: dict(inventory)
    )
    freeze._verify_manifest_environment(
        payload,
        path=tmp_path / "harness.manifest.json",
        release_worktree=inputs["release_worktree"],
        git_identity=git_identity,
        materialization_binding=binding,
    )

    if field in {
        "capture_id",
        "capture_marker_sha256",
        "normalized_content_inventory_sha256",
    }:
        payload["environment_seed"][field] = "f" * 64
    elif field == "ownership_policy_sha256":
        payload["ownership_policy"]["sha256"] = "f" * 64
    elif field == "integrity_normalization_policy_sha256":
        payload["integrity_normalization_policy"]["sha256"] = "e" * 64
    elif field == "normalization_receipt_id":
        payload["normalization_receipt"]["id"] = "f" * 64
    elif field == "conda_toolchain_binding_id":
        payload["conda_toolchain"]["binding_id"] = "f" * 64
    elif field == "conda_creation_tool_sha256":
        payload["conda_creation_tool"]["sha256"] = "f" * 64
    elif field == "conda_package_cache_sha256":
        payload["conda_package_cache_sha256"] = "f" * 64
    else:
        payload["conda_package_cache_seed_sha256"] = "f" * 64
    payload["environment_content_sha256"] = content_sha()

    with pytest.raises(
        freeze.ReleaseFreezeError,
        match="not bound to the sealed materialization evidence",
    ):
        freeze._verify_manifest_environment(
            payload,
            path=tmp_path / "harness.manifest.json",
            release_worktree=inputs["release_worktree"],
            git_identity=git_identity,
            materialization_binding=binding,
        )


def _install_real_materialization_evidence(
    tmp_path: Path,
    output: Path,
    inputs: dict,
    monkeypatch,
) -> tuple[Path, Path, Path]:
    """Install minimal sealed evidence accepted by the real bound verifier."""

    root = output.parent
    root.mkdir(parents=True, exist_ok=True)
    source_repository = tmp_path / "mutable-source-repository"
    source_harness = tmp_path / "mutable-source-harness"
    source_serving = tmp_path / "mutable-source-serving"
    for source in (source_repository, source_harness, source_serving):
        source.mkdir()
        (source / "mutable.txt").write_text("not release identity\n", encoding="utf-8")
    git_identity = freeze.verify_clean_exact_tag(inputs["release_worktree"])
    paths = {
        "source_repository": str(source_repository),
        "environment_capture_root": str(root / "environment-capture"),
        "release_worktree": str(inputs["release_worktree"]),
        "source_harness_prefix": str(source_harness),
        "source_serving_prefix": str(source_serving),
        "source_package_cache": str(tmp_path / "source-package-cache"),
        "harness_prefix": str(inputs["harness_prefix"]),
        "serving_prefix": str(inputs["serving_prefix"]),
    }
    stage_records = {}
    environment_capture, conda_creation_tool, package_cache_sha256 = (
        _v12_materialization_evidence(root)
    )
    for name, filename in materialize.STAGE_FILENAMES.items():
        stage_payload = {
            "schema_version": materialize.SCHEMA_VERSION,
            "release_id": freeze.RELEASE_ID,
            "stage": name,
        }
        if name == "package_cache":
            stage_payload["content_inventory"] = {
                "content_inventory_sha256": package_cache_sha256
            }
            stage_payload["package_cache_seed"] = {
                "seed_content_inventory_sha256": "0" * 64
            }
        stage_payload["record_sha256"] = hashlib.sha256(
            materialize._json_bytes(stage_payload)
        ).hexdigest()
        stage_path = root / filename
        stage_path.write_bytes(materialize._json_bytes(stage_payload))
        stage_path.chmod(0o444)
        stage_records[name] = {
            "filename": filename,
            "sha256": hashlib.sha256(stage_path.read_bytes()).hexdigest(),
            "record_sha256": stage_payload["record_sha256"],
        }
    marker = {
        "schema_version": materialize.SCHEMA_VERSION,
        "release_id": freeze.RELEASE_ID,
        "git_tag": freeze.REQUIRED_GIT_TAG,
        "tag_commit": git_identity["git_commit"],
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "paths": paths,
        "complete": True,
        "publication_protocol": "stage_records_fsync_marker_last",
        "stage_records": stage_records,
        "environment_capture": environment_capture,
        "conda_toolchain": _toolchain_binding(root),
        "conda_creation_tool": conda_creation_tool,
        "package_cache_seed_input": _package_cache_seed_input(root),
    }
    marker["materialization_id"] = hashlib.sha256(
        materialize._json_bytes(marker)
    ).hexdigest()
    marker_path = root / freeze.MATERIALIZATION_COMPLETE_FILENAME
    marker_path.write_bytes(materialize._json_bytes(marker))
    marker_path.chmod(0o444)
    report = {
        "status": "verified",
        "release_id": freeze.RELEASE_ID,
        "materialization_id": marker["materialization_id"],
        "tag_commit": git_identity["git_commit"],
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "paths": paths,
        "environment_capture": environment_capture,
        "conda_toolchain": _toolchain_binding(root),
        "conda_creation_tool": conda_creation_tool,
        "package_cache_seed_input": _package_cache_seed_input(root),
        "conda_package_cache_sha256": package_cache_sha256,
        "conda_package_cache_seed_sha256": "0" * 64,
    }
    monkeypatch.setattr(materialize, "verify_materialization", lambda path: dict(report))
    monkeypatch.setattr(
        freeze, "_verified_materialization_binding", _REAL_VERIFIED_MATERIALIZATION_BINDING
    )
    monkeypatch.setattr(
        freeze,
        "_verify_bound_materialization_evidence",
        _REAL_VERIFY_BOUND_MATERIALIZATION_EVIDENCE,
    )
    return source_repository, source_harness, source_serving


def _publish_bundle_consumable_by_schema5_control(
    tmp_path: Path, monkeypatch
) -> tuple[Path, dict, dict]:
    """Publish real freezer bytes and derive only the control-plane envelope.

    ``build_immutable_pins`` additionally requires all 22,680 cloned run entries and
    the complete production controller scripts.  Those are orthogonal to the release
    producer/consumer contract exercised here, so this fixture calls the exact sealed
    release-bundle validator that ``build_immutable_pins`` reaches.
    """

    inputs = _inputs(tmp_path)
    output = tmp_path / "materialization" / "identity"
    _install_real_materialization_evidence(tmp_path, output, inputs, monkeypatch)
    created = freeze.create_release_bundle(
        output,
        apply=True,
        seal_worktree=True,
        seal_environments=True,
        seal_output_root=True,
        **inputs,
    )
    identity = json.loads(
        (output / freeze.RELEASE_IDENTITY_FILENAME).read_text(encoding="utf-8")
    )
    marker = json.loads(
        (output / freeze.COMPLETE_MARKER_FILENAME).read_text(encoding="utf-8")
    )
    pins = {
        **identity["control_pin_fragment"],
        "release_bundle_root": str(output),
        "release_bundle_id": marker["release_bundle_id"],
    }
    assert created["release_bundle_id"] == marker["release_bundle_id"]
    return output, identity, pins


def _republish_mutated_bundle_for_negative_test(
    output: Path, identity: dict
) -> dict:
    """Re-address an intentionally corrupted bundle so validation reaches its cause."""

    output.chmod(0o755)
    for role, filename in (
        ("harness", freeze.HARNESS_MANIFEST_FILENAME),
        ("serving", freeze.SERVING_MANIFEST_FILENAME),
    ):
        manifest_path = output / filename
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        identity["environments"][role]["manifest_sha256"] = digest
        identity["control_pin_fragment"][f"{role}_environment_sha256"] = digest

    identity_path = output / freeze.RELEASE_IDENTITY_FILENAME
    identity_path.chmod(0o644)
    identity_path.write_bytes(freeze._json_bytes(identity))

    primary_names = (
        freeze.HARNESS_MANIFEST_FILENAME,
        freeze.SERVING_MANIFEST_FILENAME,
        freeze.RELEASE_IDENTITY_FILENAME,
    )
    for filename in primary_names:
        path = output / filename
        checksum = output / (filename + freeze.CHECKSUM_SUFFIX)
        checksum.chmod(0o644)
        checksum.write_bytes(
            freeze._checksum_bytes(filename, path.read_bytes())
        )

    marker_path = output / freeze.COMPLETE_MARKER_FILENAME
    marker_path.chmod(0o644)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifacts"] = {
        path.name: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in sorted(output.iterdir())
        if path.name != freeze.COMPLETE_MARKER_FILENAME
    }
    marker.pop("release_bundle_id", None)
    marker["release_bundle_id"] = schema5_control.sha256_value(marker)
    marker_path.write_bytes(freeze._json_bytes(marker))
    for path in output.iterdir():
        path.chmod(0o444)
    output.chmod(0o555)
    return {
        **identity["control_pin_fragment"],
        "release_bundle_root": str(output),
        "release_bundle_id": marker["release_bundle_id"],
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
    assert identity["git_tag_object"] == _run(
        "git",
        "rev-parse",
        f"refs/tags/{freeze.REQUIRED_GIT_TAG}",
        cwd=worktree,
    )
    assert identity["git_tag_object"] != identity["git_commit"]
    assert identity["source_tree_sha256"] == schema5_control.sha256_tree(worktree)
    (worktree / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(freeze.ReleaseFreezeError, match="not clean"):
        freeze.verify_clean_exact_tag(worktree)


def test_git_identity_rejects_source_or_tag_drift_during_tree_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = _tagged_worktree(tmp_path)
    original_git = freeze._git
    status_calls = 0

    def drifting_git(root: Path, *arguments: str) -> str:
        nonlocal status_calls
        result = original_git(root, *arguments)
        if arguments[:2] == (
            "status",
            "--porcelain=v1",
        ):
            status_calls += 1
            if status_calls == 2:
                return " M source.txt"
        return result

    monkeypatch.setattr(freeze, "_git", drifting_git)
    with pytest.raises(
        freeze.ReleaseFreezeError,
        match="changed during source hashing",
    ):
        freeze.verify_clean_exact_tag(worktree)


def test_git_identity_rechecks_influential_ignored_files_after_tree_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = _tagged_worktree(tmp_path)
    exclude = worktree / ".git" / "info" / "exclude"
    exclude.write_text(
        exclude.read_text(encoding="utf-8") + "\nlate-ignored.bin\n",
        encoding="utf-8",
    )
    original_tree_hash = freeze.sha256_tree
    hash_calls = 0

    def create_ignored_file_after_hash(root: Path) -> str:
        nonlocal hash_calls
        digest = original_tree_hash(root)
        hash_calls += 1
        if hash_calls == 1:
            (root / "late-ignored.bin").write_bytes(b"untracked hash input\n")
        return digest

    monkeypatch.setattr(freeze, "sha256_tree", create_ignored_file_after_hash)
    with pytest.raises(
        freeze.ReleaseFreezeError,
        match="changed during source hashing",
    ):
        freeze.verify_clean_exact_tag(worktree)


def test_git_identity_requires_stable_second_source_tree_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree = _tagged_worktree(tmp_path)
    tree_hashes = iter(("1" * 64, "2" * 64))

    monkeypatch.setattr(freeze, "sha256_tree", lambda _root: next(tree_hashes))
    with pytest.raises(
        freeze.ReleaseFreezeError,
        match="not stable across final replay",
    ):
        freeze.verify_clean_exact_tag(worktree)


def test_runtime_probe_is_isolated_and_cannot_write_bytecode(tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("python\n", encoding="utf-8")
    observed = {}
    runtime = {
        "python_version": "3.11.13",
        "python_implementation": "CPython",
        "cuda_version": None,
        "packages": {
            "torch": None,
            "vllm": None,
            "transformers": "4.57.1",
            "tokenizers": "0.22.1",
        },
    }

    def fake_run(argv, *, env=None):
        observed["argv"] = tuple(argv)
        observed["env"] = dict(env or {})
        return json.dumps(runtime)

    monkeypatch.setattr(freeze, "_run", fake_run)

    assert freeze._collect_runtime(prefix, role="harness") == runtime
    assert observed["argv"][1:3] == ("-I", "-c")
    assert observed["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert observed["env"]["PYTHONNOUSERSITE"] == "1"


def test_release_binding_requires_exact_verified_sibling_materialization(
    tmp_path, monkeypatch
):
    inputs = _inputs(tmp_path)
    output = tmp_path / "release-root" / "identity"
    root = output.parent
    root.mkdir()
    marker_path = root / freeze.MATERIALIZATION_COMPLETE_FILENAME
    git_identity = freeze.verify_clean_exact_tag(inputs["release_worktree"])
    paths = {
        "source_repository": str(inputs["release_worktree"]),
        "environment_capture_root": str(root / "environment-capture"),
        "release_worktree": str(inputs["release_worktree"]),
        "source_harness_prefix": str(tmp_path / "source-harness"),
        "source_serving_prefix": str(tmp_path / "source-serving"),
        "source_package_cache": str(tmp_path / "source-package-cache"),
        "harness_prefix": str(inputs["harness_prefix"]),
        "serving_prefix": str(inputs["serving_prefix"]),
    }
    stage_records = {}
    environment_capture, conda_creation_tool, package_cache_sha256 = (
        _v12_materialization_evidence(root)
    )
    for name, filename in materialize.STAGE_FILENAMES.items():
        stage_payload = {
            "schema_version": materialize.SCHEMA_VERSION,
            "release_id": freeze.RELEASE_ID,
            "stage": name,
        }
        if name == "package_cache":
            stage_payload["content_inventory"] = {
                "content_inventory_sha256": package_cache_sha256
            }
            stage_payload["package_cache_seed"] = {
                "seed_content_inventory_sha256": "0" * 64
            }
        stage_payload["record_sha256"] = hashlib.sha256(
            materialize._json_bytes(stage_payload)
        ).hexdigest()
        stage_path = root / filename
        stage_path.write_bytes(materialize._json_bytes(stage_payload))
        stage_path.chmod(0o444)
        stage_records[name] = {
            "filename": filename,
            "sha256": hashlib.sha256(stage_path.read_bytes()).hexdigest(),
            "record_sha256": stage_payload["record_sha256"],
        }
    marker = {
        "schema_version": materialize.SCHEMA_VERSION,
        "release_id": freeze.RELEASE_ID,
        "git_tag": freeze.REQUIRED_GIT_TAG,
        "tag_commit": git_identity["git_commit"],
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "paths": paths,
        "complete": True,
        "stage_records": stage_records,
        "environment_capture": environment_capture,
        "conda_toolchain": _toolchain_binding(root),
        "conda_creation_tool": conda_creation_tool,
        "package_cache_seed_input": _package_cache_seed_input(root),
    }
    marker["materialization_id"] = hashlib.sha256(
        materialize._json_bytes(marker)
    ).hexdigest()
    marker_path.write_bytes(materialize._json_bytes(marker))
    marker_path.chmod(0o444)
    report = {
        "status": "verified",
        "release_id": freeze.RELEASE_ID,
        "materialization_id": marker["materialization_id"],
        "tag_commit": git_identity["git_commit"],
        "source_tree_sha256": git_identity["source_tree_sha256"],
        "paths": paths,
        "environment_capture": environment_capture,
        "conda_toolchain": _toolchain_binding(root),
        "conda_creation_tool": conda_creation_tool,
        "package_cache_seed_input": _package_cache_seed_input(root),
        "conda_package_cache_sha256": package_cache_sha256,
        "conda_package_cache_seed_sha256": "0" * 64,
    }
    monkeypatch.setattr(materialize, "verify_materialization", lambda path: dict(report))

    binding = _REAL_VERIFIED_MATERIALIZATION_BINDING(
        output_root=output,
        release_worktree=inputs["release_worktree"],
        harness_prefix=inputs["harness_prefix"],
        serving_prefix=inputs["serving_prefix"],
    )

    assert binding["marker_path"] == str(marker_path)
    assert binding["marker_sha256"] == hashlib.sha256(marker_path.read_bytes()).hexdigest()
    assert binding["materialization_id"] == marker["materialization_id"]
    assert binding["stage_records"] == stage_records
    assert _REAL_VERIFY_BOUND_MATERIALIZATION_EVIDENCE(
        binding,
        output_root=output,
        release_worktree=inputs["release_worktree"],
        harness_prefix=inputs["harness_prefix"],
        serving_prefix=inputs["serving_prefix"],
        git_identity=git_identity,
    ) == binding

    stage_path = root / materialize.STAGE_FILENAMES["serving_clone"]
    stage_path.chmod(0o644)
    with pytest.raises(freeze.ReleaseFreezeError, match="stage artifact drifted"):
        _REAL_VERIFY_BOUND_MATERIALIZATION_EVIDENCE(
            binding,
            output_root=output,
            release_worktree=inputs["release_worktree"],
            harness_prefix=inputs["harness_prefix"],
            serving_prefix=inputs["serving_prefix"],
            git_identity=git_identity,
        )
    stage_path.chmod(0o444)

    marker_path.chmod(0o644)
    marker_path.unlink()
    with pytest.raises(freeze.ReleaseFreezeError, match="missing regular sibling"):
        _REAL_VERIFIED_MATERIALIZATION_BINDING(
            output_root=output,
            release_worktree=inputs["release_worktree"],
            harness_prefix=inputs["harness_prefix"],
            serving_prefix=inputs["serving_prefix"],
        )


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


def test_schema5_control_consumes_exact_v12_freezer_output(
    tmp_path, monkeypatch
):
    output, identity, pins = _publish_bundle_consumable_by_schema5_control(
        tmp_path, monkeypatch
    )

    # This is the same deep release-bundle boundary called by prepare-pins and every
    # file-verified immutable control load.  The consumer receives the producer's
    # exact control_pin_fragment rather than a hand-reconstructed schema fixture.
    schema5_control._validate_release_bundle(pins)

    assert identity["schema_version"] == 5
    assert identity["transport_uncertainty"]["binding"] == (
        schema5_control.scheduler_safety.expected_transport_uncertainty_binding()
    )
    assert identity["transport_uncertainty"]["binding_sha256"] == (
        pins["transport_uncertainty_binding_sha256"]
    )
    assert identity["transport_uncertainty"]["source_tree_sha256"] == (
        pins["source_tree_sha256"]
    )
    assert identity["materialization"]["schema_version"] == 5
    assert {
        json.loads(
            (output / filename).read_text(encoding="utf-8")
        )["schema_version"]
        for filename in (
            freeze.HARNESS_MANIFEST_FILENAME,
            freeze.SERVING_MANIFEST_FILENAME,
        )
    } == {4}

    downgraded = dict(pins)
    downgraded["release_id"] = "sweep-recovery-schema5-v1.1"
    with pytest.raises(
        schema5_control.ImmutablePinError,
        match="release completion marker identity is invalid",
    ):
        schema5_control._validate_release_bundle(downgraded)


def test_schema5_control_rejects_environment_schema1_freezer_downgrade(
    tmp_path, monkeypatch
):
    output, identity, _ = _publish_bundle_consumable_by_schema5_control(
        tmp_path, monkeypatch
    )
    manifest_path = output / freeze.HARNESS_MANIFEST_FILENAME
    output.chmod(0o755)
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest_path.write_bytes(freeze._json_bytes(manifest))
    pins = _republish_mutated_bundle_for_negative_test(output, identity)

    with pytest.raises(
        schema5_control.ImmutablePinError,
        match="harness environment manifest is not sealed schema 4",
    ):
        schema5_control._validate_release_bundle(pins)


def test_schema5_control_rejects_materialization_schema2_freezer_downgrade(
    tmp_path, monkeypatch
):
    output, identity, _ = _publish_bundle_consumable_by_schema5_control(
        tmp_path, monkeypatch
    )
    marker_path = output.parent / freeze.MATERIALIZATION_COMPLETE_FILENAME
    marker_path.chmod(0o644)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["schema_version"] = 2
    marker.pop("materialization_id")
    marker["materialization_id"] = hashlib.sha256(
        materialize._json_bytes(marker)
    ).hexdigest()
    marker_path.write_bytes(materialize._json_bytes(marker))
    marker_path.chmod(0o444)
    identity["materialization"]["schema_version"] = 2
    identity["materialization"]["materialization_id"] = marker[
        "materialization_id"
    ]
    identity["materialization"]["marker_sha256"] = hashlib.sha256(
        marker_path.read_bytes()
    ).hexdigest()
    pins = _republish_mutated_bundle_for_negative_test(output, identity)

    with pytest.raises(
        schema5_control.ImmutablePinError,
        match="release materialization binding is invalid",
    ):
        schema5_control._validate_release_bundle(pins)


def test_schema5_control_rejects_environment_seed_provenance_drift(
    tmp_path, monkeypatch
):
    output, identity, _ = _publish_bundle_consumable_by_schema5_control(
        tmp_path, monkeypatch
    )
    manifest_path = output / freeze.HARNESS_MANIFEST_FILENAME
    output.chmod(0o755)
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["environment_seed"]["capture_id"] = "f" * 64
    manifest_path.write_bytes(freeze._json_bytes(manifest))
    pins = _republish_mutated_bundle_for_negative_test(output, identity)

    with pytest.raises(
        schema5_control.ImmutablePinError,
        match="environment manifest content identity is invalid",
    ):
        schema5_control._validate_release_bundle(pins)


def test_completed_bundle_recovers_interrupted_output_root_seal(
    tmp_path, monkeypatch
):
    inputs = _inputs(tmp_path)
    output = tmp_path / "release-root" / "identity"
    real_seal = freeze._seal_output_root_read_only

    def simulate_crash(root):
        raise OSError("simulated crash after completion marker")

    monkeypatch.setattr(freeze, "_seal_output_root_read_only", simulate_crash)
    with pytest.raises(OSError, match="simulated crash"):
        freeze.create_release_bundle(
            output,
            apply=True,
            seal_output_root=True,
            **inputs,
        )

    assert (output / freeze.COMPLETE_MARKER_FILENAME).is_file()
    assert stat.S_IMODE(output.stat().st_mode) & 0o222
    monkeypatch.setattr(freeze, "_seal_output_root_read_only", real_seal)

    recovered = freeze.create_release_bundle(
        output,
        apply=True,
        seal_output_root=True,
        **inputs,
    )

    assert recovered["status"] == "already_complete"
    assert recovered["output_root_sealed_read_only"] is True
    assert stat.S_IMODE(output.stat().st_mode) & 0o222 == 0
    # Leave the temporary tree removable by the test runner.
    output.chmod(0o755)


def test_completed_bundle_rejects_incompatible_retroactive_seals(tmp_path):
    inputs = _inputs(tmp_path)
    output = tmp_path / "release-root"
    freeze.create_release_bundle(output, apply=True, **inputs)

    with pytest.raises(freeze.ReleaseFreezeError, match="unsealed worktree"):
        freeze.create_release_bundle(
            output,
            apply=True,
            seal_worktree=True,
            **inputs,
        )
    with pytest.raises(freeze.ReleaseFreezeError, match="unsealed environments"):
        freeze.create_release_bundle(
            output,
            apply=True,
            seal_environments=True,
            **inputs,
        )


def test_creation_uses_direct_conda_meta_lock_without_external_conda_query(
    tmp_path, monkeypatch
):
    inputs = _inputs(tmp_path)
    record_path = inputs["harness_prefix"] / "conda-meta" / "pip-25.1-test.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["sha256"] = "f" * 64
    record_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    real_run = freeze._run

    def reject_conda_lock_query(argv, **kwargs):
        assert not (
            "list" in argv and "--explicit" in argv
        ), "release freezing must never invoke `conda list --explicit`"
        return real_run(argv, **kwargs)

    monkeypatch.setattr(freeze, "_run", reject_conda_lock_query)

    output = tmp_path / "release-root"
    freeze.create_release_bundle(output, apply=True, **inputs)
    manifest = json.loads(
        (output / freeze.HARNESS_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert any(
        row.endswith("#" + "f" * 64)
        for row in manifest["locks"]["conda_explicit"]
    )
    assert manifest["conda_creation_tool"]["sha256"] == "d" * 64
    assert not hasattr(freeze, "_conda_lock")
    assert not hasattr(freeze, "_resolve_conda_executable")


def test_release_verifier_rejects_uninventoried_bundle_entry(
    tmp_path, monkeypatch
):
    inputs = _inputs(tmp_path)
    output = tmp_path / "release-root" / "identity"
    _install_real_materialization_evidence(tmp_path, output, inputs, monkeypatch)
    freeze.create_release_bundle(output, apply=True, **inputs)
    unexpected = output / "UNINVENTORIED"
    unexpected.write_text("not part of the release\n", encoding="utf-8")

    with pytest.raises(
        freeze.ReleaseFreezeError,
        match="unexpected or missing entries",
    ):
        freeze.verify_release_bundle(output)

    assert unexpected.read_text(encoding="utf-8") == "not part of the release\n"


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


def test_sealed_release_verifies_without_mutable_git_repository_metadata(
    tmp_path, monkeypatch
):
    inputs = _inputs(tmp_path)
    output = tmp_path / "release-root" / "identity"
    freeze.create_release_bundle(
        output,
        apply=True,
        seal_worktree=True,
        seal_environments=True,
        **inputs,
    )
    monkeypatch.setattr(
        freeze,
        "verify_clean_exact_tag",
        lambda path: (_ for _ in ()).throw(AssertionError("mutable Git lookup")),
    )

    assert freeze.verify_release_bundle(output)["status"] == "verified"


def test_sealed_release_verification_is_self_contained_after_sources_disappear(
    tmp_path, monkeypatch
):
    inputs = _inputs(tmp_path)
    output = tmp_path / "materialized-release" / "identity"
    mutable_sources = _install_real_materialization_evidence(
        tmp_path, output, inputs, monkeypatch
    )
    freeze.create_release_bundle(
        output,
        apply=True,
        seal_worktree=True,
        seal_environments=True,
        **inputs,
    )

    # The external creation tool, all mutable materialization sources, and Git's
    # repository metadata are deliberately unavailable during the future audit.
    for filename in (
        freeze.HARNESS_MANIFEST_FILENAME,
        freeze.SERVING_MANIFEST_FILENAME,
    ):
        manifest = json.loads((output / filename).read_text(encoding="utf-8"))
        assert manifest["conda_creation_tool"] == {
            "path": str(output.parent / "creation-conda"),
            "sha256": "d" * 64,
        }
        assert manifest["conda_package_cache_sha256"] == "e" * 64
        assert manifest["conda_package_cache_seed_sha256"] == "0" * 64
    for source in mutable_sources:
        shutil.rmtree(source)
    worktree = inputs["release_worktree"]
    git_metadata = worktree / ".git"
    for directory, _directory_names, _file_names in os.walk(
        git_metadata, topdown=False
    ):
        Path(directory).chmod(0o755)
    worktree.chmod(0o755)
    shutil.rmtree(git_metadata)
    freeze._seal_tree_read_only(worktree)
    assert all(not source.exists() for source in mutable_sources)
    assert not git_metadata.exists()

    verified = freeze.verify_release_bundle(output)

    assert verified["status"] == "verified"


def test_release_package_binding_deduplicates_python_directory_symlink_alias(tmp_path):
    inputs = _inputs(tmp_path)
    prefix = inputs["harness_prefix"]
    (prefix / "lib" / "python3.1").symlink_to("python3.11", target_is_directory=True)
    git_identity = freeze.verify_clean_exact_tag(inputs["release_worktree"])

    lock, binding = freeze._pip_lock_material(
        prefix,
        release_worktree=inputs["release_worktree"],
        git_identity=git_identity,
        require_release_package=True,
    )

    assert any(row.startswith("agents_scaling @ file://") for row in lock)
    assert binding is not None
    assert binding["name"] == "agents_scaling"
    assert binding["version"] == "0.1.0"


def test_release_package_binding_rejects_distinct_duplicate_distribution(tmp_path):
    inputs = _inputs(tmp_path)
    prefix = inputs["harness_prefix"]
    duplicate = (
        prefix
        / "lib"
        / "python3.10"
        / "site-packages"
        / "agents_scaling-0.1.0.dist-info"
    )
    duplicate.mkdir(parents=True)
    (duplicate / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: agents_scaling\nVersion: 0.1.0\n\n",
        encoding="utf-8",
    )
    git_identity = freeze.verify_clean_exact_tag(inputs["release_worktree"])

    with pytest.raises(
        freeze.ReleaseFreezeError,
        match="exactly one installed agents_scaling distribution",
    ):
        freeze._pip_lock_material(
            prefix,
            release_worktree=inputs["release_worktree"],
            git_identity=git_identity,
            require_release_package=True,
        )


def test_local_or_editable_pip_requirement_fails_closed(tmp_path, monkeypatch):
    prefix = _fake_environment(tmp_path, "harness", serving=False)
    original_run = freeze._run

    def fake_run(argv, *, env=None):
        if tuple(argv[1:6]) == ("-I", "-m", "pip", "freeze", "--all"):
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
        if tuple(argv[1:6]) == ("-I", "-m", "pip", "freeze", "--all"):
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
        if tuple(argv[1:3]) == ("-I", "-c"):
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


def test_release_publication_never_launders_writable_preimage(tmp_path):
    artifact = tmp_path / "RELEASE_STAGE.json"
    artifact.write_bytes(b"exact bytes\n")
    artifact.chmod(0o644)

    with pytest.raises(
        freeze.ReleaseFreezeError,
        match="conflicting immutable release artifact",
    ):
        freeze._atomic_write_exact(artifact, b"exact bytes\n")

    assert artifact.stat().st_mode & 0o222


def test_release_bundle_and_inventoried_inputs_must_be_fully_disjoint(tmp_path):
    worktree = tmp_path / "release" / "worktree"
    harness = tmp_path / "harness"
    serving = tmp_path / "serving"
    for path in (worktree, harness, serving):
        path.mkdir(parents=True)

    with pytest.raises(freeze.ReleaseFreezeError, match="overlaps"):
        freeze._verify_nonoverlap(
            tmp_path / "release",
            (worktree, harness, serving),
        )

    nested_harness = worktree / "environment"
    nested_harness.mkdir()
    with pytest.raises(
        freeze.ReleaseFreezeError,
        match="inventoried release inputs overlap",
    ):
        freeze._verify_nonoverlap(
            tmp_path / "identity",
            (worktree, nested_harness, serving),
        )
