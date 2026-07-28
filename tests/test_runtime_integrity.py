from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from agents_scaling import runtime_integrity as integrity
from slurm import dispatch_sweeps
from agents_scaling.experiment import run_one
from agents_scaling.experiment.qid_checkpoint import CoordinateAdmissionClosed


# Most tests below exercise the backward-compatible schema-1 verifier.  Keep their
# synthetic release on the retired protocol; v1.2 has dedicated schema-4 fixtures
# and must never accept a schema downgrade.
RELEASE_ID = "sweep-recovery-schema5-v1.1"
RELEASE_ID_V12 = "sweep-recovery-schema5-v1.2"
RELEASE_BUNDLE_ID = "a" * 64
IMMUTABLE_PINS_SHA256 = "b" * 64


def _environment(tmp_path: Path, role: str) -> dict[str, str]:
    prefix = tmp_path / f"{role}-prefix"
    package = prefix / "lib" / "python3.11" / "site-packages" / f"{role}_pkg.py"
    package.parent.mkdir(parents=True)
    package.write_text(f"ROLE = {role!r}\n", encoding="utf-8")
    python = prefix / "bin" / "python"
    python.parent.mkdir()
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    for path in sorted(prefix.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    python.chmod(0o555)
    prefix.chmod(0o555)
    inventory = integrity.directory_inventory(prefix)
    runtime = {"python_version": "3.11.0", "packages": {}}
    locks = {"conda_explicit": [], "pip_freeze_all": []}
    manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "role": role,
        "prefix": str(prefix.resolve()),
        "sealed_read_only": True,
        "offline_environment": {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "runtime": runtime,
        "locks": locks,
        "release_package": None,
        "directory_inventory": inventory,
    }
    manifest["environment_content_sha256"] = hashlib.sha256(
        integrity.canonical_bytes(
            {
                "runtime": runtime,
                "locks": locks,
                "release_package": None,
                "inventory_sha256": inventory["inventory_sha256"],
            }
        )
    ).hexdigest()
    manifest_path = tmp_path / f"{role}.environment.json"
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(raw)
    return {
        "prefix": str(prefix.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "nested_file": str(package),
    }


def _schema4_environment(tmp_path: Path, role: str) -> dict[str, str]:
    environment = _environment(tmp_path, role)
    manifest_path = Path(environment["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = manifest["directory_inventory"]
    locks = {
        "conda_explicit": [
            "@EXPLICIT",
            f"https://repo.example.invalid/{role}-runtime.conda#{'1' * 64}",
        ],
        "pip_freeze_all": [f"{role}-runtime==1.0.0"],
    }
    provenance = {
        "conda_creation_tool": {
            "path": f"/sealed/build-tools/{role}/conda",
            "sha256": "2" * 64,
        },
        "conda_toolchain": {
            "schema_version": 1,
            "protocol": "schema5-v1.2-r13-offline-conda-toolchain-v1",
            "release_tag": "sweep-recovery-schema5-v1.2-r13",
            "chain_namespace": "schema5-v1.2-r13",
            "toolchain_root": "/sealed/build-tools/schema5-v1.2-r13",
            "base_prefix": "/sealed/build-tools/schema5-v1.2-r13/base",
            "completion_marker": {
                "path": (
                    "/sealed/build-tools/schema5-v1.2-r13/"
                    "CONDA_TOOLCHAIN_COMPLETE.json"
                ),
                "sha256": "b" * 64,
                "size": 4096,
            },
            "marker_id": "c" * 64,
            "installer_contract": {
                "filename": "Miniforge3-Linux-x86_64.sh",
                "release": "Miniforge3-25.11.0-1",
                "sha256": "d" * 64,
                "conda_version": "25.11.0",
            },
            "intent_id": "e" * 64,
            "conda_executable": {
                "path": "/sealed/build-tools/schema5-v1.2-r13/base/bin/conda",
                "sha256": "f" * 64,
                "size": 512,
                "mode": 0o555,
                "link_count": 1,
            },
            "runtime_identity_sha256": "1" * 64,
            "complete_prefix_inventory_sha256": "2" * 64,
            "read_only_probes": {"offline": True},
            "binding_id": "3" * 64,
        },
        "environment_seed": {
            "capture_id": "3" * 64,
            "capture_marker_sha256": "4" * 64,
            "prefix": f"/retired/build-inputs/{role}-seed",
            "normalized_content_inventory_sha256": "5" * 64,
        },
        "ownership_policy": {
            "path": "/retired/build-inputs/environment_ownership_policy.v1.json",
            "sha256": "6" * 64,
        },
        "integrity_normalization_policy": {
            "path": (
                "/retired/build-inputs/"
                "environment_integrity_normalization_policy.v1.json"
            ),
            "sha256": "9" * 64,
        },
        "normalization_receipt": {"id": "7" * 64},
        "conda_package_cache_sha256": "8" * 64,
        "conda_package_cache_seed_sha256": "a" * 64,
    }
    manifest.update(
        {
            "schema_version": 4,
            "release_id": RELEASE_ID_V12,
            "offline_environment": {
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            **provenance,
            "locks": locks,
            "installed_files": {
                "inventory_sha256": inventory["inventory_sha256"],
                "entry_count": inventory["entry_count"],
                "file_count": inventory["file_count"],
                "total_file_bytes": inventory["total_file_bytes"],
            },
        }
    )
    manifest["environment_content_sha256"] = hashlib.sha256(
        integrity.canonical_bytes(
            {
                "runtime": manifest["runtime"],
                "locks": locks,
                "release_package": manifest["release_package"],
                "environment_seed": provenance["environment_seed"],
                "ownership_policy": provenance["ownership_policy"],
                "integrity_normalization_policy": provenance[
                    "integrity_normalization_policy"
                ],
                "normalization_receipt": provenance["normalization_receipt"],
                "conda_creation_tool": provenance["conda_creation_tool"],
                "conda_toolchain": provenance["conda_toolchain"],
                "conda_package_cache_sha256": provenance[
                    "conda_package_cache_sha256"
                ],
                "conda_package_cache_seed_sha256": provenance[
                    "conda_package_cache_seed_sha256"
                ],
                "inventory_sha256": inventory["inventory_sha256"],
            }
        )
    ).hexdigest()
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(raw)
    environment["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    return environment


def _rewrite_environment_manifest(
    environment: dict[str, str], transform
) -> dict[str, object]:
    path = Path(environment["manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    transform(manifest)
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    environment["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    return manifest


def test_schema4_manifest_verifies_without_mutable_build_inputs(tmp_path):
    environment = _schema4_environment(tmp_path, "harness")

    observed = integrity.verify_environment_manifest_live(
        role="harness",
        prefix=Path(environment["prefix"]),
        manifest_path=Path(environment["manifest_path"]),
        expected_manifest_sha256=environment["manifest_sha256"],
        release_id=RELEASE_ID_V12,
    )

    assert observed["manifest_sha256"] == environment["manifest_sha256"]
    # None of these recorded provenance paths exists.  Successful verification proves
    # runtime validation does not dereference mutable or retired build inputs.
    manifest = json.loads(Path(environment["manifest_path"]).read_text(encoding="utf-8"))
    assert not Path(manifest["environment_seed"]["prefix"]).exists()
    assert not Path(manifest["ownership_policy"]["path"]).exists()
    assert not Path(manifest["integrity_normalization_policy"]["path"]).exists()
    assert not Path(manifest["conda_creation_tool"]["path"]).exists()


def test_schema4_manifest_rejects_schema_downgrade(tmp_path):
    environment = _environment(tmp_path, "harness")
    _rewrite_environment_manifest(
        environment,
        lambda manifest: manifest.__setitem__("release_id", RELEASE_ID_V12),
    )

    with pytest.raises(integrity.RuntimeIntegrityError, match="schema downgrade"):
        integrity.verify_environment_manifest_live(
            role="harness",
            prefix=Path(environment["prefix"]),
            manifest_path=Path(environment["manifest_path"]),
            expected_manifest_sha256=environment["manifest_sha256"],
            release_id=RELEASE_ID_V12,
        )


@pytest.mark.parametrize(
    ("transform", "match"),
    [
        (
            lambda manifest: manifest.__setitem__("conda_toolchain", []),
            "schema-4 environment provenance is malformed",
        ),
        (
            lambda manifest: manifest["environment_seed"].__setitem__(
                "capture_id", "not-a-digest"
            ),
            "environment capture ID",
        ),
        (
            lambda manifest: manifest["ownership_policy"].__setitem__(
                "path", "relative/policy.json"
            ),
            "ownership policy must be an absolute",
        ),
        (
            lambda manifest: manifest[
                "integrity_normalization_policy"
            ].__setitem__("path", "relative/integrity-policy.json"),
            "integrity-normalization policy must be an absolute",
        ),
        (
            lambda manifest: manifest["installed_files"].__setitem__(
                "file_count", manifest["installed_files"]["file_count"] + 1
            ),
            "installed-file provenance",
        ),
    ],
)
def test_schema4_manifest_rejects_malformed_provenance(tmp_path, transform, match):
    environment = _schema4_environment(tmp_path, "serving")
    _rewrite_environment_manifest(environment, transform)

    with pytest.raises(integrity.RuntimeIntegrityError, match=match):
        integrity.verify_environment_manifest_live(
            role="serving",
            prefix=Path(environment["prefix"]),
            manifest_path=Path(environment["manifest_path"]),
            expected_manifest_sha256=environment["manifest_sha256"],
            release_id=RELEASE_ID_V12,
        )


def test_schema4_manifest_content_identity_binds_valid_provenance(tmp_path):
    environment = _schema4_environment(tmp_path, "serving")
    _rewrite_environment_manifest(
        environment,
        lambda manifest: manifest["environment_seed"].__setitem__(
            "capture_id", "9" * 64
        ),
    )

    with pytest.raises(integrity.RuntimeIntegrityError, match="content identity"):
        integrity.verify_environment_manifest_live(
            role="serving",
            prefix=Path(environment["prefix"]),
            manifest_path=Path(environment["manifest_path"]),
            expected_manifest_sha256=environment["manifest_sha256"],
            release_id=RELEASE_ID_V12,
        )


def _inputs(tmp_path: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return state_dir, {
        role: _environment(tmp_path, role) for role in ("harness", "serving")
    }


def _ensure(
    state_dir: Path, environments: dict[str, dict[str, str]], *, force_full: bool = False
):
    pins = {
        role: {
            key: environments[role][key]
            for key in ("prefix", "manifest_path", "manifest_sha256")
        }
        for role in ("harness", "serving")
    }
    attestation = integrity.ensure_generation_attestation(
        state_dir=state_dir,
        generation=1,
        release_id=RELEASE_ID,
        release_bundle_id=RELEASE_BUNDLE_ID,
        immutable_pins_sha256=IMMUTABLE_PINS_SHA256,
        environment_pins=pins,
        force_full=force_full,
    )
    lease = integrity.refresh_generation_lease(
        state_dir=state_dir,
        attestation_path=Path(attestation["path"]),
        attestation_sha256=attestation["sha256"],
        generation=1,
        release_id=RELEASE_ID,
        immutable_pins_sha256=IMMUTABLE_PINS_SHA256,
        expected_environment_hashes={
            role: environments[role]["manifest_sha256"]
            for role in ("harness", "serving")
        },
        expected_prefixes={
            role: environments[role]["prefix"] for role in ("harness", "serving")
        },
        now=time.time(),
    )
    return {**attestation, "lease_path": lease["path"], "lease": lease["record"]}


def _runtime_environment(attestation: dict, environments: dict) -> dict[str, str]:
    return {
        "ASYS_RELEASE_ID": RELEASE_ID,
        "ASYS_MODEL_CONTRACT_SHA256": "c" * 64,
        "ASYS_FLEET_CONTRACT_SHA256": "d" * 64,
        "ASYS_HARNESS_ENVIRONMENT_SHA256": environments["harness"][
            "manifest_sha256"
        ],
        "ASYS_SERVING_ENVIRONMENT_SHA256": environments["serving"][
            "manifest_sha256"
        ],
        "ASYS_ROLLOUT_GENERATION": "1",
        "ASYS_IMMUTABLE_PINS_SHA256": IMMUTABLE_PINS_SHA256,
        "ASYS_RUNTIME_ATTESTATION": attestation["path"],
        "ASYS_RUNTIME_ATTESTATION_SHA256": attestation["sha256"],
        "ASYS_RUNTIME_INTEGRITY_LEASE": attestation["lease_path"],
        "ASYS_ARTIFACT_POLICY_SHA256": "e" * 64,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def test_cached_attestation_avoids_content_rehash(tmp_path, monkeypatch):
    state_dir, environments = _inputs(tmp_path)
    created = _ensure(state_dir, environments)

    def forbidden(**_kwargs):
        raise AssertionError("cached validation unexpectedly rehashed environment bytes")

    monkeypatch.setattr(integrity, "verify_environment_manifest_live", forbidden)
    cached = _ensure(state_dir, environments)
    assert cached["cached"] is True
    assert cached["sha256"] == created["sha256"]


def test_cross_node_lock_serializes_first_full_attestation(tmp_path, monkeypatch):
    state_dir, environments = _inputs(tmp_path)
    original = integrity.verify_environment_manifest_live
    calls = 0
    calls_lock = threading.Lock()

    def counted(**kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
        return original(**kwargs)

    monkeypatch.setattr(integrity, "verify_environment_manifest_live", counted)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: _ensure(state_dir, environments), range(2)))
    assert calls == 2  # harness + serving exactly once, not once per contender
    assert {result["sha256"] for result in results} == {results[0]["sha256"]}
    assert sorted(result["cached"] for result in results) == [False, True]


@pytest.mark.parametrize("role", ["harness", "serving"])
def test_nested_owner_write_invalidates_controller_worker_cache(
    tmp_path, role, monkeypatch
):
    state_dir, environments = _inputs(tmp_path)
    attestation = _ensure(state_dir, environments)
    nested = Path(environments[role]["nested_file"])
    nested.chmod(0o644)
    nested.write_text("X" * nested.stat().st_size, encoding="utf-8")
    nested.chmod(0o444)

    with pytest.raises(integrity.RuntimeIntegrityError, match=f"{role} runtime metadata drifted"):
        integrity.refresh_generation_lease(
            state_dir=state_dir,
            attestation_path=Path(attestation["path"]),
            attestation_sha256=attestation["sha256"],
            generation=1,
            release_id=RELEASE_ID,
            immutable_pins_sha256=IMMUTABLE_PINS_SHA256,
            expected_environment_hashes={
                name: environments[name]["manifest_sha256"]
                for name in ("harness", "serving")
            },
            expected_prefixes={
                name: environments[name]["prefix"]
                for name in ("harness", "serving")
            },
            force=True,
        )
    # No new lease was published.  Once the bounded old lease expires, every new cell
    # fails closed without independently walking 142k environment paths.
    expires = float(attestation["lease"]["expires_timestamp"])
    monkeypatch.setattr(integrity.time, "time", lambda: expires + 0.001)
    with pytest.raises(dispatch_sweeps.DispatcherError, match="runtime environment integrity"):
        dispatch_sweeps._verify_runtime_environment_attestation(
            _runtime_environment(attestation, environments)
        )


def test_inode_replacement_with_same_bytes_invalidates_cache(tmp_path):
    state_dir, environments = _inputs(tmp_path)
    attestation = _ensure(state_dir, environments)
    nested = Path(environments["serving"]["nested_file"])
    payload = nested.read_bytes()
    nested.parent.chmod(0o755)
    nested.unlink()
    nested.write_bytes(payload)
    nested.chmod(0o444)
    nested.parent.chmod(0o555)
    with pytest.raises(integrity.RuntimeIntegrityError, match="serving runtime metadata drifted"):
        integrity.refresh_generation_lease(
            state_dir=state_dir,
            attestation_path=Path(attestation["path"]),
            attestation_sha256=attestation["sha256"],
            generation=1,
            release_id=RELEASE_ID,
            immutable_pins_sha256=IMMUTABLE_PINS_SHA256,
            expected_environment_hashes={
                role: environments[role]["manifest_sha256"]
                for role in ("harness", "serving")
            },
            expected_prefixes={
                role: environments[role]["prefix"]
                for role in ("harness", "serving")
            },
            force=True,
        )


def test_worker_validation_never_walks_environment_tree(tmp_path, monkeypatch):
    state_dir, environments = _inputs(tmp_path)
    attestation = _ensure(state_dir, environments)

    def forbidden(_root):
        raise AssertionError("worker performed a recursive metadata scan")

    monkeypatch.setattr(integrity, "metadata_state", forbidden)
    dispatch_sweeps._verify_runtime_environment_attestation(
        _runtime_environment(attestation, environments)
    )


def test_every_coordinate_checks_small_fresh_lease_without_tree_walk(
    tmp_path, monkeypatch
):
    state_dir, environments = _inputs(tmp_path)
    attestation = _ensure(state_dir, environments)
    runtime = _runtime_environment(attestation, environments)
    for key, value in runtime.items():
        monkeypatch.setenv(key, value)

    def forbidden(_root):
        raise AssertionError("coordinate guard performed a recursive metadata scan")

    monkeypatch.setattr(integrity, "metadata_state", forbidden)
    run_one._runtime_integrity_coordinate_guard()
    run_one._runtime_integrity_coordinate_guard()

    expires = float(attestation["lease"]["expires_timestamp"])
    monkeypatch.setattr(integrity.time, "time", lambda: expires + 0.001)
    with pytest.raises(CoordinateAdmissionClosed, match="no new coordinate admitted"):
        run_one._runtime_integrity_coordinate_guard()


def test_attestation_byte_tamper_fails_closed_without_replacement(tmp_path):
    state_dir, environments = _inputs(tmp_path)
    attestation = _ensure(state_dir, environments)
    path = Path(attestation["path"])
    original = path.read_bytes()
    path.chmod(0o644)
    payload = json.loads(original)
    payload["release_bundle_id"] = "f" * 64
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)

    with pytest.raises(integrity.RuntimeIntegrityError, match="byte hash drifted"):
        integrity.verify_generation_attestation(
            path=path,
            expected_sha256=attestation["sha256"],
            generation=1,
            release_id=RELEASE_ID,
            immutable_pins_sha256=IMMUTABLE_PINS_SHA256,
            expected_environment_hashes={
                role: environments[role]["manifest_sha256"]
                for role in ("harness", "serving")
            },
        )
    assert path.read_bytes() != original


def test_lease_tamper_and_expiry_are_not_self_healed_by_workers(tmp_path):
    state_dir, environments = _inputs(tmp_path)
    attestation = _ensure(state_dir, environments)
    lease_path = Path(attestation["lease_path"])
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease_path.chmod(0o644)
    lease["expires_timestamp"] = float(lease["expires_timestamp"]) + 86_400
    # Deliberately retain the old lease_id: a worker may validate but never resign or
    # repair controller-owned freshness evidence.
    lease_path.write_text(json.dumps(lease, sort_keys=True) + "\n", encoding="utf-8")
    lease_path.chmod(0o444)

    with pytest.raises(integrity.RuntimeIntegrityError, match="lease identity is invalid"):
        integrity.verify_generation_lease(
            lease_path=lease_path,
            attestation_path=Path(attestation["path"]),
            attestation_sha256=attestation["sha256"],
            generation=1,
            release_id=RELEASE_ID,
            immutable_pins_sha256=IMMUTABLE_PINS_SHA256,
            expected_environment_hashes={
                role: environments[role]["manifest_sha256"]
                for role in ("harness", "serving")
            },
        )


def test_manifest_drift_invalidates_cached_generation_attestation(tmp_path):
    state_dir, environments = _inputs(tmp_path)
    attestation = _ensure(state_dir, environments)
    manifest = Path(environments["harness"]["manifest_path"])
    manifest.write_bytes(manifest.read_bytes() + b" \n")

    with pytest.raises(integrity.RuntimeIntegrityError, match="harness manifest drifted"):
        integrity.verify_generation_attestation(
            path=Path(attestation["path"]),
            expected_sha256=attestation["sha256"],
            generation=1,
            release_id=RELEASE_ID,
            immutable_pins_sha256=IMMUTABLE_PINS_SHA256,
            expected_environment_hashes={
                role: environments[role]["manifest_sha256"]
                for role in ("harness", "serving")
            },
            verify_metadata=False,
        )


def test_restart_reuses_exact_generation_but_never_crosses_generation(tmp_path):
    state_dir, environments = _inputs(tmp_path)
    first = _ensure(state_dir, environments)
    pins = {
        role: {
            key: environments[role][key]
            for key in ("prefix", "manifest_path", "manifest_sha256")
        }
        for role in ("harness", "serving")
    }
    restarted = integrity.ensure_generation_attestation(
        state_dir=state_dir,
        generation=1,
        release_id=RELEASE_ID,
        release_bundle_id=RELEASE_BUNDLE_ID,
        immutable_pins_sha256=IMMUTABLE_PINS_SHA256,
        environment_pins=pins,
    )
    second_generation = integrity.ensure_generation_attestation(
        state_dir=state_dir,
        generation=2,
        release_id=RELEASE_ID,
        release_bundle_id=RELEASE_BUNDLE_ID,
        immutable_pins_sha256=IMMUTABLE_PINS_SHA256,
        environment_pins=pins,
    )

    assert restarted["cached"] is True
    assert restarted["sha256"] == first["sha256"]
    assert second_generation["path"] != first["path"]
    assert second_generation["sha256"] != first["sha256"]
    with pytest.raises(integrity.RuntimeIntegrityError, match="identity is invalid"):
        integrity.verify_generation_attestation(
            path=Path(first["path"]),
            expected_sha256=first["sha256"],
            generation=2,
            release_id=RELEASE_ID,
            immutable_pins_sha256=IMMUTABLE_PINS_SHA256,
            expected_environment_hashes={
                role: environments[role]["manifest_sha256"]
                for role in ("harness", "serving")
            },
        )
