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


RELEASE_ID = "sweep-recovery-schema5-v1.1"
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
