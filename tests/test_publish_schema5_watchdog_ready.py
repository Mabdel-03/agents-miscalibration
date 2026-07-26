from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import publish_schema5_watchdog_ready as publisher
from scripts import render_schema5_recovery_chain_v12 as renderer


COMMIT = "a" * 40
TAG_OBJECT = "b" * 40


def _self_hash(value: dict, field: str) -> str:
    return publisher._self_hash(value, field)


def _write(path: Path, value: dict) -> Path:
    path.write_bytes(publisher._canonical(value))
    path.chmod(0o444)
    return path


def _release() -> dict:
    return publisher._release_fields(
        git_commit=COMMIT, tag_object=TAG_OBJECT
    )


def _evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    identity = {
        "deployment_id": "c" * 64,
        "watchdog_code_sha256": "d" * 64,
        "immutable_release_sha256": "e" * 64,
        "control_sha256": "f" * 64,
    }
    deployment = {
        "schema_version": 1,
        "protocol": publisher.DEPLOYMENT_EVIDENCE_PROTOCOL,
        "passed": True,
        **_release(),
        **identity,
        "runtime_inventory_sha256": "2" * 64,
        "runtime_file_count": 2,
        "runtime_total_bytes": 8192,
        "vm_python_path": "/opt/agents-scaling-watchdog/python",
        "vm_python_sha256": "3" * 64,
        "liveness_email": "mabdel03@mit.edu",
        "forced_command_only": True,
        "timer_seconds": 300,
        "isolated_runtime_probe_passed": True,
        "systemd_service_loaded": True,
        "systemd_service_result": "success",
        "systemd_service_exec_main_status": 0,
        "successful_service_heartbeat_id": "4" * 64,
        "successful_service_heartbeat_sha256": "5" * 64,
        "systemd_timer_active": True,
        "systemd_timer_enabled": True,
    }
    deployment["evidence_id"] = _self_hash(deployment, "evidence_id")
    drill = {
        "schema_version": 1,
        "protocol": publisher.DRILL_EVIDENCE_PROTOCOL,
        "passed": True,
        **_release(),
        **identity,
        "scheduler_observations": [1.0, 61.0],
        "namespace_cancellation_recovery_seconds": 600.0,
        "duplicate_jobs": 0,
        "duplicate_admission_intents": 0,
        "fairness_mutations": 0,
    }
    drill["evidence_id"] = _self_hash(drill, "evidence_id")
    liveness = {
        "schema_version": 1,
        "protocol": publisher.LIVENESS_EVIDENCE_PROTOCOL,
        "passed": True,
        **_release(),
        **identity,
        "scheduler_observations": [100.0, 160.0],
        "heartbeat_sha256": "1" * 64,
        "liveness_email": "mabdel03@mit.edu",
        "liveness_email_ack": True,
    }
    liveness["evidence_id"] = _self_hash(liveness, "evidence_id")
    return (
        _write(tmp_path / "deployment.json", deployment),
        _write(tmp_path / "drill.json", drill),
        _write(tmp_path / "liveness.json", liveness),
    )


def test_publish_watchdog_ready_is_marker_last_and_idempotent(tmp_path):
    deployment, drill, liveness = _evidence(tmp_path)
    recovery = tmp_path / "recovery"
    dry = publisher.publish(
        recovery_root=recovery,
        deployment_evidence=deployment,
        drill_evidence=drill,
        liveness_evidence=liveness,
        git_commit=COMMIT,
        tag_object=TAG_OBJECT,
        apply=False,
    )
    assert dry["apply"] is False
    assert not recovery.exists()
    result = publisher.publish(
        recovery_root=recovery,
        deployment_evidence=deployment,
        drill_evidence=drill,
        liveness_evidence=liveness,
        git_commit=COMMIT,
        tag_object=TAG_OBJECT,
        apply=True,
    )
    assert result["passed"] is True
    drill_marker = recovery / publisher.DRILL_MARKER
    ready_marker = recovery / publisher.READY_MARKER
    assert drill_marker.stat().st_mode & 0o222 == 0
    assert ready_marker.stat().st_mode & 0o222 == 0
    ready = json.loads(ready_marker.read_text(encoding="utf-8"))
    sealed_drill = json.loads(drill_marker.read_text(encoding="utf-8"))
    assert sealed_drill["duplicate_admission_intents"] == 0
    assert ready["duplicate_admission_intents"] == 0
    assert ready["external_watchdog_drill"]["marker_sha256"] == publisher._sha256(
        drill_marker
    )
    repeated = publisher.publish(
        recovery_root=recovery,
        deployment_evidence=deployment,
        drill_evidence=drill,
        liveness_evidence=liveness,
        git_commit=COMMIT,
        tag_object=TAG_OBJECT,
        apply=True,
    )
    assert repeated["marker_id"] == result["marker_id"]
    paths = SimpleNamespace(
        external_watchdog_drill_marker=drill_marker,
        watchdog_ready_marker=ready_marker,
    )
    verified_drill, drill_raw = (
        renderer._validate_external_watchdog_drill_marker(
            paths,
            git_identity={"git_commit": COMMIT, "tag_object": TAG_OBJECT},
        )
    )
    verified_ready, _ = renderer._validate_watchdog_ready_marker(
        paths,
        git_identity={"git_commit": COMMIT, "tag_object": TAG_OBJECT},
        drill_marker=verified_drill,
        drill_raw=drill_raw,
    )
    assert verified_ready["marker_id"] == result["marker_id"]


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("deployment", "forced_command_only", False),
        ("deployment", "timer_seconds", 301),
        ("drill", "namespace_cancellation_recovery_seconds", 901),
        ("drill", "duplicate_admission_intents", 1),
        ("drill", "fairness_mutations", 1),
        ("liveness", "scheduler_observations", [1.0, 59.0]),
        ("liveness", "liveness_email_ack", False),
    ],
)
def test_watchdog_publisher_rejects_incomplete_evidence(
    tmp_path, target, field, value
):
    deployment, drill, liveness = _evidence(tmp_path)
    paths = {
        "deployment": deployment,
        "drill": drill,
        "liveness": liveness,
    }
    path = paths[target]
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.chmod(0o644)
    payload[field] = value
    payload["evidence_id"] = _self_hash(payload, "evidence_id")
    _write(path, payload)
    with pytest.raises(publisher.WatchdogEvidenceError):
        publisher.publish(
            recovery_root=tmp_path / "recovery",
            deployment_evidence=deployment,
            drill_evidence=drill,
            liveness_evidence=liveness,
            git_commit=COMMIT,
            tag_object=TAG_OBJECT,
            apply=True,
        )


def test_watchdog_publisher_rejects_mutable_or_noncanonical_evidence(tmp_path):
    deployment, drill, liveness = _evidence(tmp_path)
    deployment.chmod(0o644)
    with pytest.raises(publisher.WatchdogEvidenceError, match="sealed"):
        publisher.publish(
            recovery_root=tmp_path / "recovery",
            deployment_evidence=deployment,
            drill_evidence=drill,
            liveness_evidence=liveness,
            git_commit=COMMIT,
            tag_object=TAG_OBJECT,
            apply=True,
        )


def test_watchdog_publisher_recovers_linked_temp_and_rejects_symlink_parent(
    tmp_path,
):
    target = tmp_path / "WATCHDOG_READY.json"
    payload = {"schema_version": 1, "passed": True}
    encoded = publisher._canonical(payload)
    interrupted = tmp_path / ".WATCHDOG_READY.json.crashed.publishing"
    interrupted.write_bytes(encoded)
    interrupted.chmod(0o444)
    __import__("os").link(interrupted, target)
    assert target.stat().st_nlink == 2
    publisher._publish_once(target, payload)
    assert target.stat().st_nlink == 1
    assert not interrupted.exists()

    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(publisher.WatchdogEvidenceError, match="symlink"):
        publisher._publish_once(alias / "marker.json", payload)
