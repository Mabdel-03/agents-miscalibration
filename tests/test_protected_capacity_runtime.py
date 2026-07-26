from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from agents_scaling.serving import protected_capacity


COMMIT = "a" * 40
TAG_OBJECT = "b" * 40


def _marker() -> dict:
    servers = [
        {
            "partition": "gpu_protected",
            "qos": "gpu_science",
            "partition_preempt_mode": "OFF",
            "qos_preempt_mode": "OFF",
            "active_serving_gpus": 24,
            "warm_headroom_gpus": 4,
        }
    ]
    clients = [
        {
            "partition": "cpu_protected",
            "qos": "client_science",
            "partition_preempt_mode": "OFF",
            "qos_preempt_mode": "OFF",
            "slots": 384,
            "cpus": 384,
            "memory_mib": 384 * 4096,
            "reserve_jobs": 64,
            "submit_headroom": 448,
        }
    ]
    payload = {
        "schema_version": protected_capacity.SCHEMA_VERSION,
        "protocol": protected_capacity.PROTOCOL,
        "passed": True,
        "release_id": protected_capacity.RELEASE_ID,
        "release_tag": protected_capacity.RELEASE_TAG,
        "release_git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "chain_namespace": protected_capacity.CHAIN_NAMESPACE,
        "active_gpus": 24,
        "warm_headroom_gpus": 4,
        "cell_ceiling": 384,
        "reserve_jobs": 64,
        "submit_headroom": 448,
        "cpu": 384,
        "memory_mib": 384 * 4096,
        "preempt_type": "preempt/qos",
        "capacity_source": protected_capacity.CAPACITY_SOURCE,
        "scheduler_cluster": "cluster",
        "scheduler_account": "account",
        "scheduler_user": "tester",
        "scheduler_max_submit_jobs": 448,
        "partition_cpus": 384,
        "partition_memory_mib": 384 * 4096,
        "partition_gpus": 0,
        "scientific_server_preempt_mode": "OFF",
        "scientific_client_preempt_mode": "OFF",
        "squeue_complete": True,
        "sacct_complete": True,
        "scheduler_evidence_id": "c" * 64,
        "scheduler_evidence_sha256": "d" * 64,
        "canary_id": "e" * 64,
        "canary_evidence_sha256": "f" * 64,
        "fleet_contract_sha256": "1" * 64,
        "active_fleet_topology_sha256": "2" * 64,
        "scientific_server_placements": servers,
        "scientific_client_placements": clients,
    }
    payload["marker_id"] = hashlib.sha256(
        protected_capacity.canonical_bytes(payload)
    ).hexdigest()
    return payload


def _write_marker(path: Path) -> tuple[dict, str]:
    marker = _marker()
    raw = protected_capacity.canonical_bytes(marker)
    path.write_bytes(raw)
    path.chmod(0o444)
    return marker, hashlib.sha256(raw).hexdigest()


def test_load_contract_and_explicit_authorization(tmp_path: Path) -> None:
    marker, digest = _write_marker(tmp_path / protected_capacity.MARKER_FILENAME)
    contract = protected_capacity.load_contract(
        tmp_path / protected_capacity.MARKER_FILENAME,
        expected_release_git_commit=COMMIT,
        expected_release_tag_object=TAG_OBJECT,
        expected_marker_id=marker["marker_id"],
        expected_sha256=digest,
    )
    fleet = SimpleNamespace(
        replicas=(
            SimpleNamespace(
                replica_id="r0",
                partition="gpu_protected",
                qos="gpu_science",
                gpus_per_replica=2,
            ),
        )
    )
    protected_capacity.authorize_fleet(fleet, contract)
    placement = protected_capacity.authorize_client(
        contract,
        partition="cpu_protected",
        qos="client_science",
        required_slots=384,
        required_reserve_jobs=64,
    )
    assert placement.capacity["memory_mib"] == 384 * 4096


def test_missing_qos_and_uncovered_placement_fail_closed(tmp_path: Path) -> None:
    _write_marker(tmp_path / protected_capacity.MARKER_FILENAME)
    contract = protected_capacity.load_contract(
        tmp_path / protected_capacity.MARKER_FILENAME,
        expected_release_git_commit=COMMIT,
    )
    with pytest.raises(
        protected_capacity.ProtectedCapacityError, match="no explicit partition/QOS"
    ):
        protected_capacity.authorize_fleet(
            SimpleNamespace(
                replicas=(
                    SimpleNamespace(
                        replica_id="r0",
                        partition="ou_bcs_low",
                        qos=None,
                        gpus_per_replica=1,
                    ),
                )
            ),
            contract,
        )
    with pytest.raises(
        protected_capacity.ProtectedCapacityError, match="not explicitly authorized"
    ):
        protected_capacity.authorize_client(
            contract,
            partition="mit_normal",
            qos="mit_normal",
            required_slots=1,
            required_reserve_jobs=0,
        )


def _live_runner(
    *,
    qos_mode: str = "OFF",
    partition_mode: str = "OFF",
    preempt_type: str = "preempt/qos",
):
    def run(argv, *, timeout):
        del timeout
        if argv == ["scontrol", "show", "config"]:
            stdout = (
                f"PreemptType = {preempt_type}\n"
                "PreemptMode = REQUEUE\n"
                "KillWait = 30 sec\n"
            )
        elif argv[:4] == ["scontrol", "show", "partition", "gpu_protected"]:
            stdout = (
                "PartitionName=gpu_protected State=UP "
                f"PreemptMode={partition_mode} GraceTime=0 MaxTime=2-00:00:00\n"
            )
        elif argv[:5] == ["sacctmgr", "-nP", "show", "qos", "gpu_science"]:
            stdout = f"gpu_science|{qos_mode}\n"
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    return run


def _live_client_runner(
    *,
    qos_row: str = "client_science|OFF|||",
    association_row: str = (
        "cluster|account|tester|client_science|384|448"
    ),
    partition_cpus: int = 384,
    partition_memory_mib: int = 384 * 4096,
):
    def run(argv, *, timeout):
        assert timeout == 30.0
        if argv == ["scontrol", "show", "config"]:
            output = (
                "PreemptType = preempt/qos\n"
                "PreemptMode = REQUEUE\n"
                "KillWait = 30 sec\n"
            )
        elif argv == [
            "scontrol",
            "show",
            "partition",
            "cpu_protected",
            "-o",
        ]:
            output = (
                "PartitionName=cpu_protected State=UP PreemptMode=OFF "
                "GraceTime=0 MaxTime=12:00:00 "
                "AllowQos=client_science "
                f"TRES=cpu={partition_cpus},mem={partition_memory_mib}M,"
                "node=8,billing=384\n"
            )
        elif argv == [
            "sacctmgr",
            "-nP",
            "show",
            "qos",
            "client_science",
            (
                "format=Name,PreemptMode,MaxJobsPerUser,"
                "MaxSubmitJobsPerUser,MaxTRESPerUser"
            ),
        ]:
            output = qos_row + "\n"
        elif argv == [
            "sacctmgr",
            "-nP",
            "show",
            "assoc",
            "user=tester",
            "format=Cluster,Account,User,QOS,MaxJobs,MaxSubmitJobs",
        ]:
            output = association_row + "\n"
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, output, "")

    return run


def test_inherited_blank_qos_uses_sealed_canary_and_exact_association(
    tmp_path: Path,
) -> None:
    marker, digest = _write_marker(
        tmp_path / protected_capacity.MARKER_FILENAME
    )
    contract = protected_capacity.load_contract(
        tmp_path / protected_capacity.MARKER_FILENAME,
        expected_release_git_commit=COMMIT,
        expected_marker_id=marker["marker_id"],
        expected_sha256=digest,
    )
    evidence = protected_capacity.capture_live_client_capacity(
        contract,
        partition="cpu_protected",
        qos="client_science",
        required_time_limit_seconds=43_200,
        runner=_live_client_runner(),
        captured_timestamp=100.0,
    )
    summary = protected_capacity.validate_live_client_capacity_evidence(
        contract,
        evidence,
        partition="cpu_protected",
        qos="client_science",
        required_time_limit_seconds=43_200,
    )
    assert summary["capacity_source"] == protected_capacity.CAPACITY_SOURCE
    assert summary["cpu_limit"] == 384
    assert summary["scheduler_max_submit_jobs"] == 448


@pytest.mark.parametrize(
    ("runner", "message"),
    (
        (
            _live_client_runner(
                association_row=(
                    "cluster|wrong-account|tester|client_science|384|448"
                )
            ),
            "association drifted",
        ),
        (
            _live_client_runner(
                association_row=(
                    "cluster|account|wrong-user|client_science|384|448"
                )
            ),
            "association drifted",
        ),
        (
            _live_client_runner(
                association_row=(
                    "cluster|account|tester|client_science|384|447"
                )
            ),
            "association drifted",
        ),
        (
            _live_client_runner(partition_cpus=383),
            "partition inventory differs",
        ),
        (
            _live_client_runner(partition_memory_mib=384 * 4096 - 1),
            "partition inventory differs",
        ),
    ),
)
def test_live_client_capacity_rejects_association_or_inventory_drift(
    tmp_path: Path,
    runner,
    message: str,
) -> None:
    marker, _digest = _write_marker(
        tmp_path / protected_capacity.MARKER_FILENAME
    )
    contract = protected_capacity.load_contract(
        tmp_path / protected_capacity.MARKER_FILENAME,
        expected_release_git_commit=COMMIT,
        expected_marker_id=marker["marker_id"],
    )
    with pytest.raises(protected_capacity.ProtectedCapacityError, match=message):
        protected_capacity.capture_live_client_capacity(
            contract,
            partition="cpu_protected",
            qos="client_science",
            required_time_limit_seconds=43_200,
            runner=runner,
            captured_timestamp=100.0,
        )


def test_live_client_capacity_accepts_safe_explicit_qos_limits(
    tmp_path: Path,
) -> None:
    marker, _digest = _write_marker(
        tmp_path / protected_capacity.MARKER_FILENAME
    )
    contract = protected_capacity.load_contract(
        tmp_path / protected_capacity.MARKER_FILENAME,
        expected_release_git_commit=COMMIT,
        expected_marker_id=marker["marker_id"],
    )
    evidence = protected_capacity.capture_live_client_capacity(
        contract,
        partition="cpu_protected",
        qos="client_science",
        required_time_limit_seconds=43_200,
        runner=_live_client_runner(
            qos_row=(
                "client_science|OFF|384|448|cpu=384,mem=1536G"
            )
        ),
        captured_timestamp=100.0,
    )
    assert evidence["evidence_id"]


def test_live_partition_and_qos_are_rechecked(tmp_path: Path) -> None:
    _write_marker(tmp_path / protected_capacity.MARKER_FILENAME)
    contract = protected_capacity.load_contract(
        tmp_path / protected_capacity.MARKER_FILENAME,
        expected_release_git_commit=COMMIT,
    )
    report = protected_capacity.verify_live_placements(
        contract,
        role="server",
        placements=[("gpu_protected", "gpu_science")],
        required_time_limits_seconds={"gpu_protected": 86_400},
        runner=_live_runner(),
    )
    assert report["marker_id"] == contract.marker_id
    with pytest.raises(
        protected_capacity.ProtectedCapacityError, match="QOS.*drifted"
    ):
        protected_capacity.verify_live_placements(
            contract,
            role="server",
            placements=[("gpu_protected", "gpu_science")],
            required_time_limits_seconds={"gpu_protected": 86_400},
            runner=_live_runner(qos_mode="REQUEUE"),
        )
    with pytest.raises(
        protected_capacity.ProtectedCapacityError, match="partition drifted"
    ):
        protected_capacity.verify_live_placements(
            contract,
            role="server",
            placements=[("gpu_protected", "gpu_science")],
            required_time_limits_seconds={"gpu_protected": 86_400},
            runner=_live_runner(partition_mode="REQUEUE"),
        )


def test_writable_or_replaced_marker_is_not_authority(tmp_path: Path) -> None:
    marker_path = tmp_path / protected_capacity.MARKER_FILENAME
    marker, _digest = _write_marker(marker_path)
    marker_path.chmod(0o644)
    with pytest.raises(
        protected_capacity.ProtectedCapacityError, match="read-only"
    ):
        protected_capacity.load_contract(
            marker_path, expected_release_git_commit=COMMIT
        )
    marker_path.chmod(0o444)
    marker["cell_ceiling"] = 383
    marker_path.chmod(0o644)
    marker_path.write_bytes(protected_capacity.canonical_bytes(marker))
    marker_path.chmod(0o444)
    with pytest.raises(
        protected_capacity.ProtectedCapacityError, match="identity/release/seal"
    ):
        protected_capacity.load_contract(
            marker_path, expected_release_git_commit=COMMIT
        )


def test_inherited_qos_is_safe_only_for_partition_priority(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / protected_capacity.MARKER_FILENAME
    marker = _marker()
    marker["preempt_type"] = "preempt/partition_prio"
    marker["scientific_server_placements"][0]["qos_preempt_mode"] = "cluster"
    marker["scientific_client_placements"][0]["qos_preempt_mode"] = "cluster"
    marker.pop("marker_id")
    marker["marker_id"] = hashlib.sha256(
        protected_capacity.canonical_bytes(marker)
    ).hexdigest()
    marker_path.write_bytes(protected_capacity.canonical_bytes(marker))
    marker_path.chmod(0o444)
    contract = protected_capacity.load_contract(
        marker_path,
        expected_release_git_commit=COMMIT,
    )
    protected_capacity.verify_live_placements(
        contract,
        role="server",
        placements=[("gpu_protected", "gpu_science")],
        required_time_limits_seconds={"gpu_protected": 86_400},
        runner=_live_runner(
            qos_mode="cluster",
            preempt_type="preempt/partition_prio",
        ),
    )

    marker_path.chmod(0o644)
    marker["preempt_type"] = "preempt/qos"
    marker.pop("marker_id")
    marker["marker_id"] = hashlib.sha256(
        protected_capacity.canonical_bytes(marker)
    ).hexdigest()
    marker_path.write_bytes(protected_capacity.canonical_bytes(marker))
    marker_path.chmod(0o444)
    with pytest.raises(
        protected_capacity.ProtectedCapacityError,
        match="effectively nonpreemptible",
    ):
        protected_capacity.load_contract(
            marker_path,
            expected_release_git_commit=COMMIT,
        )
