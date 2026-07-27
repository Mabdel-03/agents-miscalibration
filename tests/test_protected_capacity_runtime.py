from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest

from agents_scaling.serving import protected_capacity
from agents_scaling.serving.fleet_contract import EXPECTED_COUNTS
from scripts import run_schema5_throughput_qualification as qualification


COMMIT = "a" * 40
TAG_OBJECT = "b" * 40


def test_every_production_capacity_load_binds_tag_and_frozen_sources() -> None:
    """Prevent a future caller from silently authenticating on commit alone."""

    repo = Path(__file__).resolve().parents[1]
    required_sources = {
        "expected_source_tree_sha256",
        "expected_dispatcher_source_sha256",
        "expected_qualification_runner_source_sha256",
    }
    observed: list[str] = []
    for directory in (repo / "scripts", repo / "slurm"):
        for path in sorted(directory.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load_contract"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "protected_capacity"
                ):
                    continue
                relative = str(path.relative_to(repo))
                observed.append(f"{relative}:{node.lineno}")
                keywords = {
                    keyword.arg
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                assert "expected_release_git_commit" in keywords
                assert "expected_release_tag_object" in keywords
                missing_sources = required_sources - keywords
                if missing_sources:
                    # Control alone expands one closed helper that computes all three
                    # hashes from the immutable release worktree. No other **kwargs
                    # escape hatch is accepted.
                    assert relative == "slurm/schema5_control.py"
                    assert any(keyword.arg is None for keyword in node.keywords)
    assert observed


def _sealed_json(path: Path, value: object) -> tuple[Path, str]:
    raw = protected_capacity.canonical_bytes(value)
    if not path.exists():
        path.write_bytes(raw)
        path.chmod(0o444)
    assert path.read_bytes() == raw
    return path.resolve(), hashlib.sha256(raw).hexdigest()


def _topology(
    counts: dict[str, int],
    *,
    prefix: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile in sorted(counts):
        tp = int(protected_capacity.SERVING_PROFILES[profile].tp_size)
        for index in range(counts[profile]):
            rows.append(
                {
                    "shape_id": f"{prefix}-{profile.replace('.', '_')}-{index:02d}",
                    "serving_profile": profile,
                    "tasks": 1,
                    "cpus": tp * 8,
                    "memory_mib": tp * 120 * 1024,
                    "gpus": tp,
                    "time_limit_seconds": 86_400,
                }
            )
    return rows


def _marker(root: Path) -> dict:
    base_counts = dict(EXPECTED_COUNTS)
    additions = {
        "0.6B": 3,
        "1.7B": 3,
        "4B": 3,
        "8B": 2,
        "14B": 3,
        "32B": 4,
    }
    effective_counts = {
        profile: count + additions.get(profile, 0)
        for profile, count in base_counts.items()
    }
    delta_counts = {
        profile: effective_counts[profile] - base_counts[profile]
        for profile in base_counts
    }
    base_path, base_sha256 = _sealed_json(
        root / "base-fleet.json",
        {"kind": "base-fleet", "counts": base_counts},
    )
    effective_path, effective_sha256 = _sealed_json(
        root / "effective-fleet.json",
        {"kind": "effective-fleet", "counts": effective_counts},
    )
    certificate = qualification.build_preflight_capacity_certificate(
        capacity_generation=1,
        release_git_commit=COMMIT,
        source_tree_sha256="0" * 64,
        release_fleet_contract_sha256=base_sha256,
        base_fleet_contract_sha256=base_sha256,
        proposed_effective_fleet_contract_sha256=effective_sha256,
        additive_overlay_contract_sha256=effective_sha256,
        base_profile_replicas=base_counts,
        effective_profile_replicas=effective_counts,
        dispatcher_source_sha256="1" * 64,
        qualification_runner_source_sha256="2" * 64,
    )
    certificate_path, certificate_sha256 = _sealed_json(
        root / protected_capacity.STATIC_FEASIBILITY_FILENAME,
        certificate,
    )
    base_topology = _topology(base_counts, prefix="base")
    additive_topology = _topology(delta_counts, prefix="additive")
    effective_topology = [*base_topology, *additive_topology]
    warm_topology = [
        {
            "shape_id": f"warm-tp1-{index:02d}",
            "serving_profile": "warm-tp1",
            "tasks": 1,
            "cpus": 8,
            "memory_mib": 120 * 1024,
            "gpus": 1,
            "time_limit_seconds": 86_400,
        }
        for index in range(2)
    ] + [
        {
            "shape_id": "warm-tp2-00",
            "serving_profile": "warm-tp2",
            "tasks": 1,
            "cpus": 16,
            "memory_mib": 240 * 1024,
            "gpus": 2,
            "time_limit_seconds": 86_400,
        }
    ]
    servers = [
        {
            "partition": "gpu_protected",
            "qos": "gpu_science",
            "partition_preempt_mode": "OFF",
            "qos_preempt_mode": "OFF",
            "base_active_gpus": 24,
            "reserved_additive_gpus": 18,
            "effective_active_gpus": 42,
            "retained_warm_turnover_gpus": 4,
            "attested_total_gpus": 46,
            "partition_cpus": 4096,
            "partition_memory_mib": 32 * 1024 * 1024,
            "partition_gpus": 64,
            "partition_nodes": 8,
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
        "source_tree_sha256": "0" * 64,
        "dispatcher_source_sha256": "1" * 64,
        "qualification_runner_source_sha256": "2" * 64,
        "chain_namespace": protected_capacity.CHAIN_NAMESPACE,
        "capacity_generation": 1,
        "base_fleet_contract_path": str(base_path),
        "base_fleet_contract_sha256": base_sha256,
        "effective_fleet_contract_path": str(effective_path),
        "effective_fleet_contract_sha256": effective_sha256,
        "additive_overlay_contract_path": str(effective_path),
        "additive_overlay_contract_sha256": effective_sha256,
        "static_feasibility_certificate": {
            "path": str(certificate_path),
            "sha256": certificate_sha256,
            "certificate_id": certificate["certificate_id"],
        },
        "base_active_logical_replicas": 22,
        "base_active_gpus": 24,
        "base_active_topology": base_topology,
        "base_active_topology_sha256": protected_capacity._sha256_value(
            base_topology
        ),
        "additive_reserved_logical_replicas": 18,
        "additive_reserved_gpus": 18,
        "additive_reserved_tp1_replicas": 18,
        "additive_reserved_tp2_replicas": 0,
        "additive_reserved_topology": additive_topology,
        "additive_reserved_topology_sha256": protected_capacity._sha256_value(
            additive_topology
        ),
        "effective_active_logical_replicas": 40,
        "effective_active_gpus": 42,
        "effective_active_topology": effective_topology,
        "effective_active_topology_sha256": protected_capacity._sha256_value(
            effective_topology
        ),
        "retained_warm_turnover_job_elements": 3,
        "retained_warm_turnover_gpus": 4,
        "retained_warm_turnover_tp1_allocations": 2,
        "retained_warm_turnover_tp2_allocations": 1,
        "retained_warm_turnover_topology": warm_topology,
        "retained_warm_turnover_topology_sha256": (
            protected_capacity._sha256_value(warm_topology)
        ),
        "attested_total_gpus": 46,
        "job_element_accounting": {
            "cell_job_elements": 384,
            "active_server_job_elements": 40,
            "warm_turnover_job_elements": 3,
            "controller_monitor_other_held_job_elements": 21,
            "total_non_cell_reserve_job_elements": 64,
            "total_canary_job_elements": 448,
        },
        "active_gpus": 42,
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
        "scheduler_max_jobs": 427,
        "scheduler_max_submit_jobs": 448,
        "running_scientific_jobs": 427,
        "minimum_scientific_wall_seconds": 86_400,
        "scientific_qos_contracts": [
            {
                "qos": "client_science",
                "max_wall_seconds": 86_400,
                "max_jobs_per_user": 387,
                "max_submit_jobs_per_user": 448,
                "required_wall_seconds": 43_200,
                "required_running_jobs": 384,
                "required_submit_jobs": 405,
            },
            {
                "qos": "gpu_science",
                "max_wall_seconds": 86_400,
                "max_jobs_per_user": 43,
                "max_submit_jobs_per_user": 448,
                "required_wall_seconds": 86_400,
                "required_running_jobs": 43,
                "required_submit_jobs": 43,
            },
        ],
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
        "fleet_contract_sha256": effective_sha256,
        "active_fleet_topology_sha256": (
            protected_capacity._sha256_value(effective_topology)
        ),
        "scientific_server_placements": servers,
        "scientific_client_placements": clients,
    }
    payload["marker_id"] = hashlib.sha256(
        protected_capacity.canonical_bytes(payload)
    ).hexdigest()
    return payload


def _write_marker(path: Path) -> tuple[dict, str]:
    marker = _marker(path.parent)
    raw = protected_capacity.canonical_bytes(marker)
    path.write_bytes(raw)
    path.chmod(0o444)
    return marker, hashlib.sha256(raw).hexdigest()


def _fleet(
    marker: dict,
    *,
    first_partition: str | None = None,
    first_qos: str | None = None,
) -> SimpleNamespace:
    replicas = []
    for index, row in enumerate(marker["effective_active_topology"]):
        replicas.append(
            SimpleNamespace(
                replica_id=row["shape_id"],
                partition=(
                    first_partition
                    if index == 0 and first_partition is not None
                    else "gpu_protected"
                ),
                qos=(
                    first_qos
                    if index == 0 and first_qos is not None
                    else "gpu_science"
                ),
                gpus_per_replica=row["gpus"],
                time_limit="1-00:00:00",
            )
        )
    return SimpleNamespace(
        path=Path(marker["effective_fleet_contract_path"]),
        sha256=marker["effective_fleet_contract_sha256"],
        replicas=tuple(replicas),
    )


def _trusted_provenance(
    *,
    cell_job_ids: tuple[str, ...] = (),
    fleet_job_ids: tuple[str, ...] = (),
    states: dict[str, str] | None = None,
    timestamp: float | None = None,
) -> protected_capacity.TrustedScientificJobProvenance:
    timestamp = time.time() if timestamp is None else float(timestamp)
    scheduler_states = (
        dict(states)
        if states is not None
        else {
            job_id: "RUNNING"
            for job_id in (*cell_job_ids, *fleet_job_ids)
        }
    )
    return protected_capacity.build_trusted_scientific_job_provenance(
        scheduler_job_states=scheduler_states,
        scheduler_captured_timestamp=timestamp,
        trusted_cell_job_ids=cell_job_ids,
        trusted_fleet_job_ids=fleet_job_ids,
        dispatcher_ledger_updated_timestamp=(
            None if not cell_job_ids else timestamp
        ),
        exact_cell_quiescence=not cell_job_ids,
        dispatcher_provenance_id="3" * 64,
        fleet_provenance_id="4" * 64,
        fleet_contract_sha256="1" * 64,
        fleet_generation=1,
        scheduler_truth_id="5" * 64,
        now=timestamp,
    )


def test_load_contract_and_explicit_authorization(tmp_path: Path) -> None:
    marker, digest = _write_marker(tmp_path / protected_capacity.MARKER_FILENAME)
    contract = protected_capacity.load_contract(
        tmp_path / protected_capacity.MARKER_FILENAME,
        expected_release_git_commit=COMMIT,
        expected_release_tag_object=TAG_OBJECT,
        expected_source_tree_sha256="0" * 64,
        expected_dispatcher_source_sha256="1" * 64,
        expected_qualification_runner_source_sha256="2" * 64,
        expected_marker_id=marker["marker_id"],
        expected_sha256=digest,
    )
    fleet = _fleet(marker)
    protected_capacity.authorize_fleet(fleet, contract)
    placement = protected_capacity.authorize_client(
        contract,
        partition="cpu_protected",
        qos="client_science",
        required_slots=384,
        required_reserve_jobs=64,
    )
    assert placement.capacity["memory_mib"] == 384 * 4096


@pytest.mark.parametrize(
    ("expected_field", "expected_value"),
    (
        ("expected_release_tag_object", "c" * 40),
        ("expected_source_tree_sha256", "3" * 64),
        ("expected_dispatcher_source_sha256", "4" * 64),
        ("expected_qualification_runner_source_sha256", "5" * 64),
    ),
)
def test_load_contract_rejects_tag_or_frozen_source_drift(
    tmp_path: Path,
    expected_field: str,
    expected_value: str,
) -> None:
    marker, digest = _write_marker(
        tmp_path / protected_capacity.MARKER_FILENAME
    )
    expectations = {
        "expected_release_git_commit": COMMIT,
        "expected_release_tag_object": TAG_OBJECT,
        "expected_source_tree_sha256": "0" * 64,
        "expected_dispatcher_source_sha256": "1" * 64,
        "expected_qualification_runner_source_sha256": "2" * 64,
        "expected_marker_id": marker["marker_id"],
        "expected_sha256": digest,
    }
    expectations[expected_field] = expected_value
    with pytest.raises(
        protected_capacity.ProtectedCapacityError,
        match=(
            "marker identity/release/seal is invalid|"
            "static feasibility certificate identity/binding is invalid"
        ),
    ):
        protected_capacity.load_contract(
            tmp_path / protected_capacity.MARKER_FILENAME,
            **expectations,
        )


@pytest.mark.parametrize(
    "field",
    (
        "source_tree_sha256",
        "dispatcher_source_sha256",
        "qualification_runner_source_sha256",
    ),
)
def test_load_contract_rejects_marker_source_drift_from_static_certificate(
    tmp_path: Path,
    field: str,
) -> None:
    marker = _marker(tmp_path)
    marker[field] = "9" * 64
    marker.pop("marker_id")
    marker["marker_id"] = hashlib.sha256(
        protected_capacity.canonical_bytes(marker)
    ).hexdigest()
    marker_path = tmp_path / protected_capacity.MARKER_FILENAME
    marker_path.write_bytes(protected_capacity.canonical_bytes(marker))
    marker_path.chmod(0o444)

    with pytest.raises(
        protected_capacity.ProtectedCapacityError,
        match="static feasibility certificate identity/binding is invalid",
    ):
        protected_capacity.load_contract(
            marker_path,
            expected_release_git_commit=COMMIT,
        )


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
            _fleet(
                _marker(tmp_path),
                first_partition="ou_bcs_low",
                first_qos="",
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
    partition_cpus: int = 4096,
    partition_memory_mib: int = 32 * 1024 * 1024,
    partition_gpus: int = 64,
    partition_nodes: int = 8,
):
    occupancy_rows: tuple[tuple[str, str, str, str], ...] = ()

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
                f"PreemptMode={partition_mode} GraceTime=0 MaxTime=2-00:00:00 "
                "AllowQos=gpu_science "
                f"TotalCPUs={partition_cpus} TotalNodes={partition_nodes} "
                f"TRES=cpu={partition_cpus},mem={partition_memory_mib}M,"
                f"node={partition_nodes},gres/gpu={partition_gpus}\n"
            )
        elif argv[:5] == ["sacctmgr", "-nP", "show", "qos", "gpu_science"]:
            stdout = (
                f"gpu_science|{qos_mode}|43|448||1-00:00:00\n"
            )
        elif argv == [
            "sacctmgr",
            "-nP",
            "show",
            "assoc",
            "user=tester",
            "format=Cluster,Account,User,QOS,MaxJobs,MaxSubmitJobs",
        ]:
            stdout = (
                "cluster|account|tester|client_science,gpu_science|427|448\n"
            )
        elif argv[:5] == ["squeue", "-h", "-r", "-u", "tester"]:
            stdout = "".join(
                f"{job_id}|{state}|{account}|{qos}\n"
                for job_id, state, account, qos in occupancy_rows
            )
        elif argv[:4] == ["sacct", "-nP", "-X", "--array"]:
            stdout = "".join(
                f"{job_id}|{state}|{account}|{qos}\n"
                for job_id, state, account, qos in occupancy_rows
            )
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    return run


def _live_client_runner(
    *,
    qos_row: str = "client_science|OFF|387|448||1-00:00:00",
    association_row: str = (
        "cluster|account|tester|client_science,gpu_science|427|448"
    ),
    partition_cpus: int = 384,
    partition_memory_mib: int = 384 * 4096,
    occupancy_rows: tuple[tuple[str, str, str, str], ...] = (
        ("9001", "RUNNING", "account", "client_science"),
        ("9002", "RUNNING", "account", "client_science"),
        ("9003", "RUNNING", "account", "client_science"),
    ),
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
                "MaxSubmitJobsPerUser,MaxTRESPerUser,MaxWall"
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
        elif argv[:5] == ["squeue", "-h", "-r", "-u", "tester"]:
            output = "".join(
                f"{job_id}|{state}|{account}|{qos}\n"
                for job_id, state, account, qos in occupancy_rows
            )
        elif argv[:4] == ["sacct", "-nP", "-X", "--array"]:
            output = "".join(
                f"{job_id}|{state}|{account}|{qos}\n"
                for job_id, state, account, qos in occupancy_rows
            )
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
        trusted_scientific_job_provenance=_trusted_provenance(
            cell_job_ids=("9001", "9002", "9003"),
            timestamp=100.0,
        ),
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
                    "cluster|wrong-account|tester|"
                    "client_science,gpu_science|412|448"
                )
            ),
            "association drifted",
        ),
        (
            _live_client_runner(
                association_row=(
                    "cluster|account|wrong-user|"
                    "client_science,gpu_science|412|448"
                )
            ),
            "association drifted",
        ),
        (
            _live_client_runner(
                association_row=(
                    "cluster|account|tester|"
                    "client_science,gpu_science|412|447"
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
            trusted_scientific_job_provenance=_trusted_provenance(
                cell_job_ids=("9001", "9002", "9003"),
                timestamp=100.0,
            ),
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
        trusted_scientific_job_provenance=_trusted_provenance(
            cell_job_ids=("9001", "9002", "9003"),
            timestamp=100.0,
        ),
        runner=_live_client_runner(
            qos_row=(
                "client_science|OFF|387|448|cpu=384,mem=1536G|"
                "1-00:00:00"
            )
        ),
        captured_timestamp=100.0,
    )
    assert evidence["evidence_id"]


def test_pending_trusted_jobs_count_for_submit_but_not_maxjobs(
    tmp_path: Path,
) -> None:
    marker = _marker(tmp_path)
    marker["scheduler_max_jobs"] = 427
    marker["scientific_qos_contracts"][0]["max_jobs_per_user"] = 384
    marker.pop("marker_id")
    marker["marker_id"] = hashlib.sha256(
        protected_capacity.canonical_bytes(marker)
    ).hexdigest()
    marker_path = tmp_path / protected_capacity.MARKER_FILENAME
    marker_path.write_bytes(protected_capacity.canonical_bytes(marker))
    marker_path.chmod(0o444)
    contract = protected_capacity.load_contract(
        marker_path,
        expected_release_git_commit=COMMIT,
        expected_marker_id=marker["marker_id"],
    )
    pending = ("9001", "9002", "9003")
    evidence = protected_capacity.capture_live_client_capacity(
        contract,
        partition="cpu_protected",
        qos="client_science",
        required_time_limit_seconds=43_200,
        trusted_scientific_job_provenance=_trusted_provenance(
            cell_job_ids=pending,
            states={job_id: "PENDING" for job_id in pending},
            timestamp=100.0,
        ),
        runner=_live_client_runner(
            qos_row="client_science|OFF|384|448||1-00:00:00",
            association_row=(
                "cluster|account|tester|client_science,gpu_science|427|448"
            ),
            occupancy_rows=tuple(
                (job_id, "PENDING", "account", "client_science")
                for job_id in pending
            ),
        ),
        captured_timestamp=100.0,
    )
    summary = protected_capacity.validate_live_client_capacity_evidence(
        contract,
        evidence,
        partition="cpu_protected",
        qos="client_science",
        required_time_limit_seconds=43_200,
    )
    assert summary["trusted_scientific_live_job_ids"] == list(pending)
    assert summary["trusted_scientific_running_job_ids"] == []
    assert summary["unrelated_running_association_job_elements"] == 0
    assert summary["unrelated_submitted_association_job_elements"] == 0


@pytest.mark.parametrize(
    "occupancy_rows",
    (
        (
            ("9001", "RUNNING", "account", "gpu_science"),
            ("9002", "RUNNING", "account", "gpu_science"),
        ),
        (
            ("9001", "RUNNING", "account", "client_science"),
            ("9002", "RUNNING", "account", "client_science"),
        ),
    ),
    ids=("cell-on-server-qos", "fleet-on-client-qos"),
)
def test_self_consistent_provenance_cannot_swap_scientific_qos_roles(
    tmp_path: Path,
    occupancy_rows,
) -> None:
    marker, _digest = _write_marker(
        tmp_path / protected_capacity.MARKER_FILENAME
    )
    contract = protected_capacity.load_contract(
        tmp_path / protected_capacity.MARKER_FILENAME,
        expected_release_git_commit=COMMIT,
        expected_marker_id=marker["marker_id"],
    )
    provenance = _trusted_provenance(
        cell_job_ids=("9001",),
        fleet_job_ids=("9002",),
        timestamp=100.0,
    )
    with pytest.raises(
        protected_capacity.ProtectedCapacityError,
        match="role/QOS mismatch",
    ):
        protected_capacity.capture_live_client_capacity(
            contract,
            partition="cpu_protected",
            qos="client_science",
            required_time_limit_seconds=43_200,
            trusted_scientific_job_provenance=provenance,
            runner=_live_client_runner(
                occupancy_rows=occupancy_rows,
            ),
            captured_timestamp=100.0,
        )


def test_untrusted_pending_jobs_exhaust_submit_not_maxjobs(
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
    with pytest.raises(
        protected_capacity.ProtectedCapacityError,
        match="MaxSubmit",
    ):
        protected_capacity.capture_live_client_capacity(
            contract,
            partition="cpu_protected",
            qos="client_science",
            required_time_limit_seconds=43_200,
            trusted_scientific_job_provenance=_trusted_provenance(
                timestamp=100.0
            ),
            runner=_live_client_runner(
                occupancy_rows=(
                    ("9101", "PENDING", "account", "client_science"),
                )
            ),
            captured_timestamp=100.0,
        )


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
        trusted_scientific_job_provenance=_trusted_provenance(),
        runner=_live_runner(),
    )
    assert report["marker_id"] == contract.marker_id
    assert report["live_partition_inventories"] == [
        {
            "partition": "gpu_protected",
            "qos": "gpu_science",
            "cpus": 4096,
            "memory_mib": 32 * 1024 * 1024,
            "gpus": 64,
            "nodes": 8,
            "state": "UP",
        }
    ]
    with pytest.raises(
        protected_capacity.ProtectedCapacityError, match="QOS.*drifted"
    ):
        protected_capacity.verify_live_placements(
            contract,
            role="server",
            placements=[("gpu_protected", "gpu_science")],
            required_time_limits_seconds={"gpu_protected": 86_400},
            trusted_scientific_job_provenance=_trusted_provenance(),
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
            trusted_scientific_job_provenance=_trusted_provenance(),
            runner=_live_runner(partition_mode="REQUEUE"),
        )


@pytest.mark.parametrize(
    ("runner", "drift"),
    (
        (_live_runner(partition_nodes=7), "node inventory shrink"),
        (_live_runner(partition_gpus=63), "GPU TRES drift"),
    ),
)
def test_live_server_capacity_rejects_sealed_inventory_drift(
    tmp_path: Path,
    runner,
    drift: str,
) -> None:
    del drift
    _write_marker(tmp_path / protected_capacity.MARKER_FILENAME)
    contract = protected_capacity.load_contract(
        tmp_path / protected_capacity.MARKER_FILENAME,
        expected_release_git_commit=COMMIT,
    )
    with pytest.raises(
        protected_capacity.ProtectedCapacityError,
        match="42-active plus 4-warm",
    ):
        protected_capacity.verify_live_placements(
            contract,
            role="server",
            placements=[("gpu_protected", "gpu_science")],
            required_time_limits_seconds={"gpu_protected": 86_400},
            trusted_scientific_job_provenance=_trusted_provenance(),
            runner=runner,
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
    marker = _marker(tmp_path)
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
        trusted_scientific_job_provenance=_trusted_provenance(),
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
