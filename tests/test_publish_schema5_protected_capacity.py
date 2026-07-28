"""Fail-closed tests for the offline protected-capacity attestor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import subprocess

import pytest

from scripts import publish_schema5_protected_capacity as capacity
from scripts import run_schema5_throughput_qualification as qualification
COMMIT = "1" * 40
TAG_OBJECT = "2" * 40
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FLEET_RAW = (
    REPOSITORY_ROOT / "configs" / "schema5_fleet.v1.json"
).read_text(encoding="utf-8")
if not FLEET_RAW.endswith("\n"):
    FLEET_RAW += "\n"
FLEET_SHA256 = hashlib.sha256(FLEET_RAW.encode("utf-8")).hexdigest()


def _fleet_topology(
    fleet: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    profiles = fleet["profiles"]
    assert isinstance(profiles, list)
    for profile in profiles:
        assert isinstance(profile, dict)
        for replica in profile["replicas"]:
            memory = replica["memory"]
            assert memory.endswith("G")
            rows.append(
                {
                    "shape_id": replica["replica_id"],
                    "serving_profile": profile["serving_profile"],
                    "tasks": 1,
                    "cpus": replica["cpus_per_task"],
                    "memory_mib": int(memory[:-1]) * 1024,
                    "gpus": profile["tensor_parallel_size"],
                    "time_limit_seconds": 86_400,
                }
            )
    return rows


BASE_FLEET = json.loads(FLEET_RAW)
FLEET_TOPOLOGY = _fleet_topology(BASE_FLEET)
FLEET_TOPOLOGY_SHA256 = hashlib.sha256(
    capacity.canonical_bytes(FLEET_TOPOLOGY)
).hexdigest()
BUILDER_SOURCE = "sealed builder source\n"
PUBLISHER_SOURCE = "sealed publisher source\n"
CAPACITY_GENERATION = 1
SOURCE_TREE_SHA256 = "4" * 64
DISPATCHER_SOURCE_SHA256 = "5" * 64
QUALIFICATION_RUNNER_SOURCE_SHA256 = "6" * 64


def _source_trust_kwargs() -> dict[str, str]:
    return {
        "expected_source_tree_sha256": SOURCE_TREE_SHA256,
        "expected_dispatcher_source_sha256": DISPATCHER_SOURCE_SHA256,
        "expected_qualification_runner_source_sha256": (
            QUALIFICATION_RUNNER_SOURCE_SHA256
        ),
    }


def _write_authority_file(path: Path, raw: bytes) -> Path:
    if not path.exists():
        path.write_bytes(raw)
        path.chmod(0o444)
    assert path.read_bytes() == raw
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    return path


def _capacity_authority(root: Path) -> dict[str, object]:
    """Create the real solver-certified zero-delta baseline authority."""

    base_path = _write_authority_file(
        root / "schema5_fleet.base.json",
        FLEET_RAW.encode("utf-8"),
    )
    effective = json.loads(FLEET_RAW)
    effective_raw = FLEET_RAW.encode("utf-8")
    effective_path = _write_authority_file(
        root / "schema5_fleet.capacity-v1.json",
        effective_raw,
    )
    effective_sha256 = hashlib.sha256(effective_raw).hexdigest()
    base_counts = {
        profile["serving_profile"]: len(profile["replicas"])
        for profile in BASE_FLEET["profiles"]
    }
    effective_counts = {
        profile["serving_profile"]: len(profile["replicas"])
        for profile in effective["profiles"]
    }
    certificate = qualification.build_preflight_capacity_certificate(
        capacity_generation=CAPACITY_GENERATION,
        release_git_commit=COMMIT,
        source_tree_sha256=SOURCE_TREE_SHA256,
        release_fleet_contract_sha256=FLEET_SHA256,
        base_fleet_contract_sha256=FLEET_SHA256,
        proposed_effective_fleet_contract_sha256=effective_sha256,
        additive_overlay_contract_sha256=effective_sha256,
        base_profile_replicas=base_counts,
        effective_profile_replicas=effective_counts,
        dispatcher_source_sha256=DISPATCHER_SOURCE_SHA256,
        qualification_runner_source_sha256=(
            QUALIFICATION_RUNNER_SOURCE_SHA256
        ),
    )
    certificate_raw = capacity.canonical_bytes(certificate)
    certificate_path = _write_authority_file(
        root / capacity.runtime_capacity.STATIC_FEASIBILITY_FILENAME,
        certificate_raw,
    )
    return {
        "base_path": base_path,
        "base_sha256": FLEET_SHA256,
        "base_raw": FLEET_RAW,
        "effective": effective,
        "effective_path": effective_path,
        "effective_sha256": effective_sha256,
        "effective_raw": effective_raw.decode("utf-8"),
        "certificate": certificate,
        "certificate_path": certificate_path,
        "certificate_raw": certificate_raw.decode("utf-8"),
        "certificate_sha256": hashlib.sha256(certificate_raw).hexdigest(),
    }


def _empty_occupancy_preflight() -> dict[str, object]:
    def observation(observed_at: float) -> dict[str, object]:
        identity: dict[str, object] = {
            "observed_at": observed_at,
            "job_elements": 0,
            "jobs": [],
            "squeue_sha256": "1" * 64,
            "sacct_sha256": "2" * 64,
        }
        identity["observation_id"] = hashlib.sha256(
            capacity.canonical_bytes(identity)
        ).hexdigest()
        return identity

    identity = {
        "protocol": (
            "schema5-v1.2-r13-protected-capacity-occupancy-preflight-v3"
        ),
        "plan_id": "3" * 64,
        "observation_interval_seconds": 60.0,
        "scheduler_account": "account",
        "scientific_qos": "client_science",
        "association_max_jobs": 409,
        "qos_max_jobs": None,
        "effective_max_jobs": 409,
        "association_max_submit_jobs": 448,
        "qos_max_submit_jobs": 448,
        "effective_max_submit_jobs": 448,
        "existing_job_elements": 0,
        "existing_association_job_elements": 0,
        "existing_qos_job_elements": 0,
        "existing_association_running_job_elements": 0,
        "existing_qos_running_job_elements": 0,
        "required_new_running_job_elements": 409,
        "required_new_job_elements": 448,
        "first_observation": observation(900.0),
        "second_observation": observation(960.0),
    }
    identity["preflight_id"] = hashlib.sha256(
        capacity.canonical_bytes(identity)
    ).hexdigest()
    return identity


def _source(
    command: str,
    rows: list[str],
    *,
    kind: str | None = None,
) -> dict[str, object]:
    raw = "".join(f"{row}\n" for row in rows)
    return {
        "complete": True,
        "argv": [command, f"--capture-kind={kind or command}"],
        "raw_output": raw,
        "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "record_count": len(rows),
    }


def _replace_source_rows(
    scheduler: dict[str, object],
    field: str,
    rows: list[str],
) -> None:
    source = scheduler[field]
    assert isinstance(source, dict)
    raw = "".join(f"{row}\n" for row in rows)
    source["raw_output"] = raw
    source["raw_output_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    source["record_count"] = len(rows)


def _binding() -> dict[str, object]:
    return {
        "release_id": capacity.RELEASE_ID,
        "release_tag": capacity.RELEASE_TAG,
        "release_git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "chain_namespace": capacity.CHAIN_NAMESPACE,
    }


def _scheduler_identity(root: Path) -> dict[str, object]:
    authority = _capacity_authority(root)
    effective = authority["effective"]
    assert isinstance(effective, dict)
    effective_topology = _fleet_topology(effective)
    active_rows = [
        (
            f"{101 + index}|server_active|RUNNING|gpu_protected|gpu_science|"
            f"1|{row['cpus']}|{row['memory_mib']}|{row['gpus']}|0|"
            f"86400|{row['shape_id']}|{'a' * 64}"
        )
        for index, row in enumerate(effective_topology)
    ]
    warm_rows = [
        (
            f"201|server_warm|RUNNING|gpu_protected|gpu_science|"
            f"1|8|122880|1|0|86400|warm-tp1-00|{'b' * 64}"
        ),
        (
            f"202|server_warm|RUNNING|gpu_protected|gpu_science|"
            f"1|8|122880|1|0|86400|warm-tp1-01|{'c' * 64}"
        ),
        (
            f"203|server_warm|RUNNING|gpu_protected|gpu_science|"
            f"1|16|245760|2|0|86400|warm-tp2-00|{'d' * 64}"
        ),
    ]
    client_rows = [
        (
            f"301|client|RUNNING|cpu_protected|client_science|"
            f"384|384|1572864|0|0|43200|client-000|{'e' * 64}"
        ),
        (
            f"302|reserve|PENDING|cpu_protected|client_science|"
            f"39|39|39936|0|0|43200|reserve-000|{'f' * 64}"
        ),
    ]
    servers = [
        {
            "partition": "gpu_protected",
            "qos": "gpu_science",
            "partition_preempt_mode": "OFF",
            "qos_preempt_mode": "OFF",
            "base_active_gpus": 24,
            "reserved_additive_gpus": 0,
            "effective_active_gpus": 24,
            "retained_warm_turnover_gpus": 4,
            "attested_total_gpus": 28,
            "partition_cpus": 4096,
            "partition_memory_mib": 33_554_432,
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
            "memory_mib": 1_572_864,
            "reserve_jobs": 64,
            "submit_headroom": 448,
        }
    ]
    source_fields = {
        "scheduler_configuration": _source(
            "scontrol",
            ["PreemptType|preempt/qos"],
            kind="configuration",
        ),
        "partition_configuration": _source(
            "scontrol",
            [
                "cpu_protected|OFF|UP|43200|384|1572864|0|8",
                "gpu_protected|OFF|UP|172800|4096|33554432|64|8",
            ],
            kind="partitions",
        ),
        "qos_configuration": _source(
            "sacctmgr",
            [
                "client_science|OFF|-|448|86400",
                "gpu_science|OFF|-|448|86400",
            ],
            kind="qos",
        ),
        "association_configuration": _source(
            "sacctmgr",
            [
                "cluster|account|tester|client_science,gpu_science|409|448"
            ],
            kind="associations",
        ),
        "base_fleet_contract": _source(
            "base-fleet-contract",
            str(authority["base_raw"]).rstrip("\n").splitlines(),
        ),
        "effective_fleet_contract": _source(
            "effective-fleet-contract",
            str(authority["effective_raw"]).rstrip("\n").splitlines(),
        ),
        "additive_overlay_contract": _source(
            "additive-overlay-contract",
            str(authority["effective_raw"]).rstrip("\n").splitlines(),
        ),
        "static_feasibility_certificate_source": _source(
            "static-feasibility-certificate",
            str(authority["certificate_raw"]).rstrip("\n").splitlines(),
        ),
        "builder_source": _source(
            "release-source",
            BUILDER_SOURCE.rstrip("\n").splitlines(),
            kind="builder_source",
        ),
        "publisher_source": _source(
            "release-source",
            PUBLISHER_SOURCE.rstrip("\n").splitlines(),
            kind="publisher_source",
        ),
        "squeue": _source(
            "squeue",
            [*active_rows, *warm_rows, *client_rows],
        ),
        "sacct": _source(
            "sacct",
            [*active_rows, *warm_rows, *client_rows],
        ),
    }
    seed = {
        **_binding(),
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "dispatcher_source_sha256": DISPATCHER_SOURCE_SHA256,
        "qualification_runner_source_sha256": (
            QUALIFICATION_RUNNER_SOURCE_SHA256
        ),
        "capacity_generation": CAPACITY_GENERATION,
        "base_fleet_contract_path": str(authority["base_path"]),
        "base_fleet_contract_sha256": authority["base_sha256"],
        "effective_fleet_contract_path": str(authority["effective_path"]),
        "effective_fleet_contract_sha256": authority["effective_sha256"],
        "additive_overlay_contract_path": str(authority["effective_path"]),
        "additive_overlay_contract_sha256": authority["effective_sha256"],
        "static_feasibility_certificate": {
            "path": str(authority["certificate_path"]),
            "sha256": authority["certificate_sha256"],
            "certificate_id": authority["certificate"]["certificate_id"],
        },
        "fleet_contract_sha256": authority["effective_sha256"],
        "active_fleet_topology_sha256": hashlib.sha256(
            capacity.canonical_bytes(effective_topology)
        ).hexdigest(),
        "builder_source_sha256": hashlib.sha256(
            BUILDER_SOURCE.encode("utf-8")
        ).hexdigest(),
        "publisher_source_sha256": hashlib.sha256(
            PUBLISHER_SOURCE.encode("utf-8")
        ).hexdigest(),
        "expected_total_job_elements": 448,
        "job_element_accounting": {
            "cell_job_elements": 384,
            "active_server_job_elements": 22,
            "warm_turnover_job_elements": 3,
            "controller_monitor_other_held_job_elements": 39,
            "total_non_cell_reserve_job_elements": 64,
            "total_canary_job_elements": 448,
        },
        **source_fields,
    }
    derived = capacity._derive_capacity_from_raw_sources(
        seed,
        servers=servers,
        clients=clients,
        expected_source_tree_sha256=SOURCE_TREE_SHA256,
        expected_dispatcher_source_sha256=DISPATCHER_SOURCE_SHA256,
        expected_qualification_runner_source_sha256=(
            QUALIFICATION_RUNNER_SOURCE_SHA256
        ),
    )
    derived_scheduler_fields = {
        field: value
        for field, value in derived.items()
        if field in capacity._SCHEDULER_FIELDS
    }
    return {
        "schema_version": capacity.SCHEMA_VERSION,
        "protocol": capacity.SCHEDULER_EVIDENCE_PROTOCOL,
        "passed": True,
        **_binding(),
        "observed_at": 1_000.0,
        **derived_scheduler_fields,
        **source_fields,
        "scientific_server_placements": servers,
        "scientific_client_placements": clients,
        "occupancy_preflight": _empty_occupancy_preflight(),
    }


def _scheduler(root: Path) -> dict[str, object]:
    return capacity.with_self_hash(
        _scheduler_identity(root),
        identity_field="evidence_id",
    )


def test_server_capacity_totals_are_baseline_exact_then_generation_dynamic() -> None:
    baseline = {
        "partition": "gpu_protected",
        "qos": "gpu_science",
        "partition_preempt_mode": "OFF",
        "qos_preempt_mode": "OFF",
        "base_active_gpus": 24,
        "reserved_additive_gpus": 0,
        "effective_active_gpus": 24,
        "retained_warm_turnover_gpus": 4,
        "attested_total_gpus": 28,
        "partition_cpus": 4096,
        "partition_memory_mib": 33_554_432,
        "partition_gpus": 64,
        "partition_nodes": 8,
    }
    rows, totals = capacity._validate_scheduler_server_rows(
        [baseline],
        preempt_type="preempt/qos",
        capacity_generation=1,
    )
    assert rows == [baseline]
    assert totals["effective_active_gpus"] == 24
    assert totals["attested_total_gpus"] == 28

    expanded = {
        **baseline,
        "reserved_additive_gpus": 2,
        "effective_active_gpus": 26,
        "attested_total_gpus": 30,
    }
    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="generation-scoped",
    ):
        capacity._validate_scheduler_server_rows(
            [expanded],
            preempt_type="preempt/qos",
            capacity_generation=1,
        )
    rows, totals = capacity._validate_scheduler_server_rows(
        [expanded],
        preempt_type="preempt/qos",
        capacity_generation=2,
    )
    assert rows == [expanded]
    assert totals["effective_active_gpus"] == 26
    assert totals["attested_total_gpus"] == 30


def _canary_identity(
    scheduler: dict[str, object],
) -> dict[str, object]:
    server_rows = scheduler["scientific_server_placements"]
    client_rows = scheduler["scientific_client_placements"]
    assert isinstance(server_rows, list)
    assert isinstance(client_rows, list)

    def rows(values: list[object]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for value in values:
            assert isinstance(value, dict)
            result.append(
                {
                    "partition": value["partition"],
                    "qos": value["qos"],
                    "partition_preempt_mode": value["partition_preempt_mode"],
                    "qos_preempt_mode": value["qos_preempt_mode"],
                    "effective_requeue": 0,
                }
            )
        return result

    return {
        "schema_version": capacity.SCHEMA_VERSION,
        "protocol": capacity.CANARY_EVIDENCE_PROTOCOL,
        "passed": True,
        **_binding(),
        "completed_at": 1_001.0,
        "scheduler_evidence_id": scheduler["evidence_id"],
        "scientific_server_placements": rows(server_rows),
        "scientific_client_placements": rows(client_rows),
        "squeue_complete": True,
        "sacct_complete": True,
    }


def _canary(scheduler: dict[str, object]) -> dict[str, object]:
    return capacity.with_self_hash(
        _canary_identity(scheduler),
        identity_field="canary_id",
    )


def _write_sealed(path: Path, value: object) -> Path:
    path.write_bytes(capacity.canonical_bytes(value))
    path.chmod(0o444)
    return path


def _write_evidence(
    root: Path,
    *,
    scheduler: dict[str, object] | None = None,
    canary: dict[str, object] | None = None,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    scheduler_value = _scheduler(root) if scheduler is None else scheduler
    canary_value = _canary(scheduler_value) if canary is None else canary
    scheduler_path = _write_sealed(
        root / "SCHEDULER_EVIDENCE.json",
        scheduler_value,
    )
    canary_path = _write_sealed(
        root / "CANARY_EVIDENCE.json",
        canary_value,
    )
    return scheduler_path, canary_path, scheduler_value, canary_value


def _attest(
    recovery_root: Path,
    scheduler_path: Path,
    canary_path: Path,
    *,
    apply: bool,
) -> dict[str, object]:
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    sources = {
        tuple(scheduler[field]["argv"]): scheduler[field]["raw_output"]
        for field in (
            "scheduler_configuration",
            "partition_configuration",
            "qos_configuration",
            "association_configuration",
            "base_fleet_contract",
            "effective_fleet_contract",
            "additive_overlay_contract",
            "static_feasibility_certificate_source",
            "builder_source",
            "publisher_source",
            "squeue",
            "sacct",
        )
    }

    def runner(argv, *, timeout):
        assert timeout == 300.0
        raw = sources.get(tuple(argv))
        if raw is None:
            return subprocess.CompletedProcess(argv, 1, "", "unknown capture")
        return subprocess.CompletedProcess(argv, 0, raw, "")

    return capacity.attest(
        recovery_root=recovery_root,
        scheduler_evidence_path=scheduler_path,
        canary_evidence_path=canary_path,
        expected_release_git_commit=COMMIT,
        expected_release_tag_object=TAG_OBJECT,
        **_source_trust_kwargs(),
        apply=apply,
        runner=runner,
    )


def _reseal_scheduler(
    scheduler: dict[str, object],
) -> dict[str, object]:
    identity = dict(scheduler)
    identity.pop("evidence_id", None)
    return capacity.with_self_hash(identity, identity_field="evidence_id")


def _reseal_canary(
    canary: dict[str, object],
) -> dict[str, object]:
    identity = dict(canary)
    identity.pop("canary_id", None)
    return capacity.with_self_hash(identity, identity_field="canary_id")


def test_apply_publishes_only_read_only_marker_last(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, _, _ = _write_evidence(evidence_root)

    report = _attest(
        recovery_root,
        scheduler_path,
        canary_path,
        apply=True,
    )

    marker_path = recovery_root / capacity.MARKER_FILENAME
    assert report["status"] == "complete"
    assert list(recovery_root.iterdir()) == [marker_path]
    assert stat.S_IMODE(marker_path.stat().st_mode) == 0o444
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert set(marker) == capacity._MARKER_FIELDS
    assert marker["protocol"] == capacity.PROTOCOL
    assert marker["active_gpus"] == 24
    assert marker["warm_headroom_gpus"] == 4
    assert marker["static_feasibility_wave_passed"] is False
    assert marker["static_feasibility_selected_cell_count"] == 278
    assert marker["static_feasibility_target_cell_count"] == 384
    assert marker["static_feasibility_shortfall_cells"] == 106
    assert marker["static_feasibility_configured_client_ceiling"] == 384
    assert marker["static_feasibility_certified_saturation_target"] == 278
    assert marker["cell_ceiling"] == 384
    assert marker["reserve_jobs"] == 64
    assert marker["submit_headroom"] == 448
    assert marker["cpu"] == 384
    assert marker["memory_mib"] == 1_572_864
    assert marker["source_tree_sha256"] == SOURCE_TREE_SHA256
    assert marker["dispatcher_source_sha256"] == DISPATCHER_SOURCE_SHA256
    assert (
        marker["qualification_runner_source_sha256"]
        == QUALIFICATION_RUNNER_SOURCE_SHA256
    )
    assert marker["scheduler_evidence_id"] == _scheduler(evidence_root)["evidence_id"]
    assert marker["canary_id"] == _canary(_scheduler(evidence_root))["canary_id"]
    assert (
        marker["scheduler_evidence_sha256"]
        == hashlib.sha256(
            (evidence_root / "SCHEDULER_EVIDENCE.json").read_bytes()
        ).hexdigest()
    )
    assert (
        marker["canary_evidence_sha256"]
        == hashlib.sha256(
            (evidence_root / "CANARY_EVIDENCE.json").read_bytes()
        ).hexdigest()
    )
    assert marker["scientific_server_placements"] == (
        _scheduler(evidence_root)["scientific_server_placements"]
    )
    assert marker["scientific_client_placements"] == (
        _scheduler(evidence_root)["scientific_client_placements"]
    )
    identity = dict(marker)
    marker_id = identity.pop("marker_id")
    assert marker_id == capacity.identity_sha256(identity)
    assert (
        capacity.verify_marker(
            recovery_root,
            expected_release_git_commit=COMMIT,
            expected_release_tag_object=TAG_OBJECT,
            **_source_trust_kwargs(),
        )
        == marker
    )


def test_default_is_non_mutating_dry_run(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, _, _ = _write_evidence(evidence_root)

    report = _attest(
        recovery_root,
        scheduler_path,
        canary_path,
        apply=False,
    )

    assert report["status"] == "dry_run"
    assert report["marker"]["marker_id"]
    assert not any(recovery_root.iterdir())


def test_attest_and_verify_require_all_frozen_source_trust_anchors(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, _, _ = _write_evidence(evidence_root)

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="trust anchors are all required",
    ):
        capacity.attest(
            recovery_root=recovery_root,
            scheduler_evidence_path=scheduler_path,
            canary_evidence_path=canary_path,
            expected_release_git_commit=COMMIT,
            expected_release_tag_object=TAG_OBJECT,
        )

    _attest(recovery_root, scheduler_path, canary_path, apply=True)
    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="trust anchors are all required",
    ):
        capacity.verify_marker(
            recovery_root,
            expected_release_git_commit=COMMIT,
            expected_release_tag_object=TAG_OBJECT,
        )


@pytest.mark.parametrize(
    "field",
    [
        "expected_source_tree_sha256",
        "expected_dispatcher_source_sha256",
        "expected_qualification_runner_source_sha256",
    ],
)
def test_every_frozen_source_trust_anchor_rejects_drift(
    tmp_path: Path,
    field: str,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, _, _ = _write_evidence(evidence_root)
    anchors = _source_trust_kwargs()
    anchors[field] = "f" * 64

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="source trust anchors differ",
    ):
        capacity.attest(
            recovery_root=recovery_root,
            scheduler_evidence_path=scheduler_path,
            canary_evidence_path=canary_path,
            expected_release_git_commit=COMMIT,
            expected_release_tag_object=TAG_OBJECT,
            **anchors,
        )


def test_attest_and_marker_verification_supply_all_source_loader_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, object]] = []
    original = capacity.runtime_capacity.load_static_feasibility_certificate

    def load(*args, **kwargs):
        observed.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        capacity.runtime_capacity,
        "load_static_feasibility_certificate",
        load,
    )
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, _, _ = _write_evidence(evidence_root)
    observed.clear()

    _attest(recovery_root, scheduler_path, canary_path, apply=True)

    assert len(observed) == 2
    for call in observed:
        assert call["expected_source_tree_sha256"] == SOURCE_TREE_SHA256
        assert (
            call["expected_dispatcher_source_sha256"]
            == DISPATCHER_SOURCE_SHA256
        )
        assert (
            call["expected_qualification_runner_source_sha256"]
            == QUALIFICATION_RUNNER_SOURCE_SHA256
        )


def test_higher_scheduler_limit_preserves_exact_448_element_contract(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    scheduler["scheduler_max_submit_jobs"] = 500
    _replace_source_rows(
        scheduler,
        "qos_configuration",
        [
            "client_science|OFF|-|500|86400",
            "gpu_science|OFF|-|500|86400",
        ],
    )
    _replace_source_rows(
        scheduler,
        "association_configuration",
        [
            "cluster|account|tester|client_science,gpu_science|409|500"
        ],
    )
    scheduler["scientific_qos_contracts"] = [
        {
            **row,
            "max_submit_jobs_per_user": 500,
        }
        for row in scheduler["scientific_qos_contracts"]
    ]
    scheduler["occupancy_preflight"][
        "association_max_submit_jobs"
    ] = 500
    scheduler["occupancy_preflight"]["qos_max_submit_jobs"] = 500
    scheduler["occupancy_preflight"]["effective_max_submit_jobs"] = 500
    preflight_identity = dict(scheduler["occupancy_preflight"])
    preflight_identity.pop("preflight_id")
    scheduler["occupancy_preflight"]["preflight_id"] = hashlib.sha256(
        capacity.canonical_bytes(preflight_identity)
    ).hexdigest()
    scheduler = _reseal_scheduler(scheduler)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=_canary(scheduler),
    )

    marker = _attest(
        recovery_root,
        scheduler_path,
        canary_path,
        apply=False,
    )["marker"]

    assert marker["scheduler_max_submit_jobs"] == 500
    assert marker["cell_ceiling"] == 384
    assert marker["reserve_jobs"] == 64
    assert marker["submit_headroom"] == 448


@pytest.mark.parametrize("residual", [38, 40])
def test_residual_held_reserve_must_fill_dynamic_64_job_reserve(
    tmp_path: Path,
    residual: int,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    reserve_total = 22 + 3 + residual
    total = 384 + reserve_total
    accounting = scheduler["job_element_accounting"]
    assert isinstance(accounting, dict)
    accounting["controller_monitor_other_held_job_elements"] = residual
    accounting["total_non_cell_reserve_job_elements"] = reserve_total
    accounting["total_canary_job_elements"] = total
    scheduler["expected_total_job_elements"] = total
    clients = scheduler["scientific_client_placements"]
    assert isinstance(clients, list)
    assert isinstance(clients[0], dict)
    clients[0]["reserve_jobs"] = reserve_total
    clients[0]["submit_headroom"] = total
    for field in ("squeue", "sacct"):
        source = scheduler[field]
        assert isinstance(source, dict)
        raw = source["raw_output"]
        assert isinstance(raw, str)
        changed = raw.replace(
            "302|reserve|PENDING|cpu_protected|client_science|"
            "39|39|39936|0|0|43200|reserve-000",
            "302|reserve|PENDING|cpu_protected|client_science|"
            f"{residual}|{residual}|{residual * 1024}|0|0|43200|"
            "reserve-000",
        )
        assert changed != raw
        _replace_source_rows(
            scheduler,
            field,
            changed.rstrip("\n").splitlines(),
        )
    scheduler = _reseal_scheduler(scheduler)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=_canary(scheduler),
    )

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="inclusive 64-job non-cell reserve",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=False,
        )


def test_marker_canonically_sorts_authorized_placements(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    server_rows = scheduler["scientific_server_placements"]
    assert isinstance(server_rows, list)
    scheduler["scientific_server_placements"] = [
        {
            "partition": "z_gpu",
            "qos": "z_science",
            "partition_preempt_mode": "OFF",
            "qos_preempt_mode": "OFF",
            "base_active_gpus": 0,
            "reserved_additive_gpus": 0,
            "effective_active_gpus": 0,
            "retained_warm_turnover_gpus": 0,
            "attested_total_gpus": 0,
            "partition_cpus": 512,
            "partition_memory_mib": 2_097_152,
            "partition_gpus": 64,
            "partition_nodes": 4,
        },
        *server_rows,
        {
            "partition": "a_gpu",
            "qos": "a_science",
            "partition_preempt_mode": "OFF",
            "qos_preempt_mode": "OFF",
            "base_active_gpus": 0,
            "reserved_additive_gpus": 0,
            "effective_active_gpus": 0,
            "retained_warm_turnover_gpus": 0,
            "attested_total_gpus": 0,
            "partition_cpus": 512,
            "partition_memory_mib": 2_097_152,
            "partition_gpus": 64,
            "partition_nodes": 4,
        },
    ]
    _replace_source_rows(
        scheduler,
        "partition_configuration",
        [
            "a_gpu|OFF|UP|172800|512|2097152|64|4",
            "cpu_protected|OFF|UP|43200|384|1572864|0|8",
            "gpu_protected|OFF|UP|172800|4096|33554432|64|8",
            "z_gpu|OFF|UP|172800|512|2097152|64|4",
        ],
    )
    _replace_source_rows(
        scheduler,
        "qos_configuration",
        [
            "a_science|OFF|-|448|86400",
            "client_science|OFF|-|448|86400",
            "gpu_science|OFF|-|448|86400",
            "z_science|OFF|-|448|86400",
        ],
    )
    _replace_source_rows(
        scheduler,
        "association_configuration",
            [
                "cluster|account|tester|"
                "a_science,client_science,gpu_science,z_science|409|448"
            ],
    )
    scheduler = _reseal_scheduler(scheduler)
    canary = _canary(scheduler)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=canary,
    )

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="QOS contracts|placement",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )


def test_aggregate_split_client_capacity_is_not_one_usable_placement(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    scheduler["scientific_client_placements"] = [
        {
            "partition": "client_a",
            "qos": "science_a",
            "partition_preempt_mode": "OFF",
            "qos_preempt_mode": "OFF",
            "slots": 192,
            "cpus": 192,
            "memory_mib": 192 * 4096,
            "reserve_jobs": 32,
            "submit_headroom": 224,
        },
        {
            "partition": "client_b",
            "qos": "science_b",
            "partition_preempt_mode": "OFF",
            "qos_preempt_mode": "OFF",
            "slots": 192,
            "cpus": 192,
            "memory_mib": 192 * 4096,
            "reserve_jobs": 32,
            "submit_headroom": 224,
        },
    ]
    scheduler = _reseal_scheduler(scheduler)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=_canary(scheduler),
    )
    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="exactly one scientific client",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=False,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scheduler_evidence_id", "x" * 64),
        ("scheduler_evidence_sha256", "x" * 64),
        ("canary_id", "x" * 64),
        ("canary_evidence_sha256", "x" * 64),
    ],
)
def test_marker_rejects_invalid_source_bindings(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, _, _ = _write_evidence(evidence_root)
    marker = _attest(
        recovery_root,
        scheduler_path,
        canary_path,
        apply=False,
    )["marker"]
    marker[field] = value
    marker.pop("marker_id")
    marker = capacity.with_self_hash(marker, identity_field="marker_id")

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="placement/capacity/scheduler invariants",
    ):
        capacity.validate_marker_payload(
            marker,
            **_source_trust_kwargs(),
        )


@pytest.mark.parametrize(
    ("role", "mode_field"),
    [
        ("scientific_server_placements", "partition_preempt_mode"),
        ("scientific_server_placements", "qos_preempt_mode"),
        ("scientific_client_placements", "partition_preempt_mode"),
        ("scientific_client_placements", "qos_preempt_mode"),
    ],
)
def test_every_partition_and_qos_placement_must_be_off(
    tmp_path: Path,
    role: str,
    mode_field: str,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    rows = scheduler[role]
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    rows[0][mode_field] = "REQUEUE"
    scheduler = _reseal_scheduler(scheduler)
    canary = _canary(scheduler)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=canary,
    )

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match=r"PreemptMode='REQUEUE'",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )
    assert not any(recovery_root.iterdir())


@pytest.mark.parametrize(
    ("role", "field", "value", "message"),
    [
        (
            "scientific_server_placements",
            "effective_active_gpus",
            41,
            "effective active GPUs|placement",
        ),
        (
            "scientific_server_placements",
            "retained_warm_turnover_gpus",
            3,
            "retained warm turnover|placement",
        ),
        (
            "scientific_client_placements",
            "slots",
            383,
            "client cell_ceiling",
        ),
        (
            "scientific_client_placements",
            "cpus",
            383,
            "client cpu",
        ),
        (
            "scientific_client_placements",
            "memory_mib",
            1_572_863,
            "client memory_mib",
        ),
        (
            "scientific_client_placements",
            "reserve_jobs",
            63,
            "client reserve_jobs",
        ),
        (
            "scientific_client_placements",
            "submit_headroom",
            447,
            "client submit_headroom",
        ),
    ],
)
def test_each_capacity_floor_fails_closed(
    tmp_path: Path,
    role: str,
    field: str,
    value: int,
    message: str,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    rows = scheduler[role]
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    rows[0][field] = value
    scheduler = _reseal_scheduler(scheduler)
    canary = _canary(scheduler)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=canary,
    )

    with pytest.raises(capacity.ProtectedCapacityError, match=message):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )
    assert not any(recovery_root.iterdir())


@pytest.mark.parametrize("source_name", ["squeue", "sacct"])
def test_scheduler_sources_must_be_complete(
    tmp_path: Path,
    source_name: str,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    source = scheduler[source_name]
    assert isinstance(source, dict)
    source["complete"] = False
    scheduler = _reseal_scheduler(scheduler)
    canary = _canary(scheduler)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=canary,
    )

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match=f"{source_name} evidence is incomplete",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )


@pytest.mark.parametrize("field", ["squeue_complete", "sacct_complete"])
def test_canary_must_attest_both_scheduler_sources(
    tmp_path: Path,
    field: str,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    canary = _canary(scheduler)
    canary[field] = False
    canary = _reseal_canary(canary)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=canary,
    )

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match=r"complete joined squeue\+sacct",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )


def test_scheduler_drift_after_sealing_is_rejected(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, scheduler, _ = _write_evidence(evidence_root)
    rows = scheduler["scientific_client_placements"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    rows[0]["slots"] = 385
    rows[0]["cpus"] = 385
    rows[0]["memory_mib"] = 1_576_960
    rows[0]["submit_headroom"] = 449
    _replace_source_rows(
        scheduler,
        "partition_configuration",
        [
            "cpu_protected|OFF|UP|43200|385|1576960|0|8",
            "gpu_protected|OFF|UP|172800|4096|33554432|64|8",
        ],
    )
    scheduler_path.chmod(0o644)
    scheduler_path.write_bytes(capacity.canonical_bytes(scheduler))
    scheduler_path.chmod(0o444)

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="scheduler evidence self-hash",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )


def test_drift_between_validation_and_publication_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, _, _ = _write_evidence(evidence_root)
    original = capacity.build_marker

    def build_then_drift(*args: object, **kwargs: object) -> dict[str, object]:
        marker = original(*args, **kwargs)
        current = json.loads(scheduler_path.read_text(encoding="utf-8"))
        current["observed_at"] = 1_000.5
        current = _reseal_scheduler(current)
        scheduler_path.chmod(0o644)
        scheduler_path.write_bytes(capacity.canonical_bytes(current))
        scheduler_path.chmod(0o444)
        return marker

    monkeypatch.setattr(capacity, "build_marker", build_then_drift)
    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="drifted during protected-capacity attestation",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )
    assert not any(recovery_root.iterdir())


def test_byte_only_source_drift_before_publication_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, _, _ = _write_evidence(evidence_root)
    original = capacity.build_marker

    def build_then_reformat(
        *args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        marker = original(*args, **kwargs)
        current = json.loads(scheduler_path.read_text(encoding="utf-8"))
        scheduler_path.chmod(0o644)
        scheduler_path.write_text(
            json.dumps(current, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        scheduler_path.chmod(0o444)
        return marker

    monkeypatch.setattr(capacity, "build_marker", build_then_reformat)
    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="drifted during protected-capacity attestation",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )
    assert not any(recovery_root.iterdir())


def test_rehashed_raw_capture_drift_is_rejected(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    source = scheduler["sacct"]
    assert isinstance(source, dict)
    source["raw_output"] = f"{source['raw_output']}999|forged|RUNNING\n"
    scheduler = _reseal_scheduler(scheduler)
    canary = _canary(scheduler)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=canary,
    )

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="sacct raw capture/hash/cardinality",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )


def test_canary_must_bind_exact_scheduler_identity(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    canary = _canary(scheduler)
    canary["scheduler_evidence_id"] = "f" * 64
    canary = _reseal_canary(canary)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=canary,
    )

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="exact scheduler evidence ID",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )


def test_exact_release_anchor_rejects_consistent_substitution(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    scheduler["release_git_commit"] = "3" * 40
    scheduler = _reseal_scheduler(scheduler)
    canary = _canary(scheduler)
    canary["release_git_commit"] = "3" * 40
    canary = _reseal_canary(canary)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=canary,
    )

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="exact release/tag/chain",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )


def test_unknown_fields_fail_closed_even_when_rehashed(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    scheduler["escape_hatch"] = True
    scheduler = _reseal_scheduler(scheduler)
    canary = _canary(scheduler)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=canary,
    )

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="scheduler evidence fields drifted",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, scheduler, _ = _write_evidence(evidence_root)
    raw = capacity.canonical_bytes(scheduler).decode("utf-8")
    raw = raw.replace(
        f'"schema_version": {capacity.SCHEMA_VERSION},',
        (
            f'"schema_version": {capacity.SCHEMA_VERSION}, '
            f'"schema_version": {capacity.SCHEMA_VERSION},'
        ),
        1,
    )
    scheduler_path.chmod(0o644)
    scheduler_path.write_text(raw, encoding="utf-8")
    scheduler_path.chmod(0o444)

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="duplicate JSON key",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )


def test_writable_or_symlinked_evidence_is_not_sealed(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, _, _ = _write_evidence(evidence_root)
    scheduler_path.chmod(0o644)

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="read-only regular file",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )

    scheduler_path.chmod(0o444)
    link = tmp_path / "scheduler-link.json"
    link.symlink_to(scheduler_path)
    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="traverses a symlink",
    ):
        _attest(
            recovery_root,
            link,
            canary_path,
            apply=True,
        )


def test_canary_effective_requeue_must_be_zero(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler = _scheduler(evidence_root)
    canary = _canary(scheduler)
    rows = canary["scientific_client_placements"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    rows[0]["effective_requeue"] = 1
    canary = _reseal_canary(canary)
    scheduler_path, canary_path, _, _ = _write_evidence(
        evidence_root,
        scheduler=scheduler,
        canary=canary,
    )

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="did not prove Requeue=0",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )


def test_publication_is_idempotent_but_conflicting_evidence_cannot_replace(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, scheduler, _ = _write_evidence(evidence_root)
    first = _attest(
        recovery_root,
        scheduler_path,
        canary_path,
        apply=True,
    )
    second = _attest(
        recovery_root,
        scheduler_path,
        canary_path,
        apply=True,
    )
    assert first["status"] == "complete"
    assert second["status"] == "already_complete"

    scheduler["observed_at"] = 1_000.5
    scheduler = _reseal_scheduler(scheduler)
    canary = _canary(scheduler)
    scheduler_path.chmod(0o644)
    canary_path.chmod(0o644)
    scheduler_path.write_bytes(capacity.canonical_bytes(scheduler))
    canary_path.write_bytes(capacity.canonical_bytes(canary))
    scheduler_path.chmod(0o444)
    canary_path.chmod(0o444)

    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="existing protected-capacity marker conflicts",
    ):
        _attest(
            recovery_root,
            scheduler_path,
            canary_path,
            apply=True,
        )
    marker = json.loads(
        (recovery_root / capacity.MARKER_FILENAME).read_text(encoding="utf-8")
    )
    assert marker["cell_ceiling"] == 384


def test_existing_marker_must_remain_read_only_and_self_hashed(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    recovery_root = tmp_path / "recovery"
    evidence_root.mkdir()
    recovery_root.mkdir()
    scheduler_path, canary_path, _, _ = _write_evidence(evidence_root)
    _attest(
        recovery_root,
        scheduler_path,
        canary_path,
        apply=True,
    )
    marker_path = recovery_root / capacity.MARKER_FILENAME
    marker_path.chmod(0o644)
    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="read-only regular file",
    ):
        capacity.verify_marker(
            recovery_root,
            **_source_trust_kwargs(),
        )

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["active_gpus"] = 25
    marker_path.write_bytes(capacity.canonical_bytes(marker))
    marker_path.chmod(0o444)
    with pytest.raises(
        capacity.ProtectedCapacityError,
        match="marker self-hash",
    ):
        capacity.verify_marker(
            recovery_root,
            **_source_trust_kwargs(),
        )


def test_cli_has_no_scheduler_mutation_capability() -> None:
    source = Path(capacity.__file__).read_text(encoding="utf-8")
    assert "os.system" not in source
    assert 'add_parser("submit"' not in source
    assert "sbatch" not in source
    assert "scancel" not in source


def test_publisher_and_runtime_marker_schemas_are_exactly_identical() -> None:
    assert capacity._MARKER_FIELDS == capacity.runtime_capacity._MARKER_FIELDS
