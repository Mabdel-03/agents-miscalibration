"""Focused tests for the isolated schema-5 throughput qualification producer."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from scripts import render_schema5_recovery_chain_v12 as renderer
from scripts import run_schema5_throughput_qualification as qualification


COMMIT = "1" * 40
TAG_OBJECT = "2" * 40
MANIFEST_SHA256 = "a" * 64


def _sealed(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(qualification._canonical_bytes(value))
    path.chmod(0o444)
    return path


def _prerequisite(
    recovery_root: Path,
    name: str,
    *,
    protocol: str,
    identity_field: str,
    identity: str,
) -> dict[str, object]:
    return {
        "marker": str((recovery_root / name).resolve()),
        "marker_sha256": "b" * 64,
        "marker_size": 123,
        "protocol": protocol,
        identity_field: identity,
    }


def _protected_capacity_prerequisite(
    recovery_root: Path,
) -> dict[str, object]:
    marker_identity = {
        "schema_version": qualification.protected_capacity.SCHEMA_VERSION,
        "protocol": qualification.protected_capacity.PROTOCOL,
        "passed": True,
        "release_id": qualification.protected_capacity.RELEASE_ID,
        "release_tag": qualification.protected_capacity.RELEASE_TAG,
        "release_git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "chain_namespace": qualification.protected_capacity.CHAIN_NAMESPACE,
        "active_gpus": 24,
        "warm_headroom_gpus": 4,
        "cell_ceiling": 384,
        "reserve_jobs": 64,
        "submit_headroom": 448,
        "cpu": 384,
        "memory_mib": 384 * 4_096,
        "preempt_type": "preempt/partition_prio",
        "capacity_source": qualification.protected_capacity.CAPACITY_SOURCE,
        "scheduler_cluster": "test_cluster",
        "scheduler_account": "test_account",
        "scheduler_user": "test_user",
        "scheduler_max_submit_jobs": 500,
        "partition_cpus": 384,
        "partition_memory_mib": 384 * 4_096,
        "partition_gpus": 28,
        "fleet_contract_sha256": "1" * 64,
        "active_fleet_topology_sha256": "2" * 64,
        "scientific_server_preempt_mode": "OFF",
        "scientific_client_preempt_mode": "OFF",
        "squeue_complete": True,
        "sacct_complete": True,
        "scheduler_evidence_id": "3" * 64,
        "scheduler_evidence_sha256": "4" * 64,
        "canary_id": "5" * 64,
        "canary_evidence_sha256": "6" * 64,
        "scientific_server_placements": [
            {
                "partition": "gpu_protected",
                "qos": "gpu_qos",
                "partition_preempt_mode": "OFF",
                "qos_preempt_mode": "cluster",
                "active_serving_gpus": 24,
                "warm_headroom_gpus": 4,
            }
        ],
        "scientific_client_placements": [
            {
                "partition": "ou_bcs_normal",
                "qos": "normal",
                "partition_preempt_mode": "OFF",
                "qos_preempt_mode": "cluster",
                "slots": 384,
                "cpus": 384,
                "memory_mib": 384 * 4_096,
                "reserve_jobs": 64,
                "submit_headroom": 448,
            }
        ],
    }
    marker = qualification._with_identity(marker_identity, "marker_id")
    path = _sealed(
        recovery_root / renderer.PROTECTED_CAPACITY_MARKER_NAME,
        marker,
    )
    raw = path.read_bytes()
    return {
        "marker": str(path.resolve()),
        "marker_sha256": qualification._sha256_bytes(raw),
        "marker_size": len(raw),
        "protocol": renderer.PROTECTED_CAPACITY_PROTOCOL,
        "marker_id": marker["marker_id"],
    }


def _chain_manifest(tmp_path: Path) -> Path:
    results_root = (tmp_path / "results").resolve()
    recovery_root = results_root / "recovery" / "schema5-v1"
    readiness_root = recovery_root / "readiness"
    identity = {
        "schema_version": renderer.CHAIN_SCHEMA_VERSION,
        "protocol": "schema5-v1.2-r2-recovery-chain",
        "namespace": renderer.CHAIN_NAMESPACE,
        "release_id": renderer.RELEASE_ID,
        "release_tag": renderer.RELEASE_TAG,
        "release_git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "results_root": str(results_root),
        "recovery_root": str(recovery_root),
        "readiness_root": str(readiness_root),
        "state_root": str(
            results_root / qualification.control.CONTROL_STATE_DIRNAME
        ),
        "server_pool_root": str(
            results_root / "server_pools" / "schema5-v1"
        ),
        "release_root": str(recovery_root / "release"),
        "hf_home": str((tmp_path / "hf").resolve()),
        "prerequisite_evidence": {
            "protected_capacity": _protected_capacity_prerequisite(
                recovery_root
            ),
        },
    }
    manifest = qualification._with_identity(identity, "chain_id")
    return _sealed(tmp_path / "CHAIN.json", manifest)


def _guard(*, rollout_generation: int = 0) -> dict[str, object]:
    value: dict[str, object] = {
        "immutable_sha256": "f" * 64,
        "desired_state": "paused",
        "drain_requested": False,
        "rollout_generation": rollout_generation,
        "production_run_ids": list(qualification.control.REQUIRED_RUNS),
        "admission": {"current_ceiling": 24},
        "admission_ramp": {"current_ceiling": 24},
        "admission_safety_hold": {"active": False},
    }
    value["guard_sha256"] = qualification._sha256_bytes(
        qualification._canonical_bytes(value)
    )
    return value


def _readiness_generation(tmp_path: Path) -> dict[str, object]:
    return {
        "catalog_id": "7" * 64,
        "marker_path": str((tmp_path / "TRUSTED_GENERATION.json").resolve()),
        "marker_sha256": "8" * 64,
        "inventory_sha256": "9" * 64,
        "catalog_payload_sha256": "a" * 64,
        "allowed_generation_tuple_count": 24,
        "release_fleet_contract_sha256": "b" * 64,
        "fleet_contract_sha256": "c" * 64,
        "capacity_generation": 1,
        "rollout_generation": 1,
    }


def _context_and_intent(
    tmp_path: Path,
    *,
    write_execution_authority: bool = True,
) -> tuple[qualification.QualificationContext, dict[str, object]]:
    base = qualification.load_qualification_context(
        _chain_manifest(tmp_path), verify_chain=False
    )
    readiness = _readiness_generation(tmp_path)
    context = qualification.create_or_load_attempt_context(
        base,
        control_value={},
        readiness_generation=readiness,
        now=850.0,
    )
    intent = qualification._with_identity(
        qualification._intent_identity(
            context,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            readiness_generation=readiness,
            control_guard=_guard(),
            created_timestamp=900.0,
        ),
        "intent_id",
    )
    context.qualification_root.mkdir(parents=True, exist_ok=True)
    context.run_root.mkdir(parents=True, exist_ok=True)
    qualification._write_once(
        context.qualification_root / qualification.INTENT_NAME,
        intent,
        description="test intent",
    )
    qualification._write_once(
        context.qualification_root / qualification.PLAN_NAME,
        qualification.build_load_plan(),
        description="test plan",
    )
    if write_execution_authority:
        authority = qualification._with_identity(
            {
                "schema_version": qualification.SCHEMA_VERSION,
                "protocol": qualification.EXECUTION_AUTHORITY_PROTOCOL,
                "intent_id": intent["intent_id"],
            },
            "authority_id",
        )
        qualification._write_once(
            context.qualification_root
            / qualification.EXECUTION_AUTHORITY_NAME,
            authority,
            description="test execution authority",
        )
    return context, intent


def _progress(total: int) -> dict[str, int]:
    labels = qualification.expected_stratum_labels()
    quotient, remainder = divmod(total, len(labels))
    return {
        label: quotient + int(index < remainder)
        for index, label in enumerate(labels)
    }


def _job(sequence: int) -> dict[str, str]:
    return {
        "job_id": str(10_000 + sequence),
        "job_name": f"asys-dispatch-{sequence:010d}",
        "state": "RUNNING",
        "comment": f"asys-schema5-intent:test-{sequence}",
        "command": f"/sealed/batch-{sequence}.sbatch",
        "source": "squeue",
        "dependency": "",
    }


def _record(
    context: qualification.QualificationContext,
    intent: dict[str, object],
    *,
    sequence: int,
    timestamp: float,
    ceiling: int,
    active: int,
    useful: int,
    complete: bool = False,
) -> None:
    execution_authority = (
        qualification._execution_authority_evidence_binding(intent)
    )
    jobs = [] if active == 0 else [_job(sequence)]
    scheduler = qualification.make_scheduler_evidence(
        intent_id=str(intent["intent_id"]),
        sequence=sequence,
        captured_timestamp=timestamp,
        ceiling=ceiling,
        jobs=jobs,
        qualification_job_ids=(
            [] if active == 0 else [jobs[0]["job_id"]]
        ),
        active_qualification_cells=active,
        dispatcher_ledger_sha256=(
            None if sequence == 0 else "9" * 64
        ),
        production_control_guard_sha256=str(
            intent["control_guard"]["guard_sha256"]  # type: ignore[index]
        ),
        client_partition=str(intent["client_partition"]),
        client_qos=str(intent["client_qos"]),
        protected_capacity_marker_id=str(
            intent["client_placement"]["protected_capacity_marker_id"]  # type: ignore[index]
        ),
        protected_capacity_marker_sha256=str(
            intent["client_placement"]["protected_capacity_marker_sha256"]  # type: ignore[index]
        ),
        readiness_rollout_generation=int(
            intent["readiness_generation"]["rollout_generation"]  # type: ignore[index]
        ),
        trusted_generation_catalog_id=str(
            intent["readiness_generation"]["catalog_id"]  # type: ignore[index]
        ),
        qualification_execution_authority_id=(
            execution_authority["authority_id"]
        ),
        qualification_execution_authority_sha256=(
            execution_authority["sha256"]
        ),
    )
    semantic = qualification.make_semantic_evidence(
        intent_id=str(intent["intent_id"]),
        sequence=sequence,
        captured_timestamp=timestamp,
        manifest_sha256=MANIFEST_SHA256,
        states=(
            {"complete": qualification.CELL_COUNT}
            if complete
            else {
                "active": active,
                "missing": qualification.CELL_COUNT - active,
            }
        ),
        validated_qids=useful,
        useful_qids=useful,
        strata_progress=_progress(useful),
        artifact_schema_counts=({} if useful == 0 else {"5": useful}),
    )
    qualification.record_observation(
        context.qualification_root,
        intent=intent,
        scheduler=scheduler,
        semantic=semantic,
    )


def _passing_observations(
    context: qualification.QualificationContext,
    intent: dict[str, object],
) -> list[dict[str, object]]:
    rows = [
        (1_000.0, 24, 0, 0, False),
        (1_010.0, 24, 24, 24, False),
        (1_020.0, 96, 96, 120, False),
        (1_030.0, 192, 192, 312, False),
        (1_040.0, 384, 384, 696, False),
    ]
    for index in range(1, 13):
        timestamp = 1_040.0 + 600.0 * index
        completed = index >= 10
        useful = (
            qualification.TOTAL_QIDS
            if completed
            else 696 + index * 900
        )
        rows.append(
            (
                timestamp,
                384,
                0 if completed else 384,
                useful,
                completed,
            )
        )
    for sequence, (timestamp, ceiling, active, useful, complete) in enumerate(rows):
        _record(
            context,
            intent,
            sequence=sequence,
            timestamp=timestamp,
            ceiling=ceiling,
            active=active,
            useful=useful,
            complete=complete,
        )
    return qualification.load_observations(
        context.qualification_root, intent=intent
    )


def test_load_design_is_exact_deterministic_and_balanced() -> None:
    first = qualification.generate_qualification_cells()
    second = qualification.generate_qualification_cells()

    assert first == second
    assert len(first) == 768
    assert len({cell.cell_id for cell in first}) == 768
    assert {cell.n_questions for cell in first} == {20}
    assert len({qualification._stratum_tuple(cell) for cell in first}) == 570
    assert all(
        cell.cell_id not in qualification.PRODUCTION_RUN_IDS for cell in first
    )

    plan = qualification.build_load_plan()
    assert plan["cell_count"] == 768
    assert plan["qids"] == 15_360
    assert plan["ceilings"] == [24, 96, 192, 384]
    assert plan["balance"]["benchmark"] == {
        "gpqa": 192,
        "math": 192,
        "mmlu_pro": 192,
        "truthfulqa": 192,
    }
    assert plan["balance"]["seed"] == {"0": 256, "1": 256, "2": 256}
    assert plan["balance"]["prompt_complexity_level"] == {
        "0": 270,
        "1": 228,
        "3": 270,
    }
    assert plan["balance"]["sharing_topology_context"] == {
        "artifact_only": 180,
        "plus_cot": 180,
    }
    qualification.validate_load_plan(plan)


def test_load_plan_strict_schema_and_self_hash_reject_drift() -> None:
    plan = qualification.build_load_plan()
    identity = dict(plan)
    observed = identity.pop("plan_id")
    assert observed == qualification._sha256_bytes(
        qualification._canonical_bytes(identity)
    )

    drifted = dict(plan)
    drifted["qids"] -= 1
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="invalid plan_id",
    ):
        qualification.validate_load_plan(drifted)

    extra = dict(plan)
    extra["operator_override"] = True
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="fields drifted",
    ):
        qualification.validate_load_plan(extra)


def test_sealed_json_rejects_noncanonical_and_hardlinked_files(
    tmp_path: Path,
) -> None:
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text('{"value": 1}\n', encoding="utf-8")
    noncanonical.chmod(0o444)
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="not canonically encoded",
    ):
        qualification._read_json(
            noncanonical,
            description="noncanonical fixture",
            sealed=True,
        )

    source = _sealed(tmp_path / "shared.json", {"value": 1})
    alias = tmp_path / "shared-alias.json"
    alias.hardlink_to(source)
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="unique read-only regular file",
    ):
        qualification._read_json(
            source,
            description="hardlinked fixture",
            sealed=True,
        )


def test_sealed_json_rejects_mutation_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _sealed(tmp_path / "mutable.json", {"value": 1})
    original_read = qualification.os.read
    mutated = False

    def mutate_after_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, size)
        if chunk and not mutated:
            mutated = True
            path.chmod(0o644)
            path.write_bytes(
                qualification._canonical_bytes({"value": 2})
            )
            path.chmod(0o444)
        return chunk

    monkeypatch.setattr(qualification.os, "read", mutate_after_read)
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="changed during its sealed read",
    ):
        qualification._read_json(
            path,
            description="concurrently changed fixture",
            sealed=True,
        )


def test_dry_run_and_dispatch_commands_never_name_production_runs(
    tmp_path: Path,
) -> None:
    manifest = _chain_manifest(tmp_path)
    context = qualification.load_qualification_context(
        manifest, verify_chain=False
    )
    report = qualification.dry_run_report(
        manifest,
        client_partition="ou_bcs_normal",
        client_qos="normal",
        verify_chain=False,
    )
    derived = qualification.dry_run_report(
        manifest,
        verify_chain=False,
    )

    assert report["status"] == "dry_run"
    assert report["submitted"] is False
    assert report["cells"] == 768
    assert report["qids"] == 15_360
    assert derived["client_placement"] == report["client_placement"]
    assert not context.qualification_root.exists()
    for stage in report["stages"]:
        argv = stage["dispatcher_argv"]
        joined = " ".join(argv)
        assert qualification.QUALIFICATION_RUN_ID in joined
        assert "--control-state-dir" not in argv
        assert argv[argv.index("--cell-partition") + 1] == "ou_bcs_normal"
        assert argv[argv.index("--cell-qos") + 1] == "normal"
        assert (
            argv[argv.index("--protected-capacity-marker-id") + 1]
            == context.protected_capacity_contract.marker_id
        )
        assert "--protected-capacity-release-git-commit" in argv
        assert argv[
            argv.index("--qualification-execution-authority") + 1
        ] == str(
            context.qualification_base
            / qualification.ATTEMPT_DIRECTORY
            / "VERIFIED_GENERATION_AT_EXECUTE"
            / qualification.EXECUTION_AUTHORITY_NAME
        )
        assert str(context.state_root) not in argv
        assert not any(
            run_id in joined for run_id in qualification.PRODUCTION_RUN_IDS
        )

    argv = qualification.dispatcher_command(
        context,
        client_partition="ou_bcs_normal",
        client_qos="normal",
        max_batch=7,
    )
    assert argv[argv.index("--max-batch") + 1] == "7"
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="within 1..24",
    ):
        qualification.dispatcher_command(
            context,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            max_batch=25,
        )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="must be supplied together",
    ):
        qualification.dry_run_report(
            manifest,
            client_partition="ou_bcs_normal",
            verify_chain=False,
        )


def test_marker_first_attempt_pointer_recovers_missing_current_cache(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    base = qualification.load_qualification_context(
        context.chain_manifest, verify_chain=False
    )
    pointer_path = context.attempt_pointer_path
    assert pointer_path is not None
    pointer_bytes = pointer_path.read_bytes()
    current_path = base.qualification_base / qualification.CURRENT_ATTEMPT_NAME
    current_path.unlink()

    recovered = qualification.create_or_load_attempt_context(
        base,
        control_value={},
        readiness_generation=intent["readiness_generation"],
        now=9_999.0,
    )

    assert recovered.qualification_root == context.qualification_root
    assert recovered.run_root == context.run_root
    assert recovered.attempt_pointer == context.attempt_pointer
    assert pointer_path.read_bytes() == pointer_bytes
    assert stat.S_IMODE(current_path.stat().st_mode) == 0o444
    assert len(qualification._load_attempt_pointers(base)) == 1


def test_current_attempt_cache_rejects_stale_publication_orphan(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    base = qualification.load_qualification_context(
        context.chain_manifest, verify_chain=False
    )
    current_path = base.qualification_base / qualification.CURRENT_ATTEMPT_NAME
    orphan = current_path.with_name(
        f".{current_path.name}.crashed.publishing"
    )
    orphan.write_bytes(b"incomplete")
    orphan.chmod(0o444)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="unreconciled stale publication",
    ):
        qualification.create_or_load_attempt_context(
            base,
            control_value={},
            readiness_generation=intent["readiness_generation"],
            now=9_999.0,
        )


@pytest.mark.parametrize(
    "target_name",
    [qualification.EVIDENCE_NAME, qualification.MARKER_NAME],
)
def test_completion_rejects_stale_attempt_publication_orphan(
    tmp_path: Path,
    target_name: str,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)
    if target_name == qualification.MARKER_NAME:
        evaluation = qualification.evaluate_observations(observations)
        evidence = qualification._evidence_summary(
            intent=intent,
            observations=observations,
            evaluation=evaluation,
        )
        qualification._write_once(
            context.qualification_root / qualification.EVIDENCE_NAME,
            evidence,
            description="test aggregate evidence",
        )
    target = context.qualification_root / target_name
    orphan = target.with_name(f".{target.name}.crashed.publishing")
    orphan.write_bytes(b"incomplete")
    orphan.chmod(0o444)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="unreconciled stale publication",
    ):
        qualification.publish_completion(context, intent=intent)
    assert not (
        context.qualification_base / qualification.MARKER_NAME
    ).exists()


def test_terminal_tree_seal_rejects_external_hardlink_alias(
    tmp_path: Path,
) -> None:
    context, _intent = _context_and_intent(tmp_path)
    evidence = context.run_root / "sealed-evidence.json"
    evidence.write_bytes(qualification._canonical_bytes({"sealed": True}))
    evidence.chmod(0o444)
    alias = tmp_path / "external-hardlink-alias.json"
    alias.hardlink_to(evidence)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="non-unique or unsafe member",
    ):
        qualification._seal_tree_read_only(
            context.run_root,
            description="hardlinked terminal run",
        )


def test_sealed_observations_prove_all_gates_and_publish_renderer_marker(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)

    evaluation = qualification.evaluate_observations(observations)
    assert evaluation["passed"] is True
    assert evaluation["steady_384_seconds"] == 7_200
    assert evaluation["throughput_qids_per_day"] >= 201_994
    assert evaluation["peak_active_cells"] == {
        "24": 24,
        "96": 96,
        "192": 192,
        "384": 384,
    }

    marker = qualification.publish_completion(context, intent=intent)
    marker_path = context.qualification_root / qualification.MARKER_NAME
    assert stat.S_IMODE(marker_path.stat().st_mode) == 0o444
    assert set(marker) == {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "chain_id",
        "manifest",
        "manifest_sha256",
        "protected_capacity",
        "attempt",
        "cells",
        "qids",
        "ceilings",
        "steady_384_seconds",
        "throughput_qids_per_day",
        "every_stratum_progress",
        "integrity_incidents",
        "transport_censor_incidents",
        "qualification_id",
    }
    identity = dict(marker)
    qualification_id = identity.pop("qualification_id")
    assert qualification_id == renderer._sha256_bytes(
        renderer._canonical_json(identity)
    )

    verified = qualification.verify_completed_qualification(
        context.chain_manifest,
        verify_chain=False,
        verify_renderer=False,
    )
    assert verified["qualification_id"] == qualification_id
    assert verified["cells"] == 768
    assert verified["qids"] == 15_360


def test_fixed_completion_marker_is_committed_after_attempt_trees_are_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _passing_observations(context, intent)
    root_marker = context.qualification_base / qualification.MARKER_NAME
    write_once = qualification._write_once

    def interrupt_fixed_commit(
        path: Path,
        payload: object,
        *,
        description: str,
        mode: int = 0o444,
    ) -> None:
        if path == root_marker:
            raise qualification.ThroughputQualificationError(
                "injected fixed-marker interruption"
            )
        write_once(
            path,
            payload,
            description=description,
            mode=mode,
        )

    monkeypatch.setattr(
        qualification,
        "_write_once",
        interrupt_fixed_commit,
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="injected fixed-marker interruption",
    ):
        qualification.publish_completion(context, intent=intent)
    assert not root_marker.exists()
    qualification._assert_tree_read_only(
        context.qualification_root,
        description="interrupted successful attempt",
    )
    qualification._assert_tree_read_only(
        context.run_root,
        description="interrupted successful run",
    )

    monkeypatch.setattr(qualification, "_write_once", write_once)
    marker = qualification.publish_completion(context, intent=intent)
    verified = qualification.verify_completed_qualification(
        context.chain_manifest,
        verify_chain=False,
        verify_renderer=False,
    )
    assert root_marker.read_bytes() == qualification._canonical_bytes(marker)
    assert verified["attempt_id"] == context.attempt_pointer["attempt_id"]


def test_verify_only_rejects_tampered_sealed_semantic_evidence(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _passing_observations(context, intent)
    qualification.publish_completion(context, intent=intent)

    semantic_path = (
        context.qualification_root
        / qualification.SEMANTIC_DIRECTORY
        / qualification._evidence_filename("SEMANTIC", 1)
    )
    payload = json.loads(semantic_path.read_text(encoding="utf-8"))
    payload["useful_qids"] += 1
    semantic_path.chmod(0o644)
    semantic_path.write_bytes(qualification._canonical_bytes(payload))
    semantic_path.chmod(0o444)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="invalid semantic_id",
    ):
        qualification.verify_completed_qualification(
            context.chain_manifest,
            verify_chain=False,
            verify_renderer=False,
        )


def test_partial_orphan_observation_fails_closed(tmp_path: Path) -> None:
    context, intent = _context_and_intent(tmp_path)
    orphan = (
        context.qualification_root
        / qualification.SCHEDULER_DIRECTORY
        / qualification._evidence_filename("SCHEDULER", 0)
    )
    _sealed(orphan, {"partial": True})

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="orphaned partial",
    ):
        qualification.load_observations(
            context.qualification_root, intent=intent
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("censor", "integrity, censor, or namespace incident"),
        ("stratum", "semantic progress regressed"),
        ("gap", "evidence gap"),
        ("throughput", "below 201,994"),
    ],
)
def test_acceptance_fails_closed_on_scientific_or_timing_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    observations = _passing_observations(context, intent)

    if mutation == "censor":
        observations[5]["semantic"]["transport_censor_incidents"] = 1
    elif mutation == "stratum":
        observations[6]["semantic"]["useful_qids"] = 1
    elif mutation == "gap":
        # Preserve increasing time while introducing a >660-second 384 gap.
        observations[6]["receipt"]["captured_timestamp"] = (
            observations[5]["receipt"]["captured_timestamp"] + 661
        )
    else:
        # Delay the first complete state by one normal 600-second observation,
        # preserving the independently required <=660-second evidence cadence.
        for observation in observations:
            semantic = observation["semantic"]
            if semantic["states"] == {"complete": qualification.CELL_COUNT}:
                semantic["states"] = {"partial": qualification.CELL_COUNT}
                break

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match=message,
    ):
        qualification.evaluate_observations(observations)


def test_rotation_counts_are_optimal_under_structural_context_constraints() -> None:
    cells = qualification.generate_qualification_cells()
    by_topology = Counter(cell.topology.value for cell in cells)
    assert by_topology == {
        "single_agent": 228,
        "independent": 180,
        "decentralized": 180,
        "centralized": 180,
    }
    # Context is balanced exactly where it is a real treatment.  Single-agent and
    # independent cells are correctly pinned to the canonical artifact-only value.
    sharing = [
        cell
        for cell in cells
        if cell.topology
        in {qualification.Topology.DECENTRALIZED, qualification.Topology.CENTRALIZED}
    ]
    assert Counter(cell.context_share_level.value for cell in sharing) == {
        "artifact_only": 180,
        "plus_cot": 180,
    }
    assert all(
        cell.context_share_level.value == "artifact_only"
        for cell in cells
        if cell.topology
        in {qualification.Topology.SINGLE_AGENT, qualification.Topology.INDEPENDENT}
    )


def test_paused_but_draining_control_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = qualification.load_qualification_context(
        _chain_manifest(tmp_path), verify_chain=False
    )
    control_value = {
        "desired_state": "paused",
        "drain_requested": True,
        "rollout_generation": 0,
        "immutable_sha256": "f" * 64,
        "immutable": {
            "runs": [
                {"run_id": run_id}
                for run_id in sorted(qualification.PRODUCTION_RUN_IDS)
            ]
        },
        "readiness": {"smoke_runs": {"passed": True}},
        "admission": {},
        "admission_ramp": {},
        "admission_safety_hold": {},
    }
    monkeypatch.setattr(
        qualification.control,
        "load_control",
        lambda *_args, **_kwargs: control_value,
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="requires paused",
    ):
        qualification.load_paused_control(context)


def test_runtime_generation_is_catalog_bound_not_blindly_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, intent = _context_and_intent(tmp_path)
    generation = dict(intent["readiness_generation"])
    generation.update(
        {
            "rollout_generation": 7,
            "capacity_generation": 3,
            "fleet_contract_sha256": "d" * 64,
            "release_fleet_contract_sha256": "e" * 64,
        }
    )
    runtime_intent = {**intent, "readiness_generation": generation}
    control_value = {
        "rollout_generation": 6,
        "immutable": {"model_contract_path": str(tmp_path / "models.json")},
    }
    attestation = {
        "schema_version": 1,
        "generation": 7,
        "path": str((tmp_path / "attestation.json").resolve()),
        "sha256": "2" * 64,
        "attestation_id": "3" * 64,
        "lease_path": str((tmp_path / "lease.json").resolve()),
    }
    inherited = {
        key: "x"
        for key in qualification.dispatch_sweeps.PRODUCTION_ENVIRONMENT_KEYS
    }
    inherited.update(
        {
            "ASYS_RELEASE_GIT_COMMIT": COMMIT,
            "ASYS_PROTECTED_CAPACITY_MARKER": str(
                context.protected_capacity_contract.path
            ),
            "ASYS_PROTECTED_CAPACITY_MARKER_SHA256": (
                context.protected_capacity_contract.sha256
            ),
            "ASYS_PROTECTED_CAPACITY_MARKER_ID": (
                context.protected_capacity_contract.marker_id
            ),
            "ASYS_MODEL_CONTRACT_SHA256": "4" * 64,
            "ASYS_FLEET_CONTRACT_SHA256": "d" * 64,
            "ASYS_FLEET_CONTRACT_PATH": str(
                (tmp_path / "fleet.json").resolve()
            ),
            "ASYS_RELEASE_FLEET_CONTRACT_SHA256": "e" * 64,
            "ASYS_CAPACITY_GENERATION": "3",
            "ASYS_HARNESS_ENVIRONMENT_SHA256": "5" * 64,
            "ASYS_SERVING_ENVIRONMENT_SHA256": "6" * 64,
            "ASYS_ROLLOUT_GENERATION": "7",
            "ASYS_IMMUTABLE_PINS_SHA256": "7" * 64,
            "ASYS_RUNTIME_ATTESTATION": attestation["path"],
            "ASYS_RUNTIME_ATTESTATION_SHA256": attestation["sha256"],
            "ASYS_RUNTIME_INTEGRITY_LEASE": attestation["lease_path"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    projected: dict[str, object] = {}
    monkeypatch.setattr(
        qualification.control,
        "ensure_runtime_integrity_attestation",
        lambda state, value, *, generation, force_full: (
            projected.update(
                state=state,
                source=value,
                generation=generation,
                force_full=force_full,
            )
            or attestation
        ),
    )
    monkeypatch.setattr(
        qualification.control,
        "validate_runtime_integrity_attestation",
        lambda value, *, verify_metadata: (
            projected.update(
                projected_control=value,
                verify_metadata=verify_metadata,
            )
            or attestation
        ),
    )
    monkeypatch.setattr(
        qualification.control,
        "production_environment",
        lambda value: (
            projected.update(production_control=value) or inherited
        ),
    )
    monkeypatch.setattr(
        qualification,
        "load_artifact_policy",
        lambda *_args, **_kwargs: SimpleNamespace(file_sha256="1" * 64),
    )
    environment = qualification._execution_environment(
        context,
        control_value=control_value,
        intent=runtime_intent,
    )
    assert environment["ASYS_ROLLOUT_GENERATION"] == "7"
    assert projected["generation"] == 7
    assert projected["state"] == context.qualification_root
    assert projected["force_full"] is False
    projected_control = projected["projected_control"]
    assert isinstance(projected_control, dict)
    assert projected_control["rollout_generation"] == 7
    assert (
        projected_control[qualification.control.RUNTIME_ATTESTATION_STATE_KEY]
        == attestation
    )
    assert projected["production_control"] is projected_control
    assert projected["verify_metadata"] is True
    result_row = {
        field: generation[field]
        for field in (
            "release_fleet_contract_sha256",
            "fleet_contract_sha256",
            "capacity_generation",
            "rollout_generation",
        )
    }
    assert qualification._row_matches_readiness_generation(
        result_row, generation
    )
    assert not qualification._row_matches_readiness_generation(
        {**result_row, "rollout_generation": 8},
        generation,
    )

    drifted = {
        **runtime_intent,
        "readiness_generation": {
            **generation,
            "rollout_generation": 8,
        },
    }
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="exact next rollout",
    ):
        qualification._execution_environment(
            context,
            control_value=control_value,
            intent=drifted,
        )


def test_execution_authority_is_sealed_and_dispatcher_consumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, intent = _context_and_intent(
        tmp_path, write_execution_authority=False
    )
    python = context.harness_prefix / "bin" / "python"
    dispatcher = (
        context.release_worktree / "slurm" / "dispatch_sweeps.py"
    )
    template = (
        context.release_worktree
        / "slurm"
        / "run_dispatch_batch.sbatch.tmpl"
    )
    python.parent.mkdir(parents=True)
    dispatcher.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    dispatcher.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    template.write_bytes(
        qualification.dispatch_sweeps.ARRAY_TEMPLATE.read_bytes()
    )
    python.chmod(0o555)
    dispatcher.chmod(0o444)
    template.chmod(0o444)
    environment = {
        key: "x"
        for key in qualification.dispatch_sweeps.PRODUCTION_ENVIRONMENT_KEYS
    }
    readiness = intent["readiness_generation"]
    assert isinstance(readiness, dict)
    environment.update(
        {
            "ASYS_RELEASE_GIT_COMMIT": COMMIT,
            "ASYS_PROTECTED_CAPACITY_MARKER": str(
                context.protected_capacity_contract.path
            ),
            "ASYS_PROTECTED_CAPACITY_MARKER_SHA256": (
                context.protected_capacity_contract.sha256
            ),
            "ASYS_PROTECTED_CAPACITY_MARKER_ID": (
                context.protected_capacity_contract.marker_id
            ),
            "ASYS_MODEL_CONTRACT_SHA256": "3" * 64,
            "ASYS_FLEET_CONTRACT_SHA256": readiness[
                "fleet_contract_sha256"
            ],
            "ASYS_FLEET_CONTRACT_PATH": str(
                (tmp_path / "fleet.json").resolve()
            ),
            "ASYS_RELEASE_FLEET_CONTRACT_SHA256": readiness[
                "release_fleet_contract_sha256"
            ],
            "ASYS_CAPACITY_GENERATION": str(
                readiness["capacity_generation"]
            ),
            "ASYS_HARNESS_ENVIRONMENT_SHA256": "4" * 64,
            "ASYS_SERVING_ENVIRONMENT_SHA256": "5" * 64,
            "ASYS_ROLLOUT_GENERATION": str(
                readiness["rollout_generation"]
            ),
            "ASYS_IMMUTABLE_PINS_SHA256": "6" * 64,
            "ASYS_RUNTIME_ATTESTATION": str(
                (tmp_path / "attestation.json").resolve()
            ),
            "ASYS_RUNTIME_ATTESTATION_SHA256": "7" * 64,
            "ASYS_RUNTIME_INTEGRITY_LEASE": str(
                (tmp_path / "lease.json").resolve()
            ),
            "ASYS_ARTIFACT_POLICY_SHA256": "8" * 64,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    verified: list[dict[str, object]] = []
    monkeypatch.setattr(
        qualification.dispatch_sweeps.runtime_integrity,
        "verify_generation_lease",
        lambda **kwargs: verified.append(kwargs) or {},
    )

    authority = qualification.create_or_load_execution_authority(
        context,
        intent=intent,
        control_value={},
        environment=environment,
    )
    path = context.qualification_root / qualification.EXECUTION_AUTHORITY_NAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert authority.runtime_environment == environment
    assert authority.payload["intent_id"] == intent["intent_id"]
    assert authority.execution["python"] == str(python.resolve())
    assert verified[-1]["generation"] == readiness["rollout_generation"]
    command = qualification.dispatcher_command(
        context,
        client_partition="ou_bcs_normal",
        client_qos="normal",
    )
    assert command[
        command.index("--qualification-execution-authority") + 1
    ] == str(path)


def test_terminal_failure_names_additive_bottleneck_and_requires_fresh_intent(
    tmp_path: Path,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    ledger = qualification.dispatch_sweeps._empty_ledger()
    ledger["qualification_profile_pressure"] = {
        "eight": {
            "server_pool_root": str(context.server_pool_root),
            "serving_profile": "8B",
            "eligible_cells": 10,
            "backlog_fanout_work": 100,
            "live_replicas": 2,
            "backlog_work_per_replica": 50.0,
            "observed_poll": 2,
        },
        "long": {
            "server_pool_root": str(context.server_pool_root),
            "serving_profile": "32B-long",
            "eligible_cells": 8,
            "backlog_fanout_work": 120,
            "live_replicas": 2,
            "backlog_work_per_replica": 60.0,
            "observed_poll": 3,
        },
    }
    context.dispatcher_state.mkdir(parents=True)
    qualification.dispatch_sweeps._atomic_write_json(
        context.dispatcher_state / "ledger.json", ledger
    )

    scaling = qualification._scaling_requirement(context)
    assert scaling == {
        "serving_profile": "32B-long",
        "server_pool_root": str(context.server_pool_root),
        "backlog_fanout_work": 120,
        "live_replicas": 2,
        "backlog_work_per_replica": 60.0,
        "additional_replicas": 1,
        "tensor_parallel_size": 2,
        "additional_gpus": 2,
        "requirement": "add one TP=2 replica pair (2 GPUs)",
        "capacity_mutated": False,
    }
    failure = qualification._publish_terminal_failure(
        context,
        intent=intent,
        reason="qualification throughput is below threshold",
    )
    assert failure["scheduler_capacity_mutated"] is False
    message = qualification._terminal_failure_message(context)
    assert message is not None
    assert "32B-long" in message
    assert "add one TP=2 replica pair (2 GPUs)" in message
    assert "fresh qualification namespace and intent" in message

    intent_path = context.qualification_root / qualification.INTENT_NAME
    original = intent_path.read_bytes()
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="generation changed after qualification intent",
    ):
        qualification.create_or_load_intent(
            context,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            readiness_generation={
                **intent["readiness_generation"],  # type: ignore[arg-type]
                "rollout_generation": 2,
            },
            control_guard=intent["control_guard"],  # type: ignore[arg-type]
            now=1_000.0,
        )
    assert intent_path.read_bytes() == original


def test_failed_attempt_is_preserved_and_exact_additive_generation_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context1, intent1 = _context_and_intent(
        tmp_path, write_execution_authority=False
    )
    base = qualification.load_qualification_context(
        context1.chain_manifest, verify_chain=False
    )
    old_fleet = (tmp_path / "old-fleet.json").resolve()
    new_fleet = (tmp_path / "new-fleet.json").resolve()
    replica0 = {
        "replica_index": 0,
        "replica_id": "replica-0",
        "scheduler_job_name": "server-0",
    }
    replica1 = {
        "replica_index": 1,
        "replica_id": "replica-1",
        "scheduler_job_name": "server-1",
    }
    common = {
        "schema_version": 1,
        "fleet_id": "schema5-v1",
        "release_id": renderer.RELEASE_ID,
        "model_contract_sha256": "3" * 64,
        "offline_environment": {"HF_HUB_OFFLINE": "1"},
        "server_pool": {"pool_id": "schema5-v1"},
    }
    profile = {
        "serving_profile": "8B",
        "model_size": "8B",
    }
    old_payload = {
        **common,
        "logical_replica_count": 1,
        "allocated_gpu_count": 1,
        "profiles": [{**profile, "replicas": [replica0]}],
    }
    new_payload = {
        **common,
        "logical_replica_count": 2,
        "allocated_gpu_count": 2,
        "profiles": [{**profile, "replicas": [replica0, replica1]}],
    }
    old_fleet.write_bytes(qualification._canonical_bytes(old_payload))
    new_fleet.write_bytes(qualification._canonical_bytes(new_payload))
    old_fleet.chmod(0o444)
    new_fleet.chmod(0o444)
    old_sha = qualification._sha256_file(old_fleet)
    new_sha = qualification._sha256_file(new_fleet)
    readiness1 = dict(intent1["readiness_generation"])
    readiness1["fleet_contract_sha256"] = old_sha
    # The attempt pointer is the generation authority, so its fleet digest must
    # already name the old contract used by the failed task authority.
    pointer1_path = context1.attempt_pointer_path
    assert pointer1_path is not None
    pointer1 = dict(context1.attempt_pointer)
    pointer1_identity = dict(pointer1)
    pointer1_identity.pop("pointer_id")
    pointer1_identity["readiness_generation"] = readiness1
    pointer1 = qualification._with_identity(
        pointer1_identity, "pointer_id"
    )
    pointer1_path.chmod(0o644)
    pointer1_path.write_bytes(qualification._canonical_bytes(pointer1))
    pointer1_path.chmod(0o444)
    context1 = qualification._attempt_context_from_pointer(
        base, path=pointer1_path, pointer=pointer1
    )
    # Repair the current cache to the test's exact old-fleet pointer bytes.
    current_path = base.qualification_base / qualification.CURRENT_ATTEMPT_NAME
    current_path.unlink()
    qualification._replace_current_attempt(
        base,
        path=pointer1_path,
        pointer=pointer1,
        known=[(pointer1_path, pointer1)],
    )
    intent1_identity = qualification._intent_identity(
        context1,
        client_partition="ou_bcs_normal",
        client_qos="normal",
        readiness_generation=readiness1,
        control_guard=intent1["control_guard"],
        created_timestamp=float(intent1["created_timestamp"]),
    )
    intent1 = qualification._with_identity(intent1_identity, "intent_id")
    intent1_path = context1.qualification_root / qualification.INTENT_NAME
    intent1_path.chmod(0o644)
    intent1_path.write_bytes(qualification._canonical_bytes(intent1))
    intent1_path.chmod(0o444)
    authority1 = qualification._with_identity(
        {
            "schema_version": 1,
            "protocol": qualification.EXECUTION_AUTHORITY_PROTOCOL,
            "intent_id": intent1["intent_id"],
            "release_git_commit": COMMIT,
            "runtime_environment": {
                "ASYS_FLEET_CONTRACT_PATH": str(old_fleet),
                "ASYS_FLEET_CONTRACT_SHA256": old_sha,
            },
        },
        "authority_id",
    )
    qualification._write_once(
        context1.qualification_root
        / qualification.EXECUTION_AUTHORITY_NAME,
        authority1,
        description="old attempt authority",
    )
    ledger = qualification.dispatch_sweeps._empty_ledger()
    ledger["qualification_profile_pressure"] = {
        "eight": {
            "server_pool_root": str(context1.server_pool_root),
            "serving_profile": "8B",
            "eligible_cells": 10,
            "backlog_fanout_work": 100,
            "live_replicas": 1,
            "backlog_work_per_replica": 100.0,
            "observed_poll": 1,
        }
    }
    context1.dispatcher_state.mkdir(parents=True)
    qualification.dispatch_sweeps._atomic_write_json(
        context1.dispatcher_state / "ledger.json", ledger
    )
    failure = qualification._publish_terminal_failure(
        context1,
        intent=intent1,
        reason="qualification throughput is below threshold",
    )
    old_pointer_bytes = pointer1_path.read_bytes()
    old_failure_bytes = (
        context1.qualification_root / qualification.FAILURE_NAME
    ).read_bytes()
    qualification._assert_tree_read_only(
        context1.qualification_root,
        description="failed attempt",
    )
    qualification._assert_tree_read_only(
        context1.run_root,
        description="failed run",
    )
    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="fresh additive fleet/readiness generation",
    ):
        qualification.create_or_load_attempt_context(
            base,
            control_value={"immutable": {"git_commit": COMMIT}},
            readiness_generation=readiness1,
            now=1_900.0,
        )
    assert len(
        qualification._load_attempt_pointers(base)
    ) == 1

    readiness2 = {
        **readiness1,
        "catalog_id": "d" * 64,
        "marker_path": str((tmp_path / "TRUSTED_GENERATION_2.json").resolve()),
        "marker_sha256": "e" * 64,
        "inventory_sha256": "f" * 64,
        "catalog_payload_sha256": "0" * 64,
        "fleet_contract_sha256": new_sha,
        "capacity_generation": 2,
        "rollout_generation": 2,
    }
    new_replica = SimpleNamespace(replica_index=1, gpus_per_replica=1)
    monkeypatch.setattr(
        qualification.control,
        "effective_fleet_contract_binding",
        lambda *_args, **_kwargs: {
            "capacity_generation": 2,
            "path": str(new_fleet),
            "sha256": new_sha,
            "fleet_id": "schema5-v1",
            "logical_replicas": 2,
            "allocated_gpus": 2,
            "profile_replicas": {"8B": 2},
            "is_capacity_overlay": True,
        },
    )
    monkeypatch.setattr(
        qualification.control,
        "load_effective_fleet_contract",
        lambda *_args, **_kwargs: SimpleNamespace(
            by_profile={
                "8B": (
                    SimpleNamespace(
                        replica_index=0, gpus_per_replica=1
                    ),
                    new_replica,
                )
            }
        ),
    )
    failed_comment = (
        f"asys:s5-recovery-v1.2-r2:{base.chain_id}:"
        "g0000:throughput_qualification"
    )
    receipt = qualification._with_identity(
        {
            "schema_version": 1,
            "protocol": "schema5-v1.2-r2-recovery-chain-submission",
            "passed": True,
            "chain_id": base.chain_id,
            "manifest": str(base.chain_manifest),
            "manifest_sha256": base.chain_manifest_sha256,
            "jobs": [
                {
                    "name": "throughput_qualification",
                    "job_id": "12345",
                    "comment": failed_comment,
                }
            ],
        },
        "receipt_id",
    )
    receipt_path = _sealed(tmp_path / "SUBMISSION.json", receipt)
    transition_control = {"immutable": {"git_commit": COMMIT}}
    monkeypatch.setattr(
        qualification,
        "load_paused_control",
        lambda _context: (transition_control, _guard(rollout_generation=1)),
    )
    monkeypatch.setattr(
        qualification,
        "load_readiness_generation",
        lambda _context, _control: readiness2,
    )
    transition_report = (
        qualification.publish_capacity_transition_authority(
            context1.chain_manifest,
            submission_receipt=receipt_path,
            failed_job_id="12345",
            failed_comment=failed_comment,
            apply=True,
            verify_chain=False,
            now=1_950.0,
        )
    )
    transition = transition_report["transition"]
    assert transition_report["status"] == "published"
    assert transition["failure"]["failure_id"] == failure["failure_id"]
    assert transition["additive_transition"][
        "to_fleet_contract_sha256"
    ] == new_sha
    assert transition["failed_stage"] == {
        "name": "throughput_qualification",
        "job_id": "12345",
        "comment": failed_comment,
    }
    context2 = qualification.create_or_load_attempt_context(
        base,
        control_value={"immutable": {"git_commit": COMMIT}},
        readiness_generation=readiness2,
        now=2_000.0,
    )
    assert context2.qualification_root != context1.qualification_root
    assert context2.run_root != context1.run_root
    assert context2.attempt_pointer["attempt_ordinal"] == 2
    assert context2.attempt_pointer["additive_retry"] == {
        "previous_failure_id": failure["failure_id"],
        "serving_profile": "8B",
        "additional_replicas": 1,
        "tensor_parallel_size": 1,
        "from_capacity_generation": 1,
        "to_capacity_generation": 2,
        "from_rollout_generation": 1,
        "to_rollout_generation": 2,
        "from_fleet_contract_sha256": old_sha,
        "to_fleet_contract_sha256": new_sha,
        "validation": (
            "schema5_control._assert_additive_capacity_contract+"
            "exact-required-profile-delta"
        ),
    }
    assert pointer1_path.read_bytes() == old_pointer_bytes
    assert (
        context1.qualification_root / qualification.FAILURE_NAME
    ).read_bytes() == old_failure_bytes

    guard2 = _guard(rollout_generation=1)
    intent2 = qualification.create_or_load_intent(
        context2,
        client_partition="ou_bcs_normal",
        client_qos="normal",
        readiness_generation=readiness2,
        control_guard=guard2,
        now=2_100.0,
    )
    qualification._write_once(
        context2.qualification_root / qualification.PLAN_NAME,
        qualification.build_load_plan(),
        description="second attempt plan",
    )
    authority2 = qualification._with_identity(
        {
            "schema_version": 1,
            "protocol": qualification.EXECUTION_AUTHORITY_PROTOCOL,
            "intent_id": intent2["intent_id"],
        },
        "authority_id",
    )
    qualification._write_once(
        context2.qualification_root
        / qualification.EXECUTION_AUTHORITY_NAME,
        authority2,
        description="second attempt authority",
    )
    context2.run_root.mkdir(parents=True)
    _passing_observations(context2, intent2)
    marker = qualification.publish_completion(context2, intent=intent2)
    verified = qualification.verify_completed_qualification(
        context2.chain_manifest,
        verify_chain=False,
        verify_renderer=False,
    )
    assert verified["attempt_id"] == context2.attempt_pointer["attempt_id"]
    assert marker["attempt"]["rollout_generation"] == 2
    assert marker["attempt"]["capacity_generation"] == 2
    assert (
        qualification._read_json(
            base.qualification_base / qualification.MARKER_NAME,
            description="root success marker",
            sealed=True,
        )
        == marker
    )


@pytest.mark.parametrize(
    "failure_description",
    ["semantic observation 0", "qualification observation 0"],
)
def test_observation_transaction_resumes_exact_crash_preimage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_description: str,
) -> None:
    context, intent = _context_and_intent(tmp_path)
    original = qualification._write_once
    crashed = False

    def interrupted_write(
        path: Path,
        payload: object,
        *,
        description: str,
        mode: int = 0o444,
    ) -> None:
        nonlocal crashed
        if description == failure_description and not crashed:
            crashed = True
            raise RuntimeError("injected crash")
        original(
            path,
            payload,  # type: ignore[arg-type]
            description=description,
            mode=mode,
        )

    monkeypatch.setattr(qualification, "_write_once", interrupted_write)
    with pytest.raises(RuntimeError, match="injected crash"):
        _record(
            context,
            intent,
            sequence=0,
            timestamp=1_000.0,
            ceiling=24,
            active=0,
            useful=0,
        )
    monkeypatch.setattr(qualification, "_write_once", original)

    with pytest.raises(
        qualification.ThroughputQualificationError,
        match="transaction is incomplete",
    ):
        qualification.load_observations(
            context.qualification_root, intent=intent
        )
    recovered = qualification.load_observations(
        context.qualification_root,
        intent=intent,
        recover_transactions=True,
    )
    assert len(recovered) == 1
    assert recovered[0]["scheduler"]["active_qualification_cells"] == 0
    assert recovered[0]["semantic"]["useful_qids"] == 0


@pytest.mark.parametrize(
    ("ceiling", "active", "useful", "expected"),
    [
        (24, 0, 0, 24),
        (24, 18, 100, 6),
        (96, 71, 1_000, 24),
        (96, 90, 1_100, 6),
        (384, 383, 15_000, 1),
        (384, 0, qualification.TOTAL_QIDS, 0),
    ],
)
def test_stage_fill_is_exact_when_tasks_finish_between_polls(
    ceiling: int,
    active: int,
    useful: int,
    expected: int,
) -> None:
    assert (
        qualification._stage_dispatch_batch(
            ceiling=ceiling,
            active=active,
            useful_qids=useful,
        )
        == expected
    )


def test_accepted_fast_array_is_counted_through_scheduler_visibility_grace() -> None:
    ledger = {
        "jobs": {
            "12345": {
                "state": "submitted",
                "submitted_at": 1_000.0,
                "task_count": 24,
                "tasks": [
                    {"run_id": qualification.QUALIFICATION_RUN_ID}
                    for _ in range(24)
                ],
            }
        }
    }
    active, job_ids, foreign = qualification._active_task_count(
        scheduler_jobs=[],
        ledger=ledger,
        captured_timestamp=1_001.0,
    )
    assert active == 24
    assert job_ids == ["12345"]
    assert foreign == []

    expired, _, _ = qualification._active_task_count(
        scheduler_jobs=[],
        ledger=ledger,
        captured_timestamp=1_300.0,
    )
    assert expired == 0

    stray = SimpleNamespace(
        active=True,
        job_id="99999_0",
        job_name="asys-dispatch-foreign",
        comment="",
        command="/tmp/foreign.sbatch",
    )
    _, _, foreign = qualification._active_task_count(
        scheduler_jobs=[stray],
        ledger={"jobs": {}},
        captured_timestamp=1_001.0,
    )
    assert foreign == ["unmapped-active-cell-job:99999_0"]


def test_execute_reconciles_restart_before_next_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, intent = _context_and_intent(tmp_path)
    _record(
        context,
        intent,
        sequence=0,
        timestamp=1_000.0,
        ceiling=24,
        active=0,
        useful=0,
    )
    control_value = {
        "rollout_generation": 0,
        "immutable": {"model_contract_path": str(tmp_path / "models.json")},
    }
    events: list[str] = []

    monkeypatch.setattr(
        qualification,
        "load_qualification_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        qualification,
        "load_paused_control",
        lambda _context: (control_value, intent["control_guard"]),
    )
    monkeypatch.setattr(
        qualification,
        "load_readiness_generation",
        lambda *_args, **_kwargs: intent["readiness_generation"],
    )
    monkeypatch.setattr(
        qualification,
        "create_or_load_attempt_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        qualification,
        "initialize_qualification_run",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        qualification,
        "create_or_load_execution_authority",
        lambda *_args, **_kwargs: {},
    )

    def scheduler_scan(
        _context: qualification.QualificationContext,
        *,
        intent: dict[str, object],
        sequence: int,
        captured_timestamp: float,
        ceiling: int,
        scheduler_reader: object,
    ) -> dict[str, object]:
        del _context, scheduler_reader
        events.append(f"capture:{ceiling}")
        execution_authority = (
            qualification._execution_authority_evidence_binding(intent)
        )
        return qualification.make_scheduler_evidence(
            intent_id=str(intent["intent_id"]),
            sequence=sequence,
            captured_timestamp=captured_timestamp,
            ceiling=ceiling,
            jobs=[_job(sequence)],
            qualification_job_ids=[_job(sequence)["job_id"]],
            active_qualification_cells=24,
            dispatcher_ledger_sha256="9" * 64,
            production_control_guard_sha256=str(
                intent["control_guard"]["guard_sha256"]  # type: ignore[index]
            ),
            client_partition=str(intent["client_partition"]),
            client_qos=str(intent["client_qos"]),
            protected_capacity_marker_id=str(
                intent["client_placement"]["protected_capacity_marker_id"]  # type: ignore[index]
            ),
            protected_capacity_marker_sha256=str(
                intent["client_placement"]["protected_capacity_marker_sha256"]  # type: ignore[index]
            ),
            readiness_rollout_generation=1,
            trusted_generation_catalog_id=str(
                intent["readiness_generation"]["catalog_id"]  # type: ignore[index]
            ),
            qualification_execution_authority_id=(
                execution_authority["authority_id"]
            ),
            qualification_execution_authority_sha256=(
                execution_authority["sha256"]
            ),
        )

    monkeypatch.setattr(qualification, "_scheduler_scan", scheduler_scan)

    def dispatch_once(
        _context: qualification.QualificationContext,
        *,
        intent: dict[str, object],
        control_value: dict[str, object],
        max_batch: int,
        runner: object,
    ) -> dict[str, object]:
        del _context, intent, control_value, runner
        events.append(f"dispatch:{max_batch}")
        if events.count(f"dispatch:{max_batch}") == 2:
            raise RuntimeError("second immediate dispatch")
        return {"selected": [{"run_id": qualification.QUALIFICATION_RUN_ID}]}

    monkeypatch.setattr(
        qualification, "_run_dispatcher_once", dispatch_once
    )

    def semantic_reader(**kwargs: object) -> dict[str, object]:
        return qualification.make_semantic_evidence(
            intent_id=str(intent["intent_id"]),
            sequence=int(kwargs["sequence"]),
            captured_timestamp=float(kwargs["captured_timestamp"]),
            manifest_sha256=MANIFEST_SHA256,
            states={"active": 24, "missing": qualification.CELL_COUNT - 24},
            validated_qids=0,
            useful_qids=0,
            strata_progress=_progress(0),
            artifact_schema_counts={},
        )

    def unexpected_sleep(_seconds: float) -> None:
        raise AssertionError("ramp slept before exact stage fill")

    clock_values = iter(
        (
            1_000.25,
            1_000.5,
            1_001.0,
            1_002.0,
            1_002.5,
            1_003.0,
        )
    )
    with pytest.raises(RuntimeError, match="second immediate dispatch"):
        qualification.execute_qualification(
            context.chain_manifest,
            client_partition="ou_bcs_normal",
            client_qos="normal",
            verify_chain=False,
            semantic_reader=semantic_reader,
            clock=lambda: next(clock_values),
            sleeper=unexpected_sleep,
        )

    # The stale baseline is never used for admission: restart first sees the 24 live
    # tasks, closes ceiling 24, then advances under the ceiling-96 stage.
    assert events == [
        "capture:24",
        "dispatch:24",
        "capture:96",
        "dispatch:24",
    ]
