from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from scripts import seal_recovery_evidence as evidence


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _quarantine(tmp_path: Path) -> tuple[Path, Path, Path]:
    tree = tmp_path / "quarantine" / "partial"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested" / "payload.bin").write_bytes(b"payload")
    (tree / "link").symlink_to("nested/payload.bin")
    completion = tmp_path / "quarantine.complete.json"
    _write_json(
        completion,
        {
            "passed": True,
            "destination": str(tree),
            "materialize_job_id": "123",
            "release_id": "release-v1",
        },
    )
    return tree, completion, tmp_path / "evidence"


def test_quarantine_seal_is_marker_last_read_only_and_idempotent(tmp_path):
    tree, completion, root = _quarantine(tmp_path)
    dry = evidence.seal_quarantine(
        tree=tree,
        evidence_root=root,
        release_id="release-v1",
        failed_job_id="123",
        quarantine_completion=completion,
    )
    assert dry["status"] == "dry_run"
    assert not root.exists()

    sealed = evidence.seal_quarantine(
        tree=tree,
        evidence_root=root,
        release_id="release-v1",
        failed_job_id="123",
        quarantine_completion=completion,
        apply=True,
    )
    assert sealed["status"] == "sealed"
    assert sealed["passed"] is True
    assert not evidence._writable_entries(tree)
    marker = root / "partial-job-123.sealed.json"
    inventory = root / "partial-job-123.sealed-inventory.txt"
    assert stat.S_IMODE(marker.stat().st_mode) == 0o444
    assert stat.S_IMODE(inventory.stat().st_mode) == 0o444
    assert sealed["inventory_sha256"] == hashlib.sha256(inventory.read_bytes()).hexdigest()

    again = evidence.seal_quarantine(
        tree=tree,
        evidence_root=root,
        release_id="release-v1",
        failed_job_id="123",
        quarantine_completion=completion,
        apply=True,
    )
    assert again["status"] == "already_sealed"
    assert again["seal_id"] == sealed["seal_id"]


def test_quarantine_seal_detects_content_drift(tmp_path):
    tree, completion, root = _quarantine(tmp_path)
    evidence.seal_quarantine(
        tree=tree,
        evidence_root=root,
        release_id="release-v1",
        failed_job_id="123",
        quarantine_completion=completion,
        apply=True,
    )
    payload = tree / "nested" / "payload.bin"
    os.chmod(payload, 0o644)
    payload.write_bytes(b"changed")
    with pytest.raises(evidence.EvidenceError, match="verification failed"):
        evidence.seal_quarantine(
            tree=tree,
            evidence_root=root,
            release_id="release-v1",
            failed_job_id="123",
            quarantine_completion=completion,
            apply=True,
        )


def test_record_failure_binds_exact_terminal_scheduler_and_artifacts(
    tmp_path, monkeypatch
):
    paths = {}
    for name in (
        "manifest",
        "receipt",
        "snapshot",
        "attestation",
        "seal",
        "incident",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    log_root = tmp_path / "logs"
    log_root.mkdir()
    for name in (
        "source_checkout_18555909.out",
        "maintenance_preflight_18555910.out",
        "pre_repair_snapshot_18555911.out",
        "pre_repair_snapshot_verify_18555912.out",
        "release_materialize_18555913.out",
    ):
        (log_root / name).write_text(name, encoding="utf-8")
    paths["log"] = log_root / "release_materialize_18555913.out"
    monkeypatch.setattr(
        evidence,
        "_scheduler_states",
        lambda _ids: [
            {
                "job_id": "1",
                "job_name": "asys-s5v11r1-checkout",
                "state": "COMPLETED",
                "elapsed": "1",
                "start": "s",
                "end": "e",
                "exit_code": "0:0",
                "node_list": "n",
            },
            {
                "job_id": "2",
                "job_name": "asys-s5v11r1-materialize",
                "state": "FAILED",
                "elapsed": "1",
                "start": "s",
                "end": "e",
                "exit_code": "2:0",
                "node_list": "n",
            },
        ],
    )
    output = tmp_path / "FAILED.json"
    report = evidence.record_failure(
        output=output,
        chain_manifest=paths["manifest"],
        submission_receipt=paths["receipt"],
        snapshot_marker=paths["snapshot"],
        snapshot_attestation=paths["attestation"],
        materialization_log=paths["log"],
        quarantine_seal=paths["seal"],
        incident=paths["incident"],
        job_ids=["1", "2"],
        apply=True,
    )
    assert report["classification"] == "requires_superseding_release"
    assert report["retry_same_generation"] is False
    assert len(report["scheduler_logs"]) == 5
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert output.with_suffix(".json.sha256").is_file()


def test_conda_reconciliation_incident_binds_absent_and_recovered_record(tmp_path):
    harness = tmp_path / "harness"
    serving = tmp_path / "serving"
    for prefix in (harness, serving):
        metadata = (
            prefix
            / "lib"
            / "python3.11"
            / "site-packages"
            / "setuptools-81.0.0.dist-info"
            / "METADATA"
        )
        metadata.parent.mkdir(parents=True)
        metadata.write_text("Name: setuptools\nVersion: 81.0.0\n", encoding="utf-8")
        (prefix / "conda-meta").mkdir()
    record = {
        "name": "setuptools",
        "version": "82.0.1",
        "build": "pyh332efcf_0",
        "sha256": (
            "82088a6e4daa33329a30bc26dc19a98c7c1d3f05c0f73ce9845d4eab4924e9e1"
        ),
    }
    recovered = tmp_path / "recovered.json"
    _write_json(recovered, record)
    _write_json(serving / "conda-meta" / recovered.name.replace("recovered", "setuptools-82.0.1-pyh332efcf_0"), record)
    log = tmp_path / "failure.log"
    log.write_text("pip distribution view changed", encoding="utf-8")
    output = tmp_path / "incident.json"

    result = evidence.record_conda_reconciliation_incident(
        output=output,
        harness_prefix=harness,
        serving_prefix=serving,
        recovered_harness_record=recovered,
        failed_materialization_log=log,
        observed_at="2026-07-23T11:43:55-04:00",
        apply=True,
    )
    assert result["runtime_owner"]["version"] == "81.0.0"
    assert result["harness_stale_conda_record_present"] is False
    assert result["serving_stale_conda_record_present"] is True
    assert output.with_suffix(".json.sha256").is_file()

    serving_record = (
        serving / "conda-meta" / "setuptools-82.0.1-pyh332efcf_0.json"
    )
    serving_record.unlink()
    absent_output = tmp_path / "incident-both-absent.json"
    absent = evidence.record_conda_reconciliation_incident(
        output=absent_output,
        harness_prefix=harness,
        serving_prefix=serving,
        recovered_harness_record=recovered,
        failed_materialization_log=log,
        observed_at="2026-07-23T11:43:55-04:00,2026-07-23T11:47:14-04:00",
        apply=True,
    )
    assert absent["serving_stale_conda_record_present"] is False


def _partial_canary_failure(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    tree = tmp_path / "partial-canary"
    transaction = tree / "transaction"
    transaction.mkdir(parents=True)
    comment = (
        "asys-s5-fleet:pool=schema5-v12-slurm-canary-0123456789ab;"
        "profile=canary-cpu;replica=schema5-v12-canary-0123456789ab;"
        "generation=1;intent=0123456789abcdef0123456789abcdef;"
        f"fleet={'a' * 64}"
    )
    _write_json(
        tree / "COMPOSITE_CANARY_INTENT.json",
        {
            "schema_version": 4,
            "kind": "schema5_slurm_fleet_composite_canary_intent",
            "created_at": 1_753_500_000.0,
            "canary_root": str(tree),
            "code_identity": {},
        },
    )
    _write_json(
        transaction / "CANARY_INTENT.json",
        {
            "job_name": "asys-s5-serve-canary-0123456789ab",
            "scheduler_comment": comment,
        },
    )
    _write_json(
        transaction / "attempt.json",
        {
            "job_id": "18889366",
            "job_name": "asys-s5-serve-canary-0123456789ab",
            "scheduler_comment": comment,
        },
    )
    mutation = tmp_path / "ZERO_RESULT_MUTATION.json"
    _write_json(
        mutation,
        {
            "protocol": "schema5-v1.2-r2-r1-zero-result-mutation-receipt-v1",
            "passed": True,
            "legacy_result_mutation_count": 0,
            "schema5_result_mutation_count": 0,
        },
    )
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    return tree, checkout, mutation, comment


class _TerminalCanaryScheduler:
    def __init__(self, comment: str, *, active: bool = False):
        self.comment = comment
        self.active = active
        self.commands: list[list[str]] = []

    def __call__(self, argv, **_kwargs):
        self.commands.append(list(argv))
        if argv[0] == "squeue":
            stdout = (
                "18889366|"
                f"{self.comment}|asys-s5-serve-canary-0123456789ab|RUNNING|"
                "mit_normal|2026-07-26T00:00:00|N/A|python worker.py\n"
                if self.active
                else ""
            )
        elif argv[0] == "sacct":
            stdout = (
                "18889366||asys-s5-serve-canary-0123456789ab|CANCELLED|"
                "mit_normal|Unknown|Unknown|0:15|"
                f"sbatch --comment={self.comment} immutable.sbatch\n"
            )
        else:
            raise AssertionError(f"unexpected scheduler mutation: {argv}")
        return __import__("subprocess").CompletedProcess(argv, 0, stdout, "")


class _SimulatedPowerLoss(BaseException):
    pass


def test_canary_failure_seal_is_external_marker_first_read_only_and_idempotent(
    tmp_path, monkeypatch
):
    tree, checkout, mutation, comment = _partial_canary_failure(tmp_path)
    release = {
        "release_tag": "sweep-recovery-schema5-v1.2-r2",
        "release_git_commit": "1" * 40,
        "code_hash": "2" * 64,
    }
    monkeypatch.setattr(
        evidence,
        "_validate_canary_release_identity",
        lambda **_kwargs: release,
    )
    scheduler = _TerminalCanaryScheduler(comment)
    root = tmp_path / "canary-failure-evidence"
    dry = evidence.seal_canary_failure(
        tree=tree,
        evidence_root=root,
        release_checkout=checkout,
        scheduler_user="test-user",
        error_classification="dependency_parameters_contract",
        error_summary="cluster lacks the required dependency policy",
        mutation_evidence=[mutation],
        scheduler_runner=scheduler,
    )
    assert dry["status"] == "dry_run"
    assert not root.exists()
    assert scheduler.commands == []

    sealed = evidence.seal_canary_failure(
        tree=tree,
        evidence_root=root,
        release_checkout=checkout,
        scheduler_user="test-user",
        error_classification="dependency_parameters_contract",
        error_summary="cluster lacks the required dependency policy",
        mutation_evidence=[mutation],
        apply=True,
        scheduler_runner=scheduler,
    )
    assert sealed["status"] == "sealed"
    assert sealed["known_scheduler_identity"]["job_ids"] == ["18889366"]
    assert sealed["mutation_claim"] == (
        "bound_to_preexisting_evidence_not_inferred_from_absence"
    )
    assert not evidence._writable_entries(tree)
    assert not stat.S_IMODE(root.stat().st_mode) & 0o222
    names = [path.name for path in root.iterdir()]
    assert names[-1] == "CANARY_FAILURE_SEALED.json"
    assert all(not stat.S_IMODE(path.stat().st_mode) & 0o222 for path in root.iterdir())
    assert all(command[0] in {"squeue", "sacct"} for command in scheduler.commands)

    again = evidence.seal_canary_failure(
        tree=tree,
        evidence_root=root,
        release_checkout=checkout,
        scheduler_user="test-user",
        error_classification="dependency_parameters_contract",
        error_summary="cluster lacks the required dependency policy",
        mutation_evidence=[mutation],
        apply=True,
        scheduler_runner=scheduler,
    )
    assert again["status"] == "already_sealed"
    assert again["seal_id"] == sealed["seal_id"]


@pytest.mark.parametrize(
    ("interrupted_artifact", "pending_mode", "expected_prefix_length"),
    [
        ("intent", 0o600, 0),
        ("scheduler_pre", 0o444, 2),
        ("marker", 0o444, 5),
    ],
)
def test_canary_failure_seal_recovers_sibling_staging_after_hard_crash(
    tmp_path,
    monkeypatch,
    interrupted_artifact,
    pending_mode,
    expected_prefix_length,
):
    tree, checkout, mutation, comment = _partial_canary_failure(tmp_path)
    monkeypatch.setattr(
        evidence,
        "_validate_canary_release_identity",
        lambda **_kwargs: {
            "release_tag": "sweep-recovery-schema5-v1.2-r2",
            "release_git_commit": "1" * 40,
            "code_hash": "2" * 64,
        },
    )
    scheduler = _TerminalCanaryScheduler(comment)
    root = tmp_path / "canary-failure-evidence"
    original_atomic = evidence._canary_atomic_bytes
    interrupted = False

    def crash_after_partial_sibling_write(
        path,
        payload,
        *,
        evidence_root,
        artifacts,
        mode=0o444,
    ):
        nonlocal interrupted
        if not interrupted and path == artifacts[interrupted_artifact]:
            interrupted = True
            staging = evidence._canary_failure_staging_root(evidence_root)
            staging.mkdir(mode=0o700)
            pending = evidence._canary_failure_pending_path(
                evidence_root, path
            )
            pending.write_bytes(payload[: max(1, len(payload) // 2)])
            pending.chmod(pending_mode)
            raise _SimulatedPowerLoss
        return original_atomic(
            path,
            payload,
            evidence_root=evidence_root,
            artifacts=artifacts,
            mode=mode,
        )

    monkeypatch.setattr(
        evidence, "_canary_atomic_bytes", crash_after_partial_sibling_write
    )
    with pytest.raises(_SimulatedPowerLoss):
        evidence.seal_canary_failure(
            tree=tree,
            evidence_root=root,
            release_checkout=checkout,
            scheduler_user="test-user",
            error_classification="dependency_parameters_contract",
            error_summary="deterministic canary failure",
            mutation_evidence=[mutation],
            apply=True,
            scheduler_runner=scheduler,
        )
    artifacts = evidence._canary_failure_artifact_paths(root)
    assert interrupted
    assert evidence._canary_failure_staging_root(root).is_dir()
    assert (
        evidence._validate_canary_evidence_file_set(
            root,
            artifacts,
            expected_prefix_length=expected_prefix_length,
            require_directory_read_only=False,
        )
        == expected_prefix_length
    )

    monkeypatch.setattr(evidence, "_canary_atomic_bytes", original_atomic)
    sealed = evidence.seal_canary_failure(
        tree=tree,
        evidence_root=root,
        release_checkout=checkout,
        scheduler_user="test-user",
        error_classification="dependency_parameters_contract",
        error_summary="deterministic canary failure",
        mutation_evidence=[mutation],
        apply=True,
        scheduler_runner=scheduler,
    )
    assert sealed["status"] == "sealed"
    assert not evidence._canary_failure_staging_root(root).exists()
    assert {
        candidate.name for candidate in root.iterdir()
    } == {candidate.name for candidate in artifacts.values()}
    assert (
        evidence._validate_canary_evidence_file_set(
            root,
            artifacts,
            expected_prefix_length=6,
            require_directory_read_only=True,
        )
        == 6
    )

    again = evidence.seal_canary_failure(
        tree=tree,
        evidence_root=root,
        release_checkout=checkout,
        scheduler_user="test-user",
        error_classification="dependency_parameters_contract",
        error_summary="deterministic canary failure",
        mutation_evidence=[mutation],
        apply=True,
        scheduler_runner=scheduler,
    )
    assert again["status"] == "already_sealed"
    assert again["seal_id"] == sealed["seal_id"]


@pytest.mark.parametrize(
    ("interrupted_artifact", "expected_prefix_length"),
    [("intent", 1), ("marker", 6)],
)
def test_canary_failure_seal_recovers_crash_after_artifact_rename(
    tmp_path,
    monkeypatch,
    interrupted_artifact,
    expected_prefix_length,
):
    tree, checkout, mutation, comment = _partial_canary_failure(tmp_path)
    monkeypatch.setattr(
        evidence,
        "_validate_canary_release_identity",
        lambda **_kwargs: {
            "release_tag": "sweep-recovery-schema5-v1.2-r2",
            "release_git_commit": "1" * 40,
            "code_hash": "2" * 64,
        },
    )
    scheduler = _TerminalCanaryScheduler(comment)
    root = tmp_path / "canary-failure-evidence"
    original_atomic = evidence._canary_atomic_bytes
    interrupted = False

    def crash_after_published_rename(
        path,
        payload,
        *,
        evidence_root,
        artifacts,
        mode=0o444,
    ):
        nonlocal interrupted
        result = original_atomic(
            path,
            payload,
            evidence_root=evidence_root,
            artifacts=artifacts,
            mode=mode,
        )
        if not interrupted and path == artifacts[interrupted_artifact]:
            interrupted = True
            evidence._canary_failure_staging_root(evidence_root).mkdir(
                mode=0o700
            )
            raise _SimulatedPowerLoss
        return result

    monkeypatch.setattr(
        evidence, "_canary_atomic_bytes", crash_after_published_rename
    )
    with pytest.raises(_SimulatedPowerLoss):
        evidence.seal_canary_failure(
            tree=tree,
            evidence_root=root,
            release_checkout=checkout,
            scheduler_user="test-user",
            error_classification="dependency_parameters_contract",
            error_summary="deterministic canary failure",
            mutation_evidence=[mutation],
            apply=True,
            scheduler_runner=scheduler,
        )
    artifacts = evidence._canary_failure_artifact_paths(root)
    assert interrupted
    assert evidence._canary_failure_staging_root(root).is_dir()
    assert (
        evidence._validate_canary_evidence_file_set(
            root,
            artifacts,
            expected_prefix_length=expected_prefix_length,
            require_directory_read_only=False,
        )
        == expected_prefix_length
    )

    monkeypatch.setattr(evidence, "_canary_atomic_bytes", original_atomic)
    recovered = evidence.seal_canary_failure(
        tree=tree,
        evidence_root=root,
        release_checkout=checkout,
        scheduler_user="test-user",
        error_classification="dependency_parameters_contract",
        error_summary="deterministic canary failure",
        mutation_evidence=[mutation],
        apply=True,
        scheduler_runner=scheduler,
    )
    assert recovered["status"] in {"sealed", "already_sealed"}
    assert not evidence._canary_failure_staging_root(root).exists()
    assert (
        evidence._validate_canary_evidence_file_set(
            root,
            artifacts,
            expected_prefix_length=6,
            require_directory_read_only=True,
        )
        == 6
    )


def test_canary_failure_seal_rejects_foreign_staging_and_evidence_entries(
    tmp_path, monkeypatch
):
    tree, checkout, mutation, comment = _partial_canary_failure(tmp_path)
    monkeypatch.setattr(
        evidence,
        "_validate_canary_release_identity",
        lambda **_kwargs: {"release_git_commit": "1" * 40},
    )
    scheduler = _TerminalCanaryScheduler(comment)
    root = tmp_path / "canary-failure-evidence"
    staging = evidence._canary_failure_staging_root(root)
    staging.mkdir(mode=0o700)
    (staging / "foreign.pending").write_bytes(b"foreign")
    with pytest.raises(evidence.EvidenceError, match="foreign canary failure staging"):
        evidence.seal_canary_failure(
            tree=tree,
            evidence_root=root,
            release_checkout=checkout,
            scheduler_user="test-user",
            error_classification="dependency_parameters_contract",
            error_summary="deterministic canary failure",
            mutation_evidence=[mutation],
            apply=True,
            scheduler_runner=scheduler,
        )
    assert evidence._writable_entries(tree)
    assert scheduler.commands == []

    (staging / "foreign.pending").unlink()
    staging.rmdir()
    root.mkdir(mode=0o750)
    root.chmod(0o750)
    (root / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="unexpected canary failure"):
        evidence.seal_canary_failure(
            tree=tree,
            evidence_root=root,
            release_checkout=checkout,
            scheduler_user="test-user",
            error_classification="dependency_parameters_contract",
            error_summary="deterministic canary failure",
            mutation_evidence=[mutation],
            apply=True,
            scheduler_runner=scheduler,
        )
    assert evidence._writable_entries(tree)
    assert scheduler.commands == []


def test_canary_failure_seal_refuses_active_job_before_target_mutation(
    tmp_path, monkeypatch
):
    tree, checkout, mutation, comment = _partial_canary_failure(tmp_path)
    monkeypatch.setattr(
        evidence,
        "_validate_canary_release_identity",
        lambda **_kwargs: {"release_git_commit": "1" * 40},
    )
    scheduler = _TerminalCanaryScheduler(comment, active=True)
    with pytest.raises(evidence.EvidenceError, match="active or pending"):
        evidence.seal_canary_failure(
            tree=tree,
            evidence_root=tmp_path / "failure-evidence",
            release_checkout=checkout,
            scheduler_user="test-user",
            error_classification="dependency_parameters_contract",
            error_summary="deterministic canary failure",
            mutation_evidence=[mutation],
            apply=True,
            scheduler_runner=scheduler,
        )
    assert evidence._writable_entries(tree)


def test_canary_failure_scheduler_capture_rejects_non_text_output():
    comment = (
        "asys-s5-fleet:pool=pool;profile=canary-cpu;replica=replica;"
        f"generation=1;intent={'1' * 32};fleet={'2' * 64}"
    )

    def invalid_runner(argv, **_kwargs):
        return __import__("subprocess").CompletedProcess(argv, 0, None, "")

    with pytest.raises(evidence.EvidenceError, match="invalid process result"):
        evidence._capture_canary_scheduler_evidence(
            scheduler_user="test-user",
            since="2026-07-26",
            known={
                "comments": [comment],
                "job_ids": ["18889366"],
                "job_names": ["asys-s5-serve-canary-0123456789ab"],
            },
            phase="preseal",
            runner=invalid_runner,
        )


def test_canary_scheduler_validation_reparses_derived_rows_from_raw_stdout():
    comment = (
        "asys-s5-fleet:pool=pool;profile=canary-cpu;replica=replica;"
        f"generation=1;intent={'1' * 32};fleet={'2' * 64}"
    )
    known = {
        "comments": [comment],
        "job_ids": ["18889366"],
        "job_names": ["asys-s5-serve-canary-0123456789ab"],
    }
    payload = evidence._capture_canary_scheduler_evidence(
        scheduler_user="test-user",
        since="2026-07-26",
        known=known,
        phase="preseal",
        runner=_TerminalCanaryScheduler(comment),
    )
    payload["matching_sacct_rows"][0]["state"] = "COMPLETED"
    identity = dict(payload)
    identity.pop("evidence_id")
    payload["evidence_id"] = evidence._sha256_bytes(
        evidence._canonical_json(identity)
    )

    with pytest.raises(
        evidence.EvidenceError,
        match="derived scheduler rows differ from raw evidence",
    ):
        evidence._validate_scheduler_evidence(
            payload,
            scheduler_user="test-user",
            since="2026-07-26",
            known=known,
            phase="preseal",
        )


def test_canary_failure_seal_rejects_symlink_and_mutation_basis_claims(
    tmp_path, monkeypatch
):
    tree, checkout, mutation, _comment = _partial_canary_failure(tmp_path)
    monkeypatch.setattr(
        evidence,
        "_validate_canary_release_identity",
        lambda **_kwargs: {"release_git_commit": "1" * 40},
    )
    (tree / "unsafe").symlink_to(tree / "transaction")
    with pytest.raises(evidence.EvidenceError, match="symlink"):
        evidence.seal_canary_failure(
            tree=tree,
            evidence_root=tmp_path / "failure-evidence",
            release_checkout=checkout,
            scheduler_user="test-user",
            error_classification="dependency_parameters_contract",
            error_summary="deterministic canary failure",
            mutation_evidence=[mutation],
            scheduler_runner=_TerminalCanaryScheduler("unused"),
        )

    (tree / "unsafe").unlink()
    _write_json(
        mutation,
        {
            "protocol": "handwritten-claim",
            "legacy_result_mutation_count": 0,
            "schema5_result_mutation_count": 0,
        },
    )
    with pytest.raises(evidence.EvidenceError, match="unsupported zero-mutation"):
        evidence.seal_canary_failure(
            tree=tree,
            evidence_root=tmp_path / "failure-evidence",
            release_checkout=checkout,
            scheduler_user="test-user",
            error_classification="dependency_parameters_contract",
            error_summary="deterministic canary failure",
            mutation_evidence=[mutation],
            scheduler_runner=_TerminalCanaryScheduler("unused"),
        )
