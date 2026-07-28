from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts import seal_recovery_evidence as evidence


def test_record_prelaunch_cli_keeps_subcommand_separate_from_probe_argv(
    tmp_path, monkeypatch, capsys
):
    captured: dict[str, object] = {}

    def fake_record(**kwargs):
        captured.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(evidence, "record_prelaunch_attempt", fake_record)
    rc = evidence.main(
        [
            "record-prelaunch-attempt",
            "--output",
            str(tmp_path / "attempt.json"),
            "--classification",
            "unsafe_recorded_broken_internal_symlink",
            "--cwd",
            str(tmp_path),
            "--input-root",
            "tagged-release",
            str(tmp_path),
            "--write-root",
            str(tmp_path),
            "--apply",
            "--command",
            "/sealed/python",
            "-I",
            "/tagged/probe.py",
        ]
    )

    assert rc == 0
    assert captured["command"] == [
        "/sealed/python",
        "-I",
        "/tagged/probe.py",
    ]
    assert captured["apply"] is True
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_scheduler_and_git_environment_rejects_hostile_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = {
        "PATH": "/tmp/attacker-bin",
        "BASH_ENV": "/tmp/attacker-env",
        "LD_PRELOAD": "/tmp/attacker.so",
        "GIT_DIR": "/tmp/attacker-git",
        "GIT_OBJECT_DIRECTORY": "/tmp/attacker-objects",
        "GIT_CONFIG_PARAMETERS": "'core.hooksPath=/tmp/attacker-hooks'",
        "GIT_REPLACE_REF_BASE": "refs/attacker",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": "/tmp/attacker-hooks",
        "SBATCH_PARTITION": "attacker",
        "SQUEUE_FORMAT": "attacker",
        "SACCT_FORMAT": "attacker",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    environment = evidence._sanitized_process_environment()
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert not (set(hostile) - {"PATH"}).intersection(environment)


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


def _r3_probe_paths(
    root: Path, classification: str
) -> tuple[Path, Path, Path, Path, Path, list[tuple[str, Path]], dict[str, str], list[str]]:
    recovery = root / "schema5-v1"
    r3_checkout = recovery / evidence._R3_RELEASE_CHECKOUT_DIRECTORY
    r12_checkout = recovery / evidence._R12_RELEASE_CHECKOUT_DIRECTORY
    toolchain = recovery / evidence._R12_TOOLCHAIN_RELATIVE_ROOT
    temporary = root / "schema5-r3-prelaunch-probe"
    expected_paths = {
        "tagged-r3-release-checkout": r3_checkout,
        "tagged-r12-release-checkout": r12_checkout,
        "sealed-r12-conda-toolchain": toolchain,
    }
    if classification == "unsafe_recorded_broken_internal_symlink":
        expected_paths["recorded-shared-conda-base"] = (
            evidence._R3_SHARED_CONDA_BASE
        )
    inputs = [
        (name, expected_paths[name])
        for name in evidence._R3_PROBE_INPUT_NAMES[classification]
    ]
    environment = evidence._r3_probe_expected_environment(
        temporary, classification=classification
    )
    if classification == "unsafe_recorded_broken_internal_symlink":
        argv = [
            str(toolchain / "base/bin/python"),
            "-I",
            "-B",
            str(
                r3_checkout
                / evidence._R3_RUNTIME_IDENTITY_RELATIVE_PATH
            ),
            "--conda-executable",
            str(evidence._R3_SHARED_CONDA_EXECUTABLE),
        ]
    else:
        argv = [
            str(toolchain / "base/bin/conda"),
            "create",
            "--yes",
            "--offline",
            "--clone",
            str(toolchain / "base"),
            "--prefix",
            str(temporary / "offline-clone-destination"),
        ]
    output = temporary / evidence._R3_PROBE_ENVELOPE_FILENAMES[classification]
    return (
        recovery,
        r3_checkout,
        r12_checkout,
        toolchain,
        output,
        inputs,
        environment,
        argv,
    )


def _r3_prelaunch_failure_envelope(
    tmp_path: Path,
    classification: str,
) -> Path:
    (
        _recovery,
        r3_checkout,
        _r12_checkout,
        _toolchain,
        output,
        inputs,
        environment,
        argv,
    ) = _r3_probe_paths(tmp_path, classification)
    temporary = output.parent
    stdout = ""
    if classification == "unsafe_recorded_broken_internal_symlink":
        broken = (
            evidence._R3_SHARED_CONDA_BASE
            / evidence._R3_BROKEN_SYMLINK_PATH
        )
        stderr = (
            "ERROR: "
            f"unsafe Conda runtime symlink {broken}: "
            "[Errno 2] No such file or directory: "
            f"'{evidence._R3_BROKEN_SYMLINK_MISSING_TARGET}'\n"
        )
        returncode = 2
        observations = {
            "broken_symlink": {
                "path": evidence._R3_BROKEN_SYMLINK_PATH,
                "target": evidence._R3_BROKEN_SYMLINK_TARGET,
                "target_exists": False,
            },
            "source_inventory_unchanged": True,
        }
    else:
        stdout = (
            f"Source:      {_toolchain / 'base'}\n"
            f"Destination: {temporary / 'offline-clone-destination'}\n"
            "Packages: 89\n"
            "Files: 3\n\n"
            "Downloading and Extracting Packages: ...working..."
            "\rpython-3.12 | | 0% \x1b[A done\n"
        )
        stderr = (
            "OfflineError: EnforceUnusedAdapter called with url "
            "https://conda.anaconda.org/conda-forge/linux-64/python-3.13.conda.\n"
            "This command is using a remote connection in offline mode.\n"
        )
        returncode = 1
        observations = {
            "conda_network_disabled": True,
            "offline_error_detected": True,
            "release_local_package_cache_initial_file_count": 0,
            "release_local_package_cache_seeded": False,
            "source_inventory_unchanged": True,
        }
    inventories = []
    for index, (name, path) in enumerate(inputs):
        inventory_sha = hashlib.sha256(
            f"{classification}:{index}:inventory".encode()
        ).hexdigest()
        inventories.append(
            {
                "name": name,
                "path": str(path),
                "before_sha256": inventory_sha,
                "after_sha256": inventory_sha,
                "entry_count": 10 + index,
                "total_bytes": 1024 + index,
                "unchanged": True,
            }
        )
    payload = {
        "schema_version": 1,
        "protocol": evidence._R3_PRELAUNCH_FAILURE_ENVELOPE_PROTOCOL,
        "release_id": evidence._R3_RELEASE_ID,
        "release_tag": evidence._R3_RELEASE_TAG,
        "release_git_commit": evidence._R3_RELEASE_COMMIT,
        "release_tag_object": evidence._R3_RELEASE_TAG_OBJECT,
        "classification": classification,
        "deterministic": True,
        "requires_superseding_release": True,
        "retry_in_place": False,
        "pre_scheduler_submission": True,
        "scheduler_job_ids": [],
        "command": {
            "argv": argv,
            "cwd": str(r3_checkout),
            "environment": environment,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
        },
        "input_inventories": inventories,
        "write_scope": {
            "roots": [str(temporary)],
            "all_within_temporary_storage": True,
            "results_root_touched": False,
            "release_checkout_touched": False,
            "live_prefixes_touched": False,
        },
        "observations": observations,
        "observed_at": "2026-07-27T18:00:00+00:00",
    }
    payload["failure_id"] = evidence._sha256_bytes(
        evidence._canonical_json(payload)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(evidence._canonical_json(payload))
    return output


def _record_probe_fixture(
    tmp_path: Path,
    monkeypatch,
    classification: str,
) -> tuple[dict, Path, list]:
    if classification == "unsafe_recorded_broken_internal_symlink":
        fake_shared = tmp_path / "recorded-shared-conda-base"
        monkeypatch.setattr(evidence, "_R3_SHARED_CONDA_BASE", fake_shared)
        monkeypatch.setattr(
            evidence,
            "_R3_SHARED_CONDA_EXECUTABLE",
            fake_shared / "bin/conda",
        )
        monkeypatch.setattr(
            evidence,
            "_R3_BROKEN_SYMLINK_MISSING_TARGET",
            fake_shared / "bin/x86_64-conda-linux-gnu-TOOLS=addr2line",
        )
    (
        _recovery,
        r3_checkout,
        r12_checkout,
        toolchain,
        output,
        inputs,
        environment,
        argv,
    ) = _r3_probe_paths(tmp_path, classification)
    for path in (r3_checkout, r12_checkout, toolchain):
        path.mkdir(parents=True, exist_ok=True)
        (path / "fixture").write_text(path.name, encoding="utf-8")
    temporary = output.parent
    temporary.mkdir(mode=0o700)
    if classification == "unsafe_recorded_broken_internal_symlink":
        shared = dict(inputs)["recorded-shared-conda-base"]
        link = shared / evidence._R3_BROKEN_SYMLINK_PATH
        link.parent.mkdir(parents=True)
        (shared / "bin").mkdir(exist_ok=True)
        (shared / "bin" / "conda").write_text("fixture\n", encoding="utf-8")
        link.symlink_to(evidence._R3_BROKEN_SYMLINK_TARGET)
        broken = shared / evidence._R3_BROKEN_SYMLINK_PATH
        failure = (
            "ERROR: "
            f"unsafe Conda runtime symlink {broken}: "
            "[Errno 2] No such file or directory: "
            f"'{evidence._R3_BROKEN_SYMLINK_MISSING_TARGET}'\n"
        ).encode()
        returncode = 2
    else:
        (temporary / "empty-conda-pkgs").mkdir()
        success_output = (
            f"Source:      {toolchain / 'base'}\n"
            f"Destination: {temporary / 'offline-clone-destination'}\n"
            "Packages: 89\n"
            "Files: 3\n\n"
            "Downloading and Extracting Packages: ...working..."
            "\rpython-3.12 | | 0% \x1b[A done\n"
        ).encode()
        failure = (
            b"OfflineError: EnforceUnusedAdapter called with url "
            b"https://conda.anaconda.org/conda-forge/linux-64/python.conda.\n"
            b"This command is using a remote connection in offline mode.\n"
            b"OfflineError: EnforceUnusedAdapter called with url "
            b"https://conda.anaconda.org/conda-forge/noarch/pip.conda.\n"
            b"This command is using a remote connection in offline mode.\n"
        )
        returncode = 1
    monkeypatch.setattr(
        evidence,
        "_validate_r3_probe_immutable_inputs",
        lambda **_kwargs: None,
    )
    calls = []

    def runner(command, **kwargs):
        calls.append((list(command), kwargs))
        return __import__("subprocess").CompletedProcess(
            command,
            returncode,
            success_output
            if classification == "offline_clone_unseeded_release_local_cache"
            else b"",
            failure,
        )

    kwargs = {
        "output": output,
        "classification": classification,
        "command": argv,
        "cwd": r3_checkout,
        "environment": list(environment.items()),
        "input_roots": inputs,
        "write_roots": [temporary],
        "observed_at": "2026-07-27T18:00:00+00:00",
        "runner": runner,
    }
    return kwargs, output, calls


@pytest.mark.parametrize(
    "classification",
    evidence._R3_PRELAUNCH_FAILURE_CLASSIFICATIONS,
)
def test_record_r3_prelaunch_attempt_captures_only_canonical_failure_and_replays(
    tmp_path,
    monkeypatch,
    classification,
):
    kwargs, output, calls = _record_probe_fixture(
        tmp_path, monkeypatch, classification
    )
    dry = evidence.record_prelaunch_attempt(**kwargs)
    assert dry["status"] == "dry_run"
    assert not output.exists()
    assert calls == []

    recorded = evidence.record_prelaunch_attempt(**kwargs, apply=True)
    assert recorded["status"] == "recorded"
    assert recorded["command"]["stderr_sha256"] == hashlib.sha256(
        recorded["command"]["stderr"].encode()
    ).hexdigest()
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert stat.S_IMODE(
        output.with_suffix(".json.sha256").stat().st_mode
    ) == 0o444
    evidence._validate_r3_prelaunch_failure_envelope(output)
    assert len(calls) == 1
    assert calls[0][0] == kwargs["command"]
    assert calls[0][1]["env"] == dict(kwargs["environment"])
    assert calls[0][1]["text"] is False

    replay = evidence.record_prelaunch_attempt(**kwargs, apply=True)
    assert replay["status"] == "already_recorded"
    assert replay["failure_id"] == recorded["failure_id"]
    assert len(calls) == 1


def test_record_r3_prelaunch_attempt_rejects_arbitrary_command_before_execution(
    tmp_path,
    monkeypatch,
):
    kwargs, output, calls = _record_probe_fixture(
        tmp_path,
        monkeypatch,
        "unsafe_recorded_broken_internal_symlink",
    )
    kwargs["command"] = ["/bin/false"]
    with pytest.raises(evidence.EvidenceError, match="canonical classification"):
        evidence.record_prelaunch_attempt(**kwargs, apply=True)
    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize("drift", ["python", "script"])
def test_record_r3_prelaunch_attempt_rejects_arbitrary_python_or_script(
    tmp_path,
    monkeypatch,
    drift,
):
    kwargs, output, calls = _record_probe_fixture(
        tmp_path,
        monkeypatch,
        "unsafe_recorded_broken_internal_symlink",
    )
    command = list(kwargs["command"])
    if drift == "python":
        command[0] = "/opt/arbitrary/bin/python"
    else:
        command[3] = "/opt/arbitrary/probe.py"
    kwargs["command"] = command
    with pytest.raises(evidence.EvidenceError, match="canonical classification"):
        evidence.record_prelaunch_attempt(**kwargs, apply=True)
    assert calls == []
    assert not output.exists()


def test_record_r3_prelaunch_attempt_rejects_fabricated_offline_error(
    tmp_path,
    monkeypatch,
):
    kwargs, output, calls = _record_probe_fixture(
        tmp_path,
        monkeypatch,
        "offline_clone_unseeded_release_local_cache",
    )

    def fabricated(command, **options):
        calls.append((list(command), options))
        return __import__("subprocess").CompletedProcess(
            command,
            1,
            b"",
            b"OfflineError: package cache contains no usable artifacts\n",
        )

    kwargs["runner"] = fabricated
    with pytest.raises(
        evidence.EvidenceError, match="remote-fetch/progress context"
    ):
        evidence.record_prelaunch_attempt(**kwargs, apply=True)
    assert len(calls) == 1
    assert not output.exists()


def _rewrite_r3_failure_payload(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    payload["command"]["stdout_sha256"] = hashlib.sha256(
        payload["command"]["stdout"].encode()
    ).hexdigest()
    payload["command"]["stderr_sha256"] = hashlib.sha256(
        payload["command"]["stderr"].encode()
    ).hexdigest()
    payload.pop("failure_id", None)
    payload["failure_id"] = evidence._sha256_bytes(
        evidence._canonical_json(payload)
    )
    path.write_bytes(evidence._canonical_json(payload))


def test_sealed_r3_envelope_validator_rejects_rehashed_generic_failure(
    tmp_path,
):
    envelope = _r3_prelaunch_failure_envelope(
        tmp_path, "unsafe_recorded_broken_internal_symlink"
    )
    _rewrite_r3_failure_payload(
        envelope,
        lambda payload: payload["command"].update(
            {"argv": ["/bin/false"], "stderr": ""}
        ),
    )
    with pytest.raises(evidence.EvidenceError, match="canonical classification"):
        evidence._validate_r3_prelaunch_failure_envelope(envelope)


def test_sealed_r3_envelope_validator_rejects_rehashed_fabricated_offline_error(
    tmp_path,
):
    envelope = _r3_prelaunch_failure_envelope(
        tmp_path, "offline_clone_unseeded_release_local_cache"
    )
    _rewrite_r3_failure_payload(
        envelope,
        lambda payload: payload["command"].update(
            {"stderr": "OfflineError: fabricated\n"}
        ),
    )
    with pytest.raises(
        evidence.EvidenceError, match="remote-fetch/progress context"
    ):
        evidence._validate_r3_prelaunch_failure_envelope(envelope)


@pytest.mark.parametrize(
    ("classification", "drift"),
    [
        ("unsafe_recorded_broken_internal_symlink", "stderr_suffix"),
        ("offline_clone_unseeded_release_local_cache", "stderr_prefix"),
        ("offline_clone_unseeded_release_local_cache", "stderr_suffix"),
        ("offline_clone_unseeded_release_local_cache", "stdout"),
    ],
)
def test_sealed_r3_envelope_rejects_rehashed_diagnostic_wrappers(
    tmp_path: Path,
    classification: str,
    drift: str,
) -> None:
    envelope = _r3_prelaunch_failure_envelope(tmp_path, classification)

    def mutate(payload):
        if drift == "stderr_prefix":
            payload["command"]["stderr"] = (
                "fabricated-prefix\n" + payload["command"]["stderr"]
            )
        elif drift == "stderr_suffix":
            payload["command"]["stderr"] += "fabricated-suffix\n"
        else:
            payload["command"]["stdout"] = "fabricated-stdout\n"

    _rewrite_r3_failure_payload(envelope, mutate)
    expected = (
        "runtime-identity error signature"
        if classification == "unsafe_recorded_broken_internal_symlink"
        else "remote-fetch/progress context"
    )
    with pytest.raises(evidence.EvidenceError, match=expected):
        evidence._validate_r3_prelaunch_failure_envelope(envelope)


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("extra_environment", "extra or missing"),
        ("missing_environment", "extra or missing"),
        ("unbound_input", "input names or order"),
        ("outside_output", "canonical temporary output"),
        ("outside_prefix", "canonical classification"),
    ],
)
def test_record_r3_prelaunch_attempt_rejects_contract_drift(
    tmp_path,
    monkeypatch,
    mutation,
    match,
):
    kwargs, output, calls = _record_probe_fixture(
        tmp_path,
        monkeypatch,
        "offline_clone_unseeded_release_local_cache",
    )
    if mutation == "extra_environment":
        kwargs["environment"].append(("PATH", "/usr/bin"))
    elif mutation == "missing_environment":
        kwargs["environment"] = kwargs["environment"][1:]
    elif mutation == "unbound_input":
        kwargs["input_roots"] = kwargs["input_roots"][:-1]
    elif mutation == "outside_output":
        kwargs["output"] = tmp_path / "escaped.json"
    else:
        command = list(kwargs["command"])
        command[-1] = str(tmp_path / "outside-prefix")
        kwargs["command"] = command
    with pytest.raises(evidence.EvidenceError, match=match):
        evidence.record_prelaunch_attempt(**kwargs, apply=True)
    assert calls == []
    assert not output.exists()


def test_record_r3_prelaunch_attempt_brackets_immutable_verification_with_inventories(
    tmp_path,
    monkeypatch,
):
    kwargs, output, calls = _record_probe_fixture(
        tmp_path,
        monkeypatch,
        "unsafe_recorded_broken_internal_symlink",
    )
    toolchain = dict(kwargs["input_roots"])["sealed-r12-conda-toolchain"]

    def mutating_verifier(**_kwargs):
        (toolchain / "fixture").write_text("mutated", encoding="utf-8")

    monkeypatch.setattr(
        evidence, "_validate_r3_probe_immutable_inputs", mutating_verifier
    )
    with pytest.raises(evidence.EvidenceError, match="verification mutated"):
        evidence.record_prelaunch_attempt(**kwargs)
    assert calls == []
    assert not output.exists()


def test_r3_probe_immutable_input_gate_binds_tags_sources_and_toolchain(
    tmp_path,
    monkeypatch,
):
    (
        recovery,
        r3_checkout,
        r12_checkout,
        toolchain,
        output,
        inputs,
        environment,
        argv,
    ) = _r3_probe_paths(
        tmp_path, "offline_clone_unseeded_release_local_cache"
    )
    temporary = output.parent
    temporary.mkdir(parents=True)
    for checkout in (r3_checkout, r12_checkout):
        (checkout / "scripts").mkdir(parents=True)
    r3_pilot = r3_checkout / evidence._R3_PILOT_RELATIVE_PATH
    r3_runtime = r3_checkout / evidence._R3_RUNTIME_IDENTITY_RELATIVE_PATH
    r3_pilot.write_text("exact r3 pilot\n", encoding="utf-8")
    r3_runtime.write_text("exact r3 runtime identity\n", encoding="utf-8")
    r12_sealer = r12_checkout / evidence._R12_SEALER_RELATIVE_PATH
    provisioner = (
        r12_checkout / evidence._R12_TOOLCHAIN_PROVISIONER_RELATIVE_PATH
    )
    r12_sealer.write_text("exact r12 sealer\n", encoding="utf-8")
    provisioner.write_text("exact r12 provisioner\n", encoding="utf-8")
    (toolchain / "base/bin").mkdir(parents=True)
    (toolchain / "base/bin/python").write_text(
        "#!/bin/sh\n", encoding="utf-8"
    )
    (toolchain / "base/bin/conda").write_text(
        "#!/bin/sh\n", encoding="utf-8"
    )
    monkeypatch.setattr(evidence, "__file__", str(r12_sealer))
    monkeypatch.setattr(
        evidence, "_R3_PILOT_SHA256", hashlib.sha256(r3_pilot.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        evidence,
        "_R3_RUNTIME_IDENTITY_SHA256",
        hashlib.sha256(r3_runtime.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        evidence,
        "_validate_r3_durable_release_identity",
        lambda **_kwargs: {"release_git_commit": evidence._R3_RELEASE_COMMIT},
    )

    def fake_git(_checkout, *arguments):
        if arguments[:2] == ("cat-file", "-t"):
            return "tag"
        if arguments[:2] == ("rev-parse", "--verify"):
            return "4" * 40
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(evidence, "_run_git", fake_git)
    verification_calls = []

    monkeypatch.setattr(
        evidence.conda_toolchain,
        "__file__",
        str(provisioner),
    )

    def fake_binding(root, *, exercise):
        verification_calls.append((Path(root), exercise))
        return {
            "schema_version": evidence.conda_toolchain.SCHEMA_VERSION,
            "protocol": evidence.conda_toolchain.PROTOCOL,
            "release_tag": "sweep-recovery-schema5-v1.2-r12",
            "chain_namespace": "schema5-v1.2-r12",
            "toolchain_root": str(toolchain),
            "base_prefix": str(toolchain / "base"),
            "portable_shebang": {
                "absolute_base_prefix_interpreter_required": True,
                "interpreter": str(toolchain / "base/bin/python"),
                "maximum_shebang_bytes": 127,
                "shebang_bytes": 1,
            },
            "completion_marker": {
                "path": str(
                    toolchain
                    / evidence.conda_toolchain.MARKER_NAME
                ),
                "sha256": "1" * 64,
                "size": 1,
            },
            "marker_id": "2" * 64,
            "installer_contract": {},
            "intent_id": "3" * 64,
            "conda_executable": {
                "path": str(toolchain / "base/bin/conda"),
                "sha256": "4" * 64,
                "size": 1,
                "mode": 0o555,
                "link_count": 1,
            },
            "runtime_identity_sha256": "5" * 64,
            "complete_prefix_inventory_sha256": "6" * 64,
            "read_only_probes": {},
            "binding_id": "7" * 64,
        }

    monkeypatch.setattr(
        evidence.conda_toolchain,
        "verified_conda_toolchain_binding",
        fake_binding,
    )
    contract = evidence._validated_r3_probe_contract(
        classification="offline_clone_unseeded_release_local_cache",
        argv=argv,
        cwd=r3_checkout,
        environment=environment,
        input_paths=inputs,
        write_roots=[temporary],
        output_path=output,
    )
    evidence._validate_r3_probe_immutable_inputs(
        contract=contract, environment=environment
    )

    assert recovery == contract["recovery_root"]
    assert verification_calls == [(toolchain, False)]


def test_r3_probe_input_inventory_preserves_internal_hardlink_topology(
    tmp_path,
):
    root = tmp_path / "sealed-toolchain"
    root.mkdir()
    first = root / "package-cache-payload"
    second = root / "linked-runtime-payload"
    first.write_bytes(b"same inode\n")
    os.link(first, second)

    inventory = evidence._prelaunch_input_inventory(root)

    assert inventory["file_count"] == 2
    assert inventory["entry_count"] == 3
    assert inventory["total_bytes"] == 2 * len(b"same inode\n")


def test_git_identity_queries_disable_optional_index_refresh(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    observed = {}

    def fake_run(*args, **kwargs):
        observed["args"] = args
        observed["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(evidence.subprocess, "run", fake_run)
    assert evidence._run_git(checkout, "status", "--porcelain=v1") == ""
    assert observed["environment"]["GIT_OPTIONAL_LOCKS"] == "0"
    assert observed["args"][0][:3] == (
        "/usr/bin/git",
        "-C",
        str(checkout),
    )


def test_git_identity_query_preserves_fresh_clone_index_inode(tmp_path):
    origin = tmp_path / "origin"
    clone = tmp_path / "clone"

    def git(*arguments):
        return subprocess.run(
            ["/usr/bin/git", *map(str, arguments)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    git("init", origin)
    git("-C", origin, "config", "user.name", "Schema Five Test")
    git("-C", origin, "config", "user.email", "schema5@example.invalid")
    (origin / "tracked.txt").write_text("immutable\n", encoding="utf-8")
    git("-C", origin, "add", "tracked.txt")
    git("-C", origin, "commit", "-m", "fixture")
    git("clone", "--no-local", "--no-checkout", origin, clone)
    git("-C", clone, "checkout", "--detach", "HEAD")

    index = clone / ".git/index"
    before = (index.stat().st_ino, evidence._sha256_file(index))
    assert evidence._run_git(
        clone, "status", "--porcelain=v1", "--untracked-files=all"
    ) == ""
    after = (index.stat().st_ino, evidence._sha256_file(index))
    assert after == before


def test_recorded_shared_base_inventory_matches_runtime_exclusion_boundary(
    tmp_path,
):
    root = tmp_path / "shared-base"
    (root / "libexec").mkdir(parents=True)
    (root / "libexec" / "runtime").write_text("bound\n", encoding="utf-8")
    for name in evidence._R3_SHARED_RUNTIME_EXCLUDED_TOP_LEVEL:
        directory = root / name
        directory.mkdir()
        (directory / "excluded").write_text("cache\n", encoding="utf-8")

    inventory = evidence._r3_probe_input_inventory(
        "recorded-shared-conda-base", root
    )
    (root / "pkgs" / "excluded").write_text("cache drift\n", encoding="utf-8")
    replay = evidence._r3_probe_input_inventory(
        "recorded-shared-conda-base", root
    )

    assert inventory == replay
    assert inventory["file_count"] == 1


def test_record_r3_prelaunch_attempt_withholds_envelope_on_source_mutation(
    tmp_path,
    monkeypatch,
):
    kwargs, output, _calls = _record_probe_fixture(
        tmp_path,
        monkeypatch,
        "unsafe_recorded_broken_internal_symlink",
    )
    source = dict(kwargs["input_roots"])["tagged-r3-release-checkout"]

    def runner(argv, **_kwargs):
        (source / "fixture").write_text("mutated", encoding="utf-8")
        broken = (
            evidence._R3_SHARED_CONDA_BASE
            / evidence._R3_BROKEN_SYMLINK_PATH
        )
        stderr = (
            "ERROR: "
            f"unsafe Conda runtime symlink {broken}: "
            "[Errno 2] No such file or directory: "
            f"'{evidence._R3_BROKEN_SYMLINK_MISSING_TARGET}'\n"
        ).encode()
        return __import__("subprocess").CompletedProcess(
            argv, 2, b"", stderr
        )

    kwargs["runner"] = runner
    with pytest.raises(evidence.EvidenceError, match="mutated its input"):
        evidence.record_prelaunch_attempt(**kwargs, apply=True)
    assert not output.exists()


def _r3_prelaunch_inputs(tmp_path: Path) -> tuple[list[Path], Path, Path, Path]:
    recovery = tmp_path / "schema5-v1"
    envelopes = [
        _r3_prelaunch_failure_envelope(tmp_path, classification)
        for classification in evidence._R3_PRELAUNCH_FAILURE_CLASSIFICATIONS
    ]
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
    marker = recovery / evidence._R3_DURABLE_RELEASE_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("validator fixture\n", encoding="utf-8")
    checkout = (
        recovery / "materialization_pilot_source_checkout_v1_2_r3"
    )
    checkout.mkdir(exist_ok=True)
    return envelopes, mutation, marker, checkout


def _r3_release_binding(marker: Path, checkout: Path) -> dict:
    bundle = marker.parent / "git_release" / evidence._R3_DURABLE_BUNDLE
    return {
        "release_id": evidence._R3_RELEASE_ID,
        "release_tag": evidence._R3_RELEASE_TAG,
        "chain_namespace": evidence._R3_CHAIN_NAMESPACE,
        "release_git_commit": evidence._R3_RELEASE_COMMIT,
        "release_tag_object": evidence._R3_RELEASE_TAG_OBJECT,
        "release_checkout": str(checkout),
        "durable_marker": {
            "path": str(marker),
            "sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
            "size": marker.stat().st_size,
            "marker_id": "1" * 64,
        },
        "bundle": {
            "path": str(bundle),
            "sha256": "2" * 64,
            "size": 1,
        },
        "checksum": {
            "path": str(bundle.with_suffix(bundle.suffix + ".sha256")),
            "sha256": "3" * 64,
            "size": 1,
        },
    }


def test_r3_prelaunch_failure_seal_is_marker_first_no_scheduler_and_idempotent(
    tmp_path, monkeypatch
):
    envelopes, mutation, marker, checkout = _r3_prelaunch_inputs(tmp_path)
    checkout_mode = stat.S_IMODE(checkout.stat().st_mode)
    source_bytes = {path: path.read_bytes() for path in envelopes}
    monkeypatch.setattr(
        evidence,
        "_validate_r3_durable_release_identity",
        lambda **_kwargs: _r3_release_binding(marker, checkout),
    )
    scheduler_calls = []

    def forbidden_scheduler(*args, **kwargs):
        scheduler_calls.append((args, kwargs))
        raise AssertionError("prelaunch evidence must not query Slurm")

    monkeypatch.setattr(evidence, "_scheduler_states", forbidden_scheduler)
    monkeypatch.setattr(
        evidence, "_capture_canary_scheduler_evidence", forbidden_scheduler
    )
    (tmp_path / "prelaunch-failures").mkdir()
    root = tmp_path / "prelaunch-failures" / "schema5-v1.2-r3"
    dry = evidence.seal_prelaunch_failure(
        evidence_root=root,
        durable_release_marker=marker,
        release_checkout=checkout,
        failure_envelopes=envelopes,
        mutation_evidence=[mutation],
    )
    assert dry["status"] == "dry_run"
    assert dry["scheduler_evidence_required"] is False
    assert not root.exists()

    original_atomic = evidence._prelaunch_atomic_bytes
    publication_order = []

    def observed_atomic(path, payload, *, evidence_root, artifacts):
        publication_order.append(path.name)
        return original_atomic(
            path,
            payload,
            evidence_root=evidence_root,
            artifacts=artifacts,
        )

    monkeypatch.setattr(evidence, "_prelaunch_atomic_bytes", observed_atomic)
    sealed = evidence.seal_prelaunch_failure(
        evidence_root=root,
        durable_release_marker=marker,
        release_checkout=checkout,
        failure_envelopes=envelopes,
        mutation_evidence=[mutation],
        apply=True,
    )
    artifacts = evidence._prelaunch_failure_artifact_paths(root)
    assert sealed["status"] == "sealed"
    assert sealed["classification"] == (
        "deterministic_materialization_contract_failure_sealed_fail_closed"
    )
    assert sealed["known_scheduler_job_ids"] == []
    assert publication_order == [
        artifacts[key].name
        for key in evidence._PRELAUNCH_FAILURE_ARTIFACT_ORDER
    ]
    assert not stat.S_IMODE(root.stat().st_mode) & 0o222
    assert all(
        not stat.S_IMODE(path.stat().st_mode) & 0o222
        for path in artifacts.values()
    )
    assert stat.S_IMODE(checkout.stat().st_mode) == checkout_mode
    assert {path: path.read_bytes() for path in envelopes} == source_bytes
    assert scheduler_calls == []

    publication_order.clear()
    repeated = evidence.seal_prelaunch_failure(
        evidence_root=root,
        durable_release_marker=marker,
        release_checkout=checkout,
        failure_envelopes=envelopes,
        mutation_evidence=[mutation],
        apply=True,
    )
    assert repeated["status"] == "already_sealed"
    assert repeated["seal_id"] == sealed["seal_id"]
    assert publication_order == []
    assert scheduler_calls == []


def test_r3_prelaunch_failure_seal_rejects_duplicate_or_drifted_envelopes(
    tmp_path, monkeypatch
):
    envelopes, mutation, marker, checkout = _r3_prelaunch_inputs(tmp_path)
    monkeypatch.setattr(
        evidence,
        "_validate_r3_durable_release_identity",
        lambda **_kwargs: _r3_release_binding(marker, checkout),
    )
    root = tmp_path / "evidence"
    with pytest.raises(evidence.EvidenceError, match="duplicate"):
        evidence.seal_prelaunch_failure(
            evidence_root=root,
            durable_release_marker=marker,
            release_checkout=checkout,
            failure_envelopes=[envelopes[0], envelopes[0]],
            mutation_evidence=[mutation],
        )
    assert not root.exists()

    payload = json.loads(envelopes[1].read_text(encoding="utf-8"))
    payload["observations"]["release_local_package_cache_seeded"] = True
    payload["failure_id"] = evidence._sha256_bytes(
        evidence._canonical_json(
            {key: value for key, value in payload.items() if key != "failure_id"}
        )
    )
    envelopes[1].write_bytes(evidence._canonical_json(payload))
    with pytest.raises(evidence.EvidenceError, match="empty-cache"):
        evidence.seal_prelaunch_failure(
            evidence_root=root,
            durable_release_marker=marker,
            release_checkout=checkout,
            failure_envelopes=envelopes,
            mutation_evidence=[mutation],
        )
    assert not root.exists()


def test_r3_prelaunch_failure_seal_detects_archive_drift(
    tmp_path, monkeypatch
):
    envelopes, mutation, marker, checkout = _r3_prelaunch_inputs(tmp_path)
    monkeypatch.setattr(
        evidence,
        "_validate_r3_durable_release_identity",
        lambda **_kwargs: _r3_release_binding(marker, checkout),
    )
    root = tmp_path / "evidence"
    evidence.seal_prelaunch_failure(
        evidence_root=root,
        durable_release_marker=marker,
        release_checkout=checkout,
        failure_envelopes=envelopes,
        mutation_evidence=[mutation],
        apply=True,
    )
    archive = evidence._prelaunch_failure_artifact_paths(root)["offline_cache"]
    archive.chmod(0o644)
    archive.write_bytes(b"drifted\n")
    with pytest.raises(evidence.EvidenceError, match="artifact is unsafe"):
        evidence.seal_prelaunch_failure(
            evidence_root=root,
            durable_release_marker=marker,
            release_checkout=checkout,
            failure_envelopes=envelopes,
            mutation_evidence=[mutation],
            apply=True,
        )


def _sealed_r3_prelaunch_fixture(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, dict]:
    envelopes, mutation, marker, checkout = _r3_prelaunch_inputs(tmp_path)
    recovery = tmp_path / "schema5-v1"
    monkeypatch.setattr(
        evidence,
        "_validate_r3_durable_release_identity",
        lambda **_kwargs: _r3_release_binding(marker, checkout),
    )
    root = recovery / evidence.R3_PRELAUNCH_FAILURE_RELATIVE_ROOT
    root.parent.mkdir()
    sealed = evidence.seal_prelaunch_failure(
        evidence_root=root,
        durable_release_marker=marker,
        release_checkout=checkout,
        failure_envelopes=envelopes,
        mutation_evidence=[mutation],
        apply=True,
    )
    return root, sealed


def test_verify_r3_prelaunch_failure_seal_needs_only_sealed_root(
    tmp_path, monkeypatch
):
    root, sealed = _sealed_r3_prelaunch_fixture(tmp_path, monkeypatch)

    # Prove verification does not re-enter the external release validator or
    # scheduler boundary after the seal has been published.
    monkeypatch.setattr(
        evidence,
        "_validate_r3_durable_release_identity",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("sealed verification consulted the external release")
        ),
    )
    monkeypatch.setattr(
        evidence,
        "_scheduler_states",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sealed verification queried Slurm")
        ),
    )
    binding = evidence.verify_prelaunch_failure_seal(root)
    marker = evidence._prelaunch_failure_artifact_paths(root)["marker"]

    assert binding == {
        "schema_version": 1,
        "protocol": evidence._R3_PRELAUNCH_FAILURE_BINDING_PROTOCOL,
        "root": str(root),
        "marker": {
            "path": str(marker),
            "sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
            "size": marker.stat().st_size,
            "seal_id": sealed["seal_id"],
        },
        "release_id": evidence._R3_RELEASE_ID,
        "release_tag": evidence._R3_RELEASE_TAG,
        "release_git_commit": evidence._R3_RELEASE_COMMIT,
        "release_tag_object": evidence._R3_RELEASE_TAG_OBJECT,
        "chain_namespace": evidence._R3_CHAIN_NAMESPACE,
        "classification": (
            "deterministic_materialization_contract_failure_sealed_fail_closed"
        ),
        "failure_classifications": list(
            evidence._R3_PRELAUNCH_FAILURE_CLASSIFICATIONS
        ),
        "retry_in_place": False,
        "requires_superseding_release": True,
        "pre_scheduler_submission": True,
        "scheduler_evidence_required": False,
        "known_scheduler_job_ids": [],
        "mutation_claim": (
            "bound_to_preexisting_evidence_not_inferred_from_absence"
        ),
        "archive_inventory_sha256": hashlib.sha256(
            evidence._prelaunch_failure_artifact_paths(root)[
                "inventory"
            ].read_bytes()
        ).hexdigest(),
    }


def test_verify_r3_prelaunch_failure_seal_rejects_writable_artifact(
    tmp_path, monkeypatch
):
    root, _sealed = _sealed_r3_prelaunch_fixture(tmp_path, monkeypatch)
    archive = evidence._prelaunch_failure_artifact_paths(root)["offline_cache"]
    archive.chmod(0o644)

    with pytest.raises(evidence.EvidenceError, match="artifact is unsafe"):
        evidence.verify_prelaunch_failure_seal(root)


def test_verify_r3_prelaunch_failure_seal_rejects_missing_artifact(
    tmp_path, monkeypatch
):
    root, _sealed = _sealed_r3_prelaunch_fixture(tmp_path, monkeypatch)
    inventory = evidence._prelaunch_failure_artifact_paths(root)["inventory"]
    root.chmod(0o755)
    inventory.unlink()
    root.chmod(0o555)

    with pytest.raises(
        evidence.EvidenceError,
        match="marker-last prefix|cardinality/order drifted",
    ):
        evidence.verify_prelaunch_failure_seal(root)


def test_verify_r3_prelaunch_failure_seal_rejects_inventory_tamper(
    tmp_path, monkeypatch
):
    root, _sealed = _sealed_r3_prelaunch_fixture(tmp_path, monkeypatch)
    inventory = evidence._prelaunch_failure_artifact_paths(root)["inventory"]
    rows = [
        json.loads(line)
        for line in inventory.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["size"] += 1
    inventory.chmod(0o644)
    inventory.write_bytes(
        b"".join(evidence._canonical_json(row) for row in rows)
    )
    inventory.chmod(0o444)

    with pytest.raises(evidence.EvidenceError, match="inventory drifted"):
        evidence.verify_prelaunch_failure_seal(root)


def test_verify_r3_prelaunch_failure_seal_rejects_release_substitution(
    tmp_path, monkeypatch
):
    root, _sealed = _sealed_r3_prelaunch_fixture(tmp_path, monkeypatch)
    artifacts = evidence._prelaunch_failure_artifact_paths(root)
    intent = json.loads(artifacts["intent"].read_text(encoding="utf-8"))
    marker = json.loads(artifacts["marker"].read_text(encoding="utf-8"))

    intent["release"]["release_tag"] = "sweep-recovery-schema5-v1.2-r12"
    intent.pop("intent_id")
    intent["intent_id"] = evidence._sha256_bytes(
        evidence._canonical_json(intent)
    )
    intent_raw = evidence._canonical_json(intent)
    artifacts["intent"].chmod(0o644)
    artifacts["intent"].write_bytes(intent_raw)
    artifacts["intent"].chmod(0o444)

    marker["release"]["release_tag"] = "sweep-recovery-schema5-v1.2-r12"
    marker["intent"]["sha256"] = hashlib.sha256(intent_raw).hexdigest()
    marker["intent"]["intent_id"] = intent["intent_id"]
    marker.pop("seal_id")
    marker["seal_id"] = evidence._sha256_bytes(
        evidence._canonical_json(marker)
    )
    artifacts["marker"].chmod(0o644)
    artifacts["marker"].write_bytes(evidence._canonical_json(marker))
    artifacts["marker"].chmod(0o444)

    with pytest.raises(evidence.EvidenceError, match="release identity drifted"):
        evidence.verify_prelaunch_failure_seal(root)


def test_r3_durable_release_identity_binds_exact_tag_bundle_and_checkout(
    tmp_path, monkeypatch
):
    root = tmp_path / "recovery"
    bundle = root / "git_release" / evidence._R3_DURABLE_BUNDLE
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"exact r3 bundle\n")
    bundle.chmod(0o444)
    checksum = bundle.with_suffix(bundle.suffix + ".sha256")
    checksum.write_bytes(
        f"{hashlib.sha256(bundle.read_bytes()).hexdigest()}  {bundle.name}\n".encode()
    )
    checksum.chmod(0o444)
    marker_path = root / evidence._R3_DURABLE_RELEASE_MARKER
    marker = {
        "schema_version": 1,
        "protocol": evidence._R3_DURABLE_RELEASE_PROTOCOL,
        "passed": True,
        "release_id": evidence._R3_RELEASE_ID,
        "release_tag": evidence._R3_RELEASE_TAG,
        "release_git_commit": evidence._R3_RELEASE_COMMIT,
        "release_tag_object": evidence._R3_RELEASE_TAG_OBJECT,
        "chain_namespace": evidence._R3_CHAIN_NAMESPACE,
        "clean_checkout": True,
        "annotated_tag": True,
        "remote_query_read_only": True,
        "remote": "origin",
        "remote_commit_ref": "refs/heads/schema5-v1.2-r3",
        "remote_commit": evidence._R3_RELEASE_COMMIT,
        "remote_peeled_commit": evidence._R3_RELEASE_COMMIT,
        "remote_tag_object": evidence._R3_RELEASE_TAG_OBJECT,
        "remote_url_sha256": "4" * 64,
        "bundle_path": str(bundle),
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "bundle_size": bundle.stat().st_size,
        "checksum_path": str(checksum),
        "checksum_sha256": hashlib.sha256(checksum.read_bytes()).hexdigest(),
        "published_at": "2026-07-27T17:11:50+00:00",
    }
    marker["marker_id"] = evidence._sha256_bytes(
        evidence._pretty_canonical_json(marker)
    )
    marker_path.write_bytes(evidence._pretty_canonical_json(marker))
    marker_path.chmod(0o444)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    commands = []

    def fake_git(_checkout, *arguments):
        commands.append(arguments)
        if arguments[:2] == ("cat-file", "-t"):
            return "tag"
        if arguments[:2] == ("rev-parse", "--verify"):
            requested = arguments[2]
            if requested == f"refs/tags/{evidence._R3_RELEASE_TAG}":
                return evidence._R3_RELEASE_TAG_OBJECT
            return evidence._R3_RELEASE_COMMIT
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        if arguments[:2] == ("bundle", "verify"):
            return f"{bundle} is okay"
        if arguments[:2] == ("bundle", "list-heads"):
            return (
                f"{evidence._R3_RELEASE_TAG_OBJECT} "
                f"refs/tags/{evidence._R3_RELEASE_TAG}"
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(evidence, "_run_git", fake_git)
    binding = evidence._validate_r3_durable_release_identity(
        marker_path=marker_path,
        release_checkout=checkout,
    )
    assert binding["release_git_commit"] == evidence._R3_RELEASE_COMMIT
    assert binding["release_tag_object"] == evidence._R3_RELEASE_TAG_OBJECT
    assert binding["bundle"]["sha256"] == marker["bundle_sha256"]
    assert all(command[0] != "checkout" for command in commands)
