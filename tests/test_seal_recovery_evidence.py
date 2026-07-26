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
