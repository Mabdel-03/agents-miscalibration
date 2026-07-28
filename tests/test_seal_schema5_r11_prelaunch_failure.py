from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest

from scripts import seal_schema5_r11_prelaunch_failure as seal


def test_r11_reproduction_command_is_exact(tmp_path):
    recovery = tmp_path / "results/recovery/schema5-v1"

    argv = seal._reproduction_argv(recovery, "researcher")

    assert argv[:3] == [
        seal.sys.executable,
        "-I",
        str(
            recovery
            / seal.R11_CHECKOUT
            / "scripts/seal_schema5_r9_prelaunch_failure.py"
        ),
    ]
    assert argv[3:] == [
        "seal",
        "--evidence-root",
        str(recovery / seal.R9_EVIDENCE_RELATIVE),
        "--recovery-root",
        str(recovery),
        "--scheduler-user",
        "researcher",
        "--apply",
    ]
    assert seal.EXPECTED_STDERR == (
        "ERROR: equivalent r9 OfflineError stream drifted\n"
    )


def test_r11_failure_seal_is_dry_by_default_and_marker_last(
    tmp_path, monkeypatch
):
    recovery = tmp_path / "results/recovery/schema5-v1"
    recovery.mkdir(parents=True)
    evidence = recovery / seal.R11_FAILURE_RELATIVE
    observed_evidence = {"inventory": {"entry_count": 5}}
    observed_volatile = {"inventory": {"entry_count": 9}}
    diagnostic = {
        "present_at_seal": False,
        "authoritative_failure_evidence": False,
    }
    intent = {
        "schema_version": seal.SCHEMA_VERSION,
        "protocol": f"{seal.PROTOCOL}-intent",
        "classification": seal.CLASSIFICATION,
        "source_release": {},
        "observed_incomplete_evidence": observed_evidence,
        "observed_volatile_probe": observed_volatile,
        "operator_diagnostic": diagnostic,
        "temporary_quarantine_destinations": [],
        "reproduction": {},
        "scientific_state": {
            "captured_before_r12_production_outputs": True,
            "result_mutation_count": 0,
            "scheduler_job_count": 0,
        },
        "scheduler": {
            "scheduler_user": "researcher",
            "matching_r11_jobs": [],
        },
        "evidence_root": str(evidence),
    }
    intent["intent_id"] = seal._identity(intent, "intent_id")
    monkeypatch.setattr(
        seal,
        "_validate_incomplete_evidence",
        lambda *_args: observed_evidence,
    )
    monkeypatch.setattr(
        seal, "_observed_volatile_preimage", lambda *_args: observed_volatile
    )
    monkeypatch.setattr(
        seal,
        "_operator_diagnostic_disposition",
        lambda *_args: diagnostic,
    )
    monkeypatch.setattr(
        seal,
        "_intent_payload",
        lambda *_args: json.loads(json.dumps(intent)),
    )

    dry = seal.seal_failure(
        evidence_root=evidence,
        recovery_root=recovery,
        scheduler_user="researcher",
        apply=False,
    )
    assert dry["action"] == "would_seal"
    assert not evidence.exists()

    monkeypatch.setattr(
        seal, "_execute_proof", lambda *_args: {"sealed_test_proof": True}
    )
    monkeypatch.setattr(
        seal,
        "verify_failure_seal",
        lambda *_args, **_kwargs: {"passed": True},
    )
    applied = seal.seal_failure(
        evidence_root=evidence,
        recovery_root=recovery,
        scheduler_user="researcher",
        apply=True,
    )
    assert applied["action"] == "sealed"
    marker = evidence / seal.MARKER_NAME
    assert marker.is_file()
    assert not stat.S_IMODE(marker.stat().st_mode) & 0o222
    assert not stat.S_IMODE(evidence.stat().st_mode) & 0o222


def test_r11_cleanup_disposition_uses_bound_archive_without_recreating_tmp(
    tmp_path, monkeypatch
):
    recovery = tmp_path / "recovery"
    incomplete = recovery / seal.R9_EVIDENCE_RELATIVE
    archive = incomplete / seal.R9_ARCHIVE_NAME
    archive.mkdir(parents=True)
    volatile = tmp_path / "cleaned-volatile"
    diagnostic = tmp_path / "cleaned-diagnostic"
    monkeypatch.setattr(seal, "R9_VOLATILE_ROOT", volatile)
    monkeypatch.setattr(seal, "OPERATOR_DIAGNOSTIC_ROOT", diagnostic)
    inventory = {"entry_count": 17, "inventory_sha256": "bound"}
    monkeypatch.setattr(
        seal,
        "_validate_volatile_tree",
        lambda _recovery, root: {
            "inventory": inventory,
            "validated_root": str(root),
        },
    )
    observed = {
        "transcript": {
            "sha256": "a" * 64,
            "size": 42,
            "transcript_id": "b" * 64,
        }
    }

    preimage = seal._observed_volatile_preimage(
        recovery, incomplete, observed
    )
    disposition = seal._operator_diagnostic_disposition(observed)

    assert preimage["inventory"] == inventory
    assert preimage["source_relative_path"] == seal.R9_ARCHIVE_NAME
    assert preimage["external_path_present_at_seal"] is False
    assert (
        preimage["loss_classification"]
        == seal.EXTERNAL_CLEANUP_CLASSIFICATION
    )
    assert disposition["present_at_seal"] is False
    assert disposition["authoritative_failure_evidence"] is False
    assert disposition["replacement_evidence"]["sha256"] == "a" * 64
    assert not volatile.exists()
    assert not diagnostic.exists()


def test_r11_cleanup_disposition_rejects_reappearing_external_tree(
    tmp_path, monkeypatch
):
    recovery = tmp_path / "recovery"
    incomplete = recovery / seal.R9_EVIDENCE_RELATIVE
    (incomplete / seal.R9_ARCHIVE_NAME).mkdir(parents=True)
    volatile = tmp_path / "volatile"
    volatile.mkdir()
    monkeypatch.setattr(seal, "R9_VOLATILE_ROOT", volatile)

    with pytest.raises(
        seal.R11FailureSealError,
        match="unexpectedly exists after recorded cleanup",
    ):
        seal._observed_volatile_preimage(
            recovery,
            incomplete,
            {
                "transcript": {
                    "sha256": "a" * 64,
                    "size": 1,
                    "transcript_id": "b" * 64,
                }
            },
        )
