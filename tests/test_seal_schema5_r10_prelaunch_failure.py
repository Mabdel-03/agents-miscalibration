from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest

from scripts import seal_schema5_r10_prelaunch_failure as seal


def test_r10_reproduction_command_is_exact(tmp_path):
    recovery = tmp_path / "results/recovery/schema5-v1"

    argv = seal._r10_reproduction_argv(recovery, "researcher")

    assert argv[:3] == [seal.sys.executable, "-I", str(
        recovery
        / seal.R10_CHECKOUT
        / "scripts/seal_schema5_r9_prelaunch_failure.py"
    )]
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
        "ERROR: exact r9 offline recorder rejection reproduction drifted\n"
    )


def test_operator_diagnostic_is_explicitly_non_authoritative(tmp_path):
    root = tmp_path / "diagnostic"
    cache = root / "empty-conda-pkgs"
    cache.mkdir(parents=True)
    (cache / "python.conda.partial").touch()
    destination = root / "offline-clone-destination"
    destination.mkdir()

    report = seal._validate_operator_diagnostic(root)

    assert report["authoritative_failure_evidence"] is False
    assert report["partial_artifact_count"] == 1
    assert report["inventory"]["entry_count"] == 4
    (root / "FAILURE_OFFLINE_CLONE_UNSEEDED_RELEASE_LOCAL_CACHE.source.json").touch()
    with pytest.raises(
        seal.R10FailureSealError, match="unexpectedly published an envelope"
    ):
        seal._validate_operator_diagnostic(root)


def test_volatile_archive_is_exact_and_read_only(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_bytes(b"evidence")
    (source / "link").symlink_to("payload")
    quarantine = tmp_path / "quarantine"
    archive = tmp_path / "archive"

    report = seal._archive_volatile(source, quarantine, archive)

    assert not source.exists()
    assert report["path"] == str(archive)
    assert report["inventory"] == seal._portable_inventory(archive)
    assert report["inventory"] == seal._portable_inventory(quarantine)
    seal._require_recursively_read_only(archive)
    seal._require_recursively_read_only(quarantine)


def test_r10_failure_seal_is_dry_by_default_and_marker_last(
    tmp_path, monkeypatch
):
    recovery = tmp_path / "results/recovery/schema5-v1"
    recovery.mkdir(parents=True)
    evidence = recovery / seal.R10_FAILURE_RELATIVE
    observed_evidence = {"inventory": {"entry_count": 2}}
    observed_volatile = {"inventory": {"entry_count": 9}}
    diagnostic = {
        "inventory": {"entry_count": 8},
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
        "reproduction": {},
        "scientific_state": {
            "captured_before_r11_production_outputs": True,
            "result_mutation_count": 0,
            "scheduler_job_count": 0,
        },
        "scheduler": {
            "scheduler_user": "researcher",
            "matching_r10_jobs": [],
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
        seal, "_validate_volatile_tree", lambda *_args: observed_volatile
    )
    monkeypatch.setattr(
        seal, "_validate_operator_diagnostic", lambda *_args: diagnostic
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
