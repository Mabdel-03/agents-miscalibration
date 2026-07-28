from __future__ import annotations

import json
from pathlib import Path
import stat

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
        seal, "_validate_volatile_tree", lambda *_args: observed_volatile
    )
    monkeypatch.setattr(
        seal._base,
        "_validate_operator_diagnostic",
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
