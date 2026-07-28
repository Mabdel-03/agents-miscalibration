from __future__ import annotations

import json
from pathlib import Path
import stat

from scripts import seal_schema5_r12_prelaunch_failure as seal


def test_r12_verifier_reproduction_command_is_exact(tmp_path):
    recovery = tmp_path / "results/recovery/schema5-v1"
    evidence = recovery / seal.R9_EVIDENCE_RELATIVE

    argv = seal._reproduction_argv(recovery, evidence, "researcher")

    assert argv == [
        seal.sys.executable,
        "-I",
        str(
            recovery
            / seal.R12_CHECKOUT
            / "scripts/seal_schema5_r9_prelaunch_failure.py"
        ),
        "verify",
        "--evidence-root",
        str(evidence),
        "--recovery-root",
        str(recovery),
        "--scheduler-user",
        "researcher",
    ]
    assert seal.EXPECTED_STDERR == (
        "ERROR: exact r9 recorder reproduction binding drifted\n"
    )


def test_r12_failure_seal_is_dry_by_default_and_marker_last(
    tmp_path, monkeypatch
):
    recovery = tmp_path / "results/recovery/schema5-v1"
    recovery.mkdir(parents=True)
    evidence = recovery / seal.R12_FAILURE_RELATIVE
    observed = {
        "inventory": {"entry_count": 87, "inventory_sha256": "a" * 64},
        "complete_file_reference_keys": [
            "link_count",
            "mode",
            "path",
            "sha256",
            "size",
            "transcript_id",
        ],
        "buggy_expected_reference_keys": [
            "path",
            "sha256",
            "size",
            "transcript_id",
        ],
        "immutable_r12_rejected": True,
        "reproduced_state_matches_transcript": True,
    }
    intent = {
        "schema_version": seal.SCHEMA_VERSION,
        "protocol": f"{seal.PROTOCOL}-intent",
        "classification": seal.CLASSIFICATION,
        "source_release": {},
        "observed_invalid_r9_transaction": observed,
        "reproduction": {},
        "scientific_state": {
            "captured_before_r13_production_outputs": True,
            "result_mutation_count": 0,
            "scheduler_job_count": 0,
        },
        "scheduler": {
            "scheduler_user": "researcher",
            "matching_r12_jobs": [],
        },
        "evidence_root": str(evidence),
    }
    intent["intent_id"] = seal._identity(intent, "intent_id")
    monkeypatch.setattr(
        seal,
        "_validate_invalid_transaction",
        lambda *_args: json.loads(json.dumps(observed)),
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
    assert dry["classification"] == seal.CLASSIFICATION
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


def test_r12_classification_names_only_the_complete_reference_mismatch():
    assert seal.CLASSIFICATION == (
        "deterministic_prelaunch_r9_transcript_file_reference_verifier_mismatch"
    )
    assert seal.PROTOCOL == (
        "schema5-v1.2-r12-r9-verifier-file-reference-failure-v1"
    )
    assert seal.R12_COMMIT == "a91616c412b9e36a87d8232511a1ca025f8f79de"
    assert seal.R12_TAG_OBJECT == "0615b17e4027d344fed53da8259345a2383bd33f"
