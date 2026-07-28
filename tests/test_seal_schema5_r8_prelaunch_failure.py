from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest

from scripts import seal_schema5_r8_prelaunch_failure as seal


def test_r8_failure_command_binds_exact_outer_and_inner_probes(tmp_path):
    recovery = tmp_path / "results/recovery/schema5-v1"
    contract = seal._command_contract(recovery, "researcher")
    probe = Path(contract["probe_root"])

    assert contract["outer_argv"][:4] == [
        str(seal.DEV_PYTHON),
        "-I",
        "-B",
        str(
            recovery
            / seal.R8_CHECKOUT
            / "scripts/seal_recovery_evidence.py"
        ),
    ]
    assert contract["outer_argv"][-len(contract["inner_argv"]) - 1] == "--command"
    assert contract["inner_argv"] == [
        str(recovery / seal.R8_TOOLCHAIN_RELATIVE / "base/bin/python"),
        "-I",
        "-B",
        str(
            recovery
            / seal.R3_CHECKOUT
            / "scripts/run_schema5_materialization_pilot.py"
        ),
        "conda-runtime-identity",
        "--conda-executable",
        str(seal.SHARED_CONDA_BASE / "bin/conda"),
    ]
    assert contract["expected_envelope"] == str(
        probe / seal.BROKEN_ENVELOPE_NAME
    )
    assert contract["expected_outer_stderr"] == seal.EXPECTED_RECORDER_ERROR


def test_r8_inner_traceback_requires_exact_r3_import_boundary(tmp_path):
    recovery = tmp_path / "results/recovery/schema5-v1"
    r3 = recovery / seal.R3_CHECKOUT
    traceback = (
        "Traceback (most recent call last):\n"
        f'  File "{r3}/scripts/run_schema5_materialization_pilot.py", '
        'line 63, in <module>\n'
        "    from scripts import freeze_schema5_release as freeze\n"
        f'  File "{r3}/src/agents_scaling/serving/client.py", '
        'line 26, in <module>\n'
        "    import backoff\n"
        "ModuleNotFoundError: No module named 'backoff'\n"
    )

    seal._validate_inner_stderr(traceback, recovery)
    with pytest.raises(seal.R8FailureSealError, match="signature drifted"):
        seal._validate_inner_stderr(
            traceback.replace("'backoff'", "'different'"),
            recovery,
        )


def test_r8_probe_tree_must_remain_empty(tmp_path, monkeypatch):
    probe = tmp_path / "probe"
    probe.mkdir(mode=0o700)
    monkeypatch.setattr(seal, "_probe_root", lambda _user: probe)

    report = seal._empty_probe_tree("researcher")
    assert report["entries"] == []
    assert report["mode"] == 0o700

    (probe / seal.BROKEN_ENVELOPE_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(seal.R8FailureSealError, match="is not empty"):
        seal._empty_probe_tree("researcher")


def test_r8_failure_seal_is_dry_by_default_and_marker_last(
    tmp_path, monkeypatch
):
    recovery = tmp_path / "results/recovery/schema5-v1"
    recovery.mkdir(parents=True)
    evidence = recovery / seal.EVIDENCE_RELATIVE_ROOT
    intent = {
        "schema_version": seal.SCHEMA_VERSION,
        "protocol": f"{seal.PROTOCOL}-intent",
        "classification": seal.CLASSIFICATION,
        "source_release": {},
        "sealed_r8_toolchain": {},
        "commands": {},
        "scientific_state": {},
        "scheduler": {"scheduler_user": "researcher"},
        "evidence_root": str(evidence),
    }
    intent["intent_id"] = seal._identity(intent, "intent_id")
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
        seal,
        "_execute_proof",
        lambda *_args: {"sealed_test_proof": True},
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
