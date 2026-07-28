from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess

import pytest

from scripts import seal_schema5_r5_prelaunch_failure as seal


_TEST_INVENTORY = {
    "exists": True,
    "entries": [],
    "inventory_sha256": "a" * 64,
}


def _intent(tmp_path: Path, recovery: Path) -> dict:
    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    payload = {
        "schema_version": seal.SCHEMA_VERSION,
        "protocol": f"{seal.PROTOCOL}-intent",
        "classification": seal.CLASSIFICATION,
        "source_release": {},
        "sealed_r5_toolchain": {},
        "command": {
            "argv": ["/sealed/dev-python", "record-prelaunch-attempt"],
            "cwd": str(tmp_path),
            "probe_root": str(probe_root),
            "expected_envelope": str(probe_root / "never-created.json"),
            "environment": {"PATH": "/usr/bin:/bin"},
        },
        "original_probe_tree": dict(_TEST_INVENTORY),
        "scientific_state": {
            "captured_before_r6_outputs": True,
            "result_mutation_count": 0,
            "scheduler_job_count": 0,
            "required_absent_paths_at_seal": [],
            "permanently_absent_r5_execution_paths": [],
        },
        "scheduler": {
            "scheduler_user": "researcher",
            "matching_r5_jobs": [],
        },
    }
    payload["intent_id"] = seal._identity(payload, "intent_id")
    return payload


def test_seal_is_dry_by_default_and_marker_last(tmp_path, monkeypatch):
    recovery = tmp_path / "results/recovery/schema5-v1"
    recovery.mkdir(parents=True)
    evidence = recovery / seal.EVIDENCE_RELATIVE_ROOT
    intent = _intent(tmp_path, recovery)
    original_verify = seal.verify_failure_seal
    monkeypatch.setattr(
        seal,
        "_intent_payload",
        lambda _recovery, _user: json.loads(json.dumps(intent)),
    )

    dry = seal.seal_failure(
        evidence_root=evidence,
        recovery_root=recovery,
        scheduler_user="researcher",
        apply=False,
    )
    assert dry["action"] == "would_seal"
    assert not evidence.exists()

    inventory = dict(_TEST_INVENTORY)
    monkeypatch.setattr(
        seal, "_probe_tree_inventory", lambda _root: dict(inventory)
    )
    monkeypatch.setattr(
        seal.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=b"",
            stderr=(
                f"{seal.EXPECTED_DISPATCH_LINE}\n"
                f"{seal.EXPECTED_EXCEPTION}\n"
            ).encode(),
        ),
    )
    monkeypatch.setattr(
        seal,
        "verify_failure_seal",
        lambda *_args, **_kwargs: {
            "passed": True,
            "marker_id": json.loads(
                (evidence / seal.MARKER_NAME).read_text(encoding="utf-8")
            )["marker_id"],
        },
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
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["attempt"]["returncode"] == 1
    assert payload["attempt"]["probe_tree_before"] == inventory
    assert payload["attempt"]["probe_tree_after"] == inventory
    assert payload["attempt"]["probe_tree_before"] == intent["original_probe_tree"]

    different_inventory = {
        **inventory,
        "inventory_sha256": "b" * 64,
    }
    payload["attempt"]["probe_tree_before"] = different_inventory
    payload["attempt"]["probe_tree_after"] = different_inventory
    payload["marker_id"] = seal._identity(payload, "marker_id")
    marker.chmod(0o600)
    marker.write_bytes(seal._canonical_bytes(payload))
    marker.chmod(0o444)
    monkeypatch.setattr(seal, "_validate_intent", lambda *_args, **_kwargs: None)
    with pytest.raises(
        seal.R5FailureSealError,
        match="signature drifted",
    ):
        original_verify(
            evidence,
            recovery_root=recovery,
            scheduler_user="researcher",
        )


def test_command_contract_reproduces_the_r5_argparse_collision(tmp_path):
    recovery = tmp_path / "results/recovery/schema5-v1"
    contract = seal._command_contract(recovery, "researcher")
    argv = contract["argv"]

    assert "record-prelaunch-attempt" in argv
    assert argv[argv.index("--command") + 1].endswith(
        "/toolchains/r5/conda/base/bin/python"
    )
    assert contract["expected_envelope"].endswith(
        "FAILURE_UNSAFE_RECORDED_BROKEN_INTERNAL_SYMLINK.source.json"
    )
