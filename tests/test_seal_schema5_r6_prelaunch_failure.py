from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess

import pytest

from scripts import seal_schema5_r6_prelaunch_failure as seal


_EMPTY_INVENTORY = {
    "exists": True,
    "entries": [],
    "inventory_sha256": seal.hashlib.sha256(
        seal._canonical_bytes([])
    ).hexdigest(),
}


def _binding(tmp_path: Path) -> dict:
    fields = seal.R6_EXPECTED_BINDING_FIELDS | seal.ACTUAL_ADDITIVE_FIELDS
    payload = {field: None for field in fields}
    payload.update(
        {
            "protocol": "schema5-v1.2-r6-offline-conda-toolchain-v1",
            "release_tag": seal.R6_TAG,
            "chain_namespace": seal.R6_NAMESPACE,
            "toolchain_root": str(tmp_path / "toolchain"),
            "base_prefix": str(tmp_path / "toolchain/base"),
            "portable_shebang": {
                "interpreter": str(tmp_path / "toolchain/base/bin/python")
            },
        }
    )
    return payload


def _intent(tmp_path: Path, recovery: Path) -> dict:
    probe_root = tmp_path / "probe"
    probe_root.mkdir()
    toolchain = _binding(tmp_path)
    payload = {
        "schema_version": seal.SCHEMA_VERSION,
        "protocol": f"{seal.PROTOCOL}-intent",
        "classification": seal.CLASSIFICATION,
        "source_release": {},
        "sealed_r6_toolchain": toolchain,
        "field_mismatch": seal._field_mismatch(toolchain),
        "command": {
            "argv": ["/sealed/dev-python", "record-prelaunch-attempt"],
            "cwd": str(tmp_path),
            "probe_root": str(probe_root),
            "expected_envelope": str(probe_root / "never-created.json"),
            "environment": {"PATH": "/usr/bin:/bin"},
            "timeout_seconds": 1800,
        },
        "original_probe_tree": dict(_EMPTY_INVENTORY),
        "scientific_state": {
            "captured_before_r7_outputs": True,
            "result_mutation_count": 0,
            "scheduler_job_count": 0,
            "required_absent_paths_at_seal": [],
            "permanently_absent_r6_execution_paths": [],
        },
        "scheduler": {
            "scheduler_user": "researcher",
            "matching_r6_jobs": [],
        },
    }
    payload["intent_id"] = seal._identity(payload, "intent_id")
    return payload


def test_r6_failure_seal_is_marker_last_and_binds_original_empty_tree(
    tmp_path, monkeypatch
):
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
    monkeypatch.setattr(
        seal,
        "_probe_tree_inventory",
        lambda _root: dict(_EMPTY_INVENTORY),
    )
    monkeypatch.setattr(
        seal.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=seal.EXPECTED_RETURN_CODE,
            stdout=b"",
            stderr=seal.EXPECTED_STDERR.encode(),
        ),
    )
    monkeypatch.setattr(
        seal,
        "verify_failure_seal",
        lambda *_args, **_kwargs: {"passed": True},
    )

    dry = seal.seal_failure(
        evidence_root=evidence,
        recovery_root=recovery,
        scheduler_user="researcher",
        apply=False,
    )
    assert dry["action"] == "would_seal"
    assert not evidence.exists()

    applied = seal.seal_failure(
        evidence_root=evidence,
        recovery_root=recovery,
        scheduler_user="researcher",
        apply=True,
    )
    assert applied["action"] == "sealed"
    marker_path = evidence / seal.MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["attempt"]["returncode"] == seal.EXPECTED_RETURN_CODE
    assert marker["attempt"]["probe_tree_before"] == intent["original_probe_tree"]
    assert marker["attempt"]["probe_tree_before"] == marker["attempt"]["probe_tree_after"]
    assert not stat.S_IMODE(marker_path.stat().st_mode) & 0o222
    assert not stat.S_IMODE(evidence.stat().st_mode) & 0o222

    marker["attempt"]["probe_tree_before"]["inventory_sha256"] = "f" * 64
    marker["attempt"]["probe_tree_after"]["inventory_sha256"] = "f" * 64
    marker["marker_id"] = seal._identity(marker, "marker_id")
    marker_path.chmod(0o600)
    marker_path.write_bytes(seal._canonical_bytes(marker))
    marker_path.chmod(0o444)
    monkeypatch.setattr(seal, "_validate_intent", lambda *_args, **_kwargs: None)
    with pytest.raises(seal.R6FailureSealError, match="signature drifted"):
        original_verify(
            evidence,
            recovery_root=recovery,
            scheduler_user="researcher",
        )


def test_r6_failure_classification_is_exactly_one_additive_field(tmp_path):
    binding = _binding(tmp_path)
    mismatch = seal._field_mismatch(binding)
    assert mismatch["unexpected_fields"] == ["portable_shebang"]
    assert mismatch["missing_fields"] == []

    binding["another_field"] = {}
    with pytest.raises(seal.R6FailureSealError, match="classification drifted"):
        seal._field_mismatch(binding)


def test_r6_failure_command_reproduces_dry_run_only(tmp_path):
    recovery = tmp_path / "results/recovery/schema5-v1"
    contract = seal._command_contract(recovery, "researcher")
    argv = contract["argv"]

    assert "record-prelaunch-attempt" in argv
    assert "--apply" not in argv
    assert "tagged-r6-release-checkout" in argv
    assert "sealed-r6-conda-toolchain" in argv
    assert argv[argv.index("--command") + 1].endswith(
        "/toolchains/r6/conda/base/bin/python"
    )
