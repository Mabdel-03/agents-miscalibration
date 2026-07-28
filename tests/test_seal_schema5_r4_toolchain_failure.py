from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

from scripts import seal_schema5_r4_toolchain_failure as seal


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    recovery = tmp_path / "results/recovery/schema5-v1"
    transaction = (
        recovery
        / "toolchains"
        / seal.R4_NAMESPACE
        / seal.R4_TRANSACTION_DIRECTORY
    )
    toolchain = (
        recovery
        / "toolchains"
        / seal.R4_NAMESPACE
        / seal.R4_TOOLCHAIN_DIRECTORY
    )
    transaction.mkdir(parents=True)
    toolchain.mkdir()
    (transaction / "provision.lock").write_bytes(b"")
    return recovery, recovery / "prelaunch_failures/schema5-v1.2-r4-toolchain"


def test_seal_is_dry_by_default_marker_last_and_idempotent(
    tmp_path, monkeypatch
):
    recovery, evidence = _roots(tmp_path)

    def snapshot(
        _recovery,
        *,
        scheduler_user,
        require_sealed,
        preserved_scientific_state=None,
    ):
        assert Path(_recovery) == recovery
        assert scheduler_user == "researcher"
        value = {
            "classification": seal.CLASSIFICATION,
            "failure": {"classification": seal.CLASSIFICATION},
            "scientific_state": {
                "result_mutation_count": 0,
                "scheduler_job_count": 0,
            },
            "sealed_read_only": require_sealed,
        }
        if preserved_scientific_state is not None:
            value["scientific_state"] = dict(preserved_scientific_state)
        return value

    monkeypatch.setattr(seal, "_snapshot", snapshot)

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
    marker = evidence / seal.MARKER_NAME
    assert applied["action"] == "sealed"
    assert marker.is_file()
    assert not stat.S_IMODE(marker.stat().st_mode) & 0o222
    assert not stat.S_IMODE(evidence.stat().st_mode) & 0o222
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["classification"] == seal.CLASSIFICATION
    assert payload["evidence"]["sealed_read_only"] is True

    replay = seal.seal_failure(
        evidence_root=evidence,
        recovery_root=recovery,
        scheduler_user="researcher",
        apply=True,
    )
    assert replay["action"] == "already_sealed"
    assert replay["marker_id"] == applied["marker_id"]


def test_verify_rejects_marker_tampering(tmp_path, monkeypatch):
    recovery, evidence = _roots(tmp_path)
    monkeypatch.setattr(
        seal,
        "_snapshot",
        lambda *_args, **kwargs: {
            "classification": seal.CLASSIFICATION,
            "scheduler": {"scheduler_user": "researcher"},
            "scientific_state": {
                "result_mutation_count": 0,
                "scheduler_job_count": 0,
            },
            "sealed_read_only": kwargs["require_sealed"],
        },
    )
    seal.seal_failure(
        evidence_root=evidence,
        recovery_root=recovery,
        scheduler_user="researcher",
        apply=True,
    )
    marker = evidence / seal.MARKER_NAME
    evidence.chmod(0o755)
    marker.chmod(0o644)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["classification"] = "forged"
    marker.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker.chmod(0o444)
    evidence.chmod(0o555)

    with pytest.raises(seal.FailureSealError, match="shape drifted"):
        seal.verify_failure_seal(
            evidence,
            recovery_root=recovery,
            scheduler_user="researcher",
        )


def test_verify_permits_superseding_r5_roots(tmp_path, monkeypatch):
    recovery, evidence = _roots(tmp_path)
    scientific_state = {
        "capture_scope": "pre_r5_scientific_and_operational_roots",
        "captured_before_superseding_release_outputs": True,
        "result_mutation_count": 0,
        "scheduler_job_count": 0,
        "required_absent_paths_at_seal": [],
        "permanently_absent_r4_paths": [],
    }

    def snapshot(
        _recovery,
        *,
        scheduler_user,
        require_sealed,
        preserved_scientific_state=None,
    ):
        return {
            "scheduler": {"scheduler_user": scheduler_user},
            "scientific_state": (
                dict(preserved_scientific_state)
                if preserved_scientific_state is not None
                else dict(scientific_state)
            ),
            "sealed_read_only": require_sealed,
        }

    monkeypatch.setattr(seal, "_snapshot", snapshot)
    seal.seal_failure(
        evidence_root=evidence,
        recovery_root=recovery,
        scheduler_user="researcher",
        apply=True,
    )
    (tmp_path / "results/full_sweep_schema5_v1").mkdir()
    report = seal.verify_failure_seal(
        evidence,
        recovery_root=recovery,
        scheduler_user="researcher",
    )
    assert report["passed"] is True


def test_scientific_state_preserves_prelaunch_capture_and_rejects_r4_execution(
    tmp_path,
):
    recovery = tmp_path / "results/recovery/schema5-v1"
    recovery.mkdir(parents=True)
    captured = seal._scientific_state_evidence(recovery, preserved=None)

    (tmp_path / "results/full_sweep_schema5_v1").mkdir()
    assert (
        seal._scientific_state_evidence(recovery, preserved=captured)
        == captured
    )

    (recovery / "slurm_canaries/schema5-v1.2-r4").mkdir(parents=True)
    with pytest.raises(seal.FailureSealError, match="r4 prelaunch namespace"):
        seal._scientific_state_evidence(recovery, preserved=captured)


def test_forensic_inventory_hashes_raw_symlink_text(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    payload = root / "payload"
    payload.write_bytes(b"evidence\n")
    (root / "external").symlink_to("/outside/prefix/file")

    first = seal._forensic_tree_inventory(root)
    second = seal._forensic_tree_inventory(root)

    assert first == second
    assert first["symlink_count"] == 1
    assert first["symlinks_followed"] is False


def test_forensic_inventory_rejects_external_hardlink(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    external = tmp_path / "external"
    external.write_bytes(b"shared\n")
    os.link(external, root / "shared")

    with pytest.raises(seal.FailureSealError, match="hardlinks outside"):
        seal._forensic_tree_inventory(root)
