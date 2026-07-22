from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

from scripts import create_recovery_snapshot as snapshot
from scripts import retire_legacy_runs as retire


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    results = tmp_path / "results"
    results.mkdir()
    sources = []
    for name in retire.LEGACY_ROOT_NAMES:
        root = results / name
        root.mkdir()
        nested = root / "nested"
        nested.mkdir()
        (nested / "evidence.json").write_text(f"{name}\n", encoding="utf-8")
        sources.append(snapshot.SourceSpec(retire._snapshot_source_name(name), root))
    recovery = results / "recovery" / "schema5-v1"
    recovery.mkdir(parents=True)
    operations = recovery / "operations" / "legacy_consolidation"
    operations.mkdir(parents=True)
    (operations / "legacy_semantic_audit_report.json").write_text("{}\n")
    cleanup = recovery / retire.LEGACY_CLEANUP_MARKER
    cleanup.write_text('{"passed":true}\n')
    sources.extend(
        (
            snapshot.SourceSpec("legacy_cleanup_evidence", operations),
            snapshot.SourceSpec("legacy_cleanup_complete", cleanup),
        )
    )
    consolidated = recovery / "legacy_consolidated"
    snapshot.create_snapshot(consolidated, sources, apply=True)
    snapshot.write_snapshot_attestation(
        consolidated, recovery / "legacy_consolidated.attestation.json"
    )
    return results, recovery


def _make_writable(root: Path) -> None:
    for current, directories, files in os.walk(root):
        os.chmod(current, 0o755)
        for name in files:
            os.chmod(Path(current) / name, 0o644)
        for name in directories:
            os.chmod(Path(current) / name, 0o755)


def test_retirement_is_dry_run_by_default_and_then_idempotent(tmp_path):
    results, recovery = _fixture(tmp_path)
    try:
        dry = retire.retire(results_root=results, recovery_root=recovery, apply=False)
        assert dry["status"] == "dry_run"
        assert set(dry["node_counts"]) == set(retire.LEGACY_ROOT_NAMES)
        assert all(
            stat.S_IMODE((results / name).stat().st_mode) & 0o200
            for name in retire.LEGACY_ROOT_NAMES
        )

        applied = retire.retire(results_root=results, recovery_root=recovery, apply=True)
        assert applied["status"] == "retired"
        for name in retire.LEGACY_ROOT_NAMES:
            root = results / name
            assert (root / retire.ROOT_MARKER).is_file()
            assert not stat.S_IMODE(root.stat().st_mode) & 0o222
            assert not stat.S_IMODE((root / "nested/evidence.json").stat().st_mode) & 0o222

        again = retire.retire(results_root=results, recovery_root=recovery, apply=True)
        assert again["status"] == "already_retired"
        assert again["snapshot_id"] == applied["snapshot_id"]
    finally:
        for name in retire.LEGACY_ROOT_NAMES:
            _make_writable(results / name)


def test_retirement_rejects_attestation_for_another_snapshot_identity(tmp_path):
    results, recovery = _fixture(tmp_path)
    attestation = recovery / "legacy_consolidated.attestation.json"
    os.chmod(attestation, 0o644)
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    payload["snapshot_id"] = "0" * 64
    attestation.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(retire.RetirementError, match="wrong identity"):
        retire.retire(results_root=results, recovery_root=recovery, apply=False)
