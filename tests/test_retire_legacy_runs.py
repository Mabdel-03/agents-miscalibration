from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from scripts import create_recovery_snapshot as snapshot
from scripts import retire_legacy_runs as retire


@pytest.fixture(autouse=True)
def _quiet_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "test-user")
    monkeypatch.setattr(
        retire.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )


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
        if name == retire.LEGACY_ROOT_NAMES[0]:
            lock = root / "cells" / "cell-a" / ".cell.lock"
            lock.parent.mkdir(parents=True)
            lock.touch()
        sources.append(snapshot.SourceSpec(retire._snapshot_source_name(name), root))
    recovery = results / "recovery" / "schema5-v1"
    recovery.mkdir(parents=True)
    (recovery / retire.MAINTENANCE_INTERLOCK).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "desired_state": "maintenance",
                "admission_enabled": False,
                "retired_run_ids": list(retire.LEGACY_ROOT_NAMES[:3]),
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
            marker = root / retire.ROOT_MARKER
            assert marker.is_file()
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
            assert marker_payload["source_verification"] == applied[
                "source_verification"
            ][name]
            assert len(
                marker_payload["source_verification"][
                    "source_directory_inventory_sha256"
                ]
            ) == 64
            assert not stat.S_IMODE(root.stat().st_mode) & 0o222
            assert not stat.S_IMODE((root / "nested/evidence.json").stat().st_mode) & 0o222

        again = retire.retire(results_root=results, recovery_root=recovery, apply=True)
        assert again["status"] == "already_retired"
        assert again["snapshot_id"] == applied["snapshot_id"]
    finally:
        for name in retire.LEGACY_ROOT_NAMES:
            _make_writable(results / name)


def test_apply_refuses_concurrent_global_retirement_lock(tmp_path: Path) -> None:
    results, recovery = _fixture(tmp_path)
    lock_path = recovery / retire.RETIREMENT_LOCK
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(retire.RetirementError, match="another legacy retirement"):
            retire.retire(
                results_root=results,
                recovery_root=recovery,
                apply=True,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_open_writer_during_chmod_is_caught_by_final_snapshot_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, recovery = _fixture(tmp_path)
    target = results / retire.LEGACY_ROOT_NAMES[0] / "nested" / "evidence.json"
    descriptor = os.open(target, os.O_WRONLY)
    real_verify = retire._verify_live_root
    calls = 0

    def verify_with_late_write(root: Path, inventory):
        nonlocal calls
        calls += 1
        # Four pre-chmod root hashes precede the first final post-chmod hash.
        if calls == len(retire.LEGACY_ROOT_NAMES) + 1:
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.write(descriptor, b"X")
            os.fsync(descriptor)
        return real_verify(root, inventory)

    monkeypatch.setattr(retire, "_verify_live_root", verify_with_late_write)
    try:
        with pytest.raises(retire.RetirementError, match="bytes drifted"):
            retire.retire(
                results_root=results,
                recovery_root=recovery,
                apply=True,
            )
        assert not (recovery / retire.COMPLETE_MARKER).exists()
    finally:
        os.close(descriptor)
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


def test_retirement_rejects_live_byte_drift_after_consolidated_snapshot(tmp_path):
    results, recovery = _fixture(tmp_path)
    changed = results / retire.LEGACY_ROOT_NAMES[0] / "nested" / "evidence.json"
    changed.write_text("changed after snapshot\n", encoding="utf-8")

    with pytest.raises(retire.RetirementError, match="bytes drifted"):
        retire.retire(results_root=results, recovery_root=recovery, apply=False)


def test_completed_retirement_revalidates_root_marker_and_content(tmp_path):
    results, recovery = _fixture(tmp_path)
    try:
        retire.retire(results_root=results, recovery_root=recovery, apply=True)
        root = results / retire.LEGACY_ROOT_NAMES[0]
        marker = root / retire.ROOT_MARKER
        os.chmod(root, 0o755)
        os.chmod(marker, 0o644)
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        marker_payload["snapshot_id"] = "drifted"
        marker.write_text(json.dumps(marker_payload) + "\n", encoding="utf-8")
        os.chmod(marker, 0o444)

        with pytest.raises(retire.RetirementError, match="retirement marker drifted"):
            retire.retire(results_root=results, recovery_root=recovery, apply=True)

        # Restore the exact marker, then prove payload drift is independently caught.
        os.chmod(marker, 0o644)
        proof = json.loads(
            (recovery / retire.COMPLETE_MARKER).read_text(encoding="utf-8")
        )["source_verification"][root.name]
        expected = retire._root_intent(
            root=root,
            snapshot_id=json.loads(
                (recovery / retire.COMPLETE_MARKER).read_text(encoding="utf-8")
            )["snapshot_id"],
            attestation_sha256=json.loads(
                (recovery / retire.COMPLETE_MARKER).read_text(encoding="utf-8")
            )["snapshot_attestation_sha256"],
            source_verification=proof,
        )
        marker.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        os.chmod(marker, 0o444)
        evidence = root / "nested" / "evidence.json"
        os.chmod(root / "nested", 0o755)
        os.chmod(evidence, 0o644)
        evidence.write_text("content drift\n", encoding="utf-8")

        with pytest.raises(retire.RetirementError, match="bytes drifted"):
            retire.retire(results_root=results, recovery_root=recovery, apply=True)
    finally:
        for name in retire.LEGACY_ROOT_NAMES:
            _make_writable(results / name)


def test_retirement_precheck_rejects_scheduler_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, _recovery = _fixture(tmp_path)
    monkeypatch.setattr(
        retire.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="123|asys-cells|RUNNING|full_sweep_v1\n",
            stderr="",
        ),
    )

    with pytest.raises(retire.RetirementError, match="legacy jobs remain live"):
        retire._maintenance_precheck(results)


def test_retirement_precheck_requires_exact_maintenance_interlock(tmp_path: Path) -> None:
    results, recovery = _fixture(tmp_path)
    interlock = recovery / retire.MAINTENANCE_INTERLOCK
    payload = json.loads(interlock.read_text(encoding="utf-8"))
    payload["admission_enabled"] = True
    interlock.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(retire.RetirementError, match="does not freeze the exact"):
        retire._maintenance_precheck(results, recovery)


def test_retirement_precheck_rejects_held_cell_lock(tmp_path: Path) -> None:
    results = tmp_path / "results"
    recovery = results / "recovery" / "schema5-v1"
    recovery.mkdir(parents=True)
    (recovery / retire.MAINTENANCE_INTERLOCK).write_text(
        json.dumps(
            {
                "desired_state": "maintenance",
                "admission_enabled": False,
                "retired_run_ids": list(retire.LEGACY_ROOT_NAMES[:3]),
            }
        )
        + "\n"
    )
    lock = results / retire.LEGACY_ROOT_NAMES[0] / "cells" / "cell-a" / ".cell.lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    descriptor = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(retire.RetirementError, match="held legacy cell locks"):
            retire._maintenance_precheck(results)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_apply_rechecks_quiescence_immediately_before_each_root_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, recovery = _fixture(tmp_path)
    events: list[tuple[str, str | None]] = []
    real_precheck = retire._maintenance_precheck
    real_chmod = retire.os.chmod

    def recording_precheck(
        results_root: Path,
        recovery_root: Path | None = None,
        *,
        owned_lock_paths=frozenset(),
    ):
        events.append(("precheck", None))
        return real_precheck(
            results_root,
            recovery_root,
            owned_lock_paths=owned_lock_paths,
        )

    def recording_chmod(path, mode):
        events.append(("chmod", str(path)))
        return real_chmod(path, mode)

    monkeypatch.setattr(retire, "_maintenance_precheck", recording_precheck)
    monkeypatch.setattr(retire.os, "chmod", recording_chmod)
    try:
        retire.retire(results_root=results, recovery_root=recovery, apply=True)
        # Initial audit, one final-boundary check per root, and one post-hash check.
        assert sum(kind == "precheck" for kind, _path in events) == 6

        start = 0
        for root_name in retire.LEGACY_ROOT_NAMES:
            root = str(results / root_name)
            root_chmod = next(
                index
                for index in range(start, len(events))
                if events[index] == ("chmod", root)
            )
            first_chmod = next(
                index
                for index in range(start, root_chmod + 1)
                if events[index][0] == "chmod"
            )
            assert events[first_chmod - 1] == ("precheck", None)
            start = root_chmod + 1
        assert events[-1] == ("precheck", None)
    finally:
        monkeypatch.setattr(retire.os, "chmod", real_chmod)
        for name in retire.LEGACY_ROOT_NAMES:
            _make_writable(results / name)


def test_completed_retirement_rejects_extra_or_inconsistent_completion_fields(
    tmp_path: Path,
) -> None:
    results, recovery = _fixture(tmp_path)
    try:
        retire.retire(results_root=results, recovery_root=recovery, apply=True)
        complete = recovery / retire.COMPLETE_MARKER
        payload = json.loads(complete.read_text(encoding="utf-8"))
        os.chmod(complete, 0o644)
        payload["unexpected"] = "not allowed"
        complete.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        os.chmod(complete, 0o444)

        with pytest.raises(retire.RetirementError, match="wrong fields"):
            retire.retire(results_root=results, recovery_root=recovery, apply=True)

        os.chmod(complete, 0o644)
        payload.pop("unexpected")
        payload["verified_nodes"] += 1
        complete.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        os.chmod(complete, 0o444)

        with pytest.raises(retire.RetirementError, match="verified-node count drifted"):
            retire.retire(results_root=results, recovery_root=recovery, apply=True)
    finally:
        for name in retire.LEGACY_ROOT_NAMES:
            _make_writable(results / name)


def test_completed_retirement_rejects_tampered_precheck_history(tmp_path: Path) -> None:
    results, recovery = _fixture(tmp_path)
    try:
        retire.retire(results_root=results, recovery_root=recovery, apply=True)
        complete = recovery / retire.COMPLETE_MARKER
        payload = json.loads(complete.read_text(encoding="utf-8"))
        os.chmod(complete, 0o644)
        payload["seal_prechecks"][retire.LEGACY_ROOT_NAMES[0]][
            "held_cell_locks"
        ] = 1
        complete.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        os.chmod(complete, 0o444)

        with pytest.raises(retire.RetirementError, match="did not prove quiescence"):
            retire.retire(results_root=results, recovery_root=recovery, apply=True)
    finally:
        for name in retire.LEGACY_ROOT_NAMES:
            _make_writable(results / name)


def test_completed_retirement_validation_does_not_require_current_scheduler_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results, recovery = _fixture(tmp_path)
    try:
        retire.retire(results_root=results, recovery_root=recovery, apply=True)

        def should_not_query_scheduler(*_args, **_kwargs):
            raise AssertionError("completed retirement must validate sealed history")

        monkeypatch.setattr(retire.subprocess, "run", should_not_query_scheduler)
        again = retire.retire(
            results_root=results, recovery_root=recovery, apply=True
        )
        assert again["status"] == "already_retired"
    finally:
        for name in retire.LEGACY_ROOT_NAMES:
            _make_writable(results / name)
