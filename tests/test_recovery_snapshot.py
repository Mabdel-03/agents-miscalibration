"""Focused safety tests for the byte-copy recovery snapshot."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from scripts import create_recovery_snapshot as snapshot
from scripts import restore_recovery_snapshot as restore


def _sources(tmp_path: Path) -> tuple[list[snapshot.SourceSpec], Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    (run / "cells.json").write_bytes(b"manifest bytes\n")
    nested = run / "cells" / "cell-a"
    nested.mkdir(parents=True)
    (nested / "results.jsonl").write_bytes(b'{"qid":"q0"}\n')
    metadata = tmp_path / "scheduler.txt"
    metadata.write_bytes(b"no jobs\n")
    sources = [
        snapshot.SourceSpec("run", run),
        snapshot.SourceSpec("scheduler", metadata),
    ]
    return sources, run, metadata


def test_dry_run_is_read_only_and_reports_exact_file_bytes(tmp_path):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"

    report = snapshot.create_snapshot(destination, sources, apply=False)

    assert report["status"] == "dry_run"
    assert report["file_count"] == 3
    assert report["total_bytes"] == sum(
        (
            spec.path.stat().st_size
            if spec.path.is_file()
            else sum(
                path.stat().st_size for path in spec.path.rglob("*") if path.is_file()
            )
        )
        for spec in sources
    )
    assert not destination.exists()


def test_apply_real_copies_verifies_seals_and_is_idempotent(tmp_path):
    sources, run, metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"

    report = snapshot.create_snapshot(destination, sources, apply=True)

    assert report["status"] == "created"
    assert report["verified"] is True
    assert report["file_count"] == 3
    marker = destination / snapshot.COMPLETE_MARKER_FILENAME
    assert marker.is_file()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert stat.S_IMODE(marker.stat().st_mode) == 0o444
    assert (destination / snapshot.SOURCE_INVENTORY_FILENAME).read_bytes() == (
        destination / snapshot.COPY_INVENTORY_FILENAME
    ).read_bytes()

    pairs = (
        (run / "cells.json", destination / "run/cells.json"),
        (
            run / "cells/cell-a/results.jsonl",
            destination / "run/cells/cell-a/results.jsonl",
        ),
        (metadata, destination / "scheduler"),
    )
    for source, copied in pairs:
        assert copied.read_bytes() == source.read_bytes()
        source_info = source.stat()
        copied_info = copied.stat()
        assert (source_info.st_dev, source_info.st_ino) != (
            copied_info.st_dev,
            copied_info.st_ino,
        )
        assert copied_info.st_nlink == 1

    again = snapshot.create_snapshot(destination, sources, apply=True)
    assert again["status"] == "already_complete"
    assert again["snapshot_id"] == report["snapshot_id"]


def test_resume_reuses_verified_file_and_completes(tmp_path):
    sources, run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    partial = destination / "run"
    partial.mkdir(parents=True)
    copied = partial / "cells.json"
    copied.write_bytes((run / "cells.json").read_bytes())

    report = snapshot.create_snapshot(destination, sources, apply=True)

    assert report["status"] == "created"
    assert (destination / "run/cells.json").read_bytes() == (
        run / "cells.json"
    ).read_bytes()


def test_kill_resume_removes_exact_copy_temporary_before_source_inventory(
    tmp_path, monkeypatch
):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    partial = destination / "run"
    partial.mkdir(parents=True)
    orphan = partial / ".cells.json.sdf7y855.copying"
    orphan.write_bytes(b"")
    original_inventory = snapshot._inventory_sources
    original_fsync = snapshot.os.fsync
    inventory_started = False
    cleanup_directory_fsynced = False

    def checked_fsync(descriptor):
        nonlocal cleanup_directory_fsynced
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            cleanup_directory_fsynced = True
        return original_fsync(descriptor)

    def checked_inventory(source_specs):
        nonlocal inventory_started
        inventory_started = True
        assert not orphan.exists()
        assert cleanup_directory_fsynced is True
        return original_inventory(source_specs)

    monkeypatch.setattr(snapshot.os, "fsync", checked_fsync)
    monkeypatch.setattr(snapshot, "_inventory_sources", checked_inventory)

    report = snapshot.create_snapshot(destination, sources, apply=True)

    assert inventory_started is True
    assert report["status"] == "created"
    assert (destination / "run/cells.json").read_bytes() == b"manifest bytes\n"


@pytest.mark.parametrize("control_name", sorted(snapshot._CONTROL_FILENAMES))
def test_kill_resume_removes_exact_control_temporary(tmp_path, control_name):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    destination.mkdir()
    orphan = destination / f".{control_name}.deadbeef"
    orphan.write_bytes(b"partial publication")

    report = snapshot.create_snapshot(destination, sources, apply=True)

    assert report["status"] == "created"
    assert not orphan.exists()
    assert (destination / control_name).is_file()


@pytest.mark.parametrize(
    "name",
    (
        ".cells.json.short.copying",
        ".cells.json.DEADBEEF.copying",
        ".cells.json.deadbeef.copying.extra",
        ".not-a-source.deadbeef.copying",
    ),
)
def test_temporary_lookalikes_are_rejected_and_preserved(tmp_path, name):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    partial = destination / "run"
    partial.mkdir(parents=True)
    lookalike = partial / name
    lookalike.write_bytes(b"preserve me")

    with pytest.raises(snapshot.SnapshotError, match="temporary lookalike"):
        snapshot.create_snapshot(destination, sources, apply=True)

    assert lookalike.read_bytes() == b"preserve me"
    assert not (destination / snapshot.COMPLETE_MARKER_FILENAME).exists()


def test_legitimate_source_filename_that_merely_looks_temporary_is_preserved(tmp_path):
    sources, run, _metadata = _sources(tmp_path)
    unusual = run / ".report.copying"
    unusual.write_bytes(b"legitimate source evidence")
    destination = tmp_path / "snapshot"

    report = snapshot.create_snapshot(destination, sources, apply=True)

    assert report["status"] == "created"
    assert (destination / "run/.report.copying").read_bytes() == unusual.read_bytes()


def test_ambiguous_exact_temp_and_expected_source_filename_fails_closed(tmp_path):
    sources, run, _metadata = _sources(tmp_path)
    ambiguous_source = run / ".cells.json.deadbeef.copying"
    ambiguous_source.write_bytes(b"legitimate but ambiguous source evidence")
    destination = tmp_path / "snapshot"
    partial = destination / "run"
    partial.mkdir(parents=True)
    ambiguous_destination = partial / ambiguous_source.name
    ambiguous_destination.write_bytes(ambiguous_source.read_bytes())

    with pytest.raises(snapshot.SnapshotError, match="ambiguous expected file"):
        snapshot.create_snapshot(destination, sources, apply=True)

    assert ambiguous_destination.read_bytes() == ambiguous_source.read_bytes()


def test_exact_copy_temporary_symlink_is_rejected_and_preserved(tmp_path):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    partial = destination / "run"
    partial.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside evidence")
    orphan = partial / ".cells.json.deadbeef.copying"
    orphan.symlink_to(outside)

    with pytest.raises(snapshot.SnapshotError, match="unsafe interrupted"):
        snapshot.create_snapshot(destination, sources, apply=True)

    assert orphan.is_symlink()
    assert outside.read_bytes() == b"outside evidence"


def test_exact_copy_temporary_hardlink_is_rejected_and_preserved(tmp_path):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    partial = destination / "run"
    partial.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"shared evidence")
    orphan = partial / ".cells.json.deadbeef.copying"
    os.link(outside, orphan)

    with pytest.raises(snapshot.SnapshotError, match="hardlinked interrupted"):
        snapshot.create_snapshot(destination, sources, apply=True)

    assert orphan.exists()
    assert outside.read_bytes() == b"shared evidence"
    assert outside.stat().st_nlink == 2


@pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
def test_exact_control_temporary_links_are_rejected_and_preserved(tmp_path, link_kind):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"control evidence")
    orphan = destination / f".{snapshot.CATALOG_FILENAME}.deadbeef"
    if link_kind == "symlink":
        orphan.symlink_to(outside)
        error = "unsafe interrupted"
    else:
        os.link(outside, orphan)
        error = "hardlinked interrupted"

    with pytest.raises(snapshot.SnapshotError, match=error):
        snapshot.create_snapshot(destination, sources, apply=True)

    assert os.path.lexists(orphan)
    assert outside.read_bytes() == b"control evidence"


def test_sealed_snapshot_never_cleans_an_exact_temporary_name(tmp_path):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    snapshot.create_snapshot(destination, sources, apply=True)
    partial = destination / "run"
    os.chmod(destination, 0o755)
    os.chmod(partial, 0o755)
    orphan = partial / ".cells.json.deadbeef.copying"
    orphan.write_bytes(b"post-seal evidence")

    with pytest.raises(snapshot.SnapshotError, match="unexplained entries"):
        snapshot.create_snapshot(destination, sources, apply=True)

    assert orphan.read_bytes() == b"post-seal evidence"


def test_conflicting_partial_copy_fails_without_completion_marker(tmp_path):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    partial = destination / "run"
    partial.mkdir(parents=True)
    (partial / "cells.json").write_bytes(b"wrong bytes")

    with pytest.raises(snapshot.SnapshotError, match="conflicting resumable"):
        snapshot.create_snapshot(destination, sources, apply=True)

    assert not (destination / snapshot.COMPLETE_MARKER_FILENAME).exists()


def test_unexplained_partial_entry_fails_closed(tmp_path):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    destination.mkdir()
    (destination / "not-from-a-source.txt").write_bytes(b"unknown preimage")

    with pytest.raises(snapshot.SnapshotError, match="unexplained entries"):
        snapshot.create_snapshot(destination, sources, apply=True)

    assert (destination / "not-from-a-source.txt").read_bytes() == b"unknown preimage"
    assert not (destination / snapshot.COMPLETE_MARKER_FILENAME).exists()


def test_sealed_payload_tampering_is_detected(tmp_path):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    snapshot.create_snapshot(destination, sources, apply=True)
    target = destination / "run/cells.json"
    os.chmod(destination, 0o755)
    os.chmod(destination / "run", 0o755)
    os.chmod(target, 0o644)
    target.write_bytes(b"tampered\n")

    with pytest.raises(snapshot.SnapshotError, match="differs|inventory|checksum"):
        snapshot.verify_complete_snapshot(destination, sources)


def test_source_mutation_after_complete_does_not_change_sealed_evidence(tmp_path):
    sources, run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    snapshot.create_snapshot(destination, sources, apply=True)
    (run / "cells.json").write_bytes(b"changed after snapshot\n")

    report = snapshot.create_snapshot(destination, sources, apply=True)
    assert report["status"] == "already_complete"
    assert (destination / "run/cells.json").read_bytes() == b"manifest bytes\n"


def test_symlink_source_is_rejected_without_following(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside bytes")
    (source / "link").symlink_to(outside)
    destination = tmp_path / "snapshot"

    with pytest.raises(snapshot.SnapshotError, match="symlink"):
        snapshot.create_snapshot(
            destination, [snapshot.SourceSpec("source", source)], apply=True
        )
    assert not (destination / snapshot.COMPLETE_MARKER_FILENAME).exists()


def test_external_attestation_binds_every_snapshot_control_file(tmp_path):
    sources, _run, _metadata = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    snapshot.create_snapshot(destination, sources, apply=True)
    attestation = tmp_path / "snapshot.attestation.json"

    report = snapshot.write_snapshot_attestation(destination, attestation)

    assert report["passed"] is True
    assert set(report["control_artifacts"]) == {
        snapshot.COMPLETE_FILENAME,
        snapshot.CATALOG_FILENAME,
        snapshot.SOURCE_INVENTORY_FILENAME,
        snapshot.SNAPSHOT_INVENTORY_FILENAME,
        snapshot.DIRECTORY_INVENTORY_FILENAME,
    }
    assert report["attestation_sha256"] == snapshot._sha256(attestation)
    assert not attestation.resolve().is_relative_to(destination.resolve())


def test_restore_round_trip_is_independent_exact_and_idempotent(tmp_path):
    sources, run, _metadata = _sources(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    snapshot.create_snapshot(snapshot_root, sources, apply=True)
    destination = tmp_path / "restored-run"

    report = restore.restore_source(
        snapshot_root, source_name="run", destination=destination
    )

    assert report["status"] == "restored"
    assert report["file_count"] == 2
    for relative in ("cells.json", "cells/cell-a/results.jsonl"):
        original = run / relative
        recovered = destination / relative
        assert recovered.read_bytes() == original.read_bytes()
        assert (recovered.stat().st_dev, recovered.stat().st_ino) != (
            original.stat().st_dev,
            original.stat().st_ino,
        )
    again = restore.restore_source(
        snapshot_root, source_name="run", destination=destination
    )
    assert again["status"] == "already_restored"


def test_restore_refuses_existing_unmarked_destination(tmp_path):
    sources, _run, _metadata = _sources(tmp_path)
    snapshot_root = tmp_path / "snapshot"
    snapshot.create_snapshot(snapshot_root, sources, apply=True)
    destination = tmp_path / "restored-run"
    destination.mkdir()
    (destination / "user-file").write_text("keep\n", encoding="utf-8")

    with pytest.raises(restore.RestoreError, match="refusing to overwrite"):
        restore.restore_source(
            snapshot_root, source_name="run", destination=destination
        )

    assert (destination / "user-file").read_text(encoding="utf-8") == "keep\n"
