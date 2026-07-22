from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from agents_scaling import snapshot_integrity as integrity


def _rewrite_read_only(path: Path, payload: bytes) -> None:
    path.chmod(0o644)
    path.write_bytes(payload)
    path.chmod(0o444)


def _resign(record: dict, identity_field: str) -> bytes:
    candidate = {key: value for key, value in record.items() if key != identity_field}
    record[identity_field] = integrity.sha256_bytes(
        integrity.canonical_bytes(candidate)
    )
    return integrity._json_bytes(record)


def _generation_seal(tmp_path: Path) -> tuple[dict, Path, Path]:
    state_dir = tmp_path / "state"
    evidence = tmp_path / "snapshot-gate.json"
    evidence.write_bytes(b"{}\n")
    member = tmp_path / "payload.bin"
    member.write_bytes(b"canonical payload")
    member.chmod(0o444)
    snapshots = []
    for index in (1, 2):
        root = tmp_path / f"snapshot-{index}"
        root.mkdir()
        snapshots.append(
            {
                "attestation_path": str((tmp_path / f"snapshot-{index}.json").resolve()),
                "attestation_sha256": str(index) * 64,
                "snapshot_root": str(root.resolve()),
                "snapshot_id": f"snapshot-{index}",
                "file_count": 1,
                "total_bytes": member.stat().st_size,
                "inventory_sha256": str(index + 2) * 64,
                "control_artifacts": {},
            }
        )
    sealed = integrity.ensure_generation_seal(
        state_dir=state_dir,
        generation=1,
        immutable_pins_sha256="a" * 64,
        snapshot_gate_evidence_path=evidence,
        snapshot_gate_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        snapshots=snapshots,
        sealed_members=[],
        metadata_entries=[integrity.metadata_entry(member)],
    )
    return sealed, state_dir, member


def test_sha256_with_metadata_rejects_path_replacement_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"original payload")
    target.chmod(0o444)
    replacement = tmp_path / "replacement.bin"
    original_read = integrity.os.read
    replaced = False

    def swap_after_first_chunk(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, size)
        if chunk and not replaced:
            replaced = True
            replacement.write_bytes(b"different payload")
            replacement.chmod(0o444)
            os.replace(replacement, target)
        return chunk

    monkeypatch.setattr(integrity.os, "read", swap_after_first_chunk)
    with pytest.raises(integrity.SnapshotIntegrityError, match="changed while read"):
        integrity.sha256_file_with_metadata(target)


def test_sha256_with_metadata_returns_one_fd_bound_identity(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    payload = b"stable payload"
    target.write_bytes(payload)
    target.chmod(0o444)

    digest, metadata = integrity.sha256_file_with_metadata(target)

    assert digest == hashlib.sha256(payload).hexdigest()
    assert metadata == integrity.metadata_entry(target)


def test_generation_lease_ttl_starts_after_both_metadata_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed, state_dir, _member = _generation_seal(tmp_path)
    monotonic = iter((10.0, 25.0))
    monkeypatch.setattr(integrity.time, "monotonic", lambda: next(monotonic))

    lease = integrity.refresh_generation_lease(
        state_dir=state_dir,
        seal_path=Path(sealed["path"]),
        seal_sha256=sealed["sha256"],
        generation=1,
        immutable_pins_sha256="a" * 64,
        now=1_000.0,
        force=True,
        refresh_seconds=5.0,
        ttl_seconds=integrity.LEASE_TTL_SECONDS,
    )

    assert lease["record"]["verified_timestamp"] == 1_015.0
    assert lease["record"]["expires_timestamp"] == 1_435.0
    integrity.verify_generation_lease(
        lease_path=Path(lease["path"]),
        seal_path=Path(sealed["path"]),
        seal_sha256=sealed["sha256"],
        generation=1,
        immutable_pins_sha256="a" * 64,
        now=1_434.999,
    )
    with pytest.raises(integrity.SnapshotIntegrityError, match="expired"):
        integrity.verify_generation_lease(
            lease_path=Path(lease["path"]),
            seal_path=Path(sealed["path"]),
            seal_sha256=sealed["sha256"],
            generation=1,
            immutable_pins_sha256="a" * 64,
            now=1_435.0,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"schema_version":1,"schema_version":1}\n', "duplicate JSON key"),
        (b'{"schema_version":NaN}\n', "non-finite JSON number"),
        (b'{"schema_version":1e9999}\n', "non-finite JSON number"),
    ],
)
def test_generation_artifacts_reject_ambiguous_json(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    sealed, _state_dir, _member = _generation_seal(tmp_path)
    seal_path = Path(sealed["path"])
    _rewrite_read_only(seal_path, payload)

    with pytest.raises(integrity.SnapshotIntegrityError, match=message):
        integrity.verify_generation_seal(
            path=seal_path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            generation=1,
            immutable_pins_sha256="a" * 64,
        )


def test_generation_paths_are_exact_and_symlink_free(tmp_path: Path) -> None:
    sealed, state_dir, _member = _generation_seal(tmp_path)
    seal_path = Path(sealed["path"])

    wrong_name = seal_path.with_name("snapshot.g000001.copy.json")
    wrong_name.write_bytes(seal_path.read_bytes())
    wrong_name.chmod(0o444)
    with pytest.raises(integrity.SnapshotIntegrityError, match="canonical generation path"):
        integrity.verify_generation_seal(
            path=wrong_name,
            expected_sha256=sealed["sha256"],
            generation=1,
            immutable_pins_sha256="a" * 64,
        )

    alias = tmp_path / "state-alias"
    alias.symlink_to(state_dir, target_is_directory=True)
    aliased_seal = alias / "snapshot_integrity" / seal_path.name
    with pytest.raises(integrity.SnapshotIntegrityError, match="symbolic-link ancestor"):
        integrity.verify_generation_seal(
            path=aliased_seal,
            expected_sha256=sealed["sha256"],
            generation=1,
            immutable_pins_sha256="a" * 64,
        )

    seal_record = json.loads(seal_path.read_text(encoding="utf-8"))
    seal_record["metadata_baseline_path"] = str(
        (tmp_path / "outside" / "baseline.g000001.json").resolve()
    )
    forged = _resign(seal_record, "seal_id")
    _rewrite_read_only(seal_path, forged)
    with pytest.raises(integrity.SnapshotIntegrityError, match="canonical generation path"):
        integrity.verify_generation_seal(
            path=seal_path,
            expected_sha256=hashlib.sha256(forged).hexdigest(),
            generation=1,
            immutable_pins_sha256="a" * 64,
        )


def test_lease_requires_exact_path_mode_and_single_link(tmp_path: Path) -> None:
    sealed, state_dir, _member = _generation_seal(tmp_path)
    lease = integrity.refresh_generation_lease(
        state_dir=state_dir,
        seal_path=Path(sealed["path"]),
        seal_sha256=sealed["sha256"],
        generation=1,
        immutable_pins_sha256="a" * 64,
        now=1_000.0,
        force=True,
    )
    lease_path = Path(lease["path"])

    wrong_name = lease_path.with_name("lease.g000001.copy.json")
    wrong_name.write_bytes(lease_path.read_bytes())
    wrong_name.chmod(0o444)
    with pytest.raises(integrity.SnapshotIntegrityError, match="canonical generation path"):
        integrity.verify_generation_lease(
            lease_path=wrong_name,
            seal_path=Path(sealed["path"]),
            seal_sha256=sealed["sha256"],
            generation=1,
            immutable_pins_sha256="a" * 64,
            now=1_000.0,
        )

    lease_path.chmod(0o400)
    with pytest.raises(integrity.SnapshotIntegrityError, match="mode-0444"):
        integrity.verify_generation_lease(
            lease_path=lease_path,
            seal_path=Path(sealed["path"]),
            seal_sha256=sealed["sha256"],
            generation=1,
            immutable_pins_sha256="a" * 64,
            now=1_000.0,
        )
    lease_path.chmod(0o444)
    os.link(lease_path, tmp_path / "lease-hardlink.json")
    with pytest.raises(integrity.SnapshotIntegrityError, match="hardlinked"):
        integrity.verify_generation_lease(
            lease_path=lease_path,
            seal_path=Path(sealed["path"]),
            seal_sha256=sealed["sha256"],
            generation=1,
            immutable_pins_sha256="a" * 64,
            now=1_000.0,
        )


def test_forged_long_lifetime_and_future_lease_fail_closed(tmp_path: Path) -> None:
    sealed, state_dir, _member = _generation_seal(tmp_path)
    lease = integrity.refresh_generation_lease(
        state_dir=state_dir,
        seal_path=Path(sealed["path"]),
        seal_sha256=sealed["sha256"],
        generation=1,
        immutable_pins_sha256="a" * 64,
        now=1_000.0,
        force=True,
    )
    lease_path = Path(lease["path"])
    record = json.loads(lease_path.read_text(encoding="utf-8"))
    record["expires_timestamp"] = float(record["expires_timestamp"]) + 86_400.0
    forged = _resign(record, "lease_id")
    _rewrite_read_only(lease_path, forged)
    with pytest.raises(integrity.SnapshotIntegrityError, match="lease identity is invalid"):
        integrity.verify_generation_lease(
            lease_path=lease_path,
            seal_path=Path(sealed["path"]),
            seal_sha256=sealed["sha256"],
            generation=1,
            immutable_pins_sha256="a" * 64,
            now=1_000.0,
        )

    record["verified_timestamp"] = 10**400
    record["expires_timestamp"] = 10**400 + 420
    overflow = _resign(record, "lease_id")
    _rewrite_read_only(lease_path, overflow)
    with pytest.raises(integrity.SnapshotIntegrityError, match="lease identity is invalid"):
        integrity.verify_generation_lease(
            lease_path=lease_path,
            seal_path=Path(sealed["path"]),
            seal_sha256=sealed["sha256"],
            generation=1,
            immutable_pins_sha256="a" * 64,
            now=1_000.0,
        )

    record["verified_timestamp"] = 1_031.0
    record["expires_timestamp"] = 1_451.0
    future = _resign(record, "lease_id")
    _rewrite_read_only(lease_path, future)
    with pytest.raises(integrity.SnapshotIntegrityError, match="verified in the future"):
        integrity.verify_generation_lease(
            lease_path=lease_path,
            seal_path=Path(sealed["path"]),
            seal_sha256=sealed["sha256"],
            generation=1,
            immutable_pins_sha256="a" * 64,
            now=1_000.0,
        )


def test_atomic_refresh_publishes_one_valid_exact_lifetime_lease(tmp_path: Path) -> None:
    sealed, state_dir, _member = _generation_seal(tmp_path)
    first = integrity.refresh_generation_lease(
        state_dir=state_dir,
        seal_path=Path(sealed["path"]),
        seal_sha256=sealed["sha256"],
        generation=1,
        immutable_pins_sha256="a" * 64,
        now=1_000.0,
        force=True,
    )
    second = integrity.refresh_generation_lease(
        state_dir=state_dir,
        seal_path=Path(sealed["path"]),
        seal_sha256=sealed["sha256"],
        generation=1,
        immutable_pins_sha256="a" * 64,
        now=float(first["record"]["verified_timestamp"]) + 301.0,
        force=True,
    )

    lease_path = Path(second["path"])
    assert stat.S_IMODE(lease_path.stat().st_mode) == 0o444
    assert lease_path.stat().st_nlink == 1
    assert second["record"]["sequence"] == 2
    assert (
        float(second["record"]["expires_timestamp"])
        - float(second["record"]["verified_timestamp"])
        == integrity.LEASE_TTL_SECONDS
    )
    assert integrity.verify_generation_lease(
        lease_path=lease_path,
        seal_path=Path(sealed["path"]),
        seal_sha256=sealed["sha256"],
        generation=1,
        immutable_pins_sha256="a" * 64,
        now=float(second["record"]["verified_timestamp"]),
    ) == second["record"]


def test_generation_seal_recovers_after_baseline_only_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    evidence = tmp_path / "snapshot-gate.json"
    evidence.write_bytes(b"{}\n")
    member = tmp_path / "payload.bin"
    member.write_bytes(b"payload")
    member.chmod(0o444)
    snapshots = []
    for index in (1, 2):
        root = tmp_path / f"snapshot-{index}"
        root.mkdir()
        snapshots.append(
            {
                "attestation_path": str((tmp_path / f"attestation-{index}").resolve()),
                "attestation_sha256": str(index) * 64,
                "snapshot_root": str(root.resolve()),
                "snapshot_id": f"snapshot-{index}",
                "file_count": 1,
                "total_bytes": member.stat().st_size,
                "inventory_sha256": str(index + 2) * 64,
                "control_artifacts": {},
            }
        )
    kwargs = {
        "state_dir": state_dir,
        "generation": 1,
        "immutable_pins_sha256": "a" * 64,
        "snapshot_gate_evidence_path": evidence,
        "snapshot_gate_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "snapshots": snapshots,
        "sealed_members": [],
        "metadata_entries": [integrity.metadata_entry(member)],
    }
    original_publish = integrity._atomic_publish_read_only
    calls = 0

    def crash_before_seal(path: Path, payload: bytes, *, replace: bool) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            original_publish(path, payload, replace=replace)
            return
        raise RuntimeError("simulated crash after baseline publication")

    monkeypatch.setattr(integrity, "_atomic_publish_read_only", crash_before_seal)
    with pytest.raises(RuntimeError, match="simulated crash"):
        integrity.ensure_generation_seal(**kwargs)
    monkeypatch.setattr(integrity, "_atomic_publish_read_only", original_publish)

    recovered = integrity.ensure_generation_seal(**kwargs)

    assert recovered["cached"] is False
    assert Path(recovered["path"]).is_file()
