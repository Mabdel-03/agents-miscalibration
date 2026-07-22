"""Generation-scoped integrity seals for large recovery snapshots.

The recovery snapshots are tens of GiB.  Re-hashing every payload from each
dispatcher/status/successor hot path is both unnecessary and operationally harmful.
This module turns one full, controller-owned verification into two small artifacts:

* an immutable generation seal binding the snapshot evidence, control files, selected
  sealed members, and a complete metadata baseline; and
* a short-lived lease renewed under a cross-node lock from a metadata-only scan.

Hot paths validate only the seal and lease.  A chmod, write, replacement, hard-link,
or directory-shape change alters at least one lstat identity and prevents renewal.  A
lease is refreshed at most every five minutes and expires after seven minutes, so
ordinary filesystem drift disables new admission within at most 420 seconds without
re-reading every payload byte.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence


SEAL_SCHEMA_VERSION = 1
BASELINE_SCHEMA_VERSION = 1
LEASE_SCHEMA_VERSION = 1
METADATA_ALGORITHM = "schema5-snapshot-lstat-path-type-mode-inode-nlink-size-mtime-ctime-v1"
LEASE_REFRESH_SECONDS = 300.0
LEASE_TTL_SECONDS = 420.0
LEASE_CLOCK_SKEW_SECONDS = 30.0
_SHA256_LENGTH = 64
_CHUNK_SIZE = 1024 * 1024


class SnapshotIntegrityError(RuntimeError):
    """A snapshot seal, metadata baseline, or lease cannot be trusted."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r}")
    return parsed


def _finite_nonnegative_timestamp(value: Any, *, description: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SnapshotIntegrityError(f"{description} is not a finite timestamp")
    try:
        timestamp = float(value)
    except (ValueError, OverflowError) as exc:
        raise SnapshotIntegrityError(
            f"{description} is not a finite timestamp"
        ) from exc
    if not math.isfinite(timestamp) or timestamp < 0:
        raise SnapshotIntegrityError(f"{description} is not a finite timestamp")
    return timestamp


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_sha(value: Any, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SnapshotIntegrityError(f"{description} is not lowercase SHA-256")
    return value


def _validate_generation(generation: Any) -> int:
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
    ):
        raise SnapshotIntegrityError("snapshot seal generation must be positive")
    return generation


def _absolute_lexical(path: Path) -> Path:
    """Return an absolute normalized path without resolving symbolic links."""

    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(os.fspath(expanded)))


def _reject_symlink_ancestors(path: Path, *, description: str) -> Path:
    """Reject every existing symbolic-link component above ``path``.

    ``O_NOFOLLOW`` protects only the final component.  Integrity artifacts reached
    through a symlinked state directory would otherwise pass that check while escaping
    their generation-scoped namespace.
    """

    lexical = _absolute_lexical(path)
    for ancestor in reversed(lexical.parents):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SnapshotIntegrityError(
                f"cannot inspect {description} ancestor {ancestor}: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise SnapshotIntegrityError(
                f"{description} has a symbolic-link ancestor: {ancestor}"
            )
    return lexical


def _canonical_state_dir(state_dir: Path) -> Path:
    lexical = _absolute_lexical(state_dir)
    _reject_symlink_ancestors(
        lexical / "snapshot_integrity" / ".namespace",
        description="snapshot integrity state directory",
    )
    return lexical


def _read_regular_bytes(path: Path, *, description: str) -> bytes:
    lexical = _reject_symlink_ancestors(path, description=description)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise SnapshotIntegrityError(
            f"cannot open {description} {lexical}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SnapshotIntegrityError(f"{description} is not regular: {lexical}")
        if before.st_nlink != 1:
            raise SnapshotIntegrityError(f"{description} is hardlinked: {lexical}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _CHUNK_SIZE):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise SnapshotIntegrityError(f"{description} changed while read: {lexical}")
    try:
        current = lexical.lstat()
    except OSError as exc:
        raise SnapshotIntegrityError(
            f"cannot restat {description} {lexical}: {exc}"
        ) from exc
    if _stat_identity(current) != _stat_identity(after):
        raise SnapshotIntegrityError(
            f"{description} pathname changed while read: {lexical}"
        )
    return b"".join(chunks)


def sha256_file(path: Path, *, description: str = "snapshot integrity file") -> str:
    return sha256_bytes(_read_regular_bytes(path, description=description))


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return every inode field that can invalidate one snapshot byte proof."""

    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _metadata_row(path: Path, info: os.stat_result) -> dict[str, Any]:
    """Build one validated metadata row from an already-observed inode identity."""

    if stat.S_ISLNK(info.st_mode):
        raise SnapshotIntegrityError(f"snapshot metadata member is symlinked: {path}")
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise SnapshotIntegrityError(f"snapshot metadata member is special: {path}")
    if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        raise SnapshotIntegrityError(f"snapshot regular file is hardlinked: {path}")
    if stat.S_IMODE(info.st_mode) & 0o222:
        raise SnapshotIntegrityError(f"snapshot metadata member is writable: {path}")
    return {
        "path": str(path.absolute()),
        "type": "file" if stat.S_ISREG(info.st_mode) else "directory",
        "mode": stat.S_IMODE(info.st_mode),
        "device": info.st_dev,
        "inode": info.st_ino,
        "nlink": info.st_nlink,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def sha256_file_with_metadata(
    path: Path, *, description: str = "sealed snapshot file"
) -> tuple[str, dict[str, Any]]:
    """Hash one file and return metadata proven from the same stable descriptor.

    A hash followed by an independent ``lstat`` can accidentally bless bytes written in
    the gap between those operations: the digest describes the old inode state while the
    metadata lease records the new state.  This primitive binds both facts to one open
    descriptor and then proves that the pathname still addresses that exact identity.
    Any later replacement necessarily differs from the returned baseline and is caught
    by the generation lease scan.
    """

    lexical = _reject_symlink_ancestors(path, description=description)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise SnapshotIntegrityError(
            f"cannot open {description} {lexical}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        # Validate snapshot immutability before spending time hashing the payload.
        _metadata_row(lexical, before)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise SnapshotIntegrityError(f"{description} changed while read: {lexical}")
        metadata = _metadata_row(lexical, after)
        try:
            current = lexical.lstat()
        except OSError as exc:
            raise SnapshotIntegrityError(
                f"cannot restat {description} {lexical}: {exc}"
            ) from exc
        if _stat_identity(current) != _stat_identity(after):
            raise SnapshotIntegrityError(
                f"{description} pathname changed while read: {lexical}"
            )
        return digest.hexdigest(), metadata
    finally:
        os.close(descriptor)


def read_regular_bytes(path: Path, *, description: str) -> bytes:
    """Read one stable, non-hardlinked regular file for a trusted slow-path audit."""

    return _read_regular_bytes(path, description=description)


def _read_json_object(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(path, description=description)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_finite_float,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SnapshotIntegrityError(
            f"cannot parse {description} {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SnapshotIntegrityError(f"{description} must be an object: {path}")
    return value, raw


def metadata_entry(path: Path) -> dict[str, Any]:
    """Capture one non-symlink regular-file or directory lstat identity."""

    lexical = _reject_symlink_ancestors(path, description="snapshot metadata member")
    try:
        info = lexical.lstat()
    except OSError as exc:
        raise SnapshotIntegrityError(f"cannot stat snapshot member {lexical}: {exc}") from exc
    return _metadata_row(lexical, info)


def canonical_metadata_entries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and canonicalize a complete metadata baseline."""

    required = {
        "path",
        "type",
        "mode",
        "device",
        "inode",
        "nlink",
        "size",
        "mtime_ns",
        "ctime_ns",
    }
    canonical: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or set(entry) != required:
            raise SnapshotIntegrityError(
                f"snapshot metadata entry {index} has the wrong fields"
            )
        row = dict(entry)
        path_value = row.get("path")
        if (
            not isinstance(path_value, str)
            or not Path(path_value).is_absolute()
            or row.get("type") not in {"file", "directory"}
        ):
            raise SnapshotIntegrityError(f"snapshot metadata entry {index} is invalid")
        for field in ("mode", "device", "inode", "nlink", "size", "mtime_ns", "ctime_ns"):
            if not isinstance(row.get(field), int) or isinstance(row.get(field), bool):
                raise SnapshotIntegrityError(
                    f"snapshot metadata entry {index} {field} is invalid"
                )
        if row["type"] == "file" and row["nlink"] != 1:
            raise SnapshotIntegrityError(
                f"snapshot metadata entry {index} records a hardlinked file"
            )
        canonical.append(row)
    canonical.sort(key=lambda row: row["path"])
    paths = [row["path"] for row in canonical]
    if len(paths) != len(set(paths)):
        raise SnapshotIntegrityError("snapshot metadata baseline has duplicate paths")
    return canonical


def _seal_path(state_dir: Path, generation: int) -> Path:
    return state_dir / "snapshot_integrity" / f"snapshot.g{generation:06d}.json"


def _baseline_path(state_dir: Path, generation: int) -> Path:
    return state_dir / "snapshot_integrity" / f"baseline.g{generation:06d}.json"


def generation_lease_path(state_dir: Path, generation: int) -> Path:
    return state_dir / "snapshot_integrity" / f"lease.g{generation:06d}.json"


def _canonical_generation_path(
    path: Path, *, generation: int, artifact: str
) -> tuple[Path, Path]:
    """Require the exact state-relative filename for one generation artifact."""

    generation = _validate_generation(generation)
    lexical = _reject_symlink_ancestors(
        path, description=f"snapshot generation {artifact}"
    )
    if lexical.parent.name != "snapshot_integrity":
        raise SnapshotIntegrityError(
            f"snapshot generation {artifact} is outside its canonical directory"
        )
    state_dir = lexical.parent.parent
    builders = {
        "seal": _seal_path,
        "baseline": _baseline_path,
        "lease": generation_lease_path,
    }
    try:
        expected = builders[artifact](state_dir, generation)
    except KeyError as exc:  # pragma: no cover - internal programming error
        raise ValueError(f"unknown snapshot integrity artifact {artifact!r}") from exc
    if lexical != expected:
        raise SnapshotIntegrityError(
            f"snapshot generation {artifact} is not at its canonical generation path: "
            f"expected {expected}, got {lexical}"
        )
    return lexical, state_dir


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path = _reject_symlink_ancestors(path, description="snapshot integrity lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        raise SnapshotIntegrityError(
            f"cannot open snapshot integrity lock {path}: {exc}"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise SnapshotIntegrityError(
                f"snapshot integrity lock is not regular: {path}"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()} at_ns={time.time_ns()}\n".encode())
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _atomic_publish_read_only(path: Path, payload: bytes, *, replace: bool) -> None:
    path = _reject_symlink_ancestors(
        path, description="snapshot integrity publication"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".publishing", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and os.path.lexists(path):
            raise SnapshotIntegrityError(
                f"refusing to replace snapshot integrity seal: {path}"
            )
        if replace and os.path.lexists(path):
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise SnapshotIntegrityError(
                    f"refusing to replace unsafe snapshot lease: {path}"
                )
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def ensure_generation_seal(
    *,
    state_dir: Path,
    generation: int,
    immutable_pins_sha256: str,
    snapshot_gate_evidence_path: Path,
    snapshot_gate_evidence_sha256: str,
    snapshots: Sequence[Mapping[str, Any]],
    sealed_members: Sequence[Mapping[str, Any]],
    metadata_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Publish an immutable seal from a just-completed full verification."""

    generation = _validate_generation(generation)
    immutable_pins_sha256 = _validate_sha(
        immutable_pins_sha256, description="immutable pin digest"
    )
    snapshot_gate_evidence_sha256 = _validate_sha(
        snapshot_gate_evidence_sha256, description="snapshot gate evidence digest"
    )
    evidence = _reject_symlink_ancestors(
        snapshot_gate_evidence_path,
        description="snapshot gate evidence",
    )
    if (
        sha256_file(evidence, description="snapshot gate evidence")
        != snapshot_gate_evidence_sha256
    ):
        raise SnapshotIntegrityError("snapshot gate evidence changed before sealing")
    canonical_entries = canonical_metadata_entries(metadata_entries)
    if not canonical_entries:
        raise SnapshotIntegrityError("snapshot metadata baseline is empty")
    resolved_state = _canonical_state_dir(state_dir)
    baseline_path = _baseline_path(resolved_state, generation)
    seal_path = _seal_path(resolved_state, generation)
    baseline: dict[str, Any] = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "kind": "schema5-snapshot-metadata-baseline",
        "generation": generation,
        "immutable_pins_sha256": immutable_pins_sha256,
        "metadata_algorithm": METADATA_ALGORITHM,
        "entries": canonical_entries,
    }
    baseline["baseline_id"] = sha256_bytes(canonical_bytes(baseline))
    baseline_payload = _json_bytes(baseline)
    baseline_sha256 = sha256_bytes(baseline_payload)
    snapshot_rows = sorted((dict(row) for row in snapshots), key=lambda row: row["snapshot_root"])
    member_rows = sorted(
        (dict(row) for row in sealed_members),
        key=lambda row: (row["snapshot_root"], row["logical_path"]),
    )
    seal: dict[str, Any] = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "kind": "schema5-snapshot-generation-seal",
        "generation": generation,
        "immutable_pins_sha256": immutable_pins_sha256,
        "snapshot_gate_evidence_path": str(evidence),
        "snapshot_gate_evidence_sha256": snapshot_gate_evidence_sha256,
        "metadata_algorithm": METADATA_ALGORITHM,
        "metadata_baseline_path": str(baseline_path),
        "metadata_baseline_sha256": baseline_sha256,
        "metadata_entry_count": len(canonical_entries),
        "metadata_sha256": sha256_bytes(canonical_bytes(canonical_entries)),
        "snapshots": snapshot_rows,
        "sealed_members": member_rows,
    }
    seal["seal_id"] = sha256_bytes(canonical_bytes(seal))
    seal_payload = _json_bytes(seal)
    seal_sha256 = sha256_bytes(seal_payload)
    lock = resolved_state / "locks" / "snapshot-integrity.lock"
    with _exclusive_lock(lock):
        baseline_exists = baseline_path.exists() or baseline_path.is_symlink()
        seal_exists = seal_path.exists() or seal_path.is_symlink()
        if baseline_exists:
            if sha256_file(
                baseline_path, description="existing snapshot metadata baseline"
            ) != baseline_sha256:
                raise SnapshotIntegrityError(
                    "existing generation snapshot baseline differs from full verification"
                )
        elif seal_exists:
            raise SnapshotIntegrityError(
                "snapshot generation seal exists without its metadata baseline"
            )
        else:
            _atomic_publish_read_only(baseline_path, baseline_payload, replace=False)
        if seal_exists:
            existing = verify_generation_seal(
                path=seal_path,
                expected_sha256=seal_sha256,
                generation=generation,
                immutable_pins_sha256=immutable_pins_sha256,
            )
            if existing != seal:
                raise SnapshotIntegrityError(
                    "existing generation snapshot seal differs from full verification"
                )
            return {
                "path": str(seal_path),
                "sha256": seal_sha256,
                "record": existing,
                "cached": True,
            }
        # A crash after baseline publication but before this marker is recoverable:
        # the exact immutable baseline bytes were proven above and are reused.
        _atomic_publish_read_only(seal_path, seal_payload, replace=False)
    return {
        "path": str(seal_path),
        "sha256": seal_sha256,
        "record": seal,
        "cached": False,
    }


def verify_generation_seal(
    *,
    path: Path,
    expected_sha256: str,
    generation: int,
    immutable_pins_sha256: str,
) -> dict[str, Any]:
    """Verify the compact immutable seal without traversing either snapshot."""

    generation = _validate_generation(generation)
    expected_sha256 = _validate_sha(expected_sha256, description="snapshot seal digest")
    immutable_pins_sha256 = _validate_sha(
        immutable_pins_sha256, description="immutable pin digest"
    )
    target, state_dir = _canonical_generation_path(
        path, generation=generation, artifact="seal"
    )
    record, raw = _read_json_object(target, description="snapshot generation seal")
    info = target.lstat()
    if stat.S_IMODE(info.st_mode) & 0o222:
        raise SnapshotIntegrityError(f"snapshot generation seal is writable: {target}")
    if sha256_bytes(raw) != expected_sha256:
        raise SnapshotIntegrityError("snapshot generation seal byte hash drifted")
    required = {
        "schema_version",
        "kind",
        "generation",
        "immutable_pins_sha256",
        "snapshot_gate_evidence_path",
        "snapshot_gate_evidence_sha256",
        "metadata_algorithm",
        "metadata_baseline_path",
        "metadata_baseline_sha256",
        "metadata_entry_count",
        "metadata_sha256",
        "snapshots",
        "sealed_members",
        "seal_id",
    }
    candidate = dict(record)
    seal_id = candidate.pop("seal_id", None)
    if (
        set(record) != required
        or record.get("schema_version") != SEAL_SCHEMA_VERSION
        or record.get("kind") != "schema5-snapshot-generation-seal"
        or record.get("generation") != generation
        or record.get("immutable_pins_sha256") != immutable_pins_sha256
        or record.get("metadata_algorithm") != METADATA_ALGORITHM
        or not isinstance(record.get("metadata_entry_count"), int)
        or isinstance(record.get("metadata_entry_count"), bool)
        or record["metadata_entry_count"] < 1
        or not isinstance(record.get("snapshots"), list)
        or len(record["snapshots"]) != 2
        or not isinstance(record.get("sealed_members"), list)
        or seal_id != sha256_bytes(canonical_bytes(candidate))
    ):
        raise SnapshotIntegrityError("snapshot generation seal identity is invalid")
    for field in (
        "snapshot_gate_evidence_sha256",
        "metadata_baseline_sha256",
        "metadata_sha256",
    ):
        _validate_sha(record.get(field), description=field)
    for field in ("snapshot_gate_evidence_path", "metadata_baseline_path"):
        value = record.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise SnapshotIntegrityError(f"snapshot generation seal {field} is invalid")
        _reject_symlink_ancestors(
            Path(value), description=f"snapshot generation seal {field}"
        )
    expected_baseline = _baseline_path(state_dir, generation)
    if record["metadata_baseline_path"] != str(expected_baseline):
        raise SnapshotIntegrityError(
            "snapshot generation seal metadata baseline is outside its canonical "
            "generation path"
        )
    _canonical_generation_path(
        Path(record["metadata_baseline_path"]),
        generation=generation,
        artifact="baseline",
    )
    snapshot_roots: set[str] = set()
    snapshot_ids: set[str] = set()
    for row in record["snapshots"]:
        required_snapshot = {
            "attestation_path",
            "attestation_sha256",
            "snapshot_root",
            "snapshot_id",
            "file_count",
            "total_bytes",
            "inventory_sha256",
            "control_artifacts",
        }
        if not isinstance(row, dict) or set(row) != required_snapshot:
            raise SnapshotIntegrityError("snapshot generation seal snapshot row is invalid")
        for field in ("attestation_path", "snapshot_root"):
            if not isinstance(row.get(field), str) or not Path(row[field]).is_absolute():
                raise SnapshotIntegrityError(f"snapshot generation seal {field} is invalid")
            _reject_symlink_ancestors(
                Path(row[field]), description=f"snapshot generation seal {field}"
            )
        for field in ("attestation_sha256", "inventory_sha256"):
            _validate_sha(row.get(field), description=f"snapshot {field}")
        if not isinstance(row.get("snapshot_id"), str) or not row["snapshot_id"]:
            raise SnapshotIntegrityError("snapshot generation seal snapshot ID is invalid")
        if not isinstance(row.get("control_artifacts"), dict):
            raise SnapshotIntegrityError("snapshot seal control records are invalid")
        snapshot_roots.add(row["snapshot_root"])
        snapshot_ids.add(row["snapshot_id"])
    if len(snapshot_roots) != 2 or len(snapshot_ids) != 2:
        raise SnapshotIntegrityError("snapshot generation seal does not bind two snapshots")
    member_keys: set[tuple[str, str]] = set()
    for row in record["sealed_members"]:
        if not isinstance(row, dict) or set(row) != {
            "snapshot_id",
            "snapshot_root",
            "logical_path",
            "sha256",
        }:
            raise SnapshotIntegrityError("snapshot generation sealed-member row is invalid")
        _validate_sha(row.get("sha256"), description="sealed member digest")
        key = (str(row.get("snapshot_root")), str(row.get("logical_path")))
        if key in member_keys or key[0] not in snapshot_roots:
            raise SnapshotIntegrityError("snapshot generation sealed-member identity is invalid")
        member_keys.add(key)
    return record


def _load_baseline(seal: Mapping[str, Any]) -> dict[str, Any]:
    path, _state_dir = _canonical_generation_path(
        Path(str(seal["metadata_baseline_path"])),
        generation=int(seal["generation"]),
        artifact="baseline",
    )
    baseline, raw = _read_json_object(path, description="snapshot metadata baseline")
    info = path.lstat()
    if stat.S_IMODE(info.st_mode) & 0o222:
        raise SnapshotIntegrityError(
            f"snapshot metadata baseline remains writable: {path}"
        )
    if sha256_bytes(raw) != seal["metadata_baseline_sha256"]:
        raise SnapshotIntegrityError("snapshot metadata baseline byte hash drifted")
    candidate = dict(baseline)
    baseline_id = candidate.pop("baseline_id", None)
    if (
        set(baseline)
        != {
            "schema_version",
            "kind",
            "generation",
            "immutable_pins_sha256",
            "metadata_algorithm",
            "entries",
            "baseline_id",
        }
        or baseline.get("schema_version") != BASELINE_SCHEMA_VERSION
        or baseline.get("kind") != "schema5-snapshot-metadata-baseline"
        or baseline.get("generation") != seal["generation"]
        or baseline.get("immutable_pins_sha256") != seal["immutable_pins_sha256"]
        or baseline.get("metadata_algorithm") != METADATA_ALGORITHM
        or baseline_id != sha256_bytes(canonical_bytes(candidate))
        or not isinstance(baseline.get("entries"), list)
    ):
        raise SnapshotIntegrityError("snapshot metadata baseline identity is invalid")
    entries = canonical_metadata_entries(baseline["entries"])
    if (
        len(entries) != seal["metadata_entry_count"]
        or sha256_bytes(canonical_bytes(entries)) != seal["metadata_sha256"]
    ):
        raise SnapshotIntegrityError("snapshot metadata baseline differs from seal")
    return baseline


def verify_generation_lease(
    *,
    lease_path: Path,
    seal_path: Path,
    seal_sha256: str,
    generation: int,
    immutable_pins_sha256: str,
    now: float | None = None,
    require_fresh: bool = True,
) -> dict[str, Any]:
    """Validate the small generation lease without scanning snapshot payloads."""

    try:
        timestamp = time.time() if now is None else float(now)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SnapshotIntegrityError("snapshot lease verification time is invalid") from exc
    if isinstance(now, bool) or not math.isfinite(timestamp) or timestamp < 0:
        raise SnapshotIntegrityError("snapshot lease verification time is invalid")
    canonical_seal, state_dir = _canonical_generation_path(
        seal_path, generation=generation, artifact="seal"
    )
    canonical_lease, lease_state_dir = _canonical_generation_path(
        lease_path, generation=generation, artifact="lease"
    )
    if lease_state_dir != state_dir:
        raise SnapshotIntegrityError(
            "snapshot integrity lease and seal do not share a canonical state directory"
        )
    seal = verify_generation_seal(
        path=canonical_seal,
        expected_sha256=seal_sha256,
        generation=generation,
        immutable_pins_sha256=immutable_pins_sha256,
    )
    lease, _raw = _read_json_object(
        canonical_lease, description="snapshot integrity lease"
    )
    try:
        lease_info = canonical_lease.lstat()
    except OSError as exc:
        raise SnapshotIntegrityError(
            f"cannot stat snapshot integrity lease {canonical_lease}: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(lease_info.st_mode)
        or lease_info.st_nlink != 1
        or stat.S_IMODE(lease_info.st_mode) != 0o444
    ):
        raise SnapshotIntegrityError(
            "snapshot integrity lease must be one regular, non-hardlinked mode-0444 file"
        )
    required = {
        "schema_version",
        "kind",
        "generation",
        "immutable_pins_sha256",
        "seal_path",
        "seal_sha256",
        "seal_id",
        "metadata_baseline_sha256",
        "metadata_sha256",
        "sequence",
        "verified_timestamp",
        "expires_timestamp",
        "lease_id",
    }
    candidate = dict(lease)
    lease_id = candidate.pop("lease_id", None)
    try:
        verified_timestamp = _finite_nonnegative_timestamp(
            lease.get("verified_timestamp"),
            description="snapshot lease verified time",
        )
        expires_timestamp = _finite_nonnegative_timestamp(
            lease.get("expires_timestamp"),
            description="snapshot lease expiry time",
        )
    except SnapshotIntegrityError as exc:
        raise SnapshotIntegrityError(
            "snapshot integrity lease identity is invalid"
        ) from exc
    if (
        set(lease) != required
        or lease.get("schema_version") != LEASE_SCHEMA_VERSION
        or lease.get("kind") != "schema5-snapshot-integrity-lease"
        or lease.get("generation") != generation
        or lease.get("immutable_pins_sha256") != immutable_pins_sha256
        or lease.get("seal_path") != str(canonical_seal)
        or lease.get("seal_sha256") != seal_sha256
        or lease.get("seal_id") != seal["seal_id"]
        or lease.get("metadata_baseline_sha256") != seal["metadata_baseline_sha256"]
        or lease.get("metadata_sha256") != seal["metadata_sha256"]
        or not isinstance(lease.get("sequence"), int)
        or isinstance(lease.get("sequence"), bool)
        or lease["sequence"] < 1
        or expires_timestamp != verified_timestamp + LEASE_TTL_SECONDS
        or lease_id != sha256_bytes(canonical_bytes(candidate))
    ):
        raise SnapshotIntegrityError("snapshot integrity lease identity is invalid")
    if verified_timestamp > timestamp + LEASE_CLOCK_SKEW_SECONDS:
        raise SnapshotIntegrityError("snapshot integrity lease was verified in the future")
    if require_fresh and timestamp >= expires_timestamp:
        raise SnapshotIntegrityError("snapshot integrity lease expired")
    return lease


def refresh_generation_lease(
    *,
    state_dir: Path,
    seal_path: Path,
    seal_sha256: str,
    generation: int,
    immutable_pins_sha256: str,
    now: float | None = None,
    force: bool = False,
    refresh_seconds: float = LEASE_REFRESH_SECONDS,
    ttl_seconds: float = LEASE_TTL_SECONDS,
) -> dict[str, Any]:
    """Renew the lease from a metadata-only full-tree scan when it is due."""

    try:
        timestamp = time.time() if now is None else float(now)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SnapshotIntegrityError("snapshot lease cadence/TTL is unsafe") from exc
    refresh_started_monotonic = time.monotonic()
    if (
        isinstance(now, bool)
        or not math.isfinite(timestamp)
        or timestamp < 0
        or isinstance(refresh_seconds, bool)
        or isinstance(ttl_seconds, bool)
        or not isinstance(refresh_seconds, (int, float))
        or not isinstance(ttl_seconds, (int, float))
        or not math.isfinite(float(refresh_seconds))
        or not math.isfinite(float(ttl_seconds))
        or refresh_seconds <= 0
        or ttl_seconds <= refresh_seconds
        or float(ttl_seconds) != LEASE_TTL_SECONDS
    ):
        raise SnapshotIntegrityError("snapshot lease cadence/TTL is unsafe")
    generation = _validate_generation(generation)
    resolved_state = _canonical_state_dir(state_dir)
    canonical_seal, seal_state_dir = _canonical_generation_path(
        seal_path, generation=generation, artifact="seal"
    )
    if seal_state_dir != resolved_state:
        raise SnapshotIntegrityError(
            "snapshot generation seal does not belong to the requested state directory"
        )
    lease_path = generation_lease_path(resolved_state, generation)
    lock = resolved_state / "locks" / "snapshot-integrity.lock"
    with _exclusive_lock(lock):
        evaluation_timestamp = time.time() if now is None else timestamp
        previous: dict[str, Any] | None = None
        if lease_path.exists() or lease_path.is_symlink():
            previous = verify_generation_lease(
                lease_path=lease_path,
                seal_path=canonical_seal,
                seal_sha256=seal_sha256,
                generation=generation,
                immutable_pins_sha256=immutable_pins_sha256,
                now=evaluation_timestamp,
                require_fresh=False,
            )
            age = evaluation_timestamp - float(previous["verified_timestamp"])
            if not force and 0 <= age < refresh_seconds:
                if evaluation_timestamp > float(previous["expires_timestamp"]):
                    raise SnapshotIntegrityError(
                        "snapshot lease expired inside its refresh interval"
                    )
                return {"path": str(lease_path), "record": previous, "cached": True}
        seal = verify_generation_seal(
            path=canonical_seal,
            expected_sha256=seal_sha256,
            generation=generation,
            immutable_pins_sha256=immutable_pins_sha256,
        )
        baseline = _load_baseline(seal)
        expected = canonical_metadata_entries(baseline["entries"])
        observed = [metadata_entry(Path(row["path"])) for row in expected]
        observed = canonical_metadata_entries(observed)
        # Scan twice so a concurrent tree mutation cannot be hidden by a mixed view.
        observed_again = [metadata_entry(Path(row["path"])) for row in expected]
        observed_again = canonical_metadata_entries(observed_again)
        if observed != expected or observed_again != expected:
            raise SnapshotIntegrityError(
                "sealed snapshot metadata drifted after generation verification"
            )
        # Start the TTL after both scans. Explicit ``now`` is a deterministic logical
        # clock advanced by actual monotonic scan time; production takes a new wall-
        # clock reading after the scan.
        verified_timestamp = (
            time.time()
            if now is None
            else timestamp
            + max(0.0, time.monotonic() - refresh_started_monotonic)
        )
        if previous is not None:
            verified_timestamp = max(
                verified_timestamp, float(previous["verified_timestamp"])
            )
        lease: dict[str, Any] = {
            "schema_version": LEASE_SCHEMA_VERSION,
            "kind": "schema5-snapshot-integrity-lease",
            "generation": generation,
            "immutable_pins_sha256": immutable_pins_sha256,
            "seal_path": str(canonical_seal),
            "seal_sha256": seal_sha256,
            "seal_id": seal["seal_id"],
            "metadata_baseline_sha256": seal["metadata_baseline_sha256"],
            "metadata_sha256": seal["metadata_sha256"],
            "sequence": 1 if previous is None else int(previous["sequence"]) + 1,
            "verified_timestamp": verified_timestamp,
            "expires_timestamp": verified_timestamp + LEASE_TTL_SECONDS,
        }
        lease["lease_id"] = sha256_bytes(canonical_bytes(lease))
        _atomic_publish_read_only(lease_path, _json_bytes(lease), replace=True)
        return {"path": str(lease_path), "record": lease, "cached": False}
