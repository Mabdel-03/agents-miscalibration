#!/usr/bin/env python3
"""Create or verify an independently copied, checksummed recovery snapshot.

The destination is resumable until ``SNAPSHOT_COMPLETE.json`` is published.  A
completed snapshot is immutable: every regular file and directory is made read-only,
and a later invocation is verification-only.  Symlinks, hardlinks into the source,
special files, and source mutation during copy are rejected rather than silently
changing the evidence boundary.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
from typing import Iterable, Iterator

COMPLETE_FILENAME = "SNAPSHOT_COMPLETE.json"
CATALOG_FILENAME = "SNAPSHOT_CATALOG.json"
SOURCE_INVENTORY_FILENAME = "SOURCE_INVENTORY.sha256"
SNAPSHOT_INVENTORY_FILENAME = "SNAPSHOT_INVENTORY.sha256"
DIRECTORY_INVENTORY_FILENAME = "DIRECTORY_INVENTORY.txt"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
# ``tempfile.mkstemp`` uses eight characters from this alphabet for the random
# component on the pinned production Python.  The snapshot writer has always used
# that default component verbatim.  Keeping the grammar deliberately narrow lets a
# resume distinguish its own abandoned publications from similarly named evidence.
_TEMPFILE_TOKEN = r"[a-z0-9_]{8}"
_COPY_TEMPORARY = re.compile(
    rf"\.(?P<target>[^/\r\n]+)\.(?P<token>{_TEMPFILE_TOKEN})\.copying\Z"
)
_CHUNK = 8 * 1024 * 1024
_CONTROL_FILENAMES = frozenset(
    {
        COMPLETE_FILENAME,
        CATALOG_FILENAME,
        SOURCE_INVENTORY_FILENAME,
        SNAPSHOT_INVENTORY_FILENAME,
        DIRECTORY_INVENTORY_FILENAME,
    }
)

# Public aliases used by recovery orchestration and focused tests.  Keep the on-disk
# names above stable for already-started resumable snapshots.
COMPLETE_MARKER_FILENAME = COMPLETE_FILENAME
SNAPSHOT_MANIFEST_FILENAME = CATALOG_FILENAME
COPY_INVENTORY_FILENAME = SNAPSHOT_INVENTORY_FILENAME
ATTESTATION_SCHEMA_VERSION = 1
_COMPLETION_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "snapshot_id",
        "completed_at",
        "file_count",
        "total_bytes",
        "snapshot_inventory_sha256",
        "verified",
        "read_only",
    }
)
_ATTESTATION_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "passed",
        "snapshot_root",
        "snapshot_id",
        "file_count",
        "total_bytes",
        "control_artifacts",
        "attested_at",
    }
)


class SnapshotError(RuntimeError):
    """The snapshot could not be proven complete and independent."""


@dataclass(frozen=True, order=True)
class FileRecord:
    logical_path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class Source:
    name: str
    path: Path


SourceSpec = Source


def _utc_now() -> str:
    # Readiness validation deliberately accepts one canonical representation.  Avoid
    # fractional seconds and ``+00:00`` aliases so sealed evidence is byte-stable and
    # interoperable with the rest of the schema-5 control plane.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc_timestamp(value: object, *, context: str) -> datetime:
    if not isinstance(value, str):
        raise SnapshotError(f"{context} must be a UTC timestamp string")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise SnapshotError(
            f"{context} must use canonical YYYY-MM-DDTHH:MM:SSZ form"
        ) from exc
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: object, *, mode: int = 0o644) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(path, encoded, mode=mode)


def _parse_source(value: str) -> Source:
    name, separator, raw_path = value.partition("=")
    if not separator or not _SAFE_NAME.fullmatch(name):
        raise SnapshotError(f"--source must be SAFE_NAME=/absolute/path, got {value!r}")
    path = Path(raw_path)
    if not path.is_absolute():
        raise SnapshotError(f"source path must be absolute: {raw_path!r}")
    if path.is_symlink() or not path.exists():
        raise SnapshotError(f"source is missing or a symlink: {path}")
    if not (path.is_dir() or path.is_file()):
        raise SnapshotError(f"source is not a regular file or directory: {path}")
    return Source(name=name, path=path.resolve())


def _walk_source(source: Source) -> Iterator[tuple[str, Path, bool]]:
    """Yield logical path, source path, and directory flag in stable order."""
    if source.path.is_file():
        yield source.name, source.path, False
        return
    yield source.name, source.path, True
    for root, directory_names, file_names in os.walk(source.path, followlinks=False):
        directory_names.sort()
        file_names.sort()
        root_path = Path(root)
        for directory_name in directory_names:
            path = root_path / directory_name
            if path.is_symlink() or not path.is_dir():
                raise SnapshotError(f"refusing unsafe source directory: {path}")
            relative = path.relative_to(source.path).as_posix()
            yield f"{source.name}/{relative}", path, True
        for file_name in file_names:
            path = root_path / file_name
            if path.is_symlink() or not path.is_file():
                raise SnapshotError(f"refusing symlink or special source file: {path}")
            relative = path.relative_to(source.path).as_posix()
            logical = f"{source.name}/{relative}"
            if "\n" in logical or "\r" in logical:
                raise SnapshotError(
                    f"newline in source path is not inventory-safe: {path}"
                )
            yield logical, path, False


def _copy_one(source: Path, destination: Path, logical: str) -> FileRecord:
    before = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise SnapshotError(f"source ceased to be a regular file: {source}")
    source_digest = _sha256(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    reuse = False
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise SnapshotError(f"unsafe resumable destination: {destination}")
        reuse = (
            destination.stat().st_size == before.st_size
            and _sha256(destination) == source_digest
        )
        if not reuse:
            # A resumable snapshot may reuse an exact independent copy, but it must not
            # overwrite an unexplained preimage.  The operator can preserve/remove the
            # incomplete destination explicitly after inspecting the conflict.
            raise SnapshotError(f"conflicting resumable destination: {destination}")
    if not reuse:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".copying", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
                shutil.copyfileobj(reader, writer, length=_CHUNK)
                writer.flush()
                os.fsync(writer.fileno())
            shutil.copystat(source, temporary, follow_symlinks=False)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    after = source.stat(follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise SnapshotError(f"source changed while being copied: {source}")
    copied = destination.stat(follow_symlinks=False)
    if before.st_dev == copied.st_dev and before.st_ino == copied.st_ino:
        raise SnapshotError(f"destination is a forbidden hardlink to source: {logical}")
    if copied.st_nlink != 1:
        raise SnapshotError(f"destination has multiple hardlinks: {logical}")
    destination_digest = _sha256(destination)
    if destination_digest != source_digest or copied.st_size != before.st_size:
        raise SnapshotError(f"copy verification failed: {logical}")
    return FileRecord(logical, source_digest, before.st_size)


def _inventory_bytes(records: Iterable[FileRecord]) -> bytes:
    return "".join(
        f"{record.sha256}  {record.logical_path}\n" for record in sorted(records)
    ).encode("utf-8")


def _directory_bytes(directories: Iterable[str]) -> bytes:
    return "".join(f"{path}\n" for path in sorted(directories)).encode("utf-8")


def _inventory_sources(
    sources: tuple[Source, ...],
) -> tuple[tuple[FileRecord, ...], tuple[str, ...]]:
    """Hash a complete stable view of every source before/after the copy pass."""

    records: list[FileRecord] = []
    directories: list[str] = []
    for source in sources:
        for logical, source_path, is_directory in _walk_source(source):
            before = source_path.stat(follow_symlinks=False)
            if is_directory:
                if not stat.S_ISDIR(before.st_mode):
                    raise SnapshotError(f"source directory changed type: {source_path}")
                directories.append(logical)
                continue
            digest = _sha256(source_path)
            after = source_path.stat(follow_symlinks=False)
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise SnapshotError(f"source changed during inventory: {source_path}")
            records.append(FileRecord(logical, digest, before.st_size))
    ordered_records = tuple(sorted(records))
    ordered_directories = tuple(sorted(directories))
    if len({record.logical_path for record in ordered_records}) != len(ordered_records):
        raise SnapshotError("duplicate logical file path in source inventory")
    if len(set(ordered_directories)) != len(ordered_directories):
        raise SnapshotError("duplicate logical directory path in source inventory")
    return ordered_records, ordered_directories


def _make_incomplete_snapshot_writable(snapshot_root: Path) -> None:
    """Allow a run interrupted during the final chmod pass to resume safely."""

    if not snapshot_root.exists():
        return
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise SnapshotError(f"unsafe incomplete snapshot root: {snapshot_root}")
    for root, directory_names, _file_names in os.walk(
        snapshot_root, topdown=True, followlinks=False
    ):
        root_path = Path(root)
        if root_path.is_symlink():
            raise SnapshotError(f"symlink in incomplete snapshot: {root_path}")
        root_path.chmod(stat.S_IMODE(root_path.stat().st_mode) | 0o700)
        directory_names.sort()


def _source_node_for_logical_path(
    logical_path: Path, sources_by_name: dict[str, Source]
) -> tuple[Path, int] | None:
    """Resolve one logical path without following any source-side symlink.

    This check intentionally performs no content hashing.  It is used only to prove
    that an abandoned ``.copying`` file belongs to a target the snapshot tool would
    copy on this invocation, before the potentially very expensive inventory pass.
    The full stable inventory and byte verification still follow cleanup.
    """

    parts = logical_path.parts
    if not parts:
        return None
    source = sources_by_name.get(parts[0])
    if source is None:
        return None
    candidate = source.path
    try:
        info = candidate.stat(follow_symlinks=False)
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode) or not (
        stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
    ):
        return None
    if len(parts) == 1:
        return candidate, info.st_mode
    if not stat.S_ISDIR(info.st_mode):
        return None
    for index, component in enumerate(parts[1:], 1):
        candidate = candidate / component
        try:
            info = candidate.stat(follow_symlinks=False)
        except OSError:
            return None
        if stat.S_ISLNK(info.st_mode) or not (
            stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
        ):
            return None
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            return None
    return candidate, info.st_mode


def _source_target_for_logical_path(
    logical_path: Path, sources_by_name: dict[str, Source]
) -> Path | None:
    """Resolve one logical path only when its safe source node is a regular file."""

    resolved = _source_node_for_logical_path(logical_path, sources_by_name)
    if resolved is None or not stat.S_ISREG(resolved[1]):
        return None
    return resolved[0]


def _control_temporary_target(name: str) -> str | None:
    """Return the root control target for an exact atomic-write temp name."""

    for target in _CONTROL_FILENAMES:
        prefix = f".{target}."
        if name.startswith(prefix) and re.fullmatch(
            _TEMPFILE_TOKEN, name[len(prefix) :]
        ):
            return target
    return None


def _looks_like_snapshot_temporary(name: str) -> bool:
    """Identify namespace lookalikes that must remain evidence, not be removed."""

    if name.startswith(".") and (".copying" in name or name.endswith("copying")):
        return True
    return any(name.startswith(f".{target}.") for target in _CONTROL_FILENAMES)


def _durably_remove_interrupted_temporary(path: Path) -> None:
    """Unlink one proven private regular file and persist its directory update."""

    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SnapshotError(
            f"cannot inspect interrupted snapshot temporary: {path}"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise SnapshotError(f"unsafe interrupted snapshot temporary: {path}")
    if before.st_nlink != 1:
        raise SnapshotError(f"hardlinked interrupted snapshot temporary: {path}")
    if before.st_uid != os.geteuid():
        raise SnapshotError(f"foreign-owned interrupted snapshot temporary: {path}")

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(path.parent, flags)
    except OSError as exc:
        raise SnapshotError(
            f"unsafe temporary parent directory: {path.parent}"
        ) from exc
    try:
        try:
            current = os.stat(
                path.name, dir_fd=directory_descriptor, follow_symlinks=False
            )
        except OSError as exc:
            raise SnapshotError(
                f"interrupted snapshot temporary changed before cleanup: {path}"
            ) from exc
        if (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_nlink,
            current.st_uid,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
        ):
            raise SnapshotError(
                f"interrupted snapshot temporary changed before cleanup: {path}"
            )
        os.unlink(path.name, dir_fd=directory_descriptor)
        try:
            os.fsync(directory_descriptor)
        except OSError as exc:
            raise SnapshotError(
                f"cannot durably record interrupted temporary cleanup: {path.parent}"
            ) from exc
    finally:
        os.close(directory_descriptor)


def _cleanup_interrupted_temporaries(
    snapshot_root: Path, sources: tuple[Source, ...]
) -> tuple[str, ...]:
    """Remove only exact, safely mapped temporaries from an unsealed snapshot.

    Source copy temporaries must have the precise ``mkstemp`` name generated by
    :func:`_copy_one`, live beside their intended destination, and map to a regular
    file in the current source set.  Atomic control temporaries are accepted only in
    the snapshot root.  Symlinks, multiply-linked files, special nodes, malformed
    lookalikes, and exact names without a current target all fail closed.
    """

    sources_by_name = {source.name: source for source in sources}
    removable: list[Path] = []
    for root, directory_names, file_names in os.walk(
        snapshot_root, topdown=True, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        root_path = Path(root)
        relative_parent = root_path.relative_to(snapshot_root)
        for name in directory_names:
            path = root_path / name
            if path.is_symlink():
                raise SnapshotError(f"symlink in incomplete snapshot: {path}")
            if _looks_like_snapshot_temporary(name):
                logical_node = relative_parent / name
                source_node = _source_node_for_logical_path(
                    logical_node, sources_by_name
                )
                if source_node is None or not stat.S_ISDIR(source_node[1]):
                    raise SnapshotError(
                        f"temporary lookalike is not a regular file: {path}"
                    )
        for name in file_names:
            path = root_path / name
            copy_match = _COPY_TEMPORARY.fullmatch(name)
            control_target = (
                _control_temporary_target(name)
                if relative_parent == Path(".")
                else None
            )
            looks_temporary = _looks_like_snapshot_temporary(name)
            if copy_match is None and control_target is None and not looks_temporary:
                continue
            logical_node = relative_parent / name
            source_node = _source_node_for_logical_path(logical_node, sources_by_name)
            is_expected_file = source_node is not None and stat.S_ISREG(source_node[1])
            mapped = False
            if copy_match is not None:
                logical_target = relative_parent / copy_match.group("target")
                mapped = (
                    _source_target_for_logical_path(logical_target, sources_by_name)
                    is not None
                )
            if control_target is not None:
                mapped = True
            if mapped:
                if is_expected_file:
                    raise SnapshotError(
                        f"ambiguous expected file and snapshot temporary: {path}"
                    )
                # Inspect all candidates before deleting any of them.  A hostile or
                # damaged sibling therefore fails without partially cleaning the set.
                try:
                    info = path.stat(follow_symlinks=False)
                except OSError as exc:
                    raise SnapshotError(
                        f"cannot inspect interrupted snapshot temporary: {path}"
                    ) from exc
                if path.is_symlink() or not stat.S_ISREG(info.st_mode):
                    raise SnapshotError(
                        f"unsafe interrupted snapshot temporary: {path}"
                    )
                if info.st_nlink != 1:
                    raise SnapshotError(
                        f"hardlinked interrupted snapshot temporary: {path}"
                    )
                if info.st_uid != os.geteuid():
                    raise SnapshotError(
                        f"foreign-owned interrupted snapshot temporary: {path}"
                    )
                removable.append(path)
            elif (
                copy_match is not None or control_target is not None or looks_temporary
            ) and not is_expected_file:
                raise SnapshotError(
                    f"unmapped or malformed snapshot temporary lookalike: {path}"
                )

    removed: list[str] = []
    for path in removable:
        _durably_remove_interrupted_temporary(path)
        removed.append(path.relative_to(snapshot_root).as_posix())
    return tuple(removed)


def _validate_destination_shape(
    snapshot_root: Path,
    expected_records: Iterable[FileRecord],
    expected_directories: Iterable[str],
) -> None:
    """Reject every destination node not explained by a source or control file."""

    expected_files = {record.logical_path for record in expected_records}
    expected_dirs = set(expected_directories)
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()
    for root, directory_names, file_names in os.walk(
        snapshot_root, topdown=True, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        root_path = Path(root)
        for name in directory_names:
            path = root_path / name
            if path.is_symlink():
                raise SnapshotError(f"symlink in snapshot destination: {path}")
            observed_dirs.add(path.relative_to(snapshot_root).as_posix())
        for name in file_names:
            path = root_path / name
            if path.is_symlink() or not path.is_file():
                raise SnapshotError(f"unsafe file in snapshot destination: {path}")
            relative = path.relative_to(snapshot_root).as_posix()
            if path.parent == snapshot_root and name in _CONTROL_FILENAMES:
                continue
            observed_files.add(relative)
    extra_files = sorted(observed_files - expected_files)
    extra_dirs = sorted(observed_dirs - expected_dirs)
    if extra_files or extra_dirs:
        raise SnapshotError(
            "unexplained entries in resumable snapshot destination: "
            f"files={extra_files[:5]}, directories={extra_dirs[:5]}"
        )


def _load_inventory(path: Path) -> tuple[FileRecord, ...]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise SnapshotError(f"cannot read inventory {path}: {exc}") from exc
    records: list[FileRecord] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        digest, separator, logical = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest) or not logical:
            raise SnapshotError(f"invalid inventory line {path}:{line_number}")
        relative = Path(logical)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != logical
            or logical in _CONTROL_FILENAMES
        ):
            raise SnapshotError(
                f"unsafe or non-canonical inventory path {path}:{line_number}"
            )
        candidate = path.parent / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise SnapshotError(f"inventory file is missing or unsafe: {candidate}")
        info = candidate.stat()
        if info.st_nlink != 1:
            raise SnapshotError(f"inventory file is hardlinked: {candidate}")
        records.append(FileRecord(logical, digest, info.st_size))
    result = tuple(records)
    logical_paths = tuple(record.logical_path for record in result)
    if result != tuple(sorted(result)) or len(set(logical_paths)) != len(result):
        raise SnapshotError("snapshot inventory is unsorted or contains duplicates")
    # This also rejects missing/final extra newlines, alternate separators, and other
    # non-canonical row encodings.  The source and snapshot inventories are evidence,
    # not a permissive interchange format.
    if raw != _inventory_bytes(result):
        raise SnapshotError("snapshot inventory rows are not canonically encoded")
    return result


def _require_read_only_regular(path: Path, *, context: str) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise SnapshotError(f"{context} is missing or unreadable: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise SnapshotError(f"{context} is missing or unsafe: {path}")
    if info.st_nlink != 1:
        raise SnapshotError(f"{context} is hardlinked: {path}")
    if stat.S_IMODE(info.st_mode) & 0o222:
        raise SnapshotError(f"{context} is writable: {path}")


def _load_json_object(path: Path, *, context: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot parse {context}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"{context} must be a JSON object: {path}")
    return value


def _verify_snapshot(
    snapshot_root: Path, *, _allow_writable_root: bool = False
) -> dict[str, object]:
    if snapshot_root.is_symlink() or not snapshot_root.is_dir():
        raise SnapshotError(f"snapshot root is missing or unsafe: {snapshot_root}")
    complete_path = snapshot_root / COMPLETE_FILENAME
    if complete_path.is_symlink() or not complete_path.is_file():
        raise SnapshotError(f"snapshot is not sealed: {snapshot_root}")
    for filename in sorted(_CONTROL_FILENAMES):
        _require_read_only_regular(
            snapshot_root / filename, context="sealed snapshot control artifact"
        )
    source_inventory = snapshot_root / SOURCE_INVENTORY_FILENAME
    copy_inventory = snapshot_root / SNAPSHOT_INVENTORY_FILENAME
    if source_inventory.read_bytes() != copy_inventory.read_bytes():
        raise SnapshotError("source and snapshot inventories differ")
    records = _load_inventory(copy_inventory)
    directory_path = snapshot_root / DIRECTORY_INVENTORY_FILENAME
    try:
        directory_raw = directory_path.read_bytes()
        directories = tuple(directory_raw.decode("utf-8").splitlines())
    except (OSError, UnicodeError) as exc:
        raise SnapshotError(f"cannot read directory inventory: {exc}") from exc
    if (
        tuple(sorted(directories)) != directories
        or len(set(directories)) != len(directories)
        or directory_raw != _directory_bytes(directories)
    ):
        raise SnapshotError("directory inventory is unsorted or contains duplicates")
    for logical in directories:
        relative = Path(logical)
        path = snapshot_root / relative
        if (
            not logical
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != logical
            or path.is_symlink()
            or not path.is_dir()
        ):
            raise SnapshotError(f"inventory directory is missing or unsafe: {logical}")
    _validate_destination_shape(snapshot_root, records, directories)
    for index, record in enumerate(records, 1):
        path = snapshot_root / record.logical_path
        if _sha256(path) != record.sha256:
            raise SnapshotError(f"sealed file checksum mismatch: {record.logical_path}")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise SnapshotError(f"sealed file is writable: {record.logical_path}")
        if index % 5000 == 0:
            print(f"verified {index:,}/{len(records):,} files", flush=True)
    catalog = _load_json_object(
        snapshot_root / CATALOG_FILENAME, context="snapshot catalog"
    )
    if (
        not isinstance(catalog.get("file_count"), int)
        or isinstance(catalog.get("file_count"), bool)
        or catalog.get("file_count") != len(records)
    ):
        raise SnapshotError("catalog file count disagrees with inventory")
    total_bytes = sum(record.size for record in records)
    if (
        not isinstance(catalog.get("total_bytes"), int)
        or isinstance(catalog.get("total_bytes"), bool)
        or catalog.get("total_bytes") != total_bytes
    ):
        raise SnapshotError("catalog byte count disagrees with inventory")
    marker = _load_json_object(complete_path, context="snapshot completion marker")
    if set(marker) != _COMPLETION_MARKER_KEYS:
        raise SnapshotError(
            "completion marker schema mismatch: "
            f"expected={sorted(_COMPLETION_MARKER_KEYS)}, observed={sorted(marker)}"
        )
    if (
        not isinstance(marker.get("schema_version"), int)
        or isinstance(marker.get("schema_version"), bool)
        or marker.get("schema_version") != 1
    ):
        raise SnapshotError("completion marker schema version is unsupported")
    snapshot_id = marker.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise SnapshotError("completion marker snapshot id is invalid")
    _parse_utc_timestamp(marker.get("completed_at"), context="completed_at")
    if marker.get("snapshot_id") != catalog.get("snapshot_id"):
        raise SnapshotError("completion marker snapshot id disagrees with catalog")
    expected = hashlib.sha256(copy_inventory.read_bytes()).hexdigest()
    if (
        not isinstance(marker.get("file_count"), int)
        or isinstance(marker.get("file_count"), bool)
        or marker.get("file_count") != len(records)
    ):
        raise SnapshotError("completion marker file count disagrees with inventory")
    if (
        not isinstance(marker.get("total_bytes"), int)
        or isinstance(marker.get("total_bytes"), bool)
        or marker.get("total_bytes") != total_bytes
    ):
        raise SnapshotError("completion marker byte count disagrees with inventory")
    if (
        not isinstance(marker.get("snapshot_inventory_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(marker.get("snapshot_inventory_sha256"))
        )
        or marker.get("snapshot_inventory_sha256") != expected
    ):
        raise SnapshotError("completion marker inventory hash mismatch")
    if marker.get("verified") is not True:
        raise SnapshotError("completion marker does not assert verified=true")
    if marker.get("read_only") is not True:
        raise SnapshotError("completion marker does not assert read_only=true")
    for logical in directories:
        if stat.S_IMODE((snapshot_root / logical).stat().st_mode) & 0o222:
            raise SnapshotError(f"sealed directory is writable: {logical}")
    if (
        stat.S_IMODE(snapshot_root.stat().st_mode) & 0o222
        and not _allow_writable_root
    ):
        raise SnapshotError("sealed snapshot root is writable")
    return {
        "status": "already_complete",
        "snapshot_root": str(snapshot_root),
        "snapshot_id": snapshot_id,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "snapshot_inventory_sha256": expected,
        "verified_at": _utc_now(),
    }


def verify_snapshot(snapshot_root: Path) -> dict[str, object]:
    """Strictly verify one fully sealed snapshot."""

    return _verify_snapshot(snapshot_root, _allow_writable_root=False)


def write_snapshot_attestation(
    snapshot_root: Path, attestation_path: Path
) -> dict[str, object]:
    """Publish an external checksum envelope for every snapshot control artifact.

    The first live pre-repair snapshot uses the already-running schema-1 marker.  This
    independently stored envelope closes that marker's one limitation: it binds the
    catalog and directory inventory as well as both per-file inventories.  It must live
    outside the sealed snapshot so ``SNAPSHOT_COMPLETE.json`` remains its last internal
    publication.
    """

    snapshot_root = snapshot_root.expanduser().resolve()
    attestation_path = attestation_path.expanduser().absolute()
    try:
        attestation_path.resolve().relative_to(snapshot_root)
    except ValueError:
        pass
    else:
        raise SnapshotError(
            "snapshot attestation must be stored outside the sealed snapshot"
        )
    verified = verify_snapshot(snapshot_root)
    controls: dict[str, dict[str, object]] = {}
    for filename in (
        COMPLETE_FILENAME,
        CATALOG_FILENAME,
        SOURCE_INVENTORY_FILENAME,
        SNAPSHOT_INVENTORY_FILENAME,
        DIRECTORY_INVENTORY_FILENAME,
    ):
        path = snapshot_root / filename
        controls[filename] = {"sha256": _sha256(path), "size": path.stat().st_size}
    fixed_payload: dict[str, object] = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "kind": "recovery_snapshot_external_attestation",
        "passed": True,
        "snapshot_root": str(snapshot_root),
        "snapshot_id": verified["snapshot_id"],
        "file_count": verified["file_count"],
        "total_bytes": verified["total_bytes"],
        "control_artifacts": controls,
    }

    def validate_existing() -> dict[str, object]:
        _require_read_only_regular(
            attestation_path, context="existing snapshot attestation"
        )
        try:
            raw = attestation_path.read_bytes()
        except OSError as exc:
            raise SnapshotError(
                f"cannot read existing snapshot attestation: {attestation_path}"
            ) from exc
        payload = _load_json_object(
            attestation_path, context="existing snapshot attestation"
        )
        if set(payload) != _ATTESTATION_KEYS:
            raise SnapshotError("existing snapshot attestation schema mismatch")
        attested_at = payload.get("attested_at")
        attested_timestamp = _parse_utc_timestamp(
            attested_at, context="attestation attested_at"
        )
        marker = _load_json_object(
            snapshot_root / COMPLETE_FILENAME, context="snapshot completion marker"
        )
        completed_timestamp = _parse_utc_timestamp(
            marker.get("completed_at"), context="completed_at"
        )
        if attested_timestamp < completed_timestamp:
            raise SnapshotError("snapshot attestation predates snapshot completion")
        expected_payload = fixed_payload | {"attested_at": attested_at}
        expected_bytes = (
            json.dumps(expected_payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if raw != expected_bytes:
            raise SnapshotError(
                "existing snapshot attestation differs from the exact sealed snapshot"
            )
        return payload

    if os.path.lexists(attestation_path):
        payload = validate_existing()
    else:
        payload = fixed_payload | {"attested_at": _utc_now()}
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{attestation_path.name}.", dir=attestation_path.parent
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o444)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # Hard-link publication is atomic and, unlike replace, never clobbers
                # an attestation that another process published concurrently.
                os.link(temporary, attestation_path)
                _fsync_directory(attestation_path.parent)
            except FileExistsError:
                pass
        finally:
            try:
                temporary.unlink()
                _fsync_directory(attestation_path.parent)
            except FileNotFoundError:
                pass
        payload = validate_existing()
    return payload | {
        "attestation_path": str(attestation_path),
        "attestation_sha256": _sha256(attestation_path),
    }


def verify_complete_snapshot(
    snapshot_root: Path, sources: Iterable[Source] | None = None
) -> dict[str, object]:
    """Compatibility wrapper for explicit completed-snapshot verification.

    Source paths are deliberately not re-read: the purpose of the sealed snapshot is to
    remain independently verifiable after the live sources are repaired or retired.
    """

    del sources
    return verify_snapshot(Path(snapshot_root))


def _create_snapshot_unlocked(
    snapshot_root: Path,
    sources: tuple[Source, ...] | list[Source],
    *,
    apply: bool = True,
) -> dict[str, object]:
    sources = tuple(sources)
    if len({source.name for source in sources}) != len(sources):
        raise SnapshotError("source names must be unique")
    reserved = sorted(
        source.name for source in sources if source.name in _CONTROL_FILENAMES
    )
    if reserved:
        raise SnapshotError(f"source names collide with snapshot controls: {reserved}")
    resolved_root = snapshot_root.absolute()
    for source in sources:
        try:
            resolved_root.relative_to(source.path)
        except ValueError:
            pass
        else:
            raise SnapshotError(f"snapshot destination is inside source: {source.path}")
    if snapshot_root.is_symlink():
        raise SnapshotError(f"snapshot root may not be a symlink: {snapshot_root}")
    complete = snapshot_root / COMPLETE_FILENAME
    if complete.exists() or complete.is_symlink():
        # The marker must be published before the root directory can lose its write
        # bit.  A process can therefore stop in the single crash window between those
        # operations.  On an applying retry, validate *every* marker/content/directory
        # invariant while relaxing only the root mode, finish that one pending chmod,
        # and then run the ordinary strict verifier.  Invalid marker claims or any
        # other drift still fail before a permission is changed.
        root_mode = (
            stat.S_IMODE(snapshot_root.stat().st_mode)
            if snapshot_root.is_dir() and not snapshot_root.is_symlink()
            else 0
        )
        if apply and root_mode & 0o222:
            _verify_snapshot(snapshot_root, _allow_writable_root=True)
            snapshot_root.chmod(root_mode & ~0o222)
            _fsync_directory(snapshot_root.parent)
        return verify_snapshot(snapshot_root)
    if not apply:
        expected_records, expected_directories = _inventory_sources(sources)
        return {
            "status": "dry_run",
            "snapshot_root": str(snapshot_root),
            "sources": [
                {"name": source.name, "path": str(source.path)} for source in sources
            ],
            "file_count": len(expected_records),
            "directory_count": len(expected_directories),
            "total_bytes": sum(record.size for record in expected_records),
            "would_publish_last": COMPLETE_FILENAME,
        }
    _make_incomplete_snapshot_writable(snapshot_root)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    removed_temporaries = _cleanup_interrupted_temporaries(snapshot_root, sources)
    if removed_temporaries:
        print(
            "removed interrupted snapshot temporaries before inventory: "
            + ", ".join(removed_temporaries),
            flush=True,
        )
    expected_records, expected_directories = _inventory_sources(sources)
    _validate_destination_shape(snapshot_root, expected_records, expected_directories)

    records: list[FileRecord] = []
    directories: set[str] = set()
    started = time.monotonic()
    last_report = started
    for source in sources:
        for logical, source_path, is_directory in _walk_source(source):
            destination = snapshot_root / logical
            if is_directory:
                destination.mkdir(parents=True, exist_ok=True)
                directories.add(logical)
                continue
            record = _copy_one(source_path, destination, logical)
            records.append(record)
            now = time.monotonic()
            if len(records) % 5000 == 0 or now - last_report >= 60:
                print(
                    f"copied+verified {len(records):,} files, "
                    f"{sum(item.size for item in records) / (1024**3):.2f} GiB",
                    flush=True,
                )
                last_report = now

    records.sort()
    if (
        tuple(records) != expected_records
        or tuple(sorted(directories)) != expected_directories
    ):
        raise SnapshotError("source inventory changed between audit and copy")
    final_records, final_directories = _inventory_sources(sources)
    if final_records != expected_records or final_directories != expected_directories:
        raise SnapshotError("source changed before final snapshot publication")
    _validate_destination_shape(snapshot_root, expected_records, expected_directories)

    inventory = _inventory_bytes(records)
    _atomic_bytes(snapshot_root / SOURCE_INVENTORY_FILENAME, inventory)
    _atomic_bytes(snapshot_root / SNAPSHOT_INVENTORY_FILENAME, inventory)
    _atomic_bytes(
        snapshot_root / DIRECTORY_INVENTORY_FILENAME, _directory_bytes(directories)
    )
    inventory_sha = hashlib.sha256(inventory).hexdigest()
    catalog = {
        "schema_version": 1,
        "snapshot_id": f"schema5-v1-pre-repair-{inventory_sha[:16]}",
        "created_at": _utc_now(),
        "copy_contract": "independent_regular_files_no_hardlinks_no_symlinks",
        "sources": [asdict(source) | {"path": str(source.path)} for source in sources],
        "file_count": len(records),
        "directory_count": len(directories),
        "total_bytes": sum(record.size for record in records),
        "source_inventory_sha256": inventory_sha,
        "snapshot_inventory_sha256": inventory_sha,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(snapshot_root / CATALOG_FILENAME, catalog)

    # Verify every copied byte before sealing, then remove write permission from all
    # content.  Keep the root writable only long enough to publish the marker last.
    for record in records:
        path = snapshot_root / record.logical_path
        if _sha256(path) != record.sha256:
            raise SnapshotError(f"post-copy verification failed: {record.logical_path}")
    for path in sorted(
        (candidate for candidate in snapshot_root.rglob("*") if candidate != complete),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        if path.is_file():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        elif path.is_dir():
            path.chmod(0o555)
    marker = {
        "schema_version": 1,
        "snapshot_id": catalog["snapshot_id"],
        "completed_at": _utc_now(),
        "file_count": len(records),
        "total_bytes": catalog["total_bytes"],
        "snapshot_inventory_sha256": inventory_sha,
        "verified": True,
        "read_only": True,
    }
    _atomic_json(complete, marker, mode=0o444)
    snapshot_root.chmod(0o555)
    _fsync_directory(snapshot_root.parent)
    return marker | {"status": "created", "snapshot_root": str(snapshot_root)}


def create_snapshot(
    snapshot_root: Path,
    sources: tuple[Source, ...] | list[Source],
    *,
    apply: bool = True,
) -> dict[str, object]:
    """Create/audit a snapshot under a cross-process same-parent advisory lock."""

    snapshot_root = Path(snapshot_root)
    if not apply:
        return _create_snapshot_unlocked(snapshot_root, sources, apply=False)
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = snapshot_root.parent / f".{snapshot_root.name}.snapshot.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise SnapshotError(
                    f"another snapshot process holds {lock_path}"
                ) from exc
            raise
        return _create_snapshot_unlocked(snapshot_root, sources, apply=True)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", "--destination", required=True, type=Path)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--attestation-path",
        type=Path,
        help="with --verify-only, publish an external envelope binding all control files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inventory sources and report exact bytes without writing the destination",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    snapshot_root = args.snapshot_root.absolute()
    if args.verify_only:
        result = (
            write_snapshot_attestation(snapshot_root, args.attestation_path)
            if args.attestation_path is not None
            else verify_snapshot(snapshot_root)
        )
    else:
        if args.attestation_path is not None:
            raise SnapshotError("--attestation-path requires --verify-only")
        if not args.source:
            raise SnapshotError("at least one --source NAME=/absolute/path is required")
        result = create_snapshot(
            snapshot_root,
            tuple(_parse_source(item) for item in args.source),
            apply=not args.dry_run,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
