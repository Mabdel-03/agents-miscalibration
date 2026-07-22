#!/usr/bin/env python3
"""Restore one named source from a sealed recovery snapshot without overwriting data.

The restore destination must not exist.  Bytes are copied into a same-filesystem staging
path, checked against the snapshot's sorted SHA-256 inventory, fsynced, and published by
one atomic rename.  A sibling ``.<name>.RESTORE_COMPLETE.json`` record is written last.
The operation never uses hardlinks and is idempotently verifiable after success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
try:  # package-style tests/imports
    from scripts.create_recovery_snapshot import (
        SNAPSHOT_INVENTORY_FILENAME,
        SnapshotError,
        verify_snapshot,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from create_recovery_snapshot import (  # type: ignore[no-redef]
        SNAPSHOT_INVENTORY_FILENAME,
        SnapshotError,
        verify_snapshot,
    )


_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_CHUNK = 8 * 1024 * 1024


class RestoreError(RuntimeError):
    """A restore cannot be proven exact and non-destructive."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _inventory_for_source(snapshot_root: Path, source_name: str) -> dict[str, str]:
    prefix = source_name + "/"
    records: dict[str, str] = {}
    inventory = snapshot_root / SNAPSHOT_INVENTORY_FILENAME
    for line_number, line in enumerate(
        inventory.read_text(encoding="utf-8").splitlines(), 1
    ):
        digest, separator, logical = line.partition("  ")
        if not separator or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RestoreError(f"invalid snapshot inventory line {line_number}")
        if logical == source_name:
            records[""] = digest
        elif logical.startswith(prefix):
            records[logical[len(prefix) :]] = digest
    if not records:
        raise RestoreError(f"snapshot contains no source named {source_name!r}")
    return records


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".copying", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer, length=_CHUNK)
            writer.flush()
            os.fsync(writer.fileno())
        os.chmod(temporary, stat.S_IMODE(source.stat().st_mode))
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _verify_restored(destination: Path, records: dict[str, str]) -> None:
    observed: set[str] = set()
    if "" in records:
        if destination.is_symlink() or not destination.is_file():
            raise RestoreError(f"restored file is missing or unsafe: {destination}")
        if _sha256(destination) != records[""]:
            raise RestoreError("restored file checksum mismatch")
        observed.add("")
    else:
        if destination.is_symlink() or not destination.is_dir():
            raise RestoreError(f"restored directory is missing or unsafe: {destination}")
        for path in destination.rglob("*"):
            if path.is_symlink():
                raise RestoreError(f"symlink appeared in restored tree: {path}")
            if path.is_file():
                relative = path.relative_to(destination).as_posix()
                observed.add(relative)
                if records.get(relative) != _sha256(path):
                    raise RestoreError(f"restored checksum mismatch: {relative}")
    if observed != set(records):
        raise RestoreError(
            "restored file set differs from snapshot inventory: "
            f"missing={sorted(set(records) - observed)[:5]}, "
            f"extra={sorted(observed - set(records))[:5]}"
        )


def restore_source(
    snapshot_root: Path,
    *,
    source_name: str,
    destination: Path,
) -> dict[str, object]:
    if _SAFE_NAME.fullmatch(source_name) is None:
        raise RestoreError(f"unsafe snapshot source name: {source_name!r}")
    snapshot_root = snapshot_root.expanduser().resolve()
    destination = destination.expanduser().absolute()
    if destination == Path("/"):
        raise RestoreError("refusing filesystem root as a restore destination")
    marker = destination.parent / f".{destination.name}.RESTORE_COMPLETE.json"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if (
            payload.get("snapshot_root") != str(snapshot_root)
            or payload.get("source_name") != source_name
            or payload.get("destination") != str(destination)
        ):
            raise RestoreError(f"restore marker does not describe requested target: {marker}")
        verified = verify_snapshot(snapshot_root)
        records = _inventory_for_source(snapshot_root, source_name)
        _verify_restored(destination, records)
        return payload | {"status": "already_restored", "snapshot_id": verified["snapshot_id"]}
    if destination.exists() or destination.is_symlink():
        raise RestoreError(f"refusing to overwrite existing restore destination: {destination}")
    verified = verify_snapshot(snapshot_root)
    records = _inventory_for_source(snapshot_root, source_name)
    source = snapshot_root / source_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.restore-incomplete"
    if staging.exists() or staging.is_symlink():
        raise RestoreError(f"incomplete restore staging path needs inspection: {staging}")
    if source.is_file():
        _copy_file(source, staging)
    elif source.is_dir():
        staging.mkdir(mode=0o700)
        for relative in sorted(records):
            if not relative:
                raise RestoreError("directory source has invalid root-file inventory entry")
            _copy_file(source / relative, staging / relative)
    else:
        raise RestoreError(f"snapshot source is missing or unsafe: {source}")
    _verify_restored(staging, records)
    os.replace(staging, destination)
    _fsync_directory(destination.parent)
    _verify_restored(destination, records)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "restored",
        "snapshot_root": str(snapshot_root),
        "snapshot_id": verified["snapshot_id"],
        "source_name": source_name,
        "destination": str(destination),
        "file_count": len(records),
        "total_bytes": sum(
            (destination if relative == "" else destination / relative).stat().st_size
            for relative in records
        ),
        "source_inventory_sha256": hashlib.sha256(
            "".join(f"{records[key]}  {key}\n" for key in sorted(records)).encode()
        ).hexdigest(),
    }
    _atomic_json(marker, payload)
    return payload | {"restore_marker": str(marker)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = restore_source(
            args.snapshot_root,
            source_name=args.source_name,
            destination=args.destination,
        )
    except (OSError, ValueError, SnapshotError, RestoreError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
