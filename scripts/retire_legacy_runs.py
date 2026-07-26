#!/usr/bin/env python3
"""Retire the three legacy sweep roots and dispatcher state as read-only evidence.

The operation is deliberately last in legacy consolidation.  It requires a verified
``legacy_consolidated`` snapshot and external snapshot attestation, writes an intent
marker into each exact allowlisted root, removes write bits without following symlinks,
verifies every node, and publishes one external completion record last.  Dry-run is the
default; completed retirement is idempotently revalidated.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.create_recovery_snapshot import verify_snapshot  # noqa: E402


LEGACY_ROOT_NAMES = (
    "full_sweep_v1",
    "full_sweep_agent_counts_v1",
    "full_sweep_agent_count_7_v1",
    ".dispatcher-v3",
)
ROOT_MARKER = "RETIRED_SCHEMA5_V1.json"
COMPLETE_MARKER = "LEGACY_RETIREMENT_COMPLETE.json"
LEGACY_CLEANUP_MARKER = "LEGACY_CLEANUP_COMPLETE.json"
MAINTENANCE_INTERLOCK = "MAINTENANCE_INTERLOCK.json"
RETIREMENT_LOCK = ".legacy_retirement.lock"

_PRECHECK_FIELDS = frozenset(
    {
        "maintenance_interlock",
        "scheduler_rows",
        "legacy_jobs",
        "inspected_cell_locks",
        "held_cell_locks",
    }
)
_COMPLETE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "results_root",
        "roots",
        "node_counts",
        "snapshot_id",
        "snapshot_attestation_sha256",
        "source_verification",
        "precheck",
        "seal_prechecks",
        "final_precheck",
        "verified_nodes",
    }
)


class RetirementError(RuntimeError):
    """Legacy evidence cannot be safely or completely retired."""


def _snapshot_source_name(root_name: str) -> str:
    return root_name.lstrip(".").replace("-", "_").replace(".", "_") or "dispatcher"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
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


@contextmanager
def _retirement_lock(recovery_root: Path):
    """Serialize the marker/chmod/verification transaction across nodes."""

    path = recovery_root / RETIREMENT_LOCK
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise RetirementError(
                    f"another legacy retirement owns {path}"
                ) from exc
            raise
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _cell_lock_paths(results_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for run_id in LEGACY_ROOT_NAMES[:3]:
        cells_root = results_root / run_id / "cells"
        if cells_root.is_symlink():
            raise RetirementError(f"legacy cells root is a symlink: {cells_root}")
        if not cells_root.is_dir():
            continue
        for cell_dir in sorted(cells_root.iterdir(), key=lambda item: item.name):
            if cell_dir.is_symlink() or not cell_dir.is_dir():
                continue
            lock_path = cell_dir / ".cell.lock"
            if not lock_path.exists():
                continue
            if lock_path.is_symlink() or not lock_path.is_file():
                raise RetirementError(f"unsafe legacy cell lock: {lock_path}")
            paths.append(lock_path.resolve())
    return tuple(paths)


@contextmanager
def _hold_all_cell_locks(results_root: Path):
    """Fence every extant worker lock through chmod, post-hash, and marker publish."""

    before = _cell_lock_paths(results_root)
    descriptors: list[int] = []
    try:
        for path in before:
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(descriptor)
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise RetirementError(f"legacy cell became active: {path}") from exc
                raise
            descriptors.append(descriptor)
        after = _cell_lock_paths(results_root)
        if before != after:
            raise RetirementError("legacy cell-lock set changed during retirement fencing")
        yield frozenset(before)
    finally:
        for descriptor in reversed(descriptors):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _read_strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read one regular JSON object while rejecting duplicate object keys."""

    if path.is_symlink() or not path.is_file():
        raise RetirementError(f"{label} is missing or unsafe: {path}")

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise RetirementError(f"{label} has duplicate key {key!r}: {path}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_without_duplicates,
        )
    except RetirementError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetirementError(f"cannot parse {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RetirementError(f"{label} is not a JSON object: {path}")
    return value


def _load_snapshot_attestation(
    path: Path, snapshot_root: Path, *, expected_snapshot_id: str
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetirementError(f"cannot read legacy snapshot attestation: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "recovery_snapshot_external_attestation"
        or payload.get("passed") is not True
        or Path(str(payload.get("snapshot_root", ""))).resolve() != snapshot_root
        or payload.get("snapshot_id") != expected_snapshot_id
    ):
        raise RetirementError("legacy snapshot attestation has the wrong identity")
    controls = payload.get("control_artifacts")
    if not isinstance(controls, dict) or len(controls) != 5:
        raise RetirementError("legacy snapshot attestation has an incomplete control inventory")
    for filename, record in controls.items():
        candidate = snapshot_root / filename
        if (
            not isinstance(record, dict)
            or candidate.is_symlink()
            or not candidate.is_file()
            or candidate.stat().st_size != record.get("size")
            or _sha256(candidate) != record.get("sha256")
        ):
            raise RetirementError(f"legacy snapshot control artifact drifted: {filename}")
    return payload


def _walk_nodes(root: Path) -> list[Path]:
    nodes = [root]
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                raise RetirementError(f"refusing symlink in legacy evidence: {path}")
            nodes.append(path)
        for name in file_names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise RetirementError(f"refusing unsafe legacy evidence node: {path}")
            nodes.append(path)
    return nodes


def _verify_read_only(roots: Iterable[Path]) -> int:
    count = 0
    for root in roots:
        for path in _walk_nodes(root):
            count += 1
            if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) & 0o222:
                raise RetirementError(f"retired evidence remains writable: {path}")
    return count


def _verify_consolidated_snapshot_sources(
    *, results_root: Path, recovery_root: Path, snapshot_root: Path, snapshot_id: str
) -> None:
    try:
        catalog = json.loads(
            (snapshot_root / "SNAPSHOT_CATALOG.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetirementError(f"cannot read consolidated snapshot catalog: {exc}") from exc
    expected_sources = [
        {
            "name": _snapshot_source_name(name),
            "path": str((results_root / name).resolve()),
        }
        for name in LEGACY_ROOT_NAMES
    ] + [
        {
            "name": "legacy_cleanup_evidence",
            "path": str((recovery_root / "operations" / "legacy_consolidation").resolve()),
        },
        {
            "name": "legacy_cleanup_complete",
            "path": str((recovery_root / LEGACY_CLEANUP_MARKER).resolve()),
        },
    ]
    if (
        not isinstance(catalog, dict)
        or catalog.get("schema_version") != 1
        or catalog.get("snapshot_id") != snapshot_id
        or catalog.get("copy_contract")
        != "independent_regular_files_no_hardlinks_no_symlinks"
        or catalog.get("sources") != expected_sources
    ):
        raise RetirementError(
            "legacy_consolidated snapshot does not cover the exact retired roots and cleanup evidence"
        )


def _maintenance_precheck(
    results_root: Path,
    recovery_root: Path | None = None,
    *,
    owned_lock_paths: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    """Re-prove scheduler and advisory-lock quiescence without creating lock files."""

    results_root = results_root.expanduser().resolve()
    recovery_root = (
        recovery_root.expanduser().resolve()
        if recovery_root is not None
        else results_root / "recovery" / "schema5-v1"
    )
    interlock_path = recovery_root / MAINTENANCE_INTERLOCK
    interlock = _read_strict_json_object(
        interlock_path, label="maintenance interlock"
    )
    if (
        interlock.get("desired_state") != "maintenance"
        or interlock.get("admission_enabled") is not False
        or tuple(interlock.get("retired_run_ids", ())) != LEGACY_ROOT_NAMES[:3]
    ):
        raise RetirementError(
            "maintenance interlock does not freeze the exact legacy runs"
        )

    scheduler_user = os.environ.get("USER")
    if not scheduler_user:
        raise RetirementError("USER is unset; scheduler quiescence cannot be scoped")
    proc = subprocess.run(
        ["squeue", "-u", scheduler_user, "-h", "-r", "-o", "%i|%j|%T|%o"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RetirementError(
            f"squeue failed during retirement check: {proc.stderr[:500]}"
        )
    unsafe_rows = [
        line
        for line in proc.stdout.splitlines()
        if any(run_id in line for run_id in LEGACY_ROOT_NAMES[:3])
        or any(
            prefix in line
            for prefix in ("asys-cells", "asys-dispatch", "asys-driver", "asys-serve")
        )
    ]
    if unsafe_rows:
        raise RetirementError(
            "legacy jobs remain live during retirement: "
            + "; ".join(unsafe_rows[:10])
        )

    inspected = 0
    held: list[str] = []
    for run_id in LEGACY_ROOT_NAMES[:3]:
        cells_root = results_root / run_id / "cells"
        if cells_root.is_symlink():
            raise RetirementError(f"legacy cells root is a symlink: {cells_root}")
        if not cells_root.is_dir():
            continue
        for cell_dir in cells_root.iterdir():
            if cell_dir.is_symlink() or not cell_dir.is_dir():
                continue
            lock_path = cell_dir / ".cell.lock"
            if not lock_path.exists():
                continue
            if lock_path.is_symlink() or not lock_path.is_file():
                raise RetirementError(f"unsafe legacy cell lock: {lock_path}")
            inspected += 1
            if lock_path.resolve() in owned_lock_paths:
                continue
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(lock_path, flags)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno in (errno.EACCES, errno.EAGAIN):
                        held.append(f"{run_id}/{cell_dir.name}")
                        continue
                    raise
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
    if held:
        raise RetirementError(f"held legacy cell locks remain: {held[:10]}")
    return {
        "maintenance_interlock": {
            "path": str(interlock_path),
            "sha256": _sha256(interlock_path),
        },
        "scheduler_rows": len(proc.stdout.splitlines()),
        "legacy_jobs": 0,
        "inspected_cell_locks": inspected,
        "held_cell_locks": 0,
    }


def _count_existing_cell_locks(results_root: Path) -> int:
    """Count the exact safe lock-file set without attempting scheduler admission."""

    count = 0
    for run_id in LEGACY_ROOT_NAMES[:3]:
        cells_root = results_root / run_id / "cells"
        if cells_root.is_symlink():
            raise RetirementError(f"legacy cells root is a symlink: {cells_root}")
        if not cells_root.is_dir():
            continue
        for cell_dir in cells_root.iterdir():
            if cell_dir.is_symlink() or not cell_dir.is_dir():
                continue
            lock_path = cell_dir / ".cell.lock"
            if not lock_path.exists():
                continue
            if lock_path.is_symlink() or not lock_path.is_file():
                raise RetirementError(f"unsafe legacy cell lock: {lock_path}")
            count += 1
    return count


def _validate_precheck_record(
    value: object,
    *,
    label: str,
    recovery_root: Path,
    expected_lock_count: int,
) -> None:
    """Validate one historical quiescence proof without requiring current idleness."""

    if not isinstance(value, dict) or set(value) != _PRECHECK_FIELDS:
        raise RetirementError(f"{label} has the wrong fields")
    interlock_path = recovery_root / MAINTENANCE_INTERLOCK
    expected_interlock = {
        "path": str(interlock_path),
        "sha256": _sha256(interlock_path),
    }
    integer_fields = (
        "scheduler_rows",
        "legacy_jobs",
        "inspected_cell_locks",
        "held_cell_locks",
    )
    if value.get("maintenance_interlock") != expected_interlock or any(
        isinstance(value.get(field), bool) or not isinstance(value.get(field), int)
        for field in integer_fields
    ):
        raise RetirementError(f"{label} is invalid")
    if (
        value["scheduler_rows"] < 0
        or value["legacy_jobs"] != 0
        or value["inspected_cell_locks"] != expected_lock_count
        or value["held_cell_locks"] != 0
    ):
        raise RetirementError(f"{label} did not prove quiescence")


def _load_snapshot_root_inventories(snapshot_root: Path) -> dict[str, dict[str, Any]]:
    """Load the exact four live-root subsets from the already-verified snapshot."""

    source_to_root = {
        _snapshot_source_name(root_name): root_name for root_name in LEGACY_ROOT_NAMES
    }
    inventories: dict[str, dict[str, Any]] = {
        root_name: {
            "files": {},
            "directories": set(),
            "inventory_lines": [],
            "directory_inventory_lines": [],
        }
        for root_name in LEGACY_ROOT_NAMES
    }
    try:
        file_lines = (snapshot_root / "SNAPSHOT_INVENTORY.sha256").read_text(
            encoding="utf-8"
        ).splitlines()
        directory_lines = (snapshot_root / "DIRECTORY_INVENTORY.txt").read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeError) as exc:
        raise RetirementError(f"cannot read consolidated snapshot inventories: {exc}") from exc

    for raw in file_lines:
        digest, separator, logical = raw.partition("  ")
        if not separator:
            raise RetirementError("consolidated snapshot file inventory is invalid")
        source_name, slash, suffix = logical.partition("/")
        root_name = source_to_root.get(source_name)
        if root_name is None:
            continue
        if not slash or not suffix or suffix == ROOT_MARKER:
            raise RetirementError(
                f"consolidated snapshot has an invalid retired-root file: {logical}"
            )
        snapshot_file = snapshot_root / logical
        inventories[root_name]["files"][suffix] = {
            "sha256": digest,
            "size": snapshot_file.stat().st_size,
        }
        inventories[root_name]["inventory_lines"].append(raw)

    for logical in directory_lines:
        source_name, slash, suffix = logical.partition("/")
        root_name = source_to_root.get(source_name)
        if root_name is None:
            continue
        relative = suffix if slash else "."
        inventories[root_name]["directories"].add(relative)
        inventories[root_name]["directory_inventory_lines"].append(logical)

    for root_name, inventory in inventories.items():
        if "." not in inventory["directories"]:
            raise RetirementError(
                f"consolidated snapshot omits exact root directory: {root_name}"
            )
        encoded = "".join(
            f"{line}\n" for line in sorted(inventory.pop("inventory_lines"))
        ).encode("utf-8")
        inventory["inventory_sha256"] = hashlib.sha256(encoded).hexdigest()
        directory_encoded = "".join(
            f"{line}\n"
            for line in sorted(inventory.pop("directory_inventory_lines"))
        ).encode("utf-8")
        inventory["directory_inventory_sha256"] = hashlib.sha256(
            directory_encoded
        ).hexdigest()
    return inventories


def _live_shape(root: Path) -> tuple[set[str], set[str]]:
    directories = {"."}
    files: set[str] = set()
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        for name in directory_names:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise RetirementError(f"refusing unsafe live legacy directory: {path}")
            directories.add(path.relative_to(root).as_posix())
        for name in file_names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise RetirementError(f"refusing unsafe live legacy file: {path}")
            relative = path.relative_to(root).as_posix()
            if path.parent == root and name == ROOT_MARKER:
                continue
            files.add(relative)
    return directories, files


def _stable_file_sha256(path: Path) -> tuple[str, int]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise RetirementError(f"live legacy source ceased to be regular: {path}")
    digest = _sha256(path)
    after = path.stat(follow_symlinks=False)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise RetirementError(f"live legacy source changed while hashing: {path}")
    return digest, before.st_size


def _verify_live_root(
    root: Path, inventory: Mapping[str, Any]
) -> dict[str, Any]:
    expected_files = inventory.get("files")
    expected_directories = inventory.get("directories")
    if not isinstance(expected_files, dict) or not isinstance(expected_directories, set):
        raise RetirementError(f"invalid snapshot source inventory for {root}")
    before_directories, before_files = _live_shape(root)
    if before_directories != expected_directories or before_files != set(expected_files):
        raise RetirementError(
            f"live legacy source shape drifted from consolidated snapshot: {root}"
        )
    total_bytes = 0
    for relative in sorted(expected_files):
        expected = expected_files[relative]
        digest, size = _stable_file_sha256(root / relative)
        if digest != expected.get("sha256") or size != expected.get("size"):
            raise RetirementError(
                f"live legacy source bytes drifted from consolidated snapshot: "
                f"{root / relative}"
            )
        total_bytes += size
    after_directories, after_files = _live_shape(root)
    if (after_directories, after_files) != (before_directories, before_files):
        raise RetirementError(f"live legacy source changed during inventory: {root}")
    return {
        "snapshot_source_name": _snapshot_source_name(root.name),
        "file_count": len(expected_files),
        "directory_count": len(expected_directories),
        "total_bytes": total_bytes,
        "source_inventory_sha256": inventory["inventory_sha256"],
        "source_directory_inventory_sha256": inventory[
            "directory_inventory_sha256"
        ],
    }


def _root_intent(
    *,
    root: Path,
    snapshot_id: str,
    attestation_sha256: str,
    source_verification: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "retirement_intent",
        "root": str(root),
        "snapshot_id": snapshot_id,
        "snapshot_attestation_sha256": attestation_sha256,
        "source_verification": dict(source_verification),
    }


def _validate_or_publish_root_marker(
    root: Path,
    expected: Mapping[str, Any],
    *,
    publish: bool,
    required: bool = False,
) -> None:
    marker = root / ROOT_MARKER
    if marker.exists():
        if marker.is_symlink() or not marker.is_file():
            raise RetirementError(f"unsafe retirement marker: {marker}")
        info = marker.stat(follow_symlinks=False)
        if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) & 0o222:
            raise RetirementError(f"retirement marker is not sealed: {marker}")
        observed = _read_strict_json_object(marker, label="retirement marker")
        if type(observed.get("schema_version")) is not int or observed != expected:
            raise RetirementError(f"retirement marker drifted: {marker}")
    elif publish:
        _atomic_json(marker, expected)
    elif required:
        raise RetirementError(f"retirement marker is missing: {marker}")


def _validate_completion_marker(
    payload: Mapping[str, Any],
    *,
    complete_path: Path,
    results_root: Path,
    recovery_root: Path,
    roots: tuple[Path, ...],
    snapshot_id: str,
    attestation_sha256: str,
    source_verification: Mapping[str, Any],
) -> int:
    """Strictly revalidate every durable claim in a completed retirement record."""

    _validate_completion_marker_header(
        payload,
        complete_path=complete_path,
        results_root=results_root,
        roots=roots,
        snapshot_id=snapshot_id,
        attestation_sha256=attestation_sha256,
    )
    if payload.get("source_verification") != source_verification:
        raise RetirementError("retirement completion marker has the wrong identity")

    node_counts = {root.name: len(_walk_nodes(root)) for root in roots}
    if payload.get("node_counts") != node_counts:
        raise RetirementError("retirement completion marker node counts drifted")
    verified_nodes = sum(node_counts.values())
    if payload.get("verified_nodes") != verified_nodes:
        raise RetirementError("retirement completion marker verified-node count drifted")

    lock_count = _count_existing_cell_locks(results_root)
    _validate_precheck_record(
        payload.get("precheck"),
        label="retirement initial precheck",
        recovery_root=recovery_root,
        expected_lock_count=lock_count,
    )
    seal_prechecks = payload.get("seal_prechecks")
    if not isinstance(seal_prechecks, dict) or set(seal_prechecks) != set(
        LEGACY_ROOT_NAMES
    ):
        raise RetirementError("retirement seal prechecks have the wrong roots")
    for root_name in LEGACY_ROOT_NAMES:
        _validate_precheck_record(
            seal_prechecks[root_name],
            label=f"retirement seal precheck for {root_name}",
            recovery_root=recovery_root,
            expected_lock_count=lock_count,
        )
    _validate_precheck_record(
        payload.get("final_precheck"),
        label="retirement final precheck",
        recovery_root=recovery_root,
        expected_lock_count=lock_count,
    )
    return verified_nodes


def _validate_completion_marker_header(
    payload: Mapping[str, Any],
    *,
    complete_path: Path,
    results_root: Path,
    roots: tuple[Path, ...],
    snapshot_id: str,
    attestation_sha256: str,
) -> None:
    """Fail fast on an unsafe or differently addressed completion marker."""

    info = complete_path.stat(follow_symlinks=False)
    if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) & 0o222:
        raise RetirementError("retirement completion marker is not sealed")
    if set(payload) != _COMPLETE_FIELDS:
        raise RetirementError("retirement completion marker has the wrong fields")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("status") != "retired"
        or payload.get("snapshot_id") != snapshot_id
        or payload.get("snapshot_attestation_sha256") != attestation_sha256
        or payload.get("results_root") != str(results_root)
        or payload.get("roots") != [str(root) for root in roots]
    ):
        raise RetirementError("retirement completion marker has the wrong identity")


def _retire_unlocked(
    *, results_root: Path, recovery_root: Path, apply: bool = False
) -> dict[str, Any]:
    results_root = results_root.expanduser().resolve()
    recovery_root = recovery_root.expanduser().resolve()
    roots = tuple(results_root / name for name in LEGACY_ROOT_NAMES)
    for name, root in zip(LEGACY_ROOT_NAMES, roots, strict=True):
        if root.name != name or root.is_symlink() or not root.is_dir():
            raise RetirementError(f"missing or unsafe exact legacy root: {root}")
    snapshot_root = recovery_root / "legacy_consolidated"
    verified_snapshot = verify_snapshot(snapshot_root)
    _verify_consolidated_snapshot_sources(
        results_root=results_root,
        recovery_root=recovery_root,
        snapshot_root=snapshot_root,
        snapshot_id=str(verified_snapshot["snapshot_id"]),
    )
    attestation_path = recovery_root / "legacy_consolidated.attestation.json"
    attestation = _load_snapshot_attestation(
        attestation_path,
        snapshot_root,
        expected_snapshot_id=str(verified_snapshot["snapshot_id"]),
    )
    del attestation
    source_inventories = _load_snapshot_root_inventories(snapshot_root)
    snapshot_id = str(verified_snapshot["snapshot_id"])
    attestation_sha256 = _sha256(attestation_path)
    complete_path = recovery_root / COMPLETE_MARKER
    if complete_path.is_symlink():
        raise RetirementError(f"retirement completion marker is a symlink: {complete_path}")
    if complete_path.exists() and not complete_path.is_file():
        raise RetirementError(
            f"retirement completion marker is unsafe: {complete_path}"
        )
    if complete_path.is_file():
        payload = _read_strict_json_object(
            complete_path, label="retirement completion marker"
        )
        _validate_completion_marker_header(
            payload,
            complete_path=complete_path,
            results_root=results_root,
            roots=roots,
            snapshot_id=snapshot_id,
            attestation_sha256=attestation_sha256,
        )
        source_verification: dict[str, Any] = {}
        for root in roots:
            proof = _verify_live_root(root, source_inventories[root.name])
            expected_intent = _root_intent(
                root=root,
                snapshot_id=snapshot_id,
                attestation_sha256=attestation_sha256,
                source_verification=proof,
            )
            _validate_or_publish_root_marker(
                root, expected_intent, publish=False, required=True
            )
            source_verification[root.name] = proof
        count = _validate_completion_marker(
            payload,
            complete_path=complete_path,
            results_root=results_root,
            recovery_root=recovery_root,
            roots=roots,
            snapshot_id=snapshot_id,
            attestation_sha256=attestation_sha256,
            source_verification=source_verification,
        )
        if _verify_read_only(roots) != count:
            raise RetirementError("retired evidence node count changed during validation")
        return payload | {
            "status": "already_retired",
            "verified_nodes": count,
        }

    precheck = _maintenance_precheck(results_root, recovery_root)
    source_verification: dict[str, Any] = {}
    node_counts: dict[str, int] = {}
    if not apply:
        for root in roots:
            proof = _verify_live_root(root, source_inventories[root.name])
            expected_intent = _root_intent(
                root=root,
                snapshot_id=snapshot_id,
                attestation_sha256=attestation_sha256,
                source_verification=proof,
            )
            _validate_or_publish_root_marker(root, expected_intent, publish=False)
            source_verification[root.name] = proof
            node_counts[root.name] = len(_walk_nodes(root))
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run" if not apply else "retired",
        "results_root": str(results_root),
        "roots": [str(root) for root in roots],
        "node_counts": node_counts,
        "snapshot_id": snapshot_id,
        "snapshot_attestation_sha256": attestation_sha256,
        "source_verification": source_verification,
        "precheck": precheck,
    }
    if not apply:
        return report

    seal_prechecks: dict[str, Any] = {}
    with _hold_all_cell_locks(results_root) as owned_locks:
        for root in roots:
            # Hash each root immediately before its intent publication and chmod pass.
            # The held worker locks close the scheduler/precheck TOCTOU boundary.
            proof = _verify_live_root(root, source_inventories[root.name])
            intent = _root_intent(
                root=root,
                snapshot_id=snapshot_id,
                attestation_sha256=attestation_sha256,
                source_verification=proof,
            )
            _validate_or_publish_root_marker(root, intent, publish=True)
            source_verification[root.name] = proof
            # Files first, then deepest directories, and the exact root last.
            nodes = _walk_nodes(root)
            node_counts[root.name] = len(nodes)
            # Re-query Slurm after the potentially long byte verification and node
            # walk.  Our lock descriptors are explicitly recognized as owned.
            seal_prechecks[root.name] = _maintenance_precheck(
                results_root,
                recovery_root,
                owned_lock_paths=owned_locks,
            )
            for path in sorted(nodes, key=lambda item: len(item.parts), reverse=True):
                info = path.stat(follow_symlinks=False)
                os.chmod(path, stat.S_IMODE(info.st_mode) & ~0o222)

        # Permission verification alone cannot prove that a writer with an already-open
        # descriptor did not change bytes during chmod.  Re-hash every live root against
        # the sealed snapshot while all worker locks remain held, then perform one final
        # scheduler/interlock check before publishing success.
        for root in roots:
            post_hash = _verify_live_root(root, source_inventories[root.name])
            if post_hash != source_verification[root.name]:
                raise RetirementError(
                    f"legacy source changed during retirement: {root}"
                )
        report["final_precheck"] = _maintenance_precheck(
            results_root,
            recovery_root,
            owned_lock_paths=owned_locks,
        )
        report["node_counts"] = node_counts
        report["source_verification"] = source_verification
        report["seal_prechecks"] = seal_prechecks
        report["verified_nodes"] = _verify_read_only(roots)
        _atomic_json(complete_path, report)
    return report | {"completion_marker": str(complete_path)}


def retire(
    *, results_root: Path, recovery_root: Path, apply: bool = False
) -> dict[str, Any]:
    """Audit or perform a globally serialized retirement transaction."""

    resolved_recovery = recovery_root.expanduser().resolve()
    if not apply:
        return _retire_unlocked(
            results_root=results_root,
            recovery_root=resolved_recovery,
            apply=False,
        )
    with _retirement_lock(resolved_recovery):
        return _retire_unlocked(
            results_root=results_root,
            recovery_root=resolved_recovery,
            apply=True,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--recovery-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = retire(
            results_root=args.results_root,
            recovery_root=args.recovery_root,
            apply=args.apply,
        )
    except (OSError, ValueError, RetirementError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
