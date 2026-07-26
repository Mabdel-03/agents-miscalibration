#!/usr/bin/env python3
"""Capture and normalize immutable schema-5 environment seeds.

The capture boundary is intentionally separate from release materialization.  This
program is the only v1.2 component allowed to inspect the two mutable developer
prefixes.  It never invokes Conda and copies regular files with an explicit buffered
read/write loop (not hardlinks, reflinks, ``copy_file_range``, or ``sendfile``).

Publication is marker-last.  A marker-first intent binds the live paths, policy,
reconciliation incident, recovered Conda record, and full pre-copy inventories.
Interrupted copies resume by accepting only entries whose bytes already match the
bound inventory.  The final marker is published only after a second source inventory,
an exact destination comparison, inode and symlink audits, narrowly authorized
Setuptools ownership normalization, exact RECORD repairs, exhaustive RECORD
validation, and recursive read-only sealing.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 2
RELEASE_ID = "sweep-recovery-schema5-v1.2"
POLICY_FILENAME = "environment_ownership_policy.v1.json"
INTEGRITY_POLICY_FILENAME = "environment_integrity_normalization_policy.v1.json"
COMPLETE_MARKER = "ENVIRONMENT_CAPTURE_COMPLETE.json"
INTENT_MARKER = "ENVIRONMENT_CAPTURE_INTENT.json"
SETUPTOOLS_NORMALIZATION_ID = "setuptools-conda-pip-ownership-v1"
PIP_RECORD_NORMALIZATION_ID = "pip-26.1.1-installer-launcher-record-v1"
PACKAGING_RECORD_NORMALIZATION_ID = "packaging-26.2-conda-installer-record-v1"
NUMPY_RECORD_NORMALIZATION_ID = "numpy-2.3.5-generated-bytecode-record-v1"
WHEEL_RECORD_NORMALIZATION_ID = "wheel-0.47.0-entrypoint-record-v1"
TORCH_DLPACK_RECORD_NORMALIZATION_ID = (
    "torch-c-dlpack-ext-0.1.5-shared-owner-record-v1"
)
RECORD_NORMALIZATION_IDS = (
    PIP_RECORD_NORMALIZATION_ID,
    PACKAGING_RECORD_NORMALIZATION_ID,
    NUMPY_RECORD_NORMALIZATION_ID,
    WHEEL_RECORD_NORMALIZATION_ID,
    TORCH_DLPACK_RECORD_NORMALIZATION_ID,
)
_OWNERSHIP_CONFLICT_AUTHORIZATION_FIELDS = frozenset(
    {
        "pip_version",
        "conda_version",
        "conda_build",
        "conda_record_filename",
        "conda_artifact_sha256",
    }
)
PIP_RECORD_SOURCE_SHA256 = (
    "e68380131f84450f99331db9b0e2b0158570d029838f5a485461d48c5e14bfc1"
)
PIP_RECORD_NORMALIZED_SHA256 = (
    "aaf8210b28c26dadec47702f3e44b2e1400ced83e64897b0a5aca1569ca279c9"
)
PIP_LAUNCHER_RECORD_ROWS = (
    (
        "../../../bin/pip",
        "sha256=jcrTAfiJHhhSWpU5AOpV6lUqNtGjmITeEbp0ENYkUM4",
        "452",
    ),
    (
        "../../../bin/pip3",
        "sha256=jcrTAfiJHhhSWpU5AOpV6lUqNtGjmITeEbp0ENYkUM4",
        "452",
    ),
    (
        "../../../bin/pip3.12",
        "sha256=jcrTAfiJHhhSWpU5AOpV6lUqNtGjmITeEbp0ENYkUM4",
        "452",
    ),
)
PACKAGING_RECORD_SOURCE_SHA256 = (
    "8548d2f3e6235e855b663e76d84d204b39e659080f49c80f64ec443e70363c88"
)
PACKAGING_RECORD_NORMALIZED_SHA256 = (
    "6096a53c376d8018db14fc7bf4590af42ae7b623fb41b34e39e1e13feb0ce189"
)
NUMPY_RECORD_SOURCE_SHA256 = (
    "1eec6aacf144409153747b04654267736487be2e9a1b6308c35e70b0bea723f5"
)
NUMPY_RECORD_NORMALIZED_SHA256 = (
    "c158387d76fd0e171de51b32e6f93fbc6ec40befded5a31a39d6a277f5100187"
)
WHEEL_RECORD_SOURCE_SHA256 = (
    "2f377d007e8575dcbf5aeab3bfa316c17d581fc056f3bbab7c6eea406c1f605b"
)
WHEEL_RECORD_NORMALIZED_SHA256 = (
    "8b1b67026b906c13c35dc60bfad877586cb7c7d73731f1c251448ba68a21f90e"
)
TORCH_DLPACK_RECORD_SOURCE_SHA256 = (
    "b12cd31544b3e333e4d8196afd31383f3aef633ab8669cbf62704420de716655"
)
TORCH_DLPACK_RECORD_NORMALIZED_SHA256 = (
    "5f6e642b3bf84eaee6c86cb9de3aa0a6b570347abfe3f1fb0c6248afe607f36f"
)
ROLE_MARKERS = {
    "harness": "HARNESS_SEED_COMPLETE.json",
    "serving": "SERVING_SEED_COMPLETE.json",
}
ROLES = tuple(ROLE_MARKERS)
SOURCE_INVENTORY_FILES = {
    role: f"SOURCE_{role.upper()}_INVENTORY.json" for role in ROLES
}
SOURCE_DISTRIBUTION_AUDIT_FILES = {
    role: f"SOURCE_{role.upper()}_DISTRIBUTION_AUDIT.json"
    for role in ROLES
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROJECT_NORMALIZE_RE = re.compile(r"[-_.]+")
_CHUNK_SIZE = 8 * 1024 * 1024


class EnvironmentCaptureError(RuntimeError):
    """The environment seeds cannot be captured or verified exactly."""


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: Iterable[str],
    *,
    description: str,
) -> None:
    expected_fields = set(expected)
    observed_fields = set(payload)
    if observed_fields != expected_fields:
        raise EnvironmentCaptureError(
            f"{description} field inventory drifted; "
            f"missing={sorted(expected_fields - observed_fields)!r}, "
            f"unexpected={sorted(observed_fields - expected_fields)!r}"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


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


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EnvironmentCaptureError(f"missing regular {description}: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EnvironmentCaptureError(
            f"cannot read {description} {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EnvironmentCaptureError(
            f"{description} must contain one JSON object: {path}"
        )
    return payload


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _atomic_write_once(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise EnvironmentCaptureError(
                f"conflicting immutable capture artifact: {path}"
            )
        os.chmod(path, mode)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".capturing", dir=path.parent
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
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_replace_bound(
    path: Path,
    payload: bytes,
    *,
    expected_before_sha256: str,
    description: str,
) -> str:
    """Atomically replace one regular file from one exact preimage.

    Returning ``already_replaced`` makes crash recovery idempotent.  No caller can
    use this helper as a general overwrite primitive: the current bytes must equal
    either the bound preimage or the requested postimage.
    """

    if path.is_symlink() or not path.is_file():
        raise EnvironmentCaptureError(f"missing or unsafe {description}: {path}")
    observed = _sha256_file(path)
    after_sha256 = _sha256_bytes(payload)
    if observed == after_sha256:
        return "already_replaced"
    if observed != expected_before_sha256:
        raise EnvironmentCaptureError(
            f"{description} does not match its bound preimage: {path}"
        )
    mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".normalizing", dir=path.parent
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
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if _sha256_file(path) != after_sha256:
        raise EnvironmentCaptureError(f"{description} postimage verification failed")
    return "replaced"


def _resolve_prefix(value: str | Path, *, description: str) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink() or not lexical.is_dir():
        raise EnvironmentCaptureError(f"missing or symlinked {description}: {lexical}")
    resolved = lexical.resolve()
    if resolved in {Path(resolved.anchor), Path.home().resolve()}:
        raise EnvironmentCaptureError(f"refusing unsafe broad {description}: {resolved}")
    if not (resolved / "conda-meta").is_dir():
        raise EnvironmentCaptureError(f"{description} is not a Conda prefix: {resolved}")
    return resolved


def _resolve_destination(value: str | Path, *, description: str) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink():
        raise EnvironmentCaptureError(f"symlinked {description} is forbidden: {lexical}")
    resolved = lexical.resolve()
    if resolved in {Path(resolved.anchor), Path.home().resolve()}:
        raise EnvironmentCaptureError(f"refusing unsafe broad {description}: {resolved}")
    return resolved


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_nonoverlap(paths: Mapping[str, Path]) -> None:
    rows = list(paths.items())
    for index, (left_name, left) in enumerate(rows):
        for right_name, right in rows[index + 1 :]:
            if (
                left == right
                or _is_relative_to(left, right)
                or _is_relative_to(right, left)
            ):
                raise EnvironmentCaptureError(
                    f"capture paths overlap: {left_name}={left}, "
                    f"{right_name}={right}"
                )


def _walk_entries(root: Path) -> Iterable[tuple[Path, os.stat_result]]:
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        base = Path(directory)
        for name in tuple(directory_names):
            path = base / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                # os.walk places directory symlinks in directory_names even with
                # followlinks=False.  Yield them once and prevent descent.
                directory_names.remove(name)
                yield path, info
            elif stat.S_ISDIR(info.st_mode):
                yield path, info
            else:
                raise EnvironmentCaptureError(
                    f"unsupported entry in directory traversal: {path}"
                )
        for name in file_names:
            path = base / name
            yield path, path.lstat()


def directory_inventory(root: Path, *, include_mode: bool = True) -> dict[str, Any]:
    """Return a complete sorted path/type/content inventory without following links."""

    if root.is_symlink() or not root.is_dir():
        raise EnvironmentCaptureError(f"inventory root is absent or symlinked: {root}")
    entries: list[dict[str, Any]] = []
    for path, info in _walk_entries(root):
        relative = path.relative_to(root).as_posix()
        common: dict[str, Any] = {"path": relative}
        if include_mode:
            common["mode"] = stat.S_IMODE(info.st_mode)
        if stat.S_ISREG(info.st_mode):
            entries.append(
                {
                    **common,
                    "type": "file",
                    "size": info.st_size,
                    "sha256": _sha256_file(path),
                }
            )
        elif stat.S_ISDIR(info.st_mode):
            entries.append({**common, "type": "directory"})
        elif stat.S_ISLNK(info.st_mode):
            entries.append({**common, "type": "symlink", "target": os.readlink(path)})
        else:
            raise EnvironmentCaptureError(
                f"unsupported special entry in environment prefix: {path}"
            )
    entries.sort(key=lambda row: row["path"])
    return {
        "inventory_sha256": _sha256_bytes(_canonical_bytes(entries)),
        "entry_count": len(entries),
        "file_count": sum(row["type"] == "file" for row in entries),
        "directory_count": sum(row["type"] == "directory" for row in entries),
        "symlink_count": sum(row["type"] == "symlink" for row in entries),
        "total_file_bytes": sum(
            int(row.get("size", 0)) for row in entries if row["type"] == "file"
        ),
        "entries": entries,
    }


def _validated_content_inventory(
    inventory: Mapping[str, Any], *, description: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    """Validate a full inventory and return its mode-independent content view."""

    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise EnvironmentCaptureError(f"{description} lacks full inventory entries")
    original_rows: list[dict[str, Any]] = []
    content_rows: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    for raw_row in entries:
        if not isinstance(raw_row, dict):
            raise EnvironmentCaptureError(
                f"{description} contains a non-object inventory entry"
            )
        row = dict(raw_row)
        path = row.get("path")
        entry_type = row.get("type")
        if not isinstance(path, str) or not path:
            raise EnvironmentCaptureError(
                f"{description} contains an invalid inventory path"
            )
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or path != relative.as_posix()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or path in by_path
        ):
            raise EnvironmentCaptureError(
                f"{description} contains an unsafe or duplicate path: {path!r}"
            )
        mode = row.get("mode")
        if "mode" in row and (
            isinstance(mode, bool)
            or not isinstance(mode, int)
            or mode < 0
            or mode > 0o7777
        ):
            raise EnvironmentCaptureError(
                f"{description} contains an invalid mode for {path!r}"
            )
        common_fields = {"path", "type"} | ({"mode"} if "mode" in row else set())
        if entry_type == "file":
            size = row.get("size")
            if (
                set(row) != common_fields | {"size", "sha256"}
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or _SHA256_RE.fullmatch(str(row.get("sha256", ""))) is None
            ):
                raise EnvironmentCaptureError(
                    f"{description} contains an invalid file entry for {path!r}"
                )
        elif entry_type == "directory":
            if set(row) != common_fields:
                raise EnvironmentCaptureError(
                    f"{description} contains an invalid directory entry for {path!r}"
                )
        elif entry_type == "symlink":
            if set(row) != common_fields | {"target"} or not isinstance(
                row.get("target"), str
            ):
                raise EnvironmentCaptureError(
                    f"{description} contains an invalid symlink entry for {path!r}"
                )
        else:
            raise EnvironmentCaptureError(
                f"{description} contains an unsupported entry type for {path!r}"
            )
        content_row = {key: value for key, value in row.items() if key != "mode"}
        original_rows.append(row)
        content_rows.append(content_row)
        by_path[path] = content_row
    paths = [row["path"] for row in original_rows]
    if paths != sorted(paths):
        raise EnvironmentCaptureError(f"{description} inventory is not sorted")
    expected_inventory_sha256 = _sha256_bytes(_canonical_bytes(original_rows))
    expected_summary = {
        "inventory_sha256": expected_inventory_sha256,
        "entry_count": len(original_rows),
        "file_count": sum(row["type"] == "file" for row in original_rows),
        "directory_count": sum(
            row["type"] == "directory" for row in original_rows
        ),
        "symlink_count": sum(row["type"] == "symlink" for row in original_rows),
        "total_file_bytes": sum(
            int(row.get("size", 0))
            for row in original_rows
            if row["type"] == "file"
        ),
    }
    if any(inventory.get(key) != value for key, value in expected_summary.items()):
        raise EnvironmentCaptureError(f"{description} inventory summary drifted")
    content_sha256 = _sha256_bytes(_canonical_bytes(content_rows))
    return content_rows, by_path, content_sha256


def _normalization_inventory_delta(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    role: str,
    policy: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove normalization changed only its exact checksummed metadata paths."""

    if role not in ROLES or receipt.get("role") != role:
        raise EnvironmentCaptureError("normalization inventory role drifted")
    before_rows, before_by_path, before_content_sha256 = (
        _validated_content_inventory(
            before, description=f"{role} pre-normalization"
        )
    )
    after_rows, after_by_path, after_content_sha256 = (
        _validated_content_inventory(
            after, description=f"{role} post-normalization"
        )
    )
    normalization = _policy_normalization(
        policy, SETUPTOOLS_NORMALIZATION_ID
    )
    setuptools_path = (
        f"conda-meta/{normalization['conda_record_filename']}"
    )
    if receipt.get("record_filename") != normalization["conda_record_filename"]:
        raise EnvironmentCaptureError(
            f"{role} Setuptools normalization path drifted"
        )
    source_record_state = receipt.get("source_record_state")
    expected_changed_paths: set[str] = set()
    if source_record_state == "present":
        before_entry = before_by_path.get(setuptools_path)
        if (
            before_entry is None
            or before_entry.get("type") != "file"
            or before_entry.get("sha256") != receipt.get("record_sha256")
            or setuptools_path in after_by_path
        ):
            raise EnvironmentCaptureError(
                f"{role} Setuptools inventory removal is not exact"
            )
        expected_changed_paths.add(setuptools_path)
    elif source_record_state == "absent":
        if setuptools_path in before_by_path or setuptools_path in after_by_path:
            raise EnvironmentCaptureError(
                f"{role} absent Setuptools record appeared in an inventory"
            )
    else:
        raise EnvironmentCaptureError(
            f"{role} normalization receipt has an invalid source state"
        )

    selected = _record_normalizations_for_role(policy, role=role)
    receipt_rows = receipt.get("record_normalizations")
    if not isinstance(receipt_rows, list):
        raise EnvironmentCaptureError(
            f"{role} normalization receipt lacks RECORD canonicalizations"
        )
    receipts_by_id = {
        row.get("normalization_id"): row
        for row in receipt_rows
        if isinstance(row, dict)
    }
    if set(receipts_by_id) != {
        row["normalization_id"] for row in selected
    }:
        raise EnvironmentCaptureError(
            f"{role} normalization inventory RECORD set drifted"
        )
    record_canonicalization_paths: list[str] = []
    for record_normalization in selected:
        normalization_id = record_normalization["normalization_id"]
        record_receipt = receipts_by_id[normalization_id]
        record_path = record_normalization["record_relative_path"]
        if record_receipt.get("record_relative_path") != record_path:
            raise EnvironmentCaptureError(
                f"{role} {normalization_id} inventory path drifted"
            )
        before_entry = before_by_path.get(record_path)
        after_entry = after_by_path.get(record_path)
        if record_receipt.get("applies") is True:
            if (
                before_entry is None
                or after_entry is None
                or before_entry.get("type") != "file"
                or after_entry.get("type") != "file"
                or before_entry.get("sha256")
                != record_normalization["source_record_sha256"]
                or after_entry.get("sha256")
                != record_normalization["normalized_record_sha256"]
                or {
                    key: value
                    for key, value in before_entry.items()
                    if key not in {"size", "sha256"}
                }
                != {
                    key: value
                    for key, value in after_entry.items()
                    if key not in {"size", "sha256"}
                }
            ):
                raise EnvironmentCaptureError(
                    f"{role} {normalization_id} inventory rewrite is not exact"
                )
            expected_changed_paths.add(record_path)
            record_canonicalization_paths.append(record_path)
        elif record_receipt.get("applies") is False:
            if before_entry is not None or after_entry is not None:
                raise EnvironmentCaptureError(
                    f"{role} inapplicable {normalization_id} has a RECORD path"
                )
        else:
            raise EnvironmentCaptureError(
                f"{role} {normalization_id} has an invalid applicability state"
            )

    actual_changed_paths = {
        path
        for path in set(before_by_path) | set(after_by_path)
        if before_by_path.get(path) != after_by_path.get(path)
    }
    if actual_changed_paths != expected_changed_paths:
        unexpected = sorted(actual_changed_paths - expected_changed_paths)
        missing = sorted(expected_changed_paths - actual_changed_paths)
        raise EnvironmentCaptureError(
            f"{role} normalization full-inventory delta is unauthorized; "
            f"unexpected={unexpected!r}, missing={missing!r}"
        )
    unchanged_path_count = sum(
        before_by_path[path] == after_by_path.get(path)
        for path in before_by_path
        if path not in expected_changed_paths
    )
    return {
        "protocol": "pathwise-pre-post-full-inventory-diff-v1",
        "pre_normalization_content_inventory_sha256": before_content_sha256,
        "post_normalization_content_inventory_sha256": after_content_sha256,
        "pre_normalization_entry_count": len(before_rows),
        "post_normalization_entry_count": len(after_rows),
        "setuptools_conda_record_path": setuptools_path,
        "setuptools_conda_record_removed": setuptools_path
        in actual_changed_paths,
        "record_canonicalization_paths": sorted(
            record_canonicalization_paths
        ),
        "expected_changed_paths": sorted(expected_changed_paths),
        "actual_changed_paths": sorted(actual_changed_paths),
        "runtime_changed_paths": [],
        "unchanged_path_count": unchanged_path_count,
        "exact_authorized_delta": True,
    }


def _copy_regular_file(source: Path, destination: Path, *, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        with source.open("rb", buffering=0) as reader, os.fdopen(
            descriptor, "wb", buffering=0
        ) as writer:
            while chunk := reader.read(_CHUNK_SIZE):
                writer.write(chunk)
            os.fsync(writer.fileno())
    except BaseException:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    os.chmod(destination, mode)


def _entry_matches(path: Path, row: Mapping[str, Any]) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    entry_type = row["type"]
    if entry_type == "directory":
        return stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
    if entry_type == "symlink":
        return stat.S_ISLNK(info.st_mode) and os.readlink(path) == row["target"]
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_size == row["size"]
        and _sha256_file(path) == row["sha256"]
    )


def copy_inventory_bound(
    source: Path, destination: Path, inventory: Mapping[str, Any]
) -> None:
    """Resume an exact real-copy capture without overwriting any existing entry."""

    if destination.is_symlink():
        raise EnvironmentCaptureError(f"capture destination is symlinked: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    rows = inventory.get("entries")
    if not isinstance(rows, list):
        raise EnvironmentCaptureError("bound source inventory has no entries")
    for row in rows:
        if not isinstance(row, dict):
            raise EnvironmentCaptureError("bound source inventory has a malformed entry")
        relative = PurePosixPath(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise EnvironmentCaptureError(f"unsafe inventory path: {relative}")
        src = source.joinpath(*relative.parts)
        dst = destination.joinpath(*relative.parts)
        if dst.exists() or dst.is_symlink():
            if not _entry_matches(dst, row):
                raise EnvironmentCaptureError(
                    f"partial capture conflicts with bound source bytes: {dst}"
                )
            continue
        entry_type = row.get("type")
        mode = int(row.get("mode", 0))
        if entry_type == "directory":
            dst.mkdir(mode=mode)
            os.chmod(dst, mode)
        elif entry_type == "symlink":
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(str(row["target"]))
        elif entry_type == "file":
            _copy_regular_file(src, dst, mode=mode)
        else:
            raise EnvironmentCaptureError(
                f"unsupported bound inventory entry type: {entry_type!r}"
            )


def verify_no_shared_regular_inodes(
    source: Path, destination: Path
) -> dict[str, int]:
    source_inodes: set[tuple[int, int]] = set()
    destination_inodes: set[tuple[int, int]] = set()
    source_count = 0
    destination_count = 0
    for path, info in _walk_entries(source):
        if stat.S_ISREG(info.st_mode):
            source_inodes.add((info.st_dev, info.st_ino))
            source_count += 1
    for path, info in _walk_entries(destination):
        if stat.S_ISREG(info.st_mode):
            destination_inodes.add((info.st_dev, info.st_ino))
            destination_count += 1
    shared = source_inodes & destination_inodes
    if shared:
        raise EnvironmentCaptureError(
            f"captured seed shares {len(shared)} regular-file inode(s) with {source}"
        )
    return {
        "source_regular_file_count": source_count,
        "destination_regular_file_count": destination_count,
        "shared_regular_inode_count": 0,
    }


def verify_internal_symlinks(root: Path) -> dict[str, int]:
    resolved_root = root.resolve(strict=True)
    count = 0
    for path, info in _walk_entries(root):
        if not stat.S_ISLNK(info.st_mode):
            continue
        count += 1
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise EnvironmentCaptureError(
                f"unresolvable symlink in captured seed: {path}: {exc}"
            ) from exc
        if not _is_relative_to(resolved, resolved_root):
            raise EnvironmentCaptureError(
                f"captured seed has an external symlink dependency: {path} -> {resolved}"
            )
    return {"symlink_count": count, "external_symlink_count": 0}


def _normalize_project(value: str) -> str:
    return _PROJECT_NORMALIZE_RE.sub("-", value).lower()


def _metadata_fields(path: Path) -> tuple[str, str]:
    name: str | None = None
    version: str | None = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Name:") and name is None:
                name = line.partition(":")[2].strip()
            elif line.startswith("Version:") and version is None:
                version = line.partition(":")[2].strip()
            if name is not None and version is not None:
                break
    except (OSError, UnicodeError) as exc:
        raise EnvironmentCaptureError(f"cannot read distribution metadata {path}: {exc}") from exc
    if not name or not version:
        raise EnvironmentCaptureError(f"distribution metadata lacks name/version: {path}")
    return _normalize_project(name), version


def _site_packages_roots(prefix: Path) -> list[Path]:
    roots_by_identity: dict[Path, Path] = {}
    for path in sorted((prefix / "lib").glob("python*/site-packages")):
        if not path.is_dir() or path.is_symlink():
            continue
        resolved = path.resolve()
        if not _is_relative_to(resolved, prefix):
            raise EnvironmentCaptureError(
                f"site-packages alias escapes environment prefix: {path}"
            )
        roots_by_identity.setdefault(resolved, path)
    roots = [roots_by_identity[key] for key in sorted(roots_by_identity, key=str)]
    if not roots:
        raise EnvironmentCaptureError(f"environment has no site-packages: {prefix}")
    return roots


def _read_record_rows(record_path: Path) -> list[tuple[str, str, str]]:
    if record_path.is_symlink() or not record_path.is_file():
        raise EnvironmentCaptureError(
            f"pip RECORD is missing or not a regular file: {record_path}"
        )
    rows: list[tuple[str, str, str]] = []
    try:
        with record_path.open(newline="", encoding="utf-8") as handle:
            for raw_row in csv.reader(handle, strict=True):
                if len(raw_row) != 3:
                    raise EnvironmentCaptureError(
                        f"malformed pip RECORD row in {record_path}"
                    )
                rows.append((raw_row[0], raw_row[1], raw_row[2]))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EnvironmentCaptureError(
            f"cannot parse pip RECORD {record_path}: {exc}"
        ) from exc
    if not rows:
        raise EnvironmentCaptureError(f"pip RECORD is empty: {record_path}")
    return rows


def _record_candidate(
    *,
    prefix: Path,
    site_root: Path,
    record_path: Path,
    relative: str,
    allow_missing_unhashed_runtime_cache: bool = False,
    allow_missing_authorized_record: bool = False,
) -> Path:
    if (
        not relative
        or "\x00" in relative
        or "\\" in relative
        or "\r" in relative
        or "\n" in relative
    ):
        raise EnvironmentCaptureError(
            f"unsafe pip RECORD path {relative!r} in {record_path}"
        )
    posix_path = PurePosixPath(relative)
    if posix_path.is_absolute() or not posix_path.parts:
        raise EnvironmentCaptureError(
            f"unsafe pip RECORD path {relative!r} in {record_path}"
        )
    # RECORD paths are slash-separated POSIX paths.  Requiring their canonical
    # spelling prevents two lexical aliases from claiming the same installed file.
    saw_non_parent = False
    has_embedded_parent = False
    for part in posix_path.parts:
        if part == "..":
            has_embedded_parent |= saw_non_parent
        else:
            saw_non_parent = True
    if (
        posix_path.as_posix() != relative
        or has_embedded_parent
        or "/./" in f"/{relative}/"
        or "//" in relative
    ):
        raise EnvironmentCaptureError(
            f"non-canonical pip RECORD path {relative!r} in {record_path}"
        )
    try:
        candidate = site_root.joinpath(*posix_path.parts).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise EnvironmentCaptureError(
            f"unresolvable pip RECORD path {relative!r} in {record_path}"
        ) from exc
    if not _is_relative_to(candidate, prefix.resolve()):
        raise EnvironmentCaptureError(
            f"pip RECORD escapes environment prefix: {relative!r}"
        )
    is_generated_runtime_cache = candidate.suffix in {".pyc", ".pyo"}
    if not candidate.exists() and not (
        (allow_missing_unhashed_runtime_cache and is_generated_runtime_cache)
        or allow_missing_authorized_record
    ):
        raise EnvironmentCaptureError(
            f"pip RECORD references a missing path: {candidate}"
        )
    if candidate.exists() and not candidate.is_file():
        raise EnvironmentCaptureError(
            f"pip RECORD references a non-regular path: {candidate}"
        )
    return candidate


def _normalization_rows(
    normalization: Mapping[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    if "installer_generated_launcher_rows" in normalization:
        raw_rows = normalization.get("installer_generated_launcher_rows")
    else:
        mutations = normalization.get("row_mutations")
        raw_rows = (
            [mutation.get("before") for mutation in mutations]
            if isinstance(mutations, list)
            else None
        )
    if not isinstance(raw_rows, list):
        raise EnvironmentCaptureError("RECORD normalization has no row list")
    rows: list[tuple[str, str, str]] = []
    for row in raw_rows:
        if not isinstance(row, dict) or set(row) != {"path", "hash", "size"}:
            raise EnvironmentCaptureError(
                "RECORD normalization has a malformed row"
            )
        values = (row["path"], row["hash"], row["size"])
        if not all(isinstance(value, str) for value in values):
            raise EnvironmentCaptureError(
                "RECORD normalization row fields must be strings"
            )
        rows.append(values)
    if len(rows) != len(set(rows)):
        raise EnvironmentCaptureError(
            "RECORD normalization contains duplicate rows"
        )
    return tuple(rows)


def _record_mutations(
    normalization: Mapping[str, Any],
) -> dict[tuple[str, str, str], tuple[str, str, str] | None]:
    if "installer_generated_launcher_rows" in normalization:
        return {row: None for row in _normalization_rows(normalization)}
    raw_mutations = normalization.get("row_mutations")
    if not isinstance(raw_mutations, list):
        raise EnvironmentCaptureError("RECORD normalization has no mutations")
    mutations: dict[tuple[str, str, str], tuple[str, str, str] | None] = {}
    for mutation in raw_mutations:
        if not isinstance(mutation, dict) or set(mutation) != {"before", "after"}:
            raise EnvironmentCaptureError("RECORD mutation is malformed")
        before = mutation["before"]
        after = mutation["after"]
        if not isinstance(before, dict) or set(before) != {"path", "hash", "size"}:
            raise EnvironmentCaptureError("RECORD mutation preimage row is malformed")
        before_row = (before["path"], before["hash"], before["size"])
        if not all(isinstance(value, str) for value in before_row):
            raise EnvironmentCaptureError("RECORD mutation fields must be strings")
        after_row: tuple[str, str, str] | None
        if after is None:
            after_row = None
        elif isinstance(after, dict) and set(after) == {"path", "hash", "size"}:
            after_row = (after["path"], after["hash"], after["size"])
            if not all(isinstance(value, str) for value in after_row):
                raise EnvironmentCaptureError("RECORD mutation fields must be strings")
        else:
            raise EnvironmentCaptureError("RECORD mutation postimage row is malformed")
        if before_row in mutations:
            raise EnvironmentCaptureError("RECORD mutation rows are duplicated")
        mutations[before_row] = after_row
    return mutations


def _required_retained_rows(
    normalization: Mapping[str, Any],
) -> tuple[tuple[str, str, str], ...]:
    raw_rows = normalization.get("required_retained_rows", [])
    if not isinstance(raw_rows, list):
        raise EnvironmentCaptureError("required retained RECORD rows are malformed")
    rows: list[tuple[str, str, str]] = []
    for row in raw_rows:
        if not isinstance(row, dict) or set(row) != {"path", "hash", "size"}:
            raise EnvironmentCaptureError("required retained RECORD row is malformed")
        values = (row["path"], row["hash"], row["size"])
        if not all(isinstance(value, str) for value in values):
            raise EnvironmentCaptureError("required retained RECORD row is malformed")
        rows.append(values)
    if len(rows) != len(set(rows)):
        raise EnvironmentCaptureError("required retained RECORD rows are duplicated")
    return tuple(rows)


def _normalized_record_bytes(
    preimage: bytes, normalization: Mapping[str, Any]
) -> bytes:
    mutations = _record_mutations(normalization)
    expected_rows = set(mutations)
    observed_counts = {row: 0 for row in expected_rows}
    retained_required = {
        row: 0 for row in _required_retained_rows(normalization)
    }
    retained: list[bytes] = []
    for raw_line in preimage.splitlines(keepends=True):
        content = raw_line.rstrip(b"\r\n")
        try:
            decoded = content.decode("utf-8")
            parsed = list(csv.reader([decoded], strict=True))
        except (UnicodeError, csv.Error) as exc:
            raise EnvironmentCaptureError(
                "cannot parse bound pip RECORD preimage"
            ) from exc
        if len(parsed) != 1 or len(parsed[0]) != 3:
            raise EnvironmentCaptureError(
                "bound pip RECORD preimage contains a malformed row"
            )
        row = (parsed[0][0], parsed[0][1], parsed[0][2])
        if row in expected_rows:
            observed_counts[row] += 1
            after = mutations[row]
            if after is not None:
                ending = raw_line[len(content) :]
                retained.append(",".join(after).encode("utf-8") + ending)
        else:
            retained.append(raw_line)
            if row in retained_required:
                retained_required[row] += 1
    if any(count != 1 for count in observed_counts.values()):
        raise EnvironmentCaptureError(
            "bound RECORD does not contain each authorized mutation row exactly once"
        )
    if any(count != 1 for count in retained_required.values()):
        raise EnvironmentCaptureError(
            "bound RECORD does not retain each required row exactly once"
        )
    postimage = b"".join(retained)
    if _sha256_bytes(postimage) != normalization["normalized_record_sha256"]:
        raise EnvironmentCaptureError(
            "RECORD normalized postimage does not match policy"
        )
    return postimage


def _normalized_pip_record_bytes(
    preimage: bytes, normalization: Mapping[str, Any]
) -> bytes:
    """Backward-compatible alias for the generalized RECORD normalizer."""

    return _normalized_record_bytes(preimage, normalization)


def _source_record_authorization(
    prefix: Path,
    normalization: Mapping[str, Any] | None,
    *,
    role: str | None = None,
) -> dict[str, Any] | None:
    if normalization is None:
        return None
    relative = PurePosixPath(str(normalization["record_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise EnvironmentCaptureError(
            "RECORD normalization has an unsafe record path"
        )
    record_path = prefix.joinpath(*relative.parts)
    if not record_path.parent.exists():
        # Minimal unit-test prefixes may omit pip entirely.  If the distribution
        # exists, however, an absent/unsafe RECORD remains a strict failure below.
        return None
    if record_path.is_symlink() or not record_path.is_file():
        raise EnvironmentCaptureError(
            f"RECORD normalization target is missing or unsafe: {record_path}"
        )
    metadata_path = record_path.parent / "METADATA"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise EnvironmentCaptureError(
            "RECORD normalization target has no safe METADATA"
        )
    project, version = _metadata_fields(metadata_path)
    if (
        project != _normalize_project(str(normalization["project"]))
        or version != normalization["version"]
    ):
        raise EnvironmentCaptureError(
            "RECORD normalization target has the wrong distribution identity"
        )
    if role is not None:
        role_policy = normalization.get("roles", {}).get(role, {}).get(
            "source_record_policy"
        )
        if role_policy == "forbidden":
            raise EnvironmentCaptureError(
                f"{normalization['normalization_id']} is forbidden for {role}"
            )
        if role_policy != "exact_preimage_if_present":
            raise EnvironmentCaptureError(
                f"{normalization['normalization_id']} has no valid policy for {role}"
            )
    if _sha256_file(record_path) != normalization["source_record_sha256"]:
        raise EnvironmentCaptureError(
            "RECORD source preimage does not match the checksummed policy"
        )
    preimage = record_path.read_bytes()
    postimage = _normalized_record_bytes(preimage, normalization)
    source_file_evidence = _validate_record_source_contracts(
        prefix, normalization=normalization, role=role
    )
    return {
        "normalization_id": normalization["normalization_id"],
        "record_path": record_path.resolve(strict=True),
        "source_record_sha256": normalization["source_record_sha256"],
        "normalized_record_sha256": normalization["normalized_record_sha256"],
        "rows": frozenset(_normalization_rows(normalization)),
        "mutations": _record_mutations(normalization),
        "postimage": postimage,
        "source_file_evidence": source_file_evidence,
    }


def _safe_contract_path(prefix: Path, relative: str, *, description: str) -> Path:
    posix = PurePosixPath(relative)
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or not posix.parts
        or posix.as_posix() != relative
    ):
        raise EnvironmentCaptureError(f"unsafe {description} path: {relative!r}")
    candidate = prefix.joinpath(*posix.parts)
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise EnvironmentCaptureError(
            f"unresolvable {description} path: {relative!r}"
        ) from exc
    if not _is_relative_to(resolved, prefix.resolve()):
        raise EnvironmentCaptureError(f"{description} path escapes prefix")
    return resolved


def _validate_file_contract(
    prefix: Path, contract: Mapping[str, Any], *, description: str
) -> dict[str, Any]:
    if not isinstance(contract, dict) or not isinstance(contract.get("path"), str):
        raise EnvironmentCaptureError(f"{description} file contract is malformed")
    path = _safe_contract_path(
        prefix, contract["path"], description=f"{description} file"
    )
    expected_state = contract.get("state", "regular_file")
    if expected_state == "missing":
        if path.exists() or path.is_symlink():
            raise EnvironmentCaptureError(
                f"{description} expected a missing runtime path: {path}"
            )
        return {"path": contract["path"], "state": "missing"}
    if (
        expected_state != "regular_file"
        or path.is_symlink()
        or not path.is_file()
        or contract.get("sha256") != _sha256_file(path)
        or contract.get("size") != path.stat().st_size
    ):
        raise EnvironmentCaptureError(
            f"{description} runtime file contract drifted: {path}"
        )
    return {
        "path": contract["path"],
        "state": "regular_file",
        "sha256": contract["sha256"],
        "size": contract["size"],
    }


def _validate_record_source_contracts(
    prefix: Path,
    *,
    normalization: Mapping[str, Any],
    role: str | None,
) -> list[dict[str, Any]]:
    contracts = normalization.get("source_file_contracts", [])
    if not isinstance(contracts, list):
        raise EnvironmentCaptureError("RECORD source file contracts are malformed")
    evidence = [
        _validate_file_contract(
            prefix,
            contract,
            description=str(normalization["normalization_id"]),
        )
        for contract in contracts
    ]
    role_contracts = normalization.get("role_source_file_contracts", {})
    if role_contracts:
        if role is None or not isinstance(role_contracts.get(role), list):
            raise EnvironmentCaptureError(
                "role-specific RECORD source contracts require an authorized role"
            )
        evidence.extend(
            _validate_file_contract(
                prefix,
                contract,
                description=f"{normalization['normalization_id']}:{role}",
            )
            for contract in role_contracts[role]
        )
    conda_contract = normalization.get("conda_ownership_contract")
    if conda_contract is not None:
        if not isinstance(conda_contract, dict):
            raise EnvironmentCaptureError("Conda ownership contract is malformed")
        conda_path = _safe_contract_path(
            prefix,
            str(conda_contract["record_relative_path"]),
            description="Conda ownership record",
        )
        payload = _read_json(conda_path, description="Conda ownership record")
        if (
            _sha256_file(conda_path) != conda_contract["record_sha256"]
            or payload.get("name") != conda_contract["name"]
            or payload.get("version") != conda_contract["version"]
            or payload.get("build") != conda_contract["build"]
            or payload.get("sha256") != conda_contract["artifact_sha256"]
            or conda_contract["owned_path"] not in payload.get("files", [])
        ):
            raise EnvironmentCaptureError("Conda ownership record drifted")
        owned_matches = [
            row
            for row in payload.get("paths_data", {}).get("paths", [])
            if row.get("_path") == conda_contract["owned_path"]
        ]
        if (
            len(owned_matches) != 1
            or owned_matches[0].get("sha256_in_prefix")
            != conda_contract["owned_path_sha256"]
            or owned_matches[0].get("size_in_bytes")
            != conda_contract["owned_path_size"]
        ):
            raise EnvironmentCaptureError("Conda owned-path evidence drifted")
        evidence.append(
            {
                "conda_record_path": conda_contract["record_relative_path"],
                "conda_record_sha256": conda_contract["record_sha256"],
                "artifact_sha256": conda_contract["artifact_sha256"],
                "owned_path": conda_contract["owned_path"],
                "owned_path_sha256": conda_contract["owned_path_sha256"],
                "owned_path_size": conda_contract["owned_path_size"],
            }
        )
    if normalization.get("conda_project_must_be_absent") is True:
        project = _normalize_project(str(normalization["project"]))
        if project in _conda_records(prefix):
            raise EnvironmentCaptureError(
                f"{project} unexpectedly has a Conda ownership record"
            )
        evidence.append({"conda_project": project, "state": "absent"})
    other_owner = normalization.get("other_owner_contract")
    if other_owner is not None:
        if not isinstance(other_owner, dict):
            raise EnvironmentCaptureError("other-owner contract is malformed")
        other_path = _safe_contract_path(
            prefix,
            str(other_owner["record_relative_path"]),
            description="other-owner RECORD",
        )
        if _sha256_file(other_path) != other_owner["record_sha256"]:
            raise EnvironmentCaptureError("other-owner RECORD preimage drifted")
        other_metadata = other_path.parent / "METADATA"
        if _metadata_fields(other_metadata) != (
            _normalize_project(str(other_owner["project"])),
            other_owner["version"],
        ):
            raise EnvironmentCaptureError("other-owner identity drifted")
        retained = other_owner["retained_row"]
        retained_tuple = (
            retained["path"],
            retained["hash"],
            retained["size"],
        )
        if _read_record_rows(other_path).count(retained_tuple) != 1:
            raise EnvironmentCaptureError("other-owner RECORD row drifted")
        evidence.append(
            {
                "other_owner_project": other_owner["project"],
                "other_owner_version": other_owner["version"],
                "other_owner_record_path": other_owner["record_relative_path"],
                "other_owner_record_sha256": other_owner["record_sha256"],
                "retained_row": retained,
            }
        )
    return evidence


def _pip_launcher_runtime_states(
    prefix: Path, normalization: Mapping[str, Any]
) -> list[dict[str, Any]]:
    relative = PurePosixPath(str(normalization["record_relative_path"]))
    record_path = prefix.joinpath(*relative.parts)
    site_root = record_path.parent.parent
    states: list[dict[str, Any]] = []
    for launcher_path, encoded_hash, encoded_size in _normalization_rows(
        normalization
    ):
        candidate = _record_candidate(
            prefix=prefix,
            site_root=site_root,
            record_path=record_path,
            relative=launcher_path,
            allow_missing_authorized_record=True,
        )
        exists = candidate.exists()
        states.append(
            {
                "record_path": launcher_path,
                "installed_path": candidate.relative_to(prefix).as_posix(),
                "record_hash": encoded_hash,
                "record_size": encoded_size,
                "state": "regular_file" if exists else "missing",
                "actual_sha256": _sha256_file(candidate) if exists else None,
                "actual_size": candidate.stat().st_size if exists else None,
                "actual_mode": (
                    stat.S_IMODE(candidate.stat().st_mode) if exists else None
                ),
            }
        )
    return states


def _pip_launcher_runtime_states_from_sealed_evidence(
    *,
    output_root: Path,
    role: str,
    seed: Path,
    normalization: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recreate pre-seal launcher states using bytes plus archived source modes."""

    states = _pip_launcher_runtime_states(seed, normalization)
    inventory = _read_json(
        output_root / SOURCE_INVENTORY_FILES[role],
        description=f"{role} archived source inventory",
    )
    _rows, by_path, _content_sha256 = _validated_content_inventory(
        inventory, description=f"{role} archived source"
    )
    original_by_path = {
        row["path"]: row for row in inventory["entries"]
    }
    for state in states:
        installed_path = state["installed_path"]
        source_row = by_path.get(installed_path)
        original_row = original_by_path.get(installed_path)
        if state["state"] == "missing":
            if source_row is not None:
                raise EnvironmentCaptureError(
                    f"{role} launcher source state drifted for {installed_path}"
                )
            state["actual_mode"] = None
        elif (
            source_row is None
            or source_row.get("type") != "file"
            or not isinstance(original_row, dict)
            or "mode" not in original_row
        ):
            raise EnvironmentCaptureError(
                f"{role} launcher source evidence is missing for "
                f"{installed_path}"
            )
        else:
            state["actual_mode"] = original_row["mode"]
    return states


def _decode_record_sha256(encoded_hash: str, *, record_path: Path) -> str:
    algorithm, separator, encoded = encoded_hash.partition("=")
    if separator != "=" or algorithm != "sha256" or not encoded:
        raise EnvironmentCaptureError(
            f"pip RECORD uses an invalid or non-SHA256 hash: {record_path}"
        )
    # Wheel RECORD uses URL-safe base64 without padding.  A SHA-256 digest has
    # exactly 32 bytes and therefore exactly 43 unpadded base64 characters.
    if re.fullmatch(r"[A-Za-z0-9_-]{43}", encoded) is None:
        raise EnvironmentCaptureError(
            f"invalid pip RECORD SHA256 digest: {record_path}"
        )
    try:
        decoded = base64.b64decode(
            encoded + "=",
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise EnvironmentCaptureError(
            f"invalid pip RECORD SHA256 digest: {record_path}"
        ) from exc
    if (
        len(decoded) != hashlib.sha256().digest_size
        or base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != encoded
    ):
        raise EnvironmentCaptureError(
            f"invalid pip RECORD SHA256 digest: {record_path}"
        )
    return decoded.hex()


def _validate_record_size(
    encoded_size: str, *, candidate: Path, record_path: Path
) -> None:
    if not encoded_size:
        return
    if re.fullmatch(r"(?:0|[1-9][0-9]{0,18})", encoded_size) is None:
        raise EnvironmentCaptureError(
            f"invalid pip RECORD size in {record_path}"
        )
    if candidate.stat().st_size != int(encoded_size):
        raise EnvironmentCaptureError(f"pip RECORD size mismatch: {candidate}")


def distribution_inventory(
    prefix: Path,
    *,
    source_record_normalization: Mapping[str, Any] | None = None,
    source_record_normalizations: Sequence[Mapping[str, Any]] = (),
    role: str | None = None,
) -> dict[str, Any]:
    """Validate installed distributions and every hash-bearing pip RECORD row."""

    if source_record_normalization is not None and source_record_normalizations:
        raise EnvironmentCaptureError(
            "cannot combine singular and plural RECORD normalizations"
        )
    normalizations = (
        (source_record_normalization,)
        if source_record_normalization is not None
        else tuple(source_record_normalizations)
    )
    source_authorizations = [
        authorization
        for authorization in (
            _source_record_authorization(
                prefix, normalization, role=role
            )
            for normalization in normalizations
        )
        if authorization is not None
    ]
    authorizations_by_path: dict[Path, dict[str, Any]] = {}
    for authorization in source_authorizations:
        record_path = authorization["record_path"]
        if record_path in authorizations_by_path:
            raise EnvironmentCaptureError(
                "multiple RECORD normalizations target one distribution"
            )
        authorizations_by_path[record_path] = authorization

    def authorized_row(
        record_path: Path, row: tuple[str, str, str]
    ) -> bool:
        authorization = authorizations_by_path.get(record_path.resolve(strict=True))
        return authorization is not None and row in authorization["rows"]

    def projected_row(
        record_path: Path, row: tuple[str, str, str]
    ) -> tuple[str, str, str] | None:
        authorization = authorizations_by_path.get(record_path.resolve(strict=True))
        if authorization is None or row not in authorization["rows"]:
            return row
        return authorization["mutations"][row]

    distributions: dict[str, dict[str, Any]] = {}
    conda_versions = {
        project: str(record["version"])
        for project, record in _conda_records(prefix).items()
    }
    site_roots = _site_packages_roots(prefix)
    record_owners: dict[Path, set[str]] = {}
    projected_record_owners: dict[Path, set[str]] = {}
    record_rows: dict[Path, list[tuple[str, str, str]]] = {}
    for site_root in site_roots:
        for metadata_root in sorted(site_root.glob("*.dist-info")):
            if metadata_root.is_symlink() or not metadata_root.is_dir():
                raise EnvironmentCaptureError(
                    f"unsafe installed distribution metadata: {metadata_root}"
                )
            record_path = metadata_root / "RECORD"
            rows = _read_record_rows(record_path)
            record_rows[record_path] = rows
            for relative, encoded_hash, encoded_size in rows:
                is_authorized_row = authorized_row(
                    record_path, (relative, encoded_hash, encoded_size)
                )
                candidate = _record_candidate(
                    prefix=prefix,
                    site_root=site_root,
                    record_path=record_path,
                    relative=relative,
                    allow_missing_unhashed_runtime_cache=(
                        not encoded_hash and not encoded_size
                    ),
                    allow_missing_authorized_record=is_authorized_row,
                )
                record_owners.setdefault(candidate, set()).add(str(metadata_root))
                after = projected_row(
                    record_path, (relative, encoded_hash, encoded_size)
                )
                if after is not None:
                    projected_candidate = _record_candidate(
                        prefix=prefix,
                        site_root=site_root,
                        record_path=record_path,
                        relative=after[0],
                        allow_missing_unhashed_runtime_cache=(
                            not after[1] and not after[2]
                        ),
                        allow_missing_authorized_record=False,
                    )
                    projected_record_owners.setdefault(
                        projected_candidate, set()
                    ).add(str(metadata_root))
    record_count = 0
    hashed_file_count = 0
    unhashed_record_count = 0
    runtime_cache_record_count = 0
    missing_unhashed_runtime_cache_record_count = 0
    policy_normalized_record_count = 0
    for site_root in site_roots:
        metadata_roots = sorted(site_root.glob("*.dist-info")) + sorted(
            site_root.glob("*.egg-info")
        )
        for metadata_root in metadata_roots:
            if metadata_root.is_symlink():
                raise EnvironmentCaptureError(
                    f"unsafe installed distribution metadata: {metadata_root}"
                )
            metadata_path = (
                metadata_root / "METADATA"
                if metadata_root.is_dir()
                else metadata_root
            )
            if metadata_path.is_symlink() or not metadata_path.is_file():
                # Legacy metadata without canonical METADATA is not acceptable in a
                # production seed because its project/version identity is ambiguous.
                raise EnvironmentCaptureError(
                    f"installed distribution lacks METADATA: {metadata_root}"
                )
            project, version = _metadata_fields(metadata_path)
            if project in distributions:
                raise EnvironmentCaptureError(
                    f"duplicate installed distribution {project!r} in {prefix}"
                )
            row: dict[str, Any] = {
                "project": project,
                "version": version,
                "metadata_path": metadata_path.relative_to(prefix).as_posix(),
            }
            record_path = (
                metadata_root / "RECORD" if metadata_root.is_dir() else None
            )
            if record_path is not None and record_path.is_file():
                seen_paths: dict[str, tuple[str, str]] = {}
                record_digest = hashlib.sha256()
                record_self_count = 0
                missing_unhashed_runtime_cache_paths: list[str] = []
                policy_normalized_rows: list[list[str]] = []
                for relative, encoded_hash, encoded_size in record_rows[record_path]:
                    previous = seen_paths.get(relative)
                    if previous is not None:
                        previous_hash, previous_size = previous
                        if (
                            previous_hash
                            and encoded_hash
                            and (
                                previous_hash != encoded_hash
                                or previous_size != encoded_size
                            )
                        ):
                            raise EnvironmentCaptureError(
                                "conflicting duplicate pip RECORD path "
                                f"{relative!r}: {record_path}"
                            )
                        # Pip may emit one empty bytecode row and one later
                        # hash-bearing row for the same generated ``.pyc``.  This
                        # is one file identity, not a duplicate distribution.
                        if not previous_hash and encoded_hash:
                            seen_paths[relative] = (
                                encoded_hash,
                                encoded_size,
                            )
                    else:
                        seen_paths[relative] = (encoded_hash, encoded_size)
                    is_authorized_row = authorized_row(
                        record_path, (relative, encoded_hash, encoded_size)
                    )
                    candidate = _record_candidate(
                        prefix=prefix,
                        site_root=site_root,
                        record_path=record_path,
                        relative=relative,
                        allow_missing_unhashed_runtime_cache=(
                            not encoded_hash and not encoded_size
                        ),
                        allow_missing_authorized_record=is_authorized_row,
                    )
                    if candidate == record_path.resolve(strict=True):
                        record_self_count += 1
                        if encoded_hash or encoded_size:
                            raise EnvironmentCaptureError(
                                "pip RECORD self-entry must omit hash and size: "
                                f"{record_path}"
                            )
                    is_runtime_cache = candidate.suffix in {".pyc", ".pyo"}
                    candidate_exists = candidate.exists()
                    if not candidate_exists:
                        # Wheel installers are allowed to list generated bytecode
                        # without a hash or size.  Such caches may subsequently be
                        # removed.  Preserve the exact absent claim in the audit;
                        # this exception is unreachable for any declared hash/size
                        # or any non-runtime-cache path.
                        if not is_authorized_row:
                            missing_unhashed_runtime_cache_paths.append(relative)
                            missing_unhashed_runtime_cache_record_count += 1
                    if is_authorized_row:
                        # Each exact authorized mutation row is bound to the complete
                        # source RECORD hash and its runtime/ownership evidence.  It
                        # is rewritten or removed only in the copied seed.
                        policy_normalized_rows.append(
                            [relative, encoded_hash, encoded_size]
                        )
                        policy_normalized_record_count += 1
                    else:
                        _validate_record_size(
                            encoded_size,
                            candidate=candidate,
                            record_path=record_path,
                        )
                    if encoded_hash and not is_authorized_row:
                        expected = _decode_record_sha256(
                            encoded_hash,
                            record_path=record_path,
                        )
                        if _sha256_file(candidate) != expected:
                            raise EnvironmentCaptureError(
                                f"pip RECORD hash mismatch: {candidate}"
                            )
                        hashed_file_count += 1
                    elif not encoded_hash:
                        # The wheel installation metadata permits an empty hash
                        # (notably for RECORD itself and generated bytecode).
                        # Empty means unverified by RECORD, never "skip a declared
                        # hash"; the complete capture inventory still pins bytes.
                        unhashed_record_count += 1
                        if is_runtime_cache:
                            runtime_cache_record_count += 1
                    record_digest.update(
                        _canonical_bytes(
                            [relative, encoded_hash, encoded_size]
                        )
                    )
                    record_count += 1
                if record_self_count != 1:
                    raise EnvironmentCaptureError(
                        "pip RECORD must contain exactly one unhashed self-entry: "
                        f"{record_path}"
                    )
                row["record_sha256"] = _sha256_file(record_path)
                row["record_entry_digest"] = record_digest.hexdigest()
                row["record_entry_count"] = len(seen_paths)
                row["record_validation"] = (
                    "all_non_normalized_declared_pip_record_hashes"
                    if policy_normalized_rows
                    else "all_declared_pip_record_hashes"
                )
                row["policy_normalized_record_rows"] = sorted(
                    policy_normalized_rows
                )
                row["policy_normalized_record_count"] = len(
                    policy_normalized_rows
                )
                row["missing_unhashed_runtime_cache_paths"] = sorted(
                    missing_unhashed_runtime_cache_paths
                )
                row["missing_unhashed_runtime_cache_record_count"] = len(
                    missing_unhashed_runtime_cache_paths
                )
                row["conda_owned_same_version"] = (
                    conda_versions.get(project) == version
                )
            distributions[project] = row
    normalized = [distributions[name] for name in sorted(distributions)]
    return {
        "inventory_sha256": _sha256_bytes(_canonical_bytes(normalized)),
        "distribution_count": len(normalized),
        "record_count": record_count,
        "hashed_file_count": hashed_file_count,
        "unhashed_record_count": unhashed_record_count,
        "runtime_cache_record_count": runtime_cache_record_count,
        "missing_unhashed_runtime_cache_record_count": (
            missing_unhashed_runtime_cache_record_count
        ),
        "policy_normalized_record_count": policy_normalized_record_count,
        "shared_record_path_count": sum(
            len(owners) > 1 for owners in record_owners.values()
        ),
        "projected_shared_record_path_count": sum(
            len(owners) > 1
            for owners in projected_record_owners.values()
        ),
        "distributions": normalized,
    }


def _record_metrics_from_rows(
    *,
    prefix: Path,
    site_root: Path,
    record_path: Path,
    rows: Sequence[tuple[str, str, str]],
    authorized_mutations: Mapping[
        tuple[str, str, str], tuple[str, str, str] | None
    ],
) -> dict[str, Any]:
    """Re-derive one source RECORD audit from sealed runtime and preimage bytes."""

    seen_paths: dict[str, tuple[str, str]] = {}
    record_digest = hashlib.sha256()
    record_self_count = 0
    missing_runtime_cache_paths: list[str] = []
    policy_rows: list[list[str]] = []
    hashed_file_count = 0
    unhashed_record_count = 0
    runtime_cache_record_count = 0
    for relative, encoded_hash, encoded_size in rows:
        previous = seen_paths.get(relative)
        if previous is not None:
            previous_hash, previous_size = previous
            if (
                previous_hash
                and encoded_hash
                and (
                    previous_hash != encoded_hash
                    or previous_size != encoded_size
                )
            ):
                raise EnvironmentCaptureError(
                    "conflicting duplicate pip RECORD path "
                    f"{relative!r}: {record_path}"
                )
            if not previous_hash and encoded_hash:
                seen_paths[relative] = (encoded_hash, encoded_size)
        else:
            seen_paths[relative] = (encoded_hash, encoded_size)
        row = (relative, encoded_hash, encoded_size)
        authorized = row in authorized_mutations
        candidate = _record_candidate(
            prefix=prefix,
            site_root=site_root,
            record_path=record_path,
            relative=relative,
            allow_missing_unhashed_runtime_cache=(
                not encoded_hash and not encoded_size
            ),
            allow_missing_authorized_record=authorized,
        )
        if candidate == record_path.resolve(strict=True):
            record_self_count += 1
            if encoded_hash or encoded_size:
                raise EnvironmentCaptureError(
                    "pip RECORD self-entry must omit hash and size: "
                    f"{record_path}"
                )
        is_runtime_cache = candidate.suffix in {".pyc", ".pyo"}
        if not candidate.exists() and not authorized:
            missing_runtime_cache_paths.append(relative)
        if authorized:
            policy_rows.append([relative, encoded_hash, encoded_size])
        else:
            _validate_record_size(
                encoded_size,
                candidate=candidate,
                record_path=record_path,
            )
        if encoded_hash and not authorized:
            expected = _decode_record_sha256(
                encoded_hash, record_path=record_path
            )
            if _sha256_file(candidate) != expected:
                raise EnvironmentCaptureError(
                    f"pip RECORD hash mismatch: {candidate}"
                )
            hashed_file_count += 1
        elif not encoded_hash:
            unhashed_record_count += 1
            if is_runtime_cache:
                runtime_cache_record_count += 1
        record_digest.update(
            _canonical_bytes([relative, encoded_hash, encoded_size])
        )
    if record_self_count != 1:
        raise EnvironmentCaptureError(
            "pip RECORD must contain exactly one unhashed self-entry: "
            f"{record_path}"
        )
    return {
        "record_count": len(rows),
        "hashed_file_count": hashed_file_count,
        "unhashed_record_count": unhashed_record_count,
        "runtime_cache_record_count": runtime_cache_record_count,
        "missing_unhashed_runtime_cache_record_count": len(
            missing_runtime_cache_paths
        ),
        "missing_unhashed_runtime_cache_paths": sorted(
            missing_runtime_cache_paths
        ),
        "policy_normalized_record_count": len(policy_rows),
        "policy_normalized_record_rows": sorted(policy_rows),
        "record_entry_digest": record_digest.hexdigest(),
        "record_entry_count": len(seen_paths),
    }


def _reconstruct_source_distribution_inventory(
    *,
    output_root: Path,
    role: str,
    seed: Path,
    policy: Mapping[str, Any],
    receipt: Mapping[str, Any],
    normalized_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the source distribution audit without reading live prefixes.

    Runtime bytes come from the sealed normalized seed.  Every byte changed by
    normalization comes from its checksummed archived preimage.  The exact
    full-inventory delta proves there are no other source/seed byte differences.
    """

    if role not in ROLES:
        raise EnvironmentCaptureError(f"unknown environment role: {role!r}")
    normalized_rows = normalized_report.get("distributions")
    if not isinstance(normalized_rows, list):
        raise EnvironmentCaptureError(
            f"{role} normalized distribution inventory is incomplete"
        )
    source_rows_by_project = {
        str(row.get("project")): dict(row)
        for row in normalized_rows
        if isinstance(row, dict)
    }
    if len(source_rows_by_project) != len(normalized_rows):
        raise EnvironmentCaptureError(
            f"{role} normalized distribution inventory has duplicate projects"
        )
    receipt_rows = receipt.get("record_normalizations")
    if not isinstance(receipt_rows, list):
        raise EnvironmentCaptureError(
            f"{role} normalization receipt lacks RECORD rows"
        )
    receipts_by_id = {
        row.get("normalization_id"): row
        for row in receipt_rows
        if isinstance(row, dict)
    }
    substitutions: dict[
        Path,
        tuple[
            list[tuple[str, str, str]],
            dict[tuple[str, str, str], tuple[str, str, str] | None],
            Mapping[str, Any],
        ],
    ] = {}
    totals = {
        key: int(normalized_report.get(key, -1))
        for key in (
            "record_count",
            "hashed_file_count",
            "unhashed_record_count",
            "runtime_cache_record_count",
            "missing_unhashed_runtime_cache_record_count",
            "policy_normalized_record_count",
        )
    }
    if any(value < 0 for value in totals.values()):
        raise EnvironmentCaptureError(
            f"{role} normalized distribution totals are invalid"
        )
    for normalization in _record_normalizations_for_role(policy, role=role):
        normalization_id = normalization["normalization_id"]
        record_receipt = receipts_by_id.get(normalization_id)
        if not isinstance(record_receipt, dict):
            raise EnvironmentCaptureError(
                f"{role} lacks {normalization_id} source evidence"
            )
        if record_receipt.get("applies") is not True:
            continue
        relative = PurePosixPath(normalization["record_relative_path"])
        record_path = seed.joinpath(*relative.parts)
        archive = _record_preimage_path(output_root, role, normalization)
        source_record_rows = _read_record_rows(archive)
        mutations = _record_mutations(normalization)
        substitutions[record_path.resolve(strict=True)] = (
            source_record_rows,
            mutations,
            normalization,
        )
        site_root = record_path.parent.parent
        source_metrics = _record_metrics_from_rows(
            prefix=seed,
            site_root=site_root,
            record_path=record_path,
            rows=source_record_rows,
            authorized_mutations=mutations,
        )
        normalized_record_rows = _read_record_rows(record_path)
        normalized_metrics = _record_metrics_from_rows(
            prefix=seed,
            site_root=site_root,
            record_path=record_path,
            rows=normalized_record_rows,
            authorized_mutations={},
        )
        for key in totals:
            totals[key] += source_metrics[key] - normalized_metrics[key]
        project = _normalize_project(str(normalization["project"]))
        distribution_row = source_rows_by_project.get(project)
        if distribution_row is None:
            raise EnvironmentCaptureError(
                f"{role} source audit lacks normalized project {project!r}"
            )
        distribution_row.update(
            {
                "record_sha256": normalization["source_record_sha256"],
                "record_entry_digest": source_metrics[
                    "record_entry_digest"
                ],
                "record_entry_count": source_metrics["record_entry_count"],
                "record_validation": (
                    "all_non_normalized_declared_pip_record_hashes"
                ),
                "policy_normalized_record_rows": source_metrics[
                    "policy_normalized_record_rows"
                ],
                "policy_normalized_record_count": source_metrics[
                    "policy_normalized_record_count"
                ],
                "missing_unhashed_runtime_cache_paths": source_metrics[
                    "missing_unhashed_runtime_cache_paths"
                ],
                "missing_unhashed_runtime_cache_record_count": source_metrics[
                    "missing_unhashed_runtime_cache_record_count"
                ],
            }
        )

    source_owners: dict[Path, set[str]] = {}
    projected_owners: dict[Path, set[str]] = {}
    for site_root in _site_packages_roots(seed):
        for metadata_root in sorted(site_root.glob("*.dist-info")):
            if metadata_root.is_symlink() or not metadata_root.is_dir():
                raise EnvironmentCaptureError(
                    f"unsafe installed distribution metadata: {metadata_root}"
                )
            record_path = metadata_root / "RECORD"
            target_key = record_path.resolve(strict=True)
            substitution = substitutions.get(target_key)
            if substitution is None:
                rows = _read_record_rows(record_path)
                mutations: Mapping[
                    tuple[str, str, str], tuple[str, str, str] | None
                ] = {}
            else:
                rows, mutations, _normalization = substitution
            owner = str(metadata_root)
            for row in rows:
                relative, encoded_hash, encoded_size = row
                authorized = row in mutations
                candidate = _record_candidate(
                    prefix=seed,
                    site_root=site_root,
                    record_path=record_path,
                    relative=relative,
                    allow_missing_unhashed_runtime_cache=(
                        not encoded_hash and not encoded_size
                    ),
                    allow_missing_authorized_record=authorized,
                )
                source_owners.setdefault(candidate, set()).add(owner)
                after = mutations.get(row, row)
                if after is None:
                    continue
                projected_candidate = _record_candidate(
                    prefix=seed,
                    site_root=site_root,
                    record_path=record_path,
                    relative=after[0],
                    allow_missing_unhashed_runtime_cache=(
                        not after[1] and not after[2]
                    ),
                    allow_missing_authorized_record=False,
                )
                projected_owners.setdefault(projected_candidate, set()).add(
                    owner
                )

    distributions = [
        source_rows_by_project[project]
        for project in sorted(source_rows_by_project)
    ]
    return {
        "inventory_sha256": _sha256_bytes(_canonical_bytes(distributions)),
        "distribution_count": len(distributions),
        **totals,
        "shared_record_path_count": sum(
            len(owners) > 1 for owners in source_owners.values()
        ),
        "projected_shared_record_path_count": sum(
            len(owners) > 1 for owners in projected_owners.values()
        ),
        "distributions": distributions,
    }


def _conda_records(prefix: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        payload = _read_json(path, description="Conda package record")
        name = payload.get("name")
        version = payload.get("version")
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise EnvironmentCaptureError(f"Conda record lacks name/version: {path}")
        project = _normalize_project(name)
        if project in records:
            raise EnvironmentCaptureError(
                f"duplicate Conda project record {project!r} in {prefix}"
            )
        records[project] = {
            "project": project,
            "version": version,
            "path": path.relative_to(prefix).as_posix(),
            "record_sha256": _sha256_file(path),
            "artifact_sha256": payload.get("sha256"),
            "build": payload.get("build"),
        }
    if not records:
        raise EnvironmentCaptureError(f"Conda records are empty in {prefix}")
    return records


def _policy_normalization(
    policy: Mapping[str, Any], normalization_id: str
) -> Mapping[str, Any]:
    normalizations = policy.get("normalizations")
    if not isinstance(normalizations, list):
        raise EnvironmentCaptureError(
            "environment ownership policy has no normalizations"
        )
    matches = [
        row
        for row in normalizations
        if isinstance(row, dict) and row.get("normalization_id") == normalization_id
    ]
    if len(matches) != 1:
        raise EnvironmentCaptureError(
            f"environment ownership policy must contain one {normalization_id}"
        )
    return matches[0]


def _load_policy(path: Path) -> tuple[dict[str, Any], str]:
    payload = _read_json(path, description="environment ownership policy")
    checksum_path = path.with_suffix(".sha256")
    try:
        fields = checksum_path.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError) as exc:
        raise EnvironmentCaptureError(
            f"cannot read ownership-policy checksum {checksum_path}: {exc}"
        ) from exc
    observed = _sha256_file(path)
    if (
        len(fields) != 2
        or fields[0] != observed
        or fields[1] != path.name
        or _SHA256_RE.fullmatch(fields[0]) is None
    ):
        raise EnvironmentCaptureError("ownership-policy checksum is invalid")
    if (
        payload.get("schema_version") != 1
        or payload.get("release_id") != RELEASE_ID
        or payload.get("policy_id") != "schema5-environment-ownership-v1"
        or payload.get("reject_unlisted_version_conflicts") is not True
        or not isinstance(payload.get("normalizations"), list)
        or len(payload["normalizations"]) != 1
    ):
        raise EnvironmentCaptureError("environment ownership policy is invalid")
    normalization = _policy_normalization(payload, SETUPTOOLS_NORMALIZATION_ID)
    if (
        not isinstance(normalization, dict)
        or normalization.get("normalization_id") != SETUPTOOLS_NORMALIZATION_ID
        or normalization.get("project") != "setuptools"
        or normalization.get("pip_version") != "81.0.0"
        or normalization.get("conda_version") != "82.0.1"
        or normalization.get("conda_build") != "pyh332efcf_0"
        or normalization.get("conda_record_filename")
        != "setuptools-82.0.1-pyh332efcf_0.json"
        or normalization.get("conda_artifact_sha256")
        != "82088a6e4daa33329a30bc26dc19a98c7c1d3f05c0f73ce9845d4eab4924e9e1"
        or set(normalization.get("roles", {})) != set(ROLES)
    ):
        raise EnvironmentCaptureError(
            "ownership policy does not encode the sole approved Setuptools conflict"
        )
    return payload, observed


def _load_integrity_policy(path: Path) -> tuple[dict[str, Any], str]:
    payload = _read_json(
        path, description="environment integrity-normalization policy"
    )
    checksum_path = path.with_suffix(".sha256")
    try:
        fields = checksum_path.read_text(encoding="utf-8").strip().split()
    except (OSError, UnicodeError) as exc:
        raise EnvironmentCaptureError(
            "cannot read integrity-normalization-policy checksum "
            f"{checksum_path}: {exc}"
        ) from exc
    observed = _sha256_file(path)
    if (
        len(fields) != 2
        or fields[0] != observed
        or fields[1] != path.name
        or _SHA256_RE.fullmatch(fields[0]) is None
    ):
        raise EnvironmentCaptureError(
            "integrity-normalization-policy checksum is invalid"
        )
    if (
        payload.get("schema_version") != 1
        or payload.get("release_id") != RELEASE_ID
        or payload.get("policy_id")
        != "schema5-environment-integrity-normalization-v1"
        or payload.get("require_pip_record_hashes") is not True
        or not isinstance(payload.get("normalizations"), list)
        or len(payload["normalizations"]) != len(RECORD_NORMALIZATION_IDS)
    ):
        raise EnvironmentCaptureError(
            "environment integrity-normalization policy is invalid"
        )
    expected_contracts = {
        PIP_RECORD_NORMALIZATION_ID: {
            "project": "pip",
            "version": "26.1.1",
            "record_relative_path": (
                "lib/python3.11/site-packages/pip-26.1.1.dist-info/RECORD"
            ),
            "source_record_sha256": PIP_RECORD_SOURCE_SHA256,
            "normalized_record_sha256": PIP_RECORD_NORMALIZED_SHA256,
            "archive_filename": "pip-26.1.1.RECORD",
            "roles": {
                "harness": "exact_preimage_if_present",
                "serving": "exact_preimage_if_present",
            },
        },
        PACKAGING_RECORD_NORMALIZATION_ID: {
            "project": "packaging",
            "version": "26.2",
            "record_relative_path": (
                "lib/python3.11/site-packages/packaging-26.2.dist-info/RECORD"
            ),
            "source_record_sha256": PACKAGING_RECORD_SOURCE_SHA256,
            "normalized_record_sha256": PACKAGING_RECORD_NORMALIZED_SHA256,
            "archive_filename": "packaging-26.2.RECORD",
            "roles": {
                "harness": "exact_preimage_if_present",
                "serving": "exact_preimage_if_present",
            },
        },
        NUMPY_RECORD_NORMALIZATION_ID: {
            "project": "numpy",
            "version": "2.3.5",
            "record_relative_path": (
                "lib/python3.11/site-packages/numpy-2.3.5.dist-info/RECORD"
            ),
            "source_record_sha256": NUMPY_RECORD_SOURCE_SHA256,
            "normalized_record_sha256": NUMPY_RECORD_NORMALIZED_SHA256,
            "archive_filename": "numpy-2.3.5.RECORD",
            "roles": {
                "harness": "forbidden",
                "serving": "exact_preimage_if_present",
            },
        },
        WHEEL_RECORD_NORMALIZATION_ID: {
            "project": "wheel",
            "version": "0.47.0",
            "record_relative_path": (
                "lib/python3.11/site-packages/wheel-0.47.0.dist-info/RECORD"
            ),
            "source_record_sha256": WHEEL_RECORD_SOURCE_SHA256,
            "normalized_record_sha256": WHEEL_RECORD_NORMALIZED_SHA256,
            "archive_filename": "wheel-0.47.0.RECORD",
            "roles": {
                "harness": "exact_preimage_if_present",
                "serving": "exact_preimage_if_present",
            },
        },
        TORCH_DLPACK_RECORD_NORMALIZATION_ID: {
            "project": "torch-c-dlpack-ext",
            "version": "0.1.5",
            "record_relative_path": (
                "lib/python3.11/site-packages/"
                "torch_c_dlpack_ext-0.1.5.dist-info/RECORD"
            ),
            "source_record_sha256": TORCH_DLPACK_RECORD_SOURCE_SHA256,
            "normalized_record_sha256": TORCH_DLPACK_RECORD_NORMALIZED_SHA256,
            "archive_filename": "torch_c_dlpack_ext-0.1.5.RECORD",
            "roles": {
                "harness": "forbidden",
                "serving": "exact_preimage_if_present",
            },
        },
    }
    for normalization_id, expected in expected_contracts.items():
        record_normalization = _policy_normalization(payload, normalization_id)
        collision_fields = (
            _OWNERSHIP_CONFLICT_AUTHORIZATION_FIELDS
            & set(record_normalization)
        )
        if collision_fields:
            raise EnvironmentCaptureError(
                "RECORD metadata repair cannot authorize a Conda/pip "
                f"ownership conflict: {normalization_id}:{sorted(collision_fields)}"
            )
        for key in (
            "project",
            "version",
            "record_relative_path",
            "source_record_sha256",
            "normalized_record_sha256",
            "archive_filename",
        ):
            if record_normalization.get(key) != expected[key]:
                raise EnvironmentCaptureError(
                    f"ownership policy drifted for {normalization_id}:{key}"
                )
        if {
            role: record_normalization.get("roles", {})
            .get(role, {})
            .get("source_record_policy")
            for role in ROLES
        } != expected["roles"]:
            raise EnvironmentCaptureError(
                f"ownership policy has wrong roles for {normalization_id}"
            )
        if not _record_mutations(record_normalization):
            raise EnvironmentCaptureError(
                f"ownership policy has no mutations for {normalization_id}"
            )
    pip_normalization = _policy_normalization(payload, PIP_RECORD_NORMALIZATION_ID)
    pip_mutations = _record_mutations(pip_normalization)
    if (
        not set(PIP_LAUNCHER_RECORD_ROWS).issubset(pip_mutations)
        or pip_mutations.get(
            (
                "pip-26.1.1.dist-info/INSTALLER",
                "sha256=zuuue4knoyJ-UwPPXg8fezS7VCrXJQrAP7zeNuwvFQg",
                "4",
            )
        )
        != (
            "pip-26.1.1.dist-info/INSTALLER",
            "sha256=0O3uFfkbQG8_mXJuROuZC-bjT9A0W1K5EMVo4O72oqg",
            "5",
        )
        or len(pip_mutations) != 4
    ):
        raise EnvironmentCaptureError("pip RECORD mutation contract drifted")
    packaging = _policy_normalization(
        payload, PACKAGING_RECORD_NORMALIZATION_ID
    )
    if (
        packaging.get("conda_ownership_contract", {}).get("record_sha256")
        != "0516ee64f77f9be64499d8c6a70d19343d0f67b0c4557ffef9f8ad67c5104af9"
        or packaging.get("conda_ownership_contract", {}).get(
            "owned_path_sha256"
        )
        != "bc33022edcb7639ff53355b4e91dade50a0bbf0299efeb6171d1ec0ba5029cfc"
    ):
        raise EnvironmentCaptureError("packaging Conda ownership contract drifted")
    numpy_normalization = _policy_normalization(
        payload, NUMPY_RECORD_NORMALIZATION_ID
    )
    if (
        numpy_normalization.get("conda_project_must_be_absent") is not True
        or _required_retained_rows(numpy_normalization)
        != (
            (
                "numpy/distutils/__pycache__/conv_template.cpython-311.pyc",
                "",
                "",
            ),
        )
    ):
        raise EnvironmentCaptureError("NumPy bytecode RECORD contract drifted")
    torch_normalization = _policy_normalization(
        payload, TORCH_DLPACK_RECORD_NORMALIZATION_ID
    )
    if (
        torch_normalization.get("other_owner_contract", {}).get(
            "record_sha256"
        )
        != "9a08dbf109e448dafd950cfc18256560f004677f2a6fa834c04bd0b919116155"
        or torch_normalization.get("source_file_contracts", [{}])[0].get(
            "sha256"
        )
        != "d034bb09167e69d48f61ffbddf3a0f2e4f6b4233e76f295eaa11b38ac535b657"
    ):
        raise EnvironmentCaptureError("shared-owner RECORD contract drifted")
    return payload, observed


def _combined_policy(
    ownership_policy: Mapping[str, Any],
    integrity_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose two independently closed policies for internal validation only."""

    ownership_rows = ownership_policy.get("normalizations")
    integrity_rows = integrity_policy.get("normalizations")
    if (
        not isinstance(ownership_rows, list)
        or len(ownership_rows) != 1
        or not isinstance(integrity_rows, list)
        or len(integrity_rows) != len(RECORD_NORMALIZATION_IDS)
    ):
        raise EnvironmentCaptureError(
            "environment policy composition has the wrong closed cardinality"
        )
    return {"normalizations": [*ownership_rows, *integrity_rows]}


def _validate_recovered_record(
    path: Path, normalization: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    payload = _read_json(path, description="recovered Setuptools Conda record")
    digest = _sha256_file(path)
    if (
        payload.get("name") != normalization["project"]
        or payload.get("version") != normalization["conda_version"]
        or payload.get("build") != normalization["conda_build"]
        or payload.get("sha256") != normalization["conda_artifact_sha256"]
    ):
        raise EnvironmentCaptureError(
            "recovered Conda record does not match the ownership policy"
        )
    return payload, digest


def _validate_incident(
    path: Path,
    *,
    affected_prefixes: Sequence[Path],
    artifact_sha256: str,
) -> tuple[str, dict[str, str]]:
    payload = _read_json(path, description="Conda reconciliation incident")
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise EnvironmentCaptureError("Conda reconciliation incident must be read-only")
    serialized = json.dumps(payload, sort_keys=True)
    recognized_sealed_incident = (
        payload.get("protocol")
        == "schema5-conda-pip-reconciliation-incident-v1"
        and payload.get("classification")
        == "source_metadata_reconciled_by_conda_pip_interop"
        and payload.get("live_sources_must_not_be_queried_by_conda") is True
        and payload.get("resolution")
        == "capture_live_bytes_without_conda_then_normalize_only_immutable_seeds"
        and _SHA256_RE.fullmatch(str(payload.get("incident_id", ""))) is not None
    )
    if (
        not (payload.get("complete") is True or recognized_sealed_incident)
        or "conda" not in serialized.lower()
        or "reconcil" not in serialized.lower()
        or artifact_sha256 not in serialized
        or any(str(prefix) not in serialized for prefix in affected_prefixes)
    ):
        raise EnvironmentCaptureError(
            "Conda reconciliation incident does not bind all affected prefixes "
            "and the recovered artifact"
        )
    source_record_states: dict[str, str] = {}
    expected_prefix_fields = {
        "harness": "harness_prefix",
        "serving": "serving_prefix",
    }
    expected_state_fields = {
        "harness": "harness_stale_conda_record_present",
        "serving": "serving_stale_conda_record_present",
    }
    if len(affected_prefixes) != len(ROLES):
        raise EnvironmentCaptureError(
            "Conda reconciliation incident requires both environment roles"
        )
    for role, prefix in zip(ROLES, affected_prefixes, strict=True):
        observed_prefix = payload.get(expected_prefix_fields[role])
        observed_present = payload.get(expected_state_fields[role])
        if observed_prefix != str(prefix) or type(observed_present) is not bool:
            raise EnvironmentCaptureError(
                "Conda reconciliation incident does not bind the exact "
                f"{role} source-record state"
            )
        source_record_states[role] = (
            "present" if observed_present else "absent"
        )
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists() or sidecar.is_symlink():
        if sidecar.is_symlink() or not sidecar.is_file():
            raise EnvironmentCaptureError(
                "Conda reconciliation incident checksum is unsafe"
            )
        fields = sidecar.read_text(encoding="utf-8").strip().split()
        if (
            len(fields) != 2
            or fields[0] != _sha256_file(path)
            or fields[1] != path.name
            or stat.S_IMODE(sidecar.stat().st_mode) & 0o222
        ):
            raise EnvironmentCaptureError(
                "Conda reconciliation incident checksum is invalid"
            )
    return _sha256_file(path), source_record_states


def _distribution_versions(prefix: Path) -> dict[str, str]:
    report = distribution_inventory(prefix)
    return {
        str(row["project"]): str(row["version"])
        for row in report["distributions"]
    }


def source_distribution_inventory(
    prefix: Path,
    *,
    policy: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    """Audit a mutable source with only its exact checksummed RECORD repair."""

    if role not in ROLES:
        raise EnvironmentCaptureError(f"unknown environment role: {role!r}")
    normalizations = _record_normalizations_for_role(policy, role=role)
    report = distribution_inventory(
        prefix,
        source_record_normalizations=normalizations,
        role=role,
    )
    if report["projected_shared_record_path_count"] != 0:
        raise EnvironmentCaptureError(
            f"{role} source RECORD repairs project "
            f"{report['projected_shared_record_path_count']} shared paths"
        )
    return report


def _record_normalizations_for_role(
    policy: Mapping[str, Any], *, role: str
) -> tuple[Mapping[str, Any], ...]:
    if role not in ROLES:
        raise EnvironmentCaptureError(f"unknown environment role: {role!r}")
    selected: list[Mapping[str, Any]] = []
    for normalization_id in RECORD_NORMALIZATION_IDS:
        normalization = _policy_normalization(policy, normalization_id)
        role_policy = normalization.get("roles", {}).get(role, {}).get(
            "source_record_policy"
        )
        if role_policy == "exact_preimage_if_present":
            selected.append(normalization)
        elif role_policy != "forbidden":
            raise EnvironmentCaptureError(
                f"{normalization_id} has an invalid role policy for {role}"
            )
    return tuple(selected)


def _validate_ownership(
    prefix: Path,
    *,
    normalization: Mapping[str, Any],
    allow_known_conflict: bool,
    source_record_normalization: Mapping[str, Any] | None = None,
    source_record_normalizations: Sequence[Mapping[str, Any]] = (),
    role: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    # RECORD canonicalizations are deliberately not ownership exceptions.  Enforce
    # the sole collision identity here, rather than trusting every caller to pass
    # the correct policy row.  This keeps future record-repair entries from
    # accidentally widening admission for a Conda/pip version collision.
    if (
        normalization.get("normalization_id") != SETUPTOOLS_NORMALIZATION_ID
        or normalization.get("project") != "setuptools"
        or normalization.get("pip_version") != "81.0.0"
        or normalization.get("conda_version") != "82.0.1"
        or normalization.get("conda_build") != "pyh332efcf_0"
        or normalization.get("conda_record_filename")
        != "setuptools-82.0.1-pyh332efcf_0.json"
        or normalization.get("conda_artifact_sha256")
        != "82088a6e4daa33329a30bc26dc19a98c7c1d3f05c0f73ce9845d4eab4924e9e1"
    ):
        raise EnvironmentCaptureError(
            "only the approved Setuptools normalization may authorize a "
            "Conda/pip ownership conflict"
        )
    distributions = distribution_inventory(
        prefix,
        source_record_normalization=source_record_normalization,
        source_record_normalizations=source_record_normalizations,
        role=role,
    )
    pip_versions = {
        str(row["project"]): str(row["version"])
        for row in distributions["distributions"]
    }
    conda_records = _conda_records(prefix)
    conflicts = {
        project: (pip_versions[project], record["version"])
        for project, record in conda_records.items()
        if project in pip_versions and pip_versions[project] != record["version"]
    }
    approved = {
        normalization["project"]: (
            normalization["pip_version"],
            normalization["conda_version"],
        )
    }
    if conflicts and (not allow_known_conflict or conflicts != approved):
        raise EnvironmentCaptureError(
            f"unlisted Conda/pip ownership version conflict(s): {conflicts}"
        )
    if pip_versions.get(normalization["project"]) != normalization["pip_version"]:
        raise EnvironmentCaptureError(
            "captured seed does not preserve runtime Setuptools 81.0.0"
        )
    return distributions, conda_records


def validate_normalized_seed(
    prefix: Path, *, policy: Mapping[str, Any]
) -> dict[str, Any]:
    normalization = _policy_normalization(policy, SETUPTOOLS_NORMALIZATION_ID)
    record_states: dict[str, dict[str, Any]] = {}
    for normalization_id in RECORD_NORMALIZATION_IDS:
        record_normalization = _policy_normalization(
            policy, normalization_id
        )
        relative = PurePosixPath(record_normalization["record_relative_path"])
        record_path = prefix.joinpath(*relative.parts)
        state = "distribution_absent"
        observed_sha256: str | None = None
        if record_path.parent.exists():
            if record_path.is_symlink() or not record_path.is_file():
                raise EnvironmentCaptureError(
                    f"normalized seed has an unsafe {normalization_id} RECORD"
                )
            observed_sha256 = _sha256_file(record_path)
            if observed_sha256 != record_normalization[
                "normalized_record_sha256"
            ]:
                raise EnvironmentCaptureError(
                    f"normalized seed {normalization_id} RECORD does not have "
                    "the exact approved postimage"
                )
            state = "exact_normalized_postimage"
        record_states[normalization_id] = {
            "state": state,
            "record_sha256": observed_sha256,
        }
    distributions, conda_records = _validate_ownership(
        prefix,
        normalization=normalization,
        allow_known_conflict=False,
    )
    if distributions["policy_normalized_record_count"] != 0:
        raise EnvironmentCaptureError(
            "normalized seed still depends on a RECORD policy exemption"
        )
    if distributions["shared_record_path_count"] != 0:
        raise EnvironmentCaptureError(
            "normalized seed still has shared RECORD path ownership"
        )
    if normalization["project"] in conda_records:
        raise EnvironmentCaptureError(
            "normalized seed still has a Setuptools Conda ownership record"
        )
    pip_state = record_states[PIP_RECORD_NORMALIZATION_ID]
    return {
        "setuptools_runtime_version": normalization["pip_version"],
        "setuptools_conda_record_count": 0,
        "pip_record_normalization_state": pip_state["state"],
        "pip_record_sha256": pip_state["record_sha256"],
        "record_normalization_states": record_states,
        "distribution_inventory": distributions,
        "conda_record_count": len(conda_records),
    }


def _normalization_paths(output_root: Path, role: str) -> dict[str, Path]:
    return {
        "intent": output_root / "normalization" / f"{role}.intent.json",
        "receipt": output_root / "normalization" / f"{role}.receipt.json",
        "archive": output_root
        / "normalization"
        / "preimages"
        / role
        / "setuptools-82.0.1-pyh332efcf_0.json",
        "pip_record_archive": output_root
        / "normalization"
        / "preimages"
        / role
        / "pip-26.1.1.RECORD",
    }


def _record_preimage_path(
    output_root: Path, role: str, normalization: Mapping[str, Any]
) -> Path:
    filename = normalization.get("archive_filename")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise EnvironmentCaptureError("unsafe RECORD preimage archive filename")
    return output_root / "normalization" / "preimages" / role / filename


def _restore_interrupted_normalization_preimages(
    *,
    output_root: Path,
    role: str,
    seed: Path,
    policy: Mapping[str, Any],
) -> None:
    """Restore a copied seed to source bytes before resuming its transaction."""

    paths = _normalization_paths(output_root, role)
    if not paths["intent"].is_file() or paths["intent"].is_symlink():
        return
    intent = _read_json(
        paths["intent"], description=f"{role} interrupted normalization intent"
    )
    candidate = dict(intent)
    intent_id = candidate.pop("intent_id", None)
    if intent_id != _sha256_bytes(_canonical_bytes(candidate)):
        raise EnvironmentCaptureError(
            f"{role} interrupted normalization intent is invalid"
        )
    record_intents = intent.get("record_normalizations")
    if not isinstance(record_intents, list):
        raise EnvironmentCaptureError(
            f"{role} interrupted normalization lacks RECORD evidence"
        )
    intents_by_id = {
        row.get("normalization_id"): row
        for row in record_intents
        if isinstance(row, dict)
    }
    selected = _record_normalizations_for_role(policy, role=role)
    if set(intents_by_id) != {
        row["normalization_id"] for row in selected
    }:
        raise EnvironmentCaptureError(
            f"{role} interrupted RECORD intent inventory drifted"
        )
    for normalization in selected:
        normalization_id = normalization["normalization_id"]
        record_intent = intents_by_id[normalization_id]
        if (
            record_intent.get("record_relative_path")
            != normalization["record_relative_path"]
            or record_intent.get("source_record_sha256")
            != normalization["source_record_sha256"]
            or record_intent.get("normalized_record_sha256")
            != normalization["normalized_record_sha256"]
        ):
            raise EnvironmentCaptureError(
                f"{role} interrupted {normalization_id} drifted"
            )
        if record_intent.get("applies") is not True:
            continue
        relative = PurePosixPath(normalization["record_relative_path"])
        record_path = seed.joinpath(*relative.parts)
        if record_path.is_symlink() or not record_path.is_file():
            raise EnvironmentCaptureError(
                f"{role} interrupted {normalization_id} target is unsafe"
            )
        observed = _sha256_file(record_path)
        if observed == normalization["source_record_sha256"]:
            continue
        if observed != normalization["normalized_record_sha256"]:
            raise EnvironmentCaptureError(
                f"{role} interrupted {normalization_id} has an unknown state"
            )
        archive = _record_preimage_path(
            output_root, role, normalization
        )
        if (
            archive.is_symlink()
            or not archive.is_file()
            or _sha256_file(archive)
            != normalization["source_record_sha256"]
        ):
            raise EnvironmentCaptureError(
                f"{role} interrupted {normalization_id} archive is unavailable"
            )
        _atomic_replace_bound(
            record_path,
            archive.read_bytes(),
            expected_before_sha256=normalization["normalized_record_sha256"],
            description=f"{role} interrupted {normalization_id} restoration",
        )


def _normalize_seed(
    *,
    output_root: Path,
    role: str,
    source: Path,
    seed: Path,
    policy: Mapping[str, Any],
    ownership_policy_sha256: str,
    integrity_normalization_policy_sha256: str,
    incident_path: Path,
    incident_sha256: str,
    recovered_record_path: Path,
    recovered_record_sha256: str,
) -> dict[str, Any]:
    normalization = _policy_normalization(policy, SETUPTOOLS_NORMALIZATION_ID)
    paths = _normalization_paths(output_root, role)
    role_policy = normalization["roles"][role]["source_record_policy"]
    record_path = seed / "conda-meta" / normalization["conda_record_filename"]
    observed_source_state = (
        "present"
        if record_path.is_file() and not record_path.is_symlink()
        else "absent"
    )
    recorded_intent = (
        _read_json(paths["intent"], description=f"{role} normalization intent")
        if paths["intent"].is_file() and not paths["intent"].is_symlink()
        else None
    )
    source_state = (
        str(recorded_intent.get("source_record_state"))
        if recorded_intent is not None
        else observed_source_state
    )
    # A resumed serving capture whose stale record was present recreates that exact
    # bound byte from the source inventory before returning here.  All other intent
    # states must match what is currently visible.
    if observed_source_state != source_state:
        raise EnvironmentCaptureError(
            f"{role} normalization intent/source state is inconsistent"
        )
    if source_state == "present" and role_policy == "absent_requires_incident":
        raise EnvironmentCaptureError(
            f"{role} Setuptools Conda record unexpectedly survived reconciliation"
        )
    if source_state == "absent" and role_policy not in {
        "absent_requires_incident",
        "present_or_absent_requires_incident",
    }:
        raise EnvironmentCaptureError(
            f"{role} Setuptools record absence is not authorized"
        )
    if source_state == "present":
        record_payload, record_sha256 = _validate_recovered_record(
            record_path, normalization
        )
        preimage_bytes = record_path.read_bytes()
    else:
        record_payload, _unused = _validate_recovered_record(
            recovered_record_path, normalization
        )
        record_sha256 = recovered_record_sha256
        preimage_bytes = recovered_record_path.read_bytes()
    record_work: list[dict[str, Any]] = []
    for record_normalization in _record_normalizations_for_role(
        policy, role=role
    ):
        authorization = _source_record_authorization(
            seed, record_normalization, role=role
        )
        applies = authorization is not None
        mutations = _record_mutations(record_normalization)
        mutation_rows = [
            {
                "before": {
                    "path": before[0],
                    "hash": before[1],
                    "size": before[2],
                },
                "after": (
                    {
                        "path": after[0],
                        "hash": after[1],
                        "size": after[2],
                    }
                    if after is not None
                    else None
                ),
            }
            for before, after in mutations.items()
        ]
        record_intent: dict[str, Any] = {
            "normalization_id": record_normalization["normalization_id"],
            "applies": applies,
            "record_relative_path": record_normalization["record_relative_path"],
            "source_record_sha256": record_normalization[
                "source_record_sha256"
            ],
            "normalized_record_sha256": record_normalization[
                "normalized_record_sha256"
            ],
            "mutations": mutation_rows,
            "source_file_evidence": (
                authorization["source_file_evidence"]
                if authorization is not None
                else []
            ),
            "runtime_targets": (
                _pip_launcher_runtime_states(seed, record_normalization)
                if applies
                and record_normalization["normalization_id"]
                == PIP_RECORD_NORMALIZATION_ID
                else []
            ),
            "operation": (
                "archive_complete_record_then_atomically_apply_exact_mutations"
                if applies
                else "not_applicable_distribution_absent"
            ),
        }
        record_work.append(
            {
                "normalization": record_normalization,
                "authorization": authorization,
                "intent": record_intent,
                "archive": _record_preimage_path(
                    output_root, role, record_normalization
                ),
            }
        )
    record_intents = [row["intent"] for row in record_work]
    pip_intent = next(
        row
        for row in record_intents
        if row["normalization_id"] == PIP_RECORD_NORMALIZATION_ID
    )
    intent: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "role": role,
        "source_prefix": str(source),
        "seed_prefix": str(seed),
        "ownership_policy_sha256": ownership_policy_sha256,
        "integrity_normalization_policy_sha256": (
            integrity_normalization_policy_sha256
        ),
        "incident_path": str(incident_path),
        "incident_sha256": incident_sha256,
        "source_record_state": source_state,
        "record_filename": normalization["conda_record_filename"],
        "record_sha256": record_sha256,
        "conda_artifact_sha256": normalization["conda_artifact_sha256"],
        "runtime_project": normalization["project"],
        "runtime_version": normalization["pip_version"],
        "operation": (
            "archive_preimages_then_normalize_setuptools_and_exact_records"
        ),
        "pip_record_normalization": pip_intent,
        "record_normalizations": record_intents,
    }
    intent["intent_id"] = _sha256_bytes(_canonical_bytes(intent))
    if recorded_intent is not None and recorded_intent != intent:
        raise EnvironmentCaptureError(f"{role} normalization intent drifted")
    _atomic_write_once(paths["intent"], _json_bytes(intent))
    _atomic_write_once(paths["archive"], preimage_bytes)
    if _sha256_file(paths["archive"]) != record_sha256:
        raise EnvironmentCaptureError("normalization preimage archive drifted")
    for row in record_work:
        if row["authorization"] is None:
            continue
        record_normalization = row["normalization"]
        record_relative = PurePosixPath(
            record_normalization["record_relative_path"]
        )
        record_preimage = seed.joinpath(*record_relative.parts).read_bytes()
        _atomic_write_once(row["archive"], record_preimage)
        if _sha256_file(row["archive"]) != record_normalization[
            "source_record_sha256"
        ]:
            raise EnvironmentCaptureError(
                f"{record_normalization['normalization_id']} archive drifted"
            )
    if record_path.exists() or record_path.is_symlink():
        if record_path.is_symlink() or not record_path.is_file():
            raise EnvironmentCaptureError(
                f"unsafe Setuptools Conda record path: {record_path}"
            )
        if _sha256_file(record_path) != record_sha256:
            raise EnvironmentCaptureError(
                "Setuptools Conda record changed after normalization intent"
            )
        record_path.unlink()
        _fsync_directory(record_path.parent)
    for row in record_work:
        authorization = row["authorization"]
        if authorization is None:
            continue
        record_normalization = row["normalization"]
        relative = PurePosixPath(
            record_normalization["record_relative_path"]
        )
        target_record_path = seed.joinpath(*relative.parts)
        _atomic_replace_bound(
            target_record_path,
            bytes(authorization["postimage"]),
            expected_before_sha256=record_normalization[
                "source_record_sha256"
            ],
            description=(
                f"{role} {record_normalization['normalization_id']} normalization"
            ),
        )
    normalized = validate_normalized_seed(seed, policy=policy)
    if normalized["distribution_inventory"]["policy_normalized_record_count"] != 0:
        raise EnvironmentCaptureError(
            "normalized seed still depends on a RECORD policy exemption"
        )
    record_receipts: list[dict[str, Any]] = []
    for row in record_work:
        record_normalization = row["normalization"]
        authorization = row["authorization"]
        mutations = _record_mutations(record_normalization)
        applies = authorization is not None
        record_receipts.append(
            {
                **row["intent"],
                "complete": True,
                "archived_preimage": (
                    {
                        "path": str(row["archive"]),
                        "sha256": record_normalization[
                            "source_record_sha256"
                        ],
                    }
                    if applies
                    else None
                ),
                "before_sha256": (
                    record_normalization["source_record_sha256"]
                    if applies
                    else None
                ),
                "after_sha256": (
                    record_normalization["normalized_record_sha256"]
                    if applies
                    else None
                ),
                "removed_row_count": (
                    sum(after is None for after in mutations.values())
                    if applies
                    else 0
                ),
                "rewritten_row_count": (
                    sum(after is not None for after in mutations.values())
                    if applies
                    else 0
                ),
            }
        )
    pip_receipt = next(
        row
        for row in record_receipts
        if row["normalization_id"] == PIP_RECORD_NORMALIZATION_ID
    )
    receipt: dict[str, Any] = {
        **intent,
        "complete": True,
        "archived_preimage": {
            "path": str(paths["archive"]),
            "sha256": record_sha256,
            "record": {
                "name": record_payload["name"],
                "version": record_payload["version"],
                "build": record_payload["build"],
                "artifact_sha256": record_payload["sha256"],
            },
        },
        "pip_record_normalization": pip_receipt,
        "record_normalizations": record_receipts,
        "normalized_identity": {
            "setuptools_runtime_version": normalized["setuptools_runtime_version"],
            "setuptools_conda_record_count": 0,
            "pip_record_normalization_state": normalized[
                "pip_record_normalization_state"
            ],
            "pip_record_sha256": normalized["pip_record_sha256"],
            "record_normalization_states": normalized[
                "record_normalization_states"
            ],
            "distribution_inventory_sha256": normalized[
                "distribution_inventory"
            ]["inventory_sha256"],
            "conda_record_count": normalized["conda_record_count"],
        },
    }
    receipt["receipt_id"] = _sha256_bytes(_canonical_bytes(receipt))
    _atomic_write_once(paths["receipt"], _json_bytes(receipt))
    return receipt


def _verify_normalization_receipt(
    *,
    output_root: Path,
    role: str,
    seed: Path,
    policy: Mapping[str, Any],
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = _normalization_paths(output_root, role)
    receipt = _read_json(
        paths["receipt"], description=f"{role} normalization receipt"
    )
    receipt_fields = {
        "schema_version",
        "release_id",
        "role",
        "source_prefix",
        "seed_prefix",
        "ownership_policy_sha256",
        "integrity_normalization_policy_sha256",
        "incident_path",
        "incident_sha256",
        "source_record_state",
        "record_filename",
        "record_sha256",
        "conda_artifact_sha256",
        "runtime_project",
        "runtime_version",
        "operation",
        "pip_record_normalization",
        "record_normalizations",
        "intent_id",
        "complete",
        "archived_preimage",
        "normalized_identity",
        "receipt_id",
    }
    _require_exact_fields(
        receipt, receipt_fields, description=f"{role} normalization receipt"
    )
    candidate = dict(receipt)
    receipt_id = candidate.pop("receipt_id", None)
    normalization = _policy_normalization(
        policy, SETUPTOOLS_NORMALIZATION_ID
    )
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("release_id") != RELEASE_ID
        or receipt.get("role") != role
        or receipt.get("seed_prefix") != str(seed)
        or receipt.get("record_filename")
        != normalization["conda_record_filename"]
        or receipt.get("conda_artifact_sha256")
        != normalization["conda_artifact_sha256"]
        or receipt.get("runtime_project") != normalization["project"]
        or receipt.get("runtime_version") != normalization["pip_version"]
        or receipt.get("operation")
        != "archive_preimages_then_normalize_setuptools_and_exact_records"
        or not isinstance(receipt.get("source_prefix"), str)
        or not Path(str(receipt["source_prefix"])).is_absolute()
        or not isinstance(receipt.get("ownership_policy_sha256"), str)
        or _SHA256_RE.fullmatch(receipt["ownership_policy_sha256"]) is None
        or not isinstance(
            receipt.get("integrity_normalization_policy_sha256"), str
        )
        or _SHA256_RE.fullmatch(
            receipt["integrity_normalization_policy_sha256"]
        )
        is None
        or not isinstance(receipt.get("incident_path"), str)
        or not Path(str(receipt["incident_path"])).is_absolute()
        or _SHA256_RE.fullmatch(str(receipt.get("incident_sha256", "")))
        is None
        or receipt.get("source_record_state") not in {"present", "absent"}
        or _SHA256_RE.fullmatch(str(receipt.get("record_sha256", "")))
        is None
    ):
        raise EnvironmentCaptureError(
            f"{role} normalization receipt identity is invalid"
        )
    if (
        receipt.get("complete") is not True
        or receipt_id != _sha256_bytes(_canonical_bytes(candidate))
        or stat.S_IMODE(paths["receipt"].stat().st_mode) & 0o222
    ):
        raise EnvironmentCaptureError(f"{role} normalization receipt is invalid")
    if expected_binding is not None:
        expected = {
            "source_prefix": expected_binding["source_prefix"],
            "seed_prefix": str(seed),
            "ownership_policy_sha256": expected_binding[
                "ownership_policy_sha256"
            ],
            "integrity_normalization_policy_sha256": expected_binding[
                "integrity_normalization_policy_sha256"
            ],
            "incident_path": expected_binding["incident_path"],
            "incident_sha256": expected_binding["incident_sha256"],
            "source_record_state": expected_binding["source_record_state"],
            "record_sha256": expected_binding["record_sha256"],
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise EnvironmentCaptureError(
                f"{role} normalization receipt upstream binding drifted"
            )
    archived = receipt.get("archived_preimage")
    if (
        not isinstance(archived, dict)
        or set(archived) != {"path", "sha256", "record"}
        or not isinstance(archived.get("record"), dict)
        or set(archived["record"])
        != {"name", "version", "build", "artifact_sha256"}
        or paths["archive"].is_symlink()
        or not paths["archive"].is_file()
        or archived.get("path") != str(paths["archive"])
        or archived.get("sha256") != _sha256_file(paths["archive"])
        or archived["record"]
        != {
            "name": normalization["project"],
            "version": normalization["conda_version"],
            "build": normalization["conda_build"],
            "artifact_sha256": normalization["conda_artifact_sha256"],
        }
        or receipt.get("record_sha256") != archived.get("sha256")
        or stat.S_IMODE(paths["archive"].stat().st_mode) & 0o222
    ):
        raise EnvironmentCaptureError(
            f"{role} Setuptools normalization preimage evidence is invalid"
        )
    _validate_recovered_record(paths["archive"], normalization)
    record_receipts = receipt.get("record_normalizations")
    if not isinstance(record_receipts, list):
        raise EnvironmentCaptureError(f"{role} RECORD receipts are missing")
    receipts_by_id = {
        row.get("normalization_id"): row
        for row in record_receipts
        if isinstance(row, dict)
    }
    selected = _record_normalizations_for_role(policy, role=role)
    if set(receipts_by_id) != {
        row["normalization_id"] for row in selected
    }:
        raise EnvironmentCaptureError(
            f"{role} RECORD receipt inventory drifted"
        )
    for normalization in selected:
        normalization_id = normalization["normalization_id"]
        record_receipt = receipts_by_id[normalization_id]
        mutations = _record_mutations(normalization)
        record_receipt_fields = {
            "normalization_id",
            "applies",
            "record_relative_path",
            "source_record_sha256",
            "normalized_record_sha256",
            "mutations",
            "source_file_evidence",
            "runtime_targets",
            "operation",
            "complete",
            "archived_preimage",
            "before_sha256",
            "after_sha256",
            "removed_row_count",
            "rewritten_row_count",
        }
        _require_exact_fields(
            record_receipt,
            record_receipt_fields,
            description=f"{role} {normalization_id} receipt",
        )
        expected_mutations = [
            {
                "before": {
                    "path": before[0],
                    "hash": before[1],
                    "size": before[2],
                },
                "after": (
                    {
                        "path": after[0],
                        "hash": after[1],
                        "size": after[2],
                    }
                    if after is not None
                    else None
                ),
            }
            for before, after in mutations.items()
        ]
        if (
            record_receipt.get("record_relative_path")
            != normalization["record_relative_path"]
            or record_receipt.get("source_record_sha256")
            != normalization["source_record_sha256"]
            or record_receipt.get("normalized_record_sha256")
            != normalization["normalized_record_sha256"]
            or record_receipt.get("mutations") != expected_mutations
            or record_receipt.get("complete") is not True
            or record_receipt.get("operation")
            not in {
                "archive_complete_record_then_atomically_apply_exact_mutations",
                "not_applicable_distribution_absent",
            }
        ):
            raise EnvironmentCaptureError(
                f"{role} {normalization_id} receipt is invalid"
            )
        relative = PurePosixPath(normalization["record_relative_path"])
        record_path = seed.joinpath(*relative.parts)
        archive = _record_preimage_path(output_root, role, normalization)
        if record_receipt.get("applies") is True:
            archive_receipt = record_receipt.get("archived_preimage")
            if (
                not isinstance(archive_receipt, dict)
                or archive.is_symlink()
                or not archive.is_file()
                or archive_receipt.get("path") != str(archive)
                or archive_receipt.get("sha256")
                != normalization["source_record_sha256"]
                or _sha256_file(archive)
                != normalization["source_record_sha256"]
                or record_receipt.get("before_sha256")
                != normalization["source_record_sha256"]
                or record_receipt.get("after_sha256")
                != normalization["normalized_record_sha256"]
                or record_receipt.get("removed_row_count")
                != sum(after is None for after in mutations.values())
                or record_receipt.get("rewritten_row_count")
                != sum(after is not None for after in mutations.values())
                or _sha256_file(record_path)
                != normalization["normalized_record_sha256"]
                or _normalized_record_bytes(
                    archive.read_bytes(), normalization
                )
                != record_path.read_bytes()
                or record_receipt.get("source_file_evidence")
                != _validate_record_source_contracts(
                    seed, normalization=normalization, role=role
                )
                or record_receipt.get("runtime_targets")
                != (
                    _pip_launcher_runtime_states_from_sealed_evidence(
                        output_root=output_root,
                        role=role,
                        seed=seed,
                        normalization=normalization,
                    )
                    if normalization_id == PIP_RECORD_NORMALIZATION_ID
                    else []
                )
                or record_receipt.get("operation")
                != "archive_complete_record_then_atomically_apply_exact_mutations"
                or stat.S_IMODE(archive.stat().st_mode) & 0o222
            ):
                raise EnvironmentCaptureError(
                    f"{role} {normalization_id} evidence drifted"
                )
        elif (
            record_receipt.get("applies") is not False
            or record_receipt.get("archived_preimage") is not None
            or record_receipt.get("before_sha256") is not None
            or record_receipt.get("after_sha256") is not None
            or record_receipt.get("removed_row_count") != 0
            or record_receipt.get("rewritten_row_count") != 0
            or record_receipt.get("source_file_evidence") != []
            or record_receipt.get("runtime_targets") != []
            or record_receipt.get("operation")
            != "not_applicable_distribution_absent"
            or record_path.parent.exists()
        ):
            raise EnvironmentCaptureError(
                f"{role} {normalization_id} absence receipt is inconsistent"
            )
    pip_receipt = receipts_by_id[PIP_RECORD_NORMALIZATION_ID]
    if receipt.get("pip_record_normalization") != pip_receipt:
        raise EnvironmentCaptureError(
            f"{role} pip RECORD receipt alias drifted"
        )
    normalized = validate_normalized_seed(seed, policy=policy)
    expected_normalized_identity = {
        "setuptools_runtime_version": normalized[
            "setuptools_runtime_version"
        ],
        "setuptools_conda_record_count": normalized[
            "setuptools_conda_record_count"
        ],
        "pip_record_normalization_state": normalized[
            "pip_record_normalization_state"
        ],
        "pip_record_sha256": normalized["pip_record_sha256"],
        "record_normalization_states": normalized[
            "record_normalization_states"
        ],
        "distribution_inventory_sha256": normalized[
            "distribution_inventory"
        ]["inventory_sha256"],
        "conda_record_count": normalized["conda_record_count"],
    }
    if receipt.get("normalized_identity") != expected_normalized_identity:
        raise EnvironmentCaptureError(
            f"{role} normalized identity drifted"
        )

    intent_path = paths["intent"]
    intent = _read_json(
        intent_path, description=f"{role} normalization intent"
    )
    intent_fields = receipt_fields - {
        "complete",
        "archived_preimage",
        "normalized_identity",
        "receipt_id",
    }
    _require_exact_fields(
        intent, intent_fields, description=f"{role} normalization intent"
    )
    expected_record_intents = []
    receipt_only_fields = {
        "complete",
        "archived_preimage",
        "before_sha256",
        "after_sha256",
        "removed_row_count",
        "rewritten_row_count",
    }
    for record_receipt in record_receipts:
        expected_record_intents.append(
            {
                key: value
                for key, value in record_receipt.items()
                if key not in receipt_only_fields
            }
        )
    expected_intent = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "complete",
            "archived_preimage",
            "normalized_identity",
            "receipt_id",
        }
    }
    expected_intent["record_normalizations"] = expected_record_intents
    expected_intent["pip_record_normalization"] = next(
        row
        for row in expected_record_intents
        if row["normalization_id"] == PIP_RECORD_NORMALIZATION_ID
    )
    intent_candidate = dict(intent)
    intent_id = intent_candidate.pop("intent_id", None)
    if (
        intent != expected_intent
        or intent_id != _sha256_bytes(_canonical_bytes(intent_candidate))
        or receipt.get("intent_id") != intent_id
        or stat.S_IMODE(intent_path.stat().st_mode) & 0o222
    ):
        raise EnvironmentCaptureError(
            f"{role} normalization intent/receipt binding drifted"
        )
    return receipt


def _seal_tree(root: Path) -> None:
    rows = list(_walk_entries(root))
    for path, info in reversed(rows):
        if not stat.S_ISLNK(info.st_mode):
            os.chmod(path, stat.S_IMODE(info.st_mode) & ~0o222)
    os.chmod(root, stat.S_IMODE(root.stat().st_mode) & ~0o222)
    _fsync_directory(root)


def _assert_read_only(root: Path) -> None:
    if stat.S_IMODE(root.stat().st_mode) & 0o222:
        raise EnvironmentCaptureError(f"captured seed root is writable: {root}")
    for path, info in _walk_entries(root):
        if not stat.S_ISLNK(info.st_mode) and stat.S_IMODE(info.st_mode) & 0o222:
            raise EnvironmentCaptureError(f"captured seed entry is writable: {path}")


def _role_stage_payload(
    *,
    role: str,
    source: Path,
    seed: Path,
    before: Mapping[str, Any],
    normalized_inventory: Mapping[str, Any],
    copy_report: Mapping[str, int],
    symlink_report: Mapping[str, int],
    normalization_receipt: Mapping[str, Any],
    normalization_inventory_delta: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "role": role,
        "source_prefix": str(source),
        "seed_prefix": str(seed),
        "copy_method": "buffered_read_write_exclusive_create",
        "source_before_inventory_sha256": before["inventory_sha256"],
        "source_after_inventory_sha256": before["inventory_sha256"],
        "pre_normalization_copy_inventory_sha256": before["inventory_sha256"],
        "normalized_content_inventory": normalized_inventory,
        "copy_audit": dict(copy_report),
        "symlink_audit": dict(symlink_report),
        "normalization_receipt_id": normalization_receipt["receipt_id"],
        "normalization_receipt_sha256": _sha256_file(
            _normalization_paths(seed.parent.parent, role)["receipt"]
        ),
        "normalization_inventory_delta": dict(
            normalization_inventory_delta
        ),
        "sealed_read_only": True,
    }
    payload["record_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def _capture_role(
    *,
    output_root: Path,
    role: str,
    source: Path,
    before: Mapping[str, Any],
    policy: Mapping[str, Any],
    ownership_policy_sha256: str,
    integrity_normalization_policy_sha256: str,
    incident_path: Path,
    incident_sha256: str,
    recovered_record_path: Path,
    recovered_record_sha256: str,
) -> dict[str, Any]:
    seed = output_root / "seeds" / role
    marker_path = output_root / ROLE_MARKERS[role]
    if marker_path.is_file() and not marker_path.is_symlink():
        return _verify_role_stage(output_root, role=role, source_required=False)
    live_before = directory_inventory(source)
    if live_before != before:
        raise EnvironmentCaptureError(
            f"{role} source drifted after the marker-first capture intent"
        )
    receipt_path = _normalization_paths(output_root, role)["receipt"]
    if receipt_path.is_file() and not receipt_path.is_symlink():
        receipt = _verify_normalization_receipt(
            output_root=output_root,
            role=role,
            seed=seed,
            policy=policy,
        )
        validate_normalized_seed(seed, policy=policy)
        copy_report = verify_no_shared_regular_inodes(source, seed)
        symlink_report = verify_internal_symlinks(seed)
        normalized_inventory = directory_inventory(seed, include_mode=False)
        normalization_inventory_delta = _normalization_inventory_delta(
            before=before,
            after=normalized_inventory,
            role=role,
            policy=policy,
            receipt=receipt,
        )
        _seal_tree(seed)
        _assert_read_only(seed)
        stage = _role_stage_payload(
            role=role,
            source=source,
            seed=seed,
            before=before,
            normalized_inventory=normalized_inventory,
            copy_report=copy_report,
            symlink_report=symlink_report,
            normalization_receipt=receipt,
            normalization_inventory_delta=normalization_inventory_delta,
        )
        _atomic_write_once(marker_path, _json_bytes(stage))
        return _verify_role_stage(output_root, role=role, source_required=False)
    _restore_interrupted_normalization_preimages(
        output_root=output_root,
        role=role,
        seed=seed,
        policy=policy,
    )
    copy_inventory_bound(source, seed, before)
    live_after = directory_inventory(source)
    copied = directory_inventory(seed)
    if live_after != before:
        raise EnvironmentCaptureError(f"{role} source drifted during capture")
    if copied != before:
        raise EnvironmentCaptureError(
            f"{role} captured bytes differ from the bound source inventory"
        )
    copy_report = verify_no_shared_regular_inodes(source, seed)
    symlink_report = verify_internal_symlinks(seed)
    _validate_ownership(
        seed,
        normalization=_policy_normalization(
            policy, SETUPTOOLS_NORMALIZATION_ID
        ),
        allow_known_conflict=True,
        source_record_normalizations=_record_normalizations_for_role(
            policy, role=role
        ),
        role=role,
    )
    receipt = _normalize_seed(
        output_root=output_root,
        role=role,
        source=source,
        seed=seed,
        policy=policy,
        ownership_policy_sha256=ownership_policy_sha256,
        integrity_normalization_policy_sha256=(
            integrity_normalization_policy_sha256
        ),
        incident_path=incident_path,
        incident_sha256=incident_sha256,
        recovered_record_path=recovered_record_path,
        recovered_record_sha256=recovered_record_sha256,
    )
    normalized_inventory = directory_inventory(seed, include_mode=False)
    normalization_inventory_delta = _normalization_inventory_delta(
        before=before,
        after=normalized_inventory,
        role=role,
        policy=policy,
        receipt=receipt,
    )
    _seal_tree(seed)
    _assert_read_only(seed)
    stage = _role_stage_payload(
        role=role,
        source=source,
        seed=seed,
        before=before,
        normalized_inventory=normalized_inventory,
        copy_report=copy_report,
        symlink_report=symlink_report,
        normalization_receipt=receipt,
        normalization_inventory_delta=normalization_inventory_delta,
    )
    _atomic_write_once(marker_path, _json_bytes(stage))
    return _verify_role_stage(output_root, role=role, source_required=False)


def _verify_role_stage(
    output_root: Path, *, role: str, source_required: bool = False
) -> dict[str, Any]:
    marker_path = output_root / ROLE_MARKERS[role]
    payload = _read_json(marker_path, description=f"{role} capture stage")
    _require_exact_fields(
        payload,
        {
            "schema_version",
            "release_id",
            "role",
            "source_prefix",
            "seed_prefix",
            "copy_method",
            "source_before_inventory_sha256",
            "source_after_inventory_sha256",
            "pre_normalization_copy_inventory_sha256",
            "normalized_content_inventory",
            "copy_audit",
            "symlink_audit",
            "normalization_receipt_id",
            "normalization_receipt_sha256",
            "normalization_inventory_delta",
            "sealed_read_only",
            "record_sha256",
        },
        description=f"{role} capture stage",
    )
    candidate = dict(payload)
    record_sha256 = candidate.pop("record_sha256", None)
    seed = _resolve_prefix(payload.get("seed_prefix", ""), description=f"{role} seed")
    delta = payload.get("normalization_inventory_delta")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("release_id") != RELEASE_ID
        or payload.get("role") != role
        or payload.get("copy_method") != "buffered_read_write_exclusive_create"
        or payload.get("sealed_read_only") is not True
        or payload.get("seed_prefix")
        != str(output_root / "seeds" / role)
        or not isinstance(payload.get("source_prefix"), str)
        or not Path(str(payload["source_prefix"])).is_absolute()
        or _SHA256_RE.fullmatch(
            str(payload.get("source_before_inventory_sha256", ""))
        )
        is None
        or _SHA256_RE.fullmatch(
            str(payload.get("source_after_inventory_sha256", ""))
        )
        is None
        or _SHA256_RE.fullmatch(
            str(payload.get("pre_normalization_copy_inventory_sha256", ""))
        )
        is None
        or _SHA256_RE.fullmatch(
            str(payload.get("normalization_receipt_id", ""))
        )
        is None
        or _SHA256_RE.fullmatch(
            str(payload.get("normalization_receipt_sha256", ""))
        )
        is None
        or not isinstance(delta, dict)
        or delta.get("protocol")
        != "pathwise-pre-post-full-inventory-diff-v1"
        or delta.get("exact_authorized_delta") is not True
        or delta.get("runtime_changed_paths") != []
        or record_sha256 != _sha256_bytes(_canonical_bytes(candidate))
        or stat.S_IMODE(marker_path.stat().st_mode) & 0o222
    ):
        raise EnvironmentCaptureError(f"invalid {role} capture stage")
    if source_required:
        _resolve_prefix(payload.get("source_prefix", ""), description=f"{role} source")
    _assert_read_only(seed)
    symlink_audit = verify_internal_symlinks(seed)
    if payload.get("symlink_audit") != symlink_audit:
        raise EnvironmentCaptureError(f"{role} symlink audit drifted")
    live_inventory = directory_inventory(seed, include_mode=False)
    if live_inventory != payload.get("normalized_content_inventory"):
        raise EnvironmentCaptureError(f"{role} sealed seed inventory drifted")
    return payload


def _capture_paths(
    *,
    output_root: str | Path,
    harness_source: str | Path,
    serving_source: str | Path,
) -> tuple[Path, dict[str, Path]]:
    root = _resolve_destination(output_root, description="capture output root")
    sources = {
        "harness": _resolve_prefix(harness_source, description="harness source"),
        "serving": _resolve_prefix(serving_source, description="serving source"),
    }
    _validate_nonoverlap({"output_root": root, **sources})
    return root, sources


def capture_environments(
    *,
    output_root: str | Path,
    harness_source: str | Path,
    serving_source: str | Path,
    ownership_policy: str | Path,
    integrity_normalization_policy: str | Path,
    reconciliation_incident: str | Path,
    recovered_setuptools_record: str | Path,
    apply: bool = False,
) -> dict[str, Any]:
    root, sources = _capture_paths(
        output_root=output_root,
        harness_source=harness_source,
        serving_source=serving_source,
    )
    ownership_policy_path = Path(ownership_policy).expanduser().resolve()
    ownership_policy_payload, ownership_policy_sha256 = _load_policy(
        ownership_policy_path
    )
    integrity_policy_path = (
        Path(integrity_normalization_policy).expanduser().resolve()
    )
    integrity_policy_payload, integrity_policy_sha256 = _load_integrity_policy(
        integrity_policy_path
    )
    policy = _combined_policy(
        ownership_policy_payload, integrity_policy_payload
    )
    recovered_path = Path(recovered_setuptools_record).expanduser().resolve()
    _recovered, recovered_sha256 = _validate_recovered_record(
        recovered_path,
        _policy_normalization(policy, SETUPTOOLS_NORMALIZATION_ID),
    )
    incident_path = Path(reconciliation_incident).expanduser().resolve()
    incident_sha256, incident_source_record_states = _validate_incident(
        incident_path,
        affected_prefixes=tuple(sources.values()),
        artifact_sha256=_policy_normalization(
            policy, SETUPTOOLS_NORMALIZATION_ID
        )["conda_artifact_sha256"],
    )
    setuptools_record_filename = _policy_normalization(
        policy, SETUPTOOLS_NORMALIZATION_ID
    )["conda_record_filename"]
    for role, source in sources.items():
        record_path = source / "conda-meta" / setuptools_record_filename
        observed_state = (
            "present"
            if record_path.is_file() and not record_path.is_symlink()
            else "absent"
        )
        if observed_state != incident_source_record_states[role]:
            raise EnvironmentCaptureError(
                "live source Setuptools record state differs from the sealed "
                f"reconciliation incident for {role}: "
                f"expected {incident_source_record_states[role]}, "
                f"observed {observed_state}"
            )
    marker_path = root / COMPLETE_MARKER
    if marker_path.is_file() and not marker_path.is_symlink():
        report = verify_capture(root)
        expected_sources = {role: str(path) for role, path in sources.items()}
        if (
            report["source_prefixes"] != expected_sources
            or report["ownership_policy_sha256"] != ownership_policy_sha256
            or report["integrity_normalization_policy_sha256"]
            != integrity_policy_sha256
            or report["reconciliation_incident_sha256"] != incident_sha256
            or report["recovered_record_sha256"] != recovered_sha256
        ):
            raise EnvironmentCaptureError(
                "completed environment capture belongs to different inputs"
            )
        return {**report, "status": "already_complete"}
    before = {role: directory_inventory(source) for role, source in sources.items()}
    source_distribution_reports: dict[str, dict[str, Any]] = {}
    source_distribution_audits = {}
    for role, source in sources.items():
        distributions, _conda = _validate_ownership(
            source,
            normalization=_policy_normalization(
                policy, SETUPTOOLS_NORMALIZATION_ID
            ),
            allow_known_conflict=True,
            source_record_normalizations=_record_normalizations_for_role(
                policy, role=role
            ),
            role=role,
        )
        if distributions["projected_shared_record_path_count"] != 0:
            raise EnvironmentCaptureError(
                f"{role} source RECORD repairs project "
                f"{distributions['projected_shared_record_path_count']} "
                "shared paths"
            )
        source_distribution_reports[role] = distributions
        audit_path = root / SOURCE_DISTRIBUTION_AUDIT_FILES[role]
        source_distribution_audits[role] = {
            "path": str(audit_path),
            "sha256": _sha256_bytes(_json_bytes(distributions)),
            **{
                key: distributions[key]
                for key in (
                    "inventory_sha256",
                    "distribution_count",
                    "record_count",
                    "hashed_file_count",
                    "unhashed_record_count",
                    "runtime_cache_record_count",
                    "missing_unhashed_runtime_cache_record_count",
                    "policy_normalized_record_count",
                    "shared_record_path_count",
                    "projected_shared_record_path_count",
                )
            },
        }
    archived_ownership_policy_path = (
        root / "evidence" / ownership_policy_path.name
    )
    archived_integrity_policy_path = (
        root / "evidence" / integrity_policy_path.name
    )
    archived_incident_path = root / "evidence" / "CONDA_RECONCILIATION_INCIDENT.json"
    archived_recovered_path = (
        root / "evidence" / "setuptools-82.0.1-pyh332efcf_0.json"
    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "source_prefixes": {role: str(path) for role, path in sources.items()},
        "seed_prefixes": {
            role: str(root / "seeds" / role) for role in ROLES
        },
        "ownership_policy_path": str(archived_ownership_policy_path),
        "ownership_policy_sha256": ownership_policy_sha256,
        "integrity_normalization_policy_path": str(
            archived_integrity_policy_path
        ),
        "integrity_normalization_policy_sha256": integrity_policy_sha256,
        "reconciliation_incident_path": str(archived_incident_path),
        "reconciliation_incident_sha256": incident_sha256,
        "recovered_record_path": str(archived_recovered_path),
        "recovered_record_sha256": recovered_sha256,
        "input_evidence_paths": {
            "ownership_policy": str(ownership_policy_path),
            "integrity_normalization_policy": str(integrity_policy_path),
            "reconciliation_incident": str(incident_path),
            "recovered_setuptools_record": str(recovered_path),
        },
        "copy_contract": {
            "method": "buffered_read_write_exclusive_create",
            "hardlinks": False,
            "reflinks": False,
            "shared_regular_inode_count": 0,
            "source_before_after_equal": True,
        },
        "source_inventories": {
            role: {
                "path": str(root / SOURCE_INVENTORY_FILES[role]),
                "sha256": _sha256_bytes(_json_bytes(inventory)),
                **{
                    key: value
                    for key, value in inventory.items()
                    if key != "entries"
                },
            }
            for role, inventory in before.items()
        },
        "source_distribution_audits": source_distribution_audits,
    }
    if not apply:
        return {**plan, "status": "dry_run"}
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_once(
        archived_ownership_policy_path, ownership_policy_path.read_bytes()
    )
    _atomic_write_once(
        archived_ownership_policy_path.with_suffix(".sha256"),
        (
            f"{ownership_policy_sha256}  "
            f"{archived_ownership_policy_path.name}\n"
        ).encode("utf-8"),
    )
    _atomic_write_once(
        archived_integrity_policy_path, integrity_policy_path.read_bytes()
    )
    _atomic_write_once(
        archived_integrity_policy_path.with_suffix(".sha256"),
        (
            f"{integrity_policy_sha256}  "
            f"{archived_integrity_policy_path.name}\n"
        ).encode("utf-8"),
    )
    _atomic_write_once(archived_incident_path, incident_path.read_bytes())
    _atomic_write_once(archived_recovered_path, recovered_path.read_bytes())
    for role in ROLES:
        _atomic_write_once(
            root / SOURCE_INVENTORY_FILES[role],
            _json_bytes(before[role]),
        )
        _atomic_write_once(
            root / SOURCE_DISTRIBUTION_AUDIT_FILES[role],
            _json_bytes(source_distribution_reports[role]),
        )
    ownership_policy_path = archived_ownership_policy_path
    integrity_policy_path = archived_integrity_policy_path
    incident_path = archived_incident_path
    recovered_path = archived_recovered_path
    intent_payload = {**plan, "publication_protocol": "intent_first_stages_marker_last"}
    intent_payload["intent_id"] = _sha256_bytes(_canonical_bytes(intent_payload))
    _atomic_write_once(root / INTENT_MARKER, _json_bytes(intent_payload))
    stages = {
        role: _capture_role(
            output_root=root,
            role=role,
            source=sources[role],
            before=before[role],
            policy=policy,
            ownership_policy_sha256=ownership_policy_sha256,
            integrity_normalization_policy_sha256=integrity_policy_sha256,
            incident_path=incident_path,
            incident_sha256=incident_sha256,
            recovered_record_path=recovered_path,
            recovered_record_sha256=recovered_sha256,
        )
        for role in ROLES
    }
    marker: dict[str, Any] = {
        **intent_payload,
        "complete": True,
        "stage_records": {
            role: {
                "filename": ROLE_MARKERS[role],
                "sha256": _sha256_file(root / ROLE_MARKERS[role]),
                "record_sha256": stages[role]["record_sha256"],
                "normalization_receipt_id": stages[role][
                    "normalization_receipt_id"
                ],
                "normalized_content_inventory_sha256": stages[role][
                    "normalized_content_inventory"
                ]["inventory_sha256"],
            }
            for role in ROLES
        },
    }
    marker["capture_id"] = _sha256_bytes(_canonical_bytes(marker))
    _atomic_write_once(marker_path, _json_bytes(marker))
    report = verify_capture(root)
    return {**report, "status": "created"}


def verify_capture(output_root: str | Path) -> dict[str, Any]:
    root = _resolve_destination(output_root, description="capture output root")
    if root.is_symlink() or not root.is_dir():
        raise EnvironmentCaptureError(f"missing capture output root: {root}")
    marker_path = root / COMPLETE_MARKER
    marker = _read_json(marker_path, description="environment capture marker")
    capture_plan_fields = {
        "schema_version",
        "release_id",
        "source_prefixes",
        "seed_prefixes",
        "ownership_policy_path",
        "ownership_policy_sha256",
        "integrity_normalization_policy_path",
        "integrity_normalization_policy_sha256",
        "reconciliation_incident_path",
        "reconciliation_incident_sha256",
        "recovered_record_path",
        "recovered_record_sha256",
        "input_evidence_paths",
        "copy_contract",
        "source_inventories",
        "source_distribution_audits",
        "publication_protocol",
        "intent_id",
    }
    _require_exact_fields(
        marker,
        capture_plan_fields | {"complete", "stage_records", "capture_id"},
        description="environment capture marker",
    )
    candidate = dict(marker)
    capture_id = candidate.pop("capture_id", None)
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("release_id") != RELEASE_ID
        or marker.get("complete") is not True
        or marker.get("publication_protocol")
        != "intent_first_stages_marker_last"
        or capture_id != _sha256_bytes(_canonical_bytes(candidate))
        or stat.S_IMODE(marker_path.stat().st_mode) & 0o222
    ):
        raise EnvironmentCaptureError("environment capture marker is invalid")
    intent_path = root / INTENT_MARKER
    intent = _read_json(
        intent_path, description="marker-first environment capture intent"
    )
    _require_exact_fields(
        intent, capture_plan_fields, description="environment capture intent"
    )
    intent_candidate = dict(intent)
    intent_id = intent_candidate.pop("intent_id", None)
    marker_plan = {
        key: value
        for key, value in marker.items()
        if key not in {"complete", "stage_records", "capture_id"}
    }
    if (
        intent_id != _sha256_bytes(_canonical_bytes(intent_candidate))
        or marker.get("intent_id") != intent_id
        or marker_plan != intent
        or stat.S_IMODE(intent_path.stat().st_mode) & 0o222
    ):
        raise EnvironmentCaptureError(
            "environment capture intent/marker binding drifted"
        )
    source_prefixes = marker.get("source_prefixes")
    seed_prefixes = marker.get("seed_prefixes")
    input_evidence_paths = marker.get("input_evidence_paths")
    if (
        not isinstance(source_prefixes, dict)
        or set(source_prefixes) != set(ROLES)
        or not isinstance(seed_prefixes, dict)
        or set(seed_prefixes) != set(ROLES)
        or not isinstance(input_evidence_paths, dict)
        or set(input_evidence_paths)
        != {
            "ownership_policy",
            "integrity_normalization_policy",
            "reconciliation_incident",
            "recovered_setuptools_record",
        }
        or any(
            not isinstance(source_prefixes.get(role), str)
            or not Path(source_prefixes[role]).is_absolute()
            for role in ROLES
        )
        or seed_prefixes
        != {
            role: str(root / "seeds" / role)
            for role in ROLES
        }
        or any(
            not isinstance(value, str) or not Path(value).is_absolute()
            for value in input_evidence_paths.values()
        )
        or marker.get("copy_contract")
        != {
            "method": "buffered_read_write_exclusive_create",
            "hardlinks": False,
            "reflinks": False,
            "shared_regular_inode_count": 0,
            "source_before_after_equal": True,
        }
    ):
        raise EnvironmentCaptureError(
            "environment capture exact input identity drifted"
        )
    ownership_policy_path = Path(
        str(marker.get("ownership_policy_path", ""))
    )
    integrity_policy_path = Path(
        str(marker.get("integrity_normalization_policy_path", ""))
    )
    if (
        ownership_policy_path.parent != root / "evidence"
        or ownership_policy_path.name in {"", ".", ".."}
        or stat.S_IMODE(ownership_policy_path.stat().st_mode) & 0o222
        or stat.S_IMODE(
            ownership_policy_path.with_suffix(".sha256").stat().st_mode
        )
        & 0o222
        or integrity_policy_path.parent != root / "evidence"
        or integrity_policy_path.name in {"", ".", ".."}
        or stat.S_IMODE(integrity_policy_path.stat().st_mode) & 0o222
        or stat.S_IMODE(
            integrity_policy_path.with_suffix(".sha256").stat().st_mode
        )
        & 0o222
    ):
        raise EnvironmentCaptureError(
            "captured ownership-policy path or mode drifted"
        )
    ownership_policy, ownership_policy_sha256 = _load_policy(
        ownership_policy_path
    )
    integrity_policy, integrity_policy_sha256 = _load_integrity_policy(
        integrity_policy_path
    )
    if ownership_policy_sha256 != marker.get("ownership_policy_sha256"):
        raise EnvironmentCaptureError("captured ownership policy drifted")
    if integrity_policy_sha256 != marker.get(
        "integrity_normalization_policy_sha256"
    ):
        raise EnvironmentCaptureError(
            "captured integrity-normalization policy drifted"
        )
    policy = _combined_policy(ownership_policy, integrity_policy)
    normalization = _policy_normalization(
        policy, SETUPTOOLS_NORMALIZATION_ID
    )
    incident_path = root / "evidence" / "CONDA_RECONCILIATION_INCIDENT.json"
    recovered_path = (
        root / "evidence" / "setuptools-82.0.1-pyh332efcf_0.json"
    )
    if (
        marker.get("reconciliation_incident_path") != str(incident_path)
        or marker.get("recovered_record_path") != str(recovered_path)
        or recovered_path.is_symlink()
        or not recovered_path.is_file()
        or stat.S_IMODE(recovered_path.stat().st_mode) & 0o222
    ):
        raise EnvironmentCaptureError(
            "captured reconciliation evidence path drifted"
        )
    incident_sha256, incident_source_record_states = _validate_incident(
        incident_path,
        affected_prefixes=tuple(
            Path(source_prefixes[role]) for role in ROLES
        ),
        artifact_sha256=normalization["conda_artifact_sha256"],
    )
    _recovered_record, recovered_sha256 = _validate_recovered_record(
        recovered_path, normalization
    )
    if (
        incident_sha256 != marker.get("reconciliation_incident_sha256")
        or recovered_sha256 != marker.get("recovered_record_sha256")
    ):
        raise EnvironmentCaptureError(
            "captured reconciliation evidence bytes drifted"
        )
    source_inventory_records = marker.get("source_inventories")
    source_distribution_records = marker.get(
        "source_distribution_audits"
    )
    if (
        not isinstance(source_inventory_records, dict)
        or set(source_inventory_records) != set(ROLES)
        or not isinstance(source_distribution_records, dict)
        or set(source_distribution_records) != set(ROLES)
    ):
        raise EnvironmentCaptureError(
            "capture marker has the wrong source evidence inventory"
        )
    stage_records = marker.get("stage_records")
    if not isinstance(stage_records, dict) or set(stage_records) != set(ROLES):
        raise EnvironmentCaptureError("capture marker has the wrong stage inventory")
    stages: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        stage_path = root / ROLE_MARKERS[role]
        record = stage_records[role]
        if isinstance(record, dict):
            _require_exact_fields(
                record,
                {
                    "filename",
                    "sha256",
                    "record_sha256",
                    "normalization_receipt_id",
                    "normalized_content_inventory_sha256",
                },
                description=f"{role} capture stage binding",
            )
        if (
            not isinstance(record, dict)
            or record.get("filename") != ROLE_MARKERS[role]
            or record.get("sha256") != _sha256_file(stage_path)
        ):
            raise EnvironmentCaptureError(f"{role} capture record drifted")
        stage = _verify_role_stage(root, role=role)
        receipt_path = _normalization_paths(root, role)["receipt"]
        source_inventory_path = root / SOURCE_INVENTORY_FILES[role]
        source_inventory = _read_json(
            source_inventory_path,
            description=f"{role} archived source inventory",
        )
        source_inventory_record = source_inventory_records[role]
        source_summary_fields = {
            "inventory_sha256",
            "entry_count",
            "file_count",
            "directory_count",
            "symlink_count",
            "total_file_bytes",
        }
        if not isinstance(source_inventory_record, dict):
            raise EnvironmentCaptureError(
                f"{role} source inventory binding is invalid"
            )
        _require_exact_fields(
            source_inventory_record,
            source_summary_fields | {"path", "sha256"},
            description=f"{role} source inventory binding",
        )
        _validated_content_inventory(
            source_inventory, description=f"{role} archived source"
        )
        if stat.S_IMODE(source_inventory_path.stat().st_mode) & 0o222:
            raise EnvironmentCaptureError(
                f"{role} archived source inventory is writable"
            )
        expected_source_summary = {
            key: source_inventory[key] for key in source_summary_fields
        }
        if (
            source_inventory_record.get("path")
            != str(source_inventory_path)
            or source_inventory_record.get("sha256")
            != _sha256_file(source_inventory_path)
            or {
                key: source_inventory_record[key]
                for key in source_summary_fields
            }
            != expected_source_summary
        ):
            raise EnvironmentCaptureError(
                f"{role} archived source inventory binding drifted"
        )
        source_state = incident_source_record_states[role]
        if source_state == "absent":
            record_sha256 = recovered_sha256
        else:
            record_relative = (
                "conda-meta/"
                f"{normalization['conda_record_filename']}"
            )
            record_entries = [
                row
                for row in source_inventory["entries"]
                if row.get("path") == record_relative
                and row.get("type") == "file"
            ]
            if len(record_entries) != 1:
                raise EnvironmentCaptureError(
                    f"{role} source inventory does not contain the incident-"
                    "bound Setuptools record"
                )
            record_sha256 = record_entries[0]["sha256"]
        receipt = _verify_normalization_receipt(
            output_root=root,
            role=role,
            seed=Path(stage["seed_prefix"]),
            policy=policy,
            expected_binding={
                "source_prefix": source_prefixes[role],
                "ownership_policy_sha256": ownership_policy_sha256,
                "integrity_normalization_policy_sha256": (
                    integrity_policy_sha256
                ),
                "incident_path": str(incident_path),
                "incident_sha256": incident_sha256,
                "source_record_state": source_state,
                "record_sha256": record_sha256,
            },
        )
        inventory_delta = _normalization_inventory_delta(
            before=source_inventory,
            after=stage["normalized_content_inventory"],
            role=role,
            policy=policy,
            receipt=receipt,
        )
        if (
            stage["record_sha256"] != record.get("record_sha256")
            or stage["normalization_receipt_id"]
            != record.get("normalization_receipt_id")
            or stage["normalization_receipt_id"] != receipt["receipt_id"]
            or stage["normalization_receipt_sha256"]
            != _sha256_file(receipt_path)
            or stage["normalized_content_inventory"]["inventory_sha256"]
            != record.get("normalized_content_inventory_sha256")
            or stage["source_before_inventory_sha256"]
            != source_inventory["inventory_sha256"]
            or stage["source_after_inventory_sha256"]
            != source_inventory["inventory_sha256"]
            or stage["pre_normalization_copy_inventory_sha256"]
            != source_inventory["inventory_sha256"]
            or stage["normalization_inventory_delta"] != inventory_delta
            or stage["source_prefix"] != source_prefixes[role]
            or stage["seed_prefix"] != seed_prefixes[role]
            or stage["copy_audit"]
            != {
                "source_regular_file_count": source_inventory[
                    "file_count"
                ],
                "destination_regular_file_count": source_inventory[
                    "file_count"
                ],
                "shared_regular_inode_count": 0,
            }
        ):
            raise EnvironmentCaptureError(f"{role} capture identity drifted")
        seed = Path(stage["seed_prefix"])
        normalized = validate_normalized_seed(seed, policy=policy)
        if (
            normalized["distribution_inventory"]["inventory_sha256"]
            != receipt["normalized_identity"]["distribution_inventory_sha256"]
        ):
            raise EnvironmentCaptureError(
                f"{role} normalization receipt no longer matches the seed"
            )
        source_distribution_path = (
            root / SOURCE_DISTRIBUTION_AUDIT_FILES[role]
        )
        source_distribution_record = source_distribution_records[role]
        distribution_summary_fields = {
            "inventory_sha256",
            "distribution_count",
            "record_count",
            "hashed_file_count",
            "unhashed_record_count",
            "runtime_cache_record_count",
            "missing_unhashed_runtime_cache_record_count",
            "policy_normalized_record_count",
            "shared_record_path_count",
            "projected_shared_record_path_count",
        }
        if not isinstance(source_distribution_record, dict):
            raise EnvironmentCaptureError(
                f"{role} source distribution binding is invalid"
            )
        _require_exact_fields(
            source_distribution_record,
            distribution_summary_fields | {"path", "sha256"},
            description=f"{role} source distribution binding",
        )
        source_distribution = _read_json(
            source_distribution_path,
            description=f"{role} source distribution audit",
        )
        if (
            stat.S_IMODE(source_distribution_path.stat().st_mode) & 0o222
            or source_distribution_record.get("path")
            != str(source_distribution_path)
            or source_distribution_record.get("sha256")
            != _sha256_file(source_distribution_path)
            or any(
                source_distribution_record.get(key)
                != source_distribution.get(key)
                for key in distribution_summary_fields
            )
        ):
            raise EnvironmentCaptureError(
                f"{role} source distribution evidence drifted"
            )
        reconstructed_source_distribution = (
            _reconstruct_source_distribution_inventory(
                output_root=root,
                role=role,
                seed=seed,
                policy=policy,
                receipt=receipt,
                normalized_report=normalized["distribution_inventory"],
            )
        )
        if source_distribution != reconstructed_source_distribution:
            raise EnvironmentCaptureError(
                f"{role} source distribution audit is not reconstructible "
                "from sealed evidence"
            )
        stages[role] = stage
    return {
        "status": "verified",
        "release_id": RELEASE_ID,
        "capture_id": capture_id,
        "capture_marker_sha256": _sha256_file(marker_path),
        "source_prefixes": dict(marker["source_prefixes"]),
        "seed_prefixes": dict(marker["seed_prefixes"]),
        "ownership_policy_path": str(ownership_policy_path),
        "ownership_policy_sha256": ownership_policy_sha256,
        "integrity_normalization_policy_path": str(integrity_policy_path),
        "integrity_normalization_policy_sha256": integrity_policy_sha256,
        "reconciliation_incident_path": marker["reconciliation_incident_path"],
        "reconciliation_incident_sha256": marker[
            "reconciliation_incident_sha256"
        ],
        "recovered_record_path": marker["recovered_record_path"],
        "recovered_record_sha256": marker["recovered_record_sha256"],
        "stage_records": dict(stage_records),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser(
        "capture", help="dry-run or capture immutable normalized seeds"
    )
    capture.add_argument("--output-root", required=True, type=Path)
    capture.add_argument("--harness-source", required=True, type=Path)
    capture.add_argument("--serving-source", required=True, type=Path)
    capture.add_argument(
        "--ownership-policy",
        type=Path,
        default=REPO / "configs" / POLICY_FILENAME,
    )
    capture.add_argument(
        "--integrity-normalization-policy",
        type=Path,
        default=REPO / "configs" / INTEGRITY_POLICY_FILENAME,
    )
    capture.add_argument("--reconciliation-incident", required=True, type=Path)
    capture.add_argument("--recovered-setuptools-record", required=True, type=Path)
    capture.add_argument("--apply", action="store_true")
    verify = subparsers.add_parser("verify", help="verify sealed captured seeds")
    verify.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            report = verify_capture(args.output_root)
        else:
            report = capture_environments(
                output_root=args.output_root,
                harness_source=args.harness_source,
                serving_source=args.serving_source,
                ownership_policy=args.ownership_policy,
                integrity_normalization_policy=(
                    args.integrity_normalization_policy
                ),
                reconciliation_incident=args.reconciliation_incident,
                recovered_setuptools_record=args.recovered_setuptools_record,
                apply=args.apply,
            )
    except (OSError, UnicodeError, ValueError, EnvironmentCaptureError) as exc:
        print(f"[schema5-environment-capture] ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
