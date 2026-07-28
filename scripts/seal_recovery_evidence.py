#!/usr/bin/env python3
"""Seal failed recovery artifacts without deleting or rewriting their contents.

The two operations in this module are intentionally narrow:

``seal-quarantine``
    Inventories a same-filesystem quarantined partial release, removes every write
    bit, verifies the content again, and publishes a marker outside the tree last.

``record-failure``
    Binds scheduler states and the already-sealed recovery artifacts into an
    immutable failure envelope.  It never infers that a failed deterministic stage
    is retryable.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import pwd
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


_REPOSITORY = Path(__file__).resolve().parent.parent
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from scripts import provision_schema5_conda_toolchain as conda_toolchain  # noqa: E402


class EvidenceError(RuntimeError):
    """Raised when forensic evidence cannot be sealed unambiguously."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
_CANARY_COMMENT_RE = re.compile(
    r"(?:"
    r"asys-s5-fleet:pool=[^;=\r\n]+;profile=[^;=\r\n]+;"
    r"replica=[^;=\r\n]+;generation=[0-9]+;"
    r"intent=[0-9a-f]{32};fleet=[0-9a-f]{64}"
    r"|asys:s5-dependency-canary:[0-9a-f]{32}:(?:root|child|sentinel)"
    r")\Z"
)
_CANARY_JOB_NAME_RE = re.compile(r"asys-s5-serve-[A-Za-z0-9_.-]{1,96}\Z")
_CANARY_UNIQUE_JOB_NAME_RE = re.compile(
    r"asys-s5-serve-(?:canary|turnover)-[0-9a-f]{12}\Z"
)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_ERROR_CLASSIFICATION_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,127}\Z")
_CANARY_SEAL_PROTOCOL = "schema5-v1.2-r2-partial-canary-failure-seal-v1"
_CANARY_SEAL_INTENT_PROTOCOL = (
    "schema5-v1.2-r2-partial-canary-failure-seal-intent-v1"
)
_SCHEDULER_EVIDENCE_PROTOCOL = (
    "schema5-v1.2-r2-partial-canary-scheduler-evidence-v1"
)
_R3_RELEASE_ID = "sweep-recovery-schema5-v1.2"
_R3_RELEASE_TAG = "sweep-recovery-schema5-v1.2-r3"
_R3_CHAIN_NAMESPACE = "schema5-v1.2-r3"
_R3_RELEASE_COMMIT = "acd723ba9a99d88e77f7d752268bc31205c3a808"
_TRUSTED_SYSTEM_PATH = "/usr/bin:/bin"
_UNTRUSTED_PROCESS_ENVIRONMENT_KEYS = frozenset(
    {
        "BASH_ENV",
        "CDPATH",
        "ENV",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_EXEC_PATH",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_TEMPLATE_DIR",
        "GIT_WORK_TREE",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "SLURM_CLUSTERS",
        "SLURM_CONF",
        "SLURM_TIME_FORMAT",
    }
)
_UNTRUSTED_PROCESS_ENVIRONMENT_PREFIXES = (
    "BASH_FUNC_",
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
    "SACCT_",
    "SBATCH_",
    "SCONTROL_",
    "SQUEUE_",
)
_R3_RELEASE_TAG_OBJECT = "fa87b283974afc1fa48fcb22b61e6ae7eec4bb36"
_R3_DURABLE_RELEASE_PROTOCOL = "schema5-v1.2-r3-durable-git-release-v1"
_R3_DURABLE_RELEASE_MARKER = (
    "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R3_COMPLETE.json"
)
_R3_DURABLE_BUNDLE = f"{_R3_RELEASE_TAG}.bundle"
_R3_PRELAUNCH_FAILURE_ENVELOPE_PROTOCOL = (
    "schema5-v1.2-r3-prelaunch-failure-evidence-v1"
)
_R3_PRELAUNCH_FAILURE_INTENT_PROTOCOL = (
    "schema5-v1.2-r3-prelaunch-failure-seal-intent-v1"
)
_R3_PRELAUNCH_FAILURE_SEAL_PROTOCOL = (
    "schema5-v1.2-r3-prelaunch-failure-seal-v1"
)
_R3_PRELAUNCH_FAILURE_BINDING_PROTOCOL = (
    "schema5-v1.2-r3-prelaunch-failure-seal-binding-v1"
)
R3_PRELAUNCH_FAILURE_RELATIVE_ROOT = Path(
    "prelaunch_failures/schema5-v1.2-r3"
)
_R3_PRELAUNCH_FAILURE_CLASSIFICATIONS = (
    "unsafe_recorded_broken_internal_symlink",
    "offline_clone_unseeded_release_local_cache",
)
_R3_BROKEN_SYMLINK_PATH = (
    "libexec/gcc/x86_64-conda-linux-gnu/15.2.0/TOOLS=addr2line"
)
_R3_BROKEN_SYMLINK_TARGET = (
    "../../../../bin/x86_64-conda-linux-gnu-TOOLS=addr2line"
)
_R3_RELEASE_CHECKOUT_DIRECTORY = "materialization_pilot_source_checkout_v1_2_r3"
_R10_RELEASE_CHECKOUT_DIRECTORY = "materialization_pilot_source_checkout_v1_2_r10"
_R10_TOOLCHAIN_RELATIVE_ROOT = Path(
    "toolchains/r10/conda"
)
_R3_PILOT_RELATIVE_PATH = Path("scripts/run_schema5_materialization_pilot.py")
_R3_PILOT_SHA256 = (
    "85e198c5b48070fbdd930bca639991677d8007034b0bd609e3c421ff3a8d863a"
)
_R3_RUNTIME_IDENTITY_RELATIVE_PATH = Path(
    "scripts/schema5_conda_runtime_identity.py"
)
_R3_RUNTIME_IDENTITY_SHA256 = (
    "b3a66668b09e2aa5ba67c00878a713ced94b94cb272106b217b4ce84d2c2c64d"
)
_R10_SEALER_RELATIVE_PATH = Path("scripts/seal_recovery_evidence.py")
_R10_TOOLCHAIN_PROVISIONER_RELATIVE_PATH = Path(
    "scripts/provision_schema5_conda_toolchain.py"
)
_R3_SHARED_CONDA_BASE = Path(
    "/orcd/data/lhtsai/001/om2/mabdel03/miniforge3"
)
_R3_SHARED_CONDA_EXECUTABLE = _R3_SHARED_CONDA_BASE / "bin/conda"
_R3_BROKEN_SYMLINK_MISSING_TARGET = (
    _R3_SHARED_CONDA_BASE
    / "bin"
    / "x86_64-conda-linux-gnu-TOOLS=addr2line"
)
_R3_SHARED_RUNTIME_EXCLUDED_TOP_LEVEL = frozenset(
    {".conda", "conda-bld", "envs", "pkgs"}
)
_R3_PROBE_INPUT_NAMES = {
    "unsafe_recorded_broken_internal_symlink": (
        "tagged-r3-release-checkout",
        "tagged-r10-release-checkout",
        "sealed-r10-conda-toolchain",
        "recorded-shared-conda-base",
    ),
    "offline_clone_unseeded_release_local_cache": (
        "tagged-r3-release-checkout",
        "tagged-r10-release-checkout",
        "sealed-r10-conda-toolchain",
    ),
}
_R3_PROBE_ENVELOPE_FILENAMES = {
    "unsafe_recorded_broken_internal_symlink": (
        "FAILURE_UNSAFE_RECORDED_BROKEN_INTERNAL_SYMLINK.source.json"
    ),
    "offline_clone_unseeded_release_local_cache": (
        "FAILURE_OFFLINE_CLONE_UNSEEDED_RELEASE_LOCAL_CACHE.source.json"
    ),
}
_R3_PROBE_COMMON_ENVIRONMENT_KEYS = frozenset(
    {
        "HOME",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    }
)
_R3_PROBE_OFFLINE_ENVIRONMENT_KEYS = (
    _R3_PROBE_COMMON_ENVIRONMENT_KEYS
    | {
        "CONDA_ENVS_PATH",
        "CONDA_NO_PLUGINS",
        "CONDA_OFFLINE",
        "CONDA_PIP_INTEROP_ENABLED",
        "CONDA_PKGS_DIRS",
    }
)


def _sanitized_process_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in _UNTRUSTED_PROCESS_ENVIRONMENT_KEYS
        and not key.startswith(_UNTRUSTED_PROCESS_ENVIRONMENT_PREFIXES)
    }
    environment.update(
        {
            "PATH": _TRUSTED_SYSTEM_PATH,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return environment


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any], *, mode: int) -> None:
    _atomic_bytes(path, _canonical_json(payload), mode=mode)


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{description} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{description} must be a JSON object: {path}")
    return value


def _absolute(path: str | Path, *, description: str) -> Path:
    lexical = Path(path)
    if not lexical.is_absolute() or str(lexical) != os.path.normpath(str(lexical)):
        raise EvidenceError(f"{description} must be a canonical absolute path: {path}")
    return lexical


def _safe_existing_path(
    path: str | Path, *, description: str, kind: str
) -> Path:
    """Require one lexical/physical absolute path with no symlink component."""

    value = _absolute(path, description=description)
    try:
        resolved = value.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceError(f"{description} is missing or unsafe: {value}: {exc}") from exc
    if resolved != value:
        raise EvidenceError(f"{description} contains a symlink: {value}")
    metadata = os.lstat(value)
    matches = {
        "file": stat.S_ISREG(metadata.st_mode),
        "directory": stat.S_ISDIR(metadata.st_mode),
    }
    if kind not in matches or not matches[kind]:
        raise EvidenceError(f"{description} is not a safe {kind}: {value}")
    return value


def _safe_output_directory(path: str | Path, *, description: str) -> Path:
    """Validate a possibly absent output directory without resolving it through links."""

    value = _absolute(path, description=description)
    if value.exists() or value.is_symlink():
        return _safe_existing_path(value, description=description, kind="directory")
    parent = _safe_existing_path(
        value.parent, description=f"{description} parent", kind="directory"
    )
    if value.parent != parent:
        raise EvidenceError(f"{description} parent drifted")
    return value


def _stable_file_record(path: Path, *, description: str) -> dict[str, Any]:
    """Hash a regular, singly-linked file without following a final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot open {description} {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise EvidenceError(f"{description} is not a singly-linked regular file: {path}")
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise EvidenceError(f"{description} changed while being hashed: {path}")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size": before.st_size,
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": stat.S_IMODE(before.st_mode),
    }


def _full_tree_inventory(root: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Return a stable, symlink-free inventory including the root directory."""

    root = _safe_existing_path(root, description="partial canary root", kind="directory")
    rows: list[dict[str, Any]] = []
    file_count = 0
    total_bytes = 0
    candidates = [root, *sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix())]
    for candidate in candidates:
        relative = "." if candidate == root else candidate.relative_to(root).as_posix()
        metadata = os.lstat(candidate)
        common = {
            "relative_path": relative,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
            "nlink": metadata.st_nlink,
        }
        if stat.S_ISLNK(metadata.st_mode):
            raise EvidenceError(f"partial canary tree contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            rows.append(common | {"type": "directory", "size": 0, "sha256": None})
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise EvidenceError(
                f"partial canary tree contains an unsafe entry: {relative}"
            )
        record = _stable_file_record(candidate, description="partial canary artifact")
        rows.append(
            common
            | {
                "type": "file",
                "size": record["size"],
                "sha256": record["sha256"],
            }
        )
        file_count += 1
        total_bytes += int(record["size"])
    return rows, file_count, total_bytes


def _inventory_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(dict(row)) for row in rows)


def _inventory_content_identity(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "mode"}
        for row in rows
    ]


def _validate_partially_sealed_inventory(
    current: Sequence[Mapping[str, Any]],
    before: Sequence[Mapping[str, Any]],
) -> None:
    if _inventory_content_identity(current) != _inventory_content_identity(before):
        raise EvidenceError("partial canary content identity drifted after seal intent")
    for current_row, before_row in zip(current, before):
        old_mode = before_row.get("mode")
        new_mode = current_row.get("mode")
        if (
            not isinstance(old_mode, int)
            or isinstance(old_mode, bool)
            or new_mode not in {old_mode, old_mode & ~0o222}
        ):
            raise EvidenceError(
                "partial canary mode changed outside the read-only sealing transition"
            )


def _strict_json_artifact(path: Path, *, description: str) -> dict[str, Any]:
    _safe_existing_path(path, description=description, kind="file")
    return _read_json(path, description=description)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o640)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise EvidenceError(f"unsafe evidence lock: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvidenceError(f"another evidence sealer holds {path}") from exc
        yield
    finally:
        os.close(descriptor)


def _inventory_rows(root: Path, *, include_modes: bool) -> tuple[list[str], int, int]:
    if root.is_symlink() or not root.is_dir():
        raise EvidenceError(f"quarantine root is missing or unsafe: {root}")
    rows: list[str] = []
    file_count = 0
    total_bytes = 0
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix()
        metadata = os.lstat(candidate)
        mode = f"{stat.S_IMODE(metadata.st_mode):04o}|" if include_modes else ""
        if stat.S_ISREG(metadata.st_mode):
            digest = _sha256_file(candidate)
            rows.append(f"f|{mode}{metadata.st_size}|{digest}|{relative}")
            file_count += 1
            total_bytes += metadata.st_size
        elif stat.S_ISDIR(metadata.st_mode):
            rows.append(f"d|{mode}0|-|{relative}")
        elif stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(candidate)
            if "\n" in target or "\r" in target:
                raise EvidenceError(f"newline in symlink target: {relative}")
            rows.append(f"l|{mode}{len(target.encode('utf-8'))}|{_sha256_bytes(target.encode())}|{relative}|{target}")
        else:
            raise EvidenceError(f"unsupported quarantine entry type: {relative}")
    return rows, file_count, total_bytes


def _inventory_bytes(rows: Iterable[str]) -> bytes:
    return ("\n".join(rows) + "\n").encode("utf-8")


def _content_identity(root: Path) -> tuple[str, int, int]:
    rows, file_count, total_bytes = _inventory_rows(root, include_modes=False)
    return _sha256_bytes(_inventory_bytes(rows)), file_count, total_bytes


def _remove_write_bits(root: Path) -> None:
    entries = sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True)
    for candidate in entries:
        metadata = os.lstat(candidate)
        if stat.S_ISLNK(metadata.st_mode):
            continue
        os.chmod(candidate, stat.S_IMODE(metadata.st_mode) & ~0o222)
    os.chmod(root, stat.S_IMODE(os.lstat(root).st_mode) & ~0o222)
    _fsync_directory(root.parent)


def _writable_entries(root: Path) -> list[str]:
    values: list[str] = []
    for candidate in (root, *root.rglob("*")):
        metadata = os.lstat(candidate)
        if not stat.S_ISLNK(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) & 0o222:
            values.append("." if candidate == root else candidate.relative_to(root).as_posix())
    return sorted(values)


def seal_quarantine(
    *,
    tree: str | Path,
    evidence_root: str | Path,
    release_id: str,
    failed_job_id: str,
    quarantine_completion: str | Path,
    apply: bool = False,
) -> dict[str, Any]:
    tree_path = _absolute(tree, description="quarantine tree")
    evidence = _absolute(evidence_root, description="quarantine evidence root")
    completion_path = _absolute(
        quarantine_completion, description="quarantine completion"
    )
    if not failed_job_id.isdigit() or not release_id:
        raise EvidenceError("invalid failed job or release identity")
    quarantine = _read_json(completion_path, description="quarantine completion")
    if (
        quarantine.get("passed") is not True
        or str(quarantine.get("destination")) != str(tree_path)
        or str(quarantine.get("materialize_job_id")) != failed_job_id
        or quarantine.get("release_id") != release_id
    ):
        raise EvidenceError("quarantine completion does not identify the requested tree")

    stem = f"partial-job-{failed_job_id}"
    intent_path = evidence / f"{stem}.seal-intent.json"
    inventory_path = evidence / f"{stem}.sealed-inventory.txt"
    marker_path = evidence / f"{stem}.sealed.json"
    lock_path = evidence / f".{stem}.seal.lock"
    guard = _exclusive_lock(lock_path) if apply else nullcontext()
    with guard:
        if marker_path.exists():
            marker = _read_json(marker_path, description="quarantine seal marker")
            rows, file_count, total_bytes = _inventory_rows(
                tree_path, include_modes=True
            )
            payload = _inventory_bytes(rows)
            if (
                marker.get("passed") is not True
                or marker.get("inventory_sha256") != _sha256_bytes(payload)
                or marker.get("file_count") != file_count
                or marker.get("total_bytes") != total_bytes
                or inventory_path.read_bytes() != payload
                or _writable_entries(tree_path)
            ):
                raise EvidenceError("sealed quarantine verification failed")
            return marker | {"status": "already_sealed"}

        content_sha, file_count, total_bytes = _content_identity(tree_path)
        tree_stat = os.lstat(tree_path)
        expected_intent = {
            "schema_version": 1,
            "protocol": "schema5-quarantined-materialization-seal-intent-v1",
            "release_id": release_id,
            "failed_job_id": failed_job_id,
            "tree": str(tree_path),
            "tree_device": tree_stat.st_dev,
            "tree_inode": tree_stat.st_ino,
            "content_sha256": content_sha,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "quarantine_completion": str(completion_path),
            "quarantine_completion_sha256": _sha256_file(completion_path),
        }
        if intent_path.exists():
            intent = _read_json(intent_path, description="quarantine seal intent")
            identity = dict(intent)
            intent_id = identity.pop("intent_id", None)
            created_at = identity.pop("created_at", None)
            if (
                identity != expected_intent
                or not isinstance(created_at, str)
                or intent_id
                != _sha256_bytes(
                    _canonical_json(expected_intent | {"created_at": created_at})
                )
            ):
                raise EvidenceError("quarantine seal intent drifted")
        else:
            intent = expected_intent | {"created_at": _utc_now()}
            intent["intent_id"] = _sha256_bytes(_canonical_json(intent))
            if apply:
                _atomic_json(intent_path, intent, mode=0o444)

        report = {
            "status": "dry_run" if not apply else "sealing",
            "tree": str(tree_path),
            "content_sha256": content_sha,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "intent": str(intent_path),
            "inventory": str(inventory_path),
            "marker": str(marker_path),
            "writable_entries": len(_writable_entries(tree_path)),
        }
        if not apply:
            return report

        _remove_write_bits(tree_path)
        final_content_sha, final_file_count, final_total_bytes = _content_identity(
            tree_path
        )
        if (
            final_content_sha != content_sha
            or final_file_count != file_count
            or final_total_bytes != total_bytes
            or _writable_entries(tree_path)
        ):
            raise EvidenceError("quarantine content changed while sealing")
        rows, file_count, total_bytes = _inventory_rows(tree_path, include_modes=True)
        inventory = _inventory_bytes(rows)
        _atomic_bytes(inventory_path, inventory, mode=0o444)
        marker = {
            "schema_version": 1,
            "protocol": "schema5-quarantined-materialization-seal-v1",
            "passed": True,
            "release_id": release_id,
            "failed_job_id": failed_job_id,
            "tree": str(tree_path),
            "tree_device": tree_stat.st_dev,
            "tree_inode": tree_stat.st_ino,
            "content_sha256": content_sha,
            "inventory": str(inventory_path),
            "inventory_sha256": _sha256_bytes(inventory),
            "file_count": file_count,
            "total_bytes": total_bytes,
            "intent": str(intent_path),
            "intent_sha256": _sha256_file(intent_path),
            "quarantine_completion": str(completion_path),
            "quarantine_completion_sha256": _sha256_file(completion_path),
            "sealed_at": _utc_now(),
        }
        marker["seal_id"] = _sha256_bytes(_canonical_json(marker))
        _atomic_json(marker_path, marker, mode=0o444)
        return marker | {"status": "sealed"}


def _scheduler_states(job_ids: Sequence[str]) -> list[dict[str, str]]:
    if not job_ids or any(not value.isdigit() for value in job_ids):
        raise EvidenceError("scheduler job IDs must be numeric")
    proc = subprocess.run(
        [
            "/usr/bin/sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,JobName,State,Elapsed,Start,End,ExitCode,NodeList",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=_sanitized_process_environment(),
    )
    if proc.returncode != 0:
        raise EvidenceError(f"sacct failed: {proc.stderr.strip()[:500]}")
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 8:
            raise EvidenceError(f"unexpected sacct row: {line!r}")
        row = dict(
            zip(
                (
                    "job_id",
                    "job_name",
                    "state",
                    "elapsed",
                    "start",
                    "end",
                    "exit_code",
                    "node_list",
                ),
                fields,
            )
        )
        row["state"] = row["state"].split("+", 1)[0].upper()
        rows.append(row)
    by_id = {row["job_id"]: row for row in rows}
    if set(by_id) != set(job_ids):
        raise EvidenceError(
            f"sacct did not return the exact job set: missing={sorted(set(job_ids)-set(by_id))}"
        )
    if any(row["state"] not in _TERMINAL_STATES for row in rows):
        raise EvidenceError("recovery failure cannot be recorded while a job is active")
    return [by_id[value] for value in job_ids]


def record_failure(
    *,
    output: str | Path,
    chain_manifest: str | Path,
    submission_receipt: str | Path,
    snapshot_marker: str | Path,
    snapshot_attestation: str | Path,
    materialization_log: str | Path,
    quarantine_seal: str | Path,
    incident: str | Path,
    job_ids: Sequence[str],
    apply: bool = False,
) -> dict[str, Any]:
    output_path = _absolute(output, description="failure envelope")
    inputs = {
        "chain_manifest": _absolute(chain_manifest, description="chain manifest"),
        "submission_receipt": _absolute(
            submission_receipt, description="submission receipt"
        ),
        "snapshot_marker": _absolute(snapshot_marker, description="snapshot marker"),
        "snapshot_attestation": _absolute(
            snapshot_attestation, description="snapshot attestation"
        ),
        "materialization_log": _absolute(
            materialization_log, description="materialization log"
        ),
        "quarantine_seal": _absolute(
            quarantine_seal, description="quarantine seal"
        ),
        "incident": _absolute(incident, description="operational incident"),
    }
    for name, path in inputs.items():
        if path.is_symlink() or not path.is_file():
            raise EvidenceError(f"{name} is missing or unsafe: {path}")
    log_root = inputs["materialization_log"].parent
    if log_root.is_symlink() or not log_root.is_dir():
        raise EvidenceError(f"r1 scheduler log root is missing or unsafe: {log_root}")
    scheduler_logs: list[dict[str, Any]] = []
    for candidate in sorted(log_root.rglob("*")):
        if candidate.is_symlink() or not candidate.is_file():
            raise EvidenceError(f"unsafe entry in r1 scheduler logs: {candidate}")
        scheduler_logs.append(
            {
                "relative_path": candidate.relative_to(log_root).as_posix(),
                "sha256": _sha256_file(candidate),
                "size": candidate.stat().st_size,
            }
        )
    expected_logs = {
        "source_checkout_18555909.out",
        "maintenance_preflight_18555910.out",
        "pre_repair_snapshot_18555911.out",
        "pre_repair_snapshot_verify_18555912.out",
        "release_materialize_18555913.out",
    }
    observed_logs = {row["relative_path"] for row in scheduler_logs}
    if observed_logs != expected_logs:
        raise EvidenceError(
            "r1 scheduler log set drifted: "
            f"missing={sorted(expected_logs - observed_logs)}, "
            f"unexpected={sorted(observed_logs - expected_logs)}"
        )
    states = _scheduler_states(job_ids)
    if sum(row["state"] == "FAILED" for row in states) != 1:
        raise EvidenceError("expected exactly one failed r1 recovery job")
    if not any(
        row["job_name"] == "asys-s5v11r1-materialize"
        and row["state"] == "FAILED"
        for row in states
    ):
        raise EvidenceError("r1 materialization failure is absent")
    artifacts = {
        name: {"path": str(path), "sha256": _sha256_file(path), "size": path.stat().st_size}
        for name, path in inputs.items()
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "schema5-recovery-chain-failure-v1",
        "status": "failed_closed",
        "classification": "requires_superseding_release",
        "retry_same_generation": False,
        "superseded_by": "sweep-recovery-schema5-v1.2",
        "failed_stage": "release_materialize",
        "scheduler_jobs": states,
        "scheduler_log_root": str(log_root),
        "scheduler_logs": scheduler_logs,
        "artifacts": artifacts,
        "recorded_at": _utc_now(),
    }
    payload["failure_id"] = _sha256_bytes(_canonical_json(payload))
    if output_path.exists():
        existing = _read_json(output_path, description="failure envelope")
        stable_existing = dict(existing)
        stable_payload = dict(payload)
        stable_existing.pop("recorded_at", None)
        stable_payload.pop("recorded_at", None)
        stable_existing.pop("failure_id", None)
        stable_payload.pop("failure_id", None)
        if stable_existing != stable_payload:
            raise EvidenceError("existing failure envelope drifted")
        return existing | {"status": "already_recorded"}
    report = payload | {"status": "dry_run", "output": str(output_path)}
    if not apply:
        return report
    _atomic_json(output_path, payload, mode=0o444)
    _atomic_bytes(
        output_path.with_suffix(output_path.suffix + ".sha256"),
        f"{_sha256_file(output_path)}  {output_path.name}\n".encode("utf-8"),
        mode=0o444,
    )
    return payload | {"status": "recorded", "output": str(output_path)}


def _metadata_version(prefix: Path, distribution: str) -> tuple[str, str]:
    site_packages = prefix / "lib" / "python3.11" / "site-packages"
    matches = sorted(site_packages.glob(f"{distribution}-*.dist-info/METADATA"))
    if len(matches) != 1:
        raise EvidenceError(
            f"{prefix} has {len(matches)} {distribution} METADATA records"
        )
    version = None
    for line in matches[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("Version: "):
            version = line.removeprefix("Version: ").strip()
            break
    if not version:
        raise EvidenceError(f"missing Version in {matches[0]}")
    return version, str(matches[0])


def record_conda_reconciliation_incident(
    *,
    output: str | Path,
    harness_prefix: str | Path,
    serving_prefix: str | Path,
    recovered_harness_record: str | Path,
    failed_materialization_log: str | Path,
    observed_at: str,
    apply: bool = False,
) -> dict[str, Any]:
    output_path = _absolute(output, description="incident output")
    harness = _absolute(harness_prefix, description="harness prefix")
    serving = _absolute(serving_prefix, description="serving prefix")
    recovered = _absolute(
        recovered_harness_record, description="recovered harness conda record"
    )
    failed_log = _absolute(
        failed_materialization_log, description="failed materialization log"
    )
    record_name = "setuptools-82.0.1-pyh332efcf_0.json"
    live_harness_record = harness / "conda-meta" / record_name
    live_serving_record = serving / "conda-meta" / record_name
    if live_harness_record.exists() or live_harness_record.is_symlink():
        raise EvidenceError(
            "harness stale Setuptools conda record is unexpectedly present"
        )
    if any(path.is_symlink() or not path.is_file() for path in (recovered, failed_log)):
        raise EvidenceError("incident preimage/log input is missing or unsafe")
    if live_serving_record.is_symlink():
        raise EvidenceError("serving stale Setuptools record is symlinked")
    recovered_payload = _read_json(
        recovered, description="recovered harness Setuptools record"
    )
    serving_payload = (
        _read_json(live_serving_record, description="serving Setuptools record")
        if live_serving_record.is_file()
        else None
    )
    expected_identity = {
        "name": "setuptools",
        "version": "82.0.1",
        "build": "pyh332efcf_0",
    }
    checked_records = [("recovered harness", recovered_payload)]
    if serving_payload is not None:
        checked_records.append(("live serving", serving_payload))
    for description, payload in checked_records:
        if (
            {key: payload.get(key) for key in expected_identity}
            != expected_identity
            or payload.get("sha256")
            != "82088a6e4daa33329a30bc26dc19a98c7c1d3f05c0f73ce9845d4eab4924e9e1"
        ):
            raise EvidenceError(f"{description} Setuptools record identity drifted")
    harness_version, harness_metadata = _metadata_version(harness, "setuptools")
    serving_version, serving_metadata = _metadata_version(serving, "setuptools")
    if harness_version != "81.0.0" or serving_version != "81.0.0":
        raise EvidenceError("live Setuptools runtime is not the observed pip 81.0.0")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "schema5-conda-pip-reconciliation-incident-v1",
        "classification": "source_metadata_reconciled_by_conda_pip_interop",
        "observed_at": observed_at,
        "recorded_at": _utc_now(),
        "intended_action": "read_only_environment_inventory",
        "mutation_intended": False,
        "live_sources_must_not_be_queried_by_conda": True,
        "harness_prefix": str(harness),
        "serving_prefix": str(serving),
        "harness_stale_conda_record_present": False,
        "serving_stale_conda_record_present": serving_payload is not None,
        "runtime_owner": {
            "distribution": "setuptools",
            "version": "81.0.0",
            "harness_metadata": harness_metadata,
            "harness_metadata_sha256": _sha256_file(Path(harness_metadata)),
            "serving_metadata": serving_metadata,
            "serving_metadata_sha256": _sha256_file(Path(serving_metadata)),
        },
        "superseded_conda_record": {
            **expected_identity,
            "artifact_sha256": "82088a6e4daa33329a30bc26dc19a98c7c1d3f05c0f73ce9845d4eab4924e9e1",
            "recovered_harness_path": str(recovered),
            "recovered_harness_sha256": _sha256_file(recovered),
            "serving_path": (
                str(live_serving_record) if serving_payload is not None else None
            ),
            "serving_sha256": (
                _sha256_file(live_serving_record)
                if serving_payload is not None
                else None
            ),
        },
        "failed_materialization_log": str(failed_log),
        "failed_materialization_log_sha256": _sha256_file(failed_log),
        "resolution": "capture_live_bytes_without_conda_then_normalize_only_immutable_seeds",
    }
    payload["incident_id"] = _sha256_bytes(_canonical_json(payload))
    if output_path.exists():
        existing = _read_json(output_path, description="Conda incident")
        existing_identity = dict(existing)
        existing_incident_id = existing_identity.pop("incident_id", None)
        sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
        output_metadata = os.lstat(output_path)
        try:
            sidecar_metadata = os.lstat(sidecar)
        except OSError as exc:
            raise EvidenceError(
                f"existing Conda incident checksum is missing: {sidecar}"
            ) from exc
        expected_sidecar = (
            f"{_sha256_file(output_path)}  {output_path.name}\n"
        ).encode()
        if (
            output_path.is_symlink()
            or not stat.S_ISREG(output_metadata.st_mode)
            or output_metadata.st_nlink != 1
            or stat.S_IMODE(output_metadata.st_mode) & 0o222
            or output_path.read_bytes() != _canonical_json(existing)
            or existing_incident_id
            != _sha256_bytes(_canonical_json(existing_identity))
            or sidecar.is_symlink()
            or not stat.S_ISREG(sidecar_metadata.st_mode)
            or sidecar_metadata.st_nlink != 1
            or stat.S_IMODE(sidecar_metadata.st_mode) & 0o222
            or sidecar.read_bytes() != expected_sidecar
        ):
            raise EvidenceError(
                "existing Conda incident identity/checksum is invalid"
            )
        stable_existing = dict(existing)
        stable_payload = dict(payload)
        for value in (stable_existing, stable_payload):
            value.pop("recorded_at", None)
            value.pop("incident_id", None)
        if stable_existing != stable_payload:
            raise EvidenceError("existing Conda incident drifted")
        return existing | {"status": "already_recorded"}
    if not apply:
        return payload | {"status": "dry_run", "output": str(output_path)}
    _atomic_json(output_path, payload, mode=0o444)
    _atomic_bytes(
        output_path.with_suffix(output_path.suffix + ".sha256"),
        f"{_sha256_file(output_path)}  {output_path.name}\n".encode(),
        mode=0o444,
    )
    return payload | {"status": "recorded", "output": str(output_path)}


def _run_git(checkout: Path, *arguments: str) -> str:
    environment = _sanitized_process_environment()
    # The checkout is part of the byte/inode inventory surrounding r3 probes.
    # Disable Git's optional index refresh so a read-only identity query cannot
    # atomically replace .git/index and thereby mutate its own declared input.
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        proc = subprocess.run(
            ("/usr/bin/git", "-C", str(checkout), *arguments),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvidenceError(f"Git identity query failed: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no output"
        raise EvidenceError(
            f"Git identity query failed rc={proc.returncode}: {detail[:500]}"
        )
    return proc.stdout.strip()


def _validate_canary_release_identity(
    *, tree: Path, release_checkout: Path
) -> dict[str, Any]:
    composite_path = tree / "COMPOSITE_CANARY_INTENT.json"
    composite = _strict_json_artifact(
        composite_path, description="composite canary intent"
    )
    if (
        composite.get("schema_version") != 4
        or composite.get("kind")
        != "schema5_slurm_fleet_composite_canary_intent"
        or composite.get("canary_root") != str(tree)
    ):
        raise EvidenceError("composite canary intent identity is invalid")
    identity = composite.get("code_identity")
    required = {
        "schema_version",
        "protocol",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "canary_script",
        "fleet_transactions",
        "durable_git_publisher",
        "durable_git_release",
    }
    if not isinstance(identity, dict) or set(identity) != required:
        raise EvidenceError("composite canary code identity fields drifted")
    release_tag = identity.get("release_tag")
    commit = str(identity.get("release_git_commit", ""))
    tag_object = str(identity.get("release_tag_object", ""))
    if (
        identity.get("schema_version") != 1
        or identity.get("protocol")
        != "schema5-v1.2-r2-slurm-canary-code-v3"
        or release_tag != "sweep-recovery-schema5-v1.2-r2"
        or _COMMIT_RE.fullmatch(commit) is None
        or _COMMIT_RE.fullmatch(tag_object) is None
    ):
        raise EvidenceError("composite canary release identity is invalid")
    tag_ref = f"refs/tags/{release_tag}"
    if (
        _run_git(release_checkout, "cat-file", "-t", tag_ref) != "tag"
        or _run_git(release_checkout, "rev-parse", "--verify", tag_ref)
        != tag_object
        or _run_git(release_checkout, "rev-parse", "--verify", f"{tag_ref}^{{commit}}")
        != commit
        or _run_git(release_checkout, "rev-parse", "--verify", "HEAD") != commit
        or _run_git(
            release_checkout, "status", "--porcelain=v1", "--untracked-files=all"
        )
    ):
        raise EvidenceError("release checkout is not the exact clean annotated tag")
    code_fields = {
        "canary_script": "scripts/run_schema5_slurm_fleet_canary.py",
        "fleet_transactions": "src/agents_scaling/serving/fleet_transactions.py",
        "durable_git_publisher": "scripts/publish_schema5_durable_git_release.py",
    }
    for field, expected_relative in code_fields.items():
        record = identity.get(field)
        if (
            not isinstance(record, dict)
            or set(record) != {"git_path", "sha256", "size"}
            or record.get("git_path") != expected_relative
            or _SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
            or not isinstance(record.get("size"), int)
        ):
            raise EvidenceError(f"canary code record is invalid: {field}")
        path = _safe_existing_path(
            release_checkout / expected_relative,
            description=f"release {field}",
            kind="file",
        )
        observed = _stable_file_record(path, description=f"release {field}")
        if (
            observed["sha256"] != record["sha256"]
            or observed["size"] != record["size"]
        ):
            raise EvidenceError(f"canary code hash drifted: {field}")
    durable = identity.get("durable_git_release")
    if not isinstance(durable, dict):
        raise EvidenceError("durable Git release binding is invalid")
    durable_path = _safe_existing_path(
        str(durable.get("path", "")),
        description="durable Git release marker",
        kind="file",
    )
    durable_record = _stable_file_record(
        durable_path, description="durable Git release marker"
    )
    if (
        durable_record["sha256"] != durable.get("sha256")
        or durable.get("release_git_commit") != commit
        or durable.get("release_tag_object") != tag_object
    ):
        raise EvidenceError("durable Git release binding drifted")

    # Every component that reached marker-first intent must agree with the composite.
    for candidate in sorted(tree.rglob("*.json")):
        payload = _strict_json_artifact(candidate, description="canary JSON artifact")
        if "code_identity" in payload and payload["code_identity"] != identity:
            raise EvidenceError(
                f"canary component code identity drifted: {candidate.relative_to(tree)}"
            )
    return {
        "composite_intent": _stable_file_record(
            composite_path, description="composite canary intent"
        ),
        "release_checkout": str(release_checkout),
        "release_tag": release_tag,
        "release_git_commit": commit,
        "release_tag_object": tag_object,
        "code_identity": identity,
    }


def _mutation_evidence_binding(path: Path) -> dict[str, Any]:
    payload = _strict_json_artifact(path, description="zero-mutation basis evidence")
    protocol = str(payload.get("protocol", ""))
    if (
        protocol == "schema5-v1.2-r2-r1-zero-result-mutation-receipt-v1"
        and payload.get("passed") is True
        and payload.get("legacy_result_mutation_count") == 0
        and payload.get("schema5_result_mutation_count") == 0
    ):
        kind = "zero_result_mutation_receipt"
    elif (
        path.name == "SNAPSHOT_COMPLETE.json"
        and payload.get("verified") is True
        and payload.get("read_only") is True
        and isinstance(payload.get("snapshot_id"), str)
        and isinstance(payload.get("file_count"), int)
        and isinstance(payload.get("total_bytes"), int)
        and _SHA256_RE.fullmatch(
            str(payload.get("snapshot_inventory_sha256", ""))
        )
    ):
        kind = "verified_read_only_snapshot"
    elif (
        payload.get("legacy_jobs") == 0
        and payload.get("held_cell_locks") == 0
        and isinstance(payload.get("maintenance_interlock"), dict)
    ):
        interlock_record = payload["maintenance_interlock"]
        interlock = _safe_existing_path(
            str(interlock_record.get("path", "")),
            description="maintenance interlock",
            kind="file",
        )
        if _sha256_file(interlock) != interlock_record.get("sha256"):
            raise EvidenceError("maintenance interlock hash drifted")
        state = _read_json(interlock, description="maintenance interlock")
        if (
            state.get("desired_state") != "maintenance"
            or state.get("admission_enabled") is not False
        ):
            raise EvidenceError("maintenance interlock no longer fails closed")
        kind = "maintenance_quiescence"
    else:
        raise EvidenceError(
            f"unsupported zero-mutation basis evidence contract: {path}"
        )
    return _stable_file_record(path, description="zero-mutation basis evidence") | {
        "evidence_kind": kind,
        "protocol": protocol or None,
    }


def _walk_values(value: Any, *, key: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk_values(child, key=str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child, key=key)
    else:
        yield key, value


def _known_canary_scheduler_identity(tree: Path) -> dict[str, list[str]]:
    comments: set[str] = set()
    job_ids: set[str] = set()
    job_names: set[str] = set()
    for candidate in sorted(tree.rglob("*.json")):
        payload = _strict_json_artifact(candidate, description="canary JSON artifact")
        for key, value in _walk_values(payload):
            job_id_field = (
                key == "job_id" or key.endswith("_job_id") or key == "job_ids"
            )
            if (
                job_id_field
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 1
            ):
                job_ids.add(str(value))
            if not isinstance(value, str):
                continue
            if key in {"comment", "scheduler_comment"}:
                if value.startswith(("asys-s5-fleet:", "asys:s5-dependency-canary:")):
                    if _CANARY_COMMENT_RE.fullmatch(value) is None:
                        raise EvidenceError(
                            f"malformed canary scheduler comment in {candidate}"
                        )
                    comments.add(value)
            if key == "job_name" and value.startswith("asys-s5-serve-"):
                if _CANARY_JOB_NAME_RE.fullmatch(value) is None:
                    raise EvidenceError(f"malformed canary job name in {candidate}")
                job_names.add(value)
            if job_id_field and value.isdigit():
                job_ids.add(value)
    if not comments or not job_names:
        raise EvidenceError("partial canary lacks scheduler identity")
    return {
        "comments": sorted(comments),
        "job_ids": sorted(job_ids, key=int),
        "job_names": sorted(job_names),
    }


def _submit_line_comment(value: str) -> str:
    try:
        tokens = shlex.split(value)
    except ValueError as exc:
        raise EvidenceError(f"invalid sacct SubmitLine: {exc}") from exc
    found: list[str] = []
    for index, token in enumerate(tokens):
        if token.startswith("--comment="):
            found.append(token.split("=", 1)[1])
        elif token == "--comment":
            if index + 1 >= len(tokens):
                raise EvidenceError("sacct SubmitLine has an empty --comment")
            found.append(tokens[index + 1])
    if len(found) > 1:
        raise EvidenceError("sacct SubmitLine has duplicate comments")
    return found[0] if found else ""


def _scheduler_state(value: str) -> str:
    return value.strip().split()[0].split("+", 1)[0].upper() if value.strip() else ""


def _reparse_canary_scheduler_rows(
    raw: Mapping[str, Any],
    *,
    known: Mapping[str, Sequence[str]],
) -> dict[str, list[dict[str, str]]]:
    """Derive scoped scheduler rows only from the sealed raw stdout bytes."""

    if set(raw) != {"squeue", "sacct"}:
        raise EvidenceError("canary scheduler raw evidence is incomplete")
    rows_by_source: dict[str, list[dict[str, str]]] = {
        "squeue": [],
        "sacct": [],
    }
    expected_comments = set(known["comments"])
    expected_ids = set(known["job_ids"])
    expected_names = set(known["job_names"])
    unique_names = {
        value
        for value in expected_names
        if _CANARY_UNIQUE_JOB_NAME_RE.fullmatch(value)
    }
    for source in ("squeue", "sacct"):
        record = raw[source]
        if not isinstance(record, Mapping) or not isinstance(
            record.get("stdout"), str
        ):
            raise EvidenceError(f"invalid sealed {source} scheduler stdout")
        for raw_line in record["stdout"].splitlines():
            if not raw_line.strip():
                continue
            fields = raw_line.split("|", 7 if source == "squeue" else 8)
            if len(fields) != (8 if source == "squeue" else 9):
                raise EvidenceError(
                    f"malformed sealed {source} canary row: {raw_line!r}"
                )
            if source == "squeue":
                (
                    job_id,
                    comment,
                    name,
                    state,
                    partition,
                    start,
                    end,
                    command,
                ) = fields
                exit_code = ""
            else:
                (
                    job_id,
                    comment,
                    name,
                    state,
                    partition,
                    start,
                    end,
                    exit_code,
                    command,
                ) = fields
                derived = _submit_line_comment(command)
                normalized = (
                    ""
                    if comment.lower() in {"", "(null)", "null", "none"}
                    else comment
                )
                if normalized and derived and normalized != derived:
                    raise EvidenceError(
                        f"sealed sacct comment conflict for job {job_id}"
                    )
                comment = normalized or derived
            job_id, comment, name = (
                job_id.strip(),
                comment.strip(),
                name.strip(),
            )
            scoped = (
                job_id in expected_ids
                or comment in expected_comments
                or name in unique_names
            )
            if not scoped:
                continue
            if (
                not job_id.isdigit()
                or comment not in expected_comments
                or name not in expected_names
            ):
                raise EvidenceError(
                    "ambiguous sealed scheduler row overlaps canary identity: "
                    f"{job_id}"
                )
            rows_by_source[source].append(
                {
                    "job_id": job_id,
                    "comment": comment,
                    "job_name": name,
                    "state": _scheduler_state(state),
                    "partition": partition.strip(),
                    "start": start.strip(),
                    "end": end.strip(),
                    "exit_code": exit_code.strip(),
                    "submit_line": command.strip(),
                }
            )
    if rows_by_source["squeue"]:
        raise EvidenceError(
            "sealed scheduler raw evidence contains an active matching canary job"
        )
    sacct_by_id: dict[str, dict[str, str]] = {}
    for row in rows_by_source["sacct"]:
        previous = sacct_by_id.get(row["job_id"])
        if previous is not None and previous != row:
            raise EvidenceError(
                f"ambiguous sealed sacct rows for canary job {row['job_id']}"
            )
        sacct_by_id[row["job_id"]] = row
        if row["state"] not in _TERMINAL_STATES:
            raise EvidenceError(
                "matching canary job in sealed raw evidence is not terminal: "
                f"{row['job_id']}={row['state']}"
            )
    if set(sacct_by_id) != expected_ids:
        raise EvidenceError(
            "sealed sacct raw evidence did not return every known canary job "
            "exactly"
        )
    return {
        "squeue": [],
        "sacct": [
            sacct_by_id[key] for key in sorted(sacct_by_id, key=int)
        ],
    }


def _capture_canary_scheduler_evidence(
    *,
    scheduler_user: str,
    since: str,
    known: Mapping[str, Sequence[str]],
    phase: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    commands = {
        "squeue": [
            "squeue",
            "-u",
            scheduler_user,
            "-h",
            "-r",
            "-o",
            "%i|%k|%j|%T|%P|%S|%e|%o",
        ],
        "sacct": [
            "sacct",
            "-u",
            scheduler_user,
            "-X",
            "-n",
            "-P",
            "-S",
            since,
            (
                "--format=JobIDRaw,Comment%256,JobName%64,State,Partition,"
                "Start,End,ExitCode,SubmitLine"
            ),
        ],
    }
    raw: dict[str, Any] = {}
    rows_by_source: dict[str, list[dict[str, str]]] = {"squeue": [], "sacct": []}
    expected_comments = set(known["comments"])
    expected_ids = set(known["job_ids"])
    expected_names = set(known["job_names"])
    unique_names = {
        value for value in expected_names if _CANARY_UNIQUE_JOB_NAME_RE.fullmatch(value)
    }
    for source, argv in commands.items():
        try:
            proc = runner(
                argv,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EvidenceError(f"{source} canary query failed: {exc}") from exc
        if (
            isinstance(proc.returncode, bool)
            or not isinstance(proc.returncode, int)
            or not isinstance(proc.stdout, str)
            or not isinstance(proc.stderr, str)
        ):
            raise EvidenceError(
                f"{source} canary query returned an invalid process result"
            )
        raw[source] = {
            "argv": argv,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "stdout_sha256": _sha256_bytes(proc.stdout.encode("utf-8")),
            "stderr_sha256": _sha256_bytes(proc.stderr.encode("utf-8")),
        }
        if proc.returncode != 0:
            raise EvidenceError(
                f"{source} canary query failed rc={proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )
        for raw_line in proc.stdout.splitlines():
            if not raw_line.strip():
                continue
            fields = raw_line.split("|", 7 if source == "squeue" else 8)
            if len(fields) != (8 if source == "squeue" else 9):
                raise EvidenceError(f"malformed {source} canary row: {raw_line!r}")
            if source == "squeue":
                job_id, comment, name, state, partition, start, end, command = fields
                exit_code = ""
            else:
                (
                    job_id,
                    comment,
                    name,
                    state,
                    partition,
                    start,
                    end,
                    exit_code,
                    command,
                ) = fields
                derived = _submit_line_comment(command)
                normalized = "" if comment.lower() in {"", "(null)", "null", "none"} else comment
                if normalized and derived and normalized != derived:
                    raise EvidenceError(f"sacct comment conflict for job {job_id}")
                comment = normalized or derived
            job_id, comment, name = job_id.strip(), comment.strip(), name.strip()
            scoped = (
                job_id in expected_ids
                or comment in expected_comments
                or name in unique_names
            )
            if not scoped:
                continue
            if (
                not job_id.isdigit()
                or comment not in expected_comments
                or name not in expected_names
            ):
                raise EvidenceError(
                    f"ambiguous scheduler row overlaps canary identity: {job_id}"
                )
            rows_by_source[source].append(
                {
                    "job_id": job_id,
                    "comment": comment,
                    "job_name": name,
                    "state": _scheduler_state(state),
                    "partition": partition.strip(),
                    "start": start.strip(),
                    "end": end.strip(),
                    "exit_code": exit_code.strip(),
                    "submit_line": command.strip(),
                }
            )
    if rows_by_source["squeue"]:
        raise EvidenceError(
            "active or pending matching canary jobs remain in complete squeue truth"
        )
    sacct_by_id: dict[str, dict[str, str]] = {}
    for row in rows_by_source["sacct"]:
        previous = sacct_by_id.get(row["job_id"])
        if previous is not None and previous != row:
            raise EvidenceError(f"ambiguous sacct rows for canary job {row['job_id']}")
        sacct_by_id[row["job_id"]] = row
        if row["state"] not in _TERMINAL_STATES:
            raise EvidenceError(
                f"matching canary job is not terminal: {row['job_id']}={row['state']}"
            )
    if set(sacct_by_id) != expected_ids:
        raise EvidenceError(
            "sacct did not return every known canary job exactly: "
            f"missing={sorted(expected_ids - set(sacct_by_id), key=int)}"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": _SCHEDULER_EVIDENCE_PROTOCOL,
        "phase": phase,
        "captured_at": _utc_now(),
        "scheduler_user": scheduler_user,
        "scheduler_since": since,
        "complete_squeue_truth": True,
        "complete_sacct_truth": True,
        "known_identity": {key: list(value) for key, value in known.items()},
        "matching_squeue_rows": [],
        "matching_sacct_rows": [sacct_by_id[key] for key in sorted(sacct_by_id, key=int)],
        "no_active_matching_jobs": True,
        "raw": raw,
    }
    payload["evidence_id"] = _sha256_bytes(_canonical_json(payload))
    return payload


def _validate_scheduler_evidence(
    value: Mapping[str, Any],
    *,
    scheduler_user: str,
    since: str,
    known: Mapping[str, Sequence[str]],
    phase: str,
) -> None:
    expected_fields = {
        "schema_version",
        "protocol",
        "phase",
        "captured_at",
        "scheduler_user",
        "scheduler_since",
        "complete_squeue_truth",
        "complete_sacct_truth",
        "known_identity",
        "matching_squeue_rows",
        "matching_sacct_rows",
        "no_active_matching_jobs",
        "raw",
        "evidence_id",
    }
    identity = dict(value)
    evidence_id = identity.pop("evidence_id", None)
    if (
        set(value) != expected_fields
        or value.get("schema_version") != 1
        or value.get("protocol") != _SCHEDULER_EVIDENCE_PROTOCOL
        or value.get("phase") != phase
        or not isinstance(value.get("captured_at"), str)
        or not value["captured_at"]
        or value.get("scheduler_user") != scheduler_user
        or value.get("scheduler_since") != since
        or value.get("known_identity")
        != {key: list(item) for key, item in known.items()}
        or value.get("complete_squeue_truth") is not True
        or value.get("complete_sacct_truth") is not True
        or value.get("matching_squeue_rows") != []
        or value.get("no_active_matching_jobs") is not True
        or evidence_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise EvidenceError(f"sealed {phase} scheduler evidence drifted")
    raw = value.get("raw")
    if not isinstance(raw, dict) or set(raw) != {"squeue", "sacct"}:
        raise EvidenceError(f"sealed {phase} scheduler raw evidence is incomplete")
    expected_commands = {
        "squeue": [
            "squeue",
            "-u",
            scheduler_user,
            "-h",
            "-r",
            "-o",
            "%i|%k|%j|%T|%P|%S|%e|%o",
        ],
        "sacct": [
            "sacct",
            "-u",
            scheduler_user,
            "-X",
            "-n",
            "-P",
            "-S",
            since,
            (
                "--format=JobIDRaw,Comment%256,JobName%64,State,Partition,"
                "Start,End,ExitCode,SubmitLine"
            ),
        ],
    }
    for source, record in raw.items():
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "argv",
                "returncode",
                "stdout",
                "stderr",
                "stdout_sha256",
                "stderr_sha256",
            }
            or record.get("argv") != expected_commands[source]
            or record.get("returncode") != 0
            or not isinstance(record.get("stdout"), str)
            or not isinstance(record.get("stderr"), str)
            or record.get("stdout_sha256")
            != _sha256_bytes(record["stdout"].encode("utf-8"))
            or record.get("stderr_sha256")
            != _sha256_bytes(record["stderr"].encode("utf-8"))
        ):
            raise EvidenceError(f"sealed {phase} {source} evidence drifted")
    reparsed = _reparse_canary_scheduler_rows(raw, known=known)
    if (
        value.get("matching_squeue_rows") != reparsed["squeue"]
        or value.get("matching_sacct_rows") != reparsed["sacct"]
    ):
        raise EvidenceError(
            f"sealed {phase} derived scheduler rows differ from raw evidence"
        )


def _canary_failure_artifact_paths(evidence_root: Path) -> dict[str, Path]:
    return {
        "intent": evidence_root / "CANARY_FAILURE_SEAL_INTENT.json",
        "before_inventory": evidence_root / "CANARY_FAILURE_PRESEAL_INVENTORY.jsonl",
        "scheduler_pre": evidence_root / "CANARY_FAILURE_SCHEDULER_PRE.json",
        "scheduler_post": evidence_root / "CANARY_FAILURE_SCHEDULER_POST.json",
        "after_inventory": evidence_root / "CANARY_FAILURE_SEALED_INVENTORY.jsonl",
        "marker": evidence_root / "CANARY_FAILURE_SEALED.json",
    }


_CANARY_FAILURE_ARTIFACT_ORDER = (
    "intent",
    "before_inventory",
    "scheduler_pre",
    "scheduler_post",
    "after_inventory",
    "marker",
)


def _canary_failure_staging_root(evidence_root: Path) -> Path:
    return (
        evidence_root.parent
        / f".{evidence_root.name}.canary-failure-staging"
    )


def _canary_failure_pending_path(evidence_root: Path, target: Path) -> Path:
    return _canary_failure_staging_root(evidence_root) / (
        f"{target.name}.pending"
    )


def _validate_or_recover_canary_staging(
    evidence_root: Path,
    artifacts: Mapping[str, Path],
    *,
    recover: bool,
) -> list[str]:
    """Validate, and optionally remove, only this transaction's sibling temps."""

    staging = _canary_failure_staging_root(evidence_root)
    if not staging.exists() and not staging.is_symlink():
        return []
    staging = _safe_existing_path(
        staging,
        description="canary failure sibling staging root",
        kind="directory",
    )
    metadata = os.lstat(staging)
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise EvidenceError(
            "canary failure sibling staging root has unsafe ownership or mode"
        )
    allowed = {
        _canary_failure_pending_path(evidence_root, target).name
        for target in artifacts.values()
    }
    entries = sorted(staging.iterdir(), key=lambda value: value.name)
    for candidate in entries:
        candidate_metadata = os.lstat(candidate)
        if (
            candidate.name not in allowed
            or not stat.S_ISREG(candidate_metadata.st_mode)
            or candidate_metadata.st_nlink != 1
            or candidate_metadata.st_uid != os.getuid()
        ):
            raise EvidenceError(
                f"unsafe or foreign canary failure staging entry: {candidate}"
            )
    names = [candidate.name for candidate in entries]
    if not recover:
        return names
    for candidate in entries:
        candidate.unlink()
    _fsync_directory(staging)
    staging.rmdir()
    _fsync_directory(staging.parent)
    return names


def _canary_atomic_bytes(
    path: Path,
    payload: bytes,
    *,
    evidence_root: Path,
    artifacts: Mapping[str, Path],
    mode: int = 0o444,
) -> None:
    """Publish one artifact from recoverable sibling staging.

    A hard interruption can leave only an exact ``*.pending`` file outside the
    evidence directory.  The next locked invocation discards that partial staging
    file before reconstructing the artifact from durable upstream state.
    """

    if path.parent != evidence_root or path not in artifacts.values():
        raise EvidenceError("canary atomic target escapes its evidence contract")
    if path.exists() or path.is_symlink():
        raise EvidenceError(f"canary atomic target already exists: {path}")
    if not evidence_root.exists():
        evidence_root.mkdir(mode=0o750)
        _fsync_directory(evidence_root.parent)
    _safe_existing_path(
        evidence_root,
        description="canary failure evidence root",
        kind="directory",
    )
    _validate_or_recover_canary_staging(
        evidence_root, artifacts, recover=True
    )
    staging = _canary_failure_staging_root(evidence_root)
    staging.mkdir(mode=0o700)
    _fsync_directory(staging.parent)
    pending = _canary_failure_pending_path(evidence_root, path)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(pending, flags, 0o600)
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        os.replace(pending, path)
        _fsync_directory(evidence_root)
        _fsync_directory(staging)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if pending.exists() and not pending.is_symlink():
            pending.unlink()
            _fsync_directory(staging)
        if staging.exists() and not staging.is_symlink():
            try:
                staging.rmdir()
            except OSError:
                pass
            else:
                _fsync_directory(staging.parent)
    _validate_or_recover_canary_staging(
        evidence_root, artifacts, recover=True
    )


def _canary_atomic_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    evidence_root: Path,
    artifacts: Mapping[str, Path],
) -> None:
    _canary_atomic_bytes(
        path,
        _canonical_json(payload),
        evidence_root=evidence_root,
        artifacts=artifacts,
    )


def _validate_canary_evidence_file_set(
    evidence_root: Path,
    artifacts: Mapping[str, Path],
    *,
    expected_prefix_length: int | None,
    require_directory_read_only: bool,
) -> int:
    """Require a prefix of the exact six-artifact marker-last transaction."""

    if not evidence_root.exists() and not evidence_root.is_symlink():
        if expected_prefix_length not in {None, 0}:
            raise EvidenceError("canary failure evidence root is missing")
        return 0
    evidence_root = _safe_existing_path(
        evidence_root,
        description="canary failure evidence root",
        kind="directory",
    )
    directory_metadata = os.lstat(evidence_root)
    if (
        directory_metadata.st_uid != os.getuid()
        or stat.S_IMODE(directory_metadata.st_mode) & 0o022
        or (
            require_directory_read_only
            and stat.S_IMODE(directory_metadata.st_mode) & 0o222
        )
    ):
        raise EvidenceError(
            "canary failure evidence root has unsafe ownership or mode"
        )
    actual = {candidate.name: candidate for candidate in evidence_root.iterdir()}
    expected_names = [artifacts[key].name for key in _CANARY_FAILURE_ARTIFACT_ORDER]
    if len(set(expected_names)) != len(expected_names):
        raise EvidenceError("canary failure artifact names are not unique")
    unknown = sorted(set(actual) - set(expected_names))
    if unknown:
        raise EvidenceError(
            f"unexpected canary failure evidence artifacts: {unknown}"
        )
    observed_keys = {
        key
        for key in _CANARY_FAILURE_ARTIFACT_ORDER
        if artifacts[key].name in actual
    }
    valid_prefixes = [
        set(_CANARY_FAILURE_ARTIFACT_ORDER[:length])
        for length in range(len(_CANARY_FAILURE_ARTIFACT_ORDER) + 1)
    ]
    if observed_keys not in valid_prefixes:
        raise EvidenceError(
            "canary failure evidence is not a valid marker-last prefix"
        )
    observed_length = valid_prefixes.index(observed_keys)
    if (
        expected_prefix_length is not None
        and observed_length != expected_prefix_length
    ):
        raise EvidenceError(
            "canary failure evidence artifact cardinality/order drifted"
        )
    for key in _CANARY_FAILURE_ARTIFACT_ORDER[:observed_length]:
        candidate = artifacts[key]
        metadata = os.lstat(candidate)
        if (
            candidate.resolve(strict=True) != candidate
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o222
        ):
            raise EvidenceError(
                f"canary failure evidence artifact is unsafe: {candidate}"
            )
    return observed_length


def _write_or_verify_canary_bytes(
    path: Path,
    payload: bytes,
    *,
    description: str,
    evidence_root: Path,
    artifacts: Mapping[str, Path],
) -> None:
    if path.exists() or path.is_symlink():
        _safe_existing_path(path, description=description, kind="file")
        metadata = os.lstat(path)
        if (
            path.read_bytes() != payload
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o222
        ):
            raise EvidenceError(f"{description} drifted")
        return
    _canary_atomic_bytes(
        path,
        payload,
        evidence_root=evidence_root,
        artifacts=artifacts,
    )


def seal_canary_failure(
    *,
    tree: str | Path,
    evidence_root: str | Path,
    release_checkout: str | Path,
    scheduler_user: str,
    error_classification: str,
    error_summary: str,
    mutation_evidence: Sequence[str | Path],
    apply: bool = False,
    scheduler_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Forensically seal one failed partial r2 composite canary without cancelling jobs."""

    tree_path = _safe_existing_path(
        tree, description="partial canary root", kind="directory"
    )
    evidence = _safe_output_directory(
        evidence_root, description="canary failure evidence root"
    )
    checkout = _safe_existing_path(
        release_checkout, description="release checkout", kind="directory"
    )
    if evidence == tree_path or tree_path in evidence.parents or evidence in tree_path.parents:
        raise EvidenceError("canary failure evidence root must be outside the partial tree")
    if (
        not scheduler_user
        or re.fullmatch(r"[A-Za-z0-9_.-]+", scheduler_user) is None
        or _ERROR_CLASSIFICATION_RE.fullmatch(error_classification) is None
        or not error_summary
        or len(error_summary) > 4000
        or any(ord(character) < 32 and character not in "\t\n" for character in error_summary)
        or not mutation_evidence
    ):
        raise EvidenceError("invalid canary failure classification or scheduler scope")
    if scheduler_runner is None:
        expected_user = pwd.getpwuid(os.getuid()).pw_name
        if scheduler_user != expected_user:
            raise EvidenceError(
                f"scheduler user must match the effective account: {expected_user}"
            )

        def runner(
            argv: Sequence[str], **kwargs: Any
        ) -> subprocess.CompletedProcess[Any]:
            kwargs["env"] = _sanitized_process_environment()
            return subprocess.run(argv, **kwargs)
    else:
        runner = scheduler_runner

    release = _validate_canary_release_identity(
        tree=tree_path, release_checkout=checkout
    )
    known = _known_canary_scheduler_identity(tree_path)
    bases = [
        _mutation_evidence_binding(
            _safe_existing_path(
                value, description="zero-mutation basis evidence", kind="file"
            )
        )
        for value in mutation_evidence
    ]
    if len({record["path"] for record in bases}) != len(bases):
        raise EvidenceError("duplicate zero-mutation basis evidence")
    composite = _read_json(
        tree_path / "COMPOSITE_CANARY_INTENT.json",
        description="composite canary intent",
    )
    created_at = composite.get("created_at")
    if not isinstance(created_at, (int, float)) or isinstance(created_at, bool):
        raise EvidenceError("composite canary timestamp is invalid")
    since = datetime.fromtimestamp(float(created_at), tz=timezone.utc).strftime("%Y-%m-%d")

    artifacts = _canary_failure_artifact_paths(evidence)
    lock_path = evidence.parent / f".{evidence.name}.canary-failure-seal.lock"
    guard = _exclusive_lock(lock_path) if apply else nullcontext()
    with guard:
        staged_entries = _validate_or_recover_canary_staging(
            evidence, artifacts, recover=apply
        )
        _validate_canary_evidence_file_set(
            evidence,
            artifacts,
            expected_prefix_length=None,
            require_directory_read_only=False,
        )
        current_rows, file_count, total_bytes = _full_tree_inventory(tree_path)
        # A second full pass closes the scan-to-intent mutation window.
        repeated_rows, repeated_count, repeated_bytes = _full_tree_inventory(tree_path)
        if (
            current_rows != repeated_rows
            or file_count != repeated_count
            or total_bytes != repeated_bytes
        ):
            raise EvidenceError("partial canary tree changed during preseal inventory")

        if artifacts["intent"].exists() or artifacts["intent"].is_symlink():
            intent = _strict_json_artifact(
                artifacts["intent"], description="canary failure seal intent"
            )
            before_rows = intent.get("preseal_inventory")
            stable = dict(intent)
            intent_id = stable.pop("intent_id", None)
            if (
                intent.get("schema_version") != 1
                or intent.get("protocol") != _CANARY_SEAL_INTENT_PROTOCOL
                or intent.get("tree") != str(tree_path)
                or intent.get("evidence_root") != str(evidence)
                or intent.get("release") != release
                or intent.get("known_scheduler_identity") != known
                or intent.get("scheduler_user") != scheduler_user
                or intent.get("scheduler_since") != since
                or intent.get("error_classification") != error_classification
                or intent.get("error_summary") != error_summary
                or intent.get("mutation_evidence_basis") != bases
                or not isinstance(before_rows, list)
                or intent_id != _sha256_bytes(_canonical_json(stable))
            ):
                raise EvidenceError("canary failure seal intent drifted")
            _validate_partially_sealed_inventory(current_rows, before_rows)
        else:
            if evidence.exists() and list(evidence.iterdir()):
                raise EvidenceError(
                    "canary failure evidence exists before marker-first intent"
                )
            intent = {
                "schema_version": 1,
                "protocol": _CANARY_SEAL_INTENT_PROTOCOL,
                "created_at": _utc_now(),
                "tree": str(tree_path),
                "tree_device": os.lstat(tree_path).st_dev,
                "tree_inode": os.lstat(tree_path).st_ino,
                "evidence_root": str(evidence),
                "release": release,
                "known_scheduler_identity": known,
                "scheduler_user": scheduler_user,
                "scheduler_since": since,
                "error_classification": error_classification,
                "error_summary": error_summary,
                "error_summary_sha256": _sha256_bytes(error_summary.encode("utf-8")),
                "mutation_claim": "bound_to_preexisting_evidence_not_inferred_from_absence",
                "mutation_evidence_basis": bases,
                "preseal_inventory": current_rows,
                "preseal_inventory_sha256": _sha256_bytes(
                    _inventory_payload(current_rows)
                ),
                "file_count": file_count,
                "total_bytes": total_bytes,
            }
            intent["intent_id"] = _sha256_bytes(_canonical_json(intent))
            before_rows = current_rows
            if apply:
                _canary_atomic_json(
                    artifacts["intent"],
                    intent,
                    evidence_root=evidence,
                    artifacts=artifacts,
                )

        report = {
            "status": "dry_run",
            "tree": str(tree_path),
            "evidence_root": str(evidence),
            "intent": str(artifacts["intent"]),
            "marker": str(artifacts["marker"]),
            "error_classification": error_classification,
            "known_job_ids": known["job_ids"],
            "file_count": intent["file_count"],
            "total_bytes": intent["total_bytes"],
            "recoverable_staging_entries": staged_entries,
        }
        if not apply:
            return report

        before_payload = _inventory_payload(before_rows)
        _write_or_verify_canary_bytes(
            artifacts["before_inventory"],
            before_payload,
            description="canary preseal inventory",
            evidence_root=evidence,
            artifacts=artifacts,
        )

        if artifacts["scheduler_pre"].exists() or artifacts["scheduler_pre"].is_symlink():
            scheduler_pre = _strict_json_artifact(
                artifacts["scheduler_pre"], description="preseal scheduler evidence"
            )
            _validate_scheduler_evidence(
                scheduler_pre,
                scheduler_user=scheduler_user,
                since=since,
                known=known,
                phase="preseal",
            )
        else:
            scheduler_pre = _capture_canary_scheduler_evidence(
                scheduler_user=scheduler_user,
                since=since,
                known=known,
                phase="preseal",
                runner=runner,
            )
            _canary_atomic_json(
                artifacts["scheduler_pre"],
                scheduler_pre,
                evidence_root=evidence,
                artifacts=artifacts,
            )

        _remove_write_bits(tree_path)
        sealed_rows, sealed_count, sealed_bytes = _full_tree_inventory(tree_path)
        _validate_partially_sealed_inventory(sealed_rows, before_rows)
        if any(int(row["mode"]) & 0o222 for row in sealed_rows):
            raise EvidenceError("partial canary tree remains writable after sealing")
        if (
            sealed_count != intent["file_count"]
            or sealed_bytes != intent["total_bytes"]
        ):
            raise EvidenceError("partial canary cardinality changed while sealing")

        if artifacts["scheduler_post"].exists() or artifacts["scheduler_post"].is_symlink():
            scheduler_post = _strict_json_artifact(
                artifacts["scheduler_post"], description="postseal scheduler evidence"
            )
            _validate_scheduler_evidence(
                scheduler_post,
                scheduler_user=scheduler_user,
                since=since,
                known=known,
                phase="postseal",
            )
        else:
            scheduler_post = _capture_canary_scheduler_evidence(
                scheduler_user=scheduler_user,
                since=since,
                known=known,
                phase="postseal",
                runner=runner,
            )
            _canary_atomic_json(
                artifacts["scheduler_post"],
                scheduler_post,
                evidence_root=evidence,
                artifacts=artifacts,
            )
        after_payload = _inventory_payload(sealed_rows)
        _write_or_verify_canary_bytes(
            artifacts["after_inventory"],
            after_payload,
            description="sealed canary inventory",
            evidence_root=evidence,
            artifacts=artifacts,
        )
        _validate_or_recover_canary_staging(
            evidence, artifacts, recover=True
        )
        _validate_canary_evidence_file_set(
            evidence,
            artifacts,
            expected_prefix_length=(
                len(_CANARY_FAILURE_ARTIFACT_ORDER)
                if artifacts["marker"].exists()
                or artifacts["marker"].is_symlink()
                else len(_CANARY_FAILURE_ARTIFACT_ORDER) - 1
            ),
            require_directory_read_only=False,
        )

        marker_core = {
            "schema_version": 1,
            "protocol": _CANARY_SEAL_PROTOCOL,
            "passed": True,
            "classification": "deterministic_canary_failure_sealed_fail_closed",
            "retry_in_place": False,
            "tree": str(tree_path),
            "tree_device": intent["tree_device"],
            "tree_inode": intent["tree_inode"],
            "release": release,
            "error_classification": error_classification,
            "error_summary": error_summary,
            "mutation_claim": intent["mutation_claim"],
            "mutation_evidence_basis": bases,
            "known_scheduler_identity": known,
            "intent": str(artifacts["intent"]),
            "intent_sha256": _sha256_file(artifacts["intent"]),
            "preseal_inventory": str(artifacts["before_inventory"]),
            "preseal_inventory_sha256": _sha256_bytes(before_payload),
            "sealed_inventory": str(artifacts["after_inventory"]),
            "sealed_inventory_sha256": _sha256_bytes(after_payload),
            "scheduler_pre": str(artifacts["scheduler_pre"]),
            "scheduler_pre_sha256": _sha256_file(artifacts["scheduler_pre"]),
            "scheduler_post": str(artifacts["scheduler_post"]),
            "scheduler_post_sha256": _sha256_file(artifacts["scheduler_post"]),
            "file_count": sealed_count,
            "total_bytes": sealed_bytes,
            "no_active_matching_jobs_preseal": True,
            "no_active_matching_jobs_postseal": True,
        }
        if artifacts["marker"].exists() or artifacts["marker"].is_symlink():
            marker = _strict_json_artifact(
                artifacts["marker"], description="canary failure seal marker"
            )
            stable_marker = dict(marker)
            seal_id = stable_marker.pop("seal_id", None)
            sealed_at = stable_marker.pop("sealed_at", None)
            if (
                stable_marker != marker_core
                or not isinstance(sealed_at, str)
                or seal_id
                != _sha256_bytes(
                    _canonical_json(marker_core | {"sealed_at": sealed_at})
                )
            ):
                raise EvidenceError("canary failure seal marker drifted")
            _validate_or_recover_canary_staging(
                evidence, artifacts, recover=True
            )
            _validate_canary_evidence_file_set(
                evidence,
                artifacts,
                expected_prefix_length=len(_CANARY_FAILURE_ARTIFACT_ORDER),
                require_directory_read_only=False,
            )
            if stat.S_IMODE(evidence.stat().st_mode) & 0o222:
                os.chmod(evidence, stat.S_IMODE(evidence.stat().st_mode) & ~0o222)
                _fsync_directory(evidence.parent)
            _validate_canary_evidence_file_set(
                evidence,
                artifacts,
                expected_prefix_length=len(_CANARY_FAILURE_ARTIFACT_ORDER),
                require_directory_read_only=True,
            )
            # Re-query current truth on idempotent verification without rewriting
            # the original pre/post evidence envelope.
            _capture_canary_scheduler_evidence(
                scheduler_user=scheduler_user,
                since=since,
                known=known,
                phase="verification",
                runner=runner,
            )
            return marker | {"status": "already_sealed"}

        marker = marker_core | {"sealed_at": _utc_now()}
        marker["seal_id"] = _sha256_bytes(_canonical_json(marker))
        _canary_atomic_json(
            artifacts["marker"],
            marker,
            evidence_root=evidence,
            artifacts=artifacts,
        )
        _validate_or_recover_canary_staging(
            evidence, artifacts, recover=True
        )
        _validate_canary_evidence_file_set(
            evidence,
            artifacts,
            expected_prefix_length=len(_CANARY_FAILURE_ARTIFACT_ORDER),
            require_directory_read_only=False,
        )
        os.chmod(evidence, stat.S_IMODE(evidence.stat().st_mode) & ~0o222)
        _fsync_directory(evidence.parent)
        _validate_canary_evidence_file_set(
            evidence,
            artifacts,
            expected_prefix_length=len(_CANARY_FAILURE_ARTIFACT_ORDER),
            require_directory_read_only=True,
        )
        return marker | {"status": "sealed"}


def _pretty_canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _stable_file_bytes(
    path: Path,
    *,
    description: str,
    require_read_only: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Read one regular singly-linked file through a stable no-follow fd."""

    path = _safe_existing_path(path, description=description, kind="file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    blocks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (require_read_only and stat.S_IMODE(before.st_mode) & 0o222)
        ):
            raise EvidenceError(
                f"{description} is mutable, linked, or not a regular file: {path}"
            )
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            blocks.append(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise EvidenceError(f"{description} changed while being read: {path}")
    raw = b"".join(blocks)
    return raw, {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size": before.st_size,
    }


def _decode_unique_json(
    raw: bytes,
    *,
    description: str,
) -> dict[str, Any]:
    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise EvidenceError(
                    f"{description} duplicates JSON key {key!r}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EvidenceError(
                    f"{description} contains non-finite value {token}"
                )
            ),
        )
    except EvidenceError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceError(f"{description} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{description} must be one JSON object")
    return value


def _read_canonical_failure_envelope(
    path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    description = "r3 deterministic prelaunch failure envelope"
    raw, record = _stable_file_bytes(path, description=description)
    value = _decode_unique_json(raw, description=description)
    if raw != _canonical_json(value):
        raise EvidenceError(f"{description} is not compact canonical JSON")
    return value, raw, record


def _require_sha256(value: Any, *, description: str) -> str:
    text = str(value)
    if _SHA256_RE.fullmatch(text) is None:
        raise EvidenceError(f"{description} is not a SHA-256 digest")
    return text


def _canonical_absolute_text(value: Any, *, description: str) -> str:
    if not isinstance(value, str):
        raise EvidenceError(f"{description} must be a path string")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != os.path.normpath(str(path))
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise EvidenceError(f"{description} is not a canonical absolute path")
    return value


def _r3_probe_expected_environment(
    temporary_root: Path, *, classification: str
) -> dict[str, str]:
    """Return the only environment accepted by one historical r3 probe."""

    common = {
        "HOME": str(temporary_root / "home"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TEMP": str(temporary_root / "tmp"),
        "TMP": str(temporary_root / "tmp"),
        "TMPDIR": str(temporary_root / "tmp"),
        "XDG_CACHE_HOME": str(temporary_root / "xdg-cache"),
        "XDG_CONFIG_HOME": str(temporary_root / "xdg-config"),
        "XDG_DATA_HOME": str(temporary_root / "xdg-data"),
        "XDG_STATE_HOME": str(temporary_root / "xdg-state"),
    }
    if classification == "unsafe_recorded_broken_internal_symlink":
        return common
    if classification != "offline_clone_unseeded_release_local_cache":
        raise EvidenceError("unsupported r3 prelaunch failure classification")
    return {
        **common,
        "CONDA_ENVS_PATH": str(temporary_root / "conda-envs"),
        "CONDA_NO_PLUGINS": "true",
        "CONDA_OFFLINE": "true",
        "CONDA_PIP_INTEROP_ENABLED": "false",
        "CONDA_PKGS_DIRS": str(temporary_root / "empty-conda-pkgs"),
    }


def _validated_r3_probe_contract(
    *,
    classification: str,
    argv: Sequence[str],
    cwd: str | Path,
    environment: Mapping[str, str],
    input_paths: Sequence[tuple[str, str | Path]],
    write_roots: Sequence[str | Path],
    output_path: str | Path | None,
) -> dict[str, Any]:
    """Validate the immutable command/path contract without consulting live state.

    This routine is shared by the producer and the self-contained sealed-envelope
    verifier.  It deliberately validates lexical canonical paths rather than resolving
    them: after archival, the external r3/r10 checkouts and toolchain need not remain
    available.
    """

    if classification not in _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS:
        raise EvidenceError("unsupported r3 prelaunch failure classification")
    arguments = list(argv)
    if (
        not arguments
        or any(
            not isinstance(argument, str)
            or not argument
            or any(character in argument for character in ("\x00", "\n", "\r"))
            for argument in arguments
        )
    ):
        raise EvidenceError("r3 prelaunch probe command is invalid")
    cwd_text = _canonical_absolute_text(
        str(cwd), description="r3 prelaunch probe cwd"
    )
    r3_checkout = Path(cwd_text)
    if (
        r3_checkout.name != _R3_RELEASE_CHECKOUT_DIRECTORY
        or r3_checkout.parent.name != "schema5-v1"
    ):
        raise EvidenceError(
            "r3 prelaunch probe cwd is not the canonical tagged r3 checkout"
        )
    recovery_root = r3_checkout.parent
    r10_checkout = recovery_root / _R10_RELEASE_CHECKOUT_DIRECTORY
    toolchain_root = recovery_root / _R10_TOOLCHAIN_RELATIVE_ROOT
    toolchain_base = toolchain_root / "base"
    r10_python = toolchain_base / "bin/python"
    r10_conda = toolchain_base / "bin/conda"
    r3_runtime_identity = (
        r3_checkout / _R3_RUNTIME_IDENTITY_RELATIVE_PATH
    )

    root_texts = [
        _canonical_absolute_text(
            str(value), description="r3 prelaunch probe temporary write root"
        )
        for value in write_roots
    ]
    if len(root_texts) != 1:
        raise EvidenceError(
            "r3 prelaunch probe requires exactly one temporary write root"
        )
    temporary_root = Path(root_texts[0])
    if (
        temporary_root == Path("/tmp")
        or not temporary_root.is_relative_to("/tmp")
    ):
        raise EvidenceError(
            "r3 prelaunch probe temporary root must be a private child of /tmp"
        )

    expected_environment = _r3_probe_expected_environment(
        temporary_root, classification=classification
    )
    if dict(environment) != expected_environment:
        expected_keys = (
            _R3_PROBE_COMMON_ENVIRONMENT_KEYS
            if classification == "unsafe_recorded_broken_internal_symlink"
            else _R3_PROBE_OFFLINE_ENVIRONMENT_KEYS
        )
        if set(environment) != set(expected_keys):
            raise EvidenceError(
                "r3 prelaunch probe environment has extra or missing keys"
            )
        raise EvidenceError(
            "r3 prelaunch probe environment paths or fixed values drifted"
        )

    expected_names = _R3_PROBE_INPUT_NAMES[classification]
    supplied_names = tuple(name for name, _path in input_paths)
    if supplied_names != expected_names:
        raise EvidenceError(
            "r3 prelaunch probe input names or order drifted"
        )
    supplied_paths: dict[str, Path] = {}
    for name, raw_path in input_paths:
        text = _canonical_absolute_text(
            str(raw_path), description=f"r3 prelaunch probe input {name}"
        )
        supplied_paths[name] = Path(text)
    expected_paths = {
        "tagged-r3-release-checkout": r3_checkout,
        "tagged-r10-release-checkout": r10_checkout,
        "sealed-r10-conda-toolchain": toolchain_root,
    }
    if classification == "unsafe_recorded_broken_internal_symlink":
        expected_paths["recorded-shared-conda-base"] = _R3_SHARED_CONDA_BASE
    if supplied_paths != expected_paths:
        raise EvidenceError(
            "r3 prelaunch probe input paths drifted from canonical release inputs"
        )
    for name, path in supplied_paths.items():
        if (
            path == temporary_root
            or path.is_relative_to(temporary_root)
            or temporary_root.is_relative_to(path)
        ):
            raise EvidenceError(
                f"r3 prelaunch temporary output overlaps immutable input {name}"
            )

    if classification == "unsafe_recorded_broken_internal_symlink":
        expected_argv = [
            str(r10_python),
            "-I",
            "-B",
            str(r3_runtime_identity),
            "--conda-executable",
            str(_R3_SHARED_CONDA_EXECUTABLE),
        ]
    else:
        expected_argv = [
            str(r10_conda),
            "create",
            "--yes",
            "--offline",
            "--clone",
            str(toolchain_base),
            "--prefix",
            str(temporary_root / "offline-clone-destination"),
        ]
    if arguments != expected_argv:
        raise EvidenceError(
            "r3 prelaunch probe argv is not the canonical classification command"
        )

    if output_path is not None:
        output_text = _canonical_absolute_text(
            str(output_path), description="r3 prelaunch failure output"
        )
        expected_output = (
            temporary_root / _R3_PROBE_ENVELOPE_FILENAMES[classification]
        )
        if Path(output_text) != expected_output:
            raise EvidenceError(
                "r3 prelaunch failure envelope is outside its canonical "
                "temporary output path"
            )

    return {
        "recovery_root": recovery_root,
        "r3_checkout": r3_checkout,
        "r10_checkout": r10_checkout,
        "r3_runtime_identity": r3_runtime_identity,
        "toolchain_root": toolchain_root,
        "toolchain_base": toolchain_base,
        "r10_python": r10_python,
        "r10_conda": r10_conda,
        "temporary_root": temporary_root,
        "offline_cache": temporary_root / "empty-conda-pkgs",
        "offline_destination": temporary_root / "offline-clone-destination",
    }


def _validate_r3_probe_failure_signature(
    *,
    classification: str,
    returncode: int,
    stdout: str,
    stderr: str,
    contract: Mapping[str, Any],
) -> None:
    """Reject a generic failure that did not exercise the pinned defect."""

    if classification == "unsafe_recorded_broken_internal_symlink":
        broken = (
            _R3_SHARED_CONDA_BASE / _R3_BROKEN_SYMLINK_PATH
        )
        expected_stderr = (
            "ERROR: "
            f"unsafe Conda runtime symlink {broken}: [Errno 2] "
            "No such file or directory: "
            f"'{_R3_BROKEN_SYMLINK_MISSING_TARGET}'\n"
        )
        if (
            returncode != 2
            or stdout != ""
            or stderr != expected_stderr
        ):
            raise EvidenceError(
                "r3 broken-symlink probe lacks the canonical runtime-identity "
                "error signature"
            )
        return

    expected_progress_prefix = (
        f"Source:      {contract['toolchain_base']}\n"
        f"Destination: {contract['offline_destination']}\n"
        "Packages: 89\n"
        "Files: 3\n\n"
        "Downloading and Extracting Packages: ...working..."
    )
    progress_suffix = " done\n"
    progress_body = (
        stdout[len(expected_progress_prefix) : -len(progress_suffix)]
        if stdout.startswith(expected_progress_prefix)
        and stdout.endswith(progress_suffix)
        else ""
    )
    progress_output = (
        len(stdout.encode("utf-8")) <= 262_144
        and stdout.startswith(expected_progress_prefix)
        and stdout.endswith(progress_suffix)
        and bool(progress_body)
        and re.fullmatch(
            r"[\x1b\r\n A-Za-z0-9_.+%|()[\];-]+",
            progress_body,
        )
        is not None
        and "http://" not in stdout
        and "https://" not in stdout
        and "OfflineError" not in stdout
        and "Traceback" not in stdout
    )
    remote_block = re.compile(
        r"OfflineError: EnforceUnusedAdapter called with url "
        r"(?P<url>https://conda[.]anaconda[.]org/conda-forge/"
        r"(?:linux-64|noarch)/"
        r"[A-Za-z0-9_.+%-]+(?:[.]conda|[.]tar[.]bz2))[.]\n"
        r"This command is using a remote connection in offline mode[.]\n"
    )
    remote_text = stderr[1:] if stderr.startswith("\n") else stderr
    remote_urls: list[str] = []
    offset = 0
    while offset < len(remote_text):
        match = remote_block.match(remote_text, offset)
        if match is None:
            remote_urls = []
            break
        remote_urls.append(match.group("url"))
        offset = match.end()
    remote_context = (
        bool(remote_urls)
        and offset == len(remote_text)
        and len(remote_urls) <= 512
        and len(stderr.encode("utf-8")) <= 524_288
    )
    if (
        classification != "offline_clone_unseeded_release_local_cache"
        or returncode != 1
        or not progress_output
        or not remote_context
    ):
        raise EvidenceError(
            "r3 offline-cache probe lacks the canonical Conda OfflineError "
            "remote-fetch/progress context"
        )


def _validate_r3_probe_immutable_inputs(
    *,
    contract: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    """Reverify the exact r3 tag and sealed r10 producer/toolchain before execution."""

    recovery_root = Path(contract["recovery_root"])
    r3_checkout = _safe_existing_path(
        contract["r3_checkout"],
        description="tagged r3 probe checkout",
        kind="directory",
    )
    r10_checkout = _safe_existing_path(
        contract["r10_checkout"],
        description="tagged r10 probe checkout",
        kind="directory",
    )
    _validate_r3_durable_release_identity(
        marker_path=recovery_root / _R3_DURABLE_RELEASE_MARKER,
        release_checkout=r3_checkout,
    )
    exact_r3_files = {
        _R3_PILOT_RELATIVE_PATH: _R3_PILOT_SHA256,
        _R3_RUNTIME_IDENTITY_RELATIVE_PATH: _R3_RUNTIME_IDENTITY_SHA256,
    }
    for relative, expected_sha256 in exact_r3_files.items():
        path = _safe_existing_path(
            r3_checkout / relative,
            description=f"tagged r3 probe source {relative}",
            kind="file",
        )
        if _sha256_file(path) != expected_sha256:
            raise EvidenceError(
                f"tagged r3 probe source identity drifted: {relative}"
            )

    r10_tag_ref = "refs/tags/sweep-recovery-schema5-v1.2-r10"
    if (
        _run_git(r10_checkout, "cat-file", "-t", r10_tag_ref) != "tag"
        or _run_git(r10_checkout, "rev-parse", "--verify", "HEAD")
        != _run_git(
            r10_checkout,
            "rev-parse",
            "--verify",
            f"{r10_tag_ref}^{{commit}}",
        )
        or _run_git(
            r10_checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
    ):
        raise EvidenceError("r10 probe producer checkout is not the exact clean tag")
    expected_sealer = r10_checkout / _R10_SEALER_RELATIVE_PATH
    if Path(__file__).resolve() != expected_sealer:
        raise EvidenceError(
            "r3 prelaunch probe producer is not the tagged r10 sealer"
        )
    provisioner = _safe_existing_path(
        r10_checkout / _R10_TOOLCHAIN_PROVISIONER_RELATIVE_PATH,
        description="tagged r10 toolchain verifier",
        kind="file",
    )
    if Path(conda_toolchain.__file__).resolve() != provisioner:
        raise EvidenceError(
            "r3 prelaunch probe did not import the tagged r10 toolchain verifier"
        )
    toolchain_root = _safe_existing_path(
        contract["toolchain_root"],
        description="sealed r10 Conda toolchain",
        kind="directory",
    )
    python_lexical = Path(contract["r10_python"])
    try:
        python_resolved = python_lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceError(
            f"sealed r10 Python is missing or unsafe: {python_lexical}: {exc}"
        ) from exc
    if (
        not python_resolved.is_file()
        or not python_resolved.is_relative_to(Path(contract["toolchain_base"]))
    ):
        raise EvidenceError("sealed r10 Python escapes its verified toolchain base")

    try:
        verified = conda_toolchain.verified_conda_toolchain_binding(
            toolchain_root,
            exercise=False,
        )
    except (
        OSError,
        conda_toolchain.CondaToolchainProvisionError,
    ) as exc:
        raise EvidenceError(
            f"sealed r10 Conda toolchain verification failed: {exc}"
        ) from exc
    expected_binding_fields = {
        "schema_version",
        "protocol",
        "release_tag",
        "chain_namespace",
        "toolchain_root",
        "base_prefix",
        "portable_shebang",
        "completion_marker",
        "marker_id",
        "installer_contract",
        "intent_id",
        "conda_executable",
        "runtime_identity_sha256",
        "complete_prefix_inventory_sha256",
        "read_only_probes",
        "binding_id",
    }
    executable = (
        verified.get("conda_executable")
        if isinstance(verified, dict)
        else None
    )
    portable_shebang = (
        verified.get("portable_shebang")
        if isinstance(verified, dict)
        else None
    )
    if (
        not isinstance(verified, dict)
        or set(verified) != expected_binding_fields
        or verified.get("schema_version") != conda_toolchain.SCHEMA_VERSION
        or verified.get("protocol") != conda_toolchain.PROTOCOL
        or verified.get("release_tag") != "sweep-recovery-schema5-v1.2-r10"
        or verified.get("chain_namespace") != "schema5-v1.2-r10"
        or verified.get("toolchain_root") != str(toolchain_root)
        or verified.get("base_prefix") != str(contract["toolchain_base"])
        or not isinstance(portable_shebang, dict)
        or set(portable_shebang)
        != {
            "absolute_base_prefix_interpreter_required",
            "interpreter",
            "maximum_shebang_bytes",
            "shebang_bytes",
        }
        or portable_shebang.get("absolute_base_prefix_interpreter_required")
        is not True
        or portable_shebang.get("interpreter") != str(contract["r10_python"])
        or portable_shebang.get("maximum_shebang_bytes")
        != conda_toolchain.MAX_PORTABLE_SHEBANG_BYTES
        or not isinstance(portable_shebang.get("shebang_bytes"), int)
        or isinstance(portable_shebang.get("shebang_bytes"), bool)
        or portable_shebang["shebang_bytes"] <= 0
        or portable_shebang["shebang_bytes"]
        > conda_toolchain.MAX_PORTABLE_SHEBANG_BYTES
        or not isinstance(executable, dict)
        or executable.get("path") != str(contract["r10_conda"])
        or _SHA256_RE.fullmatch(str(executable.get("sha256", ""))) is None
        or _SHA256_RE.fullmatch(
            str(verified.get("binding_id", ""))
        )
        is None
    ):
        raise EvidenceError(
            "sealed r10 Conda toolchain verification identity drifted"
        )


def _validate_r3_prelaunch_failure_envelope(
    path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    payload, raw, source_record = _read_canonical_failure_envelope(path)
    expected_fields = {
        "schema_version",
        "protocol",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "classification",
        "deterministic",
        "requires_superseding_release",
        "retry_in_place",
        "pre_scheduler_submission",
        "scheduler_job_ids",
        "command",
        "input_inventories",
        "write_scope",
        "observations",
        "observed_at",
        "failure_id",
    }
    identity = dict(payload)
    failure_id = identity.pop("failure_id", None)
    classification = payload.get("classification")
    if (
        set(payload) != expected_fields
        or payload.get("schema_version") != 1
        or payload.get("protocol")
        != _R3_PRELAUNCH_FAILURE_ENVELOPE_PROTOCOL
        or payload.get("release_id") != _R3_RELEASE_ID
        or payload.get("release_tag") != _R3_RELEASE_TAG
        or payload.get("release_git_commit") != _R3_RELEASE_COMMIT
        or payload.get("release_tag_object") != _R3_RELEASE_TAG_OBJECT
        or classification not in _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS
        or payload.get("deterministic") is not True
        or payload.get("requires_superseding_release") is not True
        or payload.get("retry_in_place") is not False
        or payload.get("pre_scheduler_submission") is not True
        or payload.get("scheduler_job_ids") != []
        or not isinstance(payload.get("observed_at"), str)
        or not payload["observed_at"]
        or failure_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise EvidenceError(
            "r3 deterministic prelaunch failure identity is invalid"
        )

    command = payload.get("command")
    command_fields = {
        "argv",
        "cwd",
        "environment",
        "returncode",
        "stdout",
        "stderr",
        "stdout_sha256",
        "stderr_sha256",
    }
    if not isinstance(command, dict) or set(command) != command_fields:
        raise EvidenceError("r3 prelaunch failure command envelope is invalid")
    argv = command.get("argv")
    environment = command.get("environment")
    if (
        not isinstance(argv, list)
        or not argv
        or any(
            not isinstance(argument, str)
            or not argument
            or any(character in argument for character in ("\x00", "\n", "\r"))
            for argument in argv
        )
        or not isinstance(environment, dict)
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or any(character in key for character in ("\x00", "=", "\n", "\r"))
            or any(character in value for character in ("\x00", "\n", "\r"))
            for key, value in environment.items()
        )
        or not isinstance(command.get("returncode"), int)
        or isinstance(command.get("returncode"), bool)
        or command["returncode"] == 0
        or not isinstance(command.get("stdout"), str)
        or not isinstance(command.get("stderr"), str)
        or command.get("stdout_sha256")
        != _sha256_bytes(command["stdout"].encode("utf-8"))
        or command.get("stderr_sha256")
        != _sha256_bytes(command["stderr"].encode("utf-8"))
    ):
        raise EvidenceError("r3 prelaunch failure command fields drifted")
    _canonical_absolute_text(command["cwd"], description="failure command cwd")
    forbidden_scheduler_commands = {"sbatch", "scancel", "scontrol", "srun"}
    if (
        Path(argv[0]).name in forbidden_scheduler_commands
        or any(Path(argument).name in forbidden_scheduler_commands for argument in argv)
        or (
            Path(argv[0]).name in {"bash", "dash", "sh", "zsh"}
            and "-c" in argv[1:]
        )
    ):
        raise EvidenceError(
            "prelaunch failure envelope may not contain a scheduler mutation "
            "or opaque shell command"
        )

    inventories = payload.get("input_inventories")
    if not isinstance(inventories, list) or not inventories:
        raise EvidenceError("r3 prelaunch failure lacks input inventories")
    inventory_names: set[str] = set()
    inventory_paths: list[tuple[str, str]] = []
    for row in inventories:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "name",
                "path",
                "before_sha256",
                "after_sha256",
                "entry_count",
                "total_bytes",
                "unchanged",
            }
            or not isinstance(row.get("name"), str)
            or not row["name"]
            or row["name"] in inventory_names
            or not isinstance(row.get("entry_count"), int)
            or isinstance(row.get("entry_count"), bool)
            or row["entry_count"] < 0
            or not isinstance(row.get("total_bytes"), int)
            or isinstance(row.get("total_bytes"), bool)
            or row["total_bytes"] < 0
            or row.get("unchanged") is not True
        ):
            raise EvidenceError("r3 prelaunch input inventory is invalid")
        inventory_names.add(row["name"])
        _canonical_absolute_text(
            row["path"], description=f"{row['name']} inventory path"
        )
        inventory_paths.append((row["name"], row["path"]))
        before = _require_sha256(
            row["before_sha256"],
            description=f"{row['name']} before inventory",
        )
        after = _require_sha256(
            row["after_sha256"],
            description=f"{row['name']} after inventory",
        )
        if before != after:
            raise EvidenceError(
                f"r3 prelaunch input changed during failure probe: {row['name']}"
            )

    write_scope = payload.get("write_scope")
    if (
        not isinstance(write_scope, dict)
        or set(write_scope)
        != {
            "roots",
            "all_within_temporary_storage",
            "results_root_touched",
            "release_checkout_touched",
            "live_prefixes_touched",
        }
        or write_scope.get("all_within_temporary_storage") is not True
        or write_scope.get("results_root_touched") is not False
        or write_scope.get("release_checkout_touched") is not False
        or write_scope.get("live_prefixes_touched") is not False
        or not isinstance(write_scope.get("roots"), list)
        or not write_scope["roots"]
    ):
        raise EvidenceError("r3 prelaunch failure write scope is invalid")
    for root in write_scope["roots"]:
        text = _canonical_absolute_text(
            root, description="prelaunch temporary write root"
        )
        if not (Path(text) == Path("/tmp") or Path(text).is_relative_to("/tmp")):
            raise EvidenceError(
                "r3 prelaunch failure wrote outside temporary storage"
            )

    probe_contract = _validated_r3_probe_contract(
        classification=classification,
        argv=argv,
        cwd=command["cwd"],
        environment=environment,
        input_paths=inventory_paths,
        write_roots=write_scope["roots"],
        output_path=None,
    )
    _validate_r3_probe_failure_signature(
        classification=classification,
        returncode=command["returncode"],
        stdout=command["stdout"],
        stderr=command["stderr"],
        contract=probe_contract,
    )

    observations = payload.get("observations")
    if not isinstance(observations, dict):
        raise EvidenceError("r3 prelaunch failure observations are invalid")
    if classification == "unsafe_recorded_broken_internal_symlink":
        broken = observations.get("broken_symlink")
        if (
            set(observations) != {"broken_symlink", "source_inventory_unchanged"}
            or observations.get("source_inventory_unchanged") is not True
            or not isinstance(broken, dict)
            or set(broken) != {"path", "target", "target_exists"}
            or broken.get("path") != _R3_BROKEN_SYMLINK_PATH
            or broken.get("target") != _R3_BROKEN_SYMLINK_TARGET
            or broken.get("target_exists") is not False
        ):
            raise EvidenceError(
                "r3 broken-symlink prelaunch observation drifted"
            )
    else:
        cache_path = str(probe_contract["offline_cache"])
        if (
            set(observations)
            != {
                "conda_network_disabled",
                "offline_error_detected",
                "release_local_package_cache_initial_file_count",
                "release_local_package_cache_seeded",
                "source_inventory_unchanged",
            }
            or observations.get("conda_network_disabled") is not True
            or observations.get("offline_error_detected") is not True
            or observations.get(
                "release_local_package_cache_initial_file_count"
            )
            != 0
            or observations.get("release_local_package_cache_seeded") is not False
            or observations.get("source_inventory_unchanged") is not True
            or environment.get("CONDA_OFFLINE") != "true"
            or environment.get("CONDA_PKGS_DIRS") != cache_path
        ):
            raise EvidenceError(
                "r3 empty-cache offline-clone observation drifted"
            )
    return payload, raw, source_record


def _prelaunch_input_inventory(
    root: Path, *, excluded_top_level: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Inventory one probe input without following its recorded symlinks."""

    root = _safe_existing_path(
        root, description="r3 prelaunch probe input root", kind="directory"
    )
    rows: list[dict[str, Any]] = []
    file_count = 0
    total_bytes = 0

    def append_entry(path: Path) -> None:
        nonlocal file_count, total_bytes
        metadata = os.lstat(path)
        relative = "." if path == root else path.relative_to(root).as_posix()
        common = {
            "relative_path": relative,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
            "nlink": metadata.st_nlink,
        }
        if stat.S_ISDIR(metadata.st_mode):
            rows.append(common | {"type": "directory", "size": 0})
        elif stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(path)
            rows.append(
                common
                | {
                    "type": "symlink",
                    "size": len(target.encode("utf-8")),
                    "target": target,
                }
            )
        elif stat.S_ISREG(metadata.st_mode):
            record = _stable_prelaunch_input_file_record(
                path, description="r3 prelaunch probe input file"
            )
            if (
                record["device"] != metadata.st_dev
                or record["inode"] != metadata.st_ino
                or record["mode"] != stat.S_IMODE(metadata.st_mode)
                or record["size"] != metadata.st_size
            ):
                raise EvidenceError(
                    f"r3 prelaunch probe input changed during inventory: {path}"
                )
            rows.append(
                common
                | {
                    "type": "file",
                    "size": record["size"],
                    "sha256": record["sha256"],
                }
            )
            file_count += 1
            total_bytes += int(record["size"])
        else:
            raise EvidenceError(
                f"r3 prelaunch probe input contains an unsafe entry: {path}"
            )

    append_entry(root)

    def walk_error(exc: OSError) -> None:
        raise EvidenceError(
            f"cannot traverse r3 prelaunch probe input {root}: {exc}"
        ) from exc

    for directory, directory_names, file_names in os.walk(
        root, followlinks=False, onerror=walk_error
    ):
        directory_names[:] = sorted(directory_names)
        selected_files = sorted(file_names)
        if Path(directory) == root and excluded_top_level:
            directory_names[:] = [
                name for name in directory_names if name not in excluded_top_level
            ]
            selected_files = [
                name for name in selected_files if name not in excluded_top_level
            ]
        for name in [*directory_names, *selected_files]:
            append_entry(Path(directory) / name)
        directory_names[:] = [
            name
            for name in directory_names
            if not (Path(directory) / name).is_symlink()
        ]
    payload = b"".join(_canonical_json(row) for row in rows)
    return {
        "path": str(root),
        "inventory_sha256": _sha256_bytes(payload),
        "entry_count": len(rows),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _r3_probe_input_inventory(name: str, root: Path) -> dict[str, Any]:
    excluded = (
        _R3_SHARED_RUNTIME_EXCLUDED_TOP_LEVEL
        if name == "recorded-shared-conda-base"
        else frozenset()
    )
    return _prelaunch_input_inventory(root, excluded_top_level=excluded)


def _stable_prelaunch_input_file_record(
    path: Path, *, description: str
) -> dict[str, Any]:
    """Hash one input file while preserving, rather than rejecting, hardlink topology."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot open {description} {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            raise EvidenceError(
                f"{description} is not a linked regular file: {path}"
            )
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise EvidenceError(f"{description} changed while being hashed: {path}")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size": before.st_size,
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": stat.S_IMODE(before.st_mode),
        "nlink": before.st_nlink,
    }


def _normalized_probe_environment(
    entries: Sequence[tuple[str, str]],
) -> dict[str, str]:
    environment: dict[str, str] = {}
    for key, value in entries:
        if (
            not key
            or key in environment
            or any(character in key for character in ("\x00", "=", "\n", "\r"))
            or any(character in value for character in ("\x00", "\n", "\r"))
        ):
            raise EvidenceError("r3 prelaunch probe environment is invalid")
        environment[key] = value
    return environment


def _normalized_probe_roots(
    entries: Sequence[tuple[str, str | Path]],
) -> list[tuple[str, Path]]:
    values: list[tuple[str, Path]] = []
    names: set[str] = set()
    paths: set[Path] = set()
    for name, raw_path in entries:
        if (
            not name
            or name in names
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None
        ):
            raise EvidenceError("r3 prelaunch probe input name is invalid")
        path = _safe_existing_path(
            raw_path,
            description=f"r3 prelaunch probe input {name}",
            kind="directory",
        )
        if path in paths:
            raise EvidenceError("duplicate r3 prelaunch probe input root")
        names.add(name)
        paths.add(path)
        values.append((name, path))
    if not values:
        raise EvidenceError("r3 prelaunch probe requires input roots")
    return values


def _normalized_temporary_write_roots(
    entries: Sequence[str | Path],
) -> list[str]:
    values: list[str] = []
    for entry in entries:
        text = _canonical_absolute_text(
            str(entry), description="r3 prelaunch probe temporary write root"
        )
        path = Path(text)
        if not (path == Path("/tmp") or path.is_relative_to("/tmp")):
            raise EvidenceError(
                "r3 prelaunch probe write root is outside /tmp"
            )
        if text in values:
            raise EvidenceError("duplicate r3 prelaunch probe write root")
        values.append(text)
    if not values:
        raise EvidenceError("r3 prelaunch probe requires temporary write roots")
    return values


def record_prelaunch_attempt(
    *,
    output: str | Path,
    classification: str,
    command: Sequence[str],
    cwd: str | Path,
    environment: Sequence[tuple[str, str]],
    input_roots: Sequence[tuple[str, str | Path]],
    write_roots: Sequence[str | Path],
    apply: bool = False,
    timeout_seconds: float = 600.0,
    observed_at: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
) -> dict[str, Any]:
    """Run and record one deterministic r3 prelaunch failure probe.

    The command is executed directly, never through a shell, with exactly the
    supplied environment.  Every declared input root is inventoried immediately
    before and after the command and must remain byte/inode identical.  Probe writes
    are declared below ``/tmp``; the canonical failure envelope is published
    read-only only after the expected deterministic failure is observed.
    """

    output_path = _absolute(output, description="r3 prelaunch failure output")
    _safe_existing_path(
        output_path.parent,
        description="r3 prelaunch failure output parent",
        kind="directory",
    )
    if classification not in _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS:
        raise EvidenceError("unsupported r3 prelaunch failure classification")
    argv = list(command)
    if (
        not argv
        or any(
            not isinstance(argument, str)
            or not argument
            or any(character in argument for character in ("\x00", "\n", "\r"))
            for argument in argv
        )
    ):
        raise EvidenceError("r3 prelaunch probe command is invalid")
    cwd_path = _safe_existing_path(
        cwd, description="r3 prelaunch probe cwd", kind="directory"
    )
    command_environment = _normalized_probe_environment(environment)
    probe_inputs = _normalized_probe_roots(input_roots)
    temporary_roots = _normalized_temporary_write_roots(write_roots)
    probe_contract = _validated_r3_probe_contract(
        classification=classification,
        argv=argv,
        cwd=cwd_path,
        environment=command_environment,
        input_paths=probe_inputs,
        write_roots=temporary_roots,
        output_path=output_path,
    )
    temporary_root = _safe_existing_path(
        probe_contract["temporary_root"],
        description="r3 prelaunch private temporary root",
        kind="directory",
    )
    temporary_metadata = os.lstat(temporary_root)
    if (
        temporary_metadata.st_uid != os.getuid()
        or stat.S_IMODE(temporary_metadata.st_mode) & 0o022
    ):
        raise EvidenceError(
            "r3 prelaunch private temporary root has unsafe ownership or mode"
        )
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
        or timeout_seconds > 3_600
    ):
        raise EvidenceError("r3 prelaunch probe timeout is invalid")

    before_verification = {
        name: _r3_probe_input_inventory(name, path)
        for name, path in probe_inputs
    }
    _validate_r3_probe_immutable_inputs(
        contract=probe_contract,
        environment=command_environment,
    )
    after_verification = {
        name: _r3_probe_input_inventory(name, path)
        for name, path in probe_inputs
    }
    if before_verification != after_verification:
        raise EvidenceError(
            "r3 prelaunch immutable-input verification mutated a probe input"
        )

    if output_path.exists() or output_path.is_symlink():
        payload, _raw, _record = _validate_r3_prelaunch_failure_envelope(
            output_path
        )
        current = after_verification
        recorded = {
            row["name"]: row for row in payload["input_inventories"]
        }
        if (
            payload["classification"] != classification
            or payload["command"]["argv"] != argv
            or payload["command"]["cwd"] != str(cwd_path)
            or payload["command"]["environment"] != command_environment
            or payload["write_scope"]["roots"] != temporary_roots
            or set(recorded) != set(current)
            or any(
                recorded[name]["after_sha256"]
                != current[name]["inventory_sha256"]
                or recorded[name]["entry_count"]
                != current[name]["entry_count"]
                or recorded[name]["total_bytes"]
                != current[name]["total_bytes"]
                for name in current
            )
        ):
            raise EvidenceError("existing r3 prelaunch failure envelope drifted")
        return payload | {"status": "already_recorded"}

    before = after_verification
    offline_cache: Path | None = None
    offline_initial_file_count: int | None = None
    if classification == "offline_clone_unseeded_release_local_cache":
        offline_cache = _safe_existing_path(
            probe_contract["offline_cache"],
            description="initially empty offline probe package cache",
            kind="directory",
        )
        offline_initial_file_count = sum(
            1
            for candidate in offline_cache.rglob("*")
            if candidate.is_file() and not candidate.is_symlink()
        )
        if any(offline_cache.iterdir()) or offline_initial_file_count != 0:
            raise EvidenceError(
                "offline probe package cache is not initially empty"
            )
        offline_destination = Path(probe_contract["offline_destination"])
        if offline_destination.exists() or offline_destination.is_symlink():
            raise EvidenceError(
                "offline probe destination is not initially absent"
            )
    plan = {
        "status": "dry_run",
        "output": str(output_path),
        "classification": classification,
        "command": argv,
        "cwd": str(cwd_path),
        "environment": command_environment,
        "input_inventories_before": before,
        "temporary_write_roots": temporary_roots,
        "timeout_seconds": float(timeout_seconds),
    }
    if not apply:
        return plan

    forbidden_scheduler_commands = {"sbatch", "scancel", "scontrol", "srun"}
    if (
        Path(argv[0]).name in forbidden_scheduler_commands
        or any(Path(argument).name in forbidden_scheduler_commands for argument in argv)
        or (
            Path(argv[0]).name in {"bash", "dash", "sh", "zsh"}
            and "-c" in argv[1:]
        )
    ):
        raise EvidenceError(
            "r3 prelaunch probe may not execute a scheduler mutation or shell"
        )
    if runner is None:
        run = subprocess.run
    else:
        run = runner
    try:
        process = run(
            argv,
            cwd=cwd_path,
            env=command_environment,
            capture_output=True,
            text=False,
            check=False,
            timeout=float(timeout_seconds),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EvidenceError(f"r3 prelaunch probe execution failed: {exc}") from exc
    if (
        not isinstance(process, subprocess.CompletedProcess)
        or not isinstance(process.returncode, int)
        or isinstance(process.returncode, bool)
        or not isinstance(process.stdout, bytes)
        or not isinstance(process.stderr, bytes)
    ):
        raise EvidenceError("r3 prelaunch probe returned an invalid process result")
    if process.returncode == 0:
        raise EvidenceError(
            "r3 prelaunch probe unexpectedly succeeded; no failure was recorded"
        )
    try:
        stdout = process.stdout.decode("utf-8")
        stderr = process.stderr.decode("utf-8")
    except UnicodeError as exc:
        raise EvidenceError(
            "r3 prelaunch probe output is not exact UTF-8"
        ) from exc
    _validate_r3_probe_failure_signature(
        classification=classification,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        contract=probe_contract,
    )

    after = {
        name: _r3_probe_input_inventory(name, path)
        for name, path in probe_inputs
    }
    changed = sorted(
        name
        for name in before
        if before[name] != after[name]
    )
    if changed:
        raise EvidenceError(
            f"r3 prelaunch probe mutated its input roots: {changed}"
        )
    inventories = [
        {
            "name": name,
            "path": before[name]["path"],
            "before_sha256": before[name]["inventory_sha256"],
            "after_sha256": after[name]["inventory_sha256"],
            "entry_count": before[name]["entry_count"],
            "total_bytes": before[name]["total_bytes"],
            "unchanged": True,
        }
        for name, _path in probe_inputs
    ]

    if classification == "unsafe_recorded_broken_internal_symlink":
        shared_base = dict(probe_inputs)["recorded-shared-conda-base"]
        link = shared_base / _R3_BROKEN_SYMLINK_PATH
        if not link.is_symlink():
            raise EvidenceError(
                "r3 broken-symlink probe did not identify the exact known link"
            )
        target = os.readlink(link)
        target_path = link.parent / target
        observations = {
            "broken_symlink": {
                "path": _R3_BROKEN_SYMLINK_PATH,
                "target": target,
                "target_exists": target_path.exists()
                or target_path.is_symlink(),
            },
            "source_inventory_unchanged": True,
        }
    else:
        assert offline_cache is not None
        assert offline_initial_file_count is not None
        observations = {
            "conda_network_disabled": (
                command_environment.get("CONDA_OFFLINE") == "true"
            ),
            "offline_error_detected": True,
            "release_local_package_cache_initial_file_count": (
                offline_initial_file_count
            ),
            "release_local_package_cache_seeded": False,
            "source_inventory_unchanged": True,
        }
    timestamp = _utc_now() if observed_at is None else observed_at
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": _R3_PRELAUNCH_FAILURE_ENVELOPE_PROTOCOL,
        "release_id": _R3_RELEASE_ID,
        "release_tag": _R3_RELEASE_TAG,
        "release_git_commit": _R3_RELEASE_COMMIT,
        "release_tag_object": _R3_RELEASE_TAG_OBJECT,
        "classification": classification,
        "deterministic": True,
        "requires_superseding_release": True,
        "retry_in_place": False,
        "pre_scheduler_submission": True,
        "scheduler_job_ids": [],
        "command": {
            "argv": argv,
            "cwd": str(cwd_path),
            "environment": command_environment,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_sha256": _sha256_bytes(process.stdout),
            "stderr_sha256": _sha256_bytes(process.stderr),
        },
        "input_inventories": inventories,
        "write_scope": {
            "roots": temporary_roots,
            "all_within_temporary_storage": True,
            "results_root_touched": False,
            "release_checkout_touched": False,
            "live_prefixes_touched": False,
        },
        "observations": observations,
        "observed_at": timestamp,
    }
    payload["failure_id"] = _sha256_bytes(_canonical_json(payload))
    # Run the same strict consumer before publication; an unexpected error string,
    # link target, cache state, or source mutation therefore withholds the record.
    temporary_raw = _canonical_json(payload)
    temporary_payload = _decode_unique_json(
        temporary_raw, description="candidate r3 prelaunch failure envelope"
    )
    if temporary_payload != payload:
        raise EvidenceError("candidate r3 prelaunch failure serialization drifted")
    # Validate through a same-directory temporary so the public validator observes
    # precisely the bytes that will become immutable.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".validation",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(temporary_raw)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_r3_prelaunch_failure_envelope(temporary_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    _atomic_bytes(output_path, temporary_raw, mode=0o444)
    sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
    _atomic_bytes(
        sidecar,
        f"{_sha256_bytes(temporary_raw)}  {output_path.name}\n".encode("ascii"),
        mode=0o444,
    )
    return payload | {"status": "recorded", "output": str(output_path)}


def _validate_r3_durable_release_identity(
    *,
    marker_path: Path,
    release_checkout: Path,
) -> dict[str, Any]:
    marker_path = _safe_existing_path(
        marker_path,
        description="r3 durable Git release marker",
        kind="file",
    )
    checkout = _safe_existing_path(
        release_checkout,
        description="r3 release checkout",
        kind="directory",
    )
    raw, marker_record = _stable_file_bytes(
        marker_path,
        description="r3 durable Git release marker",
        require_read_only=True,
    )
    marker = _decode_unique_json(raw, description="r3 durable Git release marker")
    if raw != _pretty_canonical_json(marker):
        raise EvidenceError("r3 durable Git release marker is not canonical JSON")
    marker_identity = dict(marker)
    marker_id = marker_identity.pop("marker_id", None)
    if (
        marker_path.name != _R3_DURABLE_RELEASE_MARKER
        or marker.get("schema_version") != 1
        or marker.get("protocol") != _R3_DURABLE_RELEASE_PROTOCOL
        or marker.get("passed") is not True
        or marker.get("release_id") != _R3_RELEASE_ID
        or marker.get("release_tag") != _R3_RELEASE_TAG
        or marker.get("chain_namespace") != _R3_CHAIN_NAMESPACE
        or marker.get("release_git_commit") != _R3_RELEASE_COMMIT
        or marker.get("release_tag_object") != _R3_RELEASE_TAG_OBJECT
        or marker.get("remote_commit_ref") != "refs/heads/schema5-v1.2-r3"
        or marker.get("remote_commit") != _R3_RELEASE_COMMIT
        or marker.get("remote_peeled_commit") != _R3_RELEASE_COMMIT
        or marker.get("remote_tag_object") != _R3_RELEASE_TAG_OBJECT
        or marker.get("clean_checkout") is not True
        or marker.get("annotated_tag") is not True
        or marker.get("remote_query_read_only") is not True
        or marker_id
        != _sha256_bytes(_pretty_canonical_json(marker_identity))
    ):
        raise EvidenceError("r3 durable Git release identity drifted")

    expected_bundle = marker_path.parent / "git_release" / _R3_DURABLE_BUNDLE
    expected_checksum = expected_bundle.with_suffix(expected_bundle.suffix + ".sha256")
    if (
        marker.get("bundle_path") != str(expected_bundle)
        or marker.get("checksum_path") != str(expected_checksum)
    ):
        raise EvidenceError("r3 durable Git release artifact paths drifted")
    bundle_raw, bundle_record = _stable_file_bytes(
        expected_bundle,
        description="r3 durable Git bundle",
        require_read_only=True,
    )
    checksum_raw, checksum_record = _stable_file_bytes(
        expected_checksum,
        description="r3 durable Git bundle checksum",
        require_read_only=True,
    )
    expected_checksum_raw = (
        f"{marker['bundle_sha256']}  {_R3_DURABLE_BUNDLE}\n"
    ).encode("ascii")
    if (
        marker.get("bundle_size") != len(bundle_raw)
        or marker.get("bundle_sha256") != _sha256_bytes(bundle_raw)
        or checksum_raw != expected_checksum_raw
        or marker.get("checksum_sha256") != _sha256_bytes(checksum_raw)
    ):
        raise EvidenceError("r3 durable Git bundle/checksum drifted")

    tag_ref = f"refs/tags/{_R3_RELEASE_TAG}"
    if (
        _run_git(checkout, "cat-file", "-t", tag_ref) != "tag"
        or _run_git(checkout, "rev-parse", "--verify", tag_ref)
        != _R3_RELEASE_TAG_OBJECT
        or _run_git(checkout, "rev-parse", "--verify", f"{tag_ref}^{{commit}}")
        != _R3_RELEASE_COMMIT
        or _run_git(checkout, "rev-parse", "--verify", "HEAD")
        != _R3_RELEASE_COMMIT
        or _run_git(
            checkout, "status", "--porcelain=v1", "--untracked-files=all"
        )
    ):
        raise EvidenceError("r3 release checkout is not the exact clean tag")
    _run_git(checkout, "bundle", "verify", str(expected_bundle))
    if (
        _run_git(
            checkout,
            "bundle",
            "list-heads",
            str(expected_bundle),
            tag_ref,
        )
        != f"{_R3_RELEASE_TAG_OBJECT} {tag_ref}"
    ):
        raise EvidenceError("r3 durable Git bundle does not contain the exact tag")
    return {
        "release_id": _R3_RELEASE_ID,
        "release_tag": _R3_RELEASE_TAG,
        "chain_namespace": _R3_CHAIN_NAMESPACE,
        "release_git_commit": _R3_RELEASE_COMMIT,
        "release_tag_object": _R3_RELEASE_TAG_OBJECT,
        "release_checkout": str(checkout),
        "durable_marker": marker_record | {"marker_id": marker_id},
        "bundle": bundle_record,
        "checksum": checksum_record,
    }


def _prelaunch_failure_artifact_paths(evidence_root: Path) -> dict[str, Path]:
    return {
        "intent": evidence_root / "PRELAUNCH_FAILURE_SEAL_INTENT.json",
        "broken_symlink": (
            evidence_root / "FAILURE_UNSAFE_RECORDED_BROKEN_INTERNAL_SYMLINK.json"
        ),
        "offline_cache": (
            evidence_root / "FAILURE_OFFLINE_CLONE_UNSEEDED_RELEASE_LOCAL_CACHE.json"
        ),
        "inventory": evidence_root / "PRELAUNCH_FAILURE_ARCHIVE_INVENTORY.jsonl",
        "marker": evidence_root / "PRELAUNCH_FAILURE_SEALED.json",
    }


_PRELAUNCH_FAILURE_ARTIFACT_ORDER = (
    "intent",
    "broken_symlink",
    "offline_cache",
    "inventory",
    "marker",
)


def _prelaunch_failure_staging_root(evidence_root: Path) -> Path:
    return evidence_root.parent / f".{evidence_root.name}.prelaunch-failure-staging"


def _prelaunch_failure_pending_path(evidence_root: Path, target: Path) -> Path:
    return _prelaunch_failure_staging_root(evidence_root) / f"{target.name}.pending"


def _validate_or_recover_prelaunch_staging(
    evidence_root: Path,
    artifacts: Mapping[str, Path],
    *,
    recover: bool,
) -> list[str]:
    staging = _prelaunch_failure_staging_root(evidence_root)
    if not staging.exists() and not staging.is_symlink():
        return []
    staging = _safe_existing_path(
        staging,
        description="r3 prelaunch failure sibling staging root",
        kind="directory",
    )
    metadata = os.lstat(staging)
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise EvidenceError(
            "r3 prelaunch failure staging root has unsafe ownership or mode"
        )
    allowed = {
        _prelaunch_failure_pending_path(evidence_root, target).name
        for target in artifacts.values()
    }
    entries = sorted(staging.iterdir(), key=lambda value: value.name)
    for candidate in entries:
        candidate_metadata = os.lstat(candidate)
        if (
            candidate.name not in allowed
            or not stat.S_ISREG(candidate_metadata.st_mode)
            or candidate_metadata.st_nlink != 1
            or candidate_metadata.st_uid != os.getuid()
        ):
            raise EvidenceError(
                f"unsafe or foreign r3 prelaunch staging entry: {candidate}"
            )
    names = [candidate.name for candidate in entries]
    if not recover:
        return names
    for candidate in entries:
        candidate.unlink()
    _fsync_directory(staging)
    staging.rmdir()
    _fsync_directory(staging.parent)
    return names


def _prelaunch_atomic_bytes(
    path: Path,
    payload: bytes,
    *,
    evidence_root: Path,
    artifacts: Mapping[str, Path],
) -> None:
    if path.parent != evidence_root or path not in artifacts.values():
        raise EvidenceError("r3 prelaunch atomic target escapes its contract")
    if path.exists() or path.is_symlink():
        raise EvidenceError(f"r3 prelaunch atomic target already exists: {path}")
    if not evidence_root.exists():
        evidence_root.mkdir(mode=0o750)
        _fsync_directory(evidence_root.parent)
    _safe_existing_path(
        evidence_root,
        description="r3 prelaunch failure evidence root",
        kind="directory",
    )
    _validate_or_recover_prelaunch_staging(
        evidence_root, artifacts, recover=True
    )
    staging = _prelaunch_failure_staging_root(evidence_root)
    staging.mkdir(mode=0o700)
    _fsync_directory(staging.parent)
    pending = _prelaunch_failure_pending_path(evidence_root, path)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(pending, flags, 0o600)
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        os.replace(pending, path)
        _fsync_directory(evidence_root)
        _fsync_directory(staging)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if pending.exists() and not pending.is_symlink():
            pending.unlink()
            _fsync_directory(staging)
        if staging.exists() and not staging.is_symlink():
            try:
                staging.rmdir()
            except OSError:
                pass
            else:
                _fsync_directory(staging.parent)
    _validate_or_recover_prelaunch_staging(
        evidence_root, artifacts, recover=True
    )


def _prelaunch_atomic_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    evidence_root: Path,
    artifacts: Mapping[str, Path],
) -> None:
    _prelaunch_atomic_bytes(
        path,
        _canonical_json(payload),
        evidence_root=evidence_root,
        artifacts=artifacts,
    )


def _validate_prelaunch_evidence_file_set(
    evidence_root: Path,
    artifacts: Mapping[str, Path],
    *,
    expected_prefix_length: int | None,
    require_directory_read_only: bool,
) -> int:
    if not evidence_root.exists() and not evidence_root.is_symlink():
        if expected_prefix_length not in {None, 0}:
            raise EvidenceError("r3 prelaunch failure evidence root is missing")
        return 0
    evidence_root = _safe_existing_path(
        evidence_root,
        description="r3 prelaunch failure evidence root",
        kind="directory",
    )
    metadata = os.lstat(evidence_root)
    if (
        metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or (
            require_directory_read_only
            and stat.S_IMODE(metadata.st_mode) & 0o222
        )
    ):
        raise EvidenceError(
            "r3 prelaunch failure evidence root has unsafe ownership or mode"
        )
    actual = {candidate.name for candidate in evidence_root.iterdir()}
    expected_names = [
        artifacts[key].name for key in _PRELAUNCH_FAILURE_ARTIFACT_ORDER
    ]
    unknown = sorted(actual - set(expected_names))
    if unknown:
        raise EvidenceError(
            f"unexpected r3 prelaunch failure evidence artifacts: {unknown}"
        )
    observed_keys = {
        key
        for key in _PRELAUNCH_FAILURE_ARTIFACT_ORDER
        if artifacts[key].name in actual
    }
    valid_prefixes = [
        set(_PRELAUNCH_FAILURE_ARTIFACT_ORDER[:length])
        for length in range(len(_PRELAUNCH_FAILURE_ARTIFACT_ORDER) + 1)
    ]
    if observed_keys not in valid_prefixes:
        raise EvidenceError(
            "r3 prelaunch failure evidence is not a marker-last prefix"
        )
    observed_length = valid_prefixes.index(observed_keys)
    if (
        expected_prefix_length is not None
        and observed_length != expected_prefix_length
    ):
        raise EvidenceError(
            "r3 prelaunch failure artifact cardinality/order drifted"
        )
    for key in _PRELAUNCH_FAILURE_ARTIFACT_ORDER[:observed_length]:
        candidate = artifacts[key]
        candidate_metadata = os.lstat(candidate)
        if (
            candidate.resolve(strict=True) != candidate
            or not stat.S_ISREG(candidate_metadata.st_mode)
            or candidate_metadata.st_nlink != 1
            or candidate_metadata.st_uid != os.getuid()
            or stat.S_IMODE(candidate_metadata.st_mode) & 0o222
        ):
            raise EvidenceError(
                f"r3 prelaunch failure artifact is unsafe: {candidate}"
            )
    return observed_length


def _write_or_verify_prelaunch_bytes(
    path: Path,
    payload: bytes,
    *,
    description: str,
    evidence_root: Path,
    artifacts: Mapping[str, Path],
) -> None:
    if path.exists() or path.is_symlink():
        raw, _record = _stable_file_bytes(
            path, description=description, require_read_only=True
        )
        if raw != payload:
            raise EvidenceError(f"{description} drifted")
        return
    _prelaunch_atomic_bytes(
        path,
        payload,
        evidence_root=evidence_root,
        artifacts=artifacts,
    )


def seal_prelaunch_failure(
    *,
    evidence_root: str | Path,
    durable_release_marker: str | Path,
    release_checkout: str | Path,
    failure_envelopes: Sequence[str | Path],
    mutation_evidence: Sequence[str | Path],
    apply: bool = False,
) -> dict[str, Any]:
    """Seal both deterministic r3 prelaunch failures without scheduler access.

    The two failed probes ran before any scheduler submission and wrote only below
    ``/tmp``.  Their canonical envelopes are archived into this transaction; the
    exact clean r3 tag and its durable bundle remain external immutable inputs.  The
    source checkout is queried read-only and is never chmod'ed, moved, or rewritten.
    """

    evidence = _safe_output_directory(
        evidence_root, description="r3 prelaunch failure evidence root"
    )
    release = _validate_r3_durable_release_identity(
        marker_path=_absolute(
            durable_release_marker,
            description="r3 durable Git release marker",
        ),
        release_checkout=_absolute(
            release_checkout,
            description="r3 release checkout",
        ),
    )
    if len(failure_envelopes) != len(_R3_PRELAUNCH_FAILURE_CLASSIFICATIONS):
        raise EvidenceError(
            "exactly two r3 deterministic failure envelopes are required"
        )
    failures: dict[str, dict[str, Any]] = {}
    failure_raw: dict[str, bytes] = {}
    failure_source_records: dict[str, dict[str, Any]] = {}
    for source in failure_envelopes:
        path = _absolute(source, description="r3 prelaunch failure envelope")
        payload, raw, record = _validate_r3_prelaunch_failure_envelope(path)
        classification = str(payload["classification"])
        if classification in failures:
            raise EvidenceError(
                f"duplicate r3 prelaunch failure classification: {classification}"
            )
        failures[classification] = payload
        failure_raw[classification] = raw
        failure_source_records[classification] = record
    if set(failures) != set(_R3_PRELAUNCH_FAILURE_CLASSIFICATIONS):
        raise EvidenceError("r3 prelaunch failure classification set drifted")

    bases = [
        _mutation_evidence_binding(
            _safe_existing_path(
                value,
                description="r3 prelaunch zero-mutation basis evidence",
                kind="file",
            )
        )
        for value in mutation_evidence
    ]
    if not bases or len({record["path"] for record in bases}) != len(bases):
        raise EvidenceError(
            "r3 prelaunch failure seal requires unique mutation evidence"
        )
    artifacts = _prelaunch_failure_artifact_paths(evidence)
    lock_path = evidence.parent / f".{evidence.name}.prelaunch-failure-seal.lock"
    guard = _exclusive_lock(lock_path) if apply else nullcontext()
    with guard:
        staged_entries = _validate_or_recover_prelaunch_staging(
            evidence, artifacts, recover=apply
        )
        _validate_prelaunch_evidence_file_set(
            evidence,
            artifacts,
            expected_prefix_length=None,
            require_directory_read_only=False,
        )
        source_bindings = [
            failure_source_records[classification]
            | {
                "classification": classification,
                "failure_id": failures[classification]["failure_id"],
            }
            for classification in _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS
        ]
        expected_intent = {
            "schema_version": 1,
            "protocol": _R3_PRELAUNCH_FAILURE_INTENT_PROTOCOL,
            "evidence_root": str(evidence),
            "release": release,
            "failure_sources": source_bindings,
            "failure_classifications": list(
                _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS
            ),
            "mutation_claim": (
                "bound_to_preexisting_evidence_not_inferred_from_absence"
            ),
            "mutation_evidence_basis": bases,
            "pre_scheduler_submission": True,
            "scheduler_evidence_required": False,
            "known_scheduler_job_ids": [],
            "retry_in_place": False,
            "requires_superseding_release": True,
        }
        if artifacts["intent"].exists() or artifacts["intent"].is_symlink():
            intent = _strict_json_artifact(
                artifacts["intent"],
                description="r3 prelaunch failure seal intent",
            )
            stable = dict(intent)
            intent_id = stable.pop("intent_id", None)
            created_at = stable.pop("created_at", None)
            if (
                stable != expected_intent
                or not isinstance(created_at, str)
                or intent_id
                != _sha256_bytes(
                    _canonical_json(expected_intent | {"created_at": created_at})
                )
            ):
                raise EvidenceError("r3 prelaunch failure seal intent drifted")
        else:
            if evidence.exists() and list(evidence.iterdir()):
                raise EvidenceError(
                    "r3 prelaunch evidence exists before marker-first intent"
                )
            intent = expected_intent | {"created_at": _utc_now()}
            intent["intent_id"] = _sha256_bytes(_canonical_json(intent))
            if apply:
                _prelaunch_atomic_json(
                    artifacts["intent"],
                    intent,
                    evidence_root=evidence,
                    artifacts=artifacts,
                )

        report = {
            "status": "dry_run",
            "evidence_root": str(evidence),
            "intent": str(artifacts["intent"]),
            "marker": str(artifacts["marker"]),
            "release_git_commit": _R3_RELEASE_COMMIT,
            "release_tag_object": _R3_RELEASE_TAG_OBJECT,
            "failure_classifications": list(
                _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS
            ),
            "known_scheduler_job_ids": [],
            "scheduler_evidence_required": False,
            "recoverable_staging_entries": staged_entries,
        }
        if not apply:
            return report

        archive_by_classification = {
            "unsafe_recorded_broken_internal_symlink": artifacts["broken_symlink"],
            "offline_clone_unseeded_release_local_cache": artifacts["offline_cache"],
        }
        archive_records: list[dict[str, Any]] = []
        for classification in _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS:
            archive_path = archive_by_classification[classification]
            raw = failure_raw[classification]
            _write_or_verify_prelaunch_bytes(
                archive_path,
                raw,
                description=f"archived r3 {classification} failure envelope",
                evidence_root=evidence,
                artifacts=artifacts,
            )
            archive_records.append(
                {
                    "classification": classification,
                    "filename": archive_path.name,
                    "sha256": _sha256_bytes(raw),
                    "size": len(raw),
                    "failure_id": failures[classification]["failure_id"],
                }
            )
        inventory_payload = b"".join(
            _canonical_json(row) for row in archive_records
        )
        _write_or_verify_prelaunch_bytes(
            artifacts["inventory"],
            inventory_payload,
            description="r3 prelaunch failure archive inventory",
            evidence_root=evidence,
            artifacts=artifacts,
        )
        marker_core = {
            "schema_version": 1,
            "protocol": _R3_PRELAUNCH_FAILURE_SEAL_PROTOCOL,
            "passed": True,
            "classification": (
                "deterministic_materialization_contract_failure_sealed_fail_closed"
            ),
            "retry_in_place": False,
            "requires_superseding_release": True,
            "release": release,
            "failure_classifications": list(
                _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS
            ),
            "failures": archive_records,
            "mutation_claim": intent["mutation_claim"],
            "mutation_evidence_basis": bases,
            "pre_scheduler_submission": True,
            "scheduler_evidence_required": False,
            "known_scheduler_job_ids": [],
            "intent": {
                "path": str(artifacts["intent"]),
                "sha256": _sha256_file(artifacts["intent"]),
                "intent_id": intent["intent_id"],
            },
            "archive_inventory": {
                "path": str(artifacts["inventory"]),
                "sha256": _sha256_bytes(inventory_payload),
                "size": len(inventory_payload),
            },
            "source_checkout_preserved": True,
        }
        if artifacts["marker"].exists() or artifacts["marker"].is_symlink():
            marker = _strict_json_artifact(
                artifacts["marker"],
                description="r3 prelaunch failure seal marker",
            )
            stable = dict(marker)
            seal_id = stable.pop("seal_id", None)
            sealed_at = stable.pop("sealed_at", None)
            if (
                stable != marker_core
                or not isinstance(sealed_at, str)
                or seal_id
                != _sha256_bytes(
                    _canonical_json(marker_core | {"sealed_at": sealed_at})
                )
            ):
                raise EvidenceError("r3 prelaunch failure seal marker drifted")
            _validate_prelaunch_evidence_file_set(
                evidence,
                artifacts,
                expected_prefix_length=len(_PRELAUNCH_FAILURE_ARTIFACT_ORDER),
                require_directory_read_only=True,
            )
            return marker | {"status": "already_sealed"}

        _validate_prelaunch_evidence_file_set(
            evidence,
            artifacts,
            expected_prefix_length=len(_PRELAUNCH_FAILURE_ARTIFACT_ORDER) - 1,
            require_directory_read_only=False,
        )
        marker = marker_core | {"sealed_at": _utc_now()}
        marker["seal_id"] = _sha256_bytes(_canonical_json(marker))
        _prelaunch_atomic_json(
            artifacts["marker"],
            marker,
            evidence_root=evidence,
            artifacts=artifacts,
        )
        _validate_or_recover_prelaunch_staging(
            evidence, artifacts, recover=True
        )
        _validate_prelaunch_evidence_file_set(
            evidence,
            artifacts,
            expected_prefix_length=len(_PRELAUNCH_FAILURE_ARTIFACT_ORDER),
            require_directory_read_only=False,
        )
        os.chmod(evidence, stat.S_IMODE(evidence.stat().st_mode) & ~0o222)
        _fsync_directory(evidence.parent)
        _validate_prelaunch_evidence_file_set(
            evidence,
            artifacts,
            expected_prefix_length=len(_PRELAUNCH_FAILURE_ARTIFACT_ORDER),
            require_directory_read_only=True,
        )
        return marker | {"status": "sealed"}


def _validate_sealed_r3_release_binding(
    release: object,
    *,
    evidence_root: Path,
) -> dict[str, Any]:
    """Validate the release identity embedded in the self-contained r3 seal."""

    if not isinstance(release, dict) or set(release) != {
        "release_id",
        "release_tag",
        "chain_namespace",
        "release_git_commit",
        "release_tag_object",
        "release_checkout",
        "durable_marker",
        "bundle",
        "checksum",
    }:
        raise EvidenceError("sealed r3 release binding fields drifted")
    if (
        release.get("release_id") != _R3_RELEASE_ID
        or release.get("release_tag") != _R3_RELEASE_TAG
        or release.get("chain_namespace") != _R3_CHAIN_NAMESPACE
        or release.get("release_git_commit") != _R3_RELEASE_COMMIT
        or release.get("release_tag_object") != _R3_RELEASE_TAG_OBJECT
    ):
        raise EvidenceError("sealed r3 release identity drifted")

    recovery_root = evidence_root.parent.parent
    expected_checkout = (
        recovery_root / "materialization_pilot_source_checkout_v1_2_r3"
    )
    expected_marker = recovery_root / _R3_DURABLE_RELEASE_MARKER
    expected_bundle = recovery_root / "git_release" / _R3_DURABLE_BUNDLE
    expected_checksum = expected_bundle.with_suffix(
        expected_bundle.suffix + ".sha256"
    )
    checkout = _canonical_absolute_text(
        release["release_checkout"],
        description="sealed r3 release checkout",
    )
    if checkout != str(expected_checkout):
        raise EvidenceError("sealed r3 release checkout path drifted")

    expected_records = {
        "durable_marker": (expected_marker, True),
        "bundle": (expected_bundle, False),
        "checksum": (expected_checksum, False),
    }
    for name, (expected_path, has_marker_id) in expected_records.items():
        record = release.get(name)
        expected_keys = {"path", "sha256", "size"}
        if has_marker_id:
            expected_keys.add("marker_id")
        if (
            not isinstance(record, dict)
            or set(record) != expected_keys
            or _canonical_absolute_text(
                record.get("path"),
                description=f"sealed r3 {name} path",
            )
            != str(expected_path)
            or not _SHA256_RE.fullmatch(str(record.get("sha256", "")))
            or not isinstance(record.get("size"), int)
            or isinstance(record.get("size"), bool)
            or record["size"] <= 0
            or (
                has_marker_id
                and not _SHA256_RE.fullmatch(str(record.get("marker_id", "")))
            )
        ):
            raise EvidenceError(f"sealed r3 {name} binding drifted")
    return dict(release)


def _validate_prelaunch_mutation_basis(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise EvidenceError("sealed r3 zero-mutation basis is missing")
    allowed_kinds = {
        "zero_result_mutation_receipt",
        "verified_read_only_snapshot",
        "maintenance_quiescence",
    }
    records: list[dict[str, Any]] = []
    paths: set[str] = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "sha256",
            "size",
            "device",
            "inode",
            "mode",
            "evidence_kind",
            "protocol",
        }:
            raise EvidenceError("sealed r3 zero-mutation basis fields drifted")
        path = _canonical_absolute_text(
            row.get("path"),
            description="sealed r3 zero-mutation evidence path",
        )
        if (
            path in paths
            or not _SHA256_RE.fullmatch(str(row.get("sha256", "")))
            or any(
                not isinstance(row.get(field), int)
                or isinstance(row.get(field), bool)
                or row[field] < 0
                for field in ("size", "device", "inode", "mode")
            )
            or row.get("evidence_kind") not in allowed_kinds
            or (
                row.get("protocol") is not None
                and (
                    not isinstance(row.get("protocol"), str)
                    or not row["protocol"]
                )
            )
        ):
            raise EvidenceError("sealed r3 zero-mutation basis drifted")
        if (
            row["evidence_kind"] == "zero_result_mutation_receipt"
            and row.get("protocol")
            != "schema5-v1.2-r2-r1-zero-result-mutation-receipt-v1"
        ):
            raise EvidenceError("sealed r3 zero-result mutation receipt drifted")
        paths.add(path)
        records.append(dict(row))
    if not any(
        row["evidence_kind"] == "zero_result_mutation_receipt"
        for row in records
    ):
        raise EvidenceError(
            "sealed r3 evidence lacks an explicit zero-result mutation receipt"
        )
    return records


def _read_canonical_read_only_json(
    path: Path,
    *,
    description: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw, record = _stable_file_bytes(
        path,
        description=description,
        require_read_only=True,
    )
    payload = _decode_unique_json(raw, description=description)
    if raw != _canonical_json(payload):
        raise EvidenceError(f"{description} is not canonical JSON")
    return payload, raw, record


def verify_prelaunch_failure_seal(
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Independently verify the immutable r3 prelaunch-failure seal.

    Verification reads only ``evidence_root``.  External r3 checkouts, bundles,
    source prefixes, schedulers, and result trees are deliberately not consulted.
    The returned binding is canonical-JSON-safe and can be persisted verbatim by a
    superseding recovery renderer.
    """

    evidence = _safe_existing_path(
        evidence_root,
        description="sealed r3 prelaunch failure evidence root",
        kind="directory",
    )
    artifacts = _prelaunch_failure_artifact_paths(evidence)
    _validate_prelaunch_evidence_file_set(
        evidence,
        artifacts,
        expected_prefix_length=len(_PRELAUNCH_FAILURE_ARTIFACT_ORDER),
        require_directory_read_only=True,
    )

    intent, intent_raw, intent_record = _read_canonical_read_only_json(
        artifacts["intent"],
        description="sealed r3 prelaunch failure intent",
    )
    expected_intent_fields = {
        "schema_version",
        "protocol",
        "evidence_root",
        "release",
        "failure_sources",
        "failure_classifications",
        "mutation_claim",
        "mutation_evidence_basis",
        "pre_scheduler_submission",
        "scheduler_evidence_required",
        "known_scheduler_job_ids",
        "retry_in_place",
        "requires_superseding_release",
        "created_at",
        "intent_id",
    }
    intent_identity = dict(intent)
    intent_id = intent_identity.pop("intent_id", None)
    if (
        set(intent) != expected_intent_fields
        or intent.get("schema_version") != 1
        or intent.get("protocol") != _R3_PRELAUNCH_FAILURE_INTENT_PROTOCOL
        or intent.get("evidence_root") != str(evidence)
        or intent.get("failure_classifications")
        != list(_R3_PRELAUNCH_FAILURE_CLASSIFICATIONS)
        or intent.get("mutation_claim")
        != "bound_to_preexisting_evidence_not_inferred_from_absence"
        or intent.get("pre_scheduler_submission") is not True
        or intent.get("scheduler_evidence_required") is not False
        or intent.get("known_scheduler_job_ids") != []
        or intent.get("retry_in_place") is not False
        or intent.get("requires_superseding_release") is not True
        or not isinstance(intent.get("created_at"), str)
        or not intent["created_at"]
        or intent_id != _sha256_bytes(_canonical_json(intent_identity))
    ):
        raise EvidenceError("sealed r3 prelaunch failure intent drifted")
    release = _validate_sealed_r3_release_binding(
        intent.get("release"),
        evidence_root=evidence,
    )
    mutation_basis = _validate_prelaunch_mutation_basis(
        intent.get("mutation_evidence_basis")
    )

    failure_sources = intent.get("failure_sources")
    if (
        not isinstance(failure_sources, list)
        or len(failure_sources) != len(_R3_PRELAUNCH_FAILURE_CLASSIFICATIONS)
    ):
        raise EvidenceError("sealed r3 failure-source bindings drifted")
    source_by_classification: dict[str, dict[str, Any]] = {}
    for source in failure_sources:
        if not isinstance(source, dict) or set(source) != {
            "path",
            "sha256",
            "size",
            "classification",
            "failure_id",
        }:
            raise EvidenceError("sealed r3 failure-source fields drifted")
        classification = source.get("classification")
        if (
            classification not in _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS
            or classification in source_by_classification
            or not _SHA256_RE.fullmatch(str(source.get("sha256", "")))
            or not _SHA256_RE.fullmatch(str(source.get("failure_id", "")))
            or not isinstance(source.get("size"), int)
            or isinstance(source.get("size"), bool)
            or source["size"] <= 0
        ):
            raise EvidenceError("sealed r3 failure-source binding drifted")
        _canonical_absolute_text(
            source.get("path"),
            description="sealed r3 failure-source path",
        )
        source_by_classification[str(classification)] = dict(source)
    if list(source_by_classification) != list(
        _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS
    ):
        raise EvidenceError("sealed r3 failure-source order drifted")

    archive_by_classification = {
        "unsafe_recorded_broken_internal_symlink": artifacts["broken_symlink"],
        "offline_clone_unseeded_release_local_cache": artifacts["offline_cache"],
    }
    expected_archive_records: list[dict[str, Any]] = []
    for classification in _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS:
        archive_path = archive_by_classification[classification]
        payload, raw, record = _validate_r3_prelaunch_failure_envelope(
            archive_path
        )
        if payload["classification"] != classification:
            raise EvidenceError("sealed r3 failure archive substitution detected")
        source = source_by_classification[classification]
        if (
            source["sha256"] != record["sha256"]
            or source["size"] != record["size"]
            or source["failure_id"] != payload["failure_id"]
        ):
            raise EvidenceError("sealed r3 failure source/archive binding drifted")
        expected_archive_records.append(
            {
                "classification": classification,
                "filename": archive_path.name,
                "sha256": _sha256_bytes(raw),
                "size": len(raw),
                "failure_id": payload["failure_id"],
            }
        )

    inventory_raw, inventory_record = _stable_file_bytes(
        artifacts["inventory"],
        description="sealed r3 prelaunch failure archive inventory",
        require_read_only=True,
    )
    inventory_rows: list[dict[str, Any]] = []
    for line in inventory_raw.splitlines(keepends=True):
        row = _decode_unique_json(
            line,
            description="sealed r3 prelaunch failure inventory row",
        )
        if line != _canonical_json(row):
            raise EvidenceError(
                "sealed r3 prelaunch failure inventory is not canonical JSONL"
            )
        inventory_rows.append(row)
    if inventory_rows != expected_archive_records:
        raise EvidenceError("sealed r3 prelaunch failure inventory drifted")

    marker, marker_raw, marker_record = _read_canonical_read_only_json(
        artifacts["marker"],
        description="sealed r3 prelaunch failure marker",
    )
    expected_marker_fields = {
        "schema_version",
        "protocol",
        "passed",
        "classification",
        "retry_in_place",
        "requires_superseding_release",
        "release",
        "failure_classifications",
        "failures",
        "mutation_claim",
        "mutation_evidence_basis",
        "pre_scheduler_submission",
        "scheduler_evidence_required",
        "known_scheduler_job_ids",
        "intent",
        "archive_inventory",
        "source_checkout_preserved",
        "sealed_at",
        "seal_id",
    }
    marker_identity = dict(marker)
    seal_id = marker_identity.pop("seal_id", None)
    if (
        set(marker) != expected_marker_fields
        or marker.get("schema_version") != 1
        or marker.get("protocol") != _R3_PRELAUNCH_FAILURE_SEAL_PROTOCOL
        or marker.get("passed") is not True
        or marker.get("classification")
        != "deterministic_materialization_contract_failure_sealed_fail_closed"
        or marker.get("retry_in_place") is not False
        or marker.get("requires_superseding_release") is not True
        or marker.get("release") != release
        or marker.get("failure_classifications")
        != list(_R3_PRELAUNCH_FAILURE_CLASSIFICATIONS)
        or marker.get("failures") != expected_archive_records
        or marker.get("mutation_claim") != intent["mutation_claim"]
        or marker.get("mutation_evidence_basis") != mutation_basis
        or marker.get("pre_scheduler_submission") is not True
        or marker.get("scheduler_evidence_required") is not False
        or marker.get("known_scheduler_job_ids") != []
        or marker.get("source_checkout_preserved") is not True
        or not isinstance(marker.get("sealed_at"), str)
        or not marker["sealed_at"]
        or seal_id != _sha256_bytes(_canonical_json(marker_identity))
    ):
        raise EvidenceError("sealed r3 prelaunch failure marker drifted")
    if marker.get("intent") != {
        "path": str(artifacts["intent"]),
        "sha256": _sha256_bytes(intent_raw),
        "intent_id": intent_id,
    }:
        raise EvidenceError("sealed r3 prelaunch failure intent binding drifted")
    if marker.get("archive_inventory") != {
        "path": str(artifacts["inventory"]),
        "sha256": _sha256_bytes(inventory_raw),
        "size": len(inventory_raw),
    }:
        raise EvidenceError(
            "sealed r3 prelaunch failure inventory binding drifted"
        )

    observed_records = {
        "intent": intent_record,
        "broken_symlink": expected_archive_records[0],
        "offline_cache": expected_archive_records[1],
        "inventory": inventory_record,
        "marker": marker_record,
    }
    _validate_prelaunch_evidence_file_set(
        evidence,
        artifacts,
        expected_prefix_length=len(_PRELAUNCH_FAILURE_ARTIFACT_ORDER),
        require_directory_read_only=True,
    )
    for name in _PRELAUNCH_FAILURE_ARTIFACT_ORDER:
        _raw, current = _stable_file_bytes(
            artifacts[name],
            description=f"sealed r3 prelaunch failure {name} replay",
            require_read_only=True,
        )
        if (
            current["sha256"] != observed_records[name]["sha256"]
            or current["size"] != observed_records[name]["size"]
        ):
            raise EvidenceError(
                f"sealed r3 prelaunch failure {name} changed during verification"
            )

    return {
        "schema_version": 1,
        "protocol": _R3_PRELAUNCH_FAILURE_BINDING_PROTOCOL,
        "root": str(evidence),
        "marker": marker_record | {"seal_id": seal_id},
        "release_id": _R3_RELEASE_ID,
        "release_tag": _R3_RELEASE_TAG,
        "release_git_commit": _R3_RELEASE_COMMIT,
        "release_tag_object": _R3_RELEASE_TAG_OBJECT,
        "chain_namespace": _R3_CHAIN_NAMESPACE,
        "classification": marker["classification"],
        "failure_classifications": list(
            _R3_PRELAUNCH_FAILURE_CLASSIFICATIONS
        ),
        "retry_in_place": False,
        "requires_superseding_release": True,
        "pre_scheduler_submission": True,
        "scheduler_evidence_required": False,
        "known_scheduler_job_ids": [],
        "mutation_claim": marker["mutation_claim"],
        "archive_inventory_sha256": inventory_record["sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal-quarantine")
    seal.add_argument("--tree", type=Path, required=True)
    seal.add_argument("--evidence-root", type=Path, required=True)
    seal.add_argument("--release-id", required=True)
    seal.add_argument("--failed-job-id", required=True)
    seal.add_argument("--quarantine-completion", type=Path, required=True)
    seal.add_argument("--apply", action="store_true")

    failure = subparsers.add_parser("record-failure")
    failure.add_argument("--output", type=Path, required=True)
    failure.add_argument("--chain-manifest", type=Path, required=True)
    failure.add_argument("--submission-receipt", type=Path, required=True)
    failure.add_argument("--snapshot-marker", type=Path, required=True)
    failure.add_argument("--snapshot-attestation", type=Path, required=True)
    failure.add_argument("--materialization-log", type=Path, required=True)
    failure.add_argument("--quarantine-seal", type=Path, required=True)
    failure.add_argument("--incident", type=Path, required=True)
    failure.add_argument("--job-id", action="append", required=True)
    failure.add_argument("--apply", action="store_true")

    incident = subparsers.add_parser("record-conda-incident")
    incident.add_argument("--output", type=Path, required=True)
    incident.add_argument("--harness-prefix", type=Path, required=True)
    incident.add_argument("--serving-prefix", type=Path, required=True)
    incident.add_argument("--recovered-harness-record", type=Path, required=True)
    incident.add_argument("--failed-materialization-log", type=Path, required=True)
    incident.add_argument("--observed-at", required=True)
    incident.add_argument("--apply", action="store_true")

    canary = subparsers.add_parser(
        "seal-canary-failure",
        help="seal one terminal partial r2 composite canary without cancelling jobs",
    )
    canary.add_argument("--tree", type=Path, required=True)
    canary.add_argument("--evidence-root", type=Path, required=True)
    canary.add_argument("--release-checkout", type=Path, required=True)
    canary.add_argument("--scheduler-user", required=True)
    canary.add_argument("--error-classification", required=True)
    canary.add_argument("--error-summary", required=True)
    canary.add_argument(
        "--mutation-evidence",
        type=Path,
        action="append",
        required=True,
        help=(
            "preexisting maintenance/quiescence, verified snapshot, or explicit "
            "zero-result-mutation evidence; repeat to bind multiple records"
        ),
    )
    canary.add_argument("--apply", action="store_true")

    attempt = subparsers.add_parser(
        "record-prelaunch-attempt",
        help=(
            "run one direct r3 materialization failure probe, inventory its "
            "inputs before/after, and publish a canonical envelope"
        ),
    )
    attempt.add_argument("--output", type=Path, required=True)
    attempt.add_argument(
        "--classification",
        choices=_R3_PRELAUNCH_FAILURE_CLASSIFICATIONS,
        required=True,
    )
    attempt.add_argument("--cwd", type=Path, required=True)
    attempt.add_argument(
        "--environment",
        nargs=2,
        action="append",
        default=[],
        metavar=("KEY", "VALUE"),
        help=(
            "one exact environment entry; the probe inherits no other "
            "environment variables"
        ),
    )
    attempt.add_argument(
        "--input-root",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "PATH"),
        help="one immutable probe input tree to inventory before and after",
    )
    attempt.add_argument(
        "--write-root",
        type=Path,
        action="append",
        required=True,
        help="declared probe write root below /tmp",
    )
    attempt.add_argument(
        "--timeout-seconds", type=float, default=600.0
    )
    attempt.add_argument(
        "--observed-at",
        help="optional fixed ISO-8601 timestamp; defaults to command completion",
    )
    attempt.add_argument(
        "--command",
        dest="probe_argv",
        nargs=argparse.REMAINDER,
        required=True,
        help="direct argv; place this option last and do not use a shell",
    )
    attempt.add_argument("--apply", action="store_true")

    prelaunch = subparsers.add_parser(
        "seal-prelaunch-failure",
        help=(
            "archive and seal the two deterministic r3 materialization "
            "prelaunch failures without querying Slurm"
        ),
    )
    prelaunch.add_argument("--evidence-root", type=Path, required=True)
    prelaunch.add_argument(
        "--durable-release-marker", type=Path, required=True
    )
    prelaunch.add_argument("--release-checkout", type=Path, required=True)
    prelaunch.add_argument(
        "--failure-envelope",
        type=Path,
        action="append",
        required=True,
        help=(
            "canonical r3 prelaunch failure envelope; supply exactly the broken-"
            "symlink and empty-offline-cache envelopes"
        ),
    )
    prelaunch.add_argument(
        "--mutation-evidence",
        type=Path,
        action="append",
        required=True,
        help=(
            "preexisting verified snapshot or explicit zero-result-mutation "
            "evidence; repeat to bind multiple records"
        ),
    )
    prelaunch.add_argument("--apply", action="store_true")

    verify_prelaunch = subparsers.add_parser(
        "verify-prelaunch-failure",
        help=(
            "independently verify the sealed r3 prelaunch failures using only "
            "their immutable evidence root"
        ),
    )
    verify_prelaunch.add_argument("--evidence-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal-quarantine":
            result = seal_quarantine(
                tree=args.tree,
                evidence_root=args.evidence_root,
                release_id=args.release_id,
                failed_job_id=args.failed_job_id,
                quarantine_completion=args.quarantine_completion,
                apply=args.apply,
            )
        elif args.command == "record-failure":
            result = record_failure(
                output=args.output,
                chain_manifest=args.chain_manifest,
                submission_receipt=args.submission_receipt,
                snapshot_marker=args.snapshot_marker,
                snapshot_attestation=args.snapshot_attestation,
                materialization_log=args.materialization_log,
                quarantine_seal=args.quarantine_seal,
                incident=args.incident,
                job_ids=args.job_id,
                apply=args.apply,
            )
        elif args.command == "record-conda-incident":
            result = record_conda_reconciliation_incident(
                output=args.output,
                harness_prefix=args.harness_prefix,
                serving_prefix=args.serving_prefix,
                recovered_harness_record=args.recovered_harness_record,
                failed_materialization_log=args.failed_materialization_log,
                observed_at=args.observed_at,
                apply=args.apply,
            )
        elif args.command == "record-prelaunch-attempt":
            result = record_prelaunch_attempt(
                output=args.output,
                classification=args.classification,
                command=args.probe_argv,
                cwd=args.cwd,
                environment=args.environment,
                input_roots=args.input_root,
                write_roots=args.write_root,
                apply=args.apply,
                timeout_seconds=args.timeout_seconds,
                observed_at=args.observed_at,
            )
        elif args.command == "seal-canary-failure":
            result = seal_canary_failure(
                tree=args.tree,
                evidence_root=args.evidence_root,
                release_checkout=args.release_checkout,
                scheduler_user=args.scheduler_user,
                error_classification=args.error_classification,
                error_summary=args.error_summary,
                mutation_evidence=args.mutation_evidence,
                apply=args.apply,
            )
        elif args.command == "seal-prelaunch-failure":
            result = seal_prelaunch_failure(
                evidence_root=args.evidence_root,
                durable_release_marker=args.durable_release_marker,
                release_checkout=args.release_checkout,
                failure_envelopes=args.failure_envelope,
                mutation_evidence=args.mutation_evidence,
                apply=args.apply,
            )
        else:
            result = verify_prelaunch_failure_seal(args.evidence_root)
    except (EvidenceError, OSError) as exc:
        print(f"[schema5-evidence] ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
