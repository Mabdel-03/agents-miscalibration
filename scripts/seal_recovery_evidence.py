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
import os
import pwd
import re
import shlex
import stat
import subprocess
import tempfile
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


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
            "sacct",
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
    try:
        proc = subprocess.run(
            ("git", "-C", str(checkout), *arguments),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
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
        runner = subprocess.run
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
        else:
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
    except (EvidenceError, OSError) as exc:
        print(f"[schema5-evidence] ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
