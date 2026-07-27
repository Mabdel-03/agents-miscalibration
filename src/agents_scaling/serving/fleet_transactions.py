"""Durable Slurm admission state for the canonical schema-5 serving fleet.

The Slurm submission boundary is not transactional: ``sbatch`` can accept a job and
the caller can die before receiving or persisting its job id.  This module makes that
boundary recoverable.  It writes one immutable, generation/intent-addressed batch
script and a durable ``submitting`` record *before* invoking Slurm, then reconciles the
intent against both live queue and accounting history.  A token that maps to more than
one allocation is an ambiguity, never extra capacity.

The module deliberately contains no serving-profile policy.  ``keepalive.py`` supplies
the frozen replica contract and validates Slurm's spooled script byte-for-byte.  This
separation keeps the transaction machinery small enough to test at every crash edge.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import stat
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

from agents_scaling.experiment import io


STATE_SCHEMA_VERSION = 1
STATE_DIRECTORY = ".fleet-transactions-v1"
LEDGERS_DIRECTORY = "ledgers"
ABSENCE_RECEIPTS_DIRECTORY = "submission-absence-receipts"
CURRENT_FILENAME = "CURRENT.json"
ALERTS_FILENAME = "alerts.jsonl"
LOCK_FILENAME = "fleet.lock"
SAVE_INTENT_FILENAME = "ledger-save-intent.json"
DEFAULT_VISIBILITY_GRACE_SECONDS = 300.0
STDIN_EXACT_SUBMISSION_TRANSPORT = "stdin_exact_bytes_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[0-9a-f]{32}")
_ABSENCE_RECEIPT_REFERENCE_RE = re.compile(
    r"^proven not accepted after complete squeue\+sacct visibility grace; "
    r"absence_receipt_path=(?P<path>[^;\n\r]+);"
    r"absence_receipt_sha256=(?P<sha256>[0-9a-f]{64})$"
)


class FleetTransactionError(RuntimeError):
    """Fleet admission or recovery cannot proceed without risking duplication."""


@dataclass(frozen=True)
class SchedulerRow:
    job_id: str
    job_name: str
    state: str
    partition: str
    node: str
    command: str
    comment: str
    source: str = "joined"
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    time_limit_seconds: int | None = None
    # ``None`` means the source cannot report dependencies.  This is accepted only
    # for terminal sacct-only history; every active allocation must be joined to the
    # squeue view, where an empty string means "no dependency".
    dependency: str | None = ""
    # Exact scheduler QOS is placement authority.  Keep this final/defaulted so
    # legacy positional construction cannot silently shift another field into it;
    # production reconciliation nevertheless requires a non-empty exact value.
    qos: str = ""


@dataclass(frozen=True)
class SchedulerSnapshot:
    rows: tuple[SchedulerRow, ...]
    captured_at: float
    squeue_ok: bool
    sacct_ok: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReconciledFleetAllocation:
    """One scheduler allocation bound to its exact durable admission intent."""

    replica_id: str
    profile: str
    ledger_generation: int
    row: SchedulerRow
    attempt: Mapping[str, Any]
    health: Mapping[str, Any] | None


@dataclass(frozen=True)
class FleetReconciliation:
    """Read-only scheduler/ledger join shared by readiness and the supervisor."""

    allocations: tuple[ReconciledFleetAllocation, ...]
    ignored_terminal_job_ids: tuple[str, ...]

    @property
    def active_allocations(self) -> tuple[ReconciledFleetAllocation, ...]:
        return tuple(
            allocation
            for allocation in self.allocations
            if not terminal_state(allocation.row.state)
        )

    @property
    def logical_allocations(self) -> tuple[ReconciledFleetAllocation, ...]:
        """Return exactly the promoted allocation for each logical replica.

        A rolling handoff may temporarily own two physical Slurm allocations.  The
        predecessor remains the logical allocation while the successor is ``standby``;
        after the atomic registry promotion the successor becomes logical and the
        predecessor is only ``retiring``.  Every other overlap is corruption.
        """

        by_replica: dict[str, list[ReconciledFleetAllocation]] = {}
        for allocation in self.active_allocations:
            by_replica.setdefault(allocation.replica_id, []).append(allocation)
        selected: list[ReconciledFleetAllocation] = []
        for replica_id, rows in sorted(by_replica.items()):
            candidates = [
                row
                for row in rows
                if row.attempt.get("lifecycle") in {"primary", "promoted"}
            ]
            if len(candidates) != 1:
                raise FleetTransactionError(
                    f"logical fleet allocation is ambiguous for {replica_id}: "
                    f"{[(row.row.job_id, row.attempt.get('lifecycle')) for row in rows]}"
                )
            selected.append(candidates[0])
        return tuple(selected)


@dataclass(frozen=True)
class CommittedEndpointAdmission:
    """Exact durable admission facts required to seal one endpoint registration."""

    replica_id: str
    slurm_job_id: str
    intent_token: str
    rollout_generation: int
    committed_at: float
    scheduler_comment: str
    launch_kind: str
    lifecycle: str
    sbatch_path: Path
    sbatch_sha256: str


def committed_endpoint_admission(
    directory: Path,
    ledger: Mapping[str, Any],
    *,
    replica_id: str,
    attempt: Mapping[str, Any],
    slurm_job_id: str,
) -> CommittedEndpointAdmission:
    """Export a validated committed intent without trusting a caller-made summary.

    The fleet supervisor calls this while holding the transaction lock.  It binds the
    endpoint archive to the exact in-ledger attempt and immutable local script; a copied
    dictionary, stale generation, pre-commit intent, or same-replica sibling cannot be
    substituted.
    """

    if not str(slurm_job_id).isdigit():
        raise FleetTransactionError("endpoint admission requires a numeric Slurm job id")
    generation = ledger.get("rollout_generation")
    replicas = ledger.get("replicas")
    if (
        type(generation) is not int
        or generation < 1
        or not isinstance(replicas, Mapping)
        or replica_id not in replicas
        or not isinstance(replicas[replica_id], Mapping)
    ):
        raise FleetTransactionError("endpoint admission ledger identity is invalid")
    attempts = replicas[replica_id].get("attempts")
    if not isinstance(attempts, list):
        raise FleetTransactionError("endpoint admission ledger lacks attempts")
    exact = [
        candidate
        for candidate in attempts
        if candidate is attempt
    ]
    if len(exact) != 1:
        raise FleetTransactionError(
            f"endpoint admission is not one exact ledger attempt for {replica_id}"
        )
    material = exact[0]
    _validate_attempt(
        material,
        replica_id=replica_id,
        generation=generation,
        directory=directory,
    )
    committed_at = material.get("committed_at")
    if (
        material.get("state") != "committed"
        or str(material.get("job_id") or "") != str(slurm_job_id)
        or not isinstance(committed_at, (int, float))
        or isinstance(committed_at, bool)
        or float(committed_at) <= 0
    ):
        raise FleetTransactionError(
            f"endpoint intent for {replica_id} job {slurm_job_id} is not committed"
        )
    script_path, script_raw = _stable_regular_preimage(
        str(material["sbatch_path"]),
        description=f"endpoint local script for {replica_id}",
        read_only=True,
    )
    script_sha256 = hashlib.sha256(script_raw).hexdigest()
    if script_sha256 != material["sbatch_sha256"]:
        raise FleetTransactionError(
            f"endpoint local script hash drifted for {replica_id}"
        )
    parsed = parse_intent_comment(str(material["scheduler_comment"]))
    if (
        parsed is None
        or parsed["replica"] != replica_id
        or parsed["generation"] != str(generation)
        or parsed["intent"] != material["intent_token"]
    ):
        raise FleetTransactionError(
            f"endpoint scheduler intent drifted for {replica_id}"
        )
    return CommittedEndpointAdmission(
        replica_id=replica_id,
        slurm_job_id=str(slurm_job_id),
        intent_token=str(material["intent_token"]),
        rollout_generation=generation,
        committed_at=float(committed_at),
        scheduler_comment=str(material["scheduler_comment"]),
        launch_kind=str(material["launch_kind"]),
        lifecycle=str(material["lifecycle"]),
        sbatch_path=script_path,
        sbatch_sha256=script_sha256,
    )


def state_directory(pool_root: str | Path) -> Path:
    return Path(pool_root).expanduser().resolve() / STATE_DIRECTORY


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lexical_absolute_path(path: str | Path, *, description: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise FleetTransactionError(f"{description} is not an absolute path")
    return Path(os.path.abspath(os.fspath(candidate)))


def _stable_regular_preimage(
    path: str | Path,
    *,
    description: str,
    read_only: bool,
) -> tuple[Path, bytes]:
    """Read one exact regular-file inode without following a symlink or race."""

    lexical = _lexical_absolute_path(path, description=description)
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FleetTransactionError(f"{description} is unavailable: {exc}") from exc
    if resolved != lexical or lexical.is_symlink():
        raise FleetTransactionError(f"{description} traverses a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise FleetTransactionError(f"cannot open {description}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise FleetTransactionError(
            f"{description} disappeared after its stable read: {exc}"
        ) from exc

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or identity(before) != identity(after)
        or identity(current) != identity(after)
        or (read_only and stat.S_IMODE(before.st_mode) & 0o222)
        or (read_only and stat.S_IMODE(current.st_mode) & 0o222)
    ):
        qualifier = "read-only " if read_only else ""
        raise FleetTransactionError(
            f"{description} changed or is not a one-link {qualifier}regular file"
        )
    return lexical, b"".join(blocks)


@contextmanager
def transaction_lock(pool_root: str | Path) -> Iterator[Path]:
    """Hold the one pool-scoped, cross-node advisory transaction lock."""

    directory = state_directory(pool_root)
    if directory.is_symlink():
        raise FleetTransactionError(f"fleet state directory is symlinked: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise FleetTransactionError(f"fleet state path is not a directory: {directory}")
    lock_path = directory / LOCK_FILENAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise FleetTransactionError(f"cannot open fleet lock {lock_path}: {exc}") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FleetTransactionError(
                f"another canonical fleet transaction owns {lock_path}"
            ) from exc
        yield directory
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def read_transaction_lock(pool_root: str | Path) -> Iterator[Path]:
    """Hold a shared, non-mutating view of an initialized fleet transaction."""

    directory = state_directory(pool_root)
    if directory.is_symlink() or not directory.is_dir():
        raise FleetTransactionError(
            f"fleet state directory is not initialized safely: {directory}"
        )
    lock_path = directory / LOCK_FILENAME
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as exc:
        raise FleetTransactionError(
            f"cannot open initialized fleet lock {lock_path}: {exc}"
        ) from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FleetTransactionError(
                f"fleet transaction is changing under {lock_path}"
            ) from exc
        yield directory
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ": "),
    ) + "\n"


def _strict_json_loads(payload: str, *, artifact: str) -> Any:
    """Parse controller state without accepting duplicate keys or NaN/Infinity."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        material: dict[str, Any] = {}
        for key, value in pairs:
            if key in material:
                raise ValueError(f"duplicate JSON key {key!r}")
            material[key] = value
        return material

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r}")

    try:
        material = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise FleetTransactionError(f"cannot parse {artifact}: {exc}") from exc

    def reject_nonfinite(value: Any, *, location: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise FleetTransactionError(
                f"cannot parse {artifact}: non-finite JSON number at {location}"
            )
        if isinstance(value, list):
            for index, item in enumerate(value):
                reject_nonfinite(item, location=f"{location}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                reject_nonfinite(item, location=f"{location}.{key}")

    reject_nonfinite(material, location="$")
    return material


def ledger_path(directory: Path, rollout_generation: int) -> Path:
    return directory / LEDGERS_DIRECTORY / f"g{rollout_generation:06d}.json"


def _file_digest_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise FleetTransactionError(f"fleet transaction artifact is unsafe: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_index_payload(
    *, directory: Path, material: Mapping[str, Any], ledger_sha256: str
) -> dict[str, Any]:
    generation = int(material["rollout_generation"])
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "pool_root": material.get("pool_root"),
        "pool_id": material.get("pool_id"),
        "fleet_sha256": material.get("fleet_sha256"),
        "current_generation": generation,
        "ledger_path": str(ledger_path(directory, generation).resolve()),
        "ledger_sha256": ledger_sha256,
        "updated_at": float(material["updated_at"]),
    }


def _unlink_durable(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _read_current_index_raw(directory: Path) -> dict[str, Any] | None:
    path = directory / CURRENT_FILENAME
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise FleetTransactionError(f"fleet current index is not a regular file: {path}")
    try:
        payload = _strict_json_loads(
            path.read_text(encoding="utf-8"), artifact="fleet current index"
        )
    except (OSError, UnicodeError) as exc:
        raise FleetTransactionError(f"cannot parse fleet current index: {exc}") from exc
    required = {
        "schema_version",
        "pool_root",
        "pool_id",
        "fleet_sha256",
        "current_generation",
        "ledger_path",
        "ledger_sha256",
        "updated_at",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise FleetTransactionError("fleet current-index fields drifted")
    generation = payload["current_generation"]
    if (
        payload["schema_version"] != STATE_SCHEMA_VERSION
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
        or _SHA256_RE.fullmatch(str(payload["fleet_sha256"])) is None
        or _SHA256_RE.fullmatch(str(payload["ledger_sha256"])) is None
    ):
        raise FleetTransactionError("fleet current-index identity is invalid")
    expected_path = ledger_path(directory, generation).resolve()
    if str(expected_path) != payload["ledger_path"]:
        raise FleetTransactionError("fleet current index points outside its generation")
    return payload


def _recover_interrupted_save(directory: Path) -> None:
    """Finish or roll back the one write-ahead ledger publication intent.

    A normal state transition replaces the generation ledger and then ``CURRENT.json``.
    Those are two independent atomic renames.  The write-ahead record makes a hard kill
    between them distinguishable from an unjournaled/tampered hash mismatch.
    """

    intent_path = directory / SAVE_INTENT_FILENAME
    if not intent_path.exists():
        return
    if intent_path.is_symlink() or not intent_path.is_file():
        raise FleetTransactionError(
            f"fleet ledger save intent is not a regular file: {intent_path}"
        )
    try:
        intent = _strict_json_loads(
            intent_path.read_text(encoding="utf-8"),
            artifact="fleet ledger save intent",
        )
    except (OSError, UnicodeError) as exc:
        raise FleetTransactionError(f"cannot parse fleet ledger save intent: {exc}") from exc
    required = {
        "schema_version",
        "kind",
        "rollout_generation",
        "ledger_path",
        "ledger_before_sha256",
        "ledger_after_sha256",
        "index_required",
        "index_before_sha256",
        "index_after_sha256",
        "index_after",
    }
    generation = intent.get("rollout_generation") if isinstance(intent, dict) else None
    if (
        not isinstance(intent, dict)
        or set(intent) != required
        or intent.get("schema_version") != STATE_SCHEMA_VERSION
        or intent.get("kind") != "schema5-fleet-ledger-save-v1"
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
        or intent.get("ledger_path")
        != str(ledger_path(directory, generation).resolve())
        or intent.get("ledger_before_sha256") is not None
        and _SHA256_RE.fullmatch(str(intent.get("ledger_before_sha256"))) is None
        or _SHA256_RE.fullmatch(str(intent.get("ledger_after_sha256"))) is None
        or not isinstance(intent.get("index_required"), bool)
        or intent.get("index_before_sha256") is not None
        and _SHA256_RE.fullmatch(str(intent.get("index_before_sha256"))) is None
        or intent.get("index_after_sha256") is not None
        and _SHA256_RE.fullmatch(str(intent.get("index_after_sha256"))) is None
    ):
        raise FleetTransactionError("fleet ledger save intent fields are invalid")
    target = Path(str(intent["ledger_path"]))
    observed_ledger = _file_digest_or_none(target)
    before_ledger = intent["ledger_before_sha256"]
    after_ledger = intent["ledger_after_sha256"]
    current_path = directory / CURRENT_FILENAME
    observed_index = _file_digest_or_none(current_path)
    before_index = intent["index_before_sha256"]
    after_index = intent["index_after_sha256"]

    if observed_ledger == before_ledger:
        if observed_index != before_index:
            raise FleetTransactionError(
                "fleet save intent has an impossible pre-ledger current-index state"
            )
        _unlink_durable(intent_path)
        return
    if observed_ledger != after_ledger:
        raise FleetTransactionError(
            "fleet ledger differs from both sides of its durable save intent"
        )
    if intent["index_required"]:
        index_after = intent.get("index_after")
        if not isinstance(index_after, dict):
            raise FleetTransactionError("fleet save intent lacks its current-index commit")
        encoded = _canonical_json(index_after)
        if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != after_index:
            raise FleetTransactionError("fleet save intent current-index hash is invalid")
        if observed_index == before_index:
            io.atomic_write_text(current_path, encoded)
            observed_index = _file_digest_or_none(current_path)
        if observed_index != after_index:
            raise FleetTransactionError(
                "fleet current index differs from its interrupted save transaction"
            )
    elif observed_index != before_index or after_index is not None:
        raise FleetTransactionError(
            "historical fleet ledger save unexpectedly changed CURRENT.json"
        )
    _unlink_durable(intent_path)


def _read_current_index(directory: Path) -> dict[str, Any] | None:
    _recover_interrupted_save(directory)
    return _read_current_index_raw(directory)


def save_ledger(directory: Path, ledger: Mapping[str, Any], *, now: float) -> None:
    material = dict(ledger)
    material["updated_at"] = float(now)
    generation = material.get("rollout_generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise FleetTransactionError("cannot save a fleet ledger without its generation")
    ledgers = directory / LEDGERS_DIRECTORY
    if ledgers.is_symlink():
        raise FleetTransactionError("fleet ledgers directory is symlinked")
    ledgers.mkdir(parents=True, exist_ok=True)
    path = ledger_path(directory, generation)
    payload = _canonical_json(material)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    current = _read_current_index(directory)
    index_required = current is None or int(current["current_generation"]) <= generation
    index = (
        _current_index_payload(
            directory=directory, material=material, ledger_sha256=digest
        )
        if index_required
        else None
    )
    index_payload = None if index is None else _canonical_json(index)
    intent_path = directory / SAVE_INTENT_FILENAME
    if intent_path.exists() or intent_path.is_symlink():
        raise FleetTransactionError(
            "an unreconciled fleet ledger save intent already exists"
        )
    save_intent = {
        "schema_version": STATE_SCHEMA_VERSION,
        "kind": "schema5-fleet-ledger-save-v1",
        "rollout_generation": generation,
        "ledger_path": str(path.resolve()),
        "ledger_before_sha256": _file_digest_or_none(path),
        "ledger_after_sha256": digest,
        "index_required": index_required,
        "index_before_sha256": _file_digest_or_none(directory / CURRENT_FILENAME),
        "index_after_sha256": (
            None
            if index_payload is None
            else hashlib.sha256(index_payload.encode("utf-8")).hexdigest()
        ),
        "index_after": index,
    }
    io.atomic_write_text(intent_path, _canonical_json(save_intent))
    io.atomic_write_text(path, payload)
    if index_payload is not None:
        io.atomic_write_text(directory / CURRENT_FILENAME, index_payload)
    _unlink_durable(intent_path)
    # Keep the in-memory object synchronized for callers that perform several durable
    # transitions while holding the pool lock.
    if isinstance(ledger, dict):
        ledger["updated_at"] = float(now)


def _new_ledger(
    *,
    pool_root: Path,
    pool_id: str,
    fleet_sha256: str,
    rollout_generation: int,
    replica_ids: Sequence[str],
    now: float,
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "pool_root": str(pool_root),
        "pool_id": pool_id,
        "fleet_sha256": fleet_sha256,
        "rollout_generation": rollout_generation,
        "visibility_grace_seconds": DEFAULT_VISIBILITY_GRACE_SECONDS,
        "created_at": float(now),
        "updated_at": float(now),
        "replicas": {
            replica_id: {
                "attempts": [],
                "health": None,
            }
            for replica_id in replica_ids
        },
    }


def _validate_attempt(
    attempt: Any,
    *,
    replica_id: str,
    generation: int,
    directory: Path,
) -> None:
    if not isinstance(attempt, dict):
        raise FleetTransactionError(f"non-object attempt for {replica_id}")
    required = {
        "intent_token",
        "rollout_generation",
        "state",
        "created_at",
        "submit_started_at",
        "submission_attempts",
        "sbatch_path",
        "sbatch_sha256",
        "submission_transport",
        "submission_argv_sha256",
        "scheduler_comment",
        "job_id",
        "submitted_at",
        "committed_at",
        "terminal_at",
        "last_seen_at",
        "missing_since",
        "last_error",
        "launch_kind",
        "lifecycle",
        "predecessor_job_id",
        "predecessor_end_at",
        "allocated_gpus",
        "scheduler_start_at",
        "scheduler_end_at",
        "scheduler_time_limit_seconds",
        "ready_probe_count",
        "last_ready_probe_at",
        "promoted_at",
        "retire_requested_at",
        "last_retire_attempt_at",
        "retire_attempts",
        "retire_error",
    }
    if set(attempt) != required:
        raise FleetTransactionError(
            f"attempt fields drifted for {replica_id}: {sorted(set(attempt) ^ required)}"
        )
    token = attempt["intent_token"]
    if not isinstance(token, str) or _TOKEN_RE.fullmatch(token) is None:
        raise FleetTransactionError(f"invalid fleet intent token for {replica_id}")
    if attempt["rollout_generation"] != generation:
        raise FleetTransactionError(f"attempt generation drift for {replica_id}")
    if attempt["state"] not in {
        "prepared",
        "submitting",
        "submitted",
        "committed",
        "missing",
        "terminal",
        "submission_failed",
    }:
        raise FleetTransactionError(f"invalid attempt state for {replica_id}")
    if attempt["launch_kind"] not in {"primary", "handoff"}:
        raise FleetTransactionError(f"invalid launch kind for {replica_id}")
    if attempt["lifecycle"] not in {
        "primary",
        "standby",
        "promoted",
        "retiring",
    }:
        raise FleetTransactionError(f"invalid allocation lifecycle for {replica_id}")
    if attempt["launch_kind"] == "primary":
        if (
            attempt["predecessor_job_id"] is not None
            or attempt["predecessor_end_at"] is not None
        ):
            raise FleetTransactionError(
                f"primary fleet intent unexpectedly has a predecessor for {replica_id}"
            )
        if attempt["lifecycle"] not in {"primary", "retiring"}:
            raise FleetTransactionError(
                f"primary launch has impossible lifecycle for {replica_id}"
            )
    else:
        if not str(attempt["predecessor_job_id"] or "").isdigit():
            raise FleetTransactionError(
                f"handoff intent lacks an exact predecessor for {replica_id}"
            )
        if (
            not isinstance(attempt["predecessor_end_at"], (int, float))
            or isinstance(attempt["predecessor_end_at"], bool)
        ):
            raise FleetTransactionError(
                f"handoff intent lacks an exact predecessor end for {replica_id}"
            )
        if attempt["lifecycle"] not in {"standby", "promoted", "retiring"}:
            raise FleetTransactionError(
                f"handoff launch has impossible lifecycle for {replica_id}"
            )
    for field in (
        "allocated_gpus",
        "ready_probe_count",
        "retire_attempts",
    ):
        if (
            not isinstance(attempt[field], int)
            or isinstance(attempt[field], bool)
            or attempt[field] < 0
        ):
            raise FleetTransactionError(f"invalid {field} for {replica_id}")
    if (
        not isinstance(attempt["submission_attempts"], int)
        or isinstance(attempt["submission_attempts"], bool)
        or attempt["submission_attempts"] < 0
    ):
        raise FleetTransactionError(f"invalid submission count for {replica_id}")
    if not isinstance(attempt["sbatch_path"], str) or not Path(
        attempt["sbatch_path"]
    ).is_absolute():
        raise FleetTransactionError(f"non-absolute sbatch path for {replica_id}")
    expected_parent = _lexical_absolute_path(
        directory / "sbatch" / f"g{generation:06d}",
        description=f"fleet sbatch directory for {replica_id}",
    )
    observed_path = _lexical_absolute_path(
        attempt["sbatch_path"],
        description=f"fleet sbatch path for {replica_id}",
    )
    if observed_path.parent != expected_parent:
        raise FleetTransactionError(
            f"sbatch path escapes fleet transaction root for {replica_id}"
        )
    try:
        resolved_parent = observed_path.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FleetTransactionError(
            f"sbatch parent is unavailable for {replica_id}: {exc}"
        ) from exc
    if resolved_parent != expected_parent or observed_path.parent.is_symlink():
        raise FleetTransactionError(
            f"sbatch path traverses a symlink for {replica_id}"
        )
    safe_replica = re.sub(r"[^A-Za-z0-9_.-]+", "_", replica_id)
    if observed_path.name != f"{safe_replica}.{token}.sbatch":
        raise FleetTransactionError(f"sbatch filename does not bind intent for {replica_id}")
    if _SHA256_RE.fullmatch(str(attempt["sbatch_sha256"])) is None:
        raise FleetTransactionError(f"invalid sbatch hash for {replica_id}")
    if (
        attempt["submission_transport"] != STDIN_EXACT_SUBMISSION_TRANSPORT
        or attempt["submission_argv_sha256"]
        != submission_argv_sha256(str(attempt["scheduler_comment"]))
    ):
        raise FleetTransactionError(
            f"fleet submission transport drifted for {replica_id}"
        )
    for field in (
        "created_at",
        "submit_started_at",
        "submitted_at",
        "committed_at",
        "terminal_at",
        "last_seen_at",
        "missing_since",
        "last_ready_probe_at",
        "promoted_at",
        "retire_requested_at",
        "last_retire_attempt_at",
        "predecessor_end_at",
        "scheduler_start_at",
        "scheduler_end_at",
        "scheduler_time_limit_seconds",
    ):
        value = attempt[field]
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise FleetTransactionError(f"invalid {field} for {replica_id}")
    if attempt["retire_error"] is not None and not isinstance(
        attempt["retire_error"], str
    ):
        raise FleetTransactionError(f"invalid retire_error for {replica_id}")
    if attempt["last_error"] is not None and not isinstance(
        attempt["last_error"], str
    ):
        raise FleetTransactionError(f"invalid last_error for {replica_id}")
    if attempt["lifecycle"] == "standby" and attempt["promoted_at"] is not None:
        raise FleetTransactionError(
            f"standby unexpectedly has promotion time for {replica_id}"
        )
    if attempt["lifecycle"] == "promoted" and (
        attempt["launch_kind"] != "handoff"
        or attempt["promoted_at"] is None
        or attempt["ready_probe_count"] < 2
    ):
        raise FleetTransactionError(
            f"promoted handoff lacks readiness evidence for {replica_id}"
        )
    if (attempt["retire_attempts"] == 0) != (
        attempt["last_retire_attempt_at"] is None
    ):
        raise FleetTransactionError(
            f"retirement attempt evidence drifted for {replica_id}"
        )
    if attempt["scheduler_time_limit_seconds"] is not None and (
        not isinstance(attempt["scheduler_time_limit_seconds"], int)
        or isinstance(attempt["scheduler_time_limit_seconds"], bool)
        or attempt["scheduler_time_limit_seconds"] < 1
    ):
        raise FleetTransactionError(
            f"invalid scheduler time limit for {replica_id}"
        )
    if attempt["job_id"] is not None and not str(attempt["job_id"]).isdigit():
        raise FleetTransactionError(f"invalid Slurm job id for {replica_id}")


def _validate_health(health: Any, *, replica_id: str) -> None:
    if health is None:
        return
    required = {
        "job_id",
        "endpoint",
        "observer_generation",
        "first_failure_at",
        "last_failure_at",
        "last_probe_at",
        "consecutive_failures",
        "health_failures",
        "models_failures",
        "cancel_state",
        "cancel_requested_at",
        "cancel_completed_at",
        "cancel_error",
        "cancel_attempts",
        "last_cancel_attempt_at",
        "next_cancel_eligible_at",
        "alert_id",
    }
    if not isinstance(health, dict) or set(health) != required:
        raise FleetTransactionError(f"health fields drifted for {replica_id}")
    if not str(health["job_id"]).isdigit() or not isinstance(health["endpoint"], str):
        raise FleetTransactionError(f"invalid health identity for {replica_id}")
    if health["cancel_state"] not in {
        None,
        "requested",
        "retryable",
        "accepted",
        "exhausted",
    }:
        raise FleetTransactionError(f"invalid cancellation state for {replica_id}")
    for field in (
        "observer_generation",
        "consecutive_failures",
        "health_failures",
        "models_failures",
        "cancel_attempts",
    ):
        if not isinstance(health[field], int) or isinstance(health[field], bool) or health[field] < 0:
            raise FleetTransactionError(f"invalid health counter for {replica_id}")
    if health["observer_generation"] < 1:
        raise FleetTransactionError(f"invalid health observer generation for {replica_id}")
    for field in (
        "first_failure_at",
        "last_failure_at",
        "last_probe_at",
        "cancel_requested_at",
        "cancel_completed_at",
        "last_cancel_attempt_at",
        "next_cancel_eligible_at",
    ):
        value = health[field]
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise FleetTransactionError(f"invalid health timestamp for {replica_id}")


def _validate_ledger_material(
    ledger: Any,
    *,
    directory: Path,
    canonical_root: Path,
    pool_id: str,
    fleet_sha256: str,
    rollout_generation: int,
    replica_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate one generation without repairing or publishing any state."""

    required_root = {
        "schema_version",
        "pool_root",
        "pool_id",
        "fleet_sha256",
        "rollout_generation",
        "visibility_grace_seconds",
        "created_at",
        "updated_at",
        "replicas",
    }
    if not isinstance(ledger, dict) or set(ledger) != required_root:
        raise FleetTransactionError("fleet ledger root fields drifted")
    if (
        ledger["schema_version"] != STATE_SCHEMA_VERSION
        or ledger["pool_root"] != str(canonical_root)
        or ledger["pool_id"] != pool_id
        or ledger["fleet_sha256"] != fleet_sha256
        or ledger["rollout_generation"] != rollout_generation
        or ledger["visibility_grace_seconds"] != DEFAULT_VISIBILITY_GRACE_SECONDS
        or not isinstance(ledger["replicas"], dict)
        or set(ledger["replicas"]) != set(replica_ids)
    ):
        raise FleetTransactionError("fleet ledger immutable identity drifted")
    for replica_id in replica_ids:
        record = ledger["replicas"][replica_id]
        if not isinstance(record, dict) or set(record) != {"attempts", "health"}:
            raise FleetTransactionError(
                f"replica ledger fields drifted for {replica_id}"
            )
        attempts = record["attempts"]
        if not isinstance(attempts, list):
            raise FleetTransactionError(
                f"replica attempts are not an array for {replica_id}"
            )
        tokens: set[str] = set()
        job_ids: set[str] = set()
        for attempt in attempts:
            _validate_attempt(
                attempt,
                replica_id=replica_id,
                generation=rollout_generation,
                directory=directory,
            )
            parsed_comment = parse_intent_comment(attempt["scheduler_comment"])
            if (
                parsed_comment is None
                or parsed_comment["pool"] != pool_id
                or parsed_comment["replica"] != replica_id
                or parsed_comment["generation"] != str(rollout_generation)
                or parsed_comment["intent"] != attempt["intent_token"]
                or parsed_comment["fleet"] != fleet_sha256
            ):
                raise FleetTransactionError(
                    f"scheduler comment does not bind fleet intent for {replica_id}"
                )
            receipt_path = _submission_absence_receipt_path(
                directory,
                generation=rollout_generation,
                replica_id=replica_id,
                intent_token=str(attempt["intent_token"]),
            )
            receipt_reference = _absence_receipt_reference(attempt)
            if os.path.lexists(receipt_path) or receipt_reference is not None:
                _validate_submission_absence_receipt(
                    directory,
                    ledger,
                    replica_id=replica_id,
                    attempt=attempt,
                    require_reference=receipt_reference is not None,
                )
            if attempt["intent_token"] in tokens:
                raise FleetTransactionError(
                    f"duplicate intent token for {replica_id}"
                )
            tokens.add(attempt["intent_token"])
            if attempt["job_id"] is not None:
                if str(attempt["job_id"]) in job_ids:
                    raise FleetTransactionError(
                        f"duplicate committed job id for {replica_id}"
                    )
                job_ids.add(str(attempt["job_id"]))
        nonterminal = [
            item
            for item in attempts
            if item["state"]
            in {"prepared", "submitting", "submitted", "committed", "missing"}
        ]
        if len(nonterminal) > 2:
            raise FleetTransactionError(
                f"multiple current attempts for {replica_id}"
            )
        if len(nonterminal) == 2:
            lifecycles = sorted(item["lifecycle"] for item in nonterminal)
            if lifecycles not in (
                ["primary", "standby"],
                ["promoted", "retiring"],
                ["promoted", "standby"],
                ["retiring", "standby"],
            ):
                raise FleetTransactionError(
                    f"invalid rolling-handoff overlap for {replica_id}: {lifecycles}"
                )
            standby = next(
                (item for item in nonterminal if item["lifecycle"] == "standby"),
                None,
            )
            predecessor = next(
                (
                    item
                    for item in nonterminal
                    if item["lifecycle"] in {"primary", "promoted", "retiring"}
                ),
                None,
            )
            if standby is not None and (
                predecessor is None
                or standby["predecessor_job_id"] != predecessor["job_id"]
            ):
                raise FleetTransactionError(
                    f"handoff predecessor binding drifted for {replica_id}"
                )
        _validate_health(record["health"], replica_id=replica_id)
    return ledger


def _read_ledger_file(
    path: Path,
    *,
    directory: Path,
    canonical_root: Path,
    pool_id: str,
    fleet_sha256: str,
    rollout_generation: int,
    replica_ids: Sequence[str],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FleetTransactionError(f"fleet ledger is not a regular file: {path}")
    try:
        ledger = _strict_json_loads(
            path.read_text(encoding="utf-8"), artifact=f"fleet ledger {path}"
        )
    except (OSError, UnicodeError) as exc:
        raise FleetTransactionError(f"cannot parse fleet ledger {path}: {exc}") from exc
    return _validate_ledger_material(
        ledger,
        directory=directory,
        canonical_root=canonical_root,
        pool_id=pool_id,
        fleet_sha256=fleet_sha256,
        rollout_generation=rollout_generation,
        replica_ids=replica_ids,
    )


def load_or_create_ledger(
    directory: Path,
    *,
    pool_root: str | Path,
    pool_id: str,
    fleet_sha256: str,
    rollout_generation: int,
    replica_ids: Sequence[str],
    now: float,
    allow_historical: bool = False,
) -> dict[str, Any]:
    """Load exact fleet state or atomically initialize it while the lock is held."""

    canonical_root = Path(pool_root).expanduser().resolve()
    if _SHA256_RE.fullmatch(fleet_sha256) is None:
        raise FleetTransactionError("fleet transaction requires a lowercase SHA-256")
    if not isinstance(rollout_generation, int) or rollout_generation < 1:
        raise FleetTransactionError("fleet transaction requires a positive generation")
    if len(replica_ids) != len(set(replica_ids)) or not replica_ids:
        raise FleetTransactionError("fleet transaction replica identities are not unique")
    current = _read_current_index(directory)
    if current is not None:
        if (
            current["pool_root"] != str(canonical_root)
            or current["pool_id"] != pool_id
            or current["fleet_sha256"] != fleet_sha256
        ):
            raise FleetTransactionError("fleet current-index immutable identity drifted")
        if (
            int(current["current_generation"]) > rollout_generation
            and not allow_historical
        ):
            raise FleetTransactionError(
                f"stale generation {rollout_generation} cannot supersede current fleet "
                f"generation {current['current_generation']}"
            )
    path = ledger_path(directory, rollout_generation)
    if not path.exists():
        ledger = _new_ledger(
            pool_root=canonical_root,
            pool_id=pool_id,
            fleet_sha256=fleet_sha256,
            rollout_generation=rollout_generation,
            replica_ids=replica_ids,
            now=now,
        )
        save_ledger(directory, ledger, now=now)
        return ledger
    ledger = _read_ledger_file(
        path,
        directory=directory,
        canonical_root=canonical_root,
        pool_id=pool_id,
        fleet_sha256=fleet_sha256,
        rollout_generation=rollout_generation,
        replica_ids=replica_ids,
    )
    if current is None or int(current["current_generation"]) < rollout_generation:
        # Atomically advance CURRENT only after the complete pre-existing generation
        # ledger validates.  This recovers a crash after publishing gNNNN.json but before
        # publishing its index.
        save_ledger(directory, ledger, now=now)
    elif int(current["current_generation"]) == rollout_generation:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != current["ledger_sha256"]:
            raise FleetTransactionError("fleet current ledger hash differs from index")
    return ledger


def load_generation_ledgers(
    directory: Path,
    *,
    pool_root: str | Path,
    pool_id: str,
    fleet_sha256: str,
    current_generation: int,
    replica_ids: Sequence[str],
    now: float,
) -> list[dict[str, Any]]:
    """Load every sealed generation up through the current supervisor generation."""

    # This call validates/creates the current ledger and advances the atomic index.
    current = load_or_create_ledger(
        directory,
        pool_root=pool_root,
        pool_id=pool_id,
        fleet_sha256=fleet_sha256,
        rollout_generation=current_generation,
        replica_ids=replica_ids,
        now=now,
    )
    ledger_dir = directory / LEDGERS_DIRECTORY
    paths = sorted(ledger_dir.glob("g*.json"))
    generations: dict[int, dict[str, Any]] = {current_generation: current}
    for path in paths:
        match = re.fullmatch(r"g([0-9]{6})\.json", path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise FleetTransactionError(f"invalid generation ledger entry: {path}")
        generation = int(match.group(1))
        if generation > current_generation:
            raise FleetTransactionError(
                f"future fleet ledger g{generation:06d} exceeds current generation"
            )
        if generation == current_generation:
            continue
        generations[generation] = load_or_create_ledger(
            directory,
            pool_root=pool_root,
            pool_id=pool_id,
            fleet_sha256=fleet_sha256,
            rollout_generation=generation,
            replica_ids=replica_ids,
            now=now,
            allow_historical=True,
        )
    return [generations[generation] for generation in sorted(generations)]


def read_generation_ledgers(
    directory: Path,
    *,
    pool_root: str | Path,
    pool_id: str,
    fleet_sha256: str,
    current_generation: int,
    replica_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Read and validate every generation without recovery, creation, or mutation.

    Callers must hold :func:`read_transaction_lock` (or the writer lock).  An
    interrupted two-file publication is reported rather than repaired, and the current
    ledger is cryptographically rebound to ``CURRENT.json`` after every file is read.
    """

    canonical_root = Path(pool_root).expanduser().resolve()
    if directory != state_directory(canonical_root):
        raise FleetTransactionError("fleet read directory does not match the pool root")
    if _SHA256_RE.fullmatch(fleet_sha256) is None:
        raise FleetTransactionError("fleet transaction requires a lowercase SHA-256")
    if (
        not isinstance(current_generation, int)
        or isinstance(current_generation, bool)
        or current_generation < 1
    ):
        raise FleetTransactionError("fleet transaction requires a positive generation")
    if len(replica_ids) != len(set(replica_ids)) or not replica_ids:
        raise FleetTransactionError(
            "fleet transaction replica identities are not unique"
        )
    save_intent = directory / SAVE_INTENT_FILENAME
    if save_intent.exists() or save_intent.is_symlink():
        raise FleetTransactionError(
            "fleet ledger publication is in progress or requires locked recovery"
        )
    current_before = _read_current_index_raw(directory)
    if current_before is None:
        raise FleetTransactionError("fleet transaction state lacks CURRENT.json")
    if (
        current_before["pool_root"] != str(canonical_root)
        or current_before["pool_id"] != pool_id
        or current_before["fleet_sha256"] != fleet_sha256
        or current_before["current_generation"] != current_generation
    ):
        raise FleetTransactionError("fleet current-index immutable identity drifted")
    ledger_dir = directory / LEDGERS_DIRECTORY
    if ledger_dir.is_symlink() or not ledger_dir.is_dir():
        raise FleetTransactionError("fleet generation-ledger directory is invalid")
    generations: dict[int, dict[str, Any]] = {}
    paths = sorted(ledger_dir.iterdir(), key=lambda item: item.name)
    for path in paths:
        match = re.fullmatch(r"g([0-9]{6})\.json", path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise FleetTransactionError(f"invalid generation ledger entry: {path}")
        generation = int(match.group(1))
        if generation > current_generation:
            raise FleetTransactionError(
                f"future fleet ledger g{generation:06d} exceeds current generation"
            )
        generations[generation] = _read_ledger_file(
            path,
            directory=directory,
            canonical_root=canonical_root,
            pool_id=pool_id,
            fleet_sha256=fleet_sha256,
            rollout_generation=generation,
            replica_ids=replica_ids,
        )
    if current_generation not in generations:
        raise FleetTransactionError("fleet current generation ledger is missing")
    current_path = ledger_path(directory, current_generation)
    if (
        str(current_path.resolve()) != current_before["ledger_path"]
        or hashlib.sha256(current_path.read_bytes()).hexdigest()
        != current_before["ledger_sha256"]
    ):
        raise FleetTransactionError("fleet CURRENT ledger bytes are not attested")
    current_after = _read_current_index_raw(directory)
    if current_after != current_before:
        raise FleetTransactionError("fleet CURRENT changed during read-only reconciliation")
    return tuple(generations[generation] for generation in sorted(generations))


def reconcile_scheduler_rows(
    rows: Sequence[SchedulerRow],
    ledgers: Sequence[Mapping[str, Any]],
    *,
    pool_id: str,
    fleet_sha256: str,
    replica_profiles: Mapping[str, str],
    replica_job_names: Mapping[str, str],
    replica_qos: Mapping[str, str] | None = None,
) -> FleetReconciliation:
    """Bind joined scheduler truth to durable intents without changing either side."""

    expected_ids = set(replica_profiles)
    if (
        not expected_ids
        or set(replica_job_names) != expected_ids
        or len(set(replica_job_names.values())) != len(replica_job_names)
        or (
            replica_qos is not None
            and (
                set(replica_qos) != expected_ids
                or any(
                    re.fullmatch(
                        r"[A-Za-z0-9_.-]+", str(replica_qos[item])
                    )
                    is None
                    for item in expected_ids
                )
            )
        )
    ):
        raise FleetTransactionError("fleet reconciliation identities are not bijective")
    attempts_by_token: dict[
        str, tuple[str, Mapping[str, Any], Mapping[str, Any], int]
    ] = {}
    recorded_job_ids: dict[str, str] = {}
    for ledger in ledgers:
        if (
            not isinstance(ledger, Mapping)
            or ledger.get("pool_id") != pool_id
            or ledger.get("fleet_sha256") != fleet_sha256
            or set(ledger.get("replicas", {})) != expected_ids
        ):
            raise FleetTransactionError("fleet reconciliation ledger identity drifted")
        generation = ledger.get("rollout_generation")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 1
        ):
            raise FleetTransactionError("fleet reconciliation generation is invalid")
        replicas = ledger["replicas"]
        for replica_id in sorted(expected_ids):
            record = replicas[replica_id]
            for attempt in record["attempts"]:
                token = str(attempt["intent_token"])
                if token in attempts_by_token:
                    raise FleetTransactionError(
                        f"fleet intent token {token} is not globally unique"
                    )
                attempts_by_token[token] = (
                    replica_id,
                    attempt,
                    record,
                    generation,
                )
                job_id = attempt.get("job_id")
                if job_id is not None:
                    owner = recorded_job_ids.setdefault(str(job_id), token)
                    if owner != token:
                        raise FleetTransactionError(
                            f"fleet job id {job_id} is bound to multiple intents"
                        )
                try:
                    _sbatch_path, sbatch_raw = _stable_regular_preimage(
                        str(attempt["sbatch_path"]),
                        description=(
                            f"immutable sbatch provenance for {replica_id}"
                        ),
                        read_only=True,
                    )
                except FleetTransactionError as exc:
                    raise FleetTransactionError(
                        f"immutable sbatch provenance drift for {replica_id}: "
                        f"{exc}"
                    ) from exc
                if (
                    hashlib.sha256(sbatch_raw).hexdigest()
                    != attempt["sbatch_sha256"]
                ):
                    raise FleetTransactionError(
                        f"immutable sbatch provenance drift for {replica_id}"
                    )

    scheduler_job_ids: set[str] = set()
    by_token: dict[str, list[SchedulerRow]] = {}
    active_by_replica: dict[
        str, list[tuple[SchedulerRow, Mapping[str, Any]]]
    ] = {}
    allocations: list[ReconciledFleetAllocation] = []
    ignored_terminal: list[str] = []
    for row in rows:
        if row.job_id in scheduler_job_ids:
            raise FleetTransactionError(
                f"duplicate scheduler row for fleet job {row.job_id}"
            )
        scheduler_job_ids.add(row.job_id)
        parsed = parse_intent_comment(row.comment)
        if parsed is None:
            if terminal_state(row.state):
                ignored_terminal.append(row.job_id)
                continue
            raise FleetTransactionError(
                f"unmappable schema-5 fleet job {row.job_id} lacks "
                "transactional intent provenance"
            )
        matched = attempts_by_token.get(parsed["intent"])
        if matched is None:
            if terminal_state(row.state):
                ignored_terminal.append(row.job_id)
                continue
            raise FleetTransactionError(
                f"scheduler exposes unknown fleet intent {parsed['intent']} "
                f"as job {row.job_id}"
            )
        replica_id, attempt, record, generation = matched
        expected_profile = replica_profiles[replica_id]
        if (
            row.comment != attempt["scheduler_comment"]
            or parsed["pool"] != pool_id
            or parsed["profile"] != expected_profile
            or parsed["replica"] != replica_id
            or parsed["generation"] != str(generation)
            or parsed["fleet"] != fleet_sha256
        ):
            raise FleetTransactionError(
                f"scheduler intent comment drift for job {row.job_id}"
            )
        if (
            row.job_name != replica_job_names[replica_id]
            or (
                replica_qos is not None
                and row.qos != replica_qos[replica_id]
            )
            or not command_binds_stdin_submission(
                row.command, str(attempt["scheduler_comment"])
            )
        ):
            raise FleetTransactionError(
                f"scheduler provenance drift for fleet job {row.job_id}"
            )
        if attempt.get("job_id") not in {None, str(row.job_id)}:
            raise FleetTransactionError(
                f"fleet intent {parsed['intent']} changed job id"
            )
        by_token.setdefault(parsed["intent"], []).append(row)
        if not terminal_state(row.state):
            active_by_replica.setdefault(replica_id, []).append((row, attempt))
        allocations.append(
            ReconciledFleetAllocation(
                replica_id=replica_id,
                profile=expected_profile,
                ledger_generation=generation,
                row=row,
                attempt=MappingProxyType(dict(attempt)),
                health=(
                    None
                    if record["health"] is None
                    else MappingProxyType(dict(record["health"]))
                ),
            )
        )
    duplicate_tokens = {
        token: [row.job_id for row in token_rows]
        for token, token_rows in by_token.items()
        if len(token_rows) != 1
    }
    duplicate_replicas: dict[str, list[str]] = {}
    for replica_id, replica_rows in active_by_replica.items():
        if len(replica_rows) <= 1:
            continue
        lifecycles = sorted(
            str(attempt.get("lifecycle")) for _row, attempt in replica_rows
        )
        allowed = (
            len(replica_rows) == 2
            and lifecycles
            in (
                ["primary", "standby"],
                ["promoted", "retiring"],
                ["promoted", "standby"],
                ["retiring", "standby"],
            )
        )
        if not allowed:
            duplicate_replicas[replica_id] = [
                row.job_id for row, _attempt in replica_rows
            ]
            continue
        standby = next(
            (
                (row, attempt)
                for row, attempt in replica_rows
                if attempt.get("lifecycle") == "standby"
            ),
            None,
        )
        predecessor = next(
            (
                (row, attempt)
                for row, attempt in replica_rows
                if attempt.get("lifecycle")
                in {"primary", "promoted", "retiring"}
            ),
            None,
        )
        if standby is not None and (
            predecessor is None
            or standby[1].get("predecessor_job_id") != predecessor[0].job_id
        ):
            duplicate_replicas[replica_id] = [
                row.job_id for row, _attempt in replica_rows
            ]
    if duplicate_tokens or duplicate_replicas:
        raise FleetTransactionError(
            "ambiguous duplicate fleet jobs: "
            f"tokens={duplicate_tokens}, replicas={duplicate_replicas}"
        )
    return FleetReconciliation(
        allocations=tuple(
            sorted(
                allocations,
                key=lambda item: (item.replica_id, int(item.row.job_id)),
            )
        ),
        ignored_terminal_job_ids=tuple(
            sorted(ignored_terminal, key=int)
        ),
    )


def intent_comment(
    *,
    pool_id: str,
    profile: str,
    replica_id: str,
    rollout_generation: int,
    intent_token: str,
    fleet_sha256: str,
) -> str:
    values = (pool_id, profile, replica_id)
    if any(not value or any(marker in value for marker in (";", "=", "\n", "\r")) for value in values):
        raise FleetTransactionError("unsafe fleet intent identity")
    if _TOKEN_RE.fullmatch(intent_token) is None or _SHA256_RE.fullmatch(fleet_sha256) is None:
        raise FleetTransactionError("invalid fleet intent token or hash")
    comment = (
        f"asys-s5-fleet:pool={pool_id};profile={profile};replica={replica_id};"
        f"generation={rollout_generation};intent={intent_token};fleet={fleet_sha256}"
    )
    if len(comment.encode("utf-8")) > 255:
        raise FleetTransactionError("fleet scheduler comment exceeds Slurm limit")
    return comment


def parse_intent_comment(comment: str) -> dict[str, str] | None:
    prefix = "asys-s5-fleet:"
    if not comment.startswith(prefix):
        return None
    fields: dict[str, str] = {}
    for component in comment[len(prefix) :].split(";"):
        if component.count("=") != 1:
            return None
        key, value = component.split("=", 1)
        if not key or not value or key in fields:
            return None
        fields[key] = value
    if set(fields) != {"pool", "profile", "replica", "generation", "intent", "fleet"}:
        return None
    if (
        not fields["generation"].isdigit()
        or _TOKEN_RE.fullmatch(fields["intent"]) is None
        or _SHA256_RE.fullmatch(fields["fleet"]) is None
    ):
        return None
    return fields


def submission_argv(scheduler_comment: str) -> list[str]:
    return ["sbatch", "--parsable", f"--comment={scheduler_comment}"]


def submission_argv_sha256(scheduler_comment: str) -> str:
    return hashlib.sha256(
        json.dumps(
            submission_argv(scheduler_comment),
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def command_binds_stdin_submission(
    command: str, scheduler_comment: str
) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    normalized = [
        "sbatch" if Path(tokens[0]).name == "sbatch" else tokens[0],
        *tokens[1:],
    ]
    return normalized == submission_argv(scheduler_comment)


def prepare_attempt(
    directory: Path,
    ledger: dict[str, Any],
    *,
    replica_id: str,
    profile: str,
    pool_id: str,
    fleet_sha256: str,
    rollout_generation: int,
    sbatch_text: str,
    now: float,
    token_factory: Callable[[], str] | None = None,
    launch_kind: str = "primary",
    predecessor_job_id: str | None = None,
    predecessor_end_at: float | None = None,
    predecessor_attempt: Mapping[str, Any] | None = None,
    allocated_gpus: int = 1,
) -> dict[str, Any]:
    """Publish one immutable script and its durable pre-submit intent."""

    record = ledger["replicas"][replica_id]
    current = [
        item
        for item in record["attempts"]
        if item["state"] in {"prepared", "submitting", "submitted", "committed", "missing"}
    ]
    if launch_kind not in {"primary", "handoff"}:
        raise FleetTransactionError("fleet launch kind must be primary or handoff")
    if (
        not isinstance(allocated_gpus, int)
        or isinstance(allocated_gpus, bool)
        or allocated_gpus < 0
    ):
        raise FleetTransactionError(
            "fleet allocation GPU count must be nonnegative"
        )
    if launch_kind == "primary":
        if (
            current
            or predecessor_job_id is not None
            or predecessor_end_at is not None
            or predecessor_attempt is not None
        ):
            raise FleetTransactionError(
                f"replica {replica_id} already has a current intent"
            )
    else:
        stable = [
            item
            for item in current
            if item["lifecycle"] in {"primary", "promoted"}
            and str(item.get("job_id") or "").isdigit()
        ]
        if predecessor_attempt is not None and predecessor_attempt not in stable:
            if (
                predecessor_attempt.get("lifecycle") not in {"primary", "promoted"}
                or predecessor_attempt.get("state")
                not in {"submitted", "committed", "missing"}
                or not str(predecessor_attempt.get("job_id") or "").isdigit()
            ):
                raise FleetTransactionError(
                    f"handoff for {replica_id} has an invalid external predecessor"
                )
            stable.append(predecessor_attempt)
        if (
            len(stable) != 1
            or len(current) not in {0, 1}
            or (len(current) == 1 and current[0] is not stable[0])
            or predecessor_job_id != stable[0]["job_id"]
            or not isinstance(predecessor_end_at, (int, float))
            or isinstance(predecessor_end_at, bool)
        ):
            raise FleetTransactionError(
                f"handoff for {replica_id} lacks one exact promoted predecessor"
            )
    token = (token_factory or (lambda: secrets.token_hex(16)))()
    if _TOKEN_RE.fullmatch(token) is None:
        raise FleetTransactionError("intent token factory returned an invalid token")
    comment = intent_comment(
        pool_id=pool_id,
        profile=profile,
        replica_id=replica_id,
        rollout_generation=rollout_generation,
        intent_token=token,
        fleet_sha256=fleet_sha256,
    )
    safe_replica = re.sub(r"[^A-Za-z0-9_.-]+", "_", replica_id)
    sbatch_dir = _lexical_absolute_path(
        directory / "sbatch" / f"g{rollout_generation:06d}",
        description=f"fleet sbatch directory for {replica_id}",
    )
    if sbatch_dir.is_symlink() or sbatch_dir.parent.is_symlink():
        raise FleetTransactionError("fleet sbatch directory is symlinked")
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    sbatch_path = sbatch_dir / f"{safe_replica}.{token}.sbatch"
    payload = sbatch_text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if sbatch_path.exists():
        _existing_path, existing_payload = _stable_regular_preimage(
            sbatch_path,
            description=f"fleet sbatch script for {replica_id}",
            read_only=True,
        )
        if existing_payload != payload:
            raise FleetTransactionError(f"immutable fleet script collision: {sbatch_path}")
    else:
        io.atomic_write_text(sbatch_path, sbatch_text)
        sbatch_path.chmod(stat.S_IMODE(sbatch_path.stat().st_mode) & ~0o222)
        _fsync_directory(sbatch_path.parent)
    sealed_path, sealed_payload = _stable_regular_preimage(
        sbatch_path,
        description=f"fleet sbatch script for {replica_id}",
        read_only=True,
    )
    if hashlib.sha256(sealed_payload).hexdigest() != digest:
        raise FleetTransactionError(
            f"immutable fleet script hash drifted for {replica_id}"
        )
    attempt = {
        "intent_token": token,
        "rollout_generation": rollout_generation,
        "state": "prepared",
        "created_at": float(now),
        "submit_started_at": None,
        "submission_attempts": 0,
        "sbatch_path": str(sealed_path),
        "sbatch_sha256": digest,
        "submission_transport": STDIN_EXACT_SUBMISSION_TRANSPORT,
        "submission_argv_sha256": submission_argv_sha256(comment),
        "scheduler_comment": comment,
        "job_id": None,
        "submitted_at": None,
        "committed_at": None,
        "terminal_at": None,
        "last_seen_at": None,
        "missing_since": None,
        "last_error": None,
        "launch_kind": launch_kind,
        "lifecycle": "primary" if launch_kind == "primary" else "standby",
        "predecessor_job_id": predecessor_job_id,
        "predecessor_end_at": (
            None if predecessor_end_at is None else float(predecessor_end_at)
        ),
        "allocated_gpus": allocated_gpus,
        "scheduler_start_at": None,
        "scheduler_end_at": None,
        "scheduler_time_limit_seconds": None,
        "ready_probe_count": 0,
        "last_ready_probe_at": None,
        "promoted_at": None,
        "retire_requested_at": None,
        "last_retire_attempt_at": None,
        "retire_attempts": 0,
        "retire_error": None,
    }
    record["attempts"].append(attempt)
    save_ledger(directory, ledger, now=now)
    return attempt


def _require_exact_attempt_owner(
    directory: Path,
    ledger: Mapping[str, Any],
    *,
    replica_id: str,
    attempt: Mapping[str, Any],
) -> int:
    generation = ledger.get("rollout_generation")
    replicas = ledger.get("replicas")
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
        or not isinstance(replicas, Mapping)
        or not isinstance(replicas.get(replica_id), Mapping)
        or not isinstance(replicas[replica_id].get("attempts"), list)
    ):
        raise FleetTransactionError(
            f"fleet submission ledger identity is invalid for {replica_id}"
        )
    owned = [
        candidate
        for candidate in replicas[replica_id]["attempts"]
        if candidate is attempt
    ]
    if len(owned) != 1:
        raise FleetTransactionError(
            f"fleet submission is not the exact ledger attempt for {replica_id}"
        )
    _validate_attempt(
        attempt,
        replica_id=replica_id,
        generation=generation,
        directory=directory,
    )
    parsed = parse_intent_comment(str(attempt["scheduler_comment"]))
    if (
        parsed is None
        or parsed["pool"] != str(ledger.get("pool_id"))
        or parsed["replica"] != replica_id
        or parsed["generation"] != str(generation)
        or parsed["intent"] != str(attempt["intent_token"])
        or parsed["fleet"] != str(ledger.get("fleet_sha256"))
    ):
        raise FleetTransactionError(
            f"fleet submission comment does not bind the intent for {replica_id}"
        )
    return generation


def _verify_persisted_ledger_preimage(
    directory: Path,
    ledger: Mapping[str, Any],
    *,
    generation: int,
) -> None:
    """Prove the full in-memory transaction is the durable ledger preimage."""

    _path, observed = _stable_regular_preimage(
        ledger_path(directory, generation),
        description=f"fleet generation g{generation:06d} ledger",
        read_only=False,
    )
    expected = _canonical_json(ledger).encode("utf-8")
    if observed != expected:
        raise FleetTransactionError(
            f"fleet generation g{generation:06d} durable ledger drifted"
        )


def _submission_absence_receipt_path(
    directory: Path,
    *,
    generation: int,
    replica_id: str,
    intent_token: str,
) -> Path:
    safe_replica = re.sub(r"[^A-Za-z0-9_.-]+", "_", replica_id)
    return _lexical_absolute_path(
        directory
        / ABSENCE_RECEIPTS_DIRECTORY
        / f"g{generation:06d}"
        / f"{safe_replica}.{intent_token}.json",
        description=f"submission-absence receipt for {replica_id}",
    )


def _scheduler_row_receipt_payload(row: SchedulerRow) -> dict[str, Any]:
    return {
        "job_id": str(row.job_id),
        "job_name": str(row.job_name),
        "state": str(row.state),
        "partition": str(row.partition),
        "qos": str(row.qos),
        "node": str(row.node),
        "command": str(row.command),
        "comment": str(row.comment),
        "source": str(row.source),
        "start_timestamp": row.start_timestamp,
        "end_timestamp": row.end_timestamp,
        "time_limit_seconds": row.time_limit_seconds,
        "dependency": row.dependency,
    }


def _absence_receipt_reference(
    attempt: Mapping[str, Any],
) -> tuple[Path, str] | None:
    last_error = attempt.get("last_error")
    if not isinstance(last_error, str):
        return None
    match = _ABSENCE_RECEIPT_REFERENCE_RE.fullmatch(last_error)
    if match is None:
        if "absence_receipt_" in last_error:
            raise FleetTransactionError(
                "fleet submission-absence receipt reference is malformed"
            )
        return None
    return (
        _lexical_absolute_path(
            match.group("path"),
            description="fleet submission-absence receipt reference",
        ),
        match.group("sha256"),
    )


def _validate_submission_absence_receipt(
    directory: Path,
    ledger: Mapping[str, Any],
    *,
    replica_id: str,
    attempt: Mapping[str, Any],
    require_reference: bool,
) -> tuple[Path, str, dict[str, Any]]:
    generation = int(attempt["rollout_generation"])
    token = str(attempt["intent_token"])
    expected_path = _submission_absence_receipt_path(
        directory,
        generation=generation,
        replica_id=replica_id,
        intent_token=token,
    )
    reference = _absence_receipt_reference(attempt)
    if require_reference and reference is None:
        raise FleetTransactionError(
            "proven submission absence lacks its sealed receipt reference"
        )
    if reference is not None and reference[0] != expected_path:
        raise FleetTransactionError(
            "submission-absence receipt path does not bind the fleet intent"
        )
    observed_path, raw = _stable_regular_preimage(
        expected_path,
        description=f"submission-absence receipt for {replica_id}",
        read_only=True,
    )
    digest = hashlib.sha256(raw).hexdigest()
    if reference is not None and reference[1] != digest:
        raise FleetTransactionError(
            "submission-absence receipt hash differs from its ledger binding"
        )
    try:
        payload = _strict_json_loads(
            raw.decode("utf-8"),
            artifact=f"submission-absence receipt {observed_path}",
        )
    except UnicodeError as exc:
        raise FleetTransactionError(
            f"cannot decode submission-absence receipt: {exc}"
        ) from exc
    required = {
        "schema_version",
        "protocol",
        "pool_root",
        "pool_id",
        "fleet_sha256",
        "replica_id",
        "rollout_generation",
        "intent_token",
        "scheduler_comment",
        "sbatch_path",
        "sbatch_sha256",
        "submission_transport",
        "submission_argv_sha256",
        "submit_started_at",
        "captured_at",
        "recorded_at",
        "squeue_ok",
        "sacct_ok",
        "errors",
        "rows",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise FleetTransactionError(
            "submission-absence receipt fields are invalid"
        )
    immutable_expected = {
        "schema_version": 1,
        "protocol": "schema5-fleet-submit-absence-v1",
        "pool_root": ledger.get("pool_root"),
        "pool_id": ledger.get("pool_id"),
        "fleet_sha256": ledger.get("fleet_sha256"),
        "replica_id": replica_id,
        "rollout_generation": generation,
        "intent_token": token,
        "scheduler_comment": attempt.get("scheduler_comment"),
        "sbatch_path": attempt.get("sbatch_path"),
        "sbatch_sha256": attempt.get("sbatch_sha256"),
        "submission_transport": attempt.get("submission_transport"),
        "submission_argv_sha256": attempt.get("submission_argv_sha256"),
    }
    if any(payload.get(key) != value for key, value in immutable_expected.items()):
        raise FleetTransactionError(
            "submission-absence receipt identity differs from its fleet intent"
        )
    captured_at = payload.get("captured_at")
    recorded_at = payload.get("recorded_at")
    submit_started_at = payload.get("submit_started_at")
    rows = payload.get("rows")
    if (
        payload.get("squeue_ok") is not True
        or payload.get("sacct_ok") is not True
        or payload.get("errors") != []
        or not isinstance(rows, list)
        or not isinstance(captured_at, (int, float))
        or isinstance(captured_at, bool)
        or not isinstance(recorded_at, (int, float))
        or isinstance(recorded_at, bool)
        or not isinstance(submit_started_at, (int, float))
        or isinstance(submit_started_at, bool)
        or not isinstance(attempt.get("submit_started_at"), (int, float))
        or isinstance(attempt.get("submit_started_at"), bool)
        or float(submit_started_at)
        > float(attempt["submit_started_at"])
        or (
            attempt.get("state") == "submission_failed"
            and reference is not None
            and float(submit_started_at)
            != float(attempt["submit_started_at"])
        )
        or float(captured_at) - float(submit_started_at)
        < DEFAULT_VISIBILITY_GRACE_SECONDS
        or not (0.0 <= float(recorded_at) - float(captured_at) <= 60.0)
    ):
        raise FleetTransactionError(
            "submission-absence receipt scheduler proof is invalid"
        )
    row_ids: set[str] = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "job_id",
                "job_name",
                "state",
                "partition",
                "qos",
                "node",
                "command",
                "comment",
                "source",
                "start_timestamp",
                "end_timestamp",
                "time_limit_seconds",
                "dependency",
            }
            or not isinstance(row.get("job_id"), str)
            or not row["job_id"]
            or row["job_id"] in row_ids
            or row.get("source") not in {"squeue", "sacct"}
        ):
            raise FleetTransactionError(
                "submission-absence receipt scheduler rows are invalid"
            )
        row_ids.add(row["job_id"])
        if (
            token in str(row.get("comment"))
            or token in str(row.get("command"))
            or (
                attempt.get("job_id") is not None
                and row["job_id"] == str(attempt["job_id"])
            )
        ):
            raise FleetTransactionError(
                "submission-absence receipt contains the immutable intent"
            )
    return observed_path, digest, payload


def _publish_submission_absence_receipt(
    directory: Path,
    ledger: Mapping[str, Any],
    *,
    replica_id: str,
    attempt: Mapping[str, Any],
    snapshot: SchedulerSnapshot,
    now: float,
) -> tuple[Path, str]:
    generation = int(attempt["rollout_generation"])
    path = _submission_absence_receipt_path(
        directory,
        generation=generation,
        replica_id=replica_id,
        intent_token=str(attempt["intent_token"]),
    )
    if path.exists() or path.is_symlink():
        observed_path, digest, _payload = (
            _validate_submission_absence_receipt(
                directory,
                ledger,
                replica_id=replica_id,
                attempt=attempt,
                require_reference=False,
            )
        )
        return observed_path, digest
    receipt_dir = path.parent
    if receipt_dir.is_symlink() or receipt_dir.parent.is_symlink():
        raise FleetTransactionError(
            "submission-absence receipt directory is symlinked"
        )
    receipt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": "schema5-fleet-submit-absence-v1",
        "pool_root": ledger.get("pool_root"),
        "pool_id": ledger.get("pool_id"),
        "fleet_sha256": ledger.get("fleet_sha256"),
        "replica_id": replica_id,
        "rollout_generation": generation,
        "intent_token": str(attempt["intent_token"]),
        "scheduler_comment": attempt.get("scheduler_comment"),
        "sbatch_path": attempt.get("sbatch_path"),
        "sbatch_sha256": attempt.get("sbatch_sha256"),
        "submission_transport": attempt.get("submission_transport"),
        "submission_argv_sha256": attempt.get("submission_argv_sha256"),
        "submit_started_at": attempt.get("submit_started_at"),
        "captured_at": float(snapshot.captured_at),
        "recorded_at": float(now),
        "squeue_ok": snapshot.squeue_ok,
        "sacct_ok": snapshot.sacct_ok,
        "errors": list(snapshot.errors),
        "rows": [
            _scheduler_row_receipt_payload(row)
            for row in sorted(snapshot.rows, key=lambda item: item.job_id)
        ],
    }
    io.atomic_write_text(path, _canonical_json(payload))
    path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    _fsync_directory(path.parent)
    observed_path, digest, _validated = _validate_submission_absence_receipt(
        directory,
        ledger,
        replica_id=replica_id,
        attempt=attempt,
        require_reference=False,
    )
    return observed_path, digest


def record_proven_submission_absence(
    directory: Path,
    ledger: dict[str, Any],
    *,
    replica_id: str,
    attempt: dict[str, Any],
    snapshot: SchedulerSnapshot,
    now: float,
) -> str:
    """Durably make an ambiguous submit retryable after complete scheduler absence.

    ``submitting`` means the prior process may have died after Slurm accepted the
    request.  It is therefore never a directly admissible input to
    :func:`submit_attempt`.  This transition is the sole low-level retry bridge: it
    requires complete joined queue/accounting truth, the full visibility grace, and
    absence of the exact immutable intent token.
    """

    generation = _require_exact_attempt_owner(
        directory,
        ledger,
        replica_id=replica_id,
        attempt=attempt,
    )
    if attempt["state"] != "submitting":
        raise FleetTransactionError(
            "submission-absence proof requires an ambiguous submitting intent"
        )
    if not snapshot.squeue_ok or not snapshot.sacct_ok or snapshot.errors:
        raise FleetTransactionError(
            "submission-absence proof requires complete squeue+sacct truth"
        )
    captured_at = snapshot.captured_at
    if (
        not isinstance(captured_at, (int, float))
        or isinstance(captured_at, bool)
        or not isinstance(now, (int, float))
        or isinstance(now, bool)
        or not (0.0 <= float(now) - float(captured_at) <= 60.0)
    ):
        raise FleetTransactionError(
            "submission-absence proof requires a fresh scheduler capture"
        )
    basis = attempt.get("submit_started_at")
    if (
        not isinstance(basis, (int, float))
        or isinstance(basis, bool)
        or float(captured_at) - float(basis)
        < DEFAULT_VISIBILITY_GRACE_SECONDS
    ):
        raise FleetTransactionError(
            "submission-absence proof precedes the visibility grace"
        )
    token = str(attempt["intent_token"])
    job_id = attempt.get("job_id")
    matching_rows = [
        row.job_id
        for row in snapshot.rows
        if (
            token in str(row.comment)
            or token in str(row.command)
            or (job_id is not None and str(row.job_id) == str(job_id))
        )
    ]
    if matching_rows:
        raise FleetTransactionError(
            "submission-absence proof found the immutable intent in scheduler "
            f"truth: {sorted(matching_rows)}"
        )
    row_ids = [str(row.job_id) for row in snapshot.rows]
    if (
        len(row_ids) != len(set(row_ids))
        or any(row.source not in {"squeue", "sacct"} for row in snapshot.rows)
    ):
        raise FleetTransactionError(
            "submission-absence proof has invalid joined scheduler rows"
        )
    receipt_path, evidence_sha256 = _publish_submission_absence_receipt(
        directory,
        ledger,
        replica_id=replica_id,
        attempt=attempt,
        snapshot=snapshot,
        now=float(now),
    )
    attempt["state"] = "submission_failed"
    attempt["last_error"] = (
        "proven not accepted after complete squeue+sacct visibility grace; "
        f"absence_receipt_path={receipt_path};"
        f"absence_receipt_sha256={evidence_sha256}"
    )
    save_ledger(directory, ledger, now=now)
    _verify_persisted_ledger_preimage(
        directory,
        ledger,
        generation=generation,
    )
    return evidence_sha256


def submit_attempt(
    directory: Path,
    ledger: dict[str, Any],
    *,
    replica_id: str,
    attempt: dict[str, Any],
    now: float,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> str:
    """Cross ``sbatch`` only after the exact intent is durably ``submitting``."""

    if runner is None and os.environ.get("PYTEST_CURRENT_TEST"):
        raise FleetTransactionError(
            "pytest fleet submissions must provide an injected non-Slurm runner"
        )
    generation = _require_exact_attempt_owner(
        directory,
        ledger,
        replica_id=replica_id,
        attempt=attempt,
    )
    if attempt["state"] not in {"prepared", "submission_failed"}:
        raise FleetTransactionError(
            f"cannot submit fleet intent in state {attempt['state']!r}"
        )
    absence_reference = _absence_receipt_reference(attempt)
    deterministic_receipt = _submission_absence_receipt_path(
        directory,
        generation=generation,
        replica_id=replica_id,
        intent_token=str(attempt["intent_token"]),
    )
    if absence_reference is not None or os.path.lexists(
        deterministic_receipt
    ):
        _validate_submission_absence_receipt(
            directory,
            ledger,
            replica_id=replica_id,
            attempt=attempt,
            require_reference=True,
        )
    path = _lexical_absolute_path(
        attempt["sbatch_path"],
        description=f"fleet sbatch script for {replica_id}",
    )
    attempt["state"] = "submitting"
    attempt["submit_started_at"] = float(now)
    attempt["submission_attempts"] += 1
    # Preserve the sealed absence receipt reference as durable authorization for
    # this retry.  Ordinary explicit rejections have no such audit fact to retain.
    if absence_reference is None:
        attempt["last_error"] = None
    save_ledger(directory, ledger, now=now)
    try:
        _verify_persisted_ledger_preimage(
            directory,
            ledger,
            generation=generation,
        )
        stable_path, payload = _stable_regular_preimage(
            path,
            description=f"fleet sbatch script for {replica_id}",
            read_only=True,
        )
        if (
            stable_path != path
            or hashlib.sha256(payload).hexdigest()
            != attempt["sbatch_sha256"]
        ):
            raise FleetTransactionError(
                f"immutable fleet script hash drifted for {replica_id}"
            )
    except FleetTransactionError as exc:
        # No external invocation has occurred.  Persist a deterministic rejection
        # state instead of conflating local provenance failure with an ambiguous
        # post-sbatch transport outcome.
        attempt["state"] = "submission_failed"
        attempt["last_error"] = f"pre-sbatch provenance failure: {exc}"
        save_ledger(directory, ledger, now=now)
        raise FleetTransactionError(attempt["last_error"]) from exc
    invoke = runner or subprocess.run
    try:
        submission_text = payload.decode("utf-8")
    except UnicodeError as exc:
        attempt["state"] = "submission_failed"
        attempt["last_error"] = (
            f"pre-sbatch provenance failure: fleet script is not UTF-8: {exc}"
        )
        save_ledger(directory, ledger, now=now)
        raise FleetTransactionError(attempt["last_error"]) from exc
    try:
        proc = invoke(
            submission_argv(str(attempt["scheduler_comment"])),
            capture_output=True,
            text=True,
            check=False,
            timeout=60.0,
            input=submission_text,
        )
    except BaseException:
        # The process may have died after Slurm accepted the request.  Leave the durable
        # intent in ``submitting`` so the next owner searches squeue+sacct by token.
        raise
    if proc.returncode != 0:
        attempt["state"] = "submission_failed"
        attempt["last_error"] = (
            f"sbatch rc={proc.returncode}: {proc.stderr.strip()[:500]}"
        )
        save_ledger(directory, ledger, now=now)
        raise FleetTransactionError(attempt["last_error"])
    job_id = proc.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        attempt["state"] = "submitting"
        attempt["last_error"] = f"sbatch returned invalid job id {job_id!r}"
        save_ledger(directory, ledger, now=now)
        raise FleetTransactionError(attempt["last_error"])
    try:
        spool = invoke(
            ["scontrol", "write", "batch_script", job_id, "-"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15.0,
        )
    except BaseException:
        # Slurm accepted the exact stdin payload, but the reply/spooled-script
        # proof is incomplete.  Preserve the ambiguous fence for reconciliation.
        raise
    if (
        spool.returncode != 0
        or not isinstance(spool.stdout, str)
        or spool.stdout.encode("utf-8") != payload
    ):
        attempt["state"] = "submitting"
        attempt["last_error"] = (
            "accepted fleet allocation lacks exact Slurm-spooled script proof"
        )
        save_ledger(directory, ledger, now=now)
        raise FleetTransactionError(attempt["last_error"])
    try:
        post_path, post_payload = _stable_regular_preimage(
            path,
            description=f"fleet sbatch script after submission for {replica_id}",
            read_only=True,
        )
    except FleetTransactionError as exc:
        attempt["state"] = "submitting"
        attempt["last_error"] = (
            f"fleet script changed across external submission: {exc}"
        )
        save_ledger(directory, ledger, now=now)
        raise FleetTransactionError(attempt["last_error"]) from exc
    if post_path != path or post_payload != payload:
        attempt["state"] = "submitting"
        attempt["last_error"] = (
            "fleet script changed across external submission"
        )
        save_ledger(directory, ledger, now=now)
        raise FleetTransactionError(attempt["last_error"])
    attempt["state"] = "submitted"
    attempt["job_id"] = job_id
    attempt["submitted_at"] = float(now)
    save_ledger(directory, ledger, now=now)
    return job_id


def _submit_line_comment(command: str) -> str | None:
    """Recover one exact CLI ``--comment`` from Slurm's stored SubmitLine."""

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise FleetTransactionError(f"invalid sacct SubmitLine quoting: {exc}") from exc
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token.startswith("--comment="):
            values.append(token.split("=", 1)[1])
        elif token == "--comment":
            if index + 1 >= len(tokens):
                raise FleetTransactionError("sacct SubmitLine has a valueless --comment")
            values.append(tokens[index + 1])
    if len(values) > 1:
        raise FleetTransactionError("sacct SubmitLine has duplicate --comment options")
    return values[0] if values else None


def _parse_slurm_timestamp(value: str, *, field: str) -> float | None:
    normalized = value.strip()
    if normalized.lower() in {
        "",
        "(null)",
        "null",
        "none",
        "unknown",
        "n/a",
        "notset",
    }:
        return None
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FleetTransactionError(
            f"invalid Slurm {field} timestamp {value!r}"
        ) from exc
    return parsed.timestamp()


def _parse_slurm_duration(value: str, *, source: str) -> int | None:
    normalized = value.strip()
    if normalized.lower() in {
        "",
        "(null)",
        "null",
        "none",
        "unknown",
        "n/a",
        "partition_limit",
        "unlimited",
    }:
        return None
    if normalized.isdigit():
        # On this cluster sacct's ``TimelimitRaw`` unit is minutes (e.g. 1440
        # for a 24-hour job and 10080 for seven days).  squeue's ``%l`` is the
        # formatted duration, so accepting a bare number there would be ambiguous.
        if source != "sacct":
            raise FleetTransactionError(
                f"ambiguous formatted squeue time limit {value!r}"
            )
        minutes = int(normalized)
        return minutes * 60 if minutes > 0 else None
    match = re.fullmatch(
        r"(?:(?P<days>[0-9]+)-)?(?:(?P<hours>[0-9]{1,2}):)?"
        r"(?P<minutes>[0-9]{1,2}):(?P<seconds>[0-9]{2})",
        normalized,
    )
    if match is None:
        raise FleetTransactionError(f"invalid Slurm time limit {value!r}")
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if hours >= 24 or minutes >= 60 or seconds >= 60:
        raise FleetTransactionError(f"invalid Slurm time limit {value!r}")
    total = days * 86_400 + hours * 3_600 + minutes * 60 + seconds
    return total if total > 0 else None


def _parse_rows(text: str, *, source: str) -> list[SchedulerRow]:
    rows: list[SchedulerRow] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.rstrip("\n").split("|")
        # QOS was added after the original fleet transaction format.  Retain
        # parsing of sealed historical/fixture rows, with an empty QOS that the
        # production contract rejects, while every newly queried row must carry
        # the exact scheduler-owned QOS field.
        legacy_arities = {7, 10} if source == "sacct" else {7, 11}
        qos_arities = {8, 11} if source == "sacct" else {8, 12}
        allowed_arities = legacy_arities | qos_arities
        if source not in {"sacct", "squeue"} or len(fields) not in allowed_arities:
            raise FleetTransactionError(
                f"malformed {source} fleet row {line_number}: {raw!r}"
            )
        normalized_fields = [field.strip() for field in fields]
        if len(fields) in qos_arities:
            (
                job_id,
                name,
                state,
                partition,
                qos,
                node,
                command,
                comment,
                *timing,
            ) = normalized_fields
        else:
            (
                job_id,
                name,
                state,
                partition,
                node,
                command,
                comment,
                *timing,
            ) = normalized_fields
            qos = ""
        start_timestamp: float | None = None
        end_timestamp: float | None = None
        time_limit_seconds: int | None = None
        dependency: str | None = None if source == "sacct" else ""
        if timing:
            start_timestamp = _parse_slurm_timestamp(timing[0], field="start")
            end_timestamp = _parse_slurm_timestamp(timing[1], field="end")
            time_limit_seconds = _parse_slurm_duration(timing[2], source=source)
            if source == "squeue":
                dependency = (
                    ""
                    if timing[3].lower()
                    in {
                        "",
                        "(null)",
                        "null",
                        "none",
                        "n/a",
                        "singleton",
                    }
                    else timing[3]
                )
        # Slurm renders an unset JobComment as either an empty field or ``(null)``
        # depending on whether the row came from sacct or squeue.  Normalize both before
        # joining the two scheduler views; these representations are semantically equal.
        comment = "" if comment.lower() in {"", "(null)", "null", "none"} else comment
        candidate_hint = (
            name.startswith("asys-s5-serve-")
            or comment.startswith("asys-s5-fleet:")
            or source == "sacct"
            and "asys-s5-fleet:" in command
        )
        if not candidate_hint:
            continue
        if source == "sacct":
            derived_comment = _submit_line_comment(command)
            stored_comment = comment
            if stored_comment and derived_comment and stored_comment != derived_comment:
                raise FleetTransactionError(
                    f"sacct comment/SubmitLine conflict for fleet job {job_id}"
                )
            # This cluster has AccountingStoreFlags=(null), so JobComment is absent
            # from sacct even though the immutable SubmitLine retains the exact CLI
            # token.  Recover only that token; never infer from a job name.
            comment = stored_comment or derived_comment or ""
        # Scheduler truth is account-wide, but only canonical fleet allocations belong
        # to this transaction domain.  Ignore unrelated arrays, interactive shells, and
        # experiment jobs rather than letting their legitimate Slurm-specific IDs or
        # metadata block serving reconciliation.  Keep malformed fleet comments by
        # prefix so they fail closed in the higher-level provenance validator.
        if not (
            name.startswith("asys-s5-serve-")
            or comment.startswith("asys-s5-fleet:")
        ):
            continue
        if (
            not job_id.isdigit()
            or not name
            or not state
            or not partition
            or "." in job_id
            or "_" in job_id
        ):
            raise FleetTransactionError(
                f"invalid {source} fleet row {line_number}: {raw!r}"
            )
        rows.append(
            SchedulerRow(
                job_id,
                name,
                state,
                partition,
                node,
                command,
                comment,
                source,
                start_timestamp,
                end_timestamp,
                time_limit_seconds,
                dependency,
                qos,
            )
        )
    return rows


def query_scheduler(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    user: str | None = None,
    now: float | None = None,
) -> SchedulerSnapshot:
    """Join current squeue and seven-day sacct truth, failing unless both succeed."""

    invoke = runner or subprocess.run
    scheduler_user = user or os.environ.get("USER")
    if not scheduler_user:
        raise FleetTransactionError("USER is unset; fleet scheduler query is unscoped")
    timestamp = time.time() if now is None else float(now)
    commands = (
        (
            "sacct",
            [
                "sacct",
                "-X",
                "-u",
                scheduler_user,
                "-n",
                "-P",
                "-S",
                time.strftime("%Y-%m-%d", time.localtime(timestamp - 7 * 86_400)),
                (
                    "--format=JobIDRaw,JobName,State,Partition,QOS,NodeList,"
                    "SubmitLine,Comment,Start,End,TimelimitRaw"
                ),
            ],
        ),
        (
            "squeue",
            [
                "squeue",
                "-u",
                scheduler_user,
                "-h",
                "-r",
                "-o",
                "%i|%j|%T|%P|%q|%N|%o|%k|%S|%e|%l|%E",
            ],
        ),
    )
    by_id: dict[str, SchedulerRow] = {}
    errors: list[str] = []
    success: dict[str, bool] = {"squeue": False, "sacct": False}
    for source, argv in commands:
        try:
            proc = invoke(
                argv,
                capture_output=True,
                text=True,
                check=False,
                timeout=15.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{source} failed: {exc}")
            continue
        if proc.returncode != 0:
            errors.append(
                f"{source} failed rc={proc.returncode}: {proc.stderr.strip()[:500]}"
            )
            continue
        success[source] = True
        for row in _parse_rows(proc.stdout, source=source):
            prior = by_id.get(row.job_id)
            if prior is not None and (
                prior.job_name != row.job_name
                or prior.partition != row.partition
                or prior.qos != row.qos
                or prior.comment != row.comment
            ):
                raise FleetTransactionError(
                    f"squeue/sacct identity conflict for fleet job {row.job_id}"
                )
            if prior is not None:
                for field in ("start_timestamp", "end_timestamp"):
                    left = getattr(prior, field)
                    right = getattr(row, field)
                    if (
                        left is not None
                        and right is not None
                        and abs(float(left) - float(right)) > 1.0
                    ):
                        raise FleetTransactionError(
                            f"squeue/sacct {field} conflict for fleet job {row.job_id}"
                        )
                if (
                    prior.time_limit_seconds is not None
                    and row.time_limit_seconds is not None
                    and prior.time_limit_seconds != row.time_limit_seconds
                ):
                    raise FleetTransactionError(
                        f"squeue/sacct time limit conflict for fleet job {row.job_id}"
                    )
            # The later squeue pass owns current state/node/command and its exact
            # expected walltime.  Accounting remains the terminal-history source.
            by_id[row.job_id] = row
    if errors or not all(success.values()):
        raise FleetTransactionError(
            "fleet reconciliation requires complete squeue+sacct truth: "
            + "; ".join(errors or ["one scheduler source was unavailable"])
        )
    for row in by_id.values():
        if not terminal_state(row.state):
            if row.source != "squeue" or row.dependency is None:
                raise FleetTransactionError(
                    f"active fleet job {row.job_id} lacks complete squeue timing/"
                    "dependency truth"
                )
            if row.dependency:
                raise FleetTransactionError(
                    f"active fleet job {row.job_id} has unexpected dependency "
                    f"{row.dependency!r}"
                )
    return SchedulerSnapshot(
        rows=tuple(sorted(by_id.values(), key=lambda item: int(item.job_id))),
        captured_at=timestamp,
        squeue_ok=True,
        sacct_ok=True,
    )


def command_binds_sbatch(command: str, sbatch_path: str) -> bool:
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    expected = str(Path(sbatch_path).expanduser().resolve())
    return any(
        str(Path(part).expanduser().resolve()) == expected
        for part in parts
        if part.endswith(".sbatch") and Path(part).is_absolute()
    )


def terminal_state(state: str) -> bool:
    normalized = state.upper().split("+", 1)[0].split(" ", 1)[0]
    return normalized in {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }


def append_alert_once(
    directory: Path,
    ledger: dict[str, Any],
    *,
    replica_id: str,
    health: dict[str, Any],
    now: float,
) -> str:
    """Persist one immutable hung-allocation alert before any cancellation attempt."""

    if health.get("alert_id"):
        return str(health["alert_id"])
    material = (
        f"{replica_id}\0{health['job_id']}\0{health['endpoint']}\0"
        f"{health['first_failure_at']}"
    ).encode("utf-8")
    alert_id = "fleet-hung-" + hashlib.sha256(material).hexdigest()[:20]
    event = {
        "schema_version": 1,
        "alert_id": alert_id,
        "kind": "hung_serving_allocation",
        "replica_id": replica_id,
        "job_id": str(health["job_id"]),
        "endpoint": health["endpoint"],
        "first_failure_at": health["first_failure_at"],
        "last_failure_at": health["last_failure_at"],
        "consecutive_failures": health["consecutive_failures"],
        "recorded_at": float(now),
    }
    io.append_jsonl(directory / ALERTS_FILENAME, event)
    health["alert_id"] = alert_id
    save_ledger(directory, ledger, now=now)
    return alert_id


def read_health_summary(
    pool_root: str | Path,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Read the atomic fleet ledgers for the five-minute/email monitor.

    This function never creates files or takes the writer lock.  Every individual file
    is atomically replaced by the supervisor; CURRENT additionally binds the current
    generation's exact bytes.  A cross-file transition may yield a conservative read
    error for one monitor poll, which correctly becomes a fleet-integrity alert.
    """

    directory = state_directory(pool_root)
    observed_at = time.time() if now is None else float(now)
    if not directory.exists():
        return {
            "available": False,
            "current_generation": None,
            "active_hung_allocations": [],
            "active_handoffs": [],
            "handoff_violations": [],
            "handoff_overlap_gpus": 0,
            "historical_alert_count": 0,
            "alerts_path": str(directory / ALERTS_FILENAME),
        }
    if directory.is_symlink() or not directory.is_dir():
        raise FleetTransactionError("fleet transaction state is not a regular directory")
    # Recovery of a write-ahead save intent is a writer operation and is permitted
    # only while the supervisor owns ``fleet.lock``.  A monitor racing the two-file
    # ledger/CURRENT publication must remain read-only and fail conservatively for this
    # poll instead of completing or rolling back another process's transaction.
    save_intent_path = directory / SAVE_INTENT_FILENAME
    if save_intent_path.exists() or save_intent_path.is_symlink():
        raise FleetTransactionError(
            "fleet ledger publication is in progress or requires locked recovery"
        )
    current = _read_current_index_raw(directory)
    if current is None:
        raise FleetTransactionError("fleet transaction state lacks CURRENT.json")
    current_path = Path(current["ledger_path"])
    if (
        current_path.is_symlink()
        or not current_path.is_file()
        or hashlib.sha256(current_path.read_bytes()).hexdigest()
        != current["ledger_sha256"]
    ):
        raise FleetTransactionError("fleet CURRENT ledger bytes are not attested")
    ledger_dir = directory / LEDGERS_DIRECTORY
    if ledger_dir.is_symlink() or not ledger_dir.is_dir():
        raise FleetTransactionError("fleet generation-ledger directory is invalid")
    active: list[dict[str, Any]] = []
    attempt_rows: list[tuple[int, str, dict[str, Any]]] = []
    for path in sorted(ledger_dir.glob("g*.json")):
        match = re.fullmatch(r"g([0-9]{6})\.json", path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise FleetTransactionError(f"invalid fleet ledger entry {path}")
        try:
            ledger = _strict_json_loads(
                path.read_text(encoding="utf-8"), artifact=f"fleet ledger {path}"
            )
        except (OSError, UnicodeError) as exc:
            raise FleetTransactionError(f"cannot read fleet ledger {path}: {exc}") from exc
        generation = int(match.group(1))
        if (
            not isinstance(ledger, dict)
            or ledger.get("schema_version") != STATE_SCHEMA_VERSION
            or ledger.get("pool_root") != str(Path(pool_root).expanduser().resolve())
            or ledger.get("pool_id") != current["pool_id"]
            or ledger.get("fleet_sha256") != current["fleet_sha256"]
            or ledger.get("rollout_generation") != generation
            or generation > int(current["current_generation"])
            or not isinstance(ledger.get("replicas"), dict)
        ):
            raise FleetTransactionError(f"fleet ledger identity drifted: {path}")
        _validate_ledger_material(
            ledger,
            directory=directory,
            canonical_root=Path(pool_root).expanduser().resolve(),
            pool_id=str(current["pool_id"]),
            fleet_sha256=str(current["fleet_sha256"]),
            rollout_generation=generation,
            replica_ids=sorted(str(key) for key in ledger["replicas"]),
        )
        for replica_id, record in ledger["replicas"].items():
            if not isinstance(record, dict):
                raise FleetTransactionError(f"invalid fleet replica state in {path}")
            attempt_rows.extend(
                (generation, str(replica_id), attempt)
                for attempt in record["attempts"]
            )
            health = record.get("health")
            if health is None:
                continue
            _validate_health(health, replica_id=str(replica_id))
            if health["cancel_state"] is not None:
                active.append(
                    {
                        "replica_id": str(replica_id),
                        "ledger_generation": generation,
                        "job_id": str(health["job_id"]),
                        "endpoint": health["endpoint"],
                        "cancel_state": health["cancel_state"],
                        "cancel_attempts": health["cancel_attempts"],
                        "first_failure_at": health["first_failure_at"],
                        "last_failure_at": health["last_failure_at"],
                        "alert_id": health["alert_id"],
                    }
                )
    current_states = {
        "prepared",
        "submitting",
        "submitted",
        "committed",
        "missing",
    }
    by_replica: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    active_handoffs: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for generation, replica_id, attempt in attempt_rows:
        if attempt["launch_kind"] == "handoff" and attempt["state"] in (
            current_states | {"submission_failed"}
        ):
            active_handoffs.append(
                {
                    "replica_id": replica_id,
                    "ledger_generation": generation,
                    "job_id": attempt["job_id"],
                    "state": attempt["state"],
                    "lifecycle": attempt["lifecycle"],
                    "predecessor_job_id": attempt["predecessor_job_id"],
                    "predecessor_end_at": attempt["predecessor_end_at"],
                    "ready_probe_count": attempt["ready_probe_count"],
                    "last_ready_probe_at": attempt["last_ready_probe_at"],
                    "promoted_at": attempt["promoted_at"],
                    "retire_attempts": attempt["retire_attempts"],
                    "retire_error": attempt["retire_error"],
                    "allocated_gpus": attempt["allocated_gpus"],
                }
            )
        if attempt["state"] in current_states:
            by_replica.setdefault(replica_id, []).append((generation, attempt))

    overlap_gpus = 0
    allowed_pairs = {
        ("primary", "standby"),
        ("promoted", "retiring"),
        ("promoted", "standby"),
        ("retiring", "standby"),
    }
    for replica_id, current_attempts in sorted(by_replica.items()):
        lifecycles = tuple(
            sorted(str(attempt["lifecycle"]) for _generation, attempt in current_attempts)
        )
        if len(current_attempts) > 2 or (
            len(current_attempts) == 2 and lifecycles not in allowed_pairs
        ):
            violations.append(
                {
                    "replica_id": replica_id,
                    "kind": "invalid_lifecycle_overlap",
                    "lifecycles": list(lifecycles),
                }
            )
        if len(current_attempts) == 2:
            gpu_counts = {
                int(attempt["allocated_gpus"])
                for _generation, attempt in current_attempts
            }
            if len(gpu_counts) != 1:
                violations.append(
                    {
                        "replica_id": replica_id,
                        "kind": "handoff_gpu_contract_drift",
                        "allocated_gpus": sorted(gpu_counts),
                    }
                )
            overlap_gpus += max(gpu_counts)

        standby = next(
            (
                attempt
                for _generation, attempt in current_attempts
                if attempt["lifecycle"] == "standby"
            ),
            None,
        )
        if standby is not None:
            remaining = float(standby["predecessor_end_at"]) - observed_at
            if remaining <= 0:
                violations.append(
                    {
                        "replica_id": replica_id,
                        "kind": "handoff_overdue",
                        "job_id": standby["job_id"],
                        "seconds_past_predecessor_end": -remaining,
                    }
                )
            elif (
                remaining <= 3_600
                and int(standby["ready_probe_count"]) < 2
            ):
                violations.append(
                    {
                        "replica_id": replica_id,
                        "kind": "handoff_critical_lead",
                        "job_id": standby["job_id"],
                        "seconds_remaining": remaining,
                        "ready_probe_count": standby["ready_probe_count"],
                    }
                )
            heartbeat = (
                standby["last_seen_at"]
                or standby["submit_started_at"]
                or standby["created_at"]
            )
            if observed_at - float(heartbeat) > 900:
                violations.append(
                    {
                        "replica_id": replica_id,
                        "kind": "handoff_scheduler_stale",
                        "job_id": standby["job_id"],
                        "age_seconds": observed_at - float(heartbeat),
                    }
                )

        stable = next(
            (
                attempt
                for _generation, attempt in current_attempts
                if attempt["lifecycle"] in {"primary", "promoted"}
            ),
            None,
        )
        if (
            stable is not None
            and standby is None
            and stable["scheduler_end_at"] is not None
            and float(stable["scheduler_end_at"]) - observed_at <= 3_600
        ):
            violations.append(
                {
                    "replica_id": replica_id,
                    "kind": "missed_handoff_lead",
                    "job_id": stable["job_id"],
                    "seconds_remaining": (
                        float(stable["scheduler_end_at"]) - observed_at
                    ),
                }
            )

        for _generation, attempt in current_attempts:
            if (
                attempt["lifecycle"] == "retiring"
                and attempt["retire_error"]
                and int(attempt["retire_attempts"]) >= 5
            ):
                violations.append(
                    {
                        "replica_id": replica_id,
                        "kind": "handoff_retirement_exhausted",
                        "job_id": attempt["job_id"],
                        "retire_attempts": attempt["retire_attempts"],
                        "retire_error": attempt["retire_error"],
                    }
                )
            if (
                attempt["launch_kind"] == "handoff"
                and attempt["lifecycle"] == "promoted"
            ):
                from agents_scaling.serving import registry

                try:
                    pointer = registry.read_promoted_entry(
                        pool_root,
                        str(
                            parse_intent_comment(
                                attempt["scheduler_comment"]
                            )["profile"]
                        ),
                        replica_id,
                    )
                except (TypeError, ValueError, KeyError) as exc:
                    violations.append(
                        {
                            "replica_id": replica_id,
                            "kind": "invalid_promoted_pointer",
                            "job_id": attempt["job_id"],
                            "error": str(exc),
                        }
                    )
                else:
                    if (
                        pointer is None
                        or str(pointer.slurm_job_id or "")
                        != str(attempt["job_id"])
                    ):
                        violations.append(
                            {
                                "replica_id": replica_id,
                                "kind": "invalid_promoted_pointer",
                                "job_id": attempt["job_id"],
                                "observed_job_id": (
                                    None
                                    if pointer is None
                                    else pointer.slurm_job_id
                                ),
                            }
                        )
    if overlap_gpus > 4:
        violations.append(
            {
                "replica_id": None,
                "kind": "handoff_overlap_budget_exceeded",
                "observed_gpus": overlap_gpus,
                "maximum_gpus": 4,
            }
        )
    alerts_path = directory / ALERTS_FILENAME
    alert_ids: set[str] = set()
    if alerts_path.exists():
        if alerts_path.is_symlink() or not alerts_path.is_file():
            raise FleetTransactionError("fleet alert journal is not a regular file")
        for line_number, line in enumerate(
            alerts_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            event = _strict_json_loads(
                line, artifact=f"fleet alert journal line {line_number}"
            )
            alert_id = event.get("alert_id") if isinstance(event, dict) else None
            if not isinstance(alert_id, str) or not alert_id or alert_id in alert_ids:
                raise FleetTransactionError("fleet alert journal identity is invalid")
            alert_ids.add(alert_id)
    return {
        "available": True,
        "current_generation": int(current["current_generation"]),
        "active_hung_allocations": sorted(
            active, key=lambda item: (item["replica_id"], item["job_id"])
        ),
        "active_handoffs": sorted(
            active_handoffs,
            key=lambda item: (
                item["replica_id"],
                str(item["job_id"] or ""),
            ),
        ),
        "handoff_violations": sorted(
            violations,
            key=lambda item: (
                str(item.get("replica_id") or ""),
                str(item.get("kind") or ""),
                str(item.get("job_id") or ""),
            ),
        ),
        "handoff_overlap_gpus": overlap_gpus,
        "historical_alert_count": len(alert_ids),
        "alerts_path": str(alerts_path),
    }
