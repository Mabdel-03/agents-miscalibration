#!/usr/bin/env python3
"""Record fail-fast stage observations and the aggregate schema-5 recovery outcome.

Each production stage has an independent ``afterany`` observer.  It joins the exact
immutable submission receipt to both ``squeue`` and ``sacct`` and immediately records
and alerts on that stage's terminal failure.  A separate aggregate sentinel runs after
all stages and observers; only that aggregate grants same-generation repair authority.

Dry-run mode performs the complete verification and scheduler reconciliation but
writes nothing and never sends mail.  ``--apply`` is crash-safe and idempotent:
immutable scheduler evidence is published first, at most one currently eligible
bounded mail delivery is attempted without sleeping, and the completion marker is
published immediately afterward.  Failed delivery persists its next eligibility for
a later monitor or idempotent successor; it never delays or changes classification.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid


# The chain executes this checksummed flat bundle with ``python -I`` so the sentinel
# remains available even if source checkout failed.  Isolated mode omits the script
# directory from imports; explicitly add only this immutable bundle directory.
_BUNDLE_DIRECTORY = str(Path(__file__).resolve().parent)
if _BUNDLE_DIRECTORY not in sys.path:
    sys.path.insert(0, _BUNDLE_DIRECTORY)

try:
    from scripts.verify_schema5_recovery_evidence import (
        EvidenceVerificationError,
        R2_PROTOCOL,
        verify_recovery_evidence,
    )
except ModuleNotFoundError:  # ``python -I /absolute/path/to/this_script.py``
    from verify_schema5_recovery_evidence import (  # type: ignore[no-redef]
        EvidenceVerificationError,
        R2_PROTOCOL,
        verify_recovery_evidence,
    )


SCHEDULER_EVIDENCE_NAME = "SCHEDULER_EVIDENCE.json"
MAIL_STATE_NAME = "MAIL_DELIVERY.json"
COMPLETE_MARKER_NAME = "RECOVERY_SENTINEL_COMPLETE.json"
LOCK_NAME = ".schema5_recovery_sentinel.lock"
SCHEDULER_EVIDENCE_PROTOCOL = "schema5-v1.2-r2-recovery-scheduler-evidence"
MAIL_PROTOCOL = "schema5-v1.2-r2-recovery-sentinel-mail"
MARKER_PROTOCOL = "schema5-v1.2-r2-recovery-sentinel-outcome"
STAGE_SCHEDULER_EVIDENCE_NAME = "STAGE_SCHEDULER_EVIDENCE.json"
STAGE_COMPLETE_MARKER_NAME = "STAGE_SENTINEL_COMPLETE.json"
STAGE_SCHEDULER_EVIDENCE_PROTOCOL = (
    "schema5-v1.2-r2-recovery-stage-scheduler-evidence"
)
STAGE_MARKER_PROTOCOL = "schema5-v1.2-r2-recovery-stage-sentinel-outcome"
STAGE_SENTINEL_PREFIX = "stage_failure_sentinel_"
PRODUCTION_STAGE_NAMES = (
    "source_checkout",
    "maintenance_preflight",
    "snapshot_adopt_verify",
    "environment_capture",
    "release_materialize",
    "release_freeze",
    "legacy_consolidate",
    "legacy_consolidated_snapshot",
    "legacy_consolidated_verify",
    "legacy_retire",
    "schema5_initialize",
    "static_readiness",
    "context_readiness",
    "email_readiness",
    "supplementary_cache",
    "fleet_bootstrap",
    "fleet_readiness",
    "smoke_readiness",
    "throughput_qualification",
    "controller_drill",
    "production_resume",
)
STAGE_SENTINEL_NAMES = tuple(
    f"{STAGE_SENTINEL_PREFIX}{stage}" for stage in PRODUCTION_STAGE_NAMES
)
CAPACITY_TRANSIENT_PROTOCOL = (
    "schema5-v1.2-r2-fleet-capacity-transient-receipt"
)
CAPACITY_TRANSIENT_EVIDENCE_PROTOCOL = (
    "schema5-v1.2-r2-fleet-capacity-transient-evidence"
)
CAPACITY_TRANSIENT_MARKER_NAME = "CAPACITY_TRANSIENT_COMPLETE.json"
CAPACITY_TRANSIENT_EVIDENCE_NAME = "FLEET_CAPACITY_TRANSIENT_EVIDENCE.json"
CAPACITY_PREIMAGE_ROOT_NAME = "sealed-preimages"
CAPACITY_PREIMAGE_MANIFEST_NAME = "PREIMAGE_MANIFEST.json"
CAPACITY_PREIMAGE_INVENTORY_NAME = "PREIMAGE_INVENTORY.sha256"
CAPACITY_PREIMAGE_COMPLETE_NAME = "PREIMAGE_ARCHIVE_COMPLETE.json"
CAPACITY_PREIMAGE_PROTOCOL = "schema5-v1.2-r2-capacity-preimage-archive"
CAPACITY_TRANSIENT_REASONS = frozenset({"Priority", "Resources"})
CAPACITY_TRANSIENT_EXPECTED_REPLICAS = 22
CAPACITY_TRANSIENT_EXPECTED_GPUS = 24
CAPACITY_TRANSIENT_BOUNDARY_SECONDS = 36_000
CAPACITY_TRANSIENT_BOUNDARY_TOLERANCE_SECONDS = 300
CAPACITY_TRANSIENT_RELEASE_ID = "sweep-recovery-schema5-v1.2"
QUALIFICATION_CAPACITY_EXIT_CODE = "76:0"
QUALIFICATION_ROOT_NAME = "schema5_throughput_qualification_v1"
QUALIFICATION_CURRENT_NAME = "CURRENT_ATTEMPT.json"
QUALIFICATION_FAILURE_NAME = "QUALIFICATION_FAILURE.json"
QUALIFICATION_POINTER_ROOT_NAME = "attempt-pointers"
QUALIFICATION_ATTEMPT_ROOT_NAME = "attempts"
QUALIFICATION_RUN_ROOT_NAME = "throughput-qualification-attempts"
QUALIFICATION_POINTER_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-attempt-pointer-v1"
)
QUALIFICATION_CURRENT_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-current-attempt-v1"
)
QUALIFICATION_FAILURE_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-failure-v1"
)
QUALIFICATION_POINTER_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "chain_id",
        "attempt_ordinal",
        "attempt_id",
        "attempt_root",
        "run_root",
        "dispatcher_state",
        "readiness_generation",
        "predecessor",
        "additive_retry",
        "created_at",
        "created_timestamp",
        "pointer_id",
    }
)
QUALIFICATION_CURRENT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "attempt_id",
        "pointer",
        "pointer_sha256",
        "pointer_id",
        "current_id",
    }
)
QUALIFICATION_READINESS_FIELDS = frozenset(
    {
        "catalog_id",
        "marker_path",
        "marker_sha256",
        "inventory_sha256",
        "catalog_payload_sha256",
        "allowed_generation_tuple_count",
        "release_fleet_contract_sha256",
        "fleet_contract_sha256",
        "capacity_generation",
        "rollout_generation",
    }
)
QUALIFICATION_FAILURE_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "passed",
        "intent_id",
        "attempt",
        "readiness_generation",
        "reason",
        "additive_scaling_requirement",
        "scheduler_capacity_mutated",
        "rerun_requirement",
        "failure_id",
    }
)
QUALIFICATION_SCALING_FIELDS = frozenset(
    {
        "serving_profile",
        "server_pool_root",
        "backlog_fanout_work",
        "live_replicas",
        "backlog_work_per_replica",
        "additional_replicas",
        "tensor_parallel_size",
        "additional_gpus",
        "requirement",
        "capacity_mutated",
    }
)
QUALIFICATION_SERVING_PROFILES = frozenset(
    {
        "0.6B",
        "1.7B",
        "4B",
        "8B",
        "14B",
        "32B",
        "0.6B-long",
        "1.7B-long",
        "4B-long",
        "8B-long",
        "14B-long",
        "32B-long",
    }
)
TRANSIENT_ROOT_STATES = frozenset(
    {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED", "REVOKED"}
)
ACTIVE_STATES = frozenset(
    {
        "CONFIGURING",
        "COMPLETING",
        "PENDING",
        "REQUEUED",
        "RESIZING",
        "RUNNING",
        "SUSPENDED",
    }
)
SUPERSEDING_STATES = frozenset(
    {
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "OOM",
        "OUT_OF_MEMORY",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)
RETRY_DELAYS_SECONDS = (5.0, 15.0, 30.0, 60.0, 120.0, 300.0)
IN_FLIGHT_STALE_SECONDS = 300.0
MAIL_COMMAND_TIMEOUT_SECONDS = 30.0
SYNCHRONOUS_MAIL_ATTEMPT_LIMIT = 1


class SentinelError(RuntimeError):
    """The sentinel cannot publish a truthful, unambiguous outcome."""


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
MailRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]


def _utc(timestamp: float | None = None) -> str:
    value = time.time() if timestamp is None else float(timestamp)
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _atomic_json(path: Path, payload: object, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SentinelError(f"{description} is unavailable: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SentinelError(f"{description} is not one regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SentinelError(f"cannot parse {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SentinelError(f"{description} is not a JSON object")
    return value


def _require_read_only(path: Path, *, description: str) -> None:
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise SentinelError(f"{description} must be read-only: {path}")


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().absolute()


def _prepare_output_root(path: Path) -> Path:
    path = _absolute(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o750)
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise SentinelError(
            f"sentinel output root must be one canonical non-symlink directory: {path}"
        )
    return path


@contextmanager
def _sentinel_lock(root: Path) -> Iterator[None]:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(root / LOCK_NAME, flags, 0o640)
    except OSError as exc:
        raise SentinelError(f"cannot acquire sentinel lock: {exc}") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _default_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        text=True,
        capture_output=True,
        check=False,
        timeout=120.0,
    )


def _default_mail_runner(
    argv: Sequence[str], body: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        input=body,
        text=True,
        capture_output=True,
        check=False,
        timeout=MAIL_COMMAND_TIMEOUT_SECONDS,
    )


def _normalize_state(value: str) -> str:
    return value.strip().split()[0].rstrip("+").upper() if value.strip() else ""


def _submit_line_comment(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise SentinelError(f"invalid sacct SubmitLine quoting: {exc}") from exc
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token.startswith("--comment="):
            values.append(token.split("=", 1)[1])
        elif token == "--comment":
            if index + 1 >= len(tokens):
                raise SentinelError("sacct SubmitLine has a valueless --comment")
            values.append(tokens[index + 1])
    if len(values) > 1:
        raise SentinelError("sacct SubmitLine has duplicate --comment options")
    return values[0] if values else None


def _accounting_comment(stored: str, submit_line: str, *, job_id: str) -> str:
    normalized = (
        ""
        if stored.strip().lower() in {"", "(null)", "null", "none"}
        else stored.strip()
    )
    derived = _submit_line_comment(submit_line.strip())
    if normalized and derived and normalized != derived:
        raise SentinelError(
            f"sacct comment/SubmitLine conflict for recovery job {job_id}"
        )
    return normalized or derived or ""


def _run_scheduler(
    runner: Runner, argv: list[str], *, description: str
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(argv)
    except Exception as exc:
        raise SentinelError(
            f"{description} raised {type(exc).__name__}: {exc}"
        ) from exc
    if result.returncode != 0:
        raise SentinelError(
            f"{description} failed ({result.returncode}): "
            f"{result.stderr.strip()[:500]}"
        )
    return result


def _parse_squeue(
    stdout: str,
    *,
    requested_ids: set[str],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for raw in stdout.splitlines():
        if not raw.strip():
            continue
        fields = raw.rstrip("\n").split("|")
        if len(fields) != 5:
            raise SentinelError(f"malformed squeue row: {raw[:300]!r}")
        job_id, state, reason, comment, job_name = (
            field.strip() for field in fields
        )
        if job_id not in requested_ids:
            raise SentinelError(f"squeue returned an unrequested job ID: {job_id!r}")
        if job_id in result:
            raise SentinelError(f"receipt job appears more than once in squeue: {job_id}")
        normalized = _normalize_state(state)
        if normalized not in ACTIVE_STATES:
            raise SentinelError(
                f"squeue reported non-active state {normalized!r} for {job_id}"
            )
        if not comment or not job_name:
            raise SentinelError(f"squeue omitted identity for receipt job {job_id}")
        result[job_id] = {
            "state": normalized,
            "reason": reason,
            "comment": comment,
            "job_name": job_name,
        }
    return result


def _parse_sacct(
    stdout: str,
    *,
    requested_ids: set[str],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for raw in stdout.splitlines():
        if not raw.strip():
            continue
        fields = raw.rstrip("\n").split("|", 8)
        if len(fields) != 9:
            raise SentinelError(f"malformed sacct row: {raw[:300]!r}")
        (
            job_id,
            state,
            exit_code,
            reason,
            start,
            elapsed,
            stored_comment,
            job_name,
            submit_line,
        ) = (field.strip() for field in fields)
        if "." in job_id:
            continue
        if job_id not in requested_ids:
            raise SentinelError(f"sacct returned an unrequested job ID: {job_id!r}")
        if job_id in result:
            raise SentinelError(f"receipt job has duplicate sacct rows: {job_id}")
        normalized = _normalize_state(state)
        if not normalized:
            raise SentinelError(f"sacct omitted state for receipt job {job_id}")
        comment = _accounting_comment(
            stored_comment,
            submit_line,
            job_id=job_id,
        )
        if not comment or not job_name:
            raise SentinelError(f"sacct omitted identity for receipt job {job_id}")
        result[job_id] = {
            "state": normalized,
            "exit_code": exit_code,
            "reason": reason,
            "start": start,
            "elapsed": elapsed,
            "comment": comment,
            "job_name": job_name,
        }
    return result


def _scheduler_query_contract(
    receipt: Mapping[str, Any],
) -> tuple[set[str], list[str], list[str]]:
    receipt_jobs = receipt.get("jobs")
    if not isinstance(receipt_jobs, list):
        raise SentinelError("verified receipt has no job list")
    receipt_by_id = {
        str(row["job_id"]): row for row in receipt_jobs if isinstance(row, dict)
    }
    if len(receipt_by_id) != len(receipt_jobs):
        raise SentinelError("receipt job identities are not unique")
    requested_ids = set(receipt_by_id)
    if not requested_ids or any(not job_id.isdigit() for job_id in requested_ids):
        raise SentinelError("receipt job IDs are not unique positive decimal IDs")
    joined_ids = ",".join(sorted(requested_ids, key=int))
    return (
        requested_ids,
        [
            "squeue",
            "-h",
            "-j",
            joined_ids,
            "-o",
            "%i|%T|%r|%k|%j",
        ],
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            joined_ids,
            "--format=JobIDRaw,State,ExitCode,Reason%128,Start,Elapsed,"
            "Comment%256,JobName%64,SubmitLine",
        ],
    )


def _join_scheduler_rows(
    *,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    live: Mapping[str, Mapping[str, str]],
    accounting: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Derive canonical job rows from parsed scheduler truth."""

    manifest_jobs = manifest.get("jobs")
    receipt_jobs = receipt.get("jobs")
    if not isinstance(manifest_jobs, list) or not isinstance(receipt_jobs, list):
        raise SentinelError("verified manifest or receipt has no job list")
    if len(manifest_jobs) != len(receipt_jobs):
        raise SentinelError("verified manifest and receipt cardinalities differ")
    manifest_by_name = {
        str(row["name"]): row for row in manifest_jobs if isinstance(row, dict)
    }
    if len(manifest_by_name) != len(manifest_jobs):
        raise SentinelError("manifest job identities are not unique")

    rows: list[dict[str, Any]] = []
    for receipt_row in receipt_jobs:
        if not isinstance(receipt_row, Mapping):
            raise SentinelError("receipt contains a malformed job row")
        job_id = str(receipt_row["job_id"])
        name = str(receipt_row["name"])
        manifest_row = manifest_by_name.get(name)
        if manifest_row is None:
            raise SentinelError(f"receipt job is absent from manifest: {name}")
        expected_comment = str(receipt_row["comment"])
        expected_name = str(manifest_row["job_name"])
        live_row = live.get(job_id)
        historical = accounting.get(job_id)
        if live_row is None and historical is None:
            raise SentinelError(
                f"receipt job is absent from both squeue and sacct: {job_id}"
            )
        for source, observed in (("squeue", live_row), ("sacct", historical)):
            if observed is not None and (
                observed["comment"] != expected_comment
                or observed["job_name"] != expected_name
            ):
                raise SentinelError(
                    f"{source} identity drifted for receipt job {job_id}"
                )

        # A terminal accounting row wins a legitimate active-to-terminal race.
        if historical is not None and historical["state"] not in ACTIVE_STATES:
            selected_state = historical["state"]
            selected_source = "sacct"
        elif live_row is not None:
            selected_state = live_row["state"]
            selected_source = "squeue"
        else:
            assert historical is not None
            selected_state = historical["state"]
            selected_source = "sacct"
        rows.append(
            {
                "name": name,
                "job_id": job_id,
                "dependencies": list(manifest_row["dependencies"]),
                "dependency_type": str(
                    manifest_row.get("dependency_type", "afterok")
                ),
                "comment": expected_comment,
                "job_name": expected_name,
                "state": selected_state,
                "active": selected_state in ACTIVE_STATES,
                "exit_code": (
                    None if historical is None else historical["exit_code"]
                ),
                "reason": (
                    live_row["reason"]
                    if historical is None and live_row is not None
                    else historical["reason"] if historical is not None else ""
                ),
                "start": None if historical is None else historical["start"],
                "elapsed": None if historical is None else historical["elapsed"],
                "selected_source": selected_source,
                "squeue_state": None if live_row is None else live_row["state"],
                "sacct_state": (
                    None if historical is None else historical["state"]
                ),
            }
        )
    return rows


def reconcile_receipt_jobs(
    *,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    runner: Runner | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join exact receipt allocations through successful squeue and sacct queries."""

    execute = _default_runner if runner is None else runner
    requested_ids, squeue_argv, sacct_argv = _scheduler_query_contract(receipt)
    squeue_proc = _run_scheduler(
        execute, squeue_argv, description="complete squeue receipt query"
    )
    sacct_proc = _run_scheduler(
        execute, sacct_argv, description="complete sacct receipt query"
    )
    live = _parse_squeue(squeue_proc.stdout, requested_ids=requested_ids)
    accounting = _parse_sacct(sacct_proc.stdout, requested_ids=requested_ids)

    rows = _join_scheduler_rows(
        manifest=manifest,
        receipt=receipt,
        live=live,
        accounting=accounting,
    )
    query_evidence = {
        "squeue": {
            "argv": squeue_argv,
            "stdout": squeue_proc.stdout,
            "stdout_sha256": _sha256_bytes(squeue_proc.stdout.encode("utf-8")),
            "row_count": len(live),
        },
        "sacct": {
            "argv": sacct_argv,
            "stdout": sacct_proc.stdout,
            "stdout_sha256": _sha256_bytes(sacct_proc.stdout.encode("utf-8")),
            "row_count": len(accounting),
        },
    }
    return rows, query_evidence


def _dependency_reason(value: object) -> bool:
    compact = "".join(
        character
        for character in str(value or "").lower()
        if character.isalnum()
    )
    return compact.startswith("dependency")


def _never_started(value: object) -> bool:
    return str(value or "").strip().lower() in {
        "",
        "(null)",
        "n/a",
        "na",
        "none",
        "unknown",
    }


def _zero_elapsed(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "(null)", "n/a", "na", "none", "unknown"}:
        return True
    if normalized.isdigit():
        return int(normalized) == 0
    fields = normalized.split("-")
    clock = fields[-1].split(":")
    if not all(part.isdigit() for part in clock):
        return False
    day = int(fields[0]) if len(fields) == 2 and fields[0].isdigit() else 0
    return day == 0 and all(int(part) == 0 for part in clock)


def _fleet_readiness_rows(
    *,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest_jobs = manifest.get("jobs")
    receipt_jobs = receipt.get("jobs")
    if not isinstance(manifest_jobs, list) or not isinstance(receipt_jobs, list):
        raise SentinelError("capacity receipt chain has no exact job lists")
    manifest_rows = [
        row
        for row in manifest_jobs
        if isinstance(row, Mapping) and row.get("name") == "fleet_readiness"
    ]
    receipt_rows = [
        row
        for row in receipt_jobs
        if isinstance(row, Mapping) and row.get("name") == "fleet_readiness"
    ]
    if len(manifest_rows) != 1 or len(receipt_rows) != 1:
        raise SentinelError(
            "capacity receipt chain has ambiguous fleet_readiness identity"
        )
    return manifest_rows[0], receipt_rows[0]


def _capacity_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None
    )


def _capacity_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _capacity_digit_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item.isdigit() for item in value)
        and len(value) == len(set(value))
    )


def _capacity_value_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _capacity_authority(verified: Mapping[str, Any]) -> dict[str, Any]:
    """Load the frozen control, model, and fleet authority named by the chain."""

    manifest = verified.get("manifest")
    if not isinstance(manifest, Mapping):
        raise SentinelError("capacity-transient chain manifest is absent")
    required_manifest_fields = {
        "release_id",
        "release_root",
        "state_root",
        "server_pool_root",
        "immutable_pins",
    }
    if any(
        not isinstance(manifest.get(field), str) or not manifest[field]
        for field in required_manifest_fields
    ):
        raise SentinelError(
            "capacity-transient chain omits frozen control authority paths"
        )
    if manifest["release_id"] != CAPACITY_TRANSIENT_RELEASE_ID:
        raise SentinelError("capacity-transient chain release identity drifted")
    pins_path = _absolute(str(manifest["immutable_pins"]))
    pins = _read_json(pins_path, description="chain immutable pins")
    _require_read_only(pins_path, description="chain immutable pins")
    state_root = _absolute(str(manifest["state_root"]))
    frozen_pins_path = state_root / "immutable_pins.json"
    frozen_pins = _read_json(
        frozen_pins_path, description="control frozen immutable pins"
    )
    _require_read_only(
        frozen_pins_path, description="control frozen immutable pins"
    )
    control_path = state_root / "control.json"
    control = _read_json(control_path, description="schema-5 control")
    immutable_sha256 = _capacity_value_sha256(pins)
    if (
        frozen_pins != pins
        or control.get("immutable") != pins
        or control.get("immutable_sha256") != immutable_sha256
        or pins.get("release_id") != CAPACITY_TRANSIENT_RELEASE_ID
        or pins.get("release_bundle_root") != manifest["release_root"]
        or pins.get("server_pool_root") != manifest["server_pool_root"]
    ):
        raise SentinelError("capacity-transient frozen control authority drifted")
    release_fragment_fields = {
        "release_id",
        "release_worktree",
        "git_commit",
        "source_tree_sha256",
        "model_contract_path",
        "model_contract_sha256",
        "fleet_contract_path",
        "fleet_contract_sha256",
        "harness_environment_prefix",
        "harness_environment_manifest_path",
        "harness_environment_sha256",
        "serving_environment_prefix",
        "serving_environment_manifest_path",
        "serving_environment_sha256",
    }
    if any(field not in pins for field in release_fragment_fields) or any(
        field not in pins for field in ("release_bundle_id", "release_bundle_root")
    ):
        raise SentinelError("capacity-transient release pin fragment is incomplete")
    release_root = _absolute(str(manifest["release_root"]))
    identity_path = release_root / "release_identity.schema5-v1.json"
    marker_path = release_root / "RELEASE_COMPLETE.json"
    identity = _read_json(identity_path, description="sealed release identity")
    marker = _read_json(marker_path, description="sealed release completion marker")
    _require_read_only(identity_path, description="sealed release identity")
    _require_read_only(marker_path, description="sealed release completion marker")
    marker_identity = dict(marker)
    release_bundle_id = marker_identity.pop("release_bundle_id", None)
    artifacts = marker.get("artifacts")
    identity_record = (
        artifacts.get("release_identity.schema5-v1.json")
        if isinstance(artifacts, Mapping)
        else None
    )
    expected_fragment = {field: pins[field] for field in release_fragment_fields}
    if (
        marker.get("schema_version") != 2
        or marker.get("release_id") != CAPACITY_TRANSIENT_RELEASE_ID
        or marker.get("complete") is not True
        or marker.get("publication_protocol") != "fsync_verify_marker_last"
        or marker.get("git_commit") != pins.get("git_commit")
        or marker.get("source_tree_sha256") != pins.get("source_tree_sha256")
        or release_bundle_id != pins.get("release_bundle_id")
        or release_bundle_id != _capacity_value_sha256(marker_identity)
        or not isinstance(identity_record, Mapping)
        or identity_record.get("sha256") != _sha256(identity_path)
        or identity_record.get("size") != identity_path.stat().st_size
        or identity.get("schema_version") != 2
        or identity.get("release_id") != CAPACITY_TRANSIENT_RELEASE_ID
        or identity.get("release_worktree") != pins.get("release_worktree")
        or identity.get("worktree_sealed_read_only") is not True
        or identity.get("control_pin_fragment") != expected_fragment
        or manifest.get("release_git_commit") != pins.get("git_commit")
    ):
        raise SentinelError("capacity-transient sealed release authority drifted")
    required_pin_fields = {
        "model_contract_path",
        "model_contract_sha256",
        "fleet_contract_path",
        "fleet_contract_sha256",
        "serving_environment_sha256",
        "server_pool_root",
    }
    if any(field not in pins for field in required_pin_fields):
        raise SentinelError("capacity-transient immutable pins are incomplete")
    if (
        not _capacity_hex(pins.get("model_contract_sha256"), 64)
        or not _capacity_hex(pins.get("fleet_contract_sha256"), 64)
        or not _capacity_hex(pins.get("serving_environment_sha256"), 64)
    ):
        raise SentinelError("capacity-transient immutable pin hashes are invalid")

    model_path = _absolute(str(pins["model_contract_path"]))
    fleet_path = _absolute(str(pins["fleet_contract_path"]))
    for description, artifact, expected_hash in (
        ("model contract", model_path, pins["model_contract_sha256"]),
        ("fleet contract", fleet_path, pins["fleet_contract_sha256"]),
    ):
        payload = _read_json(artifact, description=description)
        _require_read_only(artifact, description=description)
        if _sha256(artifact) != expected_hash:
            raise SentinelError(f"capacity-transient {description} bytes drifted")
        if description == "model contract":
            model_contract = payload
        else:
            fleet_contract = payload

    models = model_contract.get("models")
    profiles = fleet_contract.get("profiles")
    if (
        model_contract.get("schema_version") != 1
        or not isinstance(models, Mapping)
        or fleet_contract.get("fleet_id") != "schema5-v1"
        or fleet_contract.get("release_id") != CAPACITY_TRANSIENT_RELEASE_ID
        or fleet_contract.get("logical_replica_count")
        != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or fleet_contract.get("allocated_gpu_count")
        != CAPACITY_TRANSIENT_EXPECTED_GPUS
        or fleet_contract.get("model_contract_sha256")
        != pins["model_contract_sha256"]
        or not isinstance(profiles, list)
    ):
        raise SentinelError("capacity-transient frozen fleet/model contract drifted")
    expected: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise SentinelError("capacity-transient fleet profile is malformed")
        model_size = profile.get("model_size")
        model = models.get(model_size) if isinstance(model_size, str) else None
        replicas = profile.get("replicas")
        if (
            not isinstance(model, Mapping)
            or not isinstance(replicas, list)
            or profile.get("hf_id") != model.get("hf_id")
            or profile.get("model_revision") != model.get("model_revision")
            or profile.get("tokenizer_id") != model.get("tokenizer_id")
            or profile.get("tokenizer_revision")
            != model.get("tokenizer_revision")
            or not isinstance(profile.get("serving_profile"), str)
            or not isinstance(profile.get("served_model_name"), str)
            or not isinstance(profile.get("effective_context_limit"), int)
            or isinstance(profile.get("effective_context_limit"), bool)
            or profile["effective_context_limit"] < 1
            or not isinstance(profile.get("tensor_parallel_size"), int)
            or isinstance(profile.get("tensor_parallel_size"), bool)
            or profile["tensor_parallel_size"] < 1
            or not isinstance(profile.get("gpus_per_replica"), int)
            or isinstance(profile.get("gpus_per_replica"), bool)
            or profile["gpus_per_replica"] < 1
        ):
            raise SentinelError("capacity-transient fleet profile provenance drifted")
        for replica in replicas:
            replica_id = (
                replica.get("replica_id")
                if isinstance(replica, Mapping)
                else None
            )
            if (
                not isinstance(replica, Mapping)
                or not isinstance(replica_id, str)
                or not replica_id
                or replica_id in expected
                or replica.get("pool_id") != "schema5-v1"
                or not isinstance(replica.get("replica_index"), int)
                or isinstance(replica.get("replica_index"), bool)
                or replica["replica_index"] < 0
                or not isinstance(replica.get("scheduler_job_name"), str)
                or not replica["scheduler_job_name"]
                or not isinstance(replica.get("partition"), str)
                or not replica["partition"]
            ):
                raise SentinelError(
                    "capacity-transient frozen fleet replica is malformed"
                )
            expected[replica_id] = {
                "serving_profile": profile["serving_profile"],
                "allocated_gpus": profile["gpus_per_replica"],
                "replica_index": replica["replica_index"],
                "scheduler_job_name": replica["scheduler_job_name"],
                "partition": replica["partition"],
                "model_size": model_size,
                "hf_id": profile["hf_id"],
                "model_revision": profile["model_revision"],
                "tokenizer_id": profile["tokenizer_id"],
                "tokenizer_revision": profile["tokenizer_revision"],
                "expected_model": profile["served_model_name"],
                "effective_context_limit": profile["effective_context_limit"],
                "tensor_parallel_size": profile["tensor_parallel_size"],
            }
    if (
        len(expected) != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or sum(row["allocated_gpus"] for row in expected.values())
        != CAPACITY_TRANSIENT_EXPECTED_GPUS
    ):
        raise SentinelError("capacity-transient frozen fleet cardinality drifted")
    return {
        "immutable_sha256": immutable_sha256,
        "pins": pins,
        "model_path": str(model_path),
        "fleet_path": str(fleet_path),
        "expected_replicas": expected,
    }


def _validate_capacity_preimage_archive(
    binding: object,
    *,
    receipt_parent: Path,
    chain_id: str,
    chain_generation: int,
    readiness_job_id: str,
    fleet: Mapping[str, Any],
) -> dict[str, Any]:
    required_binding = {
        "schema_version",
        "protocol",
        "root",
        "completion",
        "completion_sha256",
        "archive_id",
        "manifest",
        "manifest_sha256",
        "manifest_id",
        "inventory",
        "inventory_sha256",
        "file_count",
        "total_bytes",
    }
    if not isinstance(binding, Mapping) or set(binding) != required_binding:
        raise SentinelError("capacity preimage archive binding schema is invalid")
    root = Path(str(binding.get("root", "")))
    archive_key = _sha256_bytes(
        _canonical_json(
            {
                "chain_id": chain_id,
                "chain_generation": chain_generation,
                "fleet_readiness_job_id": readiness_job_id,
                "rollout_generation": fleet.get("rollout_generation"),
                "current_pointer_sha256": fleet.get("current_pointer_sha256"),
                "generation_ledger_sha256": fleet.get(
                    "generation_ledger_sha256"
                ),
                "scheduler_captured_timestamp": fleet.get(
                    "scheduler_captured_timestamp"
                ),
                "captured_timestamp": fleet.get("captured_timestamp"),
            }
        )
    )
    expected_root = (
        receipt_parent / f"{CAPACITY_PREIMAGE_ROOT_NAME}-{archive_key[:24]}"
    )
    if (
        binding.get("schema_version") != 1
        or binding.get("protocol") != CAPACITY_PREIMAGE_PROTOCOL
        or not root.is_absolute()
        or root != expected_root
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve() != root
        or stat.S_IMODE(root.stat().st_mode) & 0o222
    ):
        raise SentinelError("capacity preimage archive root is invalid")
    completion_path = Path(str(binding["completion"]))
    manifest_path = Path(str(binding["manifest"]))
    inventory_path = Path(str(binding["inventory"]))
    for description, path, expected in (
        (
            "capacity preimage completion",
            completion_path,
            root / CAPACITY_PREIMAGE_COMPLETE_NAME,
        ),
        (
            "capacity preimage manifest",
            manifest_path,
            root / CAPACITY_PREIMAGE_MANIFEST_NAME,
        ),
        (
            "capacity preimage inventory",
            inventory_path,
            root / CAPACITY_PREIMAGE_INVENTORY_NAME,
        ),
    ):
        if path != expected or path.resolve() != path:
            raise SentinelError(f"{description} path escaped its archive")
        _require_read_only(path, description=description)
    if (
        _sha256(completion_path) != binding.get("completion_sha256")
        or _sha256(manifest_path) != binding.get("manifest_sha256")
        or _sha256(inventory_path) != binding.get("inventory_sha256")
    ):
        raise SentinelError("capacity preimage archive control hash drifted")
    completion = _read_json(
        completion_path, description="capacity preimage completion"
    )
    manifest = _read_json(manifest_path, description="capacity preimage manifest")
    completion_identity = dict(completion)
    archive_id = completion_identity.pop("archive_id", None)
    manifest_identity = dict(manifest)
    manifest_id = manifest_identity.pop("manifest_id", None)
    completion_fields = {
        "schema_version",
        "protocol",
        "passed",
        "root",
        "manifest",
        "manifest_sha256",
        "manifest_id",
        "inventory",
        "inventory_sha256",
        "file_count",
        "total_bytes",
        "archive_id",
    }
    manifest_fields = {
        "schema_version",
        "protocol",
        "chain_id",
        "chain_generation",
        "fleet_readiness_job_id",
        "rollout_generation",
        "fleet_contract_sha256",
        "files",
        "inventory",
        "inventory_sha256",
        "file_count",
        "total_bytes",
        "manifest_id",
    }
    if (
        set(completion) != completion_fields
        or completion.get("schema_version") != 1
        or completion.get("protocol") != CAPACITY_PREIMAGE_PROTOCOL
        or completion.get("passed") is not True
        or completion.get("root") != str(root)
        or completion.get("manifest") != str(manifest_path)
        or completion.get("manifest_sha256") != _sha256(manifest_path)
        or completion.get("manifest_id") != manifest_id
        or completion.get("inventory") != str(inventory_path)
        or completion.get("inventory_sha256") != _sha256(inventory_path)
        or not isinstance(archive_id, str)
        or archive_id
        != _sha256_bytes(_canonical_json(completion_identity))
        or archive_id != binding.get("archive_id")
        or set(manifest) != manifest_fields
        or manifest.get("schema_version") != 1
        or manifest.get("protocol") != CAPACITY_PREIMAGE_PROTOCOL
        or manifest.get("chain_id") != chain_id
        or manifest.get("chain_generation") != chain_generation
        or manifest.get("fleet_readiness_job_id") != readiness_job_id
        or manifest.get("rollout_generation")
        != fleet.get("rollout_generation")
        or manifest.get("fleet_contract_sha256")
        != fleet.get("fleet_contract_sha256")
        or manifest.get("inventory") != str(inventory_path)
        or manifest.get("inventory_sha256") != _sha256(inventory_path)
        or not isinstance(manifest_id, str)
        or manifest_id != _sha256_bytes(_canonical_json(manifest_identity))
        or manifest_id != binding.get("manifest_id")
    ):
        raise SentinelError("capacity preimage archive identity is invalid")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise SentinelError("capacity preimage archive has no file inventory")
    record_fields = {
        "kind",
        "replica_id",
        "job_id",
        "logical_path",
        "archive_path",
        "source_path",
        "source_argv",
        "sha256",
        "bytes",
    }
    expected_rows = [*fleet.get("pending", []), *fleet.get("running", [])]
    row_by_replica = {
        str(row["replica_id"]): row
        for row in expected_rows
        if isinstance(row, Mapping)
    }
    expected_counts = {
        "current_pointer": 1,
        "generation_ledger": 1,
        "local_script": len(expected_rows),
        "spooled_script": len(expected_rows),
        "registry": len(fleet.get("running", [])),
    }
    counts = {kind: 0 for kind in expected_counts}
    records: dict[tuple[str, str | None], dict[str, Any]] = {}
    logical_paths: set[str] = set()
    inodes: set[tuple[int, int]] = set()
    inventory_lines: list[tuple[str, str]] = []
    total_bytes = 0
    for value in files:
        if not isinstance(value, Mapping) or set(value) != record_fields:
            raise SentinelError("capacity preimage record schema is invalid")
        record = dict(value)
        kind = record.get("kind")
        replica_id = record.get("replica_id")
        job_id = record.get("job_id")
        logical = record.get("logical_path")
        archived = Path(str(record.get("archive_path", "")))
        digest = record.get("sha256")
        size = record.get("bytes")
        if (
            kind not in expected_counts
            or not isinstance(logical, str)
            or not logical
            or logical.startswith("/")
            or ".." in Path(logical).parts
            or logical in logical_paths
            or archived != root / logical
            or archived.resolve() != archived
            or not _capacity_hex(digest, 64)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise SentinelError("capacity preimage record identity is invalid")
        _require_read_only(
            archived, description=f"capacity preimage {logical}"
        )
        metadata = archived.stat()
        inode = (metadata.st_dev, metadata.st_ino)
        if (
            _sha256(archived) != digest
            or metadata.st_size != size
            or inode in inodes
        ):
            raise SentinelError(
                f"capacity preimage bytes/inode drifted: {logical}"
            )
        inodes.add(inode)
        source_path = record.get("source_path")
        source_argv = record.get("source_argv")
        if kind == "current_pointer":
            expected_logical = "fleet-state/CURRENT.json"
            expected_source = fleet.get("current_pointer_path")
            expected_replica = expected_job = None
            expected_argv = None
            expected_digest = fleet.get("current_pointer_sha256")
        elif kind == "generation_ledger":
            expected_logical = "fleet-state/generation-ledger.json"
            expected_source = fleet.get("generation_ledger_path")
            expected_replica = expected_job = None
            expected_argv = None
            expected_digest = fleet.get("generation_ledger_sha256")
        else:
            if not isinstance(replica_id, str) or replica_id not in row_by_replica:
                raise SentinelError(
                    "capacity preimage job record has an unknown replica"
                )
            row = row_by_replica[replica_id]
            expected_replica = replica_id
            expected_job = row["job_id"]
            if kind == "local_script":
                expected_logical = f"jobs/{replica_id}/local.sbatch"
                expected_source = row["local_script_path"]
                expected_argv = None
                expected_digest = row["local_script_sha256"]
            elif kind == "spooled_script":
                expected_logical = f"jobs/{replica_id}/spooled.sbatch"
                expected_source = None
                expected_argv = [
                    "scontrol",
                    "write",
                    "batch_script",
                    row["job_id"],
                    "-",
                ]
                expected_digest = row["spooled_script_sha256"]
            else:
                if row.get("state") != "RUNNING":
                    raise SentinelError(
                        "capacity preimage archived a pending registry"
                    )
                expected_logical = f"jobs/{replica_id}/registry.json"
                expected_source = row["registry_path"]
                expected_argv = None
                expected_digest = row["registry_sha256"]
        if (
            logical != expected_logical
            or replica_id != expected_replica
            or job_id != expected_job
            or source_path != expected_source
            or source_argv != expected_argv
            or digest != expected_digest
        ):
            raise SentinelError(
                f"capacity preimage source binding drifted: {logical}"
            )
        if isinstance(source_path, str):
            source = Path(source_path)
            if not source.is_absolute():
                raise SentinelError(
                    f"capacity preimage source path is relative: {logical}"
                )
            try:
                source_metadata = source.lstat()
            except OSError:
                source_metadata = None
            if (
                source_metadata is not None
                and stat.S_ISREG(source_metadata.st_mode)
                and (
                    source_metadata.st_dev,
                    source_metadata.st_ino,
                )
                == inode
            ):
                raise SentinelError(
                    f"capacity preimage shares its live source inode: {logical}"
                )
        key = (str(kind), None if replica_id is None else str(replica_id))
        if key in records:
            raise SentinelError("capacity preimage record identity is duplicated")
        records[key] = record
        counts[str(kind)] += 1
        logical_paths.add(logical)
        total_bytes += size
        inventory_lines.append((logical, f"{digest}  {logical}\n"))
    inventory_payload = "".join(
        line for _logical, line in sorted(inventory_lines)
    )
    if (
        counts != expected_counts
        or len(files) != sum(expected_counts.values())
        or manifest.get("file_count") != len(files)
        or manifest.get("total_bytes") != total_bytes
        or binding.get("file_count") != len(files)
        or binding.get("total_bytes") != total_bytes
        or completion.get("file_count") != len(files)
        or completion.get("total_bytes") != total_bytes
        or inventory_path.read_text(encoding="utf-8") != inventory_payload
    ):
        raise SentinelError("capacity preimage archive inventory is invalid")
    expected_files = {
        completion_path,
        manifest_path,
        inventory_path,
        *(Path(str(record["archive_path"])) for record in records.values()),
    }
    observed_files: set[Path] = set()
    for member in root.rglob("*"):
        try:
            metadata = member.lstat()
        except OSError as exc:
            raise SentinelError(
                f"cannot enumerate capacity preimage archive: {member}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise SentinelError(
                f"capacity preimage archive contains a symlink: {member}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) & 0o222:
                raise SentinelError(
                    f"capacity preimage directory is writable: {member}"
                )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise SentinelError(
                f"capacity preimage archive contains a special file: {member}"
            )
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            raise SentinelError(
                f"capacity preimage archive contains a writable file: {member}"
            )
        observed_files.add(member)
    if observed_files != expected_files:
        raise SentinelError(
            "capacity preimage archive contains unlisted/missing members"
        )
    return {
        "binding": dict(binding),
        "records": records,
    }


def _validate_capacity_spooled_script(
    row: Mapping[str, Any],
    *,
    job_id: str,
    archived_local: Path,
    archived_spooled: Path,
) -> None:
    proof = row.get("spooled_script_proof")
    local_value = row.get("local_script_path")
    if not isinstance(local_value, str) or not Path(local_value).is_absolute():
        raise SentinelError(
            f"capacity-transient local script path is invalid for job {job_id}"
        )
    local_metadata = archived_local.stat()
    spooled_metadata = archived_spooled.stat()
    local_sha256 = _sha256(archived_local)
    spooled_sha256 = _sha256(archived_spooled)
    required = {
        "argv",
        "local_path",
        "local_sha256",
        "observed_sha256",
        "observed_bytes",
        "exact_match",
    }
    if (
        not isinstance(proof, Mapping)
        or set(proof) != required
        or proof.get("argv")
        != ["scontrol", "write", "batch_script", job_id, "-"]
        or proof.get("local_path") != row.get("local_script_path")
        or proof.get("local_sha256") != row.get("local_script_sha256")
        or proof.get("local_sha256") != local_sha256
        or proof.get("observed_sha256") != row.get("spooled_script_sha256")
        or proof.get("observed_sha256") != spooled_sha256
        or row.get("spooled_script_sha256")
        != row.get("local_script_sha256")
        or not _capacity_hex(row.get("local_script_sha256"), 64)
        or not isinstance(proof.get("observed_bytes"), int)
        or isinstance(proof.get("observed_bytes"), bool)
        or proof["observed_bytes"] < 1
        or proof["observed_bytes"] != spooled_metadata.st_size
        or local_metadata.st_size != spooled_metadata.st_size
        or archived_local.read_bytes() != archived_spooled.read_bytes()
        or proof.get("exact_match") is not True
    ):
        raise SentinelError(
            f"capacity-transient spooled-script proof is invalid for job {job_id}"
        )


def _validate_capacity_provenance(
    row: Mapping[str, Any],
    *,
    fleet: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    provenance = row.get("spooled_provenance")
    expected = authority["expected_replicas"].get(row.get("replica_id"))
    pins = authority["pins"]
    required = {
        "run_root",
        "server_pool_id",
        "replica_id",
        "replica_index",
        "release_id",
        "environment_hash",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "model_contract_sha256",
        "fleet_contract_sha256",
    }
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != required
        or provenance.get("run_root") != fleet.get("server_pool_root")
        or provenance.get("server_pool_id") != fleet.get("fleet_id")
        or provenance.get("replica_id") != row.get("replica_id")
        or not isinstance(expected, Mapping)
        or provenance.get("replica_index") != expected.get("replica_index")
        or provenance.get("release_id") != CAPACITY_TRANSIENT_RELEASE_ID
        or provenance.get("environment_hash")
        != pins.get("serving_environment_sha256")
        or provenance.get("model_revision") != expected.get("model_revision")
        or provenance.get("tokenizer_id") != expected.get("tokenizer_id")
        or provenance.get("tokenizer_revision")
        != expected.get("tokenizer_revision")
        or provenance.get("model_contract_sha256")
        != fleet.get("model_contract_sha256")
        or provenance.get("fleet_contract_sha256")
        != fleet.get("fleet_contract_sha256")
    ):
        raise SentinelError(
            "capacity-transient spooled provenance is invalid for "
            f"{row.get('replica_id')}"
        )


def _validate_capacity_scontrol(
    row: Mapping[str, Any],
    *,
    state: str,
    reason: str | None,
    expected_job_name: str,
) -> None:
    job_id = str(row.get("job_id", ""))
    proof = row.get("scontrol")
    required = {
        "argv",
        "job_id",
        "job_state",
        "reason",
        "comment",
        "job_name",
        "node",
        "command",
        "effective_requeue",
        "raw_output_sha256",
    }
    expected_node = None if state == "PENDING" else row.get("node")
    if (
        not isinstance(proof, Mapping)
        or set(proof) != required
        or proof.get("argv") != ["scontrol", "show", "job", "-o", job_id]
        or proof.get("job_id") != job_id
        or proof.get("job_state") != state
        or proof.get("reason") != reason
        or proof.get("comment") != row.get("comment")
        or proof.get("job_name") != expected_job_name
        or proof.get("node") != expected_node
        or proof.get("command") != row.get("local_script_path")
        or proof.get("effective_requeue") != 0
        or not _capacity_hex(proof.get("raw_output_sha256"), 64)
    ):
        raise SentinelError(
            f"capacity-transient exact scontrol proof is invalid for job {job_id}"
        )


def _validate_capacity_common_row(
    row: Mapping[str, Any],
    *,
    fleet: Mapping[str, Any],
    authority: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> None:
    replica_id = row.get("replica_id")
    serving_profile = row.get("serving_profile")
    job_id = row.get("job_id")
    generation = fleet.get("rollout_generation")
    intent = row.get("intent_token")
    local_path = row.get("local_script_path")
    expected = authority["expected_replicas"].get(replica_id)
    expected_comment = (
        f"asys-s5-fleet:pool={fleet.get('fleet_id')};"
        f"profile={serving_profile};replica={replica_id};"
        f"generation={generation};intent={intent};"
        f"fleet={fleet.get('fleet_contract_sha256')}"
    )
    if (
        not isinstance(replica_id, str)
        or not replica_id
        or not isinstance(expected, Mapping)
        or not isinstance(serving_profile, str)
        or not serving_profile
        or serving_profile != expected.get("serving_profile")
        or row.get("allocated_gpus") != expected.get("allocated_gpus")
        or row.get("partition") != expected.get("partition")
        or not isinstance(job_id, str)
        or not job_id.isdigit()
        or not isinstance(row.get("partition"), str)
        or not row["partition"]
        or not _capacity_hex(intent, 32)
        or row.get("ledger_generation") != generation
        or not isinstance(local_path, str)
        or not Path(local_path).is_absolute()
        or row.get("comment") != expected_comment
        or len(expected_comment.encode("utf-8")) > 255
    ):
        raise SentinelError(
            f"capacity-transient ledger identity is invalid for {replica_id}"
        )
    archive_records = archive["records"]
    local_record = archive_records.get(("local_script", replica_id))
    spooled_record = archive_records.get(("spooled_script", replica_id))
    if not isinstance(local_record, Mapping) or not isinstance(
        spooled_record, Mapping
    ):
        raise SentinelError(
            f"capacity preimage omitted scripts for {replica_id}"
        )
    _validate_capacity_spooled_script(
        row,
        job_id=job_id,
        archived_local=Path(str(local_record["archive_path"])),
        archived_spooled=Path(str(spooled_record["archive_path"])),
    )
    _validate_capacity_provenance(row, fleet=fleet, authority=authority)


def _validate_capacity_pending_row(
    row: Mapping[str, Any],
    *,
    fleet: Mapping[str, Any],
    authority: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> None:
    required = {
        "replica_id",
        "serving_profile",
        "allocated_gpus",
        "job_id",
        "state",
        "reason",
        "partition",
        "comment",
        "intent_token",
        "ledger_generation",
        "local_script_path",
        "local_script_sha256",
        "spooled_script_sha256",
        "spooled_script_proof",
        "spooled_provenance",
        "scontrol",
    }
    if set(row) != required or row.get("state") != "PENDING":
        raise SentinelError("capacity-transient pending row schema/state is invalid")
    reason = row.get("reason")
    if reason not in CAPACITY_TRANSIENT_REASONS:
        raise SentinelError("capacity-transient pending reason is not admissible")
    _validate_capacity_common_row(
        row, fleet=fleet, authority=authority, archive=archive
    )
    expected = authority["expected_replicas"][row["replica_id"]]
    _validate_capacity_scontrol(
        row,
        state="PENDING",
        reason=str(reason),
        expected_job_name=str(expected["scheduler_job_name"]),
    )


def _validate_capacity_http(
    row: Mapping[str, Any],
    *,
    captured_timestamp: float,
    expected_model: str,
) -> None:
    probe = row.get("http")
    required = {
        "health_status",
        "models_status",
        "model_ids",
        "expected_model",
        "probe_started_timestamp",
        "probe_completed_timestamp",
        "healthy",
    }
    if not isinstance(probe, Mapping) or set(probe) != required:
        raise SentinelError(
            f"capacity-transient HTTP proof schema is invalid for "
            f"{row.get('replica_id')}"
        )
    started = probe.get("probe_started_timestamp")
    completed = probe.get("probe_completed_timestamp")
    model_ids = probe.get("model_ids")
    if (
        probe.get("health_status") != 200
        or probe.get("models_status") != 200
        or not isinstance(model_ids, list)
        or any(not isinstance(model, str) or not model for model in model_ids)
        or model_ids != sorted(set(model_ids))
        or probe.get("expected_model") != expected_model
        or expected_model not in model_ids
        or not _capacity_finite_number(started)
        or not _capacity_finite_number(completed)
        or float(started) > float(completed)
        or float(completed) > captured_timestamp
        or captured_timestamp - float(completed) > 600
        or probe.get("healthy") is not True
    ):
        raise SentinelError(
            f"capacity-transient dual HTTP proof is invalid for "
            f"{row.get('replica_id')}"
        )


def _validate_capacity_registry(
    row: Mapping[str, Any],
    *,
    fleet: Mapping[str, Any],
    authority: Mapping[str, Any],
    archive: Mapping[str, Any],
    captured_timestamp: float,
) -> None:
    replica_id = str(row["replica_id"])
    expected = authority["expected_replicas"][replica_id]
    registry_path = _absolute(str(row["registry_path"]))
    pool_root = _absolute(str(fleet["server_pool_root"]))
    try:
        registry_path.relative_to(pool_root / "servers" / row["serving_profile"])
    except ValueError as exc:
        raise SentinelError(
            f"capacity-transient registry escaped its profile root: {replica_id}"
        ) from exc
    registry_record = archive["records"].get(("registry", replica_id))
    if not isinstance(registry_record, Mapping):
        raise SentinelError(
            f"capacity preimage omitted registry for {replica_id}"
        )
    archived_registry = Path(str(registry_record["archive_path"]))
    registry = _read_json(
        archived_registry,
        description=f"capacity-transient registry for {replica_id}",
    )
    required = {
        "model_size",
        "hf_id",
        "host",
        "port",
        "slurm_job_id",
        "started_at",
        "serving_profile",
        "served_model_name",
        "max_model_len",
        "tp_size",
        "release_id",
        "environment_hash",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "model_contract_sha256",
        "fleet_contract_sha256",
        "server_pool_id",
        "replica_id",
        "replica_index",
    }
    started_at = registry.get("started_at")
    if (
        set(registry) != required
        or _sha256(archived_registry) != row.get("registry_sha256")
        or registry.get("model_size") != expected["model_size"]
        or registry.get("hf_id") != expected["hf_id"]
        or registry.get("host") != row.get("node")
        or not isinstance(registry.get("port"), int)
        or isinstance(registry.get("port"), bool)
        or not 1 <= registry["port"] <= 65_535
        or registry.get("slurm_job_id") != row.get("job_id")
        or not _capacity_finite_number(started_at)
        or not 0 < float(started_at) <= captured_timestamp
        or registry.get("serving_profile") != expected["serving_profile"]
        or registry.get("served_model_name") != expected["expected_model"]
        or registry.get("max_model_len") != expected["effective_context_limit"]
        or registry.get("tp_size") != expected["tensor_parallel_size"]
        or registry.get("release_id") != CAPACITY_TRANSIENT_RELEASE_ID
        or registry.get("environment_hash")
        != authority["pins"]["serving_environment_sha256"]
        or registry.get("model_revision") != expected["model_revision"]
        or registry.get("tokenizer_id") != expected["tokenizer_id"]
        or registry.get("tokenizer_revision") != expected["tokenizer_revision"]
        or registry.get("model_contract_sha256")
        != fleet["model_contract_sha256"]
        or registry.get("fleet_contract_sha256")
        != fleet["fleet_contract_sha256"]
        or registry.get("server_pool_id") != fleet["fleet_id"]
        or registry.get("replica_id") != replica_id
        or registry.get("replica_index") != expected["replica_index"]
    ):
        raise SentinelError(
            f"capacity-transient registry provenance is invalid for {replica_id}"
        )


def _validate_capacity_running_row(
    row: Mapping[str, Any],
    *,
    fleet: Mapping[str, Any],
    authority: Mapping[str, Any],
    archive: Mapping[str, Any],
    captured_timestamp: float,
) -> None:
    required = {
        "replica_id",
        "serving_profile",
        "allocated_gpus",
        "job_id",
        "state",
        "partition",
        "node",
        "comment",
        "intent_token",
        "ledger_generation",
        "local_script_path",
        "local_script_sha256",
        "spooled_script_sha256",
        "spooled_script_proof",
        "spooled_provenance",
        "scontrol",
        "registry_path",
        "registry_sha256",
        "http",
    }
    node = row.get("node")
    registry_path = row.get("registry_path")
    if (
        set(row) != required
        or row.get("state") != "RUNNING"
        or not isinstance(node, str)
        or not node
        or node in {"(null)", "N/A", "None", "None assigned"}
        or not isinstance(registry_path, str)
        or not Path(registry_path).is_absolute()
        or not _capacity_hex(row.get("registry_sha256"), 64)
    ):
        raise SentinelError("capacity-transient running row schema/state is invalid")
    _validate_capacity_common_row(
        row, fleet=fleet, authority=authority, archive=archive
    )
    expected = authority["expected_replicas"][row["replica_id"]]
    _validate_capacity_scontrol(
        row,
        state="RUNNING",
        reason=None,
        expected_job_name=str(expected["scheduler_job_name"]),
    )
    _validate_capacity_http(
        row,
        captured_timestamp=captured_timestamp,
        expected_model=str(expected["expected_model"]),
    )
    _validate_capacity_registry(
        row,
        fleet=fleet,
        authority=authority,
        archive=archive,
        captured_timestamp=captured_timestamp,
    )


def _validate_capacity_ledger(
    fleet: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    archive: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    generation = int(fleet["rollout_generation"])
    records = archive["records"]
    current_record = records.get(("current_pointer", None))
    ledger_record = records.get(("generation_ledger", None))
    if not isinstance(current_record, Mapping) or not isinstance(
        ledger_record, Mapping
    ):
        raise SentinelError("capacity preimage omitted CURRENT/generation ledger")
    current_path = Path(str(current_record["archive_path"]))
    ledger_path = Path(str(ledger_record["archive_path"]))
    current = _read_json(
        current_path, description="capacity-transient sealed CURRENT"
    )
    ledger = _read_json(
        ledger_path, description="capacity-transient sealed generation ledger"
    )
    current_fields = {
        "schema_version",
        "pool_root",
        "pool_id",
        "fleet_sha256",
        "current_generation",
        "ledger_path",
        "ledger_sha256",
        "updated_at",
    }
    if (
        set(current) != current_fields
        or current.get("schema_version") != 1
        or current.get("pool_root") != fleet["server_pool_root"]
        or current.get("pool_id") != fleet["fleet_id"]
        or current.get("fleet_sha256") != fleet["fleet_contract_sha256"]
        or current.get("current_generation") != generation
        or current.get("ledger_path") != fleet["generation_ledger_path"]
        or current.get("ledger_sha256") != fleet["generation_ledger_sha256"]
        or current_record.get("source_path") != fleet["current_pointer_path"]
        or current_record.get("sha256") != fleet["current_pointer_sha256"]
        or ledger_record.get("source_path") != fleet["generation_ledger_path"]
        or ledger_record.get("sha256") != fleet["generation_ledger_sha256"]
    ):
        raise SentinelError("capacity-transient sealed CURRENT binding is invalid")
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
    replicas = ledger.get("replicas")
    expected_ids = set(authority["expected_replicas"])
    if (
        set(ledger) != required_root
        or ledger.get("schema_version") != 1
        or ledger.get("pool_root") != fleet["server_pool_root"]
        or ledger.get("pool_id") != fleet["fleet_id"]
        or ledger.get("fleet_sha256") != fleet["fleet_contract_sha256"]
        or ledger.get("rollout_generation") != generation
        or ledger.get("visibility_grace_seconds") != 300.0
        or not _capacity_finite_number(ledger.get("created_at"))
        or not _capacity_finite_number(ledger.get("updated_at"))
        or not isinstance(replicas, Mapping)
        or set(replicas) != expected_ids
    ):
        raise SentinelError("capacity-transient generation ledger identity is invalid")
    attempt_fields = {
        "intent_token",
        "rollout_generation",
        "state",
        "created_at",
        "submit_started_at",
        "submission_attempts",
        "sbatch_path",
        "sbatch_sha256",
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
    health_fields = {
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
    by_replica = {str(row["replica_id"]): row for row in rows}
    observed_tokens: set[str] = set()
    observed_jobs: set[str] = set()
    for replica_id in sorted(expected_ids):
        record = replicas[replica_id]
        row = by_replica.get(replica_id)
        attempts = record.get("attempts") if isinstance(record, Mapping) else None
        health = record.get("health") if isinstance(record, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or set(record) != {"attempts", "health"}
            or not isinstance(row, Mapping)
            or not isinstance(attempts, list)
            or not attempts
        ):
            raise SentinelError(
                f"capacity-transient ledger replica record is invalid: {replica_id}"
            )
        active: list[Mapping[str, Any]] = []
        attempts_by_job: dict[str, Mapping[str, Any]] = {}
        for attempt in attempts:
            if not isinstance(attempt, Mapping) or set(attempt) != attempt_fields:
                raise SentinelError(
                    f"capacity-transient ledger attempt schema drifted: {replica_id}"
                )
            token = attempt.get("intent_token")
            job_id = attempt.get("job_id")
            if (
                not _capacity_hex(token, 32)
                or token in observed_tokens
                or attempt.get("rollout_generation") != generation
                or not isinstance(attempt.get("allocated_gpus"), int)
                or isinstance(attempt.get("allocated_gpus"), bool)
                or attempt["allocated_gpus"] < 1
                or not _capacity_hex(attempt.get("sbatch_sha256"), 64)
                or not isinstance(attempt.get("sbatch_path"), str)
                or not Path(attempt["sbatch_path"]).is_absolute()
                or not isinstance(attempt.get("scheduler_comment"), str)
                or attempt.get("state")
                not in {
                    "prepared",
                    "submitting",
                    "submitted",
                    "committed",
                    "missing",
                    "terminal",
                    "submission_failed",
                }
                or attempt.get("launch_kind") not in {"primary", "handoff"}
                or attempt.get("lifecycle")
                not in {"primary", "standby", "promoted", "retiring"}
                or not isinstance(attempt.get("submission_attempts"), int)
                or isinstance(attempt.get("submission_attempts"), bool)
                or attempt["submission_attempts"] < 0
                or not isinstance(attempt.get("ready_probe_count"), int)
                or isinstance(attempt.get("ready_probe_count"), bool)
                or attempt["ready_probe_count"] < 0
                or not isinstance(attempt.get("retire_attempts"), int)
                or isinstance(attempt.get("retire_attempts"), bool)
                or attempt["retire_attempts"] < 0
                or (
                    job_id is not None
                    and (
                        not isinstance(job_id, str)
                        or not job_id.isdigit()
                        or job_id in observed_jobs
                    )
                )
            ):
                raise SentinelError(
                    f"capacity-transient ledger attempt identity drifted: {replica_id}"
                )
            observed_tokens.add(str(token))
            if isinstance(job_id, str):
                observed_jobs.add(job_id)
                attempts_by_job[job_id] = attempt
            for timestamp_field in (
                "created_at",
                "submit_started_at",
                "submitted_at",
                "committed_at",
                "terminal_at",
                "last_seen_at",
                "missing_since",
                "predecessor_end_at",
                "scheduler_start_at",
                "scheduler_end_at",
                "scheduler_time_limit_seconds",
                "last_ready_probe_at",
                "promoted_at",
                "retire_requested_at",
                "last_retire_attempt_at",
            ):
                value = attempt.get(timestamp_field)
                if value is not None and not _capacity_finite_number(value):
                    raise SentinelError(
                        "capacity-transient ledger attempt timestamp drifted: "
                        f"{replica_id}"
                    )
            if attempt.get("state") in {
                "prepared",
                "submitting",
                "submitted",
                "committed",
                "missing",
            }:
                active.append(attempt)
        if len(active) != 1:
            raise SentinelError(
                f"capacity-transient ledger has ambiguous active attempt: {replica_id}"
            )
        attempt = active[0]
        terminal = [
            candidate for candidate in attempts if candidate is not attempt
        ]
        for predecessor in terminal:
            predecessor_job_id = str(predecessor.get("job_id") or "")
            successors = [
                candidate
                for candidate in attempts
                if candidate.get("launch_kind") == "handoff"
                and str(candidate.get("predecessor_job_id") or "")
                == predecessor_job_id
            ]
            if (
                predecessor.get("state") != "terminal"
                or predecessor.get("lifecycle") != "retiring"
                or not _capacity_finite_number(predecessor.get("terminal_at"))
                or predecessor.get("last_error") is not None
                or not _capacity_finite_number(
                    predecessor.get("retire_requested_at")
                )
                or predecessor.get("retire_attempts", 0) < 1
                or not _capacity_finite_number(
                    predecessor.get("last_retire_attempt_at")
                )
                or predecessor.get("retire_error") is not None
                or len(successors) != 1
            ):
                raise SentinelError(
                    "capacity-transient ledger terminal handoff history is "
                    f"unsealed: {replica_id}"
                )
            successor = successors[0]
            if (
                successor.get("state") not in {"committed", "terminal"}
                or successor.get("lifecycle") not in {"promoted", "retiring"}
                or not _capacity_finite_number(successor.get("promoted_at"))
                or successor.get("ready_probe_count", 0) < 2
                or successor.get("last_error") is not None
                or successor.get("retire_error") is not None
            ):
                raise SentinelError(
                    "capacity-transient ledger handoff successor is invalid: "
                    f"{replica_id}"
                )
            visited: set[str] = set()
            current = predecessor_job_id
            active_job_id = str(attempt.get("job_id"))
            while current != active_job_id:
                if current in visited:
                    raise SentinelError(
                        f"capacity-transient ledger handoff cycle: {replica_id}"
                    )
                visited.add(current)
                next_jobs = [
                    str(candidate.get("job_id"))
                    for candidate in attempts
                    if candidate.get("launch_kind") == "handoff"
                    and str(candidate.get("predecessor_job_id") or "")
                    == current
                ]
                if len(next_jobs) != 1 or next_jobs[0] not in attempts_by_job:
                    raise SentinelError(
                        "capacity-transient ledger handoff chain is disconnected: "
                        f"{replica_id}"
                    )
                current = next_jobs[0]
        lifecycle = attempt.get("lifecycle")
        launch_kind = attempt.get("launch_kind")
        if (
            attempt.get("state") != "committed"
            or attempt.get("intent_token") != row["intent_token"]
            or attempt.get("job_id") != row["job_id"]
            or attempt.get("sbatch_path") != row["local_script_path"]
            or attempt.get("sbatch_sha256") != row["local_script_sha256"]
            or attempt.get("scheduler_comment") != row["comment"]
            or attempt.get("allocated_gpus") != row["allocated_gpus"]
            or not _capacity_finite_number(attempt.get("committed_at"))
            or attempt.get("last_error") is not None
            or lifecycle not in {"primary", "promoted"}
            or (
                lifecycle == "primary"
                and (
                    launch_kind != "primary"
                    or attempt.get("predecessor_job_id") is not None
                    or attempt.get("promoted_at") is not None
                )
            )
            or (
                lifecycle == "promoted"
                and (
                    launch_kind != "handoff"
                    or not str(attempt.get("predecessor_job_id") or "").isdigit()
                    or not _capacity_finite_number(attempt.get("promoted_at"))
                    or not isinstance(attempt.get("ready_probe_count"), int)
                    or isinstance(attempt.get("ready_probe_count"), bool)
                    or attempt["ready_probe_count"] < 2
                )
            )
        ):
            raise SentinelError(
                f"capacity-transient row is not bound to its committed ledger "
                f"attempt: {replica_id}"
            )
        if row["state"] == "PENDING":
            if health is not None:
                raise SentinelError(
                    f"capacity-transient pending ledger health is stale: {replica_id}"
                )
        else:
            registry_record = records.get(("registry", replica_id))
            registry = (
                _read_json(
                    Path(str(registry_record["archive_path"])),
                    description=(
                        "capacity-transient sealed registry health binding "
                        f"for {replica_id}"
                    ),
                )
                if isinstance(registry_record, Mapping)
                else None
            )
            expected_endpoint = (
                f"{registry.get('host')}:{registry.get('port')}"
                if isinstance(registry, Mapping)
                else None
            )
            counters = (
                "consecutive_failures",
                "health_failures",
                "models_failures",
                "cancel_attempts",
            )
            timestamps = (
                "first_failure_at",
                "last_failure_at",
                "last_probe_at",
                "cancel_requested_at",
                "cancel_completed_at",
                "last_cancel_attempt_at",
                "next_cancel_eligible_at",
            )
            if (
                not isinstance(health, Mapping)
                or set(health) != health_fields
                or health.get("job_id") != row["job_id"]
                or health.get("observer_generation") != generation
                or health.get("consecutive_failures") != 0
                or any(
                    not isinstance(health.get(field), int)
                    or isinstance(health.get(field), bool)
                    or health[field] < 0
                    for field in counters
                )
                or any(
                    value is not None and not _capacity_finite_number(value)
                    for value in (health.get(field) for field in timestamps)
                )
                or health.get("cancel_state") is not None
                or health.get("cancel_error") is not None
                or not isinstance(health.get("endpoint"), str)
                or health.get("endpoint") != expected_endpoint
            ):
                raise SentinelError(
                    "capacity-transient running ledger health is invalid: "
                    f"{replica_id}"
                )
    return None


def _validate_capacity_fleet(
    fleet: Mapping[str, Any],
    *,
    preimage_archive: object,
    receipt_parent: Path,
    chain_id: str,
    chain_generation: int,
    readiness_job_id: str,
    evidence_observed_timestamp: object,
    verified: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _capacity_authority(verified)
    pins = authority["pins"]
    required = {
        "control_immutable_sha256",
        "rollout_generation",
        "server_pool_root",
        "fleet_id",
        "fleet_contract_path",
        "fleet_contract_sha256",
        "model_contract_sha256",
        "current_pointer_path",
        "current_pointer_sha256",
        "generation_ledger_path",
        "generation_ledger_sha256",
        "scheduler_captured_timestamp",
        "scheduler_age_seconds",
        "scheduler_sources",
        "isolated_foreign_scheduler_job_ids",
        "logical_replicas",
        "allocated_gpus",
        "running_replicas",
        "pending_replicas",
        "ignored_current_terminal_job_ids",
        "sealed_successful_handoff_terminal_job_ids",
        "overlap_replicas",
        "running",
        "pending",
        "captured_timestamp",
    }
    if set(fleet) != required:
        raise SentinelError("capacity-transient fleet schema is invalid")
    server_pool_root = fleet.get("server_pool_root")
    fleet_contract_path = fleet.get("fleet_contract_path")
    scheduler_captured = fleet.get("scheduler_captured_timestamp")
    scheduler_age = fleet.get("scheduler_age_seconds")
    captured = fleet.get("captured_timestamp")
    if (
        fleet.get("control_immutable_sha256") != authority["immutable_sha256"]
        or not isinstance(fleet.get("rollout_generation"), int)
        or isinstance(fleet.get("rollout_generation"), bool)
        or fleet["rollout_generation"] < 1
        or not isinstance(server_pool_root, str)
        or not Path(server_pool_root).is_absolute()
        or Path(server_pool_root).name != "schema5-v1"
        or fleet.get("fleet_id") != "schema5-v1"
        or fleet_contract_path != authority["fleet_path"]
        or fleet.get("fleet_contract_sha256")
        != pins.get("fleet_contract_sha256")
        or fleet.get("model_contract_sha256")
        != pins.get("model_contract_sha256")
        or not isinstance(fleet.get("current_pointer_path"), str)
        or not Path(fleet["current_pointer_path"]).is_absolute()
        or not _capacity_hex(fleet.get("current_pointer_sha256"), 64)
        or not isinstance(fleet.get("generation_ledger_path"), str)
        or not Path(fleet["generation_ledger_path"]).is_absolute()
        or not _capacity_hex(fleet.get("generation_ledger_sha256"), 64)
        or not _capacity_finite_number(scheduler_captured)
        or not _capacity_finite_number(scheduler_age)
        or not 0 <= float(scheduler_age) <= 600
        or not _capacity_finite_number(captured)
        or float(captured) < float(scheduler_captured)
        or not math.isclose(
            float(captured) - float(scheduler_captured),
            float(scheduler_age),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not _capacity_finite_number(evidence_observed_timestamp)
        or float(evidence_observed_timestamp) < float(captured)
        or float(evidence_observed_timestamp) - float(captured) > 600
        or fleet.get("scheduler_sources") != {"squeue": True, "sacct": True}
        or not _capacity_digit_list(
            fleet.get("isolated_foreign_scheduler_job_ids")
        )
        or not _capacity_digit_list(
            fleet.get("sealed_successful_handoff_terminal_job_ids")
        )
        or fleet.get("ignored_current_terminal_job_ids") != []
        or fleet.get("overlap_replicas") != []
        or fleet.get("logical_replicas")
        != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or fleet.get("allocated_gpus") != CAPACITY_TRANSIENT_EXPECTED_GPUS
    ):
        raise SentinelError("capacity-transient fleet identity/trust proof is invalid")

    archive = _validate_capacity_preimage_archive(
        preimage_archive,
        receipt_parent=receipt_parent,
        chain_id=chain_id,
        chain_generation=chain_generation,
        readiness_job_id=readiness_job_id,
        fleet=fleet,
    )
    pending_count = fleet.get("pending_replicas")
    running_count = fleet.get("running_replicas")
    pending = fleet.get("pending")
    running = fleet.get("running")
    if (
        not isinstance(pending_count, int)
        or isinstance(pending_count, bool)
        or pending_count < 1
        or not isinstance(running_count, int)
        or isinstance(running_count, bool)
        or running_count < 1
        or pending_count + running_count
        != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or not isinstance(pending, list)
        or not isinstance(running, list)
        or len(pending) != pending_count
        or len(running) != running_count
    ):
        raise SentinelError("capacity-transient fleet cardinality is invalid")
    for row in pending:
        if not isinstance(row, Mapping):
            raise SentinelError("capacity-transient pending row is not an object")
        _validate_capacity_pending_row(
            row, fleet=fleet, authority=authority, archive=archive
        )
    for row in running:
        if not isinstance(row, Mapping):
            raise SentinelError("capacity-transient running row is not an object")
        _validate_capacity_running_row(
            row,
            fleet=fleet,
            authority=authority,
            archive=archive,
            captured_timestamp=float(captured),
        )
    rows = [*pending, *running]
    replica_ids = [row["replica_id"] for row in rows]
    job_ids = [row["job_id"] for row in rows]
    allocated_gpus = [row["allocated_gpus"] for row in rows]
    foreign_ids = set(fleet["isolated_foreign_scheduler_job_ids"])
    sealed_ids = set(fleet["sealed_successful_handoff_terminal_job_ids"])
    if (
        len(set(replica_ids)) != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or set(replica_ids) != set(authority["expected_replicas"])
        or len(set(job_ids)) != CAPACITY_TRANSIENT_EXPECTED_REPLICAS
        or sum(allocated_gpus) != CAPACITY_TRANSIENT_EXPECTED_GPUS
        or set(job_ids) & foreign_ids
        or set(job_ids) & sealed_ids
        or foreign_ids & sealed_ids
    ):
        raise SentinelError("capacity-transient allocation partition is invalid")
    _validate_capacity_ledger(
        fleet,
        authority=authority,
        archive=archive,
        rows=rows,
    )
    return dict(archive["binding"])


def _validate_capacity_transient_receipt(
    path: Path,
    *,
    verified: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one marker-last capacity receipt without mutable scheduler reads."""

    path = _absolute(path)
    if path.name != CAPACITY_TRANSIENT_MARKER_NAME:
        raise SentinelError("capacity-transient receipt has the wrong marker name")
    marker = _read_json(path, description="fleet capacity-transient receipt")
    _require_read_only(path, description="fleet capacity-transient receipt")
    manifest = verified["manifest"]
    submission = verified["submission_receipt"]
    if not isinstance(manifest, Mapping) or not isinstance(submission, Mapping):
        raise SentinelError("verified chain omitted manifest/submission receipt")
    manifest_row, receipt_row = _fleet_readiness_rows(
        manifest=manifest, receipt=submission
    )
    job_rows = [
        row for row in jobs if str(row.get("name", "")) == "fleet_readiness"
    ]
    if len(job_rows) != 1:
        raise SentinelError("scheduler evidence has ambiguous fleet_readiness")
    job = job_rows[0]
    generation = receipt_row.get("generation", 0)
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
        or str(job.get("job_id")) != str(receipt_row.get("job_id"))
        or str(job.get("comment")) != str(receipt_row.get("comment"))
        or _normalize_state(str(job.get("state", ""))) != "FAILED"
        or str(job.get("exit_code") or "") != "75:0"
        or job.get("active") is not False
    ):
        raise SentinelError(
            "capacity-transient receipt is not paired with exact FAILED 75:0 "
            "fleet_readiness allocation"
        )

    marker_identity = dict(marker)
    receipt_id = marker_identity.pop("receipt_id", None)
    evidence_path = path.parent / CAPACITY_TRANSIENT_EVIDENCE_NAME
    evidence = _read_json(
        evidence_path, description="fleet capacity-transient evidence"
    )
    _require_read_only(
        evidence_path, description="fleet capacity-transient evidence"
    )
    evidence_identity = dict(evidence)
    evidence_id = evidence_identity.pop("evidence_id", None)
    fleet = evidence.get("fleet")
    preimage_archive = evidence.get("preimage_archive")
    boundary = evidence.get("boundary_proof")
    marker_fields = {
        "schema_version",
        "protocol",
        "passed",
        "capacity_transient_root",
        "chain_id",
        "chain_generation",
        "fleet_readiness_job_id",
        "fleet_readiness_comment",
        "evidence",
        "evidence_sha256",
        "evidence_id",
        "preimage_archive",
        "published_at",
        "published_timestamp",
        "receipt_id",
    }
    evidence_fields = {
        "schema_version",
        "protocol",
        "passed",
        "chain_protocol",
        "chain_id",
        "chain_generation",
        "manifest",
        "manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "fleet_readiness_job_id",
        "fleet_readiness_comment",
        "boundary_seconds",
        "boundary_proof",
        "fleet",
        "preimage_archive",
        "observed_at",
        "observed_timestamp",
        "evidence_id",
    }
    if (
        set(marker) != marker_fields
        or marker.get("schema_version") != 1
        or marker.get("protocol") != CAPACITY_TRANSIENT_PROTOCOL
        or marker.get("passed") is not True
        or marker.get("capacity_transient_root") is not True
        or marker.get("chain_id") != manifest["chain_id"]
        or marker.get("chain_generation") != generation
        or marker.get("fleet_readiness_job_id") != str(receipt_row["job_id"])
        or marker.get("fleet_readiness_comment") != receipt_row["comment"]
        or marker.get("evidence") != str(evidence_path.resolve())
        or marker.get("evidence_sha256") != _sha256(evidence_path)
        or marker.get("evidence_id") != evidence_id
        or marker.get("preimage_archive") != preimage_archive
        or not isinstance(receipt_id, str)
        or receipt_id != _sha256_bytes(_canonical_json(marker_identity))
        or set(evidence) != evidence_fields
        or evidence.get("schema_version") != 1
        or evidence.get("protocol") != CAPACITY_TRANSIENT_EVIDENCE_PROTOCOL
        or evidence.get("passed") is not True
        or evidence.get("chain_protocol") != R2_PROTOCOL
        or evidence.get("chain_id") != manifest["chain_id"]
        or evidence.get("chain_generation") != generation
        or evidence.get("manifest") != verified["manifest_path"]
        or evidence.get("manifest_sha256") != verified["manifest_sha256"]
        or evidence.get("submission_receipt")
        != verified["submission_receipt_path"]
        or evidence.get("submission_receipt_sha256")
        != verified["submission_receipt_sha256"]
        or evidence.get("submission_receipt_id") != submission["receipt_id"]
        or evidence.get("fleet_readiness_job_id") != str(receipt_row["job_id"])
        or evidence.get("fleet_readiness_comment") != receipt_row["comment"]
        or evidence.get("boundary_seconds") != CAPACITY_TRANSIENT_BOUNDARY_SECONDS
        or not isinstance(evidence_id, str)
        or evidence_id != _sha256_bytes(_canonical_json(evidence_identity))
        or not isinstance(boundary, Mapping)
        or set(boundary)
        != {
            "argv",
            "job_id",
            "comment",
            "job_name",
            "job_state",
            "effective_requeue",
            "runtime_seconds",
            "raw_output_sha256",
        }
        or boundary.get("argv")
        != [
            "scontrol",
            "show",
            "job",
            "-o",
            str(receipt_row["job_id"]),
        ]
        or boundary.get("job_id") != str(receipt_row["job_id"])
        or boundary.get("comment") != receipt_row["comment"]
        or boundary.get("job_name") != manifest_row["job_name"]
        or boundary.get("job_state") != "RUNNING"
        or boundary.get("effective_requeue") != 0
        or not isinstance(boundary.get("runtime_seconds"), int)
        or isinstance(boundary.get("runtime_seconds"), bool)
        or boundary["runtime_seconds"]
        < (
            CAPACITY_TRANSIENT_BOUNDARY_SECONDS
            - CAPACITY_TRANSIENT_BOUNDARY_TOLERANCE_SECONDS
        )
        or not _capacity_hex(boundary.get("raw_output_sha256"), 64)
        or not _capacity_finite_number(evidence.get("observed_timestamp"))
        or evidence.get("observed_at")
        != _utc(float(evidence["observed_timestamp"]))
        or not _capacity_finite_number(marker.get("published_timestamp"))
        or float(marker["published_timestamp"])
        < float(evidence["observed_timestamp"])
        or float(marker["published_timestamp"])
        - float(evidence["observed_timestamp"])
        > 600
        or marker.get("published_at")
        != _utc(float(marker["published_timestamp"]))
        or not isinstance(fleet, Mapping)
    ):
        raise SentinelError("capacity-transient receipt identity is invalid")

    archive_binding = _validate_capacity_fleet(
        fleet,
        preimage_archive=preimage_archive,
        receipt_parent=path.parent,
        chain_id=str(manifest["chain_id"]),
        chain_generation=generation,
        readiness_job_id=str(receipt_row["job_id"]),
        evidence_observed_timestamp=evidence["observed_timestamp"],
        verified=verified,
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "receipt_id": receipt_id,
        "evidence": str(evidence_path.resolve()),
        "evidence_sha256": _sha256(evidence_path),
        "evidence_id": evidence_id,
        "chain_generation": generation,
        "fleet_readiness_job_id": str(receipt_row["job_id"]),
        "preimage_archive": archive_binding,
    }


def _load_capacity_transient_receipt(
    path: Path | None,
    *,
    verified: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if path is None:
        return None
    supplied = _absolute(path)
    if not supplied.exists() and not supplied.is_symlink():
        return None
    return _validate_capacity_transient_receipt(
        supplied, verified=verified, jobs=jobs
    )


def _capacity_file_observation(path: Path) -> tuple[bool, bool, str | None]:
    try:
        metadata = path.lstat()
    except OSError:
        return False, False, None
    regular = stat.S_ISREG(metadata.st_mode)
    digest: str | None = None
    if regular:
        try:
            digest = _sha256(path)
        except OSError:
            digest = None
    return True, regular, digest


def _capacity_corruption_record(path: Path, error: Exception) -> dict[str, Any]:
    marker_path = _absolute(path)
    evidence_path = marker_path.parent / CAPACITY_TRANSIENT_EVIDENCE_NAME
    marker_present, marker_regular, marker_sha256 = _capacity_file_observation(
        marker_path
    )
    evidence_present, evidence_regular, evidence_sha256 = (
        _capacity_file_observation(evidence_path)
    )
    identity: dict[str, Any] = {
        "schema_version": 1,
        "kind": "capacity_transient_data_trust_failure",
        "path": str(marker_path),
        "present": marker_present,
        "regular_file": marker_regular,
        "sha256": marker_sha256,
        "evidence_path": str(evidence_path),
        "evidence_present": evidence_present,
        "evidence_regular_file": evidence_regular,
        "evidence_sha256": evidence_sha256,
        "error_type": type(error).__name__,
        "error": str(error)[:2_000],
    }
    identity["incident_id"] = _sha256_bytes(_canonical_json(identity))
    return identity


def _validate_capacity_corruption_record(value: object) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "path",
        "present",
        "regular_file",
        "sha256",
        "evidence_path",
        "evidence_present",
        "evidence_regular_file",
        "evidence_sha256",
        "error_type",
        "error",
        "incident_id",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise SentinelError("capacity-transient corruption record schema is invalid")
    identity = dict(value)
    incident_id = identity.pop("incident_id")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "capacity_transient_data_trust_failure"
        or not isinstance(value.get("path"), str)
        or not Path(str(value["path"])).is_absolute()
        or not isinstance(value.get("evidence_path"), str)
        or Path(str(value["evidence_path"]))
        != Path(str(value["path"])).parent / CAPACITY_TRANSIENT_EVIDENCE_NAME
        or any(
            not isinstance(value.get(field), bool)
            for field in (
                "present",
                "regular_file",
                "evidence_present",
                "evidence_regular_file",
            )
        )
        or (
            value.get("sha256") is not None
            and not _capacity_hex(value.get("sha256"), 64)
        )
        or (
            value.get("evidence_sha256") is not None
            and not _capacity_hex(value.get("evidence_sha256"), 64)
        )
        or (value.get("sha256") is not None and value.get("regular_file") is not True)
        or (
            value.get("evidence_sha256") is not None
            and value.get("evidence_regular_file") is not True
        )
        or not isinstance(value.get("error_type"), str)
        or not value["error_type"]
        or not isinstance(value.get("error"), str)
        or not value["error"]
        or not isinstance(incident_id, str)
        or incident_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise SentinelError("capacity-transient corruption record identity is invalid")
    return dict(value)


def _observe_capacity_transient_receipt(
    path: Path | None,
    *,
    verified: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    supplied = _absolute(path)
    if not supplied.exists() and not supplied.is_symlink():
        return None, None
    try:
        return (
            _validate_capacity_transient_receipt(
                supplied, verified=verified, jobs=jobs
            ),
            None,
        )
    except Exception as exc:
        # The afterany sentinel must still publish and alert when the marker exists
        # but is malformed.  It may never turn that failure into repair authority.
        return None, _capacity_corruption_record(supplied, exc)


def _bind_capacity_corruption(
    outcome: Mapping[str, Any],
    corruption: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(outcome)
    if corruption is None:
        return result
    record = _validate_capacity_corruption_record(corruption)
    superseding = result.get("superseding_failures")
    failures = list(superseding) if isinstance(superseding, list) else []
    failures.append(
        {
            "name": "capacity_transient_receipt",
            "state": "CORRUPT",
            "exit_code": "",
            "reason": str(record["error"]),
            "cause": "data_trust_failure",
        }
    )
    result.update(
        {
            "classification": "requires_superseding_release",
            "capacity_transient_roots": [],
            "superseding_failures": failures,
            "same_generation_repair_allowed": False,
            "requires_superseding_release": True,
            "run_succeeded": False,
            "capacity_transient_corruption": record,
        }
    )
    dispositions = result.get("dispositions")
    if isinstance(dispositions, Mapping) and "fleet_readiness" in dispositions:
        result["dispositions"] = dict(dispositions) | {
            "fleet_readiness": "requires_superseding_release"
        }
    return result


def _qualification_capacity_failure_binding(
    *,
    verified: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    rows = [
        row
        for row in jobs
        if row.get("name") == "throughput_qualification"
    ]
    if not rows:
        return None
    if len(rows) != 1:
        raise SentinelError(
            "qualification-capacity stage identity is ambiguous"
        )
    row = rows[0]
    if (
        _normalize_state(str(row.get("state", ""))) != "FAILED"
        or str(row.get("exit_code") or "")
        != QUALIFICATION_CAPACITY_EXIT_CODE
    ):
        return None
    manifest = verified.get("manifest")
    receipt = verified.get("submission_receipt")
    if not isinstance(manifest, Mapping) or not isinstance(receipt, Mapping):
        raise SentinelError(
            "qualification-capacity failure lacks verified chain inputs"
        )
    readiness_root_value = manifest.get("readiness_root")
    results_root_value = manifest.get("results_root")
    if (
        not isinstance(readiness_root_value, str)
        or not Path(readiness_root_value).is_absolute()
        or str(_absolute(readiness_root_value)) != readiness_root_value
        or not isinstance(results_root_value, str)
        or not Path(results_root_value).is_absolute()
        or str(_absolute(results_root_value)) != results_root_value
    ):
        raise SentinelError(
            "qualification-capacity chain roots are not lexically canonical"
        )
    readiness_root = _absolute(readiness_root_value)
    results_root = _absolute(results_root_value)
    base = readiness_root / QUALIFICATION_ROOT_NAME
    if (
        base.is_symlink()
        or not base.is_dir()
        or base.resolve() != base
        or results_root.is_symlink()
        or not results_root.is_dir()
        or results_root.resolve() != results_root
    ):
        raise SentinelError(
            "qualification-capacity artifact roots are unsafe"
        )
    current_path = base / QUALIFICATION_CURRENT_NAME
    current = _read_json(
        current_path,
        description="current qualification attempt",
    )
    _require_read_only(
        current_path,
        description="current qualification attempt",
    )
    current_metadata = current_path.lstat()
    pointer_value = current.get("pointer")
    if (
        set(current) != QUALIFICATION_CURRENT_FIELDS
        or current.get("schema_version") != 1
        or current.get("protocol") != QUALIFICATION_CURRENT_PROTOCOL
        or current_metadata.st_nlink != 1
        or not isinstance(pointer_value, str)
        or not Path(pointer_value).is_absolute()
        or str(_absolute(pointer_value)) != pointer_value
    ):
        raise SentinelError(
            "qualification-capacity current attempt is malformed"
        )
    pointer_path = _absolute(pointer_value)
    pointer_root = base / QUALIFICATION_POINTER_ROOT_NAME
    if (
        pointer_root.is_symlink()
        or not pointer_root.is_dir()
        or pointer_root.resolve() != pointer_root
        or pointer_path.parent != pointer_root
    ):
        raise SentinelError(
            "qualification-capacity pointer root is unsafe"
        )
    pointer = _read_json(
        pointer_path,
        description="qualification attempt pointer",
    )
    _require_read_only(
        pointer_path,
        description="qualification attempt pointer",
    )
    pointer_metadata = pointer_path.lstat()
    current_identity = dict(current)
    current_id = current_identity.pop("current_id", None)
    pointer_identity = dict(pointer)
    pointer_id = pointer_identity.pop("pointer_id", None)
    try:
        pointer_paths = sorted(pointer_root.iterdir())
    except OSError as exc:
        raise SentinelError(
            f"cannot inventory qualification attempt pointers: {exc}"
        ) from exc
    if (
        not pointer_paths
        or any(
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.suffix != ".json"
            for candidate in pointer_paths
        )
    ):
        raise SentinelError(
            "qualification attempt-pointer inventory is malformed"
        )
    readiness = pointer.get("readiness_generation")
    timestamp = pointer.get("created_timestamp")
    ordinal = pointer.get("attempt_ordinal")
    if (
        set(pointer) != QUALIFICATION_POINTER_FIELDS
        or pointer.get("schema_version") != 1
        or pointer.get("protocol") != QUALIFICATION_POINTER_PROTOCOL
        or pointer.get("chain_id") != manifest.get("chain_id")
        or pointer_metadata.st_nlink != 1
        or not isinstance(readiness, Mapping)
        or set(readiness) != QUALIFICATION_READINESS_FIELDS
        or any(
            not isinstance(readiness.get(field), int)
            or isinstance(readiness.get(field), bool)
            or int(readiness[field]) < 1
            for field in (
                "allowed_generation_tuple_count",
                "capacity_generation",
                "rollout_generation",
            )
        )
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(readiness.get(field, "")))
            is None
            for field in (
                "catalog_id",
                "marker_sha256",
                "inventory_sha256",
                "catalog_payload_sha256",
                "release_fleet_contract_sha256",
                "fleet_contract_sha256",
            )
        )
        or not isinstance(readiness.get("marker_path"), str)
        or not Path(str(readiness["marker_path"])).is_absolute()
        or str(_absolute(str(readiness["marker_path"])))
        != readiness["marker_path"]
        or not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or ordinal != len(pointer_paths)
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or float(timestamp) <= 0
        or pointer.get("created_at") != _utc(float(timestamp))
    ):
        raise SentinelError(
            "qualification-capacity attempt pointer is malformed"
        )
    attempt_id = (
        f"g{int(readiness['rollout_generation']):06d}-"
        f"c{int(readiness['capacity_generation']):06d}-"
        f"{readiness['catalog_id']}"
    )
    attempt_root = base / QUALIFICATION_ATTEMPT_ROOT_NAME / attempt_id
    run_root = (
        results_root
        / QUALIFICATION_RUN_ROOT_NAME
        / attempt_id
        / QUALIFICATION_ROOT_NAME
    )
    expected_pointer_path = (
        pointer_root / f"{ordinal:06d}-{attempt_id}.json"
    )
    if (
        current_id != _sha256_bytes(_canonical_json(current_identity))
        or pointer_id != _sha256_bytes(_canonical_json(pointer_identity))
        or pointer.get("attempt_id") != attempt_id
        or pointer.get("attempt_root") != str(attempt_root)
        or pointer.get("run_root") != str(run_root)
        or pointer.get("dispatcher_state")
        != str(attempt_root / "dispatcher")
        or pointer_path != expected_pointer_path
        or current.get("pointer_sha256") != _sha256(pointer_path)
        or current.get("pointer_id") != pointer_id
        or current.get("attempt_id") != attempt_id
        or pointer_path != pointer_paths[-1]
    ):
        raise SentinelError(
            "qualification-capacity current attempt binding is invalid"
        )
    for description, root in (
        ("failed qualification attempt", attempt_root),
        ("failed qualification run", run_root),
    ):
        if root.is_symlink() or not root.is_dir() or root.resolve() != root:
            raise SentinelError(f"{description} root is unsafe")
        for member in [root, *root.rglob("*")]:
            metadata = member.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not (
                    stat.S_ISDIR(metadata.st_mode)
                    or stat.S_ISREG(metadata.st_mode)
                )
                or (
                    stat.S_ISREG(metadata.st_mode)
                    and metadata.st_nlink != 1
                )
                or member.name.endswith(".publishing")
                or stat.S_IMODE(metadata.st_mode) & 0o222
            ):
                raise SentinelError(
                    f"{description} is not recursively sealed"
                )
    failure_path = attempt_root / QUALIFICATION_FAILURE_NAME
    failure = _read_json(
        failure_path,
        description="qualification capacity failure",
    )
    _require_read_only(
        failure_path,
        description="qualification capacity failure",
    )
    if failure_path.lstat().st_nlink != 1:
        raise SentinelError(
            "qualification capacity failure is hardlink-aliased"
        )
    identity = dict(failure)
    failure_id = identity.pop("failure_id", None)
    scaling = failure.get("additive_scaling_requirement")
    reason = str(failure.get("reason", ""))
    expected_attempt = {
        "path": str(pointer_path),
        "sha256": _sha256(pointer_path),
        "pointer_id": pointer_id,
        "attempt_id": attempt_id,
        "attempt_root": str(attempt_root),
        "run_root": str(run_root),
        "rollout_generation": readiness["rollout_generation"],
        "capacity_generation": readiness["capacity_generation"],
        "trusted_generation_catalog_id": readiness["catalog_id"],
    }
    profile = (
        scaling.get("serving_profile")
        if isinstance(scaling, Mapping)
        else None
    )
    tp = 2 if profile == "32B-long" else 1
    if (
        set(failure) != QUALIFICATION_FAILURE_FIELDS
        or failure.get("schema_version") != 1
        or failure.get("protocol") != QUALIFICATION_FAILURE_PROTOCOL
        or failure.get("passed") is not False
        or failure_id != _sha256_bytes(_canonical_json(identity))
        or re.fullmatch(r"[0-9a-f]{64}", str(failure.get("intent_id", "")))
        is None
        or failure.get("attempt") != expected_attempt
        or failure.get("readiness_generation")
        != readiness
        or "qualification throughput" not in reason
        or "is below" not in reason
        or failure.get("scheduler_capacity_mutated") is not False
        or not isinstance(failure.get("rerun_requirement"), str)
        or not str(failure["rerun_requirement"]).strip()
        or not isinstance(scaling, Mapping)
        or set(scaling) != QUALIFICATION_SCALING_FIELDS
        or profile not in QUALIFICATION_SERVING_PROFILES
        or scaling.get("server_pool_root")
        != manifest.get("server_pool_root")
        or scaling.get("additional_replicas") != 1
        or scaling.get("tensor_parallel_size") != tp
        or scaling.get("additional_gpus") != tp
        or scaling.get("capacity_mutated") is not False
    ):
        raise SentinelError(
            "qualification failure is not an exact throughput-only "
            "capacity-transition request"
        )
    return {
        "path": str(failure_path),
        "sha256": _sha256(failure_path),
        "failure_id": failure_id,
        "attempt": expected_attempt,
        "serving_profile": profile,
        "tensor_parallel_size": tp,
        "failed_job_id": str(row.get("job_id")),
        "failed_comment": str(row.get("comment")),
        "submission_receipt_id": receipt.get("receipt_id"),
    }


def _observe_qualification_capacity_failure(
    *,
    verified: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Treat a malformed exit-76 preimage as superseding, never as authority."""

    try:
        return _qualification_capacity_failure_binding(
            verified=verified,
            jobs=jobs,
        )
    except Exception:
        return None


def classify_recovery_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    capacity_transient_receipt: Mapping[str, Any] | None = None,
    qualification_capacity_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the fail-closed v1.2 same-generation repair contract."""

    names = [str(row.get("name", "")) for row in jobs]
    if len(names) != len(set(names)) or any(not name for name in names):
        raise SentinelError("reconciled scheduler rows have invalid job names")
    sentinel_rows = [row for row in jobs if row["name"] == "failure_sentinel"]
    if len(sentinel_rows) != 1:
        raise SentinelError("v1.2 recovery chain must contain one failure_sentinel")
    sentinel = sentinel_rows[0]
    if sentinel.get("dependency_type") != "afterany":
        raise SentinelError("failure_sentinel is not bound through afterany")
    if (
        _normalize_state(str(sentinel.get("state", ""))) != "RUNNING"
        or sentinel.get("active") is not True
    ):
        raise SentinelError(
            "the exact failure_sentinel allocation must be RUNNING while it "
            "evaluates the receipt"
        )
    by_name = {str(row["name"]): row for row in jobs}
    new_contract_detected = bool(set(STAGE_SENTINEL_NAMES) & set(by_name))
    if new_contract_detected:
        missing_production = set(PRODUCTION_STAGE_NAMES) - set(by_name)
        missing_observers = set(STAGE_SENTINEL_NAMES) - set(by_name)
        unexpected = set(by_name) - {
            *PRODUCTION_STAGE_NAMES,
            *STAGE_SENTINEL_NAMES,
            "failure_sentinel",
        }
        if missing_production or missing_observers or unexpected:
            raise SentinelError(
                "v1.2 recovery-chain stage/observer set drifted: "
                f"missing_stages={sorted(missing_production)}, "
                f"missing_observers={sorted(missing_observers)}, "
                f"unexpected={sorted(unexpected)}"
            )
        production_names = PRODUCTION_STAGE_NAMES
        observer_names = STAGE_SENTINEL_NAMES
    else:
        # Historical compact unit fixtures predate the v1.2-r2 fail-fast expansion.
        # Native chain verification can never admit this branch for production
        # evidence; keeping it here preserves pure-classifier regression coverage.
        production_names = tuple(
            name for name in by_name if name != "failure_sentinel"
        )
        observer_names = ()
    production = [by_name[name] for name in production_names]
    observers = [by_name[name] for name in observer_names]
    expected_dependencies = {*production_names, *observer_names}
    if set(str(item) for item in sentinel["dependencies"]) != expected_dependencies:
        raise SentinelError(
            "failure_sentinel does not depend on every production stage and "
            "fail-fast observer"
        )

    active = [
        str(row["name"])
        for row in (*production, *observers)
        if row.get("active") is True
    ]
    if active:
        raise SentinelError(
            "aggregate afterany sentinel observed active stage/observer jobs: "
            f"{sorted(active)}"
        )

    disposition: dict[str, str] = {}
    transient_roots: list[str] = []
    capacity_transient_roots: list[str] = []
    qualification_capacity_roots: list[str] = []
    dependency_cancelled: list[str] = []
    superseding: list[dict[str, str]] = []
    for row in production:
        name = str(row["name"])
        state = _normalize_state(str(row.get("state", "")))
        exit_code = str(row.get("exit_code") or "")
        dependencies = [str(value) for value in row.get("dependencies", [])]
        if state == "COMPLETED" and exit_code == "0:0":
            disposition[name] = "completed"
            continue
        if state in TRANSIENT_ROOT_STATES:
            disposition[name] = "transient_root"
            transient_roots.append(name)
            continue
        if (
            name == "fleet_readiness"
            and state == "FAILED"
            and exit_code == "75:0"
            and capacity_transient_receipt is not None
            and capacity_transient_receipt.get("fleet_readiness_job_id")
            == str(row.get("job_id"))
        ):
            disposition[name] = "capacity_transient_root"
            capacity_transient_roots.append(name)
            continue
        if (
            name == "throughput_qualification"
            and state == "FAILED"
            and exit_code == QUALIFICATION_CAPACITY_EXIT_CODE
            and qualification_capacity_failure is not None
            and qualification_capacity_failure.get("failed_job_id")
            == str(row.get("job_id"))
            and qualification_capacity_failure.get("failed_comment")
            == str(row.get("comment"))
        ):
            disposition[name] = "qualification_capacity_transition_root"
            qualification_capacity_roots.append(name)
            continue
        causal_dependencies = [
            dependency
            for dependency in dependencies
            if disposition.get(dependency)
            in {
                "transient_root",
                "capacity_transient_root",
                "qualification_capacity_transition_root",
                "dependency_cancelled",
            }
        ]
        if (
            state == "CANCELLED"
            and causal_dependencies
            and _dependency_reason(row.get("reason"))
            and _never_started(row.get("start"))
            and _zero_elapsed(row.get("elapsed"))
        ):
            disposition[name] = "dependency_cancelled"
            dependency_cancelled.append(name)
            continue

        disposition[name] = "requires_superseding_release"
        detail = {
            "name": name,
            "state": state or "UNKNOWN",
            "exit_code": exit_code,
            "reason": str(row.get("reason") or ""),
        }
        if state == "COMPLETED":
            detail["cause"] = "completed_with_nonzero_or_invalid_exit_code"
        elif state in SUPERSEDING_STATES:
            detail["cause"] = "deterministic_or_ambiguous_terminal_failure"
        elif state in ACTIVE_STATES:
            detail["cause"] = "active_state_without_live_scheduler_row"
        else:
            detail["cause"] = "unclassified_terminal_state"
        superseding.append(detail)

    observer_failures: list[dict[str, str]] = []
    for stage, observer_name in zip(
        production_names if observer_names else (),
        observer_names,
        strict=True,
    ):
        row = by_name[observer_name]
        state = _normalize_state(str(row.get("state", "")))
        exit_code = str(row.get("exit_code") or "")
        dependencies = [str(value) for value in row.get("dependencies", [])]
        if (
            row.get("dependency_type") != "afterany"
            or dependencies != [stage]
        ):
            raise SentinelError(
                f"stage observer dependency contract drifted for {stage}"
            )
        if state == "COMPLETED" and exit_code == "0:0":
            disposition[observer_name] = "completed"
            continue
        if state in TRANSIENT_ROOT_STATES:
            disposition[observer_name] = "transient_root"
            transient_roots.append(observer_name)
            continue
        disposition[observer_name] = "requires_superseding_release"
        detail = {
            "name": observer_name,
            "state": state or "UNKNOWN",
            "exit_code": exit_code,
            "reason": str(row.get("reason") or ""),
            "cause": "fail_fast_observer_failure",
        }
        observer_failures.append(detail)
        superseding.append(detail)

    if superseding:
        classification = "requires_superseding_release"
    elif qualification_capacity_roots:
        classification = "qualification_capacity_transition_required"
    elif transient_roots or capacity_transient_roots or dependency_cancelled:
        classification = "transient_repairable"
    else:
        classification = "complete"
    return {
        "classification": classification,
        "production_job_count": len(production),
        "stage_observer_job_count": len(observers),
        "completed_job_count": sum(
            disposition.get(name) == "completed"
            for name in production_names
        ),
        "completed_stage_observer_count": sum(
            disposition.get(name) == "completed"
            for name in observer_names
        ),
        "transient_roots": transient_roots,
        "capacity_transient_roots": capacity_transient_roots,
        "qualification_capacity_transition_roots": (
            qualification_capacity_roots
        ),
        "dependency_cancelled_suffix": dependency_cancelled,
        "superseding_failures": superseding,
        "stage_observer_failures": observer_failures,
        "dispositions": disposition,
        "same_generation_repair_allowed": (
            classification == "transient_repairable"
        ),
        "requires_superseding_release": (
            classification == "requires_superseding_release"
        ),
        "run_succeeded": classification == "complete",
        "sentinel_scheduler_state": str(sentinel["state"]),
    }


def _scheduler_evidence(
    *,
    verified: Mapping[str, Any],
    jobs: list[dict[str, Any]],
    query: Mapping[str, Any],
    classification: Mapping[str, Any],
    capacity_transient_receipt: Mapping[str, Any] | None,
    qualification_capacity_failure: Mapping[str, Any] | None,
    timestamp: float,
) -> dict[str, Any]:
    manifest = verified["manifest"]
    receipt = verified["submission_receipt"]
    assert isinstance(manifest, Mapping)
    assert isinstance(receipt, Mapping)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": SCHEDULER_EVIDENCE_PROTOCOL,
        "passed": True,
        "chain_protocol": verified["chain_protocol"],
        "chain_id": manifest["chain_id"],
        "manifest": verified["manifest_path"],
        "manifest_sha256": verified["manifest_sha256"],
        "submission_receipt": verified["submission_receipt_path"],
        "submission_receipt_sha256": verified["submission_receipt_sha256"],
        "submission_receipt_id": receipt["receipt_id"],
        "observed_at": _utc(timestamp),
        "observed_timestamp": timestamp,
        "scheduler_queries": query,
        "jobs": jobs,
        "capacity_transient_receipt": (
            None
            if capacity_transient_receipt is None
            else dict(capacity_transient_receipt)
        ),
        "qualification_capacity_failure": (
            None
            if qualification_capacity_failure is None
            else dict(qualification_capacity_failure)
        ),
        "outcome": dict(classification),
    }
    payload["evidence_id"] = _sha256_bytes(_canonical_json(payload))
    return payload


def _derive_persisted_scheduler_rows(
    evidence: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reparse archived raw scheduler output and independently rebuild job rows."""

    queries = evidence.get("scheduler_queries")
    if not isinstance(queries, Mapping) or set(queries) != {"squeue", "sacct"}:
        raise SentinelError("scheduler query evidence has the wrong sources")
    requested_ids, expected_squeue_argv, expected_sacct_argv = (
        _scheduler_query_contract(receipt)
    )
    parsed: dict[str, dict[str, dict[str, str]]] = {}
    for name, expected_argv, parser in (
        ("squeue", expected_squeue_argv, _parse_squeue),
        ("sacct", expected_sacct_argv, _parse_sacct),
    ):
        query = queries.get(name)
        if not isinstance(query, Mapping) or set(query) != {
            "argv",
            "stdout",
            "stdout_sha256",
            "row_count",
        }:
            raise SentinelError(f"{name} scheduler query evidence is incomplete")
        stdout = query.get("stdout")
        row_count = query.get("row_count")
        if (
            query.get("argv") != expected_argv
            or not isinstance(stdout, str)
            or query.get("stdout_sha256")
            != _sha256_bytes(stdout.encode("utf-8"))
            or not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count < 0
        ):
            raise SentinelError(f"{name} scheduler query evidence drifted")
        rows = parser(stdout, requested_ids=requested_ids)
        if len(rows) != row_count:
            raise SentinelError(f"{name} scheduler row count is not derived")
        parsed[name] = rows
    return _join_scheduler_rows(
        manifest=manifest,
        receipt=receipt,
        live=parsed["squeue"],
        accounting=parsed["sacct"],
    )


def classify_stage_observation(
    jobs: Sequence[Mapping[str, Any]],
    *,
    stage_name: str,
    stage_sentinel_job_id: str,
) -> dict[str, Any]:
    """Classify one terminal stage without granting recovery authority."""

    if stage_name not in PRODUCTION_STAGE_NAMES:
        raise SentinelError(f"unknown fail-fast stage target: {stage_name!r}")
    if not stage_sentinel_job_id.isdigit():
        raise SentinelError("stage-sentinel job ID must be numeric")
    by_name = {
        str(row.get("name", "")): row
        for row in jobs
        if isinstance(row, Mapping)
    }
    if len(by_name) != len(jobs):
        raise SentinelError("stage observation contains duplicate job names")
    observer_name = f"{STAGE_SENTINEL_PREFIX}{stage_name}"
    stage = by_name.get(stage_name)
    observer = by_name.get(observer_name)
    if stage is None or observer is None:
        raise SentinelError("stage observation lacks its target or exact observer")
    if (
        str(observer.get("job_id", "")) != stage_sentinel_job_id
        or observer.get("dependency_type") != "afterany"
        or [str(item) for item in observer.get("dependencies", [])]
        != [stage_name]
        or _normalize_state(str(observer.get("state", ""))) != "RUNNING"
        or observer.get("active") is not True
    ):
        raise SentinelError(
            "exact fail-fast stage observer is not the running afterany allocation"
        )
    state = _normalize_state(str(stage.get("state", "")))
    exit_code = str(stage.get("exit_code") or "")
    if stage.get("active") is True or state in ACTIVE_STATES:
        raise SentinelError(
            f"fail-fast observer ran before target became terminal: {stage_name}"
        )
    succeeded = state == "COMPLETED" and exit_code == "0:0"
    if succeeded:
        disposition = "completed"
        classification = "complete"
    elif state in TRANSIENT_ROOT_STATES:
        disposition = "transient_failure_observed"
        classification = "stage_failure_observed"
    elif (
        state == "CANCELLED"
        and _dependency_reason(stage.get("reason"))
        and _never_started(stage.get("start"))
        and _zero_elapsed(stage.get("elapsed"))
    ):
        disposition = "dependency_cancellation_observed"
        classification = "stage_failure_observed"
    else:
        disposition = "terminal_failure_observed"
        classification = "stage_failure_observed"
    return {
        "classification": classification,
        "target_stage": stage_name,
        "target_job_id": str(stage.get("job_id", "")),
        "target_state": state or "UNKNOWN",
        "target_exit_code": exit_code,
        "target_reason": str(stage.get("reason") or ""),
        "target_start": stage.get("start"),
        "target_elapsed": stage.get("elapsed"),
        "stage_disposition": disposition,
        "stage_sentinel_name": observer_name,
        "stage_sentinel_job_id": stage_sentinel_job_id,
        "stage_sentinel_scheduler_state": str(observer.get("state", "")),
        "run_succeeded": succeeded,
        # Fail-fast observers are evidence/alerting only.  The aggregate sentinel is
        # the sole authority for repair or superseding-release classification.
        "authoritative_repair_classification": False,
        "same_generation_repair_allowed": False,
        "requires_superseding_release": False,
    }


def _stage_scheduler_evidence(
    *,
    verified: Mapping[str, Any],
    jobs: list[dict[str, Any]],
    query: Mapping[str, Any],
    outcome: Mapping[str, Any],
    timestamp: float,
) -> dict[str, Any]:
    manifest = verified["manifest"]
    receipt = verified["submission_receipt"]
    assert isinstance(manifest, Mapping)
    assert isinstance(receipt, Mapping)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": STAGE_SCHEDULER_EVIDENCE_PROTOCOL,
        "passed": True,
        "chain_protocol": verified["chain_protocol"],
        "chain_id": manifest["chain_id"],
        "manifest": verified["manifest_path"],
        "manifest_sha256": verified["manifest_sha256"],
        "submission_receipt": verified["submission_receipt_path"],
        "submission_receipt_sha256": verified["submission_receipt_sha256"],
        "submission_receipt_id": receipt["receipt_id"],
        "observed_at": _utc(timestamp),
        "observed_timestamp": timestamp,
        "scheduler_queries": query,
        "jobs": jobs,
        "outcome": dict(outcome),
    }
    payload["evidence_id"] = _sha256_bytes(_canonical_json(payload))
    return payload


def _validate_stage_scheduler_evidence(
    path: Path,
    *,
    verified: Mapping[str, Any],
    stage_name: str,
    stage_sentinel_job_id: str,
) -> dict[str, Any]:
    evidence = _read_json(path, description="stage scheduler sentinel evidence")
    _require_read_only(path, description="stage scheduler sentinel evidence")
    identity = dict(evidence)
    evidence_id = identity.pop("evidence_id", None)
    manifest = verified["manifest"]
    receipt = verified["submission_receipt"]
    required = {
        "schema_version",
        "protocol",
        "passed",
        "chain_protocol",
        "chain_id",
        "manifest",
        "manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "observed_at",
        "observed_timestamp",
        "scheduler_queries",
        "jobs",
        "outcome",
        "evidence_id",
    }
    observed_timestamp = evidence.get("observed_timestamp")
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(receipt, Mapping)
        or set(evidence) != required
        or evidence.get("schema_version") != 1
        or evidence.get("protocol") != STAGE_SCHEDULER_EVIDENCE_PROTOCOL
        or evidence.get("passed") is not True
        or evidence.get("chain_protocol") != R2_PROTOCOL
        or evidence.get("chain_id") != manifest.get("chain_id")
        or evidence.get("manifest") != verified["manifest_path"]
        or evidence.get("manifest_sha256") != verified["manifest_sha256"]
        or evidence.get("submission_receipt")
        != verified["submission_receipt_path"]
        or evidence.get("submission_receipt_sha256")
        != verified["submission_receipt_sha256"]
        or evidence.get("submission_receipt_id") != receipt.get("receipt_id")
        or not isinstance(observed_timestamp, (int, float))
        or isinstance(observed_timestamp, bool)
        or not math.isfinite(float(observed_timestamp))
        or float(observed_timestamp) < 0
        or evidence.get("observed_at") != _utc(float(observed_timestamp))
        or not isinstance(evidence_id, str)
        or evidence_id != _sha256_bytes(_canonical_json(identity))
        or not isinstance(evidence.get("jobs"), list)
        or not isinstance(evidence.get("outcome"), Mapping)
    ):
        raise SentinelError("stage scheduler sentinel evidence identity is invalid")
    derived_jobs = _derive_persisted_scheduler_rows(
        evidence,
        manifest=manifest,
        receipt=receipt,
    )
    if evidence["jobs"] != derived_jobs:
        raise SentinelError(
            "stage scheduler rows are not derived from archived scheduler output"
        )
    expected = classify_stage_observation(
        derived_jobs,
        stage_name=stage_name,
        stage_sentinel_job_id=stage_sentinel_job_id,
    )
    if evidence["outcome"] != expected:
        raise SentinelError("stage sentinel outcome is not scheduler-derived")
    return evidence


def _validate_scheduler_evidence(
    path: Path,
    *,
    verified: Mapping[str, Any],
    capacity_transient_receipt_path: Path | None = None,
) -> dict[str, Any]:
    evidence = _read_json(path, description="scheduler sentinel evidence")
    _require_read_only(path, description="scheduler sentinel evidence")
    identity = dict(evidence)
    evidence_id = identity.pop("evidence_id", None)
    manifest = verified["manifest"]
    receipt = verified["submission_receipt"]
    if not isinstance(manifest, Mapping) or not isinstance(receipt, Mapping):
        raise SentinelError("verified protocol result omitted manifest or receipt")
    required = {
        "schema_version",
        "protocol",
        "passed",
        "chain_protocol",
        "chain_id",
        "manifest",
        "manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "observed_at",
        "observed_timestamp",
        "scheduler_queries",
        "jobs",
        "capacity_transient_receipt",
        "qualification_capacity_failure",
        "outcome",
        "evidence_id",
    }
    observed_timestamp = evidence.get("observed_timestamp")
    if (
        set(evidence) != required
        or evidence["schema_version"] != 1
        or evidence["protocol"] != SCHEDULER_EVIDENCE_PROTOCOL
        or evidence["passed"] is not True
        or evidence["chain_protocol"] != R2_PROTOCOL
        or evidence["chain_id"] != manifest["chain_id"]
        or evidence["manifest"] != verified["manifest_path"]
        or evidence["manifest_sha256"] != verified["manifest_sha256"]
        or evidence["submission_receipt"]
        != verified["submission_receipt_path"]
        or evidence["submission_receipt_sha256"]
        != verified["submission_receipt_sha256"]
        or evidence["submission_receipt_id"] != receipt["receipt_id"]
        or not isinstance(observed_timestamp, (int, float))
        or isinstance(observed_timestamp, bool)
        or not math.isfinite(float(observed_timestamp))
        or observed_timestamp < 0
        or evidence.get("observed_at") != _utc(float(observed_timestamp))
        or not isinstance(evidence_id, str)
        or evidence_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise SentinelError("scheduler sentinel evidence identity is invalid")
    if not isinstance(evidence["outcome"], dict):
        raise SentinelError("scheduler sentinel evidence has no outcome")
    jobs = evidence.get("jobs")
    if not isinstance(jobs, list):
        raise SentinelError("scheduler sentinel evidence has no exact job rows")
    derived_jobs = _derive_persisted_scheduler_rows(
        evidence,
        manifest=manifest,
        receipt=receipt,
    )
    if jobs != derived_jobs:
        raise SentinelError(
            "scheduler sentinel job rows are not derived from archived scheduler "
            "output"
        )
    embedded = evidence["capacity_transient_receipt"]
    corruption = evidence["outcome"].get("capacity_transient_corruption")
    if corruption is not None:
        recorded_corruption = _validate_capacity_corruption_record(corruption)
        marker_still_present = Path(recorded_corruption["path"]).exists()
        evidence_still_present = Path(
            recorded_corruption["evidence_path"]
        ).exists()
        if marker_still_present or evidence_still_present:
            observed, observed_corruption = _observe_capacity_transient_receipt(
                capacity_transient_receipt_path,
                verified=verified,
                jobs=jobs,
            )
        else:
            # The immutable scheduler artifact is the durable incident record after
            # operators archive or remove the corrupt external preimage.
            observed, observed_corruption = None, recorded_corruption
    else:
        observed, observed_corruption = _observe_capacity_transient_receipt(
            capacity_transient_receipt_path,
            verified=verified,
            jobs=jobs,
        )
    if embedded != observed or corruption != observed_corruption:
        raise SentinelError(
            "scheduler sentinel capacity-transient observation drifted"
        )
    expected_outcome = classify_recovery_jobs(
        jobs,
        capacity_transient_receipt=observed,
        qualification_capacity_failure=(
            _observe_qualification_capacity_failure(
                verified=verified,
                jobs=jobs,
            )
        ),
    )
    expected_outcome = _bind_capacity_corruption(
        expected_outcome,
        observed_corruption,
    )
    if evidence["outcome"] != expected_outcome:
        raise SentinelError(
            "scheduler sentinel outcome is not derived from exact scheduler rows"
        )
    if (
        evidence["qualification_capacity_failure"]
        != _observe_qualification_capacity_failure(
            verified=verified,
            jobs=jobs,
        )
    ):
        raise SentinelError(
            "scheduler sentinel qualification-capacity binding drifted"
        )
    return evidence


def _mail_identity(evidence: Mapping[str, Any], recipient: str) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "evidence_id": evidence["evidence_id"],
                "recipient": recipient,
            }
        )
    )


def _new_mail_state(
    evidence: Mapping[str, Any], *, recipient: str, timestamp: float
) -> dict[str, Any]:
    requested = evidence["outcome"]["classification"] != "complete"
    return {
        "schema_version": 1,
        "protocol": MAIL_PROTOCOL,
        "mail_id": _mail_identity(evidence, recipient),
        "evidence_id": evidence["evidence_id"],
        "recipient": recipient,
        "requested": requested,
        "delivered": False,
        "attempt_count": 0,
        "created_at": _utc(timestamp),
        "created_timestamp": timestamp,
        "attempted_at": None,
        "attempted_timestamp": None,
        "delivered_at": None,
        "delivered_timestamp": None,
        "next_retry_at": _utc(timestamp) if requested else None,
        "next_retry_timestamp": timestamp if requested else None,
        "in_flight_token": None,
        "error": None,
        "delivery_receipt": None,
    }


def _validate_mail_state(
    state: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    recipient: str,
) -> None:
    required = {
        "schema_version",
        "protocol",
        "mail_id",
        "evidence_id",
        "recipient",
        "requested",
        "delivered",
        "attempt_count",
        "created_at",
        "created_timestamp",
        "attempted_at",
        "attempted_timestamp",
        "delivered_at",
        "delivered_timestamp",
        "next_retry_at",
        "next_retry_timestamp",
        "in_flight_token",
        "error",
        "delivery_receipt",
    }
    expected_requested = evidence["outcome"]["classification"] != "complete"
    if (
        set(state) != required
        or state["schema_version"] != 1
        or state["protocol"] != MAIL_PROTOCOL
        or state["mail_id"] != _mail_identity(evidence, recipient)
        or state["evidence_id"] != evidence["evidence_id"]
        or state["recipient"] != recipient
        or state["requested"] is not expected_requested
        or not isinstance(state["delivered"], bool)
        or not isinstance(state["attempt_count"], int)
        or isinstance(state["attempt_count"], bool)
        or state["attempt_count"] < 0
    ):
        raise SentinelError("persisted sentinel mail state is invalid")

    def timestamp_pair(prefix: str) -> float | None:
        rendered = state[f"{prefix}_at"]
        raw = state[f"{prefix}_timestamp"]
        if rendered is None or raw is None:
            if rendered is not None or raw is not None:
                raise SentinelError(
                    f"persisted sentinel mail {prefix} timestamp is partial"
                )
            return None
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
            or raw < 0
            or rendered != _utc(float(raw))
        ):
            raise SentinelError(
                f"persisted sentinel mail {prefix} timestamp is invalid"
            )
        return float(raw)

    created = timestamp_pair("created")
    attempted = timestamp_pair("attempted")
    delivered = timestamp_pair("delivered")
    next_retry = timestamp_pair("next_retry")
    assert created is not None
    attempt_count = int(state["attempt_count"])
    in_flight = state["in_flight_token"]
    error = state["error"]
    delivery_receipt = state["delivery_receipt"]
    if (
        (in_flight is not None and (
            not isinstance(in_flight, str)
            or re.fullmatch(r"[0-9a-f]{32}", in_flight) is None
        ))
        or (
            error is not None
            and (
                not isinstance(error, str)
                or not error
                or len(error) > 500
            )
        )
    ):
        raise SentinelError("persisted sentinel mail attempt state is invalid")
    if not expected_requested:
        if (
            state["delivered"] is not False
            or attempt_count != 0
            or attempted is not None
            or delivered is not None
            or next_retry is not None
            or in_flight is not None
            or error is not None
            or delivery_receipt is not None
        ):
            raise SentinelError(
                "non-requested sentinel mail state contains delivery claims"
            )
        return
    if attempt_count == 0:
        if (
            state["delivered"] is not False
            or attempted is not None
            or delivered is not None
            or next_retry != created
            or in_flight is not None
            or error is not None
            or delivery_receipt is not None
        ):
            raise SentinelError("unattempted sentinel mail state is inconsistent")
        return
    if attempted is None or attempted < created:
        raise SentinelError("attempted sentinel mail timestamp is inconsistent")
    if state["delivered"] is True:
        expected_receipt = {
            "protocol": "mail-command-returncode-v1",
            "attempt_count": attempt_count,
            "returncode": 0,
            "completed_timestamp": delivered,
            "in_flight_token_sha256": (
                None
                if not isinstance(delivery_receipt, Mapping)
                else delivery_receipt.get("in_flight_token_sha256")
            ),
        }
        if (
            delivered is None
            or delivered < attempted
            or next_retry is not None
            or in_flight is not None
            or error is not None
            or not isinstance(delivery_receipt, Mapping)
            or set(delivery_receipt) != set(expected_receipt)
            or dict(delivery_receipt) != expected_receipt
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(delivery_receipt.get("in_flight_token_sha256", "")),
            )
            is None
        ):
            raise SentinelError(
                "delivered sentinel mail lacks a confirmed-send receipt"
            )
    elif (
        delivered is not None
        or delivery_receipt is not None
        or (
            in_flight is not None
            and (error is not None or next_retry is None or next_retry > attempted)
        )
        or (
            in_flight is None
            and (
                error is None
                or next_retry is None
                or next_retry <= attempted
            )
        )
    ):
        raise SentinelError("undelivered sentinel mail state is inconsistent")


def _load_or_create_mail_state(
    path: Path,
    *,
    evidence: Mapping[str, Any],
    recipient: str,
    timestamp: float,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        state = _read_json(path, description="sentinel mail state")
        _validate_mail_state(state, evidence=evidence, recipient=recipient)
        return state
    state = _new_mail_state(evidence, recipient=recipient, timestamp=timestamp)
    _atomic_json(path, state, mode=0o640)
    return state


def _persist_mail_state(path: Path, state: Mapping[str, Any]) -> None:
    _atomic_json(path, state, mode=0o640)


def _attempt_mail(
    path: Path,
    *,
    evidence: Mapping[str, Any],
    recipient: str,
    mail_runner: MailRunner,
    clock: Clock,
) -> tuple[dict[str, Any], bool]:
    now = float(clock())
    if not math.isfinite(now):
        raise SentinelError("mail clock returned a non-finite timestamp")
    state = _load_or_create_mail_state(
        path,
        evidence=evidence,
        recipient=recipient,
        timestamp=now,
    )
    if not state["requested"] or state["delivered"]:
        return state, False
    next_retry = state["next_retry_timestamp"]
    if isinstance(next_retry, (int, float)) and now < float(next_retry):
        return state, False
    in_flight = state["in_flight_token"]
    attempted = state["attempted_timestamp"]
    if (
        isinstance(in_flight, str)
        and in_flight
        and isinstance(attempted, (int, float))
        and now - float(attempted) < IN_FLIGHT_STALE_SECONDS
    ):
        return state, False

    token = uuid.uuid4().hex
    attempt_count = int(state["attempt_count"]) + 1
    state.update(
        {
            "attempt_count": attempt_count,
            "attempted_at": _utc(now),
            "attempted_timestamp": now,
            "in_flight_token": token,
            "error": None,
        }
    )
    _persist_mail_state(path, state)

    outcome = evidence["outcome"]
    classification = str(outcome["classification"])
    target_stage = outcome.get("target_stage")
    if isinstance(target_stage, str):
        subject = (
            f"[agents-scaling:recovery:fail-fast] "
            f"{target_stage} {classification}"
        )
        body = (
            "Schema-5 v1.2-r2 fail-fast stage observation\n\n"
            f"Stage: {target_stage}\n"
            f"Classification: {classification}\n"
            f"Target job: {outcome.get('target_job_id')}\n"
            f"Target state/exit: {outcome.get('target_state')}/"
            f"{outcome.get('target_exit_code')}\n"
            f"Chain: {evidence['chain_id']}\n"
            f"Evidence: {path.parent / STAGE_SCHEDULER_EVIDENCE_NAME}\n"
            "Repair authority: none; wait for the aggregate sentinel classifier.\n"
        )
    else:
        subject = f"[agents-scaling:recovery] {classification}"
        body = (
            "Schema-5 v1.2-r2 recovery-chain outcome\n\n"
            f"Classification: {classification}\n"
            f"Chain: {evidence['chain_id']}\n"
            f"Evidence: {path.parent / SCHEDULER_EVIDENCE_NAME}\n"
            f"Same-generation repair allowed: "
            f"{outcome['same_generation_repair_allowed']}\n"
            f"Requires superseding release: "
            f"{outcome['requires_superseding_release']}\n"
        )
    corruption = outcome.get("capacity_transient_corruption")
    if isinstance(corruption, Mapping):
        body += (
            "Capacity receipt data-trust failure: "
            f"{corruption.get('error')}\n"
            f"Capacity receipt path: {corruption.get('path')}\n"
            f"Capacity receipt SHA-256: {corruption.get('sha256')}\n"
        )
    delivered = False
    error: str | None = None
    try:
        result = mail_runner(["mail", "-s", subject, recipient], body)
        delivered = result.returncode == 0
        if not delivered:
            error = result.stderr.strip()[:500] or f"mail exited {result.returncode}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]

    completed = float(clock())
    if not math.isfinite(completed) or completed < now:
        raise SentinelError("mail completion clock is invalid")
    current = _read_json(path, description="sentinel mail state")
    _validate_mail_state(current, evidence=evidence, recipient=recipient)
    if current["in_flight_token"] != token:
        raise SentinelError("sentinel mail delivery intent was superseded")
    current["in_flight_token"] = None
    current["error"] = error
    if delivered:
        current["delivered"] = True
        current["delivered_at"] = _utc(completed)
        current["delivered_timestamp"] = completed
        current["next_retry_at"] = None
        current["next_retry_timestamp"] = None
        current["delivery_receipt"] = {
            "protocol": "mail-command-returncode-v1",
            "attempt_count": attempt_count,
            "returncode": 0,
            "completed_timestamp": completed,
            "in_flight_token_sha256": _sha256_bytes(token.encode("ascii")),
        }
    else:
        current["delivery_receipt"] = None
        delay = RETRY_DELAYS_SECONDS[
            min(attempt_count - 1, len(RETRY_DELAYS_SECONDS) - 1)
        ]
        retry_at = completed + delay
        current["next_retry_at"] = _utc(retry_at)
        current["next_retry_timestamp"] = retry_at
    _persist_mail_state(path, current)
    return current, True


def _deliver_nonblocking(
    path: Path,
    *,
    evidence: Mapping[str, Any],
    recipient: str,
    mail_runner: MailRunner,
    clock: Clock,
    maximum_attempts: int,
) -> dict[str, Any]:
    """Attempt at most one currently-eligible delivery without sleeping.

    The retry timestamp is durable scheduling state, not permission to consume a
    sentinel allocation while waiting.  A later monitor or idempotent invocation
    retries when eligible.  This keeps scheduler evidence and marker-last completion
    independent of email availability.
    """

    state = _load_or_create_mail_state(
        path,
        evidence=evidence,
        recipient=recipient,
        timestamp=float(clock()),
    )
    if (
        maximum_attempts <= 0
        or not state["requested"]
        or state["delivered"]
    ):
        return state
    now = float(clock())
    if not math.isfinite(now):
        raise SentinelError("mail clock returned a non-finite timestamp")
    next_retry = state["next_retry_timestamp"]
    in_flight = state["in_flight_token"]
    attempted = state["attempted_timestamp"]
    if isinstance(in_flight, str) and in_flight:
        if not isinstance(attempted, (int, float)):
            raise SentinelError("mail in-flight token has no attempt timestamp")
        eligible = float(attempted) + IN_FLIGHT_STALE_SECONDS
    else:
        eligible = now if next_retry is None else float(next_retry)
    if eligible > now:
        return state
    state, _attempted_now = _attempt_mail(
        path,
        evidence=evidence,
        recipient=recipient,
        mail_runner=mail_runner,
        clock=clock,
    )
    return state


def _build_marker(
    *,
    verified: Mapping[str, Any],
    evidence_path: Path,
    evidence: Mapping[str, Any],
    mail_path: Path,
    mail_state: Mapping[str, Any],
    timestamp: float,
) -> dict[str, Any]:
    outcome = evidence["outcome"]
    marker: dict[str, Any] = {
        "schema_version": 1,
        "protocol": MARKER_PROTOCOL,
        "passed": True,
        "chain_protocol": verified["chain_protocol"],
        "chain_id": evidence["chain_id"],
        "manifest": verified["manifest_path"],
        "manifest_sha256": verified["manifest_sha256"],
        "submission_receipt": verified["submission_receipt_path"],
        "submission_receipt_sha256": verified["submission_receipt_sha256"],
        "scheduler_evidence": str(evidence_path),
        "scheduler_evidence_sha256": _sha256(evidence_path),
        "scheduler_evidence_id": evidence["evidence_id"],
        "classification": outcome["classification"],
        "run_succeeded": outcome["run_succeeded"],
        "same_generation_repair_allowed": outcome[
            "same_generation_repair_allowed"
        ],
        "requires_superseding_release": outcome[
            "requires_superseding_release"
        ],
        "mail_state": str(mail_path),
        "mail_id": mail_state["mail_id"],
        "alert_requested": mail_state["requested"],
        "published_at": _utc(timestamp),
        "published_timestamp": timestamp,
    }
    marker["marker_id"] = _sha256_bytes(_canonical_json(marker))
    return marker


def _validate_marker(
    path: Path,
    *,
    verified: Mapping[str, Any],
    evidence_path: Path,
    evidence: Mapping[str, Any],
    mail_path: Path,
) -> dict[str, Any]:
    marker = _read_json(path, description="recovery sentinel completion marker")
    _require_read_only(path, description="recovery sentinel completion marker")
    identity = dict(marker)
    marker_id = identity.pop("marker_id", None)
    required = {
        "schema_version",
        "protocol",
        "passed",
        "chain_protocol",
        "chain_id",
        "manifest",
        "manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "scheduler_evidence",
        "scheduler_evidence_sha256",
        "scheduler_evidence_id",
        "classification",
        "run_succeeded",
        "same_generation_repair_allowed",
        "requires_superseding_release",
        "mail_state",
        "mail_id",
        "alert_requested",
        "published_at",
        "published_timestamp",
        "marker_id",
    }
    outcome = evidence["outcome"]
    mail_state = _read_json(
        mail_path, description="sentinel mail state"
    )
    _validate_mail_state(
        mail_state,
        evidence=evidence,
        recipient=str(mail_state.get("recipient", "")),
    )
    published_timestamp = marker.get("published_timestamp")
    if (
        set(marker) != required
        or marker["schema_version"] != 1
        or marker["protocol"] != MARKER_PROTOCOL
        or marker["passed"] is not True
        or marker["chain_protocol"] != R2_PROTOCOL
        or marker["chain_id"] != evidence["chain_id"]
        or marker["manifest"] != verified["manifest_path"]
        or marker["manifest_sha256"] != verified["manifest_sha256"]
        or marker["submission_receipt"] != verified["submission_receipt_path"]
        or marker["submission_receipt_sha256"]
        != verified["submission_receipt_sha256"]
        or marker["scheduler_evidence"] != str(evidence_path)
        or marker["scheduler_evidence_sha256"] != _sha256(evidence_path)
        or marker["scheduler_evidence_id"] != evidence["evidence_id"]
        or marker["classification"] != outcome["classification"]
        or marker["run_succeeded"] is not outcome["run_succeeded"]
        or marker["same_generation_repair_allowed"]
        is not outcome["same_generation_repair_allowed"]
        or marker["requires_superseding_release"]
        is not outcome["requires_superseding_release"]
        or marker["mail_state"] != str(mail_path)
        or marker["mail_id"]
        != _mail_identity(evidence, mail_state["recipient"])
        or marker["alert_requested"] is not mail_state["requested"]
        or not isinstance(published_timestamp, (int, float))
        or isinstance(published_timestamp, bool)
        or not math.isfinite(float(published_timestamp))
        or published_timestamp < 0
        or marker.get("published_at")
        != _utc(float(published_timestamp))
        or not isinstance(marker_id, str)
        or marker_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise SentinelError("recovery sentinel completion marker is invalid")
    return marker


def _build_stage_marker(
    *,
    verified: Mapping[str, Any],
    evidence_path: Path,
    evidence: Mapping[str, Any],
    mail_path: Path,
    mail_state: Mapping[str, Any],
    timestamp: float,
) -> dict[str, Any]:
    outcome = evidence["outcome"]
    marker: dict[str, Any] = {
        "schema_version": 1,
        "protocol": STAGE_MARKER_PROTOCOL,
        "passed": True,
        "chain_protocol": verified["chain_protocol"],
        "chain_id": evidence["chain_id"],
        "manifest": verified["manifest_path"],
        "manifest_sha256": verified["manifest_sha256"],
        "submission_receipt": verified["submission_receipt_path"],
        "submission_receipt_sha256": verified["submission_receipt_sha256"],
        "scheduler_evidence": str(evidence_path),
        "scheduler_evidence_sha256": _sha256(evidence_path),
        "scheduler_evidence_id": evidence["evidence_id"],
        "target_stage": outcome["target_stage"],
        "target_job_id": outcome["target_job_id"],
        "stage_sentinel_name": outcome["stage_sentinel_name"],
        "stage_sentinel_job_id": outcome["stage_sentinel_job_id"],
        "classification": outcome["classification"],
        "run_succeeded": outcome["run_succeeded"],
        "authoritative_repair_classification": False,
        "mail_state": str(mail_path),
        "mail_id": mail_state["mail_id"],
        "alert_requested": mail_state["requested"],
        "published_at": _utc(timestamp),
        "published_timestamp": timestamp,
    }
    marker["marker_id"] = _sha256_bytes(_canonical_json(marker))
    return marker


def _validate_stage_marker(
    path: Path,
    *,
    verified: Mapping[str, Any],
    evidence_path: Path,
    evidence: Mapping[str, Any],
    mail_path: Path,
) -> dict[str, Any]:
    marker = _read_json(path, description="stage sentinel completion marker")
    _require_read_only(path, description="stage sentinel completion marker")
    identity = dict(marker)
    marker_id = identity.pop("marker_id", None)
    outcome = evidence["outcome"]
    mail_state = _read_json(mail_path, description="stage sentinel mail state")
    _validate_mail_state(
        mail_state,
        evidence=evidence,
        recipient=str(mail_state.get("recipient", "")),
    )
    published_timestamp = marker.get("published_timestamp")
    required = {
        "schema_version",
        "protocol",
        "passed",
        "chain_protocol",
        "chain_id",
        "manifest",
        "manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "scheduler_evidence",
        "scheduler_evidence_sha256",
        "scheduler_evidence_id",
        "target_stage",
        "target_job_id",
        "stage_sentinel_name",
        "stage_sentinel_job_id",
        "classification",
        "run_succeeded",
        "authoritative_repair_classification",
        "mail_state",
        "mail_id",
        "alert_requested",
        "published_at",
        "published_timestamp",
        "marker_id",
    }
    if (
        set(marker) != required
        or marker.get("schema_version") != 1
        or marker.get("protocol") != STAGE_MARKER_PROTOCOL
        or marker.get("passed") is not True
        or marker.get("chain_protocol") != R2_PROTOCOL
        or marker.get("chain_id") != evidence["chain_id"]
        or marker.get("manifest") != verified["manifest_path"]
        or marker.get("manifest_sha256") != verified["manifest_sha256"]
        or marker.get("submission_receipt")
        != verified["submission_receipt_path"]
        or marker.get("submission_receipt_sha256")
        != verified["submission_receipt_sha256"]
        or marker.get("scheduler_evidence") != str(evidence_path)
        or marker.get("scheduler_evidence_sha256") != _sha256(evidence_path)
        or marker.get("scheduler_evidence_id") != evidence["evidence_id"]
        or marker.get("target_stage") != outcome["target_stage"]
        or marker.get("target_job_id") != outcome["target_job_id"]
        or marker.get("stage_sentinel_name")
        != outcome["stage_sentinel_name"]
        or marker.get("stage_sentinel_job_id")
        != outcome["stage_sentinel_job_id"]
        or marker.get("classification") != outcome["classification"]
        or marker.get("run_succeeded") is not outcome["run_succeeded"]
        or marker.get("authoritative_repair_classification") is not False
        or marker.get("mail_state") != str(mail_path)
        or marker.get("mail_id")
        != _mail_identity(evidence, mail_state["recipient"])
        or marker.get("alert_requested") is not mail_state["requested"]
        or not isinstance(published_timestamp, (int, float))
        or isinstance(published_timestamp, bool)
        or not math.isfinite(float(published_timestamp))
        or float(published_timestamp) < 0
        or marker.get("published_at") != _utc(float(published_timestamp))
        or not isinstance(marker_id, str)
        or marker_id != _sha256_bytes(_canonical_json(identity))
    ):
        raise SentinelError("stage sentinel completion marker is invalid")
    return marker


def _verified_inputs(
    chain_manifest: Path, submission_receipt: Path
) -> dict[str, Any]:
    try:
        verified = verify_recovery_evidence(chain_manifest, submission_receipt)
    except EvidenceVerificationError as exc:
        raise SentinelError(str(exc)) from exc
    if verified["chain_protocol"] != R2_PROTOCOL:
        raise SentinelError(
            "the afterany recovery sentinel accepts only schema5-v1.2-r2 evidence"
        )
    return verified


def run_stage_sentinel(
    *,
    chain_manifest: str | Path,
    submission_receipt: str | Path,
    output_root: str | Path,
    recipient: str,
    stage_name: str,
    stage_sentinel_job_id: str,
    apply: bool = False,
    runner: Runner | None = None,
    mail_runner: MailRunner | None = None,
    clock: Clock | None = None,
    sleeper: Sleeper | None = None,
    maximum_mail_attempts: int = SYNCHRONOUS_MAIL_ATTEMPT_LIMIT,
) -> dict[str, Any]:
    """Immediately record and alert on one exact stage's terminal state."""

    if not recipient or any(character in recipient for character in "\r\n"):
        raise SentinelError("mail recipient is empty or unsafe")
    if stage_name not in PRODUCTION_STAGE_NAMES:
        raise SentinelError(f"unknown fail-fast stage target: {stage_name!r}")
    if not stage_sentinel_job_id.isdigit():
        raise SentinelError("stage-sentinel job ID must be numeric")
    if maximum_mail_attempts < 0:
        raise SentinelError("maximum_mail_attempts cannot be negative")
    now = time.time if clock is None else clock
    # Kept as a compatibility argument for callers/tests. Sentinel publication never
    # sleeps for a future email retry.
    _ = sleeper
    send_mail = _default_mail_runner if mail_runner is None else mail_runner
    manifest_path = _absolute(chain_manifest)
    receipt_path = _absolute(submission_receipt)
    verified = _verified_inputs(manifest_path, receipt_path)

    def observe() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        jobs, query = reconcile_receipt_jobs(
            manifest=verified["manifest"],
            receipt=verified["submission_receipt"],
            runner=runner,
        )
        outcome = classify_stage_observation(
            jobs,
            stage_name=stage_name,
            stage_sentinel_job_id=stage_sentinel_job_id,
        )
        return jobs, query, outcome

    if not apply:
        _jobs, query, outcome = observe()
        return {
            "status": "dry_run",
            "passed": True,
            "writes_performed": False,
            "mail_attempted": False,
            "chain_id": verified["manifest"]["chain_id"],
            "target_stage": stage_name,
            "classification": outcome["classification"],
            "outcome": outcome,
            "scheduler_queries": query,
        }

    root = _prepare_output_root(_absolute(output_root))
    evidence_path = root / STAGE_SCHEDULER_EVIDENCE_NAME
    mail_path = root / MAIL_STATE_NAME
    marker_path = root / STAGE_COMPLETE_MARKER_NAME
    with _sentinel_lock(root):
        if evidence_path.exists() or evidence_path.is_symlink():
            evidence = _validate_stage_scheduler_evidence(
                evidence_path,
                verified=verified,
                stage_name=stage_name,
                stage_sentinel_job_id=stage_sentinel_job_id,
            )
        else:
            jobs, query, outcome = observe()
            timestamp = float(now())
            if not math.isfinite(timestamp):
                raise SentinelError(
                    "stage sentinel clock returned a non-finite timestamp"
                )
            evidence = _stage_scheduler_evidence(
                verified=verified,
                jobs=jobs,
                query=query,
                outcome=outcome,
                timestamp=timestamp,
            )
            _atomic_json(evidence_path, evidence, mode=0o444)
            evidence = _validate_stage_scheduler_evidence(
                evidence_path,
                verified=verified,
                stage_name=stage_name,
                stage_sentinel_job_id=stage_sentinel_job_id,
            )

        mail_state = _deliver_nonblocking(
            mail_path,
            evidence=evidence,
            recipient=recipient,
            mail_runner=send_mail,
            clock=now,
            maximum_attempts=min(
                maximum_mail_attempts,
                SYNCHRONOUS_MAIL_ATTEMPT_LIMIT,
            ),
        )
        if marker_path.exists() or marker_path.is_symlink():
            marker = _validate_stage_marker(
                marker_path,
                verified=verified,
                evidence_path=evidence_path,
                evidence=evidence,
                mail_path=mail_path,
            )
            status = "already_complete"
        else:
            timestamp = float(now())
            marker = _build_stage_marker(
                verified=verified,
                evidence_path=evidence_path,
                evidence=evidence,
                mail_path=mail_path,
                mail_state=mail_state,
                timestamp=timestamp,
            )
            _atomic_json(marker_path, marker, mode=0o444)
            marker = _validate_stage_marker(
                marker_path,
                verified=verified,
                evidence_path=evidence_path,
                evidence=evidence,
                mail_path=mail_path,
            )
            status = "complete"
    return {
        "status": status,
        "passed": True,
        "chain_id": marker["chain_id"],
        "target_stage": marker["target_stage"],
        "classification": marker["classification"],
        "run_succeeded": marker["run_succeeded"],
        "authoritative_repair_classification": False,
        "scheduler_evidence": str(evidence_path),
        "completion_marker": str(marker_path),
        "alert_requested": mail_state["requested"],
        "alert_delivered": mail_state["delivered"],
        "alert_retry_pending": (
            mail_state["requested"] and not mail_state["delivered"]
        ),
        "mail_next_retry_timestamp": mail_state["next_retry_timestamp"],
        "mail_attempt_count": mail_state["attempt_count"],
    }


def run_sentinel(
    *,
    chain_manifest: str | Path,
    submission_receipt: str | Path,
    output_root: str | Path,
    recipient: str,
    capacity_transient_receipt: str | Path | None = None,
    apply: bool = False,
    runner: Runner | None = None,
    mail_runner: MailRunner | None = None,
    clock: Clock | None = None,
    sleeper: Sleeper | None = None,
    maximum_mail_attempts: int = SYNCHRONOUS_MAIL_ATTEMPT_LIMIT,
) -> dict[str, Any]:
    """Evaluate or durably publish one exact v1.2-r2 recovery-chain outcome."""

    if not recipient or any(character in recipient for character in "\r\n"):
        raise SentinelError("mail recipient is empty or unsafe")
    if maximum_mail_attempts < 0:
        raise SentinelError("maximum_mail_attempts cannot be negative")
    now = time.time if clock is None else clock
    # Kept as a compatibility argument for callers/tests. Sentinel publication never
    # sleeps for a future email retry.
    _ = sleeper
    send_mail = _default_mail_runner if mail_runner is None else mail_runner
    manifest_path = _absolute(chain_manifest)
    receipt_path = _absolute(submission_receipt)
    capacity_receipt_path = (
        None
        if capacity_transient_receipt is None
        else _absolute(capacity_transient_receipt)
    )
    verified = _verified_inputs(manifest_path, receipt_path)

    if not apply:
        jobs, query = reconcile_receipt_jobs(
            manifest=verified["manifest"],
            receipt=verified["submission_receipt"],
            runner=runner,
        )
        capacity_binding, capacity_corruption = (
            _observe_capacity_transient_receipt(
            capacity_receipt_path,
            verified=verified,
            jobs=jobs,
            )
        )
        outcome = _bind_capacity_corruption(
            classify_recovery_jobs(
                jobs,
                capacity_transient_receipt=capacity_binding,
                qualification_capacity_failure=(
                    _observe_qualification_capacity_failure(
                        verified=verified,
                        jobs=jobs,
                    )
                ),
            ),
            capacity_corruption,
        )
        return {
            "status": "dry_run",
            "passed": True,
            "writes_performed": False,
            "mail_attempted": False,
            "chain_id": verified["manifest"]["chain_id"],
            "classification": outcome["classification"],
            "outcome": outcome,
            "scheduler_queries": query,
        }

    root = _prepare_output_root(_absolute(output_root))
    evidence_path = root / SCHEDULER_EVIDENCE_NAME
    mail_path = root / MAIL_STATE_NAME
    marker_path = root / COMPLETE_MARKER_NAME
    with _sentinel_lock(root):
        if evidence_path.exists() or evidence_path.is_symlink():
            evidence = _validate_scheduler_evidence(
                evidence_path,
                verified=verified,
                capacity_transient_receipt_path=capacity_receipt_path,
            )
        else:
            jobs, query = reconcile_receipt_jobs(
                manifest=verified["manifest"],
                receipt=verified["submission_receipt"],
                runner=runner,
            )
            capacity_binding, capacity_corruption = (
                _observe_capacity_transient_receipt(
                capacity_receipt_path,
                verified=verified,
                jobs=jobs,
                )
            )
            outcome = _bind_capacity_corruption(
                classify_recovery_jobs(
                    jobs,
                    capacity_transient_receipt=capacity_binding,
                    qualification_capacity_failure=(
                        _observe_qualification_capacity_failure(
                            verified=verified,
                            jobs=jobs,
                        )
                    ),
                ),
                capacity_corruption,
            )
            timestamp = float(now())
            if not math.isfinite(timestamp):
                raise SentinelError("sentinel clock returned a non-finite timestamp")
            evidence = _scheduler_evidence(
                verified=verified,
                jobs=jobs,
                query=query,
                classification=outcome,
                capacity_transient_receipt=capacity_binding,
                qualification_capacity_failure=(
                    _observe_qualification_capacity_failure(
                        verified=verified,
                        jobs=jobs,
                    )
                ),
                timestamp=timestamp,
            )
            _atomic_json(evidence_path, evidence, mode=0o444)
            evidence = _validate_scheduler_evidence(
                evidence_path,
                verified=verified,
                capacity_transient_receipt_path=capacity_receipt_path,
            )

        mail_state = _deliver_nonblocking(
            mail_path,
            evidence=evidence,
            recipient=recipient,
            mail_runner=send_mail,
            clock=now,
            maximum_attempts=min(
                maximum_mail_attempts,
                SYNCHRONOUS_MAIL_ATTEMPT_LIMIT,
            ),
        )
        if marker_path.exists() or marker_path.is_symlink():
            marker = _validate_marker(
                marker_path,
                verified=verified,
                evidence_path=evidence_path,
                evidence=evidence,
                mail_path=mail_path,
            )
            status = "already_complete"
        else:
            timestamp = float(now())
            marker = _build_marker(
                verified=verified,
                evidence_path=evidence_path,
                evidence=evidence,
                mail_path=mail_path,
                mail_state=mail_state,
                timestamp=timestamp,
            )
            _atomic_json(marker_path, marker, mode=0o444)
            marker = _validate_marker(
                marker_path,
                verified=verified,
                evidence_path=evidence_path,
                evidence=evidence,
                mail_path=mail_path,
            )
            status = "complete"
    return {
        "status": status,
        "passed": True,
        "chain_id": marker["chain_id"],
        "classification": marker["classification"],
        "run_succeeded": marker["run_succeeded"],
        "same_generation_repair_allowed": marker[
            "same_generation_repair_allowed"
        ],
        "requires_superseding_release": marker[
            "requires_superseding_release"
        ],
        "scheduler_evidence": str(evidence_path),
        "completion_marker": str(marker_path),
        "alert_requested": mail_state["requested"],
        "alert_delivered": mail_state["delivered"],
        "alert_retry_pending": (
            mail_state["requested"] and not mail_state["delivered"]
        ),
        "mail_next_retry_timestamp": mail_state["next_retry_timestamp"],
        "mail_attempt_count": mail_state["attempt_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the exact schema-5 v1.2-r2 recovery-chain receipt and publish "
            "marker-last immutable outcome evidence."
        )
    )
    parser.add_argument("--chain-manifest", type=Path, required=True)
    parser.add_argument("--submission-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recipient", required=True)
    parser.add_argument(
        "--stage-name",
        choices=PRODUCTION_STAGE_NAMES,
        help=(
            "Run as the fail-fast observer for this exact production stage. "
            "This mode records/alerts but grants no repair authority."
        ),
    )
    parser.add_argument(
        "--stage-sentinel-job-id",
        help="Exact running Slurm allocation ID for --stage-name mode.",
    )
    parser.add_argument(
        "--capacity-transient-receipt",
        type=Path,
        help=(
            "Generation/job-scoped marker-last capacity receipt. Its absence is "
            "accepted; a fleet-readiness FAILED 75:0 is repairable only when the "
            "exact valid marker exists."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish immutable evidence and attempt persisted alert delivery.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if (args.stage_name is None) != (
            args.stage_sentinel_job_id is None
        ):
            raise SentinelError(
                "--stage-name and --stage-sentinel-job-id must be supplied together"
            )
        if args.stage_name is not None:
            if args.capacity_transient_receipt is not None:
                raise SentinelError(
                    "stage-sentinel mode cannot accept a capacity-transient receipt"
                )
            report = run_stage_sentinel(
                chain_manifest=args.chain_manifest,
                submission_receipt=args.submission_receipt,
                output_root=args.output_root,
                recipient=args.recipient,
                stage_name=args.stage_name,
                stage_sentinel_job_id=args.stage_sentinel_job_id,
                apply=args.apply,
            )
        else:
            report = run_sentinel(
                chain_manifest=args.chain_manifest,
                submission_receipt=args.submission_receipt,
                output_root=args.output_root,
                recipient=args.recipient,
                capacity_transient_receipt=args.capacity_transient_receipt,
                apply=args.apply,
            )
    except SentinelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    # A classified recovery failure is a successful sentinel observation.  Returning
    # zero prevents the sentinel itself from fabricating a deterministic FAILED state.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
