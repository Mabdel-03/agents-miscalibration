#!/usr/bin/env python3
"""Durable desired-state control for the homogeneous schema-5 sweep.

This module deliberately sits *above* the experiment dispatcher and fleet manager.  It
does not select cells or servers.  Instead it provides the small, fail-closed control
plane needed to prove which immutable release is allowed to do so, keep exactly one
dispatcher and one fleet supervisor alive, and make every external scheduler boundary
recoverable after a crash.

Operator commands are idempotent::

    schema5_control.py --state-dir ... prepare-pins --release-bundle-root ... \
        --hf-home ... --output ...
    schema5_control.py --state-dir ... init --pins-json release-pins.json
    schema5_control.py --state-dir ... reconcile --all --no-admit
    schema5_control.py --state-dir ... resume
    schema5_control.py --state-dir ... pause --drain
    schema5_control.py --state-dir ... status --live
    schema5_control.py --state-dir ... repair-chain

``resume`` is a transactional desired-state transition.  It first persists a
``resuming`` intent, submits or adopts both exact controller jobs, proves both are
visible in squeue+sacct, and only then publishes ``running``.  A controller that starts
early claims its fenced identity and waits while the transition remains ``resuming``. A
running controller invokes the same reconciler for its peer after a ten-minute
stale-heartbeat threshold.  All submissions are preceded by a durable ``submitting``
intent containing a scheduler-searchable token; a restart adopts the unique matching
job rather than resubmitting.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence, TextIO

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.experiment import io
from agents_scaling import runtime_integrity, snapshot_integrity

CONTROL_SCHEMA_VERSION = 1
CONTROL_PROTOCOL = "schema5-v1"
CONTROL_FILENAME = "control.json"
IMMUTABLE_PINS_FILENAME = "immutable_pins.json"
TRANSITION_JOURNAL = "transitions.jsonl"
ALERT_JOURNAL = "alerts.jsonl"
RECONCILIATION_FILENAME = "reconciliation.json"
DRILL_STATE_FILENAME = "controller_drill.json"
DRILL_JOURNAL_FILENAME = "controller_drill.jsonl"
DRILL_COMPLETE_FILENAME = "CONTROLLER_KILL_DRILL_COMPLETE.json"
DRILL_RECONCILIATION_FILENAME = "FINAL_RECONCILIATION.json"
ROLE_NAMES = ("dispatcher", "fleet_supervisor")
ROLE_SHORT = {"dispatcher": "dispatch", "fleet_supervisor": "fleet"}
REQUIRED_RUNS = {
    "full_sweep_schema5_v1": 4_680,
    "full_sweep_agent_counts_schema5_v1": 14_400,
    "full_sweep_agent_count_7_schema5_v1": 3_600,
}
REQUIRED_RUN_QIDS = {
    "full_sweep_schema5_v1": 933_660,
    "full_sweep_agent_counts_schema5_v1": 2_872_800,
    "full_sweep_agent_count_7_schema5_v1": 718_200,
}
EXPECTED_TOTAL_CELLS = 22_680
EXPECTED_TOTAL_QIDS = 4_524_660
RUN_WEIGHT = 1.0
CONTROL_STATE_DIRNAME = ".dispatcher-schema5-v1"
RELEASE_COMPLETE_FILENAME = "RELEASE_COMPLETE.json"
MATERIALIZATION_COMPLETE_FILENAME = "MATERIALIZATION_COMPLETE.json"
MATERIALIZATION_STAGE_FILENAMES = {
    "worktree": "WORKTREE_MATERIALIZED.json",
    "harness_clone": "HARNESS_CLONE_COMPLETE.json",
    "serving_clone": "SERVING_CLONE_COMPLETE.json",
    "harness_package": "HARNESS_PACKAGE_COMPLETE.json",
}
RUNTIME_ATTESTATION_STATE_KEY = "runtime_integrity"
SNAPSHOT_ATTESTATION_STATE_KEY = "snapshot_integrity"
RELEASE_IDENTITY_FILENAME = "release_identity.schema5-v1.json"
HARNESS_ENVIRONMENT_FILENAME = "harness_environment.schema5-v1.json"
SERVING_ENVIRONMENT_FILENAME = "serving_environment.schema5-v1.json"
REQUIRED_GATES = (
    "snapshot",
    "migrations",
    "semantic_audit",
    "fleet",
    "context_audit",
    "smoke_runs",
    "email_test",
    "scheduler_reconciliation",
)
HEARTBEAT_INTERVAL_SECONDS = 60.0
STALE_HEARTBEAT_SECONDS = 600.0
SUBMISSION_VISIBILITY_GRACE_SECONDS = 180.0
ACTIVE_SCHEDULER_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "RESIZING",
    "SUSPENDED",
}
TERMINAL_SCHEDULER_STATES = {
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
CELL_JOB_PREFIXES = ("asys-cells", "asys-dispatch-")
TOKEN_PREFIX = "asys-schema5-v1"
DRILL_TOKEN_PREFIX = "asys-schema5-drill-v1"
SHARED_CONTROLLER_FENCING_PRIMITIVE = "exact-role-intent-claim-v1"
SHARED_CONTROLLER_SUBMISSION_PRIMITIVE = "durable-intent-sbatch-afterany-v1"
SHARED_CONTROLLER_SINGLETON_PRIMITIVE = "cross-node-role-flock-v1"
CELL_INTENT_PREFIX = "asys-schema5-intent:"
_EXACT_CELL_TASK_ID = re.compile(r"[0-9]+(?:_[0-9]+)?\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

DEFAULT_ADMISSION = {
    "qos_limit": 448,
    "reserve": 64,
    "maximum_cell_tasks": 384,
    "current_ceiling": 24,
    "max_batch": 24,
    "cell_cpus": 1,
    "cell_memory": "4G",
    "effective_cell_time": "12:00:00",
    "signal": "B:USR1@1200",
    "no_requeue": True,
}

ADMISSION_RAMP_SCHEMA_VERSION = 1
ADMISSION_RAMP_PROTOCOL = "schema5-admission-ramp-v1"
ADMISSION_RAMP_EVIDENCE_PROTOCOL = "schema5-admission-ramp-observation-v1"
ADMISSION_RAMP_STAGES = (24, 96, 192, 384)
# A stage's timer starts only from a semantically complete observation containing an
# exact validated-QID total for every frozen production run.  The five-minute health
# monitor then proves that the interval remained continuously healthy.
ADMISSION_RAMP_REQUIREMENTS = {
    24: {"next_ceiling": 96, "clean_seconds": 3_600.0},
    96: {"next_ceiling": 192, "clean_seconds": 21_600.0},
    192: {"next_ceiling": 384, "clean_seconds": 43_200.0},
}
ADMISSION_RAMP_MAX_OBSERVATION_GAP_SECONDS = 660.0
ADMISSION_RAMP_BLOCKING_ALERT_KEYS = frozenset(
    {"monitor:qos-memory", "monitor:starvation"}
)

# Readiness is intentionally a closed, versioned contract.  These are scientific and
# operational facts that must be independently checksummed; a generic
# ``{"passed": true}`` assertion is never sufficient to start production.
READINESS_EVIDENCE_SCHEMA_VERSION = 2
READINESS_ARTIFACT_NAMES: dict[str, tuple[str, ...]] = {
    "snapshot": (
        "pre_repair_external_attestation",
        "legacy_consolidated_external_attestation",
    ),
    "migrations": (
        "response_incident_archive_report",
        "checkpoint_migration_report",
        "permanent_ledger_archive_report",
    ),
    "semantic_audit": ("legacy_semantic_audit_report",),
    "fleet": ("fleet_health_report", "fleet_contract"),
    "context_audit": ("dense_peer_context_audit", "seven_agent_context_audit"),
    "smoke_runs": (
        "long_32b_smoke",
        "selective_long_smoke",
        "standard_canary_smoke",
    ),
    "email_test": ("email_delivery_receipt",),
}
EXPECTED_FLEET_PROFILES = {
    "0.6B": 2,
    "1.7B": 2,
    "4B": 2,
    "8B": 3,
    "14B": 2,
    "32B": 4,
    "0.6B-long": 1,
    "1.7B-long": 1,
    "4B-long": 1,
    "8B-long": 1,
    "14B-long": 1,
    "32B-long": 2,
}
SNAPSHOT_CONTROL_FILENAMES = {
    "SNAPSHOT_COMPLETE.json",
    "SNAPSHOT_CATALOG.json",
    "SOURCE_INVENTORY.sha256",
    "SNAPSHOT_INVENTORY.sha256",
    "DIRECTORY_INVENTORY.txt",
}

# Monitoring is deliberately owned by the already-supervised dispatcher controller.
# It is not a third Slurm chain.  The durable timestamps below make the three cadences
# survive ordinary 12-hour controller successors without resetting their clocks.
MONITOR_OWNER_ROLE = "dispatcher"
MONITOR_CADENCE_SECONDS = {
    "health": 300,
    "semantic": 21_600,
    "daily": 86_400,
}
MONITOR_RETRY_SECONDS = 60.0
MONITOR_SUCCESS_RETURN_CODES = frozenset({0, 2})
MONITOR_FAILURE_ALERT_PREFIX = "monitor-supervisor"


class ControlError(RuntimeError):
    """The schema-5 control plane cannot safely perform the requested transition."""


class ImmutablePinError(ControlError):
    """An immutable release, run, environment, model, or fleet pin drifted."""


class ReadinessError(ControlError):
    """Production cannot resume because at least one readiness gate is not valid."""


class SchedulerAmbiguity(ControlError):
    """Scheduler truth cannot be mapped to one unique durable intent."""


class SchedulerVisibilityPending(ControlError):
    """A recently submitted exact intent is still inside scheduler visibility grace."""


class ControllerFenced(ControlError):
    """A controller job does not own the exact durable role generation."""


@dataclass(frozen=True)
class _SnapshotLeaseRefreshTask:
    """One local asynchronous renewal; the cross-node lock remains authoritative."""

    future: Future[dict[str, Any]]
    thread: threading.Thread


@dataclass(frozen=True)
class _RuntimeLeaseRefreshTask:
    """One local asynchronous renewal; the cross-node lock remains authoritative."""

    future: Future[dict[str, Any]]
    thread: threading.Thread


@dataclass(frozen=True)
class SchedulerJob:
    job_id: str
    job_name: str
    state: str
    comment: str = ""
    command: str = ""
    source: str = "squeue"
    dependency: str = ""

    @property
    def active(self) -> bool:
        return normalize_scheduler_state(self.state) in ACTIVE_SCHEDULER_STATES

    @property
    def token(self) -> str | None:
        return self.comment if self.comment.startswith(TOKEN_PREFIX + ";") else None


@dataclass(frozen=True)
class SchedulerSnapshot:
    jobs: tuple[SchedulerJob, ...]
    captured_at: float
    squeue_ok: bool = True
    sacct_ok: bool = True
    errors: tuple[str, ...] = ()

    @property
    def active_job_ids(self) -> set[str]:
        return {job.job_id for job in self.jobs if job.active}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash an immutable release tree, excluding only VCS and runtime cache metadata."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ImmutablePinError(f"cannot hash missing release tree: {root}")
    ignored_names = {".git", ".pytest_cache", "__pycache__"}
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if not any(part in ignored_names for part in path.relative_to(root).parts)
            and path.suffix != ".pyc"
            and (path.is_file() or path.is_symlink())
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if path.is_symlink():
            payload = ("SYMLINK\0" + os.readlink(path)).encode("utf-8")
        else:
            payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def utc_timestamp(now: float | None = None) -> str:
    value = time.time() if now is None else float(now)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def normalize_scheduler_state(value: str) -> str:
    # sacct may report values such as CANCELLED+ or FAILED by 0:0.
    return value.strip().upper().split()[0].rstrip("+")


def job_token(role: str, generation: int, intent_token: str) -> str:
    _validate_role(role)
    if generation < 1:
        raise ValueError("controller generation must be positive")
    if not intent_token or any(char in intent_token for char in ";|\n\r"):
        raise ValueError("intent token must be non-empty and scheduler-safe")
    return f"{TOKEN_PREFIX};role={role};generation={generation};intent={intent_token}"


def drill_job_token(
    drill_id: str, role: str, generation: int, intent_token: str
) -> str:
    """Return a scheduler-searchable token outside the production token namespace."""

    _validate_role(role)
    if generation < 1:
        raise ValueError("drill generation must be positive")
    for label, value in (("drill_id", drill_id), ("intent_token", intent_token)):
        if not value or any(char in value for char in ";|\n\r"):
            raise ValueError(f"{label} must be non-empty and scheduler-safe")
    return (
        f"{DRILL_TOKEN_PREFIX};drill={drill_id};role={role};"
        f"generation={generation};intent={intent_token}"
    )


def parse_drill_job_token(value: str) -> dict[str, str] | None:
    if not value.startswith(DRILL_TOKEN_PREFIX + ";"):
        return None
    fields: dict[str, str] = {}
    for item in value.split(";")[1:]:
        if "=" not in item:
            return None
        key, field_value = item.split("=", 1)
        if not key or not field_value or key in fields:
            return None
        fields[key] = field_value
    if set(fields) != {"drill", "role", "generation", "intent"}:
        return None
    if fields["role"] not in ROLE_NAMES:
        return None
    try:
        if int(fields["generation"]) < 1:
            return None
    except ValueError:
        return None
    return fields


def parse_job_token(value: str) -> dict[str, str] | None:
    if not value.startswith(TOKEN_PREFIX + ";"):
        return None
    fields: dict[str, str] = {}
    for item in value.split(";")[1:]:
        if "=" not in item:
            return None
        key, field_value = item.split("=", 1)
        if not key or not field_value or key in fields:
            return None
        fields[key] = field_value
    if set(fields) != {"role", "generation", "intent"}:
        return None
    if fields["role"] not in ROLE_NAMES:
        return None
    try:
        if int(fields["generation"]) < 1:
            return None
    except ValueError:
        return None
    return fields


def _validate_role(role: str) -> None:
    if role not in ROLE_NAMES:
        raise ControlError(f"unknown controller role {role!r}; expected {ROLE_NAMES}")


def shared_controller_primitive_contract() -> dict[str, str]:
    """Return the exact production primitives exercised by isolated drill jobs."""

    return {
        "fencing": SHARED_CONTROLLER_FENCING_PRIMITIVE,
        "submission": SHARED_CONTROLLER_SUBMISSION_PRIMITIVE,
        "singleton": SHARED_CONTROLLER_SINGLETON_PRIMITIVE,
    }


def _state_path(state_dir: Path) -> Path:
    return state_dir / CONTROL_FILENAME


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    io.atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


@contextmanager
def _file_lock(path: Path, *, nonblocking: bool = False) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        try:
            fcntl.flock(fd, flags)
        except BlockingIOError as exc:
            raise ControlError(
                f"cross-node singleton lock is already held: {path}"
            ) from exc
        os.ftruncate(fd, 0)
        os.write(
            fd,
            (
                f"host={socket.gethostname()} pid={os.getpid()} "
                f"started_at={time.time():.6f}\n"
            ).encode("utf-8"),
        )
        os.fsync(fd)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def control_lock(state_dir: Path) -> Iterator[None]:
    with _file_lock(state_dir / "locks" / "control.lock"):
        yield


@contextmanager
def role_singleton_lock(state_dir: Path, role: str) -> Iterator[None]:
    _validate_role(role)
    with _file_lock(state_dir / "locks" / f"{role}.lock", nonblocking=True):
        yield


@contextmanager
def admission_boundary_lock(state_dir: Path) -> Iterator[None]:
    """Serialize the final cell ``sbatch`` boundary with ``pause --drain``.

    The dispatcher performs a fresh desired-state check while owning this lock.  Pause
    owns the same lock before its fresh squeue snapshot, so no array can be admitted in
    the otherwise-unobservable interval between snapshot and drain intent publication.
    """

    with _file_lock(state_dir / "locks" / "admission-boundary.lock"):
        yield


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    io.append_jsonl(path, dict(record))


def _read_jsonl_locked(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        if missing_ok:
            return []
        raise
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        with os.fdopen(os.dup(fd), "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    if not all(isinstance(record, dict) for record in records):
        raise ControlError(f"JSONL journal contains a non-object record: {path}")
    return records


def _history_record(
    control: Mapping[str, Any],
    *,
    event: str,
    details: Mapping[str, Any] | None,
    now: float,
) -> dict[str, Any]:
    history = list(control.get("transition_history", []))
    previous_hash = history[-1]["hash"] if history else None
    payload: dict[str, Any] = {
        "sequence": len(history) + 1,
        "at": utc_timestamp(now),
        "timestamp": now,
        "event": event,
        "details": copy.deepcopy(dict(details or {})),
        "previous_hash": previous_hash,
    }
    payload["hash"] = sha256_value(payload)
    return payload


def append_transition(
    state_dir: Path,
    control: MutableMapping[str, Any],
    *,
    event: str,
    details: Mapping[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else float(now)
    durable_history = _read_jsonl_locked(
        state_dir / TRANSITION_JOURNAL, missing_ok=True
    )
    in_memory_history = list(control.get("transition_history", []))
    if durable_history[: len(in_memory_history)] != in_memory_history:
        raise ControlError("transition journal diverged from in-memory control history")
    if len(durable_history) > len(in_memory_history):
        # A process can die after the journal append but before control.json replacement.
        # Preserve that sealed event and continue its hash chain.  The state mutation is
        # still applied idempotently by the retrying operator command.
        control["transition_history"] = copy.deepcopy(durable_history)
        validate_transition_history(control)
    record = _history_record(control, event=event, details=details, now=timestamp)
    control.setdefault("transition_history", []).append(record)
    # The separately fsynced journal makes the append-only audit trail recoverable even
    # if a future operator accidentally replaces control.json with an older copy.
    _append_jsonl(state_dir / TRANSITION_JOURNAL, record)
    return record


def validate_transition_history(control: Mapping[str, Any]) -> None:
    previous_hash: str | None = None
    history = control.get("transition_history")
    if not isinstance(history, list) or not history:
        raise ControlError("control transition history is absent")
    for expected_sequence, record in enumerate(history, start=1):
        if not isinstance(record, dict):
            raise ControlError("control transition history contains a non-object")
        candidate = dict(record)
        claimed_hash = candidate.pop("hash", None)
        if candidate.get("sequence") != expected_sequence:
            raise ControlError("control transition history sequence is not contiguous")
        if candidate.get("previous_hash") != previous_hash:
            raise ControlError("control transition history hash chain is broken")
        if claimed_hash != sha256_value(candidate):
            raise ControlError("control transition history record hash is invalid")
        previous_hash = claimed_hash


def _validate_admission(admission: Mapping[str, Any]) -> None:
    fixed = {
        "qos_limit": 448,
        "reserve": 64,
        "maximum_cell_tasks": 384,
        "max_batch": 24,
        "cell_cpus": 1,
        "cell_memory": "4G",
        "effective_cell_time": "12:00:00",
        "signal": "B:USR1@1200",
        "no_requeue": True,
    }
    for key, expected in fixed.items():
        if admission.get(key) != expected:
            raise ImmutablePinError(
                f"unsafe schema-5 admission setting {key}: "
                f"expected {expected!r}, found {admission.get(key)!r}"
            )
    ceiling = admission.get("current_ceiling")
    if not isinstance(ceiling, int) or ceiling not in {24, 96, 192, 384}:
        raise ImmutablePinError(
            "current admission ceiling must be one of staged values 24, 96, 192, 384"
        )


def _new_admission_ramp_state(
    *, rollout_generation: int, current_ceiling: int, reason: str, now: float
) -> dict[str, Any]:
    return {
        "schema_version": ADMISSION_RAMP_SCHEMA_VERSION,
        "protocol": ADMISSION_RAMP_PROTOCOL,
        "rollout_generation": int(rollout_generation),
        "current_ceiling": int(current_ceiling),
        "window": None,
        "last_observation": None,
        "last_action": {
            "action": "initialized",
            "reason": reason,
            "at": utc_timestamp(now),
            "timestamp": float(now),
        },
        "promotions": [],
        "resets": [],
    }


def _valid_run_qid_map(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == set(REQUIRED_RUNS)
        and all(
            isinstance(value[run_id], int)
            and not isinstance(value[run_id], bool)
            and 0 <= value[run_id] <= REQUIRED_RUN_QIDS[run_id]
            for run_id in REQUIRED_RUNS
        )
    )


def _validate_ramp_observation_record(value: Any) -> None:
    if not isinstance(value, dict):
        raise ControlError("admission ramp observation record is invalid")
    if (
        not isinstance(value.get("path"), str)
        or not value["path"]
        or _SHA256_RE.fullmatch(str(value.get("sha256", ""))) is None
        or not isinstance(value.get("timestamp"), (int, float))
        or isinstance(value.get("timestamp"), bool)
        or not isinstance(value.get("captured_timestamp"), (int, float))
        or isinstance(value.get("captured_timestamp"), bool)
        or float(value["captured_timestamp"]) > float(value["timestamp"])
        or value.get("cadence") not in MONITOR_CADENCE_SECONDS
        or not isinstance(value.get("clean"), bool)
    ):
        raise ControlError("admission ramp observation record fields are invalid")
    qids = value.get("run_validated_qids")
    if value["cadence"] == "health":
        if qids is not None:
            raise ControlError("health ramp observation contains semantic progress")
    elif not _valid_run_qid_map(qids):
        raise ControlError("semantic ramp observation lacks exact run progress")


def _validate_admission_ramp(
    control: Mapping[str, Any], ramp: Any
) -> None:
    """Validate the durable, generation-scoped rollout evidence accumulator."""

    if not isinstance(ramp, dict):
        raise ControlError("schema-5 admission ramp state is missing")
    if (
        ramp.get("schema_version") != ADMISSION_RAMP_SCHEMA_VERSION
        or ramp.get("protocol") != ADMISSION_RAMP_PROTOCOL
    ):
        raise ControlError("schema-5 admission ramp protocol is invalid")
    generation = ramp.get("rollout_generation")
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation != control.get("rollout_generation")
    ):
        raise ControlError("admission ramp rollout generation is not current")
    ceiling = ramp.get("current_ceiling")
    if ceiling != control.get("admission", {}).get("current_ceiling"):
        raise ControlError("admission ramp ceiling differs from dispatcher admission")
    if not isinstance(ramp.get("promotions"), list) or not isinstance(
        ramp.get("resets"), list
    ):
        raise ControlError("admission ramp promotion/reset history is invalid")
    last_observation = ramp.get("last_observation")
    if last_observation is not None:
        _validate_ramp_observation_record(last_observation)
    if not isinstance(ramp.get("last_action"), dict):
        raise ControlError("admission ramp last action is invalid")
    promotion_transitions = [
        row
        for row in control.get("transition_history", [])
        if isinstance(row, dict) and row.get("event") == "admission_ceiling_promoted"
    ]
    transition_index = 0
    for promotion in ramp["promotions"]:
        if not isinstance(promotion, dict):
            raise ControlError("admission ramp promotion record is invalid")
        source = promotion.get("from_ceiling")
        requirement = ADMISSION_RAMP_REQUIREMENTS.get(source)
        observations = promotion.get("observations")
        if (
            requirement is None
            or promotion.get("to_ceiling") != requirement["next_ceiling"]
            or not isinstance(observations, list)
            or not observations
            or promotion.get("evidence_sha256") != sha256_value(observations)
            or not _valid_run_qid_map(
                promotion.get("baseline_run_validated_qids")
            )
            or not _valid_run_qid_map(promotion.get("final_run_validated_qids"))
            or float(promotion.get("clean_seconds", -1.0))
            < float(requirement["clean_seconds"])
        ):
            raise ControlError("admission ramp promotion proof is invalid")
        for observation in observations:
            _validate_ramp_observation_record(observation)
            if observation["clean"] is not True:
                raise ControlError("admission ramp promotion includes unclean evidence")
        baseline_qids = promotion["baseline_run_validated_qids"]
        final_qids = promotion["final_run_validated_qids"]
        if any(final_qids[run_id] < baseline_qids[run_id] for run_id in REQUIRED_RUNS):
            raise ControlError("admission ramp promotion contains regressed run progress")
        if source == 24 and not all(
            final_qids[run_id] > baseline_qids[run_id] for run_id in REQUIRED_RUNS
        ):
            raise ControlError("initial admission promotion lacks all-run QID progress")
        matched = False
        while transition_index < len(promotion_transitions):
            details = promotion_transitions[transition_index].get("details", {})
            transition_index += 1
            matched = bool(
                details.get("previous") == source
                and details.get("current") == promotion["to_ceiling"]
                and details.get("rollout_generation")
                == promotion.get("rollout_generation")
                and details.get("fleet_generation")
                == promotion.get("fleet_generation")
                and details.get("throughput_epoch")
                == promotion.get("throughput_epoch")
                and details.get("evidence_sha256")
                == promotion.get("evidence_sha256")
            )
            if matched:
                break
        if not matched:
            raise ControlError("admission ramp promotion is not journal-bound")
    window = ramp.get("window")
    if window is None:
        return
    if not isinstance(window, dict):
        raise ControlError("admission ramp window is invalid")
    if (
        window.get("rollout_generation") != generation
        or window.get("ceiling") != ceiling
        or ceiling == ADMISSION_RAMP_STAGES[-1]
    ):
        raise ControlError("admission ramp window does not bind the current stage")
    if not isinstance(window.get("fleet_generation"), str) or not window.get(
        "fleet_generation"
    ):
        raise ControlError("admission ramp window lacks a fleet generation")
    if (
        not isinstance(window.get("throughput_epoch"), int)
        or isinstance(window.get("throughput_epoch"), bool)
        or window["throughput_epoch"] < 1
    ):
        raise ControlError("admission ramp window lacks a throughput epoch")
    if not isinstance(window.get("started_timestamp"), (int, float)) or isinstance(
        window.get("started_timestamp"), bool
    ):
        raise ControlError("admission ramp window start is invalid")
    if not _valid_run_qid_map(window.get("baseline_run_validated_qids")) or not (
        _valid_run_qid_map(window.get("latest_run_validated_qids"))
    ):
        raise ControlError("admission ramp run progress evidence is invalid")
    if not isinstance(window.get("observations"), list) or not window["observations"]:
        raise ControlError("admission ramp window has no clean observations")
    previous_timestamp: float | None = None
    for observation in window["observations"]:
        _validate_ramp_observation_record(observation)
        timestamp = float(observation["timestamp"])
        if observation["clean"] is not True or (
            previous_timestamp is not None and timestamp < previous_timestamp
        ):
            raise ControlError("admission ramp window evidence is unclean or reordered")
        previous_timestamp = timestamp


def _validate_monitoring_state(monitoring: Any) -> None:
    """Validate mutable cadence state without allowing cadence or ownership drift."""

    if not isinstance(monitoring, dict):
        raise ControlError("schema-5 monitoring state must be an object")
    if monitoring.get("owner_role") != MONITOR_OWNER_ROLE:
        raise ControlError(
            f"schema-5 monitoring must be owned by {MONITOR_OWNER_ROLE!r}"
        )
    cadences = monitoring.get("cadences")
    if not isinstance(cadences, dict) or set(cadences) != set(MONITOR_CADENCE_SECONDS):
        raise ControlError(
            "schema-5 monitoring must contain exactly health, semantic, and daily cadences"
        )
    for cadence, expected_interval in MONITOR_CADENCE_SECONDS.items():
        row = cadences[cadence]
        if not isinstance(row, dict):
            raise ControlError(f"monitor cadence {cadence!r} must be an object")
        if row.get("interval_seconds") != expected_interval:
            raise ControlError(
                f"monitor cadence {cadence!r} interval drifted: expected "
                f"{expected_interval}, found {row.get('interval_seconds')!r}"
            )
        next_due = row.get("next_due_timestamp")
        if not isinstance(next_due, (int, float)) or isinstance(next_due, bool):
            raise ControlError(
                f"monitor cadence {cadence!r} next_due_timestamp is invalid"
            )
        failures = row.get("consecutive_failures")
        if not isinstance(failures, int) or isinstance(failures, bool) or failures < 0:
            raise ControlError(
                f"monitor cadence {cadence!r} consecutive_failures is invalid"
            )
        for timestamp_field in (
            "last_attempt_timestamp",
            "last_success_timestamp",
            "last_failure_timestamp",
        ):
            timestamp = row.get(timestamp_field)
            if timestamp is not None and (
                not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool)
            ):
                raise ControlError(
                    f"monitor cadence {cadence!r} {timestamp_field} is invalid"
                )
        active = row.get("active_attempt")
        if active is not None:
            if not isinstance(active, dict):
                raise ControlError(
                    f"monitor cadence {cadence!r} active_attempt is invalid"
                )
            required = {
                "attempt_id",
                "started_timestamp",
                "scheduled_for_timestamp",
                "controller_generation",
                "controller_intent_token",
                "controller_job_id",
                "log_path",
            }
            if not required.issubset(active):
                raise ControlError(
                    f"monitor cadence {cadence!r} active_attempt is incomplete"
                )
            if not isinstance(active["attempt_id"], str) or not active["attempt_id"]:
                raise ControlError(f"monitor cadence {cadence!r} attempt ID is invalid")
            if not isinstance(active["started_timestamp"], (int, float)):
                raise ControlError(
                    f"monitor cadence {cadence!r} attempt timestamp is invalid"
                )


def _required_pin(mapping: Mapping[str, Any], field: str, *, context: str) -> Any:
    value = mapping.get(field)
    if value is None or value == "" or value == []:
        raise ImmutablePinError(
            f"{context} is missing required immutable field {field!r}"
        )
    return value


_IMMUTABLE_PIN_FIELDS = {
    "release_id",
    "release_bundle_root",
    "release_bundle_id",
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
    "hf_home",
    "results_root",
    "server_pool_root",
    "dispatcher_command",
    "fleet_supervisor_command",
    "runs",
}
_RUN_PIN_FIELDS = {
    "run_id",
    "run_root",
    "cell_count",
    "expected_qids",
    "weight",
    "manifest_path",
    "manifest_sha256",
    "lineage_path",
    "lineage_sha256",
    "policy_path",
    "policy_sha256",
    "benchmark_contract_path",
    "benchmark_contract_sha256",
}
_RELEASE_FRAGMENT_FIELDS = {
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


def _run_pins_by_id(pins: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    runs = pins.get("runs")
    if not isinstance(runs, list):
        raise ImmutablePinError("immutable runs must be an array")
    by_id: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict) or set(run) != _RUN_PIN_FIELDS:
            raise ImmutablePinError(
                "every immutable run pin must contain exactly "
                f"{sorted(_RUN_PIN_FIELDS)}"
            )
        run_id = str(_required_pin(run, "run_id", context="run pin"))
        if run_id in by_id:
            raise ImmutablePinError(f"duplicate immutable run pin {run_id!r}")
        by_id[run_id] = run
    return by_id


def expected_dispatcher_command(pins: Mapping[str, Any]) -> list[str]:
    """Return the one production dispatcher argv accepted by the control schema."""

    by_id = _run_pins_by_id(pins)
    release = Path(str(pins["release_worktree"])).expanduser().resolve()
    harness_python = (
        Path(str(pins["harness_environment_prefix"])).expanduser().resolve()
        / "bin"
        / "python"
    )
    results_root = Path(str(pins["results_root"])).expanduser().resolve()
    state_dir = results_root / CONTROL_STATE_DIRNAME
    pool = Path(str(pins["server_pool_root"])).expanduser().resolve()
    command = [
        str(harness_python),
        "-u",
        str(release / "slurm" / "dispatch_sweeps.py"),
        "dispatch",
        "--results-root",
        str(results_root),
        "--state-dir",
        str(state_dir),
        "--control-state-dir",
        str(state_dir),
        "--qos-limit",
        "448",
        "--reserve",
        "64",
        "--max-batch",
        "24",
        "--poll-seconds",
        "120",
        "--cell-partition",
        "mit_preemptable",
        "--cell-time",
        "12:00:00",
        "--cell-mem",
        "4G",
        "--fanout-slots-per-server",
        "24",
        "--validation-budget",
        "64",
    ]
    for run_id in REQUIRED_RUNS:
        command.extend(
            ["--run", f"{run_id}={Path(str(by_id[run_id]['run_root'])).resolve()}"]
        )
        command.extend(["--server-pool", f"{run_id}={pool}"])
        command.extend(["--weight", f"{run_id}=1"])
    command.append("--probe-servers")
    return command


def expected_fleet_supervisor_command(pins: Mapping[str, Any]) -> list[str]:
    """Return the one canonical fleet-supervisor argv accepted by production."""

    release = Path(str(pins["release_worktree"])).expanduser().resolve()
    harness_python = (
        Path(str(pins["harness_environment_prefix"])).expanduser().resolve()
        / "bin"
        / "python"
    )
    return [
        str(harness_python),
        "-u",
        str(release / "slurm" / "keepalive.py"),
        "--run-id",
        "schema5-v1",
        "--run-root",
        str(Path(str(pins["server_pool_root"])).expanduser().resolve()),
        "--release-worktree",
        str(release),
        "--release-id",
        str(pins["release_id"]),
        "--model-contract",
        str(Path(str(pins["model_contract_path"])).expanduser().resolve()),
        "--model-contract-sha256",
        str(pins["model_contract_sha256"]),
        "--fleet-contract",
        str(Path(str(pins["fleet_contract_path"])).expanduser().resolve()),
        "--fleet-contract-sha256",
        str(pins["fleet_contract_sha256"]),
        "--harness-environment-prefix",
        str(Path(str(pins["harness_environment_prefix"])).expanduser().resolve()),
        "--serving-environment-prefix",
        str(Path(str(pins["serving_environment_prefix"])).expanduser().resolve()),
        "--harness-environment-manifest",
        str(
            Path(str(pins["harness_environment_manifest_path"])).expanduser().resolve()
        ),
        "--serving-environment-manifest",
        str(
            Path(str(pins["serving_environment_manifest_path"])).expanduser().resolve()
        ),
        "--harness-environment-sha256",
        str(pins["harness_environment_sha256"]),
        "--serving-environment-sha256",
        str(pins["serving_environment_sha256"]),
        "--hf-home",
        str(Path(str(pins["hf_home"])).expanduser().resolve()),
        "--interval",
        "300",
    ]


def _validate_checksum_sidecar(
    path: Path, expected_name: str, expected_hash: str
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ImmutablePinError(f"pinned checksum sidecar is missing: {path}")
    fields = path.read_text(encoding="utf-8").strip().split()
    if fields != [expected_hash, expected_name]:
        raise ImmutablePinError(
            f"checksum sidecar does not bind {expected_name}: {path}"
        )


def _benchmark_derived_qids(run: Mapping[str, Any]) -> int:
    manifest_path = Path(str(run["manifest_path"])).resolve()
    contracts_path = Path(str(run["benchmark_contract_path"])).resolve()
    try:
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        contracts_payload = json.loads(contracts_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImmutablePinError(
            f"cannot derive benchmark QID cardinality: {exc}"
        ) from exc
    if not isinstance(rows, list) or len(rows) != run["cell_count"]:
        raise ImmutablePinError(f"manifest cardinality drifted: {manifest_path}")
    if not isinstance(contracts_payload, dict):
        raise ImmutablePinError(
            f"benchmark contracts must be an object: {contracts_path}"
        )
    if (
        contracts_payload.get("manifest_filename") != "cells.json"
        or contracts_payload.get("manifest_sha256") != run["manifest_sha256"]
        or contracts_payload.get("manifest_cell_count") != run["cell_count"]
    ):
        raise ImmutablePinError(
            "benchmark contracts do not bind the exact run manifest"
        )
    contracts = contracts_payload.get("contracts")
    if not isinstance(contracts, list) or not contracts:
        raise ImmutablePinError("benchmark contracts contain no typed contract rows")
    counts: dict[tuple[str, int | None, int], int] = {}
    for contract in contracts:
        if not isinstance(contract, dict) or not isinstance(contract.get("key"), dict):
            raise ImmutablePinError("benchmark contract row is malformed")
        key_payload = contract["key"]
        if set(key_payload) != {"benchmark", "n_questions", "seed"}:
            raise ImmutablePinError("benchmark contract key has unexpected fields")
        key = (
            str(key_payload["benchmark"]),
            key_payload["n_questions"],
            key_payload["seed"],
        )
        count = contract.get("question_count")
        qids = contract.get("ordered_qids")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            or not isinstance(qids, list)
            or len(qids) != count
            or len(qids) != len(set(qids))
            or key in counts
        ):
            raise ImmutablePinError("benchmark contract QID cardinality is malformed")
        counts[key] = count
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ImmutablePinError("manifest contains a non-object cell")
        key = (str(row.get("benchmark")), row.get("n_questions"), row.get("seed"))
        if key not in counts:
            raise ImmutablePinError(f"manifest cell lacks benchmark contract: {key!r}")
        total += counts[key]
    return total


def _assert_root_read_only(path: Path, *, description: str) -> None:
    if path.stat().st_mode & 0o222:
        raise ImmutablePinError(f"{description} is not sealed read-only: {path}")


def _validate_release_bundle(pins: Mapping[str, Any]) -> None:
    """Verify the freezer's marker-last bundle and its exact control fragment.

    The freezer performs expensive package/runtime and recursive environment checks at
    publication.  The control plane re-verifies every marker-addressed byte, the bundle
    ID, live root seals, and the identity's exact pin fragment on every file-verified
    load.  A hand-written pins file therefore cannot bypass an unsealed release.
    """

    supplied_root = Path(str(pins["release_bundle_root"])).expanduser()
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise ImmutablePinError(
            f"release bundle root is missing or unsafe: {supplied_root}"
        )
    root = supplied_root.resolve()
    _assert_root_read_only(root, description="release bundle root")
    marker_path = root / RELEASE_COMPLETE_FILENAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImmutablePinError(
            f"cannot read release completion marker: {exc}"
        ) from exc
    marker_fields = {
        "schema_version",
        "release_id",
        "complete",
        "publication_protocol",
        "artifacts",
        "git_commit",
        "source_tree_sha256",
        "release_bundle_id",
    }
    if not isinstance(marker, dict) or set(marker) != marker_fields:
        raise ImmutablePinError("release completion marker has the wrong fields")
    candidate = dict(marker)
    bundle_id = candidate.pop("release_bundle_id", None)
    if (
        marker.get("schema_version") != 1
        or marker.get("release_id") != pins["release_id"]
        or marker.get("complete") is not True
        or marker.get("publication_protocol") != "fsync_verify_marker_last"
        or marker.get("git_commit") != pins["git_commit"]
        or marker.get("source_tree_sha256") != pins["source_tree_sha256"]
        or bundle_id != sha256_value(candidate)
        or bundle_id != pins["release_bundle_id"]
    ):
        raise ImmutablePinError("release completion marker identity is invalid")
    expected_artifacts = {
        HARNESS_ENVIRONMENT_FILENAME,
        SERVING_ENVIRONMENT_FILENAME,
        RELEASE_IDENTITY_FILENAME,
        HARNESS_ENVIRONMENT_FILENAME + ".sha256",
        SERVING_ENVIRONMENT_FILENAME + ".sha256",
        RELEASE_IDENTITY_FILENAME + ".sha256",
    }
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ImmutablePinError("release marker has the wrong artifact inventory")
    for filename, record in artifacts.items():
        path = root / filename
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256", "size"}
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.get("size")
            or sha256_file(path) != record.get("sha256")
        ):
            raise ImmutablePinError(f"release marker artifact drifted: {path}")
        _assert_root_read_only(path, description="release bundle artifact")
    for filename in (
        HARNESS_ENVIRONMENT_FILENAME,
        SERVING_ENVIRONMENT_FILENAME,
        RELEASE_IDENTITY_FILENAME,
    ):
        artifact = root / filename
        expected_checksum = f"{sha256_file(artifact)}  {filename}\n"
        if (root / (filename + ".sha256")).read_text(
            encoding="utf-8"
        ) != expected_checksum:
            raise ImmutablePinError(f"release checksum sidecar is invalid: {filename}")
    try:
        identity = json.loads(
            (root / RELEASE_IDENTITY_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImmutablePinError(f"cannot parse release identity: {exc}") from exc
    identity_fields = {
        "schema_version",
        "release_id",
        "git",
        "release_worktree",
        "worktree_sealed_read_only",
        "materialization",
        "offline_environment",
        "model_contract",
        "fleet_contract",
        "environments",
        "publication",
        "control_pin_fragment",
    }
    if not isinstance(identity, dict) or set(identity) != identity_fields:
        raise ImmutablePinError("release identity has the wrong fields")
    expected_fragment = {field: pins[field] for field in _RELEASE_FRAGMENT_FIELDS}
    if (
        identity.get("schema_version") != 1
        or identity.get("release_id") != pins["release_id"]
        or identity.get("release_worktree") != pins["release_worktree"]
        or identity.get("worktree_sealed_read_only") is not True
        or identity.get("publication")
        != {
            "protocol": "fsync_verify_marker_last",
            "complete_marker": RELEASE_COMPLETE_FILENAME,
        }
        or identity.get("control_pin_fragment") != expected_fragment
    ):
        raise ImmutablePinError(
            "release identity does not bind the exact sealed control pins"
        )
    materialization = identity.get("materialization")
    materialization_fields = {
        "schema_version",
        "release_id",
        "root",
        "marker_path",
        "marker_sha256",
        "materialization_id",
        "tag_commit",
        "source_tree_sha256",
        "paths",
        "stage_records",
    }
    expected_materialization_root = root.parent
    expected_materialization_marker = (
        expected_materialization_root / MATERIALIZATION_COMPLETE_FILENAME
    )
    if (
        not isinstance(materialization, dict)
        or set(materialization) != materialization_fields
    ):
        raise ImmutablePinError("release materialization binding has the wrong fields")
    paths = materialization.get("paths")
    if (
        materialization.get("schema_version") != 2
        or materialization.get("release_id") != pins["release_id"]
        or materialization.get("root") != str(expected_materialization_root)
        or materialization.get("marker_path") != str(expected_materialization_marker)
        or materialization.get("tag_commit") != pins["git_commit"]
        or materialization.get("source_tree_sha256") != pins["source_tree_sha256"]
        or not isinstance(paths, dict)
        or paths.get("release_worktree") != pins["release_worktree"]
        or paths.get("harness_prefix") != pins["harness_environment_prefix"]
        or paths.get("serving_prefix") != pins["serving_environment_prefix"]
        or not isinstance(materialization.get("stage_records"), dict)
        or expected_materialization_marker.is_symlink()
        or not expected_materialization_marker.is_file()
        or sha256_file(expected_materialization_marker)
        != materialization.get("marker_sha256")
    ):
        raise ImmutablePinError("release materialization binding is invalid")
    _assert_root_read_only(
        expected_materialization_marker,
        description="materialization completion marker",
    )
    try:
        materialization_marker = json.loads(
            expected_materialization_marker.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImmutablePinError(f"cannot parse materialization marker: {exc}") from exc
    if not isinstance(materialization_marker, dict):
        raise ImmutablePinError("materialization marker must be an object")
    candidate_materialization = dict(materialization_marker)
    materialization_id = candidate_materialization.pop("materialization_id", None)
    expected_materialization_id = hashlib.sha256(
        (
            json.dumps(
                candidate_materialization,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if (
        materialization_marker.get("schema_version") != 2
        or materialization_marker.get("release_id") != pins["release_id"]
        or materialization_marker.get("complete") is not True
        or materialization_marker.get("paths") != paths
        or materialization_marker.get("stage_records")
        != materialization.get("stage_records")
        or materialization_id != materialization.get("materialization_id")
        or materialization_id != expected_materialization_id
    ):
        raise ImmutablePinError("materialization marker identity is invalid")
    stage_records = materialization.get("stage_records")
    if set(stage_records) != set(MATERIALIZATION_STAGE_FILENAMES):
        raise ImmutablePinError("materialization stage inventory is incomplete")
    for stage, filename in MATERIALIZATION_STAGE_FILENAMES.items():
        record = stage_records[stage]
        stage_path = expected_materialization_root / filename
        if (
            not isinstance(record, dict)
            or set(record) != {"filename", "sha256", "record_sha256"}
            or record.get("filename") != filename
            or stage_path.is_symlink()
            or not stage_path.is_file()
            or sha256_file(stage_path) != record.get("sha256")
        ):
            raise ImmutablePinError(f"materialization stage artifact drifted: {stage}")
        _assert_root_read_only(stage_path, description=f"{stage} materialization stage")
        try:
            stage_payload = json.loads(stage_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ImmutablePinError(
                f"cannot parse {stage} materialization stage: {exc}"
            ) from exc
        if not isinstance(stage_payload, dict):
            raise ImmutablePinError(f"{stage} materialization stage must be an object")
        candidate_stage = dict(stage_payload)
        record_sha256 = candidate_stage.pop("record_sha256", None)
        expected_record_sha256 = hashlib.sha256(
            (
                json.dumps(
                    candidate_stage,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        if (
            stage_payload.get("schema_version") != 2
            or stage_payload.get("release_id") != pins["release_id"]
            or stage_payload.get("stage") != stage
            or record_sha256 != record.get("record_sha256")
            or record_sha256 != expected_record_sha256
        ):
            raise ImmutablePinError(
                f"materialization stage record identity drifted: {stage}"
            )
    git_identity = identity.get("git")
    if not isinstance(git_identity, dict) or (
        git_identity.get("git_commit") != pins["git_commit"]
        or git_identity.get("source_tree_sha256") != pins["source_tree_sha256"]
    ):
        raise ImmutablePinError("release identity Git/source pin drifted")
    environment_records = identity.get("environments")
    expected_environment_paths = {
        "harness": root / HARNESS_ENVIRONMENT_FILENAME,
        "serving": root / SERVING_ENVIRONMENT_FILENAME,
    }
    if not isinstance(environment_records, dict) or set(environment_records) != {
        "harness",
        "serving",
    }:
        raise ImmutablePinError("release identity environment records are incomplete")
    for role, manifest_path in expected_environment_paths.items():
        record = environment_records.get(role)
        prefix_field = f"{role}_environment_prefix"
        sha_field = f"{role}_environment_sha256"
        manifest_field = f"{role}_environment_manifest_path"
        if (
            not isinstance(record, dict)
            or record.get("prefix") != pins[prefix_field]
            or record.get("manifest_path") != str(manifest_path)
            or record.get("manifest_sha256") != pins[sha_field]
            or pins[manifest_field] != str(manifest_path)
        ):
            raise ImmutablePinError(f"release identity {role} environment drifted")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ImmutablePinError(
                f"cannot parse {role} environment manifest: {exc}"
            ) from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("release_id") != pins["release_id"]
            or manifest.get("role") != role
            or manifest.get("prefix") != pins[prefix_field]
            or manifest.get("sealed_read_only") is not True
        ):
            raise ImmutablePinError(f"{role} environment manifest is not sealed")
        _assert_root_read_only(
            Path(str(pins[prefix_field])).resolve(), description=f"{role} environment"
        )
    _assert_root_read_only(
        Path(str(pins["release_worktree"])).resolve(), description="release worktree"
    )


def validate_immutable_pins(pins: Mapping[str, Any], *, verify_files: bool) -> None:
    """Validate the frozen schema-5 production identity and (optionally) its files."""
    if not isinstance(pins, dict):
        raise ImmutablePinError("immutable pins must be an object")
    if set(pins) != _IMMUTABLE_PIN_FIELDS:
        raise ImmutablePinError(
            "immutable pin fields differ from the closed schema: "
            f"missing={sorted(_IMMUTABLE_PIN_FIELDS - set(pins))}, "
            f"unexpected={sorted(set(pins) - _IMMUTABLE_PIN_FIELDS)}"
        )
    for field in (
        "release_id",
        "release_bundle_root",
        "release_bundle_id",
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
        "hf_home",
        "results_root",
        "server_pool_root",
        "dispatcher_command",
        "fleet_supervisor_command",
        "runs",
    ):
        _required_pin(pins, field, context="immutable pins")
    if pins["release_id"] != "sweep-recovery-schema5-v1.1":
        raise ImmutablePinError("release_id must be 'sweep-recovery-schema5-v1.1'")
    if not isinstance(pins["dispatcher_command"], list) or not all(
        isinstance(item, str) and item for item in pins["dispatcher_command"]
    ):
        raise ImmutablePinError("dispatcher_command must be a non-empty argv array")
    if not isinstance(pins["fleet_supervisor_command"], list) or not all(
        isinstance(item, str) and item for item in pins["fleet_supervisor_command"]
    ):
        raise ImmutablePinError(
            "fleet_supervisor_command must be a non-empty argv array"
        )
    for command_field in ("dispatcher_command", "fleet_supervisor_command"):
        command = pins[command_field]
        executable = Path(pins[command_field][0]).expanduser()
        if not executable.is_absolute():
            raise ImmutablePinError(
                f"{command_field} must pin an absolute executable, found {executable}"
            )
        if verify_files and not executable.is_file():
            raise ImmutablePinError(
                f"{command_field} executable is missing: {executable}"
            )
        if any(
            item == "--successor-sbatch" or "loop_keepalive.sbatch" in item
            for item in command
        ):
            raise ImmutablePinError(
                f"{command_field} must not create a competing legacy successor chain"
            )
        release_root = Path(str(pins["release_worktree"])).expanduser().resolve()
        for item in command[1:]:
            if not item.endswith((".py", ".sh")):
                continue
            script = Path(item).expanduser()
            if not script.is_absolute():
                raise ImmutablePinError(
                    f"{command_field} script arguments must be absolute: {item}"
                )
            try:
                script.resolve().relative_to(release_root)
            except ValueError as exc:
                raise ImmutablePinError(
                    f"{command_field} executes outside frozen release: {script}"
                ) from exc
            if verify_files and not script.is_file():
                raise ImmutablePinError(
                    f"pinned controller script is missing: {script}"
                )

    by_id = _run_pins_by_id(pins)
    runs = pins["runs"]
    if set(by_id) != set(REQUIRED_RUNS):
        raise ImmutablePinError(
            f"schema-5 run pins must be exactly {sorted(REQUIRED_RUNS)}, found {sorted(by_id)}"
        )
    for run_id, expected_cells in REQUIRED_RUNS.items():
        run = by_id[run_id]
        for field in (
            "run_root",
            "manifest_path",
            "manifest_sha256",
            "lineage_path",
            "lineage_sha256",
            "policy_path",
            "policy_sha256",
            "benchmark_contract_path",
            "benchmark_contract_sha256",
        ):
            _required_pin(run, field, context=f"run pin {run_id}")
        if run.get("cell_count") != expected_cells:
            raise ImmutablePinError(
                f"run {run_id} must pin {expected_cells} cells, found {run.get('cell_count')!r}"
            )
        if run.get("expected_qids") != REQUIRED_RUN_QIDS[run_id]:
            raise ImmutablePinError(
                f"run {run_id} must pin {REQUIRED_RUN_QIDS[run_id]} benchmark QIDs"
            )
        if (
            type(run.get("weight")) not in {int, float}
            or float(run["weight"]) != RUN_WEIGHT
        ):
            raise ImmutablePinError(f"run {run_id} weight must be exactly {RUN_WEIGHT}")

    if sum(int(run["cell_count"]) for run in runs) != EXPECTED_TOTAL_CELLS:
        raise ImmutablePinError("immutable runs do not total exactly 22,680 cells")
    if sum(int(run["expected_qids"]) for run in runs) != EXPECTED_TOTAL_QIDS:
        raise ImmutablePinError("immutable runs do not total exactly 4,524,660 QIDs")
    results_root = Path(str(pins["results_root"])).expanduser().resolve()
    expected_pool = results_root / "server_pools" / "schema5-v1"
    if Path(str(pins["server_pool_root"])).expanduser().resolve() != expected_pool:
        raise ImmutablePinError(f"canonical server pool must be {expected_pool}")
    for run_id, run in by_id.items():
        run_root = Path(str(run["run_root"])).expanduser().resolve()
        if run_root != results_root / run_id:
            raise ImmutablePinError(
                f"run {run_id} must live directly below the pinned results root"
            )
        expected_paths = {
            "manifest_path": run_root / "cells.json",
            "lineage_path": run_root / "lineage.schema5-v1.json",
            "policy_path": run_root / "artifact_policy.schema5-v1.json",
            "benchmark_contract_path": run_root / "benchmark_contracts.v1.json",
        }
        for field, expected_path in expected_paths.items():
            if Path(str(run[field])).expanduser().resolve() != expected_path:
                raise ImmutablePinError(f"run {run_id} {field} must be {expected_path}")
    if pins["dispatcher_command"] != expected_dispatcher_command(pins):
        raise ImmutablePinError(
            "dispatcher_command differs from the exact schema-5 topology command"
        )
    if pins["fleet_supervisor_command"] != expected_fleet_supervisor_command(pins):
        raise ImmutablePinError(
            "fleet_supervisor_command differs from the exact schema-5 topology command"
        )

    if not verify_files:
        return
    _validate_release_bundle(pins)
    for directory_field in (
        "release_bundle_root",
        "release_worktree",
        "harness_environment_prefix",
        "serving_environment_prefix",
        "hf_home",
        "results_root",
        "server_pool_root",
    ):
        path = Path(str(pins[directory_field])).expanduser().resolve()
        if not path.is_dir():
            raise ImmutablePinError(
                f"pinned directory is missing: {directory_field}={path}"
            )
    release_tree = Path(str(pins["release_worktree"])).expanduser().resolve()
    observed_source_hash = sha256_tree(release_tree)
    if observed_source_hash != pins["source_tree_sha256"]:
        raise ImmutablePinError(
            "frozen release source tree drifted: "
            f"expected {pins['source_tree_sha256']}, got {observed_source_hash}"
        )
    harness_python = Path(str(pins["harness_environment_prefix"])) / "bin" / "python"
    if not harness_python.is_file():
        raise ImmutablePinError(f"pinned harness Python is missing: {harness_python}")
    for release_relative in (
        Path("slurm") / "dispatch_sweeps.py",
        Path("slurm") / "run_dispatch_batch.sbatch.tmpl",
    ):
        release_artifact = release_tree / release_relative
        if not release_artifact.is_file():
            raise ImmutablePinError(
                f"pinned release is missing production cell runtime artifact: "
                f"{release_artifact}"
            )
    # Creation-time freezing proves the exact clean tag and records its commit.  A
    # materialized worktree's ``.git`` file may point back to the mutable source clone,
    # which is intentionally disposable after publication.  Runtime authority is the
    # sealed tree bytes above plus the marker-bound stored commit, never that mutable
    # Git administrative target.  This also prevents source-repository deletion from
    # taking a valid production release offline.
    file_pairs = [
        (pins["model_contract_path"], pins["model_contract_sha256"]),
        (pins["fleet_contract_path"], pins["fleet_contract_sha256"]),
        (
            pins["harness_environment_manifest_path"],
            pins["harness_environment_sha256"],
        ),
        (
            pins["serving_environment_manifest_path"],
            pins["serving_environment_sha256"],
        ),
    ]
    for run in runs:
        file_pairs.extend(
            (run[f"{kind}_path"], run[f"{kind}_sha256"])
            for kind in ("manifest", "lineage", "policy", "benchmark_contract")
        )
    for path_value, expected_hash in file_pairs:
        path = Path(str(path_value)).expanduser().resolve()
        if not path.is_file():
            raise ImmutablePinError(f"pinned artifact is missing: {path}")
        observed = sha256_file(path)
        if observed != expected_hash:
            raise ImmutablePinError(
                f"pinned artifact drifted: {path}; expected {expected_hash}, got {observed}"
            )
    for run_id, run in by_id.items():
        run_root = Path(str(run["run_root"])).resolve()
        for artifact_field, checksum_name in (
            ("manifest_path", "cells.sha256"),
            ("lineage_path", "lineage.schema5-v1.sha256"),
            ("policy_path", "artifact_policy.schema5-v1.sha256"),
            ("benchmark_contract_path", "benchmark_contracts.v1.sha256"),
        ):
            artifact = Path(str(run[artifact_field])).resolve()
            _validate_checksum_sidecar(
                run_root / checksum_name,
                artifact.name,
                str(run[artifact_field.replace("_path", "_sha256")]),
            )
        derived_qids = _benchmark_derived_qids(run)
        if derived_qids != REQUIRED_RUN_QIDS[run_id]:
            raise ImmutablePinError(
                f"run {run_id} benchmark-derived QIDs are {derived_qids}, "
                f"expected {REQUIRED_RUN_QIDS[run_id]}"
            )
    try:
        fleet_payload = json.loads(
            Path(str(pins["fleet_contract_path"])).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImmutablePinError(f"cannot parse fleet contract: {exc}") from exc
    if not isinstance(fleet_payload, dict):
        raise ImmutablePinError("fleet contract must be an object")
    if (
        fleet_payload.get("fleet_id") != "schema5-v1"
        or fleet_payload.get("release_id") != pins["release_id"]
        or fleet_payload.get("logical_replica_count") != 22
        or fleet_payload.get("allocated_gpu_count") != 24
        or fleet_payload.get("model_contract_sha256") != pins["model_contract_sha256"]
        or fleet_payload.get("server_pool", {}).get("root_suffix")
        != "server_pools/schema5-v1"
        or fleet_payload.get("server_pool", {}).get("no_requeue") is not True
    ):
        raise ImmutablePinError(
            "fleet contract does not encode the canonical 22/24 topology"
        )


def load_control(state_dir: Path, *, verify_files: bool = False) -> dict[str, Any]:
    path = _state_path(state_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ControlError(f"schema-5 control is not initialized: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlError(f"cannot read schema-5 control {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControlError("schema-5 control must be a JSON object")
    if value.get("schema_version") != CONTROL_SCHEMA_VERSION:
        raise ControlError(
            f"unsupported control schema {value.get('schema_version')!r}; "
            f"expected {CONTROL_SCHEMA_VERSION}"
        )
    if value.get("protocol") != CONTROL_PROTOCOL:
        raise ControlError(f"unexpected control protocol {value.get('protocol')!r}")
    immutable = value.get("immutable")
    if not isinstance(immutable, dict):
        raise ImmutablePinError("control immutable pins are missing")
    if value.get("immutable_sha256") != sha256_value(immutable):
        raise ImmutablePinError("control immutable pin digest is invalid")
    pins_path = state_dir / IMMUTABLE_PINS_FILENAME
    try:
        frozen_pins = json.loads(pins_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ImmutablePinError(
            f"frozen immutable pin copy is missing: {pins_path}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImmutablePinError(
            f"cannot read frozen immutable pins {pins_path}: {exc}"
        ) from exc
    if (
        frozen_pins != immutable
        or sha256_value(frozen_pins) != value["immutable_sha256"]
    ):
        raise ImmutablePinError(
            "control immutable pins differ from the frozen pin copy"
        )
    validate_immutable_pins(immutable, verify_files=verify_files)
    _validate_admission(value.get("admission", {}))
    _validate_monitoring_state(value.get("monitoring"))
    _validate_admission_ramp(value, value.get("admission_ramp"))
    validate_transition_history(value)
    journal_path = state_dir / TRANSITION_JOURNAL
    try:
        journal_records = _read_jsonl_locked(journal_path)
    except FileNotFoundError as exc:
        raise ControlError(
            f"append-only transition journal is missing: {journal_path}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlError(
            f"cannot validate transition journal {journal_path}: {exc}"
        ) from exc
    history = value["transition_history"]
    if (
        len(journal_records) < len(history)
        or journal_records[: len(history)] != history
    ):
        raise ControlError(
            "control transition history is not a prefix of its durable journal"
        )
    if value.get("desired_state") not in {"paused", "resuming", "running"}:
        raise ControlError("desired_state must be paused, resuming, or running")
    if (
        not isinstance(value.get("rollout_generation"), int)
        or value["rollout_generation"] < 0
    ):
        raise ControlError("rollout_generation must be a non-negative integer")
    if value["rollout_generation"] >= 1:
        validate_runtime_integrity_attestation(value, verify_metadata=False)
        validate_snapshot_integrity_attestation(
            value, state_dir=state_dir, verify_lease=False
        )
    else:
        if value.get(RUNTIME_ATTESTATION_STATE_KEY) is not None:
            raise ImmutablePinError(
                "generation-zero control must not claim a runtime integrity attestation"
            )
        if value.get(SNAPSHOT_ATTESTATION_STATE_KEY) is not None:
            if value.get("desired_state") != "paused" or value.get("drain_requested"):
                raise ReadinessError(
                    "generation-zero snapshot preseal requires paused non-draining control"
                )
            validate_snapshot_integrity_attestation(
                value, state_dir=state_dir, verify_lease=False
            )
    for gate in REQUIRED_GATES:
        if gate not in value.get("readiness", {}):
            raise ControlError(f"readiness gate {gate!r} is missing")
    for role in ROLE_NAMES:
        if role not in value.get("controllers", {}):
            raise ControlError(f"controller state for {role!r} is missing")
    resume_intent = value.get("resume_intent")
    if value.get("desired_state") == "resuming" and not isinstance(resume_intent, dict):
        raise ControlError("resuming desired state requires a durable resume_intent")
    if value.get("desired_state") == "running":
        if (
            not isinstance(resume_intent, dict)
            or resume_intent.get("state") != "complete"
        ):
            raise ControlError(
                "running desired state requires a complete resume transaction"
            )
        resume_ids = resume_intent.get("controller_job_ids")
        visible_ids = resume_intent.get("scheduler_visible_job_ids")
        if (
            not isinstance(resume_ids, dict)
            or set(resume_ids) != set(ROLE_NAMES)
            or not all(str(resume_ids[role]).isdigit() for role in ROLE_NAMES)
            or visible_ids != resume_ids
        ):
            raise ControlError(
                "running resume transaction does not bind both scheduler-visible controllers"
            )
    return value


def initialize_control(
    state_dir: Path,
    *,
    pins: Mapping[str, Any],
    alert_email: str = "mabdel03@mit.edu",
    now: float | None = None,
) -> dict[str, Any]:
    """Create the paused control atomically, or prove an existing init is identical."""
    state_dir = state_dir.expanduser().resolve()
    timestamp = time.time() if now is None else float(now)
    validate_immutable_pins(pins, verify_files=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "logs").mkdir(parents=True, exist_ok=True)
    (state_dir / "sbatch").mkdir(parents=True, exist_ok=True)
    (state_dir / "locks").mkdir(parents=True, exist_ok=True)
    with control_lock(state_dir):
        if _state_path(state_dir).exists():
            existing = load_control(state_dir, verify_files=True)
            if existing["immutable_sha256"] != sha256_value(pins):
                raise ImmutablePinError(
                    "init is idempotent only for exactly the existing immutable pins"
                )
            return existing
        if _read_jsonl_locked(state_dir / TRANSITION_JOURNAL, missing_ok=True):
            raise ControlError(
                "transition journal exists without control.json; preserve it and use a new "
                "state directory or perform explicit recovery"
            )
        control: dict[str, Any] = {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "protocol": CONTROL_PROTOCOL,
            "created_at": utc_timestamp(timestamp),
            "created_timestamp": timestamp,
            "updated_at": utc_timestamp(timestamp),
            "updated_timestamp": timestamp,
            "immutable": copy.deepcopy(dict(pins)),
            "immutable_sha256": sha256_value(pins),
            "admission": copy.deepcopy(DEFAULT_ADMISSION),
            "admission_ramp": _new_admission_ramp_state(
                rollout_generation=0,
                current_ceiling=int(DEFAULT_ADMISSION["current_ceiling"]),
                reason="initialized_paused",
                now=timestamp,
            ),
            "desired_state": "paused",
            "drain_requested": False,
            "drain_intent": None,
            "rollout_generation": 0,
            RUNTIME_ATTESTATION_STATE_KEY: None,
            SNAPSHOT_ATTESTATION_STATE_KEY: None,
            "resume_intent": None,
            "readiness": {
                gate: {"passed": False, "evidence": None, "attested_at": None}
                for gate in REQUIRED_GATES
            },
            "controllers": {
                role: {
                    "next_generation": 1,
                    "active": None,
                    "successor": None,
                    "submission_intent": None,
                    "last_exit": None,
                    "heartbeat": None,
                }
                for role in ROLE_NAMES
            },
            "transition_history": [],
            "throughput_epochs": [],
            "monitoring": {
                "owner_role": MONITOR_OWNER_ROLE,
                "cadences": {
                    cadence: {
                        "interval_seconds": interval,
                        # A newly resumed chain performs each monitor immediately.  A
                        # successful attempt advances this durable deadline; controller
                        # successors never reset it.
                        "next_due_at": utc_timestamp(timestamp),
                        "next_due_timestamp": timestamp,
                        "last_attempt_at": None,
                        "last_attempt_timestamp": None,
                        "last_success_at": None,
                        "last_success_timestamp": None,
                        "last_failure_at": None,
                        "last_failure_timestamp": None,
                        "last_returncode": None,
                        "last_error": None,
                        "consecutive_failures": 0,
                        "active_attempt": None,
                    }
                    for cadence, interval in MONITOR_CADENCE_SECONDS.items()
                },
            },
            "alerts": [],
            "alert_email": alert_email,
            "last_reconciliation": None,
        }
        append_transition(
            state_dir,
            control,
            event="initialized_paused",
            details={"immutable_sha256": control["immutable_sha256"]},
            now=timestamp,
        )
        pins_path = state_dir / IMMUTABLE_PINS_FILENAME
        _atomic_write_json(pins_path, dict(pins))
        pins_path.chmod(0o444)
        _atomic_write_json(_state_path(state_dir), control)
        return control


def _save_control(
    state_dir: Path, control: MutableMapping[str, Any], *, now: float
) -> None:
    control["updated_at"] = utc_timestamp(now)
    control["updated_timestamp"] = now
    # Refuse to persist mutation of any immutable pin or mandatory admission invariant.
    if control.get("immutable_sha256") != sha256_value(control.get("immutable")):
        raise ImmutablePinError("refusing to save mutated immutable pins")
    _validate_admission(control.get("admission", {}))
    _validate_monitoring_state(control.get("monitoring"))
    _validate_admission_ramp(control, control.get("admission_ramp"))
    validate_transition_history(control)
    _atomic_write_json(_state_path(state_dir), control)


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ReadinessError(
            f"{context} fields differ: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _exact_integer(value: Any, expected: int, *, context: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise ReadinessError(
            f"{context} must equal integer {expected}, found {value!r}"
        )


def _exact_boolean(value: Any, expected: bool, *, context: str) -> None:
    if not isinstance(value, bool) or value is not expected:
        raise ReadinessError(f"{context} must equal {expected!r}, found {value!r}")


def _validate_artifact_identity(
    control: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    *,
    name: str,
    exact_fields: set[str],
) -> Mapping[str, Any]:
    if not isinstance(payload, dict):
        raise ReadinessError(f"readiness artifact {name} must be a JSON object")
    _exact_keys(payload, exact_fields, context=f"readiness artifact {name}")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != name
        or payload.get("passed") is not True
        or payload.get("immutable_sha256") != control["immutable_sha256"]
    ):
        raise ReadinessError(
            f"readiness artifact {name} has invalid schema, identity, pass state, or pins"
        )
    return payload


def _load_only_source_report(
    wrapper: Mapping[str, Any],
    *,
    expected_name: str,
    snapshot_context: _SnapshotValidationContext,
) -> Mapping[str, Any]:
    references = wrapper.get("referenced_artifacts")
    if not isinstance(references, list) or len(references) != 1:
        raise ReadinessError(
            f"{wrapper.get('kind')} must reference exactly one source report"
        )
    reference = references[0]
    if not isinstance(reference, dict) or reference.get("name") != expected_name:
        raise ReadinessError(
            f"{wrapper.get('kind')} must reference source {expected_name!r}"
        )
    _, payload = _validate_referenced_artifact(
        reference,
        context=f"{wrapper.get('kind')} source report",
        verified=set(),
        active=set(),
        snapshot_context=snapshot_context,
    )
    if not isinstance(payload, dict):
        raise ReadinessError(f"source report {expected_name!r} must be a JSON object")
    return payload


def _validate_migration_artifacts(
    control: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, tuple[Path, Mapping[str, Any] | None]],
    *,
    snapshot_context: _SnapshotValidationContext,
) -> None:
    contracts = {
        "response_incident_archive_report": (
            "consolidated_response_report",
            {
                "protocol_incidents_total": 22,
                "protocol_already_reset": 22,
                "sealed_incident_qids": 1_064,
            },
        ),
        "checkpoint_migration_report": (
            "consolidated_checkpoint_report",
            {
                "migrated_checkpoints": 47,
                "remaining_schema1_checkpoints": 0,
                "coordinates_preserved": True,
            },
        ),
        "permanent_ledger_archive_report": (
            "consolidated_permanent_report",
            {
                "permanent_ledgers_archived": 3,
                "unresolved_permanent_ledgers": 0,
            },
        ),
    }
    for name, (source_name, expected) in contracts.items():
        wrapper = _validate_artifact_identity(
            control,
            artifacts[name][1],
            name=name,
            exact_fields={
                "schema_version",
                "kind",
                "passed",
                "immutable_sha256",
                "metrics",
                "referenced_artifacts",
            },
        )
        if wrapper.get("metrics") != expected:
            raise ReadinessError(f"{name} metrics do not match the recovery contract")
        source = _load_only_source_report(
            wrapper,
            expected_name=source_name,
            snapshot_context=snapshot_context,
        )
        if source.get("schema_version") != 1 or source.get("passed") is not True:
            raise ReadinessError(
                f"{source_name} did not report a valid passing recovery"
            )
        for key, value in expected.items():
            if source.get(key) != value:
                raise ReadinessError(
                    f"{source_name} {key} does not match its typed wrapper"
                )
            if key in metrics and metrics.get(key) != value:
                raise ReadinessError(
                    f"outer migration metric {key} is not artifact-derived"
                )


def _validate_semantic_artifact(
    control: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, tuple[Path, Mapping[str, Any] | None]],
    *,
    snapshot_context: _SnapshotValidationContext,
) -> None:
    name = "legacy_semantic_audit_report"
    wrapper = _validate_artifact_identity(
        control,
        artifacts[name][1],
        name=name,
        exact_fields={
            "schema_version",
            "kind",
            "passed",
            "immutable_sha256",
            "metrics",
            "referenced_artifacts",
        },
    )
    if wrapper.get("metrics") != dict(metrics):
        raise ReadinessError(
            "semantic audit wrapper metrics differ from the outer envelope"
        )
    source = _load_only_source_report(
        wrapper,
        expected_name="consolidated_semantic_report",
        snapshot_context=snapshot_context,
    )
    if (
        source.get("schema_version") != 1
        or source.get("kind") != "legacy_semantic_audit"
        or source.get("passed") is not True
        or source.get("manifest_cells") != EXPECTED_TOTAL_CELLS
        or source.get("invalid_rows") != 0
        or source.get("metrics") != dict(metrics)
    ):
        raise ReadinessError(
            "consolidated semantic report does not prove the exact baseline"
        )


def _validate_fleet_artifact(
    control: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, tuple[Path, Mapping[str, Any] | None]],
    *,
    snapshot_context: _SnapshotValidationContext,
) -> None:
    name = "fleet_health_report"
    report = _validate_artifact_identity(
        control,
        artifacts[name][1],
        name=name,
        exact_fields={
            "schema_version",
            "kind",
            "passed",
            "immutable_sha256",
            "server_pool_root",
            "metrics",
            "referenced_artifacts",
        },
    )
    if report.get("server_pool_root") != str(
        Path(control["immutable"]["server_pool_root"]).resolve()
    ) or report.get("metrics") != dict(metrics):
        raise ReadinessError(
            "fleet health artifact differs from the pinned pool or envelope"
        )
    raw = _load_only_source_report(
        report,
        expected_name="raw_fleet_health_probe",
        snapshot_context=snapshot_context,
    )
    _exact_keys(
        raw,
        {
            "schema_version",
            "kind",
            "passed",
            "immutable_sha256",
            "server_pool_root",
            "captured_timestamp",
            "replicas",
            "referenced_artifacts",
        },
        context="raw fleet health probe",
    )
    pool_root = str(Path(control["immutable"]["server_pool_root"]).resolve())
    if (
        raw.get("schema_version") != 1
        or raw.get("kind") != "raw_fleet_health_probe"
        or raw.get("passed") is not True
        or raw.get("immutable_sha256") != control["immutable_sha256"]
        or raw.get("server_pool_root") != pool_root
        or raw.get("captured_timestamp") != metrics.get("captured_timestamp")
    ):
        raise ReadinessError("raw fleet health probe identity differs from its wrapper")
    rows = raw.get("replicas")
    references = raw.get("referenced_artifacts")
    if not isinstance(rows, list) or len(rows) != 22:
        raise ReadinessError("raw fleet health probe must contain exactly 22 replicas")
    if not isinstance(references, list) or len(references) != 22:
        raise ReadinessError(
            "raw fleet health probe must bind exactly 22 registrations"
        )
    reference_by_name = {
        reference.get("name"): reference
        for reference in references
        if isinstance(reference, dict)
    }
    if len(reference_by_name) != 22:
        raise ReadinessError("raw fleet registration references are not unique")

    try:
        fleet_contract = json.loads(
            Path(str(control["immutable"]["fleet_contract_path"])).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"cannot parse pinned fleet topology: {exc}") from exc
    expected_replicas: dict[str, dict[str, Any]] = {}
    for profile in (
        fleet_contract.get("profiles", []) if isinstance(fleet_contract, dict) else []
    ):
        if not isinstance(profile, dict) or not isinstance(
            profile.get("replicas"), list
        ):
            raise ReadinessError("pinned fleet topology contains a malformed profile")
        for replica in profile["replicas"]:
            if not isinstance(replica, dict) or not isinstance(
                replica.get("replica_id"), str
            ):
                raise ReadinessError(
                    "pinned fleet topology contains a malformed replica"
                )
            expected_replicas[replica["replica_id"]] = {
                "serving_profile": profile.get("serving_profile"),
                "partition": replica.get("partition"),
                "replica_index": replica.get("replica_index"),
                "model_revision": profile.get("model_revision"),
                "tokenizer_id": profile.get("tokenizer_id"),
                "tokenizer_revision": profile.get("tokenizer_revision"),
            }
    if len(expected_replicas) != 22:
        raise ReadinessError(
            "pinned fleet topology does not enumerate exactly 22 replicas"
        )

    observed_ids: set[str] = set()
    profile_counts: dict[str, int] = {}
    probe_ages: list[float] = []
    replica_fields = {
        "replica_id",
        "serving_profile",
        "slurm_job_id",
        "partition",
        "node",
        "host",
        "port",
        "spooled_provenance",
        "http",
        "registry_path",
        "registry_sha256",
    }
    provenance_fields = {
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
    http_fields = {
        "health_status",
        "models_status",
        "model_ids",
        "expected_model",
        "probe_started_timestamp",
        "probe_completed_timestamp",
        "healthy",
    }
    captured = float(raw["captured_timestamp"])
    for row in rows:
        if not isinstance(row, dict):
            raise ReadinessError("raw fleet probe contains a non-object replica")
        _exact_keys(row, replica_fields, context="raw fleet replica")
        replica_id = row.get("replica_id")
        expected = expected_replicas.get(str(replica_id))
        if expected is None or replica_id in observed_ids:
            raise ReadinessError(
                f"raw fleet probe has unknown/duplicate replica {replica_id!r}"
            )
        observed_ids.add(str(replica_id))
        profile = row.get("serving_profile")
        if (
            profile != expected["serving_profile"]
            or row.get("partition") != expected["partition"]
            or not str(row.get("slurm_job_id", "")).isdigit()
            or not isinstance(row.get("node"), str)
            or not row["node"]
            or row.get("host") != row.get("node")
            or not isinstance(row.get("port"), int)
            or isinstance(row.get("port"), bool)
            or not 1 <= row["port"] <= 65_535
        ):
            raise ReadinessError(
                f"raw fleet scheduler identity drifted for {replica_id}"
            )
        profile_counts[str(profile)] = profile_counts.get(str(profile), 0) + 1

        reference_name = f"registry_{replica_id}"
        reference = reference_by_name.get(reference_name)
        if (
            not isinstance(reference, dict)
            or reference.get("path") != row.get("registry_path")
            or reference.get("sha256") != row.get("registry_sha256")
        ):
            raise ReadinessError(f"raw fleet registry binding drifted for {replica_id}")
        _, registry = _validate_referenced_artifact(
            reference,
            context=f"fleet registration {replica_id}",
            verified=set(),
            active=set(),
            snapshot_context=snapshot_context,
        )
        if (
            not isinstance(registry, dict)
            or str(registry.get("replica_id")) != replica_id
            or registry.get("serving_profile") != profile
            or str(registry.get("slurm_job_id")) != str(row["slurm_job_id"])
            or registry.get("host") != row["host"]
            or registry.get("port") != row["port"]
            or registry.get("release_id") != control["immutable"]["release_id"]
            or registry.get("environment_hash")
            != control["immutable"]["serving_environment_sha256"]
            or registry.get("model_contract_sha256")
            != control["immutable"]["model_contract_sha256"]
            or registry.get("fleet_contract_sha256")
            != control["immutable"]["fleet_contract_sha256"]
        ):
            raise ReadinessError(
                f"fleet registration provenance drifted for {replica_id}"
            )

        provenance = row.get("spooled_provenance")
        if not isinstance(provenance, dict):
            raise ReadinessError(f"spooled provenance is missing for {replica_id}")
        _exact_keys(provenance, provenance_fields, context="spooled fleet provenance")
        if (
            provenance.get("run_root") != pool_root
            or provenance.get("server_pool_id") != "schema5-v1"
            or str(provenance.get("replica_id")) != replica_id
            or provenance.get("replica_index") != expected["replica_index"]
            or provenance.get("release_id") != control["immutable"]["release_id"]
            or provenance.get("environment_hash")
            != control["immutable"]["serving_environment_sha256"]
            or provenance.get("model_revision") != expected["model_revision"]
            or provenance.get("tokenizer_id") != expected["tokenizer_id"]
            or provenance.get("tokenizer_revision") != expected["tokenizer_revision"]
            or provenance.get("model_contract_sha256")
            != control["immutable"]["model_contract_sha256"]
            or provenance.get("fleet_contract_sha256")
            != control["immutable"]["fleet_contract_sha256"]
        ):
            raise ReadinessError(f"spooled fleet provenance drifted for {replica_id}")

        http = row.get("http")
        if not isinstance(http, dict):
            raise ReadinessError(f"HTTP health evidence is missing for {replica_id}")
        _exact_keys(http, http_fields, context="fleet HTTP probe")
        completed = http.get("probe_completed_timestamp")
        started = http.get("probe_started_timestamp")
        if (
            http.get("healthy") is not True
            or http.get("health_status") != 200
            or http.get("models_status") != 200
            or not isinstance(http.get("model_ids"), list)
            or http.get("expected_model") not in http["model_ids"]
            or not isinstance(started, (int, float))
            or isinstance(started, bool)
            or not isinstance(completed, (int, float))
            or isinstance(completed, bool)
            or not float(started) <= float(completed) <= captured
        ):
            raise ReadinessError(f"fleet HTTP probe failed for {replica_id}")
        probe_ages.append(captured - float(completed))
    if observed_ids != set(expected_replicas):
        raise ReadinessError("raw fleet probe does not cover the frozen replica set")
    if profile_counts != EXPECTED_FLEET_PROFILES:
        raise ReadinessError(
            "raw fleet probe profile counts differ from the frozen topology"
        )
    if not probe_ages or max(probe_ages) != metrics.get("max_heartbeat_age_seconds"):
        raise ReadinessError(
            "fleet freshness metric is not derived from raw HTTP probes"
        )


_CONTEXT_ARTIFACT_CONTRACTS: dict[str, dict[str, Any]] = {
    "dense_peer_context_audit": {
        "source_name": "raw_dense_peer_context_audit",
        "run_id": "full_sweep_agent_count_7_schema5_v1",
        "filters": {
            "n_agents": [7],
            "reasoning": ["b2048", "b8192", "unlimited"],
            "prompt_levels": [3],
            "topologies": ["decentralized"],
            "context_levels": ["plus_cot"],
        },
        "selected_cells": 216,
        "audited_requests": 43_092,
    },
    "seven_agent_context_audit": {
        "source_name": "raw_seven_agent_context_audit",
        "run_id": "full_sweep_agent_count_7_schema5_v1",
        "filters": {
            "n_agents": [7],
            "reasoning": ["unlimited"],
            "prompt_levels": [],
            "topologies": [],
            "context_levels": ["plus_cot"],
        },
        "selected_cells": 288,
        "audited_requests": 57_456,
    },
}


def _validate_context_artifacts(
    control: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, tuple[Path, Mapping[str, Any] | None]],
    *,
    snapshot_context: _SnapshotValidationContext,
) -> None:
    total_failures = 0
    total_truncations = 0
    margins: list[int] = []
    requests: dict[str, int] = {}
    seven_run = _run_pins_by_id(control["immutable"])[
        "full_sweep_agent_count_7_schema5_v1"
    ]
    for name, contract in _CONTEXT_ARTIFACT_CONTRACTS.items():
        wrapper = _validate_artifact_identity(
            control,
            artifacts[name][1],
            name=name,
            exact_fields={
                "schema_version",
                "kind",
                "passed",
                "immutable_sha256",
                "run_id",
                "filters",
                "all_routed_profiles",
                "metrics",
                "referenced_artifacts",
            },
        )
        expected_metrics = {
            "selected_cells": contract["selected_cells"],
            "audited_requests": contract["audited_requests"],
            "failed_requests": 0,
            "failed_cells": 0,
            "minimum_context_headroom_tokens": 1_287,
            "truncation_incidents": 0,
        }
        if (
            wrapper.get("run_id") != contract["run_id"]
            or wrapper.get("filters") != contract["filters"]
            or wrapper.get("all_routed_profiles") is not True
            or wrapper.get("metrics") != expected_metrics
        ):
            raise ReadinessError(
                f"{name} selection or metrics differ from the exact audit"
            )
        source = _load_only_source_report(
            wrapper,
            expected_name=str(contract["source_name"]),
            snapshot_context=snapshot_context,
        )
        summary = source.get("summary")
        manifest = source.get("manifest")
        if (
            source.get("schema_version") != 3
            or source.get("audit") != "all_routed_profiles_context_capacity"
            or source.get("run_id") != contract["run_id"]
            or source.get("filters") != contract["filters"]
            or source.get("all_routed_profiles") is not True
            or not isinstance(summary, dict)
            or summary.get("passed") is not True
            or not isinstance(manifest, dict)
            or manifest.get("path") != str(Path(seven_run["manifest_path"]).resolve())
            or manifest.get("sha256") != seven_run["manifest_sha256"]
            or manifest.get("cells") != seven_run["cell_count"]
            or source.get("failure_groups") != []
            or source.get("failure_examples") != []
        ):
            raise ReadinessError(f"raw context report for {name} has invalid identity")
        for field in (
            "selected_cells",
            "audited_requests",
            "failed_requests",
            "failed_cells",
            "minimum_context_headroom_tokens",
        ):
            if summary.get(field) != expected_metrics[field]:
                raise ReadinessError(
                    f"raw context report {name} {field} differs from its wrapper"
                )
        requests[name] = int(expected_metrics["audited_requests"])
        total_failures += int(expected_metrics["failed_requests"])
        total_truncations += int(expected_metrics["truncation_incidents"])
        margins.append(int(expected_metrics["minimum_context_headroom_tokens"]))
    derived = {
        "dense_peer_requests": requests["dense_peer_context_audit"],
        "seven_agent_requests": requests["seven_agent_context_audit"],
        "failed_preflights": total_failures,
        "truncation_incidents": total_truncations,
        "minimum_context_margin_tokens": min(margins),
    }
    if dict(metrics) != derived:
        raise ReadinessError(
            "outer context metrics are not derived from both raw audits"
        )


def _validate_smoke_artifacts(
    control: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, tuple[Path, Mapping[str, Any] | None]],
) -> None:
    suites = {
        "long_32b_smoke": ("schema5_smoke_32b_long_v1", 15),
        "selective_long_smoke": ("schema5_smoke_selective_long_v1", 20),
        "standard_canary_smoke": ("schema5_smoke_standard_canaries_v1", 6),
    }
    expected_profiles = {
        "long_32b_smoke": {"32B-long": 15},
        "selective_long_smoke": {
            "0.6B": 1,
            "0.6B-long": 3,
            "1.7B": 1,
            "1.7B-long": 3,
            "4B": 1,
            "4B-long": 3,
            "8B": 1,
            "8B-long": 3,
            "14B": 1,
            "14B-long": 3,
        },
        "standard_canary_smoke": {
            "0.6B": 1,
            "1.7B": 1,
            "4B": 1,
            "8B": 1,
            "14B": 1,
            "32B": 1,
        },
    }
    provenance_expected = {
        "release_id": control["immutable"]["release_id"],
        "git_commit": control["immutable"]["git_commit"],
        "source_tree_sha256": control["immutable"]["source_tree_sha256"],
        "harness_environment_sha256": control["immutable"][
            "harness_environment_sha256"
        ],
        "serving_environment_sha256": control["immutable"][
            "serving_environment_sha256"
        ],
        "model_contract_sha256": control["immutable"]["model_contract_sha256"],
        "fleet_contract_sha256": control["immutable"]["fleet_contract_sha256"],
        "server_pool_root": str(
            Path(control["immutable"]["server_pool_root"]).resolve()
        ),
        "rollout_generation": 1,
    }
    derived = {
        "long_32b_cells": 0,
        "selective_long_cells": 0,
        "standard_canary_cells": 0,
        "schema5_complete_cells": 0,
        "context_incidents": 0,
        "protocol_incidents": 0,
        "truncation_incidents": 0,
        "provenance_failures": 0,
    }
    metric_names = {
        "long_32b_smoke": "long_32b_cells",
        "selective_long_smoke": "selective_long_cells",
        "standard_canary_smoke": "standard_canary_cells",
    }
    exact_fields = {
        "schema_version",
        "kind",
        "passed",
        "immutable_sha256",
        "run_id",
        "expected_cells",
        "schema5_complete_cells",
        "context_incidents",
        "protocol_incidents",
        "truncation_incidents",
        "provenance_failures",
        "semantic_validation_failures",
        "top_level_length_censored_qids",
        "top_level_protocol_censored_qids",
        "auxiliary_length_censored_draws",
        "auxiliary_protocol_censored_draws",
        "concurrency",
        "resumable",
        "execution_halted",
        "execution_halt_reason",
        "provenance",
        "suite_identity",
        "cells",
    }
    for name, (run_id, expected_cells) in suites.items():
        suite = _validate_artifact_identity(
            control,
            artifacts[name][1],
            name=name,
            exact_fields=exact_fields,
        )
        zero_fields = (
            "context_incidents",
            "protocol_incidents",
            "truncation_incidents",
            "provenance_failures",
            "semantic_validation_failures",
            "top_level_length_censored_qids",
            "top_level_protocol_censored_qids",
            "auxiliary_length_censored_draws",
            "auxiliary_protocol_censored_draws",
        )
        cells = suite.get("cells")
        identity = suite.get("suite_identity")
        identity_fields = {
            "run_id",
            "cell_count",
            "manifest_sha256",
            "benchmark_contracts_sha256",
            "artifact_policy_sha256",
            "lineage_sha256",
            "serving_profile_counts",
            "estimand_excluded",
        }
        valid_identity = (
            isinstance(identity, dict)
            and set(identity) == identity_fields
            and identity.get("run_id") == run_id
            and identity.get("cell_count") == expected_cells
            and identity.get("serving_profile_counts") == expected_profiles[name]
            and identity.get("estimand_excluded") is True
            and all(
                _SHA256_RE.fullmatch(str(identity.get(field, ""))) is not None
                for field in (
                    "manifest_sha256",
                    "benchmark_contracts_sha256",
                    "artifact_policy_sha256",
                    "lineage_sha256",
                )
            )
        )
        if (
            suite.get("run_id") != run_id
            or suite.get("expected_cells") != expected_cells
            or suite.get("schema5_complete_cells") != expected_cells
            or any(suite.get(field) != 0 for field in zero_fields)
            or suite.get("concurrency") != 1
            or suite.get("resumable") is not True
            or suite.get("execution_halted") is not False
            or suite.get("execution_halt_reason") is not None
            or suite.get("provenance") != provenance_expected
            or not valid_identity
            or not isinstance(cells, list)
            or len(cells) != expected_cells
        ):
            raise ReadinessError(
                f"smoke suite {name} does not satisfy its exact contract"
            )
        cell_fields = {
            "cell_id",
            "status",
            "valid_qids",
            "expected_qids",
            "top_level_length_censored_qids",
            "top_level_protocol_censored_qids",
            "auxiliary_length_censored_draws",
            "auxiliary_protocol_censored_draws",
            "context_incidents",
            "protocol_incidents",
            "truncation_incidents",
            "provenance_failures",
            "provenance_errors",
            "semantic_errors",
            "failure",
        }
        cell_ids: set[str] = set()
        for cell in cells:
            if (
                not isinstance(cell, dict)
                or set(cell) != cell_fields
                or not isinstance(cell.get("cell_id"), str)
                or not cell.get("cell_id")
            ):
                raise ReadinessError(
                    f"smoke suite {name} contains an invalid cell report"
                )
            cell_ids.add(cell["cell_id"])
            if (
                cell.get("status") != "complete"
                or not isinstance(cell.get("expected_qids"), int)
                or isinstance(cell.get("expected_qids"), bool)
                or cell.get("expected_qids", 0) <= 0
                or cell.get("valid_qids") != cell.get("expected_qids")
                or any(cell.get(field) != 0 for field in zero_fields if field in cell)
                or cell.get("provenance_errors") != []
                or cell.get("semantic_errors") != []
                or cell.get("failure") is not None
            ):
                raise ReadinessError(
                    f"smoke suite {name} has a non-canonical cell result"
                )
        if len(cell_ids) != expected_cells:
            raise ReadinessError(f"smoke suite {name} repeats a cell ID")
        derived[metric_names[name]] = expected_cells
        derived["schema5_complete_cells"] += expected_cells
        for field in (
            "context_incidents",
            "protocol_incidents",
            "truncation_incidents",
            "provenance_failures",
        ):
            derived[field] += int(suite[field])
    if dict(metrics) != derived:
        raise ReadinessError("outer smoke metrics are not derived from suite artifacts")


def _validate_email_artifact(
    control: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, tuple[Path, Mapping[str, Any] | None]],
    *,
    snapshot_context: _SnapshotValidationContext,
) -> None:
    name = "email_delivery_receipt"
    receipt = _validate_artifact_identity(
        control,
        artifacts[name][1],
        name=name,
        exact_fields={
            "schema_version",
            "kind",
            "passed",
            "immutable_sha256",
            "metrics",
            "referenced_artifacts",
        },
    )
    if receipt.get("metrics") != dict(metrics):
        raise ReadinessError("email receipt metrics differ from the outer envelope")
    raw = _load_only_source_report(
        receipt,
        expected_name="mail_submission_receipt",
        snapshot_context=snapshot_context,
    )
    _exact_keys(
        raw,
        {
            "schema_version",
            "kind",
            "passed",
            "immutable_sha256",
            "metrics",
            "submitted_timestamp",
            "subject",
            "message_sha256",
            "stdout",
            "stderr",
            "referenced_artifacts",
        },
        context="raw mail submission receipt",
    )
    submitted = raw.get("submitted_timestamp")
    if (
        raw.get("schema_version") != 1
        or raw.get("kind") != "mail_submission_receipt"
        or raw.get("passed") is not True
        or raw.get("immutable_sha256") != control["immutable_sha256"]
        or raw.get("metrics") != dict(metrics)
        or not isinstance(submitted, (int, float))
        or isinstance(submitted, bool)
        or submitted < 0
        or raw.get("subject") != "[agents-scaling] schema-5 readiness delivery test"
        or not isinstance(raw.get("message_sha256"), str)
        or len(raw["message_sha256"]) != 64
        or not isinstance(raw.get("stdout"), str)
        or not isinstance(raw.get("stderr"), str)
        or raw.get("referenced_artifacts") != []
    ):
        raise ReadinessError(
            "raw mail submission receipt does not prove the outer metrics"
        )


class _SnapshotValidationContext:
    """One readiness-validation pass over the two large sealed snapshots.

    ``full`` is used only while attesting generation zero or creating a new rollout
    seal. Normal dispatcher, controller, and status paths construct a compact context
    from the current generation seal and never hash snapshot payload bytes.
    """

    def __init__(
        self,
        *,
        full: bool,
        seal: Mapping[str, Any] | None = None,
        allow_inventory_lookup: bool = False,
        allowed_members: Sequence[Mapping[str, str]] = (),
    ) -> None:
        self.full = full
        self.seal = dict(seal) if seal is not None else None
        self.allow_inventory_lookup = allow_inventory_lookup
        self.proofs: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.members: dict[tuple[str, str], dict[str, str]] = {}
        self.inventories: dict[str, dict[str, str]] = {}
        self.allowed_members = {
            (str(row["snapshot_root"]), str(row["logical_path"])): dict(row)
            for row in allowed_members
        }

    def add_metadata(self, path: Path) -> None:
        try:
            row = snapshot_integrity.metadata_entry(path)
        except snapshot_integrity.SnapshotIntegrityError as exc:
            raise ReadinessError(str(exc)) from exc
        self.add_metadata_record(row)

    def add_metadata_record(self, row: Mapping[str, Any]) -> None:
        """Add metadata already proven from the descriptor used for byte hashing."""

        try:
            canonical = snapshot_integrity.canonical_metadata_entries([row])[0]
        except snapshot_integrity.SnapshotIntegrityError as exc:
            raise ReadinessError(str(exc)) from exc
        key = str(canonical["path"])
        previous = self.metadata.get(key)
        if previous is not None and previous != canonical:
            raise ReadinessError(
                f"sealed snapshot metadata changed during validation: {key}"
            )
        self.metadata[key] = canonical

    def sealed_snapshot(self, *, path: Path, digest: str) -> Mapping[str, Any] | None:
        if self.seal is None:
            return None
        matches = [
            row
            for row in self.seal.get("snapshots", [])
            if isinstance(row, dict)
            and row.get("attestation_path") == str(path.resolve())
            and row.get("attestation_sha256") == digest
        ]
        if len(matches) != 1:
            raise ReadinessError(
                f"snapshot attestation is not uniquely bound by generation seal: {path}"
            )
        return matches[0]

    def add_member(self, member: Mapping[str, str]) -> None:
        key = (str(member["snapshot_root"]), str(member["logical_path"]))
        previous = self.members.get(key)
        if previous is not None and previous != dict(member):
            raise ReadinessError(
                f"conflicting sealed snapshot membership proof: {key}"
            )
        self.members[key] = dict(member)


def _snapshot_regular_sha256(path: Path, *, context: str) -> str:
    try:
        return snapshot_integrity.sha256_file(path, description=context)
    except snapshot_integrity.SnapshotIntegrityError as exc:
        raise ReadinessError(str(exc)) from exc


def _snapshot_regular_sha256_with_metadata(
    path: Path, *, context: str
) -> tuple[str, dict[str, Any]]:
    try:
        return snapshot_integrity.sha256_file_with_metadata(
            path, description=context
        )
    except snapshot_integrity.SnapshotIntegrityError as exc:
        raise ReadinessError(str(exc)) from exc


def _validate_snapshot_external_attestation(
    path: Path,
    payload: Mapping[str, Any],
    *,
    snapshot_context: _SnapshotValidationContext,
) -> dict[str, Any]:
    """Verify the external envelope and every sealed snapshot control artifact."""
    cache_key = str(path.resolve())
    cached = snapshot_context.proofs.get(cache_key)
    if cached is not None:
        return cached
    _exact_keys(
        payload,
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
        },
        context=f"snapshot external attestation {path}",
    )
    if payload.get("schema_version") != 1:
        raise ReadinessError(f"snapshot external attestation has wrong schema: {path}")
    if payload.get("kind") != "recovery_snapshot_external_attestation":
        raise ReadinessError(
            f"snapshot artifact is not an external attestation: {path}"
        )
    if payload.get("passed") is not True:
        raise ReadinessError(f"snapshot external attestation did not pass: {path}")
    root_value = payload.get("snapshot_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ReadinessError(
            f"snapshot external attestation has non-absolute root: {path}"
        )
    supplied_root = Path(root_value)
    if supplied_root.is_symlink():
        raise ReadinessError(
            f"sealed snapshot root must not be a symlink: {supplied_root}"
        )
    root = supplied_root.resolve()
    if not root.is_dir():
        raise ReadinessError(f"sealed snapshot root is missing: {root}")
    if root.stat().st_mode & 0o222:
        raise ReadinessError(f"snapshot root is not read-only: {root}")
    controls = payload.get("control_artifacts")
    if not isinstance(controls, dict) or set(controls) != SNAPSHOT_CONTROL_FILENAMES:
        raise ReadinessError(
            f"snapshot external attestation must bind exactly "
            f"{sorted(SNAPSHOT_CONTROL_FILENAMES)}: {path}"
        )
    attestation_digest = sha256_file(path)
    sealed_row = snapshot_context.sealed_snapshot(
        path=path, digest=attestation_digest
    )
    # Running hot paths stop here.  The immutable seal was created from a full
    # payload/inventory validation and the short-lived metadata lease detects later
    # chmod/write/replace/hardlink/shape drift.  Reopening two ~201k-row inventories
    # (and their duplicate source copies) on every dispatcher/spec/successor check
    # would defeat the purpose of the compact lease.
    if not snapshot_context.full:
        if sealed_row is None:  # pragma: no cover - compact contexts always bind one
            raise ReadinessError("compact snapshot validation lacks a generation seal")
        for filename in sorted(SNAPSHOT_CONTROL_FILENAMES):
            record = controls[filename]
            if not isinstance(record, dict):
                raise ReadinessError(
                    f"invalid snapshot control record {filename}: {path}"
                )
            _exact_keys(
                record, {"sha256", "size"}, context=f"snapshot control {filename}"
            )
            if (
                _SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
                or not isinstance(record.get("size"), int)
                or isinstance(record.get("size"), bool)
                or record["size"] < 0
            ):
                raise ReadinessError(
                    f"invalid snapshot control identity for {filename}: {path}"
                )
        try:
            time.strptime(str(payload.get("attested_at")), "%Y-%m-%dT%H:%M:%SZ")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ReadinessError(
                f"snapshot attestation timestamp is invalid: {path}"
            ) from exc
        proof = {
            "attestation_path": str(path.resolve()),
            "attestation_sha256": attestation_digest,
            "snapshot_root": str(root),
            "snapshot_id": str(payload["snapshot_id"]),
            "file_count": payload.get("file_count"),
            "total_bytes": payload.get("total_bytes"),
            "inventory_sha256": sealed_row.get("inventory_sha256"),
            "control_artifacts": {
                name: dict(controls[name]) for name in sorted(controls)
            },
        }
        if proof != dict(sealed_row):
            raise ReadinessError(
                f"snapshot external attestation differs from generation seal: {root}"
            )
        snapshot_context.proofs[cache_key] = proof
        return proof
    for filename in sorted(SNAPSHOT_CONTROL_FILENAMES):
        record = controls[filename]
        if not isinstance(record, dict):
            raise ReadinessError(f"invalid snapshot control record {filename}: {path}")
        _exact_keys(record, {"sha256", "size"}, context=f"snapshot control {filename}")
        artifact = root / filename
        if not artifact.is_file() or artifact.is_symlink():
            raise ReadinessError(
                f"snapshot control artifact is missing/unsafe: {artifact}"
            )
        digest = record.get("sha256")
        size = record.get("size")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ReadinessError(f"invalid snapshot control hash for {artifact}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ReadinessError(f"invalid snapshot control size for {artifact}")
        observed_digest, metadata = _snapshot_regular_sha256_with_metadata(
            artifact, context="sealed snapshot control artifact"
        )
        if metadata["size"] != size or observed_digest != digest:
            raise ReadinessError(f"snapshot control artifact drifted: {artifact}")
        snapshot_context.add_metadata_record(metadata)

    try:
        marker = json.loads(
            (root / "SNAPSHOT_COMPLETE.json").read_text(encoding="utf-8")
        )
        catalog = json.loads(
            (root / "SNAPSHOT_CATALOG.json").read_text(encoding="utf-8")
        )
        source_inventory = (root / "SOURCE_INVENTORY.sha256").read_bytes()
        snapshot_inventory = (root / "SNAPSHOT_INVENTORY.sha256").read_bytes()
        directory_lines = (
            (root / "DIRECTORY_INVENTORY.txt").read_text(encoding="utf-8").splitlines()
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"cannot parse sealed snapshot controls: {exc}") from exc
    if not isinstance(marker, dict) or not isinstance(catalog, dict):
        raise ReadinessError("sealed snapshot marker/catalog must be JSON objects")
    if source_inventory != snapshot_inventory:
        raise ReadinessError("sealed snapshot source and copy inventories differ")
    inventory_sha = hashlib.sha256(snapshot_inventory).hexdigest()
    if snapshot_context.full:
        seen_paths: set[str] = set()
        file_count = 0
        total_bytes = 0
        try:
            inventory_lines = snapshot_inventory.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ReadinessError(
                f"sealed snapshot inventory is not UTF-8: {exc}"
            ) from exc
        for line_number, raw_line in enumerate(inventory_lines, start=1):
            digest, separator, logical = raw_line.partition("  ")
            relative = Path(logical)
            if (
                not separator
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or not logical
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != logical
                or logical in SNAPSHOT_CONTROL_FILENAMES
                or logical in seen_paths
            ):
                raise ReadinessError(
                    f"invalid sealed snapshot inventory row {line_number}: {raw_line!r}"
                )
            candidate = root / relative
            if candidate.is_symlink() or not candidate.is_file():
                raise ReadinessError(
                    f"sealed snapshot payload is missing/unsafe: {candidate}"
                )
            observed_digest, metadata = _snapshot_regular_sha256_with_metadata(
                candidate, context="sealed snapshot payload"
            )
            if observed_digest != digest:
                raise ReadinessError(f"sealed snapshot payload drifted: {candidate}")
            snapshot_context.add_metadata_record(metadata)
            seen_paths.add(logical)
            file_count += 1
            total_bytes += int(metadata["size"])
        snapshot_context.inventories[str(root)] = {
            raw_line.partition("  ")[2]: raw_line.partition("  ")[0]
            for raw_line in inventory_lines
        }
        if directory_lines != sorted(set(directory_lines)):
            raise ReadinessError(
                "sealed snapshot directory inventory is not sorted and unique"
            )
        for logical in directory_lines:
            relative = Path(logical)
            candidate = root / relative
            if (
                not logical
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != logical
                or candidate.is_symlink()
                or not candidate.is_dir()
                or candidate.stat().st_mode & 0o222
            ):
                raise ReadinessError(
                    f"sealed snapshot directory is invalid: {logical!r}"
                )
            snapshot_context.add_metadata(candidate)
        expected_files = seen_paths | set(SNAPSHOT_CONTROL_FILENAMES)
        expected_directories = set(directory_lines)
        observed_files: set[str] = set()
        observed_directories: set[str] = set()
        for directory, names, filenames in os.walk(
            root, topdown=True, followlinks=False
        ):
            names.sort()
            filenames.sort()
            directory_path = Path(directory)
            for name in names:
                candidate = directory_path / name
                if candidate.is_symlink():
                    raise ReadinessError(
                        f"symlink is forbidden in sealed snapshot: {candidate}"
                    )
                observed_directories.add(candidate.relative_to(root).as_posix())
            for name in filenames:
                candidate = directory_path / name
                if candidate.is_symlink() or not candidate.is_file():
                    raise ReadinessError(
                        f"unsafe regular entry in sealed snapshot: {candidate}"
                    )
                observed_files.add(candidate.relative_to(root).as_posix())
        if observed_files != expected_files or observed_directories != expected_directories:
            raise ReadinessError(
                "sealed snapshot topology differs from its file/directory inventories: "
                f"extra_files={sorted(observed_files - expected_files)[:5]}, "
                f"missing_files={sorted(expected_files - observed_files)[:5]}, "
                f"extra_directories={sorted(observed_directories - expected_directories)[:5]}, "
                f"missing_directories={sorted(expected_directories - observed_directories)[:5]}"
            )
        snapshot_context.add_metadata(root)
    if (
        marker.get("schema_version") != 1
        or marker.get("snapshot_id") != catalog.get("snapshot_id")
        or marker.get("snapshot_id") != payload.get("snapshot_id")
        or marker.get("file_count") != file_count
        or catalog.get("file_count") != file_count
        or payload.get("file_count") != file_count
        or marker.get("total_bytes") != total_bytes
        or catalog.get("total_bytes") != total_bytes
        or payload.get("total_bytes") != total_bytes
        or marker.get("snapshot_inventory_sha256") != inventory_sha
        or catalog.get("source_inventory_sha256") != inventory_sha
        or catalog.get("snapshot_inventory_sha256") != inventory_sha
        or marker.get("verified") is not True
        or marker.get("read_only") is not True
    ):
        raise ReadinessError("snapshot envelope claims differ from sealed inventories")
    completed_at = marker.get("completed_at")
    attested_at = payload.get("attested_at")
    try:
        completed_epoch = time.mktime(
            time.strptime(str(completed_at), "%Y-%m-%dT%H:%M:%SZ")
        )
        attested_epoch = time.mktime(
            time.strptime(str(attested_at), "%Y-%m-%dT%H:%M:%SZ")
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReadinessError(
            "snapshot completion/attestation timestamps are invalid"
        ) from exc
    if attested_epoch < completed_epoch:
        raise ReadinessError("snapshot attestation predates sealed snapshot completion")
    proof = {
        "attestation_path": str(path.resolve()),
        "attestation_sha256": attestation_digest,
        "snapshot_root": str(root),
        "snapshot_id": str(payload["snapshot_id"]),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "inventory_sha256": inventory_sha,
        "control_artifacts": {
            name: dict(controls[name]) for name in sorted(controls)
        },
    }
    if sealed_row is not None and proof != dict(sealed_row):
        raise ReadinessError(
            f"snapshot controls differ from generation seal: {root}"
        )
    snapshot_context.proofs[cache_key] = proof
    return proof


def _validate_sealed_snapshot_member(
    artifact_path: Path,
    payload: Mapping[str, Any],
    *,
    snapshot_context: _SnapshotValidationContext,
) -> None:
    raw = payload.get("sealed_snapshot_member")
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ReadinessError(
            f"sealed_snapshot_member must be an object: {artifact_path}"
        )
    _exact_keys(
        raw,
        {"snapshot_id", "snapshot_root", "logical_path", "sha256"},
        context=f"sealed_snapshot_member {artifact_path}",
    )
    snapshot_id = raw.get("snapshot_id")
    root_value = raw.get("snapshot_root")
    logical = raw.get("logical_path")
    digest = raw.get("sha256")
    if (
        not isinstance(snapshot_id, str)
        or not snapshot_id
        or not isinstance(root_value, str)
        or not Path(root_value).is_absolute()
        or not isinstance(logical, str)
        or not logical
        or not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
    ):
        raise ReadinessError(
            f"sealed_snapshot_member identity is invalid: {artifact_path}"
        )
    supplied_root = Path(root_value)
    if supplied_root.is_symlink():
        raise ReadinessError(
            f"sealed_snapshot_member root is symlinked: {supplied_root}"
        )
    root = supplied_root.resolve()
    if str(root) != root_value:
        raise ReadinessError(
            f"sealed_snapshot_member root is not canonical: {root_value}"
        )
    relative = Path(logical)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != logical
        or logical in SNAPSHOT_CONTROL_FILENAMES
    ):
        raise ReadinessError(
            f"sealed_snapshot_member logical path is unsafe: {logical!r}"
        )
    matching_proofs = [
        proof
        for proof in snapshot_context.proofs.values()
        if proof.get("snapshot_root") == str(root)
        and proof.get("snapshot_id") == snapshot_id
    ]
    if len(matching_proofs) != 1:
        raise ReadinessError(
            "sealed_snapshot_member does not address exactly one attested snapshot"
        )
    member = {
        "snapshot_id": snapshot_id,
        "snapshot_root": str(root),
        "logical_path": logical,
        "sha256": digest,
    }
    if not snapshot_context.full:
        assert snapshot_context.seal is not None
        seal_matches = [
            row
            for row in snapshot_context.seal.get("sealed_members", [])
            if isinstance(row, dict) and row == member
        ]
        allowed = snapshot_context.allowed_members.get(
            (str(root), logical)
        ) == member
        if len(seal_matches) == 1 or allowed:
            snapshot_context.add_member(member)
            return
        if not snapshot_context.allow_inventory_lookup:
            raise ReadinessError(
                "sealed_snapshot_member is not bound by the rollout seal/readiness proof"
            )
        inventory = snapshot_context.inventories.get(str(root))
        if inventory is None:
            inventory_path = root / "SNAPSHOT_INVENTORY.sha256"
            try:
                raw_inventory = snapshot_integrity.read_regular_bytes(
                    inventory_path,
                    description="sealed snapshot membership inventory",
                )
            except snapshot_integrity.SnapshotIntegrityError as exc:
                raise ReadinessError(str(exc)) from exc
            inventory_digest = hashlib.sha256(raw_inventory).hexdigest()
            expected_control_digest = matching_proofs[0]["control_artifacts"][
                "SNAPSHOT_INVENTORY.sha256"
            ]["sha256"]
            if (
                inventory_digest != matching_proofs[0]["inventory_sha256"]
                or inventory_digest != expected_control_digest
            ):
                raise ReadinessError(
                    "sealed_snapshot_member inventory differs from generation seal"
                )
            try:
                inventory_lines = raw_inventory.decode("utf-8").splitlines()
            except UnicodeDecodeError as exc:
                raise ReadinessError(
                    "sealed_snapshot_member inventory is not UTF-8"
                ) from exc
            inventory = {}
            for raw_line in inventory_lines:
                row_digest, separator, row_logical = raw_line.partition("  ")
                if (
                    not separator
                    or _SHA256_RE.fullmatch(row_digest) is None
                    or not row_logical
                    or row_logical in inventory
                ):
                    raise ReadinessError(
                        "sealed snapshot membership inventory is malformed"
                    )
                inventory[row_logical] = row_digest
            snapshot_context.inventories[str(root)] = inventory
        if inventory.get(logical) != digest:
            raise ReadinessError(
                "sealed_snapshot_member logical path/hash is absent from snapshot inventory"
            )
        snapshot_context.add_member(member)
        return
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ReadinessError(
            f"sealed_snapshot_member payload is missing/unsafe: {candidate}"
        )
    observed_digest, metadata = _snapshot_regular_sha256_with_metadata(
        candidate, context="sealed snapshot selected member"
    )
    if observed_digest != digest:
        raise ReadinessError(
            f"sealed_snapshot_member payload hash drifted: {candidate}"
        )
    inventory = snapshot_context.inventories.get(str(root))
    if inventory is None or inventory.get(logical) != digest:
        raise ReadinessError(
            "sealed_snapshot_member logical path/hash is absent from snapshot inventory"
        )
    snapshot_context.add_metadata_record(metadata)
    snapshot_context.add_member(member)


def _validate_referenced_artifact(
    reference: Mapping[str, Any],
    *,
    context: str,
    verified: set[tuple[str, str]],
    active: set[tuple[str, str]],
    snapshot_context: _SnapshotValidationContext | None = None,
) -> tuple[Path, Mapping[str, Any] | None]:
    if snapshot_context is None:
        snapshot_context = _SnapshotValidationContext(full=True)
    _exact_keys(reference, {"name", "path", "sha256"}, context=context)
    name = reference.get("name")
    path_value = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(name, str) or not name:
        raise ReadinessError(f"{context} has invalid artifact name")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ReadinessError(f"{context} artifact path must be absolute")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ReadinessError(f"{context} has invalid SHA-256")
    supplied_path = Path(path_value)
    if supplied_path.is_symlink() or not supplied_path.is_file():
        raise ReadinessError(
            f"{context} artifact is missing or a symlink: {supplied_path}"
        )
    path = supplied_path.resolve()
    observed = sha256_file(path)
    if observed != expected_hash:
        raise ReadinessError(
            f"{context} artifact drifted: {path}; expected {expected_hash}, got {observed}"
        )
    identity = (str(path), observed)
    if identity in active:
        raise ReadinessError(f"recursive artifact cycle detected at {path}")
    if identity in verified:
        return path, None
    active.add(identity)
    try:
        nested_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        active.remove(identity)
        verified.add(identity)
        return path, None
    if not isinstance(nested_payload, dict):
        active.remove(identity)
        verified.add(identity)
        return path, None
    if nested_payload.get("kind") == "recovery_snapshot_external_attestation":
        _validate_snapshot_external_attestation(
            path, nested_payload, snapshot_context=snapshot_context
        )
    nested = nested_payload.get("referenced_artifacts")
    if nested is not None:
        if not isinstance(nested, list):
            raise ReadinessError(
                f"nested referenced_artifacts must be an array: {path}"
            )
        nested_names = [
            item.get("name") if isinstance(item, dict) else None for item in nested
        ]
        if len(nested_names) != len(set(nested_names)):
            raise ReadinessError(f"nested artifact names must be unique: {path}")
        for index, item in enumerate(nested):
            if not isinstance(item, dict):
                raise ReadinessError(
                    f"nested artifact reference {index} is not an object: {path}"
                )
            _validate_referenced_artifact(
                item,
                context=f"nested artifact {path}[{index}]",
                verified=verified,
                active=active,
                snapshot_context=snapshot_context,
            )
    _validate_sealed_snapshot_member(
        path, nested_payload, snapshot_context=snapshot_context
    )
    active.remove(identity)
    verified.add(identity)
    return path, nested_payload


def _validate_gate_metrics(
    control: Mapping[str, Any],
    gate: str,
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, tuple[Path, Mapping[str, Any] | None]],
    *,
    now: float | None,
    snapshot_context: _SnapshotValidationContext,
) -> None:
    if gate == "snapshot":
        expected = {
            "snapshot_count": 2,
            "pre_repair_verified": True,
            "legacy_consolidated_verified": True,
        }
        _exact_keys(metrics, set(expected), context="snapshot metrics")
        _exact_integer(metrics["snapshot_count"], 2, context="snapshot_count")
        _exact_boolean(
            metrics["pre_repair_verified"], True, context="pre_repair_verified"
        )
        _exact_boolean(
            metrics["legacy_consolidated_verified"],
            True,
            context="legacy_consolidated_verified",
        )
        snapshot_roots: set[str] = set()
        snapshot_ids: set[str] = set()
        for name in READINESS_ARTIFACT_NAMES[gate]:
            payload = artifacts[name][1]
            if not isinstance(payload, dict):
                raise ReadinessError(f"snapshot artifact {name} must be JSON")
            _validate_snapshot_external_attestation(
                artifacts[name][0],
                payload,
                snapshot_context=snapshot_context,
            )
            snapshot_roots.add(str(Path(str(payload["snapshot_root"])).resolve()))
            snapshot_ids.add(str(payload["snapshot_id"]))
        if len(snapshot_roots) != 2 or len(snapshot_ids) != 2:
            raise ReadinessError(
                "pre-repair and consolidated snapshots must have distinct roots and IDs"
            )
        return
    exact_metrics: dict[str, Any]
    if gate == "migrations":
        exact_metrics = {
            "protocol_incidents_total": 22,
            "protocol_already_reset": 22,
            "sealed_incident_qids": 1_064,
            "migrated_checkpoints": 47,
            "remaining_schema1_checkpoints": 0,
            "permanent_ledgers_archived": 3,
            "unresolved_permanent_ledgers": 0,
        }
    elif gate == "semantic_audit":
        exact_metrics = {
            "complete_cells": 740,
            "active_validated_qids": 888_068,
            "corrupt_cells": 0,
            "permanent_cells": 0,
            "malformed_lines": 0,
            "duplicate_qids": 0,
            "unexpected_qids": 0,
            "repair_count": 0,
        }
    elif gate == "fleet":
        expected_keys = {
            "logical_replicas",
            "allocated_gpus",
            "healthy_replicas",
            "unhealthy_replicas",
            "profile_replicas",
            "revision_mismatches",
            "missing_profiles",
            "stale_registrations",
            "max_heartbeat_age_seconds",
            "captured_timestamp",
            "model_contract_sha256",
            "fleet_contract_sha256",
        }
        _exact_keys(metrics, expected_keys, context="fleet metrics")
        for key, expected in {
            "logical_replicas": 22,
            "allocated_gpus": 24,
            "healthy_replicas": 22,
            "unhealthy_replicas": 0,
            "revision_mismatches": 0,
            "missing_profiles": 0,
            "stale_registrations": 0,
        }.items():
            _exact_integer(metrics[key], expected, context=f"fleet {key}")
        if metrics.get("profile_replicas") != EXPECTED_FLEET_PROFILES:
            raise ReadinessError(
                "fleet profile replica counts do not match the 22-replica contract"
            )
        if (
            metrics.get("model_contract_sha256")
            != control["immutable"]["model_contract_sha256"]
        ):
            raise ReadinessError(
                "fleet model-contract revision/hash does not match immutable pins"
            )
        if (
            metrics.get("fleet_contract_sha256")
            != control["immutable"]["fleet_contract_sha256"]
        ):
            raise ReadinessError("fleet-contract hash does not match immutable pins")
        age = metrics.get("max_heartbeat_age_seconds")
        if (
            not isinstance(age, (int, float))
            or isinstance(age, bool)
            or not 0 <= age <= 600
        ):
            raise ReadinessError(
                "fleet max_heartbeat_age_seconds must be within [0, 600]"
            )
        captured = metrics.get("captured_timestamp")
        if (
            not isinstance(captured, (int, float))
            or isinstance(captured, bool)
            or captured < 0
        ):
            raise ReadinessError("fleet captured_timestamp is invalid")
        if now is not None and not 0 <= now - float(captured) <= 600:
            raise ReadinessError(
                "fleet readiness evidence is not recent (<=600 seconds)"
            )
        fleet_contract = artifacts["fleet_contract"][0]
        if (
            str(fleet_contract)
            != str(Path(control["immutable"]["fleet_contract_path"]).resolve())
            or sha256_file(fleet_contract)
            != control["immutable"]["fleet_contract_sha256"]
        ):
            raise ReadinessError(
                "fleet evidence does not reference the exact immutable contract"
            )
        _validate_fleet_artifact(
            control,
            metrics,
            artifacts,
            snapshot_context=snapshot_context,
        )
        return
    elif gate == "context_audit":
        exact_metrics = {
            "dense_peer_requests": 43_092,
            "seven_agent_requests": 57_456,
            "failed_preflights": 0,
            "truncation_incidents": 0,
        }
        expected_keys = set(exact_metrics) | {"minimum_context_margin_tokens"}
        _exact_keys(metrics, expected_keys, context="context-audit metrics")
        margin = metrics.get("minimum_context_margin_tokens")
        if not isinstance(margin, int) or isinstance(margin, bool) or margin != 1_287:
            raise ReadinessError(
                "minimum context margin must equal the audited 1,287 tokens"
            )
        for key, expected in exact_metrics.items():
            _exact_integer(metrics.get(key), expected, context=f"context_audit {key}")
        _validate_context_artifacts(
            control,
            metrics,
            artifacts,
            snapshot_context=snapshot_context,
        )
        return
    elif gate == "smoke_runs":
        exact_metrics = {
            "long_32b_cells": 15,
            "selective_long_cells": 20,
            "standard_canary_cells": 6,
            "schema5_complete_cells": 41,
            "context_incidents": 0,
            "protocol_incidents": 0,
            "truncation_incidents": 0,
            "provenance_failures": 0,
        }
    elif gate == "email_test":
        expected_keys = {"recipient", "delivery_succeeded", "returncode"}
        _exact_keys(metrics, expected_keys, context="email-test metrics")
        if metrics.get("recipient") != control.get("alert_email"):
            raise ReadinessError(
                "email readiness recipient does not match control alert_email"
            )
        _exact_boolean(
            metrics.get("delivery_succeeded"), True, context="email delivery"
        )
        _exact_integer(metrics.get("returncode"), 0, context="email returncode")
        _validate_email_artifact(
            control,
            metrics,
            artifacts,
            snapshot_context=snapshot_context,
        )
        return
    else:
        raise ReadinessError(f"no typed readiness contract exists for gate {gate!r}")
    _exact_keys(metrics, set(exact_metrics), context=f"{gate} metrics")
    for key, expected in exact_metrics.items():
        if isinstance(expected, bool):
            _exact_boolean(metrics.get(key), expected, context=f"{gate} {key}")
        elif isinstance(expected, int):
            _exact_integer(metrics.get(key), expected, context=f"{gate} {key}")
        elif metrics.get(key) != expected:
            raise ReadinessError(f"{gate} {key} must equal {expected!r}")
    if gate == "migrations":
        _validate_migration_artifacts(
            control,
            metrics,
            artifacts,
            snapshot_context=snapshot_context,
        )
    elif gate == "semantic_audit":
        _validate_semantic_artifact(
            control,
            metrics,
            artifacts,
            snapshot_context=snapshot_context,
        )
    elif gate == "smoke_runs":
        _validate_smoke_artifacts(control, metrics, artifacts)


def _validate_scheduler_gate_payload(
    control: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "gate",
            "immutable_sha256",
            "captured_at",
            "captured_timestamp",
            "all",
            "no_admit",
            "scheduler",
            "roles",
            "ambiguous_tokens",
            "unmappable_job_ids",
            "malformed_token_job_ids",
            "referenced_job_ids_absent_from_window",
            "errors",
            "passed",
        },
        context="scheduler reconciliation evidence",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("gate") != "scheduler_reconciliation"
    ):
        raise ReadinessError("scheduler reconciliation evidence has invalid identity")
    if payload.get("immutable_sha256") != control["immutable_sha256"]:
        raise ReadinessError("scheduler reconciliation was produced for different pins")
    if (
        payload.get("passed") is not True
        or payload.get("all") is not True
        or payload.get("no_admit") is not True
    ):
        raise ReadinessError("scheduler reconciliation must pass with --all --no-admit")
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ReadinessError("scheduler reconciliation lacks typed scheduler facts")
    _exact_keys(
        scheduler,
        {"squeue_ok", "sacct_ok", "job_count", "schema5_job_count"},
        context="scheduler reconciliation scheduler facts",
    )
    if scheduler.get("squeue_ok") is not True or scheduler.get("sacct_ok") is not True:
        raise ReadinessError(
            "scheduler reconciliation lacks complete squeue+sacct truth"
        )
    for field in ("job_count", "schema5_job_count"):
        value = scheduler.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ReadinessError(f"scheduler reconciliation {field} is invalid")
    if scheduler["schema5_job_count"] != 0:
        raise ReadinessError(
            "pre-resume scheduler reconciliation must contain no controller jobs"
        )
    for field in ("errors", "unmappable_job_ids", "malformed_token_job_ids"):
        if payload.get(field) != []:
            raise ReadinessError(f"scheduler reconciliation {field} must be empty")
    if payload.get("ambiguous_tokens") != {}:
        raise ReadinessError("scheduler reconciliation contains ambiguous intents")
    if payload.get("referenced_job_ids_absent_from_window") != []:
        raise ReadinessError(
            "scheduler reconciliation has controller IDs outside its window"
        )
    roles = payload.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(ROLE_NAMES):
        raise ReadinessError("scheduler reconciliation role coverage is incomplete")
    for role, state in roles.items():
        if not isinstance(state, dict) or set(state) != {
            "active",
            "successor",
            "submission_intent",
        }:
            raise ReadinessError(f"scheduler reconciliation role {role} is malformed")
        if any(state[field] is not None for field in state):
            raise ReadinessError(
                f"pre-resume scheduler reconciliation unexpectedly references {role} jobs"
            )
    captured = payload.get("captured_timestamp")
    if (
        not isinstance(captured, (int, float))
        or isinstance(captured, bool)
        or captured < 0
    ):
        raise ReadinessError("scheduler reconciliation captured_timestamp is invalid")
    if payload.get("captured_at") != utc_timestamp(float(captured)):
        raise ReadinessError("scheduler reconciliation timestamps disagree")


def _validate_attestation(
    control: Mapping[str, Any],
    gate: str,
    evidence_path: Path,
    expected_sha256: str,
    *,
    now: float | None = None,
    snapshot_context: _SnapshotValidationContext | None = None,
) -> None:
    if snapshot_context is None:
        snapshot_context = _SnapshotValidationContext(full=True)
    if evidence_path.is_symlink() or not evidence_path.is_file():
        raise ReadinessError(
            f"readiness evidence is missing for {gate}: {evidence_path}"
        )
    observed = sha256_file(evidence_path)
    if observed != expected_sha256:
        raise ReadinessError(
            f"readiness evidence drifted for {gate}: expected {expected_sha256}, got {observed}"
        )
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(
            f"readiness evidence for {gate} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("passed") is not True:
        raise ReadinessError(
            f"readiness evidence for {gate} does not attest passed=true"
        )
    if payload.get("gate") != gate:
        raise ReadinessError(
            f"readiness evidence gate mismatch: expected {gate!r}, got {payload.get('gate')!r}"
        )
    if payload.get("immutable_sha256") != control["immutable_sha256"]:
        raise ReadinessError(
            f"readiness evidence for {gate} was produced for different pins"
        )
    if gate == "scheduler_reconciliation":
        _validate_scheduler_gate_payload(control, payload)
        return
    _exact_keys(
        payload,
        {
            "schema_version",
            "gate",
            "passed",
            "immutable_sha256",
            "metrics",
            "artifacts",
        },
        context=f"readiness evidence for {gate}",
    )
    if payload.get("schema_version") != READINESS_EVIDENCE_SCHEMA_VERSION:
        raise ReadinessError(
            f"readiness evidence for {gate} must use schema "
            f"{READINESS_EVIDENCE_SCHEMA_VERSION}"
        )
    metrics = payload.get("metrics")
    artifact_rows = payload.get("artifacts")
    if not isinstance(metrics, dict):
        raise ReadinessError(f"readiness metrics for {gate} must be an object")
    if not isinstance(artifact_rows, list):
        raise ReadinessError(f"readiness artifacts for {gate} must be an array")
    expected_names = READINESS_ARTIFACT_NAMES[gate]
    observed_names = [
        row.get("name") if isinstance(row, dict) else None for row in artifact_rows
    ]
    if len(observed_names) != len(set(observed_names)) or set(observed_names) != set(
        expected_names
    ):
        raise ReadinessError(
            f"readiness artifacts for {gate} must be exactly {sorted(expected_names)}"
        )
    verified: dict[str, tuple[Path, Mapping[str, Any] | None]] = {}
    verified_identities: set[tuple[str, str]] = set()
    active_identities: set[tuple[str, str]] = set()
    for index, row in enumerate(artifact_rows):
        assert isinstance(row, dict)
        name = str(row["name"])
        verified[name] = _validate_referenced_artifact(
            row,
            context=f"readiness artifact {gate}[{index}]",
            verified=verified_identities,
            active=active_identities,
            snapshot_context=snapshot_context,
        )
    _validate_gate_metrics(
        control,
        gate,
        metrics,
        verified,
        now=now,
        snapshot_context=snapshot_context,
    )


def attest_gate(
    state_dir: Path,
    *,
    gate: str,
    evidence_path: Path,
    expected_sha256: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    if gate not in REQUIRED_GATES or gate == "scheduler_reconciliation":
        raise ReadinessError(
            "gate must be a non-scheduler required gate; reconciliation owns its gate"
        )
    timestamp = time.time() if now is None else float(now)
    evidence_path = evidence_path.expanduser().resolve()
    evidence_hash = expected_sha256 or sha256_file(evidence_path)
    with control_lock(state_dir):
        control = load_control(state_dir)
        snapshot_record = control.get(SNAPSHOT_ATTESTATION_STATE_KEY)
        if isinstance(snapshot_record, dict):
            # Readiness publication is a trusted slow path: renew the metadata lease
            # when due, then use the preseal. New selected-member claims may consult
            # the already-sealed inventory once and are persisted below.
            refresh_snapshot_integrity_lease(state_dir, control)
            seal = validate_snapshot_integrity_attestation(
                control, state_dir=state_dir, verify_lease=True
            )
            allowed_members = [
                member
                for binding in snapshot_record.get("member_bindings", {}).values()
                if isinstance(binding, dict)
                for member in binding.get("members", [])
                if isinstance(member, dict)
            ]
            snapshot_context = _SnapshotValidationContext(
                full=False,
                seal=seal,
                allow_inventory_lookup=True,
                allowed_members=allowed_members,
            )
        else:
            snapshot_context = _SnapshotValidationContext(full=True)
        _validate_attestation(
            control,
            gate,
            evidence_path,
            evidence_hash,
            now=timestamp,
            snapshot_context=snapshot_context,
        )
        current = control["readiness"][gate]
        replacement = {
            "passed": True,
            "evidence": str(evidence_path),
            "sha256": evidence_hash,
            "attested_at": utc_timestamp(timestamp),
            "attested_timestamp": timestamp,
        }
        control["readiness"][gate] = replacement
        if gate == "snapshot" and not isinstance(snapshot_record, dict):
            target_generation = int(control["rollout_generation"]) + 1
            control[SNAPSHOT_ATTESTATION_STATE_KEY] = (
                ensure_snapshot_integrity_attestation(
                    state_dir,
                    control,
                    generation=target_generation,
                    validation_context=snapshot_context,
                )
            )
            snapshot_record = control[SNAPSHOT_ATTESTATION_STATE_KEY]
        binding_changed = False
        if snapshot_context.members:
            if not isinstance(snapshot_record, dict):
                raise ReadinessError(
                    "sealed snapshot members cannot be bound without a generation seal"
                )
            members = sorted(
                snapshot_context.members.values(),
                key=lambda row: (row["snapshot_root"], row["logical_path"]),
            )
            binding = {
                "evidence_sha256": evidence_hash,
                "members": members,
            }
            bindings = snapshot_record.setdefault("member_bindings", {})
            binding_changed = bindings.get(gate) != binding
            bindings[gate] = binding
        elif isinstance(snapshot_record, dict):
            bindings = snapshot_record.setdefault("member_bindings", {})
            if gate in bindings:
                bindings.pop(gate)
                binding_changed = True
        if current == replacement and not binding_changed:
            return control
        append_transition(
            state_dir,
            control,
            event="readiness_attested",
            details={
                "gate": gate,
                "evidence": str(evidence_path),
                "sha256": evidence_hash,
            },
            now=timestamp,
        )
        _save_control(state_dir, control, now=timestamp)
        return control


def validate_readiness(
    control: Mapping[str, Any],
    *,
    state_dir: Path | None = None,
    verify_files: bool = True,
    now: float | None = None,
    snapshot_full: bool = False,
) -> _SnapshotValidationContext:
    validate_immutable_pins(control["immutable"], verify_files=verify_files)
    _validate_admission(control["admission"])
    if verify_files and not snapshot_full and isinstance(
        control.get(SNAPSHOT_ATTESTATION_STATE_KEY), dict
    ):
        if state_dir is None:
            raise ReadinessError(
                "compact readiness validation requires the exact control state directory"
            )
        seal = validate_snapshot_integrity_attestation(
            control, state_dir=state_dir, verify_lease=True
        )
        bindings = control[SNAPSHOT_ATTESTATION_STATE_KEY].get(
            "member_bindings", {}
        )
        allowed_members = [
            member
            for binding in bindings.values()
            if isinstance(binding, dict)
            for member in binding.get("members", [])
            if isinstance(member, dict)
        ]
        snapshot_context = _SnapshotValidationContext(
            full=False,
            seal=seal,
            allowed_members=allowed_members,
        )
    else:
        snapshot_context = _SnapshotValidationContext(full=True)
    failures: list[str] = []
    for gate in REQUIRED_GATES:
        record = control["readiness"].get(gate, {})
        if record.get("passed") is not True:
            failures.append(f"{gate}: not passed")
            continue
        if not verify_files:
            continue
        evidence = record.get("evidence")
        evidence_hash = record.get("sha256")
        if not evidence or not evidence_hash:
            failures.append(f"{gate}: missing checksummed evidence")
            continue
        try:
            _validate_attestation(
                control,
                gate,
                Path(evidence),
                str(evidence_hash),
                now=now,
                snapshot_context=snapshot_context,
            )
        except ReadinessError as exc:
            failures.append(str(exc))
    if failures:
        raise ReadinessError("resume gates failed:\n- " + "\n- ".join(failures))
    return snapshot_context


def _close_open_epoch(
    control: MutableMapping[str, Any], *, reason: str, now: float
) -> bool:
    epochs = control.setdefault("throughput_epochs", [])
    if not epochs or epochs[-1].get("closed_at") is not None:
        return False
    epochs[-1]["closed_at"] = utc_timestamp(now)
    epochs[-1]["closed_timestamp"] = now
    epochs[-1]["close_reason"] = reason
    return True


def _reset_admission_ramp(
    state_dir: Path,
    control: MutableMapping[str, Any],
    *,
    reason: str,
    now: float,
    ceiling: int | None = None,
    rollout_generation: int | None = None,
    clear_last_observation: bool = False,
) -> bool:
    """Close any clean ramp window and rebind it to safe current control identity."""

    ramp = control["admission_ramp"]
    previous_ceiling = int(control["admission"]["current_ceiling"])
    target_ceiling = previous_ceiling if ceiling is None else int(ceiling)
    target_generation = (
        int(control["rollout_generation"])
        if rollout_generation is None
        else int(rollout_generation)
    )
    generation_changed = ramp.get("rollout_generation") != target_generation
    if target_ceiling not in ADMISSION_RAMP_STAGES:
        raise ControlError("cannot reset admission ramp to an unknown stage")
    window = ramp.get("window")
    identity_changed = bool(
        generation_changed
        or ramp.get("current_ceiling") != target_ceiling
        or previous_ceiling != target_ceiling
    )
    changed = window is not None or identity_changed
    if isinstance(window, dict):
        reset = {
            "reason": reason,
            "at": utc_timestamp(now),
            "timestamp": now,
            "rollout_generation": window["rollout_generation"],
            "fleet_generation": window["fleet_generation"],
            "throughput_epoch": window["throughput_epoch"],
            "ceiling": window["ceiling"],
            "started_at": window["started_at"],
            "started_timestamp": window["started_timestamp"],
            "clean_seconds": max(0.0, now - float(window["started_timestamp"])),
            "observation_count": len(window["observations"]),
            "evidence_sha256": sha256_value(window["observations"]),
        }
        ramp["resets"].append(reset)
    control["admission"]["current_ceiling"] = target_ceiling
    ramp["rollout_generation"] = target_generation
    ramp["current_ceiling"] = target_ceiling
    ramp["window"] = None
    if clear_last_observation or generation_changed:
        ramp["last_observation"] = None
    ramp["last_action"] = {
        "action": "reset",
        "reason": reason,
        "at": utc_timestamp(now),
        "timestamp": now,
        "previous_ceiling": previous_ceiling,
        "current_ceiling": target_ceiling,
        "rollout_generation": target_generation,
    }
    if changed:
        append_transition(
            state_dir,
            control,
            event="admission_ramp_reset",
            details={
                "reason": reason,
                "previous_ceiling": previous_ceiling,
                "current_ceiling": target_ceiling,
                "rollout_generation": target_generation,
                "window_was_active": isinstance(window, dict),
            },
            now=now,
        )
    return changed


def _verify_resume_controller_visibility(
    control: Mapping[str, Any], snapshot: SchedulerSnapshot, *, now: float
) -> dict[str, SchedulerJob]:
    if not snapshot.squeue_ok or not snapshot.sacct_ok:
        raise SchedulerAmbiguity(
            "transactional resume requires complete squeue+sacct truth"
        )
    visible: dict[str, SchedulerJob] = {}
    for role in ROLE_NAMES:
        record = control["controllers"][role].get("active")
        if not isinstance(record, dict) or not record.get("job_id"):
            raise SchedulerVisibilityPending(
                f"resume has not committed the {role} job ID"
            )
        matches = [
            job
            for job in snapshot.jobs
            if job.active
            and job.job_id == str(record["job_id"])
            and job.token == record.get("job_token")
        ]
        if len(matches) > 1:
            raise SchedulerAmbiguity(
                f"resume controller {role} has duplicate exact jobs"
            )
        if not matches:
            submitted = float(
                record.get("submitted_timestamp", record.get("created_timestamp", now))
            )
            if now - submitted < SUBMISSION_VISIBILITY_GRACE_SECONDS:
                raise SchedulerVisibilityPending(
                    f"resume controller {role} is not yet scheduler-visible"
                )
            raise SchedulerAmbiguity(
                f"resume controller {role} job {record['job_id']} remained invisible beyond grace"
            )
        visible[role] = matches[0]
    return visible


def _validate_resume_scheduler_namespace(
    control: Mapping[str, Any],
    snapshot: SchedulerSnapshot,
    *,
    allow_active_controllers: bool,
    allow_crash_window_intents: bool,
) -> None:
    """Fail closed on every live job that can affect a resume transaction.

    During a crash retry, an accepted active-controller intent may be visible before
    its scheduler ID was committed.  That one exact durable token is admissible.  At
    the final commit boundary, however, only the two exact ``active`` records are
    allowed; cell jobs, drill jobs, successors, malformed tokens, and all other
    production-controller rows must be absent.
    """

    if not snapshot.squeue_ok or not snapshot.sacct_ok:
        raise SchedulerAmbiguity(
            "transactional resume requires complete squeue+sacct truth"
        )
    report = build_reconciliation_report(
        control, snapshot, all_jobs=True, no_admit=True
    )
    errors = list(report["errors"])

    exact_active: set[tuple[str, str]] = set()
    crash_tokens: set[str] = set()
    for role in ROLE_NAMES:
        role_state = control["controllers"][role]
        active = role_state.get("active")
        if isinstance(active, dict) and active.get("job_id") and active.get(
            "job_token"
        ):
            exact_active.add((str(active["job_id"]), str(active["job_token"])))
        intent = role_state.get("submission_intent")
        if (
            allow_crash_window_intents
            and isinstance(intent, dict)
            and intent.get("target") == "active"
            and intent.get("job_token")
        ):
            crash_tokens.add(str(intent["job_token"]))

    unexpected_controllers: list[str] = []
    for job in snapshot.jobs:
        if not job.active or not job.comment.startswith(TOKEN_PREFIX):
            continue
        parsed = parse_job_token(job.comment)
        exact = allow_active_controllers and (job.job_id, job.comment) in exact_active
        crash_window = parsed is not None and job.comment in crash_tokens
        if parsed is None or not (exact or crash_window):
            unexpected_controllers.append(job.job_id)
    if unexpected_controllers:
        errors.append(
            "unexpected live production controller namespace jobs: "
            + ", ".join(sorted(unexpected_controllers))
        )

    live_drills = _all_live_drill_jobs(snapshot)
    if live_drills:
        errors.append(
            "live controller-drill jobs: "
            + ", ".join(sorted(job.job_id for job in live_drills))
        )
    live_cells = sorted(
        job.job_id
        for job in snapshot.jobs
        if job.active
        and (
            job.comment.startswith(CELL_INTENT_PREFIX)
            or _cell_job(job.job_name)
        )
    )
    if live_cells:
        errors.append(
            "live schema-5 cell jobs have not drained: " + ", ".join(live_cells)
        )
    if errors:
        raise SchedulerAmbiguity(
            "transactional resume found an unclean live production join: "
            + "; ".join(dict.fromkeys(errors))
        )


def _validate_resume_drill_proof(
    state_dir: Path,
    control: Mapping[str, Any],
    *,
    expected_marker_sha256: str,
) -> None:
    """Revalidate the complete static drill proof and its marker byte identity."""

    if _SHA256_RE.fullmatch(expected_marker_sha256) is None:
        raise ReadinessError("resume transaction lacks a controller kill drill proof")
    before = _controller_drill_marker_sha256(state_dir)
    if before != expected_marker_sha256:
        raise ReadinessError("controller kill drill marker drifted during resume")
    validate_controller_drill_marker(
        state_dir,
        control,
        require_current_baseline=False,
        allow_resuming=True,
    )
    after = _controller_drill_marker_sha256(state_dir)
    if after != expected_marker_sha256 or after != before:
        raise ReadinessError("controller kill drill marker changed during validation")


def _release_resume_role(
    state_dir: Path,
    *,
    role: str,
    scheduler_reader: Callable[[], SchedulerSnapshot],
    release_runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
    now: float,
) -> None:
    control = load_control(state_dir)
    record = control["controllers"][role].get("active")
    if not isinstance(record, dict) or not str(record.get("job_id", "")).isdigit():
        raise SchedulerAmbiguity(f"cannot release uncommitted resume controller {role}")
    snapshot = scheduler_reader()
    matches = [
        job
        for job in snapshot.jobs
        if job.active
        and job.job_id == str(record["job_id"])
        and job.token == record.get("job_token")
    ]
    if len(matches) > 1:
        raise SchedulerAmbiguity(
            f"refusing to release duplicate {role} controller jobs"
        )
    if not matches:
        # Once an exact held allocation is released it may start and finish before a
        # crash-retry observes it.  Complete accounting truth plus a unique terminal
        # token is sufficient to adopt that release; absence of all token history is not.
        token_matches = _find_intent_jobs(snapshot, str(record.get("job_token", "")))
        if len(token_matches) != 1 or token_matches[0].job_id != str(record["job_id"]):
            raise SchedulerVisibilityPending(
                f"exact resume controller {role} is not visible for release"
            )
        if not token_matches[0].active:
            with control_lock(state_dir):
                updated = load_control(state_dir)
                intent = updated.get("resume_intent")
                if isinstance(intent, dict):
                    intent.setdefault("release_results", {})[role] = {
                        "job_id": str(record["job_id"]),
                        "status": "adopted_terminal_after_release",
                        "at": utc_timestamp(now),
                    }
                    _save_control(state_dir, updated, now=now)
            return
    proc = release_runner(["scontrol", "release", str(record["job_id"])])
    result = {
        "job_id": str(record["job_id"]),
        "status": "released" if proc.returncode == 0 else "release_failed",
        "returncode": int(proc.returncode),
        "stderr": proc.stderr.strip()[:500],
        "at": utc_timestamp(now),
    }
    with control_lock(state_dir):
        updated = load_control(state_dir)
        intent = updated.get("resume_intent")
        if not isinstance(intent, dict):
            raise SchedulerAmbiguity(
                "resume intent disappeared during controller release"
            )
        intent.setdefault("release_results", {})[role] = result
        append_transition(
            state_dir,
            updated,
            event="resume_controller_release_result",
            details={"role": role, **result},
            now=now,
        )
        _save_control(state_dir, updated, now=now)
    if proc.returncode != 0:
        raise ControlError(
            f"failed to release exact held {role} controller {record['job_id']}: "
            f"{proc.stderr.strip()[:500]}"
        )


def resume_control(
    state_dir: Path,
    *,
    scheduler_reader: Callable[[], SchedulerSnapshot] | None = None,
    submit_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    release_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Transactionally create both controller chains before enabling admission.

    Every scheduler boundary is restartable from ``resume_intent``.  Controllers are
    submitted normally (not held) so a starting supervisor can claim its exact intent
    and wait in ``resuming`` without introducing a second release side effect.  The
    durable desired state remains ``resuming`` until both exact job IDs are visible in
    complete squeue+sacct truth; only the final atomic control write publishes
    ``running``.  Thus neither a crash nor scheduler visibility lag can expose a
    one-controller production state to the dispatcher.
    """
    timestamp = time.time() if now is None else float(now)
    read_scheduler = scheduler_reader or (lambda: query_scheduler(now=time.time()))
    submit = submit_runner or _run_subprocess
    # Kept in the public signature for callers written against the initial held-job
    # prototype.  Release is intentionally no longer an external transaction boundary.
    del release_runner
    with control_lock(state_dir):
        control = load_control(state_dir, verify_files=True)
        if control["desired_state"] == "resuming":
            # A visibility retry may occur long after the controller jobs were accepted.
            # Expiry is not evidence of drift: first bind the immutable generation
            # records through load_control, then renew both metadata proofs and verify
            # the exact refreshed leases before trusting compact readiness.
            refresh_runtime_integrity_lease(state_dir, control, now=None)
            validate_runtime_integrity_attestation(control, verify_metadata=True)
            refresh_snapshot_integrity_lease(state_dir, control, now=None)
            validate_snapshot_integrity_attestation(
                control, state_dir=state_dir, verify_lease=True
            )
        # Fleet freshness authorizes the atomic paused -> resuming transition. Once
        # that transition is durably recorded, scheduler visibility retries must not
        # expire the already-proven launch gate; all artifact bytes and typed facts
        # are still revalidated. Running is likewise idempotent after the launch
        # evidence becomes historical.
        readiness_now = timestamp if control["desired_state"] == "paused" else None
        snapshot_validation = validate_readiness(
            control,
            state_dir=state_dir,
            verify_files=True,
            now=readiness_now,
            # Each paused -> new rollout transition gets a fresh full-byte proof.
            # Normally the marker-last snapshot gate already prepared the exact next
            # generation preseal. A manual legacy control without one falls back to a
            # full scan here; retries and running paths always use the compact seal.
            snapshot_full=(
                control["desired_state"] == "paused"
                and (
                    not isinstance(
                        control.get(SNAPSHOT_ATTESTATION_STATE_KEY), dict
                    )
                    or control[SNAPSHOT_ATTESTATION_STATE_KEY].get("generation")
                    != int(control["rollout_generation"]) + 1
                )
            ),
        )
        drill_marker_sha256: str | None = None
        if (
            control["desired_state"] == "paused"
            and int(control["rollout_generation"]) == 0
        ):
            before_marker_sha256 = _controller_drill_marker_sha256(state_dir)
            validate_controller_drill_marker(state_dir, control)
            drill_marker_sha256 = _controller_drill_marker_sha256(state_dir)
            if drill_marker_sha256 != before_marker_sha256:
                raise ReadinessError(
                    "controller kill drill marker changed during validation"
                )
        elif control["desired_state"] == "paused":
            prior_resume = control.get("resume_intent")
            prior_marker_sha256 = (
                prior_resume.get("controller_drill_marker_sha256")
                if isinstance(prior_resume, dict)
                and prior_resume.get("state") == "complete"
                else None
            )
            if (
                not isinstance(prior_marker_sha256, str)
                or _SHA256_RE.fullmatch(prior_marker_sha256) is None
            ):
                raise ReadinessError(
                    "paused production control lacks its completed controller-drill proof"
                )
            current_marker_sha256 = _controller_drill_marker_sha256(state_dir)
            if current_marker_sha256 != prior_marker_sha256:
                raise ReadinessError(
                    "historical controller kill drill marker is missing or drifted"
                )
            # The first drill baseline intentionally predates all production writes, but
            # its marker, state, journal, recovery events, and sealed reconciliation stay
            # immutable and auditable across every later pause/resume cycle.
            validate_controller_drill_marker(
                state_dir,
                control,
                require_current_baseline=False,
                allow_completed_drain=True,
            )
            drill_marker_sha256 = current_marker_sha256
        if control["desired_state"] == "paused":
            pre_resume_snapshot = read_scheduler()
            # The persisted readiness gate is not live scheduler truth.  Join the
            # current queue/accounting window before changing desired state, and require
            # the controller, drill, and cell namespaces to be completely drained.
            _validate_resume_scheduler_namespace(
                control,
                pre_resume_snapshot,
                allow_active_controllers=False,
                allow_crash_window_intents=False,
            )
        elif control["desired_state"] == "resuming":
            resume_intent = control.get("resume_intent")
            expected_marker = (
                resume_intent.get("controller_drill_marker_sha256")
                if isinstance(resume_intent, dict)
                else None
            )
            if not isinstance(expected_marker, str):
                raise ReadinessError(
                    "resume transaction lacks a controller kill drill proof"
                )
            _validate_resume_drill_proof(
                state_dir,
                control,
                expected_marker_sha256=expected_marker,
            )
            _validate_resume_scheduler_namespace(
                control,
                read_scheduler(),
                allow_active_controllers=True,
                allow_crash_window_intents=True,
            )
        if control["desired_state"] == "running":
            intent = control.get("resume_intent")
            if not isinstance(intent, dict) or intent.get("state") != "complete":
                raise SchedulerAmbiguity(
                    "running control lacks a complete resume transaction"
                )
            if not control.get("drain_requested"):
                return control
        elif control["desired_state"] == "paused":
            generation = int(control["rollout_generation"]) + 1
            # Establish the expensive byte-level environment proof before publishing
            # or submitting anything for the new rollout generation.  The immutable
            # generation file is then metadata-validated by every process at startup.
            runtime_attestation = ensure_runtime_integrity_attestation(
                state_dir,
                control,
                generation=generation,
                force_full=True,
                now=timestamp,
            )
            existing_snapshot_attestation = control.get(
                SNAPSHOT_ATTESTATION_STATE_KEY
            )
            if (
                isinstance(existing_snapshot_attestation, dict)
                and existing_snapshot_attestation.get("generation") == generation
            ):
                refresh_snapshot_integrity_lease(state_dir, control)
                validate_snapshot_integrity_attestation(
                    control, state_dir=state_dir, verify_lease=True
                )
                snapshot_attestation = copy.deepcopy(
                    existing_snapshot_attestation
                )
            else:
                snapshot_attestation = ensure_snapshot_integrity_attestation(
                    state_dir,
                    control,
                    generation=generation,
                    validation_context=snapshot_validation,
                )
            control["desired_state"] = "resuming"
            control["drain_requested"] = False
            control["drain_intent"] = None
            control["rollout_generation"] = generation
            _reset_admission_ramp(
                state_dir,
                control,
                reason="rollout_generation_started",
                now=timestamp,
                ceiling=24,
                rollout_generation=generation,
                clear_last_observation=True,
            )
            control[RUNTIME_ATTESTATION_STATE_KEY] = runtime_attestation
            control[SNAPSHOT_ATTESTATION_STATE_KEY] = snapshot_attestation
            control["resume_intent"] = {
                "resume_id": uuid.uuid4().hex,
                "state": "submitting_controllers",
                "rollout_generation": generation,
                "created_at": utc_timestamp(timestamp),
                "created_timestamp": timestamp,
                "controller_job_ids": {},
                "controller_drill_marker_sha256": drill_marker_sha256,
            }
            append_transition(
                state_dir,
                control,
                event="resume_intent_created",
                details={
                    "resume_id": control["resume_intent"]["resume_id"],
                    "rollout_generation": generation,
                    "admission_ceiling": control["admission"]["current_ceiling"],
                },
                now=timestamp,
            )
            _save_control(state_dir, control, now=timestamp)

    control = load_control(state_dir)
    if not isinstance(control.get("resume_intent"), dict):
        raise SchedulerAmbiguity("resume transaction has no durable intent")

    if control["desired_state"] == "resuming":
        expected_drill_marker = control["resume_intent"].get(
            "controller_drill_marker_sha256"
        )
        if not isinstance(expected_drill_marker, str):
            raise ReadinessError(
                "resume transaction lacks a controller kill drill proof"
            )
        _validate_resume_drill_proof(
            state_dir,
            control,
            expected_marker_sha256=expected_drill_marker,
        )
        for role in ROLE_NAMES:
            current = load_control(state_dir)["controllers"][role].get("active")
            snapshot = read_scheduler()
            if not snapshot.squeue_ok or not snapshot.sacct_ok:
                raise SchedulerAmbiguity(
                    "transactional resume requires complete squeue+sacct truth"
                )
            exact_live = bool(
                isinstance(current, dict)
                and any(
                    job.active
                    and job.job_id == str(current.get("job_id"))
                    and job.token == current.get("job_token")
                    for job in snapshot.jobs
                )
            )
            current_inside_visibility_grace = bool(
                isinstance(current, dict)
                and current.get("job_id")
                and not any(
                    job.job_id == str(current.get("job_id"))
                    and job.token == current.get("job_token")
                    for job in snapshot.jobs
                )
                and timestamp
                - float(
                    current.get(
                        "submitted_timestamp",
                        current.get("created_timestamp", timestamp),
                    )
                )
                < SUBMISSION_VISIBILITY_GRACE_SECONDS
            )
            if exact_live or current_inside_visibility_grace:
                record = current
            else:
                record = submit_controller_intent(
                    state_dir,
                    role=role,
                    target="active",
                    dependency_job_id=None,
                    scheduler=snapshot,
                    submit_runner=submit,
                    allow_resuming=True,
                    hold=False,
                    now=timestamp,
                )
            assert isinstance(record, dict)
            with control_lock(state_dir):
                updated = load_control(state_dir)
                resume_intent = updated.get("resume_intent")
                if not isinstance(resume_intent, dict):
                    raise SchedulerAmbiguity(
                        "resume intent disappeared during submission"
                    )
                resume_intent.setdefault("controller_job_ids", {})[role] = str(
                    record["job_id"]
                )
                _save_control(state_dir, updated, now=timestamp)
        with control_lock(state_dir):
            control = load_control(state_dir)
            # This scheduler read deliberately occurs while the final control lock is
            # held.  No concurrent controller-state mutation or proof replacement can
            # be interleaved between this join and the atomic running-state commit.
            snapshot = read_scheduler()
            _validate_resume_scheduler_namespace(
                control,
                snapshot,
                allow_active_controllers=True,
                allow_crash_window_intents=False,
            )
            visible = _verify_resume_controller_visibility(
                control, snapshot, now=timestamp
            )
            resume_intent = control["resume_intent"]
            expected_marker = resume_intent.get("controller_drill_marker_sha256")
            if not isinstance(expected_marker, str):
                raise ReadinessError(
                    "resume transaction lacks a controller kill drill proof"
                )
            _validate_resume_drill_proof(
                state_dir,
                control,
                expected_marker_sha256=expected_marker,
            )
            expected_ids = {role: visible[role].job_id for role in ROLE_NAMES}
            if resume_intent.get("controller_job_ids") != expected_ids:
                raise SchedulerAmbiguity(
                    "resume intent job IDs changed before the running-state commit"
                )
            if _controller_drill_marker_sha256(state_dir) != expected_marker:
                raise ReadinessError(
                    "controller kill drill marker drifted at the running-state commit"
                )
            resume_intent["state"] = "complete"
            resume_intent["scheduler_visible_job_ids"] = {
                role: visible[role].job_id for role in ROLE_NAMES
            }
            resume_intent["completed_at"] = utc_timestamp(timestamp)
            resume_intent["completed_timestamp"] = timestamp
            control["desired_state"] = "running"
            append_transition(
                state_dir,
                control,
                event="resumed",
                details={
                    "resume_id": resume_intent["resume_id"],
                    "controller_job_ids": resume_intent["scheduler_visible_job_ids"],
                    "rollout_generation": control["rollout_generation"],
                },
                now=timestamp,
            )
            _save_control(state_dir, control, now=timestamp)
    return load_control(state_dir)


def _cell_array_base_job_id(job_id: str) -> str | None:
    if _EXACT_CELL_TASK_ID.fullmatch(job_id) is None:
        return None
    return job_id.split("_", 1)[0]


def _assert_cell_submission_cut_visible(
    state_dir: Path,
    snapshot: SchedulerSnapshot,
    *,
    now: float,
    visibility_grace_seconds: float = 300.0,
) -> None:
    """Prove every intent that crossed sbatch is represented at the drain cut.

    The dispatcher fsyncs ``submitting`` before invoking sbatch and keeps the shared
    admission lock until it fsyncs the outcome.  Slurm visibility can nevertheless lag
    an accepted reply.  A pause must therefore reject an apparently empty scheduler
    snapshot while such an intent remains inside its visibility grace; otherwise the
    accepted array could begin after pause falsely reported a complete drain.
    """

    if not snapshot.squeue_ok or not snapshot.sacct_ok:
        raise SchedulerAmbiguity(
            "exact cell drain requires complete squeue+sacct transaction truth"
        )
    ledger_path = state_dir / "ledger.json"
    if not ledger_path.exists():
        return
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchedulerAmbiguity(
            f"cannot prove the cell submission cut from dispatcher ledger: {exc}"
        ) from exc
    intents = ledger.get("intents") if isinstance(ledger, dict) else None
    jobs = ledger.get("jobs") if isinstance(ledger, dict) else None
    if not isinstance(intents, dict) or not isinstance(jobs, dict):
        raise SchedulerAmbiguity(
            "dispatcher ledger has no typed intents/jobs transaction mapping"
        )

    for batch_id, intent in sorted(intents.items()):
        if not isinstance(batch_id, str) or not isinstance(intent, dict):
            raise SchedulerAmbiguity("dispatcher ledger contains an invalid cell intent")
        state = str(intent.get("state", ""))
        if state not in {"submitting", "submitted"}:
            # ``prepared`` is durable before the dispatcher changes to ``submitting``;
            # sbatch is never invoked from that state.  Reconciled/terminal states are
            # already represented in the durable jobs table.
            continue
        expected_comment = CELL_INTENT_PREFIX + batch_id
        expected_name = f"asys-dispatch-{batch_id[-10:]}"
        expected_path = str(
            Path(str(intent.get("sbatch_path", ""))).expanduser().resolve()
        )
        matching_bases: set[str] = set()
        provenance_errors: list[str] = []
        for job in snapshot.jobs:
            base_id = _cell_array_base_job_id(job.job_id)
            if base_id is None:
                continue
            try:
                command_paths = {
                    str(Path(part).expanduser().resolve())
                    for part in shlex.split(job.command)
                    if part.endswith(".sbatch") and Path(part).is_absolute()
                }
            except (OSError, ValueError):
                command_paths = set()
            namespace_match = (
                job.comment == expected_comment
                or job.job_name == expected_name
                or expected_path in command_paths
            )
            if not namespace_match:
                continue
            if (
                job.comment != expected_comment
                or job.job_name != expected_name
                or expected_path not in command_paths
            ):
                provenance_errors.append(job.job_id)
                continue
            matching_bases.add(base_id)
        if provenance_errors:
            raise SchedulerAmbiguity(
                f"cell intent {batch_id} has scheduler provenance drift for jobs "
                f"{sorted(provenance_errors)}"
            )
        if len(matching_bases) > 1:
            raise SchedulerAmbiguity(
                f"cell intent {batch_id} maps to duplicate Slurm jobs "
                f"{sorted(matching_bases)}"
            )
        recorded_job_id = intent.get("job_id")
        recorded_job = (
            jobs.get(str(recorded_job_id)) if recorded_job_id is not None else None
        )
        if recorded_job_id is not None and (
            not str(recorded_job_id).isdigit()
            or matching_bases
            and matching_bases != {str(recorded_job_id)}
        ):
            raise SchedulerAmbiguity(
                f"cell intent {batch_id} conflicts with its accepted Slurm job ID"
            )
        if matching_bases:
            continue
        if isinstance(recorded_job, dict) and recorded_job.get("state") == "inactive":
            # A prior complete dispatcher poll already observed this accepted job as
            # absent beyond its visibility reservation.  It cannot start after this
            # pause and need not remain in sacct's finite query window forever.
            continue
        submit_started_at = intent.get("submit_started_at")
        if (
            not isinstance(submit_started_at, (int, float))
            or isinstance(submit_started_at, bool)
            or not math.isfinite(float(submit_started_at))
        ):
            raise SchedulerAmbiguity(
                f"cell intent {batch_id} lacks a valid durable sbatch-boundary timestamp"
            )
        age = max(0.0, float(now) - float(submit_started_at))
        if age < visibility_grace_seconds:
            raise SchedulerVisibilityPending(
                f"cell intent {batch_id} crossed sbatch only {age:.1f}s ago but is not "
                "yet visible in complete squeue+sacct truth; retry pause after visibility"
            )
        if state == "submitted" or recorded_job_id is not None:
            raise SchedulerAmbiguity(
                f"accepted cell intent {batch_id} job {recorded_job_id} disappeared from "
                "complete scheduler history"
            )
        # An ID-less ambiguous submit that remains absent from both authoritative
        # sources beyond the visibility grace is proven not accepted for this cut.


def _exact_live_cell_task_actions(
    state_dir: Path, snapshot: SchedulerSnapshot
) -> tuple[list[str], list[str]]:
    """Map every live schema-5 cell task to a durable drain action.

    A pause signal is a mutating scheduler operation, so job-name matching alone is
    insufficient.  The live squeue row must bind the exact array base ID, batch token,
    and immutable sbatch path recorded in the dispatcher ledger.  Any scoped row that
    cannot be proven is an ambiguity and prevents *all* signalling.  Running/batch
    workers receive USR1; pending/configuring/suspended elements are canceled exactly so
    they cannot begin stochastic work after the paused snapshot.
    """

    if not snapshot.squeue_ok:
        raise SchedulerAmbiguity(
            "exact cell drain requires authoritative live squeue truth"
        )
    live_rows = [
        job
        for job in snapshot.jobs
        if job.source == "squeue"
        and normalize_scheduler_state(job.state) in ACTIVE_SCHEDULER_STATES
        and _cell_job(job.job_name)
    ]
    if not live_rows:
        return [], []
    ledger_path = state_dir / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SchedulerAmbiguity(
            f"cannot map live cell tasks without durable dispatcher ledger: {exc}"
        ) from exc
    if not isinstance(ledger, dict) or not isinstance(ledger.get("jobs"), dict):
        raise SchedulerAmbiguity("dispatcher ledger has no typed jobs mapping")
    intents = ledger.get("intents")
    if not isinstance(intents, dict):
        raise SchedulerAmbiguity("dispatcher ledger has no typed intents mapping")

    signal_ids: list[str] = []
    cancel_ids: list[str] = []
    errors: list[str] = []
    for job in live_rows:
        base_id = _cell_array_base_job_id(job.job_id)
        if base_id is None:
            errors.append(f"unsafe cell task ID {job.job_id!r}")
            continue
        if not job.job_name.startswith("asys-dispatch-"):
            errors.append(f"unscoped legacy cell task {job.job_id} ({job.job_name})")
            continue
        if not job.comment.startswith(CELL_INTENT_PREFIX):
            errors.append(f"cell task {job.job_id} lacks a schema-5 batch token")
            continue
        batch_id = job.comment[len(CELL_INTENT_PREFIX) :]
        record = ledger["jobs"].get(base_id)
        intent = intents.get(batch_id)
        if not isinstance(intent, dict):
            errors.append(
                f"cell task {job.job_id} is absent from its durable intent ledger"
            )
            continue
        expected_name = f"asys-dispatch-{batch_id[-10:]}"
        if job.job_name != expected_name:
            errors.append(
                f"cell task {job.job_id} has unexpected job name {job.job_name!r}"
            )
            continue
        intent_job_id = intent.get("job_id")
        if intent_job_id is not None and str(intent_job_id) != base_id:
            errors.append(
                f"cell task {job.job_id} conflicts with batch intent {batch_id}"
            )
            continue
        if isinstance(record, dict):
            if (
                str(record.get("job_id")) != base_id
                or str(record.get("batch_id")) != batch_id
                or str(record.get("sbatch_path")) != str(intent.get("sbatch_path"))
            ):
                errors.append(
                    f"cell task {job.job_id} conflicts with batch intent {batch_id}"
                )
                continue
        elif str(intent.get("state")) != "submitting":
            errors.append(
                f"cell task {job.job_id} has no accepted or ambiguous durable job record"
            )
            continue
        sbatch_path = str(
            Path(str(intent.get("sbatch_path", ""))).expanduser().resolve()
        )
        try:
            command_paths = {
                str(Path(part).expanduser().resolve())
                for part in shlex.split(job.command)
                if part.endswith(".sbatch")
            }
        except (OSError, ValueError) as exc:
            errors.append(
                f"cell task {job.job_id} has an invalid submit command: {exc}"
            )
            continue
        if sbatch_path not in command_paths:
            errors.append(
                f"cell task {job.job_id} does not bind its immutable sbatch path"
            )
            continue
        if normalize_scheduler_state(job.state) in {"RUNNING", "COMPLETING"}:
            signal_ids.append(job.job_id)
        else:
            cancel_ids.append(job.job_id)
    if errors:
        raise SchedulerAmbiguity(
            "exact cell drain mapping failed: " + "; ".join(errors)
        )
    all_ids = signal_ids + cancel_ids
    if len(all_ids) != len(set(all_ids)):
        raise SchedulerAmbiguity("exact cell drain contains duplicate task IDs")
    return sorted(signal_ids), sorted(cancel_ids)


def _exact_running_cell_task_ids(
    state_dir: Path, snapshot: SchedulerSnapshot
) -> list[str]:
    """Compatibility helper returning the signalable subset of an exact drain."""

    signal_ids, _cancel_ids = _exact_live_cell_task_actions(state_dir, snapshot)
    return signal_ids


def _persist_drain_result(
    state_dir: Path,
    *,
    category: str,
    job_id: str,
    command: Sequence[str],
    proc: subprocess.CompletedProcess[str],
    now: float,
) -> None:
    with control_lock(state_dir):
        control = load_control(state_dir)
        intent = control.get("drain_intent")
        if not isinstance(intent, dict) or intent.get("state") != "signaling":
            raise SchedulerAmbiguity(
                "pause drain intent changed during scheduler mutation"
            )
        key = f"{category}:{job_id}"
        intent.setdefault("results", {})[key] = {
            "category": category,
            "job_id": job_id,
            "command": list(command),
            "returncode": int(proc.returncode),
            "stderr": proc.stderr.strip()[:1000],
            "completed_at": utc_timestamp(now),
            "completed_timestamp": now,
        }
        _save_control(state_dir, control, now=now)


def _pause_control_locked(
    state_dir: Path,
    *,
    drain: bool,
    scheduler: SchedulerSnapshot | None = None,
    cancel_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    now: float | None = None,
) -> dict[str, Any]:
    if not drain:
        raise ControlError("schema-5 production may only be paused with --drain")
    timestamp = time.time() if now is None else float(now)
    successor_jobs: list[tuple[str, str]] = []
    with control_lock(state_dir):
        control = load_control(state_dir)
        changed = control["desired_state"] != "paused" or not control.get(
            "drain_requested"
        )
        control["desired_state"] = "paused"
        control["drain_requested"] = True
        _close_open_epoch(control, reason="pause", now=timestamp)
        _reset_admission_ramp(
            state_dir,
            control,
            reason="pause",
            now=timestamp,
            ceiling=24,
            clear_last_observation=True,
        )
        for role in ROLE_NAMES:
            successor = control["controllers"][role].get("successor")
            if isinstance(successor, dict) and successor.get("job_id"):
                successor_jobs.append(
                    (str(successor["job_id"]), str(successor["job_token"]))
                )
        if changed:
            append_transition(
                state_dir,
                control,
                event="pause_drain_requested",
                details={"successor_job_ids": [item[0] for item in successor_jobs]},
                now=timestamp,
            )
        _save_control(state_dir, control, now=timestamp)

    if scheduler is None:
        return load_control(state_dir)

    # Resolve every exact target before crossing the first external mutation boundary.
    _assert_cell_submission_cut_visible(
        state_dir,
        scheduler,
        now=timestamp,
    )
    cell_task_ids, pending_cell_task_ids = _exact_live_cell_task_actions(
        state_dir, scheduler
    )
    by_id = {job.job_id: job for job in scheduler.jobs}
    verified_successors: list[str] = []
    for job_id, expected_token in successor_jobs:
        job = by_id.get(job_id)
        if job is None or not job.active:
            continue
        if job.token != expected_token:
            raise SchedulerAmbiguity(
                f"refusing to cancel successor {job_id}: scheduler token does not match"
            )
        verified_successors.append(job_id)
    with control_lock(state_dir):
        control = load_control(state_dir)
        previous = control.get("drain_intent")
        previous_results = (
            copy.deepcopy(previous.get("results", {}))
            if isinstance(previous, dict)
            else {}
        )
        control["drain_intent"] = {
            "drain_id": (
                str(previous.get("drain_id"))
                if isinstance(previous, dict) and previous.get("drain_id")
                else uuid.uuid4().hex
            ),
            "state": "signaling",
            "rollout_generation": control["rollout_generation"],
            "created_at": (
                previous.get("created_at")
                if isinstance(previous, dict)
                else utc_timestamp(timestamp)
            ),
            "created_timestamp": (
                previous.get("created_timestamp")
                if isinstance(previous, dict)
                else timestamp
            ),
            "cell_task_ids": cell_task_ids,
            "pending_cell_task_ids": pending_cell_task_ids,
            "successor_job_ids": sorted(verified_successors),
            "results": previous_results,
        }
        append_transition(
            state_dir,
            control,
            event="pause_exact_targets_committed",
            details={
                "cell_task_ids": cell_task_ids,
                "pending_cell_task_ids": pending_cell_task_ids,
                "successor_job_ids": sorted(verified_successors),
            },
            now=timestamp,
        )
        _save_control(state_dir, control, now=timestamp)

    runner = cancel_runner or _run_subprocess
    for category, identifiers, signal_argv in (
        # The production batch script ``exec``s Python as the Slurm batch step.  Slurm
        # excludes that step from non-KILL signals unless --batch/--full is explicit.
        ("cell_usr1", cell_task_ids, ("scancel", "--batch", "--signal=USR1")),
        ("pending_cell_cancel", pending_cell_task_ids, ("scancel",)),
        ("successor_cancel", sorted(verified_successors), ("scancel",)),
    ):
        for job_id in identifiers:
            current = load_control(state_dir)["drain_intent"]
            key = f"{category}:{job_id}"
            prior = current.get("results", {}).get(key)
            if isinstance(prior, dict) and prior.get("returncode") == 0:
                continue
            command = [*signal_argv, job_id]
            proc = runner(command)
            completed = time.time() if now is None else timestamp
            _persist_drain_result(
                state_dir,
                category=category,
                job_id=job_id,
                command=command,
                proc=proc,
                now=completed,
            )
            if proc.returncode != 0:
                raise ControlError(
                    f"failed exact pause action for {job_id}: {proc.stderr.strip()[:500]}"
                )
    with control_lock(state_dir):
        control = load_control(state_dir)
        intent = control.get("drain_intent")
        if not isinstance(intent, dict):
            raise SchedulerAmbiguity("pause drain intent disappeared")
        intent["state"] = "complete"
        intent["completed_at"] = utc_timestamp(timestamp)
        intent["completed_timestamp"] = timestamp
        append_transition(
            state_dir,
            control,
            event="pause_drain_signals_complete",
            details={
                "cell_task_count": len(cell_task_ids),
                "pending_cell_task_count": len(pending_cell_task_ids),
                "successor_count": len(verified_successors),
            },
            now=timestamp,
        )
        _save_control(state_dir, control, now=timestamp)
    return load_control(state_dir)


def pause_control(
    state_dir: Path,
    *,
    drain: bool,
    scheduler: SchedulerSnapshot | None = None,
    scheduler_reader: Callable[[], SchedulerSnapshot] | None = None,
    cancel_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Fence admission, take fresh scheduler truth, and drain without a race."""

    if scheduler is not None and scheduler_reader is not None:
        raise ControlError("pause accepts scheduler or scheduler_reader, not both")
    with admission_boundary_lock(state_dir):
        snapshot = scheduler_reader() if scheduler_reader is not None else scheduler
        return _pause_control_locked(
            state_dir,
            drain=drain,
            scheduler=snapshot,
            cancel_runner=cancel_runner,
            now=now,
        )


def set_admission_ceiling(
    state_dir: Path, *, ceiling: int, now: float | None = None
) -> dict[str, Any]:
    if ceiling not in ADMISSION_RAMP_STAGES:
        raise ControlError("schema-5 admission ceiling must be 24, 96, 192, or 384")
    timestamp = time.time() if now is None else float(now)
    with control_lock(state_dir):
        control = load_control(state_dir)
        previous = control["admission"]["current_ceiling"]
        if previous == ceiling:
            return control
        if ceiling > previous:
            raise ControlError(
                "raising admission is monitor-controlled and requires a complete "
                "checksummed clean-window proof"
            )
        _reset_admission_ramp(
            state_dir,
            control,
            reason="manual_ceiling_decrease",
            now=timestamp,
            ceiling=ceiling,
            clear_last_observation=True,
        )
        append_transition(
            state_dir,
            control,
            event="admission_ceiling_changed",
            details={"previous": previous, "current": ceiling},
            now=timestamp,
        )
        _save_control(state_dir, control, now=timestamp)
        return control


def _active_critical_alerts(control: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(alert.get("dedupe_key"))
        for alert in control.get("alerts", [])
        if isinstance(alert, dict)
        and alert.get("resolved_at") is None
        and alert.get("severity") == "critical"
    )


def _active_ramp_blocking_alerts(control: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(alert.get("dedupe_key"))
        for alert in control.get("alerts", [])
        if isinstance(alert, dict)
        and alert.get("resolved_at") is None
        and (
            alert.get("severity") == "critical"
            or alert.get("dedupe_key") in ADMISSION_RAMP_BLOCKING_ALERT_KEYS
        )
    )


def _read_admission_ramp_evidence(
    state_dir: Path, evidence_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = evidence_path.expanduser().resolve()
    root = (state_dir.expanduser().resolve() / "monitoring").resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ControlError(
            f"admission ramp evidence must be under {root}, found {path}"
        ) from exc
    if evidence_path.is_symlink() or not path.is_file():
        raise ControlError("admission ramp evidence must be a regular non-symlink file")
    if path.stat().st_nlink != 1:
        raise ControlError("admission ramp evidence must not have hardlink aliases")
    if path.stat().st_mode & 0o222:
        raise ControlError("admission ramp evidence must be sealed read-only")
    try:
        payload = path.read_bytes()
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        report = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlError(f"cannot read admission ramp evidence {path}: {exc}") from exc
    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ControlError("admission ramp evidence SHA-256 is invalid")
    if observed_sha256 != expected_sha256:
        raise ControlError(
            f"admission ramp evidence hash mismatch: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    if not isinstance(report, dict):
        raise ControlError("admission ramp evidence report must be an object")
    observation = report.get("ramp_observation")
    expected_fields = {
        "schema_version",
        "protocol",
        "captured_timestamp",
        "committed_timestamp",
        "cadence",
        "control_immutable_sha256",
        "rollout_generation",
        "admission_ceiling",
        "fleet_generation",
        "production_health_clean",
        "semantic_integrity_clean",
        "critical_finding_keys",
        "promotion_blocking_finding_keys",
        "run_validated_qids",
    }
    if not isinstance(observation, dict) or set(observation) != expected_fields:
        raise ControlError("admission ramp observation fields are not exact")
    if (
        observation.get("schema_version") != 1
        or observation.get("protocol") != ADMISSION_RAMP_EVIDENCE_PROTOCOL
        or observation.get("cadence") not in MONITOR_CADENCE_SECONDS
        or observation.get("captured_timestamp") != report.get("captured_timestamp")
        or not isinstance(observation.get("committed_timestamp"), (int, float))
        or isinstance(observation.get("committed_timestamp"), bool)
        or float(observation["committed_timestamp"])
        < float(observation["captured_timestamp"])
        or observation.get("cadence") != report.get("cadence")
    ):
        raise ControlError("admission ramp observation identity is invalid")
    critical_keys = observation.get("critical_finding_keys")
    if (
        not isinstance(critical_keys, list)
        or critical_keys != sorted(set(critical_keys))
        or not all(isinstance(value, str) and value for value in critical_keys)
    ):
        raise ControlError("admission ramp critical-finding evidence is invalid")
    blocking_keys = observation.get("promotion_blocking_finding_keys")
    if (
        not isinstance(blocking_keys, list)
        or blocking_keys != sorted(set(blocking_keys))
        or not all(isinstance(value, str) and value for value in blocking_keys)
        or not set(critical_keys).issubset(blocking_keys)
    ):
        raise ControlError("admission ramp promotion-blocking evidence is invalid")
    health = report.get("health")
    live = health.get("control") if isinstance(health, dict) else None
    if (
        not isinstance(health, dict)
        or not isinstance(live, dict)
        or observation.get("fleet_generation") != health.get("fleet_generation")
        or observation.get("rollout_generation") != live.get("rollout_generation")
        or observation.get("admission_ceiling")
        != live.get("admission", {}).get("current_ceiling")
    ):
        raise ControlError("admission ramp evidence disagrees with live report fields")
    cadence = str(observation["cadence"])
    if cadence == "health":
        if (
            observation.get("semantic_integrity_clean") is not None
            or observation.get("run_validated_qids") is not None
        ):
            raise ControlError("health ramp evidence must not claim semantic progress")
    else:
        semantic = report.get("semantic")
        runs = semantic.get("runs") if isinstance(semantic, dict) else None
        if not isinstance(runs, dict) or set(runs) != set(REQUIRED_RUNS):
            raise ControlError("semantic ramp evidence does not cover all production runs")
        qids: dict[str, int] = {}
        for run_id in REQUIRED_RUNS:
            outcomes = (
                runs[run_id].get("outcomes")
                if isinstance(runs[run_id], dict)
                else None
            )
            value = outcomes.get("validated_qids") if isinstance(outcomes, dict) else None
            if not isinstance(value, int) or isinstance(value, bool):
                raise ControlError(f"semantic ramp evidence lacks QIDs for {run_id}")
            qids[run_id] = value
        if not _valid_run_qid_map(qids) or observation.get("run_validated_qids") != qids:
            raise ControlError("semantic ramp per-run validated-QID evidence is invalid")
        total = semantic.get("outcomes", {}).get("validated_qids")
        if total != sum(qids.values()):
            raise ControlError("semantic ramp aggregate QIDs disagree with its run totals")
        if not isinstance(observation.get("semantic_integrity_clean"), bool):
            raise ControlError("semantic ramp integrity evidence must be boolean")
    return report, observation, observed_sha256


def _ramp_observation_record(
    *, path: Path, sha256: str, observation: Mapping[str, Any], clean: bool
) -> dict[str, Any]:
    return {
        "path": str(path.expanduser().resolve()),
        "sha256": sha256,
        "captured_timestamp": float(observation["captured_timestamp"]),
        "timestamp": float(observation["committed_timestamp"]),
        "cadence": str(observation["cadence"]),
        "clean": bool(clean),
        "run_validated_qids": copy.deepcopy(
            observation.get("run_validated_qids")
        ),
    }


def _start_admission_ramp_window(
    ramp: MutableMapping[str, Any],
    *,
    control: Mapping[str, Any],
    observation: Mapping[str, Any],
    record: Mapping[str, Any],
    epoch: Mapping[str, Any],
    now: float,
) -> None:
    qids = observation.get("run_validated_qids")
    if not _valid_run_qid_map(qids):
        raise ControlError("a ramp window can start only from complete per-run QID evidence")
    ramp["window"] = {
        "rollout_generation": int(control["rollout_generation"]),
        "fleet_generation": str(observation["fleet_generation"]),
        "throughput_epoch": int(epoch["epoch"]),
        "ceiling": int(control["admission"]["current_ceiling"]),
        "started_at": utc_timestamp(now),
        "started_timestamp": now,
        "baseline_run_validated_qids": copy.deepcopy(qids),
        "latest_run_validated_qids": copy.deepcopy(qids),
        "observations": [copy.deepcopy(dict(record))],
    }


def _verify_admission_ramp_window_evidence(
    state_dir: Path, observations: Sequence[Mapping[str, Any]]
) -> str:
    """Re-open every sealed report before using its chain to raise admission."""

    identities: set[tuple[str, str]] = set()
    for record in observations:
        path = Path(str(record["path"]))
        digest = str(record["sha256"])
        identity = (str(path.expanduser().resolve()), digest)
        if identity in identities:
            raise ControlError("admission ramp window repeats one evidence file")
        identities.add(identity)
        _report, observation, observed_sha256 = _read_admission_ramp_evidence(
            state_dir, path, digest
        )
        if (
            observed_sha256 != digest
            or float(observation["captured_timestamp"])
            != float(record["captured_timestamp"])
            or float(observation["committed_timestamp"])
            != float(record["timestamp"])
            or observation["cadence"] != record["cadence"]
            or observation.get("run_validated_qids")
            != record.get("run_validated_qids")
        ):
            raise ControlError("admission ramp evidence chain record drifted")
    return sha256_value(observations)


def record_successful_poll(
    state_dir: Path,
    *,
    validated_qids: int,
    fleet_generation: str,
    strata_with_throughput: int | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Append generation-scoped throughput accounting after a successful live poll."""
    if validated_qids < 0:
        raise ControlError("validated_qids must be non-negative")
    if not fleet_generation:
        raise ControlError("fleet_generation must be non-empty")
    timestamp = time.time() if now is None else float(now)
    with control_lock(state_dir):
        control = load_control(state_dir)
        try:
            _require_production_running(control)
        except ControlError as exc:
            raise ControlError(
                "throughput epochs cannot start or advance while paused or resuming"
            ) from exc
        critical_alerts = _active_critical_alerts(control)
        if critical_alerts:
            raise ControlError(
                "throughput epochs cannot advance with unresolved critical alerts: "
                + ", ".join(critical_alerts)
            )
        epochs = control["throughput_epochs"]
        open_epoch = (
            epochs[-1] if epochs and epochs[-1].get("closed_at") is None else None
        )
        if (
            open_epoch is not None
            and open_epoch["fleet_generation"] != fleet_generation
        ):
            _close_open_epoch(control, reason="material_fleet_change", now=timestamp)
            _reset_admission_ramp(
                state_dir,
                control,
                reason="material_fleet_change",
                now=timestamp,
                ceiling=24,
                clear_last_observation=True,
            )
            open_epoch = None
        if open_epoch is None:
            open_epoch = {
                "epoch": len(epochs) + 1,
                "rollout_generation": control["rollout_generation"],
                "fleet_generation": fleet_generation,
                "started_at": utc_timestamp(timestamp),
                "started_timestamp": timestamp,
                "closed_at": None,
                "samples": [],
            }
            epochs.append(open_epoch)
            append_transition(
                state_dir,
                control,
                event="throughput_epoch_started",
                details={
                    "epoch": open_epoch["epoch"],
                    "rollout_generation": control["rollout_generation"],
                    "fleet_generation": fleet_generation,
                },
                now=timestamp,
            )
        samples = open_epoch["samples"]
        if samples and validated_qids < samples[-1]["validated_qids"]:
            raise ControlError(
                "validated QID total regressed within one throughput epoch"
            )
        sample: dict[str, Any] = {
            "at": utc_timestamp(timestamp),
            "timestamp": timestamp,
            "validated_qids": validated_qids,
        }
        if strata_with_throughput is not None:
            if strata_with_throughput < 0:
                raise ControlError("strata_with_throughput must be non-negative")
            sample["strata_with_throughput"] = strata_with_throughput
        samples.append(sample)
        append_transition(
            state_dir,
            control,
            event="successful_production_poll",
            details={
                "epoch": open_epoch["epoch"],
                "validated_qids": validated_qids,
            },
            now=timestamp,
        )
        _save_control(state_dir, control, now=timestamp)
        return control


def record_admission_ramp_observation(
    state_dir: Path,
    *,
    evidence_path: Path,
    evidence_sha256: str,
    now: float | None = None,
) -> dict[str, Any]:
    """Record one immutable monitor observation and promote only from a full proof.

    Promotion is intentionally not an operator switch.  It is derived under the
    control lock from a read-only checksummed report, the current rollout generation,
    the open material-fleet throughput epoch, alert state, continuous five-minute
    health evidence, and exact per-run semantic progress.
    """

    _report, observation, observed_sha256 = _read_admission_ramp_evidence(
        state_dir, evidence_path, evidence_sha256
    )
    timestamp = (
        float(observation["committed_timestamp"]) if now is None else float(now)
    )
    if timestamp != float(observation["committed_timestamp"]):
        raise ControlError("admission ramp evidence timestamp differs from commit time")
    resolved_path = evidence_path.expanduser().resolve()
    with control_lock(state_dir):
        control = load_control(state_dir)
        ramp = control["admission_ramp"]
        prior = ramp.get("last_observation")
        if isinstance(prior, dict) and (
            prior.get("path") == str(resolved_path)
            and prior.get("sha256") == observed_sha256
        ):
            return control
        if isinstance(prior, dict) and float(prior.get("timestamp", -1.0)) >= timestamp:
            raise ControlError("admission ramp evidence is stale or reordered")

        record = _ramp_observation_record(
            path=resolved_path,
            sha256=observed_sha256,
            observation=observation,
            clean=False,
        )
        ramp["last_observation"] = record
        identity_matches = bool(
            observation.get("control_immutable_sha256")
            == control["immutable_sha256"]
            and observation.get("rollout_generation")
            == control["rollout_generation"]
            and observation.get("admission_ceiling")
            == control["admission"]["current_ceiling"]
        )
        if not identity_matches or control["desired_state"] != "running":
            _reset_admission_ramp(
                state_dir,
                control,
                reason=(
                    "production_not_running"
                    if control["desired_state"] != "running"
                    else "monitor_control_identity_changed"
                ),
                now=timestamp,
            )
            ramp["last_observation"] = record
            _save_control(state_dir, control, now=timestamp)
            return control

        epochs = control["throughput_epochs"]
        epoch = epochs[-1] if epochs and epochs[-1].get("closed_at") is None else None
        fleet_generation = str(observation.get("fleet_generation", ""))
        if isinstance(epoch, dict) and epoch.get("fleet_generation") != fleet_generation:
            _close_open_epoch(control, reason="material_fleet_change", now=timestamp)
            _reset_admission_ramp(
                state_dir,
                control,
                reason="material_fleet_change",
                now=timestamp,
                ceiling=24,
                clear_last_observation=True,
            )
            ramp["last_observation"] = record
            _save_control(state_dir, control, now=timestamp)
            return control
        epoch_matches = bool(
            isinstance(epoch, dict)
            and epoch.get("fleet_generation") == fleet_generation
            and epoch.get("rollout_generation") == control["rollout_generation"]
        )

        critical_evidence = list(observation["critical_finding_keys"])
        blocking_evidence = list(observation["promotion_blocking_finding_keys"])
        active_blocking = _active_ramp_blocking_alerts(control)
        semantic_clean = observation.get("semantic_integrity_clean")
        evidence_clean = bool(
            observation.get("production_health_clean") is True
            and not blocking_evidence
            and (
                observation["cadence"] == "health"
                or semantic_clean is True
            )
        )
        clean = bool(evidence_clean and not active_blocking and epoch_matches)
        record["clean"] = clean
        ramp["last_observation"] = record

        window = ramp.get("window")
        if isinstance(window, dict):
            last = window["observations"][-1]
            gap = timestamp - float(last["timestamp"])
            window_identity_matches = bool(
                window["rollout_generation"] == control["rollout_generation"]
                and window["fleet_generation"] == fleet_generation
                and isinstance(epoch, dict)
                and window["throughput_epoch"] == epoch.get("epoch")
                and window["ceiling"] == control["admission"]["current_ceiling"]
            )
            if not window_identity_matches:
                _reset_admission_ramp(
                    state_dir,
                    control,
                    reason="ramp_identity_changed",
                    now=timestamp,
                    ceiling=(24 if window["fleet_generation"] != fleet_generation else None),
                )
                window = None
            elif gap > ADMISSION_RAMP_MAX_OBSERVATION_GAP_SECONDS:
                _reset_admission_ramp(
                    state_dir,
                    control,
                    reason="health_observation_gap",
                    now=timestamp,
                )
                window = None
            elif not clean:
                _reset_admission_ramp(
                    state_dir,
                    control,
                    reason=(
                        "promotion_blocking_alert"
                        if blocking_evidence or active_blocking
                        else "unclean_monitor_observation"
                    ),
                    now=timestamp,
                    ceiling=(24 if blocking_evidence or active_blocking else None),
                )
                window = None

        if not clean:
            if (
                (blocking_evidence or active_blocking)
                and control["admission"]["current_ceiling"] != 24
            ):
                _reset_admission_ramp(
                    state_dir,
                    control,
                    reason="promotion_blocking_alert",
                    now=timestamp,
                    ceiling=24,
                )
            ramp["last_action"] = {
                "action": "blocked",
                "reason": (
                    "promotion_blocking_alert"
                    if blocking_evidence or active_blocking
                    else "unclean_monitor_observation"
                ),
                "at": utc_timestamp(timestamp),
                "timestamp": timestamp,
                "critical_finding_keys": critical_evidence,
                "promotion_blocking_finding_keys": blocking_evidence,
                "active_promotion_blocking_alerts": active_blocking,
                "evidence": str(resolved_path),
                "evidence_sha256": observed_sha256,
            }
            _save_control(state_dir, control, now=timestamp)
            return control

        if control["admission"]["current_ceiling"] == ADMISSION_RAMP_STAGES[-1]:
            ramp["window"] = None
            ramp["last_action"] = {
                "action": "terminal_stage_healthy",
                "at": utc_timestamp(timestamp),
                "timestamp": timestamp,
                "ceiling": ADMISSION_RAMP_STAGES[-1],
                "evidence": str(resolved_path),
                "evidence_sha256": observed_sha256,
            }
            _save_control(state_dir, control, now=timestamp)
            return control

        if ramp.get("window") is None:
            if observation["cadence"] not in {"semantic", "daily"}:
                ramp["last_action"] = {
                    "action": "waiting_for_semantic_baseline",
                    "at": utc_timestamp(timestamp),
                    "timestamp": timestamp,
                    "evidence": str(resolved_path),
                    "evidence_sha256": observed_sha256,
                }
                _save_control(state_dir, control, now=timestamp)
                return control
            assert isinstance(epoch, dict)
            _start_admission_ramp_window(
                ramp,
                control=control,
                observation=observation,
                record=record,
                epoch=epoch,
                now=timestamp,
            )
            append_transition(
                state_dir,
                control,
                event="admission_ramp_window_started",
                details={
                    "ceiling": control["admission"]["current_ceiling"],
                    "rollout_generation": control["rollout_generation"],
                    "fleet_generation": fleet_generation,
                    "throughput_epoch": epoch["epoch"],
                    "baseline_run_validated_qids": observation[
                        "run_validated_qids"
                    ],
                    "evidence_sha256": observed_sha256,
                },
                now=timestamp,
            )
            ramp["last_action"] = {
                "action": "window_started",
                "at": utc_timestamp(timestamp),
                "timestamp": timestamp,
                "ceiling": control["admission"]["current_ceiling"],
            }
            _save_control(state_dir, control, now=timestamp)
            return control

        window = ramp["window"]
        window["observations"].append(record)
        if observation["cadence"] in {"semantic", "daily"}:
            qids = observation["run_validated_qids"]
            assert _valid_run_qid_map(qids)
            if any(
                qids[run_id] < window["latest_run_validated_qids"][run_id]
                for run_id in REQUIRED_RUNS
            ):
                _reset_admission_ramp(
                    state_dir,
                    control,
                    reason="per_run_validated_qid_regression",
                    now=timestamp,
                )
                _save_control(state_dir, control, now=timestamp)
                return control
            window["latest_run_validated_qids"] = copy.deepcopy(qids)

        ceiling = int(control["admission"]["current_ceiling"])
        requirement = ADMISSION_RAMP_REQUIREMENTS.get(ceiling)
        if requirement is None:
            _reset_admission_ramp(
                state_dir,
                control,
                reason="terminal_stage",
                now=timestamp,
            )
            _save_control(state_dir, control, now=timestamp)
            return control
        elapsed = timestamp - float(window["started_timestamp"])
        all_runs_progressed = all(
            window["latest_run_validated_qids"][run_id]
            > window["baseline_run_validated_qids"][run_id]
            for run_id in REQUIRED_RUNS
        )
        all_run_progress_required = ceiling == 24
        progress_gate_passed = bool(
            all_runs_progressed or not all_run_progress_required
        )
        can_promote = bool(
            observation["cadence"] in {"semantic", "daily"}
            and elapsed >= float(requirement["clean_seconds"])
            and progress_gate_passed
        )
        if not can_promote:
            ramp["last_action"] = {
                "action": "window_advanced",
                "at": utc_timestamp(timestamp),
                "timestamp": timestamp,
                "ceiling": ceiling,
                "clean_seconds": elapsed,
                "required_clean_seconds": requirement["clean_seconds"],
                "all_runs_progressed": all_runs_progressed,
                "all_run_progress_required": all_run_progress_required,
            }
            _save_control(state_dir, control, now=timestamp)
            return control

        evidence_chain_sha256 = _verify_admission_ramp_window_evidence(
            state_dir, window["observations"]
        )
        validate_readiness(control, state_dir=state_dir, verify_files=True)
        next_ceiling = int(requirement["next_ceiling"])
        promotion = {
            "from_ceiling": ceiling,
            "to_ceiling": next_ceiling,
            "at": utc_timestamp(timestamp),
            "timestamp": timestamp,
            "rollout_generation": control["rollout_generation"],
            "fleet_generation": fleet_generation,
            "throughput_epoch": epoch["epoch"],
            "clean_seconds": elapsed,
            "required_clean_seconds": requirement["clean_seconds"],
            "all_runs_progressed": all_runs_progressed,
            "all_run_progress_required": all_run_progress_required,
            "baseline_run_validated_qids": copy.deepcopy(
                window["baseline_run_validated_qids"]
            ),
            "final_run_validated_qids": copy.deepcopy(
                window["latest_run_validated_qids"]
            ),
            "observations": copy.deepcopy(window["observations"]),
            "evidence_sha256": evidence_chain_sha256,
        }
        ramp["promotions"].append(promotion)
        control["admission"]["current_ceiling"] = next_ceiling
        ramp["current_ceiling"] = next_ceiling
        if next_ceiling == ADMISSION_RAMP_STAGES[-1]:
            ramp["window"] = None
        else:
            _start_admission_ramp_window(
                ramp,
                control=control,
                observation=observation,
                record=record,
                epoch=epoch,
                now=timestamp,
            )
        ramp["last_action"] = {
            "action": "promoted",
            "at": utc_timestamp(timestamp),
            "timestamp": timestamp,
            "previous_ceiling": ceiling,
            "current_ceiling": next_ceiling,
            "evidence_sha256": promotion["evidence_sha256"],
        }
        append_transition(
            state_dir,
            control,
            event="admission_ceiling_promoted",
            details={
                "previous": ceiling,
                "current": next_ceiling,
                "rollout_generation": control["rollout_generation"],
                "fleet_generation": fleet_generation,
                "throughput_epoch": epoch["epoch"],
                "clean_seconds": elapsed,
                "evidence_sha256": promotion["evidence_sha256"],
            },
            now=timestamp,
        )
        _save_control(state_dir, control, now=timestamp)
        return control


def _submit_line_comment(command: str, *, source: str) -> str | None:
    """Recover one scheduler token from an accounting SubmitLine.

    ``AccountingStoreFlags=(null)`` on the production cluster means sacct's Comment
    field is empty.  Controller/fleet submissions also place the exact token on the
    sbatch CLI, whose SubmitLine *is* retained.  Duplicate or malformed options are an
    ambiguity rather than permission to guess.
    """

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise SchedulerAmbiguity(f"invalid {source} SubmitLine quoting: {exc}") from exc
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token.startswith("--comment="):
            values.append(token.split("=", 1)[1])
        elif token == "--comment":
            if index + 1 >= len(tokens):
                raise SchedulerAmbiguity(
                    f"{source} SubmitLine contains a valueless --comment"
                )
            values.append(tokens[index + 1])
    if len(values) > 1:
        raise SchedulerAmbiguity(
            f"{source} SubmitLine contains duplicate --comment options"
        )
    return values[0] if values else None


def parse_scheduler_rows(text: str, *, source: str) -> list[SchedulerJob]:
    """Parse the common scheduler row, including scheduler-owned dependency state."""
    rows: list[SchedulerJob] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.rstrip("\n").split("|", 5)
        if len(fields) != 6:
            raise SchedulerAmbiguity(
                f"malformed {source} scheduler row {line_number}: {raw!r}"
            )
        job_id, name, state, comment, command, dependency = (
            field.strip() for field in fields
        )
        if source == "sacct":
            derived = _submit_line_comment(command, source=source)
            stored = "" if comment.lower() in {"", "(null)", "null", "none"} else comment
            if stored and derived and stored != derived:
                raise SchedulerAmbiguity(
                    f"sacct Comment/SubmitLine conflict for job {job_id}"
                )
            comment = stored or derived or ""
        if not job_id or not name or not state:
            raise SchedulerAmbiguity(
                f"incomplete {source} scheduler row {line_number}: {raw!r}"
            )
        rows.append(
            SchedulerJob(
                job_id=job_id,
                job_name=name,
                state=state,
                comment=comment,
                command=command,
                source=source,
                dependency=dependency,
            )
        )
    return rows


def _run_subprocess(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        check=False,
        timeout=15.0,
    )


def query_scheduler(
    *,
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
    user: str | None = None,
    now: float | None = None,
    tolerate_errors: bool = False,
) -> SchedulerSnapshot:
    """Join live ``squeue`` and accounting ``sacct`` truth by exact raw job ID."""
    invoke = runner or _run_subprocess
    scheduler_user = user or os.environ.get("USER")
    if not scheduler_user:
        raise ControlError("USER is unset; scheduler queries cannot be scoped safely")
    timestamp = time.time() if now is None else float(now)
    commands = (
        (
            "squeue",
            [
                "squeue",
                "-u",
                scheduler_user,
                "-h",
                "-r",
                "-o",
                "%i|%j|%T|%k|%o|%E",
            ],
        ),
        (
            "sacct",
            [
                "sacct",
                "-u",
                scheduler_user,
                "-n",
                "-P",
                "-S",
                time.strftime("%Y-%m-%d", time.localtime(timestamp - 7 * 86_400)),
                "--format=JobIDRaw,JobName,State,Comment,SubmitLine,Dependency",
            ],
        ),
    )
    by_id: dict[str, SchedulerJob] = {}
    errors: list[str] = []
    succeeded: dict[str, bool] = {"squeue": False, "sacct": False}
    # Accounting is loaded first; live queue rows then override stale accounting state.
    outputs: dict[str, str] = {}
    for source, command in commands:
        try:
            proc = invoke(command)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{source} failed: {exc}")
            continue
        if proc.returncode != 0:
            errors.append(
                f"{source} failed rc={proc.returncode}: {proc.stderr.strip()[:500]}"
            )
            continue
        succeeded[source] = True
        outputs[source] = proc.stdout
    for source in ("sacct", "squeue"):
        if source not in outputs:
            continue
        for row in parse_scheduler_rows(outputs[source], source=source):
            # sacct includes job steps; they are never independently submitted control
            # jobs and would make a parent job appear duplicated.
            if source == "sacct" and "." in row.job_id:
                continue
            prior = by_id.get(row.job_id)
            if source == "squeue" and prior is not None and (
                prior.job_name != row.job_name
                or (
                    bool(prior.comment)
                    and prior.comment != row.comment
                )
            ):
                raise SchedulerAmbiguity(
                    f"squeue/sacct identity conflict for scheduler job {row.job_id}"
                )
            if (
                source == "squeue"
                and prior is not None
                and row.dependency.strip().lower() in {"", "(null)", "null", "none"}
                and prior.dependency.strip().lower()
                not in {"", "(null)", "null", "none"}
            ):
                # A satisfied dependency may disappear from live ``squeue %E`` while
                # accounting retains the immutable submission edge.  Keep live state
                # and command provenance, but preserve that accounting dependency.
                row = SchedulerJob(
                    job_id=row.job_id,
                    job_name=row.job_name,
                    state=row.state,
                    comment=row.comment,
                    command=row.command,
                    source=row.source,
                    dependency=prior.dependency,
                )
            by_id[row.job_id] = row
    if errors and not tolerate_errors:
        raise ControlError("; ".join(errors))
    return SchedulerSnapshot(
        jobs=tuple(sorted(by_id.values(), key=lambda job: job.job_id)),
        captured_at=timestamp,
        squeue_ok=succeeded["squeue"],
        sacct_ok=succeeded["sacct"],
        errors=tuple(errors),
    )


def _scoped_control_jobs(snapshot: SchedulerSnapshot) -> list[SchedulerJob]:
    return [job for job in snapshot.jobs if parse_job_token(job.comment) is not None]


def _controller_references(control: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    job_ids: set[str] = set()
    tokens: set[str] = set()
    for role in ROLE_NAMES:
        role_state = control["controllers"][role]
        for field in ("active", "successor", "submission_intent"):
            record = role_state.get(field)
            if not isinstance(record, dict):
                continue
            if record.get("job_id"):
                job_ids.add(str(record["job_id"]))
            if record.get("job_token"):
                tokens.add(str(record["job_token"]))
    return job_ids, tokens


def build_reconciliation_report(
    control: Mapping[str, Any],
    snapshot: SchedulerSnapshot,
    *,
    all_jobs: bool,
    no_admit: bool,
) -> dict[str, Any]:
    """Build a fail-closed scheduler/control join without changing admission state."""
    if not all_jobs or not no_admit:
        raise ControlError("schema-5 reconciliation requires both --all and --no-admit")
    referenced_ids, referenced_tokens = _controller_references(control)
    scoped = _scoped_control_jobs(snapshot)
    active_by_token: dict[str, list[SchedulerJob]] = {}
    all_by_token: dict[str, list[SchedulerJob]] = {}
    malformed: list[str] = []
    for job in snapshot.jobs:
        if (
            job.comment.startswith(TOKEN_PREFIX)
            and parse_job_token(job.comment) is None
        ):
            malformed.append(job.job_id)
    for job in scoped:
        assert job.token is not None
        all_by_token.setdefault(job.token, []).append(job)
        if job.active:
            active_by_token.setdefault(job.token, []).append(job)
    ambiguous = {
        token: [job.job_id for job in jobs]
        for token, jobs in active_by_token.items()
        if len(jobs) > 1
    }
    unmappable = [
        job.job_id
        for job in scoped
        if job.active
        and job.job_id not in referenced_ids
        and job.token not in referenced_tokens
    ]
    missing_referenced: list[str] = []
    by_id = {job.job_id: job for job in snapshot.jobs}
    for job_id in referenced_ids:
        record = by_id.get(job_id)
        # Old terminal records in sacct are legitimate history; only references claiming
        # to be active/pending are checked below in the per-role summary.
        if record is None:
            missing_referenced.append(job_id)
    roles: dict[str, Any] = {}
    for role in ROLE_NAMES:
        role_state = control["controllers"][role]
        role_summary: dict[str, Any] = {}
        for field in ("active", "successor", "submission_intent"):
            expected = role_state.get(field)
            if not isinstance(expected, dict):
                role_summary[field] = None
                continue
            matches = all_by_token.get(str(expected.get("job_token", "")), [])
            live_matches = [job for job in matches if job.active]
            role_summary[field] = {
                "recorded_job_id": expected.get("job_id"),
                "intent_token": expected.get("intent_token"),
                "matching_job_ids": [job.job_id for job in matches],
                "live_job_ids": [job.job_id for job in live_matches],
            }
        roles[role] = role_summary
    errors = list(snapshot.errors)
    if not snapshot.squeue_ok:
        errors.append("live squeue truth is unavailable")
    if not snapshot.sacct_ok:
        errors.append("sacct transaction history is unavailable")
    if ambiguous:
        errors.append("duplicate active controller intent tokens")
    if unmappable:
        errors.append("unmappable active schema-5 jobs")
    if malformed:
        errors.append("malformed schema-5 scheduler tokens")
    report = {
        "schema_version": 1,
        "gate": "scheduler_reconciliation",
        "immutable_sha256": control["immutable_sha256"],
        "captured_at": utc_timestamp(snapshot.captured_at),
        "captured_timestamp": snapshot.captured_at,
        "all": True,
        "no_admit": True,
        "scheduler": {
            "squeue_ok": snapshot.squeue_ok,
            "sacct_ok": snapshot.sacct_ok,
            "job_count": len(snapshot.jobs),
            "schema5_job_count": len(scoped),
        },
        "roles": roles,
        "ambiguous_tokens": ambiguous,
        "unmappable_job_ids": sorted(unmappable),
        "malformed_token_job_ids": sorted(malformed),
        "referenced_job_ids_absent_from_window": sorted(missing_referenced),
        "errors": errors,
        "passed": not errors,
    }
    return report


def _adopt_unique_intents(
    control: MutableMapping[str, Any], snapshot: SchedulerSnapshot
) -> list[dict[str, Any]]:
    """Adopt jobs accepted between ``sbatch`` and durable job-ID commit."""
    adopted: list[dict[str, Any]] = []
    by_token: dict[str, list[SchedulerJob]] = {}
    for job in _scoped_control_jobs(snapshot):
        if job.token:
            by_token.setdefault(job.token, []).append(job)
    for role in ROLE_NAMES:
        role_state = control["controllers"][role]
        intent = role_state.get("submission_intent")
        if not isinstance(intent, dict) or intent.get("state") != "submitting":
            continue
        matches = by_token.get(str(intent.get("job_token", "")), [])
        active = [job for job in matches if job.active]
        if len(active) > 1:
            raise SchedulerAmbiguity(
                f"intent {intent.get('intent_token')} maps to multiple active jobs: "
                + ", ".join(job.job_id for job in active)
            )
        match = active[0] if active else (matches[-1] if len(matches) == 1 else None)
        if match is None:
            continue
        intent["state"] = "submitted"
        intent["job_id"] = match.job_id
        intent["adopted_from_scheduler"] = True
        target = str(intent.get("target", "active"))
        if target not in {"active", "successor"}:
            raise ControlError(f"invalid controller intent target {target!r}")
        role_state[target] = copy.deepcopy(intent)
        adopted.append(
            {
                "role": role,
                "target": target,
                "job_id": match.job_id,
                "token": match.token,
            }
        )
    return adopted


def reconcile_control(
    state_dir: Path,
    *,
    snapshot: SchedulerSnapshot,
    all_jobs: bool,
    no_admit: bool,
    now: float | None = None,
) -> dict[str, Any]:
    timestamp = snapshot.captured_at if now is None else float(now)
    with control_lock(state_dir):
        control = load_control(state_dir)
        adopted = _adopt_unique_intents(control, snapshot)
        report = build_reconciliation_report(
            control, snapshot, all_jobs=all_jobs, no_admit=no_admit
        )
        if adopted:
            report["adopted_intents"] = adopted
        _atomic_write_json(state_dir / RECONCILIATION_FILENAME, report)
        report_hash = sha256_file(state_dir / RECONCILIATION_FILENAME)
        control["last_reconciliation"] = {
            "passed": report["passed"],
            "evidence": str((state_dir / RECONCILIATION_FILENAME).resolve()),
            "sha256": report_hash,
            "at": report["captured_at"],
        }
        control["readiness"]["scheduler_reconciliation"] = {
            "passed": report["passed"],
            "evidence": str((state_dir / RECONCILIATION_FILENAME).resolve()),
            "sha256": report_hash,
            "attested_at": utc_timestamp(timestamp),
            "attested_timestamp": timestamp,
        }
        append_transition(
            state_dir,
            control,
            event="scheduler_reconciled",
            details={
                "passed": report["passed"],
                "adopted": adopted,
                "errors": report["errors"],
            },
            now=timestamp,
        )
        _save_control(state_dir, control, now=timestamp)
    return report


def _controller_resource(control: Mapping[str, Any], role: str) -> dict[str, str]:
    resources = control["immutable"].get("controller_resources", {})
    role_resources = resources.get(role, {}) if isinstance(resources, dict) else {}
    return {
        "partition": str(
            role_resources.get("partition", "mit_preemptable,mit_normal,mit_normal_gpu")
        ),
        "memory": str(role_resources.get("memory", "4G")),
        "time_limit": str(role_resources.get("time_limit", "12:00:00")),
    }


def _runtime_environment_pins(control: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    immutable = control["immutable"]
    return {
        role: {
            "prefix": str(immutable[f"{role}_environment_prefix"]),
            "manifest_path": str(immutable[f"{role}_environment_manifest_path"]),
            "manifest_sha256": str(immutable[f"{role}_environment_sha256"]),
        }
        for role in ("harness", "serving")
    }


def ensure_runtime_integrity_attestation(
    state_dir: Path,
    control: Mapping[str, Any],
    *,
    generation: int,
    force_full: bool,
    now: float | None = None,
) -> dict[str, Any]:
    """Create/verify the one full-inventory attestation for a rollout generation."""

    try:
        result = runtime_integrity.ensure_generation_attestation(
            state_dir=state_dir,
            generation=generation,
            release_id=str(control["immutable"]["release_id"]),
            release_bundle_id=str(control["immutable"]["release_bundle_id"]),
            immutable_pins_sha256=str(control["immutable_sha256"]),
            environment_pins=_runtime_environment_pins(control),
            force_full=force_full,
        )
    except runtime_integrity.RuntimeIntegrityError as exc:
        raise ImmutablePinError(f"runtime environment integrity failed: {exc}") from exc
    try:
        lease = runtime_integrity.refresh_generation_lease(
            state_dir=state_dir,
            attestation_path=Path(str(result["path"])),
            attestation_sha256=str(result["sha256"]),
            generation=generation,
            release_id=str(control["immutable"]["release_id"]),
            immutable_pins_sha256=str(control["immutable_sha256"]),
            expected_environment_hashes={
                role: str(control["immutable"][f"{role}_environment_sha256"])
                for role in ("harness", "serving")
            },
            expected_prefixes={
                role: str(control["immutable"][f"{role}_environment_prefix"])
                for role in ("harness", "serving")
            },
            now=now,
        )
    except runtime_integrity.RuntimeIntegrityError as exc:
        raise ImmutablePinError(f"runtime environment lease failed: {exc}") from exc
    return {
        "schema_version": runtime_integrity.ATTESTATION_SCHEMA_VERSION,
        "generation": generation,
        "path": str(result["path"]),
        "sha256": str(result["sha256"]),
        "attestation_id": str(result["record"]["attestation_id"]),
        "lease_path": str(lease["path"]),
    }


def ensure_snapshot_integrity_attestation(
    state_dir: Path,
    control: Mapping[str, Any],
    *,
    generation: int,
    validation_context: _SnapshotValidationContext,
) -> dict[str, Any]:
    """Seal one already-completed full snapshot validation for a rollout."""

    if not validation_context.full or len(validation_context.proofs) != 2:
        raise ReadinessError(
            "snapshot generation sealing requires one full validation of two snapshots"
        )
    gate = control["readiness"].get("snapshot", {})
    evidence = gate.get("evidence")
    evidence_sha256 = gate.get("sha256")
    if (
        not isinstance(evidence, str)
        or not Path(evidence).is_absolute()
        or not isinstance(evidence_sha256, str)
        or _SHA256_RE.fullmatch(evidence_sha256) is None
    ):
        raise ReadinessError("snapshot gate lacks checksummed evidence for sealing")
    try:
        sealed = snapshot_integrity.ensure_generation_seal(
            state_dir=state_dir,
            generation=generation,
            immutable_pins_sha256=str(control["immutable_sha256"]),
            snapshot_gate_evidence_path=Path(evidence),
            snapshot_gate_evidence_sha256=evidence_sha256,
            snapshots=list(validation_context.proofs.values()),
            sealed_members=list(validation_context.members.values()),
            metadata_entries=list(validation_context.metadata.values()),
        )
        lease = snapshot_integrity.refresh_generation_lease(
            state_dir=state_dir,
            seal_path=Path(str(sealed["path"])),
            seal_sha256=str(sealed["sha256"]),
            generation=generation,
            immutable_pins_sha256=str(control["immutable_sha256"]),
            # Lease time is an actual filesystem-observation time, not the optional
            # transaction timestamp used by deterministic controller tests/logs.
            now=None,
            force=True,
        )
    except snapshot_integrity.SnapshotIntegrityError as exc:
        raise ReadinessError(f"snapshot generation sealing failed: {exc}") from exc
    return {
        "schema_version": snapshot_integrity.SEAL_SCHEMA_VERSION,
        "generation": generation,
        "path": str(sealed["path"]),
        "sha256": str(sealed["sha256"]),
        "seal_id": str(sealed["record"]["seal_id"]),
        "lease_path": str(lease["path"]),
        "member_bindings": {},
    }


def prepare_snapshot_integrity_preseal(
    state_dir: Path,
    *,
    evidence_path: Path,
    evidence_sha256: str,
    validation_context: _SnapshotValidationContext,
    now: float | None = None,
) -> dict[str, Any]:
    """Persist the initial full proof for the paused control's next generation.

    The snapshot readiness builder has already performed the expensive validation on
    its marker-last candidate.  Publishing this preseal after the candidate is renamed
    lets the subsequent ``attest`` command and all other readiness builders reuse that
    proof instead of re-reading ~96 GiB.  Admission remains disabled and the seal is
    not authoritative until the snapshot gate binds the same evidence hash.
    """

    timestamp = time.time() if now is None else float(now)
    evidence = evidence_path.expanduser().resolve()
    if sha256_file(evidence) != evidence_sha256:
        raise ReadinessError("snapshot preseal evidence changed before publication")
    with control_lock(state_dir):
        current = load_control(state_dir)
        if current["desired_state"] != "paused" or current.get("drain_requested"):
            raise ReadinessError("snapshot preseal requires paused, non-draining control")
        target_generation = int(current["rollout_generation"]) + 1
        existing = current.get(SNAPSHOT_ATTESTATION_STATE_KEY)
        if isinstance(existing, dict):
            seal = validate_snapshot_integrity_attestation(
                current, state_dir=state_dir, verify_lease=False
            )
            if (
                existing.get("generation") != target_generation
                or seal.get("snapshot_gate_evidence_path") != str(evidence)
                or seal.get("snapshot_gate_evidence_sha256") != evidence_sha256
            ):
                raise ReadinessError("existing snapshot preseal binds different evidence")
            refresh_snapshot_integrity_lease(state_dir, current)
            return copy.deepcopy(existing)
        projected = copy.deepcopy(current)
        projected["readiness"]["snapshot"] = {
            "passed": True,
            "evidence": str(evidence),
            "sha256": evidence_sha256,
            "attested_at": utc_timestamp(timestamp),
            "attested_timestamp": timestamp,
        }
        record = ensure_snapshot_integrity_attestation(
            state_dir,
            projected,
            generation=target_generation,
            validation_context=validation_context,
        )
        current[SNAPSHOT_ATTESTATION_STATE_KEY] = record
        append_transition(
            state_dir,
            current,
            event="snapshot_integrity_presealed",
            details={
                "generation": target_generation,
                "evidence": str(evidence),
                "sha256": evidence_sha256,
                "seal_id": record["seal_id"],
            },
            now=timestamp,
        )
        _save_control(state_dir, current, now=timestamp)
        return copy.deepcopy(record)


def validate_snapshot_integrity_attestation(
    control: Mapping[str, Any], *, state_dir: Path, verify_lease: bool
) -> dict[str, Any]:
    """Validate the compact rollout seal and, on hot paths, its bounded lease."""

    record = control.get(SNAPSHOT_ATTESTATION_STATE_KEY)
    generation = int(control.get("rollout_generation", 0))
    required = {
        "schema_version",
        "generation",
        "path",
        "sha256",
        "seal_id",
        "lease_path",
        "member_bindings",
    }
    if (
        not isinstance(record, dict)
        or set(record) != required
        or record.get("schema_version") != snapshot_integrity.SEAL_SCHEMA_VERSION
        or record.get("generation")
        != (
            generation
            if generation >= 1
            else generation + 1
        )
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
        or not isinstance(record.get("lease_path"), str)
        or not isinstance(record.get("member_bindings"), dict)
    ):
        raise ReadinessError("control snapshot integrity seal is malformed")
    resolved_state = Path(state_dir).expanduser().resolve()
    record_generation = int(record["generation"])
    expected_seal_path = (
        resolved_state
        / "snapshot_integrity"
        / f"snapshot.g{record_generation:06d}.json"
    )
    expected_lease_path = snapshot_integrity.generation_lease_path(
        resolved_state, record_generation
    ).resolve()
    if (
        Path(str(record["path"])).expanduser().resolve() != expected_seal_path
        or Path(str(record["lease_path"])).expanduser().resolve()
        != expected_lease_path
        or str(record["path"]) != str(expected_seal_path)
        or str(record["lease_path"]) != str(expected_lease_path)
    ):
        raise ReadinessError(
            "control snapshot integrity seal/lease path escapes its state generation"
        )
    try:
        seal = snapshot_integrity.verify_generation_seal(
            path=Path(record["path"]),
            expected_sha256=str(record["sha256"]),
            generation=int(record["generation"]),
            immutable_pins_sha256=str(control["immutable_sha256"]),
        )
        if verify_lease:
            snapshot_integrity.verify_generation_lease(
                lease_path=Path(record["lease_path"]),
                seal_path=Path(record["path"]),
                seal_sha256=str(record["sha256"]),
                generation=int(record["generation"]),
                immutable_pins_sha256=str(control["immutable_sha256"]),
            )
    except snapshot_integrity.SnapshotIntegrityError as exc:
        raise ReadinessError(f"snapshot integrity validation failed: {exc}") from exc
    gate = control.get("readiness", {}).get("snapshot", {})
    if seal.get("seal_id") != record.get("seal_id"):
        raise ReadinessError("control snapshot seal ID drifted")
    if gate.get("passed") is True and (
        seal.get("snapshot_gate_evidence_path") != gate.get("evidence")
        or seal.get("snapshot_gate_evidence_sha256") != gate.get("sha256")
    ):
        raise ReadinessError("control snapshot seal differs from snapshot gate evidence")
    snapshot_id_by_root = {
        str(row["snapshot_root"]): str(row["snapshot_id"])
        for row in seal.get("snapshots", [])
        if isinstance(row, dict)
    }
    for bound_gate, binding in record["member_bindings"].items():
        readiness_gate = control.get("readiness", {}).get(bound_gate, {})
        if (
            bound_gate not in REQUIRED_GATES
            or not isinstance(binding, dict)
            or set(binding) != {"evidence_sha256", "members"}
            or binding.get("evidence_sha256") != readiness_gate.get("sha256")
            or readiness_gate.get("passed") is not True
            or not isinstance(binding.get("members"), list)
        ):
            raise ReadinessError("control snapshot member binding is malformed or stale")
        seen: set[tuple[str, str]] = set()
        for member in binding["members"]:
            if (
                not isinstance(member, dict)
                or set(member)
                != {"snapshot_id", "snapshot_root", "logical_path", "sha256"}
                or snapshot_id_by_root.get(str(member.get("snapshot_root")))
                != member.get("snapshot_id")
                or _SHA256_RE.fullmatch(str(member.get("sha256", ""))) is None
                or not isinstance(member.get("logical_path"), str)
                or not member["logical_path"]
            ):
                raise ReadinessError("control snapshot member binding row is invalid")
            key = (str(member["snapshot_root"]), str(member["logical_path"]))
            if key in seen:
                raise ReadinessError("control snapshot member binding repeats a path")
            seen.add(key)
    return seal


def refresh_snapshot_integrity_lease(
    state_dir: Path,
    control: Mapping[str, Any],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Centrally metadata-scan snapshots at most once per five-minute interval."""

    validate_snapshot_integrity_attestation(
        control, state_dir=state_dir, verify_lease=False
    )
    record = control[SNAPSHOT_ATTESTATION_STATE_KEY]
    try:
        return snapshot_integrity.refresh_generation_lease(
            state_dir=state_dir,
            seal_path=Path(str(record["path"])),
            seal_sha256=str(record["sha256"]),
            generation=int(record["generation"]),
            immutable_pins_sha256=str(control["immutable_sha256"]),
            now=now,
        )
    except snapshot_integrity.SnapshotIntegrityError as exc:
        raise ReadinessError(f"snapshot integrity lease refresh failed: {exc}") from exc


def _start_snapshot_lease_refresh(
    state_dir: Path,
    control: Mapping[str, Any],
) -> _SnapshotLeaseRefreshTask:
    """Start one daemon renewal without blocking the controller heartbeat loop.

    ``refresh_generation_lease`` retains the cross-node singleton lock and five-minute
    cache.  The thread only removes filesystem latency and lock wait from the controller
    liveness path; it does not permit a second local renewal while its task is retained.
    Production intentionally supplies no synthetic timestamp so the lease reflects an
    actual filesystem observation.
    """

    future: Future[dict[str, Any]] = Future()
    frozen_control = copy.deepcopy(dict(control))

    def renew() -> None:
        try:
            result = refresh_snapshot_integrity_lease(
                state_dir,
                frozen_control,
                now=None,
            )
        except BaseException as exc:  # preserve every worker failure for the owner
            future.set_exception(exc)
        else:
            future.set_result(result)

    thread = threading.Thread(
        target=renew,
        name=(
            "schema5-snapshot-lease-"
            f"g{int(control.get('rollout_generation', 0)):06d}"
        ),
        daemon=True,
    )
    thread.start()
    return _SnapshotLeaseRefreshTask(future=future, thread=thread)


def _await_snapshot_lease_refresh_with_heartbeats(
    task: _SnapshotLeaseRefreshTask,
    *,
    state_dir: Path,
    role: str,
    generation: int,
    intent_token: str,
    job_id: str,
    heartbeat_interval: float,
) -> dict[str, Any]:
    """Wait for startup renewal while proving the exact controller remains alive."""

    if heartbeat_interval <= 0:
        raise ControlError("snapshot renewal heartbeat interval must be positive")
    # The prior operation (notably runtime-environment validation) may itself have used
    # most of an interval.  Heartbeat immediately, then at every timeout until renewal
    # completes.  ``heartbeat_controller`` rechecks the fenced role identity each time.
    heartbeat_controller(
        state_dir,
        role=role,
        generation=generation,
        intent_token=intent_token,
        job_id=job_id,
        now=time.time(),
    )
    while True:
        try:
            # Half-interval waits leave headroom for control-lock and shared-filesystem
            # latency inside the heartbeat write itself while preserving the advertised
            # maximum cadence.
            return task.future.result(timeout=heartbeat_interval / 2.0)
        except FutureTimeoutError:
            heartbeat_controller(
                state_dir,
                role=role,
                generation=generation,
                intent_token=intent_token,
                job_id=job_id,
                now=time.time(),
            )


def validate_runtime_integrity_attestation(
    control: Mapping[str, Any], *, verify_metadata: bool
) -> dict[str, Any]:
    """Validate the control-bound attestation without ever refreshing drift."""

    record = control.get(RUNTIME_ATTESTATION_STATE_KEY)
    generation = int(control.get("rollout_generation", 0))
    if not isinstance(record, dict):
        raise ImmutablePinError("control lacks a runtime integrity attestation")
    required = {
        "schema_version",
        "generation",
        "path",
        "sha256",
        "attestation_id",
        "lease_path",
    }
    if (
        set(record) != required
        or record.get("schema_version") != runtime_integrity.ATTESTATION_SCHEMA_VERSION
        or record.get("generation") != generation
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("sha256"), str)
    ):
        raise ImmutablePinError("control runtime integrity attestation is malformed")
    immutable = control["immutable"]
    try:
        attestation = runtime_integrity.verify_generation_attestation(
            path=Path(record["path"]),
            expected_sha256=str(record["sha256"]),
            generation=generation,
            release_id=str(immutable["release_id"]),
            immutable_pins_sha256=str(control["immutable_sha256"]),
            expected_environment_hashes={
                role: str(immutable[f"{role}_environment_sha256"])
                for role in ("harness", "serving")
            },
            expected_prefixes={
                role: str(immutable[f"{role}_environment_prefix"])
                for role in ("harness", "serving")
            },
            verify_metadata=False,
        )
    except runtime_integrity.RuntimeIntegrityError as exc:
        raise ImmutablePinError(f"runtime environment integrity failed: {exc}") from exc
    if attestation.get("attestation_id") != record.get("attestation_id"):
        raise ImmutablePinError("control runtime attestation ID drifted")
    if attestation.get("release_bundle_id") != immutable["release_bundle_id"]:
        raise ImmutablePinError("control runtime attestation release bundle drifted")
    if verify_metadata:
        try:
            runtime_integrity.verify_generation_lease(
                lease_path=Path(str(record["lease_path"])),
                attestation_path=Path(str(record["path"])),
                attestation_sha256=str(record["sha256"]),
                generation=generation,
                release_id=str(immutable["release_id"]),
                immutable_pins_sha256=str(control["immutable_sha256"]),
                expected_environment_hashes={
                    role: str(immutable[f"{role}_environment_sha256"])
                    for role in ("harness", "serving")
                },
                expected_prefixes={
                    role: str(immutable[f"{role}_environment_prefix"])
                    for role in ("harness", "serving")
                },
            )
        except runtime_integrity.RuntimeIntegrityError as exc:
            raise ImmutablePinError(f"runtime environment lease failed: {exc}") from exc
    return copy.deepcopy(record)


def refresh_runtime_integrity_lease(
    state_dir: Path,
    control: Mapping[str, Any],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Run the centralized metadata scan only when the five-minute lease is due."""

    record = validate_runtime_integrity_attestation(control, verify_metadata=False)
    immutable = control["immutable"]
    try:
        return runtime_integrity.refresh_generation_lease(
            state_dir=state_dir,
            attestation_path=Path(str(record["path"])),
            attestation_sha256=str(record["sha256"]),
            generation=int(control["rollout_generation"]),
            release_id=str(immutable["release_id"]),
            immutable_pins_sha256=str(control["immutable_sha256"]),
            expected_environment_hashes={
                role: str(immutable[f"{role}_environment_sha256"])
                for role in ("harness", "serving")
            },
            expected_prefixes={
                role: str(immutable[f"{role}_environment_prefix"])
                for role in ("harness", "serving")
            },
            now=now,
        )
    except runtime_integrity.RuntimeIntegrityError as exc:
        raise ImmutablePinError(f"runtime environment lease refresh failed: {exc}") from exc


def _start_runtime_lease_refresh(
    state_dir: Path,
    control: Mapping[str, Any],
) -> _RuntimeLeaseRefreshTask:
    """Start one daemon renewal without blocking controller heartbeats.

    ``refresh_generation_lease`` still owns the cross-node singleton lock and its
    five-minute cache.  Retaining the returned task prevents a controller from
    starting a second local renewal while metadata verification or lock acquisition
    is still in progress.  Production deliberately uses the observation time from
    inside the renewal rather than a timestamp captured before a possible stall.
    """

    future: Future[dict[str, Any]] = Future()
    frozen_control = copy.deepcopy(dict(control))

    def renew() -> None:
        try:
            result = refresh_runtime_integrity_lease(
                state_dir,
                frozen_control,
                now=None,
            )
        except BaseException as exc:  # preserve every worker failure for the owner
            future.set_exception(exc)
        else:
            future.set_result(result)

    thread = threading.Thread(
        target=renew,
        name=(
            "schema5-runtime-lease-"
            f"g{int(control.get('rollout_generation', 0)):06d}"
        ),
        daemon=True,
    )
    thread.start()
    return _RuntimeLeaseRefreshTask(future=future, thread=thread)


def _await_runtime_lease_refresh_with_heartbeats(
    task: _RuntimeLeaseRefreshTask,
    *,
    state_dir: Path,
    role: str,
    generation: int,
    intent_token: str,
    job_id: str,
    heartbeat_interval: float,
) -> dict[str, Any]:
    """Wait for startup renewal while proving the exact controller remains alive."""

    if heartbeat_interval <= 0:
        raise ControlError("runtime renewal heartbeat interval must be positive")
    # Callers fence and heartbeat immediately before starting the task.  Heartbeat
    # again here so this helper is safe on its own, then at half-interval while the
    # cross-node lock or metadata scan remains stalled.
    heartbeat_controller(
        state_dir,
        role=role,
        generation=generation,
        intent_token=intent_token,
        job_id=job_id,
        now=time.time(),
    )
    while True:
        try:
            return task.future.result(timeout=heartbeat_interval / 2.0)
        except FutureTimeoutError:
            heartbeat_controller(
                state_dir,
                role=role,
                generation=generation,
                intent_token=intent_token,
                job_id=job_id,
                now=time.time(),
            )


def production_environment(
    control: Mapping[str, Any], *, run_id: str | None = None
) -> dict[str, str]:
    """Return the exact provenance environment for controllers or one cell batch.

    The narrow dispatcher integration should call this function immediately before
    rendering a batch.  Supplying ``run_id`` adds the immutable per-run policy digest;
    multi-run arrays must instead be split by run so every worker receives one policy.
    """
    immutable = control["immutable"]
    attestation = (
        validate_runtime_integrity_attestation(control, verify_metadata=False)
        if int(control.get("rollout_generation", 0)) >= 1
        else None
    )
    environment = {
        "ASYS_RELEASE_ID": str(immutable["release_id"]),
        "ASYS_MODEL_CONTRACT_SHA256": str(immutable["model_contract_sha256"]),
        "ASYS_FLEET_CONTRACT_SHA256": str(immutable["fleet_contract_sha256"]),
        "ASYS_HARNESS_ENVIRONMENT_SHA256": str(immutable["harness_environment_sha256"]),
        "ASYS_SERVING_ENVIRONMENT_SHA256": str(immutable["serving_environment_sha256"]),
        "ASYS_ROLLOUT_GENERATION": str(control["rollout_generation"]),
        "ASYS_IMMUTABLE_PINS_SHA256": str(control["immutable_sha256"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    if attestation is not None:
        environment.update(
            {
                "ASYS_RUNTIME_ATTESTATION": str(attestation["path"]),
                "ASYS_RUNTIME_ATTESTATION_SHA256": str(attestation["sha256"]),
                "ASYS_RUNTIME_INTEGRITY_LEASE": str(attestation["lease_path"]),
            }
        )
    if run_id is not None:
        matches = [run for run in immutable["runs"] if run["run_id"] == run_id]
        if len(matches) != 1:
            raise ImmutablePinError(f"run {run_id!r} is not uniquely pinned by control")
        environment["ASYS_ARTIFACT_POLICY_SHA256"] = str(matches[0]["policy_sha256"])
    return environment


def _require_production_running(control: Mapping[str, Any]) -> None:
    """Fence every admission-facing hook until transactional resume is complete."""

    if control.get("desired_state") != "running" or control.get("drain_requested"):
        raise ControlError("schema-5 admission is disabled by desired state")
    intent = control.get("resume_intent")
    if not isinstance(intent, dict) or intent.get("state") != "complete":
        raise ControlError(
            "schema-5 admission is disabled by incomplete resume transaction"
        )
    controller_ids = intent.get("controller_job_ids")
    if (
        not isinstance(controller_ids, dict)
        or set(controller_ids) != set(ROLE_NAMES)
        or not all(str(controller_ids[role]).isdigit() for role in ROLE_NAMES)
        or intent.get("scheduler_visible_job_ids") != controller_ids
    ):
        raise ControlError(
            "schema-5 admission is disabled without both controller chains"
        )
    for role in ROLE_NAMES:
        active = control["controllers"][role].get("active")
        if not isinstance(active, dict) or not str(active.get("job_id", "")):
            raise ControlError(
                f"schema-5 admission is disabled without a current {role} controller"
            )


def production_environment_from_state(
    state_dir: Path, *, run_id: str | None = None, require_running: bool = True
) -> dict[str, str]:
    """Fail-closed public hook for dispatcher batch rendering."""
    control = load_control(state_dir, verify_files=True)
    if require_running:
        _require_production_running(control)
        validate_readiness(control, state_dir=state_dir, verify_files=True)
    return production_environment(control, run_id=run_id)


def production_cell_execution(control: Mapping[str, Any]) -> dict[str, str]:
    """Return the only interpreter and release entry point allowed for cell arrays.

    Provenance fields in result metadata are not sufficient if Slurm first starts the
    worker from an ambient Conda environment or a mutable checkout.  This contract is
    deliberately path-addressed: the array invokes the frozen harness Python directly,
    and that process verifies both ``sys.prefix`` and the executing dispatcher source
    root before it is allowed to mutate a cell.
    """

    immutable = control["immutable"]
    harness_prefix = (
        Path(str(immutable["harness_environment_prefix"])).expanduser().resolve()
    )
    release_worktree = Path(str(immutable["release_worktree"])).expanduser().resolve()
    hf_home = Path(str(immutable["hf_home"])).expanduser().resolve()
    python_path = (harness_prefix / "bin" / "python").resolve()
    dispatcher_script = (release_worktree / "slurm" / "dispatch_sweeps.py").resolve()
    batch_template = (
        release_worktree / "slurm" / "run_dispatch_batch.sbatch.tmpl"
    ).resolve()
    if not hf_home.is_dir():
        raise ImmutablePinError(f"pinned Hugging Face home is missing: {hf_home}")
    for label, path in (
        ("pinned harness Python", python_path),
        ("pinned release dispatcher", dispatcher_script),
        ("pinned release cell template", batch_template),
    ):
        if not path.is_file():
            raise ImmutablePinError(f"{label} is missing: {path}")
    try:
        dispatcher_script.relative_to(release_worktree)
        batch_template.relative_to(release_worktree)
    except ValueError as exc:
        raise ImmutablePinError(
            "production cell runtime resolves outside the frozen release worktree"
        ) from exc
    try:
        python_path.relative_to(harness_prefix)
    except ValueError as exc:
        raise ImmutablePinError(
            "production cell interpreter resolves outside the frozen harness prefix"
        ) from exc
    unsafe_environment_paths = {
        label: str(path)
        for label, path in (
            ("release worktree", release_worktree),
            ("Hugging Face home", hf_home),
        )
        if any(
            character in str(path) for character in ('"', "\\", "$", "`", "\n", "\r")
        )
    }
    if unsafe_environment_paths:
        raise ImmutablePinError(
            "production environment paths are unsafe for exact shell export: "
            + ", ".join(sorted(unsafe_environment_paths))
        )
    return {
        "harness_prefix": str(harness_prefix),
        "python": str(python_path),
        "release_worktree": str(release_worktree),
        "hf_home": str(hf_home),
        "dispatcher_script": str(dispatcher_script),
        "batch_template": str(batch_template),
    }


def production_cell_execution_from_state(
    state_dir: Path, *, require_running: bool = True
) -> dict[str, str]:
    """Fail-closed public hook used when rendering a schema-5 cell array."""

    control = load_control(state_dir, verify_files=True)
    if require_running:
        _require_production_running(control)
        validate_readiness(control, state_dir=state_dir, verify_files=True)
    return production_cell_execution(control)


def admission_contract_from_state(state_dir: Path) -> dict[str, Any]:
    """Return the live, fail-closed admission contract for one dispatcher poll."""
    control = load_control(state_dir, verify_files=True)
    _require_production_running(control)
    validate_readiness(control, state_dir=state_dir, verify_files=True)
    return {
        **copy.deepcopy(control["admission"]),
        "rollout_generation": control["rollout_generation"],
        "immutable_sha256": control["immutable_sha256"],
    }


def validate_production_batch_sbatch(
    control: Mapping[str, Any], *, payload: str, task_count: int
) -> None:
    """Reject a rendered cell array that weakens any production resource invariant."""
    if task_count < 1 or task_count > int(control["admission"]["max_batch"]):
        raise ControlError(
            f"production array task count {task_count} exceeds 1..{control['admission']['max_batch']}"
        )
    required_directives = {
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=4G",
        "#SBATCH --time=12:00:00",
        "#SBATCH --signal=B:USR1@1200",
        "#SBATCH --no-requeue",
    }
    stripped_lines = [line.strip() for line in payload.splitlines()]
    lines = set(stripped_lines)
    missing = sorted(required_directives - lines)
    if missing:
        raise ControlError(
            "rendered production cell batch weakens required Slurm contract; missing "
            + ", ".join(missing)
        )
    manifest_hash_arguments = [
        line
        for line in stripped_lines
        if line.startswith("--batch-manifest-sha256 ")
    ]
    if len(manifest_hash_arguments) != 1 or re.fullmatch(
        r'--batch-manifest-sha256 "[0-9a-f]{64}" \\',
        manifest_hash_arguments[0],
    ) is None:
        raise ControlError(
            "rendered production cell batch lacks one exact manifest digest"
        )
    arrays = [line for line in lines if line.startswith("#SBATCH --array=")]
    if len(arrays) != 1:
        raise ControlError(
            "rendered production cell batch must contain exactly one array"
        )
    range_text = arrays[0].split("=", 1)[1].split("%", 1)[0]
    try:
        first_text, last_text = range_text.split("-", 1)
        first, last = int(first_text), int(last_text)
    except (ValueError, TypeError) as exc:
        raise ControlError(f"invalid production array range {range_text!r}") from exc
    if first != 0 or last + 1 != task_count:
        raise ControlError(
            f"production array range {range_text!r} does not encode {task_count} tasks"
        )

    execution = production_cell_execution(control)
    required_fragments = {
        "exact pinned cell entry point": (
            "exec "
            + shlex.quote(execution["python"])
            + " -u "
            + shlex.quote(execution["dispatcher_script"])
            + " run-task"
        ),
        "runtime release verification": (
            "--expected-release-root " + shlex.quote(execution["release_worktree"])
        ),
        "runtime harness verification": (
            "--expected-harness-prefix " + shlex.quote(execution["harness_prefix"])
        ),
        "ambient Python/Conda sanitization": (
            "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV"
        ),
        "bytecode isolation": "export PYTHONDONTWRITEBYTECODE=1",
        "user-site isolation": "export PYTHONNOUSERSITE=1",
        "safe import path": "export PYTHONSAFEPATH=1",
        "offline Hugging Face": "export HF_HUB_OFFLINE=1",
        "offline Transformers": "export TRANSFORMERS_OFFLINE=1",
        "offline datasets": "export HF_DATASETS_OFFLINE=1",
    }
    required_exact_exports = {
        "exact pinned Hugging Face home": f'export HF_HOME="{execution["hf_home"]}"',
        "exact pinned release worktree": (
            f'export ASYS_RELEASE_WORKTREE="{execution["release_worktree"]}"'
        ),
    }
    absent = [
        label
        for label, fragment in required_fragments.items()
        if fragment not in payload
    ]
    absent.extend(
        label for label, line in required_exact_exports.items() if line not in lines
    )
    for variable, expected_line in (
        ("HF_HOME", required_exact_exports["exact pinned Hugging Face home"]),
        (
            "ASYS_RELEASE_WORKTREE",
            required_exact_exports["exact pinned release worktree"],
        ),
    ):
        assignments = [
            line
            for line in stripped_lines
            if line.startswith((f"export {variable}=", f"{variable}="))
        ]
        if assignments != [expected_line]:
            absent.append(f"single exact {variable} assignment")
    if absent:
        raise ControlError(
            "rendered production cell batch lacks immutable runtime contract: "
            + ", ".join(sorted(absent))
        )
    forbidden_fragments = (
        "mamba activate",
        "conda activate",
        "$ASYS_HARNESS_ENV",
        "${ASYS_HARNESS_ENV",
        'source "' + str(REPO / "slurm" / "common.sh") + '"',
    )
    present = [fragment for fragment in forbidden_fragments if fragment in payload]
    present.extend(
        line.strip()
        for line in payload.splitlines()
        if line.strip().startswith(("source ", ". "))
    )
    if present:
        raise ControlError(
            "rendered production cell batch depends on ambient environment activation: "
            + ", ".join(present)
        )


def validate_controller_generation_sbatch(
    control: Mapping[str, Any],
    *,
    payload: str,
    state_dir: Path,
    role: str,
    generation: int,
    intent_token: str,
) -> None:
    """Prove a spooled controller starts only from the frozen runtime.

    Controller jobs are long-lived trust roots.  Executing the pinned Python binary is
    not sufficient if a mutable shell bootstrap first changes imports, ``HF_HOME``, or
    result routing.  This validator is intentionally byte-oriented and is run before
    every generation file is published.
    """

    _validate_role(role)
    immutable = control["immutable"]
    release_worktree = Path(str(immutable["release_worktree"])).resolve()
    python_path = (
        Path(str(immutable["harness_environment_prefix"])).resolve() / "bin" / "python"
    )
    control_script = release_worktree / "slurm" / "schema5_control.py"
    exact_lines = {
        "#SBATCH --no-requeue",
        "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV",
        "export PYTHONDONTWRITEBYTECODE=1",
        "export PYTHONNOUSERSITE=1",
        "export PYTHONSAFEPATH=1",
        f'export HF_HOME="{Path(str(immutable["hf_home"])).resolve()}"',
        f'export ASYS_RESULTS_ROOT="{Path(str(immutable["results_root"])).resolve()}"',
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        "export HF_DATASETS_OFFLINE=1",
        f'export ASYS_RELEASE_ID="{immutable["release_id"]}"',
        f'export ASYS_RELEASE_WORKTREE="{release_worktree}"',
        (
            'export ASYS_MODEL_CONTRACT_SHA256="'
            + str(immutable["model_contract_sha256"])
            + '"'
        ),
        (
            'export ASYS_FLEET_CONTRACT_SHA256="'
            + str(immutable["fleet_contract_sha256"])
            + '"'
        ),
        (
            'export ASYS_HARNESS_ENVIRONMENT_SHA256="'
            + str(immutable["harness_environment_sha256"])
            + '"'
        ),
        (
            'export ASYS_SERVING_ENVIRONMENT_SHA256="'
            + str(immutable["serving_environment_sha256"])
            + '"'
        ),
        f'export ASYS_ROLLOUT_GENERATION="{control["rollout_generation"]}"',
        f'export ASYS_IMMUTABLE_PINS_SHA256="{control["immutable_sha256"]}"',
        (
            f'exec "{python_path}" -u "{control_script}" '
            f'--state-dir "{state_dir.resolve()}" supervise \\'
        ),
        (
            f"--role {role} --generation {generation} "
            f'--intent-token "{intent_token}"'
        ),
    }
    ordered_stripped_lines = [line.strip() for line in payload.splitlines()]
    stripped_lines = set(ordered_stripped_lines)
    missing = sorted(exact_lines - stripped_lines)
    if missing:
        raise ControlError(
            "rendered controller generation lacks the immutable runtime contract: "
            + ", ".join(missing)
        )
    for variable, expected_line in (
        ("HF_HOME", f'export HF_HOME="{Path(str(immutable["hf_home"])).resolve()}"'),
        (
            "ASYS_RELEASE_WORKTREE",
            f'export ASYS_RELEASE_WORKTREE="{release_worktree}"',
        ),
    ):
        assignments = [
            line
            for line in ordered_stripped_lines
            if line.startswith((f"export {variable}=", f"{variable}="))
        ]
        if assignments != [expected_line]:
            raise ControlError(
                f"rendered controller generation must contain one exact {variable} assignment"
            )

    forbidden_lines = []
    forbidden_fragments = (
        "mamba activate",
        "conda activate",
        "micromamba activate",
        "source ",
        ". ",
        "PYTHONPATH=",
        "PYTHONHOME=",
        "VIRTUAL_ENV=",
        "CONDA_PREFIX=",
        "CONDA_DEFAULT_ENV=",
    )
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if any(fragment.lower() in lowered for fragment in forbidden_fragments):
            # The single exact unset command is the only permitted mention of these
            # ambient variables.
            if (
                line
                == "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV"
            ):
                continue
            forbidden_lines.append(line)
    if forbidden_lines:
        raise ControlError(
            "rendered controller generation depends on ambient environment activation: "
            + ", ".join(forbidden_lines)
        )

    exec_lines = [line for line in stripped_lines if line.startswith("exec ")]
    if exec_lines != [
        f'exec "{python_path}" -u "{control_script}" '
        f'--state-dir "{state_dir.resolve()}" supervise \\'
    ]:
        raise ControlError(
            "rendered controller generation has a non-canonical entry point"
        )


def render_generation_sbatch(
    state_dir: Path,
    control: Mapping[str, Any],
    *,
    role: str,
    generation: int,
    intent_token: str,
) -> Path:
    """Render one immutable generation-addressed controller sbatch."""
    _validate_role(role)
    token = job_token(role, generation, intent_token)
    short = ROLE_SHORT[role]
    template_name = (
        "schema5_dispatcher.sbatch.tmpl"
        if role == "dispatcher"
        else "schema5_fleet_supervisor.sbatch.tmpl"
    )
    release_worktree = Path(control["immutable"]["release_worktree"]).resolve()
    template = release_worktree / "slurm" / template_name
    if not template.is_file():
        raise ImmutablePinError(
            f"controller template missing from frozen release: {template}"
        )
    target = state_dir / "sbatch" / f"{short}.g{generation:06d}.{intent_token}.sbatch"
    resource = _controller_resource(control, role)
    python_path = (
        Path(control["immutable"]["harness_environment_prefix"]) / "bin" / "python"
    )
    provenance = production_environment(control)
    replacements = {
        "JOB_NAME": f"asys-s5-{short}-g{generation:06d}-{intent_token[:8]}",
        "JOB_TOKEN": token,
        "PARTITION": resource["partition"],
        "MEMORY": resource["memory"],
        "TIME_LIMIT": resource["time_limit"],
        "LOG_PATH": str(state_dir / "logs" / f"{short}.g{generation:06d}.%j.out"),
        "ALERT_EMAIL": str(control["alert_email"]),
        "PYTHON": str(python_path),
        "CONTROL_SCRIPT": str(release_worktree / "slurm" / "schema5_control.py"),
        "STATE_DIR": str(state_dir.resolve()),
        "GENERATION": str(generation),
        "INTENT_TOKEN": intent_token,
        "RELEASE_ID": provenance["ASYS_RELEASE_ID"],
        "RELEASE_WORKTREE": str(release_worktree),
        "MODEL_CONTRACT_SHA256": provenance["ASYS_MODEL_CONTRACT_SHA256"],
        "FLEET_CONTRACT_SHA256": provenance["ASYS_FLEET_CONTRACT_SHA256"],
        "HARNESS_ENVIRONMENT_SHA256": provenance["ASYS_HARNESS_ENVIRONMENT_SHA256"],
        "SERVING_ENVIRONMENT_SHA256": provenance["ASYS_SERVING_ENVIRONMENT_SHA256"],
        "ROLLOUT_GENERATION": provenance["ASYS_ROLLOUT_GENERATION"],
        "IMMUTABLE_PINS_SHA256": provenance["ASYS_IMMUTABLE_PINS_SHA256"],
        "HF_HOME": str(Path(control["immutable"]["hf_home"]).resolve()),
        "RESULTS_ROOT": str(Path(control["immutable"]["results_root"]).resolve()),
    }
    unsafe = {
        name: value
        for name, value in replacements.items()
        if any(character in value for character in ('"', "\\", "$", "`", "\n", "\r"))
    }
    if unsafe:
        raise ImmutablePinError(
            "controller template values are not safe for exact double-quoted rendering: "
            + ", ".join(sorted(unsafe))
        )
    payload = template.read_text(encoding="utf-8")
    for name, value in replacements.items():
        if "{" + name + "}" not in payload:
            raise ControlError(f"controller template omitted placeholder {name}")
        payload = payload.replace("{" + name + "}", value)
    if "{" in payload or "}" in payload:
        raise ControlError(
            f"unresolved placeholder in rendered controller sbatch {target}"
        )
    validate_controller_generation_sbatch(
        control,
        payload=payload,
        state_dir=state_dir,
        role=role,
        generation=generation,
        intent_token=intent_token,
    )
    if target.exists():
        if target.read_text(encoding="utf-8") != payload:
            raise ImmutablePinError(
                f"generation sbatch already exists with different bytes: {target}"
            )
        return target
    io.atomic_write_text(target, payload)
    # Make the immutable intent artifact read-only.  A retry reuses identical bytes.
    target.chmod(0o444)
    return target


def _submit_argv(
    path: Path, *, token: str, dependency_job_id: str | None, hold: bool = False
) -> list[str]:
    argv = ["sbatch", "--parsable", f"--comment={token}"]
    if hold:
        argv.append("--hold")
    if dependency_job_id:
        if not dependency_job_id.isdigit():
            raise ControlError(f"unsafe dependency job ID {dependency_job_id!r}")
        argv.append(f"--dependency=afterany:{dependency_job_id}")
    argv.append(str(path))
    return argv


def _find_intent_jobs(snapshot: SchedulerSnapshot, token: str) -> list[SchedulerJob]:
    return [job for job in snapshot.jobs if job.token == token]


def submit_controller_intent(
    state_dir: Path,
    *,
    role: str,
    target: str,
    dependency_job_id: str | None,
    scheduler: SchedulerSnapshot,
    submit_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    allow_resuming: bool = False,
    hold: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """Recover or submit one role intent, committing scheduler identity exactly once."""
    _validate_role(role)
    if target not in {"active", "successor"}:
        raise ControlError("controller intent target must be active or successor")
    timestamp = time.time() if now is None else float(now)
    runner = submit_runner or _run_subprocess
    if not scheduler.squeue_ok or not scheduler.sacct_ok:
        raise SchedulerAmbiguity(
            "controller submission requires complete squeue+sacct truth"
        )

    # Phase one: under the control lock, adopt a crash-window job or persist the exact
    # submitting intent before crossing the external sbatch boundary.
    with control_lock(state_dir):
        control = load_control(state_dir)
        allowed_states = {"running", "resuming"} if allow_resuming else {"running"}
        if control["desired_state"] not in allowed_states:
            raise ControlError("desired_state=paused disables controller resurrection")
        # A fresh fleet probe is a launch gate, not a renewable lease on the
        # controller chain. ``resume_control`` proves the <=10-minute fleet view
        # before entering ``resuming`` and records the resulting transaction. Once
        # production is ``running``, successors and stale-heartbeat takeovers keep
        # verifying every immutable/readiness artifact without aging out that
        # historical launch proof; live fleet health is owned by the fleet supervisor
        # and continuous monitors. Requiring freshness here would let a nominal
        # 12-hour controller submit its first successor but prevent that successor
        # from renewing the chain after the readiness artifact became old.
        validate_readiness(
            control, state_dir=state_dir, verify_files=True, now=None
        )
        role_state = control["controllers"][role]
        current_target = role_state.get(target)
        if isinstance(current_target, dict) and current_target.get("job_id"):
            exact_matches = [
                job
                for job in scheduler.jobs
                if job.job_id == str(current_target["job_id"])
                and job.token == current_target.get("job_token")
            ]
            if len(exact_matches) > 1:
                raise SchedulerAmbiguity(
                    f"controller job {current_target['job_id']} for {role} appears more than once"
                )
            same_id_wrong_token = [
                job
                for job in scheduler.jobs
                if job.job_id == str(current_target["job_id"])
                and job.token != current_target.get("job_token")
            ]
            if same_id_wrong_token:
                raise SchedulerAmbiguity(
                    f"controller job {current_target['job_id']} for {role} has a token mismatch"
                )
            if exact_matches and exact_matches[0].active:
                return copy.deepcopy(current_target)
            if exact_matches:
                # Accounting proves the previous exact allocation was accepted and is
                # terminal.  It is safe to create a new generation immediately.
                role_state[target] = None
                if role_state.get("submission_intent") is current_target or (
                    isinstance(role_state.get("submission_intent"), dict)
                    and role_state["submission_intent"].get("intent_token")
                    == current_target.get("intent_token")
                ):
                    role_state["submission_intent"] = None
                current_target = None
            visible_since = float(
                (current_target or {}).get(
                    "submitted_timestamp",
                    (current_target or {}).get("created_timestamp", 0.0),
                )
            )
            if (
                current_target is not None
                and timestamp - visible_since < SUBMISSION_VISIBILITY_GRACE_SECONDS
            ):
                raise SchedulerVisibilityPending(
                    f"controller job {current_target['job_id']} for {role} is inside "
                    "scheduler visibility grace"
                )
        intent = role_state.get("submission_intent")
        if isinstance(intent, dict) and intent.get("state") == "submitting":
            if bool(intent.get("hold", False)) != bool(hold):
                raise SchedulerAmbiguity(
                    "controller retry changed immutable hold semantics"
                )
            token = str(intent["job_token"])
            matches = _find_intent_jobs(scheduler, token)
            if len(matches) > 1:
                raise SchedulerAmbiguity(
                    f"controller intent {token} maps to multiple scheduler jobs"
                )
            active = [job for job in matches if job.active]
            if len(active) == 1:
                intent["state"] = "submitted"
                intent["job_id"] = active[0].job_id
                intent["adopted_from_scheduler"] = True
                role_state[str(intent["target"])] = copy.deepcopy(intent)
                append_transition(
                    state_dir,
                    control,
                    event="controller_submission_adopted",
                    details={"role": role, "job_id": active[0].job_id, "token": token},
                    now=timestamp,
                )
                _save_control(state_dir, control, now=timestamp)
                return copy.deepcopy(intent)
            if matches:
                terminal = matches[0]
                intent["state"] = "terminal"
                intent["job_id"] = terminal.job_id
                intent["terminal_scheduler_state"] = normalize_scheduler_state(
                    terminal.state
                )
                intent["adopted_from_scheduler"] = True
                role_state[str(intent["target"])] = copy.deepcopy(intent)
                role_state["submission_intent"] = None
                role_state["next_generation"] = max(
                    int(role_state["next_generation"]), int(intent["generation"]) + 1
                )
                append_transition(
                    state_dir,
                    control,
                    event="controller_terminal_submission_adopted",
                    details={
                        "role": role,
                        "job_id": terminal.job_id,
                        "token": token,
                        "state": normalize_scheduler_state(terminal.state),
                    },
                    now=timestamp,
                )
                intent = None
            # With complete squeue+sacct truth, a prior retryable non-acceptance may be
            # retried using the *same* generation and token.  An incomplete snapshot is
            # ambiguous and therefore fail-closed.
            if intent is not None:
                age = timestamp - float(intent.get("created_timestamp", timestamp))
                if (
                    not intent.get("last_submission_rejected", False)
                    and age < SUBMISSION_VISIBILITY_GRACE_SECONDS
                ):
                    raise SchedulerVisibilityPending(
                        f"controller intent {token} is inside scheduler visibility grace "
                        f"({max(0.0, age):.1f}s/{SUBMISSION_VISIBILITY_GRACE_SECONDS:.1f}s)"
                    )
        if not (isinstance(intent, dict) and intent.get("state") == "submitting"):
            generation = int(role_state["next_generation"])
            intent_token = uuid.uuid4().hex
            token = job_token(role, generation, intent_token)
            sbatch_path = render_generation_sbatch(
                state_dir,
                control,
                role=role,
                generation=generation,
                intent_token=intent_token,
            )
            intent = {
                "state": "submitting",
                "role": role,
                "target": target,
                "generation": generation,
                "intent_token": intent_token,
                "job_token": token,
                "sbatch_path": str(sbatch_path),
                "dependency_job_id": dependency_job_id,
                "controller_primitives": shared_controller_primitive_contract(),
                "hold": bool(hold),
                "created_at": utc_timestamp(timestamp),
                "created_timestamp": timestamp,
                "attempts": 0,
            }
            role_state["submission_intent"] = intent
            append_transition(
                state_dir,
                control,
                event="controller_submission_intent",
                details={
                    "role": role,
                    "target": target,
                    "generation": generation,
                    "intent_token": intent_token,
                    "dependency_job_id": dependency_job_id,
                },
                now=timestamp,
            )
            _save_control(state_dir, control, now=timestamp)
        intent_copy = copy.deepcopy(intent)

    sbatch_path = Path(intent_copy["sbatch_path"])
    argv = _submit_argv(
        sbatch_path,
        token=str(intent_copy["job_token"]),
        dependency_job_id=intent_copy.get("dependency_job_id"),
        hold=bool(intent_copy.get("hold", False)),
    )
    proc = runner(argv)
    completed_at = time.time() if now is None else timestamp

    # Phase two: commit acceptance.  A nonzero return is known rejection and remains a
    # retryable submitting intent.  A killed process never reaches this phase; the next
    # caller searches both scheduler views by token before doing anything else.
    with control_lock(state_dir):
        control = load_control(state_dir)
        role_state = control["controllers"][role]
        current_intent = role_state.get("submission_intent")
        if not isinstance(current_intent, dict) or (
            current_intent.get("intent_token") != intent_copy["intent_token"]
        ):
            raise SchedulerAmbiguity(
                "controller submission intent changed during sbatch"
            )
        current_intent["attempts"] = int(current_intent.get("attempts", 0)) + 1
        current_intent["last_attempt_at"] = utc_timestamp(completed_at)
        if proc.returncode != 0:
            current_intent["last_error"] = proc.stderr.strip()[:1000]
            current_intent["last_submission_rejected"] = True
            append_transition(
                state_dir,
                control,
                event="controller_submission_retryable",
                details={
                    "role": role,
                    "intent_token": current_intent["intent_token"],
                    "returncode": proc.returncode,
                    "error": current_intent["last_error"],
                },
                now=completed_at,
            )
            _save_control(state_dir, control, now=completed_at)
            raise ControlError(
                f"sbatch rejected controller intent for {role}: "
                f"{proc.stderr.strip()[:500]}"
            )
        job_id = proc.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            current_intent["last_error"] = f"invalid sbatch job id {job_id!r}"
            _save_control(state_dir, control, now=completed_at)
            raise SchedulerAmbiguity(
                f"sbatch returned invalid controller job id {job_id!r}"
            )
        current_intent["state"] = "submitted"
        current_intent["last_submission_rejected"] = False
        current_intent["job_id"] = job_id
        current_intent["submitted_at"] = utc_timestamp(completed_at)
        current_intent["submitted_timestamp"] = completed_at
        role_state[target] = copy.deepcopy(current_intent)
        role_state["next_generation"] = max(
            int(role_state["next_generation"]), int(current_intent["generation"]) + 1
        )
        append_transition(
            state_dir,
            control,
            event="controller_submitted",
            details={
                "role": role,
                "target": target,
                "generation": current_intent["generation"],
                "intent_token": current_intent["intent_token"],
                "job_id": job_id,
            },
            now=completed_at,
        )
        _save_control(state_dir, control, now=completed_at)
        return copy.deepcopy(current_intent)


def _drill_state_path(state_dir: Path) -> Path:
    return state_dir / DRILL_STATE_FILENAME


def _drill_reconciliation_path(state_dir: Path, drill_id: str) -> Path:
    return state_dir / "drills" / drill_id / DRILL_RECONCILIATION_FILENAME


@contextmanager
def drill_lock(state_dir: Path) -> Iterator[None]:
    with _file_lock(state_dir / "locks" / "controller-drill.lock"):
        yield


@contextmanager
def drill_submission_boundary_lock(state_dir: Path) -> Iterator[None]:
    """Serialize the complete external ``sbatch`` transaction with drill stopping.

    The lock deliberately spans both the durable intent write and the external scheduler
    call.  ``finish_controller_drill`` takes the same lock before publishing ``stopping``.
    Consequently, a successful finish can neither overtake an in-flight submission nor
    permit a new one to start after the stopping transition.
    """

    with _file_lock(state_dir / "locks" / "controller-drill-submission-boundary.lock"):
        yield


def _append_drill_event(
    state_dir: Path,
    state: MutableMapping[str, Any],
    *,
    event: str,
    details: Mapping[str, Any] | None = None,
    now: float,
) -> None:
    durable = _read_jsonl_locked(state_dir / DRILL_JOURNAL_FILENAME, missing_ok=True)
    memory = list(state.get("transition_history", []))
    if durable[: len(memory)] != memory:
        raise ControlError("controller-drill journal diverged from its state")
    if len(durable) > len(memory):
        state["transition_history"] = copy.deepcopy(durable)
        validate_transition_history(state)
    record = _history_record(state, event=event, details=details, now=now)
    state.setdefault("transition_history", []).append(record)
    _append_jsonl(state_dir / DRILL_JOURNAL_FILENAME, record)


def _validate_drill_state(state: Mapping[str, Any]) -> None:
    if (
        state.get("schema_version") != 1
        or state.get("protocol") != "schema5-controller-drill-v1"
    ):
        raise ControlError("controller-drill state has an unsupported schema")
    if not isinstance(state.get("drill_id"), str) or not state["drill_id"]:
        raise ControlError("controller-drill state has no drill ID")
    if state.get("shared_fencing_primitive") != SHARED_CONTROLLER_FENCING_PRIMITIVE:
        raise ControlError(
            "controller-drill state has the wrong shared fencing primitive"
        )
    if (
        state.get("shared_controller_primitives")
        != shared_controller_primitive_contract()
    ):
        raise ControlError(
            "controller-drill state has the wrong shared primitive contract"
        )
    if state.get("phase") not in {"running", "stopping", "completed"}:
        raise ControlError("controller-drill phase is invalid")
    if state.get("phase") == "completed":
        if (
            not isinstance(state.get("final_reconciliation_path"), str)
            or _SHA256_RE.fullmatch(str(state.get("final_reconciliation_sha256", "")))
            is None
        ):
            raise ControlError(
                "completed controller-drill reconciliation proof is missing"
            )
    elif (
        state.get("final_reconciliation_path") is not None
        or state.get("final_reconciliation_sha256") is not None
    ):
        raise ControlError(
            "unfinished controller drill cannot carry final reconciliation"
        )
    if not isinstance(state.get("immutable_sha256"), str):
        raise ControlError("controller-drill immutable identity is missing")
    roles = state.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(ROLE_NAMES):
        raise ControlError("controller-drill role state is incomplete")
    for role, row in roles.items():
        if not isinstance(row, dict) or set(row) != {
            "next_generation",
            "active",
            "successor",
            "submission_intent",
            "heartbeat",
            "last_exit",
            "kill",
            "recovery",
        }:
            raise ControlError(f"controller-drill role state is malformed: {role}")
        if not isinstance(row["next_generation"], int) or row["next_generation"] < 1:
            raise ControlError(f"controller-drill next generation is invalid: {role}")
    if not isinstance(state.get("baseline"), dict):
        raise ControlError("controller-drill baseline is missing")
    validate_transition_history(state)


def load_drill_state(state_dir: Path) -> dict[str, Any]:
    path = _drill_state_path(state_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ControlError(f"controller drill has not been started: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlError(f"cannot read controller-drill state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControlError("controller-drill state must be a JSON object")
    _validate_drill_state(value)
    durable = _read_jsonl_locked(state_dir / DRILL_JOURNAL_FILENAME)
    if durable[: len(value["transition_history"])] != value["transition_history"]:
        raise ControlError("controller-drill state is not a prefix of its journal")
    return value


def _save_drill_state(
    state_dir: Path, state: MutableMapping[str, Any], *, now: float
) -> None:
    state["updated_at"] = utc_timestamp(now)
    state["updated_timestamp"] = now
    _validate_drill_state(state)
    _atomic_write_json(_drill_state_path(state_dir), state)


def _optional_file_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "sha256": None}
    if path.is_symlink() or not path.is_file():
        raise ControlError(f"drill baseline target is not a regular file: {path}")
    return {"exists": True, "sha256": sha256_file(path)}


def controller_drill_baseline(
    state_dir: Path, control: Mapping[str, Any]
) -> dict[str, Any]:
    """Hash every production-owned state surface a no-admission drill may not alter."""

    run_trees = {
        str(run["run_id"]): sha256_tree(Path(str(run["run_root"])).resolve())
        for run in control["immutable"]["runs"]
    }
    projection = {
        "rollout_generation": control["rollout_generation"],
        "admission": control["admission"],
        "throughput_epochs": control["throughput_epochs"],
        "resume_intent": control["resume_intent"],
        "drain_requested": control["drain_requested"],
        "controllers": control["controllers"],
    }
    return {
        "immutable_sha256": control["immutable_sha256"],
        "production_state_sha256": sha256_value(projection),
        "dispatcher_ledger": _optional_file_identity(state_dir / "ledger.json"),
        "run_tree_sha256": run_trees,
    }


def _assert_drill_control_paused(
    state_dir: Path, *, expected_immutable_sha256: str | None = None
) -> dict[str, Any]:
    control = load_control(state_dir, verify_files=True)
    if control["desired_state"] != "paused" or control.get("drain_requested"):
        raise ControlError(
            "controller kill drill requires desired_state=paused without a drain intent"
        )
    if (
        expected_immutable_sha256
        and control["immutable_sha256"] != expected_immutable_sha256
    ):
        raise ImmutablePinError("controller-drill immutable pins differ from control")
    if any(
        role_state.get(field) is not None
        for role_state in control["controllers"].values()
        for field in ("active", "successor", "submission_intent")
    ):
        raise ControlError(
            "controller kill drill requires empty production controller state"
        )
    return control


def _render_drill_sbatch(
    state_dir: Path,
    control: Mapping[str, Any],
    *,
    drill_id: str,
    role: str,
    generation: int,
    intent_token: str,
) -> Path:
    token = drill_job_token(drill_id, role, generation, intent_token)
    short = ROLE_SHORT[role]
    target_dir = state_dir / "drills" / drill_id / "sbatch"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{short}.g{generation:06d}.{intent_token}.sbatch"
    python = Path(control["immutable"]["harness_environment_prefix"]) / "bin" / "python"
    script = (
        Path(control["immutable"]["release_worktree"]) / "slurm" / "schema5_control.py"
    )
    log_path = (
        state_dir / "drills" / drill_id / "logs" / f"{short}.g{generation:06d}.%j.out"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(python),
        "-I",
        "-u",
        str(script),
        "--state-dir",
        str(state_dir.resolve()),
        "supervise-drill",
        "--drill-id",
        drill_id,
        "--role",
        role,
        "--generation",
        str(generation),
        "--intent-token",
        intent_token,
    ]
    payload = "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name=asys-s5-drill-{short}-{generation:06d}",
            f"#SBATCH --comment={token}",
            "#SBATCH --partition=mit_preemptable,mit_normal,mit_normal_gpu",
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --mem=4G",
            "#SBATCH --time=01:00:00",
            "#SBATCH --no-requeue",
            f"#SBATCH --output={log_path}",
            "set -euo pipefail",
            "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV",
            "export PYTHONDONTWRITEBYTECODE=1",
            "export PYTHONNOUSERSITE=1",
            "export PYTHONSAFEPATH=1",
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "export HF_DATASETS_OFFLINE=1",
            "exec " + " ".join(shlex.quote(item) for item in argv),
            "",
        ]
    )
    if target.exists():
        if target.read_text(encoding="utf-8") != payload:
            raise ImmutablePinError(
                f"drill sbatch already exists with different bytes: {target}"
            )
        return target
    io.atomic_write_text(target, payload)
    target.chmod(0o444)
    return target


def _drill_token_jobs(snapshot: SchedulerSnapshot, token: str) -> list[SchedulerJob]:
    return [job for job in snapshot.jobs if job.comment == token]


def _command_binds_exact_sbatch(command: str, sbatch_path: str) -> bool:
    """Return whether scheduler command provenance contains the exact sbatch argument."""

    try:
        return sbatch_path in shlex.split(command)
    except ValueError:
        return False


def _dependency_binds_exact_afterany(dependency: str, parent_job_id: str) -> bool:
    """Accept only the one Slurm after-any edge frozen for a successor."""

    if not parent_job_id.isdigit():
        return False
    return (
        re.fullmatch(
            rf"afterany:{re.escape(parent_job_id)}(?:\([^()]*\))?",
            dependency.strip(),
        )
        is not None
    )


def _all_live_drill_jobs(snapshot: SchedulerSnapshot) -> list[SchedulerJob]:
    """Return every live row claiming the drill namespace, parsed or malformed.

    Namespace emptiness is a scheduler safety property, not a parser convenience.  A
    malformed comment must therefore fence start/resume/finish just as strongly as a
    well-formed but unmapped drill token.
    """

    return [
        job
        for job in snapshot.jobs
        if job.active and job.comment.startswith(DRILL_TOKEN_PREFIX)
    ]


def _unique_exact_drill_job(
    snapshot: SchedulerSnapshot,
    *,
    job_id: str,
    job_token: str,
    sbatch_path: str,
    require_active: bool | None,
    expected_dependency_job_id: str | None = None,
) -> SchedulerJob:
    """Resolve one token and bind its ID, sbatch, and optional parent exactly."""

    token_matches = _drill_token_jobs(snapshot, job_token)
    if len(token_matches) != 1:
        raise SchedulerAmbiguity(
            f"drill token must map to exactly one scheduler job; found "
            f"{[job.job_id for job in token_matches]}"
        )
    job = token_matches[0]
    if job.job_id != str(job_id):
        raise SchedulerAmbiguity(
            f"drill token maps to job {job.job_id}, not recorded exact job {job_id}"
        )
    if not _command_binds_exact_sbatch(job.command, sbatch_path):
        raise SchedulerAmbiguity(
            f"drill job {job_id} command does not bind recorded exact sbatch {sbatch_path}"
        )
    if expected_dependency_job_id is not None and not _dependency_binds_exact_afterany(
        job.dependency, expected_dependency_job_id
    ):
        raise SchedulerAmbiguity(
            f"drill job {job_id} dependency {job.dependency!r} does not bind exact "
            f"afterany parent {expected_dependency_job_id}"
        )
    if require_active is True and not job.active:
        raise SchedulerAmbiguity(f"drill job {job_id} is not active")
    if require_active is False and job.active:
        raise SchedulerAmbiguity(f"drill job {job_id} is still active")
    return job


def _reconcile_stopping_drill_intents(
    state_dir: Path,
    *,
    snapshot: SchedulerSnapshot,
    now: float,
) -> None:
    """Adopt or close every pre-stop submission intent under complete scheduler truth."""

    if not snapshot.squeue_ok or not snapshot.sacct_ok:
        raise SchedulerAmbiguity(
            "stopping drill intent reconciliation requires full truth"
        )
    with drill_lock(state_dir):
        state = load_drill_state(state_dir)
        if state["phase"] not in {"stopping", "completed"}:
            raise ControlError(
                "stopping-intent reconciliation requires a stopping drill"
            )
        changed = False
        for role, row in state["roles"].items():
            intent = row.get("submission_intent")
            if not isinstance(intent, dict):
                continue
            token = str(intent.get("job_token", ""))
            matches = _drill_token_jobs(snapshot, token)
            if len(matches) > 1:
                raise SchedulerAmbiguity(
                    f"stopping {role} intent maps to duplicate jobs: "
                    f"{[job.job_id for job in matches]}"
                )
            target = str(intent.get("target", ""))
            if target not in {"active", "successor"}:
                raise ControlError(
                    f"stopping {role} intent has invalid target {target!r}"
                )
            if matches:
                job = matches[0]
                if not _command_binds_exact_sbatch(
                    job.command, str(intent.get("sbatch_path", ""))
                ):
                    raise SchedulerAmbiguity(
                        f"stopping {role} intent has mismatched scheduler command"
                    )
                expected_parent = intent.get("dependency_job_id")
                if expected_parent is not None and not _dependency_binds_exact_afterany(
                    job.dependency, str(expected_parent)
                ):
                    raise SchedulerAmbiguity(
                        f"stopping {role} intent has mismatched scheduler dependency"
                    )
                existing = row.get(target)
                if isinstance(existing, dict) and existing.get(
                    "intent_token"
                ) != intent.get("intent_token"):
                    raise SchedulerAmbiguity(
                        f"stopping {role} intent would overwrite another {target}"
                    )
                intent["state"] = "submitted"
                intent["job_id"] = job.job_id
                intent["adopted_from_scheduler"] = True
                row[target] = copy.deepcopy(intent)
                event = "stopping_submission_adopted"
                details = {"role": role, "target": target, "job_id": job.job_id}
            else:
                age = now - float(intent.get("created_timestamp", now))
                if (
                    not intent.get("last_submission_rejected", False)
                    and age < SUBMISSION_VISIBILITY_GRACE_SECONDS
                ):
                    raise SchedulerVisibilityPending(
                        f"stopping {role} intent remains inside scheduler visibility grace "
                        f"({max(0.0, age):.1f}s/{SUBMISSION_VISIBILITY_GRACE_SECONDS:.1f}s)"
                    )
                event = "stopping_submission_closed_absent"
                details = {"role": role, "target": target, "age_seconds": age}
            row["submission_intent"] = None
            row["next_generation"] = max(
                int(row["next_generation"]), int(intent.get("generation", 0)) + 1
            )
            _append_drill_event(
                state_dir,
                state,
                event=event,
                details=details,
                now=now,
            )
            changed = True
        if changed:
            _save_drill_state(state_dir, state, now=now)


def submit_drill_intent(
    state_dir: Path,
    *,
    role: str,
    target: str,
    dependency_job_id: str | None,
    scheduler: SchedulerSnapshot,
    submit_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Transactionally submit or adopt one no-op drill controller intent."""

    # This boundary is intentionally outside ``drill_lock`` and spans ``sbatch``.  The
    # finish path acquires it before changing the phase to ``stopping``, which rules out
    # a late accepted successor appearing after the completion marker.
    with drill_submission_boundary_lock(state_dir):
        return _submit_drill_intent_inside_boundary(
            state_dir,
            role=role,
            target=target,
            dependency_job_id=dependency_job_id,
            scheduler=scheduler,
            submit_runner=submit_runner,
            now=now,
        )


def _submit_drill_intent_inside_boundary(
    state_dir: Path,
    *,
    role: str,
    target: str,
    dependency_job_id: str | None,
    scheduler: SchedulerSnapshot,
    submit_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Implementation of :func:`submit_drill_intent` with its boundary held."""

    _validate_role(role)
    if target not in {"active", "successor"}:
        raise ControlError("drill intent target must be active or successor")
    if not scheduler.squeue_ok or not scheduler.sacct_ok:
        raise SchedulerAmbiguity(
            "drill submission requires complete squeue+sacct truth"
        )
    timestamp = time.time() if now is None else float(now)
    runner = submit_runner or _run_subprocess
    with drill_lock(state_dir):
        state = load_drill_state(state_dir)
        _assert_drill_control_paused(
            state_dir, expected_immutable_sha256=state["immutable_sha256"]
        )
        if state["phase"] != "running":
            raise ControlError("controller-drill submission is disabled while stopping")
        row = state["roles"][role]
        current = row.get(target)
        if isinstance(current, dict) and current.get("job_id"):
            matches = [
                job
                for job in scheduler.jobs
                if job.job_id == str(current["job_id"])
                and job.comment == current.get("job_token")
            ]
            if len(matches) > 1:
                raise SchedulerAmbiguity("drill job ID appears more than once")
            wrong_token = [
                job
                for job in scheduler.jobs
                if job.job_id == str(current["job_id"])
                and job.comment != current.get("job_token")
            ]
            if wrong_token:
                raise SchedulerAmbiguity("drill job ID has a scheduler token mismatch")
            if matches and matches[0].active:
                _unique_exact_drill_job(
                    scheduler,
                    job_id=str(current["job_id"]),
                    job_token=str(current["job_token"]),
                    sbatch_path=str(current["sbatch_path"]),
                    require_active=True,
                    expected_dependency_job_id=(
                        str(current["dependency_job_id"])
                        if current.get("dependency_job_id") is not None
                        else None
                    ),
                )
                return copy.deepcopy(current)
            if matches:
                row[target] = None
            else:
                age = timestamp - float(
                    current.get(
                        "submitted_timestamp",
                        current.get("created_timestamp", timestamp),
                    )
                )
                if age < SUBMISSION_VISIBILITY_GRACE_SECONDS:
                    raise SchedulerVisibilityPending(
                        "recorded drill job is inside scheduler visibility grace"
                    )
                row[target] = None
        intent = row.get("submission_intent")
        if isinstance(intent, dict):
            matches = _drill_token_jobs(scheduler, str(intent["job_token"]))
            if len(matches) > 1:
                raise SchedulerAmbiguity("drill intent maps to multiple scheduler jobs")
            if len(matches) == 1 and matches[0].active:
                _unique_exact_drill_job(
                    scheduler,
                    job_id=matches[0].job_id,
                    job_token=str(intent["job_token"]),
                    sbatch_path=str(intent["sbatch_path"]),
                    require_active=True,
                    expected_dependency_job_id=(
                        str(intent["dependency_job_id"])
                        if intent.get("dependency_job_id") is not None
                        else None
                    ),
                )
                intent["state"] = "submitted"
                intent["job_id"] = matches[0].job_id
                intent["adopted_from_scheduler"] = True
                row[str(intent["target"])] = copy.deepcopy(intent)
                row["submission_intent"] = None
                row["next_generation"] = max(
                    int(row["next_generation"]), int(intent["generation"]) + 1
                )
                _append_drill_event(
                    state_dir,
                    state,
                    event="submission_adopted",
                    details={"role": role, "job_id": matches[0].job_id},
                    now=timestamp,
                )
                _save_drill_state(state_dir, state, now=timestamp)
                return copy.deepcopy(intent)
            if matches:
                row["submission_intent"] = None
                row["next_generation"] = max(
                    int(row["next_generation"]), int(intent["generation"]) + 1
                )
                intent = None
            elif (
                not intent.get("last_submission_rejected", False)
                and timestamp - float(intent["created_timestamp"])
                < SUBMISSION_VISIBILITY_GRACE_SECONDS
            ):
                raise SchedulerVisibilityPending(
                    "drill submission is inside visibility grace"
                )
        if not isinstance(intent, dict):
            generation = int(row["next_generation"])
            intent_token = uuid.uuid4().hex
            token = drill_job_token(state["drill_id"], role, generation, intent_token)
            sbatch_path = _render_drill_sbatch(
                state_dir,
                load_control(state_dir),
                drill_id=state["drill_id"],
                role=role,
                generation=generation,
                intent_token=intent_token,
            )
            intent = {
                "state": "submitting",
                "role": role,
                "target": target,
                "generation": generation,
                "intent_token": intent_token,
                "job_token": token,
                "sbatch_path": str(sbatch_path),
                "dependency_job_id": dependency_job_id,
                "controller_primitives": shared_controller_primitive_contract(),
                "created_at": utc_timestamp(timestamp),
                "created_timestamp": timestamp,
                "attempts": 0,
            }
            row["submission_intent"] = intent
            _append_drill_event(
                state_dir,
                state,
                event="submission_intent",
                details={"role": role, "target": target, "generation": generation},
                now=timestamp,
            )
            _save_drill_state(state_dir, state, now=timestamp)
        intent_copy = copy.deepcopy(intent)
    proc = runner(
        _submit_argv(
            Path(intent_copy["sbatch_path"]),
            token=str(intent_copy["job_token"]),
            dependency_job_id=intent_copy.get("dependency_job_id"),
        )
    )
    completed = time.time() if now is None else timestamp
    with drill_lock(state_dir):
        state = load_drill_state(state_dir)
        _assert_drill_control_paused(
            state_dir, expected_immutable_sha256=state["immutable_sha256"]
        )
        row = state["roles"][role]
        current = row.get("submission_intent")
        if (
            not isinstance(current, dict)
            or current.get("intent_token") != intent_copy["intent_token"]
        ):
            raise SchedulerAmbiguity("drill submission intent changed during sbatch")
        current["attempts"] = int(current.get("attempts", 0)) + 1
        if proc.returncode != 0:
            current["last_submission_rejected"] = True
            current["last_error"] = proc.stderr.strip()[:1000]
            _save_drill_state(state_dir, state, now=completed)
            raise ControlError(
                f"sbatch rejected drill intent: {current['last_error'][:500]}"
            )
        job_id = proc.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise SchedulerAmbiguity(f"sbatch returned invalid drill job ID {job_id!r}")
        current["state"] = "submitted"
        current["job_id"] = job_id
        current["submitted_at"] = utc_timestamp(completed)
        current["submitted_timestamp"] = completed
        row[target] = copy.deepcopy(current)
        row["submission_intent"] = None
        row["next_generation"] = max(
            int(row["next_generation"]), int(current["generation"]) + 1
        )
        late_for_cleanup = state["phase"] != "running"
        _append_drill_event(
            state_dir,
            state,
            event=(
                "late_submission_committed_for_cleanup"
                if late_for_cleanup
                else "submitted"
            ),
            details={"role": role, "target": target, "job_id": job_id},
            now=completed,
        )
        _save_drill_state(state_dir, state, now=completed)
        result = copy.deepcopy(current)
    if late_for_cleanup:
        raise ControlError(
            f"drill submission {job_id} was accepted after stopping and is committed for cleanup"
        )
    return result


def start_controller_drill(
    state_dir: Path,
    *,
    scheduler: SchedulerSnapshot,
    submit_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    now: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else float(now)
    control = _assert_drill_control_paused(state_dir)
    validate_readiness(
        control, state_dir=state_dir, verify_files=True, now=timestamp
    )
    if not scheduler.squeue_ok or not scheduler.sacct_ok:
        raise SchedulerAmbiguity("controller drill requires complete scheduler truth")
    active_production = [
        job.job_id for job in _scoped_control_jobs(scheduler) if job.active
    ]
    if active_production:
        raise SchedulerAmbiguity(
            f"controller drill found live production controller jobs: {active_production}"
        )
    live_drill_rows = _all_live_drill_jobs(scheduler)
    live_drills = [job.job_id for job in live_drill_rows]
    malformed_drills = [
        job.job_id
        for job in live_drill_rows
        if parse_drill_job_token(job.comment) is None
    ]
    if malformed_drills:
        raise SchedulerAmbiguity(
            "malformed live controller-drill namespace jobs: "
            + ", ".join(sorted(malformed_drills))
        )
    path = _drill_state_path(state_dir)
    if path.exists():
        state = load_drill_state(state_dir)
        if state["phase"] == "completed":
            if live_drills:
                raise SchedulerAmbiguity(
                    f"completed controller drill still has live jobs: {sorted(live_drills)}"
                )
            validate_controller_drill_marker(
                state_dir,
                control,
                require_current_baseline=int(control["rollout_generation"]) == 0,
            )
            return state
        if live_drills and any(
            parse_drill_job_token(job.comment).get("drill") != state["drill_id"]
            for job in scheduler.jobs
            if job.active and parse_drill_job_token(job.comment) is not None
        ):
            raise SchedulerAmbiguity("another controller drill is live")
    else:
        if live_drills:
            raise SchedulerAmbiguity("unmapped controller-drill jobs are live")
        drill_id = uuid.uuid4().hex
        state = {
            "schema_version": 1,
            "protocol": "schema5-controller-drill-v1",
            "drill_id": drill_id,
            "shared_fencing_primitive": SHARED_CONTROLLER_FENCING_PRIMITIVE,
            "shared_controller_primitives": shared_controller_primitive_contract(),
            "phase": "running",
            "immutable_sha256": control["immutable_sha256"],
            "created_at": utc_timestamp(timestamp),
            "created_timestamp": timestamp,
            "updated_at": utc_timestamp(timestamp),
            "updated_timestamp": timestamp,
            "baseline": controller_drill_baseline(state_dir, control),
            "roles": {
                role: {
                    "next_generation": 1,
                    "active": None,
                    "successor": None,
                    "submission_intent": None,
                    "heartbeat": None,
                    "last_exit": None,
                    "kill": None,
                    "recovery": None,
                }
                for role in ROLE_NAMES
            },
            "transition_history": [],
            "final_reconciliation_path": None,
            "final_reconciliation_sha256": None,
        }
        with drill_lock(state_dir):
            if path.exists():
                state = load_drill_state(state_dir)
            else:
                _append_drill_event(
                    state_dir,
                    state,
                    event="started_paused",
                    details={"desired_state": "paused"},
                    now=timestamp,
                )
                _save_drill_state(state_dir, state, now=timestamp)
    for role in ROLE_NAMES:
        submit_drill_intent(
            state_dir,
            role=role,
            target="active",
            dependency_job_id=None,
            scheduler=scheduler,
            submit_runner=submit_runner,
            now=timestamp,
        )
    return load_drill_state(state_dir)


def _resolve_fenced_role_claim(
    role_state: Mapping[str, Any],
    *,
    role: str,
    expected_token: str,
    generation: int,
    intent_token: str,
    job_id: str,
    active_scheduler_job_ids: set[str],
    context: str,
    now: float,
) -> dict[str, Any]:
    """Shared exact-intent fencing primitive for production and drill controllers."""

    candidates = [
        record
        for field in ("active", "successor", "submission_intent")
        if isinstance((record := role_state.get(field)), dict)
        and record.get("job_token") == expected_token
        and int(record.get("generation", -1)) == generation
        and record.get("intent_token") == intent_token
    ]
    if not candidates:
        raise ControllerFenced(
            f"{context} job {job_id} has no durable exact intent for {role} "
            f"generation {generation}"
        )
    if any(
        record.get("controller_primitives") != shared_controller_primitive_contract()
        for record in candidates
    ):
        raise ControllerFenced(
            f"{context} job {job_id} has the wrong primitive contract"
        )
    exact_ids = {
        str(record.get("job_id")) for record in candidates if record.get("job_id")
    }
    if exact_ids and job_id not in exact_ids:
        raise ControllerFenced(
            f"{context} job {job_id} does not match recorded exact IDs {sorted(exact_ids)}"
        )
    active = role_state.get("active")
    if isinstance(active, dict):
        active_id = str(active.get("job_id", ""))
        if active_id and active_id != job_id and active_id in active_scheduler_job_ids:
            raise ControllerFenced(
                f"live {context} job {active_id} already owns controller role {role}"
            )
    claimed = copy.deepcopy(candidates[0])
    claimed.update(
        {
            "state": "running",
            "job_id": job_id,
            "claimed_at": utc_timestamp(now),
            "claimed_timestamp": now,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "fencing_primitive": SHARED_CONTROLLER_FENCING_PRIMITIVE,
        }
    )
    return claimed


def claim_drill_controller(
    state_dir: Path,
    *,
    drill_id: str,
    role: str,
    generation: int,
    intent_token: str,
    job_id: str,
    active_scheduler_job_ids: set[str],
    now: float | None = None,
) -> dict[str, Any]:
    _validate_role(role)
    if not job_id.isdigit():
        raise ControllerFenced(f"invalid drill SLURM_JOB_ID {job_id!r}")
    timestamp = time.time() if now is None else float(now)
    expected = drill_job_token(drill_id, role, generation, intent_token)
    with drill_lock(state_dir):
        state = load_drill_state(state_dir)
        _assert_drill_control_paused(
            state_dir, expected_immutable_sha256=state["immutable_sha256"]
        )
        if state["drill_id"] != drill_id or state["phase"] != "running":
            raise ControllerFenced("controller drill is not accepting claims")
        row = state["roles"][role]
        claimed = _resolve_fenced_role_claim(
            row,
            role=role,
            expected_token=expected,
            generation=generation,
            intent_token=intent_token,
            job_id=job_id,
            active_scheduler_job_ids=active_scheduler_job_ids,
            context="drill",
            now=timestamp,
        )
        row["active"] = claimed
        if (
            isinstance(row.get("successor"), dict)
            and row["successor"].get("intent_token") == intent_token
        ):
            row["successor"] = None
        row["heartbeat"] = {
            "job_id": job_id,
            "generation": generation,
            "intent_token": intent_token,
            "timestamp": timestamp,
            "at": utc_timestamp(timestamp),
        }
        _append_drill_event(
            state_dir,
            state,
            event="claimed",
            details={"role": role, "generation": generation, "job_id": job_id},
            now=timestamp,
        )
        _save_drill_state(state_dir, state, now=timestamp)
        return claimed


def heartbeat_drill_controller(
    state_dir: Path,
    *,
    drill_id: str,
    role: str,
    generation: int,
    intent_token: str,
    job_id: str,
    now: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else float(now)
    with drill_lock(state_dir):
        state = load_drill_state(state_dir)
        _assert_drill_control_paused(
            state_dir, expected_immutable_sha256=state["immutable_sha256"]
        )
        if state["drill_id"] != drill_id or state["phase"] != "running":
            raise ControllerFenced("controller drill has stopped")
        active = state["roles"][role]["active"]
        identity = (job_id, generation, intent_token)
        recorded = (
            (
                str(active.get("job_id", "")),
                int(active.get("generation", -1)),
                str(active.get("intent_token", "")),
            )
            if isinstance(active, dict)
            else None
        )
        if identity != recorded:
            raise ControllerFenced("drill heartbeat no longer owns the exact role")
        heartbeat = {
            "job_id": job_id,
            "generation": generation,
            "intent_token": intent_token,
            "timestamp": timestamp,
            "at": utc_timestamp(timestamp),
        }
        state["roles"][role]["heartbeat"] = heartbeat
        _save_drill_state(state_dir, state, now=timestamp)
        return heartbeat


def _record_drill_exit(
    state_dir: Path,
    *,
    drill_id: str,
    role: str,
    generation: int,
    intent_token: str,
    job_id: str,
    reason: str,
    now: float | None = None,
) -> None:
    timestamp = time.time() if now is None else float(now)
    with drill_lock(state_dir):
        state = load_drill_state(state_dir)
        if state["drill_id"] != drill_id:
            return
        row = state["roles"][role]
        active = row.get("active")
        if isinstance(active, dict) and (
            str(active.get("job_id")) == job_id
            and int(active.get("generation", -1)) == generation
            and str(active.get("intent_token")) == intent_token
        ):
            row["active"] = None
        row["last_exit"] = {
            "job_id": job_id,
            "generation": generation,
            "intent_token": intent_token,
            "reason": reason,
            "timestamp": timestamp,
            "at": utc_timestamp(timestamp),
        }
        _append_drill_event(
            state_dir,
            state,
            event="exited",
            details={"role": role, "job_id": job_id, "reason": reason},
            now=timestamp,
        )
        _save_drill_state(state_dir, state, now=timestamp)


def supervise_drill(
    state_dir: Path,
    *,
    drill_id: str,
    role: str,
    generation: int,
    intent_token: str,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> int:
    """Exercise the real fencing/successor boundary without starting a managed plane."""

    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id:
        raise ControllerFenced("supervise-drill must run inside Slurm")
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    previous = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    reason = "error"
    try:
        with role_singleton_lock(state_dir, role):
            snapshot = query_scheduler()
            claim_drill_controller(
                state_dir,
                drill_id=drill_id,
                role=role,
                generation=generation,
                intent_token=intent_token,
                job_id=job_id,
                active_scheduler_job_ids=snapshot.active_job_ids | {job_id},
            )
            submit_drill_intent(
                state_dir,
                role=role,
                target="successor",
                dependency_job_id=job_id,
                scheduler=snapshot,
            )
            last_heartbeat = 0.0
            while not stop:
                state = load_drill_state(state_dir)
                if state["phase"] != "running":
                    reason = "drill_stopping"
                    return 0
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    heartbeat_drill_controller(
                        state_dir,
                        drill_id=drill_id,
                        role=role,
                        generation=generation,
                        intent_token=intent_token,
                        job_id=job_id,
                        now=now,
                    )
                    snapshot = query_scheduler(tolerate_errors=True, now=now)
                    try:
                        submit_drill_intent(
                            state_dir,
                            role=role,
                            target="successor",
                            dependency_job_id=job_id,
                            scheduler=snapshot,
                            now=now,
                        )
                    except (ControlError, OSError) as exc:
                        print(
                            f"[schema5-control] drill successor retry required: {exc}",
                            file=sys.stderr,
                        )
                    last_heartbeat = now
                time.sleep(min(1.0, heartbeat_interval))
            reason = "signal"
            return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        try:
            _record_drill_exit(
                state_dir,
                drill_id=drill_id,
                role=role,
                generation=generation,
                intent_token=intent_token,
                job_id=job_id,
                reason=reason,
            )
        except ControlError as exc:
            print(
                f"[schema5-control] could not record drill exit: {exc}", file=sys.stderr
            )


def controller_drill_status(
    state_dir: Path, *, snapshot: SchedulerSnapshot, now: float | None = None
) -> dict[str, Any]:
    timestamp = snapshot.captured_at if now is None else float(now)
    state = load_drill_state(state_dir)
    control = _assert_drill_control_paused(
        state_dir, expected_immutable_sha256=state["immutable_sha256"]
    )
    namespace_jobs = [
        job for job in snapshot.jobs if job.comment.startswith(DRILL_TOKEN_PREFIX)
    ]
    valid_namespace_jobs = [
        job for job in namespace_jobs if parse_drill_job_token(job.comment) is not None
    ]
    live = [job for job in namespace_jobs if job.active]
    valid_live = [
        job for job in live if parse_drill_job_token(job.comment) is not None
    ]
    malformed_live = [
        job.job_id for job in live if parse_drill_job_token(job.comment) is None
    ]
    foreign = [
        job.job_id
        for job in valid_live
        if parse_drill_job_token(job.comment).get("drill") != state["drill_id"]
    ]
    roles: dict[str, Any] = {}
    errors: list[str] = list(snapshot.errors)
    if not snapshot.squeue_ok or not snapshot.sacct_ok:
        errors.append("controller drill lacks complete scheduler truth")
    if malformed_live:
        errors.append(
            f"malformed live controller-drill namespace jobs: {sorted(malformed_live)}"
        )
    if foreign:
        errors.append(f"foreign live controller drill jobs: {sorted(foreign)}")
    recorded_tokens: set[str] = set()
    for role, row in state["roles"].items():
        role_status: dict[str, Any] = {}
        for field in ("active", "successor", "submission_intent"):
            record = row.get(field)
            if not isinstance(record, dict):
                role_status[field] = None
                continue
            token = str(record.get("job_token", ""))
            recorded_tokens.add(token)
            all_matches = [job for job in valid_namespace_jobs if job.comment == token]
            matches = [job for job in all_matches if job.active]
            if len(all_matches) > 1:
                errors.append(f"duplicate drill token history for {role} {field}")
            recorded_id = record.get("job_id")
            if (
                recorded_id
                and matches
                and any(job.job_id != str(recorded_id) for job in matches)
            ):
                errors.append(f"scheduler ID mismatch for {role} {field}")
            if recorded_id:
                exact = [job for job in matches if job.job_id == str(recorded_id)]
                if exact and not _command_binds_exact_sbatch(
                    exact[0].command, str(record.get("sbatch_path", ""))
                ):
                    errors.append(
                        f"scheduler command does not bind recorded sbatch for {role} {field}"
                    )
                expected_parent = record.get("dependency_job_id")
                if (
                    exact
                    and expected_parent is not None
                    and not _dependency_binds_exact_afterany(
                        exact[0].dependency, str(expected_parent)
                    )
                ):
                    errors.append(
                        f"scheduler dependency does not bind recorded parent for {role} {field}"
                    )
            role_status[field] = {
                "recorded_job_id": recorded_id,
                "matching_job_ids": [job.job_id for job in all_matches],
                "live_job_ids": [job.job_id for job in matches],
            }
        heartbeat = row.get("heartbeat")
        role_status["heartbeat_age_seconds"] = (
            timestamp - float(heartbeat["timestamp"])
            if isinstance(heartbeat, dict)
            else None
        )
        role_status["kill"] = copy.deepcopy(row.get("kill"))
        role_status["recovery"] = copy.deepcopy(row.get("recovery"))
        roles[role] = role_status
    unexpected = [job.job_id for job in live if job.comment not in recorded_tokens]
    if unexpected:
        errors.append(f"unmapped live controller-drill jobs: {sorted(unexpected)}")
    ready = state["phase"] == "running" and all(
        roles[role]["active"] is not None
        and len(roles[role]["active"]["live_job_ids"]) == 1
        and roles[role]["successor"] is not None
        and len(roles[role]["successor"]["live_job_ids"]) == 1
        and roles[role]["heartbeat_age_seconds"] is not None
        and 0 <= roles[role]["heartbeat_age_seconds"] <= 2 * HEARTBEAT_INTERVAL_SECONDS
        for role in ROLE_NAMES
    )
    completed = False
    if state["phase"] == "completed":
        if live:
            errors.append(
                f"completed controller drill still has live jobs: "
                f"{sorted(job.job_id for job in live)}"
            )
        try:
            validate_controller_drill_marker(
                state_dir,
                control,
                require_current_baseline=int(control["rollout_generation"]) == 0,
            )
        except ControlError as exc:
            errors.append(f"completed controller drill proof is invalid: {exc}")
        completed = not errors
    return {
        "schema_version": 1,
        "drill_id": state["drill_id"],
        "phase": state["phase"],
        "desired_state": control["desired_state"],
        "roles": roles,
        "live_job_ids": sorted(job.job_id for job in live),
        "errors": errors,
        "healthy": not errors,
        "ready_for_kill": not errors and ready,
        "completed": completed,
    }


def kill_drill_controller(
    state_dir: Path,
    *,
    role: str,
    snapshot: SchedulerSnapshot,
    cancel_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Persist and then cancel exactly one token-and-command-verified drill job."""

    _validate_role(role)
    if not snapshot.squeue_ok or not snapshot.sacct_ok:
        raise SchedulerAmbiguity("drill kill requires complete scheduler truth")
    timestamp = snapshot.captured_at if now is None else float(now)
    status = controller_drill_status(state_dir, snapshot=snapshot, now=timestamp)
    if status["errors"]:
        raise SchedulerAmbiguity("; ".join(status["errors"]))
    runner = cancel_runner or _run_subprocess
    with drill_lock(state_dir):
        state = load_drill_state(state_dir)
        _assert_drill_control_paused(
            state_dir, expected_immutable_sha256=state["immutable_sha256"]
        )
        if state["phase"] != "running":
            raise ControlError("controller drill is not running")
        row = state["roles"][role]
        if isinstance(row.get("kill"), dict) and row["kill"].get("returncode") == 0:
            return copy.deepcopy(row["kill"])
        existing_kill = row.get("kill")
        active = row.get("active")
        successor = row.get("successor")
        if isinstance(existing_kill, dict):
            kill = existing_kill
            sbatch_path = str(kill.get("sbatch_path", ""))
            if not sbatch_path and isinstance(active, dict):
                sbatch_path = str(active.get("sbatch_path", ""))
            if not sbatch_path:
                raise SchedulerAmbiguity(
                    f"{role} persisted kill has no sbatch provenance"
                )
            target_job = _unique_exact_drill_job(
                snapshot,
                job_id=str(kill.get("job_id", "")),
                job_token=str(kill.get("job_token", "")),
                sbatch_path=sbatch_path,
                require_active=None,
            )
            if not target_job.active:
                terminal_state = normalize_scheduler_state(target_job.state)
                if terminal_state != "CANCELLED":
                    raise SchedulerAmbiguity(
                        f"persisted exact kill target {kill['job_id']} is {terminal_state}, "
                        "which does not prove scancel acceptance"
                    )
                kill["returncode"] = 0
                kill["stderr"] = ""
                kill["completed_at"] = utc_timestamp(timestamp)
                kill["completed_timestamp"] = timestamp
                kill["state"] = "cancelled_reconciled_terminal"
                kill["scheduler_terminal_state"] = terminal_state
                kill["reconciled_from_scheduler"] = True
                kill["sbatch_path"] = sbatch_path
                _append_drill_event(
                    state_dir,
                    state,
                    event="exact_kill_reconciled_terminal",
                    details={
                        "role": role,
                        "job_id": kill["job_id"],
                        "scheduler_state": kill["scheduler_terminal_state"],
                    },
                    now=timestamp,
                )
                _save_drill_state(state_dir, state, now=timestamp)
                return copy.deepcopy(kill)
            kill["state"] = "cancelling"
            kill["sbatch_path"] = sbatch_path
            kill["retry_count"] = int(kill.get("retry_count", 0)) + 1
            _append_drill_event(
                state_dir,
                state,
                event="exact_kill_retry",
                details={"role": role, "job_id": kill["job_id"]},
                now=timestamp,
            )
            _save_drill_state(state_dir, state, now=timestamp)
        else:
            if not status["ready_for_kill"]:
                raise SchedulerAmbiguity(
                    "controller drill is not globally ready for an exact kill"
                )
            if not isinstance(active, dict) or not active.get("job_id"):
                raise SchedulerAmbiguity(f"{role} has no exact active drill job")
            if not isinstance(successor, dict) or not successor.get("job_id"):
                raise SchedulerAmbiguity(f"{role} has no recorded drill successor")
            if str(successor.get("dependency_job_id", "")) != str(active["job_id"]):
                raise SchedulerAmbiguity(
                    f"{role} successor does not bind the exact active parent"
                )
            _unique_exact_drill_job(
                snapshot,
                job_id=str(active["job_id"]),
                job_token=str(active["job_token"]),
                sbatch_path=str(active["sbatch_path"]),
                require_active=True,
            )
            _unique_exact_drill_job(
                snapshot,
                job_id=str(successor["job_id"]),
                job_token=str(successor["job_token"]),
                sbatch_path=str(successor["sbatch_path"]),
                require_active=True,
                expected_dependency_job_id=str(active["job_id"]),
            )
            kill = {
                "state": "cancelling",
                "job_id": str(active["job_id"]),
                "job_token": str(active["job_token"]),
                "sbatch_path": str(active["sbatch_path"]),
                "fencing_primitive": active.get("fencing_primitive"),
                "controller_primitives": active.get("controller_primitives"),
                "successor_job_id": str(successor["job_id"]),
                "successor_job_token": str(successor["job_token"]),
                "successor_sbatch_path": str(successor["sbatch_path"]),
                "successor_dependency_job_id": str(active["job_id"]),
                "started_at": utc_timestamp(timestamp),
                "started_timestamp": timestamp,
            }
            row["kill"] = kill
            _append_drill_event(
                state_dir,
                state,
                event="exact_kill_intent",
                details={"role": role, "job_id": kill["job_id"]},
                now=timestamp,
            )
            _save_drill_state(state_dir, state, now=timestamp)
    proc = runner(["scancel", kill["job_id"]])
    completed = time.time() if now is None else timestamp
    with drill_lock(state_dir):
        state = load_drill_state(state_dir)
        current = state["roles"][role].get("kill")
        if not isinstance(current, dict) or current.get("job_id") != kill["job_id"]:
            raise SchedulerAmbiguity("drill kill intent changed during scancel")
        current["returncode"] = int(proc.returncode)
        current["stderr"] = proc.stderr.strip()[:1000]
        current["completed_at"] = utc_timestamp(completed)
        current["completed_timestamp"] = completed
        current["state"] = "cancelled" if proc.returncode == 0 else "failed"
        _append_drill_event(
            state_dir,
            state,
            event="exact_kill_result",
            details={
                "role": role,
                "job_id": kill["job_id"],
                "returncode": proc.returncode,
            },
            now=completed,
        )
        _save_drill_state(state_dir, state, now=completed)
        result = copy.deepcopy(current)
    if proc.returncode != 0:
        raise ControlError(
            f"failed exact drill kill {kill['job_id']}: {proc.stderr.strip()[:500]}"
        )
    return result


def record_drill_recovery(
    state_dir: Path,
    *,
    role: str,
    snapshot: SchedulerSnapshot,
    max_seconds: float = 900.0,
    now: float | None = None,
) -> dict[str, Any]:
    _validate_role(role)
    timestamp = snapshot.captured_at if now is None else float(now)
    status = controller_drill_status(state_dir, snapshot=snapshot, now=timestamp)
    if status["errors"]:
        raise SchedulerAmbiguity("; ".join(status["errors"]))
    with drill_lock(state_dir):
        state = load_drill_state(state_dir)
        row = state["roles"][role]
        kill = row.get("kill")
        active = row.get("active")
        heartbeat = row.get("heartbeat")
        if not isinstance(kill, dict) or kill.get("returncode") != 0:
            raise ControlError(f"{role} has no successful exact kill")
        if not isinstance(active, dict) or not isinstance(heartbeat, dict):
            raise ControlError(f"{role} successor has not claimed and heartbeated")
        if str(active.get("job_id")) == str(kill.get("job_id")):
            raise ControlError(f"{role} still reports the killed allocation")
        expected_successor = (
            str(kill.get("successor_job_id", "")),
            str(kill.get("successor_job_token", "")),
            str(Path(str(kill.get("successor_sbatch_path", ""))).expanduser().resolve()),
            str(kill.get("successor_dependency_job_id", "")),
        )
        active_successor = (
            str(active.get("job_id", "")),
            str(active.get("job_token", "")),
            str(Path(str(active.get("sbatch_path", ""))).expanduser().resolve()),
            str(active.get("dependency_job_id", "")),
        )
        if not all(expected_successor) or active_successor != expected_successor:
            raise ControllerFenced(
                f"{role} recovery did not claim the exact successor frozen by the kill intent"
            )
        active_heartbeat_identity = (
            str(active.get("job_id", "")),
            int(active.get("generation", -1)),
            str(active.get("intent_token", "")),
        )
        heartbeat_identity = (
            str(heartbeat.get("job_id", "")),
            int(heartbeat.get("generation", -1)),
            str(heartbeat.get("intent_token", "")),
        )
        if active_heartbeat_identity != heartbeat_identity:
            raise ControllerFenced(
                f"{role} successor heartbeat does not own the active role"
            )
        _unique_exact_drill_job(
            snapshot,
            job_id=expected_successor[0],
            job_token=expected_successor[1],
            sbatch_path=expected_successor[2],
            require_active=True,
            expected_dependency_job_id=expected_successor[3],
        )
        recovered_at = float(heartbeat["timestamp"])
        recovery_seconds = recovered_at - float(kill["started_timestamp"])
        if recovery_seconds < 0 or recovery_seconds > max_seconds:
            raise ControlError(
                f"{role} recovery took {recovery_seconds:.1f}s; limit is {max_seconds:.1f}s"
            )
        recovery = {
            "passed": True,
            "killed_job_id": str(kill["job_id"]),
            "successor_job_id": str(active["job_id"]),
            "successor_job_token": str(active["job_token"]),
            "successor_sbatch_path": str(active["sbatch_path"]),
            "successor_dependency_job_id": str(active["dependency_job_id"]),
            "successor_generation": int(active["generation"]),
            "successor_intent_token": str(active["intent_token"]),
            "recovered_at": utc_timestamp(recovered_at),
            "recovered_timestamp": recovered_at,
            "recovery_seconds": recovery_seconds,
            "maximum_seconds": max_seconds,
            "fencing_primitive": active.get("fencing_primitive"),
            "controller_primitives": active.get("controller_primitives"),
        }
        row["recovery"] = recovery
        _append_drill_event(
            state_dir,
            state,
            event="recovery_verified",
            details={"role": role, **recovery},
            now=timestamp,
        )
        _save_drill_state(state_dir, state, now=timestamp)
        return copy.deepcopy(recovery)


def wait_for_drill_recovery(
    state_dir: Path,
    *,
    role: str,
    scheduler_reader: Callable[[], SchedulerSnapshot] | None = None,
    timeout_seconds: float = 900.0,
    poll_seconds: float = 5.0,
) -> dict[str, Any]:
    reader = scheduler_reader or query_scheduler
    started = time.monotonic()
    last_error: Exception | None = None
    while time.monotonic() - started <= timeout_seconds:
        snapshot = reader()
        try:
            return record_drill_recovery(
                state_dir,
                role=role,
                snapshot=snapshot,
                max_seconds=timeout_seconds,
            )
        except (ControlError, OSError) as exc:
            last_error = exc
        time.sleep(min(poll_seconds, max(0.0, timeout_seconds)))
    raise ControlError(f"{role} drill recovery timed out: {last_error}")


def _live_drill_jobs(snapshot: SchedulerSnapshot, drill_id: str) -> list[SchedulerJob]:
    return [
        job
        for job in snapshot.jobs
        if job.active
        and (parsed := parse_drill_job_token(job.comment)) is not None
        and parsed["drill"] == drill_id
    ]


def _controller_drill_marker_sha256(state_dir: Path) -> str:
    marker_path = state_dir / DRILL_COMPLETE_FILENAME
    try:
        return sha256_file(marker_path)
    except FileNotFoundError as exc:
        raise ReadinessError(
            f"controller kill drill marker is missing: {marker_path}"
        ) from exc
    except OSError as exc:
        raise ReadinessError(
            f"controller kill drill marker cannot be read: {marker_path}: {exc}"
        ) from exc


def _publish_drill_reconciliation(state_dir: Path, drill_id: str) -> tuple[Path, str]:
    """Seal or recover the immutable final no-admission reconciliation report.

    A prior attempt may have published the sealed copy and died before committing the
    completed drill state.  In that case the immutable target is the transaction's
    durable preimage: validate and reuse it instead of comparing it with a newly
    timestamped live reconciliation.
    """

    target = _drill_reconciliation_path(state_dir, drill_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target_present = target.exists() or target.is_symlink()
    if target_present and (
        target.is_symlink()
        or not target.is_file()
        or target.stat().st_mode & 0o222
    ):
        raise ImmutablePinError(
            f"drill reconciliation is not an immutable regular file: {target}"
        )
    source = target if target_present else state_dir / RECONCILIATION_FILENAME
    try:
        payload = source.read_text(encoding="utf-8")
        report = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlError(f"cannot seal drill reconciliation {source}: {exc}") from exc
    control = load_control(state_dir)
    if (
        not isinstance(report, dict)
        or report.get("passed") is not True
        or report.get("immutable_sha256") != control["immutable_sha256"]
        or report.get("all") is not True
        or report.get("no_admit") is not True
        or report.get("errors") != []
    ):
        raise ControlError("cannot seal a non-clean drill reconciliation")
    if not target_present:
        io.atomic_write_text(target, payload)
        target.chmod(0o444)
    return target.resolve(), sha256_file(target)


def _validate_completed_drill_history(state: Mapping[str, Any]) -> None:
    """Validate the semantic event/role proof, not only JSON shape and hashes."""

    history = state["transition_history"]
    events = [str(record.get("event", "")) for record in history]
    if not events or events[0] != "started_paused" or events[-1] != "completed":
        raise ReadinessError(
            "controller kill drill history lacks start/completion boundaries"
        )
    if events.count("completed") != 1:
        raise ReadinessError(
            "controller kill drill history must contain exactly one completion"
        )
    try:
        stopping_index = max(
            index for index, event in enumerate(events) if event == "stopping"
        )
        completed_index = len(events) - 1
    except ValueError as exc:
        raise ReadinessError(
            "controller kill drill history lacks a stopping transition"
        ) from exc
    if stopping_index >= completed_index:
        raise ReadinessError(
            "controller kill drill stopping/completion order is invalid"
        )
    for role in ROLE_NAMES:
        row = state["roles"][role]
        kill = row.get("kill")
        recovery = row.get("recovery")
        if (
            not isinstance(kill, dict)
            or kill.get("returncode") != 0
            or kill.get("fencing_primitive") != SHARED_CONTROLLER_FENCING_PRIMITIVE
            or kill.get("controller_primitives")
            != shared_controller_primitive_contract()
            or not isinstance(recovery, dict)
            or recovery.get("passed") is not True
            or recovery.get("fencing_primitive") != SHARED_CONTROLLER_FENCING_PRIMITIVE
            or recovery.get("controller_primitives")
            != shared_controller_primitive_contract()
            or str(kill.get("job_id")) != str(recovery.get("killed_job_id"))
            or str(kill.get("successor_job_id"))
            != str(recovery.get("successor_job_id"))
            or str(kill.get("successor_job_token"))
            != str(recovery.get("successor_job_token"))
            or str(Path(str(kill.get("successor_sbatch_path", ""))).resolve())
            != str(Path(str(recovery.get("successor_sbatch_path", ""))).resolve())
            or str(kill.get("successor_dependency_job_id", ""))
            != str(recovery.get("successor_dependency_job_id", ""))
            or str(recovery.get("successor_dependency_job_id", ""))
            != str(recovery.get("killed_job_id", ""))
            or str(recovery.get("successor_job_id"))
            == str(recovery.get("killed_job_id"))
        ):
            raise ReadinessError(
                f"controller kill drill state proof is invalid for {role}"
            )
        intent_indexes = [
            index
            for index, record in enumerate(history)
            if record.get("event") == "exact_kill_intent"
            and record.get("details", {}).get("role") == role
            and str(record.get("details", {}).get("job_id")) == str(kill["job_id"])
        ]
        result_indexes = [
            index
            for index, record in enumerate(history)
            if (
                (
                    record.get("event") == "exact_kill_result"
                    and record.get("details", {}).get("returncode") == 0
                )
                or (
                    record.get("event") == "exact_kill_reconciled_terminal"
                    and record.get("details", {}).get("scheduler_state") == "CANCELLED"
                )
            )
            and record.get("details", {}).get("role") == role
            and str(record.get("details", {}).get("job_id")) == str(kill["job_id"])
        ]
        recovery_indexes = [
            index
            for index, record in enumerate(history)
            if record.get("event") == "recovery_verified"
            and record.get("details", {}).get("role") == role
            and str(record.get("details", {}).get("killed_job_id"))
            == str(kill["job_id"])
            and str(record.get("details", {}).get("successor_job_id"))
            == str(recovery["successor_job_id"])
        ]
        if not intent_indexes or not result_indexes or not recovery_indexes:
            raise ReadinessError(
                f"controller kill drill event proof is incomplete for {role}"
            )
        if not (
            min(intent_indexes)
            < max(result_indexes)
            < max(recovery_indexes)
            < stopping_index
        ):
            raise ReadinessError(
                f"controller kill drill event order is invalid for {role}"
            )


def _recover_completed_drill_journal_suffix(
    state_dir: Path, state: MutableMapping[str, Any]
) -> dict[str, Any]:
    """Replay the sole completed event if death preceded the state replacement.

    Drill events are journaled before the JSON state is atomically replaced.  The
    completed event contains every field needed to finish that one interrupted state
    transaction.  Replaying it avoids emitting a second semantic completion on retry.
    The caller holds ``drill_lock``.
    """

    durable = _read_jsonl_locked(state_dir / DRILL_JOURNAL_FILENAME)
    memory = list(state["transition_history"])
    if durable[: len(memory)] != memory:
        raise ControlError("controller-drill journal diverged from its state")
    suffix = durable[len(memory) :]
    if not suffix:
        return dict(state)
    completed = [record for record in suffix if record.get("event") == "completed"]
    if not completed:
        return dict(state)
    if state.get("phase") != "stopping" or len(suffix) != 1 or len(completed) != 1:
        raise ReadinessError(
            "controller drill has an ambiguous journal suffix around completion"
        )
    record = completed[0]
    details = record.get("details")
    if not isinstance(details, dict) or set(details) != {
        "final_reconciliation_path",
        "final_reconciliation_sha256",
    }:
        raise ReadinessError("controller drill completion journal payload is invalid")
    raw_path = Path(str(details["final_reconciliation_path"]))
    expected_path = _drill_reconciliation_path(
        state_dir, str(state["drill_id"])
    ).resolve()
    expected_sha = str(details["final_reconciliation_sha256"])
    if (
        raw_path.is_symlink()
        or raw_path.resolve() != expected_path
        or not expected_path.is_file()
        or expected_path.stat().st_mode & 0o222
        or _SHA256_RE.fullmatch(expected_sha) is None
        or sha256_file(expected_path) != expected_sha
    ):
        raise ReadinessError(
            "controller drill completion journal lacks its sealed reconciliation"
        )
    recovered = copy.deepcopy(dict(state))
    recovered["transition_history"] = copy.deepcopy(durable)
    recovered["phase"] = "completed"
    recovered["final_reconciliation_path"] = str(expected_path)
    recovered["final_reconciliation_sha256"] = expected_sha
    recovered["completed_at"] = record.get("at")
    recovered["completed_timestamp"] = record.get("timestamp")
    _validate_completed_drill_history(recovered)
    _save_drill_state(
        state_dir,
        recovered,
        now=float(record.get("timestamp")),
    )
    return recovered


def validate_controller_drill_marker(
    state_dir: Path,
    control: Mapping[str, Any],
    *,
    require_current_baseline: bool = True,
    allow_completed_drain: bool = False,
    allow_resuming: bool = False,
) -> dict[str, Any]:
    """Require marker-last proof that both paused kill drills recovered safely."""

    marker_path = state_dir / DRILL_COMPLETE_FILENAME
    if marker_path.is_symlink():
        raise ReadinessError(
            f"controller kill drill marker cannot be a symlink: {marker_path}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReadinessError(
            f"controller kill drill marker is missing: {marker_path}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"controller kill drill marker is invalid: {exc}") from exc
    expected_fields = {
        "schema_version",
        "protocol",
        "passed",
        "drill_id",
        "shared_fencing_primitive",
        "shared_controller_primitives",
        "immutable_sha256",
        "completed_at",
        "completed_timestamp",
        "baseline",
        "roles",
        "final_reconciliation_path",
        "final_reconciliation_sha256",
        "drill_state_path",
        "drill_state_sha256",
    }
    if not isinstance(marker, dict) or set(marker) != expected_fields:
        raise ReadinessError("controller kill drill marker fields differ from schema")
    if (
        marker["schema_version"] != 1
        or marker["protocol"] != "schema5-controller-kill-drill-v1"
        or marker["passed"] is not True
        or marker["immutable_sha256"] != control["immutable_sha256"]
        or marker["shared_fencing_primitive"] != SHARED_CONTROLLER_FENCING_PRIMITIVE
        or marker["shared_controller_primitives"]
        != shared_controller_primitive_contract()
    ):
        raise ReadinessError("controller kill drill marker identity is invalid")
    desired_state = control.get("desired_state")
    if desired_state != "paused" and not (
        allow_resuming and desired_state == "resuming"
    ):
        raise ReadinessError("controller kill drill can only authorize a paused resume")
    if control.get("drain_requested") and (
        not allow_completed_drain
        or not isinstance(control.get("drain_intent"), dict)
        or control["drain_intent"].get("state") != "complete"
    ):
        raise ReadinessError(
            "controller kill drill cannot authorize an incomplete pause drain"
        )
    if set(marker.get("roles", {})) != set(ROLE_NAMES):
        raise ReadinessError(
            "controller kill drill did not cover both controller roles"
        )
    for role, recovery in marker["roles"].items():
        parsed_successor = (
            parse_drill_job_token(str(recovery.get("successor_job_token", "")))
            if isinstance(recovery, dict)
            else None
        )
        if (
            not isinstance(recovery, dict)
            or recovery.get("passed") is not True
            or not str(recovery.get("killed_job_id", "")).isdigit()
            or not str(recovery.get("successor_job_id", "")).isdigit()
            or recovery.get("killed_job_id") == recovery.get("successor_job_id")
            or not isinstance(recovery.get("recovery_seconds"), (int, float))
            or float(recovery["recovery_seconds"]) < 0
            or float(recovery["recovery_seconds"]) > 900.0
            or recovery.get("fencing_primitive") != SHARED_CONTROLLER_FENCING_PRIMITIVE
            or recovery.get("controller_primitives")
            != shared_controller_primitive_contract()
            or parsed_successor is None
            or parsed_successor.get("role") != role
            or int(parsed_successor["generation"])
            != recovery.get("successor_generation")
            or parsed_successor.get("intent")
            != recovery.get("successor_intent_token")
            or not Path(
                str(recovery.get("successor_sbatch_path", ""))
            ).is_absolute()
            or recovery.get("successor_dependency_job_id")
            != recovery.get("killed_job_id")
        ):
            raise ReadinessError(
                f"controller kill drill recovery is invalid for {role}"
            )
    raw_drill_path = Path(str(marker["drill_state_path"]))
    drill_path = raw_drill_path.resolve()
    if (
        raw_drill_path.is_symlink()
        or drill_path != _drill_state_path(state_dir).resolve()
        or not drill_path.is_file()
    ):
        raise ReadinessError("controller kill drill state path is invalid")
    if sha256_file(drill_path) != marker["drill_state_sha256"]:
        raise ReadinessError("controller kill drill state drifted after completion")
    try:
        state = load_drill_state(state_dir)
    except ControlError as exc:
        raise ReadinessError(f"controller kill drill state is invalid: {exc}") from exc
    journal = _read_jsonl_locked(state_dir / DRILL_JOURNAL_FILENAME)
    if journal != state["transition_history"]:
        raise ReadinessError(
            "completed controller kill drill state does not exactly match its journal"
        )
    if state["phase"] != "completed" or state["drill_id"] != marker["drill_id"]:
        raise ReadinessError("controller kill drill state is not completed")
    if (
        state.get("immutable_sha256") != marker["immutable_sha256"]
        or state.get("shared_fencing_primitive") != marker["shared_fencing_primitive"]
        or state.get("shared_controller_primitives")
        != marker["shared_controller_primitives"]
        or state.get("baseline") != marker["baseline"]
        or state.get("completed_at") != marker["completed_at"]
        or state.get("completed_timestamp") != marker["completed_timestamp"]
        or {role: state["roles"][role].get("recovery") for role in ROLE_NAMES}
        != marker["roles"]
        or state.get("final_reconciliation_path") != marker["final_reconciliation_path"]
        or state.get("final_reconciliation_sha256")
        != marker["final_reconciliation_sha256"]
    ):
        raise ReadinessError(
            "controller kill drill marker does not exactly match state"
        )
    _validate_completed_drill_history(state)
    raw_reconciliation_path = Path(str(marker["final_reconciliation_path"]))
    reconciliation_path = raw_reconciliation_path.resolve()
    if (
        raw_reconciliation_path.is_symlink()
        or reconciliation_path
        != _drill_reconciliation_path(state_dir, state["drill_id"]).resolve()
        or not reconciliation_path.is_file()
        or reconciliation_path.stat().st_mode & 0o222
        or sha256_file(reconciliation_path) != marker["final_reconciliation_sha256"]
    ):
        raise ReadinessError(
            "controller kill drill lacks its final clean reconciliation"
        )
    try:
        sealed_reconciliation = json.loads(
            reconciliation_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"sealed drill reconciliation is invalid: {exc}") from exc
    if (
        not isinstance(sealed_reconciliation, dict)
        or sealed_reconciliation.get("passed") is not True
        or sealed_reconciliation.get("immutable_sha256") != control["immutable_sha256"]
        or sealed_reconciliation.get("all") is not True
        or sealed_reconciliation.get("no_admit") is not True
        or sealed_reconciliation.get("errors") != []
    ):
        raise ReadinessError(
            "sealed drill reconciliation does not prove a clean no-admit join"
        )
    if require_current_baseline:
        current_baseline = controller_drill_baseline(state_dir, control)
        if current_baseline != marker["baseline"]:
            raise ReadinessError(
                "production state changed after the controller kill drill"
            )
    return marker


def _completed_drill_marker_payload(
    state_dir: Path, state: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the sole completion payload from an already durable completed state."""

    if state.get("phase") != "completed":
        raise ControlError("controller drill state is not ready for marker publication")
    _validate_completed_drill_history(state)
    reconciliation_path, reconciliation_sha = _publish_drill_reconciliation(
        state_dir, str(state["drill_id"])
    )
    if (
        state.get("final_reconciliation_path") != str(reconciliation_path)
        or state.get("final_reconciliation_sha256") != reconciliation_sha
    ):
        raise ImmutablePinError(
            "completed controller drill state does not bind its sealed reconciliation"
        )
    return {
        "schema_version": 1,
        "protocol": "schema5-controller-kill-drill-v1",
        "passed": True,
        "drill_id": state["drill_id"],
        "shared_fencing_primitive": state["shared_fencing_primitive"],
        "shared_controller_primitives": state["shared_controller_primitives"],
        "immutable_sha256": state["immutable_sha256"],
        "completed_at": state["completed_at"],
        "completed_timestamp": state["completed_timestamp"],
        "baseline": state["baseline"],
        "roles": {
            role: copy.deepcopy(state["roles"][role]["recovery"])
            for role in ROLE_NAMES
        },
        "final_reconciliation_path": str(reconciliation_path),
        "final_reconciliation_sha256": reconciliation_sha,
        "drill_state_path": str(_drill_state_path(state_dir).resolve()),
        "drill_state_sha256": sha256_file(_drill_state_path(state_dir)),
    }


def finish_controller_drill(
    state_dir: Path,
    *,
    scheduler_reader: Callable[[], SchedulerSnapshot] | None = None,
    cancel_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    timeout_seconds: float = 120.0,
    poll_seconds: float = 2.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Stop drill chains, prove no mutation, reconcile, and publish completion last."""

    reader = scheduler_reader or query_scheduler
    runner = cancel_runner or _run_subprocess
    timestamp = time.time() if now is None else float(now)
    marker_path = state_dir / DRILL_COMPLETE_FILENAME
    if marker_path.exists() or marker_path.is_symlink():
        if marker_path.is_symlink() or not marker_path.is_file():
            raise ReadinessError(
                f"controller kill drill marker is not a regular file: {marker_path}"
            )
        control = _assert_drill_control_paused(state_dir)
        completed_snapshot = reader()
        if not completed_snapshot.squeue_ok or not completed_snapshot.sacct_ok:
            raise SchedulerAmbiguity(
                "completed drill validation requires complete scheduler truth"
            )
        lingering = _all_live_drill_jobs(completed_snapshot)
        if lingering:
            raise SchedulerAmbiguity(
                f"completed controller drill still has live jobs: "
                f"{sorted(job.job_id for job in lingering)}"
            )
        return validate_controller_drill_marker(
            state_dir,
            control,
            require_current_baseline=int(control["rollout_generation"]) == 0,
        )
    # Marker publication is a separate final filesystem transaction.  If the prior
    # process died after committing the completed state, re-prove namespace emptiness
    # and the unchanged baseline, then publish the exact marker without replaying the
    # timestamped reconciliation/completion transition.
    with drill_lock(state_dir):
        prior_state = load_drill_state(state_dir)
        prior_state = _recover_completed_drill_journal_suffix(
            state_dir, prior_state
        )
    if prior_state["phase"] == "completed":
        completed_snapshot = reader()
        if not completed_snapshot.squeue_ok or not completed_snapshot.sacct_ok:
            raise SchedulerAmbiguity(
                "completed drill recovery requires complete scheduler truth"
            )
        lingering = _all_live_drill_jobs(completed_snapshot)
        if lingering:
            raise SchedulerAmbiguity(
                "completed controller drill recovery found live namespace jobs: "
                + ", ".join(sorted(job.job_id for job in lingering))
            )
        control = _assert_drill_control_paused(
            state_dir,
            expected_immutable_sha256=prior_state["immutable_sha256"],
        )
        if controller_drill_baseline(state_dir, control) != prior_state["baseline"]:
            raise ControlError(
                "production state changed after the completed controller drill"
            )
        marker = _completed_drill_marker_payload(state_dir, prior_state)
        _atomic_write_json(marker_path, marker)
        return validate_controller_drill_marker(
            state_dir,
            control,
            require_current_baseline=int(control["rollout_generation"]) == 0,
        )
    # Acquiring the same boundary used across every external sbatch call proves there
    # is no live submitter before the durable stopping transition is published.
    with drill_submission_boundary_lock(state_dir):
        with drill_lock(state_dir):
            state = load_drill_state(state_dir)
            _assert_drill_control_paused(
                state_dir, expected_immutable_sha256=state["immutable_sha256"]
            )
            if any(
                not isinstance(state["roles"][role].get("recovery"), dict)
                or state["roles"][role]["recovery"].get("passed") is not True
                for role in ROLE_NAMES
            ):
                raise ControlError(
                    "both controller drill recoveries must pass before finish"
                )
            if state["phase"] == "running":
                state["phase"] = "stopping"
                _append_drill_event(
                    state_dir,
                    state,
                    event="stopping",
                    details={"reason": "both_recoveries_passed"},
                    now=timestamp,
                )
                _save_drill_state(state_dir, state, now=timestamp)
    deadline = time.monotonic() + timeout_seconds
    cancelled_job_ids: set[str] = set()
    clean_snapshots = 0
    final_snapshot = reader()
    state = load_drill_state(state_dir)
    drill_id = state["drill_id"]
    while True:
        if not final_snapshot.squeue_ok or not final_snapshot.sacct_ok:
            raise SchedulerAmbiguity("drill finish requires complete scheduler truth")
        try:
            _reconcile_stopping_drill_intents(
                state_dir,
                snapshot=final_snapshot,
                now=final_snapshot.captured_at,
            )
        except SchedulerVisibilityPending:
            if time.monotonic() >= deadline:
                raise
            if poll_seconds > 0:
                time.sleep(poll_seconds)
            final_snapshot = reader()
            continue
        state = load_drill_state(state_dir)
        all_live_drills = _all_live_drill_jobs(final_snapshot)
        malformed = [
            job.job_id
            for job in all_live_drills
            if parse_drill_job_token(job.comment) is None
        ]
        if malformed:
            raise SchedulerAmbiguity(
                "malformed controller-drill jobs remain live during finish: "
                + ", ".join(sorted(malformed))
            )
        foreign = [
            job.job_id
            for job in all_live_drills
            if parse_drill_job_token(job.comment).get("drill") != drill_id
        ]
        if foreign:
            raise SchedulerAmbiguity(
                f"foreign controller-drill jobs remain live during finish: {sorted(foreign)}"
            )
        live = _live_drill_jobs(final_snapshot, drill_id)
        if live:
            clean_snapshots = 0
            recorded: dict[str, dict[str, Any]] = {}
            for row in state["roles"].values():
                for field in ("active", "successor", "submission_intent"):
                    record = row.get(field)
                    if isinstance(record, dict) and record.get("job_id"):
                        recorded[str(record["job_id"])] = record
            for job in live:
                record = recorded.get(job.job_id)
                if (
                    record is None
                    or job.comment != record.get("job_token")
                    or not _command_binds_exact_sbatch(
                        job.command, str(record.get("sbatch_path", ""))
                    )
                    or (
                        record.get("dependency_job_id") is not None
                        and not _dependency_binds_exact_afterany(
                            job.dependency, str(record["dependency_job_id"])
                        )
                    )
                ):
                    raise SchedulerAmbiguity(
                        f"refusing drill cleanup of unmapped exact job {job.job_id}"
                    )
            new_targets = sorted(
                (job for job in live if job.job_id not in cancelled_job_ids),
                key=lambda item: item.job_id,
            )
            if new_targets:
                with drill_lock(state_dir):
                    state = load_drill_state(state_dir)
                    cleanup_ids = set(state.get("cleanup_job_ids", []))
                    cleanup_ids.update(job.job_id for job in new_targets)
                    state["cleanup_job_ids"] = sorted(cleanup_ids)
                    _append_drill_event(
                        state_dir,
                        state,
                        event="cleanup_targets_committed",
                        details={"job_ids": [job.job_id for job in new_targets]},
                        now=timestamp,
                    )
                    _save_drill_state(state_dir, state, now=timestamp)
                for job in new_targets:
                    proc = runner(["scancel", job.job_id])
                    if proc.returncode != 0:
                        raise ControlError(
                            f"failed exact drill cleanup {job.job_id}: "
                            f"{proc.stderr.strip()[:500]}"
                        )
                    cancelled_job_ids.add(job.job_id)
        else:
            clean_snapshots += 1
            # A second independent, complete scheduler read closes the narrow window
            # in which an after-any successor is released as its parent disappears.
            if clean_snapshots >= 2:
                break
        if time.monotonic() >= deadline:
            ids = [job.job_id for job in _live_drill_jobs(final_snapshot, drill_id)]
            raise ControlError(f"controller-drill cleanup timed out for jobs {ids}")
        if poll_seconds > 0:
            time.sleep(poll_seconds)
        final_snapshot = reader()
    control = _assert_drill_control_paused(
        state_dir, expected_immutable_sha256=state["immutable_sha256"]
    )
    current_baseline = controller_drill_baseline(state_dir, control)
    if current_baseline != state["baseline"]:
        raise ControlError(
            "controller drill changed production fairness, ledger, or run data"
        )
    reconciliation = reconcile_control(
        state_dir,
        snapshot=final_snapshot,
        all_jobs=True,
        no_admit=True,
        now=timestamp,
    )
    if not reconciliation["passed"]:
        raise SchedulerAmbiguity(
            "post-drill scheduler reconciliation failed: "
            + "; ".join(reconciliation["errors"])
        )
    reconciliation_path, reconciliation_sha = _publish_drill_reconciliation(
        state_dir, drill_id
    )
    with drill_lock(state_dir):
        state = load_drill_state(state_dir)
        state["phase"] = "completed"
        state["final_reconciliation_path"] = str(reconciliation_path)
        state["final_reconciliation_sha256"] = reconciliation_sha
        state["completed_at"] = utc_timestamp(timestamp)
        state["completed_timestamp"] = timestamp
        _append_drill_event(
            state_dir,
            state,
            event="completed",
            details={
                "final_reconciliation_path": str(reconciliation_path),
                "final_reconciliation_sha256": reconciliation_sha,
            },
            now=timestamp,
        )
        _save_drill_state(state_dir, state, now=timestamp)
        _validate_completed_drill_history(state)
    marker = _completed_drill_marker_payload(state_dir, state)
    # All job cleanup, reconciliation, and state fsyncs precede this marker-last write.
    _atomic_write_json(marker_path, marker)
    return marker


def record_alert(
    state_dir: Path,
    *,
    kind: str,
    severity: str,
    message: str,
    dedupe_key: str,
    send_email: bool = False,
    mail_runner: (
        Callable[[Sequence[str], str], subprocess.CompletedProcess[str]] | None
    ) = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Persist a deduplicated alert before attempting best-effort email delivery."""
    if severity not in {"warning", "critical"}:
        raise ControlError("alert severity must be warning or critical")
    if not kind or not message or not dedupe_key:
        raise ControlError("alert kind, message, and dedupe key must be non-empty")
    timestamp = time.time() if now is None else float(now)
    with control_lock(state_dir):
        control = load_control(state_dir)
        active = next(
            (
                alert
                for alert in reversed(control["alerts"])
                if alert.get("dedupe_key") == dedupe_key
                and alert.get("resolved_at") is None
            ),
            None,
        )
        if active is None:
            payload: dict[str, Any] = {
                "alert_id": uuid.uuid4().hex,
                "kind": kind,
                "severity": severity,
                "message": message,
                "dedupe_key": dedupe_key,
                "first_at": utc_timestamp(timestamp),
                "first_timestamp": timestamp,
                "last_at": utc_timestamp(timestamp),
                "last_timestamp": timestamp,
                "occurrences": 1,
                "resolved_at": None,
                "email": {"attempted": False, "delivered": False},
            }
            control["alerts"].append(payload)
        else:
            active["occurrences"] = int(active.get("occurrences", 1)) + 1
            active["last_at"] = utc_timestamp(timestamp)
            active["last_timestamp"] = timestamp
            active["message"] = message
            payload = active
        journal_record = {
            "at": utc_timestamp(timestamp),
            "timestamp": timestamp,
            "action": "raised",
            "alert_id": payload["alert_id"],
            "kind": kind,
            "severity": severity,
            "message": message,
            "dedupe_key": dedupe_key,
            "occurrences": payload["occurrences"],
        }
        _append_jsonl(state_dir / ALERT_JOURNAL, journal_record)
        append_transition(
            state_dir,
            control,
            event="alert_raised",
            details={
                "alert_id": payload["alert_id"],
                "kind": kind,
                "severity": severity,
                "dedupe_key": dedupe_key,
            },
            now=timestamp,
        )
        _save_control(state_dir, control, now=timestamp)
        alert_id = str(payload["alert_id"])
        email = str(control["alert_email"])

    if send_email:
        subject = f"[agents-scaling:{severity}] {kind}"
        body = (
            f"Schema-5 sweep alert\n\nKind: {kind}\nSeverity: {severity}\n"
            f"Time: {utc_timestamp(timestamp)}\nState: {state_dir}\n\n{message}\n"
        )
        delivered = False
        error: str | None = None
        if mail_runner is not None:
            proc = mail_runner(["mail", "-s", subject, email], body)
            delivered = proc.returncode == 0
            error = None if delivered else proc.stderr.strip()[:500]
        else:
            try:
                proc = subprocess.run(
                    ["mail", "-s", subject, email],
                    input=body,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                delivered = proc.returncode == 0
                error = None if delivered else proc.stderr.strip()[:500]
            except OSError as exc:
                error = str(exc)
        with control_lock(state_dir):
            control = load_control(state_dir)
            match = next(
                alert for alert in control["alerts"] if alert["alert_id"] == alert_id
            )
            match["email"] = {
                "attempted": True,
                "delivered": delivered,
                "attempted_at": utc_timestamp(),
                "error": error,
            }
            _save_control(state_dir, control, now=time.time())
    return next(
        alert
        for alert in load_control(state_dir)["alerts"]
        if alert["alert_id"] == alert_id
    )


def resolve_alert(
    state_dir: Path, *, dedupe_key: str, now: float | None = None
) -> dict[str, Any]:
    timestamp = time.time() if now is None else float(now)
    with control_lock(state_dir):
        control = load_control(state_dir)
        matches = [
            alert
            for alert in control["alerts"]
            if alert.get("dedupe_key") == dedupe_key
            and alert.get("resolved_at") is None
        ]
        for alert in matches:
            alert["resolved_at"] = utc_timestamp(timestamp)
            alert["resolved_timestamp"] = timestamp
            _append_jsonl(
                state_dir / ALERT_JOURNAL,
                {
                    "at": utc_timestamp(timestamp),
                    "timestamp": timestamp,
                    "action": "resolved",
                    "alert_id": alert["alert_id"],
                    "dedupe_key": dedupe_key,
                },
            )
        if matches:
            append_transition(
                state_dir,
                control,
                event="alert_resolved",
                details={"dedupe_key": dedupe_key, "count": len(matches)},
                now=timestamp,
            )
            _save_control(state_dir, control, now=timestamp)
        return control


def _heartbeat_age(record: Any, now: float) -> float | None:
    if not isinstance(record, dict) or not isinstance(
        record.get("timestamp"), (int, float)
    ):
        return None
    return max(0.0, now - float(record["timestamp"]))


def controller_liveness_age(role_state: Mapping[str, Any], now: float) -> float | None:
    """Age a role from heartbeat, falling back to its most recent durable submission."""
    heartbeat_age = _heartbeat_age(role_state.get("heartbeat"), now)
    if heartbeat_age is not None:
        return heartbeat_age
    candidates: list[float] = []
    for field in ("active", "successor", "submission_intent"):
        record = role_state.get(field)
        if not isinstance(record, dict):
            continue
        for timestamp_field in (
            "claimed_timestamp",
            "submitted_timestamp",
            "created_timestamp",
        ):
            value = record.get(timestamp_field)
            if isinstance(value, (int, float)):
                candidates.append(float(value))
                break
    return max(0.0, now - max(candidates)) if candidates else None


def _cell_job(name: str) -> bool:
    return name == "asys-cells" or any(
        name.startswith(prefix) for prefix in CELL_JOB_PREFIXES
    )


def live_status(
    state_dir: Path,
    *,
    snapshot: SchedulerSnapshot,
    now: float | None = None,
) -> dict[str, Any]:
    """Return scheduler-authoritative live status without mutating cached state."""
    timestamp = snapshot.captured_at if now is None else float(now)
    control = load_control(state_dir)
    jobs_by_id = {job.job_id: job for job in snapshot.jobs}
    roles: dict[str, Any] = {}
    for role in ROLE_NAMES:
        state = control["controllers"][role]
        active = state.get("active") if isinstance(state.get("active"), dict) else {}
        successor = (
            state.get("successor") if isinstance(state.get("successor"), dict) else {}
        )
        active_job = jobs_by_id.get(str(active.get("job_id", "")))
        successor_job = jobs_by_id.get(str(successor.get("job_id", "")))
        age = controller_liveness_age(state, timestamp)
        roles[role] = {
            "recorded_active_job_id": active.get("job_id"),
            "live_active": bool(active_job and active_job.active),
            "scheduler_state": active_job.state if active_job else None,
            "recorded_successor_job_id": successor.get("job_id"),
            "successor_live": bool(successor_job and successor_job.active),
            "successor_scheduler_state": successor_job.state if successor_job else None,
            "heartbeat": state.get("heartbeat"),
            "heartbeat_age_seconds": age,
            "heartbeat_stale": age is None or age > STALE_HEARTBEAT_SECONDS,
        }
    active_cell_jobs = [
        job for job in snapshot.jobs if job.active and _cell_job(job.job_name)
    ]
    ledger_path = state_dir / "ledger.json"
    cached: dict[str, Any] = {"available": False, "path": str(ledger_path)}
    if ledger_path.is_file():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            cached = {
                "available": True,
                "path": str(ledger_path),
                "updated_at": ledger.get("updated_at"),
                "poll_number": ledger.get("poll_number"),
                "cached_job_records": len(ledger.get("jobs", {})),
                "cached_cell_records": len(ledger.get("cells", {})),
                "cached_state_only": True,
            }
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            cached = {"available": False, "path": str(ledger_path), "error": str(exc)}
    active_alerts = [
        alert for alert in control["alerts"] if alert.get("resolved_at") is None
    ]
    controllers_healthy = control["desired_state"] == "paused" or all(
        not role["heartbeat_stale"] and role["live_active"] for role in roles.values()
    )
    healthy = (
        snapshot.squeue_ok
        and snapshot.sacct_ok
        and not snapshot.errors
        and controllers_healthy
        and not any(alert.get("severity") == "critical" for alert in active_alerts)
    )
    return {
        "schema_version": 1,
        "captured_at": utc_timestamp(timestamp),
        "desired_state": control["desired_state"],
        "drain_requested": control["drain_requested"],
        "rollout_generation": control["rollout_generation"],
        "admission": copy.deepcopy(control["admission"]),
        "admission_ramp": copy.deepcopy(control["admission_ramp"]),
        "healthy": healthy,
        "scheduler": {
            "squeue_ok": snapshot.squeue_ok,
            "sacct_ok": snapshot.sacct_ok,
            "errors": list(snapshot.errors),
            "total_jobs": len(snapshot.jobs),
            # This is the only field named active_cell_tasks/jobs: it comes directly
            # from scheduler truth, never the potentially stale dispatcher ledger.
            "active_cell_tasks": len(active_cell_jobs),
            "active_cell_job_ids": [job.job_id for job in active_cell_jobs],
        },
        "controllers": roles,
        "monitoring": {
            "owner_role": control["monitoring"]["owner_role"],
            "cadences": {
                cadence: {
                    **copy.deepcopy(row),
                    "overdue": (
                        control["desired_state"] == "running"
                        and row["active_attempt"] is None
                        and float(row["next_due_timestamp"]) <= timestamp
                    ),
                }
                for cadence, row in control["monitoring"]["cadences"].items()
            },
        },
        "dispatcher_cache": cached,
        "readiness": copy.deepcopy(control["readiness"]),
        "open_throughput_epoch": (
            copy.deepcopy(control["throughput_epochs"][-1])
            if control["throughput_epochs"]
            and control["throughput_epochs"][-1].get("closed_at") is None
            else None
        ),
        "active_alerts": copy.deepcopy(active_alerts),
    }


def _role_has_live_chain(
    control: Mapping[str, Any], role: str, snapshot: SchedulerSnapshot
) -> bool:
    active_ids = snapshot.active_job_ids
    state = control["controllers"][role]
    for field in ("active", "successor", "submission_intent"):
        record = state.get(field)
        if isinstance(record, dict) and str(record.get("job_id", "")) in active_ids:
            return True
        if isinstance(record, dict) and record.get("state") == "submitting":
            matches = _find_intent_jobs(snapshot, str(record.get("job_token", "")))
            if any(job.active for job in matches):
                return True
    return False


def trigger_stale_takeover(
    state_dir: Path,
    *,
    role: str,
    snapshot: SchedulerSnapshot,
    cancel_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    submit_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    stale_seconds: float = STALE_HEARTBEAT_SECONDS,
    now: float | None = None,
) -> dict[str, Any]:
    """Fence a >10-minute stale exact job and release or create its one successor."""
    _validate_role(role)
    timestamp = snapshot.captured_at if now is None else float(now)
    control = load_control(state_dir)
    if control["desired_state"] != "running":
        return {"role": role, "action": "disabled_while_paused"}
    state = control["controllers"][role]
    age = controller_liveness_age(state, timestamp)
    if age is not None and age <= stale_seconds:
        return {"role": role, "action": "heartbeat_fresh", "age_seconds": age}
    active = state.get("active")
    cancelled: str | None = None
    if isinstance(active, dict) and active.get("job_id"):
        active_id = str(active["job_id"])
        scheduler_job = next(
            (job for job in snapshot.jobs if job.job_id == active_id and job.active),
            None,
        )
        if scheduler_job is not None:
            if scheduler_job.token != active.get("job_token"):
                raise SchedulerAmbiguity(
                    f"refusing stale takeover of {active_id}: exact scheduler token mismatch"
                )
            invoke = cancel_runner or _run_subprocess
            proc = invoke(["scancel", active_id])
            if proc.returncode != 0:
                raise ControlError(
                    f"failed to fence stale exact controller {active_id}: "
                    f"{proc.stderr.strip()[:500]}"
                )
            cancelled = active_id
            with control_lock(state_dir):
                updated = load_control(state_dir)
                append_transition(
                    state_dir,
                    updated,
                    event="stale_controller_fenced",
                    details={"role": role, "job_id": active_id, "age_seconds": age},
                    now=timestamp,
                )
                _save_control(state_dir, updated, now=timestamp)

    # Exclude the just-fenced allocation from the scheduler view.  A recorded pending
    # successor is already the takeover and its afterany dependency will now release.
    remaining = SchedulerSnapshot(
        tuple(job for job in snapshot.jobs if job.job_id != cancelled),
        snapshot.captured_at,
        snapshot.squeue_ok,
        snapshot.sacct_ok,
        snapshot.errors,
    )
    refreshed = load_control(state_dir)
    successor = refreshed["controllers"][role].get("successor")
    if isinstance(successor, dict) and any(
        job.job_id == str(successor.get("job_id")) and job.active
        for job in remaining.jobs
    ):
        return {
            "role": role,
            "action": "released_recorded_successor",
            "cancelled_job_id": cancelled,
            "successor_job_id": successor.get("job_id"),
        }
    result = submit_controller_intent(
        state_dir,
        role=role,
        target="active",
        dependency_job_id=None,
        scheduler=remaining,
        submit_runner=submit_runner,
        now=timestamp,
    )
    return {
        "role": role,
        "action": "submitted_takeover",
        "cancelled_job_id": cancelled,
        "successor_job_id": result["job_id"],
    }


def repair_chains(
    state_dir: Path,
    *,
    snapshot: SchedulerSnapshot,
    submit_runner: (
        Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None
    ) = None,
    only_roles: Sequence[str] = ROLE_NAMES,
    now: float | None = None,
) -> dict[str, Any]:
    """Idempotently ensure each requested role has one uniquely mapped live chain."""
    for role in only_roles:
        _validate_role(role)
    control = load_control(state_dir, verify_files=True)
    if control["desired_state"] != "running":
        return {"desired_state": "paused", "submitted": [], "existing": []}
    validate_readiness(control, state_dir=state_dir, verify_files=True)
    provisional = build_reconciliation_report(
        control, snapshot, all_jobs=True, no_admit=True
    )
    if provisional["errors"]:
        raise SchedulerAmbiguity(
            "repair-chain refused ambiguous scheduler state: "
            + "; ".join(provisional["errors"])
        )
    submitted: list[dict[str, Any]] = []
    existing: list[str] = []
    for role in only_roles:
        # Reload between roles because each submission commits a new durable identity.
        current = load_control(state_dir)
        if _role_has_live_chain(current, role, snapshot):
            existing.append(role)
            continue
        record = submit_controller_intent(
            state_dir,
            role=role,
            target="active",
            dependency_job_id=None,
            scheduler=snapshot,
            submit_runner=submit_runner,
            now=now,
        )
        submitted.append({"role": role, "job_id": record["job_id"]})
    return {"desired_state": "running", "submitted": submitted, "existing": existing}


def claim_controller(
    state_dir: Path,
    *,
    role: str,
    generation: int,
    intent_token: str,
    job_id: str,
    active_scheduler_job_ids: set[str],
    now: float | None = None,
) -> dict[str, Any]:
    """Fence a starting Slurm job to its exact durable generation and intent."""
    _validate_role(role)
    if not job_id.isdigit():
        raise ControllerFenced(f"invalid SLURM_JOB_ID {job_id!r}")
    timestamp = time.time() if now is None else float(now)
    expected_token = job_token(role, generation, intent_token)
    with control_lock(state_dir):
        control = load_control(state_dir)
        if control["desired_state"] not in {"resuming", "running"}:
            raise ControllerFenced(
                "desired_state=paused fences all starting controllers"
            )
        if control["desired_state"] == "resuming":
            resume_intent = control.get("resume_intent")
            if (
                not isinstance(resume_intent, dict)
                or resume_intent.get("state") != "submitting_controllers"
                or int(resume_intent.get("rollout_generation", -1))
                != int(control["rollout_generation"])
            ):
                raise ControllerFenced(
                    "controller is not part of the active resume transaction"
                )
        state = control["controllers"][role]
        claimed = _resolve_fenced_role_claim(
            state,
            role=role,
            expected_token=expected_token,
            generation=generation,
            intent_token=intent_token,
            job_id=job_id,
            active_scheduler_job_ids=active_scheduler_job_ids,
            context="production",
            now=timestamp,
        )
        state["active"] = claimed
        if isinstance(state.get("successor"), dict) and (
            state["successor"].get("intent_token") == intent_token
        ):
            state["successor"] = None
        state["heartbeat"] = {
            "job_id": job_id,
            "generation": generation,
            "intent_token": intent_token,
            "at": utc_timestamp(timestamp),
            "timestamp": timestamp,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        append_transition(
            state_dir,
            control,
            event="controller_claimed",
            details={"role": role, "generation": generation, "job_id": job_id},
            now=timestamp,
        )
        _save_control(state_dir, control, now=timestamp)
        return claimed


def heartbeat_controller(
    state_dir: Path,
    *,
    role: str,
    generation: int,
    intent_token: str,
    job_id: str,
    now: float | None = None,
) -> dict[str, Any]:
    timestamp = time.time() if now is None else float(now)
    with control_lock(state_dir):
        control = load_control(state_dir)
        state = control["controllers"][role]
        active = state.get("active")
        identity = (job_id, generation, intent_token)
        recorded = (
            (
                str(active.get("job_id", "")),
                int(active.get("generation", -1)),
                str(active.get("intent_token", "")),
            )
            if isinstance(active, dict)
            else None
        )
        if recorded != identity:
            raise ControllerFenced(
                f"heartbeat identity {identity!r} no longer owns {role}; recorded={recorded!r}"
            )
        heartbeat = {
            "job_id": job_id,
            "generation": generation,
            "intent_token": intent_token,
            "at": utc_timestamp(timestamp),
            "timestamp": timestamp,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        state["heartbeat"] = heartbeat
        _save_control(state_dir, control, now=timestamp)
        return heartbeat


def record_controller_exit(
    state_dir: Path,
    *,
    role: str,
    generation: int,
    intent_token: str,
    job_id: str,
    returncode: int,
    reason: str,
    now: float | None = None,
) -> None:
    timestamp = time.time() if now is None else float(now)
    with control_lock(state_dir):
        control = load_control(state_dir)
        state = control["controllers"][role]
        active = state.get("active")
        if isinstance(active, dict) and (
            str(active.get("job_id")) == job_id
            and int(active.get("generation", -1)) == generation
            and str(active.get("intent_token")) == intent_token
        ):
            state["active"] = None
        state["last_exit"] = {
            "job_id": job_id,
            "generation": generation,
            "intent_token": intent_token,
            "returncode": returncode,
            "reason": reason,
            "at": utc_timestamp(timestamp),
            "timestamp": timestamp,
        }
        append_transition(
            state_dir,
            control,
            event="controller_exited",
            details={
                "role": role,
                "generation": generation,
                "job_id": job_id,
                "returncode": returncode,
                "reason": reason,
            },
            now=timestamp,
        )
        _save_control(state_dir, control, now=timestamp)


def _ensure_own_successor(
    state_dir: Path,
    *,
    role: str,
    job_id: str,
    snapshot: SchedulerSnapshot,
) -> dict[str, Any] | None:
    control = load_control(state_dir)
    if control["desired_state"] != "running":
        return None
    successor = control["controllers"][role].get("successor")
    if isinstance(successor, dict) and successor.get("job_id"):
        if any(
            job.job_id == str(successor["job_id"]) and job.active
            for job in snapshot.jobs
        ):
            return successor
    return submit_controller_intent(
        state_dir,
        role=role,
        target="successor",
        dependency_job_id=job_id,
        scheduler=snapshot,
    )


def _signal_process_group(child: subprocess.Popen[Any], sig: signal.Signals) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, sig)
    except ProcessLookupError:
        return


@dataclass
class _ManagedMonitorProcess:
    cadence: str
    attempt_id: str
    process: subprocess.Popen[Any]
    output: TextIO


def _validate_monitor_cadence(cadence: str) -> None:
    if cadence not in MONITOR_CADENCE_SECONDS:
        raise ControlError(
            f"unknown monitor cadence {cadence!r}; expected "
            f"{tuple(MONITOR_CADENCE_SECONDS)}"
        )


def schema5_monitor_command(
    state_dir: Path,
    control: Mapping[str, Any],
    *,
    cadence: str,
) -> list[str]:
    """Build the exact frozen monitor argv used by the dispatcher supervisor."""

    _validate_monitor_cadence(cadence)
    immutable = control["immutable"]
    results_root = Path(str(immutable["results_root"])).expanduser().resolve()
    resolved_state_dir = state_dir.expanduser().resolve()
    expected_state_dir = results_root / ".dispatcher-schema5-v1"
    if resolved_state_dir != expected_state_dir:
        raise ImmutablePinError(
            "monitor state directory must be the pinned results-root control directory: "
            f"expected {expected_state_dir}, found {resolved_state_dir}"
        )
    release_worktree = Path(str(immutable["release_worktree"])).expanduser().resolve()
    python = (
        Path(str(immutable["harness_environment_prefix"])).expanduser().resolve()
        / "bin"
        / "python"
    )
    script = release_worktree / "scripts" / "schema5_monitor.py"
    config = release_worktree / "configs" / "schema5_monitoring.v1.json"
    for path, description in (
        (python, "immutable harness Python"),
        (script, "frozen schema-5 monitor"),
        (config, "frozen schema-5 monitoring contract"),
    ):
        if not path.is_file():
            raise ImmutablePinError(f"{description} is missing: {path}")
    command = [
        str(python),
        "-u",
        str(script),
        "--cadence",
        cadence,
        "--config",
        str(config),
        "--results-root",
        str(results_root),
        "--state-dir",
        str(resolved_state_dir),
        "--persist",
        "--send-email",
    ]
    if cadence == "health":
        command.append("--probe-endpoints")
    return command


def _next_aligned_deadline(
    *, scheduled_for: float, interval: float, now: float
) -> float:
    deadline = float(scheduled_for) + float(interval)
    if deadline <= now:
        skipped = int((now - deadline) // interval) + 1
        deadline += skipped * interval
    return deadline


def monitor_due_cadences(control: Mapping[str, Any], *, now: float) -> tuple[str, ...]:
    """Return due, non-overlapping cadences from durable control state."""

    if control.get("desired_state") != "running":
        return ()
    rows = control["monitoring"]["cadences"]
    return tuple(
        cadence
        for cadence in MONITOR_CADENCE_SECONDS
        if rows[cadence]["active_attempt"] is None
        and float(rows[cadence]["next_due_timestamp"]) <= now
    )


def begin_monitor_attempt(
    state_dir: Path,
    *,
    cadence: str,
    controller_generation: int,
    controller_intent_token: str,
    controller_job_id: str,
    now: float | None = None,
) -> dict[str, Any]:
    """Persist a cadence intent before starting its monitor subprocess."""

    _validate_monitor_cadence(cadence)
    timestamp = time.time() if now is None else float(now)
    with control_lock(state_dir):
        control = load_control(state_dir)
        if control["desired_state"] != "running":
            raise ControlError("cannot start a schema-5 monitor while paused")
        row = control["monitoring"]["cadences"][cadence]
        if row["active_attempt"] is not None:
            raise ControlError(
                f"monitor cadence {cadence!r} already has an active attempt"
            )
        if float(row["next_due_timestamp"]) > timestamp:
            raise ControlError(f"monitor cadence {cadence!r} is not due")
        scheduled_for = float(row["next_due_timestamp"])
        attempt_id = uuid.uuid4().hex
        log_path = (
            state_dir
            / "monitoring"
            / "controller_logs"
            / cadence
            / f"{int(timestamp)}.{attempt_id}.log"
        ).resolve()
        attempt = {
            "attempt_id": attempt_id,
            "cadence": cadence,
            "scheduled_for_at": utc_timestamp(scheduled_for),
            "scheduled_for_timestamp": scheduled_for,
            "started_at": utc_timestamp(timestamp),
            "started_timestamp": timestamp,
            "controller_generation": int(controller_generation),
            "controller_intent_token": str(controller_intent_token),
            "controller_job_id": str(controller_job_id),
            "log_path": str(log_path),
        }
        row["active_attempt"] = attempt
        row["last_attempt_at"] = utc_timestamp(timestamp)
        row["last_attempt_timestamp"] = timestamp
        deadline = _next_aligned_deadline(
            scheduled_for=scheduled_for,
            interval=float(row["interval_seconds"]),
            now=timestamp,
        )
        row["next_due_at"] = utc_timestamp(deadline)
        row["next_due_timestamp"] = deadline
        _save_control(state_dir, control, now=timestamp)
        return copy.deepcopy(attempt)


def complete_monitor_attempt(
    state_dir: Path,
    *,
    cadence: str,
    attempt_id: str,
    returncode: int | None,
    error: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Fence and durably finish one monitor attempt.

    Exit code 2 is a successful scan with critical scientific findings; the monitor has
    already persisted and emailed those findings.  Only execution failures are retried
    early and raised as supervisor alerts.
    """

    _validate_monitor_cadence(cadence)
    timestamp = time.time() if now is None else float(now)
    with control_lock(state_dir):
        control = load_control(state_dir)
        row = control["monitoring"]["cadences"][cadence]
        active = row.get("active_attempt")
        if not isinstance(active, dict) or active.get("attempt_id") != attempt_id:
            raise ControllerFenced(
                f"monitor attempt {attempt_id!r} no longer owns cadence {cadence!r}"
            )
        successful = returncode in MONITOR_SUCCESS_RETURN_CODES and error is None
        row["active_attempt"] = None
        row["last_returncode"] = returncode
        if successful:
            row["last_success_at"] = utc_timestamp(timestamp)
            row["last_success_timestamp"] = timestamp
            row["last_error"] = None
            row["consecutive_failures"] = 0
        else:
            message = error or f"monitor exited with code {returncode}"
            row["last_failure_at"] = utc_timestamp(timestamp)
            row["last_failure_timestamp"] = timestamp
            row["last_error"] = message[:2000]
            row["consecutive_failures"] = int(row["consecutive_failures"]) + 1
            retry_at = timestamp + MONITOR_RETRY_SECONDS
            if retry_at < float(row["next_due_timestamp"]):
                row["next_due_at"] = utc_timestamp(retry_at)
                row["next_due_timestamp"] = retry_at
        if float(row["next_due_timestamp"]) <= timestamp:
            deadline = _next_aligned_deadline(
                scheduled_for=float(row["next_due_timestamp"]),
                interval=float(row["interval_seconds"]),
                now=timestamp,
            )
            row["next_due_at"] = utc_timestamp(deadline)
            row["next_due_timestamp"] = deadline
        _save_control(state_dir, control, now=timestamp)
        return copy.deepcopy(row)


def recover_abandoned_monitor_attempts(
    state_dir: Path,
    *,
    controller_generation: int,
    controller_intent_token: str,
    controller_job_id: str,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Clear monitor intents left by a terminated dispatcher controller."""

    timestamp = time.time() if now is None else float(now)
    recovered: list[dict[str, Any]] = []
    with control_lock(state_dir):
        control = load_control(state_dir)
        for cadence, row in control["monitoring"]["cadences"].items():
            active = row.get("active_attempt")
            if not isinstance(active, dict):
                continue
            recovered.append(copy.deepcopy(active))
            row["active_attempt"] = None
            row["last_failure_at"] = utc_timestamp(timestamp)
            row["last_failure_timestamp"] = timestamp
            row["last_error"] = (
                "abandoned monitor attempt recovered by dispatcher "
                f"generation={controller_generation} intent={controller_intent_token} "
                f"job={controller_job_id}"
            )
            row["consecutive_failures"] = int(row["consecutive_failures"]) + 1
            row["next_due_at"] = utc_timestamp(timestamp)
            row["next_due_timestamp"] = timestamp
        if recovered:
            _save_control(state_dir, control, now=timestamp)
    return recovered


def abandon_monitor_attempt(
    state_dir: Path,
    *,
    cadence: str,
    attempt_id: str,
    reason: str,
    now: float | None = None,
) -> None:
    """Clear an orderly controller-drain attempt without treating it as a failure."""

    timestamp = time.time() if now is None else float(now)
    with control_lock(state_dir):
        control = load_control(state_dir)
        row = control["monitoring"]["cadences"][cadence]
        active = row.get("active_attempt")
        if not isinstance(active, dict) or active.get("attempt_id") != attempt_id:
            return
        row["active_attempt"] = None
        row["last_error"] = reason[:2000]
        row["next_due_at"] = utc_timestamp(timestamp)
        row["next_due_timestamp"] = timestamp
        _save_control(state_dir, control, now=timestamp)


def _monitor_failure_alert(
    state_dir: Path, *, cadence: str, message: str, now: float
) -> None:
    """Best-effort deduplicated email path; never propagate into supervision."""

    dedupe_key = f"{MONITOR_FAILURE_ALERT_PREFIX}:{cadence}"
    try:
        active = any(
            alert.get("dedupe_key") == dedupe_key and alert.get("resolved_at") is None
            for alert in load_control(state_dir)["alerts"]
        )
        record_alert(
            state_dir,
            kind="monitor-execution-failure",
            severity="critical",
            message=message,
            dedupe_key=dedupe_key,
            send_email=not active,
            now=now,
        )
    except Exception as exc:
        print(
            f"[schema5-control] could not persist {cadence} monitor alert: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _start_monitor_process(
    state_dir: Path,
    *,
    cadence: str,
    controller_generation: int,
    controller_intent_token: str,
    controller_job_id: str,
    now: float,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> _ManagedMonitorProcess | None:
    attempt: dict[str, Any] | None = None
    output: TextIO | None = None
    try:
        attempt = begin_monitor_attempt(
            state_dir,
            cadence=cadence,
            controller_generation=controller_generation,
            controller_intent_token=controller_intent_token,
            controller_job_id=controller_job_id,
            now=now,
        )
        # Re-verify the frozen release and environment/model/fleet/run pins at every
        # monitor launch, not merely when this 12-hour controller generation began.
        control = load_control(state_dir, verify_files=True)
        command = schema5_monitor_command(state_dir, control, cadence=cadence)
        log_path = Path(str(attempt["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output = log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(production_environment(control))
        environment["ASYS_SCHEMA5_CONTROL"] = str(_state_path(state_dir).resolve())
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process = popen_factory(
            command,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return _ManagedMonitorProcess(
            cadence=cadence,
            attempt_id=str(attempt["attempt_id"]),
            process=process,
            output=output,
        )
    except Exception as exc:
        if output is not None:
            output.close()
        message = f"could not start {cadence} monitor: {type(exc).__name__}: {exc}"
        if attempt is not None:
            try:
                complete_monitor_attempt(
                    state_dir,
                    cadence=cadence,
                    attempt_id=str(attempt["attempt_id"]),
                    returncode=None,
                    error=message,
                    now=now,
                )
            except Exception as completion_exc:
                message += f"; completion persistence failed: {completion_exc}"
        _monitor_failure_alert(state_dir, cadence=cadence, message=message, now=now)
        print(f"[schema5-control] {message}", file=sys.stderr, flush=True)
        return None


def _service_schema5_monitors(
    state_dir: Path,
    *,
    processes: MutableMapping[str, _ManagedMonitorProcess],
    controller_generation: int,
    controller_intent_token: str,
    controller_job_id: str,
    now: float,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> None:
    """Poll and launch monitors; all per-cadence errors are controller-safe."""

    for cadence, managed in list(processes.items()):
        try:
            returncode = managed.process.poll()
            if returncode is None:
                continue
            managed.output.close()
            row = complete_monitor_attempt(
                state_dir,
                cadence=cadence,
                attempt_id=managed.attempt_id,
                returncode=int(returncode),
                now=now,
            )
            processes.pop(cadence, None)
            if returncode in MONITOR_SUCCESS_RETURN_CODES:
                resolve_alert(
                    state_dir,
                    dedupe_key=f"{MONITOR_FAILURE_ALERT_PREFIX}:{cadence}",
                    now=now,
                )
            else:
                _monitor_failure_alert(
                    state_dir,
                    cadence=cadence,
                    message=(
                        f"{cadence} monitor exited with code {returncode}; "
                        f"log={managed.output.name}; failures={row['consecutive_failures']}"
                    ),
                    now=now,
                )
        except Exception as exc:
            # One broken cadence must neither stop admission nor suppress the other
            # monitors.  Keep an unpolled process tracked; a completed but unrecordable
            # process is retried/recovered by the next controller.
            _monitor_failure_alert(
                state_dir,
                cadence=cadence,
                message=f"could not service {cadence} monitor: {type(exc).__name__}: {exc}",
                now=now,
            )
            print(
                f"[schema5-control] monitor service error ({cadence}): {exc}",
                file=sys.stderr,
                flush=True,
            )

    try:
        current = load_control(state_dir)
        due = monitor_due_cadences(current, now=now)
    except Exception as exc:
        _monitor_failure_alert(
            state_dir,
            cadence="health",
            message=f"could not load monitor cadence state: {type(exc).__name__}: {exc}",
            now=now,
        )
        return
    for cadence in due:
        if cadence in processes:
            continue
        # Daily already performs the complete semantic scan.  Keep those two heavy
        # scans serialized while allowing the lightweight five-minute health probe to
        # continue independently.
        if cadence in {"semantic", "daily"} and any(
            active in processes for active in ("semantic", "daily")
        ):
            continue
        managed = _start_monitor_process(
            state_dir,
            cadence=cadence,
            controller_generation=controller_generation,
            controller_intent_token=controller_intent_token,
            controller_job_id=controller_job_id,
            now=now,
            popen_factory=popen_factory,
        )
        if managed is not None:
            processes[cadence] = managed


def _stop_schema5_monitors(
    state_dir: Path,
    processes: MutableMapping[str, _ManagedMonitorProcess],
    *,
    reason: str,
) -> None:
    """Terminate controller-owned monitors and leave every cadence immediately due."""

    for cadence, managed in list(processes.items()):
        try:
            _signal_process_group(managed.process, signal.SIGTERM)
            try:
                managed.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                _signal_process_group(managed.process, signal.SIGKILL)
        except Exception as exc:
            print(
                f"[schema5-control] could not stop {cadence} monitor: {exc}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            managed.output.close()
            try:
                abandon_monitor_attempt(
                    state_dir,
                    cadence=cadence,
                    attempt_id=managed.attempt_id,
                    reason=reason,
                )
            except Exception as exc:
                print(
                    f"[schema5-control] could not abandon {cadence} monitor: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            processes.pop(cadence, None)


def supervise(
    state_dir: Path,
    *,
    role: str,
    generation: int,
    intent_token: str,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
    stale_seconds: float = STALE_HEARTBEAT_SECONDS,
    drain_grace_seconds: float = 90.0,
) -> int:
    """Run one fenced controller process and continuously supervise its peer role."""
    _validate_role(role)
    if heartbeat_interval <= 0 or stale_seconds < heartbeat_interval:
        raise ControlError("invalid controller heartbeat/staleness intervals")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id:
        raise ControllerFenced("supervise must run inside a Slurm allocation")
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    child: subprocess.Popen[Any] | None = None
    runtime_refresh: _RuntimeLeaseRefreshTask | None = None
    snapshot_refresh: _SnapshotLeaseRefreshTask | None = None
    monitor_processes: dict[str, _ManagedMonitorProcess] = {}
    exit_reason = "supervisor_error"
    returncode = 1
    try:
        with role_singleton_lock(state_dir, role):
            initial_snapshot = query_scheduler(tolerate_errors=True)
            claim_controller(
                state_dir,
                role=role,
                generation=generation,
                intent_token=intent_token,
                job_id=job_id,
                active_scheduler_job_ids=initial_snapshot.active_job_ids | {job_id},
            )
            # A resume-submitted job may begin before its peer is visible.  Claim the
            # exact durable identity so no duplicate can take the role, but do not
            # start either managed plane until the atomic resume commit publishes
            # ``running``.  This is what makes unheld submission both restartable and
            # safe from one-controller admission.
            wait_heartbeat = 0.0
            while True:
                waiting_control = load_control(state_dir)
                if waiting_control["desired_state"] == "running":
                    break
                if stop_requested or waiting_control["desired_state"] == "paused":
                    returncode = 0
                    exit_reason = (
                        "desired_state_paused"
                        if waiting_control["desired_state"] == "paused"
                        else "signal"
                    )
                    return returncode
                wait_now = time.time()
                if wait_now - wait_heartbeat >= heartbeat_interval:
                    heartbeat_controller(
                        state_dir,
                        role=role,
                        generation=generation,
                        intent_token=intent_token,
                        job_id=job_id,
                        now=wait_now,
                    )
                    # Scheduler visibility may legitimately take longer than either
                    # seven-minute integrity lease.  Renew and verify both exact
                    # generation proofs on every waiting heartbeat; the snapshot scan
                    # runs in a daemon task while the waiter continues fenced
                    # heartbeats, so liveness does not weaken the drift boundary.
                    runtime_refresh = _start_runtime_lease_refresh(
                        state_dir, waiting_control
                    )
                    _await_runtime_lease_refresh_with_heartbeats(
                        runtime_refresh,
                        state_dir=state_dir,
                        role=role,
                        generation=generation,
                        intent_token=intent_token,
                        job_id=job_id,
                        heartbeat_interval=heartbeat_interval,
                    )
                    runtime_refresh = None
                    validate_runtime_integrity_attestation(
                        waiting_control, verify_metadata=True
                    )
                    snapshot_refresh = _start_snapshot_lease_refresh(
                        state_dir, waiting_control
                    )
                    _await_snapshot_lease_refresh_with_heartbeats(
                        snapshot_refresh,
                        state_dir=state_dir,
                        role=role,
                        generation=generation,
                        intent_token=intent_token,
                        job_id=job_id,
                        heartbeat_interval=heartbeat_interval,
                    )
                    snapshot_refresh = None
                    validate_snapshot_integrity_attestation(
                        waiting_control,
                        state_dir=state_dir,
                        verify_lease=True,
                    )
                    wait_heartbeat = time.time()
                time.sleep(min(1.0, heartbeat_interval))
            # Queue exactly one afterany successor before starting the managed plane.  A
            # transient submission failure is retried on every heartbeat below.
            try:
                _ensure_own_successor(
                    state_dir, role=role, job_id=job_id, snapshot=initial_snapshot
                )
            except (ControlError, OSError) as exc:
                print(
                    f"[schema5-control] successor retry required: {exc}",
                    file=sys.stderr,
                )

            control = load_control(state_dir, verify_files=True)
            # This is the last gate before either managed plane executes.  One
            # controller refreshes the centralized metadata lease when due; all other
            # starts validate that small, generation-bound lease in O(1).
            heartbeat_controller(
                state_dir,
                role=role,
                generation=generation,
                intent_token=intent_token,
                job_id=job_id,
                now=time.time(),
            )
            runtime_refresh = _start_runtime_lease_refresh(state_dir, control)
            _await_runtime_lease_refresh_with_heartbeats(
                runtime_refresh,
                state_dir=state_dir,
                role=role,
                generation=generation,
                intent_token=intent_token,
                job_id=job_id,
                heartbeat_interval=heartbeat_interval,
            )
            runtime_refresh = None
            validate_runtime_integrity_attestation(control, verify_metadata=True)
            snapshot_refresh = _start_snapshot_lease_refresh(state_dir, control)
            _await_snapshot_lease_refresh_with_heartbeats(
                snapshot_refresh,
                state_dir=state_dir,
                role=role,
                generation=generation,
                intent_token=intent_token,
                job_id=job_id,
                heartbeat_interval=heartbeat_interval,
            )
            snapshot_refresh = None
            validate_snapshot_integrity_attestation(
                control, state_dir=state_dir, verify_lease=True
            )
            command_field = (
                "dispatcher_command"
                if role == "dispatcher"
                else "fleet_supervisor_command"
            )
            command = list(control["immutable"][command_field])
            environment = os.environ.copy()
            environment.update(production_environment(control))
            environment["ASYS_SCHEMA5_CONTROL"] = str(_state_path(state_dir).resolve())
            child = subprocess.Popen(command, env=environment, start_new_session=True)
            if role == MONITOR_OWNER_ROLE:
                try:
                    recovered = recover_abandoned_monitor_attempts(
                        state_dir,
                        controller_generation=generation,
                        controller_intent_token=intent_token,
                        controller_job_id=job_id,
                    )
                except Exception as exc:
                    recovered = []
                    _monitor_failure_alert(
                        state_dir,
                        cadence="health",
                        message=(
                            "could not reconcile restart-safe monitor intents: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        now=time.time(),
                    )
                    print(
                        f"[schema5-control] monitor recovery failed safely: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                for attempt in recovered:
                    cadence = str(attempt.get("cadence", "health"))
                    _monitor_failure_alert(
                        state_dir,
                        cadence=cadence,
                        message=(
                            "dispatcher successor recovered an abandoned monitor attempt: "
                            f"attempt={attempt.get('attempt_id')} "
                            f"prior_job={attempt.get('controller_job_id')}"
                        ),
                        now=time.time(),
                    )
            last_heartbeat = 0.0
            last_peer_check = 0.0
            drain_started: float | None = None
            while True:
                now = time.time()
                control = load_control(state_dir)
                if runtime_refresh is not None and runtime_refresh.future.done():
                    # The completed renewal either publishes a new immutable lease or
                    # raises here.  On error the prior lease is never extended and its
                    # TTL remains the fail-closed boundary for every controller.
                    runtime_refresh.future.result()
                    runtime_refresh = None
                if snapshot_refresh is not None and snapshot_refresh.future.done():
                    # Surface drift/I/O failures promptly.  The previous lease remains
                    # immutable and expires naturally, so every admission path fails
                    # closed even if a successor also encounters the same outage.
                    snapshot_refresh.future.result()
                    snapshot_refresh = None
                desired_pause = control["desired_state"] == "paused"
                if stop_requested or desired_pause:
                    if drain_started is None:
                        drain_started = now
                        # Managed cell workers receive USR1 independently from Slurm.
                        # Controllers stop admission/supervision on TERM and exit only
                        # after their own cleanup handlers run.
                        _signal_process_group(child, signal.SIGTERM)
                    if child.poll() is not None:
                        returncode = int(child.returncode or 0)
                        exit_reason = (
                            "desired_state_paused" if desired_pause else "signal"
                        )
                        break
                    if now - drain_started >= drain_grace_seconds:
                        _signal_process_group(child, signal.SIGKILL)
                    time.sleep(min(1.0, heartbeat_interval))
                    continue

                polled = child.poll()
                if polled is not None:
                    returncode = int(polled)
                    exit_reason = "managed_process_exited"
                    break
                if now - last_heartbeat >= heartbeat_interval:
                    # Liveness is independent of filesystem latency: publish the exact
                    # fenced heartbeat before any validation or renewal work.  Snapshot
                    # renewal runs in at most one local daemon task; its cross-node lock
                    # and cache still allow only one double metadata scan fleet-wide.
                    heartbeat_now = time.time()
                    heartbeat_controller(
                        state_dir,
                        role=role,
                        generation=generation,
                        intent_token=intent_token,
                        job_id=job_id,
                        now=heartbeat_now,
                    )
                    last_heartbeat = heartbeat_now
                    # This validates only the compact generation lease.  It is
                    # intentionally performed after the fenced heartbeat and before
                    # launching renewal, so an expired proof stops the controller
                    # rather than being papered over by a late metadata scan.
                    validate_runtime_integrity_attestation(
                        control, verify_metadata=True
                    )
                    if runtime_refresh is None:
                        runtime_refresh = _start_runtime_lease_refresh(
                            state_dir, control
                        )
                    # While a slow renewal is in flight, expiry remains the strict
                    # safety bound.  This compact check never scans payload metadata.
                    validate_snapshot_integrity_attestation(
                        control, state_dir=state_dir, verify_lease=True
                    )
                    if snapshot_refresh is None:
                        snapshot_refresh = _start_snapshot_lease_refresh(
                            state_dir, control
                        )
                    scheduler_now = time.time()
                    snapshot = query_scheduler(
                        tolerate_errors=True, now=scheduler_now
                    )
                    try:
                        _ensure_own_successor(
                            state_dir, role=role, job_id=job_id, snapshot=snapshot
                        )
                    except (ControlError, OSError) as exc:
                        print(
                            f"[schema5-control] successor retry required for {role}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                if now - last_peer_check >= heartbeat_interval:
                    last_peer_check = now
                    peer = next(
                        candidate for candidate in ROLE_NAMES if candidate != role
                    )
                    peer_state = control["controllers"][peer]
                    age = controller_liveness_age(peer_state, now)
                    snapshot = query_scheduler(tolerate_errors=True, now=now)
                    has_chain = _role_has_live_chain(control, peer, snapshot)
                    stale = age is None or age > stale_seconds
                    if stale:
                        record_alert(
                            state_dir,
                            kind="stale_controller_heartbeat",
                            severity="critical",
                            message=(
                                f"{peer} has no live chain and heartbeat age is "
                                f"{age if age is not None else 'missing'} seconds; {role} "
                                "is attempting exact-token takeover"
                            ),
                            dedupe_key=f"stale-controller:{peer}",
                            send_email=True,
                            now=now,
                        )
                        if snapshot.squeue_ok and snapshot.sacct_ok:
                            try:
                                trigger_stale_takeover(
                                    state_dir,
                                    role=peer,
                                    snapshot=snapshot,
                                    stale_seconds=stale_seconds,
                                    now=now,
                                )
                            except ControlError as exc:
                                print(
                                    f"[schema5-control] peer repair failed closed: {exc}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                    elif has_chain:
                        resolve_alert(
                            state_dir, dedupe_key=f"stale-controller:{peer}", now=now
                        )
                if role == MONITOR_OWNER_ROLE:
                    _service_schema5_monitors(
                        state_dir,
                        processes=monitor_processes,
                        controller_generation=generation,
                        controller_intent_token=intent_token,
                        controller_job_id=job_id,
                        now=now,
                    )
                time.sleep(min(1.0, heartbeat_interval))
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if child is not None and child.poll() is None:
            _signal_process_group(child, signal.SIGTERM)
        if monitor_processes:
            _stop_schema5_monitors(
                state_dir,
                monitor_processes,
                reason=f"dispatcher supervisor exiting: {exit_reason}",
            )
        try:
            record_controller_exit(
                state_dir,
                role=role,
                generation=generation,
                intent_token=intent_token,
                job_id=job_id,
                returncode=returncode,
                reason=exit_reason,
            )
        except ControlError as exc:
            print(
                f"[schema5-control] could not record controller exit: {exc}",
                file=sys.stderr,
            )
    return returncode


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_nonfinite(value: str) -> Any:
        raise ValueError(f"non-finite JSON number {value!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ControlError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControlError(f"{description} must be a JSON object: {path}")
    return value


def build_immutable_pins(
    *,
    state_dir: Path,
    release_bundle_root: Path,
    hf_home: Path,
) -> dict[str, Any]:
    """Derive the complete closed pin schema from published authority artifacts.

    The release freezer owns code, model, fleet, and environment identity.  The three
    cloned run roots own their manifest/lineage/policy/benchmark identity.  This builder
    joins those two sources without accepting any free-form command, run, weight, count,
    or hash override from the operator, then runs the same deep validator used by
    ``init`` before returning a byte-publishable object.
    """

    raw_state_dir = state_dir.expanduser()
    if raw_state_dir.is_symlink():
        raise ImmutablePinError(
            f"control state directory cannot be a symlink: {raw_state_dir}"
        )
    resolved_state_dir = raw_state_dir.resolve()
    if resolved_state_dir.name != CONTROL_STATE_DIRNAME:
        raise ImmutablePinError(
            f"control state directory must end in {CONTROL_STATE_DIRNAME!r}: "
            f"{resolved_state_dir}"
        )
    results_root = resolved_state_dir.parent
    if results_root.is_symlink() or not results_root.is_dir():
        raise ImmutablePinError(f"results root is missing or symlinked: {results_root}")

    raw_bundle = release_bundle_root.expanduser()
    if raw_bundle.is_symlink() or not raw_bundle.is_dir():
        raise ImmutablePinError(
            f"release bundle root is missing or symlinked: {raw_bundle}"
        )
    bundle = raw_bundle.resolve()
    marker_path = bundle / RELEASE_COMPLETE_FILENAME
    identity_path = bundle / RELEASE_IDENTITY_FILENAME
    for description, path in (
        ("release completion marker", marker_path),
        ("release identity", identity_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise ImmutablePinError(f"{description} is missing or symlinked: {path}")
    marker = _load_json_object(marker_path, description="release completion marker")
    identity = _load_json_object(identity_path, description="release identity")
    fragment = identity.get("control_pin_fragment")
    if not isinstance(fragment, dict) or set(fragment) != _RELEASE_FRAGMENT_FIELDS:
        raise ImmutablePinError(
            "release identity does not contain the exact control pin fragment"
        )
    if (
        marker.get("complete") is not True
        or marker.get("release_id") != fragment.get("release_id")
        or marker.get("release_bundle_id") in {None, ""}
    ):
        raise ImmutablePinError("release marker and control pin fragment disagree")

    raw_hf_home = hf_home.expanduser()
    if raw_hf_home.is_symlink() or not raw_hf_home.is_dir():
        raise ImmutablePinError(
            f"Hugging Face home is missing or symlinked: {raw_hf_home}"
        )
    resolved_hf_home = raw_hf_home.resolve()
    server_pool_root = results_root / "server_pools" / "schema5-v1"
    if server_pool_root.is_symlink() or not server_pool_root.is_dir():
        raise ImmutablePinError(
            f"canonical schema-5 server pool is missing or symlinked: {server_pool_root}"
        )

    runs: list[dict[str, Any]] = []
    for run_id, expected_cells in REQUIRED_RUNS.items():
        run_root = results_root / run_id
        if run_root.is_symlink() or not run_root.is_dir():
            raise ImmutablePinError(
                f"schema-5 run root is missing or symlinked: {run_root}"
            )
        paths = {
            "manifest": run_root / "cells.json",
            "lineage": run_root / "lineage.schema5-v1.json",
            "policy": run_root / "artifact_policy.schema5-v1.json",
            "benchmark_contract": run_root / "benchmark_contracts.v1.json",
        }
        for description, path in paths.items():
            if path.is_symlink() or not path.is_file():
                raise ImmutablePinError(
                    f"run {run_id} {description} is missing or symlinked: {path}"
                )
        hashes = {name: sha256_file(path) for name, path in paths.items()}
        try:
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ImmutablePinError(
                f"cannot parse run manifest {paths['manifest']}: {exc}"
            ) from exc
        if not isinstance(manifest, list) or len(manifest) != expected_cells:
            raise ImmutablePinError(
                f"run {run_id} manifest must contain exactly {expected_cells} cells"
            )
        lineage = _load_json_object(paths["lineage"], description=f"{run_id} lineage")
        policy = _load_json_object(paths["policy"], description=f"{run_id} policy")
        target_artifacts = lineage.get("target_artifacts")
        target_manifest = (
            target_artifacts.get("cells.json")
            if isinstance(target_artifacts, dict)
            else None
        )
        target_benchmarks = (
            target_artifacts.get("benchmark_contracts.v1.json")
            if isinstance(target_artifacts, dict)
            else None
        )
        if (
            lineage.get("schema_version") != 1
            or lineage.get("clone_mode") != "byte_for_byte_scientific_contract"
            or lineage.get("target_run_id") != run_id
            or lineage.get("manifest_cell_count") != expected_cells
            or lineage.get("imported_result_rows") != 0
            or lineage.get("transformations") != []
            or not isinstance(target_artifacts, dict)
            or not isinstance(target_manifest, dict)
            or target_manifest.get("sha256") != hashes["manifest"]
            or not isinstance(target_benchmarks, dict)
            or target_benchmarks.get("sha256") != hashes["benchmark_contract"]
        ):
            raise ImmutablePinError(
                f"run {run_id} lineage does not bind its frozen clone"
            )
        policy_release = policy.get("release")
        policy_environment = policy.get("environment")
        if (
            policy.get("schema_version") != 1
            or policy.get("run_id") != run_id
            or policy.get("authoritative") is not True
            or policy.get("required_artifact_schema_version") != 5
            or policy.get("accepted_manifest_sha256") != hashes["manifest"]
            or policy.get("accepted_benchmark_contracts_sha256")
            != hashes["benchmark_contract"]
            or policy.get("accepted_model_contract_sha256")
            != fragment["model_contract_sha256"]
            or policy.get("legacy_result_import_allowed") is not False
            or not isinstance(policy_release, dict)
            or policy_release.get("release_id") != fragment["release_id"]
            or policy_release.get("git_commit") != fragment["git_commit"]
            or policy_release.get("source_tree_sha256")
            != fragment["source_tree_sha256"]
            or not isinstance(policy_environment, dict)
            or policy_environment.get("harness_sha256")
            != fragment["harness_environment_sha256"]
            or policy_environment.get("serving_sha256")
            != fragment["serving_environment_sha256"]
        ):
            raise ImmutablePinError(
                f"run {run_id} policy does not bind the frozen release"
            )
        runs.append(
            {
                "run_id": run_id,
                "run_root": str(run_root),
                "cell_count": expected_cells,
                "expected_qids": REQUIRED_RUN_QIDS[run_id],
                "weight": RUN_WEIGHT,
                "manifest_path": str(paths["manifest"]),
                "manifest_sha256": hashes["manifest"],
                "lineage_path": str(paths["lineage"]),
                "lineage_sha256": hashes["lineage"],
                "policy_path": str(paths["policy"]),
                "policy_sha256": hashes["policy"],
                "benchmark_contract_path": str(paths["benchmark_contract"]),
                "benchmark_contract_sha256": hashes["benchmark_contract"],
            }
        )

    pins: dict[str, Any] = {
        **copy.deepcopy(fragment),
        "release_bundle_root": str(bundle),
        "release_bundle_id": str(marker["release_bundle_id"]),
        "hf_home": str(resolved_hf_home),
        "results_root": str(results_root),
        "server_pool_root": str(server_pool_root),
        "dispatcher_command": [],
        "fleet_supervisor_command": [],
        "runs": runs,
    }
    pins["dispatcher_command"] = expected_dispatcher_command(pins)
    pins["fleet_supervisor_command"] = expected_fleet_supervisor_command(pins)
    validate_immutable_pins(pins, verify_files=True)
    return pins


def publish_immutable_pins(path: Path, pins: Mapping[str, Any]) -> dict[str, Any]:
    """Publish one deterministic, checksummed, read-only pins document."""

    validate_immutable_pins(pins, verify_files=True)
    raw_path = path.expanduser()
    if raw_path.is_symlink() or raw_path.is_dir():
        raise ImmutablePinError(
            f"pins output is symlinked or not a file target: {raw_path}"
        )
    target = raw_path.resolve()
    payload = json.dumps(dict(pins), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists():
        try:
            existing = target.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ImmutablePinError(
                f"cannot read existing pins output {target}: {exc}"
            ) from exc
        if existing != payload:
            raise ImmutablePinError(
                f"refusing to replace non-identical immutable pins output: {target}"
            )
    else:
        io.atomic_write_text(target, payload)
    target.chmod((target.stat().st_mode & 0o7777) & ~0o222)
    digest = sha256_file(target)
    checksum_path = target.with_name(target.name + ".sha256")
    checksum_payload = f"{digest}  {target.name}\n"
    if checksum_path.is_symlink() or checksum_path.is_dir():
        raise ImmutablePinError(f"pins checksum target is unsafe: {checksum_path}")
    if checksum_path.exists():
        try:
            existing_checksum = checksum_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ImmutablePinError(
                f"cannot read existing pins checksum {checksum_path}: {exc}"
            ) from exc
        if existing_checksum != checksum_payload:
            raise ImmutablePinError(
                f"refusing to replace non-identical pins checksum: {checksum_path}"
            )
    else:
        io.atomic_write_text(checksum_path, checksum_payload)
    checksum_path.chmod((checksum_path.stat().st_mode & 0o7777) & ~0o222)
    return {
        "output": str(target),
        "sha256": digest,
        "checksum": str(checksum_path),
        "immutable_sha256": sha256_value(pins),
        "run_count": len(pins["runs"]),
        "cell_count": sum(int(run["cell_count"]) for run in pins["runs"]),
        "expected_qids": sum(int(run["expected_qids"]) for run in pins["runs"]),
    }


def _scheduler_for_cli(*, tolerate_errors: bool = False) -> SchedulerSnapshot:
    return query_scheduler(tolerate_errors=tolerate_errors)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-pins",
        help="derive and publish the closed immutable pins consumed by init",
    )
    prepare.add_argument("--release-bundle-root", type=Path, required=True)
    prepare.add_argument("--hf-home", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    init = subparsers.add_parser("init", help="initialize an immutable paused control")
    init.add_argument("--pins-json", type=Path, required=True)
    init.add_argument("--alert-email", default="mabdel03@mit.edu")

    reconcile = subparsers.add_parser(
        "reconcile",
        help="join all scheduler truth with control state, without admission",
    )
    reconcile.add_argument("--all", action="store_true", required=True)
    reconcile.add_argument("--no-admit", action="store_true", required=True)

    subparsers.add_parser("resume", help="fail closed, then set desired_state=running")

    pause = subparsers.add_parser("pause", help="pause admission and gracefully drain")
    pause.add_argument("--drain", action="store_true", required=True)

    status = subparsers.add_parser(
        "status", help="report scheduler-authoritative status"
    )
    status.add_argument("--live", action="store_true", required=True)

    repair = subparsers.add_parser(
        "repair-chain", help="idempotently recreate missing controller chains"
    )
    repair.add_argument("--dry-run", action="store_true")

    attest = subparsers.add_parser(
        "attest", help="attach checksummed readiness evidence"
    )
    attest.add_argument("--gate", required=True, choices=REQUIRED_GATES[:-1])
    attest.add_argument("--evidence", type=Path, required=True)
    attest.add_argument("--sha256")

    ceiling = subparsers.add_parser(
        "set-ceiling",
        help="hold or reduce staged admission (monitor evidence alone may raise it)",
    )
    ceiling.add_argument("--value", type=int, required=True, choices=(24, 96, 192, 384))

    poll = subparsers.add_parser(
        "record-poll", help="append one successful generation-scoped throughput sample"
    )
    poll.add_argument("--validated-qids", type=int, required=True)
    poll.add_argument("--fleet-generation", required=True)
    poll.add_argument("--strata-with-throughput", type=int)

    drill = subparsers.add_parser(
        "drill", help="run the paused, no-admission controller kill drill"
    )
    drill.add_argument("action", choices=("start", "status", "kill", "wait", "finish"))
    drill.add_argument("--role", choices=ROLE_NAMES)
    drill.add_argument("--live", action="store_true")
    drill.add_argument("--timeout", type=float, default=900.0)

    worker = subparsers.add_parser("supervise", help=argparse.SUPPRESS)
    worker.add_argument("--role", required=True, choices=ROLE_NAMES)
    worker.add_argument("--generation", type=int, required=True)
    worker.add_argument("--intent-token", required=True)
    worker.add_argument(
        "--heartbeat-interval", type=float, default=HEARTBEAT_INTERVAL_SECONDS
    )
    worker.add_argument("--stale-seconds", type=float, default=STALE_HEARTBEAT_SECONDS)
    worker.add_argument("--drain-grace-seconds", type=float, default=90.0)
    drill_worker = subparsers.add_parser("supervise-drill", help=argparse.SUPPRESS)
    drill_worker.add_argument("--drill-id", required=True)
    drill_worker.add_argument("--role", required=True, choices=ROLE_NAMES)
    drill_worker.add_argument("--generation", type=int, required=True)
    drill_worker.add_argument("--intent-token", required=True)
    drill_worker.add_argument(
        "--heartbeat-interval", type=float, default=HEARTBEAT_INTERVAL_SECONDS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    state_dir = args.state_dir.expanduser().resolve()
    try:
        if args.command == "prepare-pins":
            canonical_pool = state_dir.parent / "server_pools" / "schema5-v1"
            if canonical_pool.is_symlink():
                raise ImmutablePinError(
                    f"canonical schema-5 server pool cannot be a symlink: {canonical_pool}"
                )
            canonical_pool.mkdir(parents=True, exist_ok=True)
            pins = build_immutable_pins(
                state_dir=state_dir,
                release_bundle_root=args.release_bundle_root,
                hf_home=args.hf_home,
            )
            report = publish_immutable_pins(args.output, pins)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "init":
            pins = _load_json_object(
                args.pins_json.expanduser().resolve(), description="pins"
            )
            control = initialize_control(
                state_dir, pins=pins, alert_email=args.alert_email
            )
            print(
                json.dumps(
                    {"initialized": True, "control": control}, indent=2, sort_keys=True
                )
            )
            return 0
        if args.command == "reconcile":
            report = reconcile_control(
                state_dir,
                snapshot=_scheduler_for_cli(),
                all_jobs=args.all,
                no_admit=args.no_admit,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["passed"] else 2
        if args.command == "resume":
            control = resume_control(state_dir)
            snapshot = _scheduler_for_cli()
            chains = repair_chains(state_dir, snapshot=snapshot)
            print(
                json.dumps(
                    {
                        "desired_state": control["desired_state"],
                        "rollout_generation": control["rollout_generation"],
                        "chains": chains,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "pause":
            control = pause_control(
                state_dir,
                drain=args.drain,
                scheduler_reader=lambda: _scheduler_for_cli(tolerate_errors=True),
            )
            print(
                json.dumps(
                    {
                        "desired_state": control["desired_state"],
                        "drain_requested": control["drain_requested"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "status":
            report = live_status(
                state_dir, snapshot=_scheduler_for_cli(tolerate_errors=True)
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["healthy"] else 1
        if args.command == "repair-chain":
            snapshot = _scheduler_for_cli()
            control = load_control(state_dir, verify_files=True)
            provisional = build_reconciliation_report(
                control, snapshot, all_jobs=True, no_admit=True
            )
            if args.dry_run:
                result = {
                    "dry_run": True,
                    "would_submit": [
                        role
                        for role in ROLE_NAMES
                        if control["desired_state"] == "running"
                        and not _role_has_live_chain(control, role, snapshot)
                    ],
                    "reconciliation": provisional,
                }
            else:
                result = repair_chains(state_dir, snapshot=snapshot)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "attest":
            control = attest_gate(
                state_dir,
                gate=args.gate,
                evidence_path=args.evidence,
                expected_sha256=args.sha256,
            )
            print(json.dumps(control["readiness"][args.gate], indent=2, sort_keys=True))
            return 0
        if args.command == "set-ceiling":
            control = set_admission_ceiling(state_dir, ceiling=args.value)
            print(json.dumps(control["admission"], indent=2, sort_keys=True))
            return 0
        if args.command == "record-poll":
            control = record_successful_poll(
                state_dir,
                validated_qids=args.validated_qids,
                fleet_generation=args.fleet_generation,
                strata_with_throughput=args.strata_with_throughput,
            )
            print(
                json.dumps(control["throughput_epochs"][-1], indent=2, sort_keys=True)
            )
            return 0
        if args.command == "drill":
            if args.action in {"kill", "wait"} and args.role is None:
                raise ControlError(f"drill {args.action} requires --role")
            if args.action == "status" and not args.live:
                raise ControlError("drill status requires --live")
            if args.action == "start":
                state = start_controller_drill(
                    state_dir, scheduler=_scheduler_for_cli()
                )
                result = {
                    "drill_id": state["drill_id"],
                    "phase": state["phase"],
                    "job_ids": {
                        role: state["roles"][role]["active"]["job_id"]
                        for role in ROLE_NAMES
                    },
                }
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0
            if args.action == "status":
                result = controller_drill_status(
                    state_dir, snapshot=_scheduler_for_cli(tolerate_errors=True)
                )
                print(json.dumps(result, indent=2, sort_keys=True))
                return (
                    0
                    if result["healthy"]
                    and (result["ready_for_kill"] or result["completed"])
                    else 1
                )
            if args.action == "kill":
                result = kill_drill_controller(
                    state_dir, role=args.role, snapshot=_scheduler_for_cli()
                )
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0
            if args.action == "wait":
                result = wait_for_drill_recovery(
                    state_dir, role=args.role, timeout_seconds=args.timeout
                )
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0
            if args.action == "finish":
                result = finish_controller_drill(
                    state_dir, timeout_seconds=min(args.timeout, 900.0)
                )
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0
            raise AssertionError(f"unhandled drill action {args.action}")
        if args.command == "supervise":
            return supervise(
                state_dir,
                role=args.role,
                generation=args.generation,
                intent_token=args.intent_token,
                heartbeat_interval=args.heartbeat_interval,
                stale_seconds=args.stale_seconds,
                drain_grace_seconds=args.drain_grace_seconds,
            )
        if args.command == "supervise-drill":
            return supervise_drill(
                state_dir,
                drill_id=args.drill_id,
                role=args.role,
                generation=args.generation,
                intent_token=args.intent_token,
                heartbeat_interval=args.heartbeat_interval,
            )
        raise AssertionError(f"unhandled command {args.command}")
    except (ControlError, OSError) as exc:
        print(f"[schema5-control] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
