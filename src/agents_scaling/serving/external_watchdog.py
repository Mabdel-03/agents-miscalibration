"""Fail-closed decision logic for the schema-5 external dead-man watchdog.

The watchdog runs outside Slurm.  It is deliberately much less capable than the
in-cluster control plane: it may inspect the frozen live-status contract and invoke
one of two idempotent reconciliation operations.  It cannot resume a paused run,
clear a hold, alter capacity, or submit scientific work directly.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


WATCHDOG_PROTOCOL = "schema5-v1.2-r9-external-watchdog-v1"
WATCHDOG_SCHEMA_VERSION = 1
WATCHDOG_INTERVAL_SECONDS = 300
WATCHDOG_OBSERVATION_GAP_SECONDS = 60
WATCHDOG_STALE_SECONDS = 600
WATCHDOG_RELEASE_TAG = "sweep-recovery-schema5-v1.2-r9"
WATCHDOG_CLUSTER_CYCLE_PROTOCOL = (
    "schema5-v1.2-r9-external-watchdog-cycle-receipt-v1"
)
WATCHDOG_CLUSTER_LATEST_PROTOCOL = (
    "schema5-v1.2-r9-external-watchdog-latest-pointer-v1"
)
CONTROLLER_ROLES = ("dispatcher", "fleet_supervisor")
WATCHDOG_DRILL_PROTOCOL = "schema5-external-watchdog-drill-intent-v1"
FINALIZER_JOB_TOKEN_PREFIX = "asys-schema5-finalizer-v1"
EXACT_SBATCH_SUBMISSION_TRANSPORT = "stdin_exact_bytes_held_v1"
ACTIVE_SCHEDULER_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "RESIZING",
        "SUSPENDED",
    }
)
TERMINAL_SCHEDULER_STATES = frozenset(
    {
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
)
ACTIVE_FINALIZATION_STATES = frozenset(
    {
        "requested",
        "draining",
        "validating",
        "retiring_fleet",
        "snapshotting",
    }
)


class WatchdogError(RuntimeError):
    """The external watchdog cannot establish a safe decision."""


@dataclass(frozen=True)
class WatchdogDecision:
    """One closed watchdog decision derived from two complete observations."""

    action: str | None
    reason: str
    desired_state: str
    finalization_state: str
    observed_at: float


def _timestamp(report: Mapping[str, Any]) -> float:
    value = report.get("captured_timestamp")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WatchdogError("watchdog status lacks a numeric captured_timestamp")
    return float(value)


def _validate_scheduler(report: Mapping[str, Any]) -> None:
    scheduler = report.get("scheduler")
    if not isinstance(scheduler, Mapping):
        raise WatchdogError("watchdog status lacks scheduler truth")
    if (
        scheduler.get("squeue_ok") is not True
        or scheduler.get("sacct_ok") is not True
        or scheduler.get("errors") not in ([], ())
    ):
        raise WatchdogError("watchdog requires complete, unambiguous squeue+sacct truth")


def _validate_identity(
    report: Mapping[str, Any],
    *,
    expected_release_id: str,
    expected_git_commit: str,
    expected_control_sha256: str,
) -> None:
    if report.get("schema_version") != 1:
        raise WatchdogError("unsupported schema-5 live-status schema")
    if report.get("release_id") != expected_release_id:
        raise WatchdogError("watchdog live status release identity drifted")
    if report.get("git_commit") != expected_git_commit:
        raise WatchdogError("watchdog live status Git identity drifted")
    if report.get("immutable_sha256") != expected_control_sha256:
        raise WatchdogError("watchdog live status control identity drifted")


def _finalization_state(report: Mapping[str, Any]) -> str:
    finalization = report.get("finalization")
    if not isinstance(finalization, Mapping):
        raise WatchdogError("watchdog live status lacks finalization state")
    state = finalization.get("state")
    if state not in {
        "idle",
        "requested",
        "draining",
        "validating",
        "retiring_fleet",
        "snapshotting",
        "complete",
        "blocked",
    }:
        raise WatchdogError("watchdog live status has an invalid finalization state")
    return str(state)


def _validate_finalizer_job_record(
    value: Any,
    *,
    intent_id: str,
    role: str,
) -> dict[str, Any] | None:
    """Validate the durable identity used by the frozen reconciler.

    The external host must not attempt to reproduce Slurm reconciliation.  It does,
    however, require the status projection to carry an internally exact intent,
    token, sbatch, dependency, and numeric-job identity before delegating to the
    cluster-side idempotent reconciler.
    """

    if value is None:
        return None
    expected = {
        "attempt",
        "intent_token",
        "job_token",
        "sbatch_path",
        "sbatch_sha256",
        "submission_transport",
        "submission_argv",
        "submission_argv_sha256",
        "persistent_hold",
        "spooled_receipt_path",
        "spooled_receipt_sha256",
        "released_at",
        "released_timestamp",
        "dependency_job_id",
        "job_id",
        "state",
        "created_at",
        "created_timestamp",
        "submitted_at",
        "submitted_timestamp",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise WatchdogError(f"watchdog {role} finalizer identity is incomplete")
    attempt = value.get("attempt")
    intent_token = value.get("intent_token")
    dependency = value.get("dependency_job_id")
    job_id = value.get("job_id")
    state = value.get("state")
    created = value.get("created_timestamp")
    submitted = value.get("submitted_timestamp")
    released = value.get("released_timestamp")
    sbatch_path = str(value.get("sbatch_path", ""))
    expected_token = (
        f"{FINALIZER_JOB_TOKEN_PREFIX};intent={intent_id};"
        f"attempt={attempt};token={intent_token}"
    )
    submission_argv = value.get("submission_argv")
    expected_submission_argv = [
        "sbatch",
        "--parsable",
        f"--comment={expected_token}",
        "--hold",
    ]
    if dependency is not None:
        expected_submission_argv.append(
            f"--dependency=afterany:{dependency}"
        )
    expected_argv_sha256 = hashlib.sha256(
        json.dumps(
            expected_submission_argv,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt_path = value.get("spooled_receipt_path")
    receipt_sha256 = value.get("spooled_receipt_sha256")
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 1
        or re.fullmatch(r"[0-9a-f]{32}", str(intent_token)) is None
        or value.get("job_token") != expected_token
        or not Path(sbatch_path).is_absolute()
        or any(character in sbatch_path for character in ("\x00", "\n", "\r"))
        or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sbatch_sha256", "")))
        is None
        or value.get("submission_transport")
        != EXACT_SBATCH_SUBMISSION_TRANSPORT
        or submission_argv != expected_submission_argv
        or value.get("submission_argv_sha256") != expected_argv_sha256
        or value.get("persistent_hold") is not False
        or (
            dependency is not None
            and (not isinstance(dependency, str) or not dependency.isdigit())
        )
        or (
            job_id is not None
            and (not isinstance(job_id, str) or not job_id.isdigit())
        )
        or state
        not in {"submitting", "submitted", "started", "terminal", "cancelled"}
        or not isinstance(created, (int, float))
        or isinstance(created, bool)
        or not isinstance(value.get("created_at"), str)
        or (
            submitted is not None
            and (
                not isinstance(submitted, (int, float))
                or isinstance(submitted, bool)
                or not isinstance(value.get("submitted_at"), str)
            )
        )
        or (submitted is None and value.get("submitted_at") is not None)
        or (
            state in {"submitted", "started", "terminal", "cancelled"}
            and (
                job_id is None
                or submitted is None
                or not isinstance(receipt_path, str)
                or not Path(receipt_path).is_absolute()
                or re.fullmatch(r"[0-9a-f]{64}", str(receipt_sha256 or ""))
                is None
                or not isinstance(released, (int, float))
                or isinstance(released, bool)
                or not isinstance(value.get("released_at"), str)
            )
        )
        or (
            job_id is None
            and (
                receipt_path is not None
                or receipt_sha256 is not None
                or released is not None
                or value.get("released_at") is not None
            )
        )
    ):
        raise WatchdogError(f"watchdog {role} finalizer identity is invalid")
    return dict(value)


def _validate_finalizer_status_row(
    finalization: Mapping[str, Any],
    *,
    intent_id: str | None,
    field: str,
) -> dict[str, Any]:
    expected = {
        "recorded",
        "recorded_job_id",
        "scheduler_job_id",
        "attempt",
        "job_token",
        "sbatch_path",
        "dependency_job_id",
        "live",
        "scheduler_state",
        "identity_mismatch",
        "adoptable",
        "missing",
        "missing_after_grace",
        "visibility_pending",
        "terminal",
        "provenance",
    }
    row = finalization.get(field)
    if not isinstance(row, Mapping) or set(row) != expected:
        raise WatchdogError(
            f"watchdog {field} finalizer scheduler projection is incomplete"
        )
    for boolean_field in (
        "recorded",
        "live",
        "identity_mismatch",
        "adoptable",
        "missing",
        "missing_after_grace",
        "visibility_pending",
        "terminal",
    ):
        if not isinstance(row.get(boolean_field), bool):
            raise WatchdogError(
                f"watchdog {field} finalizer scheduler projection is invalid"
            )
    raw_record = finalization.get(f"{field}_job")
    if raw_record is not None and intent_id is None:
        raise WatchdogError(
            f"watchdog {field} finalizer lacks its parent intent identity"
        )
    record = (
        _validate_finalizer_job_record(
            raw_record,
            intent_id=str(intent_id),
            role=field,
        )
        if raw_record is not None
        else None
    )
    if row["recorded"] is not (record is not None):
        raise WatchdogError(
            f"watchdog {field} finalizer ledger projection is contradictory"
        )
    if record is None:
        if dict(row) != {
            "recorded": False,
            "recorded_job_id": None,
            "scheduler_job_id": None,
            "attempt": None,
            "job_token": None,
            "sbatch_path": None,
            "dependency_job_id": None,
            "live": False,
            "scheduler_state": None,
            "identity_mismatch": False,
            "adoptable": False,
            "missing": False,
            "missing_after_grace": False,
            "visibility_pending": False,
            "terminal": False,
            "provenance": None,
        }:
            raise WatchdogError(
                f"watchdog unrecorded {field} finalizer projection is ambiguous"
            )
        return {"record": None, "scheduler": dict(row)}

    projected = {
        "recorded_job_id": record.get("job_id"),
        "attempt": record["attempt"],
        "job_token": record["job_token"],
        "sbatch_path": record["sbatch_path"],
        "dependency_job_id": record["dependency_job_id"],
    }
    if any(row.get(key) != value for key, value in projected.items()):
        raise WatchdogError(
            f"watchdog {field} finalizer identity projection drifted"
        )
    scheduler_job_id = row.get("scheduler_job_id")
    if scheduler_job_id is not None and (
        not isinstance(scheduler_job_id, str) or not scheduler_job_id.isdigit()
    ):
        raise WatchdogError(
            f"watchdog {field} finalizer scheduler job ID is invalid"
        )
    if row["adoptable"]:
        if (
            record.get("job_id") is not None
            or record.get("state") != "submitting"
            or scheduler_job_id is None
        ):
            raise WatchdogError(
                f"watchdog {field} finalizer adoption identity is invalid"
            )
    elif scheduler_job_id != (
        record.get("job_id")
        if row["live"] or row["terminal"] or row["identity_mismatch"]
        else None
    ):
        raise WatchdogError(
            f"watchdog {field} finalizer scheduler job ID drifted"
        )
    state = row.get("scheduler_state")
    if state is not None and state not in (
        ACTIVE_SCHEDULER_STATES | TERMINAL_SCHEDULER_STATES
    ):
        raise WatchdogError(
            f"watchdog {field} finalizer scheduler state is ambiguous"
        )
    truth_count = sum(
        bool(row[key])
        for key in ("live", "identity_mismatch", "missing", "terminal")
    )
    if truth_count != 1:
        raise WatchdogError(
            f"watchdog {field} finalizer scheduler truth is contradictory"
        )
    if row["missing_after_grace"] and not row["missing"]:
        raise WatchdogError(
            f"watchdog {field} finalizer missing-after-grace state is invalid"
        )
    if row["visibility_pending"] and (
        not row["missing"] or row["missing_after_grace"]
    ):
        raise WatchdogError(
            f"watchdog {field} finalizer visibility state is invalid"
        )
    if row["missing"] and not (
        row["missing_after_grace"] or row["visibility_pending"]
    ):
        raise WatchdogError(
            f"watchdog {field} finalizer missing state lacks visibility authority"
        )
    if (
        row["live"] is not (state in ACTIVE_SCHEDULER_STATES)
        or row["terminal"] is not (state in TERMINAL_SCHEDULER_STATES)
        or (
            (row["missing"] or row["identity_mismatch"])
            and row["provenance"] is not None
        )
    ):
        raise WatchdogError(
            f"watchdog {field} finalizer scheduler classification is invalid"
        )
    provenance = row.get("provenance")
    if provenance is not None:
        expected_provenance = {
            "job_id",
            "job_name",
            "state",
            "comment",
            "command",
            "source",
            "dependency",
        }
        if not isinstance(provenance, Mapping):
            raise WatchdogError(
                f"watchdog {field} finalizer scheduler provenance is invalid"
            )
        command = provenance.get("command")
        dependency_provenance = provenance.get("dependency")
        try:
            command_arguments = (
                shlex.split(command) if isinstance(command, str) else []
            )
        except ValueError:
            command_arguments = []
        dependency_valid = (
            isinstance(dependency_provenance, str)
            and (
                (
                    record.get("dependency_job_id") is None
                    and dependency_provenance.strip().lower()
                    in {"", "(null)", "null", "none"}
                )
                or (
                    record.get("dependency_job_id") is not None
                    and re.fullmatch(
                        rf"afterany:{re.escape(str(record['dependency_job_id']))}"
                        r"(?:\([^()]*\))?",
                        dependency_provenance.strip(),
                    )
                    is not None
                )
            )
        )
        if (
            set(provenance) != expected_provenance
            or provenance.get("job_id") != scheduler_job_id
            or provenance.get("job_name")
            != f"asys-s5-final-a{int(record['attempt']):06d}"
            or provenance.get("state") != state
            or provenance.get("comment") != record.get("job_token")
            or record["sbatch_path"] not in command_arguments
            or provenance.get("source") not in {"squeue", "sacct"}
            or not dependency_valid
        ):
            raise WatchdogError(
                f"watchdog {field} finalizer scheduler provenance is invalid"
            )
    return {
        "record": record,
        "scheduler": {
            key: (
                {
                    provenance_key: provenance[provenance_key]
                    for provenance_key in (
                        "job_id",
                        "job_name",
                        "state",
                        "comment",
                        "command",
                        "dependency",
                    )
                }
                if key == "provenance" and provenance is not None
                else row[key]
            )
            for key in expected
        },
    }


def _validate_pending_finalization_request(
    finalization: Mapping[str, Any],
) -> dict[str, Any] | None:
    value = finalization.get("pending_request")
    if value is None:
        return None
    expected = {
        "validated",
        "intent_id",
        "requested_timestamp",
        "semantic_evidence_sha256",
        "request_payload_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or value.get("validated") is not True
        or re.fullmatch(r"[0-9a-f]{64}", str(value.get("intent_id", "")))
        is None
        or not isinstance(value.get("requested_timestamp"), (int, float))
        or isinstance(value.get("requested_timestamp"), bool)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(value.get("semantic_evidence_sha256", "")),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(value.get("request_payload_sha256", "")),
        )
        is None
    ):
        raise WatchdogError(
            "watchdog pending finalization request identity is invalid"
        )
    return dict(value)


def _finalizer_chain_assessment(
    report: Mapping[str, Any],
    *,
    finalization_state: str,
) -> dict[str, Any]:
    """Validate and classify one scheduler-authoritative finalizer projection."""

    finalization = report["finalization"]
    intent_id_value = finalization.get("intent_id")
    intent_id = (
        str(intent_id_value)
        if isinstance(intent_id_value, str)
        and re.fullmatch(r"[0-9a-f]{64}", intent_id_value) is not None
        else None
    )
    if (
        finalization_state in ACTIVE_FINALIZATION_STATES
        or finalization_state == "complete"
    ) and intent_id is None:
        raise WatchdogError(
            "watchdog active or complete finalization intent identity is invalid"
        )
    if finalization.get("scheduler_complete") is not True or finalization.get(
        "scheduler_error"
    ) is not None:
        raise WatchdogError(
            "watchdog finalizer requires complete, unambiguous scheduler truth"
        )
    active = _validate_finalizer_status_row(
        finalization,
        intent_id=intent_id,
        field="active",
    )
    successor = _validate_finalizer_status_row(
        finalization,
        intent_id=intent_id,
        field="successor",
    )
    active_record = active["record"]
    successor_record = successor["record"]
    if successor_record is not None and active_record is None:
        raise WatchdogError(
            "watchdog finalizer successor exists without an active identity"
        )
    if (
        active_record is not None
        and active_record.get("dependency_job_id") is not None
    ):
        raise WatchdogError("watchdog active finalizer has an unexpected dependency")
    if successor_record is not None:
        active_job_id = active_record.get("job_id")
        if (
            active_job_id is None
            or successor_record.get("dependency_job_id") != active_job_id
            or successor_record.get("attempt") == active_record.get("attempt")
            or successor_record.get("intent_token")
            == active_record.get("intent_token")
            or (
                successor_record.get("job_id") is not None
                and successor_record.get("job_id") == active_job_id
            )
        ):
            raise WatchdogError(
                "watchdog finalizer successor identity is ambiguous"
            )
    namespace_value = finalization.get("namespace_active_job_ids")
    unexpected_value = finalization.get("unexpected_active_job_ids")
    for value in (namespace_value, unexpected_value):
        if (
            not isinstance(value, list)
            or any(
                not isinstance(job_id, str) or not job_id.isdigit()
                for job_id in value
            )
            or value != sorted(set(value), key=int)
        ):
            raise WatchdogError(
                "watchdog finalizer namespace identity is invalid"
            )
    namespace = list(namespace_value)
    unexpected = list(unexpected_value)
    if set(unexpected) - set(namespace):
        raise WatchdogError("watchdog finalizer namespace identity is invalid")
    expected_job_ids = {
        str(row["scheduler"]["scheduler_job_id"])
        for row in (active, successor)
        if isinstance(row["scheduler"].get("scheduler_job_id"), str)
        and str(row["scheduler"]["scheduler_job_id"]).isdigit()
    }
    computed_unexpected = sorted(set(namespace) - expected_job_ids, key=int)
    if unexpected != computed_unexpected:
        raise WatchdogError(
            "watchdog finalizer namespace projection is contradictory"
        )
    if unexpected:
        raise WatchdogError(
            "watchdog finalizer namespace contains an untrusted active identity"
        )
    if active["scheduler"]["identity_mismatch"] or successor["scheduler"][
        "identity_mismatch"
    ]:
        raise WatchdogError("watchdog finalizer scheduler identity is ambiguous")
    if (
        active["scheduler"]["live"]
        and successor["scheduler"]["terminal"]
        and successor["scheduler"]["scheduler_state"]
        not in {"CANCELLED", "REVOKED"}
    ):
        raise WatchdogError(
            "watchdog finalizer successor executed while its publisher remains live"
        )
    pending_request = _validate_pending_finalization_request(finalization)
    if pending_request is not None and (
        finalization_state != "idle"
        or intent_id is not None
        or active_record is not None
        or successor_record is not None
        or namespace
        or report.get("desired_state") != "paused"
        or report.get("drain_requested") is not True
    ):
        raise WatchdogError(
            "watchdog pending finalization request lost its admission fence"
        )
    completion_marker = finalization.get("completion_marker")
    successor_retirement = finalization.get("successor_retirement")
    complete_terminal_healthy = bool(
        finalization_state == "complete"
        and pending_request is None
        and isinstance(completion_marker, Mapping)
        and set(completion_marker) == {"path", "sha256", "final_id"}
        and isinstance(completion_marker.get("path"), str)
        and Path(str(completion_marker["path"])).is_absolute()
        and re.fullmatch(
            r"[0-9a-f]{64}", str(completion_marker.get("sha256", ""))
        )
        is not None
        and re.fullmatch(
            r"[0-9a-f]{64}", str(completion_marker.get("final_id", ""))
        )
        is not None
        and isinstance(successor_retirement, Mapping)
        and isinstance(successor_retirement.get("receipt_path"), str)
        and Path(str(successor_retirement["receipt_path"])).is_absolute()
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(successor_retirement.get("receipt_sha256", "")),
        )
        is not None
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(successor_retirement.get("receipt_id", "")),
        )
        is not None
        and active_record is not None
        and successor_record is not None
        and successor_record.get("state") in {"terminal", "cancelled"}
        and not active["scheduler"]["live"]
        and not successor["scheduler"]["live"]
        and not active["scheduler"]["identity_mismatch"]
        and not successor["scheduler"]["identity_mismatch"]
        and not active["scheduler"]["adoptable"]
        and not successor["scheduler"]["adoptable"]
        and not namespace
    )
    if finalization_state == "complete" and not complete_terminal_healthy:
        raise WatchdogError(
            "watchdog complete finalization lacks terminal marker, retirement, "
            "or quiescent scheduler proof"
        )
    computed_healthy = bool(
        (
            finalization_state == "idle"
            and pending_request is None
            and active_record is None
            and successor_record is None
            and not namespace
        )
        or (
            finalization_state in ACTIVE_FINALIZATION_STATES
            and active["scheduler"]["live"]
            and successor["scheduler"]["live"]
            and successor_record is not None
            and active_record is not None
            and not active["scheduler"]["adoptable"]
            and not successor["scheduler"]["adoptable"]
            and successor_record["dependency_job_id"]
            == active_record["job_id"]
            and bool(namespace)
        )
        or complete_terminal_healthy
    )
    # The second branch above requires both exact job IDs to comprise the namespace.
    if (
        finalization_state in ACTIVE_FINALIZATION_STATES
        and computed_healthy
        and set(namespace)
        != {str(active_record["job_id"]), str(successor_record["job_id"])}
    ):
        computed_healthy = False
    if finalization.get("chain_healthy") is not computed_healthy:
        raise WatchdogError(
            "watchdog finalizer health projection is contradictory"
        )
    visibility_pending = bool(
        active["scheduler"]["visibility_pending"]
        or successor["scheduler"]["visibility_pending"]
    )
    adoptable = bool(
        active["scheduler"]["adoptable"]
        or successor["scheduler"]["adoptable"]
    )
    if pending_request is not None or adoptable:
        disposition = "recoverable"
    elif finalization_state in ACTIVE_FINALIZATION_STATES:
        if computed_healthy:
            disposition = "healthy"
        elif visibility_pending:
            disposition = "visibility_pending"
        else:
            disposition = "recoverable"
    else:
        disposition = "terminal" if finalization_state != "idle" else "healthy"
    return {
        "intent_id": intent_id,
        "pending_request": pending_request,
        "active_job": active,
        "successor_job": successor,
        "namespace_active_job_ids": namespace,
        "chain_healthy": computed_healthy,
        "disposition": disposition,
    }


def _controller_chain_absent_or_stale(
    report: Mapping[str, Any], *, stale_seconds: float
) -> bool:
    controllers = report.get("controllers")
    if not isinstance(controllers, Mapping) or set(controllers) != set(
        CONTROLLER_ROLES
    ):
        raise WatchdogError("watchdog status has an incomplete controller map")
    all_absent_or_stale = True
    for role in CONTROLLER_ROLES:
        row = controllers[role]
        if not isinstance(row, Mapping):
            raise WatchdogError(f"watchdog controller record is invalid: {role}")
        if (
            row.get("active_identity_mismatch") is True
            or row.get("successor_identity_mismatch") is True
        ):
            raise WatchdogError(
                f"watchdog refuses ambiguous controller identity for {role}"
            )
        # A live exact successor is already the recovery mechanism.  A merely recorded
        # successor is not: after an exact recovery-namespace cancellation the durable
        # control record remains while two complete scheduler observations prove that
        # its exact allocation is gone.  Treating the stale record itself as live would
        # make the external dead-man unable to repair the exact failure it covers.
        if row.get("successor_live") is True:
            all_absent_or_stale = False
            continue
        successor_state = row.get("successor_scheduler_state")
        if successor_state not in (
            None,
            "",
            "BOOT_FAIL",
            "CANCELLED",
            "COMPLETED",
            "DEADLINE",
            "FAILED",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "TIMEOUT",
        ):
            raise WatchdogError(
                f"watchdog successor scheduler state is ambiguous for {role}"
            )
        if row.get("live_active") is True:
            age = row.get("heartbeat_age_seconds")
            if (
                isinstance(age, (int, float))
                and not isinstance(age, bool)
                and float(age) > stale_seconds
            ):
                continue
            all_absent_or_stale = False
            continue
        # A missing allocation is actionable only when its heartbeat is absent/stale.
        age = row.get("heartbeat_age_seconds")
        if age is None:
            continue
        if not isinstance(age, (int, float)) or isinstance(age, bool):
            raise WatchdogError(f"watchdog heartbeat age is invalid for {role}")
        if float(age) <= stale_seconds:
            all_absent_or_stale = False
    return all_absent_or_stale


def _active_watchdog_drill(
    report: Mapping[str, Any], *, observed_at: float
) -> dict[str, Any] | None:
    """Validate the sole paused-state exception to ordinary watchdog authority."""

    value = report.get("watchdog_drill")
    if value is None:
        return None
    required = {
        "schema_version",
        "protocol",
        "active",
        "intent_id",
        "drill_id",
        "phase",
        "created_timestamp",
        "expires_timestamp",
        "maximum_recovery_seconds",
        "admission_zero",
        "production_baseline_unchanged",
        "namespace_job_ids",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise WatchdogError("watchdog drill status fields differ from the closed schema")
    job_ids = value.get("namespace_job_ids")
    created = value.get("created_timestamp")
    expires = value.get("expires_timestamp")
    if (
        value.get("schema_version") != 1
        or value.get("protocol") != WATCHDOG_DRILL_PROTOCOL
        or value.get("active") is not True
        or not isinstance(value.get("intent_id"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(value["intent_id"])) is None
        or not isinstance(value.get("drill_id"), str)
        or not value["drill_id"]
        or value.get("phase")
        not in {"armed", "cancelling", "cancelled", "recovering"}
        or not isinstance(created, (int, float))
        or isinstance(created, bool)
        or not isinstance(expires, (int, float))
        or isinstance(expires, bool)
        or not float(created) < observed_at < float(expires)
        or value.get("maximum_recovery_seconds") != 900.0
        or value.get("admission_zero") is not True
        or value.get("production_baseline_unchanged") is not True
        or not isinstance(job_ids, list)
        or len(job_ids) != 4
        or len(set(job_ids)) != 4
        or any(not isinstance(job_id, str) or not job_id.isdigit() for job_id in job_ids)
    ):
        raise WatchdogError("watchdog drill is expired, malformed, or not fail closed")
    return dict(value)


def decide(
    observations: Sequence[Mapping[str, Any]],
    *,
    expected_release_id: str,
    expected_git_commit: str,
    expected_control_sha256: str,
    minimum_gap_seconds: float = WATCHDOG_OBSERVATION_GAP_SECONDS,
    stale_seconds: float = WATCHDOG_STALE_SECONDS,
) -> WatchdogDecision:
    """Return the only action permitted by two independent status observations."""

    if len(observations) != 2:
        raise WatchdogError("watchdog requires exactly two scheduler observations")
    first, second = observations
    for report in observations:
        _validate_identity(
            report,
            expected_release_id=expected_release_id,
            expected_git_commit=expected_git_commit,
            expected_control_sha256=expected_control_sha256,
        )
        _validate_scheduler(report)
    first_time = _timestamp(first)
    second_time = _timestamp(second)
    if second_time - first_time < float(minimum_gap_seconds):
        raise WatchdogError("watchdog observations are not sufficiently independent")

    desired_states = {str(report.get("desired_state")) for report in observations}
    finalization_states = {_finalization_state(report) for report in observations}
    if len(desired_states) != 1 or len(finalization_states) != 1:
        raise WatchdogError("watchdog control state changed between observations")
    desired_state = desired_states.pop()
    finalization_state = finalization_states.pop()
    finalizer_chains = [
        _finalizer_chain_assessment(
            report,
            finalization_state=finalization_state,
        )
        for report in observations
    ]
    if finalizer_chains[0] != finalizer_chains[1]:
        raise WatchdogError(
            "watchdog finalizer identity changed between observations"
        )
    drill_states = [
        _active_watchdog_drill(report, observed_at=_timestamp(report))
        for report in observations
    ]
    if (drill_states[0] is None) != (drill_states[1] is None):
        raise WatchdogError("watchdog drill state changed between observations")
    watchdog_drill = drill_states[1]
    if watchdog_drill is not None and drill_states[0] != watchdog_drill:
        raise WatchdogError("watchdog drill identity changed between observations")

    if finalization_state == "blocked":
        return WatchdogDecision(
            None,
            "scientific finalization is blocked and requires human acknowledgement",
            desired_state,
            finalization_state,
            second_time,
        )
    finalization_active = finalization_state in ACTIVE_FINALIZATION_STATES
    controller_chains_absent_or_stale = [
        _controller_chain_absent_or_stale(
            report,
            stale_seconds=stale_seconds,
        )
        for report in observations
    ]
    if finalization_state == "complete":
        return WatchdogDecision(
            None,
            "scientific finalization is already complete",
            desired_state,
            finalization_state,
            second_time,
        )
    finalizer_disposition = finalizer_chains[1]["disposition"]
    pending_finalization = finalizer_chains[1]["pending_request"] is not None
    if finalization_active or pending_finalization:
        if finalizer_disposition in {"healthy", "visibility_pending"}:
            return WatchdogDecision(
                None,
                (
                    "the exact finalizer publisher and successor are live"
                    if finalizer_disposition == "healthy"
                    else "an exact finalizer submission remains inside scheduler "
                    "visibility grace"
                ),
                desired_state,
                finalization_state,
                second_time,
            )
        if finalizer_disposition != "recoverable":
            raise WatchdogError(
                "watchdog finalizer recovery disposition is ambiguous"
            )
        # The external host does not submit a finalizer.  Two complete, stable status
        # cuts authorize only the frozen cluster-side reconciler, which transactionally
        # adopts or repairs the exact request and scheduler identities.  Controller
        # health is deliberately irrelevant: controllers can remain live after the
        # admission fence/request commit while the finalizer chain is absent.
        return WatchdogDecision(
            "finalizer-reconcile",
            (
                "two complete observations prove the exact finalizer chain "
                "recoverably absent or terminal"
            ),
            desired_state,
            finalization_state,
            second_time,
        )
    paused_drill_active = desired_state == "paused" and watchdog_drill is not None
    if desired_state != "running" and not finalization_active and not paused_drill_active:
        return WatchdogDecision(
            None,
            "desired_state is not running and finalization is not active",
            desired_state,
            finalization_state,
            second_time,
        )
    if not all(controller_chains_absent_or_stale):
        return WatchdogDecision(
            None,
            "an exact controller or recorded successor remains valid",
            desired_state,
            finalization_state,
            second_time,
        )
    return WatchdogDecision(
        "repair-chain",
        (
            "both observations prove the exact paused drill namespace absent "
            "with no successor"
            if paused_drill_active
            else "both observations prove absent/stale controller chains with no successor"
        ),
        desired_state,
        finalization_state,
        second_time,
    )
