"""Fail-closed decision logic for the schema-5 external dead-man watchdog.

The watchdog runs outside Slurm.  It is deliberately much less capable than the
in-cluster control plane: it may inspect the frozen live-status contract and invoke
one of two idempotent reconciliation operations.  It cannot resume a paused run,
clear a hold, alter capacity, or submit scientific work directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


WATCHDOG_PROTOCOL = "schema5-v1.2-r2-external-watchdog-v1"
WATCHDOG_SCHEMA_VERSION = 1
WATCHDOG_INTERVAL_SECONDS = 300
WATCHDOG_OBSERVATION_GAP_SECONDS = 60
WATCHDOG_STALE_SECONDS = 600
CONTROLLER_ROLES = ("dispatcher", "fleet_supervisor")
WATCHDOG_DRILL_PROTOCOL = "schema5-external-watchdog-drill-intent-v1"
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


def _controller_chain_absent_or_stale(
    report: Mapping[str, Any], *, stale_seconds: float
) -> bool:
    controllers = report.get("controllers")
    if not isinstance(controllers, Mapping) or set(controllers) != set(
        CONTROLLER_ROLES
    ):
        raise WatchdogError("watchdog status has an incomplete controller map")
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
        # successor is not: after an account-wide cancellation the durable control
        # record remains while two complete scheduler observations prove that its exact
        # allocation is gone.  Treating the stale record itself as live would make the
        # external dead-man unable to repair the exact failure it exists to cover.
        if row.get("successor_live") is True:
            return False
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
            return False
        # A missing allocation is actionable only when its heartbeat is absent/stale.
        age = row.get("heartbeat_age_seconds")
        if age is None:
            continue
        if not isinstance(age, (int, float)) or isinstance(age, bool):
            raise WatchdogError(f"watchdog heartbeat age is invalid for {role}")
        if float(age) <= stale_seconds:
            return False
    return True


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
    paused_drill_active = desired_state == "paused" and watchdog_drill is not None
    if desired_state != "running" and not finalization_active and not paused_drill_active:
        return WatchdogDecision(
            None,
            "desired_state is not running and finalization is not active",
            desired_state,
            finalization_state,
            second_time,
        )
    if not all(
        _controller_chain_absent_or_stale(report, stale_seconds=stale_seconds)
        for report in observations
    ):
        return WatchdogDecision(
            None,
            "an exact controller or recorded successor remains valid",
            desired_state,
            finalization_state,
            second_time,
        )
    return WatchdogDecision(
        "finalizer-reconcile" if finalization_active else "repair-chain",
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
