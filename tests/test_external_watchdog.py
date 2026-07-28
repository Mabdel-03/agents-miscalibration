from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from agents_scaling.serving import external_watchdog as watchdog
from scripts import build_schema5_watchdog_deployment as deployment
from scripts import schema5_external_watchdog as runner
from scripts import schema5_watchdog_forced_command as forced


RELEASE = "sweep-recovery-schema5-v1.2"
COMMIT = "a" * 40
CONTROL = "b" * 64
TAG_OBJECT = "c" * 40
FINALIZER_INTENT = "d" * 64


def test_external_watchdog_fresh_protocol_is_r9_end_to_end():
    assert (
        watchdog.WATCHDOG_PROTOCOL
        == "schema5-v1.2-r9-external-watchdog-v1"
    )
    assert runner.WATCHDOG_PROTOCOL == watchdog.WATCHDOG_PROTOCOL


def _finalizer_job(
    *,
    attempt: int,
    token_character: str,
    job_id: str | None,
    dependency_job_id: str | None,
    state: str = "submitted",
) -> dict:
    intent_token = token_character * 32
    submitted_timestamp = 2.0 if job_id is not None else None
    job_token = (
        f"{watchdog.FINALIZER_JOB_TOKEN_PREFIX};intent={FINALIZER_INTENT};"
        f"attempt={attempt};token={intent_token}"
    )
    submission_argv = [
        "sbatch",
        "--parsable",
        f"--comment={job_token}",
        "--hold",
    ]
    if dependency_job_id is not None:
        submission_argv.append(
            f"--dependency=afterany:{dependency_job_id}"
        )
    value = {
        "attempt": attempt,
        "intent_token": intent_token,
        "job_token": job_token,
        "sbatch_path": (
            f"/immutable/finalizer/finalizer.a{attempt:06d}."
            f"{intent_token}.sbatch"
        ),
        "sbatch_sha256": token_character * 64,
        "submission_transport": watchdog.EXACT_SBATCH_SUBMISSION_TRANSPORT,
        "submission_argv": submission_argv,
        "submission_argv_sha256": hashlib.sha256(
            json.dumps(
                submission_argv,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
        "persistent_hold": False,
        "spooled_receipt_path": (
            f"/immutable/finalizer/spool/{job_id}.json"
            if job_id is not None
            else None
        ),
        "spooled_receipt_sha256": (
            token_character * 64 if job_id is not None else None
        ),
        "released_at": (
            "1970-01-01T00:00:02+00:00"
            if job_id is not None
            else None
        ),
        "released_timestamp": submitted_timestamp,
        "dependency_job_id": dependency_job_id,
        "job_id": job_id,
        "state": state,
        "created_at": "1970-01-01T00:00:01+00:00",
        "created_timestamp": 1.0,
        "submitted_at": (
            "1970-01-01T00:00:02+00:00"
            if submitted_timestamp is not None
            else None
        ),
        "submitted_timestamp": submitted_timestamp,
    }
    return value


def _finalizer_status_row(
    record: dict | None,
    *,
    field: str,
) -> dict:
    if record is None:
        return {
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
        }
    dependency = record["dependency_job_id"]
    state = "PENDING" if field == "successor" else "RUNNING"
    return {
        "recorded": True,
        "recorded_job_id": record["job_id"],
        "scheduler_job_id": record["job_id"],
        "attempt": record["attempt"],
        "job_token": record["job_token"],
        "sbatch_path": record["sbatch_path"],
        "dependency_job_id": dependency,
        "live": True,
        "scheduler_state": state,
        "identity_mismatch": False,
        "adoptable": False,
        "missing": False,
        "missing_after_grace": False,
        "visibility_pending": False,
        "terminal": False,
        "provenance": {
            "job_id": record["job_id"],
            "job_name": f"asys-s5-final-a{record['attempt']:06d}",
            "state": state,
            "comment": record["job_token"],
            "command": record["sbatch_path"],
            "source": "squeue",
            "dependency": (
                f"afterany:{dependency}" if dependency is not None else ""
            ),
        },
    }


def _status(
    captured: float,
    *,
    desired: str = "running",
    finalization: str = "idle",
    successor: str | None = None,
    successor_state: str | None = None,
    active: bool = False,
    age: float | None = 700.0,
    finalizer_chain: bool = False,
    pending_finalization: bool = False,
) -> dict:
    roles = {
        role: {
            "recorded_active_job_id": None,
            "live_active": active,
            "active_identity_mismatch": False,
            "recorded_successor_job_id": successor,
            "successor_live": successor is not None,
            "successor_identity_mismatch": False,
            "successor_scheduler_state": successor_state,
            "heartbeat_age_seconds": age,
        }
        for role in watchdog.CONTROLLER_ROLES
    }
    active_record = None
    successor_record = None
    if finalization in watchdog.ACTIVE_FINALIZATION_STATES:
        if finalizer_chain:
            active_record = _finalizer_job(
                attempt=1,
                token_character="e",
                job_id="901",
                dependency_job_id=None,
            )
            successor_record = _finalizer_job(
                attempt=2,
                token_character="f",
                job_id="902",
                dependency_job_id="901",
            )
    namespace = ["901", "902"] if finalizer_chain else []
    pending_request = (
        {
            "validated": True,
            "intent_id": FINALIZER_INTENT,
            "requested_timestamp": 1.0,
            "semantic_evidence_sha256": "1" * 64,
            "request_payload_sha256": "2" * 64,
        }
        if pending_finalization
        else None
    )
    finalization_value = {
        "state": finalization,
        "intent_id": (
            FINALIZER_INTENT
            if finalization in watchdog.ACTIVE_FINALIZATION_STATES
            else None
        ),
        "active_job": active_record,
        "successor_job": successor_record,
        "scheduler_complete": True,
        "pending_request": pending_request,
        "active": _finalizer_status_row(active_record, field="active"),
        "successor": _finalizer_status_row(
            successor_record,
            field="successor",
        ),
        "namespace_active_job_ids": namespace,
        "unexpected_active_job_ids": [],
        "scheduler_error": None,
        "chain_healthy": bool(
            finalizer_chain
            or (
                finalization == "idle"
                and not pending_finalization
            )
        ),
    }
    return {
        "schema_version": 1,
        "captured_timestamp": captured,
        "release_id": RELEASE,
        "git_commit": COMMIT,
        "immutable_sha256": CONTROL,
        "desired_state": desired,
        "rollout_generation": 0,
        "drain_requested": pending_finalization,
        "finalization": finalization_value,
        "scheduler": {"squeue_ok": True, "sacct_ok": True, "errors": []},
        "controllers": roles,
    }


def _cluster_mirror_response(
    first: dict,
    second: dict,
    *,
    action_binding: dict | None = None,
    sequence: int = 1,
) -> dict:
    cycle_identity = {
        "release_id": RELEASE,
        "release_tag": watchdog.WATCHDOG_RELEASE_TAG,
        "release_tag_object": TAG_OBJECT,
        "git_commit": COMMIT,
        "control_sha256": CONTROL,
        "desired_state": second["desired_state"],
        "rollout_generation": second["rollout_generation"],
        "finalization": {
            "state": second["finalization"]["state"],
            "intent_id": second["finalization"]["intent_id"],
            "pending_request_intent_id": None,
        },
    }
    if action_binding is not None:
        action_binding = dict(action_binding)
        action_binding.setdefault("identity_before", cycle_identity)
        action_binding.setdefault("identity_after", cycle_identity)
        action_binding.setdefault("outcome", "completed")
    receipt = {
        "schema_version": 1,
        "protocol": watchdog.WATCHDOG_CLUSTER_CYCLE_PROTOCOL,
        "sequence": sequence,
        "predecessor": None,
        "intent": {
            "path": f"/cluster/transactions/{sequence:012d}/INTENT.json",
            "sha256": "1" * 64,
            "intent_id": "2" * 64,
        },
        "identity": cycle_identity,
        "status_observations": [
            {
                "sequence": sequence * 2 - 1,
                "path": f"/cluster/status/{sequence * 2 - 1}.json",
                "sha256": "3" * 64,
                "observation_id": "4" * 64,
                "report_sha256": hashlib.sha256(
                    runner._canonical(first)
                ).hexdigest(),
                "report_captured_timestamp": first["captured_timestamp"],
            },
            {
                "sequence": sequence * 2,
                "path": f"/cluster/status/{sequence * 2}.json",
                "sha256": "5" * 64,
                "observation_id": "6" * 64,
                "report_sha256": hashlib.sha256(
                    runner._canonical(second)
                ).hexdigest(),
                "report_captured_timestamp": second["captured_timestamp"],
            },
        ],
        "consumed_status_sequence": sequence * 2,
        "action_receipt": action_binding,
        "consumed_action_sequence": (
            int(action_binding["sequence"])
            if action_binding is not None
            else 0
        ),
        "cluster_server_timestamp": second["captured_timestamp"] + 1.0,
    }
    receipt["receipt_id"] = runner._self_hash(receipt, "receipt_id")
    pointer = {
        "schema_version": 1,
        "protocol": watchdog.WATCHDOG_CLUSTER_LATEST_PROTOCOL,
        "sequence": sequence,
        "receipt_path": f"/cluster/cycles/{receipt['receipt_id']}.json",
        "receipt_sha256": "7" * 64,
        "receipt_id": receipt["receipt_id"],
        "cluster_server_timestamp": receipt["cluster_server_timestamp"],
    }
    pointer["pointer_id"] = runner._self_hash(pointer, "pointer_id")
    return {
        "published": True,
        "recovered": False,
        "receipt": receipt,
        "pointer": pointer,
    }


def _set_finalizer_missing(
    report: dict,
    *,
    field: str,
    visibility_pending: bool,
) -> None:
    row = report["finalization"][field]
    job_id = row["recorded_job_id"]
    row.update(
        {
            "scheduler_job_id": None,
            "live": False,
            "scheduler_state": None,
            "identity_mismatch": False,
            "missing": True,
            "missing_after_grace": not visibility_pending,
            "visibility_pending": visibility_pending,
            "terminal": False,
            "provenance": None,
        }
    )
    report["finalization"]["namespace_active_job_ids"] = [
        candidate
        for candidate in report["finalization"]["namespace_active_job_ids"]
        if candidate != job_id
    ]
    report["finalization"]["chain_healthy"] = False


def _set_finalizer_adoptable(report: dict) -> None:
    record = _finalizer_job(
        attempt=1,
        token_character="e",
        job_id=None,
        dependency_job_id=None,
        state="submitting",
    )
    row = _finalizer_status_row(record, field="active")
    row["scheduler_job_id"] = "901"
    row["adoptable"] = True
    row["provenance"]["job_id"] = "901"
    report["finalization"].update(
        {
            "active_job": record,
            "active": row,
            "namespace_active_job_ids": ["901"],
            "chain_healthy": False,
        }
    )


def _set_finalizer_terminal(
    report: dict,
    *,
    field: str,
    scheduler_state: str,
) -> None:
    row = report["finalization"][field]
    job_id = row["scheduler_job_id"]
    row.update(
        {
            "live": False,
            "scheduler_state": scheduler_state,
            "identity_mismatch": False,
            "adoptable": False,
            "missing": False,
            "missing_after_grace": False,
            "visibility_pending": False,
            "terminal": True,
        }
    )
    row["provenance"]["state"] = scheduler_state
    report["finalization"]["namespace_active_job_ids"] = [
        candidate
        for candidate in report["finalization"]["namespace_active_job_ids"]
        if candidate != job_id
    ]
    report["finalization"]["chain_healthy"] = False


def _complete_status(captured: float) -> dict:
    """Mirror the real live-status projection after marker-last completion."""

    report = _status(
        captured,
        desired="paused",
        finalization="complete",
    )
    active = _finalizer_job(
        attempt=1,
        token_character="e",
        job_id="901",
        dependency_job_id=None,
        state="started",
    )
    successor = _finalizer_job(
        attempt=2,
        token_character="f",
        job_id="902",
        dependency_job_id="901",
        state="cancelled",
    )
    active_row = _finalizer_status_row(active, field="active")
    successor_row = _finalizer_status_row(successor, field="successor")
    for row in (active_row, successor_row):
        row.update(
            {
                "scheduler_job_id": None,
                "live": False,
                "scheduler_state": None,
                "identity_mismatch": False,
                "adoptable": False,
                "missing": True,
                "missing_after_grace": True,
                "visibility_pending": False,
                "terminal": False,
                "provenance": None,
            }
        )
    report["drain_requested"] = True
    report["finalization"].update(
        {
            "intent_id": FINALIZER_INTENT,
            "active_job": active,
            "successor_job": successor,
            "active": active_row,
            "successor": successor_row,
            "completion_marker": {
                "path": "/results/final/FINAL_COMPLETE.json",
                "sha256": "1" * 64,
                "final_id": "2" * 64,
            },
            "successor_retirement": {
                "receipt_path": "/results/finalizer/SUCCESSOR_RETIREMENT_COMPLETE.json",
                "receipt_sha256": "3" * 64,
                "receipt_id": "4" * 64,
            },
            "namespace_active_job_ids": [],
            "unexpected_active_job_ids": [],
            "chain_healthy": True,
        }
    )
    return report


def test_watchdog_complete_terminal_chain_is_a_verified_noop():
    decision = watchdog.decide(
        (_complete_status(10.0), _complete_status(70.0)),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert decision.action is None
    assert decision.reason == "scientific finalization is already complete"


@pytest.mark.parametrize(
    "mutator",
    (
        lambda report: report["finalization"].pop("completion_marker"),
        lambda report: report["finalization"]["successor_retirement"].update(
            {"receipt_path": None}
        ),
        lambda report: report["finalization"].update(
            {"namespace_active_job_ids": ["902"], "chain_healthy": False}
        ),
    ),
)
def test_watchdog_malformed_complete_terminal_state_fails_closed(mutator):
    first = _complete_status(10.0)
    second = _complete_status(70.0)
    mutator(first)
    mutator(second)
    with pytest.raises(watchdog.WatchdogError):
        watchdog.decide(
            (first, second),
            expected_release_id=RELEASE,
            expected_git_commit=COMMIT,
            expected_control_sha256=CONTROL,
        )


def test_watchdog_repairs_only_after_two_complete_stale_observations():
    decision = watchdog.decide(
        (_status(10.0), _status(70.0)),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert decision.action == "repair-chain"

    final = watchdog.decide(
        (
            _status(10.0, desired="paused", finalization="snapshotting"),
            _status(70.0, desired="paused", finalization="snapshotting"),
        ),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert final.action == "finalizer-reconcile"


def test_watchdog_recovers_request_saved_before_chain_with_healthy_controllers():
    decision = watchdog.decide(
        (
            _status(
                10.0,
                desired="paused",
                finalization="requested",
                active=True,
                age=1.0,
            ),
            _status(
                70.0,
                desired="paused",
                finalization="requested",
                active=True,
                age=1.0,
            ),
        ),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert decision.action == "finalizer-reconcile"
    assert "exact finalizer chain" in decision.reason


def test_watchdog_recovers_pending_request_fence_before_active_state_commit():
    decision = watchdog.decide(
        (
            _status(
                10.0,
                desired="paused",
                active=True,
                age=1.0,
                pending_finalization=True,
            ),
            _status(
                70.0,
                desired="paused",
                active=True,
                age=1.0,
                pending_finalization=True,
            ),
        ),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert decision.action == "finalizer-reconcile"
    assert decision.desired_state == "paused"
    assert decision.finalization_state == "idle"


def test_watchdog_noops_for_healthy_exact_finalizer_and_controllers():
    decision = watchdog.decide(
        (
            _status(
                10.0,
                desired="paused",
                finalization="snapshotting",
                active=True,
                age=1.0,
                finalizer_chain=True,
            ),
            _status(
                70.0,
                desired="paused",
                finalization="snapshotting",
                active=True,
                age=1.0,
                finalizer_chain=True,
            ),
        ),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert decision.action is None
    assert "publisher and successor are live" in decision.reason


def test_watchdog_repairs_only_after_two_stable_missing_after_grace_cuts():
    first = _status(
        10.0,
        desired="paused",
        finalization="draining",
        active=True,
        age=1.0,
        finalizer_chain=True,
    )
    second = _status(
        70.0,
        desired="paused",
        finalization="draining",
        active=True,
        age=1.0,
        finalizer_chain=True,
    )
    for report in (first, second):
        _set_finalizer_missing(
            report,
            field="successor",
            visibility_pending=False,
        )
    decision = watchdog.decide(
        (first, second),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert decision.action == "finalizer-reconcile"


def test_watchdog_reconciles_exact_sbatch_accepted_before_job_id_commit():
    first = _status(
        10.0,
        desired="paused",
        finalization="requested",
        active=True,
        age=1.0,
    )
    second = _status(
        70.0,
        desired="paused",
        finalization="requested",
        active=True,
        age=1.0,
    )
    for report in (first, second):
        _set_finalizer_adoptable(report)
    decision = watchdog.decide(
        (first, second),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert decision.action == "finalizer-reconcile"


def test_watchdog_replaces_exact_cancelled_finalizer_successor():
    first = _status(
        10.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    second = _status(
        70.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    for report in (first, second):
        _set_finalizer_terminal(
            report,
            field="successor",
            scheduler_state="CANCELLED",
        )
    decision = watchdog.decide(
        (first, second),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert decision.action == "finalizer-reconcile"


def test_watchdog_fails_closed_if_successor_executes_while_publisher_is_live():
    first = _status(
        10.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    second = _status(
        70.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    for report in (first, second):
        _set_finalizer_terminal(
            report,
            field="successor",
            scheduler_state="FAILED",
        )
    with pytest.raises(watchdog.WatchdogError, match="publisher remains live"):
        watchdog.decide(
            (first, second),
            expected_release_id=RELEASE,
            expected_git_commit=COMMIT,
            expected_control_sha256=CONTROL,
        )


def test_watchdog_waits_for_exact_finalizer_visibility_grace():
    first = _status(
        10.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    second = _status(
        70.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    for report in (first, second):
        _set_finalizer_missing(
            report,
            field="successor",
            visibility_pending=True,
        )
    decision = watchdog.decide(
        (first, second),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert decision.action is None
    assert "visibility grace" in decision.reason


@pytest.mark.parametrize(
    "mutator",
    [
        lambda report: report["finalization"].update(
            {"scheduler_complete": False}
        ),
        lambda report: report["finalization"].update(
            {
                "namespace_active_job_ids": ["901", "902", "999"],
                "unexpected_active_job_ids": ["999"],
                "chain_healthy": False,
            }
        ),
    ],
)
def test_watchdog_fails_closed_on_incomplete_or_untrusted_finalizer_truth(
    mutator,
):
    first = _status(
        10.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    second = _status(
        70.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    mutator(first)
    mutator(second)
    with pytest.raises(watchdog.WatchdogError):
        watchdog.decide(
            (first, second),
            expected_release_id=RELEASE,
            expected_git_commit=COMMIT,
            expected_control_sha256=CONTROL,
        )


def test_watchdog_rejects_changed_or_ambiguous_finalizer_identity():
    first = _status(
        10.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    changed = _status(
        70.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    changed["finalization"]["successor_job"]["job_id"] = "903"
    changed["finalization"]["successor"]["recorded_job_id"] = "903"
    changed["finalization"]["successor"]["scheduler_job_id"] = "903"
    changed["finalization"]["successor"]["provenance"]["job_id"] = "903"
    changed["finalization"]["namespace_active_job_ids"] = ["901", "903"]
    with pytest.raises(watchdog.WatchdogError, match="identity changed"):
        watchdog.decide(
            (first, changed),
            expected_release_id=RELEASE,
            expected_git_commit=COMMIT,
            expected_control_sha256=CONTROL,
        )

    malformed = _status(
        70.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    malformed["finalization"]["successor_job"]["job_token"] = (
        malformed["finalization"]["active_job"]["job_token"]
    )
    with pytest.raises(watchdog.WatchdogError, match="identity is invalid"):
        watchdog.decide(
            (first, malformed),
            expected_release_id=RELEASE,
            expected_git_commit=COMMIT,
            expected_control_sha256=CONTROL,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_transport",
        "wrong_transport",
        "argv_mismatch",
        "argv_hash_mismatch",
        "persistent_hold",
        "relative_receipt",
        "receipt_hash_mismatch",
        "missing_release",
    ),
)
def test_watchdog_rejects_malformed_finalizer_spool_release_provenance(
    mutation,
):
    first = _status(
        10.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    second = _status(
        70.0,
        desired="paused",
        finalization="draining",
        finalizer_chain=True,
    )
    record = second["finalization"]["successor_job"]
    if mutation == "missing_transport":
        record.pop("submission_transport")
    elif mutation == "wrong_transport":
        record["submission_transport"] = "path_reopened"
    elif mutation == "argv_mismatch":
        record["submission_argv"][-1] = "--dependency=afterany:999"
    elif mutation == "argv_hash_mismatch":
        record["submission_argv_sha256"] = "0" * 64
    elif mutation == "persistent_hold":
        record["persistent_hold"] = True
    elif mutation == "relative_receipt":
        record["spooled_receipt_path"] = "relative.json"
    elif mutation == "receipt_hash_mismatch":
        record["spooled_receipt_sha256"] = "not-a-sha256"
    else:
        record["released_at"] = None
        record["released_timestamp"] = None
    with pytest.raises(
        watchdog.WatchdogError,
        match="identity is (?:incomplete|invalid)",
    ):
        watchdog.decide(
            (first, second),
            expected_release_id=RELEASE,
            expected_git_commit=COMMIT,
            expected_control_sha256=CONTROL,
        )


@pytest.mark.parametrize(
    "first,second",
    [
        (_status(10.0), _status(69.0)),
        (_status(10.0, successor="12"), _status(70.0, successor="12")),
        (_status(10.0, active=True, age=1.0), _status(70.0, active=True, age=1.0)),
    ],
)
def test_watchdog_refuses_insufficient_or_already_recovering_state(first, second):
    if second["captured_timestamp"] - first["captured_timestamp"] < 60:
        with pytest.raises(watchdog.WatchdogError):
            watchdog.decide(
                (first, second),
                expected_release_id=RELEASE,
                expected_git_commit=COMMIT,
                expected_control_sha256=CONTROL,
            )
    else:
        decision = watchdog.decide(
            (first, second),
            expected_release_id=RELEASE,
            expected_git_commit=COMMIT,
            expected_control_sha256=CONTROL,
        )
        assert decision.action is None


def test_watchdog_never_resumes_paused_or_blocked_state():
    paused = watchdog.decide(
        (
            _status(10.0, desired="paused"),
            _status(70.0, desired="paused"),
        ),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert paused.action is None
    blocked = watchdog.decide(
        (
            _status(10.0, desired="paused", finalization="blocked"),
            _status(70.0, desired="paused", finalization="blocked"),
        ),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert blocked.action is None


def test_watchdog_paused_exception_is_only_one_sealed_active_drill():
    first = _status(10.0, desired="paused")
    second = _status(70.0, desired="paused")
    drill = {
        "schema_version": 1,
        "protocol": watchdog.WATCHDOG_DRILL_PROTOCOL,
        "active": True,
        "intent_id": "d" * 64,
        "drill_id": "isolated-drill",
        "phase": "cancelled",
        "created_timestamp": 1.0,
        "expires_timestamp": 1_000.0,
        "maximum_recovery_seconds": 900.0,
        "admission_zero": True,
        "production_baseline_unchanged": True,
        "namespace_job_ids": ["101", "102", "103", "104"],
    }
    first["watchdog_drill"] = dict(drill)
    second["watchdog_drill"] = dict(drill)
    decision = watchdog.decide(
        (first, second),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert decision.action == "repair-chain"
    assert decision.desired_state == "paused"

    second["watchdog_drill"]["production_baseline_unchanged"] = False
    with pytest.raises(watchdog.WatchdogError, match="fail closed"):
        watchdog.decide(
            (first, second),
            expected_release_id=RELEASE,
            expected_git_commit=COMMIT,
            expected_control_sha256=CONTROL,
        )


def test_watchdog_repairs_after_scheduler_proves_recorded_successors_are_gone():
    first = _status(10.0, successor=None)
    second = _status(70.0, successor=None)
    for report, state in ((first, "CANCELLED"), (second, None)):
        for role in watchdog.CONTROLLER_ROLES:
            report["controllers"][role].update(
                {
                    "recorded_successor_job_id": "42",
                    "successor_live": False,
                    "successor_scheduler_state": state,
                }
            )
    decision = watchdog.decide(
        (first, second),
        expected_release_id=RELEASE,
        expected_git_commit=COMMIT,
        expected_control_sha256=CONTROL,
    )
    assert decision.action == "repair-chain"


def test_watchdog_rejects_ambiguous_nonlive_successor_state():
    first = _status(10.0, successor=None)
    second = _status(70.0, successor=None)
    for report in (first, second):
        for role in watchdog.CONTROLLER_ROLES:
            report["controllers"][role].update(
                {
                    "recorded_successor_job_id": "42",
                    "successor_live": False,
                    "successor_scheduler_state": "CONFIGURING",
                }
            )
    with pytest.raises(watchdog.WatchdogError, match="successor scheduler state"):
        watchdog.decide(
            (first, second),
            expected_release_id=RELEASE,
            expected_git_commit=COMMIT,
            expected_control_sha256=CONTROL,
        )


def _forced_command_fixture(tmp_path):
    release = tmp_path / "release"
    repository = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/build_schema5_watchdog_deployment.py",
        "src/agents_scaling/serving/external_watchdog.py",
    ):
        source = repository / relative
        target = release / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o444)
    control_script = release / "slurm" / "schema5_control.py"
    control_script.parent.mkdir(parents=True)
    control_script.write_text("# fixture\n", encoding="utf-8")
    control_script.chmod(0o444)
    prefix = tmp_path / "harness"
    python = prefix / "bin" / "python"
    target = python.with_name("python3.11")
    python.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o555)
    python.symlink_to(target.name)
    critical = prefix / "lib" / "critical.py"
    critical.parent.mkdir()
    critical.write_text("VALUE = 1\n", encoding="utf-8")
    critical.chmod(0o444)
    critical.parent.chmod(0o555)
    python.parent.chmod(0o555)
    prefix.chmod(0o555)
    entries = [
        {
            "path": "bin",
            "type": "directory",
            "mode": python.parent.stat().st_mode & 0o7777,
        },
        {
            "path": "bin/python",
            "type": "symlink",
            "mode": python.lstat().st_mode & 0o7777,
            "target": target.name,
        },
        {
            "path": "bin/python3.11",
            "type": "file",
            "mode": target.stat().st_mode & 0o7777,
            "size": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        {
            "path": "lib",
            "type": "directory",
            "mode": critical.parent.stat().st_mode & 0o7777,
        },
        {
            "path": "lib/critical.py",
            "type": "file",
            "mode": critical.stat().st_mode & 0o7777,
            "size": critical.stat().st_size,
            "sha256": hashlib.sha256(critical.read_bytes()).hexdigest(),
        },
    ]
    inventory = {
        "entries": entries,
        "inventory_sha256": hashlib.sha256(
            deployment._compact_canonical(entries)
        ).hexdigest(),
        "entry_count": 5,
        "file_count": 2,
        "directory_count": 2,
        "symlink_count": 1,
        "total_file_bytes": target.stat().st_size + critical.stat().st_size,
    }
    manifest = release / "harness_environment.schema5-v1.json"
    manifest.write_bytes(
        deployment._canonical(
            {
                "schema_version": 3,
                "release_id": deployment.RELEASE_ID,
                "role": "harness",
                "prefix": str(prefix),
                "sealed_read_only": True,
                "directory_inventory": inventory,
            }
        )
    )
    manifest.chmod(0o444)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    state = tmp_path / "state"
    state.mkdir()
    (state / "control.json").write_bytes(
        deployment._canonical(
            {
                "immutable_sha256": CONTROL,
                "immutable": {
                    "harness_environment_prefix": str(prefix),
                    "harness_environment_manifest_path": str(manifest),
                    "harness_environment_sha256": manifest_sha256,
                },
                "desired_state": "paused",
                "drain_requested": False,
                "finalization": {"state": "idle"},
            }
        )
    )
    for directory in (
        release / "scripts",
        release / "src" / "agents_scaling" / "serving",
        control_script.parent,
        release,
    ):
        directory.chmod(0o555)
    return {
        "state_dir": state,
        "release_root": release,
        "harness_python": python,
        "resolved_harness_python": target,
        "harness_environment_manifest": manifest,
        "harness_environment_sha256": manifest_sha256,
        "control_sha256": CONTROL,
    }


def test_forced_command_has_four_exact_selectors(tmp_path):
    fixture = _forced_command_fixture(tmp_path)
    command = forced.command_for(
        "schema5-watchdog finalizer-reconcile",
        **fixture,
    )
    assert command[0] == str(fixture["resolved_harness_python"])
    assert command[-1] == "watchdog-finalizer-reconcile"
    observe = forced.command_for("schema5-watchdog observe", **fixture)
    assert observe[-1] == "watchdog-observe"
    assert set(forced.ALLOWED) == {
        "schema5-watchdog status",
        "schema5-watchdog observe",
        "schema5-watchdog repair-chain",
        "schema5-watchdog finalizer-reconcile",
    }
    with pytest.raises(forced.ForcedCommandError):
        forced.command_for(
            "schema5-watchdog resume",
            **fixture,
        )


def test_forced_command_rejects_mutated_resolved_python(tmp_path):
    fixture = _forced_command_fixture(tmp_path)
    fixture["resolved_harness_python"].chmod(0o755)
    with pytest.raises(forced.ForcedCommandError, match="verification failed"):
        forced.command_for("schema5-watchdog status", **fixture)


def test_forced_command_rejects_same_mode_environment_file_replacement(
    tmp_path,
):
    fixture = _forced_command_fixture(tmp_path)
    critical = fixture["harness_python"].parents[1] / "lib" / "critical.py"
    critical.chmod(0o644)
    critical.write_text("VALUE = 2\n", encoding="utf-8")
    critical.chmod(0o444)
    with pytest.raises(forced.ForcedCommandError, match="inventory drifted"):
        forced.command_for("schema5-watchdog status", **fixture)


def test_forced_command_rejects_uninventoried_environment_entry(tmp_path):
    fixture = _forced_command_fixture(tmp_path)
    library = fixture["harness_python"].parents[1] / "lib"
    library.chmod(0o755)
    extra = library / "injected.py"
    extra.write_text("VALUE = 1\n", encoding="utf-8")
    extra.chmod(0o444)
    library.chmod(0o555)
    with pytest.raises(forced.ForcedCommandError, match="inventory drifted"):
        forced.command_for("schema5-watchdog status", **fixture)


def test_bootstrap_forced_command_has_separate_two_selector_authority(
    tmp_path, monkeypatch,
):
    release = tmp_path / "release"
    scripts = release / "scripts"
    scripts.mkdir(parents=True)
    renderer = scripts / "render_schema5_recovery_chain_v12.py"
    renderer.write_text("# sealed renderer\n", encoding="utf-8")
    renderer.chmod(0o444)
    for name in (
        "build_schema5_watchdog_deployment.py",
        "schema5_watchdog_forced_command.py",
        "schema5_bootstrap_watchdog.py",
    ):
        source = scripts / name
        source.write_text(f"# sealed {name}\n", encoding="utf-8")
        source.chmod(0o444)
    watchdog_runtime = (
        release
        / "src"
        / "agents_scaling"
        / "serving"
        / "external_watchdog.py"
    )
    watchdog_runtime.parent.mkdir(parents=True)
    watchdog_runtime.write_text("# sealed external runtime\n", encoding="utf-8")
    watchdog_runtime.chmod(0o444)
    renderer_import = (
        release
        / "src"
        / "agents_scaling"
        / "experiment"
        / "completion.py"
    )
    renderer_import.parent.mkdir(parents=True)
    renderer_import.write_text(
        "# sealed renderer import\n", encoding="utf-8"
    )
    renderer_import.chmod(0o444)
    cached_bytecode = (
        scripts / "__pycache__" / "renderer-import.cpython-311.pyc"
    )
    cached_bytecode.parent.mkdir()
    cached_bytecode.write_bytes(b"sealed-bytecode-fixture")
    cached_bytecode.chmod(0o444)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o555)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(deployment._canonical({"chain_id": "c" * 64}))
    manifest.chmod(0o444)
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(deployment._canonical({"receipt_id": "d" * 64}))
    receipt.chmod(0o444)
    isolated_root = tmp_path / "isolated_cancellation_drill"
    isolated_root.mkdir()
    isolated_manifest = isolated_root / manifest.name
    isolated_manifest.write_bytes(
        deployment._canonical({"chain_id": "e" * 64})
    )
    isolated_manifest.chmod(0o444)
    isolated_receipt = isolated_root / receipt.name
    isolated_receipt.write_bytes(
        deployment._canonical({"receipt_id": "f" * 64})
    )
    isolated_receipt.chmod(0o444)
    environment_manifest = tmp_path / "harness.json"
    environment_manifest.write_bytes(deployment._canonical({"sealed": True}))
    environment_manifest.chmod(0o444)
    pilot_marker = tmp_path / "PILOT_COMPLETE.json"
    pilot = {
        "kind": "schema5-materialization-pilot-completion",
        "complete": True,
        "layout": {"harness_prefix": str(tmp_path)},
    }
    pilot["pilot_id"] = hashlib.sha256(
        deployment._compact_canonical(pilot)
    ).hexdigest()
    pilot_marker.write_bytes(deployment._canonical(pilot))
    pilot_marker.chmod(0o444)
    monkeypatch.setattr(
        forced,
        "_load_deployment_runtime",
        lambda _release: type(
            "Runtime",
            (),
            {
                "_resolve_inventory_pinned_harness_python": staticmethod(
                    lambda **_kwargs: (
                        python,
                        {
                            "manifest_path": str(environment_manifest),
                            "manifest_sha256": hashlib.sha256(
                                environment_manifest.read_bytes()
                            ).hexdigest(),
                        },
                    )
                )
            },
        ),
    )
    for directory in sorted(
        (path for path in release.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    release.chmod(0o555)
    kwargs = {
        "release_root": release,
        "harness_python": python,
        "resolved_harness_python": python,
        "harness_environment_manifest": environment_manifest,
        "harness_environment_sha256": hashlib.sha256(
            environment_manifest.read_bytes()
        ).hexdigest(),
        "materialization_pilot_marker": pilot_marker,
        "materialization_pilot_sha256": hashlib.sha256(
            pilot_marker.read_bytes()
        ).hexdigest(),
        "materialization_pilot_id": pilot["pilot_id"],
        "chain_manifest": manifest,
        "chain_manifest_sha256": hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
        "submission_receipt": receipt,
        "submission_receipt_sha256": hashlib.sha256(
            receipt.read_bytes()
        ).hexdigest(),
        "isolated_drill_chain_manifest": isolated_manifest,
        "isolated_drill_chain_manifest_sha256": hashlib.sha256(
            isolated_manifest.read_bytes()
        ).hexdigest(),
        "isolated_drill_submission_receipt": isolated_receipt,
        "isolated_drill_submission_receipt_sha256": hashlib.sha256(
            isolated_receipt.read_bytes()
        ).hexdigest(),
    }
    code_records = forced._bootstrap_tagged_source_records(release)
    code_by_path = {item["path"]: item for item in code_records}
    assert (
        "scripts/__pycache__/renderer-import.cpython-311.pyc"
        in code_by_path
    )
    kwargs.update(
        {
            "bootstrap_runtime_sha256": code_by_path[
                "scripts/schema5_bootstrap_watchdog.py"
            ]["sha256"],
            "forced_command_sha256": code_by_path[
                "scripts/schema5_watchdog_forced_command.py"
            ]["sha256"],
            "renderer_sha256": code_by_path[
                "scripts/render_schema5_recovery_chain_v12.py"
            ]["sha256"],
            "deployment_verifier_sha256": code_by_path[
                "scripts/build_schema5_watchdog_deployment.py"
            ]["sha256"],
            "tagged_source_inventory_sha256": hashlib.sha256(
                deployment._canonical(code_records)
            ).hexdigest(),
        }
    )
    status = forced.bootstrap_command_for(
        "schema5-bootstrap-watchdog status", **kwargs
    )
    repair = forced.bootstrap_command_for(
        "schema5-bootstrap-watchdog repair", **kwargs
    )
    drill_status = forced.bootstrap_command_for(
        "schema5-bootstrap-watchdog drill-status", **kwargs
    )
    drill_repair = forced.bootstrap_command_for(
        "schema5-bootstrap-watchdog drill-repair", **kwargs
    )
    assert "bootstrap-status" in status
    assert "bootstrap-repair" in repair
    assert repair[-1] == "--apply"
    assert str(isolated_manifest) in drill_status
    assert "bootstrap-status" in drill_status
    assert str(isolated_manifest) in drill_repair
    assert drill_repair[-2:] == [
        "--result-output",
        str(isolated_root / "BOOTSTRAP_REPAIR_RESULT.json"),
    ]
    assert all("release-root" not in value for value in repair)
    assert all("release-root" not in value for value in drill_repair)
    with pytest.raises(forced.ForcedCommandError):
        forced.bootstrap_command_for(
            "schema5-bootstrap-watchdog release-root", **kwargs
        )
    renderer.chmod(0o644)
    renderer.write_text("# same mode replacement\n", encoding="utf-8")
    renderer.chmod(0o444)
    with pytest.raises(
        forced.ForcedCommandError, match="tagged source binding drifted"
    ):
        forced.bootstrap_command_for(
            "schema5-bootstrap-watchdog status", **kwargs
        )
    renderer.chmod(0o644)
    renderer.write_text("# sealed renderer\n", encoding="utf-8")
    renderer.chmod(0o444)
    watchdog_runtime.chmod(0o644)
    watchdog_runtime.write_text(
        "# same-mode imported runtime replacement\n", encoding="utf-8"
    )
    watchdog_runtime.chmod(0o444)
    with pytest.raises(
        forced.ForcedCommandError,
        match="tagged source inventory binding drifted",
    ):
        forced.bootstrap_command_for(
            "schema5-bootstrap-watchdog status", **kwargs
        )
    watchdog_runtime.chmod(0o644)
    watchdog_runtime.write_text(
        "# sealed external runtime\n", encoding="utf-8"
    )
    watchdog_runtime.chmod(0o444)
    renderer_import.chmod(0o644)
    renderer_import.write_text(
        "# same-mode renderer import replacement\n", encoding="utf-8"
    )
    renderer_import.chmod(0o444)
    with pytest.raises(
        forced.ForcedCommandError,
        match="tagged source inventory binding drifted",
    ):
        forced.bootstrap_command_for(
            "schema5-bootstrap-watchdog status", **kwargs
        )


@pytest.mark.parametrize(
    ("selector", "desired_state", "drain_requested", "finalization_state"),
    (
        ("schema5-watchdog status", "running", False, "idle"),
        ("schema5-watchdog repair-chain", "running", False, "idle"),
        (
            "schema5-watchdog finalizer-reconcile",
            "paused",
            True,
            "snapshotting",
        ),
    ),
)
def test_forced_command_revalidates_runtime_in_operational_states(
    tmp_path,
    selector,
    desired_state,
    drain_requested,
    finalization_state,
):
    fixture = _forced_command_fixture(tmp_path)
    control_path = fixture["state_dir"] / "control.json"
    control_value = json.loads(control_path.read_text(encoding="utf-8"))
    control_value["desired_state"] = desired_state
    control_value["drain_requested"] = drain_requested
    control_value["finalization"]["state"] = finalization_state
    control_path.write_bytes(deployment._canonical(control_value))
    command = forced.command_for(selector, **fixture)
    assert command[0] == str(fixture["resolved_harness_python"])
    assert command[-1] == forced.ALLOWED[selector][-1]


def test_external_runner_persists_action_and_heartbeat(tmp_path):
    identity = tmp_path / "id"
    known_hosts = tmp_path / "known_hosts"
    identity.write_text("fixture", encoding="utf-8")
    known_hosts.write_text("fixture", encoding="utf-8")
    config = {
        "schema_version": 1,
        "protocol": watchdog.WATCHDOG_PROTOCOL,
        "release_id": RELEASE,
        "git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "control_sha256": CONTROL,
        "remote": {
            "host": "cluster.example",
            "user": "watchdog",
            "identity_file": str(identity),
            "known_hosts_file": str(known_hosts),
        },
        "state_root": str(tmp_path / "state"),
        "liveness_email": "mabdel03@mit.edu",
        "interval_seconds": 300,
        "observation_gap_seconds": 60,
        "stale_seconds": 600,
    }
    first = _status(10.0)
    second = _status(70.0)
    action_binding = {
        "sequence": 1,
        "path": "/cluster/actions/1.json",
        "sha256": "8" * 64,
        "action": "repair-chain",
        "action_receipt_id": "9" * 64,
        "cluster_server_timestamp": 71.0,
    }
    action_result = {
        "result": {
            "desired_state": "running",
            "submitted": [{"role": "dispatcher"}],
        },
        "watchdog_action_receipt": {
            key: action_binding[key]
            for key in ("path", "sha256", "action", "action_receipt_id")
        },
    }
    responses = [
        first,
        second,
        action_result,
        _cluster_mirror_response(
            first, second, action_binding=action_binding
        ),
    ]

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, value):
            self.stdout = json.dumps(value)

    def fake_run(*_args, **_kwargs):
        return Result(responses.pop(0))

    mail_calls = []

    def fake_mail(argv, **kwargs):
        mail_calls.append((argv, kwargs))
        return Result({})

    result = runner.run_once(
        config,
        runner=fake_run,
        mail_runner=fake_mail,
        sleeper=lambda _seconds: None,
        now=lambda: 80.0,
    )
    assert result["action"] == "repair-chain"
    assert result["liveness_email_sent"] is True
    assert len(mail_calls) == 1
    assert (tmp_path / "state" / "WATCHDOG_HEARTBEAT.json").is_file()
    assert len(
        (tmp_path / "state" / "watchdog_actions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 1

    paused_first = _status(100.0, desired="paused")
    paused_second = _status(160.0, desired="paused")
    responses.extend(
        [
            paused_first,
            paused_second,
            _cluster_mirror_response(
                paused_first, paused_second, sequence=2
            ),
        ]
    )
    again = runner.run_once(
        config,
        runner=fake_run,
        mail_runner=fake_mail,
        sleeper=lambda _seconds: None,
        now=lambda: 100.0,
    )
    assert again["liveness_email_sent"] is False
    assert len(mail_calls) == 1


def test_external_runner_adopts_lost_action_reply_then_mirrors_current_cuts(
    tmp_path,
):
    state = tmp_path / "state"
    state.mkdir()
    config = {
        "release_id": RELEASE,
        "git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "control_sha256": CONTROL,
        "remote": {
            "host": "cluster.example",
            "user": "watchdog",
            "identity_file": "/fixed/id",
            "known_hosts_file": "/fixed/known-hosts",
        },
        "state_root": str(state),
        "liveness_email": "mabdel03@mit.edu",
        "observation_gap_seconds": 60,
        "stale_seconds": 600,
    }
    old_first = _status(10.0)
    old_second = _status(70.0)
    current_first = _status(400.0, active=True, age=1.0)
    current_second = _status(460.0, active=True, age=1.0)
    pending_action = {
        "sequence": 1,
        "path": "/cluster/actions/1.json",
        "sha256": "8" * 64,
        "action": "repair-chain",
        "action_receipt_id": "9" * 64,
        "cluster_server_timestamp": 71.0,
    }
    responses = [
        current_first,
        current_second,
        _cluster_mirror_response(
            old_first,
            old_second,
            action_binding=pending_action,
        ),
        _cluster_mirror_response(
            current_first, current_second, sequence=2
        ),
    ]
    selectors = []

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, value):
            self.stdout = json.dumps(value)

    def fake_run(argv, **_kwargs):
        selectors.append(argv[-1])
        return Result(responses.pop(0))

    result = runner.run_once(
        config,
        runner=fake_run,
        mail_runner=lambda *_args, **_kwargs: Result({}),
        sleeper=lambda _seconds: None,
        now=lambda: 461.0,
    )
    assert selectors == [
        "schema5-watchdog status",
        "schema5-watchdog status",
        "schema5-watchdog observe",
        "schema5-watchdog observe",
    ]
    assert result["action"] is None
    assert result["action_executed"] is False
    assert (
        result["cluster_mirror"]["recovered_prior_action"][
            "action_receipt_id"
        ]
        == pending_action["action_receipt_id"]
    )
    assert (
        result["cluster_mirror"]["current"]["recovered_prior_action"]
        is False
    )


@pytest.mark.parametrize(
    ("current_finalization", "requested_action"),
    [
        ("idle", "repair-chain"),
        ("requested", "finalizer-reconcile"),
    ],
)
def test_external_runner_adopts_pending_action_without_reexecution(
    tmp_path, current_finalization, requested_action
):
    state = tmp_path / "state"
    state.mkdir()
    config = {
        "release_id": RELEASE,
        "git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "control_sha256": CONTROL,
        "remote": {
            "host": "cluster.example",
            "user": "watchdog",
            "identity_file": "/fixed/id",
            "known_hosts_file": "/fixed/known-hosts",
        },
        "state_root": str(state),
        "liveness_email": "mabdel03@mit.edu",
        "observation_gap_seconds": 60,
        "stale_seconds": 600,
    }
    old_first = _status(10.0)
    old_second = _status(70.0)
    current_first = _status(400.0, finalization=current_finalization)
    current_second = _status(460.0, finalization=current_finalization)
    pending_action = {
        "sequence": 1,
        "path": "/cluster/actions/1.json",
        "sha256": "8" * 64,
        "action": "repair-chain",
        "action_receipt_id": "9" * 64,
        "cluster_server_timestamp": 71.0,
    }
    deferred = {
        "requested_action": requested_action,
        "executed": False,
        "result": {"deferred": True},
        "watchdog_action_receipt": pending_action,
    }
    responses = [
        current_first,
        current_second,
        deferred,
        _cluster_mirror_response(
            old_first,
            old_second,
            action_binding=pending_action,
        ),
        _cluster_mirror_response(
            current_first, current_second, sequence=2
        ),
    ]
    selectors = []

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, value):
            self.stdout = json.dumps(value)

    def fake_run(argv, **_kwargs):
        selectors.append(argv[-1])
        return Result(responses.pop(0))

    result = runner.run_once(
        config,
        runner=fake_run,
        mail_runner=lambda *_args, **_kwargs: Result({}),
        sleeper=lambda _seconds: None,
        now=lambda: 461.0,
    )
    assert selectors.count(f"schema5-watchdog {requested_action}") == 1
    assert result["action"] == requested_action
    assert result["action_executed"] is False
    assert result["cluster_mirror"]["current"]["receipt_id"]


def test_external_runner_accepts_incomplete_intent_recovered_after_current_cuts(
    tmp_path,
):
    state = tmp_path / "state"
    state.mkdir()
    config = {
        "release_id": RELEASE,
        "git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "control_sha256": CONTROL,
        "remote": {
            "host": "cluster.example",
            "user": "watchdog",
            "identity_file": "/fixed/id",
            "known_hosts_file": "/fixed/known-hosts",
        },
        "state_root": str(state),
        "liveness_email": "mabdel03@mit.edu",
        "observation_gap_seconds": 60,
        "stale_seconds": 600,
    }
    first = _status(400.0)
    second = _status(460.0)
    recovered = {
        "sequence": 1,
        "path": "/cluster/actions/1.json",
        "sha256": "8" * 64,
        "action": "repair-chain",
        "action_receipt_id": "9" * 64,
        "cluster_server_timestamp": 461.0,
        "outcome": "recovered_interrupted",
    }
    deferred = {
        "requested_action": "repair-chain",
        "executed": False,
        "result": {"deferred": True},
        "watchdog_action_receipt": recovered,
    }
    responses = [
        first,
        second,
        deferred,
        _cluster_mirror_response(
            first, second, action_binding=recovered
        ),
    ]
    selectors = []

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, value):
            self.stdout = json.dumps(value)

    def fake_run(argv, **_kwargs):
        selectors.append(argv[-1])
        return Result(responses.pop(0))

    result = runner.run_once(
        config,
        runner=fake_run,
        mail_runner=lambda *_args, **_kwargs: Result({}),
        sleeper=lambda _seconds: None,
        now=lambda: 462.0,
    )
    assert selectors == [
        "schema5-watchdog status",
        "schema5-watchdog status",
        "schema5-watchdog repair-chain",
        "schema5-watchdog observe",
    ]
    current = result["cluster_mirror"]["current"]
    assert current["recovered_current_interrupted"] is True
    assert current["recovered_prior_action"] is False
    assert result["action_executed"] is False


def test_external_runner_never_commits_vm_success_when_cluster_observe_fails(
    tmp_path,
):
    state = tmp_path / "state"
    state.mkdir()
    config = {
        "release_id": RELEASE,
        "git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "control_sha256": CONTROL,
        "remote": {
            "host": "cluster.example",
            "user": "watchdog",
            "identity_file": "/fixed/id",
            "known_hosts_file": "/fixed/known-hosts",
        },
        "state_root": str(state),
        "liveness_email": "mabdel03@mit.edu",
        "observation_gap_seconds": 60,
        "stale_seconds": 600,
    }
    first = _status(10.0)
    second = _status(70.0)
    action_binding = {
        "path": "/cluster/actions/1.json",
        "sha256": "8" * 64,
        "action": "repair-chain",
        "action_receipt_id": "9" * 64,
    }
    responses = [
        (0, first),
        (0, second),
        (
            0,
            {
                "requested_action": "repair-chain",
                "executed": True,
                "result": {"submitted": []},
                "watchdog_action_receipt": action_binding,
            },
        ),
        (1, {"error": "cluster mirror unavailable"}),
    ]

    class Result:
        stderr = "synthetic cluster logging failure"

        def __init__(self, returncode, value):
            self.returncode = returncode
            self.stdout = json.dumps(value)

    def fake_run(*_args, **_kwargs):
        return Result(*responses.pop(0))

    with pytest.raises(watchdog.WatchdogError, match="remote observe failed"):
        runner.run_once(
            config,
            runner=fake_run,
            mail_runner=lambda *_args, **_kwargs: pytest.fail(
                "liveness email must wait for cluster logging"
            ),
            sleeper=lambda _seconds: None,
            now=lambda: 80.0,
        )
    assert not (state / "WATCHDOG_HEARTBEAT.json").exists()
    assert not (state / "watchdog_actions.jsonl").exists()


def test_external_runner_config_rejects_symlink_and_unsafe_ssh_components(
    tmp_path,
):
    identity = tmp_path / "id"
    identity.write_text("private", encoding="utf-8")
    identity.chmod(0o600)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("cluster fixture\n", encoding="utf-8")
    known_hosts.chmod(0o644)
    state = tmp_path / "state"
    state.mkdir()
    value = {
        "schema_version": 1,
        "protocol": watchdog.WATCHDOG_PROTOCOL,
        "release_id": RELEASE,
        "git_commit": COMMIT,
        "release_tag_object": TAG_OBJECT,
        "control_sha256": CONTROL,
        "remote": {
            "host": "cluster.example",
            "user": "watchdog",
            "identity_file": str(identity),
            "known_hosts_file": str(known_hosts),
        },
        "state_root": str(state),
        "liveness_email": "mabdel03@mit.edu",
        "interval_seconds": 300,
        "observation_gap_seconds": 60,
        "stale_seconds": 600,
    }
    config = tmp_path / "watchdog.json"
    config.write_bytes(runner._canonical(value))
    config.chmod(0o444)
    assert runner.load_config(config)["remote"]["host"] == "cluster.example"

    alias = tmp_path / "watchdog-alias.json"
    alias.symlink_to(config)
    with pytest.raises(watchdog.WatchdogError, match="symlink"):
        runner.load_config(alias)

    shared = tmp_path / "watchdog-shared.json"
    shared.hardlink_to(config)
    with pytest.raises(watchdog.WatchdogError, match="shared-linked"):
        runner.load_config(config)
    shared.unlink()

    canonical = runner._canonical(value).decode("utf-8")
    duplicate = canonical.replace(
        f'  "git_commit": "{COMMIT}",',
        f'  "git_commit": "{COMMIT}",\n  "git_commit": "{COMMIT}",',
        1,
    )
    config.chmod(0o644)
    config.write_text(duplicate, encoding="utf-8")
    config.chmod(0o444)
    with pytest.raises(watchdog.WatchdogError, match="duplicates JSON key"):
        runner.load_config(config)

    config.chmod(0o644)
    config.write_text(
        canonical.replace('  "stale_seconds": 600', '  "stale_seconds": NaN'),
        encoding="utf-8",
    )
    config.chmod(0o444)
    with pytest.raises(watchdog.WatchdogError, match="non-finite"):
        runner.load_config(config)

    config.chmod(0o644)
    config.write_text(json.dumps(value), encoding="utf-8")
    config.chmod(0o444)
    with pytest.raises(watchdog.WatchdogError, match="not canonical"):
        runner.load_config(config)

    value["remote"]["host"] = "-oProxyCommand=bad"
    config.chmod(0o644)
    config.write_bytes(runner._canonical(value))
    config.chmod(0o444)
    with pytest.raises(watchdog.WatchdogError, match="host is unsafe"):
        runner.load_config(config)


@pytest.mark.parametrize(
    "payload,match",
    [
        ('{"ok": true, "ok": false}', "duplicates JSON key"),
        ('{"value": NaN}', "non-finite"),
    ],
)
def test_external_runner_rejects_ambiguous_remote_json(payload, match):
    class Result:
        returncode = 0
        stderr = ""
        stdout = payload

    config = {
        "remote": {
            "host": "cluster.example",
            "user": "watchdog",
            "identity_file": "/fixed/id",
            "known_hosts_file": "/fixed/known_hosts",
        }
    }
    with pytest.raises(watchdog.WatchdogError, match=match):
        runner._remote_json(
            config,
            "status",
            runner=lambda *_args, **_kwargs: Result(),
        )


def test_liveness_email_failure_uses_persisted_bounded_backoff(tmp_path):
    config = {
        "state_root": str(tmp_path / "state"),
        "liveness_email": "mabdel03@mit.edu",
    }
    heartbeat = {"heartbeat_id": "a" * 64, "observed_timestamp": 1.0, "action": None}

    class Failure:
        returncode = 1
        stderr = "mail unavailable"

    calls = []

    def fail(argv, **kwargs):
        calls.append((argv, kwargs))
        return Failure()

    assert (
        runner._maybe_send_liveness_email(
            config, heartbeat=heartbeat, now=100.0, mail_runner=fail
        )
        is False
    )
    retry_path = (
        tmp_path / "state" / runner.LIVENESS_EMAIL_RETRY_FILENAME
    )
    retry = json.loads(retry_path.read_text(encoding="utf-8"))
    assert retry["attempts"] == 1
    assert retry["backoff_seconds"] == 300
    assert retry["next_eligible_timestamp"] == 400.0

    assert (
        runner._maybe_send_liveness_email(
            config, heartbeat=heartbeat, now=399.0, mail_runner=fail
        )
        is False
    )
    assert len(calls) == 1

    for attempt in range(2, 10):
        retry["next_eligible_timestamp"] = float(attempt * 1_000)
        retry["attempts"] = attempt - 1
        runner._atomic_json(retry_path, retry)
        runner._maybe_send_liveness_email(
            config,
            heartbeat=heartbeat,
            now=float(attempt * 1_000),
            mail_runner=fail,
        )
    retry = json.loads(retry_path.read_text(encoding="utf-8"))
    assert retry["backoff_seconds"] == runner.LIVENESS_EMAIL_MAX_BACKOFF_SECONDS


def test_confirmed_liveness_email_resets_retry_state(tmp_path):
    config = {
        "state_root": str(tmp_path / "state"),
        "liveness_email": "mabdel03@mit.edu",
    }
    heartbeat = {"heartbeat_id": "b" * 64, "observed_timestamp": 2.0, "action": None}
    retry_path = (
        tmp_path / "state" / runner.LIVENESS_EMAIL_RETRY_FILENAME
    )
    runner._atomic_json(
        retry_path,
        {
            "schema_version": 1,
            "status": "retryable",
            "attempts": 4,
            "last_attempt_timestamp": 0.0,
            "backoff_seconds": 2_400,
            "next_eligible_timestamp": 50.0,
            "recipient": config["liveness_email"],
            "heartbeat_id": "c" * 64,
        },
    )

    class Success:
        returncode = 0
        stderr = ""

    assert (
        runner._maybe_send_liveness_email(
            config,
            heartbeat=heartbeat,
            now=100.0,
            mail_runner=lambda *_args, **_kwargs: Success(),
        )
        is True
    )
    retry = json.loads(retry_path.read_text(encoding="utf-8"))
    assert retry["status"] == "confirmed"
    assert retry["attempts"] == 0
    assert retry["next_eligible_timestamp"] == 86_500.0


def test_liveness_retry_clamps_huge_persisted_attempt_count(tmp_path):
    config = {
        "state_root": str(tmp_path / "state"),
        "liveness_email": "mabdel03@mit.edu",
    }
    heartbeat = {"heartbeat_id": "d" * 64, "observed_timestamp": 3.0, "action": None}
    retry_path = (
        tmp_path / "state" / runner.LIVENESS_EMAIL_RETRY_FILENAME
    )
    huge = 10**100
    runner._atomic_json(
        retry_path,
        {
            "schema_version": 1,
            "status": "retryable",
            "attempts": huge,
            "last_attempt_timestamp": 0.0,
            "backoff_seconds": runner.LIVENESS_EMAIL_MAX_BACKOFF_SECONDS,
            "next_eligible_timestamp": 0.0,
            "recipient": config["liveness_email"],
            "heartbeat_id": "e" * 64,
        },
    )

    class Failure:
        returncode = 1
        stderr = "still unavailable"

    assert (
        runner._maybe_send_liveness_email(
            config,
            heartbeat=heartbeat,
            now=100.0,
            mail_runner=lambda *_args, **_kwargs: Failure(),
        )
        is False
    )
    retry = json.loads(retry_path.read_text(encoding="utf-8"))
    assert retry["attempts"] == huge + 1
    assert retry["backoff_seconds"] == runner.LIVENESS_EMAIL_MAX_BACKOFF_SECONDS
