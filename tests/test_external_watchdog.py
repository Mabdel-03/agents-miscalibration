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


def _status(
    captured: float,
    *,
    desired: str = "running",
    finalization: str = "idle",
    successor: str | None = None,
    successor_state: str | None = None,
    active: bool = False,
    age: float | None = 700.0,
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
    return {
        "schema_version": 1,
        "captured_timestamp": captured,
        "release_id": RELEASE,
        "git_commit": COMMIT,
        "immutable_sha256": CONTROL,
        "desired_state": desired,
        "finalization": {"state": finalization},
        "scheduler": {"squeue_ok": True, "sacct_ok": True, "errors": []},
        "controllers": roles,
    }


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
    ]
    inventory = {
        "entries": entries,
        "inventory_sha256": hashlib.sha256(
            deployment._compact_canonical(entries)
        ).hexdigest(),
        "entry_count": 3,
        "file_count": 1,
        "directory_count": 1,
        "symlink_count": 1,
        "total_file_bytes": target.stat().st_size,
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


def test_forced_command_has_three_exact_selectors(tmp_path):
    fixture = _forced_command_fixture(tmp_path)
    command = forced.command_for(
        "schema5-watchdog finalizer-reconcile",
        **fixture,
    )
    assert command[0] == str(fixture["resolved_harness_python"])
    assert command[-1] == "finalizer-reconcile"
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
    responses = [
        _status(10.0),
        _status(70.0),
        {"desired_state": "running", "submitted": [{"role": "dispatcher"}]},
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

    responses.extend(
        [
            _status(100.0, desired="paused"),
            _status(160.0, desired="paused"),
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
