#!/usr/bin/env python3
"""Run one five-minute external schema-5 watchdog transaction.

This program is intended for an institutional VM, not a Slurm node.  SSH is bound to
an account-side forced command, so the strings sent here are selectors rather than
arbitrary remote shell commands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_RELEASE_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_PATH = (
    _RELEASE_ROOT
    / "src"
    / "agents_scaling"
    / "serving"
    / "external_watchdog.py"
)
_RUNTIME = sys.modules.get("agents_scaling.serving.external_watchdog")
if (
    _RUNTIME is None
    or Path(str(getattr(_RUNTIME, "__file__", ""))).resolve()
    != _RUNTIME_PATH.resolve()
):
    _RUNTIME_SPEC = importlib.util.spec_from_file_location(
        "_schema5_external_watchdog_runtime", _RUNTIME_PATH
    )
    if _RUNTIME_SPEC is None or _RUNTIME_SPEC.loader is None:
        raise ImportError(f"cannot load sealed watchdog runtime: {_RUNTIME_PATH}")
    _RUNTIME = importlib.util.module_from_spec(_RUNTIME_SPEC)
    sys.modules[_RUNTIME_SPEC.name] = _RUNTIME
    _RUNTIME_SPEC.loader.exec_module(_RUNTIME)

WATCHDOG_INTERVAL_SECONDS = _RUNTIME.WATCHDOG_INTERVAL_SECONDS
WATCHDOG_OBSERVATION_GAP_SECONDS = _RUNTIME.WATCHDOG_OBSERVATION_GAP_SECONDS
WATCHDOG_PROTOCOL = _RUNTIME.WATCHDOG_PROTOCOL
WATCHDOG_SCHEMA_VERSION = _RUNTIME.WATCHDOG_SCHEMA_VERSION
WATCHDOG_RELEASE_TAG = _RUNTIME.WATCHDOG_RELEASE_TAG
WATCHDOG_CLUSTER_CYCLE_PROTOCOL = _RUNTIME.WATCHDOG_CLUSTER_CYCLE_PROTOCOL
WATCHDOG_CLUSTER_LATEST_PROTOCOL = _RUNTIME.WATCHDOG_CLUSTER_LATEST_PROTOCOL
WatchdogError = _RUNTIME.WatchdogError
decide = _RUNTIME.decide

LIVENESS_EMAIL_RETRY_FILENAME = "LIVENESS_EMAIL_RETRY.json"
LIVENESS_EMAIL_INITIAL_BACKOFF_SECONDS = 300
LIVENESS_EMAIL_MAX_BACKOFF_SECONDS = 21_600
RELEASE_ID = "sweep-recovery-schema5-v1.2"
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SSH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,252}\Z")
_EMAIL = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?\Z"
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        payload = json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_path(path: Path, *, description: str, kind: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if any(character in str(lexical) for character in ("\x00", "\n", "\r")):
        raise WatchdogError(f"{description} path is unsafe")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WatchdogError(f"{description} is unavailable: {exc}") from exc
    if resolved != lexical:
        raise WatchdogError(f"{description} traverses a symlink")
    metadata = lexical.stat(follow_symlinks=False)
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise WatchdogError(f"{description} is not a regular file")
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise WatchdogError(f"{description} is not a directory")
    return lexical


def _stable_bytes(
    path: Path, *, description: str, require_read_only: bool = False
) -> bytes:
    canonical = _canonical_path(path, description=description, kind="file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(canonical, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
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
    current = canonical.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or identity(before) != identity(after)
        or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
        or (require_read_only and stat.S_IMODE(before.st_mode) & 0o222)
        or before.st_nlink != 1
    ):
        raise WatchdogError(
            f"{description} is mutable, shared-linked, or changed while being read"
        )
    return b"".join(chunks)


def _json_object(
    payload: bytes | str,
    *,
    description: str,
    require_canonical: bool,
) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WatchdogError(
                    f"{description} duplicates JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        raise WatchdogError(f"{description} contains non-finite value {token}")

    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except WatchdogError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise WatchdogError(f"{description} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WatchdogError(f"{description} returned a non-object")
    if require_canonical and (
        not isinstance(payload, bytes) or payload != _canonical(value)
    ):
        raise WatchdogError(f"{description} is not canonical JSON")
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = _stable_bytes(
            path,
            description="watchdog configuration",
            require_read_only=True,
        )
        value = _json_object(
            raw,
            description="watchdog configuration",
            require_canonical=True,
        )
    except (OSError, WatchdogError) as exc:
        raise WatchdogError(f"cannot load watchdog configuration: {exc}") from exc
    expected = {
        "schema_version",
        "protocol",
        "release_id",
        "git_commit",
        "release_tag_object",
        "control_sha256",
        "remote",
        "state_root",
        "liveness_email",
        "interval_seconds",
        "observation_gap_seconds",
        "stale_seconds",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise WatchdogError("watchdog configuration fields differ from the closed schema")
    remote = value.get("remote")
    if not isinstance(remote, dict) or set(remote) != {
        "host",
        "user",
        "identity_file",
        "known_hosts_file",
    }:
        raise WatchdogError("watchdog remote configuration is invalid")
    if (
        value.get("schema_version") != WATCHDOG_SCHEMA_VERSION
        or value.get("protocol") != WATCHDOG_PROTOCOL
        or value.get("release_id") != RELEASE_ID
        or value.get("interval_seconds") != WATCHDOG_INTERVAL_SECONDS
        or value.get("observation_gap_seconds")
        != WATCHDOG_OBSERVATION_GAP_SECONDS
        or value.get("stale_seconds") != 600
        or not isinstance(value.get("git_commit"), str)
        or _GIT_COMMIT.fullmatch(value["git_commit"]) is None
        or not isinstance(value.get("release_tag_object"), str)
        or _GIT_COMMIT.fullmatch(value["release_tag_object"]) is None
        or not isinstance(value.get("control_sha256"), str)
        or _SHA256.fullmatch(value["control_sha256"]) is None
        or not isinstance(value.get("liveness_email"), str)
        or _EMAIL.fullmatch(value["liveness_email"]) is None
    ):
        raise WatchdogError("watchdog configuration identity is invalid")
    for field in ("host", "user"):
        candidate = remote.get(field)
        if (
            not isinstance(candidate, str)
            or _SSH_COMPONENT.fullmatch(candidate) is None
            or candidate.startswith("-")
        ):
            raise WatchdogError(f"watchdog remote {field} is unsafe")
    for field in ("identity_file", "known_hosts_file"):
        candidate = Path(str(remote[field])).expanduser()
        if not candidate.is_absolute():
            raise WatchdogError(f"watchdog {field} must be absolute")
        candidate = _canonical_path(
            candidate, description=f"watchdog {field}", kind="file"
        )
        mode = stat.S_IMODE(candidate.stat(follow_symlinks=False).st_mode)
        if field == "identity_file" and mode & 0o077:
            raise WatchdogError("watchdog identity_file permissions are too broad")
        if field == "known_hosts_file" and mode & 0o022:
            raise WatchdogError("watchdog known_hosts_file is group/world writable")
    state_root = Path(str(value["state_root"])).expanduser()
    if not state_root.is_absolute():
        raise WatchdogError("watchdog state_root must be absolute")
    _canonical_path(
        state_root, description="watchdog state_root", kind="directory"
    )
    return value


def _ssh_argv(config: Mapping[str, Any], selector: str) -> list[str]:
    if selector not in {
        "status",
        "observe",
        "repair-chain",
        "finalizer-reconcile",
    }:
        raise WatchdogError(f"watchdog selector is not allowed: {selector}")
    remote = config["remote"]
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        f"UserKnownHostsFile={remote['known_hosts_file']}",
        "-i",
        str(remote["identity_file"]),
        "--",
        f"{remote['user']}@{remote['host']}",
        f"schema5-watchdog {selector}",
    ]


def _remote_json(
    config: Mapping[str, Any],
    selector: str,
    *,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    proc = runner(
        _ssh_argv(config, selector),
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    accepted = {0, 1} if selector == "status" else {0}
    if proc.returncode not in accepted:
        raise WatchdogError(
            f"remote {selector} failed rc={proc.returncode}: {proc.stderr[:500]}"
        )
    try:
        return _json_object(
            proc.stdout,
            description=f"remote {selector} response",
            require_canonical=False,
        )
    except WatchdogError as exc:
        raise WatchdogError(f"remote {selector} returned invalid JSON: {exc}") from exc


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    identity = dict(value)
    identity.pop(field, None)
    return hashlib.sha256(_canonical(identity)).hexdigest()


def _validate_cluster_mirror(
    value: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    action: str | None,
    action_result: Mapping[str, Any] | None,
    allow_recovered_prior_action: bool = False,
) -> dict[str, Any]:
    """Require the cluster journal to bind this exact two-cut VM transaction."""

    if set(value) != {"published", "recovered", "receipt", "pointer"}:
        raise WatchdogError("cluster mirror response fields differ from the closed schema")
    receipt = value.get("receipt")
    pointer = value.get("pointer")
    if (
        value.get("published") is not True
        or not isinstance(value.get("recovered"), bool)
        or not isinstance(receipt, Mapping)
        or not isinstance(pointer, Mapping)
    ):
        raise WatchdogError("cluster mirror response is incomplete")
    expected_receipt_fields = {
        "schema_version",
        "protocol",
        "sequence",
        "predecessor",
        "intent",
        "identity",
        "status_observations",
        "consumed_status_sequence",
        "action_receipt",
        "consumed_action_sequence",
        "cluster_server_timestamp",
        "receipt_id",
    }
    identity = receipt.get("identity")
    finalization = identity.get("finalization") if isinstance(identity, Mapping) else None
    observations = receipt.get("status_observations")
    action_binding = receipt.get("action_receipt")

    def matches_status_identity(
        candidate: Any, status: Mapping[str, Any]
    ) -> bool:
        status_finalization = status.get("finalization")
        candidate_finalization = (
            candidate.get("finalization")
            if isinstance(candidate, Mapping)
            else None
        )
        return bool(
            isinstance(candidate, Mapping)
            and candidate.get("release_id") == config["release_id"]
            and candidate.get("release_tag") == WATCHDOG_RELEASE_TAG
            and candidate.get("release_tag_object")
            == config["release_tag_object"]
            and candidate.get("git_commit") == config["git_commit"]
            and candidate.get("control_sha256") == config["control_sha256"]
            and candidate.get("desired_state") == status.get("desired_state")
            and candidate.get("rollout_generation")
            == status.get("rollout_generation")
            and isinstance(candidate_finalization, Mapping)
            and isinstance(status_finalization, Mapping)
            and candidate_finalization.get("state")
            == status_finalization.get("state")
            and candidate_finalization.get("intent_id")
            == status_finalization.get("intent_id")
        )

    def matches_release_identity(candidate: Any) -> bool:
        return bool(
            isinstance(candidate, Mapping)
            and candidate.get("release_id") == config["release_id"]
            and candidate.get("release_tag") == WATCHDOG_RELEASE_TAG
            and candidate.get("release_tag_object")
            == config["release_tag_object"]
            and candidate.get("git_commit") == config["git_commit"]
            and candidate.get("control_sha256") == config["control_sha256"]
        )

    first_hash = hashlib.sha256(_canonical(dict(first))).hexdigest()
    second_hash = hashlib.sha256(_canonical(dict(second))).hexdigest()
    exact_status_cuts = bool(
        isinstance(observations, list)
        and len(observations) == 2
        and observations[0].get("report_sha256") == first_hash
        and observations[1].get("report_sha256") == second_hash
        and observations[0].get("report_captured_timestamp")
        == first.get("captured_timestamp")
        and observations[1].get("report_captured_timestamp")
        == second.get("captured_timestamp")
    )
    if (
        set(receipt) != expected_receipt_fields
        or receipt.get("schema_version") != 1
        or receipt.get("protocol") != WATCHDOG_CLUSTER_CYCLE_PROTOCOL
        or receipt.get("receipt_id") != _self_hash(receipt, "receipt_id")
        or not isinstance(identity, Mapping)
        or identity.get("release_id") != config["release_id"]
        or identity.get("release_tag") != WATCHDOG_RELEASE_TAG
        or identity.get("release_tag_object") != config["release_tag_object"]
        or identity.get("git_commit") != config["git_commit"]
        or identity.get("control_sha256") != config["control_sha256"]
        or not isinstance(finalization, Mapping)
        or not isinstance(observations, list)
        or len(observations) != 2
    ):
        raise WatchdogError(
            "cluster mirror did not bind the exact release and two status cuts"
        )
    if action_binding is None:
        if not matches_status_identity(identity, second):
            raise WatchdogError(
                "cluster mirror no-action identity differs from cut two"
            )
    else:
        expected_action_fields = {
            "sequence",
            "path",
            "sha256",
            "action",
            "action_receipt_id",
            "cluster_server_timestamp",
            "identity_before",
            "identity_after",
            "outcome",
        }
        if (
            not isinstance(action_binding, Mapping)
            or set(action_binding) != expected_action_fields
            or not (
                (
                    matches_status_identity(
                        action_binding.get("identity_after"), second
                    )
                    if action_binding.get("outcome")
                    == "recovered_interrupted"
                    else matches_status_identity(
                        action_binding.get("identity_before"), second
                    )
                )
                if exact_status_cuts
                else matches_release_identity(
                    action_binding.get("identity_before")
                )
            )
            or action_binding.get("identity_after") != identity
        ):
            raise WatchdogError(
                "cluster mirror action transition identity is invalid"
            )
    action_executed = bool(
        action is not None
        and isinstance(action_result, Mapping)
        and action_result.get("executed", True) is True
    )
    external_binding = (
        action_result.get("watchdog_action_receipt")
        if isinstance(action_result, Mapping)
        else None
    )
    external_binding_matches = bool(
        isinstance(external_binding, Mapping)
        and isinstance(action_binding, Mapping)
        and action_binding.get("action")
        == external_binding.get("action")
        and action_binding.get("action_receipt_id")
        == external_binding.get("action_receipt_id")
        and action_binding.get("sha256")
        == external_binding.get("sha256")
        and action_binding.get("path")
        == external_binding.get("path")
    )
    recovered_current_interrupted = bool(
        exact_status_cuts
        and not action_executed
        and isinstance(action_binding, Mapping)
        and action_binding.get("outcome") == "recovered_interrupted"
        and external_binding_matches
    )
    recovered_prior_action = bool(
        allow_recovered_prior_action
        and not exact_status_cuts
        and isinstance(action_binding, Mapping)
        and (
            action is None
            or not action_executed
        )
        and isinstance(action_binding.get("cluster_server_timestamp"), (int, float))
        and not isinstance(action_binding.get("cluster_server_timestamp"), bool)
        and float(action_binding["cluster_server_timestamp"])
        <= float(first["captured_timestamp"])
        and (
            external_binding is None
            or external_binding_matches
        )
    )
    if recovered_prior_action:
        pass
    elif recovered_current_interrupted:
        pass
    elif not exact_status_cuts:
        raise WatchdogError(
            "cluster mirror did not bind this VM cycle's exact status cuts"
        )
    elif action is None:
        if action_binding is not None:
            raise WatchdogError(
                "cluster mirror bound an action not performed by this VM cycle"
            )
    elif action_executed:
        if (
            not isinstance(external_binding, Mapping)
            or not isinstance(action_binding, Mapping)
            or action_binding.get("action") != action
            or action_binding.get("action_receipt_id")
            != external_binding.get("action_receipt_id")
            or action_binding.get("sha256") != external_binding.get("sha256")
            or action_binding.get("path") != external_binding.get("path")
        ):
            raise WatchdogError(
                "cluster mirror action binding differs from the action receipt"
            )
    else:
        raise WatchdogError(
            "a deferred action must first mirror its prior action receipt"
        )
    expected_pointer_fields = {
        "schema_version",
        "protocol",
        "sequence",
        "receipt_path",
        "receipt_sha256",
        "receipt_id",
        "cluster_server_timestamp",
        "pointer_id",
    }
    if (
        set(pointer) != expected_pointer_fields
        or pointer.get("schema_version") != 1
        or pointer.get("protocol") != WATCHDOG_CLUSTER_LATEST_PROTOCOL
        or pointer.get("sequence") != receipt.get("sequence")
        or pointer.get("receipt_id") != receipt.get("receipt_id")
        or pointer.get("cluster_server_timestamp")
        != receipt.get("cluster_server_timestamp")
        or pointer.get("pointer_id") != _self_hash(pointer, "pointer_id")
        or not isinstance(pointer.get("receipt_path"), str)
        or not Path(str(pointer["receipt_path"])).is_absolute()
        or _SHA256.fullmatch(str(pointer.get("receipt_sha256", ""))) is None
    ):
        raise WatchdogError("cluster mirror latest pointer is invalid")
    return {
        "receipt_id": receipt["receipt_id"],
        "sequence": receipt["sequence"],
        "cluster_server_timestamp": receipt["cluster_server_timestamp"],
        "pointer_id": pointer["pointer_id"],
        "action_receipt_id": (
            action_binding.get("action_receipt_id")
            if isinstance(action_binding, Mapping)
            else None
        ),
        "recovered_prior_action": recovered_prior_action,
        "recovered_current_interrupted": recovered_current_interrupted,
    }


def _maybe_send_liveness_email(
    config: Mapping[str, Any],
    *,
    heartbeat: Mapping[str, Any],
    now: float,
    mail_runner: Any,
) -> bool:
    state_root = Path(str(config["state_root"]))
    receipt_path = state_root / "LIVENESS_EMAIL_CONFIRMED.json"
    retry_path = state_root / LIVENESS_EMAIL_RETRY_FILENAME
    last_confirmed: float | None = None
    if receipt_path.is_file() and not receipt_path.is_symlink():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            receipt = None
        candidate = (
            receipt.get("confirmed_timestamp")
            if isinstance(receipt, Mapping)
            else None
        )
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            last_confirmed = float(candidate)
    if last_confirmed is not None and now - last_confirmed < 86_400:
        return False
    prior_attempts = 0
    if retry_path.is_file() and not retry_path.is_symlink():
        try:
            retry = json.loads(retry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            retry = None
        if isinstance(retry, Mapping):
            next_eligible = retry.get("next_eligible_timestamp")
            attempts = retry.get("attempts")
            if (
                isinstance(next_eligible, (int, float))
                and not isinstance(next_eligible, bool)
                and float(next_eligible) > now
            ):
                return False
            if (
                isinstance(attempts, int)
                and not isinstance(attempts, bool)
                and attempts >= 0
            ):
                prior_attempts = attempts
    recipient = str(config["liveness_email"])
    body = (
        "Schema-5 external watchdog is live.\n"
        f"heartbeat_id={heartbeat['heartbeat_id']}\n"
        f"observed_timestamp={heartbeat['observed_timestamp']}\n"
        f"action={heartbeat['action']}\n"
    )
    proc = mail_runner(
        [
            "mail",
            "-s",
            "[schema5-watchdog] daily liveness",
            recipient,
        ],
        input=body,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        attempts = prior_attempts + 1
        maximum_exponent = max(
            0,
            (
                LIVENESS_EMAIL_MAX_BACKOFF_SECONDS
                // LIVENESS_EMAIL_INITIAL_BACKOFF_SECONDS
            ).bit_length(),
        )
        delay = min(
            LIVENESS_EMAIL_INITIAL_BACKOFF_SECONDS
            * (2 ** min(max(attempts - 1, 0), maximum_exponent)),
            LIVENESS_EMAIL_MAX_BACKOFF_SECONDS,
        )
        _append_jsonl(
            state_root / "liveness_email_failures.jsonl",
            {
                "attempted_timestamp": now,
                "attempts": attempts,
                "backoff_seconds": delay,
                "next_eligible_timestamp": now + delay,
                "recipient": recipient,
                "returncode": int(proc.returncode),
                "stderr": proc.stderr[-500:],
                "heartbeat_id": heartbeat["heartbeat_id"],
            },
        )
        _atomic_json(
            retry_path,
            {
                "schema_version": 1,
                "status": "retryable",
                "attempts": attempts,
                "last_attempt_timestamp": now,
                "backoff_seconds": delay,
                "next_eligible_timestamp": now + delay,
                "recipient": recipient,
                "heartbeat_id": heartbeat["heartbeat_id"],
            },
        )
        return False
    _atomic_json(
        receipt_path,
        {
            "schema_version": 1,
            "confirmed_timestamp": now,
            "recipient": recipient,
            "heartbeat_id": heartbeat["heartbeat_id"],
        },
    )
    _atomic_json(
        retry_path,
        {
            "schema_version": 1,
            "status": "confirmed",
            "attempts": 0,
            "last_attempt_timestamp": now,
            "backoff_seconds": 0,
            "next_eligible_timestamp": now + 86_400,
            "recipient": recipient,
            "heartbeat_id": heartbeat["heartbeat_id"],
        },
    )
    return True


def run_once(
    config: Mapping[str, Any],
    *,
    runner: Any = subprocess.run,
    mail_runner: Any = subprocess.run,
    sleeper: Any = time.sleep,
    now: Any = time.time,
) -> dict[str, Any]:
    first = _remote_json(config, "status", runner=runner)
    sleeper(float(config["observation_gap_seconds"]))
    second = _remote_json(config, "status", runner=runner)
    decision = decide(
        (first, second),
        expected_release_id=str(config["release_id"]),
        expected_git_commit=str(config["git_commit"]),
        expected_control_sha256=str(config["control_sha256"]),
        minimum_gap_seconds=float(config["observation_gap_seconds"]),
        stale_seconds=float(config["stale_seconds"]),
    )
    action_result = (
        _remote_json(config, decision.action, runner=runner)
        if decision.action is not None
        else None
    )
    mirror_response = _remote_json(config, "observe", runner=runner)
    first_mirror = _validate_cluster_mirror(
        mirror_response,
        config=config,
        first=first,
        second=second,
        action=decision.action,
        action_result=action_result,
        allow_recovered_prior_action=True,
    )
    if first_mirror["recovered_prior_action"]:
        current_mirror_response = _remote_json(
            config, "observe", runner=runner
        )
        current_mirror = _validate_cluster_mirror(
            current_mirror_response,
            config=config,
            first=first,
            second=second,
            action=None,
            action_result=None,
        )
        mirror: dict[str, Any] = {
            "current": current_mirror,
            "recovered_prior_action": first_mirror,
        }
    else:
        mirror = {
            "current": first_mirror,
            "recovered_prior_action": None,
        }
    timestamp = float(now())
    record: dict[str, Any] = {
        "schema_version": 1,
        "protocol": WATCHDOG_PROTOCOL,
        "observed_timestamp": timestamp,
        "release_id": config["release_id"],
        "git_commit": config["git_commit"],
        "release_tag_object": config["release_tag_object"],
        "control_sha256": config["control_sha256"],
        "first_status_sha256": hashlib.sha256(_canonical(first)).hexdigest(),
        "second_status_sha256": hashlib.sha256(_canonical(second)).hexdigest(),
        "action": decision.action,
        "action_executed": bool(
            decision.action is not None
            and isinstance(action_result, Mapping)
            and action_result.get("executed", True) is True
        ),
        "reason": decision.reason,
        "desired_state": decision.desired_state,
        "finalization_state": decision.finalization_state,
        "action_result": action_result,
        "cluster_mirror": mirror,
    }
    without_id = dict(record)
    record["heartbeat_id"] = hashlib.sha256(_canonical(without_id)).hexdigest()
    state_root = Path(str(config["state_root"]))
    _append_jsonl(state_root / "watchdog_actions.jsonl", record)
    _atomic_json(state_root / "WATCHDOG_HEARTBEAT.json", record)
    record["liveness_email_sent"] = _maybe_send_liveness_email(
        config,
        heartbeat=record,
        now=timestamp,
        mail_runner=mail_runner,
    )
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config.expanduser().absolute())
    result = run_once(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
