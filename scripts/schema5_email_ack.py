#!/usr/bin/env python3
"""Consume one active schema-5 email challenge exactly once.

The emailed token is never written by the recovery system.  The durable request
contains only a random-salt, random-nonce verifier bound to the immutable release,
the recovery chain, the challenge generation, and the intended recipient.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence


class AcknowledgementError(RuntimeError):
    """The acknowledgement does not match the active immutable email request."""


EMAIL_CHALLENGE_SCHEMA_VERSION = 2
EMAIL_CHALLENGE_VERIFIER_ALGORITHM = "sha256-salted-nonce-bound-v1"
EMAIL_CHALLENGE_LOCK_NAME = ".email-challenge.lock"
EMAIL_CHALLENGE_SUBJECT = (
    "[agents-scaling] schema-5 readiness acknowledgement required"
)
EMAIL_CHALLENGE_BODY_TEMPLATE = "schema5-email-acknowledgement-v2"

_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,128}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_REQUEST_ID = re.compile(r"[0-9a-f]{64}\Z")
_RELEASE_TAG = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_VERIFIER_DOMAIN = b"schema5-email-challenge-verifier-v2\0"
_CHALLENGE_ID_DOMAIN = b"schema5-email-challenge-id-v2\0"

EMAIL_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "passed",
        "immutable_sha256",
        "chain_id",
        "request_id",
        "release_tag",
        "release_git_commit",
        "recipient",
        "challenge_generation",
        "challenge_nonce",
        "challenge_salt",
        "challenge_verifier_algorithm",
        "challenge_verifier",
        "challenge_id",
        "active_challenge",
        "acknowledgement",
        "delivery_succeeded",
        "returncode",
        "delivery_attempts",
        "submitted_timestamp",
        "expires_timestamp",
        "subject",
        "body_template",
        "acknowledgement_tool",
    }
)

ACTIVE_CHALLENGE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "passed",
        "immutable_sha256",
        "chain_id",
        "request_id",
        "release_tag",
        "release_git_commit",
        "recipient",
        "challenge_generation",
        "challenge_id",
        "request",
        "request_sha256",
        "acknowledgement",
        "supersedes",
        "activated_timestamp",
        "active_challenge_id",
    }
)

ACKNOWLEDGEMENT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "passed",
        "immutable_sha256",
        "chain_id",
        "request_id",
        "release_tag",
        "release_git_commit",
        "recipient",
        "challenge_generation",
        "challenge_id",
        "request",
        "request_sha256",
        "operator",
        "acknowledged_at",
        "acknowledged_timestamp",
        "acknowledgement_id",
    }
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def identity_sha256(value: Mapping[str, Any]) -> str:
    """Hash one closed-schema identity with the challenge canonical encoding."""

    return _sha256_bytes(_canonical(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_safe_directory(path: Path, *, description: str) -> Path:
    """Require a canonical existing directory with no symlink in its ancestry."""

    supplied = path.expanduser().absolute()
    if (
        supplied.is_symlink()
        or not supplied.is_dir()
        or supplied.resolve() != supplied
    ):
        raise AcknowledgementError(
            f"{description} is missing or traverses a symlink: {supplied}"
        )
    return supplied


def _read(path: Path, *, description: str) -> dict[str, Any]:
    supplied = path.expanduser().absolute()
    require_safe_directory(
        supplied.parent, description=f"{description} parent"
    )
    if supplied.is_symlink() or not supplied.is_file():
        raise AcknowledgementError(f"{description} is missing or unsafe: {supplied}")
    try:
        payload = json.loads(
            supplied.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AcknowledgementError(
            f"cannot read {description} {supplied}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AcknowledgementError(f"{description} must be one JSON object")
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_output(path: Path) -> Path:
    path = path.expanduser().absolute()
    require_safe_directory(
        path.parent, description="acknowledgement output parent"
    )
    if path.is_symlink():
        raise AcknowledgementError(f"acknowledgement output is a symlink: {path}")
    return path


def _atomic_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete marker with create-if-absent semantics.

    ``os.link`` is the no-overwrite commit point.  A concurrent or replayed
    acknowledgement therefore cannot replace the first valid consumption marker.
    """

    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path = _safe_output(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise AcknowledgementError(
                "email challenge was already consumed; replay is forbidden"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def challenge_lock(active_challenge: Path) -> Iterator[None]:
    """Serialize acknowledgement and challenge supersession across nodes."""

    active_challenge = active_challenge.expanduser().absolute()
    root = require_safe_directory(
        active_challenge.parent, description="email challenge root"
    )
    lock_path = root / EMAIL_CHALLENGE_LOCK_NAME
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _binding(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "immutable_sha256": record["immutable_sha256"],
        "chain_id": record["chain_id"],
        "request_id": record["request_id"],
        "release_tag": record["release_tag"],
        "release_git_commit": record["release_git_commit"],
        "recipient": record["recipient"],
        "challenge_generation": record["challenge_generation"],
        "challenge_nonce": record["challenge_nonce"],
        "challenge_salt": record["challenge_salt"],
    }


def challenge_id_for(record: Mapping[str, Any]) -> str:
    """Return the non-secret identity of one fully bound challenge."""

    return _sha256_bytes(_CHALLENGE_ID_DOMAIN + _canonical(_binding(record)))


def challenge_verifier_for(token: str, record: Mapping[str, Any]) -> str:
    """Return the only durable token verifier.

    The verifier is salted, nonce-bound, and domain-separated.  The 256-bit token
    remains useful only while held by the live mail process or by the recipient.
    """

    if _TOKEN.fullmatch(token) is None:
        raise AcknowledgementError("acknowledgement token has an unsafe format")
    salt = bytes.fromhex(str(record["challenge_salt"]))
    material = (
        _VERIFIER_DOMAIN
        + salt
        + _canonical(_binding(record))
        + b"\0"
        + token.encode("ascii")
    )
    return _sha256_bytes(material)


def validate_request_payload(record: Mapping[str, Any]) -> None:
    if set(record) != EMAIL_REQUEST_FIELDS:
        raise AcknowledgementError(
            "email request fields differ from the closed schema"
        )
    attempts = record.get("delivery_attempts")
    tool = record.get("acknowledgement_tool")
    submitted = record.get("submitted_timestamp")
    expires = record.get("expires_timestamp")
    if (
        record.get("schema_version") != EMAIL_CHALLENGE_SCHEMA_VERSION
        or record.get("kind") != "schema5_email_ack_request"
        or record.get("passed") is not True
        or _SHA256.fullmatch(str(record.get("immutable_sha256", ""))) is None
        or _SHA256.fullmatch(str(record.get("chain_id", ""))) is None
        or _REQUEST_ID.fullmatch(str(record.get("request_id", ""))) is None
        or _RELEASE_TAG.fullmatch(str(record.get("release_tag", ""))) is None
        or _GIT_COMMIT.fullmatch(str(record.get("release_git_commit", ""))) is None
        or not isinstance(record.get("recipient"), str)
        or not record["recipient"]
        or any(character in record["recipient"] for character in "\r\n")
        or not isinstance(record.get("challenge_generation"), int)
        or isinstance(record.get("challenge_generation"), bool)
        or record["challenge_generation"] < 0
        or _SHA256.fullmatch(str(record.get("challenge_nonce", ""))) is None
        or _SHA256.fullmatch(str(record.get("challenge_salt", ""))) is None
        or record.get("challenge_verifier_algorithm")
        != EMAIL_CHALLENGE_VERIFIER_ALGORITHM
        or _SHA256.fullmatch(str(record.get("challenge_verifier", ""))) is None
        or _SHA256.fullmatch(str(record.get("challenge_id", ""))) is None
        or record.get("challenge_id") != challenge_id_for(record)
        or not isinstance(record.get("active_challenge"), str)
        or not record["active_challenge"]
        or not Path(record["active_challenge"]).is_absolute()
        or any(
            character in record["active_challenge"] for character in "\r\n"
        )
        or not isinstance(record.get("acknowledgement"), str)
        or not record["acknowledgement"]
        or not Path(record["acknowledgement"]).is_absolute()
        or any(character in record["acknowledgement"] for character in "\r\n")
        or record.get("delivery_succeeded") is not True
        or record.get("returncode") != 0
        or not isinstance(attempts, list)
        or not attempts
        or not isinstance(submitted, (int, float))
        or isinstance(submitted, bool)
        or submitted < 0
        or not isinstance(expires, (int, float))
        or isinstance(expires, bool)
        or expires <= submitted
        or record.get("subject") != EMAIL_CHALLENGE_SUBJECT
        or record.get("body_template") != EMAIL_CHALLENGE_BODY_TEMPLATE
        or not isinstance(tool, dict)
        or set(tool) != {"path", "sha256"}
        or not isinstance(tool.get("path"), str)
        or not tool["path"]
        or not Path(tool["path"]).is_absolute()
        or any(character in tool["path"] for character in "\r\n")
        or _SHA256.fullmatch(str(tool.get("sha256", ""))) is None
    ):
        raise AcknowledgementError("email request identity is invalid")
    for index, attempt in enumerate(attempts, start=1):
        if (
            not isinstance(attempt, dict)
            or set(attempt) != {"attempt", "returncode", "timed_out"}
            or attempt.get("attempt") != index
            or not isinstance(attempt.get("returncode"), int)
            or isinstance(attempt.get("returncode"), bool)
            or not isinstance(attempt.get("timed_out"), bool)
            or (
                attempt.get("timed_out") is True
                and attempt.get("returncode") != 124
            )
            or (
                index < len(attempts)
                and attempt.get("returncode") == 0
                and attempt.get("timed_out") is False
            )
        ):
            raise AcknowledgementError("email delivery attempt history is invalid")
    if attempts[-1] != {
        "attempt": len(attempts),
        "returncode": 0,
        "timed_out": False,
    }:
        raise AcknowledgementError(
            "email request does not end in a confirmed delivery submission"
        )


def validate_active_challenge(
    active: Mapping[str, Any],
    *,
    active_path: Path,
    request: Mapping[str, Any],
    request_path: Path,
) -> None:
    if set(active) != ACTIVE_CHALLENGE_FIELDS:
        raise AcknowledgementError(
            "active email challenge fields differ from the closed schema"
        )
    stable = dict(active)
    active_id = stable.pop("active_challenge_id", None)
    supersedes = active.get("supersedes")
    if supersedes is not None and (
        not isinstance(supersedes, dict)
        or set(supersedes)
        != {
            "challenge_generation",
            "challenge_id",
            "request_id",
            "request",
            "request_sha256",
        }
        or not isinstance(supersedes.get("challenge_generation"), int)
        or supersedes["challenge_generation"] < 0
        or _SHA256.fullmatch(str(supersedes.get("challenge_id", ""))) is None
        or _REQUEST_ID.fullmatch(str(supersedes.get("request_id", ""))) is None
        or not isinstance(supersedes.get("request"), str)
        or not Path(str(supersedes.get("request", ""))).is_absolute()
        or any(
            character in str(supersedes.get("request", ""))
            for character in "\r\n"
        )
        or _SHA256.fullmatch(str(supersedes.get("request_sha256", ""))) is None
    ):
        raise AcknowledgementError("active email challenge supersession is invalid")
    if (
        active.get("immutable_sha256") != request.get("immutable_sha256")
        or active.get("chain_id") != request.get("chain_id")
        or active.get("request_id") != request.get("request_id")
        or active.get("release_tag") != request.get("release_tag")
        or active.get("release_git_commit")
        != request.get("release_git_commit")
        or active.get("recipient") != request.get("recipient")
        or active.get("challenge_generation")
        != request.get("challenge_generation")
        or active.get("challenge_id") != request.get("challenge_id")
        or active.get("request") != str(request_path.resolve())
        or active.get("request_sha256") != _sha256(request_path)
        or active.get("acknowledgement") != request.get("acknowledgement")
    ):
        raise AcknowledgementError(
            "request is not the exact currently active email challenge"
        )
    if supersedes is not None:
        previous_path = Path(supersedes["request"]).expanduser().absolute()
        previous = _read(
            previous_path, description="superseded email acknowledgement request"
        )
        validate_request_payload(previous)
        if (
            _sha256(previous_path) != supersedes["request_sha256"]
            or previous["challenge_generation"]
            != supersedes["challenge_generation"]
            or previous["challenge_id"] != supersedes["challenge_id"]
            or previous["request_id"] != supersedes["request_id"]
            or previous["immutable_sha256"] != request["immutable_sha256"]
            or previous["chain_id"] != request["chain_id"]
            or previous["release_tag"] != request["release_tag"]
            or previous["release_git_commit"]
            != request["release_git_commit"]
            or previous["recipient"] != request["recipient"]
            or previous["active_challenge"] != str(active_path.resolve())
            or previous["challenge_generation"]
            > request["challenge_generation"]
            or previous["request_id"] == request["request_id"]
        ):
            raise AcknowledgementError(
                "active email challenge supersession preimage is invalid"
            )
    expected = {
        "schema_version": EMAIL_CHALLENGE_SCHEMA_VERSION,
        "kind": "schema5_email_active_challenge",
        "passed": True,
        "immutable_sha256": request["immutable_sha256"],
        "chain_id": request["chain_id"],
        "request_id": request["request_id"],
        "release_tag": request["release_tag"],
        "release_git_commit": request["release_git_commit"],
        "recipient": request["recipient"],
        "challenge_generation": request["challenge_generation"],
        "challenge_id": request["challenge_id"],
        "request": str(request_path.resolve()),
        "request_sha256": _sha256(request_path),
        "acknowledgement": request["acknowledgement"],
        "supersedes": supersedes,
        "activated_timestamp": active.get("activated_timestamp"),
    }
    activated = active.get("activated_timestamp")
    if (
        dict(stable) != expected
        or not isinstance(activated, (int, float))
        or isinstance(activated, bool)
        or activated < 0
        or activated != request["submitted_timestamp"]
        or active_id != identity_sha256(stable)
        or request.get("active_challenge") != str(active_path.resolve())
    ):
        raise AcknowledgementError(
            "request is not the exact currently active email challenge"
        )


def validate_acknowledgement(
    acknowledgement: Mapping[str, Any],
    *,
    acknowledgement_path: Path,
    request: Mapping[str, Any],
    request_path: Path,
) -> None:
    """Validate one consumed challenge against its exact immutable request."""

    validate_request_payload(request)
    acknowledgement_path = acknowledgement_path.expanduser().absolute()
    request_path = request_path.expanduser().absolute()
    if set(acknowledgement) != ACKNOWLEDGEMENT_FIELDS:
        raise AcknowledgementError(
            "email acknowledgement fields differ from the closed schema"
        )
    stable = dict(acknowledgement)
    acknowledgement_id = stable.pop("acknowledgement_id", None)
    timestamp = acknowledgement.get("acknowledged_timestamp")
    acknowledged_at = acknowledgement.get("acknowledged_at")
    try:
        canonical_acknowledged_at = datetime.fromtimestamp(
            float(timestamp), tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, OSError, OverflowError) as exc:
        raise AcknowledgementError(
            "email acknowledgement timestamp is invalid"
        ) from exc
    if (
        acknowledgement.get("schema_version")
        != EMAIL_CHALLENGE_SCHEMA_VERSION
        or acknowledgement.get("kind") != "schema5_email_acknowledgement"
        or acknowledgement.get("passed") is not True
        or acknowledgement.get("immutable_sha256")
        != request["immutable_sha256"]
        or acknowledgement.get("chain_id") != request["chain_id"]
        or acknowledgement.get("request_id") != request["request_id"]
        or acknowledgement.get("release_tag") != request["release_tag"]
        or acknowledgement.get("release_git_commit")
        != request["release_git_commit"]
        or acknowledgement.get("recipient") != request["recipient"]
        or acknowledgement.get("challenge_generation")
        != request["challenge_generation"]
        or acknowledgement.get("challenge_id") != request["challenge_id"]
        or acknowledgement.get("request") != str(request_path.resolve())
        or acknowledgement.get("request_sha256") != _sha256(request_path)
        or not isinstance(acknowledgement.get("operator"), str)
        or not acknowledgement["operator"]
        or any(
            character in acknowledgement["operator"]
            for character in "\r\n"
        )
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or timestamp < request["submitted_timestamp"]
        or timestamp > request["expires_timestamp"]
        or acknowledged_at != canonical_acknowledged_at
        or acknowledgement_id != identity_sha256(stable)
        or acknowledgement_path.resolve()
        != Path(str(request["acknowledgement"])).resolve()
    ):
        raise AcknowledgementError(
            "email acknowledgement is not bound to the active request"
        )


def acknowledge(
    *,
    request: Path,
    output: Path,
    token: str,
    operator: str,
    apply: bool = False,
    now: float | None = None,
    executing_tool: Path | None = None,
) -> dict[str, Any]:
    if _TOKEN.fullmatch(token) is None:
        raise AcknowledgementError("acknowledgement token has an unsafe format")
    if not operator or "\n" in operator or "\r" in operator:
        raise AcknowledgementError("operator identity is invalid")
    supplied_request = request.expanduser().absolute()
    record = _read(supplied_request, description="email acknowledgement request")
    validate_request_payload(record)
    if executing_tool is None:
        executing_tool = Path(__file__)
    executing_tool = executing_tool.expanduser().absolute()
    require_safe_directory(
        executing_tool.parent,
        description="email acknowledgement tool parent",
    )
    tool = record["acknowledgement_tool"]
    if (
        executing_tool.is_symlink()
        or not executing_tool.is_file()
        or executing_tool.resolve() != executing_tool
        or str(executing_tool) != tool["path"]
        or _sha256(executing_tool) != tool["sha256"]
    ):
        raise AcknowledgementError(
            "executing acknowledgement tool differs from the sealed request"
        )
    request = supplied_request.resolve()
    supplied_output = output.expanduser().absolute()
    if str(supplied_output) != record["acknowledgement"]:
        raise AcknowledgementError(
            "acknowledgement output differs from the immutable request"
        )
    output = _safe_output(supplied_output)
    active_path = Path(record["active_challenge"]).expanduser().absolute()
    timestamp = float(time.time() if now is None else now)
    with challenge_lock(active_path):
        # Re-read all mutable identities while holding the shared supersession lock.
        record = _read(request, description="email acknowledgement request")
        validate_request_payload(record)
        active = _read(active_path, description="active email challenge")
        validate_active_challenge(
            active,
            active_path=active_path,
            request=record,
            request_path=request,
        )
        if timestamp > float(record["expires_timestamp"]):
            raise AcknowledgementError("email challenge has expired")
        expected_verifier = challenge_verifier_for(token, record)
        if not hmac.compare_digest(
            expected_verifier, str(record["challenge_verifier"])
        ):
            raise AcknowledgementError("token or email-request identity is invalid")
        if output.exists() or output.is_symlink():
            raise AcknowledgementError(
                "email challenge was already consumed; replay is forbidden"
            )
        acknowledged_at = datetime.fromtimestamp(
            timestamp, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        stable = {
            "schema_version": EMAIL_CHALLENGE_SCHEMA_VERSION,
            "kind": "schema5_email_acknowledgement",
            "passed": True,
            "immutable_sha256": record["immutable_sha256"],
            "chain_id": record["chain_id"],
            "request_id": record["request_id"],
            "release_tag": record["release_tag"],
            "release_git_commit": record["release_git_commit"],
            "recipient": record["recipient"],
            "challenge_generation": record["challenge_generation"],
            "challenge_id": record["challenge_id"],
            "request": str(request),
            "request_sha256": _sha256(request),
            "operator": operator,
            "acknowledged_at": acknowledged_at,
            "acknowledged_timestamp": timestamp,
        }
        payload = stable | {
            "acknowledgement_id": identity_sha256(stable)
        }
        if not apply:
            return payload | {"status": "dry_run", "output": str(output)}
        _atomic_once(output, payload)
    return payload | {"status": "acknowledged", "output": str(output)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--token", required=True)
    parser.add_argument("--operator", default=getpass.getuser())
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = acknowledge(
            request=args.request,
            output=args.output,
            token=args.token,
            operator=args.operator,
            apply=args.apply,
        )
    except (AcknowledgementError, OSError, ValueError) as exc:
        print(f"[schema5-email-ack] ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
