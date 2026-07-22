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
import os
import re
import secrets
import shlex
import stat
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from agents_scaling.experiment import io


STATE_SCHEMA_VERSION = 1
STATE_DIRECTORY = ".fleet-transactions-v1"
LEDGERS_DIRECTORY = "ledgers"
CURRENT_FILENAME = "CURRENT.json"
ALERTS_FILENAME = "alerts.jsonl"
LOCK_FILENAME = "fleet.lock"
SAVE_INTENT_FILENAME = "ledger-save-intent.json"
DEFAULT_VISIBILITY_GRACE_SECONDS = 300.0
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[0-9a-f]{32}")


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


@dataclass(frozen=True)
class SchedulerSnapshot:
    rows: tuple[SchedulerRow, ...]
    captured_at: float
    squeue_ok: bool
    sacct_ok: bool
    errors: tuple[str, ...] = ()


def state_directory(pool_root: str | Path) -> Path:
    return Path(pool_root).expanduser().resolve() / STATE_DIRECTORY


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        return json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise FleetTransactionError(f"cannot parse {artifact}: {exc}") from exc


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
        "scheduler_comment",
        "job_id",
        "submitted_at",
        "committed_at",
        "terminal_at",
        "last_seen_at",
        "missing_since",
        "last_error",
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
    expected_parent = (directory / "sbatch" / f"g{generation:06d}").resolve()
    observed_path = Path(attempt["sbatch_path"])
    try:
        observed_path.parent.resolve().relative_to(expected_parent)
    except ValueError as exc:
        raise FleetTransactionError(
            f"sbatch path escapes fleet transaction root for {replica_id}"
        ) from exc
    if observed_path.parent.resolve() != expected_parent:
        raise FleetTransactionError(f"sbatch path nesting drifted for {replica_id}")
    safe_replica = re.sub(r"[^A-Za-z0-9_.-]+", "_", replica_id)
    if observed_path.name != f"{safe_replica}.{token}.sbatch":
        raise FleetTransactionError(f"sbatch filename does not bind intent for {replica_id}")
    if _SHA256_RE.fullmatch(str(attempt["sbatch_sha256"])) is None:
        raise FleetTransactionError(f"invalid sbatch hash for {replica_id}")
    for field in (
        "created_at",
        "submit_started_at",
        "submitted_at",
        "committed_at",
        "terminal_at",
        "last_seen_at",
        "missing_since",
    ):
        value = attempt[field]
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise FleetTransactionError(f"invalid {field} for {replica_id}")
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
    if path.is_symlink() or not path.is_file():
        raise FleetTransactionError(f"fleet ledger is not a regular file: {path}")
    try:
        ledger = _strict_json_loads(
            path.read_text(encoding="utf-8"), artifact=f"fleet ledger {path}"
        )
    except (OSError, UnicodeError) as exc:
        raise FleetTransactionError(f"cannot parse fleet ledger {path}: {exc}") from exc
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
            raise FleetTransactionError(f"replica ledger fields drifted for {replica_id}")
        attempts = record["attempts"]
        if not isinstance(attempts, list):
            raise FleetTransactionError(f"replica attempts are not an array for {replica_id}")
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
            if attempt["intent_token"] in tokens:
                raise FleetTransactionError(f"duplicate intent token for {replica_id}")
            tokens.add(attempt["intent_token"])
            if attempt["job_id"] is not None:
                if str(attempt["job_id"]) in job_ids:
                    raise FleetTransactionError(f"duplicate committed job id for {replica_id}")
                job_ids.add(str(attempt["job_id"]))
        nonterminal = [
            item
            for item in attempts
            if item["state"]
            in {"prepared", "submitting", "submitted", "committed", "missing"}
        ]
        if len(nonterminal) > 1:
            raise FleetTransactionError(f"multiple current attempts for {replica_id}")
        _validate_health(record["health"], replica_id=replica_id)
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
) -> dict[str, Any]:
    """Publish one immutable script and its durable pre-submit intent."""

    record = ledger["replicas"][replica_id]
    current = [
        item
        for item in record["attempts"]
        if item["state"] in {"prepared", "submitting", "submitted", "committed", "missing"}
    ]
    if current:
        raise FleetTransactionError(f"replica {replica_id} already has a current intent")
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
    sbatch_dir = directory / "sbatch" / f"g{rollout_generation:06d}"
    if sbatch_dir.is_symlink() or sbatch_dir.parent.is_symlink():
        raise FleetTransactionError("fleet sbatch directory is symlinked")
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    sbatch_path = sbatch_dir / f"{safe_replica}.{token}.sbatch"
    payload = sbatch_text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if sbatch_path.exists():
        if sbatch_path.is_symlink() or sbatch_path.read_bytes() != payload:
            raise FleetTransactionError(f"immutable fleet script collision: {sbatch_path}")
    else:
        io.atomic_write_text(sbatch_path, sbatch_text)
        sbatch_path.chmod(stat.S_IMODE(sbatch_path.stat().st_mode) & ~0o222)
        _fsync_directory(sbatch_path.parent)
    attempt = {
        "intent_token": token,
        "rollout_generation": rollout_generation,
        "state": "prepared",
        "created_at": float(now),
        "submit_started_at": None,
        "submission_attempts": 0,
        "sbatch_path": str(sbatch_path.resolve()),
        "sbatch_sha256": digest,
        "scheduler_comment": comment,
        "job_id": None,
        "submitted_at": None,
        "committed_at": None,
        "terminal_at": None,
        "last_seen_at": None,
        "missing_since": None,
        "last_error": None,
    }
    record["attempts"].append(attempt)
    save_ledger(directory, ledger, now=now)
    return attempt


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
    if attempt["state"] not in {"prepared", "submitting", "submission_failed"}:
        raise FleetTransactionError(
            f"cannot submit fleet intent in state {attempt['state']!r}"
        )
    path = Path(attempt["sbatch_path"])
    if (
        path.is_symlink()
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != attempt["sbatch_sha256"]
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise FleetTransactionError(f"immutable fleet script drifted: {path}")
    attempt["state"] = "submitting"
    attempt["submit_started_at"] = float(now)
    attempt["submission_attempts"] += 1
    attempt["last_error"] = None
    save_ledger(directory, ledger, now=now)
    invoke = runner or subprocess.run
    try:
        proc = invoke(
            [
                "sbatch",
                "--parsable",
                f"--comment={attempt['scheduler_comment']}",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60.0,
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


def _parse_rows(text: str, *, source: str) -> list[SchedulerRow]:
    rows: list[SchedulerRow] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.rstrip("\n").split("|", 6)
        if len(fields) != 7:
            raise FleetTransactionError(
                f"malformed {source} fleet row {line_number}: {raw!r}"
            )
        job_id, name, state, partition, node, command, comment = (
            field.strip() for field in fields
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
            SchedulerRow(job_id, name, state, partition, node, command, comment, source)
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
                "--format=JobIDRaw,JobName,State,Partition,NodeList,SubmitLine,Comment",
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
                "%i|%j|%T|%P|%N|%o|%k",
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
                or prior.comment != row.comment
            ):
                raise FleetTransactionError(
                    f"squeue/sacct identity conflict for fleet job {row.job_id}"
                )
            # The later squeue pass owns current state/node/command.
            by_id[row.job_id] = row
    if errors or not all(success.values()):
        raise FleetTransactionError(
            "fleet reconciliation requires complete squeue+sacct truth: "
            + "; ".join(errors or ["one scheduler source was unavailable"])
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


def read_health_summary(pool_root: str | Path) -> dict[str, Any]:
    """Read the atomic fleet ledgers for the five-minute/email monitor.

    This function never creates files or takes the writer lock.  Every individual file
    is atomically replaced by the supervisor; CURRENT additionally binds the current
    generation's exact bytes.  A cross-file transition may yield a conservative read
    error for one monitor poll, which correctly becomes a fleet-integrity alert.
    """

    directory = state_directory(pool_root)
    if not directory.exists():
        return {
            "available": False,
            "current_generation": None,
            "active_hung_allocations": [],
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
        for replica_id, record in ledger["replicas"].items():
            if not isinstance(record, dict):
                raise FleetTransactionError(f"invalid fleet replica state in {path}")
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
        "historical_alert_count": len(alert_ids),
        "alerts_path": str(alerts_path),
    }
