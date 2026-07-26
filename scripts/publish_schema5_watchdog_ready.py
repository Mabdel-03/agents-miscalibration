#!/usr/bin/env python3
"""Publish sealed external-watchdog drill and readiness markers.

The deployment and drill happen outside Slurm.  This tool only validates their sealed
evidence after it has been copied into recovery storage; it has no SSH, scheduler,
systemd, acknowledgement, or repair capability of its own.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


RELEASE_ID = "sweep-recovery-schema5-v1.2"
RELEASE_TAG = "sweep-recovery-schema5-v1.2-r2"
CHAIN_NAMESPACE = "schema5-v1.2-r2"
DRILL_PROTOCOL = "schema5-v1.2-r2-external-watchdog-drill-v1"
READY_PROTOCOL = "schema5-v1.2-r2-external-watchdog-v1"
DEPLOYMENT_EVIDENCE_PROTOCOL = "schema5-external-watchdog-deployment-evidence-v1"
DRILL_EVIDENCE_PROTOCOL = "schema5-external-watchdog-drill-evidence-v1"
LIVENESS_EVIDENCE_PROTOCOL = "schema5-external-watchdog-liveness-evidence-v1"
DRILL_MARKER = "EXTERNAL_WATCHDOG_KILL_DRILL_COMPLETE.json"
READY_MARKER = "WATCHDOG_READY.json"
SHA256 = __import__("re").compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT = __import__("re").compile(r"[0-9a-f]{40}\Z")


class WatchdogEvidenceError(RuntimeError):
    """External watchdog evidence is incomplete, mutable, or inconsistent."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(
        _stable_bytes(path, description=f"hash input {path.name}")
    ).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return _sha256_bytes(_canonical(payload))


def _canonical_path(path: Path, *, description: str, kind: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if any(character in str(lexical) for character in ("\x00", "\n", "\r")):
        raise WatchdogEvidenceError(f"{description} path is unsafe")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise WatchdogEvidenceError(f"{description} is unavailable: {exc}") from exc
    if resolved != lexical:
        raise WatchdogEvidenceError(f"{description} traverses a symlink")
    metadata = lexical.stat(follow_symlinks=False)
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise WatchdogEvidenceError(f"{description} is not a regular file")
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise WatchdogEvidenceError(f"{description} is not a directory")
    return lexical


def _stable_bytes(
    path: Path,
    *,
    description: str,
    require_read_only: bool = False,
    require_single_link: bool = False,
) -> bytes:
    canonical = _canonical_path(path, description=description, kind="file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(canonical, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    current = canonical.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or identity(before) != identity(after)
        or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
        or (require_read_only and stat.S_IMODE(before.st_mode) & 0o222)
        or (require_single_link and before.st_nlink != 1)
    ):
        raise WatchdogEvidenceError(
            f"{description} is not sealed: mutable, linked, or changed while read"
        )
    return b"".join(chunks)


def _read_sealed(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    raw = _stable_bytes(
        path,
        description=description,
        require_read_only=True,
        require_single_link=True,
    )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WatchdogEvidenceError(
                    f"{description} duplicates JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                WatchdogEvidenceError(
                    f"{description} contains non-finite value {token}"
                )
            ),
        )
    except WatchdogEvidenceError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise WatchdogEvidenceError(f"{description} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or raw != _canonical(value):
        raise WatchdogEvidenceError(f"{description} is not canonical JSON")
    return value, raw


def _validate_self_hashed(
    value: Mapping[str, Any], *, field: str, description: str
) -> None:
    observed = value.get(field)
    if (
        not isinstance(observed, str)
        or SHA256.fullmatch(observed) is None
        or observed != _self_hash(value, field)
    ):
        raise WatchdogEvidenceError(f"{description} self-hash is invalid")


def _release_fields(
    *, git_commit: str, tag_object: str
) -> dict[str, Any]:
    if (
        GIT_OBJECT.fullmatch(git_commit) is None
        or GIT_OBJECT.fullmatch(tag_object) is None
    ):
        raise WatchdogEvidenceError("release commit/tag-object binding is malformed")
    return {
        "release_id": RELEASE_ID,
        "release_tag": RELEASE_TAG,
        "release_git_commit": git_commit,
        "release_tag_object": tag_object,
        "chain_namespace": CHAIN_NAMESPACE,
    }


def _validate_release_evidence(
    value: Mapping[str, Any],
    *,
    protocol: str,
    git_commit: str,
    tag_object: str,
    description: str,
) -> None:
    if (
        value.get("schema_version") != 1
        or value.get("protocol") != protocol
        or value.get("passed") is not True
        or any(
            value.get(field) != expected
            for field, expected in _release_fields(
                git_commit=git_commit, tag_object=tag_object
            ).items()
        )
    ):
        raise WatchdogEvidenceError(
            f"{description} does not bind the exact immutable release"
        )


def _deployment(
    path: Path, *, git_commit: str, tag_object: str
) -> tuple[dict[str, Any], bytes]:
    value, raw = _read_sealed(path, description="watchdog deployment evidence")
    required = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "deployment_id",
        "watchdog_code_sha256",
        "immutable_release_sha256",
        "runtime_inventory_sha256",
        "runtime_file_count",
        "runtime_total_bytes",
        "vm_python_path",
        "vm_python_sha256",
        "control_sha256",
        "liveness_email",
        "forced_command_only",
        "timer_seconds",
        "isolated_runtime_probe_passed",
        "systemd_service_loaded",
        "systemd_service_result",
        "systemd_service_exec_main_status",
        "successful_service_heartbeat_id",
        "successful_service_heartbeat_sha256",
        "systemd_timer_active",
        "systemd_timer_enabled",
        "evidence_id",
    }
    if set(value) != required:
        raise WatchdogEvidenceError("watchdog deployment evidence fields drifted")
    _validate_release_evidence(
        value,
        protocol=DEPLOYMENT_EVIDENCE_PROTOCOL,
        git_commit=git_commit,
        tag_object=tag_object,
        description="watchdog deployment evidence",
    )
    _validate_self_hashed(
        value, field="evidence_id", description="watchdog deployment evidence"
    )
    if (
        any(
            SHA256.fullmatch(str(value.get(field, ""))) is None
            for field in (
                "deployment_id",
                "watchdog_code_sha256",
                "immutable_release_sha256",
                "runtime_inventory_sha256",
                "vm_python_sha256",
                "control_sha256",
                "successful_service_heartbeat_id",
                "successful_service_heartbeat_sha256",
            )
        )
        or not isinstance(value.get("runtime_file_count"), int)
        or isinstance(value.get("runtime_file_count"), bool)
        or value["runtime_file_count"] <= 0
        or not isinstance(value.get("runtime_total_bytes"), int)
        or isinstance(value.get("runtime_total_bytes"), bool)
        or value["runtime_total_bytes"] <= 0
        or not isinstance(value.get("vm_python_path"), str)
        or not Path(value["vm_python_path"]).is_absolute()
        or value.get("forced_command_only") is not True
        or not isinstance(value.get("liveness_email"), str)
        or "@" not in value["liveness_email"]
        or value.get("timer_seconds") != 300
        or value.get("isolated_runtime_probe_passed") is not True
        or value.get("systemd_service_loaded") is not True
        or value.get("systemd_service_result") != "success"
        or value.get("systemd_service_exec_main_status") != 0
        or value.get("systemd_timer_active") is not True
        or value.get("systemd_timer_enabled") is not True
    ):
        raise WatchdogEvidenceError("watchdog deployment is not ready")
    return value, raw


def _drill_evidence(
    path: Path,
    *,
    deployment: Mapping[str, Any],
    git_commit: str,
    tag_object: str,
) -> tuple[dict[str, Any], bytes]:
    value, raw = _read_sealed(path, description="watchdog cancellation-drill evidence")
    required = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "deployment_id",
        "watchdog_code_sha256",
        "immutable_release_sha256",
        "control_sha256",
        "scheduler_observations",
        "namespace_cancellation_recovery_seconds",
        "duplicate_jobs",
        "duplicate_admission_intents",
        "fairness_mutations",
        "evidence_id",
    }
    if set(value) != required:
        raise WatchdogEvidenceError("watchdog drill evidence fields drifted")
    _validate_release_evidence(
        value,
        protocol=DRILL_EVIDENCE_PROTOCOL,
        git_commit=git_commit,
        tag_object=tag_object,
        description="watchdog drill evidence",
    )
    _validate_self_hashed(
        value, field="evidence_id", description="watchdog drill evidence"
    )
    observations = value.get("scheduler_observations")
    recovery = value.get("namespace_cancellation_recovery_seconds")
    if (
        any(
            value.get(field) != deployment.get(field)
            for field in (
                "deployment_id",
                "watchdog_code_sha256",
                "immutable_release_sha256",
                "control_sha256",
            )
        )
        or not isinstance(observations, list)
        or len(observations) < 2
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in observations
        )
        or float(observations[-1]) - float(observations[0]) < 60
        or not isinstance(recovery, (int, float))
        or isinstance(recovery, bool)
        or not math.isfinite(float(recovery))
        or not 0 <= float(recovery) <= 900
        or value.get("duplicate_jobs") != 0
        or value.get("duplicate_admission_intents") != 0
        or value.get("fairness_mutations") != 0
    ):
        raise WatchdogEvidenceError(
            "watchdog drill did not prove bounded exact single recovery"
        )
    return value, raw


def _liveness_evidence(
    path: Path,
    *,
    deployment: Mapping[str, Any],
    git_commit: str,
    tag_object: str,
) -> tuple[dict[str, Any], bytes]:
    value, raw = _read_sealed(path, description="watchdog liveness evidence")
    required = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "deployment_id",
        "watchdog_code_sha256",
        "immutable_release_sha256",
        "control_sha256",
        "scheduler_observations",
        "heartbeat_sha256",
        "liveness_email",
        "liveness_email_ack",
        "evidence_id",
    }
    if set(value) != required:
        raise WatchdogEvidenceError("watchdog liveness evidence fields drifted")
    _validate_release_evidence(
        value,
        protocol=LIVENESS_EVIDENCE_PROTOCOL,
        git_commit=git_commit,
        tag_object=tag_object,
        description="watchdog liveness evidence",
    )
    _validate_self_hashed(
        value, field="evidence_id", description="watchdog liveness evidence"
    )
    observations = value.get("scheduler_observations")
    if (
        any(
            value.get(field) != deployment.get(field)
            for field in (
                "deployment_id",
                "watchdog_code_sha256",
                "immutable_release_sha256",
                "control_sha256",
            )
        )
        or not isinstance(observations, list)
        or len(observations) != 2
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in observations
        )
        or float(observations[1]) - float(observations[0]) < 60
        or SHA256.fullmatch(str(value.get("heartbeat_sha256", ""))) is None
        or value.get("liveness_email") != deployment.get("liveness_email")
        or value.get("liveness_email_ack") is not True
    ):
        raise WatchdogEvidenceError(
            "watchdog liveness/email evidence is incomplete"
        )
    return value, raw


def _publish_once(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = _canonical_path(
        path.parent, description=f"{path.name} parent", kind="directory"
    )
    encoded = _canonical(payload)
    lock_descriptor = os.open(
        parent / f".{path.name}.publish.lock",
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        pattern = f".{path.name}."
        candidates = sorted(
            child
            for child in parent.iterdir()
            if child.name.startswith(pattern) and child.name.endswith(".publishing")
        )
        target_metadata = (
            path.stat(follow_symlinks=False)
            if path.exists() and not path.is_symlink()
            else None
        )
        if path.is_symlink():
            raise WatchdogEvidenceError(
                f"immutable marker is symlinked: {path}"
            )
        exact_candidates: list[Path] = []
        for candidate in candidates:
            if candidate.is_symlink() or not candidate.is_file():
                raise WatchdogEvidenceError(
                    f"unsafe interrupted marker publication: {candidate}"
                )
            candidate_raw = _stable_bytes(
                candidate,
                description=f"interrupted marker {candidate.name}",
                require_read_only=True,
            )
            candidate_metadata = candidate.stat(follow_symlinks=False)
            if target_metadata is not None and (
                candidate_metadata.st_dev,
                candidate_metadata.st_ino,
            ) == (target_metadata.st_dev, target_metadata.st_ino):
                candidate.unlink()
            elif candidate_raw == encoded:
                exact_candidates.append(candidate)
            else:
                raise WatchdogEvidenceError(
                    f"conflicting interrupted marker publication: {candidate}"
                )
        if not path.exists() and exact_candidates:
            survivor = exact_candidates.pop(0)
            os.link(survivor, path, follow_symlinks=False)
            survivor.unlink()
        for candidate in exact_candidates:
            candidate.unlink()
        if path.exists():
            if _stable_bytes(
                path,
                description=f"existing immutable marker {path.name}",
                require_read_only=True,
                require_single_link=True,
            ) != encoded:
                raise WatchdogEvidenceError(
                    f"immutable marker already exists with different bytes: {path}"
                )
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".publishing",
                dir=parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.chmod(0o444)
                os.link(temporary, path, follow_symlinks=False)
                temporary.unlink()
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        directory = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def publish(
    *,
    recovery_root: Path,
    deployment_evidence: Path,
    drill_evidence: Path,
    liveness_evidence: Path,
    git_commit: str,
    tag_object: str,
    apply: bool,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(recovery_root.expanduser())))
    if root.exists() or root.is_symlink():
        root = _canonical_path(
            root, description="watchdog recovery root", kind="directory"
        )
    deployment, deployment_raw = _deployment(
        deployment_evidence,
        git_commit=git_commit,
        tag_object=tag_object,
    )
    drill, drill_raw = _drill_evidence(
        drill_evidence,
        deployment=deployment,
        git_commit=git_commit,
        tag_object=tag_object,
    )
    liveness, liveness_raw = _liveness_evidence(
        liveness_evidence,
        deployment=deployment,
        git_commit=git_commit,
        tag_object=tag_object,
    )
    drill_marker: dict[str, Any] = {
        "schema_version": 1,
        "protocol": DRILL_PROTOCOL,
        "passed": True,
        **_release_fields(git_commit=git_commit, tag_object=tag_object),
        "deployment_id": deployment["deployment_id"],
        "watchdog_code_sha256": deployment["watchdog_code_sha256"],
        "immutable_release_sha256": deployment["immutable_release_sha256"],
        "control_sha256": deployment["control_sha256"],
        "namespace_cancellation_recovery_seconds": drill[
            "namespace_cancellation_recovery_seconds"
        ],
        "duplicate_jobs": 0,
        "duplicate_admission_intents": 0,
        "fairness_mutations": 0,
    }
    drill_marker["drill_id"] = _self_hash(drill_marker, "drill_id")
    drill_path = root / DRILL_MARKER
    drill_marker_raw = _canonical(drill_marker)
    ready_marker: dict[str, Any] = {
        "schema_version": 1,
        "protocol": READY_PROTOCOL,
        "passed": True,
        **_release_fields(git_commit=git_commit, tag_object=tag_object),
        "deployment_id": deployment["deployment_id"],
        "watchdog_code_sha256": deployment["watchdog_code_sha256"],
        "immutable_release_sha256": deployment["immutable_release_sha256"],
        "control_sha256": deployment["control_sha256"],
        "forced_command_only": True,
        "timer_seconds": 300,
        "scheduler_observations": liveness["scheduler_observations"],
        "liveness_email_ack": True,
        "external_watchdog_drill": {
            "marker": str(drill_path),
            "marker_sha256": _sha256_bytes(drill_marker_raw),
            "drill_id": drill_marker["drill_id"],
        },
        "namespace_cancellation_recovery_seconds": drill[
            "namespace_cancellation_recovery_seconds"
        ],
        "duplicate_jobs": 0,
        "duplicate_admission_intents": 0,
        "fairness_mutations": 0,
    }
    ready_marker["marker_id"] = _self_hash(ready_marker, "marker_id")
    ready_path = root / READY_MARKER
    result = {
        "passed": True,
        "apply": apply,
        "deployment_evidence_sha256": _sha256_bytes(deployment_raw),
        "drill_evidence_sha256": _sha256_bytes(drill_raw),
        "liveness_evidence_sha256": _sha256_bytes(liveness_raw),
        "drill_marker": str(drill_path),
        "drill_id": drill_marker["drill_id"],
        "ready_marker": str(ready_path),
        "marker_id": ready_marker["marker_id"],
    }
    if apply:
        _publish_once(drill_path, drill_marker)
        _publish_once(ready_path, ready_marker)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--deployment-evidence", type=Path, required=True)
    parser.add_argument("--drill-evidence", type=Path, required=True)
    parser.add_argument("--liveness-evidence", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--tag-object", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = publish(
        recovery_root=args.recovery_root,
        deployment_evidence=args.deployment_evidence,
        drill_evidence=args.drill_evidence,
        liveness_evidence=args.liveness_evidence,
        git_commit=args.git_commit,
        tag_object=args.tag_object,
        apply=args.apply,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
