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
import importlib.util
import json
import math
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


RELEASE_ID = "sweep-recovery-schema5-v1.2"
RELEASE_TAG = "sweep-recovery-schema5-v1.2-r12"
CHAIN_NAMESPACE = "schema5-v1.2-r12"
DRILL_PROTOCOL = "schema5-v1.2-r12-external-watchdog-drill-v1"
READY_PROTOCOL = "schema5-v1.2-r12-external-watchdog-v1"
DEPLOYMENT_EVIDENCE_PROTOCOL = "schema5-external-watchdog-deployment-evidence-v1"
DRILL_EVIDENCE_PROTOCOL = "schema5-external-watchdog-drill-evidence-v1"
LIVENESS_EVIDENCE_PROTOCOL = "schema5-external-watchdog-liveness-evidence-v1"
DRILL_MARKER = "EXTERNAL_WATCHDOG_KILL_DRILL_COMPLETE.json"
READY_MARKER = "WATCHDOG_READY.json"
BOOTSTRAP_ATTESTATION_PROTOCOL = (
    "schema5-v1.2-r12-bootstrap-watchdog-attestation-v1"
)
BOOTSTRAP_READY_PROTOCOL = (
    "schema5-v1.2-r12-bootstrap-watchdog-deployment-ready-v1"
)
BOOTSTRAP_READY_MARKER = (
    "RECOVERY_CHAIN_BOOTSTRAP_WATCHDOG_READY.json"
)
BOOTSTRAP_ARM_INTENT_PROTOCOL = (
    "schema5-v1.2-r12-bootstrap-watchdog-arm-intent-v1"
)
BOOTSTRAP_ARM_INTENT_MARKER = (
    "RECOVERY_CHAIN_BOOTSTRAP_WATCHDOG_ARM_INTENT.json"
)
BOOTSTRAP_ARMED_PROTOCOL = (
    "schema5-v1.2-r12-bootstrap-watchdog-armed-v1"
)
BOOTSTRAP_ARMED_MARKER = (
    "RECOVERY_CHAIN_BOOTSTRAP_WATCHDOG_ARMED.json"
)
BOOTSTRAP_HEARTBEAT_MAX_AGE_SECONDS = 600
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


def _revalidate_bootstrap_drill_preimages(
    *,
    drill: Mapping[str, Any],
    bundle: Mapping[str, Any],
    deployment: Mapping[str, Any],
) -> None:
    """Independently re-read every drill preimage before READY/ARMED.

    The deployment builder owns the receipt/provenance validation contract.  Load
    that exact sibling source by path rather than accepting an import from a
    mutable search path; bootstrap's tagged-source inventory already binds the
    complete ``scripts``/``src`` closure before the renderer can invoke us.
    """

    source = _canonical_path(
        Path(__file__).resolve().with_name(
            "build_schema5_watchdog_deployment.py"
        ),
        description="bootstrap drill validator source",
        kind="file",
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_bootstrap_drill_ready_validator", source
    )
    if spec is None or spec.loader is None:
        raise WatchdogEvidenceError(
            "bootstrap drill validator cannot be loaded"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        module._validate_bootstrap_drill_preimages(
            drill=drill,
            bundle=bundle,
            deployment=deployment,
        )
    except Exception as exc:
        raise WatchdogEvidenceError(
            f"bootstrap drill sealed preimages are invalid: {exc}"
        ) from exc


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
                require_read_only=False,
            )
            candidate_metadata = candidate.stat(follow_symlinks=False)
            candidate_mode = stat.S_IMODE(candidate_metadata.st_mode)
            same_as_target = target_metadata is not None and (
                candidate_metadata.st_dev,
                candidate_metadata.st_ino,
            ) == (target_metadata.st_dev, target_metadata.st_ino)
            if (
                candidate_metadata.st_uid != os.geteuid()
                or candidate_mode not in {0o600, 0o444}
                or (
                    same_as_target
                    and (
                        candidate_mode != 0o444
                        or candidate_metadata.st_nlink != 2
                    )
                )
                or (
                    not same_as_target
                    and candidate_metadata.st_nlink != 1
                )
            ):
                raise WatchdogEvidenceError(
                    "unsafe interrupted marker ownership, mode, or link "
                    f"count: {candidate}"
                )
            if same_as_target:
                if candidate_raw != encoded:
                    raise WatchdogEvidenceError(
                        f"conflicting linked marker publication: {candidate}"
                    )
                candidate.unlink()
            elif candidate_raw == encoded:
                if candidate_mode == 0o600:
                    candidate.chmod(0o444)
                    descriptor = os.open(
                        candidate,
                        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
                    )
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    if (
                        _stable_bytes(
                            candidate,
                            description=(
                                f"sealed interrupted marker {candidate.name}"
                            ),
                            require_read_only=True,
                            require_single_link=True,
                        )
                        != encoded
                    ):
                        raise WatchdogEvidenceError(
                            f"interrupted marker changed while sealed: "
                            f"{candidate}"
                        )
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


def publish_bootstrap(
    *,
    recovery_root: Path,
    chain_manifest: Path,
    submission_receipt: Path,
    bootstrap_attestation: Path,
    git_commit: str,
    tag_object: str,
    apply: bool,
) -> dict[str, Any]:
    """Publish the pre-control watchdog gate for the exact held transaction."""

    root = _canonical_path(
        Path(os.path.abspath(os.fspath(recovery_root.expanduser()))),
        description="bootstrap watchdog recovery root",
        kind="directory",
    )
    manifest, manifest_raw = _read_sealed(
        chain_manifest, description="recovery-chain manifest"
    )
    receipt, receipt_raw = _read_sealed(
        submission_receipt, description="recovery-chain submission receipt"
    )
    attestation, attestation_raw = _read_sealed(
        bootstrap_attestation,
        description="bootstrap watchdog attestation",
    )
    manifest_identity = dict(manifest)
    chain_id = manifest_identity.pop("chain_id", None)
    receipt_identity = dict(receipt)
    receipt_id = receipt_identity.pop("receipt_id", None)
    if (
        Path(chain_manifest).resolve(strict=True) != Path(chain_manifest)
        or Path(submission_receipt).resolve(strict=True)
        != Path(submission_receipt)
        or Path(submission_receipt).parent != root
        or not isinstance(chain_id, str)
        or SHA256.fullmatch(chain_id) is None
        or chain_id != _sha256_bytes(_canonical(manifest_identity))
        or not isinstance(receipt_id, str)
        or SHA256.fullmatch(receipt_id) is None
        or receipt_id != _sha256_bytes(_canonical(receipt_identity))
        or receipt.get("manifest") != str(chain_manifest)
        or receipt.get("manifest_sha256")
        != _sha256_bytes(manifest_raw)
        or receipt.get("chain_id") != chain_id
        or receipt.get("root_initial_hold") is not True
        or receipt.get("no_requeue") is not True
        or manifest.get("release_git_commit") != git_commit
        or manifest.get("release_tag_object") != tag_object
    ):
        raise WatchdogEvidenceError(
            "bootstrap manifest/receipt identity is invalid"
        )
    jobs = receipt.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 43:
        raise WatchdogEvidenceError(
            "bootstrap receipt must bind exactly 43 jobs"
        )
    namespace: list[dict[str, str]] = []
    observed_ids: set[str] = set()
    observed_comments: set[str] = set()
    for row in jobs:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or not isinstance(row.get("job_id"), str)
            or not row["job_id"].isdigit()
            or not isinstance(row.get("comment"), str)
            or row["job_id"] in observed_ids
            or row["comment"] in observed_comments
        ):
            raise WatchdogEvidenceError(
                "bootstrap receipt job namespace is invalid"
            )
        observed_ids.add(row["job_id"])
        observed_comments.add(row["comment"])
        namespace.append(
            {
                "name": row["name"],
                "job_id": row["job_id"],
                "comment": row["comment"],
            }
        )
    root_rows = [row for row in namespace if row["name"] == "source_checkout"]
    if len(root_rows) != 1:
        raise WatchdogEvidenceError(
            "bootstrap receipt has no exact held source root"
        )
    required_attestation = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "chain_manifest_sha256",
        "submission_receipt_sha256",
        "deployment_id",
        "deployment_evidence",
        "deployment_evidence_sha256",
        "deployment_evidence_id",
        "watchdog_code_sha256",
        "bundle_manifest",
        "bundle_manifest_sha256",
        "bundle_id",
        "ssh_public_key_sha256",
        "separate_bootstrap_key",
        "forced_command_argv",
        "systemd_service_loaded",
        "systemd_timer_active",
        "forced_command_only",
        "forced_operation",
        "scheduler_observations",
        "cancellation_drill",
        "cancellation_drill_evidence",
        "cancellation_drill_evidence_sha256",
        "cancellation_drill_evidence_id",
        "root_release_authority",
        "prelaunch_root_release",
        "descendant_rearm_authority",
        "release_intent_continuation_authority",
        "scientific_admission_direct",
        "production_control_mutation",
        "safety_hold_clear_authority",
        "handoff_stage",
        "evidence_id",
    }
    _validate_self_hashed(
        attestation,
        field="evidence_id",
        description="bootstrap watchdog attestation",
    )
    observation_bindings = attestation.get("scheduler_observations")
    observation_binding_fields = {
        "artifact",
        "artifact_sha256",
        "observation_id",
    }
    observation_fields = {
        "schema_version",
        "protocol",
        "passed",
        "observed_at_timestamp",
        "release_git_commit",
        "release_tag_object",
        "chain_id",
        "chain_manifest",
        "chain_manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "generation_provenance",
        "generation_provenance_sha256",
        "generation_provenance_id",
        "anchor_submission_receipt",
        "anchor_submission_receipt_sha256",
        "anchor_submission_receipt_id",
        "descendant_chain_validated",
        "squeue_complete",
        "sacct_complete",
        "submission_receipt_id",
        "repair_generation",
        "root_name",
        "root_job_id",
        "root_names",
        "root_job_ids",
        "roots_held",
        "job_count",
        "ambiguous_jobs",
        "root_held",
        "namespace_scan_complete",
        "namespace_lineage_receipt_ids",
        "namespace_lineage_job_ids",
        "namespace_bound_job_ids",
        "jobs",
        "recovery_namespace_cancelled",
        "anchor_launch_authorized",
        "anchor_launch_id",
        "anchor_armed_authorized",
        "anchor_armed_id",
        "anchor_release_intent_authorized",
        "anchor_release_intent_sha256",
        "descendant_armed",
        "descendant_armed_id",
        "descendant_rearm_required",
        "descendant_release_required",
        "handoff_complete",
        "watchdog_scientific_jobs_submitted",
        "observation_id",
    }
    observation_payloads: list[dict[str, Any]] = []
    observation_provenance_valid: list[bool] = []
    if isinstance(observation_bindings, list):
        for binding in observation_bindings:
            if (
                not isinstance(binding, dict)
                or set(binding) != observation_binding_fields
            ):
                break
            artifact, raw = _read_sealed(
                Path(str(binding["artifact"])),
                description="bootstrap scheduler observation",
            )
            _validate_self_hashed(
                artifact,
                field="observation_id",
                description="bootstrap scheduler observation",
            )
            if (
                binding["artifact_sha256"] != _sha256_bytes(raw)
                or binding["observation_id"]
                != artifact.get("observation_id")
            ):
                break
            try:
                provenance, provenance_raw = _read_sealed(
                    Path(str(artifact["generation_provenance"])),
                    description=(
                        "bootstrap generation scheduler provenance"
                    ),
                )
                provenance_identity = dict(provenance)
                provenance_id = provenance_identity.pop(
                    "provenance_id", None
                )
                provenance_jobs = provenance.get("jobs")
                provenance_ok = (
                    artifact.get("generation_provenance_sha256")
                    == _sha256_bytes(provenance_raw)
                    and artifact.get("generation_provenance_id")
                    == provenance_id
                    and provenance_id
                    == _sha256_bytes(_canonical(provenance_identity))
                    and provenance.get("protocol")
                    == (
                        "schema5-v1.2-r12-bootstrap-"
                        "generation-provenance-v1"
                    )
                    and provenance.get("passed") is True
                    and provenance.get("chain_id") == chain_id
                    and provenance.get("chain_manifest_sha256")
                    == _sha256_bytes(manifest_raw)
                    and provenance.get("submission_receipt_id")
                    == receipt_id
                    and provenance.get("submission_receipt_sha256")
                    == _sha256_bytes(receipt_raw)
                    and provenance.get("repair_generation") == 0
                    and provenance.get("namespace_scan_complete") is True
                    and provenance.get("namespace_lineage_receipt_ids")
                    == [receipt_id]
                    and provenance.get("namespace_bound_job_ids")
                    == sorted(
                        {item["job_id"] for item in namespace}, key=int
                    )
                    and provenance.get("generation_root_names")
                    == ["source_checkout"]
                    and provenance.get("current_generation_names")
                    == [item["name"] for item in namespace]
                    and provenance.get("scheduler_topology_valid") is True
                    and isinstance(provenance_jobs, list)
                    and len(provenance_jobs) == len(namespace)
                )
            except (KeyError, OSError, WatchdogEvidenceError):
                provenance_ok = False
            observation_payloads.append(artifact)
            observation_provenance_valid.append(provenance_ok)
    observations = [
        {
            "observed_at_timestamp": item.get("observed_at_timestamp"),
            "squeue_complete": item.get("squeue_complete"),
            "sacct_complete": item.get("sacct_complete"),
            "submission_receipt_id": item.get("submission_receipt_id"),
            "job_count": item.get("job_count"),
            "ambiguous_jobs": item.get("ambiguous_jobs"),
            "root_held": item.get("root_held"),
        }
        for item in observation_payloads
    ]
    drill = attestation.get("cancellation_drill")
    valid_observations = (
        isinstance(observation_bindings, list)
        and len(observation_payloads) == len(observation_bindings) == 2
        and len(observation_provenance_valid) == 2
        and all(observation_provenance_valid)
        and all(
            set(row) == observation_fields
            and row.get("schema_version") == 1
            and row.get("protocol")
            == "schema5-v1.2-r12-bootstrap-status-v1"
            and row.get("passed") is True
            and isinstance(row.get("observed_at_timestamp"), (int, float))
            and not isinstance(row.get("observed_at_timestamp"), bool)
            and math.isfinite(float(row["observed_at_timestamp"]))
            and row.get("squeue_complete") is True
            and row.get("sacct_complete") is True
            and row.get("submission_receipt_id") == receipt_id
            and row.get("job_count") == len(namespace)
            and row.get("ambiguous_jobs") == 0
            and row.get("root_held") is True
            and row.get("release_git_commit") == git_commit
            and row.get("release_tag_object") == tag_object
            and row.get("chain_id") == chain_id
            and row.get("chain_manifest") == str(chain_manifest)
            and row.get("chain_manifest_sha256")
            == _sha256_bytes(manifest_raw)
            and row.get("submission_receipt")
            == str(submission_receipt)
            and row.get("submission_receipt_sha256")
            == _sha256_bytes(receipt_raw)
            and row.get("anchor_submission_receipt")
            == str(submission_receipt)
            and row.get("anchor_submission_receipt_sha256")
            == _sha256_bytes(receipt_raw)
            and row.get("anchor_submission_receipt_id") == receipt_id
            and row.get("descendant_chain_validated") is True
            and row.get("repair_generation") == 0
            and row.get("root_name") == root_rows[0]["name"]
            and row.get("root_job_id") == root_rows[0]["job_id"]
            and row.get("root_names") == [root_rows[0]["name"]]
            and row.get("root_job_ids") == [root_rows[0]["job_id"]]
            and row.get("roots_held") is True
            and row.get("namespace_scan_complete") is True
            and row.get("namespace_lineage_receipt_ids") == [receipt_id]
            and row.get("namespace_lineage_job_ids")
            == [[item["job_id"] for item in namespace]]
            and row.get("namespace_bound_job_ids") == sorted(
                {item["job_id"] for item in namespace}, key=int
            )
            and row.get("jobs")
            == [
                {
                    "name": receipt_row["name"],
                    "job_id": receipt_row["job_id"],
                    "comment": receipt_row["comment"],
                    "job_name": manifest_row["job_name"],
                    "state": "PENDING",
                    "active": True,
                    "script_sha256": manifest_row["script_sha256"],
                    "submit_line_sha256": row["submit_line_sha256"],
                    "submit_line_exact": True,
                    "scontrol_command": receipt_row["script"],
                    "scontrol_requeue": 0,
                    "spooled_script_sha256": manifest_row[
                        "script_sha256"
                    ],
                    "spooled_script_exact_match": True,
                }
                for row, manifest_row, receipt_row in zip(
                    row["jobs"],
                    manifest["jobs"],
                    receipt["jobs"],
                    strict=True,
                )
            ]
            and all(
                SHA256.fullmatch(
                    str(item.get("submit_line_sha256", ""))
                )
                is not None
                and item.get("submit_line_exact") is True
                and item.get("spooled_script_exact_match") is True
                for item in row["jobs"]
            )
            and row.get("recovery_namespace_cancelled") is False
            and row.get("anchor_launch_authorized") is False
            and row.get("anchor_launch_id") is None
            and row.get("anchor_armed_authorized") is False
            and row.get("anchor_armed_id") is None
            and row.get("anchor_release_intent_authorized") is False
            and row.get("anchor_release_intent_sha256") is None
            and row.get("descendant_armed") is None
            and row.get("descendant_armed_id") is None
            and row.get("descendant_rearm_required") is False
            and row.get("descendant_release_required") is False
            and row.get("handoff_complete") is False
            and row.get("watchdog_scientific_jobs_submitted") == 0
            for row in observation_payloads
        )
        and float(observations[1]["observed_at_timestamp"])
        - float(observations[0]["observed_at_timestamp"])
        >= 60.0
    )
    valid_drill = (
        isinstance(drill, dict)
        and set(drill)
        == {
            "recovery_namespace_cancellation_recovery_seconds",
            "isolated_cancellation_drill",
            "isolation_id",
            "isolated_drill_root",
            "isolated_chain_id",
            "isolated_anchor_submission_receipt_id",
            "canonical_submission_receipt_id",
            "canonical_job_id_overlap",
            "canonical_comment_overlap",
            "canonical_control_paths_absent",
            "squeue_complete",
            "sacct_complete",
            "duplicate_jobs",
            "duplicate_submission_intents",
            "root_remained_held",
            "scientific_jobs_started",
        }
        and isinstance(
            drill.get("recovery_namespace_cancellation_recovery_seconds"),
            (int, float),
        )
        and not isinstance(
            drill.get("recovery_namespace_cancellation_recovery_seconds"), bool
        )
        and math.isfinite(
            float(drill["recovery_namespace_cancellation_recovery_seconds"])
        )
        and 0
        <= float(drill["recovery_namespace_cancellation_recovery_seconds"])
        <= 900
        and drill.get("isolated_cancellation_drill") is True
        and SHA256.fullmatch(str(drill.get("isolation_id", ""))) is not None
        and isinstance(drill.get("isolated_drill_root"), str)
        and bool(drill["isolated_drill_root"])
        and SHA256.fullmatch(str(drill.get("isolated_chain_id", "")))
        is not None
        and SHA256.fullmatch(
            str(drill.get("isolated_anchor_submission_receipt_id", ""))
        )
        is not None
        and SHA256.fullmatch(
            str(drill.get("canonical_submission_receipt_id", ""))
        )
        is not None
        and drill.get("isolated_chain_id") != chain_id
        and drill.get("isolated_anchor_submission_receipt_id") != receipt_id
        and drill.get("canonical_submission_receipt_id") == receipt_id
        and drill.get("canonical_job_id_overlap") == 0
        and drill.get("canonical_comment_overlap") == 0
        and drill.get("canonical_control_paths_absent") is True
        and drill.get("squeue_complete") is True
        and drill.get("sacct_complete") is True
        and drill.get("duplicate_jobs") == 0
        and drill.get("duplicate_submission_intents") == 0
        and drill.get("root_remained_held") is True
        and drill.get("scientific_jobs_started") == 0
    )
    bundle_path = Path(str(attestation.get("bundle_manifest", "")))
    deployment_path = Path(
        str(attestation.get("deployment_evidence", ""))
    )
    drill_path = Path(
        str(attestation.get("cancellation_drill_evidence", ""))
    )
    bundle: dict[str, Any] = {}
    bundle_raw = b""
    deployment: dict[str, Any] = {}
    deployment_raw = b""
    drill_evidence: dict[str, Any] = {}
    drill_evidence_raw = b""
    try:
        bundle, bundle_raw = _read_sealed(
            bundle_path,
            description="bootstrap watchdog bundle manifest",
        )
        deployment, deployment_raw = _read_sealed(
            deployment_path,
            description="bootstrap watchdog deployment evidence",
        )
        drill_evidence, drill_evidence_raw = _read_sealed(
            drill_path,
            description="bootstrap watchdog cancellation drill evidence",
        )
    except (OSError, WatchdogEvidenceError):
        if set(attestation) == required_attestation:
            raise
    _revalidate_bootstrap_drill_preimages(
        drill=drill_evidence,
        bundle=bundle,
        deployment=deployment,
    )
    bundle_identity = dict(bundle)
    bundle_id = bundle_identity.pop("bundle_id", None)
    deployment_identity = dict(deployment)
    deployment_evidence_id = deployment_identity.pop(
        "evidence_id", None
    )
    drill_identity = dict(drill_evidence)
    drill_evidence_id = drill_identity.pop("evidence_id", None)
    forced_argv = attestation.get("forced_command_argv")
    bundle_harness = bundle.get("harness_environment_binding")
    bundle_pilot = bundle.get("materialization_pilot")

    def forced_value(flag: str) -> str | None:
        if not isinstance(forced_argv, list):
            return None
        try:
            index = forced_argv.index(flag)
        except ValueError:
            return None
        if index + 1 >= len(forced_argv):
            return None
        value = forced_argv[index + 1]
        return value if isinstance(value, str) else None

    if (
        set(attestation) != required_attestation
        or attestation.get("schema_version") != 1
        or attestation.get("protocol")
        != BOOTSTRAP_ATTESTATION_PROTOCOL
        or attestation.get("passed") is not True
        or any(
            attestation.get(field) != expected
            for field, expected in _release_fields(
                git_commit=git_commit, tag_object=tag_object
            ).items()
        )
        or attestation.get("chain_manifest_sha256")
        != _sha256_bytes(manifest_raw)
        or attestation.get("submission_receipt_sha256")
        != _sha256_bytes(receipt_raw)
        or SHA256.fullmatch(
            str(attestation.get("deployment_id", ""))
        )
        is None
        or SHA256.fullmatch(
            str(attestation.get("watchdog_code_sha256", ""))
        )
        is None
        or attestation.get("bundle_manifest_sha256")
        != _sha256_bytes(bundle_raw)
        or attestation.get("bundle_id") != bundle_id
        or not isinstance(bundle_id, str)
        or SHA256.fullmatch(bundle_id) is None
        or bundle_id != _sha256_bytes(_canonical(bundle_identity))
        or bundle.get("protocol")
        != "schema5-v1.2-r12-bootstrap-watchdog-bundle-v1"
        or bundle.get("release_git_commit") != git_commit
        or bundle.get("release_tag_object") != tag_object
        or bundle.get("chain_manifest_sha256")
        != _sha256_bytes(manifest_raw)
        or bundle.get("watchdog_code_sha256")
        != attestation.get("watchdog_code_sha256")
        or bundle.get("ssh_public_key_sha256")
        != attestation.get("ssh_public_key_sha256")
        or SHA256.fullmatch(
            str(attestation.get("ssh_public_key_sha256", ""))
        )
        is None
        or attestation.get("separate_bootstrap_key") is not True
        or bundle.get("separate_bootstrap_key") is not True
        or attestation.get("deployment_evidence_sha256")
        != _sha256_bytes(deployment_raw)
        or attestation.get("deployment_evidence_id")
        != deployment_evidence_id
        or deployment_evidence_id
        != _sha256_bytes(_canonical(deployment_identity))
        or deployment.get("protocol")
        != "schema5-v1.2-r12-bootstrap-watchdog-deployment-evidence-v1"
        or deployment.get("passed") is not True
        or deployment.get("bundle_id") != bundle_id
        or deployment.get("bundle_sha256")
        != _sha256_bytes(bundle_raw)
        or deployment.get("deployment_id")
        != attestation.get("deployment_id")
        or deployment.get("watchdog_code_sha256")
        != attestation.get("watchdog_code_sha256")
        or deployment.get("release_git_commit") != git_commit
        or deployment.get("release_tag_object") != tag_object
        or deployment.get("chain_id") != chain_id
        or deployment.get("chain_manifest_sha256")
        != _sha256_bytes(manifest_raw)
        or deployment.get("submission_receipt_id") != receipt_id
        or deployment.get("submission_receipt_sha256")
        != _sha256_bytes(receipt_raw)
        or deployment.get("runtime_inventory_sha256")
        != bundle.get("runtime_inventory_sha256")
        or deployment.get("runtime_file_count")
        != bundle.get("runtime_file_count")
        or deployment.get("runtime_total_bytes")
        != bundle.get("runtime_total_bytes")
        or deployment.get("vm_python_path") != bundle.get("vm_python")
        or SHA256.fullmatch(
            str(deployment.get("vm_python_sha256", ""))
        )
        is None
        or deployment.get("vm_python_immutable") is not True
        or not isinstance(
            deployment.get("heartbeat_freshness_seconds"), (int, float)
        )
        or isinstance(
            deployment.get("heartbeat_freshness_seconds"), bool
        )
        or not math.isfinite(
            float(deployment.get("heartbeat_freshness_seconds", -1))
        )
        or not 0
        <= float(deployment["heartbeat_freshness_seconds"])
        <= BOOTSTRAP_HEARTBEAT_MAX_AGE_SECONDS
        or deployment.get("heartbeat_max_age_seconds")
        != BOOTSTRAP_HEARTBEAT_MAX_AGE_SECONDS
        or deployment.get("ssh_public_key_sha256")
        != attestation.get("ssh_public_key_sha256")
        or deployment.get("separate_bootstrap_key") is not True
        or deployment.get("forced_command_only") is not True
        or deployment.get("systemd_service_loaded") is not True
        or deployment.get("systemd_timer_active") is not True
        or attestation.get("cancellation_drill_evidence_sha256")
        != _sha256_bytes(drill_evidence_raw)
        or attestation.get("cancellation_drill_evidence_id")
        != drill_evidence_id
        or drill_evidence_id
        != _sha256_bytes(_canonical(drill_identity))
        or drill_evidence.get("protocol")
        != "schema5-v1.2-r12-bootstrap-watchdog-drill-evidence-v1"
        or drill_evidence.get("passed") is not True
        or drill_evidence.get("bundle_id") != bundle_id
        or drill_evidence.get("deployment_id")
        != attestation.get("deployment_id")
        or any(
            drill_evidence.get(field) != drill.get(field)
            for field in (
                "recovery_namespace_cancellation_recovery_seconds",
                "isolated_cancellation_drill",
                "isolation_id",
                "isolated_drill_root",
                "isolated_chain_id",
                "isolated_anchor_submission_receipt_id",
                "canonical_submission_receipt_id",
                "canonical_job_id_overlap",
                "canonical_comment_overlap",
                "canonical_control_paths_absent",
                "squeue_complete",
                "sacct_complete",
                "duplicate_jobs",
                "duplicate_submission_intents",
                "root_remained_held",
                "scientific_jobs_started",
            )
        )
        or not isinstance(forced_argv, list)
        or forced_argv != bundle.get("forced_command_argv")
        or len(forced_argv) < 6
        or any(not isinstance(item, str) or not item for item in forced_argv)
        or forced_argv[-1] != "bootstrap-dispatch"
        or attestation.get("systemd_service_loaded") is not True
        or attestation.get("systemd_timer_active") is not True
        or attestation.get("forced_command_only") is not True
        or attestation.get("forced_operation") != "bootstrap-repair"
        or attestation.get("root_release_authority")
        != (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        )
        or attestation.get("prelaunch_root_release") is not False
        or attestation.get("descendant_rearm_authority") is not True
        or attestation.get("release_intent_continuation_authority") is not True
        or attestation.get("scientific_admission_direct") is not False
        or attestation.get("production_control_mutation") is not False
        or attestation.get("safety_hold_clear_authority") is not False
        or bundle.get("root_release_authority")
        != (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        )
        or bundle.get("prelaunch_root_release") is not False
        or bundle.get("descendant_rearm_authority") is not True
        or bundle.get("release_intent_continuation_authority") is not True
        or bundle.get("scientific_admission_direct") is not False
        or bundle.get("production_control_mutation") is not False
        or bundle.get("safety_hold_clear_authority") is not False
        or not isinstance(bundle_harness, dict)
        or SHA256.fullmatch(
            str(bundle_harness.get("manifest_sha256", ""))
        )
        is None
        or SHA256.fullmatch(
            str(bundle_harness.get("inventory_sha256", ""))
        )
        is None
        or not isinstance(bundle_harness.get("lexical_path"), str)
        or not isinstance(bundle_harness.get("resolved_path"), str)
        or not isinstance(bundle_harness.get("manifest_path"), str)
        or not isinstance(bundle_pilot, dict)
        or set(bundle_pilot)
        != {
            "marker",
            "marker_sha256",
            "pilot_id",
            "pilot_root",
            "release_worktree",
            "release_bundle",
            "source_tree_sha256",
            "release_bundle_id",
            "release_identity_sha256",
            "release_completion_sha256",
        }
        or SHA256.fullmatch(
            str(bundle_pilot.get("marker_sha256", ""))
        )
        is None
        or SHA256.fullmatch(str(bundle_pilot.get("pilot_id", ""))) is None
        or any(
            SHA256.fullmatch(str(bundle_pilot.get(field, ""))) is None
            for field in (
                "source_tree_sha256",
                "release_bundle_id",
                "release_identity_sha256",
                "release_completion_sha256",
            )
        )
        or any(
            not isinstance(bundle_pilot.get(field), str)
            or not bundle_pilot[field]
            for field in (
                "pilot_root",
                "release_worktree",
                "release_bundle",
            )
        )
        or forced_value("--harness-python")
        != bundle_harness.get("lexical_path")
        or forced_value("--resolved-harness-python")
        != bundle_harness.get("resolved_path")
        or forced_value("--harness-environment-manifest")
        != bundle_harness.get("manifest_path")
        or forced_value("--harness-environment-sha256")
        != bundle_harness.get("manifest_sha256")
        or forced_value("--materialization-pilot-marker")
        != bundle_pilot.get("marker")
        or forced_value("--materialization-pilot-sha256")
        != bundle_pilot.get("marker_sha256")
        or forced_value("--materialization-pilot-id")
        != bundle_pilot.get("pilot_id")
        or attestation.get("handoff_stage") != "watchdog_readiness"
        or not valid_observations
        or not valid_drill
    ):
        raise WatchdogEvidenceError(
            "bootstrap watchdog attestation is incomplete or unsafe"
        )
    root_record = root_rows[0]
    ready_marker: dict[str, Any] = {
        "schema_version": 1,
        "protocol": BOOTSTRAP_READY_PROTOCOL,
        "passed": True,
        **_release_fields(git_commit=git_commit, tag_object=tag_object),
        "chain_manifest": str(chain_manifest),
        "chain_manifest_sha256": _sha256_bytes(manifest_raw),
        "chain_id": chain_id,
        "deployment_id": attestation["deployment_id"],
        "watchdog_code_sha256": attestation["watchdog_code_sha256"],
        "bootstrap_attestation": str(bootstrap_attestation),
        "bootstrap_attestation_sha256": _sha256_bytes(attestation_raw),
        "bootstrap_attestation_id": attestation["evidence_id"],
        "bundle_manifest": str(bundle_path),
        "bundle_manifest_sha256": _sha256_bytes(bundle_raw),
        "bundle_id": bundle_id,
        "ssh_public_key_sha256": attestation[
            "ssh_public_key_sha256"
        ],
        "separate_bootstrap_key": True,
        "forced_command_only": True,
        "allowed_operations": ["bootstrap-status", "bootstrap-repair"],
        "timer_seconds": 300,
        "root_release_authority": (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        ),
        "prelaunch_root_release": False,
        "descendant_rearm_authority": True,
        "release_intent_continuation_authority": True,
        "scientific_admission_direct": False,
        "production_control_mutation": False,
        "safety_hold_clear_authority": False,
    }
    ready_marker["ready_id"] = _self_hash(ready_marker, "ready_id")
    ready_path = root / BOOTSTRAP_READY_MARKER
    arm_intent: dict[str, Any] = {
        "schema_version": 1,
        "protocol": BOOTSTRAP_ARM_INTENT_PROTOCOL,
        "chain_id": chain_id,
        "submission_receipt": str(submission_receipt),
        "submission_receipt_sha256": _sha256_bytes(receipt_raw),
        "submission_receipt_id": receipt_id,
        "root_job_id": root_record["job_id"],
        "bootstrap_watchdog_ready": str(ready_path),
        "bootstrap_watchdog_ready_sha256": _sha256_bytes(
            _canonical(ready_marker)
        ),
        "bootstrap_watchdog_ready_id": ready_marker["ready_id"],
    }
    arm_intent["arm_intent_id"] = _self_hash(
        arm_intent, "arm_intent_id"
    )
    arm_intent_path = root / BOOTSTRAP_ARM_INTENT_MARKER
    armed_marker: dict[str, Any] = {
        "schema_version": 1,
        "protocol": BOOTSTRAP_ARMED_PROTOCOL,
        "passed": True,
        **_release_fields(git_commit=git_commit, tag_object=tag_object),
        "bootstrap_watchdog_ready": str(ready_path),
        "bootstrap_watchdog_ready_sha256": _sha256_bytes(
            _canonical(ready_marker)
        ),
        "bootstrap_watchdog_ready_id": ready_marker["ready_id"],
        "arm_intent": str(arm_intent_path),
        "arm_intent_sha256": _sha256_bytes(_canonical(arm_intent)),
        "arm_intent_id": arm_intent["arm_intent_id"],
        "chain_manifest": str(chain_manifest),
        "chain_manifest_sha256": _sha256_bytes(manifest_raw),
        "chain_id": chain_id,
        "submission_receipt": str(submission_receipt),
        "submission_receipt_sha256": _sha256_bytes(receipt_raw),
        "submission_receipt_id": receipt_id,
        "root_name": root_record["name"],
        "root_job_id": root_record["job_id"],
        "root_comment": root_record["comment"],
        "job_namespace": namespace,
        "job_count": len(namespace),
        "forced_command_only": True,
        "forced_operation": "bootstrap-repair",
        "scheduler_observations": observations,
        "scheduler_observation_artifacts": observation_bindings,
        "cancellation_drill": drill,
        "root_release_authority": (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        ),
        "prelaunch_root_release": False,
        "descendant_rearm_authority": True,
        "release_intent_continuation_authority": True,
        "scientific_admission_direct": False,
        "production_control_mutation": False,
        "safety_hold_clear_authority": False,
        "handoff_stage": "watchdog_readiness",
    }
    armed_marker["marker_id"] = _self_hash(armed_marker, "marker_id")
    armed_path = root / BOOTSTRAP_ARMED_MARKER
    result = {
        "passed": True,
        "apply": apply,
        "bootstrap_attestation": str(bootstrap_attestation),
        "bootstrap_attestation_sha256": _sha256_bytes(attestation_raw),
        "ready_marker": str(ready_path),
        "ready_id": ready_marker["ready_id"],
        "arm_intent": str(arm_intent_path),
        "arm_intent_id": arm_intent["arm_intent_id"],
        "armed_marker": str(armed_path),
        "marker_id": armed_marker["marker_id"],
        "submission_receipt_id": receipt_id,
        "root_job_id": root_record["job_id"],
    }
    if apply:
        _publish_once(ready_path, ready_marker)
        _publish_once(arm_intent_path, arm_intent)
        _publish_once(armed_path, armed_marker)
    return result


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
        "duplicate_jobs": drill["duplicate_jobs"],
        "duplicate_admission_intents": drill[
            "duplicate_admission_intents"
        ],
        "fairness_mutations": drill["fairness_mutations"],
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
        "duplicate_jobs": drill["duplicate_jobs"],
        "duplicate_admission_intents": drill[
            "duplicate_admission_intents"
        ],
        "fairness_mutations": drill["fairness_mutations"],
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
    parser.add_argument("--deployment-evidence", type=Path)
    parser.add_argument("--drill-evidence", type=Path)
    parser.add_argument("--liveness-evidence", type=Path)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--chain-manifest", type=Path)
    parser.add_argument("--submission-receipt", type=Path)
    parser.add_argument("--bootstrap-attestation", type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--tag-object", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.bootstrap:
        if (
            args.chain_manifest is None
            or args.submission_receipt is None
            or args.bootstrap_attestation is None
        ):
            raise WatchdogEvidenceError(
                "bootstrap publication requires manifest, receipt, and attestation"
            )
        result = publish_bootstrap(
            recovery_root=args.recovery_root,
            chain_manifest=args.chain_manifest,
            submission_receipt=args.submission_receipt,
            bootstrap_attestation=args.bootstrap_attestation,
            git_commit=args.git_commit,
            tag_object=args.tag_object,
            apply=args.apply,
        )
    else:
        if (
            args.deployment_evidence is None
            or args.drill_evidence is None
            or args.liveness_evidence is None
        ):
            raise WatchdogEvidenceError(
                "production publication requires deployment, drill, and liveness evidence"
            )
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
