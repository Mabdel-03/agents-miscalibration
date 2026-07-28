#!/usr/bin/env python3
"""Run and attest the schema-5 v1.2 two-prefix materialization pilot.

The pilot is a marker-last orchestration transaction around the independently
testable capture, materialization, and release-freeze tools.  It is dry-run by
default.  Under ``--apply`` it:

* verifies a clean checkout at an explicitly supplied annotated tag and commit;
* inventories both live developer prefixes before any stage;
* runs capture dry/apply/verify without invoking Conda;
* runs materialization dry/apply/verify, allowing Conda to clone only normalized
  seeds through an independently copied, immutable package-cache seed inside the
  isolated pilot root;
* runs release freeze dry/apply/verify without invoking Conda;
* proves the live prefixes did not change across the complete transaction;
* proves exactly one Setuptools 81 distribution and no Setuptools 82 ownership
  record or versioned path remain in every seed and materialized prefix;
* proves all relevant copy boundaries have zero shared regular-file inodes; and
* repeats sealed capture, materialization, and release verification before
  publishing ``PILOT_COMPLETE.json`` as the final evidence file.

Completed evidence can be verified without the mutable live prefixes, the original
checkout, or an external Conda executable.  The release-local sealed Conda toolchain
is completely reverified and may operate only against isolated scratch.

The narrow ``quarantine`` command preserves a canonical pilot interrupted by a
proved scheduler/node transient or explicit external cancellation.  It binds the
exact rendered sbatch receipt to terminal ``squeue``/``sacct`` truth, renames the
whole partial pilot tree on the same filesystem, and seals it before the exact tagged
job may be retried.  Deterministic, timeout, and OOM failures require a superseding
release.

The ``submit-sbatch`` command is the only production submission path. It persists
marker-first intent, reconciles complete ``squeue`` plus ``sacct`` truth, adopts one
exact job after a submit-boundary crash, and publishes immutable acceptance last.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import capture_schema5_environments as capture  # noqa: E402
from scripts import freeze_schema5_release as freeze  # noqa: E402
from scripts import materialize_schema5_release as materialize  # noqa: E402
from scripts import provision_schema5_conda_toolchain as conda_toolchain  # noqa: E402
from scripts import publish_schema5_durable_git_release as durable_git  # noqa: E402
from scripts import seal_recovery_evidence as recovery_evidence  # noqa: E402
from slurm import schema5_control as control  # noqa: E402


SCHEMA_VERSION = 5
RELEASE_ID = freeze.RELEASE_ID
REQUIRED_TAG = freeze.REQUIRED_GIT_TAG
COMPLETE_MARKER = "PILOT_COMPLETE.json"
INTENT_MARKER = "PILOT_INTENT.json"
AUDIT_FILENAME = "PILOT_AUDIT.json"
LIVE_INVENTORY_FILENAMES = {
    ("harness", "before"): "LIVE_HARNESS_BEFORE.json",
    ("serving", "before"): "LIVE_SERVING_BEFORE.json",
    ("harness", "after"): "LIVE_HARNESS_AFTER.json",
    ("serving", "after"): "LIVE_SERVING_AFTER.json",
}
STAGE_EVIDENCE_FILENAMES = {
    ("capture", "dry_run"): "CAPTURE_DRY_RUN.json",
    ("capture", "apply"): "CAPTURE_APPLY.json",
    ("capture", "verify"): "CAPTURE_VERIFY.json",
    ("materialization", "dry_run"): "MATERIALIZATION_DRY_RUN.json",
    ("materialization", "apply"): "MATERIALIZATION_APPLY.json",
    ("materialization", "verify"): "MATERIALIZATION_VERIFY.json",
    ("freeze", "dry_run"): "FREEZE_DRY_RUN.json",
    ("freeze", "apply"): "FREEZE_APPLY.json",
    ("freeze", "verify"): "FREEZE_VERIFY.json",
}
EVIDENCE_FILENAMES = {
    INTENT_MARKER,
    AUDIT_FILENAME,
    *LIVE_INVENTORY_FILENAMES.values(),
    *STAGE_EVIDENCE_FILENAMES.values(),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}\Z")
_SETUPTOOLS_82_PATH_RE = re.compile(
    r"setuptools(?:[-_.])82(?:[-_.])0(?:[-_.])1", re.IGNORECASE
)
_CHUNK_SIZE = 8 * 1024 * 1024
SBATCH_TIME_LIMIT = "11:30:00"
SBATCH_JOB_NAME = "asys-s5-materialization-pilot-r4"
_SBATCH_TOKEN_RE = re.compile(r"[A-Za-z0-9_.%/+=:-]+\Z")
PILOT_QUARANTINE_PROTOCOL = "schema5-v1.2-r4-materialization-pilot-quarantine"
PILOT_QUARANTINE_INTENT_PROTOCOL = (
    "schema5-v1.2-r4-materialization-pilot-quarantine-intent"
)
SCHEDULER_ATTEMPTS_DIRECTORY = "scheduler_attempts"
SCHEDULER_INTENT_FILENAME = "SCHEDULER_INTENT.json"
SCHEDULER_SPOOLED_SCRIPT_FILENAME = "SPOOLED_BATCH_SCRIPT.sbatch"
SCHEDULER_ACTIVE_FILENAME = "SCHEDULER_ACTIVE.json"
SCHEDULER_ACCEPTANCE_FILENAME = "PILOT_SCHEDULER_ACCEPTED.json"
SCHEDULER_INTENT_PROTOCOL = "schema5-v1.2-r4-pilot-scheduler-intent"
SCHEDULER_ACTIVE_PROTOCOL = "schema5-v1.2-r4-pilot-scheduler-active"
SCHEDULER_ACCEPTANCE_PROTOCOL = "schema5-v1.2-r4-pilot-scheduler-acceptance"
SUBMISSION_INTENT_FILENAME = "PILOT_SUBMISSION_INTENT.json"
SUBMISSION_ACCEPTED_FILENAME = "PILOT_SUBMISSION_ACCEPTED.json"
SUBMISSION_DIRECTORY = "scheduler_submission"
SUBMISSION_ATTEMPTS_DIRECTORY = "attempts"
SUBMISSION_ATTEMPT_INTENT_FILENAME = "ATTEMPT_INTENT.json"
SUBMISSION_RESULT_FILENAME = "SBATCH_RESULT.json"
SUBMISSION_ABSENT_FILENAME = "NO_ACCEPTED_JOB.json"
SUBMISSION_INTENT_PROTOCOL = "schema5-v1.2-r4-pilot-submission-intent"
SUBMISSION_ATTEMPT_PROTOCOL = "schema5-v1.2-r4-pilot-submission-attempt"
SUBMISSION_RESULT_PROTOCOL = "schema5-v1.2-r4-pilot-submission-result"
SUBMISSION_ABSENT_PROTOCOL = "schema5-v1.2-r4-pilot-submission-absent"
SUBMISSION_ACCEPTED_PROTOCOL = "schema5-v1.2-r4-pilot-submission-accepted"
DEFAULT_SUBMISSION_VISIBILITY_TIMEOUT = 900.0
DEFAULT_SUBMISSION_POLL_SECONDS = 2.0
TRUSTED_SYSTEM_PATH = "/usr/bin:/bin"
_BLOCKED_PROCESS_ENVIRONMENT = {
    "BASH_ENV",
    "CDPATH",
    "ENV",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DIR",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    "GIT_EXEC_PATH",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_TEMPLATE_DIR",
    "GIT_WORK_TREE",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "SLURM_CLUSTERS",
    "SLURM_CONF",
    "SLURM_TIME_FORMAT",
}


def _sanitized_process_environment() -> dict[str, str]:
    """Return a deterministic command environment for Git and Slurm clients."""

    environment = dict(os.environ)
    for key in tuple(environment):
        if (
            key in _BLOCKED_PROCESS_ENVIRONMENT
            or key.startswith("BASH_FUNC_")
            or key.startswith("GIT_CONFIG_KEY_")
            or key.startswith("GIT_CONFIG_VALUE_")
            or key.startswith("SBATCH_")
            or key.startswith("SACCT_")
            or key.startswith("SCONTROL_")
            or key.startswith("SQUEUE_")
        ):
            environment.pop(key, None)
    environment.update(
        {
            "PATH": TRUSTED_SYSTEM_PATH,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return environment
_SLURM_MEMORY_RE = re.compile(
    r"(?P<amount>[0-9]+(?:\.[0-9]+)?)(?P<unit>[KMGT]?)"
    r"(?P<scope>[cn]?)\Z",
    re.IGNORECASE,
)
_INHERENT_TRANSIENT_STATES = frozenset(
    {"BOOT_FAIL", "NODE_FAIL", "PREEMPTED", "REVOKED"}
)
_SLURM_EXIT_CODE_RE = re.compile(r"[0-9]+:[0-9]+\Z")
_EXTERNAL_CANCELLATION_RE = re.compile(r"CANCELLED by [0-9]+\Z")


class MaterializationPilotError(RuntimeError):
    """The two-prefix pilot cannot be completed or independently verified."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise MaterializationPilotError(f"cannot open regular evidence {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MaterializationPilotError(f"evidence is not a regular file: {path}")
        while chunk := os.read(descriptor, _CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (  # noqa: E731 - compact immutable identity
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise MaterializationPilotError(f"evidence changed while being hashed: {path}")
    return digest.hexdigest()


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MaterializationPilotError(f"missing regular {description}: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise MaterializationPilotError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationPilotError(f"{description} must contain one JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _atomic_write_once(path: Path, payload: bytes) -> None:
    """Publish one immutable artifact, accepting only an exact prior write."""

    if path.exists() or path.is_symlink():
        info = path.stat(follow_symlinks=False)
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o444
            or _sha256_file(path) != _sha256_bytes(payload)
            or path.read_bytes() != payload
        ):
            raise MaterializationPilotError(
                f"conflicting immutable pilot artifact: {path}"
            )
        return
    if path.parent.is_symlink():
        raise MaterializationPilotError(
            f"pilot evidence directory is symlinked: {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".publishing", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise MaterializationPilotError(
                f"refusing to replace pilot artifact published concurrently: {path}"
            )
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _existing_directory(value: str | Path, *, description: str) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink() or not lexical.is_dir():
        raise MaterializationPilotError(f"missing or symlinked {description}: {lexical}")
    resolved = lexical.resolve()
    if resolved in {Path(resolved.anchor), Path.home().resolve()}:
        raise MaterializationPilotError(f"refusing unsafe broad {description}: {resolved}")
    return resolved


def _existing_file(
    value: str | Path, *, description: str, executable: bool = False
) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink() or not lexical.is_file():
        raise MaterializationPilotError(f"missing or symlinked {description}: {lexical}")
    resolved = lexical.resolve()
    if executable and not os.access(resolved, os.X_OK):
        raise MaterializationPilotError(f"{description} is not executable: {resolved}")
    return resolved


def _destination_root(value: str | Path) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink():
        raise MaterializationPilotError(f"pilot root is symlinked: {lexical}")
    resolved = lexical.resolve()
    if resolved in {Path(resolved.anchor), Path.home().resolve()}:
        raise MaterializationPilotError(f"refusing unsafe broad pilot root: {resolved}")
    return resolved


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_isolation(
    pilot_root: Path,
    *,
    release_checkout: Path,
    harness_source: Path,
    serving_source: Path,
    source_package_cache: Path,
    conda_toolchain_root: Path,
) -> None:
    observed = {
        "release checkout": release_checkout,
        "harness source": harness_source,
        "serving source": serving_source,
        "source package cache": source_package_cache,
        "sealed Conda toolchain": conda_toolchain_root,
    }
    roots = list(observed.items())
    for index, (left_name, left) in enumerate(roots):
        for right_name, right in roots[index + 1 :]:
            if left == right or _is_relative_to(left, right) or _is_relative_to(right, left):
                raise MaterializationPilotError(
                    f"pilot inputs overlap: {left_name}={left}, {right_name}={right}"
                )
    for description, root in observed.items():
        if (
            pilot_root == root
            or _is_relative_to(pilot_root, root)
            or _is_relative_to(root, pilot_root)
        ):
            raise MaterializationPilotError(
                f"isolated pilot root overlaps {description}: {pilot_root} and {root}"
            )


def _git(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("/usr/bin/git", "-C", str(repository), *arguments),
            capture_output=True,
            text=True,
            check=False,
            env=_sanitized_process_environment(),
        )
    except OSError as exc:
        raise MaterializationPilotError(f"cannot execute Git: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise MaterializationPilotError(
            f"Git command failed ({completed.returncode}): {arguments!r}: {detail}"
        )
    return completed.stdout.strip()


def _verify_exact_annotated_checkout(
    release_checkout: Path,
    *,
    expected_tag: str,
    expected_commit: str,
) -> dict[str, str]:
    if expected_tag != REQUIRED_TAG:
        raise MaterializationPilotError(
            f"pilot tag must be the frozen operational tag {REQUIRED_TAG!r}"
        )
    if _COMMIT_RE.fullmatch(expected_commit) is None:
        raise MaterializationPilotError("expected commit is not an exact Git object ID")
    try:
        identity = freeze.verify_clean_exact_tag(release_checkout)
    except freeze.ReleaseFreezeError as exc:
        raise MaterializationPilotError(str(exc)) from exc
    if identity.get("git_commit") != expected_commit:
        raise MaterializationPilotError(
            "release checkout HEAD does not match the explicitly expected commit"
        )
    tag_ref = f"refs/tags/{expected_tag}"
    tag_type = _git(release_checkout, "cat-file", "-t", tag_ref)
    if tag_type != "tag":
        raise MaterializationPilotError(
            f"release tag {expected_tag!r} is not an annotated tag object"
        )
    tag_object = _git(release_checkout, "rev-parse", "--verify", tag_ref)
    tag_commit = _git(
        release_checkout, "rev-parse", "--verify", f"{tag_ref}^{{commit}}"
    )
    if (
        _COMMIT_RE.fullmatch(tag_object) is None
        or tag_commit != expected_commit
        or identity.get("git_tag") != expected_tag
    ):
        raise MaterializationPilotError("annotated tag/commit identity is inconsistent")
    return {
        **identity,
        "tag_object": tag_object,
        "tag_object_type": tag_type,
    }


def _layout(root: Path) -> dict[str, str]:
    materialization_root = root / "materialization"
    return {
        "pilot_root": str(root),
        "environment_capture_root": str(root / "environment-capture"),
        "materialization_root": str(materialization_root),
        "release_worktree": str(materialization_root / "release-worktree"),
        "harness_prefix": str(materialization_root / "harness-environment"),
        "serving_prefix": str(materialization_root / "serving-environment"),
        "conda_package_cache_seed": str(
            materialization_root / materialize.PACKAGE_CACHE_SEED_DIRECTORY
        ),
        "conda_package_cache": str(
            materialization_root / materialize.PACKAGE_CACHE_DIRECTORY
        ),
        # The freezer deliberately discovers its materialization proof in the parent
        # of the release directory.
        "release_bundle": str(materialization_root / "release"),
    }


def _inventory_summary(inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in inventory.items() if key != "entries"}


def _input_binding(
    *,
    release_checkout: Path,
    harness_source: Path,
    serving_source: Path,
    ownership_policy: Path,
    integrity_normalization_policy: Path,
    reconciliation_incident: Path,
    recovered_setuptools_record: Path,
    conda_toolchain_root: Path,
    source_package_cache: Path,
    durable_git_release_marker: Path,
) -> dict[str, Any]:
    durable_binding = durable_git.marker_binding(durable_git_release_marker)
    try:
        toolchain_binding = (
            conda_toolchain.verified_conda_toolchain_binding(
                conda_toolchain_root,
                exercise=True,
            )
        )
    except (
        OSError,
        conda_toolchain.CondaToolchainProvisionError,
    ) as exc:
        raise MaterializationPilotError(
            f"sealed Conda toolchain verification failed: {exc}"
        ) from exc
    return {
        "release_checkout": str(release_checkout),
        "harness_source": str(harness_source),
        "serving_source": str(serving_source),
        "source_package_cache": str(source_package_cache),
        "ownership_policy": {
            "path": str(ownership_policy),
            "sha256": _sha256_file(ownership_policy),
        },
        "integrity_normalization_policy": {
            "path": str(integrity_normalization_policy),
            "sha256": _sha256_file(integrity_normalization_policy),
        },
        "reconciliation_incident": {
            "path": str(reconciliation_incident),
            "sha256": _sha256_file(reconciliation_incident),
        },
        "recovered_setuptools_record": {
            "path": str(recovered_setuptools_record),
            "sha256": _sha256_file(recovered_setuptools_record),
        },
        "conda_toolchain": toolchain_binding,
        "durable_git_release": durable_binding,
    }


def _without_status(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "status"}


def _stage_evidence_payload(
    *, stage: str, phase: str, report: Mapping[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "kind": "schema5-materialization-pilot-stage",
        "stage": stage,
        "phase": phase,
        "report": _without_status(report),
    }
    payload["evidence_id"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def _read_stage_evidence(root: Path, *, stage: str, phase: str) -> dict[str, Any]:
    filename = STAGE_EVIDENCE_FILENAMES[(stage, phase)]
    payload = _read_json(root / filename, description=f"{stage} {phase} evidence")
    candidate = dict(payload)
    evidence_id = candidate.pop("evidence_id", None)
    if (
        set(payload)
        != {
            "schema_version",
            "release_id",
            "kind",
            "stage",
            "phase",
            "report",
            "evidence_id",
        }
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("release_id") != RELEASE_ID
        or payload.get("kind") != "schema5-materialization-pilot-stage"
        or payload.get("stage") != stage
        or payload.get("phase") != phase
        or not isinstance(payload.get("report"), dict)
        or evidence_id != _sha256_bytes(_canonical_bytes(candidate))
    ):
        raise MaterializationPilotError(f"{stage} {phase} evidence is invalid")
    return payload


def _record_stage_evidence(
    root: Path, *, stage: str, phase: str, report: Mapping[str, Any]
) -> dict[str, Any]:
    payload = _stage_evidence_payload(stage=stage, phase=phase, report=report)
    _atomic_write_once(
        root / STAGE_EVIDENCE_FILENAMES[(stage, phase)], _json_bytes(payload)
    )
    return _read_stage_evidence(root, stage=stage, phase=phase)


_MATERIALIZATION_PHASE_BINDING_FIELDS = (
    "release_id",
    "tag_commit",
    "source_tree_sha256",
    "paths",
    "environment_capture",
    "conda_toolchain",
    "conda_creation_tool",
    "package_cache_seed_input",
)


def _materialization_phase_binding(
    report: Mapping[str, Any],
    *,
    description: str,
) -> dict[str, Any]:
    """Extract the immutable inputs shared by dry/apply/verify reports.

    A pilot may resume after a crash and therefore may encounter an authentic,
    checksummed dry-run record from an earlier attempt.  Its cache selection or
    toolchain can nevertheless be stale.  Treating the phase file's own digest as
    sufficient would allow apply to run with inputs other than those preflighted.
    """

    missing = [
        field
        for field in _MATERIALIZATION_PHASE_BINDING_FIELDS
        if field not in report
    ]
    if missing:
        raise MaterializationPilotError(
            f"{description} lacks immutable phase binding fields: "
            + ", ".join(missing)
        )
    binding = {
        field: report[field]
        for field in _MATERIALIZATION_PHASE_BINDING_FIELDS
    }
    if (
        binding["release_id"] != RELEASE_ID
        or _COMMIT_RE.fullmatch(str(binding["tag_commit"])) is None
        or _SHA256_RE.fullmatch(
            str(binding["source_tree_sha256"])
        )
        is None
        or not isinstance(binding["paths"], dict)
        or not isinstance(binding["environment_capture"], dict)
        or not isinstance(binding["conda_toolchain"], dict)
        or not isinstance(binding["conda_creation_tool"], dict)
        or not isinstance(binding["package_cache_seed_input"], dict)
    ):
        raise MaterializationPilotError(
            f"{description} has a malformed immutable phase binding"
        )
    return binding


def _require_identical_materialization_phase_bindings(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not reports:
        raise MaterializationPilotError(
            "materialization phase binding comparison has no reports"
        )
    bindings = {
        phase: _materialization_phase_binding(
            report,
            description=f"materialization {phase} report",
        )
        for phase, report in reports.items()
    }
    reference_phase, reference = next(iter(bindings.items()))
    for phase, binding in bindings.items():
        if binding != reference:
            raise MaterializationPilotError(
                "materialization dry/apply/verify immutable phase binding "
                f"differs between {reference_phase} and {phase}"
            )
    return reference


def _run_dry_once(
    root: Path,
    *,
    stage: str,
    operation: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    path = root / STAGE_EVIDENCE_FILENAMES[(stage, "dry_run")]
    if path.exists() or path.is_symlink():
        return _read_stage_evidence(root, stage=stage, phase="dry_run")
    report = dict(operation())
    if report.get("status") != "dry_run":
        raise MaterializationPilotError(
            f"{stage} dry-run did not return dry_run status"
        )
    return _record_stage_evidence(
        root, stage=stage, phase="dry_run", report=report
    )


def _run_apply_and_verify(
    root: Path,
    *,
    stage: str,
    apply_operation: Callable[[], Mapping[str, Any]],
    verify_operation: Callable[[], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    apply_report = dict(apply_operation())
    if apply_report.get("status") not in {"created", "already_complete"}:
        raise MaterializationPilotError(
            f"{stage} apply did not create or adopt exact completed evidence"
        )
    apply_evidence = _record_stage_evidence(
        root, stage=stage, phase="apply", report=apply_report
    )
    verify_report = dict(verify_operation())
    if verify_report.get("status") != "verified":
        raise MaterializationPilotError(f"{stage} verification did not succeed")
    verify_evidence = _record_stage_evidence(
        root, stage=stage, phase="verify", report=verify_report
    )
    if apply_evidence["report"].get(
        {
            "capture": "capture_id",
            "materialization": "materialization_id",
            "freeze": "release_bundle_id",
        }[stage]
    ) != verify_evidence["report"].get(
        {
            "capture": "capture_id",
            "materialization": "materialization_id",
            "freeze": "release_bundle_id",
        }[stage]
    ):
        raise MaterializationPilotError(f"{stage} apply/verify identity differs")
    return apply_evidence, verify_evidence


def _setuptools_82_versioned_paths(prefix: Path) -> list[str]:
    matches: list[str] = []
    for directory, directory_names, file_names in os.walk(prefix, followlinks=False):
        directory_names[:] = sorted(directory_names)
        for name in [*directory_names, *sorted(file_names)]:
            path = Path(directory) / name
            relative = path.relative_to(prefix).as_posix()
            if _SETUPTOOLS_82_PATH_RE.search(relative):
                matches.append(relative)
    return sorted(set(matches))


def _setuptools_pip_owned_paths(
    prefix: Path, *, distribution: Mapping[str, Any]
) -> set[str]:
    metadata_path = prefix / str(distribution.get("metadata_path", ""))
    record_path = metadata_path.parent / "RECORD"
    if (
        metadata_path.name != "METADATA"
        or record_path.is_symlink()
        or not record_path.is_file()
    ):
        raise MaterializationPilotError(
            f"Setuptools 81 distribution lacks a regular RECORD: {prefix}"
        )
    site_packages = metadata_path.parent.parent
    owned: set[str] = set()
    try:
        with record_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) != 3:
                    raise MaterializationPilotError(
                        f"malformed Setuptools 81 RECORD: {record_path}"
                    )
                candidate = (site_packages / row[0]).resolve()
                if not _is_relative_to(candidate, prefix):
                    raise MaterializationPilotError(
                        f"Setuptools 81 RECORD escapes prefix: {row[0]!r}"
                    )
                owned.add(candidate.relative_to(prefix).as_posix())
    except (OSError, UnicodeError) as exc:
        raise MaterializationPilotError(
            f"cannot read Setuptools 81 RECORD {record_path}: {exc}"
        ) from exc
    return owned


def _recovered_setuptools_paths(record: Mapping[str, Any]) -> set[str]:
    candidates: set[str] = set()
    files = record.get("files", [])
    if files is not None:
        if not isinstance(files, list) or not all(
            isinstance(value, str) for value in files
        ):
            raise MaterializationPilotError(
                "recovered Setuptools record has malformed file ownership"
            )
        candidates.update(files)
    paths_data = record.get("paths_data")
    if paths_data is not None:
        paths = paths_data.get("paths") if isinstance(paths_data, dict) else None
        if not isinstance(paths, list):
            raise MaterializationPilotError(
                "recovered Setuptools paths_data is malformed"
            )
        for row in paths:
            if not isinstance(row, dict) or not isinstance(row.get("_path"), str):
                raise MaterializationPilotError(
                    "recovered Setuptools paths_data row is malformed"
                )
            candidates.add(row["_path"])
    normalized: set[str] = set()
    for value in candidates:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise MaterializationPilotError(
                f"recovered Setuptools path is unsafe: {value!r}"
            )
        normalized.add(path.as_posix())
    return normalized


def _setuptools_audit(
    prefix: Path,
    *,
    policy: Mapping[str, Any],
    recovered_record: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        normalized = capture.validate_normalized_seed(prefix, policy=policy)
    except capture.EnvironmentCaptureError as exc:
        raise MaterializationPilotError(
            f"Setuptools ownership audit failed for {prefix}: {exc}"
        ) from exc
    distributions = normalized["distribution_inventory"]["distributions"]
    setuptools = [
        row for row in distributions if str(row.get("project")) == "setuptools"
    ]
    versioned_82_paths = _setuptools_82_versioned_paths(prefix)
    pip_owned_paths = (
        _setuptools_pip_owned_paths(prefix, distribution=setuptools[0])
        if len(setuptools) == 1
        else set()
    )
    recovered_owned_paths = _recovered_setuptools_paths(recovered_record)
    unclaimed_existing_paths = sorted(
        relative
        for relative in recovered_owned_paths
        if (prefix / relative).exists() and relative not in pip_owned_paths
    )
    if (
        len(setuptools) != 1
        or setuptools[0].get("version") != "81.0.0"
        or normalized.get("setuptools_conda_record_count") != 0
        or versioned_82_paths
        or unclaimed_existing_paths
    ):
        raise MaterializationPilotError(
            f"Setuptools 81/82 ownership contract failed for {prefix}"
        )
    return {
        "prefix": str(prefix),
        "setuptools_81_distribution_count": 1,
        "setuptools_81_version": "81.0.0",
        "setuptools_82_conda_record_count": 0,
        "setuptools_82_versioned_path_count": 0,
        "setuptools_82_versioned_paths": [],
        "setuptools_82_unclaimed_existing_file_count": 0,
        "setuptools_82_unclaimed_existing_files": [],
        "recovered_setuptools_owned_path_count": len(recovered_owned_paths),
        "setuptools_81_record_owned_path_count": len(pip_owned_paths),
        "distribution_inventory_sha256": normalized["distribution_inventory"][
            "inventory_sha256"
        ],
    }


def _regular_inode_set(root: Path) -> tuple[set[tuple[int, int]], int]:
    identities: set[tuple[int, int]] = set()
    count = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = sorted(directory_names)
        for name in sorted(file_names):
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                identities.add((info.st_dev, info.st_ino))
                count += 1
    return identities, count


def _inode_edges(
    roots: Mapping[str, Path], edges: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    inventories = {
        name: _regular_inode_set(root)
        for name, root in sorted(roots.items())
    }
    reports: dict[str, Any] = {}
    for left, right in edges:
        shared = inventories[left][0] & inventories[right][0]
        if shared:
            raise MaterializationPilotError(
                f"pilot copy edge {left}->{right} shares "
                f"{len(shared)} regular-file inode(s)"
            )
        reports[f"{left}__{right}"] = {
            "source_regular_file_count": inventories[left][1],
            "destination_regular_file_count": inventories[right][1],
            "shared_regular_inode_count": 0,
        }
    return reports


_SEALED_INODE_EDGES = (
    ("harness_seed", "harness_prefix"),
    ("serving_seed", "serving_prefix"),
    ("harness_seed", "serving_seed"),
    ("harness_prefix", "serving_prefix"),
    ("conda_package_cache_seed", "conda_package_cache"),
    ("conda_package_cache_seed", "harness_prefix"),
    ("conda_package_cache_seed", "serving_prefix"),
    ("conda_package_cache", "harness_prefix"),
    ("conda_package_cache", "serving_prefix"),
)
_TRANSACTION_INODE_EDGES = (
    ("release_checkout", "release_worktree"),
    ("harness_source", "harness_seed"),
    ("serving_source", "serving_seed"),
    ("harness_source", "harness_prefix"),
    ("serving_source", "serving_prefix"),
    ("source_package_cache", "conda_package_cache_seed"),
    ("source_package_cache", "conda_package_cache"),
    ("source_package_cache", "harness_prefix"),
    ("source_package_cache", "serving_prefix"),
    *_SEALED_INODE_EDGES,
)


def _pilot_roots(
    *,
    layout: Mapping[str, str],
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    capture_root = Path(layout["environment_capture_root"])
    roots = {
        "release_worktree": Path(layout["release_worktree"]),
        "harness_seed": capture_root / "seeds" / "harness",
        "serving_seed": capture_root / "seeds" / "serving",
        "harness_prefix": Path(layout["harness_prefix"]),
        "serving_prefix": Path(layout["serving_prefix"]),
        "conda_package_cache_seed": Path(
            layout["conda_package_cache_seed"]
        ),
        "conda_package_cache": Path(layout["conda_package_cache"]),
    }
    if inputs is not None:
        roots.update(
            {
                "release_checkout": Path(str(inputs["release_checkout"])),
                "harness_source": Path(str(inputs["harness_source"])),
                "serving_source": Path(str(inputs["serving_source"])),
                "source_package_cache": Path(
                    str(inputs["source_package_cache"])
                ),
            }
        )
    for name, root in roots.items():
        if root.is_symlink() or not root.is_dir():
            raise MaterializationPilotError(
                f"pilot inode-audit root is absent or symlinked: {name}={root}"
            )
    return roots


def _sealed_artifact_audits(
    *,
    layout: Mapping[str, str],
) -> dict[str, Any]:
    capture_root = Path(layout["environment_capture_root"])
    capture_report = capture.verify_capture(capture_root)
    ownership_policy_path = Path(capture_report["ownership_policy_path"])
    integrity_policy_path = Path(
        capture_report["integrity_normalization_policy_path"]
    )
    try:
        ownership_policy, ownership_policy_sha256 = capture._load_policy(
            ownership_policy_path
        )
        integrity_policy, integrity_policy_sha256 = (
            capture._load_integrity_policy(integrity_policy_path)
        )
        policy = capture._combined_policy(
            ownership_policy, integrity_policy
        )
    except capture.EnvironmentCaptureError as exc:
        raise MaterializationPilotError(str(exc)) from exc
    recovered_record = _read_json(
        Path(capture_report["recovered_record_path"]),
        description="captured recovered Setuptools record",
    )
    roots = _pilot_roots(layout=layout)
    setuptools = {
        role: _setuptools_audit(
            roots[role],
            policy=policy,
            recovered_record=recovered_record,
        )
        for role in (
            "harness_seed",
            "serving_seed",
            "harness_prefix",
            "serving_prefix",
        )
    }
    return {
        "ownership_policy_sha256": ownership_policy_sha256,
        "integrity_normalization_policy_sha256": integrity_policy_sha256,
        "setuptools": setuptools,
        "sealed_inode_edges": _inode_edges(roots, _SEALED_INODE_EDGES),
        "control_consumer": _control_consumer_audit(
            Path(layout["release_bundle"])
        ),
    }


def _control_consumer_audit(release_bundle: Path) -> dict[str, Any]:
    """Feed the freezer's actual identity fragment to the production consumer."""

    identity_path = release_bundle / freeze.RELEASE_IDENTITY_FILENAME
    marker_path = release_bundle / freeze.COMPLETE_MARKER_FILENAME
    identity = _read_json(identity_path, description="frozen release identity")
    marker = _read_json(marker_path, description="frozen release completion marker")
    fragment = identity.get("control_pin_fragment")
    bundle_id = marker.get("release_bundle_id")
    if not isinstance(fragment, dict) or not isinstance(bundle_id, str):
        raise MaterializationPilotError(
            "frozen release lacks a control-consumable pin fragment"
        )
    pins = {
        **fragment,
        "release_bundle_root": str(release_bundle),
        "release_bundle_id": bundle_id,
    }
    try:
        # This is the same sealed-bundle boundary reached by prepare-pins before it
        # joins the three production manifests.  The pilot intentionally stops at
        # this static boundary so it never weakens or fabricates run cardinalities.
        control._validate_release_bundle(pins)
        control._validate_release_bundle(pins)
    except control.ImmutablePinError as exc:
        raise MaterializationPilotError(
            f"schema5_control rejected the frozen release: {exc}"
        ) from exc
    return {
        "validator": "schema5_control._validate_release_bundle",
        "validated_twice": True,
        "release_bundle_root": str(release_bundle),
        "release_bundle_id": bundle_id,
        "control_pin_fragment_sha256": _sha256_bytes(
            _canonical_bytes(fragment)
        ),
        "control_pin_fragment_fields": sorted(fragment),
    }


def _repeatable_verification(layout: Mapping[str, str]) -> dict[str, Any]:
    operations: dict[str, Callable[[], Mapping[str, Any]]] = {
        "capture": lambda: capture.verify_capture(
            layout["environment_capture_root"]
        ),
        "materialization": lambda: materialize.verify_materialization(
            layout["materialization_root"]
        ),
        "freeze": lambda: freeze.verify_release_bundle(layout["release_bundle"]),
    }
    reports: dict[str, Any] = {}
    for stage, operation in operations.items():
        first = _without_status(dict(operation()))
        second = _without_status(dict(operation()))
        if first != second:
            raise MaterializationPilotError(
                f"{stage} sealed verification is not repeatable"
            )
        reports[stage] = {
            "equal": True,
            "report_sha256": _sha256_bytes(_canonical_bytes(first)),
            "identity": first[
                {
                    "capture": "capture_id",
                    "materialization": "materialization_id",
                    "freeze": "release_bundle_id",
                }[stage]
            ],
        }
    return reports


def _sealed_verifier_runtime(layout: Mapping[str, str]) -> dict[str, Any]:
    """Return the exact sealed pilot harness used for independent verification."""

    prefix = _existing_directory(
        layout["harness_prefix"], description="pilot verifier harness"
    )
    release_root = _existing_directory(
        layout["release_bundle"], description="pilot release bundle"
    )
    identity = _read_json(
        release_root / freeze.RELEASE_IDENTITY_FILENAME,
        description="pilot frozen release identity",
    )
    environment = identity.get("environments", {}).get("harness")
    if not isinstance(environment, dict):
        raise MaterializationPilotError(
            "pilot release lacks a harness verifier environment"
        )
    manifest_path = Path(str(environment.get("manifest_path", ""))).resolve()
    expected_manifest = release_root / freeze.HARNESS_MANIFEST_FILENAME
    python_alias = prefix / "bin" / "python"
    try:
        python = python_alias.resolve(strict=True)
        python.relative_to(prefix)
    except (OSError, RuntimeError, ValueError) as exc:
        raise MaterializationPilotError(
            "pilot verifier Python escapes the sealed harness"
        ) from exc
    library = prefix / "lib"
    if (
        environment.get("prefix") != str(prefix)
        or manifest_path != expected_manifest
        or environment.get("manifest_sha256") != _sha256_file(manifest_path)
        or _SHA256_RE.fullmatch(
            str(environment.get("directory_inventory_sha256", ""))
        )
        is None
        or python.is_symlink()
        or not python.is_file()
        or not os.access(python, os.X_OK)
        or library.is_symlink()
        or not library.is_dir()
        or stat.S_IMODE(release_root.stat().st_mode) & 0o222
        or any(
            stat.S_IMODE(path.stat().st_mode) & 0o222
            for path in (prefix, python, library, manifest_path)
        )
    ):
        raise MaterializationPilotError(
            "pilot verifier runtime is not exact and read-only"
        )
    return {
        "harness_prefix": str(prefix),
        "environment_manifest_path": str(manifest_path),
        "environment_manifest_sha256": environment["manifest_sha256"],
        "directory_inventory_sha256": environment[
            "directory_inventory_sha256"
        ],
        "python_path": str(python),
        "python_sha256": _sha256_file(python),
        "python_size": python.stat().st_size,
        "library_path": str(library),
    }


def _write_inventory_evidence(
    root: Path,
    *,
    role: str,
    phase: str,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    filename = LIVE_INVENTORY_FILENAMES[(role, phase)]
    payload = _json_bytes(inventory)
    _atomic_write_once(root / filename, payload)
    return {
        "filename": filename,
        "sha256": _sha256_bytes(payload),
        "inventory_sha256": inventory["inventory_sha256"],
        "entry_count": inventory["entry_count"],
        "file_count": inventory["file_count"],
        "total_file_bytes": inventory["total_file_bytes"],
    }


def _read_inventory_evidence(
    root: Path, *, role: str, phase: str
) -> dict[str, Any]:
    filename = LIVE_INVENTORY_FILENAMES[(role, phase)]
    return _read_json(root / filename, description=f"{role} live {phase} inventory")


def _artifact_inventory(root: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for filename in sorted(EVIDENCE_FILENAMES):
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise MaterializationPilotError(
                f"pilot evidence is missing before completion: {path}"
            )
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise MaterializationPilotError(f"pilot evidence remains writable: {path}")
        artifacts[filename] = {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
    return artifacts


def _validate_artifact_inventory(root: Path, artifacts: Any) -> None:
    if not isinstance(artifacts, dict) or set(artifacts) != EVIDENCE_FILENAMES:
        raise MaterializationPilotError("pilot completion has the wrong evidence inventory")
    for filename, record in artifacts.items():
        path = root / filename
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256", "size"}
            or _SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
            or not isinstance(record.get("size"), int)
            or isinstance(record.get("size"), bool)
            or record["size"] < 1
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["size"]
            or _sha256_file(path) != record["sha256"]
            or stat.S_IMODE(path.stat().st_mode) & 0o222
        ):
            raise MaterializationPilotError(
                f"pilot evidence inventory drifted: {filename}"
            )


def _validate_intent(payload: Mapping[str, Any]) -> None:
    candidate = dict(payload)
    intent_id = candidate.pop("intent_id", None)
    inputs = payload.get("inputs")
    expected_input_fields = {
        "release_checkout",
        "harness_source",
        "serving_source",
        "source_package_cache",
        "ownership_policy",
        "integrity_normalization_policy",
        "reconciliation_incident",
        "recovered_setuptools_record",
        "conda_toolchain",
        "durable_git_release",
    }
    if (
        set(payload)
        != {
            "schema_version",
            "release_id",
            "kind",
            "publication_protocol",
            "expected_tag",
            "expected_commit",
            "git_identity",
            "inputs",
            "layout",
            "live_source_inventories",
            "conda_scope_contract",
            "intent_id",
        }
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("release_id") != RELEASE_ID
        or payload.get("kind") != "schema5-materialization-pilot-intent"
        or payload.get("publication_protocol")
        != "intent_first_stage_evidence_pilot_marker_last"
        or payload.get("expected_tag") != REQUIRED_TAG
        or not isinstance(payload.get("git_identity"), dict)
        or not isinstance(inputs, dict)
        or set(inputs) != expected_input_fields
        or not isinstance(payload.get("layout"), dict)
        or not isinstance(payload.get("live_source_inventories"), dict)
        or payload.get("conda_scope_contract")
        != {
            "capture_invokes_conda": False,
            "freeze_invokes_conda": False,
            "materialization_conda_source": (
                "normalized_read_only_seeds_and_immutable_preseeded_cache_only"
            ),
            "live_prefix_conda_queries": 0,
        }
        or intent_id != _sha256_bytes(_canonical_bytes(candidate))
    ):
        raise MaterializationPilotError("materialization pilot intent is invalid")


def _verify_completed_inputs(
    marker: Mapping[str, Any],
    *,
    expected_tag: str,
    expected_commit: str,
    git_identity: Mapping[str, Any],
    inputs: Mapping[str, Any],
    layout: Mapping[str, str],
) -> None:
    if (
        marker.get("expected_tag") != expected_tag
        or marker.get("expected_commit") != expected_commit
        or marker.get("git_identity") != git_identity
        or marker.get("inputs") != inputs
        or marker.get("layout") != layout
    ):
        raise MaterializationPilotError(
            "completed pilot belongs to different explicit inputs"
        )


def _safe_output_file(value: str | Path, *, description: str) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink() or lexical.is_dir():
        raise MaterializationPilotError(
            f"{description} is symlinked or a directory: {lexical}"
        )
    resolved = lexical.resolve()
    if resolved in {Path(resolved.anchor), Path.home().resolve()}:
        raise MaterializationPilotError(
            f"refusing unsafe broad {description}: {resolved}"
        )
    return resolved


def _safe_log_directory(value: str | Path) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink():
        raise MaterializationPilotError(f"pilot log directory is symlinked: {lexical}")
    resolved = lexical.resolve()
    if resolved in {Path(resolved.anchor), Path.home().resolve()}:
        raise MaterializationPilotError(
            f"refusing unsafe broad pilot log directory: {resolved}"
        )
    return resolved


def _sbatch_directive_value(value: str, *, description: str) -> str:
    if not isinstance(value, str) or _SBATCH_TOKEN_RE.fullmatch(value) is None:
        raise MaterializationPilotError(
            f"{description} contains unsafe Slurm-directive characters"
        )
    return value


def render_materialization_pilot_sbatch(
    *,
    sbatch_path: str | Path,
    log_dir: str | Path,
    partition: str,
    python_executable: str | Path,
    pilot_root: str | Path,
    release_checkout: str | Path,
    expected_tag: str,
    expected_commit: str,
    harness_source: str | Path,
    serving_source: str | Path,
    ownership_policy: str | Path,
    integrity_normalization_policy: str | Path,
    reconciliation_incident: str | Path,
    recovered_setuptools_record: str | Path,
    conda_toolchain_root: str | Path,
    source_package_cache: str | Path,
    durable_git_release_marker: str | Path,
    apply: bool = False,
) -> dict[str, Any]:
    """Render one immutable job for the transactional submit-sbatch workflow."""

    root = _destination_root(pilot_root)
    checkout = _existing_directory(release_checkout, description="release checkout")
    harness = _existing_directory(harness_source, description="live harness prefix")
    serving = _existing_directory(serving_source, description="live serving prefix")
    policy = _existing_file(ownership_policy, description="ownership policy")
    integrity_policy = _existing_file(
        integrity_normalization_policy,
        description="integrity-normalization policy",
    )
    incident = _existing_file(
        reconciliation_incident, description="Conda reconciliation incident"
    )
    recovered = _existing_file(
        recovered_setuptools_record,
        description="recovered Setuptools Conda record",
    )
    toolchain_root = _existing_directory(
        conda_toolchain_root,
        description="sealed Conda toolchain root",
    )
    package_cache_source = _existing_directory(
        source_package_cache, description="source Conda package cache"
    )
    durable_marker = _existing_file(
        durable_git_release_marker,
        description="durable Git release marker",
    )
    python = _existing_file(
        python_executable, description="pilot Python executable", executable=True
    )
    _validate_isolation(
        root,
        release_checkout=checkout,
        harness_source=harness,
        serving_source=serving,
        source_package_cache=package_cache_source,
        conda_toolchain_root=toolchain_root,
    )
    git_identity = _verify_exact_annotated_checkout(
        checkout, expected_tag=expected_tag, expected_commit=expected_commit
    )
    pilot_script = checkout / "scripts" / Path(__file__).name
    if pilot_script.is_symlink() or not pilot_script.is_file():
        raise MaterializationPilotError(
            f"exact tagged checkout lacks the pilot script: {pilot_script}"
        )
    tracked = _git(
        checkout,
        "ls-files",
        "--error-unmatch",
        pilot_script.relative_to(checkout).as_posix(),
    )
    if tracked != pilot_script.relative_to(checkout).as_posix():
        raise MaterializationPilotError(
            "pilot script is not tracked by the exact tagged checkout"
        )

    target = _safe_output_file(sbatch_path, description="pilot sbatch output")
    logs = _safe_log_directory(log_dir)
    for description, candidate in (
        ("pilot sbatch output", target),
        ("pilot log directory", logs),
    ):
        if (
            _is_relative_to(candidate, root)
            or _is_relative_to(candidate, checkout)
            or _is_relative_to(candidate, harness)
            or _is_relative_to(candidate, serving)
            or _is_relative_to(candidate, package_cache_source)
        ):
            raise MaterializationPilotError(
                f"{description} must remain outside pilot inputs and output root"
            )
    partition_value = _sbatch_directive_value(
        partition, description="Slurm partition"
    )
    output_log = logs / "schema5-materialization-pilot-%j.out"
    error_log = logs / "schema5-materialization-pilot-%j.err"
    for description, value in (
        ("Slurm output log", str(output_log)),
        ("Slurm error log", str(error_log)),
        ("Slurm checkout", str(checkout)),
    ):
        _sbatch_directive_value(value, description=description)

    inputs = _input_binding(
        release_checkout=checkout,
        harness_source=harness,
        serving_source=serving,
        ownership_policy=policy,
        integrity_normalization_policy=integrity_policy,
        reconciliation_incident=incident,
        recovered_setuptools_record=recovered,
        conda_toolchain_root=toolchain_root,
        source_package_cache=package_cache_source,
        durable_git_release_marker=durable_marker,
    )
    if (
        inputs["durable_git_release"]["release_git_commit"] != expected_commit
        or inputs["durable_git_release"]["release_tag_object"]
        != git_identity["tag_object"]
    ):
        raise MaterializationPilotError(
            "durable Git release marker belongs to another release identity"
        )
    launch_binding: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "kind": "schema5-materialization-pilot-sbatch",
        "expected_tag": expected_tag,
        "expected_commit": expected_commit,
        "git_identity": git_identity,
        "inputs": inputs,
        "pilot_root": str(root),
        "pilot_script": {
            "path": str(pilot_script),
            "sha256": _sha256_file(pilot_script),
        },
        "python_executable": {
            "path": str(python),
            "sha256": _sha256_file(python),
        },
        "slurm": {
            "job_name": SBATCH_JOB_NAME,
            "partition": partition_value,
            "time_limit": SBATCH_TIME_LIMIT,
            "no_requeue": True,
            "cpus_per_task": 1,
            "memory": "8G",
            "chdir": str(checkout),
            "output": str(output_log),
            "error": str(error_log),
        },
    }
    launch_binding["launch_id"] = _sha256_bytes(_canonical_bytes(launch_binding))
    comment = (
        "asys-s5-pilot:r4:"
        f"commit={expected_commit[:12]}:"
        f"launch={launch_binding['launch_id'][:16]}"
    )
    _sbatch_directive_value(comment, description="Slurm comment")
    launch_binding["slurm"]["comment"] = comment

    command = [
        str(python),
        str(pilot_script),
        "run",
        "--pilot-root",
        str(root),
        "--release-checkout",
        str(checkout),
        "--expected-tag",
        expected_tag,
        "--expected-commit",
        expected_commit,
        "--harness-source",
        str(harness),
        "--serving-source",
        str(serving),
        "--ownership-policy",
        str(policy),
        "--integrity-normalization-policy",
        str(integrity_policy),
        "--reconciliation-incident",
        str(incident),
        "--recovered-setuptools-record",
        str(recovered),
        "--conda-toolchain-root",
        str(toolchain_root),
        "--source-package-cache",
        str(package_cache_source),
        "--durable-git-release-marker",
        str(durable_marker),
        "--sbatch-receipt",
        str(target) + ".receipt.json",
        "--apply",
    ]
    script = "\n".join(
        (
            "#!/bin/bash",
            f"#SBATCH --job-name={SBATCH_JOB_NAME}",
            f"#SBATCH --partition={partition_value}",
            f"#SBATCH --time={SBATCH_TIME_LIMIT}",
            "#SBATCH --no-requeue",
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --mem=8G",
            "#SBATCH --open-mode=append",
            "#SBATCH --export=NONE",
            f"#SBATCH --chdir={checkout}",
            f"#SBATCH --output={output_log}",
            f"#SBATCH --error={error_log}",
            f"#SBATCH --comment={comment}",
            "",
            "set -euo pipefail",
            (
                "unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX "
                "CONDA_DEFAULT_ENV LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT "
                "BASH_ENV ENV CDPATH GIT_DIR GIT_WORK_TREE GIT_COMMON_DIR "
                "GIT_INDEX_FILE GIT_OBJECT_DIRECTORY "
                "GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_EXEC_PATH "
                "GIT_TEMPLATE_DIR GIT_NAMESPACE GIT_SSH GIT_SSH_COMMAND "
                "GIT_CONFIG GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS "
                "GIT_CEILING_DIRECTORIES GIT_DISCOVERY_ACROSS_FILESYSTEM "
                "GIT_REPLACE_REF_BASE GIT_SHALLOW_FILE SLURM_CONF "
                "SLURM_CLUSTERS SLURM_TIME_FORMAT"
            ),
            (
                "while IFS= read -r ambient_name; do case \"$ambient_name\" in "
                "PIP_*|CONDA_*|BASH_FUNC_*|GIT_CONFIG_KEY_*|"
                "GIT_CONFIG_VALUE_*|SBATCH_*|SACCT_*|SCONTROL_*|SQUEUE_*) "
                "unset \"$ambient_name\" ;; esac; "
                "done < <(compgen -e)"
            ),
            "export PYTHONNOUSERSITE=1",
            "export PYTHONDONTWRITEBYTECODE=1",
            f"export PATH={TRUSTED_SYSTEM_PATH}",
            "readonly PATH",
            (
                "export GIT_CONFIG_GLOBAL=/dev/null "
                "GIT_CONFIG_NOSYSTEM=1 GIT_ATTR_NOSYSTEM=1 "
                "GIT_NO_REPLACE_OBJECTS=1 GIT_TERMINAL_PROMPT=0 "
                "LC_ALL=C LANG=C"
            ),
            (
                "export PIP_CONFIG_FILE=/dev/null PIP_NO_INPUT=1 "
                "PIP_DISABLE_PIP_VERSION_CHECK=1"
            ),
            "export CONDARC=/dev/null CONDA_NO_PLUGINS=true",
            shlex.join(command),
            "",
        )
    ).encode("utf-8")
    script_sha256 = _sha256_bytes(script)
    receipt: dict[str, Any] = {
        **launch_binding,
        "sbatch_path": str(target),
        "sbatch_sha256": script_sha256,
        "publication_protocol": "sbatch_checksum_receipt_last",
        "submit_command": ["sbatch", str(target)],
    }
    receipt["receipt_id"] = _sha256_bytes(_canonical_bytes(receipt))
    report = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "launch_id": launch_binding["launch_id"],
        "receipt_id": receipt["receipt_id"],
        "expected_tag": expected_tag,
        "expected_commit": expected_commit,
        "sbatch_path": str(target),
        "sbatch_sha256": script_sha256,
        "receipt_path": str(target) + ".receipt.json",
        "log_paths": {
            "output": str(output_log),
            "error": str(error_log),
        },
        "slurm_comment": comment,
        "time_limit": SBATCH_TIME_LIMIT,
        "no_requeue": True,
        "submit_command": ["sbatch", str(target)],
    }
    if not apply:
        return {**report, "status": "dry_run"}

    target.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    was_complete = all(
        path.is_file() and not path.is_symlink()
        for path in (
            target,
            Path(str(target) + ".sha256"),
            Path(str(target) + ".receipt.json"),
        )
    )
    _atomic_write_once(target, script)
    _atomic_write_once(
        Path(str(target) + ".sha256"),
        f"{script_sha256}  {target.name}\n".encode("utf-8"),
    )
    _atomic_write_once(
        Path(str(target) + ".receipt.json"), _json_bytes(receipt)
    )
    if (
        _sha256_file(target) != script_sha256
        or _read_json(
            Path(str(target) + ".receipt.json"),
            description="pilot sbatch receipt",
        )
        != receipt
    ):
        raise MaterializationPilotError("published pilot sbatch evidence drifted")
    return {**report, "status": "already_complete" if was_complete else "created"}


def _validate_pilot_sbatch_receipt(
    receipt_path: str | Path,
    *,
    pilot_root: Path,
) -> tuple[dict[str, Any], Path]:
    path = _existing_file(
        receipt_path, description="materialization pilot sbatch receipt"
    )
    if stat.S_IMODE(path.stat().st_mode) & 0o222:
        raise MaterializationPilotError(
            "materialization pilot sbatch receipt must be read-only"
        )
    receipt = _read_json(path, description="materialization pilot sbatch receipt")
    identity = dict(receipt)
    receipt_id = identity.pop("receipt_id", None)
    sbatch_path = Path(str(receipt.get("sbatch_path", ""))).expanduser()
    try:
        sbatch_path = sbatch_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MaterializationPilotError(
            f"materialization pilot sbatch path is unavailable: {exc}"
        ) from exc
    slurm = receipt.get("slurm")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("release_id") != RELEASE_ID
        or receipt.get("kind") != "schema5-materialization-pilot-sbatch"
        or receipt.get("expected_tag") != REQUIRED_TAG
        or receipt.get("pilot_root") != str(pilot_root)
        or not isinstance(receipt_id, str)
        or _SHA256_RE.fullmatch(receipt_id) is None
        or receipt_id != _sha256_bytes(_canonical_bytes(identity))
        or not isinstance(slurm, dict)
        or slurm.get("job_name") != SBATCH_JOB_NAME
        or slurm.get("no_requeue") is not True
        or slurm.get("time_limit") != SBATCH_TIME_LIMIT
        or slurm.get("cpus_per_task") != 1
        or slurm.get("memory") != "8G"
        or receipt.get("submit_command") != ["sbatch", str(sbatch_path)]
        or receipt.get("sbatch_path") != str(sbatch_path)
        or _SHA256_RE.fullmatch(str(receipt.get("sbatch_sha256", ""))) is None
        or sbatch_path.is_symlink()
        or not sbatch_path.is_file()
        or stat.S_IMODE(sbatch_path.stat().st_mode) & 0o222
        or _sha256_file(sbatch_path) != receipt.get("sbatch_sha256")
    ):
        raise MaterializationPilotError(
            "materialization pilot sbatch receipt identity is invalid"
        )
    sbatch_text = sbatch_path.read_text(encoding="utf-8")
    if (
        "#SBATCH --no-requeue\n" not in sbatch_text
        or f"#SBATCH --time={SBATCH_TIME_LIMIT}\n" not in sbatch_text
        or f"#SBATCH --job-name={SBATCH_JOB_NAME}\n" not in sbatch_text
        or "#SBATCH --cpus-per-task=1\n" not in sbatch_text
        or "#SBATCH --mem=8G\n" not in sbatch_text
        or "#SBATCH --export=NONE\n" not in sbatch_text
        or f"#SBATCH --comment={slurm.get('comment')}\n" not in sbatch_text
    ):
        raise MaterializationPilotError(
            "materialization pilot sbatch no-requeue identity drifted"
        )
    return receipt, path


def _scheduler_runner(
    runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] | None,
) -> Callable[[Sequence[str]], subprocess.CompletedProcess[str]]:
    if runner is not None:
        return runner
    return lambda argv: subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        env=_sanitized_process_environment(),
    )


def _scheduler_query_evidence(
    argv: Sequence[str],
    process: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    """Retain exact scheduler query bytes next to every derived scheduler fact."""

    stdout = process.stdout
    stderr = process.stderr
    if (
        not isinstance(process.returncode, int)
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
    ):
        raise MaterializationPilotError(
            "scheduler query did not return text-mode evidence"
        )
    return {
        "argv": list(argv),
        "returncode": process.returncode,
        "stdout": stdout,
        "stdout_sha256": _sha256_bytes(stdout.encode("utf-8")),
        "stderr": stderr,
        "stderr_sha256": _sha256_bytes(stderr.encode("utf-8")),
    }


def _verify_scheduler_query_evidence(
    value: Any,
    *,
    expected_argv: Sequence[str],
    description: str,
) -> dict[str, Any]:
    required = {
        "argv",
        "returncode",
        "stdout",
        "stdout_sha256",
        "stderr",
        "stderr_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise MaterializationPilotError(
            f"{description} scheduler query evidence has invalid fields"
        )
    stdout = value.get("stdout")
    stderr = value.get("stderr")
    if (
        value.get("argv") != list(expected_argv)
        or not isinstance(value.get("returncode"), int)
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or value.get("stdout_sha256")
        != _sha256_bytes(stdout.encode("utf-8"))
        or value.get("stderr_sha256")
        != _sha256_bytes(stderr.encode("utf-8"))
    ):
        raise MaterializationPilotError(
            f"{description} scheduler query evidence drifted"
        )
    return value


def _submission_lock_path(root: Path) -> Path:
    return root.parent / f".{root.name}.pilot-submission.lock"


@contextmanager
def _pilot_submission_lock(root: Path):
    """Serialize the one canonical pilot submission across login nodes."""

    parent = root.parent
    if parent.is_symlink():
        raise MaterializationPilotError(
            f"pilot submission parent is symlinked: {parent}"
        )
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = _submission_lock_path(root)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise MaterializationPilotError(
            f"cannot open pilot submission lock {lock_path}: {exc}"
        ) from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MaterializationPilotError(
                f"another pilot submission transaction owns {lock_path}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _submission_scheduler_since(timestamp: float) -> str:
    if (
        not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or timestamp < 0
    ):
        raise MaterializationPilotError(
            "pilot submission timestamp is invalid"
        )
    # Include a small clock-skew margin. The immutable receipt comment and exact
    # SubmitLine, rather than this time window, identify the allocation.
    # Slurm parses an offsetless ``sacct -S`` value in the scheduler client's
    # local timezone. Emitting a UTC wall clock without an offset can therefore
    # move the requested start into the future on an EDT/EST login node.
    return datetime.fromtimestamp(
        max(0.0, float(timestamp) - 300.0), tz=timezone.utc
    ).astimezone().strftime("%Y-%m-%dT%H:%M:%S")


def _validate_retired_quarantine_scheduler(
    value: Any,
    *,
    job_id: str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-prove the terminal transient bound into one sealed quarantine."""

    required = {
        "job_id",
        "job_name",
        "raw_state",
        "normalized_state",
        "exit_code",
        "reason",
        "accounting_comment",
        "expected_comment",
        "submit_line",
        "transient_classification",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise MaterializationPilotError(
            "retired pilot quarantine scheduler evidence has invalid fields"
        )
    raw_state = value.get("raw_state")
    normalized_state = (
        raw_state.split()[0].rstrip("+")
        if isinstance(raw_state, str) and raw_state
        else ""
    )
    if normalized_state in _INHERENT_TRANSIENT_STATES:
        expected_classification = "scheduler_or_node_transient"
    elif (
        normalized_state == "CANCELLED"
        and isinstance(raw_state, str)
        and _EXTERNAL_CANCELLATION_RE.fullmatch(raw_state) is not None
    ):
        expected_classification = "external_cancellation"
    else:
        raise MaterializationPilotError(
            "retired pilot quarantine is not an allowed scheduler transient"
        )
    try:
        submit_tokens = shlex.split(str(value.get("submit_line", "")))
    except ValueError as exc:
        raise MaterializationPilotError(
            f"retired pilot quarantine SubmitLine is malformed: {exc}"
        ) from exc
    expected_comment = str(receipt["slurm"]["comment"])
    accounting_comment = value.get("accounting_comment")
    if (
        value.get("job_id") != job_id
        or value.get("job_name") != SBATCH_JOB_NAME
        or value.get("normalized_state") != normalized_state
        or value.get("transient_classification")
        != expected_classification
        or not isinstance(value.get("exit_code"), str)
        or _SLURM_EXIT_CODE_RE.fullmatch(value["exit_code"]) is None
        or not isinstance(value.get("reason"), str)
        or not value["reason"]
        or value.get("expected_comment") != expected_comment
        or accounting_comment not in {"", expected_comment}
        or submit_tokens != ["sbatch", str(receipt["sbatch_path"])]
    ):
        raise MaterializationPilotError(
            "retired pilot quarantine scheduler identity drifted"
        )
    return dict(value)


def _validate_retired_quarantine(
    seal_value: str | Path,
    *,
    root: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    lineage_stack: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    """Verify one explicit marker-last quarantine lineage entry."""

    seal_path = _existing_file(
        seal_value, description="retired pilot quarantine seal"
    )
    if seal_path in lineage_stack:
        raise MaterializationPilotError(
            "retired pilot quarantine lineage contains a cycle"
        )
    if seal_path.is_symlink() or stat.S_IMODE(seal_path.stat().st_mode) & 0o222:
        raise MaterializationPilotError(
            "retired pilot quarantine seal must be a read-only regular file"
        )
    seal = _read_json(
        seal_path, description="retired pilot quarantine seal"
    )
    seal_identity = dict(seal)
    seal_id = seal_identity.pop("seal_id", None)
    required_seal = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "failed_job_id",
        "tree",
        "tree_device",
        "tree_inode",
        "content_sha256",
        "inventory",
        "inventory_sha256",
        "file_count",
        "total_bytes",
        "intent",
        "intent_sha256",
        "quarantine_completion",
        "quarantine_completion_sha256",
        "sealed_at",
    }
    job_id = str(seal.get("failed_job_id", ""))
    expected_seal_path = (
        root.parent
        / "quarantine_evidence"
        / f"partial-job-{job_id}.sealed.json"
    )
    expected_tree = (
        root.parent
        / "quarantine"
        / f"{root.name}.partial-job-{job_id}"
    )
    tree_path = _existing_directory(
        expected_tree, description="retired pilot quarantine tree"
    )
    tree_stat = os.lstat(tree_path)
    expected_inventory = (
        seal_path.parent / f"partial-job-{job_id}.sealed-inventory.txt"
    )
    expected_seal_intent = (
        seal_path.parent / f"partial-job-{job_id}.seal-intent.json"
    )
    inventory_path = _existing_file(
        Path(str(seal.get("inventory", ""))),
        description="retired pilot quarantine inventory",
    )
    seal_intent_path = _existing_file(
        Path(str(seal.get("intent", ""))),
        description="retired pilot quarantine seal intent",
    )
    if (
        set(seal_identity) != required_seal
        or seal.get("schema_version") != 1
        or seal.get("protocol")
        != "schema5-quarantined-materialization-seal-v1"
        or seal.get("passed") is not True
        or seal.get("release_id") != RELEASE_ID
        or not job_id.isdigit()
        or seal_path != expected_seal_path
        or seal.get("tree") != str(expected_tree)
        or tree_path != expected_tree
        or seal.get("tree_device") != tree_stat.st_dev
        or seal.get("tree_inode") != tree_stat.st_ino
        or inventory_path != expected_inventory
        or seal.get("inventory_sha256") != _sha256_file(inventory_path)
        or stat.S_IMODE(inventory_path.stat().st_mode) & 0o222
        or seal_intent_path != expected_seal_intent
        or seal.get("intent_sha256") != _sha256_file(seal_intent_path)
        or stat.S_IMODE(seal_intent_path.stat().st_mode) & 0o222
        or not isinstance(seal_id, str)
        or _SHA256_RE.fullmatch(seal_id) is None
        or seal_id
        != _sha256_bytes(recovery_evidence._canonical_json(seal_identity))
    ):
        raise MaterializationPilotError(
            "retired pilot quarantine seal identity drifted"
        )

    completion_path = _existing_file(
        Path(str(seal["quarantine_completion"])),
        description="retired pilot quarantine completion",
    )
    expected_completion_path = (
        root.parent
        / "quarantine_evidence"
        / f"partial-job-{job_id}.complete.json"
    )
    completion = _read_json(
        completion_path, description="retired pilot quarantine completion"
    )
    completion_identity = dict(completion)
    completion_id = completion_identity.pop("completion_id", None)
    required_completion = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "materialize_job_id",
        "pilot_root",
        "source",
        "destination",
        "sbatch_receipt",
        "sbatch_receipt_sha256",
        "intent",
        "intent_sha256",
        "intent_id",
        "scheduler",
        "source_device",
        "source_inode",
        "completed_at",
    }
    if (
        set(completion_identity) != required_completion
        or completion.get("schema_version") != 1
        or completion.get("protocol") != PILOT_QUARANTINE_PROTOCOL
        or completion.get("passed") is not True
        or completion.get("release_id") != RELEASE_ID
        or completion.get("release_tag") != REQUIRED_TAG
        or completion.get("materialize_job_id") != job_id
        or completion.get("pilot_root") != str(root)
        or completion.get("source") != str(root)
        or completion.get("destination") != str(expected_tree)
        or completion.get("sbatch_receipt") != str(receipt_path)
        or completion.get("sbatch_receipt_sha256")
        != _sha256_file(receipt_path)
        or completion.get("source_device") != tree_stat.st_dev
        or completion.get("source_inode") != tree_stat.st_ino
        or not isinstance(completion.get("completed_at"), str)
        or completion_path != expected_completion_path
        or seal.get("quarantine_completion") != str(completion_path)
        or seal.get("quarantine_completion_sha256")
        != _sha256_file(completion_path)
        or not isinstance(completion_id, str)
        or _SHA256_RE.fullmatch(completion_id) is None
        or completion_id
        != _sha256_bytes(_canonical_bytes(completion_identity))
        or completion_path.is_symlink()
        or stat.S_IMODE(completion_path.stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError(
            "retired pilot quarantine completion identity drifted"
        )

    scheduler = _validate_retired_quarantine_scheduler(
        completion.get("scheduler"),
        job_id=job_id,
        receipt=receipt,
    )
    intent_path = _existing_file(
        Path(str(completion["intent"])),
        description="retired pilot quarantine intent",
    )
    intent = _read_json(
        intent_path, description="retired pilot quarantine intent"
    )
    intent_identity = dict(intent)
    intent_id = intent_identity.pop("intent_id", None)
    required_intent = {
        "schema_version",
        "protocol",
        "release_id",
        "release_tag",
        "pilot_root",
        "destination",
        "sbatch_receipt",
        "sbatch_receipt_sha256",
        "sbatch_receipt_id",
        "scheduler",
        "source_device",
        "source_inode",
        "created_at",
    }
    expected_intent_path = (
        root.parent
        / "quarantine_evidence"
        / f"partial-job-{job_id}.intent.json"
    )
    if (
        set(intent_identity) != required_intent
        or intent.get("schema_version") != 1
        or intent.get("protocol") != PILOT_QUARANTINE_INTENT_PROTOCOL
        or intent.get("release_id") != RELEASE_ID
        or intent.get("release_tag") != REQUIRED_TAG
        or intent.get("pilot_root") != str(root)
        or intent.get("destination") != str(expected_tree)
        or intent.get("sbatch_receipt") != str(receipt_path)
        or intent.get("sbatch_receipt_sha256")
        != _sha256_file(receipt_path)
        or intent.get("sbatch_receipt_id") != receipt["receipt_id"]
        or intent.get("scheduler") != scheduler
        or intent.get("source_device") != tree_stat.st_dev
        or intent.get("source_inode") != tree_stat.st_ino
        or not isinstance(intent.get("created_at"), str)
        or intent_path != expected_intent_path
        or completion.get("intent") != str(intent_path)
        or completion.get("intent_sha256") != _sha256_file(intent_path)
        or completion.get("intent_id") != intent_id
        or not isinstance(intent_id, str)
        or _SHA256_RE.fullmatch(intent_id) is None
        or intent_id != _sha256_bytes(_canonical_bytes(intent_identity))
        or intent_path.is_symlink()
        or stat.S_IMODE(intent_path.stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError(
            "retired pilot quarantine intent identity drifted"
        )

    submission_marker_path = _existing_file(
        tree_path / SUBMISSION_ACCEPTED_FILENAME,
        description="retired pilot accepted submission",
    )
    submission_marker = _read_json(
        submission_marker_path,
        description="retired pilot accepted submission",
    )
    submission_identity = dict(submission_marker)
    acceptance_id = submission_identity.pop("acceptance_id", None)
    required_submission = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "pilot_root",
        "submission_intent",
        "submission_intent_sha256",
        "submission_intent_id",
        "attempt",
        "attempt_intent",
        "attempt_intent_sha256",
        "attempt_id",
        "sbatch_result",
        "sbatch_result_sha256",
        "sbatch_result_id",
        "sbatch_receipt",
        "sbatch_receipt_sha256",
        "sbatch_receipt_id",
        "sbatch_path",
        "sbatch_sha256",
        "scheduler_user",
        "scheduler_comment",
        "job_id",
        "job_name",
        "partition",
        "time_limit",
        "cpus_per_task",
        "memory",
        "scheduler_sources",
        "scheduler_snapshot",
        "accepted_at",
    }
    relocated_intent_path = _existing_file(
        tree_path / SUBMISSION_INTENT_FILENAME,
        description="retired pilot submission intent",
    )
    raw_submission_intent = _read_json(
        relocated_intent_path,
        description="retired pilot submission intent",
    )
    prior_lineage = raw_submission_intent.get("retired_quarantines")
    if not isinstance(prior_lineage, list) or any(
        not isinstance(record, dict)
        or not isinstance(record.get("quarantine_seal"), str)
        for record in prior_lineage
    ):
        raise MaterializationPilotError(
            "retired pilot submission ancestry is invalid"
        )
    submission_intent = _load_submission_intent(
        relocated_intent_path,
        expected=_submission_intent_stable(
            root=root,
            receipt=receipt,
            receipt_path=receipt_path,
            scheduler_user=str(
                raw_submission_intent.get("scheduler_user", "")
            ),
            prior_quarantine_seals=[
                record["quarantine_seal"] for record in prior_lineage
            ],
            _lineage_stack=lineage_stack | {seal_path},
        ),
    )
    scheduler_candidates = _submission_candidates_from_snapshot(
        submission_marker.get("scheduler_snapshot"),
        intent=submission_intent,
    )
    attempt_number = submission_marker.get("attempt")
    if (
        not isinstance(attempt_number, int)
        or isinstance(attempt_number, bool)
        or attempt_number <= 0
    ):
        raise MaterializationPilotError(
            "retired pilot accepted submission attempt is invalid"
        )

    def relocated_recorded_file(
        recorded: Any,
        *,
        description: str,
    ) -> Path:
        if not isinstance(recorded, str):
            raise MaterializationPilotError(
                f"{description} path is invalid"
            )
        recorded_path = Path(recorded)
        try:
            relative = recorded_path.relative_to(root)
        except ValueError as exc:
            raise MaterializationPilotError(
                f"{description} escaped the canonical pilot root"
            ) from exc
        relocated = _existing_file(
            tree_path / relative,
            description=description,
        )
        if not _is_relative_to(relocated, tree_path):
            raise MaterializationPilotError(
                f"{description} escaped the quarantined pilot tree"
            )
        return relocated

    relocated_attempt_path = relocated_recorded_file(
        submission_marker.get("attempt_intent"),
        description="retired pilot submission attempt intent",
    )
    attempt_intent = _load_submission_attempt_intent(
        relocated_attempt_path,
        global_intent=submission_intent,
        attempt=attempt_number,
    )
    recorded_result = submission_marker.get("sbatch_result")
    if recorded_result is None:
        submission_result = None
        relocated_result_path = None
    else:
        relocated_result_path = relocated_recorded_file(
            recorded_result,
            description="retired pilot sbatch result",
        )
        submission_result = _load_submission_result(
            relocated_result_path,
            attempt_intent=attempt_intent,
        )
    if (
        set(submission_identity) != required_submission
        or submission_marker.get("schema_version") != 1
        or submission_marker.get("protocol") != SUBMISSION_ACCEPTED_PROTOCOL
        or submission_marker.get("passed") is not True
        or submission_marker.get("release_id") != RELEASE_ID
        or submission_marker.get("release_tag") != REQUIRED_TAG
        or submission_marker.get("pilot_root") != str(root)
        or submission_marker.get("submission_intent")
        != str(root / SUBMISSION_INTENT_FILENAME)
        or submission_marker.get("submission_intent_sha256")
        != _sha256_file(relocated_intent_path)
        or submission_marker.get("submission_intent_id")
        != submission_intent["intent_id"]
        or submission_marker.get("attempt_intent_sha256")
        != _sha256_file(relocated_attempt_path)
        or submission_marker.get("attempt_id")
        != attempt_intent["attempt_id"]
        or submission_marker.get("sbatch_result_sha256")
        != (
            None
            if relocated_result_path is None
            else _sha256_file(relocated_result_path)
        )
        or submission_marker.get("sbatch_result_id")
        != (
            None
            if submission_result is None
            else submission_result["result_id"]
        )
        or submission_marker.get("sbatch_receipt") != str(receipt_path)
        or submission_marker.get("sbatch_receipt_sha256")
        != _sha256_file(receipt_path)
        or submission_marker.get("sbatch_receipt_id")
        != receipt["receipt_id"]
        or submission_marker.get("sbatch_path") != receipt["sbatch_path"]
        or submission_marker.get("sbatch_sha256")
        != receipt["sbatch_sha256"]
        or submission_marker.get("scheduler_comment")
        != receipt["slurm"]["comment"]
        or submission_marker.get("job_id") != job_id
        or submission_marker.get("job_name") != SBATCH_JOB_NAME
        or submission_marker.get("partition")
        != receipt["slurm"]["partition"]
        or submission_marker.get("time_limit")
        != receipt["slurm"]["time_limit"]
        or submission_marker.get("cpus_per_task")
        != receipt["slurm"]["cpus_per_task"]
        or submission_marker.get("memory") != receipt["slurm"]["memory"]
        or len(scheduler_candidates) != 1
        or scheduler_candidates[0].get("job_id") != job_id
        or submission_marker.get("scheduler_sources")
        != scheduler_candidates[0].get("sources")
        or not isinstance(submission_marker.get("accepted_at"), str)
        or not isinstance(acceptance_id, str)
        or _SHA256_RE.fullmatch(acceptance_id) is None
        or acceptance_id
        != _sha256_bytes(_canonical_bytes(submission_identity))
        or stat.S_IMODE(submission_marker_path.stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError(
            "retired pilot accepted submission identity drifted"
        )

    try:
        verified_seal = recovery_evidence.seal_quarantine(
            tree=expected_tree,
            evidence_root=seal_path.parent,
            release_id=RELEASE_ID,
            failed_job_id=job_id,
            quarantine_completion=completion_path,
            apply=False,
        )
    except recovery_evidence.EvidenceError as exc:
        raise MaterializationPilotError(
            f"retired pilot quarantine seal verification failed: {exc}"
        ) from exc
    if (
        verified_seal.get("status") != "already_sealed"
        or verified_seal.get("seal_id") != seal_id
    ):
        raise MaterializationPilotError(
            "retired pilot quarantine seal did not reverify"
        )
    return {
        "job_id": job_id,
        "quarantine_seal": str(seal_path),
        "quarantine_seal_sha256": _sha256_file(seal_path),
        "seal_id": seal_id,
        "quarantine_completion": str(completion_path),
        "quarantine_completion_sha256": _sha256_file(completion_path),
        "completion_id": completion_id,
        "quarantine_intent": str(intent_path),
        "quarantine_intent_sha256": _sha256_file(intent_path),
        "intent_id": intent_id,
        "tree": str(expected_tree),
        "content_sha256": seal["content_sha256"],
        "inventory_sha256": seal["inventory_sha256"],
        "terminal": {
            "raw_state": scheduler["raw_state"],
            "normalized_state": scheduler["normalized_state"],
            "exit_code": scheduler["exit_code"],
            "reason": scheduler["reason"],
            "transient_classification": scheduler[
                "transient_classification"
            ],
        },
        "submission_ancestors": [
            {
                "job_id": record["job_id"],
                "quarantine_seal": record["quarantine_seal"],
                "quarantine_seal_sha256": record[
                    "quarantine_seal_sha256"
                ],
                "seal_id": record["seal_id"],
            }
            for record in prior_lineage
        ],
    }


def _validate_retired_quarantines(
    values: Sequence[str | Path],
    *,
    root: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    lineage_stack: frozenset[Path] = frozenset(),
) -> list[dict[str, Any]]:
    if isinstance(values, (str, bytes, Path)):
        raise MaterializationPilotError(
            "retired pilot quarantine lineage must be a sequence of seals"
        )
    records = [
        _validate_retired_quarantine(
            value,
            root=root,
            receipt=receipt,
            receipt_path=receipt_path,
            lineage_stack=lineage_stack,
        )
        for value in values
    ]
    records.sort(key=lambda value: (int(value["job_id"]), value["quarantine_seal"]))
    if len({record["job_id"] for record in records}) != len(records):
        raise MaterializationPilotError(
            "retired pilot quarantine lineage repeats a job ID"
        )
    if len({record["quarantine_seal"] for record in records}) != len(records):
        raise MaterializationPilotError(
            "retired pilot quarantine lineage repeats a seal"
        )
    for index, record in enumerate(records):
        ancestors = record.get("submission_ancestors")
        if not isinstance(ancestors, list):
            raise MaterializationPilotError(
                "retired pilot quarantine ancestry is invalid"
            )
        if any(
            not isinstance(ancestor, dict)
            or set(ancestor)
            != {
                "job_id",
                "quarantine_seal",
                "quarantine_seal_sha256",
                "seal_id",
            }
            for ancestor in ancestors
        ):
            raise MaterializationPilotError(
                "retired pilot quarantine ancestry is invalid"
            )
        expected_ancestors = [
            {
                "job_id": current["job_id"],
                "quarantine_seal": current["quarantine_seal"],
                "quarantine_seal_sha256": current[
                    "quarantine_seal_sha256"
                ],
                "seal_id": current["seal_id"],
            }
            for current in records[:index]
        ]
        if ancestors != expected_ancestors:
            raise MaterializationPilotError(
                "retired pilot quarantine lineage is not the complete ordered "
                "prefix of accepted retries"
            )
    return records


def _submission_intent_stable(
    *,
    root: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    scheduler_user: str,
    prior_quarantine_seals: Sequence[str | Path] = (),
    _lineage_stack: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    if not scheduler_user or not re.fullmatch(r"[A-Za-z0-9_.-]+", scheduler_user):
        raise MaterializationPilotError(
            "pilot submission scheduler user is invalid"
        )
    if receipt.get("submit_command") != [
        "sbatch",
        receipt.get("sbatch_path"),
    ]:
        raise MaterializationPilotError(
            "pilot submission requires an exact two-token sbatch command"
        )
    return {
        "schema_version": 1,
        "protocol": SUBMISSION_INTENT_PROTOCOL,
        "release_id": RELEASE_ID,
        "release_tag": REQUIRED_TAG,
        "pilot_root": str(root),
        "scheduler_user": scheduler_user,
        "job_name": SBATCH_JOB_NAME,
        "scheduler_comment": receipt["slurm"]["comment"],
        "partition": receipt["slurm"]["partition"],
        "time_limit": receipt["slurm"]["time_limit"],
        "cpus_per_task": receipt["slurm"]["cpus_per_task"],
        "memory": receipt["slurm"]["memory"],
        "sbatch_receipt": str(receipt_path),
        "sbatch_receipt_sha256": _sha256_file(receipt_path),
        "sbatch_receipt_id": receipt["receipt_id"],
        "sbatch_path": receipt["sbatch_path"],
        "sbatch_sha256": receipt["sbatch_sha256"],
        "submit_argv": receipt["submit_command"],
        "retired_quarantines": _validate_retired_quarantines(
            prior_quarantine_seals,
            root=root,
            receipt=receipt,
            receipt_path=receipt_path,
            lineage_stack=_lineage_stack,
        ),
    }


def _load_submission_intent(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path, description="pilot submission intent")
    candidate = dict(payload)
    intent_id = candidate.pop("intent_id", None)
    created_at = candidate.pop("created_at", None)
    created_timestamp = candidate.pop("created_timestamp", None)
    scheduler_since = candidate.pop("scheduler_since", None)
    if (
        candidate != dict(expected)
        or not isinstance(created_at, str)
        or not isinstance(created_timestamp, (int, float))
        or isinstance(created_timestamp, bool)
        or scheduler_since
        != _submission_scheduler_since(float(created_timestamp))
        or not isinstance(intent_id, str)
        or _SHA256_RE.fullmatch(intent_id) is None
        or intent_id
        != _sha256_bytes(
            _canonical_bytes(
                dict(expected)
                | {
                    "created_at": created_at,
                    "created_timestamp": float(created_timestamp),
                    "scheduler_since": scheduler_since,
                }
            )
        )
        or path.is_symlink()
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError(
            "pilot submission intent identity drifted"
        )
    return payload


def _load_or_create_submission_intent(
    *,
    root: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    scheduler_user: str,
    prior_quarantine_seals: Sequence[str | Path],
    now: float,
) -> tuple[dict[str, Any], bool]:
    path = root / SUBMISSION_INTENT_FILENAME
    expected = _submission_intent_stable(
        root=root,
        receipt=receipt,
        receipt_path=receipt_path,
        scheduler_user=scheduler_user,
        prior_quarantine_seals=prior_quarantine_seals,
    )
    if path.exists() or path.is_symlink():
        return _load_submission_intent(path, expected=expected), False
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise MaterializationPilotError(
                f"pilot submission root is unsafe: {root}"
            )
        unexpected = [item.name for item in root.iterdir()]
        if unexpected:
            raise MaterializationPilotError(
                "pilot submission intent must precede every pilot-root artifact: "
                + ", ".join(sorted(unexpected))
            )
    else:
        root.mkdir(parents=True)
    created_timestamp = float(now)
    payload = {
        **expected,
        "created_at": datetime.fromtimestamp(
            created_timestamp, tz=timezone.utc
        ).isoformat(),
        "created_timestamp": created_timestamp,
        "scheduler_since": _submission_scheduler_since(created_timestamp),
    }
    payload["intent_id"] = _sha256_bytes(_canonical_bytes(payload))
    _atomic_write_once(path, _json_bytes(payload))
    return _load_submission_intent(path, expected=expected), True


def _submission_attempt_directory(root: Path, attempt: int) -> Path:
    return (
        root
        / SUBMISSION_DIRECTORY
        / SUBMISSION_ATTEMPTS_DIRECTORY
        / f"a{attempt:04d}"
    )


def _submission_attempt_paths(root: Path, attempt: int) -> dict[str, Path]:
    directory = _submission_attempt_directory(root, attempt)
    return {
        "root": directory,
        "intent": directory / SUBMISSION_ATTEMPT_INTENT_FILENAME,
        "result": directory / SUBMISSION_RESULT_FILENAME,
        "absent": directory / SUBMISSION_ABSENT_FILENAME,
    }


def _load_submission_attempt_intent(
    path: Path,
    *,
    global_intent: Mapping[str, Any],
    attempt: int,
) -> dict[str, Any]:
    payload = _read_json(path, description="pilot submission attempt intent")
    candidate = dict(payload)
    attempt_id = candidate.pop("attempt_id", None)
    expected = {
        "schema_version": 1,
        "protocol": SUBMISSION_ATTEMPT_PROTOCOL,
        "pilot_root": global_intent["pilot_root"],
        "submission_intent": str(
            Path(global_intent["pilot_root"]) / SUBMISSION_INTENT_FILENAME
        ),
        "submission_intent_id": global_intent["intent_id"],
        "attempt": attempt,
        "submit_argv": global_intent["submit_argv"],
        "scheduler_comment": global_intent["scheduler_comment"],
        "started_timestamp": payload.get("started_timestamp"),
        "visibility_deadline": payload.get("visibility_deadline"),
    }
    if (
        set(candidate) != set(expected)
        or candidate != expected
        or not isinstance(candidate["started_timestamp"], (int, float))
        or isinstance(candidate["started_timestamp"], bool)
        or not isinstance(candidate["visibility_deadline"], (int, float))
        or isinstance(candidate["visibility_deadline"], bool)
        or float(candidate["visibility_deadline"])
        < float(candidate["started_timestamp"])
        or not isinstance(attempt_id, str)
        or _SHA256_RE.fullmatch(attempt_id) is None
        or attempt_id != _sha256_bytes(_canonical_bytes(candidate))
        or path.is_symlink()
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError(
            "pilot submission attempt intent identity drifted"
        )
    return payload


def _load_submission_result(
    path: Path,
    *,
    attempt_intent: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path, description="pilot sbatch result")
    candidate = dict(payload)
    result_id = candidate.pop("result_id", None)
    stdout = candidate.get("stdout")
    stderr = candidate.get("stderr")
    if (
        set(candidate)
        != {
            "schema_version",
            "protocol",
            "attempt",
            "attempt_id",
            "argv",
            "returncode",
            "stdout",
            "stdout_sha256",
            "stderr",
            "stderr_sha256",
            "reported_job_id",
            "completed_at",
        }
        or candidate.get("schema_version") != 1
        or candidate.get("protocol") != SUBMISSION_RESULT_PROTOCOL
        or candidate.get("attempt") != attempt_intent["attempt"]
        or candidate.get("attempt_id") != attempt_intent["attempt_id"]
        or candidate.get("argv") != attempt_intent["submit_argv"]
        or not isinstance(candidate.get("returncode"), int)
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or candidate.get("stdout_sha256")
        != _sha256_bytes(stdout.encode("utf-8"))
        or candidate.get("stderr_sha256")
        != _sha256_bytes(stderr.encode("utf-8"))
        or (
            candidate.get("reported_job_id") is not None
            and (
                not isinstance(candidate["reported_job_id"], str)
                or not candidate["reported_job_id"].isdigit()
            )
        )
        or not isinstance(candidate.get("completed_at"), str)
        or not isinstance(result_id, str)
        or _SHA256_RE.fullmatch(result_id) is None
        or result_id != _sha256_bytes(_canonical_bytes(candidate))
        or path.is_symlink()
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError(
            "pilot sbatch result identity drifted"
        )
    return payload


def _submission_query_argv(intent: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        "squeue": [
            "squeue",
            "-u",
            str(intent["scheduler_user"]),
            "-h",
            "-r",
            "-o",
            "%i|%j|%T|%k|%P|%l|%C|%m|%o",
        ],
        "sacct": [
            "sacct",
            "-u",
            str(intent["scheduler_user"]),
            "-X",
            "-n",
            "-P",
            "-S",
            str(intent["scheduler_since"]),
            (
                "--format=JobIDRaw,JobName%64,State,ExitCode,Reason,"
                "Comment%256,Partition,Timelimit,SubmitLine%1024"
            ),
        ],
    }


def _normalize_scheduler_comment(value: str) -> str:
    stripped = value.strip()
    return "" if stripped.lower() in {"", "(null)", "null", "none"} else stripped


def _slurm_memory_bytes(value: str) -> int:
    """Normalize the common squeue memory spellings to an exact byte count."""

    match = _SLURM_MEMORY_RE.fullmatch(value.strip())
    if match is None:
        raise MaterializationPilotError(
            f"invalid Slurm memory value in pilot submission truth: {value!r}"
        )
    amount = float(match.group("amount"))
    unit = match.group("unit").upper()
    multiplier = {
        "": 1024**2,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }[unit]
    normalized = amount * multiplier
    if not normalized.is_integer():
        raise MaterializationPilotError(
            f"non-integral Slurm memory value in pilot submission truth: {value!r}"
        )
    return int(normalized)


def _submission_candidates_from_snapshot(
    snapshot: Any,
    *,
    intent: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if (
        not isinstance(snapshot, dict)
        or set(snapshot) != {"captured_at", "squeue", "sacct"}
        or not isinstance(snapshot.get("captured_at"), (int, float))
        or isinstance(snapshot.get("captured_at"), bool)
    ):
        raise MaterializationPilotError(
            "pilot submission scheduler snapshot has invalid fields"
        )
    commands = _submission_query_argv(intent)
    query_records = {
        source: _verify_scheduler_query_evidence(
            snapshot.get(source),
            expected_argv=commands[source],
            description=f"pilot submission {source}",
        )
        for source in ("squeue", "sacct")
    }
    if any(record["returncode"] != 0 for record in query_records.values()):
        raise MaterializationPilotError(
            "pilot submission requires complete squeue+sacct truth"
        )

    expected_comment = str(intent["scheduler_comment"])
    expected_name = str(intent["job_name"])
    expected_partition = str(intent["partition"])
    expected_limit = str(intent["time_limit"])
    expected_cpus = str(intent["cpus_per_task"])
    expected_memory = str(intent["memory"])
    expected_sbatch = str(intent["sbatch_path"])
    expected_submit = [str(item) for item in intent["submit_argv"]]
    rows_by_source: dict[str, dict[str, dict[str, Any]]] = {
        "squeue": {},
        "sacct": {},
    }

    for raw in query_records["squeue"]["stdout"].splitlines():
        if not raw.strip():
            continue
        fields = raw.split("|", 8)
        if len(fields) != 9:
            raise MaterializationPilotError(
                f"malformed pilot submission squeue row: {raw!r}"
            )
        (
            job_id,
            job_name,
            state,
            comment,
            partition,
            time_limit,
            cpus,
            memory,
            command,
        ) = (field.strip() for field in fields)
        if not job_id.isdigit():
            continue
        normalized_comment = _normalize_scheduler_comment(comment)
        related = normalized_comment == expected_comment or (
            job_name == expected_name and command == expected_sbatch
        )
        if not related:
            continue
        if (
            normalized_comment != expected_comment
            or job_name != expected_name
            or command != expected_sbatch
            or partition != expected_partition
            or time_limit != expected_limit
            or cpus != expected_cpus
            or _slurm_memory_bytes(memory)
            != _slurm_memory_bytes(expected_memory)
        ):
            raise MaterializationPilotError(
                f"pilot submission scheduler identity conflict for job {job_id}"
            )
        row = {
            "job_id": job_id,
            "job_name": job_name,
            "state": state,
            "comment": normalized_comment,
            "partition": partition,
            "time_limit": time_limit,
            "command": command,
            "submit_line": "",
            "source": "squeue",
            "cpus": cpus,
            "memory": memory,
        }
        previous = rows_by_source["squeue"].get(job_id)
        if previous is not None and previous != row:
            raise MaterializationPilotError(
                f"ambiguous pilot submission squeue rows for job {job_id}"
            )
        rows_by_source["squeue"][job_id] = row

    for raw in query_records["sacct"]["stdout"].splitlines():
        if not raw.strip():
            continue
        fields = raw.split("|", 8)
        if len(fields) != 9:
            raise MaterializationPilotError(
                f"malformed pilot submission sacct row: {raw!r}"
            )
        (
            job_id,
            job_name,
            state,
            exit_code,
            reason,
            comment,
            partition,
            time_limit,
            submit_line,
        ) = (field.strip() for field in fields)
        if not job_id.isdigit():
            continue
        try:
            submit_tokens = shlex.split(submit_line)
        except ValueError as exc:
            raise MaterializationPilotError(
                f"malformed pilot submission SubmitLine for job {job_id}: {exc}"
            ) from exc
        normalized_comment = _normalize_scheduler_comment(comment)
        exact_submit = submit_tokens == expected_submit
        related = normalized_comment == expected_comment or exact_submit
        if not related:
            continue
        if (
            normalized_comment not in {"", expected_comment}
            or not exact_submit
            or job_name != expected_name
            or partition != expected_partition
            or time_limit != expected_limit
        ):
            raise MaterializationPilotError(
                f"pilot submission accounting identity conflict for job {job_id}"
            )
        row = {
            "job_id": job_id,
            "job_name": job_name,
            "state": state,
            "comment": normalized_comment,
            "partition": partition,
            "time_limit": time_limit,
            "command": expected_sbatch,
            "submit_line": submit_line,
            "source": "sacct",
            # The exact two-token SubmitLine binds the immutable, checksummed
            # sbatch file whose directives carry this resource contract.
            "cpus": expected_cpus,
            "memory": expected_memory,
            "exit_code": exit_code,
            "reason": reason,
        }
        previous = rows_by_source["sacct"].get(job_id)
        if previous is not None and previous != row:
            raise MaterializationPilotError(
                f"ambiguous pilot submission sacct rows for job {job_id}"
            )
        rows_by_source["sacct"][job_id] = row

    candidates: list[dict[str, Any]] = []
    for job_id in sorted(
        set(rows_by_source["squeue"]) | set(rows_by_source["sacct"]),
        key=int,
    ):
        live = rows_by_source["squeue"].get(job_id)
        history = rows_by_source["sacct"].get(job_id)
        if live is not None and history is not None:
            if (
                live["job_name"] != history["job_name"]
                or live["partition"] != history["partition"]
                or live["time_limit"] != history["time_limit"]
                or history["comment"] not in {"", live["comment"]}
            ):
                raise MaterializationPilotError(
                    f"pilot submission squeue/sacct identity conflict for job {job_id}"
                )
        selected = dict(live or history or {})
        selected["sources"] = sorted(
            source
            for source, rows in rows_by_source.items()
            if job_id in rows
        )
        selected["cpus"] = expected_cpus
        selected["memory"] = expected_memory
        candidates.append(selected)

    lineage = intent.get("retired_quarantines")
    if not isinstance(lineage, list):
        raise MaterializationPilotError(
            "pilot submission intent lacks retired-quarantine lineage"
        )
    retired_by_id = {
        str(record.get("job_id", "")): record
        for record in lineage
        if isinstance(record, dict)
    }
    if (
        len(retired_by_id) != len(lineage)
        or any(not job_id.isdigit() for job_id in retired_by_id)
    ):
        raise MaterializationPilotError(
            "pilot submission retired-quarantine lineage is ambiguous"
        )
    active_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        retired = retired_by_id.get(candidate["job_id"])
        if retired is None:
            active_candidates.append(candidate)
            continue
        terminal = retired.get("terminal")
        if (
            candidate.get("sources") != ["sacct"]
            or not isinstance(terminal, dict)
            or candidate.get("state") != terminal.get("raw_state")
            or candidate.get("exit_code") != terminal.get("exit_code")
            or candidate.get("reason") != terminal.get("reason")
        ):
            raise MaterializationPilotError(
                "retired pilot job reappeared with scheduler identity drift"
            )
    if len(active_candidates) > 1:
        raise MaterializationPilotError(
            "pilot submission intent maps to multiple scheduler jobs: "
            + ", ".join(row["job_id"] for row in active_candidates)
        )
    return active_candidates


def _query_submission_scheduler(
    *,
    intent: Mapping[str, Any],
    runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
    now: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    commands = _submission_query_argv(intent)
    snapshot = {
        "captured_at": float(now),
        "squeue": _scheduler_query_evidence(
            commands["squeue"], runner(commands["squeue"])
        ),
        "sacct": _scheduler_query_evidence(
            commands["sacct"], runner(commands["sacct"])
        ),
    }
    return snapshot, _submission_candidates_from_snapshot(
        snapshot, intent=intent
    )


def _load_submission_absent(
    path: Path,
    *,
    attempt_intent: Mapping[str, Any],
    global_intent: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path, description="pilot submission absence evidence")
    candidate = dict(payload)
    absent_id = candidate.pop("absent_id", None)
    snapshot = candidate.get("scheduler_snapshot")
    if (
        set(candidate)
        != {
            "schema_version",
            "protocol",
            "attempt",
            "attempt_id",
            "observed_at",
            "no_accepted_job",
            "scheduler_snapshot",
        }
        or candidate.get("schema_version") != 1
        or candidate.get("protocol") != SUBMISSION_ABSENT_PROTOCOL
        or candidate.get("attempt") != attempt_intent["attempt"]
        or candidate.get("attempt_id") != attempt_intent["attempt_id"]
        or candidate.get("no_accepted_job") is not True
        or not isinstance(candidate.get("observed_at"), str)
        or _submission_candidates_from_snapshot(
            snapshot, intent=global_intent
        )
        or not isinstance(absent_id, str)
        or _SHA256_RE.fullmatch(absent_id) is None
        or absent_id != _sha256_bytes(_canonical_bytes(candidate))
        or path.is_symlink()
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError(
            "pilot submission absence evidence drifted"
        )
    return payload


def _submission_attempts(
    root: Path,
    *,
    global_intent: Mapping[str, Any],
) -> list[dict[str, Any]]:
    parent = (
        root / SUBMISSION_DIRECTORY / SUBMISSION_ATTEMPTS_DIRECTORY
    )
    submission_root = root / SUBMISSION_DIRECTORY
    if submission_root.exists() or submission_root.is_symlink():
        if submission_root.is_symlink() or not submission_root.is_dir():
            raise MaterializationPilotError(
                "pilot submission transaction directory is unsafe"
            )
        unexpected_submission = [
            path.name
            for path in submission_root.iterdir()
            if path.name != SUBMISSION_ATTEMPTS_DIRECTORY
        ]
        if unexpected_submission:
            raise MaterializationPilotError(
                "pilot submission transaction contains unexpected artifacts: "
                + ", ".join(sorted(unexpected_submission))
            )
    if not parent.exists():
        return []
    if parent.is_symlink() or not parent.is_dir():
        raise MaterializationPilotError(
            "pilot submission attempts directory is unsafe"
        )
    entries = sorted(parent.iterdir(), key=lambda path: path.name)
    expected_names = [f"a{index:04d}" for index in range(1, len(entries) + 1)]
    if [entry.name for entry in entries] != expected_names:
        raise MaterializationPilotError(
            "pilot submission attempts are not a contiguous transaction"
        )
    attempts: list[dict[str, Any]] = []
    for index, directory in enumerate(entries, start=1):
        if directory.is_symlink() or not directory.is_dir():
            raise MaterializationPilotError(
                f"pilot submission attempt is unsafe: {directory}"
            )
        allowed = {
            SUBMISSION_ATTEMPT_INTENT_FILENAME,
            SUBMISSION_RESULT_FILENAME,
            SUBMISSION_ABSENT_FILENAME,
        }
        unexpected_attempt = [
            path.name for path in directory.iterdir() if path.name not in allowed
        ]
        if unexpected_attempt:
            raise MaterializationPilotError(
                "pilot submission attempt contains unexpected artifacts: "
                + ", ".join(sorted(unexpected_attempt))
            )
        paths = _submission_attempt_paths(root, index)
        intent = _load_submission_attempt_intent(
            paths["intent"],
            global_intent=global_intent,
            attempt=index,
        )
        result = (
            _load_submission_result(
                paths["result"], attempt_intent=intent
            )
            if paths["result"].exists() or paths["result"].is_symlink()
            else None
        )
        absent = (
            _load_submission_absent(
                paths["absent"],
                attempt_intent=intent,
                global_intent=global_intent,
            )
            if paths["absent"].exists() or paths["absent"].is_symlink()
            else None
        )
        if absent is not None and result is not None and result["returncode"] == 0:
            raise MaterializationPilotError(
                "successful sbatch output cannot be marked scheduler-absent"
            )
        if index < len(entries) and absent is None:
            raise MaterializationPilotError(
                "a later pilot submission attempt bypassed unresolved ambiguity"
            )
        attempts.append(
            {
                "attempt": index,
                "paths": paths,
                "intent": intent,
                "result": result,
                "absent": absent,
            }
        )
    return attempts


def _create_submission_attempt(
    *,
    root: Path,
    global_intent: Mapping[str, Any],
    attempt: int,
    now: float,
    visibility_timeout: float,
) -> dict[str, Any]:
    paths = _submission_attempt_paths(root, attempt)
    if paths["root"].exists() or paths["root"].is_symlink():
        raise MaterializationPilotError(
            f"pilot submission attempt already exists: {paths['root']}"
        )
    started = float(now)
    stable = {
        "schema_version": 1,
        "protocol": SUBMISSION_ATTEMPT_PROTOCOL,
        "pilot_root": str(root),
        "submission_intent": str(root / SUBMISSION_INTENT_FILENAME),
        "submission_intent_id": global_intent["intent_id"],
        "attempt": attempt,
        "submit_argv": global_intent["submit_argv"],
        "scheduler_comment": global_intent["scheduler_comment"],
        "started_timestamp": started,
        "visibility_deadline": started + float(visibility_timeout),
    }
    stable["attempt_id"] = _sha256_bytes(_canonical_bytes(stable))
    _atomic_write_once(paths["intent"], _json_bytes(stable))
    return _load_submission_attempt_intent(
        paths["intent"],
        global_intent=global_intent,
        attempt=attempt,
    )


def _reported_sbatch_job_id(stdout: str) -> str | None:
    match = re.fullmatch(
        r"Submitted[ \t]+batch[ \t]+job[ \t]+([0-9]+)[ \t]*\n?",
        stdout,
    )
    return None if match is None else match.group(1)


def _record_submission_result(
    path: Path,
    *,
    attempt_intent: Mapping[str, Any],
    process: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    if (
        not isinstance(process.returncode, int)
        or not isinstance(process.stdout, str)
        or not isinstance(process.stderr, str)
    ):
        raise MaterializationPilotError(
            "pilot sbatch did not return text-mode evidence"
        )
    stable = {
        "schema_version": 1,
        "protocol": SUBMISSION_RESULT_PROTOCOL,
        "attempt": attempt_intent["attempt"],
        "attempt_id": attempt_intent["attempt_id"],
        "argv": attempt_intent["submit_argv"],
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stdout_sha256": _sha256_bytes(process.stdout.encode("utf-8")),
        "stderr": process.stderr,
        "stderr_sha256": _sha256_bytes(process.stderr.encode("utf-8")),
        "reported_job_id": _reported_sbatch_job_id(process.stdout),
        "completed_at": _utc_now(),
    }
    stable["result_id"] = _sha256_bytes(_canonical_bytes(stable))
    _atomic_write_once(path, _json_bytes(stable))
    return _load_submission_result(path, attempt_intent=attempt_intent)


def _record_submission_absent(
    path: Path,
    *,
    attempt_intent: Mapping[str, Any],
    global_intent: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    if _submission_candidates_from_snapshot(snapshot, intent=global_intent):
        raise MaterializationPilotError(
            "cannot record pilot submission absence for a visible job"
        )
    stable = {
        "schema_version": 1,
        "protocol": SUBMISSION_ABSENT_PROTOCOL,
        "attempt": attempt_intent["attempt"],
        "attempt_id": attempt_intent["attempt_id"],
        "observed_at": _utc_now(),
        "no_accepted_job": True,
        "scheduler_snapshot": dict(snapshot),
    }
    stable["absent_id"] = _sha256_bytes(_canonical_bytes(stable))
    _atomic_write_once(path, _json_bytes(stable))
    return _load_submission_absent(
        path,
        attempt_intent=attempt_intent,
        global_intent=global_intent,
    )


def _submission_attempt_for_candidate(
    attempts: Sequence[Mapping[str, Any]],
    *,
    job_id: str,
) -> Mapping[str, Any]:
    matches = [
        attempt
        for attempt in attempts
        if attempt.get("result") is not None
        and attempt["result"].get("reported_job_id") == job_id
    ]
    if len(matches) > 1:
        raise MaterializationPilotError(
            f"pilot job {job_id} maps to multiple sbatch attempts"
        )
    if matches:
        if matches[0].get("absent") is not None:
            raise MaterializationPilotError(
                f"pilot job {job_id} contradicts sealed scheduler-absence evidence"
            )
        return matches[0]
    unresolved = [
        attempt
        for attempt in attempts
        if attempt.get("absent") is None
    ]
    if len(unresolved) != 1:
        raise MaterializationPilotError(
            "visible pilot job cannot be assigned to exactly one durable attempt"
        )
    reported_job_id = (
        None
        if unresolved[0].get("result") is None
        else unresolved[0]["result"].get("reported_job_id")
    )
    if reported_job_id is not None and reported_job_id != job_id:
        raise MaterializationPilotError(
            "sbatch stdout and scheduler truth disagree on pilot job ID"
        )
    return unresolved[0]


def _publish_submission_acceptance(
    *,
    root: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
    global_intent: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    job_id = str(candidate["job_id"])
    attempt = _submission_attempt_for_candidate(
        attempts, job_id=job_id
    )
    attempt_intent = attempt["intent"]
    result = attempt.get("result")
    stable = {
        "schema_version": 1,
        "protocol": SUBMISSION_ACCEPTED_PROTOCOL,
        "passed": True,
        "release_id": RELEASE_ID,
        "release_tag": REQUIRED_TAG,
        "pilot_root": str(root),
        "submission_intent": str(root / SUBMISSION_INTENT_FILENAME),
        "submission_intent_sha256": _sha256_file(
            root / SUBMISSION_INTENT_FILENAME
        ),
        "submission_intent_id": global_intent["intent_id"],
        "attempt": attempt_intent["attempt"],
        "attempt_intent": str(attempt["paths"]["intent"]),
        "attempt_intent_sha256": _sha256_file(attempt["paths"]["intent"]),
        "attempt_id": attempt_intent["attempt_id"],
        "sbatch_result": (
            None if result is None else str(attempt["paths"]["result"])
        ),
        "sbatch_result_sha256": (
            None
            if result is None
            else _sha256_file(attempt["paths"]["result"])
        ),
        "sbatch_result_id": (
            None if result is None else result["result_id"]
        ),
        "sbatch_receipt": str(receipt_path),
        "sbatch_receipt_sha256": _sha256_file(receipt_path),
        "sbatch_receipt_id": receipt["receipt_id"],
        "sbatch_path": receipt["sbatch_path"],
        "sbatch_sha256": receipt["sbatch_sha256"],
        "scheduler_user": global_intent["scheduler_user"],
        "scheduler_comment": global_intent["scheduler_comment"],
        "job_id": job_id,
        "job_name": candidate["job_name"],
        "partition": candidate["partition"],
        "time_limit": candidate["time_limit"],
        "cpus_per_task": global_intent["cpus_per_task"],
        "memory": global_intent["memory"],
        "scheduler_sources": candidate["sources"],
        "scheduler_snapshot": dict(snapshot),
        "accepted_at": _utc_now(),
    }
    stable["acceptance_id"] = _sha256_bytes(_canonical_bytes(stable))
    marker = root / SUBMISSION_ACCEPTED_FILENAME
    _atomic_write_once(marker, _json_bytes(stable))
    return _verify_pilot_submission_acceptance(
        root=root,
        receipt=receipt,
        receipt_path=receipt_path,
    )


def _verify_pilot_submission_acceptance(
    *,
    root: Path,
    receipt: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    intent_path = root / SUBMISSION_INTENT_FILENAME
    raw_intent = _read_json(
        intent_path, description="pilot submission intent"
    )
    raw_lineage = raw_intent.get("retired_quarantines")
    if not isinstance(raw_lineage, list) or any(
        not isinstance(record, dict)
        or not isinstance(record.get("quarantine_seal"), str)
        for record in raw_lineage
    ):
        raise MaterializationPilotError(
            "pilot submission retired-quarantine lineage is invalid"
        )
    intent = _load_submission_intent(
        intent_path,
        expected=_submission_intent_stable(
            root=root,
            receipt=receipt,
            receipt_path=receipt_path,
            scheduler_user=str(
                _read_json(
                    intent_path, description="pilot submission intent"
                ).get("scheduler_user", "")
            ),
            prior_quarantine_seals=[
                record["quarantine_seal"] for record in raw_lineage
            ],
        ),
    )
    attempts = _submission_attempts(root, global_intent=intent)
    marker_path = root / SUBMISSION_ACCEPTED_FILENAME
    marker = _read_json(
        marker_path, description="pilot submission acceptance"
    )
    candidate = dict(marker)
    acceptance_id = candidate.pop("acceptance_id", None)
    required = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "pilot_root",
        "submission_intent",
        "submission_intent_sha256",
        "submission_intent_id",
        "attempt",
        "attempt_intent",
        "attempt_intent_sha256",
        "attempt_id",
        "sbatch_result",
        "sbatch_result_sha256",
        "sbatch_result_id",
        "sbatch_receipt",
        "sbatch_receipt_sha256",
        "sbatch_receipt_id",
        "sbatch_path",
        "sbatch_sha256",
        "scheduler_user",
        "scheduler_comment",
        "job_id",
        "job_name",
        "partition",
        "time_limit",
        "cpus_per_task",
        "memory",
        "scheduler_sources",
        "scheduler_snapshot",
        "accepted_at",
    }
    if set(candidate) != required:
        raise MaterializationPilotError(
            "pilot submission acceptance fields drifted"
        )
    scheduler_candidates = _submission_candidates_from_snapshot(
        marker.get("scheduler_snapshot"), intent=intent
    )
    if len(scheduler_candidates) != 1:
        raise MaterializationPilotError(
            "pilot submission acceptance lacks exactly one scheduler job"
        )
    scheduler_job = scheduler_candidates[0]
    attempt_number = marker.get("attempt")
    if (
        not isinstance(attempt_number, int)
        or isinstance(attempt_number, bool)
        or not 1 <= attempt_number <= len(attempts)
    ):
        raise MaterializationPilotError(
            "pilot submission acceptance attempt is invalid"
        )
    attempt = attempts[attempt_number - 1]
    result = attempt.get("result")
    if (
        marker.get("schema_version") != 1
        or marker.get("protocol") != SUBMISSION_ACCEPTED_PROTOCOL
        or marker.get("passed") is not True
        or marker.get("release_id") != RELEASE_ID
        or marker.get("release_tag") != REQUIRED_TAG
        or marker.get("pilot_root") != str(root)
        or marker.get("submission_intent") != str(intent_path)
        or marker.get("submission_intent_sha256")
        != _sha256_file(intent_path)
        or marker.get("submission_intent_id") != intent["intent_id"]
        or marker.get("attempt_intent")
        != str(attempt["paths"]["intent"])
        or marker.get("attempt_intent_sha256")
        != _sha256_file(attempt["paths"]["intent"])
        or marker.get("attempt_id") != attempt["intent"]["attempt_id"]
        or marker.get("sbatch_result")
        != (None if result is None else str(attempt["paths"]["result"]))
        or marker.get("sbatch_result_sha256")
        != (
            None
            if result is None
            else _sha256_file(attempt["paths"]["result"])
        )
        or marker.get("sbatch_result_id")
        != (None if result is None else result["result_id"])
        or marker.get("sbatch_receipt") != str(receipt_path)
        or marker.get("sbatch_receipt_sha256")
        != _sha256_file(receipt_path)
        or marker.get("sbatch_receipt_id") != receipt["receipt_id"]
        or marker.get("sbatch_path") != receipt["sbatch_path"]
        or marker.get("sbatch_sha256") != receipt["sbatch_sha256"]
        or marker.get("scheduler_user") != intent["scheduler_user"]
        or marker.get("scheduler_comment") != intent["scheduler_comment"]
        or marker.get("job_id") != scheduler_job["job_id"]
        or marker.get("job_name") != scheduler_job["job_name"]
        or marker.get("partition") != scheduler_job["partition"]
        or marker.get("time_limit") != scheduler_job["time_limit"]
        or marker.get("cpus_per_task") != intent["cpus_per_task"]
        or marker.get("memory") != intent["memory"]
        or scheduler_job.get("cpus") != str(intent["cpus_per_task"])
        or _slurm_memory_bytes(str(scheduler_job.get("memory", "")))
        != _slurm_memory_bytes(str(intent["memory"]))
        or marker.get("scheduler_sources") != scheduler_job["sources"]
        or not isinstance(marker.get("accepted_at"), str)
        or not isinstance(acceptance_id, str)
        or _SHA256_RE.fullmatch(acceptance_id) is None
        or acceptance_id != _sha256_bytes(_canonical_bytes(candidate))
        or marker_path.is_symlink()
        or stat.S_IMODE(marker_path.stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError(
            "pilot submission acceptance identity drifted"
        )
    mapped_attempt = _submission_attempt_for_candidate(
        attempts, job_id=scheduler_job["job_id"]
    )
    if mapped_attempt["attempt"] != attempt_number:
        raise MaterializationPilotError(
            "pilot submission acceptance selected the wrong attempt"
        )
    return marker


def submit_materialization_pilot_sbatch(
    *,
    pilot_root: str | Path,
    sbatch_receipt: str | Path,
    scheduler_user: str,
    prior_quarantine_seals: Sequence[str | Path] = (),
    visibility_timeout: float = DEFAULT_SUBMISSION_VISIBILITY_TIMEOUT,
    poll_seconds: float = DEFAULT_SUBMISSION_POLL_SECONDS,
    apply: bool = False,
    runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] | None = None,
    now_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Transactionally submit or adopt the exact rendered pilot sbatch."""

    root = _destination_root(pilot_root)
    receipt, receipt_path = _validate_pilot_sbatch_receipt(
        sbatch_receipt, pilot_root=root
    )
    planned_identity = _submission_intent_stable(
        root=root,
        receipt=receipt,
        receipt_path=receipt_path,
        scheduler_user=scheduler_user,
        prior_quarantine_seals=prior_quarantine_seals,
    )
    if (
        not isinstance(visibility_timeout, (int, float))
        or isinstance(visibility_timeout, bool)
        or float(visibility_timeout) < 0
        or not isinstance(poll_seconds, (int, float))
        or isinstance(poll_seconds, bool)
        or float(poll_seconds) <= 0
    ):
        raise MaterializationPilotError(
            "pilot submission polling contract is invalid"
        )
    plan = {
        "schema_version": 1,
        "protocol": SUBMISSION_INTENT_PROTOCOL,
        "status": "dry_run",
        "pilot_root": str(root),
        "submission_intent": str(root / SUBMISSION_INTENT_FILENAME),
        "submission_acceptance": str(root / SUBMISSION_ACCEPTED_FILENAME),
        "sbatch_receipt": str(receipt_path),
        "sbatch_receipt_id": receipt["receipt_id"],
        "sbatch_path": receipt["sbatch_path"],
        "sbatch_sha256": receipt["sbatch_sha256"],
        "scheduler_user": planned_identity["scheduler_user"],
        "scheduler_comment": planned_identity["scheduler_comment"],
        "submit_argv": planned_identity["submit_argv"],
        "retired_quarantines": planned_identity["retired_quarantines"],
        "visibility_timeout": float(visibility_timeout),
        "poll_seconds": float(poll_seconds),
        "marker_first": True,
    }
    if not apply:
        return plan

    run = _scheduler_runner(runner)
    with _pilot_submission_lock(root):
        # Revalidate immutable input under the cross-node lock.
        receipt, receipt_path = _validate_pilot_sbatch_receipt(
            sbatch_receipt, pilot_root=root
        )
        intent, _created = _load_or_create_submission_intent(
            root=root,
            receipt=receipt,
            receipt_path=receipt_path,
            scheduler_user=scheduler_user,
            prior_quarantine_seals=prior_quarantine_seals,
            now=now_fn(),
        )
        marker = root / SUBMISSION_ACCEPTED_FILENAME
        if marker.exists() or marker.is_symlink():
            accepted = _verify_pilot_submission_acceptance(
                root=root,
                receipt=receipt,
                receipt_path=receipt_path,
            )
            return accepted | {
                "status": "already_submitted",
                "submission_acceptance": str(marker),
            }

        attempts = _submission_attempts(root, global_intent=intent)
        snapshot, candidates = _query_submission_scheduler(
            intent=intent, runner=run, now=now_fn()
        )
        if candidates:
            if not attempts:
                raise MaterializationPilotError(
                    "visible pilot job lacks a marker-first sbatch attempt"
                )
            accepted = _publish_submission_acceptance(
                root=root,
                receipt=receipt,
                receipt_path=receipt_path,
                global_intent=intent,
                attempts=attempts,
                candidate=candidates[0],
                snapshot=snapshot,
            )
            return accepted | {
                "status": "adopted",
                "submission_acceptance": str(marker),
            }

        if attempts and attempts[-1]["absent"] is None:
            last = attempts[-1]
            deadline = float(last["intent"]["visibility_deadline"])
            while not candidates and now_fn() < deadline:
                sleep_fn(
                    min(
                        float(poll_seconds),
                        max(0.0, deadline - now_fn()),
                    )
                )
                snapshot, candidates = _query_submission_scheduler(
                    intent=intent, runner=run, now=now_fn()
                )
            if candidates:
                accepted = _publish_submission_acceptance(
                    root=root,
                    receipt=receipt,
                    receipt_path=receipt_path,
                    global_intent=intent,
                    attempts=attempts,
                    candidate=candidates[0],
                    snapshot=snapshot,
                )
                return accepted | {
                    "status": "adopted",
                    "submission_acceptance": str(marker),
                }
            result = last.get("result")
            if result is not None and result["returncode"] == 0:
                raise MaterializationPilotError(
                    "sbatch reported an accepted pilot job that is absent from "
                    "complete squeue+sacct truth"
                )
            _record_submission_absent(
                last["paths"]["absent"],
                attempt_intent=last["intent"],
                global_intent=intent,
                snapshot=snapshot,
            )
            attempts = _submission_attempts(
                root, global_intent=intent
            )

        attempt_number = len(attempts) + 1
        attempt_intent = _create_submission_attempt(
            root=root,
            global_intent=intent,
            attempt=attempt_number,
            now=now_fn(),
            visibility_timeout=float(visibility_timeout),
        )
        attempt_paths = _submission_attempt_paths(root, attempt_number)
        # Close the scheduler-visibility race between the preceding reconciliation
        # and the irreversible submission boundary. The attempt marker already
        # exists, so an exact job that becomes visible here is safely adopted.
        attempts = _submission_attempts(root, global_intent=intent)
        snapshot, candidates = _query_submission_scheduler(
            intent=intent, runner=run, now=now_fn()
        )
        if candidates:
            accepted = _publish_submission_acceptance(
                root=root,
                receipt=receipt,
                receipt_path=receipt_path,
                global_intent=intent,
                attempts=attempts,
                candidate=candidates[0],
                snapshot=snapshot,
            )
            return accepted | {
                "status": "adopted",
                "submission_acceptance": str(marker),
            }
        # The immutable attempt intent is fsynced before this non-transactional
        # boundary. A process death here is recovered by exact comment/SubmitLine.
        process = run(list(receipt["submit_command"]))
        result = _record_submission_result(
            attempt_paths["result"],
            attempt_intent=attempt_intent,
            process=process,
        )
        attempts = _submission_attempts(root, global_intent=intent)
        deadline = float(attempt_intent["visibility_deadline"])
        snapshot, candidates = _query_submission_scheduler(
            intent=intent, runner=run, now=now_fn()
        )
        while not candidates and now_fn() < deadline:
            sleep_fn(
                min(
                    float(poll_seconds),
                    max(0.0, deadline - now_fn()),
                )
            )
            snapshot, candidates = _query_submission_scheduler(
                intent=intent, runner=run, now=now_fn()
            )
        if candidates:
            if (
                result["reported_job_id"] is not None
                and result["reported_job_id"] != candidates[0]["job_id"]
            ):
                raise MaterializationPilotError(
                    "sbatch stdout and scheduler truth disagree on pilot job ID"
                )
            accepted = _publish_submission_acceptance(
                root=root,
                receipt=receipt,
                receipt_path=receipt_path,
                global_intent=intent,
                attempts=attempts,
                candidate=candidates[0],
                snapshot=snapshot,
            )
            return accepted | {
                "status": "submitted",
                "submission_acceptance": str(marker),
            }
        if result["returncode"] == 0:
            raise MaterializationPilotError(
                "sbatch reported success but the pilot job is absent from "
                "complete squeue+sacct truth"
            )
        _record_submission_absent(
            attempt_paths["absent"],
            attempt_intent=attempt_intent,
            global_intent=intent,
            snapshot=snapshot,
        )
        raise MaterializationPilotError(
            f"pilot sbatch failed rc={result['returncode']}: "
            f"{result['stderr'].strip()[:500]}"
        )


def _scontrol_field(record: str, name: str) -> str:
    match = re.search(
        rf"(?:^|[ ]){re.escape(name)}=([^ ]*)",
        record.strip(),
    )
    if match is None or not match.group(1):
        raise MaterializationPilotError(
            f"active pilot scontrol record lacks {name}"
        )
    return match.group(1)


def _scheduler_attempt_paths(root: Path, job_id: str) -> dict[str, Path]:
    attempt = root / SCHEDULER_ATTEMPTS_DIRECTORY / job_id
    return {
        "root": attempt,
        "intent": attempt / SCHEDULER_INTENT_FILENAME,
        "spooled": attempt / SCHEDULER_SPOOLED_SCRIPT_FILENAME,
        "active": attempt / SCHEDULER_ACTIVE_FILENAME,
    }


def _scheduler_intent_base(
    *,
    root: Path,
    job_id: str,
    receipt: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": SCHEDULER_INTENT_PROTOCOL,
        "release_id": RELEASE_ID,
        "release_tag": REQUIRED_TAG,
        "pilot_root": str(root),
        "job_id": job_id,
        "job_name": SBATCH_JOB_NAME,
        "scheduler_comment": receipt["slurm"]["comment"],
        "sbatch_receipt": str(receipt_path),
        "sbatch_receipt_sha256": _sha256_file(receipt_path),
        "sbatch_receipt_id": receipt["receipt_id"],
        "sbatch_path": receipt["sbatch_path"],
        "sbatch_sha256": receipt["sbatch_sha256"],
        "expected_requeue": 0,
    }


def _load_scheduler_intent(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path, description="pilot scheduler intent")
    stable = dict(payload)
    intent_id = stable.pop("intent_id", None)
    created_at = stable.pop("created_at", None)
    if (
        stable != dict(expected)
        or not isinstance(created_at, str)
        or not isinstance(intent_id, str)
        or _SHA256_RE.fullmatch(intent_id) is None
        or intent_id
        != _sha256_bytes(_canonical_bytes(dict(expected) | {"created_at": created_at}))
        or path.is_symlink()
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError("pilot scheduler intent identity drifted")
    return payload


def _verify_active_scheduler_attempt(
    *,
    root: Path,
    job_id: str,
    receipt: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    paths = _scheduler_attempt_paths(root, job_id)
    for description, path in (
        ("pilot scheduler intent attempt directory", paths["root"]),
        ("pilot scheduler intent", paths["intent"]),
        ("pilot spooled sbatch", paths["spooled"]),
        ("pilot active scheduler evidence", paths["active"]),
    ):
        if path.is_symlink() or not path.exists():
            raise MaterializationPilotError(f"{description} is missing or unsafe")
    intent = _load_scheduler_intent(
        paths["intent"],
        expected=_scheduler_intent_base(
            root=root,
            job_id=job_id,
            receipt=receipt,
            receipt_path=receipt_path,
        ),
    )
    spooled = paths["spooled"]
    if (
        not spooled.is_file()
        or stat.S_IMODE(spooled.stat().st_mode) & 0o222
        or _sha256_file(spooled) != receipt["sbatch_sha256"]
        or spooled.read_bytes() != Path(receipt["sbatch_path"]).read_bytes()
    ):
        raise MaterializationPilotError(
            "active pilot spooled batch script differs from the receipt"
        )
    active = _read_json(paths["active"], description="active pilot scheduler evidence")
    candidate = dict(active)
    active_id = candidate.pop("active_id", None)
    recorded_at = candidate.pop("recorded_at", None)
    required = {
        "schema_version",
        "protocol",
        "release_id",
        "release_tag",
        "pilot_root",
        "job_id",
        "intent",
        "intent_sha256",
        "intent_id",
        "sbatch_receipt",
        "sbatch_receipt_sha256",
        "sbatch_receipt_id",
        "sbatch_path",
        "sbatch_sha256",
        "spooled_script",
        "spooled_script_sha256",
        "spooled_script_size",
        "spool_query",
        "scontrol",
        "squeue",
        "effective_requeue",
    }
    if (
        not isinstance(active.get("scontrol"), dict)
        or not isinstance(active.get("squeue"), dict)
    ):
        raise MaterializationPilotError(
            "active pilot scheduler evidence lacks raw query envelopes"
        )
    queue_argv = [
        "squeue",
        "-h",
        "-j",
        job_id,
        "-o",
        "%i|%j|%T|%k|%P|%l|%C|%m",
    ]
    queue_query = _verify_scheduler_query_evidence(
        active["squeue"].get("query"),
        expected_argv=queue_argv,
        description="active pilot squeue",
    )
    if queue_query["returncode"] != 0:
        raise MaterializationPilotError(
            "active pilot squeue evidence records a failed query"
        )
    queue_rows = [
        [field.strip() for field in row.split("|")]
        for row in queue_query["stdout"].splitlines()
        if row.strip()
    ]
    if len(queue_rows) != 1 or len(queue_rows[0]) != 8:
        raise MaterializationPilotError(
            "active pilot raw squeue evidence is absent or ambiguous"
        )
    (
        queue_id,
        queue_name,
        queue_state,
        queue_comment,
        queue_partition,
        queue_limit,
        queue_cpus,
        queue_memory,
    ) = queue_rows[0]
    expected_squeue = {
        "job_id": queue_id,
        "job_name": queue_name,
        "state": queue_state,
        "comment": queue_comment,
        "partition": queue_partition,
        "time_limit": queue_limit,
        "cpus": queue_cpus,
        "memory": queue_memory,
        "query": queue_query,
    }
    scontrol_argv = ["scontrol", "show", "job", "-o", job_id]
    scontrol_query = _verify_scheduler_query_evidence(
        active["scontrol"].get("query"),
        expected_argv=scontrol_argv,
        description="active pilot scontrol",
    )
    if scontrol_query["returncode"] != 0:
        raise MaterializationPilotError(
            "active pilot scontrol evidence records a failed query"
        )
    record = scontrol_query["stdout"].strip()
    try:
        requeue = int(_scontrol_field(record, "Requeue"))
    except ValueError as exc:
        raise MaterializationPilotError(
            "active pilot raw scontrol evidence has an invalid Requeue value"
        ) from exc
    expected_scontrol = {
        "job_id": _scontrol_field(record, "JobId"),
        "job_name": _scontrol_field(record, "JobName"),
        "state": _scontrol_field(record, "JobState"),
        "partition": _scontrol_field(record, "Partition"),
        "time_limit": _scontrol_field(record, "TimeLimit"),
        "requeue": requeue,
        "comment": _scontrol_field(record, "Comment"),
        "command": _scontrol_field(record, "Command"),
        "query": scontrol_query,
    }
    raw_spool_query = active.get("spool_query")
    spool_expected_argv = (
        raw_spool_query.get("argv")
        if isinstance(raw_spool_query, dict)
        and isinstance(raw_spool_query.get("argv"), list)
        else []
    )
    spool_query = _verify_scheduler_query_evidence(
        raw_spool_query,
        expected_argv=spool_expected_argv,
        description="active pilot batch-script spool",
    )
    # The actual temporary basename is deliberately randomized.  Bind all immutable
    # command components while accepting only that one same-directory basename.
    spool_argv = spool_query["argv"]
    if (
        len(spool_argv) != 5
        or spool_argv[:4]
        != ["scontrol", "write", "batch_script", job_id]
        or Path(spool_argv[4]).parent != paths["root"]
        or not Path(spool_argv[4]).name.startswith(".pilot-spooled.")
        or not Path(spool_argv[4]).name.endswith(".sbatch")
        or spool_query["returncode"] != 0
    ):
        raise MaterializationPilotError(
            "active pilot batch-script spool command drifted"
        )
    if (
        set(candidate) != required
        or active.get("schema_version") != 1
        or active.get("protocol") != SCHEDULER_ACTIVE_PROTOCOL
        or active.get("release_id") != RELEASE_ID
        or active.get("release_tag") != REQUIRED_TAG
        or active.get("pilot_root") != str(root)
        or active.get("job_id") != job_id
        or active.get("intent") != str(paths["intent"])
        or active.get("intent_sha256") != _sha256_file(paths["intent"])
        or active.get("intent_id") != intent["intent_id"]
        or active.get("sbatch_receipt") != str(receipt_path)
        or active.get("sbatch_receipt_sha256") != _sha256_file(receipt_path)
        or active.get("sbatch_receipt_id") != receipt["receipt_id"]
        or active.get("sbatch_path") != receipt["sbatch_path"]
        or active.get("sbatch_sha256") != receipt["sbatch_sha256"]
        or active.get("spooled_script") != str(spooled)
        or active.get("spooled_script_sha256") != _sha256_file(spooled)
        or active.get("spooled_script_size") != spooled.stat().st_size
        or active.get("effective_requeue") != 0
        or active["scontrol"] != expected_scontrol
        or active["squeue"] != expected_squeue
        or active["scontrol"].get("job_id") != job_id
        or active["scontrol"].get("job_name") != SBATCH_JOB_NAME
        or active["scontrol"].get("requeue") != 0
        or active["scontrol"].get("comment") != receipt["slurm"]["comment"]
        or active["scontrol"].get("command") != receipt["sbatch_path"]
        or active["scontrol"].get("partition") != receipt["slurm"]["partition"]
        or active["scontrol"].get("time_limit") != SBATCH_TIME_LIMIT
        or active["squeue"].get("job_id") != job_id
        or active["squeue"].get("job_name") != SBATCH_JOB_NAME
        or active["squeue"].get("comment") != receipt["slurm"]["comment"]
        or active["squeue"].get("partition") != receipt["slurm"]["partition"]
        or active["squeue"].get("time_limit") != SBATCH_TIME_LIMIT
        or active["squeue"].get("cpus")
        != str(receipt["slurm"]["cpus_per_task"])
        or _slurm_memory_bytes(
            str(active["squeue"].get("memory", ""))
        )
        != _slurm_memory_bytes(str(receipt["slurm"]["memory"]))
        or active["squeue"].get("state") not in {"CONFIGURING", "RUNNING"}
        or active["scontrol"].get("state") not in {"CONFIGURING", "RUNNING"}
        or not isinstance(recorded_at, str)
        or not isinstance(active_id, str)
        or _SHA256_RE.fullmatch(active_id) is None
        or active_id
        != _sha256_bytes(
            _canonical_bytes(candidate | {"recorded_at": recorded_at})
        )
        or paths["active"].is_symlink()
        or stat.S_IMODE(paths["active"].stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError(
            "active pilot scheduler evidence identity drifted"
        )
    return active


def _record_active_scheduler_attempt(
    *,
    root: Path,
    sbatch_receipt: str | Path,
    job_id: str,
    runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] | None = None,
) -> dict[str, Any]:
    """Record effective no-requeue and exact spooled bytes from inside the job."""

    receipt, receipt_path = _validate_pilot_sbatch_receipt(
        sbatch_receipt, pilot_root=root
    )
    if not job_id.isdigit():
        raise MaterializationPilotError(
            "materialization pilot requires a numeric SLURM_JOB_ID"
        )
    paths = _scheduler_attempt_paths(root, job_id)
    paths["root"].mkdir(parents=True, exist_ok=True, mode=0o750)
    expected_intent = _scheduler_intent_base(
        root=root,
        job_id=job_id,
        receipt=receipt,
        receipt_path=receipt_path,
    )
    if paths["intent"].exists() or paths["intent"].is_symlink():
        intent = _load_scheduler_intent(
            paths["intent"], expected=expected_intent
        )
    else:
        intent = expected_intent | {"created_at": _utc_now()}
        intent["intent_id"] = _sha256_bytes(_canonical_bytes(intent))
        _atomic_write_once(paths["intent"], _json_bytes(intent))
    if paths["active"].exists() or paths["active"].is_symlink():
        return _verify_active_scheduler_attempt(
            root=root,
            job_id=job_id,
            receipt=receipt,
            receipt_path=receipt_path,
        )

    run = _scheduler_runner(runner)
    queue_argv = [
        "squeue",
        "-h",
        "-j",
        job_id,
        "-o",
        "%i|%j|%T|%k|%P|%l|%C|%m",
    ]
    queue = run(queue_argv)
    queue_query = _scheduler_query_evidence(queue_argv, queue)
    if queue.returncode != 0:
        raise MaterializationPilotError(
            f"cannot query active pilot allocation: {queue.stderr[:500]}"
        )
    queue_rows = [
        [field.strip() for field in row.split("|")]
        for row in queue.stdout.splitlines()
        if row.strip()
    ]
    if len(queue_rows) != 1 or len(queue_rows[0]) != 8:
        raise MaterializationPilotError(
            "active pilot allocation is absent or ambiguous in squeue"
        )
    (
        queue_id,
        queue_name,
        queue_state,
        queue_comment,
        queue_partition,
        queue_limit,
        queue_cpus,
        queue_memory,
    ) = queue_rows[0]
    if (
        queue_id != job_id
        or queue_name != SBATCH_JOB_NAME
        or queue_state not in {"CONFIGURING", "RUNNING"}
        or queue_comment != receipt["slurm"]["comment"]
        or queue_partition != receipt["slurm"]["partition"]
        or queue_limit != SBATCH_TIME_LIMIT
        or queue_cpus != str(receipt["slurm"]["cpus_per_task"])
        or _slurm_memory_bytes(queue_memory)
        != _slurm_memory_bytes(str(receipt["slurm"]["memory"]))
    ):
        raise MaterializationPilotError(
            "active pilot squeue identity differs from the rendered receipt"
        )

    scontrol_argv = ["scontrol", "show", "job", "-o", job_id]
    effective = run(scontrol_argv)
    scontrol_query = _scheduler_query_evidence(scontrol_argv, effective)
    if effective.returncode != 0:
        raise MaterializationPilotError(
            f"cannot query effective pilot job: {effective.stderr[:500]}"
        )
    record = effective.stdout.strip()
    try:
        requeue = int(_scontrol_field(record, "Requeue"))
    except ValueError as exc:
        raise MaterializationPilotError(
            "active pilot has an invalid effective Requeue value"
        ) from exc
    scontrol_record = {
        "job_id": _scontrol_field(record, "JobId"),
        "job_name": _scontrol_field(record, "JobName"),
        "state": _scontrol_field(record, "JobState"),
        "partition": _scontrol_field(record, "Partition"),
        "time_limit": _scontrol_field(record, "TimeLimit"),
        "requeue": requeue,
        "comment": _scontrol_field(record, "Comment"),
        "command": _scontrol_field(record, "Command"),
        "query": scontrol_query,
    }
    if (
        scontrol_record["job_id"] != job_id
        or scontrol_record["job_name"] != SBATCH_JOB_NAME
        or scontrol_record["state"] not in {"CONFIGURING", "RUNNING"}
        or scontrol_record["partition"] != receipt["slurm"]["partition"]
        or scontrol_record["time_limit"] != SBATCH_TIME_LIMIT
        or scontrol_record["requeue"] != 0
        or scontrol_record["comment"] != receipt["slurm"]["comment"]
        or scontrol_record["command"] != receipt["sbatch_path"]
    ):
        raise MaterializationPilotError(
            "active pilot effective Slurm identity differs from the receipt"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".pilot-spooled.", suffix=".sbatch", dir=paths["root"]
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        spool_argv = [
            "scontrol",
            "write",
            "batch_script",
            job_id,
            str(temporary),
        ]
        written = run(spool_argv)
        spool_query = _scheduler_query_evidence(spool_argv, written)
        if written.returncode != 0:
            raise MaterializationPilotError(
                f"cannot spool active pilot batch script: {written.stderr[:500]}"
            )
        if temporary.is_symlink() or not temporary.is_file():
            raise MaterializationPilotError(
                "scontrol did not publish a regular pilot batch script"
            )
        payload = temporary.read_bytes()
        if (
            _sha256_bytes(payload) != receipt["sbatch_sha256"]
            or payload != Path(receipt["sbatch_path"]).read_bytes()
        ):
            raise MaterializationPilotError(
                "active pilot spooled batch bytes differ from the receipt"
            )
        _atomic_write_once(paths["spooled"], payload)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    active_stable = {
        "schema_version": 1,
        "protocol": SCHEDULER_ACTIVE_PROTOCOL,
        "release_id": RELEASE_ID,
        "release_tag": REQUIRED_TAG,
        "pilot_root": str(root),
        "job_id": job_id,
        "intent": str(paths["intent"]),
        "intent_sha256": _sha256_file(paths["intent"]),
        "intent_id": intent["intent_id"],
        "sbatch_receipt": str(receipt_path),
        "sbatch_receipt_sha256": _sha256_file(receipt_path),
        "sbatch_receipt_id": receipt["receipt_id"],
        "sbatch_path": receipt["sbatch_path"],
        "sbatch_sha256": receipt["sbatch_sha256"],
        "spooled_script": str(paths["spooled"]),
        "spooled_script_sha256": _sha256_file(paths["spooled"]),
        "spooled_script_size": paths["spooled"].stat().st_size,
        "spool_query": spool_query,
        "scontrol": scontrol_record,
        "squeue": {
            "job_id": queue_id,
            "job_name": queue_name,
            "state": queue_state,
            "comment": queue_comment,
            "partition": queue_partition,
            "time_limit": queue_limit,
            "cpus": queue_cpus,
            "memory": queue_memory,
            "query": queue_query,
        },
        "effective_requeue": 0,
        "recorded_at": _utc_now(),
    }
    active_stable["active_id"] = _sha256_bytes(
        _canonical_bytes(active_stable)
    )
    _atomic_write_once(paths["active"], _json_bytes(active_stable))
    return _verify_active_scheduler_attempt(
        root=root,
        job_id=job_id,
        receipt=receipt,
        receipt_path=receipt_path,
    )


def _terminal_transient_pilot_job(
    *,
    job_id: str,
    receipt: Mapping[str, Any],
    runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ],
) -> dict[str, Any]:
    if not job_id.isdigit():
        raise MaterializationPilotError("pilot Slurm job ID must be numeric")
    active = runner(
        ["squeue", "-h", "-j", job_id, "-o", "%i|%j|%T|%k"]
    )
    if active.returncode != 0:
        raise MaterializationPilotError(
            f"cannot query active pilot job {job_id}: {active.stderr[:500]}"
        )
    if active.stdout.strip():
        raise MaterializationPilotError(
            f"cannot quarantine while pilot job {job_id} remains active"
        )
    accounting = runner(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=JobIDRaw,JobName%64,State,ExitCode,Reason,Comment%256,SubmitLine",
        ]
    )
    if accounting.returncode != 0:
        raise MaterializationPilotError(
            f"cannot query terminal pilot job {job_id}: {accounting.stderr[:500]}"
        )
    rows: list[list[str]] = []
    for raw in accounting.stdout.splitlines():
        fields = [value.strip() for value in raw.split("|", 6)]
        if len(fields) < 7 or fields[0] != job_id:
            continue
        rows.append(fields)
    if len(rows) != 1:
        raise MaterializationPilotError(
            f"pilot job {job_id} has {len(rows)} unambiguous top-level sacct rows"
        )
    (
        observed_job_id,
        job_name,
        raw_state,
        exit_code,
        reason,
        accounting_comment,
        submit_line,
    ) = rows[0]
    del observed_job_id
    normalized_state = raw_state.split()[0].rstrip("+") if raw_state else ""
    if normalized_state in _INHERENT_TRANSIENT_STATES:
        transient_classification = "scheduler_or_node_transient"
    elif (
        normalized_state == "CANCELLED"
        and _EXTERNAL_CANCELLATION_RE.fullmatch(raw_state) is not None
    ):
        transient_classification = "external_cancellation"
    else:
        raise MaterializationPilotError(
            "same-tag pilot retry is restricted to scheduler/node transients or "
            "an explicit external cancellation; deterministic, timeout, OOM, and "
            f"unattributed terminal state {raw_state!r} require a superseding release"
        )
    if _SLURM_EXIT_CODE_RE.fullmatch(exit_code) is None or not reason:
        raise MaterializationPilotError(
            "terminal pilot accounting lacks an exact exit code or reason"
        )
    expected_comment = str(receipt["slurm"]["comment"])
    if accounting_comment.lower() in {"", "(null)", "null", "none"}:
        accounting_comment = ""
    elif accounting_comment != expected_comment:
        raise MaterializationPilotError(
            "terminal pilot accounting comment differs from the rendered receipt"
        )
    try:
        submit_tokens = shlex.split(submit_line)
    except ValueError as exc:
        raise MaterializationPilotError(
            f"terminal pilot SubmitLine is malformed: {exc}"
        ) from exc
    expected_sbatch = str(receipt["sbatch_path"])
    if (
        job_name != SBATCH_JOB_NAME
        or len(submit_tokens) != 2
        or Path(submit_tokens[0]).name != "sbatch"
        or submit_tokens[1] != expected_sbatch
    ):
        raise MaterializationPilotError(
            "terminal pilot job is not bound to the exact rendered sbatch receipt"
        )
    return {
        "job_id": job_id,
        "job_name": job_name,
        "raw_state": raw_state,
        "normalized_state": normalized_state,
        "exit_code": exit_code,
        "reason": reason,
        "accounting_comment": accounting_comment,
        "expected_comment": expected_comment,
        "submit_line": submit_line,
        "transient_classification": transient_classification,
    }


def _completed_terminal_from_query_evidence(
    *,
    job_id: str,
    receipt: Mapping[str, Any],
    squeue_query: Any,
    sacct_query: Any,
) -> dict[str, Any]:
    if not job_id.isdigit():
        raise MaterializationPilotError("pilot Slurm job ID must be numeric")
    squeue_argv = [
        "squeue",
        "-h",
        "-j",
        job_id,
        "-o",
        "%i|%j|%T|%k",
    ]
    active = _verify_scheduler_query_evidence(
        squeue_query,
        expected_argv=squeue_argv,
        description="terminal pilot squeue",
    )
    if active["returncode"] != 0:
        raise MaterializationPilotError(
            f"cannot query active pilot job {job_id}: {active['stderr'][:500]}"
        )
    if active["stdout"].strip():
        raise MaterializationPilotError(
            f"pilot job {job_id} is not terminal"
        )
    sacct_argv = [
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        job_id,
        "--format=JobIDRaw,JobName%64,State,ExitCode,Reason,Comment%256,SubmitLine",
    ]
    accounting = _verify_scheduler_query_evidence(
        sacct_query,
        expected_argv=sacct_argv,
        description="terminal pilot sacct",
    )
    if accounting["returncode"] != 0:
        raise MaterializationPilotError(
            f"cannot query completed pilot job {job_id}: "
            f"{accounting['stderr'][:500]}"
        )
    rows: list[list[str]] = []
    for raw in accounting["stdout"].splitlines():
        fields = [value.strip() for value in raw.split("|", 6)]
        if len(fields) >= 7 and fields[0] == job_id:
            rows.append(fields)
    if len(rows) != 1:
        raise MaterializationPilotError(
            f"pilot job {job_id} has {len(rows)} unambiguous top-level sacct rows"
        )
    (
        _observed_job_id,
        job_name,
        raw_state,
        exit_code,
        reason,
        accounting_comment,
        submit_line,
    ) = rows[0]
    normalized_state = raw_state.split()[0].rstrip("+") if raw_state else ""
    if normalized_state != "COMPLETED" or exit_code != "0:0" or not reason:
        raise MaterializationPilotError(
            "pilot scheduler acceptance requires one terminal COMPLETED|0:0 "
            f"allocation, got {raw_state}|{exit_code}|{reason}"
        )
    expected_comment = str(receipt["slurm"]["comment"])
    if accounting_comment.lower() in {"", "(null)", "null", "none"}:
        accounting_comment = ""
    elif accounting_comment != expected_comment:
        raise MaterializationPilotError(
            "completed pilot accounting comment differs from the receipt"
        )
    try:
        submit_tokens = shlex.split(submit_line)
    except ValueError as exc:
        raise MaterializationPilotError(
            f"completed pilot SubmitLine is malformed: {exc}"
        ) from exc
    if (
        job_name != SBATCH_JOB_NAME
        or len(submit_tokens) != 2
        or Path(submit_tokens[0]).name != "sbatch"
        or submit_tokens[1] != receipt["sbatch_path"]
    ):
        raise MaterializationPilotError(
            "completed pilot job is not bound to the exact rendered sbatch"
        )
    return {
        "job_id": job_id,
        "job_name": job_name,
        "raw_state": raw_state,
        "normalized_state": normalized_state,
        "exit_code": exit_code,
        "reason": reason,
        "accounting_comment": accounting_comment,
        "expected_comment": expected_comment,
        "submit_line": submit_line,
        "squeue_query": active,
        "sacct_query": accounting,
    }


def _terminal_completed_pilot_job(
    *,
    job_id: str,
    receipt: Mapping[str, Any],
    runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ],
) -> dict[str, Any]:
    """Query once, retain exact scheduler output, and derive the terminal record."""

    squeue_argv = [
        "squeue",
        "-h",
        "-j",
        job_id,
        "-o",
        "%i|%j|%T|%k",
    ]
    sacct_argv = [
        "sacct",
        "-X",
        "-n",
        "-P",
        "-j",
        job_id,
        "--format=JobIDRaw,JobName%64,State,ExitCode,Reason,Comment%256,SubmitLine",
    ]
    squeue_process = runner(squeue_argv)
    sacct_process = runner(sacct_argv)
    return _completed_terminal_from_query_evidence(
        job_id=job_id,
        receipt=receipt,
        squeue_query=_scheduler_query_evidence(
            squeue_argv, squeue_process
        ),
        sacct_query=_scheduler_query_evidence(
            sacct_argv, sacct_process
        ),
    )


def _verify_pilot_scheduler_acceptance(
    *,
    root: Path,
    semantic: Mapping[str, Any],
) -> dict[str, Any]:
    marker_path = root / SCHEDULER_ACCEPTANCE_FILENAME
    marker = _read_json(
        marker_path, description="materialization pilot scheduler acceptance"
    )
    candidate = dict(marker)
    acceptance_id = candidate.pop("acceptance_id", None)
    accepted_at = candidate.pop("accepted_at", None)
    required = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "pilot_root",
        "pilot_id",
        "pilot_marker",
        "pilot_marker_sha256",
        "job_id",
        "sbatch_receipt",
        "sbatch_receipt_sha256",
        "sbatch_receipt_id",
        "sbatch_path",
        "sbatch_sha256",
        "submission_acceptance",
        "submission_acceptance_sha256",
        "submission_acceptance_id",
        "active_evidence",
        "active_evidence_sha256",
        "active_id",
        "spooled_script",
        "spooled_script_sha256",
        "effective_requeue",
        "terminal",
    }
    if set(candidate) != required:
        raise MaterializationPilotError(
            "pilot scheduler acceptance fields drifted"
        )
    job_id = str(marker.get("job_id", ""))
    receipt_path = Path(str(marker.get("sbatch_receipt", "")))
    receipt, canonical_receipt = _validate_pilot_sbatch_receipt(
        receipt_path, pilot_root=root
    )
    submission = _verify_pilot_submission_acceptance(
        root=root,
        receipt=receipt,
        receipt_path=canonical_receipt,
    )
    active = _verify_active_scheduler_attempt(
        root=root,
        job_id=job_id,
        receipt=receipt,
        receipt_path=canonical_receipt,
    )
    paths = _scheduler_attempt_paths(root, job_id)
    terminal = marker.get("terminal")
    if (
        marker.get("schema_version") != 1
        or marker.get("protocol") != SCHEDULER_ACCEPTANCE_PROTOCOL
        or marker.get("passed") is not True
        or marker.get("release_id") != RELEASE_ID
        or marker.get("release_tag") != REQUIRED_TAG
        or marker.get("pilot_root") != str(root)
        or marker.get("pilot_id") != semantic["pilot_id"]
        or marker.get("pilot_marker") != str(root / COMPLETE_MARKER)
        or marker.get("pilot_marker_sha256")
        != _sha256_file(root / COMPLETE_MARKER)
        or not job_id.isdigit()
        or marker.get("sbatch_receipt") != str(canonical_receipt)
        or marker.get("sbatch_receipt_sha256")
        != _sha256_file(canonical_receipt)
        or marker.get("sbatch_receipt_id") != receipt["receipt_id"]
        or marker.get("sbatch_path") != receipt["sbatch_path"]
        or marker.get("sbatch_sha256") != receipt["sbatch_sha256"]
        or marker.get("submission_acceptance")
        != str(root / SUBMISSION_ACCEPTED_FILENAME)
        or marker.get("submission_acceptance_sha256")
        != _sha256_file(root / SUBMISSION_ACCEPTED_FILENAME)
        or marker.get("submission_acceptance_id")
        != submission["acceptance_id"]
        or submission.get("job_id") != job_id
        or marker.get("active_evidence") != str(paths["active"])
        or marker.get("active_evidence_sha256")
        != _sha256_file(paths["active"])
        or marker.get("active_id") != active["active_id"]
        or marker.get("spooled_script") != str(paths["spooled"])
        or marker.get("spooled_script_sha256")
        != _sha256_file(paths["spooled"])
        or marker.get("effective_requeue") != 0
        or not isinstance(terminal, dict)
        or set(terminal)
        != {
            "job_id",
            "job_name",
            "raw_state",
            "normalized_state",
            "exit_code",
            "reason",
            "accounting_comment",
            "expected_comment",
            "submit_line",
            "squeue_query",
            "sacct_query",
        }
        or terminal
        != _completed_terminal_from_query_evidence(
            job_id=job_id,
            receipt=receipt,
            squeue_query=terminal.get("squeue_query"),
            sacct_query=terminal.get("sacct_query"),
        )
        or terminal.get("job_id") != job_id
        or terminal.get("job_name") != SBATCH_JOB_NAME
        or terminal.get("normalized_state") != "COMPLETED"
        or terminal.get("exit_code") != "0:0"
        or terminal.get("expected_comment") != receipt["slurm"]["comment"]
        or terminal.get("submit_line")
        != shlex.join(receipt["submit_command"])
        or not isinstance(accepted_at, str)
        or not isinstance(acceptance_id, str)
        or _SHA256_RE.fullmatch(acceptance_id) is None
        or acceptance_id
        != _sha256_bytes(
            _canonical_bytes(candidate | {"accepted_at": accepted_at})
        )
        or marker_path.is_symlink()
        or stat.S_IMODE(marker_path.stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError(
            "pilot scheduler acceptance identity drifted"
        )
    return marker


def accept_pilot_scheduler(
    *,
    pilot_root: str | Path,
    sbatch_receipt: str | Path,
    job_id: str,
    apply: bool = False,
    runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] | None = None,
) -> dict[str, Any]:
    """Join active no-requeue evidence to terminal COMPLETED accounting."""

    root = _existing_directory(
        pilot_root, description="materialization pilot root"
    )
    semantic = _verify_materialization_pilot_semantic(root)
    receipt, receipt_path = _validate_pilot_sbatch_receipt(
        sbatch_receipt, pilot_root=root
    )
    submission = _verify_pilot_submission_acceptance(
        root=root,
        receipt=receipt,
        receipt_path=receipt_path,
    )
    if submission.get("job_id") != job_id:
        raise MaterializationPilotError(
            "pilot scheduler acceptance job differs from the transactionally "
            "accepted submission"
        )
    marker_path = root / SCHEDULER_ACCEPTANCE_FILENAME
    if marker_path.exists() or marker_path.is_symlink():
        verified = _verify_pilot_scheduler_acceptance(
            root=root, semantic=semantic
        )
        if (
            verified.get("job_id") != job_id
            or verified.get("sbatch_receipt") != str(receipt_path)
            or verified.get("sbatch_receipt_sha256")
            != _sha256_file(receipt_path)
            or verified.get("sbatch_receipt_id") != receipt["receipt_id"]
        ):
            raise MaterializationPilotError(
                "existing pilot scheduler acceptance binds a different job "
                "or sbatch receipt"
            )
        return verified | {
            "status": "already_accepted",
            "acceptance_marker": str(marker_path),
        }
    active = _verify_active_scheduler_attempt(
        root=root,
        job_id=job_id,
        receipt=receipt,
        receipt_path=receipt_path,
    )
    terminal = _terminal_completed_pilot_job(
        job_id=job_id,
        receipt=receipt,
        runner=_scheduler_runner(runner),
    )
    paths = _scheduler_attempt_paths(root, job_id)
    stable = {
        "schema_version": 1,
        "protocol": SCHEDULER_ACCEPTANCE_PROTOCOL,
        "passed": True,
        "release_id": RELEASE_ID,
        "release_tag": REQUIRED_TAG,
        "pilot_root": str(root),
        "pilot_id": semantic["pilot_id"],
        "pilot_marker": str(root / COMPLETE_MARKER),
        "pilot_marker_sha256": _sha256_file(root / COMPLETE_MARKER),
        "job_id": job_id,
        "sbatch_receipt": str(receipt_path),
        "sbatch_receipt_sha256": _sha256_file(receipt_path),
        "sbatch_receipt_id": receipt["receipt_id"],
        "sbatch_path": receipt["sbatch_path"],
        "sbatch_sha256": receipt["sbatch_sha256"],
        "submission_acceptance": str(
            root / SUBMISSION_ACCEPTED_FILENAME
        ),
        "submission_acceptance_sha256": _sha256_file(
            root / SUBMISSION_ACCEPTED_FILENAME
        ),
        "submission_acceptance_id": submission["acceptance_id"],
        "active_evidence": str(paths["active"]),
        "active_evidence_sha256": _sha256_file(paths["active"]),
        "active_id": active["active_id"],
        "spooled_script": str(paths["spooled"]),
        "spooled_script_sha256": _sha256_file(paths["spooled"]),
        "effective_requeue": 0,
        "terminal": terminal,
        "accepted_at": _utc_now(),
    }
    stable["acceptance_id"] = _sha256_bytes(_canonical_bytes(stable))
    if not apply:
        return stable | {
            "status": "dry_run",
            "acceptance_marker": str(marker_path),
        }
    _atomic_write_once(marker_path, _json_bytes(stable))
    verified = _verify_pilot_scheduler_acceptance(
        root=root, semantic=semantic
    )
    return verified | {
        "status": "accepted",
        "acceptance_marker": str(marker_path),
    }


def _validated_quarantine_intent(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path, description="pilot quarantine intent")
    identity = dict(payload)
    intent_id = identity.pop("intent_id", None)
    created_at = identity.pop("created_at", None)
    if (
        identity != dict(expected)
        or not isinstance(created_at, str)
        or not isinstance(intent_id, str)
        or _SHA256_RE.fullmatch(intent_id) is None
        or intent_id
        != _sha256_bytes(_canonical_bytes(dict(expected) | {"created_at": created_at}))
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise MaterializationPilotError("pilot quarantine intent identity drifted")
    return payload


def quarantine_interrupted_pilot(
    *,
    pilot_root: str | Path,
    sbatch_receipt: str | Path,
    job_id: str,
    apply: bool = False,
    runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] | None = None,
) -> dict[str, Any]:
    """Preserve one transiently interrupted canonical pilot before exact retry."""

    supplied_root = Path(pilot_root).expanduser()
    if supplied_root.is_symlink():
        raise MaterializationPilotError(
            f"materialization pilot root is symlinked: {supplied_root}"
        )
    root = supplied_root.resolve()
    if root in {Path(root.anchor), Path.home().resolve()}:
        raise MaterializationPilotError(
            f"refusing unsafe broad materialization pilot root: {root}"
        )
    receipt, receipt_path = _validate_pilot_sbatch_receipt(
        sbatch_receipt, pilot_root=root
    )
    scheduler_runner = (
        (
            lambda argv: subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                env=_sanitized_process_environment(),
            )
        )
        if runner is None
        else runner
    )
    scheduler = _terminal_transient_pilot_job(
        job_id=job_id,
        receipt=receipt,
        runner=scheduler_runner,
    )

    quarantine_parent = root.parent / "quarantine"
    evidence_root = root.parent / "quarantine_evidence"
    destination = quarantine_parent / f"{root.name}.partial-job-{job_id}"
    intent_path = evidence_root / f"partial-job-{job_id}.intent.json"
    completion_path = evidence_root / f"partial-job-{job_id}.complete.json"
    source_exists = root.exists() or root.is_symlink()
    destination_exists = destination.exists() or destination.is_symlink()
    if source_exists and destination_exists:
        raise MaterializationPilotError(
            "interrupted pilot exists at both source and quarantine paths"
        )
    if source_exists:
        if root.is_symlink() or not root.is_dir():
            raise MaterializationPilotError("interrupted pilot source is unsafe")
        partial = root
    elif destination_exists:
        if destination.is_symlink() or not destination.is_dir():
            raise MaterializationPilotError("interrupted pilot quarantine is unsafe")
        partial = destination
    else:
        raise MaterializationPilotError(
            f"no interrupted pilot exists at {root} or {destination}"
        )
    if (partial / COMPLETE_MARKER).exists() or (partial / COMPLETE_MARKER).is_symlink():
        raise MaterializationPilotError(
            "refusing to quarantine a completed materialization pilot"
        )
    partial_stat = os.lstat(partial)
    expected_intent = {
        "schema_version": 1,
        "protocol": PILOT_QUARANTINE_INTENT_PROTOCOL,
        "release_id": RELEASE_ID,
        "release_tag": REQUIRED_TAG,
        "pilot_root": str(root),
        "destination": str(destination),
        "sbatch_receipt": str(receipt_path),
        "sbatch_receipt_sha256": _sha256_file(receipt_path),
        "sbatch_receipt_id": receipt["receipt_id"],
        "scheduler": scheduler,
        "source_device": partial_stat.st_dev,
        "source_inode": partial_stat.st_ino,
    }
    if intent_path.exists() or intent_path.is_symlink():
        if intent_path.is_symlink() or not intent_path.is_file():
            raise MaterializationPilotError("pilot quarantine intent is unsafe")
        intent = _validated_quarantine_intent(
            intent_path, expected=expected_intent
        )
    else:
        if destination_exists:
            raise MaterializationPilotError(
                "quarantined pilot exists without its marker-first intent"
            )
        intent = expected_intent | {"created_at": _utc_now()}
        intent["intent_id"] = _sha256_bytes(_canonical_bytes(intent))

    if completion_path.exists() or completion_path.is_symlink():
        if completion_path.is_symlink() or not completion_path.is_file():
            raise MaterializationPilotError("pilot quarantine completion is unsafe")
        completion = _read_json(
            completion_path, description="pilot quarantine completion"
        )
        stable = dict(completion)
        completion_id = stable.pop("completion_id", None)
        completed_at = stable.pop("completed_at", None)
        expected_completion = {
            "schema_version": 1,
            "protocol": PILOT_QUARANTINE_PROTOCOL,
            "passed": True,
            "release_id": RELEASE_ID,
            "release_tag": REQUIRED_TAG,
            "materialize_job_id": job_id,
            "pilot_root": str(root),
            "source": str(root),
            "destination": str(destination),
            "sbatch_receipt": str(receipt_path),
            "sbatch_receipt_sha256": _sha256_file(receipt_path),
            "intent": str(intent_path),
            "intent_sha256": _sha256_file(intent_path),
            "intent_id": intent["intent_id"],
            "scheduler": scheduler,
            "source_device": partial_stat.st_dev,
            "source_inode": partial_stat.st_ino,
        }
        if (
            stable != expected_completion
            or not isinstance(completed_at, str)
            or not isinstance(completion_id, str)
            or _SHA256_RE.fullmatch(completion_id) is None
            or completion_id
            != _sha256_bytes(
                _canonical_bytes(
                    expected_completion | {"completed_at": completed_at}
                )
            )
            or source_exists
            or not destination_exists
            or stat.S_IMODE(completion_path.stat().st_mode) & 0o222
        ):
            raise MaterializationPilotError(
                "pilot quarantine completion identity drifted"
            )
        seal = recovery_evidence.seal_quarantine(
            tree=destination,
            evidence_root=evidence_root,
            release_id=RELEASE_ID,
            failed_job_id=job_id,
            quarantine_completion=completion_path,
            apply=apply,
        )
        return completion | {
            "status": (
                "already_quarantined_and_sealed"
                if apply
                else "already_quarantined_would_verify"
            ),
            "quarantine_seal": seal,
        }

    report = {
        "status": "dry_run" if not apply else "quarantining",
        "pilot_root": str(root),
        "destination": str(destination),
        "job_id": job_id,
        "scheduler": scheduler,
        "would_resume_incomplete_rename": destination_exists,
    }
    if not apply:
        return report

    evidence_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    quarantine_parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if (
        evidence_root.is_symlink()
        or quarantine_parent.is_symlink()
        or evidence_root.resolve() != evidence_root
        or quarantine_parent.resolve() != quarantine_parent
    ):
        raise MaterializationPilotError("pilot quarantine ancestry is unsafe")
    if os.lstat(quarantine_parent).st_dev != partial_stat.st_dev:
        raise MaterializationPilotError(
            "pilot quarantine must use a same-filesystem rename"
        )
    if not intent_path.exists():
        _atomic_write_once(intent_path, _json_bytes(intent))

    if source_exists:
        # Repeat the scheduler check at the mutation boundary.
        current_scheduler = _terminal_transient_pilot_job(
            job_id=job_id,
            receipt=receipt,
            runner=scheduler_runner,
        )
        if current_scheduler != scheduler:
            raise MaterializationPilotError(
                "pilot scheduler evidence changed before quarantine"
            )
        current_stat = os.lstat(root)
        if (
            current_stat.st_dev != partial_stat.st_dev
            or current_stat.st_ino != partial_stat.st_ino
            or (root / COMPLETE_MARKER).exists()
            or (root / COMPLETE_MARKER).is_symlink()
        ):
            raise MaterializationPilotError(
                "interrupted pilot changed before quarantine"
            )
        os.rename(root, destination)
        _fsync_directory(root.parent)
        _fsync_directory(quarantine_parent)

    destination_stat = os.lstat(destination)
    if (
        destination_stat.st_dev != partial_stat.st_dev
        or destination_stat.st_ino != partial_stat.st_ino
        or root.exists()
        or root.is_symlink()
    ):
        raise MaterializationPilotError(
            "pilot quarantine rename did not preserve source identity"
        )
    completion = {
        "schema_version": 1,
        "protocol": PILOT_QUARANTINE_PROTOCOL,
        "passed": True,
        "release_id": RELEASE_ID,
        "release_tag": REQUIRED_TAG,
        "materialize_job_id": job_id,
        "pilot_root": str(root),
        "source": str(root),
        "destination": str(destination),
        "sbatch_receipt": str(receipt_path),
        "sbatch_receipt_sha256": _sha256_file(receipt_path),
        "intent": str(intent_path),
        "intent_sha256": _sha256_file(intent_path),
        "intent_id": intent["intent_id"],
        "scheduler": scheduler,
        "source_device": partial_stat.st_dev,
        "source_inode": partial_stat.st_ino,
        "completed_at": _utc_now(),
    }
    completion["completion_id"] = _sha256_bytes(_canonical_bytes(completion))
    _atomic_write_once(completion_path, _json_bytes(completion))
    seal = recovery_evidence.seal_quarantine(
        tree=destination,
        evidence_root=evidence_root,
        release_id=RELEASE_ID,
        failed_job_id=job_id,
        quarantine_completion=completion_path,
        apply=True,
    )
    if seal.get("passed") is not True:
        raise MaterializationPilotError(
            "interrupted pilot quarantine did not seal"
        )
    return completion | {
        "status": "quarantined_and_sealed",
        "quarantine_seal": seal,
    }


def run_materialization_pilot(
    *,
    pilot_root: str | Path,
    release_checkout: str | Path,
    expected_tag: str,
    expected_commit: str,
    harness_source: str | Path,
    serving_source: str | Path,
    ownership_policy: str | Path,
    integrity_normalization_policy: str | Path,
    reconciliation_incident: str | Path,
    recovered_setuptools_record: str | Path,
    conda_toolchain_root: str | Path,
    source_package_cache: str | Path,
    durable_git_release_marker: str | Path,
    sbatch_receipt: str | Path | None = None,
    scheduler_job_id: str | None = None,
    scheduler_runner: Callable[
        [Sequence[str]], subprocess.CompletedProcess[str]
    ] | None = None,
    require_scheduler: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    root = _destination_root(pilot_root)
    checkout = _existing_directory(release_checkout, description="release checkout")
    harness = _existing_directory(harness_source, description="live harness prefix")
    serving = _existing_directory(serving_source, description="live serving prefix")
    policy = _existing_file(ownership_policy, description="ownership policy")
    integrity_policy = _existing_file(
        integrity_normalization_policy,
        description="integrity-normalization policy",
    )
    incident = _existing_file(
        reconciliation_incident, description="Conda reconciliation incident"
    )
    recovered = _existing_file(
        recovered_setuptools_record,
        description="recovered Setuptools Conda record",
    )
    toolchain_root = _existing_directory(
        conda_toolchain_root,
        description="sealed Conda toolchain root",
    )
    package_cache_source = _existing_directory(
        source_package_cache, description="source Conda package cache"
    )
    durable_marker = _existing_file(
        durable_git_release_marker,
        description="durable Git release marker",
    )
    _validate_isolation(
        root,
        release_checkout=checkout,
        harness_source=harness,
        serving_source=serving,
        source_package_cache=package_cache_source,
        conda_toolchain_root=toolchain_root,
    )
    git_identity = _verify_exact_annotated_checkout(
        checkout, expected_tag=expected_tag, expected_commit=expected_commit
    )
    inputs = _input_binding(
        release_checkout=checkout,
        harness_source=harness,
        serving_source=serving,
        ownership_policy=policy,
        integrity_normalization_policy=integrity_policy,
        reconciliation_incident=incident,
        recovered_setuptools_record=recovered,
        conda_toolchain_root=toolchain_root,
        source_package_cache=package_cache_source,
        durable_git_release_marker=durable_marker,
    )
    if (
        inputs["durable_git_release"]["release_git_commit"] != expected_commit
        or inputs["durable_git_release"]["release_tag_object"]
        != git_identity["tag_object"]
    ):
        raise MaterializationPilotError(
            "durable Git release marker belongs to another release identity"
        )
    layout = _layout(root)
    if apply and require_scheduler:
        if sbatch_receipt is None:
            raise MaterializationPilotError(
                "production pilot apply requires the rendered sbatch receipt"
            )
        scheduler_job_id = (
            os.environ.get("SLURM_JOB_ID")
            if scheduler_job_id is None
            else scheduler_job_id
        )
        if not isinstance(scheduler_job_id, str) or not scheduler_job_id.isdigit():
            raise MaterializationPilotError(
                "production pilot apply must execute inside its exact Slurm job"
            )

    marker_path = root / COMPLETE_MARKER
    if marker_path.exists() or marker_path.is_symlink():
        if apply and require_scheduler:
            _record_active_scheduler_attempt(
                root=root,
                sbatch_receipt=sbatch_receipt,
                job_id=scheduler_job_id,
                runner=scheduler_runner,
            )
        verified = _verify_materialization_pilot_semantic(root)
        marker = _read_json(marker_path, description="pilot completion marker")
        _verify_completed_inputs(
            marker,
            expected_tag=expected_tag,
            expected_commit=expected_commit,
            git_identity=git_identity,
            inputs=inputs,
            layout=layout,
        )
        return {**verified, "status": "already_complete"}

    capture_kwargs = {
        "output_root": layout["environment_capture_root"],
        "harness_source": harness,
        "serving_source": serving,
        "ownership_policy": policy,
        "integrity_normalization_policy": integrity_policy,
        "reconciliation_incident": incident,
        "recovered_setuptools_record": recovered,
    }
    if not apply:
        capture_plan = capture.capture_environments(**capture_kwargs, apply=False)
        if capture_plan.get("status") != "dry_run":
            raise MaterializationPilotError("capture pilot preflight was not a dry-run")
        return {
            "status": "dry_run",
            "schema_version": SCHEMA_VERSION,
            "release_id": RELEASE_ID,
            "expected_tag": expected_tag,
            "expected_commit": expected_commit,
            "git_identity": git_identity,
            "inputs": inputs,
            "layout": layout,
            "capture_dry_run": _without_status(capture_plan),
            "would_run": [
                "capture:dry_run/apply/verify",
                "materialization:dry_run/apply/verify",
                "freeze:dry_run/apply/verify",
                "live-source-before/after-byte-inventory",
                "setuptools-and-inode-audits",
                "repeatable-sealed-verification",
                "active-scheduler-no-requeue-and-spooled-script-evidence",
                "post-terminal-scheduler-acceptance",
                COMPLETE_MARKER,
            ],
        }

    before = {
        "harness": capture.directory_inventory(harness),
        "serving": capture.directory_inventory(serving),
    }
    before_records = {
        role: {
            "filename": LIVE_INVENTORY_FILENAMES[(role, "before")],
            "sha256": _sha256_bytes(_json_bytes(inventory)),
            "inventory_sha256": inventory["inventory_sha256"],
        }
        for role, inventory in before.items()
    }
    intent: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "kind": "schema5-materialization-pilot-intent",
        "publication_protocol": "intent_first_stage_evidence_pilot_marker_last",
        "expected_tag": expected_tag,
        "expected_commit": expected_commit,
        "git_identity": git_identity,
        "inputs": inputs,
        "layout": layout,
        "live_source_inventories": before_records,
        "conda_scope_contract": {
            "capture_invokes_conda": False,
            "freeze_invokes_conda": False,
            "materialization_conda_source": (
                "normalized_read_only_seeds_and_immutable_preseeded_cache_only"
            ),
            "live_prefix_conda_queries": 0,
        },
    }
    intent["intent_id"] = _sha256_bytes(_canonical_bytes(intent))
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_once(root / INTENT_MARKER, _json_bytes(intent))
    _validate_intent(_read_json(root / INTENT_MARKER, description="pilot intent"))
    if require_scheduler:
        _record_active_scheduler_attempt(
            root=root,
            sbatch_receipt=sbatch_receipt,
            job_id=scheduler_job_id,
            runner=scheduler_runner,
        )
    for role in ("harness", "serving"):
        record = _write_inventory_evidence(
            root, role=role, phase="before", inventory=before[role]
        )
        if (
            record["sha256"] != before_records[role]["sha256"]
            or record["inventory_sha256"]
            != before_records[role]["inventory_sha256"]
        ):
            raise MaterializationPilotError(
                f"{role} live before-inventory publication drifted"
            )

    _run_dry_once(
        root,
        stage="capture",
        operation=lambda: capture.capture_environments(**capture_kwargs, apply=False),
    )
    _run_apply_and_verify(
        root,
        stage="capture",
        apply_operation=lambda: capture.capture_environments(
            **capture_kwargs, apply=True
        ),
        verify_operation=lambda: capture.verify_capture(
            layout["environment_capture_root"]
        ),
    )
    capture_report = capture.verify_capture(layout["environment_capture_root"])
    seeds = capture_report["seed_prefixes"]
    expected_package_cache_seed_input = (
        materialize.selected_package_cache_input_binding(
            source_package_cache=package_cache_source,
            harness_seed=seeds["harness"],
            serving_seed=seeds["serving"],
        )
    )

    materialize_kwargs = {
        "output_root": layout["materialization_root"],
        "environment_capture_root": layout["environment_capture_root"],
        "source_repository": checkout,
        "release_worktree": layout["release_worktree"],
        "source_harness_prefix": seeds["harness"],
        "source_serving_prefix": seeds["serving"],
        "source_package_cache": package_cache_source,
        "harness_prefix": layout["harness_prefix"],
        "serving_prefix": layout["serving_prefix"],
        "conda_toolchain_root": toolchain_root,
        "expected_package_cache_seed_input": (
            expected_package_cache_seed_input
        ),
    }

    def gated_materialize(*, apply_stage: bool) -> Mapping[str, Any]:
        try:
            live_toolchain = (
                conda_toolchain.verified_conda_toolchain_binding(
                    toolchain_root,
                    exercise=True,
                )
            )
        except (
            OSError,
            conda_toolchain.CondaToolchainProvisionError,
        ) as exc:
            raise MaterializationPilotError(
                f"sealed Conda toolchain verification failed: {exc}"
            ) from exc
        if live_toolchain != inputs["conda_toolchain"]:
            raise MaterializationPilotError(
                "sealed Conda toolchain drifted immediately before "
                "materialization"
            )
        return materialize.materialize_release(
            **materialize_kwargs,
            apply=apply_stage,
        )

    current_materialization_dry_report = dict(
        gated_materialize(apply_stage=False)
    )
    if current_materialization_dry_report.get("status") != "dry_run":
        raise MaterializationPilotError(
            "materialization current preflight did not return dry_run status"
        )
    materialization_dry_evidence = _run_dry_once(
        root,
        stage="materialization",
        operation=lambda: current_materialization_dry_report,
    )
    # Compare a replayed dry-run with a freshly recomputed binding before the
    # first mutating materialization operation.
    _require_identical_materialization_phase_bindings(
        {
            "persisted-dry-run": materialization_dry_evidence["report"],
            "current-dry-run": _without_status(
                current_materialization_dry_report
            ),
        }
    )
    materialization_apply_evidence, materialization_verify_evidence = (
        _run_apply_and_verify(
            root,
            stage="materialization",
            apply_operation=lambda: gated_materialize(apply_stage=True),
            verify_operation=lambda: materialize.verify_materialization(
                layout["materialization_root"]
            ),
        )
    )
    _require_identical_materialization_phase_bindings(
        {
            "dry-run": materialization_dry_evidence["report"],
            "apply": materialization_apply_evidence["report"],
            "verify": materialization_verify_evidence["report"],
        }
    )

    freeze_kwargs = {
        "output_root": layout["release_bundle"],
        "release_worktree": layout["release_worktree"],
        "harness_prefix": layout["harness_prefix"],
        "serving_prefix": layout["serving_prefix"],
        "model_contract_path": (
            Path(layout["release_worktree"])
            / "configs"
            / "model_contracts.v1.json"
        ),
        "fleet_contract_path": (
            Path(layout["release_worktree"])
            / "configs"
            / "schema5_fleet.v1.json"
        ),
        "seal_worktree": True,
        "seal_environments": True,
        "seal_output_root": True,
    }
    _run_dry_once(
        root,
        stage="freeze",
        operation=lambda: freeze.create_release_bundle(**freeze_kwargs, apply=False),
    )
    _run_apply_and_verify(
        root,
        stage="freeze",
        apply_operation=lambda: freeze.create_release_bundle(
            **freeze_kwargs, apply=True
        ),
        verify_operation=lambda: freeze.verify_release_bundle(
            layout["release_bundle"]
        ),
    )

    after = {
        "harness": capture.directory_inventory(harness),
        "serving": capture.directory_inventory(serving),
    }
    for role in ("harness", "serving"):
        _write_inventory_evidence(
            root, role=role, phase="after", inventory=after[role]
        )
        if before[role] != after[role]:
            raise MaterializationPilotError(
                f"live {role} prefix changed across the materialization pilot"
            )

    sealed_audits = _sealed_artifact_audits(layout=layout)
    all_roots = _pilot_roots(layout=layout, inputs=inputs)
    transaction_inode_edges = _inode_edges(all_roots, _TRANSACTION_INODE_EDGES)
    repeatable = _repeatable_verification(layout)
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "kind": "schema5-materialization-pilot-audit",
        "live_source_inventories_equal": True,
        "live_source_inventory_sha256": {
            role: before[role]["inventory_sha256"]
            for role in ("harness", "serving")
        },
        **sealed_audits,
        "transaction_inode_edges": transaction_inode_edges,
        "repeatable_sealed_verification": repeatable,
        "conda_scope_contract": intent["conda_scope_contract"],
    }
    audit["audit_id"] = _sha256_bytes(_canonical_bytes(audit))
    _atomic_write_once(root / AUDIT_FILENAME, _json_bytes(audit))

    artifacts = _artifact_inventory(root)
    stage_reports = {
        stage: _read_stage_evidence(root, stage=stage, phase="verify")["report"]
        for stage in ("capture", "materialization", "freeze")
    }
    marker: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "kind": "schema5-materialization-pilot-completion",
        "complete": True,
        "publication_protocol": "intent_first_stage_evidence_pilot_marker_last",
        "pilot_root": str(root),
        "expected_tag": expected_tag,
        "expected_commit": expected_commit,
        "git_identity": git_identity,
        "inputs": inputs,
        "layout": layout,
        "artifacts": artifacts,
        "live_source_inventories": {
            role: {
                "before": before[role]["inventory_sha256"],
                "after": after[role]["inventory_sha256"],
                "equal": before[role] == after[role],
            }
            for role in ("harness", "serving")
        },
        "stage_ids": {
            "capture_id": stage_reports["capture"]["capture_id"],
            "materialization_id": stage_reports["materialization"][
                "materialization_id"
            ],
            "release_bundle_id": stage_reports["freeze"]["release_bundle_id"],
        },
        "package_cache_seed_input": stage_reports[
            "materialization"
        ]["package_cache_seed_input"],
        "audit_id": audit["audit_id"],
    }
    marker["pilot_id"] = _sha256_bytes(_canonical_bytes(marker))
    # Semantic evidence is now complete.  The only permitted later publication is
    # the separately verified post-terminal scheduler-acceptance marker.
    _atomic_write_once(marker_path, _json_bytes(marker))
    verified = _verify_materialization_pilot_semantic(root)
    return {**verified, "status": "created"}


def _verify_materialization_pilot_semantic(
    pilot_root: str | Path,
) -> dict[str, Any]:
    """Verify semantic pilot evidence without mutable build inputs or Conda."""

    root = _existing_directory(pilot_root, description="materialization pilot root")
    marker_path = root / COMPLETE_MARKER
    marker = _read_json(marker_path, description="pilot completion marker")
    if stat.S_IMODE(marker_path.stat().st_mode) & 0o222:
        raise MaterializationPilotError("pilot completion marker remains writable")
    candidate = dict(marker)
    pilot_id = candidate.pop("pilot_id", None)
    required = {
        "schema_version",
        "release_id",
        "kind",
        "complete",
        "publication_protocol",
        "pilot_root",
        "expected_tag",
        "expected_commit",
        "git_identity",
        "inputs",
        "layout",
        "artifacts",
        "live_source_inventories",
        "stage_ids",
        "package_cache_seed_input",
        "audit_id",
        "pilot_id",
    }
    layout = marker.get("layout")
    if (
        set(marker) != required
        or marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("release_id") != RELEASE_ID
        or marker.get("kind") != "schema5-materialization-pilot-completion"
        or marker.get("complete") is not True
        or marker.get("publication_protocol")
        != "intent_first_stage_evidence_pilot_marker_last"
        or marker.get("pilot_root") != str(root)
        or marker.get("expected_tag") != REQUIRED_TAG
        or _COMMIT_RE.fullmatch(str(marker.get("expected_commit", ""))) is None
        or not isinstance(layout, dict)
        or layout != _layout(root)
        or pilot_id != _sha256_bytes(_canonical_bytes(candidate))
    ):
        raise MaterializationPilotError("pilot completion marker is invalid")
    _validate_artifact_inventory(root, marker.get("artifacts"))

    intent = _read_json(root / INTENT_MARKER, description="pilot intent")
    _validate_intent(intent)
    recorded_toolchain = intent.get("inputs", {}).get("conda_toolchain")
    source_package_cache = intent.get("inputs", {}).get(
        "source_package_cache"
    )
    if (
        not isinstance(recorded_toolchain, dict)
        or not isinstance(recorded_toolchain.get("toolchain_root"), str)
        or not Path(recorded_toolchain["toolchain_root"]).is_absolute()
    ):
        raise MaterializationPilotError(
            "pilot sealed Conda toolchain binding is invalid"
        )
    try:
        live_toolchain = conda_toolchain.verified_conda_toolchain_binding(
            recorded_toolchain["toolchain_root"],
            exercise=True,
        )
    except (
        OSError,
        conda_toolchain.CondaToolchainProvisionError,
    ) as exc:
        raise MaterializationPilotError(
            f"sealed Conda toolchain verification failed: {exc}"
        ) from exc
    if live_toolchain != recorded_toolchain:
        raise MaterializationPilotError(
            "pilot sealed Conda toolchain binding drifted"
        )
    if (
        not isinstance(source_package_cache, str)
        or not Path(source_package_cache).is_absolute()
        or "\n" in source_package_cache
        or "\r" in source_package_cache
    ):
        raise MaterializationPilotError(
            "pilot source package-cache binding is invalid"
        )
    if any(
        marker.get(field) != intent.get(field)
        for field in (
            "expected_tag",
            "expected_commit",
            "git_identity",
            "inputs",
            "layout",
        )
    ):
        raise MaterializationPilotError("pilot completion no longer matches its intent")

    inventories: dict[str, dict[str, Any]] = {}
    for role in ("harness", "serving"):
        before = _read_inventory_evidence(root, role=role, phase="before")
        after = _read_inventory_evidence(root, role=role, phase="after")
        inventory_record = marker.get("live_source_inventories", {}).get(role)
        if (
            before != after
            or not isinstance(inventory_record, dict)
            or inventory_record
            != {
                "before": before.get("inventory_sha256"),
                "after": after.get("inventory_sha256"),
                "equal": True,
            }
            or intent["live_source_inventories"][role].get("inventory_sha256")
            != before.get("inventory_sha256")
            or intent["live_source_inventories"][role].get("sha256")
            != _sha256_file(
                root / LIVE_INVENTORY_FILENAMES[(role, "before")]
            )
        ):
            raise MaterializationPilotError(
                f"live {role} before/after inventory evidence is inconsistent"
            )
        inventories[role] = before

    phase_evidence: dict[str, dict[str, dict[str, Any]]] = {}
    for stage in ("capture", "materialization", "freeze"):
        phase_evidence[stage] = {}
        for phase in ("dry_run", "apply", "verify"):
            phase_evidence[stage][phase] = _read_stage_evidence(
                root, stage=stage, phase=phase
            )

    current_reports = {
        "capture": _without_status(
            capture.verify_capture(layout["environment_capture_root"])
        ),
        "materialization": _without_status(
            materialize.verify_materialization(layout["materialization_root"])
        ),
        "freeze": _without_status(
            freeze.verify_release_bundle(layout["release_bundle"])
        ),
    }
    for stage, report in current_reports.items():
        stored = _read_stage_evidence(
            root, stage=stage, phase="verify"
        )["report"]
        if report != stored:
            raise MaterializationPilotError(
                f"{stage} sealed verification report drifted"
            )
    materialization_report = current_reports["materialization"]
    _require_identical_materialization_phase_bindings(
        {
            "dry-run": phase_evidence["materialization"]["dry_run"][
                "report"
            ],
            "apply": phase_evidence["materialization"]["apply"]["report"],
            "verify": phase_evidence["materialization"]["verify"]["report"],
            "current-verify": materialization_report,
        }
    )
    package_cache_seed_input = materialization_report.get(
        "package_cache_seed_input"
    )
    if (
        materialization_report.get("conda_toolchain") != live_toolchain
        or not isinstance(package_cache_seed_input, dict)
        or marker.get("package_cache_seed_input")
        != package_cache_seed_input
    ):
        raise MaterializationPilotError(
            "pilot materialization toolchain or selected package-cache "
            "input binding drifted"
        )
    capture_report = current_reports["capture"]
    incident_path = Path(capture_report["reconciliation_incident_path"])
    incident = _read_json(
        incident_path,
        description="pilot-captured Conda reconciliation incident",
    )
    reconciliation_incident = {
        "path": str(incident_path),
        "sha256": capture_report["reconciliation_incident_sha256"],
        "incident_id": incident.get("incident_id"),
        "harness_stale_conda_record_present": incident.get(
            "harness_stale_conda_record_present"
        ),
        "serving_stale_conda_record_present": incident.get(
            "serving_stale_conda_record_present"
        ),
    }
    if (
        _SHA256_RE.fullmatch(str(reconciliation_incident["sha256"])) is None
        or _SHA256_RE.fullmatch(
            str(reconciliation_incident["incident_id"])
        )
        is None
        or type(
            reconciliation_incident["harness_stale_conda_record_present"]
        )
        is not bool
        or type(
            reconciliation_incident["serving_stale_conda_record_present"]
        )
        is not bool
    ):
        raise MaterializationPilotError(
            "pilot-captured Conda reconciliation identity is malformed"
        )
    expected_stage_ids = {
        "capture_id": current_reports["capture"]["capture_id"],
        "materialization_id": current_reports["materialization"][
            "materialization_id"
        ],
        "release_bundle_id": current_reports["freeze"]["release_bundle_id"],
    }
    package_cache_seed_sha256 = current_reports["materialization"].get(
        "conda_package_cache_seed_sha256"
    )
    if _SHA256_RE.fullmatch(str(package_cache_seed_sha256)) is None:
        raise MaterializationPilotError(
            "materialization lacks immutable package-cache seed identity"
        )
    if marker.get("stage_ids") != expected_stage_ids:
        raise MaterializationPilotError("pilot stage identity binding drifted")

    audit = _read_json(root / AUDIT_FILENAME, description="pilot audit")
    audit_candidate = dict(audit)
    audit_id = audit_candidate.pop("audit_id", None)
    if (
        audit.get("schema_version") != SCHEMA_VERSION
        or audit.get("release_id") != RELEASE_ID
        or audit.get("kind") != "schema5-materialization-pilot-audit"
        or audit.get("live_source_inventories_equal") is not True
        or audit.get("live_source_inventory_sha256")
        != {
            role: inventories[role]["inventory_sha256"]
            for role in ("harness", "serving")
        }
        or audit.get("conda_scope_contract") != intent["conda_scope_contract"]
        or audit_id != _sha256_bytes(_canonical_bytes(audit_candidate))
        or marker.get("audit_id") != audit_id
    ):
        raise MaterializationPilotError("pilot audit identity is invalid")

    sealed_audits = _sealed_artifact_audits(layout=layout)
    for field in (
        "ownership_policy_sha256",
        "integrity_normalization_policy_sha256",
        "setuptools",
        "sealed_inode_edges",
        "control_consumer",
    ):
        if audit.get(field) != sealed_audits[field]:
            raise MaterializationPilotError(f"sealed pilot {field} audit drifted")
    repeated = _repeatable_verification(layout)
    if audit.get("repeatable_sealed_verification") != repeated:
        raise MaterializationPilotError(
            "repeatable sealed-verification evidence drifted"
        )
    transaction_edges = audit.get("transaction_inode_edges")
    if (
        not isinstance(transaction_edges, dict)
        or set(transaction_edges)
        != {f"{left}__{right}" for left, right in _TRANSACTION_INODE_EDGES}
        or any(
            not isinstance(record, dict)
            or record.get("shared_regular_inode_count") != 0
            for record in transaction_edges.values()
        )
    ):
        raise MaterializationPilotError("transaction inode evidence is invalid")
    return {
        "status": "verified",
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "pilot_id": pilot_id,
        "pilot_marker_sha256": _sha256_file(marker_path),
        "expected_tag": marker["expected_tag"],
        "expected_commit": marker["expected_commit"],
        "capture_id": expected_stage_ids["capture_id"],
        "materialization_id": expected_stage_ids["materialization_id"],
        "release_bundle_id": expected_stage_ids["release_bundle_id"],
        "audit_id": audit_id,
        "live_source_inventory_sha256": audit[
            "live_source_inventory_sha256"
        ],
        "ownership_policy_sha256": audit["ownership_policy_sha256"],
        "integrity_normalization_policy_sha256": audit[
            "integrity_normalization_policy_sha256"
        ],
        "conda_toolchain": live_toolchain,
        "source_package_cache": source_package_cache,
        "package_cache_seed_input": dict(package_cache_seed_input),
        "conda_package_cache_seed_sha256": package_cache_seed_sha256,
        "reconciliation_incident": reconciliation_incident,
        "durable_git_release": dict(
            intent["inputs"]["durable_git_release"]
        ),
        "verifier_runtime": _sealed_verifier_runtime(layout),
        "setuptools_contract": {
            "runtime_version": "81.0.0",
            "distribution_count_per_prefix": 1,
            "setuptools_82_conda_record_count": 0,
            "setuptools_82_versioned_path_count": 0,
        },
        "shared_regular_inode_count": 0,
        "sealed_verification_repeatable": True,
    }


def verify_materialization_pilot(
    pilot_root: str | Path,
) -> dict[str, Any]:
    """Require both semantic completion and durable Slurm acceptance."""

    root = _existing_directory(
        pilot_root, description="materialization pilot root"
    )
    semantic = _verify_materialization_pilot_semantic(root)
    scheduler = _verify_pilot_scheduler_acceptance(
        root=root, semantic=semantic
    )
    return {
        **semantic,
        "scheduler_acceptance": {
            "acceptance_id": scheduler["acceptance_id"],
            "marker": str(root / SCHEDULER_ACCEPTANCE_FILENAME),
            "marker_sha256": _sha256_file(
                root / SCHEDULER_ACCEPTANCE_FILENAME
            ),
            "job_id": scheduler["job_id"],
            "receipt": scheduler["sbatch_receipt"],
            "receipt_sha256": scheduler["sbatch_receipt_sha256"],
            "receipt_id": scheduler["sbatch_receipt_id"],
            "effective_requeue": scheduler["effective_requeue"],
            "spooled_script_sha256": scheduler[
                "spooled_script_sha256"
            ],
            "terminal_state": scheduler["terminal"][
                "normalized_state"
            ],
            "exit_code": scheduler["terminal"]["exit_code"],
            "reason": scheduler["terminal"]["reason"],
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="dry-run or apply the two-prefix pilot")
    run.add_argument("--pilot-root", type=Path, required=True)
    run.add_argument("--release-checkout", type=Path, required=True)
    run.add_argument("--expected-tag", required=True)
    run.add_argument("--expected-commit", required=True)
    run.add_argument("--harness-source", type=Path, required=True)
    run.add_argument("--serving-source", type=Path, required=True)
    run.add_argument("--ownership-policy", type=Path, required=True)
    run.add_argument(
        "--integrity-normalization-policy", type=Path, required=True
    )
    run.add_argument("--reconciliation-incident", type=Path, required=True)
    run.add_argument("--recovered-setuptools-record", type=Path, required=True)
    run.add_argument("--conda-toolchain-root", type=Path, required=True)
    run.add_argument("--source-package-cache", type=Path, required=True)
    run.add_argument(
        "--durable-git-release-marker", type=Path, required=True
    )
    run.add_argument("--sbatch-receipt", type=Path)
    run.add_argument("--apply", action="store_true")
    render = subparsers.add_parser(
        "render-sbatch",
        help="dry-run or publish an immutable --no-requeue pilot sbatch",
    )
    render.add_argument("--sbatch-path", type=Path, required=True)
    render.add_argument("--log-dir", type=Path, required=True)
    render.add_argument("--partition", required=True)
    render.add_argument("--python-executable", type=Path, required=True)
    render.add_argument("--pilot-root", type=Path, required=True)
    render.add_argument("--release-checkout", type=Path, required=True)
    render.add_argument("--expected-tag", required=True)
    render.add_argument("--expected-commit", required=True)
    render.add_argument("--harness-source", type=Path, required=True)
    render.add_argument("--serving-source", type=Path, required=True)
    render.add_argument("--ownership-policy", type=Path, required=True)
    render.add_argument(
        "--integrity-normalization-policy", type=Path, required=True
    )
    render.add_argument("--reconciliation-incident", type=Path, required=True)
    render.add_argument("--recovered-setuptools-record", type=Path, required=True)
    render.add_argument("--conda-toolchain-root", type=Path, required=True)
    render.add_argument("--source-package-cache", type=Path, required=True)
    render.add_argument(
        "--durable-git-release-marker", type=Path, required=True
    )
    render.add_argument("--apply", action="store_true")
    submit = subparsers.add_parser(
        "submit-sbatch",
        help=(
            "transactionally submit or adopt the exact immutable pilot "
            "sbatch through complete squeue+sacct reconciliation"
        ),
    )
    submit.add_argument("--pilot-root", type=Path, required=True)
    submit.add_argument("--sbatch-receipt", type=Path, required=True)
    submit.add_argument("--scheduler-user", required=True)
    submit.add_argument(
        "--prior-quarantine-seal",
        type=Path,
        action="append",
        default=[],
        help=(
            "repeatable explicit marker-last seal for a previously quarantined "
            "scheduler-transient pilot allocation"
        ),
    )
    submit.add_argument(
        "--visibility-timeout",
        type=float,
        default=DEFAULT_SUBMISSION_VISIBILITY_TIMEOUT,
    )
    submit.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_SUBMISSION_POLL_SECONDS,
    )
    submit.add_argument("--apply", action="store_true")
    verify = subparsers.add_parser("verify", help="verify completed pilot evidence")
    verify.add_argument("--pilot-root", type=Path, required=True)
    acceptance = subparsers.add_parser(
        "accept-scheduler",
        help=(
            "bind one completed no-requeue Slurm allocation to the semantic "
            "pilot before production acceptance"
        ),
    )
    acceptance.add_argument("--pilot-root", type=Path, required=True)
    acceptance.add_argument("--sbatch-receipt", type=Path, required=True)
    acceptance.add_argument("--job-id", required=True)
    acceptance.add_argument("--apply", action="store_true")
    runtime = subparsers.add_parser(
        "conda-toolchain-binding",
        help="fully verify and bind the sealed release-local Conda toolchain",
    )
    runtime.add_argument("--conda-toolchain-root", type=Path, required=True)
    quarantine = subparsers.add_parser(
        "quarantine",
        help=(
            "preserve a transiently interrupted canonical pilot before retrying "
            "the exact tagged sbatch"
        ),
    )
    quarantine.add_argument("--pilot-root", type=Path, required=True)
    quarantine.add_argument("--sbatch-receipt", type=Path, required=True)
    quarantine.add_argument("--job-id", required=True)
    quarantine.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            report = verify_materialization_pilot(args.pilot_root)
        elif args.command == "accept-scheduler":
            report = accept_pilot_scheduler(
                pilot_root=args.pilot_root,
                sbatch_receipt=args.sbatch_receipt,
                job_id=args.job_id,
                apply=args.apply,
            )
        elif args.command == "submit-sbatch":
            report = submit_materialization_pilot_sbatch(
                pilot_root=args.pilot_root,
                sbatch_receipt=args.sbatch_receipt,
                scheduler_user=args.scheduler_user,
                prior_quarantine_seals=args.prior_quarantine_seal,
                visibility_timeout=args.visibility_timeout,
                poll_seconds=args.poll_seconds,
                apply=args.apply,
            )
        elif args.command == "conda-toolchain-binding":
            report = conda_toolchain.verified_conda_toolchain_binding(
                args.conda_toolchain_root,
                exercise=True,
            )
        elif args.command == "quarantine":
            report = quarantine_interrupted_pilot(
                pilot_root=args.pilot_root,
                sbatch_receipt=args.sbatch_receipt,
                job_id=args.job_id,
                apply=args.apply,
            )
        elif args.command == "render-sbatch":
            report = render_materialization_pilot_sbatch(
                sbatch_path=args.sbatch_path,
                log_dir=args.log_dir,
                partition=args.partition,
                python_executable=args.python_executable,
                pilot_root=args.pilot_root,
                release_checkout=args.release_checkout,
                expected_tag=args.expected_tag,
                expected_commit=args.expected_commit,
                harness_source=args.harness_source,
                serving_source=args.serving_source,
                ownership_policy=args.ownership_policy,
                integrity_normalization_policy=(
                    args.integrity_normalization_policy
                ),
                reconciliation_incident=args.reconciliation_incident,
                recovered_setuptools_record=args.recovered_setuptools_record,
                conda_toolchain_root=args.conda_toolchain_root,
                source_package_cache=args.source_package_cache,
                durable_git_release_marker=args.durable_git_release_marker,
                apply=args.apply,
            )
        else:
            report = run_materialization_pilot(
                pilot_root=args.pilot_root,
                release_checkout=args.release_checkout,
                expected_tag=args.expected_tag,
                expected_commit=args.expected_commit,
                harness_source=args.harness_source,
                serving_source=args.serving_source,
                ownership_policy=args.ownership_policy,
                integrity_normalization_policy=(
                    args.integrity_normalization_policy
                ),
                reconciliation_incident=args.reconciliation_incident,
                recovered_setuptools_record=args.recovered_setuptools_record,
                conda_toolchain_root=args.conda_toolchain_root,
                source_package_cache=args.source_package_cache,
                durable_git_release_marker=args.durable_git_release_marker,
                sbatch_receipt=args.sbatch_receipt,
                require_scheduler=True,
                apply=args.apply,
            )
    except (
        OSError,
        UnicodeError,
        ValueError,
        capture.EnvironmentCaptureError,
        materialize.MaterializationError,
        freeze.ReleaseFreezeError,
        conda_toolchain.CondaToolchainProvisionError,
        MaterializationPilotError,
    ) as exc:
        print(f"[schema5-materialization-pilot] ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
