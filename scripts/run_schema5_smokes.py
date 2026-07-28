#!/usr/bin/env python3
"""Run or verify the 41 immutable schema-5 smoke cells at concurrency one.

The command invokes the frozen dispatcher through the exact immutable harness Python,
using the same verified batch-task boundary as production. Each attempt starts from
empty, attempt-scoped run roots. An interrupted or transport-censored attempt is
inventoried and sealed before a successor may draw; rows never cross attempts.
``--apply`` is required to contact servers. The marker-last CURRENT selector is
published only when all 15 + 20 + 6 cells are semantically complete under schema 5
with trusted capacity/rollout provenance and zero censor/protocol incidents.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
for value in (REPO, SOURCE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog  # noqa: E402
from agents_scaling import runtime_integrity  # noqa: E402
from agents_scaling.experiment.analyze import _censor_accounting  # noqa: E402
from agents_scaling.experiment.artifact_policy import (  # noqa: E402
    ArtifactPolicyError,
    load_artifact_policy,
)
from agents_scaling.experiment.completion import (  # noqa: E402
    CompletionState,
    get_completion_status,
    read_canonical_results,
    trusted_generation_catalog_errors,
)
from agents_scaling.experiment.manifest import load_manifest  # noqa: E402
from agents_scaling.experiment.transport_censor import (  # noqa: E402
    TRANSPORT_CENSOR_PROTOCOL_HASH,
    TRANSPORT_CENSOR_PROTOCOL_VERSION,
)
from agents_scaling.serving.profiles import serving_profile_for_cell  # noqa: E402
from agents_scaling.serving.generation_catalog import (  # noqa: E402
    TrustedGenerationCatalog,
)
from scripts.init_schema5_smokes import (  # noqa: E402
    ATTEMPT_BINDING_PROTOCOL,
    SMOKE_SUITES,
    SmokeInitializationError,
    _verify_suite,
    initialize_all,
)
from slurm import schema5_control  # noqa: E402


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUNNABLE_STATES = frozenset(
    {CompletionState.MISSING, CompletionState.PARTIAL, CompletionState.RETRYABLE}
)
_SUITE_ARTIFACT_NAMES = {
    "schema5_smoke_32b_long_v1": "long_32b_smoke",
    "schema5_smoke_selective_long_v1": "selective_long_smoke",
    "schema5_smoke_standard_canaries_v1": "standard_canary_smoke",
}
_PROVENANCE_ERROR_MARKERS = (
    "artifact policy",
    "authoritative policy",
    "endpoint_generation",
    "environment",
    "git_commit",
    "model contract",
    "model_revision",
    "provenance",
    "release_id",
    "rollout_generation",
    "schema-5",
    "serving_profile",
    "tokenizer_revision",
)
_PROTOCOL_FAILURE_TYPES = frozenset(
    {
        "GenerationProtocolCensorError",
        "ServerResponseProtocolError",
        "ThinkingBudgetProtocolError",
    }
)
_TRUNCATION_FAILURE_TYPES = frozenset({"GenerationTruncationError"})

SMOKE_ATTEMPT_BASE_NAME = "schema5-smoke-readiness-v1"
SMOKE_ATTEMPT_RUNS_NAME = "schema5-smoke-attempt-runs-v1"
ATTEMPTS_DIRECTORY = "attempts"
ATTEMPT_POINTERS_DIRECTORY = "attempt-pointers"
CURRENT_SELECTOR_NAME = "CURRENT.json"
ATTEMPT_COMPLETE_NAME = "SMOKE_ATTEMPT_COMPLETE.json"
ATTEMPT_FAILURE_NAME = "SMOKE_ATTEMPT_FAILURE.json"
ATTEMPT_EVIDENCE_NAME = "smoke_runs.json"
ATTEMPT_POINTER_PROTOCOL = "schema5-v1.2-r13-smoke-attempt-pointer-v1"
CURRENT_SELECTOR_PROTOCOL = "schema5-v1.2-r13-smoke-current-selector-v1"
ATTEMPT_COMPLETE_PROTOCOL = "schema5-v1.2-r13-smoke-attempt-complete-v1"
ATTEMPT_FAILURE_PROTOCOL = "schema5-v1.2-r13-smoke-attempt-failure-v1"
MAX_AUTOMATIC_ATTEMPTS = 3

_ATTEMPT_POINTER_FIELDS = {
    "schema_version",
    "protocol",
    "attempt_id",
    "attempt_ordinal",
    "attempt_root",
    "runs_root",
    "execution_root",
    "immutable_sha256",
    "readiness_generation",
    "protected_capacity",
    "predecessor",
    "created_at",
    "created_timestamp",
    "pointer_id",
}
_CURRENT_SELECTOR_FIELDS = {
    "schema_version",
    "protocol",
    "attempt_id",
    "attempt_pointer",
    "attempt_pointer_sha256",
    "pointer_id",
    "completion_marker",
    "completion_marker_sha256",
    "completion_id",
    "evidence",
    "evidence_sha256",
    "selector_id",
}


class SmokeRunError(RuntimeError):
    """Smoke execution or semantic verification failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _with_identity(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = dict(value)
    payload[field] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def _verify_identity(
    value: Mapping[str, Any], field: str, *, description: str
) -> None:
    identity = dict(value)
    observed = identity.pop(field, None)
    if (
        not isinstance(observed, str)
        or _SHA256.fullmatch(observed) is None
        or observed != _sha256_bytes(_canonical_bytes(identity))
    ):
        raise SmokeRunError(f"{description} identity is invalid")


def _utc(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _read_sealed_json(path: Path, *, description: str) -> dict[str, Any]:
    lexical = _lexical_absolute(path)
    try:
        metadata = lexical.lstat()
    except OSError as exc:
        raise SmokeRunError(f"{description} is unavailable: {lexical}: {exc}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o222
        or lexical.resolve(strict=True) != lexical
    ):
        raise SmokeRunError(f"{description} is not a canonical sealed file: {lexical}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical, flags)
    try:
        before = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise SmokeRunError(f"{description} changed during its sealed read")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeRunError(f"{description} is invalid JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise SmokeRunError(f"{description} is not canonical JSON")
    return value


def _write_once_sealed(path: Path, value: Mapping[str, Any], *, description: str) -> None:
    lexical = _lexical_absolute(path)
    encoded = _canonical_bytes(value)
    lexical.parent.mkdir(parents=True, exist_ok=True)
    if lexical.exists() or lexical.is_symlink():
        existing = _read_sealed_json(lexical, description=description)
        if existing != dict(value):
            raise SmokeRunError(f"{description} conflicts with its sealed preimage")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{lexical.name}.", suffix=".publishing", dir=lexical.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, lexical)
        except FileExistsError:
            existing = _read_sealed_json(lexical, description=description)
            if existing != dict(value):
                raise SmokeRunError(f"{description} appeared with different bytes")
        _fsync_directory(lexical.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        _fsync_directory(lexical.parent)


def _replace_sealed_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace the fixed CURRENT cache after its attempt is sealed."""

    lexical = _lexical_absolute(path)
    lexical.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{lexical.name}.", suffix=".publishing", dir=lexical.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, lexical)
        _fsync_directory(lexical.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _assert_tree_read_only(root: Path, *, description: str) -> None:
    lexical = _lexical_absolute(root)
    if lexical.is_symlink() or not lexical.is_dir() or lexical.resolve(strict=True) != lexical:
        raise SmokeRunError(f"{description} root is unsafe: {lexical}")
    seen: set[tuple[int, int]] = set()
    for path in (lexical, *sorted(lexical.rglob("*"))):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise SmokeRunError(f"{description} contains a symlink: {path}")
        if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
            raise SmokeRunError(f"{description} contains a special file: {path}")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise SmokeRunError(f"{description} contains a hardlink: {path}")
            inode = (metadata.st_dev, metadata.st_ino)
            if inode in seen:
                raise SmokeRunError(f"{description} aliases an inode: {path}")
            seen.add(inode)
        if stat.S_IMODE(metadata.st_mode) & 0o222:
            raise SmokeRunError(f"{description} remains writable: {path}")


def _seal_tree_read_only(root: Path, *, description: str) -> None:
    lexical = _lexical_absolute(root)
    if lexical.is_symlink() or not lexical.is_dir() or lexical.resolve(strict=True) != lexical:
        raise SmokeRunError(f"{description} root is unsafe: {lexical}")
    paths = sorted(lexical.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise SmokeRunError(f"{description} contains a symlink: {path}")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise SmokeRunError(f"{description} contains a hardlink: {path}")
            path.chmod(stat.S_IMODE(metadata.st_mode) & ~0o222)
        elif stat.S_ISDIR(metadata.st_mode):
            path.chmod(stat.S_IMODE(metadata.st_mode) & ~0o222)
        else:
            raise SmokeRunError(f"{description} contains a special file: {path}")
    lexical.chmod(stat.S_IMODE(lexical.stat().st_mode) & ~0o222)
    _assert_tree_read_only(lexical, description=description)


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    lexical = _lexical_absolute(root)
    records: list[dict[str, Any]] = []
    if not lexical.exists():
        return records
    for path in sorted(lexical.rglob("*")):
        metadata = path.lstat()
        relative = path.relative_to(lexical).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise SmokeRunError(f"smoke attempt contains a symlink: {path}")
        if stat.S_ISREG(metadata.st_mode):
            records.append(
                {
                    "path": relative,
                    "bytes": metadata.st_size,
                    "sha256": _sha256(path),
                }
            )
    return records


def _tree_directories(root: Path) -> list[str]:
    lexical = _lexical_absolute(root)
    if not lexical.exists():
        return []
    directories = ["."]
    directories.extend(
        path.relative_to(lexical).as_posix()
        for path in sorted(lexical.rglob("*"))
        if path.is_dir() and not path.is_symlink()
    )
    return directories


def _attempt_inventory(
    *,
    attempt_root: Path,
    runs_root: Path,
    exclude_attempt_files: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    attempt_files = [
        record
        for record in _tree_inventory(attempt_root)
        if record["path"] not in exclude_attempt_files
    ]
    return {
        "attempt_artifacts": attempt_files,
        "attempt_directories": _tree_directories(attempt_root),
        "run_artifacts": _tree_inventory(runs_root),
        "run_directories": _tree_directories(runs_root),
    }


@contextmanager
def _exclusive_smoke_lock(state_root: Path) -> Iterator[None]:
    """Fence the complete smoke pass so two operators cannot create concurrency two."""

    state_root.mkdir(parents=True, exist_ok=True)
    path = state_root / ".schema5-smoke.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SmokeRunError(
                f"another schema-5 smoke runner owns the concurrency-one lock: {path}"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            (
                f"host={socket.gethostname()} pid={os.getpid()} "
                f"release=run_schema5_smokes_v1\n"
            ).encode("utf-8"),
        )
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _runtime_environment(
    *,
    policy: Any,
    release_git_commit: str,
    protected_capacity_marker_path: str,
    protected_capacity_marker_sha256: str,
    protected_capacity_marker_id: str,
    immutable_pins_sha256: str,
    fleet_contract_path: str,
    fleet_contract_sha256: str,
    release_fleet_contract_sha256: str,
    capacity_generation: int,
    rollout_generation: int,
    runtime_attestation: Mapping[str, Any],
) -> dict[str, str]:
    required_attestation = {"path", "sha256", "lease_path"}
    if not isinstance(runtime_attestation, Mapping) or not required_attestation.issubset(
        runtime_attestation
    ):
        raise SmokeRunError("smoke runtime attestation is incomplete")
    return {
        "ASYS_RELEASE_ID": policy.release.release_id,
        "ASYS_RELEASE_GIT_COMMIT": release_git_commit,
        "ASYS_PROTECTED_CAPACITY_MARKER": protected_capacity_marker_path,
        "ASYS_PROTECTED_CAPACITY_MARKER_SHA256": (
            protected_capacity_marker_sha256
        ),
        "ASYS_PROTECTED_CAPACITY_MARKER_ID": protected_capacity_marker_id,
        "ASYS_MODEL_CONTRACT_SHA256": policy.accepted_model_contract_sha256,
        "ASYS_FLEET_CONTRACT_SHA256": fleet_contract_sha256,
        "ASYS_FLEET_CONTRACT_PATH": fleet_contract_path,
        "ASYS_RELEASE_FLEET_CONTRACT_SHA256": release_fleet_contract_sha256,
        "ASYS_CAPACITY_GENERATION": str(capacity_generation),
        "ASYS_HARNESS_ENVIRONMENT_SHA256": policy.environment.harness_sha256,
        "ASYS_SERVING_ENVIRONMENT_SHA256": policy.environment.serving_sha256,
        "ASYS_ROLLOUT_GENERATION": str(rollout_generation),
        "ASYS_IMMUTABLE_PINS_SHA256": immutable_pins_sha256,
        "ASYS_RUNTIME_ATTESTATION": str(runtime_attestation["path"]),
        "ASYS_RUNTIME_ATTESTATION_SHA256": str(runtime_attestation["sha256"]),
        "ASYS_RUNTIME_INTEGRITY_LEASE": str(runtime_attestation["lease_path"]),
        "ASYS_ARTIFACT_POLICY_SHA256": policy.file_sha256,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def _task(
    *,
    run_root: Path,
    source_index: int,
    server_pool_root: Path,
    release_git_commit: str,
    protected_capacity_marker_path: str,
    protected_capacity_marker_sha256: str,
    protected_capacity_marker_id: str,
    immutable_pins_sha256: str,
    fleet_contract_path: str,
    fleet_contract_sha256: str,
    release_fleet_contract_sha256: str,
    capacity_generation: int,
    rollout_generation: int,
    runtime_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = load_manifest(run_root, verify_frozen=True)
    catalog = VerifiedQuestionCatalog(run_root, snapshot=snapshot)
    policy = load_artifact_policy(run_root, required=True)
    assert policy is not None
    cell = snapshot.cells[source_index]
    profile = serving_profile_for_cell(cell)
    return {
        "run_id": run_root.name,
        "run_root": str(run_root),
        "source_index": source_index,
        "cell_id": cell.cell_id,
        "config_hash": cell.config_hash(),
        "manifest_sha256": snapshot.sha256,
        "benchmark_contracts_sha256": catalog.sidecar_sha256,
        "model_size": cell.model_size,
        "serving_profile": profile.name,
        "fanout_cost": cell.n_agents,
        "server_pool_id": "schema5-v1",
        "server_run_id": str(server_pool_root),
        "server_pool_root": str(server_pool_root),
        "runtime_environment": _runtime_environment(
            policy=policy,
            release_git_commit=release_git_commit,
            protected_capacity_marker_path=protected_capacity_marker_path,
            protected_capacity_marker_sha256=protected_capacity_marker_sha256,
            protected_capacity_marker_id=protected_capacity_marker_id,
            immutable_pins_sha256=immutable_pins_sha256,
            fleet_contract_path=fleet_contract_path,
            fleet_contract_sha256=fleet_contract_sha256,
            release_fleet_contract_sha256=release_fleet_contract_sha256,
            capacity_generation=capacity_generation,
            rollout_generation=rollout_generation,
            runtime_attestation=runtime_attestation,
        ),
    }


def _sanitized_environment(*, hf_home: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("ASYS_"):
            environment.pop(key, None)
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(hf_home),
        }
    )
    return environment


def _load_immutable_context(
    *,
    state_root: Path,
    results_root: Path,
    server_pool_root: Path,
    release_worktree: Path,
    harness_prefix: Path,
    immutable_pins_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and fully verify the control-pinned release used by every smoke cell."""

    try:
        control = schema5_control.load_control(state_root, verify_files=True)
    except schema5_control.ControlError as exc:
        raise SmokeRunError(f"schema-5 immutable control verification failed: {exc}") from exc
    if control.get("immutable_sha256") != immutable_pins_sha256:
        raise SmokeRunError(
            "smoke immutable_pins_sha256 differs from the verified control state"
        )
    immutable = dict(control["immutable"])
    release_fleet_contract_sha256 = str(immutable["fleet_contract_sha256"])
    fleet_binding = schema5_control.effective_fleet_contract_binding(
        control, verify_files=True
    )
    immutable.update(
        {
            "fleet_contract_path": fleet_binding["path"],
            "fleet_contract_sha256": fleet_binding["sha256"],
            "release_fleet_contract_sha256": release_fleet_contract_sha256,
            "capacity_generation": int(fleet_binding["capacity_generation"]),
        }
    )
    expected_paths = {
        "results_root": results_root,
        "server_pool_root": server_pool_root,
        "release_worktree": release_worktree,
        "harness_environment_prefix": harness_prefix,
    }
    mismatches: dict[str, tuple[str, str]] = {}
    for field, observed in expected_paths.items():
        expected = Path(str(immutable[field])).expanduser().resolve()
        if expected != observed:
            mismatches[field] = (str(expected), str(observed))
    if mismatches:
        raise SmokeRunError(
            "smoke execution paths differ from immutable control pins: "
            + "; ".join(
                f"{field} expected {expected}, observed {observed}"
                for field, (expected, observed) in sorted(mismatches.items())
            )
        )
    for field in (
        "source_tree_sha256",
        "model_contract_sha256",
        "fleet_contract_sha256",
        "release_fleet_contract_sha256",
        "harness_environment_sha256",
        "serving_environment_sha256",
    ):
        if _SHA256.fullmatch(str(immutable.get(field, ""))) is None:
            raise SmokeRunError(f"immutable control {field} is not a lowercase SHA-256")
    if (
        not isinstance(immutable.get("capacity_generation"), int)
        or isinstance(immutable["capacity_generation"], bool)
        or int(immutable["capacity_generation"]) < 1
    ):
        raise SmokeRunError("effective capacity generation must be a positive integer")
    return control, immutable


def _validate_preproduction_rollout_generation(
    control: Mapping[str, Any], rollout_generation: int
) -> None:
    """Bind smoke provenance to the first generation after a paused control.

    Smoke cells exercise the exact generation that the next successful ``resume`` will
    publish, while admission and both production controllers are still disabled.  A fresh
    control therefore records smoke rows at generation one even though its durable current
    generation is zero.  Refusing every other desired state and generation prevents an
    operator from blessing rows produced during (or after) a live production rollout.
    """

    desired_state = control.get("desired_state")
    if desired_state != "paused":
        raise SmokeRunError(
            "schema-5 smoke execution requires desired_state=paused; "
            f"observed {desired_state!r}"
        )
    if control.get("drain_requested") is not False:
        try:
            schema5_control.validate_capacity_readiness_pending_state(
                control,
                require_fleet_gate=True,
                # _load_immutable_context already loaded this control with full
                # immutable/effective-fleet verification.
                verify_files=False,
            )
        except schema5_control.ControlError as exc:
            raise SmokeRunError(
                "schema-5 smoke execution is forbidden while draining unless "
                f"capacity readiness is exact: {exc}"
            ) from exc
    current_generation = control.get("rollout_generation")
    if (
        not isinstance(current_generation, int)
        or isinstance(current_generation, bool)
        or current_generation < 0
    ):
        raise SmokeRunError("verified control has an invalid rollout_generation")
    expected_generation = current_generation + 1
    if rollout_generation != expected_generation:
        raise SmokeRunError(
            "smoke rollout_generation must equal paused control rollout_generation + 1: "
            f"expected {expected_generation}, observed {rollout_generation}"
        )


def _trusted_readiness_generation(
    *,
    state_root: Path,
    control: Mapping[str, Any],
    server_pool_root: Path,
    rollout_generation: int,
) -> tuple[dict[str, Any], TrustedGenerationCatalog]:
    """Bind the attempt to the exact endpoint catalog consumed by resume."""

    try:
        catalog = schema5_control.load_trusted_generation_catalog(
            state_root,
            server_pool_root=server_pool_root,
        )
        binding = schema5_control._validate_resume_catalog_authority(  # noqa: SLF001
            state_root,
            control,
            target_rollout_generation=rollout_generation,
            catalog=catalog,
        )
    except (
        OSError,
        ValueError,
        schema5_control.ControlError,
        schema5_control.GenerationCatalogError,
    ) as exc:
        raise SmokeRunError(
            f"smoke cannot prove its trusted serving generation: {exc}"
        ) from exc
    required = {
        "catalog_id",
        "marker_path",
        "marker_sha256",
        "inventory_sha256",
        "catalog_payload_sha256",
        "allowed_generation_tuple_count",
        "release_fleet_contract_sha256",
        "fleet_contract_sha256",
        "capacity_generation",
        "rollout_generation",
    }
    if set(binding) != required:
        raise SmokeRunError("trusted smoke generation binding fields drifted")
    value = dict(binding)
    if (
        value.get("rollout_generation") != rollout_generation
        or any(
            not isinstance(value.get(field), int)
            or isinstance(value[field], bool)
            or int(value[field]) < 1
            for field in (
                "allowed_generation_tuple_count",
                "capacity_generation",
                "rollout_generation",
            )
        )
        or not isinstance(value.get("marker_path"), str)
        or not Path(str(value["marker_path"])).is_absolute()
        or any(
            _SHA256.fullmatch(str(value.get(field, ""))) is None
            for field in (
                "catalog_id",
                "marker_sha256",
                "inventory_sha256",
                "catalog_payload_sha256",
                "release_fleet_contract_sha256",
                "fleet_contract_sha256",
            )
        )
    ):
        raise SmokeRunError("trusted smoke generation binding is malformed")
    if (
        catalog.catalog_id != value["catalog_id"]
        or catalog.marker_sha256 != value["marker_sha256"]
        or catalog.inventory_sha256 != value["inventory_sha256"]
        or catalog.catalog_sha256 != value["catalog_payload_sha256"]
        or len(catalog.allowed_generation_tuples)
        != value["allowed_generation_tuple_count"]
        or catalog.marker_path != Path(str(value["marker_path"])).resolve()
    ):
        raise SmokeRunError(
            "trusted smoke generation binding differs from its validated catalog"
        )
    return value, catalog


def _attempt_base(state_root: Path) -> Path:
    return state_root / "readiness" / SMOKE_ATTEMPT_BASE_NAME


def _pointer_reference(path: Path, pointer: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "pointer_id": str(pointer["pointer_id"]),
        "attempt_id": str(pointer["attempt_id"]),
    }


def _attempt_provenance(
    *,
    immutable_sha256: str,
    immutable: Mapping[str, Any],
    readiness_generation: Mapping[str, Any],
    rollout_generation: int,
) -> dict[str, Any]:
    value = {
        "immutable_sha256": immutable_sha256,
        "release_id": immutable["release_id"],
        "git_commit": immutable["git_commit"],
        "source_tree_sha256": immutable["source_tree_sha256"],
        "harness_environment_sha256": immutable[
            "harness_environment_sha256"
        ],
        "serving_environment_sha256": immutable[
            "serving_environment_sha256"
        ],
        "model_contract_sha256": immutable["model_contract_sha256"],
        "release_fleet_contract_sha256": immutable[
            "release_fleet_contract_sha256"
        ],
        "fleet_contract_path": immutable["fleet_contract_path"],
        "fleet_contract_sha256": immutable["fleet_contract_sha256"],
        "capacity_generation": immutable["capacity_generation"],
        "rollout_generation": rollout_generation,
        "protected_capacity_marker_path": immutable[
            "protected_capacity_marker_path"
        ],
        "protected_capacity_marker_sha256": immutable[
            "protected_capacity_marker_sha256"
        ],
        "protected_capacity_marker_id": immutable[
            "protected_capacity_marker_id"
        ],
        "trusted_generation": dict(readiness_generation),
    }
    if (
        value["fleet_contract_sha256"]
        != readiness_generation["fleet_contract_sha256"]
        or value["release_fleet_contract_sha256"]
        != readiness_generation["release_fleet_contract_sha256"]
        or value["capacity_generation"]
        != readiness_generation["capacity_generation"]
        or value["rollout_generation"]
        != readiness_generation["rollout_generation"]
    ):
        raise SmokeRunError(
            "smoke immutable fleet and trusted endpoint generation disagree"
        )
    return value


def _attempt_id(
    *,
    ordinal: int,
    readiness_generation: Mapping[str, Any],
) -> str:
    return (
        f"a{ordinal:06d}-"
        f"g{int(readiness_generation['rollout_generation']):06d}-"
        f"c{int(readiness_generation['capacity_generation']):06d}-"
        f"{str(readiness_generation['catalog_id'])[:16]}"
    )


def _validate_attempt_pointer(
    value: Mapping[str, Any],
    *,
    base: Path,
    results_root: Path,
    path: Path,
    expected_ordinal: int,
    predecessor: Mapping[str, str] | None,
) -> dict[str, Any]:
    if set(value) != _ATTEMPT_POINTER_FIELDS:
        raise SmokeRunError("smoke attempt-pointer fields drifted")
    _verify_identity(value, "pointer_id", description="smoke attempt pointer")
    readiness = value.get("readiness_generation")
    if not isinstance(readiness, Mapping):
        raise SmokeRunError("smoke attempt pointer lacks readiness generation")
    expected_id = _attempt_id(
        ordinal=expected_ordinal, readiness_generation=readiness
    )
    expected_root = base / ATTEMPTS_DIRECTORY / expected_id
    expected_runs = (
        results_root / SMOKE_ATTEMPT_RUNS_NAME / expected_id
    )
    expected_path = (
        base
        / ATTEMPT_POINTERS_DIRECTORY
        / f"{expected_ordinal:06d}-{expected_id}.json"
    )
    timestamp = value.get("created_timestamp")
    protected = value.get("protected_capacity")
    if (
        value.get("schema_version") != 1
        or value.get("protocol") != ATTEMPT_POINTER_PROTOCOL
        or value.get("attempt_id") != expected_id
        or value.get("attempt_ordinal") != expected_ordinal
        or path != expected_path
        or value.get("attempt_root") != str(expected_root)
        or value.get("runs_root") != str(expected_runs)
        or value.get("execution_root") != str(expected_root / "execution")
        or _SHA256.fullmatch(str(value.get("immutable_sha256", ""))) is None
        or value.get("predecessor") != predecessor
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or float(timestamp) <= 0
        or value.get("created_at") != _utc(float(timestamp))
        or not isinstance(protected, Mapping)
        or set(protected)
        != {"path", "sha256", "marker_id"}
        or any(
            _SHA256.fullmatch(str(protected.get(field, ""))) is None
            for field in ("sha256", "marker_id")
        )
        or not isinstance(protected.get("path"), str)
        or not Path(str(protected["path"])).is_absolute()
    ):
        raise SmokeRunError("smoke attempt-pointer identity is invalid")
    # Reuse the strict generation validator independent of mutable control.
    generation = dict(readiness)
    if (
        set(generation)
        != {
            "catalog_id",
            "marker_path",
            "marker_sha256",
            "inventory_sha256",
            "catalog_payload_sha256",
            "allowed_generation_tuple_count",
            "release_fleet_contract_sha256",
            "fleet_contract_sha256",
            "capacity_generation",
            "rollout_generation",
        }
        or any(
            _SHA256.fullmatch(str(generation.get(field, ""))) is None
            for field in (
                "catalog_id",
                "marker_sha256",
                "inventory_sha256",
                "catalog_payload_sha256",
                "release_fleet_contract_sha256",
                "fleet_contract_sha256",
            )
        )
    ):
        raise SmokeRunError("smoke attempt readiness generation is invalid")
    return dict(value)


def _attempt_terminal_kind(pointer: Mapping[str, Any]) -> str | None:
    root = Path(str(pointer["attempt_root"]))
    complete = root / ATTEMPT_COMPLETE_NAME
    failure = root / ATTEMPT_FAILURE_NAME
    if complete.exists() and failure.exists():
        raise SmokeRunError("smoke attempt has conflicting terminal markers")
    if complete.exists():
        return "complete"
    if failure.exists():
        return "failure"
    return None


def _load_attempt_failure(
    pointer_path: Path, pointer: Mapping[str, Any]
) -> dict[str, Any]:
    failure = _read_sealed_json(
        Path(str(pointer["attempt_root"])) / ATTEMPT_FAILURE_NAME,
        description="smoke attempt failure",
    )
    _verify_identity(
        failure, "failure_id", description="smoke attempt failure"
    )
    inventory = failure.get("inventory")
    if (
        set(failure)
        != {
            "schema_version",
            "protocol",
            "passed",
            "attempt",
            "reason",
            "retryable",
            "inventory",
            "failure_id",
        }
        or failure.get("schema_version") != 1
        or failure.get("protocol") != ATTEMPT_FAILURE_PROTOCOL
        or failure.get("passed") is not False
        or failure.get("attempt") != _pointer_reference(pointer_path, pointer)
        or not isinstance(failure.get("reason"), str)
        or not failure["reason"]
        or not isinstance(failure.get("retryable"), bool)
        or not isinstance(inventory, Mapping)
        or set(inventory)
        != {
            "attempt_artifacts",
            "attempt_directories",
            "run_artifacts",
            "run_directories",
        }
        or any(
            not isinstance(inventory.get(field), list)
            for field in (
                "attempt_artifacts",
                "attempt_directories",
                "run_artifacts",
                "run_directories",
            )
        )
    ):
        raise SmokeRunError("smoke attempt failure contract is invalid")
    expected_inventory = _attempt_inventory(
        attempt_root=Path(str(pointer["attempt_root"])),
        runs_root=Path(str(pointer["runs_root"])),
        exclude_attempt_files=frozenset({ATTEMPT_FAILURE_NAME}),
    )
    if dict(inventory) != expected_inventory:
        raise SmokeRunError("smoke attempt failure inventory drifted")
    return failure


def _load_attempt_pointers(
    *,
    base: Path,
    results_root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    root = base / ATTEMPT_POINTERS_DIRECTORY
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise SmokeRunError("smoke attempt-pointer directory is unsafe")
    paths = sorted(root.iterdir())
    if any(path.suffix != ".json" for path in paths):
        raise SmokeRunError("smoke attempt-pointer directory contains unexpected files")
    loaded: list[tuple[Path, dict[str, Any]]] = []
    predecessor: dict[str, str] | None = None
    prior_pointer: Mapping[str, Any] | None = None
    for ordinal, path in enumerate(paths, start=1):
        pointer = _validate_attempt_pointer(
            _read_sealed_json(
                path, description=f"smoke attempt pointer {ordinal}"
            ),
            base=base,
            results_root=results_root,
            path=path,
            expected_ordinal=ordinal,
            predecessor=predecessor,
        )
        if prior_pointer is not None:
            prior_generation = prior_pointer["readiness_generation"]
            generation = pointer["readiness_generation"]
            same_generation = generation == prior_generation
            if same_generation:
                failure = _load_attempt_failure(
                    loaded[-1][0], prior_pointer
                )
                if failure.get("retryable") is not True:
                    raise SmokeRunError(
                        "same-generation smoke retry follows a non-retryable attempt"
                    )
            elif (
                int(generation["rollout_generation"])
                <= int(prior_generation["rollout_generation"])
                or int(generation["capacity_generation"])
                <= int(prior_generation["capacity_generation"])
                or generation["catalog_id"] == prior_generation["catalog_id"]
                or generation["fleet_contract_sha256"]
                == prior_generation["fleet_contract_sha256"]
            ):
                raise SmokeRunError(
                    "smoke capacity-generation attempts are not strictly increasing"
                )
        loaded.append((path, pointer))
        predecessor = _pointer_reference(path, pointer)
        prior_pointer = pointer
    for _, pointer in loaded[:-1]:
        if _attempt_terminal_kind(pointer) is None:
            raise SmokeRunError(
                "a superseded smoke attempt lacks a terminal marker"
            )
        _assert_tree_read_only(
            Path(str(pointer["attempt_root"])),
            description="superseded smoke attempt evidence",
        )
        _assert_tree_read_only(
            Path(str(pointer["runs_root"])),
            description="superseded smoke attempt runs",
        )
    return loaded


def _attempt_binding(pointer: Mapping[str, Any]) -> dict[str, Any]:
    generation = pointer["readiness_generation"]
    return {
        "protocol": ATTEMPT_BINDING_PROTOCOL,
        "attempt_id": pointer["attempt_id"],
        "attempt_ordinal": pointer["attempt_ordinal"],
        "immutable_sha256": pointer["immutable_sha256"],
        "capacity_generation": generation["capacity_generation"],
        "rollout_generation": generation["rollout_generation"],
        "fleet_contract_sha256": generation["fleet_contract_sha256"],
        "release_fleet_contract_sha256": generation[
            "release_fleet_contract_sha256"
        ],
        "trusted_catalog_id": generation["catalog_id"],
    }


def _publish_failure(
    pointer_path: Path,
    pointer: Mapping[str, Any],
    *,
    reason: str,
    retryable: bool,
) -> dict[str, Any]:
    attempt_root = Path(str(pointer["attempt_root"]))
    runs_root = Path(str(pointer["runs_root"]))
    for description, root in (
        ("smoke attempt", attempt_root),
        ("smoke attempt runs", runs_root),
    ):
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise SmokeRunError(f"{description} root is unsafe: {root}")
        root.mkdir(parents=True, exist_ok=True)
    inventory = _attempt_inventory(
        attempt_root=attempt_root,
        runs_root=runs_root,
    )
    payload = _with_identity(
        {
            "schema_version": 1,
            "protocol": ATTEMPT_FAILURE_PROTOCOL,
            "passed": False,
            "attempt": _pointer_reference(pointer_path, pointer),
            "reason": reason,
            "retryable": retryable,
            "inventory": inventory,
        },
        "failure_id",
    )
    _write_once_sealed(
        attempt_root / ATTEMPT_FAILURE_NAME,
        payload,
        description="smoke attempt failure",
    )
    _seal_tree_read_only(runs_root, description="failed smoke attempt runs")
    _seal_tree_read_only(attempt_root, description="failed smoke attempt evidence")
    return payload


def _create_attempt(
    *,
    base: Path,
    results_root: Path,
    release_worktree: Path,
    immutable_sha256: str,
    immutable: Mapping[str, Any],
    readiness_generation: Mapping[str, Any],
    now: float,
) -> tuple[Path, dict[str, Any]]:
    known = _load_attempt_pointers(base=base, results_root=results_root)
    if known:
        latest_path, latest = known[-1]
        terminal_kind = _attempt_terminal_kind(latest)
        if terminal_kind is None:
            # A previous allocation died before it could publish a terminal marker.
            # Seal every retained row/log, then start from a fresh empty run root.
            _publish_failure(
                latest_path,
                latest,
                reason="interrupted_before_terminal_publication",
                retryable=True,
            )
        else:
            if terminal_kind == "failure":
                _load_attempt_failure(latest_path, latest)
            else:
                _validate_completion(
                    pointer_path=latest_path,
                    pointer=latest,
                    require_sealed=False,
                )
            _seal_tree_read_only(
                Path(str(latest["runs_root"])),
                description="terminal smoke attempt runs",
            )
            _seal_tree_read_only(
                Path(str(latest["attempt_root"])),
                description="terminal smoke attempt evidence",
            )
            _assert_tree_read_only(
                Path(str(latest["attempt_root"])),
                description="terminal smoke attempt evidence",
            )
            _assert_tree_read_only(
                Path(str(latest["runs_root"])),
                description="terminal smoke attempt runs",
            )
        latest_generation = latest["readiness_generation"]
        if dict(latest_generation) == dict(readiness_generation):
            latest_failure_path = (
                Path(str(latest["attempt_root"])) / ATTEMPT_FAILURE_NAME
            )
            if not latest_failure_path.is_file():
                raise SmokeRunError(
                    "a successful smoke generation cannot be redrawn"
                )
            latest_failure = _load_attempt_failure(latest_path, latest)
            if latest_failure.get("retryable") is not True:
                raise SmokeRunError(
                    "non-retryable smoke failure requires a superseding release "
                    "or new capacity generation"
                )
        elif (
            int(readiness_generation["rollout_generation"])
            <= int(latest_generation["rollout_generation"])
            or int(readiness_generation["capacity_generation"])
            <= int(latest_generation["capacity_generation"])
            or readiness_generation["catalog_id"]
            == latest_generation["catalog_id"]
            or readiness_generation["fleet_contract_sha256"]
            == latest_generation["fleet_contract_sha256"]
        ):
            raise SmokeRunError(
                "new smoke capacity generation does not strictly supersede "
                "its predecessor"
            )
        predecessor = _pointer_reference(latest_path, latest)
    else:
        predecessor = None
    ordinal = len(known) + 1
    attempt_id = _attempt_id(
        ordinal=ordinal, readiness_generation=readiness_generation
    )
    attempt_root = base / ATTEMPTS_DIRECTORY / attempt_id
    runs_root = results_root / SMOKE_ATTEMPT_RUNS_NAME / attempt_id
    pointer_path = (
        base
        / ATTEMPT_POINTERS_DIRECTORY
        / f"{ordinal:06d}-{attempt_id}.json"
    )
    if (
        attempt_root.exists()
        or attempt_root.is_symlink()
        or runs_root.exists()
        or runs_root.is_symlink()
    ):
        raise SmokeRunError("smoke attempt artifacts predate their marker-first pointer")
    protected = {
        "path": str(immutable["protected_capacity_marker_path"]),
        "sha256": str(immutable["protected_capacity_marker_sha256"]),
        "marker_id": str(immutable["protected_capacity_marker_id"]),
    }
    pointer = _with_identity(
        {
            "schema_version": 1,
            "protocol": ATTEMPT_POINTER_PROTOCOL,
            "attempt_id": attempt_id,
            "attempt_ordinal": ordinal,
            "attempt_root": str(attempt_root),
            "runs_root": str(runs_root),
            "execution_root": str(attempt_root / "execution"),
            "immutable_sha256": immutable_sha256,
            "readiness_generation": dict(readiness_generation),
            "protected_capacity": protected,
            "predecessor": predecessor,
            "created_at": _utc(now),
            "created_timestamp": now,
        },
        "pointer_id",
    )
    _write_once_sealed(
        pointer_path, pointer, description="smoke attempt pointer"
    )
    attempt_root.mkdir(parents=True)
    runs_root.mkdir(parents=True)
    try:
        initialize_all(
            results_root=runs_root,
            release_worktree=release_worktree,
            model_contract_path=release_worktree
            / "configs"
            / "model_contracts.v1.json",
            release_id=str(immutable["release_id"]),
            git_commit=str(immutable["git_commit"]),
            source_tree_sha256=str(immutable["source_tree_sha256"]),
            harness_environment_sha256=str(
                immutable["harness_environment_sha256"]
            ),
            serving_environment_sha256=str(
                immutable["serving_environment_sha256"]
            ),
            apply=True,
            attempt_binding=_attempt_binding(pointer),
        )
    except Exception as exc:
        _publish_failure(
            pointer_path,
            pointer,
            reason=f"attempt_initialization_failed:{type(exc).__name__}:{exc}",
            retryable=False,
        )
        raise
    return pointer_path, pointer


def _read_meta(run_root: Path, cell_id: str) -> Mapping[str, Any] | None:
    path = run_root / "cells" / cell_id / "meta.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        # The completion validator supplies the authoritative diagnostic.  Returning
        # ``None`` here prevents this secondary provenance census from masking it.
        return None
    return value if isinstance(value, dict) else None


def _provenance_errors(
    *,
    run_root: Path,
    cell: Any,
    status: Any,
    rows: Sequence[Mapping[str, Any]],
    immutable: Mapping[str, Any],
    policy: Any,
    rollout_generation: int,
) -> tuple[str, ...]:
    """Return explicit immutable-lineage violations in retained smoke artifacts."""

    errors = [
        str(error)
        for error in status.errors
        if any(marker in str(error).lower() for marker in _PROVENANCE_ERROR_MARKERS)
    ]
    expected_profile = serving_profile_for_cell(cell).name
    row_fixed = {
        "schema_version": 5,
        "release_id": immutable["release_id"],
        "environment_hash": immutable["harness_environment_sha256"],
        "model_contract_sha256": immutable["model_contract_sha256"],
        "fleet_contract_sha256": immutable["fleet_contract_sha256"],
        "release_fleet_contract_sha256": immutable[
            "release_fleet_contract_sha256"
        ],
        "capacity_generation": immutable["capacity_generation"],
        "serving_profile": expected_profile,
        "rollout_generation": rollout_generation,
    }
    for index, row in enumerate(rows):
        for field, expected in row_fixed.items():
            if row.get(field) != expected:
                errors.append(
                    f"record {index} {field} expected {expected!r}, "
                    f"observed {row.get(field)!r}"
                )
        if not isinstance(row.get("endpoint_generation"), str) or not row.get(
            "endpoint_generation"
        ):
            errors.append(f"record {index} lacks an exact endpoint_generation")

    meta = _read_meta(run_root, cell.cell_id)
    if meta is not None:
        meta_fixed = {
            "schema_version": 5,
            "release_id": immutable["release_id"],
            "environment_hash": immutable["harness_environment_sha256"],
            "serving_environment_hash": immutable["serving_environment_sha256"],
            "model_contract_sha256": immutable["model_contract_sha256"],
            "fleet_contract_sha256": immutable["fleet_contract_sha256"],
            "release_fleet_contract_sha256": immutable[
                "release_fleet_contract_sha256"
            ],
            "capacity_generation": immutable["capacity_generation"],
            "artifact_policy_sha256": policy.file_sha256,
            "git_commit": immutable["git_commit"],
            "serving_profile": expected_profile,
            "rollout_generation": rollout_generation,
        }
        for field, expected in meta_fixed.items():
            if meta.get(field) != expected:
                errors.append(
                    f"meta {field} expected {expected!r}, observed {meta.get(field)!r}"
                )
    elif status.status is CompletionState.COMPLETE:
        errors.append("complete cell lacks readable schema-5 meta.json")
    return tuple(dict.fromkeys(errors))


def _incident_accounting(rows: Sequence[Mapping[str, Any]], status: Any) -> dict[str, int]:
    """Count retained censors and unresolved current failure-ledger incidents."""

    accounting = _censor_accounting(list(rows))
    failure = status.failure
    classification = None if failure is None else failure.classification
    failure_type = (
        None if failure is None else str(failure.last_error.get("type", ""))
    )
    status_errors = tuple(str(error).lower() for error in status.errors)
    context_failure = int(
        classification == "context_capacity"
        or failure_type in {"ContextCapacityError", "GenerationTruncationError"}
    )
    protocol_failure = int(
        failure_type in _PROTOCOL_FAILURE_TYPES
        or any("protocol" in error for error in status_errors)
    )
    truncation_failure = int(
        failure_type in _TRUNCATION_FAILURE_TYPES
        or any("truncat" in error for error in status_errors)
    )
    top_length = int(accounting["n_length_censored_questions"])
    top_protocol = int(accounting["n_protocol_censored_questions"])
    top_transport = int(accounting["n_transport_censored_questions"])
    auxiliary_length = int(accounting["n_auxiliary_length_censors"])
    auxiliary_protocol = int(accounting["n_auxiliary_protocol_censors"])
    auxiliary_transport = int(accounting["n_auxiliary_transport_censors"])
    transport_coordinates = int(
        accounting["n_transport_censored_coordinates"]
    )
    transport_affected = int(accounting["n_transport_affected_questions"])
    return {
        "top_level_length_censored_qids": top_length,
        "top_level_protocol_censored_qids": top_protocol,
        "top_level_transport_censored_qids": top_transport,
        "transport_affected_qids": transport_affected,
        "transport_censored_coordinates": transport_coordinates,
        "auxiliary_length_censored_draws": auxiliary_length,
        "auxiliary_protocol_censored_draws": auxiliary_protocol,
        "auxiliary_transport_censored_draws": auxiliary_transport,
        "context_incidents": context_failure,
        "protocol_incidents": top_protocol + auxiliary_protocol + protocol_failure,
        "transport_incidents": transport_coordinates,
        "truncation_incidents": top_length + auxiliary_length + truncation_failure,
    }


def _runnable(status: Any) -> bool:
    if status.status not in _RUNNABLE_STATES:
        return False
    return status.status is not CompletionState.RETRYABLE or bool(
        status.eligible_for_retry
    )


def _run_task(
    task: Mapping[str, Any],
    *,
    state_root: Path,
    execution_root: Path | None = None,
    release_worktree: Path,
    harness_prefix: Path,
    hf_home: Path,
    runtime_attestation: Mapping[str, Any],
    immutable: Mapping[str, Any],
) -> dict[str, Any]:
    cell_id = str(task["cell_id"])
    output_root = state_root if execution_root is None else execution_root
    batch_path = output_root / "batches" / str(task["run_id"]) / f"{cell_id}.json"
    log_path = output_root / "logs" / str(task["run_id"]) / f"{cell_id}.json"
    _atomic_json(batch_path, {"schema_version": 1, "tasks": [dict(task)]})
    python = (harness_prefix / "bin" / "python").resolve()
    dispatcher = (release_worktree / "slurm" / "dispatch_sweeps.py").resolve()
    command = [
        str(python),
        "-u",
        str(dispatcher),
        "run-task",
        "--batch-manifest",
        str(batch_path),
        "--index",
        "0",
        "--expected-release-root",
        str(release_worktree),
        "--expected-harness-prefix",
        str(harness_prefix),
    ]

    def refresh_lease() -> None:
        try:
            runtime_integrity.refresh_generation_lease(
                state_dir=state_root,
                attestation_path=Path(str(runtime_attestation["path"])),
                attestation_sha256=str(runtime_attestation["sha256"]),
                generation=int(runtime_attestation["generation"]),
                release_id=str(immutable["release_id"]),
                immutable_pins_sha256=str(
                    task["runtime_environment"]["ASYS_IMMUTABLE_PINS_SHA256"]
                ),
                expected_environment_hashes={
                    role: str(immutable[f"{role}_environment_sha256"])
                    for role in ("harness", "serving")
                },
                expected_prefixes={
                    role: str(immutable[f"{role}_environment_prefix"])
                    for role in ("harness", "serving")
                },
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            runtime_integrity.RuntimeIntegrityError,
        ) as exc:
            raise SmokeRunError(
                f"smoke runtime integrity lease refresh failed: {exc}"
            ) from exc

    # Production controllers refresh this lease every heartbeat.  They intentionally
    # do not exist yet during paused, estimand-excluded smokes, so this parent owns the
    # identical refresh operation while its one worker is active.  Polling once per
    # minute leaves ample margin around the five-minute scan / seven-minute TTL.
    refresh_lease()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_sanitized_environment(hf_home=hf_home),
    )
    lease_error: SmokeRunError | None = None
    while True:
        try:
            stdout, stderr = process.communicate(timeout=60.0)
            break
        except subprocess.TimeoutExpired:
            try:
                refresh_lease()
            except SmokeRunError as exc:
                lease_error = exc
                # The worker's existing signal handler finishes its current bounded
                # request and refuses every later coordinate before exiting cleanly.
                if process.poll() is None:
                    try:
                        process.send_signal(signal.SIGUSR1)
                    except ProcessLookupError:
                        pass
                stdout, stderr = process.communicate()
                break
    report = {
        "cell_id": cell_id,
        "command": command,
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "batch_manifest": str(batch_path),
        "batch_manifest_sha256": _sha256(batch_path),
    }
    _atomic_json(log_path, report)
    if lease_error is not None:
        raise SmokeRunError(f"{lease_error}; log={log_path}") from lease_error
    if process.returncode != 0:
        raise SmokeRunError(
            f"smoke worker failed for {cell_id} rc={process.returncode}; log={log_path}"
        )
    return report


def _status(
    run_root: Path,
    source_index: int,
    *,
    model_contract_path: Path,
    trusted_generation_tuples: frozenset[
        tuple[str, str, int, int, str]
    ],
) -> tuple[Any, list[dict[str, Any]]]:
    snapshot = load_manifest(run_root, verify_frozen=True)
    catalog = VerifiedQuestionCatalog(run_root, snapshot=snapshot)
    cell = snapshot.cells[source_index]
    questions = catalog.questions_for(cell)
    status = get_completion_status(
        cell,
        run_root / "cells" / cell.cell_id,
        expected_qids=tuple(question.qid for question in questions),
        expected_questions=questions,
        verified_benchmark_contracts=catalog.frozen,
        verified_manifest=catalog.snapshot,
        check_active=True,
        model_contract_path=model_contract_path,
        trusted_generation_tuples=trusted_generation_tuples,
    )
    rows: list[dict[str, Any]] = []
    if status.valid_count:
        rows = list(
            read_canonical_results(
                cell,
                run_root / "cells" / cell.cell_id,
                expected_qids=tuple(question.qid for question in questions),
                expected_questions=questions,
                verified_benchmark_contracts=catalog.frozen,
                verified_manifest=catalog.snapshot,
            ).records
        )
    return status, rows


def _write_json_with_checksum(path: Path, payload: Mapping[str, Any]) -> str:
    _atomic_json(path, payload)
    digest = _sha256(path)
    _atomic_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
    return digest


def _write_smoke_evidence(
    *,
    state_root: Path,
    immutable_sha256: str,
    suites: Sequence[Mapping[str, Any]],
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Publish the controller's exact typed smoke readiness envelope."""

    required_scalar_fields = {
        "schema5_complete_cells",
        "context_incidents",
        "protocol_incidents",
        "transport_incidents",
        "truncation_incidents",
        "provenance_failures",
        "trusted_generation_failures",
    }
    for index, suite in enumerate(suites):
        if not isinstance(suite, Mapping):
            raise SmokeRunError(
                f"smoke suite {index} is not a semantic report object"
            )
        missing = (
            {"kind", "run_id", "passed"} | required_scalar_fields
        ) - set(suite)
        if missing:
            raise SmokeRunError(
                f"smoke suite {index} is missing required semantic fields: "
                f"{sorted(missing)}"
            )
        if (
            not isinstance(suite["kind"], str)
            or not suite["kind"]
            or not isinstance(suite["run_id"], str)
            or not suite["run_id"]
            or not isinstance(suite["passed"], bool)
            or any(
                not isinstance(suite[field], int)
                or isinstance(suite[field], bool)
                or int(suite[field]) < 0
                for field in required_scalar_fields
            )
        ):
            raise SmokeRunError(
                f"smoke suite {index} has malformed required semantic fields"
            )
    observed_kinds = [str(suite["kind"]) for suite in suites]
    observed_runs = [str(suite["run_id"]) for suite in suites]
    if (
        len(suites) != 3
        or len(set(observed_kinds)) != 3
        or len(set(observed_runs)) != 3
        or set(observed_kinds) != set(_SUITE_ARTIFACT_NAMES.values())
        or set(observed_runs) != set(_SUITE_ARTIFACT_NAMES)
    ):
        raise SmokeRunError("smoke suite artifacts do not cover the exact three-suite contract")
    publication_root = (
        state_root / "readiness"
        if output_root is None
        else output_root
    )
    artifact_directory = publication_root / "artifacts"
    references: list[dict[str, str]] = []
    by_kind = {str(suite["kind"]): suite for suite in suites}
    for kind in (
        "long_32b_smoke",
        "selective_long_smoke",
        "standard_canary_smoke",
    ):
        path = (artifact_directory / f"{kind}.json").resolve()
        digest = _write_json_with_checksum(path, by_kind[kind])
        references.append({"name": kind, "path": str(path), "sha256": digest})

    by_run = {str(suite["run_id"]): suite for suite in suites}
    metrics = {
        "long_32b_cells": int(
            by_run["schema5_smoke_32b_long_v1"]["schema5_complete_cells"]
        ),
        "selective_long_cells": int(
            by_run["schema5_smoke_selective_long_v1"]["schema5_complete_cells"]
        ),
        "standard_canary_cells": int(
            by_run["schema5_smoke_standard_canaries_v1"]["schema5_complete_cells"]
        ),
        "schema5_complete_cells": sum(
            int(suite["schema5_complete_cells"]) for suite in suites
        ),
        "context_incidents": sum(int(suite["context_incidents"]) for suite in suites),
        "protocol_incidents": sum(
            int(suite["protocol_incidents"]) for suite in suites
        ),
        "transport_incidents": sum(
            int(suite["transport_incidents"]) for suite in suites
        ),
        "truncation_incidents": sum(
            int(suite["truncation_incidents"]) for suite in suites
        ),
        "provenance_failures": sum(
            int(suite["provenance_failures"]) for suite in suites
        ),
        "trusted_generation_failures": sum(
            int(suite["trusted_generation_failures"]) for suite in suites
        ),
    }
    expected = {
        "long_32b_cells": 15,
        "selective_long_cells": 20,
        "standard_canary_cells": 6,
        "schema5_complete_cells": 41,
        "context_incidents": 0,
        "protocol_incidents": 0,
        "transport_incidents": 0,
        "truncation_incidents": 0,
        "provenance_failures": 0,
        "trusted_generation_failures": 0,
    }
    envelope = {
        "schema_version": 2,
        "gate": "smoke_runs",
        "passed": metrics == expected and all(suite["passed"] is True for suite in suites),
        "immutable_sha256": immutable_sha256,
        "metrics": metrics,
        "artifacts": references,
    }
    _write_json_with_checksum(publication_root / ATTEMPT_EVIDENCE_NAME, envelope)
    return envelope


def _completion_payload(
    *,
    pointer_path: Path,
    pointer: Mapping[str, Any],
    evidence_path: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_root = Path(str(pointer["attempt_root"]))
    runs_root = Path(str(pointer["runs_root"]))
    return _with_identity(
        {
            "schema_version": 1,
            "protocol": ATTEMPT_COMPLETE_PROTOCOL,
            "passed": True,
            "attempt": _pointer_reference(pointer_path, pointer),
            "evidence": str(evidence_path),
            "evidence_sha256": _sha256(evidence_path),
            "cells": 41,
            "suite_cells": {
                "schema5_smoke_32b_long_v1": 15,
                "schema5_smoke_selective_long_v1": 20,
                "schema5_smoke_standard_canaries_v1": 6,
            },
            "inventory": _attempt_inventory(
                attempt_root=attempt_root,
                runs_root=runs_root,
            ),
            "provenance": dict(provenance),
        },
        "completion_id",
    )


def _validate_completion(
    *,
    pointer_path: Path,
    pointer: Mapping[str, Any],
    require_sealed: bool = True,
) -> dict[str, Any]:
    attempt_root = Path(str(pointer["attempt_root"]))
    runs_root = Path(str(pointer["runs_root"]))
    marker_path = attempt_root / ATTEMPT_COMPLETE_NAME
    marker = _read_sealed_json(
        marker_path, description="smoke attempt completion marker"
    )
    _verify_identity(
        marker, "completion_id", description="smoke attempt completion"
    )
    evidence_path = attempt_root / ATTEMPT_EVIDENCE_NAME
    expected_attempt = _pointer_reference(pointer_path, pointer)
    if (
        marker.get("schema_version") != 1
        or marker.get("protocol") != ATTEMPT_COMPLETE_PROTOCOL
        or marker.get("passed") is not True
        or marker.get("attempt") != expected_attempt
        or marker.get("evidence") != str(evidence_path)
        or marker.get("evidence_sha256") != _sha256(evidence_path)
        or marker.get("cells") != 41
        or marker.get("suite_cells")
        != {
            "schema5_smoke_32b_long_v1": 15,
            "schema5_smoke_selective_long_v1": 20,
            "schema5_smoke_standard_canaries_v1": 6,
        }
        or marker.get("inventory")
        != _attempt_inventory(
            attempt_root=attempt_root,
            runs_root=runs_root,
            exclude_attempt_files=frozenset({ATTEMPT_COMPLETE_NAME}),
        )
    ):
        raise SmokeRunError("smoke attempt completion marker is invalid")
    if require_sealed:
        _assert_tree_read_only(
            attempt_root, description="successful smoke attempt evidence"
        )
        _assert_tree_read_only(
            runs_root, description="successful smoke attempt runs"
        )
    return marker


def _selector_payload(
    *,
    pointer_path: Path,
    pointer: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    completion_path = Path(str(pointer["attempt_root"])) / ATTEMPT_COMPLETE_NAME
    return _with_identity(
        {
            "schema_version": 1,
            "protocol": CURRENT_SELECTOR_PROTOCOL,
            "attempt_id": pointer["attempt_id"],
            "attempt_pointer": str(pointer_path),
            "attempt_pointer_sha256": _sha256(pointer_path),
            "pointer_id": pointer["pointer_id"],
            "completion_marker": str(completion_path),
            "completion_marker_sha256": _sha256(completion_path),
            "completion_id": completion["completion_id"],
            "evidence": completion["evidence"],
            "evidence_sha256": completion["evidence_sha256"],
        },
        "selector_id",
    )


def _publish_success(
    *,
    base: Path,
    pointer_path: Path,
    pointer: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_root = Path(str(pointer["attempt_root"]))
    runs_root = Path(str(pointer["runs_root"]))
    evidence_path = attempt_root / ATTEMPT_EVIDENCE_NAME
    completion = _completion_payload(
        pointer_path=pointer_path,
        pointer=pointer,
        evidence_path=evidence_path,
        provenance=provenance,
    )
    _write_once_sealed(
        attempt_root / ATTEMPT_COMPLETE_NAME,
        completion,
        description="smoke attempt completion marker",
    )
    _seal_tree_read_only(runs_root, description="successful smoke attempt runs")
    _seal_tree_read_only(
        attempt_root, description="successful smoke attempt evidence"
    )
    # CURRENT is the marker-last cache. It becomes visible only after the selected
    # pointer, completion, evidence, and all run artifacts are sealed.
    selector = _selector_payload(
        pointer_path=pointer_path,
        pointer=pointer,
        completion=completion,
    )
    _replace_sealed_json(base / CURRENT_SELECTOR_NAME, selector)
    return selector


def _load_current_attempt(
    *,
    state_root: Path,
    results_root: Path,
    expected_provenance: Mapping[str, Any] | None = None,
    require_latest: bool = True,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = _attempt_base(state_root)
    known = _load_attempt_pointers(base=base, results_root=results_root)
    if not known:
        raise SmokeRunError("no generation-scoped smoke attempt exists")
    selector = _read_sealed_json(
        base / CURRENT_SELECTOR_NAME,
        description="current smoke attempt selector",
    )
    if set(selector) != _CURRENT_SELECTOR_FIELDS:
        raise SmokeRunError("current smoke selector fields drifted")
    _verify_identity(selector, "selector_id", description="current smoke selector")
    matches = [
        (path, pointer)
        for path, pointer in known
        if (
            selector.get("attempt_pointer") == str(path)
            and selector.get("attempt_pointer_sha256") == _sha256(path)
            and selector.get("pointer_id") == pointer["pointer_id"]
            and selector.get("attempt_id") == pointer["attempt_id"]
        )
    ]
    if (
        selector.get("schema_version") != 1
        or selector.get("protocol") != CURRENT_SELECTOR_PROTOCOL
        or len(matches) != 1
    ):
        raise SmokeRunError("current smoke selector is invalid")
    pointer_path, pointer = matches[0]
    if require_latest and pointer_path != known[-1][0]:
        raise SmokeRunError("current smoke selector replays a stale attempt")
    completion = _validate_completion(
        pointer_path=pointer_path, pointer=pointer
    )
    expected_selector = _selector_payload(
        pointer_path=pointer_path,
        pointer=pointer,
        completion=completion,
    )
    if selector != expected_selector:
        raise SmokeRunError("current smoke selector differs from sealed completion")
    if (
        expected_provenance is not None
        and completion.get("provenance") != dict(expected_provenance)
    ):
        raise SmokeRunError(
            "current smoke attempt was produced for a stale capacity/rollout generation"
        )
    evidence_path = Path(str(selector["evidence"]))
    if (
        evidence_path.is_symlink()
        or not evidence_path.is_file()
        or stat.S_IMODE(evidence_path.stat().st_mode) & 0o222
        or evidence_path.stat().st_nlink != 1
        or _sha256(evidence_path) != selector["evidence_sha256"]
    ):
        raise SmokeRunError("current smoke evidence is not exact and sealed")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SmokeRunError("current smoke evidence is unreadable") from exc
    if (
        not isinstance(evidence, dict)
        or evidence.get("passed") is not True
        or evidence.get("metrics", {}).get("schema5_complete_cells") != 41
    ):
        raise SmokeRunError("current smoke evidence does not prove 41 cells")
    return pointer_path, pointer, completion, selector


def verify_current_attempt(
    *,
    results_root: Path,
    server_pool_root: Path,
    release_worktree: Path,
    harness_prefix: Path,
    state_root: Path,
    immutable_pins_sha256: str,
    rollout_generation: int,
) -> dict[str, Any]:
    """Resolve CURRENT through its sealed pointer and current live generation."""

    results_root = results_root.expanduser().resolve()
    server_pool_root = server_pool_root.expanduser().resolve()
    release_worktree = release_worktree.expanduser().resolve()
    harness_prefix = harness_prefix.expanduser().resolve()
    state_root = state_root.expanduser().resolve()
    control, immutable = _load_immutable_context(
        state_root=state_root,
        results_root=results_root,
        server_pool_root=server_pool_root,
        release_worktree=release_worktree,
        harness_prefix=harness_prefix,
        immutable_pins_sha256=immutable_pins_sha256,
    )
    _validate_preproduction_rollout_generation(control, rollout_generation)
    readiness_generation, _trusted_catalog = _trusted_readiness_generation(
        state_root=state_root,
        control=control,
        server_pool_root=server_pool_root,
        rollout_generation=rollout_generation,
    )
    provenance = _attempt_provenance(
        immutable_sha256=immutable_pins_sha256,
        immutable=immutable,
        readiness_generation=readiness_generation,
        rollout_generation=rollout_generation,
    )
    pointer_path, pointer, completion, selector = _load_current_attempt(
        state_root=state_root,
        results_root=results_root,
        expected_provenance=provenance,
    )
    return {
        "status": "verified",
        "passed": True,
        "attempt_id": pointer["attempt_id"],
        "attempt_pointer": str(pointer_path),
        "pointer_id": pointer["pointer_id"],
        "completion_id": completion["completion_id"],
        "selector_id": selector["selector_id"],
        "evidence": selector["evidence"],
        "evidence_sha256": selector["evidence_sha256"],
        "capacity_generation": readiness_generation["capacity_generation"],
        "rollout_generation": readiness_generation["rollout_generation"],
        "trusted_catalog_id": readiness_generation["catalog_id"],
    }


def _execute_attempt(
    *,
    pointer: Mapping[str, Any],
    results_root: Path,
    server_pool_root: Path,
    release_worktree: Path,
    harness_prefix: Path,
    state_root: Path,
    immutable_pins_sha256: str,
    rollout_generation: int,
    runtime_attestation: Mapping[str, Any],
    immutable: Mapping[str, Any],
    trusted_catalog: TrustedGenerationCatalog,
) -> dict[str, Any]:
    """Execute and census one fresh 41-cell attempt."""

    runs_root = Path(str(pointer["runs_root"]))
    attempt_root = Path(str(pointer["attempt_root"]))
    execution_root = Path(str(pointer["execution_root"]))
    readiness_generation = pointer.get("readiness_generation")
    if not isinstance(readiness_generation, Mapping):
        raise SmokeRunError("smoke attempt lacks its trusted readiness generation")
    selected_prefix = (
        readiness_generation.get("release_fleet_contract_sha256"),
        readiness_generation.get("fleet_contract_sha256"),
        readiness_generation.get("capacity_generation"),
        readiness_generation.get("rollout_generation"),
    )
    trusted_generation_tuples = frozenset(
        identity
        for identity in trusted_catalog.allowed_generation_tuples
        if identity[:4] == selected_prefix
    )
    if not trusted_generation_tuples:
        raise SmokeRunError(
            "smoke trusted catalog contains no endpoint for its selected generation"
        )
    suites: list[dict[str, Any]] = []
    execution_halted: str | None = None
    for run_id, relative_config, expected_cells in SMOKE_SUITES:
        run_root = runs_root / run_id
        identity = _verify_suite(
            run_root,
            config_path=release_worktree / relative_config,
            expected_cells=expected_cells,
            expected_release_id=str(immutable["release_id"]),
            expected_git_commit=str(immutable["git_commit"]),
            expected_source_tree_sha256=str(immutable["source_tree_sha256"]),
            expected_harness_environment_sha256=str(
                immutable["harness_environment_sha256"]
            ),
            expected_serving_environment_sha256=str(
                immutable["serving_environment_sha256"]
            ),
            expected_model_contract_sha256=str(
                immutable["model_contract_sha256"]
            ),
            expected_attempt_binding=_attempt_binding(pointer),
        )
        snapshot = load_manifest(run_root, verify_frozen=True)
        policy = load_artifact_policy(run_root, required=True)
        assert policy is not None
        cell_reports: list[dict[str, Any]] = []
        for index, cell in enumerate(snapshot.cells):
            status, rows = _status(
                run_root,
                index,
                model_contract_path=(
                    release_worktree / "configs" / "model_contracts.v1.json"
                ),
                trusted_generation_tuples=trusted_generation_tuples,
            )
            if execution_halted is None and _runnable(status):
                task = _task(
                    run_root=run_root,
                    source_index=index,
                    server_pool_root=server_pool_root,
                    release_git_commit=str(immutable["git_commit"]),
                    protected_capacity_marker_path=str(
                        immutable["protected_capacity_marker_path"]
                    ),
                    protected_capacity_marker_sha256=str(
                        immutable["protected_capacity_marker_sha256"]
                    ),
                    protected_capacity_marker_id=str(
                        immutable["protected_capacity_marker_id"]
                    ),
                    immutable_pins_sha256=immutable_pins_sha256,
                    fleet_contract_path=str(immutable["fleet_contract_path"]),
                    fleet_contract_sha256=str(
                        immutable["fleet_contract_sha256"]
                    ),
                    release_fleet_contract_sha256=str(
                        immutable["release_fleet_contract_sha256"]
                    ),
                    capacity_generation=int(immutable["capacity_generation"]),
                    rollout_generation=rollout_generation,
                    runtime_attestation=runtime_attestation,
                )
                try:
                    _run_task(
                        task,
                        state_root=state_root,
                        execution_root=execution_root,
                        release_worktree=release_worktree,
                        harness_prefix=harness_prefix,
                        hf_home=Path(str(immutable["hf_home"])).resolve(),
                        runtime_attestation=runtime_attestation,
                        immutable=immutable,
                    )
                except SmokeRunError as exc:
                    execution_halted = str(exc)
                status, rows = _status(
                    run_root,
                    index,
                    model_contract_path=(
                        release_worktree / "configs" / "model_contracts.v1.json"
                    ),
                    trusted_generation_tuples=(
                        trusted_generation_tuples
                    ),
                )

            incidents = _incident_accounting(rows, status)
            trusted_errors = trusted_generation_catalog_errors(
                rows, trusted_generation_tuples
            )
            provenance_errors = _provenance_errors(
                run_root=run_root,
                cell=cell,
                status=status,
                rows=rows,
                immutable=immutable,
                policy=policy,
                rollout_generation=rollout_generation,
            )
            failure = status.failure
            failure_summary = None
            if failure is not None:
                failure_summary = {
                    "classification": failure.classification,
                    "disposition": failure.disposition,
                    "attempts": failure.attempts,
                    "last_error_type": failure.last_error.get("type"),
                    "last_error_message": failure.last_error.get("message"),
                }
            cell_reports.append(
                {
                    "cell_id": cell.cell_id,
                    "status": status.status.value,
                    "valid_qids": status.valid_count,
                    "expected_qids": status.expected_count,
                    **incidents,
                    "provenance_failures": int(bool(provenance_errors)),
                    "provenance_errors": list(provenance_errors),
                    "trusted_generation_errors": list(trusted_errors),
                    "results_artifact": (
                        None
                        if not (
                            run_root
                            / "cells"
                            / cell.cell_id
                            / "results.jsonl"
                        ).is_file()
                        else {
                            "path": str(
                                (
                                    run_root
                                    / "cells"
                                    / cell.cell_id
                                    / "results.jsonl"
                                ).resolve()
                            ),
                            "sha256": _sha256(
                                run_root
                                / "cells"
                                / cell.cell_id
                                / "results.jsonl"
                            ),
                            "row_count": len(rows),
                        }
                    ),
                    "semantic_errors": list(status.errors),
                    "failure": failure_summary,
                }
            )

        complete_cells = sum(
            row["status"] == CompletionState.COMPLETE.value
            for row in cell_reports
        )
        context_incidents = sum(row["context_incidents"] for row in cell_reports)
        protocol_incidents = sum(row["protocol_incidents"] for row in cell_reports)
        transport_incidents = sum(row["transport_incidents"] for row in cell_reports)
        truncation_incidents = sum(row["truncation_incidents"] for row in cell_reports)
        provenance_failures = sum(row["provenance_failures"] for row in cell_reports)
        trusted_generation_failures = sum(
            bool(row["trusted_generation_errors"]) for row in cell_reports
        )
        semantic_validation_failures = sum(
            bool(row["semantic_errors"]) for row in cell_reports
        )
        suite_passed = (
            complete_cells == expected_cells
            and context_incidents == 0
            and protocol_incidents == 0
            and transport_incidents == 0
            and truncation_incidents == 0
            and provenance_failures == 0
            and trusted_generation_failures == 0
            and semantic_validation_failures == 0
            and execution_halted is None
        )
        suites.append(
            {
                "schema_version": 1,
                "kind": _SUITE_ARTIFACT_NAMES[run_id],
                "passed": suite_passed,
                "immutable_sha256": immutable_pins_sha256,
                "run_id": run_id,
                "expected_cells": expected_cells,
                "schema5_complete_cells": complete_cells,
                "context_incidents": context_incidents,
                "protocol_incidents": protocol_incidents,
                "transport_incidents": transport_incidents,
                "truncation_incidents": truncation_incidents,
                "provenance_failures": provenance_failures,
                "trusted_generation_failures": trusted_generation_failures,
                "semantic_validation_failures": semantic_validation_failures,
                "top_level_length_censored_qids": sum(
                    row["top_level_length_censored_qids"] for row in cell_reports
                ),
                "top_level_protocol_censored_qids": sum(
                    row["top_level_protocol_censored_qids"] for row in cell_reports
                ),
                "top_level_transport_censored_qids": sum(
                    row["top_level_transport_censored_qids"] for row in cell_reports
                ),
                "transport_affected_qids": sum(
                    row["transport_affected_qids"] for row in cell_reports
                ),
                "transport_censored_coordinates": sum(
                    row["transport_censored_coordinates"] for row in cell_reports
                ),
                "auxiliary_length_censored_draws": sum(
                    row["auxiliary_length_censored_draws"] for row in cell_reports
                ),
                "auxiliary_protocol_censored_draws": sum(
                    row["auxiliary_protocol_censored_draws"] for row in cell_reports
                ),
                "auxiliary_transport_censored_draws": sum(
                    row["auxiliary_transport_censored_draws"] for row in cell_reports
                ),
                "transport_censor_protocol_version": TRANSPORT_CENSOR_PROTOCOL_VERSION,
                "transport_censor_protocol_hash": TRANSPORT_CENSOR_PROTOCOL_HASH,
                "concurrency": 1,
                # The readiness operation is resumable through a fresh, immutable
                # successor attempt; rows within this attempt are never reused.
                "resumable": True,
                "execution_halted": execution_halted is not None,
                "execution_halt_reason": execution_halted,
                "provenance": {
                    "release_id": immutable["release_id"],
                    "git_commit": immutable["git_commit"],
                    "source_tree_sha256": immutable["source_tree_sha256"],
                    "harness_environment_sha256": immutable[
                        "harness_environment_sha256"
                    ],
                    "serving_environment_sha256": immutable[
                        "serving_environment_sha256"
                    ],
                    "model_contract_sha256": immutable["model_contract_sha256"],
                    "fleet_contract_sha256": immutable["fleet_contract_sha256"],
                    "release_fleet_contract_sha256": immutable[
                        "release_fleet_contract_sha256"
                    ],
                    "capacity_generation": immutable["capacity_generation"],
                    "server_pool_root": str(server_pool_root),
                    "rollout_generation": rollout_generation,
                    "trusted_generation_catalog": {
                        "catalog_id": trusted_catalog.catalog_id,
                        "marker_path": str(trusted_catalog.marker_path),
                        "marker_sha256": trusted_catalog.marker_sha256,
                        "inventory_sha256": trusted_catalog.inventory_sha256,
                        "catalog_payload_sha256": trusted_catalog.catalog_sha256,
                        "allowed_generation_tuple_count": len(
                            trusted_catalog.allowed_generation_tuples
                        ),
                    },
                },
                "suite_identity": identity,
                "cells": cell_reports,
            }
        )
    return _write_smoke_evidence(
        state_root=state_root,
        immutable_sha256=immutable_pins_sha256,
        suites=suites,
        output_root=attempt_root,
    )


def _retryable_attempt_failure(envelope: Mapping[str, Any]) -> tuple[bool, str]:
    metrics = envelope.get("metrics")
    if not isinstance(metrics, Mapping):
        return False, "malformed_smoke_evidence"
    if int(metrics.get("transport_incidents", 0)) > 0:
        return True, "transport_censor_observed"
    artifacts = envelope.get("artifacts")
    if isinstance(artifacts, list):
        for reference in artifacts:
            if not isinstance(reference, Mapping):
                continue
            try:
                suite = json.loads(
                    Path(str(reference["path"])).read_text(encoding="utf-8")
                )
            except (KeyError, OSError, UnicodeError, json.JSONDecodeError):
                return False, "malformed_smoke_suite_evidence"
            if suite.get("execution_halted") is True:
                return True, "worker_interrupted_or_failed"
            if (
                suite.get("protocol_incidents") == 0
                and suite.get("truncation_incidents") == 0
                and suite.get("context_incidents") == 0
                and suite.get("provenance_failures") == 0
                and suite.get("trusted_generation_failures") == 0
                and suite.get("schema5_complete_cells")
                != suite.get("expected_cells")
            ):
                return True, "incomplete_attempt"
    return False, "scientific_readiness_failure"


def run_or_verify(
    *,
    results_root: Path,
    server_pool_root: Path,
    release_worktree: Path,
    harness_prefix: Path,
    state_root: Path,
    immutable_pins_sha256: str,
    rollout_generation: int,
    apply: bool,
) -> dict[str, Any]:
    results_root = results_root.expanduser().resolve()
    server_pool_root = server_pool_root.expanduser().resolve()
    release_worktree = release_worktree.expanduser().resolve()
    harness_prefix = harness_prefix.expanduser().resolve()
    state_root = state_root.expanduser().resolve()
    if _SHA256.fullmatch(immutable_pins_sha256) is None:
        raise SmokeRunError("immutable_pins_sha256 is not a lowercase SHA-256")
    if (
        not isinstance(rollout_generation, int)
        or isinstance(rollout_generation, bool)
        or rollout_generation < 1
    ):
        raise SmokeRunError("rollout_generation must be positive")
    if not server_pool_root.is_dir():
        raise SmokeRunError(f"canonical server pool is missing: {server_pool_root}")
    if not (harness_prefix / "bin/python").is_file():
        raise SmokeRunError(f"immutable harness Python is missing: {harness_prefix}")
    if not (release_worktree / "slurm/dispatch_sweeps.py").is_file():
        raise SmokeRunError(f"immutable dispatcher is missing: {release_worktree}")
    if not (state_root / schema5_control.CONTROL_FILENAME).is_file():
        raise SmokeRunError(f"schema-5 control state is missing: {state_root}")
    if not apply:
        return verify_current_attempt(
            results_root=results_root,
            server_pool_root=server_pool_root,
            release_worktree=release_worktree,
            harness_prefix=harness_prefix,
            state_root=state_root,
            immutable_pins_sha256=immutable_pins_sha256,
            rollout_generation=rollout_generation,
        )
    # Hold the same cross-node control lock used by ``resume`` for every attempt.
    # This closes the check/use race across fleet, attempt initialization, draws,
    # sealing, and CURRENT publication.
    with schema5_control.control_lock(state_root), _exclusive_smoke_lock(state_root):
        control, immutable = _load_immutable_context(
            state_root=state_root,
            results_root=results_root,
            server_pool_root=server_pool_root,
            release_worktree=release_worktree,
            harness_prefix=harness_prefix,
            immutable_pins_sha256=immutable_pins_sha256,
        )
        _validate_preproduction_rollout_generation(control, rollout_generation)
        readiness_generation, trusted_catalog = _trusted_readiness_generation(
            state_root=state_root,
            control=control,
            server_pool_root=server_pool_root,
            rollout_generation=rollout_generation,
        )
        provenance = _attempt_provenance(
            immutable_sha256=immutable_pins_sha256,
            immutable=immutable,
            readiness_generation=readiness_generation,
            rollout_generation=rollout_generation,
        )
        base = _attempt_base(state_root)
        known_before = _load_attempt_pointers(
            base=base, results_root=results_root
        )
        try:
            # Idempotent success reruns are verification-only and never redraw.
            return verify_current_attempt(
                results_root=results_root,
                server_pool_root=server_pool_root,
                release_worktree=release_worktree,
                harness_prefix=harness_prefix,
                state_root=state_root,
                immutable_pins_sha256=immutable_pins_sha256,
                rollout_generation=rollout_generation,
            )
        except SmokeRunError:
            current_path = base / CURRENT_SELECTOR_NAME
            if current_path.exists() or current_path.is_symlink():
                # A valid selector for the most recent prior success may be stale
                # only because capacity/rollout advanced. Any malformed selector
                # or replay of an older success is an integrity failure.
                selected_path, _, selected_completion, _ = _load_current_attempt(
                    state_root=state_root,
                    results_root=results_root,
                    expected_provenance=None,
                    require_latest=False,
                )
                latest_success_path = next(
                    (
                        path
                        for path, item in reversed(known_before)
                        if _attempt_terminal_kind(item) == "complete"
                    ),
                    None,
                )
                if (
                    selected_path != latest_success_path
                    or selected_completion.get("provenance") == provenance
                ):
                    raise
            if known_before:
                latest_path, latest = known_before[-1]
                if _attempt_terminal_kind(latest) == "complete":
                    latest_completion = _validate_completion(
                        pointer_path=latest_path,
                        pointer=latest,
                        require_sealed=False,
                    )
                    if latest_completion.get("provenance") == provenance:
                        # The only recoverable success crash boundary is after the
                        # attempt was sealed but before marker-last CURRENT. A present
                        # but invalid selector is tampering/replay and fails closed.
                        selector = _selector_payload(
                            pointer_path=latest_path,
                            pointer=latest,
                            completion=latest_completion,
                        )
                        _seal_tree_read_only(
                            Path(str(latest["runs_root"])),
                            description="recovered successful smoke attempt runs",
                        )
                        _seal_tree_read_only(
                            Path(str(latest["attempt_root"])),
                            description="recovered successful smoke attempt evidence",
                        )
                        _replace_sealed_json(current_path, selector)
                        return verify_current_attempt(
                            results_root=results_root,
                            server_pool_root=server_pool_root,
                            release_worktree=release_worktree,
                            harness_prefix=harness_prefix,
                            state_root=state_root,
                            immutable_pins_sha256=immutable_pins_sha256,
                            rollout_generation=rollout_generation,
                        )
        try:
            runtime_attestation = schema5_control.ensure_runtime_integrity_attestation(
                state_root,
                control,
                generation=rollout_generation,
                force_full=True,
            )
        except schema5_control.ControlError as exc:
            raise SmokeRunError(
                f"schema-5 smoke runtime integrity failed: {exc}"
            ) from exc
        if set(_SUITE_ARTIFACT_NAMES) != {suite[0] for suite in SMOKE_SUITES}:
            raise SmokeRunError(
                "smoke suite IDs differ from the closed readiness contract"
            )
        last_failure: str | None = None
        for _ in range(MAX_AUTOMATIC_ATTEMPTS):
            pointer_path, pointer = _create_attempt(
                base=base,
                results_root=results_root,
                release_worktree=release_worktree,
                immutable_sha256=immutable_pins_sha256,
                immutable=immutable,
                readiness_generation=readiness_generation,
                now=time.time(),
            )
            envelope = _execute_attempt(
                pointer=pointer,
                results_root=results_root,
                server_pool_root=server_pool_root,
                release_worktree=release_worktree,
                harness_prefix=harness_prefix,
                state_root=state_root,
                immutable_pins_sha256=immutable_pins_sha256,
                rollout_generation=rollout_generation,
                runtime_attestation=runtime_attestation,
                immutable=immutable,
                trusted_catalog=trusted_catalog,
            )
            if envelope.get("passed") is True:
                _publish_success(
                    base=base,
                    pointer_path=pointer_path,
                    pointer=pointer,
                    provenance=provenance,
                )
                return verify_current_attempt(
                    results_root=results_root,
                    server_pool_root=server_pool_root,
                    release_worktree=release_worktree,
                    harness_prefix=harness_prefix,
                    state_root=state_root,
                    immutable_pins_sha256=immutable_pins_sha256,
                    rollout_generation=rollout_generation,
                )
            retryable, reason = _retryable_attempt_failure(envelope)
            _publish_failure(
                pointer_path,
                pointer,
                reason=reason,
                retryable=retryable,
            )
            last_failure = reason
            if not retryable:
                raise SmokeRunError(
                    f"smoke attempt failed scientific readiness: {reason}"
                )
        raise SmokeRunError(
            "smoke automatic retry budget exhausted after evidence-preserving "
            f"attempts: {last_failure}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--server-pool-root", required=True, type=Path)
    parser.add_argument("--release-worktree", required=True, type=Path)
    parser.add_argument("--harness-prefix", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--immutable-pins-sha256", required=True)
    parser.add_argument("--rollout-generation", required=True, type=int)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run_or_verify(
            results_root=args.results_root,
            server_pool_root=args.server_pool_root,
            release_worktree=args.release_worktree,
            harness_prefix=args.harness_prefix,
            state_root=args.state_root,
            immutable_pins_sha256=args.immutable_pins_sha256,
            rollout_generation=args.rollout_generation,
            apply=args.apply,
        )
    except (
        ArtifactPolicyError,
        OSError,
        SmokeInitializationError,
        SmokeRunError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
