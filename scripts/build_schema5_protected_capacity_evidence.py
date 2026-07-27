#!/usr/bin/env python3
"""Run and seal the real schema-5 protected-capacity canary transaction.

The default command is non-mutating and prints the exact four-role plan.  The active
role contains every allocation in the solver-qualified effective fleet, the warm role
contains the separately retained ``2xTP1 + 1xTP2`` turnover topology (four GPUs), and
the client role contains 384 tasks.  Active and warm job elements consume the inclusive
64-job non-cell reserve, leaving ``64 - active - 3`` held controller/monitor/other
placeholders.  The resulting canary therefore always contains exactly 448 job elements,
split into arrays of at most 24 elements.  ``run --apply``
creates a marker-first transaction, submits the generation/comment-addressed Slurm
jobs, waits until all protected resources are simultaneously visible, seals normalized
raw scheduler evidence, and delegates marker-last publication to
``publish_schema5_protected_capacity``.

No success artifact is written when capacity is short, scheduler truth is incomplete,
an array is requeue-enabled, or any partition/QOS/resource fact drifts.  The historical
CPU-only fleet-turnover canary is deliberately not an input to this protocol.
Before the first ``sbatch``, two complete squeue/sacct occupancy observations 60
seconds apart must be identical, and the existing user elements plus all 448 canary
elements must fit the effective association/QOS submit limit.  Thus a 448 limit
requires exact account quiescence; a larger limit may safely accommodate only its
proven residual headroom.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    # The production launcher invokes this script with ``python -I`` from the
    # immutable release checkout.  Add that exact checkout so the builder calls
    # the control plane's canonical tree-hash implementation instead of carrying
    # a second, potentially divergent implementation.
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agents_scaling.serving.fleet_contract import (
    FleetContractError,
    FrozenFleetContract,
    load_fleet_contract,
)
from agents_scaling.serving.model_contracts import (
    ModelContractError,
    load_model_contracts,
)
from agents_scaling.serving import protected_capacity as runtime_capacity
from slurm.schema5_control import sha256_tree


def _load_sibling_publisher() -> Any:
    path = Path(__file__).resolve().with_name(
        "publish_schema5_protected_capacity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_protected_capacity_publisher",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protected-capacity publisher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = _load_sibling_publisher()


PROTOCOL = "schema5-v1.2-r3-protected-capacity-builder-v4"
INTENT_FILENAME = "PROTECTED_CAPACITY_BUILD_INTENT.json"
LEDGER_FILENAME = "protected_capacity_build_ledger.json"
SCHEDULER_EVIDENCE_FILENAME = "PROTECTED_CAPACITY_SCHEDULER_EVIDENCE.json"
CANARY_EVIDENCE_FILENAME = "PROTECTED_CAPACITY_CANARY_EVIDENCE.json"
OCCUPANCY_PREFLIGHT_FILENAME = "PROTECTED_CAPACITY_OCCUPANCY_PREFLIGHT.json"
READY_DIRECTORY = "ready"
RELEASE_FILENAME = "RELEASE"
ROLE_ORDER = ("server_active", "server_warm", "client", "reserve")
MAX_ARRAY_TASKS = 24
_PLACEMENT_NAME_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z"
)
BASE_ACTIVE_SERVER_JOB_ELEMENTS = 22
BASE_ACTIVE_GPUS = 24
WARM_TURNOVER_JOB_ELEMENTS = 3
WARM_TURNOVER_GPUS = 4
CLIENT_JOB_ELEMENTS = 384
TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS = 64
EXPECTED_TOTAL_JOB_ELEMENTS = (
    CLIENT_JOB_ELEMENTS + TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS
)
CLIENT_TIME_LIMIT = "12:00:00"
SERVER_TIME_LIMIT = "1-00:00:00"
CLIENT_TIME_LIMIT_SECONDS = 43_200
SERVER_TIME_LIMIT_SECONDS = 86_400
MAX_JOBS_CONSUMING_STATES = frozenset(
    {
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "RESIZING",
        "SUSPENDED",
    }
)
OCCUPANCY_OBSERVATION_INTERVAL_SECONDS = 60.0
ROLE_HELD = {
    "server_active": False,
    "server_warm": False,
    "client": False,
    "reserve": True,
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_JOB_ID_RE = re.compile(r"([0-9]+)(?:_([0-9]+))?\Z")


class ProtectedCapacityBuildError(RuntimeError):
    """The real capacity transaction cannot be proven."""


def canonical_bytes(value: Any) -> bytes:
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lexical_absolute(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if any(character in str(lexical) for character in ("\x00", "\n", "\r")):
        raise ProtectedCapacityBuildError(
            f"capacity evidence path contains unsafe characters: {lexical}"
        )
    return lexical


def _canonical_existing_path(
    path: Path,
    *,
    description: str,
    kind: str,
) -> Path:
    lexical = _lexical_absolute(path)
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProtectedCapacityBuildError(
            f"{description} is unavailable: {exc}"
        ) from exc
    if resolved != lexical:
        raise ProtectedCapacityBuildError(
            f"{description} traverses a symlink"
        )
    metadata = lexical.stat(follow_symlinks=False)
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise ProtectedCapacityBuildError(
            f"{description} is not a regular file"
        )
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise ProtectedCapacityBuildError(
            f"{description} is not a directory"
        )
    return lexical


def _stable_bytes(
    path: Path,
    *,
    description: str,
    require_read_only: bool,
    require_single_link: bool,
) -> bytes:
    canonical = _canonical_existing_path(
        path,
        description=description,
        kind="file",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(canonical, flags)
    except OSError as exc:
        raise ProtectedCapacityBuildError(
            f"cannot open {description} safely: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = canonical.stat(follow_symlinks=False)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        not stat.S_ISREG(before.st_mode)
        or identity(before) != identity(after)
        or (current.st_dev, current.st_ino)
        != (after.st_dev, after.st_ino)
        or (require_read_only and stat.S_IMODE(before.st_mode) & 0o222)
        or (require_single_link and before.st_nlink != 1)
    ):
        raise ProtectedCapacityBuildError(
            f"{description} is mutable, linked, or changed while reading"
        )
    return b"".join(chunks)


def _atomic_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish_sealed_once(path: Path, value: Mapping[str, Any]) -> None:
    target = _lexical_absolute(path)
    parent = _canonical_existing_path(
        target.parent,
        description=f"{target.name} parent",
        kind="directory",
    )
    payload = canonical_bytes(dict(value))
    lock_flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    lock_path = parent / f".{target.name}.publish.lock"
    try:
        lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    except OSError as exc:
        raise ProtectedCapacityBuildError(
            f"cannot lock sealed publication {target}: {exc}"
        ) from exc
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if target.is_symlink():
            raise ProtectedCapacityBuildError(
                f"sealed evidence target is symlinked: {target}"
            )
        target_metadata = (
            target.stat(follow_symlinks=False)
            if target.exists()
            else None
        )
        exact_candidates: list[Path] = []
        for candidate in sorted(
            child
            for child in parent.iterdir()
            if child.name.startswith(f".{target.name}.")
            and child.name.endswith(".publishing")
        ):
            candidate_raw = _stable_bytes(
                candidate,
                description=(
                    f"interrupted {target.name} publication "
                    f"{candidate.name}"
                ),
                require_read_only=True,
                require_single_link=False,
            )
            candidate_metadata = candidate.stat(follow_symlinks=False)
            if target_metadata is not None and (
                candidate_metadata.st_dev,
                candidate_metadata.st_ino,
            ) == (
                target_metadata.st_dev,
                target_metadata.st_ino,
            ):
                # A crash after link(2), before temporary unlink, leaves these
                # names on the same inode.  Removing the temporary restores the
                # target's required single-link identity.
                candidate.unlink()
            elif candidate_raw == payload:
                exact_candidates.append(candidate)
            else:
                raise ProtectedCapacityBuildError(
                    "conflicting interrupted sealed publication artifact: "
                    f"{candidate}"
                )
        if not target.exists() and exact_candidates:
            survivor = exact_candidates.pop(0)
            os.link(survivor, target, follow_symlinks=False)
            survivor.unlink()
        for candidate in exact_candidates:
            candidate.unlink()
        if target.exists():
            if (
                _stable_bytes(
                    target,
                    description=f"existing sealed evidence {target.name}",
                    require_read_only=True,
                    require_single_link=True,
                )
                != payload
            ):
                raise ProtectedCapacityBuildError(
                    f"sealed evidence conflicts at {target}"
                )
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=parent,
                prefix=f".{target.name}.",
                suffix=".publishing",
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.chmod(0o444)
                os.link(temporary, target, follow_symlinks=False)
                temporary.unlink()
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        _fsync_directory(parent)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _read_json(
    path: Path,
    *,
    description: str,
    sealed: bool,
) -> dict[str, Any]:
    raw = _stable_bytes(
        path,
        description=description,
        require_read_only=sealed,
        require_single_link=True,
    )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ProtectedCapacityBuildError(
                    f"{description} duplicates JSON key {key!r}"
                )
            value[key] = item
        return value

    def reject_nonfinite(token: str) -> Any:
        raise ProtectedCapacityBuildError(
            f"{description} contains non-finite JSON value {token}"
        )

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except ProtectedCapacityBuildError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtectedCapacityBuildError(
            f"cannot read {description}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ProtectedCapacityBuildError(f"{description} must be an object")
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise ProtectedCapacityBuildError(
                f"{description} contains a non-finite number"
            )
        if isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    if sealed and raw != canonical_bytes(value):
        raise ProtectedCapacityBuildError(
            f"{description} is not canonical JSON"
        )
    return value


def _invoke(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    argv: Sequence[str],
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    try:
        result = (
            subprocess.run(
                list(argv),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if runner is None
            else runner(list(argv), timeout=timeout)
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProtectedCapacityBuildError(
            f"scheduler command failed: {list(argv)!r}: {exc}"
        ) from exc
    if (
        not isinstance(result, subprocess.CompletedProcess)
        or result.returncode != 0
        or not isinstance(result.stdout, str)
        or not isinstance(result.stderr, str)
    ):
        raise ProtectedCapacityBuildError(
            f"scheduler command failed: {list(argv)!r}: "
            f"rc={getattr(result, 'returncode', 'invalid')} "
            f"{str(getattr(result, 'stderr', ''))[:500]}"
        )
    return result


def _time_seconds(value: str) -> int:
    raw = value.strip()
    if raw.upper() in {"UNLIMITED", "INFINITE"}:
        return 2**31 - 1
    days = 0
    if "-" in raw:
        day_raw, raw = raw.split("-", 1)
        days = int(day_raw)
    fields = [int(item) for item in raw.split(":")]
    if len(fields) == 2:
        hours = 0
        minutes, seconds = fields
    elif len(fields) == 3:
        hours, minutes, seconds = fields
    else:
        raise ProtectedCapacityBuildError(f"invalid Slurm time {value!r}")
    return days * 86_400 + hours * 3600 + minutes * 60 + seconds


def _memory_mib(value: str) -> int:
    match = re.fullmatch(
        r"([0-9]+)([KMGTP]?)(?:[cn])?",
        value.strip(),
        re.IGNORECASE,
    )
    if match is None:
        raise ProtectedCapacityBuildError(f"invalid Slurm memory {value!r}")
    amount = int(match.group(1))
    suffix = match.group(2).upper()
    factor = {
        "": 1,
        "K": 1 / 1024,
        "M": 1,
        "G": 1024,
        "T": 1024**2,
        "P": 1024**3,
    }[suffix]
    result = amount * factor
    if result != int(result):
        raise ProtectedCapacityBuildError(
            f"non-integral MiB memory {value!r}"
        )
    return int(result)


def _gpus_from_tres(value: str) -> int:
    raw = value.strip()
    if not raw or raw.upper() in {"N/A", "NONE", "(NULL)"}:
        return 0
    result = 0
    for item in raw.split(","):
        match = re.fullmatch(
            r"(?:gres/)?gpu(?::[A-Za-z0-9_.-]+)?(?::|=)([0-9]+)",
            item.strip(),
        )
        if match is not None:
            result += int(match.group(1))
    return result


def _load_frozen_fleet(
    *,
    fleet_contract_path: Path,
    fleet_contract_sha256: str,
    model_contract_path: Path,
    model_contract_sha256: str,
    allow_capacity_layout: bool,
) -> FrozenFleetContract:
    if (
        _SHA256_RE.fullmatch(fleet_contract_sha256) is None
        or _SHA256_RE.fullmatch(model_contract_sha256) is None
    ):
        raise ProtectedCapacityBuildError(
            "fleet/model contract hashes must be lowercase SHA-256"
        )
    try:
        models = load_model_contracts(
            model_contract_path,
            expected_sha256=model_contract_sha256,
        )
        fleet = load_fleet_contract(
            fleet_contract_path,
            model_contracts=models,
            expected_sha256=fleet_contract_sha256,
            allow_capacity_layout=allow_capacity_layout,
        )
    except (FleetContractError, ModelContractError, OSError, ValueError) as exc:
        raise ProtectedCapacityBuildError(
            f"frozen fleet contract cannot be loaded: {exc}"
        ) from exc
    if not allow_capacity_layout and (
        len(fleet.replicas) != BASE_ACTIVE_SERVER_JOB_ELEMENTS
        or sum(replica.gpus_per_replica for replica in fleet.replicas)
        != BASE_ACTIVE_GPUS
    ):
        raise ProtectedCapacityBuildError(
            "frozen fleet must contain exactly 22 logical replicas using 24 GPUs"
        )
    return fleet


def _profile_counts(fleet: FrozenFleetContract) -> dict[str, int]:
    return {
        profile: len(replicas)
        for profile, replicas in sorted(fleet.by_profile.items())
    }


def _validate_effective_capacity_authority(
    *,
    base_fleet: FrozenFleetContract,
    effective_fleet: FrozenFleetContract,
    additive_overlay_path: Path,
    additive_overlay_sha256: str,
    static_feasibility_certificate_path: Path,
    static_feasibility_certificate_sha256: str,
    static_feasibility_certificate_id: str,
    capacity_generation: int,
    release_git_commit: str,
    source_tree_sha256: str,
    dispatcher_source_sha256: str,
    qualification_runner_source_sha256: str,
) -> runtime_capacity.StaticFeasibilityCertificate:
    """Prove one exact additive fleet and its zero-QID admission certificate."""

    if (
        type(capacity_generation) is not int
        or capacity_generation < 1
        or _SHA256_RE.fullmatch(additive_overlay_sha256) is None
        or _SHA256_RE.fullmatch(static_feasibility_certificate_sha256) is None
        or _SHA256_RE.fullmatch(static_feasibility_certificate_id) is None
    ):
        raise ProtectedCapacityBuildError(
            "capacity generation/certificate/overlay identity is malformed"
        )
    overlay = _canonical_existing_path(
        additive_overlay_path,
        description="additive overlay contract",
        kind="file",
    )
    if (
        overlay != effective_fleet.path
        or additive_overlay_sha256 != effective_fleet.sha256
    ):
        raise ProtectedCapacityBuildError(
            "current overlay protocol requires the exact effective fleet contract "
            "as its additive-overlay authority"
        )
    base_ids = {replica.replica_id: replica for replica in base_fleet.replicas}
    effective_ids = {
        replica.replica_id: replica for replica in effective_fleet.replicas
    }
    if (
        not set(base_ids).issubset(effective_ids)
        or any(effective_ids[key] != row for key, row in base_ids.items())
    ):
        raise ProtectedCapacityBuildError(
            "effective fleet is not an exact additive extension of the frozen base"
        )
    base_counts = _profile_counts(base_fleet)
    effective_counts = _profile_counts(effective_fleet)
    if any(
        effective_counts[profile] < base_counts[profile]
        for profile in base_counts
    ) or effective_counts == base_counts:
        raise ProtectedCapacityBuildError(
            "effective fleet must be a nonempty additive extension of the base"
        )
    try:
        certificate = runtime_capacity.load_static_feasibility_certificate(
            static_feasibility_certificate_path,
            expected_sha256=static_feasibility_certificate_sha256,
            expected_certificate_id=static_feasibility_certificate_id,
            expected_capacity_generation=capacity_generation,
            expected_base_fleet_contract_sha256=base_fleet.sha256,
            expected_effective_fleet_contract_sha256=effective_fleet.sha256,
            expected_additive_overlay_contract_sha256=additive_overlay_sha256,
            expected_release_git_commit=release_git_commit,
            expected_source_tree_sha256=source_tree_sha256,
            expected_dispatcher_source_sha256=dispatcher_source_sha256,
            expected_qualification_runner_source_sha256=(
                qualification_runner_source_sha256
            ),
        )
    except (
        runtime_capacity.ProtectedCapacityError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise ProtectedCapacityBuildError(
            f"static feasibility certificate cannot authorize capacity: {exc}"
        ) from exc
    if (
        dict(certificate.base_profile_replicas) != base_counts
        or dict(certificate.effective_profile_replicas) != effective_counts
        or certificate.base_logical_replicas != len(base_fleet.replicas)
        or certificate.base_allocated_gpus
        != sum(row.gpus_per_replica for row in base_fleet.replicas)
        or certificate.effective_logical_replicas
        != len(effective_fleet.replicas)
        or certificate.effective_active_gpus
        != sum(row.gpus_per_replica for row in effective_fleet.replicas)
    ):
        raise ProtectedCapacityBuildError(
            "static feasibility topology differs from the supplied fleet contracts"
        )
    residual = (
        TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS
        - len(effective_fleet.replicas)
        - WARM_TURNOVER_JOB_ELEMENTS
    )
    if residual <= 0:
        raise ProtectedCapacityBuildError(
            "effective fleet exhausts the inclusive 64-job non-cell reserve"
        )
    return certificate


def _verify_release_checkout(
    *,
    release_worktree: Path,
    release_git_commit: str,
    release_tag_object: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> dict[str, str]:
    worktree = _canonical_existing_path(
        release_worktree,
        description="release worktree",
        kind="directory",
    )

    def git(*arguments: str) -> str:
        return _invoke(
            runner,
            ["git", "-C", str(worktree), *arguments],
            timeout=60.0,
        ).stdout.strip()

    head = git("rev-parse", "HEAD")
    tag_object = git("rev-parse", f"refs/tags/{publisher.RELEASE_TAG}")
    peeled = git("rev-parse", f"refs/tags/{publisher.RELEASE_TAG}^{{}}")
    tag_type = git("cat-file", "-t", tag_object)
    dirty = git("status", "--porcelain=v1", "--untracked-files=all")
    if (
        head != release_git_commit
        or tag_object != release_tag_object
        or peeled != release_git_commit
        or tag_type != "tag"
        or dirty
    ):
        raise ProtectedCapacityBuildError(
            "capacity builder is not executing against the exact clean annotated "
            "release checkout"
        )
    builder_path = _canonical_existing_path(
        worktree / "scripts" / Path(__file__).name,
        description="tagged capacity builder source",
        kind="file",
    )
    publisher_path = _canonical_existing_path(
        worktree / "scripts" / Path(publisher.__file__).name,
        description="tagged capacity publisher source",
        kind="file",
    )
    dispatcher_path = _canonical_existing_path(
        worktree / "slurm" / "dispatch_sweeps.py",
        description="tagged dispatcher source",
        kind="file",
    )
    qualification_runner_path = _canonical_existing_path(
        worktree / "scripts" / "run_schema5_throughput_qualification.py",
        description="tagged qualification runner source",
        kind="file",
    )
    control_source_path = _canonical_existing_path(
        worktree / "slurm" / "schema5_control.py",
        description="tagged control-plane source",
        kind="file",
    )
    executing_builder = _canonical_existing_path(
        Path(__file__),
        description="executing capacity builder source",
        kind="file",
    )
    executing_publisher = _canonical_existing_path(
        Path(publisher.__file__),
        description="executing capacity publisher source",
        kind="file",
    )
    executing_control_source = _canonical_existing_path(
        Path(sha256_tree.__code__.co_filename),
        description="executing control-plane source",
        kind="file",
    )
    tagged_builder_bytes = _stable_bytes(
        builder_path,
        description="tagged capacity builder source",
        require_read_only=False,
        require_single_link=False,
    )
    tagged_publisher_bytes = _stable_bytes(
        publisher_path,
        description="tagged capacity publisher source",
        require_read_only=False,
        require_single_link=False,
    )
    tagged_control_bytes = _stable_bytes(
        control_source_path,
        description="tagged control-plane source",
        require_read_only=False,
        require_single_link=False,
    )
    if (
        tagged_builder_bytes
        != _stable_bytes(
            executing_builder,
            description="executing capacity builder source",
            require_read_only=False,
            require_single_link=False,
        )
        or tagged_publisher_bytes
        != _stable_bytes(
            executing_publisher,
            description="executing capacity publisher source",
            require_read_only=False,
            require_single_link=False,
        )
        or tagged_control_bytes
        != _stable_bytes(
            executing_control_source,
            description="executing control-plane source",
            require_read_only=False,
            require_single_link=False,
        )
    ):
        raise ProtectedCapacityBuildError(
            "executing builder/publisher/control bytes differ from the tagged "
            "worktree"
        )
    try:
        source_tree_sha256 = sha256_tree(worktree)
        # A clean-status/tree-hash replay closes the interval in which the three
        # source anchors are read.  The certificate is allowed to authorize this
        # transaction only when all anchors come from one stable annotated tree.
        dispatcher_source_sha256 = sha256_bytes(
            _stable_bytes(
                dispatcher_path,
                description="tagged dispatcher source",
                require_read_only=False,
                require_single_link=False,
            )
        )
        qualification_runner_source_sha256 = sha256_bytes(
            _stable_bytes(
                qualification_runner_path,
                description="tagged qualification runner source",
                require_read_only=False,
                require_single_link=False,
            )
        )
        replayed_source_tree_sha256 = sha256_tree(worktree)
        replayed_dirty = git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        replayed_head = git("rev-parse", "HEAD")
        replayed_tag_object = git(
            "rev-parse",
            f"refs/tags/{publisher.RELEASE_TAG}",
        )
        replayed_peeled = git(
            "rev-parse",
            f"refs/tags/{publisher.RELEASE_TAG}^{{}}",
        )
        replayed_tag_type = git("cat-file", "-t", replayed_tag_object)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProtectedCapacityBuildError(
            f"cannot compute frozen release source authority: {exc}"
        ) from exc
    if (
        source_tree_sha256 != replayed_source_tree_sha256
        or replayed_dirty
        or replayed_head != head
        or replayed_tag_object != tag_object
        or replayed_peeled != peeled
        or replayed_tag_type != tag_type
        or any(
            _SHA256_RE.fullmatch(value) is None
            for value in (
                source_tree_sha256,
                dispatcher_source_sha256,
                qualification_runner_source_sha256,
            )
        )
    ):
        raise ProtectedCapacityBuildError(
            "annotated release tree drifted while computing source authority"
        )
    return {
        "release_worktree": str(worktree),
        "source_tree_sha256": source_tree_sha256,
        "builder_source_path": str(builder_path),
        "builder_source_sha256": sha256_bytes(tagged_builder_bytes),
        "publisher_source_path": str(publisher_path),
        "publisher_source_sha256": sha256_bytes(tagged_publisher_bytes),
        "control_source_path": str(control_source_path),
        "control_source_sha256": sha256_bytes(tagged_control_bytes),
        "dispatcher_source_path": str(dispatcher_path),
        "dispatcher_source_sha256": dispatcher_source_sha256,
        "qualification_runner_source_path": str(
            qualification_runner_path
        ),
        "qualification_runner_source_sha256": (
            qualification_runner_source_sha256
        ),
    }


def _active_shapes(fleet: FrozenFleetContract) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    for replica in fleet.replicas:
        shapes.append(
            {
                "shape_id": replica.replica_id,
                "serving_profile": replica.serving_profile,
                "tasks": 1,
                "cpus": replica.cpus_per_task,
                "memory_mib": _memory_mib(replica.memory),
                "gpus": replica.gpus_per_replica,
                "time_limit": replica.time_limit,
            }
        )
    return shapes


def _active_topology(shapes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **{
                key: row[key]
                for key in (
                    "shape_id",
                    "serving_profile",
                    "tasks",
                    "cpus",
                    "memory_mib",
                    "gpus",
                )
            },
            "time_limit_seconds": _time_seconds(str(row["time_limit"])),
        }
        for row in shapes
    ]


def _warm_shapes() -> list[dict[str, Any]]:
    return [
        {
            "shape_id": "warm-tp1-00",
            "serving_profile": "warm-tp1",
            "tasks": 1,
            "cpus": 8,
            "memory_mib": 120 * 1024,
            "gpus": 1,
            "time_limit": SERVER_TIME_LIMIT,
        },
        {
            "shape_id": "warm-tp1-01",
            "serving_profile": "warm-tp1",
            "tasks": 1,
            "cpus": 8,
            "memory_mib": 120 * 1024,
            "gpus": 1,
            "time_limit": SERVER_TIME_LIMIT,
        },
        {
            "shape_id": "warm-tp2-00",
            "serving_profile": "warm-tp2",
            "tasks": 1,
            "cpus": 16,
            "memory_mib": 240 * 1024,
            "gpus": 2,
            "time_limit": SERVER_TIME_LIMIT,
        },
    ]


def _key_values(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in raw.replace("\n", " ").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key or not value:
            continue
        if key in values and values[key] != value:
            raise ProtectedCapacityBuildError(
                f"scheduler output duplicates {key!r}"
            )
        values[key] = value
    return values


def _capture_configuration(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> str:
    raw = _invoke(
        runner,
        ["scontrol", "show", "config"],
        timeout=30.0,
    ).stdout
    match = re.search(
        r"(?m)^\s*PreemptType\s*=\s*(\S+)\s*$",
        raw,
    )
    if match is None:
        raise ProtectedCapacityBuildError(
            "scontrol configuration lacks one PreemptType"
        )
    return f"PreemptType|{match.group(1)}\n"


def _capture_partition(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    *,
    partition: str,
) -> str:
    argv = ["scontrol", "show", "partition", partition, "-o"]
    raw = _invoke(runner, argv, timeout=30.0).stdout.strip()
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ProtectedCapacityBuildError(
            "partition query did not return exactly one row"
        )
    values = _key_values(lines[0])
    if values.get("PartitionName") != partition:
        raise ProtectedCapacityBuildError("partition identity drifted")
    tres: dict[str, str] = {}
    for item in values.get("TRES", "").split(","):
        key, separator, value = item.rpartition("=")
        if separator and key and value:
            tres[key] = value
    memory_raw = tres.get("mem")
    cpu_raw = tres.get("cpu")
    node_raw = tres.get("node")
    gpu_raw = (
        tres.get("gres/gpu")
        or tres.get("gpu")
        or "0"
    )
    if memory_raw is None or cpu_raw is None or node_raw is None:
        raise ProtectedCapacityBuildError(
            "partition TRES does not report CPU, memory, and node inventory"
        )
    cpus = int(cpu_raw)
    nodes = int(node_raw)
    if (
        cpus < 0
        or nodes <= 0
        or (
            "TotalCPUs" in values
            and int(values["TotalCPUs"]) != cpus
        )
        or (
            "TotalNodes" in values
            and int(values["TotalNodes"]) != nodes
        )
    ):
        raise ProtectedCapacityBuildError(
            "partition aggregate TRES conflicts with TotalCPUs/TotalNodes"
        )
    return (
        f"{partition}|{values.get('PreemptMode', '')}|"
        f"{values.get('State', '')}|"
        f"{_time_seconds(values.get('MaxTime', ''))}|"
        f"{cpus}|{_memory_mib(memory_raw)}|{int(gpu_raw)}|{nodes}\n"
    )


def _capture_qos(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    *,
    qos: str,
) -> str:
    argv = [
        "sacctmgr",
        "-nP",
        "show",
        "qos",
        qos,
        (
            "format=Name,PreemptMode,MaxJobsPerUser,"
            "MaxSubmitJobsPerUser,MaxWall"
        ),
    ]
    raw = _invoke(runner, argv, timeout=30.0).stdout
    rows = [line.strip().split("|") for line in raw.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) < 5 or rows[0][0] != qos:
        raise ProtectedCapacityBuildError(
            "QOS query did not return one exact row"
        )
    mode = rows[0][1] or "cluster"
    max_jobs = rows[0][2] or "-"
    max_submit = rows[0][3] or "-"
    max_wall_raw = rows[0][4]
    max_wall = (
        "-"
        if not max_wall_raw
        or max_wall_raw.upper() in {"UNLIMITED", "INFINITE"}
        else str(_time_seconds(max_wall_raw))
    )
    return f"{qos}|{mode}|{max_jobs}|{max_submit}|{max_wall}\n"


def _capture_association(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    *,
    scheduler_user: str,
    qos: str,
) -> str:
    argv = [
        "sacctmgr",
        "-nP",
        "show",
        "assoc",
        f"user={scheduler_user}",
        "format=Cluster,Account,User,QOS,MaxJobs,MaxSubmitJobs",
    ]
    raw = _invoke(runner, argv, timeout=30.0).stdout
    candidates: list[list[str]] = []
    for line in raw.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 6 or fields[2] != scheduler_user:
            continue
        qoses = set(filter(None, fields[3].split(",")))
        if qos in qoses:
            candidates.append(fields)
    if len(candidates) != 1:
        raise ProtectedCapacityBuildError(
            "association query does not yield one exact user/QOS row"
        )
    row = candidates[0]
    return (
        f"{row[0]}|{row[1]}|{row[2]}|"
        f"{','.join(sorted(set(filter(None, row[3].split(',')))))}|"
        f"{row[4] or '-'}|{row[5] or '-'}\n"
    )


def _job_comment(token: str, role: str) -> str:
    return f"asys-s5-capacity:{token}:{role}"


def _chunk_sizes(total_tasks: int) -> tuple[int, ...]:
    if total_tasks <= 0:
        raise ProtectedCapacityBuildError("canary task count must be positive")
    full_chunks, remainder = divmod(total_tasks, MAX_ARRAY_TASKS)
    values = [MAX_ARRAY_TASKS] * full_chunks
    if remainder:
        values.append(remainder)
    return tuple(values)


def _job_element_accounting(
    active_server_job_elements: int,
) -> dict[str, int]:
    residual_held = (
        TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS
        - active_server_job_elements
        - WARM_TURNOVER_JOB_ELEMENTS
    )
    if (
        type(active_server_job_elements) is not int
        or active_server_job_elements < BASE_ACTIVE_SERVER_JOB_ELEMENTS
        or residual_held <= 0
        or active_server_job_elements + WARM_TURNOVER_JOB_ELEMENTS
        + residual_held
        != TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS
        or CLIENT_JOB_ELEMENTS + TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS
        != EXPECTED_TOTAL_JOB_ELEMENTS
    ):
        raise ProtectedCapacityBuildError(
            "protected-capacity residual reserve formula is invalid"
        )
    return {
        "cell_job_elements": CLIENT_JOB_ELEMENTS,
        "active_server_job_elements": active_server_job_elements,
        "warm_turnover_job_elements": WARM_TURNOVER_JOB_ELEMENTS,
        "controller_monitor_other_held_job_elements": residual_held,
        "total_non_cell_reserve_job_elements": (
            TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS
        ),
        "total_canary_job_elements": EXPECTED_TOTAL_JOB_ELEMENTS,
    }


def _chunk_key(role: str, chunk_index: int) -> str:
    return f"{role}:{chunk_index:03d}"


def _chunk_comment(token: str, role: str, chunk_index: int) -> str:
    return f"{_job_comment(token, role)}:{chunk_index:03d}"


def _render_script(
    *,
    root: Path,
    token: str,
    role: str,
    chunk_index: int,
    chunk_tasks: int,
    cpus_per_task: int,
    memory_mib_per_task: int,
    gpus_per_task: int,
    time_limit: str,
    shape_id: str,
    partition: str,
    qos: str,
) -> str:
    gres = (
        f"#SBATCH --gres=gpu:a100:{gpus_per_task}\n"
        if gpus_per_task > 0
        else ""
    )
    ready = root / READY_DIRECTORY / role / f"{chunk_index:03d}"
    release = root / RELEASE_FILENAME
    receipt_format = (
        '{"array_job_id":"%s","plan_id":"%s","role":"%s",'
        '"script_sha256":"%s","shape_id":"%s","task_id":%s,"token":"%s"}'
    )
    return (
        "#!/bin/bash\n"
        f"#SBATCH --job-name=asys-s5-cap-{role.replace('_', '-')}\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --qos={qos}\n"
        f"#SBATCH --cpus-per-task={cpus_per_task}\n"
        f"#SBATCH --mem={memory_mib_per_task}M\n"
        f"#SBATCH --time={time_limit}\n"
        "#SBATCH --no-requeue\n"
        f"#SBATCH --array=0-{chunk_tasks - 1}%{chunk_tasks}\n"
        f"{gres}"
        "set -euo pipefail\n"
        "[[ \"${ASYS_CAPACITY_PLAN_ID:-}\" =~ ^[0-9a-f]{64}$ ]]\n"
        f"readonly ASYS_CAPACITY_INTENT_TOKEN={_shell_quote(token)}\n"
        "script_sha256=\"$(sha256sum -- \"$0\" | awk '{print $1}')\"\n"
        f"mkdir -p -- {_shell_quote(str(ready))}\n"
        "receipt=\"$(printf "
        f"{_shell_quote(receipt_format)}"
        " \"${SLURM_ARRAY_JOB_ID}\" \"${ASYS_CAPACITY_PLAN_ID}\" "
        f"{_shell_quote(role)} \"${{script_sha256}}\" "
        f"{_shell_quote(shape_id)} \"${{SLURM_ARRAY_TASK_ID}}\" "
        "\"${ASYS_CAPACITY_INTENT_TOKEN}\")\"\n"
        f"receipt_path={_shell_quote(str(ready))}/\"${{SLURM_ARRAY_TASK_ID}}\"\n"
        "if [[ -e \"${receipt_path}\" ]]; then\n"
        "  [[ \"$(cat -- \"${receipt_path}\")\" == \"${receipt}\" ]]\n"
        "else\n"
        "  receipt_tmp=\"${receipt_path}.tmp.${SLURM_JOB_ID}\"\n"
        "  printf '%s\\n' \"${receipt}\" > \"${receipt_tmp}\"\n"
        "  chmod 0444 \"${receipt_tmp}\"\n"
        "  mv -n -- \"${receipt_tmp}\" \"${receipt_path}\"\n"
        "  rm -f -- \"${receipt_tmp}\"\n"
        "  [[ \"$(cat -- \"${receipt_path}\")\" == \"${receipt}\" ]]\n"
        "fi\n"
        f"while [[ ! -f {_shell_quote(str(release))} ]]; do sleep 2; done\n"
    )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_plan(
    *,
    root: Path,
    release_git_commit: str,
    release_tag_object: str,
    partition: str,
    qos: str,
    scheduler_user: str,
    token: str,
    fleet_contract_path: Path,
    fleet_contract_sha256: str,
    effective_fleet_contract_path: Path,
    effective_fleet_contract_sha256: str,
    additive_overlay_contract_path: Path,
    additive_overlay_contract_sha256: str,
    static_feasibility_certificate_path: Path,
    static_feasibility_certificate_sha256: str,
    static_feasibility_certificate_id: str,
    capacity_generation: int,
    model_contract_path: Path,
    model_contract_sha256: str,
    release_worktree: Path,
    identity_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    if (
        _GIT_RE.fullmatch(release_git_commit) is None
        or _GIT_RE.fullmatch(release_tag_object) is None
        or _SAFE_NAME_RE.fullmatch(partition) is None
        or _SAFE_NAME_RE.fullmatch(qos) is None
        or _SAFE_NAME_RE.fullmatch(scheduler_user) is None
        or re.fullmatch(r"[0-9a-f]{32}", token) is None
    ):
        raise ProtectedCapacityBuildError(
            "release, placement, user, or token identity is unsafe"
        )
    release_identity = _verify_release_checkout(
        release_worktree=release_worktree,
        release_git_commit=release_git_commit,
        release_tag_object=release_tag_object,
        runner=identity_runner,
    )
    worktree = Path(release_identity["release_worktree"])
    expected_fleet = _canonical_existing_path(
        worktree / "configs" / "schema5_fleet.v1.json",
        description="tagged fleet contract",
        kind="file",
    )
    expected_models = _canonical_existing_path(
        worktree / "configs" / "model_contracts.v1.json",
        description="tagged model contract",
        kind="file",
    )
    supplied_fleet = _canonical_existing_path(
        fleet_contract_path,
        description="supplied fleet contract",
        kind="file",
    )
    supplied_models = _canonical_existing_path(
        model_contract_path,
        description="supplied model contract",
        kind="file",
    )
    if supplied_fleet != expected_fleet or supplied_models != expected_models:
        raise ProtectedCapacityBuildError(
            "capacity contracts must be the exact tagged release authorities"
        )
    base_fleet = _load_frozen_fleet(
        fleet_contract_path=supplied_fleet,
        fleet_contract_sha256=fleet_contract_sha256,
        model_contract_path=supplied_models,
        model_contract_sha256=model_contract_sha256,
        allow_capacity_layout=False,
    )
    effective_fleet = _load_frozen_fleet(
        fleet_contract_path=effective_fleet_contract_path,
        fleet_contract_sha256=effective_fleet_contract_sha256,
        model_contract_path=supplied_models,
        model_contract_sha256=model_contract_sha256,
        allow_capacity_layout=True,
    )
    certificate = _validate_effective_capacity_authority(
        base_fleet=base_fleet,
        effective_fleet=effective_fleet,
        additive_overlay_path=additive_overlay_contract_path,
        additive_overlay_sha256=additive_overlay_contract_sha256,
        static_feasibility_certificate_path=(
            static_feasibility_certificate_path
        ),
        static_feasibility_certificate_sha256=(
            static_feasibility_certificate_sha256
        ),
        static_feasibility_certificate_id=(
            static_feasibility_certificate_id
        ),
        capacity_generation=capacity_generation,
        release_git_commit=release_git_commit,
        source_tree_sha256=release_identity["source_tree_sha256"],
        dispatcher_source_sha256=release_identity[
            "dispatcher_source_sha256"
        ],
        qualification_runner_source_sha256=release_identity[
            "qualification_runner_source_sha256"
        ],
    )
    root = _lexical_absolute(root)
    if root.exists() or root.is_symlink():
        _canonical_existing_path(
            root,
            description="capacity build root",
            kind="directory",
        )
    else:
        _canonical_existing_path(
            root.parent,
            description="capacity build parent",
            kind="directory",
        )
    active_server_job_elements = len(effective_fleet.replicas)
    residual_held_reserve = (
        TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS
        - active_server_job_elements
        - WARM_TURNOVER_JOB_ELEMENTS
    )
    accounting = _job_element_accounting(active_server_job_elements)
    role_shapes: dict[str, list[dict[str, Any]]] = {
        "server_active": _active_shapes(effective_fleet),
        "server_warm": _warm_shapes(),
        "client": [
            {
                "shape_id": f"client-{index:03d}",
                "serving_profile": "client",
                "tasks": tasks,
                "cpus": tasks,
                "memory_mib": tasks * 4096,
                "gpus": 0,
                "time_limit": CLIENT_TIME_LIMIT,
            }
            for index, tasks in enumerate(_chunk_sizes(CLIENT_JOB_ELEMENTS))
        ],
        "reserve": [
            {
                "shape_id": f"reserve-{index:03d}",
                "serving_profile": "reserve",
                "tasks": tasks,
                "cpus": tasks,
                "memory_mib": tasks * 1024,
                "gpus": 0,
                "time_limit": CLIENT_TIME_LIMIT,
            }
            for index, tasks in enumerate(
                _chunk_sizes(residual_held_reserve)
            )
        ],
    }
    active_shapes = role_shapes["server_active"]
    if (
        len(active_shapes) != certificate.effective_logical_replicas
        or sum(int(row["gpus"]) for row in active_shapes)
        != certificate.effective_active_gpus
        or {
            _time_seconds(str(row["time_limit"])) for row in active_shapes
        }
        != {SERVER_TIME_LIMIT_SECONDS}
    ):
        raise ProtectedCapacityBuildError(
            "effective active fleet differs from its certified resource topology"
        )
    scripts: dict[str, str] = {}
    role_chunks: dict[str, list[dict[str, Any]]] = {}
    for role in ROLE_ORDER:
        chunks: list[dict[str, Any]] = []
        for chunk_index, shape in enumerate(role_shapes[role]):
            chunk_tasks = int(shape["tasks"])
            key = _chunk_key(role, chunk_index)
            if (
                chunk_tasks < 1
                or chunk_tasks > MAX_ARRAY_TASKS
                or int(shape["cpus"]) % chunk_tasks
                or int(shape["memory_mib"]) % chunk_tasks
                or int(shape["gpus"]) % chunk_tasks
                or _time_seconds(str(shape["time_limit"]))
                != (
                    SERVER_TIME_LIMIT_SECONDS
                    if role in {"server_active", "server_warm"}
                    else CLIENT_TIME_LIMIT_SECONDS
                )
            ):
                raise ProtectedCapacityBuildError(
                    f"canary shape {shape['shape_id']} cannot be represented "
                    "as an exact bounded array"
                )
            script = _render_script(
                root=root,
                token=token,
                role=role,
                chunk_index=chunk_index,
                chunk_tasks=chunk_tasks,
                cpus_per_task=int(shape["cpus"]) // chunk_tasks,
                memory_mib_per_task=int(shape["memory_mib"]) // chunk_tasks,
                gpus_per_task=int(shape["gpus"]) // chunk_tasks,
                time_limit=str(shape["time_limit"]),
                shape_id=str(shape["shape_id"]),
                partition=partition,
                qos=qos,
            )
            scripts[key] = script
            chunks.append(
                {
                    "key": key,
                    "index": chunk_index,
                    "tasks": chunk_tasks,
                    "cpus": int(shape["cpus"]),
                    "memory_mib": int(shape["memory_mib"]),
                    "gpus": int(shape["gpus"]),
                    "time_limit": str(shape["time_limit"]),
                    "shape_id": str(shape["shape_id"]),
                    "serving_profile": str(shape["serving_profile"]),
                    "comment": _chunk_comment(token, role, chunk_index),
                    "script": str(
                        (root / "sbatch" / f"{key.replace(':', '-')}.sbatch").resolve()
                    ),
                    "script_sha256": sha256_bytes(script.encode("utf-8")),
                }
            )
        role_chunks[role] = chunks
    role_totals = {
        role: sum(int(row["tasks"]) for row in role_shapes[role])
        for role in ROLE_ORDER
    }
    expected_role_totals = {
        "server_active": active_server_job_elements,
        "server_warm": WARM_TURNOVER_JOB_ELEMENTS,
        "client": CLIENT_JOB_ELEMENTS,
        "reserve": residual_held_reserve,
    }
    if (
        role_totals != expected_role_totals
        or sum(role_totals.values()) != EXPECTED_TOTAL_JOB_ELEMENTS
    ):
        raise ProtectedCapacityBuildError(
            "protected-capacity role totals do not realize the exact 448-element "
            "contract"
        )
    plan: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "release_id": publisher.RELEASE_ID,
        "release_tag": publisher.RELEASE_TAG,
        "release_git_commit": release_git_commit,
        "release_tag_object": release_tag_object,
        "chain_namespace": publisher.CHAIN_NAMESPACE,
        "root": str(root),
        "partition": partition,
        "qos": qos,
        "scheduler_user": scheduler_user,
        "token": token,
        "capacity_generation": capacity_generation,
        "base_fleet_contract_path": str(base_fleet.path),
        "base_fleet_contract_sha256": base_fleet.sha256,
        "effective_fleet_contract_path": str(effective_fleet.path),
        "effective_fleet_contract_sha256": effective_fleet.sha256,
        "additive_overlay_contract_path": str(effective_fleet.path),
        "additive_overlay_contract_sha256": effective_fleet.sha256,
        "static_feasibility_certificate": {
            "path": str(certificate.path),
            "sha256": certificate.sha256,
            "certificate_id": certificate.certificate_id,
        },
        # Compatibility aliases remain exact-effective, never base-only.
        "fleet_contract_path": str(effective_fleet.path),
        "fleet_contract_sha256": effective_fleet.sha256,
        "model_contract_path": str(supplied_models),
        "model_contract_sha256": model_contract_sha256,
        **release_identity,
        "base_active_logical_replicas": len(base_fleet.replicas),
        "base_active_gpus": sum(
            row.gpus_per_replica for row in base_fleet.replicas
        ),
        "base_active_topology": _active_topology(
            _active_shapes(base_fleet)
        ),
        "base_active_topology_sha256": sha256_bytes(
            canonical_bytes(_active_topology(_active_shapes(base_fleet)))
        ),
        "additive_reserved_logical_replicas": (
            len(effective_fleet.replicas) - len(base_fleet.replicas)
        ),
        "additive_reserved_gpus": certificate.additive_allocated_gpus,
        "additive_reserved_tp1_replicas": (
            certificate.additive_tp1_logical_replicas
        ),
        "additive_reserved_tp2_replicas": (
            certificate.additive_tp2_logical_replicas
        ),
        "additive_reserved_topology": [
            row
            for row in _active_topology(active_shapes)
            if row["shape_id"]
            not in {replica.replica_id for replica in base_fleet.replicas}
        ],
        "effective_active_logical_replicas": len(
            effective_fleet.replicas
        ),
        "effective_active_gpus": sum(
            row.gpus_per_replica for row in effective_fleet.replicas
        ),
        "effective_active_topology": _active_topology(active_shapes),
        "retained_warm_turnover_topology": _active_topology(
            role_shapes["server_warm"]
        ),
        "attested_total_gpus": (
            sum(row.gpus_per_replica for row in effective_fleet.replicas)
            + WARM_TURNOVER_GPUS
        ),
        "active_fleet_topology_sha256": sha256_bytes(
            canonical_bytes(_active_topology(active_shapes))
        ),
        "expected_total_job_elements": EXPECTED_TOTAL_JOB_ELEMENTS,
        "running_scientific_jobs": (
            CLIENT_JOB_ELEMENTS
            + active_server_job_elements
            + WARM_TURNOVER_JOB_ELEMENTS
        ),
        "job_element_accounting": accounting,
        "roles": {
            role: {
                "tasks": sum(int(row["tasks"]) for row in role_shapes[role]),
                "cpus": sum(int(row["cpus"]) for row in role_shapes[role]),
                "memory_mib": sum(
                    int(row["memory_mib"]) for row in role_shapes[role]
                ),
                "gpus": sum(int(row["gpus"]) for row in role_shapes[role]),
                "held": ROLE_HELD[role],
                "chunks": role_chunks[role],
            }
            for role in ROLE_ORDER
        },
    }
    plan["plan_id"] = sha256_bytes(canonical_bytes(plan))
    plan["_scripts"] = scripts
    return plan


def _public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "_scripts"}


def prepare(
    plan: Mapping[str, Any],
    *,
    apply: bool,
) -> dict[str, Any]:
    public = _public_plan(plan)
    root = Path(str(public["root"]))
    if not apply:
        return {"status": "dry_run", "plan": public}
    if root.is_symlink():
        raise ProtectedCapacityBuildError("build root cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    intent_path = root / INTENT_FILENAME
    if not intent_path.exists():
        permitted_publication_names = {
            f".{INTENT_FILENAME}.publish.lock",
        }
        existing = sorted(
            path.name
            for path in root.iterdir()
            if path.name not in permitted_publication_names
            and not (
                path.name.startswith(f".{INTENT_FILENAME}.")
                and path.name.endswith(".publishing")
            )
        )
        if existing:
            raise ProtectedCapacityBuildError(
                f"new build root is not empty: {existing[:8]}"
            )
        _publish_sealed_once(intent_path, public)
    elif _load_plan(root) != public:
        raise ProtectedCapacityBuildError(
            "existing protected-capacity build intent conflicts"
        )
    scripts = plan.get("_scripts")
    if not isinstance(scripts, Mapping):
        raise ProtectedCapacityBuildError("prepared plan lacks script bytes")
    script_root = root / "sbatch"
    script_root.mkdir(exist_ok=True)
    for key, script in sorted(scripts.items()):
        path = script_root / f"{key.replace(':', '-')}.sbatch"
        payload = str(script).encode("utf-8")
        if path.exists():
            if path.is_symlink() or path.read_bytes() != payload:
                raise ProtectedCapacityBuildError(
                    f"immutable {key} canary script conflicts"
                )
        else:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o444,
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        path.chmod(0o444)
    ledger_path = root / LEDGER_FILENAME
    if not ledger_path.exists():
        jobs: dict[str, dict[str, Any]] = {}
        for role in ROLE_ORDER:
            role_record = public["roles"][role]
            for chunk in role_record["chunks"]:
                key = str(chunk["key"])
                jobs[key] = {
                    "state": "prepared",
                    "role": role,
                    "chunk_index": int(chunk["index"]),
                    "tasks": int(chunk["tasks"]),
                    "cpus": int(chunk["cpus"]),
                    "memory_mib": int(chunk["memory_mib"]),
                    "gpus": int(chunk["gpus"]),
                    "time_limit": str(chunk["time_limit"]),
                    "shape_id": str(chunk["shape_id"]),
                    "serving_profile": str(chunk["serving_profile"]),
                    "held": bool(role_record["held"]),
                    "job_id": None,
                    "comment": chunk["comment"],
                    "script": chunk["script"],
                    "script_sha256": chunk["script_sha256"],
                    "submitted_at": None,
                    "submission_intent_at": None,
                    "submission_receipt_at": None,
                    "attempt": 0,
                    "last_error": None,
                    "cleanup_state": "pending",
                    "cleanup_receipt": None,
                }
        ledger = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "plan_id": public["plan_id"],
            "occupancy_preflight": None,
            "jobs": jobs,
        }
        _atomic_json(ledger_path, ledger)
    return {"status": "prepared", "plan": public}


def _load_plan(root: Path) -> dict[str, Any]:
    path = root / INTENT_FILENAME
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) & 0o222
    ):
        raise ProtectedCapacityBuildError(
            "build intent is missing, mutable, or unsafe"
        )
    plan = _read_json(path, description="build intent", sealed=True)
    plan_id = plan.get("plan_id")
    identity = dict(plan)
    identity.pop("plan_id", None)
    if (
        _SHA256_RE.fullmatch(str(plan_id)) is None
        or sha256_bytes(canonical_bytes(identity)) != plan_id
        or Path(str(plan.get("root", ""))).resolve() != root.resolve()
    ):
        raise ProtectedCapacityBuildError(
            "build intent self-hash/root binding is invalid"
        )
    return plan


def _validate_occupancy_preflight(
    value: Any,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    expected_running = plan.get("running_scientific_jobs")
    expected_accounting = plan.get("job_element_accounting")
    if (
        type(expected_running) is not int
        or expected_running < CLIENT_JOB_ELEMENTS
        or not isinstance(expected_accounting, Mapping)
    ):
        raise ProtectedCapacityBuildError(
            "build intent lacks dynamic scientific job accounting"
        )
    required = {
        "protocol",
        "plan_id",
        "observation_interval_seconds",
        "scheduler_account",
        "scientific_qos",
        "association_max_jobs",
        "qos_max_jobs",
        "effective_max_jobs",
        "association_max_submit_jobs",
        "qos_max_submit_jobs",
        "effective_max_submit_jobs",
        "existing_job_elements",
        "existing_association_job_elements",
        "existing_qos_job_elements",
        "existing_association_running_job_elements",
        "existing_qos_running_job_elements",
        "required_new_running_job_elements",
        "required_new_job_elements",
        "first_observation",
        "second_observation",
        "preflight_id",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("protocol")
        != "schema5-v1.2-r3-protected-capacity-occupancy-preflight-v3"
        or value.get("plan_id") != plan.get("plan_id")
        or value.get("scientific_qos") != plan.get("qos")
        or not isinstance(value.get("scheduler_account"), str)
        or _PLACEMENT_NAME_RE.fullmatch(value["scheduler_account"]) is None
        or value.get("required_new_running_job_elements")
        != expected_running
        or value.get("required_new_job_elements")
        != EXPECTED_TOTAL_JOB_ELEMENTS
        or value.get("preflight_id")
        != sha256_bytes(
            canonical_bytes(
                {key: item for key, item in value.items() if key != "preflight_id"}
            )
        )
    ):
        raise ProtectedCapacityBuildError(
            "build ledger occupancy preflight is malformed"
        )
    interval = value.get("observation_interval_seconds")
    association_max_jobs = value.get("association_max_jobs")
    qos_max_jobs = value.get("qos_max_jobs")
    effective_max_jobs = value.get("effective_max_jobs")
    association_limit = value.get("association_max_submit_jobs")
    qos_limit = value.get("qos_max_submit_jobs")
    effective_limit = value.get("effective_max_submit_jobs")
    existing = value.get("existing_job_elements")
    existing_association = value.get(
        "existing_association_job_elements"
    )
    existing_qos = value.get("existing_qos_job_elements")
    existing_association_running = value.get(
        "existing_association_running_job_elements"
    )
    existing_qos_running = value.get("existing_qos_running_job_elements")
    if (
        not isinstance(interval, (int, float))
        or isinstance(interval, bool)
        or not math.isfinite(float(interval))
        or float(interval) < 0
        or not isinstance(association_limit, int)
        or isinstance(association_limit, bool)
        or association_limit < 1
        or (
            qos_limit is not None
            and (
                not isinstance(qos_limit, int)
                or isinstance(qos_limit, bool)
                or qos_limit < 1
            )
        )
        or effective_limit
        != min(
            association_limit,
            association_limit if qos_limit is None else qos_limit,
        )
        or not isinstance(existing, int)
        or isinstance(existing, bool)
        or existing < 0
        or not isinstance(existing_association, int)
        or isinstance(existing_association, bool)
        or existing_association < 0
        or not isinstance(existing_qos, int)
        or isinstance(existing_qos, bool)
        or existing_qos < 0
        or existing_qos > existing_association
        or existing_association > existing
        or existing_association + EXPECTED_TOTAL_JOB_ELEMENTS
        > association_limit
        or (
            qos_limit is not None
            and existing_qos + EXPECTED_TOTAL_JOB_ELEMENTS > qos_limit
        )
        or (
            association_max_jobs is not None
            and (
                not isinstance(association_max_jobs, int)
                or isinstance(association_max_jobs, bool)
                or association_max_jobs < 1
            )
        )
        or (
            qos_max_jobs is not None
            and (
                not isinstance(qos_max_jobs, int)
                or isinstance(qos_max_jobs, bool)
                or qos_max_jobs < 1
            )
        )
        or effective_max_jobs
        != (
            None
            if association_max_jobs is None and qos_max_jobs is None
            else min(
                limit
                for limit in (association_max_jobs, qos_max_jobs)
                if limit is not None
            )
        )
        or not isinstance(existing_association_running, int)
        or isinstance(existing_association_running, bool)
        or existing_association_running < 0
        or not isinstance(existing_qos_running, int)
        or isinstance(existing_qos_running, bool)
        or existing_qos_running < 0
        or existing_qos_running > existing_association_running
        or existing_association_running > existing
        or (
            association_max_jobs is not None
            and existing_association_running
            + expected_running
            > association_max_jobs
        )
        or (
            qos_max_jobs is not None
            and existing_qos_running + expected_running
            > qos_max_jobs
        )
    ):
        raise ProtectedCapacityBuildError(
            "build ledger occupancy limits are invalid"
        )
    observations: list[dict[str, Any]] = []
    for field in ("first_observation", "second_observation"):
        observation = value.get(field)
        if (
            not isinstance(observation, dict)
            or set(observation)
            != {
                "observed_at",
                "job_elements",
                "jobs",
                "squeue_sha256",
                "sacct_sha256",
                "observation_id",
            }
            or not isinstance(observation.get("observed_at"), (int, float))
            or isinstance(observation.get("observed_at"), bool)
            or not math.isfinite(float(observation["observed_at"]))
            or not isinstance(observation.get("jobs"), list)
            or observation.get("job_elements") != len(observation["jobs"])
            or _SHA256_RE.fullmatch(str(observation.get("squeue_sha256", "")))
            is None
            or _SHA256_RE.fullmatch(str(observation.get("sacct_sha256", "")))
            is None
            or observation.get("observation_id")
            != sha256_bytes(
                canonical_bytes(
                    {
                        key: item
                        for key, item in observation.items()
                        if key != "observation_id"
                    }
                )
            )
        ):
            raise ProtectedCapacityBuildError(
                f"build ledger {field} is malformed"
            )
        identities: set[str] = set()
        for row in observation["jobs"]:
            if (
                not isinstance(row, dict)
                or set(row)
                != {"job_id", "state", "account", "qos", "comment"}
                or _JOB_ID_RE.fullmatch(str(row.get("job_id", ""))) is None
                or row["job_id"] in identities
                or not isinstance(row.get("state"), str)
                or not isinstance(row.get("account"), str)
                or not isinstance(row.get("qos"), str)
                or not isinstance(row.get("comment"), str)
            ):
                raise ProtectedCapacityBuildError(
                    f"build ledger {field} job row is malformed"
                )
            identities.add(row["job_id"])
        observations.append(observation)
    if (
        observations[0]["jobs"] != observations[1]["jobs"]
        or observations[0]["job_elements"] != existing
        or observations[1]["job_elements"] != existing
        or sum(
            row["account"] == value["scheduler_account"]
            for row in observations[1]["jobs"]
        )
        != existing_association
        or sum(
            row["account"] == value["scheduler_account"]
            and row["qos"] == value["scientific_qos"]
            for row in observations[1]["jobs"]
        )
        != existing_qos
        or sum(
            row["state"] in MAX_JOBS_CONSUMING_STATES
            and row["account"] == value["scheduler_account"]
            for row in observations[1]["jobs"]
        )
        != existing_association_running
        or sum(
            row["state"] in MAX_JOBS_CONSUMING_STATES
            and row["account"] == value["scheduler_account"]
            and row["qos"] == value["scientific_qos"]
            for row in observations[1]["jobs"]
        )
        != existing_qos_running
        or float(observations[1]["observed_at"])
        - float(observations[0]["observed_at"])
        < float(interval)
    ):
        raise ProtectedCapacityBuildError(
            "build ledger occupancy observations are not stable and separated"
        )
    return value


def _load_ledger(root: Path, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    path = root / LEDGER_FILENAME
    if path.is_symlink() or not path.is_file():
        raise ProtectedCapacityBuildError("build ledger is missing or unsafe")
    ledger = _read_json(path, description="build ledger", sealed=False)
    if set(ledger) != {
        "schema_version",
        "protocol",
        "plan_id",
        "occupancy_preflight",
        "jobs",
    } or (
        ledger.get("schema_version") != 1
        or ledger.get("protocol") != PROTOCOL
        or ledger.get("plan_id") != plan.get("plan_id")
        or not isinstance(ledger.get("jobs"), dict)
    ):
        raise ProtectedCapacityBuildError("build ledger envelope is malformed")
    occupancy_preflight = ledger.get("occupancy_preflight")
    if occupancy_preflight is not None:
        _validate_occupancy_preflight(occupancy_preflight, plan=plan)
    expected_chunks: dict[str, tuple[str, Mapping[str, Any], Mapping[str, Any]]] = {}
    roles = plan.get("roles")
    if not isinstance(roles, Mapping):
        raise ProtectedCapacityBuildError("build intent roles are malformed")
    for role, role_record in roles.items():
        if role not in ROLE_ORDER or not isinstance(role_record, Mapping):
            raise ProtectedCapacityBuildError("build intent role is malformed")
        chunks = role_record.get("chunks")
        if not isinstance(chunks, list):
            raise ProtectedCapacityBuildError("build intent chunks are malformed")
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise ProtectedCapacityBuildError(
                    "build intent chunk is malformed"
                )
            expected_chunks[str(chunk["key"])] = (role, role_record, chunk)
    if set(ledger["jobs"]) != set(expected_chunks):
        raise ProtectedCapacityBuildError(
            "build ledger job set differs from immutable intent"
        )
    required_job_fields = {
        "state",
        "role",
        "chunk_index",
        "tasks",
        "cpus",
        "memory_mib",
        "gpus",
        "time_limit",
        "shape_id",
        "serving_profile",
        "held",
        "job_id",
        "comment",
        "script",
        "script_sha256",
        "submitted_at",
        "submission_intent_at",
        "submission_receipt_at",
        "attempt",
        "last_error",
        "cleanup_state",
        "cleanup_receipt",
    }
    for key, record in ledger["jobs"].items():
        if not isinstance(record, dict) or set(record) != required_job_fields:
            raise ProtectedCapacityBuildError(
                f"build ledger job {key} has invalid fields"
            )
        role, role_record, chunk = expected_chunks[key]
        immutable_expected = {
            "role": role,
            "chunk_index": int(chunk["index"]),
            "tasks": int(chunk["tasks"]),
            "cpus": int(chunk["cpus"]),
            "memory_mib": int(chunk["memory_mib"]),
            "gpus": int(chunk["gpus"]),
            "time_limit": str(chunk["time_limit"]),
            "shape_id": str(chunk["shape_id"]),
            "serving_profile": str(chunk["serving_profile"]),
            "held": bool(role_record["held"]),
            "comment": str(chunk["comment"]),
            "script": str(chunk["script"]),
            "script_sha256": str(chunk["script_sha256"]),
        }
        if any(record[field] != value for field, value in immutable_expected.items()):
            raise ProtectedCapacityBuildError(
                f"build ledger job {key} differs from immutable intent"
            )
        if record["state"] not in {"prepared", "submitting", "submitted"}:
            raise ProtectedCapacityBuildError(
                f"build ledger job {key} has invalid state"
            )
        if (
            not isinstance(record["attempt"], int)
            or isinstance(record["attempt"], bool)
            or record["attempt"] < 0
            or record["cleanup_state"]
            not in {"pending", "released", "cancelling", "cancelled"}
            or (
                record["cleanup_receipt"] is not None
                and not isinstance(record["cleanup_receipt"], dict)
            )
        ):
            raise ProtectedCapacityBuildError(
                f"build ledger job {key} mutable state is malformed"
            )
        if (
            record["state"] == "submitting"
            and not isinstance(record["submission_intent_at"], (int, float))
        ):
            raise ProtectedCapacityBuildError(
                f"build ledger job {key} lacks submission intent time"
            )
        job_id = record["job_id"]
        if (
            job_id is not None
            and (not isinstance(job_id, str) or not job_id.isdigit())
        ):
            raise ProtectedCapacityBuildError(
                f"build ledger job {key} has invalid job ID"
            )
    return ledger


def _capture_current_user_occupancy(
    *,
    plan: Mapping[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    observed_at: float,
) -> dict[str, Any]:
    """Join complete live squeue identities to exact-ID sacct provenance."""

    user = str(plan["scheduler_user"])
    queue_argv = [
        "squeue",
        "-h",
        "-r",
        "-u",
        user,
        "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,RESIZING,SUSPENDED",
        "-o",
        "%i|%T|%a|%q|%k",
    ]
    queue_raw = _invoke(runner, queue_argv, timeout=60.0).stdout
    active_states = {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "RESIZING",
        "SUSPENDED",
    }
    queued: dict[str, dict[str, str]] = {}
    for line_number, raw in enumerate(queue_raw.splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.rstrip("\n").split("|")
        if len(fields) != 5:
            raise ProtectedCapacityBuildError(
                f"occupancy squeue row {line_number} is malformed"
            )
        job_id, state, account, qos, comment = (
            field.strip() for field in fields
        )
        if (
            _JOB_ID_RE.fullmatch(job_id) is None
            or state.upper() not in active_states
            or _PLACEMENT_NAME_RE.fullmatch(account) is None
            or _PLACEMENT_NAME_RE.fullmatch(qos) is None
            or job_id in queued
        ):
            raise ProtectedCapacityBuildError(
                f"occupancy squeue row {line_number} is ambiguous"
            )
        queued[job_id] = {
            "job_id": job_id,
            "state": state.upper(),
            "account": account,
            "qos": qos,
            "comment": comment,
        }

    # Query accounting by the exact current squeue identities, not by an arbitrary
    # lookback window: serving jobs may legitimately survive longer than a week.  The
    # bounded chunks keep argv size portable at the full 448-element ceiling.
    accounting_captures: list[dict[str, Any]] = []
    queued_ids = sorted(queued)
    job_id_chunks = [
        queued_ids[index : index + 128]
        for index in range(0, len(queued_ids), 128)
    ]
    if not job_id_chunks:
        # A successful empty-current-day query establishes sacct availability while
        # complete squeue is the authoritative proof of current quiescence.
        accounting_argvs = [
            [
                "sacct",
                "-nP",
                "-X",
                "--array",
                "-u",
                user,
                "-S",
                datetime.fromtimestamp(
                    observed_at,
                    timezone.utc,
                ).astimezone().strftime("%Y-%m-%d"),
                "-o",
                "JobID,State,Account,QOS,Comment",
            ]
        ]
    else:
        accounting_argvs = [
            [
                "sacct",
                "-nP",
                "-X",
                "--array",
                "-j",
                ",".join(chunk),
                "-o",
                "JobID,State,Account,QOS,Comment",
            ]
            for chunk in job_id_chunks
        ]
    accounting_outputs: list[str] = []
    for accounting_argv in accounting_argvs:
        accounting_raw = _invoke(
            runner,
            accounting_argv,
            timeout=60.0,
        ).stdout
        accounting_outputs.append(accounting_raw)
        accounting_captures.append(
            {
                "argv": accounting_argv,
                "raw_output_sha256": sha256_bytes(
                    accounting_raw.encode("utf-8")
                ),
            }
        )

    accounted: dict[str, dict[str, str]] = {}
    for capture_index, accounting_raw in enumerate(accounting_outputs):
        for line_number, raw in enumerate(
            accounting_raw.splitlines(),
            start=1,
        ):
            if not raw.strip():
                continue
            fields = raw.rstrip("\n").split("|")
            if len(fields) != 5:
                raise ProtectedCapacityBuildError(
                    "occupancy sacct row "
                    f"{capture_index}:{line_number} is malformed"
                )
            job_id, state, account, qos, comment = (
                field.strip() for field in fields
            )
            if "." in job_id:
                continue
            if _JOB_ID_RE.fullmatch(job_id) is None:
                raise ProtectedCapacityBuildError(
                    "occupancy sacct row "
                    f"{capture_index}:{line_number} has an invalid job ID"
                )
            if (
                _PLACEMENT_NAME_RE.fullmatch(account) is None
                or _PLACEMENT_NAME_RE.fullmatch(qos) is None
            ):
                raise ProtectedCapacityBuildError(
                    "occupancy sacct row "
                    f"{capture_index}:{line_number} has invalid account/QOS"
                )
            normalized_state = state.upper().split()[0].rstrip("+")
            if normalized_state not in active_states:
                continue
            if job_id in accounted:
                raise ProtectedCapacityBuildError(
                    f"occupancy sacct duplicates active job {job_id}"
                )
            accounted[job_id] = {
                "job_id": job_id,
                "state": normalized_state,
                "account": account,
                "qos": qos,
                "comment": comment,
            }

    for job_id, queue_row in queued.items():
        accounting_row = accounted.get(job_id)
        if accounting_row is None:
            raise ProtectedCapacityBuildError(
                f"complete sacct truth omits live squeue job element {job_id}"
            )
        if (
            accounting_row["comment"]
            and queue_row["comment"]
            and accounting_row["comment"] != queue_row["comment"]
        ) or (
            accounting_row["account"] != queue_row["account"]
            or accounting_row["qos"] != queue_row["qos"]
        ):
            raise ProtectedCapacityBuildError(
                f"squeue/sacct occupancy provenance conflicts for {job_id}"
            )
    # sacct may retain an active array-parent record in addition to the task rows
    # returned by ``squeue -r``.  Every other active accounting-only identity means
    # current occupancy truth is incomplete.
    queue_array_parents = {
        job_id.split("_", 1)[0]
        for job_id in queued
        if "_" in job_id
    }
    unexplained = sorted(
        job_id
        for job_id in accounted
        if job_id not in queued and job_id not in queue_array_parents
    )
    if unexplained:
        raise ProtectedCapacityBuildError(
            "sacct reports active jobs absent from complete squeue truth: "
            + ", ".join(unexplained[:8])
        )
    jobs = [queued[job_id] for job_id in sorted(queued)]
    observation = {
        "observed_at": float(observed_at),
        "job_elements": len(jobs),
        "jobs": jobs,
        "squeue_sha256": sha256_bytes(queue_raw.encode("utf-8")),
        "sacct_sha256": sha256_bytes(
            canonical_bytes(accounting_captures)
        ),
    }
    observation["observation_id"] = sha256_bytes(canonical_bytes(observation))
    return observation


def _verify_submit_headroom_preflight(
    *,
    plan: Mapping[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    now: Callable[[], float],
    sleep: Callable[[float], None],
    observation_interval_seconds: float = (
        OCCUPANCY_OBSERVATION_INTERVAL_SECONDS
    ),
) -> dict[str, Any]:
    active_jobs = int(
        plan["job_element_accounting"]["active_server_job_elements"]
    )
    expected_accounting = _job_element_accounting(active_jobs)
    expected_running = int(plan["running_scientific_jobs"])
    if (
        plan.get("expected_total_job_elements")
        != EXPECTED_TOTAL_JOB_ELEMENTS
        or plan.get("job_element_accounting") != expected_accounting
    ):
        raise ProtectedCapacityBuildError(
            "immutable canary intent does not bind the exact job-element "
            "accounting"
        )
    qos_fields = _capture_qos(
        runner,
        qos=str(plan["qos"]),
    ).strip().split("|")
    association_fields = _capture_association(
        runner,
        scheduler_user=str(plan["scheduler_user"]),
        qos=str(plan["qos"]),
    ).strip().split("|")
    if len(qos_fields) != 5 or len(association_fields) != 6:
        raise ProtectedCapacityBuildError(
            "submit-headroom preflight returned malformed scheduler authority"
        )
    try:
        association_max_jobs = (
            None
            if association_fields[4] == "-"
            else int(association_fields[4])
        )
        association_limit = int(association_fields[5])
        qos_max_jobs = (
            None if qos_fields[2] == "-" else int(qos_fields[2])
        )
        qos_limit = (
            None if qos_fields[3] == "-" else int(qos_fields[3])
        )
        qos_max_wall = (
            None if qos_fields[4] == "-" else int(qos_fields[4])
        )
    except ValueError as exc:
        raise ProtectedCapacityBuildError(
            "submit-headroom preflight limits are not integers"
        ) from exc
    if (
        qos_max_wall is not None
        and qos_max_wall < SERVER_TIME_LIMIT_SECONDS
    ):
        raise ProtectedCapacityBuildError(
            "protected-capacity QOS walltime authority cannot sustain the "
            "24-hour scientific serving allocation"
        )
    if (
        not isinstance(observation_interval_seconds, (int, float))
        or isinstance(observation_interval_seconds, bool)
        or not math.isfinite(float(observation_interval_seconds))
        or float(observation_interval_seconds) < 0
    ):
        raise ProtectedCapacityBuildError(
            "occupancy observation interval is invalid"
        )
    first_at = float(now())
    first = _capture_current_user_occupancy(
        plan=plan,
        runner=runner,
        observed_at=first_at,
    )
    sleep(float(observation_interval_seconds))
    second_at = float(now())
    if (
        not math.isfinite(first_at)
        or not math.isfinite(second_at)
        or second_at - first_at
        < float(observation_interval_seconds)
    ):
        raise ProtectedCapacityBuildError(
            "protected-capacity occupancy observations were not separated by "
            f"{float(observation_interval_seconds):g} seconds"
        )
    second = _capture_current_user_occupancy(
        plan=plan,
        runner=runner,
        observed_at=second_at,
    )
    stable_fields = ("job_elements", "jobs")
    if any(first[field] != second[field] for field in stable_fields):
        raise ProtectedCapacityBuildError(
            "protected-capacity occupancy changed across the two complete "
            "scheduler observations"
        )
    effective_limit = min(
        association_limit,
        association_limit if qos_limit is None else qos_limit,
    )
    existing_job_elements = int(second["job_elements"])
    scheduler_account = association_fields[1]
    scientific_qos = str(plan["qos"])
    existing_association = sum(
        row["account"] == scheduler_account for row in second["jobs"]
    )
    existing_qos = sum(
        row["account"] == scheduler_account
        and row["qos"] == scientific_qos
        for row in second["jobs"]
    )
    existing_association_running = sum(
        row["state"] in MAX_JOBS_CONSUMING_STATES
        and row["account"] == scheduler_account
        for row in second["jobs"]
    )
    existing_qos_running = sum(
        row["state"] in MAX_JOBS_CONSUMING_STATES
        and row["account"] == scheduler_account
        and row["qos"] == scientific_qos
        for row in second["jobs"]
    )
    if (
        association_max_jobs is not None
        and existing_association_running + expected_running
        > association_max_jobs
    ) or (
        qos_max_jobs is not None
        and existing_qos_running + expected_running
        > qos_max_jobs
    ):
        raise ProtectedCapacityBuildError(
            "protected-capacity canary requires contemporaneous unrelated "
            f"running jobs plus {expected_running} scientific allocations to fit both "
            "association and QOS MaxJobs limits: "
            f"association_existing={existing_association_running}, "
            f"association_limit={association_max_jobs}, "
            f"qos_existing={existing_qos_running}, qos_limit={qos_max_jobs}"
        )
    if (
        existing_association + EXPECTED_TOTAL_JOB_ELEMENTS
        > association_limit
    ) or (
        qos_limit is not None
        and existing_qos + EXPECTED_TOTAL_JOB_ELEMENTS > qos_limit
    ):
        raise ProtectedCapacityBuildError(
            "protected-capacity canary requires same-account/QOS existing user "
            "job elements plus 448 scientific allocations to fit the residual "
            "submit limits"
        )
    receipt: dict[str, Any] = {
        "protocol": (
            "schema5-v1.2-r3-protected-capacity-occupancy-preflight-v3"
        ),
        "plan_id": str(plan["plan_id"]),
        "observation_interval_seconds": float(observation_interval_seconds),
        "scheduler_account": scheduler_account,
        "scientific_qos": scientific_qos,
        "association_max_jobs": association_max_jobs,
        "qos_max_jobs": qos_max_jobs,
        "effective_max_jobs": (
            None
            if association_max_jobs is None and qos_max_jobs is None
            else min(
                limit
                for limit in (association_max_jobs, qos_max_jobs)
                if limit is not None
            )
        ),
        "association_max_submit_jobs": association_limit,
        "qos_max_submit_jobs": qos_limit,
        "effective_max_submit_jobs": effective_limit,
        "existing_job_elements": existing_job_elements,
        "existing_association_job_elements": existing_association,
        "existing_qos_job_elements": existing_qos,
        "existing_association_running_job_elements": (
            existing_association_running
        ),
        "existing_qos_running_job_elements": existing_qos_running,
        "required_new_running_job_elements": (
            expected_running
        ),
        "required_new_job_elements": EXPECTED_TOTAL_JOB_ELEMENTS,
        "first_observation": first,
        "second_observation": second,
    }
    receipt["preflight_id"] = sha256_bytes(canonical_bytes(receipt))
    return receipt


def _submission_truth(
    *,
    plan: Mapping[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    since: float,
) -> dict[str, dict[str, set[str]]]:
    user = str(plan["scheduler_user"])
    queue_raw = _invoke(
        runner,
        ["squeue", "-h", "-u", user, "-o", "%A|%k"],
        timeout=60.0,
    ).stdout
    start = datetime.fromtimestamp(
        max(0.0, since - 300.0),
        timezone.utc,
    ).astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    accounting_raw = _invoke(
        runner,
        [
            "sacct",
            "-nP",
            "--array",
            "-u",
            user,
            "-S",
            start,
            "-o",
            "JobID,Comment,State",
        ],
        timeout=60.0,
    ).stdout
    comments = {
        str(chunk["comment"])
        for role in plan["roles"].values()
        for chunk in role["chunks"]
    }
    queue: dict[str, set[str]] = {comment: set() for comment in comments}
    accounted: dict[str, set[str]] = {comment: set() for comment in comments}
    terminal: dict[str, set[str]] = {comment: set() for comment in comments}
    for line in queue_raw.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 2:
            raise ProtectedCapacityBuildError(
                "complete squeue comment reconciliation output is malformed"
            )
        job_id, comment = fields
        if comment in comments:
            match = re.fullmatch(r"([0-9]+)(?:_[0-9]+)?", job_id)
            if match is None:
                raise ProtectedCapacityBuildError(
                    "squeue reconciliation returned an invalid job ID"
                )
            queue[comment].add(match.group(1))
    for line in accounting_raw.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 3:
            raise ProtectedCapacityBuildError(
                "complete sacct comment reconciliation output is malformed"
            )
        job_id, comment, state = fields
        if comment not in comments:
            continue
        match = re.fullmatch(r"([0-9]+)(?:_[0-9]+)?", job_id)
        if match is None:
            # Ignore batch/extern steps, but reject other nonempty identities.
            if "." in job_id:
                continue
            raise ProtectedCapacityBuildError(
                "sacct reconciliation returned an invalid job ID"
            )
        normalized_state = state.split("+", 1)[0]
        if normalized_state in {
            "PENDING",
            "RUNNING",
            "CONFIGURING",
            "COMPLETING",
        }:
            accounted[comment].add(match.group(1))
        else:
            terminal[comment].add(match.group(1))
    result: dict[str, dict[str, set[str]]] = {}
    for comment in comments:
        if queue[comment] != accounted[comment]:
            raise ProtectedCapacityBuildError(
                f"squeue/sacct reconciliation is incomplete for {comment}"
            )
        if queue[comment] & terminal[comment]:
            raise ProtectedCapacityBuildError(
                f"sacct reports live and terminal states for {comment}"
            )
        result[comment] = {
            "live": queue[comment],
            "terminal": terminal[comment],
        }
    return result


def _verify_reconciled_job(
    *,
    record: Mapping[str, Any],
    job_id: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> None:
    details = _invoke(
        runner,
        ["scontrol", "show", "job", "-o", job_id],
        timeout=60.0,
    ).stdout
    if (
        f"Comment={record['comment']}" not in details
        or "Requeue=0" not in details
    ):
        raise ProtectedCapacityBuildError(
            f"reconciled job {job_id} lacks exact comment/Requeue=0"
        )
    spooled = _invoke(
        runner,
        ["scontrol", "write", "batch_script", job_id, "-"],
        timeout=60.0,
    ).stdout.encode("utf-8")
    if sha256_bytes(spooled) != record["script_sha256"]:
        raise ProtectedCapacityBuildError(
            f"reconciled job {job_id} has the wrong spooled script"
        )


def _reconcile_submitting(
    root: Path,
    *,
    plan: Mapping[str, Any],
    ledger: dict[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    now: Callable[[], float],
    sleep: Callable[[float], None],
    observation_interval_seconds: float,
) -> dict[str, Any]:
    pending = {
        key: record
        for key, record in ledger["jobs"].items()
        if record["state"] == "submitting"
    }
    if not pending:
        return ledger
    since = min(
        float(record["submission_intent_at"])
        for record in pending.values()
        if isinstance(record.get("submission_intent_at"), (int, float))
    )
    first = _submission_truth(plan=plan, runner=runner, since=since)
    sleep(observation_interval_seconds)
    second = _submission_truth(plan=plan, runner=runner, since=since)
    if first != second:
        raise ProtectedCapacityBuildError(
            "submission reconciliation changed across complete observations"
        )
    ledger_path = root / LEDGER_FILENAME
    for key, record in sorted(pending.items()):
        truth = second[str(record["comment"])]
        candidates = truth["live"]
        terminal = truth["terminal"]
        if terminal:
            raise ProtectedCapacityBuildError(
                f"terminal comment-matched job(s) {sorted(terminal)} exist for "
                f"{key}; refusing silent resubmission"
            )
        if len(candidates) > 1:
            raise ProtectedCapacityBuildError(
                f"ambiguous duplicate jobs exist for immutable intent {key}"
            )
        if not candidates:
            record.update(
                {
                    "state": "prepared",
                    "job_id": None,
                    "submitted_at": None,
                    "submission_intent_at": None,
                    "submission_receipt_at": None,
                    "last_error": "no accepted job after two complete observations",
                }
            )
        else:
            job_id = next(iter(candidates))
            _verify_reconciled_job(
                record=record,
                job_id=job_id,
                runner=runner,
            )
            receipt_at = float(now())
            record.update(
                {
                    "state": "submitted",
                    "job_id": job_id,
                    "submitted_at": receipt_at,
                    "submission_receipt_at": receipt_at,
                    "last_error": None,
                }
            )
        _atomic_json(ledger_path, ledger)
    return _load_ledger(root, plan=plan)


def _submit_jobs(
    root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    now: Callable[[], float],
    sleep: Callable[[float], None],
    reconciliation_interval_seconds: float = 60.0,
    after_sbatch: Callable[[Mapping[str, Any], str], None] | None = None,
) -> dict[str, Any]:
    plan = _load_plan(root)
    ledger_path = root / LEDGER_FILENAME
    ledger = _load_ledger(root, plan=plan)
    jobs = ledger.get("jobs")
    if not isinstance(jobs, dict):
        raise ProtectedCapacityBuildError("build ledger jobs are malformed")
    ledger = _reconcile_submitting(
        root,
        plan=plan,
        ledger=ledger,
        runner=runner,
        now=now,
        sleep=sleep,
        observation_interval_seconds=reconciliation_interval_seconds,
    )
    jobs = ledger["jobs"]
    for key, record in sorted(jobs.items()):
        if not isinstance(record, dict):
            raise ProtectedCapacityBuildError(f"build ledger job {key} is malformed")
        role = str(record.get("role", ""))
        if role not in ROLE_ORDER or key != _chunk_key(
            role, int(record.get("chunk_index", -1))
        ):
            raise ProtectedCapacityBuildError(
                f"build ledger job identity is malformed: {key}"
            )
        if record["state"] == "submitted":
            continue
        if record["state"] == "submitting":
            raise ProtectedCapacityBuildError(
                f"{role} remains ambiguous after complete reconciliation"
            )
        intent_at = float(now())
        record["state"] = "submitting"
        record["attempt"] += 1
        record["submission_intent_at"] = intent_at
        record["submission_receipt_at"] = None
        record["last_error"] = None
        _atomic_json(ledger_path, ledger)
        argv = ["sbatch", "--parsable"]
        if record["held"]:
            argv.append("--hold")
        argv.extend(
            [
                f"--comment={record['comment']}",
                f"--export=ASYS_CAPACITY_PLAN_ID={plan['plan_id']}",
                str(record["script"]),
            ]
        )
        try:
            result = _invoke(runner, argv, timeout=60.0)
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise ProtectedCapacityBuildError(
                    f"sbatch returned invalid job ID for {role}"
                )
            if after_sbatch is not None:
                after_sbatch(record, job_id)
        except Exception as exc:
            record["last_error"] = str(exc)
            _atomic_json(ledger_path, ledger)
            raise
        record.update(
            {
                "state": "submitted",
                "job_id": job_id,
                "submitted_at": float(now()),
                "submission_receipt_at": float(now()),
            }
        )
        _atomic_json(ledger_path, ledger)
    return ledger


def _ready_count(
    root: Path,
    role: str,
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> int:
    directory = root / READY_DIRECTORY / role
    if not directory.exists():
        return 0
    if directory.is_symlink() or not directory.is_dir():
        raise ProtectedCapacityBuildError(
            f"{role} readiness root is not a safe directory"
        )
    jobs = ledger.get("jobs")
    if not isinstance(jobs, Mapping):
        raise ProtectedCapacityBuildError("build ledger jobs are malformed")
    expected_paths: set[Path] = set()
    valid = 0
    for key, record in jobs.items():
        if not isinstance(record, Mapping) or record.get("role") != role:
            continue
        job_id = str(record.get("job_id") or "")
        if not job_id.isdigit() or record.get("state") != "submitted":
            continue
        chunk_index = int(record["chunk_index"])
        for task_id in range(int(record["tasks"])):
            path = directory / f"{chunk_index:03d}" / str(task_id)
            expected_paths.add(path)
            if not path.exists():
                continue
            if (
                path.is_symlink()
                or not path.is_file()
                or stat.S_IMODE(path.stat().st_mode) & 0o222
            ):
                raise ProtectedCapacityBuildError(
                    f"{role} readiness receipt is mutable or unsafe: {path}"
                )
            expected = {
                "array_job_id": job_id,
                "plan_id": str(plan["plan_id"]),
                "role": role,
                "script_sha256": str(record["script_sha256"]),
                "shape_id": str(record["shape_id"]),
                "task_id": task_id,
                "token": str(plan["token"]),
            }
            raw = path.read_bytes()
            compact = (
                json.dumps(
                    expected,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            if raw != compact:
                raise ProtectedCapacityBuildError(
                    f"{role} readiness receipt differs from live ledger: {path}"
                )
            valid += 1
    observed_paths = {
        path
        for path in directory.glob("*/*")
        if path.is_file() or path.is_symlink()
    }
    if not observed_paths.issubset(expected_paths):
        raise ProtectedCapacityBuildError(
            f"{role} readiness contains an unexpected stale receipt"
        )
    return valid


def _wait_until_live(
    root: Path,
    *,
    deadline_seconds: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    plan = _load_plan(root)
    ledger = _load_ledger(root, plan=plan)
    deadline = float(now()) + deadline_seconds
    while float(now()) < deadline:
        if all(
            plan["roles"][role]["held"]
            or _ready_count(
                root,
                role,
                plan=plan,
                ledger=ledger,
            )
            == plan["roles"][role]["tasks"]
            for role in ROLE_ORDER
        ):
            return
        sleep(5.0)
    counts = {
        role: _ready_count(root, role, plan=plan, ledger=ledger)
        for role in ROLE_ORDER
        if not plan["roles"][role]["held"]
    }
    raise ProtectedCapacityBuildError(
        f"protected-capacity arrays did not all run before deadline: {counts}"
    )


def _capture_job_rows(
    root: Path,
    *,
    command: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> str:
    plan = _load_plan(root)
    ledger = _load_ledger(root, plan=plan)
    jobs = ledger.get("jobs")
    if not isinstance(jobs, dict):
        raise ProtectedCapacityBuildError("build ledger jobs are malformed")
    rows: list[str] = []
    for key, record in sorted(jobs.items()):
        if not isinstance(record, dict):
            raise ProtectedCapacityBuildError(f"build ledger job {key} is malformed")
        role = str(record.get("role", ""))
        chunk_index = int(record.get("chunk_index", -1))
        if role not in ROLE_ORDER or key != _chunk_key(role, chunk_index):
            raise ProtectedCapacityBuildError(
                f"build ledger job identity is malformed: {key}"
            )
        job_id = str(record.get("job_id") or "")
        if not job_id.isdigit():
            raise ProtectedCapacityBuildError(
                f"{key} lacks an exact submitted job ID"
            )
        if command == "squeue":
            argv = [
                "squeue",
                "-h",
                "-r",
                "-j",
                job_id,
                "-o",
                "%A|%a|%T|%P|%q|%C|%m|%b|%l|%k",
            ]
        else:
            argv = [
                "sacct",
                "-nP",
                "--array",
                "-j",
                job_id,
                "-o",
                (
                    "JobID,State,Partition,QOS,ReqCPUS,ReqMem,ReqTRES,"
                    "Timelimit,Comment"
                ),
            ]
        raw = _invoke(runner, argv, timeout=60.0).stdout
        observed: set[int] = set()
        states: set[str] = set()
        partitions: set[str] = set()
        qoses: set[str] = set()
        time_limits: set[int] = set()
        total_cpus = 0
        total_memory_mib = 0
        total_gpus = 0
        for line in raw.splitlines():
            fields = line.strip().split("|")
            if len(fields) < (10 if command == "squeue" else 9):
                continue
            identity = fields[0]
            if command == "squeue":
                base = fields[0]
                task_raw = fields[1]
                state, partition, qos = fields[2:5]
                cpu_raw, memory_raw, tres_raw, time_raw, comment = fields[5:10]
            else:
                match = _JOB_ID_RE.fullmatch(identity)
                if match is None or match.group(2) is None:
                    continue
                base = match.group(1)
                task_raw = match.group(2)
                state, partition, qos = fields[1:4]
                cpu_raw, memory_raw, tres_raw, time_raw, comment = fields[4:9]
            if base != job_id or not task_raw.isdigit():
                continue
            if comment != record["comment"]:
                raise ProtectedCapacityBuildError(
                    f"{command} {key} comment binding drifted"
                )
            task_id = int(task_raw)
            if task_id in observed:
                raise ProtectedCapacityBuildError(
                    f"{command} duplicates {key} task {task_id}"
                )
            observed.add(task_id)
            states.add(state.split("+", 1)[0])
            partitions.add(partition)
            qoses.add(qos)
            try:
                total_cpus += int(cpu_raw)
                total_memory_mib += _memory_mib(memory_raw)
                total_gpus += _gpus_from_tres(tres_raw)
                time_limits.add(_time_seconds(time_raw))
            except (TypeError, ValueError) as exc:
                raise ProtectedCapacityBuildError(
                    f"{command} {key} resource fields are malformed"
                ) from exc
        expected_tasks = int(record.get("tasks", 0))
        if observed != set(range(expected_tasks)):
            raise ProtectedCapacityBuildError(
                f"{command} does not expose all {key} array elements"
            )
        if (
            len(states) != 1
            or len(partitions) != 1
            or len(qoses) != 1
            or len(time_limits) != 1
        ):
            raise ProtectedCapacityBuildError(
                f"{command} {key} scheduler facts are ambiguous"
            )
        state = next(iter(states))
        expected_state = "PENDING" if record["held"] else "RUNNING"
        if state not in {
            expected_state,
            "HELD" if record["held"] else "MIXED",
        }:
            raise ProtectedCapacityBuildError(
                f"{command} {key} is {state}, expected {expected_state}"
            )
        partition = next(iter(partitions))
        qos = next(iter(qoses))
        if partition != plan["partition"] or qos != plan["qos"]:
            raise ProtectedCapacityBuildError(
                f"{command} {key} placement drifted"
            )
        expected_cpus = int(record.get("cpus", -1))
        expected_memory_mib = int(record.get("memory_mib", -1))
        expected_gpus = int(record.get("gpus", -1))
        expected_time_limit = _time_seconds(
            str(record.get("time_limit", ""))
        )
        if (
            total_cpus != expected_cpus
            or total_memory_mib != expected_memory_mib
            or total_gpus != expected_gpus
            or next(iter(time_limits)) != expected_time_limit
        ):
            raise ProtectedCapacityBuildError(
                f"{command} {key} resource request drifted: "
                f"cpus={total_cpus}, memory_mib={total_memory_mib}, "
                f"gpus={total_gpus}"
            )
        job_details = _invoke(
            runner,
            ["scontrol", "show", "job", "-o", job_id],
            timeout=60.0,
        ).stdout
        details = [
            item
            for item in job_details.splitlines()
            if f"ArrayJobId={job_id}" in item or f"JobId={job_id}" in item
        ]
        if not details or any(
            "Requeue=0" not in item
            or f"Comment={record['comment']}" not in item
            or f"TimeLimit={record['time_limit']}" not in item
            for item in details
        ):
            raise ProtectedCapacityBuildError(
                f"{key} does not prove effective Requeue=0 and exact comment"
            )
        spooled_script = _invoke(
            runner,
            ["scontrol", "write", "batch_script", job_id, "-"],
            timeout=60.0,
        ).stdout.encode("utf-8")
        spooled_sha256 = sha256_bytes(spooled_script)
        if spooled_sha256 != record["script_sha256"]:
            raise ProtectedCapacityBuildError(
                f"{key} spooled batch script differs from immutable intent"
            )
        rows.append(
            "|".join(
                [
                    job_id,
                    role,
                    state,
                    partition,
                    qos,
                    str(expected_tasks),
                    str(total_cpus),
                    str(total_memory_mib),
                    str(total_gpus),
                    "0",
                    str(expected_time_limit),
                    str(record["shape_id"]),
                    spooled_sha256,
                ]
            )
        )
    return "\n".join(rows) + "\n"


def capture_source(
    *,
    root: Path,
    kind: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> str:
    plan = _load_plan(root)
    if kind == "scheduler_configuration":
        return _capture_configuration(runner)
    if kind == "partition_configuration":
        return _capture_partition(
            runner,
            partition=str(plan["partition"]),
        )
    if kind == "qos_configuration":
        return _capture_qos(runner, qos=str(plan["qos"]))
    if kind == "association_configuration":
        return _capture_association(
            runner,
            scheduler_user=str(plan["scheduler_user"]),
            qos=str(plan["qos"]),
        )
    contract_sources = {
        "base_fleet_contract": (
            "base_fleet_contract_path",
            "base_fleet_contract_sha256",
        ),
        "effective_fleet_contract": (
            "effective_fleet_contract_path",
            "effective_fleet_contract_sha256",
        ),
        "additive_overlay_contract": (
            "additive_overlay_contract_path",
            "additive_overlay_contract_sha256",
        ),
    }
    if kind in contract_sources:
        path_field, sha_field = contract_sources[kind]
        path = Path(str(plan[path_field])).resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != plan[sha_field]
        ):
            raise ProtectedCapacityBuildError(
                f"{kind.replace('_', ' ')} drifted during capacity capture"
            )
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ProtectedCapacityBuildError(
                f"cannot read {kind.replace('_', ' ')}: {exc}"
            ) from exc
        if not raw.endswith("\n"):
            raw += "\n"
        return raw
    if kind == "static_feasibility_certificate_source":
        binding = plan.get("static_feasibility_certificate")
        if not isinstance(binding, Mapping):
            raise ProtectedCapacityBuildError(
                "capacity plan lacks static feasibility certificate binding"
            )
        path = Path(str(binding["path"])).resolve()
        raw = _stable_bytes(
            path,
            description="static feasibility certificate",
            require_read_only=True,
            require_single_link=True,
        )
        if sha256_bytes(raw) != binding["sha256"]:
            raise ProtectedCapacityBuildError(
                "static feasibility certificate drifted during capacity capture"
            )
        try:
            return raw.decode("utf-8")
        except UnicodeError as exc:
            raise ProtectedCapacityBuildError(
                f"static feasibility certificate is not UTF-8: {exc}"
            ) from exc
    if kind in {"builder_source", "publisher_source"}:
        path_field = f"{kind}_path"
        sha_field = f"{kind}_sha256"
        path = Path(str(plan[path_field])).resolve()
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != plan[sha_field]
        ):
            raise ProtectedCapacityBuildError(
                f"tagged {kind} bytes drifted during capacity capture"
            )
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ProtectedCapacityBuildError(
                f"cannot read tagged {kind}: {exc}"
            ) from exc
    if kind in {"squeue", "sacct"}:
        return _capture_job_rows(root, command=kind, runner=runner)
    raise ProtectedCapacityBuildError(f"unknown capture kind {kind!r}")


def _capture_argv(root: Path, kind: str) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).resolve()),
        "capture-source",
        "--root",
        str(root.resolve()),
        "--kind",
        kind,
    ]


def _source_record(
    *,
    root: Path,
    kind: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> dict[str, Any]:
    raw = capture_source(root=root, kind=kind, runner=runner)
    command = (
        "scontrol"
        if kind in {"scheduler_configuration", "partition_configuration"}
        else "sacctmgr"
        if kind in {"qos_configuration", "association_configuration"}
        else {
            "base_fleet_contract": "base-fleet-contract",
            "effective_fleet_contract": "effective-fleet-contract",
            "additive_overlay_contract": "additive-overlay-contract",
            "static_feasibility_certificate_source": (
                "static-feasibility-certificate"
            ),
        }[kind]
        if kind
        in {
            "base_fleet_contract",
            "effective_fleet_contract",
            "additive_overlay_contract",
            "static_feasibility_certificate_source",
        }
        else "release-source"
        if kind in {"builder_source", "publisher_source"}
        else kind
    )
    argv = _capture_argv(root, kind)
    # The command name is an explicit argument so the publisher can enforce the
    # closed source role even though the immutable capture wrapper is argv[0].
    argv.extend(["--source-command", command])
    return {
        "complete": True,
        "argv": argv,
        "raw_output": raw,
        "raw_output_sha256": sha256_bytes(raw.encode("utf-8")),
        "record_count": len([line for line in raw.splitlines() if line.strip()]),
    }


def _build_evidence(
    root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    now: Callable[[], float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _load_plan(root)
    occupancy_preflight = _validate_occupancy_preflight(
        _load_ledger(root, plan=plan).get("occupancy_preflight"),
        plan=plan,
    )
    source_kinds = (
        "scheduler_configuration",
        "partition_configuration",
        "qos_configuration",
        "association_configuration",
        "base_fleet_contract",
        "effective_fleet_contract",
        "additive_overlay_contract",
        "static_feasibility_certificate_source",
        "builder_source",
        "publisher_source",
        "squeue",
        "sacct",
    )
    sources = {
        kind: _source_record(root=root, kind=kind, runner=runner)
        for kind in source_kinds
    }
    config_fields = sources["scheduler_configuration"]["raw_output"].strip().split("|")
    partition_fields = (
        sources["partition_configuration"]["raw_output"].strip().split("|")
    )
    qos_fields = sources["qos_configuration"]["raw_output"].strip().split("|")
    association_fields = (
        sources["association_configuration"]["raw_output"].strip().split("|")
    )
    if (
        len(config_fields) != 2
        or len(partition_fields) != 8
        or len(qos_fields) != 5
        or len(association_fields) != 6
        or association_fields[2] != plan["scheduler_user"]
        or plan["qos"] not in set(association_fields[3].split(","))
    ):
        raise ProtectedCapacityBuildError(
            "normalized scheduler configuration has invalid cardinality"
        )
    optional_limit = lambda value: (  # noqa: E731 - local normalization
        None if value == "-" else int(value)
    )
    running_scientific_jobs = int(plan["running_scientific_jobs"])
    qos_contract = {
        "qos": str(plan["qos"]),
        "max_wall_seconds": optional_limit(qos_fields[4]),
        "max_jobs_per_user": optional_limit(qos_fields[2]),
        "max_submit_jobs_per_user": optional_limit(qos_fields[3]),
        "required_wall_seconds": SERVER_TIME_LIMIT_SECONDS,
        "required_running_jobs": running_scientific_jobs,
        "required_submit_jobs": EXPECTED_TOTAL_JOB_ELEMENTS,
    }
    binding = {
        "release_id": publisher.RELEASE_ID,
        "release_tag": publisher.RELEASE_TAG,
        "release_git_commit": plan["release_git_commit"],
        "release_tag_object": plan["release_tag_object"],
        "chain_namespace": publisher.CHAIN_NAMESPACE,
    }
    scheduler: dict[str, Any] = {
        "schema_version": publisher.SCHEMA_VERSION,
        "protocol": publisher.SCHEDULER_EVIDENCE_PROTOCOL,
        "passed": True,
        **binding,
        "observed_at": float(now()),
        "preempt_type": config_fields[1],
        "capacity_source": publisher.CAPACITY_SOURCE,
        "scheduler_cluster": association_fields[0],
        "scheduler_account": association_fields[1],
        "scheduler_user": association_fields[2],
        "scheduler_max_jobs": optional_limit(association_fields[4]),
        "scheduler_max_submit_jobs": int(association_fields[5]),
        "running_scientific_jobs": running_scientific_jobs,
        "minimum_scientific_wall_seconds": SERVER_TIME_LIMIT_SECONDS,
        "scientific_qos_contracts": [qos_contract],
        "partition_cpus": int(partition_fields[4]),
        "partition_memory_mib": int(partition_fields[5]),
        "partition_gpus": int(partition_fields[6]),
        "capacity_generation": plan["capacity_generation"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "dispatcher_source_sha256": plan[
            "dispatcher_source_sha256"
        ],
        "qualification_runner_source_sha256": plan[
            "qualification_runner_source_sha256"
        ],
        "base_fleet_contract_path": plan["base_fleet_contract_path"],
        "base_fleet_contract_sha256": plan[
            "base_fleet_contract_sha256"
        ],
        "effective_fleet_contract_path": plan[
            "effective_fleet_contract_path"
        ],
        "effective_fleet_contract_sha256": plan[
            "effective_fleet_contract_sha256"
        ],
        "additive_overlay_contract_path": plan[
            "additive_overlay_contract_path"
        ],
        "additive_overlay_contract_sha256": plan[
            "additive_overlay_contract_sha256"
        ],
        "static_feasibility_certificate": plan[
            "static_feasibility_certificate"
        ],
        "base_active_logical_replicas": plan[
            "base_active_logical_replicas"
        ],
        "base_active_gpus": plan["base_active_gpus"],
        "base_active_topology": plan["base_active_topology"],
        "base_active_topology_sha256": plan[
            "base_active_topology_sha256"
        ],
        "additive_reserved_logical_replicas": plan[
            "additive_reserved_logical_replicas"
        ],
        "additive_reserved_gpus": plan["additive_reserved_gpus"],
        "additive_reserved_tp1_replicas": plan[
            "additive_reserved_tp1_replicas"
        ],
        "additive_reserved_tp2_replicas": plan[
            "additive_reserved_tp2_replicas"
        ],
        "additive_reserved_topology": plan[
            "additive_reserved_topology"
        ],
        "additive_reserved_topology_sha256": sha256_bytes(
            canonical_bytes(plan["additive_reserved_topology"])
        ),
        "effective_active_logical_replicas": plan[
            "effective_active_logical_replicas"
        ],
        "effective_active_gpus": plan["effective_active_gpus"],
        "effective_active_topology": plan["effective_active_topology"],
        "effective_active_topology_sha256": plan[
            "active_fleet_topology_sha256"
        ],
        "retained_warm_turnover_job_elements": (
            WARM_TURNOVER_JOB_ELEMENTS
        ),
        "retained_warm_turnover_gpus": WARM_TURNOVER_GPUS,
        "retained_warm_turnover_tp1_allocations": 2,
        "retained_warm_turnover_tp2_allocations": 1,
        "retained_warm_turnover_topology": plan[
            "retained_warm_turnover_topology"
        ],
        "retained_warm_turnover_topology_sha256": sha256_bytes(
            canonical_bytes(plan["retained_warm_turnover_topology"])
        ),
        "attested_total_gpus": plan["attested_total_gpus"],
        "fleet_contract_sha256": plan["effective_fleet_contract_sha256"],
        "active_fleet_topology_sha256": plan[
            "active_fleet_topology_sha256"
        ],
        "builder_source_sha256": plan["builder_source_sha256"],
        "publisher_source_sha256": plan["publisher_source_sha256"],
        "expected_total_job_elements": plan[
            "expected_total_job_elements"
        ],
        "job_element_accounting": plan["job_element_accounting"],
        "occupancy_preflight": occupancy_preflight,
        **sources,
        "scientific_server_placements": [
            {
                "partition": plan["partition"],
                "qos": plan["qos"],
                "partition_preempt_mode": partition_fields[1],
                "qos_preempt_mode": qos_fields[1],
                "base_active_gpus": plan["base_active_gpus"],
                "reserved_additive_gpus": plan[
                    "additive_reserved_gpus"
                ],
                "effective_active_gpus": plan[
                    "effective_active_gpus"
                ],
                "retained_warm_turnover_gpus": WARM_TURNOVER_GPUS,
                "attested_total_gpus": plan["attested_total_gpus"],
                "partition_cpus": int(partition_fields[4]),
                "partition_memory_mib": int(partition_fields[5]),
                "partition_gpus": int(partition_fields[6]),
                "partition_nodes": int(partition_fields[7]),
            }
        ],
        "scientific_client_placements": [
            {
                "partition": plan["partition"],
                "qos": plan["qos"],
                "partition_preempt_mode": partition_fields[1],
                "qos_preempt_mode": qos_fields[1],
                "slots": CLIENT_JOB_ELEMENTS,
                "cpus": CLIENT_JOB_ELEMENTS,
                "memory_mib": CLIENT_JOB_ELEMENTS * 4096,
                "reserve_jobs": TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS,
                "submit_headroom": EXPECTED_TOTAL_JOB_ELEMENTS,
            }
        ],
    }
    scheduler = publisher.with_self_hash(
        scheduler,
        identity_field="evidence_id",
    )
    canary_rows = [
        {
            "partition": plan["partition"],
            "qos": plan["qos"],
            "partition_preempt_mode": partition_fields[1],
            "qos_preempt_mode": qos_fields[1],
            "effective_requeue": 0,
        }
    ]
    canary: dict[str, Any] = {
        "schema_version": publisher.SCHEMA_VERSION,
        "protocol": publisher.CANARY_EVIDENCE_PROTOCOL,
        "passed": True,
        **binding,
        "completed_at": float(now()),
        "scheduler_evidence_id": scheduler["evidence_id"],
        "scientific_server_placements": canary_rows,
        "scientific_client_placements": canary_rows,
        "squeue_complete": True,
        "sacct_complete": True,
    }
    canary = publisher.with_self_hash(canary, identity_field="canary_id")
    return scheduler, canary


def _terminal_cleanup_receipt(
    *,
    job_id: str,
    comment: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
) -> dict[str, Any] | None:
    try:
        raw = _invoke(
            runner,
            [
                "sacct",
                "-nP",
                "--array",
                "-j",
                job_id,
                "-o",
                "JobID,State,Comment",
            ],
            timeout=60.0,
        ).stdout
    except ProtectedCapacityBuildError:
        return None
    terminal_states: set[str] = set()
    for line in raw.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 3:
            continue
        identity, state, observed_comment = fields
        match = re.fullmatch(r"([0-9]+)(?:_[0-9]+)?", identity)
        if (
            match is None
            or match.group(1) != job_id
            or observed_comment != comment
        ):
            continue
        normalized = state.split("+", 1)[0]
        if normalized not in {
            "PENDING",
            "RUNNING",
            "CONFIGURING",
            "COMPLETING",
        }:
            terminal_states.add(normalized)
    if not terminal_states:
        return None
    return {
        "job_id": job_id,
        "comment": comment,
        "terminal_states": sorted(terminal_states),
        "source": "complete_sacct_reconciliation",
    }


def _cleanup_transaction(
    *,
    root: Path,
    marker: Mapping[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    now: Callable[[], float],
) -> dict[str, Any]:
    plan = _load_plan(root)
    ledger_path = root / LEDGER_FILENAME
    ledger = _load_ledger(root, plan=plan)
    release_payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "plan_id": plan["plan_id"],
        "marker_id": marker["marker_id"],
    }
    release_payload["release_id"] = sha256_bytes(
        canonical_bytes(release_payload)
    )
    release = root / RELEASE_FILENAME
    _publish_sealed_once(release, release_payload)
    release_sha256 = sha256_file(release)
    for key, record in sorted(ledger["jobs"].items()):
        if record["role"] != "reserve":
            if record["cleanup_state"] != "released":
                record["cleanup_state"] = "released"
                record["cleanup_receipt"] = {
                    "release_path": str(release),
                    "release_sha256": release_sha256,
                    "recorded_at": float(now()),
                }
                _atomic_json(ledger_path, ledger)
            continue
        if record["cleanup_state"] == "cancelled":
            continue
        job_id = str(record.get("job_id") or "")
        if not job_id.isdigit():
            raise ProtectedCapacityBuildError(
                f"reserve canary {key} lacks an exact cleanup job ID"
            )
        if record["cleanup_state"] == "cancelling":
            terminal = _terminal_cleanup_receipt(
                job_id=job_id,
                comment=str(record["comment"]),
                runner=runner,
            )
            if terminal is not None:
                terminal["recorded_at"] = float(now())
                record["cleanup_state"] = "cancelled"
                record["cleanup_receipt"] = terminal
                _atomic_json(ledger_path, ledger)
                continue
        record["cleanup_state"] = "cancelling"
        record["cleanup_receipt"] = {
            "job_id": job_id,
            "comment": record["comment"],
            "intent_at": float(now()),
        }
        _atomic_json(ledger_path, ledger)
        try:
            _invoke(runner, ["scancel", job_id], timeout=60.0)
        except ProtectedCapacityBuildError:
            terminal = _terminal_cleanup_receipt(
                job_id=job_id,
                comment=str(record["comment"]),
                runner=runner,
            )
            if terminal is None:
                raise
            receipt = terminal
        else:
            receipt = {
                "job_id": job_id,
                "comment": record["comment"],
                "source": "exact_scancel_receipt",
            }
        receipt["recorded_at"] = float(now())
        record["cleanup_state"] = "cancelled"
        record["cleanup_receipt"] = receipt
        _atomic_json(ledger_path, ledger)
    return _load_ledger(root, plan=plan)


def apply_transaction(
    *,
    root: Path,
    recovery_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    deadline_seconds: float = 1800.0,
    occupancy_observation_interval_seconds: float = (
        OCCUPANCY_OBSERVATION_INTERVAL_SECONDS
    ),
) -> dict[str, Any]:
    plan = _load_plan(root)
    marker_path = recovery_root / publisher.MARKER_FILENAME
    if marker_path.exists() or marker_path.is_symlink():
        marker = publisher.verify_marker(
            recovery_root,
            expected_release_git_commit=str(plan["release_git_commit"]),
            expected_release_tag_object=str(plan["release_tag_object"]),
            expected_source_tree_sha256=str(plan["source_tree_sha256"]),
            expected_dispatcher_source_sha256=str(
                plan["dispatcher_source_sha256"]
            ),
            expected_qualification_runner_source_sha256=str(
                plan["qualification_runner_source_sha256"]
            ),
        )
        if marker["fleet_contract_sha256"] != plan["fleet_contract_sha256"]:
            raise ProtectedCapacityBuildError(
                "existing protected-capacity marker binds a different fleet"
            )
        _cleanup_transaction(
            root=root,
            marker=marker,
            runner=runner,
            now=now,
        )
        return {
            "status": "already_complete",
            "marker": marker,
            "marker_path": str(marker_path),
        }
    ledger = _load_ledger(root, plan=plan)
    crossed_submission_boundary = any(
        record.get("state") != "prepared"
        or int(record.get("attempt", 0)) > 0
        or record.get("job_id") is not None
        for record in ledger["jobs"].values()
    )
    if not crossed_submission_boundary:
        ledger["occupancy_preflight"] = _verify_submit_headroom_preflight(
            plan=plan,
            runner=runner,
            now=now,
            sleep=sleep,
            observation_interval_seconds=(
                occupancy_observation_interval_seconds
            ),
        )
        _atomic_json(root / LEDGER_FILENAME, ledger)
    elif ledger.get("occupancy_preflight") is None:
        raise ProtectedCapacityBuildError(
            "capacity submission crossed sbatch without a durable two-observation "
            "occupancy preflight"
        )
    _submit_jobs(root, runner=runner, now=now, sleep=sleep)
    _wait_until_live(
        root,
        deadline_seconds=deadline_seconds,
        now=now,
        sleep=sleep,
    )
    scheduler, canary = _build_evidence(root, runner=runner, now=now)
    final_ledger = _load_ledger(root, plan=plan)
    occupancy_preflight = _validate_occupancy_preflight(
        final_ledger.get("occupancy_preflight"),
        plan=plan,
    )
    occupancy_path = root / OCCUPANCY_PREFLIGHT_FILENAME
    scheduler_path = root / SCHEDULER_EVIDENCE_FILENAME
    canary_path = root / CANARY_EVIDENCE_FILENAME
    _publish_sealed_once(occupancy_path, occupancy_preflight)
    _publish_sealed_once(scheduler_path, scheduler)
    _publish_sealed_once(canary_path, canary)

    def recapture(
        argv: Sequence[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del timeout
        args = list(argv)
        try:
            kind = args[args.index("--kind") + 1]
            observed_root = Path(args[args.index("--root") + 1]).resolve()
        except (ValueError, IndexError) as exc:
            return subprocess.CompletedProcess(args, 2, "", str(exc))
        if observed_root != root.resolve():
            return subprocess.CompletedProcess(
                args, 2, "", "capture root drifted"
            )
        try:
            raw = capture_source(
                root=observed_root,
                kind=kind,
                runner=runner,
            )
        except ProtectedCapacityBuildError as exc:
            return subprocess.CompletedProcess(args, 2, "", str(exc))
        return subprocess.CompletedProcess(args, 0, raw, "")

    result = publisher.attest(
        recovery_root=recovery_root,
        scheduler_evidence_path=scheduler_path,
        canary_evidence_path=canary_path,
        expected_release_git_commit=str(plan["release_git_commit"]),
        expected_release_tag_object=str(plan["release_tag_object"]),
        expected_source_tree_sha256=str(plan["source_tree_sha256"]),
        expected_dispatcher_source_sha256=str(
            plan["dispatcher_source_sha256"]
        ),
        expected_qualification_runner_source_sha256=str(
            plan["qualification_runner_source_sha256"]
        ),
        apply=True,
        runner=recapture,
    )
    # Release only after the marker has been independently reparsed, recaptured,
    # and published. Cleanup is a resumable exact-ID transaction.
    _cleanup_transaction(
        root=root,
        marker=result["marker"],
        runner=runner,
        now=now,
    )
    return {
        "status": "complete",
        "marker": result,
        "scheduler_evidence": str(scheduler_path),
        "scheduler_evidence_sha256": sha256_file(scheduler_path),
        "occupancy_preflight": str(occupancy_path),
        "occupancy_preflight_sha256": sha256_file(occupancy_path),
        "canary_evidence": str(canary_path),
        "canary_evidence_sha256": sha256_file(canary_path),
    }


def run_operation(
    *,
    root: Path,
    recovery_root: Path,
    release_git_commit: str,
    release_tag_object: str,
    partition: str,
    qos: str,
    scheduler_user: str,
    token: str,
    fleet_contract_path: Path,
    fleet_contract_sha256: str,
    effective_fleet_contract_path: Path,
    effective_fleet_contract_sha256: str,
    additive_overlay_contract_path: Path,
    additive_overlay_contract_sha256: str,
    static_feasibility_certificate_path: Path,
    static_feasibility_certificate_sha256: str,
    static_feasibility_certificate_id: str,
    capacity_generation: int,
    model_contract_path: Path,
    model_contract_sha256: str,
    release_worktree: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    deadline_seconds: float = 1800.0,
    occupancy_observation_interval_seconds: float = (
        OCCUPANCY_OBSERVATION_INTERVAL_SECONDS
    ),
) -> dict[str, Any]:
    """Prepare or resume one durable real-capacity transaction."""

    plan = build_plan(
        root=root,
        release_git_commit=release_git_commit,
        release_tag_object=release_tag_object,
        partition=partition,
        qos=qos,
        scheduler_user=scheduler_user,
        token=token,
        fleet_contract_path=fleet_contract_path,
        fleet_contract_sha256=fleet_contract_sha256,
        effective_fleet_contract_path=effective_fleet_contract_path,
        effective_fleet_contract_sha256=effective_fleet_contract_sha256,
        additive_overlay_contract_path=additive_overlay_contract_path,
        additive_overlay_contract_sha256=additive_overlay_contract_sha256,
        static_feasibility_certificate_path=(
            static_feasibility_certificate_path
        ),
        static_feasibility_certificate_sha256=(
            static_feasibility_certificate_sha256
        ),
        static_feasibility_certificate_id=(
            static_feasibility_certificate_id
        ),
        capacity_generation=capacity_generation,
        model_contract_path=model_contract_path,
        model_contract_sha256=model_contract_sha256,
        release_worktree=release_worktree,
        identity_runner=runner,
    )
    prepare(plan, apply=True)
    return apply_transaction(
        root=root.resolve(),
        recovery_root=recovery_root.resolve(),
        runner=runner,
        now=now,
        sleep=sleep,
        deadline_seconds=deadline_seconds,
        occupancy_observation_interval_seconds=(
            occupancy_observation_interval_seconds
        ),
    )


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--release-git-commit", required=True)
    parser.add_argument("--release-tag-object", required=True)
    parser.add_argument("--partition", default="ou_bcs_normal")
    parser.add_argument("--qos", default="normal")
    parser.add_argument("--scheduler-user", default=os.environ.get("USER", ""))
    parser.add_argument("--token", required=True)
    parser.add_argument("--fleet-contract", required=True, type=Path)
    parser.add_argument("--fleet-contract-sha256", required=True)
    parser.add_argument(
        "--effective-fleet-contract",
        required=True,
        type=Path,
    )
    parser.add_argument("--effective-fleet-contract-sha256", required=True)
    parser.add_argument(
        "--additive-overlay-contract",
        required=True,
        type=Path,
    )
    parser.add_argument("--additive-overlay-contract-sha256", required=True)
    parser.add_argument(
        "--static-feasibility-certificate",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--static-feasibility-certificate-sha256",
        required=True,
    )
    parser.add_argument(
        "--static-feasibility-certificate-id",
        required=True,
    )
    parser.add_argument("--capacity-generation", required=True, type=int)
    parser.add_argument("--model-contract", required=True, type=Path)
    parser.add_argument("--model-contract-sha256", required=True)
    parser.add_argument("--release-worktree", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    _add_plan_arguments(plan)

    run = subparsers.add_parser(
        "run",
        help="prepare or resume the real canary and publish evidence marker-last",
    )
    _add_plan_arguments(run)
    run.add_argument(
        "--apply",
        action="store_true",
        help="submit real Slurm canaries; default only prints the exact plan",
    )
    run.add_argument("--recovery-root", required=True, type=Path)
    run.add_argument("--deadline-seconds", type=float, default=1800.0)

    capture = subparsers.add_parser("capture-source")
    capture.add_argument("--root", required=True, type=Path)
    capture.add_argument(
        "--kind",
        required=True,
        choices=(
            "scheduler_configuration",
            "partition_configuration",
            "qos_configuration",
            "association_configuration",
            "base_fleet_contract",
            "effective_fleet_contract",
            "additive_overlay_contract",
            "static_feasibility_certificate_source",
            "builder_source",
            "publisher_source",
            "squeue",
            "sacct",
        ),
    )
    # This inert identity argument lets the publisher bind each normalized source
    # to the underlying scheduler command without executing arbitrary shell.
    capture.add_argument(
        "--source-command",
        choices=(
            "scontrol",
            "sacctmgr",
            "squeue",
            "sacct",
            "base-fleet-contract",
            "effective-fleet-contract",
            "additive-overlay-contract",
            "static-feasibility-certificate",
            "release-source",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "capture-source":
        sys.stdout.write(capture_source(root=args.root, kind=args.kind))
        return 0
    plan = build_plan(
        root=args.root,
        release_git_commit=args.release_git_commit,
        release_tag_object=args.release_tag_object,
        partition=args.partition,
        qos=args.qos,
        scheduler_user=args.scheduler_user,
        token=args.token,
        fleet_contract_path=args.fleet_contract,
        fleet_contract_sha256=args.fleet_contract_sha256,
        effective_fleet_contract_path=args.effective_fleet_contract,
        effective_fleet_contract_sha256=(
            args.effective_fleet_contract_sha256
        ),
        additive_overlay_contract_path=args.additive_overlay_contract,
        additive_overlay_contract_sha256=(
            args.additive_overlay_contract_sha256
        ),
        static_feasibility_certificate_path=(
            args.static_feasibility_certificate
        ),
        static_feasibility_certificate_sha256=(
            args.static_feasibility_certificate_sha256
        ),
        static_feasibility_certificate_id=(
            args.static_feasibility_certificate_id
        ),
        capacity_generation=args.capacity_generation,
        model_contract_path=args.model_contract,
        model_contract_sha256=args.model_contract_sha256,
        release_worktree=args.release_worktree,
    )
    if args.command == "plan" or not args.apply:
        prepared = prepare(plan, apply=False)
        print(json.dumps(prepared, indent=2, sort_keys=True))
        return 0
    result = run_operation(
        root=args.root,
        recovery_root=args.recovery_root,
        release_git_commit=args.release_git_commit,
        release_tag_object=args.release_tag_object,
        partition=args.partition,
        qos=args.qos,
        scheduler_user=args.scheduler_user,
        token=args.token,
        fleet_contract_path=args.fleet_contract,
        fleet_contract_sha256=args.fleet_contract_sha256,
        effective_fleet_contract_path=args.effective_fleet_contract,
        effective_fleet_contract_sha256=(
            args.effective_fleet_contract_sha256
        ),
        additive_overlay_contract_path=args.additive_overlay_contract,
        additive_overlay_contract_sha256=(
            args.additive_overlay_contract_sha256
        ),
        static_feasibility_certificate_path=(
            args.static_feasibility_certificate
        ),
        static_feasibility_certificate_sha256=(
            args.static_feasibility_certificate_sha256
        ),
        static_feasibility_certificate_id=(
            args.static_feasibility_certificate_id
        ),
        capacity_generation=args.capacity_generation,
        model_contract_path=args.model_contract,
        model_contract_sha256=args.model_contract_sha256,
        release_worktree=args.release_worktree,
        deadline_seconds=float(args.deadline_seconds),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtectedCapacityBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
