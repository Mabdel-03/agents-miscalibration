#!/usr/bin/env python3
"""Run, seal, or verify the estimand-excluded schema-5 throughput qualification.

The qualification is deliberately outside the three production run namespaces.  It
uses one immutable 768-cell manifest (20 QIDs per cell), a private dispatcher ledger,
and the frozen dispatcher's normal ``run-task`` boundary.  The production control is
read only: it must remain paused and its admission/ramp projection must not change.

``dry-run`` is entirely non-mutating.  ``execute`` is the sole mode that may initialize
the qualification run or submit qualification arrays.  It advances isolated active-cell
limits through 24, 96, 192, and 384, recording immutable scheduler and semantic
observations at every poll.  A completion marker is published last only after:

* all 768 cells and 15,360 QIDs are schema-5 complete;
* every nonempty model x reasoning x topology x agent-count stratum progressed;
* scheduler and semantic evidence contain no integrity or transport-censor incident;
* ceiling 384 remained continuously clean for at least two hours; and
* useful throughput is at least 201,994 QIDs/day.

``verify-only`` reads only sealed qualification evidence.  Re-running ``execute`` after
completion takes the same verification-only path and never contacts Slurm.  Immutable
write-once artifacts are adopted only when their bytes and modes are exact; orphaned or
conflicting partial observations fail closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
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
from typing import Any, Callable, Iterator, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
for _path in (REPO, SOURCE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from agents_scaling.benchmarks.contracts import (  # noqa: E402
    build_run_benchmark_contracts,
    freeze_benchmark_contracts,
)
from agents_scaling.benchmarks.runtime_contracts import (  # noqa: E402
    VerifiedQuestionCatalog,
)
from agents_scaling.config import (  # noqa: E402
    ContextShareLevel,
    ExperimentCell,
    ReasoningLevel,
    Topology,
)
from agents_scaling.experiment.analyze import _censor_accounting  # noqa: E402
from agents_scaling.experiment.artifact_policy import (  # noqa: E402
    POLICY_CHECKSUM_FILENAME,
    POLICY_DOMAIN,
    POLICY_FILENAME,
    REQUIRED_METADATA_FIELDS,
    load_artifact_policy,
)
from agents_scaling.experiment.completion import (  # noqa: E402
    CompletionState,
    get_completion_status,
    read_canonical_results,
)
from agents_scaling.experiment.manifest import (  # noqa: E402
    ManifestSnapshot,
    freeze_manifest,
    load_manifest,
)
from agents_scaling.experiment import io as experiment_io  # noqa: E402
from agents_scaling.models import QWEN3_LADDER  # noqa: E402
from agents_scaling.serving import protected_capacity  # noqa: E402
from agents_scaling.serving.profiles import serving_profile_for_cell  # noqa: E402
from slurm import dispatch_sweeps  # noqa: E402
from slurm import schema5_control as control  # noqa: E402
from scripts import render_schema5_recovery_chain_v12 as renderer  # noqa: E402


SCHEMA_VERSION = 1
QUALIFICATION_RUN_ID = "schema5_throughput_qualification_v1"
QUALIFICATION_ROOT_NAME = QUALIFICATION_RUN_ID
MARKER_NAME = "THROUGHPUT_QUALIFICATION_COMPLETE.json"
PROTOCOL = "schema5-v1.2-r2-throughput-qualification-v1"
PLAN_PROTOCOL = "schema5-v1.2-r2-throughput-qualification-load-plan-v1"
INTENT_PROTOCOL = "schema5-v1.2-r2-throughput-qualification-intent-v1"
SCHEDULER_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-scheduler-evidence-v1"
)
SEMANTIC_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-semantic-evidence-v1"
)
OBSERVATION_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-observation-v1"
)
OBSERVATION_TRANSACTION_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-observation-transaction-v1"
)
EVIDENCE_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-evidence-v1"
)
LINEAGE_PROTOCOL = "schema5-v1.2-r2-throughput-qualification-lineage-v1"
EXECUTION_AUTHORITY_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-execution-authority-v1"
)
ATTEMPT_POINTER_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-attempt-pointer-v1"
)
CURRENT_ATTEMPT_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-current-attempt-v1"
)

CELL_COUNT = 768
QIDS_PER_CELL = 20
TOTAL_QIDS = CELL_COUNT * QIDS_PER_CELL
CEILINGS = (24, 96, 192, 384)
MAX_BATCH = 24
QOS_LIMIT = 448
QOS_RESERVE = 64
STEADY_384_SECONDS = 7_200
MAX_OBSERVATION_GAP_SECONDS = 660
MIN_QIDS_PER_DAY = 201_994
DEFAULT_POLL_SECONDS = 300.0
DEFAULT_TIMEOUT_SECONDS = 36_000.0

PLAN_NAME = "LOAD_PLAN.json"
INTENT_NAME = "QUALIFICATION_INTENT.json"
EVIDENCE_NAME = "QUALIFICATION_EVIDENCE.json"
LINEAGE_NAME = "qualification_lineage.schema5-v1.json"
INITIALIZED_NAME = "SCHEMA5_THROUGHPUT_QUALIFICATION_INITIALIZED.json"
EXECUTION_AUTHORITY_NAME = "QUALIFICATION_EXECUTION_AUTHORITY.json"
FAILURE_NAME = "QUALIFICATION_FAILURE.json"
FAILURE_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-failure-v1"
)
CAPACITY_TRANSITION_NAME = "QUALIFICATION_CAPACITY_TRANSITION_COMPLETE.json"
CAPACITY_TRANSITION_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-capacity-transition-v1"
)
LOCK_NAME = ".throughput-qualification.lock"
CURRENT_ATTEMPT_NAME = "CURRENT_ATTEMPT.json"
ATTEMPT_DIRECTORY = "attempts"
ATTEMPT_POINTER_DIRECTORY = "attempt-pointers"
ATTEMPT_RUN_DIRECTORY = "throughput-qualification-attempts"
SCHEDULER_DIRECTORY = "scheduler"
SEMANTIC_DIRECTORY = "semantic"
OBSERVATION_DIRECTORY = "observations"
OBSERVATION_TRANSACTION_DIRECTORY = "observation-transactions"
DISPATCHER_STATE_DIRECTORY = "dispatcher"

_FAILURE_FIELDS = {
    "schema_version",
    "protocol",
    "passed",
    "intent_id",
    "attempt",
    "readiness_generation",
    "reason",
    "additive_scaling_requirement",
    "scheduler_capacity_mutated",
    "rerun_requirement",
    "failure_id",
}
_SCALING_REQUIREMENT_FIELDS = {
    "serving_profile",
    "server_pool_root",
    "backlog_fanout_work",
    "live_replicas",
    "backlog_work_per_replica",
    "additional_replicas",
    "tensor_parallel_size",
    "additional_gpus",
    "requirement",
    "capacity_mutated",
}

MODELS = tuple(QWEN3_LADDER)
SERVING_PROFILES = (
    "0.6B",
    "1.7B",
    "4B",
    "8B",
    "14B",
    "32B",
    "0.6B-long",
    "1.7B-long",
    "4B-long",
    "8B-long",
    "14B-long",
    "32B-long",
)
REASONING_LEVELS = tuple(level.value for level in ReasoningLevel)
BENCHMARKS = ("gpqa", "mmlu_pro", "math", "truthfulqa")
SEEDS = (0, 1, 2)
MAS_TOPOLOGIES = (
    Topology.INDEPENDENT.value,
    Topology.DECENTRALIZED.value,
    Topology.CENTRALIZED.value,
)
MAS_AGENT_COUNTS = (2, 3, 4, 5, 6, 7)
STRATUM_DIMENSIONS = (
    "model_size",
    "reasoning_level",
    "topology",
    "agent_count",
)
ROTATION_DIMENSIONS = (
    *STRATUM_DIMENSIONS,
    "benchmark",
    "context_share_level",
    "prompt_complexity_level",
    "seed",
)
EXPECTED_STRATA = len(MODELS) * len(REASONING_LEVELS) * (
    1 + len(MAS_TOPOLOGIES) * len(MAS_AGENT_COUNTS)
)
PRODUCTION_RUN_IDS = frozenset(control.REQUIRED_RUNS)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_PARTITION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class ThroughputQualificationError(RuntimeError):
    """The qualification cannot proceed without weakening its evidence contract."""


class QualificationCapacityTransitionRequired(ThroughputQualificationError):
    """A sealed throughput-only shortfall requires one exact additive transition."""


@dataclass(frozen=True)
class QualificationContext:
    """Immutable chain paths and prerequisite bindings used by the producer."""

    chain_manifest: Path
    chain_manifest_sha256: str
    chain_id: str
    results_root: Path
    recovery_root: Path
    readiness_root: Path
    qualification_base: Path
    qualification_root: Path
    run_root: Path
    dispatcher_state: Path
    state_root: Path
    server_pool_root: Path
    release_worktree: Path
    harness_prefix: Path
    hf_home: Path
    release_git_commit: str
    release_tag_object: str
    protected_capacity: Mapping[str, str]
    protected_capacity_contract: protected_capacity.ProtectedCapacityContract
    attempt_pointer_path: Path | None = None
    attempt_pointer: Mapping[str, Any] | None = None


def _canonical_bytes(value: Any) -> bytes:
    """Use the renderer's exact pretty canonical form for every self-hash."""

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


def _compact_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ThroughputQualificationError(
            f"cannot open qualification artifact {path}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ThroughputQualificationError(
                f"qualification artifact is not regular: {path}"
            )
        for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (  # noqa: E731 - compact immutable-read identity
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise ThroughputQualificationError(
            f"qualification artifact changed while hashing: {path}"
        )
    return digest.hexdigest()


def _with_identity(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise ThroughputQualificationError(
            f"identity input already contains {field!r}"
        )
    result = dict(value)
    result[field] = _sha256_bytes(_canonical_bytes(result))
    return result


def _verify_identity(
    value: Mapping[str, Any], field: str, *, description: str
) -> None:
    identity = dict(value)
    observed = identity.pop(field, None)
    if (
        not isinstance(observed, str)
        or _SHA256_RE.fullmatch(observed) is None
        or observed != _sha256_bytes(_canonical_bytes(identity))
    ):
        raise ThroughputQualificationError(
            f"{description} has an invalid {field} self-hash"
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ThroughputQualificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ThroughputQualificationError(f"non-finite JSON number {token!r}")


def _read_json(
    path: Path,
    *,
    description: str,
    sealed: bool = False,
) -> dict[str, Any]:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if lexical.is_symlink() or not lexical.is_file():
        raise ThroughputQualificationError(
            f"{description} is missing, non-regular, or symlinked: {lexical}"
        )
    raw: bytes
    if sealed:
        try:
            if lexical.resolve(strict=True) != lexical:
                raise ThroughputQualificationError(
                    f"{description} traverses a symlink: {lexical}"
                )
        except (OSError, RuntimeError) as exc:
            raise ThroughputQualificationError(
                f"{description} is not a canonical sealed path: {lexical}: {exc}"
            ) from exc
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lexical, flags)
        except OSError as exc:
            raise ThroughputQualificationError(
                f"cannot open sealed {description} {lexical}: {exc}"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) & 0o222
            ):
                raise ThroughputQualificationError(
                    f"{description} is not a unique read-only regular file: "
                    f"{lexical}"
                )
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            first_raw = b"".join(chunks)
            os.lseek(descriptor, 0, os.SEEK_SET)
            repeated_chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                repeated_chunks.append(chunk)
            repeated_raw = b"".join(repeated_chunks)
            after = os.fstat(descriptor)
            current = lexical.stat(follow_symlinks=False)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
                "st_nlink",
                "st_mode",
            )
            if any(
                getattr(before, field) != getattr(after, field)
                or getattr(after, field) != getattr(current, field)
                for field in stable_fields
            ) or first_raw != repeated_raw:
                raise ThroughputQualificationError(
                    f"{description} changed during its sealed read: {lexical}"
                )
            raw = repeated_raw
        finally:
            os.close(descriptor)
    else:
        try:
            raw = lexical.read_bytes()
        except OSError as exc:
            raise ThroughputQualificationError(
                f"cannot read {description} {lexical}: {exc}"
            ) from exc
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except ThroughputQualificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ThroughputQualificationError(
            f"cannot parse {description} {lexical}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ThroughputQualificationError(
            f"{description} must contain exactly one JSON object"
        )
    if sealed and raw != _canonical_bytes(value):
        raise ThroughputQualificationError(
            f"{description} is not canonically encoded: {lexical}"
        )
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


def _seal_tree_read_only(root: Path, *, description: str) -> None:
    """Recursively preserve one terminal attempt without deleting any bytes."""

    if root.is_symlink() or not root.is_dir():
        raise ThroughputQualificationError(
            f"{description} root is missing, non-directory, or symlinked: {root}"
        )
    paths = [root, *root.rglob("*")]
    for path in paths:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not (
                stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISREG(metadata.st_mode)
            )
            or (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_nlink != 1
            )
            or path.name.endswith(".publishing")
        ):
            raise ThroughputQualificationError(
                f"{description} contains a non-unique or unsafe member: {path}"
            )
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode & ~0o222)
    _fsync_directory(root.parent)


def _assert_tree_read_only(root: Path, *, description: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ThroughputQualificationError(
            f"{description} root is unsafe: {root}"
        )
    for path in [root, *root.rglob("*")]:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not (
                stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISREG(metadata.st_mode)
            )
            or (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_nlink != 1
            )
            or path.name.endswith(".publishing")
            or stat.S_IMODE(metadata.st_mode) & 0o222
        ):
            raise ThroughputQualificationError(
                f"{description} is not recursively read-only: {path}"
            )


def _assert_no_stale_publications(
    path: Path,
    *,
    description: str,
) -> None:
    parent = path.parent
    if not parent.exists():
        return
    if parent.is_symlink() or not parent.is_dir():
        raise ThroughputQualificationError(
            f"{description} parent is unsafe: {parent}"
        )
    prefix = f".{path.name}."
    stale = sorted(
        child
        for child in parent.iterdir()
        if child.name.startswith(prefix)
        and child.name.endswith(".publishing")
    )
    if stale:
        raise ThroughputQualificationError(
            f"{description} has an unreconciled stale publication: {stale[0]}"
        )


def _write_once(
    path: Path,
    payload: Mapping[str, Any] | bytes,
    *,
    description: str,
    mode: int = 0o444,
) -> None:
    encoded = bytes(payload) if isinstance(payload, bytes) else _canonical_bytes(payload)
    _assert_no_stale_publications(path, description=description)
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != encoded
            or stat.S_IMODE(path.stat().st_mode) & 0o222
        ):
            raise ThroughputQualificationError(
                f"existing {description} conflicts with the immutable transaction: {path}"
            )
        return
    if path.parent.is_symlink():
        raise ThroughputQualificationError(
            f"{description} parent is symlinked: {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".publishing", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        if path.exists() or path.is_symlink():
            raise ThroughputQualificationError(
                f"{description} appeared concurrently: {path}"
            )
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def _absolute_path(value: object, *, description: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or any(character in value for character in ("\x00", "\n", "\r"))
    ):
        raise ThroughputQualificationError(
            f"{description} is not one safe absolute path"
        )
    path = Path(value)
    if not path.is_absolute() or str(Path(os.path.abspath(value))) != value:
        raise ThroughputQualificationError(
            f"{description} is not lexically canonical: {value!r}"
        )
    if path.is_symlink():
        raise ThroughputQualificationError(f"{description} is symlinked: {path}")
    return path.resolve(strict=False)


def _stable_order(parts: Sequence[object]) -> str:
    return hashlib.sha256(
        "\0".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def _stratum_tuple(cell: ExperimentCell) -> tuple[str, str, str, int]:
    return (
        cell.model_size,
        cell.reasoning_level.value,
        cell.topology.value,
        cell.n_agents,
    )


def _stratum_label(value: Sequence[object]) -> str:
    return "|".join(str(item) for item in value)


def expected_stratum_labels() -> tuple[str, ...]:
    labels = [
        _stratum_label((model, reasoning, Topology.SINGLE_AGENT.value, 1))
        for model in MODELS
        for reasoning in REASONING_LEVELS
    ]
    labels.extend(
        _stratum_label((model, reasoning, topology, count))
        for model in MODELS
        for reasoning in REASONING_LEVELS
        for topology in MAS_TOPOLOGIES
        for count in MAS_AGENT_COUNTS
    )
    return tuple(sorted(labels))


def _rotation_schedule() -> tuple[tuple[str, int, int], ...]:
    """A 24-row factorial schedule whose first half is marginally balanced."""

    first = [
        (benchmark, seed, (benchmark_index + seed_index) % 2)
        for benchmark_index, benchmark in enumerate(BENCHMARKS)
        for seed_index, seed in enumerate(SEEDS)
    ]
    second = [
        (benchmark, seed, 1 - prompt_index)
        for benchmark, seed, prompt_index in first
    ]
    return tuple(first + second)


def generate_qualification_cells() -> tuple[ExperimentCell, ...]:
    """Return the single deterministic 768-cell, 20-QID load design.

    The 540 multi-agent strata each receive one cell.  Their 24-row factorial
    schedule is exactly balanced over benchmark, seed, and prompt endpoints and,
    conditional on a context-sharing topology, over both context endpoints.

    The remaining 228 cells are the only structurally available prompt-level-1
    treatments: 7 or 8 single-agent cells per stratum.  A deterministic bipartite
    balancing pass selects every benchmark x seed variant exactly 19 times.  This is
    the closest feasible global prompt balance after mandatory coverage of all 540
    multi-agent strata.
    """

    mas_strata = [
        (model, reasoning, topology, count)
        for model in MODELS
        for reasoning in REASONING_LEVELS
        for topology in MAS_TOPOLOGIES
        for count in MAS_AGENT_COUNTS
    ]
    mas_strata.sort(key=_stable_order)
    rotations = _rotation_schedule()
    sharing_index = 0
    cells: list[ExperimentCell] = []
    for index, (model, reasoning, topology, count) in enumerate(mas_strata):
        benchmark, seed, prompt_index = rotations[index % len(rotations)]
        if topology == Topology.INDEPENDENT.value:
            context_level = ContextShareLevel.ARTIFACT_ONLY
        else:
            context_level = (
                ContextShareLevel.ARTIFACT_ONLY
                if sharing_index % 2 == 0
                else ContextShareLevel.PLUS_COT
            )
            sharing_index += 1
        cells.append(
            ExperimentCell(
                model_size=model,
                context_share_level=context_level,
                prompt_complexity_level=(0, 3)[prompt_index],
                reasoning_level=ReasoningLevel(reasoning),
                topology=Topology(topology),
                benchmark=benchmark,
                n_agents=count,
                rounds=2,
                n_samples=5,
                temperature=0.7,
                n_questions=QIDS_PER_CELL,
                seed=seed,
            )
        )

    single_strata = [
        (model, reasoning)
        for model in MODELS
        for reasoning in REASONING_LEVELS
    ]
    single_strata.sort(key=_stable_order)
    variants = tuple(
        (benchmark, seed) for benchmark in BENCHMARKS for seed in SEEDS
    )
    variant_counts: Counter[int] = Counter()
    for stratum_index, (model, reasoning) in enumerate(single_strata):
        quota = 8 if stratum_index < 18 else 7
        selected_variants: set[int] = set()
        offset = stratum_index % len(variants)
        for _ in range(quota):
            available = [
                index
                for index in range(len(variants))
                if index not in selected_variants
            ]
            variant_index = min(
                available,
                key=lambda value: (
                    variant_counts[value],
                    (value - offset) % len(variants),
                ),
            )
            selected_variants.add(variant_index)
            variant_counts[variant_index] += 1
            benchmark, seed = variants[variant_index]
            cells.append(
                ExperimentCell(
                    model_size=model,
                    context_share_level=ContextShareLevel.ARTIFACT_ONLY,
                    prompt_complexity_level=1,
                    reasoning_level=ReasoningLevel(reasoning),
                    topology=Topology.SINGLE_AGENT,
                    benchmark=benchmark,
                    n_agents=1,
                    rounds=1,
                    n_samples=5,
                    temperature=0.7,
                    n_questions=QIDS_PER_CELL,
                    seed=seed,
                )
            )

    cells.sort(
        key=lambda cell: _stable_order(
            (
                *_stratum_tuple(cell),
                cell.benchmark,
                cell.context_share_level.value,
                cell.prompt_complexity_level,
                cell.seed,
                cell.cell_id,
            )
        )
    )
    if len(cells) != CELL_COUNT:
        raise AssertionError(f"qualification design has {len(cells)} cells")
    if len({cell.cell_id for cell in cells}) != CELL_COUNT:
        raise AssertionError("qualification design contains duplicate cell IDs")
    strata = {_stratum_tuple(cell) for cell in cells}
    if len(strata) != EXPECTED_STRATA:
        raise AssertionError(
            f"qualification design covers {len(strata)} of {EXPECTED_STRATA} strata"
        )
    if set(variant_counts.values()) != {19}:
        raise AssertionError("single-agent benchmark/seed rotation is not exact")
    return tuple(cells)


def _balance_report(cells: Sequence[ExperimentCell]) -> dict[str, Any]:
    strata = Counter(_stratum_label(_stratum_tuple(cell)) for cell in cells)
    sharing = [
        cell
        for cell in cells
        if cell.topology in {Topology.DECENTRALIZED, Topology.CENTRALIZED}
    ]
    return {
        "benchmark": dict(
            sorted(Counter(cell.benchmark for cell in cells).items())
        ),
        "context_share_level": dict(
            sorted(
                Counter(cell.context_share_level.value for cell in cells).items()
            )
        ),
        "sharing_topology_context": dict(
            sorted(
                Counter(
                    cell.context_share_level.value for cell in sharing
                ).items()
            )
        ),
        "prompt_complexity_level": {
            str(key): value
            for key, value in sorted(
                Counter(cell.prompt_complexity_level for cell in cells).items()
            )
        },
        "seed": {
            str(key): value
            for key, value in sorted(Counter(cell.seed for cell in cells).items())
        },
        "stratum_cell_minimum": min(strata.values()),
        "stratum_cell_maximum": max(strata.values()),
    }


def build_load_plan() -> dict[str, Any]:
    """Build and self-hash the closed qualification load contract."""

    cells = generate_qualification_cells()
    identity: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PLAN_PROTOCOL,
        "run_id": QUALIFICATION_RUN_ID,
        "estimand_excluded": True,
        "primary_analysis_eligible": False,
        "cells": [cell.to_dict() for cell in cells],
        "cell_count": CELL_COUNT,
        "qids_per_cell": QIDS_PER_CELL,
        "qids": TOTAL_QIDS,
        "stratum_dimensions": list(STRATUM_DIMENSIONS),
        "rotation_dimensions": list(ROTATION_DIMENSIONS),
        "strata": EXPECTED_STRATA,
        "stratum_labels": list(expected_stratum_labels()),
        "balance": _balance_report(cells),
        "ceilings": list(CEILINGS),
        "maximum_microbatch": MAX_BATCH,
        "steady_384_seconds": STEADY_384_SECONDS,
        "maximum_observation_gap_seconds": MAX_OBSERVATION_GAP_SECONDS,
        "minimum_qids_per_day": MIN_QIDS_PER_DAY,
    }
    return _with_identity(identity, "plan_id")


_PLAN_FIELDS = {
    "schema_version",
    "protocol",
    "run_id",
    "estimand_excluded",
    "primary_analysis_eligible",
    "cells",
    "cell_count",
    "qids_per_cell",
    "qids",
    "stratum_dimensions",
    "rotation_dimensions",
    "strata",
    "stratum_labels",
    "balance",
    "ceilings",
    "maximum_microbatch",
    "steady_384_seconds",
    "maximum_observation_gap_seconds",
    "minimum_qids_per_day",
    "plan_id",
}


def validate_load_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != _PLAN_FIELDS:
        raise ThroughputQualificationError("qualification load-plan fields drifted")
    _verify_identity(value, "plan_id", description="qualification load plan")
    expected = build_load_plan()
    if dict(value) != expected:
        raise ThroughputQualificationError(
            "qualification load plan differs from the closed 768-cell design"
        )
    return expected


def _compact_prerequisite(
    manifest: Mapping[str, Any],
    name: str,
    *,
    identity_field: str,
) -> dict[str, str]:
    prerequisite = manifest.get("prerequisite_evidence")
    record = prerequisite.get(name) if isinstance(prerequisite, Mapping) else None
    expected = {
        "marker",
        "marker_sha256",
        "marker_size",
        "protocol",
        identity_field,
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected
        or not isinstance(record.get("marker"), str)
        or not Path(str(record["marker"])).is_absolute()
        or _SHA256_RE.fullmatch(str(record.get("marker_sha256", ""))) is None
        or _SHA256_RE.fullmatch(str(record.get(identity_field, ""))) is None
    ):
        raise ThroughputQualificationError(
            f"chain manifest {name} prerequisite binding is malformed"
        )
    return {
        "marker": str(record["marker"]),
        "marker_sha256": str(record["marker_sha256"]),
        identity_field: str(record[identity_field]),
    }


def load_qualification_context(
    chain_manifest: Path,
    *,
    verify_chain: bool = True,
) -> QualificationContext:
    """Load the renderer-authenticated chain namespace used by qualification."""

    lexical = Path(os.path.abspath(os.fspath(chain_manifest.expanduser())))
    if lexical.is_symlink() or not lexical.is_file():
        raise ThroughputQualificationError(
            f"chain manifest is missing or symlinked: {lexical}"
        )
    if stat.S_IMODE(lexical.stat().st_mode) & 0o222:
        raise ThroughputQualificationError("chain manifest must be read-only")
    if verify_chain:
        try:
            renderer.verify_chain(lexical)
        except renderer.ChainError as exc:
            raise ThroughputQualificationError(
                f"recovery chain verification failed: {exc}"
            ) from exc
    manifest = _read_json(
        lexical, description="throughput-qualification chain manifest", sealed=True
    )
    identity = dict(manifest)
    chain_id = identity.pop("chain_id", None)
    if (
        manifest.get("schema_version") != renderer.CHAIN_SCHEMA_VERSION
        or manifest.get("protocol") != "schema5-v1.2-r2-recovery-chain"
        or manifest.get("namespace") != renderer.CHAIN_NAMESPACE
        or manifest.get("release_id") != renderer.RELEASE_ID
        or manifest.get("release_tag") != renderer.RELEASE_TAG
        or not isinstance(chain_id, str)
        or _SHA256_RE.fullmatch(chain_id) is None
        or chain_id != _sha256_bytes(_canonical_bytes(identity))
    ):
        raise ThroughputQualificationError(
            "recovery chain identity is invalid"
        )
    results_root = _absolute_path(
        manifest.get("results_root"), description="chain results root"
    )
    recovery_root = _absolute_path(
        manifest.get("recovery_root"), description="chain recovery root"
    )
    readiness_root = _absolute_path(
        manifest.get("readiness_root"), description="chain readiness root"
    )
    state_root = _absolute_path(
        manifest.get("state_root"), description="chain state root"
    )
    server_pool_root = _absolute_path(
        manifest.get("server_pool_root"), description="chain server pool root"
    )
    release_root = _absolute_path(
        manifest.get("release_root"), description="chain release root"
    )
    hf_home = _absolute_path(
        manifest.get("hf_home"), description="chain Hugging Face root"
    )
    if (
        recovery_root != results_root / "recovery" / "schema5-v1"
        or readiness_root != recovery_root / "readiness"
        or state_root != results_root / control.CONTROL_STATE_DIRNAME
        or server_pool_root != results_root / "server_pools" / "schema5-v1"
    ):
        raise ThroughputQualificationError(
            "chain paths do not match the canonical schema-5 namespace"
        )
    qualification_root = readiness_root / QUALIFICATION_ROOT_NAME
    run_root = results_root / QUALIFICATION_RUN_ID
    if run_root.name in PRODUCTION_RUN_IDS:
        raise ThroughputQualificationError(
            "qualification run aliases a production run ID"
        )
    release_git_commit = str(manifest.get("release_git_commit", ""))
    release_tag_object = str(manifest.get("release_tag_object", ""))
    protected_binding = _compact_prerequisite(
        manifest, "protected_capacity", identity_field="marker_id"
    )
    try:
        protected_contract = protected_capacity.load_contract(
            protected_binding["marker"],
            expected_release_git_commit=release_git_commit,
            expected_release_tag_object=release_tag_object,
            expected_marker_id=protected_binding["marker_id"],
            expected_sha256=protected_binding["marker_sha256"],
        )
    except protected_capacity.ProtectedCapacityError as exc:
        raise ThroughputQualificationError(
            f"protected scientific placement authority is invalid: {exc}"
        ) from exc
    return QualificationContext(
        chain_manifest=lexical,
        chain_manifest_sha256=_sha256_file(lexical),
        chain_id=chain_id,
        results_root=results_root,
        recovery_root=recovery_root,
        readiness_root=readiness_root,
        qualification_base=qualification_root,
        qualification_root=qualification_root,
        run_root=run_root,
        dispatcher_state=qualification_root / DISPATCHER_STATE_DIRECTORY,
        state_root=state_root,
        server_pool_root=server_pool_root,
        release_worktree=release_root / "worktree",
        harness_prefix=release_root / "environments" / "harness",
        hf_home=hf_home,
        release_git_commit=release_git_commit,
        release_tag_object=release_tag_object,
        protected_capacity=protected_binding,
        protected_capacity_contract=protected_contract,
    )


def _control_guard(value: Mapping[str, Any]) -> dict[str, Any]:
    immutable = value.get("immutable")
    pinned_runs = immutable.get("runs") if isinstance(immutable, Mapping) else None
    run_ids = (
        [str(record.get("run_id")) for record in pinned_runs]
        if isinstance(pinned_runs, list)
        and all(isinstance(record, Mapping) for record in pinned_runs)
        else []
    )
    guard = {
        "immutable_sha256": value.get("immutable_sha256"),
        "desired_state": value.get("desired_state"),
        "drain_requested": value.get("drain_requested"),
        "rollout_generation": value.get("rollout_generation"),
        "production_run_ids": run_ids,
        "admission": value.get("admission"),
        "admission_ramp": value.get("admission_ramp"),
        "admission_safety_hold": value.get("admission_safety_hold"),
    }
    guard["guard_sha256"] = _sha256_bytes(_canonical_bytes(guard))
    return guard


def load_paused_control(context: QualificationContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify, but never mutate, the production control used as provenance."""

    try:
        value = control.load_control(context.state_root, verify_files=True)
    except control.ControlError as exc:
        raise ThroughputQualificationError(
            f"production control verification failed: {exc}"
        ) from exc
    guard = _control_guard(value)
    smoke = value.get("readiness", {}).get("smoke_runs")
    if (
        value.get("desired_state") != "paused"
        or value.get("drain_requested") is not False
        or set(guard["production_run_ids"]) != PRODUCTION_RUN_IDS
        or QUALIFICATION_RUN_ID in guard["production_run_ids"]
        or not isinstance(smoke, Mapping)
        or smoke.get("passed") is not True
        or _SHA256_RE.fullmatch(str(value.get("immutable_sha256", ""))) is None
    ):
        raise ThroughputQualificationError(
            "qualification requires paused, smoke-ready production control with "
            "exactly the three frozen production run IDs"
        )
    return value, guard


_READINESS_GENERATION_FIELDS = {
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


def _validate_readiness_generation(
    value: Mapping[str, Any],
    *,
    control_value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact serving generation proved for the next paused rollout."""

    current = control_value.get("rollout_generation")
    integer_fields = (
        "allowed_generation_tuple_count",
        "capacity_generation",
        "rollout_generation",
    )
    if (
        set(value) != _READINESS_GENERATION_FIELDS
        or not isinstance(current, int)
        or isinstance(current, bool)
        or current < 0
        or value.get("rollout_generation") != current + 1
        or any(
            not isinstance(value.get(field), int)
            or isinstance(value.get(field), bool)
            or int(value[field]) < 1
            for field in integer_fields
        )
        or not isinstance(value.get("marker_path"), str)
        or not Path(str(value["marker_path"])).is_absolute()
        or any(
            _SHA256_RE.fullmatch(str(value.get(field, ""))) is None
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
        raise ThroughputQualificationError(
            "trusted serving/readiness generation is malformed or is not the "
            "paused control's exact next rollout"
        )
    return dict(value)


def load_readiness_generation(
    context: QualificationContext,
    control_value: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the sealed endpoint catalog and prove its exact fleet/readiness join."""

    target = int(control_value["rollout_generation"]) + 1
    try:
        catalog = control.load_trusted_generation_catalog(
            context.state_root,
            server_pool_root=context.server_pool_root,
        )
        # This is the same read-only authority check used immediately before resume:
        # it joins every endpoint to the effective fleet, current fleet-readiness
        # evidence, capacity marker, and exact rollout generation.
        binding = control._validate_resume_catalog_authority(  # noqa: SLF001
            context.state_root,
            control_value,
            target_rollout_generation=target,
            catalog=catalog,
        )
    except (
        control.ControlError,
        control.GenerationCatalogError,
        OSError,
        ValueError,
    ) as exc:
        raise ThroughputQualificationError(
            f"cannot prove the live serving/readiness generation: {exc}"
        ) from exc
    return _validate_readiness_generation(
        binding, control_value=control_value
    )


_ATTEMPT_POINTER_FIELDS = {
    "schema_version",
    "protocol",
    "chain_id",
    "attempt_ordinal",
    "attempt_id",
    "attempt_root",
    "run_root",
    "dispatcher_state",
    "readiness_generation",
    "predecessor",
    "additive_retry",
    "created_at",
    "created_timestamp",
    "pointer_id",
}
_CURRENT_ATTEMPT_FIELDS = {
    "schema_version",
    "protocol",
    "attempt_id",
    "pointer",
    "pointer_sha256",
    "pointer_id",
    "current_id",
}


def _attempt_id(readiness_generation: Mapping[str, Any]) -> str:
    return (
        f"g{int(readiness_generation['rollout_generation']):06d}-"
        f"c{int(readiness_generation['capacity_generation']):06d}-"
        f"{str(readiness_generation['catalog_id'])}"
    )


def _attempt_pointer_ref(
    path: Path, pointer: Mapping[str, Any]
) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "pointer_id": str(pointer["pointer_id"]),
        "attempt_id": str(pointer["attempt_id"]),
    }


def _attempt_context_from_pointer(
    base: QualificationContext,
    *,
    path: Path,
    pointer: Mapping[str, Any],
) -> QualificationContext:
    return replace(
        base,
        qualification_root=Path(str(pointer["attempt_root"])),
        run_root=Path(str(pointer["run_root"])),
        dispatcher_state=Path(str(pointer["dispatcher_state"])),
        attempt_pointer_path=path,
        attempt_pointer=dict(pointer),
    )


def _validate_attempt_pointer(
    value: Mapping[str, Any],
    *,
    base: QualificationContext,
    path: Path,
    expected_ordinal: int,
    predecessor: Mapping[str, str] | None,
) -> dict[str, Any]:
    if set(value) != _ATTEMPT_POINTER_FIELDS:
        raise ThroughputQualificationError(
            "qualification attempt-pointer fields drifted"
        )
    _verify_identity(
        value,
        "pointer_id",
        description="qualification attempt pointer",
    )
    timestamp = value.get("created_timestamp")
    readiness = value.get("readiness_generation")
    attempt_id = (
        _attempt_id(readiness)
        if isinstance(readiness, Mapping)
        else ""
    )
    expected_attempt_root = (
        base.qualification_base / ATTEMPT_DIRECTORY / attempt_id
    )
    expected_run_root = (
        base.results_root
        / ATTEMPT_RUN_DIRECTORY
        / attempt_id
        / QUALIFICATION_RUN_ID
    )
    expected_path = (
        base.qualification_base
        / ATTEMPT_POINTER_DIRECTORY
        / f"{expected_ordinal:06d}-{attempt_id}.json"
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol") != ATTEMPT_POINTER_PROTOCOL
        or value.get("chain_id") != base.chain_id
        or value.get("attempt_ordinal") != expected_ordinal
        or value.get("attempt_id") != attempt_id
        or path != expected_path
        or value.get("attempt_root") != str(expected_attempt_root)
        or value.get("run_root") != str(expected_run_root)
        or value.get("dispatcher_state")
        != str(expected_attempt_root / DISPATCHER_STATE_DIRECTORY)
        or value.get("predecessor") != predecessor
        or (
            expected_ordinal == 1
            and value.get("additive_retry") is not None
        )
        or (
            expected_ordinal > 1
            and not isinstance(value.get("additive_retry"), Mapping)
        )
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or float(timestamp) <= 0
        or value.get("created_at") != _utc(float(timestamp))
    ):
        raise ThroughputQualificationError(
            "qualification attempt pointer identity is invalid"
        )
    assert isinstance(readiness, Mapping)
    # A pointer proves a generation identity independently of the current control.
    if (
        set(readiness) != _READINESS_GENERATION_FIELDS
        or any(
            not isinstance(readiness.get(field), int)
            or isinstance(readiness.get(field), bool)
            or int(readiness[field]) < 1
            for field in (
                "allowed_generation_tuple_count",
                "capacity_generation",
                "rollout_generation",
            )
        )
        or not isinstance(readiness.get("marker_path"), str)
        or not Path(str(readiness["marker_path"])).is_absolute()
        or any(
            _SHA256_RE.fullmatch(str(readiness.get(field, ""))) is None
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
        raise ThroughputQualificationError(
            "qualification attempt readiness generation is malformed"
        )
    return dict(value)


def _load_attempt_pointers(
    base: QualificationContext,
) -> list[tuple[Path, dict[str, Any]]]:
    root = base.qualification_base / ATTEMPT_POINTER_DIRECTORY
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ThroughputQualificationError(
            f"qualification attempt-pointer directory is unsafe: {root}"
        )
    paths = sorted(root.iterdir())
    if any(path.suffix != ".json" for path in paths):
        raise ThroughputQualificationError(
            "qualification attempt-pointer directory contains unexpected files"
        )
    loaded: list[tuple[Path, dict[str, Any]]] = []
    predecessor: dict[str, str] | None = None
    prior_readiness: Mapping[str, Any] | None = None
    for ordinal, path in enumerate(paths, start=1):
        pointer = _validate_attempt_pointer(
            _read_json(
                path,
                description=f"qualification attempt pointer {ordinal}",
                sealed=True,
            ),
            base=base,
            path=path,
            expected_ordinal=ordinal,
            predecessor=predecessor,
        )
        readiness = pointer["readiness_generation"]
        if prior_readiness is not None and (
            int(readiness["rollout_generation"])
            <= int(prior_readiness["rollout_generation"])
            or int(readiness["capacity_generation"])
            <= int(prior_readiness["capacity_generation"])
            or readiness["catalog_id"] == prior_readiness["catalog_id"]
        ):
            raise ThroughputQualificationError(
                "qualification attempt generations are not strictly increasing"
            )
        loaded.append((path, pointer))
        predecessor = _attempt_pointer_ref(path, pointer)
        prior_readiness = readiness
    for path, pointer in loaded[:-1]:
        context = _attempt_context_from_pointer(
            base, path=path, pointer=pointer
        )
        failure_path = context.qualification_root / FAILURE_NAME
        if not failure_path.is_file() or failure_path.is_symlink():
            raise ThroughputQualificationError(
                "a superseded qualification attempt is not terminally failed"
            )
        _load_terminal_failure(context)
        _assert_tree_read_only(
            context.qualification_root,
            description="superseded qualification attempt",
        )
        _assert_tree_read_only(
            context.run_root,
            description="superseded qualification run",
        )
    return loaded


def _current_attempt_payload(
    path: Path, pointer: Mapping[str, Any]
) -> dict[str, Any]:
    reference = _attempt_pointer_ref(path, pointer)
    return _with_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": CURRENT_ATTEMPT_PROTOCOL,
            "attempt_id": reference["attempt_id"],
            "pointer": reference["path"],
            "pointer_sha256": reference["sha256"],
            "pointer_id": reference["pointer_id"],
        },
        "current_id",
    )


def _validate_current_attempt(
    value: Mapping[str, Any],
    *,
    known: Sequence[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(value) != _CURRENT_ATTEMPT_FIELDS:
        raise ThroughputQualificationError(
            "current qualification-attempt fields drifted"
        )
    _verify_identity(
        value,
        "current_id",
        description="current qualification attempt",
    )
    matches = [
        (path, pointer)
        for path, pointer in known
        if (
            value.get("pointer") == str(path)
            and value.get("pointer_sha256") == _sha256_file(path)
            and value.get("pointer_id") == pointer["pointer_id"]
            and value.get("attempt_id") == pointer["attempt_id"]
        )
    ]
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol") != CURRENT_ATTEMPT_PROTOCOL
        or len(matches) != 1
    ):
        raise ThroughputQualificationError(
            "current qualification-attempt pointer is invalid"
        )
    return dict(value)


def _replace_current_attempt(
    base: QualificationContext,
    *,
    path: Path,
    pointer: Mapping[str, Any],
    known: Sequence[tuple[Path, Mapping[str, Any]]],
) -> None:
    destination = base.qualification_base / CURRENT_ATTEMPT_NAME
    payload = _current_attempt_payload(path, pointer)
    _assert_no_stale_publications(
        destination,
        description="current qualification attempt",
    )
    if destination.exists() or destination.is_symlink():
        existing = _validate_current_attempt(
            _read_json(
                destination,
                description="current qualification attempt",
                sealed=True,
            ),
            known=known,
        )
        if existing == payload:
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".publishing",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        encoded = _canonical_bytes(payload)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _profile_counts_from_fleet_payload(
    payload: Mapping[str, Any],
) -> dict[str, int]:
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ThroughputQualificationError(
            "fleet contract profiles are malformed"
        )
    counts: dict[str, int] = {}
    for row in profiles:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("serving_profile"), str)
            or row["serving_profile"] in counts
            or not isinstance(row.get("replicas"), list)
        ):
            raise ThroughputQualificationError(
                "fleet contract profile/replica layout is malformed"
            )
        counts[str(row["serving_profile"])] = len(row["replicas"])
    return counts


def _validate_additive_retry(
    base: QualificationContext,
    *,
    previous_path: Path,
    previous: Mapping[str, Any],
    control_value: Mapping[str, Any],
    readiness_generation: Mapping[str, Any],
) -> dict[str, Any]:
    previous_context = _attempt_context_from_pointer(
        base, path=previous_path, pointer=previous
    )
    failure = _load_terminal_failure(previous_context)
    scaling = failure.get("additive_scaling_requirement")
    assert isinstance(scaling, Mapping)
    authority = _read_json(
        previous_context.qualification_root / EXECUTION_AUTHORITY_NAME,
        description="prior qualification execution authority",
        sealed=True,
    )
    _verify_identity(
        authority,
        "authority_id",
        description="prior qualification execution authority",
    )
    old_environment = authority.get("runtime_environment")
    immutable = control_value.get("immutable")
    if (
        not isinstance(old_environment, Mapping)
        or authority.get("release_git_commit") != base.release_git_commit
        or not isinstance(immutable, Mapping)
        or immutable.get("git_commit") != base.release_git_commit
    ):
        raise ThroughputQualificationError(
            "retry would supersede the sealed release or lacks its prior fleet "
            "environment"
        )
    old_path = Path(
        str(old_environment.get("ASYS_FLEET_CONTRACT_PATH", ""))
    )
    old_sha256 = str(
        old_environment.get("ASYS_FLEET_CONTRACT_SHA256", "")
    )
    previous_readiness = previous["readiness_generation"]
    if (
        not old_path.is_absolute()
        or old_path.is_symlink()
        or not old_path.is_file()
        or _sha256_file(old_path) != old_sha256
        or old_sha256 != previous_readiness["fleet_contract_sha256"]
        or readiness_generation["release_fleet_contract_sha256"]
        != previous_readiness["release_fleet_contract_sha256"]
        or int(readiness_generation["capacity_generation"])
        != int(previous_readiness["capacity_generation"]) + 1
        or int(readiness_generation["rollout_generation"])
        <= int(previous_readiness["rollout_generation"])
        or readiness_generation["catalog_id"]
        == previous_readiness["catalog_id"]
    ):
        raise ThroughputQualificationError(
            "repeat qualification requires a fresh additive fleet/readiness "
            "generation under the same immutable release"
        )
    try:
        new_binding = control.effective_fleet_contract_binding(
            control_value, verify_files=True
        )
    except control.ControlError as exc:
        raise ThroughputQualificationError(
            f"fresh additive fleet binding is invalid: {exc}"
        ) from exc
    if (
        readiness_generation["fleet_contract_sha256"]
        != new_binding["sha256"]
        or int(readiness_generation["capacity_generation"])
        != int(new_binding["capacity_generation"])
    ):
        raise ThroughputQualificationError(
            "fresh readiness does not bind the effective additive fleet"
        )
    new_path = Path(str(new_binding["path"])).resolve()
    try:
        control._assert_additive_capacity_contract(  # noqa: SLF001
            old_path.resolve(), new_path
        )
        new_fleet = control.load_effective_fleet_contract(
            control_value, verify_files=True
        )
    except control.ControlError as exc:
        raise ThroughputQualificationError(
            f"retry fleet is not a verified additive contract: {exc}"
        ) from exc
    old_payload = _read_json(
        old_path,
        description="prior effective fleet contract",
        sealed=True,
    )
    old_counts = _profile_counts_from_fleet_payload(old_payload)
    new_counts = {
        str(key): int(value)
        for key, value in dict(new_binding["profile_replicas"]).items()
    }
    profile = str(scaling.get("serving_profile", ""))
    tensor_parallel_size = int(scaling.get("tensor_parallel_size", 0))
    if profile not in old_counts or tensor_parallel_size not in {1, 2}:
        raise ThroughputQualificationError(
            "prior failure's additive profile requirement is not represented "
            "by the sealed fleet contract"
        )
    required_deltas = {
        name: int(name == profile) for name in old_counts
    }
    observed_deltas = {
        name: new_counts.get(name, -1) - old_counts[name]
        for name in old_counts
    }
    appended = (
        new_fleet.by_profile.get(profile, ())[-1:]
        if profile in new_fleet.by_profile
        else ()
    )
    if (
        set(new_counts) != set(old_counts)
        or observed_deltas != required_deltas
        or len(appended) != 1
        or appended[0].replica_index != old_counts[profile]
        or appended[0].gpus_per_replica != tensor_parallel_size
        or int(new_binding["logical_replicas"])
        != int(old_payload.get("logical_replica_count", -1)) + 1
        or int(new_binding["allocated_gpus"])
        != int(old_payload.get("allocated_gpu_count", -1))
        + tensor_parallel_size
    ):
        raise ThroughputQualificationError(
            "fresh fleet generation does not implement exactly the failed "
            "attempt's additive replica requirement"
        )
    return {
        "previous_failure_id": str(failure["failure_id"]),
        "serving_profile": profile,
        "additional_replicas": 1,
        "tensor_parallel_size": tensor_parallel_size,
        "from_capacity_generation": int(
            previous_readiness["capacity_generation"]
        ),
        "to_capacity_generation": int(
            readiness_generation["capacity_generation"]
        ),
        "from_rollout_generation": int(
            previous_readiness["rollout_generation"]
        ),
        "to_rollout_generation": int(
            readiness_generation["rollout_generation"]
        ),
        "from_fleet_contract_sha256": old_sha256,
        "to_fleet_contract_sha256": str(new_binding["sha256"]),
        "validation": (
            "schema5_control._assert_additive_capacity_contract+"
            "exact-required-profile-delta"
        ),
    }


def publish_capacity_transition_authority(
    chain_manifest: Path,
    *,
    submission_receipt: Path,
    failed_job_id: str,
    failed_comment: str,
    apply: bool,
    verify_chain: bool = True,
    now: float | None = None,
) -> dict[str, Any]:
    """Seal post-transition authority for a same-release stage-18 suffix repair."""

    base = load_qualification_context(
        chain_manifest,
        verify_chain=verify_chain,
    )
    if (
        not failed_job_id.isdigit()
        or not failed_comment
        or any(character in failed_comment for character in "\r\n")
    ):
        raise ThroughputQualificationError(
            "failed qualification job identity is malformed"
        )
    receipt_path = Path(
        os.path.abspath(os.fspath(submission_receipt.expanduser()))
    )
    receipt = _read_json(
        receipt_path,
        description="qualification failure submission receipt",
        sealed=True,
    )
    receipt_sha256 = _sha256_file(receipt_path)
    _verify_identity(
        receipt,
        "receipt_id",
        description="qualification failure submission receipt",
    )
    rows = [
        row
        for row in receipt.get("jobs", [])
        if isinstance(row, Mapping)
        and row.get("name") == "throughput_qualification"
    ]
    if (
        receipt.get("chain_id") != base.chain_id
        or receipt.get("passed") is not True
        or receipt.get("manifest") != str(base.chain_manifest)
        or receipt.get("manifest_sha256") != base.chain_manifest_sha256
        or len(rows) != 1
        or str(rows[0].get("job_id", "")) != failed_job_id
        or rows[0].get("comment") != failed_comment
    ):
        raise ThroughputQualificationError(
            "submission receipt does not bind the exact failed stage-18 job"
        )
    timestamp = time.time() if now is None else float(now)
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise ThroughputQualificationError(
            "capacity-transition timestamp is invalid"
        )
    destination = base.qualification_base / CAPACITY_TRANSITION_NAME
    with _qualification_lock(base.qualification_base):
        locked_receipt = _read_json(
            receipt_path,
            description="qualification failure submission receipt",
            sealed=True,
        )
        if (
            locked_receipt != receipt
            or _sha256_file(receipt_path) != receipt_sha256
        ):
            raise ThroughputQualificationError(
                "qualification failure submission receipt changed before "
                "capacity-transition publication"
            )
        if (
            base.qualification_base / MARKER_NAME
        ).exists() or (
            base.qualification_base / MARKER_NAME
        ).is_symlink():
            raise ThroughputQualificationError(
                "successful qualification cannot publish capacity transition"
            )
        control_value, guard = load_paused_control(base)
        readiness = load_readiness_generation(base, control_value)
        pointers = _load_attempt_pointers(base)
        if not pointers:
            raise ThroughputQualificationError(
                "capacity transition has no failed qualification attempt"
            )
        previous_path, previous = pointers[-1]
        current = load_current_attempt_context(
            base,
            require_success=False,
        )
        if current.attempt_pointer_path != previous_path:
            raise ThroughputQualificationError(
                "capacity transition does not start from the current attempt"
            )
        failure = _load_terminal_failure(current)
        _assert_tree_read_only(
            current.qualification_root,
            description="failed qualification attempt",
        )
        _assert_tree_read_only(
            current.run_root,
            description="failed qualification run",
        )
        additive = _validate_additive_retry(
            base,
            previous_path=previous_path,
            previous=previous,
            control_value=control_value,
            readiness_generation=readiness,
        )
        identity = {
            "schema_version": SCHEMA_VERSION,
            "protocol": CAPACITY_TRANSITION_PROTOCOL,
            "passed": True,
            "release_id": renderer.RELEASE_ID,
            "release_tag": renderer.RELEASE_TAG,
            "release_git_commit": base.release_git_commit,
            "release_tag_object": base.release_tag_object,
            "chain_namespace": renderer.CHAIN_NAMESPACE,
            "chain_id": base.chain_id,
            "manifest": str(base.chain_manifest),
            "manifest_sha256": base.chain_manifest_sha256,
            "protected_capacity": dict(base.protected_capacity),
            "failed_attempt": _completion_attempt_binding(current),
            "failure": {
                "path": str(
                    current.qualification_root / FAILURE_NAME
                ),
                "sha256": _sha256_file(
                    current.qualification_root / FAILURE_NAME
                ),
                "failure_id": failure["failure_id"],
            },
            "submission_receipt": {
                "path": str(receipt_path),
                "sha256": _sha256_file(receipt_path),
                "receipt_id": receipt["receipt_id"],
            },
            "failed_stage": {
                "name": "throughput_qualification",
                "job_id": failed_job_id,
                "comment": failed_comment,
            },
            "paused_control": guard,
            "additive_transition": additive,
            "readiness_generation": dict(readiness),
            "created_at": _utc(timestamp),
            "created_timestamp": timestamp,
        }
        authority = _with_identity(identity, "transition_id")
        if apply:
            # The transition proof is useful only while the exact paused
            # control/readiness pair that authorized it remains current.
            final_control, final_guard = load_paused_control(base)
            final_readiness = load_readiness_generation(
                base,
                final_control,
            )
            if (
                final_control != control_value
                or final_guard != guard
                or final_readiness != readiness
                or _read_json(
                    receipt_path,
                    description="qualification failure submission receipt",
                    sealed=True,
                )
                != receipt
                or _sha256_file(receipt_path) != receipt_sha256
            ):
                raise ThroughputQualificationError(
                    "qualification transition authority drifted before its "
                    "marker-last commit"
                )
            _write_once(
                destination,
                authority,
                description="qualification capacity-transition authority",
            )
        return {
            "status": "published" if apply else "dry_run",
            "writes_performed": apply,
            "transition": authority,
            "transition_marker": str(destination),
        }


def create_or_load_attempt_context(
    base: QualificationContext,
    *,
    control_value: Mapping[str, Any],
    readiness_generation: Mapping[str, Any],
    now: float,
) -> QualificationContext:
    """Select or create one marker-first, generation-scoped attempt."""

    if base.qualification_root != base.qualification_base:
        raise ThroughputQualificationError(
            "attempt selection requires the fixed qualification base context"
        )
    known = _load_attempt_pointers(base)
    if not known and base.qualification_base.exists():
        allowed = {
            LOCK_NAME,
            ATTEMPT_POINTER_DIRECTORY,
            CURRENT_ATTEMPT_NAME,
        }
        legacy = sorted(
            path.name
            for path in base.qualification_base.iterdir()
            if path.name not in allowed
        )
        if legacy:
            raise ThroughputQualificationError(
                "legacy unscoped qualification evidence requires an explicit "
                "read-only migration before generation-scoped attempts: "
                + ", ".join(legacy)
            )
    if known:
        latest_path, latest = known[-1]
        latest_context = _attempt_context_from_pointer(
            base, path=latest_path, pointer=latest
        )
        failure_path = latest_context.qualification_root / FAILURE_NAME
        success_path = latest_context.qualification_root / MARKER_NAME
        if success_path.exists() or success_path.is_symlink():
            if dict(latest["readiness_generation"]) != dict(
                readiness_generation
            ):
                raise ThroughputQualificationError(
                    "a successful qualification already owns this release chain"
                )
            _replace_current_attempt(
                base,
                path=latest_path,
                pointer=latest,
                known=known,
            )
            return latest_context
        if not failure_path.exists() and not failure_path.is_symlink():
            if dict(latest["readiness_generation"]) != dict(
                readiness_generation
            ):
                raise ThroughputQualificationError(
                    "an unfinished qualification attempt cannot be abandoned "
                    "for a different generation"
                )
            _replace_current_attempt(
                base,
                path=latest_path,
                pointer=latest,
                known=known,
            )
            return latest_context
        # Failure publication is a marker-first transaction: a process may have
        # died after the sealed failure marker became durable but before the two
        # attempt trees were chmod-sealed.  Validate the exact terminal marker,
        # then finish only that idempotent local sealing step before considering
        # a successor generation.
        _load_terminal_failure(latest_context)
        _seal_tree_read_only(
            latest_context.run_root,
            description="terminal qualification run",
        )
        _seal_tree_read_only(
            latest_context.qualification_root,
            description="terminal qualification attempt",
        )
        _assert_tree_read_only(
            latest_context.qualification_root,
            description="terminal qualification attempt",
        )
        _assert_tree_read_only(
            latest_context.run_root,
            description="terminal qualification run",
        )
        additive_retry = _validate_additive_retry(
            base,
            previous_path=latest_path,
            previous=latest,
            control_value=control_value,
            readiness_generation=readiness_generation,
        )
        predecessor = _attempt_pointer_ref(latest_path, latest)
    else:
        additive_retry = None
        predecessor = None
    ordinal = len(known) + 1
    attempt_id = _attempt_id(readiness_generation)
    attempt_root = (
        base.qualification_base / ATTEMPT_DIRECTORY / attempt_id
    )
    run_root = (
        base.results_root
        / ATTEMPT_RUN_DIRECTORY
        / attempt_id
        / QUALIFICATION_RUN_ID
    )
    pointer_path = (
        base.qualification_base
        / ATTEMPT_POINTER_DIRECTORY
        / f"{ordinal:06d}-{attempt_id}.json"
    )
    if (
        attempt_root.exists()
        or attempt_root.is_symlink()
        or run_root.exists()
        or run_root.is_symlink()
    ):
        raise ThroughputQualificationError(
            "attempt artifacts exist before their marker-first pointer"
        )
    pointer = _with_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": ATTEMPT_POINTER_PROTOCOL,
            "chain_id": base.chain_id,
            "attempt_ordinal": ordinal,
            "attempt_id": attempt_id,
            "attempt_root": str(attempt_root),
            "run_root": str(run_root),
            "dispatcher_state": str(
                attempt_root / DISPATCHER_STATE_DIRECTORY
            ),
            "readiness_generation": dict(readiness_generation),
            "predecessor": predecessor,
            "additive_retry": additive_retry,
            "created_at": _utc(now),
            "created_timestamp": float(now),
        },
        "pointer_id",
    )
    _write_once(
        pointer_path,
        pointer,
        description="qualification attempt pointer",
    )
    known = [*known, (pointer_path, pointer)]
    _replace_current_attempt(
        base,
        path=pointer_path,
        pointer=pointer,
        known=known,
    )
    return _attempt_context_from_pointer(
        base, path=pointer_path, pointer=pointer
    )


def load_current_attempt_context(
    base: QualificationContext,
    *,
    require_success: bool,
) -> QualificationContext:
    """Resolve only the latest immutable pointer through the fixed current cache."""

    known = _load_attempt_pointers(base)
    if not known:
        raise ThroughputQualificationError(
            "qualification has no generation-scoped attempt pointer"
        )
    current_path = base.qualification_base / CURRENT_ATTEMPT_NAME
    current = _validate_current_attempt(
        _read_json(
            current_path,
            description="current qualification attempt",
            sealed=True,
        ),
        known=known,
    )
    latest_path, latest = known[-1]
    expected = _current_attempt_payload(latest_path, latest)
    if current != expected:
        raise ThroughputQualificationError(
            "current qualification pointer does not select the latest attempt"
        )
    context = _attempt_context_from_pointer(
        base, path=latest_path, pointer=latest
    )
    if require_success and not (
        context.qualification_root / MARKER_NAME
    ).is_file():
        raise ThroughputQualificationError(
            "current qualification attempt is not successfully complete"
        )
    return context


def _completion_attempt_binding(
    context: QualificationContext,
) -> dict[str, Any]:
    if (
        context.attempt_pointer_path is None
        or context.attempt_pointer is None
    ):
        raise ThroughputQualificationError(
            "qualification completion requires a generation-scoped attempt"
        )
    readiness = context.attempt_pointer["readiness_generation"]
    reference = _attempt_pointer_ref(
        context.attempt_pointer_path, context.attempt_pointer
    )
    return {
        **reference,
        "attempt_root": str(context.qualification_root),
        "run_root": str(context.run_root),
        "rollout_generation": int(readiness["rollout_generation"]),
        "capacity_generation": int(readiness["capacity_generation"]),
        "trusted_generation_catalog_id": str(readiness["catalog_id"]),
    }


def authorize_client_placement(
    context: QualificationContext,
    *,
    client_partition: str,
    client_qos: str,
) -> dict[str, Any]:
    """Require one exact marker-authorized 384-cell client placement."""

    if (
        _PARTITION_RE.fullmatch(client_partition) is None
        or _PARTITION_RE.fullmatch(client_qos) is None
    ):
        raise ThroughputQualificationError(
            "qualification client partition/QOS is unsafe"
        )
    try:
        placement = protected_capacity.authorize_client(
            context.protected_capacity_contract,
            partition=client_partition,
            qos=client_qos,
            required_slots=CEILINGS[-1],
            required_reserve_jobs=QOS_RESERVE,
        )
    except protected_capacity.ProtectedCapacityError as exc:
        raise ThroughputQualificationError(
            f"qualification client placement is not authorized: {exc}"
        ) from exc
    return {
        "partition": placement.partition,
        "qos": placement.qos,
        "slots": int(placement.capacity["slots"]),
        "reserve_jobs": int(placement.capacity["reserve_jobs"]),
        "submit_headroom": int(placement.capacity["submit_headroom"]),
        "protected_capacity_marker_id": (
            context.protected_capacity_contract.marker_id
        ),
        "protected_capacity_marker_sha256": (
            context.protected_capacity_contract.sha256
        ),
    }


def resolve_client_placement(
    context: QualificationContext,
    *,
    client_partition: str | None,
    client_qos: str | None,
) -> dict[str, Any]:
    """Resolve an explicit pair or the unique full-capacity sealed marker row."""

    if (client_partition is None) != (client_qos is None):
        raise ThroughputQualificationError(
            "client partition and QOS must be supplied together"
        )
    if client_partition is not None and client_qos is not None:
        return authorize_client_placement(
            context,
            client_partition=client_partition,
            client_qos=client_qos,
        )
    authorized: list[dict[str, Any]] = []
    for placement in context.protected_capacity_contract.client_placements:
        try:
            authorized.append(
                authorize_client_placement(
                    context,
                    client_partition=placement.partition,
                    client_qos=placement.qos,
                )
            )
        except ThroughputQualificationError:
            continue
    if len(authorized) != 1:
        raise ThroughputQualificationError(
            "sealed protected-capacity marker must contain exactly one client "
            "placement that independently authorizes 384 cells plus 64 reserve jobs"
        )
    return authorized[0]


def _intent_identity(
    context: QualificationContext,
    *,
    client_partition: str,
    client_qos: str,
    readiness_generation: Mapping[str, Any],
    control_guard: Mapping[str, Any],
    created_timestamp: float,
) -> dict[str, Any]:
    plan = build_load_plan()
    placement = authorize_client_placement(
        context,
        client_partition=client_partition,
        client_qos=client_qos,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": INTENT_PROTOCOL,
        "run_id": QUALIFICATION_RUN_ID,
        "estimand_excluded": True,
        "primary_analysis_eligible": False,
        "chain_id": context.chain_id,
        "chain_manifest": str(context.chain_manifest),
        "chain_manifest_sha256": context.chain_manifest_sha256,
        "plan": str(context.qualification_root / PLAN_NAME),
        "plan_id": plan["plan_id"],
        "results_root": str(context.results_root),
        "run_root": str(context.run_root),
        "dispatcher_state": str(context.dispatcher_state),
        "production_state_root": str(context.state_root),
        "server_pool_root": str(context.server_pool_root),
        "release_worktree": str(context.release_worktree),
        "harness_prefix": str(context.harness_prefix),
        "hf_home": str(context.hf_home),
        "client_partition": client_partition,
        "client_qos": client_qos,
        "client_placement": placement,
        "readiness_generation": dict(readiness_generation),
        "control_guard": dict(control_guard),
        "protected_capacity": dict(context.protected_capacity),
        "execution_authority": {
            "path": str(
                context.qualification_root / EXECUTION_AUTHORITY_NAME
            ),
            "protocol": EXECUTION_AUTHORITY_PROTOCOL,
        },
        "created_at": _utc(created_timestamp),
        "created_timestamp": float(created_timestamp),
    }


_INTENT_FIELDS = {
    "schema_version",
    "protocol",
    "run_id",
    "estimand_excluded",
    "primary_analysis_eligible",
    "chain_id",
    "chain_manifest",
    "chain_manifest_sha256",
    "plan",
    "plan_id",
    "results_root",
    "run_root",
    "dispatcher_state",
    "production_state_root",
    "server_pool_root",
    "release_worktree",
    "harness_prefix",
    "hf_home",
    "client_partition",
    "client_qos",
    "client_placement",
    "readiness_generation",
    "control_guard",
    "protected_capacity",
    "execution_authority",
    "created_at",
    "created_timestamp",
    "intent_id",
}


def validate_intent(
    value: Mapping[str, Any],
    *,
    context: QualificationContext,
    client_partition: str | None = None,
    client_qos: str | None = None,
) -> dict[str, Any]:
    if set(value) != _INTENT_FIELDS:
        raise ThroughputQualificationError("qualification intent fields drifted")
    _verify_identity(value, "intent_id", description="qualification intent")
    timestamp = value.get("created_timestamp")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol") != INTENT_PROTOCOL
        or value.get("run_id") != QUALIFICATION_RUN_ID
        or value.get("estimand_excluded") is not True
        or value.get("primary_analysis_eligible") is not False
        or value.get("chain_id") != context.chain_id
        or value.get("chain_manifest") != str(context.chain_manifest)
        or value.get("chain_manifest_sha256") != context.chain_manifest_sha256
        or value.get("plan")
        != str(context.qualification_root / PLAN_NAME)
        or value.get("plan_id") != build_load_plan()["plan_id"]
        or value.get("results_root") != str(context.results_root)
        or value.get("run_root") != str(context.run_root)
        or value.get("dispatcher_state") != str(context.dispatcher_state)
        or value.get("production_state_root") != str(context.state_root)
        or value.get("server_pool_root") != str(context.server_pool_root)
        or value.get("release_worktree") != str(context.release_worktree)
        or value.get("harness_prefix") != str(context.harness_prefix)
        or value.get("hf_home") != str(context.hf_home)
        or not isinstance(value.get("client_partition"), str)
        or _PARTITION_RE.fullmatch(str(value["client_partition"])) is None
        or not isinstance(value.get("client_qos"), str)
        or _PARTITION_RE.fullmatch(str(value["client_qos"])) is None
        or (
            client_partition is not None
            and value.get("client_partition") != client_partition
        )
        or (
            client_qos is not None
            and value.get("client_qos") != client_qos
        )
        or value.get("client_placement")
        != authorize_client_placement(
            context,
            client_partition=str(value.get("client_partition", "")),
            client_qos=str(value.get("client_qos", "")),
        )
        or value.get("protected_capacity") != dict(context.protected_capacity)
        or value.get("execution_authority")
        != {
            "path": str(
                context.qualification_root / EXECUTION_AUTHORITY_NAME
            ),
            "protocol": EXECUTION_AUTHORITY_PROTOCOL,
        }
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or float(timestamp) <= 0
        or value.get("created_at") != _utc(float(timestamp))
    ):
        raise ThroughputQualificationError("qualification intent is invalid")
    guard = value.get("control_guard")
    if (
        not isinstance(guard, Mapping)
        or guard.get("desired_state") != "paused"
        or guard.get("drain_requested") is not False
        or set(guard.get("production_run_ids", [])) != PRODUCTION_RUN_IDS
        or QUALIFICATION_RUN_ID in guard.get("production_run_ids", [])
    ):
        raise ThroughputQualificationError(
            "qualification intent does not bind paused production control"
        )
    guard_identity = dict(guard)
    observed_guard_hash = guard_identity.pop("guard_sha256", None)
    if observed_guard_hash != _sha256_bytes(_canonical_bytes(guard_identity)):
        raise ThroughputQualificationError(
            "qualification production-control guard identity drifted"
        )
    readiness_generation = value.get("readiness_generation")
    if not isinstance(readiness_generation, Mapping):
        raise ThroughputQualificationError(
            "qualification intent lacks its trusted serving generation"
        )
    _validate_readiness_generation(
        readiness_generation,
        control_value={"rollout_generation": guard.get("rollout_generation")},
    )
    return dict(value)


def create_or_load_intent(
    context: QualificationContext,
    *,
    client_partition: str,
    client_qos: str,
    readiness_generation: Mapping[str, Any],
    control_guard: Mapping[str, Any],
    now: float,
) -> dict[str, Any]:
    authorize_client_placement(
        context,
        client_partition=client_partition,
        client_qos=client_qos,
    )
    root = context.qualification_root
    intent_path = root / INTENT_NAME
    if intent_path.exists() or intent_path.is_symlink():
        existing = validate_intent(
            _read_json(
                intent_path,
                description="qualification intent",
                sealed=True,
            ),
            context=context,
            client_partition=client_partition,
            client_qos=client_qos,
        )
        if existing["readiness_generation"] != dict(readiness_generation):
            raise ThroughputQualificationError(
                "trusted serving/readiness generation changed after qualification "
                "intent publication"
            )
        return existing
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ThroughputQualificationError(
                f"qualification root is unsafe: {root}"
            )
        unexpected = sorted(
            path.name for path in root.iterdir() if path.name != LOCK_NAME
        )
        if unexpected:
            raise ThroughputQualificationError(
                "qualification artifacts exist before the marker-first intent: "
                + ", ".join(unexpected)
            )
    else:
        root.mkdir(parents=True)
    intent = _with_identity(
        _intent_identity(
            context,
            client_partition=client_partition,
            client_qos=client_qos,
            readiness_generation=readiness_generation,
            control_guard=control_guard,
            created_timestamp=now,
        ),
        "intent_id",
    )
    _write_once(
        intent_path,
        intent,
        description="qualification marker-first intent",
    )
    return validate_intent(
        _read_json(
            intent_path, description="qualification intent", sealed=True
        ),
        context=context,
        client_partition=client_partition,
        client_qos=client_qos,
    )


def _policy_payload(
    *,
    control_value: Mapping[str, Any],
    manifest_sha256: str,
    benchmark_sha256: str,
) -> dict[str, Any]:
    immutable = control_value["immutable"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": QUALIFICATION_RUN_ID,
        "authoritative": True,
        "required_artifact_schema_version": 5,
        "accepted_manifest_sha256": manifest_sha256,
        "accepted_benchmark_contracts_sha256": benchmark_sha256,
        "accepted_model_contract_sha256": immutable["model_contract_sha256"],
        "release": {
            "release_id": immutable["release_id"],
            "git_commit": immutable["git_commit"],
            "source_tree_sha256": immutable["source_tree_sha256"],
        },
        "environment": {
            "harness_sha256": immutable["harness_environment_sha256"],
            "serving_sha256": immutable["serving_environment_sha256"],
        },
        "required_metadata_fields": list(REQUIRED_METADATA_FIELDS),
        "legacy_result_import_allowed": False,
    }
    payload["policy_id"] = _sha256_bytes(
        _compact_bytes({"domain": POLICY_DOMAIN, "payload": payload})
    )
    return payload


def _qualification_lineage(
    *,
    plan: Mapping[str, Any],
    manifest_sha256: str,
    benchmark_sha256: str,
) -> dict[str, Any]:
    routes = Counter(
        serving_profile_for_cell(cell).name
        for cell in generate_qualification_cells()
    )
    return _with_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": LINEAGE_PROTOCOL,
            "run_id": QUALIFICATION_RUN_ID,
            "estimand_excluded": True,
            "primary_analysis_eligible": False,
            "load_plan_id": plan["plan_id"],
            "manifest_sha256": manifest_sha256,
            "benchmark_contracts_sha256": benchmark_sha256,
            "cell_count": CELL_COUNT,
            "qids_per_cell": QIDS_PER_CELL,
            "qids": TOTAL_QIDS,
            "serving_profile_counts": dict(sorted(routes.items())),
        },
        "lineage_id",
    )


def initialize_qualification_run(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    control_value: Mapping[str, Any],
    benchmark_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Atomically initialize, or verify, the isolated estimand-excluded run."""

    plan = validate_load_plan(build_load_plan())
    _write_once(
        context.qualification_root / PLAN_NAME,
        plan,
        description="qualification load plan",
    )
    target = context.run_root
    if target.exists() or target.is_symlink():
        return verify_qualification_run(
            context, intent=intent, control_value=control_value
        ) | {"status": "already_initialized"}
    context.results_root.mkdir(parents=True, exist_ok=True)
    stage_parent = context.results_root / (
        f".{QUALIFICATION_RUN_ID}.initialization-incomplete"
    )
    if stage_parent.exists() or stage_parent.is_symlink():
        raise ThroughputQualificationError(
            f"incomplete qualification-run staging requires inspection: {stage_parent}"
        )
    stage_parent.mkdir(mode=0o700)
    staged = stage_parent / QUALIFICATION_RUN_ID
    staged.mkdir(mode=0o700)
    try:
        cells = generate_qualification_cells()
        manifest_payload = (
            json.dumps(
                [cell.to_dict() for cell in cells],
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        experiment_io.atomic_write_text(staged / "cells.json", manifest_payload)
        freeze_manifest(staged)
        snapshot = load_manifest(staged)
        contract_kwargs: dict[str, Any] = {}
        if benchmark_loader is not None:
            contract_kwargs["benchmark_loader"] = benchmark_loader
        contract_payload = build_run_benchmark_contracts(
            snapshot, **contract_kwargs
        )
        contracts = freeze_benchmark_contracts(
            staged, snapshot=snapshot, payload=contract_payload
        )
        policy = _policy_payload(
            control_value=control_value,
            manifest_sha256=snapshot.sha256,
            benchmark_sha256=contracts.sidecar_sha256,
        )
        policy_bytes = _canonical_bytes(policy)
        _write_once(
            staged / POLICY_FILENAME,
            policy_bytes,
            description="qualification artifact policy",
        )
        _write_once(
            staged / POLICY_CHECKSUM_FILENAME,
            (
                f"{_sha256_bytes(policy_bytes)}  {POLICY_FILENAME}\n"
            ).encode("utf-8"),
            description="qualification artifact-policy checksum",
        )
        lineage = _qualification_lineage(
            plan=plan,
            manifest_sha256=snapshot.sha256,
            benchmark_sha256=contracts.sidecar_sha256,
        )
        _write_once(
            staged / LINEAGE_NAME,
            lineage,
            description="qualification lineage",
        )
        (staged / "cells").mkdir(mode=0o755)
        marker = _with_identity(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": QUALIFICATION_RUN_ID,
                "intent_id": intent["intent_id"],
                "plan_id": plan["plan_id"],
                "manifest_sha256": snapshot.sha256,
                "benchmark_contracts_sha256": contracts.sidecar_sha256,
                "artifact_policy_sha256": _sha256_bytes(policy_bytes),
                "lineage_id": lineage["lineage_id"],
                "cell_count": CELL_COUNT,
                "qids": TOTAL_QIDS,
                "estimand_excluded": True,
                "primary_analysis_eligible": False,
                "cells_directory_empty_at_initialization": True,
            },
            "initialization_id",
        )
        _write_once(
            staged / INITIALIZED_NAME,
            marker,
            description="qualification initialization marker",
        )
        verify_qualification_run(
            QualificationContext(
                **{
                    **context.__dict__,
                    "run_root": staged,
                }
            ),
            intent=intent,
            control_value=control_value,
        )
        if target.exists() or target.is_symlink():
            raise ThroughputQualificationError(
                f"qualification run appeared during staging: {target}"
            )
        os.replace(staged, target)
        _fsync_directory(context.results_root)
        stage_parent.rmdir()
    except Exception:
        # Preserve the complete preimage for explicit fail-closed inspection.
        raise
    return verify_qualification_run(
        context, intent=intent, control_value=control_value
    ) | {"status": "initialized"}


def verify_qualification_run(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    control_value: Mapping[str, Any],
) -> dict[str, Any]:
    root = context.run_root
    if root.is_symlink() or not root.is_dir() or root.name != QUALIFICATION_RUN_ID:
        raise ThroughputQualificationError(
            f"qualification run root is missing or unsafe: {root}"
        )
    snapshot = load_manifest(root, verify_frozen=True)
    expected_cells = generate_qualification_cells()
    if snapshot.cells != expected_cells or len(snapshot.cells) != CELL_COUNT:
        raise ThroughputQualificationError(
            "qualification run manifest differs from the closed load plan"
        )
    catalog = VerifiedQuestionCatalog(root, snapshot=snapshot)
    policy = load_artifact_policy(root, required=True)
    assert policy is not None
    immutable = control_value["immutable"]
    if (
        policy.run_id != QUALIFICATION_RUN_ID
        or policy.accepted_manifest_sha256 != snapshot.sha256
        or policy.accepted_benchmark_contracts_sha256
        != catalog.sidecar_sha256
        or policy.accepted_model_contract_sha256
        != immutable["model_contract_sha256"]
        or policy.release.release_id != immutable["release_id"]
        or policy.release.git_commit != immutable["git_commit"]
        or policy.release.source_tree_sha256
        != immutable["source_tree_sha256"]
        or policy.environment.harness_sha256
        != immutable["harness_environment_sha256"]
        or policy.environment.serving_sha256
        != immutable["serving_environment_sha256"]
        or policy.legacy_result_import_allowed is not False
    ):
        raise ThroughputQualificationError(
            "qualification artifact policy differs from paused control provenance"
        )
    lineage = _read_json(
        root / LINEAGE_NAME,
        description="qualification lineage",
        sealed=True,
    )
    _verify_identity(lineage, "lineage_id", description="qualification lineage")
    initialization = _read_json(
        root / INITIALIZED_NAME,
        description="qualification initialization marker",
        sealed=True,
    )
    _verify_identity(
        initialization,
        "initialization_id",
        description="qualification initialization marker",
    )
    if (
        lineage.get("protocol") != LINEAGE_PROTOCOL
        or lineage.get("run_id") != QUALIFICATION_RUN_ID
        or lineage.get("estimand_excluded") is not True
        or lineage.get("primary_analysis_eligible") is not False
        or lineage.get("load_plan_id") != build_load_plan()["plan_id"]
        or lineage.get("manifest_sha256") != snapshot.sha256
        or lineage.get("benchmark_contracts_sha256")
        != catalog.sidecar_sha256
        or initialization.get("intent_id") != intent["intent_id"]
        or initialization.get("plan_id") != build_load_plan()["plan_id"]
        or initialization.get("manifest_sha256") != snapshot.sha256
        or initialization.get("benchmark_contracts_sha256")
        != catalog.sidecar_sha256
        or initialization.get("artifact_policy_sha256") != policy.file_sha256
        or initialization.get("cell_count") != CELL_COUNT
        or initialization.get("qids") != TOTAL_QIDS
        or initialization.get("estimand_excluded") is not True
        or initialization.get("primary_analysis_eligible") is not False
    ):
        raise ThroughputQualificationError(
            "qualification run exclusion or initialization identity drifted"
        )
    return {
        "run_id": QUALIFICATION_RUN_ID,
        "run_root": str(root),
        "manifest_sha256": snapshot.sha256,
        "benchmark_contracts_sha256": catalog.sidecar_sha256,
        "artifact_policy_sha256": policy.file_sha256,
        "cell_count": CELL_COUNT,
        "qids": TOTAL_QIDS,
        "estimand_excluded": True,
    }


_SCHEDULER_FIELDS = {
    "schema_version",
    "protocol",
    "intent_id",
    "sequence",
    "captured_timestamp",
    "ceiling",
    "squeue_complete",
    "sacct_complete",
    "errors",
    "jobs",
    "qualification_job_ids",
    "active_qualification_cells",
    "qualification_tasks_only",
    "production_run_ids",
    "dispatcher_ledger_sha256",
    "production_control_guard_sha256",
    "client_partition",
    "client_qos",
    "protected_capacity_marker_id",
    "protected_capacity_marker_sha256",
    "readiness_rollout_generation",
    "trusted_generation_catalog_id",
    "qualification_execution_authority_id",
    "qualification_execution_authority_sha256",
    "scheduler_id",
}
_SEMANTIC_FIELDS = {
    "schema_version",
    "protocol",
    "intent_id",
    "sequence",
    "captured_timestamp",
    "run_id",
    "manifest_sha256",
    "cells",
    "qids_per_cell",
    "expected_qids",
    "states",
    "validated_qids",
    "useful_qids",
    "strata_total",
    "strata_progress",
    "every_stratum_progress",
    "artifact_schema_counts",
    "integrity_incidents",
    "transport_censor_incidents",
    "semantic_id",
}
_OBSERVATION_FIELDS = {
    "schema_version",
    "protocol",
    "intent_id",
    "sequence",
    "captured_timestamp",
    "ceiling",
    "scheduler",
    "semantic",
    "observation_id",
}


def _execution_authority_evidence_binding(
    intent: Mapping[str, Any],
) -> dict[str, str]:
    """Read the sealed authority identity without renewing its runtime lease."""

    reference = intent.get("execution_authority")
    if (
        not isinstance(reference, Mapping)
        or set(reference) != {"path", "protocol"}
        or reference.get("protocol") != EXECUTION_AUTHORITY_PROTOCOL
        or not isinstance(reference.get("path"), str)
        or not Path(str(reference["path"])).is_absolute()
    ):
        raise ThroughputQualificationError(
            "qualification intent lacks its execution authority reference"
        )
    path = Path(str(reference["path"]))
    authority = _read_json(
        path,
        description="qualification execution authority",
        sealed=True,
    )
    if (
        authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("protocol") != EXECUTION_AUTHORITY_PROTOCOL
        or authority.get("intent_id") != intent.get("intent_id")
    ):
        raise ThroughputQualificationError(
            "qualification execution authority identity drifted"
        )
    _verify_identity(
        authority,
        "authority_id",
        description="qualification execution authority",
    )
    return {
        "path": str(path),
        "authority_id": str(authority["authority_id"]),
        "sha256": _sha256_file(path),
    }


def make_scheduler_evidence(
    *,
    intent_id: str,
    sequence: int,
    captured_timestamp: float,
    ceiling: int,
    jobs: Sequence[Mapping[str, Any]],
    qualification_job_ids: Sequence[str],
    active_qualification_cells: int,
    dispatcher_ledger_sha256: str | None,
    production_control_guard_sha256: str,
    client_partition: str,
    client_qos: str,
    protected_capacity_marker_id: str,
    protected_capacity_marker_sha256: str,
    readiness_rollout_generation: int,
    trusted_generation_catalog_id: str,
    qualification_execution_authority_id: str,
    qualification_execution_authority_sha256: str,
    squeue_complete: bool = True,
    sacct_complete: bool = True,
    errors: Sequence[str] = (),
    qualification_tasks_only: bool = True,
    production_run_ids: Sequence[str] = (),
) -> dict[str, Any]:
    return _with_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": SCHEDULER_PROTOCOL,
            "intent_id": intent_id,
            "sequence": sequence,
            "captured_timestamp": float(captured_timestamp),
            "ceiling": ceiling,
            "squeue_complete": squeue_complete,
            "sacct_complete": sacct_complete,
            "errors": list(errors),
            "jobs": [dict(job) for job in jobs],
            "qualification_job_ids": list(qualification_job_ids),
            "active_qualification_cells": active_qualification_cells,
            "qualification_tasks_only": qualification_tasks_only,
            "production_run_ids": list(production_run_ids),
            "dispatcher_ledger_sha256": dispatcher_ledger_sha256,
            "production_control_guard_sha256": production_control_guard_sha256,
            "client_partition": client_partition,
            "client_qos": client_qos,
            "protected_capacity_marker_id": protected_capacity_marker_id,
            "protected_capacity_marker_sha256": (
                protected_capacity_marker_sha256
            ),
            "readiness_rollout_generation": readiness_rollout_generation,
            "trusted_generation_catalog_id": trusted_generation_catalog_id,
            "qualification_execution_authority_id": (
                qualification_execution_authority_id
            ),
            "qualification_execution_authority_sha256": (
                qualification_execution_authority_sha256
            ),
        },
        "scheduler_id",
    )


def make_semantic_evidence(
    *,
    intent_id: str,
    sequence: int,
    captured_timestamp: float,
    manifest_sha256: str,
    states: Mapping[str, int],
    validated_qids: int,
    useful_qids: int,
    strata_progress: Mapping[str, int],
    artifact_schema_counts: Mapping[str, int],
    integrity_incidents: int = 0,
    transport_censor_incidents: int = 0,
) -> dict[str, Any]:
    progress = dict(sorted((str(key), int(value)) for key, value in strata_progress.items()))
    return _with_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": SEMANTIC_PROTOCOL,
            "intent_id": intent_id,
            "sequence": sequence,
            "captured_timestamp": float(captured_timestamp),
            "run_id": QUALIFICATION_RUN_ID,
            "manifest_sha256": manifest_sha256,
            "cells": CELL_COUNT,
            "qids_per_cell": QIDS_PER_CELL,
            "expected_qids": TOTAL_QIDS,
            "states": dict(sorted((str(key), int(value)) for key, value in states.items())),
            "validated_qids": validated_qids,
            "useful_qids": useful_qids,
            "strata_total": EXPECTED_STRATA,
            "strata_progress": progress,
            "every_stratum_progress": bool(progress)
            and all(value > 0 for value in progress.values()),
            "artifact_schema_counts": dict(
                sorted((str(key), int(value)) for key, value in artifact_schema_counts.items())
            ),
            "integrity_incidents": integrity_incidents,
            "transport_censor_incidents": transport_censor_incidents,
        },
        "semantic_id",
    )


def validate_scheduler_evidence(
    value: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    sequence: int | None = None,
) -> dict[str, Any]:
    if set(value) != _SCHEDULER_FIELDS:
        raise ThroughputQualificationError("scheduler evidence fields drifted")
    _verify_identity(value, "scheduler_id", description="scheduler evidence")
    timestamp = value.get("captured_timestamp")
    jobs = value.get("jobs")
    execution_authority = _execution_authority_evidence_binding(intent)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol") != SCHEDULER_PROTOCOL
        or value.get("intent_id") != intent["intent_id"]
        or not isinstance(value.get("sequence"), int)
        or isinstance(value.get("sequence"), bool)
        or value["sequence"] < 0
        or (sequence is not None and value["sequence"] != sequence)
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or float(timestamp) <= 0
        or value.get("ceiling") not in CEILINGS
        or value.get("squeue_complete") is not True
        or value.get("sacct_complete") is not True
        or value.get("errors") != []
        or not isinstance(jobs, list)
        or not all(
            isinstance(job, Mapping)
            and set(job)
            == {
                "job_id",
                "job_name",
                "state",
                "comment",
                "command",
                "source",
                "dependency",
            }
            and all(isinstance(job[field], str) for field in job)
            for job in jobs
        )
        or not isinstance(value.get("qualification_job_ids"), list)
        or not all(
            isinstance(job_id, str) and job_id
            for job_id in value["qualification_job_ids"]
        )
        or len(value["qualification_job_ids"])
        != len(set(value["qualification_job_ids"]))
        or not isinstance(value.get("active_qualification_cells"), int)
        or isinstance(value.get("active_qualification_cells"), bool)
        or not 0 <= value["active_qualification_cells"] <= value["ceiling"]
        or value.get("qualification_tasks_only") is not True
        or value.get("production_run_ids") != []
        or (
            value.get("dispatcher_ledger_sha256") is not None
            and _SHA256_RE.fullmatch(
                str(value["dispatcher_ledger_sha256"])
            )
            is None
        )
        or (
            value.get("active_qualification_cells", 0) > 0
            and value.get("dispatcher_ledger_sha256") is None
        )
        or value.get("production_control_guard_sha256")
        != intent["control_guard"]["guard_sha256"]
        or value.get("client_partition") != intent["client_partition"]
        or value.get("client_qos") != intent["client_qos"]
        or value.get("protected_capacity_marker_id")
        != intent["client_placement"]["protected_capacity_marker_id"]
        or value.get("protected_capacity_marker_sha256")
        != intent["client_placement"]["protected_capacity_marker_sha256"]
        or value.get("readiness_rollout_generation")
        != intent["readiness_generation"]["rollout_generation"]
        or value.get("trusted_generation_catalog_id")
        != intent["readiness_generation"]["catalog_id"]
        or value.get("qualification_execution_authority_id")
        != execution_authority["authority_id"]
        or value.get("qualification_execution_authority_sha256")
        != execution_authority["sha256"]
    ):
        raise ThroughputQualificationError(
            "scheduler evidence is incomplete, unclean, or exceeds its ceiling"
        )
    return dict(value)


def validate_semantic_evidence(
    value: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    sequence: int | None = None,
) -> dict[str, Any]:
    if set(value) != _SEMANTIC_FIELDS:
        raise ThroughputQualificationError("semantic evidence fields drifted")
    _verify_identity(value, "semantic_id", description="semantic evidence")
    timestamp = value.get("captured_timestamp")
    states = value.get("states")
    progress = value.get("strata_progress")
    schemas = value.get("artifact_schema_counts")
    integer_fields = (
        "validated_qids",
        "useful_qids",
        "integrity_incidents",
        "transport_censor_incidents",
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol") != SEMANTIC_PROTOCOL
        or value.get("intent_id") != intent["intent_id"]
        or not isinstance(value.get("sequence"), int)
        or isinstance(value.get("sequence"), bool)
        or value["sequence"] < 0
        or (sequence is not None and value["sequence"] != sequence)
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or float(timestamp) <= 0
        or value.get("run_id") != QUALIFICATION_RUN_ID
        or _SHA256_RE.fullmatch(str(value.get("manifest_sha256", ""))) is None
        or value.get("cells") != CELL_COUNT
        or value.get("qids_per_cell") != QIDS_PER_CELL
        or value.get("expected_qids") != TOTAL_QIDS
        or not isinstance(states, Mapping)
        or not states
        or not set(states).issubset(
            {state.value for state in CompletionState}
        )
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for count in states.values()
        )
        or sum(states.values()) != CELL_COUNT
        or any(
            not isinstance(value.get(field), int)
            or isinstance(value.get(field), bool)
            or not 0 <= value[field] <= TOTAL_QIDS
            for field in integer_fields
        )
        or value["useful_qids"] > value["validated_qids"]
        or value.get("strata_total") != EXPECTED_STRATA
        or not isinstance(progress, Mapping)
        or set(progress) != set(expected_stratum_labels())
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for count in progress.values()
        )
        or sum(progress.values()) != value["useful_qids"]
        or value.get("every_stratum_progress")
        is not all(count > 0 for count in progress.values())
        or not isinstance(schemas, Mapping)
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for count in schemas.values()
        )
        or sum(schemas.values()) != value["validated_qids"]
    ):
        raise ThroughputQualificationError(
            "semantic evidence violates the 768-cell/15,360-QID contract"
        )
    return dict(value)


def _evidence_filename(prefix: str, sequence: int) -> str:
    return f"{prefix}_{sequence:06d}.json"


def record_observation(
    root: Path,
    *,
    intent: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    semantic: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish one resumable marker-first scheduler+semantic transaction."""

    sequence = scheduler.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise ThroughputQualificationError("observation sequence is invalid")
    scheduler_value = validate_scheduler_evidence(
        scheduler, intent=intent, sequence=sequence
    )
    semantic_value = validate_semantic_evidence(
        semantic, intent=intent, sequence=sequence
    )
    if (
        scheduler_value["captured_timestamp"]
        != semantic_value["captured_timestamp"]
    ):
        raise ThroughputQualificationError(
            "scheduler and semantic evidence timestamps differ"
        )
    transaction = _with_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": OBSERVATION_TRANSACTION_PROTOCOL,
            "intent_id": intent["intent_id"],
            "sequence": sequence,
            # The exact validated payloads are the recovery preimage.  A crash after
            # either child write can finish from these bytes without recapturing a
            # different scheduler instant or silently discarding evidence.
            "scheduler": scheduler_value,
            "semantic": semantic_value,
        },
        "transaction_id",
    )
    transaction_path = (
        root
        / OBSERVATION_TRANSACTION_DIRECTORY
        / _evidence_filename("TRANSACTION", sequence)
    )
    _write_once(
        transaction_path,
        transaction,
        description=f"observation transaction {sequence}",
    )
    return _finalize_observation_transaction(
        root, intent=intent, transaction=transaction
    )


def _validate_observation_transaction(
    value: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    sequence: int,
) -> dict[str, Any]:
    if set(value) != {
        "schema_version",
        "protocol",
        "intent_id",
        "sequence",
        "scheduler",
        "semantic",
        "transaction_id",
    }:
        raise ThroughputQualificationError(
            f"observation transaction {sequence} fields drifted"
        )
    _verify_identity(
        value,
        "transaction_id",
        description=f"observation transaction {sequence}",
    )
    scheduler = value.get("scheduler")
    semantic = value.get("semantic")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol") != OBSERVATION_TRANSACTION_PROTOCOL
        or value.get("intent_id") != intent["intent_id"]
        or value.get("sequence") != sequence
        or not isinstance(scheduler, Mapping)
        or not isinstance(semantic, Mapping)
    ):
        raise ThroughputQualificationError(
            f"observation transaction {sequence} identity is invalid"
        )
    scheduler_value = validate_scheduler_evidence(
        scheduler, intent=intent, sequence=sequence
    )
    semantic_value = validate_semantic_evidence(
        semantic, intent=intent, sequence=sequence
    )
    if (
        scheduler_value["captured_timestamp"]
        != semantic_value["captured_timestamp"]
    ):
        raise ThroughputQualificationError(
            f"observation transaction {sequence} timestamps differ"
        )
    return dict(value)


def _finalize_observation_transaction(
    root: Path,
    *,
    intent: Mapping[str, Any],
    transaction: Mapping[str, Any],
) -> dict[str, Any]:
    sequence = int(transaction["sequence"])
    transaction_value = _validate_observation_transaction(
        transaction, intent=intent, sequence=sequence
    )
    scheduler_value = dict(transaction_value["scheduler"])
    semantic_value = dict(transaction_value["semantic"])
    scheduler_path = (
        root
        / SCHEDULER_DIRECTORY
        / _evidence_filename("SCHEDULER", sequence)
    )
    semantic_path = (
        root
        / SEMANTIC_DIRECTORY
        / _evidence_filename("SEMANTIC", sequence)
    )
    observation_path = (
        root
        / OBSERVATION_DIRECTORY
        / _evidence_filename("OBSERVATION", sequence)
    )
    _write_once(
        scheduler_path,
        scheduler_value,
        description=f"scheduler observation {sequence}",
    )
    _write_once(
        semantic_path,
        semantic_value,
        description=f"semantic observation {sequence}",
    )
    observation = _with_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": OBSERVATION_PROTOCOL,
            "intent_id": intent["intent_id"],
            "sequence": sequence,
            "captured_timestamp": scheduler_value["captured_timestamp"],
            "ceiling": scheduler_value["ceiling"],
            "scheduler": {
                "path": str(scheduler_path.resolve()),
                "sha256": _sha256_file(scheduler_path),
                "scheduler_id": scheduler_value["scheduler_id"],
            },
            "semantic": {
                "path": str(semantic_path.resolve()),
                "sha256": _sha256_file(semantic_path),
                "semantic_id": semantic_value["semantic_id"],
            },
        },
        "observation_id",
    )
    _write_once(
        observation_path,
        observation,
        description=f"qualification observation {sequence}",
    )
    return observation


def _numbered_files(directory: Path, prefix: str) -> dict[int, Path]:
    if not directory.exists():
        return {}
    if directory.is_symlink() or not directory.is_dir():
        raise ThroughputQualificationError(
            f"qualification evidence directory is unsafe: {directory}"
        )
    pattern = re.compile(rf"{re.escape(prefix)}_([0-9]{{6}})\.json\Z")
    result: dict[int, Path] = {}
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ThroughputQualificationError(
                f"unrecognized partial qualification evidence: {path}"
            )
        sequence = int(match.group(1))
        if sequence in result:
            raise ThroughputQualificationError(
                f"duplicate qualification evidence sequence {sequence}"
            )
        result[sequence] = path
    return result


def load_observations(
    root: Path,
    *,
    intent: Mapping[str, Any],
    recover_transactions: bool = False,
) -> list[dict[str, Any]]:
    """Load a contiguous, fully sealed observation journal or fail closed."""

    transaction_files = _numbered_files(
        root / OBSERVATION_TRANSACTION_DIRECTORY, "TRANSACTION"
    )
    scheduler_files = _numbered_files(root / SCHEDULER_DIRECTORY, "SCHEDULER")
    semantic_files = _numbered_files(root / SEMANTIC_DIRECTORY, "SEMANTIC")
    observation_files = _numbered_files(
        root / OBSERVATION_DIRECTORY, "OBSERVATION"
    )
    partial_sequences = (
        set(transaction_files)
        | set(scheduler_files)
        | set(semantic_files)
        | set(observation_files)
    )
    if partial_sequences != set(transaction_files):
        raise ThroughputQualificationError(
            "orphaned partial scheduler/semantic observation has no marker-first "
            "transaction"
        )
    if recover_transactions:
        for sequence, transaction_path in sorted(transaction_files.items()):
            transaction = _validate_observation_transaction(
                _read_json(
                    transaction_path,
                    description=f"observation transaction {sequence}",
                    sealed=True,
                ),
                intent=intent,
                sequence=sequence,
            )
            _finalize_observation_transaction(
                root, intent=intent, transaction=transaction
            )
        scheduler_files = _numbered_files(
            root / SCHEDULER_DIRECTORY, "SCHEDULER"
        )
        semantic_files = _numbered_files(
            root / SEMANTIC_DIRECTORY, "SEMANTIC"
        )
        observation_files = _numbered_files(
            root / OBSERVATION_DIRECTORY, "OBSERVATION"
        )
    if not (
        set(transaction_files)
        == set(scheduler_files)
        == set(semantic_files)
        == set(observation_files)
    ):
        raise ThroughputQualificationError(
            "marker-first observation transaction is incomplete; execute may "
            "resumably adopt its exact preimage"
        )
    sequences = sorted(observation_files)
    if sequences and sequences != list(range(len(sequences))):
        raise ThroughputQualificationError(
            "qualification observation sequence is not contiguous from zero"
        )
    loaded: list[dict[str, Any]] = []
    prior_timestamp = -math.inf
    for sequence in sequences:
        scheduler = validate_scheduler_evidence(
            _read_json(
                scheduler_files[sequence],
                description=f"scheduler evidence {sequence}",
                sealed=True,
            ),
            intent=intent,
            sequence=sequence,
        )
        semantic = validate_semantic_evidence(
            _read_json(
                semantic_files[sequence],
                description=f"semantic evidence {sequence}",
                sealed=True,
            ),
            intent=intent,
            sequence=sequence,
        )
        receipt = _read_json(
            observation_files[sequence],
            description=f"observation receipt {sequence}",
            sealed=True,
        )
        if set(receipt) != _OBSERVATION_FIELDS:
            raise ThroughputQualificationError(
                f"observation receipt {sequence} fields drifted"
            )
        _verify_identity(
            receipt,
            "observation_id",
            description=f"observation receipt {sequence}",
        )
        expected_scheduler_ref = {
            "path": str(scheduler_files[sequence].resolve()),
            "sha256": _sha256_file(scheduler_files[sequence]),
            "scheduler_id": scheduler["scheduler_id"],
        }
        expected_semantic_ref = {
            "path": str(semantic_files[sequence].resolve()),
            "sha256": _sha256_file(semantic_files[sequence]),
            "semantic_id": semantic["semantic_id"],
        }
        timestamp = float(receipt.get("captured_timestamp", -1))
        if (
            receipt.get("schema_version") != SCHEMA_VERSION
            or receipt.get("protocol") != OBSERVATION_PROTOCOL
            or receipt.get("intent_id") != intent["intent_id"]
            or receipt.get("sequence") != sequence
            or receipt.get("ceiling") != scheduler["ceiling"]
            or receipt.get("captured_timestamp")
            != scheduler["captured_timestamp"]
            or receipt.get("captured_timestamp")
            != semantic["captured_timestamp"]
            or receipt.get("scheduler") != expected_scheduler_ref
            or receipt.get("semantic") != expected_semantic_ref
            or timestamp <= prior_timestamp
        ):
            raise ThroughputQualificationError(
                f"observation receipt {sequence} binding is invalid"
            )
        prior_timestamp = timestamp
        loaded.append(
            {
                "receipt": receipt,
                "scheduler": scheduler,
                "semantic": semantic,
                "receipt_path": observation_files[sequence],
                "scheduler_path": scheduler_files[sequence],
                "semantic_path": semantic_files[sequence],
            }
        )
    return loaded


def evaluate_observations(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the complete scientific and scheduler acceptance contract."""

    if not observations:
        raise ThroughputQualificationError(
            "qualification has no scheduler/semantic evidence"
        )
    ceilings_seen: list[int] = []
    peak_by_ceiling = {ceiling: 0 for ceiling in CEILINGS}
    prior_useful = -1
    prior_validated = -1
    first_timestamp = float(observations[0]["receipt"]["captured_timestamp"])
    first_semantic = observations[0]["semantic"]
    if (
        observations[0]["scheduler"]["ceiling"] != CEILINGS[0]
        or observations[0]["scheduler"]["active_qualification_cells"] != 0
        or first_semantic["validated_qids"] != 0
        or first_semantic["useful_qids"] != 0
    ):
        raise ThroughputQualificationError(
            "qualification evidence must begin with a clean zero-QID ceiling-24 baseline"
        )
    completion_timestamp: float | None = None
    clean_384: list[Mapping[str, Any]] = []
    for observation in observations:
        scheduler = observation["scheduler"]
        semantic = observation["semantic"]
        ceiling = int(scheduler["ceiling"])
        if not ceilings_seen or ceilings_seen[-1] != ceiling:
            ceilings_seen.append(ceiling)
        if ceilings_seen != list(CEILINGS[: len(ceilings_seen)]):
            raise ThroughputQualificationError(
                "qualification ceilings were skipped, regressed, or reordered"
            )
        peak_by_ceiling[ceiling] = max(
            peak_by_ceiling[ceiling],
            int(scheduler["active_qualification_cells"]),
        )
        useful = int(semantic["useful_qids"])
        validated = int(semantic["validated_qids"])
        if useful < prior_useful or validated < prior_validated:
            raise ThroughputQualificationError(
                "qualification semantic progress regressed"
            )
        prior_useful, prior_validated = useful, validated
        if (
            semantic["integrity_incidents"] != 0
            or semantic["transport_censor_incidents"] != 0
            or scheduler["production_run_ids"] != []
            or scheduler["qualification_tasks_only"] is not True
        ):
            raise ThroughputQualificationError(
                "qualification contains an integrity, censor, or namespace incident"
            )
        if (
            completion_timestamp is None
            and semantic["states"] == {"complete": CELL_COUNT}
            and validated == TOTAL_QIDS
            and useful == TOTAL_QIDS
            and semantic["every_stratum_progress"] is True
            and semantic["artifact_schema_counts"] == {"5": TOTAL_QIDS}
        ):
            completion_timestamp = float(
                observation["receipt"]["captured_timestamp"]
            )
        if ceiling == 384:
            clean_384.append(observation)

    if ceilings_seen != list(CEILINGS):
        raise ThroughputQualificationError(
            f"qualification did not exercise every ceiling: {ceilings_seen}"
        )
    if any(peak_by_ceiling[ceiling] != ceiling for ceiling in CEILINGS):
        raise ThroughputQualificationError(
            f"qualification did not reconcile to every exact ceiling: {peak_by_ceiling}"
        )
    if not clean_384:
        raise ThroughputQualificationError(
            "qualification has no ceiling-384 evidence"
        )
    for left, right in zip(clean_384, clean_384[1:]):
        gap = (
            float(right["receipt"]["captured_timestamp"])
            - float(left["receipt"]["captured_timestamp"])
        )
        if gap > MAX_OBSERVATION_GAP_SECONDS:
            raise ThroughputQualificationError(
                f"ceiling-384 clean window contains a {gap:g}-second evidence gap"
            )
    steady_seconds = (
        float(clean_384[-1]["receipt"]["captured_timestamp"])
        - float(clean_384[0]["receipt"]["captured_timestamp"])
    )
    if steady_seconds < STEADY_384_SECONDS:
        raise ThroughputQualificationError(
            f"ceiling 384 is clean for only {steady_seconds:g} seconds"
        )
    if completion_timestamp is None:
        raise ThroughputQualificationError(
            "qualification is not semantically complete"
        )
    duration = completion_timestamp - first_timestamp
    if duration <= 0:
        raise ThroughputQualificationError(
            "qualification completion does not follow its baseline"
        )
    throughput = math.floor(TOTAL_QIDS * 86_400.0 / duration)
    if throughput < MIN_QIDS_PER_DAY:
        raise ThroughputQualificationError(
            f"qualification throughput {throughput:,} QIDs/day is below "
            f"{MIN_QIDS_PER_DAY:,}"
        )
    final_scheduler = observations[-1]["scheduler"]
    final_semantic = observations[-1]["semantic"]
    if (
        final_scheduler["active_qualification_cells"] != 0
        or final_semantic["states"] != {"complete": CELL_COUNT}
        or final_semantic["validated_qids"] != TOTAL_QIDS
        or final_semantic["useful_qids"] != TOTAL_QIDS
        or final_semantic["every_stratum_progress"] is not True
        or final_semantic["artifact_schema_counts"] != {"5": TOTAL_QIDS}
    ):
        raise ThroughputQualificationError(
            "final qualification observation is not quiescent and schema-5 complete"
        )
    return {
        "passed": True,
        "cells": CELL_COUNT,
        "qids": TOTAL_QIDS,
        "ceilings": list(CEILINGS),
        "peak_active_cells": {
            str(key): value for key, value in peak_by_ceiling.items()
        },
        "steady_384_seconds": math.floor(steady_seconds),
        "throughput_qids_per_day": throughput,
        "every_stratum_progress": True,
        "integrity_incidents": 0,
        "transport_censor_incidents": 0,
        "completion_timestamp": completion_timestamp,
        "observation_count": len(observations),
    }


def _evidence_summary(
    *,
    intent: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = []
    for observation in observations:
        inventory.append(
            {
                "sequence": observation["receipt"]["sequence"],
                "observation": {
                    "path": str(observation["receipt_path"].resolve()),
                    "sha256": _sha256_file(observation["receipt_path"]),
                    "observation_id": observation["receipt"]["observation_id"],
                },
                "scheduler": {
                    "path": str(observation["scheduler_path"].resolve()),
                    "sha256": _sha256_file(observation["scheduler_path"]),
                    "scheduler_id": observation["scheduler"]["scheduler_id"],
                },
                "semantic": {
                    "path": str(observation["semantic_path"].resolve()),
                    "sha256": _sha256_file(observation["semantic_path"]),
                    "semantic_id": observation["semantic"]["semantic_id"],
                },
            }
        )
    return _with_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": EVIDENCE_PROTOCOL,
            "intent_id": intent["intent_id"],
            "plan_id": intent["plan_id"],
            "run_id": QUALIFICATION_RUN_ID,
            "estimand_excluded": True,
            "primary_analysis_eligible": False,
            "observations": inventory,
            "evaluation": dict(evaluation),
        },
        "evidence_id",
    )


def _marker_payload(
    *,
    context: QualificationContext,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the renderer's exact completion-marker schema."""

    marker = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "passed": True,
        "release_id": renderer.RELEASE_ID,
        "release_tag": renderer.RELEASE_TAG,
        "release_git_commit": context.release_git_commit,
        "release_tag_object": context.release_tag_object,
        "chain_namespace": renderer.CHAIN_NAMESPACE,
        "chain_id": context.chain_id,
        "manifest": str(context.chain_manifest),
        "manifest_sha256": context.chain_manifest_sha256,
        "protected_capacity": dict(context.protected_capacity),
        "attempt": _completion_attempt_binding(context),
        "cells": CELL_COUNT,
        "qids": TOTAL_QIDS,
        "ceilings": list(CEILINGS),
        "steady_384_seconds": evaluation["steady_384_seconds"],
        "throughput_qids_per_day": evaluation["throughput_qids_per_day"],
        "every_stratum_progress": True,
        "integrity_incidents": 0,
        "transport_censor_incidents": 0,
    }
    return _with_identity(marker, "qualification_id")


def publish_completion(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish evidence summary then the renderer-compatible marker last."""

    if (
        context.qualification_root / FAILURE_NAME
    ).exists() or (
        context.qualification_root / FAILURE_NAME
    ).is_symlink():
        raise ThroughputQualificationError(
            "a terminally failed qualification intent cannot publish completion"
        )
    observations = load_observations(
        context.qualification_root, intent=intent
    )
    evaluation = evaluate_observations(observations)
    evidence = _evidence_summary(
        intent=intent,
        observations=observations,
        evaluation=evaluation,
    )
    _write_once(
        context.qualification_root / EVIDENCE_NAME,
        evidence,
        description="qualification aggregate evidence",
    )
    marker = _marker_payload(context=context, evaluation=evaluation)
    marker_path = context.qualification_root / MARKER_NAME
    _write_once(
        marker_path,
        marker,
        description="throughput qualification completion marker",
    )
    _seal_tree_read_only(
        context.run_root,
        description="successful qualification run",
    )
    _seal_tree_read_only(
        context.qualification_root,
        description="successful qualification attempt",
    )
    # The fixed-root marker is the chain-visible commit point and therefore must
    # be published only after both generation-scoped trees are durably sealed.
    root_marker_path = context.qualification_base / MARKER_NAME
    _write_once(
        root_marker_path,
        marker,
        description="current-generation throughput qualification marker",
    )
    return marker


def _load_and_verify_evidence(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    observations = load_observations(
        context.qualification_root, intent=intent
    )
    evaluation = evaluate_observations(observations)
    evidence_path = context.qualification_root / EVIDENCE_NAME
    evidence = _read_json(
        evidence_path,
        description="qualification aggregate evidence",
        sealed=True,
    )
    _verify_identity(
        evidence, "evidence_id", description="qualification aggregate evidence"
    )
    expected = _evidence_summary(
        intent=intent,
        observations=observations,
        evaluation=evaluation,
    )
    if evidence != expected:
        raise ThroughputQualificationError(
            "qualification aggregate evidence inventory drifted"
        )
    return evaluation, evidence


def verify_completed_qualification(
    chain_manifest: Path,
    *,
    verify_chain: bool = True,
    verify_renderer: bool = True,
) -> dict[str, Any]:
    """Verify all sealed evidence without contacting Slurm or mutable results."""

    base = load_qualification_context(
        chain_manifest, verify_chain=verify_chain
    )
    context = load_current_attempt_context(base, require_success=True)
    _assert_tree_read_only(
        context.run_root,
        description="successful qualification run",
    )
    _assert_tree_read_only(
        context.qualification_root,
        description="successful qualification attempt",
    )
    intent = validate_intent(
        _read_json(
            context.qualification_root / INTENT_NAME,
            description="qualification intent",
            sealed=True,
        ),
        context=context,
    )
    plan = validate_load_plan(
        _read_json(
            context.qualification_root / PLAN_NAME,
            description="qualification load plan",
            sealed=True,
        )
    )
    if plan["plan_id"] != intent["plan_id"]:
        raise ThroughputQualificationError(
            "qualification intent and load plan identities differ"
        )
    evaluation, evidence = _load_and_verify_evidence(
        context, intent=intent
    )
    attempt_marker_path = context.qualification_root / MARKER_NAME
    attempt_marker = _read_json(
        attempt_marker_path,
        description="throughput qualification completion marker",
        sealed=True,
    )
    marker_path = context.qualification_base / MARKER_NAME
    marker = _read_json(
        marker_path,
        description="current-generation throughput qualification marker",
        sealed=True,
    )
    expected_marker = _marker_payload(
        context=context, evaluation=evaluation
    )
    if marker != expected_marker or attempt_marker != marker:
        raise ThroughputQualificationError(
            "throughput qualification completion marker drifted"
        )
    if verify_renderer:
        try:
            rendered = renderer.verify_throughput_qualification(
                context.chain_manifest
            )
        except renderer.ChainError as exc:
            raise ThroughputQualificationError(
                f"renderer rejected throughput qualification: {exc}"
            ) from exc
        if rendered.get("qualification_id") != marker["qualification_id"]:
            raise ThroughputQualificationError(
                "renderer qualification identity differs from sealed evidence"
            )
    return {
        "status": "verified",
        "passed": True,
        "qualification_marker": str(marker_path),
        "attempt_marker": str(attempt_marker_path),
        "attempt_id": context.attempt_pointer["attempt_id"],
        "qualification_id": marker["qualification_id"],
        "evidence_id": evidence["evidence_id"],
        **{
            key: evaluation[key]
            for key in (
                "cells",
                "qids",
                "ceilings",
                "steady_384_seconds",
                "throughput_qids_per_day",
                "every_stratum_progress",
                "integrity_incidents",
                "transport_censor_incidents",
                "observation_count",
            )
        },
    }


def dispatcher_command(
    context: QualificationContext,
    *,
    client_partition: str,
    client_qos: str,
    max_batch: int = MAX_BATCH,
) -> list[str]:
    """Return the one isolated frozen-dispatcher poll command.

    Stage ceilings are enforced by the parent orchestrator before each maximum-24
    microbatch.  The dispatcher retains the protected global 448-minus-64 contract.
    No production run ID or production dispatcher state directory appears here.
    """

    if (
        not isinstance(max_batch, int)
        or isinstance(max_batch, bool)
        or not 1 <= max_batch <= MAX_BATCH
    ):
        raise ThroughputQualificationError(
            f"qualification microbatch must be within 1..{MAX_BATCH}"
        )
    authorize_client_placement(
        context,
        client_partition=client_partition,
        client_qos=client_qos,
    )
    command = [
        str((context.harness_prefix / "bin" / "python").resolve()),
        "-u",
        str((context.release_worktree / "slurm" / "dispatch_sweeps.py").resolve()),
        "dispatch",
        "--run",
        f"{QUALIFICATION_RUN_ID}={context.run_root}",
        "--server-pool",
        f"{QUALIFICATION_RUN_ID}={context.server_pool_root}",
        "--weight",
        f"{QUALIFICATION_RUN_ID}=1",
        "--results-root",
        str(context.results_root),
        "--state-dir",
        str(context.dispatcher_state),
        "--qos-limit",
        str(QOS_LIMIT),
        "--reserve",
        str(QOS_RESERVE),
        "--max-batch",
        str(max_batch),
        "--poll-seconds",
        "120",
        "--cell-partition",
        client_partition,
        "--cell-qos",
        client_qos,
        "--protected-capacity-marker",
        str(context.protected_capacity_contract.path),
        "--protected-capacity-marker-sha256",
        context.protected_capacity_contract.sha256,
        "--protected-capacity-marker-id",
        context.protected_capacity_contract.marker_id,
        "--protected-capacity-release-git-commit",
        context.release_git_commit,
        "--qualification-execution-authority",
        str(context.qualification_root / EXECUTION_AUTHORITY_NAME),
        "--cell-time",
        "12:00:00",
        "--cell-mem",
        "4G",
        "--fanout-slots-per-server",
        "24",
        "--validation-budget",
        "768",
        "--probe-servers",
        "--once",
    ]
    joined = "\0".join(command)
    if any(run_id in joined for run_id in PRODUCTION_RUN_IDS):
        raise ThroughputQualificationError(
            "qualification dispatcher command contains a production run ID"
        )
    if str(context.state_root) in command:
        raise ThroughputQualificationError(
            "qualification dispatcher command aliases production control state"
        )
    return command


def dry_run_report(
    chain_manifest: Path,
    *,
    client_partition: str | None = None,
    client_qos: str | None = None,
    verify_chain: bool = True,
) -> dict[str, Any]:
    """Render the exact non-mutating qualification plan."""

    base = load_qualification_context(
        chain_manifest, verify_chain=verify_chain
    )
    placement = resolve_client_placement(
        base,
        client_partition=client_partition,
        client_qos=client_qos,
    )
    selected_partition = str(placement["partition"])
    selected_qos = str(placement["qos"])
    plan = build_load_plan()
    known = _load_attempt_pointers(base)
    if known and (base.qualification_base / CURRENT_ATTEMPT_NAME).exists():
        context = load_current_attempt_context(
            base, require_success=False
        )
        namespace_status = "existing-current-attempt"
    else:
        placeholder = "VERIFIED_GENERATION_AT_EXECUTE"
        attempt_root = (
            base.qualification_base / ATTEMPT_DIRECTORY / placeholder
        )
        context = replace(
            base,
            qualification_root=attempt_root,
            run_root=(
                base.results_root
                / ATTEMPT_RUN_DIRECTORY
                / placeholder
                / QUALIFICATION_RUN_ID
            ),
            dispatcher_state=(
                attempt_root / DISPATCHER_STATE_DIRECTORY
            ),
        )
        namespace_status = "marker-first-at-execute"
    command = dispatcher_command(
        context,
        client_partition=selected_partition,
        client_qos=selected_qos,
    )
    return {
        "status": "dry_run",
        "submitted": False,
        "run_id": QUALIFICATION_RUN_ID,
        "estimand_excluded": True,
        "primary_analysis_eligible": False,
        "qualification_base": str(base.qualification_base),
        "qualification_root": str(context.qualification_root),
        "run_root": str(context.run_root),
        "dispatcher_state": str(context.dispatcher_state),
        "attempt_namespace": namespace_status,
        "current_attempt_pointer": str(
            base.qualification_base / CURRENT_ATTEMPT_NAME
        ),
        "client_placement": placement,
        "plan_id": plan["plan_id"],
        "cells": CELL_COUNT,
        "qids_per_cell": QIDS_PER_CELL,
        "qids": TOTAL_QIDS,
        "strata": EXPECTED_STRATA,
        "balance": plan["balance"],
        "stages": [
            {
                "ceiling": ceiling,
                "maximum_active_cells": ceiling,
                "maximum_microbatch": MAX_BATCH,
                "dispatcher_argv": command,
            }
            for ceiling in CEILINGS
        ],
        "steady_384_seconds": STEADY_384_SECONDS,
        "minimum_qids_per_day": MIN_QIDS_PER_DAY,
    }


def _row_matches_readiness_generation(
    row: Mapping[str, Any],
    readiness_generation: Mapping[str, Any],
) -> bool:
    return all(
        row.get(field) == readiness_generation[field]
        for field in (
            "release_fleet_contract_sha256",
            "fleet_contract_sha256",
            "capacity_generation",
            "rollout_generation",
        )
    )


def _semantic_scan(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    sequence: int,
    captured_timestamp: float,
    model_contract_path: Path,
) -> dict[str, Any]:
    """Collect one policy-aware semantic snapshot of the qualification run."""

    snapshot = load_manifest(context.run_root, verify_frozen=True)
    if snapshot.cells != generate_qualification_cells():
        raise ThroughputQualificationError(
            "qualification manifest drifted before semantic scan"
        )
    catalog = VerifiedQuestionCatalog(context.run_root, snapshot=snapshot)
    progress = {label: 0 for label in expected_stratum_labels()}
    states: Counter[str] = Counter()
    schemas: Counter[str] = Counter()
    validated = 0
    useful = 0
    integrity_incidents = 0
    transport_incidents = 0
    readiness_generation = intent["readiness_generation"]
    for cell in snapshot.cells:
        questions = catalog.questions_for(cell)
        cell_directory = context.run_root / "cells" / cell.cell_id
        status = get_completion_status(
            cell,
            cell_directory,
            expected_qids=tuple(question.qid for question in questions),
            expected_questions=questions,
            verified_benchmark_contracts=catalog.frozen,
            verified_manifest=snapshot,
            model_contract_path=model_contract_path,
            check_active=True,
        )
        states[status.status.value] += 1
        rows: list[dict[str, Any]] = []
        if status.valid_count:
            rows = list(
                read_canonical_results(
                    cell,
                    cell_directory,
                    expected_qids=tuple(question.qid for question in questions),
                    expected_questions=questions,
                    verified_benchmark_contracts=catalog.frozen,
                    verified_manifest=snapshot,
                ).records
            )
        validated += len(rows)
        for row in rows:
            schemas[str(row.get("schema_version"))] += 1
            accounting = _censor_accounting([row])
            affected = int(accounting["n_transport_affected_questions"])
            runtime_matches = _row_matches_readiness_generation(
                row, readiness_generation
            )
            if affected == 0 and runtime_matches:
                useful += 1
                progress[_stratum_label(_stratum_tuple(cell))] += 1
            if not runtime_matches:
                integrity_incidents += 1
            transport_incidents += int(
                accounting["n_transport_censored_coordinates"]
            )
            integrity_incidents += (
                int(accounting["n_length_censored_questions"])
                + int(accounting["n_protocol_censored_questions"])
                + int(accounting["n_auxiliary_length_censors"])
                + int(accounting["n_auxiliary_protocol_censors"])
            )
        integrity_incidents += (
            int(status.malformed_lines)
            + int(status.invalid_rows)
            + len(status.duplicate_qids)
            + len(status.unexpected_qids)
            + int(
                status.status
                in {CompletionState.CORRUPT, CompletionState.PERMANENT}
            )
        )
    return make_semantic_evidence(
        intent_id=str(intent["intent_id"]),
        sequence=sequence,
        captured_timestamp=captured_timestamp,
        manifest_sha256=snapshot.sha256,
        states=dict(states),
        validated_qids=validated,
        useful_qids=useful,
        strata_progress=progress,
        artifact_schema_counts=dict(schemas),
        integrity_incidents=integrity_incidents,
        transport_censor_incidents=transport_incidents,
    )


def _active_task_count(
    *,
    scheduler_jobs: Sequence[Any],
    ledger: Mapping[str, Any],
    captured_timestamp: float,
) -> tuple[int, list[str], list[str]]:
    jobs = ledger.get("jobs", {})
    if not isinstance(jobs, Mapping):
        raise ThroughputQualificationError(
            "qualification dispatcher ledger has invalid jobs"
        )
    production_ids: set[str] = set()
    for record in jobs.values():
        if not isinstance(record, Mapping):
            raise ThroughputQualificationError(
                "qualification dispatcher job record is malformed"
            )
        for task in record.get("tasks", []):
            run_id = str(task.get("run_id", "")) if isinstance(task, Mapping) else ""
            if run_id != QUALIFICATION_RUN_ID:
                production_ids.add(run_id)
    active_count = 0
    qualification_job_ids = sorted(str(job_id) for job_id in jobs)
    known_job_ids = {str(job_id) for job_id in jobs}
    for job in scheduler_jobs:
        if not job.active:
            continue
        provenance = "\0".join(
            (
                str(job.job_name),
                str(job.comment),
                str(job.command),
            )
        )
        production_ids.update(
            run_id for run_id in PRODUCTION_RUN_IDS if run_id in provenance
        )
        belongs_to_qualification = any(
            job.job_id == base_id or job.job_id.startswith(f"{base_id}_")
            for base_id in known_job_ids
        )
        if (
            dispatch_sweeps.is_cell_job_name(str(job.job_name))
            and not belongs_to_qualification
        ):
            production_ids.add(f"unmapped-active-cell-job:{job.job_id}")
    for base_id, record in jobs.items():
        rows = [
            job
            for job in scheduler_jobs
            if job.job_id == str(base_id)
            or job.job_id.startswith(f"{base_id}_")
        ]
        task_rows = [
            job
            for job in rows
            if "_" in job.job_id and job.active
        ]
        if task_rows:
            active_count += len({job.job_id for job in task_rows})
        elif any(job.active for job in rows):
            active_count += int(record.get("task_count", 0))
        elif record.get("state") in {
            "submitted",
            "active",
            "visibility_grace",
        }:
            submitted_at = record.get("submitted_at")
            if (
                isinstance(submitted_at, (int, float))
                and not isinstance(submitted_at, bool)
                and 0
                <= captured_timestamp - float(submitted_at)
                < 300.0
            ):
                # A numeric sbatch acceptance is durable before Slurm necessarily
                # exposes array tasks.  Count that exact task reservation during the
                # same visibility grace used by the dispatcher, so fast cells cannot
                # disappear between acceptance and the first scheduler sample.
                active_count += int(record.get("task_count", 0))
    return active_count, qualification_job_ids, sorted(production_ids)


def _scheduler_scan(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    sequence: int,
    captured_timestamp: float,
    ceiling: int,
    scheduler_reader: Callable[..., Any] = control.query_scheduler,
) -> dict[str, Any]:
    try:
        execution_authority = (
            dispatch_sweeps.load_qualification_execution_authority(
                context.qualification_root / EXECUTION_AUTHORITY_NAME
            )
        )
        expected_execution_authority = (
            dispatch_sweeps._qualification_execution_authority_binding(  # noqa: SLF001
                execution_authority
            )
        )
    except dispatch_sweeps.DispatcherError as exc:
        raise ThroughputQualificationError(
            f"qualification execution authority failed scheduler join: {exc}"
        ) from exc
    try:
        snapshot = scheduler_reader(
            now=captured_timestamp, tolerate_errors=False
        )
    except (control.ControlError, OSError, subprocess.TimeoutExpired) as exc:
        raise ThroughputQualificationError(
            f"qualification scheduler capture failed: {exc}"
        ) from exc
    ledger_path = context.dispatcher_state / "ledger.json"
    if ledger_path.exists():
        ledger = dispatch_sweeps._load_ledger(ledger_path)
        expected_authority = {
            "path": str(context.protected_capacity_contract.path),
            "sha256": context.protected_capacity_contract.sha256,
            "marker_id": context.protected_capacity_contract.marker_id,
            "release_git_commit": context.release_git_commit,
            "partition": str(intent["client_partition"]),
            "qos": str(intent["client_qos"]),
            "authorized_cell_slots": CEILINGS[-1],
            "reserve_jobs": QOS_RESERVE,
        }
        if ledger.get("protected_capacity_authority") != expected_authority:
            raise ThroughputQualificationError(
                "isolated dispatcher ledger is not bound to the intent's exact "
                "protected partition/QOS authority"
            )
        if (
            ledger.get("qualification_execution_authority")
            != expected_execution_authority
        ):
            raise ThroughputQualificationError(
                "isolated dispatcher ledger is not bound to the intent's exact "
                "qualification execution authority"
            )
        ledger_sha256: str | None = _sha256_file(ledger_path)
    else:
        ledger = dispatch_sweeps._empty_ledger(captured_timestamp)
        ledger_sha256 = None
    active, job_ids, foreign = _active_task_count(
        scheduler_jobs=snapshot.jobs,
        ledger=ledger,
        captured_timestamp=captured_timestamp,
    )
    jobs = [
        {
            "job_id": job.job_id,
            "job_name": job.job_name,
            "state": job.state,
            "comment": job.comment,
            "command": job.command,
            "source": job.source,
            "dependency": job.dependency,
        }
        for job in snapshot.jobs
        if any(
            job.job_id == base or job.job_id.startswith(f"{base}_")
            for base in job_ids
        )
    ]
    return make_scheduler_evidence(
        intent_id=str(intent["intent_id"]),
        sequence=sequence,
        captured_timestamp=captured_timestamp,
        ceiling=ceiling,
        jobs=jobs,
        qualification_job_ids=job_ids,
        active_qualification_cells=active,
        dispatcher_ledger_sha256=ledger_sha256,
        production_control_guard_sha256=str(
            intent["control_guard"]["guard_sha256"]
        ),
        client_partition=str(intent["client_partition"]),
        client_qos=str(intent["client_qos"]),
        protected_capacity_marker_id=str(
            intent["client_placement"]["protected_capacity_marker_id"]
        ),
        protected_capacity_marker_sha256=str(
            intent["client_placement"]["protected_capacity_marker_sha256"]
        ),
        readiness_rollout_generation=int(
            intent["readiness_generation"]["rollout_generation"]
        ),
        trusted_generation_catalog_id=str(
            intent["readiness_generation"]["catalog_id"]
        ),
        qualification_execution_authority_id=(
            execution_authority.authority_id
        ),
        qualification_execution_authority_sha256=(
            execution_authority.sha256
        ),
        squeue_complete=bool(snapshot.squeue_ok),
        sacct_complete=bool(snapshot.sacct_ok),
        errors=list(snapshot.errors),
        qualification_tasks_only=not foreign,
        production_run_ids=foreign,
    )


def _execution_environment(
    context: QualificationContext,
    *,
    control_value: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, str]:
    """Build the exact projected-generation environment for qualification cells.

    The paused production control remains read only.  Its immutable pins and verified
    effective fleet are projected to the catalog-proved next rollout generation, while
    the corresponding runtime attestation and renewable lease live only below the
    isolated qualification root.
    """

    immutable = control_value["immutable"]
    readiness_generation = _validate_readiness_generation(
        intent["readiness_generation"],
        control_value=control_value,
    )
    target_generation = int(readiness_generation["rollout_generation"])
    try:
        runtime_attestation = control.ensure_runtime_integrity_attestation(
            context.qualification_root,
            control_value,
            generation=target_generation,
            force_full=False,
        )
        projected_control = copy.deepcopy(control_value)
        projected_control["rollout_generation"] = target_generation
        projected_control[control.RUNTIME_ATTESTATION_STATE_KEY] = (
            runtime_attestation
        )
        validated_attestation = (
            control.validate_runtime_integrity_attestation(
                projected_control,
                verify_metadata=True,
            )
        )
        if validated_attestation != runtime_attestation:
            raise ThroughputQualificationError(
                "projected runtime attestation changed during validation"
            )
        environment = control.production_environment(projected_control)
    except ThroughputQualificationError:
        raise
    except (control.ControlError, OSError, ValueError) as exc:
        raise ThroughputQualificationError(
            f"cannot derive qualification runtime provenance: {exc}"
        ) from exc
    expected_environment = {
        "ASYS_RELEASE_GIT_COMMIT": context.release_git_commit,
        "ASYS_PROTECTED_CAPACITY_MARKER": str(
            context.protected_capacity_contract.path
        ),
        "ASYS_PROTECTED_CAPACITY_MARKER_SHA256": (
            context.protected_capacity_contract.sha256
        ),
        "ASYS_PROTECTED_CAPACITY_MARKER_ID": (
            context.protected_capacity_contract.marker_id
        ),
        "ASYS_FLEET_CONTRACT_SHA256": str(
            readiness_generation["fleet_contract_sha256"]
        ),
        "ASYS_RELEASE_FLEET_CONTRACT_SHA256": str(
            readiness_generation["release_fleet_contract_sha256"]
        ),
        "ASYS_CAPACITY_GENERATION": str(
            readiness_generation["capacity_generation"]
        ),
        "ASYS_ROLLOUT_GENERATION": str(target_generation),
        "ASYS_RUNTIME_ATTESTATION": str(runtime_attestation["path"]),
        "ASYS_RUNTIME_ATTESTATION_SHA256": str(
            runtime_attestation["sha256"]
        ),
        "ASYS_RUNTIME_INTEGRITY_LEASE": str(
            runtime_attestation["lease_path"]
        ),
    }
    mismatches = {
        key: (expected, environment.get(key))
        for key, expected in expected_environment.items()
        if environment.get(key) != expected
    }
    if mismatches:
        raise ThroughputQualificationError(
            "qualification runtime differs from its sealed placement/fleet "
            f"authority: {mismatches}"
        )
    policy = load_artifact_policy(context.run_root, required=True)
    assert policy is not None
    environment = dict(environment)
    environment["ASYS_ARTIFACT_POLICY_SHA256"] = policy.file_sha256
    missing_task_environment = sorted(
        dispatch_sweeps.PRODUCTION_ENVIRONMENT_KEYS - set(environment)
    )
    if missing_task_environment:
        raise ThroughputQualificationError(
            "projected qualification runtime lacks required schema-5 task "
            f"fields: {missing_task_environment}"
        )
    task_environment = {
        key: str(environment[key])
        for key in sorted(dispatch_sweeps.PRODUCTION_ENVIRONMENT_KEYS)
    }
    try:
        dispatch_sweeps._runtime_environment(  # noqa: SLF001
            {"runtime_environment": task_environment}
        )
    except dispatch_sweeps.DispatcherError as exc:
        raise ThroughputQualificationError(
            f"projected qualification task environment is invalid: {exc}"
        ) from exc
    result = dict(os.environ)
    for name in list(result):
        if name.startswith("ASYS_") or name in {
            "PYTHONHOME",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
        }:
            result.pop(name, None)
    result.update(task_environment)
    result.update(
        {
            "ASYS_RESULTS_ROOT": str(context.results_root),
            "ASYS_RELEASE_WORKTREE": str(context.release_worktree),
            "ASYS_HARNESS_ENV": str(context.harness_prefix),
            "ASYS_HARNESS_ENVIRONMENT_PREFIX": str(context.harness_prefix),
            "ASYS_MODEL_CONTRACT": str(
                Path(immutable["model_contract_path"]).resolve()
            ),
            "HF_HOME": str(context.hf_home),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    return result


def create_or_load_execution_authority(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    control_value: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
) -> dispatch_sweeps.QualificationExecutionAuthority:
    """Publish and revalidate the marker-first qualification cell authority."""

    effective_environment = (
        _execution_environment(
            context,
            control_value=control_value,
            intent=intent,
        )
        if environment is None
        else dict(environment)
    )
    try:
        runtime_environment = {
            key: str(effective_environment[key])
            for key in sorted(
                dispatch_sweeps.PRODUCTION_ENVIRONMENT_KEYS
            )
        }
    except KeyError as exc:
        raise ThroughputQualificationError(
            f"qualification execution environment lacks {exc.args[0]!r}"
        ) from exc
    release_worktree = context.release_worktree.resolve()
    harness_prefix = context.harness_prefix.resolve()
    python_path = (harness_prefix / "bin" / "python").resolve()
    dispatcher_path = (
        release_worktree / "slurm" / "dispatch_sweeps.py"
    ).resolve()
    template_path = (
        release_worktree / "slurm" / "run_dispatch_batch.sbatch.tmpl"
    ).resolve()
    try:
        execution = {
            "release_worktree": str(release_worktree),
            "harness_prefix": str(harness_prefix),
            "hf_home": str(context.hf_home.resolve()),
            "python": str(python_path),
            "python_sha256": dispatch_sweeps._sealed_artifact_sha256(  # noqa: SLF001
                python_path
            ),
            "dispatcher_script": str(dispatcher_path),
            "dispatcher_script_sha256": (
                dispatch_sweeps._sealed_artifact_sha256(  # noqa: SLF001
                    dispatcher_path
                )
            ),
            "batch_template": str(template_path),
            "batch_template_sha256": (
                dispatch_sweeps._sealed_artifact_sha256(  # noqa: SLF001
                    template_path
                )
            ),
        }
    except dispatch_sweeps.DispatcherError as exc:
        raise ThroughputQualificationError(
            f"qualification execution artifact is not sealed: {exc}"
        ) from exc
    payload = _with_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": EXECUTION_AUTHORITY_PROTOCOL,
            "intent_id": str(intent["intent_id"]),
            "chain_id": context.chain_id,
            "run_id": QUALIFICATION_RUN_ID,
            "run_root": str(context.run_root),
            "release_git_commit": context.release_git_commit,
            "protected_capacity": {
                "path": str(context.protected_capacity_contract.path),
                "sha256": context.protected_capacity_contract.sha256,
                "marker_id": (
                    context.protected_capacity_contract.marker_id
                ),
            },
            "readiness_generation": dict(
                intent["readiness_generation"]
            ),
            "execution": execution,
            "runtime_environment": runtime_environment,
        },
        "authority_id",
    )
    path = context.qualification_root / EXECUTION_AUTHORITY_NAME
    _write_once(
        path,
        payload,
        description="qualification execution authority",
    )
    try:
        authority = (
            dispatch_sweeps.load_qualification_execution_authority(path)
        )
    except dispatch_sweeps.DispatcherError as exc:
        raise ThroughputQualificationError(
            f"qualification execution authority is invalid: {exc}"
        ) from exc
    expected_intent_authority = {
        "path": str(path),
        "protocol": EXECUTION_AUTHORITY_PROTOCOL,
    }
    if (
        intent.get("execution_authority") != expected_intent_authority
        or authority.payload != payload
    ):
        raise ThroughputQualificationError(
            "qualification execution authority differs from its sealed intent"
        )
    return authority


def _run_dispatcher_once(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    control_value: Mapping[str, Any],
    max_batch: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    environment = _execution_environment(
        context,
        control_value=control_value,
        intent=intent,
    )
    create_or_load_execution_authority(
        context,
        intent=intent,
        control_value=control_value,
        environment=environment,
    )
    argv = dispatcher_command(
        context,
        client_partition=str(intent["client_partition"]),
        client_qos=str(intent["client_qos"]),
        max_batch=max_batch,
    )
    try:
        completed = runner(
            argv,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
    except OSError as exc:
        raise ThroughputQualificationError(
            f"cannot execute frozen qualification dispatcher: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise ThroughputQualificationError(
            "frozen qualification dispatcher failed "
            f"({completed.returncode}): {completed.stderr.strip()[:1000]}"
        )
    try:
        report = json.loads(completed.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ThroughputQualificationError(
            f"qualification dispatcher returned invalid JSON: {exc}"
        ) from exc
    selected = report.get("selected") if isinstance(report, Mapping) else None
    submission = (
        report.get("submission") if isinstance(report, Mapping) else None
    )
    profile_backlog = (
        report.get("profile_backlog")
        if isinstance(report, Mapping)
        else None
    )
    if (
        not isinstance(report, Mapping)
        or report.get("dry_run") is not False
        or not isinstance(selected, list)
        or any(
            not isinstance(task, Mapping)
            or task.get("run_id") != QUALIFICATION_RUN_ID
            for task in selected
        )
        or len(selected) > max_batch
        or report.get("submission_error") is not None
        or (
            bool(selected)
            and (
                not isinstance(submission, Mapping)
                or not str(submission.get("job_id", "")).isdigit()
                or submission.get("tasks") != len(selected)
            )
        )
        or (not selected and submission is not None)
        or not isinstance(profile_backlog, list)
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "server_pool_root",
                "serving_profile",
                "eligible_cells",
                "backlog_fanout_work",
                "live_replicas",
                "backlog_work_per_replica",
            }
            or not isinstance(row["server_pool_root"], str)
            or not Path(row["server_pool_root"]).is_absolute()
            or not isinstance(row["serving_profile"], str)
            or not row["serving_profile"]
            or any(
                not isinstance(row[field], int)
                or isinstance(row[field], bool)
                or row[field] < 0
                for field in (
                    "eligible_cells",
                    "backlog_fanout_work",
                    "live_replicas",
                )
            )
            or (
                row["live_replicas"] == 0
                and row["backlog_work_per_replica"] is not None
            )
            or (
                row["live_replicas"] > 0
                and (
                    not isinstance(
                        row["backlog_work_per_replica"], (int, float)
                    )
                    or isinstance(
                        row["backlog_work_per_replica"], bool
                    )
                    or not math.isfinite(
                        float(row["backlog_work_per_replica"])
                    )
                    or float(row["backlog_work_per_replica"])
                    != row["backlog_fanout_work"]
                    / row["live_replicas"]
                )
            )
            for row in profile_backlog
        )
    ):
        raise ThroughputQualificationError(
            "qualification dispatcher report escaped its isolated run/microbatch "
            "or lacks unambiguous numeric sbatch acceptance"
        )
    return dict(report)


def _scaling_requirement(
    context: QualificationContext,
) -> dict[str, Any]:
    """Name the highest observed isolated backlog pressure without mutating it."""

    ledger_path = context.dispatcher_state / "ledger.json"
    try:
        ledger = dispatch_sweeps._load_ledger(ledger_path)  # noqa: SLF001
    except (dispatch_sweeps.DispatcherError, OSError) as exc:
        raise ThroughputQualificationError(
            f"cannot derive qualification scaling requirement: {exc}"
        ) from exc
    history = ledger.get("qualification_profile_pressure")
    if not isinstance(history, Mapping) or not history:
        raise ThroughputQualificationError(
            "qualification failed before any per-profile backlog pressure was "
            "sealed in its isolated ledger"
        )
    rows = [row for row in history.values() if isinstance(row, Mapping)]
    if len(rows) != len(history):
        raise ThroughputQualificationError(
            "qualification profile-pressure history is malformed"
        )

    def higher(
        candidate: Mapping[str, Any], incumbent: Mapping[str, Any]
    ) -> bool:
        candidate_replicas = int(candidate["live_replicas"])
        incumbent_replicas = int(incumbent["live_replicas"])
        if (candidate_replicas == 0) != (incumbent_replicas == 0):
            return candidate_replicas == 0
        left = int(candidate["backlog_fanout_work"]) * max(
            1, incumbent_replicas
        )
        right = int(incumbent["backlog_fanout_work"]) * max(
            1, candidate_replicas
        )
        if left != right:
            return left > right
        return str(candidate["serving_profile"]) < str(
            incumbent["serving_profile"]
        )

    bottleneck = rows[0]
    for row in rows[1:]:
        if higher(row, bottleneck):
            bottleneck = row
    profile = str(bottleneck["serving_profile"])
    tp_size = 2 if profile == "32B-long" else 1
    return {
        "serving_profile": profile,
        "server_pool_root": str(bottleneck["server_pool_root"]),
        "backlog_fanout_work": int(
            bottleneck["backlog_fanout_work"]
        ),
        "live_replicas": int(bottleneck["live_replicas"]),
        "backlog_work_per_replica": (
            None
            if int(bottleneck["live_replicas"]) == 0
            else int(bottleneck["backlog_fanout_work"])
            / int(bottleneck["live_replicas"])
        ),
        "additional_replicas": 1,
        "tensor_parallel_size": tp_size,
        "additional_gpus": tp_size,
        "requirement": (
            "add one TP=2 replica pair (2 GPUs)"
            if tp_size == 2
            else "add one replica (1 GPU)"
        ),
        "capacity_mutated": False,
    }


def _publish_terminal_failure(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    scaling = _scaling_requirement(context)
    payload = _with_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": FAILURE_PROTOCOL,
            "passed": False,
            "intent_id": str(intent["intent_id"]),
            "attempt": _completion_attempt_binding(context),
            "readiness_generation": dict(
                intent["readiness_generation"]
            ),
            "reason": reason,
            "additive_scaling_requirement": scaling,
            "scheduler_capacity_mutated": False,
            "rerun_requirement": (
                "publish a fresh serving/capacity and trusted-catalog rollout "
                "generation, then create a fresh qualification namespace and intent; "
                "this failed intent and its observations cannot be reused"
            ),
        },
        "failure_id",
    )
    _write_once(
        context.qualification_root / FAILURE_NAME,
        payload,
        description="terminal qualification failure",
    )
    _load_terminal_failure(context)
    _seal_tree_read_only(
        context.run_root,
        description="failed qualification run",
    )
    _seal_tree_read_only(
        context.qualification_root,
        description="failed qualification attempt",
    )
    return payload


def _load_terminal_failure(
    context: QualificationContext,
) -> dict[str, Any]:
    path = context.qualification_root / FAILURE_NAME
    failure = _read_json(
        path,
        description="terminal qualification failure",
        sealed=True,
    )
    if set(failure) != _FAILURE_FIELDS:
        raise ThroughputQualificationError(
            "terminal qualification failure marker fields drifted"
        )
    _verify_identity(
        failure,
        "failure_id",
        description="terminal qualification failure",
    )
    scaling = failure.get("additive_scaling_requirement")
    if not isinstance(scaling, Mapping):
        raise ThroughputQualificationError(
            "terminal qualification failure marker is malformed"
        )
    profile = scaling.get("serving_profile")
    live_replicas = scaling.get("live_replicas")
    backlog = scaling.get("backlog_fanout_work")
    pressure = scaling.get("backlog_work_per_replica")
    tensor_parallel_size = 2 if profile == "32B-long" else 1
    expected_requirement = (
        "add one TP=2 replica pair (2 GPUs)"
        if tensor_parallel_size == 2
        else "add one replica (1 GPU)"
    )
    expected_pressure = (
        None
        if live_replicas == 0
        else (
            int(backlog) / int(live_replicas)
            if (
                isinstance(backlog, int)
                and not isinstance(backlog, bool)
                and isinstance(live_replicas, int)
                and not isinstance(live_replicas, bool)
                and live_replicas > 0
            )
            else object()
        )
    )
    if (
        set(scaling) != _SCALING_REQUIREMENT_FIELDS
        or failure.get("schema_version") != SCHEMA_VERSION
        or failure.get("protocol") != FAILURE_PROTOCOL
        or failure.get("passed") is not False
        or _SHA256_RE.fullmatch(str(failure.get("intent_id", ""))) is None
        or failure.get("attempt") != _completion_attempt_binding(context)
        or (
            context.attempt_pointer is None
            or failure.get("readiness_generation")
            != context.attempt_pointer["readiness_generation"]
        )
        or not isinstance(failure.get("reason"), str)
        or not str(failure["reason"]).strip()
        or not isinstance(failure.get("rerun_requirement"), str)
        or "fresh qualification namespace and intent"
        not in str(failure["rerun_requirement"])
        or failure.get("scheduler_capacity_mutated") is not False
        or profile not in SERVING_PROFILES
        or scaling.get("server_pool_root") != str(context.server_pool_root)
        or not isinstance(backlog, int)
        or isinstance(backlog, bool)
        or backlog < 0
        or not isinstance(live_replicas, int)
        or isinstance(live_replicas, bool)
        or live_replicas < 0
        or pressure != expected_pressure
        or scaling.get("additional_replicas") != 1
        or scaling.get("tensor_parallel_size") != tensor_parallel_size
        or scaling.get("additional_gpus") != tensor_parallel_size
        or scaling.get("requirement") != expected_requirement
        or scaling.get("capacity_mutated") is not False
    ):
        raise ThroughputQualificationError(
            "terminal qualification failure marker is malformed"
        )
    return failure


def _terminal_failure_message(context: QualificationContext) -> str | None:
    path = context.qualification_root / FAILURE_NAME
    if not path.exists() and not path.is_symlink():
        return None
    failure = _load_terminal_failure(context)
    scaling = failure["additive_scaling_requirement"]
    return (
        f"{failure.get('reason')}; bottleneck profile "
        f"{scaling.get('serving_profile')} had the largest observed "
        "backlog-work-per-replica; additive requirement: "
        f"{scaling.get('requirement')}. No scheduler capacity was mutated. "
        f"{failure.get('rerun_requirement')}"
    )


def _next_ceiling(observations: Sequence[Mapping[str, Any]]) -> int:
    peaks = {ceiling: 0 for ceiling in CEILINGS}
    for observation in observations:
        scheduler = observation["scheduler"]
        peaks[int(scheduler["ceiling"])] = max(
            peaks[int(scheduler["ceiling"])],
            int(scheduler["active_qualification_cells"]),
        )
    for ceiling in CEILINGS:
        if peaks[ceiling] != ceiling:
            return ceiling
    return 384


def _stage_dispatch_batch(*, ceiling: int, active: int, useful_qids: int) -> int:
    """Return the exact next microbatch without crossing the isolated stage."""

    if (
        ceiling not in CEILINGS
        or not isinstance(active, int)
        or isinstance(active, bool)
        or not 0 <= active <= ceiling
        or not isinstance(useful_qids, int)
        or isinstance(useful_qids, bool)
        or not 0 <= useful_qids <= TOTAL_QIDS
    ):
        raise ThroughputQualificationError(
            "qualification stage admission inputs are invalid"
        )
    if useful_qids == TOTAL_QIDS:
        return 0
    return min(MAX_BATCH, ceiling - active)


@contextmanager
def _qualification_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ThroughputQualificationError(
                "another throughput qualification process holds the execution lock"
            ) from exc
        yield
    finally:
        os.close(descriptor)


def execute_qualification(
    chain_manifest: Path,
    *,
    client_partition: str | None = None,
    client_qos: str | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    verify_chain: bool = True,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    scheduler_reader: Callable[..., Any] = control.query_scheduler,
    semantic_reader: Callable[..., Mapping[str, Any]] | None = None,
    clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
    benchmark_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute the isolated restartable qualification transaction."""

    if (
        not math.isfinite(float(poll_seconds))
        or not 0 < float(poll_seconds) <= MAX_OBSERVATION_GAP_SECONDS
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) < STEADY_384_SECONDS
    ):
        raise ThroughputQualificationError(
            "poll/timeout bounds cannot prove the two-hour clean window"
        )
    base_context = load_qualification_context(
        chain_manifest, verify_chain=verify_chain
    )
    selected_placement = resolve_client_placement(
        base_context,
        client_partition=client_partition,
        client_qos=client_qos,
    )
    selected_partition = str(selected_placement["partition"])
    selected_qos = str(selected_placement["qos"])
    marker_path = base_context.qualification_base / MARKER_NAME
    if marker_path.exists() or marker_path.is_symlink():
        return verify_completed_qualification(
            chain_manifest,
            verify_chain=verify_chain,
            verify_renderer=verify_chain,
        )
    with _qualification_lock(base_context.qualification_base):
        # Recheck after lock acquisition: another exact runner may have completed.
        if marker_path.exists() or marker_path.is_symlink():
            return verify_completed_qualification(
                chain_manifest,
                verify_chain=verify_chain,
                verify_renderer=verify_chain,
            )
        control_value, guard = load_paused_control(base_context)
        readiness_generation = load_readiness_generation(
            base_context, control_value
        )
        context = create_or_load_attempt_context(
            base_context,
            control_value=control_value,
            readiness_generation=readiness_generation,
            now=clock(),
        )
        attempt_marker_path = context.qualification_root / MARKER_NAME
        if attempt_marker_path.exists() or attempt_marker_path.is_symlink():
            # Recover the narrow crash window after the generation-scoped
            # completion marker but before the fixed-root commit point.  This
            # path performs no dispatch or scheduler read: it only recomputes
            # sealed evidence, finishes recursive sealing, and publishes the
            # identical fixed-root marker last.
            recovered_intent = validate_intent(
                _read_json(
                    context.qualification_root / INTENT_NAME,
                    description="qualification intent",
                    sealed=True,
                ),
                context=context,
            )
            marker = publish_completion(
                context,
                intent=recovered_intent,
            )
            verified = verify_completed_qualification(
                chain_manifest,
                verify_chain=verify_chain,
                verify_renderer=verify_chain,
            )
            return {
                "status": "complete",
                **verified,
                "marker": marker,
            }
        intent = create_or_load_intent(
            context,
            client_partition=selected_partition,
            client_qos=selected_qos,
            readiness_generation=readiness_generation,
            control_guard=guard,
            now=clock(),
        )
        if intent["control_guard"] != guard:
            raise ThroughputQualificationError(
                "production admission/ramp state changed after qualification intent"
            )
        initialize_qualification_run(
            context,
            intent=intent,
            control_value=control_value,
            benchmark_loader=benchmark_loader,
        )
        create_or_load_execution_authority(
            context,
            intent=intent,
            control_value=control_value,
        )
        observations = load_observations(
            context.qualification_root,
            intent=intent,
            recover_transactions=True,
        )
        # The marker-first timestamp is the restart-stable outer timeout boundary.
        # A process restart must not buy another ten-hour execution window.
        started = float(intent["created_timestamp"])
        model_contract_path = Path(
            str(control_value["immutable"]["model_contract_path"])
        ).resolve()

        def capture(ceiling: int) -> None:
            nonlocal observations
            sequence = len(observations)
            captured = clock()
            scheduler = _scheduler_scan(
                context,
                intent=intent,
                sequence=sequence,
                captured_timestamp=captured,
                ceiling=ceiling,
                scheduler_reader=scheduler_reader,
            )
            semantic = (
                dict(
                    semantic_reader(
                        context=context,
                        intent=intent,
                        sequence=sequence,
                        captured_timestamp=captured,
                    )
                )
                if semantic_reader is not None
                else _semantic_scan(
                    context,
                    intent=intent,
                    sequence=sequence,
                    captured_timestamp=captured,
                    model_contract_path=model_contract_path,
                )
            )
            record_observation(
                context.qualification_root,
                intent=intent,
                scheduler=scheduler,
                semantic=semantic,
            )
            observations = load_observations(
                context.qualification_root,
                intent=intent,
                recover_transactions=True,
            )

        if not observations:
            # This sealed zero baseline is an executable precondition for the first
            # submission.  A runner cannot reach the dispatcher before it exists.
            capture(24)
        else:
            # A process may have died after the dispatcher crossed sbatch but before
            # the next observation.  Reconcile scheduler+ledger truth on every restart
            # before computing stage room; never admit from a stale receipt.
            capture(_next_ceiling(observations))
        while True:
            try:
                marker = publish_completion(context, intent=intent)
            except ThroughputQualificationError as exc:
                incomplete_markers = (
                    "does not exercise every ceiling",
                    "did not exercise every ceiling",
                    "did not reconcile to every exact ceiling",
                    "is clean for only",
                    "is not semantically complete",
                    "final qualification observation is not quiescent",
                )
                if not any(marker in str(exc) for marker in incomplete_markers):
                    if (
                        "qualification throughput" in str(exc)
                        and "is below" in str(exc)
                    ):
                        failure = _publish_terminal_failure(
                            context,
                            intent=intent,
                            reason=str(exc),
                        )
                        scaling = failure[
                            "additive_scaling_requirement"
                        ]
                        raise QualificationCapacityTransitionRequired(
                            f"{exc}; bottleneck profile "
                            f"{scaling['serving_profile']} had the largest "
                            "observed backlog-work-per-replica; additive "
                            f"requirement: {scaling['requirement']}. No "
                            "scheduler capacity was mutated; a repeated "
                            "qualification requires a fresh generation, "
                            "namespace, and intent."
                        ) from exc
                    raise
            else:
                verified = verify_completed_qualification(
                    chain_manifest,
                    verify_chain=verify_chain,
                    verify_renderer=verify_chain,
                )
                return {"status": "complete", **verified, "marker": marker}

            if clock() - started > timeout_seconds:
                reason = (
                    "throughput qualification exceeded its fixed execution "
                    "timeout"
                )
                failure = _publish_terminal_failure(
                    context,
                    intent=intent,
                    reason=reason,
                )
                scaling = failure["additive_scaling_requirement"]
                raise ThroughputQualificationError(
                    f"{reason}; bottleneck profile "
                    f"{scaling['serving_profile']} had the largest observed "
                    "backlog-work-per-replica; additive requirement: "
                    f"{scaling['requirement']}. No scheduler capacity was "
                    "mutated; a repeated qualification requires a fresh "
                    "generation, namespace, and intent."
                )
            current_control, current_guard = load_paused_control(context)
            if current_guard != intent["control_guard"]:
                raise ThroughputQualificationError(
                    "production admission/ramp state changed during qualification"
                )
            current_generation = load_readiness_generation(
                context, current_control
            )
            if current_generation != intent["readiness_generation"]:
                raise ThroughputQualificationError(
                    "trusted serving/readiness generation changed during "
                    "qualification"
                )
            target = _next_ceiling(observations)
            latest = observations[-1]
            active = int(
                latest["scheduler"]["active_qualification_cells"]
            )
            useful = int(latest["semantic"]["useful_qids"])
            # One existing dispatcher poll can submit at most 24 tasks.  Clamp that
            # existing interface to the exact remaining stage room before crossing
            # its submission boundary, without modifying production control.
            next_batch = _stage_dispatch_batch(
                ceiling=target,
                active=active,
                useful_qids=useful,
            )
            if next_batch > 0:
                dispatch_report = _run_dispatcher_once(
                    context,
                    intent=intent,
                    control_value=current_control,
                    max_batch=next_batch,
                    runner=runner,
                )
                # Join the accepted numeric array through the isolated ledger
                # immediately.  The ledger's bounded scheduler-visibility reservation
                # proves the exact submitted task count even if sub-second cells finish
                # before squeue first exposes them.
                capture(target)
                target = _next_ceiling(observations)
                latest_after_dispatch = observations[-1]
                if (
                    dispatch_report.get("selected")
                    and int(
                        latest_after_dispatch["semantic"]["useful_qids"]
                    )
                    < TOTAL_QIDS
                    and int(
                        latest_after_dispatch["scheduler"][
                            "active_qualification_cells"
                        ]
                    )
                    < target
                ):
                    # Fill the remainder of a ramp with back-to-back <=24 arrays.
                    # Sleeping here would let fast cells vanish before ceilings 96,
                    # 192, or 384 could ever be durably reconciled.
                    continue
            sleeper(float(poll_seconds))
            capture(target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "execute"):
        command = subparsers.add_parser(name)
        command.add_argument("--chain-manifest", required=True, type=Path)
        command.add_argument(
            "--client-partition",
            help=(
                "explicit sealed-marker client partition; omit with --client-qos "
                "to resolve the unique full-capacity row"
            ),
        )
        command.add_argument(
            "--client-qos",
            help=(
                "explicit sealed-marker client QOS; omit with --client-partition "
                "to resolve the unique full-capacity row"
            ),
        )
        if name == "execute":
            command.add_argument(
                "--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS
            )
            command.add_argument(
                "--timeout-seconds",
                type=float,
                default=DEFAULT_TIMEOUT_SECONDS,
            )
    verify = subparsers.add_parser(
        "verify-only",
        help="verify sealed completion evidence without contacting Slurm",
    )
    verify.add_argument("--chain-manifest", required=True, type=Path)
    transition = subparsers.add_parser(
        "publish-capacity-transition",
        help=(
            "verify and seal the exact paused additive generation required "
            "before same-release stage-18 suffix repair"
        ),
    )
    transition.add_argument("--chain-manifest", required=True, type=Path)
    transition.add_argument("--submission-receipt", required=True, type=Path)
    transition.add_argument("--failed-job-id", required=True)
    transition.add_argument("--failed-comment", required=True)
    transition.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "dry-run":
            report = dry_run_report(
                args.chain_manifest,
                client_partition=args.client_partition,
                client_qos=args.client_qos,
            )
        elif args.command == "execute":
            report = execute_qualification(
                args.chain_manifest,
                client_partition=args.client_partition,
                client_qos=args.client_qos,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        elif args.command == "verify-only":
            report = verify_completed_qualification(args.chain_manifest)
        else:
            report = publish_capacity_transition_authority(
                args.chain_manifest,
                submission_receipt=args.submission_receipt,
                failed_job_id=args.failed_job_id,
                failed_comment=args.failed_comment,
                apply=args.apply,
            )
    except QualificationCapacityTransitionRequired as exc:
        print(
            f"throughput qualification requires exact additive capacity "
            f"transition: {exc}",
            file=sys.stderr,
        )
        return 76
    except ThroughputQualificationError as exc:
        print(
            f"throughput qualification failed closed: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
