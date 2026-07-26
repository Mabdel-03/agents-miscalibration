#!/usr/bin/env python3
"""Fair, global SLURM admission control for all active sweep manifests.

The legacy sweep drivers each reasoned about only one run.  Two drivers could therefore
compete for the same per-user QOS and starve the run whose next array happened to be
larger.  This coordinator owns one global admission decision, submits only small arrays,
and records every decision in an atomic ledger.

Typical use (after stopping the legacy driver chains)::

    python -u slurm/dispatch_sweeps.py dispatch \
      --run full_sweep_v1 \
      --run full_sweep_agent_counts_v1 \
      --server-pool full_sweep_agent_count_7_v1=full_sweep_agent_counts_v1

``dispatch --dry-run`` performs all validation and planning but neither writes state nor
calls ``sbatch``.  ``status`` is also read-only.  This module intentionally never cancels
jobs; controlled migration away from legacy arrays remains an explicit operator action.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
import stat
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

# Permit the documented ``python slurm/dispatch_sweeps.py`` invocation from a fresh
# checkout as well as the normal editable/conda installation used inside SLURM jobs.
REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.config import (
    DEFAULT_DISPATCHER_STATE_DIRNAME,
    DEFAULT_RESULTS_ROOT,
    ExperimentCell,
)
from agents_scaling.benchmarks.contracts import (
    BenchmarkContractError,
    load_frozen_benchmark_contracts,
)
from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog
from agents_scaling.experiment import io
from agents_scaling.experiment.completion import (
    CompletionState,
    get_completion_status,
    is_cell_active,
)
from agents_scaling.experiment.manifest import ManifestSnapshot, load_manifest
from agents_scaling.serving import healthcheck
from agents_scaling.serving import protected_capacity, scheduler_safety
from agents_scaling.serving.registry import (
    ServerEntry,
    active_slurm_allocations,
    entry_has_current_provenance,
    endpoint_history_for_entry,
    endpoint_instance_id,
    server_pool_id,
    server_pool_generation,
)
from agents_scaling.serving.fleet_contract import (
    FleetContractError,
    FrozenFleetContract,
    load_fleet_contract,
)
from agents_scaling.serving.launch_server import _port_for
from agents_scaling.serving.model_contracts import ModelContractError, load_model_contracts
from agents_scaling.serving.profiles import serving_profile_for_cell
from agents_scaling import runtime_integrity


ARRAY_TEMPLATE = REPO / "slurm" / "run_dispatch_batch.sbatch.tmpl"
LEDGER_SCHEMA_VERSION = 1
QOS_LIMIT_DEFAULT = 448
QOS_RESERVE_DEFAULT = 64
MAX_BATCH_DEFAULT = 24
CELL_CPUS_DEFAULT = 1
# The first controlled 2G production pilot reached 1,561,824 KiB MaxRSS on a
# 14B client (2026-07-18), crossing the rollout's 1.5 GB promotion threshold.
# Keep 2G available for explicit low-memory pilots, but use 4G for production.
CELL_MEM_DEFAULT = "4G"
CELL_TIME_DEFAULT = "12:00:00"
CELL_JOB_PREFIXES = ("asys-cells", "asys-dispatch-")
AUTHORITATIVE_SCHEMA5_RUN_IDS = frozenset(
    {
        "full_sweep_schema5_v1",
        "full_sweep_agent_counts_schema5_v1",
        "full_sweep_agent_count_7_schema5_v1",
    }
)
QUALIFICATION_EXECUTION_AUTHORITY_PROTOCOL = (
    "schema5-v1.2-r2-throughput-qualification-execution-authority-v1"
)
ELIGIBLE_STATES = {
    CompletionState.MISSING.value,
    CompletionState.PARTIAL.value,
    CompletionState.CORRUPT.value,
}
PRODUCTION_ENVIRONMENT_KEYS = frozenset(
    {
        "ASYS_RELEASE_ID",
        "ASYS_RELEASE_GIT_COMMIT",
        "ASYS_PROTECTED_CAPACITY_MARKER",
        "ASYS_PROTECTED_CAPACITY_MARKER_SHA256",
        "ASYS_PROTECTED_CAPACITY_MARKER_ID",
        "ASYS_MODEL_CONTRACT_SHA256",
        "ASYS_FLEET_CONTRACT_SHA256",
        "ASYS_FLEET_CONTRACT_PATH",
        "ASYS_RELEASE_FLEET_CONTRACT_SHA256",
        "ASYS_CAPACITY_GENERATION",
        "ASYS_HARNESS_ENVIRONMENT_SHA256",
        "ASYS_SERVING_ENVIRONMENT_SHA256",
        "ASYS_ROLLOUT_GENERATION",
        "ASYS_IMMUTABLE_PINS_SHA256",
        "ASYS_RUNTIME_ATTESTATION",
        "ASYS_RUNTIME_ATTESTATION_SHA256",
        "ASYS_RUNTIME_INTEGRITY_LEASE",
        "ASYS_ARTIFACT_POLICY_SHA256",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    }
)

GroupKey = tuple[str, str, str]  # run id, canonical server-pool root, serving profile
ProfileKey = tuple[str, str]  # canonical server-pool root, serving profile


class DispatcherError(RuntimeError):
    """An admission invariant was violated."""


class SingletonAlreadyRunning(DispatcherError):
    """Another live dispatcher owns the state-directory lock."""


@dataclass(frozen=True)
class Candidate:
    """One schedulable manifest cell; all fields are deterministic and test friendly."""

    run_id: str
    run_root: str
    source_index: int
    cell: ExperimentCell
    manifest_sha256: str
    server_pool_arg: str | None
    server_pool_root: str
    serving_profile: str
    fanout_cost: int
    benchmark_contracts_sha256: str | None = None
    runtime_environment: tuple[tuple[str, str], ...] = ()

    @property
    def cell_id(self) -> str:
        return self.cell.cell_id

    @property
    def group_key(self) -> GroupKey:
        return (self.run_id, self.server_pool_root, self.serving_profile)

    @property
    def profile_key(self) -> ProfileKey:
        return (self.server_pool_root, self.serving_profile)


@dataclass(frozen=True)
class AdmissionResult:
    selected: tuple[Candidate, ...]
    deficits: dict[str, float]
    cursor: int
    remaining_profile_headroom: dict[str, int]


@dataclass(frozen=True)
class ValidationAllocation:
    """One poll's work-conserving, restart-safe semantic-validation allocation."""

    allocations: dict[str, int]
    next_run_id: str | None
    run_order: tuple[str, ...]


@dataclass(frozen=True)
class QueueRow:
    array_job_id: str
    array_task_id: int | None
    job_id: str
    job_name: str
    state: str
    command: str = ""
    comment: str = ""


@dataclass(frozen=True)
class StableAdmissionOccupancy:
    """One target-usage capture bracketed by stable global scheduler truth."""

    scheduler_snapshot: Any
    rows: tuple[QueueRow, ...]
    usage: dict[str, Any]
    usage_summary: dict[str, Any]
    live_job_ids: tuple[str, ...]
    attempts: int


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    run_root: Path
    manifest: ManifestSnapshot
    server_pool_arg: str | None
    server_pool_root: Path
    weight: float = 1.0
    question_catalog: VerifiedQuestionCatalog | None = None
    runtime_environment: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CapacitySnapshot:
    live_servers: dict[ProfileKey, int]
    server_pool_generation: dict[ProfileKey, str]


@dataclass(frozen=True)
class QualificationExecutionAuthority:
    """One sealed, self-hashed qualification-only cell execution boundary."""

    path: Path
    sha256: str
    authority_id: str
    payload: dict[str, Any]
    runtime_environment: dict[str, str]
    execution: dict[str, str]


@dataclass
class _CellScan:
    """Cheap scan result, finalized after fair semantic-validation allocation."""

    spec: RunSpec
    candidate: Candidate
    cell_key: str
    cell_dir: Path | None
    record: dict[str, Any]
    fingerprint: list[list[int | str]] | None
    validation_context: dict[str, Any] | None
    state: str
    eligible: bool
    next_eligible: float | None
    needs_validation: bool = False


def _key(parts: Sequence[str]) -> str:
    """Stable JSON key for tuple-valued scheduler state."""
    return json.dumps(list(parts), separators=(",", ":"), ensure_ascii=True)


def _unkey(value: str) -> tuple[str, ...]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"invalid encoded scheduler key: {value!r}")
    return tuple(parsed)


def fanout_cost(cell: ExperimentCell) -> int:
    """Approximate serving work used by fair admission.

    A single-agent cell consumes one unit.  Every multi-agent topology launches one
    inference stream per configured agent, so its admission cost is ``n_agents``.  The
    runner's finer request scheduling remains responsible for round-level concurrency.
    """
    return 1 if not cell.topology.is_multi_agent else max(1, int(cell.n_agents))


def available_cell_slots(
    *,
    total_jobs: int,
    active_cell_jobs: int,
    qos_limit: int = QOS_LIMIT_DEFAULT,
    reserve: int = QOS_RESERVE_DEFAULT,
    max_batch: int = MAX_BATCH_DEFAULT,
) -> int:
    """Pure global QOS gate.

    The reserve is held even when today's non-cell overhead is smaller, while the
    absolute gate also protects against a temporary increase in unrelated/serve jobs.
    """
    if min(total_jobs, active_cell_jobs, qos_limit, reserve, max_batch) < 0:
        raise ValueError("job counts and limits must be non-negative")
    if reserve > qos_limit:
        raise ValueError("reserve cannot exceed qos_limit")
    cell_ceiling = qos_limit - reserve
    return max(
        0,
        min(max_batch, cell_ceiling - active_cell_jobs, qos_limit - total_jobs),
    )


def plan_validation_allocation(
    run_ids: Sequence[str],
    *,
    demand: Mapping[str, int],
    budget: int,
    next_run_id: str | None,
) -> ValidationAllocation:
    """Allocate bounded validation work one token per backlogged run at a time.

    Run IDs are canonicalized so CLI ordering cannot change the allocation after a
    coordinator restart.  ``next_run_id`` identifies the next position in that stable
    ring; storing it in the atomic dispatcher ledger prevents a small remainder (for
    example, 64 tokens across three runs) from always favoring the same run.  Runs with
    no validation demand are skipped, making the allocation work-conserving.
    """
    if budget < 0:
        raise ValueError("validation budget must be non-negative")
    canonical = sorted(set(run_ids))
    if len(canonical) != len(run_ids):
        raise ValueError("validation run IDs must be unique")
    unknown = set(demand) - set(canonical)
    if unknown:
        raise ValueError(f"validation demand references unknown runs: {sorted(unknown)}")
    remaining = {run_id: int(demand.get(run_id, 0)) for run_id in canonical}
    if any(value < 0 for value in remaining.values()):
        raise ValueError("validation demand must be non-negative")
    allocations = {run_id: 0 for run_id in canonical}
    if not canonical:
        return ValidationAllocation(allocations, None, ())

    if next_run_id in canonical:
        cursor = canonical.index(str(next_run_id))
    elif next_run_id is None:
        cursor = 0
    else:
        # A run may have been deliberately added or retired between restarts.  Resume
        # at the lexical successor instead of silently resetting priority to run zero.
        cursor = next(
            (index for index, run_id in enumerate(canonical) if run_id >= next_run_id),
            0,
        )

    run_order: list[str] = []
    for _ in range(budget):
        selected_run: str | None = None
        for _ in canonical:
            run_id = canonical[cursor]
            cursor = (cursor + 1) % len(canonical)
            if allocations[run_id] < remaining[run_id]:
                selected_run = run_id
                break
        if selected_run is None:
            break
        allocations[selected_run] += 1
        run_order.append(selected_run)

    return ValidationAllocation(allocations, canonical[cursor], tuple(run_order))


def _candidate_sort_key(candidate: Candidate) -> tuple[int, str]:
    return candidate.source_index, candidate.cell_id


def _axis_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _scientific_stratum(candidate: Candidate) -> tuple[Any, ...]:
    cell = candidate.cell
    effective_agents = 1 if not cell.topology.is_multi_agent else int(cell.n_agents)
    return (
        _axis_value(cell.model_size),
        _axis_value(cell.reasoning_level),
        _axis_value(cell.topology),
        effective_agents,
        _axis_value(cell.benchmark),
        int(cell.seed),
        _axis_value(cell.context_share_level),
        int(cell.prompt_complexity_level),
    )


def _stratified_candidate_order(
    candidates: Sequence[Candidate], *, interleave_prefix: int = 256
) -> list[Candidate]:
    """Greedily balance every scientific axis before the source-order tail.

    A composite lexicographic stratum sort still drains early reasoning/benchmark
    values.  Instead, each next bucket minimizes its accumulated marginal exposure
    across model, reasoning, topology, agent count, benchmark, seed, context, and prompt
    level.  Only a bounded prefix needs this richer ordering because a poll admits at
    most 24 tasks; bounding it keeps planning linear enough for the 22,680-cell grid.
    """

    if interleave_prefix < 0:
        raise ValueError("interleave_prefix must be non-negative")

    buckets: dict[tuple[Any, ...], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[_scientific_stratum(candidate)].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=_candidate_sort_key)
    ordered: list[Candidate] = []
    marginal_counts: dict[tuple[int, Any], int] = defaultdict(int)
    limit = min(len(candidates), interleave_prefix)
    for _ in range(limit):
        active_strata = [stratum for stratum, bucket in buckets.items() if bucket]
        selected_stratum = min(
            active_strata,
            key=lambda stratum: (
                sum(marginal_counts[(axis, value)] for axis, value in enumerate(stratum)),
                max(
                    (marginal_counts[(axis, value)] for axis, value in enumerate(stratum)),
                    default=0,
                ),
                tuple(
                    marginal_counts[(axis, value)]
                    for axis, value in enumerate(stratum)
                ),
                buckets[stratum][0].source_index,
                buckets[stratum][0].cell_id,
            ),
        )
        selected = buckets[selected_stratum].pop(0)
        ordered.append(selected)
        for axis, value in enumerate(selected_stratum):
            marginal_counts[(axis, value)] += 1
    tail = sorted(
        (candidate for bucket in buckets.values() for candidate in bucket),
        key=_candidate_sort_key,
    )
    return ordered + tail


def plan_admission(
    candidates: Sequence[Candidate],
    *,
    deficits: Mapping[str, float] | None,
    cursor: int,
    max_tasks: int,
    live_servers: Mapping[ProfileKey, int],
    profile_headroom: Mapping[ProfileKey, int],
    run_weights: Mapping[str, float] | None = None,
) -> AdmissionResult:
    """Weighted deficit round-robin across ``(run, pool, serving-profile)`` groups.

    This function is deliberately free of filesystem and SLURM access.  A group's
    quantum is its run weight times the number of live replicas.  A cell spends one
    deficit unit per agent, so high-fanout cells cannot monopolize a shared model pool.
    ``profile_headroom`` is a hard shared budget and already accounts for active cells.

    Returned deficits and cursor can be serialized in the ledger and supplied to the
    next call, preserving fairness across coordinator restarts.
    """
    if max_tasks < 0:
        raise ValueError("max_tasks must be non-negative")
    weights = dict(run_weights or {})
    queues: dict[GroupKey, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.fanout_cost < 1:
            raise ValueError("candidate fanout_cost must be >= 1")
        queues[candidate.group_key].append(candidate)
    for group, queue in tuple(queues.items()):
        queues[group] = _stratified_candidate_order(queue)

    groups = sorted(queues)
    new_deficits = {str(k): float(v) for k, v in (deficits or {}).items()}
    remaining = {
        _key(profile): max(0, int(slots)) for profile, slots in profile_headroom.items()
    }
    if not groups or max_tasks == 0:
        return AdmissionResult((), new_deficits, 0 if not groups else cursor % len(groups), remaining)

    start = cursor % len(groups)
    order = groups[start:] + groups[:start]
    selected: list[Candidate] = []
    positive_quanta: list[float] = []
    for group in order:
        capacity = max(0, int(live_servers.get((group[1], group[2]), 0)))
        weight = float(weights.get(group[0], 1.0))
        if weight <= 0 or not math.isfinite(weight):
            raise ValueError(f"run weight must be finite and positive: {group[0]}={weight}")
        if capacity:
            positive_quanta.append(capacity * weight)

    # Enough cycles to accumulate the largest cell cost even with the smallest quantum,
    # plus one cycle per potential selection.  This is a safety bound, not a policy cap.
    max_cost = max(candidate.fanout_cost for candidate in candidates)
    min_quantum = min(positive_quanta, default=1.0)
    max_cycles = int(math.ceil(max_cost / min_quantum)) + max_tasks + 2

    for _ in range(max_cycles):
        admitted_this_cycle = False
        feasible_this_cycle = False
        for group in order:
            if len(selected) >= max_tasks:
                break
            queue = queues[group]
            if not queue:
                continue
            profile = (group[1], group[2])
            capacity = max(0, int(live_servers.get(profile, 0)))
            if capacity == 0:
                continue
            weight = float(weights.get(group[0], 1.0))
            quantum = capacity * weight
            encoded_group = _key(group)
            new_deficits[encoded_group] = new_deficits.get(encoded_group, 0.0) + quantum
            encoded_profile = _key(profile)
            # Canonical DRR drains as much of this group's quantum as its queued work
            # and shared profile budget allow.  Rotating the first group each poll keeps
            # even a large quantum from starving later groups at a small batch boundary.
            while queue and len(selected) < max_tasks:
                room = remaining.get(encoded_profile, 0)
                if any(candidate.fanout_cost <= room for candidate in queue):
                    feasible_this_cycle = True
                selected_index = next(
                    (
                        index
                        for index, candidate in enumerate(queue)
                        if candidate.fanout_cost <= room
                        and candidate.fanout_cost
                        <= new_deficits[encoded_group] + 1e-12
                    ),
                    None,
                )
                if selected_index is None:
                    break
                candidate = queue.pop(selected_index)
                new_deficits[encoded_group] -= candidate.fanout_cost
                remaining[encoded_profile] = room - candidate.fanout_cost
                selected.append(candidate)
                admitted_this_cycle = True
        if len(selected) >= max_tasks:
            break
        if not any(queues.values()):
            break
        # No profile has enough hard headroom; more deficit cannot help.
        if not admitted_this_cycle and not feasible_this_cycle:
            break

    return AdmissionResult(
        selected=tuple(selected),
        deficits=new_deficits,
        cursor=(start + 1) % len(groups),
        remaining_profile_headroom=remaining,
    )


def update_starvation_counters(
    run_records: Mapping[str, Mapping[str, Any]],
    *,
    backlogged_runs: Iterable[str],
    admitted_runs: Iterable[str],
    poll_number: int,
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    """Pure two-poll starvation accounting used by the daemon and tests."""
    backlog = set(backlogged_runs)
    admitted = set(admitted_runs)
    updated = {run_id: dict(record) for run_id, record in run_records.items()}
    warnings: list[str] = []
    for run_id, record in updated.items():
        previous = int(record.get("backlogged_polls_without_admission", 0))
        if run_id not in backlog or run_id in admitted:
            record["backlogged_polls_without_admission"] = 0
            if run_id in admitted:
                record["last_admitted_poll"] = poll_number
            continue
        current = previous + 1
        record["backlogged_polls_without_admission"] = current
        if current >= 2 and previous < 2:
            record["last_starvation_warning_poll"] = poll_number
            warnings.append(run_id)
    return updated, tuple(warnings)


def parse_squeue(text: str) -> list[QueueRow]:
    """Parse ``squeue -o '%F|%K|%i|%j|%T|%o'`` output.

    Five-column input remains accepted for isolated unit tests/backward compatibility,
    but live queries always include ``%o`` so bare legacy job names can be resolved from
    their run-scoped sbatch path.
    """
    rows: list[QueueRow] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.rstrip("\n").split("|", 5)
        if len(fields) not in {5, 6}:
            raise DispatcherError(f"malformed squeue row {line_number}: {raw!r}")
        if len(fields) == 5:
            fields.append("")
        array_job_id, task, job_id, name, state, command = (
            field.strip() for field in fields
        )
        task_id = None if task in {"", "N/A"} else int(task)
        rows.append(QueueRow(array_job_id, task_id, job_id, name, state, command))
    return rows


def is_cell_job_name(name: str) -> bool:
    return name == "asys-cells" or any(name.startswith(prefix) for prefix in CELL_JOB_PREFIXES)


def _empty_ledger(now: float | None = None) -> dict[str, Any]:
    timestamp = time.time() if now is None else now
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "created_at": timestamp,
        "updated_at": timestamp,
        "poll_number": 0,
        "fairness": {"cursor": 0, "deficits": {}},
        "validation_fairness": {"next_run_id": None},
        "runs": {},
        "jobs": {},
        "intents": {},
        "cells": {},
    }


LEDGER_MAPPING_FIELDS = (
    "runs",
    "jobs",
    "intents",
    "cells",
    "fairness",
    "validation_fairness",
)
LEDGER_RECORD_MAPPING_FIELDS = ("runs", "jobs", "intents", "cells")


def validate_ledger_structure(
    value: Any, *, source: str = "dispatcher ledger"
) -> dict[str, Any]:
    """Return the production-normalized ledger or reject it fail closed.

    This is deliberately shared with the production monitor.  A merely fresh JSON
    object is not scheduler truth: every consumer must agree on the schema and the
    mapping fields whose contents drive admission and health decisions.
    """

    if not isinstance(value, dict):
        raise DispatcherError(f"{source} must be a JSON object")
    if value.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise DispatcherError(
            f"unsupported dispatcher ledger schema in {source}: "
            f"{value.get('schema_version')!r}"
        )
    normalized = dict(value)
    # These two fields were introduced after the first schema-1 ledger was emitted.
    # Production loading has always upgraded their absence in memory; the monitor
    # must use exactly the same compatibility boundary.
    normalized.setdefault("intents", {})
    normalized.setdefault("validation_fairness", {"next_run_id": None})
    for field in LEDGER_MAPPING_FIELDS:
        if not isinstance(normalized.get(field), dict):
            raise DispatcherError(
                f"dispatcher ledger field {field!r} in {source} must be an object"
            )
    for field in LEDGER_RECORD_MAPPING_FIELDS:
        for key, record in normalized[field].items():
            if not isinstance(key, str) or not isinstance(record, dict):
                raise DispatcherError(
                    f"dispatcher ledger {field} record {key!r} in {source} "
                    "must be an object under a string key"
                )
    return normalized


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_ledger()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DispatcherError(f"cannot read dispatcher ledger {path}: {exc}") from exc
    return validate_ledger_structure(value, source=str(path))


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace one JSON file without exposing a partial ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _sealed_artifact_sha256(path: Path) -> str:
    """Hash one regular, non-symlink, read-only dispatcher transaction artifact."""

    if path.is_symlink() or not path.is_file():
        raise DispatcherError(f"dispatcher transaction artifact is unsafe: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o222:
        raise DispatcherError(f"dispatcher transaction artifact is writable: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_dispatch_artifact(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise DispatcherError(f"cannot seal dispatcher transaction artifact: {path}")
    path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return _sealed_artifact_sha256(path)


@contextmanager
def _exclusive_lock(lock_path: Path, *, scope: str) -> Iterator[None]:
    """Hold one non-blocking filesystem lock for the caller's full lifetime."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SingletonAlreadyRunning(
                f"another dispatcher owns the {scope} lock {lock_path}; "
                "use the read-only status command"
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} started_at={time.time():.6f}\n".encode())
        os.fsync(fd)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def global_dispatcher_lock(results_root: Path) -> Iterator[None]:
    """Serialize every coordinator state directory under one results root.

    The state ledger is intentionally replaceable across protocol migrations.  It cannot
    therefore be the only singleton authority: otherwise ``.dispatcher`` and
    ``.dispatcher-v3`` can independently admit against the same user QOS.  This root-level
    lock closes that split-brain path while keeping dry-run/status entirely read-only.
    """
    with _exclusive_lock(
        results_root / ".global-dispatcher.lock", scope="results-root global dispatcher"
    ):
        yield


@contextmanager
def singleton_lock(state_dir: Path) -> Iterator[None]:
    """Serialize restarts that share one durable state directory."""
    with _exclusive_lock(state_dir / "dispatcher.lock", scope="state-directory"):
        yield


def _parse_assignments(values: Sequence[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise DispatcherError(f"{option} expects RUN_ID=VALUE, got {value!r}")
        run_id, assigned = value.split("=", 1)
        if not run_id or not assigned:
            raise DispatcherError(f"{option} expects non-empty RUN_ID=VALUE")
        if run_id in parsed:
            raise DispatcherError(f"duplicate {option} assignment for {run_id!r}")
        parsed[run_id] = assigned
    return parsed


def _resolve_run_root(value: str, results_root: Path) -> tuple[str, Path]:
    if "=" in value:
        run_id, root_text = value.split("=", 1)
        if not run_id or not root_text:
            raise DispatcherError(f"--run expects RUN_ID or RUN_ID=RUN_ROOT, got {value!r}")
        return run_id, Path(root_text).expanduser().resolve()
    return value, (results_root / value).resolve()


def _resolve_server_pool(value: str, results_root: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (results_root / value).resolve()


def load_run_specs(
    run_values: Sequence[str],
    *,
    results_root: Path,
    server_pool_values: Sequence[str] = (),
    weight_values: Sequence[str] = (),
) -> list[RunSpec]:
    """Load and checksum immutable manifests through the shared manifest API."""
    pools = _parse_assignments(server_pool_values, option="--server-pool")
    raw_weights = _parse_assignments(weight_values, option="--weight")
    resolved = [_resolve_run_root(value, results_root) for value in run_values]
    if len({run_id for run_id, _ in resolved}) != len(resolved):
        raise DispatcherError("duplicate --run id")
    unknown = (set(pools) | set(raw_weights)) - {run_id for run_id, _ in resolved}
    if unknown:
        raise DispatcherError(f"assignments reference unknown runs: {sorted(unknown)}")

    specs: list[RunSpec] = []
    for run_id, run_root in resolved:
        if run_root.name != run_id:
            raise DispatcherError(
                f"run root basename must equal run id so workers cannot redirect results: "
                f"{run_id!r} -> {run_root}"
            )
        manifest = load_manifest(run_root)
        try:
            question_catalog = VerifiedQuestionCatalog(run_root, snapshot=manifest)
        except BenchmarkContractError as exc:
            raise DispatcherError(
                f"run {run_id!r} has no valid frozen benchmark Question contract: {exc}"
            ) from exc
        pool_arg = pools.get(run_id)
        pool_root = run_root if pool_arg is None else _resolve_server_pool(pool_arg, results_root)
        try:
            weight = float(raw_weights.get(run_id, "1"))
        except ValueError as exc:
            raise DispatcherError(f"invalid --weight for {run_id!r}") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise DispatcherError(f"--weight must be finite and positive for {run_id!r}")
        specs.append(
            RunSpec(
                run_id,
                run_root,
                manifest,
                pool_arg,
                pool_root,
                weight,
                question_catalog,
            )
        )
    return specs


def _register_run_pins(ledger: dict[str, Any], specs: Sequence[RunSpec]) -> None:
    """Refuse manifest drift even when an operator removes ``cells.sha256``."""
    for spec in specs:
        previous = ledger["runs"].get(spec.run_id)
        if previous and previous.get("manifest_sha256") != spec.manifest.sha256:
            raise DispatcherError(
                f"immutable manifest changed for run {spec.run_id!r}: ledger pins "
                f"{previous.get('manifest_sha256')}, disk has {spec.manifest.sha256}"
            )
        contract_sha256 = (
            spec.question_catalog.sidecar_sha256
            if spec.question_catalog is not None
            else None
        )
        if (
            previous
            and previous.get("benchmark_contracts_sha256") is not None
            and previous.get("benchmark_contracts_sha256") != contract_sha256
        ):
            raise DispatcherError(
                f"immutable benchmark contract changed for run {spec.run_id!r}: "
                f"ledger pins {previous.get('benchmark_contracts_sha256')}, disk has "
                f"{contract_sha256}"
            )
        record = dict(previous or {})
        record.update(
            {
                "run_root": str(spec.run_root),
                "manifest_path": str(spec.manifest.path),
                "manifest_sha256": spec.manifest.sha256,
                "manifest_cells": len(spec.manifest.cells),
                "benchmark_contracts_sha256": contract_sha256,
                "server_pool_arg": spec.server_pool_arg,
                "server_pool_root": str(spec.server_pool_root),
                "weight": spec.weight,
            }
        )
        record.setdefault("backlogged_polls_without_admission", 0)
        ledger["runs"][spec.run_id] = record


def _query_squeue() -> list[QueueRow]:
    user = os.environ.get("USER")
    if not user:
        raise DispatcherError("USER is unset; cannot scope squeue safely")
    proc = subprocess.run(
        ["squeue", "-u", user, "-h", "-r", "-o", "%F|%K|%i|%j|%T|%o"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise DispatcherError(f"squeue failed: {proc.stderr.strip()[:500]}")
    return parse_squeue(proc.stdout)


def _array_identity(raw_job_id: str) -> tuple[str, int | None]:
    base, separator, suffix = raw_job_id.partition("_")
    if separator and suffix.isdigit():
        return base, int(suffix)
    return raw_job_id, None


def _queue_rows_from_scheduler_snapshot(snapshot: Any) -> list[QueueRow]:
    rows: list[QueueRow] = []
    for job in snapshot.jobs:
        if job.source != "squeue" or not job.active:
            continue
        base, index = _array_identity(str(job.job_id))
        rows.append(
            QueueRow(
                array_job_id=base,
                array_task_id=index,
                job_id=str(job.job_id),
                job_name=str(job.job_name),
                state=str(job.state),
                command=str(job.command),
                comment=str(job.comment),
            )
        )
    return rows


def _complete_scheduler_rows(
    snapshot: Any,
    *,
    observation: str,
) -> tuple[tuple[QueueRow, ...], tuple[str, ...]]:
    """Extract one complete, expanded, logical-ID squeue observation."""

    if (
        getattr(snapshot, "squeue_ok", False) is not True
        or getattr(snapshot, "sacct_ok", False) is not True
        or tuple(getattr(snapshot, "errors", ()))
    ):
        raise DispatcherError(
            f"{observation} lacks complete squeue+sacct scheduler truth"
        )
    rows = tuple(_queue_rows_from_scheduler_snapshot(snapshot))
    live_ids = tuple(sorted(row.job_id for row in rows))
    if len(live_ids) != len(set(live_ids)):
        raise DispatcherError(
            f"{observation} repeats a live logical Slurm job element"
        )
    return rows, live_ids


def _capture_stable_admission_occupancy(
    *,
    user: str,
    partition: str,
    max_attempts: int = 2,
    scheduler_reader: Any | None = None,
    usage_reader: Any | None = None,
) -> StableAdmissionOccupancy:
    """Bracket target TRES usage with identical global logical job-element IDs.

    The partition usage query and the global squeue+sacct join are separate Slurm
    RPCs.  A job admitted between them could otherwise be charged to only the global
    submit count or only the target CPU/memory envelope.  Two complete scheduler
    snapshots surrounding the usage capture close that skew.  Churn is retried once
    and then fails the admission boundary closed; the next dispatcher poll may retry.
    """

    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or max_attempts < 1
    ):
        raise DispatcherError("stable occupancy max_attempts must be positive")
    if scheduler_reader is None:
        from slurm.schema5_control import query_scheduler

        scheduler_reader = lambda: query_scheduler(tolerate_errors=False)
    if usage_reader is None:
        usage_reader = lambda: scheduler_safety.capture_user_partition_usage(
            user=user,
            partition=partition,
        )

    last_churn = "unknown scheduler churn"
    for attempt in range(1, max_attempts + 1):
        first = scheduler_reader()
        _first_rows, first_ids = _complete_scheduler_rows(
            first,
            observation=f"stable occupancy attempt {attempt} first observation",
        )
        usage = usage_reader()
        try:
            usage_summary = scheduler_safety.validate_user_partition_usage(
                usage
            )
        except scheduler_safety.SchedulerSafetyError as exc:
            raise DispatcherError(
                f"stable occupancy target usage is invalid: {exc}"
            ) from exc
        second = scheduler_reader()
        second_rows, second_ids = _complete_scheduler_rows(
            second,
            observation=f"stable occupancy attempt {attempt} second observation",
        )
        usage_ids = {
            str(row["job_id"]) for row in usage_summary["jobs"]
        }
        stable_ids = set(second_ids)
        added = sorted(stable_ids - set(first_ids))
        removed = sorted(set(first_ids) - stable_ids)
        usage_only = sorted(usage_ids - stable_ids)
        if not added and not removed and not usage_only:
            return StableAdmissionOccupancy(
                scheduler_snapshot=second,
                rows=second_rows,
                usage=dict(usage),
                usage_summary=dict(usage_summary),
                live_job_ids=second_ids,
                attempts=attempt,
            )
        last_churn = (
            f"added={added[:8]}, removed={removed[:8]}, "
            f"target_usage_only={usage_only[:8]}"
        )
    raise DispatcherError(
        "scheduler occupancy changed across the target-usage admission "
        f"observation after {max_attempts} attempts: {last_churn}"
    )


def _command_binds_exact_sbatch(command: str, expected_path: str) -> bool:
    try:
        expected = str(Path(expected_path).expanduser().resolve())
        return any(
            Path(token).is_absolute()
            and str(Path(token).expanduser().resolve()) == expected
            for token in shlex.split(command)
            if token.endswith(".sbatch")
        )
    except (OSError, ValueError):
        return False


def _intent_visibility_started_at(intent: Mapping[str, Any]) -> float | None:
    """Return the timestamp at which an intent could first have reached Slurm.

    A dispatcher poll can spend minutes validating a large manifest before admission.
    Using the poll/intent creation time would consume the entire scheduler-visibility
    grace before ``sbatch`` is invoked and could misclassify a newly accepted job as
    absent.  Prepared intents have not crossed that boundary; submitting intents must
    carry the separately fsynced boundary timestamp.
    """

    field = (
        "submit_started_at"
        if intent.get("state") in {"submitting", "submitted", "reconciled"}
        else "created_at"
    )
    value = intent.get(field)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _invisible_reservation_count(
    ledger: Mapping[str, Any],
    *,
    now: float,
    exclude_intent_ids: frozenset[str] = frozenset(),
) -> int:
    """Count scheduler-invisible cell tasks once at an admission boundary."""

    job_reservations = sum(
        int(record.get("task_count", len(record.get("tasks", []))))
        for record in ledger.get("jobs", {}).values()
        if isinstance(record, Mapping)
        and record.get("state") == "visibility_grace"
    )
    intent_reservations = 0
    for intent_id, intent in ledger.get("intents", {}).items():
        if (
            intent_id in exclude_intent_ids
            or not isinstance(intent, Mapping)
            or intent.get("state") not in {"prepared", "submitting"}
        ):
            continue
        visibility_started_at = _intent_visibility_started_at(intent)
        if (
            visibility_started_at is not None
            and now - visibility_started_at < 300.0
        ):
            tasks = intent.get("tasks", [])
            if not isinstance(tasks, list):
                raise DispatcherError(
                    f"invisible intent {intent_id} has an invalid task payload"
                )
            intent_reservations += len(tasks)
    return job_reservations + intent_reservations


def _reconcile_schema5_intents(
    ledger: dict[str, Any],
    *,
    scheduler_snapshot: Any,
    now: float,
    visibility_grace_s: float = 300.0,
) -> tuple[list[str], list[str]]:
    """Resolve every pre-sbatch intent through joined squeue+sacct authority."""

    warnings: list[str] = []
    errors: list[str] = []
    for batch_id, intent in ledger.get("intents", {}).items():
        intent_state = intent.get("state")
        if intent_state not in {
            "prepared",
            "submitting",
            "submitted",
            "reconciled",
        }:
            continue
        visibility_started_at = _intent_visibility_started_at(intent)
        if visibility_started_at is None:
            errors.append(
                f"intent {batch_id} lacks a valid scheduler-boundary timestamp"
            )
            continue
        expected_comment = f"asys-schema5-intent:{batch_id}"
        expected_path = str(Path(str(intent.get("sbatch_path", ""))).resolve())
        expected_name = f"asys-dispatch-{batch_id[-10:]}"
        artifacts = (
            ("batch_manifest", "batch_manifest_sha256"),
            ("sbatch_path", "sbatch_sha256"),
        )
        artifact_error: str | None = None
        for path_field, hash_field in artifacts:
            expected_hash = intent.get(hash_field)
            try:
                observed_hash = _sealed_artifact_sha256(
                    Path(str(intent.get(path_field, ""))).expanduser().resolve()
                )
            except (DispatcherError, OSError) as exc:
                artifact_error = str(exc)
                break
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash)
                or observed_hash != expected_hash
            ):
                artifact_error = f"{path_field} bytes differ from durable intent"
                break
        if artifact_error is not None:
            errors.append(f"intent {batch_id} artifact drift: {artifact_error}")
            continue
        matching_by_base: dict[str, list[Any]] = defaultdict(list)
        for job in scheduler_snapshot.jobs:
            comment = str(job.comment)
            path_matches = _command_binds_exact_sbatch(
                str(job.command), expected_path
            )
            namespace_match = (
                comment == expected_comment
                or path_matches
                or str(job.job_name) == expected_name
            )
            if not namespace_match:
                continue
            if (
                str(job.job_name) != expected_name
                or not path_matches
                or (comment and comment != expected_comment)
                or (str(job.source) == "squeue" and comment != expected_comment)
            ):
                errors.append(
                    f"intent {batch_id} scheduler provenance drift for job "
                    f"{job.job_id}: name={job.job_name!r} comment={comment!r}"
                )
                continue
            base, _ = _array_identity(str(job.job_id))
            matching_by_base[base].append(job)
        if len(matching_by_base) > 1:
            errors.append(
                f"intent {batch_id} ambiguously maps to jobs "
                f"{sorted(matching_by_base)} across squeue/sacct"
            )
            continue
        if not matching_by_base:
            age = max(0.0, now - visibility_started_at)
            recorded_job_id = intent.get("job_id")
            recorded_job = (
                ledger.get("jobs", {}).get(str(recorded_job_id))
                if recorded_job_id is not None
                else None
            )
            if intent_state in {"submitted", "reconciled"}:
                if (
                    isinstance(recorded_job, dict)
                    and recorded_job.get("state") in {"terminal", "inactive"}
                ):
                    continue
                if age >= visibility_grace_s:
                    errors.append(
                        f"accepted intent {batch_id} job {recorded_job_id} disappeared "
                        "from complete squeue+sacct truth"
                    )
                continue
            if age >= visibility_grace_s:
                intent.update(
                    {
                        "state": "not_accepted",
                        "reconciled_at": now,
                        "error": "absent from complete squeue+sacct transaction history",
                    }
                )
                warnings.append(f"intent {batch_id} was not accepted by Slurm")
            continue
        job_id, matching = next(iter(matching_by_base.items()))
        recorded_job_id = intent.get("job_id")
        if recorded_job_id is not None and str(recorded_job_id) != job_id:
            errors.append(
                f"intent {batch_id} records job {recorded_job_id} but scheduler "
                f"provenance maps job {job_id}"
            )
            continue
        tasks = intent.get("tasks")
        if not isinstance(tasks, list):
            errors.append(f"intent {batch_id} has invalid task payload")
            continue
        active = any(job.active for job in matching)
        states = sorted({str(job.state) for job in matching})
        record = ledger["jobs"].get(job_id)
        expected_record = {
            "job_id": job_id,
            "batch_id": batch_id,
            "batch_manifest": str(intent.get("batch_manifest")),
            "batch_manifest_sha256": str(intent.get("batch_manifest_sha256")),
            "sbatch_path": str(intent.get("sbatch_path")),
            "sbatch_sha256": str(intent.get("sbatch_sha256")),
            "submitted_at": float(intent.get("created_at", now)),
            "last_seen_at": now,
            "reconciled_at": now,
            "state": "active" if active else "terminal",
            "scheduler_states": states,
            "task_count": len(tasks),
            "tasks": tasks,
        }
        if record is not None and (
            record.get("batch_id") != batch_id
            or record.get("tasks") != tasks
            or record.get("batch_manifest_sha256")
            != intent.get("batch_manifest_sha256")
            or record.get("sbatch_sha256") != intent.get("sbatch_sha256")
        ):
            errors.append(f"job {job_id} conflicts with intent {batch_id}")
            continue
        if record is None:
            ledger["jobs"][job_id] = expected_record
        else:
            record.update(
                {
                    "last_seen_at": now,
                    "reconciled_at": now,
                    "state": expected_record["state"],
                    "scheduler_states": states,
                }
            )
        intent.update(
            {
                "state": "reconciled",
                "job_id": job_id,
                "reconciled_at": now,
            }
        )
        _commit_intent_fairness(ledger, batch_id)
        warnings.append(f"reconciled intent {batch_id} to Slurm job {job_id}")
    return warnings, errors


def _task_from_candidate(candidate: Candidate) -> dict[str, Any]:
    task = {
        "run_id": candidate.run_id,
        "run_root": candidate.run_root,
        "source_index": candidate.source_index,
        "cell_id": candidate.cell_id,
        "config_hash": candidate.cell.config_hash(),
        "manifest_sha256": candidate.manifest_sha256,
        "benchmark_contracts_sha256": candidate.benchmark_contracts_sha256,
        "model_size": candidate.cell.model_size,
        "serving_profile": candidate.serving_profile,
        "fanout_cost": candidate.fanout_cost,
        "server_pool_id": candidate.server_pool_arg,
        # Use the absolute pool root at execution time.  This remains correct when a
        # run was registered from a non-default RESULTS_ROOT.
        "server_run_id": (
            candidate.server_pool_root if candidate.server_pool_arg is not None else None
        ),
        "server_pool_root": candidate.server_pool_root,
    }
    if candidate.runtime_environment:
        task["runtime_environment"] = dict(candidate.runtime_environment)
    return task


def _candidate_from_task(task: Mapping[str, Any], specs: Mapping[str, RunSpec]) -> Candidate | None:
    """Reconstruct a task only when every immutable field matches current pins.

    Persisted job/intent payloads are untrusted recovery inputs.  In particular they
    must not redirect result writes or endpoint lookup, understate fan-out, or smuggle
    a stale runtime generation merely because their run/index/cell tuple is valid.
    Rebuild the sole canonical payload from the frozen RunSpec and compare it exactly.
    """

    run_id = str(task.get("run_id", ""))
    spec = specs.get(run_id)
    if spec is None:
        return None
    try:
        index = int(task["source_index"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= index < len(spec.manifest.cells):
        return None
    cell = spec.manifest.cells[index]
    if cell.cell_id != task.get("cell_id"):
        return None
    contract_sha256 = (
        spec.question_catalog.sidecar_sha256
        if spec.question_catalog is not None
        else None
    )
    if task.get("benchmark_contracts_sha256") != contract_sha256:
        return None
    profile = serving_profile_for_cell(cell).registry_key
    candidate = Candidate(
        run_id=run_id,
        run_root=str(spec.run_root),
        source_index=index,
        cell=cell,
        manifest_sha256=spec.manifest.sha256,
        server_pool_arg=spec.server_pool_arg,
        server_pool_root=str(spec.server_pool_root),
        serving_profile=profile,
        fanout_cost=fanout_cost(cell),
        benchmark_contracts_sha256=contract_sha256,
        runtime_environment=spec.runtime_environment,
    )
    if dict(task) != _task_from_candidate(candidate):
        return None
    return candidate


def _active_cells(
    rows: Sequence[QueueRow],
    ledger: dict[str, Any],
    specs: Sequence[RunSpec],
    *,
    now: float,
    visibility_grace_s: float = 300.0,
    schema5_strict: bool = False,
    trusted_cell_bindings: dict[str, dict[str, str]] | None = None,
) -> tuple[
    dict[tuple[str, str], Candidate],
    dict[ProfileKey, int],
    list[str],
    list[str],
]:
    """Join live array indices to ledger batch manifests and run-scoped legacy manifests."""
    by_run = {spec.run_id: spec for spec in specs}
    active: dict[tuple[str, str], Candidate] = {}
    active_load: dict[ProfileKey, int] = defaultdict(int)
    warnings: list[str] = []
    unmappable: list[str] = []
    verified_job_artifacts: dict[str, str | None] = {}

    # One immutable sbatch/intent may map to exactly one Slurm array allocation.  Array
    # task rows legitimately repeat the same base id; two distinct base ids for the same
    # script are an ambiguous duplicate admission and globally fence new work.
    dispatcher_jobs_by_sbatch: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not row.job_name.startswith("asys-dispatch-"):
            continue
        try:
            sbatch_path = next(
                str(Path(part).expanduser().resolve())
                for part in shlex.split(row.command)
                if part.endswith(".sbatch")
            )
        except (StopIteration, OSError, ValueError):
            continue
        dispatcher_jobs_by_sbatch[sbatch_path].add(row.array_job_id)
    ambiguous_sbatch_paths = {
        path: sorted(job_ids)
        for path, job_ids in dispatcher_jobs_by_sbatch.items()
        if len(job_ids) > 1
    }
    for path, job_ids in sorted(ambiguous_sbatch_paths.items()):
        unmappable.append(
            f"ambiguous duplicate dispatcher intent {path}: jobs {job_ids}"
        )

    def add_active(candidate: Candidate) -> None:
        active[(candidate.run_id, candidate.cell_id)] = candidate
        # Deliberately count every live/reserved task, even when duplicate legacy arrays
        # point at the same cell.  The advisory lock prevents duplicate mutation, but all
        # such clients still consume QOS and can briefly consume endpoint capacity.
        active_load[candidate.profile_key] += candidate.fanout_cost

    def reserve_tasks(tasks: Any, label: str) -> None:
        if not isinstance(tasks, list):
            unmappable.append(f"{label}: tasks are not a list")
            return
        for task in tasks:
            if not isinstance(task, dict):
                unmappable.append(f"{label}: task is not an object")
                continue
            candidate = _candidate_from_task(task, by_run)
            if candidate is None:
                unmappable.append(f"{label}: task does not match a registered manifest")
            else:
                add_active(candidate)

    live_array_ids = {row.array_job_id for row in rows}
    for job_id, record in ledger["jobs"].items():
        prior_state = record.get("state")
        record["state"] = "active" if job_id in live_array_ids else "inactive"
        if job_id in live_array_ids:
            record["last_seen_at"] = now
        elif (
            prior_state in {"submitted", "active"}
            and now - float(record.get("submitted_at", 0.0)) < visibility_grace_s
        ):
            record["state"] = "visibility_grace"
            reserve_tasks(record.get("tasks"), f"job {job_id} visibility reservation")

    for row in rows:
        if row.job_name.startswith("asys-dispatch-"):
            try:
                row_sbatch = next(
                    str(Path(part).expanduser().resolve())
                    for part in shlex.split(row.command)
                    if part.endswith(".sbatch")
                )
            except (StopIteration, OSError, ValueError):
                row_sbatch = None
            if row_sbatch in ambiguous_sbatch_paths:
                continue
        job_record = ledger["jobs"].get(row.array_job_id)
        if job_record is None and row.job_name.startswith("asys-dispatch-"):
            if schema5_strict:
                # A schema-5 intent is fsynced before sbatch.  Therefore an active
                # production array without a reconciled ledger record is foreign or
                # ambiguous; reading an attacker-chosen adjacent JSON file is never a
                # valid crash-recovery path.
                unmappable.append(
                    f"{row.job_id} ({row.job_name}): no durable schema-5 intent record"
                )
                continue
            # Recovery for the narrow crash window after sbatch accepts a microbatch but
            # before its job id reaches the ledger.  Slurm retains the submitted sbatch
            # path in %o; our .json batch manifest has the same basename.
            try:
                command_parts = shlex.split(row.command)
                sbatch_path = next(
                    Path(part) for part in command_parts if part.endswith(".sbatch")
                )
                manifest_path = sbatch_path.with_suffix(".json")
                batch = json.loads(manifest_path.read_text(encoding="utf-8"))
                tasks = batch["tasks"]
                if batch.get("schema_version") != 1 or not isinstance(tasks, list):
                    raise ValueError("invalid recovered batch schema")
                job_record = {
                    "job_id": row.array_job_id,
                    "batch_id": str(batch.get("batch_id", manifest_path.stem)),
                    "batch_manifest": str(manifest_path),
                    "sbatch_path": str(sbatch_path),
                    "submitted_at": now,
                    "last_seen_at": now,
                    "reconciled_at": now,
                    "state": "active",
                    "task_count": len(tasks),
                    "tasks": tasks,
                }
                ledger["jobs"][row.array_job_id] = job_record
                if job_record["batch_id"] in ledger.get("intents", {}):
                    ledger["intents"][job_record["batch_id"]].update(
                        {"state": "reconciled", "job_id": row.array_job_id}
                    )
                    _commit_intent_fairness(ledger, job_record["batch_id"])
                warnings.append(
                    f"reconciled dispatcher array {row.array_job_id} from {manifest_path}"
                )
            except (StopIteration, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                unmappable.append(f"{row.job_id} ({row.job_name}): {exc}")
                continue
        if job_record is not None:
            if schema5_strict and is_cell_job_name(row.job_name):
                batch_id = str(job_record.get("batch_id", ""))
                expected_name = f"asys-dispatch-{batch_id[-10:]}"
                expected_comment = f"asys-schema5-intent:{batch_id}"
                artifact_error = verified_job_artifacts.get(row.array_job_id)
                if row.array_job_id not in verified_job_artifacts:
                    artifact_error = None
                    for path_field, hash_field in (
                        ("batch_manifest", "batch_manifest_sha256"),
                        ("sbatch_path", "sbatch_sha256"),
                    ):
                        expected_hash = job_record.get(hash_field)
                        try:
                            observed_hash = _sealed_artifact_sha256(
                                Path(str(job_record.get(path_field, "")))
                                .expanduser()
                                .resolve()
                            )
                        except (DispatcherError, OSError) as exc:
                            artifact_error = str(exc)
                            break
                        if observed_hash != expected_hash:
                            artifact_error = (
                                f"{path_field} differs from its durable hash"
                            )
                            break
                    verified_job_artifacts[row.array_job_id] = artifact_error
                if (
                    not batch_id
                    or row.job_name != expected_name
                    or row.comment != expected_comment
                    or artifact_error is not None
                    or not _command_binds_exact_sbatch(
                        row.command, str(job_record.get("sbatch_path", ""))
                    )
                ):
                    unmappable.append(
                        f"{row.job_id} ({row.job_name}): live scheduler provenance "
                        "differs from durable schema-5 intent"
                    )
                    continue
            tasks = job_record.get("tasks", [])
            indices = range(len(tasks)) if row.array_task_id is None else (row.array_task_id,)
            mapped = False
            for index in indices:
                if 0 <= index < len(tasks):
                    candidate = _candidate_from_task(tasks[index], by_run)
                    if candidate is not None:
                        add_active(candidate)
                        mapped = True
            if not mapped and is_cell_job_name(row.job_name):
                unmappable.append(f"{row.job_id} ({row.job_name}): invalid ledger task mapping")
            elif mapped and schema5_strict and trusted_cell_bindings is not None:
                binding = {
                    "job_name": row.job_name,
                    "comment": row.comment,
                }
                prior = trusted_cell_bindings.setdefault(row.job_id, binding)
                if prior != binding:
                    unmappable.append(
                        f"{row.job_id} ({row.job_name}): scheduler identity has "
                        "conflicting trusted client provenance"
                    )
            continue

        # Legacy arrays used their run's canonical cells.json index directly.  Exact
        # run-scoped names make the join unambiguous during controlled drain-down.
        spec: RunSpec | None = None
        if row.job_name.startswith("asys-cells-"):
            spec = by_run.get(row.job_name[len("asys-cells-"):])
        elif row.job_name == "asys-cells":
            command_parts = shlex.split(row.command)
            command_paths = [Path(part).expanduser() for part in command_parts]
            owners = [
                candidate_spec
                for candidate_spec in specs
                if any(
                    path.is_absolute()
                    and (
                        path == candidate_spec.run_root
                        or candidate_spec.run_root in path.parents
                    )
                    for path in command_paths
                )
            ]
            if len(owners) == 1:
                spec = owners[0]
            elif len(specs) == 1 and not row.command:
                # Backward-compatible only for synthetic/older squeue output.  Live
                # queries always include %o and therefore never need this ambiguity.
                spec = specs[0]
        if not is_cell_job_name(row.job_name):
            continue
        if spec is None or row.array_task_id is None:
            unmappable.append(f"{row.job_id} ({row.job_name}): cannot resolve run/index")
            continue
        if 0 <= row.array_task_id < len(spec.manifest.cells):
            cell = spec.manifest.cells[row.array_task_id]
            profile = serving_profile_for_cell(cell).registry_key
            candidate = Candidate(
                run_id=spec.run_id,
                run_root=str(spec.run_root),
                source_index=row.array_task_id,
                cell=cell,
                manifest_sha256=spec.manifest.sha256,
                # Legacy templates always discover endpoints under their result run,
                # regardless of a new dispatcher's cross-run server-pool mapping.
                server_pool_arg=None,
                server_pool_root=str(spec.run_root),
                serving_profile=profile,
                fanout_cost=fanout_cost(cell),
                benchmark_contracts_sha256=(
                    spec.question_catalog.sidecar_sha256
                    if spec.question_catalog is not None
                    else None
                ),
                runtime_environment=spec.runtime_environment,
            )
            add_active(candidate)
        else:
            unmappable.append(f"{row.job_id} ({row.job_name}): array index out of range")

    # A durable pre-submit intent closes both sides of the sbatch crash window.  Before
    # its job id is known, reserve the tasks briefly; if sbatch actually succeeded, %o
    # reconciliation above adopts the array.  A rejected/abandoned intent is not held.
    for intent_id, intent in ledger.get("intents", {}).items():
        if intent.get("state") not in {"prepared", "submitting"}:
            continue
        visibility_started_at = _intent_visibility_started_at(intent)
        if (
            visibility_started_at is not None
            and now - visibility_started_at < visibility_grace_s
        ):
            reserve_tasks(intent.get("tasks"), f"intent {intent_id}")
    return (
        active,
        dict(active_load),
        list(dict.fromkeys(warnings)),
        list(dict.fromkeys(unmappable)),
    )


def _registered_endpoints(
    pool_root: Path,
    profile: str,
    *,
    frozen_fleet: FrozenFleetContract | None = None,
) -> list[ServerEntry]:
    """Read truthful endpoint records without creating or pruning registry state."""
    server_root = pool_root / "servers"
    paths = sorted((server_root / profile).glob("*.json")) if (server_root / profile).is_dir() else []
    legacy = server_root / f"{profile}.json"
    if legacy.is_file():
        paths.append(legacy)
    endpoints: dict[str, ServerEntry] = {}
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            entry = ServerEntry(**value)
            if not entry_has_current_provenance(entry, profile):
                continue
            if frozen_fleet is not None:
                frozen_fleet.verify_pool_root(pool_root)
                if not isinstance(entry.replica_index, int) or isinstance(
                    entry.replica_index, bool
                ):
                    continue
                replica = frozen_fleet.for_replica(profile, entry.replica_index)
                if not (
                    entry.server_pool_id == server_pool_id(pool_root) == replica.pool_id
                    and entry.replica_id == replica.replica_id
                    and entry.fleet_contract_sha256 == frozen_fleet.sha256
                    and entry.port == _port_for(profile, replica.replica_index)
                ):
                    continue
            host, port = str(entry.host), int(entry.port)
            if host and 0 < port <= 65535:
                endpoints[endpoint_instance_id(entry)] = entry
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return [endpoints[key] for key in sorted(endpoints)]


def _trusted_server_scheduler_bindings(
    specs: Sequence[RunSpec],
    *,
    expected_fleet_sha256: str,
    frozen_fleet: FrozenFleetContract | None,
    scheduler_rows: Sequence[QueueRow],
) -> dict[str, dict[str, str]]:
    """Classify only servers with an immutable job/name/comment preimage."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_fleet_sha256 or "") is None:
        raise DispatcherError("trusted server classification lacks a fleet hash")
    live_by_id: dict[str, QueueRow] = {}
    for row in scheduler_rows:
        if row.job_id in live_by_id:
            raise DispatcherError(
                f"scheduler repeats live job identity {row.job_id}"
            )
        live_by_id[row.job_id] = row
    keys = sorted(
        {
            (
                str(spec.server_pool_root.resolve()),
                serving_profile_for_cell(cell).registry_key,
            )
            for spec in specs
            for cell in spec.manifest.cells
        }
    )
    bindings: dict[str, dict[str, str]] = {}
    try:
        for pool_text, profile in keys:
            pool_root = Path(pool_text)
            for entry in _registered_endpoints(
                pool_root,
                profile,
                frozen_fleet=frozen_fleet,
            ):
                if (
                    entry.fleet_contract_sha256 != expected_fleet_sha256
                    or not str(entry.slurm_job_id or "").isdigit()
                ):
                    continue
                history = endpoint_history_for_entry(pool_root, entry)
                if history is None:
                    continue
                row = live_by_id.get(str(entry.slurm_job_id))
                if row is None:
                    continue
                binding = {
                    "job_name": str(history.binding["scheduler_job_name"]),
                    "comment": str(history.binding["scheduler_comment"]),
                }
                if (
                    row.job_name != binding["job_name"]
                    or row.comment != binding["comment"]
                    or not _command_binds_exact_sbatch(
                        row.command,
                        str(history.binding["local_script_path"]),
                    )
                ):
                    continue
                job_id = str(entry.slurm_job_id)
                prior = bindings.setdefault(job_id, binding)
                if prior != binding:
                    raise DispatcherError(
                        f"server job {job_id} has conflicting immutable provenance"
                    )
    except (OSError, TypeError, ValueError, FleetContractError) as exc:
        raise DispatcherError(
            f"cannot classify protected server allocations: {exc}"
        ) from exc
    return bindings


def discover_capacity(
    keys: Iterable[ProfileKey],
    *,
    probe: bool = True,
    probe_timeout: float = 1.0,
    overrides: Mapping[ProfileKey, int] | None = None,
    active_job_ids: set[str] | None = None,
    probe_attempts: int = 3,
    frozen_fleet: FrozenFleetContract | None = None,
) -> CapacitySnapshot:
    """Count authoritative replicas and hash endpoint identity.

    Registry entries carrying a SLURM job id are accepted only while that job remains in
    ``squeue``.  This avoids both stale-file inflation and the false negatives produced by
    aggressive HTTP probes when a healthy vLLM engine is saturated.  ``probe=True`` is an
    optional stricter diagnostic, not the production default.
    """
    overrides = dict(overrides or {})
    unique_keys = sorted(set(keys))
    endpoints = {
        key: _registered_endpoints(
            Path(key[0]), key[1], frozen_fleet=frozen_fleet
        )
        for key in unique_keys
    }
    if probe_attempts < 1:
        raise ValueError("probe_attempts must be >= 1")
    active_bases = (
        None
        if active_job_ids is None
        else {identifier.split("_", 1)[0] for identifier in active_job_ids}
    )
    job_ids_to_resolve = {
        job_id.split("_", 1)[0]
        for values in endpoints.values()
        for entry in values
        if entry.slurm_job_id is not None
        and (
            active_bases is None
            or str(entry.slurm_job_id).split("_", 1)[0] in active_bases
        )
        for job_id in (str(entry.slurm_job_id),)
    }
    allocations = (
        active_slurm_allocations(job_ids_to_resolve)
        if active_bases is not None
        else None
    )
    authoritative: dict[ProfileKey, list[ServerEntry]] = {}
    probe_required: set[str] = set()
    for key, values in endpoints.items():
        accepted: list[ServerEntry] = []
        for entry in values:
            host = entry.host
            job_id = entry.slurm_job_id
            if job_id is None:
                accepted.append(entry)
                probe_required.add(endpoint_instance_id(entry))
                continue
            base_job_id = str(job_id).split("_", 1)[0]
            if active_bases is not None and base_job_id not in active_bases:
                continue
            if allocations is None:
                # Scheduler authority could not be resolved (or no active snapshot was
                # supplied): require tolerant HTTP evidence rather than trusting a stale id.
                accepted.append(entry)
                probe_required.add(endpoint_instance_id(entry))
                continue
            allocated_nodes = allocations.get(base_job_id)
            if allocated_nodes is None:
                continue
            if not allocated_nodes:
                accepted.append(entry)
                probe_required.add(endpoint_instance_id(entry))
                continue
            if host in allocated_nodes:
                accepted.append(entry)
            # A live job id on another node is a stale pre-requeue registry record.
        authoritative[key] = accepted
    alive: dict[tuple[str, int], bool] = {}
    # Null-id files come from legacy keepalive re-registration and cannot be proven live
    # through Slurm.  Never count them on registry presence alone: use several probes and
    # accept any success so a single saturated-engine timeout does not flap capacity.
    must_probe = sorted(
        {
            (entry.host, entry.port)
            for values in authoritative.values()
            for entry in values
            if probe or endpoint_instance_id(entry) in probe_required
        }
    )

    def tolerant_probe(host_port: tuple[str, int]) -> bool:
        return any(
            healthcheck.is_alive(host_port[0], host_port[1], timeout=probe_timeout)
            for _ in range(probe_attempts)
        )

    if must_probe:
        with ThreadPoolExecutor(max_workers=min(32, len(must_probe))) as executor:
            alive = dict(zip(must_probe, executor.map(tolerant_probe, must_probe)))

    counts: dict[ProfileKey, int] = {}
    generations: dict[ProfileKey, str] = {}
    for key in unique_keys:
        live = [
            entry
            for entry in authoritative[key]
            if (
                not probe
                and endpoint_instance_id(entry) not in probe_required
            )
            or alive.get((entry.host, entry.port), False)
        ]
        generations[key] = server_pool_generation(live, profile_name=key[1])
        counts[key] = int(overrides[key]) if key in overrides else len(live)
        if counts[key] < 0:
            raise ValueError(f"server capacity override must be non-negative: {key}")
    return CapacitySnapshot(counts, generations)


def _artifact_fingerprint(cell_dir: Path) -> list[list[int | str]]:
    value: list[list[int | str]] = []
    for name in ("results.jsonl", "meta.json", "failure.json"):
        path = cell_dir / name
        try:
            stat = path.stat()
            value.append([name, stat.st_mtime_ns, stat.st_size])
        except FileNotFoundError:
            value.append([name, -1, -1])
    return value


def _validate_pending_scan(
    scan: _CellScan,
    *,
    active: dict[tuple[str, str], Candidate],
    active_cost: dict[ProfileKey, int],
    now: float,
    code_version: str | None,
    model_contract_path: str | None,
    validation_retry_s: float,
) -> str | None:
    """Semantically validate one fairly selected artifact-bearing cell."""
    spec = scan.spec
    cell = scan.candidate.cell
    if (
        scan.cell_dir is None
        or scan.fingerprint is None
        or scan.validation_context is None
    ):
        raise DispatcherError(
            f"internal validation plan lost artifact context for "
            f"{spec.run_id}/{cell.cell_id}"
        )
    if spec.question_catalog is None:
        raise DispatcherError(f"run {spec.run_id!r} has no verified Question catalog")
    try:
        questions = spec.question_catalog.questions_for(cell)
    except BenchmarkContractError as exc:
        raise DispatcherError(
            f"run {spec.run_id!r} benchmark Question drift for {cell.cell_id}: {exc}"
        ) from exc
    try:
        status = get_completion_status(
            cell,
            scan.cell_dir,
            expected_qids=tuple(question.qid for question in questions),
            expected_questions=questions,
            verified_benchmark_contracts=spec.question_catalog.frozen,
            verified_manifest=spec.question_catalog.snapshot,
            check_active=True,
            now=now,
            serving_profile=scan.candidate.serving_profile,
            code_version=code_version,
            server_pool_generation=scan.validation_context["server_pool_generation"],
            model_contract_path=model_contract_path,
        )
    except Exception as exc:  # validation failure must not become blind work
        scan.state = "validation_error"
        scan.eligible = False
        scan.next_eligible = None
        # Cache deterministic/environmental validator failures long enough for other
        # manifest cells to consume the next bounded validation wave.  Fingerprint
        # changes still force an immediate new attempt.
        scan.record["artifact_fingerprint"] = scan.fingerprint
        scan.record["validation_context"] = scan.validation_context
        scan.record["eligible_for_retry"] = False
        scan.record["validation_retry_at"] = now + validation_retry_s
        return f"{spec.run_id}/{cell.cell_id}: completion validation failed: {exc}"

    scan.state = status.status.value
    scan.eligible = bool(status.eligible_for_retry)
    scan.next_eligible = status.next_eligible_at
    scan.record["artifact_fingerprint"] = scan.fingerprint
    scan.record["validation_context"] = scan.validation_context
    scan.record["eligible_for_retry"] = scan.eligible
    scan.record.pop("validation_retry_at", None)
    if scan.state == CompletionState.ACTIVE.value:
        cell_key_tuple = (spec.run_id, cell.cell_id)
        active[cell_key_tuple] = scan.candidate
        active_cost[scan.candidate.profile_key] += scan.candidate.fanout_cost
    return None


def _scan_candidates(
    specs: Sequence[RunSpec],
    *,
    active: dict[tuple[str, str], Candidate],
    active_task_load: Mapping[ProfileKey, int],
    capacity: CapacitySnapshot,
    ledger: dict[str, Any],
    now: float,
    code_version: str | None,
    validation_budget: int,
    model_contract_path: str | None = None,
    validation_retry_s: float = 3600.0,
) -> tuple[list[Candidate], dict[str, dict[str, int]], dict[ProfileKey, int], list[str]]:
    """Incrementally validate artifacts and return cells eligible for admission.

    Results JSONLs are large enough that a full semantic audit takes minutes.  The
    dispatcher therefore reuses a status only when the three artifact fingerprints are
    unchanged, validates at most ``validation_budget`` changed/unknown artifact-bearing
    cells per poll, and classifies truly artifact-free cells as ``missing`` without
    loading a benchmark.  The bounded semantic work is allocated in a work-conserving
    round robin across registered runs; its cursor is separate from admission WDRR and
    is persisted in the atomic ledger.  Advisory-lock checks remain live on every poll.
    A separate audit/repair command remains the right tool for a deliberate full
    validation pass.
    """
    if validation_budget < 0:
        raise ValueError("validation_budget must be non-negative")
    if validation_retry_s < 0:
        raise ValueError("validation_retry_s must be non-negative")
    scans: list[_CellScan] = []
    pending_by_run: dict[str, list[_CellScan]] = defaultdict(list)
    active_cost: dict[ProfileKey, int] = defaultdict(int, active_task_load)
    errors: list[str] = []

    # First do only cheap fingerprint/cache/lock classification.  Semantic validation
    # cannot be allocated fairly until demand from every registered run is known.
    for spec in specs:
        for index, cell in enumerate(spec.manifest.cells):
            profile = serving_profile_for_cell(cell).registry_key
            candidate = Candidate(
                run_id=spec.run_id,
                run_root=str(spec.run_root),
                source_index=index,
                cell=cell,
                manifest_sha256=spec.manifest.sha256,
                server_pool_arg=spec.server_pool_arg,
                server_pool_root=str(spec.server_pool_root),
                serving_profile=profile,
                fanout_cost=fanout_cost(cell),
                benchmark_contracts_sha256=(
                    spec.question_catalog.sidecar_sha256
                    if spec.question_catalog is not None
                    else None
                ),
                runtime_environment=spec.runtime_environment,
            )
            cell_key_tuple = (spec.run_id, cell.cell_id)
            cell_key = _key(cell_key_tuple)
            record = dict(ledger["cells"].get(cell_key, {}))
            fingerprint: list[list[int | str]] | None = None
            validation_context: dict[str, Any] | None = None
            cell_dir: Path | None = None
            needs_validation = False
            if cell_key_tuple in active:
                state = CompletionState.ACTIVE.value
                eligible = False
                next_eligible = None
            else:
                cell_dir = spec.run_root / "cells" / cell.cell_id
                fingerprint = _artifact_fingerprint(cell_dir)
                state_on_disk = str(record.get("completion_state", ""))
                pool_generation = capacity.server_pool_generation.get(candidate.profile_key)
                validation_context = {
                    "manifest_sha256": spec.manifest.sha256,
                    "benchmark_contracts_sha256": (
                        spec.question_catalog.sidecar_sha256
                        if spec.question_catalog is not None
                        else None
                    ),
                    "serving_profile": profile,
                    "code_version": code_version,
                    "server_pool_generation": pool_generation,
                }
                has_artifacts = any(int(item[1]) >= 0 for item in fingerprint)
                # Lock probes are cheap and cannot be cached: a worker may acquire or
                # release the advisory lock without modifying an artifact yet.
                locked = cell_dir.exists() and is_cell_active(cell_dir)
                context_sensitive = state_on_disk in {
                    CompletionState.RETRYABLE.value,
                    CompletionState.PERMANENT.value,
                }
                cached_context = record.get("validation_context")
                base_context_matches = bool(
                    isinstance(cached_context, Mapping)
                    and cached_context.get("manifest_sha256") == spec.manifest.sha256
                    and cached_context.get("benchmark_contracts_sha256")
                    == validation_context["benchmark_contracts_sha256"]
                    and cached_context.get("serving_profile") == profile
                    and cached_context.get("code_version") == code_version
                )
                pool_context_matches = bool(
                    base_context_matches
                    and cached_context.get("server_pool_generation")
                    == pool_generation
                )
                validation_error_cooldown = (
                    state_on_disk == "validation_error"
                    and now < float(record.get("validation_retry_at", 0.0))
                )
                reusable = (
                    record.get("artifact_fingerprint") == fingerprint
                    and state_on_disk
                    not in {
                        "",
                        CompletionState.ACTIVE.value,
                        "unvalidated",
                    }
                    and (
                        state_on_disk != "validation_error" or validation_error_cooldown
                    )
                    and (
                        pool_context_matches
                        if context_sensitive
                        else base_context_matches
                    )
                )
                if locked:
                    state = CompletionState.ACTIVE.value
                    eligible = False
                    next_eligible = None
                    active[cell_key_tuple] = candidate
                    active_cost[candidate.profile_key] += candidate.fanout_cost
                elif not has_artifacts:
                    # This is the one state whose semantic answer is certain without
                    # parsing a results file or loading expected benchmark QIDs.
                    state = CompletionState.MISSING.value
                    eligible = True
                    next_eligible = None
                    record["artifact_fingerprint"] = fingerprint
                    record["validation_context"] = validation_context
                    record["eligible_for_retry"] = True
                elif reusable:
                    state = str(record["completion_state"])
                    next_eligible = record.get("next_eligible_at")
                    eligible = bool(record.get("eligible_for_retry", False))
                    if state == CompletionState.RETRYABLE.value and next_eligible is not None:
                        eligible = now >= float(next_eligible)
                else:
                    # Do not update the fingerprint: the changed cell remains at the
                    # front of a future incremental-validation wave.
                    state = "unvalidated"
                    eligible = False
                    next_eligible = None
                    needs_validation = True

            scan = _CellScan(
                spec=spec,
                candidate=candidate,
                cell_key=cell_key,
                cell_dir=cell_dir,
                record=record,
                fingerprint=fingerprint,
                validation_context=validation_context,
                state=state,
                eligible=eligible,
                next_eligible=next_eligible,
                needs_validation=needs_validation,
            )
            scans.append(scan)
            if needs_validation:
                pending_by_run[spec.run_id].append(scan)

    validation_fairness = ledger.setdefault(
        "validation_fairness", {"next_run_id": None}
    )
    if not isinstance(validation_fairness, dict):
        raise DispatcherError("dispatcher validation_fairness ledger field must be an object")
    demand = {spec.run_id: len(pending_by_run[spec.run_id]) for spec in specs}
    allocation = plan_validation_allocation(
        [spec.run_id for spec in specs],
        demand=demand,
        budget=validation_budget,
        next_run_id=validation_fairness.get("next_run_id"),
    )
    validation_fairness.update(
        {
            "next_run_id": allocation.next_run_id,
            "last_budget": validation_budget,
            "last_demand": demand,
            "last_allocations": allocation.allocations,
            "last_run_order": list(allocation.run_order),
            "last_used": sum(allocation.allocations.values()),
            "last_planned_at": now,
        }
    )

    # Execute in token order rather than grouping by run.  Even if a control job is
    # interrupted before its ledger write, it cannot repeatedly perform only run zero's
    # expensive validations.  Per-run pending queues remain in immutable manifest order.
    pending_offsets: dict[str, int] = defaultdict(int)
    for run_id in allocation.run_order:
        offset = pending_offsets[run_id]
        scan = pending_by_run[run_id][offset]
        pending_offsets[run_id] += 1
        error = _validate_pending_scan(
            scan,
            active=active,
            active_cost=active_cost,
            now=now,
            code_version=code_version,
            model_contract_path=model_contract_path,
            validation_retry_s=validation_retry_s,
        )
        if error is not None:
            errors.append(error)

    candidates: list[Candidate] = []
    count_builders: dict[str, dict[str, int]] = {
        spec.run_id: defaultdict(int) for spec in specs
    }
    for scan in scans:
        cell = scan.candidate.cell
        count_builders[scan.spec.run_id][scan.state] += 1
        scan.record.update(
            {
                "run_id": scan.spec.run_id,
                "cell_id": cell.cell_id,
                "source_index": scan.candidate.source_index,
                "model_size": cell.model_size,
                "serving_profile": scan.candidate.serving_profile,
                "fanout_cost": scan.candidate.fanout_cost,
                "completion_state": scan.state,
                "next_eligible_at": scan.next_eligible,
                "last_checked_at": now,
            }
        )
        ledger["cells"][scan.cell_key] = scan.record

        if scan.state in ELIGIBLE_STATES or (
            scan.state == CompletionState.RETRYABLE.value and scan.eligible
        ):
            candidates.append(scan.candidate)

    counts = {
        run_id: dict(sorted(state_counts.items()))
        for run_id, state_counts in count_builders.items()
    }
    return candidates, counts, dict(active_cost), errors


def _parse_capacity_overrides(
    values: Sequence[str], specs: Sequence[RunSpec]
) -> dict[ProfileKey, int]:
    """Parse repeatable ``RUN_ID:PROFILE=LIVE_SERVERS`` test/operator overrides."""
    by_run = {spec.run_id: spec for spec in specs}
    out: dict[ProfileKey, int] = {}
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[0]:
            raise DispatcherError(
                f"--server-capacity expects RUN_ID:PROFILE=N, got {value!r}"
            )
        left, raw_count = value.split("=", 1)
        run_id, profile = left.split(":", 1)
        if run_id not in by_run:
            raise DispatcherError(f"--server-capacity references unknown run {run_id!r}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise DispatcherError(f"invalid server capacity {value!r}") from exc
        key = (str(by_run[run_id].server_pool_root), profile)
        if key in out and out[key] != count:
            raise DispatcherError(f"conflicting capacity override for shared pool/profile {key}")
        out[key] = count
    return out


def _render_batch_sbatch(
    batch_manifest: Path,
    *,
    n_tasks: int,
    partition: str,
    qos: str | None = None,
    time_limit: str,
    memory: str,
    log_dir: Path,
    batch_tag: str,
    batch_id: str | None = None,
    batch_manifest_sha256: str = "0" * 64,
    control_state_dir: Path | None = None,
    qualification_execution: Mapping[str, str] | None = None,
) -> str:
    if not 1 <= n_tasks <= MAX_BATCH_DEFAULT:
        raise ValueError(f"microbatch size must be in [1, {MAX_BATCH_DEFAULT}]")
    if control_state_dir is not None and qualification_execution is not None:
        raise DispatcherError(
            "production control and qualification execution authorities are "
            "mutually exclusive"
        )
    production_execution: dict[str, str] | None = None
    if control_state_dir is not None:
        from slurm.schema5_control import production_cell_execution_from_state

        production_execution = production_cell_execution_from_state(control_state_dir)
    pinned_execution: Mapping[str, str] | None = (
        production_execution
        if production_execution is not None
        else qualification_execution
    )
    if pinned_execution is not None:
        template_path = Path(pinned_execution["batch_template"])
        runtime_setup = "\n".join(
            (
                "# Schema-5 cells never source common.sh or activate an ambient env.",
                "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV",
                "export PYTHONDONTWRITEBYTECODE=1",
                "export PYTHONNOUSERSITE=1",
                "export PYTHONSAFEPATH=1",
                "export HF_HUB_OFFLINE=1",
                "export TRANSFORMERS_OFFLINE=1",
                "export HF_DATASETS_OFFLINE=1",
                (
                    'export HF_HOME="'
                    + pinned_execution["hf_home"]
                    + '"'
                ),
                (
                    'export ASYS_RELEASE_WORKTREE="'
                    + pinned_execution["release_worktree"]
                    + '"'
                ),
            )
        )
        python_command = shlex.quote(pinned_execution["python"])
        dispatch_script = shlex.quote(pinned_execution["dispatcher_script"])
        runtime_arguments = (
            " \\\n  --expected-release-root "
            + shlex.quote(pinned_execution["release_worktree"])
            + " \\\n  --expected-harness-prefix "
            + shlex.quote(pinned_execution["harness_prefix"])
        )
    else:
        # Explicit legacy compatibility: old, non-control dispatchers still activate
        # their configured harness environment.  The production validator rejects this
        # branch, so a schema-5 array can never inherit it accidentally.
        template_path = ARRAY_TEMPLATE
        runtime_setup = "\n".join(
            (
                f'source "{REPO}/slurm/common.sh"',
                'mamba activate "$ASYS_HARNESS_ENV"',
                "export PYTHONDONTWRITEBYTECODE=1",
            )
        )
        python_command = "python"
        dispatch_script = shlex.quote(str(REPO / "slurm" / "dispatch_sweeps.py"))
        runtime_arguments = ""

    effective_qos = partition if qos is None else qos
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", effective_qos) is None:
        raise ValueError(f"invalid Slurm QOS {effective_qos!r}")
    text = template_path.read_text(encoding="utf-8")
    replacements = {
        "BATCH_TAG": batch_tag,
        "BATCH_ID": batch_id or batch_tag,
        "PARTITION": partition,
        "QOS": effective_qos,
        "CPUS": str(CELL_CPUS_DEFAULT),
        "MEM": memory,
        "TIME": time_limit,
        "LAST_INDEX": str(n_tasks - 1),
        "THROTTLE": str(n_tasks),
        "LOG_DIR": str(log_dir),
        "REPO": str(REPO),
        "BATCH_MANIFEST": str(batch_manifest),
        "BATCH_MANIFEST_SHA256": batch_manifest_sha256,
        "RUNTIME_SETUP": runtime_setup,
        "PYTHON": python_command,
        "DISPATCH_SCRIPT": dispatch_script,
        "RUNTIME_ARGUMENTS": runtime_arguments,
    }
    for name, value in replacements.items():
        text = text.replace("{" + name + "}", value)
    unresolved = [token for token in replacements if "{" + token + "}" in text]
    if unresolved:
        raise DispatcherError(f"unresolved batch template fields: {unresolved}")
    if control_state_dir is not None:
        from slurm.schema5_control import load_control, validate_production_batch_sbatch

        control = load_control(control_state_dir, verify_files=True)
        validate_production_batch_sbatch(control, payload=text, task_count=n_tasks)
    return text


def _write_batch(
    state_dir: Path,
    selected: Sequence[Candidate],
    *,
    partition: str,
    qos: str | None = None,
    time_limit: str,
    memory: str,
    now: float,
    control_state_dir: Path | None = None,
    qualification_execution: Mapping[str, str] | None = None,
) -> tuple[str, Path, Path, dict[str, Any]]:
    batch_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(now))}-{uuid.uuid4().hex[:10]}"
    batch_dir = state_dir / "batches"
    log_dir = state_dir / "logs"
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / f"batch-{batch_id}.json"
    sbatch_path = batch_dir / f"batch-{batch_id}.sbatch"
    batch = {
        "schema_version": 1,
        "batch_id": batch_id,
        "created_at": now,
        "tasks": [_task_from_candidate(candidate) for candidate in selected],
    }
    _atomic_write_json(manifest_path, batch)
    manifest_sha256 = _seal_dispatch_artifact(manifest_path)
    sbatch_text = _render_batch_sbatch(
        manifest_path,
        n_tasks=len(selected),
        partition=partition,
        qos=qos,
        time_limit=time_limit,
        memory=memory,
        log_dir=log_dir,
        batch_tag=batch_id[-10:],
        batch_id=batch_id,
        batch_manifest_sha256=manifest_sha256,
        control_state_dir=control_state_dir,
        qualification_execution=qualification_execution,
    )
    io.atomic_write_text(sbatch_path, sbatch_text)
    # These generation/intent-addressed files are the recovery authority on both sides
    # of sbatch.  Seal them before their hashes enter the durable ledger; subsequent
    # reconciliation refuses mode or byte drift.
    _seal_dispatch_artifact(sbatch_path)
    return batch_id, manifest_path, sbatch_path, batch


def _submit_sbatch(path: Path) -> str:
    if not path.name.startswith("batch-") or path.suffix != ".sbatch":
        raise DispatcherError(f"dispatcher sbatch path lacks an intent identity: {path}")
    batch_id = path.name[len("batch-") : -len(".sbatch")]
    if not batch_id or any(character in batch_id for character in "|;\n\r"):
        raise DispatcherError(f"unsafe dispatcher batch intent {batch_id!r}")
    proc = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--comment=asys-schema5-intent:{batch_id}",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    if proc.returncode != 0:
        raise DispatcherError(
            f"sbatch rejected {path.name} (rc={proc.returncode}): {proc.stderr.strip()[:500]}"
        )
    job_id = proc.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise DispatcherError(f"sbatch returned invalid job id {job_id!r} for {path}")
    return job_id


def _queue_successor(path: Path) -> str:
    """Queue exactly one afterany successor after the singleton lock is acquired."""
    current_job = os.environ.get("SLURM_JOB_ID")
    if not current_job:
        raise DispatcherError("--successor-sbatch requires SLURM_JOB_ID")
    proc = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{current_job}",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise DispatcherError(
            f"failed to queue dispatcher successor: {proc.stderr.strip()[:500]}"
        )
    successor = proc.stdout.strip().split(";", 1)[0]
    if not successor:
        raise DispatcherError("sbatch returned no dispatcher successor job id")
    return successor


def _record_submission(
    ledger: dict[str, Any],
    *,
    job_id: str,
    batch_id: str,
    manifest_path: Path,
    sbatch_path: Path,
    batch: Mapping[str, Any],
    now: float,
) -> None:
    tasks = list(batch["tasks"])
    intent = ledger.get("intents", {}).get(batch_id)
    if not isinstance(intent, dict):
        raise DispatcherError(f"submission {batch_id} lacks its durable intent")
    ledger["jobs"][job_id] = {
        "job_id": job_id,
        "batch_id": batch_id,
        "batch_manifest": str(manifest_path),
        "batch_manifest_sha256": str(intent.get("batch_manifest_sha256")),
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": str(intent.get("sbatch_sha256")),
        "submitted_at": now,
        "last_seen_at": now,
        "state": "submitted",
        "task_count": len(tasks),
        "tasks": tasks,
    }
    if batch_id in ledger.get("intents", {}):
        ledger["intents"][batch_id].update(
            {"state": "submitted", "job_id": job_id, "submitted_at": now}
        )
        _commit_intent_fairness(ledger, batch_id)
    for task in tasks:
        cell_key = _key((str(task["run_id"]), str(task["cell_id"])))
        record = dict(ledger["cells"].get(cell_key, {}))
        record["submission_attempts"] = int(record.get("submission_attempts", 0)) + 1
        record["last_job_id"] = job_id
        record["last_submitted_at"] = now
        ledger["cells"][cell_key] = record


def _commit_intent_fairness(ledger: dict[str, Any], batch_id: str) -> None:
    """Apply one accepted intent's post-admission fairness state exactly once."""

    intent = ledger.get("intents", {}).get(batch_id)
    if not isinstance(intent, dict) or intent.get("fairness_committed") is True:
        return
    fairness_after = intent.get("fairness_after")
    if not isinstance(fairness_after, dict):
        raise DispatcherError(f"accepted intent {batch_id} lacks durable fairness state")
    cursor = fairness_after.get("cursor")
    deficits = fairness_after.get("deficits")
    if not isinstance(cursor, int) or not isinstance(deficits, dict):
        raise DispatcherError(f"accepted intent {batch_id} has invalid fairness state")
    ledger["fairness"] = copy.deepcopy(fairness_after)
    intent["fairness_committed"] = True


def _profile_headroom(
    capacity: CapacitySnapshot,
    active_cost: Mapping[ProfileKey, int],
    *,
    fanout_slots_per_server: int,
) -> dict[ProfileKey, int]:
    if fanout_slots_per_server < 1:
        raise ValueError("fanout_slots_per_server must be >= 1")
    return {
        key: max(0, replicas * fanout_slots_per_server - int(active_cost.get(key, 0)))
        for key, replicas in capacity.live_servers.items()
    }


DISPATCHER_SCHEDULER_AMBIGUITY_ALERT = "dispatcher:scheduler-ambiguity"


def _dispatcher_safety_findings(
    *,
    state_counts: Mapping[str, Mapping[str, int]],
    validation_errors: Sequence[str],
    unmappable_jobs: Sequence[str],
) -> list[dict[str, str]]:
    """Translate admission-visible integrity failures into durable alert facts."""

    totals: Counter[str] = Counter()
    for run_counts in state_counts.values():
        if not isinstance(run_counts, Mapping):
            continue
        for state, count in run_counts.items():
            try:
                totals[str(state)] += int(count)
            except (TypeError, ValueError):
                # The ledger/schema validator protects persisted structure.  A corrupt
                # per-run count is still an integrity finding and cannot be ignored.
                totals["validation_error"] += 1
    corrupt = totals[CompletionState.CORRUPT.value]
    validation = max(totals["validation_error"], len(validation_errors))
    findings: list[dict[str, str]] = []
    if corrupt or validation:
        examples = "; ".join(str(item) for item in validation_errors[:3])
        suffix = f"; examples: {examples}" if examples else ""
        findings.append(
            {
                "dedupe_key": "monitor:corrupt",
                # Match the semantic monitor's ownership identity so the same durable
                # incident can be refreshed and later resolved without a kind clash.
                "kind": "corrupt-artifacts",
                "message": (
                    "dispatcher observed admission-visible corrupt/validation-error "
                    f"cells: corrupt={corrupt}, validation_error={validation}{suffix}"
                ),
            }
        )
    permanent = totals[CompletionState.PERMANENT.value]
    if permanent:
        findings.append(
            {
                "dedupe_key": "monitor:permanent",
                "kind": "permanent-failures",
                "message": (
                    "dispatcher observed admission-visible permanent cells: "
                    f"permanent={permanent}"
                ),
            }
        )
    if unmappable_jobs:
        findings.append(
            {
                "dedupe_key": DISPATCHER_SCHEDULER_AMBIGUITY_ALERT,
                "kind": "dispatcher-scheduler-ambiguity",
                "message": (
                    "active or historical schema-5 cell jobs could not be mapped to "
                    "one immutable admission intent: "
                    + "; ".join(str(item) for item in unmappable_jobs[:10])
                ),
            }
        )
    return findings


def _persist_dispatcher_safety_findings(
    state_dir: Path,
    *,
    findings: Sequence[Mapping[str, str]],
    now: float,
) -> None:
    """Fence admission transactionally before the dispatcher can plan an array.

    Critical ``record_alert`` owns the same cross-node admission boundary as sbatch
    submission.  Any persistence or delivery-path exception propagates so the poll
    fails closed.  Artifact alerts are resolved only by monitor-owned semantic/health
    reconciliation; this dispatcher exclusively resolves its scheduler ambiguity.
    """

    from slurm.schema5_control import record_alert, resolve_alert

    keys = {str(finding["dedupe_key"]) for finding in findings}
    for finding in findings:
        record_alert(
            state_dir,
            kind=str(finding["kind"]),
            severity="critical",
            message=str(finding["message"]),
            dedupe_key=str(finding["dedupe_key"]),
            send_email=True,
            now=now,
        )
    if DISPATCHER_SCHEDULER_AMBIGUITY_ALERT not in keys:
        resolve_alert(
            state_dir,
            dedupe_key=DISPATCHER_SCHEDULER_AMBIGUITY_ALERT,
            now=now,
        )


def _dispatch_poll(
    args: argparse.Namespace,
    specs: Sequence[RunSpec],
    ledger: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    now = time.time()
    production_contract: dict[str, Any] | None = None
    production_fleet: FrozenFleetContract | None = None
    production_code_version: str | None = None
    production_model_contract_path: str | None = None
    production_client_capacity: dict[str, Any] | None = None
    production_partition_usage: dict[str, Any] | None = None
    control: dict[str, Any] | None = None
    production_capacity_contract: (
        protected_capacity.ProtectedCapacityContract | None
    ) = None
    qualification_capacity_contract: (
        protected_capacity.ProtectedCapacityContract | None
    ) = None
    qualification_client_placement = None
    qualification_client_capacity: dict[str, Any] | None = None
    qualification_partition_usage: dict[str, Any] | None = None
    qualification_execution_authority: (
        QualificationExecutionAuthority | None
    ) = None
    qualification_execution_binding: dict[str, Any] | None = None
    protected_task_headroom: int | None = None
    control_state_dir = getattr(args, "control_state_dir", None)
    if control_state_dir is not None:
        from slurm.schema5_control import (
            admission_contract_from_state,
            effective_fleet_contract_binding,
            load_control,
            production_environment_from_state,
        )

        production_contract = admission_contract_from_state(control_state_dir)
        control = load_control(control_state_dir, verify_files=True)
        immutable = control["immutable"]
        protected_ref = production_contract["protected_capacity"]
        production_capacity_contract = protected_capacity.load_contract(
            protected_ref["path"],
            expected_release_git_commit=str(immutable["git_commit"]),
            expected_marker_id=str(protected_ref["marker_id"]),
            expected_sha256=str(protected_ref["sha256"]),
        )
        fleet_binding = effective_fleet_contract_binding(
            control, verify_files=True
        )
        production_code_version = (
            str(immutable["git_commit"])
            + "+source."
            + str(immutable["source_tree_sha256"])[:16]
        )
        production_model_contract_path = str(
            Path(str(immutable["model_contract_path"])).resolve()
        )
        expected_pool_root = Path(immutable["server_pool_root"]).resolve()
        mismatched_pools = {
            spec.run_id: str(spec.server_pool_root)
            for spec in specs
            if spec.server_pool_root.resolve() != expected_pool_root
        }
        if mismatched_pools:
            raise DispatcherError(
                "schema-5 runs must use the one control-pinned canonical server pool: "
                f"expected {expected_pool_root}, observed {mismatched_pools}"
            )
        try:
            model_contracts = load_model_contracts(
                immutable["model_contract_path"],
                expected_sha256=immutable["model_contract_sha256"],
            )
            production_fleet = load_fleet_contract(
                fleet_binding["path"],
                model_contracts=model_contracts,
                expected_sha256=fleet_binding["sha256"],
                allow_capacity_layout=True,
            )
            production_fleet.verify_pool_root(expected_pool_root)
        except (FleetContractError, ModelContractError, OSError, ValueError) as exc:
            raise DispatcherError(
                f"schema-5 immutable fleet contract failed closed: {exc}"
            ) from exc
        required_cli = {
            "qos_limit": args.qos_limit,
            "reserve": args.reserve,
            "max_batch": args.max_batch,
            "cell_cpus": CELL_CPUS_DEFAULT,
            "cell_memory": args.cell_mem,
            "effective_cell_time": args.cell_time,
        }
        mismatches = {
            key: (production_contract.get(key), observed)
            for key, observed in required_cli.items()
            if production_contract.get(key) != observed
        }
        if mismatches:
            raise DispatcherError(
                f"dispatcher CLI weakens or drifts from schema-5 control: {mismatches}"
            )
        specs = tuple(
            replace(
                spec,
                runtime_environment=tuple(
                    sorted(
                        production_environment_from_state(
                            control_state_dir, run_id=spec.run_id
                        ).items()
                    )
                ),
            )
            for spec in specs
        )
    qualification_values = {
        "path": getattr(args, "protected_capacity_marker", None),
        "sha256": getattr(args, "protected_capacity_marker_sha256", None),
        "marker_id": getattr(args, "protected_capacity_marker_id", None),
        "release_git_commit": getattr(
            args, "protected_capacity_release_git_commit", None
        ),
    }
    qualification_execution_path = getattr(
        args, "qualification_execution_authority", None
    )
    has_qualification_capacity = any(qualification_values.values())
    authoritative_runs = sorted(
        {
            spec.run_id
            for spec in specs
            if spec.run_id in AUTHORITATIVE_SCHEMA5_RUN_IDS
        }
    )
    has_complete_qualification_authority = bool(
        qualification_execution_path
        and all(qualification_values.values())
    )
    if (
        authoritative_runs
        and control_state_dir is None
        and not has_complete_qualification_authority
    ):
        raise DispatcherError(
            "authoritative schema-5 runs require --control-state-dir or the "
            "complete isolated qualification authority before admission: "
            f"{authoritative_runs}"
        )
    if bool(qualification_execution_path) != has_qualification_capacity:
        raise DispatcherError(
            "qualification execution and protected-capacity authorities must "
            "be supplied together"
        )
    if has_qualification_capacity:
        if production_contract is not None:
            raise DispatcherError(
                "qualification authorities cannot be combined with production "
                "--control-state-dir"
            )
        missing = [
            field for field, value in qualification_values.items() if not value
        ]
        if missing:
            raise DispatcherError(
                "qualification protected-capacity authority is all-or-none; "
                f"missing {missing}"
            )
        qualification_qos = getattr(args, "cell_qos", None)
        if not qualification_qos:
            raise DispatcherError(
                "protected qualification requires explicit --cell-qos"
            )
        if args.qos_limit - args.reserve != 384:
            raise DispatcherError(
                "protected qualification must reserve 64 of 448 jobs, leaving "
                "exactly 384 scientific-client slots"
            )
        try:
            qualification_capacity_contract = protected_capacity.load_contract(
                qualification_values["path"],
                expected_release_git_commit=str(
                    qualification_values["release_git_commit"]
                ),
                expected_marker_id=str(qualification_values["marker_id"]),
                expected_sha256=str(qualification_values["sha256"]),
            )
            qualification_client_placement = protected_capacity.authorize_client(
                qualification_capacity_contract,
                partition=str(args.cell_partition),
                qos=str(qualification_qos),
                required_slots=384,
                required_reserve_jobs=64,
            )
            protected_capacity.verify_live_placements(
                qualification_capacity_contract,
                role="client",
                placements=[
                    (str(args.cell_partition), str(qualification_qos))
                ],
                required_time_limits_seconds={
                    str(args.cell_partition): (
                        scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                    )
                },
            )
            qualification_client_capacity = (
                protected_capacity.capture_live_client_capacity(
                    qualification_capacity_contract,
                    partition=str(args.cell_partition),
                    qos=str(qualification_qos),
                    required_time_limit_seconds=(
                        scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                    ),
                    captured_timestamp=now,
                )
            )
            qualification_execution_authority = (
                load_qualification_execution_authority(
                    str(qualification_execution_path)
                )
            )
        except (
            DispatcherError,
            scheduler_safety.SchedulerSafetyError,
            protected_capacity.ProtectedCapacityError,
        ) as exc:
            raise DispatcherError(
                f"protected qualification placement failed closed: {exc}"
            ) from exc
        if len(specs) != 1:
            raise DispatcherError(
                "qualification execution authority permits exactly one isolated run"
            )
        qualification_spec = specs[0]
        authority_payload = qualification_execution_authority.payload
        protected_binding = authority_payload["protected_capacity"]
        authority_mismatches = {
            field: (expected, observed)
            for field, expected, observed in (
                (
                    "run_id",
                    qualification_spec.run_id,
                    authority_payload["run_id"],
                ),
                (
                    "run_root",
                    str(qualification_spec.run_root.resolve()),
                    str(Path(str(authority_payload["run_root"])).resolve()),
                ),
                (
                    "release_git_commit",
                    str(qualification_values["release_git_commit"]),
                    authority_payload["release_git_commit"],
                ),
                (
                    "protected_capacity_path",
                    str(qualification_capacity_contract.path),
                    protected_binding["path"],
                ),
                (
                    "protected_capacity_sha256",
                    qualification_capacity_contract.sha256,
                    protected_binding["sha256"],
                ),
                (
                    "protected_capacity_marker_id",
                    qualification_capacity_contract.marker_id,
                    protected_binding["marker_id"],
                ),
            )
            if expected != observed
        }
        if authority_mismatches:
            raise DispatcherError(
                "qualification execution authority differs from its isolated "
                f"run/capacity pins: {authority_mismatches}"
            )
        specs = (
            replace(
                qualification_spec,
                runtime_environment=tuple(
                    sorted(
                        qualification_execution_authority.runtime_environment.items()
                    )
                ),
            ),
        )
        qualification_execution_binding = (
            _qualification_execution_authority_binding(
                qualification_execution_authority
            )
        )
    for spec in specs:
        if spec.question_catalog is None:
            raise DispatcherError(
                f"run {spec.run_id!r} has no verified frozen benchmark Question contract"
            )
        try:
            spec.question_catalog.verify_unchanged()
        except BenchmarkContractError as exc:
            raise DispatcherError(
                f"run {spec.run_id!r} benchmark Question contract verification failed: {exc}"
            ) from exc
    work = copy.deepcopy(ledger)
    if qualification_capacity_contract is not None:
        authority = {
            "path": str(qualification_capacity_contract.path),
            "sha256": qualification_capacity_contract.sha256,
            "marker_id": qualification_capacity_contract.marker_id,
            "release_git_commit": (
                qualification_capacity_contract.release_git_commit
            ),
            "partition": str(args.cell_partition),
            "qos": str(args.cell_qos),
            "authorized_cell_slots": 384,
            "reserve_jobs": 64,
        }
        existing_authority = work.get("protected_capacity_authority")
        if existing_authority is not None and existing_authority != authority:
            raise DispatcherError(
                "dispatcher ledger is bound to a different protected-capacity "
                "qualification authority"
            )
        work["protected_capacity_authority"] = authority
        assert qualification_execution_binding is not None
        existing_execution_authority = work.get(
            "qualification_execution_authority"
        )
        if (
            existing_execution_authority is not None
            and existing_execution_authority
            != qualification_execution_binding
        ):
            raise DispatcherError(
                "dispatcher ledger is bound to a different qualification "
                "execution authority"
            )
        work["qualification_execution_authority"] = copy.deepcopy(
            qualification_execution_binding
        )
    if not dry_run and "throughput_observation_started_at" not in work:
        # This is the post-remediation throughput epoch, distinct from ``created_at``
        # on a preserved ledger that may predate a long pause.  It is established only
        # by the first live poll and survives every coordinator/control-job restart.
        work["throughput_observation_started_at"] = now
    _register_run_pins(work, specs)
    work["poll_number"] = int(work.get("poll_number", 0)) + 1
    poll_number = work["poll_number"]

    scheduler_reconcile_warnings: list[str] = []
    scheduler_reconcile_errors: list[str] = []
    protected_admission = (
        production_contract is not None
        or qualification_capacity_contract is not None
    )
    if protected_admission:
        if args.assume_total_jobs is not None or args.assume_cell_jobs is not None:
            raise DispatcherError(
                "protected schema-5 admission cannot replace scheduler authority "
                "with assumed counts"
            )
        if production_contract is not None:
            scheduler_user = os.environ.get("USER", "")
            client_contract = production_contract.get("client_capacity")
            if not isinstance(client_contract, Mapping):
                raise DispatcherError(
                    "schema-5 admission contract lacks client-capacity authority"
                )
            try:
                assert production_capacity_contract is not None
                production_client_capacity = (
                    protected_capacity.capture_live_client_capacity(
                        production_capacity_contract,
                        partition=str(client_contract["partition"]),
                        qos=str(client_contract["qos"]),
                        required_time_limit_seconds=(
                            scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                        ),
                        captured_timestamp=now,
                    )
                )
            except (
                scheduler_safety.SchedulerSafetyError,
                protected_capacity.ProtectedCapacityError,
            ) as exc:
                raise DispatcherError(
                    f"schema-5 client QOS/TRES authority failed closed: {exc}"
                ) from exc
            occupancy_partition = str(client_contract["partition"])
        else:
            scheduler_user = os.environ.get("USER", "")
            occupancy_partition = str(args.cell_partition)
        try:
            initial_occupancy = _capture_stable_admission_occupancy(
                user=scheduler_user,
                partition=occupancy_partition,
            )
        except (
            DispatcherError,
            scheduler_safety.SchedulerSafetyError,
        ) as exc:
            raise DispatcherError(
                f"schema-5 stable scheduler occupancy failed closed: {exc}"
            ) from exc
        scheduler_snapshot = initial_occupancy.scheduler_snapshot
        rows = list(initial_occupancy.rows)
        if production_contract is not None:
            production_partition_usage = initial_occupancy.usage
        else:
            qualification_partition_usage = initial_occupancy.usage
        scheduler_reconcile_warnings, scheduler_reconcile_errors = (
            _reconcile_schema5_intents(
                work,
                scheduler_snapshot=scheduler_snapshot,
                now=now,
            )
        )
    else:
        rows = (
            []
            if args.assume_total_jobs is not None and args.assume_cell_jobs is not None
            else _query_squeue()
        )
    total_jobs = len(rows) if args.assume_total_jobs is None else args.assume_total_jobs
    trusted_cell_bindings: dict[str, dict[str, str]] = {}
    active, active_task_load, join_warnings, unmappable_jobs = _active_cells(
        rows,
        work,
        specs,
        now=now,
        schema5_strict=protected_admission,
        trusted_cell_bindings=trusted_cell_bindings,
    )
    active_cell_jobs = (
        len(trusted_cell_bindings)
        if args.assume_cell_jobs is None
        else args.assume_cell_jobs
    )
    join_warnings = scheduler_reconcile_warnings + join_warnings
    unmappable_jobs = scheduler_reconcile_errors + unmappable_jobs
    invisible_reservations = _invisible_reservation_count(
        work,
        now=now,
    )
    total_jobs += invisible_reservations
    active_cell_jobs += invisible_reservations

    profile_keys = {
        (str(spec.server_pool_root), serving_profile_for_cell(cell).registry_key)
        for spec in specs
        for cell in spec.manifest.cells
    }
    capacity = discover_capacity(
        profile_keys,
        probe=args.probe_servers,
        probe_timeout=args.probe_timeout,
        overrides=_parse_capacity_overrides(args.server_capacity, specs),
        active_job_ids={
            identifier
            for row in rows
            if row.state.upper() in {"RUNNING", "COMPLETING", "CONFIGURING"}
            for identifier in (row.array_job_id, row.job_id)
        },
        frozen_fleet=production_fleet,
    )
    # ``io.git_commit`` intentionally inspects the source tree containing the imported
    # package.  Under the non-editable schema-5 harness that is site-packages and has no
    # Git checkout, so production derives the identical frozen identity from control.
    code_version = (
        production_code_version
        if production_code_version is not None
        else io.git_commit()
    )
    candidates, state_counts, active_cost, validation_errors = _scan_candidates(
        specs,
        active=active,
        active_task_load=active_task_load,
        capacity=capacity,
        ledger=work,
        now=now,
        code_version=code_version,
        model_contract_path=production_model_contract_path,
        validation_budget=args.validation_budget,
    )
    trusted_nonclient_bindings: dict[str, dict[str, str]] = {}
    if protected_admission:
        if production_fleet is not None:
            expected_fleet_sha256 = production_fleet.sha256
        else:
            qualification_fleet_hashes = {
                dict(spec.runtime_environment).get(
                    "ASYS_FLEET_CONTRACT_SHA256", ""
                )
                for spec in specs
            }
            if (
                len(qualification_fleet_hashes) != 1
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    next(iter(qualification_fleet_hashes), ""),
                )
                is None
            ):
                raise DispatcherError(
                    "qualification server classification lacks one exact fleet hash"
                )
            expected_fleet_sha256 = next(iter(qualification_fleet_hashes))
        trusted_nonclient_bindings = _trusted_server_scheduler_bindings(
            specs,
            expected_fleet_sha256=expected_fleet_sha256,
            frozen_fleet=production_fleet,
            scheduler_rows=rows,
        )

        def capture_protected_boundary(
            *,
            partition: str,
            cpu_limit: int,
            memory_limit_mib: int,
            max_submit_jobs: int,
            reserve_jobs: int,
            exclude_intent_ids: frozenset[str] = frozenset(),
        ) -> tuple[
            StableAdmissionOccupancy,
            int,
            int,
            dict[str, dict[str, str]],
            dict[str, dict[str, str]],
        ]:
            """Recompute one complete protected headroom decision from stable truth."""

            stable = _capture_stable_admission_occupancy(
                user=os.environ.get("USER", ""),
                partition=partition,
            )
            boundary_work = copy.deepcopy(work)
            _reconcile_warnings, reconcile_errors = _reconcile_schema5_intents(
                boundary_work,
                scheduler_snapshot=stable.scheduler_snapshot,
                now=time.time(),
            )
            fresh_client_bindings: dict[str, dict[str, str]] = {}
            (
                _fresh_active,
                _fresh_load,
                _fresh_warnings,
                fresh_unmappable,
            ) = _active_cells(
                stable.rows,
                boundary_work,
                specs,
                now=time.time(),
                schema5_strict=True,
                trusted_cell_bindings=fresh_client_bindings,
            )
            boundary_errors = list(reconcile_errors) + list(fresh_unmappable)
            if boundary_errors:
                raise DispatcherError(
                    "stable admission occupancy contains unmappable scheduler "
                    "state: "
                    + "; ".join(boundary_errors[:10])
                )
            fresh_nonclient_bindings = _trusted_server_scheduler_bindings(
                specs,
                expected_fleet_sha256=expected_fleet_sha256,
                frozen_fleet=production_fleet,
                scheduler_rows=stable.rows,
            )
            boundary_now = time.time()
            boundary_invisible = _invisible_reservation_count(
                boundary_work,
                now=boundary_now,
                exclude_intent_ids=exclude_intent_ids,
            )
            headroom = scheduler_safety.client_task_headroom(
                stable.usage,
                cell_cpus=CELL_CPUS_DEFAULT,
                cell_memory_mib=4 * 1024,
                invisible_reserved_tasks=boundary_invisible,
                cpu_limit=cpu_limit,
                memory_limit_mib=memory_limit_mib,
                max_submit_jobs=max_submit_jobs,
                reserve_jobs=reserve_jobs,
                cell_ceiling=384,
                absolute_job_ceiling=448,
                live_user_job_elements=len(stable.rows),
                trusted_client_jobs=fresh_client_bindings,
                trusted_nonclient_jobs=fresh_nonclient_bindings,
            )
            active_cells_at_boundary = (
                len(fresh_client_bindings) + boundary_invisible
            )
            return (
                stable,
                headroom,
                active_cells_at_boundary,
                fresh_client_bindings,
                fresh_nonclient_bindings,
            )
    safety_findings = _dispatcher_safety_findings(
        state_counts=state_counts,
        validation_errors=validation_errors,
        unmappable_jobs=unmappable_jobs,
    )
    if production_contract is not None and not dry_run:
        # This must precede both WDRR planning and the external sbatch boundary.
        # Persisting a critical alert transactionally drops the effective ceiling to
        # zero; then refresh the contract so even unrelated pre-existing holds remain
        # visible to this poll.
        _persist_dispatcher_safety_findings(
            control_state_dir,
            findings=safety_findings,
            now=now,
        )
        from slurm.schema5_control import admission_contract_from_state

        production_contract = admission_contract_from_state(control_state_dir)
    slots = available_cell_slots(
        total_jobs=total_jobs,
        active_cell_jobs=active_cell_jobs,
        qos_limit=args.qos_limit,
        reserve=args.reserve,
        max_batch=args.max_batch,
    )
    if production_contract is not None:
        slots = min(
            slots,
            max(
                0,
                int(production_contract["current_ceiling"]) - active_cell_jobs,
            ),
        )
        assert production_client_capacity is not None
        assert production_partition_usage is not None
        client_contract = production_contract["client_capacity"]
        assert production_capacity_contract is not None
        protected_capacity.validate_live_client_capacity_evidence(
            production_capacity_contract,
            production_client_capacity,
            partition=str(client_contract["partition"]),
            qos=str(client_contract["qos"]),
            required_time_limit_seconds=(
                scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
            ),
        )
        protected_task_headroom = scheduler_safety.client_task_headroom(
            production_partition_usage,
            cell_cpus=CELL_CPUS_DEFAULT,
            cell_memory_mib=4 * 1024,
            invisible_reserved_tasks=invisible_reservations,
            cpu_limit=int(client_contract["cpu_limit"]),
            memory_limit_mib=int(client_contract["memory_limit_mib"]),
            max_submit_jobs=int(client_contract["max_submit_jobs"]),
            reserve_jobs=int(client_contract["reserve_jobs"]),
            cell_ceiling=384,
            absolute_job_ceiling=448,
            live_user_job_elements=total_jobs - invisible_reservations,
            trusted_client_jobs=trusted_cell_bindings,
            trusted_nonclient_jobs=trusted_nonclient_bindings,
        )
        slots = min(slots, protected_task_headroom)
    elif qualification_capacity_contract is not None:
        assert qualification_client_placement is not None
        assert qualification_client_capacity is not None
        assert qualification_partition_usage is not None
        protected_task_headroom = scheduler_safety.client_task_headroom(
            qualification_partition_usage,
            cell_cpus=CELL_CPUS_DEFAULT,
            cell_memory_mib=4 * 1024,
            invisible_reserved_tasks=invisible_reservations,
            cpu_limit=int(
                qualification_client_placement.capacity["cpus"]
            ),
            memory_limit_mib=int(
                qualification_client_placement.capacity["memory_mib"]
            ),
            max_submit_jobs=int(
                qualification_client_placement.capacity[
                    "submit_headroom"
                ]
            ),
            reserve_jobs=64,
            cell_ceiling=384,
            absolute_job_ceiling=448,
            live_user_job_elements=total_jobs - invisible_reservations,
            trusted_client_jobs=trusted_cell_bindings,
            trusted_nonclient_jobs=trusted_nonclient_bindings,
        )
        slots = min(slots, protected_task_headroom)
    if safety_findings:
        # Production findings are already durable above.  Preserve fail-closed
        # behavior in dry-run/non-production simulations without mutating control.
        slots = 0
    if unmappable_jobs:
        # Fail closed: a cell job that cannot be joined to an immutable manifest may be
        # mutating any candidate.  Advisory locking protects new runners, but legacy
        # workers predate it, so global admission must stop until ownership is resolved.
        slots = 0
        join_warnings.append(
            "global admission disabled because active cell jobs could not be mapped: "
            + "; ".join(unmappable_jobs[:10])
        )
    fairness = work["fairness"]
    admission = plan_admission(
        candidates,
        deficits=fairness.get("deficits", {}),
        cursor=int(fairness.get("cursor", 0)),
        max_tasks=slots,
        live_servers=capacity.live_servers,
        profile_headroom=_profile_headroom(
            capacity, active_cost, fanout_slots_per_server=args.fanout_slots_per_server
        ),
        run_weights={spec.run_id: spec.weight for spec in specs},
    )

    selected = admission.selected
    submission: dict[str, Any] | None = None
    submission_error: str | None = None
    submitted = False
    if selected and not dry_run:
        # The production pause path owns this same cross-node lock before taking its
        # scheduler snapshot.  Re-read the fail-closed contract while holding it, then
        # retain ownership through the durable intent and external sbatch boundary.
        # Thus pause can neither miss a just-admitted array nor race a stale plan.
        guard = nullcontext()
        if control_state_dir is not None:
            from slurm.schema5_control import admission_boundary_lock

            guard = admission_boundary_lock(control_state_dir)
        with guard:
            if production_contract is not None:
                from slurm.schema5_control import admission_contract_from_state

                fresh_contract = admission_contract_from_state(control_state_dir)
                if fresh_contract != production_contract:
                    raise DispatcherError(
                        "schema-5 admission contract changed after planning; "
                        "discarding the stale plan before rendering or sbatch"
                    )
                # The cross-node admission lock serializes schema-5 dispatchers, but it
                # cannot prevent the user from submitting an unrelated job to the
                # currently authorized client partition.
                # Re-read scheduler-authoritative QOS and TRES immediately before
                # crossing sbatch; a stale resource plan is discarded, never held.
                try:
                    client_contract = fresh_contract["client_capacity"]
                    assert production_capacity_contract is not None
                    fresh_client_capacity = (
                        protected_capacity.capture_live_client_capacity(
                            production_capacity_contract,
                            partition=str(client_contract["partition"]),
                            qos=str(client_contract["qos"]),
                            required_time_limit_seconds=(
                                scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                            ),
                        )
                    )
                    (
                        fresh_occupancy,
                        fresh_resource_slots,
                        fresh_active_cell_jobs,
                        _fresh_client_bindings,
                        _fresh_nonclient_bindings,
                    ) = capture_protected_boundary(
                        partition=str(client_contract["partition"]),
                        cpu_limit=int(client_contract["cpu_limit"]),
                        memory_limit_mib=int(
                            client_contract["memory_limit_mib"]
                        ),
                        max_submit_jobs=int(
                            client_contract["max_submit_jobs"]
                        ),
                        reserve_jobs=int(client_contract["reserve_jobs"]),
                    )
                except (
                    DispatcherError,
                    scheduler_safety.SchedulerSafetyError,
                    protected_capacity.ProtectedCapacityError,
                ) as exc:
                    raise DispatcherError(
                        "schema-5 fresh client QOS/TRES authority failed closed "
                        f"before sbatch: {exc}"
                    ) from exc
                if fresh_active_cell_jobs + len(selected) > int(
                    fresh_contract["current_ceiling"]
                ):
                    raise DispatcherError(
                        "schema-5 selected tasks exceed the freshly fenced cell ceiling"
                    )
                if len(selected) > fresh_resource_slots:
                    raise DispatcherError(
                        "schema-5 selected tasks exceed freshly observed authorized "
                        f"{client_contract['partition']} "
                        f"CPU/memory/MaxSubmitJobs headroom: selected={len(selected)}, "
                        f"headroom={fresh_resource_slots}"
                    )
                production_partition_usage = fresh_occupancy.usage
                production_client_capacity = fresh_client_capacity
                protected_task_headroom = fresh_resource_slots
            elif qualification_capacity_contract is not None:
                assert qualification_client_placement is not None
                try:
                    fresh_qualification_capacity = (
                        protected_capacity.capture_live_client_capacity(
                            qualification_capacity_contract,
                            partition=str(args.cell_partition),
                            qos=str(args.cell_qos),
                            required_time_limit_seconds=(
                                scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                            ),
                        )
                    )
                    (
                        fresh_qualification_occupancy,
                        fresh_resource_slots,
                        _fresh_active_cell_jobs,
                        _fresh_client_bindings,
                        _fresh_nonclient_bindings,
                    ) = capture_protected_boundary(
                        partition=str(args.cell_partition),
                        cpu_limit=int(
                            qualification_client_placement.capacity["cpus"]
                        ),
                        memory_limit_mib=int(
                            qualification_client_placement.capacity[
                                "memory_mib"
                            ]
                        ),
                        max_submit_jobs=int(
                            qualification_client_placement.capacity[
                                "submit_headroom"
                            ]
                        ),
                        reserve_jobs=64,
                    )
                except (
                    DispatcherError,
                    scheduler_safety.SchedulerSafetyError,
                    protected_capacity.ProtectedCapacityError,
                ) as exc:
                    raise DispatcherError(
                        "protected qualification fresh client QOS/TRES "
                        f"authority failed closed before sbatch: {exc}"
                    ) from exc
                if len(selected) > fresh_resource_slots:
                    raise DispatcherError(
                        "protected qualification selected tasks exceed freshly "
                        "observed CPU/memory/MaxSubmitJobs headroom: "
                        f"selected={len(selected)}, headroom={fresh_resource_slots}"
                    )
                qualification_client_capacity = fresh_qualification_capacity
                qualification_partition_usage = (
                    fresh_qualification_occupancy.usage
                )
                protected_task_headroom = fresh_resource_slots

            batch_id, manifest_path, sbatch_path, batch = _write_batch(
                args.state_dir,
                selected,
                partition=(
                    str(production_contract["client_capacity"]["partition"])
                    if production_contract is not None
                    else args.cell_partition
                ),
                qos=(
                    str(production_contract["client_capacity"]["qos"])
                    if production_contract is not None
                    else (getattr(args, "cell_qos", None) or args.cell_partition)
                ),
                time_limit=args.cell_time,
                memory=args.cell_mem,
                now=now,
                control_state_dir=control_state_dir,
                qualification_execution=(
                    qualification_execution_authority.execution
                    if qualification_execution_authority is not None
                    else None
                ),
            )
            work["intents"][batch_id] = {
                "state": "prepared",
                "created_at": now,
                "batch_manifest": str(manifest_path),
                "batch_manifest_sha256": _sealed_artifact_sha256(manifest_path),
                "sbatch_path": str(sbatch_path),
                "sbatch_sha256": _sealed_artifact_sha256(sbatch_path),
                "tasks": list(batch["tasks"]),
                "fairness_after": {
                    "cursor": admission.cursor,
                    "deficits": admission.deficits,
                },
                "fairness_committed": False,
                "protected_capacity_authority": (
                    copy.deepcopy(work.get("protected_capacity_authority"))
                    if qualification_capacity_contract is not None
                    else None
                ),
                "qualification_execution_authority": (
                    copy.deepcopy(
                        work.get("qualification_execution_authority")
                    )
                    if qualification_execution_authority is not None
                    else None
                ),
            }
            # Persist the exact task reservation before crossing the external sbatch
            # boundary.  If the process dies after acceptance, the next coordinator
            # joins the array through the intent token and exact immutable sbatch path.
            _atomic_write_json(args.ledger_path, work)
            if production_contract is not None:
                try:
                    protected_ref = production_contract["protected_capacity"]
                    live_contract = protected_capacity.load_contract(
                        protected_ref["path"],
                        expected_release_git_commit=str(
                            control["immutable"]["git_commit"]
                        ),
                        expected_marker_id=str(protected_ref["marker_id"]),
                        expected_sha256=str(protected_ref["sha256"]),
                    )
                    client_contract = production_contract["client_capacity"]
                    protected_capacity.authorize_client(
                        live_contract,
                        partition=str(client_contract["partition"]),
                        qos=str(client_contract["qos"]),
                        required_slots=int(
                            production_contract["configured_ceiling"]
                        ),
                        required_reserve_jobs=int(
                            production_contract["reserve"]
                        ),
                    )
                    fresh_production_capacity = (
                        protected_capacity.capture_live_client_capacity(
                            live_contract,
                            partition=str(client_contract["partition"]),
                            qos=str(client_contract["qos"]),
                            required_time_limit_seconds=(
                                scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                            ),
                        )
                    )
                    (
                        fresh_production_occupancy,
                        fresh_headroom,
                        fresh_active_cell_jobs,
                        _fresh_client_bindings,
                        _fresh_nonclient_bindings,
                    ) = capture_protected_boundary(
                        partition=str(client_contract["partition"]),
                        cpu_limit=int(client_contract["cpu_limit"]),
                        memory_limit_mib=int(
                            client_contract["memory_limit_mib"]
                        ),
                        max_submit_jobs=int(
                            client_contract["max_submit_jobs"]
                        ),
                        reserve_jobs=int(client_contract["reserve_jobs"]),
                        exclude_intent_ids=frozenset({batch_id}),
                    )
                    if fresh_active_cell_jobs + len(selected) > int(
                        production_contract["current_ceiling"]
                    ):
                        raise scheduler_safety.SchedulerSafetyError(
                            "scientific client ceiling shrank below the "
                            "prepared microbatch before sbatch"
                        )
                    if len(selected) > fresh_headroom:
                        raise scheduler_safety.SchedulerSafetyError(
                            "scientific client headroom shrank below the "
                            "prepared microbatch before sbatch"
                        )
                    production_client_capacity = fresh_production_capacity
                    production_partition_usage = (
                        fresh_production_occupancy.usage
                    )
                    protected_task_headroom = fresh_headroom
                except (
                    KeyError,
                    DispatcherError,
                    scheduler_safety.SchedulerSafetyError,
                    protected_capacity.ProtectedCapacityError,
                ) as exc:
                    work["intents"][batch_id].update(
                        {
                            "state": "prepared",
                            "error": (
                                "protected client placement drifted before sbatch: "
                                + str(exc)
                            ),
                        }
                    )
                    _atomic_write_json(args.ledger_path, work)
                    raise DispatcherError(
                        "protected client partition/QOS drifted immediately "
                        f"before sbatch: {exc}"
                    ) from exc
            elif qualification_capacity_contract is not None:
                try:
                    assert qualification_execution_authority is not None
                    assert qualification_execution_binding is not None
                    fresh_execution_authority = (
                        load_qualification_execution_authority(
                            str(qualification_execution_path)
                        )
                    )
                    if (
                        fresh_execution_authority.payload
                        != qualification_execution_authority.payload
                        or _qualification_execution_authority_binding(
                            fresh_execution_authority
                        )
                        != qualification_execution_binding
                    ):
                        raise DispatcherError(
                            "qualification execution authority changed after "
                            "planning"
                        )
                    qualification_capacity_contract = (
                        protected_capacity.load_contract(
                            qualification_values["path"],
                            expected_release_git_commit=str(
                                qualification_values["release_git_commit"]
                            ),
                            expected_marker_id=str(
                                qualification_values["marker_id"]
                            ),
                            expected_sha256=str(
                                qualification_values["sha256"]
                            ),
                        )
                    )
                    protected_capacity.authorize_client(
                        qualification_capacity_contract,
                        partition=str(args.cell_partition),
                        qos=str(args.cell_qos),
                        required_slots=384,
                        required_reserve_jobs=64,
                    )
                    fresh_qualification_capacity = (
                        protected_capacity.capture_live_client_capacity(
                            qualification_capacity_contract,
                            partition=str(args.cell_partition),
                            qos=str(args.cell_qos),
                            required_time_limit_seconds=(
                                scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                            ),
                        )
                    )
                    (
                        fresh_qualification_occupancy,
                        fresh_headroom,
                        _fresh_active_cell_jobs,
                        _fresh_client_bindings,
                        _fresh_nonclient_bindings,
                    ) = capture_protected_boundary(
                        partition=str(args.cell_partition),
                        cpu_limit=int(
                            qualification_client_placement.capacity["cpus"]
                        ),
                        memory_limit_mib=int(
                            qualification_client_placement.capacity[
                                "memory_mib"
                            ]
                        ),
                        max_submit_jobs=int(
                            qualification_client_placement.capacity[
                                "submit_headroom"
                            ]
                        ),
                        reserve_jobs=64,
                        exclude_intent_ids=frozenset({batch_id}),
                    )
                    if len(selected) > fresh_headroom:
                        raise scheduler_safety.SchedulerSafetyError(
                            "protected qualification client headroom shrank "
                            "below the prepared microbatch before sbatch"
                        )
                    qualification_client_capacity = (
                        fresh_qualification_capacity
                    )
                    qualification_partition_usage = (
                        fresh_qualification_occupancy.usage
                    )
                    protected_task_headroom = fresh_headroom
                except (
                    DispatcherError,
                    scheduler_safety.SchedulerSafetyError,
                    protected_capacity.ProtectedCapacityError,
                ) as exc:
                    work["intents"][batch_id].update(
                        {
                            "state": "prepared",
                            "error": (
                                "protected qualification placement drifted "
                                "before sbatch: "
                                + str(exc)
                            ),
                        }
                    )
                    _atomic_write_json(args.ledger_path, work)
                    raise DispatcherError(
                        "protected qualification partition/QOS drifted "
                        f"immediately before sbatch: {exc}"
                    ) from exc
            work["intents"][batch_id]["state"] = "submitting"
            work["intents"][batch_id]["submit_started_at"] = time.time()
            _atomic_write_json(args.ledger_path, work)
            try:
                job_id = _submit_sbatch(sbatch_path)
            except (DispatcherError, OSError, subprocess.TimeoutExpired) as exc:
                # Any error after invoking sbatch is an ambiguous external boundary:
                # Slurm may have accepted the array before the client lost/failed to
                # parse its reply.  Preserve ``submitting`` until complete squeue+sacct
                # truth proves absence after the visibility grace.
                submission_error = str(exc)
                work["intents"][batch_id].update(
                    {
                        "state": "submitting",
                        "error": submission_error,
                        "last_submit_error_at": time.time(),
                    }
                )
                # Publish the ambiguous outcome before releasing the pause boundary.
                # A concurrent drain must see this reservation and wait for joined
                # scheduler visibility instead of declaring an empty cut.
                _atomic_write_json(args.ledger_path, work)
            else:
                accepted_at = time.time()
                _record_submission(
                    work,
                    job_id=job_id,
                    batch_id=batch_id,
                    manifest_path=manifest_path,
                    sbatch_path=sbatch_path,
                    batch=batch,
                    now=accepted_at,
                )
                submission = {
                    "job_id": job_id,
                    "batch_id": batch_id,
                    "tasks": len(selected),
                }
                submitted = True
                # Bind the accepted numeric job ID durably while admission is still
                # fenced.  Pause may now cancel or wait for this exact allocation even
                # if squeue has not exposed it yet.
                _atomic_write_json(args.ledger_path, work)

    if dry_run:
        # Dry-run reports the exact next fairness state without persisting it.
        work["fairness"] = {
            "cursor": admission.cursor,
            "deficits": admission.deficits,
        }

    backlogged_runs = {candidate.run_id for candidate in candidates}
    profile_backlog_cells: Counter[ProfileKey] = Counter(
        candidate.profile_key for candidate in candidates
    )
    profile_backlog_work: Counter[ProfileKey] = Counter()
    for candidate in candidates:
        profile_backlog_work[candidate.profile_key] += candidate.fanout_cost
    profile_backlog = [
        {
            "server_pool_root": key[0],
            "serving_profile": key[1],
            "eligible_cells": profile_backlog_cells[key],
            "backlog_fanout_work": profile_backlog_work[key],
            "live_replicas": int(capacity.live_servers.get(key, 0)),
            "backlog_work_per_replica": (
                None
                if int(capacity.live_servers.get(key, 0)) == 0
                else profile_backlog_work[key]
                / int(capacity.live_servers[key])
            ),
        }
        for key in sorted(profile_backlog_work)
    ]
    if qualification_execution_authority is not None:
        pressure_history = work.setdefault(
            "qualification_profile_pressure", {}
        )
        if not isinstance(pressure_history, dict):
            raise DispatcherError(
                "qualification profile-pressure ledger is malformed"
            )
        for row in profile_backlog:
            if int(row["backlog_fanout_work"]) <= 0:
                continue
            pressure_key = _key(
                (
                    str(row["server_pool_root"]),
                    str(row["serving_profile"]),
                )
            )
            candidate_pressure = {
                **row,
                "observed_poll": poll_number,
            }
            previous = pressure_history.get(pressure_key)
            replace_pressure = not isinstance(previous, Mapping)
            if isinstance(previous, Mapping):
                new_replicas = int(candidate_pressure["live_replicas"])
                old_replicas = int(previous["live_replicas"])
                new_work = int(candidate_pressure["backlog_fanout_work"])
                old_work = int(previous["backlog_fanout_work"])
                replace_pressure = (
                    (new_replicas == 0 and old_replicas != 0)
                    or (
                        (new_replicas == 0) == (old_replicas == 0)
                        and new_work * max(1, old_replicas)
                        > old_work * max(1, new_replicas)
                    )
                )
            if replace_pressure:
                pressure_history[pressure_key] = candidate_pressure
    admitted_runs = {candidate.run_id for candidate in selected} if (submitted or dry_run) else set()
    updated_runs, starvation = update_starvation_counters(
        work["runs"],
        backlogged_runs=backlogged_runs,
        admitted_runs=admitted_runs,
        poll_number=poll_number,
    )
    work["runs"] = updated_runs
    work["updated_at"] = now
    report = {
        "poll_number": poll_number,
        "dry_run": dry_run,
        "qos": {
            "total_jobs": total_jobs,
            "active_cell_jobs": active_cell_jobs,
            "qos_limit": args.qos_limit,
            "reserve": args.reserve,
            "available_slots": slots,
            "invisible_reserved_tasks": invisible_reservations,
            "client_capacity_contract": (
                None
                if production_client_capacity is None
                else {
                    "evidence_id": production_client_capacity["evidence_id"],
                    "partition": production_contract["client_capacity"][
                        "partition"
                    ],
                    "qos": production_contract["client_capacity"]["qos"],
                    "capacity_generation": production_contract[
                        "client_capacity"
                    ]["capacity_generation"],
                    "authorization_sha256": production_contract[
                        "client_capacity"
                    ]["authorization_sha256"],
                    "cpu_limit": production_contract["client_capacity"][
                        "cpu_limit"
                    ],
                    "memory_limit_mib": production_contract[
                        "client_capacity"
                    ]["memory_limit_mib"],
                    "max_submit_jobs": production_contract[
                        "client_capacity"
                    ]["max_submit_jobs"],
                }
            ),
            # Retain the historical field name for report readers, but include the
            # exact dynamic partition identity instead of implying mit_normal.
            "mit_normal_usage": (
                None
                if production_partition_usage is None
                else {
                    "evidence_id": production_partition_usage["evidence_id"],
                    "partition": production_partition_usage["partition"],
                    "job_count": production_partition_usage["job_count"],
                    "used_cpus": production_partition_usage["used_cpus"],
                    "used_memory_mib": production_partition_usage[
                        "used_memory_mib"
                    ],
                    "task_headroom": protected_task_headroom,
                }
            ),
        },
        "live_servers": {
            _key(key): value for key, value in sorted(capacity.live_servers.items())
        },
        "active_manifest_cells": len(active),
        "active_fanout_load": {
            _key(key): value for key, value in sorted(active_task_load.items())
        },
        "validation_fairness": copy.deepcopy(work["validation_fairness"]),
        "eligible_cells": len(candidates),
        "profile_backlog": profile_backlog,
        "state_counts": state_counts,
        "selected": [_task_from_candidate(candidate) for candidate in selected],
        "submission": submission,
        "submission_error": submission_error,
        "starvation_warnings": list(starvation),
        "warnings": join_warnings,
        "unmappable_cell_jobs": unmappable_jobs,
        "validation_errors": validation_errors,
        "safety_alert_keys": [
            finding["dedupe_key"] for finding in safety_findings
        ],
    }
    return {"ledger": work, "report": report}


def _run_dispatch(args: argparse.Namespace) -> int:
    results_root = Path(args.results_root).expanduser().resolve()
    args.state_dir = Path(args.state_dir).expanduser().resolve()
    args.control_state_dir = (
        None
        if args.control_state_dir is None
        else Path(args.control_state_dir).expanduser().resolve()
    )
    args.qualification_execution_authority = (
        None
        if args.qualification_execution_authority is None
        else Path(args.qualification_execution_authority).expanduser().resolve()
    )
    specs = load_run_specs(
        args.run,
        results_root=results_root,
        server_pool_values=args.server_pool,
        weight_values=args.weight,
    )
    if not specs:
        raise DispatcherError("dispatch requires at least one --run")
    if args.max_batch > MAX_BATCH_DEFAULT:
        raise DispatcherError(f"--max-batch may not exceed {MAX_BATCH_DEFAULT}")
    if args.cell_mem not in {"2G", "4G"}:
        raise DispatcherError("--cell-mem must be 2G, or 4G after a MaxRSS pilot")
    if args.control_state_dir is not None and (
        args.cell_mem != "4G"
        or args.cell_time != CELL_TIME_DEFAULT
    ):
        raise DispatcherError(
            "schema-5 production requires --cell-mem=4G and "
            "--cell-time=12:00:00; the cell partition is ignored and derived from "
            "the reverified control capacity generation"
        )
    args.ledger_path = args.state_dir / "ledger.json"
    if args.dry_run:
        args.once = True
        outcome = _dispatch_poll(args, specs, _load_ledger(args.state_dir / "ledger.json"), dry_run=True)
        print(json.dumps(outcome["report"], indent=2, sort_keys=True))
        return 0

    ledger_path = args.ledger_path
    # Universal order is root lock -> state lock.  Queue the successor only after both
    # are owned so an alternate state directory cannot create an overlapping chain.
    with global_dispatcher_lock(results_root):
        with singleton_lock(args.state_dir):
            ledger = _load_ledger(ledger_path)
            successor_queued = args.successor_sbatch is None
            while True:
                if not successor_queued:
                    try:
                        successor = _queue_successor(Path(args.successor_sbatch).resolve())
                    except (DispatcherError, OSError) as exc:
                        print(
                            f"[dispatcher] SUCCESSOR RETRY: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        successor_queued = True
                        print(
                            f"[dispatcher] durable successor queued as job {successor}",
                            flush=True,
                        )
                outcome = _dispatch_poll(args, specs, ledger, dry_run=False)
                ledger = outcome["ledger"]
                _atomic_write_json(ledger_path, ledger)
                report = outcome["report"]
                print(json.dumps(report, sort_keys=True), flush=True)
                for run_id in report["starvation_warnings"]:
                    print(
                        f"[dispatcher] WARNING: run {run_id!r} remained backlogged with no "
                        "admission for two consecutive polls",
                        file=sys.stderr,
                        flush=True,
                    )
                for warning in report["warnings"]:
                    print(f"[dispatcher] WARNING: {warning}", file=sys.stderr, flush=True)
                if report["submission_error"]:
                    print(
                        f"[dispatcher] SUBMISSION RETRY: {report['submission_error']}",
                        file=sys.stderr,
                        flush=True,
                    )
                for error in report["validation_errors"][:20]:
                    print(
                        f"[dispatcher] VALIDATION ERROR: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                if args.once:
                    return 0
                time.sleep(args.poll_seconds)


def _run_status(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).expanduser().resolve()
    ledger = _load_ledger(state_dir / "ledger.json")
    jobs = ledger.get("jobs", {})
    runs = ledger.get("runs", {})
    cells = ledger.get("cells", {})
    by_state: dict[str, int] = defaultdict(int)
    for record in cells.values():
        by_state[str(record.get("completion_state", "unknown"))] += 1
    report = {
        "state_dir": str(state_dir),
        "schema_version": ledger["schema_version"],
        "updated_at": ledger.get("updated_at"),
        "poll_number": ledger.get("poll_number", 0),
        "validation_fairness": ledger.get(
            "validation_fairness", {"next_run_id": None}
        ),
        "runs": runs,
        "jobs": {
            "total": len(jobs),
            "active_at_last_poll": sum(record.get("state") == "active" for record in jobs.values()),
        },
        "cells_by_state": dict(sorted(by_state.items())),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _load_batch_task(
    batch_manifest: Path, index: int, *, expected_sha256: str
) -> Mapping[str, Any]:
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or batch_manifest.is_symlink()
        or not batch_manifest.is_file()
        or stat.S_IMODE(batch_manifest.stat().st_mode) & 0o222
    ):
        raise DispatcherError(f"unsafe or unpinned batch manifest: {batch_manifest}")
    raw = batch_manifest.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise DispatcherError(
            f"batch manifest drifted: expected {expected_sha256}, got {observed_sha256}"
        )
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DispatcherError(f"invalid batch manifest JSON: {batch_manifest}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise DispatcherError(f"invalid batch manifest schema: {batch_manifest}")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not 0 <= index < len(tasks):
        raise DispatcherError(f"batch task index {index} out of range in {batch_manifest}")
    task = tasks[index]
    if not isinstance(task, dict):
        raise DispatcherError(f"batch task {index} is not an object")
    _runtime_environment(task)
    return task


def _runtime_environment(task: Mapping[str, Any]) -> dict[str, str]:
    value = task.get("runtime_environment")
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) != PRODUCTION_ENVIRONMENT_KEYS:
        raise DispatcherError(
            "schema-5 task runtime environment is missing required keys or contains "
            "untrusted keys"
        )
    if any(not isinstance(item, str) or not item for item in value.values()):
        raise DispatcherError("schema-5 task runtime environment values must be non-empty text")
    if any(
        value[key] != "1"
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    ):
        raise DispatcherError("schema-5 task must enforce offline Hugging Face operation")
    try:
        rollout_generation = int(value["ASYS_ROLLOUT_GENERATION"])
    except ValueError as exc:
        raise DispatcherError("schema-5 rollout generation is not an integer") from exc
    if rollout_generation < 1:
        raise DispatcherError("schema-5 rollout generation must be positive")
    try:
        capacity_generation = int(value["ASYS_CAPACITY_GENERATION"])
    except ValueError as exc:
        raise DispatcherError("schema-5 capacity generation is not an integer") from exc
    if capacity_generation < 1:
        raise DispatcherError("schema-5 capacity generation must be positive")
    if not Path(value["ASYS_FLEET_CONTRACT_PATH"]).is_absolute():
        raise DispatcherError("schema-5 fleet contract path must be absolute")
    for field in (
        "ASYS_FLEET_CONTRACT_SHA256",
        "ASYS_RELEASE_FLEET_CONTRACT_SHA256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value[field]) is None:
            raise DispatcherError(f"schema-5 {field} is not a lowercase SHA-256")
    return dict(value)


def build_run_one_command(
    task: Mapping[str, Any], *, release_worktree: str | Path | None = None
) -> list[str]:
    """Verify a task and build its worker command from an explicit release root.

    Production executes the dispatcher from a non-editable installation under ``-I``.
    Repository-level contracts and prompt files therefore cannot be discovered relative
    to the installed package.  The Slurm-spooled array entry point supplies the already
    verified release worktree; pass every runtime resource below that exact root to the
    worker rather than falling back to package-relative paths.
    """
    run_root = Path(str(task["run_root"]))
    run_id = str(task["run_id"])
    if run_root.name != run_id:
        raise DispatcherError(
            f"task run root {run_root} does not end in its run id {run_id!r}"
        )
    snapshot = load_manifest(run_root)
    if snapshot.sha256 != task.get("manifest_sha256"):
        raise DispatcherError(
            f"batch pins manifest {task.get('manifest_sha256')}, disk has {snapshot.sha256}"
        )
    frozen = load_frozen_benchmark_contracts(run_root, snapshot=snapshot)
    if frozen.sidecar_sha256 != task.get("benchmark_contracts_sha256"):
        raise DispatcherError(
            "batch benchmark contract pin no longer matches the frozen run sidecar: "
            f"task has {task.get('benchmark_contracts_sha256')}, disk has "
            f"{frozen.sidecar_sha256}"
        )
    index = int(task["source_index"])
    if not 0 <= index < len(snapshot.cells):
        raise DispatcherError(f"source index {index} out of range in {snapshot.path}")
    cell = snapshot.cells[index]
    if cell.cell_id != task.get("cell_id") or cell.config_hash() != task.get("config_hash"):
        raise DispatcherError("batch task no longer matches its frozen manifest cell")
    runtime_environment = _runtime_environment(task)
    if runtime_environment:
        if release_worktree is None:
            raise DispatcherError(
                "schema-5 task lacks its explicit immutable release worktree"
            )
        raw_release_worktree = Path(release_worktree).expanduser()
        if not raw_release_worktree.is_absolute():
            raise DispatcherError(
                "schema-5 release worktree must be an absolute path"
            )
        release_root = raw_release_worktree.resolve()
        model_contract_path = release_root / "configs" / "model_contracts.v1.json"
        fleet_contract_path = Path(
            runtime_environment["ASYS_FLEET_CONTRACT_PATH"]
        ).expanduser()
        if not fleet_contract_path.is_absolute():
            raise DispatcherError(
                "schema-5 fleet contract path must be absolute"
            )
        fleet_contract_path = fleet_contract_path.resolve()
        prompt_root = release_root / "configs" / "prompts"
    elif release_worktree is not None:
        raise DispatcherError(
            "an immutable release worktree is forbidden for a legacy task"
        )
    command = [sys.executable]
    if runtime_environment:
        # Isolated mode ignores PYTHON* variables and the user site for the long-lived
        # cell process.  Its packages therefore come only from the attested harness
        # prefix, rather than a submit host's ambient PYTHONPATH.
        command.append("-I")
    command.extend([
        "-m",
        "agents_scaling.experiment.run_one",
        "--run-id",
        run_id,
        "--cells-file",
        str(snapshot.path),
        "--index",
        str(index),
        "--benchmark-contracts-sha256",
        frozen.sidecar_sha256,
    ])
    if runtime_environment:
        command.extend(
            [
                "--release-worktree",
                str(release_root),
                "--model-contract",
                str(model_contract_path),
                "--fleet-contract",
                str(fleet_contract_path),
                "--prompt-root",
                str(prompt_root),
            ]
        )
    server_run_id = task.get("server_run_id")
    if server_run_id:
        command.extend(["--server-pool", str(server_run_id)])
    return command


def _verify_pinned_cell_runtime(
    *, expected_release_root: str, expected_harness_prefix: str
) -> None:
    """Prove this worker is executing from the control plane's immutable pins."""

    raw_release = Path(expected_release_root).expanduser()
    raw_harness = Path(expected_harness_prefix).expanduser()
    if not raw_release.is_absolute() or not raw_harness.is_absolute():
        raise DispatcherError("production execution pins must be absolute paths")
    release_root = raw_release.resolve()
    harness_prefix = raw_harness.resolve()
    expected_script = (release_root / "slurm" / "dispatch_sweeps.py").resolve()
    expected_python = (harness_prefix / "bin" / "python").resolve()
    observed_script = Path(__file__).resolve()
    observed_release = REPO.resolve()
    observed_python = Path(sys.executable).resolve()
    observed_prefix = Path(sys.prefix).resolve()
    mismatches: dict[str, tuple[str, str]] = {}
    for field, expected, observed in (
        ("release_worktree", release_root, observed_release),
        ("dispatcher_script", expected_script, observed_script),
        ("harness_prefix", harness_prefix, observed_prefix),
        ("harness_python", expected_python, observed_python),
    ):
        if expected != observed:
            mismatches[field] = (str(expected), str(observed))
    if mismatches:
        raise DispatcherError(
            "schema-5 cell runtime differs from immutable execution pins: "
            + "; ".join(
                f"{field} expected {expected}, observed {observed}"
                for field, (expected, observed) in sorted(mismatches.items())
            )
        )


def _verify_runtime_environment_attestation(
    runtime_environment: Mapping[str, str],
) -> None:
    """Reject a cell before execution if either frozen prefix changed."""

    try:
        runtime_integrity.verify_generation_lease(
            lease_path=Path(runtime_environment["ASYS_RUNTIME_INTEGRITY_LEASE"]),
            attestation_path=Path(runtime_environment["ASYS_RUNTIME_ATTESTATION"]),
            attestation_sha256=runtime_environment["ASYS_RUNTIME_ATTESTATION_SHA256"],
            generation=int(runtime_environment["ASYS_ROLLOUT_GENERATION"]),
            release_id=runtime_environment["ASYS_RELEASE_ID"],
            immutable_pins_sha256=runtime_environment["ASYS_IMMUTABLE_PINS_SHA256"],
            expected_environment_hashes={
                "harness": runtime_environment["ASYS_HARNESS_ENVIRONMENT_SHA256"],
                "serving": runtime_environment["ASYS_SERVING_ENVIRONMENT_SHA256"],
            },
        )
    except (KeyError, ValueError, runtime_integrity.RuntimeIntegrityError) as exc:
        raise DispatcherError(
            f"schema-5 runtime environment integrity failed: {exc}"
        ) from exc


_QUALIFICATION_EXECUTION_AUTHORITY_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "intent_id",
        "chain_id",
        "run_id",
        "run_root",
        "release_git_commit",
        "protected_capacity",
        "readiness_generation",
        "execution",
        "runtime_environment",
        "authority_id",
    }
)
_QUALIFICATION_EXECUTION_FIELDS = frozenset(
    {
        "release_worktree",
        "harness_prefix",
        "hf_home",
        "python",
        "python_sha256",
        "dispatcher_script",
        "dispatcher_script_sha256",
        "batch_template",
        "batch_template_sha256",
    }
)
_QUALIFICATION_READINESS_FIELDS = frozenset(
    {
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
)


def _qualification_canonical_bytes(value: Any) -> bytes:
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


def _qualification_runtime_environment_sha256(
    runtime_environment: Mapping[str, str],
) -> str:
    return hashlib.sha256(
        _qualification_canonical_bytes(dict(runtime_environment))
    ).hexdigest()


def _qualification_execution_authority_binding(
    authority: QualificationExecutionAuthority,
) -> dict[str, Any]:
    return {
        "path": str(authority.path),
        "sha256": authority.sha256,
        "authority_id": authority.authority_id,
        "intent_id": str(authority.payload["intent_id"]),
        "chain_id": str(authority.payload["chain_id"]),
        "run_id": str(authority.payload["run_id"]),
        "run_root": str(authority.payload["run_root"]),
        "release_git_commit": str(
            authority.payload["release_git_commit"]
        ),
        "runtime_environment_sha256": (
            _qualification_runtime_environment_sha256(
                authority.runtime_environment
            )
        ),
        "release_worktree": authority.execution["release_worktree"],
        "harness_prefix": authority.execution["harness_prefix"],
        "python": authority.execution["python"],
        "dispatcher_script": authority.execution["dispatcher_script"],
        "batch_template": authority.execution["batch_template"],
    }


def load_qualification_execution_authority(
    path: str | Path,
) -> QualificationExecutionAuthority:
    """Load and revalidate the qualification's immutable cell authority.

    This is deliberately independent of production desired state.  The authority
    binds one estimand-excluded run to the same schema-5 worker boundary—sealed
    template, dispatcher, interpreter, release, runtime generation, and live
    attestation lease—without granting access to any production run or ledger.
    """

    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise DispatcherError(
            "qualification execution authority path must be absolute"
        )
    resolved = supplied.resolve()
    if (
        supplied != resolved
        or supplied.is_symlink()
        or not supplied.is_file()
        or stat.S_IMODE(supplied.stat().st_mode) & 0o222
    ):
        raise DispatcherError(
            f"qualification execution authority is unsafe: {supplied}"
        )
    raw = supplied.read_bytes()

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DispatcherError(
                    f"qualification execution authority repeats {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DispatcherError(
                    "qualification execution authority contains "
                    f"non-finite JSON number {token!r}"
                )
            ),
        )
    except DispatcherError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DispatcherError(
            f"invalid qualification execution authority {supplied}: {exc}"
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != _QUALIFICATION_EXECUTION_AUTHORITY_FIELDS
        or value.get("schema_version") != 1
        or value.get("protocol")
        != QUALIFICATION_EXECUTION_AUTHORITY_PROTOCOL
        or raw != _qualification_canonical_bytes(value)
    ):
        raise DispatcherError(
            "qualification execution authority schema or canonical bytes drifted"
        )
    identity = dict(value)
    authority_id = identity.pop("authority_id", None)
    expected_id = hashlib.sha256(
        _qualification_canonical_bytes(identity)
    ).hexdigest()
    if (
        not isinstance(authority_id, str)
        or not re.fullmatch(r"[0-9a-f]{64}", authority_id)
        or authority_id != expected_id
    ):
        raise DispatcherError(
            "qualification execution authority self-hash is invalid"
        )
    for field in ("intent_id", "chain_id"):
        if re.fullmatch(r"[0-9a-f]{64}", str(value.get(field, ""))) is None:
            raise DispatcherError(
                f"qualification execution authority has invalid {field}"
            )
    run_id = value.get("run_id")
    raw_run_root = value.get("run_root")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(raw_run_root, str)
        or not Path(raw_run_root).is_absolute()
        or Path(raw_run_root).resolve().name != run_id
        or re.fullmatch(
            r"[0-9a-f]{40}", str(value.get("release_git_commit", ""))
        )
        is None
    ):
        raise DispatcherError(
            "qualification execution authority run/release identity is invalid"
        )
    protected = value.get("protected_capacity")
    if (
        not isinstance(protected, dict)
        or set(protected) != {"path", "sha256", "marker_id"}
        or not isinstance(protected.get("path"), str)
        or not Path(str(protected["path"])).is_absolute()
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(protected.get(field, "")))
            is None
            for field in ("sha256", "marker_id")
        )
    ):
        raise DispatcherError(
            "qualification execution authority protected-capacity binding is invalid"
        )
    readiness = value.get("readiness_generation")
    if (
        not isinstance(readiness, dict)
        or set(readiness) != _QUALIFICATION_READINESS_FIELDS
        or not isinstance(readiness.get("marker_path"), str)
        or not Path(str(readiness["marker_path"])).is_absolute()
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(readiness.get(field, "")))
            is None
            for field in (
                "catalog_id",
                "marker_sha256",
                "inventory_sha256",
                "catalog_payload_sha256",
                "release_fleet_contract_sha256",
                "fleet_contract_sha256",
            )
        )
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
    ):
        raise DispatcherError(
            "qualification execution authority readiness binding is invalid"
        )
    execution_value = value.get("execution")
    if (
        not isinstance(execution_value, dict)
        or set(execution_value) != _QUALIFICATION_EXECUTION_FIELDS
        or any(
            not isinstance(execution_value.get(field), str)
            or not execution_value[field]
            for field in _QUALIFICATION_EXECUTION_FIELDS
        )
    ):
        raise DispatcherError(
            "qualification execution authority execution pins are invalid"
        )
    execution = {
        field: str(execution_value[field])
        for field in _QUALIFICATION_EXECUTION_FIELDS
    }
    absolute_fields = (
        "release_worktree",
        "harness_prefix",
        "hf_home",
        "python",
        "dispatcher_script",
        "batch_template",
    )
    if any(
        not Path(execution[field]).is_absolute()
        or Path(execution[field]).resolve() != Path(execution[field])
        for field in absolute_fields
    ):
        raise DispatcherError(
            "qualification execution authority paths are not canonical absolute paths"
        )
    release_root = Path(execution["release_worktree"])
    harness_prefix = Path(execution["harness_prefix"])
    expected_paths = {
        "python": (harness_prefix / "bin" / "python").resolve(),
        "dispatcher_script": (
            release_root / "slurm" / "dispatch_sweeps.py"
        ).resolve(),
        "batch_template": (
            release_root / "slurm" / "run_dispatch_batch.sbatch.tmpl"
        ).resolve(),
    }
    for field, expected in expected_paths.items():
        artifact = Path(execution[field])
        digest_field = f"{field}_sha256"
        if (
            artifact != expected
            or artifact.is_symlink()
            or not artifact.is_file()
            or stat.S_IMODE(artifact.stat().st_mode) & 0o222
            or re.fullmatch(
                r"[0-9a-f]{64}", execution[digest_field]
            )
            is None
            or _sealed_artifact_sha256(artifact)
            != execution[digest_field]
        ):
            raise DispatcherError(
                f"qualification execution artifact drifted: {field}"
            )
    runtime_environment = _runtime_environment(value)
    expected_runtime = {
        "ASYS_RELEASE_GIT_COMMIT": str(value["release_git_commit"]),
        "ASYS_PROTECTED_CAPACITY_MARKER": str(protected["path"]),
        "ASYS_PROTECTED_CAPACITY_MARKER_SHA256": str(protected["sha256"]),
        "ASYS_PROTECTED_CAPACITY_MARKER_ID": str(protected["marker_id"]),
        "ASYS_RELEASE_FLEET_CONTRACT_SHA256": str(
            readiness["release_fleet_contract_sha256"]
        ),
        "ASYS_FLEET_CONTRACT_SHA256": str(
            readiness["fleet_contract_sha256"]
        ),
        "ASYS_CAPACITY_GENERATION": str(
            readiness["capacity_generation"]
        ),
        "ASYS_ROLLOUT_GENERATION": str(readiness["rollout_generation"]),
    }
    mismatches = {
        key: (expected, runtime_environment.get(key))
        for key, expected in expected_runtime.items()
        if runtime_environment.get(key) != expected
    }
    for field in (
        "ASYS_MODEL_CONTRACT_SHA256",
        "ASYS_HARNESS_ENVIRONMENT_SHA256",
        "ASYS_SERVING_ENVIRONMENT_SHA256",
        "ASYS_IMMUTABLE_PINS_SHA256",
        "ASYS_RUNTIME_ATTESTATION_SHA256",
        "ASYS_ARTIFACT_POLICY_SHA256",
    ):
        if re.fullmatch(
            r"[0-9a-f]{64}", runtime_environment[field]
        ) is None:
            mismatches[field] = ("lowercase SHA-256", runtime_environment[field])
    for field in (
        "ASYS_PROTECTED_CAPACITY_MARKER",
        "ASYS_FLEET_CONTRACT_PATH",
        "ASYS_RUNTIME_ATTESTATION",
        "ASYS_RUNTIME_INTEGRITY_LEASE",
    ):
        if not Path(runtime_environment[field]).is_absolute():
            mismatches[field] = ("absolute path", runtime_environment[field])
    if mismatches:
        raise DispatcherError(
            "qualification execution authority runtime binding drifted: "
            f"{mismatches}"
        )
    _verify_runtime_environment_attestation(runtime_environment)
    return QualificationExecutionAuthority(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        authority_id=authority_id,
        payload=copy.deepcopy(value),
        runtime_environment=runtime_environment,
        execution=execution,
    )


def _run_task(args: argparse.Namespace) -> int:
    task = _load_batch_task(
        Path(args.batch_manifest),
        args.index,
        expected_sha256=args.batch_manifest_sha256,
    )
    runtime_environment = _runtime_environment(task)
    expected_release_root = getattr(args, "expected_release_root", None)
    expected_harness_prefix = getattr(args, "expected_harness_prefix", None)
    has_execution_pin = bool(expected_release_root or expected_harness_prefix)
    if bool(expected_release_root) != bool(expected_harness_prefix):
        raise DispatcherError(
            "production cell runtime requires both release-root and harness-prefix pins"
        )
    if runtime_environment and not has_execution_pin:
        raise DispatcherError(
            "schema-5 task cannot run without immutable release and harness execution pins"
        )
    if has_execution_pin and not runtime_environment:
        raise DispatcherError(
            "immutable production execution pins cannot be used for a legacy task"
        )
    if has_execution_pin:
        _verify_pinned_cell_runtime(
            expected_release_root=str(expected_release_root),
            expected_harness_prefix=str(expected_harness_prefix),
        )
        # Smoke jobs invoke this entry point from a deliberately sparse environment.
        # Publish the already-verified root locally before exec so every installed
        # package component sees the same repository-resource authority.
        os.environ["ASYS_RELEASE_WORKTREE"] = str(
            Path(str(expected_release_root)).expanduser().resolve()
        )
        _verify_runtime_environment_attestation(runtime_environment)
    command = build_run_one_command(
        task,
        release_worktree=(
            None if expected_release_root is None else str(expected_release_root)
        ),
    )
    # run_one/io resolve result locations from ASYS_RESULTS_ROOT.  Pin it to the parent of
    # the already-verified immutable run root, rather than trusting the worker's ambient
    # environment (which may differ from the coordinator's).
    os.environ["ASYS_RESULTS_ROOT"] = str(Path(str(task["run_root"])).resolve().parent)
    for key, value in runtime_environment.items():
        os.environ[key] = value
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if runtime_environment:
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.environ["PYTHONSAFEPATH"] = "1"
    os.execv(command[0], command)
    return 127  # pragma: no cover - execv replaces the process


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dispatch = sub.add_parser("dispatch", help="run the singleton global coordinator")
    dispatch.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="RUN_ID[=RUN_ROOT]",
        help="repeat for every immutable run manifest",
    )
    dispatch.add_argument(
        "--server-pool",
        action="append",
        default=[],
        metavar="RUN_ID=SERVER_RUN_ID",
        help="reuse another run's endpoint registry (passed explicitly to run_one --server-pool)",
    )
    dispatch.add_argument(
        "--weight", action="append", default=[], metavar="RUN_ID=WEIGHT"
    )
    dispatch.add_argument(
        "--results-root",
        default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT),
    )
    dispatch.add_argument(
        "--state-dir",
        default=None,
        help=(
            "global ledger/batch directory "
            f"(default: RESULTS_ROOT/{DEFAULT_DISPATCHER_STATE_DIRNAME})"
        ),
    )
    dispatch.add_argument(
        "--control-state-dir",
        default=None,
        help=(
            "schema-5 desired-state directory; required for production and used to "
            "fail closed on pin, readiness, rollout, or staged-ceiling drift"
        ),
    )
    dispatch.add_argument("--qos-limit", type=int, default=QOS_LIMIT_DEFAULT)
    dispatch.add_argument("--reserve", type=int, default=QOS_RESERVE_DEFAULT)
    dispatch.add_argument("--max-batch", type=int, default=MAX_BATCH_DEFAULT)
    dispatch.add_argument("--poll-seconds", type=float, default=120.0)
    dispatch.add_argument("--once", action="store_true")
    dispatch.add_argument("--dry-run", action="store_true")
    dispatch.add_argument(
        "--successor-sbatch",
        help="control sbatch to queue afterany once the singleton lock is held",
    )
    dispatch.add_argument(
        "--cell-partition",
        default="mit_normal",
        help=(
            "non-production fallback; schema-5 production ignores this value and "
            "uses the sealed client-capacity generation"
        ),
    )
    dispatch.add_argument(
        "--cell-qos",
        default=None,
        help=(
            "explicit non-production scientific-client QOS; defaults to "
            "--cell-partition. Schema-5 production uses sealed capacity authority."
        ),
    )
    dispatch.add_argument(
        "--protected-capacity-marker",
        help=(
            "sealed qualification-only PROTECTED_CAPACITY_COMPLETE.json; "
            "requires all protected-capacity identity flags and no control state"
        ),
    )
    dispatch.add_argument("--protected-capacity-marker-sha256")
    dispatch.add_argument("--protected-capacity-marker-id")
    dispatch.add_argument("--protected-capacity-release-git-commit")
    dispatch.add_argument(
        "--qualification-execution-authority",
        help=(
            "sealed qualification-only runtime/template/interpreter authority; "
            "requires the complete protected-capacity authority and no control state"
        ),
    )
    dispatch.add_argument("--cell-time", default=CELL_TIME_DEFAULT)
    dispatch.add_argument("--cell-mem", default=CELL_MEM_DEFAULT)
    dispatch.add_argument(
        "--fanout-slots-per-server",
        type=int,
        default=24,
        help="hard per-replica admission budget measured in agent fanout units",
    )
    dispatch.add_argument("--probe-timeout", type=float, default=1.0)
    dispatch.add_argument(
        "--validation-budget",
        type=int,
        default=64,
        help="maximum changed/unknown artifact-bearing cells semantically parsed per poll",
    )
    probe_group = dispatch.add_mutually_exclusive_group()
    probe_group.add_argument(
        "--probe-servers",
        action="store_true",
        help="also require a tolerant /health probe (diagnostic; Slurm authority is default)",
    )
    probe_group.add_argument(
        "--no-probe-servers",
        dest="probe_servers",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    dispatch.set_defaults(probe_servers=False)
    dispatch.add_argument(
        "--server-capacity",
        action="append",
        default=[],
        metavar="RUN_ID:PROFILE=N",
        help="override live replica count, primarily for deterministic dry runs",
    )
    dispatch.add_argument("--assume-total-jobs", type=int)
    dispatch.add_argument("--assume-cell-jobs", type=int)

    status = sub.add_parser("status", help="read the persistent ledger without locking or writing")
    status.add_argument(
        "--state-dir",
        default=str(
            Path(os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
            / DEFAULT_DISPATCHER_STATE_DIRNAME
        ),
    )

    task = sub.add_parser("run-task", help=argparse.SUPPRESS)
    task.add_argument("--batch-manifest", required=True)
    task.add_argument("--batch-manifest-sha256", required=True)
    task.add_argument("--index", required=True, type=int)
    task.add_argument("--expected-release-root", help=argparse.SUPPRESS)
    task.add_argument("--expected-harness-prefix", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "dispatch":
        if args.state_dir is None:
            args.state_dir = str(
                Path(args.results_root) / DEFAULT_DISPATCHER_STATE_DIRNAME
            )
        return _run_dispatch(args)
    if args.command == "status":
        return _run_status(args)
    if args.command == "run-task":
        return _run_task(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SingletonAlreadyRunning as exc:
        print(f"[dispatcher] singleton already active: {exc}", file=sys.stderr)
        raise SystemExit(75) from exc
    except DispatcherError as exc:
        print(f"[dispatcher] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
