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
import json
import math
import os
import shlex
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
from agents_scaling.serving.registry import (
    ServerEntry,
    active_slurm_allocations,
    entry_has_current_provenance,
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
ELIGIBLE_STATES = {
    CompletionState.MISSING.value,
    CompletionState.PARTIAL.value,
    CompletionState.CORRUPT.value,
}
PRODUCTION_ENVIRONMENT_KEYS = frozenset(
    {
        "ASYS_RELEASE_ID",
        "ASYS_MODEL_CONTRACT_SHA256",
        "ASYS_FLEET_CONTRACT_SHA256",
        "ASYS_HARNESS_ENVIRONMENT_SHA256",
        "ASYS_SERVING_ENVIRONMENT_SHA256",
        "ASYS_ROLLOUT_GENERATION",
        "ASYS_IMMUTABLE_PINS_SHA256",
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


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_ledger()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DispatcherError(f"cannot read dispatcher ledger {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise DispatcherError(
            f"unsupported dispatcher ledger schema in {path}: {value.get('schema_version')!r}"
        )
    value.setdefault("intents", {})
    value.setdefault("validation_fairness", {"next_run_id": None})
    for field in (
        "runs",
        "jobs",
        "intents",
        "cells",
        "fairness",
        "validation_fairness",
    ):
        if not isinstance(value.get(field), dict):
            raise DispatcherError(f"dispatcher ledger field {field!r} must be an object")
    return value


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
            )
        )
    return rows


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
        if intent.get("state") not in {"prepared", "submitting"}:
            continue
        expected_comment = f"asys-schema5-intent:{batch_id}"
        expected_path = str(Path(str(intent.get("sbatch_path", ""))).resolve())
        matching_by_base: dict[str, list[Any]] = defaultdict(list)
        for job in scheduler_snapshot.jobs:
            command = str(job.command)
            if str(job.comment) != expected_comment and expected_path not in command:
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
            age = now - float(intent.get("created_at", 0.0))
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
            "sbatch_path": str(intent.get("sbatch_path")),
            "submitted_at": float(intent.get("created_at", now)),
            "last_seen_at": now,
            "reconciled_at": now,
            "state": "active" if active else "terminal",
            "scheduler_states": states,
            "task_count": len(tasks),
            "tasks": tasks,
        }
        if record is not None and (
            record.get("batch_id") != batch_id or record.get("tasks") != tasks
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
    return Candidate(
        run_id=run_id,
        run_root=str(spec.run_root),
        source_index=index,
        cell=cell,
        manifest_sha256=spec.manifest.sha256,
        server_pool_arg=(
            None if task.get("server_pool_id") is None else str(task.get("server_pool_id"))
        ),
        server_pool_root=str(task.get("server_pool_root", spec.server_pool_root)),
        serving_profile=profile,
        fanout_cost=fanout_cost(cell),
        benchmark_contracts_sha256=contract_sha256,
        runtime_environment=tuple(
            sorted(
                (str(key), str(value))
                for key, value in dict(task.get("runtime_environment", {})).items()
            )
        ),
    )


def _active_cells(
    rows: Sequence[QueueRow],
    ledger: dict[str, Any],
    specs: Sequence[RunSpec],
    *,
    now: float,
    visibility_grace_s: float = 300.0,
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
        if now - float(intent.get("created_at", 0.0)) < visibility_grace_s:
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
    time_limit: str,
    memory: str,
    log_dir: Path,
    batch_tag: str,
    batch_id: str | None = None,
    control_state_dir: Path | None = None,
) -> str:
    if not 1 <= n_tasks <= MAX_BATCH_DEFAULT:
        raise ValueError(f"microbatch size must be in [1, {MAX_BATCH_DEFAULT}]")
    production_execution: dict[str, str] | None = None
    if control_state_dir is not None:
        from slurm.schema5_control import production_cell_execution_from_state

        production_execution = production_cell_execution_from_state(control_state_dir)
        template_path = Path(production_execution["batch_template"])
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
                    + production_execution["hf_home"]
                    + '"'
                ),
                (
                    'export ASYS_RELEASE_WORKTREE="'
                    + production_execution["release_worktree"]
                    + '"'
                ),
            )
        )
        python_command = shlex.quote(production_execution["python"])
        dispatch_script = shlex.quote(production_execution["dispatcher_script"])
        runtime_arguments = (
            " \\\n  --expected-release-root "
            + shlex.quote(production_execution["release_worktree"])
            + " \\\n  --expected-harness-prefix "
            + shlex.quote(production_execution["harness_prefix"])
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

    text = template_path.read_text(encoding="utf-8")
    replacements = {
        "BATCH_TAG": batch_tag,
        "BATCH_ID": batch_id or batch_tag,
        "PARTITION": partition,
        "CPUS": str(CELL_CPUS_DEFAULT),
        "MEM": memory,
        "TIME": time_limit,
        "LAST_INDEX": str(n_tasks - 1),
        "THROTTLE": str(n_tasks),
        "LOG_DIR": str(log_dir),
        "REPO": str(REPO),
        "BATCH_MANIFEST": str(batch_manifest),
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
    time_limit: str,
    memory: str,
    now: float,
    control_state_dir: Path | None = None,
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
    sbatch_text = _render_batch_sbatch(
        manifest_path,
        n_tasks=len(selected),
        partition=partition,
        time_limit=time_limit,
        memory=memory,
        log_dir=log_dir,
        batch_tag=batch_id[-10:],
        batch_id=batch_id,
        control_state_dir=control_state_dir,
    )
    io.atomic_write_text(sbatch_path, sbatch_text)
    return batch_id, manifest_path, sbatch_path, batch


def _submit_sbatch(path: Path) -> str:
    proc = subprocess.run(
        ["sbatch", "--parsable", str(path)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise DispatcherError(
            f"sbatch rejected {path.name} (rc={proc.returncode}): {proc.stderr.strip()[:500]}"
        )
    job_id = proc.stdout.strip().split(";", 1)[0]
    if not job_id:
        raise DispatcherError(f"sbatch returned no job id for {path}")
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
    ledger["jobs"][job_id] = {
        "job_id": job_id,
        "batch_id": batch_id,
        "batch_manifest": str(manifest_path),
        "sbatch_path": str(sbatch_path),
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
    control_state_dir = getattr(args, "control_state_dir", None)
    if control_state_dir is not None:
        from slurm.schema5_control import (
            admission_contract_from_state,
            load_control,
            production_environment_from_state,
        )

        production_contract = admission_contract_from_state(control_state_dir)
        control = load_control(control_state_dir, verify_files=True)
        immutable = control["immutable"]
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
                immutable["fleet_contract_path"],
                model_contracts=model_contracts,
                expected_sha256=immutable["fleet_contract_sha256"],
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
    if production_contract is not None:
        if args.assume_total_jobs is not None or args.assume_cell_jobs is not None:
            raise DispatcherError(
                "schema-5 production cannot replace scheduler authority with assumed counts"
            )
        from slurm.schema5_control import query_scheduler

        scheduler_snapshot = query_scheduler(now=now, tolerate_errors=False)
        if not scheduler_snapshot.squeue_ok or not scheduler_snapshot.sacct_ok:
            raise DispatcherError("schema-5 admission requires both squeue and sacct truth")
        rows = _queue_rows_from_scheduler_snapshot(scheduler_snapshot)
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
    active_cell_jobs = (
        sum(is_cell_job_name(row.job_name) for row in rows)
        if args.assume_cell_jobs is None
        else args.assume_cell_jobs
    )
    active, active_task_load, join_warnings, unmappable_jobs = _active_cells(
        rows, work, specs, now=now
    )
    join_warnings = scheduler_reconcile_warnings + join_warnings
    unmappable_jobs = scheduler_reconcile_errors + unmappable_jobs
    invisible_reservations = sum(
        int(record.get("task_count", len(record.get("tasks", []))))
        for record in work["jobs"].values()
        if record.get("state") == "visibility_grace"
    ) + sum(
        len(intent.get("tasks", []))
        for intent in work.get("intents", {}).values()
        if intent.get("state") in {"prepared", "submitting"}
        and now - float(intent.get("created_at", 0.0)) < 300.0
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
        batch_id, manifest_path, sbatch_path, batch = _write_batch(
            args.state_dir,
            selected,
            partition=args.cell_partition,
            time_limit=args.cell_time,
            memory=args.cell_mem,
            now=now,
            control_state_dir=control_state_dir,
        )
        work["intents"][batch_id] = {
            "state": "prepared",
            "created_at": now,
            "batch_manifest": str(manifest_path),
            "sbatch_path": str(sbatch_path),
            "tasks": list(batch["tasks"]),
            "fairness_after": {
                "cursor": admission.cursor,
                "deficits": admission.deficits,
            },
            "fairness_committed": False,
        }
        # Persist the exact task reservation before crossing the external sbatch boundary.
        # If the process dies after acceptance, the next coordinator either joins the
        # array via squeue %o or holds this intent through the visibility grace period.
        _atomic_write_json(args.ledger_path, work)
        work["intents"][batch_id]["state"] = "submitting"
        # The external side effect happens only after the durable state says
        # ``submitting``.  A crash at any later instruction is reconcilable through the
        # immutable batch path and intent token in Slurm job metadata.
        _atomic_write_json(args.ledger_path, work)
        try:
            job_id = _submit_sbatch(sbatch_path)
        except (DispatcherError, OSError) as exc:
            # QOS races and transient scheduler outages are expected operational events;
            # preserve the rendered audit artifact and retry a fresh plan next poll.
            submission_error = str(exc)
            work["intents"][batch_id].update(
                {"state": "rejected", "error": submission_error, "failed_at": time.time()}
            )
        else:
            _record_submission(
                work,
                job_id=job_id,
                batch_id=batch_id,
                manifest_path=manifest_path,
                sbatch_path=sbatch_path,
                batch=batch,
                now=now,
            )
            submission = {"job_id": job_id, "batch_id": batch_id, "tasks": len(selected)}
            submitted = True

    if dry_run:
        # Dry-run reports the exact next fairness state without persisting it.
        work["fairness"] = {
            "cursor": admission.cursor,
            "deficits": admission.deficits,
        }

    backlogged_runs = {candidate.run_id for candidate in candidates}
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
        "state_counts": state_counts,
        "selected": [_task_from_candidate(candidate) for candidate in selected],
        "submission": submission,
        "submission_error": submission_error,
        "starvation_warnings": list(starvation),
        "warnings": join_warnings,
        "unmappable_cell_jobs": unmappable_jobs,
        "validation_errors": validation_errors,
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
        args.cell_mem != "4G" or args.cell_time != CELL_TIME_DEFAULT
    ):
        raise DispatcherError(
            "schema-5 production requires --cell-mem=4G and --cell-time=12:00:00"
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


def _load_batch_task(batch_manifest: Path, index: int) -> Mapping[str, Any]:
    value = json.loads(batch_manifest.read_text(encoding="utf-8"))
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
        fleet_contract_path = release_root / "configs" / "schema5_fleet.v1.json"
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


def _run_task(args: argparse.Namespace) -> int:
    task = _load_batch_task(Path(args.batch_manifest), args.index)
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
    dispatch.add_argument("--cell-partition", default="mit_preemptable")
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
