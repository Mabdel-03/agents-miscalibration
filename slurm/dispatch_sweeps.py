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
from agents_scaling.serving import (
    fleet_transactions,
    protected_capacity,
    scheduler_safety,
)
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
LEGACY_LEDGER_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 2
QOS_LIMIT_DEFAULT = 448
QOS_RESERVE_DEFAULT = 64
MAX_BATCH_DEFAULT = 24
CELL_CPUS_DEFAULT = 1
# The first controlled 2G production pilot reached 1,561,824 KiB MaxRSS on a
# 14B client (2026-07-18), crossing the rollout's 1.5 GB promotion threshold.
# Keep 2G available for explicit low-memory pilots, but use 4G for production.
CELL_MEM_DEFAULT = "4G"
CELL_TIME_DEFAULT = "12:00:00"
STDIN_EXACT_SUBMISSION_TRANSPORT = "stdin_exact_bytes_v1"
INTEGRITY_RETIREMENT_DIRNAME = "integrity-retirements"
INTEGRITY_RETIREMENT_INTENT_FILENAME = "RETIREMENT_INTENT.json"
INTEGRITY_RETIREMENT_SCHEDULER_FILENAME = "SCHEDULER_ABSENCE.json"
INTEGRITY_RETIREMENT_COMPLETE_FILENAME = "RETIREMENT_COMPLETE.json"
INTEGRITY_RETIREMENT_SCHEMA_VERSION = 1
CELL_JOB_PREFIXES = ("asys-cells", "asys-dispatch-")
AUTHORITATIVE_SCHEMA5_RUN_IDS = frozenset(
    {
        "full_sweep_schema5_v1",
        "full_sweep_agent_counts_schema5_v1",
        "full_sweep_agent_count_7_schema5_v1",
    }
)
QUALIFICATION_EXECUTION_AUTHORITY_PROTOCOL = (
    "schema5-v1.2-r11-throughput-qualification-execution-authority-v2"
)
QUALIFICATION_EXECUTION_AUTHORITY_SCHEMA_VERSION = 2
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


class SubmissionPreflightError(DispatcherError):
    """Immutable local admission provenance failed before invoking Slurm."""


class SubmissionRejectedError(DispatcherError):
    """Slurm returned an explicit non-acceptance response."""


class SubmissionAmbiguousError(DispatcherError):
    """The external submission may have been accepted without a usable reply."""


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
    partition: str = ""
    qos: str = ""


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
    """Parse ``squeue -o '%F|%K|%i|%j|%T|%o|%P|%q'`` output.

    Five/six-column input remains accepted for isolated tests/backward compatibility,
    but live queries always include ``%o`` so bare legacy job names can be resolved from
    their run-scoped sbatch path, plus exact partition and QOS placement.
    """
    rows: list[QueueRow] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.rstrip("\n").split("|", 7)
        if len(fields) not in {5, 6, 8}:
            raise DispatcherError(f"malformed squeue row {line_number}: {raw!r}")
        if len(fields) == 5:
            fields.append("")
        if len(fields) == 6:
            fields.extend(("", ""))
        array_job_id, task, job_id, name, state, command, partition, qos = (
            field.strip() for field in fields
        )
        task_id = None if task in {"", "N/A"} else int(task)
        rows.append(
            QueueRow(
                array_job_id,
                task_id,
                job_id,
                name,
                state,
                command,
                "",
                partition,
                qos,
            )
        )
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
    if value.get("schema_version") not in {
        LEGACY_LEDGER_SCHEMA_VERSION,
        LEDGER_SCHEMA_VERSION,
    }:
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


def validate_production_ledger_structure(
    value: Any, *, source: str = "production dispatcher ledger"
) -> dict[str, Any]:
    """Validate every admission-relevant transaction field fail closed."""

    normalized = validate_ledger_structure(value, source=source)
    if normalized.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise DispatcherError(
            f"{source} uses legacy dispatcher schema "
            f"{normalized.get('schema_version')!r}; production requires "
            f"schema {LEDGER_SCHEMA_VERSION}"
        )

    def finite_number(item: Any) -> bool:
        return (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
        )

    if (
        not finite_number(normalized.get("created_at"))
        or not finite_number(normalized.get("updated_at"))
        or not isinstance(normalized.get("poll_number"), int)
        or isinstance(normalized.get("poll_number"), bool)
        or normalized["poll_number"] < 0
    ):
        raise DispatcherError(
            f"{source} has invalid creation/update/poll metadata"
        )
    fairness = normalized["fairness"]
    if (
        set(fairness) != {"cursor", "deficits"}
        or not isinstance(fairness["cursor"], int)
        or isinstance(fairness["cursor"], bool)
        or fairness["cursor"] < 0
        or not isinstance(fairness["deficits"], dict)
        or any(
            not isinstance(run_id, str)
            or not run_id
            or not finite_number(deficit)
            for run_id, deficit in fairness["deficits"].items()
        )
    ):
        raise DispatcherError(f"{source} has invalid fairness state")
    validation_fairness = normalized["validation_fairness"]
    validation_fairness_allowed = {
        "next_run_id",
        "last_budget",
        "last_demand",
        "last_allocations",
        "last_run_order",
        "last_used",
        "last_planned_at",
    }
    if (
        "next_run_id" not in validation_fairness
        or not set(validation_fairness) <= validation_fairness_allowed
        or (
            validation_fairness["next_run_id"] is not None
            and (
                not isinstance(validation_fairness["next_run_id"], str)
                or not validation_fairness["next_run_id"]
            )
        )
    ):
        raise DispatcherError(
            f"{source} has invalid validation-fairness state"
        )
    if set(validation_fairness) != {"next_run_id"}:
        demand = validation_fairness.get("last_demand")
        allocations = validation_fairness.get("last_allocations")
        run_order = validation_fairness.get("last_run_order")
        if (
            not isinstance(validation_fairness.get("last_budget"), int)
            or isinstance(validation_fairness.get("last_budget"), bool)
            or validation_fairness["last_budget"] < 0
            or not isinstance(demand, dict)
            or not isinstance(allocations, dict)
            or set(demand) != set(allocations)
            or any(
                not isinstance(run_id, str)
                or not run_id
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                for mapping in (demand, allocations)
                for run_id, count in mapping.items()
            )
            or not isinstance(run_order, list)
            or not all(
                isinstance(run_id, str) and run_id in demand
                for run_id in run_order
            )
            or not isinstance(validation_fairness.get("last_used"), int)
            or isinstance(validation_fairness.get("last_used"), bool)
            or validation_fairness["last_used"] < 0
            or not finite_number(
                validation_fairness.get("last_planned_at")
            )
        ):
            raise DispatcherError(
                f"{source} has malformed validation allocation history"
            )

    run_required = {
        "run_root",
        "manifest_path",
        "manifest_sha256",
        "manifest_cells",
        "benchmark_contracts_sha256",
        "server_pool_arg",
        "server_pool_root",
        "weight",
        "backlogged_polls_without_admission",
    }
    run_optional = {
        "last_admitted_poll",
        "last_starvation_warning_poll",
    }
    for run_id, record in normalized["runs"].items():
        if (
            not run_id
            or not run_required <= set(record)
            or not set(record) <= run_required | run_optional
            or not isinstance(record["run_root"], str)
            or not Path(record["run_root"]).is_absolute()
            or not isinstance(record["manifest_path"], str)
            or not Path(record["manifest_path"]).is_absolute()
            or re.fullmatch(
                r"[0-9a-f]{64}", str(record["manifest_sha256"])
            )
            is None
            or not isinstance(record["manifest_cells"], int)
            or isinstance(record["manifest_cells"], bool)
            or record["manifest_cells"] < 1
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(record["benchmark_contracts_sha256"]),
            )
            is None
            or (
                record["server_pool_arg"] is not None
                and not isinstance(record["server_pool_arg"], str)
            )
            or not isinstance(record["server_pool_root"], str)
            or not Path(record["server_pool_root"]).is_absolute()
            or not finite_number(record["weight"])
            or float(record["weight"]) <= 0
            or not isinstance(
                record["backlogged_polls_without_admission"], int
            )
            or isinstance(
                record["backlogged_polls_without_admission"], bool
            )
            or record["backlogged_polls_without_admission"] < 0
            or any(
                not isinstance(record[field], int)
                or isinstance(record[field], bool)
                or record[field] < 0
                for field in run_optional & set(record)
            )
        ):
            raise DispatcherError(
                f"{source} run {run_id!r} is not a closed manifest record"
            )

    task_required = {
        "run_id",
        "run_root",
        "source_index",
        "cell_id",
        "config_hash",
        "manifest_sha256",
        "benchmark_contracts_sha256",
        "model_size",
        "serving_profile",
        "fanout_cost",
        "server_pool_id",
        "server_run_id",
        "server_pool_root",
    }

    def validate_task(task: Any, *, context: str) -> None:
        if (
            not isinstance(task, dict)
            or not task_required <= set(task)
            or not set(task) <= task_required | {"runtime_environment"}
            or not isinstance(task["run_id"], str)
            or not task["run_id"]
            or not isinstance(task["run_root"], str)
            or not Path(task["run_root"]).is_absolute()
            or not isinstance(task["source_index"], int)
            or isinstance(task["source_index"], bool)
            or task["source_index"] < 0
            or not isinstance(task["cell_id"], str)
            or not task["cell_id"]
            or re.fullmatch(r"[0-9a-f]{12}", str(task["config_hash"]))
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(task["manifest_sha256"])
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(task["benchmark_contracts_sha256"]),
            )
            is None
            or not isinstance(task["model_size"], str)
            or not task["model_size"]
            or not isinstance(task["serving_profile"], str)
            or not task["serving_profile"]
            or not isinstance(task["fanout_cost"], int)
            or isinstance(task["fanout_cost"], bool)
            or task["fanout_cost"] < 1
            or (
                task["server_pool_id"] is not None
                and not isinstance(task["server_pool_id"], str)
            )
            or (
                task["server_run_id"] is not None
                and not isinstance(task["server_run_id"], str)
            )
            or not isinstance(task["server_pool_root"], str)
            or not Path(task["server_pool_root"]).is_absolute()
            or (
                "runtime_environment" in task
                and (
                    not isinstance(task["runtime_environment"], dict)
                    or not all(
                        isinstance(key, str)
                        and key
                        and isinstance(item, str)
                        for key, item in task["runtime_environment"].items()
                    )
                )
            )
        ):
            raise DispatcherError(f"{context} has invalid coordinate identity")

    allowed_intent_states = {
        "prepared",
        "submitting",
        "submitted",
        "reconciled",
        "not_accepted",
        "submission_rejected",
        "integrity_blocked",
        "integrity_retired",
    }
    accepted_intent_states = {"submitted", "reconciled"}
    intent_common_required = {
        "state",
        "created_at",
        "batch_manifest",
        "batch_manifest_sha256",
        "sbatch_path",
        "sbatch_sha256",
        "submission_transport",
        "submission_argv_sha256",
        "tasks",
        "fairness_after",
        "fairness_committed",
    }
    intent_common_optional = {
        "task_count",
        "protected_capacity_authority",
        "qualification_execution_authority",
    }
    intent_state_required = {
        "prepared": set(),
        "submitting": {"submit_started_at"},
        "submitted": {
            "submit_started_at",
            "submitted_at",
            "job_id",
            "spooled_sbatch_sha256",
            "spooled_receipt_path",
            "spooled_receipt_sha256",
        },
        "reconciled": {
            "submit_started_at",
            "reconciled_at",
            "job_id",
            "spooled_sbatch_sha256",
            "spooled_receipt_path",
            "spooled_receipt_sha256",
        },
        "not_accepted": {"reconciled_at", "error"},
        "submission_rejected": {
            "submit_started_at",
            "error",
            "last_submit_error_at",
        },
        "integrity_blocked": {
            "error",
            "integrity_blocked_at",
            "integrity_alert_key",
        },
        "integrity_retired": {
            "error",
            "integrity_blocked_at",
            "integrity_alert_key",
            "integrity_retired_at",
            "retirement_id",
            "retirement_receipt_path",
            "retirement_receipt_sha256",
            "retirement_semantic_report_path",
            "retirement_semantic_report_sha256",
            "retirement_scheduler_absence_sha256",
            "retirement_operator_note_sha256",
            "retirement_identity_changes",
        },
    }
    intent_state_optional = {
        "prepared": {"error"},
        "submitting": {"error", "last_submit_error_at"},
        "submitted": set(),
        # A reconciled scheduler adoption may either follow a locally receipted
        # submission or recover a lost reply.  ``submitted_at`` is retained only in
        # the former case and is never synthesized during adoption.
        "reconciled": {"submitted_at"},
        # Prepared work can be proven absent without ever crossing sbatch.  A
        # submitting intent additionally retains its exact boundary timestamp.
        "not_accepted": {"submit_started_at"},
        "submission_rejected": set(),
        "integrity_blocked": set(),
        "integrity_retired": set(),
    }
    intents = normalized["intents"]
    for batch_id, intent in intents.items():
        state = intent.get("state")
        tasks = intent.get("tasks")
        manifest_path = intent.get("batch_manifest")
        sbatch_path = intent.get("sbatch_path")
        manifest_sha256 = intent.get("batch_manifest_sha256")
        sbatch_sha256 = intent.get("sbatch_sha256")
        fairness_after = intent.get("fairness_after")
        required_fields = (
            intent_common_required
            | intent_state_required.get(str(state), set())
        )
        allowed_fields = (
            required_fields
            | intent_common_optional
            | intent_state_optional.get(str(state), set())
        )
        if state == "integrity_blocked" and (
            set(intent)
            & {
                "job_id",
                "submit_started_at",
                "submitted_at",
                "reconciled_at",
                "spooled_sbatch_sha256",
                "spooled_receipt_path",
                "spooled_receipt_sha256",
            }
            or any(
                isinstance(job, dict) and job.get("batch_id") == batch_id
                for job in normalized["jobs"].values()
            )
        ):
            raise DispatcherError(
                f"{source} integrity-blocked intent {batch_id!r} is not a "
                "pre-boundary durable incident"
            )
        if (
            not batch_id
            or any(character in batch_id for character in "|;\n\r")
            or state not in allowed_intent_states
            or not required_fields <= set(intent)
            or not set(intent) <= allowed_fields
            or not finite_number(intent.get("created_at"))
            or not isinstance(tasks, list)
            or not isinstance(manifest_path, str)
            or not Path(manifest_path).is_absolute()
            or not isinstance(sbatch_path, str)
            or not Path(sbatch_path).is_absolute()
            or Path(manifest_path) != Path(sbatch_path).with_suffix(".json")
            or re.fullmatch(r"[0-9a-f]{64}", str(manifest_sha256 or ""))
            is None
            or re.fullmatch(r"[0-9a-f]{64}", str(sbatch_sha256 or ""))
            is None
            or intent.get("submission_transport")
            != STDIN_EXACT_SUBMISSION_TRANSPORT
            or intent.get("submission_argv_sha256")
            != _stdin_submission_argv_sha256(batch_id)
            or not isinstance(fairness_after, dict)
            or set(fairness_after) != {"cursor", "deficits"}
            or not isinstance(fairness_after["cursor"], int)
            or isinstance(fairness_after["cursor"], bool)
            or fairness_after["cursor"] < 0
            or not isinstance(fairness_after["deficits"], dict)
            or any(
                not isinstance(run_id, str)
                or not run_id
                or not finite_number(deficit)
                for run_id, deficit in fairness_after["deficits"].items()
            )
            or not isinstance(intent.get("fairness_committed"), bool)
        ):
            raise DispatcherError(
                f"{source} intent {batch_id!r} is not a closed transaction record"
            )
        for authority_field in (
            "protected_capacity_authority",
            "qualification_execution_authority",
        ):
            authority = intent.get(authority_field)
            if authority is not None and not isinstance(authority, dict):
                raise DispatcherError(
                    f"{source} intent {batch_id!r} has malformed "
                    f"{authority_field}"
                )
        for task_index, task in enumerate(tasks):
            validate_task(
                task,
                context=(
                    f"{source} intent {batch_id!r} task {task_index}"
                ),
            )
        task_count = intent.get("task_count")
        if task_count is not None and (
            not isinstance(task_count, int)
            or isinstance(task_count, bool)
            or task_count != len(tasks)
        ):
            raise DispatcherError(
                f"{source} intent {batch_id!r} has invalid task_count"
            )
        if (
            "submit_started_at" in intent
            and not finite_number(intent.get("submit_started_at"))
        ):
            raise DispatcherError(
                f"{source} intent {batch_id!r} has an invalid "
                "scheduler-boundary timestamp"
            )
        if (
            "submitted_at" in intent
            and not finite_number(intent.get("submitted_at"))
        ):
            raise DispatcherError(
                f"{source} intent {batch_id!r} has an invalid submitted timestamp"
            )
        if (
            "error" in intent
            and (
                not isinstance(intent["error"], str)
                or not intent["error"]
            )
        ):
            raise DispatcherError(
                f"{source} intent {batch_id!r} has an invalid error record"
            )
        if (
            "last_submit_error_at" in intent
            and not finite_number(intent.get("last_submit_error_at"))
        ):
            raise DispatcherError(
                f"{source} intent {batch_id!r} has an invalid submission-error "
                "timestamp"
            )
        if state == "submitting" and (
            ("error" in intent) != ("last_submit_error_at" in intent)
        ):
            raise DispatcherError(
                f"{source} submitting intent {batch_id!r} has an incomplete "
                "ambiguous-outcome record"
            )
        recorded_job_id = intent.get("job_id")
        if recorded_job_id is not None and (
            not isinstance(recorded_job_id, str)
            or not recorded_job_id.isdigit()
        ):
            raise DispatcherError(
                f"{source} intent {batch_id!r} has an invalid job ID"
            )
        if state in accepted_intent_states:
            if (
                recorded_job_id is None
                or intent.get("fairness_committed") is not True
                or intent.get("spooled_sbatch_sha256") != sbatch_sha256
                or not isinstance(intent.get("spooled_receipt_path"), str)
                or not Path(intent["spooled_receipt_path"]).is_absolute()
                or Path(intent["spooled_receipt_path"])
                != Path(sbatch_path).with_suffix(".spooled.json")
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(intent.get("spooled_receipt_sha256", "")),
                )
                is None
            ):
                raise DispatcherError(
                    f"{source} accepted intent {batch_id!r} lacks sealed spool proof"
                )
            state_timestamp = (
                intent.get("submitted_at")
                if state == "submitted"
                else intent.get("reconciled_at")
            )
            if not finite_number(state_timestamp):
                raise DispatcherError(
                    f"{source} accepted intent {batch_id!r} lacks its "
                    f"{state} timestamp"
                )
        elif intent.get("fairness_committed") is not False:
            raise DispatcherError(
                f"{source} unaccepted intent {batch_id!r} committed fairness"
            )
        if state in {
            "not_accepted",
            "submission_rejected",
            "integrity_blocked",
        } and (
            not isinstance(intent.get("error"), str)
            or not intent["error"]
        ):
            raise DispatcherError(
                f"{source} failed intent {batch_id!r} lacks its error"
            )
        if state == "not_accepted" and not finite_number(
            intent.get("reconciled_at")
        ):
            raise DispatcherError(
                f"{source} proven-absent intent {batch_id!r} lacks its "
                "reconciliation timestamp"
            )
        if state == "submission_rejected" and not finite_number(
            intent.get("last_submit_error_at")
        ):
            raise DispatcherError(
                f"{source} explicitly rejected intent {batch_id!r} lacks its "
                "failure timestamp"
            )
        if state == "integrity_blocked" and (
            not finite_number(intent.get("integrity_blocked_at"))
            or intent.get("integrity_alert_key")
            != DISPATCHER_SUBMISSION_INTEGRITY_ALERT
            or intent.get("fairness_committed") is not False
            or any(
                intent.get(field) is not None
                for field in (
                    "job_id",
                    "submit_started_at",
                    "submitted_at",
                    "reconciled_at",
                    "spooled_sbatch_sha256",
                    "spooled_receipt_path",
                    "spooled_receipt_sha256",
                )
            )
            or any(
                isinstance(job, dict) and job.get("batch_id") == batch_id
                for job in normalized["jobs"].values()
            )
        ):
            raise DispatcherError(
                f"{source} integrity-blocked intent {batch_id!r} is not a "
                "pre-boundary durable incident"
            )
        if state == "integrity_retired":
            identity_changes = intent.get("retirement_identity_changes")
            if (
                not finite_number(intent.get("integrity_blocked_at"))
                or intent.get("integrity_alert_key")
                != DISPATCHER_SUBMISSION_INTEGRITY_ALERT
                or not finite_number(intent.get("integrity_retired_at"))
                or re.fullmatch(
                    r"[0-9a-f]{64}", str(intent.get("retirement_id", ""))
                )
                is None
                or not isinstance(intent.get("retirement_receipt_path"), str)
                or not Path(intent["retirement_receipt_path"]).is_absolute()
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(intent.get("retirement_receipt_sha256", "")),
                )
                is None
                or not isinstance(
                    intent.get("retirement_semantic_report_path"), str
                )
                or not Path(
                    intent["retirement_semantic_report_path"]
                ).is_absolute()
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(intent.get("retirement_semantic_report_sha256", "")),
                )
                is None
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(intent.get("retirement_scheduler_absence_sha256", "")),
                )
                is None
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(intent.get("retirement_operator_note_sha256", "")),
                )
                is None
                or not isinstance(identity_changes, list)
                or not identity_changes
                or identity_changes != sorted(set(identity_changes))
                or not all(
                    isinstance(field, str) and field
                    for field in identity_changes
                )
            ):
                raise DispatcherError(
                    f"{source} integrity-retired intent {batch_id!r} lacks its "
                    "sealed remediation receipt"
                )
        linked_job_ids = {
            str(job_id)
            for job_id, record in normalized["jobs"].items()
            if record.get("batch_id") == batch_id
        }
        if state in accepted_intent_states:
            if linked_job_ids != {str(recorded_job_id)}:
                raise DispatcherError(
                    f"{source} accepted intent {batch_id!r} does not bind "
                    "exactly one matching job record"
                )
        elif linked_job_ids:
            raise DispatcherError(
                f"{source} unaccepted intent {batch_id!r} retains accepted "
                "job artifacts"
            )

    allowed_job_states = {
        "submitted",
        "active",
        "inactive",
        "terminal",
        "visibility_grace",
    }
    job_required = {
        "job_id",
        "batch_id",
        "batch_manifest",
        "batch_manifest_sha256",
        "sbatch_path",
        "sbatch_sha256",
        "spooled_sbatch_sha256",
        "spooled_receipt_path",
        "spooled_receipt_sha256",
        "submission_transport",
        "submission_argv_sha256",
        "submitted_at",
        "last_seen_at",
        "state",
        "task_count",
        "tasks",
    }
    job_optional = {
        "reconciled_at",
        "scheduler_states",
        "inactive_since_at",
    }
    for job_id, record in normalized["jobs"].items():
        batch_id = record.get("batch_id")
        intent = intents.get(batch_id)
        tasks = record.get("tasks")
        if (
            not job_id.isdigit()
            or not job_required <= set(record)
            or not set(record) <= job_required | job_optional
            or record.get("job_id") != job_id
            or record.get("state") not in allowed_job_states
            or not isinstance(batch_id, str)
            or not isinstance(intent, dict)
            or not isinstance(tasks, list)
            or tasks != intent.get("tasks")
            or record.get("task_count") != len(tasks)
            or record.get("batch_manifest") != intent.get("batch_manifest")
            or record.get("batch_manifest_sha256")
            != intent.get("batch_manifest_sha256")
            or record.get("sbatch_path") != intent.get("sbatch_path")
            or record.get("sbatch_sha256") != intent.get("sbatch_sha256")
            or record.get("spooled_sbatch_sha256")
            != intent.get("sbatch_sha256")
            or not finite_number(record.get("submitted_at"))
            or not finite_number(record.get("last_seen_at"))
            or (
                record.get("reconciled_at") is not None
                and not finite_number(record.get("reconciled_at"))
            )
            or (
                record.get("scheduler_states") is not None
                and (
                    not isinstance(record["scheduler_states"], list)
                    or not record["scheduler_states"]
                    or not all(
                        isinstance(state, str) and state
                        for state in record["scheduler_states"]
                    )
                )
            )
            or (
                record.get("state") == "inactive"
                and not finite_number(record.get("inactive_since_at"))
            )
            or record.get("submission_transport")
            != STDIN_EXACT_SUBMISSION_TRANSPORT
            or record.get("submission_argv_sha256")
            != _stdin_submission_argv_sha256(batch_id)
            or not isinstance(record.get("spooled_receipt_path"), str)
            or record.get("spooled_receipt_path")
            != str(Path(str(record["sbatch_path"])).with_suffix(".spooled.json"))
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(record.get("spooled_receipt_sha256", "")),
            )
            is None
        ):
            raise DispatcherError(
                f"{source} job {job_id!r} is not a closed accepted transaction"
            )

    allowed_cell_states = {
        *(state.value for state in CompletionState),
        "unvalidated",
        "validation_error",
    }
    cell_required = {
        "run_id",
        "cell_id",
        "source_index",
        "model_size",
        "serving_profile",
        "fanout_cost",
        "completion_state",
        "next_eligible_at",
        "last_checked_at",
    }
    cell_optional = {
        "artifact_fingerprint",
        "validation_context",
        "eligible_for_retry",
        "validation_retry_at",
        "submission_attempts",
        "last_job_id",
        "last_submitted_at",
    }
    for cell_key, record in normalized["cells"].items():
        if (
            not cell_required <= set(record)
            or not set(record) <= cell_required | cell_optional
            or cell_key != _key((record["run_id"], record["cell_id"]))
            or record["run_id"] not in normalized["runs"]
            or not isinstance(record["source_index"], int)
            or isinstance(record["source_index"], bool)
            or record["source_index"] < 0
            or not isinstance(record["model_size"], str)
            or not record["model_size"]
            or not isinstance(record["serving_profile"], str)
            or not record["serving_profile"]
            or not isinstance(record["fanout_cost"], int)
            or isinstance(record["fanout_cost"], bool)
            or record["fanout_cost"] < 1
            or record["completion_state"] not in allowed_cell_states
            or (
                record["next_eligible_at"] is not None
                and not finite_number(record["next_eligible_at"])
            )
            or not finite_number(record["last_checked_at"])
            or (
                "eligible_for_retry" in record
                and not isinstance(record["eligible_for_retry"], bool)
            )
            or (
                "validation_retry_at" in record
                and not finite_number(record["validation_retry_at"])
            )
            or (
                "submission_attempts" in record
                and (
                    not isinstance(record["submission_attempts"], int)
                    or isinstance(record["submission_attempts"], bool)
                    or record["submission_attempts"] < 0
                )
            )
            or (
                "last_job_id" in record
                and (
                    not isinstance(record["last_job_id"], str)
                    or not record["last_job_id"].isdigit()
                )
            )
            or (
                "last_submitted_at" in record
                and not finite_number(record["last_submitted_at"])
            )
        ):
            raise DispatcherError(
                f"{source} cell {cell_key!r} is not a closed retry record"
            )
        fingerprint = record.get("artifact_fingerprint")
        if fingerprint is not None and (
            not isinstance(fingerprint, list)
            or len(fingerprint) != 3
            or [item[0] for item in fingerprint if isinstance(item, list)]
            != ["results.jsonl", "meta.json", "failure.json"]
            or any(
                not isinstance(item, list)
                or len(item) != 3
                or not isinstance(item[1], int)
                or isinstance(item[1], bool)
                or not isinstance(item[2], int)
                or isinstance(item[2], bool)
                for item in fingerprint
            )
        ):
            raise DispatcherError(
                f"{source} cell {cell_key!r} has an invalid artifact fingerprint"
            )
        validation_context = record.get("validation_context")
        if validation_context is not None and (
            not isinstance(validation_context, dict)
            or set(validation_context)
            != {
                "manifest_sha256",
                "benchmark_contracts_sha256",
                "serving_profile",
                "code_version",
                "server_pool_generation",
            }
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(validation_context["manifest_sha256"]),
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(validation_context["benchmark_contracts_sha256"]),
            )
            is None
            or validation_context["serving_profile"]
            != record["serving_profile"]
            or (
                validation_context["code_version"] is not None
                and not isinstance(validation_context["code_version"], str)
            )
            or (
                validation_context["server_pool_generation"] is not None
                and not isinstance(
                    validation_context["server_pool_generation"], str
                )
            )
        ):
            raise DispatcherError(
                f"{source} cell {cell_key!r} has invalid validation provenance"
            )
        if (
            record["completion_state"] == "validation_error"
            and (
                fingerprint is None
                or validation_context is None
                or record.get("eligible_for_retry") is not False
                or not finite_number(record.get("validation_retry_at"))
            )
        ):
            raise DispatcherError(
                f"{source} validation-error cell {cell_key!r} lacks retry fencing"
            )
    return normalized


def _load_ledger(path: Path) -> dict[str, Any]:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if not os.path.lexists(lexical):
        return _empty_ledger()
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DispatcherError(
            f"cannot resolve dispatcher ledger {lexical}: {exc}"
        ) from exc
    if resolved != lexical or lexical.is_symlink():
        raise DispatcherError(
            f"dispatcher ledger traverses a symlink: {lexical}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise DispatcherError(
            f"cannot open dispatcher ledger {lexical}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise DispatcherError(
            f"dispatcher ledger disappeared after read: {exc}"
        ) from exc

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or identity(before) != identity(after)
        or identity(current) != identity(after)
    ):
        raise DispatcherError(
            "dispatcher ledger changed or is not a one-link regular file"
        )
    raw = b"".join(blocks)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise DispatcherError(
                    f"dispatcher ledger repeats JSON field {key!r}"
                )
            result[key] = item
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise DispatcherError(
                "dispatcher ledger contains non-finite JSON number "
                f"{token!r}"
            )
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_float=finite_float,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DispatcherError(
                    "dispatcher ledger contains non-finite JSON number "
                    f"{token!r}"
                )
            ),
        )
    except DispatcherError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DispatcherError(
            f"cannot read dispatcher ledger {lexical}: {exc}"
        ) from exc
    return validate_ledger_structure(value, source=str(path))


def load_production_ledger(path: Path) -> dict[str, Any]:
    """Load one mandatory descriptor-stable schema-5 production ledger."""

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if not os.path.lexists(lexical):
        raise FileNotFoundError(lexical)
    return validate_production_ledger_structure(
        _load_ledger(lexical), source=str(lexical)
    )


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


def _publish_readonly_json_once(
    path: Path,
    value: Mapping[str, Any],
    *,
    compatible_orphan: Any | None = None,
) -> None:
    """Publish a complete, sealed JSON artifact without clobbering a winner.

    The temporary inode is fully written, fsynced, and made read-only before its
    name becomes visible.  ``link(2)`` supplies a no-clobber publication point:
    concurrent owners may race, but only the first complete inode can win.  A
    process death after the link and before temporary-name cleanup leaves two
    names for the same sealed inode; replay safely removes only our
    deterministic temporary-link namespace before descriptor-stable validation.
    """

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    parent = lexical.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        if parent.resolve(strict=True) != parent or parent.is_symlink():
            raise DispatcherError(
                f"read-only artifact directory traverses a symlink: {parent}"
            )
    except (OSError, RuntimeError) as exc:
        raise DispatcherError(
            f"cannot verify read-only artifact directory {parent}: {exc}"
        ) from exc

    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _publish_readonly_bytes_once(
        lexical,
        payload,
        compatible_orphan=compatible_orphan,
    )


def _fsync_directory_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_readonly_bytes_once(
    path: Path,
    payload: bytes,
    *,
    compatible_orphan: Any | None = None,
    crash_hook: Any | None = None,
) -> None:
    """Publish exact bytes through a marker-first private transaction directory."""

    if not isinstance(payload, bytes):
        raise DispatcherError("read-only artifact payload must be exact bytes")
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    parent = lexical.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        if parent.resolve(strict=True) != parent or parent.is_symlink():
            raise DispatcherError(
                f"read-only artifact directory traverses a symlink: {parent}"
            )
    except (OSError, RuntimeError) as exc:
        raise DispatcherError(
            f"cannot verify read-only artifact directory {parent}: {exc}"
        ) from exc

    _recover_prelink_readonly_publish(
        lexical,
        expected_payload=payload,
        compatible_orphan=compatible_orphan,
    )
    if os.path.lexists(lexical):
        _observed_path, observed = _stable_readonly_artifact(
            lexical, description=f"published read-only artifact {lexical}"
        )
        compatible = (
            observed == payload
            if compatible_orphan is None
            else bool(compatible_orphan(observed))
        )
        if not compatible:
            raise DispatcherError(
                f"published read-only artifact conflicts with {lexical}"
            )
        return

    transaction = parent / (
        f".{lexical.name}.publish.{os.getpid()}.{uuid.uuid4().hex}.txn"
    )
    transaction.mkdir(mode=0o700)
    _fsync_directory_path(parent)
    temporary = transaction / "PAYLOAD"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(
        os, "O_CLOEXEC", 0
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    if crash_hook is not None:
        crash_hook("open")
    try:
        offset = 0
        first_write = True
        while offset < len(payload):
            remaining = len(payload) - offset
            requested = (
                max(1, remaining // 2)
                if first_write and remaining > 1
                else remaining
            )
            written = os.write(
                descriptor, payload[offset : offset + requested]
            )
            if written <= 0:
                raise DispatcherError(
                    "temporary read-only artifact write made no progress"
                )
            offset += written
            if first_write:
                first_write = False
                if crash_hook is not None:
                    crash_hook("partial_write")
        os.fsync(descriptor)
        if crash_hook is not None:
            crash_hook("post_fsync")
            crash_hook("pre_fchmod")
        os.fchmod(descriptor, 0o444)
        if crash_hook is not None:
            crash_hook("post_fchmod")
        os.fsync(descriptor)
        sealed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(sealed.st_mode)
            or sealed.st_nlink != 1
            or stat.S_IMODE(sealed.st_mode) != 0o444
            or sealed.st_size != len(payload)
        ):
            raise DispatcherError(
                "temporary read-only artifact did not seal as one regular inode"
            )
    finally:
        os.close(descriptor)

    if crash_hook is not None:
        crash_hook("prelink")
    try:
        os.link(temporary, lexical, follow_symlinks=False)
    except FileExistsError:
        pass
    _fsync_directory_path(parent)
    if crash_hook is not None:
        crash_hook("postlink")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    try:
        transaction.rmdir()
    except FileNotFoundError:
        pass
    _fsync_directory_path(parent)
    _recover_prelink_readonly_publish(
        lexical,
        expected_payload=payload,
        compatible_orphan=compatible_orphan,
    )
    _observed_path, observed = _stable_readonly_artifact(
        lexical, description=f"published read-only artifact {lexical}"
    )
    compatible = (
        observed == payload
        if compatible_orphan is None
        else bool(compatible_orphan(observed))
    )
    if not compatible:
        raise DispatcherError(
            f"published read-only artifact conflicts with {lexical}"
        )


def _recover_prelink_readonly_publish(
    path: Path,
    *,
    expected_payload: bytes,
    compatible_orphan: Any | None,
) -> None:
    """Adopt a sealed owned transaction or remove an incomplete owned payload."""

    prefix = f".{path.name}.publish."
    candidates: list[tuple[Path, Path, bytes]] = []
    try:
        entries = tuple(os.scandir(path.parent))
    except OSError as exc:
        raise DispatcherError(
            f"cannot inspect interrupted publication for {path}: {exc}"
        ) from exc
    for entry in sorted(entries, key=lambda item: item.name):
        if not entry.name.startswith(prefix):
            continue
        transaction = path.parent / entry.name
        if (
            not entry.name.endswith(".txn")
            or entry.is_symlink()
            or not entry.is_dir(follow_symlinks=False)
        ):
            raise DispatcherError(
                "interrupted publication contains a foreign namespace entry: "
                f"{transaction}"
            )
        transaction_info = transaction.stat(follow_symlinks=False)
        if stat.S_IMODE(transaction_info.st_mode) != 0o700:
            raise DispatcherError(
                f"interrupted publication transaction is not private: "
                f"{transaction}"
            )
        children = tuple(transaction.iterdir())
        if not children:
            transaction.rmdir()
            continue
        if len(children) != 1 or children[0].name != "PAYLOAD":
            raise DispatcherError(
                f"interrupted publication transaction has foreign entries: "
                f"{transaction}"
            )
        candidate = children[0]
        try:
            info = candidate.stat(follow_symlinks=False)
        except OSError as exc:
            raise DispatcherError(
                f"cannot inspect interrupted publication payload: {exc}"
            ) from exc
        if candidate.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise DispatcherError(
                f"interrupted publication payload is unsafe: {candidate}"
            )
        raw = candidate.read_bytes()
        sealed = (
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_nlink in {1, 2}
        )
        compatible = (
            raw == expected_payload
            if compatible_orphan is None
            else bool(compatible_orphan(raw))
        )
        if not sealed:
            candidate.unlink()
            transaction.rmdir()
            continue
        if not compatible:
            raise DispatcherError(
                f"interrupted publication temp conflicts with {path}"
            )
        candidates.append((transaction, candidate, raw))
    if not candidates:
        _fsync_directory_path(path.parent)
        return
    if not os.path.lexists(path):
        try:
            os.link(candidates[0][1], path, follow_symlinks=False)
        except FileExistsError:
            pass
    # The destination, whether adopted here or won concurrently, is validated by
    # the caller.  Remove only compatible, sealed files in our private namespace.
    for transaction, candidate, _raw in candidates:
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
        try:
            transaction.rmdir()
        except FileNotFoundError:
            pass
    _fsync_directory_path(path.parent)


def _finish_interrupted_readonly_publish(path: Path) -> None:
    """Remove only stale owned hardlinks to an already-published inode."""

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        target = lexical.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if lexical.is_symlink() or not stat.S_ISREG(target.st_mode):
        return
    prefix = f".{lexical.name}.publish."
    removed = False
    try:
        entries = tuple(os.scandir(lexical.parent))
    except OSError as exc:
        raise DispatcherError(
            f"cannot inspect interrupted publication for {lexical}: {exc}"
        ) from exc
    for entry in entries:
        if not entry.name.startswith(prefix) or not entry.name.endswith(".txn"):
            continue
        transaction = lexical.parent / entry.name
        if (
            entry.is_symlink()
            or not entry.is_dir(follow_symlinks=False)
            or stat.S_IMODE(
                transaction.stat(follow_symlinks=False).st_mode
            )
            != 0o700
        ):
            continue
        children = tuple(transaction.iterdir())
        if len(children) != 1 or children[0].name != "PAYLOAD":
            continue
        candidate = children[0]
        try:
            observed = candidate.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if (
            stat.S_ISREG(observed.st_mode)
            and observed.st_dev == target.st_dev
            and observed.st_ino == target.st_ino
            and stat.S_IMODE(observed.st_mode) & 0o222 == 0
        ):
            candidate.unlink()
            transaction.rmdir()
            removed = True
    if removed:
        _fsync_directory_path(lexical.parent)


def _stable_readonly_artifact(
    path: Path, *, description: str
) -> tuple[Path, bytes]:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DispatcherError(f"{description} is unavailable: {exc}") from exc
    if resolved != lexical or lexical.is_symlink():
        raise DispatcherError(f"{description} traverses a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise DispatcherError(f"cannot open {description}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise DispatcherError(
            f"{description} disappeared after read: {exc}"
        ) from exc

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o222
        or identity(before) != identity(after)
        or identity(current) != identity(after)
        or stat.S_IMODE(current.st_mode) & 0o222
    ):
        raise DispatcherError(
            f"{description} changed or is not a one-link read-only regular file"
        )
    return lexical, b"".join(blocks)


def _sealed_artifact_sha256(path: Path) -> str:
    """Hash one descriptor-stable sealed dispatcher transaction artifact."""

    _lexical, raw = _stable_readonly_artifact(
        path, description="dispatcher transaction artifact"
    )
    return hashlib.sha256(raw).hexdigest()


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
        [
            "squeue",
            "-u",
            user,
            "-h",
            "-r",
            "-o",
            "%F|%K|%i|%j|%T|%o|%P|%q",
        ],
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
                partition=str(job.partition),
                qos=str(job.qos),
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
    snapshot_jobs = tuple(getattr(snapshot, "jobs", ()))
    live_squeue_ids = {
        str(job.job_id)
        for job in snapshot_jobs
        if str(getattr(job, "source", "")) == "squeue"
        and bool(getattr(job, "active", False))
    }
    active_sacct_only = []
    for job in snapshot_jobs:
        if (
            str(getattr(job, "source", "")) != "sacct"
            or not bool(getattr(job, "active", False))
        ):
            continue
        prefix = str(job.job_id) + "_"
        if not any(
            live_id.startswith(prefix)
            and live_id[len(prefix) :].isdigit()
            for live_id in live_squeue_ids
        ):
            active_sacct_only.append(str(job.job_id))
    if active_sacct_only:
        raise DispatcherError(
            f"{observation} exposes active sacct-only jobs absent from live "
            "squeue truth: " + ", ".join(sorted(active_sacct_only)[:8])
        )
    rows = tuple(_queue_rows_from_scheduler_snapshot(snapshot))
    unsafe_placement = [
        row.job_id
        for row in rows
        if re.fullmatch(r"[A-Za-z0-9_.-]+", row.partition) is None
        or re.fullmatch(r"[A-Za-z0-9_.-]+", row.qos) is None
    ]
    if unsafe_placement:
        raise DispatcherError(
            f"{observation} lacks exact partition/QOS placement for live jobs: "
            + ", ".join(sorted(unsafe_placement)[:8])
        )
    live_ids = tuple(sorted(row.job_id for row in rows))
    if len(live_ids) != len(set(live_ids)):
        raise DispatcherError(
            f"{observation} repeats a live logical Slurm job element"
        )
    return rows, live_ids


def _scheduler_occupancy_bindings(
    rows: Sequence[QueueRow],
) -> dict[str, tuple[str, str, str, str, str, str]]:
    """Return the immutable/relevant scheduler view bracketing one usage query."""

    return {
        row.job_id: (
            row.job_name,
            row.comment,
            row.command,
            row.partition,
            row.qos,
            row.state.upper(),
        )
        for row in rows
    }


def _capture_stable_admission_occupancy(
    *,
    user: str,
    partition: str,
    max_attempts: int = 2,
    scheduler_reader: Any | None = None,
    usage_reader: Any | None = None,
) -> StableAdmissionOccupancy:
    """Bracket target TRES usage with identical global job-element bindings.

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
        first_rows, first_ids = _complete_scheduler_rows(
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
        usage_by_id = {
            str(row["job_id"]): row for row in usage_summary["jobs"]
        }
        stable_ids = set(second_ids)
        target_ids = {
            row.job_id for row in second_rows if row.partition == partition
        }
        added = sorted(stable_ids - set(first_ids))
        removed = sorted(set(first_ids) - stable_ids)
        usage_only = sorted(usage_ids - stable_ids)
        target_missing_usage = sorted(target_ids - usage_ids)
        first_bindings = _scheduler_occupancy_bindings(first_rows)
        second_bindings = _scheduler_occupancy_bindings(second_rows)
        changed = sorted(
            job_id
            for job_id in set(first_bindings) & set(second_bindings)
            if first_bindings[job_id] != second_bindings[job_id]
        )
        second_by_id = {row.job_id: row for row in second_rows}
        inconsistent_usage = sorted(
            job_id
            for job_id in usage_ids & stable_ids
            if (
                second_by_id[job_id].partition != partition
                or second_by_id[job_id].job_name
                != usage_by_id[job_id]["job_name"]
                or second_by_id[job_id].comment
                != usage_by_id[job_id]["comment"]
                or second_by_id[job_id].qos
                != usage_by_id[job_id]["qos"]
            )
        )
        usage_partition = (
            usage.get("partition")
            if isinstance(usage, Mapping)
            else None
        )
        if (
            usage_partition == partition
            and not added
            and not removed
            and not changed
            and not usage_only
            and not target_missing_usage
            and not inconsistent_usage
        ):
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
            f"changed={changed[:8]}, target_usage_only={usage_only[:8]}, "
            f"target_missing_usage={target_missing_usage[:8]}, "
            f"target_usage_mismatch={inconsistent_usage[:8]}, "
            f"target_partition={usage_partition!r}"
        )
    raise DispatcherError(
        "scheduler occupancy changed across the target-usage admission "
        f"observation after {max_attempts} attempts: {last_churn}"
    )


def _command_binds_exact_sbatch(command: str, expected_path: str) -> bool:
    try:
        expected_candidate = Path(expected_path).expanduser()
        if not expected_candidate.is_absolute():
            return False
        expected = str(
            Path(os.path.abspath(os.fspath(expected_candidate)))
        )
        candidates = [
            str(
                Path(
                    os.path.abspath(
                        os.fspath(Path(token).expanduser())
                    )
                )
            )
            for token in shlex.split(command)
            if token.endswith(".sbatch") and Path(token).is_absolute()
        ]
        return candidates == [expected]
    except (OSError, ValueError):
        return False


def _stdin_submission_argv(batch_id: str) -> list[str]:
    return [
        "sbatch",
        "--parsable",
        f"--comment=asys-schema5-intent:{batch_id}",
    ]


def _stdin_submission_argv_sha256(batch_id: str) -> str:
    return hashlib.sha256(
        json.dumps(
            _stdin_submission_argv(batch_id),
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _command_binds_stdin_submission(command: str, batch_id: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    normalized = ["sbatch" if Path(tokens[0]).name == "sbatch" else tokens[0], *tokens[1:]]
    return normalized == _stdin_submission_argv(batch_id)


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


def _spooled_script_receipt(
    *,
    batch_id: str,
    job_id: str,
    expected_name: str,
    expected_comment: str,
    sbatch_path: Path,
    sbatch_sha256: str,
    spooled_script_reader: Any | None,
    now: float,
) -> tuple[str, str]:
    """Publish/replay one immutable proof that Slurm spooled the validated bytes."""

    receipt_path = sbatch_path.with_suffix(".spooled.json")
    expected_identity = {
        "schema_version": 1,
        "kind": "schema5_dispatch_spooled_script_receipt",
        "batch_id": batch_id,
        "job_id": job_id,
        "job_name": expected_name,
        "scheduler_comment": expected_comment,
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": sbatch_sha256,
        "spooled_sbatch_sha256": sbatch_sha256,
    }
    def parse_receipt(raw: bytes) -> dict[str, Any]:
        def unique_object(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise DispatcherError(
                        f"spooled-script receipt repeats field {key!r}"
                    )
                value[key] = item
            return value

        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    DispatcherError(
                        "spooled-script receipt contains non-finite "
                        f"number {token!r}"
                    )
                ),
            )
        except DispatcherError:
            raise
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise DispatcherError(
                f"spooled-script receipt is invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise DispatcherError(
                "spooled-script receipt is not a JSON object"
            )
        return value

    def compatible_receipt(raw: bytes) -> bool:
        try:
            receipt = parse_receipt(raw)
        except DispatcherError:
            return False
        return bool(
            set(receipt) == set(expected_identity) | {"verified_at"}
            and all(
                receipt.get(key) == value
                for key, value in expected_identity.items()
            )
            and isinstance(receipt.get("verified_at"), (int, float))
            and not isinstance(receipt.get("verified_at"), bool)
            and math.isfinite(float(receipt["verified_at"]))
        )

    _finish_interrupted_readonly_publish(receipt_path)
    _recover_prelink_readonly_publish(
        receipt_path,
        expected_payload=b"",
        compatible_orphan=compatible_receipt,
    )
    if not os.path.lexists(receipt_path):
        spool_raw = (
            _read_spooled_batch_script(job_id)
            if spooled_script_reader is None
            else spooled_script_reader(job_id)
        )
        if not isinstance(spool_raw, (bytes, bytearray)):
            raise DispatcherError(
                "spooled-script reader returned a non-byte payload"
            )
        if hashlib.sha256(bytes(spool_raw)).hexdigest() != sbatch_sha256:
            raise DispatcherError(
                "Slurm-spooled script differs from the durable intent"
            )
        _publish_readonly_json_once(
            receipt_path,
            {
                **expected_identity,
                "verified_at": float(now),
            },
            compatible_orphan=compatible_receipt,
        )
    _finish_interrupted_readonly_publish(receipt_path)
    lexical, raw = _stable_readonly_artifact(
        receipt_path,
        description=f"intent {batch_id} spooled-script receipt",
    )
    if lexical != receipt_path:
        raise DispatcherError("spooled-script receipt lexical identity drifted")
    if not compatible_receipt(raw):
        raise DispatcherError(
            f"spooled-script receipt identity drifted for intent {batch_id}"
        )
    return str(receipt_path), hashlib.sha256(raw).hexdigest()


def _reconcile_schema5_intents(
    ledger: dict[str, Any],
    *,
    scheduler_snapshot: Any,
    now: float,
    visibility_grace_s: float = 300.0,
    spooled_script_reader: Any | None = None,
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
        expected_name = f"asys-dispatch-{batch_id[-10:]}"
        if (
            intent.get("submission_transport")
            != STDIN_EXACT_SUBMISSION_TRANSPORT
            or intent.get("submission_argv_sha256")
            != _stdin_submission_argv_sha256(batch_id)
        ):
            errors.append(
                f"intent {batch_id} lacks the closed exact-stdin submission "
                "transport contract"
            )
            continue
        artifacts = (
            ("batch_manifest", "batch_manifest_sha256"),
            ("sbatch_path", "sbatch_sha256"),
        )
        stable_artifact_paths: dict[str, Path] = {}

        def verify_artifacts() -> str | None:
            observed_paths: dict[str, Path] = {}
            for path_field, hash_field in artifacts:
                expected_hash = intent.get(hash_field)
                raw_path = intent.get(path_field)
                if (
                    not isinstance(raw_path, str)
                    or not Path(raw_path).is_absolute()
                ):
                    return f"{path_field} is not a lexical absolute path"
                try:
                    lexical, raw = _stable_readonly_artifact(
                        Path(raw_path),
                        description=(
                            f"intent {batch_id} {path_field} artifact"
                        ),
                    )
                except (DispatcherError, OSError) as exc:
                    return str(exc)
                if str(lexical) != raw_path:
                    return f"{path_field} lexical identity drifted"
                if (
                    not isinstance(expected_hash, str)
                    or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
                    or hashlib.sha256(raw).hexdigest() != expected_hash
                ):
                    return (
                        f"{path_field} bytes differ from durable intent"
                    )
                observed_paths[path_field] = lexical
            if observed_paths["batch_manifest"] != observed_paths[
                "sbatch_path"
            ].with_suffix(".json"):
                return "batch manifest is not the exact sbatch sibling"
            stable_artifact_paths.clear()
            stable_artifact_paths.update(observed_paths)
            return None

        artifact_error = verify_artifacts()
        if artifact_error is not None:
            errors.append(f"intent {batch_id} artifact drift: {artifact_error}")
            continue
        matching_by_base: dict[str, list[Any]] = defaultdict(list)
        for job in scheduler_snapshot.jobs:
            comment = str(job.comment)
            namespace_match = (
                comment == expected_comment
                or str(job.job_name) == expected_name
            )
            if not namespace_match:
                continue
            if (
                str(job.job_name) != expected_name
                or not _command_binds_stdin_submission(
                    str(job.command), batch_id
                )
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
                if intent_state == "submitting":
                    accounting_start = getattr(
                        scheduler_snapshot,
                        "accounting_start_timestamp",
                        None,
                    )
                    submit_started = intent.get("submit_started_at")
                    if (
                        not isinstance(accounting_start, (int, float))
                        or isinstance(accounting_start, bool)
                        or not math.isfinite(float(accounting_start))
                        or not isinstance(submit_started, (int, float))
                        or isinstance(submit_started, bool)
                        or float(accounting_start) > float(submit_started)
                    ):
                        intent["error"] = (
                            "scheduler absence is not authoritative: the complete "
                            "sacct window does not cover submit_started_at"
                        )
                        errors.append(
                            f"intent {batch_id} remains integrity-ambiguous because "
                            "complete accounting does not cover its submission boundary"
                        )
                        continue
                intent.update(
                    {
                        "state": "not_accepted",
                        "reconciled_at": now,
                        "error": "absent from complete squeue+sacct transaction history",
                    }
                )
                for field in (
                    "job_id",
                    "submitted_at",
                    "spooled_sbatch_sha256",
                    "spooled_receipt_path",
                    "spooled_receipt_sha256",
                    "last_submit_error_at",
                ):
                    intent.pop(field, None)
                warnings.append(f"intent {batch_id} was not accepted by Slurm")
            continue
        artifact_error = verify_artifacts()
        if artifact_error is not None:
            errors.append(
                f"intent {batch_id} artifact drift before adoption: "
                f"{artifact_error}"
            )
            continue
        job_id, matching = next(iter(matching_by_base.items()))
        try:
            spooled_sha256 = str(intent.get("sbatch_sha256"))
            receipt_path, receipt_sha256 = _spooled_script_receipt(
                batch_id=batch_id,
                job_id=job_id,
                expected_name=expected_name,
                expected_comment=expected_comment,
                sbatch_path=stable_artifact_paths["sbatch_path"],
                sbatch_sha256=spooled_sha256,
                spooled_script_reader=spooled_script_reader,
                now=now,
            )
        except (DispatcherError, OSError, subprocess.TimeoutExpired) as exc:
            errors.append(
                f"intent {batch_id} lacks exact Slurm-spooled script proof "
                f"for job {job_id}: {exc}"
            )
            continue
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
            "spooled_sbatch_sha256": spooled_sha256,
            "spooled_receipt_path": receipt_path,
            "spooled_receipt_sha256": receipt_sha256,
            "submission_transport": STDIN_EXACT_SUBMISSION_TRANSPORT,
            "submission_argv_sha256": _stdin_submission_argv_sha256(batch_id),
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
            or record.get("spooled_sbatch_sha256") != spooled_sha256
            or record.get("spooled_receipt_path") != receipt_path
            or record.get("spooled_receipt_sha256") != receipt_sha256
            or record.get("submission_transport")
            != STDIN_EXACT_SUBMISSION_TRANSPORT
            or record.get("submission_argv_sha256")
            != _stdin_submission_argv_sha256(batch_id)
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
                "spooled_sbatch_sha256": spooled_sha256,
                "spooled_receipt_path": receipt_path,
                "spooled_receipt_sha256": receipt_sha256,
            }
        )
        intent.pop("error", None)
        intent.pop("last_submit_error_at", None)
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
    dispatcher_jobs_by_intent: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if not row.job_name.startswith("asys-dispatch-"):
            continue
        if not row.comment.startswith("asys-schema5-intent:"):
            continue
        intent_id = row.comment[len("asys-schema5-intent:") :]
        dispatcher_jobs_by_intent[intent_id].add(row.array_job_id)
    ambiguous_intents = {
        intent_id: sorted(job_ids)
        for intent_id, job_ids in dispatcher_jobs_by_intent.items()
        if len(job_ids) > 1
    }
    for intent_id, job_ids in sorted(ambiguous_intents.items()):
        unmappable.append(
            f"ambiguous duplicate dispatcher intent {intent_id}: jobs {job_ids}"
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
            record.pop("inactive_since_at", None)
        elif (
            prior_state in {"submitted", "active"}
            and now - float(record.get("submitted_at", 0.0)) < visibility_grace_s
        ):
            record["state"] = "visibility_grace"
            reserve_tasks(record.get("tasks"), f"job {job_id} visibility reservation")
        elif prior_state != "inactive":
            record["inactive_since_at"] = now

    for row in rows:
        if row.job_name.startswith("asys-dispatch-"):
            row_intent = (
                row.comment[len("asys-schema5-intent:") :]
                if row.comment.startswith("asys-schema5-intent:")
                else None
            )
            if row_intent in ambiguous_intents:
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
                    if artifact_error is None:
                        try:
                            (
                                observed_receipt_path,
                                observed_receipt_sha256,
                            ) = _spooled_script_receipt(
                                batch_id=batch_id,
                                job_id=row.array_job_id,
                                expected_name=expected_name,
                                expected_comment=expected_comment,
                                sbatch_path=Path(
                                    str(job_record.get("sbatch_path", ""))
                                ),
                                sbatch_sha256=str(
                                    job_record.get("sbatch_sha256", "")
                                ),
                                spooled_script_reader=lambda _job_id: (
                                    _ for _ in ()
                                ).throw(
                                    DispatcherError(
                                        "sealed spooled-script receipt is missing"
                                    )
                                ),
                                now=now,
                            )
                            if (
                                observed_receipt_path
                                != job_record.get("spooled_receipt_path")
                                or observed_receipt_sha256
                                != job_record.get("spooled_receipt_sha256")
                            ):
                                artifact_error = (
                                    "spooled-script receipt differs from its "
                                    "durable job record"
                                )
                        except DispatcherError as exc:
                            artifact_error = str(exc)
                    verified_job_artifacts[row.array_job_id] = artifact_error
                if (
                    not batch_id
                    or row.job_name != expected_name
                    or row.comment != expected_comment
                    or artifact_error is not None
                    or job_record.get("spooled_sbatch_sha256")
                    != job_record.get("sbatch_sha256")
                    or job_record.get("submission_transport")
                    != STDIN_EXACT_SUBMISSION_TRANSPORT
                    or job_record.get("submission_argv_sha256")
                    != _stdin_submission_argv_sha256(batch_id)
                    or not _command_binds_stdin_submission(
                        row.command, batch_id
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
                    "sbatch_path": str(job_record["sbatch_path"]),
                    "sbatch_sha256": str(job_record["sbatch_sha256"]),
                    "batch_id": str(job_record["batch_id"]),
                    "submission_transport": str(
                        job_record["submission_transport"]
                    ),
                    "submission_argv_sha256": str(
                        job_record["submission_argv_sha256"]
                    ),
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
        if intent.get("state") == "integrity_blocked":
            # A deterministic local provenance failure occurred before Slurm was
            # invoked.  Keep the exact coordinates fenced indefinitely while the
            # scientific-integrity safety hold awaits explicit acknowledgement.
            reserve_tasks(intent.get("tasks"), f"blocked intent {intent_id}")
            continue
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
) -> dict[str, dict[str, Any]]:
    """Classify only servers with an immutable job/name/comment preimage."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_fleet_sha256 or "") is None:
        raise DispatcherError("trusted server classification lacks a fleet hash")
    if frozen_fleet is None:
        raise DispatcherError(
            "trusted server classification lacks the frozen placement contract"
        )
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
    bindings: dict[str, dict[str, Any]] = {}
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
                if not isinstance(entry.replica_index, int) or isinstance(
                    entry.replica_index, bool
                ):
                    continue
                replica = frozen_fleet.for_replica(
                    profile, entry.replica_index
                )
                if not isinstance(replica.qos, str) or not replica.qos:
                    raise DispatcherError(
                        f"frozen fleet replica {replica.replica_id} lacks an "
                        "exact protected QOS"
                    )
                binding = {
                    "job_name": str(history.binding["scheduler_job_name"]),
                    "comment": str(history.binding["scheduler_comment"]),
                    "sbatch_path": str(history.binding["local_script_path"]),
                    "sbatch_sha256": str(
                        history.binding["local_script_sha256"]
                    ),
                    "intent_token": str(history.binding["intent_token"]),
                    "replica_id": str(history.binding["replica_id"]),
                    "serving_profile": replica.serving_profile,
                    "ledger_generation": int(
                        history.binding["ledger_generation"]
                    ),
                    "partition": replica.partition,
                    "qos": replica.qos,
                    "allocated_gpus": replica.gpus_per_replica,
                    "gpu_type": replica.gpu_type,
                }
                transaction_directory = Path(
                    binding["sbatch_path"]
                ).parents[2]
                generation_ledger_path = Path(
                    os.path.abspath(
                        os.fspath(
                            fleet_transactions.ledger_path(
                                transaction_directory,
                                binding["ledger_generation"],
                            ).expanduser()
                        )
                    )
                )
                try:
                    generation_ledger_raw = (
                        generation_ledger_path.read_bytes()
                    )
                except OSError as exc:
                    raise DispatcherError(
                        "server job "
                        f"{entry.slurm_job_id} lacks its generation fleet "
                        f"ledger: {exc}"
                    ) from exc
                binding.update(
                    {
                        "ledger_path": str(generation_ledger_path),
                        "ledger_sha256": hashlib.sha256(
                            generation_ledger_raw
                        ).hexdigest(),
                    }
                )
                if (
                    row.job_name != binding["job_name"]
                    or row.comment != binding["comment"]
                    or row.partition != binding["partition"]
                    or row.qos != binding["qos"]
                    or not fleet_transactions.command_binds_stdin_submission(
                        row.command, binding["comment"]
                    )
                ):
                    continue
                job_id = str(entry.slurm_job_id)
                prior = bindings.setdefault(job_id, binding)
                if prior != binding:
                    raise DispatcherError(
                        f"server job {job_id} has conflicting immutable provenance"
                    )
    except (
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        FleetContractError,
    ) as exc:
        raise DispatcherError(
            f"cannot classify protected server allocations: {exc}"
        ) from exc
    unbound = sorted(
        row.job_id
        for row in scheduler_rows
        if row.job_name.startswith("asys-s5-serve-")
        and row.job_id not in bindings
    )
    if unbound:
        raise DispatcherError(
            "live scientific fleet jobs lack exact ledger/registry/history "
            f"provenance: {unbound}"
        )
    return bindings


def _require_exact_trusted_scientific_binding_sets(
    provenance: protected_capacity.TrustedScientificJobProvenance,
    *,
    client_bindings: Mapping[str, Any],
    nonclient_bindings: Mapping[str, Any],
) -> None:
    """Keep residual MaxJobs and TRES headroom on one exact scientific set."""

    if (
        set(provenance.payload["trusted_cell_job_ids"])
        != set(client_bindings)
        or set(provenance.payload["trusted_fleet_job_ids"])
        != set(nonclient_bindings)
    ):
        raise DispatcherError(
            "shared trusted scientific reconciliation differs from the "
            "dispatcher's exact active mappings"
        )


def _scheduler_headroom_binding_projection(
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Project rich ledger provenance onto scheduler_safety's closed ABI."""

    projected: dict[str, dict[str, str]] = {}
    for job_id, binding in bindings.items():
        if (
            not isinstance(job_id, str)
            or not isinstance(binding, Mapping)
            or not isinstance(binding.get("job_name"), str)
            or not binding["job_name"]
            or not isinstance(binding.get("comment"), str)
            or not binding["comment"]
        ):
            raise DispatcherError(
                "cannot project malformed trusted scientific scheduler binding"
            )
        projected[job_id] = {
            "job_name": str(binding["job_name"]),
            "comment": str(binding["comment"]),
        }
    return projected


def _require_trusted_client_target_placement(
    occupancy: StableAdmissionOccupancy,
    *,
    trusted_client_bindings: Mapping[str, Mapping[str, Any]],
    expected_partition: str,
    expected_qos: str,
) -> None:
    """Prove every trusted client consumes the exact attested client envelope."""

    try:
        usage = scheduler_safety.validate_user_partition_usage(occupancy.usage)
    except scheduler_safety.SchedulerSafetyError as exc:
        raise DispatcherError(
            f"trusted client target-placement evidence is invalid: {exc}"
        ) from exc
    if (
        occupancy.usage.get("partition") != expected_partition
        or re.fullmatch(r"[A-Za-z0-9_.-]+", expected_qos) is None
    ):
        raise DispatcherError(
            "trusted client target-placement authority differs from the expected "
            "partition/QOS"
        )
    scheduler_by_id = {row.job_id: row for row in occupancy.rows}
    usage_by_id = {str(row["job_id"]): row for row in usage["jobs"]}
    if len(scheduler_by_id) != len(occupancy.rows) or len(usage_by_id) != len(
        usage["jobs"]
    ):
        raise DispatcherError(
            "trusted client target-placement evidence repeats scheduler identities"
        )
    errors: list[str] = []
    for job_id, binding in sorted(trusted_client_bindings.items()):
        scheduler_row = scheduler_by_id.get(job_id)
        usage_row = usage_by_id.get(job_id)
        if (
            scheduler_row is None
            or usage_row is None
            or scheduler_row.partition != expected_partition
            or scheduler_row.qos != expected_qos
            or scheduler_row.job_name != binding.get("job_name")
            or scheduler_row.comment != binding.get("comment")
            or usage_row["job_name"] != binding.get("job_name")
            or usage_row["comment"] != binding.get("comment")
            or usage_row["qos"] != expected_qos
            or usage_row["state"] != scheduler_row.state.upper()
            or usage_row["cpus"] != CELL_CPUS_DEFAULT
            or usage_row["memory_mib"] != 4 * 1024
        ):
            errors.append(job_id)
    if errors:
        raise DispatcherError(
            "trusted client jobs drifted from their exact protected "
            "partition/QOS/name/comment/1-CPU/4096-MiB placement: "
            + ", ".join(errors[:8])
        )


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
                "umask 027",
                "unset BASH_ENV CDPATH ENV LD_AUDIT LD_LIBRARY_PATH LD_PRELOAD",
                (
                    "unset GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_ATTR_NOSYSTEM "
                    "GIT_CEILING_DIRECTORIES"
                ),
                (
                    "unset GIT_COMMON_DIR GIT_CONFIG GIT_CONFIG_COUNT "
                    "GIT_CONFIG_GLOBAL"
                ),
                (
                    "unset GIT_CONFIG_NOSYSTEM GIT_CONFIG_PARAMETERS "
                    "GIT_CONFIG_SYSTEM GIT_DIR"
                ),
                (
                    "unset GIT_DISCOVERY_ACROSS_FILESYSTEM GIT_EXEC_PATH "
                    "GIT_INDEX_FILE GIT_NAMESPACE"
                ),
                (
                    "unset GIT_NO_REPLACE_OBJECTS GIT_OBJECT_DIRECTORY "
                    "GIT_REPLACE_REF_BASE"
                ),
                (
                    "unset GIT_SHALLOW_FILE GIT_SSH GIT_SSH_COMMAND "
                    "GIT_TEMPLATE_DIR GIT_WORK_TREE"
                ),
                "unset SLURM_CLUSTERS SLURM_CONF SLURM_EXIT_ERROR SLURM_TIME_FORMAT",
                "while IFS= read -r ambient_name; do",
                '  case "$ambient_name" in',
                (
                    "    ASYS_*|BASH_FUNC_*|PIP_*|PYTHON*|CONDA_*|HF_*|"
                    "TRANSFORMERS_*|VLLM_*|GIT_CONFIG_KEY_*|GIT_CONFIG_VALUE_*|"
                    "GIT_TRACE*|SACCT_*|SBATCH_*|SCONTROL_*|SQUEUE_*)"
                ),
                (
                    '      builtin unset -v "$ambient_name" '
                    "2>/dev/null || true"
                ),
                "      ;;",
                "  esac",
                "done < <(compgen -e)",
                "export PATH=/usr/bin:/bin",
                "readonly PATH",
                "while read -r _ _ ambient_function; do",
                '  builtin unset -f "$ambient_function"',
                "done < <(builtin declare -F)",
                "export LANG=C LC_ALL=C",
                (
                    "export GIT_ATTR_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null "
                    "GIT_CONFIG_NOSYSTEM=1"
                ),
                (
                    "export GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 "
                    "GIT_TERMINAL_PROMPT=0"
                ),
                "export GIT_PAGER=cat PAGER=cat",
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


def _read_spooled_batch_script(
    job_id: str,
    *,
    runner: Any | None = None,
) -> bytes:
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise DispatcherError("spooled-script proof requires a numeric Slurm job id")
    invoke = subprocess.run if runner is None else runner
    try:
        proc = invoke(
            ["scontrol", "write", "batch_script", job_id, "-"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DispatcherError(
            f"cannot retrieve Slurm-spooled script for job {job_id}: {exc}"
        ) from exc
    if proc.returncode != 0 or not isinstance(proc.stdout, str):
        raise DispatcherError(
            f"cannot retrieve Slurm-spooled script for job {job_id}: "
            f"rc={proc.returncode}, stderr={str(proc.stderr)[:500]}"
        )
    return proc.stdout.encode("utf-8")


def _submit_sbatch(
    path: Path,
    *,
    expected_sbatch_sha256: str,
    batch_manifest_path: Path,
    expected_batch_manifest_sha256: str,
) -> str:
    try:
        lexical_path, sbatch_raw = _stable_readonly_artifact(
            path, description="dispatcher sbatch at external submission boundary"
        )
        lexical_manifest, manifest_raw = _stable_readonly_artifact(
            batch_manifest_path,
            description="dispatcher batch manifest at external submission boundary",
        )
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected_sbatch_sha256) is None
            or re.fullmatch(
                r"[0-9a-f]{64}", expected_batch_manifest_sha256
            )
            is None
            or hashlib.sha256(sbatch_raw).hexdigest()
            != expected_sbatch_sha256
            or hashlib.sha256(manifest_raw).hexdigest()
            != expected_batch_manifest_sha256
        ):
            raise DispatcherError(
                "dispatcher transaction artifacts drifted immediately before sbatch"
            )
        if (
            not lexical_path.name.startswith("batch-")
            or lexical_path.suffix != ".sbatch"
            or lexical_manifest != lexical_path.with_suffix(".json")
        ):
            raise DispatcherError(
                f"dispatcher sbatch path lacks an intent identity: {path}"
            )
        batch_id = lexical_path.name[len("batch-") : -len(".sbatch")]
        if not batch_id or any(character in batch_id for character in "|;\n\r"):
            raise DispatcherError(f"unsafe dispatcher batch intent {batch_id!r}")
    except DispatcherError as exc:
        raise SubmissionPreflightError(str(exc)) from exc

    argv = _stdin_submission_argv(batch_id)
    try:
        submission_text = sbatch_raw.decode("utf-8")
    except UnicodeError as exc:
        raise SubmissionPreflightError(
            f"dispatcher sbatch is not UTF-8: {exc}"
        ) from exc
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=60.0,
            input=submission_text,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SubmissionAmbiguousError(
            f"sbatch reply was unavailable for {lexical_path.name}: {exc}"
        ) from exc
    if proc.returncode != 0:
        raise SubmissionRejectedError(
            f"sbatch rejected {lexical_path.name} (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:500]}"
        )
    job_id = proc.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise SubmissionAmbiguousError(
            f"sbatch returned invalid job id {job_id!r} for {lexical_path}"
        )
    try:
        spooled = _read_spooled_batch_script(job_id)
    except DispatcherError as exc:
        raise SubmissionAmbiguousError(str(exc)) from exc
    if spooled != sbatch_raw:
        raise SubmissionAmbiguousError(
            f"Slurm-spooled script differs from validated bytes for job {job_id}"
        )
    try:
        post_path, post_raw = _stable_readonly_artifact(
            lexical_path,
            description="dispatcher sbatch after external submission boundary",
        )
    except DispatcherError as exc:
        raise SubmissionAmbiguousError(str(exc)) from exc
    if post_path != lexical_path or post_raw != sbatch_raw:
        raise SubmissionAmbiguousError(
            "dispatcher sbatch lexical artifact changed across submission"
        )
    try:
        _spooled_script_receipt(
            batch_id=batch_id,
            job_id=job_id,
            expected_name=f"asys-dispatch-{batch_id[-10:]}",
            expected_comment=f"asys-schema5-intent:{batch_id}",
            sbatch_path=lexical_path,
            sbatch_sha256=expected_sbatch_sha256,
            spooled_script_reader=lambda _job_id: spooled,
            now=time.time(),
        )
    except DispatcherError as exc:
        raise SubmissionAmbiguousError(
            f"cannot seal Slurm-spooled script proof for job {job_id}: {exc}"
        ) from exc
    return job_id


def _successor_submission_argv(
    *, current_job: str, scheduler_comment: str
) -> list[str]:
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--dependency=afterany:{current_job}",
        f"--comment={scheduler_comment}",
    ]


def _successor_command_matches(
    command: str, *, current_job: str, scheduler_comment: str
) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    normalized = [
        "sbatch" if Path(tokens[0]).name == "sbatch" else tokens[0],
        *tokens[1:],
    ]
    return normalized == _successor_submission_argv(
        current_job=current_job,
        scheduler_comment=scheduler_comment,
    )


def _verify_successor_release_complete(
    path: Path,
    *,
    job_id: str,
    release_intent_path: Path,
    release_intent_sha256: str,
) -> str:
    value, raw = _strict_readonly_json_artifact(
        path, description="dispatcher successor release completion"
    )
    required = {
        "schema_version",
        "kind",
        "job_id",
        "release_intent_path",
        "release_intent_sha256",
        "release_attempt_path",
        "release_attempt_sha256",
        "release_result_path",
        "release_result_sha256",
        "scheduler_job_ids",
        "scheduler_states",
        "released_this_call",
        "completed_at",
    }
    if (
        set(value) != required
        or value.get("schema_version") != 1
        or value.get("kind")
        != "schema5_dispatcher_successor_release_complete"
        or value.get("job_id") != job_id
        or value.get("release_intent_path")
        != str(release_intent_path.resolve())
        or value.get("release_intent_sha256") != release_intent_sha256
        or not isinstance(value.get("scheduler_job_ids"), list)
        or job_id
        not in {
            _array_identity(str(observed))[0]
            for observed in value["scheduler_job_ids"]
        }
        or not isinstance(value.get("scheduler_states"), list)
        or not isinstance(value.get("released_this_call"), bool)
        or not isinstance(value.get("completed_at"), (int, float))
        or isinstance(value.get("completed_at"), bool)
    ):
        raise DispatcherError(
            "dispatcher successor release completion is malformed"
        )
    for prefix in ("release_attempt", "release_result"):
        artifact_path = value.get(f"{prefix}_path")
        artifact_sha256 = value.get(f"{prefix}_sha256")
        if artifact_path is None:
            if artifact_sha256 is not None:
                raise DispatcherError(
                    "dispatcher successor release completion has a partial "
                    f"{prefix} binding"
                )
            continue
        lexical = Path(str(artifact_path))
        if (
            not lexical.is_absolute()
            or lexical.parent != path.parent
            or re.fullmatch(r"[0-9a-f]{64}", str(artifact_sha256 or ""))
            is None
            or _sealed_artifact_sha256(lexical) != artifact_sha256
        ):
            raise DispatcherError(
                f"dispatcher successor release {prefix} binding drifted"
            )
    if value["release_attempt_path"] is None:
        raise DispatcherError(
            "dispatcher successor release completion lacks its durable attempt"
        )
    return hashlib.sha256(raw).hexdigest()


def _queue_successor(
    path: Path,
    *,
    state_dir: Path | None = None,
    runner: Any | None = None,
    scheduler_reader: Any | None = None,
    now: float | None = None,
    crash_hook: Any | None = None,
) -> str:
    """Queue/adopt one held, exact-stdin afterany successor transaction."""

    current_job = os.environ.get("SLURM_JOB_ID")
    if not current_job or not current_job.isdigit():
        raise DispatcherError("--successor-sbatch requires numeric SLURM_JOB_ID")
    timestamp = time.time() if now is None else float(now)
    invoke = subprocess.run if runner is None else runner
    lexical, script_raw = _stable_readonly_artifact(
        path, description="dispatcher successor sbatch"
    )
    script_sha256 = hashlib.sha256(script_raw).hexdigest()
    try:
        script_text = script_raw.decode("utf-8")
    except UnicodeError as exc:
        raise DispatcherError(
            f"dispatcher successor sbatch is not UTF-8: {exc}"
        ) from exc
    transaction_root = (
        (state_dir or lexical.parent)
        / "successor-transactions"
        / f"after-{current_job}"
    ).resolve()
    transaction_root.mkdir(parents=True, exist_ok=True)
    intent_path = transaction_root / "INTENT.json"
    if intent_path.exists():
        if (
            intent_path.is_symlink()
            or not intent_path.is_file()
            or intent_path.stat(follow_symlinks=False).st_nlink != 1
        ):
            raise DispatcherError(
                "dispatcher successor intent is not a one-link regular file"
            )
        try:
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DispatcherError(
                f"cannot read dispatcher successor intent: {exc}"
            ) from exc
        if not isinstance(intent, dict):
            raise DispatcherError(
                "dispatcher successor intent must be one JSON object"
            )
    else:
        token = uuid.uuid4().hex
        comment = f"asys-schema5-successor:{current_job}:{token}"
        argv = _successor_submission_argv(
            current_job=current_job,
            scheduler_comment=comment,
        )
        intent = {
            "schema_version": 1,
            "kind": "schema5_dispatcher_successor_intent",
            "state": "prepared",
            "created_at": timestamp,
            "current_job_id": current_job,
            "intent_token": token,
            "scheduler_comment": comment,
            "sbatch_path": str(lexical),
            "sbatch_sha256": script_sha256,
            "submission_transport": STDIN_EXACT_SUBMISSION_TRANSPORT,
            "submission_argv": argv,
            "submission_argv_sha256": hashlib.sha256(
                json.dumps(argv, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "job_id": None,
            "submit_started_at": None,
            "spooled_path": None,
            "spooled_sha256": None,
            "release_intent_path": None,
            "release_intent_sha256": None,
            "release_complete_path": None,
            "release_complete_sha256": None,
            "released_at": None,
            "last_error": None,
        }
        _atomic_write_json(intent_path, intent)
    expected_argv = _successor_submission_argv(
        current_job=current_job,
        scheduler_comment=str(intent.get("scheduler_comment", "")),
    )
    if (
        intent.get("schema_version") != 1
        or intent.get("kind") != "schema5_dispatcher_successor_intent"
        or intent.get("current_job_id") != current_job
        or intent.get("sbatch_path") != str(lexical)
        or intent.get("sbatch_sha256") != script_sha256
        or intent.get("submission_transport")
        != STDIN_EXACT_SUBMISSION_TRANSPORT
        or intent.get("submission_argv") != expected_argv
        or intent.get("submission_argv_sha256")
        != hashlib.sha256(
            json.dumps(expected_argv, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        or re.fullmatch(r"[0-9a-f]{32}", str(intent.get("intent_token", "")))
        is None
    ):
        raise DispatcherError("dispatcher successor intent identity drifted")
    if intent.get("state") == "committed":
        job_id = str(intent.get("job_id", ""))
        spool_path = Path(str(intent.get("spooled_path", "")))
        release_complete_path = Path(
            str(intent.get("release_complete_path", ""))
        )
        release_intent_path = Path(
            str(intent.get("release_intent_path", ""))
        )
        release_intent_sha256 = str(
            intent.get("release_intent_sha256", "")
        )
        if (
            not job_id.isdigit()
            or _sealed_artifact_sha256(spool_path) != script_sha256
            or intent.get("spooled_sha256") != script_sha256
            or not isinstance(intent.get("released_at"), (int, float))
            or _sealed_artifact_sha256(release_intent_path)
            != release_intent_sha256
            or _verify_successor_release_complete(
                release_complete_path,
                job_id=job_id,
                release_intent_path=release_intent_path,
                release_intent_sha256=release_intent_sha256,
            )
            != intent.get("release_complete_sha256")
        ):
            raise DispatcherError(
                "committed dispatcher successor proof drifted"
            )
        return job_id

    adopted_job_id: str | None = None
    if intent.get("state") == "integrity_blocked":
        raise DispatcherError(
            f"dispatcher successor integrity remains blocked: "
            f"{intent.get('last_error')}"
        )
    accepted_held_replay = intent.get("state") == "accepted_held"
    if accepted_held_replay:
        adopted_job_id = str(intent.get("job_id", ""))
        spool_path = Path(str(intent.get("spooled_path", "")))
        if (
            not adopted_job_id.isdigit()
            or intent.get("spooled_sha256") != script_sha256
            or _sealed_artifact_sha256(spool_path) != script_sha256
        ):
            raise DispatcherError(
                "accepted dispatcher successor spool proof drifted"
            )
    if intent.get("state") == "submitting":
        if scheduler_reader is None:
            from slurm.schema5_control import query_scheduler

            scheduler_reader = query_scheduler
        snapshot = scheduler_reader()
        if (
            getattr(snapshot, "squeue_ok", False) is not True
            or getattr(snapshot, "sacct_ok", False) is not True
            or getattr(snapshot, "errors", ())
        ):
            raise DispatcherError(
                "dispatcher successor recovery lacks complete scheduler truth"
            )
        matches = [
            job
            for job in snapshot.jobs
            if str(job.comment) == intent["scheduler_comment"]
        ]
        base_ids = {_array_identity(str(job.job_id))[0] for job in matches}
        if len(base_ids) > 1:
            raise DispatcherError(
                "dispatcher successor intent maps to duplicate scheduler jobs"
            )
        if matches:
            adopted_job_id = next(iter(base_ids))
            if any(
                not _successor_command_matches(
                    str(job.command),
                    current_job=current_job,
                    scheduler_comment=intent["scheduler_comment"],
                )
                for job in matches
            ):
                raise DispatcherError(
                    "dispatcher successor scheduler provenance drifted"
                )
        else:
            submit_started = intent.get("submit_started_at")
            if (
                not isinstance(submit_started, (int, float))
                or timestamp - float(submit_started) < 300.0
            ):
                raise DispatcherError(
                    "dispatcher successor acceptance remains inside visibility grace"
                )
            accounting_start = getattr(
                snapshot, "accounting_start_timestamp", None
            )
            if (
                not isinstance(accounting_start, (int, float))
                or float(accounting_start) > float(submit_started)
            ):
                raise DispatcherError(
                    "dispatcher successor absence is outside accounting coverage"
                )
            intent["state"] = "prepared"
            intent["last_error"] = (
                "prior exact submission proven absent by complete accounting"
            )
            _atomic_write_json(intent_path, intent)

    if adopted_job_id is None and intent.get("state") == "prepared":
        intent["state"] = "submitting"
        intent["submit_started_at"] = timestamp
        intent["last_error"] = None
        _atomic_write_json(intent_path, intent)
        if crash_hook is not None:
            crash_hook("after_submitting")
        try:
            proc = invoke(
                expected_argv,
                capture_output=True,
                text=True,
                check=False,
                timeout=60.0,
                input=script_text,
            )
        except BaseException:
            raise
        if proc.returncode != 0:
            intent["state"] = "rejected"
            intent["last_error"] = proc.stderr.strip()[:500]
            _atomic_write_json(intent_path, intent)
            raise DispatcherError(
                f"failed to queue dispatcher successor: {intent['last_error']}"
            )
        adopted_job_id = proc.stdout.strip().split(";", 1)[0]
        if not adopted_job_id.isdigit():
            raise DispatcherError(
                "dispatcher successor sbatch returned an ambiguous job id"
            )
        if crash_hook is not None:
            crash_hook("after_acceptance")

    if adopted_job_id is None:
        raise DispatcherError(
            f"dispatcher successor transaction is blocked in state "
            f"{intent.get('state')!r}"
        )
    if accepted_held_replay:
        spool_path = Path(str(intent["spooled_path"]))
        spooled = _stable_readonly_artifact(
            spool_path,
            description="sealed dispatcher successor spooled script",
        )[1]
    else:
        spooled = _read_spooled_batch_script(
            adopted_job_id, runner=invoke
        )
        if spooled != script_raw:
            intent["state"] = "integrity_blocked"
            intent["job_id"] = adopted_job_id
            intent["last_error"] = "Slurm-spooled successor script differs"
            _atomic_write_json(intent_path, intent)
            raise DispatcherError(intent["last_error"])
    post_path, post_raw = _stable_readonly_artifact(
        lexical, description="dispatcher successor sbatch after submission"
    )
    if post_path != lexical or post_raw != script_raw:
        raise DispatcherError(
            "dispatcher successor sbatch changed across submission boundary"
        )
    spool_path = transaction_root / "SPOOLED_SCRIPT.sbatch"
    _publish_readonly_bytes_once(spool_path, spooled)
    if not accepted_held_replay:
        show = invoke(
            ["scontrol", "show", "job", "-o", adopted_job_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
        if (
            show.returncode != 0
            or "Reason=JobHeldUser" not in show.stdout
            or re.search(
                rf"Dependency=[^ ]*afterany:{re.escape(current_job)}",
                show.stdout,
            )
            is None
        ):
            raise DispatcherError(
                "dispatcher successor was not held with its exact afterany "
                "dependency"
            )
        intent.update(
            {
                "state": "accepted_held",
                "job_id": adopted_job_id,
                "spooled_path": str(spool_path.resolve()),
                "spooled_sha256": script_sha256,
            }
        )
        _atomic_write_json(intent_path, intent)

    release_argv = ["scontrol", "release", adopted_job_id]
    release_intent_path = transaction_root / "RELEASE_INTENT.json"
    release_intent = {
        "schema_version": 1,
        "kind": "schema5_dispatcher_successor_release_intent",
        "job_id": adopted_job_id,
        "current_job_id": current_job,
        "scheduler_comment": intent["scheduler_comment"],
        "dependency": f"afterany:{current_job}",
        "spooled_path": str(spool_path.resolve()),
        "spooled_sha256": script_sha256,
        "command": release_argv,
        "created_at": float(intent.get("submit_started_at") or timestamp),
    }
    _publish_readonly_json_once(release_intent_path, release_intent)
    release_intent_sha256 = _sealed_artifact_sha256(release_intent_path)
    intent["release_intent_path"] = str(release_intent_path.resolve())
    intent["release_intent_sha256"] = release_intent_sha256
    _atomic_write_json(intent_path, intent)

    def show_successor() -> subprocess.CompletedProcess[str]:
        return invoke(
            ["scontrol", "show", "job", "-o", adopted_job_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )

    def exact_scheduler_rows() -> tuple[Any, ...]:
        reader = scheduler_reader
        if reader is None:
            from slurm.schema5_control import query_scheduler

            reader = query_scheduler
        snapshot = reader()
        if (
            getattr(snapshot, "squeue_ok", False) is not True
            or getattr(snapshot, "sacct_ok", False) is not True
            or getattr(snapshot, "errors", ())
        ):
            raise DispatcherError(
                "dispatcher successor release recovery lacks complete "
                "scheduler truth"
            )
        matching = tuple(
            job
            for job in snapshot.jobs
            if _array_identity(str(job.job_id))[0] == adopted_job_id
            and str(job.comment) == intent["scheduler_comment"]
        )
        if not matching or any(
            not _successor_command_matches(
                str(job.command),
                current_job=current_job,
                scheduler_comment=intent["scheduler_comment"],
            )
            for job in matching
        ):
            raise DispatcherError(
                "dispatcher successor release scheduler provenance drifted"
            )
        return matching

    release_attempt_path = transaction_root / "RELEASE_ATTEMPT.json"
    release_result_path = transaction_root / "RELEASE_RESULT.json"
    prior_release_attempt: dict[str, Any] | None = None
    if release_attempt_path.exists():
        prior_release_attempt, _attempt_raw = (
            _strict_readonly_json_artifact(
                release_attempt_path,
                description="dispatcher successor release attempt",
            )
        )
        if (
            set(prior_release_attempt)
            != {
                "schema_version",
                "kind",
                "job_id",
                "release_intent_path",
                "release_intent_sha256",
                "command",
                "started_at",
            }
            or prior_release_attempt.get("schema_version") != 1
            or prior_release_attempt.get("kind")
            != "schema5_dispatcher_successor_release_attempt"
            or prior_release_attempt.get("job_id") != adopted_job_id
            or prior_release_attempt.get("release_intent_path")
            != str(release_intent_path.resolve())
            or prior_release_attempt.get("release_intent_sha256")
            != release_intent_sha256
            or prior_release_attempt.get("command") != release_argv
            or not isinstance(
                prior_release_attempt.get("started_at"), (int, float)
            )
            or isinstance(prior_release_attempt.get("started_at"), bool)
        ):
            raise DispatcherError(
                "dispatcher successor release attempt drifted"
            )
    prior_release_result: dict[str, Any] | None = None
    if release_result_path.exists():
        prior_release_result, _result_raw = _strict_readonly_json_artifact(
            release_result_path,
            description="dispatcher successor release result",
        )
        if (
            set(prior_release_result)
            != {
                "schema_version",
                "kind",
                "job_id",
                "attempt_sha256",
                "returncode",
                "stdout",
                "stderr",
                "completed_at",
            }
            or prior_release_result.get("schema_version") != 1
            or prior_release_result.get("kind")
            != "schema5_dispatcher_successor_release_result"
            or prior_release_result.get("job_id") != adopted_job_id
            or prior_release_attempt is None
            or prior_release_result.get("attempt_sha256")
            != _sealed_artifact_sha256(release_attempt_path)
            or prior_release_result.get("returncode") != 0
            or not isinstance(prior_release_result.get("stdout"), str)
            or not isinstance(prior_release_result.get("stderr"), str)
            or not isinstance(
                prior_release_result.get("completed_at"), (int, float)
            )
            or isinstance(prior_release_result.get("completed_at"), bool)
        ):
            raise DispatcherError(
                "dispatcher successor release result drifted"
            )
    before = show_successor()
    held = (
        before.returncode == 0
        and "Reason=JobHeldUser" in before.stdout
        and re.search(
            rf"Dependency=[^ ]*afterany:{re.escape(current_job)}",
            before.stdout,
        )
        is not None
    )
    released_this_call = False
    should_release = False
    if held and not os.path.lexists(release_attempt_path):
        _publish_readonly_json_once(
            release_attempt_path,
            {
                "schema_version": 1,
                "kind": "schema5_dispatcher_successor_release_attempt",
                "job_id": adopted_job_id,
                "release_intent_path": str(release_intent_path.resolve()),
                "release_intent_sha256": release_intent_sha256,
                "command": release_argv,
                "started_at": timestamp,
            },
        )
        should_release = True
    elif held and prior_release_result is None:
        assert prior_release_attempt is not None
        started_at = prior_release_attempt.get("started_at")
        if (
            not isinstance(started_at, (int, float))
            or isinstance(started_at, bool)
            or timestamp - float(started_at) < 300.0
        ):
            raise DispatcherError(
                "dispatcher successor release remains inside scheduler "
                "visibility grace"
            )
        should_release = True
    if should_release:
        if crash_hook is not None:
            crash_hook("before_release")
        release = invoke(
            release_argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
        released_this_call = True
        if crash_hook is not None:
            crash_hook("after_release_command")
        if release.returncode != 0:
            raise DispatcherError(
                f"failed to release exact dispatcher successor "
                f"{adopted_job_id}: {release.stderr.strip()[:500]}"
            )
        _publish_readonly_json_once(
            release_result_path,
            {
                "schema_version": 1,
                "kind": "schema5_dispatcher_successor_release_result",
                "job_id": adopted_job_id,
                "attempt_sha256": _sealed_artifact_sha256(
                    release_attempt_path
                ),
                "returncode": int(release.returncode),
                "stdout": release.stdout,
                "stderr": release.stderr,
                "completed_at": timestamp,
            },
        )
        if crash_hook is not None:
            crash_hook("after_release")
        before = show_successor()
        held = (
            before.returncode == 0
            and "Reason=JobHeldUser" in before.stdout
        )
    elif held and prior_release_attempt is not None:
        raise DispatcherError(
            "dispatcher successor release was attempted but remains held; "
            "waiting for unambiguous scheduler reconciliation"
        )
    elif not held and prior_release_attempt is None:
        raise DispatcherError(
            "dispatcher successor became unheld without its durable exact "
            "release attempt"
        )

    scheduler_rows = (
        ()
        if released_this_call and not accepted_held_replay
        else exact_scheduler_rows()
    )
    if held:
        raise DispatcherError(
            "dispatcher successor remains user-held after exact release"
        )
    if before.returncode == 0:
        if re.search(
            rf"Dependency=[^ ]*afterany:{re.escape(current_job)}",
            before.stdout,
        ) is None:
            raise DispatcherError(
                "released dispatcher successor lost its exact dependency"
            )
    elif any(bool(getattr(job, "active", False)) for job in scheduler_rows):
        raise DispatcherError(
            "active dispatcher successor disappeared from scontrol truth"
        )

    release_complete_path = transaction_root / "RELEASE_COMPLETE.json"
    if release_complete_path.exists():
        release_complete, _release_complete_raw = (
            _strict_readonly_json_artifact(
                release_complete_path,
                description="dispatcher successor release completion",
            )
        )
        release_complete_sha256 = _verify_successor_release_complete(
            release_complete_path,
            job_id=adopted_job_id,
            release_intent_path=release_intent_path,
            release_intent_sha256=release_intent_sha256,
        )
    else:
        release_complete = {
            "schema_version": 1,
            "kind": "schema5_dispatcher_successor_release_complete",
            "job_id": adopted_job_id,
            "release_intent_path": str(release_intent_path.resolve()),
            "release_intent_sha256": release_intent_sha256,
            "release_attempt_path": (
                str(release_attempt_path.resolve())
                if release_attempt_path.exists()
                else None
            ),
            "release_attempt_sha256": (
                _sealed_artifact_sha256(release_attempt_path)
                if release_attempt_path.exists()
                else None
            ),
            "release_result_path": (
                str(release_result_path.resolve())
                if release_result_path.exists()
                else None
            ),
            "release_result_sha256": (
                _sealed_artifact_sha256(release_result_path)
                if release_result_path.exists()
                else None
            ),
            "scheduler_job_ids": (
                sorted({str(job.job_id) for job in scheduler_rows})
                or [adopted_job_id]
            ),
            "scheduler_states": sorted(
                {str(getattr(job, "state", "")) for job in scheduler_rows}
            ),
            "released_this_call": released_this_call,
            "completed_at": timestamp,
        }
        _publish_readonly_json_once(
            release_complete_path, release_complete
        )
        release_complete_sha256 = _verify_successor_release_complete(
            release_complete_path,
            job_id=adopted_job_id,
            release_intent_path=release_intent_path,
            release_intent_sha256=release_intent_sha256,
        )
    intent.update(
        {
            "state": "committed",
            "released_at": timestamp,
            "release_complete_path": str(release_complete_path.resolve()),
            "release_complete_sha256": release_complete_sha256,
        }
    )
    _atomic_write_json(intent_path, intent)
    return adopted_job_id


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
    receipt_path = sbatch_path.with_suffix(".spooled.json")
    _receipt_lexical, receipt_raw = _stable_readonly_artifact(
        receipt_path,
        description=f"submission {batch_id} spooled-script receipt",
    )
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    ledger["jobs"][job_id] = {
        "job_id": job_id,
        "batch_id": batch_id,
        "batch_manifest": str(manifest_path),
        "batch_manifest_sha256": str(intent.get("batch_manifest_sha256")),
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": str(intent.get("sbatch_sha256")),
        "spooled_sbatch_sha256": str(intent.get("sbatch_sha256")),
        "spooled_receipt_path": str(receipt_path),
        "spooled_receipt_sha256": receipt_sha256,
        "submission_transport": STDIN_EXACT_SUBMISSION_TRANSPORT,
        "submission_argv_sha256": _stdin_submission_argv_sha256(batch_id),
        "submitted_at": now,
        "last_seen_at": now,
        "state": "submitted",
        "task_count": len(tasks),
        "tasks": tasks,
    }
    if batch_id in ledger.get("intents", {}):
        ledger["intents"][batch_id].update(
            {
                "state": "submitted",
                "job_id": job_id,
                "submitted_at": now,
                "spooled_sbatch_sha256": str(
                    intent.get("sbatch_sha256")
                ),
                "spooled_receipt_path": str(receipt_path),
                "spooled_receipt_sha256": receipt_sha256,
            }
        )
        ledger["intents"][batch_id].pop("error", None)
        ledger["intents"][batch_id].pop("last_submit_error_at", None)
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
DISPATCHER_SUBMISSION_INTEGRITY_ALERT = "dispatcher:submission-integrity"


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


def _persist_dispatcher_submission_integrity_hold(
    state_dir: Path,
    *,
    message: str,
    now: float,
) -> None:
    """Latch artifact drift as a human-acknowledged scientific-integrity hold."""

    from slurm.schema5_control import record_alert

    record_alert(
        state_dir,
        kind="dispatcher-submission-integrity",
        severity="critical",
        message=message,
        dedupe_key=DISPATCHER_SUBMISSION_INTEGRITY_ALERT,
        send_email=True,
        now=now,
    )


def _strict_readonly_json_artifact(
    path: Path, *, description: str
) -> tuple[dict[str, Any], bytes]:
    _lexical, raw = _stable_readonly_artifact(path, description=description)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise DispatcherError(
                    f"{description} repeats JSON field {key!r}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DispatcherError(
                    f"{description} contains non-finite number {token!r}"
                )
            ),
        )
    except DispatcherError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DispatcherError(f"cannot parse {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise DispatcherError(f"{description} must be one JSON object")
    return value, raw


def _integrity_retirement_root(state_dir: Path, batch_id: str) -> Path:
    if (
        not isinstance(batch_id, str)
        or not batch_id
        or re.fullmatch(r"[A-Za-z0-9_.-]+", batch_id) is None
    ):
        raise DispatcherError("integrity retirement batch ID is unsafe")
    lexical_state = Path(
        os.path.abspath(os.fspath(state_dir.expanduser()))
    )
    return (
        lexical_state / INTEGRITY_RETIREMENT_DIRNAME / batch_id
    )


def _ensure_integrity_retirement_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        if (
            path.is_symlink()
            or not path.is_dir()
            or path.resolve(strict=True) != path
        ):
            raise DispatcherError(
                f"integrity retirement directory is unsafe: {path}"
            )
    except (OSError, RuntimeError) as exc:
        raise DispatcherError(
            f"cannot verify integrity retirement directory {path}: {exc}"
        ) from exc


def _seal_integrity_retirement_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise DispatcherError("integrity retirement tree is missing or unsafe")
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise DispatcherError(
                f"integrity retirement tree contains symlink {path}"
            )
        if path.is_file():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        elif path.is_dir():
            path.chmod(0o555)
        else:
            raise DispatcherError(
                f"integrity retirement tree contains special object {path}"
            )
    root.chmod(0o555)
    directory_fd = os.open(root.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _blocked_integrity_runtime_identities(
    intent: Mapping[str, Any],
) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    tasks = intent.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise DispatcherError("integrity-blocked intent has no coordinates")
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            raise DispatcherError(
                f"integrity-blocked task {index} is not an object"
            )
        environment = _runtime_environment(task)
        if not environment:
            raise DispatcherError(
                "production integrity retirement requires schema-5 runtime identity"
            )
        identities.append(
            {
                "task_index": index,
                "run_id": str(task.get("run_id", "")),
                "cell_id": str(task.get("cell_id", "")),
                "serving_profile": str(task.get("serving_profile", "")),
                "runtime_environment": {
                    key: environment[key]
                    for key in sorted(PRODUCTION_ENVIRONMENT_KEYS)
                },
            }
        )
    return identities


def _current_integrity_runtime_identities(
    control_state: Mapping[str, Any],
    blocked: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from slurm import schema5_control as control

    current: list[dict[str, Any]] = []
    for identity in blocked:
        run_id = str(identity["run_id"])
        environment = control.production_environment(
            control_state, run_id=run_id
        )
        missing = sorted(PRODUCTION_ENVIRONMENT_KEYS - set(environment))
        if missing:
            raise DispatcherError(
                "current schema-5 runtime identity is incomplete: "
                + ", ".join(missing)
            )
        current.append(
            {
                "task_index": int(identity["task_index"]),
                "run_id": run_id,
                "cell_id": str(identity["cell_id"]),
                "serving_profile": str(identity["serving_profile"]),
                "runtime_environment": {
                    key: str(environment[key])
                    for key in sorted(PRODUCTION_ENVIRONMENT_KEYS)
                },
            }
        )
    return current


def _integrity_identity_changes(
    blocked: Sequence[Mapping[str, Any]],
    current: Sequence[Mapping[str, Any]],
) -> list[str]:
    if len(blocked) != len(current):
        raise DispatcherError("integrity retirement task identity cardinality drifted")
    changes: list[str] = []
    for old, new in zip(blocked, current, strict=True):
        if (
            old.get("task_index") != new.get("task_index")
            or old.get("run_id") != new.get("run_id")
            or old.get("cell_id") != new.get("cell_id")
        ):
            raise DispatcherError(
                "integrity retirement coordinate identity changed unexpectedly"
            )
        task_changes: list[str] = []
        if old.get("serving_profile") != new.get("serving_profile"):
            task_changes.append("serving_profile")
        old_environment = old.get("runtime_environment")
        new_environment = new.get("runtime_environment")
        if not isinstance(old_environment, Mapping) or not isinstance(
            new_environment, Mapping
        ):
            raise DispatcherError("integrity retirement runtime identity is malformed")
        task_changes.extend(
            key
            for key in sorted(PRODUCTION_ENVIRONMENT_KEYS)
            if old_environment.get(key) != new_environment.get(key)
        )
        if not task_changes:
            raise DispatcherError(
                "integrity-blocked coordinates cannot be retired until their "
                "release, rollout, fleet, environment, policy, or profile identity "
                "changes"
            )
        prefix = (
            f"{int(old['task_index'])}:{old['run_id']}:{old['cell_id']}:"
        )
        changes.extend(prefix + field for field in task_changes)
    return sorted(set(changes))


def _validated_integrity_retirement_semantic_evidence(
    control_state_dir: Path,
    *,
    evidence_path: Path,
    expected_sha256: str,
    blocked_at: float,
) -> dict[str, Any]:
    from slurm import schema5_control as control

    report, observation, observed_sha256 = control._read_admission_ramp_evidence(
        control_state_dir,
        evidence_path,
        expected_sha256,
    )
    if (
        observation.get("cadence") not in {"semantic", "daily"}
        or observation.get("semantic_integrity_clean") is not True
        or float(observation.get("committed_timestamp", -1)) < blocked_at
    ):
        raise DispatcherError(
            "integrity retirement requires a clean sealed semantic scan after "
            "the blocked admission incident"
        )
    return {
        "path": str(evidence_path.expanduser().resolve()),
        "sha256": observed_sha256,
        "captured_timestamp": float(observation["captured_timestamp"]),
        "committed_timestamp": float(observation["committed_timestamp"]),
        "cadence": str(observation["cadence"]),
        "control_immutable_sha256": str(
            observation["control_immutable_sha256"]
        ),
        "rollout_generation": int(observation["rollout_generation"]),
        "fleet_generation": str(observation["fleet_generation"]),
        "report_sha256": hashlib.sha256(
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def _bind_integrity_retirement_semantic_to_current_control(
    control_state: Mapping[str, Any],
    semantic_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a clean scan to the exact post-remediation control/fleet identity."""

    immutable_sha256 = str(control_state.get("immutable_sha256", ""))
    rollout_generation = control_state.get("rollout_generation")
    if (
        re.fullmatch(r"[0-9a-f]{64}", immutable_sha256) is None
        or not isinstance(rollout_generation, int)
        or isinstance(rollout_generation, bool)
        or rollout_generation < 1
        or semantic_evidence.get("control_immutable_sha256")
        != immutable_sha256
        or semantic_evidence.get("rollout_generation")
        != rollout_generation
    ):
        raise DispatcherError(
            "integrity retirement semantic scan does not bind the current "
            "immutable control and rollout generation"
        )

    open_epochs = [
        epoch
        for epoch in control_state.get("throughput_epochs", [])
        if isinstance(epoch, Mapping) and epoch.get("closed_at") is None
    ]
    if len(open_epochs) != 1:
        raise DispatcherError(
            "integrity retirement requires exactly one current throughput epoch"
        )
    epoch = open_epochs[0]
    fleet_generation = semantic_evidence.get("fleet_generation")
    if (
        not isinstance(fleet_generation, str)
        or not fleet_generation
        or epoch.get("fleet_generation") != fleet_generation
        or epoch.get("rollout_generation") != rollout_generation
        or not isinstance(epoch.get("started_timestamp"), (int, float))
        or isinstance(epoch.get("started_timestamp"), bool)
    ):
        raise DispatcherError(
            "integrity retirement semantic scan does not bind the current "
            "fleet generation"
        )

    ramp = control_state.get("admission_ramp")
    last_observation = (
        ramp.get("last_observation") if isinstance(ramp, Mapping) else None
    )
    if (
        not isinstance(last_observation, Mapping)
        or last_observation.get("path") != semantic_evidence.get("path")
        or last_observation.get("sha256") != semantic_evidence.get("sha256")
        or last_observation.get("cadence") != semantic_evidence.get("cadence")
        or last_observation.get("captured_timestamp")
        != semantic_evidence.get("captured_timestamp")
        or last_observation.get("timestamp")
        != semantic_evidence.get("committed_timestamp")
    ):
        raise DispatcherError(
            "integrity retirement semantic scan is not the current sealed "
            "control observation"
        )

    identity_timestamps: list[float] = []
    created_timestamp = control_state.get("created_timestamp")
    if (
        isinstance(created_timestamp, (int, float))
        and not isinstance(created_timestamp, bool)
        and math.isfinite(float(created_timestamp))
    ):
        identity_timestamps.append(float(created_timestamp))
    resume_intent = control_state.get("resume_intent")
    if (
        isinstance(resume_intent, Mapping)
        and resume_intent.get("rollout_generation") == rollout_generation
        and isinstance(resume_intent.get("created_timestamp"), (int, float))
        and not isinstance(resume_intent.get("created_timestamp"), bool)
    ):
        identity_timestamps.append(float(resume_intent["created_timestamp"]))
    capacity = control_state.get("capacity")
    current_contract = (
        capacity.get("current_contract")
        if isinstance(capacity, Mapping)
        else None
    )
    if isinstance(current_contract, Mapping):
        activated_timestamp = current_contract.get("activated_timestamp")
        if (
            not isinstance(activated_timestamp, (int, float))
            or isinstance(activated_timestamp, bool)
            or not math.isfinite(float(activated_timestamp))
        ):
            raise DispatcherError(
                "current capacity contract lacks its activation timestamp"
            )
        identity_timestamps.append(float(activated_timestamp))
    if not identity_timestamps:
        raise DispatcherError(
            "current post-remediation control identity has no durable start time"
        )
    identity_effective_timestamp = max(identity_timestamps)
    captured = float(semantic_evidence["captured_timestamp"])
    committed = float(semantic_evidence["committed_timestamp"])
    if (
        captured < identity_effective_timestamp
        or committed < identity_effective_timestamp
        or committed < float(epoch["started_timestamp"])
    ):
        raise DispatcherError(
            "integrity retirement semantic scan predates the current "
            "post-remediation identity"
        )
    return {
        **copy.deepcopy(dict(semantic_evidence)),
        "identity_effective_timestamp": identity_effective_timestamp,
        "throughput_epoch": int(epoch["epoch"]),
        "throughput_epoch_started_timestamp": float(
            epoch["started_timestamp"]
        ),
    }


def _matching_integrity_acknowledgement(
    control_state: Mapping[str, Any],
    *,
    note: str,
    blocked_at: float,
    semantic_committed_at: float,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for row in control_state.get("transition_history", []):
        details = row.get("details") if isinstance(row, Mapping) else None
        timestamp = row.get("timestamp") if isinstance(row, Mapping) else None
        if (
            row.get("event") != "admission_safety_hold_acknowledged"
            or not isinstance(details, Mapping)
            or not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or float(timestamp) < max(blocked_at, semantic_committed_at)
            or details.get("operator_note") != note
            or DISPATCHER_SUBMISSION_INTEGRITY_ALERT
            not in details.get("previous_reasons", [])
        ):
            continue
        matches.append(
            {
                "timestamp": float(timestamp),
                "at": str(row.get("at", "")),
                "operator_note": note,
                "transition_sha256": hashlib.sha256(
                    (
                        json.dumps(row, sort_keys=True, separators=(",", ":"))
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    if len(matches) > 1:
        raise DispatcherError(
            "multiple integrity acknowledgements ambiguously match this incident"
        )
    return None if not matches else matches[0]


def _ensure_integrity_retirement_acknowledgement(
    control_state_dir: Path,
    *,
    note: str,
    blocked_at: float,
    semantic_evidence: Mapping[str, Any],
    now: float,
) -> dict[str, Any]:
    """Use the ordinary integrity-hold workflow and return its exact audit binding."""

    from slurm import schema5_control as control

    current = control.load_control(control_state_dir, verify_files=True)
    prior = _matching_integrity_acknowledgement(
        current,
        note=note,
        blocked_at=blocked_at,
        semantic_committed_at=float(
            semantic_evidence["committed_timestamp"]
        ),
    )
    if prior is not None:
        return prior
    hold = current.get("admission_safety_hold")
    if (
        not isinstance(hold, Mapping)
        or hold.get("active") is not True
        or DISPATCHER_SUBMISSION_INTEGRITY_ALERT
        not in hold.get("reasons", [])
    ):
        raise DispatcherError(
            "integrity retirement lacks an active or previously acknowledged "
            "dispatcher integrity hold"
        )
    control.resolve_alert(
        control_state_dir,
        dedupe_key=DISPATCHER_SUBMISSION_INTEGRITY_ALERT,
        now=now,
    )
    control.update_admission_safety_hold(
        control_state_dir,
        semantic_scan_clean=True,
        now=now,
    )
    control.acknowledge_admission_hold(
        control_state_dir,
        note=note,
        now=now,
    )
    current = control.load_control(control_state_dir, verify_files=True)
    acknowledgement = _matching_integrity_acknowledgement(
        current,
        note=note,
        blocked_at=blocked_at,
        semantic_committed_at=float(
            semantic_evidence["committed_timestamp"]
        ),
    )
    if acknowledgement is None:
        raise DispatcherError(
            "integrity-hold acknowledgement was not durably recorded"
        )
    return acknowledgement


def _integrity_scheduler_absence(
    batch_id: str,
    *,
    scheduler_snapshot: Any,
) -> dict[str, Any]:
    rows, live_ids = _complete_scheduler_rows(
        scheduler_snapshot,
        observation=f"integrity retirement {batch_id}",
    )
    expected_name = f"asys-dispatch-{batch_id[-10:]}"
    expected_comment = f"asys-schema5-intent:{batch_id}"
    related: list[str] = []
    normalized_jobs: list[dict[str, Any]] = []
    for job in tuple(getattr(scheduler_snapshot, "jobs", ())):
        row = {
            "job_id": str(getattr(job, "job_id", "")),
            "source": str(getattr(job, "source", "")),
            "active": bool(getattr(job, "active", False)),
            "job_name": str(getattr(job, "job_name", "")),
            "comment": str(getattr(job, "comment", "")),
            "command": str(getattr(job, "command", "")),
        }
        normalized_jobs.append(row)
        if (
            row["job_name"] == expected_name
            or row["comment"] == expected_comment
            or _command_binds_stdin_submission(row["command"], batch_id)
        ):
            related.append(f"{row['source']}:{row['job_id']}")
    if related:
        raise DispatcherError(
            "integrity retirement cannot prove the blocked admission absent "
            "from joined scheduler truth: " + ", ".join(sorted(related))
        )
    normalized_jobs.sort(
        key=lambda row: (
            row["job_id"],
            row["source"],
            row["job_name"],
            row["comment"],
            row["command"],
        )
    )
    captured_at = float(getattr(scheduler_snapshot, "captured_at"))
    return {
        "schema_version": INTEGRITY_RETIREMENT_SCHEMA_VERSION,
        "kind": "schema5_dispatch_integrity_scheduler_absence",
        "batch_id": batch_id,
        "expected_job_name": expected_name,
        "expected_scheduler_comment": expected_comment,
        "expected_submission_argv": _stdin_submission_argv(batch_id),
        "expected_submission_argv_sha256": _stdin_submission_argv_sha256(
            batch_id
        ),
        "captured_at": captured_at,
        "squeue_ok": True,
        "sacct_ok": True,
        "related_job_ids": [],
        "live_job_ids": list(live_ids),
        "live_row_count": len(rows),
        "joined_scheduler_identity_sha256": hashlib.sha256(
            json.dumps(
                normalized_jobs,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _verify_integrity_retirement_receipt(
    batch_id: str,
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = Path(str(intent.get("retirement_receipt_path", "")))
    receipt, receipt_raw = _strict_readonly_json_artifact(
        receipt_path,
        description=f"integrity retirement receipt {batch_id}",
    )
    retirement_root = receipt_path.parent
    expected_root = _integrity_retirement_root(
        retirement_root.parent.parent, batch_id
    )
    if retirement_root != expected_root:
        raise DispatcherError(
            f"integrity retirement receipt {batch_id} escaped its state root"
        )
    required = {
        "schema_version",
        "kind",
        "retirement_id",
        "batch_id",
        "retirement_intent_path",
        "retirement_intent_sha256",
        "scheduler_absence_path",
        "scheduler_absence_sha256",
        "semantic_evidence",
        "acknowledgement",
        "identity_changes",
        "archived_preimages",
        "completed_at",
        "completed_timestamp",
        "completion_id",
    }
    stable = dict(receipt)
    completion_id = stable.pop("completion_id", None)
    if (
        set(receipt) != required
        or receipt.get("schema_version")
        != INTEGRITY_RETIREMENT_SCHEMA_VERSION
        or receipt.get("kind")
        != "schema5_dispatch_integrity_retirement"
        or receipt.get("batch_id") != batch_id
        or receipt.get("retirement_id") != intent.get("retirement_id")
        or completion_id
        != hashlib.sha256(
            json.dumps(
                stable, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        or hashlib.sha256(receipt_raw).hexdigest()
        != intent.get("retirement_receipt_sha256")
    ):
        raise DispatcherError(
            f"integrity retirement receipt {batch_id} identity drifted"
        )
    root_objects = (
        (
            "retirement intent",
            receipt.get("retirement_intent_path"),
            receipt.get("retirement_intent_sha256"),
        ),
        (
            "scheduler absence",
            receipt.get("scheduler_absence_path"),
            receipt.get("scheduler_absence_sha256"),
        ),
    )
    loaded_root_objects: dict[str, dict[str, Any]] = {}
    for description, path_value, expected_sha256 in root_objects:
        path = Path(str(path_value))
        if path.parent != retirement_root:
            raise DispatcherError(
                f"integrity retirement {description} escaped its archive"
            )
        value, raw = _strict_readonly_json_artifact(
            path, description=f"{description} {batch_id}"
        )
        loaded_root_objects[description] = value
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise DispatcherError(
                f"integrity retirement {description} hash drifted"
            )
    retirement_intent = loaded_root_objects["retirement intent"]
    scheduler_absence = loaded_root_objects["scheduler absence"]
    if (
        retirement_intent.get("kind")
        != "schema5_dispatch_integrity_retirement_intent"
        or retirement_intent.get("batch_id") != batch_id
        or retirement_intent.get("retirement_id")
        != receipt.get("retirement_id")
        or scheduler_absence.get("kind")
        != "schema5_dispatch_integrity_scheduler_absence"
        or scheduler_absence.get("batch_id") != batch_id
        or scheduler_absence.get("related_job_ids") != []
        or scheduler_absence.get("squeue_ok") is not True
        or scheduler_absence.get("sacct_ok") is not True
        or scheduler_absence.get("expected_submission_argv_sha256")
        != _stdin_submission_argv_sha256(batch_id)
    ):
        raise DispatcherError(
            f"integrity retirement {batch_id} evidence contract drifted"
        )
    semantic = receipt.get("semantic_evidence")
    semantic_required = {
        "path",
        "sha256",
        "captured_timestamp",
        "committed_timestamp",
        "cadence",
        "control_immutable_sha256",
        "rollout_generation",
        "fleet_generation",
        "report_sha256",
        "identity_effective_timestamp",
        "throughput_epoch",
        "throughput_epoch_started_timestamp",
    }
    if (
        not isinstance(semantic, Mapping)
        or set(semantic) != semantic_required
        or retirement_intent.get("semantic_evidence") != semantic
        or not isinstance(semantic.get("path"), str)
        or not Path(semantic["path"]).is_absolute()
        or re.fullmatch(r"[0-9a-f]{64}", str(semantic.get("sha256", "")))
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(semantic.get("control_immutable_sha256", "")),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(semantic.get("report_sha256", ""))
        )
        is None
        or semantic.get("cadence") not in {"semantic", "daily"}
        or not isinstance(semantic.get("rollout_generation"), int)
        or isinstance(semantic.get("rollout_generation"), bool)
        or semantic["rollout_generation"] < 1
        or not isinstance(semantic.get("fleet_generation"), str)
        or not semantic["fleet_generation"]
        or not isinstance(semantic.get("throughput_epoch"), int)
        or isinstance(semantic.get("throughput_epoch"), bool)
        or semantic["throughput_epoch"] < 1
        or any(
            not isinstance(semantic.get(field), (int, float))
            or isinstance(semantic.get(field), bool)
            or not math.isfinite(float(semantic[field]))
            for field in (
                "captured_timestamp",
                "committed_timestamp",
                "identity_effective_timestamp",
                "throughput_epoch_started_timestamp",
            )
        )
        or float(semantic["captured_timestamp"])
        < float(semantic["identity_effective_timestamp"])
        or float(semantic["committed_timestamp"])
        < float(semantic["captured_timestamp"])
        or float(semantic["committed_timestamp"])
        < float(semantic["throughput_epoch_started_timestamp"])
    ):
        raise DispatcherError(
            f"integrity retirement {batch_id} semantic binding is invalid"
        )
    _semantic_value, semantic_raw = _strict_readonly_json_artifact(
        Path(semantic["path"]),
        description=f"integrity retirement semantic evidence {batch_id}",
    )
    if hashlib.sha256(semantic_raw).hexdigest() != semantic["sha256"]:
        raise DispatcherError(
            f"integrity retirement {batch_id} semantic evidence drifted"
        )
    archived = receipt.get("archived_preimages")
    if not isinstance(archived, list) or len(archived) != 3:
        raise DispatcherError(
            f"integrity retirement {batch_id} has incomplete preimages"
        )
    observed_names: set[str] = set()
    for record in archived:
        if not isinstance(record, Mapping) or set(record) != {
            "logical_name",
            "source_path",
            "source_sha256",
            "source_size",
            "archive_path",
            "archive_sha256",
            "archive_size",
        }:
            raise DispatcherError(
                f"integrity retirement {batch_id} preimage record is malformed"
            )
        name = str(record["logical_name"])
        archive_path = Path(str(record["archive_path"]))
        if (
            name in observed_names
            or archive_path.parent != retirement_root / "preimages"
            or record["archive_sha256"] != record["source_sha256"]
            or record["archive_size"] != record["source_size"]
        ):
            raise DispatcherError(
                f"integrity retirement {batch_id} preimage identity drifted"
            )
        observed_names.add(name)
        _lexical, raw = _stable_readonly_artifact(
            archive_path,
            description=f"integrity retirement {name} preimage",
        )
        if (
            hashlib.sha256(raw).hexdigest() != record["archive_sha256"]
            or len(raw) != record["archive_size"]
        ):
            raise DispatcherError(
                f"integrity retirement {batch_id} preimage hash drifted"
            )
    if observed_names != {"blocked_intent", "batch_manifest", "sbatch"}:
        raise DispatcherError(
            f"integrity retirement {batch_id} preimage set drifted"
        )
    for directory in (retirement_root, retirement_root / "preimages"):
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or stat.S_IMODE(directory.stat().st_mode) & 0o222
        ):
            raise DispatcherError(
                f"integrity retirement {batch_id} archive is not sealed"
            )
    return receipt


def retire_integrity_blocked_intent(
    state_dir: Path,
    *,
    control_state_dir: Path,
    batch_id: str,
    semantic_evidence_path: Path,
    semantic_evidence_sha256: str,
    operator_note: str,
    scheduler_reader: Any | None = None,
    now: float | None = None,
    crash_hook: Any | None = None,
) -> dict[str, Any]:
    """Evidence-preservingly retire one deterministic pre-sbatch failure."""

    from slurm import schema5_control as control

    if not isinstance(operator_note, str) or not operator_note.strip():
        raise DispatcherError("integrity retirement requires an operator note")
    note = operator_note.strip()
    timestamp = time.time() if now is None else float(now)
    state_dir = Path(
        os.path.abspath(os.fspath(state_dir.expanduser()))
    )
    control_state_dir = Path(
        os.path.abspath(os.fspath(control_state_dir.expanduser()))
    )
    ledger_path = state_dir / "ledger.json"
    initial = load_production_ledger(ledger_path)
    initial_intent = initial.get("intents", {}).get(batch_id)
    if not isinstance(initial_intent, Mapping):
        raise DispatcherError(f"unknown dispatcher intent {batch_id!r}")
    if initial_intent.get("state") == "integrity_retired":
        return _verify_integrity_retirement_receipt(batch_id, initial_intent)
    if initial_intent.get("state") != "integrity_blocked":
        raise DispatcherError(
            f"dispatcher intent {batch_id!r} is not integrity-blocked"
        )
    blocked_at = float(initial_intent["integrity_blocked_at"])
    semantic_source = _validated_integrity_retirement_semantic_evidence(
        control_state_dir,
        evidence_path=semantic_evidence_path,
        expected_sha256=semantic_evidence_sha256,
        blocked_at=blocked_at,
    )
    acknowledgement = _ensure_integrity_retirement_acknowledgement(
        control_state_dir,
        note=note,
        blocked_at=blocked_at,
        semantic_evidence=semantic_source,
        now=timestamp,
    )

    with global_dispatcher_lock(state_dir.parent):
        with singleton_lock(state_dir):
            with control.admission_boundary_lock(control_state_dir):
                ledger = load_production_ledger(ledger_path)
                blocked_intent = ledger.get("intents", {}).get(batch_id)
                if not isinstance(blocked_intent, Mapping):
                    raise DispatcherError(
                        f"dispatcher intent {batch_id!r} disappeared"
                    )
                if blocked_intent.get("state") == "integrity_retired":
                    return _verify_integrity_retirement_receipt(
                        batch_id, blocked_intent
                    )
                if blocked_intent.get("state") != "integrity_blocked":
                    raise DispatcherError(
                        f"dispatcher intent {batch_id!r} changed state"
                    )
                if dict(blocked_intent) != dict(initial_intent):
                    raise DispatcherError(
                        "integrity-blocked intent changed during remediation"
                    )
                control_state = control.load_control(
                    control_state_dir, verify_files=True
                )
                fresh_semantic_source = (
                    _validated_integrity_retirement_semantic_evidence(
                        control_state_dir,
                        evidence_path=semantic_evidence_path,
                        expected_sha256=semantic_evidence_sha256,
                        blocked_at=blocked_at,
                    )
                )
                if fresh_semantic_source != semantic_source:
                    raise DispatcherError(
                        "integrity retirement semantic evidence changed before "
                        "the admission-locked transaction"
                    )
                semantic = (
                    _bind_integrity_retirement_semantic_to_current_control(
                        control_state,
                        fresh_semantic_source,
                    )
                )
                confirmed_ack = _matching_integrity_acknowledgement(
                    control_state,
                    note=note,
                    blocked_at=blocked_at,
                    semantic_committed_at=float(
                        semantic["committed_timestamp"]
                    ),
                )
                if confirmed_ack != acknowledgement:
                    raise DispatcherError(
                        "integrity acknowledgement changed before retirement"
                    )
                blocked_identities = _blocked_integrity_runtime_identities(
                    blocked_intent
                )
                current_identities = _current_integrity_runtime_identities(
                    control_state, blocked_identities
                )
                identity_changes = _integrity_identity_changes(
                    blocked_identities, current_identities
                )
                manifest_path, manifest_raw = _stable_readonly_artifact(
                    Path(str(blocked_intent["batch_manifest"])),
                    description=f"blocked batch {batch_id} manifest",
                )
                sbatch_path, sbatch_raw = _stable_readonly_artifact(
                    Path(str(blocked_intent["sbatch_path"])),
                    description=f"blocked batch {batch_id} sbatch",
                )
                blocked_raw = (
                    json.dumps(
                        dict(blocked_intent), indent=2, sort_keys=True
                    )
                    + "\n"
                ).encode("utf-8")
                source_preimages = (
                    (
                        "blocked_intent",
                        str(ledger_path),
                        blocked_raw,
                        "blocked_intent.json",
                    ),
                    (
                        "batch_manifest",
                        str(manifest_path),
                        manifest_raw,
                        "batch_manifest.preimage",
                    ),
                    (
                        "sbatch",
                        str(sbatch_path),
                        sbatch_raw,
                        "sbatch.preimage",
                    ),
                )
                root = _integrity_retirement_root(state_dir, batch_id)
                _ensure_integrity_retirement_directory(root)
                preimages_root = root / "preimages"
                _ensure_integrity_retirement_directory(preimages_root)
                source_records = [
                    {
                        "logical_name": logical_name,
                        "source_path": source_path,
                        "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "source_size": len(raw),
                        "archive_path": str(preimages_root / archive_name),
                    }
                    for logical_name, source_path, raw, archive_name
                    in source_preimages
                ]
                intent_identity = {
                    "schema_version": INTEGRITY_RETIREMENT_SCHEMA_VERSION,
                    "kind": "schema5_dispatch_integrity_retirement_intent",
                    "batch_id": batch_id,
                    "blocked_at": blocked_at,
                    "blocked_intent_sha256": hashlib.sha256(
                        blocked_raw
                    ).hexdigest(),
                    "source_preimages": source_records,
                    "semantic_evidence": semantic,
                    "acknowledgement": acknowledgement,
                    "operator_note": note,
                    "operator_note_sha256": hashlib.sha256(
                        note.encode("utf-8")
                    ).hexdigest(),
                    "blocked_runtime_identities": blocked_identities,
                    "current_runtime_identities": current_identities,
                    "identity_changes": identity_changes,
                    "control_immutable_sha256": str(
                        control_state["immutable_sha256"]
                    ),
                    "control_rollout_generation": int(
                        control_state["rollout_generation"]
                    ),
                    "expected_scheduler_comment": (
                        f"asys-schema5-intent:{batch_id}"
                    ),
                    "expected_submission_argv_sha256": (
                        _stdin_submission_argv_sha256(batch_id)
                    ),
                }
                retirement_id = hashlib.sha256(
                    json.dumps(
                        intent_identity,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                retirement_intent = {
                    **intent_identity,
                    "retirement_id": retirement_id,
                    # A retry must render the exact same marker-first intent.
                    "created_at": float(acknowledgement["timestamp"]),
                }
                retirement_intent_path = (
                    root / INTEGRITY_RETIREMENT_INTENT_FILENAME
                )
                _publish_readonly_json_once(
                    retirement_intent_path, retirement_intent
                )
                observed_retirement_intent, retirement_intent_raw = (
                    _strict_readonly_json_artifact(
                        retirement_intent_path,
                        description=f"integrity retirement intent {batch_id}",
                    )
                )
                if observed_retirement_intent != retirement_intent:
                    raise DispatcherError(
                        "integrity retirement intent conflicts with replay"
                    )
                if crash_hook is not None:
                    crash_hook("after_intent")
                archived_records: list[dict[str, Any]] = []
                for (
                    logical_name,
                    source_path,
                    raw,
                    archive_name,
                ) in source_preimages:
                    archive_path = preimages_root / archive_name
                    _publish_readonly_bytes_once(archive_path, raw)
                    _lexical, archived_raw = _stable_readonly_artifact(
                        archive_path,
                        description=(
                            f"integrity retirement {logical_name} preimage"
                        ),
                    )
                    if archived_raw != raw:
                        raise DispatcherError(
                            f"integrity retirement {logical_name} preimage drifted"
                        )
                    digest = hashlib.sha256(raw).hexdigest()
                    archived_records.append(
                        {
                            "logical_name": logical_name,
                            "source_path": source_path,
                            "source_sha256": digest,
                            "source_size": len(raw),
                            "archive_path": str(archive_path),
                            "archive_sha256": digest,
                            "archive_size": len(raw),
                        }
                    )
                archived_records.sort(key=lambda row: row["logical_name"])
                if crash_hook is not None:
                    crash_hook("after_preimages")
                snapshot = (
                    control.query_scheduler(tolerate_errors=False)
                    if scheduler_reader is None
                    else scheduler_reader()
                )
                fresh_scheduler_absence = _integrity_scheduler_absence(
                    batch_id, scheduler_snapshot=snapshot
                )
                scheduler_path = (
                    root / INTEGRITY_RETIREMENT_SCHEDULER_FILENAME
                )
                if os.path.lexists(scheduler_path):
                    scheduler_absence, scheduler_raw = (
                        _strict_readonly_json_artifact(
                            scheduler_path,
                            description=(
                                "integrity retirement scheduler absence "
                                f"{batch_id}"
                            ),
                        )
                    )
                    stable_scheduler_fields = {
                        "schema_version",
                        "kind",
                        "batch_id",
                        "expected_job_name",
                        "expected_scheduler_comment",
                        "expected_submission_argv",
                        "expected_submission_argv_sha256",
                        "squeue_ok",
                        "sacct_ok",
                        "related_job_ids",
                    }
                    if any(
                        scheduler_absence.get(field)
                        != fresh_scheduler_absence.get(field)
                        for field in stable_scheduler_fields
                    ):
                        raise DispatcherError(
                            "integrity retirement scheduler evidence identity "
                            "drifted on replay"
                        )
                else:
                    scheduler_absence = fresh_scheduler_absence
                    _publish_readonly_json_once(
                        scheduler_path, scheduler_absence
                    )
                    observed_scheduler, scheduler_raw = (
                        _strict_readonly_json_artifact(
                            scheduler_path,
                            description=(
                                "integrity retirement scheduler absence "
                                f"{batch_id}"
                            ),
                        )
                    )
                    if observed_scheduler != scheduler_absence:
                        raise DispatcherError(
                            "integrity retirement scheduler evidence conflicts "
                            "with replay"
                        )
                if crash_hook is not None:
                    crash_hook("after_scheduler_absence")
                # Re-read every source after the external observation.  A path
                # replacement cannot inherit authority from the archived preimage.
                if (
                    _stable_readonly_artifact(
                        manifest_path,
                        description=f"blocked batch {batch_id} manifest replay",
                    )[1]
                    != manifest_raw
                    or _stable_readonly_artifact(
                        sbatch_path,
                        description=f"blocked batch {batch_id} sbatch replay",
                    )[1]
                    != sbatch_raw
                ):
                    raise DispatcherError(
                        "blocked dispatcher artifacts changed during retirement"
                    )
                final_semantic_source = (
                    _validated_integrity_retirement_semantic_evidence(
                        control_state_dir,
                        evidence_path=semantic_evidence_path,
                        expected_sha256=semantic_evidence_sha256,
                        blocked_at=blocked_at,
                    )
                )
                final_semantic = (
                    _bind_integrity_retirement_semantic_to_current_control(
                        control_state,
                        final_semantic_source,
                    )
                )
                if (
                    final_semantic_source != semantic_source
                    or final_semantic != semantic
                ):
                    raise DispatcherError(
                        "integrity retirement semantic/current-control binding "
                        "changed before receipt publication"
                    )
                stable = {
                    "schema_version": INTEGRITY_RETIREMENT_SCHEMA_VERSION,
                    "kind": "schema5_dispatch_integrity_retirement",
                    "retirement_id": retirement_id,
                    "batch_id": batch_id,
                    "retirement_intent_path": str(
                        retirement_intent_path
                    ),
                    "retirement_intent_sha256": hashlib.sha256(
                        retirement_intent_raw
                    ).hexdigest(),
                    "scheduler_absence_path": str(scheduler_path),
                    "scheduler_absence_sha256": hashlib.sha256(
                        scheduler_raw
                    ).hexdigest(),
                    "semantic_evidence": semantic,
                    "acknowledgement": acknowledgement,
                    "identity_changes": identity_changes,
                    "archived_preimages": archived_records,
                    "completed_at": float(scheduler_absence["captured_at"]),
                    "completed_timestamp": float(
                        scheduler_absence["captured_at"]
                    ),
                }
                receipt = {
                    **stable,
                    "completion_id": hashlib.sha256(
                        json.dumps(
                            stable,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
                receipt_path = (
                    root / INTEGRITY_RETIREMENT_COMPLETE_FILENAME
                )
                _publish_readonly_json_once(receipt_path, receipt)
                _seal_integrity_retirement_tree(root)
                if crash_hook is not None:
                    crash_hook("after_receipt")
                receipt_raw = _stable_readonly_artifact(
                    receipt_path,
                    description=f"integrity retirement receipt {batch_id}",
                )[1]
                updated = copy.deepcopy(ledger)
                target = updated["intents"][batch_id]
                if target.get("fairness_committed") is not False:
                    raise DispatcherError(
                        "integrity retirement cannot preserve committed fairness"
                    )
                target.update(
                    {
                        "state": "integrity_retired",
                        "integrity_retired_at": timestamp,
                        "retirement_id": retirement_id,
                        "retirement_receipt_path": str(receipt_path),
                        "retirement_receipt_sha256": hashlib.sha256(
                            receipt_raw
                        ).hexdigest(),
                        "retirement_semantic_report_path": semantic["path"],
                        "retirement_semantic_report_sha256": semantic["sha256"],
                        "retirement_scheduler_absence_sha256": (
                            receipt["scheduler_absence_sha256"]
                        ),
                        "retirement_operator_note_sha256": hashlib.sha256(
                            note.encode("utf-8")
                        ).hexdigest(),
                        "retirement_identity_changes": identity_changes,
                    }
                )
                updated["updated_at"] = timestamp
                validate_production_ledger_structure(
                    updated, source=str(ledger_path)
                )
                _atomic_write_json(ledger_path, updated)
                persisted = load_production_ledger(ledger_path)
                persisted_intent = persisted["intents"][batch_id]
                verified = _verify_integrity_retirement_receipt(
                    batch_id, persisted_intent
                )
                return verified


def _recover_dispatcher_submission_integrity_hold(
    state_dir: Path,
    *,
    ledger: Mapping[str, Any],
    now: float,
) -> tuple[str, ...]:
    """Reconstruct the global hold from every durable blocked admission intent.

    The intent is written while the dispatcher owns the admission boundary, whereas
    alert persistence acquires that boundary itself.  A hard kill between those two
    transactions is therefore recoverable only if every later production poll checks
    the ledger *before* reading capacity or planning WDRR.  Malformed blocked records
    fail closed instead of being silently ignored.
    """

    blocked: list[str] = []
    for batch_id, intent in ledger.get("intents", {}).items():
        if not isinstance(intent, Mapping):
            continue
        if intent.get("state") == "integrity_retired":
            try:
                _verify_integrity_retirement_receipt(batch_id, intent)
            except DispatcherError as exc:
                _persist_dispatcher_submission_integrity_hold(
                    state_dir,
                    message=(
                        "dispatcher retired-integrity archive failed sealed "
                        f"verification for {batch_id}; admission remains fenced: {exc}"
                    ),
                    now=now,
                )
                raise SubmissionPreflightError(str(exc)) from exc
            continue
        if intent.get("state") != "integrity_blocked":
            continue
        if (
            not isinstance(batch_id, str)
            or not batch_id
            or intent.get("integrity_alert_key")
            != DISPATCHER_SUBMISSION_INTEGRITY_ALERT
            or not isinstance(intent.get("error"), str)
            or not intent["error"]
            or not isinstance(intent.get("tasks"), list)
            or not intent["tasks"]
        ):
            raise SubmissionPreflightError(
                "durable integrity-blocked dispatcher intent is malformed"
            )
        blocked.append(batch_id)
    if not blocked:
        return ()
    _persist_dispatcher_submission_integrity_hold(
        state_dir,
        message=(
            "dispatcher recovered durable immutable-admission provenance "
            "failures before admission; exact coordinates remain fenced for "
            "human review: "
            + ", ".join(sorted(blocked))
        ),
        now=now,
    )
    return tuple(sorted(blocked))


def _load_qualification_capacity_contract(
    values: Mapping[str, Any],
    authority: QualificationExecutionAuthority,
) -> protected_capacity.ProtectedCapacityContract:
    """Join the isolated qualification marker to its frozen execution source.

    Qualification runs intentionally have no production control directory, so their
    sealed execution authority is the sole equivalent of immutable control pins.  The
    marker CLI binding must agree exactly with that authority before the certificate
    may authenticate the annotated tag, full source tree, dispatcher, and
    qualification-runner bytes.
    """

    protected = authority.payload.get("protected_capacity")
    if not isinstance(protected, Mapping):
        raise DispatcherError(
            "qualification execution authority has no protected-capacity binding"
        )
    try:
        supplied_path = Path(str(values["path"])).expanduser().resolve()
        authorized_path = Path(str(protected["path"])).expanduser().resolve()
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DispatcherError(
            f"qualification protected-capacity path binding is invalid: {exc}"
        ) from exc
    if (
        supplied_path != authorized_path
        or str(values.get("sha256")) != str(protected.get("sha256"))
        or str(values.get("marker_id")) != str(protected.get("marker_id"))
        or str(values.get("release_git_commit"))
        != str(authority.payload.get("release_git_commit"))
    ):
        raise DispatcherError(
            "qualification protected-capacity CLI differs from the sealed "
            "execution authority"
        )
    return protected_capacity.load_contract(
        authorized_path,
        expected_release_git_commit=str(
            authority.payload["release_git_commit"]
        ),
        expected_release_tag_object=str(
            authority.payload["release_tag_object"]
        ),
        expected_marker_id=str(protected["marker_id"]),
        expected_sha256=str(protected["sha256"]),
        expected_source_tree_sha256=str(
            authority.payload["source_tree_sha256"]
        ),
        expected_dispatcher_source_sha256=str(
            authority.execution["dispatcher_script_sha256"]
        ),
        expected_qualification_runner_source_sha256=str(
            authority.execution["qualification_runner_script_sha256"]
        ),
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
        ledger = validate_production_ledger_structure(
            ledger,
            source=str(Path(control_state_dir) / "ledger.json"),
        )
    if control_state_dir is not None and not dry_run:
        # This recovery cut must precede even the first admission-contract read.
        # Thus a crash after the blocked intent fsync but before alert publication
        # cannot let a successor observe the old nonzero ceiling or admit other cells.
        _recover_dispatcher_submission_integrity_hold(
            Path(control_state_dir),
            ledger=ledger,
            now=now,
        )
    if control_state_dir is not None:
        from slurm.schema5_control import (
            admission_contract_from_state,
            effective_fleet_contract_binding,
            load_effective_protected_capacity_contract,
            load_control,
            production_environment_from_state,
        )

        production_contract = admission_contract_from_state(control_state_dir)
        control = load_control(control_state_dir, verify_files=True)
        immutable = control["immutable"]
        production_capacity_contract = (
            load_effective_protected_capacity_contract(
                control, verify_files=True
            )
        )
        protected_ref = production_contract["protected_capacity"]
        if {
            "path": str(production_capacity_contract.path),
            "sha256": production_capacity_contract.sha256,
            "marker_id": production_capacity_contract.marker_id,
        } != protected_ref:
            raise DispatcherError(
                "production admission and live protected-capacity authorities "
                "changed across the control read"
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
            qualification_execution_authority = (
                load_qualification_execution_authority(
                    str(qualification_execution_path)
                )
            )
            qualification_capacity_contract = (
                _load_qualification_capacity_contract(
                    qualification_values,
                    qualification_execution_authority,
                )
            )
            qualification_client_placement = protected_capacity.authorize_client(
                qualification_capacity_contract,
                partition=str(args.cell_partition),
                qos=str(qualification_qos),
                required_slots=384,
                required_reserve_jobs=64,
            )
            qualification_release_root = Path(
                qualification_execution_authority.execution[
                    "release_worktree"
                ]
            )
            qualification_model_contracts = load_model_contracts(
                qualification_release_root / "configs" / "model_contracts.v1.json",
                expected_sha256=(
                    qualification_execution_authority.runtime_environment[
                        "ASYS_MODEL_CONTRACT_SHA256"
                    ]
                ),
            )
            production_fleet = load_fleet_contract(
                qualification_execution_authority.runtime_environment[
                    "ASYS_FLEET_CONTRACT_PATH"
                ],
                model_contracts=qualification_model_contracts,
                expected_sha256=(
                    qualification_execution_authority.runtime_environment[
                        "ASYS_FLEET_CONTRACT_SHA256"
                    ]
                ),
                allow_capacity_layout=True,
            )
            for qualification_pool in {
                spec.server_pool_root.resolve() for spec in specs
            }:
                production_fleet.verify_pool_root(qualification_pool)
        except (
            DispatcherError,
            FleetContractError,
            ModelContractError,
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
    if protected_admission and not dry_run:
        # The shared protected-capacity reconciler accepts only the stable
        # descriptor preimage on disk.  Publish this poll's fully reconciled state
        # before asking it to discount any scientific allocation; an in-memory poll
        # number or timestamp is never capacity evidence.
        work["updated_at"] = time.time()
        _atomic_write_json(args.ledger_path, work)
    trusted_nonclient_bindings: dict[str, dict[str, Any]] = {}
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

        def trusted_scientific_provenance(
            scheduler_rows: Sequence[QueueRow],
            *,
            client_bindings: Mapping[str, Any],
            nonclient_bindings: Mapping[str, Any],
            scheduler_snapshot: Any,
            reconciled_at: float,
        ) -> protected_capacity.TrustedScientificJobProvenance:
            from slurm.schema5_control import (
                SchedulerAmbiguity,
                reconcile_trusted_scientific_job_provenance,
            )

            if production_contract is not None:
                assert control is not None
                fleet_generation = int(control["rollout_generation"])
            else:
                generations = {
                    dict(spec.runtime_environment).get(
                        "ASYS_ROLLOUT_GENERATION"
                    )
                    for spec in specs
                }
                if (
                    len(generations) != 1
                    or not str(next(iter(generations), "")).isdigit()
                ):
                    raise DispatcherError(
                        "qualification scientific provenance lacks one rollout "
                        "generation"
                    )
                fleet_generation = int(next(iter(generations)))
            try:
                snapshot_ids = {
                    str(job.job_id)
                    for job in scheduler_snapshot.jobs
                    if job.source == "squeue" and job.active
                }
                row_ids = {row.job_id for row in scheduler_rows}
                if row_ids != snapshot_ids:
                    raise DispatcherError(
                        "trusted scientific scheduler rows differ from the "
                        "stable scheduler snapshot"
                    )
                provenance = (
                    reconcile_trusted_scientific_job_provenance(
                        args.state_dir,
                        fleet_bindings=nonclient_bindings,
                        fleet_contract_sha256=expected_fleet_sha256,
                        fleet_generation=fleet_generation,
                        scheduler_snapshot=scheduler_snapshot,
                        now=reconciled_at,
                        allow_exact_cell_quiescence=False,
                    )
                )
                _require_exact_trusted_scientific_binding_sets(
                    provenance,
                    client_bindings=client_bindings,
                    nonclient_bindings=nonclient_bindings,
                )
                return provenance
            except (
                SchedulerAmbiguity,
                protected_capacity.ProtectedCapacityError,
            ) as exc:
                raise DispatcherError(
                    f"trusted scientific scheduler provenance failed: {exc}"
                ) from exc

        initial_boundary_now = time.time()
        initial_trusted_provenance = trusted_scientific_provenance(
            rows,
            client_bindings=trusted_cell_bindings,
            nonclient_bindings=trusted_nonclient_bindings,
            scheduler_snapshot=initial_occupancy.scheduler_snapshot,
            reconciled_at=initial_boundary_now,
        )
        initial_client_partition = (
            str(production_contract["client_capacity"]["partition"])
            if production_contract is not None
            else str(args.cell_partition)
        )
        initial_client_qos = (
            str(production_contract["client_capacity"]["qos"])
            if production_contract is not None
            else str(args.cell_qos)
        )
        _require_trusted_client_target_placement(
            initial_occupancy,
            trusted_client_bindings=trusted_cell_bindings,
            expected_partition=initial_client_partition,
            expected_qos=initial_client_qos,
        )
        try:
            if production_contract is not None:
                assert production_capacity_contract is not None
                client_contract = production_contract["client_capacity"]
                protected_capacity.verify_live_placements(
                    production_capacity_contract,
                    role="client",
                    placements=[
                        (
                            str(client_contract["partition"]),
                            str(client_contract["qos"]),
                        )
                    ],
                    required_time_limits_seconds={
                        str(client_contract["partition"]): (
                            scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                        )
                    },
                    trusted_scientific_job_provenance=(
                        initial_trusted_provenance
                    ),
                )
                production_client_capacity = (
                    protected_capacity.capture_live_client_capacity(
                        production_capacity_contract,
                        partition=str(client_contract["partition"]),
                        qos=str(client_contract["qos"]),
                        required_time_limit_seconds=(
                            scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                        ),
                        trusted_scientific_job_provenance=(
                            initial_trusted_provenance
                        ),
                        captured_timestamp=initial_boundary_now,
                    )
                )
            else:
                assert qualification_capacity_contract is not None
                protected_capacity.verify_live_placements(
                    qualification_capacity_contract,
                    role="client",
                    placements=[
                        (str(args.cell_partition), str(args.cell_qos))
                    ],
                    required_time_limits_seconds={
                        str(args.cell_partition): (
                            scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                        )
                    },
                    trusted_scientific_job_provenance=(
                        initial_trusted_provenance
                    ),
                )
                qualification_client_capacity = (
                    protected_capacity.capture_live_client_capacity(
                        qualification_capacity_contract,
                        partition=str(args.cell_partition),
                        qos=str(args.cell_qos),
                        required_time_limit_seconds=(
                            scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                        ),
                        trusted_scientific_job_provenance=(
                            initial_trusted_provenance
                        ),
                        captured_timestamp=initial_boundary_now,
                    )
                )
        except (
            scheduler_safety.SchedulerSafetyError,
            protected_capacity.ProtectedCapacityError,
        ) as exc:
            raise DispatcherError(
                f"schema-5 client QOS/TRES/MaxJobs authority failed closed: {exc}"
            ) from exc

        def capture_protected_boundary(
            *,
            partition: str,
            qos: str,
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
            dict[str, dict[str, Any]],
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
            _require_trusted_client_target_placement(
                stable,
                trusted_client_bindings=fresh_client_bindings,
                expected_partition=partition,
                expected_qos=qos,
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
                trusted_client_jobs=_scheduler_headroom_binding_projection(
                    fresh_client_bindings
                ),
                trusted_nonclient_jobs=_scheduler_headroom_binding_projection(
                    fresh_nonclient_bindings
                ),
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
            trusted_client_jobs=_scheduler_headroom_binding_projection(
                trusted_cell_bindings
            ),
            trusted_nonclient_jobs=_scheduler_headroom_binding_projection(
                trusted_nonclient_bindings
            ),
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
            trusted_client_jobs=_scheduler_headroom_binding_projection(
                trusted_cell_bindings
            ),
            trusted_nonclient_jobs=_scheduler_headroom_binding_projection(
                trusted_nonclient_bindings
            ),
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
    unresolved_admission_intents = sorted(
        intent_id
        for intent_id, intent in work.get("intents", {}).items()
        if isinstance(intent, Mapping)
        and intent.get("state") in {"prepared", "submitting"}
    )
    if unresolved_admission_intents:
        # Fairness is committed only after scheduler acceptance.  A second intent
        # planned from the old fairness snapshot could otherwise commit first and
        # then be overwritten when this older intent is adopted.  Serialize the
        # unresolved external boundary instead of trying to compose stale snapshots.
        slots = 0
        join_warnings.append(
            "global admission fenced until unresolved intent(s) reconcile: "
            + ", ".join(unresolved_admission_intents[:8])
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
    submission_integrity_error: str | None = None
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
                    (
                        fresh_occupancy,
                        fresh_resource_slots,
                        fresh_active_cell_jobs,
                        fresh_client_bindings,
                        fresh_nonclient_bindings,
                    ) = capture_protected_boundary(
                        partition=str(client_contract["partition"]),
                        qos=str(client_contract["qos"]),
                        cpu_limit=int(client_contract["cpu_limit"]),
                        memory_limit_mib=int(
                            client_contract["memory_limit_mib"]
                        ),
                        max_submit_jobs=int(
                            client_contract["max_submit_jobs"]
                        ),
                        reserve_jobs=int(client_contract["reserve_jobs"]),
                    )
                    fresh_client_capacity = (
                        protected_capacity.capture_live_client_capacity(
                            production_capacity_contract,
                            partition=str(client_contract["partition"]),
                            qos=str(client_contract["qos"]),
                            required_time_limit_seconds=(
                                scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                            ),
                            trusted_scientific_job_provenance=(
                                trusted_scientific_provenance(
                                    fresh_occupancy.rows,
                                    client_bindings=fresh_client_bindings,
                                    nonclient_bindings=(
                                        fresh_nonclient_bindings
                                    ),
                                    scheduler_snapshot=(
                                        fresh_occupancy.scheduler_snapshot
                                    ),
                                    reconciled_at=time.time(),
                                )
                            ),
                        )
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
                    (
                        fresh_qualification_occupancy,
                        fresh_resource_slots,
                        _fresh_active_cell_jobs,
                        fresh_client_bindings,
                        fresh_nonclient_bindings,
                    ) = capture_protected_boundary(
                        partition=str(args.cell_partition),
                        qos=str(args.cell_qos),
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
                    fresh_qualification_capacity = (
                        protected_capacity.capture_live_client_capacity(
                            qualification_capacity_contract,
                            partition=str(args.cell_partition),
                            qos=str(args.cell_qos),
                            required_time_limit_seconds=(
                                scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                            ),
                            trusted_scientific_job_provenance=(
                                trusted_scientific_provenance(
                                    fresh_qualification_occupancy.rows,
                                    client_bindings=fresh_client_bindings,
                                    nonclient_bindings=(
                                        fresh_nonclient_bindings
                                    ),
                                    scheduler_snapshot=(
                                        fresh_qualification_occupancy.scheduler_snapshot
                                    ),
                                    reconciled_at=time.time(),
                                )
                            ),
                        )
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
                "submission_transport": STDIN_EXACT_SUBMISSION_TRANSPORT,
                "submission_argv_sha256": _stdin_submission_argv_sha256(
                    batch_id
                ),
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
            work["updated_at"] = time.time()
            _atomic_write_json(args.ledger_path, work)
            # Fence the ambiguous external-call boundary before the final stable
            # scheduler/resource cut.  A crash during that cut may conservatively
            # wait out visibility grace, but can never redraw tasks accepted by an
            # unobserved sbatch.  The shared provenance reconciler below therefore
            # hashes this exact last durable ``submitting`` preimage.
            work["intents"][batch_id]["state"] = "submitting"
            work["intents"][batch_id]["submit_started_at"] = time.time()
            work["updated_at"] = time.time()
            _atomic_write_json(args.ledger_path, work)
            if production_contract is not None:
                try:
                    protected_ref = production_contract["protected_capacity"]
                    live_contract = (
                        load_effective_protected_capacity_contract(
                            control, verify_files=True
                        )
                    )
                    if {
                        "path": str(live_contract.path),
                        "sha256": live_contract.sha256,
                        "marker_id": live_contract.marker_id,
                    } != protected_ref:
                        raise DispatcherError(
                            "protected-capacity authority changed immediately "
                            "before sbatch"
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
                    (
                        fresh_production_occupancy,
                        fresh_headroom,
                        fresh_active_cell_jobs,
                        fresh_client_bindings,
                        fresh_nonclient_bindings,
                    ) = capture_protected_boundary(
                        partition=str(client_contract["partition"]),
                        qos=str(client_contract["qos"]),
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
                    fresh_production_capacity = (
                        protected_capacity.capture_live_client_capacity(
                            live_contract,
                            partition=str(client_contract["partition"]),
                            qos=str(client_contract["qos"]),
                            required_time_limit_seconds=(
                                scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                            ),
                            trusted_scientific_job_provenance=(
                                trusted_scientific_provenance(
                                    fresh_production_occupancy.rows,
                                    client_bindings=fresh_client_bindings,
                                    nonclient_bindings=(
                                        fresh_nonclient_bindings
                                    ),
                                    scheduler_snapshot=(
                                        fresh_production_occupancy.scheduler_snapshot
                                    ),
                                    reconciled_at=time.time(),
                                )
                            ),
                        )
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
                    prepared_intent = work["intents"][batch_id]
                    prepared_intent.pop("submit_started_at", None)
                    prepared_intent.pop("last_submit_error_at", None)
                    prepared_intent.update(
                        {
                            "state": "prepared",
                            "error": (
                                "protected client placement drifted before sbatch: "
                                + str(exc)
                            ),
                        }
                    )
                    work["updated_at"] = time.time()
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
                        _load_qualification_capacity_contract(
                            qualification_values,
                            fresh_execution_authority,
                        )
                    )
                    protected_capacity.authorize_client(
                        qualification_capacity_contract,
                        partition=str(args.cell_partition),
                        qos=str(args.cell_qos),
                        required_slots=384,
                        required_reserve_jobs=64,
                    )
                    (
                        fresh_qualification_occupancy,
                        fresh_headroom,
                        _fresh_active_cell_jobs,
                        fresh_client_bindings,
                        fresh_nonclient_bindings,
                    ) = capture_protected_boundary(
                        partition=str(args.cell_partition),
                        qos=str(args.cell_qos),
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
                    fresh_qualification_capacity = (
                        protected_capacity.capture_live_client_capacity(
                            qualification_capacity_contract,
                            partition=str(args.cell_partition),
                            qos=str(args.cell_qos),
                            required_time_limit_seconds=(
                                scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                            ),
                            trusted_scientific_job_provenance=(
                                trusted_scientific_provenance(
                                    fresh_qualification_occupancy.rows,
                                    client_bindings=fresh_client_bindings,
                                    nonclient_bindings=(
                                        fresh_nonclient_bindings
                                    ),
                                    scheduler_snapshot=(
                                        fresh_qualification_occupancy.scheduler_snapshot
                                    ),
                                    reconciled_at=time.time(),
                                )
                            ),
                        )
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
                    prepared_intent = work["intents"][batch_id]
                    prepared_intent.pop("submit_started_at", None)
                    prepared_intent.pop("last_submit_error_at", None)
                    prepared_intent.update(
                        {
                            "state": "prepared",
                            "error": (
                                "protected qualification placement drifted "
                                "before sbatch: "
                                + str(exc)
                            ),
                        }
                    )
                    work["updated_at"] = time.time()
                    _atomic_write_json(args.ledger_path, work)
                    raise DispatcherError(
                        "protected qualification partition/QOS drifted "
                        f"immediately before sbatch: {exc}"
                    ) from exc
            try:
                job_id = _submit_sbatch(
                    sbatch_path,
                    expected_sbatch_sha256=str(
                        work["intents"][batch_id]["sbatch_sha256"]
                    ),
                    batch_manifest_path=manifest_path,
                    expected_batch_manifest_sha256=str(
                        work["intents"][batch_id][
                            "batch_manifest_sha256"
                        ]
                    ),
                )
            except SubmissionPreflightError as exc:
                # The external boundary was never crossed.  Preserve the exact
                # coordinates indefinitely and latch a scientific-integrity hold
                # after releasing this already-owned admission lock.
                submission_integrity_error = str(exc)
                blocked_intent = work["intents"][batch_id]
                for field in (
                    "job_id",
                    "submit_started_at",
                    "submitted_at",
                    "reconciled_at",
                    "spooled_sbatch_sha256",
                    "spooled_receipt_path",
                    "spooled_receipt_sha256",
                    "last_submit_error_at",
                ):
                    blocked_intent.pop(field, None)
                for job_id, record in list(work["jobs"].items()):
                    if (
                        isinstance(record, Mapping)
                        and record.get("batch_id") == batch_id
                    ):
                        del work["jobs"][job_id]
                blocked_intent.update(
                    {
                        "state": "integrity_blocked",
                        "fairness_committed": False,
                        "error": submission_integrity_error,
                        "integrity_blocked_at": time.time(),
                        "integrity_alert_key": (
                            DISPATCHER_SUBMISSION_INTEGRITY_ALERT
                        ),
                    }
                )
                work["updated_at"] = time.time()
                _atomic_write_json(args.ledger_path, work)
            except SubmissionRejectedError as exc:
                # Slurm explicitly rejected this request.  No allocation can exist,
                # so this is retryable without the ambiguous-visibility fence.
                submission_error = str(exc)
                work["intents"][batch_id].update(
                    {
                        "state": "submission_rejected",
                        "error": submission_error,
                        "last_submit_error_at": time.time(),
                    }
                )
                work["updated_at"] = time.time()
                _atomic_write_json(args.ledger_path, work)
            except (
                SubmissionAmbiguousError,
                DispatcherError,
                OSError,
                subprocess.TimeoutExpired,
            ) as exc:
                # An error after invoking sbatch is an ambiguous external boundary:
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
                work["updated_at"] = time.time()
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
                work["updated_at"] = accepted_at
                _atomic_write_json(args.ledger_path, work)

    if submission_integrity_error is not None:
        if control_state_dir is not None:
            _persist_dispatcher_submission_integrity_hold(
                control_state_dir,
                message=(
                    "dispatcher refused to invoke sbatch because immutable "
                    "admission provenance drifted; the exact coordinates remain "
                    f"fenced: {submission_integrity_error}"
                ),
                now=time.time(),
            )
        raise SubmissionPreflightError(submission_integrity_error)

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
    work["updated_at"] = max(
        float(work.get("updated_at", now)),
        now,
    )
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
                        successor = _queue_successor(
                            Path(args.successor_sbatch).resolve(),
                            state_dir=args.state_dir,
                        )
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
    if ledger.get("schema_version") == LEDGER_SCHEMA_VERSION:
        ledger = validate_production_ledger_structure(
            ledger, source=str(state_dir / "ledger.json")
        )
    jobs = ledger.get("jobs", {})
    runs = ledger.get("runs", {})
    cells = ledger.get("cells", {})
    intents = ledger.get("intents", {})
    by_state: dict[str, int] = defaultdict(int)
    for record in cells.values():
        by_state[str(record.get("completion_state", "unknown"))] += 1
    blocked = sorted(
        batch_id
        for batch_id, record in intents.items()
        if record.get("state") == "integrity_blocked"
    )
    retired: list[str] = []
    invalid_retired: dict[str, str] = {}
    for batch_id, record in sorted(intents.items()):
        if record.get("state") != "integrity_retired":
            continue
        try:
            _verify_integrity_retirement_receipt(batch_id, record)
        except DispatcherError as exc:
            invalid_retired[batch_id] = str(exc)
        else:
            retired.append(batch_id)
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
        "integrity_intents": {
            "blocked": blocked,
            "blocked_count": len(blocked),
            "retired": retired,
            "retired_count": len(retired),
            "invalid_retired": invalid_retired,
        },
        "cells_by_state": dict(sorted(by_state.items())),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _run_retire_integrity(args: argparse.Namespace) -> int:
    receipt = retire_integrity_blocked_intent(
        Path(args.state_dir),
        control_state_dir=Path(args.control_state_dir),
        batch_id=str(args.batch_id),
        semantic_evidence_path=Path(args.semantic_evidence),
        semantic_evidence_sha256=str(args.semantic_evidence_sha256),
        operator_note=str(args.note),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
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
        "release_tag_object",
        "source_tree_sha256",
        "qualification_runner_source_sha256",
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
        "qualification_runner_script",
        "qualification_runner_script_sha256",
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
        "release_tag_object": str(
            authority.payload["release_tag_object"]
        ),
        "source_tree_sha256": str(
            authority.payload["source_tree_sha256"]
        ),
        "qualification_runner_source_sha256": str(
            authority.payload["qualification_runner_source_sha256"]
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
        or value.get("schema_version")
        != QUALIFICATION_EXECUTION_AUTHORITY_SCHEMA_VERSION
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
        or re.fullmatch(
            r"[0-9a-f]{40}", str(value.get("release_tag_object", ""))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("source_tree_sha256", ""))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(value.get("qualification_runner_source_sha256", "")),
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
        "qualification_runner_script",
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
        "qualification_runner_script": (
            release_root
            / "scripts"
            / "run_schema5_throughput_qualification.py"
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
    if (
        execution["qualification_runner_script_sha256"]
        != value["qualification_runner_source_sha256"]
    ):
        raise DispatcherError(
            "qualification runner differs from its release source authority"
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

    retire_integrity = sub.add_parser(
        "retire-integrity",
        help=(
            "archive and retire one pre-sbatch integrity-blocked intent after "
            "clean semantic evidence, human acknowledgement, identity change, "
            "and complete scheduler absence"
        ),
    )
    retire_integrity.add_argument("--state-dir", required=True)
    retire_integrity.add_argument("--control-state-dir", required=True)
    retire_integrity.add_argument("--batch-id", required=True)
    retire_integrity.add_argument(
        "--semantic-evidence", required=True, type=Path
    )
    retire_integrity.add_argument(
        "--semantic-evidence-sha256", required=True
    )
    retire_integrity.add_argument("--note", required=True)

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
    if args.command == "retire-integrity":
        return _run_retire_integrity(args)
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
