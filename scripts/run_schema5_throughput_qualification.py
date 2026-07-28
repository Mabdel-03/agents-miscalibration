#!/usr/bin/env python3
"""Run, seal, or verify the estimand-excluded schema-5 throughput qualification.

The qualification is deliberately outside the three production run namespaces.  Its
scientific reference is one immutable 768-cell manifest (20 QIDs per cell).  Sustained
load is measured separately with cycle-scoped, estimand-excluded replays of that exact
design.  Every replay has an immutable intent, run root, dispatcher ledger, and frozen
dispatcher authority.  The production control is read only: it must remain paused and
its admission/ramp projection must not change.

``dry-run`` is entirely non-mutating.  ``execute`` is the sole mode that may initialize
the qualification run or submit qualification arrays.  It advances isolated active-cell
limits through 24, 96, 192, and 384, recording immutable scheduler and semantic
observations at every poll.  A completion marker is published last only after:

* the 768-cell, 15,360-QID semantic reference cycle is schema-5 complete;
* every nonempty model x reasoning x topology x agent-count stratum progressed;
* scheduler and semantic evidence contain no integrity or transport-censor incident;
* one immutable, non-resettable load-window intent begins at the first clean
  exact-384 observation;
* the resulting contiguous window lasts at least 7,200 seconds, has observation gaps
  no larger than 660 seconds, holds exactly 384 active assignments with at least 384
  unfinished assignments, and makes trusted progress in every stratum; and
* trusted schema-5 execution-event throughput is at least 201,994 events/day.

The rate numerator is explicitly an execution-event count.  It may include the same
scientific coordinate in different replay cycles, but never twice within one cycle.
Unique scientific coverage remains exactly 15,360 QIDs and is never inflated by load
replays.  After qualification, replay cycles are gracefully drained; partial cycles
are sealed as ``load_window_drained`` and remain ineligible for analysis.

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
import shutil
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
from agents_scaling.serving.fleet_contract import (  # noqa: E402
    EXPECTED_COUNTS,
    FleetContractError,
    FrozenFleetContract,
    load_fleet_contract,
)
from agents_scaling.serving.model_contracts import (  # noqa: E402
    ModelContractError,
    load_model_contracts,
)
from agents_scaling.serving.profiles import (  # noqa: E402
    SERVING_PROFILES as SERVING_PROFILE_REGISTRY,
    serving_profile_for_cell,
)
from slurm import dispatch_sweeps  # noqa: E402
from slurm import schema5_control as control  # noqa: E402
from scripts import render_schema5_recovery_chain_v12 as renderer  # noqa: E402


# Qualification artifacts are created by the active r10 chain.  Their schema version
# remains v1 where appropriate, but their protocol identity must never claim r2.
SCHEMA_VERSION = 1
RECOVERY_CHAIN_PROTOCOL = "schema5-v1.2-r10-recovery-chain"
LOAD_ACCOUNTING_SCHEMA_VERSION = 3
QUALIFICATION_RUN_ID = "schema5_throughput_qualification_v1"
QUALIFICATION_ROOT_NAME = QUALIFICATION_RUN_ID
MARKER_NAME = "THROUGHPUT_QUALIFICATION_COMPLETE.json"
PROTOCOL = "schema5-v1.2-r10-throughput-qualification-v3"
PLAN_PROTOCOL = "schema5-v1.2-r10-throughput-qualification-load-plan-v3"
INTENT_PROTOCOL = "schema5-v1.2-r10-throughput-qualification-intent-v3"
SCHEDULER_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-scheduler-evidence-v3"
)
SEMANTIC_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-semantic-evidence-v3"
)
OBSERVATION_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-observation-v3"
)
OBSERVATION_TRANSACTION_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-observation-transaction-v3"
)
EVIDENCE_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-evidence-v3"
)
LINEAGE_PROTOCOL = "schema5-v1.2-r10-throughput-qualification-lineage-v3"
EXECUTION_AUTHORITY_SCHEMA_VERSION = 2
EXECUTION_AUTHORITY_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-execution-authority-v2"
)
ATTEMPT_POINTER_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-attempt-pointer-v1"
)
CURRENT_ATTEMPT_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-current-attempt-v1"
)

CELL_COUNT = 768
QIDS_PER_CELL = 20
TOTAL_QIDS = CELL_COUNT * QIDS_PER_CELL
CEILINGS = (24, 96, 192, 384)
MAX_BATCH = 24
QOS_LIMIT = 448
QOS_RESERVE = 64
HEALTH_SOAK_384_SECONDS = 7_200
MIN_LOADED_384_OBSERVATIONS = 2
MAX_OBSERVATION_GAP_SECONDS = 660
MIN_QIDS_PER_DAY = 201_994
MIN_LOAD_WINDOW_EXECUTION_EVENTS = math.ceil(
    MIN_QIDS_PER_DAY * HEALTH_SOAK_384_SECONDS / 86_400
)
DEFAULT_POLL_SECONDS = 300.0
DEFAULT_TIMEOUT_SECONDS = 36_000.0

PLAN_NAME = "LOAD_PLAN.json"
INTENT_NAME = "QUALIFICATION_INTENT.json"
EVIDENCE_NAME = "QUALIFICATION_EVIDENCE.json"
LINEAGE_NAME = "qualification_lineage.schema5-v1.json"
INITIALIZED_NAME = "SCHEMA5_THROUGHPUT_QUALIFICATION_INITIALIZED.json"
EXECUTION_AUTHORITY_NAME = "QUALIFICATION_EXECUTION_AUTHORITY.json"
FAILURE_NAME = "QUALIFICATION_FAILURE.json"
FAILURE_DRAIN_INTENT_NAME = "QUALIFICATION_FAILURE_DRAIN_INTENT.json"
FAILURE_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-failure-v4"
)
FAILURE_DRAIN_INTENT_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-failure-drain-intent-v3"
)
CAPACITY_TRANSITION_DIRECTORY = "capacity-transitions"
CURRENT_CAPACITY_TRANSITION_NAME = "CURRENT_CAPACITY_TRANSITION.json"
CAPACITY_TRANSITION_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-capacity-transition-v2"
)
CURRENT_CAPACITY_TRANSITION_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-current-capacity-transition-v1"
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
LOAD_CYCLE_DIRECTORY = "load-cycles"
LOAD_REPLAY_RUN_DIRECTORY = "load-replay-runs"
CYCLE_INTENT_NAME = "CYCLE_INTENT.json"
CYCLE_INITIALIZED_NAME = "CYCLE_INITIALIZED.json"
CYCLE_EVENTS_DIRECTORY = "trusted-events"
CYCLE_DRAIN_NAME = "CYCLE_DRAIN_COMPLETE.json"
LOAD_WINDOW_INTENT_NAME = "LOAD_WINDOW_INTENT.json"
LOAD_WINDOW_END_INTENT_NAME = "LOAD_WINDOW_END_INTENT.json"
LOAD_WINDOW_DRAIN_NAME = "LOAD_WINDOW_DRAIN_COMPLETE.json"
REFILL_RECONCILIATION_DIRECTORY = "refill-reconciliations"
PREFLIGHT_CAPACITY_CERTIFICATE_NAME = (
    "PREFLIGHT_CAPACITY_CERTIFICATE.json"
)
PREFLIGHT_CAPACITY_SHORTFALL_NAME = "PREFLIGHT_CAPACITY_SHORTFALL.json"
PREFLIGHT_CAPACITY_PROTOCOL = (
    "schema5-v1.2-r10-throughput-preflight-capacity-certificate-v1"
)
PREFLIGHT_CAPACITY_SHORTFALL_PROTOCOL = (
    "schema5-v1.2-r10-throughput-preflight-capacity-shortfall-v1"
)
PREFLIGHT_CAPACITY_ALGORITHM = (
    "dispatch_sweeps.plan_admission-sequential-wdrr-v1"
)
FANOUT_SLOTS_PER_REPLICA = 24
MIN_RETAINED_WARM_TURNOVER_GPUS = 4
CYCLE_INTENT_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-cycle-intent-v3"
)
CYCLE_INITIALIZED_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-cycle-initialized-v3"
)
LOAD_EVENT_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-trusted-event-v3"
)
LOAD_WINDOW_INTENT_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-load-window-intent-v3"
)
LOAD_WINDOW_END_INTENT_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-load-window-end-intent-v3"
)
CYCLE_DRAIN_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-cycle-drain-v3"
)
LOAD_WINDOW_DRAIN_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-load-window-drain-v3"
)
REFILL_RECONCILIATION_PROTOCOL = (
    "schema5-v1.2-r10-throughput-qualification-refill-reconciliation-v3"
)

_FAILURE_FIELDS = {
    "schema_version",
    "protocol",
    "passed",
    "intent_id",
    "attempt",
    "readiness_generation",
    "reason",
    "admission_capacity_certificate",
    "additive_scaling_requirement",
    "scheduler_capacity_mutated",
    "rerun_requirement",
    "failure_drain_intent",
    "cycle_run_roots",
    "refill_reconciliations",
    "failure_id",
}
_ADMISSION_CAPACITY_CERTIFICATE_FIELDS = {
    "path",
    "sha256",
    "certificate_id",
    "capacity_generation",
    "effective_fleet_contract_sha256",
    "effective_logical_replicas",
    "effective_active_gpus",
    "wave_passed",
    "selected_cell_count",
    "target_cell_count",
    "shortfall_cells",
    "theoretical_packing_upper_bound",
}
_CURRENT_CAPACITY_TRANSITION_FIELDS = {
    "schema_version",
    "protocol",
    "path",
    "sha256",
    "transition_id",
    "from_capacity_generation",
    "to_capacity_generation",
    "failed_attempt_id",
    "pointer_id",
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
_CAPACITY_SHORTFALL_REASON_RE = re.compile(
    r"qualification throughput from "
    r"(?P<events>[0-9][0-9,]*) trusted execution events in "
    r"(?P<duration>[0-9]+(?:\.[0-9]+)?) seconds "
    r"\((?P<rate>[0-9][0-9,]*)/day\) is below "
    r"(?P<threshold>[0-9][0-9,]*)\Z"
)
_STATIC_ADMISSION_CAPACITY_SHORTFALL_REASON_RE = re.compile(
    r"static admission capacity shortfall in generation "
    r"(?P<generation>[1-9][0-9]*): signed certificate selected "
    r"(?P<selected>[0-9][0-9,]*)/(?P<target>[0-9][0-9,]*) cells "
    r"with a (?P<shortfall>[0-9][0-9,]*)-cell shortfall; "
    r"theoretical packing upper bound "
    r"(?P<upper_bound>[0-9][0-9,]*)\Z"
)


class ThroughputQualificationError(RuntimeError):
    """The qualification cannot proceed without weakening its evidence contract."""


class QualificationCapacityTransitionRequired(ThroughputQualificationError):
    """A sealed throughput-only shortfall requires one exact additive transition."""


class QualificationWindowContinuityError(ThroughputQualificationError):
    """The immutable full-load window could not be truthfully maintained."""


def _capacity_shortfall_reason(
    execution_events: int,
    duration_seconds: float,
) -> str:
    """Return the one producer-authenticated throughput-capacity reason."""

    if (
        not isinstance(execution_events, int)
        or isinstance(execution_events, bool)
        or execution_events < 0
        or not isinstance(duration_seconds, (int, float))
        or isinstance(duration_seconds, bool)
        or not math.isfinite(float(duration_seconds))
        or float(duration_seconds) < HEALTH_SOAK_384_SECONDS
    ):
        raise ThroughputQualificationError(
            "throughput-capacity shortfall evidence is invalid"
        )
    throughput = math.floor(
        execution_events * 86_400.0 / float(duration_seconds)
    )
    if throughput >= MIN_QIDS_PER_DAY:
        raise ThroughputQualificationError(
            "throughput-capacity transition requires a measured rate below "
            f"{MIN_QIDS_PER_DAY:,}"
        )
    return (
        f"qualification throughput from {execution_events:,} trusted "
        f"execution events in {float(duration_seconds):g} seconds "
        f"({throughput:,}/day) is below {MIN_QIDS_PER_DAY:,}"
    )


def _is_capacity_shortfall_reason(reason: str) -> bool:
    """Recognize only the exact canonical reason emitted above."""

    match = _CAPACITY_SHORTFALL_REASON_RE.fullmatch(reason)
    if match is None:
        return False
    try:
        execution_events = int(match.group("events").replace(",", ""))
        duration_seconds = float(match.group("duration"))
        throughput = int(match.group("rate").replace(",", ""))
        threshold = int(match.group("threshold").replace(",", ""))
        expected = _capacity_shortfall_reason(
            execution_events,
            duration_seconds,
        )
    except (ValueError, ThroughputQualificationError):
        return False
    return (
        threshold == MIN_QIDS_PER_DAY
        and throughput
        == math.floor(execution_events * 86_400.0 / duration_seconds)
        and reason == expected
    )


def _static_admission_capacity_shortfall_reason(
    binding: Mapping[str, Any],
) -> str:
    """Return the canonical signed-certificate admission shortfall reason."""

    if set(binding) != _ADMISSION_CAPACITY_CERTIFICATE_FIELDS:
        raise ThroughputQualificationError(
            "static admission shortfall lacks its exact certificate binding"
        )
    generation = binding.get("capacity_generation")
    selected = binding.get("selected_cell_count")
    target = binding.get("target_cell_count")
    shortfall = binding.get("shortfall_cells")
    upper_bound = binding.get("theoretical_packing_upper_bound")
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
        or binding.get("wave_passed") is not False
        or not isinstance(selected, int)
        or isinstance(selected, bool)
        or selected < 0
        or not isinstance(target, int)
        or isinstance(target, bool)
        or target != CEILINGS[-1]
        or not isinstance(shortfall, int)
        or isinstance(shortfall, bool)
        or shortfall != target - selected
        or shortfall <= 0
        or not isinstance(upper_bound, int)
        or isinstance(upper_bound, bool)
        or not selected <= upper_bound < target
    ):
        raise ThroughputQualificationError(
            "static admission shortfall certificate is not a genuine "
            "sub-384 capacity proof"
        )
    return (
        f"static admission capacity shortfall in generation {generation}: "
        f"signed certificate selected {selected:,}/{target:,} cells with a "
        f"{shortfall:,}-cell shortfall; theoretical packing upper bound "
        f"{upper_bound:,}"
    )


def _is_static_admission_capacity_shortfall_reason(
    reason: str,
    binding: Mapping[str, Any],
) -> bool:
    """Recognize only the canonical reason derived from one sealed certificate."""

    if _STATIC_ADMISSION_CAPACITY_SHORTFALL_REASON_RE.fullmatch(reason) is None:
        return False
    try:
        expected = _static_admission_capacity_shortfall_reason(binding)
    except ThroughputQualificationError:
        return False
    return reason == expected


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
    source_tree_sha256: str
    dispatcher_source_sha256: str
    qualification_runner_source_sha256: str
    protected_capacity: Mapping[str, str]
    protected_capacity_contract: protected_capacity.ProtectedCapacityContract
    attempt_pointer_path: Path | None = None
    attempt_pointer: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LoadCycleContext:
    """One independently authorized, analysis-excluded load execution cycle."""

    cycle_index: int
    cycle_id: str
    run_id: str
    evidence_root: Path
    run_root: Path
    dispatcher_state: Path
    execution_authority_path: Path
    intent_path: Path
    intent: Mapping[str, Any]
    semantic_reference: bool


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


def _tree_content_inventory(
    root: Path,
    *,
    description: str,
) -> dict[str, Any]:
    """Hash every regular file by relative path and exact bytes."""

    if root.is_symlink() or not root.is_dir():
        raise ThroughputQualificationError(
            f"{description} is missing or unsafe: {root}"
        )
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ThroughputQualificationError(
                f"{description} contains symlink: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ThroughputQualificationError(
                f"{description} contains non-regular member: {path}"
            )
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": relative,
                "size": size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "root": str(root.resolve()),
        "files": len(files),
        "bytes": total_bytes,
        "inventory_sha256": _sha256_bytes(_canonical_bytes(files)),
    }


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
        "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
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
        "health_soak_384_seconds": HEALTH_SOAK_384_SECONDS,
        "minimum_loaded_384_observations": MIN_LOADED_384_OBSERVATIONS,
        "maximum_observation_gap_seconds": MAX_OBSERVATION_GAP_SECONDS,
        "minimum_qids_per_day": MIN_QIDS_PER_DAY,
        "minimum_load_window_execution_events": (
            MIN_LOAD_WINDOW_EXECUTION_EVENTS
        ),
        "unique_design": {
            "cells": CELL_COUNT,
            "qids": TOTAL_QIDS,
            "semantic_reference_cycle": 0,
        },
        "load_execution": {
            "unit": "trusted_qid_execution_events",
            "repeated_coordinates": True,
            "uniqueness_key": ["attempt", "cycle", "cell", "qid"],
            "configured_client_ceiling": CEILINGS[-1],
            "certified_saturation_target_source": (
                "protected_capacity.static_feasibility_certificate."
                "selected_cell_count"
            ),
            "measurement_contract": (
                "signed_fleet_specific_saturation_cuts_with_sealed_"
                "work_conserving_refill"
            ),
            "minimum_unfinished_assignments_source": (
                "certified_saturation_target"
            ),
            "window_intent_policy": (
                "first_clean_certified_saturation_cut_nonresettable"
            ),
            "rate_denominator_includes_refill_wall_time": True,
            "graceful_drain_required": True,
        },
        "preflight_capacity_contract": {
            "algorithm": PREFLIGHT_CAPACITY_ALGORITHM,
            "configured_client_ceiling": CEILINGS[-1],
            "certificate_selected_count_semantics": (
                "fleet_specific_work_conserving_saturation_target"
            ),
            "maximum_microbatch": MAX_BATCH,
            "fanout_slots_per_replica": FANOUT_SLOTS_PER_REPLICA,
            "completion_assumption": "no_completions_during_certified_wave",
            "fairness_state": "persist_deficits_and_cursor_between_batches",
            "required_batches": math.ceil(
                CEILINGS[-1] / MAX_BATCH
            ),
            "requires_exact_selected_cell_ids": True,
            "requires_effective_fleet_binding": True,
            "requires_disjoint_retained_warm_turnover": True,
            "minimum_retained_warm_turnover_gpus": (
                MIN_RETAINED_WARM_TURNOVER_GPUS
            ),
        },
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
    "health_soak_384_seconds",
    "minimum_loaded_384_observations",
    "maximum_observation_gap_seconds",
    "minimum_qids_per_day",
    "minimum_load_window_execution_events",
    "unique_design",
    "load_execution",
    "preflight_capacity_contract",
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


def simulate_preflight_capacity_wave(
    profile_replicas: Mapping[str, int],
    *,
    server_pool_root: str = "/schema5-capacity-certificate",
) -> dict[str, Any]:
    """Run the exact sequential dispatcher policy without assuming completions."""

    if (
        set(profile_replicas) != set(SERVING_PROFILES)
        or not Path(server_pool_root).is_absolute()
        or any(
            not isinstance(replicas, int)
            or isinstance(replicas, bool)
            or replicas < 1
            for replicas in profile_replicas.values()
        )
    ):
        raise ThroughputQualificationError(
            "capacity certificate requires one positive replica count for "
            "every serving profile and an absolute pool root"
        )
    cells = generate_qualification_cells()
    candidates = [
        dispatch_sweeps.Candidate(
            run_id=QUALIFICATION_RUN_ID,
            run_root="/schema5-capacity-certificate/run",
            source_index=index,
            cell=cell,
            manifest_sha256="0" * 64,
            server_pool_arg=server_pool_root,
            server_pool_root=server_pool_root,
            serving_profile=serving_profile_for_cell(cell).name,
            fanout_cost=dispatch_sweeps.fanout_cost(cell),
            benchmark_contracts_sha256="1" * 64,
        )
        for index, cell in enumerate(cells)
    ]
    plan_profile_cells = Counter(
        candidate.serving_profile for candidate in candidates
    )
    plan_profile_fanout = Counter()
    for candidate in candidates:
        plan_profile_fanout[candidate.serving_profile] += (
            candidate.fanout_cost
        )
    live_servers = {
        (server_pool_root, profile): int(replicas)
        for profile, replicas in profile_replicas.items()
    }
    remaining = list(candidates)
    deficits: dict[str, float] = {}
    cursor = 0
    cumulative_cost: Counter[str] = Counter()
    selected_ids: set[str] = set()
    selected_wave: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    target = CEILINGS[-1]
    while len(selected_wave) < target:
        headroom = {
            (server_pool_root, profile): (
                int(replicas) * FANOUT_SLOTS_PER_REPLICA
                - int(cumulative_cost[profile])
            )
            for profile, replicas in profile_replicas.items()
        }
        deficits_before = dict(sorted(deficits.items()))
        cursor_before = cursor
        admission = dispatch_sweeps.plan_admission(
            remaining,
            deficits=deficits,
            cursor=cursor,
            max_tasks=min(MAX_BATCH, target - len(selected_wave)),
            live_servers=live_servers,
            profile_headroom=headroom,
            run_weights={QUALIFICATION_RUN_ID: 1.0},
        )
        if not admission.selected:
            break
        batch_selected: list[dict[str, Any]] = []
        for candidate in admission.selected:
            if candidate.cell_id in selected_ids:
                raise ThroughputQualificationError(
                    "sequential capacity simulation selected a cell twice"
                )
            selected_ids.add(candidate.cell_id)
            cumulative_cost[candidate.serving_profile] += (
                candidate.fanout_cost
            )
            record = {
                "source_index": candidate.source_index,
                "cell_id": candidate.cell_id,
                "serving_profile": candidate.serving_profile,
                "fanout_cost": candidate.fanout_cost,
            }
            selected_wave.append(record)
            batch_selected.append(record)
        remaining = [
            candidate
            for candidate in remaining
            if candidate.cell_id not in selected_ids
        ]
        deficits = dict(admission.deficits)
        cursor = int(admission.cursor)
        batches.append(
            {
                "batch_index": len(batches),
                "cursor_before": cursor_before,
                "cursor_after": cursor,
                "deficits_before": deficits_before,
                "deficits_after": dict(sorted(deficits.items())),
                "selected": batch_selected,
                "selected_count": len(batch_selected),
                "cumulative_selected_count": len(selected_wave),
                "profile_headroom_before": {
                    profile: int(headroom[(server_pool_root, profile)])
                    for profile in sorted(profile_replicas)
                },
                "profile_headroom_after": {
                    profile: (
                        int(profile_replicas[profile])
                        * FANOUT_SLOTS_PER_REPLICA
                        - int(cumulative_cost[profile])
                    )
                    for profile in sorted(profile_replicas)
                },
            }
        )
    profile_summary = {
        profile: {
            "replicas": int(profile_replicas[profile]),
            "fanout_capacity": (
                int(profile_replicas[profile])
                * FANOUT_SLOTS_PER_REPLICA
            ),
            "plan_cells": int(plan_profile_cells[profile]),
            "plan_fanout": int(plan_profile_fanout[profile]),
            "selected_cells": sum(
                record["serving_profile"] == profile
                for record in selected_wave
            ),
            "selected_fanout": int(cumulative_cost[profile]),
            "ending_headroom": (
                int(profile_replicas[profile])
                * FANOUT_SLOTS_PER_REPLICA
                - int(cumulative_cost[profile])
            ),
            "remaining_candidate_cells": sum(
                candidate.serving_profile == profile
                for candidate in remaining
            ),
        }
        for profile in sorted(profile_replicas)
    }
    passed = len(selected_wave) == target
    return {
        "algorithm": PREFLIGHT_CAPACITY_ALGORITHM,
        "passed": passed,
        "target_active_cells": target,
        "maximum_microbatch": MAX_BATCH,
        "fanout_slots_per_replica": FANOUT_SLOTS_PER_REPLICA,
        "completion_assumption": "no_completions_during_certified_wave",
        "fairness_state": "persist_deficits_and_cursor_between_batches",
        "plan_cell_count": len(candidates),
        "plan_fanout_total": sum(
            candidate.fanout_cost for candidate in candidates
        ),
        "profile_replicas": {
            profile: int(profile_replicas[profile])
            for profile in sorted(profile_replicas)
        },
        "profile_summary": profile_summary,
        "selected_cell_count": len(selected_wave),
        "selected_cell_ids_sha256": _sha256_bytes(
            _canonical_bytes(
                [record["cell_id"] for record in selected_wave]
            )
        ),
        "selected_wave": selected_wave,
        "microbatch_count": len(batches),
        "microbatches": batches,
        "ending_cursor": cursor,
        "ending_deficits": dict(sorted(deficits.items())),
        "shortfall_cells": target - len(selected_wave),
    }


def theoretical_profile_packing_upper_bound(
    profile_replicas: Mapping[str, int],
) -> dict[str, Any]:
    """Return the cheapest-cell packing bound, never an admission certificate."""

    if (
        set(profile_replicas) != set(SERVING_PROFILES)
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            for count in profile_replicas.values()
        )
    ):
        raise ThroughputQualificationError(
            "theoretical packing bound requires positive profile replicas"
        )
    costs: dict[str, list[int]] = {
        profile: [] for profile in SERVING_PROFILES
    }
    for cell in generate_qualification_cells():
        costs[serving_profile_for_cell(cell).name].append(
            dispatch_sweeps.fanout_cost(cell)
        )
    profile_fit: dict[str, int] = {}
    for profile in sorted(costs):
        budget = (
            int(profile_replicas[profile])
            * FANOUT_SLOTS_PER_REPLICA
        )
        fitted = 0
        for cost in sorted(costs[profile]):
            if cost > budget:
                break
            budget -= cost
            fitted += 1
        profile_fit[profile] = fitted
    return {
        "kind": "non_executable_cheapest_cell_upper_bound",
        "profile_fit": profile_fit,
        "total_fit": sum(profile_fit.values()),
    }


def validate_preflight_capacity_wave(
    value: Mapping[str, Any],
    *,
    allow_generation_one_baseline_shortfall: bool = False,
    allow_intermediate_generation_shortfall: bool = False,
) -> dict[str, Any]:
    replicas = value.get("profile_replicas")
    if not isinstance(replicas, Mapping):
        raise ThroughputQualificationError(
            "preflight capacity wave lacks profile replicas"
        )
    expected = simulate_preflight_capacity_wave(
        {
            str(profile): int(count)
            for profile, count in replicas.items()
            if isinstance(count, int) and not isinstance(count, bool)
        }
    )
    if dict(value) != expected:
        raise ThroughputQualificationError(
            "preflight capacity wave differs from exact sequential WDRR "
            "recomputation"
        )
    if (
        allow_generation_one_baseline_shortfall
        and (
            expected["profile_replicas"] != dict(EXPECTED_COUNTS)
            or expected["passed"] is not False
            or expected["selected_cell_count"] != 278
            or expected["shortfall_cells"] != 106
            or expected["microbatch_count"] != 12
            or [
                batch["selected_count"]
                for batch in expected["microbatches"]
            ]
            != [24] * 11 + [14]
        )
    ):
        raise ThroughputQualificationError(
            "generation-one baseline capacity wave differs from the exact "
            "22-replica/24-GPU zero-delta shortfall"
        )
    if (
        not allow_generation_one_baseline_shortfall
        and not allow_intermediate_generation_shortfall
        and expected["passed"] is not True
    ):
        raise ThroughputQualificationError(
            "effective serving fleet cannot admit the exact 384-cell "
            "qualification wave"
        )
    if (
        expected["passed"] is True
        and not allow_generation_one_baseline_shortfall
        and (
        expected["microbatch_count"]
        != math.ceil(CEILINGS[-1] / MAX_BATCH)
        or any(
            batch["selected_count"] != MAX_BATCH
            for batch in expected["microbatches"]
        )
        or len(
            {
                record["cell_id"]
                for record in expected["selected_wave"]
            }
        )
        != CEILINGS[-1]
        or any(
            summary["selected_fanout"]
            > summary["fanout_capacity"]
            for summary in expected["profile_summary"].values()
        )
        )
    ):
        raise ThroughputQualificationError(
            "preflight capacity wave does not prove sixteen exact "
            "<=24-task batches within profile fanout limits"
        )
    return expected


_PREFLIGHT_CAPACITY_CERTIFICATE_FIELDS = {
    "schema_version",
    "protocol",
    "passed",
    "release_git_commit",
    "source_tree_sha256",
    "capacity_generation",
    "release_fleet_contract_sha256",
    "base_fleet_contract_sha256",
    "proposed_effective_fleet_contract_sha256",
    "additive_overlay_contract_sha256",
    "base_logical_replicas",
    "base_allocated_gpus",
    "effective_logical_replicas",
    "effective_active_gpus",
    "base_profile_replicas",
    "effective_profile_replicas",
    "additive_profile_delta",
    "additive_tp1_logical_replicas",
    "additive_tp2_logical_replicas",
    "additive_allocated_gpus",
    "additive_topology_sha256",
    "dispatcher_policy",
    "dispatcher_source_sha256",
    "qualification_plan_id",
    "qualification_runner_source_sha256",
    "fanout_slots_per_replica",
    "run_weights",
    "initial_fairness",
    "final_fairness",
    "microbatch_trace_sha256",
    "selected_cell_count",
    "selected_cell_ids_sha256",
    "wave",
    "certificate_id",
}


def build_preflight_capacity_certificate(
    *,
    capacity_generation: int,
    release_git_commit: str,
    source_tree_sha256: str,
    release_fleet_contract_sha256: str,
    base_fleet_contract_sha256: str,
    proposed_effective_fleet_contract_sha256: str,
    additive_overlay_contract_sha256: str,
    base_profile_replicas: Mapping[str, int],
    effective_profile_replicas: Mapping[str, int],
    dispatcher_source_sha256: str,
    qualification_runner_source_sha256: str,
) -> dict[str, Any]:
    """Build the marker-independent certificate consumed by protected capacity."""

    hashes = (
        source_tree_sha256,
        release_fleet_contract_sha256,
        base_fleet_contract_sha256,
        proposed_effective_fleet_contract_sha256,
        additive_overlay_contract_sha256,
        dispatcher_source_sha256,
        qualification_runner_source_sha256,
    )
    if (
        not isinstance(capacity_generation, int)
        or isinstance(capacity_generation, bool)
        or capacity_generation < 1
        or re.fullmatch(r"[0-9a-f]{40}", release_git_commit) is None
        or any(_SHA256_RE.fullmatch(value) is None for value in hashes)
        or release_fleet_contract_sha256
        != base_fleet_contract_sha256
        or additive_overlay_contract_sha256
        != proposed_effective_fleet_contract_sha256
        or set(base_profile_replicas) != set(SERVING_PROFILES)
        or set(effective_profile_replicas) != set(SERVING_PROFILES)
    ):
        raise ThroughputQualificationError(
            "preflight capacity certificate inputs are malformed"
        )
    base = {
        profile: int(base_profile_replicas[profile])
        for profile in sorted(SERVING_PROFILES)
    }
    effective = {
        profile: int(effective_profile_replicas[profile])
        for profile in sorted(SERVING_PROFILES)
    }
    if any(
        not isinstance(base_profile_replicas[profile], int)
        or isinstance(base_profile_replicas[profile], bool)
        or not isinstance(effective_profile_replicas[profile], int)
        or isinstance(effective_profile_replicas[profile], bool)
        or base[profile] < 1
        or effective[profile] < base[profile]
        for profile in SERVING_PROFILES
    ):
        raise ThroughputQualificationError(
            "effective fleet is not a profile-preserving non-reducing "
            "extension of the base fleet"
        )
    delta_counts = {
        profile: effective[profile] - base[profile]
        for profile in sorted(SERVING_PROFILES)
    }
    tp_sizes = {
        profile: int(SERVING_PROFILE_REGISTRY[profile].tp_size)
        for profile in SERVING_PROFILES
    }
    base_gpus = sum(
        base[profile] * tp_sizes[profile] for profile in SERVING_PROFILES
    )
    effective_gpus = sum(
        effective[profile] * tp_sizes[profile]
        for profile in SERVING_PROFILES
    )
    delta_tp1 = sum(
        delta_counts[profile]
        for profile in SERVING_PROFILES
        if tp_sizes[profile] == 1
    )
    delta_tp2 = sum(
        delta_counts[profile]
        for profile in SERVING_PROFILES
        if tp_sizes[profile] == 2
    )
    delta = {
        profile: {
            "logical_replicas": delta_counts[profile],
            "tensor_parallel_size": tp_sizes[profile],
            "allocated_gpus": (
                delta_counts[profile] * tp_sizes[profile]
            ),
        }
        for profile in sorted(SERVING_PROFILES)
    }
    zero_delta = not any(delta_counts.values())
    generation_one_baseline = (
        capacity_generation == 1
        and zero_delta
        and base == dict(EXPECTED_COUNTS)
        and effective == base
        and sum(base.values()) == 22
        and base_gpus == 24
        and proposed_effective_fleet_contract_sha256
        == base_fleet_contract_sha256
    )
    if capacity_generation == 1 and not generation_one_baseline:
        raise ThroughputQualificationError(
            "capacity generation one must bind the exact zero-delta "
            "22-replica/24-GPU base fleet"
        )
    if capacity_generation > 1 and (
        zero_delta
        or proposed_effective_fleet_contract_sha256
        == base_fleet_contract_sha256
    ):
        raise ThroughputQualificationError(
            "post-baseline capacity generations require a positive additive "
            "fleet transition"
        )
    wave = validate_preflight_capacity_wave(
        simulate_preflight_capacity_wave(effective),
        allow_generation_one_baseline_shortfall=generation_one_baseline,
        allow_intermediate_generation_shortfall=capacity_generation > 1,
    )
    initial_fairness = {"cursor": 0, "deficits": {}}
    final_fairness = {
        "cursor": wave["ending_cursor"],
        "deficits": wave["ending_deficits"],
    }
    identity = {
        "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
        "protocol": PREFLIGHT_CAPACITY_PROTOCOL,
        "passed": True,
        "release_git_commit": release_git_commit,
        "source_tree_sha256": source_tree_sha256,
        "capacity_generation": capacity_generation,
        "release_fleet_contract_sha256": (
            release_fleet_contract_sha256
        ),
        "base_fleet_contract_sha256": base_fleet_contract_sha256,
        "proposed_effective_fleet_contract_sha256": (
            proposed_effective_fleet_contract_sha256
        ),
        "additive_overlay_contract_sha256": (
            additive_overlay_contract_sha256
        ),
        "base_logical_replicas": sum(base.values()),
        "base_allocated_gpus": base_gpus,
        "effective_logical_replicas": sum(effective.values()),
        "effective_active_gpus": effective_gpus,
        "base_profile_replicas": base,
        "effective_profile_replicas": effective,
        "additive_profile_delta": delta,
        "additive_tp1_logical_replicas": delta_tp1,
        "additive_tp2_logical_replicas": delta_tp2,
        "additive_allocated_gpus": effective_gpus - base_gpus,
        "additive_topology_sha256": _sha256_bytes(
            _canonical_bytes(
                [
                    {
                        "serving_profile": profile,
                        "replicas": delta_counts[profile],
                        "tensor_parallel_size": tp_sizes[profile],
                        "allocated_gpus": (
                            delta_counts[profile] * tp_sizes[profile]
                        ),
                    }
                    for profile in sorted(SERVING_PROFILES)
                    if delta_counts[profile]
                ]
            )
        ),
        "dispatcher_policy": PREFLIGHT_CAPACITY_ALGORITHM,
        "dispatcher_source_sha256": dispatcher_source_sha256,
        "qualification_plan_id": build_load_plan()["plan_id"],
        "qualification_runner_source_sha256": (
            qualification_runner_source_sha256
        ),
        "fanout_slots_per_replica": FANOUT_SLOTS_PER_REPLICA,
        "run_weights": {QUALIFICATION_RUN_ID: 1.0},
        "initial_fairness": initial_fairness,
        "final_fairness": final_fairness,
        "microbatch_trace_sha256": _sha256_bytes(
            _canonical_bytes(wave["microbatches"])
        ),
        "selected_cell_count": wave["selected_cell_count"],
        "selected_cell_ids_sha256": wave[
            "selected_cell_ids_sha256"
        ],
        "wave": wave,
    }
    return _with_identity(identity, "certificate_id")


def validate_preflight_capacity_certificate(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if set(value) != _PREFLIGHT_CAPACITY_CERTIFICATE_FIELDS:
        raise ThroughputQualificationError(
            "preflight capacity certificate fields drifted"
        )
    _verify_identity(
        value,
        "certificate_id",
        description="preflight capacity certificate",
    )
    rebuilt = build_preflight_capacity_certificate(
        capacity_generation=int(value["capacity_generation"]),
        release_git_commit=str(value["release_git_commit"]),
        source_tree_sha256=str(value["source_tree_sha256"]),
        release_fleet_contract_sha256=str(
            value["release_fleet_contract_sha256"]
        ),
        base_fleet_contract_sha256=str(
            value["base_fleet_contract_sha256"]
        ),
        proposed_effective_fleet_contract_sha256=str(
            value["proposed_effective_fleet_contract_sha256"]
        ),
        additive_overlay_contract_sha256=str(
            value["additive_overlay_contract_sha256"]
        ),
        base_profile_replicas=value["base_profile_replicas"],
        effective_profile_replicas=value[
            "effective_profile_replicas"
        ],
        dispatcher_source_sha256=str(value["dispatcher_source_sha256"]),
        qualification_runner_source_sha256=str(
            value["qualification_runner_source_sha256"]
        ),
    )
    if dict(value) != rebuilt:
        raise ThroughputQualificationError(
            "preflight capacity certificate differs from exact source-bound "
            "sequential-policy recomputation"
        )
    return rebuilt


def _verify_preflight_release_source_binding(
    *,
    release_git_commit: str,
    source_tree_sha256: str,
    dispatcher_source: Path,
    qualification_runner_source: Path,
) -> dict[str, str]:
    """Bind certificate source hashes to the clean exact annotated release."""

    if (
        dispatcher_source
        != (REPO / "slurm" / "dispatch_sweeps.py").resolve()
        or qualification_runner_source != Path(__file__).resolve()
    ):
        raise ThroughputQualificationError(
            "preflight sources are not the exact release-worktree paths"
        )

    def run_git(
        *arguments: str,
        text: bool,
    ) -> subprocess.CompletedProcess[Any]:
        try:
            result = subprocess.run(
                ["git", "-C", str(REPO), *arguments],
                check=False,
                capture_output=True,
                text=text,
                timeout=60.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ThroughputQualificationError(
                f"cannot verify preflight release Git identity: {exc}"
            ) from exc
        if result.returncode != 0:
            stderr = result.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            raise ThroughputQualificationError(
                "cannot verify preflight release Git identity: "
                + str(stderr).strip()
            )
        return result

    def git(*arguments: str) -> str:
        output = run_git(*arguments, text=True).stdout
        if not isinstance(output, str):
            raise ThroughputQualificationError(
                "preflight Git text query returned non-text output"
            )
        return output.strip()

    def git_bytes(*arguments: str) -> bytes:
        output = run_git(*arguments, text=False).stdout
        if not isinstance(output, bytes):
            raise ThroughputQualificationError(
                "preflight Git blob query returned non-byte output"
            )
        return output

    head = git("rev-parse", "HEAD")
    top_level = Path(git("rev-parse", "--show-toplevel")).resolve()
    tag_object = git("rev-parse", f"refs/tags/{renderer.RELEASE_TAG}")
    tag_type = git("cat-file", "-t", tag_object)
    peeled = git("rev-parse", f"{tag_object}^{{commit}}")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    tagged_sources: dict[str, bytes] = {}
    for relative in (
        "slurm/dispatch_sweeps.py",
        "scripts/run_schema5_throughput_qualification.py",
    ):
        blob_id = git("rev-parse", f"{peeled}:{relative}")
        if (
            re.fullmatch(r"[0-9a-f]{40}", blob_id) is None
            or git("cat-file", "-t", blob_id) != "blob"
        ):
            raise ThroughputQualificationError(
                f"preflight tagged source is not one Git blob: {relative}"
            )
        tagged_sources[relative] = git_bytes(
            "cat-file", "blob", blob_id
        )
    final_head = git("rev-parse", "HEAD")
    final_tag_object = git(
        "rev-parse", f"refs/tags/{renderer.RELEASE_TAG}"
    )
    try:
        observed_source_tree = control.sha256_tree(REPO)
    except (OSError, control.ControlError) as exc:
        raise ThroughputQualificationError(
            f"cannot hash exact preflight release source tree: {exc}"
        ) from exc
    try:
        dispatcher_bytes = dispatcher_source.read_bytes()
        qualification_runner_bytes = qualification_runner_source.read_bytes()
    except OSError as exc:
        raise ThroughputQualificationError(
            f"cannot read exact preflight release sources: {exc}"
        ) from exc
    if (
        re.fullmatch(r"[0-9a-f]{40}", release_git_commit) is None
        or _SHA256_RE.fullmatch(source_tree_sha256) is None
        or top_level != REPO.resolve()
        or head != release_git_commit
        or peeled != release_git_commit
        or tag_type != "tag"
        or re.fullmatch(r"[0-9a-f]{40}", tag_object) is None
        or status
        or final_head != head
        or final_tag_object != tag_object
        or observed_source_tree != source_tree_sha256
        or dispatcher_bytes != tagged_sources["slurm/dispatch_sweeps.py"]
        or qualification_runner_bytes
        != tagged_sources[
            "scripts/run_schema5_throughput_qualification.py"
        ]
    ):
        raise ThroughputQualificationError(
            "preflight certificate is not bound to the clean exact annotated "
            "release and its computed source-tree hash"
        )
    return {
        "release_git_commit": head,
        "release_tag_object": tag_object,
        "source_tree_sha256": observed_source_tree,
        "dispatcher_source_sha256": _sha256_bytes(
            tagged_sources["slurm/dispatch_sweeps.py"]
        ),
        "qualification_runner_source_sha256": _sha256_bytes(
            tagged_sources[
                "scripts/run_schema5_throughput_qualification.py"
            ]
        ),
    }


def preflight_capacity_report(
    *,
    base_fleet_contract: Path,
    effective_fleet_contract: Path,
    additive_overlay_contract: Path,
    capacity_generation: int,
    release_git_commit: str,
    source_tree_sha256: str,
    dispatcher_source: Path,
    qualification_runner_source: Path,
    output: Path,
    apply: bool,
) -> dict[str, Any]:
    """Certify a zero-QID effective overlay before any qualification attempt."""

    paths = {
        "base fleet contract": base_fleet_contract,
        "effective fleet contract": effective_fleet_contract,
        "additive overlay contract": additive_overlay_contract,
        "dispatcher source": dispatcher_source,
        "qualification runner source": qualification_runner_source,
    }
    resolved: dict[str, Path] = {}
    for description, path in paths.items():
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise ThroughputQualificationError(
                f"{description} path is not absolute"
            )
        lexical = Path(os.path.abspath(os.fspath(candidate)))
        if (
            not lexical.is_file()
            or lexical.is_symlink()
            or lexical.resolve(strict=True) != lexical
        ):
            raise ThroughputQualificationError(
                f"{description} is missing, aliased, or unsafe"
            )
        resolved[description] = lexical
    expected_sources = {
        "dispatcher source": Path(
            str(dispatch_sweeps.__file__)
        ).resolve(),
        "qualification runner source": Path(__file__).resolve(),
    }
    for description, expected_path in expected_sources.items():
        source_path = resolved[description]
        metadata = source_path.stat()
        if (
            source_path != expected_path
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o222
        ):
            raise ThroughputQualificationError(
                f"{description} is not the exact immutable imported source"
            )
    source_binding = _verify_preflight_release_source_binding(
        release_git_commit=release_git_commit,
        source_tree_sha256=source_tree_sha256,
        dispatcher_source=resolved["dispatcher source"],
        qualification_runner_source=resolved[
            "qualification runner source"
        ],
    )
    output = Path(os.path.abspath(os.fspath(output.expanduser())))
    if (
        not output.is_absolute()
        or output.name != PREFLIGHT_CAPACITY_CERTIFICATE_NAME
        or output.is_symlink()
        or (
            capacity_generation > 1
            and (
                output.parent.name
                != f"c{capacity_generation:06d}"
                or output.parent.parent.name != "capacity-generations"
            )
        )
    ):
        raise ThroughputQualificationError(
            "preflight certificate output must be an absolute canonical "
            f"{PREFLIGHT_CAPACITY_CERTIFICATE_NAME} path"
        )
    base_path = resolved["base fleet contract"]
    effective_path = resolved["effective fleet contract"]
    overlay_path = resolved["additive overlay contract"]
    if overlay_path != effective_path:
        raise ThroughputQualificationError(
            "the additive overlay must be the exact proposed effective fleet "
            "contract; unrelated overlay hashes are forbidden"
        )
    base_contract, effective_contract = _load_preflight_fleet_contracts(
        base_path=base_path,
        effective_path=effective_path,
    )
    if (
        _sha256_file(resolved["dispatcher source"])
        != source_binding["dispatcher_source_sha256"]
        or _sha256_file(resolved["qualification runner source"])
        != source_binding["qualification_runner_source_sha256"]
    ):
        raise ThroughputQualificationError(
            "preflight tagged release sources changed after verification"
        )
    base_replicas = {
        profile: len(base_contract.by_profile[profile])
        for profile in sorted(SERVING_PROFILES)
    }
    effective_replicas = {
        profile: len(effective_contract.by_profile[profile])
        for profile in sorted(SERVING_PROFILES)
    }
    wave = simulate_preflight_capacity_wave(effective_replicas)
    certificate = build_preflight_capacity_certificate(
        capacity_generation=capacity_generation,
        release_git_commit=release_git_commit,
        source_tree_sha256=source_tree_sha256,
        release_fleet_contract_sha256=_sha256_file(base_path),
        base_fleet_contract_sha256=_sha256_file(base_path),
        proposed_effective_fleet_contract_sha256=_sha256_file(
            effective_path
        ),
        additive_overlay_contract_sha256=_sha256_file(overlay_path),
        base_profile_replicas=base_replicas,
        effective_profile_replicas=effective_replicas,
        dispatcher_source_sha256=source_binding[
            "dispatcher_source_sha256"
        ],
        qualification_runner_source_sha256=source_binding[
            "qualification_runner_source_sha256"
        ],
    )
    if apply:
        _write_once(
            output,
            certificate,
            description="preflight capacity certificate",
        )
        observed = validate_preflight_capacity_certificate(
            _read_json(
                output,
                description="preflight capacity certificate",
                sealed=True,
            )
        )
        if observed != certificate:
            raise ThroughputQualificationError(
                "published preflight capacity certificate replay drifted"
            )
    return {
        "status": "complete" if apply else "dry_run",
        "passed": True,
        "submitted": False,
        "configured_client_ceiling": wave["target_active_cells"],
        "certified_saturation_target": wave["selected_cell_count"],
        "full_ceiling_fit": wave["passed"],
        "certificate": certificate,
        "certificate_path": str(output) if apply else None,
    }


def verify_authorized_preflight_capacity(
    context: QualificationContext,
    *,
    control_value: Mapping[str, Any],
    readiness_generation: Mapping[str, Any],
) -> dict[str, Any]:
    """Join the marker-bound certificate to exact fleet and release bytes."""

    marker = _read_json(
        context.protected_capacity_contract.path,
        description="protected capacity authority",
        sealed=True,
    )
    certificate_binding = marker.get(
        "static_feasibility_certificate"
    )
    if (
        not isinstance(certificate_binding, Mapping)
        or set(certificate_binding)
        != {"path", "sha256", "certificate_id"}
    ):
        raise ThroughputQualificationError(
            "protected capacity lacks the static feasibility certificate"
        )
    certificate_path = Path(str(certificate_binding.get("path", "")))
    expected_generation = int(
        context.protected_capacity_contract.capacity_generation
    )
    generation_one_path = (
        context.readiness_root / PREFLIGHT_CAPACITY_CERTIFICATE_NAME
    ).resolve()
    expected_path = (
        context.protected_capacity_contract
        .static_feasibility_certificate_path
        .resolve()
    )
    try:
        expected_path.relative_to(context.readiness_root)
    except ValueError as exc:
        raise ThroughputQualificationError(
            "static feasibility certificate escapes the readiness root"
        ) from exc
    if (
        not certificate_path.is_absolute()
        or certificate_path != expected_path
        or (
            expected_generation == 1
            and certificate_path != generation_one_path
        )
        or (
            expected_generation > 1
            and certificate_path == generation_one_path
        )
        or not certificate_path.is_file()
        or certificate_path.is_symlink()
        or certificate_path.stat().st_nlink != 1
        or stat.S_IMODE(certificate_path.stat().st_mode) & 0o222
        or _sha256_file(certificate_path)
        != certificate_binding.get("sha256")
    ):
        raise ThroughputQualificationError(
            "static feasibility certificate path/bytes are unsafe or drifted"
        )
    certificate = validate_preflight_capacity_certificate(
        _read_json(
            certificate_path,
            description="static feasibility certificate",
            sealed=True,
        )
    )
    if (
        certificate["certificate_id"]
        != certificate_binding.get("certificate_id")
    ):
        raise ThroughputQualificationError(
            "protected capacity binds a different feasibility certificate"
        )
    immutable = control_value.get("immutable")
    if not isinstance(immutable, Mapping):
        raise ThroughputQualificationError(
            "paused control lacks immutable release pins"
        )
    try:
        effective = control.effective_fleet_contract_binding(
            control_value, verify_files=True
        )
    except control.ControlError as exc:
        raise ThroughputQualificationError(
            f"effective fleet cannot be joined to capacity certificate: {exc}"
        ) from exc
    base_path = Path(str(immutable.get("fleet_contract_path", "")))
    effective_path = Path(str(effective.get("path", "")))
    if (
        not base_path.is_absolute()
        or not effective_path.is_absolute()
        or not base_path.is_file()
        or base_path.is_symlink()
        or not effective_path.is_file()
        or effective_path.is_symlink()
    ):
        raise ThroughputQualificationError(
            "certificate fleet paths are missing or unsafe"
        )
    base_contract, effective_contract = _load_preflight_fleet_contracts(
        base_path=base_path.resolve(),
        effective_path=effective_path.resolve(),
    )
    base_counts = {
        profile: len(base_contract.by_profile[profile])
        for profile in sorted(SERVING_PROFILES)
    }
    effective_counts = {
        profile: len(effective_contract.by_profile[profile])
        for profile in sorted(SERVING_PROFILES)
    }
    if (
        context.protected_capacity_contract.base_fleet_contract_path
        != base_path.resolve()
        or context.protected_capacity_contract.effective_fleet_contract_path
        != effective_path.resolve()
        or context.protected_capacity_contract.additive_overlay_contract_path
        != effective_path.resolve()
    ):
        raise ThroughputQualificationError(
            "protected capacity and paused control bind different fleet paths"
        )
    dispatcher_source = (
        context.release_worktree / "slurm" / "dispatch_sweeps.py"
    ).resolve()
    runner_source = (
        context.release_worktree
        / "scripts"
        / Path(__file__).name
    ).resolve()
    for description, source in (
        ("dispatcher", dispatcher_source),
        ("qualification runner", runner_source),
    ):
        if (
            not source.is_file()
            or source.is_symlink()
            or source.stat().st_nlink != 1
            or stat.S_IMODE(source.stat().st_mode) & 0o222
        ):
            raise ThroughputQualificationError(
                f"frozen {description} source is missing or mutable"
            )
    rebuilt = build_preflight_capacity_certificate(
        capacity_generation=int(effective["capacity_generation"]),
        release_git_commit=str(immutable.get("git_commit", "")),
        source_tree_sha256=str(
            immutable.get("source_tree_sha256", "")
        ),
        release_fleet_contract_sha256=_sha256_file(base_path),
        base_fleet_contract_sha256=_sha256_file(base_path),
        proposed_effective_fleet_contract_sha256=_sha256_file(
            effective_path
        ),
        additive_overlay_contract_sha256=_sha256_file(effective_path),
        base_profile_replicas=base_counts,
        effective_profile_replicas=effective_counts,
        dispatcher_source_sha256=_sha256_file(dispatcher_source),
        qualification_runner_source_sha256=_sha256_file(
            runner_source
        ),
    )
    if (
        certificate != rebuilt
        or certificate["qualification_plan_id"]
        != build_load_plan()["plan_id"]
        or readiness_generation["release_fleet_contract_sha256"]
        != certificate["release_fleet_contract_sha256"]
        or readiness_generation["fleet_contract_sha256"]
        != certificate["proposed_effective_fleet_contract_sha256"]
        or readiness_generation["capacity_generation"]
        != certificate["capacity_generation"]
        or effective.get("profile_replicas")
        != certificate["effective_profile_replicas"]
        or effective.get("logical_replicas")
        != certificate["effective_logical_replicas"]
        or effective.get("allocated_gpus")
        != certificate["effective_active_gpus"]
    ):
        raise ThroughputQualificationError(
            "static feasibility certificate is not the exact release, "
            "plan, fleet, source, and trusted readiness generation"
        )
    packing = theoretical_profile_packing_upper_bound(effective_counts)
    wave = certificate["wave"]
    if (
        not isinstance(wave, Mapping)
        or not isinstance(wave.get("passed"), bool)
        or not isinstance(wave.get("target_active_cells"), int)
        or isinstance(wave.get("target_active_cells"), bool)
        or not isinstance(wave.get("shortfall_cells"), int)
        or isinstance(wave.get("shortfall_cells"), bool)
        or packing.get("kind")
        != "non_executable_cheapest_cell_upper_bound"
        or not isinstance(packing.get("total_fit"), int)
    ):
        raise ThroughputQualificationError(
            "static feasibility certificate lacks exact admission accounting"
        )
    return {
        "path": str(certificate_path),
        "sha256": _sha256_file(certificate_path),
        "certificate_id": certificate["certificate_id"],
        "capacity_generation": certificate["capacity_generation"],
        "effective_fleet_contract_sha256": certificate[
            "proposed_effective_fleet_contract_sha256"
        ],
        "effective_logical_replicas": certificate[
            "effective_logical_replicas"
        ],
        "effective_active_gpus": certificate["effective_active_gpus"],
        "wave_passed": wave["passed"],
        "selected_cell_count": certificate["selected_cell_count"],
        "target_cell_count": wave["target_active_cells"],
        "shortfall_cells": wave["shortfall_cells"],
        "theoretical_packing_upper_bound": packing["total_fit"],
        "selected_cell_ids_sha256": certificate[
            "selected_cell_ids_sha256"
        ],
    }


def _validated_admission_capacity_certificate_binding(
    context: QualificationContext,
    value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and compact one attempt's immutable admission certificate."""

    if value is None:
        certificate_path = (
            context.protected_capacity_contract
            .static_feasibility_certificate_path
            .resolve()
        )
        raw_binding: Mapping[str, Any] = {
            "path": str(certificate_path),
            "sha256": (
                context.protected_capacity_contract
                .static_feasibility_certificate_sha256
            ),
            "certificate_id": (
                context.protected_capacity_contract
                .static_feasibility_certificate_id
            ),
        }
    else:
        raw_binding = value
        certificate_path = Path(str(raw_binding.get("path", "")))
    if (
        not certificate_path.is_absolute()
        or certificate_path.is_symlink()
        or not certificate_path.is_file()
        or certificate_path.resolve() != certificate_path
        or certificate_path.stat().st_nlink != 1
        or stat.S_IMODE(certificate_path.stat().st_mode) & 0o222
    ):
        raise ThroughputQualificationError(
            "admission capacity certificate path is unsafe"
        )
    try:
        certificate_path.relative_to(context.readiness_root)
    except ValueError as exc:
        raise ThroughputQualificationError(
            "admission capacity certificate escapes the readiness root"
        ) from exc
    certificate = validate_preflight_capacity_certificate(
        _read_json(
            certificate_path,
            description="attempt admission capacity certificate",
            sealed=True,
        )
    )
    effective_counts = {
        str(profile): int(count)
        for profile, count in dict(
            certificate["effective_profile_replicas"]
        ).items()
    }
    packing = theoretical_profile_packing_upper_bound(effective_counts)
    wave = certificate["wave"]
    compact = {
        "path": str(certificate_path),
        "sha256": _sha256_file(certificate_path),
        "certificate_id": certificate["certificate_id"],
        "capacity_generation": certificate["capacity_generation"],
        "effective_fleet_contract_sha256": certificate[
            "proposed_effective_fleet_contract_sha256"
        ],
        "effective_logical_replicas": certificate[
            "effective_logical_replicas"
        ],
        "effective_active_gpus": certificate["effective_active_gpus"],
        "wave_passed": wave["passed"],
        "selected_cell_count": certificate["selected_cell_count"],
        "target_cell_count": wave["target_active_cells"],
        "shortfall_cells": wave["shortfall_cells"],
        "theoretical_packing_upper_bound": packing["total_fit"],
    }
    readiness = (
        context.attempt_pointer.get("readiness_generation")
        if isinstance(context.attempt_pointer, Mapping)
        else None
    )
    generation_one_path = (
        context.readiness_root / PREFLIGHT_CAPACITY_CERTIFICATE_NAME
    ).resolve()
    if (
        set(compact) != _ADMISSION_CAPACITY_CERTIFICATE_FIELDS
        or raw_binding.get("path") != compact["path"]
        or raw_binding.get("sha256") != compact["sha256"]
        or raw_binding.get("certificate_id") != compact["certificate_id"]
        or certificate.get("release_git_commit")
        != context.release_git_commit
        or certificate.get("source_tree_sha256")
        != context.source_tree_sha256
        or certificate.get("dispatcher_source_sha256")
        != context.dispatcher_source_sha256
        or certificate.get("qualification_runner_source_sha256")
        != context.qualification_runner_source_sha256
        or (
            compact["capacity_generation"] == 1
            and certificate_path != generation_one_path
        )
        or (
            compact["capacity_generation"] > 1
            and certificate_path == generation_one_path
        )
        or (
            isinstance(readiness, Mapping)
            and (
                readiness.get("capacity_generation")
                != compact["capacity_generation"]
                or readiness.get("fleet_contract_sha256")
                != compact["effective_fleet_contract_sha256"]
            )
        )
        or (
            value is not None
            and (
                set(value) != _ADMISSION_CAPACITY_CERTIFICATE_FIELDS
                or dict(value) != compact
            )
        )
    ):
        raise ThroughputQualificationError(
            "attempt admission capacity certificate binding drifted"
        )
    return compact


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


def _load_frozen_release_source_authority(
    *,
    release_root: Path,
    release_worktree: Path,
    expected_release_git_commit: str,
    expected_release_tag_object: str,
) -> dict[str, str]:
    """Derive capacity source pins from the sealed release, never its marker."""

    release_root = Path(release_root).resolve()
    release_worktree = Path(release_worktree).resolve()
    identity_path = (
        release_root
        / "identity"
        / "release_identity.schema5-v1.json"
    )
    checksum_path = Path(str(identity_path) + ".sha256")
    _, _, identity_sha256 = _require_sealed_preflight_file(
        identity_path,
        description="frozen release identity",
    )
    _require_sealed_preflight_file(
        checksum_path,
        description="frozen release identity checksum",
    )
    try:
        checksum_fields = checksum_path.read_text(
            encoding="utf-8"
        ).strip().split()
    except (OSError, UnicodeError) as exc:
        raise ThroughputQualificationError(
            f"cannot read frozen release identity checksum: {exc}"
        ) from exc
    identity = _read_json(
        identity_path,
        description="frozen release identity",
        sealed=True,
    )
    git_identity = identity.get("git")
    fragment = identity.get("control_pin_fragment")
    source_tree_sha256 = (
        str(git_identity.get("source_tree_sha256", ""))
        if isinstance(git_identity, Mapping)
        else ""
    )
    if (
        checksum_fields != [identity_sha256, identity_path.name]
        or identity.get("release_id") != renderer.RELEASE_ID
        or identity.get("release_worktree") != str(release_worktree)
        or identity.get("worktree_sealed_read_only") is not True
        or not isinstance(git_identity, Mapping)
        or set(git_identity)
        != {
            "git_commit",
            "git_tag",
            "git_tag_object",
            "source_tree_sha256",
        }
        or git_identity.get("git_commit")
        != expected_release_git_commit
        or git_identity.get("git_tag") != renderer.RELEASE_TAG
        or git_identity.get("git_tag_object")
        != expected_release_tag_object
        or _SHA256_RE.fullmatch(source_tree_sha256) is None
        or not isinstance(fragment, Mapping)
        or fragment.get("release_id") != renderer.RELEASE_ID
        or fragment.get("release_worktree") != str(release_worktree)
        or fragment.get("git_commit") != expected_release_git_commit
        or fragment.get("release_tag_object")
        != expected_release_tag_object
        or fragment.get("source_tree_sha256")
        != source_tree_sha256
    ):
        raise ThroughputQualificationError(
            "frozen release identity does not bind the exact annotated tag "
            "object, worktree, and control source pins"
        )
    try:
        observed_tree_sha256 = control.sha256_tree(release_worktree)
    except (OSError, control.ControlError) as exc:
        raise ThroughputQualificationError(
            f"cannot hash frozen release worktree: {exc}"
        ) from exc
    dispatcher = release_worktree / "slurm" / "dispatch_sweeps.py"
    qualification_runner = (
        release_worktree
        / "scripts"
        / "run_schema5_throughput_qualification.py"
    )
    sources: dict[str, str] = {}
    for key, path, description in (
        (
            "dispatcher_source_sha256",
            dispatcher,
            "frozen dispatcher source",
        ),
        (
            "qualification_runner_source_sha256",
            qualification_runner,
            "frozen qualification runner source",
        ),
    ):
        resolved, _, digest = _require_sealed_preflight_file(
            path,
            description=description,
        )
        if resolved != path:
            raise ThroughputQualificationError(
                f"{description} does not occupy its canonical release path"
            )
        sources[key] = digest
    if (
        observed_tree_sha256 != source_tree_sha256
        or _sha256_file(identity_path) != identity_sha256
    ):
        raise ThroughputQualificationError(
            "frozen release source or identity changed during authority load"
        )
    return {
        "release_git_commit": expected_release_git_commit,
        "release_tag_object": expected_release_tag_object,
        "source_tree_sha256": source_tree_sha256,
        **sources,
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
        or manifest.get("protocol") != RECOVERY_CHAIN_PROTOCOL
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
        source_authority = _load_frozen_release_source_authority(
            release_root=release_root,
            release_worktree=release_root / "worktree",
            expected_release_git_commit=release_git_commit,
            expected_release_tag_object=release_tag_object,
        )
        protected_contract = protected_capacity.load_contract(
            protected_binding["marker"],
            expected_release_git_commit=release_git_commit,
            expected_release_tag_object=release_tag_object,
            expected_marker_id=protected_binding["marker_id"],
            expected_sha256=protected_binding["marker_sha256"],
            expected_source_tree_sha256=source_authority[
                "source_tree_sha256"
            ],
            expected_dispatcher_source_sha256=source_authority[
                "dispatcher_source_sha256"
            ],
            expected_qualification_runner_source_sha256=source_authority[
                "qualification_runner_source_sha256"
            ],
        )
    except (
        OSError,
        protected_capacity.ProtectedCapacityError,
    ) as exc:
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
        source_tree_sha256=source_authority["source_tree_sha256"],
        dispatcher_source_sha256=source_authority[
            "dispatcher_source_sha256"
        ],
        qualification_runner_source_sha256=source_authority[
            "qualification_runner_source_sha256"
        ],
        protected_capacity=protected_binding,
        protected_capacity_contract=protected_contract,
    )


def _with_effective_protected_capacity(
    context: QualificationContext,
    control_value: Mapping[str, Any],
) -> QualificationContext:
    """Bind an attempt to the current generation, preserving gen-1 manifest pins."""

    try:
        authority = control.effective_protected_capacity_binding(
            control_value,
            verify_files=True,
        )
        contract = control.load_effective_protected_capacity_contract(
            control_value,
            verify_files=True,
        )
    except control.ControlError as exc:
        raise ThroughputQualificationError(
            f"current protected-capacity authority is invalid: {exc}"
        ) from exc
    compact = {
        "marker": str(Path(str(authority["path"])).resolve()),
        "marker_sha256": str(authority["sha256"]),
        "marker_id": str(authority["marker_id"]),
    }
    if (
        contract.path != Path(compact["marker"])
        or contract.sha256 != compact["marker_sha256"]
        or contract.marker_id != compact["marker_id"]
        or contract.capacity_generation
        != int(authority["capacity_generation"])
    ):
        raise ThroughputQualificationError(
            "current protected-capacity marker and contract disagree"
        )
    return replace(
        context,
        protected_capacity=compact,
        protected_capacity_contract=contract,
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
    readiness = value.get("readiness")
    required_gates = (
        "static_feasibility_certificate",
        "protected_capacity",
        "smoke_runs",
    )
    if (
        value.get("desired_state") != "paused"
        or value.get("drain_requested") is not False
        or set(guard["production_run_ids"]) != PRODUCTION_RUN_IDS
        or QUALIFICATION_RUN_ID in guard["production_run_ids"]
        or not isinstance(readiness, Mapping)
        or any(
            not isinstance(readiness.get(gate), Mapping)
            or readiness[gate].get("passed") is not True
            for gate in required_gates
        )
        or _SHA256_RE.fullmatch(str(value.get("immutable_sha256", ""))) is None
    ):
        raise ThroughputQualificationError(
            "qualification requires paused production control with current "
            "static-capacity, protected-capacity, and smoke readiness for "
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
    context = replace(
        base,
        qualification_root=Path(str(pointer["attempt_root"])),
        run_root=Path(str(pointer["run_root"])),
        dispatcher_state=Path(str(pointer["dispatcher_state"])),
        attempt_pointer_path=path,
        attempt_pointer=dict(pointer),
    )
    intent_path = context.qualification_root / INTENT_NAME
    if not intent_path.exists() and not intent_path.is_symlink():
        return context
    intent = _read_json(
        intent_path,
        description="generation-scoped qualification intent",
        sealed=True,
    )
    binding = intent.get("protected_capacity")
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"marker", "marker_sha256", "marker_id"}
    ):
        raise ThroughputQualificationError(
            "qualification intent protected-capacity binding is malformed"
        )
    try:
        contract = protected_capacity.load_contract(
            binding["marker"],
            expected_release_git_commit=context.release_git_commit,
            expected_release_tag_object=context.release_tag_object,
            expected_marker_id=str(binding["marker_id"]),
            expected_sha256=str(binding["marker_sha256"]),
            expected_source_tree_sha256=context.source_tree_sha256,
            expected_dispatcher_source_sha256=(
                context.dispatcher_source_sha256
            ),
            expected_qualification_runner_source_sha256=(
                context.qualification_runner_source_sha256
            ),
        )
    except protected_capacity.ProtectedCapacityError as exc:
        raise ThroughputQualificationError(
            f"attempt protected-capacity authority is invalid: {exc}"
        ) from exc
    readiness = pointer["readiness_generation"]
    if (
        contract.capacity_generation
        != readiness["capacity_generation"]
        or contract.effective_fleet_contract_sha256
        != readiness["fleet_contract_sha256"]
    ):
        raise ThroughputQualificationError(
            "attempt protected-capacity and readiness generations disagree"
        )
    return replace(
        context,
        protected_capacity=dict(binding),
        protected_capacity_contract=contract,
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


def _require_sealed_preflight_file(
    path: Path,
    *,
    description: str,
) -> tuple[Path, os.stat_result, str]:
    """Return one stable, canonical, unique, read-only preflight artifact."""

    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        before = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ThroughputQualificationError(
            f"{description} is unavailable or noncanonical: {exc}"
        ) from exc
    if (
        lexical != resolved
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) & 0o222
    ):
        raise ThroughputQualificationError(
            f"{description} is not one canonical unique read-only file"
        )
    first_sha256 = _sha256_file(lexical)
    after = lexical.stat(follow_symlinks=False)
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
        for field in stable_fields
    ) or _sha256_file(lexical) != first_sha256:
        raise ThroughputQualificationError(
            f"{description} changed during its sealed read"
        )
    return lexical, after, first_sha256


def _load_preflight_fleet_contracts(
    *,
    base_path: Path,
    effective_path: Path,
) -> tuple[FrozenFleetContract, FrozenFleetContract]:
    """Parse the exact release model/base/effective topology before simulation."""

    base_path = Path(base_path).resolve()
    effective_path = Path(effective_path).resolve()
    model_path = (REPO / "configs" / "model_contracts.v1.json").resolve()
    sealed_inputs = (
        (model_path, "release model contract"),
        (
            model_path.with_name("model_contracts.v1.sha256"),
            "release model-contract checksum",
        ),
        (base_path, "base release fleet contract"),
        (
            base_path.with_suffix(".sha256"),
            "base release fleet checksum",
        ),
        (effective_path, "effective additive fleet contract"),
        (
            effective_path.with_suffix(".sha256"),
            "effective additive fleet checksum",
        ),
    )
    input_hashes: dict[Path, str] = {}
    for path, description in sealed_inputs:
        resolved, _metadata, digest = _require_sealed_preflight_file(
            path,
            description=description,
        )
        input_hashes[resolved] = digest
    try:
        models = load_model_contracts(model_path)
        base = load_fleet_contract(
            base_path,
            model_contracts=models,
            expected_sha256=input_hashes[base_path],
            allow_capacity_layout=False,
        )
        effective = load_fleet_contract(
            effective_path,
            model_contracts=models,
            expected_sha256=input_hashes[effective_path],
            allow_capacity_layout=True,
        )
        control._assert_additive_capacity_contract(  # noqa: SLF001
            base_path, effective_path
        )
    except (
        FleetContractError,
        ModelContractError,
        OSError,
        ValueError,
        control.ControlError,
    ) as exc:
        raise ThroughputQualificationError(
            f"preflight fleet/model topology failed closed: {exc}"
        ) from exc
    for path, expected in input_hashes.items():
        if _sha256_file(path) != expected:
            raise ThroughputQualificationError(
                "preflight fleet/model artifact changed during parsing"
            )
    base_counts = {
        profile: len(base.by_profile[profile])
        for profile in sorted(SERVING_PROFILES)
    }
    effective_counts = {
        profile: len(effective.by_profile[profile])
        for profile in sorted(SERVING_PROFILES)
    }
    if (
        set(base.by_profile) != set(SERVING_PROFILES)
        or set(effective.by_profile) != set(SERVING_PROFILES)
        or base_counts != _profile_counts_from_fleet_payload(
            _read_json(
                base_path,
                description="parsed base fleet contract",
            )
        )
        or effective_counts != _profile_counts_from_fleet_payload(
            _read_json(
                effective_path,
                description="parsed effective fleet contract",
            )
        )
        or len(base.replicas) != sum(base_counts.values())
        or len(effective.replicas) != sum(effective_counts.values())
        or sum(replica.gpus_per_replica for replica in base.replicas) != 24
        or sum(
            replica.gpus_per_replica
            for replica in effective.replicas
        )
        < 24
    ):
        raise ThroughputQualificationError(
            "parsed preflight fleet topology/cardinality drifted"
        )
    return base, effective


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
        or authority.get("release_tag_object")
        != base.release_tag_object
        or authority.get("source_tree_sha256")
        != base.source_tree_sha256
        or authority.get("qualification_runner_source_sha256")
        != base.qualification_runner_source_sha256
        or not isinstance(immutable, Mapping)
        or immutable.get("git_commit") != base.release_git_commit
        or immutable.get("release_tag_object")
        != base.release_tag_object
        or immutable.get("source_tree_sha256")
        != base.source_tree_sha256
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


def _capacity_transition_path(
    base: QualificationContext,
    *,
    from_generation: int,
    to_generation: int,
) -> Path:
    if (
        from_generation < 1
        or to_generation != from_generation + 1
    ):
        raise ThroughputQualificationError(
            "capacity-transition generations must be one contiguous step"
        )
    return (
        base.qualification_base
        / CAPACITY_TRANSITION_DIRECTORY
        / (
            f"c{from_generation:06d}-to-"
            f"c{to_generation:06d}.json"
        )
    )


def _capacity_transition_pointer_payload(
    path: Path,
    transition: Mapping[str, Any],
) -> dict[str, Any]:
    additive = transition.get("additive_transition")
    failed_attempt = transition.get("failed_attempt")
    if not isinstance(additive, Mapping) or not isinstance(
        failed_attempt,
        Mapping,
    ):
        raise ThroughputQualificationError(
            "capacity-transition pointer source is malformed"
        )
    return _with_identity(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": CURRENT_CAPACITY_TRANSITION_PROTOCOL,
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "transition_id": transition["transition_id"],
            "from_capacity_generation": additive[
                "from_capacity_generation"
            ],
            "to_capacity_generation": additive[
                "to_capacity_generation"
            ],
            "failed_attempt_id": failed_attempt["attempt_id"],
        },
        "pointer_id",
    )


def _load_capacity_transition_journal(
    base: QualificationContext,
    *,
    recover_unpublished_head: bool = False,
) -> list[tuple[Path, dict[str, Any]]]:
    """Load a contiguous immutable transition journal and its marker-last head."""

    root = base.qualification_base / CAPACITY_TRANSITION_DIRECTORY
    pointer_path = (
        base.qualification_base / CURRENT_CAPACITY_TRANSITION_NAME
    )
    if not root.exists():
        if pointer_path.exists() or pointer_path.is_symlink():
            raise ThroughputQualificationError(
                "capacity-transition head exists without its journal"
            )
        return []
    if root.is_symlink() or not root.is_dir():
        raise ThroughputQualificationError(
            "capacity-transition journal root is unsafe"
        )
    records: list[tuple[Path, dict[str, Any]]] = []
    expected_from = 1
    for path in sorted(root.iterdir()):
        match = re.fullmatch(
            r"c([0-9]{6})-to-c([0-9]{6})[.]json",
            path.name,
        )
        if (
            match is None
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_nlink != 1
            or stat.S_IMODE(path.stat().st_mode) & 0o222
        ):
            raise ThroughputQualificationError(
                "capacity-transition journal contains an unsafe member"
            )
        from_generation = int(match.group(1))
        to_generation = int(match.group(2))
        record = _read_json(
            path,
            description="capacity-transition journal record",
            sealed=True,
        )
        _verify_identity(
            record,
            "transition_id",
            description="capacity-transition journal record",
        )
        additive = record.get("additive_transition")
        if (
            record.get("schema_version") != SCHEMA_VERSION
            or record.get("protocol") != CAPACITY_TRANSITION_PROTOCOL
            or record.get("passed") is not True
            or not isinstance(additive, Mapping)
            or additive.get("from_capacity_generation")
            != from_generation
            or additive.get("to_capacity_generation")
            != to_generation
            or from_generation != expected_from
            or to_generation != from_generation + 1
        ):
            raise ThroughputQualificationError(
                "capacity-transition journal skips, duplicates, or drifts "
                "from its generation identity"
            )
        records.append((path, record))
        expected_from = to_generation
    if not records:
        raise ThroughputQualificationError(
            "capacity-transition journal directory is empty"
        )
    if not pointer_path.exists() and not pointer_path.is_symlink():
        if recover_unpublished_head:
            return records
        raise ThroughputQualificationError(
            "capacity-transition journal lacks its marker-last head"
        )
    pointer = _read_json(
        pointer_path,
        description="current capacity-transition pointer",
        sealed=True,
    )
    if set(pointer) != _CURRENT_CAPACITY_TRANSITION_FIELDS:
        raise ThroughputQualificationError(
            "current capacity-transition pointer fields drifted"
        )
    _verify_identity(
        pointer,
        "pointer_id",
        description="current capacity-transition pointer",
    )
    matching_indexes = [
        index
        for index, (path, record) in enumerate(records)
        if pointer == _capacity_transition_pointer_payload(path, record)
    ]
    if (
        len(matching_indexes) != 1
        or (
            matching_indexes[0] != len(records) - 1
            and not (
                recover_unpublished_head
                and matching_indexes[0] == len(records) - 2
            )
        )
    ):
        raise ThroughputQualificationError(
            "current capacity-transition pointer does not bind the journal head"
        )
    return records


def _replace_current_capacity_transition(
    base: QualificationContext,
    *,
    path: Path,
    transition: Mapping[str, Any],
) -> None:
    destination = (
        base.qualification_base / CURRENT_CAPACITY_TRANSITION_NAME
    )
    payload = _capacity_transition_pointer_payload(path, transition)
    encoded = _canonical_bytes(payload)
    existing: dict[str, Any] | None = None
    if destination.exists() or destination.is_symlink():
        existing = _read_json(
            destination,
            description="current capacity-transition pointer",
            sealed=True,
        )
        if existing == payload:
            return
        if set(existing) != _CURRENT_CAPACITY_TRANSITION_FIELDS:
            raise ThroughputQualificationError(
                "current capacity-transition pointer fields drifted"
            )
        _verify_identity(
            existing,
            "pointer_id",
            description="current capacity-transition pointer",
        )
        if (
            existing.get("to_capacity_generation")
            != payload["from_capacity_generation"]
            or payload["to_capacity_generation"]
            != int(existing["to_capacity_generation"]) + 1
        ):
            raise ThroughputQualificationError(
                "capacity-transition pointer replacement would roll back or "
                "skip the immutable journal head"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stale = sorted(
        child
        for child in destination.parent.iterdir()
        if child.name.startswith(f".{destination.name}.")
        and child.name.endswith(".publishing")
    )
    if stale:
        if (
            len(stale) != 1
            or stale[0].is_symlink()
            or not stale[0].is_file()
            or stale[0].stat().st_nlink != 1
            or stat.S_IMODE(stale[0].stat().st_mode) & 0o222
            or stale[0].read_bytes() != encoded
        ):
            raise ThroughputQualificationError(
                "current capacity-transition pointer has an ambiguous or "
                "mismatched crash preimage"
            )
        descriptor = os.open(
            stale[0],
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(stale[0], destination)
        _fsync_directory(destination.parent)
        observed = _read_json(
            destination,
            description="recovered current capacity-transition pointer",
            sealed=True,
        )
        if observed != payload:
            raise ThroughputQualificationError(
                "recovered capacity-transition pointer bytes drifted"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".publishing",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


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
        effective_base = _with_effective_protected_capacity(
            base,
            control_value,
        )
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
        from_generation = int(
            additive["from_capacity_generation"]
        )
        to_generation = int(additive["to_capacity_generation"])
        destination = _capacity_transition_path(
            base,
            from_generation=from_generation,
            to_generation=to_generation,
        )
        if destination.exists() or destination.is_symlink():
            existing_transition = _read_json(
                destination,
                description="existing capacity-transition journal record",
                sealed=True,
            )
            _verify_identity(
                existing_transition,
                "transition_id",
                description="existing capacity-transition journal record",
            )
            existing_timestamp = existing_transition.get(
                "created_timestamp"
            )
            if (
                existing_transition.get("protocol")
                != CAPACITY_TRANSITION_PROTOCOL
                or not isinstance(existing_timestamp, (int, float))
                or isinstance(existing_timestamp, bool)
                or not math.isfinite(float(existing_timestamp))
                or float(existing_timestamp) <= 0
            ):
                raise ThroughputQualificationError(
                    "existing capacity-transition journal record is malformed"
                )
            timestamp = float(existing_timestamp)
        journal = _load_capacity_transition_journal(
            base,
            recover_unpublished_head=True,
        )
        if (
            (not journal and from_generation != 1)
            or (
                journal
                and int(
                    journal[-1][1]["additive_transition"][
                        "to_capacity_generation"
                    ]
                )
                not in {from_generation, to_generation}
            )
        ):
            raise ThroughputQualificationError(
                "capacity-transition publication skips or duplicates a "
                "generation"
            )
        raw_to_certificate = verify_authorized_preflight_capacity(
            effective_base,
            control_value=control_value,
            readiness_generation=readiness,
        )
        to_certificate = (
            _validated_admission_capacity_certificate_binding(
                effective_base,
                {
                    field: raw_to_certificate[field]
                    for field in _ADMISSION_CAPACITY_CERTIFICATE_FIELDS
                },
            )
        )
        from_certificate = dict(
            failure["admission_capacity_certificate"]
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
            "from_protected_capacity": dict(
                current.protected_capacity
            ),
            "to_protected_capacity": dict(
                effective_base.protected_capacity
            ),
            "from_admission_capacity_certificate": (
                from_certificate
            ),
            "to_admission_capacity_certificate": to_certificate,
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
            "from_readiness_generation": dict(
                previous["readiness_generation"]
            ),
            "to_readiness_generation": dict(readiness),
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
            _replace_current_capacity_transition(
                base,
                path=destination,
                transition=authority,
            )
            _load_capacity_transition_journal(base)
        return {
            "status": "published" if apply else "dry_run",
            "writes_performed": apply,
            "transition": authority,
            "transition_marker": str(destination),
            "transition_pointer": str(
                base.qualification_base
                / CURRENT_CAPACITY_TRANSITION_NAME
            ),
        }


def _require_capacity_transition_for_retry(
    base: QualificationContext,
    *,
    previous_path: Path,
    previous: Mapping[str, Any],
    control_value: Mapping[str, Any],
    readiness_generation: Mapping[str, Any],
    additive_retry: Mapping[str, Any],
) -> None:
    """Require the journal head to authorize exactly this one-generation retry."""

    journal = _load_capacity_transition_journal(base)
    if not journal:
        raise ThroughputQualificationError(
            "additive qualification retry lacks a capacity-transition journal"
        )
    transition_path, transition = journal[-1]
    previous_context = _attempt_context_from_pointer(
        base,
        path=previous_path,
        pointer=previous,
    )
    failure = _load_terminal_failure(previous_context)
    effective = _with_effective_protected_capacity(base, control_value)
    raw_to_certificate = verify_authorized_preflight_capacity(
        effective,
        control_value=control_value,
        readiness_generation=readiness_generation,
    )
    to_certificate = _validated_admission_capacity_certificate_binding(
        effective,
        {
            field: raw_to_certificate[field]
            for field in _ADMISSION_CAPACITY_CERTIFICATE_FIELDS
        },
    )
    expected_path = _capacity_transition_path(
        base,
        from_generation=int(
            additive_retry["from_capacity_generation"]
        ),
        to_generation=int(additive_retry["to_capacity_generation"]),
    )
    if (
        transition_path != expected_path
        or transition.get("additive_transition")
        != dict(additive_retry)
        or transition.get("failed_attempt")
        != _completion_attempt_binding(previous_context)
        or transition.get("failure", {}).get("failure_id")
        != failure["failure_id"]
        or transition.get("from_protected_capacity")
        != dict(previous_context.protected_capacity)
        or transition.get("to_protected_capacity")
        != dict(effective.protected_capacity)
        or transition.get("from_admission_capacity_certificate")
        != failure["admission_capacity_certificate"]
        or transition.get("to_admission_capacity_certificate")
        != to_certificate
        or transition.get("from_readiness_generation")
        != previous["readiness_generation"]
        or transition.get("to_readiness_generation")
        != dict(readiness_generation)
    ):
        raise ThroughputQualificationError(
            "capacity-transition journal head does not authorize the exact "
            "one-generation retry"
        )


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
            CAPACITY_TRANSITION_DIRECTORY,
            CURRENT_CAPACITY_TRANSITION_NAME,
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
        _require_capacity_transition_for_retry(
            base,
            previous_path=latest_path,
            previous=latest,
            control_value=control_value,
            readiness_generation=readiness_generation,
            additive_retry=additive_retry,
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
    admission_capacity_certificate: Mapping[str, Any],
    created_timestamp: float,
) -> dict[str, Any]:
    plan = build_load_plan()
    placement = authorize_client_placement(
        context,
        client_partition=client_partition,
        client_qos=client_qos,
    )
    return {
        "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
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
        "configured_client_ceiling": CEILINGS[-1],
        "certified_saturation_target": int(
            admission_capacity_certificate["selected_cell_count"]
        ),
        "admission_capacity_certificate": dict(
            admission_capacity_certificate
        ),
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
    "configured_client_ceiling",
    "certified_saturation_target",
    "admission_capacity_certificate",
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
    admission_certificate = value.get("admission_capacity_certificate")
    expected_admission_certificate: dict[str, Any] | None = None
    if isinstance(admission_certificate, Mapping):
        expected_admission_certificate = (
            _validated_admission_capacity_certificate_binding(
                context,
                admission_certificate,
            )
        )
    if (
        value.get("schema_version") != LOAD_ACCOUNTING_SCHEMA_VERSION
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
        or value.get("configured_client_ceiling") != CEILINGS[-1]
        or expected_admission_certificate is None
        or value.get("admission_capacity_certificate")
        != expected_admission_certificate
        or value.get("certified_saturation_target")
        != expected_admission_certificate["selected_cell_count"]
        or not 0
        < int(value.get("certified_saturation_target", 0))
        <= CEILINGS[-1]
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
    admission_capacity_certificate: Mapping[str, Any],
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
        if existing["admission_capacity_certificate"] != dict(
            admission_capacity_certificate
        ):
            raise ThroughputQualificationError(
                "signed saturation target changed after qualification intent "
                "publication"
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
            admission_capacity_certificate=(
                admission_capacity_certificate
            ),
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
    run_id: str = QUALIFICATION_RUN_ID,
) -> dict[str, Any]:
    immutable = control_value["immutable"]
    payload: dict[str, Any] = {
        "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
        "run_id": run_id,
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
    run_id: str = QUALIFICATION_RUN_ID,
    cycle_index: int = 0,
    cycle_id: str | None = None,
    semantic_reference: bool = True,
) -> dict[str, Any]:
    routes = Counter(
        serving_profile_for_cell(cell).name
        for cell in generate_qualification_cells()
    )
    return _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": LINEAGE_PROTOCOL,
            "run_id": run_id,
            "estimand_excluded": True,
            "primary_analysis_eligible": False,
            "cycle_index": cycle_index,
            "cycle_id": cycle_id,
            "semantic_reference": semantic_reference,
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
                "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
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
        "lineage_id": lineage["lineage_id"],
        "cell_count": CELL_COUNT,
        "qids": TOTAL_QIDS,
        "estimand_excluded": True,
    }


def _cycle_run_id(intent: Mapping[str, Any], cycle_index: int) -> str:
    if (
        not isinstance(cycle_index, int)
        or isinstance(cycle_index, bool)
        or cycle_index < 0
    ):
        raise ThroughputQualificationError("load cycle index must be nonnegative")
    if cycle_index == 0:
        return QUALIFICATION_RUN_ID
    return (
        f"{QUALIFICATION_RUN_ID}__{str(intent['intent_id'])[:12]}"
        f"__cycle_{cycle_index:06d}"
    )


def _cycle_paths(
    context: QualificationContext,
    intent: Mapping[str, Any],
    cycle_index: int,
) -> dict[str, Path | str | bool]:
    run_id = _cycle_run_id(intent, cycle_index)
    evidence_root = (
        context.qualification_root
        / LOAD_CYCLE_DIRECTORY
        / f"cycle-{cycle_index:06d}"
    )
    if cycle_index == 0:
        run_root = context.run_root
        dispatcher_state = context.dispatcher_state
        authority_path = context.qualification_root / EXECUTION_AUTHORITY_NAME
    else:
        run_root = (
            context.run_root.parent / LOAD_REPLAY_RUN_DIRECTORY / run_id
        )
        dispatcher_state = evidence_root / DISPATCHER_STATE_DIRECTORY
        authority_path = evidence_root / EXECUTION_AUTHORITY_NAME
    return {
        "run_id": run_id,
        "evidence_root": evidence_root,
        "run_root": run_root,
        "dispatcher_state": dispatcher_state,
        "execution_authority_path": authority_path,
        "intent_path": evidence_root / CYCLE_INTENT_NAME,
        "semantic_reference": cycle_index == 0,
    }


def _cycle_predecessor(
    context: QualificationContext,
    intent: Mapping[str, Any],
    cycle_index: int,
) -> dict[str, Any] | None:
    if cycle_index == 0:
        return None
    previous = load_cycle_context(
        context,
        intent=intent,
        cycle_index=cycle_index - 1,
        require_initialized=False,
    )
    return {
        "cycle_index": previous.cycle_index,
        "cycle_id": previous.cycle_id,
        "intent_path": str(previous.intent_path),
        "intent_sha256": _sha256_file(previous.intent_path),
    }


_CYCLE_INTENT_FIELDS = {
    "schema_version",
    "protocol",
    "qualification_intent_id",
    "attempt_id",
    "readiness_generation",
    "cycle_index",
    "run_id",
    "run_root",
    "dispatcher_state",
    "execution_authority",
    "semantic_reference",
    "predecessor",
    "plan_id",
    "estimand_excluded",
    "primary_analysis_eligible",
    "created_at",
    "created_timestamp",
    "cycle_id",
}


def _cycle_intent_identity(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    cycle_index: int,
    created_timestamp: float,
) -> dict[str, Any]:
    if context.attempt_pointer is None:
        raise ThroughputQualificationError(
            "load cycles require a generation-scoped qualification attempt"
        )
    paths = _cycle_paths(context, intent, cycle_index)
    return {
        "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
        "protocol": CYCLE_INTENT_PROTOCOL,
        "qualification_intent_id": intent["intent_id"],
        "attempt_id": context.attempt_pointer["attempt_id"],
        "readiness_generation": dict(intent["readiness_generation"]),
        "cycle_index": cycle_index,
        "run_id": paths["run_id"],
        "run_root": str(paths["run_root"]),
        "dispatcher_state": str(paths["dispatcher_state"]),
        "execution_authority": str(paths["execution_authority_path"]),
        "semantic_reference": bool(paths["semantic_reference"]),
        "predecessor": _cycle_predecessor(
            context, intent, cycle_index
        ),
        "plan_id": intent["plan_id"],
        "estimand_excluded": True,
        "primary_analysis_eligible": False,
        "created_at": _utc(created_timestamp),
        "created_timestamp": float(created_timestamp),
    }


def validate_cycle_intent(
    value: Mapping[str, Any],
    *,
    context: QualificationContext,
    intent: Mapping[str, Any],
    cycle_index: int,
) -> dict[str, Any]:
    if set(value) != _CYCLE_INTENT_FIELDS:
        raise ThroughputQualificationError(
            f"load cycle {cycle_index} intent fields drifted"
        )
    _verify_identity(
        value,
        "cycle_id",
        description=f"load cycle {cycle_index} intent",
    )
    timestamp = value.get("created_timestamp")
    paths = _cycle_paths(context, intent, cycle_index)
    expected_predecessor = _cycle_predecessor(
        context, intent, cycle_index
    )
    if (
        value.get("schema_version") != LOAD_ACCOUNTING_SCHEMA_VERSION
        or value.get("protocol") != CYCLE_INTENT_PROTOCOL
        or value.get("qualification_intent_id") != intent["intent_id"]
        or context.attempt_pointer is None
        or value.get("attempt_id")
        != context.attempt_pointer.get("attempt_id")
        or value.get("readiness_generation")
        != intent["readiness_generation"]
        or value.get("cycle_index") != cycle_index
        or value.get("run_id") != paths["run_id"]
        or value.get("run_root") != str(paths["run_root"])
        or value.get("dispatcher_state")
        != str(paths["dispatcher_state"])
        or value.get("execution_authority")
        != str(paths["execution_authority_path"])
        or value.get("semantic_reference")
        is not bool(paths["semantic_reference"])
        or value.get("predecessor") != expected_predecessor
        or value.get("plan_id") != intent["plan_id"]
        or value.get("estimand_excluded") is not True
        or value.get("primary_analysis_eligible") is not False
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or float(timestamp) <= 0
        or value.get("created_at") != _utc(float(timestamp))
    ):
        raise ThroughputQualificationError(
            f"load cycle {cycle_index} intent is invalid"
        )
    return dict(value)


def create_or_load_cycle_intent(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    cycle_index: int,
    now: float,
) -> LoadCycleContext:
    """Publish one marker-first cycle intent, adopting exact replay safely."""

    paths = _cycle_paths(context, intent, cycle_index)
    intent_path = Path(paths["intent_path"])
    if intent_path.exists() or intent_path.is_symlink():
        value = validate_cycle_intent(
            _read_json(
                intent_path,
                description=f"load cycle {cycle_index} intent",
                sealed=True,
            ),
            context=context,
            intent=intent,
            cycle_index=cycle_index,
        )
    else:
        if cycle_index > 0:
            load_cycle_context(
                context,
                intent=intent,
                cycle_index=cycle_index - 1,
                require_initialized=True,
            )
        intent_path.parent.mkdir(parents=True, exist_ok=True)
        value = _with_identity(
            _cycle_intent_identity(
                context,
                intent=intent,
                cycle_index=cycle_index,
                created_timestamp=now,
            ),
            "cycle_id",
        )
        _write_once(
            intent_path,
            value,
            description=f"load cycle {cycle_index} marker-first intent",
        )
        value = validate_cycle_intent(
            _read_json(
                intent_path,
                description=f"load cycle {cycle_index} intent",
                sealed=True,
            ),
            context=context,
            intent=intent,
            cycle_index=cycle_index,
        )
    return LoadCycleContext(
        cycle_index=cycle_index,
        cycle_id=str(value["cycle_id"]),
        run_id=str(value["run_id"]),
        evidence_root=Path(str(paths["evidence_root"])),
        run_root=Path(str(value["run_root"])),
        dispatcher_state=Path(str(value["dispatcher_state"])),
        execution_authority_path=Path(str(value["execution_authority"])),
        intent_path=intent_path,
        intent=value,
        semantic_reference=bool(value["semantic_reference"]),
    )


def load_cycle_context(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    cycle_index: int,
    require_initialized: bool = True,
) -> LoadCycleContext:
    paths = _cycle_paths(context, intent, cycle_index)
    intent_path = Path(paths["intent_path"])
    if not intent_path.exists() and not intent_path.is_symlink():
        raise ThroughputQualificationError(
            f"load cycle {cycle_index} intent is missing"
        )
    cycle = create_or_load_cycle_intent(
        context,
        intent=intent,
        cycle_index=cycle_index,
        now=1.0,  # Existing sealed intent makes this value observationally inert.
    )
    if require_initialized:
        marker_path = cycle.evidence_root / CYCLE_INITIALIZED_NAME
        if not marker_path.is_file() or marker_path.is_symlink():
            raise ThroughputQualificationError(
                f"load cycle {cycle_index} is not initialized"
            )
        marker = _read_json(
            marker_path,
            description=f"load cycle {cycle_index} initialization",
            sealed=True,
        )
        _verify_identity(
            marker,
            "initialization_id",
            description=f"load cycle {cycle_index} initialization",
        )
        if (
            marker.get("schema_version") != LOAD_ACCOUNTING_SCHEMA_VERSION
            or marker.get("protocol") != CYCLE_INITIALIZED_PROTOCOL
            or marker.get("cycle_id") != cycle.cycle_id
            or marker.get("run_id") != cycle.run_id
            or marker.get("run_root") != str(cycle.run_root)
            or marker.get("estimand_excluded") is not True
            or marker.get("primary_analysis_eligible") is not False
        ):
            raise ThroughputQualificationError(
                f"load cycle {cycle_index} initialization is invalid"
            )
    return cycle


def load_cycle_inventory(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    require_initialized: bool = True,
) -> list[LoadCycleContext]:
    directory = context.qualification_root / LOAD_CYCLE_DIRECTORY
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ThroughputQualificationError("load cycle directory is unsafe")
    indexes: list[int] = []
    pattern = re.compile(r"cycle-([0-9]{6})\Z")
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            raise ThroughputQualificationError(
                f"unrecognized load cycle artifact: {path}"
            )
        indexes.append(int(match.group(1)))
    indexes.sort()
    if indexes != list(range(len(indexes))):
        raise ThroughputQualificationError(
            "load cycle indexes are not contiguous from zero"
        )
    return [
        load_cycle_context(
            context,
            intent=intent,
            cycle_index=index,
            require_initialized=require_initialized,
        )
        for index in indexes
    ]


def _copy_sealed_once(
    source: Path,
    destination: Path,
    *,
    description: str,
) -> None:
    """Copy exact bytes to a distinct sealed inode and adopt exact replay."""

    if source.is_symlink() or not source.is_file():
        raise ThroughputQualificationError(
            f"{description} source is missing or unsafe: {source}"
        )
    _write_once(
        destination,
        source.read_bytes(),
        description=description,
    )
    if os.stat(source).st_ino == os.stat(destination).st_ino:
        raise ThroughputQualificationError(
            f"{description} unexpectedly shares an inode with its source"
        )


def _verify_replay_cycle_run(
    cycle: LoadCycleContext,
    *,
    reference_root: Path,
    control_value: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = load_manifest(cycle.run_root, verify_frozen=True)
    reference = load_manifest(reference_root, verify_frozen=True)
    if (
        snapshot.cells != reference.cells
        or snapshot.sha256 != reference.sha256
        or snapshot.cells != generate_qualification_cells()
    ):
        raise ThroughputQualificationError(
            f"load cycle {cycle.cycle_index} manifest drifted"
        )
    catalog = VerifiedQuestionCatalog(cycle.run_root, snapshot=snapshot)
    reference_catalog = VerifiedQuestionCatalog(
        reference_root, snapshot=reference
    )
    policy = load_artifact_policy(cycle.run_root, required=True)
    assert policy is not None
    immutable = control_value["immutable"]
    if (
        catalog.sidecar_sha256 != reference_catalog.sidecar_sha256
        or policy.run_id != cycle.run_id
        or policy.accepted_manifest_sha256 != snapshot.sha256
        or policy.accepted_benchmark_contracts_sha256
        != catalog.sidecar_sha256
        or policy.accepted_model_contract_sha256
        != immutable["model_contract_sha256"]
        or policy.legacy_result_import_allowed is not False
    ):
        raise ThroughputQualificationError(
            f"load cycle {cycle.cycle_index} provenance drifted"
        )
    lineage = _read_json(
        cycle.run_root / LINEAGE_NAME,
        description=f"load cycle {cycle.cycle_index} lineage",
        sealed=True,
    )
    _verify_identity(
        lineage,
        "lineage_id",
        description=f"load cycle {cycle.cycle_index} lineage",
    )
    if (
        lineage.get("protocol") != LINEAGE_PROTOCOL
        or lineage.get("run_id") != cycle.run_id
        or lineage.get("cycle_index") != cycle.cycle_index
        or lineage.get("cycle_id") != cycle.cycle_id
        or lineage.get("semantic_reference") is not cycle.semantic_reference
        or lineage.get("estimand_excluded") is not True
        or lineage.get("primary_analysis_eligible") is not False
    ):
        raise ThroughputQualificationError(
            f"load cycle {cycle.cycle_index} analysis isolation drifted"
        )
    return {
        "manifest_sha256": snapshot.sha256,
        "benchmark_contracts_sha256": catalog.sidecar_sha256,
        "artifact_policy_sha256": policy.file_sha256,
        "lineage_id": lineage["lineage_id"],
    }


def initialize_load_cycle(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    control_value: Mapping[str, Any],
    cycle_index: int,
    now: float,
) -> LoadCycleContext:
    """Initialize or resume one byte-identical, independently owned cycle."""

    cycle = create_or_load_cycle_intent(
        context,
        intent=intent,
        cycle_index=cycle_index,
        now=now,
    )
    evidence_marker = cycle.evidence_root / CYCLE_INITIALIZED_NAME
    if evidence_marker.exists() or evidence_marker.is_symlink():
        load_cycle_context(
            context,
            intent=intent,
            cycle_index=cycle_index,
            require_initialized=True,
        )
        if cycle_index > 0:
            _verify_replay_cycle_run(
                cycle,
                reference_root=context.run_root,
                control_value=control_value,
            )
        return cycle

    if cycle_index == 0:
        run_evidence = verify_qualification_run(
            context, intent=intent, control_value=control_value
        )
    else:
        cycle.run_root.parent.mkdir(parents=True, exist_ok=True)
        stage = cycle.run_root.parent / (
            f".{cycle.run_id}.initialization-v3"
        )
        if cycle.run_root.exists() or cycle.run_root.is_symlink():
            run_evidence = _verify_replay_cycle_run(
                cycle,
                reference_root=context.run_root,
                control_value=control_value,
            )
        else:
            if stage.is_symlink():
                raise ThroughputQualificationError(
                    f"load cycle {cycle_index} staging path is unsafe"
                )
            stage.mkdir(parents=True, exist_ok=True)
            for name in (
                "cells.json",
                "cells.sha256",
                "benchmark_contracts.v1.json",
                "benchmark_contracts.v1.sha256",
            ):
                _copy_sealed_once(
                    context.run_root / name,
                    stage / name,
                    description=f"load cycle {cycle_index} {name}",
                )
            snapshot = load_manifest(stage, verify_frozen=True)
            catalog = VerifiedQuestionCatalog(stage, snapshot=snapshot)
            policy = _policy_payload(
                control_value=control_value,
                manifest_sha256=snapshot.sha256,
                benchmark_sha256=catalog.sidecar_sha256,
                run_id=cycle.run_id,
            )
            policy_bytes = _canonical_bytes(policy)
            _write_once(
                stage / POLICY_FILENAME,
                policy_bytes,
                description=f"load cycle {cycle_index} artifact policy",
            )
            _write_once(
                stage / POLICY_CHECKSUM_FILENAME,
                f"{_sha256_bytes(policy_bytes)}  {POLICY_FILENAME}\n".encode(),
                description=f"load cycle {cycle_index} policy checksum",
            )
            lineage = _qualification_lineage(
                plan=build_load_plan(),
                manifest_sha256=snapshot.sha256,
                benchmark_sha256=catalog.sidecar_sha256,
                run_id=cycle.run_id,
                cycle_index=cycle_index,
                cycle_id=cycle.cycle_id,
                semantic_reference=False,
            )
            _write_once(
                stage / LINEAGE_NAME,
                lineage,
                description=f"load cycle {cycle_index} lineage",
            )
            (stage / "cells").mkdir(exist_ok=True)
            run_marker = _with_identity(
                {
                    "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
                    "protocol": CYCLE_INITIALIZED_PROTOCOL,
                    "cycle_id": cycle.cycle_id,
                    "run_id": cycle.run_id,
                    "run_root": str(cycle.run_root),
                    "manifest_sha256": snapshot.sha256,
                    "benchmark_contracts_sha256": catalog.sidecar_sha256,
                    "artifact_policy_sha256": _sha256_bytes(policy_bytes),
                    "lineage_id": lineage["lineage_id"],
                    "estimand_excluded": True,
                    "primary_analysis_eligible": False,
                },
                "initialization_id",
            )
            _write_once(
                stage / CYCLE_INITIALIZED_NAME,
                run_marker,
                description=f"load cycle {cycle_index} run initialization",
            )
            staged_cycle = replace(cycle, run_root=stage)
            _verify_replay_cycle_run(
                staged_cycle,
                reference_root=context.run_root,
                control_value=control_value,
            )
            os.replace(stage, cycle.run_root)
            _fsync_directory(cycle.run_root.parent)
            run_evidence = _verify_replay_cycle_run(
                cycle,
                reference_root=context.run_root,
                control_value=control_value,
            )
    marker = _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": CYCLE_INITIALIZED_PROTOCOL,
            "cycle_id": cycle.cycle_id,
            "run_id": cycle.run_id,
            "run_root": str(cycle.run_root),
            "manifest_sha256": run_evidence["manifest_sha256"],
            "benchmark_contracts_sha256": run_evidence[
                "benchmark_contracts_sha256"
            ],
            "artifact_policy_sha256": run_evidence[
                "artifact_policy_sha256"
            ],
            "lineage_id": run_evidence["lineage_id"],
            "estimand_excluded": True,
            "primary_analysis_eligible": False,
        },
        "initialization_id",
    )
    _write_once(
        evidence_marker,
        marker,
        description=f"load cycle {cycle_index} initialization receipt",
    )
    return load_cycle_context(
        context,
        intent=intent,
        cycle_index=cycle_index,
        require_initialized=True,
    )


_LOAD_EVENT_FIELDS = {
    "schema_version",
    "protocol",
    "qualification_intent_id",
    "attempt_id",
    "cycle_id",
    "cycle_index",
    "run_id",
    "cell_id",
    "qid",
    "stratum",
    "artifact_schema_version",
    "result_record_sha256",
    "trusted",
    "length_censors",
    "protocol_censors",
    "transport_censors",
    "event_id",
}


def _load_event_path(
    cycle: LoadCycleContext,
    *,
    cell_id: str,
    qid: str,
) -> Path:
    cell_component = hashlib.sha256(cell_id.encode("utf-8")).hexdigest()
    qid_component = hashlib.sha256(qid.encode("utf-8")).hexdigest()
    return (
        cycle.evidence_root
        / CYCLE_EVENTS_DIRECTORY
        / cell_component
        / f"{qid_component}.json"
    )


def validate_load_execution_event(
    value: Mapping[str, Any],
    *,
    context: QualificationContext,
    intent: Mapping[str, Any],
    cycle: LoadCycleContext,
) -> dict[str, Any]:
    if set(value) != _LOAD_EVENT_FIELDS:
        raise ThroughputQualificationError(
            f"load cycle {cycle.cycle_index} event fields drifted"
        )
    _verify_identity(
        value,
        "event_id",
        description=f"load cycle {cycle.cycle_index} event",
    )
    if (
        value.get("schema_version") != LOAD_ACCOUNTING_SCHEMA_VERSION
        or value.get("protocol") != LOAD_EVENT_PROTOCOL
        or value.get("qualification_intent_id") != intent["intent_id"]
        or context.attempt_pointer is None
        or value.get("attempt_id")
        != context.attempt_pointer.get("attempt_id")
        or value.get("cycle_id") != cycle.cycle_id
        or value.get("cycle_index") != cycle.cycle_index
        or value.get("run_id") != cycle.run_id
        or not isinstance(value.get("cell_id"), str)
        or not value["cell_id"]
        or not isinstance(value.get("qid"), str)
        or not value["qid"]
        or value.get("stratum") not in expected_stratum_labels()
        or value.get("artifact_schema_version") != 5
        or _SHA256_RE.fullmatch(
            str(value.get("result_record_sha256", ""))
        )
        is None
        or value.get("trusted") is not True
        or value.get("length_censors") != 0
        or value.get("protocol_censors") != 0
        or value.get("transport_censors") != 0
    ):
        raise ThroughputQualificationError(
            f"load cycle {cycle.cycle_index} event is untrusted"
        )
    return dict(value)


def record_load_execution_event(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    cycle: LoadCycleContext,
    cell: ExperimentCell,
    qid: str,
    result_record_sha256: str,
) -> dict[str, Any]:
    """Journal one trusted coordinate exactly once within its cycle.

    The deterministic path is the transaction key.  Repeating the coordinate in
    another cycle is valid because that cycle owns a different evidence root.
    Conflicting redraws within one cycle collide with the same path and fail closed.
    """

    if cell.cell_id not in {item.cell_id for item in generate_qualification_cells()}:
        raise ThroughputQualificationError(
            "load event cell is outside the immutable qualification design"
        )
    payload = _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": LOAD_EVENT_PROTOCOL,
            "qualification_intent_id": intent["intent_id"],
            "attempt_id": (
                context.attempt_pointer["attempt_id"]
                if context.attempt_pointer is not None
                else None
            ),
            "cycle_id": cycle.cycle_id,
            "cycle_index": cycle.cycle_index,
            "run_id": cycle.run_id,
            "cell_id": cell.cell_id,
            "qid": str(qid),
            "stratum": _stratum_label(_stratum_tuple(cell)),
            "artifact_schema_version": 5,
            "result_record_sha256": result_record_sha256,
            "trusted": True,
            "length_censors": 0,
            "protocol_censors": 0,
            "transport_censors": 0,
        },
        "event_id",
    )
    validate_load_execution_event(
        payload,
        context=context,
        intent=intent,
        cycle=cycle,
    )
    path = _load_event_path(
        cycle, cell_id=cell.cell_id, qid=str(qid)
    )
    _write_once(
        path,
        payload,
        description=(
            f"load cycle {cycle.cycle_index} trusted event "
            f"{cell.cell_id}/{qid}"
        ),
    )
    return validate_load_execution_event(
        _read_json(
            path,
            description=f"load cycle {cycle.cycle_index} trusted event",
            sealed=True,
        ),
        context=context,
        intent=intent,
        cycle=cycle,
    )


def load_cycle_execution_events(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    cycle: LoadCycleContext,
) -> list[dict[str, Any]]:
    root = cycle.evidence_root / CYCLE_EVENTS_DIRECTORY
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise ThroughputQualificationError(
            f"load cycle {cycle.cycle_index} event root is unsafe"
        )
    events: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} event path is unsafe"
            )
        event = validate_load_execution_event(
            _read_json(
                path,
                description=f"load cycle {cycle.cycle_index} trusted event",
                sealed=True,
            ),
            context=context,
            intent=intent,
            cycle=cycle,
        )
        expected_path = _load_event_path(
            cycle,
            cell_id=str(event["cell_id"]),
            qid=str(event["qid"]),
        )
        key = (str(event["cell_id"]), str(event["qid"]))
        if path != expected_path or key in keys:
            raise ThroughputQualificationError(
                f"duplicate or misaddressed coordinate in load cycle "
                f"{cycle.cycle_index}: {key}"
            )
        keys.add(key)
        events.append(event)
    unexpected = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix != ".json"
    ]
    if unexpected:
        raise ThroughputQualificationError(
            f"load cycle {cycle.cycle_index} contains unrecognized events"
        )
    return events


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
    "unfinished_load_assignments",
    "active_cycle_ids",
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
    "cycle_execution_authorities",
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
    "semantic_reference_cycle",
    "trusted_qid_execution_events",
    "replay_qid_execution_events",
    "load_strata_progress",
    "load_cycle_inventory",
    "unfinished_load_assignments",
    "load_integrity_incidents",
    "load_censor_incidents",
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
_LOAD_CYCLE_INVENTORY_FIELDS = {
    "cycle_index",
    "cycle_id",
    "run_id",
    "semantic_reference",
    "status",
    "validated_execution_events",
    "unfinished_assignments",
    "estimand_excluded",
    "primary_analysis_eligible",
}
_LOAD_CYCLE_STATUSES = {
    "initialized",
    "active",
    "complete",
    "load_window_drained",
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
        authority.get("schema_version")
        != EXECUTION_AUTHORITY_SCHEMA_VERSION
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
    unfinished_load_assignments: int = CELL_COUNT,
    active_cycle_ids: Sequence[str] = (),
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
    cycle_execution_authorities: Sequence[Mapping[str, str]] | None = None,
    squeue_complete: bool = True,
    sacct_complete: bool = True,
    errors: Sequence[str] = (),
    qualification_tasks_only: bool = True,
    production_run_ids: Sequence[str] = (),
) -> dict[str, Any]:
    return _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
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
            "unfinished_load_assignments": unfinished_load_assignments,
            "active_cycle_ids": list(active_cycle_ids),
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
            "cycle_execution_authorities": [
                dict(record)
                for record in (
                    cycle_execution_authorities
                    if cycle_execution_authorities is not None
                    else [
                        {
                            "cycle_id": "0" * 64,
                            "run_id": QUALIFICATION_RUN_ID,
                            "authority_id": (
                                qualification_execution_authority_id
                            ),
                            "authority_sha256": (
                                qualification_execution_authority_sha256
                            ),
                        }
                    ]
                )
            ],
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
    semantic_reference_cycle: str = "0" * 64,
    trusted_qid_execution_events: int | None = None,
    replay_qid_execution_events: int = 0,
    load_strata_progress: Mapping[str, int] | None = None,
    load_cycle_inventory: Sequence[Mapping[str, Any]] | None = None,
    unfinished_load_assignments: int | None = None,
    load_integrity_incidents: int = 0,
    load_censor_incidents: int = 0,
) -> dict[str, Any]:
    progress = dict(sorted((str(key), int(value)) for key, value in strata_progress.items()))
    execution_events = (
        useful_qids
        if trusted_qid_execution_events is None
        else trusted_qid_execution_events
    )
    load_progress = dict(
        sorted(
            (str(key), int(value))
            for key, value in (
                strata_progress
                if load_strata_progress is None
                else load_strata_progress
            ).items()
        )
    )
    if unfinished_load_assignments is None:
        complete_cells = int(states.get("complete", 0))
        unfinished_load_assignments = max(0, CELL_COUNT - complete_cells)
    if load_cycle_inventory is None:
        load_cycle_inventory = [
            {
                "cycle_index": 0,
                "cycle_id": semantic_reference_cycle,
                "run_id": QUALIFICATION_RUN_ID,
                "semantic_reference": True,
                "status": (
                    "complete"
                    if states == {"complete": CELL_COUNT}
                    else "active"
                ),
                "validated_execution_events": execution_events,
                "unfinished_assignments": unfinished_load_assignments,
                "estimand_excluded": True,
                "primary_analysis_eligible": False,
            }
        ]
    return _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
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
            "semantic_reference_cycle": semantic_reference_cycle,
            "trusted_qid_execution_events": execution_events,
            "replay_qid_execution_events": replay_qid_execution_events,
            "load_strata_progress": load_progress,
            "load_cycle_inventory": [
                dict(record) for record in load_cycle_inventory
            ],
            "unfinished_load_assignments": unfinished_load_assignments,
            "load_integrity_incidents": load_integrity_incidents,
            "load_censor_incidents": load_censor_incidents,
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
        value.get("schema_version") != LOAD_ACCOUNTING_SCHEMA_VERSION
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
        or not isinstance(value.get("unfinished_load_assignments"), int)
        or isinstance(value.get("unfinished_load_assignments"), bool)
        or value["unfinished_load_assignments"] < 0
        or not isinstance(value.get("active_cycle_ids"), list)
        or not all(
            isinstance(cycle_id, str)
            and _SHA256_RE.fullmatch(cycle_id) is not None
            for cycle_id in value["active_cycle_ids"]
        )
        or len(value["active_cycle_ids"])
        != len(set(value["active_cycle_ids"]))
        or (
            value["active_qualification_cells"] > 0
            and not value["active_cycle_ids"]
        )
        or (
            value["active_qualification_cells"] == 0
            and value["active_cycle_ids"]
        )
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
        or not isinstance(
            value.get("cycle_execution_authorities"), list
        )
        or not value["cycle_execution_authorities"]
        or any(
            not isinstance(record, Mapping)
            or set(record)
            != {
                "cycle_id",
                "run_id",
                "authority_id",
                "authority_sha256",
            }
            or _SHA256_RE.fullmatch(
                str(record.get("cycle_id", ""))
            )
            is None
            or not isinstance(record.get("run_id"), str)
            or not record["run_id"]
            or _SHA256_RE.fullmatch(
                str(record.get("authority_id", ""))
            )
            is None
            or _SHA256_RE.fullmatch(
                str(record.get("authority_sha256", ""))
            )
            is None
            for record in value["cycle_execution_authorities"]
        )
        or len(
            {
                record["cycle_id"]
                for record in value["cycle_execution_authorities"]
            }
        )
        != len(value["cycle_execution_authorities"])
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
    load_progress = value.get("load_strata_progress")
    cycle_inventory = value.get("load_cycle_inventory")
    schemas = value.get("artifact_schema_counts")
    integer_fields = (
        "validated_qids",
        "useful_qids",
        "integrity_incidents",
        "transport_censor_incidents",
    )
    load_integer_fields = (
        "trusted_qid_execution_events",
        "replay_qid_execution_events",
        "unfinished_load_assignments",
        "load_integrity_incidents",
        "load_censor_incidents",
    )
    if (
        value.get("schema_version") != LOAD_ACCOUNTING_SCHEMA_VERSION
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
        or _SHA256_RE.fullmatch(
            str(value.get("semantic_reference_cycle", ""))
        )
        is None
        or any(
            not isinstance(value.get(field), int)
            or isinstance(value.get(field), bool)
            or value[field] < 0
            for field in load_integer_fields
        )
        or value["replay_qid_execution_events"]
        > value["trusted_qid_execution_events"]
        or not isinstance(load_progress, Mapping)
        or set(load_progress) != set(expected_stratum_labels())
        or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for count in load_progress.values()
        )
        or sum(load_progress.values())
        != value["trusted_qid_execution_events"]
        or not isinstance(cycle_inventory, list)
        or not cycle_inventory
        or any(
            not isinstance(record, Mapping)
            or set(record) != _LOAD_CYCLE_INVENTORY_FIELDS
            or not isinstance(record.get("cycle_index"), int)
            or isinstance(record.get("cycle_index"), bool)
            or record["cycle_index"] < 0
            or _SHA256_RE.fullmatch(
                str(record.get("cycle_id", ""))
            )
            is None
            or not isinstance(record.get("run_id"), str)
            or not record["run_id"]
            or not isinstance(record.get("semantic_reference"), bool)
            or record.get("status") not in _LOAD_CYCLE_STATUSES
            or not isinstance(
                record.get("validated_execution_events"), int
            )
            or isinstance(
                record.get("validated_execution_events"), bool
            )
            or record["validated_execution_events"] < 0
            or not isinstance(record.get("unfinished_assignments"), int)
            or isinstance(record.get("unfinished_assignments"), bool)
            or record["unfinished_assignments"] < 0
            or record.get("estimand_excluded") is not True
            or record.get("primary_analysis_eligible") is not False
            for record in cycle_inventory
        )
        or [record["cycle_index"] for record in cycle_inventory]
        != list(range(len(cycle_inventory)))
        or len({record["cycle_id"] for record in cycle_inventory})
        != len(cycle_inventory)
        or len(
            [
                record
                for record in cycle_inventory
                if record["semantic_reference"] is True
            ]
        )
        != 1
        or cycle_inventory[0]["semantic_reference"] is not True
        or cycle_inventory[0]["cycle_id"]
        != value["semantic_reference_cycle"]
        or sum(
            record["validated_execution_events"]
            for record in cycle_inventory
        )
        != value["trusted_qid_execution_events"]
        or sum(
            record["validated_execution_events"]
            for record in cycle_inventory
            if not record["semantic_reference"]
        )
        != value["replay_qid_execution_events"]
        or sum(
            record["unfinished_assignments"]
            for record in cycle_inventory
        )
        != value["unfinished_load_assignments"]
    ):
        raise ThroughputQualificationError(
            "semantic evidence violates the 768-cell/15,360-QID contract"
        )
    return dict(value)


_REFILL_RECONCILIATION_FIELDS = {
    "schema_version",
    "protocol",
    "intent_id",
    "refill_index",
    "measurement_sequence",
    "captured_timestamp",
    "scheduler",
    "semantic",
    "preceding_dispatch",
    "unresolved_intent_reservations",
    "prior_refill_id",
    "active_deficit",
    "measurement_eligible",
    "refill_id",
}
_REFILL_DISPATCH_FIELDS = {
    "cycle_id",
    "run_id",
    "ledger_path",
    "requested_tasks",
    "selected_tasks",
    "job_id",
    "batch_id",
    "accepted_tasks",
    "intent_state",
    "job_task_count",
    "tasks_sha256",
    "batch_manifest",
    "batch_manifest_sha256",
    "sbatch_path",
    "sbatch_sha256",
    "spooled_sbatch_sha256",
    "spooled_receipt_path",
    "spooled_receipt_sha256",
    "submission_transport",
    "submission_argv_sha256",
}
_REFILL_UNRESOLVED_RESERVATION_FIELDS = {
    "cycle_id",
    "run_id",
    "ledger_path",
    "batch_id",
    "intent_state",
    "task_count",
    "tasks_sha256",
}


def _refill_dispatch_from_ledger(
    *,
    cycle: LoadCycleContext,
    ledger: Mapping[str, Any],
    job_id: str,
    requested_tasks: int,
) -> dict[str, Any]:
    """Bind one accepted admission to its final durable transaction records."""

    jobs = ledger.get("jobs")
    intents = ledger.get("intents")
    job = jobs.get(job_id) if isinstance(jobs, Mapping) else None
    batch_id = str(job.get("batch_id", "")) if isinstance(job, Mapping) else ""
    intent_record = (
        intents.get(batch_id) if isinstance(intents, Mapping) else None
    )
    tasks = job.get("tasks") if isinstance(job, Mapping) else None
    if (
        not isinstance(requested_tasks, int)
        or isinstance(requested_tasks, bool)
        or not 1 <= requested_tasks <= MAX_BATCH
        or not isinstance(job, Mapping)
        or not isinstance(intent_record, Mapping)
        or intent_record.get("state") not in {"submitted", "reconciled"}
        or str(intent_record.get("job_id", "")) != job_id
        or intent_record.get("fairness_committed") is not True
        or not isinstance(tasks, list)
        or len(tasks) != requested_tasks
        or intent_record.get("tasks") != tasks
        or job.get("task_count") != requested_tasks
        or any(
            not isinstance(task, Mapping)
            or task.get("run_id") != cycle.run_id
            for task in tasks
        )
    ):
        raise QualificationWindowContinuityError(
            "refill admission is not a closed accepted intent/job/task "
            "transaction"
        )
    artifact_fields = (
        "batch_manifest",
        "batch_manifest_sha256",
        "sbatch_path",
        "sbatch_sha256",
        "spooled_sbatch_sha256",
        "spooled_receipt_path",
        "spooled_receipt_sha256",
        "submission_transport",
        "submission_argv_sha256",
    )
    if any(job.get(field) != intent_record.get(field) for field in artifact_fields):
        raise QualificationWindowContinuityError(
            "refill admission job and intent artifact provenance differ"
        )
    for path_field in (
        "batch_manifest",
        "sbatch_path",
        "spooled_receipt_path",
    ):
        raw_path = job.get(path_field)
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise QualificationWindowContinuityError(
                f"refill admission {path_field} is not an absolute path"
            )
    for hash_field in (
        "batch_manifest_sha256",
        "sbatch_sha256",
        "spooled_sbatch_sha256",
        "spooled_receipt_sha256",
        "submission_argv_sha256",
    ):
        if _SHA256_RE.fullmatch(str(job.get(hash_field, ""))) is None:
            raise QualificationWindowContinuityError(
                f"refill admission {hash_field} is malformed"
            )
    return {
        "cycle_id": cycle.cycle_id,
        "run_id": cycle.run_id,
        "ledger_path": str(
            (cycle.dispatcher_state / "ledger.json").resolve()
        ),
        "requested_tasks": requested_tasks,
        "selected_tasks": requested_tasks,
        "job_id": job_id,
        "batch_id": batch_id,
        "accepted_tasks": requested_tasks,
        # The mutable dispatcher state may advance submitted -> reconciled.  The
        # journal binds the stable semantic state and verify-only reopens the
        # final sealed ledger to prove it remains accepted.
        "intent_state": "accepted",
        "job_task_count": requested_tasks,
        "tasks_sha256": _sha256_bytes(_canonical_bytes(tasks)),
        **{field: str(job[field]) for field in artifact_fields},
    }


def _refill_dispatch_binding(
    report: Mapping[str, Any],
    *,
    requested_tasks: int,
    cycle: LoadCycleContext,
) -> dict[str, Any]:
    selected = report.get("selected")
    submission = report.get("submission")
    if (
        not isinstance(requested_tasks, int)
        or isinstance(requested_tasks, bool)
        or not 1 <= requested_tasks <= MAX_BATCH
        or not isinstance(selected, list)
        or len(selected) != requested_tasks
        or not isinstance(submission, Mapping)
        or set(submission) != {"job_id", "batch_id", "tasks"}
        or not str(submission.get("job_id", "")).isdigit()
        or _SHA256_RE.fullmatch(
            str(submission.get("batch_id", ""))
        )
        is None
        or submission.get("tasks") != len(selected)
    ):
        raise QualificationWindowContinuityError(
            "refill dispatch lacks one exact accepted <=24-task transaction"
        )
    job_id = str(submission["job_id"])
    batch_id = str(submission["batch_id"])
    ledger_path = cycle.dispatcher_state / "ledger.json"
    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise QualificationWindowContinuityError(
            "refill dispatch accepted a job without its durable ledger"
        )
    try:
        ledger = dispatch_sweeps._load_ledger(ledger_path)  # noqa: SLF001
    except (dispatch_sweeps.DispatcherError, OSError) as exc:
        raise QualificationWindowContinuityError(
            f"refill dispatch ledger cannot be verified: {exc}"
        ) from exc
    binding = _refill_dispatch_from_ledger(
        cycle=cycle,
        ledger=ledger,
        job_id=job_id,
        requested_tasks=requested_tasks,
    )
    if binding["batch_id"] != batch_id:
        raise QualificationWindowContinuityError(
            "refill dispatch report is not bijective with its exact durable "
            "intent/job/task record"
        )
    return binding


def _validate_refill_reconciliation(
    value: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    refill_index: int,
    prior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if set(value) != _REFILL_RECONCILIATION_FIELDS:
        raise ThroughputQualificationError(
            f"refill reconciliation {refill_index} fields drifted"
        )
    _verify_identity(
        value,
        "refill_id",
        description=f"refill reconciliation {refill_index}",
    )
    scheduler = value.get("scheduler")
    semantic = value.get("semantic")
    timestamp = value.get("captured_timestamp")
    sequence = value.get("measurement_sequence")
    preceding_dispatch = value.get("preceding_dispatch")
    unresolved_reservations = value.get("unresolved_intent_reservations")
    if (
        value.get("schema_version") != LOAD_ACCOUNTING_SCHEMA_VERSION
        or value.get("protocol") != REFILL_RECONCILIATION_PROTOCOL
        or value.get("intent_id") != intent["intent_id"]
        or value.get("refill_index") != refill_index
        or not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence < 0
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or float(timestamp) <= 0
        or not isinstance(scheduler, Mapping)
        or not isinstance(semantic, Mapping)
        or not isinstance(unresolved_reservations, list)
    ):
        raise ThroughputQualificationError(
            f"refill reconciliation {refill_index} identity is invalid"
        )
    scheduler_value = validate_scheduler_evidence(
        scheduler, intent=intent, sequence=sequence
    )
    semantic_value = validate_semantic_evidence(
        semantic, intent=intent, sequence=sequence
    )
    active = int(scheduler_value["active_qualification_cells"])
    configured_ceiling, saturation_target = _capacity_targets(intent)
    expected_eligible = bool(
        scheduler_value["ceiling"] == configured_ceiling
        and active == saturation_target
        and scheduler_value["unfinished_load_assignments"]
        >= saturation_target
        and not unresolved_reservations
        and semantic_value["integrity_incidents"] == 0
        and semantic_value["transport_censor_incidents"] == 0
        and semantic_value["load_integrity_incidents"] == 0
        and semantic_value["load_censor_incidents"] == 0
    )
    if (
        scheduler_value["captured_timestamp"] != timestamp
        or semantic_value["captured_timestamp"] != timestamp
        or scheduler_value["unfinished_load_assignments"]
        != semantic_value["unfinished_load_assignments"]
        or value.get("active_deficit") != saturation_target - active
        or value.get("measurement_eligible") is not expected_eligible
        or value.get("prior_refill_id")
        != (None if prior is None else prior["refill_id"])
        or (
            prior is not None
            and (
                float(timestamp) <= float(prior["captured_timestamp"])
                or int(sequence) < int(prior["measurement_sequence"])
                or int(
                    semantic_value["trusted_qid_execution_events"]
                )
                < int(
                    prior["semantic"][
                        "trusted_qid_execution_events"
                    ]
                )
                or any(
                    int(
                        semantic_value["load_strata_progress"][label]
                    )
                    < int(
                        prior["semantic"]["load_strata_progress"][
                            label
                        ]
                    )
                    for label in semantic_value[
                        "load_strata_progress"
                    ]
                )
            )
        )
    ):
        raise ThroughputQualificationError(
            f"refill reconciliation {refill_index} accounting drifted"
        )
    if preceding_dispatch is not None:
        preceding_batch_id = (
            preceding_dispatch.get("batch_id")
            if isinstance(preceding_dispatch, Mapping)
            else None
        )
        matching_prior_reservation = (
            None
            if prior is None
            else next(
                (
                    reservation
                    for reservation in prior[
                        "unresolved_intent_reservations"
                    ]
                    if reservation["batch_id"]
                    == preceding_batch_id
                ),
                None,
            )
        )
        expected_requested_tasks = (
            None
            if prior is None
            else (
                min(MAX_BATCH, int(prior["active_deficit"]))
                if int(prior["active_deficit"]) > 0
                else (
                    matching_prior_reservation["task_count"]
                    if matching_prior_reservation is not None
                    else None
                )
            )
        )
        consumed_jobs = {
            str(record["preceding_dispatch"]["job_id"])
            for record in ([] if prior is None else [prior])
            if record.get("preceding_dispatch") is not None
        }
        consumed_batches = {
            str(record["preceding_dispatch"]["batch_id"])
            for record in ([] if prior is None else [prior])
            if record.get("preceding_dispatch") is not None
        }
        if (
            not isinstance(preceding_dispatch, Mapping)
            or set(preceding_dispatch) != _REFILL_DISPATCH_FIELDS
            or prior is None
            or expected_requested_tasks is None
            or sequence != prior["measurement_sequence"]
            or _SHA256_RE.fullmatch(
                str(preceding_dispatch.get("cycle_id", ""))
            )
            is None
            or not isinstance(preceding_dispatch.get("run_id"), str)
            or not preceding_dispatch["run_id"]
            or not isinstance(
                preceding_dispatch.get("ledger_path"), str
            )
            or not Path(preceding_dispatch["ledger_path"]).is_absolute()
            or any(
                not isinstance(preceding_dispatch.get(field), int)
                or isinstance(preceding_dispatch.get(field), bool)
                or preceding_dispatch[field] <= 0
                for field in (
                    "requested_tasks",
                    "selected_tasks",
                    "accepted_tasks",
                )
            )
            or preceding_dispatch["requested_tasks"] > MAX_BATCH
            or preceding_dispatch["requested_tasks"]
            != expected_requested_tasks
            or preceding_dispatch["selected_tasks"]
            != preceding_dispatch["requested_tasks"]
            or preceding_dispatch["accepted_tasks"]
            != preceding_dispatch["selected_tasks"]
            or preceding_dispatch.get("intent_state")
            != "accepted"
            or preceding_dispatch.get("job_task_count")
            != preceding_dispatch["accepted_tasks"]
            or not str(preceding_dispatch.get("job_id", "")).isdigit()
            or str(preceding_dispatch["job_id"])
            not in scheduler_value["qualification_job_ids"]
            or _SHA256_RE.fullmatch(
                str(preceding_dispatch.get("batch_id", ""))
            )
            is None
            or preceding_dispatch["job_id"] in consumed_jobs
            or preceding_dispatch["batch_id"] in consumed_batches
            or _SHA256_RE.fullmatch(
                str(preceding_dispatch.get("tasks_sha256", ""))
            )
            is None
            or any(
                not isinstance(preceding_dispatch.get(field), str)
                or not Path(preceding_dispatch[field]).is_absolute()
                for field in (
                    "batch_manifest",
                    "sbatch_path",
                    "spooled_receipt_path",
                )
            )
            or any(
                _SHA256_RE.fullmatch(
                    str(preceding_dispatch.get(field, ""))
                )
                is None
                for field in (
                    "batch_manifest_sha256",
                    "sbatch_sha256",
                    "spooled_sbatch_sha256",
                    "spooled_receipt_sha256",
                    "submission_argv_sha256",
                )
            )
            or preceding_dispatch.get("submission_transport")
            != dispatch_sweeps.STDIN_EXACT_SUBMISSION_TRANSPORT
        ):
            raise ThroughputQualificationError(
                f"refill reconciliation {refill_index} dispatch binding "
                "is invalid"
            )
    reservation_batch_ids: set[str] = set()
    reservation_tasks = 0
    for reservation in unresolved_reservations:
        if (
            not isinstance(reservation, Mapping)
            or set(reservation) != _REFILL_UNRESOLVED_RESERVATION_FIELDS
            or _SHA256_RE.fullmatch(
                str(reservation.get("cycle_id", ""))
            )
            is None
            or not isinstance(reservation.get("run_id"), str)
            or not reservation["run_id"]
            or not isinstance(reservation.get("ledger_path"), str)
            or not Path(reservation["ledger_path"]).is_absolute()
            or _SHA256_RE.fullmatch(
                str(reservation.get("batch_id", ""))
            )
            is None
            or reservation.get("intent_state")
            not in {"prepared", "submitting"}
            or not isinstance(reservation.get("task_count"), int)
            or isinstance(reservation.get("task_count"), bool)
            or reservation["task_count"] <= 0
            or _SHA256_RE.fullmatch(
                str(reservation.get("tasks_sha256", ""))
            )
            is None
            or reservation["batch_id"] in reservation_batch_ids
        ):
            raise ThroughputQualificationError(
                f"refill reconciliation {refill_index} unresolved "
                "reservation is invalid"
            )
        reservation_batch_ids.add(str(reservation["batch_id"]))
        reservation_tasks += int(reservation["task_count"])
    if reservation_tasks > active:
        raise ThroughputQualificationError(
            f"refill reconciliation {refill_index} reserves more invisible "
            "tasks than its scheduler cut"
        )
    return dict(value)


def load_refill_reconciliations(
    root: Path,
    *,
    intent: Mapping[str, Any],
) -> list[dict[str, Any]]:
    files = _numbered_files(
        root / REFILL_RECONCILIATION_DIRECTORY,
        "REFILL",
    )
    result: list[dict[str, Any]] = []
    consumed_jobs: set[str] = set()
    consumed_batches: set[str] = set()
    for index, path in sorted(files.items()):
        if index != len(result):
            raise ThroughputQualificationError(
                "refill reconciliation journal is not contiguous from zero"
            )
        record = _validate_refill_reconciliation(
                _read_json(
                    path,
                    description=f"refill reconciliation {index}",
                    sealed=True,
                ),
                intent=intent,
                refill_index=index,
                prior=(None if not result else result[-1]),
            )
        dispatch = record["preceding_dispatch"]
        if dispatch is not None:
            job_id = str(dispatch["job_id"])
            batch_id = str(dispatch["batch_id"])
            if job_id in consumed_jobs or batch_id in consumed_batches:
                raise ThroughputQualificationError(
                    "refill reconciliation consumes an admission more than once"
                )
            consumed_jobs.add(job_id)
            consumed_batches.add(batch_id)
        result.append(
            record
        )
    return result


def record_refill_reconciliation(
    root: Path,
    *,
    intent: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    semantic: Mapping[str, Any],
    preceding_dispatch: Mapping[str, Any] | None,
    unresolved_intent_reservations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Seal every pre/post-refill scheduler cut outside the load numerator."""

    records = load_refill_reconciliations(root, intent=intent)
    index = len(records)
    scheduler_value = validate_scheduler_evidence(
        scheduler,
        intent=intent,
    )
    semantic_value = validate_semantic_evidence(
        semantic,
        intent=intent,
    )
    active = int(scheduler_value["active_qualification_cells"])
    configured_ceiling, saturation_target = _capacity_targets(intent)
    eligible = bool(
        scheduler_value["ceiling"] == configured_ceiling
        and active == saturation_target
        and scheduler_value["unfinished_load_assignments"]
        >= saturation_target
        and semantic_value["integrity_incidents"] == 0
        and semantic_value["transport_censor_incidents"] == 0
        and semantic_value["load_integrity_incidents"] == 0
        and semantic_value["load_censor_incidents"] == 0
        and not unresolved_intent_reservations
    )
    payload = _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": REFILL_RECONCILIATION_PROTOCOL,
            "intent_id": intent["intent_id"],
            "refill_index": index,
            "measurement_sequence": scheduler_value["sequence"],
            "captured_timestamp": scheduler_value[
                "captured_timestamp"
            ],
            "scheduler": scheduler_value,
            "semantic": semantic_value,
            "preceding_dispatch": (
                None
                if preceding_dispatch is None
                else dict(preceding_dispatch)
            ),
            "unresolved_intent_reservations": [
                dict(record)
                for record in unresolved_intent_reservations
            ],
            "prior_refill_id": (
                None if not records else records[-1]["refill_id"]
            ),
            "active_deficit": saturation_target - active,
            "measurement_eligible": eligible,
        },
        "refill_id",
    )
    path = (
        root
        / REFILL_RECONCILIATION_DIRECTORY
        / _evidence_filename("REFILL", index)
    )
    _write_once(
        path,
        payload,
        description=f"refill reconciliation {index}",
    )
    return _validate_refill_reconciliation(
        _read_json(
            path,
            description=f"refill reconciliation {index}",
            sealed=True,
        ),
        intent=intent,
        refill_index=index,
        prior=(None if not records else records[-1]),
    )


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
    cycle_ids = {
        str(record["cycle_id"])
        for record in semantic_value["load_cycle_inventory"]
    }
    authority_cycle_ids = {
        str(record["cycle_id"])
        for record in scheduler_value["cycle_execution_authorities"]
    }
    if (
        scheduler_value["unfinished_load_assignments"]
        != semantic_value["unfinished_load_assignments"]
        or not set(scheduler_value["active_cycle_ids"]).issubset(cycle_ids)
        or not set(scheduler_value["active_cycle_ids"]).issubset(
            authority_cycle_ids
        )
    ):
        raise ThroughputQualificationError(
            "scheduler and semantic load-cycle accounting differ"
        )
    transaction = _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
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
    receipt = _finalize_observation_transaction(
        root, intent=intent, transaction=transaction
    )
    _maybe_publish_load_window_intent(
        root,
        intent=intent,
        observation={
            "receipt": receipt,
            "scheduler": scheduler_value,
            "semantic": semantic_value,
            "receipt_path": (
                root
                / OBSERVATION_DIRECTORY
                / _evidence_filename("OBSERVATION", sequence)
            ),
        },
    )
    return receipt


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
        value.get("schema_version") != LOAD_ACCOUNTING_SCHEMA_VERSION
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
    cycle_ids = {
        str(record["cycle_id"])
        for record in semantic_value["load_cycle_inventory"]
    }
    authority_cycle_ids = {
        str(record["cycle_id"])
        for record in scheduler_value["cycle_execution_authorities"]
    }
    if (
        scheduler_value["unfinished_load_assignments"]
        != semantic_value["unfinished_load_assignments"]
        or not set(scheduler_value["active_cycle_ids"]).issubset(cycle_ids)
        or not set(scheduler_value["active_cycle_ids"]).issubset(
            authority_cycle_ids
        )
    ):
        raise ThroughputQualificationError(
            f"observation transaction {sequence} load accounting differs"
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
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
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


_LOAD_WINDOW_INTENT_FIELDS = {
    "schema_version",
    "protocol",
    "qualification_intent_id",
    "attempt_id",
    "plan_id",
    "event_unit",
    "start_sequence",
    "start_timestamp",
    "start_observation",
    "start_trusted_qid_execution_events",
    "start_load_strata_progress",
    "start_cycle_ids",
    "configured_client_ceiling",
    "certified_saturation_target",
    "required_active_assignments",
    "minimum_unfinished_assignments",
    "minimum_duration_seconds",
    "maximum_observation_gap_seconds",
    "minimum_execution_events",
    "minimum_events_per_day",
    "window_intent_id",
}


def _is_clean_load_window_start(
    observation: Mapping[str, Any],
    *,
    configured_client_ceiling: int = CEILINGS[-1],
    certified_saturation_target: int = CEILINGS[-1],
) -> bool:
    scheduler = observation["scheduler"]
    semantic = observation["semantic"]
    return bool(
        scheduler["ceiling"] == configured_client_ceiling
        and scheduler["active_qualification_cells"]
        == certified_saturation_target
        and scheduler["unfinished_load_assignments"]
        >= certified_saturation_target
        and scheduler["qualification_tasks_only"] is True
        and scheduler["production_run_ids"] == []
        and semantic["integrity_incidents"] == 0
        and semantic["transport_censor_incidents"] == 0
        and semantic["load_integrity_incidents"] == 0
        and semantic["load_censor_incidents"] == 0
    )


def _load_window_intent_payload(
    root: Path,
    *,
    intent: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    configured_ceiling, saturation_target = _capacity_targets(intent)
    if not _is_clean_load_window_start(
        observation,
        configured_client_ceiling=configured_ceiling,
        certified_saturation_target=saturation_target,
    ):
        raise ThroughputQualificationError(
            "load-window intent requires a clean signed saturation cut"
        )
    receipt = observation["receipt"]
    semantic = observation["semantic"]
    receipt_path = Path(str(observation["receipt_path"])).resolve()
    expected_path = (
        root
        / OBSERVATION_DIRECTORY
        / _evidence_filename("OBSERVATION", int(receipt["sequence"]))
    ).resolve()
    if receipt_path != expected_path:
        raise ThroughputQualificationError(
            "load-window start observation path escaped its attempt"
        )
    attempt_id = (
        intent.get("attempt_id")
        or (
            Path(str(intent["run_root"])).parent.name
            if isinstance(intent.get("run_root"), str)
            else None
        )
    )
    # The generation-scoped run-root parent is the attempt ID in production.
    # Tests may supply only the sealed intent, so the value remains an immutable
    # string binding rather than a guessed ordinal.
    if not isinstance(attempt_id, str) or not attempt_id:
        attempt_id = _sha256_bytes(
            _canonical_bytes(
                {
                    "intent_id": intent["intent_id"],
                    "run_root": intent["run_root"],
                }
            )
        )
    return _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": LOAD_WINDOW_INTENT_PROTOCOL,
            "qualification_intent_id": intent["intent_id"],
            "attempt_id": attempt_id,
            "plan_id": intent["plan_id"],
            "event_unit": "trusted_qid_execution_events",
            "start_sequence": receipt["sequence"],
            "start_timestamp": receipt["captured_timestamp"],
            "start_observation": {
                "path": str(receipt_path),
                "sha256": _sha256_file(receipt_path),
                "observation_id": receipt["observation_id"],
            },
            "start_trusted_qid_execution_events": semantic[
                "trusted_qid_execution_events"
            ],
            "start_load_strata_progress": dict(
                semantic["load_strata_progress"]
            ),
            "start_cycle_ids": [
                record["cycle_id"]
                for record in semantic["load_cycle_inventory"]
            ],
            "configured_client_ceiling": configured_ceiling,
            "certified_saturation_target": saturation_target,
            "required_active_assignments": saturation_target,
            "minimum_unfinished_assignments": saturation_target,
            "minimum_duration_seconds": HEALTH_SOAK_384_SECONDS,
            "maximum_observation_gap_seconds": (
                MAX_OBSERVATION_GAP_SECONDS
            ),
            "minimum_execution_events": (
                MIN_LOAD_WINDOW_EXECUTION_EVENTS
            ),
            "minimum_events_per_day": MIN_QIDS_PER_DAY,
        },
        "window_intent_id",
    )


def validate_load_window_intent(
    value: Mapping[str, Any],
    *,
    root: Path,
    intent: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if set(value) != _LOAD_WINDOW_INTENT_FIELDS:
        raise ThroughputQualificationError(
            "load-window intent fields drifted"
        )
    _verify_identity(
        value,
        "window_intent_id",
        description="load-window intent",
    )
    configured_ceiling, saturation_target = _capacity_targets(intent)
    eligible = [
        observation
        for observation in observations
        if _is_clean_load_window_start(
            observation,
            configured_client_ceiling=configured_ceiling,
            certified_saturation_target=saturation_target,
        )
    ]
    if not eligible:
        raise ThroughputQualificationError(
            "load-window intent exists without an eligible observation"
        )
    expected = _load_window_intent_payload(
        root,
        intent=intent,
        observation=eligible[0],
    )
    if dict(value) != expected:
        raise ThroughputQualificationError(
            "load-window intent is not bound to the first clean signed "
            "saturation "
            "observation; reset or cherry-pick is forbidden"
        )
    return expected


def _maybe_publish_load_window_intent(
    root: Path,
    *,
    intent: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    configured_ceiling, saturation_target = _capacity_targets(intent)
    if not _is_clean_load_window_start(
        observation,
        configured_client_ceiling=configured_ceiling,
        certified_saturation_target=saturation_target,
    ):
        return None
    path = root / LOAD_WINDOW_INTENT_NAME
    if path.exists() or path.is_symlink():
        existing = _read_json(
            path,
            description="load-window intent",
            sealed=True,
        )
        _verify_identity(
            existing,
            "window_intent_id",
            description="load-window intent",
        )
        return existing
    payload = _load_window_intent_payload(
        root, intent=intent, observation=observation
    )
    _write_once(
        path,
        payload,
        description="non-resettable load-window intent",
    )
    return payload


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
            receipt.get("schema_version") != LOAD_ACCOUNTING_SCHEMA_VERSION
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
    window_path = root / LOAD_WINDOW_INTENT_NAME
    configured_ceiling, saturation_target = _capacity_targets(intent)
    eligible = [
        observation
        for observation in loaded
        if _is_clean_load_window_start(
            observation,
            configured_client_ceiling=configured_ceiling,
            certified_saturation_target=saturation_target,
        )
    ]
    if window_path.exists() or window_path.is_symlink():
        validate_load_window_intent(
            _read_json(
                window_path,
                description="load-window intent",
                sealed=True,
            ),
            root=root,
            intent=intent,
            observations=loaded,
        )
    elif eligible and recover_transactions:
        _maybe_publish_load_window_intent(
            root, intent=intent, observation=eligible[0]
        )
    return loaded


def evaluate_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    load_window_intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate unique semantics and repeated load execution independently."""

    if not observations:
        raise ThroughputQualificationError(
            "qualification has no scheduler/semantic evidence"
        )
    first = observations[0]
    first_semantic = first["semantic"]
    if (
        first["scheduler"]["ceiling"] != CEILINGS[0]
        or first["scheduler"]["active_qualification_cells"] != 0
        or first_semantic["validated_qids"] != 0
        or first_semantic["useful_qids"] != 0
        or first_semantic["trusted_qid_execution_events"] != 0
    ):
        raise ThroughputQualificationError(
            "qualification evidence must begin with a clean zero-event "
            "ceiling-24 baseline"
        )

    ceilings_seen: list[int] = []
    peak_by_ceiling = {ceiling: 0 for ceiling in CEILINGS}
    prior_useful = -1
    prior_validated = -1
    prior_events = -1
    prior_load_progress = {
        label: 0 for label in expected_stratum_labels()
    }
    prior_cycle_ids: list[str] = []
    completion_timestamp: float | None = None
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
        events = int(semantic["trusted_qid_execution_events"])
        cycle_ids = [
            str(record["cycle_id"])
            for record in semantic["load_cycle_inventory"]
        ]
        if (
            useful < prior_useful
            or validated < prior_validated
            or events < prior_events
            or cycle_ids[: len(prior_cycle_ids)] != prior_cycle_ids
            or any(
                int(semantic["load_strata_progress"][label])
                < prior_load_progress[label]
                for label in prior_load_progress
            )
        ):
            raise ThroughputQualificationError(
                "qualification semantic or execution-event progress regressed"
            )
        prior_useful = useful
        prior_validated = validated
        prior_events = events
        prior_cycle_ids = cycle_ids
        prior_load_progress = {
            label: int(semantic["load_strata_progress"][label])
            for label in expected_stratum_labels()
        }
        if (
            semantic["integrity_incidents"] != 0
            or semantic["transport_censor_incidents"] != 0
            or semantic["load_integrity_incidents"] != 0
            or semantic["load_censor_incidents"] != 0
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

    if ceilings_seen != list(CEILINGS):
        raise ThroughputQualificationError(
            f"qualification did not exercise every ceiling: {ceilings_seen}"
        )
    if completion_timestamp is None:
        raise ThroughputQualificationError(
            "semantic reference cycle is not schema-5 complete"
        )

    if load_window_intent is None:
        receipt_path = observations[0].get("receipt_path")
        if not isinstance(receipt_path, Path):
            raise ThroughputQualificationError(
                "qualification lacks its immutable load-window intent"
            )
        root = receipt_path.parent.parent
        path = root / LOAD_WINDOW_INTENT_NAME
        if not path.is_file() or path.is_symlink():
            raise ThroughputQualificationError(
                "qualification lacks its immutable load-window intent"
            )
        load_window_intent = _read_json(
            path,
            description="load-window intent",
            sealed=True,
        )
    start_sequence = load_window_intent.get("start_sequence")
    configured_ceiling = load_window_intent.get(
        "configured_client_ceiling"
    )
    saturation_target = load_window_intent.get(
        "certified_saturation_target"
    )
    _verify_identity(
        load_window_intent,
        "window_intent_id",
        description="load-window intent",
    )
    if (
        load_window_intent.get("schema_version")
        != LOAD_ACCOUNTING_SCHEMA_VERSION
        or load_window_intent.get("protocol")
        != LOAD_WINDOW_INTENT_PROTOCOL
        or load_window_intent.get("event_unit")
        != "trusted_qid_execution_events"
        or configured_ceiling != CEILINGS[-1]
        or not isinstance(saturation_target, int)
        or isinstance(saturation_target, bool)
        or not 0 < saturation_target <= configured_ceiling
        or load_window_intent.get("required_active_assignments")
        != saturation_target
        or load_window_intent.get("minimum_unfinished_assignments")
        != saturation_target
        or not isinstance(start_sequence, int)
        or isinstance(start_sequence, bool)
        or not 0 <= start_sequence < len(observations)
        or load_window_intent.get("minimum_execution_events")
        != MIN_LOAD_WINDOW_EXECUTION_EVENTS
        or load_window_intent.get("minimum_duration_seconds")
        != HEALTH_SOAK_384_SECONDS
        or load_window_intent.get("maximum_observation_gap_seconds")
        != MAX_OBSERVATION_GAP_SECONDS
    ):
        raise ThroughputQualificationError(
            "load-window intent schema or accounting unit drifted"
        )
    expected_peaks = {
        ceiling: _phase_active_target(ceiling, saturation_target)
        for ceiling in CEILINGS
    }
    if peak_by_ceiling != expected_peaks:
        raise ThroughputQualificationError(
            "qualification did not reconcile every configured ceiling to "
            f"its signed saturation target: {peak_by_ceiling} != "
            f"{expected_peaks}"
        )
    start = observations[start_sequence]
    if (
        start["receipt"]["observation_id"]
        != load_window_intent["start_observation"]["observation_id"]
        or start["receipt"]["captured_timestamp"]
        != load_window_intent["start_timestamp"]
        or start["semantic"]["trusted_qid_execution_events"]
        != load_window_intent["start_trusted_qid_execution_events"]
        or start["semantic"]["load_strata_progress"]
        != load_window_intent["start_load_strata_progress"]
        or [
            record["cycle_id"]
            for record in start["semantic"]["load_cycle_inventory"]
        ]
        != load_window_intent["start_cycle_ids"]
    ):
        raise ThroughputQualificationError(
            "load-window start observation differs from its immutable intent"
        )
    if not _is_clean_load_window_start(
        start,
        configured_client_ceiling=configured_ceiling,
        certified_saturation_target=saturation_target,
    ):
        raise ThroughputQualificationError(
            "load-window start is no longer a clean signed saturation cut"
        )

    window: list[Mapping[str, Any]] = [start]
    end: Mapping[str, Any] | None = None
    last_mature_candidate: tuple[int, float, int, bool] | None = None
    for observation in observations[start_sequence + 1 :]:
        prior = window[-1]
        gap = (
            float(observation["receipt"]["captured_timestamp"])
            - float(prior["receipt"]["captured_timestamp"])
        )
        if gap > MAX_OBSERVATION_GAP_SECONDS:
            raise ThroughputQualificationError(
                f"load window contains a {gap:g}-second evidence gap"
        )
        scheduler = observation["scheduler"]
        semantic = observation["semantic"]
        if (
            semantic["integrity_incidents"] != 0
            or semantic["transport_censor_incidents"] != 0
            or semantic["load_integrity_incidents"] != 0
            or semantic["load_censor_incidents"] != 0
        ):
            raise ThroughputQualificationError(
                "qualification contains an integrity or censor incident "
                "inside the non-resettable load window"
            )
        if (
            scheduler["ceiling"] != configured_ceiling
            or scheduler["active_qualification_cells"]
            != saturation_target
            or scheduler["unfinished_load_assignments"]
            < saturation_target
        ):
            if last_mature_candidate is not None:
                (
                    candidate_events,
                    candidate_duration,
                    _,
                    candidate_all_strata,
                ) = last_mature_candidate
                if not candidate_all_strata:
                    raise ThroughputQualificationError(
                        "load window did not make positive trusted execution "
                        "progress in every stratum"
                    )
                raise QualificationCapacityTransitionRequired(
                    _capacity_shortfall_reason(
                        candidate_events,
                        candidate_duration,
                    )
                )
            raise ThroughputQualificationError(
                "the non-resettable load window lost its certified capacity "
                "target cut, backlog, "
                "or scientific integrity before 7,200 seconds"
            )
        window.append(observation)
        duration = (
            float(observation["receipt"]["captured_timestamp"])
            - float(start["receipt"]["captured_timestamp"])
        )
        if duration >= HEALTH_SOAK_384_SECONDS:
            candidate_events = (
                int(
                    observation["semantic"][
                        "trusted_qid_execution_events"
                    ]
                )
                - int(
                    start["semantic"][
                        "trusted_qid_execution_events"
                    ]
                )
            )
            candidate_rate = math.floor(
                candidate_events * 86_400.0 / duration
            )
            candidate_all_strata = all(
                int(
                    observation["semantic"][
                        "load_strata_progress"
                    ][label]
                )
                > int(
                    start["semantic"]["load_strata_progress"][label]
                )
                for label in expected_stratum_labels()
            )
            last_mature_candidate = (
                candidate_events,
                duration,
                candidate_rate,
                candidate_all_strata,
            )
            if (
                candidate_events >= MIN_LOAD_WINDOW_EXECUTION_EVENTS
                and candidate_rate >= MIN_QIDS_PER_DAY
                and candidate_all_strata
            ):
                end = observation
                break
    if end is None:
        if last_mature_candidate is not None:
            (
                candidate_events,
                candidate_duration,
                _,
                candidate_all_strata,
            ) = last_mature_candidate
            if not candidate_all_strata:
                raise ThroughputQualificationError(
                    "load window did not make positive trusted execution "
                    "progress in every stratum"
                )
            raise QualificationCapacityTransitionRequired(
                _capacity_shortfall_reason(
                    candidate_events,
                    candidate_duration,
                )
            )
        covered = (
            float(window[-1]["receipt"]["captured_timestamp"])
            - float(start["receipt"]["captured_timestamp"])
        )
        raise ThroughputQualificationError(
            f"load window covers only {covered:g} seconds"
        )

    loaded_seconds = (
        float(end["receipt"]["captured_timestamp"])
        - float(start["receipt"]["captured_timestamp"])
    )
    execution_events = (
        int(end["semantic"]["trusted_qid_execution_events"])
        - int(start["semantic"]["trusted_qid_execution_events"])
    )
    stratum_deltas = {
        label: (
            int(end["semantic"]["load_strata_progress"][label])
            - int(start["semantic"]["load_strata_progress"][label])
        )
        for label in expected_stratum_labels()
    }
    if any(delta <= 0 for delta in stratum_deltas.values()):
        raise ThroughputQualificationError(
            "load window did not make positive trusted execution progress in "
            "every stratum"
        )
    if execution_events < MIN_LOAD_WINDOW_EXECUTION_EVENTS:
        raise ThroughputQualificationError(
            f"load window recorded {execution_events:,} trusted execution "
            f"events; at least {MIN_LOAD_WINDOW_EXECUTION_EVENTS:,} are "
            "required for the exact 7,200-second threshold"
        )
    throughput = math.floor(execution_events * 86_400.0 / loaded_seconds)
    if throughput < MIN_QIDS_PER_DAY:
        raise QualificationCapacityTransitionRequired(
            _capacity_shortfall_reason(
                execution_events,
                loaded_seconds,
            )
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
            "final qualification observation is not quiescent and the semantic "
            "reference is not schema-5 complete"
        )
    final_cycles = final_semantic["load_cycle_inventory"]
    if any(
        record["status"] not in {"complete", "load_window_drained"}
        for record in final_cycles
    ):
        raise ThroughputQualificationError(
            "replay cycles were not gracefully completed or load-window drained"
        )
    receipt_path = observations[0].get("receipt_path")
    if not isinstance(receipt_path, Path):
        raise ThroughputQualificationError(
            "qualification observations lack their sealed refill journal root"
        )
    root = receipt_path.parent.parent
    refill_intent = _read_json(
        root / INTENT_NAME,
        description="qualification intent for refill accounting",
        sealed=True,
    )
    if refill_intent.get("intent_id") != first["receipt"]["intent_id"]:
        raise ThroughputQualificationError(
            "refill journal intent differs from observation evidence"
        )
    refill_records = [
        record
        for record in load_refill_reconciliations(
            root, intent=refill_intent
        )
        if float(start["receipt"]["captured_timestamp"])
        <= float(record["captured_timestamp"])
        <= float(end["receipt"]["captured_timestamp"])
    ]
    deficit_records = [
        record
        for record in refill_records
        if int(record["active_deficit"]) > 0
    ]
    refill_wall_seconds = 0.0
    by_measurement: dict[int, list[Mapping[str, Any]]] = {}
    for record in refill_records:
        by_measurement.setdefault(
            int(record["measurement_sequence"]), []
        ).append(record)
    for sequence, records in by_measurement.items():
        deficits = [
            record for record in records if record["active_deficit"] > 0
        ]
        if not deficits:
            continue
        eligible = [
            record
            for record in records
            if record["measurement_eligible"] is True
        ]
        if (
            not eligible
            or float(eligible[-1]["captured_timestamp"])
            <= float(deficits[0]["captured_timestamp"])
        ):
            raise ThroughputQualificationError(
                f"refill sequence {sequence} lacks a later signed saturation "
                "cut"
            )
        refill_wall_seconds += (
            float(eligible[-1]["captured_timestamp"])
            - float(deficits[0]["captured_timestamp"])
        )

    unique_design = {
        "cells": CELL_COUNT,
        "qids": TOTAL_QIDS,
        "semantic_reference_cycle": final_semantic[
            "semantic_reference_cycle"
        ],
    }
    load_execution = {
        "unit": "trusted_qid_execution_events",
        "repeated_coordinates": True,
        "window_intent_id": load_window_intent["window_intent_id"],
        "window_start_sequence": start_sequence,
        "window_end_sequence": end["receipt"]["sequence"],
        "window_start_timestamp": start["receipt"]["captured_timestamp"],
        "window_end_timestamp": end["receipt"]["captured_timestamp"],
        "window_duration_seconds": math.floor(loaded_seconds),
        "trusted_execution_events": execution_events,
        "replay_execution_events_total": final_semantic[
            "replay_qid_execution_events"
        ],
        "cycle_count": len(final_cycles),
        "cycle_inventory": [dict(record) for record in final_cycles],
        "configured_client_ceiling": configured_ceiling,
        "certified_saturation_target": saturation_target,
        "certified_saturation_target_cuts": True,
        "work_conserving_refill": True,
        "sealed_refill_deficit_journal": True,
        "refill_deficit_scan_count": len(deficit_records),
        "refill_wall_seconds": refill_wall_seconds,
        "rate_denominator_includes_refill_wall_time": True,
        "minimum_unfinished_assignments": saturation_target,
        "all_strata_progress": True,
        "stratum_execution_event_deltas": stratum_deltas,
        "throughput_events_per_day": throughput,
    }
    return {
        "passed": True,
        "cells": CELL_COUNT,
        "qids": TOTAL_QIDS,
        "ceilings": list(CEILINGS),
        "peak_active_cells": {
            str(key): value for key, value in peak_by_ceiling.items()
        },
        "unique_design": unique_design,
        "load_execution": load_execution,
        # Compatibility aliases are explicitly execution-event quantities.
        "health_soak_384_seconds": math.floor(loaded_seconds),
        "loaded_384_seconds": math.floor(loaded_seconds),
        "loaded_384_useful_qids": execution_events,
        "loaded_384_observation_count": len(window),
        "configured_client_ceiling": configured_ceiling,
        "certified_saturation_target": saturation_target,
        "certified_saturation_target_cuts": True,
        "throughput_qids_per_day": throughput,
        "throughput_unit": "trusted_qid_execution_events",
        "every_stratum_progress": True,
        "integrity_incidents": 0,
        "transport_censor_incidents": 0,
        "completion_timestamp": completion_timestamp,
        "observation_count": len(observations),
    }


def load_window_ready(
    observations: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether the immutable window has already met the r10 load gate.

    A started window that loses its capacity-target cut, backlog, cadence, or
    integrity raises
    instead of silently searching for a later interval.
    """

    if not observations:
        return False
    receipt_path = observations[0].get("receipt_path")
    if not isinstance(receipt_path, Path):
        return False
    path = receipt_path.parent.parent / LOAD_WINDOW_INTENT_NAME
    if not path.exists():
        return False
    window_intent = _read_json(
        path, description="load-window intent", sealed=True
    )
    configured_ceiling = int(
        window_intent["configured_client_ceiling"]
    )
    saturation_target = int(
        window_intent["certified_saturation_target"]
    )
    start_sequence = int(window_intent["start_sequence"])
    start = observations[start_sequence]
    prior = start
    for observation in observations[start_sequence + 1 :]:
        gap = (
            float(observation["receipt"]["captured_timestamp"])
            - float(prior["receipt"]["captured_timestamp"])
        )
        scheduler = observation["scheduler"]
        semantic = observation["semantic"]
        if gap > MAX_OBSERVATION_GAP_SECONDS:
            raise ThroughputQualificationError(
                "non-resettable load window exceeded its observation cadence"
            )
        if (
            scheduler["ceiling"] != configured_ceiling
            or scheduler["active_qualification_cells"]
            != saturation_target
            or scheduler["unfinished_load_assignments"]
            < saturation_target
            or semantic["integrity_incidents"] != 0
            or semantic["transport_censor_incidents"] != 0
            or semantic["load_integrity_incidents"] != 0
            or semantic["load_censor_incidents"] != 0
        ):
            return False
        duration = (
            float(observation["receipt"]["captured_timestamp"])
            - float(start["receipt"]["captured_timestamp"])
        )
        events = (
            int(semantic["trusted_qid_execution_events"])
            - int(
                start["semantic"]["trusted_qid_execution_events"]
            )
        )
        if duration >= HEALTH_SOAK_384_SECONDS:
            rate = math.floor(events * 86_400.0 / duration)
            if (
                events >= MIN_LOAD_WINDOW_EXECUTION_EVENTS
                and rate >= MIN_QIDS_PER_DAY
                and all(
                    int(semantic["load_strata_progress"][label])
                    > int(
                        start["semantic"]["load_strata_progress"][
                            label
                        ]
                    )
                    for label in expected_stratum_labels()
                )
            ):
                return True
        prior = observation
    return False


_LOAD_WINDOW_END_INTENT_FIELDS = {
    "schema_version",
    "protocol",
    "qualification_intent_id",
    "window_intent_id",
    "end_observation",
    "end_sequence",
    "end_timestamp",
    "trusted_execution_events",
    "throughput_events_per_day",
    "event_unit",
    "admission_closed",
    "state",
    "end_intent_id",
}


def publish_load_window_end_intent(
    root: Path,
    *,
    intent: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Commit qualification before draining, so a crash cannot readmit work."""

    if not load_window_ready(observations):
        raise ThroughputQualificationError(
            "load-window end cannot be committed before qualification"
        )
    window = _read_json(
        root / LOAD_WINDOW_INTENT_NAME,
        description="load-window intent",
        sealed=True,
    )
    start = observations[int(window["start_sequence"])]
    qualifying: Mapping[str, Any] | None = None
    for observation in observations[int(window["start_sequence"]) + 1 :]:
        duration = (
            float(observation["receipt"]["captured_timestamp"])
            - float(start["receipt"]["captured_timestamp"])
        )
        events = (
            int(
                observation["semantic"][
                    "trusted_qid_execution_events"
                ]
            )
            - int(
                start["semantic"]["trusted_qid_execution_events"]
            )
        )
        if (
            duration >= HEALTH_SOAK_384_SECONDS
            and math.floor(events * 86_400.0 / duration)
            >= MIN_QIDS_PER_DAY
            and all(
                int(
                    observation["semantic"][
                        "load_strata_progress"
                    ][label]
                )
                > int(
                    start["semantic"]["load_strata_progress"][label]
                )
                for label in expected_stratum_labels()
            )
        ):
            qualifying = observation
            break
    if qualifying is None:
        raise ThroughputQualificationError(
            "load-window readiness did not yield a qualifying end"
        )
    duration = (
        float(qualifying["receipt"]["captured_timestamp"])
        - float(start["receipt"]["captured_timestamp"])
    )
    events = (
        int(
            qualifying["semantic"]["trusted_qid_execution_events"]
        )
        - int(start["semantic"]["trusted_qid_execution_events"])
    )
    receipt_path = Path(str(qualifying["receipt_path"])).resolve()
    payload = _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": LOAD_WINDOW_END_INTENT_PROTOCOL,
            "qualification_intent_id": intent["intent_id"],
            "window_intent_id": window["window_intent_id"],
            "end_observation": {
                "path": str(receipt_path),
                "sha256": _sha256_file(receipt_path),
                "observation_id": qualifying["receipt"][
                    "observation_id"
                ],
            },
            "end_sequence": qualifying["receipt"]["sequence"],
            "end_timestamp": qualifying["receipt"][
                "captured_timestamp"
            ],
            "trusted_execution_events": events,
            "throughput_events_per_day": math.floor(
                events * 86_400.0 / duration
            ),
            "event_unit": "trusted_qid_execution_events",
            "admission_closed": True,
            "state": "draining",
        },
        "end_intent_id",
    )
    path = root / LOAD_WINDOW_END_INTENT_NAME
    _write_once(
        path,
        payload,
        description="load-window end-before-drain intent",
    )
    observed = _read_json(
        path,
        description="load-window end-before-drain intent",
        sealed=True,
    )
    if set(observed) != _LOAD_WINDOW_END_INTENT_FIELDS:
        raise ThroughputQualificationError(
            "load-window end intent fields drifted"
        )
    _verify_identity(
        observed,
        "end_intent_id",
        description="load-window end-before-drain intent",
    )
    if observed != payload:
        raise ThroughputQualificationError(
            "load-window end intent conflicts with the first qualifying end"
        )
    return observed


_LOAD_WINDOW_DRAIN_FIELDS = {
    "schema_version",
    "protocol",
    "qualification_intent_id",
    "window_intent_id",
    "drain_observation",
    "active_assignments",
    "cycle_inventory",
    "partial_cycle_status",
    "analysis_ingestion_allowed",
    "drain_id",
}


def publish_load_window_drain(
    root: Path,
    *,
    intent: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the graceful no-new-admission drain after the accepted window."""

    if not observations:
        raise ThroughputQualificationError(
            "cannot drain a qualification without observations"
        )
    final = observations[-1]
    scheduler = final["scheduler"]
    semantic = final["semantic"]
    if scheduler["active_qualification_cells"] != 0:
        raise ThroughputQualificationError(
            "load replay drain requires zero active assignments"
        )
    inventory = [dict(record) for record in semantic["load_cycle_inventory"]]
    if any(
        record["status"] not in {"complete", "load_window_drained"}
        for record in inventory
    ):
        raise ThroughputQualificationError(
            "load replay drain contains an active or unclassified cycle"
        )
    receipt_path = Path(str(final["receipt_path"])).resolve()
    payload = _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": LOAD_WINDOW_DRAIN_PROTOCOL,
            "qualification_intent_id": intent["intent_id"],
            "window_intent_id": evaluation["load_execution"][
                "window_intent_id"
            ],
            "drain_observation": {
                "path": str(receipt_path),
                "sha256": _sha256_file(receipt_path),
                "observation_id": final["receipt"]["observation_id"],
            },
            "active_assignments": 0,
            "cycle_inventory": inventory,
            "partial_cycle_status": "load_window_drained",
            "analysis_ingestion_allowed": False,
        },
        "drain_id",
    )
    path = root / LOAD_WINDOW_DRAIN_NAME
    _write_once(
        path,
        payload,
        description="load-window graceful drain",
    )
    return validate_load_window_drain(
        _read_json(
            path,
            description="load-window graceful drain",
            sealed=True,
        ),
        root=root,
        intent=intent,
        observations=observations,
        evaluation=evaluation,
    )


def validate_load_window_drain(
    value: Mapping[str, Any],
    *,
    root: Path,
    intent: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    if set(value) != _LOAD_WINDOW_DRAIN_FIELDS:
        raise ThroughputQualificationError(
            "load-window drain fields drifted"
        )
    _verify_identity(
        value,
        "drain_id",
        description="load-window graceful drain",
    )
    final = observations[-1]
    receipt_path = Path(str(final["receipt_path"])).resolve()
    expected = _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": LOAD_WINDOW_DRAIN_PROTOCOL,
            "qualification_intent_id": intent["intent_id"],
            "window_intent_id": evaluation["load_execution"][
                "window_intent_id"
            ],
            "drain_observation": {
                "path": str(receipt_path),
                "sha256": _sha256_file(receipt_path),
                "observation_id": final["receipt"]["observation_id"],
            },
            "active_assignments": 0,
            "cycle_inventory": [
                dict(record)
                for record in final["semantic"]["load_cycle_inventory"]
            ],
            "partial_cycle_status": "load_window_drained",
            "analysis_ingestion_allowed": False,
        },
        "drain_id",
    )
    if dict(value) != expected:
        raise ThroughputQualificationError(
            "load-window drain does not bind the final quiescent cycle state"
        )
    return expected


def _cycle_run_root_inventory(
    root: Path,
    cycle_inventory: Sequence[Mapping[str, Any]],
    *,
    require_read_only: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in cycle_inventory:
        index = int(record["cycle_index"])
        intent_path = (
            root
            / LOAD_CYCLE_DIRECTORY
            / f"cycle-{index:06d}"
            / CYCLE_INTENT_NAME
        )
        cycle_intent = _read_json(
            intent_path,
            description=f"load cycle {index} intent",
            sealed=True,
        )
        _verify_identity(
            cycle_intent,
            "cycle_id",
            description=f"load cycle {index} intent",
        )
        run_root = Path(str(cycle_intent.get("run_root", "")))
        if (
            cycle_intent.get("protocol") != CYCLE_INTENT_PROTOCOL
            or cycle_intent.get("cycle_index") != index
            or cycle_intent.get("cycle_id") != record["cycle_id"]
            or cycle_intent.get("run_id") != record["run_id"]
            or cycle_intent.get("estimand_excluded") is not True
            or cycle_intent.get("primary_analysis_eligible") is not False
            or not run_root.is_absolute()
            or run_root.name != record["run_id"]
        ):
            raise ThroughputQualificationError(
                f"load cycle {index} run-root binding drifted"
            )
        if require_read_only:
            _assert_tree_read_only(
                run_root,
                description=f"sealed load cycle {index} run",
            )
        result.append(
            {
                "cycle_index": index,
                "cycle_id": record["cycle_id"],
                "run_id": record["run_id"],
                "semantic_reference": record["semantic_reference"],
                "estimand_excluded": True,
                "primary_analysis_eligible": False,
                **_tree_content_inventory(
                    run_root,
                    description=f"load cycle {index} run",
                ),
            }
        )
    return result


def _refill_evidence_inventory(
    root: Path,
    *,
    intent: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for refill in load_refill_reconciliations(root, intent=intent):
        path = (
            root
            / REFILL_RECONCILIATION_DIRECTORY
            / _evidence_filename("REFILL", refill["refill_index"])
        )
        inventory.append(
            {
                "refill_index": refill["refill_index"],
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "refill_id": refill["refill_id"],
            }
        )
    return inventory


def _verify_refill_dispatch_ledger_bindings(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    require_read_only: bool,
) -> dict[str, Any]:
    """Reopen final cycle ledgers and prove exactly-once refill consumption."""

    records = load_refill_reconciliations(
        context.qualification_root, intent=intent
    )
    if not records:
        return {
            "consumed_jobs": [],
            "consumed_batches": [],
            "final_ledger_sha256": {},
        }
    cycles = {
        cycle.cycle_id: cycle
        for cycle in load_cycle_inventory(context, intent=intent)
    }
    ledgers: dict[str, Mapping[str, Any]] = {}
    ledger_hashes: dict[str, str] = {}
    for cycle_id, cycle in cycles.items():
        ledger_path = (cycle.dispatcher_state / "ledger.json").resolve()
        if not ledger_path.is_file() or ledger_path.is_symlink():
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} final dispatcher ledger "
                "is missing or unsafe"
            )
        if require_read_only and stat.S_IMODE(
            ledger_path.stat().st_mode
        ) & 0o222:
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} final dispatcher ledger "
                "is writable"
            )
        ledgers[cycle_id] = dispatch_sweeps._load_ledger(  # noqa: SLF001
            ledger_path
        )
        ledger_hashes[cycle_id] = _sha256_file(ledger_path)
    consumed_jobs: set[str] = set()
    consumed_batches: set[str] = set()
    for record in records:
        binding = record["preceding_dispatch"]
        if binding is None:
            continue
        cycle_id = str(binding["cycle_id"])
        cycle = cycles.get(cycle_id)
        ledger = ledgers.get(cycle_id)
        if (
            cycle is None
            or ledger is None
            or binding["run_id"] != cycle.run_id
            or Path(binding["ledger_path"])
            != (cycle.dispatcher_state / "ledger.json").resolve()
        ):
            raise ThroughputQualificationError(
                "refill dispatch does not bind its exact final cycle ledger"
            )
        expected = _refill_dispatch_from_ledger(
            cycle=cycle,
            ledger=ledger,
            job_id=str(binding["job_id"]),
            requested_tasks=int(binding["requested_tasks"]),
        )
        if binding != expected:
            raise ThroughputQualificationError(
                "refill dispatch differs from its final accepted "
                "intent/job/task transaction"
            )
        job_id = str(binding["job_id"])
        batch_id = str(binding["batch_id"])
        if job_id in consumed_jobs or batch_id in consumed_batches:
            raise ThroughputQualificationError(
                "refill admission is consumed more than once"
            )
        consumed_jobs.add(job_id)
        consumed_batches.add(batch_id)
        for path_field, hash_field in (
            ("batch_manifest", "batch_manifest_sha256"),
            ("sbatch_path", "sbatch_sha256"),
            ("spooled_receipt_path", "spooled_receipt_sha256"),
        ):
            artifact_path = Path(str(binding[path_field]))
            try:
                lexical, raw = dispatch_sweeps._stable_readonly_artifact(  # noqa: SLF001
                    artifact_path,
                    description=(
                        f"refill admission {batch_id} {path_field}"
                    ),
                )
            except (dispatch_sweeps.DispatcherError, OSError) as exc:
                raise ThroughputQualificationError(
                    f"refill admission artifact cannot be verified: {exc}"
                ) from exc
            if (
                lexical != artifact_path
                or _sha256_bytes(raw) != binding[hash_field]
            ):
                raise ThroughputQualificationError(
                    f"refill admission {path_field} bytes drifted"
                )
        try:
            spool_receipt = json.loads(
                Path(binding["spooled_receipt_path"]).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ThroughputQualificationError(
                f"refill admission spool receipt is invalid: {exc}"
            ) from exc
        if (
            not isinstance(spool_receipt, Mapping)
            or spool_receipt.get("batch_id") != batch_id
            or str(spool_receipt.get("job_id", "")) != job_id
            or spool_receipt.get("sbatch_path") != binding["sbatch_path"]
            or spool_receipt.get("sbatch_sha256")
            != binding["sbatch_sha256"]
            or spool_receipt.get("spooled_sbatch_sha256")
            != binding["spooled_sbatch_sha256"]
        ):
            raise ThroughputQualificationError(
                "refill admission spool receipt provenance drifted"
            )
    baseline_job_ids = set(records[0]["scheduler"]["qualification_job_ids"])
    final_accepted_jobs = {
        str(job_id)
        for ledger in ledgers.values()
        for job_id, job in ledger["jobs"].items()
        if isinstance(job, Mapping)
        and str(job_id) not in baseline_job_ids
        and (
            isinstance(ledger["intents"].get(str(job["batch_id"])), Mapping)
            and ledger["intents"][str(job["batch_id"])].get("state")
            in {"submitted", "reconciled"}
        )
    }
    if consumed_jobs != final_accepted_jobs:
        raise ThroughputQualificationError(
            "refill journal does not consume every post-baseline accepted "
            "admission exactly once"
        )
    return {
        "consumed_jobs": sorted(consumed_jobs),
        "consumed_batches": sorted(consumed_batches),
        "final_ledger_sha256": ledger_hashes,
    }


def _evidence_summary(
    *,
    intent: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    root = observations[0]["receipt_path"].parent.parent
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
    refill_inventory = _refill_evidence_inventory(root, intent=intent)
    window_path = observations[0]["receipt_path"].parent.parent / (
        LOAD_WINDOW_INTENT_NAME
    )
    drain_path = observations[0]["receipt_path"].parent.parent / (
        LOAD_WINDOW_DRAIN_NAME
    )
    end_path = observations[0]["receipt_path"].parent.parent / (
        LOAD_WINDOW_END_INTENT_NAME
    )
    window = _read_json(
        window_path,
        description="load-window intent",
        sealed=True,
    )
    drain = _read_json(
        drain_path,
        description="load-window graceful drain",
        sealed=True,
    )
    end_intent = _read_json(
        end_path,
        description="load-window end-before-drain intent",
        sealed=True,
    )
    cycle_run_roots = _cycle_run_root_inventory(
        observations[0]["receipt_path"].parent.parent,
        evaluation["load_execution"]["cycle_inventory"],
        require_read_only=True,
    )
    return _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": EVIDENCE_PROTOCOL,
            "intent_id": intent["intent_id"],
            "plan_id": intent["plan_id"],
            "run_id": QUALIFICATION_RUN_ID,
            "estimand_excluded": True,
            "primary_analysis_eligible": False,
            "unique_design": dict(evaluation["unique_design"]),
            "load_execution": dict(evaluation["load_execution"]),
            "load_window_intent": {
                "path": str(window_path.resolve()),
                "sha256": _sha256_file(window_path),
                "window_intent_id": window["window_intent_id"],
            },
            "load_window_drain": {
                "path": str(drain_path.resolve()),
                "sha256": _sha256_file(drain_path),
                "drain_id": drain["drain_id"],
            },
            "load_window_end_intent": {
                "path": str(end_path.resolve()),
                "sha256": _sha256_file(end_path),
                "end_intent_id": end_intent["end_intent_id"],
            },
            "cycle_run_roots": cycle_run_roots,
            "refill_reconciliations": refill_inventory,
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

    evidence_path = context.qualification_root / EVIDENCE_NAME
    evidence = _read_json(
        evidence_path,
        description="qualification aggregate evidence",
        sealed=True,
    )
    _verify_identity(
        evidence,
        "evidence_id",
        description="qualification aggregate evidence",
    )
    marker = {
        "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
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
        "evidence": {
            "path": str(evidence_path.resolve()),
            "sha256": _sha256_file(evidence_path),
            "evidence_id": evidence["evidence_id"],
        },
        "cells": CELL_COUNT,
        "qids": TOTAL_QIDS,
        "unique_design": dict(evaluation["unique_design"]),
        "load_execution": dict(evaluation["load_execution"]),
        "ceilings": list(CEILINGS),
        "health_soak_384_seconds": evaluation[
            "health_soak_384_seconds"
        ],
        "loaded_384_seconds": evaluation["loaded_384_seconds"],
        "loaded_384_useful_qids": evaluation["loaded_384_useful_qids"],
        "loaded_384_observation_count": evaluation[
            "loaded_384_observation_count"
        ],
        "configured_client_ceiling": evaluation[
            "configured_client_ceiling"
        ],
        "certified_saturation_target": evaluation[
            "certified_saturation_target"
        ],
        "certified_saturation_target_cuts": evaluation[
            "certified_saturation_target_cuts"
        ],
        "throughput_qids_per_day": evaluation["throughput_qids_per_day"],
        "throughput_unit": evaluation["throughput_unit"],
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
    publish_load_window_end_intent(
        context.qualification_root,
        intent=intent,
        observations=observations,
    )
    publish_load_window_drain(
        context.qualification_root,
        intent=intent,
        observations=observations,
        evaluation=evaluation,
    )
    cycles = load_cycle_inventory(context, intent=intent)
    expected_cycle_ids = [
        record["cycle_id"]
        for record in evaluation["load_execution"]["cycle_inventory"]
    ]
    if [cycle.cycle_id for cycle in cycles] != expected_cycle_ids:
        raise ThroughputQualificationError(
            "load-cycle roots changed before terminal sealing"
        )
    _verify_refill_dispatch_ledger_bindings(
        context,
        intent=intent,
        require_read_only=False,
    )
    for cycle in cycles:
        _seal_tree_read_only(
            cycle.run_root,
            description=f"successful load cycle {cycle.cycle_index} run",
        )
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
    # The semantic-reference root is cycle zero and was sealed with every replay.
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
    validate_load_window_drain(
        _read_json(
            context.qualification_root / LOAD_WINDOW_DRAIN_NAME,
            description="load-window graceful drain",
            sealed=True,
        ),
        root=context.qualification_root,
        intent=intent,
        observations=observations,
        evaluation=evaluation,
    )
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
    cycles = load_cycle_inventory(context, intent=intent)
    if not cycles:
        raise ThroughputQualificationError(
            "completed qualification lacks sealed load-cycle roots"
        )
    for cycle in cycles:
        _assert_tree_read_only(
            cycle.run_root,
            description=f"successful load cycle {cycle.cycle_index} run",
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
    _verify_refill_dispatch_ledger_bindings(
        context,
        intent=intent,
        require_read_only=True,
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
                "unique_design",
                "load_execution",
                "health_soak_384_seconds",
                "loaded_384_seconds",
                "loaded_384_useful_qids",
                "loaded_384_observation_count",
                "configured_client_ceiling",
                "certified_saturation_target",
                "certified_saturation_target_cuts",
                "throughput_qids_per_day",
                "throughput_unit",
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
    cycle: LoadCycleContext | None = None,
) -> list[str]:
    """Return the one isolated frozen-dispatcher poll command.

    Stage ceilings are enforced by the parent orchestrator before each maximum-24
    microbatch.  The dispatcher retains the protected global 448-minus-64 contract.
    No production run ID or production dispatcher state directory appears here.
    """

    if (
        not isinstance(max_batch, int)
        or isinstance(max_batch, bool)
        or not 0 <= max_batch <= MAX_BATCH
    ):
        raise ThroughputQualificationError(
            f"qualification microbatch must be within 0..{MAX_BATCH}"
        )
    authorize_client_placement(
        context,
        client_partition=client_partition,
        client_qos=client_qos,
    )
    run_id = QUALIFICATION_RUN_ID if cycle is None else cycle.run_id
    run_root = context.run_root if cycle is None else cycle.run_root
    dispatcher_state = (
        context.dispatcher_state
        if cycle is None
        else cycle.dispatcher_state
    )
    execution_authority_path = (
        context.qualification_root / EXECUTION_AUTHORITY_NAME
        if cycle is None
        else cycle.execution_authority_path
    )
    command = [
        str((context.harness_prefix / "bin" / "python").resolve()),
        "-u",
        str((context.release_worktree / "slurm" / "dispatch_sweeps.py").resolve()),
        "dispatch",
        "--run",
        f"{run_id}={run_root}",
        "--server-pool",
        f"{run_id}={context.server_pool_root}",
        "--weight",
        f"{run_id}=1",
        "--results-root",
        str(context.results_root),
        "--state-dir",
        str(dispatcher_state),
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
        str(execution_authority_path),
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
        "health_soak_384_seconds": HEALTH_SOAK_384_SECONDS,
        "loaded_384_contract": {
            "minimum_observations": MIN_LOADED_384_OBSERVATIONS,
            "active_cells": CEILINGS[-1],
            "minimum_unfinished_cells": CEILINGS[-1],
            "requires_trusted_qid_progress": True,
        },
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


def mark_load_cycles_drained(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    cycle_inventory: Sequence[Mapping[str, Any]],
    now: float,
) -> list[dict[str, Any]]:
    """Seal every partial replay after scheduler truth proves zero writers."""

    cycles = load_cycle_inventory(context, intent=intent)
    if len(cycles) != len(cycle_inventory):
        raise ThroughputQualificationError(
            "load-cycle inventory changed at the drain boundary"
        )
    receipts: list[dict[str, Any]] = []
    for cycle, record in zip(cycles, cycle_inventory):
        if (
            record.get("cycle_id") != cycle.cycle_id
            or record.get("cycle_index") != cycle.cycle_index
        ):
            raise ThroughputQualificationError(
                "load-cycle drain inventory identity drifted"
            )
        if record.get("status") == "complete":
            continue
        path = cycle.evidence_root / CYCLE_DRAIN_NAME
        if path.exists() or path.is_symlink():
            observed = _read_json(
                path,
                description=f"load cycle {cycle.cycle_index} graceful drain",
                sealed=True,
            )
            _verify_identity(
                observed,
                "drain_id",
                description=f"load cycle {cycle.cycle_index} graceful drain",
            )
            if (
                observed.get("schema_version")
                != LOAD_ACCOUNTING_SCHEMA_VERSION
                or observed.get("protocol") != CYCLE_DRAIN_PROTOCOL
                or observed.get("qualification_intent_id")
                != intent["intent_id"]
                or observed.get("cycle_id") != cycle.cycle_id
                or observed.get("cycle_index") != cycle.cycle_index
                or observed.get("run_id") != cycle.run_id
                or observed.get("status") != "load_window_drained"
                or observed.get("no_new_admission") is not True
                or observed.get("validated_execution_events")
                != record["validated_execution_events"]
                or observed.get("unfinished_assignments")
                != record["unfinished_assignments"]
                or observed.get("estimand_excluded") is not True
                or observed.get("primary_analysis_eligible") is not False
            ):
                raise ThroughputQualificationError(
                    f"load cycle {cycle.cycle_index} drain replay conflicts"
                )
            receipts.append(observed)
            continue
        payload = _with_identity(
            {
                "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
                "protocol": CYCLE_DRAIN_PROTOCOL,
                "qualification_intent_id": intent["intent_id"],
                "cycle_id": cycle.cycle_id,
                "cycle_index": cycle.cycle_index,
                "run_id": cycle.run_id,
                "status": "load_window_drained",
                "no_new_admission": True,
                "validated_execution_events": record[
                    "validated_execution_events"
                ],
                "unfinished_assignments": record[
                    "unfinished_assignments"
                ],
                "drained_at": _utc(now),
                "drained_timestamp": float(now),
                "estimand_excluded": True,
                "primary_analysis_eligible": False,
            },
            "drain_id",
        )
        _write_once(
            path,
            payload,
            description=f"load cycle {cycle.cycle_index} graceful drain",
        )
        observed = _read_json(
            path,
            description=f"load cycle {cycle.cycle_index} graceful drain",
            sealed=True,
        )
        _verify_identity(
            observed,
            "drain_id",
            description=f"load cycle {cycle.cycle_index} graceful drain",
        )
        if observed != payload:
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} drain replay conflicts"
            )
        receipts.append(observed)
    return receipts


def _scan_cycle_semantics(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    cycle: LoadCycleContext,
    model_contract_path: Path,
) -> dict[str, Any]:
    snapshot = load_manifest(cycle.run_root, verify_frozen=True)
    if snapshot.cells != generate_qualification_cells():
        raise ThroughputQualificationError(
            f"load cycle {cycle.cycle_index} manifest drifted before scan"
        )
    catalog = VerifiedQuestionCatalog(cycle.run_root, snapshot=snapshot)
    states: Counter[str] = Counter()
    progress = {label: 0 for label in expected_stratum_labels()}
    schemas: Counter[str] = Counter()
    validated = 0
    useful = 0
    integrity_incidents = 0
    transport_incidents = 0
    complete_cells = 0
    readiness_generation = intent["readiness_generation"]
    for cell in snapshot.cells:
        questions = catalog.questions_for(cell)
        cell_directory = cycle.run_root / "cells" / cell.cell_id
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
        complete_cells += int(status.status is CompletionState.COMPLETE)
        rows: list[dict[str, Any]] = []
        if status.valid_count:
            rows = list(
                read_canonical_results(
                    cell,
                    cell_directory,
                    expected_qids=tuple(
                        question.qid for question in questions
                    ),
                    expected_questions=questions,
                    verified_benchmark_contracts=catalog.frozen,
                    verified_manifest=snapshot,
                ).records
            )
        validated += len(rows)
        for row in rows:
            schemas[str(row.get("schema_version"))] += 1
            accounting = _censor_accounting([row])
            transport = int(
                accounting["n_transport_censored_coordinates"]
            )
            scientific_censors = (
                int(accounting["n_length_censored_questions"])
                + int(accounting["n_protocol_censored_questions"])
                + int(accounting["n_auxiliary_length_censors"])
                + int(accounting["n_auxiliary_protocol_censors"])
            )
            runtime_matches = _row_matches_readiness_generation(
                row, readiness_generation
            )
            trusted = bool(
                row.get("schema_version") == 5
                and runtime_matches
                and transport == 0
                and scientific_censors == 0
            )
            if trusted:
                qid = row.get("qid")
                if not isinstance(qid, str) or not qid:
                    integrity_incidents += 1
                else:
                    useful += 1
                    progress[_stratum_label(_stratum_tuple(cell))] += 1
                    record_load_execution_event(
                        context,
                        intent=intent,
                        cycle=cycle,
                        cell=cell,
                        qid=qid,
                        result_record_sha256=_sha256_bytes(
                            _compact_bytes(row)
                        ),
                    )
            if not runtime_matches or row.get("schema_version") != 5:
                integrity_incidents += 1
            transport_incidents += transport
            integrity_incidents += scientific_censors
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
    events = load_cycle_execution_events(
        context, intent=intent, cycle=cycle
    )
    event_progress = Counter(
        str(event["stratum"]) for event in events
    )
    if dict(sorted(event_progress.items())) != {
        label: count for label, count in progress.items() if count
    }:
        raise ThroughputQualificationError(
            f"load cycle {cycle.cycle_index} event journal differs from "
            "canonical trusted rows"
        )
    drain_path = cycle.evidence_root / CYCLE_DRAIN_NAME
    if complete_cells == CELL_COUNT:
        cycle_status = "complete"
    elif drain_path.exists() or drain_path.is_symlink():
        drain = _read_json(
            drain_path,
            description=f"load cycle {cycle.cycle_index} drain",
            sealed=True,
        )
        _verify_identity(
            drain,
            "drain_id",
            description=f"load cycle {cycle.cycle_index} drain",
        )
        if (
            drain.get("protocol") != CYCLE_DRAIN_PROTOCOL
            or drain.get("cycle_id") != cycle.cycle_id
            or drain.get("status") != "load_window_drained"
            or drain.get("no_new_admission") is not True
            or drain.get("validated_execution_events") != len(events)
            or drain.get("unfinished_assignments")
            != CELL_COUNT - complete_cells
        ):
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} drain is invalid"
            )
        cycle_status = "load_window_drained"
    else:
        cycle_status = "active"
    return {
        "manifest_sha256": snapshot.sha256,
        "states": dict(states),
        "validated_qids": validated,
        "useful_qids": useful,
        "strata_progress": progress,
        "artifact_schema_counts": dict(schemas),
        "integrity_incidents": integrity_incidents,
        "transport_censor_incidents": transport_incidents,
        "event_count": len(events),
        "event_progress": {
            label: int(event_progress.get(label, 0))
            for label in expected_stratum_labels()
        },
        "inventory": {
            "cycle_index": cycle.cycle_index,
            "cycle_id": cycle.cycle_id,
            "run_id": cycle.run_id,
            "semantic_reference": cycle.semantic_reference,
            "status": cycle_status,
            "validated_execution_events": len(events),
            "unfinished_assignments": CELL_COUNT - complete_cells,
            "estimand_excluded": True,
            "primary_analysis_eligible": False,
        },
    }


def _semantic_scan(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    sequence: int,
    captured_timestamp: float,
    model_contract_path: Path,
) -> dict[str, Any]:
    """Collect unique reference semantics plus cycle-scoped load events."""

    cycles = load_cycle_inventory(context, intent=intent)
    if not cycles or not cycles[0].semantic_reference:
        raise ThroughputQualificationError(
            "qualification lacks its initialized semantic-reference cycle"
        )
    scans = [
        _scan_cycle_semantics(
            context,
            intent=intent,
            cycle=cycle,
            model_contract_path=model_contract_path,
        )
        for cycle in cycles
    ]
    reference = scans[0]
    load_progress = {
        label: sum(
            int(scan["event_progress"][label]) for scan in scans
        )
        for label in expected_stratum_labels()
    }
    load_integrity = sum(
        int(scan["integrity_incidents"]) for scan in scans
    )
    load_censors = sum(
        int(scan["transport_censor_incidents"]) for scan in scans
    )
    return make_semantic_evidence(
        intent_id=str(intent["intent_id"]),
        sequence=sequence,
        captured_timestamp=captured_timestamp,
        manifest_sha256=str(reference["manifest_sha256"]),
        states=reference["states"],
        validated_qids=int(reference["validated_qids"]),
        useful_qids=int(reference["useful_qids"]),
        strata_progress=reference["strata_progress"],
        artifact_schema_counts=reference["artifact_schema_counts"],
        integrity_incidents=int(reference["integrity_incidents"]),
        transport_censor_incidents=int(
            reference["transport_censor_incidents"]
        ),
        semantic_reference_cycle=cycles[0].cycle_id,
        trusted_qid_execution_events=sum(
            int(scan["event_count"]) for scan in scans
        ),
        replay_qid_execution_events=sum(
            int(scan["event_count"]) for scan in scans[1:]
        ),
        load_strata_progress=load_progress,
        load_cycle_inventory=[
            scan["inventory"] for scan in scans
        ],
        unfinished_load_assignments=sum(
            int(scan["inventory"]["unfinished_assignments"])
            for scan in scans
        ),
        load_integrity_incidents=load_integrity,
        load_censor_incidents=load_censors,
    )


def _active_task_count(
    *,
    scheduler_jobs: Sequence[Any],
    ledger: Mapping[str, Any],
    captured_timestamp: float,
    allowed_run_ids: Sequence[str] = (QUALIFICATION_RUN_ID,),
) -> tuple[int, list[str], list[str]]:
    jobs = ledger.get("jobs", {})
    intents = ledger.get("intents", {})
    if not isinstance(jobs, Mapping):
        raise ThroughputQualificationError(
            "qualification dispatcher ledger has invalid jobs"
        )
    if not isinstance(intents, Mapping):
        raise ThroughputQualificationError(
            "qualification dispatcher ledger has invalid intents"
        )
    production_ids: set[str] = set()
    for record in jobs.values():
        if not isinstance(record, Mapping):
            raise ThroughputQualificationError(
                "qualification dispatcher job record is malformed"
            )
        for task in record.get("tasks", []):
            run_id = str(task.get("run_id", "")) if isinstance(task, Mapping) else ""
            if run_id not in set(allowed_run_ids):
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
        elif (
            not rows
            and record.get("state")
            in {
                "submitted",
                "active",
                "visibility_grace",
            }
        ):
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
    for batch_id, record in intents.items():
        if (
            not isinstance(record, Mapping)
            or record.get("state") not in {"prepared", "submitting"}
        ):
            continue
        started_at = dispatch_sweeps._intent_visibility_started_at(  # noqa: SLF001
            record
        )
        tasks = record.get("tasks")
        if (
            started_at is None
            or not 0 <= captured_timestamp - started_at < 300.0
            or not isinstance(tasks, list)
            or any(not isinstance(task, Mapping) for task in tasks)
        ):
            continue
        run_ids = {str(task.get("run_id", "")) for task in tasks}
        unexpected = run_ids.difference(allowed_run_ids)
        if unexpected:
            production_ids.update(unexpected)
            continue
        # An intent that has crossed (or may cross) sbatch remains an exact
        # invisible reservation until complete scheduler truth adopts or rejects
        # it.  It must consume capacity but cannot certify a measurement cut.
        active_count += len(tasks)
    return active_count, qualification_job_ids, sorted(production_ids)


def _scheduler_scan(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    sequence: int,
    captured_timestamp: float,
    ceiling: int,
    unfinished_load_assignments: int = CELL_COUNT,
    active_cycle_ids: Sequence[str] | None = None,
    scheduler_reader: Callable[..., Any] = control.query_scheduler,
) -> dict[str, Any]:
    cycles = load_cycle_inventory(context, intent=intent)
    if not cycles:
        raise ThroughputQualificationError(
            "scheduler scan requires an initialized load cycle"
        )
    authorities: dict[int, dispatch_sweeps.QualificationExecutionAuthority] = {}
    authority_catalog: list[dict[str, str]] = []
    for cycle in cycles:
        path = cycle.execution_authority_path
        if not path.exists() and not path.is_symlink():
            if cycle.cycle_index == 0:
                raise ThroughputQualificationError(
                    "semantic-reference execution authority is missing"
                )
            continue
        try:
            authority = (
                dispatch_sweeps.load_qualification_execution_authority(path)
            )
        except dispatch_sweeps.DispatcherError as exc:
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} execution authority failed "
                f"scheduler join: {exc}"
            ) from exc
        if (
            authority.payload["run_id"] != cycle.run_id
            or authority.payload["run_root"] != str(cycle.run_root)
            or authority.payload["intent_id"] != intent["intent_id"]
        ):
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} execution authority drifted"
            )
        authorities[cycle.cycle_index] = authority
        authority_catalog.append(
            {
                "cycle_id": cycle.cycle_id,
                "run_id": cycle.run_id,
                "authority_id": authority.authority_id,
                "authority_sha256": authority.sha256,
            }
        )
    execution_authority = authorities[0]
    try:
        snapshot = scheduler_reader(
            now=captured_timestamp, tolerate_errors=False
        )
    except (control.ControlError, OSError, subprocess.TimeoutExpired) as exc:
        raise ThroughputQualificationError(
            f"qualification scheduler capture failed: {exc}"
        ) from exc
    expected_capacity_authority = {
        "path": str(context.protected_capacity_contract.path),
        "sha256": context.protected_capacity_contract.sha256,
        "marker_id": context.protected_capacity_contract.marker_id,
        "release_git_commit": context.release_git_commit,
        "partition": str(intent["client_partition"]),
        "qos": str(intent["client_qos"]),
        "authorized_cell_slots": CEILINGS[-1],
        "reserve_jobs": QOS_RESERVE,
    }
    aggregate_jobs: dict[str, Any] = {}
    aggregate_intents: dict[str, Any] = {}
    ledger_hashes: list[dict[str, str]] = []
    cycle_ledgers: dict[int, Mapping[str, Any]] = {}
    for cycle in cycles:
        ledger_path = cycle.dispatcher_state / "ledger.json"
        if not ledger_path.exists():
            cycle_ledgers[cycle.cycle_index] = (
                dispatch_sweeps._empty_ledger(captured_timestamp)
            )
            continue
        if cycle.cycle_index not in authorities:
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} ledger lacks its authority"
            )
        ledger = dispatch_sweeps._load_ledger(ledger_path)
        expected_execution = (
            dispatch_sweeps._qualification_execution_authority_binding(  # noqa: SLF001
                authorities[cycle.cycle_index]
            )
        )
        if (
            ledger.get("protected_capacity_authority")
            != expected_capacity_authority
            or ledger.get("qualification_execution_authority")
            != expected_execution
        ):
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} dispatcher ledger authority "
                "drifted"
            )
        jobs_value = ledger.get("jobs")
        intents_value = ledger.get("intents")
        if not isinstance(jobs_value, Mapping) or not isinstance(
            intents_value, Mapping
        ):
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} dispatcher transactions "
                "are malformed"
            )
        collisions = set(aggregate_jobs).intersection(jobs_value)
        if collisions:
            raise ThroughputQualificationError(
                f"load cycle ledgers duplicate scheduler IDs: "
                f"{sorted(collisions)}"
            )
        aggregate_jobs.update(jobs_value)
        intent_collisions = set(aggregate_intents).intersection(
            intents_value
        )
        if intent_collisions:
            raise ThroughputQualificationError(
                f"load cycle ledgers duplicate admission intents: "
                f"{sorted(intent_collisions)}"
            )
        aggregate_intents.update(intents_value)
        cycle_ledgers[cycle.cycle_index] = ledger
        ledger_hashes.append(
            {
                "cycle_id": cycle.cycle_id,
                "path": str(ledger_path.resolve()),
                "sha256": _sha256_file(ledger_path),
            }
        )
    ledger = dispatch_sweeps._empty_ledger(captured_timestamp)
    ledger["jobs"] = aggregate_jobs
    ledger["intents"] = aggregate_intents
    ledger_sha256 = (
        _sha256_bytes(_canonical_bytes(ledger_hashes))
        if ledger_hashes
        else None
    )
    active, job_ids, foreign = _active_task_count(
        scheduler_jobs=snapshot.jobs,
        ledger=ledger,
        captured_timestamp=captured_timestamp,
        allowed_run_ids=[cycle.run_id for cycle in cycles],
    )
    observed_active_cycle_ids: list[str] = []
    for cycle in cycles:
        cycle_ledger = cycle_ledgers[cycle.cycle_index]
        cycle_job_ids = set(str(job_id) for job_id in cycle_ledger["jobs"])
        cycle_rows = [
            job
            for job in snapshot.jobs
            if any(
                job.job_id == job_id
                or job.job_id.startswith(f"{job_id}_")
                for job_id in cycle_job_ids
            )
        ]
        cycle_active, _, _ = _active_task_count(
            scheduler_jobs=cycle_rows,
            ledger=cycle_ledger,
            captured_timestamp=captured_timestamp,
            allowed_run_ids=[cycle.run_id],
        )
        if cycle_active:
            observed_active_cycle_ids.append(cycle.cycle_id)
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
        unfinished_load_assignments=unfinished_load_assignments,
        active_cycle_ids=(
            list(active_cycle_ids)
            if active_cycle_ids is not None
            else observed_active_cycle_ids
        ),
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
        cycle_execution_authorities=authority_catalog,
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
    cycle: LoadCycleContext | None = None,
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
    policy = load_artifact_policy(
        context.run_root if cycle is None else cycle.run_root,
        required=True,
    )
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
    cycle: LoadCycleContext | None = None,
) -> dispatch_sweeps.QualificationExecutionAuthority:
    """Publish and revalidate the marker-first qualification cell authority."""

    effective_environment = (
        _execution_environment(
            context,
            control_value=control_value,
            intent=intent,
            cycle=cycle,
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
    qualification_runner_path = (
        release_worktree
        / "scripts"
        / "run_schema5_throughput_qualification.py"
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
            "qualification_runner_script": str(
                qualification_runner_path
            ),
            "qualification_runner_script_sha256": (
                dispatch_sweeps._sealed_artifact_sha256(  # noqa: SLF001
                    qualification_runner_path
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
    if (
        execution["dispatcher_script_sha256"]
        != context.dispatcher_source_sha256
        or execution["qualification_runner_script_sha256"]
        != context.qualification_runner_source_sha256
    ):
        raise ThroughputQualificationError(
            "qualification execution sources differ from the frozen release "
            "authority used to validate protected capacity"
        )
    run_id = QUALIFICATION_RUN_ID if cycle is None else cycle.run_id
    run_root = context.run_root if cycle is None else cycle.run_root
    payload = _with_identity(
        {
            "schema_version": EXECUTION_AUTHORITY_SCHEMA_VERSION,
            "protocol": EXECUTION_AUTHORITY_PROTOCOL,
            "intent_id": str(intent["intent_id"]),
            "chain_id": context.chain_id,
            "run_id": run_id,
            "run_root": str(run_root),
            "release_git_commit": context.release_git_commit,
            "release_tag_object": context.release_tag_object,
            "source_tree_sha256": context.source_tree_sha256,
            "qualification_runner_source_sha256": (
                context.qualification_runner_source_sha256
            ),
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
    path = (
        context.qualification_root / EXECUTION_AUTHORITY_NAME
        if cycle is None
        else cycle.execution_authority_path
    )
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
        (cycle is None or cycle.cycle_index == 0)
        and intent.get("execution_authority") != expected_intent_authority
    ) or authority.payload != payload:
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
    cycle: LoadCycleContext | None = None,
) -> dict[str, Any]:
    environment = _execution_environment(
        context,
        control_value=control_value,
        intent=intent,
        cycle=cycle,
    )
    create_or_load_execution_authority(
        context,
        intent=intent,
        control_value=control_value,
        environment=environment,
        cycle=cycle,
    )
    argv = dispatcher_command(
        context,
        client_partition=str(intent["client_partition"]),
        client_qos=str(intent["client_qos"]),
        max_batch=max_batch,
        cycle=cycle,
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
    expected_run_id = (
        QUALIFICATION_RUN_ID if cycle is None else cycle.run_id
    )
    if (
        not isinstance(report, Mapping)
        or report.get("dry_run") is not False
        or not isinstance(selected, list)
        or any(
            not isinstance(task, Mapping)
            or task.get("run_id") != expected_run_id
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


def _reconcile_dispatchers_no_admit(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    control_value: Mapping[str, Any],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    """Adopt every ambiguous intent before scheduler foreign-job classification."""

    cycles = load_cycle_inventory(context, intent=intent)
    if not cycles:
        raise ThroughputQualificationError(
            "pre-scan dispatcher reconciliation lacks an initialized cycle"
        )
    records: list[dict[str, Any]] = []
    for cycle in cycles:
        report = _run_dispatcher_once(
            context,
            intent=intent,
            control_value=control_value,
            max_batch=0,
            runner=runner,
            cycle=cycle,
        )
        if (
            report.get("selected") != []
            or report.get("submission") is not None
            or report.get("submission_error") is not None
            or report.get("unmappable_cell_jobs") != []
            or report.get("validation_errors") != []
            or not isinstance(report.get("poll_number"), int)
            or isinstance(report.get("poll_number"), bool)
            or report["poll_number"] < 1
        ):
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} no-admit reconciliation "
                "is incomplete, ambiguous, or attempted admission"
            )
        ledger_path = cycle.dispatcher_state / "ledger.json"
        if not ledger_path.is_file() or ledger_path.is_symlink():
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} reconciliation did not "
                "publish its durable ledger"
            )
        ledger = dispatch_sweeps._load_ledger(ledger_path)  # noqa: SLF001
        intents = ledger.get("intents")
        jobs = ledger.get("jobs")
        if not isinstance(intents, Mapping) or not isinstance(jobs, Mapping):
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} reconciled ledger is malformed"
            )
        ledger_now = ledger.get("updated_at")
        if (
            not isinstance(ledger_now, (int, float))
            or isinstance(ledger_now, bool)
            or not math.isfinite(float(ledger_now))
        ):
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} reconciled ledger lacks "
                "its exact poll timestamp"
            )
        unresolved_reservations: list[dict[str, Any]] = []
        for batch_id, record in intents.items():
            if (
                not isinstance(record, Mapping)
                or record.get("state") not in {"prepared", "submitting"}
            ):
                continue
            started_at = dispatch_sweeps._intent_visibility_started_at(  # noqa: SLF001
                record
            )
            tasks = record.get("tasks")
            if (
                started_at is None
                or float(ledger_now) - started_at >= 300.0
                or not isinstance(tasks, list)
                or not tasks
                or any(
                    not isinstance(task, Mapping)
                    or task.get("run_id") != cycle.run_id
                    for task in tasks
                )
            ):
                raise ThroughputQualificationError(
                    f"load cycle {cycle.cycle_index} retains an ambiguous "
                    f"intent outside exact scheduler visibility grace: "
                    f"{batch_id}"
                )
            unresolved_reservations.append(
                {
                    "cycle_id": cycle.cycle_id,
                    "run_id": cycle.run_id,
                    "ledger_path": str(ledger_path.resolve()),
                    "batch_id": str(batch_id),
                    "intent_state": str(record["state"]),
                    "task_count": len(tasks),
                    "tasks_sha256": _sha256_bytes(
                        _canonical_bytes(tasks)
                    ),
                }
            )
        records.append(
            {
                "cycle_index": cycle.cycle_index,
                "cycle_id": cycle.cycle_id,
                "run_id": cycle.run_id,
                "poll_number": report["poll_number"],
                "ledger_path": str(ledger_path.resolve()),
                "ledger_sha256": _sha256_file(ledger_path),
                "intent_count": len(intents),
                "job_count": len(jobs),
                "selected_tasks": 0,
                "unresolved_intent_reservations": sorted(
                    unresolved_reservations,
                    key=lambda item: item["batch_id"],
                ),
            }
        )
    return {
        "cycles": records,
        "unresolved_intent_reservations": [
            reservation
            for record in records
            for reservation in record["unresolved_intent_reservations"]
        ],
        "aggregate_ledger_sha256": _sha256_bytes(
            _canonical_bytes(
                [
                    {
                        "cycle_id": record["cycle_id"],
                        "path": record["ledger_path"],
                        "sha256": record["ledger_sha256"],
                    }
                    for record in records
                ]
            )
        ),
    }


def _recover_unconsumed_refill_dispatch(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    proposed: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Recover one accepted refill admission after any process-boundary crash.

    The dispatcher ledger is authoritative.  A local return value from ``sbatch``
    is merely a consistency check: it is never the only evidence carried into the
    immutable refill journal.
    """

    records = load_refill_reconciliations(
        context.qualification_root, intent=intent
    )
    unresolved = reconciliation.get("unresolved_intent_reservations")
    cycle_records = reconciliation.get("cycles")
    if not isinstance(unresolved, list) or not isinstance(
        cycle_records, list
    ):
        raise QualificationWindowContinuityError(
            "pre-refill reconciliation lacks durable intent reservations"
        )
    consumed_jobs = {
        str(record["preceding_dispatch"]["job_id"])
        for record in records
        if record["preceding_dispatch"] is not None
    }
    consumed_batches = {
        str(record["preceding_dispatch"]["batch_id"])
        for record in records
        if record["preceding_dispatch"] is not None
    }
    current_job_ids = set(scheduler.get("qualification_job_ids", []))
    prior_job_ids = (
        set()
        if not records
        else set(records[-1]["scheduler"]["qualification_job_ids"])
    )
    cycles = {
        cycle.cycle_id: cycle
        for cycle in load_cycle_inventory(context, intent=intent)
    }
    candidates: list[tuple[LoadCycleContext, Mapping[str, Any], str]] = []
    if records:
        for cycle_record in cycle_records:
            if not isinstance(cycle_record, Mapping):
                raise QualificationWindowContinuityError(
                    "pre-refill cycle reconciliation is malformed"
                )
            cycle_id = str(cycle_record.get("cycle_id", ""))
            cycle = cycles.get(cycle_id)
            ledger_path = Path(str(cycle_record.get("ledger_path", "")))
            if (
                cycle is None
                or cycle_record.get("run_id") != cycle.run_id
                or ledger_path
                != (cycle.dispatcher_state / "ledger.json").resolve()
                or not ledger_path.is_file()
                or ledger_path.is_symlink()
            ):
                raise QualificationWindowContinuityError(
                    "pre-refill reconciliation does not bind an exact cycle "
                    "ledger"
                )
            ledger = dispatch_sweeps._load_ledger(ledger_path)  # noqa: SLF001
            intents = ledger.get("intents")
            jobs = ledger.get("jobs")
            if not isinstance(intents, Mapping) or not isinstance(
                jobs, Mapping
            ):
                raise QualificationWindowContinuityError(
                    "pre-refill cycle ledger is malformed"
                )
            for batch_id, intent_record in intents.items():
                if (
                    not isinstance(intent_record, Mapping)
                    or intent_record.get("state")
                    not in {"submitted", "reconciled"}
                ):
                    continue
                job_id = str(intent_record.get("job_id", ""))
                if (
                    batch_id in consumed_batches
                    or job_id in consumed_jobs
                    or job_id in prior_job_ids
                    or job_id not in current_job_ids
                ):
                    continue
                job = jobs.get(job_id)
                if (
                    not isinstance(job, Mapping)
                    or job.get("batch_id") != batch_id
                ):
                    raise QualificationWindowContinuityError(
                        "accepted refill intent lacks its exact job record"
                    )
                candidates.append((cycle, ledger, job_id))
    if len(candidates) > 1:
        raise QualificationWindowContinuityError(
            "more than one unconsumed admission appeared between refill cuts"
        )
    recovered: dict[str, Any] | None = None
    if candidates:
        cycle, ledger, job_id = candidates[0]
        prior = records[-1]
        job = ledger["jobs"][job_id]
        batch_id = str(job["batch_id"])
        matching_reservation = next(
            (
                reservation
                for reservation in prior[
                    "unresolved_intent_reservations"
                ]
                if reservation["batch_id"] == batch_id
            ),
            None,
        )
        requested = (
            min(MAX_BATCH, int(prior["active_deficit"]))
            if int(prior["active_deficit"]) > 0
            else (
                int(matching_reservation["task_count"])
                if matching_reservation is not None
                else 0
            )
        )
        if requested <= 0:
            raise QualificationWindowContinuityError(
                "accepted unconsumed admission has no preceding deficit or "
                "invisible reservation"
            )
        recovered = _refill_dispatch_from_ledger(
            cycle=cycle,
            ledger=ledger,
            job_id=job_id,
            requested_tasks=requested,
        )
    if proposed is not None and recovered != dict(proposed):
        raise QualificationWindowContinuityError(
            "local refill acceptance differs from reconciled durable ledger"
        )
    normalized_unresolved = [
        dict(record)
        for record in unresolved
        if isinstance(record, Mapping)
    ]
    if len(normalized_unresolved) != len(unresolved):
        raise QualificationWindowContinuityError(
            "pre-refill unresolved reservation inventory is malformed"
        )
    if {
        str(record["batch_id"]) for record in normalized_unresolved
    }.intersection(consumed_batches):
        raise QualificationWindowContinuityError(
            "a consumed admission reappeared as an unresolved reservation"
        )
    return recovered, normalized_unresolved


def _scaling_requirement(
    context: QualificationContext,
) -> dict[str, Any]:
    """Name the highest provenance-bound pressure across every load cycle."""

    intent = validate_intent(
        _read_json(
            context.qualification_root / INTENT_NAME,
            description="qualification intent",
            sealed=True,
        ),
        context=context,
    )
    cycles = load_cycle_inventory(context, intent=intent)
    if not cycles:
        raise ThroughputQualificationError(
            "qualification scaling requires an initialized load cycle"
        )
    histories: list[tuple[LoadCycleContext, Mapping[str, Any]]] = []
    expected_capacity_authority = {
        "path": str(context.protected_capacity_contract.path),
        "sha256": context.protected_capacity_contract.sha256,
        "marker_id": context.protected_capacity_contract.marker_id,
        "release_git_commit": context.release_git_commit,
        "partition": str(intent["client_partition"]),
        "qos": str(intent["client_qos"]),
        "authorized_cell_slots": CEILINGS[-1],
        "reserve_jobs": QOS_RESERVE,
    }
    for cycle in cycles:
        ledger_path = cycle.dispatcher_state / "ledger.json"
        if not ledger_path.exists():
            continue
        try:
            ledger = dispatch_sweeps._load_ledger(ledger_path)  # noqa: SLF001
        except (dispatch_sweeps.DispatcherError, OSError) as exc:
            raise ThroughputQualificationError(
                f"cannot derive qualification scaling requirement: {exc}"
            ) from exc
        authority_path = cycle.execution_authority_path
        authority = _read_json(
            authority_path,
            description=(
                f"load cycle {cycle.cycle_index} execution authority"
            ),
            sealed=True,
        )
        _verify_identity(
            authority,
            "authority_id",
            description=(
                f"load cycle {cycle.cycle_index} execution authority"
            ),
        )
        expected_execution_binding = {
            "path": str(authority_path),
            "sha256": _sha256_file(authority_path),
            "authority_id": authority["authority_id"],
            "intent_id": intent["intent_id"],
            "chain_id": context.chain_id,
            "run_id": cycle.run_id,
            "run_root": str(cycle.run_root),
        }
        observed_execution_binding = ledger.get(
            "qualification_execution_authority"
        )
        if (
            authority.get("schema_version")
            != EXECUTION_AUTHORITY_SCHEMA_VERSION
            or authority.get("protocol") != EXECUTION_AUTHORITY_PROTOCOL
            or authority.get("intent_id") != intent["intent_id"]
            or authority.get("chain_id") != context.chain_id
            or authority.get("run_id") != cycle.run_id
            or authority.get("run_root") != str(cycle.run_root)
            or authority.get("readiness_generation")
            != intent["readiness_generation"]
            or not isinstance(observed_execution_binding, Mapping)
            or any(
                observed_execution_binding.get(field) != expected
                for field, expected in expected_execution_binding.items()
            )
            or ledger.get("protected_capacity_authority")
            != expected_capacity_authority
        ):
            raise ThroughputQualificationError(
                f"load cycle {cycle.cycle_index} pressure ledger provenance "
                "drifted"
            )
        history = ledger.get("qualification_profile_pressure")
        if isinstance(history, Mapping):
            histories.append((cycle, history))
    if not histories:
        raise ThroughputQualificationError(
            "qualification failed before any per-profile backlog pressure was "
            "sealed in its isolated ledger"
        )
    raw_rows = [
        (cycle, row)
        for cycle, history in histories
        for row in history.values()
        if isinstance(row, Mapping)
    ]
    if len(raw_rows) != sum(len(history) for _, history in histories):
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

    # A profile may appear in every replay-cycle ledger.  Collapse exact
    # generation/profile duplicates by their highest observed pressure before
    # selecting the global bottleneck, so replay count cannot bias the result.
    deduplicated: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for cycle, row in raw_rows:
        key = (
            str(cycle.intent["readiness_generation"]["catalog_id"]),
            str(row.get("server_pool_root", "")),
            str(row.get("serving_profile", "")),
        )
        previous = deduplicated.get(key)
        if previous is None or higher(row, previous):
            deduplicated[key] = row
    rows = list(deduplicated.values())
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


def _static_admission_scaling_requirement(
    context: QualificationContext,
    certificate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one additive replica from the signed no-QID wave pressure."""

    binding = _validated_admission_capacity_certificate_binding(
        context,
        certificate_binding,
    )
    if binding["wave_passed"] is not False:
        raise ThroughputQualificationError(
            "static scaling requires a signed admission shortfall"
        )
    certificate = validate_preflight_capacity_certificate(
        _read_json(
            Path(str(binding["path"])),
            description="static-shortfall admission certificate",
            sealed=True,
        )
    )
    summary = certificate.get("wave", {}).get("profile_summary")
    if not isinstance(summary, Mapping) or set(summary) != set(
        SERVING_PROFILES
    ):
        raise ThroughputQualificationError(
            "static-shortfall profile pressure is malformed"
        )
    candidates: list[tuple[str, int, int]] = []
    for profile in sorted(SERVING_PROFILES):
        row = summary.get(profile)
        if not isinstance(row, Mapping):
            raise ThroughputQualificationError(
                "static-shortfall profile pressure row is malformed"
            )
        replicas = row.get("replicas")
        plan_fanout = row.get("plan_fanout")
        selected_fanout = row.get("selected_fanout")
        if (
            not isinstance(replicas, int)
            or isinstance(replicas, bool)
            or replicas <= 0
            or not isinstance(plan_fanout, int)
            or isinstance(plan_fanout, bool)
            or not isinstance(selected_fanout, int)
            or isinstance(selected_fanout, bool)
            or not 0 <= selected_fanout <= plan_fanout
        ):
            raise ThroughputQualificationError(
                "static-shortfall profile pressure values are invalid"
            )
        candidates.append(
            (profile, plan_fanout - selected_fanout, replicas)
        )
    profile, backlog, live_replicas = candidates[0]
    for candidate_profile, candidate_backlog, candidate_replicas in candidates[1:]:
        if (
            candidate_backlog * live_replicas
            > backlog * candidate_replicas
        ):
            profile, backlog, live_replicas = (
                candidate_profile,
                candidate_backlog,
                candidate_replicas,
            )
    tp_size = int(SERVING_PROFILE_REGISTRY[profile].tp_size)
    return {
        "serving_profile": profile,
        "server_pool_root": str(context.server_pool_root),
        "backlog_fanout_work": backlog,
        "live_replicas": live_replicas,
        "backlog_work_per_replica": backlog / live_replicas,
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


_FAILURE_DRAIN_INTENT_FIELDS = {
    "schema_version",
    "protocol",
    "qualification_intent_id",
    "reason",
    "admission_closed",
    "state",
    "requested_at",
    "requested_timestamp",
    "failure_drain_intent_id",
}


def request_failure_drain(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    reason: str,
    now: float,
) -> dict[str, Any]:
    """Fence admission durably before waiting for scientific writers."""

    if (
        not isinstance(now, (int, float))
        or isinstance(now, bool)
        or not math.isfinite(float(now))
        or float(now) <= 0
    ):
        raise ThroughputQualificationError(
            "qualification failure-drain timestamp is invalid"
        )
    path = context.qualification_root / FAILURE_DRAIN_INTENT_NAME
    if path.exists() or path.is_symlink():
        observed = _read_json(
            path,
            description="qualification failure-drain intent",
            sealed=True,
        )
        if set(observed) != _FAILURE_DRAIN_INTENT_FIELDS:
            raise ThroughputQualificationError(
                "qualification failure-drain intent fields drifted"
            )
        _verify_identity(
            observed,
            "failure_drain_intent_id",
            description="qualification failure-drain intent",
        )
        if (
            observed.get("schema_version")
            != LOAD_ACCOUNTING_SCHEMA_VERSION
            or observed.get("protocol") != FAILURE_DRAIN_INTENT_PROTOCOL
            or observed.get("qualification_intent_id")
            != intent["intent_id"]
            or observed.get("reason") != reason
            or observed.get("admission_closed") is not True
            or observed.get("state") != "draining"
            or not isinstance(
                observed.get("requested_timestamp"), (int, float)
            )
            or isinstance(observed.get("requested_timestamp"), bool)
            or not math.isfinite(
                float(observed["requested_timestamp"])
            )
            or float(observed["requested_timestamp"]) <= 0
            or observed.get("requested_at")
            != _utc(float(observed["requested_timestamp"]))
        ):
            raise ThroughputQualificationError(
                "qualification failure-drain replay conflicts"
            )
        return observed
    payload = _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": FAILURE_DRAIN_INTENT_PROTOCOL,
            "qualification_intent_id": intent["intent_id"],
            "reason": reason,
            "admission_closed": True,
            "state": "draining",
            "requested_at": _utc(now),
            "requested_timestamp": float(now),
        },
        "failure_drain_intent_id",
    )
    _write_once(
        path,
        payload,
        description="qualification failure-drain admission fence",
    )
    return request_failure_drain(
        context,
        intent=intent,
        reason=reason,
        now=now,
    )


def _require_terminal_scheduler_quiescence(
    scheduler: Mapping[str, Any],
) -> None:
    """Require complete namespace truth with no mapped or unmapped writer."""

    if (
        scheduler.get("active_qualification_cells") != 0
        or scheduler.get("qualification_tasks_only") is not True
        or scheduler.get("production_run_ids") != []
        or scheduler.get("active_cycle_ids") != []
        or scheduler.get("squeue_complete") is not True
        or scheduler.get("sacct_complete") is not True
        or scheduler.get("errors") != []
    ):
        raise ThroughputQualificationError(
            "terminal failure cannot seal while scientific writers are active "
            "or scheduler namespace truth is incomplete"
        )


def _validate_failure_cycle_drain_state(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    cycle_records: Sequence[Mapping[str, Any]],
) -> list[LoadCycleContext]:
    if any(
        record["status"] not in {"complete", "load_window_drained"}
        for record in cycle_records
    ):
        raise ThroughputQualificationError(
            "terminal failure requires every partial cycle to be drained"
        )
    cycles = load_cycle_inventory(context, intent=intent)
    if [cycle.cycle_id for cycle in cycles] != [
        record["cycle_id"] for record in cycle_records
    ]:
        raise ThroughputQualificationError(
            "terminal failure cycle inventory drifted before sealing"
        )
    for cycle, record in zip(cycles, cycle_records):
        if (
            record.get("cycle_index") != cycle.cycle_index
            or record.get("run_id") != cycle.run_id
            or record.get("semantic_reference")
            is not cycle.semantic_reference
        ):
            raise ThroughputQualificationError(
                f"terminal failure load cycle {cycle.cycle_index} identity "
                "drifted"
            )
        if record["status"] != "load_window_drained":
            continue
        cycle_drain_path = cycle.evidence_root / CYCLE_DRAIN_NAME
        if (
            not cycle_drain_path.is_file()
            or cycle_drain_path.is_symlink()
        ):
            raise ThroughputQualificationError(
                f"terminal failure load cycle {cycle.cycle_index} lacks "
                "its drain receipt"
            )
        drain = _read_json(
            cycle_drain_path,
            description=f"load cycle {cycle.cycle_index} drain receipt",
            sealed=True,
        )
        _verify_identity(
            drain,
            "drain_id",
            description=f"load cycle {cycle.cycle_index} drain receipt",
        )
        if (
            drain.get("protocol") != CYCLE_DRAIN_PROTOCOL
            or drain.get("cycle_id") != cycle.cycle_id
            or drain.get("status") != "load_window_drained"
            or drain.get("validated_execution_events")
            != record["validated_execution_events"]
            or drain.get("unfinished_assignments")
            != record["unfinished_assignments"]
        ):
            raise ThroughputQualificationError(
                f"terminal failure load cycle {cycle.cycle_index} drain "
                "receipt drifted"
            )
    return cycles


def _publish_terminal_failure(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    drain_path = context.qualification_root / FAILURE_DRAIN_INTENT_NAME
    if not drain_path.is_file() or drain_path.is_symlink():
        raise ThroughputQualificationError(
            "terminal failure requires a durable failure-drain admission fence"
        )
    drain_intent = request_failure_drain(
        context,
        intent=intent,
        reason=reason,
        now=1.0,
    )
    observations = load_observations(
        context.qualification_root, intent=intent
    )
    if not observations:
        raise ThroughputQualificationError(
            "terminal failure requires scheduler-authoritative drain evidence"
        )
    final = observations[-1]
    _require_terminal_scheduler_quiescence(final["scheduler"])
    if (
        float(final["receipt"]["captured_timestamp"])
        < float(drain_intent["requested_timestamp"])
    ):
        raise ThroughputQualificationError(
            "terminal failure drain evidence predates its admission fence"
        )
    cycle_records = final["semantic"]["load_cycle_inventory"]
    cycles = _validate_failure_cycle_drain_state(
        context,
        intent=intent,
        cycle_records=cycle_records,
    )
    _verify_refill_dispatch_ledger_bindings(
        context,
        intent=intent,
        require_read_only=False,
    )
    for cycle in cycles:
        _seal_tree_read_only(
            cycle.run_root,
            description=f"failed load cycle {cycle.cycle_index} run",
        )
    cycle_run_roots = _cycle_run_root_inventory(
        context.qualification_root,
        cycle_records,
        require_read_only=True,
    )
    admission_certificate = (
        _validated_admission_capacity_certificate_binding(context)
    )
    scaling = _scaling_requirement(context)
    receipt_path = Path(str(final["receipt_path"])).resolve()
    payload = _with_identity(
        {
            "schema_version": LOAD_ACCOUNTING_SCHEMA_VERSION,
            "protocol": FAILURE_PROTOCOL,
            "passed": False,
            "intent_id": str(intent["intent_id"]),
            "attempt": _completion_attempt_binding(context),
            "readiness_generation": dict(
                intent["readiness_generation"]
            ),
            "reason": reason,
            "admission_capacity_certificate": admission_certificate,
            "additive_scaling_requirement": scaling,
            "scheduler_capacity_mutated": False,
            "rerun_requirement": (
                "publish a fresh serving/capacity and trusted-catalog rollout "
                "generation, then create a fresh qualification namespace and intent; "
                "this failed intent and its observations cannot be reused"
            ),
            "failure_drain_intent": {
                "path": str(drain_path.resolve()),
                "sha256": _sha256_file(drain_path),
                "failure_drain_intent_id": drain_intent[
                    "failure_drain_intent_id"
                ],
                "drain_observation": {
                    "path": str(receipt_path),
                    "sha256": _sha256_file(receipt_path),
                    "observation_id": final["receipt"]["observation_id"],
                },
            },
            "cycle_run_roots": cycle_run_roots,
            "refill_reconciliations": _refill_evidence_inventory(
                context.qualification_root,
                intent=intent,
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
    # Every cycle root, including the semantic reference, was sealed above.
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
    intent = validate_intent(
        _read_json(
            context.qualification_root / INTENT_NAME,
            description="qualification intent",
            sealed=True,
        ),
        context=context,
    )
    observations = load_observations(
        context.qualification_root,
        intent=intent,
    )
    if not observations:
        raise ThroughputQualificationError(
            "terminal qualification failure lacks its drain observation"
        )
    final = observations[-1]
    _require_terminal_scheduler_quiescence(final["scheduler"])
    cycle_records = final["semantic"]["load_cycle_inventory"]
    _validate_failure_cycle_drain_state(
        context,
        intent=intent,
        cycle_records=cycle_records,
    )
    _verify_refill_dispatch_ledger_bindings(
        context,
        intent=intent,
        require_read_only=not bool(
            stat.S_IMODE(context.qualification_root.stat().st_mode)
            & 0o222
        ),
    )
    scaling = failure.get("additive_scaling_requirement")
    if not isinstance(scaling, Mapping):
        raise ThroughputQualificationError(
            "terminal qualification failure marker is malformed"
        )
    raw_admission_certificate = failure.get(
        "admission_capacity_certificate"
    )
    try:
        admission_certificate = (
            _validated_admission_capacity_certificate_binding(
                context,
                (
                    raw_admission_certificate
                    if isinstance(raw_admission_certificate, Mapping)
                    else None
                ),
            )
        )
    except ThroughputQualificationError:
        admission_certificate = None
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
    drain_binding = failure.get("failure_drain_intent")
    cycle_roots = failure.get("cycle_run_roots")
    drain_valid = False
    if isinstance(drain_binding, Mapping):
        drain_path = Path(str(drain_binding.get("path", "")))
        drain_observation = drain_binding.get("drain_observation")
        if (
            set(drain_binding)
            == {
                "path",
                "sha256",
                "failure_drain_intent_id",
                "drain_observation",
            }
            and drain_path
            == context.qualification_root / FAILURE_DRAIN_INTENT_NAME
            and drain_path.is_file()
            and not drain_path.is_symlink()
            and drain_binding.get("sha256") == _sha256_file(drain_path)
            and isinstance(drain_observation, Mapping)
            and set(drain_observation)
            == {"path", "sha256", "observation_id"}
        ):
            drain = _read_json(
                drain_path,
                description="qualification failure-drain intent",
                sealed=True,
            )
            _verify_identity(
                drain,
                "failure_drain_intent_id",
                description="qualification failure-drain intent",
            )
            drain_valid = bool(
                drain.get("protocol") == FAILURE_DRAIN_INTENT_PROTOCOL
                and drain.get("qualification_intent_id")
                == intent["intent_id"]
                and drain.get("reason") == failure.get("reason")
                and drain.get("admission_closed") is True
                and drain.get("state") == "draining"
                and isinstance(
                    drain.get("requested_timestamp"), (int, float)
                )
                and not isinstance(
                    drain.get("requested_timestamp"), bool
                )
                and math.isfinite(
                    float(drain["requested_timestamp"])
                )
                and float(drain["requested_timestamp"]) > 0
                and drain.get("requested_at")
                == _utc(float(drain["requested_timestamp"]))
                and float(final["receipt"]["captured_timestamp"])
                >= float(drain["requested_timestamp"])
                and drain_binding.get("failure_drain_intent_id")
                == drain.get("failure_drain_intent_id")
                and drain_observation
                == {
                    "path": str(final["receipt_path"].resolve()),
                    "sha256": _sha256_file(final["receipt_path"]),
                    "observation_id": final["receipt"][
                        "observation_id"
                    ],
                }
            )
    try:
        expected_cycle_roots = _cycle_run_root_inventory(
            context.qualification_root,
            cycle_records,
            require_read_only=True,
        )
    except ThroughputQualificationError:
        expected_cycle_roots = None
    cycle_roots_valid = (
        isinstance(cycle_roots, list)
        and expected_cycle_roots is not None
        and cycle_roots == expected_cycle_roots
    )
    try:
        expected_refill_reconciliations = _refill_evidence_inventory(
            context.qualification_root,
            intent=intent,
        )
    except ThroughputQualificationError:
        expected_refill_reconciliations = None
    try:
        expected_scaling = _scaling_requirement(context)
    except ThroughputQualificationError:
        expected_scaling = None
    if (
        set(scaling) != _SCALING_REQUIREMENT_FIELDS
        or failure.get("schema_version")
        != LOAD_ACCOUNTING_SCHEMA_VERSION
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
        or admission_certificate is None
        or failure.get("admission_capacity_certificate")
        != admission_certificate
        or not isinstance(failure.get("rerun_requirement"), str)
        or "fresh qualification namespace and intent"
        not in str(failure["rerun_requirement"])
        or failure.get("scheduler_capacity_mutated") is not False
        or not drain_valid
        or not cycle_roots_valid
        or expected_refill_reconciliations is None
        or failure.get("refill_reconciliations")
        != expected_refill_reconciliations
        or expected_scaling is None
        or dict(scaling) != expected_scaling
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
    return _format_terminal_failure_message(failure)


def _format_terminal_failure_message(
    failure: Mapping[str, Any],
) -> str:
    scaling = failure["additive_scaling_requirement"]
    return (
        f"{failure.get('reason')}; bottleneck profile "
        f"{scaling.get('serving_profile')} had the largest observed "
        "backlog-work-per-replica; additive requirement: "
        f"{scaling.get('requirement')}. No scheduler capacity was mutated. "
        f"{failure.get('rerun_requirement')}"
    )


def _raise_terminal_failure(context: QualificationContext) -> None:
    """Replay one sealed failure with its exact scheduler exit disposition."""

    failure = _load_terminal_failure(context)
    message = _format_terminal_failure_message(failure)
    if _is_capacity_shortfall_reason(str(failure["reason"])):
        raise QualificationCapacityTransitionRequired(message)
    raise ThroughputQualificationError(message)


def _publish_and_raise_terminal_failure(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    reason: str,
) -> None:
    """Seal a drained failure, then use the same disposition as replay."""

    _publish_terminal_failure(
        context,
        intent=intent,
        reason=reason,
    )
    _raise_terminal_failure(context)


def _capacity_targets(intent: Mapping[str, Any]) -> tuple[int, int]:
    """Return the configured client ceiling and signed fleet saturation cut."""

    ceiling = intent.get("configured_client_ceiling")
    saturation = intent.get("certified_saturation_target")
    certificate = intent.get("admission_capacity_certificate")
    if (
        not isinstance(ceiling, int)
        or isinstance(ceiling, bool)
        or ceiling != CEILINGS[-1]
        or not isinstance(saturation, int)
        or isinstance(saturation, bool)
        or not 0 < saturation <= ceiling
        or not isinstance(certificate, Mapping)
        or certificate.get("target_cell_count") != ceiling
        or certificate.get("selected_cell_count") != saturation
    ):
        raise ThroughputQualificationError(
            "qualification capacity ceiling/saturation binding is invalid"
        )
    return ceiling, saturation


def _phase_active_target(ceiling: int, saturation_target: int) -> int:
    if ceiling not in CEILINGS or not 0 < saturation_target <= CEILINGS[-1]:
        raise ThroughputQualificationError(
            "qualification phase capacity target is invalid"
        )
    return min(ceiling, saturation_target)


def _next_ceiling(
    observations: Sequence[Mapping[str, Any]],
    *,
    saturation_target: int = CEILINGS[-1],
) -> int:
    peaks = {ceiling: 0 for ceiling in CEILINGS}
    for observation in observations:
        scheduler = observation["scheduler"]
        peaks[int(scheduler["ceiling"])] = max(
            peaks[int(scheduler["ceiling"])],
            int(scheduler["active_qualification_cells"]),
        )
    for ceiling in CEILINGS:
        if peaks[ceiling] != _phase_active_target(
            ceiling, saturation_target
        ):
            return ceiling
    return CEILINGS[-1]


def _stage_dispatch_batch(
    *,
    ceiling: int,
    active: int,
    useful_qids: int,
    saturation_target: int = CEILINGS[-1],
    admission_closed: bool = False,
) -> int:
    """Return the exact next microbatch without crossing the isolated stage."""

    if (
        ceiling not in CEILINGS
        or not isinstance(active, int)
        or isinstance(active, bool)
        or not 0 <= active <= _phase_active_target(
            ceiling, saturation_target
        )
        or not isinstance(useful_qids, int)
        or isinstance(useful_qids, bool)
        or not 0 <= useful_qids <= TOTAL_QIDS
    ):
        raise ThroughputQualificationError(
            "qualification stage admission inputs are invalid"
        )
    if admission_closed:
        return 0
    return min(
        MAX_BATCH,
        _phase_active_target(ceiling, saturation_target) - active,
    )


def _load_window_refill_permitted(root: Path) -> bool:
    """Return whether exact-load refill admission remains durably authorized."""

    window = root / LOAD_WINDOW_INTENT_NAME
    if window.is_symlink():
        raise ThroughputQualificationError(
            "load-window intent cannot be symlinked"
        )
    if not window.is_file():
        return False
    for name in (LOAD_WINDOW_END_INTENT_NAME, FAILURE_DRAIN_INTENT_NAME):
        fence = root / name
        if fence.is_symlink():
            raise ThroughputQualificationError(
                f"qualification admission fence cannot be symlinked: {fence}"
            )
        if fence.exists():
            return False
    return True


def _ensure_load_backlog(
    context: QualificationContext,
    *,
    intent: Mapping[str, Any],
    control_value: Mapping[str, Any],
    semantic: Mapping[str, Any],
    now: float,
    saturation_target: int = CEILINGS[-1],
) -> list[LoadCycleContext]:
    """Keep one full future wave available without mutating a prior cycle."""

    cycles = load_cycle_inventory(context, intent=intent)
    unfinished = int(semantic["unfinished_load_assignments"])
    # Two signed saturation cuts keep a complete future refill wave available.
    while unfinished < 2 * saturation_target:
        cycle = initialize_load_cycle(
            context,
            intent=intent,
            control_value=control_value,
            cycle_index=len(cycles),
            now=now,
        )
        cycles.append(cycle)
        unfinished += CELL_COUNT
    return cycles


def _select_dispatch_cycle(
    cycles: Sequence[LoadCycleContext],
    semantic: Mapping[str, Any],
) -> LoadCycleContext:
    by_id = {cycle.cycle_id: cycle for cycle in cycles}
    candidates = [
        record
        for record in semantic["load_cycle_inventory"]
        if record["status"] == "active"
        and int(record["unfinished_assignments"]) > 0
    ]
    if not candidates:
        # A newly initialized cycle is not represented until the next semantic
        # observation.  It is nevertheless safe to dispatch because its sealed
        # manifest/intent has already committed.
        represented = {
            record["cycle_id"]
            for record in semantic["load_cycle_inventory"]
        }
        new_cycles = [
            cycle for cycle in cycles if cycle.cycle_id not in represented
        ]
        if new_cycles:
            return new_cycles[0]
        raise ThroughputQualificationError(
            "qualification has no unfinished initialized load cycle"
        )
    selected = min(
        candidates,
        key=lambda record: (
            int(record["validated_execution_events"]),
            int(record["cycle_index"]),
        ),
    )
    return by_id[str(selected["cycle_id"])]


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
    capacity_certificate_loader: Callable[..., Mapping[str, Any]] = (
        verify_authorized_preflight_capacity
    ),
) -> dict[str, Any]:
    """Execute the isolated restartable qualification transaction."""

    if (
        not math.isfinite(float(poll_seconds))
        or not 0 < float(poll_seconds) <= MAX_OBSERVATION_GAP_SECONDS
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) < HEALTH_SOAK_384_SECONDS
    ):
        raise ThroughputQualificationError(
            "poll/timeout bounds cannot prove the two-hour ceiling-384 "
            "health soak"
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
        if capacity_certificate_loader is verify_authorized_preflight_capacity:
            base_context = _with_effective_protected_capacity(
                base_context,
                control_value,
            )
        current_attempt_path = (
            base_context.qualification_base / CURRENT_ATTEMPT_NAME
        )
        if (
            current_attempt_path.exists()
            or current_attempt_path.is_symlink()
        ):
            replay_context = load_current_attempt_context(
                base_context,
                require_success=False,
            )
            replay_failure_path = (
                replay_context.qualification_root / FAILURE_NAME
            )
            if (
                replay_failure_path.exists()
                or replay_failure_path.is_symlink()
            ) and (
                replay_context.attempt_pointer is not None
                and replay_context.attempt_pointer[
                    "readiness_generation"
                ]
                == readiness_generation
            ):
                # Replaying the same sealed failed generation must not depend
                # on mutable runtime capacity checks and must not attempt to
                # create a successor namespace.
                _raise_terminal_failure(replay_context)
        raw_capacity_certificate = capacity_certificate_loader(
            base_context,
            control_value=control_value,
            readiness_generation=readiness_generation,
        )
        context = create_or_load_attempt_context(
            base_context,
            control_value=control_value,
            readiness_generation=readiness_generation,
            now=clock(),
        )
        admission_certificate: dict[str, Any] | None = None
        if (
            isinstance(raw_capacity_certificate, Mapping)
            and _ADMISSION_CAPACITY_CERTIFICATE_FIELDS.issubset(
                raw_capacity_certificate
            )
        ):
            admission_certificate = (
                _validated_admission_capacity_certificate_binding(
                    context,
                    {
                        field: raw_capacity_certificate[field]
                        for field in _ADMISSION_CAPACITY_CERTIFICATE_FIELDS
                    },
                )
            )
        if admission_certificate is None:
            raise ThroughputQualificationError(
                "qualification lacks its exact signed fleet saturation target"
            )
        failure_path = context.qualification_root / FAILURE_NAME
        if failure_path.exists() or failure_path.is_symlink():
            # A restart of a sealed failed attempt is a read-only replay.  In
            # particular, it must preserve the original scheduler disposition
            # instead of falling through into initialization under a sealed root.
            _raise_terminal_failure(context)
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
            admission_capacity_certificate=admission_certificate,
            now=clock(),
        )
        if intent["control_guard"] != guard:
            raise ThroughputQualificationError(
                "production admission/ramp state changed after qualification intent"
            )
        configured_client_ceiling, saturation_target = _capacity_targets(
            intent
        )
        initialize_qualification_run(
            context,
            intent=intent,
            control_value=control_value,
            benchmark_loader=benchmark_loader,
        )
        initialize_load_cycle(
            context,
            intent=intent,
            control_value=control_value,
            cycle_index=0,
            now=clock(),
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

        def scan(
            ceiling: int,
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            """Capture one complete scheduler/semantic cut without publishing it."""

            reconciliation = _reconcile_dispatchers_no_admit(
                context,
                intent=intent,
                control_value=control_value,
                runner=runner,
            )
            sequence = len(observations)
            captured = clock()
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
            scheduler = _scheduler_scan(
                context,
                intent=intent,
                sequence=sequence,
                captured_timestamp=captured,
                ceiling=ceiling,
                unfinished_load_assignments=int(
                    semantic["unfinished_load_assignments"]
                ),
                scheduler_reader=scheduler_reader,
            )
            if (
                scheduler.get("dispatcher_ledger_sha256")
                != reconciliation["aggregate_ledger_sha256"]
            ):
                raise ThroughputQualificationError(
                    "scheduler scan is not bound to its immediately preceding "
                    "no-admit dispatcher reconciliation"
                )
            return scheduler, semantic, reconciliation

        def commit_scan(
            scheduler: Mapping[str, Any],
            semantic: Mapping[str, Any],
        ) -> None:
            nonlocal observations
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

        def capture(ceiling: int) -> None:
            scheduler, semantic, _ = scan(ceiling)
            commit_scan(scheduler, semantic)

        def refill_and_capture_load_window(
            current_control: Mapping[str, Any],
        ) -> None:
            """Refill and commit one signed fleet-saturation cut at ceiling 384."""

            if not observations:
                raise QualificationWindowContinuityError(
                    "load-window refill lacks a prior committed observation"
                )
            last_cut_timestamp = float(
                observations[-1]["receipt"]["captured_timestamp"]
            )
            preceding_dispatch: Mapping[str, Any] | None = None
            for _ in range(
                2 * math.ceil(saturation_target / MAX_BATCH)
            ):
                scheduler, semantic, reconciliation = scan(
                    configured_client_ceiling
                )
                recovered_dispatch, unresolved_reservations = (
                    _recover_unconsumed_refill_dispatch(
                        context,
                        intent=intent,
                        reconciliation=reconciliation,
                        scheduler=scheduler,
                        proposed=preceding_dispatch,
                    )
                )
                refill = record_refill_reconciliation(
                    context.qualification_root,
                    intent=intent,
                    scheduler=scheduler,
                    semantic=semantic,
                    preceding_dispatch=recovered_dispatch,
                    unresolved_intent_reservations=(
                        unresolved_reservations
                    ),
                )
                preceding_dispatch = None
                elapsed = (
                    float(refill["captured_timestamp"])
                    - last_cut_timestamp
                )
                if elapsed > MAX_OBSERVATION_GAP_SECONDS:
                    raise QualificationWindowContinuityError(
                        f"refill convergence exceeded the "
                        f"{MAX_OBSERVATION_GAP_SECONDS}-second evidence cadence"
                    )
                active = int(
                    scheduler["active_qualification_cells"]
                )
                if refill["measurement_eligible"] is True:
                    commit_scan(scheduler, semantic)
                    return
                if unresolved_reservations:
                    if active > saturation_target:
                        raise QualificationWindowContinuityError(
                            "invisible intent reservations exceed the "
                            "capacity target"
                        )
                    sleeper(float(poll_seconds))
                    continue
                if (
                    scheduler["ceiling"] != configured_client_ceiling
                    or scheduler["qualification_tasks_only"] is not True
                    or scheduler["production_run_ids"] != []
                    or scheduler["squeue_complete"] is not True
                    or scheduler["sacct_complete"] is not True
                    or scheduler["errors"] != []
                    or active >= saturation_target
                ):
                    raise QualificationWindowContinuityError(
                        "refill scan is ambiguous, foreign, or exceeds the "
                        "exact-384 ceiling"
                    )
                cycles = _ensure_load_backlog(
                    context,
                    intent=intent,
                    control_value=current_control,
                    semantic=semantic,
                    saturation_target=saturation_target,
                    now=clock(),
                )
                requested = _stage_dispatch_batch(
                    ceiling=configured_client_ceiling,
                    active=active,
                    useful_qids=int(semantic["useful_qids"]),
                    saturation_target=saturation_target,
                )
                if requested <= 0:
                    raise QualificationWindowContinuityError(
                        "refill scan has a deficit but no safe admission room"
                    )
                dispatch_cycle = _select_dispatch_cycle(
                    cycles, semantic
                )
                report = _run_dispatcher_once(
                    context,
                    intent=intent,
                    control_value=current_control,
                    max_batch=requested,
                    runner=runner,
                    cycle=dispatch_cycle,
                )
                preceding_dispatch = _refill_dispatch_binding(
                    report,
                    requested_tasks=requested,
                    cycle=dispatch_cycle,
                )
            raise QualificationWindowContinuityError(
                "refill did not converge to the signed saturation target "
                "within bounded transactional microbatches"
            )

        if not observations:
            # This sealed zero baseline is an executable precondition for the first
            # submission.  A runner cannot reach the dispatcher before it exists.
            capture(24)
        elif _load_window_refill_permitted(
            context.qualification_root
        ):
            # A process may have died after the dispatcher crossed sbatch but before
            # the next observation.  Preserve every reconciliation scan in the refill
            # journal, restore the exact capacity target, then publish the cut.
            try:
                refill_and_capture_load_window(control_value)
            except QualificationWindowContinuityError as exc:
                request_failure_drain(
                    context,
                    intent=intent,
                    reason=f"load-window continuity failure: {exc}",
                    now=clock(),
                )
        else:
            # Reconcile scheduler+ledger truth on every restart before computing
            # stage room; never admit from a stale receipt.
            capture(
                _next_ceiling(
                    observations,
                    saturation_target=saturation_target,
                )
            )
        while True:
            failure_drain_path = (
                context.qualification_root / FAILURE_DRAIN_INTENT_NAME
            )
            if not failure_drain_path.exists():
                try:
                    marker = publish_completion(context, intent=intent)
                except QualificationCapacityTransitionRequired as exc:
                    request_failure_drain(
                        context,
                        intent=intent,
                        reason=str(exc),
                        now=clock(),
                    )
                except ThroughputQualificationError as exc:
                    incomplete_markers = (
                        "load-window end cannot be committed",
                        "lacks its immutable load-window intent",
                        "does not exercise every ceiling",
                        "did not exercise every ceiling",
                        "did not reconcile to every exact ceiling",
                        "load window covers only",
                        "semantic reference cycle is not",
                        "is not semantically complete",
                        "final qualification observation is not quiescent",
                        "replay cycles were not gracefully",
                    )
                    if not any(
                        marker in str(exc)
                        for marker in incomplete_markers
                    ):
                        raise
                else:
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

            if (
                not failure_drain_path.exists()
                and clock() - started > timeout_seconds
            ):
                request_failure_drain(
                    context,
                    intent=intent,
                    reason=(
                        "throughput qualification exceeded its fixed "
                        "execution timeout"
                    ),
                    now=clock(),
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
            target = _next_ceiling(
                observations,
                saturation_target=saturation_target,
            )
            latest = observations[-1]
            active = int(
                latest["scheduler"]["active_qualification_cells"]
            )
            useful = int(latest["semantic"]["useful_qids"])
            if failure_drain_path.exists() or failure_drain_path.is_symlink():
                drain_intent = _read_json(
                    failure_drain_path,
                    description="qualification failure-drain intent",
                    sealed=True,
                )
                reason = str(drain_intent["reason"])
                # Admission is already fenced.  Scheduler truth, not elapsed
                # wall time, controls the transition to terminal sealing.
                if active > 0:
                    sleeper(float(poll_seconds))
                    capture(
                        _next_ceiling(
                            observations,
                            saturation_target=saturation_target,
                        )
                    )
                    continue
                if any(
                    record["status"] not in {
                        "complete",
                        "load_window_drained",
                    }
                    for record in latest["semantic"][
                        "load_cycle_inventory"
                    ]
                ):
                    mark_load_cycles_drained(
                        context,
                        intent=intent,
                        cycle_inventory=latest["semantic"][
                            "load_cycle_inventory"
                        ],
                        now=clock(),
                    )
                    capture(
                        _next_ceiling(
                            observations,
                            saturation_target=saturation_target,
                        )
                    )
                    continue
                _publish_and_raise_terminal_failure(
                    context,
                    intent=intent,
                    reason=reason,
                )
            end_path = (
                context.qualification_root
                / LOAD_WINDOW_END_INTENT_NAME
            )
            reference_complete = bool(
                latest["semantic"]["states"]
                == {"complete": CELL_COUNT}
                and useful == TOTAL_QIDS
                and latest["semantic"]["validated_qids"] == TOTAL_QIDS
            )
            if (
                not end_path.exists()
                and load_window_ready(observations)
                and reference_complete
            ):
                publish_load_window_end_intent(
                    context.qualification_root,
                    intent=intent,
                    observations=observations,
                )
            if end_path.exists() or end_path.is_symlink():
                # The immutable end intent is the admission fence across process
                # death.  Never invoke a dispatcher after it appears.
                if active > 0:
                    sleeper(float(poll_seconds))
                    capture(configured_client_ceiling)
                    continue
                if any(
                    record["status"] not in {
                        "complete",
                        "load_window_drained",
                    }
                    for record in latest["semantic"][
                        "load_cycle_inventory"
                    ]
                ):
                    mark_load_cycles_drained(
                        context,
                        intent=intent,
                        cycle_inventory=latest["semantic"][
                            "load_cycle_inventory"
                        ],
                        now=clock(),
                    )
                    capture(configured_client_ceiling)
                    continue
                # The next loop publishes the aggregate drain, evidence, and
                # completion marker without any scientific admission.
                continue
            cycles = _ensure_load_backlog(
                context,
                intent=intent,
                control_value=current_control,
                semantic=latest["semantic"],
                saturation_target=saturation_target,
                now=clock(),
            )
            # One existing dispatcher poll can submit at most 24 tasks.  Clamp that
            # existing interface to the exact remaining stage room before crossing
            # its submission boundary, without modifying production control.
            next_batch = _stage_dispatch_batch(
                ceiling=target,
                active=active,
                useful_qids=useful,
                saturation_target=saturation_target,
            )
            if next_batch > 0:
                dispatch_cycle = _select_dispatch_cycle(
                    cycles, latest["semantic"]
                )
                dispatch_report = _run_dispatcher_once(
                    context,
                    intent=intent,
                    control_value=current_control,
                    max_batch=next_batch,
                    runner=runner,
                    cycle=dispatch_cycle,
                )
                # Join the accepted numeric array through the isolated ledger
                # immediately.  The ledger's bounded scheduler-visibility reservation
                # proves the exact submitted task count even if sub-second cells finish
                # before squeue first exposes them.
                capture(target)
                target = _next_ceiling(
                    observations,
                    saturation_target=saturation_target,
                )
                latest_after_dispatch = observations[-1]
                if (
                    dispatch_report.get("selected")
                    and int(
                        latest_after_dispatch["scheduler"][
                            "active_qualification_cells"
                        ]
                    )
                    < _phase_active_target(target, saturation_target)
                ):
                    # Fill the remainder of a ramp with back-to-back <=24 arrays.
                    # Sleeping here would let fast cells vanish before ceilings 96,
                    # 192, or the signed ceiling-384 saturation cut could ever
                    # be durably reconciled.
                    continue
            sleeper(float(poll_seconds))
            if _load_window_refill_permitted(
                context.qualification_root
            ):
                try:
                    refill_and_capture_load_window(current_control)
                except QualificationWindowContinuityError as exc:
                    request_failure_drain(
                        context,
                        intent=intent,
                        reason=f"load-window continuity failure: {exc}",
                        now=clock(),
                    )
            else:
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
    capacity = subparsers.add_parser(
        "preflight-capacity",
        help=(
            "seal the exact fleet-specific WDRR saturation cut beneath the "
            "configured 384-client ceiling before any scientific QID"
        ),
    )
    capacity.add_argument(
        "--base-fleet-contract", required=True, type=Path
    )
    capacity.add_argument(
        "--effective-fleet-contract", required=True, type=Path
    )
    capacity.add_argument(
        "--additive-overlay-contract", required=True, type=Path
    )
    capacity.add_argument(
        "--capacity-generation", required=True, type=int
    )
    capacity.add_argument("--release-git-commit", required=True)
    capacity.add_argument("--source-tree-sha256", required=True)
    capacity.add_argument(
        "--dispatcher-source", required=True, type=Path
    )
    capacity.add_argument(
        "--qualification-runner-source", required=True, type=Path
    )
    capacity.add_argument("--output", required=True, type=Path)
    capacity.add_argument("--apply", action="store_true")
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
        elif args.command == "publish-capacity-transition":
            report = publish_capacity_transition_authority(
                args.chain_manifest,
                submission_receipt=args.submission_receipt,
                failed_job_id=args.failed_job_id,
                failed_comment=args.failed_comment,
                apply=args.apply,
            )
        else:
            report = preflight_capacity_report(
                base_fleet_contract=args.base_fleet_contract,
                effective_fleet_contract=args.effective_fleet_contract,
                additive_overlay_contract=args.additive_overlay_contract,
                capacity_generation=args.capacity_generation,
                release_git_commit=args.release_git_commit,
                source_tree_sha256=args.source_tree_sha256,
                dispatcher_source=args.dispatcher_source,
                qualification_runner_source=(
                    args.qualification_runner_source
                ),
                output=args.output,
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
    return 0 if report.get("passed", True) else 75


if __name__ == "__main__":
    raise SystemExit(main())
