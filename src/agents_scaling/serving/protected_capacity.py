"""Immutable protected-placement authority for schema-5 scientific jobs.

The production fleet and cell dispatcher are not allowed to infer safety from a
partition name, from ``--no-requeue``, or from an old readiness report.  This module
loads the marker-last ``PROTECTED_CAPACITY_COMPLETE.json`` contract, binds it to the
exact release, and rechecks every partition and QOS immediately before an ``sbatch``
boundary.

Controllers are deliberately outside this contract: they perform no scientific
requests and have their own successor/watchdog recovery policy.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time
from typing import Any

from agents_scaling.serving import scheduler_safety
from agents_scaling.serving.fleet_contract import EXPECTED_COUNTS
from agents_scaling.serving.profiles import SERVING_PROFILES


SCHEMA_VERSION = 4
PROTOCOL = "schema5-v1.2-r12-protected-capacity-v4"
STATIC_FEASIBILITY_SCHEMA_VERSION = 3
STATIC_FEASIBILITY_PROTOCOL = (
    "schema5-v1.2-r12-throughput-preflight-capacity-certificate-v1"
)
LIVE_CLIENT_CAPACITY_PROTOCOL = (
    "schema5-v1.2-r12-live-protected-client-capacity-v3"
)
TRUSTED_SCIENTIFIC_PROVENANCE_PROTOCOL = (
    "schema5-v1.2-r12-trusted-scientific-job-provenance-v1"
)
CAPACITY_SOURCE = (
    "sealed_protected_canary+partition_inventory+association"
)
RELEASE_ID = "sweep-recovery-schema5-v1.2"
RELEASE_TAG = "sweep-recovery-schema5-v1.2-r12"
CHAIN_NAMESPACE = "schema5-v1.2-r12"
MARKER_FILENAME = "PROTECTED_CAPACITY_COMPLETE.json"
STATIC_FEASIBILITY_FILENAME = "PREFLIGHT_CAPACITY_CERTIFICATE.json"
BASE_LOGICAL_REPLICAS = 22
BASE_ACTIVE_GPUS = 24
PRODUCTION_ACTIVE_GPUS = BASE_ACTIVE_GPUS
RETAINED_WARM_TURNOVER_JOB_ELEMENTS = 3
RETAINED_WARM_TURNOVER_GPUS = 4
PRODUCTION_ATTESTED_GPUS = (
    PRODUCTION_ACTIVE_GPUS + RETAINED_WARM_TURNOVER_GPUS
)
CLIENT_JOB_ELEMENTS = 384
PREQUALIFICATION_SELECTED_CELL_COUNT = 278
PREQUALIFICATION_SHORTFALL_CELL_COUNT = (
    CLIENT_JOB_ELEMENTS - PREQUALIFICATION_SELECTED_CELL_COUNT
)
TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS = 64
PREQUALIFICATION_HELD_NON_CELL_JOB_ELEMENTS = (
    TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS
    - BASE_LOGICAL_REPLICAS
    - RETAINED_WARM_TURNOVER_JOB_ELEMENTS
)
TOTAL_SUBMIT_HEADROOM = CLIENT_JOB_ELEMENTS + TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS
PRODUCTION_SERVING_PROFILES = tuple(sorted(EXPECTED_COUNTS))
MIN_SCIENTIFIC_WALL_SECONDS = 86_400
CLIENT_WALL_SECONDS = 43_200
MAX_TRUSTED_SCHEDULER_AGE_SECONDS = 60.0
MAX_DISPATCHER_LEDGER_AGE_SECONDS = 360.0
_MAX_JOBS_CONSUMING_STATES = frozenset(
    {
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "RESIZING",
        "SUSPENDED",
    }
)
_LIVE_JOB_STATES = _MAX_JOBS_CONSUMING_STATES | {"PENDING"}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}\Z")
_PLACEMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MARKER_FIELDS = {
    "schema_version",
    "protocol",
    "passed",
    "release_id",
    "release_tag",
    "release_git_commit",
    "release_tag_object",
    "source_tree_sha256",
    "dispatcher_source_sha256",
    "qualification_runner_source_sha256",
    "chain_namespace",
    "capacity_generation",
    "base_fleet_contract_path",
    "base_fleet_contract_sha256",
    "effective_fleet_contract_path",
    "effective_fleet_contract_sha256",
    "additive_overlay_contract_path",
    "additive_overlay_contract_sha256",
    "static_feasibility_certificate",
    "static_feasibility_wave_passed",
    "static_feasibility_selected_cell_count",
    "static_feasibility_target_cell_count",
    "static_feasibility_shortfall_cells",
    "static_feasibility_configured_client_ceiling",
    "static_feasibility_certified_saturation_target",
    "base_active_logical_replicas",
    "base_active_gpus",
    "base_active_topology",
    "base_active_topology_sha256",
    "additive_reserved_logical_replicas",
    "additive_reserved_gpus",
    "additive_reserved_tp1_replicas",
    "additive_reserved_tp2_replicas",
    "additive_reserved_topology",
    "additive_reserved_topology_sha256",
    "effective_active_logical_replicas",
    "effective_active_gpus",
    "effective_active_topology",
    "effective_active_topology_sha256",
    "retained_warm_turnover_job_elements",
    "retained_warm_turnover_gpus",
    "retained_warm_turnover_tp1_allocations",
    "retained_warm_turnover_tp2_allocations",
    "retained_warm_turnover_topology",
    "retained_warm_turnover_topology_sha256",
    "attested_total_gpus",
    "job_element_accounting",
    "active_gpus",
    "warm_headroom_gpus",
    "cell_ceiling",
    "reserve_jobs",
    "submit_headroom",
    "cpu",
    "memory_mib",
    "preempt_type",
    "capacity_source",
    "scheduler_cluster",
    "scheduler_account",
    "scheduler_user",
    "scheduler_max_jobs",
    "scheduler_max_submit_jobs",
    "running_scientific_jobs",
    "minimum_scientific_wall_seconds",
    "scientific_qos_contracts",
    "partition_cpus",
    "partition_memory_mib",
    "partition_gpus",
    "fleet_contract_sha256",
    "active_fleet_topology_sha256",
    "scientific_server_preempt_mode",
    "scientific_client_preempt_mode",
    "squeue_complete",
    "sacct_complete",
    "scheduler_evidence_id",
    "scheduler_evidence_sha256",
    "canary_id",
    "canary_evidence_sha256",
    "scientific_server_placements",
    "scientific_client_placements",
    "marker_id",
}
_SERVER_FIELDS = {
    "partition",
    "qos",
    "partition_preempt_mode",
    "qos_preempt_mode",
    "base_active_gpus",
    "reserved_additive_gpus",
    "effective_active_gpus",
    "retained_warm_turnover_gpus",
    "attested_total_gpus",
    "partition_cpus",
    "partition_memory_mib",
    "partition_gpus",
    "partition_nodes",
}
_CLIENT_FIELDS = {
    "partition",
    "qos",
    "partition_preempt_mode",
    "qos_preempt_mode",
    "slots",
    "cpus",
    "memory_mib",
    "reserve_jobs",
    "submit_headroom",
}
_QOS_CONTRACT_FIELDS = {
    "qos",
    "max_wall_seconds",
    "max_jobs_per_user",
    "max_submit_jobs_per_user",
    "required_wall_seconds",
    "required_running_jobs",
    "required_submit_jobs",
}
_STATIC_FEASIBILITY_BINDING_FIELDS = {
    "path",
    "sha256",
    "certificate_id",
}
_STATIC_FEASIBILITY_FIELDS = {
    "schema_version",
    "protocol",
    "passed",
    "capacity_generation",
    "release_git_commit",
    "source_tree_sha256",
    "qualification_plan_id",
    "qualification_runner_source_sha256",
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
_TOPOLOGY_FIELDS = {
    "shape_id",
    "serving_profile",
    "tasks",
    "cpus",
    "memory_mib",
    "gpus",
    "time_limit_seconds",
}
_JOB_ELEMENT_ACCOUNTING_FIELDS = {
    "cell_job_elements",
    "active_server_job_elements",
    "warm_turnover_job_elements",
    "controller_monitor_other_held_job_elements",
    "total_non_cell_reserve_job_elements",
    "total_canary_job_elements",
}


class ProtectedCapacityError(RuntimeError):
    """Protected placement is absent, drifted, or no longer live."""


@dataclass(frozen=True)
class ProtectedPlacement:
    partition: str
    qos: str
    capacity: Mapping[str, int]


@dataclass(frozen=True)
class ProtectedCapacityContract:
    path: Path
    sha256: str
    marker_id: str
    release_git_commit: str
    release_tag_object: str
    scheduler_evidence_id: str
    scheduler_evidence_sha256: str
    canary_id: str
    canary_evidence_sha256: str
    capacity_generation: int
    base_fleet_contract_path: Path
    base_fleet_contract_sha256: str
    effective_fleet_contract_path: Path
    effective_fleet_contract_sha256: str
    additive_overlay_contract_path: Path
    additive_overlay_contract_sha256: str
    static_feasibility_certificate_path: Path
    static_feasibility_certificate_sha256: str
    static_feasibility_certificate_id: str
    static_feasibility_wave_passed: bool
    static_feasibility_selected_cell_count: int
    static_feasibility_target_cell_count: int
    static_feasibility_shortfall_cells: int
    static_feasibility_configured_client_ceiling: int
    static_feasibility_certified_saturation_target: int
    fleet_contract_sha256: str
    active_fleet_topology_sha256: str
    base_active_logical_replicas: int
    base_active_gpus: int
    additive_reserved_logical_replicas: int
    additive_reserved_gpus: int
    effective_active_logical_replicas: int
    effective_active_gpus: int
    retained_warm_turnover_job_elements: int
    retained_warm_turnover_gpus: int
    attested_total_gpus: int
    job_element_accounting: Mapping[str, int]
    preempt_type: str
    capacity_source: str
    scheduler_cluster: str
    scheduler_account: str
    scheduler_user: str
    scheduler_max_jobs: int | None
    scheduler_max_submit_jobs: int
    running_scientific_jobs: int
    scientific_qos_contracts: tuple[Mapping[str, Any], ...]
    partition_cpus: int
    partition_memory_mib: int
    partition_gpus: int
    server_placements: tuple[ProtectedPlacement, ...]
    client_placements: tuple[ProtectedPlacement, ...]

    def placement(self, *, role: str, partition: str, qos: str) -> ProtectedPlacement:
        rows = (
            self.server_placements
            if role == "server"
            else self.client_placements
            if role == "client"
            else ()
        )
        matches = [
            row
            for row in rows
            if row.partition == partition and row.qos == qos
        ]
        if len(matches) != 1:
            raise ProtectedCapacityError(
                f"scientific {role} placement {partition}/{qos} is not explicitly "
                f"authorized by protected-capacity marker {self.marker_id}"
            )
        return matches[0]


@dataclass(frozen=True)
class StaticFeasibilityCertificate:
    """Sealed zero-QID trace of the exact sequential-WDRR admission wave.

    Generation one truthfully records the unscaled base fleet's 278-cell
    saturation cut against the separately configured 384-client Slurm ceiling.
    Later controlled additive generations may preserve a different authenticated
    cut.  A sub-ceiling cut is valid capacity evidence: it is filled and refilled
    during qualification, while scaling is driven only by measured throughput.
    """

    path: Path
    sha256: str
    certificate_id: str
    capacity_generation: int
    base_fleet_contract_sha256: str
    effective_fleet_contract_sha256: str
    additive_overlay_contract_sha256: str
    base_logical_replicas: int
    base_allocated_gpus: int
    effective_logical_replicas: int
    effective_active_gpus: int
    base_profile_replicas: Mapping[str, int]
    effective_profile_replicas: Mapping[str, int]
    additive_profile_delta: Mapping[str, Mapping[str, int]]
    additive_tp1_logical_replicas: int
    additive_tp2_logical_replicas: int
    additive_allocated_gpus: int
    additive_topology_sha256: str
    selected_cell_ids_sha256: str
    wave_passed: bool
    selected_cell_count: int
    target_cell_count: int
    shortfall_cells: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class TrustedScientificJobProvenance:
    """Exact caller reconciliation used to discount current scientific jobs.

    The sealed capacity contract describes the dynamic running/448-submitted
    scientific envelope.  Existing schema-5 allocations are already part of that
    envelope and must not be charged a second time when residual association/QOS
    headroom is evaluated.  Conversely, a job inferred from a name or comment alone
    must never be discounted.  Callers therefore reconcile their durable dispatcher
    and fleet transactions against one complete, current squeue+sacct snapshot and
    bind the resulting exact IDs here.
    """

    payload: Mapping[str, Any]

    @property
    def job_ids(self) -> tuple[str, ...]:
        return tuple(self.payload["trusted_live_job_ids"])

    @property
    def provenance_id(self) -> str:
        return str(self.payload["provenance_id"])


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


def _identity_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(dict(value))).hexdigest()


def _normalized_live_job_states(
    value: Mapping[str, str],
    *,
    description: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ProtectedCapacityError(f"{description} must be a mapping")
    normalized: dict[str, str] = {}
    for raw_job_id, raw_state in value.items():
        if (
            not isinstance(raw_job_id, str)
            or re.fullmatch(r"[0-9]+(?:_[0-9]+)?", raw_job_id) is None
            or not isinstance(raw_state, str)
        ):
            raise ProtectedCapacityError(
                f"{description} contains a malformed job identity/state"
            )
        state = raw_state.upper().split()[0].rstrip("+")
        if state not in _LIVE_JOB_STATES or raw_job_id in normalized:
            raise ProtectedCapacityError(
                f"{description} contains a non-live or duplicated job"
            )
        normalized[raw_job_id] = state
    return dict(sorted(normalized.items()))


def _normalized_job_ids(
    value: Collection[str],
    *,
    description: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise ProtectedCapacityError(f"{description} must be a collection")
    normalized = sorted(value)
    if (
        any(
            not isinstance(job_id, str)
            or re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) is None
            for job_id in normalized
        )
        or len(normalized) != len(set(normalized))
    ):
        raise ProtectedCapacityError(
            f"{description} contains malformed or duplicated identities"
        )
    return tuple(normalized)


def build_trusted_scientific_job_provenance(
    *,
    scheduler_job_states: Mapping[str, str],
    scheduler_captured_timestamp: float,
    trusted_cell_job_ids: Collection[str],
    trusted_fleet_job_ids: Collection[str],
    dispatcher_ledger_updated_timestamp: float | None,
    exact_cell_quiescence: bool,
    dispatcher_provenance_id: str,
    fleet_provenance_id: str,
    fleet_contract_sha256: str,
    fleet_generation: int,
    scheduler_truth_id: str,
    now: float | None = None,
    provenance_errors: Sequence[str] = (),
) -> TrustedScientificJobProvenance:
    """Build the sole exact scientific-occupancy discount authority.

    ``scheduler_job_states`` must be the complete current live-job intersection from
    a successful joined squeue+sacct observation.  The caller owns the domain-specific
    ledger/intent/registry/spooled-script join and binds it through the three SHA-256
    identities.  This function owns freshness, uniqueness, exact-set, and state
    validation shared by every scientific scheduler caller.
    """

    timestamp = time.time() if now is None else float(now)
    if (
        not math.isfinite(timestamp)
        or not isinstance(scheduler_captured_timestamp, (int, float))
        or isinstance(scheduler_captured_timestamp, bool)
        or not math.isfinite(float(scheduler_captured_timestamp))
        or float(scheduler_captured_timestamp) <= 0
        or float(scheduler_captured_timestamp) > timestamp + 1.0
        or timestamp - float(scheduler_captured_timestamp)
        > MAX_TRUSTED_SCHEDULER_AGE_SECONDS
    ):
        raise ProtectedCapacityError(
            "trusted scientific scheduler reconciliation is missing, future, "
            "or stale"
        )
    errors = tuple(provenance_errors)
    if (
        any(not isinstance(error, str) or not error for error in errors)
        or errors
    ):
        raise ProtectedCapacityError(
            "trusted scientific scheduler reconciliation is ambiguous: "
            + "; ".join(str(error) for error in errors[:10])
        )
    states = _normalized_live_job_states(
        scheduler_job_states,
        description="trusted scientific complete scheduler truth",
    )
    cells = _normalized_job_ids(
        trusted_cell_job_ids,
        description="trusted scientific cell identities",
    )
    fleet = _normalized_job_ids(
        trusted_fleet_job_ids,
        description="trusted scientific fleet identities",
    )
    overlap = set(cells) & set(fleet)
    if overlap:
        raise ProtectedCapacityError(
            "trusted scientific cell/fleet identities overlap"
        )
    trusted = tuple(sorted((*cells, *fleet)))
    absent = sorted(set(trusted) - set(states))
    if absent:
        raise ProtectedCapacityError(
            "trusted scientific identities are absent from the caller's complete "
            f"current scheduler truth: {absent[:10]}"
        )
    if not isinstance(exact_cell_quiescence, bool):
        raise ProtectedCapacityError(
            "trusted scientific exact-cell-quiescence flag is invalid"
        )
    if exact_cell_quiescence and cells:
        raise ProtectedCapacityError(
            "trusted scientific cell quiescence conflicts with live cell IDs"
        )
    if not exact_cell_quiescence:
        if (
            not isinstance(dispatcher_ledger_updated_timestamp, (int, float))
            or isinstance(dispatcher_ledger_updated_timestamp, bool)
            or not math.isfinite(float(dispatcher_ledger_updated_timestamp))
            or float(dispatcher_ledger_updated_timestamp) <= 0
            or float(dispatcher_ledger_updated_timestamp) > timestamp + 1.0
            or timestamp - float(dispatcher_ledger_updated_timestamp)
            > MAX_DISPATCHER_LEDGER_AGE_SECONDS
        ):
            raise ProtectedCapacityError(
                "trusted scientific dispatcher ledger is missing, future, or "
                "older than 360 seconds"
            )
    elif dispatcher_ledger_updated_timestamp is not None and (
        not isinstance(dispatcher_ledger_updated_timestamp, (int, float))
        or isinstance(dispatcher_ledger_updated_timestamp, bool)
        or not math.isfinite(float(dispatcher_ledger_updated_timestamp))
        or float(dispatcher_ledger_updated_timestamp) <= 0
        or float(dispatcher_ledger_updated_timestamp) > timestamp + 1.0
    ):
        raise ProtectedCapacityError(
            "trusted scientific quiescent dispatcher timestamp is invalid"
        )
    if (
        not isinstance(fleet_generation, int)
        or isinstance(fleet_generation, bool)
        or fleet_generation < 1
        or any(
            _SHA256_RE.fullmatch(value) is None
            for value in (
                dispatcher_provenance_id,
                fleet_provenance_id,
                fleet_contract_sha256,
                scheduler_truth_id,
            )
        )
    ):
        raise ProtectedCapacityError(
            "trusted scientific immutable provenance bindings are invalid"
        )
    body: dict[str, Any] = {
        "schema_version": 1,
        "protocol": TRUSTED_SCIENTIFIC_PROVENANCE_PROTOCOL,
        "scheduler_captured_timestamp": float(scheduler_captured_timestamp),
        "dispatcher_ledger_updated_timestamp": (
            None
            if dispatcher_ledger_updated_timestamp is None
            else float(dispatcher_ledger_updated_timestamp)
        ),
        "exact_cell_quiescence": exact_cell_quiescence,
        "dispatcher_provenance_id": dispatcher_provenance_id,
        "fleet_provenance_id": fleet_provenance_id,
        "fleet_contract_sha256": fleet_contract_sha256,
        "fleet_generation": fleet_generation,
        "scheduler_truth_id": scheduler_truth_id,
        "trusted_cell_job_ids": list(cells),
        "trusted_fleet_job_ids": list(fleet),
        "trusted_live_job_ids": list(trusted),
    }
    body["provenance_id"] = _identity_sha256(body)
    return TrustedScientificJobProvenance(payload=body)


def validate_trusted_scientific_job_provenance(
    value: TrustedScientificJobProvenance | Mapping[str, Any],
    *,
    validation_timestamp: float,
) -> TrustedScientificJobProvenance:
    """Revalidate one stored provenance envelope without mutable source files."""

    raw = value.payload if isinstance(value, TrustedScientificJobProvenance) else value
    expected_fields = {
        "schema_version",
        "protocol",
        "scheduler_captured_timestamp",
        "dispatcher_ledger_updated_timestamp",
        "exact_cell_quiescence",
        "dispatcher_provenance_id",
        "fleet_provenance_id",
        "fleet_contract_sha256",
        "fleet_generation",
        "scheduler_truth_id",
        "trusted_cell_job_ids",
        "trusted_fleet_job_ids",
        "trusted_live_job_ids",
        "provenance_id",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != expected_fields
        or raw.get("schema_version") != 1
        or raw.get("protocol") != TRUSTED_SCIENTIFIC_PROVENANCE_PROTOCOL
        or raw.get("provenance_id")
        != _identity_sha256(
            {key: item for key, item in raw.items() if key != "provenance_id"}
        )
    ):
        raise ProtectedCapacityError(
            "trusted scientific provenance envelope is invalid"
        )
    reconstructed = build_trusted_scientific_job_provenance(
        scheduler_job_states={
            str(job_id): "RUNNING"
            for job_id in raw.get("trusted_live_job_ids", [])
        },
        scheduler_captured_timestamp=float(
            raw.get("scheduler_captured_timestamp", 0.0)
        ),
        trusted_cell_job_ids=raw.get("trusted_cell_job_ids", []),
        trusted_fleet_job_ids=raw.get("trusted_fleet_job_ids", []),
        dispatcher_ledger_updated_timestamp=raw.get(
            "dispatcher_ledger_updated_timestamp"
        ),
        exact_cell_quiescence=raw.get("exact_cell_quiescence"),
        dispatcher_provenance_id=str(
            raw.get("dispatcher_provenance_id", "")
        ),
        fleet_provenance_id=str(raw.get("fleet_provenance_id", "")),
        fleet_contract_sha256=str(raw.get("fleet_contract_sha256", "")),
        fleet_generation=raw.get("fleet_generation"),
        scheduler_truth_id=str(raw.get("scheduler_truth_id", "")),
        now=float(validation_timestamp),
    )
    if reconstructed.payload != dict(raw):
        raise ProtectedCapacityError(
            "trusted scientific provenance fields are noncanonical"
        )
    return reconstructed


def canonical_marker_path(results_root: str | Path) -> Path:
    """Return the one protected-capacity marker location for a result hierarchy."""

    return (
        Path(results_root).expanduser().resolve()
        / "recovery"
        / "schema5-v1"
        / MARKER_FILENAME
    )


def _read_sealed_json(path: Path) -> tuple[Path, bytes, dict[str, Any]]:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProtectedCapacityError(
            f"protected-capacity marker is missing or unsafe: {lexical}: {exc}"
        ) from exc
    if resolved != lexical or lexical.is_symlink() or not lexical.is_file():
        raise ProtectedCapacityError(
            f"protected-capacity marker must be a non-symlink regular file: {lexical}"
        )
    descriptor_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        descriptor_flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical, descriptor_flags)
    except OSError as exc:
        raise ProtectedCapacityError(
            f"cannot open protected-capacity marker {lexical}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) & 0o222
        or before.st_nlink != 1
        or (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    ):
        raise ProtectedCapacityError(
            "protected-capacity marker must be a stable, read-only, single-link file"
        )
    raw = b"".join(chunks)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtectedCapacityError(
                    f"protected-capacity marker duplicates JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        raise ProtectedCapacityError(
            f"protected-capacity marker contains {token!r}"
        )

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except ProtectedCapacityError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtectedCapacityError(
            f"invalid protected-capacity marker JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProtectedCapacityError(
            "protected-capacity marker must contain one JSON object"
        )
    current = lexical.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino):
        raise ProtectedCapacityError(
            "protected-capacity marker was replaced while being read"
        )
    return lexical, raw, payload


def _positive_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ProtectedCapacityError(
            f"protected-capacity {field} must be an integer >= {minimum}"
        )
    return value


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _profile_replica_counts(
    value: Any,
    *,
    field: str,
    minimum: int,
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(PRODUCTION_SERVING_PROFILES):
        raise ProtectedCapacityError(
            f"static feasibility {field} does not contain the exact profile set"
        )
    counts: dict[str, int] = {}
    for profile in PRODUCTION_SERVING_PROFILES:
        counts[profile] = _positive_integer(
            value.get(profile),
            field=f"static feasibility {field} {profile}",
            minimum=minimum,
        )
    return counts


def load_static_feasibility_certificate(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_certificate_id: str | None = None,
    expected_capacity_generation: int | None = None,
    expected_base_fleet_contract_sha256: str | None = None,
    expected_effective_fleet_contract_sha256: str | None = None,
    expected_additive_overlay_contract_sha256: str | None = None,
    expected_release_git_commit: str | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_dispatcher_source_sha256: str | None = None,
    expected_qualification_runner_source_sha256: str | None = None,
) -> StaticFeasibilityCertificate:
    """Load the sealed zero-QID sequential-WDRR admission certificate.

    The producer independently recomputes the exact dispatcher wave.  Runtime
    validation deliberately repeats its closed cardinality, topology, trace-hash,
    and fairness invariants so a self-hashed but semantically hollow object cannot
    become serving authority.
    """

    certificate_path, raw, payload = _read_sealed_json(Path(path))
    digest = hashlib.sha256(raw).hexdigest()
    if set(payload) != _STATIC_FEASIBILITY_FIELDS:
        raise ProtectedCapacityError(
            "static feasibility certificate fields differ from the closed schema"
        )
    identity = dict(payload)
    certificate_id = identity.pop("certificate_id", None)
    hash_fields = (
        "release_fleet_contract_sha256",
        "base_fleet_contract_sha256",
        "proposed_effective_fleet_contract_sha256",
        "additive_overlay_contract_sha256",
        "additive_topology_sha256",
        "dispatcher_source_sha256",
        "microbatch_trace_sha256",
        "selected_cell_ids_sha256",
        "source_tree_sha256",
        "qualification_plan_id",
        "qualification_runner_source_sha256",
    )
    capacity_generation = _positive_integer(
        payload.get("capacity_generation"),
        field="static feasibility capacity_generation",
        minimum=1,
    )
    generation_directory = f"c{capacity_generation:06d}"
    if (
        certificate_path.name != STATIC_FEASIBILITY_FILENAME
        or (
            capacity_generation == 1
            and certificate_path.parent.name == generation_directory
            and certificate_path.parent.parent.name == "capacity-generations"
        )
        or (
            capacity_generation > 1
            and (
                certificate_path.parent.name != generation_directory
                or certificate_path.parent.parent.name
                != "capacity-generations"
            )
        )
    ):
        raise ProtectedCapacityError(
            "static feasibility certificate path is not generation-addressed"
        )
    if (
        payload.get("schema_version") != STATIC_FEASIBILITY_SCHEMA_VERSION
        or payload.get("protocol") != STATIC_FEASIBILITY_PROTOCOL
        or payload.get("passed") is not True
        or not isinstance(certificate_id, str)
        or _SHA256_RE.fullmatch(certificate_id) is None
        or certificate_id != _identity_sha256(identity)
        or any(
            _SHA256_RE.fullmatch(str(payload.get(field, ""))) is None
            for field in hash_fields
        )
        or payload["release_fleet_contract_sha256"]
        != payload["base_fleet_contract_sha256"]
        or payload["additive_overlay_contract_sha256"]
        != payload["proposed_effective_fleet_contract_sha256"]
        or _GIT_OBJECT_RE.fullmatch(
            str(payload.get("release_git_commit", ""))
        )
        is None
        or payload.get("dispatcher_policy")
        != "dispatch_sweeps.plan_admission-sequential-wdrr-v1"
        or payload.get("fanout_slots_per_replica") != 24
        or payload.get("run_weights")
        != {"schema5_throughput_qualification_v1": 1.0}
        or payload.get("initial_fairness") != {"cursor": 0, "deficits": {}}
        or (expected_sha256 is not None and digest != expected_sha256)
        or (
            expected_certificate_id is not None
            and certificate_id != expected_certificate_id
        )
        or (
            expected_capacity_generation is not None
            and capacity_generation != expected_capacity_generation
        )
        or (
            expected_base_fleet_contract_sha256 is not None
            and payload["base_fleet_contract_sha256"]
            != expected_base_fleet_contract_sha256
        )
        or (
            expected_effective_fleet_contract_sha256 is not None
            and payload["proposed_effective_fleet_contract_sha256"]
            != expected_effective_fleet_contract_sha256
        )
        or (
            expected_additive_overlay_contract_sha256 is not None
            and payload["additive_overlay_contract_sha256"]
            != expected_additive_overlay_contract_sha256
        )
        or (
            expected_release_git_commit is not None
            and payload["release_git_commit"] != expected_release_git_commit
        )
        or (
            expected_source_tree_sha256 is not None
            and payload["source_tree_sha256"]
            != expected_source_tree_sha256
        )
        or (
            expected_dispatcher_source_sha256 is not None
            and payload["dispatcher_source_sha256"]
            != expected_dispatcher_source_sha256
        )
        or (
            expected_qualification_runner_source_sha256 is not None
            and payload["qualification_runner_source_sha256"]
            != expected_qualification_runner_source_sha256
        )
    ):
        raise ProtectedCapacityError(
            "static feasibility certificate identity/binding is invalid"
        )
    base = _profile_replica_counts(
        payload.get("base_profile_replicas"),
        field="base_profile_replicas",
        minimum=1,
    )
    effective = _profile_replica_counts(
        payload.get("effective_profile_replicas"),
        field="effective_profile_replicas",
        minimum=1,
    )
    raw_delta = payload.get("additive_profile_delta")
    if (
        not isinstance(raw_delta, dict)
        or set(raw_delta) != set(PRODUCTION_SERVING_PROFILES)
    ):
        raise ProtectedCapacityError(
            "static feasibility additive profile delta is malformed"
        )
    delta: dict[str, Mapping[str, int]] = {}
    tp1 = 0
    tp2 = 0
    additive_gpus = 0
    for profile in PRODUCTION_SERVING_PROFILES:
        row = raw_delta.get(profile)
        expected_tp = int(SERVING_PROFILES[profile].tp_size)
        if (
            not isinstance(row, dict)
            or set(row)
            != {"logical_replicas", "tensor_parallel_size", "allocated_gpus"}
        ):
            raise ProtectedCapacityError(
                f"static feasibility additive delta for {profile} is malformed"
            )
        logical = _positive_integer(
            row.get("logical_replicas"),
            field=f"static feasibility {profile} additive replicas",
        )
        allocated = _positive_integer(
            row.get("allocated_gpus"),
            field=f"static feasibility {profile} additive GPUs",
        )
        if (
            row.get("tensor_parallel_size") != expected_tp
            or effective[profile] != base[profile] + logical
            or allocated != logical * expected_tp
        ):
            raise ProtectedCapacityError(
                f"static feasibility additive topology for {profile} is inconsistent"
            )
        if expected_tp == 1:
            tp1 += logical
        else:
            tp2 += logical
        additive_gpus += allocated
        delta[profile] = {
            "logical_replicas": logical,
            "tensor_parallel_size": expected_tp,
            "allocated_gpus": allocated,
        }
    base_gpus = sum(
        base[profile] * int(SERVING_PROFILES[profile].tp_size)
        for profile in PRODUCTION_SERVING_PROFILES
    )
    effective_gpus = sum(
        effective[profile] * int(SERVING_PROFILES[profile].tp_size)
        for profile in PRODUCTION_SERVING_PROFILES
    )
    additive_topology = [
        {
            "serving_profile": profile,
            "replicas": int(delta[profile]["logical_replicas"]),
            "tensor_parallel_size": int(
                delta[profile]["tensor_parallel_size"]
            ),
            "allocated_gpus": int(delta[profile]["allocated_gpus"]),
        }
        for profile in PRODUCTION_SERVING_PROFILES
        if int(delta[profile]["logical_replicas"])
    ]
    wave = payload.get("wave")
    final_fairness = payload.get("final_fairness")
    if not isinstance(wave, dict) or not isinstance(final_fairness, dict):
        raise ProtectedCapacityError(
            "static feasibility wave/final fairness is malformed"
        )
    microbatches = wave.get("microbatches")
    selected_wave = wave.get("selected_wave")
    deficits = final_fairness.get("deficits")
    selected_cell_count = payload.get("selected_cell_count")
    wave_selected_cell_count = wave.get("selected_cell_count")
    target_cell_count = wave.get("target_active_cells")
    shortfall_cells = wave.get("shortfall_cells")
    wave_passed = wave.get("passed")
    if (
        not isinstance(selected_cell_count, int)
        or isinstance(selected_cell_count, bool)
        or not 1 <= selected_cell_count <= CLIENT_JOB_ELEMENTS
        or wave_selected_cell_count != selected_cell_count
        or target_cell_count != CLIENT_JOB_ELEMENTS
        or shortfall_cells != CLIENT_JOB_ELEMENTS - selected_cell_count
        or wave_passed
        is not (
            selected_cell_count == CLIENT_JOB_ELEMENTS
            and shortfall_cells == 0
        )
    ):
        raise ProtectedCapacityError(
            "static feasibility wave cardinality/pass state is inconsistent"
        )
    full_batches, final_batch = divmod(selected_cell_count, 24)
    expected_microbatch_sizes = [24] * full_batches
    if final_batch:
        expected_microbatch_sizes.append(final_batch)
    if (
        payload.get("base_logical_replicas") != sum(base.values())
        or payload.get("base_allocated_gpus") != base_gpus
        or payload.get("effective_logical_replicas") != sum(effective.values())
        or payload.get("effective_active_gpus") != effective_gpus
        or payload.get("additive_tp1_logical_replicas") != tp1
        or payload.get("additive_tp2_logical_replicas") != tp2
        or payload.get("additive_allocated_gpus") != additive_gpus
        or additive_gpus != effective_gpus - base_gpus
        or payload.get("additive_topology_sha256")
        != _sha256_value(additive_topology)
        or not isinstance(microbatches, list)
        or len(microbatches) != len(expected_microbatch_sizes)
        or payload.get("microbatch_trace_sha256")
        != _sha256_value(microbatches)
        or not isinstance(selected_wave, list)
        or len(selected_wave) != selected_cell_count
        or len(
            {
                row.get("cell_id")
                for row in selected_wave
                if isinstance(row, dict)
                and isinstance(row.get("cell_id"), str)
            }
        )
        != selected_cell_count
        or payload.get("selected_cell_ids_sha256")
        != _sha256_value([row["cell_id"] for row in selected_wave])
        or wave.get("maximum_microbatch") != 24
        or wave.get("fanout_slots_per_replica") != 24
        or wave.get("microbatch_count") != len(expected_microbatch_sizes)
        or [
            batch.get("selected_count")
            if isinstance(batch, dict)
            else None
            for batch in microbatches
        ]
        != expected_microbatch_sizes
        or wave.get("profile_replicas") != effective
        or final_fairness.get("cursor") != wave.get("ending_cursor")
        or deficits != wave.get("ending_deficits")
        or not isinstance(deficits, dict)
        or any(
            not isinstance(key, str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for key, value in deficits.items()
        )
        or (
            capacity_generation == 1
            and (
                payload["proposed_effective_fleet_contract_sha256"]
                != payload["base_fleet_contract_sha256"]
                or payload["additive_overlay_contract_sha256"]
                != payload["base_fleet_contract_sha256"]
                or effective != base
                or tp1 != 0
                or tp2 != 0
                or additive_gpus != 0
                or additive_topology != []
                or selected_cell_count
                != PREQUALIFICATION_SELECTED_CELL_COUNT
                or wave_passed is not False
            )
        )
        or (
            capacity_generation > 1
            and (
                tp1 + tp2 < 1
                or additive_gpus < 1
                or effective_gpus <= base_gpus
            )
        )
    ):
        raise ProtectedCapacityError(
            "static feasibility certificate does not prove the generation's "
            "exact sequential admission wave"
        )
    return StaticFeasibilityCertificate(
        path=certificate_path,
        sha256=digest,
        certificate_id=certificate_id,
        capacity_generation=capacity_generation,
        base_fleet_contract_sha256=str(
            payload["base_fleet_contract_sha256"]
        ),
        effective_fleet_contract_sha256=str(
            payload["proposed_effective_fleet_contract_sha256"]
        ),
        additive_overlay_contract_sha256=str(
            payload["additive_overlay_contract_sha256"]
        ),
        base_logical_replicas=int(payload["base_logical_replicas"]),
        base_allocated_gpus=int(payload["base_allocated_gpus"]),
        effective_logical_replicas=int(
            payload["effective_logical_replicas"]
        ),
        effective_active_gpus=int(payload["effective_active_gpus"]),
        base_profile_replicas=base,
        effective_profile_replicas=effective,
        additive_profile_delta=delta,
        additive_tp1_logical_replicas=tp1,
        additive_tp2_logical_replicas=tp2,
        additive_allocated_gpus=additive_gpus,
        additive_topology_sha256=str(payload["additive_topology_sha256"]),
        selected_cell_ids_sha256=str(payload["selected_cell_ids_sha256"]),
        wave_passed=bool(wave_passed),
        selected_cell_count=int(selected_cell_count),
        target_cell_count=int(target_cell_count),
        shortfall_cells=int(shortfall_cells),
        payload=dict(payload),
    )


def _validated_topology(
    value: Any,
    *,
    field: str,
    expected_count: int,
    expected_gpus: int,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ProtectedCapacityError(
            f"protected-capacity {field} has the wrong replica count"
        )
    rows: list[Mapping[str, Any]] = []
    identities: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != _TOPOLOGY_FIELDS:
            raise ProtectedCapacityError(
                f"protected-capacity {field} row {index} fields drifted"
            )
        shape_id = row.get("shape_id")
        profile = row.get("serving_profile")
        if (
            not isinstance(shape_id, str)
            or _PLACEMENT_RE.fullmatch(shape_id) is None
            or shape_id in identities
            or not isinstance(profile, str)
            or _PLACEMENT_RE.fullmatch(profile) is None
            or row.get("tasks") != 1
        ):
            raise ProtectedCapacityError(
                f"protected-capacity {field} row {index} identity is invalid"
            )
        identities.add(shape_id)
        for numeric in (
            "cpus",
            "memory_mib",
            "gpus",
            "time_limit_seconds",
        ):
            _positive_integer(
                row.get(numeric),
                field=f"protected-capacity {field} row {index} {numeric}",
                minimum=1,
            )
        if row["time_limit_seconds"] != MIN_SCIENTIFIC_WALL_SECONDS:
            raise ProtectedCapacityError(
                f"protected-capacity {field} row {index} walltime drifted"
            )
        rows.append(dict(row))
    if sum(int(row["gpus"]) for row in rows) != expected_gpus:
        raise ProtectedCapacityError(
            f"protected-capacity {field} has the wrong GPU total"
        )
    return tuple(rows)


def _placements(
    value: Any,
    *,
    role: str,
    preempt_type: str,
) -> tuple[ProtectedPlacement, ...]:
    expected_fields = _SERVER_FIELDS if role == "server" else _CLIENT_FIELDS
    capacity_fields = (
        (
            "base_active_gpus",
            "reserved_additive_gpus",
            "effective_active_gpus",
            "retained_warm_turnover_gpus",
            "attested_total_gpus",
            "partition_cpus",
            "partition_memory_mib",
            "partition_gpus",
            "partition_nodes",
        )
        if role == "server"
        else ("slots", "cpus", "memory_mib", "reserve_jobs", "submit_headroom")
    )
    if not isinstance(value, list) or not value:
        raise ProtectedCapacityError(
            f"protected-capacity marker has no scientific {role} placements"
        )
    parsed: list[ProtectedPlacement] = []
    seen: set[tuple[str, str]] = set()
    previous_sort_key: tuple[str, str, bytes] | None = None
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ProtectedCapacityError(
                f"scientific {role} placement {index} fields drifted"
            )
        partition, qos = row.get("partition"), row.get("qos")
        if (
            not isinstance(partition, str)
            or _PLACEMENT_RE.fullmatch(partition) is None
            or not isinstance(qos, str)
            or _PLACEMENT_RE.fullmatch(qos) is None
            or row.get("partition_preempt_mode") != "OFF"
            or (
                row.get("qos_preempt_mode") != "OFF"
                and not (
                    preempt_type == "preempt/partition_prio"
                    and row.get("qos_preempt_mode") == "cluster"
                )
            )
        ):
            raise ProtectedCapacityError(
                f"scientific {role} placement {index} is not explicitly "
                "effectively nonpreemptible under the bound PreemptType"
            )
        identity = (partition, qos)
        if identity in seen:
            raise ProtectedCapacityError(
                f"duplicate scientific {role} placement {partition}/{qos}"
            )
        seen.add(identity)
        sort_key = (partition, qos, canonical_bytes(row))
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            raise ProtectedCapacityError(
                f"scientific {role} placements are not in canonical order"
            )
        previous_sort_key = sort_key
        parsed.append(
            ProtectedPlacement(
                partition=partition,
                qos=qos,
                capacity={
                    field: _positive_integer(
                        row.get(field),
                        field=f"{role} placement {partition}/{qos} {field}",
                        minimum=(
                            1
                            if role == "server"
                            and field
                            in {
                                "partition_cpus",
                                "partition_memory_mib",
                                "partition_nodes",
                            }
                            else 0
                        ),
                    )
                    for field in capacity_fields
                },
            )
        )
    return tuple(parsed)


def _qos_contracts(
    value: Any,
    *,
    expected_running_jobs: int,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ProtectedCapacityError(
            "protected-capacity marker has no scientific QOS contracts"
        )
    parsed: list[Mapping[str, Any]] = []
    prior = ""
    for index, row in enumerate(value):
        if (
            not isinstance(row, dict)
            or set(row) != _QOS_CONTRACT_FIELDS
        ):
            raise ProtectedCapacityError(
                f"scientific QOS contract {index} fields drifted"
            )
        qos = row.get("qos")
        if (
            not isinstance(qos, str)
            or _PLACEMENT_RE.fullmatch(qos) is None
            or qos <= prior
        ):
            raise ProtectedCapacityError(
                "scientific QOS contracts are duplicated or not canonical"
            )
        prior = qos
        optional: dict[str, int | None] = {}
        for field in (
            "max_wall_seconds",
            "max_jobs_per_user",
            "max_submit_jobs_per_user",
        ):
            observed = row.get(field)
            optional[field] = (
                None
                if observed is None
                else _positive_integer(
                    observed, field=f"{qos} {field}", minimum=1
                )
            )
        required_wall = _positive_integer(
            row.get("required_wall_seconds"),
            field=f"{qos} required walltime",
            minimum=CLIENT_WALL_SECONDS,
        )
        required_running = _positive_integer(
            row.get("required_running_jobs"),
            field=f"{qos} required running jobs",
        )
        required_submit = _positive_integer(
            row.get("required_submit_jobs"),
            field=f"{qos} required submit jobs",
            minimum=1,
        )
        if (
            optional["max_wall_seconds"] is not None
            and optional["max_wall_seconds"] < required_wall
        ) or (
            optional["max_jobs_per_user"] is not None
            and optional["max_jobs_per_user"] < required_running
        ) or (
            optional["max_submit_jobs_per_user"] is not None
            and optional["max_submit_jobs_per_user"] < required_submit
        ):
            raise ProtectedCapacityError(
                f"scientific QOS contract {qos} cannot sustain its load"
            )
        parsed.append(dict(row))
    if (
        sum(int(row["required_running_jobs"]) for row in parsed)
        != expected_running_jobs
        or sum(int(row["required_submit_jobs"]) for row in parsed)
        != TOTAL_SUBMIT_HEADROOM
        or max(int(row["required_wall_seconds"]) for row in parsed)
        != MIN_SCIENTIFIC_WALL_SECONDS
    ):
        raise ProtectedCapacityError(
            "scientific QOS contracts do not prove the dynamic running-job, "
            "448 submitted-job, and 24-hour serving envelope"
        )
    return tuple(parsed)


def load_contract(
    path: str | Path,
    *,
    expected_release_git_commit: str,
    expected_release_tag_object: str | None = None,
    expected_marker_id: str | None = None,
    expected_sha256: str | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_dispatcher_source_sha256: str | None = None,
    expected_qualification_runner_source_sha256: str | None = None,
) -> ProtectedCapacityContract:
    """Load and verify the exact sealed protected-capacity authority."""

    marker_path, raw, payload = _read_sealed_json(Path(path))
    if set(payload) != _MARKER_FIELDS:
        raise ProtectedCapacityError(
            "protected-capacity marker fields differ from the closed schema"
        )
    identity = dict(payload)
    marker_id = identity.pop("marker_id", None)
    digest = hashlib.sha256(raw).hexdigest()
    capacity_generation = _positive_integer(
        payload.get("capacity_generation"),
        field="capacity_generation",
        minimum=1,
    )
    effective_logical_replicas = _positive_integer(
        payload.get("effective_active_logical_replicas"),
        field="effective_active_logical_replicas",
        minimum=BASE_LOGICAL_REPLICAS,
    )
    effective_active_gpus = _positive_integer(
        payload.get("effective_active_gpus"),
        field="effective_active_gpus",
        minimum=BASE_ACTIVE_GPUS,
    )
    expected_running_jobs = (
        CLIENT_JOB_ELEMENTS
        + effective_logical_replicas
        + RETAINED_WARM_TURNOVER_JOB_ELEMENTS
    )
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("protocol") != PROTOCOL
        or payload.get("passed") is not True
        or payload.get("release_id") != RELEASE_ID
        or payload.get("release_tag") != RELEASE_TAG
        or payload.get("chain_namespace") != CHAIN_NAMESPACE
        or not isinstance(expected_release_git_commit, str)
        or _GIT_OBJECT_RE.fullmatch(expected_release_git_commit) is None
        or payload.get("release_git_commit") != expected_release_git_commit
        or (
            expected_release_tag_object is not None
            and payload.get("release_tag_object") != expected_release_tag_object
        )
        or (
            expected_source_tree_sha256 is not None
            and payload.get("source_tree_sha256")
            != expected_source_tree_sha256
        )
        or (
            expected_dispatcher_source_sha256 is not None
            and payload.get("dispatcher_source_sha256")
            != expected_dispatcher_source_sha256
        )
        or (
            expected_qualification_runner_source_sha256 is not None
            and payload.get("qualification_runner_source_sha256")
            != expected_qualification_runner_source_sha256
        )
        or _GIT_OBJECT_RE.fullmatch(str(payload.get("release_tag_object", "")))
        is None
        or not isinstance(marker_id, str)
        or _SHA256_RE.fullmatch(marker_id) is None
        or marker_id != _identity_sha256(identity)
        or (
            expected_marker_id is not None and marker_id != expected_marker_id
        )
        or (expected_sha256 is not None and digest != expected_sha256)
        or payload.get("scientific_server_preempt_mode") != "OFF"
        or payload.get("scientific_client_preempt_mode") != "OFF"
        or payload.get("capacity_source") != CAPACITY_SOURCE
        or payload.get("running_scientific_jobs")
        != expected_running_jobs
        or payload.get("minimum_scientific_wall_seconds")
        != MIN_SCIENTIFIC_WALL_SECONDS
        or (
            payload.get("scheduler_max_jobs") is not None
            and (
                not isinstance(payload.get("scheduler_max_jobs"), int)
                or isinstance(payload.get("scheduler_max_jobs"), bool)
                or int(payload["scheduler_max_jobs"])
                < expected_running_jobs
            )
        )
        or any(
            not isinstance(payload.get(field), str)
            or _PLACEMENT_RE.fullmatch(str(payload.get(field))) is None
            for field in (
                "scheduler_cluster",
                "scheduler_account",
                "scheduler_user",
            )
        )
        or any(
            not isinstance(payload.get(field), int)
            or isinstance(payload.get(field), bool)
            or int(payload[field]) < minimum
            for field, minimum in (
                ("scheduler_max_submit_jobs", TOTAL_SUBMIT_HEADROOM),
                ("partition_cpus", CLIENT_JOB_ELEMENTS),
                (
                    "partition_memory_mib",
                    CLIENT_JOB_ELEMENTS * 4096,
                ),
                # This aggregate describes the client partition inventory.  The
                # protected GPU envelope is independently and exactly attested by
                # the server placement rows and may live on a different partition.
                ("partition_gpus", 0),
            )
        )
        or payload.get("squeue_complete") is not True
        or payload.get("sacct_complete") is not True
    ):
        raise ProtectedCapacityError(
            "protected-capacity marker identity/release/seal is invalid"
        )
    for field in (
        "scheduler_evidence_id",
        "scheduler_evidence_sha256",
        "canary_id",
        "canary_evidence_sha256",
        "base_fleet_contract_sha256",
        "effective_fleet_contract_sha256",
        "additive_overlay_contract_sha256",
        "fleet_contract_sha256",
        "base_active_topology_sha256",
        "additive_reserved_topology_sha256",
        "effective_active_topology_sha256",
        "retained_warm_turnover_topology_sha256",
        "active_fleet_topology_sha256",
        "source_tree_sha256",
        "dispatcher_source_sha256",
        "qualification_runner_source_sha256",
    ):
        if _SHA256_RE.fullmatch(str(payload.get(field, ""))) is None:
            raise ProtectedCapacityError(
                f"protected-capacity marker {field} is not SHA-256"
            )
    bound_paths: dict[str, Path] = {}
    for field, hash_field in (
        ("base_fleet_contract_path", "base_fleet_contract_sha256"),
        ("effective_fleet_contract_path", "effective_fleet_contract_sha256"),
        ("additive_overlay_contract_path", "additive_overlay_contract_sha256"),
    ):
        raw_path = payload.get(field)
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ProtectedCapacityError(
                f"protected-capacity marker {field} is not absolute"
            )
        path_value, file_raw, _file_payload = _read_sealed_json(Path(raw_path))
        if hashlib.sha256(file_raw).hexdigest() != payload[hash_field]:
            raise ProtectedCapacityError(
                f"protected-capacity marker {field} bytes drifted"
            )
        bound_paths[field] = path_value
    certificate_binding = payload.get("static_feasibility_certificate")
    if (
        not isinstance(certificate_binding, dict)
        or set(certificate_binding) != _STATIC_FEASIBILITY_BINDING_FIELDS
        or not isinstance(certificate_binding.get("path"), str)
        or not Path(certificate_binding["path"]).is_absolute()
        or _SHA256_RE.fullmatch(
            str(certificate_binding.get("sha256", ""))
        )
        is None
        or _SHA256_RE.fullmatch(
            str(certificate_binding.get("certificate_id", ""))
        )
        is None
    ):
        raise ProtectedCapacityError(
            "protected-capacity static feasibility binding is malformed"
        )
    certificate = load_static_feasibility_certificate(
        certificate_binding["path"],
        expected_sha256=str(certificate_binding["sha256"]),
        expected_certificate_id=str(certificate_binding["certificate_id"]),
        expected_capacity_generation=capacity_generation,
        expected_base_fleet_contract_sha256=str(
            payload["base_fleet_contract_sha256"]
        ),
        expected_effective_fleet_contract_sha256=str(
            payload["effective_fleet_contract_sha256"]
        ),
        expected_additive_overlay_contract_sha256=str(
            payload["additive_overlay_contract_sha256"]
        ),
        expected_release_git_commit=expected_release_git_commit,
        expected_source_tree_sha256=str(payload["source_tree_sha256"]),
        expected_dispatcher_source_sha256=str(
            payload["dispatcher_source_sha256"]
        ),
        expected_qualification_runner_source_sha256=(
            str(payload["qualification_runner_source_sha256"])
        ),
    )
    base_count = _positive_integer(
        payload.get("base_active_logical_replicas"),
        field="base_active_logical_replicas",
        minimum=BASE_LOGICAL_REPLICAS,
    )
    base_gpus = _positive_integer(
        payload.get("base_active_gpus"),
        field="base_active_gpus",
        minimum=BASE_ACTIVE_GPUS,
    )
    additive_count = _positive_integer(
        payload.get("additive_reserved_logical_replicas"),
        field="additive_reserved_logical_replicas",
    )
    additive_gpus = _positive_integer(
        payload.get("additive_reserved_gpus"),
        field="additive_reserved_gpus",
    )
    additive_tp1 = _positive_integer(
        payload.get("additive_reserved_tp1_replicas"),
        field="additive_reserved_tp1_replicas",
    )
    additive_tp2 = _positive_integer(
        payload.get("additive_reserved_tp2_replicas"),
        field="additive_reserved_tp2_replicas",
    )
    warm_count = _positive_integer(
        payload.get("retained_warm_turnover_job_elements"),
        field="retained_warm_turnover_job_elements",
        minimum=1,
    )
    warm_gpus = _positive_integer(
        payload.get("retained_warm_turnover_gpus"),
        field="retained_warm_turnover_gpus",
        minimum=1,
    )
    base_topology = _validated_topology(
        payload.get("base_active_topology"),
        field="base_active_topology",
        expected_count=base_count,
        expected_gpus=base_gpus,
    )
    additive_topology = _validated_topology(
        payload.get("additive_reserved_topology"),
        field="additive_reserved_topology",
        expected_count=additive_count,
        expected_gpus=additive_gpus,
    )
    effective_topology = _validated_topology(
        payload.get("effective_active_topology"),
        field="effective_active_topology",
        expected_count=effective_logical_replicas,
        expected_gpus=effective_active_gpus,
    )
    warm_topology = _validated_topology(
        payload.get("retained_warm_turnover_topology"),
        field="retained_warm_turnover_topology",
        expected_count=warm_count,
        expected_gpus=warm_gpus,
    )
    base_ids = {str(row["shape_id"]) for row in base_topology}
    additive_ids = {str(row["shape_id"]) for row in additive_topology}
    effective_by_id = {
        str(row["shape_id"]): row for row in effective_topology
    }
    combined_by_id = {
        str(row["shape_id"]): row
        for row in (*base_topology, *additive_topology)
    }
    accounting = payload.get("job_element_accounting")
    expected_held = (
        TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS
        - effective_logical_replicas
        - RETAINED_WARM_TURNOVER_JOB_ELEMENTS
    )
    expected_accounting = {
        "cell_job_elements": CLIENT_JOB_ELEMENTS,
        "active_server_job_elements": effective_logical_replicas,
        "warm_turnover_job_elements": RETAINED_WARM_TURNOVER_JOB_ELEMENTS,
        "controller_monitor_other_held_job_elements": expected_held,
        "total_non_cell_reserve_job_elements": (
            TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS
        ),
        "total_canary_job_elements": TOTAL_SUBMIT_HEADROOM,
    }
    attested_total_gpus = _positive_integer(
        payload.get("attested_total_gpus"),
        field="attested_total_gpus",
        minimum=effective_active_gpus + RETAINED_WARM_TURNOVER_GPUS,
    )
    if (
        payload.get("base_fleet_contract_sha256")
        != certificate.base_fleet_contract_sha256
        or payload.get("effective_fleet_contract_sha256")
        != certificate.effective_fleet_contract_sha256
        or payload.get("fleet_contract_sha256")
        != payload.get("effective_fleet_contract_sha256")
        or payload.get("active_fleet_topology_sha256")
        != payload.get("effective_active_topology_sha256")
        or base_count != BASE_LOGICAL_REPLICAS
        or base_gpus != BASE_ACTIVE_GPUS
        or base_count != certificate.base_logical_replicas
        or base_gpus != certificate.base_allocated_gpus
        or effective_logical_replicas
        != certificate.effective_logical_replicas
        or effective_active_gpus != certificate.effective_active_gpus
        or payload.get("static_feasibility_wave_passed")
        is not certificate.wave_passed
        or payload.get("static_feasibility_selected_cell_count")
        != certificate.selected_cell_count
        or payload.get("static_feasibility_target_cell_count")
        != certificate.target_cell_count
        or payload.get("static_feasibility_shortfall_cells")
        != certificate.shortfall_cells
        or payload.get("static_feasibility_configured_client_ceiling")
        != certificate.target_cell_count
        or payload.get("static_feasibility_certified_saturation_target")
        != certificate.selected_cell_count
        or payload.get("cell_ceiling") != certificate.target_cell_count
        or additive_count
        != additive_tp1 + additive_tp2
        or additive_count
        != certificate.additive_tp1_logical_replicas
        + certificate.additive_tp2_logical_replicas
        or additive_gpus != additive_tp1 + 2 * additive_tp2
        or additive_gpus != certificate.additive_allocated_gpus
        or additive_tp1 != certificate.additive_tp1_logical_replicas
        or additive_tp2 != certificate.additive_tp2_logical_replicas
        or effective_logical_replicas != base_count + additive_count
        or effective_active_gpus != base_gpus + additive_gpus
        or (
            capacity_generation == 1
            and (
                effective_logical_replicas != BASE_LOGICAL_REPLICAS
                or effective_active_gpus != PRODUCTION_ACTIVE_GPUS
                or additive_count != 0
                or additive_gpus != 0
                or additive_tp1 != 0
                or additive_tp2 != 0
                or payload.get("effective_fleet_contract_sha256")
                != payload.get("base_fleet_contract_sha256")
                or payload.get("additive_overlay_contract_path")
                != payload.get("effective_fleet_contract_path")
                or payload.get("additive_overlay_contract_sha256")
                != payload.get("base_fleet_contract_sha256")
            )
        )
        or (
            capacity_generation > 1
            and (additive_count < 1 or additive_gpus < 1)
        )
        or base_ids.intersection(additive_ids)
        or effective_by_id != combined_by_id
        or warm_count != RETAINED_WARM_TURNOVER_JOB_ELEMENTS
        or warm_gpus != RETAINED_WARM_TURNOVER_GPUS
        or payload.get("retained_warm_turnover_tp1_allocations") != 2
        or payload.get("retained_warm_turnover_tp2_allocations") != 1
        or sorted(int(row["gpus"]) for row in warm_topology) != [1, 1, 2]
        or attested_total_gpus != effective_active_gpus + warm_gpus
        or (
            capacity_generation == 1
            and attested_total_gpus != PRODUCTION_ATTESTED_GPUS
        )
        or payload.get("active_gpus") != effective_active_gpus
        or payload.get("warm_headroom_gpus") != warm_gpus
        or not isinstance(accounting, dict)
        or set(accounting) != _JOB_ELEMENT_ACCOUNTING_FIELDS
        or accounting != expected_accounting
        or expected_held < 0
        or (
            capacity_generation == 1
            and expected_held
            != PREQUALIFICATION_HELD_NON_CELL_JOB_ELEMENTS
        )
        or any(
            payload.get(hash_field) != _sha256_value(payload.get(value_field))
            for value_field, hash_field in (
                ("base_active_topology", "base_active_topology_sha256"),
                (
                    "additive_reserved_topology",
                    "additive_reserved_topology_sha256",
                ),
                (
                    "effective_active_topology",
                    "effective_active_topology_sha256",
                ),
                (
                    "retained_warm_turnover_topology",
                    "retained_warm_turnover_topology_sha256",
                ),
            )
        )
    ):
        raise ProtectedCapacityError(
            "protected-capacity fleet/certificate/topology envelope is inconsistent"
        )
    preempt_type = payload.get("preempt_type")
    if preempt_type not in {"preempt/partition_prio", "preempt/qos"}:
        raise ProtectedCapacityError(
            "protected-capacity marker has unsupported PreemptType"
        )
    server_placements = _placements(
        payload["scientific_server_placements"],
        role="server",
        preempt_type=str(preempt_type),
    )
    client_placements = _placements(
        payload["scientific_client_placements"],
        role="client",
        preempt_type=str(preempt_type),
    )
    qos_contracts = _qos_contracts(
        payload["scientific_qos_contracts"],
        expected_running_jobs=expected_running_jobs,
    )
    qos_contract_by_name = {
        str(row["qos"]): row for row in qos_contracts
    }
    if (
        len(server_placements) != 1
        or sum(row.capacity["base_active_gpus"] for row in server_placements)
        != base_gpus
        or sum(
            row.capacity["reserved_additive_gpus"]
            for row in server_placements
        )
        != additive_gpus
        or sum(
            row.capacity["effective_active_gpus"]
            for row in server_placements
        )
        != effective_active_gpus
        or sum(
            row.capacity["retained_warm_turnover_gpus"]
            for row in server_placements
        )
        != warm_gpus
        or sum(
            row.capacity["attested_total_gpus"]
            for row in server_placements
        )
        != attested_total_gpus
        or any(
            row.capacity["effective_active_gpus"]
            + row.capacity["retained_warm_turnover_gpus"]
            > row.capacity["attested_total_gpus"]
            or row.capacity["partition_gpus"]
            < row.capacity["attested_total_gpus"]
            or row.capacity["base_active_gpus"]
            + row.capacity["reserved_additive_gpus"]
            != row.capacity["effective_active_gpus"]
            for row in server_placements
        )
        or sum(row.capacity["slots"] for row in client_placements)
        != _positive_integer(
            payload.get("cell_ceiling"),
            field="cell_ceiling",
            minimum=CLIENT_JOB_ELEMENTS,
        )
        or sum(row.capacity["cpus"] for row in client_placements)
        != _positive_integer(
            payload.get("cpu"), field="cpu", minimum=CLIENT_JOB_ELEMENTS
        )
        or sum(row.capacity["memory_mib"] for row in client_placements)
        != _positive_integer(
            payload.get("memory_mib"),
            field="memory_mib",
            minimum=CLIENT_JOB_ELEMENTS * 4096,
        )
        or sum(row.capacity["reserve_jobs"] for row in client_placements)
        != _positive_integer(
            payload.get("reserve_jobs"),
            field="reserve_jobs",
            minimum=TOTAL_NON_CELL_RESERVE_JOB_ELEMENTS,
        )
        or sum(row.capacity["submit_headroom"] for row in client_placements)
        != _positive_integer(
            payload.get("submit_headroom"),
            field="submit_headroom",
            minimum=TOTAL_SUBMIT_HEADROOM,
        )
        or {str(row["qos"]) for row in qos_contracts}
        != {
            row.qos
            for row in (*server_placements, *client_placements)
        }
        or any(
            int(qos_contract_by_name[row.qos]["required_wall_seconds"])
            != MIN_SCIENTIFIC_WALL_SECONDS
            for row in server_placements
        )
        or any(
            int(qos_contract_by_name[row.qos]["required_wall_seconds"])
            < CLIENT_WALL_SECONDS
            for row in client_placements
        )
    ):
        raise ProtectedCapacityError(
            "protected-capacity placement totals differ from the marker envelope"
        )
    return ProtectedCapacityContract(
        path=marker_path,
        sha256=digest,
        marker_id=marker_id,
        release_git_commit=expected_release_git_commit,
        release_tag_object=str(payload["release_tag_object"]),
        scheduler_evidence_id=str(payload["scheduler_evidence_id"]),
        scheduler_evidence_sha256=str(payload["scheduler_evidence_sha256"]),
        canary_id=str(payload["canary_id"]),
        canary_evidence_sha256=str(payload["canary_evidence_sha256"]),
        capacity_generation=capacity_generation,
        base_fleet_contract_path=bound_paths["base_fleet_contract_path"],
        base_fleet_contract_sha256=str(
            payload["base_fleet_contract_sha256"]
        ),
        effective_fleet_contract_path=bound_paths[
            "effective_fleet_contract_path"
        ],
        effective_fleet_contract_sha256=str(
            payload["effective_fleet_contract_sha256"]
        ),
        additive_overlay_contract_path=bound_paths[
            "additive_overlay_contract_path"
        ],
        additive_overlay_contract_sha256=str(
            payload["additive_overlay_contract_sha256"]
        ),
        static_feasibility_certificate_path=certificate.path,
        static_feasibility_certificate_sha256=certificate.sha256,
        static_feasibility_certificate_id=certificate.certificate_id,
        static_feasibility_wave_passed=certificate.wave_passed,
        static_feasibility_selected_cell_count=certificate.selected_cell_count,
        static_feasibility_target_cell_count=certificate.target_cell_count,
        static_feasibility_shortfall_cells=certificate.shortfall_cells,
        static_feasibility_configured_client_ceiling=(
            certificate.target_cell_count
        ),
        static_feasibility_certified_saturation_target=(
            certificate.selected_cell_count
        ),
        fleet_contract_sha256=str(payload["fleet_contract_sha256"]),
        active_fleet_topology_sha256=str(
            payload["active_fleet_topology_sha256"]
        ),
        base_active_logical_replicas=base_count,
        base_active_gpus=base_gpus,
        additive_reserved_logical_replicas=additive_count,
        additive_reserved_gpus=additive_gpus,
        effective_active_logical_replicas=effective_logical_replicas,
        effective_active_gpus=effective_active_gpus,
        retained_warm_turnover_job_elements=warm_count,
        retained_warm_turnover_gpus=warm_gpus,
        attested_total_gpus=attested_total_gpus,
        job_element_accounting=dict(accounting),
        preempt_type=str(preempt_type),
        capacity_source=str(payload["capacity_source"]),
        scheduler_cluster=str(payload["scheduler_cluster"]),
        scheduler_account=str(payload["scheduler_account"]),
        scheduler_user=str(payload["scheduler_user"]),
        scheduler_max_jobs=(
            None
            if payload["scheduler_max_jobs"] is None
            else int(payload["scheduler_max_jobs"])
        ),
        scheduler_max_submit_jobs=int(
            payload["scheduler_max_submit_jobs"]
        ),
        running_scientific_jobs=expected_running_jobs,
        scientific_qos_contracts=qos_contracts,
        partition_cpus=int(payload["partition_cpus"]),
        partition_memory_mib=int(payload["partition_memory_mib"]),
        partition_gpus=int(payload["partition_gpus"]),
        server_placements=server_placements,
        client_placements=client_placements,
    )


def authorize_fleet(
    fleet: Any,
    contract: ProtectedCapacityContract,
) -> None:
    """Require explicit QOS and marker coverage for every scientific replica."""

    replicas = getattr(fleet, "replicas", None)
    if not isinstance(replicas, tuple) or not replicas:
        raise ProtectedCapacityError("frozen serving fleet is missing replicas")
    fleet_sha256 = getattr(fleet, "sha256", None)
    fleet_path = getattr(fleet, "path", None)
    try:
        resolved_fleet_path = Path(fleet_path).expanduser().resolve()
    except (TypeError, OSError, RuntimeError) as exc:
        raise ProtectedCapacityError(
            "effective serving fleet has no exact contract path"
        ) from exc
    required_total_gpus = sum(
        int(getattr(replica, "gpus_per_replica", 0))
        for replica in replicas
    )
    if (
        fleet_sha256 != contract.effective_fleet_contract_sha256
        or fleet_sha256 != contract.fleet_contract_sha256
        or resolved_fleet_path != contract.effective_fleet_contract_path
        or len(replicas) != contract.effective_active_logical_replicas
        or required_total_gpus != contract.effective_active_gpus
    ):
        raise ProtectedCapacityError(
            "protected-capacity marker is not bound to the exact effective fleet"
        )
    active_by_placement: dict[tuple[str, str], int] = {}
    for replica in replicas:
        partition = getattr(replica, "partition", None)
        qos = getattr(replica, "qos", None)
        if not isinstance(partition, str) or not isinstance(qos, str) or not qos:
            raise ProtectedCapacityError(
                f"scientific server {getattr(replica, 'replica_id', '<unknown>')} "
                "has no explicit partition/QOS placement"
            )
        contract.placement(role="server", partition=partition, qos=qos)
        time_limit = getattr(replica, "time_limit", None)
        if time_limit is not None:
            try:
                time_limit_seconds = scheduler_safety.slurm_time_limit_seconds(
                    str(time_limit)
                )
            except scheduler_safety.SchedulerSafetyError as exc:
                raise ProtectedCapacityError(
                    f"scientific server "
                    f"{getattr(replica, 'replica_id', '<unknown>')} has an "
                    f"invalid walltime: {exc}"
                ) from exc
            if time_limit_seconds != MIN_SCIENTIFIC_WALL_SECONDS:
                raise ProtectedCapacityError(
                    f"scientific server "
                    f"{getattr(replica, 'replica_id', '<unknown>')} must use "
                    "the exact protected 24-hour walltime"
                )
        active_by_placement[(partition, qos)] = (
            active_by_placement.get((partition, qos), 0)
            + int(getattr(replica, "gpus_per_replica", 0))
        )
    for identity, required_gpus in active_by_placement.items():
        placement = contract.placement(
            role="server", partition=identity[0], qos=identity[1]
        )
        authorized_active_gpus = placement.capacity[
            "effective_active_gpus"
        ]
        if authorized_active_gpus != required_gpus:
            raise ProtectedCapacityError(
                f"protected placement {identity[0]}/{identity[1]} authorizes "
                f"{authorized_active_gpus} effective active GPUs, "
                f"but the fleet requires {required_gpus}"
            )
    unused_placements = {
        (placement.partition, placement.qos)
        for placement in contract.server_placements
    } - set(active_by_placement)
    if unused_placements:
        raise ProtectedCapacityError(
            "protected-capacity marker contains active server placement capacity "
            f"outside the exact effective fleet: {sorted(unused_placements)}"
        )


def authorize_client(
    contract: ProtectedCapacityContract,
    *,
    partition: str,
    qos: str,
    required_slots: int,
    required_reserve_jobs: int,
) -> ProtectedPlacement:
    placement = contract.placement(role="client", partition=partition, qos=qos)
    required_slots = _positive_integer(
        required_slots, field="required client slots", minimum=1
    )
    required_reserve_jobs = _positive_integer(
        required_reserve_jobs, field="required client reserve", minimum=0
    )
    capacity = placement.capacity
    if (
        capacity["slots"] < required_slots
        or capacity["cpus"] < required_slots
        or capacity["memory_mib"] < required_slots * 4096
        or capacity["reserve_jobs"] < required_reserve_jobs
        or capacity["submit_headroom"] < required_slots + required_reserve_jobs
    ):
        raise ProtectedCapacityError(
            f"protected client placement {partition}/{qos} cannot authorize "
            f"{required_slots} cells plus {required_reserve_jobs} reserved jobs"
        )
    return placement


def _invoke(
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    argv: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    try:
        result = (
            subprocess.run(
                list(argv),
                check=False,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            if runner is None
            else runner(list(argv), timeout=30.0)
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProtectedCapacityError(
            f"protected placement live query failed for {list(argv)!r}: {exc}"
        ) from exc
    if (
        not isinstance(result, subprocess.CompletedProcess)
        or result.returncode != 0
        or not isinstance(result.stdout, str)
        or not isinstance(result.stderr, str)
    ):
        raise ProtectedCapacityError(
            f"protected placement live query failed for {list(argv)!r}: "
            f"rc={getattr(result, 'returncode', 'invalid')} "
            f"{str(getattr(result, 'stderr', ''))[:500]}"
        )
    return result


def _validate_live_qos(
    qos: str,
    *,
    contract: ProtectedCapacityContract,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    preempt_type: str,
) -> None:
    argv = [
        "sacctmgr",
        "-nP",
        "show",
        "qos",
        qos,
        (
            "format=Name,PreemptMode,MaxJobsPerUser,"
            "MaxSubmitJobsPerUser,MaxTRESPerUser,MaxWall"
        ),
    ]
    raw = _invoke(runner, argv).stdout
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ProtectedCapacityError(
            f"live QOS query for {qos!r} did not return exactly one row"
        )
    fields = lines[0].split("|")
    qos_mode = fields[1] if len(fields) == 6 else ""
    effectively_safe = qos_mode.upper() == "OFF" or (
        preempt_type == "preempt/partition_prio"
        and qos_mode.lower() == "cluster"
    )
    qos_contract = _qos_contract_for(contract, qos)
    if (
        len(fields) != 6
        or fields[0] != qos
        or not effectively_safe
        or _optional_limit(
            fields[2], description=f"{qos} MaxJobsPerUser"
        )
        != qos_contract["max_jobs_per_user"]
        or _optional_limit(
            fields[3], description=f"{qos} MaxSubmitJobsPerUser"
        )
        != qos_contract["max_submit_jobs_per_user"]
        or _optional_slurm_time_seconds(
            fields[5], description=f"{qos} MaxWall"
        )
        != qos_contract["max_wall_seconds"]
    ):
        raise ProtectedCapacityError(
            f"scientific QOS {qos!r} drifted from its sealed "
            "preemption/running/submit/walltime contract"
        )


def verify_live_placements(
    contract: ProtectedCapacityContract,
    *,
    role: str,
    placements: Sequence[tuple[str, str]],
    required_time_limits_seconds: Mapping[str, int],
    trusted_scientific_job_provenance: (
        TrustedScientificJobProvenance | Mapping[str, Any] | None
    ) = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Requery and reject partition/QOS drift immediately before ``sbatch``."""

    unique = sorted(set(placements))
    if not unique or len(unique) != len(placements):
        raise ProtectedCapacityError(
            f"scientific {role} live placement list is empty or ambiguous"
        )
    for partition, qos in unique:
        contract.placement(role=role, partition=partition, qos=qos)
    partitions = sorted({partition for partition, _qos in unique})
    if set(required_time_limits_seconds) != set(partitions):
        raise ProtectedCapacityError(
            "live protected-placement time requirements differ from partitions"
        )
    exact_wall = (
        MIN_SCIENTIFIC_WALL_SECONDS
        if role == "server"
        else CLIENT_WALL_SECONDS
        if role == "client"
        else None
    )
    if exact_wall is None or any(
        not isinstance(seconds, int)
        or isinstance(seconds, bool)
        or seconds != exact_wall
        for seconds in required_time_limits_seconds.values()
    ):
        raise ProtectedCapacityError(
            f"scientific {role} placement must use the exact protected "
            f"{'24-hour' if role == 'server' else '12-hour'} walltime"
        )
    try:
        evidence = scheduler_safety.capture_scheduler_safety_evidence(
            partitions,
            runner=runner,
        )
        policy = scheduler_safety.validate_scheduler_safety_evidence(
            evidence,
            expected_partitions=partitions,
            required_time_limits_seconds=required_time_limits_seconds,
        )
    except scheduler_safety.SchedulerSafetyError as exc:
        raise ProtectedCapacityError(
            f"scientific {role} partition drifted from protected contract: {exc}"
        ) from exc
    if policy["preemptible_partitions"]:
        raise ProtectedCapacityError(
            f"scientific {role} partition became preemptible"
        )
    if policy["preempt_type"] != contract.preempt_type:
        raise ProtectedCapacityError(
            "live scheduler PreemptType drifted from the sealed "
            "protected-capacity contract"
        )
    live_partition_inventories: list[dict[str, Any]] = []
    if role == "server":
        for partition, qos in unique:
            placement = contract.placement(
                role="server",
                partition=partition,
                qos=qos,
            )
            inventory = _partition_inventory(
                evidence,
                partition=partition,
                qos=qos,
            )
            expected_inventory = {
                "cpus": placement.capacity["partition_cpus"],
                "memory_mib": placement.capacity["partition_memory_mib"],
                "gpus": placement.capacity["partition_gpus"],
                "nodes": placement.capacity["partition_nodes"],
                "state": "UP",
            }
            observed_inventory = {
                field: inventory[field] for field in expected_inventory
            }
            if (
                observed_inventory != expected_inventory
                or placement.capacity["effective_active_gpus"]
                != contract.effective_active_gpus
                or placement.capacity["retained_warm_turnover_gpus"]
                != RETAINED_WARM_TURNOVER_GPUS
                or placement.capacity["attested_total_gpus"]
                != contract.attested_total_gpus
                or inventory["gpus"] < contract.attested_total_gpus
            ):
                raise ProtectedCapacityError(
                    "live protected server partition TRES/node inventory "
                    "differs from the sealed effective-active plus 4-warm capacity "
                    "authority"
                )
            live_partition_inventories.append(
                {
                    "partition": partition,
                    "qos": qos,
                    **observed_inventory,
                }
            )
    qoses = sorted({qos for _partition, qos in unique})
    for qos in qoses:
        _validate_live_qos(
            qos,
            contract=contract,
            runner=runner,
            preempt_type=str(policy["preempt_type"]),
        )
    association_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "assoc",
        f"user={contract.scheduler_user}",
        "format=Cluster,Account,User,QOS,MaxJobs,MaxSubmitJobs",
    ]
    association_row = _single_pipe_row(
        _invoke(runner, association_argv).stdout,
        width=6,
        description="live protected association",
    )
    if (
        association_row[:3]
        != [
            contract.scheduler_cluster,
            contract.scheduler_account,
            contract.scheduler_user,
        ]
        or not set(qoses).issubset(
            set(filter(None, association_row[3].split(",")))
        )
        or _optional_limit(
            association_row[4],
            description="association MaxJobs",
        )
        != contract.scheduler_max_jobs
        or _optional_limit(
            association_row[5],
            description="association MaxSubmitJobs",
        )
        != contract.scheduler_max_submit_jobs
    ):
        raise ProtectedCapacityError(
            "live protected association running/submit authority drifted"
        )
    captured_timestamp = time.time()
    queue_source, accounting_sources = (
        _capture_live_user_occupancy_sources(
            contract,
            runner=runner,
            captured_timestamp=captured_timestamp,
        )
    )
    occupancy_rows = _parse_live_user_occupancy_sources(
        contract,
        queue_source=queue_source,
        accounting_sources=accounting_sources,
        captured_timestamp=captured_timestamp,
    )
    residual = _validate_residual_max_jobs(
        contract,
        rows=occupancy_rows,
        trusted_scientific_job_provenance=(
            trusted_scientific_job_provenance
        ),
        qoses=qoses,
        validation_timestamp=captured_timestamp,
    )
    return {
        "marker_id": contract.marker_id,
        "marker_sha256": contract.sha256,
        "role": role,
        "placements": [
            {"partition": partition, "qos": qos} for partition, qos in unique
        ],
        "scheduler_evidence_id": policy["scheduler_evidence_id"],
        "scheduler_policy_id": policy["policy_id"],
        "live_partition_inventories": live_partition_inventories,
        **residual,
    }


def _single_pipe_row(
    raw_output: str,
    *,
    width: int,
    description: str,
) -> list[str]:
    if not isinstance(raw_output, str) or "\x00" in raw_output:
        raise ProtectedCapacityError(f"{description} is not safe text")
    rows = [
        line.strip().split("|")
        for line in raw_output.splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or len(rows[0]) != width:
        raise ProtectedCapacityError(
            f"{description} must contain exactly one {width}-field row"
        )
    return rows[0]


def _optional_limit(value: str, *, description: str) -> int | None:
    if value == "":
        return None
    if not value.isdigit():
        raise ProtectedCapacityError(f"{description} is malformed")
    return int(value)


def _optional_slurm_time_seconds(
    value: str, *, description: str
) -> int | None:
    raw = value.strip()
    if not raw or raw.upper() in {"UNLIMITED", "INFINITE"}:
        return None
    try:
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
            raise ValueError
        result = (
            days * 86_400
            + hours * 3_600
            + minutes * 60
            + seconds
        )
    except ValueError as exc:
        raise ProtectedCapacityError(
            f"{description} is malformed"
        ) from exc
    if result <= 0:
        raise ProtectedCapacityError(f"{description} must be positive")
    return result


def _qos_contract_for(
    contract: ProtectedCapacityContract, qos: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in contract.scientific_qos_contracts
        if row["qos"] == qos
    ]
    if len(matches) != 1:
        raise ProtectedCapacityError(
            f"scientific QOS {qos!r} lacks one sealed capacity contract"
        )
    return matches[0]


def _memory_mib(value: str, *, description: str) -> int:
    match = re.fullmatch(r"([0-9]+)([KMGTP]?)", value, re.IGNORECASE)
    if match is None:
        raise ProtectedCapacityError(f"{description} is malformed")
    amount = int(match.group(1))
    factor = {
        "": 1,
        "K": 1 / 1024,
        "M": 1,
        "G": 1024,
        "T": 1024**2,
        "P": 1024**3,
    }[match.group(2).upper()]
    converted = amount * factor
    if converted != int(converted) or converted < 0:
        raise ProtectedCapacityError(f"{description} is not integral MiB")
    return int(converted)


def _key_value_fields(raw_output: str, *, description: str) -> dict[str, str]:
    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ProtectedCapacityError(
            f"{description} must contain one scheduler record"
        )
    try:
        tokens = shlex.split(lines[0], posix=True)
    except (TypeError, ValueError) as exc:
        raise ProtectedCapacityError(
            f"cannot parse {description}: {exc}"
        ) from exc
    fields: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise ProtectedCapacityError(
                f"{description} duplicates scheduler field {key!r}"
            )
        fields[key] = value.strip('"')
    return fields


def _partition_inventory(
    scheduler_evidence: Mapping[str, Any],
    *,
    partition: str,
    qos: str,
) -> dict[str, Any]:
    rows = scheduler_evidence.get("partitions")
    if not isinstance(rows, list):
        raise ProtectedCapacityError(
            "live protected scheduler evidence lacks partition rows"
        )
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("partition") == partition
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("raw_output"), str):
        raise ProtectedCapacityError(
            "live protected scheduler evidence has ambiguous partition inventory"
        )
    fields = _key_value_fields(
        str(matches[0]["raw_output"]),
        description=f"partition {partition} inventory",
    )
    tres: dict[str, str] = {}
    for item in fields.get("TRES", "").split(","):
        key, separator, value = item.rpartition("=")
        if not separator or not key or not value or key in tres:
            raise ProtectedCapacityError(
                f"partition {partition} TRES inventory is malformed"
            )
        tres[key] = value
    try:
        cpus = int(tres["cpu"])
        gpus = int(tres.get("gres/gpu", tres.get("gpu", "0")))
        nodes = int(tres["node"])
        memory_mib = _memory_mib(
            tres["mem"],
            description=f"partition {partition} memory",
        )
    except (KeyError, ValueError) as exc:
        raise ProtectedCapacityError(
            f"partition {partition} inventory is incomplete"
        ) from exc
    try:
        total_cpus = (
            cpus
            if "TotalCPUs" not in fields
            else int(fields["TotalCPUs"])
        )
        total_nodes = (
            nodes
            if "TotalNodes" not in fields
            else int(fields["TotalNodes"])
        )
    except ValueError as exc:
        raise ProtectedCapacityError(
            f"partition {partition} aggregate inventory is malformed"
        ) from exc
    if (
        cpus < 0
        or gpus < 0
        or memory_mib < 0
        or nodes <= 0
        or total_cpus != cpus
        or total_nodes != nodes
        or fields.get("State") != "UP"
    ):
        raise ProtectedCapacityError(
            f"partition {partition} aggregate TRES/node/state inventory is "
            "inconsistent"
        )
    allowed_qoses = set(
        filter(None, fields.get("AllowQos", "").split(","))
    )
    if "ALL" not in allowed_qoses and qos not in allowed_qoses:
        raise ProtectedCapacityError(
            f"partition {partition} no longer allows QOS {qos}"
        )
    return {
        "cpus": cpus,
        "memory_mib": memory_mib,
        "gpus": gpus,
        "nodes": nodes,
        "state": fields["State"],
        "allow_qos": sorted(allowed_qoses),
    }


def _source_record(
    *,
    argv: Sequence[str],
    raw_output: str,
) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "raw_output": raw_output,
        "raw_output_sha256": hashlib.sha256(
            raw_output.encode("utf-8")
        ).hexdigest(),
    }


def _validate_source_record(
    value: Any,
    *,
    argv: Sequence[str],
    description: str,
) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"argv", "raw_output", "raw_output_sha256"}
        or value.get("argv") != list(argv)
        or not isinstance(value.get("raw_output"), str)
        or value.get("raw_output_sha256")
        != hashlib.sha256(
            str(value.get("raw_output", "")).encode("utf-8")
        ).hexdigest()
    ):
        raise ProtectedCapacityError(
            f"{description} source record is invalid"
        )
    return str(value["raw_output"])


def _capture_live_user_occupancy_sources(
    contract: ProtectedCapacityContract,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None,
    captured_timestamp: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queue_argv = [
        "squeue",
        "-h",
        "-r",
        "-u",
        contract.scheduler_user,
        "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,RESIZING,SUSPENDED",
        "-o",
        "%i|%T|%a|%q",
    ]
    queue_raw = _invoke(runner, queue_argv).stdout
    queue_rows = [
        line.strip().split("|")
        for line in queue_raw.splitlines()
        if line.strip()
    ]
    if any(len(row) != 4 for row in queue_rows):
        raise ProtectedCapacityError(
            "live protected occupancy squeue output is malformed"
        )
    identities = sorted(row[0] for row in queue_rows)
    if (
        any(
            re.fullmatch(r"[0-9]+(?:_[0-9]+)?", identity) is None
            for identity in identities
        )
        or len(identities) != len(set(identities))
    ):
        raise ProtectedCapacityError(
            "live protected occupancy squeue identities are ambiguous"
        )
    chunks = [
        identities[index : index + 128]
        for index in range(0, len(identities), 128)
    ]
    if chunks:
        accounting_argvs = [
            [
                "sacct",
                "-nP",
                "-X",
                "--array",
                "-j",
                ",".join(chunk),
                "-o",
                "JobID,State,Account,QOS",
            ]
            for chunk in chunks
        ]
    else:
        accounting_argvs = [
            [
                "sacct",
                "-nP",
                "-X",
                "--array",
                "-u",
                contract.scheduler_user,
                "-S",
                datetime.fromtimestamp(captured_timestamp).strftime(
                    "%Y-%m-%d"
                ),
                "-o",
                "JobID,State,Account,QOS",
            ]
        ]
    accounting_sources: list[dict[str, Any]] = []
    for argv in accounting_argvs:
        raw = _invoke(runner, argv).stdout
        accounting_sources.append(
            _source_record(argv=argv, raw_output=raw)
        )
    return (
        _source_record(argv=queue_argv, raw_output=queue_raw),
        accounting_sources,
    )


def _parse_live_user_occupancy_sources(
    contract: ProtectedCapacityContract,
    *,
    queue_source: Any,
    accounting_sources: Any,
    captured_timestamp: float,
) -> list[dict[str, str]]:
    queue_argv = [
        "squeue",
        "-h",
        "-r",
        "-u",
        contract.scheduler_user,
        "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,RESIZING,SUSPENDED",
        "-o",
        "%i|%T|%a|%q",
    ]
    queue_raw = _validate_source_record(
        queue_source,
        argv=queue_argv,
        description="live protected occupancy squeue",
    )
    queued: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(queue_raw.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 4:
            raise ProtectedCapacityError(
                f"live protected occupancy squeue row {line_number} is malformed"
            )
        job_id, state, account, qos = fields
        if (
            re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) is None
            or state.upper() not in _LIVE_JOB_STATES
            or _PLACEMENT_RE.fullmatch(account) is None
            or _PLACEMENT_RE.fullmatch(qos) is None
            or job_id in queued
        ):
            raise ProtectedCapacityError(
                f"live protected occupancy squeue row {line_number} is "
                "ambiguous"
            )
        queued[job_id] = {
            "job_id": job_id,
            "state": state.upper(),
            "account": account,
            "qos": qos,
        }
    identities = sorted(queued)
    chunks = [
        identities[index : index + 128]
        for index in range(0, len(identities), 128)
    ]
    if chunks:
        expected_argvs = [
            [
                "sacct",
                "-nP",
                "-X",
                "--array",
                "-j",
                ",".join(chunk),
                "-o",
                "JobID,State,Account,QOS",
            ]
            for chunk in chunks
        ]
    else:
        expected_argvs = [
            [
                "sacct",
                "-nP",
                "-X",
                "--array",
                "-u",
                contract.scheduler_user,
                "-S",
                datetime.fromtimestamp(captured_timestamp).strftime(
                    "%Y-%m-%d"
                ),
                "-o",
                "JobID,State,Account,QOS",
            ]
        ]
    if (
        not isinstance(accounting_sources, list)
        or len(accounting_sources) != len(expected_argvs)
    ):
        raise ProtectedCapacityError(
            "live protected occupancy sacct source cardinality drifted"
        )
    accounted: dict[str, dict[str, str]] = {}
    for source_index, (source, argv) in enumerate(
        zip(accounting_sources, expected_argvs, strict=True)
    ):
        raw = _validate_source_record(
            source,
            argv=argv,
            description=f"live protected occupancy sacct {source_index}",
        )
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split("|")]
            if len(fields) != 4:
                raise ProtectedCapacityError(
                    "live protected occupancy sacct row "
                    f"{source_index}:{line_number} is malformed"
                )
            job_id, state, account, qos = fields
            if "." in job_id:
                continue
            if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) is None:
                raise ProtectedCapacityError(
                    "live protected occupancy sacct identity is malformed"
                )
            normalized_state = state.upper().split()[0].rstrip("+")
            if normalized_state not in _LIVE_JOB_STATES:
                continue
            if (
                _PLACEMENT_RE.fullmatch(account) is None
                or _PLACEMENT_RE.fullmatch(qos) is None
                or job_id in accounted
            ):
                raise ProtectedCapacityError(
                    "live protected occupancy sacct row is ambiguous"
                )
            accounted[job_id] = {
                "job_id": job_id,
                "state": normalized_state,
                "account": account,
                "qos": qos,
            }
    queue_array_parents = {
        job_id.split("_", 1)[0]
        for job_id in queued
        if "_" in job_id
    }
    unexplained = set(accounted) - set(queued) - queue_array_parents
    if unexplained:
        raise ProtectedCapacityError(
            "live protected occupancy sacct contains active identities absent "
            "from complete squeue truth"
        )
    for job_id, row in queued.items():
        accounting = accounted.get(job_id)
        if accounting != row:
            raise ProtectedCapacityError(
                f"live protected occupancy squeue/sacct disagree for {job_id}"
            )
    return [queued[job_id] for job_id in identities]


def _validate_residual_max_jobs(
    contract: ProtectedCapacityContract,
    *,
    rows: Sequence[Mapping[str, str]],
    trusted_scientific_job_provenance: (
        TrustedScientificJobProvenance | Mapping[str, Any] | None
    ),
    qoses: Collection[str],
    validation_timestamp: float,
) -> dict[str, Any]:
    qos_names = sorted(set(qoses))
    if not qos_names:
        raise ProtectedCapacityError(
            "residual MaxJobs validation has no scientific QOS"
        )
    qos_contracts = {
        qos: _qos_contract_for(contract, qos) for qos in qos_names
    }
    finite = (
        contract.scheduler_max_jobs is not None
        or contract.scheduler_max_submit_jobs is not None
        or any(
            row["max_jobs_per_user"] is not None
            or row["max_submit_jobs_per_user"] is not None
            for row in qos_contracts.values()
        )
    )
    if trusted_scientific_job_provenance is None:
        if finite:
            raise ProtectedCapacityError(
                "finite MaxJobs/MaxSubmitJobs authority requires exact, fresh "
                "caller-reconciled scientific provenance"
            )
        provenance = build_trusted_scientific_job_provenance(
            scheduler_job_states={},
            scheduler_captured_timestamp=validation_timestamp,
            trusted_cell_job_ids=(),
            trusted_fleet_job_ids=(),
            dispatcher_ledger_updated_timestamp=None,
            exact_cell_quiescence=True,
            dispatcher_provenance_id=hashlib.sha256(b"unbounded").hexdigest(),
            fleet_provenance_id=hashlib.sha256(b"unbounded-fleet").hexdigest(),
            fleet_contract_sha256=contract.fleet_contract_sha256,
            fleet_generation=1,
            scheduler_truth_id=hashlib.sha256(b"unbounded-truth").hexdigest(),
            now=validation_timestamp,
        )
    else:
        provenance = validate_trusted_scientific_job_provenance(
            trusted_scientific_job_provenance,
            validation_timestamp=validation_timestamp,
        )
    trusted = provenance.job_ids
    by_id = {str(row["job_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ProtectedCapacityError(
            "live protected occupancy repeats scheduler identities"
        )
    scientific_qoses = {
        str(contract_row["qos"])
        for contract_row in contract.scientific_qos_contracts
    }
    role_qoses = {
        "cell": {
            placement.qos for placement in contract.client_placements
        },
        "fleet": {
            placement.qos for placement in contract.server_placements
        },
    }
    role_ids = {
        "cell": tuple(provenance.payload["trusted_cell_job_ids"]),
        "fleet": tuple(provenance.payload["trusted_fleet_job_ids"]),
    }
    for role, job_ids in role_ids.items():
        for job_id in job_ids:
            row = by_id.get(job_id)
            if row is None or row["qos"] not in role_qoses[role]:
                raise ProtectedCapacityError(
                    "caller-reconciled trusted scientific job has a "
                    f"{role} role/QOS mismatch: {job_id}"
                )
    for job_id in trusted:
        row = by_id.get(job_id)
        if (
            row is None
            or row["state"] not in _LIVE_JOB_STATES
            or row["account"] != contract.scheduler_account
            or row["qos"] not in scientific_qoses
        ):
            raise ProtectedCapacityError(
                "caller-reconciled trusted scientific live job set is not an "
                "exact subset of current scientific account/QOS allocations"
            )
    trusted_set = set(trusted)
    trusted_running = {
        job_id
        for job_id in trusted_set
        if by_id[job_id]["state"] in _MAX_JOBS_CONSUMING_STATES
    }
    association_live = {
        str(row["job_id"])
        for row in rows
        if row["state"] in _LIVE_JOB_STATES
        and row["account"] == contract.scheduler_account
    }
    association_running = {
        str(row["job_id"])
        for row in rows
        if row["state"] in _MAX_JOBS_CONSUMING_STATES
        and row["account"] == contract.scheduler_account
    }
    unrelated_association = association_running - trusted_running
    unrelated_submitted_association = association_live - trusted_set
    if (
        contract.scheduler_max_jobs is not None
        and len(unrelated_association) + contract.running_scientific_jobs
        > contract.scheduler_max_jobs
    ):
        raise ProtectedCapacityError(
            "live association MaxJobs lacks residual capacity for the dynamic "
            "scientific allocation envelope plus contemporaneous unrelated running "
            f"jobs: unrelated={len(unrelated_association)}, "
            f"limit={contract.scheduler_max_jobs}"
        )
    if (
        contract.scheduler_max_submit_jobs is not None
        and len(unrelated_submitted_association) + 448
        > contract.scheduler_max_submit_jobs
    ):
        raise ProtectedCapacityError(
            "live association MaxSubmitJobs lacks residual capacity for 448 "
            "scientific allocations plus contemporaneous unrelated submitted "
            f"jobs: unrelated={len(unrelated_submitted_association)}, "
            f"limit={contract.scheduler_max_submit_jobs}"
        )
    unrelated_by_qos: dict[str, int] = {}
    unrelated_submitted_by_qos: dict[str, int] = {}
    for qos, qos_contract in qos_contracts.items():
        qos_running = {
            str(row["job_id"])
            for row in rows
            if row["state"] in _MAX_JOBS_CONSUMING_STATES
            and row["account"] == contract.scheduler_account
            and row["qos"] == qos
        }
        qos_live = {
            str(row["job_id"])
            for row in rows
            if row["state"] in _LIVE_JOB_STATES
            and row["account"] == contract.scheduler_account
            and row["qos"] == qos
        }
        unrelated = qos_running - trusted_running
        unrelated_submitted = qos_live - trusted_set
        unrelated_by_qos[qos] = len(unrelated)
        unrelated_submitted_by_qos[qos] = len(unrelated_submitted)
        max_jobs = qos_contract["max_jobs_per_user"]
        if (
            max_jobs is not None
            and len(unrelated) + int(qos_contract["required_running_jobs"])
            > max_jobs
        ):
            raise ProtectedCapacityError(
                f"live QOS {qos} MaxJobs lacks residual capacity for its "
                "scientific envelope plus contemporaneous unrelated running "
                f"jobs: unrelated={len(unrelated)}, limit={max_jobs}"
            )
        max_submit = qos_contract["max_submit_jobs_per_user"]
        if (
            max_submit is not None
            and len(unrelated_submitted)
            + int(qos_contract["required_submit_jobs"])
            > max_submit
        ):
            raise ProtectedCapacityError(
                f"live QOS {qos} MaxSubmitJobs lacks residual capacity for its "
                "scientific envelope plus contemporaneous unrelated submitted "
                f"jobs: unrelated={len(unrelated_submitted)}, limit={max_submit}"
            )
    return {
        "trusted_scientific_provenance_id": provenance.provenance_id,
        "trusted_scientific_live_job_ids": list(trusted),
        "trusted_scientific_running_job_ids": sorted(trusted_running),
        "live_submitted_association_job_elements": len(association_live),
        "live_running_association_job_elements": len(association_running),
        "unrelated_submitted_association_job_elements": len(
            unrelated_submitted_association
        ),
        "unrelated_running_association_job_elements": len(
            unrelated_association
        ),
        "unrelated_submitted_qos_job_elements": unrelated_submitted_by_qos,
        "unrelated_running_qos_job_elements": unrelated_by_qos,
    }


def validate_live_client_capacity_evidence(
    contract: ProtectedCapacityContract,
    evidence: Mapping[str, Any],
    *,
    partition: str,
    qos: str,
    required_time_limit_seconds: int,
) -> dict[str, Any]:
    """Reparse one live recapture against the sealed simultaneous canary.

    Blank inherited QOS limits are never treated as infinity.  CPU and memory
    authority comes only from the sealed simultaneous client canary; the live
    partition inventory must remain byte-semantically equal to its attested
    configured totals, and the exact user/account association must retain the
    attested submit limit.
    """

    placement = contract.placement(
        role="client",
        partition=partition,
        qos=qos,
    )
    fields = {
        "schema_version",
        "protocol",
        "capacity_source",
        "marker_id",
        "marker_sha256",
        "scheduler_evidence_id",
        "canary_id",
        "partition",
        "qos",
        "scheduler_cluster",
        "scheduler_account",
        "scheduler_user",
        "scheduler_max_submit_jobs",
        "authorized_capacity",
        "captured_timestamp",
        "scheduler_policy_evidence",
        "qos_source",
        "association_source",
        "occupancy_squeue_source",
        "occupancy_sacct_sources",
        "trusted_scientific_job_provenance",
        "evidence_id",
    }
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != fields
        or evidence.get("schema_version") != 1
        or evidence.get("protocol") != LIVE_CLIENT_CAPACITY_PROTOCOL
        or evidence.get("capacity_source") != CAPACITY_SOURCE
        or evidence.get("marker_id") != contract.marker_id
        or evidence.get("marker_sha256") != contract.sha256
        or evidence.get("scheduler_evidence_id")
        != contract.scheduler_evidence_id
        or evidence.get("canary_id") != contract.canary_id
        or evidence.get("partition") != partition
        or evidence.get("qos") != qos
        or evidence.get("scheduler_cluster") != contract.scheduler_cluster
        or evidence.get("scheduler_account") != contract.scheduler_account
        or evidence.get("scheduler_user") != contract.scheduler_user
        or evidence.get("scheduler_max_submit_jobs")
        != contract.scheduler_max_submit_jobs
        or evidence.get("authorized_capacity")
        != dict(placement.capacity)
        or not isinstance(evidence.get("captured_timestamp"), (int, float))
        or isinstance(evidence.get("captured_timestamp"), bool)
        or not math.isfinite(float(evidence["captured_timestamp"]))
        or float(evidence["captured_timestamp"]) <= 0
        or evidence.get("evidence_id")
        != _identity_sha256(
            {
                key: value
                for key, value in evidence.items()
                if key != "evidence_id"
            }
        )
    ):
        raise ProtectedCapacityError(
            "live protected client-capacity evidence envelope drifted"
        )
    scheduler_evidence = evidence.get("scheduler_policy_evidence")
    if not isinstance(scheduler_evidence, Mapping):
        raise ProtectedCapacityError(
            "live protected client-capacity scheduler evidence is absent"
        )
    if scheduler_evidence.get("captured_timestamp") != evidence["captured_timestamp"]:
        raise ProtectedCapacityError(
            "live protected client-capacity timestamps disagree"
        )
    try:
        policy = scheduler_safety.validate_scheduler_safety_evidence(
            scheduler_evidence,
            expected_partitions=[partition],
            required_time_limits_seconds={
                partition: required_time_limit_seconds
            },
        )
    except scheduler_safety.SchedulerSafetyError as exc:
        raise ProtectedCapacityError(
            f"live protected client partition policy drifted: {exc}"
        ) from exc
    if (
        policy["preemptible_partitions"]
        or policy["preempt_type"] != contract.preempt_type
    ):
        raise ProtectedCapacityError(
            "live protected client partition became preemptible"
        )
    inventory = _partition_inventory(
        scheduler_evidence,
        partition=partition,
        qos=qos,
    )
    if (
        inventory["cpus"] != contract.partition_cpus
        or inventory["memory_mib"] != contract.partition_memory_mib
        or inventory["gpus"] != contract.partition_gpus
        or inventory["cpus"] < placement.capacity["cpus"]
        or inventory["memory_mib"] < placement.capacity["memory_mib"]
    ):
        raise ProtectedCapacityError(
            "live partition inventory differs from the sealed protected canary "
            "authority"
        )

    qos_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "qos",
        qos,
        (
            "format=Name,PreemptMode,MaxJobsPerUser,"
            "MaxSubmitJobsPerUser,MaxTRESPerUser,MaxWall"
        ),
    ]
    qos_raw = _validate_source_record(
        evidence.get("qos_source"),
        argv=qos_argv,
        description="live protected QOS",
    )
    qos_row = _single_pipe_row(
        qos_raw,
        width=6,
        description="live protected QOS",
    )
    qos_mode = qos_row[1]
    qos_safe = qos_mode.upper() == "OFF" or (
        contract.preempt_type == "preempt/partition_prio"
        and qos_mode.lower() == "cluster"
    )
    if qos_row[0] != qos or not qos_safe:
        raise ProtectedCapacityError(
            "live protected QOS identity/preemption mode drifted"
        )
    qos_max_jobs = _optional_limit(
        qos_row[2],
        description=f"{qos} MaxJobsPerUser",
    )
    qos_max_submit = _optional_limit(
        qos_row[3],
        description=f"{qos} MaxSubmitJobsPerUser",
    )
    qos_max_wall = _optional_slurm_time_seconds(
        qos_row[5],
        description=f"{qos} MaxWall",
    )
    qos_contract = _qos_contract_for(contract, qos)
    if (
        qos_max_jobs != qos_contract["max_jobs_per_user"]
        or qos_max_submit
        != qos_contract["max_submit_jobs_per_user"]
        or qos_max_wall != qos_contract["max_wall_seconds"]
    ):
        raise ProtectedCapacityError(
            "live protected QOS running/submit/walltime limits drifted "
            "from the sealed canary"
        )
    if qos_row[4]:
        qos_tres: dict[str, str] = {}
        for item in qos_row[4].split(","):
            key, separator, value = item.partition("=")
            if not separator or not key or not value or key in qos_tres:
                raise ProtectedCapacityError(
                    "live protected QOS MaxTRESPerUser is malformed"
                )
            qos_tres[key] = value
        try:
            qos_cpu_limit = (
                int(qos_tres["cpu"]) if "cpu" in qos_tres else None
            )
        except ValueError as exc:
            raise ProtectedCapacityError(
                f"{qos} MaxTRESPerUser cpu is malformed"
            ) from exc
        if (
            qos_cpu_limit is not None
            and qos_cpu_limit < placement.capacity["cpus"]
        ) or (
            "mem" in qos_tres
            and _memory_mib(
                qos_tres["mem"],
                description=f"{qos} MaxTRESPerUser memory",
            )
            < placement.capacity["memory_mib"]
        ):
            raise ProtectedCapacityError(
                "live protected QOS MaxTRESPerUser fell below the sealed canary"
            )

    association_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "assoc",
        f"user={contract.scheduler_user}",
        "format=Cluster,Account,User,QOS,MaxJobs,MaxSubmitJobs",
    ]
    association_raw = _validate_source_record(
        evidence.get("association_source"),
        argv=association_argv,
        description="live protected association",
    )
    association_row = _single_pipe_row(
        association_raw,
        width=6,
        description="live protected association",
    )
    association_qoses = set(filter(None, association_row[3].split(",")))
    association_max_jobs = _optional_limit(
        association_row[4],
        description="association MaxJobs",
    )
    association_max_submit = _optional_limit(
        association_row[5],
        description="association MaxSubmitJobs",
    )
    if (
        association_row[:3]
        != [
            contract.scheduler_cluster,
            contract.scheduler_account,
            contract.scheduler_user,
        ]
        or qos not in association_qoses
        or association_max_submit != contract.scheduler_max_submit_jobs
        or association_max_submit < placement.capacity["submit_headroom"]
        or association_max_jobs != contract.scheduler_max_jobs
        or (
            association_max_jobs is not None
            and association_max_jobs < contract.running_scientific_jobs
        )
    ):
        raise ProtectedCapacityError(
            "live protected user/account/QOS association drifted"
        )
    occupancy_rows = _parse_live_user_occupancy_sources(
        contract,
        queue_source=evidence.get("occupancy_squeue_source"),
        accounting_sources=evidence.get("occupancy_sacct_sources"),
        captured_timestamp=float(evidence["captured_timestamp"]),
    )
    residual = _validate_residual_max_jobs(
        contract,
        rows=occupancy_rows,
        trusted_scientific_job_provenance=evidence.get(
            "trusted_scientific_job_provenance"
        ),
        qoses={qos},
        validation_timestamp=float(evidence["captured_timestamp"]),
    )
    return {
        "evidence_id": evidence["evidence_id"],
        "capacity_source": CAPACITY_SOURCE,
        "marker_id": contract.marker_id,
        "partition": partition,
        "qos": qos,
        "scheduler_cluster": contract.scheduler_cluster,
        "scheduler_account": contract.scheduler_account,
        "scheduler_user": contract.scheduler_user,
        "scheduler_max_submit_jobs": contract.scheduler_max_submit_jobs,
        "scheduler_max_jobs": contract.scheduler_max_jobs,
        **residual,
        "cpu_limit": placement.capacity["cpus"],
        "memory_limit_mib": placement.capacity["memory_mib"],
        "max_submit_jobs": placement.capacity["submit_headroom"],
        "partition_cpus": inventory["cpus"],
        "partition_memory_mib": inventory["memory_mib"],
        "partition_gpus": inventory["gpus"],
        "scheduler_policy_id": policy["policy_id"],
        "scheduler_policy_contract_id": policy["policy_contract_id"],
    }


def capture_live_client_capacity(
    contract: ProtectedCapacityContract,
    *,
    partition: str,
    qos: str,
    required_time_limit_seconds: int,
    trusted_scientific_job_provenance: (
        TrustedScientificJobProvenance | Mapping[str, Any] | None
    ) = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    captured_timestamp: float | None = None,
) -> dict[str, Any]:
    """Capture and validate live inherited-QOS truth against the sealed canary."""

    placement = contract.placement(
        role="client",
        partition=partition,
        qos=qos,
    )
    try:
        scheduler_evidence = (
            scheduler_safety.capture_scheduler_safety_evidence(
                [partition],
                runner=runner,
                captured_timestamp=captured_timestamp,
            )
        )
    except scheduler_safety.SchedulerSafetyError as exc:
        raise ProtectedCapacityError(
            f"cannot capture protected client partition policy: {exc}"
        ) from exc
    qos_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "qos",
        qos,
        (
            "format=Name,PreemptMode,MaxJobsPerUser,"
            "MaxSubmitJobsPerUser,MaxTRESPerUser,MaxWall"
        ),
    ]
    association_argv = [
        "sacctmgr",
        "-nP",
        "show",
        "assoc",
        f"user={contract.scheduler_user}",
        "format=Cluster,Account,User,QOS,MaxJobs,MaxSubmitJobs",
    ]
    qos_raw = _invoke(runner, qos_argv).stdout
    association_raw = _invoke(runner, association_argv).stdout
    timestamp = (
        time.time()
        if captured_timestamp is None
        else float(captured_timestamp)
    )
    queue_source, accounting_sources = (
        _capture_live_user_occupancy_sources(
            contract,
            runner=runner,
            captured_timestamp=timestamp,
        )
    )
    # Fail before publishing a reusable live authority when a finite running or
    # submit limit has no exact, fresh caller reconciliation.
    validated_provenance = (
        None
        if trusted_scientific_job_provenance is None
        else validate_trusted_scientific_job_provenance(
            trusted_scientific_job_provenance,
            validation_timestamp=timestamp,
        )
    )
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol": LIVE_CLIENT_CAPACITY_PROTOCOL,
        "capacity_source": CAPACITY_SOURCE,
        "marker_id": contract.marker_id,
        "marker_sha256": contract.sha256,
        "scheduler_evidence_id": contract.scheduler_evidence_id,
        "canary_id": contract.canary_id,
        "partition": partition,
        "qos": qos,
        "scheduler_cluster": contract.scheduler_cluster,
        "scheduler_account": contract.scheduler_account,
        "scheduler_user": contract.scheduler_user,
        "scheduler_max_submit_jobs": (
            contract.scheduler_max_submit_jobs
        ),
        "authorized_capacity": dict(placement.capacity),
        "captured_timestamp": timestamp,
        "scheduler_policy_evidence": scheduler_evidence,
        "qos_source": _source_record(argv=qos_argv, raw_output=qos_raw),
        "association_source": _source_record(
            argv=association_argv,
            raw_output=association_raw,
        ),
        "occupancy_squeue_source": queue_source,
        "occupancy_sacct_sources": accounting_sources,
        "trusted_scientific_job_provenance": (
            None
            if validated_provenance is None
            else dict(validated_provenance.payload)
        ),
    }
    evidence["evidence_id"] = _identity_sha256(evidence)
    validate_live_client_capacity_evidence(
        contract,
        evidence,
        partition=partition,
        qos=qos,
        required_time_limit_seconds=required_time_limit_seconds,
    )
    return evidence
