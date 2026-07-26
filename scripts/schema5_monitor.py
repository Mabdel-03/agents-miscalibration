#!/usr/bin/env python3
"""Generation-scoped health, semantic, and acceptance monitoring for schema-5.

Report generation is read-only.  ``--persist`` atomically publishes a report, appends a
successful validated-QID sample to the durable control plane, and raises/resolves alerts
through its existing append-only journal.  Top-level QIDs and auxiliary draws always use
separate denominators.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "src"
for candidate in (str(REPO), str(SOURCE)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog  # noqa: E402
from agents_scaling.config import DEFAULT_RESULTS_ROOT, ExperimentCell  # noqa: E402
from agents_scaling.experiment.analyze import _censor_accounting  # noqa: E402
from agents_scaling.experiment.artifact_policy import load_artifact_policy  # noqa: E402
from agents_scaling.experiment.completion import (  # noqa: E402
    CompletionState,
    CorruptArtifactError,
    canonical_coordinate_provenance_counts,
    expected_result_coordinate_count,
    get_completion_status,
    read_canonical_results,
    result_coordinate_endpoint_counts,
    singleton_coordinate_provenance,
    transport_censored_coordinate_count,
    trusted_generation_catalog_errors,
    visible_endpoint_counts_are_bounded,
)
from agents_scaling.experiment.manifest import load_manifest  # noqa: E402
from agents_scaling.experiment.result_schema import (  # noqa: E402
    TERMINATION_COMPLETED,
    TERMINATION_LENGTH_CENSORED,
    TERMINATION_PROTOCOL_CENSORED,
    TERMINATION_TRANSPORT_CENSORED,
)
from agents_scaling.experiment.transport_censor import (  # noqa: E402
    TRANSPORT_CENSOR_PROTOCOL_HASH,
    TRANSPORT_CENSOR_PROTOCOL_VERSION,
)
from agents_scaling.serving.model_contracts import load_model_contracts  # noqa: E402
from agents_scaling.serving import (  # noqa: E402
    fleet_transactions,
    protected_capacity,
    scheduler_safety,
)
from slurm.dispatch_sweeps import (  # noqa: E402
    DispatcherError as DispatcherLedgerError,
    validate_ledger_structure,
)
from slurm import schema5_control as control  # noqa: E402
from scripts import monitor_run  # noqa: E402


DEFAULT_CONFIG = REPO / "configs" / "schema5_monitoring.v1.json"
DEFAULT_STATE_DIRNAME = ".dispatcher-schema5-v1"
MONITORING_DIRNAME = "monitoring"
LATEST_POINTER_PROTOCOL = "schema5-monitor-latest-pointer-v1"
HEALTH_ALERT_KEYS = frozenset(
    {
        "monitor:scheduler",
        "monitor:controllers",
        "monitor:fleet",
        "monitor:fleet-hung",
        "monitor:qos-memory",
        "monitor:disk",
        "monitor:inodes",
        "monitor:ledger",
        "monitor:starvation",
        "monitor:semantic-stale",
        "monitor:ramp-stall",
        "monitor:hold-drain",
        "monitor:capacity-gate",
        # Dispatcher-ledger states provide the five-minute early-warning path; the
        # complete semantic scan independently confirms them every six hours.
        "monitor:corrupt",
        "monitor:permanent",
        "monitor:retryable",
    }
)
SEMANTIC_ALERT_KEYS = HEALTH_ALERT_KEYS | frozenset(
    {
        # Progress is established only by a complete semantic scan.  A five-minute
        # health poll therefore neither owns nor resolves this incident.
        "monitor:no-progress",
        "monitor:untrusted",
        "monitor:throughput",
        "monitor:context-protocol",
        "monitor:transport-censor",
    }
)


class MonitorError(RuntimeError):
    """A report cannot be made truthful from the frozen production state."""


@contextmanager
def _monitor_persist_lock(state_dir: Path):
    """Serialize independently scheduled health/semantic/daily commit boundaries."""

    root = state_dir / MONITORING_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".persist.lock"
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class AlertFinding:
    dedupe_key: str
    kind: str
    severity: str
    message: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def load_monitor_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MonitorError(f"cannot read monitoring contract {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise MonitorError("unsupported schema-5 monitoring contract")
    runs = value.get("runs")
    if not isinstance(runs, list) or not all(isinstance(row, dict) for row in runs):
        raise MonitorError("monitoring contract runs must be an array of objects")
    run_ids = [row.get("run_id") for row in runs]
    if set(run_ids) != set(control.REQUIRED_RUNS) or len(run_ids) != len(set(run_ids)):
        raise MonitorError("monitoring contract must contain exactly the schema-5 runs")
    expected_cells = sum(int(row.get("expected_cells", -1)) for row in runs)
    expected_qids = sum(int(row.get("expected_qids", -1)) for row in runs)
    if expected_cells != value.get("expected_total_cells") or expected_cells != 22_680:
        raise MonitorError("monitoring cell cardinality must be exactly 22,680")
    if expected_qids != value.get("expected_total_qids") or expected_qids != 4_524_660:
        raise MonitorError("monitoring QID cardinality must be exactly 4,524,660")
    for run in runs:
        if int(run["expected_cells"]) != control.REQUIRED_RUNS[str(run["run_id"])]:
            raise MonitorError(f"monitoring cell count drift for {run['run_id']}")
    cadence = value.get("cadence_seconds")
    if cadence != {"health": 300, "semantic": 21600, "daily": 86400}:
        raise MonitorError("monitoring cadences must be 5 minutes, 6 hours, and daily")
    if value.get("execution_deadline_seconds") != {
        "health": 240,
        "semantic": 18_000,
        "daily": 18_000,
        "terminate_grace": 30,
    }:
        raise MonitorError(
            "monitor execution deadlines must be 240 seconds and five hours "
            "with a 30-second TERM grace"
        )
    throughput = value.get("throughput")
    if not isinstance(throughput, dict) or throughput.get("minimum_qids_per_day") != 161_595:
        raise MonitorError("monitoring throughput floor must be 161,595 QIDs/day")
    if throughput.get("target_completion_days") != 28:
        raise MonitorError("monitoring completion target must be 28 days")
    material_generation = throughput.get("material_capacity_layout_generation")
    if (
        not isinstance(material_generation, int)
        or isinstance(material_generation, bool)
        or material_generation < 1
    ):
        raise MonitorError(
            "monitoring material capacity/layout generation must be a positive integer"
        )
    fleet_replicas = value.get("fleet_replicas")
    if fleet_replicas != control.EXPECTED_FLEET_PROFILES:
        raise MonitorError(
            "monitoring fleet replica layout must exactly match the canonical fleet"
        )
    alerts = value.get("alerts")
    if not isinstance(alerts, dict):
        raise MonitorError("monitoring alert policy must be an object")
    required_alerts = {
        "dispatcher_ledger_stale_seconds": 360,
        "email_retry_delays_seconds": [60, 300, 900, 3600],
        "minimum_free_inodes": 1_000_000,
        "minimum_free_inode_fraction": 0.05,
        "no_progress_semantic_scans": 2,
        "ramp_stall_seconds": {
            "24": 28_800,
            "96": 46_800,
            "192": 68_400,
        },
    }
    for field, expected in required_alerts.items():
        if alerts.get(field) != expected:
            raise MonitorError(
                f"monitoring alert setting {field!r} must be {expected!r}"
            )
    return value


def _effective_agents(cell: ExperimentCell) -> int:
    return 1 if cell.topology.value == "single_agent" else int(cell.n_agents)


def _dimension_value(cell: ExperimentCell, dimension: str) -> Any:
    values = {
        "model_size": cell.model_size,
        "reasoning_level": cell.reasoning_level.value,
        "topology": cell.topology.value,
        "agent_count": _effective_agents(cell),
        "benchmark": cell.benchmark,
        "context_share_level": cell.context_share_level.value,
        "prompt_complexity_level": cell.prompt_complexity_level,
        "seed": cell.seed,
    }
    try:
        return values[dimension]
    except KeyError as exc:
        raise MonitorError(f"unsupported throughput dimension {dimension!r}") from exc


def _stratum_key(cell: ExperimentCell, dimensions: Sequence[str]) -> str:
    return json.dumps(
        {dimension: _dimension_value(cell, dimension) for dimension in dimensions},
        sort_keys=True,
        separators=(",", ":"),
    )


def _finite_timestamp(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _record_matches_policy(
    record: Mapping[str, Any],
    cell: ExperimentCell,
    policy,
    contracts,
    *,
    release_fleet_contract_sha256: str,
    trusted_generation_tuples: frozenset[
        tuple[str, str, int, int, str]
    ],
) -> bool:
    """Apply immutable row pins even while a worker holds the cell advisory lock."""

    try:
        identity = contracts.for_size(cell.model_size)
    except Exception:
        return False
    fixed = {
        "schema_version": 5,
        "release_id": policy.release.release_id,
        "environment_hash": policy.environment.harness_sha256,
        "model_revision": identity.model_revision,
        "tokenizer_revision": identity.tokenizer_revision,
        "model_contract_sha256": policy.accepted_model_contract_sha256,
    }
    if any(record.get(field) != expected for field, expected in fixed.items()):
        return False
    if record.get("effective_context") != record.get("effective_context_limit"):
        return False
    try:
        counts = canonical_coordinate_provenance_counts(
            record.get("coordinate_provenance_counts")
        )
        endpoints = result_coordinate_endpoint_counts(record)
        expected_count = expected_result_coordinate_count(record)
    except CorruptArtifactError:
        return False
    if trusted_generation_catalog_errors(
        [record], trusted_generation_tuples
    ):
        return False
    authoritative_endpoints = counts["endpoint_generation"]
    if sum(authoritative_endpoints.values()) != expected_count:
        return False
    if not visible_endpoint_counts_are_bounded(
        authoritative_endpoints, endpoints
    ):
        return False
    expected_endpoint = (
        next(iter(authoritative_endpoints))
        if len(authoritative_endpoints) == 1
        else "mixed"
    )
    if record.get("endpoint_generation") != expected_endpoint:
        return False
    for count_field, scalar_field, integer in (
        ("fleet_contract_sha256", "fleet_contract_sha256", False),
        ("capacity_generation", "capacity_generation", True),
        ("rollout_generation", "rollout_generation", True),
    ):
        expected = singleton_coordinate_provenance(counts, count_field)
        if integer and expected is not None:
            expected = int(expected)
        if record.get(scalar_field) != expected:
            return False
    release_lineage = singleton_coordinate_provenance(
        counts, "release_fleet_contract_sha256"
    )
    return bool(
        release_lineage == release_fleet_contract_sha256
        and record.get("release_fleet_contract_sha256") == release_lineage
    )


def _empty_progress() -> dict[str, Any]:
    return {
        "expected": 0,
        "validated": 0,
        "useful_validated": 0,
        "timestamps": [],
    }


def _record_is_useful(record: Mapping[str, Any]) -> bool:
    """Return whether a trusted QID is eligible for throughput evidence.

    A top-level completed answer can still depend on an ambiguous sibling topology
    wave or self-consistency draw.  Any retained transport censor therefore excludes
    the whole QID from useful throughput while leaving trusted cardinality unchanged.
    """

    return bool(
        record.get("termination_status") != TERMINATION_TRANSPORT_CENSORED
        and transport_censored_coordinate_count([record]) == 0
    )


def collect_semantic_state(
    *,
    results_root: Path,
    state_dir: Path,
    config: Mapping[str, Any],
    now: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Perform one complete policy-aware scan of all three frozen manifests."""

    state = control.load_control(state_dir)
    pinned_runs = {
        str(row["run_id"]): row for row in state["immutable"]["runs"]
    }
    dimensions = tuple(config["throughput"]["stratum_dimensions"])
    rotation_dimensions = tuple(config["throughput"]["rotation_dimensions"])
    progress: dict[str, dict[str, Any]] = defaultdict(_empty_progress)
    rotation: dict[str, dict[str, Any]] = defaultdict(_empty_progress)
    contracts = load_model_contracts(
        state["immutable"]["model_contract_path"],
        expected_sha256=state["immutable"]["model_contract_sha256"]
    )
    totals: Counter[str] = Counter()
    states: Counter[str] = Counter()
    artifact_schemas: Counter[str] = Counter()
    run_reports: dict[str, Any] = {}
    scan_errors: list[str] = []
    trusted_generation_tuples: frozenset[
        tuple[str, str, int, int, str]
    ] = frozenset()
    trusted_catalog_binding: dict[str, Any] | None = None
    try:
        trusted_catalog = control.refresh_trusted_generation_catalog(
            state_dir,
            target_rollout_generation=int(state["rollout_generation"]),
            now=now,
        )
        trusted_generation_tuples = trusted_catalog.allowed_generation_tuples
        trusted_catalog_binding = {
            "catalog_id": trusted_catalog.catalog_id,
            "marker_path": str(trusted_catalog.marker_path),
            "marker_sha256": trusted_catalog.marker_sha256,
            "inventory_sha256": trusted_catalog.inventory_sha256,
            "catalog_payload_sha256": trusted_catalog.catalog_sha256,
            "allowed_generation_tuple_count": len(
                trusted_catalog.allowed_generation_tuples
            ),
        }
    except Exception as exc:
        scan_errors.append(
            "trusted-generation-catalog: "
            f"{type(exc).__name__}: {exc}"
        )

    for run_spec in config["runs"]:
        run_id = str(run_spec["run_id"])
        run_root = (results_root / run_id).resolve()
        snapshot = load_manifest(run_root, verify_frozen=True)
        catalog = VerifiedQuestionCatalog(run_root, snapshot=snapshot)
        policy = load_artifact_policy(run_root, required=True)
        assert policy is not None
        pin = pinned_runs.get(run_id)
        if pin is None:
            raise MonitorError(f"control has no immutable run pin for {run_id}")
        contract_errors: list[str] = []
        checks = {
            "manifest_sha256": snapshot.sha256,
            "benchmark_contract_sha256": catalog.sidecar_sha256,
            "policy_sha256": policy.file_sha256,
        }
        for field, observed in checks.items():
            if pin.get(field) != observed:
                contract_errors.append(
                    f"{field}: pinned {pin.get(field)!r}, observed {observed!r}"
                )
        if policy.accepted_manifest_sha256 != snapshot.sha256:
            contract_errors.append("policy does not bind manifest")
        if policy.accepted_benchmark_contracts_sha256 != catalog.sidecar_sha256:
            contract_errors.append("policy does not bind benchmark contract")
        if policy.accepted_model_contract_sha256 != contracts.sha256:
            contract_errors.append("policy does not bind model contract")
        if len(snapshot.cells) != int(run_spec["expected_cells"]):
            contract_errors.append("manifest cell count differs from monitor contract")

        cells_root = run_root / "cells"
        present = (
            {path.name for path in cells_root.iterdir() if path.is_dir()}
            if cells_root.is_dir()
            else set()
        )
        stale = sorted(present - set(snapshot.ids))
        run_states: Counter[str] = Counter()
        run_totals: Counter[str] = Counter()
        run_schemas: Counter[str] = Counter()
        expected_qids = 0

        for cell in snapshot.cells:
            questions = catalog.questions_for(cell)
            qids = tuple(question.qid for question in questions)
            expected_qids += len(qids)
            stratum = _stratum_key(cell, dimensions)
            rotation_key = _stratum_key(cell, rotation_dimensions)
            progress[stratum]["expected"] += len(qids)
            rotation[rotation_key]["expected"] += len(qids)
            cdir = cells_root / cell.cell_id
            try:
                status = get_completion_status(
                    cell,
                    cdir,
                    expected_qids=qids,
                    expected_questions=questions,
                    verified_benchmark_contracts=catalog.frozen,
                    verified_manifest=catalog.snapshot,
                    check_active=True,
                    now=now,
                    model_contract_path=state["immutable"]["model_contract_path"],
                    trusted_generation_tuples=trusted_generation_tuples,
                )
                canonical = (
                    read_canonical_results(
                        cell,
                        cdir,
                        expected_qids=qids,
                        expected_questions=questions,
                        verified_benchmark_contracts=catalog.frozen,
                        verified_manifest=catalog.snapshot,
                    )
                    if status.valid_count
                    else None
                )
            except Exception as exc:
                run_states["validation_error"] += 1
                scan_errors.append(f"{run_id}/{cell.cell_id}: {type(exc).__name__}: {exc}")
                continue
            run_states[status.status.value] += 1
            run_totals["malformed_lines"] += status.malformed_lines
            run_totals["duplicate_qids"] += len(status.duplicate_qids)
            run_totals["unexpected_qids"] += len(status.unexpected_qids)
            run_totals["invalid_rows"] += status.invalid_rows

            records = [] if canonical is None else list(canonical.records[: status.valid_count])
            trusted = [
                record
                for record in records
                if _record_matches_policy(
                    record,
                    cell,
                    policy,
                    contracts,
                    release_fleet_contract_sha256=state["immutable"][
                        "fleet_contract_sha256"
                    ],
                    trusted_generation_tuples=trusted_generation_tuples,
                )
            ]
            run_totals["untrusted_valid_rows"] += len(records) - len(trusted)
            run_totals["validated_qids"] += len(trusted)
            useful = [row for row in trusted if _record_is_useful(row)]
            run_totals["useful_qids"] += len(useful)
            run_totals["transport_affected_qids"] += sum(
                transport_censored_coordinate_count([row]) > 0
                for row in trusted
            )
            progress[stratum]["validated"] += len(trusted)
            progress[stratum]["useful_validated"] += len(useful)
            rotation[rotation_key]["validated"] += len(trusted)
            rotation[rotation_key]["useful_validated"] += len(useful)
            timestamps = [
                timestamp
                for timestamp in (
                    _finite_timestamp(row.get("timestamp")) for row in useful
                )
                if timestamp is not None
            ]
            progress[stratum]["timestamps"].extend(timestamps)
            rotation[rotation_key]["timestamps"].extend(timestamps)
            for row in trusted:
                run_schemas[str(row.get("schema_version", "legacy"))] += 1
                termination = row.get("termination_status", TERMINATION_COMPLETED)
                if termination == TERMINATION_COMPLETED:
                    run_totals["completed_qids"] += 1
                elif termination == TERMINATION_LENGTH_CENSORED:
                    run_totals["length_censored_qids"] += 1
                elif termination == TERMINATION_PROTOCOL_CENSORED:
                    run_totals["protocol_censored_qids"] += 1
                elif termination == TERMINATION_TRANSPORT_CENSORED:
                    run_totals["transport_censored_qids"] += 1
            accounting = _censor_accounting(trusted)
            run_totals["auxiliary_outcomes"] += accounting["n_auxiliary_samples"]
            run_totals["auxiliary_completed"] += accounting[
                "n_auxiliary_completed_samples"
            ]
            run_totals["auxiliary_length_censored"] += accounting[
                "n_auxiliary_length_censors"
            ]
            run_totals["auxiliary_protocol_censored"] += accounting[
                "n_auxiliary_protocol_censors"
            ]
            run_totals["auxiliary_transport_censored"] += accounting[
                "n_auxiliary_transport_censors"
            ]
            run_totals["topology_transport_censored_coordinates"] += accounting[
                "n_topology_transport_censored_coordinates"
            ]
            run_totals["transport_censored_coordinates"] += accounting[
                "n_transport_censored_coordinates"
            ]

        if expected_qids != int(run_spec["expected_qids"]):
            contract_errors.append(
                f"expected QID cardinality {expected_qids} != {run_spec['expected_qids']}"
            )
        if (
            run_totals["validated_qids"]
            != run_totals["completed_qids"]
            + run_totals["length_censored_qids"]
            + run_totals["protocol_censored_qids"]
            + run_totals["transport_censored_qids"]
        ):
            contract_errors.append("top-level termination partition is not exhaustive")
        if (
            run_totals["auxiliary_outcomes"]
            != run_totals["auxiliary_completed"]
            + run_totals["auxiliary_length_censored"]
            + run_totals["auxiliary_protocol_censored"]
            + run_totals["auxiliary_transport_censored"]
        ):
            contract_errors.append("auxiliary termination partition is not exhaustive")
        run_reports[run_id] = {
            "run_root": str(run_root),
            "manifest_cells": len(snapshot.cells),
            "expected_qids": expected_qids,
            "manifest_sha256": snapshot.sha256,
            "benchmark_contract_sha256": catalog.sidecar_sha256,
            "artifact_policy_sha256": policy.file_sha256,
            "artifact_policy_id": policy.policy_id,
            "contract_errors": contract_errors,
            "states": dict(sorted(run_states.items())),
            "outcomes": dict(sorted(run_totals.items())),
            "artifact_schema_counts": dict(sorted(run_schemas.items())),
            "stale_unmanifested_dirs": len(stale),
            "stale_unmanifested_examples": stale[:20],
        }
        states.update(run_states)
        totals.update(run_totals)
        artifact_schemas.update(run_schemas)
        totals["stale_unmanifested_dirs"] += len(stale)
        totals["contract_errors"] += len(contract_errors)

    report = {
        "scan_successful": not scan_errors,
        "scan_errors": scan_errors[:100],
        "runs": run_reports,
        "states": dict(sorted(states.items())),
        "outcomes": dict(sorted(totals.items())),
        "artifact_schema_counts": dict(sorted(artifact_schemas.items())),
        "trusted_generation_catalog": trusted_catalog_binding,
        "transport_censor_protocol": {
            "version": TRANSPORT_CENSOR_PROTOCOL_VERSION,
            "hash": TRANSPORT_CENSOR_PROTOCOL_HASH,
        },
    }
    return report, dict(progress), dict(rotation)


def throughput_projection(
    progress: Mapping[str, Mapping[str, Any]],
    rotation: Mapping[str, Mapping[str, Any]],
    *,
    epoch_started_at: float | None,
    now: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    policy = config["throughput"]
    minimum_hours = float(policy["minimum_observation_hours"])
    window_hours = float(policy["window_hours"])
    target_days = float(policy["target_completion_days"])
    minimum_rate = float(policy["minimum_qids_per_day"])
    observation_hours = (
        max(0.0, (now - epoch_started_at) / 3600.0)
        if epoch_started_at is not None
        else 0.0
    )
    window_start = (
        max(epoch_started_at, now - window_hours * 3600.0)
        if epoch_started_at is not None
        else now
    )
    window_days = max(0.0, (now - window_start) / 86400.0)
    rows: list[dict[str, Any]] = []
    for key in sorted(progress):
        value = progress[key]
        expected = int(value["expected"])
        validated = int(value["validated"])
        useful_validated = int(value.get("useful_validated", validated))
        remaining = max(0, expected - validated)
        observed = sum(
            window_start <= timestamp <= now for timestamp in value["timestamps"]
        )
        rate = observed / window_days if window_days > 0 else None
        eta = 0.0 if remaining == 0 else (
            remaining / rate if rate is not None and rate > 0 else None
        )
        rows.append(
            {
                "stratum": json.loads(key),
                "expected_qids": expected,
                "validated_qids": validated,
                "useful_validated_qids": useful_validated,
                "remaining_qids": remaining,
                "window_qids": observed,
                "qids_per_day": rate,
                "eta_days": eta,
            }
        )
    expected_total = sum(row["expected_qids"] for row in rows)
    validated_total = sum(row["validated_qids"] for row in rows)
    useful_validated_total = sum(
        row["useful_validated_qids"] for row in rows
    )
    remaining_total = sum(row["remaining_qids"] for row in rows)
    window_total = sum(row["window_qids"] for row in rows)
    overall_rate = window_total / window_days if window_days > 0 else None
    overall_eta = 0.0 if remaining_total == 0 else (
        remaining_total / overall_rate
        if overall_rate is not None and overall_rate > 0
        else None
    )
    unfinished = [row for row in rows if row["remaining_qids"] > 0]
    no_throughput = [row for row in unfinished if not row["qids_per_day"]]
    estimated_etas = [float(row["eta_days"]) for row in unfinished if row["eta_days"] is not None]
    max_eta = max(estimated_etas, default=0.0) if not no_throughput else None
    ready = epoch_started_at is not None and observation_hours >= minimum_hours

    rotation_observed = 0
    for value in rotation.values():
        if any(window_start <= timestamp <= now for timestamp in value["timestamps"]):
            rotation_observed += 1
    all_rotation_strata_observed = rotation_observed == len(rotation)
    within_target = bool(
        ready
        and not no_throughput
        and all_rotation_strata_observed
        and max_eta is not None
        and max_eta <= target_days
        and overall_eta is not None
        and overall_eta <= target_days
        and overall_rate is not None
        and overall_rate >= minimum_rate
    )
    return {
        "epoch_started_at": _iso(epoch_started_at) if epoch_started_at is not None else None,
        "epoch_started_timestamp": epoch_started_at,
        "observation_hours": observation_hours,
        "minimum_observation_hours": minimum_hours,
        "window_hours": window_hours,
        "window_start": _iso(window_start),
        "window_days": window_days,
        "target_completion_days": target_days,
        "minimum_qids_per_day": minimum_rate,
        "overall": {
            "expected_qids": expected_total,
            "validated_qids": validated_total,
            "useful_validated_qids": useful_validated_total,
            "remaining_qids": remaining_total,
            "window_qids": window_total,
            "qids_per_day": overall_rate,
            "eta_days": overall_eta,
        },
        "strata": rows,
        "rotation_coverage": {
            "total_strata": len(rotation),
            "strata_with_window_progress": rotation_observed,
            "all_strata_observed": all_rotation_strata_observed,
        },
        "acceptance": {
            "observation_ready": ready,
            "unfinished_strata": len(unfinished),
            "strata_with_observed_throughput": len(unfinished) - len(no_throughput),
            "strata_without_observed_throughput": len(no_throughput),
            "all_rotation_strata_observed": all_rotation_strata_observed,
            "max_stratum_eta_days": max_eta,
            "overall_rate_meets_floor": bool(
                overall_rate is not None and overall_rate >= minimum_rate
            ),
            "projected_completion_within_target": within_target,
        },
    }


def _ledger_health(
    state_dir: Path, *, now: float, stale_seconds: float
) -> dict[str, Any]:
    path = state_dir / "ledger.json"
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "available": False,
            "healthy": False,
            "status": "missing",
            "path": str(path),
            "starved_runs": [],
            "cached_state_counts": {},
        }
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "healthy": False,
            "status": "invalid",
            "path": str(path),
            "error": str(exc),
            "starved_runs": [],
            "cached_state_counts": {},
        }
    try:
        ledger = validate_ledger_structure(ledger, source=str(path))
    except DispatcherLedgerError as exc:
        return {
            "available": False,
            "healthy": False,
            "status": "invalid",
            "path": str(path),
            "error": str(exc),
            "starved_runs": [],
            "cached_state_counts": {},
        }
    updated_at = _finite_timestamp(ledger.get("updated_at"))
    age = None if updated_at is None else max(0.0, now - updated_at)
    status = (
        "invalid"
        if updated_at is None
        else ("stale" if age is not None and age > stale_seconds else "healthy")
    )
    run_rows = ledger.get("runs", {}) if isinstance(ledger, dict) else {}
    starved = sorted(
        run_id
        for run_id, row in run_rows.items()
        if isinstance(row, dict)
        and int(row.get("backlogged_polls_without_admission", 0)) >= 2
    )
    cell_rows = ledger.get("cells", {}) if isinstance(ledger, dict) else {}
    cached_states = Counter(
        str(row.get("completion_state", "unknown"))
        for row in cell_rows.values()
        if isinstance(row, dict)
    )
    return {
        "available": True,
        "healthy": status == "healthy",
        "status": status,
        "path": str(path),
        "poll_number": ledger.get("poll_number"),
        "updated_at": ledger.get("updated_at"),
        "age_seconds": age,
        "stale_after_seconds": stale_seconds,
        "starved_runs": starved,
        "cached_state_counts": dict(sorted(cached_states.items())),
    }


def _material_fleet_generation(
    *, fleet_contract_sha256: str, material_capacity_layout_generation: int
) -> str:
    """Derive the throughput epoch identity from material, durable fleet policy.

    Endpoint registrations deliberately do not participate.  Their allocation IDs and
    process timestamps must continue to change whenever a 24-hour server job is
    replaced, and remain available separately as ``server_pool_generations`` for
    request/retry provenance.  Only an immutable fleet-contract change or an explicit
    capacity/layout generation bump changes this identity.
    """

    if (
        not isinstance(fleet_contract_sha256, str)
        or len(fleet_contract_sha256) != 64
        or any(character not in "0123456789abcdef" for character in fleet_contract_sha256)
    ):
        raise MonitorError("immutable fleet-contract SHA-256 is invalid")
    if (
        not isinstance(material_capacity_layout_generation, int)
        or isinstance(material_capacity_layout_generation, bool)
        or material_capacity_layout_generation < 1
    ):
        raise MonitorError("material capacity/layout generation must be a positive integer")
    identity = {
        "protocol": "schema5-material-fleet-generation-v1",
        "fleet_contract_sha256": fleet_contract_sha256,
        "material_capacity_layout_generation": material_capacity_layout_generation,
    }
    return "schema5-material-fleet-v1:" + _sha256_value(identity)


def _verified_material_fleet_identity(
    state: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Verify the effective capacity contract and return its material identity."""

    immutable = state.get("immutable")
    if not isinstance(immutable, Mapping):
        raise MonitorError("control immutable fleet pins are unavailable")
    try:
        binding = control.effective_fleet_contract_binding(
            state, verify_files=True
        )
        fleet = control.load_effective_fleet_contract(state, verify_files=True)
    except (control.ControlError, OSError, ValueError) as exc:
        raise MonitorError(f"cannot verify effective fleet contract: {exc}") from exc
    observed_layout = {
        profile: len(replicas)
        for profile, replicas in sorted(fleet.by_profile.items())
    }
    bound_layout = binding.get("profile_replicas")
    if observed_layout != bound_layout:
        raise MonitorError(
            "effective fleet contract and capacity binding differ: "
            f"contract={observed_layout}, binding={bound_layout}"
        )
    if set(observed_layout) != set(config.get("fleet_replicas", {})):
        raise MonitorError(
            "effective fleet profile universe differs from monitoring policy"
        )
    material_generation = binding["capacity_generation"]
    effective_sha256 = str(binding["sha256"])
    release_sha256 = immutable.get("fleet_contract_sha256")
    if (
        not isinstance(release_sha256, str)
        or len(release_sha256) != 64
        or any(character not in "0123456789abcdef" for character in release_sha256)
    ):
        raise MonitorError("release fleet-contract lineage hash is invalid")
    generation = _material_fleet_generation(
        fleet_contract_sha256=effective_sha256,
        material_capacity_layout_generation=material_generation,
    )
    return generation, {
        "fleet_contract_path": str(binding["path"]),
        "fleet_contract_sha256": effective_sha256,
        "release_fleet_contract_sha256": release_sha256,
        "is_capacity_overlay": bool(binding["is_capacity_overlay"]),
        "capacity_generation": material_generation,
        "material_capacity_layout_generation": material_generation,
        "profile_replicas": dict(sorted(observed_layout.items())),
    }


def collect_health_state(
    *,
    results_root: Path,
    state_dir: Path,
    config: Mapping[str, Any],
    now: float,
    probe_endpoints: bool,
    scheduler_safety_runner=None,
) -> dict[str, Any]:
    state = control.load_control(state_dir)
    scheduler_snapshot = control.query_scheduler(now=now, tolerate_errors=True)
    live = control.live_status(state_dir, snapshot=scheduler_snapshot, now=now)
    server_pool = Path(state["immutable"]["server_pool_root"]).resolve()
    fleet_generation, material_fleet_identity = _verified_material_fleet_identity(
        state, config
    )
    try:
        attested_scheduler = control.fleet_scheduler_policy_binding(
            state, require_attested=True
        )
        assert attested_scheduler is not None
        effective_fleet = control.load_effective_fleet_contract(
            state, verify_files=True
        )
        partition_time_requirements = (
            scheduler_safety.fleet_partition_time_requirements(
                effective_fleet.replicas
            )
        )
        live_scheduler_evidence = (
            scheduler_safety.capture_scheduler_safety_evidence(
                list(partition_time_requirements),
                runner=scheduler_safety_runner,
                captured_timestamp=now,
            )
        )
        live_scheduler_policy = (
            scheduler_safety.validate_scheduler_safety_evidence(
                live_scheduler_evidence,
                expected_partitions=list(partition_time_requirements),
                required_time_limits_seconds=partition_time_requirements,
            )
        )
        scheduler_policy_health = {
            "available": True,
            "drift": (
                live_scheduler_policy["policy_contract_id"]
                != attested_scheduler[
                    "scheduler_safety_policy_contract_id"
                ]
            ),
            "scheduler_evidence_id": live_scheduler_policy[
                "scheduler_evidence_id"
            ],
            "scheduler_policy_id": live_scheduler_policy["policy_id"],
            "scheduler_policy_contract_id": live_scheduler_policy[
                "policy_contract_id"
            ],
            "attested_scheduler_policy_contract_id": attested_scheduler[
                "scheduler_safety_policy_contract_id"
            ],
            "transport_uncertainty_binding_sha256": state["immutable"][
                "transport_uncertainty_binding_sha256"
            ],
            "preemptible_partitions": live_scheduler_policy[
                "preemptible_partitions"
            ],
            "partition_time_requirements_seconds": (
                partition_time_requirements
            ),
            "error": None,
        }
    except (
        KeyError,
        OSError,
        control.ImmutablePinError,
        control.ReadinessError,
        scheduler_safety.SchedulerSafetyError,
    ) as exc:
        scheduler_policy_health = {
            "available": False,
            "drift": True,
            "scheduler_evidence_id": None,
            "scheduler_policy_id": None,
            "scheduler_policy_contract_id": None,
            "attested_scheduler_policy_contract_id": None,
            "transport_uncertainty_binding_sha256": state.get(
                "immutable", {}
            ).get("transport_uncertainty_binding_sha256"),
            "preemptible_partitions": None,
            "partition_time_requirements_seconds": None,
            "error": str(exc),
        }
    admission_state = state.get("admission")
    if not isinstance(admission_state, Mapping):
        admission_state = live.get("admission")
    configured_ceiling = int(
        admission_state.get(
            "current_ceiling",
            control.PRODUCTION_NONPREEMPTIBLE_CELL_CAP,
        )
        if isinstance(admission_state, Mapping)
        else control.PRODUCTION_NONPREEMPTIBLE_CELL_CAP
    )
    try:
        client_contract = control.client_capacity_contract_from_state(
            state_dir
        )
        capacity_contract = protected_capacity.load_contract(
            state["immutable"]["protected_capacity_marker_path"],
            expected_release_git_commit=str(
                state["immutable"]["git_commit"]
            ),
            expected_marker_id=str(
                state["immutable"]["protected_capacity_marker_id"]
            ),
            expected_sha256=str(
                state["immutable"][
                    "protected_capacity_marker_sha256"
                ]
            ),
        )
        live_client_capacity = (
            protected_capacity.capture_live_client_capacity(
                capacity_contract,
                partition=str(client_contract["partition"]),
                qos=str(client_contract["qos"]),
                required_time_limit_seconds=(
                    scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                ),
                runner=scheduler_safety_runner,
                captured_timestamp=now,
            )
        )
        live_client_summary = (
            protected_capacity.validate_live_client_capacity_evidence(
                capacity_contract,
                live_client_capacity,
                partition=str(client_contract["partition"]),
                qos=str(client_contract["qos"]),
                required_time_limit_seconds=(
                    scheduler_safety.CLIENT_JOB_TIME_LIMIT_SECONDS
                ),
            )
        )
        client_capacity_health = {
            "required": True,
            "available": True,
            "drift": False,
            "configured_ceiling": configured_ceiling,
            "authorization_sha256": client_contract[
                "authorization_sha256"
            ],
            "capacity_generation": client_contract[
                "capacity_generation"
            ],
            **live_client_summary,
            "error": None,
        }
    except (
        KeyError,
        OSError,
        control.ControlError,
        scheduler_safety.SchedulerSafetyError,
        protected_capacity.ProtectedCapacityError,
    ) as exc:
        client_capacity_health = {
            "required": True,
            "available": False,
            "drift": True,
            "configured_ceiling": configured_ceiling,
            "authorization_sha256": None,
            "capacity_generation": state.get("capacity", {}).get(
                "current_generation"
            ),
            "error": str(exc),
        }
    expected_replicas = material_fleet_identity["profile_replicas"]
    profile_names = tuple(expected_replicas)
    endpoints, generations = monitor_run._endpoint_stats(
        server_pool,
        profile_names,
        probe_http=probe_endpoints,
        timeout=3.0,
    )
    fleet_mismatches = {
        profile: {
            "expected": int(expected_replicas[profile]),
            "live": int(endpoints.get(profile, {}).get("live", 0)),
        }
        for profile in expected_replicas
        if int(endpoints.get(profile, {}).get("live", 0))
        != int(expected_replicas[profile])
    }
    http_fleet_mismatches = (
        {
            profile: {
                "expected": int(expected_replicas[profile]),
                "http_healthy": int(endpoints.get(profile, {}).get("http_healthy", -1)),
            }
            for profile in expected_replicas
            if int(endpoints.get(profile, {}).get("http_healthy", -1))
            != int(expected_replicas[profile])
        }
        if probe_endpoints
        else None
    )
    disk = shutil.disk_usage(results_root)
    filesystem = os.statvfs(results_root)
    free_inodes = int(filesystem.f_favail)
    total_inodes = int(filesystem.f_files)
    scheduler = monitor_run._scheduler_stats()
    ledger = _ledger_health(
        state_dir,
        now=now,
        stale_seconds=float(config["alerts"]["dispatcher_ledger_stale_seconds"]),
    )
    try:
        fleet_transaction_health = fleet_transactions.read_health_summary(server_pool)
    except fleet_transactions.FleetTransactionError as exc:
        fleet_transaction_health = {
            "available": False,
            "current_generation": None,
            "active_hung_allocations": [],
            "active_handoffs": [],
            "handoff_violations": [],
            "handoff_overlap_gpus": 0,
            "historical_alert_count": 0,
            "alerts_path": str(
                fleet_transactions.state_directory(server_pool)
                / fleet_transactions.ALERTS_FILENAME
            ),
            "error": str(exc),
        }
    monitoring_root = state_dir / MONITORING_DIRNAME
    latest_semantic = monitoring_root / "semantic.latest.json"
    semantic_age_seconds: float | None = None
    if latest_semantic.is_file():
        latest_report = _load_latest_report(
            latest_semantic, cadence="semantic"
        )
        captured = _finite_timestamp(latest_report.get("captured_timestamp"))
        if captured is None:
            raise MonitorError(
                "latest semantic history has no finite capture timestamp"
            )
        semantic_age_seconds = max(0.0, now - captured)
    elif live.get("open_throughput_epoch") is not None:
        started = live["open_throughput_epoch"].get("started_timestamp")
        if isinstance(started, (int, float)):
            semantic_age_seconds = max(0.0, now - float(started))
    return {
        "control": live,
        "scheduler": scheduler,
        "endpoints": endpoints,
        "fleet_generation": fleet_generation,
        "material_fleet_identity": material_fleet_identity,
        # These volatile per-process generations remain visible for response/retry
        # provenance, but never define a 48-hour throughput epoch.
        "server_pool_generations": dict(sorted(generations.items())),
        "fleet_mismatches": fleet_mismatches,
        "http_probes_performed": bool(probe_endpoints),
        "http_fleet_mismatches": http_fleet_mismatches,
        "ledger": ledger,
        "fleet_transactions": fleet_transaction_health,
        "fleet_scheduler_policy": scheduler_policy_health,
        "client_capacity_health": client_capacity_health,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "free_fraction": disk.free / disk.total if disk.total else 0.0,
            "total_inodes": total_inodes,
            "free_inodes": free_inodes,
            "free_inode_fraction": (
                free_inodes / total_inodes if total_inodes else 0.0
            ),
        },
        "latest_semantic_report_age_seconds": semantic_age_seconds,
    }


def _epoch_start_for_generation(
    state: Mapping[str, Any], fleet_generation: str, *, prospective_now: float | None
) -> float | None:
    epochs = state.get("throughput_epochs", [])
    if epochs:
        latest = epochs[-1]
        if latest.get("closed_at") is None and latest.get("fleet_generation") == fleet_generation:
            value = latest.get("started_timestamp")
            if isinstance(value, (int, float)):
                return float(value)
    return prospective_now if state.get("desired_state") == "running" else None


def final_acceptance(
    semantic: Mapping[str, Any], throughput: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    states = Counter(semantic["states"])
    outcomes = Counter(semantic["outcomes"])
    checks = {
        "exact_complete_cells": states[CompletionState.COMPLETE.value]
        == int(config["expected_total_cells"]),
        "no_noncomplete_cells": sum(
            count for state, count in states.items() if state != CompletionState.COMPLETE.value
        )
        == 0,
        "exact_validated_qids": outcomes["validated_qids"]
        == int(config["expected_total_qids"]),
        "top_level_partition_exact": outcomes["validated_qids"]
        == outcomes["completed_qids"]
        + outcomes["length_censored_qids"]
        + outcomes["protocol_censored_qids"]
        + outcomes["transport_censored_qids"],
        "auxiliary_partition_exact": outcomes["auxiliary_outcomes"]
        == outcomes["auxiliary_completed"]
        + outcomes["auxiliary_length_censored"]
        + outcomes["auxiliary_protocol_censored"]
        + outcomes["auxiliary_transport_censored"],
        "useful_transport_partition_exact": outcomes["validated_qids"]
        == outcomes["useful_qids"] + outcomes["transport_affected_qids"],
        "transport_coordinate_partition_exact": outcomes[
            "transport_censored_coordinates"
        ]
        == outcomes["topology_transport_censored_coordinates"]
        + outcomes["auxiliary_transport_censored"],
        "topology_transport_coordinate_qid_bound": outcomes[
            "topology_transport_censored_coordinates"
        ]
        >= outcomes["transport_censored_qids"],
        "transport_affected_qid_bounds_exact": outcomes[
            "transport_censored_qids"
        ]
        <= outcomes["transport_affected_qids"]
        <= outcomes["transport_censored_coordinates"],
        "transport_protocol_exact": semantic.get("transport_censor_protocol")
        == {
            "version": TRANSPORT_CENSOR_PROTOCOL_VERSION,
            "hash": TRANSPORT_CENSOR_PROTOCOL_HASH,
        },
        "zero_integrity_errors": sum(
            outcomes[name]
            for name in (
                "malformed_lines",
                "duplicate_qids",
                "unexpected_qids",
                "invalid_rows",
                "untrusted_valid_rows",
                "stale_unmanifested_dirs",
                "contract_errors",
            )
        )
        == 0,
        "schema5_only": semantic.get("artifact_schema_counts")
        == {"5": int(config["expected_total_qids"])},
        "throughput_projection_acceptable": bool(
            throughput["acceptance"]["projected_completion_within_target"]
            or outcomes["validated_qids"] == int(config["expected_total_qids"])
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _semantic_payload_issues(report: Mapping[str, Any]) -> tuple[str, ...]:
    """Return structural defects that make semantic alert resolution unsafe."""

    issues: list[str] = []
    semantic = report.get("semantic")
    if not isinstance(semantic, Mapping):
        issues.append("semantic must be an object")
    else:
        if not isinstance(semantic.get("scan_successful"), bool):
            issues.append("semantic.scan_successful must be a boolean")
        for field in ("states", "outcomes"):
            if not isinstance(semantic.get(field), Mapping):
                issues.append(f"semantic.{field} must be an object")

    throughput = report.get("throughput")
    if not isinstance(throughput, Mapping):
        issues.append("throughput must be an object")
    else:
        acceptance = throughput.get("acceptance")
        if not isinstance(acceptance, Mapping):
            issues.append("throughput.acceptance must be an object")
        else:
            for field in (
                "observation_ready",
                "projected_completion_within_target",
            ):
                if not isinstance(acceptance.get(field), bool):
                    issues.append(
                        f"throughput.acceptance.{field} must be a boolean"
                    )
    return tuple(issues)


def _cadence_payload_complete(
    report: Mapping[str, Any], *, cadence: str
) -> bool:
    if cadence in {"semantic", "daily"}:
        return not _semantic_payload_issues(report)
    return isinstance(report.get("health"), Mapping)


def _cadence_execution_succeeded(
    report: Mapping[str, Any], *, cadence: str
) -> bool:
    if not _cadence_payload_complete(report, cadence=cadence):
        return False
    if cadence in {"semantic", "daily"}:
        semantic = report.get("semantic")
        return bool(
            isinstance(semantic, Mapping)
            and semantic.get("scan_successful") is True
        )
    return True


def evaluate_alerts(
    report: Mapping[str, Any], *, cadence: str, config: Mapping[str, Any]
) -> list[AlertFinding]:
    findings: list[AlertFinding] = []
    health = report["health"]
    control_live = health["control"]
    if not health["scheduler"].get("query_ok", False) or not (
        control_live["scheduler"]["squeue_ok"] and control_live["scheduler"]["sacct_ok"]
    ):
        findings.append(AlertFinding("monitor:scheduler", "scheduler-query", "critical", "scheduler truth is unavailable or incomplete"))
    scheduler_policy_health = health.get("fleet_scheduler_policy")
    if isinstance(scheduler_policy_health, Mapping) and (
        scheduler_policy_health.get("available") is not True
        or scheduler_policy_health.get("drift") is not False
    ):
        findings.append(
            AlertFinding(
                "monitor:scheduler",
                "scheduler-policy-drift",
                "critical",
                "live fleet partition/preemption policy is unavailable or "
                f"differs from readiness: {scheduler_policy_health!r}",
            )
        )
    if control_live["desired_state"] == "running" and any(
        row["heartbeat_stale"] or not row["live_active"]
        for row in control_live["controllers"].values()
    ):
        findings.append(AlertFinding("monitor:controllers", "controller-health", "critical", "one or more running controllers are absent or stale"))
    live_mismatches = health.get("fleet_mismatches")
    http_probes_performed = health.get("http_probes_performed")
    http_mismatches = health.get("http_fleet_mismatches")
    transaction_health = health.get("fleet_transactions")
    handoff_violations = (
        transaction_health.get("handoff_violations")
        if isinstance(transaction_health, Mapping)
        else None
    )
    missing_health_probe = cadence == "health" and http_probes_performed is not True
    invalid_http_truth = http_probes_performed is True and not isinstance(
        http_mismatches, Mapping
    )
    if control_live["desired_state"] == "running" and (
        bool(live_mismatches)
        or missing_health_probe
        or invalid_http_truth
        or (isinstance(http_mismatches, Mapping) and bool(http_mismatches))
        or bool(handoff_violations)
    ):
        findings.append(
            AlertFinding(
                "monitor:fleet",
                "fleet-capacity",
                "critical",
                "canonical fleet health mismatch: "
                f"scheduler_live={live_mismatches}, "
                f"http_probes_performed={http_probes_performed!r}, "
                f"http_healthy={http_mismatches!r}, "
                f"handoff_violations={handoff_violations!r}",
            )
        )
    if control_live["desired_state"] == "running" and (
        not isinstance(transaction_health, Mapping)
        or transaction_health.get("available") is not True
        or bool(transaction_health.get("active_hung_allocations"))
    ):
        findings.append(
            AlertFinding(
                "monitor:fleet-hung",
                "hung-serving-allocation",
                "critical",
                "canonical fleet transaction/hung-allocation state is unhealthy: "
                f"{transaction_health!r}",
            )
        )
    if int(health["scheduler"].get("qos_max_memory_per_user_cell_holds", 0)):
        findings.append(AlertFinding("monitor:qos-memory", "qos-hold", "warning", "cell jobs are held by QOSMaxMemoryPerUser"))
    disk_policy = config["alerts"]
    if health["disk"]["free_bytes"] < int(disk_policy["minimum_disk_free_bytes"]) or health["disk"]["free_fraction"] < float(disk_policy["minimum_disk_free_fraction"]):
        findings.append(AlertFinding("monitor:disk", "disk-capacity", "critical", f"results filesystem headroom is low: {health['disk']}"))
    if (
        int(health["disk"].get("free_inodes", 0))
        < int(disk_policy["minimum_free_inodes"])
        or float(health["disk"].get("free_inode_fraction", 0.0))
        < float(disk_policy["minimum_free_inode_fraction"])
    ):
        findings.append(
            AlertFinding(
                "monitor:inodes",
                "inode-capacity",
                "critical",
                f"results filesystem inode headroom is low: {health['disk']}",
            )
        )
    if (
        control_live["desired_state"] == "running"
        and health["ledger"].get("healthy") is not True
    ):
        findings.append(
            AlertFinding(
                "monitor:ledger",
                "dispatcher-ledger-health",
                "critical",
                "dispatcher ledger is missing, invalid, or stale: "
                f"{health['ledger']}",
            )
        )
    if health["ledger"]["starved_runs"]:
        findings.append(AlertFinding("monitor:starvation", "run-starvation", "warning", f"backlogged runs received no admission for at least two polls: {health['ledger']['starved_runs']}"))
    cached_states = Counter(health["ledger"].get("cached_state_counts", {}))
    if cached_states[CompletionState.CORRUPT.value] or cached_states.get("validation_error", 0):
        findings.append(AlertFinding("monitor:corrupt", "corrupt-artifacts", "critical", f"dispatcher cache reports corrupt/validation-error cells: {cached_states[CompletionState.CORRUPT.value] + cached_states.get('validation_error', 0)}"))
    if cached_states[CompletionState.PERMANENT.value]:
        findings.append(AlertFinding("monitor:permanent", "permanent-failures", "critical", f"dispatcher cache reports permanent cells: {cached_states[CompletionState.PERMANENT.value]}"))
    if cached_states[CompletionState.RETRYABLE.value]:
        findings.append(AlertFinding("monitor:retryable", "retryable-failures", "warning", f"dispatcher cache reports retryable cells: {cached_states[CompletionState.RETRYABLE.value]}"))
    semantic_age = health.get("latest_semantic_report_age_seconds")
    if control_live["desired_state"] == "running" and semantic_age is not None and semantic_age > 7 * 3600:
        findings.append(AlertFinding("monitor:semantic-stale", "semantic-monitor-stale", "warning", f"latest six-hour semantic report is {semantic_age / 3600:.1f} hours old"))
    ramp = control_live.get("admission_ramp")
    admission = control_live.get("admission")
    safety_hold = control_live.get("admission_safety_hold")
    safety_hold_drain = control_live.get("safety_hold_drain_intent")
    capacity_gate = (
        ramp.get("capacity_gate") if isinstance(ramp, Mapping) else None
    )
    if isinstance(capacity_gate, Mapping):
        findings.append(
            AlertFinding(
                "monitor:capacity-gate",
                "client-capacity-generation",
                "critical",
                "higher admission stage is held pending a controlled "
                "non-preemptible client-placement capacity transition: "
                f"{dict(capacity_gate)!r}",
            )
        )
    client_capacity_health = health.get("client_capacity_health")
    if (
        not isinstance(capacity_gate, Mapping)
        and isinstance(client_capacity_health, Mapping)
        and client_capacity_health.get("required") is True
        and (
            client_capacity_health.get("available") is not True
            or client_capacity_health.get("drift") is True
        )
    ):
        findings.append(
            AlertFinding(
                "monitor:capacity-gate",
                "client-capacity-drift",
                "critical",
                "the authorization-bound client partition/QOS no longer "
                f"matches live scheduler truth: {dict(client_capacity_health)!r}",
            )
        )
    if (
        isinstance(safety_hold, Mapping)
        and safety_hold.get("active") is True
        and (
            not isinstance(safety_hold_drain, Mapping)
            or safety_hold_drain.get("state") != "complete"
        )
    ):
        findings.append(
            AlertFinding(
                "monitor:hold-drain",
                "safety-hold-drain",
                "critical",
                "admission safety hold is active but its exact scheduler drain "
                f"transaction is not complete: {safety_hold_drain!r}",
            )
        )
    latched_hold_reasons = (
        set(safety_hold.get("reasons", ()))
        if isinstance(safety_hold, Mapping)
        and safety_hold.get("active") is True
        and isinstance(safety_hold.get("reasons"), list)
        else set()
    )
    ramp_stall_latched = "monitor:ramp-stall" in latched_hold_reasons
    ceiling = (
        admission.get("current_ceiling")
        if isinstance(admission, Mapping)
        else None
    )
    threshold = disk_policy["ramp_stall_seconds"].get(str(ceiling))
    if ramp_stall_latched:
        deadline = (
            f"{float(threshold) / 3600:.1f} hours"
            if threshold is not None
            else "its configured evidence deadline"
        )
        findings.append(
            AlertFinding(
                "monitor:ramp-stall",
                "admission-ramp-stall",
                "critical",
                f"admission ceiling {ceiling!r} remains latched after failing "
                f"its evidence gate within {deadline}",
            )
        )
    elif (
        control_live["desired_state"] == "running"
        and isinstance(ramp, Mapping)
        and isinstance(admission, Mapping)
    ):
        action = ramp.get("last_action")
        started = control_live.get("admission_stage_started_timestamp")
        if started is None:
            started = (
                ramp.get("window", {}).get("started_timestamp")
                if isinstance(ramp.get("window"), Mapping)
                else None
            )
        if started is None and isinstance(action, Mapping):
            started = action.get("timestamp")
        captured = _finite_timestamp(report.get("captured_timestamp"))
        if (
            threshold is not None
            and captured is not None
            and isinstance(started, (int, float))
            and captured - float(started) > float(threshold)
        ):
            findings.append(
                AlertFinding(
                    "monitor:ramp-stall",
                    "admission-ramp-stall",
                    "critical",
                    f"admission ceiling {ceiling} has not passed its evidence gate "
                    f"within {float(threshold) / 3600:.1f} hours",
                )
            )

    if cadence in {"semantic", "daily"}:
        throughput_latched = "monitor:throughput" in latched_hold_reasons
        payload_issues = _semantic_payload_issues(report)
        if payload_issues:
            findings.append(
                AlertFinding(
                    f"monitor:execution:{cadence}",
                    "monitor-execution-failure",
                    "critical",
                    "semantic monitor produced an incomplete report: "
                    + "; ".join(payload_issues),
                )
            )
            if throughput_latched:
                findings.append(
                    AlertFinding(
                        "monitor:throughput",
                        "capacity-gate",
                        "critical",
                        "48-hour QID throughput gate remains latched while "
                        "fresh semantic throughput evidence is incomplete",
                    )
                )
            return findings

        semantic = report["semantic"]
        states = Counter(semantic["states"])
        outcomes = Counter(semantic["outcomes"])
        if states[CompletionState.CORRUPT.value] or states.get("validation_error", 0):
            findings.append(AlertFinding("monitor:corrupt", "corrupt-artifacts", "critical", f"corrupt/validation-error cells: {states[CompletionState.CORRUPT.value] + states.get('validation_error', 0)}"))
        if states[CompletionState.PERMANENT.value]:
            findings.append(AlertFinding("monitor:permanent", "permanent-failures", "critical", f"permanent cells: {states[CompletionState.PERMANENT.value]}"))
        if states[CompletionState.RETRYABLE.value]:
            findings.append(AlertFinding("monitor:retryable", "retryable-failures", "warning", f"retryable cells: {states[CompletionState.RETRYABLE.value]}"))
        catalog_scan_errors = [
            str(error)
            for error in semantic.get("scan_errors", [])
            if str(error).startswith("trusted-generation-catalog:")
        ]
        if (
            outcomes["untrusted_valid_rows"]
            or outcomes["contract_errors"]
            or semantic.get("trusted_generation_catalog") is None
            or catalog_scan_errors
        ):
            findings.append(
                AlertFinding(
                    "monitor:untrusted",
                    "untrusted-responses",
                    "critical",
                    "trusted-generation authority failed: "
                    f"untrusted rows={outcomes['untrusted_valid_rows']}, "
                    f"contract errors={outcomes['contract_errors']}, "
                    f"catalog errors={catalog_scan_errors[:3]}",
                )
            )
        if outcomes["protocol_censored_qids"]:
            findings.append(AlertFinding("monitor:context-protocol", "protocol-censors", "warning", f"protocol-censored top-level QIDs: {outcomes['protocol_censored_qids']}"))
        transport_watch = report.get("transport_censor_watch", {})
        if int(transport_watch.get("new_coordinates", 0)) > 0:
            findings.append(
                AlertFinding(
                    "monitor:transport-censor",
                    "transport-censor",
                    "critical",
                    "new trusted transport-censored stochastic coordinates were "
                    f"observed: {transport_watch}",
                )
            )
        acceptance = report["throughput"]["acceptance"]
        if (
            (
                acceptance["observation_ready"]
                and not acceptance["projected_completion_within_target"]
            )
            or throughput_latched
        ):
            findings.append(AlertFinding("monitor:throughput", "capacity-gate", "critical", f"48-hour QID throughput gate failed: {acceptance}"))
        progress_watch = report.get("progress_watch", {})
        if (
            int(progress_watch.get("consecutive_no_progress_scans", 0))
            >= int(disk_policy["no_progress_semantic_scans"])
            and int(progress_watch.get("useful_qids", 0))
            < int(config["expected_total_qids"])
        ):
            findings.append(
                AlertFinding(
                    "monitor:no-progress",
                    "trusted-qid-progress-stall",
                    "critical",
                    "two consecutive semantic scans produced no new trusted QIDs: "
                    f"{progress_watch}",
                )
            )
    return findings


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a monitor evidence file once without any overwrite window."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != temporary.read_bytes():
                raise MonitorError(
                    f"immutable monitor evidence already exists with different bytes: {path}"
                )
            if path.stat().st_mode & 0o222:
                raise MonitorError(
                    f"existing monitor evidence is not sealed read-only: {path}"
                )
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _latest_pointer(
    *,
    cadence: str,
    history: Path,
    history_sha256: str,
    captured_timestamp: float,
    committed_timestamp: float,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "protocol": LATEST_POINTER_PROTOCOL,
        "cadence": cadence,
        "history": str(history.resolve()),
        "history_sha256": history_sha256,
        "captured_timestamp": captured_timestamp,
        "committed_timestamp": committed_timestamp,
    }
    return identity | {"pointer_id": _sha256_value(identity)}


def _load_latest_report(path: Path, *, cadence: str) -> dict[str, Any]:
    """Resolve a mutable latest pointer only through immutable report evidence."""

    supplied = path.expanduser().absolute()
    if supplied.is_symlink() or not supplied.is_file():
        raise MonitorError(f"{cadence} latest pointer is missing or unsafe: {supplied}")
    try:
        pointer = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MonitorError(f"cannot parse {cadence} latest pointer: {exc}") from exc
    required = {
        "schema_version",
        "protocol",
        "cadence",
        "history",
        "history_sha256",
        "captured_timestamp",
        "committed_timestamp",
        "pointer_id",
    }
    identity = dict(pointer) if isinstance(pointer, dict) else {}
    pointer_id = identity.pop("pointer_id", None)
    captured = pointer.get("captured_timestamp") if isinstance(pointer, dict) else None
    committed = (
        pointer.get("committed_timestamp") if isinstance(pointer, dict) else None
    )
    if (
        not isinstance(pointer, dict)
        or set(pointer) != required
        or pointer.get("schema_version") != 1
        or pointer.get("protocol") != LATEST_POINTER_PROTOCOL
        or pointer.get("cadence") != cadence
        or not isinstance(captured, (int, float))
        or isinstance(captured, bool)
        or not math.isfinite(float(captured))
        or captured < 0
        or not isinstance(committed, (int, float))
        or isinstance(committed, bool)
        or not math.isfinite(float(committed))
        or committed < captured
        or not isinstance(pointer_id, str)
        or pointer_id != _sha256_value(identity)
    ):
        raise MonitorError(f"{cadence} latest pointer identity is invalid")
    history = Path(str(pointer["history"])).expanduser().absolute()
    expected_parent = supplied.parent / cadence
    expected_name = f"{int(float(committed) * 1_000_000):020d}.json"
    if (
        history.parent != expected_parent
        or history.name != expected_name
        or history.is_symlink()
        or not history.is_file()
        or history.resolve() != history
        or history.stat().st_mode & 0o222
        or not isinstance(pointer.get("history_sha256"), str)
        or len(pointer["history_sha256"]) != 64
        or _sha256_file(history) != pointer["history_sha256"]
    ):
        raise MonitorError(f"{cadence} immutable history binding is invalid")
    try:
        report = json.loads(history.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MonitorError(f"cannot parse {cadence} immutable history: {exc}") from exc
    if (
        not isinstance(report, dict)
        or report.get("cadence") != cadence
        or report.get("captured_timestamp") != captured
    ):
        raise MonitorError(f"{cadence} immutable history identity is invalid")
    return report


def _production_poll_health_clean(report: Mapping[str, Any]) -> bool:
    """Require live controller, scheduler, and fleet truth before timing throughput.

    A semantically clean scan alone is not a successful *production* poll when the
    control plane is absent, scheduler truth is incomplete, or the canonical fleet is
    under capacity. Starting an epoch in any of those states would charge outage time
    to the scientific throughput estimate and could conceal a broken relaunch.
    """

    health = report.get("health")
    if not isinstance(health, Mapping):
        return False
    live = health.get("control")
    scheduler = health.get("scheduler")
    if not isinstance(live, Mapping) or not isinstance(scheduler, Mapping):
        return False
    live_scheduler = live.get("scheduler")
    controllers = live.get("controllers")
    fleet_scheduler_policy = health.get("fleet_scheduler_policy")
    http_probes_performed = health.get("http_probes_performed")
    if (
        live.get("desired_state") != "running"
        or scheduler.get("query_ok") is not True
        or not isinstance(live_scheduler, Mapping)
        or live_scheduler.get("squeue_ok") is not True
        or live_scheduler.get("sacct_ok") is not True
        or not isinstance(controllers, Mapping)
        or set(controllers) != {"dispatcher", "fleet_supervisor"}
        or health.get("fleet_mismatches") != {}
        or not isinstance(fleet_scheduler_policy, Mapping)
        or fleet_scheduler_policy.get("available") is not True
        or fleet_scheduler_policy.get("drift") is not False
        or not isinstance(health.get("fleet_transactions"), Mapping)
        or health["fleet_transactions"].get("available") is not True
        or health["fleet_transactions"].get("active_hung_allocations") != []
        or health["fleet_transactions"].get("handoff_violations", []) != []
        or not isinstance(http_probes_performed, bool)
        or (
            http_probes_performed
            and health.get("http_fleet_mismatches") != {}
        )
    ):
        return False
    return all(
        isinstance(row, Mapping)
        and row.get("live_active") is True
        and row.get("heartbeat_stale") is False
        for row in controllers.values()
    )


def _semantic_integrity_clean(report: Mapping[str, Any]) -> bool:
    semantic = report.get("semantic")
    if not isinstance(semantic, Mapping):
        return False
    outcomes = Counter(semantic.get("outcomes", {}))
    return bool(
        semantic.get("scan_successful") is True
        and outcomes["contract_errors"] == 0
        and outcomes["untrusted_valid_rows"] == 0
        and outcomes["malformed_lines"] == 0
        and outcomes["duplicate_qids"] == 0
        and outcomes["unexpected_qids"] == 0
        and outcomes["invalid_rows"] == 0
        and outcomes["stale_unmanifested_dirs"] == 0
    )


def _semantic_run_qids(
    report: Mapping[str, Any], *, field: str
) -> dict[str, int]:
    semantic = report.get("semantic")
    runs = semantic.get("runs") if isinstance(semantic, Mapping) else None
    if not isinstance(runs, Mapping) or set(runs) != set(control.REQUIRED_RUNS):
        raise MonitorError("semantic report must cover all three production runs")
    result: dict[str, int] = {}
    for run_id in control.REQUIRED_RUNS:
        row = runs[run_id]
        outcomes = row.get("outcomes") if isinstance(row, Mapping) else None
        value = outcomes.get(field) if isinstance(outcomes, Mapping) else None
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > control.REQUIRED_RUN_QIDS[run_id]
        ):
            raise MonitorError(
                f"semantic {field} total is invalid for {run_id}"
            )
        result[run_id] = value
    total = semantic.get("outcomes", {}).get(field)
    if total != sum(result.values()):
        raise MonitorError(
            f"semantic aggregate {field} differ from exact per-run totals"
        )
    return result


def _semantic_run_validated_qids(report: Mapping[str, Any]) -> dict[str, int]:
    return _semantic_run_qids(report, field="validated_qids")


def _semantic_run_useful_qids(report: Mapping[str, Any]) -> dict[str, int]:
    return _semantic_run_qids(report, field="useful_qids")


def _admission_ramp_observation(
    report: Mapping[str, Any],
    *,
    cadence: str,
    state: Mapping[str, Any],
    findings: Sequence[AlertFinding],
    captured_at: float,
    committed_at: float,
) -> dict[str, Any]:
    health = report.get("health")
    live = health.get("control") if isinstance(health, Mapping) else None
    if not isinstance(live, Mapping):
        raise MonitorError("ramp evidence requires scheduler-authoritative control state")
    admission = live.get("admission")
    if not isinstance(admission, Mapping):
        raise MonitorError("ramp evidence requires live admission state")
    semantic_clean: bool | None = None
    run_validated_qids: dict[str, int] | None = None
    run_useful_qids: dict[str, int] | None = None
    if cadence in {"semantic", "daily"}:
        semantic_clean = _semantic_integrity_clean(report)
        run_validated_qids = _semantic_run_validated_qids(report)
        run_useful_qids = _semantic_run_useful_qids(report)
    critical = sorted(
        {finding.dedupe_key for finding in findings if finding.severity == "critical"}
    )
    promotion_blocking = sorted(
        set(critical)
        | {
            finding.dedupe_key
            for finding in findings
            if finding.dedupe_key in control.ADMISSION_RAMP_BLOCKING_ALERT_KEYS
        }
    )
    return {
        "schema_version": 1,
        "protocol": control.ADMISSION_RAMP_EVIDENCE_PROTOCOL,
        "captured_timestamp": captured_at,
        "committed_timestamp": committed_at,
        "cadence": cadence,
        "control_immutable_sha256": state["immutable_sha256"],
        "rollout_generation": live.get("rollout_generation"),
        "admission_ceiling": admission.get("current_ceiling"),
        "fleet_generation": health.get("fleet_generation"),
        "production_health_clean": _production_poll_health_clean(report),
        "semantic_integrity_clean": semantic_clean,
        "critical_finding_keys": critical,
        "promotion_blocking_finding_keys": promotion_blocking,
        "run_validated_qids": run_validated_qids,
        "run_useful_qids": run_useful_qids,
    }


def persist_report(
    report: dict[str, Any],
    *,
    cadence: str,
    state_dir: Path,
    findings: Sequence[AlertFinding],
    send_email: bool,
    now: float,
    committed_at: float | None = None,
) -> dict[str, Any]:
    with _monitor_persist_lock(state_dir):
        timestamp = time.time() if committed_at is None else float(committed_at)
        return _persist_report_locked(
            report,
            cadence=cadence,
            state_dir=state_dir,
            findings=findings,
            send_email=send_email,
            captured_at=float(now),
            committed_at=timestamp,
        )


def _persist_report_locked(
    report: dict[str, Any],
    *,
    cadence: str,
    state_dir: Path,
    findings: Sequence[AlertFinding],
    send_email: bool,
    captured_at: float,
    committed_at: float,
) -> dict[str, Any]:
    state = control.load_control(state_dir)
    recorded_cadence = report.setdefault("cadence", cadence)
    recorded_captured = report.setdefault("captured_timestamp", captured_at)
    report.setdefault("captured_at", _iso(captured_at))
    if (
        recorded_cadence != cadence
        or not isinstance(recorded_captured, (int, float))
        or isinstance(recorded_captured, bool)
        or not math.isfinite(float(recorded_captured))
        or float(recorded_captured) != captured_at
        or report.get("captured_at") != _iso(captured_at)
    ):
        raise MonitorError("persisted monitor report capture identity is invalid")
    report["production_poll_recorded"] = False
    report["admission_ramp_recorded"] = False
    report["ramp_observation"] = _admission_ramp_observation(
        report,
        cadence=cadence,
        state=state,
        findings=findings,
        captured_at=captured_at,
        committed_at=committed_at,
    )

    root = state_dir / MONITORING_DIRNAME
    history = root / cadence / f"{int(committed_at * 1_000_000):020d}.json"
    latest = root / f"{cadence}.latest.json"
    _immutable_json(history, report)
    evidence_sha256 = _sha256_file(history)

    active_before = {
        alert["dedupe_key"]
        for alert in control.load_control(state_dir)["alerts"]
        if alert.get("resolved_at") is None
    }
    current = {finding.dedupe_key for finding in findings}
    for finding in findings:
        control.record_alert(
            state_dir,
            kind=finding.kind,
            severity=finding.severity,
            message=finding.message,
            dedupe_key=finding.dedupe_key,
            send_email=send_email and finding.dedupe_key not in active_before,
            now=committed_at,
        )
    owned_keys = HEALTH_ALERT_KEYS if cadence == "health" else SEMANTIC_ALERT_KEYS
    if _cadence_payload_complete(report, cadence=cadence):
        for dedupe_key in sorted(owned_keys - current):
            control.resolve_alert(
                state_dir, dedupe_key=dedupe_key, now=committed_at
            )
    execution_key = f"monitor:execution:{cadence}"
    if (
        execution_key not in current
        and _cadence_execution_succeeded(report, cadence=cadence)
    ):
        control.resolve_alert(
            state_dir,
            dedupe_key=execution_key,
            now=committed_at,
        )

    current_state = control.load_control(state_dir)
    active_critical = {
        alert["dedupe_key"]
        for alert in current_state["alerts"]
        if alert.get("resolved_at") is None and alert.get("severity") == "critical"
    }
    observation = report["ramp_observation"]
    if "admission_safety_hold" in current_state:
        current_state = control.update_admission_safety_hold(
            state_dir,
            active_critical_keys=sorted(active_critical),
            clean_poll=bool(
                cadence == "health"
                and
                observation["production_health_clean"] is True
                and not observation["critical_finding_keys"]
                and not active_critical
            ),
            semantic_scan_clean=observation["semantic_integrity_clean"],
            now=committed_at,
        )
        active_critical = {
            alert["dedupe_key"]
            for alert in current_state["alerts"]
            if alert.get("resolved_at") is None
            and alert.get("severity") == "critical"
        }
    if send_email and "admission_safety_hold" in current_state:
        control.retry_pending_alert_emails(state_dir, now=committed_at)
        current_state = control.load_control(state_dir)
    if (
        cadence in {"semantic", "daily"}
        and observation["semantic_integrity_clean"] is True
        and observation["production_health_clean"] is True
        and not observation["critical_finding_keys"]
        and not active_critical
        and current_state["desired_state"] == "running"
        and not current_state.get("admission_safety_hold", {}).get("active", False)
    ):
        control.record_successful_poll(
            state_dir,
            useful_qids=int(
                report["semantic"]["outcomes"].get("useful_qids", 0)
            ),
            fleet_generation=report["health"]["fleet_generation"],
            strata_with_throughput=int(
                report["throughput"]["acceptance"][
                    "strata_with_observed_throughput"
                ]
            ),
            now=committed_at,
        )
        report["production_poll_recorded"] = True

    ramp_state = control.record_admission_ramp_observation(
        state_dir,
        evidence_path=history,
        evidence_sha256=evidence_sha256,
        now=committed_at,
    )
    report["admission_ramp_recorded"] = True
    report["admission_ramp_action"] = copy.deepcopy(
        ramp_state["admission_ramp"]["last_action"]
    )
    report["admission_ceiling"] = ramp_state["admission"]["current_ceiling"]
    _atomic_json(
        latest,
        _latest_pointer(
            cadence=cadence,
            history=history,
            history_sha256=evidence_sha256,
            captured_timestamp=float(recorded_captured),
            committed_timestamp=committed_at,
        ),
    )
    finalization_requested = False
    if (
        cadence in {"semantic", "daily"}
        and report.get("final_acceptance", {}).get("passed") is True
        and report.get("semantic", {}).get("states")
        == {"complete": control.EXPECTED_TOTAL_CELLS}
        and report.get("semantic", {}).get("outcomes", {}).get(
            "validated_qids"
        )
        == control.EXPECTED_TOTAL_QIDS
    ):
        control.request_autonomous_finalization(
            state_dir,
            semantic_report_path=history,
            now=committed_at,
        )
        finalization_requested = True
    return {
        "history": str(history),
        "history_sha256": evidence_sha256,
        "latest": str(latest),
        "finalization_requested": finalization_requested,
    }


def _semantic_progress_watch(
    state_dir: Path, *, useful_qids: int, expected_qids: int, now: float
) -> dict[str, Any]:
    """Derive a durable no-progress counter from useful trusted outcomes only."""

    previous_reports: list[Mapping[str, Any]] = []
    root = state_dir / MONITORING_DIRNAME
    for cadence in ("semantic", "daily"):
        path = root / f"{cadence}.latest.json"
        try:
            value = _load_latest_report(path, cadence=cadence)
        except FileNotFoundError:
            continue
        except MonitorError as exc:
            if not path.exists() and not path.is_symlink():
                continue
            raise MonitorError(
                f"cannot read prior semantic progress evidence {path}: {exc}"
            ) from exc
        if isinstance(value, Mapping) and isinstance(
            value.get("captured_timestamp"), (int, float)
        ):
            previous_reports.append(value)
    previous = (
        max(previous_reports, key=lambda row: float(row["captured_timestamp"]))
        if previous_reports
        else None
    )
    previous_qids: int | None = None
    previous_count = 0
    if previous is not None:
        semantic = previous.get("semantic")
        outcomes = semantic.get("outcomes") if isinstance(semantic, Mapping) else None
        candidate = (
            outcomes.get("useful_qids") if isinstance(outcomes, Mapping) else None
        )
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            previous_qids = candidate
        prior_watch = previous.get("progress_watch")
        if isinstance(prior_watch, Mapping):
            prior_count = prior_watch.get("consecutive_no_progress_scans")
            if isinstance(prior_count, int) and not isinstance(prior_count, bool):
                previous_count = max(0, prior_count)
    if useful_qids >= expected_qids or previous_qids is None:
        consecutive = 0
    elif useful_qids > previous_qids:
        consecutive = 0
    elif useful_qids == previous_qids:
        consecutive = previous_count + 1
    else:
        # Regression is never "progress"; the semantic integrity checks will expose
        # the underlying missing/untrusted records while this guard immediately holds.
        consecutive = previous_count + 1
    return {
        "observed_at": _iso(now),
        "observed_timestamp": now,
        "useful_qids": useful_qids,
        "previous_useful_qids": previous_qids,
        "consecutive_no_progress_scans": consecutive,
        "regressed": bool(
            previous_qids is not None and useful_qids < previous_qids
        ),
    }


def _transport_censor_watch(
    state_dir: Path,
    *,
    coordinate_count: int,
    affected_qids: int,
    now: float,
) -> dict[str, Any]:
    """Compare exact retained transport censors with the last committed scan."""

    previous_count = 0
    previous_reports: list[Mapping[str, Any]] = []
    root = state_dir / MONITORING_DIRNAME
    for cadence in ("semantic", "daily"):
        path = root / f"{cadence}.latest.json"
        try:
            value = _load_latest_report(path, cadence=cadence)
        except FileNotFoundError:
            continue
        except MonitorError:
            if not path.exists() and not path.is_symlink():
                continue
            raise
        if isinstance(value, Mapping) and isinstance(
            value.get("captured_timestamp"), (int, float)
        ):
            previous_reports.append(value)
    if previous_reports:
        previous = max(
            previous_reports,
            key=lambda row: float(row["captured_timestamp"]),
        )
        watch = previous.get("transport_censor_watch")
        candidate = (
            watch.get("coordinate_count")
            if isinstance(watch, Mapping)
            else None
        )
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            previous_count = max(0, candidate)
    return {
        "observed_at": _iso(now),
        "observed_timestamp": now,
        "coordinate_count": coordinate_count,
        "affected_qids": affected_qids,
        "previous_coordinate_count": previous_count,
        "new_coordinates": max(0, coordinate_count - previous_count),
        "regressed": coordinate_count < previous_count,
    }


def build_report(
    *,
    cadence: str,
    results_root: Path,
    state_dir: Path,
    config: Mapping[str, Any],
    now: float,
    probe_endpoints: bool,
    prospective_epoch: bool,
) -> dict[str, Any]:
    health = collect_health_state(
        results_root=results_root,
        state_dir=state_dir,
        config=config,
        now=now,
        probe_endpoints=probe_endpoints,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "cadence": cadence,
        "captured_at": _iso(now),
        "captured_timestamp": now,
        "monitoring_contract_sha256": _sha256_file(DEFAULT_CONFIG),
        "health": health,
    }
    if cadence in {"semantic", "daily"}:
        semantic, progress, rotation = collect_semantic_state(
            results_root=results_root,
            state_dir=state_dir,
            config=config,
            now=now,
        )
        state = control.load_control(state_dir)
        epoch_start = _epoch_start_for_generation(
            state,
            health["fleet_generation"],
            prospective_now=now if prospective_epoch else None,
        )
        projection = throughput_projection(
            progress,
            rotation,
            epoch_started_at=epoch_start,
            now=now,
            config=config,
        )
        report["semantic"] = semantic
        report["throughput"] = projection
        report["progress_watch"] = _semantic_progress_watch(
            state_dir,
            useful_qids=int(semantic["outcomes"].get("useful_qids", 0)),
            expected_qids=(
                int(config["expected_total_qids"])
                - (
                    int(semantic["outcomes"].get("validated_qids", 0))
                    - int(semantic["outcomes"].get("useful_qids", 0))
                )
            ),
            now=now,
        )
        report["transport_censor_watch"] = _transport_censor_watch(
            state_dir,
            coordinate_count=int(
                semantic["outcomes"].get(
                    "transport_censored_coordinates", 0
                )
            ),
            affected_qids=(
                int(semantic["outcomes"].get("validated_qids", 0))
                - int(semantic["outcomes"].get("useful_qids", 0))
            ),
            now=now,
        )
        # Either complete semantic cadence may request autonomous finalization.  The
        # finalizer reruns the full scan before retiring serving capacity, so this
        # marker is an admission/drain trigger rather than the terminal acceptance
        # proof itself.
        report["final_acceptance"] = final_acceptance(
            semantic, projection, config
        )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", choices=("health", "semantic", "daily"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=Path(os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)))
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--probe-endpoints", action="store_true")
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--send-email", action="store_true")
    parser.add_argument("--now", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    now = time.time() if args.now is None else float(args.now)
    results_root = args.results_root.expanduser().resolve()
    state_dir = (
        args.state_dir.expanduser().resolve()
        if args.state_dir is not None
        else results_root / DEFAULT_STATE_DIRNAME
    )
    config_path = args.config.expanduser().resolve()
    config = load_monitor_config(config_path)
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config_path
    try:
        report = build_report(
            cadence=args.cadence,
            results_root=results_root,
            state_dir=state_dir,
            config=config,
            now=now,
            probe_endpoints=args.probe_endpoints,
            prospective_epoch=args.persist,
        )
        findings = evaluate_alerts(report, cadence=args.cadence, config=config)
        report["alert_findings"] = [finding.__dict__ for finding in findings]
        if args.persist:
            report["persisted_paths"] = persist_report(
                report,
                cadence=args.cadence,
                state_dir=state_dir,
                findings=findings,
                send_email=args.send_email,
                now=now,
            )
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if not any(item.severity == "critical" for item in findings) else 2
    except Exception as exc:
        if args.persist:
            try:
                control.record_alert(
                    state_dir,
                    kind="monitor-execution-failure",
                    severity="critical",
                    message=f"{type(exc).__name__}: {exc}",
                    dedupe_key=f"monitor:execution:{args.cadence}",
                    send_email=args.send_email,
                    now=now,
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
