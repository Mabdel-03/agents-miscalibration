#!/usr/bin/env python3
"""Generation-scoped health, semantic, and acceptance monitoring for schema-5.

Report generation is read-only.  ``--persist`` atomically publishes a report, appends a
successful validated-QID sample to the durable control plane, and raises/resolves alerts
through its existing append-only journal.  Top-level QIDs and auxiliary draws always use
separate denominators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
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
    get_completion_status,
    read_canonical_results,
)
from agents_scaling.experiment.manifest import load_manifest  # noqa: E402
from agents_scaling.experiment.result_schema import (  # noqa: E402
    TERMINATION_COMPLETED,
    TERMINATION_LENGTH_CENSORED,
    TERMINATION_PROTOCOL_CENSORED,
)
from agents_scaling.serving.model_contracts import load_model_contracts  # noqa: E402
from slurm import schema5_control as control  # noqa: E402
from scripts import monitor_run  # noqa: E402


DEFAULT_CONFIG = REPO / "configs" / "schema5_monitoring.v1.json"
DEFAULT_STATE_DIRNAME = ".dispatcher-schema5-v1"
MONITORING_DIRNAME = "monitoring"
HEALTH_ALERT_KEYS = frozenset(
    {
        "monitor:scheduler",
        "monitor:controllers",
        "monitor:fleet",
        "monitor:qos-memory",
        "monitor:disk",
        "monitor:starvation",
        "monitor:semantic-stale",
        # Dispatcher-ledger states provide the five-minute early-warning path; the
        # complete semantic scan independently confirms them every six hours.
        "monitor:corrupt",
        "monitor:permanent",
        "monitor:retryable",
    }
)
SEMANTIC_ALERT_KEYS = HEALTH_ALERT_KEYS | frozenset(
    {
        "monitor:untrusted",
        "monitor:throughput",
        "monitor:context-protocol",
    }
)


class MonitorError(RuntimeError):
    """A report cannot be made truthful from the frozen production state."""


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
    throughput = value.get("throughput")
    if not isinstance(throughput, dict) or throughput.get("minimum_qids_per_day") != 161_595:
        raise MonitorError("monitoring throughput floor must be 161,595 QIDs/day")
    if throughput.get("target_completion_days") != 28:
        raise MonitorError("monitoring completion target must be 28 days")
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


def _record_matches_policy(record: Mapping[str, Any], cell: ExperimentCell, policy, contracts) -> bool:
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
    return bool(
        isinstance(record.get("rollout_generation"), int)
        and not isinstance(record.get("rollout_generation"), bool)
        and record["rollout_generation"] > 0
        and record.get("effective_context") == record.get("effective_context_limit")
        and isinstance(record.get("endpoint_generation"), str)
        and bool(record.get("endpoint_generation"))
    )


def _empty_progress() -> dict[str, Any]:
    return {"expected": 0, "validated": 0, "timestamps": []}


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
                if _record_matches_policy(record, cell, policy, contracts)
            ]
            run_totals["untrusted_valid_rows"] += len(records) - len(trusted)
            run_totals["validated_qids"] += len(trusted)
            progress[stratum]["validated"] += len(trusted)
            rotation[rotation_key]["validated"] += len(trusted)
            timestamps = [
                timestamp
                for timestamp in (_finite_timestamp(row.get("timestamp")) for row in trusted)
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

        if expected_qids != int(run_spec["expected_qids"]):
            contract_errors.append(
                f"expected QID cardinality {expected_qids} != {run_spec['expected_qids']}"
            )
        if (
            run_totals["validated_qids"]
            != run_totals["completed_qids"]
            + run_totals["length_censored_qids"]
            + run_totals["protocol_censored_qids"]
        ):
            contract_errors.append("top-level termination partition is not exhaustive")
        if (
            run_totals["auxiliary_outcomes"]
            != run_totals["auxiliary_completed"]
            + run_totals["auxiliary_length_censored"]
            + run_totals["auxiliary_protocol_censored"]
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
                "remaining_qids": remaining,
                "window_qids": observed,
                "qids_per_day": rate,
                "eta_days": eta,
            }
        )
    expected_total = sum(row["expected_qids"] for row in rows)
    validated_total = sum(row["validated_qids"] for row in rows)
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


def _ledger_health(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "ledger.json"
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "available": False,
            "path": str(path),
            "starved_runs": [],
            "cached_state_counts": {},
        }
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "path": str(path),
            "error": str(exc),
            "starved_runs": [],
            "cached_state_counts": {},
        }
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
        "path": str(path),
        "poll_number": ledger.get("poll_number"),
        "updated_at": ledger.get("updated_at"),
        "starved_runs": starved,
        "cached_state_counts": dict(sorted(cached_states.items())),
    }


def collect_health_state(
    *,
    results_root: Path,
    state_dir: Path,
    config: Mapping[str, Any],
    now: float,
    probe_endpoints: bool,
) -> dict[str, Any]:
    state = control.load_control(state_dir)
    scheduler_snapshot = control.query_scheduler(now=now, tolerate_errors=True)
    live = control.live_status(state_dir, snapshot=scheduler_snapshot, now=now)
    server_pool = Path(state["immutable"]["server_pool_root"]).resolve()
    profile_names = tuple(config["fleet_replicas"])
    endpoints, generations = monitor_run._endpoint_stats(
        server_pool,
        profile_names,
        probe_http=probe_endpoints,
        timeout=3.0,
    )
    expected_replicas = config["fleet_replicas"]
    fleet_mismatches = {
        profile: {
            "expected": int(expected_replicas[profile]),
            "live": int(endpoints.get(profile, {}).get("live", 0)),
        }
        for profile in expected_replicas
        if int(endpoints.get(profile, {}).get("live", 0))
        != int(expected_replicas[profile])
    }
    fleet_generation = "schema5-fleet-v1:" + _sha256_value(generations)
    disk = shutil.disk_usage(results_root)
    scheduler = monitor_run._scheduler_stats()
    ledger = _ledger_health(state_dir)
    monitoring_root = state_dir / MONITORING_DIRNAME
    latest_semantic = monitoring_root / "semantic.latest.json"
    semantic_age_seconds: float | None = None
    if latest_semantic.is_file():
        semantic_age_seconds = max(0.0, now - latest_semantic.stat().st_mtime)
    elif live.get("open_throughput_epoch") is not None:
        started = live["open_throughput_epoch"].get("started_timestamp")
        if isinstance(started, (int, float)):
            semantic_age_seconds = max(0.0, now - float(started))
    return {
        "control": live,
        "scheduler": scheduler,
        "endpoints": endpoints,
        "fleet_generation": fleet_generation,
        "fleet_mismatches": fleet_mismatches,
        "ledger": ledger,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "free_fraction": disk.free / disk.total if disk.total else 0.0,
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
        + outcomes["protocol_censored_qids"],
        "auxiliary_partition_exact": outcomes["auxiliary_outcomes"]
        == outcomes["auxiliary_completed"]
        + outcomes["auxiliary_length_censored"]
        + outcomes["auxiliary_protocol_censored"],
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
    if control_live["desired_state"] == "running" and any(
        row["heartbeat_stale"] or not row["live_active"]
        for row in control_live["controllers"].values()
    ):
        findings.append(AlertFinding("monitor:controllers", "controller-health", "critical", "one or more running controllers are absent or stale"))
    if control_live["desired_state"] == "running" and health["fleet_mismatches"]:
        findings.append(AlertFinding("monitor:fleet", "fleet-capacity", "critical", f"canonical fleet mismatch: {health['fleet_mismatches']}"))
    if int(health["scheduler"].get("qos_max_memory_per_user_cell_holds", 0)):
        findings.append(AlertFinding("monitor:qos-memory", "qos-hold", "warning", "cell jobs are held by QOSMaxMemoryPerUser"))
    disk_policy = config["alerts"]
    if health["disk"]["free_bytes"] < int(disk_policy["minimum_disk_free_bytes"]) or health["disk"]["free_fraction"] < float(disk_policy["minimum_disk_free_fraction"]):
        findings.append(AlertFinding("monitor:disk", "disk-capacity", "critical", f"results filesystem headroom is low: {health['disk']}"))
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

    if cadence in {"semantic", "daily"}:
        semantic = report["semantic"]
        states = Counter(semantic["states"])
        outcomes = Counter(semantic["outcomes"])
        if states[CompletionState.CORRUPT.value] or states.get("validation_error", 0):
            findings.append(AlertFinding("monitor:corrupt", "corrupt-artifacts", "critical", f"corrupt/validation-error cells: {states[CompletionState.CORRUPT.value] + states.get('validation_error', 0)}"))
        if states[CompletionState.PERMANENT.value]:
            findings.append(AlertFinding("monitor:permanent", "permanent-failures", "critical", f"permanent cells: {states[CompletionState.PERMANENT.value]}"))
        if states[CompletionState.RETRYABLE.value]:
            findings.append(AlertFinding("monitor:retryable", "retryable-failures", "warning", f"retryable cells: {states[CompletionState.RETRYABLE.value]}"))
        if outcomes["untrusted_valid_rows"] or outcomes["contract_errors"]:
            findings.append(AlertFinding("monitor:untrusted", "untrusted-responses", "critical", f"untrusted rows={outcomes['untrusted_valid_rows']}, contract errors={outcomes['contract_errors']}"))
        if outcomes["protocol_censored_qids"]:
            findings.append(AlertFinding("monitor:context-protocol", "protocol-censors", "warning", f"protocol-censored top-level QIDs: {outcomes['protocol_censored_qids']}"))
        acceptance = report["throughput"]["acceptance"]
        if acceptance["observation_ready"] and not acceptance["projected_completion_within_target"]:
            findings.append(AlertFinding("monitor:throughput", "capacity-gate", "critical", f"48-hour QID throughput gate failed: {acceptance}"))
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
    if (
        live.get("desired_state") != "running"
        or scheduler.get("query_ok") is not True
        or not isinstance(live_scheduler, Mapping)
        or live_scheduler.get("squeue_ok") is not True
        or live_scheduler.get("sacct_ok") is not True
        or not isinstance(controllers, Mapping)
        or set(controllers) != {"dispatcher", "fleet_supervisor"}
        or health.get("fleet_mismatches") != {}
    ):
        return False
    return all(
        isinstance(row, Mapping)
        and row.get("live_active") is True
        and row.get("heartbeat_stale") is False
        for row in controllers.values()
    )


def persist_report(
    report: dict[str, Any],
    *,
    cadence: str,
    state_dir: Path,
    findings: Sequence[AlertFinding],
    send_email: bool,
    now: float,
) -> dict[str, Any]:
    state = control.load_control(state_dir)
    semantic = report.get("semantic", {})
    semantic_outcomes = Counter(semantic.get("outcomes", {}))
    integrity_clean = (
        semantic.get("scan_successful") is True
        and semantic_outcomes["contract_errors"] == 0
        and semantic_outcomes["untrusted_valid_rows"] == 0
        and semantic_outcomes["malformed_lines"] == 0
        and semantic_outcomes["duplicate_qids"] == 0
        and semantic_outcomes["unexpected_qids"] == 0
        and semantic_outcomes["invalid_rows"] == 0
        and semantic_outcomes["stale_unmanifested_dirs"] == 0
    )
    if (
        cadence in {"semantic", "daily"}
        and integrity_clean
        and state["desired_state"] == "running"
        and _production_poll_health_clean(report)
    ):
        control.record_successful_poll(
            state_dir,
            validated_qids=int(report["semantic"]["outcomes"].get("validated_qids", 0)),
            fleet_generation=report["health"]["fleet_generation"],
            strata_with_throughput=int(report["throughput"]["acceptance"]["strata_with_observed_throughput"]),
            now=now,
        )
        report["production_poll_recorded"] = True
    else:
        report["production_poll_recorded"] = False

    root = state_dir / MONITORING_DIRNAME
    history = root / cadence / f"{int(now)}.json"
    latest = root / f"{cadence}.latest.json"
    _atomic_json(history, report)
    _atomic_json(latest, report)

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
            now=now,
        )
    owned_keys = HEALTH_ALERT_KEYS if cadence == "health" else SEMANTIC_ALERT_KEYS
    for dedupe_key in sorted(owned_keys - current):
        control.resolve_alert(state_dir, dedupe_key=dedupe_key, now=now)
    return {"history": str(history), "latest": str(latest)}


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
        if cadence == "daily":
            report["final_acceptance"] = final_acceptance(semantic, projection, config)
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
                    kind="monitor-failure",
                    severity="critical",
                    message=f"{type(exc).__name__}: {exc}",
                    dedupe_key="monitor:untrusted",
                    send_email=args.send_email,
                    now=now,
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
