#!/usr/bin/env python
"""Semantic sweep health, scheduler, endpoint, and 48-hour ETA report.

The monitor uses the same immutable manifest and ``CompletionStatus`` contract as the
runner/dispatcher.  It is read-only: no artifact, registry, ledger, or Slurm state is
mutated.  Relative server-pool mappings resolve under ``ASYS_RESULTS_ROOT`` so a run such
as the seven-agent tranche can truthfully report the shared agent-count serving fleet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agents_scaling.config import (
    DEFAULT_DISPATCHER_STATE_DIRNAME,
    DEFAULT_RESULTS_ROOT,
    ExperimentCell,
)
from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog
from agents_scaling.experiment import io
from agents_scaling.experiment.analyze import _censor_accounting
from agents_scaling.experiment.completion import (
    CompletionState,
    get_completion_status,
    read_canonical_results,
)
from agents_scaling.experiment.manifest import ManifestSnapshot, load_manifest
from agents_scaling.serving import healthcheck, registry
from agents_scaling.serving.profiles import SERVING_PROFILES, serving_profile_for_cell
from agents_scaling.serving.registry import server_pool_generation


LOG_SIGNATURES: Mapping[str, re.Pattern[str]] = {
    "connection_or_timeout": re.compile(
        r"\b(?:APITimeoutError|ReadTimeout|APIConnectionError|RateLimitError|ConnectError)\b"
    ),
    "context_capacity": re.compile(r"\bContextCapacityError\b"),
    "generation_truncation": re.compile(r"\bGenerationTruncationError\b"),
    "retained_length_censor": re.compile(r"\blength[_-]censored\b"),
    "retained_protocol_censor": re.compile(r"\bprotocol[_-]censored\b"),
    "thinking_budget_protocol": re.compile(
        r"\b(?:ThinkingBudgetProtocolError|ServerResponseProtocolError)\b"
    ),
    "tokenizer_initialization": re.compile(
        r"\bTokenizerInitializationError\b|cannot import name ['\"]AutoTokenizer['\"]"
    ),
    "bad_request": re.compile(
        r"\bBadRequestError\b|(?:status|error code|HTTP)[^\n]{0,30}\b400\b",
        re.IGNORECASE,
    ),
}

CELL_JOB_PREFIXES = ("asys-cells", "asys-dispatch-")


@dataclass(frozen=True)
class CellObservation:
    cell: ExperimentCell
    state: str
    finished_at: float | None
    duration_h: float | None


def _pct(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(quantile * (len(ordered) - 1)))]


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _meta_timing(path: Path) -> tuple[float | None, float | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, None
    if not isinstance(value, dict):
        return None, None
    started = _finite_positive(value.get("started_at"))
    finished = _finite_positive(value.get("finished_at"))
    if started is None or finished is None or finished < started:
        return finished, None
    return finished, (finished - started) / 3600.0


def _cell_stats(
    run_root: Path,
    snapshot: ManifestSnapshot,
    *,
    question_catalog: VerifiedQuestionCatalog,
    code_version: str | None,
    server_pool_generations: Mapping[str, str],
    now: float,
) -> tuple[dict[str, Any], list[CellObservation]]:
    cells_root = run_root / "cells"
    states: Counter[str] = Counter()
    failure_classes: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    bad_lines = 0
    completed_questions = 0
    length_censored_questions = 0
    protocol_censored_questions = 0
    auxiliary_samples = 0
    auxiliary_completed_samples = 0
    auxiliary_length_censors = 0
    auxiliary_protocol_censors = 0
    auxiliary_censor_affected_cells = 0
    durations_h: list[float] = []
    observations: list[CellObservation] = []

    for cell in snapshot.cells:
        cdir = cells_root / cell.cell_id
        profile = serving_profile_for_cell(cell)
        questions = question_catalog.questions_for(cell)
        status = get_completion_status(
            cell,
            cdir,
            expected_qids=tuple(question.qid for question in questions),
            expected_questions=questions,
            verified_benchmark_contracts=question_catalog.frozen,
            verified_manifest=question_catalog.snapshot,
            serving_profile=profile.name,
            code_version=code_version,
            server_pool_generation=server_pool_generations.get(profile.name),
            now=now,
        )
        states[status.status.value] += 1
        bad_lines += status.malformed_lines
        completed_questions += status.completed_question_count
        length_censored_questions += status.length_censored_question_count
        protocol_censored_questions += status.protocol_censored_question_count
        # CompletionStatus deliberately keeps its stable top-level QID interface.  Read
        # the already validated canonical records only for cells whose frozen design
        # schedules auxiliary self-consistency draws, and expose those draws under a
        # separate denominator.  Slice to the status snapshot so a concurrently active
        # writer cannot make auxiliary counts run ahead of the top-level counters.
        if (
            status.valid_count
            and cell.topology.value == "single_agent"
            and cell.n_samples > 1
        ):
            canonical = read_canonical_results(
                cell,
                cdir,
                expected_qids=tuple(question.qid for question in questions),
                expected_questions=questions,
                verified_benchmark_contracts=question_catalog.frozen,
                verified_manifest=question_catalog.snapshot,
            )
            accounting = _censor_accounting(
                list(canonical.records[: status.valid_count])
            )
            auxiliary_samples += accounting["n_auxiliary_samples"]
            auxiliary_completed_samples += accounting[
                "n_auxiliary_completed_samples"
            ]
            auxiliary_length_censors += accounting[
                "n_auxiliary_length_censors"
            ]
            auxiliary_protocol_censors += accounting[
                "n_auxiliary_protocol_censors"
            ]
            auxiliary_censor_affected_cells += int(
                accounting["n_auxiliary_length_censors"]
                + accounting["n_auxiliary_protocol_censors"]
                > 0
            )
        if status.failure is not None:
            failure_classes[status.failure.classification] += 1
            failure_types[status.failure.last_error.get("type", "unknown")] += 1

        finished_at: float | None = None
        duration_h: float | None = None
        if status.is_complete:
            finished_at, duration_h = _meta_timing(cdir / "meta.json")
            if duration_h is not None:
                durations_h.append(duration_h)
        observations.append(
            CellObservation(cell, status.status.value, finished_at, duration_h)
        )

    present = (
        {path.name for path in cells_root.iterdir() if path.is_dir()}
        if cells_root.exists()
        else set()
    )
    stale = present - set(snapshot.ids)
    top_level_outcomes = (
        completed_questions
        + length_censored_questions
        + protocol_censored_questions
    )
    auxiliary_any_censors = (
        auxiliary_length_censors + auxiliary_protocol_censors
    )
    all_generation_outcomes = top_level_outcomes + auxiliary_samples
    return (
        {
            "manifest_cells": len(snapshot.cells),
            "manifest_sha256": snapshot.sha256,
            "benchmark_contracts_sha256": question_catalog.sidecar_sha256,
            "states": dict(sorted(states.items())),
            "complete": states[CompletionState.COMPLETE.value],
            "partial": states[CompletionState.PARTIAL.value],
            "corrupt": states[CompletionState.CORRUPT.value],
            "retryable": states[CompletionState.RETRYABLE.value],
            "permanent": states[CompletionState.PERMANENT.value],
            "active": states[CompletionState.ACTIVE.value],
            "missing": states[CompletionState.MISSING.value],
            "stale_unmanifested_dirs": len(stale),
            "bad_result_lines": bad_lines,
            "completed_question_outcomes": completed_questions,
            "length_censored_question_outcomes": length_censored_questions,
            "protocol_censored_question_outcomes": protocol_censored_questions,
            "top_level_any_censored_question_outcomes": (
                length_censored_questions + protocol_censored_questions
            ),
            "length_censor_rate": (
                length_censored_questions / top_level_outcomes
                if top_level_outcomes
                else 0.0
            ),
            "protocol_censor_rate": (
                protocol_censored_questions / top_level_outcomes
                if top_level_outcomes
                else 0.0
            ),
            # Explicit aliases make clear that the compatibility rates immediately
            # above use top-level QIDs, never auxiliary self-consistency draws.
            "top_level_length_censor_rate": (
                length_censored_questions / top_level_outcomes
                if top_level_outcomes
                else 0.0
            ),
            "top_level_protocol_censor_rate": (
                protocol_censored_questions / top_level_outcomes
                if top_level_outcomes
                else 0.0
            ),
            "top_level_any_censor_rate": (
                (length_censored_questions + protocol_censored_questions)
                / top_level_outcomes
                if top_level_outcomes
                else 0.0
            ),
            "auxiliary_sample_outcomes": auxiliary_samples,
            "auxiliary_completed_sample_outcomes": auxiliary_completed_samples,
            "auxiliary_length_censored_sample_outcomes": (
                auxiliary_length_censors
            ),
            "auxiliary_protocol_censored_sample_outcomes": (
                auxiliary_protocol_censors
            ),
            "auxiliary_any_censored_sample_outcomes": auxiliary_any_censors,
            "auxiliary_censor_affected_cells": auxiliary_censor_affected_cells,
            "auxiliary_length_censor_rate": (
                auxiliary_length_censors / auxiliary_samples
                if auxiliary_samples
                else None
            ),
            "auxiliary_protocol_censor_rate": (
                auxiliary_protocol_censors / auxiliary_samples
                if auxiliary_samples
                else None
            ),
            "auxiliary_any_censor_rate": (
                auxiliary_any_censors / auxiliary_samples
                if auxiliary_samples
                else None
            ),
            "all_generation_length_censor_rate": (
                (length_censored_questions + auxiliary_length_censors)
                / all_generation_outcomes
                if all_generation_outcomes
                else 0.0
            ),
            "all_generation_protocol_censor_rate": (
                (protocol_censored_questions + auxiliary_protocol_censors)
                / all_generation_outcomes
                if all_generation_outcomes
                else 0.0
            ),
            "failure_classes": dict(sorted(failure_classes.items())),
            "failure_error_types": dict(sorted(failure_types.items())),
            "wall_h_p50": statistics.median(durations_h) if durations_h else None,
            "wall_h_p95": _pct(durations_h, 0.95),
            "wall_h_max": max(durations_h) if durations_h else None,
        },
        observations,
    )


def _dispatcher_log_paths(state_dir: Path, run_id: str) -> set[Path]:
    """Resolve mixed dispatcher-array logs back to one ledger-pinned run."""
    try:
        ledger = json.loads((state_dir / "ledger.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return set()
    jobs = ledger.get("jobs") if isinstance(ledger, dict) else None
    if not isinstance(jobs, dict):
        return set()
    paths: set[Path] = set()
    for job_id, record in jobs.items():
        tasks = record.get("tasks") if isinstance(record, dict) else None
        if not isinstance(tasks, list):
            continue
        for index, task in enumerate(tasks):
            if isinstance(task, dict) and task.get("run_id") == run_id:
                paths.add(state_dir / "logs" / f"dispatch_{job_id}_{index}.out")
    return paths


def _log_stats(
    log_dirs: Iterable[Path], *, extra_paths: Iterable[Path] = ()
) -> dict[str, Any]:
    paths: set[Path] = {path for path in extra_paths if path.is_file()}
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        paths.update(log_dir.glob("cell_*_*.out"))
        paths.update(log_dir.glob("dispatch_*_*.out"))

    occurrences: Counter[str] = Counter()
    affected_files: dict[str, set[str]] = defaultdict(set)
    unreadable = 0
    for path in sorted(paths):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            unreadable += 1
            continue
        for label, pattern in LOG_SIGNATURES.items():
            count = len(pattern.findall(text))
            if count:
                occurrences[label] += count
                affected_files[label].add(str(path))

    signatures = {
        label: {
            "occurrences": occurrences[label],
            "affected_files": len(affected_files[label]),
        }
        for label in LOG_SIGNATURES
    }
    return {
        "cell_or_dispatch_log_files": len(paths),
        "unreadable_log_files": unreadable,
        "failure_signatures": signatures,
        # Compatibility with the old monitor key.
        "timeout_or_connection_errors": occurrences["connection_or_timeout"],
    }


def _endpoint_stats(
    server_pool_root: Path,
    profile_names: Iterable[str],
    *,
    probe_http: bool,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    out: dict[str, Any] = {}
    generations: dict[str, str] = {}
    for profile_name in sorted(set(profile_names)):
        if profile_name not in SERVING_PROFILES:
            continue
        raw = registry.list_servers(str(server_pool_root), profile_name)
        valid = [
            entry
            for entry in raw
            if registry.entry_matches_profile(entry, profile_name)
        ]
        provenance_valid = [
            entry
            for entry in raw
            if registry.entry_has_current_provenance(entry, profile_name)
        ]
        scheduler_live = registry.list_live_servers(
            str(server_pool_root),
            profile_name,
            probe_timeout=timeout,
            require_current_provenance=True,
        )
        generations[profile_name] = server_pool_generation(
            scheduler_live, profile_name=profile_name
        )
        row: dict[str, Any] = {
            "registered": len(raw),
            "profile_valid_registered": len(valid),
            "current_provenance_valid_registered": len(provenance_valid),
            "live": len(scheduler_live),
            "server_pool_generation": generations[profile_name],
        }
        if probe_http:
            row["http_healthy"] = sum(
                healthcheck.is_alive(entry.host, entry.port, timeout=timeout)
                for entry in scheduler_live
            )
        out[profile_name] = row
    return out, generations


def _is_cell_job(name: str) -> bool:
    return name == "asys-cells" or any(name.startswith(prefix) for prefix in CELL_JOB_PREFIXES)


def _scheduler_stats(*, timeout: float = 10.0) -> dict[str, Any]:
    user = os.environ.get("USER")
    if not user:
        return {"scope": "user", "query_ok": False, "error": "USER is unset"}
    try:
        proc = subprocess.run(
            [
                "squeue",
                "-u",
                user,
                "-h",
                "-r",
                "-o",
                "%i|%T|%R|%j|%m|%C",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"scope": "user", "query_ok": False, "error": str(exc)}
    if proc.returncode != 0:
        return {
            "scope": "user",
            "query_ok": False,
            "error": proc.stderr.strip()[:500],
        }

    total = running = pending = cell_total = cell_running = cell_pending = 0
    pending_reasons: Counter[str] = Counter()
    pending_cell_reasons: Counter[str] = Counter()
    malformed = 0
    for raw in proc.stdout.splitlines():
        if not raw.strip():
            continue
        fields = raw.split("|", 5)
        if len(fields) != 6:
            malformed += 1
            continue
        _job_id, state, reason, name, _memory, _cpus = (
            field.strip() for field in fields
        )
        reason = reason.strip("()") or "None"
        total += 1
        is_cell = _is_cell_job(name)
        if is_cell:
            cell_total += 1
        if state == "RUNNING":
            running += 1
            if is_cell:
                cell_running += 1
        elif state == "PENDING":
            pending += 1
            pending_reasons[reason] += 1
            if is_cell:
                cell_pending += 1
                pending_cell_reasons[reason] += 1

    memory_holds = sum(
        count
        for reason, count in pending_cell_reasons.items()
        if reason.startswith("QOSMaxMemoryPerUser")
    )
    return {
        "scope": "user",
        "query_ok": True,
        "total_jobs": total,
        "running_jobs": running,
        "pending_jobs": pending,
        "cell_jobs": cell_total,
        "running_cell_jobs": cell_running,
        "pending_cell_jobs": cell_pending,
        "pending_reasons": dict(sorted(pending_reasons.items())),
        "pending_cell_hold_reasons": dict(sorted(pending_cell_reasons.items())),
        "qos_max_memory_per_user_cell_holds": memory_holds,
        "malformed_squeue_rows": malformed,
    }


def _effective_agent_count(cell: ExperimentCell) -> int:
    return 1 if cell.topology.value == "single_agent" else cell.n_agents


def _iso_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _eta_row(
    observations: Sequence[CellObservation],
    *,
    window_start: float,
    window_days: float,
    now: float,
) -> dict[str, Any]:
    states = Counter(item.state for item in observations)
    complete = states[CompletionState.COMPLETE.value]
    remaining = len(observations) - complete
    window_completions = sum(
        item.state == CompletionState.COMPLETE.value
        and item.finished_at is not None
        and window_start <= item.finished_at <= now
        for item in observations
    )
    rate = window_completions / window_days if window_days > 0 else None
    eta_days = (
        0.0
        if remaining == 0
        else (remaining / rate if rate is not None and rate > 0 else None)
    )
    return {
        "cells": len(observations),
        "complete": complete,
        "remaining": remaining,
        "states": dict(sorted(states.items())),
        "window_completions": window_completions,
        "throughput_cells_per_day": rate,
        "eta_days": eta_days,
        "projected_completion_at": (
            _iso_timestamp(now + eta_days * 86400.0)
            if eta_days is not None
            else None
        ),
    }


def _eta_report(
    observations: Sequence[CellObservation],
    *,
    now: float,
    started_at: float | None,
    window_hours: float,
    min_observation_hours: float,
    target_days: float,
) -> dict[str, Any]:
    if min(window_hours, min_observation_hours, target_days) <= 0:
        raise ValueError("ETA window, minimum observation, and target must be positive")
    observation_hours = (
        max(0.0, (now - started_at) / 3600.0) if started_at is not None else 0.0
    )
    window_start = max(
        started_at if started_at is not None else now,
        now - window_hours * 3600.0,
    )
    window_days = max(0.0, (now - window_start) / 86400.0)

    grouped: dict[tuple[str, str, str, int], list[CellObservation]] = defaultdict(list)
    for item in observations:
        key = (
            item.cell.model_size,
            item.cell.reasoning_level.value,
            item.cell.topology.value,
            _effective_agent_count(item.cell),
        )
        grouped[key].append(item)

    strata: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: (value[0], value[1], value[2], value[3])):
        model_size, reasoning_level, topology, agent_count = key
        strata.append(
            {
                "model_size": model_size,
                "reasoning_level": reasoning_level,
                "topology": topology,
                "agent_count": agent_count,
                **_eta_row(
                    grouped[key],
                    window_start=window_start,
                    window_days=window_days,
                    now=now,
                ),
            }
        )

    overall = _eta_row(
        observations, window_start=window_start, window_days=window_days, now=now
    )
    remaining_rows = [row for row in strata if row["remaining"] > 0]
    unestimated = sum(row["eta_days"] is None for row in remaining_rows)
    estimated_etas = [
        float(row["eta_days"])
        for row in remaining_rows
        if row["eta_days"] is not None
    ]
    max_eta = max(estimated_etas, default=0.0) if not unestimated else None
    ready = started_at is not None and observation_hours >= min_observation_hours
    within_target = bool(
        ready
        and not unestimated
        and max_eta is not None
        and max_eta <= target_days
    )
    return {
        "started_at": _iso_timestamp(started_at) if started_at is not None else None,
        "observation_hours": observation_hours,
        "minimum_observation_hours": min_observation_hours,
        "window_hours": window_hours,
        "window_start": _iso_timestamp(window_start),
        "window_days_observed": window_days,
        "target_completion_days": target_days,
        "overall": overall,
        "strata": strata,
        "acceptance": {
            "observation_ready": ready,
            "remaining_strata": len(remaining_rows),
            "strata_without_observed_throughput": unestimated,
            "max_stratum_eta_days": max_eta,
            "projected_completion_within_target": within_target,
        },
    }


def _parse_timestamp(value: str) -> float:
    try:
        timestamp = float(value)
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamp = parsed.timestamp()
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise ValueError("timestamp must be finite and positive")
    return timestamp


def _eta_started_at(explicit: str | None, state_dir: Path) -> float | None:
    if explicit is not None:
        return _parse_timestamp(explicit)
    try:
        value = json.loads((state_dir / "ledger.json").read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        if "throughput_observation_started_at" in value:
            # A present-but-invalid rollout epoch fails closed.  ``created_at`` is only
            # a compatibility fallback for ledgers written before this field existed.
            return _finite_positive(value.get("throughput_observation_started_at"))
        return _finite_positive(value.get("created_at"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _dispatcher_state_dir(explicit: Path | None, results_root: Path) -> Path:
    """Resolve the durable dispatcher state directory shared by rollout tooling."""

    return (
        explicit.expanduser().resolve()
        if explicit is not None
        else results_root / DEFAULT_DISPATCHER_STATE_DIRNAME
    )


def _resolve_server_pool(
    run_id: str,
    assignment: str | None,
    *,
    results_root: Path,
    run_root: Path,
) -> Path:
    if assignment is None:
        return run_root.resolve()
    if "=" not in assignment:
        raise ValueError("--server-pool expects RUN_ID=POOL_ID_OR_PATH")
    assigned_run, pool_value = assignment.split("=", 1)
    if assigned_run != run_id or not pool_value:
        raise ValueError(
            f"--server-pool mapping must target {run_id!r}, got {assignment!r}"
        )
    candidate = Path(pool_value).expanduser()
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (results_root / candidate).resolve()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", default=None)
    parser.add_argument(
        "--server-pool",
        help="RUN_ID=POOL_ID_OR_PATH mapping, matching the global dispatcher interface",
    )
    parser.add_argument(
        "--dispatcher-state-dir",
        type=Path,
        help=(
            "dispatcher state/log directory "
            f"(default: RESULTS_ROOT/{DEFAULT_DISPATCHER_STATE_DIRNAME})"
        ),
    )
    parser.add_argument(
        "--probe-endpoints",
        action="store_true",
        help="add direct /health checks to scheduler-validated endpoint counts",
    )
    parser.add_argument("--health-timeout", type=float, default=3.0)
    parser.add_argument("--eta-window-hours", type=float, default=48.0)
    parser.add_argument("--eta-min-observation-hours", type=float, default=48.0)
    parser.add_argument("--eta-target-days", type=float, default=28.0)
    parser.add_argument(
        "--eta-started-at",
        help=(
            "unified-dispatch Unix timestamp or ISO-8601 time; defaults to the "
            "ledger throughput_observation_started_at rollout epoch (legacy ledgers "
            "fall back to created_at)"
        ),
    )
    args = parser.parse_args()

    now = time.time()
    results_root = Path(os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)).resolve()
    run_root = Path(args.run_root).resolve() if args.run_root else results_root / args.run_id
    state_dir = _dispatcher_state_dir(args.dispatcher_state_dir, results_root)
    server_pool_root = _resolve_server_pool(
        args.run_id,
        args.server_pool,
        results_root=results_root,
        run_root=run_root,
    )
    snapshot = load_manifest(run_root)
    question_catalog = VerifiedQuestionCatalog(run_root, snapshot=snapshot)
    profile_names = {
        serving_profile_for_cell(cell).name for cell in snapshot.cells
    }
    endpoints, server_pool_generations = _endpoint_stats(
        server_pool_root,
        profile_names,
        probe_http=args.probe_endpoints,
        timeout=args.health_timeout,
    )
    code_version = io.git_commit()
    cell_report, observations = _cell_stats(
        run_root,
        snapshot,
        question_catalog=question_catalog,
        code_version=code_version,
        server_pool_generations=server_pool_generations,
        now=now,
    )
    report = {
        "run_id": args.run_id,
        "run_root": str(run_root),
        "server_pool_root": str(server_pool_root),
        "code_version": code_version,
        **cell_report,
        **_log_stats(
            (run_root / "logs",),
            extra_paths=_dispatcher_log_paths(state_dir, args.run_id),
        ),
        "endpoints": endpoints,
        "scheduler": _scheduler_stats(),
        "eta": _eta_report(
            observations,
            now=now,
            started_at=_eta_started_at(args.eta_started_at, state_dir),
            window_hours=args.eta_window_hours,
            min_observation_hours=args.eta_min_observation_hours,
            target_days=args.eta_target_days,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
