#!/usr/bin/env python3
"""Run or verify the 41 immutable schema-5 smoke cells at concurrency one.

The command invokes the frozen dispatcher through the exact immutable harness Python,
using the same verified batch-task boundary as production.  Completed cells are skipped;
partial cells resume from durable QID checkpoints.  ``--apply`` is required to contact
servers.  The final report passes only when all 15 + 20 + 6 cells are semantically
complete under schema 5 with trusted provenance and zero censored/protocol incidents.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
for value in (REPO, SOURCE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from agents_scaling.benchmarks.runtime_contracts import VerifiedQuestionCatalog  # noqa: E402
from agents_scaling.experiment.analyze import _censor_accounting  # noqa: E402
from agents_scaling.experiment.artifact_policy import (  # noqa: E402
    ArtifactPolicyError,
    load_artifact_policy,
)
from agents_scaling.experiment.completion import (  # noqa: E402
    CompletionState,
    get_completion_status,
    read_canonical_results,
)
from agents_scaling.experiment.manifest import load_manifest  # noqa: E402
from agents_scaling.serving.profiles import serving_profile_for_cell  # noqa: E402
from scripts.init_schema5_smokes import (  # noqa: E402
    SMOKE_SUITES,
    SmokeInitializationError,
    _verify_suite,
)
from slurm import schema5_control  # noqa: E402


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUNNABLE_STATES = frozenset(
    {CompletionState.MISSING, CompletionState.PARTIAL, CompletionState.RETRYABLE}
)
_SUITE_ARTIFACT_NAMES = {
    "schema5_smoke_32b_long_v1": "long_32b_smoke",
    "schema5_smoke_selective_long_v1": "selective_long_smoke",
    "schema5_smoke_standard_canaries_v1": "standard_canary_smoke",
}
_PROVENANCE_ERROR_MARKERS = (
    "artifact policy",
    "authoritative policy",
    "endpoint_generation",
    "environment",
    "git_commit",
    "model contract",
    "model_revision",
    "provenance",
    "release_id",
    "rollout_generation",
    "schema-5",
    "serving_profile",
    "tokenizer_revision",
)
_PROTOCOL_FAILURE_TYPES = frozenset(
    {
        "GenerationProtocolCensorError",
        "ServerResponseProtocolError",
        "ThinkingBudgetProtocolError",
    }
)
_TRUNCATION_FAILURE_TYPES = frozenset({"GenerationTruncationError"})


class SmokeRunError(RuntimeError):
    """Smoke execution or semantic verification failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _exclusive_smoke_lock(state_root: Path) -> Iterator[None]:
    """Fence the complete smoke pass so two operators cannot create concurrency two."""

    state_root.mkdir(parents=True, exist_ok=True)
    path = state_root / ".schema5-smoke.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SmokeRunError(
                f"another schema-5 smoke runner owns the concurrency-one lock: {path}"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            (
                f"host={socket.gethostname()} pid={os.getpid()} "
                f"release=run_schema5_smokes_v1\n"
            ).encode("utf-8"),
        )
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _runtime_environment(
    *,
    policy: Any,
    immutable_pins_sha256: str,
    fleet_contract_sha256: str,
    rollout_generation: int,
) -> dict[str, str]:
    return {
        "ASYS_RELEASE_ID": policy.release.release_id,
        "ASYS_MODEL_CONTRACT_SHA256": policy.accepted_model_contract_sha256,
        "ASYS_FLEET_CONTRACT_SHA256": fleet_contract_sha256,
        "ASYS_HARNESS_ENVIRONMENT_SHA256": policy.environment.harness_sha256,
        "ASYS_SERVING_ENVIRONMENT_SHA256": policy.environment.serving_sha256,
        "ASYS_ROLLOUT_GENERATION": str(rollout_generation),
        "ASYS_IMMUTABLE_PINS_SHA256": immutable_pins_sha256,
        "ASYS_ARTIFACT_POLICY_SHA256": policy.file_sha256,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


def _task(
    *,
    run_root: Path,
    source_index: int,
    server_pool_root: Path,
    immutable_pins_sha256: str,
    fleet_contract_sha256: str,
    rollout_generation: int,
) -> dict[str, Any]:
    snapshot = load_manifest(run_root, verify_frozen=True)
    catalog = VerifiedQuestionCatalog(run_root, snapshot=snapshot)
    policy = load_artifact_policy(run_root, required=True)
    assert policy is not None
    cell = snapshot.cells[source_index]
    profile = serving_profile_for_cell(cell)
    return {
        "run_id": run_root.name,
        "run_root": str(run_root),
        "source_index": source_index,
        "cell_id": cell.cell_id,
        "config_hash": cell.config_hash(),
        "manifest_sha256": snapshot.sha256,
        "benchmark_contracts_sha256": catalog.sidecar_sha256,
        "model_size": cell.model_size,
        "serving_profile": profile.name,
        "fanout_cost": cell.n_agents,
        "server_pool_id": "schema5-v1",
        "server_run_id": str(server_pool_root),
        "server_pool_root": str(server_pool_root),
        "runtime_environment": _runtime_environment(
            policy=policy,
            immutable_pins_sha256=immutable_pins_sha256,
            fleet_contract_sha256=fleet_contract_sha256,
            rollout_generation=rollout_generation,
        ),
    }


def _sanitized_environment(*, hf_home: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for key in list(environment):
        if key.startswith("ASYS_"):
            environment.pop(key, None)
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(hf_home),
        }
    )
    return environment


def _load_immutable_context(
    *,
    state_root: Path,
    results_root: Path,
    server_pool_root: Path,
    release_worktree: Path,
    harness_prefix: Path,
    immutable_pins_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and fully verify the control-pinned release used by every smoke cell."""

    try:
        control = schema5_control.load_control(state_root, verify_files=True)
    except schema5_control.ControlError as exc:
        raise SmokeRunError(f"schema-5 immutable control verification failed: {exc}") from exc
    if control.get("immutable_sha256") != immutable_pins_sha256:
        raise SmokeRunError(
            "smoke immutable_pins_sha256 differs from the verified control state"
        )
    immutable = control["immutable"]
    expected_paths = {
        "results_root": results_root,
        "server_pool_root": server_pool_root,
        "release_worktree": release_worktree,
        "harness_environment_prefix": harness_prefix,
    }
    mismatches: dict[str, tuple[str, str]] = {}
    for field, observed in expected_paths.items():
        expected = Path(str(immutable[field])).expanduser().resolve()
        if expected != observed:
            mismatches[field] = (str(expected), str(observed))
    if mismatches:
        raise SmokeRunError(
            "smoke execution paths differ from immutable control pins: "
            + "; ".join(
                f"{field} expected {expected}, observed {observed}"
                for field, (expected, observed) in sorted(mismatches.items())
            )
        )
    for field in (
        "source_tree_sha256",
        "model_contract_sha256",
        "fleet_contract_sha256",
        "harness_environment_sha256",
        "serving_environment_sha256",
    ):
        if _SHA256.fullmatch(str(immutable.get(field, ""))) is None:
            raise SmokeRunError(f"immutable control {field} is not a lowercase SHA-256")
    return control, immutable


def _read_meta(run_root: Path, cell_id: str) -> Mapping[str, Any] | None:
    path = run_root / "cells" / cell_id / "meta.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        # The completion validator supplies the authoritative diagnostic.  Returning
        # ``None`` here prevents this secondary provenance census from masking it.
        return None
    return value if isinstance(value, dict) else None


def _provenance_errors(
    *,
    run_root: Path,
    cell: Any,
    status: Any,
    rows: Sequence[Mapping[str, Any]],
    immutable: Mapping[str, Any],
    policy: Any,
    rollout_generation: int,
) -> tuple[str, ...]:
    """Return explicit immutable-lineage violations in retained smoke artifacts."""

    errors = [
        str(error)
        for error in status.errors
        if any(marker in str(error).lower() for marker in _PROVENANCE_ERROR_MARKERS)
    ]
    expected_profile = serving_profile_for_cell(cell).name
    row_fixed = {
        "schema_version": 5,
        "release_id": immutable["release_id"],
        "environment_hash": immutable["harness_environment_sha256"],
        "model_contract_sha256": immutable["model_contract_sha256"],
        "serving_profile": expected_profile,
        "rollout_generation": rollout_generation,
    }
    for index, row in enumerate(rows):
        for field, expected in row_fixed.items():
            if row.get(field) != expected:
                errors.append(
                    f"record {index} {field} expected {expected!r}, "
                    f"observed {row.get(field)!r}"
                )
        if not isinstance(row.get("endpoint_generation"), str) or not row.get(
            "endpoint_generation"
        ):
            errors.append(f"record {index} lacks an exact endpoint_generation")

    meta = _read_meta(run_root, cell.cell_id)
    if meta is not None:
        meta_fixed = {
            "schema_version": 5,
            "release_id": immutable["release_id"],
            "environment_hash": immutable["harness_environment_sha256"],
            "serving_environment_hash": immutable["serving_environment_sha256"],
            "model_contract_sha256": immutable["model_contract_sha256"],
            "artifact_policy_sha256": policy.file_sha256,
            "git_commit": immutable["git_commit"],
            "serving_profile": expected_profile,
            "rollout_generation": rollout_generation,
        }
        for field, expected in meta_fixed.items():
            if meta.get(field) != expected:
                errors.append(
                    f"meta {field} expected {expected!r}, observed {meta.get(field)!r}"
                )
    elif status.status is CompletionState.COMPLETE:
        errors.append("complete cell lacks readable schema-5 meta.json")
    return tuple(dict.fromkeys(errors))


def _incident_accounting(rows: Sequence[Mapping[str, Any]], status: Any) -> dict[str, int]:
    """Count retained censors and unresolved current failure-ledger incidents."""

    accounting = _censor_accounting(list(rows))
    failure = status.failure
    classification = None if failure is None else failure.classification
    failure_type = (
        None if failure is None else str(failure.last_error.get("type", ""))
    )
    status_errors = tuple(str(error).lower() for error in status.errors)
    context_failure = int(
        classification == "context_capacity"
        or failure_type in {"ContextCapacityError", "GenerationTruncationError"}
    )
    protocol_failure = int(
        failure_type in _PROTOCOL_FAILURE_TYPES
        or any("protocol" in error for error in status_errors)
    )
    truncation_failure = int(
        failure_type in _TRUNCATION_FAILURE_TYPES
        or any("truncat" in error for error in status_errors)
    )
    top_length = int(accounting["n_length_censored_questions"])
    top_protocol = int(accounting["n_protocol_censored_questions"])
    auxiliary_length = int(accounting["n_auxiliary_length_censors"])
    auxiliary_protocol = int(accounting["n_auxiliary_protocol_censors"])
    return {
        "top_level_length_censored_qids": top_length,
        "top_level_protocol_censored_qids": top_protocol,
        "auxiliary_length_censored_draws": auxiliary_length,
        "auxiliary_protocol_censored_draws": auxiliary_protocol,
        "context_incidents": context_failure,
        "protocol_incidents": top_protocol + auxiliary_protocol + protocol_failure,
        "truncation_incidents": top_length + auxiliary_length + truncation_failure,
    }


def _runnable(status: Any) -> bool:
    if status.status not in _RUNNABLE_STATES:
        return False
    return status.status is not CompletionState.RETRYABLE or bool(
        status.eligible_for_retry
    )


def _run_task(
    task: Mapping[str, Any],
    *,
    state_root: Path,
    release_worktree: Path,
    harness_prefix: Path,
    hf_home: Path,
) -> dict[str, Any]:
    cell_id = str(task["cell_id"])
    batch_path = state_root / "batches" / str(task["run_id"]) / f"{cell_id}.json"
    log_path = state_root / "logs" / str(task["run_id"]) / f"{cell_id}.json"
    _atomic_json(batch_path, {"schema_version": 1, "tasks": [dict(task)]})
    python = (harness_prefix / "bin" / "python").resolve()
    dispatcher = (release_worktree / "slurm" / "dispatch_sweeps.py").resolve()
    command = [
        str(python),
        "-u",
        str(dispatcher),
        "run-task",
        "--batch-manifest",
        str(batch_path),
        "--index",
        "0",
        "--expected-release-root",
        str(release_worktree),
        "--expected-harness-prefix",
        str(harness_prefix),
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=_sanitized_environment(hf_home=hf_home),
    )
    report = {
        "cell_id": cell_id,
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "batch_manifest": str(batch_path),
        "batch_manifest_sha256": _sha256(batch_path),
    }
    _atomic_json(log_path, report)
    if process.returncode != 0:
        raise SmokeRunError(
            f"smoke worker failed for {cell_id} rc={process.returncode}; log={log_path}"
        )
    return report


def _status(
    run_root: Path,
    source_index: int,
    *,
    model_contract_path: Path,
) -> tuple[Any, list[dict[str, Any]]]:
    snapshot = load_manifest(run_root, verify_frozen=True)
    catalog = VerifiedQuestionCatalog(run_root, snapshot=snapshot)
    cell = snapshot.cells[source_index]
    questions = catalog.questions_for(cell)
    status = get_completion_status(
        cell,
        run_root / "cells" / cell.cell_id,
        expected_qids=tuple(question.qid for question in questions),
        expected_questions=questions,
        verified_benchmark_contracts=catalog.frozen,
        verified_manifest=catalog.snapshot,
        check_active=True,
        model_contract_path=model_contract_path,
    )
    rows: list[dict[str, Any]] = []
    if status.valid_count:
        rows = list(
            read_canonical_results(
                cell,
                run_root / "cells" / cell.cell_id,
                expected_qids=tuple(question.qid for question in questions),
                expected_questions=questions,
                verified_benchmark_contracts=catalog.frozen,
                verified_manifest=catalog.snapshot,
            ).records
        )
    return status, rows


def _write_json_with_checksum(path: Path, payload: Mapping[str, Any]) -> str:
    _atomic_json(path, payload)
    digest = _sha256(path)
    _atomic_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
    return digest


def _write_smoke_evidence(
    *,
    state_root: Path,
    immutable_sha256: str,
    suites: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Publish the controller's exact typed smoke readiness envelope."""

    observed_kinds = [str(suite["kind"]) for suite in suites]
    observed_runs = [str(suite["run_id"]) for suite in suites]
    if (
        len(suites) != 3
        or len(set(observed_kinds)) != 3
        or len(set(observed_runs)) != 3
        or set(observed_kinds) != set(_SUITE_ARTIFACT_NAMES.values())
        or set(observed_runs) != set(_SUITE_ARTIFACT_NAMES)
    ):
        raise SmokeRunError("smoke suite artifacts do not cover the exact three-suite contract")
    artifact_directory = state_root / "readiness" / "artifacts"
    references: list[dict[str, str]] = []
    by_kind = {str(suite["kind"]): suite for suite in suites}
    for kind in (
        "long_32b_smoke",
        "selective_long_smoke",
        "standard_canary_smoke",
    ):
        path = (artifact_directory / f"{kind}.json").resolve()
        digest = _write_json_with_checksum(path, by_kind[kind])
        references.append({"name": kind, "path": str(path), "sha256": digest})

    by_run = {str(suite["run_id"]): suite for suite in suites}
    metrics = {
        "long_32b_cells": int(
            by_run["schema5_smoke_32b_long_v1"]["schema5_complete_cells"]
        ),
        "selective_long_cells": int(
            by_run["schema5_smoke_selective_long_v1"]["schema5_complete_cells"]
        ),
        "standard_canary_cells": int(
            by_run["schema5_smoke_standard_canaries_v1"]["schema5_complete_cells"]
        ),
        "schema5_complete_cells": sum(
            int(suite["schema5_complete_cells"]) for suite in suites
        ),
        "context_incidents": sum(int(suite["context_incidents"]) for suite in suites),
        "protocol_incidents": sum(
            int(suite["protocol_incidents"]) for suite in suites
        ),
        "truncation_incidents": sum(
            int(suite["truncation_incidents"]) for suite in suites
        ),
        "provenance_failures": sum(
            int(suite["provenance_failures"]) for suite in suites
        ),
    }
    expected = {
        "long_32b_cells": 15,
        "selective_long_cells": 20,
        "standard_canary_cells": 6,
        "schema5_complete_cells": 41,
        "context_incidents": 0,
        "protocol_incidents": 0,
        "truncation_incidents": 0,
        "provenance_failures": 0,
    }
    envelope = {
        "schema_version": 2,
        "gate": "smoke_runs",
        "passed": metrics == expected and all(suite["passed"] is True for suite in suites),
        "immutable_sha256": immutable_sha256,
        "metrics": metrics,
        "artifacts": references,
    }
    _write_json_with_checksum(state_root / "readiness" / "smoke_runs.json", envelope)
    return envelope


def run_or_verify(
    *,
    results_root: Path,
    server_pool_root: Path,
    release_worktree: Path,
    harness_prefix: Path,
    state_root: Path,
    immutable_pins_sha256: str,
    rollout_generation: int,
    apply: bool,
) -> dict[str, Any]:
    results_root = results_root.expanduser().resolve()
    server_pool_root = server_pool_root.expanduser().resolve()
    release_worktree = release_worktree.expanduser().resolve()
    harness_prefix = harness_prefix.expanduser().resolve()
    state_root = state_root.expanduser().resolve()
    if _SHA256.fullmatch(immutable_pins_sha256) is None:
        raise SmokeRunError("immutable_pins_sha256 is not a lowercase SHA-256")
    if rollout_generation < 1:
        raise SmokeRunError("rollout_generation must be positive")
    if not server_pool_root.is_dir():
        raise SmokeRunError(f"canonical server pool is missing: {server_pool_root}")
    if not (harness_prefix / "bin/python").is_file():
        raise SmokeRunError(f"immutable harness Python is missing: {harness_prefix}")
    if not (release_worktree / "slurm/dispatch_sweeps.py").is_file():
        raise SmokeRunError(f"immutable dispatcher is missing: {release_worktree}")
    if not (state_root / schema5_control.CONTROL_FILENAME).is_file():
        raise SmokeRunError(f"schema-5 control state is missing: {state_root}")
    with _exclusive_smoke_lock(state_root):
        _, immutable = _load_immutable_context(
            state_root=state_root,
            results_root=results_root,
            server_pool_root=server_pool_root,
            release_worktree=release_worktree,
            harness_prefix=harness_prefix,
            immutable_pins_sha256=immutable_pins_sha256,
        )
        if set(_SUITE_ARTIFACT_NAMES) != {suite[0] for suite in SMOKE_SUITES}:
            raise SmokeRunError("smoke suite IDs differ from the closed readiness contract")

        suites: list[dict[str, Any]] = []
        execution_halted: str | None = None
        for run_id, relative_config, expected_cells in SMOKE_SUITES:
            run_root = results_root / run_id
            identity = _verify_suite(
                run_root,
                config_path=release_worktree / relative_config,
                expected_cells=expected_cells,
                expected_release_id=str(immutable["release_id"]),
                expected_git_commit=str(immutable["git_commit"]),
                expected_source_tree_sha256=str(immutable["source_tree_sha256"]),
                expected_harness_environment_sha256=str(
                    immutable["harness_environment_sha256"]
                ),
                expected_serving_environment_sha256=str(
                    immutable["serving_environment_sha256"]
                ),
                expected_model_contract_sha256=str(
                    immutable["model_contract_sha256"]
                ),
            )
            snapshot = load_manifest(run_root, verify_frozen=True)
            policy = load_artifact_policy(run_root, required=True)
            assert policy is not None
            cell_reports: list[dict[str, Any]] = []
            for index, cell in enumerate(snapshot.cells):
                status, rows = _status(
                    run_root,
                    index,
                    model_contract_path=(
                        release_worktree / "configs" / "model_contracts.v1.json"
                    ),
                )
                if apply and execution_halted is None and _runnable(status):
                    task = _task(
                        run_root=run_root,
                        source_index=index,
                        server_pool_root=server_pool_root,
                        immutable_pins_sha256=immutable_pins_sha256,
                        fleet_contract_sha256=str(
                            immutable["fleet_contract_sha256"]
                        ),
                        rollout_generation=rollout_generation,
                    )
                    try:
                        _run_task(
                            task,
                            state_root=state_root,
                            release_worktree=release_worktree,
                            harness_prefix=harness_prefix,
                            hf_home=Path(str(immutable["hf_home"])).resolve(),
                        )
                    except SmokeRunError as exc:
                        # Stop new draws after the first worker failure, but finish a
                        # read-only census so the operator receives attestable evidence.
                        execution_halted = str(exc)
                    status, rows = _status(
                        run_root,
                        index,
                        model_contract_path=(
                            release_worktree / "configs" / "model_contracts.v1.json"
                        ),
                    )

                incidents = _incident_accounting(rows, status)
                provenance_errors = _provenance_errors(
                    run_root=run_root,
                    cell=cell,
                    status=status,
                    rows=rows,
                    immutable=immutable,
                    policy=policy,
                    rollout_generation=rollout_generation,
                )
                failure = status.failure
                failure_summary = None
                if failure is not None:
                    failure_summary = {
                        "classification": failure.classification,
                        "disposition": failure.disposition,
                        "attempts": failure.attempts,
                        "last_error_type": failure.last_error.get("type"),
                        "last_error_message": failure.last_error.get("message"),
                    }
                cell_reports.append(
                    {
                        "cell_id": cell.cell_id,
                        "status": status.status.value,
                        "valid_qids": status.valid_count,
                        "expected_qids": status.expected_count,
                        **incidents,
                        "provenance_failures": int(bool(provenance_errors)),
                        "provenance_errors": list(provenance_errors),
                        "semantic_errors": list(status.errors),
                        "failure": failure_summary,
                    }
                )

            complete_cells = sum(
                row["status"] == CompletionState.COMPLETE.value for row in cell_reports
            )
            context_incidents = sum(row["context_incidents"] for row in cell_reports)
            protocol_incidents = sum(row["protocol_incidents"] for row in cell_reports)
            truncation_incidents = sum(
                row["truncation_incidents"] for row in cell_reports
            )
            provenance_failures = sum(
                row["provenance_failures"] for row in cell_reports
            )
            semantic_validation_failures = sum(
                bool(row["semantic_errors"]) for row in cell_reports
            )
            suite_passed = (
                complete_cells == expected_cells
                and context_incidents == 0
                and protocol_incidents == 0
                and truncation_incidents == 0
                and provenance_failures == 0
                and semantic_validation_failures == 0
                and execution_halted is None
            )
            suites.append(
                {
                    "schema_version": 1,
                    "kind": _SUITE_ARTIFACT_NAMES[run_id],
                    "passed": suite_passed,
                    "immutable_sha256": immutable_pins_sha256,
                    "run_id": run_id,
                    "expected_cells": expected_cells,
                    "schema5_complete_cells": complete_cells,
                    "context_incidents": context_incidents,
                    "protocol_incidents": protocol_incidents,
                    "truncation_incidents": truncation_incidents,
                    "provenance_failures": provenance_failures,
                    "semantic_validation_failures": semantic_validation_failures,
                    "top_level_length_censored_qids": sum(
                        row["top_level_length_censored_qids"] for row in cell_reports
                    ),
                    "top_level_protocol_censored_qids": sum(
                        row["top_level_protocol_censored_qids"] for row in cell_reports
                    ),
                    "auxiliary_length_censored_draws": sum(
                        row["auxiliary_length_censored_draws"] for row in cell_reports
                    ),
                    "auxiliary_protocol_censored_draws": sum(
                        row["auxiliary_protocol_censored_draws"] for row in cell_reports
                    ),
                    "concurrency": 1,
                    "resumable": True,
                    "execution_halted": execution_halted is not None,
                    "execution_halt_reason": execution_halted,
                    "provenance": {
                        "release_id": immutable["release_id"],
                        "git_commit": immutable["git_commit"],
                        "source_tree_sha256": immutable["source_tree_sha256"],
                        "harness_environment_sha256": immutable[
                            "harness_environment_sha256"
                        ],
                        "serving_environment_sha256": immutable[
                            "serving_environment_sha256"
                        ],
                        "model_contract_sha256": immutable[
                            "model_contract_sha256"
                        ],
                        "fleet_contract_sha256": immutable[
                            "fleet_contract_sha256"
                        ],
                        "server_pool_root": str(server_pool_root),
                        "rollout_generation": rollout_generation,
                    },
                    "suite_identity": identity,
                    "cells": cell_reports,
                }
            )
        return _write_smoke_evidence(
            state_root=state_root,
            immutable_sha256=immutable_pins_sha256,
            suites=suites,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--server-pool-root", required=True, type=Path)
    parser.add_argument("--release-worktree", required=True, type=Path)
    parser.add_argument("--harness-prefix", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--immutable-pins-sha256", required=True)
    parser.add_argument("--rollout-generation", required=True, type=int)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run_or_verify(
            results_root=args.results_root,
            server_pool_root=args.server_pool_root,
            release_worktree=args.release_worktree,
            harness_prefix=args.harness_prefix,
            state_root=args.state_root,
            immutable_pins_sha256=args.immutable_pins_sha256,
            rollout_generation=args.rollout_generation,
            apply=args.apply,
        )
    except (
        ArtifactPolicyError,
        OSError,
        SmokeInitializationError,
        SmokeRunError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
