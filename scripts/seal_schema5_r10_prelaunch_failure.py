#!/usr/bin/env python3
"""Seal r10's deterministic r9-failure-sealer wrapper defect for r11."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import types
from typing import Any, Mapping


SCHEMA_VERSION = 1
PROTOCOL = "schema5-v1.2-r10-r9-failure-sealer-wrapper-failure-v1"
CLASSIFICATION = "deterministic_prelaunch_r9_sealer_stderr_prefix_mismatch"
R10_TAG = "sweep-recovery-schema5-v1.2-r10"
R10_NAMESPACE = "schema5-v1.2-r10"
R10_COMMIT = "ea2e9b8adc33e4f16dd4d6672bc2ca4c65b78a11"
R10_TAG_OBJECT = "1b1f568ae68bfac09248b9d4b03c7d9a613a8d1d"
R10_DURABLE_MARKER = "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R10_COMPLETE.json"
R10_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r10"
R9_EVIDENCE_RELATIVE = Path(
    "prelaunch_failures/schema5-v1.2-r9-offline-diagnostic"
)
R9_INTENT_NAME = "PRELAUNCH_OFFLINE_DIAGNOSTIC_FAILURE_INTENT.json"
R9_MARKER_NAME = "PRELAUNCH_OFFLINE_DIAGNOSTIC_FAILURE_SEALED.json"
R9_VOLATILE_ROOT = Path("/tmp/schema5-r3-prelaunch-mabdel03-r9")
OPERATOR_DIAGNOSTIC_ROOT = Path(
    "/tmp/schema5-r3-prelaunch-mabdel03-r9-diagnose"
)
R10_FAILURE_RELATIVE = Path(
    "prelaunch_failures/schema5-v1.2-r10-r9-sealer"
)
MARKER_NAME = "PRELAUNCH_R9_SEALER_FAILURE_SEALED.json"
INTENT_NAME = "PRELAUNCH_R9_SEALER_FAILURE_INTENT.json"
OBSERVED_EVIDENCE_NAME = "observed_incomplete_r9_seal"
OBSERVED_VOLATILE_NAME = "observed_volatile_probe"
DIAGNOSTIC_NAME = "operator_diagnostic_non_authoritative"
REPRODUCED_EVIDENCE_NAME = "reproduced_incomplete_r9_seal"
REPRODUCED_VOLATILE_NAME = "reproduced_volatile_probe"
STDOUT_NAME = "r10-sealer.stdout"
STDERR_NAME = "r10-sealer.stderr"
EXPECTED_STDERR = (
    "ERROR: exact r9 offline recorder rejection reproduction drifted\n"
)
_CHUNK_SIZE = 8 * 1024 * 1024


class R10FailureSealError(RuntimeError):
    """The r10 deterministic prelaunch failure cannot be sealed exactly."""


def _load_helpers():
    path = Path(__file__).resolve().with_name(
        "seal_schema5_r8_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r10_failure_helpers", path
    )
    if spec is None or spec.loader is None:
        raise R10FailureSealError("cannot load marker-last evidence helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_helpers()
MarkerLastEvidenceError = (
    _helpers.R8FailureSealError,
    *_helpers.MarkerLastEvidenceError,
)
_canonical_bytes = _helpers._canonical_bytes
_identity = _helpers._identity
_safe_directory = _helpers._safe_directory
_read_json = _helpers._read_json
_file_ref = _helpers._file_ref
_publish = _helpers._publish
_fsync_directory = _helpers._fsync_directory
_run_git = _helpers._helpers._run_git
_require_recursively_read_only = (
    _helpers._helpers._require_recursively_read_only
)


def _git(checkout: Path, *arguments: str) -> str:
    return _run_git(checkout, *arguments, optional_locks=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_inventory(root: Path) -> dict[str, Any]:
    root = _safe_directory(root, description="r10 failure evidence tree")
    rows: list[dict[str, Any]] = []
    for path in [root, *sorted(root.rglob("*"))]:
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        row: dict[str, Any] = {"path": relative}
        if stat.S_ISDIR(metadata.st_mode):
            row["type"] = "directory"
        elif stat.S_ISREG(metadata.st_mode):
            row.update(
                {
                    "type": "file",
                    "size": metadata.st_size,
                    "sha256": _sha256_file(path),
                }
            )
        elif stat.S_ISLNK(metadata.st_mode):
            row.update({"type": "symlink", "target": os.readlink(path)})
        else:
            raise R10FailureSealError(f"unsupported evidence entry: {path}")
        rows.append(row)
    return {
        "entry_count": len(rows),
        "rows": rows,
        "inventory_sha256": hashlib.sha256(
            _canonical_bytes(rows)
        ).hexdigest(),
    }


def _remove_write_bits(root: Path) -> None:
    for path in sorted(
        [root, *root.rglob("*")],
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        metadata = path.lstat()
        if not stat.S_ISLNK(metadata.st_mode):
            os.chmod(path, stat.S_IMODE(metadata.st_mode) & ~0o222)


def _release_binding(recovery: Path) -> dict[str, Any]:
    checkout = _safe_directory(
        recovery / R10_CHECKOUT, description="r10 checkout"
    )
    tag_ref = f"refs/tags/{R10_TAG}"
    if (
        _git(checkout, "cat-file", "-t", tag_ref) != "tag"
        or _git(checkout, "rev-parse", tag_ref) != R10_TAG_OBJECT
        or _git(checkout, "rev-parse", f"{tag_ref}^{{commit}}")
        != R10_COMMIT
        or _git(checkout, "rev-parse", "HEAD") != R10_COMMIT
        or _git(
            checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        or (checkout / ".git/objects/info/alternates").exists()
    ):
        raise R10FailureSealError("r10 checkout identity drifted")
    marker_path = recovery / R10_DURABLE_MARKER
    marker, marker_raw = _read_json(
        marker_path, description="r10 durable release marker"
    )
    bundle = Path(str(marker.get("bundle_path", "")))
    if (
        marker.get("passed") is not True
        or marker.get("release_tag") != R10_TAG
        or marker.get("chain_namespace") != R10_NAMESPACE
        or marker.get("release_git_commit") != R10_COMMIT
        or marker.get("release_tag_object") != R10_TAG_OBJECT
        or marker.get("remote_commit") != R10_COMMIT
        or marker.get("remote_peeled_commit") != R10_COMMIT
        or marker.get("remote_tag_object") != R10_TAG_OBJECT
        or not bundle.is_file()
        or bundle.is_symlink()
        or _sha256_file(bundle) != marker.get("bundle_sha256")
    ):
        raise R10FailureSealError("r10 durable release identity drifted")
    return {
        "checkout": str(checkout),
        "release_tag": R10_TAG,
        "release_git_commit": R10_COMMIT,
        "release_tag_object": R10_TAG_OBJECT,
        "failed_sealer": _file_ref(
            checkout / "scripts/seal_schema5_r9_prelaunch_failure.py",
            description="immutable r10 failed sealer",
        ),
        "durable_marker": {
            "path": str(marker_path),
            "sha256": hashlib.sha256(marker_raw).hexdigest(),
            "size": len(marker_raw),
            "marker_id": marker.get("marker_id"),
        },
        "durable_bundle": _file_ref(
            bundle, description="r10 durable Git bundle"
        ),
    }


def _load_r10_sealer(recovery: Path):
    path = (
        recovery
        / R10_CHECKOUT
        / "scripts/seal_schema5_r9_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r10_failed_r9_sealer", path
    )
    if spec is None or spec.loader is None:
        raise R10FailureSealError("cannot load immutable r10 failed sealer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise R10FailureSealError("r10 failed-sealer source drifted")
    return module


def _validate_incomplete_evidence(
    recovery: Path, root: Path, scheduler_user: str
) -> dict[str, Any]:
    root = _safe_directory(root, description="incomplete r10 r9 seal")
    if {path.name for path in root.iterdir()} != {R9_INTENT_NAME}:
        raise R10FailureSealError("r10 incomplete r9-seal topology drifted")
    if (root / R9_MARKER_NAME).exists() or (root / R9_MARKER_NAME).is_symlink():
        raise R10FailureSealError("r10 unexpectedly published its r9 seal")
    intent, intent_raw = _read_json(
        root / R9_INTENT_NAME, description="r10 incomplete r9-seal intent"
    )
    module = _load_r10_sealer(recovery)
    try:
        module._validate_intent(
            intent,
            recovery,
            recovery / R9_EVIDENCE_RELATIVE,
            scheduler_user,
        )
    except Exception as exc:
        raise R10FailureSealError(
            f"r10 incomplete r9-seal intent is invalid: {exc}"
        ) from exc
    return {
        "path": str(root),
        "intent": {
            "path": str(root / R9_INTENT_NAME),
            "sha256": hashlib.sha256(intent_raw).hexdigest(),
            "size": len(intent_raw),
            "intent_id": intent.get("intent_id"),
        },
        "inventory": _portable_inventory(root),
        "completion_marker_published": False,
    }


def _validate_volatile_tree(recovery: Path, root: Path) -> dict[str, Any]:
    root = _safe_directory(root, description="r10 failed volatile probe")
    module = _load_r10_sealer(recovery)
    saved = module.R9_PROBE_ROOT
    try:
        module.R9_PROBE_ROOT = root
        report = module._validate_original_probe_tree(recovery)
    except Exception as exc:
        raise R10FailureSealError(
            f"r10 failed volatile probe is invalid: {exc}"
        ) from exc
    finally:
        module.R9_PROBE_ROOT = saved
    if report.get("offline_envelope_published") is not False:
        raise R10FailureSealError("r10 unexpectedly published an offline envelope")
    return report


def _validate_operator_diagnostic(root: Path) -> dict[str, Any]:
    root = _safe_directory(
        root, description="non-authoritative operator diagnostic"
    )
    if any(
        candidate.name.startswith(
            "FAILURE_OFFLINE_CLONE_UNSEEDED_RELEASE_LOCAL_CACHE"
        )
        for candidate in root.iterdir()
    ):
        raise R10FailureSealError(
            "operator diagnostic unexpectedly published an envelope"
        )
    cache = _safe_directory(
        root / "empty-conda-pkgs",
        description="operator diagnostic partial cache",
    )
    partials = sorted(
        path.name for path in cache.iterdir() if path.name.endswith(".partial")
    )
    if not partials:
        raise R10FailureSealError("operator diagnostic has no partial artifacts")
    return {
        "path": str(root),
        "authoritative_failure_evidence": False,
        "purpose": "identified_exact_r9_cli_error_prefix",
        "partial_artifact_count": len(partials),
        "inventory": _portable_inventory(root),
    }


def _scheduler_quiescence(user: str) -> dict[str, Any]:
    process = subprocess.run(
        ["/usr/bin/squeue", "-h", "-r", "-u", user, "-o", "%i|%j|%k|%T"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
    )
    if process.returncode != 0:
        raise R10FailureSealError(f"squeue failed: {process.stderr.strip()}")
    matching = [
        row
        for row in process.stdout.splitlines()
        if "s5v12r10" in row
        or "schema5-v1.2-r10" in row
        or "sweep-recovery-schema5-v1.2-r10" in row
    ]
    if matching:
        raise R10FailureSealError(f"r10 scheduler jobs remain live: {matching}")
    return {"scheduler_user": user, "matching_r10_jobs": []}


def _scientific_state(recovery: Path) -> dict[str, Any]:
    results = recovery.parent.parent
    paths = [
        results / ".dispatcher-schema5-v1",
        results / "server_pools/schema5-v1",
        results / "full_sweep_schema5_v1",
        results / "full_sweep_agent_counts_schema5_v1",
        results / "full_sweep_agent_count_7_schema5_v1",
        recovery / "slurm_canaries/schema5-v1.2-r10",
        recovery / "materialization_pilots/schema5-v1.2-r10",
        recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R10.json",
    ]
    present = [
        str(path) for path in paths if path.exists() or path.is_symlink()
    ]
    if present:
        raise R10FailureSealError(
            f"r10 wrapper failure is not zero-result evidence: {present}"
        )
    return {
        "captured_before_r11_production_outputs": True,
        "result_mutation_count": 0,
        "scheduler_job_count": 0,
        "required_absent_paths_at_seal": [str(path) for path in paths],
    }


def _r10_reproduction_argv(
    recovery: Path, scheduler_user: str
) -> list[str]:
    evidence = recovery / R9_EVIDENCE_RELATIVE
    return [
        sys.executable,
        "-I",
        str(
            recovery
            / R10_CHECKOUT
            / "scripts/seal_schema5_r9_prelaunch_failure.py"
        ),
        "seal",
        "--evidence-root",
        str(evidence),
        "--recovery-root",
        str(recovery),
        "--scheduler-user",
        scheduler_user,
        "--apply",
    ]


def _temporary_quarantine_paths() -> tuple[Path, ...]:
    return (
        Path(f"{R9_VOLATILE_ROOT}-r10-failed-observed"),
        Path(f"{R9_VOLATILE_ROOT}-r10-failed-reproduced"),
        Path(f"{OPERATOR_DIAGNOSTIC_ROOT}-r10-observed"),
    )


def _validate_temporary_quarantines_absent() -> list[str]:
    paths = _temporary_quarantine_paths()
    present = [
        str(path) for path in paths if path.exists() or path.is_symlink()
    ]
    if present:
        raise R10FailureSealError(
            f"r10 temporary quarantine destination already exists: {present}"
        )
    return [str(path) for path in paths]


def _intent_payload(
    recovery: Path,
    evidence: Path,
    scheduler_user: str,
    observed_evidence: Mapping[str, Any],
    observed_volatile: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": f"{PROTOCOL}-intent",
        "classification": CLASSIFICATION,
        "source_release": _release_binding(recovery),
        "observed_incomplete_evidence": observed_evidence,
        "observed_volatile_probe": observed_volatile,
        "operator_diagnostic": diagnostic,
        "temporary_quarantine_destinations": (
            _validate_temporary_quarantines_absent()
        ),
        "reproduction": {
            "argv": _r10_reproduction_argv(recovery, scheduler_user),
            "cwd": str(recovery / R10_CHECKOUT),
            "expected_returncode": 2,
            "expected_stdout": "",
            "expected_stderr": EXPECTED_STDERR,
        },
        "scientific_state": _scientific_state(recovery),
        "scheduler": _scheduler_quiescence(scheduler_user),
        "evidence_root": str(evidence),
    }
    payload["intent_id"] = _identity(payload, "intent_id")
    return payload


def _validate_intent(
    intent: Mapping[str, Any],
    recovery: Path,
    evidence: Path,
    scheduler_user: str,
) -> None:
    if (
        intent.get("schema_version") != SCHEMA_VERSION
        or intent.get("protocol") != f"{PROTOCOL}-intent"
        or intent.get("classification") != CLASSIFICATION
        or intent.get("intent_id") != _identity(intent, "intent_id")
        or intent.get("source_release") != _release_binding(recovery)
        or intent.get("evidence_root") != str(evidence)
        or intent.get("scheduler") != _scheduler_quiescence(scheduler_user)
        or intent.get("temporary_quarantine_destinations")
        != [str(path) for path in _temporary_quarantine_paths()]
        or intent.get("reproduction")
        != {
            "argv": _r10_reproduction_argv(recovery, scheduler_user),
            "cwd": str(recovery / R10_CHECKOUT),
            "expected_returncode": 2,
            "expected_stdout": "",
            "expected_stderr": EXPECTED_STDERR,
        }
    ):
        raise R10FailureSealError("r10 failure-seal intent drifted")
    state = intent.get("scientific_state")
    if (
        not isinstance(state, dict)
        or state.get("captured_before_r11_production_outputs") is not True
        or state.get("result_mutation_count") != 0
        or state.get("scheduler_job_count") != 0
    ):
        raise R10FailureSealError("r10 scientific-state evidence drifted")


def _move_directory(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise R10FailureSealError(f"archive destination already exists: {destination}")
    try:
        os.replace(source, destination)
    except OSError as exc:
        raise R10FailureSealError(
            f"cannot transactionally archive {source}: {exc}"
        ) from exc
    _fsync_directory(destination.parent)


def _archive_volatile(
    source: Path, temporary_quarantine: Path, durable_archive: Path
) -> dict[str, Any]:
    _move_directory(source, temporary_quarantine)
    shutil.copytree(
        temporary_quarantine,
        durable_archive,
        symlinks=True,
        copy_function=shutil.copy2,
    )
    source_inventory = _portable_inventory(temporary_quarantine)
    archive_inventory = _portable_inventory(durable_archive)
    if source_inventory != archive_inventory:
        raise R10FailureSealError("durable volatile archive drifted")
    _remove_write_bits(temporary_quarantine)
    _remove_write_bits(durable_archive)
    _require_recursively_read_only(temporary_quarantine)
    _require_recursively_read_only(durable_archive)
    return {
        "path": str(durable_archive),
        "temporary_quarantine": str(temporary_quarantine),
        "durable_archive": str(durable_archive),
        "inventory": archive_inventory,
        "recursively_read_only": True,
    }


def _execute_proof(
    evidence: Path,
    recovery: Path,
    intent: Mapping[str, Any],
    scheduler_user: str,
) -> dict[str, Any]:
    observed_evidence = recovery / R9_EVIDENCE_RELATIVE
    observed_volatile = R9_VOLATILE_ROOT
    diagnostic = OPERATOR_DIAGNOSTIC_ROOT
    _validate_temporary_quarantines_absent()
    archived_observed_evidence = evidence / OBSERVED_EVIDENCE_NAME
    _move_directory(observed_evidence, archived_observed_evidence)
    observed_volatile_archive = _archive_volatile(
        observed_volatile,
        Path(f"{observed_volatile}-r10-failed-observed"),
        evidence / OBSERVED_VOLATILE_NAME,
    )
    diagnostic_archive = _archive_volatile(
        diagnostic,
        Path(f"{diagnostic}-r10-observed"),
        evidence / DIAGNOSTIC_NAME,
    )
    if (
        _validate_incomplete_evidence(
            recovery, archived_observed_evidence, scheduler_user
        )["inventory"]
        != intent["observed_incomplete_evidence"]["inventory"]
        or _validate_volatile_tree(
            recovery, evidence / OBSERVED_VOLATILE_NAME
        )["inventory"]
        != intent["observed_volatile_probe"]["inventory"]
        or _validate_operator_diagnostic(
            evidence / DIAGNOSTIC_NAME
        )["inventory"]
        != intent["operator_diagnostic"]["inventory"]
    ):
        raise R10FailureSealError("observed r10 failure archive drifted")
    process = subprocess.run(
        list(intent["reproduction"]["argv"]),
        cwd=intent["reproduction"]["cwd"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=1_800,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    )
    if (
        process.returncode != 2
        or process.stdout != b""
        or process.stderr.decode("utf-8", errors="strict") != EXPECTED_STDERR
    ):
        raise R10FailureSealError("exact r10 wrapper failure reproduction drifted")
    stdout_path = evidence / STDOUT_NAME
    stderr_path = evidence / STDERR_NAME
    _publish(stdout_path, process.stdout)
    _publish(stderr_path, process.stderr)
    reproduced_evidence = recovery / R9_EVIDENCE_RELATIVE
    reproduced_volatile = R9_VOLATILE_ROOT
    reproduced_evidence_state = _validate_incomplete_evidence(
        recovery, reproduced_evidence, scheduler_user
    )
    reproduced_volatile_state = _validate_volatile_tree(
        recovery, reproduced_volatile
    )
    archived_reproduced_evidence = evidence / REPRODUCED_EVIDENCE_NAME
    _move_directory(reproduced_evidence, archived_reproduced_evidence)
    reproduced_volatile_archive = _archive_volatile(
        reproduced_volatile,
        Path(f"{reproduced_volatile}-r10-failed-reproduced"),
        evidence / REPRODUCED_VOLATILE_NAME,
    )
    if (
        _portable_inventory(archived_reproduced_evidence)
        != reproduced_evidence_state["inventory"]
        or _portable_inventory(evidence / REPRODUCED_VOLATILE_NAME)
        != reproduced_volatile_state["inventory"]
    ):
        raise R10FailureSealError("reproduced r10 failure archive drifted")
    for root in (
        archived_observed_evidence,
        archived_reproduced_evidence,
    ):
        _remove_write_bits(root)
        _require_recursively_read_only(root)
    return {
        "returncode": process.returncode,
        "stdout": _file_ref(stdout_path, description="r10 sealer stdout"),
        "stderr": _file_ref(stderr_path, description="r10 sealer stderr"),
        "exact_error_matched": True,
        "failed_condition": (
            "r9_cli_stderr_used_schema5_evidence_prefix_but_r10_expected_bare_error"
        ),
        "observed_incomplete_evidence": {
            "path": str(archived_observed_evidence),
            "inventory": _portable_inventory(archived_observed_evidence),
            "recursively_read_only": True,
        },
        "observed_volatile_probe": observed_volatile_archive,
        "operator_diagnostic": diagnostic_archive
        | {"authoritative_failure_evidence": False},
        "reproduced_incomplete_evidence": {
            "path": str(archived_reproduced_evidence),
            "inventory": _portable_inventory(archived_reproduced_evidence),
            "recursively_read_only": True,
        },
        "reproduced_volatile_probe": reproduced_volatile_archive,
        "canonical_r9_evidence_path_absent": True,
        "canonical_r9_volatile_path_absent": True,
        "known_scheduler_job_ids": [],
        "result_mutation_count": 0,
    }


def _verify_archive(root: Path, record: Mapping[str, Any]) -> None:
    if (
        record.get("path") != str(root)
        or record.get("inventory") != _portable_inventory(root)
        or record.get("recursively_read_only") is not True
    ):
        raise R10FailureSealError(f"r10 archive binding drifted: {root}")
    _require_recursively_read_only(root)


def _verify_proof(
    evidence: Path,
    recovery: Path,
    intent: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> None:
    expected_keys = {
        "returncode",
        "stdout",
        "stderr",
        "exact_error_matched",
        "failed_condition",
        "observed_incomplete_evidence",
        "observed_volatile_probe",
        "operator_diagnostic",
        "reproduced_incomplete_evidence",
        "reproduced_volatile_probe",
        "canonical_r9_evidence_path_absent",
        "canonical_r9_volatile_path_absent",
        "known_scheduler_job_ids",
        "result_mutation_count",
    }
    if (
        set(proof) != expected_keys
        or proof.get("returncode") != 2
        or proof.get("exact_error_matched") is not True
        or proof.get("failed_condition")
        != "r9_cli_stderr_used_schema5_evidence_prefix_but_r10_expected_bare_error"
        or proof.get("canonical_r9_evidence_path_absent") is not True
        or proof.get("canonical_r9_volatile_path_absent") is not True
        or proof.get("known_scheduler_job_ids") != []
        or proof.get("result_mutation_count") != 0
        or (recovery / R9_EVIDENCE_RELATIVE).exists()
        or (recovery / R9_EVIDENCE_RELATIVE).is_symlink()
        or R9_VOLATILE_ROOT.exists()
        or R9_VOLATILE_ROOT.is_symlink()
    ):
        raise R10FailureSealError("r10 failure proof drifted")
    if proof.get("stdout") != _file_ref(
        evidence / STDOUT_NAME, description="sealed r10 sealer stdout"
    ) or proof.get("stderr") != _file_ref(
        evidence / STDERR_NAME, description="sealed r10 sealer stderr"
    ):
        raise R10FailureSealError("r10 process-output binding drifted")
    if (evidence / STDOUT_NAME).read_bytes() != b"" or (
        evidence / STDERR_NAME
    ).read_text(encoding="utf-8") != EXPECTED_STDERR:
        raise R10FailureSealError("r10 process output drifted")
    _verify_archive(
        evidence / OBSERVED_EVIDENCE_NAME,
        proof["observed_incomplete_evidence"],
    )
    _verify_archive(
        evidence / REPRODUCED_EVIDENCE_NAME,
        proof["reproduced_incomplete_evidence"],
    )
    for name, field in (
        (OBSERVED_VOLATILE_NAME, "observed_volatile_probe"),
        (REPRODUCED_VOLATILE_NAME, "reproduced_volatile_probe"),
        (DIAGNOSTIC_NAME, "operator_diagnostic"),
    ):
        record = proof.get(field)
        if not isinstance(record, dict):
            raise R10FailureSealError(f"r10 proof field is invalid: {field}")
        _verify_archive(evidence / name, record)
    if (
        proof["observed_incomplete_evidence"].get("inventory")
        != intent["observed_incomplete_evidence"].get("inventory")
        or proof["observed_volatile_probe"].get("inventory")
        != intent["observed_volatile_probe"].get("inventory")
        or proof["operator_diagnostic"].get("inventory")
        != intent["operator_diagnostic"].get("inventory")
    ):
        raise R10FailureSealError("r10 observed/preimage binding drifted")
    if (
        proof["operator_diagnostic"].get("authoritative_failure_evidence")
        is not False
    ):
        raise R10FailureSealError("operator diagnostic authority drifted")


def verify_failure_seal(
    evidence_root: str | Path,
    *,
    recovery_root: str | Path,
    scheduler_user: str | None = None,
) -> dict[str, Any]:
    evidence = _safe_directory(
        evidence_root, description="r10 wrapper-failure evidence"
    )
    recovery = _safe_directory(
        recovery_root, description="schema-5 recovery root"
    )
    marker, marker_raw = _read_json(
        evidence / MARKER_NAME, description="r10 failure marker"
    )
    if (
        set(marker)
        != {
            "schema_version",
            "protocol",
            "classification",
            "intent",
            "proof",
            "marker_id",
        }
        or marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("protocol") != PROTOCOL
        or marker.get("classification") != CLASSIFICATION
        or marker.get("marker_id") != _identity(marker, "marker_id")
    ):
        raise R10FailureSealError("r10 failure marker drifted")
    intent, intent_raw = _read_json(
        evidence / INTENT_NAME, description="r10 failure intent"
    )
    sealed_user = intent.get("scheduler", {}).get("scheduler_user")
    if scheduler_user is None:
        scheduler_user = sealed_user
    if not isinstance(scheduler_user, str) or scheduler_user != sealed_user:
        raise R10FailureSealError("r10 scheduler-user binding drifted")
    _validate_intent(intent, recovery, evidence, scheduler_user)
    if marker.get("intent") != {
        "path": str(evidence / INTENT_NAME),
        "sha256": hashlib.sha256(intent_raw).hexdigest(),
        "size": len(intent_raw),
        "intent_id": intent["intent_id"],
    }:
        raise R10FailureSealError("r10 marker intent binding drifted")
    proof = marker.get("proof")
    if not isinstance(proof, dict):
        raise R10FailureSealError("r10 failure proof is malformed")
    _verify_proof(evidence, recovery, intent, proof)
    _require_recursively_read_only(evidence)
    return {
        "passed": True,
        "root": str(evidence),
        "protocol": PROTOCOL,
        "release_tag": R10_TAG,
        "release_git_commit": R10_COMMIT,
        "chain_namespace": R10_NAMESPACE,
        "marker": str(evidence / MARKER_NAME),
        "marker_sha256": hashlib.sha256(marker_raw).hexdigest(),
        "marker_size": len(marker_raw),
        "marker_id": marker["marker_id"],
        "classification": CLASSIFICATION,
        "retry_in_place": False,
        "requires_superseding_release": True,
        "pre_scheduler_submission": True,
        "known_scheduler_job_ids": [],
    }


def seal_failure(
    *,
    evidence_root: str | Path,
    recovery_root: str | Path,
    scheduler_user: str,
    apply: bool,
) -> dict[str, Any]:
    recovery = _safe_directory(
        recovery_root, description="schema-5 recovery root"
    )
    evidence = Path(evidence_root).expanduser().absolute()
    expected = recovery / R10_FAILURE_RELATIVE
    if evidence != expected:
        raise R10FailureSealError(f"r10 failure evidence root must be {expected}")
    marker = evidence / MARKER_NAME
    if marker.exists() or marker.is_symlink():
        return {
            "action": "already_sealed",
            **verify_failure_seal(
                evidence,
                recovery_root=recovery,
                scheduler_user=scheduler_user,
            ),
        }
    observed_evidence = _validate_incomplete_evidence(
        recovery, recovery / R9_EVIDENCE_RELATIVE, scheduler_user
    )
    observed_volatile = _validate_volatile_tree(
        recovery, R9_VOLATILE_ROOT
    )
    diagnostic = _validate_operator_diagnostic(OPERATOR_DIAGNOSTIC_ROOT)
    intent = _intent_payload(
        recovery,
        evidence,
        scheduler_user,
        observed_evidence,
        observed_volatile,
        diagnostic,
    )
    if not apply:
        return {
            "action": "would_seal",
            "classification": CLASSIFICATION,
            "observed_incomplete_evidence": observed_evidence,
            "observed_volatile_probe": observed_volatile,
            "operator_diagnostic": diagnostic,
            "scientific_state": intent["scientific_state"],
            "scheduler": intent["scheduler"],
        }
    if evidence.exists() or evidence.is_symlink():
        raise R10FailureSealError(
            "incomplete r10 failure evidence root requires quarantine"
        )
    evidence.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    evidence.mkdir(mode=0o700)
    intent_path = evidence / INTENT_NAME
    intent_raw = _canonical_bytes(intent)
    _publish(intent_path, intent_raw)
    proof = _execute_proof(evidence, recovery, intent, scheduler_user)
    marker_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "classification": CLASSIFICATION,
        "intent": {
            "path": str(intent_path),
            "sha256": hashlib.sha256(intent_raw).hexdigest(),
            "size": len(intent_raw),
            "intent_id": intent["intent_id"],
        },
        "proof": proof,
    }
    marker_payload["marker_id"] = _identity(marker_payload, "marker_id")
    _publish(marker, _canonical_bytes(marker_payload))
    _remove_write_bits(evidence)
    _fsync_directory(evidence.parent)
    return {
        "action": "sealed",
        **verify_failure_seal(
            evidence,
            recovery_root=recovery,
            scheduler_user=scheduler_user,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="subcommand", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--evidence-root", type=Path, required=True)
    seal.add_argument("--recovery-root", type=Path, required=True)
    seal.add_argument("--scheduler-user", required=True)
    seal.add_argument("--apply", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--evidence-root", type=Path, required=True)
    verify.add_argument("--recovery-root", type=Path, required=True)
    verify.add_argument("--scheduler-user")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.subcommand == "seal":
            report = seal_failure(
                evidence_root=args.evidence_root,
                recovery_root=args.recovery_root,
                scheduler_user=args.scheduler_user,
                apply=args.apply,
            )
        else:
            report = verify_failure_seal(
                args.evidence_root,
                recovery_root=args.recovery_root,
                scheduler_user=args.scheduler_user,
            )
    except (
        R10FailureSealError,
        *MarkerLastEvidenceError,
        OSError,
        UnicodeError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
