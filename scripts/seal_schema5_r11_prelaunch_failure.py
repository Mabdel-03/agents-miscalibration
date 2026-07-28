#!/usr/bin/env python3
"""Seal r11's deterministic equivalent-diagnostic delimiter defect for r12."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping


SCHEMA_VERSION = 1
PROTOCOL = "schema5-v1.2-r11-equivalent-diagnostic-delimiter-failure-v1"
CLASSIFICATION = (
    "deterministic_prelaunch_equivalent_diagnostic_trailing_blank_rejection"
)
R11_TAG = "sweep-recovery-schema5-v1.2-r11"
R11_NAMESPACE = "schema5-v1.2-r11"
R11_COMMIT = "6a4b100b3c2c866fe0154007fce42f90c2763864"
R11_TAG_OBJECT = "f9d657147ff43d2abc16389e54f8323a012b6c68"
R11_DURABLE_MARKER = "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R11_COMPLETE.json"
R11_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r11"
R9_EVIDENCE_RELATIVE = Path(
    "prelaunch_failures/schema5-v1.2-r9-offline-diagnostic"
)
R9_INTENT_NAME = "PRELAUNCH_OFFLINE_DIAGNOSTIC_FAILURE_INTENT.json"
R9_MARKER_NAME = "PRELAUNCH_OFFLINE_DIAGNOSTIC_FAILURE_SEALED.json"
R9_TRANSCRIPT_NAME = "r9_recorder_reproduction.json"
R9_ARCHIVE_NAME = "original_probe_tree"
R9_REPRODUCTION_NAME = "equivalent_offline_reproduction"
R9_VOLATILE_ROOT = Path("/tmp/schema5-r3-prelaunch-mabdel03-r9")
OPERATOR_DIAGNOSTIC_ROOT = Path(
    "/tmp/schema5-r9-equivalent-diagnose-r11"
)
R11_FAILURE_RELATIVE = Path(
    "prelaunch_failures/schema5-v1.2-r11-r9-sealer"
)
MARKER_NAME = "PRELAUNCH_EQUIVALENT_DIAGNOSTIC_FAILURE_SEALED.json"
INTENT_NAME = "PRELAUNCH_EQUIVALENT_DIAGNOSTIC_FAILURE_INTENT.json"
OBSERVED_EVIDENCE_NAME = "observed_incomplete_r9_seal"
OBSERVED_VOLATILE_NAME = "observed_volatile_probe"
DIAGNOSTIC_NAME = "operator_diagnostic_non_authoritative"
REPRODUCED_EVIDENCE_NAME = "reproduced_incomplete_r9_seal"
REPRODUCED_VOLATILE_NAME = "reproduced_volatile_probe"
STDOUT_NAME = "r11-sealer.stdout"
STDERR_NAME = "r11-sealer.stderr"
EXPECTED_STDERR = "ERROR: equivalent r9 OfflineError stream drifted\n"


class R11FailureSealError(RuntimeError):
    """The r11 deterministic prelaunch failure cannot be sealed exactly."""


def _load_base():
    path = Path(__file__).resolve().with_name(
        "seal_schema5_r10_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r11_failure_base", path
    )
    if spec is None or spec.loader is None:
        raise R11FailureSealError("cannot load marker-last failure helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base()
MarkerLastEvidenceError = (
    R11FailureSealError,
    _base.R10FailureSealError,
    *_base.MarkerLastEvidenceError,
)
_canonical_bytes = _base._canonical_bytes
_identity = _base._identity
_safe_directory = _base._safe_directory
_read_json = _base._read_json
_file_ref = _base._file_ref
_publish = _base._publish
_fsync_directory = _base._fsync_directory
_git = _base._git
_portable_inventory = _base._portable_inventory
_remove_write_bits = _base._remove_write_bits
_move_directory = _base._move_directory
_archive_volatile = _base._archive_volatile
_require_recursively_read_only = _base._require_recursively_read_only


def _release_binding(recovery: Path) -> dict[str, Any]:
    checkout = _safe_directory(
        recovery / R11_CHECKOUT, description="r11 checkout"
    )
    tag_ref = f"refs/tags/{R11_TAG}"
    if (
        _git(checkout, "cat-file", "-t", tag_ref) != "tag"
        or _git(checkout, "rev-parse", tag_ref) != R11_TAG_OBJECT
        or _git(checkout, "rev-parse", f"{tag_ref}^{{commit}}")
        != R11_COMMIT
        or _git(checkout, "rev-parse", "HEAD") != R11_COMMIT
        or _git(
            checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        or (checkout / ".git/objects/info/alternates").exists()
    ):
        raise R11FailureSealError("r11 checkout identity drifted")
    marker_path = recovery / R11_DURABLE_MARKER
    marker, marker_raw = _read_json(
        marker_path, description="r11 durable release marker"
    )
    bundle = Path(str(marker.get("bundle_path", "")))
    if (
        marker.get("passed") is not True
        or marker.get("release_tag") != R11_TAG
        or marker.get("chain_namespace") != R11_NAMESPACE
        or marker.get("release_git_commit") != R11_COMMIT
        or marker.get("release_tag_object") != R11_TAG_OBJECT
        or marker.get("remote_commit") != R11_COMMIT
        or marker.get("remote_peeled_commit") != R11_COMMIT
        or marker.get("remote_tag_object") != R11_TAG_OBJECT
        or not bundle.is_file()
        or bundle.is_symlink()
        or _base._sha256_file(bundle) != marker.get("bundle_sha256")
    ):
        raise R11FailureSealError("r11 durable release identity drifted")
    return {
        "checkout": str(checkout),
        "release_tag": R11_TAG,
        "release_git_commit": R11_COMMIT,
        "release_tag_object": R11_TAG_OBJECT,
        "failed_sealer": _file_ref(
            checkout / "scripts/seal_schema5_r9_prelaunch_failure.py",
            description="immutable r11 failed sealer",
        ),
        "durable_marker": {
            "path": str(marker_path),
            "sha256": hashlib.sha256(marker_raw).hexdigest(),
            "size": len(marker_raw),
            "marker_id": marker.get("marker_id"),
        },
        "durable_bundle": _file_ref(
            bundle, description="r11 durable Git bundle"
        ),
    }


def _load_r11_sealer(recovery: Path):
    path = (
        recovery
        / R11_CHECKOUT
        / "scripts/seal_schema5_r9_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r11_failed_r9_sealer", path
    )
    if spec is None or spec.loader is None:
        raise R11FailureSealError("cannot load immutable r11 failed sealer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise R11FailureSealError("r11 failed-sealer source drifted")
    return module


def _validate_incomplete_evidence(
    recovery: Path, root: Path, scheduler_user: str
) -> dict[str, Any]:
    root = _safe_directory(root, description="incomplete r11 r9 seal")
    expected = {
        R9_INTENT_NAME,
        R9_TRANSCRIPT_NAME,
        R9_ARCHIVE_NAME,
        R9_REPRODUCTION_NAME,
    }
    if {path.name for path in root.iterdir()} != expected:
        raise R11FailureSealError("r11 incomplete r9-seal topology drifted")
    if (root / R9_MARKER_NAME).exists() or (root / R9_MARKER_NAME).is_symlink():
        raise R11FailureSealError("r11 unexpectedly published its r9 seal")
    module = _load_r11_sealer(recovery)
    intent, intent_raw = _read_json(
        root / R9_INTENT_NAME, description="r11 incomplete r9-seal intent"
    )
    r10_root = recovery / _base.R10_FAILURE_RELATIVE
    try:
        r10_binding = _base.verify_failure_seal(
            r10_root,
            recovery_root=recovery,
            scheduler_user=scheduler_user,
        )
        if intent.get("sealed_r10_wrapper_failure") != r10_binding:
            raise R11FailureSealError(
                "r11 intent's r10 failure binding drifted"
            )
        saved_r10_binding = module._r10_failure_binding
        module._r10_failure_binding = (
            lambda _recovery, _user: json.loads(
                json.dumps(r10_binding, sort_keys=True)
            )
        )
        module._validate_intent(
            intent,
            recovery,
            recovery / R9_EVIDENCE_RELATIVE,
            scheduler_user,
        )
    except Exception as exc:
        raise R11FailureSealError(
            f"r11 incomplete r9-seal intent is invalid: {exc}"
        ) from exc
    finally:
        if "saved_r10_binding" in locals():
            module._r10_failure_binding = saved_r10_binding
    transcript, transcript_raw = _read_json(
        root / R9_TRANSCRIPT_NAME,
        description="r11 exact-recorder transcript",
    )
    if (
        transcript.get("protocol")
        != f"{module.PROTOCOL}-exact-r9-recorder-reproduction"
        or transcript.get("source_release") != module.R9_TAG
        or transcript.get("volatile_root_recreated_from_empty") is not True
        or transcript.get("transcript_id")
        != module._identity(transcript, "transcript_id")
        or not isinstance(transcript.get("records"), list)
        or len(transcript["records"]) != 5
        or transcript["records"][-1].get("stderr")
        != module.EXPECTED_R9_CLI_STDERR
    ):
        raise R11FailureSealError("r11 exact-recorder transcript drifted")
    archive_inventory = _portable_inventory(root / R9_ARCHIVE_NAME)
    if archive_inventory != transcript.get("resulting_probe_state", {}).get(
        "inventory"
    ):
        raise R11FailureSealError("r11 reproduced probe archive drifted")
    reproduction = _safe_directory(
        root / R9_REPRODUCTION_NAME,
        description="r11 incomplete equivalent reproduction",
    )
    if not (reproduction / "empty-conda-pkgs").is_dir() or not (
        reproduction / "offline-clone-destination"
    ).is_dir():
        raise R11FailureSealError("r11 equivalent reproduction topology drifted")
    return {
        "path": str(root),
        "intent": {
            "path": str(root / R9_INTENT_NAME),
            "sha256": hashlib.sha256(intent_raw).hexdigest(),
            "size": len(intent_raw),
            "intent_id": intent.get("intent_id"),
        },
        "transcript": {
            "path": str(root / R9_TRANSCRIPT_NAME),
            "sha256": hashlib.sha256(transcript_raw).hexdigest(),
            "size": len(transcript_raw),
            "transcript_id": transcript.get("transcript_id"),
        },
        "inventory": _portable_inventory(root),
        "completion_marker_published": False,
        "conda_output_files_published": False,
    }


def _validate_volatile_tree(recovery: Path, root: Path) -> dict[str, Any]:
    root = _safe_directory(root, description="r11 failed volatile probe")
    module = _load_r11_sealer(recovery)
    saved = module.R9_PROBE_ROOT
    try:
        module.R9_PROBE_ROOT = root
        report = module._validate_original_probe_tree(recovery)
    except Exception as exc:
        raise R11FailureSealError(
            f"r11 failed volatile probe is invalid: {exc}"
        ) from exc
    finally:
        module.R9_PROBE_ROOT = saved
    return report


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
        raise R11FailureSealError(f"squeue failed: {process.stderr.strip()}")
    matching = [
        row
        for row in process.stdout.splitlines()
        if "s5v12r11" in row
        or "schema5-v1.2-r11" in row
        or "sweep-recovery-schema5-v1.2-r11" in row
    ]
    if matching:
        raise R11FailureSealError(f"r11 scheduler jobs remain live: {matching}")
    return {"scheduler_user": user, "matching_r11_jobs": []}


def _scientific_state(recovery: Path) -> dict[str, Any]:
    results = recovery.parent.parent
    paths = [
        results / ".dispatcher-schema5-v1",
        results / "server_pools/schema5-v1",
        results / "full_sweep_schema5_v1",
        results / "full_sweep_agent_counts_schema5_v1",
        results / "full_sweep_agent_count_7_schema5_v1",
        recovery / "slurm_canaries/schema5-v1.2-r11",
        recovery / "materialization_pilots/schema5-v1.2-r11",
        recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R11.json",
    ]
    present = [
        str(path) for path in paths if path.exists() or path.is_symlink()
    ]
    if present:
        raise R11FailureSealError(
            f"r11 delimiter failure is not zero-result evidence: {present}"
        )
    return {
        "captured_before_r12_production_outputs": True,
        "result_mutation_count": 0,
        "scheduler_job_count": 0,
        "required_absent_paths_at_seal": [str(path) for path in paths],
    }


def _reproduction_argv(recovery: Path, scheduler_user: str) -> list[str]:
    return [
        sys.executable,
        "-I",
        str(
            recovery
            / R11_CHECKOUT
            / "scripts/seal_schema5_r9_prelaunch_failure.py"
        ),
        "seal",
        "--evidence-root",
        str(recovery / R9_EVIDENCE_RELATIVE),
        "--recovery-root",
        str(recovery),
        "--scheduler-user",
        scheduler_user,
        "--apply",
    ]


def _temporary_quarantine_paths() -> tuple[Path, ...]:
    return (
        Path(f"{R9_VOLATILE_ROOT}-r11-failed-observed"),
        Path(f"{R9_VOLATILE_ROOT}-r11-failed-reproduced"),
        Path(f"{OPERATOR_DIAGNOSTIC_ROOT}-observed"),
    )


def _validate_quarantines_absent() -> list[str]:
    paths = _temporary_quarantine_paths()
    present = [
        str(path) for path in paths if path.exists() or path.is_symlink()
    ]
    if present:
        raise R11FailureSealError(
            f"r11 temporary quarantine destination exists: {present}"
        )
    return [str(path) for path in paths]


def _intent_payload(
    recovery: Path,
    evidence: Path,
    user: str,
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
        "temporary_quarantine_destinations": _validate_quarantines_absent(),
        "reproduction": {
            "argv": _reproduction_argv(recovery, user),
            "cwd": str(recovery / R11_CHECKOUT),
            "expected_returncode": 2,
            "expected_stdout": "",
            "expected_stderr": EXPECTED_STDERR,
        },
        "scientific_state": _scientific_state(recovery),
        "scheduler": _scheduler_quiescence(user),
        "evidence_root": str(evidence),
    }
    payload["intent_id"] = _identity(payload, "intent_id")
    return payload


def _validate_intent(
    intent: Mapping[str, Any],
    recovery: Path,
    evidence: Path,
    user: str,
) -> None:
    if (
        intent.get("schema_version") != SCHEMA_VERSION
        or intent.get("protocol") != f"{PROTOCOL}-intent"
        or intent.get("classification") != CLASSIFICATION
        or intent.get("intent_id") != _identity(intent, "intent_id")
        or intent.get("source_release") != _release_binding(recovery)
        or intent.get("temporary_quarantine_destinations")
        != [str(path) for path in _temporary_quarantine_paths()]
        or intent.get("reproduction")
        != {
            "argv": _reproduction_argv(recovery, user),
            "cwd": str(recovery / R11_CHECKOUT),
            "expected_returncode": 2,
            "expected_stdout": "",
            "expected_stderr": EXPECTED_STDERR,
        }
        or intent.get("scheduler") != _scheduler_quiescence(user)
        or intent.get("evidence_root") != str(evidence)
    ):
        raise R11FailureSealError("r11 failure-seal intent drifted")
    state = intent.get("scientific_state")
    if (
        not isinstance(state, dict)
        or state.get("captured_before_r12_production_outputs") is not True
        or state.get("result_mutation_count") != 0
        or state.get("scheduler_job_count") != 0
    ):
        raise R11FailureSealError("r11 scientific-state evidence drifted")


def _execute_proof(
    evidence: Path,
    recovery: Path,
    intent: Mapping[str, Any],
    user: str,
) -> dict[str, Any]:
    _validate_quarantines_absent()
    observed_evidence = recovery / R9_EVIDENCE_RELATIVE
    archived_observed = evidence / OBSERVED_EVIDENCE_NAME
    _move_directory(observed_evidence, archived_observed)
    observed_volatile = _archive_volatile(
        R9_VOLATILE_ROOT,
        Path(f"{R9_VOLATILE_ROOT}-r11-failed-observed"),
        evidence / OBSERVED_VOLATILE_NAME,
    )
    diagnostic = _archive_volatile(
        OPERATOR_DIAGNOSTIC_ROOT,
        Path(f"{OPERATOR_DIAGNOSTIC_ROOT}-observed"),
        evidence / DIAGNOSTIC_NAME,
    )
    if (
        _validate_incomplete_evidence(
            recovery, archived_observed, user
        )["inventory"]
        != intent["observed_incomplete_evidence"]["inventory"]
        or _validate_volatile_tree(
            recovery, evidence / OBSERVED_VOLATILE_NAME
        )["inventory"]
        != intent["observed_volatile_probe"]["inventory"]
        or _portable_inventory(evidence / DIAGNOSTIC_NAME)
        != intent["operator_diagnostic"]["inventory"]
    ):
        raise R11FailureSealError("observed r11 failure archive drifted")
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
        raise R11FailureSealError("exact r11 delimiter failure drifted")
    stdout_path = evidence / STDOUT_NAME
    stderr_path = evidence / STDERR_NAME
    _publish(stdout_path, process.stdout)
    _publish(stderr_path, process.stderr)
    reproduced_evidence = recovery / R9_EVIDENCE_RELATIVE
    reproduced_state = _validate_incomplete_evidence(
        recovery, reproduced_evidence, user
    )
    reproduced_volatile_state = _validate_volatile_tree(
        recovery, R9_VOLATILE_ROOT
    )
    archived_reproduced = evidence / REPRODUCED_EVIDENCE_NAME
    _move_directory(reproduced_evidence, archived_reproduced)
    reproduced_volatile = _archive_volatile(
        R9_VOLATILE_ROOT,
        Path(f"{R9_VOLATILE_ROOT}-r11-failed-reproduced"),
        evidence / REPRODUCED_VOLATILE_NAME,
    )
    if (
        _portable_inventory(archived_reproduced)
        != reproduced_state["inventory"]
        or _portable_inventory(evidence / REPRODUCED_VOLATILE_NAME)
        != reproduced_volatile_state["inventory"]
    ):
        raise R11FailureSealError("reproduced r11 failure archive drifted")
    for root in (archived_observed, archived_reproduced):
        _remove_write_bits(root)
        _require_recursively_read_only(root)
    return {
        "returncode": process.returncode,
        "stdout": _file_ref(stdout_path, description="r11 sealer stdout"),
        "stderr": _file_ref(stderr_path, description="r11 sealer stderr"),
        "exact_error_matched": True,
        "failed_condition": (
            "canonical_offline_blocks_had_one_trailing_blank_delimiter"
        ),
        "observed_incomplete_evidence": {
            "path": str(archived_observed),
            "inventory": _portable_inventory(archived_observed),
            "recursively_read_only": True,
        },
        "observed_volatile_probe": observed_volatile,
        "operator_diagnostic": diagnostic
        | {"authoritative_failure_evidence": False},
        "reproduced_incomplete_evidence": {
            "path": str(archived_reproduced),
            "inventory": _portable_inventory(archived_reproduced),
            "recursively_read_only": True,
        },
        "reproduced_volatile_probe": reproduced_volatile,
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
        raise R11FailureSealError(f"r11 archive binding drifted: {root}")
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
        != "canonical_offline_blocks_had_one_trailing_blank_delimiter"
        or proof.get("canonical_r9_evidence_path_absent") is not True
        or proof.get("canonical_r9_volatile_path_absent") is not True
        or proof.get("known_scheduler_job_ids") != []
        or proof.get("result_mutation_count") != 0
    ):
        raise R11FailureSealError("r11 failure proof drifted")
    if proof.get("stdout") != _file_ref(
        evidence / STDOUT_NAME, description="sealed r11 sealer stdout"
    ) or proof.get("stderr") != _file_ref(
        evidence / STDERR_NAME, description="sealed r11 sealer stderr"
    ):
        raise R11FailureSealError("r11 process-output binding drifted")
    if (evidence / STDOUT_NAME).read_bytes() != b"" or (
        evidence / STDERR_NAME
    ).read_text(encoding="utf-8") != EXPECTED_STDERR:
        raise R11FailureSealError("r11 process output drifted")
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
            raise R11FailureSealError(f"r11 proof field is invalid: {field}")
        _verify_archive(evidence / name, record)
    if (
        proof["operator_diagnostic"].get("authoritative_failure_evidence")
        is not False
        or proof["observed_incomplete_evidence"].get("inventory")
        != intent["observed_incomplete_evidence"].get("inventory")
        or proof["observed_volatile_probe"].get("inventory")
        != intent["observed_volatile_probe"].get("inventory")
        or proof["operator_diagnostic"].get("inventory")
        != intent["operator_diagnostic"].get("inventory")
    ):
        raise R11FailureSealError("r11 observed/preimage binding drifted")


def verify_failure_seal(
    evidence_root: str | Path,
    *,
    recovery_root: str | Path,
    scheduler_user: str | None = None,
) -> dict[str, Any]:
    evidence = _safe_directory(
        evidence_root, description="r11 delimiter-failure evidence"
    )
    recovery = _safe_directory(
        recovery_root, description="schema-5 recovery root"
    )
    marker, marker_raw = _read_json(
        evidence / MARKER_NAME, description="r11 failure marker"
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
        raise R11FailureSealError("r11 failure marker drifted")
    intent, intent_raw = _read_json(
        evidence / INTENT_NAME, description="r11 failure intent"
    )
    sealed_user = intent.get("scheduler", {}).get("scheduler_user")
    if scheduler_user is None:
        scheduler_user = sealed_user
    if not isinstance(scheduler_user, str) or scheduler_user != sealed_user:
        raise R11FailureSealError("r11 scheduler-user binding drifted")
    _validate_intent(intent, recovery, evidence, scheduler_user)
    if marker.get("intent") != {
        "path": str(evidence / INTENT_NAME),
        "sha256": hashlib.sha256(intent_raw).hexdigest(),
        "size": len(intent_raw),
        "intent_id": intent["intent_id"],
    }:
        raise R11FailureSealError("r11 marker intent binding drifted")
    proof = marker.get("proof")
    if not isinstance(proof, dict):
        raise R11FailureSealError("r11 failure proof is malformed")
    _verify_proof(evidence, recovery, intent, proof)
    _require_recursively_read_only(evidence)
    return {
        "passed": True,
        "root": str(evidence),
        "protocol": PROTOCOL,
        "release_tag": R11_TAG,
        "release_git_commit": R11_COMMIT,
        "chain_namespace": R11_NAMESPACE,
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
    expected = recovery / R11_FAILURE_RELATIVE
    if evidence != expected:
        raise R11FailureSealError(f"r11 failure evidence root must be {expected}")
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
    diagnostic = _base._validate_operator_diagnostic(
        OPERATOR_DIAGNOSTIC_ROOT
    )
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
        raise R11FailureSealError(
            "incomplete r11 failure evidence root requires quarantine"
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
