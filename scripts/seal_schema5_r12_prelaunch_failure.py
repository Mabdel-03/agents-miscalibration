#!/usr/bin/env python3
"""Seal r12's deterministic r9 transcript-reference verifier defect for r13."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping


SCHEMA_VERSION = 1
PROTOCOL = "schema5-v1.2-r12-r9-verifier-file-reference-failure-v1"
CLASSIFICATION = (
    "deterministic_prelaunch_r9_transcript_file_reference_verifier_mismatch"
)
R12_TAG = "sweep-recovery-schema5-v1.2-r12"
R12_NAMESPACE = "schema5-v1.2-r12"
R12_COMMIT = "a91616c412b9e36a87d8232511a1ca025f8f79de"
R12_TAG_OBJECT = "0615b17e4027d344fed53da8259345a2383bd33f"
R12_DURABLE_MARKER = "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R12_COMPLETE.json"
R12_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r12"
R9_EVIDENCE_RELATIVE = Path(
    "prelaunch_failures/schema5-v1.2-r9-offline-diagnostic"
)
R9_MARKER_NAME = "PRELAUNCH_OFFLINE_DIAGNOSTIC_FAILURE_SEALED.json"
R9_INTENT_NAME = "PRELAUNCH_OFFLINE_DIAGNOSTIC_FAILURE_INTENT.json"
R9_TRANSCRIPT_NAME = "r9_recorder_reproduction.json"
R9_ARCHIVE_NAME = "original_probe_tree"
R9_REPRODUCTION_NAME = "equivalent_offline_reproduction"
R9_STDOUT_NAME = "conda.stdout"
R9_STDERR_NAME = "conda.stderr"
R12_FAILURE_RELATIVE = Path(
    "prelaunch_failures/schema5-v1.2-r12-r9-verifier"
)
MARKER_NAME = "PRELAUNCH_R9_VERIFIER_FAILURE_SEALED.json"
INTENT_NAME = "PRELAUNCH_R9_VERIFIER_FAILURE_INTENT.json"
ARCHIVE_NAME = "observed_invalid_r9_transaction"
STDOUT_NAME = "r12-verifier.stdout"
STDERR_NAME = "r12-verifier.stderr"
EXPECTED_STDERR = "ERROR: exact r9 recorder reproduction binding drifted\n"


class R12FailureSealError(RuntimeError):
    """The immutable r12 verifier failure cannot be sealed exactly."""


def _load_helpers():
    path = Path(__file__).resolve().with_name(
        "seal_schema5_r10_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r12_failure_helpers", path
    )
    if spec is None or spec.loader is None:
        raise R12FailureSealError("cannot load marker-last failure helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_helpers()
MarkerLastEvidenceError = (
    R12FailureSealError,
    _helpers.R10FailureSealError,
    *_helpers.MarkerLastEvidenceError,
)
_canonical_bytes = _helpers._canonical_bytes
_identity = _helpers._identity
_safe_directory = _helpers._safe_directory
_read_json = _helpers._read_json
_file_ref = _helpers._file_ref
_publish = _helpers._publish
_fsync_directory = _helpers._fsync_directory
_git = _helpers._git
_portable_inventory = _helpers._portable_inventory
_remove_write_bits = _helpers._remove_write_bits
_require_recursively_read_only = _helpers._require_recursively_read_only


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise R12FailureSealError(f"cannot load verifier module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise R12FailureSealError(f"verifier source drifted: {path}")
    return module


def _r12_module(recovery: Path):
    return _load_module(
        recovery
        / R12_CHECKOUT
        / "scripts/seal_schema5_r9_prelaunch_failure.py",
        "_schema5_immutable_r12_r9_verifier",
    )


def _corrected_module():
    return _load_module(
        Path(__file__).resolve().with_name(
            "seal_schema5_r9_prelaunch_failure.py"
        ),
        "_schema5_corrected_r13_r9_verifier",
    )


def _release_binding(recovery: Path) -> dict[str, Any]:
    checkout = _safe_directory(
        recovery / R12_CHECKOUT, description="r12 checkout"
    )
    tag_ref = f"refs/tags/{R12_TAG}"
    if (
        _git(checkout, "cat-file", "-t", tag_ref) != "tag"
        or _git(checkout, "rev-parse", tag_ref) != R12_TAG_OBJECT
        or _git(checkout, "rev-parse", f"{tag_ref}^{{commit}}")
        != R12_COMMIT
        or _git(checkout, "rev-parse", "HEAD") != R12_COMMIT
        or _git(
            checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        or (checkout / ".git/objects/info/alternates").exists()
    ):
        raise R12FailureSealError("r12 checkout identity drifted")
    marker_path = recovery / R12_DURABLE_MARKER
    marker, marker_raw = _read_json(
        marker_path, description="r12 durable release marker"
    )
    bundle = Path(str(marker.get("bundle_path", "")))
    if (
        marker.get("passed") is not True
        or marker.get("release_tag") != R12_TAG
        or marker.get("chain_namespace") != R12_NAMESPACE
        or marker.get("release_git_commit") != R12_COMMIT
        or marker.get("release_tag_object") != R12_TAG_OBJECT
        or marker.get("remote_commit") != R12_COMMIT
        or marker.get("remote_peeled_commit") != R12_COMMIT
        or marker.get("remote_tag_object") != R12_TAG_OBJECT
        or not bundle.is_file()
        or bundle.is_symlink()
        or _helpers._sha256_file(bundle) != marker.get("bundle_sha256")
    ):
        raise R12FailureSealError("r12 durable release identity drifted")
    return {
        "checkout": str(checkout),
        "release_tag": R12_TAG,
        "release_git_commit": R12_COMMIT,
        "release_tag_object": R12_TAG_OBJECT,
        "failed_verifier": _file_ref(
            checkout / "scripts/seal_schema5_r9_prelaunch_failure.py",
            description="immutable r12 failed r9 verifier",
        ),
        "durable_marker": {
            "path": str(marker_path),
            "sha256": hashlib.sha256(marker_raw).hexdigest(),
            "size": len(marker_raw),
            "marker_id": marker.get("marker_id"),
        },
        "durable_bundle": _file_ref(
            bundle, description="r12 durable Git bundle"
        ),
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
        raise R12FailureSealError(f"squeue failed: {process.stderr.strip()}")
    matching = [
        row
        for row in process.stdout.splitlines()
        if "s5v12r12" in row
        or "schema5-v1.2-r12" in row
        or "sweep-recovery-schema5-v1.2-r12" in row
    ]
    if matching:
        raise R12FailureSealError(f"r12 scheduler jobs remain live: {matching}")
    return {"scheduler_user": user, "matching_r12_jobs": []}


def _scientific_state(recovery: Path) -> dict[str, Any]:
    results = recovery.parent.parent
    paths = [
        results / ".dispatcher-schema5-v1",
        results / "server_pools/schema5-v1",
        results / "full_sweep_schema5_v1",
        results / "full_sweep_agent_counts_schema5_v1",
        results / "full_sweep_agent_count_7_schema5_v1",
        recovery / "slurm_canaries/schema5-v1.2-r12",
        recovery / "materialization_pilots/schema5-v1.2-r12",
        recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R12.json",
    ]
    present = [
        str(path) for path in paths if path.exists() or path.is_symlink()
    ]
    if present:
        raise R12FailureSealError(
            f"r12 verifier failure is not zero-result evidence: {present}"
        )
    return {
        "captured_before_r13_production_outputs": True,
        "result_mutation_count": 0,
        "scheduler_job_count": 0,
        "required_absent_paths_at_seal": [str(path) for path in paths],
    }


def _verify_immutable_rejection(
    module, evidence: Path, recovery: Path, user: str
) -> None:
    try:
        module.verify_failure_seal(
            evidence,
            recovery_root=recovery,
            scheduler_user=user,
        )
    except module.R9FailureSealError as exc:
        if str(exc) != EXPECTED_STDERR.removeprefix("ERROR: ").rstrip("\n"):
            raise R12FailureSealError(
                f"immutable r12 rejection drifted: {exc}"
            ) from exc
    else:
        raise R12FailureSealError(
            "immutable r12 verifier unexpectedly accepted its r9 marker"
        )


def _validate_transcript_reference_mismatch(
    *,
    actual_reference: Mapping[str, Any],
    buggy_expected: Mapping[str, Any],
    transcript_path: Path,
    reproduced_state: Any,
    recorded_state: Any,
) -> None:
    if (
        set(actual_reference) != set(buggy_expected) | {"mode", "link_count"}
        or {
            key: value
            for key, value in actual_reference.items()
            if key not in {"mode", "link_count"}
        }
        != buggy_expected
        or actual_reference.get("mode")
        != stat.S_IMODE(transcript_path.stat().st_mode)
        or actual_reference.get("link_count")
        != transcript_path.stat().st_nlink
        or reproduced_state != recorded_state
    ):
        raise R12FailureSealError(
            "r12 transcript-reference mismatch classification drifted"
        )


def _validate_invalid_transaction(
    recovery: Path, evidence: Path, user: str
) -> dict[str, Any]:
    evidence = _safe_directory(
        evidence, description="r12 invalid-under-verifier r9 transaction"
    )
    expected = {
        R9_MARKER_NAME,
        R9_INTENT_NAME,
        R9_TRANSCRIPT_NAME,
        R9_ARCHIVE_NAME,
        R9_REPRODUCTION_NAME,
        R9_STDOUT_NAME,
        R9_STDERR_NAME,
    }
    if {path.name for path in evidence.iterdir()} != expected:
        raise R12FailureSealError("r12 r9 transaction topology drifted")
    immutable = _r12_module(recovery)
    _verify_immutable_rejection(immutable, evidence, recovery, user)
    corrected = _corrected_module()
    corrected_binding = corrected.verify_failure_seal(
        evidence,
        recovery_root=recovery,
        scheduler_user=user,
    )
    marker, marker_raw = _read_json(
        evidence / R9_MARKER_NAME, description="r12-published r9 marker"
    )
    transcript, transcript_raw = _read_json(
        evidence / R9_TRANSCRIPT_NAME,
        description="r12 exact-recorder transcript",
    )
    proof = marker.get("proof")
    if not isinstance(proof, dict):
        raise R12FailureSealError("r12 r9 marker proof is malformed")
    actual_reference = proof.get("historical_recorder_transcript")
    if not isinstance(actual_reference, dict):
        raise R12FailureSealError("r12 transcript reference is malformed")
    buggy_expected = {
        "path": str(evidence / R9_TRANSCRIPT_NAME),
        "sha256": hashlib.sha256(transcript_raw).hexdigest(),
        "size": len(transcript_raw),
        "transcript_id": transcript.get("transcript_id"),
    }
    _validate_transcript_reference_mismatch(
        actual_reference=actual_reference,
        buggy_expected=buggy_expected,
        transcript_path=evidence / R9_TRANSCRIPT_NAME,
        reproduced_state=proof.get("reproduced_probe_state"),
        recorded_state=transcript.get("resulting_probe_state"),
    )
    return {
        "path": str(evidence),
        "inventory": _portable_inventory(evidence),
        "marker": {
            "path": str(evidence / R9_MARKER_NAME),
            "sha256": hashlib.sha256(marker_raw).hexdigest(),
            "size": len(marker_raw),
            "marker_id": marker.get("marker_id"),
        },
        "transcript": {
            "path": str(evidence / R9_TRANSCRIPT_NAME),
            "sha256": hashlib.sha256(transcript_raw).hexdigest(),
            "size": len(transcript_raw),
            "transcript_id": transcript.get("transcript_id"),
        },
        "complete_file_reference_keys": sorted(actual_reference),
        "buggy_expected_reference_keys": sorted(buggy_expected),
        "reproduced_state_matches_transcript": True,
        "immutable_r12_rejected": True,
        "corrected_r13_verifier": corrected_binding,
    }


def _reproduction_argv(
    recovery: Path, evidence: Path, user: str
) -> list[str]:
    return [
        sys.executable,
        "-I",
        str(
            recovery
            / R12_CHECKOUT
            / "scripts/seal_schema5_r9_prelaunch_failure.py"
        ),
        "verify",
        "--evidence-root",
        str(evidence),
        "--recovery-root",
        str(recovery),
        "--scheduler-user",
        user,
    ]


def _intent_payload(
    recovery: Path,
    evidence: Path,
    user: str,
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": f"{PROTOCOL}-intent",
        "classification": CLASSIFICATION,
        "source_release": _release_binding(recovery),
        "observed_invalid_r9_transaction": observed,
        "reproduction": {
            "argv": _reproduction_argv(
                recovery, recovery / R9_EVIDENCE_RELATIVE, user
            ),
            "cwd": str(recovery / R12_CHECKOUT),
            "expected_returncode": 2,
            "expected_stdout": "",
            "expected_stderr": EXPECTED_STDERR,
            "read_only_replay": True,
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
        or intent.get("reproduction")
        != {
            "argv": _reproduction_argv(
                recovery, recovery / R9_EVIDENCE_RELATIVE, user
            ),
            "cwd": str(recovery / R12_CHECKOUT),
            "expected_returncode": 2,
            "expected_stdout": "",
            "expected_stderr": EXPECTED_STDERR,
            "read_only_replay": True,
        }
        or intent.get("evidence_root") != str(evidence)
    ):
        raise R12FailureSealError("r12 failure-seal intent drifted")
    state = intent.get("scientific_state")
    scheduler = intent.get("scheduler")
    if (
        not isinstance(state, dict)
        or state.get("captured_before_r13_production_outputs") is not True
        or state.get("result_mutation_count") != 0
        or state.get("scheduler_job_count") != 0
        or not isinstance(scheduler, dict)
        or scheduler.get("scheduler_user") != user
        or scheduler.get("matching_r12_jobs") != []
    ):
        raise R12FailureSealError("r12 sealed prelaunch state drifted")


def _execute_proof(
    evidence: Path,
    recovery: Path,
    intent: Mapping[str, Any],
    user: str,
) -> dict[str, Any]:
    canonical = recovery / R9_EVIDENCE_RELATIVE
    before = _portable_inventory(canonical)
    process = subprocess.run(
        list(intent["reproduction"]["argv"]),
        cwd=intent["reproduction"]["cwd"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=600,
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
        or _portable_inventory(canonical) != before
    ):
        raise R12FailureSealError(
            "exact read-only r12 verifier replay drifted"
        )
    stdout_path = evidence / STDOUT_NAME
    stderr_path = evidence / STDERR_NAME
    _publish(stdout_path, process.stdout)
    _publish(stderr_path, process.stderr)
    archive = evidence / ARCHIVE_NAME
    shutil.copytree(
        canonical,
        archive,
        symlinks=True,
        copy_function=shutil.copy2,
    )
    if _portable_inventory(archive) != before:
        raise R12FailureSealError("r12 invalid transaction archive drifted")
    _remove_write_bits(archive)
    _require_recursively_read_only(archive)
    observed = intent["observed_invalid_r9_transaction"]
    if observed.get("inventory") != before:
        raise R12FailureSealError("r12 observed transaction preimage drifted")
    return {
        "returncode": process.returncode,
        "stdout": _file_ref(
            stdout_path, description="r12 failed verifier stdout"
        ),
        "stderr": _file_ref(
            stderr_path, description="r12 failed verifier stderr"
        ),
        "exact_error_matched": True,
        "read_only_replay": True,
        "failed_condition": (
            "complete_file_reference_included_mode_and_link_count_"
            "while_verifier_expected_subset"
        ),
        "observed_invalid_r9_transaction": observed,
        "durable_archive": {
            "path": str(archive),
            "inventory": _portable_inventory(archive),
            "recursively_read_only": True,
        },
        "canonical_transaction_preserved": True,
        "known_scheduler_job_ids": [],
        "result_mutation_count": 0,
    }


def _verify_proof(
    evidence: Path,
    recovery: Path,
    intent: Mapping[str, Any],
    proof: Mapping[str, Any],
    user: str,
) -> None:
    expected_keys = {
        "returncode",
        "stdout",
        "stderr",
        "exact_error_matched",
        "read_only_replay",
        "failed_condition",
        "observed_invalid_r9_transaction",
        "durable_archive",
        "canonical_transaction_preserved",
        "known_scheduler_job_ids",
        "result_mutation_count",
    }
    if (
        set(proof) != expected_keys
        or proof.get("returncode") != 2
        or proof.get("exact_error_matched") is not True
        or proof.get("read_only_replay") is not True
        or proof.get("failed_condition")
        != (
            "complete_file_reference_included_mode_and_link_count_"
            "while_verifier_expected_subset"
        )
        or proof.get("canonical_transaction_preserved") is not True
        or proof.get("known_scheduler_job_ids") != []
        or proof.get("result_mutation_count") != 0
        or proof.get("observed_invalid_r9_transaction")
        != intent.get("observed_invalid_r9_transaction")
    ):
        raise R12FailureSealError("r12 failure proof drifted")
    if proof.get("stdout") != _file_ref(
        evidence / STDOUT_NAME, description="sealed r12 verifier stdout"
    ) or proof.get("stderr") != _file_ref(
        evidence / STDERR_NAME, description="sealed r12 verifier stderr"
    ):
        raise R12FailureSealError("r12 verifier output binding drifted")
    if (evidence / STDOUT_NAME).read_bytes() != b"" or (
        evidence / STDERR_NAME
    ).read_text(encoding="utf-8") != EXPECTED_STDERR:
        raise R12FailureSealError("r12 verifier output drifted")
    archive = evidence / ARCHIVE_NAME
    archive_record = proof.get("durable_archive")
    if (
        not isinstance(archive_record, dict)
        or archive_record.get("path") != str(archive)
        or archive_record.get("inventory") != _portable_inventory(archive)
        or archive_record.get("inventory")
        != intent["observed_invalid_r9_transaction"]["inventory"]
        or archive_record.get("recursively_read_only") is not True
    ):
        raise R12FailureSealError("r12 failure archive binding drifted")
    _require_recursively_read_only(archive)
    canonical = recovery / R9_EVIDENCE_RELATIVE
    current = _validate_invalid_transaction(recovery, canonical, user)
    if (
        current.get("inventory")
        != intent["observed_invalid_r9_transaction"]["inventory"]
    ):
        raise R12FailureSealError("canonical r9 transaction drifted")


def verify_failure_seal(
    evidence_root: str | Path,
    *,
    recovery_root: str | Path,
    scheduler_user: str | None = None,
) -> dict[str, Any]:
    evidence = _safe_directory(
        evidence_root, description="r12 r9-verifier failure evidence"
    )
    recovery = _safe_directory(
        recovery_root, description="schema-5 recovery root"
    )
    marker, marker_raw = _read_json(
        evidence / MARKER_NAME, description="r12 failure marker"
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
        raise R12FailureSealError("r12 failure marker drifted")
    intent, intent_raw = _read_json(
        evidence / INTENT_NAME, description="r12 failure intent"
    )
    sealed_user = intent.get("scheduler", {}).get("scheduler_user")
    if scheduler_user is None:
        scheduler_user = sealed_user
    if not isinstance(scheduler_user, str) or scheduler_user != sealed_user:
        raise R12FailureSealError("r12 scheduler-user binding drifted")
    _validate_intent(intent, recovery, evidence, scheduler_user)
    if marker.get("intent") != {
        "path": str(evidence / INTENT_NAME),
        "sha256": hashlib.sha256(intent_raw).hexdigest(),
        "size": len(intent_raw),
        "intent_id": intent["intent_id"],
    }:
        raise R12FailureSealError("r12 marker intent binding drifted")
    proof = marker.get("proof")
    if not isinstance(proof, dict):
        raise R12FailureSealError("r12 failure proof is malformed")
    _verify_proof(evidence, recovery, intent, proof, scheduler_user)
    _require_recursively_read_only(evidence)
    return {
        "passed": True,
        "root": str(evidence),
        "protocol": PROTOCOL,
        "release_tag": R12_TAG,
        "release_git_commit": R12_COMMIT,
        "chain_namespace": R12_NAMESPACE,
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
    expected = recovery / R12_FAILURE_RELATIVE
    if evidence != expected:
        raise R12FailureSealError(
            f"r12 failure evidence root must be {expected}"
        )
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
    canonical = recovery / R9_EVIDENCE_RELATIVE
    observed = _validate_invalid_transaction(
        recovery, canonical, scheduler_user
    )
    intent = _intent_payload(
        recovery, evidence, scheduler_user, observed
    )
    if not apply:
        return {
            "action": "would_seal",
            "classification": CLASSIFICATION,
            "observed_invalid_r9_transaction": observed,
            "scientific_state": intent["scientific_state"],
            "scheduler": intent["scheduler"],
        }
    if evidence.exists() or evidence.is_symlink():
        raise R12FailureSealError(
            "incomplete r12 failure evidence root requires quarantine"
        )
    evidence.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    evidence.mkdir(mode=0o700)
    intent_path = evidence / INTENT_NAME
    intent_raw = _canonical_bytes(intent)
    _publish(intent_path, intent_raw)
    proof = _execute_proof(
        evidence, recovery, intent, scheduler_user
    )
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
    marker_payload["marker_id"] = _identity(
        marker_payload, "marker_id"
    )
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
    except MarkerLastEvidenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
