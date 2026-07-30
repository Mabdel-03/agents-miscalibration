#!/usr/bin/env python3
"""Seal r13's deterministic r3 offline-diagnostic delimiter defect for r14."""

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
from typing import Any, Mapping


SCHEMA_VERSION = 1
PROTOCOL = "schema5-v1.2-r13-r3-offline-delimiter-failure-v1"
CLASSIFICATION = (
    "deterministic_prelaunch_r3_offline_diagnostic_trailing_blank_rejection"
)
R13_TAG = "sweep-recovery-schema5-v1.2-r13"
R13_NAMESPACE = "schema5-v1.2-r13"
R13_COMMIT = "bb692d908a04e44567a19753340b7947827476ab"
R13_TAG_OBJECT = "9b50e98034ca0c6f269d5cd6360c47a4864bbc1d"
R13_DURABLE_MARKER = "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R13_COMPLETE.json"
R13_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r13"
R13_TOOLCHAIN_RELATIVE = Path("toolchains/r13/conda")
R3_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r3"
R13_PROBE_ROOT = Path("/tmp/schema5-r3-prelaunch-mabdel03-r13")
REPRODUCTION_ROOT = Path(
    "/tmp/schema5-r13-r3-offline-reproduction-mabdel03-r14"
)
EVIDENCE_RELATIVE_ROOT = Path(
    "prelaunch_failures/schema5-v1.2-r13-r3-offline-delimiter"
)
MARKER_NAME = "PRELAUNCH_R3_OFFLINE_DELIMITER_FAILURE_SEALED.json"
INTENT_NAME = "PRELAUNCH_R3_OFFLINE_DELIMITER_FAILURE_INTENT.json"
OBSERVED_ARCHIVE_NAME = "observed_failed_probe_tree"
REPRODUCED_ARCHIVE_NAME = "reproduced_failed_probe_tree"
STDOUT_NAME = "conda.stdout"
STDERR_NAME = "conda.stderr"
RECORDER_STDERR_NAME = "r13-recorder.stderr"
RECORDER_STDOUT_NAME = "r13-recorder.stdout"
BROKEN_ENVELOPE = (
    "FAILURE_UNSAFE_RECORDED_BROKEN_INTERNAL_SYMLINK.source.json"
)
OFFLINE_ENVELOPE = (
    "FAILURE_OFFLINE_CLONE_UNSEEDED_RELEASE_LOCAL_CACHE.source.json"
)
EXPECTED_RECORDER_ERROR = (
    "r3 offline-cache probe lacks the canonical Conda OfflineError "
    "remote-fetch/progress context"
)


class R13FailureSealError(RuntimeError):
    """The immutable r13 prelaunch failure cannot be sealed exactly."""


def _load_helpers():
    path = Path(__file__).resolve().with_name(
        "seal_schema5_r12_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r13_failure_helpers", path
    )
    if spec is None or spec.loader is None:
        raise R13FailureSealError("cannot load marker-last failure helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_helpers()
MarkerLastEvidenceError = (
    R13FailureSealError,
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
_sha256_file = _helpers._helpers._sha256_file


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise R13FailureSealError(f"cannot load immutable module: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    if Path(module.__file__).resolve() != path.resolve():
        raise R13FailureSealError(f"module source drifted: {path}")
    return module


def _r13_evidence_module(recovery: Path):
    return _load_module(
        recovery / R13_CHECKOUT / "scripts/seal_recovery_evidence.py",
        "_schema5_immutable_r13_recovery_evidence",
    )


def _corrected_evidence_module():
    return _load_module(
        Path(__file__).resolve().with_name("seal_recovery_evidence.py"),
        "_schema5_corrected_r14_recovery_evidence",
    )


def _r13_toolchain_binding(recovery: Path) -> dict[str, Any]:
    provisioner = _load_module(
        recovery
        / R13_CHECKOUT
        / "scripts/provision_schema5_conda_toolchain.py",
        "_schema5_immutable_r13_toolchain_verifier",
    )
    try:
        binding = provisioner.verified_conda_toolchain_binding(
            recovery / R13_TOOLCHAIN_RELATIVE,
            exercise=True,
        )
    except Exception as exc:
        raise R13FailureSealError(
            f"r13 Conda toolchain verification failed: {exc}"
        ) from exc
    if (
        not isinstance(binding, dict)
        or binding.get("protocol")
        != "schema5-v1.2-r13-offline-conda-toolchain-v1"
        or binding.get("release_tag") != R13_TAG
        or binding.get("chain_namespace") != R13_NAMESPACE
        or binding.get("toolchain_root")
        != str(recovery / R13_TOOLCHAIN_RELATIVE)
        or binding.get("base_prefix")
        != str(recovery / R13_TOOLCHAIN_RELATIVE / "base")
    ):
        raise R13FailureSealError("r13 Conda toolchain binding drifted")
    return json.loads(json.dumps(binding, sort_keys=True))


def _release_binding(recovery: Path) -> dict[str, Any]:
    checkout = _safe_directory(
        recovery / R13_CHECKOUT, description="r13 checkout"
    )
    tag_ref = f"refs/tags/{R13_TAG}"
    if (
        _git(checkout, "cat-file", "-t", tag_ref) != "tag"
        or _git(checkout, "rev-parse", tag_ref) != R13_TAG_OBJECT
        or _git(checkout, "rev-parse", f"{tag_ref}^{{commit}}")
        != R13_COMMIT
        or _git(checkout, "rev-parse", "HEAD") != R13_COMMIT
        or _git(
            checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        or (checkout / ".git/objects/info/alternates").exists()
    ):
        raise R13FailureSealError("r13 checkout identity drifted")
    marker_path = recovery / R13_DURABLE_MARKER
    marker, marker_raw = _read_json(
        marker_path, description="r13 durable release marker"
    )
    bundle = Path(str(marker.get("bundle_path", "")))
    if (
        marker.get("passed") is not True
        or marker.get("release_tag") != R13_TAG
        or marker.get("chain_namespace") != R13_NAMESPACE
        or marker.get("release_git_commit") != R13_COMMIT
        or marker.get("release_tag_object") != R13_TAG_OBJECT
        or marker.get("remote_commit") != R13_COMMIT
        or marker.get("remote_peeled_commit") != R13_COMMIT
        or marker.get("remote_tag_object") != R13_TAG_OBJECT
        or not bundle.is_file()
        or bundle.is_symlink()
        or _sha256_file(bundle) != marker.get("bundle_sha256")
    ):
        raise R13FailureSealError("r13 durable release identity drifted")
    return {
        "checkout": str(checkout),
        "release_tag": R13_TAG,
        "release_git_commit": R13_COMMIT,
        "release_tag_object": R13_TAG_OBJECT,
        "failed_recorder": _file_ref(
            checkout / "scripts/seal_recovery_evidence.py",
            description="immutable r13 failed recorder",
        ),
        "durable_marker": {
            "path": str(marker_path),
            "sha256": hashlib.sha256(marker_raw).hexdigest(),
            "size": len(marker_raw),
            "marker_id": marker.get("marker_id"),
        },
        "durable_bundle": _file_ref(
            bundle, description="r13 durable Git bundle"
        ),
        "conda_toolchain": _r13_toolchain_binding(recovery),
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
        raise R13FailureSealError(f"squeue failed: {process.stderr.strip()}")
    matching = [
        row
        for row in process.stdout.splitlines()
        if "s5v12r13" in row
        or "schema5-v1.2-r13" in row
        or "sweep-recovery-schema5-v1.2-r13" in row
    ]
    if matching:
        raise R13FailureSealError(f"r13 scheduler jobs remain live: {matching}")
    return {"scheduler_user": user, "matching_r13_jobs": []}


def _scientific_state(recovery: Path) -> dict[str, Any]:
    results = recovery.parent.parent
    paths = [
        results / ".dispatcher-schema5-v1",
        results / "server_pools/schema5-v1",
        results / "full_sweep_schema5_v1",
        results / "full_sweep_agent_counts_schema5_v1",
        results / "full_sweep_agent_count_7_schema5_v1",
        recovery / "slurm_canaries/schema5-v1.2-r13",
        recovery / "materialization_pilots/schema5-v1.2-r13",
        recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R13.json",
    ]
    present = [
        str(path) for path in paths if path.exists() or path.is_symlink()
    ]
    if present:
        raise R13FailureSealError(
            f"r13 failure is not zero-result evidence: {present}"
        )
    return {
        "captured_before_r14_production_outputs": True,
        "result_mutation_count": 0,
        "scheduler_job_count": 0,
        "required_absent_paths_at_seal": [str(path) for path in paths],
    }


def _validate_observed_probe_tree(
    recovery: Path, root: Path
) -> dict[str, Any]:
    root = _safe_directory(root, description="observed r13 probe tree")
    expected = {
        BROKEN_ENVELOPE,
        f"{BROKEN_ENVELOPE}.sha256",
        "empty-conda-pkgs",
        "offline-clone-destination",
    }
    if {path.name for path in root.iterdir()} != expected:
        raise R13FailureSealError("observed r13 probe topology drifted")
    if (root / OFFLINE_ENVELOPE).exists() or (
        root / OFFLINE_ENVELOPE
    ).is_symlink():
        raise R13FailureSealError(
            "r13 unexpectedly published its offline envelope"
        )
    immutable = _r13_evidence_module(recovery)
    try:
        payload, _raw, _record = (
            immutable._validate_r3_prelaunch_failure_envelope(
                root / BROKEN_ENVELOPE
            )
        )
    except Exception as exc:
        raise R13FailureSealError(
            f"r13 broken-link envelope is invalid: {exc}"
        ) from exc
    if (
        payload.get("classification")
        != "unsafe_recorded_broken_internal_symlink"
        or payload.get("pre_scheduler_submission") is not True
        or payload.get("scheduler_job_ids") != []
    ):
        raise R13FailureSealError("r13 broken-link envelope drifted")
    cache = _safe_directory(
        root / "empty-conda-pkgs",
        description="observed r13 partial package cache",
    )
    destination = _safe_directory(
        root / "offline-clone-destination",
        description="observed r13 partial clone destination",
    )
    if (
        not any(cache.iterdir())
        or not (destination / ".condarc").is_file()
        or not (destination / "micromamba").is_file()
        or not (destination / "_conda").is_symlink()
    ):
        raise R13FailureSealError("r13 offline residue drifted")
    return {
        "path": str(root),
        "inventory": _portable_inventory(root),
        "broken_envelope": _file_ref(
            root / BROKEN_ENVELOPE,
            description="r13 broken-link envelope",
        )
        | {"failure_id": payload.get("failure_id")},
        "offline_envelope_absent": True,
        "partial_cache_entry_count": sum(1 for _ in cache.iterdir()),
        "partial_destination_entry_count": sum(
            1 for _ in destination.iterdir()
        ),
    }


def _probe_arguments(
    module, recovery: Path, root: Path
) -> dict[str, Any]:
    classification = "offline_clone_unseeded_release_local_cache"
    environment = module._r3_probe_expected_environment(
        root, classification=classification
    )
    toolchain = recovery / R13_TOOLCHAIN_RELATIVE
    command = [
        str(toolchain / "base/bin/conda"),
        "create",
        "--yes",
        "--offline",
        "--clone",
        str(toolchain / "base"),
        "--prefix",
        str(root / "offline-clone-destination"),
    ]
    return {
        "output": root / OFFLINE_ENVELOPE,
        "classification": classification,
        "command": command,
        "cwd": recovery / R3_CHECKOUT,
        "environment": list(environment.items()),
        "input_roots": [
            ("tagged-r3-release-checkout", recovery / R3_CHECKOUT),
            ("tagged-r13-release-checkout", recovery / R13_CHECKOUT),
            ("sealed-r13-conda-toolchain", toolchain),
        ],
        "write_roots": [root],
        "apply": True,
    }


def _grammar_proof(
    *,
    immutable,
    corrected,
    completed: subprocess.CompletedProcess[bytes],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        stdout = completed.stdout.decode("utf-8")
        stderr = completed.stderr.decode("utf-8")
    except UnicodeError as exc:
        raise R13FailureSealError("r13 probe output is not UTF-8") from exc
    if (
        completed.returncode != 1
        or not stderr.startswith("\n")
        or not stderr.endswith("\n\n")
        or stderr.endswith("\n\n\n")
    ):
        raise R13FailureSealError(
            "r13 trailing-delimiter reproduction drifted"
        )
    try:
        immutable._validate_r3_probe_failure_signature(
            classification="offline_clone_unseeded_release_local_cache",
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            contract=contract,
        )
    except immutable.EvidenceError as exc:
        if str(exc) != EXPECTED_RECORDER_ERROR:
            raise R13FailureSealError(
                f"immutable r13 rejection drifted: {exc}"
            ) from exc
    else:
        raise R13FailureSealError(
            "immutable r13 unexpectedly accepted the raw diagnostic"
        )
    immutable._validate_r3_probe_failure_signature(
        classification="offline_clone_unseeded_release_local_cache",
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr[:-1],
        contract=contract,
    )
    corrected._validate_r3_probe_failure_signature(
        classification="offline_clone_unseeded_release_local_cache",
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        contract=contract,
    )
    return {
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stdout_size": len(completed.stdout),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stderr_size": len(completed.stderr),
        "leading_blank_delimiter_count": 1,
        "trailing_blank_delimiter_count": 1,
        "immutable_r13_rejected_raw": True,
        "immutable_r13_accepted_after_one_final_newline_removed": True,
        "corrected_r14_accepted_raw": True,
    }


def _reproduce(
    evidence: Path, recovery: Path
) -> dict[str, Any]:
    root = REPRODUCTION_ROOT
    if root.exists() or root.is_symlink():
        raise R13FailureSealError(
            f"r13 reproduction root requires quarantine: {root}"
        )
    root.mkdir(mode=0o700)
    (root / "empty-conda-pkgs").mkdir(mode=0o700)
    immutable = _r13_evidence_module(recovery)
    corrected = _corrected_evidence_module()
    arguments = _probe_arguments(immutable, recovery, root)
    captured: list[subprocess.CompletedProcess[bytes]] = []

    def runner(command, **options):
        process = subprocess.run(command, **options)
        captured.append(process)
        return process

    try:
        immutable.record_prelaunch_attempt(
            **arguments,
            runner=runner,
        )
    except immutable.EvidenceError as exc:
        if str(exc) != EXPECTED_RECORDER_ERROR:
            raise R13FailureSealError(
                f"immutable r13 recorder rejection drifted: {exc}"
            ) from exc
    else:
        raise R13FailureSealError(
            "immutable r13 recorder unexpectedly published an envelope"
        )
    if len(captured) != 1:
        raise R13FailureSealError("r13 recorder command count drifted")
    completed = captured[0]
    contract = immutable._validated_r3_probe_contract(
        classification=arguments["classification"],
        argv=arguments["command"],
        cwd=arguments["cwd"],
        environment=dict(arguments["environment"]),
        input_paths=arguments["input_roots"],
        write_roots=arguments["write_roots"],
        output_path=arguments["output"],
    )
    grammar = _grammar_proof(
        immutable=immutable,
        corrected=corrected,
        completed=completed,
        contract=contract,
    )
    if Path(arguments["output"]).exists() or Path(
        arguments["output"]
    ).is_symlink():
        raise R13FailureSealError("r13 failed recorder published an envelope")
    stdout_path = evidence / STDOUT_NAME
    stderr_path = evidence / STDERR_NAME
    recorder_stdout_path = evidence / RECORDER_STDOUT_NAME
    recorder_stderr_path = evidence / RECORDER_STDERR_NAME
    _publish(stdout_path, completed.stdout)
    _publish(stderr_path, completed.stderr)
    _publish(recorder_stdout_path, b"")
    _publish(
        recorder_stderr_path,
        f"[schema5-evidence] ERROR: {EXPECTED_RECORDER_ERROR}\n".encode(),
    )
    archive = evidence / REPRODUCED_ARCHIVE_NAME
    shutil.move(str(root), str(archive))
    _remove_write_bits(archive)
    _require_recursively_read_only(archive)
    return {
        "grammar": grammar,
        "stdout": _file_ref(
            stdout_path, description="r13 Conda stdout"
        ),
        "stderr": _file_ref(
            stderr_path, description="r13 Conda stderr"
        ),
        "recorder_stdout": _file_ref(
            recorder_stdout_path, description="r13 recorder stdout"
        ),
        "recorder_stderr": _file_ref(
            recorder_stderr_path, description="r13 recorder stderr"
        ),
        "reproduced_probe_tree": {
            "path": str(archive),
            "inventory": _portable_inventory(archive),
            "recursively_read_only": True,
        },
        "offline_envelope_absent": True,
        "known_scheduler_job_ids": [],
        "result_mutation_count": 0,
    }


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
        "observed_failed_probe_tree": observed,
        "reproduction_root": str(REPRODUCTION_ROOT),
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
        or intent.get("reproduction_root") != str(REPRODUCTION_ROOT)
        or intent.get("evidence_root") != str(evidence)
    ):
        raise R13FailureSealError("r13 failure-seal intent drifted")
    state = intent.get("scientific_state")
    scheduler = intent.get("scheduler")
    if (
        not isinstance(state, dict)
        or state.get("captured_before_r14_production_outputs") is not True
        or state.get("result_mutation_count") != 0
        or state.get("scheduler_job_count") != 0
        or not isinstance(scheduler, dict)
        or scheduler.get("scheduler_user") != user
        or scheduler.get("matching_r13_jobs") != []
    ):
        raise R13FailureSealError("r13 sealed prelaunch state drifted")


def _archive_observed(
    evidence: Path, observed: Mapping[str, Any]
) -> dict[str, Any]:
    source = _safe_directory(
        R13_PROBE_ROOT, description="observed r13 probe root"
    )
    before = _portable_inventory(source)
    if observed.get("inventory") != before:
        raise R13FailureSealError("observed r13 probe preimage drifted")
    archive = evidence / OBSERVED_ARCHIVE_NAME
    shutil.copytree(
        source,
        archive,
        symlinks=True,
        copy_function=shutil.copy2,
    )
    if _portable_inventory(archive) != before:
        raise R13FailureSealError("observed r13 probe archive drifted")
    _remove_write_bits(archive)
    _require_recursively_read_only(archive)
    return {
        "path": str(archive),
        "inventory": _portable_inventory(archive),
        "recursively_read_only": True,
        "original_temporary_tree_preserved": True,
    }


def _verify_proof(
    evidence: Path,
    recovery: Path,
    intent: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> None:
    if (
        not isinstance(proof.get("grammar"), dict)
        or proof["grammar"].get("immutable_r13_rejected_raw") is not True
        or proof["grammar"].get(
            "immutable_r13_accepted_after_one_final_newline_removed"
        )
        is not True
        or proof["grammar"].get("corrected_r14_accepted_raw") is not True
        or proof.get("offline_envelope_absent") is not True
        or proof.get("known_scheduler_job_ids") != []
        or proof.get("result_mutation_count") != 0
    ):
        raise R13FailureSealError("r13 failure proof drifted")
    stdout_path = evidence / STDOUT_NAME
    stderr_path = evidence / STDERR_NAME
    if proof.get("stdout") != _file_ref(
        stdout_path, description="sealed r13 Conda stdout"
    ) or proof.get("stderr") != _file_ref(
        stderr_path, description="sealed r13 Conda stderr"
    ):
        raise R13FailureSealError("r13 Conda output binding drifted")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=stdout_path.read_bytes(),
        stderr=stderr_path.read_bytes(),
    )
    immutable = _r13_evidence_module(recovery)
    corrected = _corrected_evidence_module()
    reproduced = evidence / REPRODUCED_ARCHIVE_NAME
    arguments = _probe_arguments(immutable, recovery, reproduced)
    contract = immutable._validated_r3_probe_contract(
        classification=arguments["classification"],
        argv=[
            value.replace(str(reproduced), str(REPRODUCTION_ROOT))
            for value in arguments["command"]
        ],
        cwd=arguments["cwd"],
        environment={
            key: value.replace(str(reproduced), str(REPRODUCTION_ROOT))
            for key, value in arguments["environment"]
        },
        input_paths=arguments["input_roots"],
        write_roots=[REPRODUCTION_ROOT],
        output_path=REPRODUCTION_ROOT / OFFLINE_ENVELOPE,
    )
    grammar = _grammar_proof(
        immutable=immutable,
        corrected=corrected,
        completed=completed,
        contract=contract,
    )
    if grammar != proof["grammar"]:
        raise R13FailureSealError("r13 grammar proof binding drifted")
    for key, name in (
        ("observed_probe_tree", OBSERVED_ARCHIVE_NAME),
        ("reproduced_probe_tree", REPRODUCED_ARCHIVE_NAME),
    ):
        record = proof.get(key)
        root = evidence / name
        if (
            not isinstance(record, dict)
            or record.get("path") != str(root)
            or record.get("inventory") != _portable_inventory(root)
            or record.get("recursively_read_only") is not True
        ):
            raise R13FailureSealError(f"r13 {key} binding drifted")
        _require_recursively_read_only(root)
    if (
        proof["observed_probe_tree"]["inventory"]
        != intent["observed_failed_probe_tree"]["inventory"]
    ):
        raise R13FailureSealError("r13 observed archive differs from intent")


def verify_failure_seal(
    evidence_root: str | Path,
    *,
    recovery_root: str | Path,
    scheduler_user: str | None = None,
) -> dict[str, Any]:
    evidence = _safe_directory(
        evidence_root, description="r13 offline-delimiter failure evidence"
    )
    recovery = _safe_directory(
        recovery_root, description="schema-5 recovery root"
    )
    marker, marker_raw = _read_json(
        evidence / MARKER_NAME, description="r13 failure marker"
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
        raise R13FailureSealError("r13 failure marker drifted")
    intent, intent_raw = _read_json(
        evidence / INTENT_NAME, description="r13 failure intent"
    )
    sealed_user = intent.get("scheduler", {}).get("scheduler_user")
    if scheduler_user is None:
        scheduler_user = sealed_user
    if not isinstance(scheduler_user, str) or scheduler_user != sealed_user:
        raise R13FailureSealError("r13 scheduler-user binding drifted")
    _validate_intent(intent, recovery, evidence, scheduler_user)
    if marker.get("intent") != {
        "path": str(evidence / INTENT_NAME),
        "sha256": hashlib.sha256(intent_raw).hexdigest(),
        "size": len(intent_raw),
        "intent_id": intent["intent_id"],
    }:
        raise R13FailureSealError("r13 marker intent binding drifted")
    proof = marker.get("proof")
    if not isinstance(proof, dict):
        raise R13FailureSealError("r13 failure proof is malformed")
    _verify_proof(evidence, recovery, intent, proof)
    _require_recursively_read_only(evidence)
    return {
        "passed": True,
        "root": str(evidence),
        "protocol": PROTOCOL,
        "release_tag": R13_TAG,
        "release_git_commit": R13_COMMIT,
        "chain_namespace": R13_NAMESPACE,
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
    expected = recovery / EVIDENCE_RELATIVE_ROOT
    if evidence != expected:
        raise R13FailureSealError(
            f"r13 failure evidence root must be {expected}"
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
    observed = _validate_observed_probe_tree(recovery, R13_PROBE_ROOT)
    intent = _intent_payload(
        recovery, evidence, scheduler_user, observed
    )
    if not apply:
        return {
            "action": "would_seal",
            "classification": CLASSIFICATION,
            "observed_failed_probe_tree": observed,
            "scientific_state": intent["scientific_state"],
            "scheduler": intent["scheduler"],
        }
    if evidence.exists() or evidence.is_symlink():
        raise R13FailureSealError(
            "incomplete r13 failure evidence root requires quarantine"
        )
    if REPRODUCTION_ROOT.exists() or REPRODUCTION_ROOT.is_symlink():
        raise R13FailureSealError(
            "incomplete r13 reproduction root requires quarantine"
        )
    evidence.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    evidence.mkdir(mode=0o700)
    intent_path = evidence / INTENT_NAME
    intent_raw = _canonical_bytes(intent)
    _publish(intent_path, intent_raw)
    proof = _reproduce(evidence, recovery)
    proof["observed_probe_tree"] = _archive_observed(
        evidence, observed
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
