#!/usr/bin/env python3
"""Seal the deterministic r6 prelaunch toolchain-binding failure for r7."""

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
import types
from typing import Any, Mapping


SCHEMA_VERSION = 1
PROTOCOL = "schema5-v1.2-r6-prelaunch-toolchain-binding-failure-seal-v1"
CLASSIFICATION = (
    "deterministic_prelaunch_toolchain_binding_field_set_mismatch"
)
R6_TAG = "sweep-recovery-schema5-v1.2-r6"
R6_NAMESPACE = "schema5-v1.2-r6"
R6_COMMIT = "9b40414a0d90f569cf021ac152503746bedda6c7"
R6_TAG_OBJECT = "e6b9c2f2548f7ef58ad5163b65a25b164013a4e6"
R6_DURABLE_MARKER = "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R6_COMPLETE.json"
R6_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r6"
R6_TOOLCHAIN_RELATIVE = Path("toolchains/r6/conda")
R3_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r3"
EVIDENCE_RELATIVE_ROOT = Path(
    "prelaunch_failures/schema5-v1.2-r6-toolchain-binding"
)
MARKER_NAME = "PRELAUNCH_TOOLCHAIN_BINDING_FAILURE_SEALED.json"
INTENT_NAME = "PRELAUNCH_TOOLCHAIN_BINDING_FAILURE_INTENT.json"
STDOUT_NAME = "attempt.stdout"
STDERR_NAME = "attempt.stderr"
DEV_PYTHON = Path(
    "/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python3.11"
)
EXPECTED_STDERR = (
    "[schema5-evidence] ERROR: sealed r6 Conda toolchain verification "
    "identity drifted\n"
)
EXPECTED_RETURN_CODE = 2
R6_EXPECTED_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "release_tag",
        "chain_namespace",
        "toolchain_root",
        "base_prefix",
        "completion_marker",
        "marker_id",
        "installer_contract",
        "intent_id",
        "conda_executable",
        "runtime_identity_sha256",
        "complete_prefix_inventory_sha256",
        "read_only_probes",
        "binding_id",
    }
)
ACTUAL_ADDITIVE_FIELDS = frozenset({"portable_shebang"})


class R6FailureSealError(RuntimeError):
    """The r6 prelaunch failure cannot be sealed or verified exactly."""


def _load_historical_helpers():
    path = Path(__file__).resolve().with_name(
        "seal_schema5_r5_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r5_failure_helpers", path
    )
    if spec is None or spec.loader is None:
        raise R6FailureSealError("cannot load marker-last evidence helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_historical_helpers()
MarkerLastEvidenceError = _helpers.R5FailureSealError
_canonical_bytes = _helpers._canonical_bytes
_identity = _helpers._identity
_safe_directory = _helpers._safe_directory
_read_json = _helpers._read_json
_file_ref = _helpers._file_ref
_publish = _helpers._publish
_fsync_directory = _helpers._fsync_directory
_git = _helpers._git
_probe_tree_inventory = _helpers._probe_tree_inventory


def _release_binding(recovery: Path) -> dict[str, Any]:
    checkout = _safe_directory(recovery / R6_CHECKOUT, description="r6 checkout")
    tag_ref = f"refs/tags/{R6_TAG}"
    if (
        _git(checkout, "cat-file", "-t", tag_ref) != "tag"
        or _git(checkout, "rev-parse", tag_ref) != R6_TAG_OBJECT
        or _git(checkout, "rev-parse", f"{tag_ref}^{{commit}}") != R6_COMMIT
        or _git(checkout, "rev-parse", "HEAD") != R6_COMMIT
        or _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
        or (checkout / ".git/objects/info/alternates").exists()
    ):
        raise R6FailureSealError("r6 checkout identity drifted")
    durable_path = recovery / R6_DURABLE_MARKER
    durable, durable_raw = _read_json(
        durable_path, description="r6 durable release marker"
    )
    if (
        durable.get("passed") is not True
        or durable.get("release_tag") != R6_TAG
        or durable.get("chain_namespace") != R6_NAMESPACE
        or durable.get("release_git_commit") != R6_COMMIT
        or durable.get("release_tag_object") != R6_TAG_OBJECT
        or durable.get("remote_commit") != R6_COMMIT
        or durable.get("remote_peeled_commit") != R6_COMMIT
        or durable.get("remote_tag_object") != R6_TAG_OBJECT
    ):
        raise R6FailureSealError("r6 durable release identity drifted")
    return {
        "checkout": str(checkout),
        "release_tag": R6_TAG,
        "release_git_commit": R6_COMMIT,
        "release_tag_object": R6_TAG_OBJECT,
        "recorder": _file_ref(
            checkout / "scripts/seal_recovery_evidence.py",
            description="r6 recovery-evidence recorder",
        ),
        "provisioner": _file_ref(
            checkout / "scripts/provision_schema5_conda_toolchain.py",
            description="r6 Conda provisioner",
        ),
        "durable_marker": {
            "path": str(durable_path),
            "sha256": hashlib.sha256(durable_raw).hexdigest(),
            "size": len(durable_raw),
            "marker_id": durable.get("marker_id"),
        },
    }


def _toolchain_binding(recovery: Path) -> dict[str, Any]:
    checkout = recovery / R6_CHECKOUT
    provisioner = checkout / "scripts/provision_schema5_conda_toolchain.py"
    runtime_identity = checkout / "scripts/schema5_conda_runtime_identity.py"
    toolchain = recovery / R6_TOOLCHAIN_RELATIVE
    names = (
        "scripts",
        "scripts.schema5_conda_runtime_identity",
        "scripts.provision_schema5_conda_toolchain",
    )
    saved = {name: sys.modules.get(name) for name in names}
    try:
        package = types.ModuleType("scripts")
        package.__path__ = []
        sys.modules["scripts"] = package
        runtime_spec = importlib.util.spec_from_file_location(
            "scripts.schema5_conda_runtime_identity", runtime_identity
        )
        if runtime_spec is None or runtime_spec.loader is None:
            raise R6FailureSealError("cannot load r6 runtime-identity verifier")
        runtime_module = importlib.util.module_from_spec(runtime_spec)
        sys.modules[runtime_spec.name] = runtime_module
        runtime_spec.loader.exec_module(runtime_module)
        provisioner_spec = importlib.util.spec_from_file_location(
            "scripts.provision_schema5_conda_toolchain", provisioner
        )
        if provisioner_spec is None or provisioner_spec.loader is None:
            raise R6FailureSealError("cannot load r6 toolchain verifier")
        provisioner_module = importlib.util.module_from_spec(provisioner_spec)
        sys.modules[provisioner_spec.name] = provisioner_module
        provisioner_spec.loader.exec_module(provisioner_module)
        binding = provisioner_module.verified_conda_toolchain_binding(
            toolchain, exercise=False
        )
    except Exception as exc:
        if isinstance(exc, R6FailureSealError):
            raise
        raise R6FailureSealError(
            f"r6 sealed toolchain verification failed: {exc}"
        ) from exc
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    actual_fields = frozenset(binding) if isinstance(binding, dict) else frozenset()
    if (
        not isinstance(binding, dict)
        or actual_fields != R6_EXPECTED_BINDING_FIELDS | ACTUAL_ADDITIVE_FIELDS
        or binding.get("protocol")
        != "schema5-v1.2-r6-offline-conda-toolchain-v1"
        or binding.get("release_tag") != R6_TAG
        or binding.get("chain_namespace") != R6_NAMESPACE
        or binding.get("toolchain_root") != str(toolchain)
        or binding.get("base_prefix") != str(toolchain / "base")
        or binding.get("portable_shebang", {}).get("interpreter")
        != str(toolchain / "base/bin/python")
    ):
        raise R6FailureSealError("r6 sealed toolchain binding drifted")
    return json.loads(json.dumps(binding, sort_keys=True))


def _scheduler_quiescence(user: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["/usr/bin/squeue", "-h", "-r", "-u", user, "-o", "%i|%j|%k|%T"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise R6FailureSealError(f"squeue failed: {completed.stderr.strip()}")
    matching = [
        row
        for row in completed.stdout.splitlines()
        if "s5v12r6" in row
        or "schema5-v1.2-r6" in row
        or "sweep-recovery-schema5-v1.2-r6" in row
    ]
    if matching:
        raise R6FailureSealError(f"r6 scheduler jobs remain live: {matching}")
    return {"scheduler_user": user, "matching_r6_jobs": []}


def _historical_scientific_paths(recovery: Path) -> list[str]:
    results = recovery.parent.parent
    return [
        str(path)
        for path in (
            results / ".dispatcher-schema5-v1",
            results / "server_pools/schema5-v1",
            results / "full_sweep_schema5_v1",
            results / "full_sweep_agent_counts_schema5_v1",
            results / "full_sweep_agent_count_7_schema5_v1",
            recovery / "slurm_canaries/schema5-v1.2-r6",
            recovery / "materialization_pilots/schema5-v1.2-r6",
            recovery / "PROTECTED_CAPACITY_COMPLETE.json",
            recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R6.json",
            recovery / "prelaunch_failures/schema5-v1.2-r3",
        )
    ]


def _permanently_absent_paths(recovery: Path, user: str) -> list[str]:
    probe_root = Path(f"/tmp/schema5-r3-prelaunch-{user}-r6")
    return [
        str(recovery / "slurm_canaries/schema5-v1.2-r6"),
        str(recovery / "materialization_pilots/schema5-v1.2-r6"),
        str(recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R6.json"),
        str(
            probe_root
            / "FAILURE_UNSAFE_RECORDED_BROKEN_INTERNAL_SYMLINK.source.json"
        ),
    ]


def _scientific_state(recovery: Path, user: str) -> dict[str, Any]:
    historical = _historical_scientific_paths(recovery)
    present = [
        path for path in historical if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise R6FailureSealError(
            f"r6 failure is not prelaunch zero-result evidence: {present}"
        )
    return {
        "captured_before_r7_outputs": True,
        "result_mutation_count": 0,
        "scheduler_job_count": 0,
        "required_absent_paths_at_seal": historical,
        "permanently_absent_r6_execution_paths": _permanently_absent_paths(
            recovery, user
        ),
    }


def _validate_permanent_absence(
    recovery: Path, user: str, state: Mapping[str, Any]
) -> None:
    expected = _permanently_absent_paths(recovery, user)
    if state.get("permanently_absent_r6_execution_paths") != expected:
        raise R6FailureSealError("sealed r6 permanent-absence contract drifted")
    present = [
        path for path in expected if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise R6FailureSealError(f"r6 execution namespace appeared later: {present}")


def _command_contract(recovery: Path, user: str) -> dict[str, Any]:
    r6_checkout = recovery / R6_CHECKOUT
    r3_checkout = recovery / R3_CHECKOUT
    toolchain = recovery / R6_TOOLCHAIN_RELATIVE
    probe_root = Path(f"/tmp/schema5-r3-prelaunch-{user}-r6")
    envelope = (
        probe_root
        / "FAILURE_UNSAFE_RECORDED_BROKEN_INTERNAL_SYMLINK.source.json"
    )
    common = [
        "--environment", "HOME", str(probe_root / "home"),
        "--environment", "PYTHONDONTWRITEBYTECODE", "1",
        "--environment", "PYTHONNOUSERSITE", "1",
        "--environment", "TEMP", str(probe_root / "tmp"),
        "--environment", "TMP", str(probe_root / "tmp"),
        "--environment", "TMPDIR", str(probe_root / "tmp"),
        "--environment", "XDG_CACHE_HOME", str(probe_root / "xdg-cache"),
        "--environment", "XDG_CONFIG_HOME", str(probe_root / "xdg-config"),
        "--environment", "XDG_DATA_HOME", str(probe_root / "xdg-data"),
        "--environment", "XDG_STATE_HOME", str(probe_root / "xdg-state"),
    ]
    argv = [
        str(DEV_PYTHON),
        "-I",
        "-B",
        str(r6_checkout / "scripts/seal_recovery_evidence.py"),
        "record-prelaunch-attempt",
        "--output",
        str(envelope),
        "--classification",
        "unsafe_recorded_broken_internal_symlink",
        "--cwd",
        str(r3_checkout),
        *common,
        "--input-root",
        "tagged-r3-release-checkout",
        str(r3_checkout),
        "--input-root",
        "tagged-r6-release-checkout",
        str(r6_checkout),
        "--input-root",
        "sealed-r6-conda-toolchain",
        str(toolchain),
        "--input-root",
        "recorded-shared-conda-base",
        "/orcd/data/lhtsai/001/om2/mabdel03/miniforge3",
        "--write-root",
        str(probe_root),
        "--command",
        str(toolchain / "base/bin/python"),
        "-I",
        "-B",
        str(r3_checkout / "scripts/run_schema5_materialization_pilot.py"),
        "conda-runtime-identity",
        "--conda-executable",
        "/orcd/data/lhtsai/001/om2/mabdel03/miniforge3/bin/conda",
    ]
    return {
        "argv": argv,
        "cwd": "/orcd/data/tpoggio/001/mabdel03/agents_scaling",
        "probe_root": str(probe_root),
        "expected_envelope": str(envelope),
        "environment": {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
        "timeout_seconds": 1800,
    }


def _validate_original_probe_tree(value: Any) -> dict[str, Any]:
    expected_entries: list[dict[str, Any]] = []
    expected = {
        "exists": True,
        "entries": expected_entries,
        "inventory_sha256": hashlib.sha256(
            _canonical_bytes(expected_entries)
        ).hexdigest(),
    }
    if value != expected:
        raise R6FailureSealError("original failed r6 probe tree is not empty")
    return json.loads(json.dumps(value, sort_keys=True))


def _field_mismatch(binding: Mapping[str, Any]) -> dict[str, Any]:
    actual = frozenset(binding)
    report = {
        "r6_recorder_expected_fields": sorted(R6_EXPECTED_BINDING_FIELDS),
        "verified_binding_fields": sorted(actual),
        "unexpected_fields": sorted(actual - R6_EXPECTED_BINDING_FIELDS),
        "missing_fields": sorted(R6_EXPECTED_BINDING_FIELDS - actual),
    }
    if (
        report["unexpected_fields"] != sorted(ACTUAL_ADDITIVE_FIELDS)
        or report["missing_fields"] != []
    ):
        raise R6FailureSealError("r6 binding mismatch classification drifted")
    return report


def _intent_payload(recovery: Path, user: str) -> dict[str, Any]:
    command = _command_contract(recovery, user)
    toolchain = _toolchain_binding(recovery)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": f"{PROTOCOL}-intent",
        "classification": CLASSIFICATION,
        "source_release": _release_binding(recovery),
        "sealed_r6_toolchain": toolchain,
        "field_mismatch": _field_mismatch(toolchain),
        "command": command,
        "original_probe_tree": _validate_original_probe_tree(
            _probe_tree_inventory(Path(command["probe_root"]))
        ),
        "scientific_state": _scientific_state(recovery, user),
        "scheduler": _scheduler_quiescence(user),
    }
    payload["intent_id"] = _identity(payload, "intent_id")
    return payload


def _validate_intent(
    intent: Mapping[str, Any], recovery: Path, user: str
) -> None:
    if (
        intent.get("schema_version") != SCHEMA_VERSION
        or intent.get("protocol") != f"{PROTOCOL}-intent"
        or intent.get("classification") != CLASSIFICATION
        or intent.get("intent_id") != _identity(intent, "intent_id")
        or intent.get("source_release") != _release_binding(recovery)
        or intent.get("sealed_r6_toolchain") != _toolchain_binding(recovery)
        or intent.get("field_mismatch")
        != _field_mismatch(intent.get("sealed_r6_toolchain", {}))
        or intent.get("command") != _command_contract(recovery, user)
        or _validate_original_probe_tree(intent.get("original_probe_tree"))
        != intent.get("original_probe_tree")
        or intent.get("scheduler") != _scheduler_quiescence(user)
    ):
        raise R6FailureSealError("r6 failure-seal intent drifted")
    state = intent.get("scientific_state")
    if not isinstance(state, dict):
        raise R6FailureSealError("r6 failure-seal intent lacks scientific state")
    if (
        state.get("captured_before_r7_outputs") is not True
        or state.get("result_mutation_count") != 0
        or state.get("scheduler_job_count") != 0
        or state.get("required_absent_paths_at_seal")
        != _historical_scientific_paths(recovery)
    ):
        raise R6FailureSealError("r6 prelaunch scientific-state evidence drifted")
    _validate_permanent_absence(recovery, user, state)


def _binding(
    evidence: Path, marker: Mapping[str, Any], marker_raw: bytes
) -> dict[str, Any]:
    return {
        "passed": True,
        "root": str(evidence),
        "protocol": PROTOCOL,
        "release_tag": R6_TAG,
        "release_git_commit": R6_COMMIT,
        "chain_namespace": R6_NAMESPACE,
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


def verify_failure_seal(
    evidence_root: str | Path,
    *,
    recovery_root: str | Path,
    scheduler_user: str | None = None,
) -> dict[str, Any]:
    evidence = _safe_directory(evidence_root, description="r6 failure evidence")
    recovery = _safe_directory(recovery_root, description="schema-5 recovery root")
    marker, marker_raw = _read_json(
        evidence / MARKER_NAME, description="r6 prelaunch failure marker"
    )
    if (
        set(marker)
        != {
            "schema_version",
            "protocol",
            "classification",
            "intent",
            "attempt",
            "marker_id",
        }
        or marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("protocol") != PROTOCOL
        or marker.get("classification") != CLASSIFICATION
        or marker.get("marker_id") != _identity(marker, "marker_id")
    ):
        raise R6FailureSealError("r6 prelaunch failure marker drifted")
    intent, intent_raw = _read_json(
        evidence / INTENT_NAME, description="r6 prelaunch failure intent"
    )
    sealed_user = intent.get("scheduler", {}).get("scheduler_user")
    if scheduler_user is None:
        scheduler_user = sealed_user
    if not isinstance(scheduler_user, str) or scheduler_user != sealed_user:
        raise R6FailureSealError("r6 failure scheduler-user binding drifted")
    _validate_intent(intent, recovery, scheduler_user)
    if marker.get("intent") != {
        "path": str(evidence / INTENT_NAME),
        "sha256": hashlib.sha256(intent_raw).hexdigest(),
        "size": len(intent_raw),
        "intent_id": intent["intent_id"],
    }:
        raise R6FailureSealError("r6 failure marker intent binding drifted")
    stdout_ref = _file_ref(evidence / STDOUT_NAME, description="r6 failure stdout")
    stderr_ref = _file_ref(evidence / STDERR_NAME, description="r6 failure stderr")
    attempt = marker.get("attempt")
    if not isinstance(attempt, dict):
        raise R6FailureSealError("r6 failure attempt binding is malformed")
    if (
        attempt
        != {
            "returncode": EXPECTED_RETURN_CODE,
            "stdout": stdout_ref,
            "stderr": stderr_ref,
            "expected_stderr": EXPECTED_STDERR,
            "probe_tree_before": attempt.get("probe_tree_before"),
            "probe_tree_after": attempt.get("probe_tree_after"),
            "expected_envelope_absent": True,
        }
        or stdout_ref["size"] != 0
        or (evidence / STDERR_NAME).read_text(encoding="utf-8")
        != EXPECTED_STDERR
        or attempt.get("probe_tree_before") != intent.get("original_probe_tree")
        or attempt.get("probe_tree_before") != attempt.get("probe_tree_after")
    ):
        raise R6FailureSealError("r6 failure signature drifted")
    if Path(intent["command"]["expected_envelope"]).exists():
        raise R6FailureSealError("r6 recorder unexpectedly produced an envelope")
    for path in (
        evidence,
        evidence / INTENT_NAME,
        evidence / STDOUT_NAME,
        evidence / STDERR_NAME,
        evidence / MARKER_NAME,
    ):
        if stat.S_IMODE(path.lstat().st_mode) & 0o222:
            raise R6FailureSealError(f"r6 failure evidence is writable: {path}")
    return _binding(evidence, marker, marker_raw)


def seal_failure(
    *,
    evidence_root: str | Path,
    recovery_root: str | Path,
    scheduler_user: str,
    apply: bool,
) -> dict[str, Any]:
    recovery = _safe_directory(recovery_root, description="schema-5 recovery root")
    expected_root = recovery / EVIDENCE_RELATIVE_ROOT
    evidence = Path(evidence_root).expanduser().absolute()
    if evidence != expected_root:
        raise R6FailureSealError(f"r6 failure evidence root must be {expected_root}")
    marker_path = evidence / MARKER_NAME
    if marker_path.exists() or marker_path.is_symlink():
        return {
            "action": "already_sealed",
            **verify_failure_seal(
                evidence,
                recovery_root=recovery,
                scheduler_user=scheduler_user,
            ),
        }
    intent = _intent_payload(recovery, scheduler_user)
    if not apply:
        return {
            "action": "would_seal",
            "classification": CLASSIFICATION,
            "field_mismatch": intent["field_mismatch"],
            "command": intent["command"],
            "scientific_state": intent["scientific_state"],
            "scheduler": intent["scheduler"],
        }
    if evidence.exists() or evidence.is_symlink():
        raise R6FailureSealError(
            "incomplete r6 failure evidence root requires quarantine"
        )
    evidence.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    evidence.mkdir(mode=0o700)
    intent_path = evidence / INTENT_NAME
    intent_raw = _canonical_bytes(intent)
    _publish(intent_path, intent_raw)
    contract = intent["command"]
    before = _probe_tree_inventory(Path(contract["probe_root"]))
    if before != intent["original_probe_tree"]:
        raise R6FailureSealError("r6 failed probe tree changed before reproduction")
    completed = subprocess.run(
        contract["argv"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=contract["cwd"],
        env=contract["environment"],
        timeout=contract["timeout_seconds"],
        check=False,
    )
    after = _probe_tree_inventory(Path(contract["probe_root"]))
    if (
        completed.returncode != EXPECTED_RETURN_CODE
        or completed.stdout
        or completed.stderr.decode("utf-8", "strict") != EXPECTED_STDERR
        or before != after
        or Path(contract["expected_envelope"]).exists()
    ):
        raise R6FailureSealError("r6 failure reproduction drifted")
    _publish(evidence / STDOUT_NAME, completed.stdout)
    _publish(evidence / STDERR_NAME, completed.stderr)
    stdout_ref = _file_ref(evidence / STDOUT_NAME, description="r6 failure stdout")
    stderr_ref = _file_ref(evidence / STDERR_NAME, description="r6 failure stderr")
    marker: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "classification": CLASSIFICATION,
        "intent": {
            "path": str(intent_path),
            "sha256": hashlib.sha256(intent_raw).hexdigest(),
            "size": len(intent_raw),
            "intent_id": intent["intent_id"],
        },
        "attempt": {
            "returncode": completed.returncode,
            "stdout": stdout_ref,
            "stderr": stderr_ref,
            "expected_stderr": EXPECTED_STDERR,
            "probe_tree_before": before,
            "probe_tree_after": after,
            "expected_envelope_absent": True,
        },
    }
    marker["marker_id"] = _identity(marker, "marker_id")
    _publish(marker_path, _canonical_bytes(marker))
    os.chmod(evidence, 0o555)
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
        R6FailureSealError,
        MarkerLastEvidenceError,
        OSError,
        UnicodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
