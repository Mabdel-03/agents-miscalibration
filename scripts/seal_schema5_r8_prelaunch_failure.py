#!/usr/bin/env python3
"""Seal r8's deterministic r3-probe dependency-import failure for r9."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
from typing import Any, Mapping


SCHEMA_VERSION = 1
PROTOCOL = "schema5-v1.2-r8-prelaunch-r3-probe-import-failure-seal-v1"
CLASSIFICATION = "deterministic_prelaunch_r3_probe_dependency_import_failure"
R8_TAG = "sweep-recovery-schema5-v1.2-r8"
R8_NAMESPACE = "schema5-v1.2-r8"
R8_COMMIT = "ed18bb969c6632d2b2643024064a2b4d1b63f6de"
R8_TAG_OBJECT = "c31692215b98c176bdd138109b0c2b6ec0a63e1f"
R8_DURABLE_MARKER = "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R8_COMPLETE.json"
R8_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r8"
R8_TOOLCHAIN_RELATIVE = Path("toolchains/r8/conda")
R3_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r3"
EVIDENCE_RELATIVE_ROOT = Path(
    "prelaunch_failures/schema5-v1.2-r8-r3-probe-import"
)
MARKER_NAME = "PRELAUNCH_R3_PROBE_IMPORT_FAILURE_SEALED.json"
INTENT_NAME = "PRELAUNCH_R3_PROBE_IMPORT_FAILURE_INTENT.json"
OUTER_STDOUT_NAME = "recorder.stdout"
OUTER_STDERR_NAME = "recorder.stderr"
INNER_STDOUT_NAME = "r3-probe.stdout"
INNER_STDERR_NAME = "r3-probe.stderr"
DEV_PYTHON = Path(
    "/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python"
)
SHARED_CONDA_BASE = Path(
    "/orcd/data/lhtsai/001/om2/mabdel03/miniforge3"
)
BROKEN_ENVELOPE_NAME = (
    "FAILURE_UNSAFE_RECORDED_BROKEN_INTERNAL_SYMLINK.source.json"
)
OFFLINE_ENVELOPE_NAME = (
    "FAILURE_OFFLINE_CLONE_UNSEEDED_RELEASE_LOCAL_CACHE.source.json"
)
EXPECTED_RECORDER_ERROR = (
    "[schema5-evidence] ERROR: r3 broken-symlink probe lacks the "
    "canonical runtime-identity error signature\n"
)
EXPECTED_IMPORT_ERROR = "ModuleNotFoundError: No module named 'backoff'\n"


class R8FailureSealError(RuntimeError):
    """The r8 prelaunch failure cannot be sealed or verified exactly."""


def _load_helpers():
    path = Path(__file__).resolve().with_name(
        "seal_schema5_r7_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r7_failure_helpers", path
    )
    if spec is None or spec.loader is None:
        raise R8FailureSealError("cannot load marker-last evidence helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_helpers()
MarkerLastEvidenceError = (
    _helpers.R7FailureSealError,
    *_helpers.MarkerLastEvidenceError,
)
_canonical_bytes = _helpers._canonical_bytes
_identity = _helpers._identity
_safe_directory = _helpers._safe_directory
_read_json = _helpers._read_json
_file_ref = _helpers._file_ref
_publish = _helpers._publish
_fsync_directory = _helpers._fsync_directory


def _git(checkout: Path, *arguments: str) -> str:
    return _helpers._run_git(
        checkout,
        *arguments,
        optional_locks=False,
    )


def _release_binding(recovery: Path) -> dict[str, Any]:
    checkout = _safe_directory(recovery / R8_CHECKOUT, description="r8 checkout")
    tag_ref = f"refs/tags/{R8_TAG}"
    if (
        _git(checkout, "cat-file", "-t", tag_ref) != "tag"
        or _git(checkout, "rev-parse", tag_ref) != R8_TAG_OBJECT
        or _git(checkout, "rev-parse", f"{tag_ref}^{{commit}}") != R8_COMMIT
        or _git(checkout, "rev-parse", "HEAD") != R8_COMMIT
        or _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
        or (checkout / ".git/objects/info/alternates").exists()
    ):
        raise R8FailureSealError("r8 checkout identity drifted")
    durable_path = recovery / R8_DURABLE_MARKER
    durable, durable_raw = _read_json(
        durable_path, description="r8 durable release marker"
    )
    bundle_path = Path(str(durable.get("bundle_path", "")))
    if (
        durable.get("passed") is not True
        or durable.get("release_tag") != R8_TAG
        or durable.get("chain_namespace") != R8_NAMESPACE
        or durable.get("release_git_commit") != R8_COMMIT
        or durable.get("release_tag_object") != R8_TAG_OBJECT
        or durable.get("remote_commit") != R8_COMMIT
        or durable.get("remote_peeled_commit") != R8_COMMIT
        or durable.get("remote_tag_object") != R8_TAG_OBJECT
        or not bundle_path.is_file()
        or bundle_path.is_symlink()
        or _file_ref(bundle_path, description="r8 durable Git bundle")["sha256"]
        != durable.get("bundle_sha256")
    ):
        raise R8FailureSealError("r8 durable release identity drifted")
    return {
        "checkout": str(checkout),
        "release_tag": R8_TAG,
        "release_git_commit": R8_COMMIT,
        "release_tag_object": R8_TAG_OBJECT,
        "recorder": _file_ref(
            checkout / "scripts/seal_recovery_evidence.py",
            description="r8 recovery-evidence recorder",
        ),
        "r3_pilot": _file_ref(
            recovery
            / R3_CHECKOUT
            / "scripts/run_schema5_materialization_pilot.py",
            description="immutable r3 materialization pilot",
        ),
        "r3_runtime_identity": _file_ref(
            recovery
            / R3_CHECKOUT
            / "scripts/schema5_conda_runtime_identity.py",
            description="immutable r3 Conda runtime-identity tool",
        ),
        "durable_marker": {
            "path": str(durable_path),
            "sha256": hashlib.sha256(durable_raw).hexdigest(),
            "size": len(durable_raw),
            "marker_id": durable.get("marker_id"),
        },
        "durable_bundle": _file_ref(
            bundle_path, description="r8 durable Git bundle"
        ),
    }


def _toolchain_binding(recovery: Path) -> dict[str, Any]:
    checkout = recovery / R8_CHECKOUT
    provisioner = checkout / "scripts/provision_schema5_conda_toolchain.py"
    runtime_identity = checkout / "scripts/schema5_conda_runtime_identity.py"
    toolchain = recovery / R8_TOOLCHAIN_RELATIVE
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
            raise R8FailureSealError("cannot load r8 runtime-identity verifier")
        runtime_module = importlib.util.module_from_spec(runtime_spec)
        sys.modules[runtime_spec.name] = runtime_module
        runtime_spec.loader.exec_module(runtime_module)
        provisioner_spec = importlib.util.spec_from_file_location(
            "scripts.provision_schema5_conda_toolchain", provisioner
        )
        if provisioner_spec is None or provisioner_spec.loader is None:
            raise R8FailureSealError("cannot load r8 toolchain verifier")
        provisioner_module = importlib.util.module_from_spec(provisioner_spec)
        sys.modules[provisioner_spec.name] = provisioner_module
        provisioner_spec.loader.exec_module(provisioner_module)
        binding = provisioner_module.verified_conda_toolchain_binding(
            toolchain, exercise=False
        )
    except Exception as exc:
        if isinstance(exc, R8FailureSealError):
            raise
        raise R8FailureSealError(
            f"r8 sealed toolchain verification failed: {exc}"
        ) from exc
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    if (
        not isinstance(binding, dict)
        or binding.get("protocol")
        != "schema5-v1.2-r8-offline-conda-toolchain-v1"
        or binding.get("release_tag") != R8_TAG
        or binding.get("chain_namespace") != R8_NAMESPACE
        or binding.get("toolchain_root") != str(toolchain)
        or binding.get("portable_shebang", {}).get("interpreter")
        != str(toolchain / "base/bin/python")
    ):
        raise R8FailureSealError("r8 sealed toolchain binding drifted")
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
        raise R8FailureSealError(f"squeue failed: {completed.stderr.strip()}")
    matching = [
        row
        for row in completed.stdout.splitlines()
        if "s5v12r8" in row
        or "schema5-v1.2-r8" in row
        or "sweep-recovery-schema5-v1.2-r8" in row
    ]
    if matching:
        raise R8FailureSealError(f"r8 scheduler jobs remain live: {matching}")
    return {"scheduler_user": user, "matching_r8_jobs": []}


def _probe_root(user: str) -> Path:
    return Path(f"/tmp/schema5-r3-prelaunch-{user}-r8")


def _empty_probe_tree(user: str) -> dict[str, Any]:
    root = _safe_directory(
        _probe_root(user), description="original failed r8 probe root"
    )
    entries = sorted(path.name for path in root.iterdir())
    if entries:
        raise R8FailureSealError(
            f"original failed r8 probe root is not empty: {entries}"
        )
    metadata = root.lstat()
    return {
        "path": str(root),
        "exists": True,
        "type": "directory",
        "mode": stat.S_IMODE(metadata.st_mode),
        "entries": [],
        "inventory_sha256": hashlib.sha256(
            _canonical_bytes([])
        ).hexdigest(),
    }


def _permanently_absent_paths(recovery: Path, user: str) -> list[str]:
    probe = _probe_root(user)
    return [
        str(recovery / "slurm_canaries/schema5-v1.2-r8"),
        str(recovery / "materialization_pilots/schema5-v1.2-r8"),
        str(recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R8.json"),
        str(probe / BROKEN_ENVELOPE_NAME),
        str(probe / OFFLINE_ENVELOPE_NAME),
    ]


def _scientific_state(recovery: Path, user: str) -> dict[str, Any]:
    results = recovery.parent.parent
    required_absent = [
        str(path)
        for path in (
            results / ".dispatcher-schema5-v1",
            results / "server_pools/schema5-v1",
            results / "full_sweep_schema5_v1",
            results / "full_sweep_agent_counts_schema5_v1",
            results / "full_sweep_agent_count_7_schema5_v1",
            recovery / "slurm_canaries/schema5-v1.2-r8",
            recovery / "materialization_pilots/schema5-v1.2-r8",
            recovery / "PROTECTED_CAPACITY_COMPLETE.json",
            recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R8.json",
            recovery / "prelaunch_failures/schema5-v1.2-r3",
        )
    ]
    present = [
        path
        for path in required_absent
        if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise R8FailureSealError(
            f"r8 failure is not prelaunch zero-result evidence: {present}"
        )
    return {
        "captured_before_r9_outputs": True,
        "result_mutation_count": 0,
        "scheduler_job_count": 0,
        "required_absent_paths_at_seal": required_absent,
        "original_r8_probe_tree": _empty_probe_tree(user),
        "permanently_absent_r8_execution_paths": _permanently_absent_paths(
            recovery, user
        ),
    }


def _validate_permanent_absence(
    recovery: Path, user: str, state: Mapping[str, Any]
) -> None:
    expected = _permanently_absent_paths(recovery, user)
    if state.get("permanently_absent_r8_execution_paths") != expected:
        raise R8FailureSealError("sealed r8 permanent-absence contract drifted")
    present = [
        path for path in expected if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise R8FailureSealError(f"r8 execution namespace appeared later: {present}")
    if state.get("original_r8_probe_tree") != _empty_probe_tree(user):
        raise R8FailureSealError("original failed r8 probe tree drifted")


def _probe_environment(user: str) -> dict[str, str]:
    root = _probe_root(user)
    return {
        "HOME": str(root / "home"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TEMP": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "TMPDIR": str(root / "tmp"),
        "XDG_CACHE_HOME": str(root / "xdg-cache"),
        "XDG_CONFIG_HOME": str(root / "xdg-config"),
        "XDG_DATA_HOME": str(root / "xdg-data"),
        "XDG_STATE_HOME": str(root / "xdg-state"),
    }


def _outer_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }


def _command_contract(recovery: Path, user: str) -> dict[str, Any]:
    r8_checkout = recovery / R8_CHECKOUT
    r3_checkout = recovery / R3_CHECKOUT
    toolchain = recovery / R8_TOOLCHAIN_RELATIVE
    probe = _probe_root(user)
    envelope = probe / BROKEN_ENVELOPE_NAME
    environment = _probe_environment(user)
    common: list[str] = []
    for key in (
        "HOME",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    ):
        common.extend(("--environment", key, environment[key]))
    inner = [
        str(toolchain / "base/bin/python"),
        "-I",
        "-B",
        str(r3_checkout / "scripts/run_schema5_materialization_pilot.py"),
        "conda-runtime-identity",
        "--conda-executable",
        str(SHARED_CONDA_BASE / "bin/conda"),
    ]
    outer = [
        str(DEV_PYTHON),
        "-I",
        "-B",
        str(r8_checkout / "scripts/seal_recovery_evidence.py"),
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
        "tagged-r8-release-checkout",
        str(r8_checkout),
        "--input-root",
        "sealed-r8-conda-toolchain",
        str(toolchain),
        "--input-root",
        "recorded-shared-conda-base",
        str(SHARED_CONDA_BASE),
        "--write-root",
        str(probe),
        "--apply",
        "--command",
        *inner,
    ]
    return {
        "outer_argv": outer,
        "outer_cwd": str(r8_checkout),
        "outer_environment": _outer_environment(),
        "inner_argv": inner,
        "inner_cwd": str(r3_checkout),
        "inner_environment": environment,
        "probe_root": str(probe),
        "expected_envelope": str(envelope),
        "expected_outer_returncode": 2,
        "expected_outer_stderr": EXPECTED_RECORDER_ERROR,
        "expected_inner_returncode": 1,
        "expected_inner_error": EXPECTED_IMPORT_ERROR.rstrip(),
    }


def _intent_payload(recovery: Path, evidence: Path, user: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": f"{PROTOCOL}-intent",
        "classification": CLASSIFICATION,
        "source_release": _release_binding(recovery),
        "sealed_r8_toolchain": _toolchain_binding(recovery),
        "commands": _command_contract(recovery, user),
        "scientific_state": _scientific_state(recovery, user),
        "scheduler": _scheduler_quiescence(user),
        "evidence_root": str(evidence),
    }
    payload["intent_id"] = _identity(payload, "intent_id")
    return payload


def _validate_intent(
    intent: Mapping[str, Any], recovery: Path, evidence: Path, user: str
) -> None:
    if (
        intent.get("schema_version") != SCHEMA_VERSION
        or intent.get("protocol") != f"{PROTOCOL}-intent"
        or intent.get("classification") != CLASSIFICATION
        or intent.get("intent_id") != _identity(intent, "intent_id")
        or intent.get("source_release") != _release_binding(recovery)
        or intent.get("sealed_r8_toolchain") != _toolchain_binding(recovery)
        or intent.get("commands") != _command_contract(recovery, user)
        or intent.get("evidence_root") != str(evidence)
        or intent.get("scheduler") != _scheduler_quiescence(user)
    ):
        raise R8FailureSealError("r8 failure-seal intent drifted")
    state = intent.get("scientific_state")
    if (
        not isinstance(state, dict)
        or state.get("captured_before_r9_outputs") is not True
        or state.get("result_mutation_count") != 0
        or state.get("scheduler_job_count") != 0
    ):
        raise R8FailureSealError("r8 prelaunch scientific-state evidence drifted")
    _validate_permanent_absence(recovery, user, state)


def _run(
    argv: list[str], *, cwd: str, environment: Mapping[str, str], timeout: int
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(environment),
        timeout=timeout,
        check=False,
    )


def _validate_inner_stderr(stderr: str, recovery: Path) -> None:
    r3 = re.escape(str(recovery / R3_CHECKOUT))
    if (
        not stderr.startswith("Traceback (most recent call last):\n")
        or not stderr.endswith(EXPECTED_IMPORT_ERROR)
        or re.search(
            rf'{r3}/scripts/run_schema5_materialization_pilot[.]py", line 63, '
            r"in <module>",
            stderr,
        )
        is None
        or re.search(
            rf'{r3}/src/agents_scaling/serving/client[.]py", line 26, '
            r"in <module>\n    import backoff",
            stderr,
        )
        is None
        or stderr.count("Traceback (most recent call last):") != 1
    ):
        raise R8FailureSealError(
            "r8 inner r3-probe dependency-import signature drifted"
        )


def _execute_proof(
    evidence: Path, recovery: Path, intent: Mapping[str, Any], user: str
) -> dict[str, Any]:
    commands = intent["commands"]
    before = _empty_probe_tree(user)
    outer = _run(
        list(commands["outer_argv"]),
        cwd=commands["outer_cwd"],
        environment=commands["outer_environment"],
        timeout=1800,
    )
    outer_stdout = outer.stdout.decode("utf-8", errors="strict")
    outer_stderr = outer.stderr.decode("utf-8", errors="strict")
    if (
        outer.returncode != commands["expected_outer_returncode"]
        or outer_stdout != ""
        or outer_stderr != EXPECTED_RECORDER_ERROR
    ):
        raise R8FailureSealError("r8 recorder failure signature drifted")
    after_outer = _empty_probe_tree(user)
    inner_environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        **commands["inner_environment"],
    }
    inner = _run(
        list(commands["inner_argv"]),
        cwd=commands["inner_cwd"],
        environment=inner_environment,
        timeout=600,
    )
    inner_stdout = inner.stdout.decode("utf-8", errors="strict")
    inner_stderr = inner.stderr.decode("utf-8", errors="strict")
    if inner.returncode != 1 or inner_stdout != "":
        raise R8FailureSealError("r8 inner r3-probe return signature drifted")
    _validate_inner_stderr(inner_stderr, recovery)
    after_inner = _empty_probe_tree(user)
    payloads = {
        OUTER_STDOUT_NAME: outer.stdout,
        OUTER_STDERR_NAME: outer.stderr,
        INNER_STDOUT_NAME: inner.stdout,
        INNER_STDERR_NAME: inner.stderr,
    }
    refs: dict[str, Any] = {}
    for name, payload in payloads.items():
        path = evidence / name
        _publish(path, payload)
        refs[name] = _file_ref(path, description=f"r8 failure proof {name}")
    return {
        "probe_tree_before": before,
        "probe_tree_after_outer": after_outer,
        "probe_tree_after_inner": after_inner,
        "outer_returncode": outer.returncode,
        "inner_returncode": inner.returncode,
        "outer_stdout": refs[OUTER_STDOUT_NAME],
        "outer_stderr": refs[OUTER_STDERR_NAME],
        "inner_stdout": refs[INNER_STDOUT_NAME],
        "inner_stderr": refs[INNER_STDERR_NAME],
        "envelope_published": False,
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
    empty = _empty_probe_tree(user)
    expected_keys = {
        "probe_tree_before",
        "probe_tree_after_outer",
        "probe_tree_after_inner",
        "outer_returncode",
        "inner_returncode",
        "outer_stdout",
        "outer_stderr",
        "inner_stdout",
        "inner_stderr",
        "envelope_published",
        "known_scheduler_job_ids",
        "result_mutation_count",
    }
    if (
        set(proof) != expected_keys
        or proof.get("probe_tree_before") != empty
        or proof.get("probe_tree_after_outer") != empty
        or proof.get("probe_tree_after_inner") != empty
        or proof.get("outer_returncode") != 2
        or proof.get("inner_returncode") != 1
        or proof.get("envelope_published") is not False
        or proof.get("known_scheduler_job_ids") != []
        or proof.get("result_mutation_count") != 0
    ):
        raise R8FailureSealError("r8 prelaunch failure proof drifted")
    expected_files = {
        "outer_stdout": (OUTER_STDOUT_NAME, b""),
        "outer_stderr": (
            OUTER_STDERR_NAME,
            EXPECTED_RECORDER_ERROR.encode("utf-8"),
        ),
        "inner_stdout": (INNER_STDOUT_NAME, b""),
    }
    for field, (name, expected) in expected_files.items():
        observed = _file_ref(
            evidence / name, description=f"r8 failure proof {name}"
        )
        if proof.get(field) != observed or (evidence / name).read_bytes() != expected:
            raise R8FailureSealError(f"r8 failure proof drifted: {field}")
    inner_ref = _file_ref(
        evidence / INNER_STDERR_NAME,
        description="r8 failure proof inner stderr",
    )
    if proof.get("inner_stderr") != inner_ref:
        raise R8FailureSealError("r8 failure proof inner stderr binding drifted")
    _validate_inner_stderr(
        (evidence / INNER_STDERR_NAME).read_text(encoding="utf-8"),
        recovery,
    )
    if intent.get("commands") != _command_contract(recovery, user):
        raise R8FailureSealError("r8 failure proof command binding drifted")


def _binding(
    evidence: Path, marker: Mapping[str, Any], marker_raw: bytes
) -> dict[str, Any]:
    return {
        "passed": True,
        "root": str(evidence),
        "protocol": PROTOCOL,
        "release_tag": R8_TAG,
        "release_git_commit": R8_COMMIT,
        "chain_namespace": R8_NAMESPACE,
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
    evidence = _safe_directory(evidence_root, description="r8 failure evidence")
    recovery = _safe_directory(recovery_root, description="schema-5 recovery root")
    marker, marker_raw = _read_json(
        evidence / MARKER_NAME, description="r8 prelaunch failure marker"
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
        raise R8FailureSealError("r8 prelaunch failure marker drifted")
    intent, intent_raw = _read_json(
        evidence / INTENT_NAME, description="r8 prelaunch failure intent"
    )
    sealed_user = intent.get("scheduler", {}).get("scheduler_user")
    if scheduler_user is None:
        scheduler_user = sealed_user
    if not isinstance(scheduler_user, str) or scheduler_user != sealed_user:
        raise R8FailureSealError("r8 failure scheduler-user binding drifted")
    _validate_intent(intent, recovery, evidence, scheduler_user)
    if marker.get("intent") != {
        "path": str(evidence / INTENT_NAME),
        "sha256": hashlib.sha256(intent_raw).hexdigest(),
        "size": len(intent_raw),
        "intent_id": intent["intent_id"],
    }:
        raise R8FailureSealError("r8 failure marker intent binding drifted")
    proof = marker.get("proof")
    if not isinstance(proof, dict):
        raise R8FailureSealError("r8 failure marker proof is malformed")
    _verify_proof(evidence, recovery, intent, proof, scheduler_user)
    _helpers._require_recursively_read_only(evidence)
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
        raise R8FailureSealError(f"r8 failure evidence root must be {expected_root}")
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
    intent = _intent_payload(recovery, evidence, scheduler_user)
    if not apply:
        return {
            "action": "would_seal",
            "classification": CLASSIFICATION,
            "commands": intent["commands"],
            "scientific_state": intent["scientific_state"],
            "scheduler": intent["scheduler"],
        }
    if evidence.exists() or evidence.is_symlink():
        raise R8FailureSealError(
            "incomplete r8 failure evidence root requires quarantine"
        )
    evidence.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    evidence.mkdir(mode=0o700)
    intent_path = evidence / INTENT_NAME
    intent_raw = _canonical_bytes(intent)
    _publish(intent_path, intent_raw)
    proof = _execute_proof(evidence, recovery, intent, scheduler_user)
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
        "proof": proof,
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
        R8FailureSealError,
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
