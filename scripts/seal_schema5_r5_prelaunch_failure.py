#!/usr/bin/env python3
"""Seal the deterministic r5 prelaunch CLI-dispatch failure for r6."""

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
PROTOCOL = "schema5-v1.2-r5-prelaunch-cli-dispatch-failure-seal-v1"
CLASSIFICATION = "deterministic_prelaunch_cli_probe_argv_dispatch_collision"
R5_TAG = "sweep-recovery-schema5-v1.2-r5"
R5_NAMESPACE = "schema5-v1.2-r5"
R5_COMMIT = "fa50d619579d6ffde9ab768d342bb65f1b98d844"
R5_TAG_OBJECT = "51267bb1bbe59833bfadc96b4d3ad74355ce1d39"
R5_DURABLE_MARKER = "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R5_COMPLETE.json"
R5_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r5"
R5_TOOLCHAIN_RELATIVE = Path("toolchains/r5/conda")
R3_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r3"
EVIDENCE_RELATIVE_ROOT = Path("prelaunch_failures/schema5-v1.2-r5-cli")
MARKER_NAME = "PRELAUNCH_CLI_FAILURE_SEALED.json"
INTENT_NAME = "PRELAUNCH_CLI_FAILURE_INTENT.json"
STDOUT_NAME = "attempt.stdout"
STDERR_NAME = "attempt.stderr"
DEV_PYTHON = Path(
    "/orcd/home/002/mabdel03/conda_envs/asys_env/bin/python3.11"
)
EXPECTED_EXCEPTION = (
    "AttributeError: 'Namespace' object has no attribute 'evidence_root'"
)
EXPECTED_DISPATCH_LINE = (
    "result = verify_prelaunch_failure_seal(args.evidence_root)"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class R5FailureSealError(RuntimeError):
    """The r5 failure cannot be reproduced, classified, or verified exactly."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _identity(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_directory(path: str | Path, *, description: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    try:
        metadata = lexical.lstat()
    except OSError as exc:
        raise R5FailureSealError(f"missing {description}: {lexical}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise R5FailureSealError(f"{description} is not a real directory: {lexical}")
    if lexical.resolve(strict=True) != lexical:
        raise R5FailureSealError(f"{description} is not canonical: {lexical}")
    return lexical


def _read_json(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise R5FailureSealError(f"cannot read {description}: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise R5FailureSealError(f"{description} is not a regular file: {path}")
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise R5FailureSealError(f"{description} is not canonical JSON: {path}")
    return value, raw


def _file_ref(path: Path, *, description: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise R5FailureSealError(f"cannot stat {description}: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise R5FailureSealError(f"{description} is not a regular file: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
        "link_count": metadata.st_nlink,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(path: Path, raw: bytes, *, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(raw)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise R5FailureSealError(f"short write while publishing {path}")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(checkout), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        },
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise R5FailureSealError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _release_binding(recovery: Path) -> dict[str, Any]:
    checkout = _safe_directory(recovery / R5_CHECKOUT, description="r5 checkout")
    tag_ref = f"refs/tags/{R5_TAG}"
    if (
        _git(checkout, "cat-file", "-t", tag_ref) != "tag"
        or _git(checkout, "rev-parse", tag_ref) != R5_TAG_OBJECT
        or _git(checkout, "rev-parse", f"{tag_ref}^{{commit}}") != R5_COMMIT
        or _git(checkout, "rev-parse", "HEAD") != R5_COMMIT
        or _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
        or (checkout / ".git/objects/info/alternates").exists()
    ):
        raise R5FailureSealError("r5 checkout identity drifted")
    durable_path = recovery / R5_DURABLE_MARKER
    durable, durable_raw = _read_json(
        durable_path, description="r5 durable release marker"
    )
    if (
        durable.get("passed") is not True
        or durable.get("release_tag") != R5_TAG
        or durable.get("chain_namespace") != R5_NAMESPACE
        or durable.get("release_git_commit") != R5_COMMIT
        or durable.get("release_tag_object") != R5_TAG_OBJECT
        or durable.get("remote_commit") != R5_COMMIT
        or durable.get("remote_peeled_commit") != R5_COMMIT
        or durable.get("remote_tag_object") != R5_TAG_OBJECT
    ):
        raise R5FailureSealError("r5 durable release identity drifted")
    sealer = checkout / "scripts/seal_recovery_evidence.py"
    provisioner = checkout / "scripts/provision_schema5_conda_toolchain.py"
    return {
        "checkout": str(checkout),
        "release_tag": R5_TAG,
        "release_git_commit": R5_COMMIT,
        "release_tag_object": R5_TAG_OBJECT,
        "sealer": _file_ref(sealer, description="r5 recovery sealer"),
        "provisioner": _file_ref(provisioner, description="r5 provisioner"),
        "durable_marker": {
            "path": str(durable_path),
            "sha256": hashlib.sha256(durable_raw).hexdigest(),
            "size": len(durable_raw),
            "marker_id": durable.get("marker_id"),
        },
    }


def _toolchain_binding(recovery: Path) -> dict[str, Any]:
    checkout = recovery / R5_CHECKOUT
    provisioner = checkout / "scripts/provision_schema5_conda_toolchain.py"
    runtime_identity = checkout / "scripts/schema5_conda_runtime_identity.py"
    toolchain = recovery / R5_TOOLCHAIN_RELATIVE
    try:
        saved = {
            name: sys.modules.get(name)
            for name in (
                "scripts",
                "scripts.schema5_conda_runtime_identity",
                "scripts.provision_schema5_conda_toolchain",
            )
        }
        package = types.ModuleType("scripts")
        package.__path__ = []
        sys.modules["scripts"] = package
        runtime_spec = importlib.util.spec_from_file_location(
            "scripts.schema5_conda_runtime_identity",
            runtime_identity,
        )
        if runtime_spec is None or runtime_spec.loader is None:
            raise R5FailureSealError("cannot load r5 runtime-identity verifier")
        runtime_module = importlib.util.module_from_spec(runtime_spec)
        sys.modules[runtime_spec.name] = runtime_module
        runtime_spec.loader.exec_module(runtime_module)
        provisioner_spec = importlib.util.spec_from_file_location(
            "scripts.provision_schema5_conda_toolchain",
            provisioner,
        )
        if provisioner_spec is None or provisioner_spec.loader is None:
            raise R5FailureSealError("cannot load r5 toolchain verifier")
        provisioner_module = importlib.util.module_from_spec(provisioner_spec)
        sys.modules[provisioner_spec.name] = provisioner_module
        provisioner_spec.loader.exec_module(provisioner_module)
        report = provisioner_module.verify_conda_toolchain(toolchain)
    except (OSError, ValueError) as exc:
        raise R5FailureSealError(
            f"r5 sealed toolchain verification failed: {exc}"
        ) from exc
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    if (
        not isinstance(report, dict)
        or report.get("protocol")
        != "schema5-v1.2-r5-offline-conda-toolchain-v1"
        or report.get("release_tag") != R5_TAG
        or report.get("chain_namespace") != R5_NAMESPACE
        or report.get("sealed_read_only") is not True
        or report.get("toolchain_root") != str(toolchain)
    ):
        raise R5FailureSealError("r5 sealed toolchain binding drifted")
    canonical = _canonical_bytes(report)
    marker = toolchain / "CONDA_TOOLCHAIN_COMPLETE.json"
    return {
        "root": str(toolchain),
        "verification_sha256": hashlib.sha256(canonical).hexdigest(),
        "marker": _file_ref(marker, description="r5 toolchain marker"),
        "runtime_identity_sha256": report["runtime_identity"]["first"][
            "identity_sha256"
        ],
        "prefix_inventory_sha256": report["complete_prefix_inventory"]["after"][
            "inventory_sha256"
        ],
    }


def _probe_tree_inventory(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"exists": False, "entries": [], "inventory_sha256": hashlib.sha256(
            _canonical_bytes([])
        ).hexdigest()}
    root = _safe_directory(root, description="r5 failed probe root")
    rows: list[dict[str, Any]] = []
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in sorted([*names, *files]):
            path = base / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                rows.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": os.readlink(path),
                    }
                )
            elif stat.S_ISDIR(metadata.st_mode):
                rows.append(
                    {
                        "path": relative,
                        "type": "directory",
                        "mode": stat.S_IMODE(metadata.st_mode),
                    }
                )
            elif stat.S_ISREG(metadata.st_mode):
                rows.append(
                    {
                        "path": relative,
                        "type": "file",
                        "mode": stat.S_IMODE(metadata.st_mode),
                        "size": metadata.st_size,
                        "sha256": _sha256(path),
                    }
                )
            else:
                raise R5FailureSealError(f"unsafe entry in r5 probe root: {path}")
    rows.sort(key=lambda row: (row["path"], row["type"]))
    return {
        "exists": True,
        "entries": rows,
        "inventory_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def _command_contract(recovery: Path, scheduler_user: str) -> dict[str, Any]:
    r5_checkout = recovery / R5_CHECKOUT
    r3_checkout = recovery / R3_CHECKOUT
    toolchain = recovery / R5_TOOLCHAIN_RELATIVE
    probe_root = Path(f"/tmp/schema5-r3-prelaunch-{scheduler_user}-r5")
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
        str(r5_checkout / "scripts/seal_recovery_evidence.py"),
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
        "tagged-r5-release-checkout",
        str(r5_checkout),
        "--input-root",
        "sealed-r5-conda-toolchain",
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
        "cwd": str(
            Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling")
        ),
        "probe_root": str(probe_root),
        "expected_envelope": str(envelope),
        "environment": {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    }


def _validate_original_probe_tree_snapshot(value: Any) -> dict[str, Any]:
    expected_entries = [
        {
            "path": "empty-conda-pkgs",
            "type": "directory",
            "mode": 0o700,
        }
    ]
    if (
        not isinstance(value, dict)
        or value.get("exists") is not True
        or value.get("entries") != expected_entries
        or value.get("inventory_sha256")
        != hashlib.sha256(_canonical_bytes(expected_entries)).hexdigest()
    ):
        raise R5FailureSealError(
            "original failed r5 probe tree is not the exact empty-cache prefix"
        )
    return json.loads(json.dumps(value, sort_keys=True))


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
        raise R5FailureSealError(f"squeue failed: {completed.stderr.strip()}")
    matching = [
        row
        for row in completed.stdout.splitlines()
        if "s5v12r5" in row
        or "schema5-v1.2-r5" in row
        or "sweep-recovery-schema5-v1.2-r5" in row
    ]
    if matching:
        raise R5FailureSealError(f"r5 scheduler jobs remain live: {matching}")
    return {"scheduler_user": user, "matching_r5_jobs": []}


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
            recovery / "slurm_canaries/schema5-v1.2-r5",
            recovery / "materialization_pilots/schema5-v1.2-r5",
            recovery / "PROTECTED_CAPACITY_COMPLETE.json",
            recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R5.json",
            recovery / "prelaunch_failures/schema5-v1.2-r3",
        )
    ]


def _scientific_state(recovery: Path) -> dict[str, Any]:
    historical = _historical_scientific_paths(recovery)
    present = [
        path for path in historical if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise R5FailureSealError(
            f"r5 failure is not prelaunch zero-result evidence: {present}"
        )
    return {
        "captured_before_r6_outputs": True,
        "result_mutation_count": 0,
        "scheduler_job_count": 0,
        "required_absent_paths_at_seal": historical,
        "permanently_absent_r5_execution_paths": [
            str(recovery / "slurm_canaries/schema5-v1.2-r5"),
            str(recovery / "materialization_pilots/schema5-v1.2-r5"),
            str(recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R5.json"),
        ],
    }


def _validate_permanent_absence(
    recovery: Path, state: Mapping[str, Any]
) -> None:
    expected = _scientific_state_paths(recovery)
    if state.get("permanently_absent_r5_execution_paths") != expected:
        raise R5FailureSealError("sealed r5 permanent-absence contract drifted")
    present = [
        path for path in expected if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise R5FailureSealError(f"r5 execution namespace appeared later: {present}")


def _scientific_state_paths(recovery: Path) -> list[str]:
    return [
        str(recovery / "slurm_canaries/schema5-v1.2-r5"),
        str(recovery / "materialization_pilots/schema5-v1.2-r5"),
        str(recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R5.json"),
    ]


def _intent_payload(recovery: Path, user: str) -> dict[str, Any]:
    command = _command_contract(recovery, user)
    original_probe_tree = _validate_original_probe_tree_snapshot(
        _probe_tree_inventory(Path(command["probe_root"]))
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": f"{PROTOCOL}-intent",
        "classification": CLASSIFICATION,
        "source_release": _release_binding(recovery),
        "sealed_r5_toolchain": _toolchain_binding(recovery),
        "command": command,
        "original_probe_tree": original_probe_tree,
        "scientific_state": _scientific_state(recovery),
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
        or intent.get("sealed_r5_toolchain") != _toolchain_binding(recovery)
        or intent.get("command") != _command_contract(recovery, user)
        or _validate_original_probe_tree_snapshot(
            intent.get("original_probe_tree")
        )
        != intent.get("original_probe_tree")
        or intent.get("scheduler") != _scheduler_quiescence(user)
    ):
        raise R5FailureSealError("r5 failure-seal intent drifted")
    state = intent.get("scientific_state")
    if not isinstance(state, dict):
        raise R5FailureSealError("r5 failure-seal intent lacks scientific state")
    if (
        state.get("captured_before_r6_outputs") is not True
        or state.get("result_mutation_count") != 0
        or state.get("scheduler_job_count") != 0
        or state.get("required_absent_paths_at_seal")
        != _historical_scientific_paths(recovery)
    ):
        raise R5FailureSealError("r5 prelaunch scientific-state evidence drifted")
    _validate_permanent_absence(recovery, state)


def _binding(
    evidence: Path, marker: Mapping[str, Any], marker_raw: bytes
) -> dict[str, Any]:
    return {
        "passed": True,
        "root": str(evidence),
        "protocol": PROTOCOL,
        "release_tag": R5_TAG,
        "release_git_commit": R5_COMMIT,
        "chain_namespace": R5_NAMESPACE,
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
    evidence = _safe_directory(evidence_root, description="r5 failure evidence")
    recovery = _safe_directory(recovery_root, description="schema-5 recovery root")
    marker, marker_raw = _read_json(
        evidence / MARKER_NAME, description="r5 prelaunch failure marker"
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
        raise R5FailureSealError("r5 prelaunch failure marker drifted")
    intent, intent_raw = _read_json(
        evidence / INTENT_NAME, description="r5 prelaunch failure intent"
    )
    sealed_user = intent.get("scheduler", {}).get("scheduler_user")
    if scheduler_user is None:
        scheduler_user = sealed_user
    if not isinstance(scheduler_user, str) or scheduler_user != sealed_user:
        raise R5FailureSealError("r5 failure scheduler-user binding drifted")
    _validate_intent(intent, recovery, scheduler_user)
    if marker.get("intent") != {
        "path": str(evidence / INTENT_NAME),
        "sha256": hashlib.sha256(intent_raw).hexdigest(),
        "size": len(intent_raw),
        "intent_id": intent["intent_id"],
    }:
        raise R5FailureSealError("r5 failure marker intent binding drifted")
    stdout_ref = _file_ref(evidence / STDOUT_NAME, description="r5 failure stdout")
    stderr_ref = _file_ref(evidence / STDERR_NAME, description="r5 failure stderr")
    stderr = (evidence / STDERR_NAME).read_text(encoding="utf-8")
    attempt = marker.get("attempt")
    if not isinstance(attempt, dict):
        raise R5FailureSealError("r5 CLI failure attempt binding is malformed")
    if (
        attempt
        != {
            "returncode": 1,
            "stdout": stdout_ref,
            "stderr": stderr_ref,
            "expected_exception": EXPECTED_EXCEPTION,
            "expected_dispatch_line": EXPECTED_DISPATCH_LINE,
            "probe_tree_before": attempt.get("probe_tree_before"),
            "probe_tree_after": attempt.get("probe_tree_after"),
            "expected_envelope_absent": True,
        }
        or stdout_ref["size"] != 0
        or EXPECTED_EXCEPTION not in stderr
        or EXPECTED_DISPATCH_LINE not in stderr
        or attempt.get("probe_tree_before") != intent.get("original_probe_tree")
        or attempt.get("probe_tree_before") != attempt.get("probe_tree_after")
    ):
        raise R5FailureSealError("r5 CLI failure signature drifted")
    command = intent["command"]
    if Path(command["expected_envelope"]).exists():
        raise R5FailureSealError("r5 failed CLI unexpectedly produced an envelope")
    for path in (evidence, evidence / INTENT_NAME, evidence / STDOUT_NAME,
                 evidence / STDERR_NAME, evidence / MARKER_NAME):
        if stat.S_IMODE(path.lstat().st_mode) & 0o222:
            raise R5FailureSealError(f"r5 failure evidence is writable: {path}")
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
        raise R5FailureSealError(f"r5 failure evidence root must be {expected_root}")
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
            "command": intent["command"],
            "scientific_state": intent["scientific_state"],
            "scheduler": intent["scheduler"],
        }
    if evidence.exists() or evidence.is_symlink():
        raise R5FailureSealError(
            "incomplete r5 failure evidence root requires quarantine"
        )
    evidence.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    evidence.mkdir(mode=0o700)
    intent_path = evidence / INTENT_NAME
    intent_raw = _canonical_bytes(intent)
    _publish(intent_path, intent_raw)
    contract = intent["command"]
    before = _probe_tree_inventory(Path(contract["probe_root"]))
    if before != intent["original_probe_tree"]:
        raise R5FailureSealError("r5 failed probe tree changed before reproduction")
    completed = subprocess.run(
        contract["argv"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=contract["cwd"],
        env=contract["environment"],
        timeout=120,
        check=False,
    )
    after = _probe_tree_inventory(Path(contract["probe_root"]))
    if (
        completed.returncode != 1
        or completed.stdout
        or before != after
        or EXPECTED_EXCEPTION.encode() not in completed.stderr
        or EXPECTED_DISPATCH_LINE.encode() not in completed.stderr
        or Path(contract["expected_envelope"]).exists()
    ):
        raise R5FailureSealError("r5 CLI failure reproduction drifted")
    _publish(evidence / STDOUT_NAME, completed.stdout)
    _publish(evidence / STDERR_NAME, completed.stderr)
    stdout_ref = _file_ref(evidence / STDOUT_NAME, description="r5 failure stdout")
    stderr_ref = _file_ref(evidence / STDERR_NAME, description="r5 failure stderr")
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
            "expected_exception": EXPECTED_EXCEPTION,
            "expected_dispatch_line": EXPECTED_DISPATCH_LINE,
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
    except (R5FailureSealError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
