#!/usr/bin/env python3
"""Seal r7's deterministic prelaunch Git index-refresh failure for r8."""

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
PROTOCOL = "schema5-v1.2-r7-prelaunch-git-index-refresh-failure-seal-v1"
CLASSIFICATION = "deterministic_prelaunch_git_optional_index_refresh"
R7_TAG = "sweep-recovery-schema5-v1.2-r7"
R7_NAMESPACE = "schema5-v1.2-r7"
R7_COMMIT = "acc0beb98cfbb5017d5d5b5e60567aa43229e248"
R7_TAG_OBJECT = "95f8f3852498c4ab8e3026f29c183ecdd58678f4"
R7_DURABLE_MARKER = "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R7_COMPLETE.json"
R7_CHECKOUT = "materialization_pilot_source_checkout_v1_2_r7"
R7_TOOLCHAIN_RELATIVE = Path("toolchains/r7/conda")
EVIDENCE_RELATIVE_ROOT = Path(
    "prelaunch_failures/schema5-v1.2-r7-git-index-refresh"
)
MARKER_NAME = "PRELAUNCH_GIT_INDEX_REFRESH_FAILURE_SEALED.json"
INTENT_NAME = "PRELAUNCH_GIT_INDEX_REFRESH_FAILURE_INTENT.json"
INDEX_BEFORE_NAME = "git-index.before"
INDEX_AFTER_NAME = "git-index.after"
REPRODUCTION_CHECKOUT_NAME = "reproduction_checkout"
EXPECTED_RECORDER_ERROR = (
    "r3 prelaunch immutable-input verification mutated a probe input"
)
R7_PROBE_ENVELOPE_NAMES = (
    "FAILURE_UNSAFE_RECORDED_BROKEN_INTERNAL_SYMLINK.source.json",
    "FAILURE_OFFLINE_CLONE_UNSEEDED_RELEASE_LOCAL_CACHE.source.json",
)
_CHUNK_SIZE = 8 * 1024 * 1024


class R7FailureSealError(RuntimeError):
    """The r7 Git-index failure cannot be sealed or verified exactly."""


def _load_helpers():
    path = Path(__file__).resolve().with_name(
        "seal_schema5_r6_prelaunch_failure.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_schema5_r6_failure_helpers", path
    )
    if spec is None or spec.loader is None:
        raise R7FailureSealError("cannot load marker-last evidence helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_helpers()
MarkerLastEvidenceError = (
    _helpers.R6FailureSealError,
    _helpers.MarkerLastEvidenceError,
)
_canonical_bytes = _helpers._canonical_bytes
_identity = _helpers._identity
_safe_directory = _helpers._safe_directory
_read_json = _helpers._read_json
_file_ref = _helpers._file_ref
_publish = _helpers._publish
_fsync_directory = _helpers._fsync_directory


def _git_environment(*, optional_locks: bool) -> dict[str, str]:
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    if not optional_locks:
        environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def _run_git(
    checkout: Path,
    *arguments: str,
    optional_locks: bool = False,
    timeout: int = 120,
) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(checkout), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_git_environment(optional_locks=optional_locks),
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise R7FailureSealError(
            f"git {' '.join(arguments)} failed: {detail[:1000]}"
        )
    return completed.stdout.strip()


def _release_binding(recovery: Path) -> dict[str, Any]:
    checkout = _safe_directory(recovery / R7_CHECKOUT, description="r7 checkout")
    tag_ref = f"refs/tags/{R7_TAG}"
    if (
        _run_git(checkout, "cat-file", "-t", tag_ref) != "tag"
        or _run_git(checkout, "rev-parse", tag_ref) != R7_TAG_OBJECT
        or _run_git(checkout, "rev-parse", f"{tag_ref}^{{commit}}") != R7_COMMIT
        or _run_git(checkout, "rev-parse", "HEAD") != R7_COMMIT
        or _run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
        or (checkout / ".git/objects/info/alternates").exists()
    ):
        raise R7FailureSealError("r7 checkout identity drifted")
    durable_path = recovery / R7_DURABLE_MARKER
    durable, durable_raw = _read_json(
        durable_path, description="r7 durable release marker"
    )
    bundle_path = Path(str(durable.get("bundle_path", "")))
    if (
        durable.get("passed") is not True
        or durable.get("release_tag") != R7_TAG
        or durable.get("chain_namespace") != R7_NAMESPACE
        or durable.get("release_git_commit") != R7_COMMIT
        or durable.get("release_tag_object") != R7_TAG_OBJECT
        or durable.get("remote_commit") != R7_COMMIT
        or durable.get("remote_peeled_commit") != R7_COMMIT
        or durable.get("remote_tag_object") != R7_TAG_OBJECT
        or not bundle_path.is_file()
        or bundle_path.is_symlink()
        or _file_ref(bundle_path, description="r7 durable Git bundle")["sha256"]
        != durable.get("bundle_sha256")
    ):
        raise R7FailureSealError("r7 durable release identity drifted")
    return {
        "checkout": str(checkout),
        "release_tag": R7_TAG,
        "release_git_commit": R7_COMMIT,
        "release_tag_object": R7_TAG_OBJECT,
        "recorder": _file_ref(
            checkout / "scripts/seal_recovery_evidence.py",
            description="r7 recovery-evidence recorder",
        ),
        "provisioner": _file_ref(
            checkout / "scripts/provision_schema5_conda_toolchain.py",
            description="r7 Conda provisioner",
        ),
        "durable_marker": {
            "path": str(durable_path),
            "sha256": hashlib.sha256(durable_raw).hexdigest(),
            "size": len(durable_raw),
            "marker_id": durable.get("marker_id"),
        },
        "durable_bundle": _file_ref(
            bundle_path, description="r7 durable Git bundle"
        ),
    }


def _toolchain_binding(recovery: Path) -> dict[str, Any]:
    checkout = recovery / R7_CHECKOUT
    provisioner = checkout / "scripts/provision_schema5_conda_toolchain.py"
    runtime_identity = checkout / "scripts/schema5_conda_runtime_identity.py"
    toolchain = recovery / R7_TOOLCHAIN_RELATIVE
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
            raise R7FailureSealError("cannot load r7 runtime-identity verifier")
        runtime_module = importlib.util.module_from_spec(runtime_spec)
        sys.modules[runtime_spec.name] = runtime_module
        runtime_spec.loader.exec_module(runtime_module)
        provisioner_spec = importlib.util.spec_from_file_location(
            "scripts.provision_schema5_conda_toolchain", provisioner
        )
        if provisioner_spec is None or provisioner_spec.loader is None:
            raise R7FailureSealError("cannot load r7 toolchain verifier")
        provisioner_module = importlib.util.module_from_spec(provisioner_spec)
        sys.modules[provisioner_spec.name] = provisioner_module
        provisioner_spec.loader.exec_module(provisioner_module)
        binding = provisioner_module.verified_conda_toolchain_binding(
            toolchain, exercise=False
        )
    except Exception as exc:
        if isinstance(exc, R7FailureSealError):
            raise
        raise R7FailureSealError(
            f"r7 sealed toolchain verification failed: {exc}"
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
        != "schema5-v1.2-r7-offline-conda-toolchain-v1"
        or binding.get("release_tag") != R7_TAG
        or binding.get("chain_namespace") != R7_NAMESPACE
        or binding.get("toolchain_root") != str(toolchain)
        or binding.get("portable_shebang", {}).get("interpreter")
        != str(toolchain / "base/bin/python")
    ):
        raise R7FailureSealError("r7 sealed toolchain binding drifted")
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
        raise R7FailureSealError(f"squeue failed: {completed.stderr.strip()}")
    matching = [
        row
        for row in completed.stdout.splitlines()
        if "s5v12r7" in row
        or "schema5-v1.2-r7" in row
        or "sweep-recovery-schema5-v1.2-r7" in row
    ]
    if matching:
        raise R7FailureSealError(f"r7 scheduler jobs remain live: {matching}")
    return {"scheduler_user": user, "matching_r7_jobs": []}


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
            recovery / "slurm_canaries/schema5-v1.2-r7",
            recovery / "materialization_pilots/schema5-v1.2-r7",
            recovery / "PROTECTED_CAPACITY_COMPLETE.json",
            recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R7.json",
            recovery / "prelaunch_failures/schema5-v1.2-r3",
        )
    ]


def _r7_probe_root(user: str) -> Path:
    return Path(f"/tmp/schema5-r3-prelaunch-{user}-r7")


def _empty_r7_probe_tree(user: str) -> dict[str, Any]:
    root = _safe_directory(
        _r7_probe_root(user), description="original failed r7 probe root"
    )
    entries = sorted(path.name for path in root.iterdir())
    if entries:
        raise R7FailureSealError(
            f"original failed r7 probe root is not empty: {entries}"
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
    probe_root = _r7_probe_root(user)
    return [
        str(recovery / "slurm_canaries/schema5-v1.2-r7"),
        str(recovery / "materialization_pilots/schema5-v1.2-r7"),
        str(recovery / "RECOVERY_CHAIN_SCHEMA5_V1_2_R7.json"),
        *(str(probe_root / name) for name in R7_PROBE_ENVELOPE_NAMES),
    ]


def _scientific_state(recovery: Path, user: str) -> dict[str, Any]:
    historical = _historical_scientific_paths(recovery)
    present = [
        path for path in historical if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise R7FailureSealError(
            f"r7 failure is not prelaunch zero-result evidence: {present}"
        )
    return {
        "captured_before_r8_outputs": True,
        "result_mutation_count": 0,
        "scheduler_job_count": 0,
        "required_absent_paths_at_seal": historical,
        "original_r7_probe_tree": _empty_r7_probe_tree(user),
        "permanently_absent_r7_execution_paths": _permanently_absent_paths(
            recovery, user
        ),
    }


def _validate_permanent_absence(
    recovery: Path, user: str, state: Mapping[str, Any]
) -> None:
    expected = _permanently_absent_paths(recovery, user)
    if state.get("permanently_absent_r7_execution_paths") != expected:
        raise R7FailureSealError("sealed r7 permanent-absence contract drifted")
    present = [
        path for path in expected if Path(path).exists() or Path(path).is_symlink()
    ]
    if present:
        raise R7FailureSealError(f"r7 execution namespace appeared later: {present}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _index_observation(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise R7FailureSealError(f"Git index is not a regular file: {path}")
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": stat.S_IMODE(metadata.st_mode),
        "link_count": metadata.st_nlink,
        "size": metadata.st_size,
        "sha256": _sha256(path),
    }


def _tree_inventory(root: Path, *, exclude_index: bool) -> dict[str, Any]:
    root = _safe_directory(root, description="r7 proof checkout")
    rows: list[dict[str, Any]] = []
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in sorted([*names, *files]):
            path = base / name
            relative = path.relative_to(root).as_posix()
            if exclude_index and relative == ".git/index":
                continue
            metadata = path.lstat()
            common = {
                "path": relative,
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "mode": stat.S_IMODE(metadata.st_mode),
                "link_count": metadata.st_nlink,
            }
            if stat.S_ISLNK(metadata.st_mode):
                rows.append(
                    common
                    | {
                        "type": "symlink",
                        "target": os.readlink(path),
                    }
                )
            elif stat.S_ISDIR(metadata.st_mode):
                rows.append(common | {"type": "directory"})
            elif stat.S_ISREG(metadata.st_mode):
                rows.append(
                    common
                    | {
                        "type": "file",
                        "size": metadata.st_size,
                        "sha256": _sha256(path),
                    }
                )
            else:
                raise R7FailureSealError(f"unsafe proof-checkout entry: {path}")
    rows.sort(key=lambda row: (row["path"], row["type"]))
    return {
        "entry_count": len(rows),
        "inventory_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def _copy_preimage(source: Path, destination: Path) -> None:
    data = source.read_bytes()
    _publish(destination, data)


def _recursively_read_only(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            path = base / name
            if not path.is_symlink():
                os.chmod(path, stat.S_IMODE(path.lstat().st_mode) & ~0o222)
        for name in names:
            path = base / name
            if not path.is_symlink():
                os.chmod(path, stat.S_IMODE(path.lstat().st_mode) & ~0o222)
        os.chmod(base, stat.S_IMODE(base.lstat().st_mode) & ~0o222)


def _require_recursively_read_only(root: Path) -> None:
    for directory, names, files in os.walk(root, followlinks=False):
        for path in [Path(directory), *[Path(directory) / name for name in names + files]]:
            if not path.is_symlink() and stat.S_IMODE(path.lstat().st_mode) & 0o222:
                raise R7FailureSealError(f"r7 failure evidence is writable: {path}")


def _proof_contract(evidence: Path, release: Mapping[str, Any]) -> dict[str, Any]:
    checkout = evidence / REPRODUCTION_CHECKOUT_NAME
    return {
        "reproduction_checkout": str(checkout),
        "durable_bundle": release["durable_bundle"],
        "clone_argv": [
            "/usr/bin/git",
            "clone",
            "--no-local",
            "--no-checkout",
            release["durable_bundle"]["path"],
            str(checkout),
        ],
        "checkout_argv": [
            "/usr/bin/git",
            "-C",
            str(checkout),
            "checkout",
            "--detach",
            R7_COMMIT,
        ],
        "identity_queries": [
            ["cat-file", "-t", f"refs/tags/{R7_TAG}"],
            ["rev-parse", "--verify", "HEAD"],
            ["rev-parse", "--verify", f"refs/tags/{R7_TAG}^{{commit}}"],
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ],
        "r7_git_environment": _git_environment(optional_locks=True),
        "r8_corrected_environment": _git_environment(optional_locks=False),
        "expected_results": ["tag", R7_COMMIT, R7_COMMIT, ""],
        "expected_recorder_error": EXPECTED_RECORDER_ERROR,
        "expected_mutation": {
            "path": ".git/index",
            "same_device": True,
            "same_bytes": True,
            "same_size": True,
            "inode_replaced": True,
            "all_other_checkout_entries_unchanged": True,
        },
    }


def _intent_payload(recovery: Path, evidence: Path, user: str) -> dict[str, Any]:
    release = _release_binding(recovery)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": f"{PROTOCOL}-intent",
        "classification": CLASSIFICATION,
        "source_release": release,
        "sealed_r7_toolchain": _toolchain_binding(recovery),
        "proof": _proof_contract(evidence, release),
        "scientific_state": _scientific_state(recovery, user),
        "scheduler": _scheduler_quiescence(user),
    }
    payload["intent_id"] = _identity(payload, "intent_id")
    return payload


def _validate_intent(
    intent: Mapping[str, Any], recovery: Path, evidence: Path, user: str
) -> None:
    release = _release_binding(recovery)
    if (
        intent.get("schema_version") != SCHEMA_VERSION
        or intent.get("protocol") != f"{PROTOCOL}-intent"
        or intent.get("classification") != CLASSIFICATION
        or intent.get("intent_id") != _identity(intent, "intent_id")
        or intent.get("source_release") != release
        or intent.get("sealed_r7_toolchain") != _toolchain_binding(recovery)
        or intent.get("proof") != _proof_contract(evidence, release)
        or intent.get("scheduler") != _scheduler_quiescence(user)
    ):
        raise R7FailureSealError("r7 failure-seal intent drifted")
    state = intent.get("scientific_state")
    if (
        not isinstance(state, dict)
        or state.get("captured_before_r8_outputs") is not True
        or state.get("result_mutation_count") != 0
        or state.get("scheduler_job_count") != 0
        or state.get("required_absent_paths_at_seal")
        != _historical_scientific_paths(recovery)
        or state.get("original_r7_probe_tree") != _empty_r7_probe_tree(user)
    ):
        raise R7FailureSealError("r7 prelaunch scientific-state evidence drifted")
    _validate_permanent_absence(recovery, user, state)


def _binding(
    evidence: Path, marker: Mapping[str, Any], marker_raw: bytes
) -> dict[str, Any]:
    return {
        "passed": True,
        "root": str(evidence),
        "protocol": PROTOCOL,
        "release_tag": R7_TAG,
        "release_git_commit": R7_COMMIT,
        "chain_namespace": R7_NAMESPACE,
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


def _verify_proof(
    evidence: Path, intent: Mapping[str, Any], proof: Mapping[str, Any]
) -> None:
    checkout = _safe_directory(
        evidence / REPRODUCTION_CHECKOUT_NAME,
        description="sealed r7 index-refresh proof checkout",
    )
    before_copy = _file_ref(
        evidence / INDEX_BEFORE_NAME, description="r7 index preimage"
    )
    after_copy = _file_ref(
        evidence / INDEX_AFTER_NAME, description="r7 index postimage"
    )
    before = proof.get("index_before")
    after = proof.get("index_after")
    if (
        proof.get("git_results")
        != intent["proof"]["expected_results"]
        or proof.get("inventory_before_excluding_index")
        != proof.get("inventory_after_excluding_index")
        or not isinstance(before, dict)
        or not isinstance(after, dict)
        or before.get("device") != after.get("device")
        or before.get("inode") == after.get("inode")
        or before.get("size") != after.get("size")
        or before.get("sha256") != after.get("sha256")
        or before_copy["sha256"] != before.get("sha256")
        or after_copy["sha256"] != after.get("sha256")
        or before_copy["size"] != before.get("size")
        or after_copy["size"] != after.get("size")
        or proof.get("corrected_query_preserved_index") is not True
    ):
        raise R7FailureSealError("r7 Git-index mutation proof drifted")
    live_index = _index_observation(checkout / ".git/index")
    if (
        live_index["device"] != after["device"]
        or live_index["inode"] != after["inode"]
        or live_index["size"] != after["size"]
        or live_index["sha256"] != after["sha256"]
        or _tree_inventory(checkout, exclude_index=False)
        != proof.get("sealed_checkout_inventory")
    ):
        raise R7FailureSealError("sealed r7 proof checkout drifted")
    index_before_query = _index_observation(checkout / ".git/index")
    if _run_git(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        optional_locks=False,
    ):
        raise R7FailureSealError("sealed r7 proof checkout is dirty")
    index_after_query = _index_observation(checkout / ".git/index")
    if index_before_query != index_after_query:
        raise R7FailureSealError("corrected Git identity query mutated its index")
    if (
        _run_git(checkout, "cat-file", "-t", f"refs/tags/{R7_TAG}") != "tag"
        or _run_git(checkout, "rev-parse", f"refs/tags/{R7_TAG}") != R7_TAG_OBJECT
        or _run_git(checkout, "rev-parse", "HEAD") != R7_COMMIT
        or (checkout / ".git/objects/info/alternates").exists()
    ):
        raise R7FailureSealError("sealed r7 proof checkout identity drifted")


def verify_failure_seal(
    evidence_root: str | Path,
    *,
    recovery_root: str | Path,
    scheduler_user: str | None = None,
) -> dict[str, Any]:
    evidence = _safe_directory(evidence_root, description="r7 failure evidence")
    recovery = _safe_directory(recovery_root, description="schema-5 recovery root")
    marker, marker_raw = _read_json(
        evidence / MARKER_NAME, description="r7 prelaunch failure marker"
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
        raise R7FailureSealError("r7 prelaunch failure marker drifted")
    intent, intent_raw = _read_json(
        evidence / INTENT_NAME, description="r7 prelaunch failure intent"
    )
    sealed_user = intent.get("scheduler", {}).get("scheduler_user")
    if scheduler_user is None:
        scheduler_user = sealed_user
    if not isinstance(scheduler_user, str) or scheduler_user != sealed_user:
        raise R7FailureSealError("r7 failure scheduler-user binding drifted")
    _validate_intent(intent, recovery, evidence, scheduler_user)
    if marker.get("intent") != {
        "path": str(evidence / INTENT_NAME),
        "sha256": hashlib.sha256(intent_raw).hexdigest(),
        "size": len(intent_raw),
        "intent_id": intent["intent_id"],
    }:
        raise R7FailureSealError("r7 failure marker intent binding drifted")
    proof = marker.get("proof")
    if not isinstance(proof, dict):
        raise R7FailureSealError("r7 failure marker proof is malformed")
    _verify_proof(evidence, intent, proof)
    _require_recursively_read_only(evidence)
    return _binding(evidence, marker, marker_raw)


def _execute_proof(
    evidence: Path, intent: Mapping[str, Any]
) -> dict[str, Any]:
    contract = intent["proof"]
    checkout = evidence / REPRODUCTION_CHECKOUT_NAME
    for argv in (contract["clone_argv"], contract["checkout_argv"]):
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_git_environment(optional_locks=False),
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            raise R7FailureSealError(
                f"r7 proof setup failed: {completed.stderr.strip()[:1000]}"
            )
    if (checkout / ".git/objects/info/alternates").exists():
        raise R7FailureSealError("r7 proof checkout has object alternates")
    before_inventory = _tree_inventory(checkout, exclude_index=True)
    index = checkout / ".git/index"
    before = _index_observation(index)
    _copy_preimage(index, evidence / INDEX_BEFORE_NAME)
    results = [
        _run_git(checkout, *arguments, optional_locks=True)
        for arguments in contract["identity_queries"]
    ]
    after = _index_observation(index)
    after_inventory = _tree_inventory(checkout, exclude_index=True)
    _copy_preimage(index, evidence / INDEX_AFTER_NAME)
    if (
        results != contract["expected_results"]
        or before_inventory != after_inventory
        or before["device"] != after["device"]
        or before["inode"] == after["inode"]
        or before["size"] != after["size"]
        or before["sha256"] != after["sha256"]
    ):
        raise R7FailureSealError("r7 optional-index-refresh reproduction drifted")
    corrected_before = _index_observation(index)
    if _run_git(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        optional_locks=False,
    ):
        raise R7FailureSealError("r7 proof checkout is dirty")
    corrected_after = _index_observation(index)
    if corrected_before != corrected_after:
        raise R7FailureSealError("r8 corrected Git query mutated the index")
    _recursively_read_only(checkout)
    return {
        "git_results": results,
        "index_before": before,
        "index_after": after,
        "inventory_before_excluding_index": before_inventory,
        "inventory_after_excluding_index": after_inventory,
        "corrected_query_preserved_index": True,
        "sealed_checkout_inventory": _tree_inventory(
            checkout, exclude_index=False
        ),
        "expected_recorder_error": EXPECTED_RECORDER_ERROR,
    }


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
        raise R7FailureSealError(f"r7 failure evidence root must be {expected_root}")
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
            "proof": intent["proof"],
            "scientific_state": intent["scientific_state"],
            "scheduler": intent["scheduler"],
        }
    if evidence.exists() or evidence.is_symlink():
        raise R7FailureSealError(
            "incomplete r7 failure evidence root requires quarantine"
        )
    evidence.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    evidence.mkdir(mode=0o700)
    intent_path = evidence / INTENT_NAME
    intent_raw = _canonical_bytes(intent)
    _publish(intent_path, intent_raw)
    proof = _execute_proof(evidence, intent)
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
        R7FailureSealError,
        *MarkerLastEvidenceError,
        OSError,
        UnicodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
