#!/usr/bin/env python3
"""Seal the deterministic r4 overlong-prefix Conda failure before r5 launch."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping


SCHEMA_VERSION = 1
PROTOCOL = "schema5-v1.2-r4-overlong-conda-prefix-failure-seal-v1"
CLASSIFICATION = (
    "deterministic_overlong_conda_prefix_failure_sealed_fail_closed"
)
R4_TAG = "sweep-recovery-schema5-v1.2-r4"
R4_NAMESPACE = "schema5-v1.2-r4"
R4_COMMIT = "f23cf1c4b2d2bf606afd013123b9afe9c614bb6c"
R4_TOOLCHAIN_DIRECTORY = "conda-toolchain-miniforge3-25.11.0-1"
R4_TRANSACTION_DIRECTORY = f".{R4_TOOLCHAIN_DIRECTORY}.provisioning"
R4_TOOLCHAIN_PROTOCOL = "schema5-v1.2-r4-offline-conda-toolchain-v1"
MARKER_NAME = "TOOLCHAIN_FAILURE_SEALED.json"
EXPECTED_INSTALLER_SHA256 = (
    "be1bad9d4e67a8753eb76fb4940e9a08036786675c7adf060627e55791bf110d"
)
PINNED_INSTALLER_FILENAME = "Miniforge3-Linux-x86_64.sh"
PINNED_INSTALLER = Path(
    "/orcd/data/lhtsai/001/om2/mabdel03/Miniforge3-Linux-x86_64.sh"
)
PINNED_INSTALLER_CONTRACT = {
    "conda_version": "25.11.0",
    "filename": PINNED_INSTALLER_FILENAME,
    "release": "Miniforge3-25.11.0-1",
    "sha256": EXPECTED_INSTALLER_SHA256,
}
FORBIDDEN_PREFIXES = (
    Path("/orcd/home/002/mabdel03/conda_envs/asys_env"),
    Path("/orcd/home/002/mabdel03/conda_envs/serve_env"),
    Path("/orcd/data/lhtsai/001/om2/mabdel03/miniforge3"),
)
R4_COMPLETION_MARKER_NAME = "CONDA_TOOLCHAIN_COMPLETE.json"
OBSERVED_FALLBACK_SHEBANG = "#!/usr/bin/env python"
PORTABLE_SHEBANG_LIMIT = 127


class FailureSealError(RuntimeError):
    """The r4 failure cannot be classified or sealed without ambiguity."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _safe_root(path: str | Path, *, description: str) -> Path:
    lexical = Path(path).expanduser().absolute()
    try:
        metadata = lexical.lstat()
    except OSError as exc:
        raise FailureSealError(f"missing {description}: {lexical}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FailureSealError(f"{description} is not a real directory: {lexical}")
    if lexical.resolve(strict=True) != lexical:
        raise FailureSealError(f"{description} is not canonical: {lexical}")
    return lexical


def _read_json(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FailureSealError(f"cannot read {description}: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FailureSealError(f"{description} is not a regular file: {path}")
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise FailureSealError(f"{description} is not canonical JSON: {path}")
    return value, raw


def _file_ref(path: Path, *, description: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise FailureSealError(f"cannot read {description}: {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FailureSealError(f"{description} is not a regular file: {path}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "mode": stat.S_IMODE(metadata.st_mode),
        "link_count": metadata.st_nlink,
    }


def _children(path: Path, expected: set[str], *, description: str) -> None:
    observed = {item.name for item in path.iterdir()}
    if observed != expected:
        raise FailureSealError(
            f"{description} children drifted: expected {sorted(expected)}, "
            f"got {sorted(observed)}"
        )


def _forensic_tree_inventory(root: Path) -> dict[str, Any]:
    """Hash bytes and raw link text without following relocated-prefix links."""

    records: list[dict[str, Any]] = []
    inode_paths: dict[tuple[int, int], list[str]] = {}
    inode_link_counts: dict[tuple[int, int], int] = {}
    file_count = 0
    directory_count = 0
    symlink_count = 0
    total_bytes = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        for child in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(child.path)
            relative = path.relative_to(root).as_posix()
            metadata = child.stat(follow_symlinks=False)
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                directory_count += 1
                records.append(
                    {"path": relative, "type": "directory", "mode": mode}
                )
                stack.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                raw = path.read_bytes()
                file_count += 1
                total_bytes += len(raw)
                records.append(
                    {
                        "path": relative,
                        "type": "file",
                        "mode": mode,
                        "size": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
                inode = (metadata.st_dev, metadata.st_ino)
                inode_paths.setdefault(inode, []).append(relative)
                inode_link_counts[inode] = metadata.st_nlink
            elif stat.S_ISLNK(metadata.st_mode):
                symlink_count += 1
                records.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "mode": mode,
                        "target": os.readlink(path),
                    }
                )
            else:
                raise FailureSealError(
                    f"unsupported entry in forensic tree: {path}"
                )
    external_hardlinks = [
        sorted(paths)
        for inode, paths in inode_paths.items()
        if inode_link_counts[inode] != len(paths)
    ]
    if external_hardlinks:
        raise FailureSealError(
            "forensic tree has regular-file hardlinks outside its root"
        )
    records.sort(key=lambda row: (row["path"], row["type"]))
    return {
        "inventory_sha256": hashlib.sha256(_canonical_bytes(records)).hexdigest(),
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "total_bytes": total_bytes,
        "hardlink_group_count": sum(
            len(paths) > 1 for paths in inode_paths.values()
        ),
        "external_shared_inode_count": 0,
        "symlinks_followed": False,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _seal_recursively(root: Path) -> None:
    entries: list[Path] = []
    for directory, names, files in os.walk(root, topdown=False, followlinks=False):
        current = Path(directory)
        entries.extend(current / name for name in files)
        entries.extend(current / name for name in names)
        entries.append(current)
    for path in entries:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            continue
        os.chmod(path, stat.S_IMODE(metadata.st_mode) & ~0o222)
    _fsync_directory(root.parent)


def _require_recursively_read_only(root: Path, *, description: str) -> None:
    for directory, names, files in os.walk(root, followlinks=False):
        for name in [".", *names, *files]:
            path = Path(directory) if name == "." else Path(directory) / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                continue
            if stat.S_IMODE(metadata.st_mode) & 0o222:
                raise FailureSealError(f"{description} has a writable entry: {path}")


def _publish_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FailureSealError(f"short write while publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _git(checkout: Path, *args: str) -> str:
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
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(checkout), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise FailureSealError(
            f"git {' '.join(args)} failed rc={completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _validate_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    observed = value.get(field)
    if not isinstance(observed, str) or not re.fullmatch(r"[0-9a-f]{64}", observed):
        raise FailureSealError(f"{label} lacks a valid {field}")
    if observed != _self_hash(value, field):
        raise FailureSealError(f"{label} {field} drifted")


def _scheduler_evidence(user: str) -> dict[str, Any]:
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
        raise FailureSealError(
            f"squeue failed rc={completed.returncode}: {completed.stderr.strip()}"
        )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    matching = [
        line
        for line in rows
        if "s5v12r4" in line
        or "schema5-v1.2-r4" in line
        or "sweep-recovery-schema5-v1.2-r4" in line
    ]
    if matching:
        raise FailureSealError(f"r4 scheduler jobs remain live: {matching}")
    return {
        "scheduler_user": user,
        "query": ["/usr/bin/squeue", "-h", "-r", "-u", user],
        "matching_r4_jobs": [],
    }


def _validate_r4_source(recovery_root: Path) -> dict[str, Any]:
    checkout = _safe_root(
        recovery_root / "materialization_pilot_source_checkout_v1_2_r4",
        description="r4 pilot checkout",
    )
    tag_ref = f"refs/tags/{R4_TAG}"
    if (
        _git(checkout, "cat-file", "-t", tag_ref) != "tag"
        or _git(checkout, "rev-parse", f"{tag_ref}^{{commit}}") != R4_COMMIT
        or _git(checkout, "rev-parse", "HEAD") != R4_COMMIT
        or _git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise FailureSealError("r4 pilot checkout is not the exact clean tag")
    if (checkout / ".git/objects/info/alternates").exists():
        raise FailureSealError("r4 pilot checkout uses object alternates")
    provisioner = checkout / "scripts/provision_schema5_conda_toolchain.py"
    identity = checkout / "scripts/schema5_conda_runtime_identity.py"
    durable = recovery_root / "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R4_COMPLETE.json"
    marker, marker_raw = _read_json(durable, description="r4 durable Git marker")
    if (
        marker.get("passed") is not True
        or marker.get("annotated_tag") is not True
        or marker.get("clean_checkout") is not True
        or marker.get("release_tag") != R4_TAG
        or marker.get("chain_namespace") != R4_NAMESPACE
        or marker.get("release_git_commit") != R4_COMMIT
        or marker.get("remote_commit") != R4_COMMIT
        or marker.get("remote_peeled_commit") != R4_COMMIT
    ):
        raise FailureSealError("r4 durable Git marker identity drifted")
    return {
        "checkout": str(checkout),
        "commit": R4_COMMIT,
        "tag": R4_TAG,
        "tag_object": _git(checkout, "rev-parse", tag_ref),
        "provisioner": _file_ref(provisioner, description="r4 provisioner"),
        "runtime_identity": _file_ref(identity, description="r4 runtime identity"),
        "durable_marker": {
            "path": str(durable),
            "sha256": hashlib.sha256(marker_raw).hexdigest(),
            "size": len(marker_raw),
            "marker_id": marker.get("marker_id"),
        },
    }


def _validate_intent(
    intent: Mapping[str, Any],
    *,
    namespace: Path,
    toolchain: Path,
    transaction: Path,
) -> None:
    expected = {
        "schema_version": 1,
        "protocol": f"{R4_TOOLCHAIN_PROTOCOL}-intent",
        "release_tag": R4_TAG,
        "chain_namespace": R4_NAMESPACE,
        "namespace_root": str(namespace),
        "toolchain_root": str(toolchain),
        "base_prefix": str(toolchain / "base"),
        "installer_source": str(PINNED_INSTALLER),
        "installer_contract": dict(PINNED_INSTALLER_CONTRACT),
        "forbidden_prefixes": [str(path) for path in FORBIDDEN_PREFIXES],
        "mutation_scope": [str(toolchain), str(transaction)],
        "live_prefix_queries_permitted": False,
        "shared_base_queries_permitted": False,
    }
    unsigned = dict(intent)
    intent_id = unsigned.pop("intent_id", None)
    if unsigned != expected or intent_id != _self_hash(intent, "intent_id"):
        raise FailureSealError("r4 provision intent identity drifted")


def _validate_attempt(
    path: Path,
    *,
    generation: str,
    toolchain: Path,
    intent_id: str,
) -> dict[str, Any]:
    _children(
        path,
        {
            "ATTEMPT_INTENT.json",
            "installer.stdout",
            "installer.stderr",
            "scratch",
        },
        description=f"r4 {generation} attempt",
    )
    scratch = _safe_root(path / "scratch", description=f"r4 {generation} scratch")
    scratch_directories: list[str] = []
    for directory, names, files in os.walk(scratch, followlinks=False):
        current = Path(directory)
        if files:
            raise FailureSealError(
                f"r4 {generation} scratch contains files: {files}"
            )
        for name in names:
            child = current / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise FailureSealError(
                    f"r4 {generation} scratch has an unsafe entry: {child}"
                )
        scratch_directories.append(current.relative_to(scratch).as_posix())
    expected_scratch = [
        ".",
        "envs",
        "home",
        "home/.conda",
        "pkgs",
        "tmp",
        "xdg-cache",
        "xdg-config",
        "xdg-data",
    ]
    if sorted(scratch_directories) != expected_scratch:
        raise FailureSealError(f"r4 {generation} scratch inventory drifted")
    value, _ = _read_json(
        path / "ATTEMPT_INTENT.json",
        description=f"r4 {generation} attempt intent",
    )
    expected = {
        "schema_version": 1,
        "protocol": f"{R4_TOOLCHAIN_PROTOCOL}-installer-attempt",
        "generation": generation,
        "toolchain_root": str(toolchain),
        "base_prefix": str(toolchain / "base"),
        "installer_sha256": EXPECTED_INSTALLER_SHA256,
        "intent_id": intent_id,
    }
    unsigned = dict(value)
    attempt_id = unsigned.pop("attempt_id", None)
    if unsigned != expected or attempt_id != _self_hash(value, "attempt_id"):
        raise FailureSealError(f"r4 {generation} attempt identity drifted")
    return {
        "generation": generation,
        "attempt_id": attempt_id,
        "intent": _file_ref(
            path / "ATTEMPT_INTENT.json",
            description=f"r4 {generation} attempt intent",
        ),
        "stdout": _file_ref(
            path / "installer.stdout",
            description=f"r4 {generation} installer stdout",
        ),
        "stderr": _file_ref(
            path / "installer.stderr",
            description=f"r4 {generation} installer stderr",
        ),
        "scratch_directories": expected_scratch,
    }


def _scientific_state_evidence(
    recovery_root: Path,
    *,
    preserved: Mapping[str, Any] | None,
) -> dict[str, Any]:
    results_root = recovery_root.parent.parent
    superseding_release_paths = [
        results_root / ".dispatcher-schema5-v1",
        results_root / "server_pools/schema5-v1",
        results_root / "full_sweep_schema5_v1",
        results_root / "full_sweep_agent_counts_schema5_v1",
        results_root / "full_sweep_agent_count_7_schema5_v1",
        recovery_root / "PROTECTED_CAPACITY_COMPLETE.json",
    ]
    permanently_absent_r4_paths = [
        recovery_root / "slurm_canaries/schema5-v1.2-r4",
        recovery_root / "materialization_pilots/schema5-v1.2-r4",
        recovery_root / "RECOVERY_CHAIN_SCHEMA5_V1_2_R4.json",
    ]
    historical_absent_paths = [
        *superseding_release_paths,
        *permanently_absent_r4_paths,
    ]
    permanent_present = [
        str(path)
        for path in permanently_absent_r4_paths
        if path.exists() or path.is_symlink()
    ]
    if permanent_present:
        raise FailureSealError(
            "an immutable r4 prelaunch namespace appeared after failure: "
            f"{permanent_present}"
        )
    expected = {
        "capture_scope": "pre_r5_scientific_and_operational_roots",
        "captured_before_superseding_release_outputs": True,
        "result_mutation_count": 0,
        "scheduler_job_count": 0,
        "required_absent_paths_at_seal": [
            str(path) for path in historical_absent_paths
        ],
        "permanently_absent_r4_paths": [
            str(path) for path in permanently_absent_r4_paths
        ],
    }
    if preserved is None:
        present = [
            str(path)
            for path in historical_absent_paths
            if path.exists() or path.is_symlink()
        ]
        if present:
            raise FailureSealError(
                f"r4 failure is not prelaunch zero-result evidence: {present}"
            )
    elif dict(preserved) != expected:
        raise FailureSealError(
            "sealed r4 prelaunch scientific-state contract drifted"
        )
    # The marker proves that all historical roots were absent at publication.
    # Subsequent r5 roots have their own lineage; r4 execution roots stay absent.
    return expected


def _snapshot(
    recovery_root: Path,
    *,
    scheduler_user: str,
    require_sealed: bool,
    preserved_scientific_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = _validate_r4_source(recovery_root)
    namespace = _safe_root(
        recovery_root / "toolchains" / R4_NAMESPACE,
        description="r4 toolchain namespace",
    )
    toolchain = _safe_root(
        namespace / R4_TOOLCHAIN_DIRECTORY,
        description="failed r4 toolchain root",
    )
    transaction = _safe_root(
        namespace / R4_TRANSACTION_DIRECTORY,
        description="r4 toolchain transaction root",
    )
    _children(
        toolchain,
        {"CONDA_TOOLCHAIN_PROVISION_INTENT.json", "base", "inputs"},
        description="failed r4 toolchain",
    )
    _children(
        transaction,
        {"attempts", "provision.lock", "quarantine"},
        description="r4 toolchain transaction",
    )
    attempts_root = _safe_root(
        transaction / "attempts", description="r4 installer attempts"
    )
    quarantine_root = _safe_root(
        transaction / "quarantine", description="r4 prefix quarantine"
    )
    _children(attempts_root, {"g0001", "g0002"}, description="r4 attempts")
    _children(quarantine_root, {"g0001"}, description="r4 quarantine")
    quarantine = _safe_root(
        quarantine_root / "g0001", description="r4 g0001 quarantine"
    )
    _children(
        quarantine,
        {"base", "QUARANTINE_COMPLETE.json"},
        description="r4 g0001 quarantine",
    )
    inputs = _safe_root(toolchain / "inputs", description="r4 installer input")
    _children(
        inputs,
        {PINNED_INSTALLER_FILENAME},
        description="r4 installer input",
    )
    if (toolchain / R4_COMPLETION_MARKER_NAME).exists():
        raise FailureSealError("failed r4 toolchain unexpectedly has a completion marker")

    intent, _ = _read_json(
        toolchain / "CONDA_TOOLCHAIN_PROVISION_INTENT.json",
        description="r4 toolchain provision intent",
    )
    _validate_intent(
        intent,
        namespace=namespace,
        toolchain=toolchain,
        transaction=transaction,
    )
    attempts = [
        _validate_attempt(
            attempts_root / generation,
            generation=generation,
            toolchain=toolchain,
            intent_id=str(intent["intent_id"]),
        )
        for generation in ("g0001", "g0002")
    ]
    receipt, _ = _read_json(
        quarantine / "QUARANTINE_COMPLETE.json",
        description="r4 incomplete-prefix quarantine receipt",
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol")
        != f"{R4_TOOLCHAIN_PROTOCOL}-incomplete-prefix-quarantine"
        or receipt.get("generation") != "g0001"
        or receipt.get("source_toolchain_root") != str(toolchain)
        or receipt.get("moved") != ["base"]
    ):
        raise FailureSealError("r4 incomplete-prefix quarantine receipt drifted")
    _validate_self_hash(receipt, "receipt_id", "r4 quarantine receipt")

    conda = toolchain / "base/bin/conda"
    try:
        first_line = conda.open("rb").readline(16 * 1024).decode("utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise FailureSealError(f"cannot read failed r4 Conda entrypoint: {exc}") from exc
    interpreter = toolchain / "base/bin/python"
    required_shebang_bytes = len(f"#!{interpreter}\n".encode("utf-8"))
    if (
        first_line != OBSERVED_FALLBACK_SHEBANG
        or required_shebang_bytes <= PORTABLE_SHEBANG_LIMIT
    ):
        raise FailureSealError("r4 overlong-prefix failure signature drifted")

    quarantine_receipt = _file_ref(
        quarantine / "QUARANTINE_COMPLETE.json",
        description="r4 quarantine receipt",
    )
    installer_copy = _file_ref(
        inputs / PINNED_INSTALLER_FILENAME,
        description="r4 copied installer",
    )
    if installer_copy["sha256"] != EXPECTED_INSTALLER_SHA256:
        raise FailureSealError("r4 copied installer digest drifted")

    current_inventory = _forensic_tree_inventory(toolchain / "base")
    quarantine_inventory = _forensic_tree_inventory(quarantine / "base")
    if require_sealed:
        _require_recursively_read_only(
            toolchain, description="sealed failed r4 toolchain"
        )
        _require_recursively_read_only(
            transaction, description="sealed r4 toolchain transaction"
        )

    scientific_state = _scientific_state_evidence(
        recovery_root,
        preserved=preserved_scientific_state,
    )

    return {
        "source_release": source,
        "failure": {
            "classification": CLASSIFICATION,
            "toolchain_root": str(toolchain),
            "transaction_root": str(transaction),
            "completion_marker_absent": True,
            "observed_conda_shebang": first_line,
            "required_interpreter": str(interpreter),
            "required_absolute_shebang_bytes": required_shebang_bytes,
            "portable_shebang_limit": PORTABLE_SHEBANG_LIMIT,
            "failure_message": (
                "isolated Conda runtime symlink audit failed: Conda entrypoint "
                "must name one absolute base-prefix interpreter"
            ),
        },
        "provision_intent": {
            "intent_id": intent["intent_id"],
            "file": _file_ref(
                toolchain / "CONDA_TOOLCHAIN_PROVISION_INTENT.json",
                description="r4 provision intent",
            ),
            "mutation_scope": list(intent["mutation_scope"]),
            "live_prefix_queries_permitted": False,
            "shared_base_queries_permitted": False,
        },
        "installer_copy": installer_copy,
        "attempts": attempts,
        "quarantine": {
            "receipt_id": receipt["receipt_id"],
            "receipt": quarantine_receipt,
            "base_inventory": quarantine_inventory,
        },
        "failed_base_inventory": current_inventory,
        "scheduler": _scheduler_evidence(scheduler_user),
        "scientific_state": scientific_state,
        "sealed_read_only": require_sealed,
    }


def verify_failure_seal(
    evidence_root: str | Path,
    *,
    recovery_root: str | Path,
    scheduler_user: str | None = None,
) -> dict[str, Any]:
    evidence = _safe_root(evidence_root, description="r4 failure evidence root")
    recovery = _safe_root(recovery_root, description="schema-5 recovery root")
    marker_path = evidence / MARKER_NAME
    marker, raw = _read_json(marker_path, description="r4 toolchain failure seal")
    expected_keys = {
        "schema_version",
        "protocol",
        "classification",
        "evidence",
        "marker_id",
    }
    if (
        set(marker) != expected_keys
        or marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("protocol") != PROTOCOL
        or marker.get("classification") != CLASSIFICATION
    ):
        raise FailureSealError("r4 toolchain failure seal shape drifted")
    _validate_self_hash(marker, "marker_id", "r4 toolchain failure seal")
    sealed_scheduler = marker.get("evidence", {}).get("scheduler", {})
    if scheduler_user is None:
        scheduler_user = sealed_scheduler.get("scheduler_user")
    if not isinstance(scheduler_user, str) or not scheduler_user:
        raise FailureSealError("r4 toolchain failure seal lacks a scheduler user")
    sealed_evidence = marker.get("evidence")
    if not isinstance(sealed_evidence, dict):
        raise FailureSealError("r4 toolchain failure seal lacks evidence")
    sealed_scientific_state = sealed_evidence.get("scientific_state")
    if not isinstance(sealed_scientific_state, dict):
        raise FailureSealError(
            "r4 toolchain failure seal lacks scientific-state evidence"
        )
    observed = _snapshot(
        recovery,
        scheduler_user=scheduler_user,
        require_sealed=True,
        preserved_scientific_state=sealed_scientific_state,
    )
    if marker.get("evidence") != observed:
        raise FailureSealError("sealed r4 toolchain failure evidence drifted")
    if stat.S_IMODE(marker_path.stat().st_mode) & 0o222:
        raise FailureSealError("r4 toolchain failure seal is writable")
    if stat.S_IMODE(evidence.stat().st_mode) & 0o222:
        raise FailureSealError("r4 toolchain failure evidence root is writable")
    return {
        "passed": True,
        "root": str(evidence),
        "protocol": PROTOCOL,
        "release_tag": R4_TAG,
        "release_git_commit": R4_COMMIT,
        "chain_namespace": R4_NAMESPACE,
        "marker": str(marker_path),
        "marker_sha256": hashlib.sha256(raw).hexdigest(),
        "marker_size": len(raw),
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
    recovery = _safe_root(recovery_root, description="schema-5 recovery root")
    expected_evidence = recovery / "prelaunch_failures/schema5-v1.2-r4-toolchain"
    lexical_evidence = Path(evidence_root).expanduser().absolute()
    if lexical_evidence != expected_evidence:
        raise FailureSealError(
            f"r4 failure evidence root must be {expected_evidence}"
        )
    marker_path = lexical_evidence / MARKER_NAME
    if marker_path.exists() or marker_path.is_symlink():
        return {
            "action": "already_sealed",
            **verify_failure_seal(
                lexical_evidence,
                recovery_root=recovery,
                scheduler_user=scheduler_user,
            ),
        }

    namespace = _safe_root(
        recovery / "toolchains" / R4_NAMESPACE,
        description="r4 toolchain namespace",
    )
    transaction = _safe_root(
        namespace / R4_TRANSACTION_DIRECTORY,
        description="r4 transaction root",
    )
    lock_path = transaction / "provision.lock"
    lock_descriptor = os.open(
        lock_path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FailureSealError("r4 toolchain transaction is still active") from exc
        preview = _snapshot(
            recovery, scheduler_user=scheduler_user, require_sealed=False
        )
        if not apply:
            return {
                "action": "would_seal",
                "classification": CLASSIFICATION,
                "failure": preview["failure"],
                "scientific_state": preview["scientific_state"],
            }
        toolchain = _safe_root(
            namespace / R4_TOOLCHAIN_DIRECTORY,
            description="failed r4 toolchain root",
        )
        _seal_recursively(toolchain)
        _seal_recursively(transaction)
        evidence = _snapshot(
            recovery, scheduler_user=scheduler_user, require_sealed=True
        )
        lexical_evidence.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        lexical_evidence.mkdir(mode=0o755)
        marker: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "classification": CLASSIFICATION,
            "evidence": evidence,
        }
        marker["marker_id"] = _self_hash(marker, "marker_id")
        _publish_once(marker_path, marker)
        os.chmod(lexical_evidence, 0o555)
        _fsync_directory(lexical_evidence.parent)
    finally:
        os.close(lock_descriptor)
    return {
        "action": "sealed",
        **verify_failure_seal(
            lexical_evidence,
            recovery_root=recovery,
            scheduler_user=scheduler_user,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("seal", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--evidence-root", required=True, type=Path)
        command.add_argument("--recovery-root", required=True, type=Path)
        if name == "seal":
            command.add_argument("--scheduler-user", required=True)
            command.add_argument("--apply", action="store_true")
        else:
            command.add_argument("--scheduler-user")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal":
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
    except (FailureSealError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
