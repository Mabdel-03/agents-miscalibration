#!/usr/bin/env python3
"""Build and verify the external schema-5 watchdog deployment.

The bundle command only writes an offline deployment bundle.  It does not contact a
host, install a key, invoke systemd, or alter Slurm.  Evidence commands inspect an
already installed deployment through no-follow descriptors and publish immutable,
marker-last JSON for the recovery chain.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_RELEASE_ROOT = Path(__file__).resolve().parent.parent
_WATCHDOG_RUNTIME = (
    _RELEASE_ROOT
    / "src"
    / "agents_scaling"
    / "serving"
    / "external_watchdog.py"
)
_RUNTIME_SPEC = importlib.util.spec_from_file_location(
    "_schema5_watchdog_deployment_runtime", _WATCHDOG_RUNTIME
)
if _RUNTIME_SPEC is None or _RUNTIME_SPEC.loader is None:
    raise ImportError(f"cannot load sealed watchdog runtime: {_WATCHDOG_RUNTIME}")
_RUNTIME = importlib.util.module_from_spec(_RUNTIME_SPEC)
sys.modules[_RUNTIME_SPEC.name] = _RUNTIME
_RUNTIME_SPEC.loader.exec_module(_RUNTIME)
WATCHDOG_INTERVAL_SECONDS = _RUNTIME.WATCHDOG_INTERVAL_SECONDS
WATCHDOG_OBSERVATION_GAP_SECONDS = (
    _RUNTIME.WATCHDOG_OBSERVATION_GAP_SECONDS
)
WATCHDOG_PROTOCOL = _RUNTIME.WATCHDOG_PROTOCOL
WATCHDOG_SCHEMA_VERSION = _RUNTIME.WATCHDOG_SCHEMA_VERSION


RELEASE_ID = "sweep-recovery-schema5-v1.2"
RELEASE_TAG = "sweep-recovery-schema5-v1.2-r2"
CHAIN_NAMESPACE = "schema5-v1.2-r2"
DEPLOYMENT_PROTOCOL = "schema5-external-watchdog-deployment-evidence-v1"
LIVENESS_PROTOCOL = "schema5-external-watchdog-liveness-evidence-v1"
BUNDLE_PROTOCOL = "schema5-external-watchdog-deployment-bundle-v3"
ACK_PROTOCOL = "schema5-external-watchdog-liveness-ack-v1"
HEARTBEAT_PROTOCOL = "schema5-v1.2-r2-external-watchdog-v1"
SERVICE_NAME = "agents-scaling-schema5-watchdog.service"
TIMER_NAME = "agents-scaling-schema5-watchdog.timer"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
SSH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,252}\Z")
EMAIL = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?\Z"
)
PUBLIC_KEY = re.compile(
    r"(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(?:256|384|521)) "
    r"([A-Za-z0-9+/]+={0,2})(?: [^\r\n]+)?\Z"
)
WATCHDOG_RUNTIME_PATHS = (
    Path("scripts/build_schema5_watchdog_deployment.py"),
    Path("scripts/schema5_external_watchdog.py"),
    Path("src/agents_scaling/serving/external_watchdog.py"),
)


class DeploymentError(RuntimeError):
    """The offline bundle or inspected deployment is unsafe or inconsistent."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _canonical_path(
    path: Path, *, description: str, kind: str, create: bool = False
) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if any(character in str(lexical) for character in ("\x00", "\n", "\r")):
        raise DeploymentError(f"{description} path contains unsafe characters")
    if create:
        lexical.mkdir(parents=True, exist_ok=True)
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DeploymentError(f"{description} is unavailable: {exc}") from exc
    if resolved != lexical:
        raise DeploymentError(f"{description} traverses a symlink")
    metadata = lexical.stat(follow_symlinks=False)
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise DeploymentError(f"{description} is not a regular file")
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise DeploymentError(f"{description} is not a directory")
    return lexical


def _stable_bytes(
    path: Path,
    *,
    description: str,
    require_read_only: bool = False,
    require_single_link: bool = True,
) -> bytes:
    canonical = _canonical_path(path, description=description, kind="file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(canonical, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    current = canonical.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or identity(before) != identity(after)
        or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
        or (require_read_only and stat.S_IMODE(before.st_mode) & 0o222)
        or (require_single_link and before.st_nlink != 1)
    ):
        raise DeploymentError(f"{description} is mutable, linked, or changed")
    return b"".join(chunks)


def _sealed_json(
    path: Path,
    *,
    description: str,
    require_read_only: bool = True,
) -> tuple[dict[str, Any], bytes]:
    raw = _stable_bytes(
        path,
        description=description,
        require_read_only=require_read_only,
        require_single_link=True,
    )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DeploymentError(
                    f"{description} duplicates JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DeploymentError(
                    f"{description} contains non-finite value {token}"
                )
            ),
        )
    except DeploymentError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DeploymentError(f"{description} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or raw != _canonical(value):
        raise DeploymentError(f"{description} is not canonical JSON")
    return value, raw


def _compact_canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _resolve_inventory_pinned_harness_python(
    *,
    control_state_dir: Path,
    harness_python: Path,
    control_sha256: str,
    require_pristine_paused: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Resolve Conda's ``bin/python`` through the sealed environment inventory.

    The lexical Conda alias is intentionally retained as provenance, but the
    authorized-keys command executes the immutable regular target.  This closes both
    failure modes of accepting an unpinned alias and rejecting every normal Conda
    environment merely because ``bin/python`` is a same-prefix symlink.
    """

    control_path = control_state_dir / "control.json"
    control, _ = _sealed_json(
        control_path,
        description="paused schema-5 control",
        require_read_only=False,
    )
    immutable = control.get("immutable")
    finalization = control.get("finalization")
    if (
        control.get("immutable_sha256") != control_sha256
        or not isinstance(immutable, Mapping)
    ):
        raise DeploymentError(
            "watchdog harness resolution does not bind the expected control"
        )
    if require_pristine_paused and (
        control.get("desired_state") != "paused"
        or control.get("drain_requested") is not False
        or not isinstance(finalization, Mapping)
        or finalization.get("state") != "idle"
    ):
        raise DeploymentError(
            "watchdog harness resolution requires the exact pristine paused control"
        )

    try:
        prefix = _canonical_path(
            Path(str(immutable["harness_environment_prefix"])),
            description="sealed harness prefix",
            kind="directory",
        )
        manifest_path = _canonical_path(
            Path(str(immutable["harness_environment_manifest_path"])),
            description="sealed harness environment manifest",
            kind="file",
        )
        manifest_sha256 = str(immutable["harness_environment_sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentError(
            "paused control lacks the sealed harness environment binding"
        ) from exc
    if (
        SHA256.fullmatch(manifest_sha256) is None
        or stat.S_IMODE(prefix.stat(follow_symlinks=False).st_mode) & 0o222
    ):
        raise DeploymentError("sealed harness prefix binding is invalid")

    lexical = Path(os.path.abspath(os.fspath(harness_python.expanduser())))
    expected_lexical = prefix / "bin" / "python"
    if lexical != expected_lexical:
        raise DeploymentError(
            "watchdog harness Python must be the pinned prefix's bin/python alias"
        )
    try:
        lexical_info = lexical.lstat()
        target = lexical.resolve(strict=True)
        relative_target = target.relative_to(prefix)
        relative_lexical = lexical.relative_to(prefix)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DeploymentError(
            "watchdog harness Python is unavailable or escapes its sealed prefix"
        ) from exc
    if target == prefix:
        raise DeploymentError("watchdog harness Python resolves to its prefix root")

    manifest, manifest_raw = _sealed_json(
        manifest_path,
        description="sealed harness environment manifest",
    )
    if hashlib.sha256(manifest_raw).hexdigest() != manifest_sha256:
        raise DeploymentError("sealed harness environment manifest hash drifted")
    inventory = manifest.get("directory_inventory")
    entries = inventory.get("entries") if isinstance(inventory, Mapping) else None
    if (
        manifest.get("schema_version") != 3
        or manifest.get("role") != "harness"
        or manifest.get("sealed_read_only") is not True
        or Path(str(manifest.get("prefix", ""))).expanduser().resolve()
        != prefix
        or not isinstance(inventory, Mapping)
        or set(inventory)
        != {
            "entries",
            "inventory_sha256",
            "entry_count",
            "file_count",
            "directory_count",
            "symlink_count",
            "total_file_bytes",
        }
        or not isinstance(entries, list)
        or inventory.get("inventory_sha256")
        != hashlib.sha256(_compact_canonical(entries)).hexdigest()
        or inventory.get("entry_count") != len(entries)
        or inventory.get("file_count")
        != sum(
            isinstance(entry, Mapping) and entry.get("type") == "file"
            for entry in entries
        )
        or inventory.get("directory_count")
        != sum(
            isinstance(entry, Mapping) and entry.get("type") == "directory"
            for entry in entries
        )
        or inventory.get("symlink_count")
        != sum(
            isinstance(entry, Mapping) and entry.get("type") == "symlink"
            for entry in entries
        )
    ):
        raise DeploymentError("sealed harness environment inventory is invalid")

    by_path: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("path"), str)
            or entry["path"] in by_path
        ):
            raise DeploymentError("sealed harness environment entries are malformed")
        by_path[str(entry["path"])] = entry
    if list(by_path) != sorted(by_path):
        raise DeploymentError("sealed harness environment entries are not sorted")

    bin_path = lexical.parent
    bin_entry = by_path.get("bin")
    bin_info = bin_path.stat(follow_symlinks=False)
    if (
        bin_path.is_symlink()
        or not stat.S_ISDIR(bin_info.st_mode)
        or stat.S_IMODE(bin_info.st_mode) & 0o222
        or not isinstance(bin_entry, Mapping)
        or dict(bin_entry)
        != {
            "path": "bin",
            "type": "directory",
            "mode": stat.S_IMODE(bin_info.st_mode),
        }
    ):
        raise DeploymentError("sealed harness bin directory is mutable or unpinned")

    lexical_entry = by_path.get(relative_lexical.as_posix())
    if stat.S_ISLNK(lexical_info.st_mode):
        try:
            link_target = os.readlink(lexical)
        except OSError as exc:
            raise DeploymentError(
                "watchdog harness Python symlink cannot be inspected"
            ) from exc
        direct_target = (
            Path(link_target)
            if Path(link_target).is_absolute()
            else lexical.parent / link_target
        )
        direct_target = Path(os.path.abspath(os.fspath(direct_target)))
        if (
            direct_target != target
            or direct_target.is_symlink()
            or not isinstance(lexical_entry, Mapping)
            or dict(lexical_entry)
            != {
                "path": relative_lexical.as_posix(),
                "type": "symlink",
                "mode": stat.S_IMODE(lexical_info.st_mode),
                "target": link_target,
            }
        ):
            raise DeploymentError(
                "watchdog harness Python symlink is chained or not inventory-pinned"
            )
    elif (
        not stat.S_ISREG(lexical_info.st_mode)
        or target != lexical
        or not isinstance(lexical_entry, Mapping)
    ):
        raise DeploymentError(
            "watchdog harness Python lexical entry is unsafe or unpinned"
        )

    target_raw = _stable_bytes(
        target,
        description="resolved sealed harness Python",
        require_read_only=True,
        require_single_link=True,
    )
    target_info = target.stat(follow_symlinks=False)
    target_entry = by_path.get(relative_target.as_posix())
    expected_target_entry = {
        "path": relative_target.as_posix(),
        "type": "file",
        "mode": stat.S_IMODE(target_info.st_mode),
        "size": len(target_raw),
        "sha256": hashlib.sha256(target_raw).hexdigest(),
    }
    if (
        not stat.S_ISREG(target_info.st_mode)
        or stat.S_IMODE(target_info.st_mode) & 0o111 == 0
        or not isinstance(target_entry, Mapping)
        or dict(target_entry) != expected_target_entry
    ):
        raise DeploymentError(
            "resolved watchdog harness Python is unexecutable or not inventory-pinned"
        )
    if not stat.S_ISLNK(lexical_info.st_mode) and dict(lexical_entry) != (
        expected_target_entry
    ):
        raise DeploymentError(
            "regular watchdog harness Python alias is not inventory-pinned"
        )

    binding = {
        "prefix": str(prefix),
        "lexical_path": str(lexical),
        "resolved_path": str(target),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "inventory_sha256": str(inventory["inventory_sha256"]),
        "lexical_entry": dict(lexical_entry),
        "resolved_entry": dict(target_entry),
    }
    return target, binding


def _publish_once(path: Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    target = Path(os.path.abspath(os.fspath(path.expanduser())))
    parent = _canonical_path(
        target.parent,
        description=f"{target.name} parent",
        kind="directory",
        create=True,
    )
    payload = _canonical(value)
    lock_path = parent / f".{target.name}.publish.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        pattern = f".{target.name}."
        candidates = sorted(
            child
            for child in parent.iterdir()
            if child.name.startswith(pattern)
            and child.name.endswith(".publishing")
        )
        exact_candidates: list[Path] = []
        target_metadata = (
            target.stat(follow_symlinks=False)
            if target.exists() and not target.is_symlink()
            else None
        )
        if target.is_symlink():
            raise DeploymentError(f"immutable output is symlinked: {target}")
        for candidate in candidates:
            if candidate.is_symlink() or not candidate.is_file():
                raise DeploymentError(
                    f"unsafe interrupted publication artifact: {candidate}"
                )
            candidate_raw = _stable_bytes(
                candidate,
                description=f"interrupted publication {candidate.name}",
                require_read_only=True,
                require_single_link=False,
            )
            candidate_metadata = candidate.stat(follow_symlinks=False)
            if target_metadata is not None and (
                candidate_metadata.st_dev,
                candidate_metadata.st_ino,
            ) == (target_metadata.st_dev, target_metadata.st_ino):
                candidate.unlink()
            elif candidate_raw == payload:
                exact_candidates.append(candidate)
            else:
                raise DeploymentError(
                    f"conflicting interrupted publication artifact: {candidate}"
                )
        if not target.exists() and exact_candidates:
            survivor = exact_candidates.pop(0)
            os.link(survivor, target, follow_symlinks=False)
            survivor.unlink()
        for candidate in exact_candidates:
            candidate.unlink()
        if target.exists():
            if _stable_bytes(
                target,
                description=f"existing {target.name}",
                require_read_only=True,
                require_single_link=True,
            ) != payload:
                raise DeploymentError(
                    f"immutable output already has different bytes: {target}"
                )
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=parent, prefix=f".{target.name}.", suffix=".publishing"
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.chmod(0o444)
                os.link(temporary, target, follow_symlinks=False)
                temporary.unlink()
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        directory = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
    return target, hashlib.sha256(payload).hexdigest()


def _safe_fixed_path(path: Path, *, description: str) -> str:
    value = str(path)
    if (
        not path.is_absolute()
        or re.fullmatch(r"/[A-Za-z0-9_./-]+", value) is None
        or ".." in path.parts
    ):
        raise DeploymentError(f"{description} is not a safe fixed absolute path")
    return value


def _release_fields(git_commit: str, tag_object: str) -> dict[str, Any]:
    if GIT_OBJECT.fullmatch(git_commit) is None or GIT_OBJECT.fullmatch(
        tag_object
    ) is None:
        raise DeploymentError("release commit or annotated-tag object is malformed")
    return {
        "release_id": RELEASE_ID,
        "release_tag": RELEASE_TAG,
        "release_git_commit": git_commit,
        "release_tag_object": tag_object,
        "chain_namespace": CHAIN_NAMESPACE,
    }


def _hash_release_sources(release_root: Path) -> tuple[str, list[dict[str, Any]]]:
    root = _canonical_path(
        release_root, description="immutable release root", kind="directory"
    )
    records: list[dict[str, Any]] = []
    for relative in (
        Path("scripts/build_schema5_watchdog_deployment.py"),
        Path("scripts/schema5_external_watchdog.py"),
        Path("scripts/schema5_watchdog_forced_command.py"),
        Path("src/agents_scaling/serving/external_watchdog.py"),
        Path("slurm/schema5_control.py"),
    ):
        path = root / relative
        raw = _stable_bytes(
            path,
            description=f"release source {relative}",
            require_read_only=True,
            require_single_link=True,
        )
        records.append(
            {
                "path": relative.as_posix(),
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return hashlib.sha256(_canonical(records)).hexdigest(), records


def _copy_runtime_tree(
    *, release_root: Path, bundle_root: Path
) -> tuple[Path, list[dict[str, Any]], str, int]:
    """Install the exact isolated watchdog runtime into the offline bundle."""

    runtime_root = bundle_root / "release"
    runtime_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for relative in WATCHDOG_RUNTIME_PATHS:
        source = release_root / relative
        payload = _stable_bytes(
            source,
            description=f"watchdog runtime source {relative}",
            require_read_only=True,
            require_single_link=True,
        )
        target = runtime_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or _stable_bytes(
                target,
                description=f"bundled watchdog runtime {relative}",
                require_read_only=True,
                require_single_link=True,
            ) != payload:
                raise DeploymentError(
                    f"bundled watchdog runtime conflicts: {target}"
                )
        else:
            descriptor = os.open(
                target,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0),
                0o400,
            )
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            target.chmod(0o444)
        record = {
            "path": relative.as_posix(),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        records.append(record)
        total_bytes += len(payload)
    records.sort(key=lambda row: str(row["path"]))
    inventory_sha256 = hashlib.sha256(_canonical(records)).hexdigest()
    return runtime_root, records, inventory_sha256, total_bytes


def _render_systemd_units(
    *,
    vm_python: Path,
    vm_release_root: Path,
    vm_config_path: Path,
    service_user: str,
    external_state_root: Path,
) -> tuple[bytes, bytes]:
    """Render the sole canonical service/timer source used by every deployment."""

    service = f"""[Unit]
Description=Schema-5 external dead-man watchdog
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User={service_user}
Group={service_user}
ExecStart={vm_python} -I -u {vm_release_root}/scripts/schema5_external_watchdog.py --config {vm_config_path}
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={external_state_root}
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
LockPersonality=true
MemoryDenyWriteExecute=true
""".encode("utf-8")
    timer = f"""[Unit]
Description=Run the schema-5 external watchdog every five minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
AccuracySec=5s
Persistent=true
Unit={SERVICE_NAME}

[Install]
WantedBy=timers.target
""".encode("utf-8")
    return service, timer


def build_bundle(
    *,
    output_root: Path,
    release_root: Path,
    harness_python: Path,
    control_state_dir: Path,
    git_commit: str,
    tag_object: str,
    control_sha256: str,
    remote_host: str,
    remote_user: str,
    identity_file: Path,
    known_hosts_file: Path,
    external_state_root: Path,
    public_key_file: Path,
    vm_python: Path = Path("/opt/agents-scaling-watchdog/venv/bin/python"),
    vm_release_root: Path = Path("/opt/agents-scaling-watchdog/release"),
    vm_config_path: Path = Path("/etc/agents-scaling-watchdog/watchdog.json"),
    service_user: str = "agents-scaling-watchdog",
    liveness_email: str = "mabdel03@mit.edu",
) -> dict[str, Any]:
    """Create a sealed, installable bundle without performing deployment."""

    if (
        SSH_COMPONENT.fullmatch(remote_host) is None
        or SSH_COMPONENT.fullmatch(remote_user) is None
        or remote_host.startswith("-")
        or remote_user.startswith("-")
        or SSH_COMPONENT.fullmatch(service_user) is None
        or EMAIL.fullmatch(liveness_email) is None
        or SHA256.fullmatch(control_sha256) is None
    ):
        raise DeploymentError("host, user, email, or control identity is unsafe")
    release_root = _canonical_path(
        release_root, description="immutable release root", kind="directory"
    )
    control_state_dir = _canonical_path(
        control_state_dir, description="paused control state", kind="directory"
    )
    resolved_python, harness_runtime = (
        _resolve_inventory_pinned_harness_python(
            control_state_dir=control_state_dir,
            harness_python=harness_python,
            control_sha256=control_sha256,
        )
    )
    if not os.access(resolved_python, os.X_OK):
        raise DeploymentError("immutable harness Python is not executable")
    release_digest, code_records = _hash_release_sources(release_root)
    public_key_raw = _stable_bytes(
        public_key_file,
        description="watchdog public key",
        require_single_link=True,
    )
    try:
        public_key = public_key_raw.decode("ascii").strip()
    except UnicodeError as exc:
        raise DeploymentError("watchdog public key is not ASCII") from exc
    match = PUBLIC_KEY.fullmatch(public_key)
    if match is None:
        raise DeploymentError("watchdog public key has an unsupported format")
    try:
        decoded_key = base64.b64decode(match.group(2), validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise DeploymentError("watchdog public key payload is invalid base64") from exc
    if len(decoded_key) < 4:
        raise DeploymentError("watchdog public key payload is truncated")
    bare_key = f"{match.group(1)} {match.group(2)}"
    fixed_release = _safe_fixed_path(release_root, description="release root")
    fixed_python = _safe_fixed_path(
        resolved_python, description="resolved harness Python"
    )
    lexical_python = _safe_fixed_path(
        Path(str(harness_runtime["lexical_path"])),
        description="lexical harness Python",
    )
    harness_manifest = _safe_fixed_path(
        Path(str(harness_runtime["manifest_path"])),
        description="harness environment manifest",
    )
    fixed_state = _safe_fixed_path(control_state_dir, description="control state")
    forced_script = _safe_fixed_path(
        release_root / "scripts" / "schema5_watchdog_forced_command.py",
        description="forced-command script",
    )
    forced_argv = [
        fixed_python,
        "-I",
        "-u",
        forced_script,
        "--state-dir",
        fixed_state,
        "--release-root",
        fixed_release,
        "--harness-python",
        lexical_python,
        "--resolved-harness-python",
        fixed_python,
        "--harness-environment-manifest",
        harness_manifest,
        "--harness-environment-sha256",
        str(harness_runtime["manifest_sha256"]),
        "--control-sha256",
        control_sha256,
    ]
    forced_command = " ".join(shlex.quote(item) for item in forced_argv)
    if '"' in forced_command or "\\" in forced_command:
        raise DeploymentError("forced command requires unsupported quoting")
    authorized_line = (
        'restrict,command="' + forced_command + '" ' + bare_key + "\n"
    ).encode("ascii")

    for path, description in (
        (identity_file, "external private-key path"),
        (known_hosts_file, "external known-hosts path"),
        (external_state_root, "external state-root path"),
        (vm_python, "VM Python path"),
        (vm_release_root, "VM release path"),
        (vm_config_path, "VM config path"),
    ):
        _safe_fixed_path(path, description=description)
    config = {
        "schema_version": WATCHDOG_SCHEMA_VERSION,
        "protocol": WATCHDOG_PROTOCOL,
        "release_id": RELEASE_ID,
        "git_commit": git_commit,
        "release_tag_object": tag_object,
        "control_sha256": control_sha256,
        "remote": {
            "host": remote_host,
            "user": remote_user,
            "identity_file": str(identity_file),
            "known_hosts_file": str(known_hosts_file),
        },
        "state_root": str(external_state_root),
        "liveness_email": liveness_email,
        "interval_seconds": WATCHDOG_INTERVAL_SECONDS,
        "observation_gap_seconds": WATCHDOG_OBSERVATION_GAP_SECONDS,
        "stale_seconds": 600,
    }
    root = Path(os.path.abspath(os.fspath(output_root.expanduser())))
    if root.exists():
        root = _canonical_path(root, description="bundle root", kind="directory")
    else:
        root.mkdir(parents=True, mode=0o700)
        root = _canonical_path(root, description="bundle root", kind="directory")
    (
        runtime_root,
        runtime_records,
        runtime_inventory_sha256,
        runtime_total_bytes,
    ) = _copy_runtime_tree(release_root=release_root, bundle_root=root)
    service, timer = _render_systemd_units(
        vm_python=vm_python,
        vm_release_root=vm_release_root,
        vm_config_path=vm_config_path,
        service_user=service_user,
        external_state_root=external_state_root,
    )
    payloads = {
        "watchdog.json": _canonical(config),
        SERVICE_NAME: service,
        TIMER_NAME: timer,
        "authorized_keys.line": authorized_line,
    }
    files: list[dict[str, Any]] = []
    for name, payload in payloads.items():
        path = root / name
        if path.exists() or path.is_symlink():
            existing = _stable_bytes(path, description=f"bundle {name}")
            if existing != payload:
                raise DeploymentError(f"bundle file conflicts: {path}")
        else:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o400,
            )
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        path.chmod(0o444)
        files.append(
            {
                "name": name,
                "path": name,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "protocol": BUNDLE_PROTOCOL,
        **_release_fields(git_commit, tag_object),
        "control_sha256": control_sha256,
        "immutable_release_sha256": release_digest,
        "watchdog_code_sha256": hashlib.sha256(
            _canonical(code_records)
        ).hexdigest(),
        "code_records": code_records,
        "runtime_root": runtime_root.relative_to(root).as_posix(),
        "runtime_inventory": runtime_records,
        "runtime_inventory_sha256": runtime_inventory_sha256,
        "runtime_file_count": len(runtime_records),
        "runtime_total_bytes": runtime_total_bytes,
        "cluster_harness_runtime": harness_runtime,
        "vm_python": str(vm_python),
        "vm_release_root": str(vm_release_root),
        "forced_command_only": True,
        "timer_seconds": 300,
        "files": files,
    }
    manifest["bundle_id"] = _self_hash(manifest, "bundle_id")
    manifest_path, manifest_sha = _publish_once(
        root / "BUNDLE.json", manifest
    )
    return {
        "passed": True,
        "bundle_root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "bundle_id": manifest["bundle_id"],
    }


def _validate_bundle(path: Path) -> tuple[dict[str, Any], bytes]:
    value, raw = _sealed_json(path, description="watchdog bundle manifest")
    required = {
        "schema_version",
        "protocol",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "control_sha256",
        "immutable_release_sha256",
        "watchdog_code_sha256",
        "code_records",
        "runtime_root",
        "runtime_inventory",
        "runtime_inventory_sha256",
        "runtime_file_count",
        "runtime_total_bytes",
        "cluster_harness_runtime",
        "vm_python",
        "vm_release_root",
        "forced_command_only",
        "timer_seconds",
        "files",
        "bundle_id",
    }
    if (
        set(value) != required
        or value.get("schema_version") != 3
        or value.get("protocol") != BUNDLE_PROTOCOL
        or value.get("release_id") != RELEASE_ID
        or value.get("release_tag") != RELEASE_TAG
        or value.get("chain_namespace") != CHAIN_NAMESPACE
        or GIT_OBJECT.fullmatch(
            str(value.get("release_git_commit", ""))
        )
        is None
        or GIT_OBJECT.fullmatch(
            str(value.get("release_tag_object", ""))
        )
        is None
        or any(
            SHA256.fullmatch(str(value.get(field, ""))) is None
            for field in (
                "control_sha256",
                "immutable_release_sha256",
                "watchdog_code_sha256",
                "runtime_inventory_sha256",
                "bundle_id",
            )
        )
        or value.get("forced_command_only") is not True
        or value.get("timer_seconds") != 300
        or value.get("bundle_id") != _self_hash(value, "bundle_id")
    ):
        raise DeploymentError("watchdog bundle manifest is invalid")
    harness_runtime = value.get("cluster_harness_runtime")
    lexical_path = Path(
        str(harness_runtime.get("lexical_path", ""))
        if isinstance(harness_runtime, Mapping)
        else ""
    )
    resolved_path = Path(
        str(harness_runtime.get("resolved_path", ""))
        if isinstance(harness_runtime, Mapping)
        else ""
    )
    prefix_path = Path(
        str(harness_runtime.get("prefix", ""))
        if isinstance(harness_runtime, Mapping)
        else ""
    )
    manifest_path = Path(
        str(harness_runtime.get("manifest_path", ""))
        if isinstance(harness_runtime, Mapping)
        else ""
    )
    lexical_entry = (
        harness_runtime.get("lexical_entry")
        if isinstance(harness_runtime, Mapping)
        else None
    )
    resolved_entry = (
        harness_runtime.get("resolved_entry")
        if isinstance(harness_runtime, Mapping)
        else None
    )
    if (
        not isinstance(harness_runtime, Mapping)
        or set(harness_runtime)
        != {
            "prefix",
            "lexical_path",
            "resolved_path",
            "manifest_path",
            "manifest_sha256",
            "inventory_sha256",
            "lexical_entry",
            "resolved_entry",
        }
        or any(
            not isinstance(harness_runtime.get(field), str)
            or not Path(str(harness_runtime[field])).is_absolute()
            for field in (
                "prefix",
                "lexical_path",
                "resolved_path",
                "manifest_path",
            )
        )
        or any(
            SHA256.fullmatch(str(harness_runtime.get(field, ""))) is None
            for field in ("manifest_sha256", "inventory_sha256")
        )
        or not isinstance(harness_runtime.get("lexical_entry"), Mapping)
        or not isinstance(harness_runtime.get("resolved_entry"), Mapping)
        or lexical_path != prefix_path / "bin" / "python"
        or resolved_path == prefix_path
        or prefix_path not in resolved_path.parents
        or not isinstance(lexical_entry, Mapping)
        or lexical_entry.get("path") != "bin/python"
        or lexical_entry.get("type") not in {"file", "symlink"}
        or not isinstance(resolved_entry, Mapping)
        or resolved_entry.get("path")
        != resolved_path.relative_to(prefix_path).as_posix()
        or resolved_entry.get("type") != "file"
        or not isinstance(resolved_entry.get("mode"), int)
        or isinstance(resolved_entry.get("mode"), bool)
        or int(resolved_entry["mode"]) & 0o222
        or int(resolved_entry["mode"]) & 0o111 == 0
        or not isinstance(resolved_entry.get("size"), int)
        or isinstance(resolved_entry.get("size"), bool)
        or int(resolved_entry["size"]) <= 0
        or SHA256.fullmatch(str(resolved_entry.get("sha256", ""))) is None
    ):
        raise DeploymentError(
            "watchdog bundle harness-runtime binding is invalid"
        )
    records = value.get("files")
    if (
        not isinstance(records, list)
        or {record.get("name") for record in records if isinstance(record, dict)}
        != {
            "watchdog.json",
            SERVICE_NAME,
            TIMER_NAME,
            "authorized_keys.line",
        }
    ):
        raise DeploymentError("watchdog bundle file inventory is incomplete")
    file_payloads: dict[str, bytes] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"name", "path", "size", "sha256"}
            or Path(str(record["path"])).is_absolute()
            or ".." in Path(str(record["path"])).parts
            or str(record["path"]) != str(record["name"])
            or not isinstance(record["size"], int)
            or isinstance(record["size"], bool)
            or record["size"] <= 0
            or SHA256.fullmatch(str(record["sha256"])) is None
        ):
            raise DeploymentError("watchdog bundle file record is malformed")
        payload = _stable_bytes(
            path.parent / Path(str(record["path"])),
            description=f"bundle file {record['name']}",
            require_read_only=True,
            require_single_link=True,
        )
        if (
            len(payload) != record["size"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise DeploymentError("watchdog bundle file bytes drifted")
        file_payloads[str(record["name"])] = payload
    try:
        authorized_line = file_payloads["authorized_keys.line"].decode("ascii")
    except (KeyError, UnicodeError) as exc:
        raise DeploymentError(
            "watchdog bundle authorized-keys line is not ASCII"
        ) from exc
    command_bindings = (
        f'restrict,command="{resolved_path} -I -u ',
        f"--harness-python {lexical_path}",
        f"--resolved-harness-python {resolved_path}",
        f"--harness-environment-manifest {manifest_path}",
        "--harness-environment-sha256 "
        + str(harness_runtime["manifest_sha256"]),
        "--control-sha256 " + str(value["control_sha256"]),
    )
    if (
        not authorized_line.endswith("\n")
        or authorized_line.count("\n") != 1
        or not all(fragment in authorized_line for fragment in command_bindings)
    ):
        raise DeploymentError(
            "watchdog authorized-keys line does not bind the pinned harness target"
        )
    relative_runtime_root = Path(str(value.get("runtime_root", "")))
    if (
        relative_runtime_root.is_absolute()
        or ".." in relative_runtime_root.parts
        or relative_runtime_root.as_posix() != "release"
    ):
        raise DeploymentError("bundled watchdog runtime root is not relocatable")
    runtime_root = _canonical_path(
        path.parent / relative_runtime_root,
        description="bundled watchdog runtime root",
        kind="directory",
    )
    runtime_records = value.get("runtime_inventory")
    expected_runtime_paths = sorted(
        path.as_posix() for path in WATCHDOG_RUNTIME_PATHS
    )
    if (
        not isinstance(runtime_records, list)
        or len(runtime_records) != len(expected_runtime_paths)
        or value.get("runtime_file_count") != len(runtime_records)
        or not isinstance(value.get("runtime_total_bytes"), int)
        or isinstance(value.get("runtime_total_bytes"), bool)
        or value["runtime_total_bytes"] <= 0
        or sorted(
            str(record.get("path"))
            for record in runtime_records
            if isinstance(record, dict)
        )
        != expected_runtime_paths
    ):
        raise DeploymentError("watchdog runtime inventory is incomplete")
    runtime_total = 0
    for record in runtime_records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size", "sha256"}
            or not isinstance(record.get("size"), int)
            or isinstance(record.get("size"), bool)
            or int(record["size"]) <= 0
            or SHA256.fullmatch(str(record.get("sha256", ""))) is None
        ):
            raise DeploymentError("watchdog runtime inventory record is malformed")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise DeploymentError("watchdog runtime inventory path is unsafe")
        payload = _stable_bytes(
            runtime_root / relative,
            description=f"bundled watchdog runtime {relative}",
            require_read_only=True,
            require_single_link=True,
        )
        if (
            len(payload) != record["size"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise DeploymentError("bundled watchdog runtime bytes drifted")
        runtime_total += len(payload)
    if (
        runtime_total != value["runtime_total_bytes"]
        or hashlib.sha256(_canonical(runtime_records)).hexdigest()
        != value["runtime_inventory_sha256"]
    ):
        raise DeploymentError("watchdog runtime inventory identity is invalid")
    for field in ("vm_python", "vm_release_root"):
        raw_path = Path(str(value.get(field, "")))
        if (
            not raw_path.is_absolute()
            or _safe_fixed_path(raw_path, description=field) != str(raw_path)
        ):
            raise DeploymentError(f"watchdog {field} path is unsafe")
    return value, raw


def _verify_installed_runtime(
    bundle: Mapping[str, Any], *, installed_release_root: Path
) -> list[dict[str, Any]]:
    root = _canonical_path(
        installed_release_root,
        description="installed watchdog release root",
        kind="directory",
    )
    if str(root) != str(bundle["vm_release_root"]):
        raise DeploymentError(
            "installed watchdog release root differs from the sealed service path"
        )
    observed_paths: list[str] = []
    for directory, directories, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in directories:
            child = parent / name
            if child.is_symlink():
                raise DeploymentError(
                    f"installed watchdog runtime contains symlinked directory: {child}"
                )
        for name in filenames:
            child = parent / name
            if child.is_symlink() or not child.is_file():
                raise DeploymentError(
                    f"installed watchdog runtime contains unsafe file: {child}"
                )
            observed_paths.append(child.relative_to(root).as_posix())
    expected = [str(row["path"]) for row in bundle["runtime_inventory"]]
    if sorted(observed_paths) != sorted(expected):
        raise DeploymentError(
            "installed watchdog runtime file set differs from the sealed bundle"
        )
    inspected: list[dict[str, Any]] = []
    by_path = {
        str(record["path"]): record for record in bundle["runtime_inventory"]
    }
    for relative in sorted(expected):
        path = root / relative
        raw = _stable_bytes(
            path,
            description=f"installed watchdog runtime {relative}",
            require_read_only=True,
            require_single_link=True,
        )
        record = by_path[relative]
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != record["size"] or digest != record["sha256"]:
            raise DeploymentError(
                f"installed watchdog runtime differs from bundle: {relative}"
            )
        inspected.append(
            {
                "path": relative,
                "size": len(raw),
                "sha256": digest,
            }
        )
    if hashlib.sha256(_canonical(inspected)).hexdigest() != bundle[
        "runtime_inventory_sha256"
    ]:
        raise DeploymentError("installed watchdog runtime inventory hash drifted")
    return inspected


def _validate_heartbeat(
    path: Path,
    *,
    bundle: Mapping[str, Any],
    description: str,
) -> tuple[dict[str, Any], str]:
    value, raw = _sealed_json(path, description=description)
    required = {
        "schema_version",
        "protocol",
        "observed_timestamp",
        "release_id",
        "git_commit",
        "release_tag_object",
        "control_sha256",
        "first_status_sha256",
        "second_status_sha256",
        "action",
        "reason",
        "desired_state",
        "finalization_state",
        "action_result",
        "heartbeat_id",
    }
    identity = dict(value)
    heartbeat_id = identity.pop("heartbeat_id", None)
    if (
        set(value) != required
        or value.get("schema_version") != 1
        or value.get("protocol") != HEARTBEAT_PROTOCOL
        or value.get("release_id") != bundle["release_id"]
        or value.get("git_commit") != bundle["release_git_commit"]
        or value.get("release_tag_object") != bundle["release_tag_object"]
        or value.get("control_sha256") != bundle["control_sha256"]
        or heartbeat_id != hashlib.sha256(_canonical(identity)).hexdigest()
        or any(
            SHA256.fullmatch(str(value.get(field, ""))) is None
            for field in ("first_status_sha256", "second_status_sha256")
        )
        or value.get("action")
        not in {None, "repair-chain", "finalizer-reconcile"}
        or not isinstance(value.get("reason"), str)
        or value.get("desired_state") not in {"paused", "running", "resuming"}
        or value.get("finalization_state")
        not in {
            "idle",
            "requested",
            "draining",
            "validating",
            "retiring_fleet",
            "snapshotting",
            "complete",
            "blocked",
        }
        or not isinstance(value.get("observed_timestamp"), (int, float))
        or isinstance(value.get("observed_timestamp"), bool)
    ):
        raise DeploymentError(f"{description} identity is invalid")
    return value, hashlib.sha256(raw).hexdigest()


def _systemctl_value(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    *,
    unit: str,
    property_name: str,
) -> str:
    proc = runner(
        [
            "systemctl",
            "show",
            f"--property={property_name}",
            "--value",
            unit,
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise DeploymentError(
            f"cannot read systemd {unit} {property_name}: {proc.stderr[:500]}"
        )
    return proc.stdout.strip()


def capture_deployment_evidence(
    *,
    bundle_manifest: Path,
    installed_release_root: Path,
    vm_python: Path,
    installed_config: Path,
    installed_service: Path,
    installed_timer: Path,
    installed_authorized_keys: Path,
    service_heartbeat: Path,
    output: Path,
    systemctl_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    probe_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Verify exact runtime bytes, one successful service run, then seal evidence."""

    bundle, bundle_raw = _validate_bundle(bundle_manifest)
    runtime_inventory = _verify_installed_runtime(
        bundle, installed_release_root=installed_release_root
    )
    canonical_python = _canonical_path(
        vm_python, description="installed watchdog Python", kind="file"
    )
    if str(canonical_python) != str(bundle["vm_python"]) or not os.access(
        canonical_python, os.X_OK
    ):
        raise DeploymentError(
            "installed watchdog Python differs from the sealed service interpreter"
        )
    python_raw = _stable_bytes(
        canonical_python,
        description="installed watchdog Python",
        require_read_only=True,
        require_single_link=True,
    )
    watchdog_script = (
        _canonical_path(
            installed_release_root,
            description="installed watchdog release root",
            kind="directory",
        )
        / "scripts"
        / "schema5_external_watchdog.py"
    )
    probe = probe_runner(
        [
            str(canonical_python),
            "-I",
            "-u",
            str(watchdog_script),
            "--help",
        ],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        },
    )
    if (
        probe.returncode != 0
        or "Run one five-minute external schema-5 watchdog transaction"
        not in probe.stdout
    ):
        raise DeploymentError(
            "installed watchdog isolated-runtime probe failed: "
            f"rc={probe.returncode}; {probe.stderr[:500]}"
        )
    heartbeat, heartbeat_sha256 = _validate_heartbeat(
        service_heartbeat,
        bundle=bundle,
        description="successful watchdog service heartbeat",
    )
    bundle_root = _canonical_path(
        bundle_manifest.parent,
        description="watchdog bundle root",
        kind="directory",
    )
    by_name = {record["name"]: record for record in bundle["files"]}
    installed = {
        "watchdog.json": installed_config,
        SERVICE_NAME: installed_service,
        TIMER_NAME: installed_timer,
    }
    inspected: list[dict[str, Any]] = []
    for name, path in installed.items():
        raw = _stable_bytes(path, description=f"installed {name}")
        if hashlib.sha256(raw).hexdigest() != by_name[name]["sha256"]:
            raise DeploymentError(f"installed {name} differs from sealed bundle")
        inspected.append(
            {
                "name": name,
                "path": str(_canonical_path(path, description=name, kind="file")),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    try:
        config_value = json.loads(
            _stable_bytes(
                installed_config,
                description="installed watchdog configuration",
            ).decode("utf-8")
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DeploymentError(
            f"installed watchdog configuration is invalid: {exc}"
        ) from exc
    remote = (
        config_value.get("remote")
        if isinstance(config_value, dict)
        else None
    )
    if not isinstance(remote, dict):
        raise DeploymentError("installed watchdog remote configuration is invalid")
    if (
        config_value.get("schema_version") != WATCHDOG_SCHEMA_VERSION
        or config_value.get("protocol") != WATCHDOG_PROTOCOL
        or config_value.get("release_id") != bundle["release_id"]
        or config_value.get("git_commit") != bundle["release_git_commit"]
        or config_value.get("release_tag_object")
        != bundle["release_tag_object"]
        or config_value.get("control_sha256") != bundle["control_sha256"]
        or config_value.get("interval_seconds") != WATCHDOG_INTERVAL_SECONDS
        or config_value.get("observation_gap_seconds")
        != WATCHDOG_OBSERVATION_GAP_SECONDS
        or config_value.get("stale_seconds") != 600
    ):
        raise DeploymentError(
            "installed watchdog configuration release identity is invalid"
        )
    identity_path = _canonical_path(
        Path(str(remote.get("identity_file", ""))),
        description="installed watchdog private key",
        kind="file",
    )
    known_hosts_path = _canonical_path(
        Path(str(remote.get("known_hosts_file", ""))),
        description="installed watchdog known_hosts",
        kind="file",
    )
    identity_raw = _stable_bytes(
        identity_path,
        description="installed watchdog private key",
        require_single_link=True,
    )
    known_hosts_raw = _stable_bytes(
        known_hosts_path,
        description="installed watchdog known_hosts",
        require_single_link=True,
    )
    if stat.S_IMODE(identity_path.stat().st_mode) & 0o077:
        raise DeploymentError("installed watchdog private key permissions are broad")
    if stat.S_IMODE(known_hosts_path.stat().st_mode) & 0o022:
        raise DeploymentError("installed watchdog known_hosts is group/world writable")
    _canonical_path(
        Path(str(config_value.get("state_root", ""))),
        description="installed watchdog state root",
        kind="directory",
    )
    inspected.extend(
        [
            {
                "name": "identity_file",
                "path": str(identity_path),
                "sha256": hashlib.sha256(identity_raw).hexdigest(),
            },
            {
                "name": "known_hosts_file",
                "path": str(known_hosts_path),
                "sha256": hashlib.sha256(known_hosts_raw).hexdigest(),
            },
        ]
    )
    authorized = _stable_bytes(
        installed_authorized_keys,
        description="installed authorized_keys",
        require_single_link=True,
    )
    expected_line = _stable_bytes(
        bundle_root / Path(by_name["authorized_keys.line"]["path"]),
        description="bundled authorized_keys line",
        require_read_only=True,
        require_single_link=True,
    )
    if authorized.splitlines(keepends=True).count(expected_line) != 1:
        raise DeploymentError(
            "installed authorized_keys lacks exactly one sealed forced-command line"
        )
    load_state = _systemctl_value(
        systemctl_runner, unit=SERVICE_NAME, property_name="LoadState"
    )
    service_result = _systemctl_value(
        systemctl_runner, unit=SERVICE_NAME, property_name="Result"
    )
    service_status = _systemctl_value(
        systemctl_runner, unit=SERVICE_NAME, property_name="ExecMainStatus"
    )
    if load_state != "loaded":
        raise DeploymentError("watchdog systemd service is not loaded")
    if service_result != "success" or service_status != "0":
        raise DeploymentError(
            "watchdog systemd service has not completed successfully: "
            f"Result={service_result!r}, ExecMainStatus={service_status!r}"
        )
    active = systemctl_runner(
        ["systemctl", "is-active", "--quiet", TIMER_NAME],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if active.returncode != 0:
        raise DeploymentError("watchdog systemd timer is not active")
    enabled = systemctl_runner(
        ["systemctl", "is-enabled", "--quiet", TIMER_NAME],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if enabled.returncode != 0:
        raise DeploymentError("watchdog timer is not enabled")
    deployment_id = hashlib.sha256(
        _canonical(
            {
                "bundle_id": bundle["bundle_id"],
                "bundle_sha256": hashlib.sha256(bundle_raw).hexdigest(),
                "installed": inspected,
                "runtime_inventory": runtime_inventory,
                "runtime_inventory_sha256": bundle[
                    "runtime_inventory_sha256"
                ],
                "vm_python_sha256": hashlib.sha256(python_raw).hexdigest(),
                "service_heartbeat_sha256": heartbeat_sha256,
                "service_heartbeat_id": heartbeat["heartbeat_id"],
                "authorized_keys_sha256": hashlib.sha256(authorized).hexdigest(),
            }
        )
    ).hexdigest()
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol": DEPLOYMENT_PROTOCOL,
        "passed": True,
        **_release_fields(
            str(bundle["release_git_commit"]),
            str(bundle["release_tag_object"]),
        ),
        "deployment_id": deployment_id,
        "watchdog_code_sha256": bundle["watchdog_code_sha256"],
        "immutable_release_sha256": bundle["immutable_release_sha256"],
        "runtime_inventory_sha256": bundle["runtime_inventory_sha256"],
        "runtime_file_count": bundle["runtime_file_count"],
        "runtime_total_bytes": bundle["runtime_total_bytes"],
        "vm_python_path": str(canonical_python),
        "vm_python_sha256": hashlib.sha256(python_raw).hexdigest(),
        "control_sha256": bundle["control_sha256"],
        "liveness_email": str(config_value["liveness_email"]),
        "forced_command_only": True,
        "timer_seconds": 300,
        "isolated_runtime_probe_passed": True,
        "systemd_service_loaded": True,
        "systemd_service_result": service_result,
        "systemd_service_exec_main_status": int(service_status),
        "successful_service_heartbeat_id": heartbeat["heartbeat_id"],
        "successful_service_heartbeat_sha256": heartbeat_sha256,
        "systemd_timer_active": True,
        "systemd_timer_enabled": True,
    }
    evidence["evidence_id"] = _self_hash(evidence, "evidence_id")
    _publish_once(output, evidence)
    return evidence


def capture_liveness_evidence(
    *,
    deployment_evidence: Path,
    heartbeat_paths: Sequence[Path],
    acknowledgement: Path,
    output: Path,
) -> dict[str, Any]:
    """Seal two independent heartbeats plus an explicit human email acknowledgement."""

    deployment, _ = _sealed_json(
        deployment_evidence, description="watchdog deployment evidence"
    )
    if (
        deployment.get("protocol") != DEPLOYMENT_PROTOCOL
        or deployment.get("evidence_id")
        != _self_hash(deployment, "evidence_id")
    ):
        raise DeploymentError("watchdog deployment evidence is invalid")
    if len(heartbeat_paths) != 2:
        raise DeploymentError("liveness requires exactly two sealed heartbeats")
    heartbeats: list[dict[str, Any]] = []
    heartbeat_raw: list[bytes] = []
    for index, path in enumerate(heartbeat_paths):
        value, raw = _sealed_json(
            path, description=f"watchdog heartbeat {index + 1}"
        )
        required = {
            "schema_version",
            "protocol",
            "observed_timestamp",
            "release_id",
            "git_commit",
            "release_tag_object",
            "control_sha256",
            "first_status_sha256",
            "second_status_sha256",
            "action",
            "reason",
            "desired_state",
            "finalization_state",
            "action_result",
            "heartbeat_id",
        }
        identity = dict(value)
        heartbeat_id = identity.pop("heartbeat_id", None)
        if (
            set(value) != required
            or value.get("schema_version") != 1
            or value.get("protocol") != HEARTBEAT_PROTOCOL
            or value.get("release_id") != RELEASE_ID
            or value.get("git_commit") != deployment["release_git_commit"]
            or value.get("release_tag_object")
            != deployment["release_tag_object"]
            or value.get("control_sha256") != deployment["control_sha256"]
            or heartbeat_id != hashlib.sha256(_canonical(identity)).hexdigest()
            or SHA256.fullmatch(
                str(value.get("first_status_sha256", ""))
            )
            is None
            or SHA256.fullmatch(
                str(value.get("second_status_sha256", ""))
            )
            is None
            or value.get("action")
            not in {None, "repair-chain", "finalizer-reconcile"}
            or not isinstance(value.get("reason"), str)
            or value.get("desired_state")
            not in {"paused", "running", "resuming"}
            or value.get("finalization_state")
            not in {
                "idle",
                "requested",
                "draining",
                "validating",
                "retiring_fleet",
                "snapshotting",
                "complete",
                "blocked",
            }
            or not isinstance(value.get("observed_timestamp"), (int, float))
            or isinstance(value.get("observed_timestamp"), bool)
        ):
            raise DeploymentError("watchdog heartbeat identity is invalid")
        heartbeats.append(value)
        heartbeat_raw.append(raw)
    observations = [
        float(heartbeat["observed_timestamp"]) for heartbeat in heartbeats
    ]
    if observations[1] - observations[0] < 60:
        raise DeploymentError("watchdog heartbeats are not 60 seconds apart")
    ack, _ = _sealed_json(
        acknowledgement, description="watchdog liveness acknowledgement"
    )
    if (
        set(ack)
        != {
            "schema_version",
            "protocol",
            "passed",
            "deployment_id",
            "recipient",
            "heartbeat_ids",
            "acknowledged_at",
            "acknowledged_timestamp",
            "operator",
            "acknowledgement_id",
        }
        or ack.get("schema_version") != 1
        or ack.get("protocol") != ACK_PROTOCOL
        or ack.get("passed") is not True
        or ack.get("deployment_id") != deployment["deployment_id"]
        or ack.get("recipient") != deployment.get("liveness_email")
        or ack.get("heartbeat_ids")
        != [heartbeat["heartbeat_id"] for heartbeat in heartbeats]
        or not isinstance(ack.get("acknowledged_at"), str)
        or not ack["acknowledged_at"]
        or not isinstance(ack.get("acknowledged_timestamp"), (int, float))
        or isinstance(ack.get("acknowledged_timestamp"), bool)
        or float(ack["acknowledged_timestamp"]) < observations[1]
        or not isinstance(ack.get("operator"), str)
        or not ack["operator"].strip()
        or ack.get("acknowledgement_id")
        != _self_hash(ack, "acknowledgement_id")
    ):
        raise DeploymentError("watchdog liveness acknowledgement is invalid")
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol": LIVENESS_PROTOCOL,
        "passed": True,
        **{
            field: deployment[field]
            for field in (
                "release_id",
                "release_tag",
                "release_git_commit",
                "release_tag_object",
                "chain_namespace",
                "deployment_id",
                "watchdog_code_sha256",
                "immutable_release_sha256",
                "control_sha256",
            )
        },
        "scheduler_observations": observations,
        "heartbeat_sha256": hashlib.sha256(
            b"".join(heartbeat_raw)
        ).hexdigest(),
        "liveness_email": deployment["liveness_email"],
        "liveness_email_ack": True,
    }
    evidence["evidence_id"] = _self_hash(evidence, "evidence_id")
    _publish_once(output, evidence)
    return evidence


def acknowledge_liveness_email(
    *,
    deployment_evidence: Path,
    heartbeat_paths: Sequence[Path],
    operator: str,
    output: Path,
    confirm_email_received: bool,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Publish an explicit human acknowledgement for two verified heartbeats."""

    if not confirm_email_received:
        raise DeploymentError(
            "liveness acknowledgement requires --confirm-email-received"
        )
    if not operator.strip() or any(
        character in operator for character in ("\x00", "\n", "\r")
    ):
        raise DeploymentError("liveness acknowledgement operator is invalid")
    deployment, _ = _sealed_json(
        deployment_evidence, description="watchdog deployment evidence"
    )
    if (
        deployment.get("protocol") != DEPLOYMENT_PROTOCOL
        or deployment.get("evidence_id")
        != _self_hash(deployment, "evidence_id")
    ):
        raise DeploymentError("watchdog deployment evidence is invalid")
    if len(heartbeat_paths) != 2:
        raise DeploymentError(
            "liveness acknowledgement requires exactly two heartbeats"
        )
    heartbeats = [
        _validate_heartbeat(
            path,
            bundle=deployment,
            description=f"watchdog heartbeat {index}",
        )[0]
        for index, path in enumerate(heartbeat_paths, start=1)
    ]
    observations = [
        float(heartbeat["observed_timestamp"]) for heartbeat in heartbeats
    ]
    if observations[1] - observations[0] < 60:
        raise DeploymentError("watchdog heartbeats are not 60 seconds apart")
    acknowledged_timestamp = max(float(now()), observations[1])
    acknowledgement: dict[str, Any] = {
        "schema_version": 1,
        "protocol": ACK_PROTOCOL,
        "passed": True,
        "deployment_id": deployment["deployment_id"],
        "recipient": deployment["liveness_email"],
        "heartbeat_ids": [
            heartbeat["heartbeat_id"] for heartbeat in heartbeats
        ],
        "acknowledged_at": datetime.fromtimestamp(
            acknowledged_timestamp, timezone.utc
        ).isoformat(),
        "acknowledged_timestamp": acknowledged_timestamp,
        "operator": operator.strip(),
    }
    acknowledgement["acknowledgement_id"] = _self_hash(
        acknowledgement, "acknowledgement_id"
    )
    _publish_once(output, acknowledgement)
    return acknowledgement


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--output-root", type=Path, required=True)
    bundle.add_argument("--release-root", type=Path, required=True)
    bundle.add_argument("--harness-python", type=Path, required=True)
    bundle.add_argument("--control-state-dir", type=Path, required=True)
    bundle.add_argument("--git-commit", required=True)
    bundle.add_argument("--tag-object", required=True)
    bundle.add_argument("--control-sha256", required=True)
    bundle.add_argument("--remote-host", required=True)
    bundle.add_argument("--remote-user", required=True)
    bundle.add_argument("--identity-file", type=Path, required=True)
    bundle.add_argument("--known-hosts-file", type=Path, required=True)
    bundle.add_argument("--external-state-root", type=Path, required=True)
    bundle.add_argument("--public-key-file", type=Path, required=True)
    bundle.add_argument("--vm-python", type=Path, default=Path("/opt/agents-scaling-watchdog/venv/bin/python"))
    bundle.add_argument("--vm-release-root", type=Path, default=Path("/opt/agents-scaling-watchdog/release"))
    bundle.add_argument("--vm-config-path", type=Path, default=Path("/etc/agents-scaling-watchdog/watchdog.json"))
    bundle.add_argument("--service-user", default="agents-scaling-watchdog")
    bundle.add_argument("--liveness-email", default="mabdel03@mit.edu")
    deployment = subparsers.add_parser("deployment-evidence")
    deployment.add_argument("--bundle-manifest", type=Path, required=True)
    deployment.add_argument(
        "--installed-release-root", type=Path, required=True
    )
    deployment.add_argument("--vm-python", type=Path, required=True)
    deployment.add_argument("--installed-config", type=Path, required=True)
    deployment.add_argument("--installed-service", type=Path, required=True)
    deployment.add_argument("--installed-timer", type=Path, required=True)
    deployment.add_argument("--installed-authorized-keys", type=Path, required=True)
    deployment.add_argument("--service-heartbeat", type=Path, required=True)
    deployment.add_argument("--output", type=Path, required=True)
    liveness = subparsers.add_parser("liveness-evidence")
    liveness.add_argument("--deployment-evidence", type=Path, required=True)
    liveness.add_argument("--heartbeat", type=Path, action="append", required=True)
    liveness.add_argument("--acknowledgement", type=Path, required=True)
    liveness.add_argument("--output", type=Path, required=True)
    acknowledge = subparsers.add_parser("acknowledge-liveness")
    acknowledge.add_argument(
        "--deployment-evidence", type=Path, required=True
    )
    acknowledge.add_argument(
        "--heartbeat", type=Path, action="append", required=True
    )
    acknowledge.add_argument("--operator", required=True)
    acknowledge.add_argument(
        "--confirm-email-received", action="store_true"
    )
    acknowledge.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "bundle":
            result = build_bundle(
                output_root=args.output_root,
                release_root=args.release_root,
                harness_python=args.harness_python,
                control_state_dir=args.control_state_dir,
                git_commit=args.git_commit,
                tag_object=args.tag_object,
                control_sha256=args.control_sha256,
                remote_host=args.remote_host,
                remote_user=args.remote_user,
                identity_file=args.identity_file,
                known_hosts_file=args.known_hosts_file,
                external_state_root=args.external_state_root,
                public_key_file=args.public_key_file,
                vm_python=args.vm_python,
                vm_release_root=args.vm_release_root,
                vm_config_path=args.vm_config_path,
                service_user=args.service_user,
                liveness_email=args.liveness_email,
            )
        elif args.command == "deployment-evidence":
            result = capture_deployment_evidence(
                bundle_manifest=args.bundle_manifest,
                installed_release_root=args.installed_release_root,
                vm_python=args.vm_python,
                installed_config=args.installed_config,
                installed_service=args.installed_service,
                installed_timer=args.installed_timer,
                installed_authorized_keys=args.installed_authorized_keys,
                service_heartbeat=args.service_heartbeat,
                output=args.output,
            )
        elif args.command == "liveness-evidence":
            result = capture_liveness_evidence(
                deployment_evidence=args.deployment_evidence,
                heartbeat_paths=args.heartbeat,
                acknowledgement=args.acknowledgement,
                output=args.output,
            )
        elif args.command == "acknowledge-liveness":
            result = acknowledge_liveness_email(
                deployment_evidence=args.deployment_evidence,
                heartbeat_paths=args.heartbeat,
                operator=args.operator,
                output=args.output,
                confirm_email_received=args.confirm_email_received,
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (DeploymentError, OSError, ValueError) as exc:
        print(f"[schema5-watchdog-deployment] ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
