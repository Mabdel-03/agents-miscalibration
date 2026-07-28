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
import binascii
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
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
RELEASE_TAG = "sweep-recovery-schema5-v1.2-r5"
CHAIN_NAMESPACE = "schema5-v1.2-r5"
DEPLOYMENT_PROTOCOL = "schema5-external-watchdog-deployment-evidence-v1"
LIVENESS_PROTOCOL = "schema5-external-watchdog-liveness-evidence-v1"
BUNDLE_PROTOCOL = "schema5-external-watchdog-deployment-bundle-v3"
BOOTSTRAP_BUNDLE_PROTOCOL = (
    "schema5-v1.2-r5-bootstrap-watchdog-bundle-v1"
)
BOOTSTRAP_DEPLOYMENT_EVIDENCE_PROTOCOL = (
    "schema5-v1.2-r5-bootstrap-watchdog-deployment-evidence-v1"
)
BOOTSTRAP_DRILL_EVIDENCE_PROTOCOL = (
    "schema5-v1.2-r5-bootstrap-watchdog-drill-evidence-v1"
)
BOOTSTRAP_ATTESTATION_PROTOCOL = (
    "schema5-v1.2-r5-bootstrap-watchdog-attestation-v1"
)
BOOTSTRAP_SERVICE_NAME = (
    "agents-scaling-schema5-bootstrap-watchdog.service"
)
BOOTSTRAP_TIMER_NAME = "agents-scaling-schema5-bootstrap-watchdog.timer"
BOOTSTRAP_HEARTBEAT_MAX_AGE_SECONDS = 600
BOOTSTRAP_ISOLATED_DRILL_DIRECTORY = "isolated_cancellation_drill"
BOOTSTRAP_CHAIN_MANIFEST_NAME = "RECOVERY_CHAIN_SCHEMA5_V1_2_R5.json"
BOOTSTRAP_SUBMISSION_RECEIPT_NAME = (
    "RECOVERY_CHAIN_SCHEMA5_V1_2_R5_SUBMISSION.json"
)
BOOTSTRAP_CANONICAL_REPAIR_DIRECTORY = "recovery_chain_repairs_v1_2_r5"
BOOTSTRAP_CANONICAL_ROOT_RELEASE_MARKER = (
    "RECOVERY_CHAIN_ROOT_RELEASE_COMPLETE.json"
)
BOOTSTRAP_CANONICAL_LAUNCH_MARKER = (
    "RECOVERY_CHAIN_SCHEMA5_V1_2_R5_LAUNCHED.json"
)
ACK_PROTOCOL = "schema5-external-watchdog-liveness-ack-v1"
HEARTBEAT_PROTOCOL = "schema5-v1.2-r5-external-watchdog-v1"
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
PUBLIC_KEY_ALGORITHM = re.compile(
    r"(?:ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(?:256|384|521))\Z"
)
WATCHDOG_RUNTIME_PATHS = (
    Path("scripts/build_schema5_watchdog_deployment.py"),
    Path("scripts/schema5_external_watchdog.py"),
    Path("src/agents_scaling/serving/external_watchdog.py"),
)
BOOTSTRAP_TAGGED_SOURCE_ROOTS = (Path("scripts"), Path("src"))


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


def _public_key_identity(
    raw: bytes, *, description: str
) -> tuple[str, bytes, str]:
    """Return the SSH algorithm, decoded blob, and canonical public-key text."""

    try:
        text = raw.decode("ascii").strip()
    except UnicodeError as exc:
        raise DeploymentError(f"{description} is not ASCII") from exc
    match = PUBLIC_KEY.fullmatch(text)
    if match is None:
        raise DeploymentError(f"{description} has an unsupported format")
    try:
        blob = base64.b64decode(match.group(2), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise DeploymentError(f"{description} payload is invalid base64") from exc
    if len(blob) < 4:
        raise DeploymentError(f"{description} payload is truncated")
    algorithm = match.group(1)
    return algorithm, blob, f"{algorithm} {match.group(2)}"


def _authorized_key_identity(
    line: bytes, *, description: str
) -> tuple[str, bytes]:
    """Parse one authorized_keys line without treating its comment as identity."""

    try:
        tokens = shlex.split(line.decode("ascii").strip(), posix=True)
    except (UnicodeError, ValueError) as exc:
        raise DeploymentError(f"{description} is malformed") from exc
    if not tokens:
        raise DeploymentError(f"{description} is empty")
    key_index = 0 if PUBLIC_KEY_ALGORITHM.fullmatch(tokens[0]) else 1
    if (
        key_index >= len(tokens)
        or PUBLIC_KEY_ALGORITHM.fullmatch(tokens[key_index]) is None
        or key_index + 1 >= len(tokens)
    ):
        raise DeploymentError(f"{description} lacks a supported SSH public key")
    algorithm = tokens[key_index]
    try:
        blob = base64.b64decode(tokens[key_index + 1], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise DeploymentError(
            f"{description} public-key payload is invalid base64"
        ) from exc
    if len(blob) < 4:
        raise DeploymentError(f"{description} public-key payload is truncated")
    return algorithm, blob


def _require_single_authorized_key_identity(
    installed: bytes, expected_line: bytes, *, description: str
) -> None:
    """Require one exact restricted line and no alias of its key identity."""

    expected_identity = _authorized_key_identity(
        expected_line, description=f"expected {description}"
    )
    exact_count = installed.splitlines(keepends=True).count(expected_line)
    identity_count = 0
    for index, line in enumerate(installed.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(b"#"):
            continue
        try:
            identity = _authorized_key_identity(
                stripped, description=f"{description} line {index}"
            )
        except DeploymentError:
            # An unrelated supported authorized_keys syntax is outside this
            # bundle's scope. A line that names our key type/blob, however, must
            # parse exactly so an unrestricted alias cannot hide in malformed
            # options.
            try:
                text = stripped.decode("ascii")
            except UnicodeError:
                continue
            if (
                expected_line.decode("ascii").split()[-2] in text
                and expected_line.decode("ascii").split()[-1] in text
            ):
                raise
            continue
        if identity == expected_identity:
            identity_count += 1
    if exact_count != 1 or identity_count != 1:
        raise DeploymentError(
            f"{description} must contain exactly one restricted instance of "
            "the sealed SSH key identity"
        )


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


def _bootstrap_tagged_source_records(
    release_root: Path,
) -> list[dict[str, Any]]:
    """Hash a conservative closure of every tagged file bootstrap can execute.

    Bootstrap enters through the forced command, imports the deployment verifier,
    and then executes the recovery renderer.  Those modules contain both ordinary
    and function-local imports.  Binding every Python source below ``scripts`` and
    ``src`` is intentionally stronger than trying to predict which conditional
    repair path will execute.  Binding non-``.py`` files as well covers cached
    bytecode and any data-driven import/runtime path; adding or replacing any
    tagged file is a fail-closed change to the source inventory.
    """

    root = _canonical_path(
        release_root,
        description="bootstrap tagged release root",
        kind="directory",
    )
    records: list[dict[str, Any]] = []
    for relative_root in BOOTSTRAP_TAGGED_SOURCE_ROOTS:
        source_root = _canonical_path(
            root / relative_root,
            description=f"bootstrap tagged source root {relative_root}",
            kind="directory",
        )
        for current, directory_names, file_names in os.walk(
            source_root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            current_info = current_path.stat(follow_symlinks=False)
            if (
                current_path.is_symlink()
                or not stat.S_ISDIR(current_info.st_mode)
                or stat.S_IMODE(current_info.st_mode) & 0o222
            ):
                raise DeploymentError(
                    f"bootstrap tagged source directory is mutable or unsafe: "
                    f"{current_path}"
                )
            directory_names.sort()
            file_names.sort()
            for name in directory_names:
                child = current_path / name
                child_info = child.stat(follow_symlinks=False)
                if child.is_symlink() or not stat.S_ISDIR(child_info.st_mode):
                    raise DeploymentError(
                        f"bootstrap tagged source tree contains an unsafe "
                        f"directory entry: {child}"
                    )
            for name in file_names:
                source = current_path / name
                relative = source.relative_to(root)
                raw = _stable_bytes(
                    source,
                    description=f"bootstrap tagged source {relative}",
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
    records.sort(key=lambda item: str(item["path"]))
    required = {
        "scripts/schema5_bootstrap_watchdog.py",
        "scripts/schema5_watchdog_forced_command.py",
        "scripts/render_schema5_recovery_chain_v12.py",
        "scripts/build_schema5_watchdog_deployment.py",
        "src/agents_scaling/serving/external_watchdog.py",
    }
    observed = {str(item["path"]) for item in records}
    missing = sorted(required - observed)
    if missing:
        raise DeploymentError(
            "bootstrap tagged-source execution closure is incomplete: "
            + ", ".join(missing)
        )
    return records


def _sealed_source_tree_sha256(root: Path) -> str:
    """Recompute ``schema5_control.sha256_tree`` from sealed source bytes."""

    root = _canonical_path(
        root, description="bootstrap pilot release worktree", kind="directory"
    )
    ignored_names = {".git", ".pytest_cache", "__pycache__"}
    paths: list[Path] = []
    for current, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        parent = Path(current)
        parent_info = parent.stat(follow_symlinks=False)
        if (
            parent.is_symlink()
            or not stat.S_ISDIR(parent_info.st_mode)
            or stat.S_IMODE(parent_info.st_mode) & 0o222
        ):
            raise DeploymentError(
                f"bootstrap pilot release directory is mutable or unsafe: {parent}"
            )
        directory_names[:] = sorted(
            name for name in directory_names if name not in ignored_names
        )
        file_names.sort()
        for name in directory_names:
            child = parent / name
            child_info = child.stat(follow_symlinks=False)
            if child.is_symlink() or not stat.S_ISDIR(child_info.st_mode):
                raise DeploymentError(
                    "bootstrap pilot release contains an unsafe directory "
                    f"entry: {child}"
                )
        for name in file_names:
            path = parent / name
            relative = path.relative_to(root)
            if (
                any(part in ignored_names for part in relative.parts)
                or path.suffix == ".pyc"
            ):
                continue
            info = path.stat(follow_symlinks=False)
            if path.is_symlink():
                target = os.readlink(path)
                if any(character in target for character in ("\x00", "\n", "\r")):
                    raise DeploymentError(
                        f"bootstrap pilot release symlink is unsafe: {path}"
                    )
            elif (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) & 0o222
            ):
                raise DeploymentError(
                    f"bootstrap pilot release file is mutable or unsafe: {path}"
                )
            paths.append(path)
    digest = hashlib.sha256()
    for path in sorted(
        paths, key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if path.is_symlink():
            payload = ("SYMLINK\0" + os.readlink(path)).encode("utf-8")
        else:
            payload = _stable_bytes(
                path,
                description=f"bootstrap pilot release {path.relative_to(root)}",
                require_read_only=True,
                require_single_link=True,
            )
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _bootstrap_pilot_release_binding(
    *,
    pilot: Mapping[str, Any],
    pilot_marker_path: Path,
    release_root: Path,
    git_commit: str,
    tag_object: str,
) -> dict[str, Any]:
    """Authenticate the exact r5 pilot layout and sealed source bundle."""

    pilot_root = pilot_marker_path.parent
    materialization_root = pilot_root / "materialization"
    expected_layout = {
        "pilot_root": str(pilot_root),
        "environment_capture_root": str(pilot_root / "environment-capture"),
        "materialization_root": str(materialization_root),
        "release_worktree": str(materialization_root / "release-worktree"),
        "harness_prefix": str(materialization_root / "harness-environment"),
        "serving_prefix": str(materialization_root / "serving-environment"),
        "conda_package_cache": str(materialization_root / "conda-package-cache"),
        "release_bundle": str(materialization_root / "release"),
    }
    layout = pilot.get("layout")
    git_identity = pilot.get("git_identity")
    stage_ids = pilot.get("stage_ids")
    if (
        layout != expected_layout
        or release_root != Path(expected_layout["release_worktree"])
        or not isinstance(git_identity, Mapping)
        or set(git_identity)
        != {
            "git_commit",
            "git_tag",
            "git_tag_object",
            "source_tree_sha256",
            "tag_object",
            "tag_object_type",
        }
        or git_identity.get("git_commit") != git_commit
        or git_identity.get("git_tag") != RELEASE_TAG
        or git_identity.get("git_tag_object") != tag_object
        or git_identity.get("tag_object") != tag_object
        or git_identity.get("tag_object_type") != "tag"
        or SHA256.fullmatch(
            str(git_identity.get("source_tree_sha256", ""))
        )
        is None
        or not isinstance(stage_ids, Mapping)
        or set(stage_ids)
        != {"capture_id", "materialization_id", "release_bundle_id"}
        or any(
            SHA256.fullmatch(str(stage_ids.get(field, ""))) is None
            for field in stage_ids
        )
    ):
        raise DeploymentError(
            "bootstrap materialization pilot layout or annotated source "
            "identity is invalid"
        )
    source_tree_sha256 = _sealed_source_tree_sha256(release_root)
    if source_tree_sha256 != git_identity["source_tree_sha256"]:
        raise DeploymentError(
            "bootstrap materialization pilot source tree identity drifted"
        )
    release_bundle = _canonical_path(
        Path(expected_layout["release_bundle"]),
        description="bootstrap pilot release bundle",
        kind="directory",
    )
    if stat.S_IMODE(
        release_bundle.stat(follow_symlinks=False).st_mode
    ) & 0o222:
        raise DeploymentError("bootstrap pilot release bundle is mutable")
    identity_path = release_bundle / "release_identity.schema5-v1.json"
    marker_path = release_bundle / "RELEASE_COMPLETE.json"
    identity, identity_raw = _sealed_json(
        identity_path, description="bootstrap pilot release identity"
    )
    release_marker, release_marker_raw = _sealed_json(
        marker_path, description="bootstrap pilot release completion"
    )
    stored_git = identity.get("git")
    marker_identity = dict(release_marker)
    release_bundle_id = marker_identity.pop("release_bundle_id", None)
    artifacts = release_marker.get("artifacts")
    identity_artifact = (
        artifacts.get(identity_path.name)
        if isinstance(artifacts, Mapping)
        else None
    )
    if (
        identity.get("release_worktree") != str(release_root)
        or identity.get("worktree_sealed_read_only") is not True
        or stored_git
        != {
            "git_commit": git_commit,
            "git_tag": RELEASE_TAG,
            "git_tag_object": tag_object,
            "source_tree_sha256": source_tree_sha256,
        }
        or release_marker.get("complete") is not True
        or release_marker.get("publication_protocol")
        != "fsync_verify_marker_last"
        or release_marker.get("git_commit") != git_commit
        or release_marker.get("source_tree_sha256") != source_tree_sha256
        or release_bundle_id != stage_ids["release_bundle_id"]
        or release_bundle_id
        != hashlib.sha256(_compact_canonical(marker_identity)).hexdigest()
        or identity_artifact
        != {
            "sha256": hashlib.sha256(identity_raw).hexdigest(),
            "size": len(identity_raw),
        }
    ):
        raise DeploymentError(
            "bootstrap pilot sealed release bundle identity is invalid"
        )
    return {
        "pilot_root": str(pilot_root),
        "release_worktree": str(release_root),
        "release_bundle": str(release_bundle),
        "source_tree_sha256": source_tree_sha256,
        "release_bundle_id": release_bundle_id,
        "release_identity_sha256": hashlib.sha256(identity_raw).hexdigest(),
        "release_completion_sha256": hashlib.sha256(
            release_marker_raw
        ).hexdigest(),
    }


def _bootstrap_chain_anchor(
    *,
    manifest_path: Path,
    receipt_path: Path,
    description: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    """Validate one exact held generation-zero recovery transaction."""

    manifest, manifest_raw = _sealed_json(
        manifest_path, description=f"{description} chain manifest"
    )
    receipt, receipt_raw = _sealed_json(
        receipt_path, description=f"{description} submission receipt"
    )
    manifest_identity = dict(manifest)
    chain_id = manifest_identity.pop("chain_id", None)
    receipt_identity = dict(receipt)
    receipt_id = receipt_identity.pop("receipt_id", None)
    manifest_jobs = manifest.get("jobs")
    receipt_jobs = receipt.get("jobs")
    if (
        manifest.get("protocol") != "schema5-v1.2-r5-recovery-chain"
        or manifest.get("release_git_commit") is None
        or manifest.get("release_tag_object") is None
        or chain_id != hashlib.sha256(_canonical(manifest_identity)).hexdigest()
        or receipt.get("protocol")
        != "schema5-v1.2-r5-recovery-chain-submission"
        or receipt_id
        != hashlib.sha256(_canonical(receipt_identity)).hexdigest()
        or receipt.get("chain_id") != chain_id
        or receipt.get("manifest") != str(manifest_path)
        or receipt.get("manifest_sha256")
        != hashlib.sha256(manifest_raw).hexdigest()
        or receipt.get("root_initial_hold") is not True
        or receipt.get("no_requeue") is not True
        or not isinstance(manifest_jobs, list)
        or len(manifest_jobs) != 43
        or not isinstance(receipt_jobs, list)
        or len(receipt_jobs) != len(manifest_jobs)
    ):
        raise DeploymentError(f"{description} held chain identity is invalid")
    for manifest_row, receipt_row in zip(
        manifest_jobs, receipt_jobs, strict=True
    ):
        if (
            not isinstance(manifest_row, Mapping)
            or not isinstance(receipt_row, Mapping)
            or receipt_row.get("name") != manifest_row.get("name")
            or receipt_row.get("dependencies")
            != manifest_row.get("dependencies")
            or receipt_row.get("script") != manifest_row.get("script")
            or receipt_row.get("script_sha256")
            != manifest_row.get("script_sha256")
            or not str(receipt_row.get("job_id", "")).isdigit()
            or not isinstance(receipt_row.get("comment"), str)
        ):
            raise DeploymentError(
                f"{description} held chain scheduler topology drifted"
            )
    return manifest, manifest_raw, receipt, receipt_raw


def _bootstrap_isolated_drill_contract(
    *,
    canonical_manifest_path: Path,
    canonical_manifest: Mapping[str, Any],
    canonical_receipt_path: Path,
    canonical_receipt: Mapping[str, Any],
    isolated_manifest_path: Path,
    isolated_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate and bind a separately rendered sibling recovery namespace."""

    canonical_root = canonical_manifest_path.parent
    isolated_root = canonical_root / BOOTSTRAP_ISOLATED_DRILL_DIRECTORY
    if (
        isolated_manifest_path
        != isolated_root / BOOTSTRAP_CHAIN_MANIFEST_NAME
        or isolated_receipt_path
        != isolated_root / BOOTSTRAP_SUBMISSION_RECEIPT_NAME
    ):
        raise DeploymentError(
            "bootstrap drill manifest/receipt are outside the exact isolated "
            "sibling namespace"
        )
    (
        isolated_manifest,
        isolated_manifest_raw,
        isolated_receipt,
        isolated_receipt_raw,
    ) = _bootstrap_chain_anchor(
        manifest_path=isolated_manifest_path,
        receipt_path=isolated_receipt_path,
        description="isolated bootstrap drill",
    )
    topology_fields = (
        "name",
        "job_name",
        "dependencies",
        "dependency_type",
        "cpus",
        "memory",
        "time_limit",
    )
    canonical_topology = [
        {field: row.get(field) for field in topology_fields}
        for row in canonical_manifest["jobs"]
    ]
    isolated_topology = [
        {field: row.get(field) for field in topology_fields}
        for row in isolated_manifest["jobs"]
    ]
    canonical_ids = {
        str(row["job_id"]) for row in canonical_receipt["jobs"]
    }
    isolated_ids = {
        str(row["job_id"]) for row in isolated_receipt["jobs"]
    }
    canonical_comments = {
        str(row["comment"]) for row in canonical_receipt["jobs"]
    }
    isolated_comments = {
        str(row["comment"]) for row in isolated_receipt["jobs"]
    }
    isolated_prefix = (
        "asys:s5-recovery-v1.2-r5:"
        f"{isolated_manifest['chain_id']}:g0000:"
    )
    canonical_repair_root = (
        canonical_root / BOOTSTRAP_CANONICAL_REPAIR_DIRECTORY
    )
    canonical_markers = (
        canonical_root / BOOTSTRAP_CANONICAL_ROOT_RELEASE_MARKER,
        canonical_root / BOOTSTRAP_CANONICAL_LAUNCH_MARKER,
    )
    if (
        canonical_manifest.get("release_git_commit")
        != isolated_manifest.get("release_git_commit")
        or canonical_manifest.get("release_tag_object")
        != isolated_manifest.get("release_tag_object")
        or canonical_manifest.get("chain_id")
        == isolated_manifest.get("chain_id")
        or canonical_topology != isolated_topology
        or canonical_receipt.get("receipt_id")
        == isolated_receipt.get("receipt_id")
        or canonical_ids & isolated_ids
        or canonical_comments & isolated_comments
        or any(
            not comment.startswith(isolated_prefix)
            for comment in isolated_comments
        )
        or canonical_repair_root.exists()
        or canonical_repair_root.is_symlink()
        or any(path.exists() or path.is_symlink() for path in canonical_markers)
    ):
        raise DeploymentError(
            "bootstrap cancellation drill overlaps or pollutes the canonical "
            "prelaunch recovery namespace"
        )
    contract: dict[str, Any] = {
        "schema_version": 1,
        "protocol": (
            "schema5-v1.2-r5-bootstrap-isolated-cancellation-drill-v1"
        ),
        "isolated_drill_root": str(isolated_root),
        "chain_manifest": str(isolated_manifest_path),
        "chain_manifest_sha256": hashlib.sha256(
            isolated_manifest_raw
        ).hexdigest(),
        "chain_id": isolated_manifest["chain_id"],
        "anchor_submission_receipt": str(isolated_receipt_path),
        "anchor_submission_receipt_sha256": hashlib.sha256(
            isolated_receipt_raw
        ).hexdigest(),
        "anchor_submission_receipt_id": isolated_receipt["receipt_id"],
        "scheduler_comment_prefix": (
            "asys:s5-recovery-v1.2-r5:"
            f"{isolated_manifest['chain_id']}:"
        ),
        "canonical_chain_manifest": str(canonical_manifest_path),
        "canonical_submission_receipt": str(canonical_receipt_path),
        "canonical_submission_receipt_id": canonical_receipt["receipt_id"],
        "canonical_repair_root": str(canonical_repair_root),
        "canonical_root_release_marker": str(canonical_markers[0]),
        "canonical_launch_marker": str(canonical_markers[1]),
        "canonical_job_id_overlap": 0,
        "canonical_comment_overlap": 0,
    }
    contract["isolation_id"] = _self_hash(contract, "isolation_id")
    return contract, isolated_manifest, isolated_receipt


def _observed_directory_inventory(prefix: Path) -> dict[str, Any]:
    """Recompute a sealed environment's complete byte-level inventory."""

    root = _canonical_path(
        prefix, description="sealed harness inventory root", kind="directory"
    )

    def metadata_state() -> list[tuple[Any, ...]]:
        root_info = root.stat(follow_symlinks=False)
        signatures: list[tuple[Any, ...]] = [
            (
                ".",
                "directory",
                root_info.st_mode,
                root_info.st_dev,
                root_info.st_ino,
                root_info.st_nlink,
                root_info.st_size,
                root_info.st_mtime_ns,
                root_info.st_ctime_ns,
                None,
            )
        ]
        for current, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            directory_names.sort()
            file_names.sort()
            for name in [*directory_names, *file_names]:
                child = current_path / name
                info = child.stat(follow_symlinks=False)
                relative = child.relative_to(root).as_posix()
                if stat.S_ISLNK(info.st_mode):
                    kind = "symlink"
                    target: str | None = os.readlink(child)
                elif stat.S_ISDIR(info.st_mode):
                    kind = "directory"
                    target = None
                elif stat.S_ISREG(info.st_mode):
                    kind = "file"
                    target = None
                else:
                    raise DeploymentError(
                        "sealed harness inventory contains a special entry: "
                        f"{child}"
                    )
                signatures.append(
                    (
                        relative,
                        kind,
                        info.st_mode,
                        info.st_dev,
                        info.st_ino,
                        info.st_nlink,
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ctime_ns,
                        target,
                    )
                )
        signatures.sort(key=lambda item: str(item[0]))
        return signatures

    before = metadata_state()
    entries: list[dict[str, Any]] = []
    total_file_bytes = 0
    for current, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        current_info = current_path.stat(follow_symlinks=False)
        if (
            current_path.is_symlink()
            or not stat.S_ISDIR(current_info.st_mode)
            or stat.S_IMODE(current_info.st_mode) & 0o222
        ):
            raise DeploymentError(
                f"sealed harness directory is mutable or unsafe: {current_path}"
            )
        directory_names.sort()
        file_names.sort()
        retained_directories: list[str] = []
        for name in directory_names:
            child = current_path / name
            info = child.stat(follow_symlinks=False)
            relative = child.relative_to(root).as_posix()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                entries.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "mode": mode,
                        "target": os.readlink(child),
                    }
                )
            elif stat.S_ISDIR(info.st_mode):
                if mode & 0o222:
                    raise DeploymentError(
                        f"sealed harness directory is mutable: {child}"
                    )
                entries.append(
                    {"path": relative, "type": "directory", "mode": mode}
                )
                retained_directories.append(name)
            else:
                raise DeploymentError(
                    f"sealed harness inventory contains a special entry: {child}"
                )
        directory_names[:] = retained_directories
        for name in file_names:
            child = current_path / name
            info = child.stat(follow_symlinks=False)
            relative = child.relative_to(root).as_posix()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                entries.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "mode": mode,
                        "target": os.readlink(child),
                    }
                )
            elif stat.S_ISREG(info.st_mode):
                raw = _stable_bytes(
                    child,
                    description=f"sealed harness inventory file {relative}",
                    require_read_only=True,
                    require_single_link=True,
                )
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "mode": mode,
                        "size": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
                total_file_bytes += len(raw)
            else:
                raise DeploymentError(
                    f"sealed harness inventory contains a special entry: {child}"
                )
    entries.sort(key=lambda item: str(item["path"]))
    after = metadata_state()
    if before != after:
        raise DeploymentError(
            "sealed harness environment changed while fully inventoried"
        )
    return {
        "entries": entries,
        "inventory_sha256": hashlib.sha256(
            _compact_canonical(entries)
        ).hexdigest(),
        "entry_count": len(entries),
        "file_count": sum(item["type"] == "file" for item in entries),
        "directory_count": sum(
            item["type"] == "directory" for item in entries
        ),
        "symlink_count": sum(item["type"] == "symlink" for item in entries),
        "total_file_bytes": total_file_bytes,
    }


def _resolve_inventory_pinned_harness_python(
    *,
    control_state_dir: Path | None,
    harness_python: Path,
    control_sha256: str | None,
    require_pristine_paused: bool = True,
    environment_prefix: Path | None = None,
    environment_manifest_path: Path | None = None,
    environment_manifest_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve Conda's ``bin/python`` through the sealed environment inventory.

    The lexical Conda alias is intentionally retained as provenance, but the
    authorized-keys command executes the immutable regular target.  This closes both
    failure modes of accepting an unpinned alias and rejecting every normal Conda
    environment merely because ``bin/python`` is a same-prefix symlink.
    """

    direct_binding = any(
        value is not None
        for value in (
            environment_prefix,
            environment_manifest_path,
            environment_manifest_sha256,
        )
    )
    if direct_binding:
        if (
            control_state_dir is not None
            or control_sha256 is not None
            or environment_prefix is None
            or environment_manifest_path is None
            or environment_manifest_sha256 is None
        ):
            raise DeploymentError(
                "direct harness environment binding is incomplete or mixed "
                "with mutable control"
            )
        prefix = _canonical_path(
            environment_prefix,
            description="sealed harness prefix",
            kind="directory",
        )
        manifest_path = _canonical_path(
            environment_manifest_path,
            description="sealed harness environment manifest",
            kind="file",
        )
        manifest_sha256 = environment_manifest_sha256
    else:
        if control_state_dir is None or control_sha256 is None:
            raise DeploymentError(
                "watchdog harness resolution lacks a sealed binding"
            )
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
    observed_inventory = _observed_directory_inventory(prefix)
    if observed_inventory != dict(inventory):
        raise DeploymentError(
            "sealed harness environment directory inventory drifted"
        )

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
                require_read_only=False,
                require_single_link=False,
            )
            candidate_metadata = candidate.stat(follow_symlinks=False)
            candidate_mode = stat.S_IMODE(candidate_metadata.st_mode)
            same_as_target = target_metadata is not None and (
                candidate_metadata.st_dev,
                candidate_metadata.st_ino,
            ) == (target_metadata.st_dev, target_metadata.st_ino)
            if (
                candidate_metadata.st_uid != os.geteuid()
                or candidate_mode not in {0o600, 0o444}
                or (
                    same_as_target
                    and (
                        candidate_mode != 0o444
                        or candidate_metadata.st_nlink != 2
                    )
                )
                or (
                    not same_as_target
                    and candidate_metadata.st_nlink != 1
                )
            ):
                raise DeploymentError(
                    f"unsafe interrupted publication ownership, mode, or "
                    f"link count: {candidate}"
                )
            if same_as_target:
                if candidate_raw != payload:
                    raise DeploymentError(
                        f"conflicting linked publication artifact: {candidate}"
                    )
                candidate.unlink()
            elif candidate_raw == payload:
                if candidate_mode == 0o600:
                    candidate.chmod(0o444)
                    descriptor = os.open(
                        candidate,
                        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
                    )
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    if (
                        _stable_bytes(
                            candidate,
                            description=(
                                f"sealed interrupted publication "
                                f"{candidate.name}"
                            ),
                            require_read_only=True,
                            require_single_link=True,
                        )
                        != payload
                    ):
                        raise DeploymentError(
                            f"interrupted publication changed while sealed: "
                            f"{candidate}"
                        )
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
    _public_key_algorithm, _public_key_blob, bare_key = _public_key_identity(
        public_key_raw, description="watchdog public key"
    )
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


def build_bootstrap_bundle(
    *,
    output_root: Path,
    release_root: Path,
    harness_python: Path,
    harness_environment_manifest: Path,
    materialization_pilot_marker: Path,
    chain_manifest: Path,
    submission_receipt: Path,
    isolated_drill_chain_manifest: Path,
    isolated_drill_submission_receipt: Path,
    git_commit: str,
    tag_object: str,
    remote_host: str,
    remote_user: str,
    identity_file: Path,
    known_hosts_file: Path,
    external_state_root: Path,
    bootstrap_public_key_file: Path,
    production_public_key_file: Path,
    vm_python: Path = Path(
        "/opt/agents-scaling-bootstrap-watchdog/venv/bin/python"
    ),
    vm_release_root: Path = Path(
        "/opt/agents-scaling-bootstrap-watchdog/release"
    ),
    vm_config_path: Path = Path(
        "/etc/agents-scaling-bootstrap-watchdog/watchdog.json"
    ),
    service_user: str = "agents-scaling-bootstrap-watchdog",
) -> dict[str, Any]:
    """Build the separate, pre-control recovery watchdog bundle.

    Unlike :func:`build_bundle`, this path has no ``control.json`` dependency.
    Its cluster key can select only receipt-bound scheduler status and an
    idempotent recovery-DAG reconstruction operation.
    """

    release_root = _canonical_path(
        release_root, description="bootstrap immutable release", kind="directory"
    )
    harness_python = Path(
        os.path.abspath(os.fspath(harness_python.expanduser()))
    )
    try:
        harness_python.lstat()
    except OSError as exc:
        raise DeploymentError(
            f"bootstrap harness Python is unavailable: {exc}"
        ) from exc
    if any(character in str(harness_python) for character in ("\x00", "\n", "\r")):
        raise DeploymentError(
            "bootstrap harness Python path contains unsafe characters"
        )
    environment_manifest_path = _canonical_path(
        harness_environment_manifest,
        description="bootstrap pilot harness environment manifest",
        kind="file",
    )
    pilot_marker_path = _canonical_path(
        materialization_pilot_marker,
        description="bootstrap materialization pilot completion",
        kind="file",
    )
    manifest_path = _canonical_path(
        chain_manifest, description="bootstrap chain manifest", kind="file"
    )
    receipt_path = _canonical_path(
        submission_receipt,
        description="bootstrap submission receipt",
        kind="file",
    )
    isolated_manifest_path = _canonical_path(
        isolated_drill_chain_manifest,
        description="isolated bootstrap drill chain manifest",
        kind="file",
    )
    isolated_receipt_path = _canonical_path(
        isolated_drill_submission_receipt,
        description="isolated bootstrap drill submission receipt",
        kind="file",
    )
    for path, description in (
        (release_root, "bootstrap immutable release"),
        (
            environment_manifest_path,
            "bootstrap pilot harness environment manifest",
        ),
        (pilot_marker_path, "bootstrap materialization pilot completion"),
        (manifest_path, "bootstrap chain manifest"),
        (receipt_path, "bootstrap submission receipt"),
        (isolated_manifest_path, "isolated bootstrap drill chain manifest"),
        (isolated_receipt_path, "isolated bootstrap drill submission receipt"),
    ):
        if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) & 0o222:
            raise DeploymentError(f"{description} must be read-only")
    pilot_raw = _stable_bytes(
        pilot_marker_path,
        description="bootstrap materialization pilot completion",
        require_read_only=True,
        require_single_link=True,
    )
    try:
        pilot = json.loads(pilot_raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DeploymentError(
            f"bootstrap materialization pilot completion is invalid JSON: {exc}"
        ) from exc
    pilot_identity = dict(pilot) if isinstance(pilot, dict) else {}
    pilot_id = pilot_identity.pop("pilot_id", None)
    pilot_layout = pilot.get("layout") if isinstance(pilot, dict) else None
    if (
        not isinstance(pilot, dict)
        or pilot_raw != _canonical(pilot)
        or pilot.get("kind")
        != "schema5-materialization-pilot-completion"
        or pilot.get("complete") is not True
        or pilot.get("publication_protocol")
        != "intent_first_stage_evidence_pilot_marker_last"
        or pilot.get("expected_tag") != RELEASE_TAG
        or pilot.get("expected_commit") != git_commit
        or pilot.get("git_identity", {}).get("tag_object") != tag_object
        or pilot.get("pilot_root") != str(pilot_marker_path.parent)
        or not isinstance(pilot_layout, Mapping)
        or not isinstance(pilot_id, str)
        or SHA256.fullmatch(pilot_id) is None
        or pilot_id
        != hashlib.sha256(_compact_canonical(pilot_identity)).hexdigest()
    ):
        raise DeploymentError(
            "bootstrap materialization pilot completion identity is invalid"
        )
    pilot_release_binding = _bootstrap_pilot_release_binding(
        pilot=pilot,
        pilot_marker_path=pilot_marker_path,
        release_root=release_root,
        git_commit=git_commit,
        tag_object=tag_object,
    )
    pilot_prefix = _canonical_path(
        Path(str(pilot_layout["harness_prefix"])),
        description="bootstrap pilot harness prefix",
        kind="directory",
    )
    pilot_release_bundle = _canonical_path(
        Path(str(pilot_layout["release_bundle"])),
        description="bootstrap pilot release bundle",
        kind="directory",
    )
    if environment_manifest_path.parent != pilot_release_bundle:
        raise DeploymentError(
            "bootstrap harness manifest is outside the sealed pilot release"
        )
    environment_manifest_raw = _stable_bytes(
        environment_manifest_path,
        description="bootstrap pilot harness environment manifest",
        require_read_only=True,
        require_single_link=True,
    )
    resolved_python, harness_binding = (
        _resolve_inventory_pinned_harness_python(
            control_state_dir=None,
            harness_python=harness_python,
            control_sha256=None,
            require_pristine_paused=False,
            environment_prefix=pilot_prefix,
            environment_manifest_path=environment_manifest_path,
            environment_manifest_sha256=hashlib.sha256(
                environment_manifest_raw
            ).hexdigest(),
        )
    )
    manifest, manifest_raw, receipt, receipt_raw = _bootstrap_chain_anchor(
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        description="canonical bootstrap",
    )
    if (
        manifest.get("release_git_commit") != git_commit
        or manifest.get("release_tag_object") != tag_object
    ):
        raise DeploymentError(
            "bootstrap bundle does not bind the exact held 43-job transaction"
        )
    (
        isolated_drill_contract,
        _isolated_manifest,
        _isolated_receipt,
    ) = _bootstrap_isolated_drill_contract(
        canonical_manifest_path=manifest_path,
        canonical_manifest=manifest,
        canonical_receipt_path=receipt_path,
        canonical_receipt=receipt,
        isolated_manifest_path=isolated_manifest_path,
        isolated_receipt_path=isolated_receipt_path,
    )
    bootstrap_key = _stable_bytes(
        bootstrap_public_key_file,
        description="bootstrap watchdog public key",
        require_single_link=True,
    ).strip()
    production_key = _stable_bytes(
        production_public_key_file,
        description="production watchdog public key",
        require_single_link=True,
    ).strip()
    (
        bootstrap_key_algorithm,
        bootstrap_key_blob,
        bootstrap_bare_key,
    ) = _public_key_identity(
        bootstrap_key, description="bootstrap watchdog public key"
    )
    production_key_algorithm, production_key_blob, _production_bare_key = (
        _public_key_identity(
            production_key, description="production watchdog public key"
        )
    )
    if (bootstrap_key_algorithm, bootstrap_key_blob) == (
        production_key_algorithm,
        production_key_blob,
    ):
        raise DeploymentError(
            "bootstrap and production watchdogs require distinct SSH keys"
        )
    fixed_release = _safe_fixed_path(
        release_root, description="bootstrap release root"
    )
    code_records = _bootstrap_tagged_source_records(release_root)
    code_by_path = {
        str(record["path"]): record for record in code_records
    }
    watchdog_code_sha256 = hashlib.sha256(
        _canonical(code_records)
    ).hexdigest()
    fixed_python = _safe_fixed_path(
        resolved_python, description="bootstrap resolved harness Python"
    )
    forced_script = _safe_fixed_path(
        release_root / "scripts" / "schema5_watchdog_forced_command.py",
        description="bootstrap forced-command script",
    )
    forced_argv = [
        fixed_python,
        "-I",
        "-u",
        forced_script,
        "--release-root",
        fixed_release,
        "--harness-python",
        str(harness_python),
        "--resolved-harness-python",
        fixed_python,
        "--harness-environment-manifest",
        str(environment_manifest_path),
        "--harness-environment-sha256",
        hashlib.sha256(environment_manifest_raw).hexdigest(),
        "--materialization-pilot-marker",
        str(pilot_marker_path),
        "--materialization-pilot-sha256",
        hashlib.sha256(pilot_raw).hexdigest(),
        "--materialization-pilot-id",
        pilot_id,
        "--renderer-sha256",
        code_by_path[
            "scripts/render_schema5_recovery_chain_v12.py"
        ]["sha256"],
        "--deployment-verifier-sha256",
        code_by_path[
            "scripts/build_schema5_watchdog_deployment.py"
        ]["sha256"],
        "--forced-command-sha256",
        code_by_path[
            "scripts/schema5_watchdog_forced_command.py"
        ]["sha256"],
        "--bootstrap-runtime-sha256",
        code_by_path[
            "scripts/schema5_bootstrap_watchdog.py"
        ]["sha256"],
        "--tagged-source-inventory-sha256",
        watchdog_code_sha256,
        "--chain-manifest",
        str(manifest_path),
        "--chain-manifest-sha256",
        hashlib.sha256(manifest_raw).hexdigest(),
        "--submission-receipt",
        str(receipt_path),
        "--submission-receipt-sha256",
        hashlib.sha256(receipt_raw).hexdigest(),
        "--isolated-drill-chain-manifest",
        str(isolated_manifest_path),
        "--isolated-drill-chain-manifest-sha256",
        hashlib.sha256(
            _stable_bytes(
                isolated_manifest_path,
                description="isolated bootstrap drill chain manifest",
                require_read_only=True,
                require_single_link=True,
            )
        ).hexdigest(),
        "--isolated-drill-submission-receipt",
        str(isolated_receipt_path),
        "--isolated-drill-submission-receipt-sha256",
        hashlib.sha256(
            _stable_bytes(
                isolated_receipt_path,
                description="isolated bootstrap drill submission receipt",
                require_read_only=True,
                require_single_link=True,
            )
        ).hexdigest(),
        "bootstrap-dispatch",
    ]
    forced_command = " ".join(shlex.quote(item) for item in forced_argv)
    if '"' in forced_command or "\\" in forced_command:
        raise DeploymentError(
            "bootstrap forced command requires unsupported quoting"
        )
    authorized_line = (
        'restrict,command="'
        + forced_command
        + '" '
        + f"{bootstrap_bare_key}\n"
    ).encode("ascii")
    config = {
        "schema_version": 1,
        "protocol": "schema5-v1.2-r5-bootstrap-watchdog-config-v1",
        "release_git_commit": git_commit,
        "release_tag_object": tag_object,
        "chain_id": manifest["chain_id"],
        "chain_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "anchor_submission_receipt_id": receipt["receipt_id"],
        "harness_environment_binding": harness_binding,
        "materialization_pilot": {
            "marker": str(pilot_marker_path),
            "marker_sha256": hashlib.sha256(pilot_raw).hexdigest(),
            "pilot_id": pilot_id,
            **pilot_release_binding,
        },
        "anchor_submission_receipt_sha256": hashlib.sha256(
            receipt_raw
        ).hexdigest(),
        "remote": {
            "host": remote_host,
            "user": remote_user,
            "identity_file": str(identity_file),
            "known_hosts_file": str(known_hosts_file),
        },
        "state_root": str(external_state_root),
        "observation_gap_seconds": 60,
        "timer_seconds": 300,
    }
    service = f"""[Unit]
Description=Schema-5 pre-control recovery watchdog
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User={service_user}
Group={service_user}
ExecStart={vm_python} -I -u {vm_release_root}/scripts/schema5_bootstrap_watchdog.py --config {vm_config_path}
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={external_state_root}
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
LockPersonality=true
MemoryDenyWriteExecute=true
""".encode()
    timer = f"""[Unit]
Description=Run the schema-5 pre-control watchdog every five minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
AccuracySec=5s
Persistent=true
Unit={BOOTSTRAP_SERVICE_NAME}

[Install]
WantedBy=timers.target
""".encode()
    root = Path(os.path.abspath(os.fspath(output_root.expanduser())))
    root.mkdir(parents=True, exist_ok=True)
    root = _canonical_path(
        root, description="bootstrap bundle root", kind="directory"
    )
    runtime_relatives = (
        Path("scripts/build_schema5_watchdog_deployment.py"),
        Path("scripts/schema5_bootstrap_watchdog.py"),
        Path("src/agents_scaling/serving/external_watchdog.py"),
    )
    runtime_inventory: list[dict[str, Any]] = []
    for relative in runtime_relatives:
        payload = _stable_bytes(
            release_root / relative,
            description=f"bootstrap runtime {relative}",
            require_read_only=True,
            require_single_link=True,
        )
        target = root / "release" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if _stable_bytes(
                target,
                description=f"bundled bootstrap runtime {relative}",
                require_read_only=True,
            ) != payload:
                raise DeploymentError(
                    f"bootstrap runtime conflicts: {target}"
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
        runtime_inventory.append(
            {
                "path": relative.as_posix(),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    runtime_inventory.sort(key=lambda item: str(item["path"]))
    runtime_inventory_sha256 = hashlib.sha256(
        _canonical(runtime_inventory)
    ).hexdigest()
    runtime_total_bytes = sum(int(item["size"]) for item in runtime_inventory)
    payloads = {
        "bootstrap-watchdog.json": _canonical(config),
        BOOTSTRAP_SERVICE_NAME: service,
        BOOTSTRAP_TIMER_NAME: timer,
        "authorized_keys.bootstrap.line": authorized_line,
    }
    files: list[dict[str, Any]] = []
    for name, payload in payloads.items():
        target = root / name
        if target.exists() or target.is_symlink():
            if _stable_bytes(
                target, description=f"bootstrap bundle {name}"
            ) != payload:
                raise DeploymentError(
                    f"bootstrap bundle file conflicts: {target}"
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
        files.append(
            {
                "name": name,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "protocol": BOOTSTRAP_BUNDLE_PROTOCOL,
        **_release_fields(git_commit, tag_object),
        "chain_manifest": str(manifest_path),
        "chain_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "chain_id": manifest["chain_id"],
        "submission_receipt": str(receipt_path),
        "submission_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "submission_receipt_id": receipt["receipt_id"],
        "isolated_cancellation_drill": isolated_drill_contract,
        "harness_environment_binding": harness_binding,
        "materialization_pilot": {
            "marker": str(pilot_marker_path),
            "marker_sha256": hashlib.sha256(pilot_raw).hexdigest(),
            "pilot_id": pilot_id,
            **pilot_release_binding,
        },
        "watchdog_code_sha256": watchdog_code_sha256,
        "code_records": code_records,
        "runtime_root": "release",
        "runtime_inventory": runtime_inventory,
        "runtime_inventory_sha256": runtime_inventory_sha256,
        "runtime_file_count": len(runtime_inventory),
        "runtime_total_bytes": runtime_total_bytes,
        "vm_python": str(vm_python),
        "vm_release_root": str(vm_release_root),
        "vm_config_path": str(vm_config_path),
        "ssh_public_key_sha256": hashlib.sha256(
            bootstrap_bare_key.encode("ascii")
        ).hexdigest(),
        "separate_bootstrap_key": True,
        "forced_command_argv": forced_argv,
        "allowed_operations": [
            "bootstrap-status",
            "bootstrap-repair",
            "bootstrap-drill-status",
            "bootstrap-drill-repair",
        ],
        "root_release_authority": (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        ),
        "prelaunch_root_release": False,
        "descendant_rearm_authority": True,
        "release_intent_continuation_authority": True,
        "scientific_admission_direct": False,
        "production_control_mutation": False,
        "safety_hold_clear_authority": False,
        "timer_seconds": 300,
        "files": files,
    }
    bundle["bundle_id"] = _self_hash(bundle, "bundle_id")
    path, digest = _publish_once(root / "BUNDLE.json", bundle)
    return {
        "passed": True,
        "bundle_root": str(root),
        "manifest": str(path),
        "manifest_sha256": digest,
        "bundle_id": bundle["bundle_id"],
    }


def capture_bootstrap_attestation(
    *,
    bundle_manifest: Path,
    deployment_evidence: Path,
    scheduler_observations: Sequence[Path],
    cancellation_drill_evidence: Path,
    output: Path,
) -> dict[str, Any]:
    """Bind external deployment, two held cuts, and cancellation drill."""

    bundle, bundle_raw = _sealed_json(
        bundle_manifest,
        description="bootstrap watchdog bundle manifest",
    )
    deployment, deployment_raw = _sealed_json(
        deployment_evidence,
        description="bootstrap watchdog deployment evidence",
    )
    drill, drill_raw = _sealed_json(
        cancellation_drill_evidence,
        description="bootstrap watchdog cancellation drill",
    )
    bundle_identity = dict(bundle)
    bundle_id = bundle_identity.pop("bundle_id", None)
    deployment_identity = dict(deployment)
    deployment_id = deployment_identity.pop("evidence_id", None)
    drill_identity = dict(drill)
    drill_id = drill_identity.pop("evidence_id", None)
    if (
        bundle.get("protocol") != BOOTSTRAP_BUNDLE_PROTOCOL
        or bundle_id != _self_hash(bundle, "bundle_id")
        or bundle.get("root_release_authority")
        != (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        )
        or bundle.get("prelaunch_root_release") is not False
        or bundle.get("descendant_rearm_authority") is not True
        or bundle.get("release_intent_continuation_authority") is not True
        or bundle.get("scientific_admission_direct") is not False
        or bundle.get("production_control_mutation") is not False
        or bundle.get("safety_hold_clear_authority") is not False
        or not isinstance(bundle.get("harness_environment_binding"), Mapping)
        or not isinstance(bundle.get("materialization_pilot"), Mapping)
        or deployment.get("protocol")
        != BOOTSTRAP_DEPLOYMENT_EVIDENCE_PROTOCOL
        or deployment.get("passed") is not True
        or deployment_id != _self_hash(deployment, "evidence_id")
        or deployment.get("bundle_id") != bundle_id
        or deployment.get("bundle_sha256")
        != hashlib.sha256(bundle_raw).hexdigest()
        or deployment.get("watchdog_code_sha256")
        != bundle.get("watchdog_code_sha256")
        or deployment.get("release_git_commit")
        != bundle.get("release_git_commit")
        or deployment.get("release_tag_object")
        != bundle.get("release_tag_object")
        or deployment.get("chain_id") != bundle.get("chain_id")
        or deployment.get("chain_manifest_sha256")
        != bundle.get("chain_manifest_sha256")
        or deployment.get("submission_receipt_id")
        != bundle.get("submission_receipt_id")
        or deployment.get("submission_receipt_sha256")
        != bundle.get("submission_receipt_sha256")
        or deployment.get("runtime_inventory_sha256")
        != bundle.get("runtime_inventory_sha256")
        or deployment.get("runtime_file_count")
        != bundle.get("runtime_file_count")
        or deployment.get("runtime_total_bytes")
        != bundle.get("runtime_total_bytes")
        or deployment.get("vm_python_immutable") is not True
        or SHA256.fullmatch(
            str(deployment.get("vm_python_sha256", ""))
        )
        is None
        or not isinstance(
            deployment.get("heartbeat_freshness_seconds"), (int, float)
        )
        or isinstance(
            deployment.get("heartbeat_freshness_seconds"), bool
        )
        or not 0
        <= float(deployment["heartbeat_freshness_seconds"])
        <= BOOTSTRAP_HEARTBEAT_MAX_AGE_SECONDS
        or deployment.get("heartbeat_max_age_seconds")
        != BOOTSTRAP_HEARTBEAT_MAX_AGE_SECONDS
        or deployment.get("ssh_public_key_sha256")
        != bundle.get("ssh_public_key_sha256")
        or deployment.get("separate_bootstrap_key") is not True
        or deployment.get("forced_command_only") is not True
        or deployment.get("systemd_service_loaded") is not True
        or deployment.get("systemd_timer_active") is not True
        or drill.get("protocol") != BOOTSTRAP_DRILL_EVIDENCE_PROTOCOL
        or drill.get("passed") is not True
        or drill_id != _self_hash(drill, "evidence_id")
        or drill.get("bundle_id") != bundle_id
        or drill.get("deployment_id") != deployment.get("deployment_id")
        or drill.get("root_remained_held") is not True
        or drill.get("scientific_jobs_started") != 0
        or drill.get("duplicate_jobs") != 0
        or drill.get("duplicate_submission_intents") != 0
        or drill.get("isolated_cancellation_drill") is not True
        or not isinstance(
            bundle.get("isolated_cancellation_drill"), Mapping
        )
        or drill.get("isolation_id")
        != bundle["isolated_cancellation_drill"].get("isolation_id")
        or drill.get("canonical_job_id_overlap") != 0
        or drill.get("canonical_comment_overlap") != 0
        or drill.get("canonical_control_paths_absent") is not True
        or drill.get("squeue_complete") is not True
        or drill.get("sacct_complete") is not True
        or not isinstance(
            drill.get("recovery_namespace_cancellation_recovery_seconds"),
            (int, float),
        )
        or isinstance(
            drill.get("recovery_namespace_cancellation_recovery_seconds"), bool
        )
        or not 0
        <= float(drill["recovery_namespace_cancellation_recovery_seconds"])
        <= 900
    ):
        raise DeploymentError(
            "bootstrap external deployment or drill evidence is invalid"
        )
    manifest, _ = _sealed_json(
        Path(str(bundle["chain_manifest"])),
        description="bootstrap attestation chain manifest",
    )
    receipt, _ = _sealed_json(
        Path(str(bundle["submission_receipt"])),
        description="bootstrap attestation submission receipt",
    )
    _validate_bootstrap_drill_preimages(
        drill=drill,
        bundle=bundle,
        deployment=deployment,
    )
    if len(scheduler_observations) != 2:
        raise DeploymentError(
            "bootstrap arming requires exactly two scheduler observations"
        )
    bindings: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for index, path in enumerate(scheduler_observations, start=1):
        value, raw = _sealed_json(
            path,
            description=f"bootstrap scheduler observation {index}",
        )
        (
            checked_value,
            checked_raw,
            checked_lineage,
            _checked_provenance,
            _checked_provenance_raw,
            checked_duplicates,
        ) = _validate_bootstrap_observation(
            path=path,
            manifest=manifest,
            manifest_path=Path(str(bundle["chain_manifest"])),
            manifest_raw=_canonical(manifest),
            anchor_path=Path(str(bundle["submission_receipt"])),
            anchor=receipt,
            cancelled=False,
        )
        if (
            checked_value != value
            or checked_raw != raw
            or len(checked_lineage) != 1
            or checked_duplicates != 0
        ):
            raise DeploymentError(
                f"bootstrap scheduler observation {index} is not the exact "
                "held anchor generation"
            )
        if (
            value.get("protocol")
            != "schema5-v1.2-r5-bootstrap-status-v1"
            or value.get("passed") is not True
            or value.get("chain_id") != bundle.get("chain_id")
            or value.get("chain_manifest_sha256")
            != bundle.get("chain_manifest_sha256")
            or value.get("submission_receipt_id")
            != bundle.get("submission_receipt_id")
            or value.get("submission_receipt_sha256")
            != bundle.get("submission_receipt_sha256")
            or value.get("anchor_submission_receipt_id")
            != bundle.get("submission_receipt_id")
            or value.get("anchor_submission_receipt_sha256")
            != bundle.get("submission_receipt_sha256")
            or value.get("descendant_chain_validated") is not True
            or value.get("anchor_launch_authorized") is not False
            or value.get("anchor_launch_id") is not None
            or value.get("descendant_release_required") is not False
            or value.get("root_held") is not True
            or value.get("squeue_complete") is not True
            or value.get("sacct_complete") is not True
            or value.get("ambiguous_jobs") != 0
            or value.get("handoff_complete") is not False
            or value.get("watchdog_scientific_jobs_submitted") != 0
            or value.get("observation_id")
            != _self_hash(value, "observation_id")
            or not isinstance(value.get("jobs"), list)
            or len(value["jobs"]) != len(manifest.get("jobs", []))
            or any(
                not isinstance(job, Mapping)
                or job.get("name") != manifest_row.get("name")
                or job.get("job_id") != receipt_row.get("job_id")
                or job.get("comment") != receipt_row.get("comment")
                or job.get("job_name") != manifest_row.get("job_name")
                or job.get("state") != "PENDING"
                or job.get("active") is not True
                or job.get("script_sha256")
                != manifest_row.get("script_sha256")
                or job.get("scontrol_command")
                != receipt_row.get("script")
                or job.get("scontrol_requeue") != 0
                or job.get("spooled_script_sha256")
                != manifest_row.get("script_sha256")
                or job.get("spooled_script_exact_match") is not True
                or job.get("submit_line_exact") is not True
                or SHA256.fullmatch(
                    str(job.get("submit_line_sha256", ""))
                )
                is None
                for job, manifest_row, receipt_row in zip(
                    value["jobs"],
                    manifest["jobs"],
                    receipt["jobs"],
                    strict=True,
                )
            )
        ):
            raise DeploymentError(
                f"bootstrap scheduler observation {index} is invalid"
            )
        provenance, provenance_raw = _sealed_json(
            Path(str(value.get("generation_provenance", ""))),
            description=(
                f"bootstrap scheduler provenance {index}"
            ),
        )
        if (
            value.get("generation_provenance_sha256")
            != hashlib.sha256(provenance_raw).hexdigest()
            or value.get("generation_provenance_id")
            != provenance.get("provenance_id")
            or provenance.get("provenance_id")
            != _self_hash(provenance, "provenance_id")
            or provenance.get("protocol")
            != (
                "schema5-v1.2-r5-bootstrap-"
                "generation-provenance-v1"
            )
            or provenance.get("chain_id") != bundle.get("chain_id")
            or provenance.get("submission_receipt_id")
            != receipt.get("receipt_id")
            or provenance.get("submission_receipt_sha256")
            != hashlib.sha256(
                _canonical(receipt)
            ).hexdigest()
            or provenance.get("repair_generation") != 0
        ):
            raise DeploymentError(
                f"bootstrap scheduler provenance {index} is invalid"
            )
        payloads.append(value)
        bindings.append(
            {
                "artifact": str(
                    _canonical_path(
                        path,
                        description=(
                            f"bootstrap scheduler observation {index}"
                        ),
                        kind="file",
                    )
                ),
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "observation_id": value["observation_id"],
            }
        )
    first = payloads[0].get("observed_at_timestamp")
    second = payloads[1].get("observed_at_timestamp")
    if (
        not isinstance(first, (int, float))
        or isinstance(first, bool)
        or not isinstance(second, (int, float))
        or isinstance(second, bool)
        or not math.isfinite(float(first))
        or not math.isfinite(float(second))
        or float(second) - float(first) < 60
    ):
        raise DeploymentError(
            "bootstrap scheduler observations are not 60 seconds apart"
        )
    attestation: dict[str, Any] = {
        "schema_version": 1,
        "protocol": BOOTSTRAP_ATTESTATION_PROTOCOL,
        "passed": True,
        "release_id": bundle["release_id"],
        "release_tag": bundle["release_tag"],
        "release_git_commit": bundle["release_git_commit"],
        "release_tag_object": bundle["release_tag_object"],
        "chain_namespace": bundle["chain_namespace"],
        "chain_manifest_sha256": bundle["chain_manifest_sha256"],
        "submission_receipt_sha256": bundle[
            "submission_receipt_sha256"
        ],
        "deployment_id": deployment["deployment_id"],
        "deployment_evidence": str(
            _canonical_path(
                deployment_evidence,
                description="bootstrap deployment evidence",
                kind="file",
            )
        ),
        "deployment_evidence_sha256": hashlib.sha256(
            deployment_raw
        ).hexdigest(),
        "deployment_evidence_id": deployment["evidence_id"],
        "watchdog_code_sha256": bundle["watchdog_code_sha256"],
        "bundle_manifest": str(
            _canonical_path(
                bundle_manifest,
                description="bootstrap bundle manifest",
                kind="file",
            )
        ),
        "bundle_manifest_sha256": hashlib.sha256(bundle_raw).hexdigest(),
        "bundle_id": bundle_id,
        "ssh_public_key_sha256": bundle["ssh_public_key_sha256"],
        "separate_bootstrap_key": True,
        "forced_command_argv": bundle["forced_command_argv"],
        "systemd_service_loaded": True,
        "systemd_timer_active": True,
        "forced_command_only": True,
        "forced_operation": "bootstrap-repair",
        "scheduler_observations": bindings,
        "cancellation_drill": {
            "recovery_namespace_cancellation_recovery_seconds": drill[
                "recovery_namespace_cancellation_recovery_seconds"
            ],
            "isolated_cancellation_drill": True,
            "isolation_id": drill["isolation_id"],
            "isolated_drill_root": drill["isolated_drill_root"],
            "isolated_chain_id": drill["isolated_chain_id"],
            "isolated_anchor_submission_receipt_id": drill[
                "isolated_anchor_submission_receipt_id"
            ],
            "canonical_submission_receipt_id": drill[
                "canonical_submission_receipt_id"
            ],
            "canonical_job_id_overlap": 0,
            "canonical_comment_overlap": 0,
            "canonical_control_paths_absent": True,
            "squeue_complete": True,
            "sacct_complete": True,
            "duplicate_jobs": drill["duplicate_jobs"],
            "duplicate_submission_intents": drill[
                "duplicate_submission_intents"
            ],
            "root_remained_held": drill["root_remained_held"],
            "scientific_jobs_started": drill[
                "scientific_jobs_started"
            ],
        },
        "cancellation_drill_evidence": str(
            _canonical_path(
                cancellation_drill_evidence,
                description="bootstrap cancellation drill evidence",
                kind="file",
            )
        ),
        "cancellation_drill_evidence_sha256": hashlib.sha256(
            drill_raw
        ).hexdigest(),
        "cancellation_drill_evidence_id": drill["evidence_id"],
        "root_release_authority": (
            "armed-descendant-rearm-or-release-intent-"
            "continuation-only"
        ),
        "prelaunch_root_release": False,
        "descendant_rearm_authority": True,
        "release_intent_continuation_authority": True,
        "scientific_admission_direct": False,
        "production_control_mutation": False,
        "safety_hold_clear_authority": False,
        "handoff_stage": "watchdog_readiness",
    }
    attestation["evidence_id"] = _self_hash(
        attestation, "evidence_id"
    )
    path, digest = _publish_once(output, attestation)
    return {
        "passed": True,
        "attestation": str(path),
        "attestation_sha256": digest,
        "evidence_id": attestation["evidence_id"],
        "deployment_evidence_sha256": hashlib.sha256(
            deployment_raw
        ).hexdigest(),
        "cancellation_drill_evidence_sha256": hashlib.sha256(
            drill_raw
        ).hexdigest(),
    }


def capture_bootstrap_deployment_evidence(
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
    systemctl_runner: Callable[..., subprocess.CompletedProcess[str]] = (
        subprocess.run
    ),
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Inspect the separately installed bootstrap deployment marker-last."""

    bundle, bundle_raw = _sealed_json(
        bundle_manifest,
        description="bootstrap watchdog bundle manifest",
    )
    if (
        bundle.get("protocol") != BOOTSTRAP_BUNDLE_PROTOCOL
        or bundle.get("bundle_id") != _self_hash(bundle, "bundle_id")
        or bundle.get("runtime_root") != "release"
        or not isinstance(bundle.get("runtime_inventory"), list)
        or bundle.get("runtime_file_count")
        != len(bundle.get("runtime_inventory", []))
        or SHA256.fullmatch(
            str(bundle.get("runtime_inventory_sha256", ""))
        )
        is None
        or not isinstance(bundle.get("runtime_total_bytes"), int)
        or isinstance(bundle.get("runtime_total_bytes"), bool)
        or int(bundle.get("runtime_total_bytes", 0)) <= 0
    ):
        raise DeploymentError("bootstrap bundle identity is invalid")
    runtime_inventory = _verify_installed_runtime(
        bundle, installed_release_root=installed_release_root
    )
    canonical_python = _canonical_path(
        vm_python, description="installed bootstrap watchdog Python", kind="file"
    )
    python_raw = _stable_bytes(
        canonical_python,
        description="installed bootstrap watchdog Python",
        require_read_only=True,
        require_single_link=True,
    )
    if (
        str(canonical_python) != str(bundle.get("vm_python", ""))
        or not os.access(canonical_python, os.X_OK)
        or stat.S_IMODE(canonical_python.stat(follow_symlinks=False).st_mode)
        & 0o222
    ):
        raise DeploymentError(
            "installed bootstrap Python differs from the immutable service path"
        )
    bundle_root = _canonical_path(
        bundle_manifest,
        description="bootstrap bundle manifest",
        kind="file",
    ).parent
    bundled = {
        str(row["name"]): row
        for row in bundle.get("files", [])
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    installed = {
        "bootstrap-watchdog.json": installed_config,
        BOOTSTRAP_SERVICE_NAME: installed_service,
        BOOTSTRAP_TIMER_NAME: installed_timer,
        "authorized_keys.bootstrap.line": installed_authorized_keys,
    }
    installed_records: list[dict[str, Any]] = []
    installed_payloads: dict[str, bytes] = {}
    for name, path in installed.items():
        record = bundled.get(name)
        if not isinstance(record, Mapping):
            raise DeploymentError(
                f"bootstrap bundle lacks installed artifact {name}"
            )
        raw = _stable_bytes(
            path,
            description=f"installed bootstrap artifact {name}",
            require_single_link=True,
        )
        expected_path = bundle_root / name
        expected = _stable_bytes(
            expected_path,
            description=f"bundled bootstrap artifact {name}",
            require_read_only=True,
            require_single_link=True,
        )
        if name == "authorized_keys.bootstrap.line":
            _require_single_authorized_key_identity(
                raw,
                expected,
                description="installed bootstrap authorized_keys",
            )
        elif raw != expected:
            raise DeploymentError(
                f"installed bootstrap artifact drifted: {name}"
            )
        installed_records.append(
            {
                "name": name,
                "path": str(
                    _canonical_path(
                        path,
                        description=f"installed bootstrap artifact {name}",
                        kind="file",
                    )
                ),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        installed_payloads[name] = raw
    try:
        installed_config_value = json.loads(
            installed_payloads["bootstrap-watchdog.json"].decode("utf-8")
        )
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise DeploymentError(
            f"installed bootstrap configuration is invalid: {exc}"
        ) from exc
    if (
        not isinstance(installed_config_value, dict)
        or installed_config_value.get("protocol")
        != "schema5-v1.2-r5-bootstrap-watchdog-config-v1"
        or installed_config_value.get("release_git_commit")
        != bundle.get("release_git_commit")
        or installed_config_value.get("release_tag_object")
        != bundle.get("release_tag_object")
        or installed_config_value.get("chain_id") != bundle.get("chain_id")
        or installed_config_value.get("chain_manifest_sha256")
        != bundle.get("chain_manifest_sha256")
        or installed_config_value.get("anchor_submission_receipt_id")
        != bundle.get("submission_receipt_id")
        or installed_config_value.get("anchor_submission_receipt_sha256")
        != bundle.get("submission_receipt_sha256")
    ):
        raise DeploymentError(
            "installed bootstrap configuration chain identity drifted"
        )
    heartbeat, heartbeat_raw = _sealed_json(
        service_heartbeat,
        description="bootstrap watchdog service heartbeat",
    )
    heartbeat_required = {
        "schema_version",
        "protocol",
        "passed",
        "observed_at_timestamp",
        "configuration_sha256",
        "release_git_commit",
        "release_tag_object",
        "chain_id",
        "chain_manifest_sha256",
        "anchor_submission_receipt_id",
        "anchor_submission_receipt_sha256",
        "first_observation_id",
        "second_observation_id",
        "action",
        "recovery_root_release_performed",
        "production_control_mutated",
        "scientific_jobs_submitted",
        "safety_hold_cleared",
        "heartbeat_id",
    }
    observed_at = heartbeat.get("observed_at_timestamp")
    captured_at = float(clock())
    if (
        set(heartbeat) != heartbeat_required
        or heartbeat.get("schema_version") != 1
        or heartbeat.get("protocol")
        != "schema5-v1.2-r5-bootstrap-watchdog-heartbeat-v1"
        or heartbeat.get("passed") is not True
        or heartbeat.get("configuration_sha256")
        != hashlib.sha256(
            installed_payloads["bootstrap-watchdog.json"]
        ).hexdigest()
        or heartbeat.get("release_git_commit")
        != bundle.get("release_git_commit")
        or heartbeat.get("release_tag_object")
        != bundle.get("release_tag_object")
        or heartbeat.get("chain_id") != bundle.get("chain_id")
        or heartbeat.get("chain_manifest_sha256")
        != bundle.get("chain_manifest_sha256")
        or heartbeat.get("anchor_submission_receipt_id")
        != bundle.get("submission_receipt_id")
        or heartbeat.get("anchor_submission_receipt_sha256")
        != bundle.get("submission_receipt_sha256")
        or any(
            SHA256.fullmatch(str(heartbeat.get(field, ""))) is None
            for field in (
                "configuration_sha256",
                "chain_id",
                "chain_manifest_sha256",
                "anchor_submission_receipt_id",
                "anchor_submission_receipt_sha256",
                "first_observation_id",
                "second_observation_id",
            )
        )
        or heartbeat.get("action")
        not in {
            "healthy_noop",
            "handoff_complete",
            "bootstrap_repair",
            "bootstrap_descendant_rearm",
            "bootstrap_release_reconcile",
        }
        or heartbeat.get("recovery_root_release_performed") is not False
        or heartbeat.get("production_control_mutated") is not False
        or heartbeat.get("scientific_jobs_submitted") != 0
        or heartbeat.get("safety_hold_cleared") is not False
        or heartbeat.get("heartbeat_id")
        != _self_hash(heartbeat, "heartbeat_id")
        or not isinstance(observed_at, (int, float))
        or isinstance(observed_at, bool)
        or not math.isfinite(float(observed_at))
        or not math.isfinite(captured_at)
        or float(observed_at) > captured_at + 5.0
        or captured_at - float(observed_at)
        > BOOTSTRAP_HEARTBEAT_MAX_AGE_SECONDS
    ):
        raise DeploymentError(
            "bootstrap watchdog service heartbeat is invalid"
        )
    loaded = _systemctl_value(
        systemctl_runner,
        unit=BOOTSTRAP_SERVICE_NAME,
        property_name="LoadState",
    )
    service_result = _systemctl_value(
        systemctl_runner,
        unit=BOOTSTRAP_SERVICE_NAME,
        property_name="Result",
    )
    status = _systemctl_value(
        systemctl_runner,
        unit=BOOTSTRAP_SERVICE_NAME,
        property_name="ExecMainStatus",
    )
    timer_active = systemctl_runner(
        ["systemctl", "is-active", "--quiet", BOOTSTRAP_TIMER_NAME],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    timer_enabled = systemctl_runner(
        ["systemctl", "is-enabled", "--quiet", BOOTSTRAP_TIMER_NAME],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if (
        loaded != "loaded"
        or service_result != "success"
        or status != "0"
        or timer_active.returncode != 0
        or timer_enabled.returncode != 0
    ):
        raise DeploymentError(
            "bootstrap watchdog systemd service/timer is not healthy"
        )
    deployment_projection = {
        "bundle_id": bundle["bundle_id"],
        "bundle_sha256": hashlib.sha256(bundle_raw).hexdigest(),
        "installed": installed_records,
        "runtime_inventory": runtime_inventory,
        "runtime_inventory_sha256": bundle["runtime_inventory_sha256"],
        "vm_python_path": str(canonical_python),
        "vm_python_sha256": hashlib.sha256(python_raw).hexdigest(),
        "heartbeat_id": heartbeat["heartbeat_id"],
        "heartbeat_sha256": hashlib.sha256(heartbeat_raw).hexdigest(),
    }
    deployment_id = hashlib.sha256(
        _canonical(deployment_projection)
    ).hexdigest()
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol": BOOTSTRAP_DEPLOYMENT_EVIDENCE_PROTOCOL,
        "passed": True,
        "bundle_id": bundle["bundle_id"],
        "bundle_sha256": hashlib.sha256(bundle_raw).hexdigest(),
        "deployment_id": deployment_id,
        "watchdog_code_sha256": bundle["watchdog_code_sha256"],
        "release_git_commit": bundle["release_git_commit"],
        "release_tag_object": bundle["release_tag_object"],
        "chain_id": bundle["chain_id"],
        "chain_manifest_sha256": bundle["chain_manifest_sha256"],
        "submission_receipt_id": bundle["submission_receipt_id"],
        "submission_receipt_sha256": bundle["submission_receipt_sha256"],
        "runtime_inventory_sha256": bundle["runtime_inventory_sha256"],
        "runtime_file_count": bundle["runtime_file_count"],
        "runtime_total_bytes": bundle["runtime_total_bytes"],
        "vm_python_path": str(canonical_python),
        "vm_python_sha256": hashlib.sha256(python_raw).hexdigest(),
        "vm_python_immutable": True,
        "heartbeat_freshness_seconds": captured_at - float(observed_at),
        "heartbeat_max_age_seconds": BOOTSTRAP_HEARTBEAT_MAX_AGE_SECONDS,
        "ssh_public_key_sha256": bundle["ssh_public_key_sha256"],
        "separate_bootstrap_key": True,
        "forced_command_only": True,
        "systemd_service_loaded": True,
        "systemd_service_result": "success",
        "systemd_service_exec_main_status": 0,
        "systemd_timer_active": True,
        "systemd_timer_enabled": True,
        "successful_service_heartbeat_id": heartbeat["heartbeat_id"],
        "successful_service_heartbeat_sha256": hashlib.sha256(
            heartbeat_raw
        ).hexdigest(),
        "installed_artifacts": installed_records,
    }
    evidence["evidence_id"] = _self_hash(evidence, "evidence_id")
    path, digest = _publish_once(output, evidence)
    return {
        "passed": True,
        "deployment_evidence": str(path),
        "deployment_evidence_sha256": digest,
        "deployment_id": deployment_id,
        "evidence_id": evidence["evidence_id"],
    }


def _artifact_binding(
    path: Path,
    raw: bytes,
    *,
    identity_field: str | None = None,
    value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "artifact": str(
            _canonical_path(
                path, description="bootstrap drill artifact", kind="file"
            )
        ),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if identity_field is not None:
        if value is None or not isinstance(value.get(identity_field), str):
            raise DeploymentError(
                f"bootstrap drill artifact lacks {identity_field}"
            )
        binding[identity_field] = value[identity_field]
    return binding


def _bootstrap_drill_manifest_and_anchor(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_manifest_path = Path(str(bundle.get("chain_manifest", "")))
    canonical_anchor_path = Path(str(bundle.get("submission_receipt", "")))
    (
        canonical_manifest,
        canonical_manifest_raw,
        canonical_anchor,
        canonical_anchor_raw,
    ) = _bootstrap_chain_anchor(
        manifest_path=canonical_manifest_path,
        receipt_path=canonical_anchor_path,
        description="canonical bootstrap drill binding",
    )
    contract = bundle.get("isolated_cancellation_drill")
    if (
        bundle.get("bundle_id") != _self_hash(bundle, "bundle_id")
        or bundle.get("protocol") != BOOTSTRAP_BUNDLE_PROTOCOL
        or not isinstance(contract, Mapping)
        or contract.get("isolation_id")
        != _self_hash(contract, "isolation_id")
        or bundle.get("chain_id") != canonical_manifest.get("chain_id")
        or bundle.get("chain_manifest_sha256")
        != hashlib.sha256(canonical_manifest_raw).hexdigest()
        or bundle.get("submission_receipt_id")
        != canonical_anchor.get("receipt_id")
        or bundle.get("submission_receipt_sha256")
        != hashlib.sha256(canonical_anchor_raw).hexdigest()
    ):
        raise DeploymentError(
            "bootstrap drill canonical bundle binding is invalid"
        )
    isolated_manifest_path = Path(str(contract.get("chain_manifest", "")))
    isolated_anchor_path = Path(
        str(contract.get("anchor_submission_receipt", ""))
    )
    (
        expected_contract,
        isolated_manifest,
        isolated_anchor,
    ) = _bootstrap_isolated_drill_contract(
        canonical_manifest_path=canonical_manifest_path,
        canonical_manifest=canonical_manifest,
        canonical_receipt_path=canonical_anchor_path,
        canonical_receipt=canonical_anchor,
        isolated_manifest_path=isolated_manifest_path,
        isolated_receipt_path=isolated_anchor_path,
    )
    if dict(contract) != expected_contract:
        raise DeploymentError(
            "bootstrap isolated cancellation-drill contract drifted"
        )
    isolated_manifest_raw = _stable_bytes(
        isolated_manifest_path,
        description="isolated bootstrap drill chain manifest",
        require_read_only=True,
        require_single_link=True,
    )
    isolated_anchor_raw = _stable_bytes(
        isolated_anchor_path,
        description="isolated bootstrap drill anchor receipt",
        require_read_only=True,
        require_single_link=True,
    )
    return {
        "manifest": isolated_manifest,
        "manifest_path": isolated_manifest_path,
        "manifest_raw": isolated_manifest_raw,
        "anchor": isolated_anchor,
        "anchor_path": isolated_anchor_path,
        "anchor_raw": isolated_anchor_raw,
        "contract": dict(contract),
        "canonical_manifest": canonical_manifest,
        "canonical_anchor": canonical_anchor,
        "canonical_job_ids": {
            str(row["job_id"]) for row in canonical_anchor["jobs"]
        },
        "canonical_comments": {
            str(row["comment"]) for row in canonical_anchor["jobs"]
        },
    }


def _ordered_repair_frontier(
    manifest: Mapping[str, Any], repair_names: Sequence[str]
) -> list[str]:
    """Return the exact ordered minimal frontier of a resubmitted induced DAG."""

    manifest_rows = manifest.get("jobs")
    if (
        not isinstance(manifest_rows, list)
        or not repair_names
        or len(repair_names) != len(set(repair_names))
    ):
        raise DeploymentError("bootstrap repair frontier inputs are invalid")
    repair_set = set(repair_names)
    manifest_names = [
        str(row.get("name"))
        for row in manifest_rows
        if isinstance(row, Mapping)
    ]
    if (
        len(manifest_names) != len(manifest_rows)
        or len(manifest_names) != len(set(manifest_names))
        or not repair_set <= set(manifest_names)
    ):
        raise DeploymentError("bootstrap repair frontier names are invalid")
    ancestors_by_name: dict[str, set[str]] = {}
    frontier: list[str] = []
    for row in manifest_rows:
        name = str(row["name"])
        dependencies = row.get("dependencies")
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) for item in dependencies
        ):
            raise DeploymentError(
                "bootstrap repair manifest dependencies are invalid"
            )
        if (
            len(dependencies) != len(set(dependencies))
            or any(item not in ancestors_by_name for item in dependencies)
        ):
            raise DeploymentError(
                "bootstrap repair manifest is not a canonical topological DAG"
            )
        ancestors = set(dependencies)
        for dependency in dependencies:
            ancestors.update(ancestors_by_name[dependency])
        ancestors_by_name[name] = ancestors
        if name in repair_set and not repair_set.intersection(ancestors):
            frontier.append(name)
    if not frontier:
        raise DeploymentError("bootstrap repair frontier is empty")
    return frontier


def _validate_bootstrap_receipt_lineage(
    *,
    current_path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_raw: bytes,
    anchor_path: Path,
    anchor_id: str,
    scheduler_comment_prefix: str | None = None,
    forbidden_job_ids: set[str] | None = None,
    forbidden_comments: set[str] | None = None,
) -> list[tuple[Path, dict[str, Any], bytes]]:
    """Read and validate every receipt from the anchor to one generation."""

    backwards: list[tuple[Path, dict[str, Any], bytes]] = []
    seen: set[Path] = set()
    path = current_path
    expected_generation: int | None = None
    while True:
        canonical = _canonical_path(
            path, description="bootstrap drill receipt", kind="file"
        )
        if canonical in seen:
            raise DeploymentError("bootstrap receipt lineage contains a cycle")
        seen.add(canonical)
        receipt, raw = _sealed_json(
            canonical, description="bootstrap drill receipt"
        )
        identity = dict(receipt)
        receipt_id = identity.pop("receipt_id", None)
        generation = receipt.get("repair_generation", 0)
        rows = receipt.get("jobs")
        manifest_rows = manifest.get("jobs")
        expected_path = (
            anchor_path
            if generation == 0
            else (
                anchor_path.parent
                / BOOTSTRAP_CANONICAL_REPAIR_DIRECTORY
                / f"g{generation:04d}"
                / BOOTSTRAP_SUBMISSION_RECEIPT_NAME
            )
        )
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 0
            or (
                expected_generation is not None
                and generation != expected_generation
            )
            or receipt_id != hashlib.sha256(_canonical(identity)).hexdigest()
            or receipt.get("chain_id") != manifest.get("chain_id")
            or receipt.get("manifest") != str(manifest_path)
            or receipt.get("manifest_sha256")
            != hashlib.sha256(manifest_raw).hexdigest()
            or receipt.get("root_initial_hold") is not True
            or receipt.get("no_requeue") is not True
            or not isinstance(rows, list)
            or not isinstance(manifest_rows, list)
            or len(rows) != len(manifest_rows)
            or canonical != expected_path
        ):
            raise DeploymentError("bootstrap drill receipt identity drifted")
        names: list[str] = []
        job_ids: list[str] = []
        comments: list[str] = []
        submitted: dict[str, str] = {}
        resubmitted_names: list[str] = []
        for row, manifest_row in zip(rows, manifest_rows, strict=True):
            dependencies = manifest_row.get("dependencies")
            if (
                not isinstance(row, Mapping)
                or not isinstance(manifest_row, Mapping)
                or row.get("name") != manifest_row.get("name")
                or not isinstance(dependencies, list)
                or any(
                    not isinstance(item, str) or item not in submitted
                    for item in dependencies
                )
                or row.get("dependencies") != dependencies
                or row.get("dependency_job_ids")
                != [submitted[item] for item in dependencies]
                or not isinstance(row.get("job_id"), str)
                or not str(row["job_id"]).isdigit()
                or not isinstance(row.get("comment"), str)
                or row.get("script") != manifest_row.get("script")
                or row.get("script_sha256")
                != manifest_row.get("script_sha256")
                or (
                    forbidden_job_ids is not None
                    and str(row.get("job_id", "")) in forbidden_job_ids
                )
                or (
                    forbidden_comments is not None
                    and str(row.get("comment", "")) in forbidden_comments
                )
            ):
                raise DeploymentError(
                    "bootstrap drill receipt job namespace drifted"
                )
            origin = row.get("generation", 0)
            if (
                not isinstance(origin, int)
                or isinstance(origin, bool)
                or not 0 <= origin <= generation
                or (
                    scheduler_comment_prefix is not None
                    and not str(row["comment"]).startswith(
                        scheduler_comment_prefix + f"g{origin:04d}:"
                    )
                )
            ):
                raise DeploymentError(
                    "bootstrap receipt job generation is invalid"
                )
            if generation > 0:
                disposition = row.get("disposition")
                if (
                    disposition not in {"reused_completed", "resubmitted"}
                    or (
                        disposition == "resubmitted"
                        and origin != generation
                    )
                    or (
                        disposition == "reused_completed"
                        and (
                            origin >= generation
                            or any(
                                item in set(resubmitted_names)
                                for item in dependencies
                            )
                        )
                    )
                ):
                    raise DeploymentError(
                        "bootstrap receipt disposition lineage is invalid"
                    )
                if disposition == "resubmitted":
                    resubmitted_names.append(str(row["name"]))
            names.append(str(row["name"]))
            job_ids.append(str(row["job_id"]))
            comments.append(str(row["comment"]))
            submitted[str(row["name"])] = str(row["job_id"])
        if (
            len(names) != len(set(names))
            or len(job_ids) != len(set(job_ids))
            or len(comments) != len(set(comments))
        ):
            raise DeploymentError(
                "bootstrap receipt contains duplicate jobs or intents"
            )
        if generation > 0:
            expected_frontier = _ordered_repair_frontier(
                manifest, resubmitted_names
            )
            if receipt.get("held_root_names") != expected_frontier:
                raise DeploymentError(
                    "bootstrap receipt held-root frontier drifted"
                )
        backwards.append((canonical, receipt, raw))
        if generation == 0:
            if canonical != anchor_path or receipt_id != anchor_id:
                raise DeploymentError(
                    "bootstrap receipt lineage does not terminate at the anchor"
                )
            break
        parent_path = Path(str(receipt.get("parent_receipt", "")))
        parent, parent_raw = _sealed_json(
            parent_path, description="bootstrap drill parent receipt"
        )
        if (
            receipt.get("parent_receipt_sha256")
            != hashlib.sha256(parent_raw).hexdigest()
            or parent.get("receipt_id")
            != _self_hash(parent, "receipt_id")
            or parent.get("repair_generation", 0) != generation - 1
        ):
            raise DeploymentError(
                "bootstrap repair receipt parent lineage drifted"
            )
        path = parent_path
        expected_generation = generation - 1
    backwards.reverse()
    return backwards


def _validate_bootstrap_generation_provenance(
    *,
    provenance_path: Path,
    lineage: Sequence[tuple[Path, dict[str, Any], bytes]],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_raw: bytes,
) -> tuple[dict[str, Any], bytes]:
    """Validate the complete self-hashed scheduler provenance lineage."""

    current: tuple[dict[str, Any], bytes] | None = None
    provenance_paths: list[Path] = [provenance_path]
    probe, _ = _sealed_json(
        provenance_path, description="bootstrap drill generation provenance"
    )
    while probe.get("parent_provenance") is not None:
        parent_path = Path(str(probe["parent_provenance"]))
        provenance_paths.append(parent_path)
        probe, _ = _sealed_json(
            parent_path,
            description="bootstrap drill parent generation provenance",
        )
        if len(provenance_paths) > len(lineage):
            raise DeploymentError(
                "bootstrap generation provenance lineage contains a cycle"
            )
    provenance_paths.reverse()
    if len(provenance_paths) != len(lineage):
        raise DeploymentError(
            "bootstrap generation provenance lineage length drifted"
        )
    parent_value: dict[str, Any] | None = None
    parent_raw: bytes | None = None
    parent_path: Path | None = None
    for generation, (
        path,
        (receipt_path, receipt, receipt_raw),
    ) in enumerate(zip(provenance_paths, lineage, strict=True)):
        value, raw = _sealed_json(
            path, description="bootstrap drill generation provenance"
        )
        rows = value.get("jobs")
        manifest_rows = manifest.get("jobs")
        receipt_rows = receipt.get("jobs")
        expected_lineage_ids = [
            item[1]["receipt_id"] for item in lineage[: generation + 1]
        ]
        expected_job_ids = sorted(
            {
                str(row["job_id"])
                for _, lineage_receipt, _ in lineage[: generation + 1]
                for row in lineage_receipt["jobs"]
            },
            key=int,
        )
        current_names = [
            str(item["name"])
            for item in receipt_rows
            if generation == 0
            or int(item.get("generation", 0)) == generation
        ]
        root_names = (
            ["source_checkout"]
            if generation == 0
            else list(receipt.get("held_root_names", []))
        )
        if (
            value.get("protocol")
            != "schema5-v1.2-r5-bootstrap-generation-provenance-v1"
            or value.get("passed") is not True
            or value.get("provenance_id")
            != _self_hash(value, "provenance_id")
            or value.get("release_git_commit")
            != manifest.get("release_git_commit")
            or value.get("release_tag_object")
            != manifest.get("release_tag_object")
            or value.get("chain_id") != manifest.get("chain_id")
            or value.get("chain_manifest") != str(manifest_path)
            or value.get("chain_manifest_sha256")
            != hashlib.sha256(manifest_raw).hexdigest()
            or value.get("submission_receipt") != str(receipt_path)
            or value.get("submission_receipt_sha256")
            != hashlib.sha256(receipt_raw).hexdigest()
            or value.get("submission_receipt_id") != receipt.get("receipt_id")
            or value.get("repair_generation") != generation
            or value.get("parent_provenance")
            != (None if parent_path is None else str(parent_path))
            or value.get("parent_provenance_sha256")
            != (
                None
                if parent_raw is None
                else hashlib.sha256(parent_raw).hexdigest()
            )
            or value.get("parent_provenance_id")
            != (
                None
                if parent_value is None
                else parent_value.get("provenance_id")
            )
            or value.get("namespace_scan_complete") is not True
            or value.get("namespace_lineage_receipt_ids")
            != expected_lineage_ids
            or value.get("namespace_bound_job_ids") != expected_job_ids
            or value.get("generation_root_names") != root_names
            or value.get("current_generation_names") != current_names
            or value.get("scheduler_topology_valid") is not True
            or not isinstance(rows, list)
            or not isinstance(manifest_rows, list)
            or len(rows) != len(manifest_rows)
        ):
            raise DeploymentError(
                "bootstrap generation scheduler provenance drifted"
            )
        for row, manifest_row, receipt_row in zip(
            rows, manifest_rows, receipt_rows, strict=True
        ):
            if (
                not isinstance(row, Mapping)
                or row.get("name") != manifest_row.get("name")
                or row.get("job_id") != receipt_row.get("job_id")
                or row.get("comment") != receipt_row.get("comment")
                or row.get("job_name") != manifest_row.get("job_name")
                or row.get("script_sha256")
                != manifest_row.get("script_sha256")
                or row.get("scontrol_command") != receipt_row.get("script")
                or row.get("scontrol_requeue") != 0
                or row.get("spooled_script_sha256")
                != manifest_row.get("script_sha256")
                or row.get("submit_line_exact") is not True
                or row.get("spooled_script_exact_match") is not True
                or SHA256.fullmatch(
                    str(row.get("submit_line_sha256", ""))
                )
                is None
                or row.get("origin_generation")
                != int(receipt_row.get("generation", 0))
            ):
                raise DeploymentError(
                    "bootstrap generation job provenance drifted"
                )
        parent_path, parent_value, parent_raw = path, value, raw
        current = (value, raw)
    if current is None:
        raise DeploymentError("bootstrap generation provenance is unavailable")
    return current


def _validate_bootstrap_observation(
    *,
    path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    manifest_raw: bytes,
    anchor_path: Path,
    anchor: Mapping[str, Any],
    cancelled: bool,
    scheduler_comment_prefix: str | None = None,
    forbidden_job_ids: set[str] | None = None,
    forbidden_comments: set[str] | None = None,
) -> tuple[
    dict[str, Any],
    bytes,
    list[tuple[Path, dict[str, Any], bytes]],
    dict[str, Any],
    bytes,
    int,
]:
    value, raw = _sealed_json(
        path, description="bootstrap drill scheduler observation"
    )
    receipt_path = Path(str(value.get("submission_receipt", "")))
    lineage = _validate_bootstrap_receipt_lineage(
        current_path=receipt_path,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_raw=manifest_raw,
        anchor_path=anchor_path,
        anchor_id=str(anchor["receipt_id"]),
        scheduler_comment_prefix=scheduler_comment_prefix,
        forbidden_job_ids=forbidden_job_ids,
        forbidden_comments=forbidden_comments,
    )
    _, receipt, receipt_raw = lineage[-1]
    generation = len(lineage) - 1
    provenance, provenance_raw = _validate_bootstrap_generation_provenance(
        provenance_path=Path(str(value.get("generation_provenance", ""))),
        lineage=lineage,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_raw=manifest_raw,
    )
    receipt_rows = receipt["jobs"]
    manifest_rows = manifest["jobs"]
    rows = value.get("jobs")
    expected_lineage_ids = [item[1]["receipt_id"] for item in lineage]
    expected_lineage_job_ids = [
        [str(row["job_id"]) for row in lineage_receipt["jobs"]]
        for _, lineage_receipt, _ in lineage
    ]
    expected_job_ids = sorted(
        {
            str(row["job_id"])
            for _, lineage_receipt, _ in lineage
            for row in lineage_receipt["jobs"]
        },
        key=int,
    )
    current_names = {
        str(item["name"])
        for item in receipt_rows
        if generation == 0
        or int(item.get("generation", 0)) == generation
    }
    root_names = (
        ["source_checkout"]
        if generation == 0
        else list(receipt.get("held_root_names", []))
    )
    root_rows = [
        next(
            (
                item
                for item in receipt_rows
                if item.get("name") == root_name
            ),
            None,
        )
        for root_name in root_names
    ]
    root_job_ids = [
        str(item["job_id"])
        for item in root_rows
        if isinstance(item, Mapping)
    ]
    provenance_rows = provenance.get("jobs")
    timestamp = value.get("observed_at_timestamp")
    if (
        value.get("protocol") != "schema5-v1.2-r5-bootstrap-status-v1"
        or value.get("passed") is not True
        or value.get("observation_id")
        != _self_hash(value, "observation_id")
        or not isinstance(timestamp, (int, float))
        or isinstance(timestamp, bool)
        or not math.isfinite(float(timestamp))
        or value.get("release_git_commit")
        != manifest.get("release_git_commit")
        or value.get("release_tag_object")
        != manifest.get("release_tag_object")
        or value.get("chain_id") != manifest.get("chain_id")
        or value.get("chain_manifest") != str(manifest_path)
        or value.get("chain_manifest_sha256")
        != hashlib.sha256(manifest_raw).hexdigest()
        or value.get("submission_receipt_sha256")
        != hashlib.sha256(receipt_raw).hexdigest()
        or value.get("submission_receipt_id") != receipt.get("receipt_id")
        or value.get("generation_provenance_sha256")
        != hashlib.sha256(provenance_raw).hexdigest()
        or value.get("generation_provenance_id")
        != provenance.get("provenance_id")
        or value.get("anchor_submission_receipt") != str(anchor_path)
        or value.get("anchor_submission_receipt_sha256")
        != hashlib.sha256(_canonical(anchor)).hexdigest()
        or value.get("anchor_submission_receipt_id")
        != anchor.get("receipt_id")
        or value.get("descendant_chain_validated") is not True
        or value.get("repair_generation") != generation
        or not root_names
        or len(root_rows) != len(root_job_ids)
        or value.get("root_name") != root_names[0]
        or value.get("root_job_id") != root_job_ids[0]
        or value.get("root_names") != root_names
        or value.get("root_job_ids") != root_job_ids
        or value.get("roots_held") is not (not cancelled)
        or value.get("squeue_complete") is not True
        or value.get("sacct_complete") is not True
        or value.get("namespace_scan_complete") is not True
        or value.get("namespace_lineage_receipt_ids")
        != expected_lineage_ids
        or value.get("namespace_lineage_job_ids")
        != expected_lineage_job_ids
        or value.get("namespace_bound_job_ids") != expected_job_ids
        or value.get("ambiguous_jobs") != 0
        or value.get("job_count") != len(manifest_rows)
        or not isinstance(rows, list)
        or len(rows) != len(manifest_rows)
        or not isinstance(provenance_rows, list)
        or len(provenance_rows) != len(manifest_rows)
        or value.get("recovery_namespace_cancelled") is not cancelled
        or value.get("root_held") is cancelled
        or value.get("anchor_launch_authorized") is not False
        or value.get("anchor_launch_id") is not None
        or value.get("anchor_armed_authorized") is not False
        or value.get("anchor_armed_id") is not None
        or value.get("anchor_release_intent_authorized") is not False
        or value.get("anchor_release_intent_sha256") is not None
        or value.get("descendant_armed") is not None
        or value.get("descendant_armed_id") is not None
        or value.get("descendant_rearm_required") is not False
        or value.get("descendant_release_required") is not False
        or value.get("handoff_complete") is not False
        or not isinstance(
            value.get("watchdog_scientific_jobs_submitted"), int
        )
        or isinstance(
            value.get("watchdog_scientific_jobs_submitted"), bool
        )
        or value.get("watchdog_scientific_jobs_submitted") < 0
    ):
        raise DeploymentError(
            "bootstrap drill scheduler observation identity drifted"
        )
    terminal = {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
        "COMPLETED",
    }
    duplicate_keys: list[tuple[str, str, str]] = []
    for row, provenance_row, manifest_row, receipt_row in zip(
        rows,
        provenance_rows,
        manifest_rows,
        receipt_rows,
        strict=True,
    ):
        name = str(receipt_row["name"])
        expected_active = not cancelled and name in current_names
        expected_state = (
            None
            if cancelled
            else ("PENDING" if expected_active else "COMPLETED")
        )
        if (
            not isinstance(row, Mapping)
            or not isinstance(provenance_row, Mapping)
            or row.get("name") != name
            or any(
                row.get(field) != provenance_row.get(field)
                for field in (
                    "name",
                    "job_id",
                    "comment",
                    "job_name",
                    "script_sha256",
                    "submit_line_sha256",
                    "submit_line_exact",
                    "scontrol_command",
                    "scontrol_requeue",
                    "spooled_script_sha256",
                    "spooled_script_exact_match",
                )
            )
            or row.get("job_id") != receipt_row.get("job_id")
            or row.get("comment") != receipt_row.get("comment")
            or row.get("job_name") != manifest_row.get("job_name")
            or row.get("script_sha256")
            != manifest_row.get("script_sha256")
            or row.get("scontrol_command") != receipt_row.get("script")
            or row.get("scontrol_requeue") != 0
            or row.get("spooled_script_sha256")
            != manifest_row.get("script_sha256")
            or row.get("submit_line_exact") is not True
            or row.get("spooled_script_exact_match") is not True
            or SHA256.fullmatch(
                str(row.get("submit_line_sha256", ""))
            )
            is None
            or (
                cancelled
                and (
                    row.get("active") is not False
                    or row.get("state") not in terminal
                )
            )
            or (
                not cancelled
                and (
                    row.get("active") is not expected_active
                    or row.get("state") != expected_state
                )
            )
        ):
            raise DeploymentError(
                "bootstrap drill scheduler job evidence drifted"
            )
        duplicate_keys.append(
            (
                str(row["job_id"]),
                str(row["comment"]),
                str(row["name"]),
            )
        )
    duplicate_jobs = len(duplicate_keys) - len(set(duplicate_keys))
    return (
        value,
        raw,
        lineage,
        provenance,
        provenance_raw,
        duplicate_jobs,
    )


def _inspect_bootstrap_drill(
    *,
    bundle: Mapping[str, Any],
    deployment: Mapping[str, Any],
    cancelled_paths: Sequence[Path],
    repair_path: Path,
    recovered_path: Path,
    recovery_seconds: float,
) -> dict[str, Any]:
    drill_context = _bootstrap_drill_manifest_and_anchor(bundle)
    manifest = drill_context["manifest"]
    manifest_path = drill_context["manifest_path"]
    manifest_raw = drill_context["manifest_raw"]
    anchor = drill_context["anchor"]
    anchor_path = drill_context["anchor_path"]
    anchor_raw = drill_context["anchor_raw"]
    contract = drill_context["contract"]
    canonical_job_ids = drill_context["canonical_job_ids"]
    canonical_comments = drill_context["canonical_comments"]
    isolated_root = _canonical_path(
        Path(str(contract["isolated_drill_root"])),
        description="isolated bootstrap drill root",
        kind="directory",
    )

    def require_isolated_artifact(path: Path, description: str) -> Path:
        canonical = _canonical_path(
            path, description=description, kind="file"
        )
        try:
            canonical.relative_to(isolated_root)
        except ValueError as exc:
            raise DeploymentError(
                f"{description} is outside the isolated cancellation-drill root"
            ) from exc
        return canonical

    for index, path in enumerate(cancelled_paths, start=1):
        require_isolated_artifact(
            path, f"cancelled isolated drill observation {index}"
        )
    require_isolated_artifact(repair_path, "isolated bootstrap repair result")
    require_isolated_artifact(
        recovered_path, "isolated bootstrap recovered observation"
    )
    if (
        deployment.get("protocol")
        != BOOTSTRAP_DEPLOYMENT_EVIDENCE_PROTOCOL
        or deployment.get("passed") is not True
        or deployment.get("evidence_id")
        != _self_hash(deployment, "evidence_id")
        or deployment.get("bundle_id") != bundle.get("bundle_id")
        or len(cancelled_paths) != 2
    ):
        raise DeploymentError("bootstrap drill deployment binding is invalid")
    cancelled_records = [
        _validate_bootstrap_observation(
            path=path,
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_raw=manifest_raw,
            anchor_path=anchor_path,
            anchor=anchor,
            cancelled=True,
            scheduler_comment_prefix=str(
                contract["scheduler_comment_prefix"]
            ),
            forbidden_job_ids=canonical_job_ids,
            forbidden_comments=canonical_comments,
        )
        for path in cancelled_paths
    ]
    first, second = (item[0] for item in cancelled_records)
    same_fields = (
        "release_git_commit",
        "release_tag_object",
        "chain_id",
        "chain_manifest",
        "chain_manifest_sha256",
        "submission_receipt",
        "submission_receipt_sha256",
        "submission_receipt_id",
        "generation_provenance",
        "generation_provenance_sha256",
        "generation_provenance_id",
        "anchor_submission_receipt",
        "anchor_submission_receipt_sha256",
        "anchor_submission_receipt_id",
        "repair_generation",
        "root_name",
        "root_job_id",
        "root_names",
        "root_job_ids",
        "roots_held",
        "anchor_armed_authorized",
        "anchor_armed_id",
        "anchor_release_intent_authorized",
        "anchor_release_intent_sha256",
        "descendant_armed",
        "descendant_armed_id",
        "descendant_rearm_required",
        "namespace_lineage_receipt_ids",
        "namespace_lineage_job_ids",
        "namespace_bound_job_ids",
        "job_count",
        "jobs",
    )
    first_time = float(first["observed_at_timestamp"])
    second_time = float(second["observed_at_timestamp"])
    if (
        any(first.get(field) != second.get(field) for field in same_fields)
        or first["observation_id"] == second["observation_id"]
        or second_time - first_time < 60.0
    ):
        raise DeploymentError(
            "cancelled bootstrap observations are not the same generation "
            "with a complete 60-second scheduler cut"
        )
    cancelled_lineage = cancelled_records[0][2]
    for lineage_path, _lineage_receipt, _lineage_raw in cancelled_lineage:
        require_isolated_artifact(
            lineage_path, "isolated bootstrap cancelled receipt lineage"
        )
    require_isolated_artifact(
        Path(str(first["generation_provenance"])),
        "isolated bootstrap cancelled provenance",
    )
    cancelled_receipt_path, cancelled_receipt, cancelled_receipt_raw = (
        cancelled_lineage[-1]
    )
    repair, repair_raw = _sealed_json(
        repair_path, description="bootstrap repair result"
    )
    repair_receipt_path = Path(str(repair.get("submission_receipt", "")))
    require_isolated_artifact(
        repair_receipt_path, "isolated bootstrap repair receipt"
    )
    repair_lineage = _validate_bootstrap_receipt_lineage(
        current_path=repair_receipt_path,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_raw=manifest_raw,
        anchor_path=anchor_path,
        anchor_id=str(anchor["receipt_id"]),
        scheduler_comment_prefix=str(contract["scheduler_comment_prefix"]),
        forbidden_job_ids=canonical_job_ids,
        forbidden_comments=canonical_comments,
    )
    _, repair_receipt, repair_receipt_raw = repair_lineage[-1]
    repair_generation = len(repair_lineage) - 1
    repair_provenance, repair_provenance_raw = (
        _validate_bootstrap_generation_provenance(
            provenance_path=Path(
                str(repair.get("generation_provenance", ""))
            ),
            lineage=repair_lineage,
            manifest=manifest,
            manifest_path=manifest_path,
            manifest_raw=manifest_raw,
        )
    )
    require_isolated_artifact(
        Path(str(repair.get("generation_provenance", ""))),
        "isolated bootstrap repair provenance",
    )
    resubmitted = [
        row
        for row in repair_receipt["jobs"]
        if row.get("generation") == repair_generation
        and row.get("disposition") == "resubmitted"
    ]
    held_root_names = list(repair_receipt.get("held_root_names", []))
    held_root_rows = [
        next(
            (
                row
                for row in repair_receipt["jobs"]
                if row.get("name") == name
            ),
            None,
        )
        for name in held_root_names
    ]
    held_root_job_ids = [
        str(row["job_id"])
        for row in held_root_rows
        if isinstance(row, Mapping)
    ]
    if (
        repair.get("passed") is not True
        or repair.get("status")
        not in {"bootstrap_repaired_held", "bootstrap_repair_reconciled"}
        or any(
            repair.get(key) != value
            for key, value in repair_receipt.items()
        )
        or repair_generation
        != int(cancelled_receipt.get("repair_generation", 0)) + 1
        or repair_receipt.get("parent_receipt")
        != str(cancelled_receipt_path)
        or repair_receipt.get("parent_receipt_sha256")
        != hashlib.sha256(cancelled_receipt_raw).hexdigest()
        or repair.get("parent_submission_receipt_id")
        != cancelled_receipt.get("receipt_id")
        or repair.get("anchor_submission_receipt_id")
        != anchor.get("receipt_id")
        or repair.get("anchor_submission_receipt_sha256")
        != hashlib.sha256(anchor_raw).hexdigest()
        or repair.get("submission_receipt_sha256")
        != hashlib.sha256(repair_receipt_raw).hexdigest()
        or repair.get("submission_receipt_id")
        != repair_receipt.get("receipt_id")
        or repair.get("generation_provenance_sha256")
        != hashlib.sha256(repair_provenance_raw).hexdigest()
        or repair.get("generation_provenance_id")
        != repair_provenance.get("provenance_id")
        or not held_root_names
        or len(held_root_rows) != len(held_root_job_ids)
        or repair.get("root_name") != held_root_names[0]
        or repair.get("root_job_id") != held_root_job_ids[0]
        or repair.get("root_names") != held_root_names
        or repair.get("root_job_ids") != held_root_job_ids
        or repair.get("roots_held") is not True
        or repair.get("roots_held") is not repair.get("root_held")
        or repair.get("root_held") is not True
        or repair.get("root_released") is not False
        or repair.get("root_release_id") is not None
        or repair.get("launch_id") is not None
        or repair.get("anchor_launch_authorized") is not False
        or repair.get("anchor_launch_id") is not None
        or repair.get("anchor_armed_authorized") is not False
        or repair.get("anchor_armed_id") is not None
        or repair.get("anchor_release_intent_authorized") is not False
        or repair.get("descendant_armed") is not None
        or repair.get("descendant_armed_id") is not None
        or not isinstance(
            repair.get("watchdog_scientific_jobs_submitted"), int
        )
        or isinstance(
            repair.get("watchdog_scientific_jobs_submitted"), bool
        )
        or repair.get("watchdog_scientific_jobs_submitted") < 0
    ):
        raise DeploymentError(
            "bootstrap repair result or receipt lineage is invalid"
        )
    journal_path = Path(str(repair_receipt.get("submission_journal", "")))
    require_isolated_artifact(
        journal_path, "isolated bootstrap repair submission journal"
    )
    journal, journal_raw = _sealed_json(
        journal_path, description="bootstrap repair submission journal"
    )
    journal_jobs = journal.get("jobs")
    repair_names = [str(row["name"]) for row in resubmitted]
    if (
        repair_receipt.get("submission_journal_sha256")
        != hashlib.sha256(journal_raw).hexdigest()
        or journal.get("chain_id") != manifest.get("chain_id")
        or journal.get("repair_generation") != repair_generation
        or journal.get("base_receipt") != str(cancelled_receipt_path)
        or journal.get("base_receipt_sha256")
        != hashlib.sha256(cancelled_receipt_raw).hexdigest()
        or journal.get("repair_jobs") != repair_names
        or journal.get("held_root_names") != held_root_names
        or not isinstance(journal_jobs, Mapping)
        or set(journal_jobs) != set(repair_names)
    ):
        raise DeploymentError(
            "bootstrap repair submission journal lineage is invalid"
        )
    intent_keys: list[tuple[str, str, str]] = []
    receipt_by_name = {
        str(row["name"]): row for row in repair_receipt["jobs"]
    }
    for name in repair_names:
        record = journal_jobs[name]
        receipt_row = receipt_by_name[name]
        if (
            not isinstance(record, Mapping)
            or record.get("name") != name
            or record.get("comment") != receipt_row.get("comment")
            or record.get("job_id") != receipt_row.get("job_id")
            or record.get("submission_boundary_state") != "committed"
        ):
            raise DeploymentError(
                "bootstrap repair submission intent is not exactly committed"
            )
        intent_keys.append(
            (
                str(record["job_id"]),
                str(record["comment"]),
                str(record["name"]),
            )
        )
    duplicate_submission_intents = len(intent_keys) - len(set(intent_keys))
    recovered_record = _validate_bootstrap_observation(
        path=recovered_path,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_raw=manifest_raw,
        anchor_path=anchor_path,
        anchor=anchor,
        cancelled=False,
        scheduler_comment_prefix=str(contract["scheduler_comment_prefix"]),
        forbidden_job_ids=canonical_job_ids,
        forbidden_comments=canonical_comments,
    )
    recovered, recovered_raw = recovered_record[:2]
    if (
        recovered.get("submission_receipt_id")
        != repair_receipt.get("receipt_id")
        or recovered.get("submission_receipt_sha256")
        != hashlib.sha256(repair_receipt_raw).hexdigest()
        or recovered.get("repair_generation") != repair_generation
        or recovered.get("generation_provenance_id")
        != repair_provenance.get("provenance_id")
        or recovered.get("generation_provenance_sha256")
        != hashlib.sha256(repair_provenance_raw).hexdigest()
    ):
        raise DeploymentError(
            "recovered observation is unrelated to the exact repair generation"
        )
    recovered_time = float(recovered["observed_at_timestamp"])
    observed_recovery_seconds = recovered_time - first_time
    if (
        not math.isfinite(float(recovery_seconds))
        or not 0 <= float(recovery_seconds) <= 900
        or recovered_time < second_time
        or not 0 <= observed_recovery_seconds <= 900
        or not math.isclose(
            float(recovery_seconds),
            observed_recovery_seconds,
            rel_tol=0.0,
            abs_tol=1.0,
        )
    ):
        raise DeploymentError(
            "bootstrap drill recovery timing is invalid or not observation-bound"
        )
    duplicate_jobs = sum(item[5] for item in cancelled_records)
    duplicate_jobs += recovered_record[5]
    scientific_jobs_started = max(
        [
            int(item[0]["watchdog_scientific_jobs_submitted"])
            for item in cancelled_records
        ]
        + [
            int(repair["watchdog_scientific_jobs_submitted"]),
            int(recovered["watchdog_scientific_jobs_submitted"]),
        ]
    )
    root_remained_held = bool(
        repair.get("root_held")
        and not repair.get("root_released")
        and recovered.get("root_held")
    )
    isolated_job_ids = {
        str(row["job_id"])
        for _path, lineage_receipt, _raw in repair_lineage
        for row in lineage_receipt["jobs"]
    }
    isolated_comments = {
        str(row["comment"])
        for _path, lineage_receipt, _raw in repair_lineage
        for row in lineage_receipt["jobs"]
    }
    job_id_overlap = len(isolated_job_ids & canonical_job_ids)
    comment_overlap = len(isolated_comments & canonical_comments)
    canonical_control_paths = [
        Path(str(contract[field]))
        for field in (
            "canonical_repair_root",
            "canonical_root_release_marker",
            "canonical_launch_marker",
        )
    ]
    canonical_control_paths_absent = not any(
        path.exists() or path.is_symlink() for path in canonical_control_paths
    )
    squeue_complete = all(
        item[0].get("squeue_complete") is True
        for item in [*cancelled_records, recovered_record]
    )
    sacct_complete = all(
        item[0].get("sacct_complete") is True
        for item in [*cancelled_records, recovered_record]
    )
    if (
        job_id_overlap
        or comment_overlap
        or not canonical_control_paths_absent
        or not squeue_complete
        or not sacct_complete
    ):
        raise DeploymentError(
            "isolated cancellation drill overlaps or pollutes canonical recovery"
        )
    return {
        "recovery_namespace_cancellation_recovery_seconds": (
            observed_recovery_seconds
        ),
        "isolated_cancellation_drill": True,
        "isolation_id": contract["isolation_id"],
        "isolated_drill_root": str(isolated_root),
        "isolated_chain_id": manifest["chain_id"],
        "isolated_anchor_submission_receipt_id": anchor["receipt_id"],
        "canonical_submission_receipt_id": contract[
            "canonical_submission_receipt_id"
        ],
        "canonical_job_id_overlap": job_id_overlap,
        "canonical_comment_overlap": comment_overlap,
        "canonical_control_paths_absent": canonical_control_paths_absent,
        "squeue_complete": squeue_complete,
        "sacct_complete": sacct_complete,
        "cancelled_observation_gap_seconds": second_time - first_time,
        "cancelled_observations": [
            _artifact_binding(
                path,
                record[1],
                identity_field="observation_id",
                value=record[0],
            )
            for path, record in zip(
                cancelled_paths, cancelled_records, strict=True
            )
        ],
        "cancelled_observation_ids": [
            record[0]["observation_id"] for record in cancelled_records
        ],
        "cancelled_submission_receipt": _artifact_binding(
            cancelled_receipt_path,
            cancelled_receipt_raw,
            identity_field="receipt_id",
            value=cancelled_receipt,
        ),
        "cancelled_generation_provenance": _artifact_binding(
            Path(str(first["generation_provenance"])),
            cancelled_records[0][4],
            identity_field="provenance_id",
            value=cancelled_records[0][3],
        ),
        "repair_result": _artifact_binding(repair_path, repair_raw),
        "repair_result_sha256": hashlib.sha256(repair_raw).hexdigest(),
        "repair_result_id": hashlib.sha256(repair_raw).hexdigest(),
        "repair_submission_receipt": _artifact_binding(
            repair_receipt_path,
            repair_receipt_raw,
            identity_field="receipt_id",
            value=repair_receipt,
        ),
        "repair_submission_journal": _artifact_binding(
            journal_path, journal_raw
        ),
        "repair_generation_provenance": _artifact_binding(
            Path(str(repair["generation_provenance"])),
            repair_provenance_raw,
            identity_field="provenance_id",
            value=repair_provenance,
        ),
        "recovered_observation": _artifact_binding(
            recovered_path,
            recovered_raw,
            identity_field="observation_id",
            value=recovered,
        ),
        "recovered_observation_id": recovered["observation_id"],
        "recovered_observation_sha256": hashlib.sha256(
            recovered_raw
        ).hexdigest(),
        "repair_generation": repair_generation,
        "held_root_names": held_root_names,
        "held_root_job_ids": held_root_job_ids,
        "duplicate_jobs": duplicate_jobs,
        "duplicate_submission_intents": duplicate_submission_intents,
        "root_remained_held": root_remained_held,
        "scientific_jobs_started": scientific_jobs_started,
        "namespace_scan_complete": all(
            item[0].get("namespace_scan_complete") is True
            for item in cancelled_records
        )
        and recovered.get("namespace_scan_complete") is True,
    }


def _validate_bootstrap_drill_preimages(
    *,
    drill: Mapping[str, Any],
    bundle: Mapping[str, Any],
    deployment: Mapping[str, Any],
) -> dict[str, Any]:
    cancelled = drill.get("cancelled_observations")
    repair = drill.get("repair_result")
    recovered = drill.get("recovered_observation")
    if (
        not isinstance(cancelled, list)
        or len(cancelled) != 2
        or any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("artifact"), str)
            for item in cancelled
        )
        or not isinstance(repair, Mapping)
        or not isinstance(repair.get("artifact"), str)
        or not isinstance(recovered, Mapping)
        or not isinstance(recovered.get("artifact"), str)
    ):
        raise DeploymentError(
            "bootstrap drill evidence lacks exact artifact bindings"
        )
    inspected = _inspect_bootstrap_drill(
        bundle=bundle,
        deployment=deployment,
        cancelled_paths=[
            Path(str(item["artifact"])) for item in cancelled
        ],
        repair_path=Path(str(repair["artifact"])),
        recovered_path=Path(str(recovered["artifact"])),
        recovery_seconds=float(
            drill.get("recovery_namespace_cancellation_recovery_seconds", -1)
        ),
    )
    if any(drill.get(field) != value for field, value in inspected.items()):
        raise DeploymentError(
            "bootstrap drill evidence does not match its sealed preimages"
        )
    return inspected


def capture_bootstrap_drill_evidence(
    *,
    bundle_manifest: Path,
    deployment_evidence: Path,
    cancelled_observations: Sequence[Path],
    repair_result: Path,
    recovered_observation: Path,
    recovery_seconds: float,
    output: Path,
) -> dict[str, Any]:
    """Seal an isolated account-cancellation drill for bootstrap recovery."""

    bundle, _ = _sealed_json(
        bundle_manifest,
        description="bootstrap drill bundle manifest",
    )
    deployment, _ = _sealed_json(
        deployment_evidence,
        description="bootstrap drill deployment evidence",
    )
    if (
        bundle.get("protocol") != BOOTSTRAP_BUNDLE_PROTOCOL
        or deployment.get("protocol")
        != BOOTSTRAP_DEPLOYMENT_EVIDENCE_PROTOCOL
        or deployment.get("passed") is not True
        or deployment.get("bundle_id") != bundle.get("bundle_id")
        or len(cancelled_observations) != 2
    ):
        raise DeploymentError("bootstrap drill inputs are invalid")
    inspected = _inspect_bootstrap_drill(
        bundle=bundle,
        deployment=deployment,
        cancelled_paths=cancelled_observations,
        repair_path=repair_result,
        recovered_path=recovered_observation,
        recovery_seconds=recovery_seconds,
    )
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol": BOOTSTRAP_DRILL_EVIDENCE_PROTOCOL,
        "passed": True,
        "bundle_id": bundle["bundle_id"],
        "deployment_id": deployment["deployment_id"],
        **inspected,
    }
    evidence["evidence_id"] = _self_hash(evidence, "evidence_id")
    path, digest = _publish_once(output, evidence)
    return {
        "passed": True,
        "drill_evidence": str(path),
        "drill_evidence_sha256": digest,
        "evidence_id": evidence["evidence_id"],
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
    if stat.S_IMODE(root.stat(follow_symlinks=False).st_mode) & 0o222:
        raise DeploymentError("installed watchdog release root is mutable")
    observed_paths: list[str] = []
    for directory, directories, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        parent_info = parent.stat(follow_symlinks=False)
        if (
            parent.is_symlink()
            or not stat.S_ISDIR(parent_info.st_mode)
            or stat.S_IMODE(parent_info.st_mode) & 0o222
        ):
            raise DeploymentError(
                f"installed watchdog runtime directory is mutable: {parent}"
            )
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
    _require_single_authorized_key_identity(
        authorized,
        expected_line,
        description="installed production authorized_keys",
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
    bootstrap = subparsers.add_parser("bootstrap-bundle")
    bootstrap.add_argument("--output-root", type=Path, required=True)
    bootstrap.add_argument("--release-root", type=Path, required=True)
    bootstrap.add_argument("--harness-python", type=Path, required=True)
    bootstrap.add_argument(
        "--harness-environment-manifest", type=Path, required=True
    )
    bootstrap.add_argument(
        "--materialization-pilot-marker", type=Path, required=True
    )
    bootstrap.add_argument("--chain-manifest", type=Path, required=True)
    bootstrap.add_argument("--submission-receipt", type=Path, required=True)
    bootstrap.add_argument(
        "--isolated-drill-chain-manifest", type=Path, required=True
    )
    bootstrap.add_argument(
        "--isolated-drill-submission-receipt", type=Path, required=True
    )
    bootstrap.add_argument("--git-commit", required=True)
    bootstrap.add_argument("--tag-object", required=True)
    bootstrap.add_argument("--remote-host", required=True)
    bootstrap.add_argument("--remote-user", required=True)
    bootstrap.add_argument("--identity-file", type=Path, required=True)
    bootstrap.add_argument("--known-hosts-file", type=Path, required=True)
    bootstrap.add_argument("--external-state-root", type=Path, required=True)
    bootstrap.add_argument(
        "--bootstrap-public-key-file", type=Path, required=True
    )
    bootstrap.add_argument(
        "--production-public-key-file", type=Path, required=True
    )
    bootstrap_attest = subparsers.add_parser("bootstrap-attestation")
    bootstrap_attest.add_argument(
        "--bundle-manifest", type=Path, required=True
    )
    bootstrap_attest.add_argument(
        "--deployment-evidence", type=Path, required=True
    )
    bootstrap_attest.add_argument(
        "--scheduler-observation",
        type=Path,
        action="append",
        required=True,
    )
    bootstrap_attest.add_argument(
        "--cancellation-drill-evidence", type=Path, required=True
    )
    bootstrap_attest.add_argument("--output", type=Path, required=True)
    bootstrap_deployment = subparsers.add_parser(
        "bootstrap-deployment-evidence"
    )
    bootstrap_deployment.add_argument(
        "--bundle-manifest", type=Path, required=True
    )
    bootstrap_deployment.add_argument(
        "--installed-release-root", type=Path, required=True
    )
    bootstrap_deployment.add_argument(
        "--vm-python", type=Path, required=True
    )
    bootstrap_deployment.add_argument(
        "--installed-config", type=Path, required=True
    )
    bootstrap_deployment.add_argument(
        "--installed-service", type=Path, required=True
    )
    bootstrap_deployment.add_argument(
        "--installed-timer", type=Path, required=True
    )
    bootstrap_deployment.add_argument(
        "--installed-authorized-keys", type=Path, required=True
    )
    bootstrap_deployment.add_argument(
        "--service-heartbeat", type=Path, required=True
    )
    bootstrap_deployment.add_argument("--output", type=Path, required=True)
    bootstrap_drill = subparsers.add_parser(
        "bootstrap-drill-evidence"
    )
    bootstrap_drill.add_argument(
        "--bundle-manifest", type=Path, required=True
    )
    bootstrap_drill.add_argument(
        "--deployment-evidence", type=Path, required=True
    )
    bootstrap_drill.add_argument(
        "--cancelled-observation",
        type=Path,
        action="append",
        required=True,
    )
    bootstrap_drill.add_argument(
        "--repair-result", type=Path, required=True
    )
    bootstrap_drill.add_argument(
        "--recovered-observation", type=Path, required=True
    )
    bootstrap_drill.add_argument(
        "--recovery-seconds", type=float, required=True
    )
    bootstrap_drill.add_argument("--output", type=Path, required=True)
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
        elif args.command == "bootstrap-bundle":
            result = build_bootstrap_bundle(
                output_root=args.output_root,
                release_root=args.release_root,
                harness_python=args.harness_python,
                harness_environment_manifest=(
                    args.harness_environment_manifest
                ),
                materialization_pilot_marker=(
                    args.materialization_pilot_marker
                ),
                chain_manifest=args.chain_manifest,
                submission_receipt=args.submission_receipt,
                isolated_drill_chain_manifest=(
                    args.isolated_drill_chain_manifest
                ),
                isolated_drill_submission_receipt=(
                    args.isolated_drill_submission_receipt
                ),
                git_commit=args.git_commit,
                tag_object=args.tag_object,
                remote_host=args.remote_host,
                remote_user=args.remote_user,
                identity_file=args.identity_file,
                known_hosts_file=args.known_hosts_file,
                external_state_root=args.external_state_root,
                bootstrap_public_key_file=(
                    args.bootstrap_public_key_file
                ),
                production_public_key_file=(
                    args.production_public_key_file
                ),
            )
        elif args.command == "bootstrap-attestation":
            result = capture_bootstrap_attestation(
                bundle_manifest=args.bundle_manifest,
                deployment_evidence=args.deployment_evidence,
                scheduler_observations=args.scheduler_observation,
                cancellation_drill_evidence=(
                    args.cancellation_drill_evidence
                ),
                output=args.output,
            )
        elif args.command == "bootstrap-deployment-evidence":
            result = capture_bootstrap_deployment_evidence(
                bundle_manifest=args.bundle_manifest,
                installed_release_root=args.installed_release_root,
                vm_python=args.vm_python,
                installed_config=args.installed_config,
                installed_service=args.installed_service,
                installed_timer=args.installed_timer,
                installed_authorized_keys=(
                    args.installed_authorized_keys
                ),
                service_heartbeat=args.service_heartbeat,
                output=args.output,
            )
        elif args.command == "bootstrap-drill-evidence":
            result = capture_bootstrap_drill_evidence(
                bundle_manifest=args.bundle_manifest,
                deployment_evidence=args.deployment_evidence,
                cancelled_observations=args.cancelled_observation,
                repair_result=args.repair_result,
                recovered_observation=args.recovered_observation,
                recovery_seconds=args.recovery_seconds,
                output=args.output,
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
