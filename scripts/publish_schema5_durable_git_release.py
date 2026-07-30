#!/usr/bin/env python3
"""Publish and verify marker-last evidence for the durable schema-5 Git release.

This tool never pushes and never changes a Git ref.  It performs read-only local
and remote Git queries, creates a self-contained bundle in recovery storage, verifies
that bundle, seals a checksum sidecar, and publishes the completion marker last.
The default command is a dry-run; ``--apply`` permits only local evidence writes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


RELEASE_ID = "sweep-recovery-schema5-v1.2"
RELEASE_TAG = "sweep-recovery-schema5-v1.2-r14"
CHAIN_NAMESPACE = "schema5-v1.2-r14"
DURABLE_COMMIT_REF = "refs/heads/schema5-v1.2-r14"
PROTOCOL = "schema5-v1.2-r14-durable-git-release-v1"
MARKER_NAME = "DURABLE_GIT_RELEASE_SCHEMA5_V1_2_R14_COMPLETE.json"
BUNDLE_DIRECTORY = "git_release"
BUNDLE_NAME = f"{RELEASE_TAG}.bundle"
CHECKSUM_NAME = f"{BUNDLE_NAME}.sha256"
_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REMOTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_REMOTE_REF = re.compile(r"refs/(?:heads|releases)/[A-Za-z0-9._/+@-]+\Z")
_TRUSTED_SYSTEM_PATH = "/usr/bin:/bin"
_UNTRUSTED_PROCESS_ENVIRONMENT_KEYS = frozenset(
    {
        "BASH_ENV",
        "CDPATH",
        "ENV",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CEILING_DIRECTORIES",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_EXEC_PATH",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_TEMPLATE_DIR",
        "GIT_WORK_TREE",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "SLURM_CLUSTERS",
        "SLURM_CONF",
        "SLURM_TIME_FORMAT",
    }
)
_UNTRUSTED_PROCESS_ENVIRONMENT_PREFIXES = (
    "BASH_FUNC_",
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
    "SACCT_",
    "SBATCH_",
    "SCONTROL_",
    "SQUEUE_",
)


class DurableGitReleaseError(RuntimeError):
    """The release is not durably and unambiguously published."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sanitized_process_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in _UNTRUSTED_PROCESS_ENVIRONMENT_KEYS
        and not key.startswith(_UNTRUSTED_PROCESS_ENVIRONMENT_PREFIXES)
    }
    environment.update(
        {
            "PATH": _TRUSTED_SYSTEM_PATH,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return environment


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    candidate = dict(value)
    candidate.pop(field, None)
    return _sha256_bytes(_canonical(candidate))


def _canonical_path(path: Path, *, description: str, kind: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if any(character in str(lexical) for character in ("\x00", "\n", "\r")):
        raise DurableGitReleaseError(f"{description} path is unsafe")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DurableGitReleaseError(
            f"{description} is unavailable: {lexical}: {exc}"
        ) from exc
    if lexical != resolved:
        raise DurableGitReleaseError(f"{description} traverses a symlink")
    metadata = lexical.stat(follow_symlinks=False)
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise DurableGitReleaseError(f"{description} is not a regular file")
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise DurableGitReleaseError(f"{description} is not a directory")
    return lexical


def _stable_bytes(
    path: Path, *, description: str, require_read_only: bool = True
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
    current = canonical.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or identity(before) != identity(after)
        or (current.st_dev, current.st_ino) != (after.st_dev, after.st_ino)
        or before.st_nlink != 1
        or (require_read_only and stat.S_IMODE(before.st_mode) & 0o222)
    ):
        raise DurableGitReleaseError(
            f"{description} is mutable, linked, or changed while being read"
        )
    return b"".join(chunks)


def _read_canonical_json(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    raw = _stable_bytes(path, description=description)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DurableGitReleaseError(
                    f"{description} duplicates JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DurableGitReleaseError(
                    f"{description} contains non-finite value {token}"
                )
            ),
        )
    except DurableGitReleaseError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DurableGitReleaseError(f"{description} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or raw != _canonical(value):
        raise DurableGitReleaseError(f"{description} is not canonical JSON")
    return value, raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    target = Path(os.path.abspath(os.fspath(path.expanduser())))
    target.parent.mkdir(parents=True, exist_ok=True)
    parent = _canonical_path(
        target.parent, description=f"{target.name} parent", kind="directory"
    )
    lock = parent / f".{target.name}.publish.lock"
    descriptor = os.open(
        lock,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if target.exists() or target.is_symlink():
            if target.is_symlink() or _stable_bytes(
                target, description=f"existing {target.name}"
            ) != payload:
                raise DurableGitReleaseError(
                    f"immutable release evidence conflicts: {target}"
                )
            return
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".publishing",
            dir=parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(temporary_descriptor, mode)
            with os.fdopen(temporary_descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            _fsync_directory(parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    finally:
        os.close(descriptor)


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    description: str,
    timeout: float = 300,
) -> str:
    environment = _sanitized_process_environment()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DurableGitReleaseError(f"{description} failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise DurableGitReleaseError(
            f"{description} failed rc={completed.returncode}: {detail[:1000]}"
        )
    return completed.stdout.strip()


def _local_identity(repository: Path) -> dict[str, str]:
    repository = _canonical_path(
        repository, description="release repository", kind="directory"
    )
    if not (repository / ".git").exists():
        raise DurableGitReleaseError("release repository is not a Git checkout")
    tag_ref = f"refs/tags/{RELEASE_TAG}"
    tag_type = _run(
        ["/usr/bin/git", "cat-file", "-t", tag_ref],
        cwd=repository,
        description="annotated-tag type query",
    )
    tag_object = _run(
        ["/usr/bin/git", "rev-parse", "--verify", tag_ref],
        cwd=repository,
        description="annotated-tag object query",
    )
    commit = _run(
        ["/usr/bin/git", "rev-parse", "--verify", f"{tag_ref}^{{commit}}"],
        cwd=repository,
        description="annotated-tag commit query",
    )
    head = _run(
        ["/usr/bin/git", "rev-parse", "--verify", "HEAD"],
        cwd=repository,
        description="HEAD query",
    )
    status = _run(
        ["/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        description="clean-worktree query",
    )
    replacement_refs = _run(
        [
            "/usr/bin/git",
            "for-each-ref",
            "--format=%(refname)",
            "refs/replace",
        ],
        cwd=repository,
        description="replacement-ref query",
    )
    if replacement_refs:
        raise DurableGitReleaseError(
            "release checkout contains forbidden Git replacement refs"
        )
    if (
        tag_type != "tag"
        or _OBJECT.fullmatch(tag_object) is None
        or _OBJECT.fullmatch(commit) is None
        or head != commit
        or status
    ):
        raise DurableGitReleaseError(
            "release checkout must be clean at the exact annotated release tag"
        )
    return {
        "release_git_commit": commit,
        "release_tag_object": tag_object,
    }


def _remote_identity(
    repository: Path,
    *,
    remote: str,
    remote_commit_ref: str,
    local: Mapping[str, str],
) -> dict[str, str]:
    if (
        _REMOTE_NAME.fullmatch(remote) is None
        or _REMOTE_REF.fullmatch(remote_commit_ref) is None
        or remote_commit_ref != DURABLE_COMMIT_REF
    ):
        raise DurableGitReleaseError(
            "remote name is unsafe or durable commit ref is not the exact "
            f"{DURABLE_COMMIT_REF}"
        )
    remote_url = _run(
        ["/usr/bin/git", "remote", "get-url", remote],
        cwd=repository,
        description="remote URL query",
    )
    tag_ref = f"refs/tags/{RELEASE_TAG}"
    output = _run(
        [
            "/usr/bin/git",
            "ls-remote",
            "--exit-code",
            remote,
            remote_commit_ref,
            tag_ref,
            f"{tag_ref}^{{}}",
        ],
        cwd=repository,
        description="read-only remote release query",
    )
    rows: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split("\t")
        if (
            len(fields) != 2
            or _OBJECT.fullmatch(fields[0]) is None
            or fields[1] in rows
        ):
            raise DurableGitReleaseError("remote release query is ambiguous")
        rows[fields[1]] = fields[0]
    expected = {
        remote_commit_ref: local["release_git_commit"],
        tag_ref: local["release_tag_object"],
        f"{tag_ref}^{{}}": local["release_git_commit"],
    }
    if rows != expected:
        raise DurableGitReleaseError(
            "remote does not advertise the exact commit, tag object, and peeled tag"
        )
    return {
        "remote": remote,
        "remote_commit_ref": remote_commit_ref,
        "remote_commit": rows[remote_commit_ref],
        "remote_tag_object": rows[tag_ref],
        "remote_peeled_commit": rows[f"{tag_ref}^{{}}"],
        "remote_url_sha256": _sha256_bytes(remote_url.encode("utf-8")),
    }


def _verify_bundle(
    bundle: Path,
    *,
    repository: Path,
    expected_tag_object: str,
) -> tuple[str, int]:
    raw = _stable_bytes(bundle, description="Git release bundle")
    _run(
        ["/usr/bin/git", "bundle", "verify", str(bundle)],
        cwd=repository,
        description="Git bundle verification",
    )
    heads = _run(
        [
            "/usr/bin/git",
            "bundle",
            "list-heads",
            str(bundle),
            f"refs/tags/{RELEASE_TAG}",
        ],
        cwd=repository,
        description="Git bundle tag query",
    )
    if heads != f"{expected_tag_object} refs/tags/{RELEASE_TAG}":
        raise DurableGitReleaseError("Git bundle does not contain the exact tag object")
    return _sha256_bytes(raw), len(raw)


def _validate_marker_fields(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "protocol",
        "passed",
        "release_id",
        "release_tag",
        "release_git_commit",
        "release_tag_object",
        "chain_namespace",
        "clean_checkout",
        "annotated_tag",
        "remote_query_read_only",
        "remote",
        "remote_commit_ref",
        "remote_commit",
        "remote_tag_object",
        "remote_peeled_commit",
        "remote_url_sha256",
        "bundle_path",
        "bundle_sha256",
        "bundle_size",
        "checksum_path",
        "checksum_sha256",
        "published_at",
        "marker_id",
    }
    if (
        set(value) != required
        or value.get("schema_version") != 1
        or value.get("protocol") != PROTOCOL
        or value.get("passed") is not True
        or value.get("release_id") != RELEASE_ID
        or value.get("release_tag") != RELEASE_TAG
        or value.get("chain_namespace") != CHAIN_NAMESPACE
        or value.get("clean_checkout") is not True
        or value.get("annotated_tag") is not True
        or value.get("remote_query_read_only") is not True
        or value.get("remote_commit_ref") != DURABLE_COMMIT_REF
        or any(
            _OBJECT.fullmatch(str(value.get(field, ""))) is None
            for field in (
                "release_git_commit",
                "release_tag_object",
                "remote_commit",
                "remote_tag_object",
                "remote_peeled_commit",
            )
        )
        or value.get("remote_commit") != value.get("release_git_commit")
        or value.get("remote_peeled_commit") != value.get("release_git_commit")
        or value.get("remote_tag_object") != value.get("release_tag_object")
        or _REMOTE_NAME.fullmatch(str(value.get("remote", ""))) is None
        or _REMOTE_REF.fullmatch(str(value.get("remote_commit_ref", ""))) is None
        or any(
            _SHA256.fullmatch(str(value.get(field, ""))) is None
            for field in (
                "remote_url_sha256",
                "bundle_sha256",
                "checksum_sha256",
                "marker_id",
            )
        )
        or not isinstance(value.get("bundle_size"), int)
        or isinstance(value.get("bundle_size"), bool)
        or value["bundle_size"] <= 0
        or not isinstance(value.get("published_at"), str)
        or not value["published_at"]
        or value.get("marker_id") != _self_hash(value, "marker_id")
    ):
        raise DurableGitReleaseError("durable Git release marker identity is invalid")


def marker_binding(marker_path: Path) -> dict[str, Any]:
    """Verify sealed marker/bundle bytes and return a portable launch binding."""

    marker, marker_raw = _read_canonical_json(
        marker_path, description="durable Git release marker"
    )
    _validate_marker_fields(marker)
    marker_path = _canonical_path(
        marker_path, description="durable Git release marker", kind="file"
    )
    root = marker_path.parent
    expected_bundle = root / BUNDLE_DIRECTORY / BUNDLE_NAME
    expected_checksum = root / BUNDLE_DIRECTORY / CHECKSUM_NAME
    bundle_path = Path(str(marker["bundle_path"]))
    checksum_path = Path(str(marker["checksum_path"]))
    if bundle_path != expected_bundle or checksum_path != expected_checksum:
        raise DurableGitReleaseError("durable Git release artifact paths drifted")
    bundle_raw = _stable_bytes(bundle_path, description="durable Git bundle")
    checksum_raw = _stable_bytes(
        checksum_path, description="durable Git bundle checksum"
    )
    expected_checksum_raw = (
        f"{marker['bundle_sha256']}  {BUNDLE_NAME}\n".encode("ascii")
    )
    if (
        len(bundle_raw) != marker["bundle_size"]
        or _sha256_bytes(bundle_raw) != marker["bundle_sha256"]
        or checksum_raw != expected_checksum_raw
        or _sha256_bytes(checksum_raw) != marker["checksum_sha256"]
    ):
        raise DurableGitReleaseError("durable Git bundle or checksum drifted")
    return {
        "path": str(marker_path),
        "sha256": _sha256_bytes(marker_raw),
        "marker_id": marker["marker_id"],
        "release_git_commit": marker["release_git_commit"],
        "release_tag_object": marker["release_tag_object"],
        "bundle_sha256": marker["bundle_sha256"],
    }


def publish(
    *,
    repository: Path,
    recovery_root: Path,
    remote: str,
    remote_commit_ref: str,
    apply: bool = False,
) -> dict[str, Any]:
    """Verify local/remote release identity and optionally publish sealed evidence."""

    repository = _canonical_path(
        repository, description="release repository", kind="directory"
    )
    recovery_root = Path(os.path.abspath(os.fspath(recovery_root.expanduser())))
    local = _local_identity(repository)
    remote_identity = _remote_identity(
        repository,
        remote=remote,
        remote_commit_ref=remote_commit_ref,
        local=local,
    )
    marker_path = recovery_root / MARKER_NAME
    bundle_path = recovery_root / BUNDLE_DIRECTORY / BUNDLE_NAME
    checksum_path = recovery_root / BUNDLE_DIRECTORY / CHECKSUM_NAME
    if marker_path.exists() or marker_path.is_symlink():
        marker, _marker_raw = _read_canonical_json(
            marker_path, description="durable Git release marker"
        )
        _validate_marker_fields(marker)
        binding = marker_binding(marker_path)
        if (
            binding["release_git_commit"] != local["release_git_commit"]
            or binding["release_tag_object"] != local["release_tag_object"]
            or any(
                marker.get(field) != remote_identity.get(field)
                for field in (
                    "remote",
                    "remote_commit_ref",
                    "remote_commit",
                    "remote_tag_object",
                    "remote_peeled_commit",
                    "remote_url_sha256",
                )
            )
        ):
            raise DurableGitReleaseError(
                "existing durable release belongs to another local or remote identity"
            )
        return {
            "status": "already_complete",
            "apply": apply,
            "passed": True,
            **binding,
        }
    if not apply:
        return {
            "status": "dry_run",
            "apply": False,
            "passed": True,
            **local,
            **remote_identity,
            "would_publish": [
                str(bundle_path),
                str(checksum_path),
                str(marker_path),
            ],
        }

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_parent = _canonical_path(
        bundle_path.parent, description="Git bundle evidence root", kind="directory"
    )
    if bundle_path.exists() or bundle_path.is_symlink():
        if bundle_path.is_symlink():
            raise DurableGitReleaseError("existing Git bundle is symlinked")
    else:
        with tempfile.TemporaryDirectory(
            prefix=".git-release-bundle.", dir=bundle_parent
        ) as temporary_name:
            temporary_bundle = Path(temporary_name) / BUNDLE_NAME
            _run(
                [
                    "/usr/bin/git",
                    "bundle",
                    "create",
                    str(temporary_bundle),
                    f"refs/tags/{RELEASE_TAG}",
                ],
                cwd=repository,
                description="Git bundle creation",
            )
            temporary_bundle.chmod(0o444)
            _verify_bundle(
                temporary_bundle,
                repository=repository,
                expected_tag_object=local["release_tag_object"],
            )
            os.replace(temporary_bundle, bundle_path)
            _fsync_directory(bundle_parent)
    bundle_sha256, bundle_size = _verify_bundle(
        bundle_path,
        repository=repository,
        expected_tag_object=local["release_tag_object"],
    )
    checksum_raw = f"{bundle_sha256}  {BUNDLE_NAME}\n".encode("ascii")
    _publish_once(checksum_path, checksum_raw)
    marker: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "passed": True,
        "release_id": RELEASE_ID,
        "release_tag": RELEASE_TAG,
        **local,
        "chain_namespace": CHAIN_NAMESPACE,
        "clean_checkout": True,
        "annotated_tag": True,
        "remote_query_read_only": True,
        **remote_identity,
        "bundle_path": str(bundle_path),
        "bundle_sha256": bundle_sha256,
        "bundle_size": bundle_size,
        "checksum_path": str(checksum_path),
        "checksum_sha256": _sha256_bytes(checksum_raw),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    marker["marker_id"] = _self_hash(marker, "marker_id")
    _publish_once(marker_path, _canonical(marker))
    binding = marker_binding(marker_path)
    return {
        "status": "complete",
        "apply": True,
        "passed": True,
        **binding,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--remote-commit-ref", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = publish(
            repository=args.repository,
            recovery_root=args.recovery_root,
            remote=args.remote,
            remote_commit_ref=args.remote_commit_ref,
            apply=args.apply,
        )
    except (DurableGitReleaseError, OSError, ValueError) as exc:
        print(f"[schema5-durable-git-release] ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
