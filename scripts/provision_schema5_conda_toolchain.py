#!/usr/bin/env python3
"""Provision and verify the isolated offline Conda toolchain for schema-5 r9.

The cached Miniforge installer is treated as an immutable input, never as an
authorization to repair or query an existing Conda installation.  ``provision`` is
read-only unless ``--apply`` is supplied.  The apply path:

* copies the exact pinned installer with buffered read/write I/O;
* installs it into a fresh, release-namespace-local prefix;
* rejects unsafe links and regular files whose inodes are shared outside that prefix;
* removes write bits, inventories the complete prefix before and after two offline
  command probes, and requires the inventories to be identical; and
* publishes ``CONDA_TOOLCHAIN_COMPLETE.json`` last.

The completion marker and copied installer are sufficient for later verification.
Verification invokes only the sealed toolchain itself; it does not consult the cached
installer, any developer environment, or an external/shared Conda base.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


_REPOSITORY = Path(__file__).resolve().parent.parent
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from scripts import schema5_conda_runtime_identity as runtime_identity  # noqa: E402


SCHEMA_VERSION = 1
PROTOCOL = "schema5-v1.2-r9-offline-conda-toolchain-v1"
RELEASE_TAG = "sweep-recovery-schema5-v1.2-r9"
CHAIN_NAMESPACE = "schema5-v1.2-r9"
MARKER_NAME = "CONDA_TOOLCHAIN_COMPLETE.json"
INTENT_NAME = "CONDA_TOOLCHAIN_PROVISION_INTENT.json"
TOOLCHAIN_NAMESPACE_DIRECTORY = "r9"
TOOLCHAIN_DIRECTORY_NAME = "conda"
TRANSACTION_DIRECTORY_NAME = f".{TOOLCHAIN_DIRECTORY_NAME}.provisioning"
PINNED_INSTALLER_FILENAME = "Miniforge3-Linux-x86_64.sh"
PINNED_INSTALLER_RELEASE = "Miniforge3-25.11.0-1"
PINNED_INSTALLER_SHA256 = (
    "be1bad9d4e67a8753eb76fb4940e9a08036786675c7adf060627e55791bf110d"
)
PINNED_CONDA_VERSION = "25.11.0"
DEFAULT_INSTALLER = Path(
    "/orcd/data/lhtsai/001/om2/mabdel03/Miniforge3-Linux-x86_64.sh"
)
DEFAULT_FORBIDDEN_PREFIXES = (
    Path("/orcd/home/002/mabdel03/conda_envs/asys_env"),
    Path("/orcd/home/002/mabdel03/conda_envs/serve_env"),
    Path("/orcd/data/lhtsai/001/om2/mabdel03/miniforge3"),
)
MAX_PORTABLE_SHEBANG_BYTES = 127
_CHUNK_SIZE = 8 * 1024 * 1024


class CondaToolchainProvisionError(RuntimeError):
    """The isolated Conda toolchain cannot be trusted or safely provisioned."""


@dataclass(frozen=True)
class InstallerContract:
    filename: str
    release: str
    sha256: str
    conda_version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "filename": self.filename,
            "release": self.release,
            "sha256": self.sha256,
            "conda_version": self.conda_version,
        }


PINNED_INSTALLER_CONTRACT = InstallerContract(
    filename=PINNED_INSTALLER_FILENAME,
    release=PINNED_INSTALLER_RELEASE,
    sha256=PINNED_INSTALLER_SHA256,
    conda_version=PINNED_CONDA_VERSION,
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    candidate = dict(value)
    candidate.pop(field, None)
    return _sha256_bytes(_canonical_bytes(candidate))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_lexical(path: str | Path, *, description: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if any(character in str(candidate) for character in ("\x00", "\n", "\r")):
        raise CondaToolchainProvisionError(f"{description} path is unsafe")
    return candidate


def _existing_canonical(
    path: str | Path, *, description: str, kind: str
) -> Path:
    candidate = _safe_lexical(path, description=description)
    try:
        resolved = candidate.resolve(strict=True)
        metadata = candidate.stat(follow_symlinks=False)
    except (OSError, RuntimeError) as exc:
        raise CondaToolchainProvisionError(
            f"{description} is unavailable: {candidate}: {exc}"
        ) from exc
    if candidate != resolved or stat.S_ISLNK(metadata.st_mode):
        raise CondaToolchainProvisionError(f"{description} traverses a symlink")
    if kind == "file" and not stat.S_ISREG(metadata.st_mode):
        raise CondaToolchainProvisionError(f"{description} is not a regular file")
    if kind == "directory" and not stat.S_ISDIR(metadata.st_mode):
        raise CondaToolchainProvisionError(f"{description} is not a directory")
    return candidate


def _new_child_path(
    parent: Path, name: str, *, description: str, allow_existing: bool
) -> Path:
    canonical_parent = _existing_canonical(
        parent, description=f"{description} parent", kind="directory"
    )
    candidate = _safe_lexical(canonical_parent / name, description=description)
    if candidate.parent != canonical_parent or candidate.name != name:
        raise CondaToolchainProvisionError(f"{description} is not an exact child")
    if candidate.is_symlink():
        raise CondaToolchainProvisionError(f"{description} is symlinked")
    if candidate.exists():
        if not allow_existing:
            raise CondaToolchainProvisionError(f"{description} already exists")
        _existing_canonical(candidate, description=description, kind="directory")
    return candidate


def _validate_contract(contract: InstallerContract) -> None:
    if (
        not contract.filename
        or Path(contract.filename).name != contract.filename
        or "/" in contract.filename
        or len(contract.sha256) != 64
        or any(character not in "0123456789abcdef" for character in contract.sha256)
        or not contract.release
        or not contract.conda_version
    ):
        raise CondaToolchainProvisionError("installer contract is malformed")


def _stable_regular(
    path: str | Path,
    *,
    description: str,
    require_read_only: bool = False,
    require_single_link: bool = False,
) -> tuple[Path, os.stat_result, bytes]:
    canonical = _existing_canonical(path, description=description, kind="file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(canonical, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while block := os.read(descriptor, _CHUNK_SIZE):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        not stat.S_ISREG(before.st_mode)
        or identity(before) != identity(after)
        or (
            canonical.stat(follow_symlinks=False).st_dev,
            canonical.stat(follow_symlinks=False).st_ino,
        )
        != (after.st_dev, after.st_ino)
        or (require_read_only and stat.S_IMODE(after.st_mode) & 0o222)
        or (require_single_link and after.st_nlink != 1)
    ):
        raise CondaToolchainProvisionError(
            f"{description} is mutable, linked, or changed while being read"
        )
    return canonical, after, b"".join(chunks)


def _stable_file_identity(
    path: str | Path,
    *,
    description: str,
    require_read_only: bool = False,
    require_single_link: bool = False,
) -> dict[str, Any]:
    canonical, metadata, raw = _stable_regular(
        path,
        description=description,
        require_read_only=require_read_only,
        require_single_link=require_single_link,
    )
    return {
        "path": str(canonical),
        "sha256": _sha256_bytes(raw),
        "size": len(raw),
        "mode": stat.S_IMODE(metadata.st_mode),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "link_count": metadata.st_nlink,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(
    path: Path,
    value: Mapping[str, Any],
    *,
    description: str,
    mode: int = 0o444,
) -> bytes:
    payload = _canonical_bytes(value)
    parent = _existing_canonical(
        path.parent, description=f"{description} parent", kind="directory"
    )
    if path.exists() or path.is_symlink():
        _, _, current = _stable_regular(
            path, description=f"existing {description}", require_read_only=True
        )
        if current != payload:
            raise CondaToolchainProvisionError(
                f"immutable {description} conflicts: {path}"
            )
        return current
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".publishing", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path)
        _fsync_directory(parent)
    except FileExistsError as exc:
        raise CondaToolchainProvisionError(
            f"{description} appeared concurrently: {path}"
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return payload


def _read_canonical_json(
    path: Path, *, description: str, require_read_only: bool = True
) -> tuple[dict[str, Any], bytes]:
    _, _, raw = _stable_regular(
        path, description=description, require_read_only=require_read_only
    )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CondaToolchainProvisionError(
                    f"{description} duplicates JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CondaToolchainProvisionError(
                    f"{description} contains non-finite value {token}"
                )
            ),
        )
    except CondaToolchainProvisionError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CondaToolchainProvisionError(
            f"{description} is invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise CondaToolchainProvisionError(f"{description} is not canonical JSON")
    return value, raw


def _buffered_copy(
    source: Path,
    destination: Path,
    *,
    description: str,
    staging_directory: Path | None = None,
) -> dict[str, Any]:
    source_before = _stable_file_identity(source, description=description)
    if destination.exists() or destination.is_symlink():
        destination_identity = _stable_file_identity(
            destination,
            description=f"existing copied {description}",
            require_read_only=True,
            require_single_link=True,
        )
        if (
            destination_identity["sha256"] != source_before["sha256"]
            or destination_identity["size"] != source_before["size"]
        ):
            raise CondaToolchainProvisionError(
                f"existing copied {description} conflicts"
            )
    else:
        staging_parent = _existing_canonical(
            staging_directory or destination.parent,
            description=f"{description} copy-staging directory",
            kind="directory",
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".buffered-copy",
            dir=staging_parent,
        )
        temporary = Path(temporary_name)
        try:
            source_descriptor = os.open(
                source,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                before = os.fstat(source_descriptor)
                with os.fdopen(descriptor, "wb") as output:
                    while block := os.read(source_descriptor, _CHUNK_SIZE):
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                after = os.fstat(source_descriptor)
            finally:
                os.close(source_descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise CondaToolchainProvisionError(
                    f"{description} changed during buffered copy"
                )
            os.chmod(temporary, 0o400)
            os.link(temporary, destination)
            _fsync_directory(destination.parent)
        except FileExistsError as exc:
            raise CondaToolchainProvisionError(
                f"copied {description} appeared concurrently"
            ) from exc
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        destination_identity = _stable_file_identity(
            destination,
            description=f"copied {description}",
            require_read_only=True,
            require_single_link=True,
        )
    source_after = _stable_file_identity(source, description=description)
    if (
        source_before != source_after
        or source_before["sha256"] != destination_identity["sha256"]
        or source_before["size"] != destination_identity["size"]
        or (source_before["device"], source_before["inode"])
        == (destination_identity["device"], destination_identity["inode"])
    ):
        raise CondaToolchainProvisionError(
            f"{description} copy is unstable, non-identical, or inode-shared"
        )
    return {
        "source": source_before,
        "copy": destination_identity,
        "buffered_copy": True,
        "shared_inode": False,
        "source_reverified_after_copy": True,
    }


def _hash_regular(path: Path) -> tuple[os.stat_result, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CondaToolchainProvisionError(
                f"toolchain entry is not a regular file: {path}"
            )
        while block := os.read(descriptor, _CHUNK_SIZE):
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise CondaToolchainProvisionError(
            f"toolchain file changed while hashing: {path}"
        )
    return after, digest.hexdigest()


def _complete_prefix_inventory(prefix: Path) -> dict[str, Any]:
    root = _existing_canonical(
        prefix, description="isolated Conda base prefix", kind="directory"
    )
    records: list[dict[str, Any]] = []
    inode_paths: dict[tuple[int, int], list[str]] = {}
    inode_link_counts: dict[tuple[int, int], int] = {}
    file_count = 0
    directory_count = 0
    symlink_count = 0
    runtime_symlink_count = 0
    excluded_cache_symlink_count = 0
    unresolved_cache_symlink_count = 0
    total_bytes = 0

    def walk_error(error: OSError) -> None:
        raise CondaToolchainProvisionError(
            f"cannot traverse isolated Conda base: {error}"
        ) from error

    for directory, directory_names, file_names in os.walk(
        root, followlinks=False, onerror=walk_error
    ):
        directory_names[:] = sorted(directory_names)
        for name in sorted((*directory_names, *file_names)):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                directory_count += 1
                records.append(
                    {"path": relative, "type": "directory", "mode": mode}
                )
            elif stat.S_ISREG(metadata.st_mode):
                stable, digest = _hash_regular(path)
                file_count += 1
                total_bytes += stable.st_size
                inode = (stable.st_dev, stable.st_ino)
                inode_paths.setdefault(inode, []).append(relative)
                inode_link_counts[inode] = stable.st_nlink
                records.append(
                    {
                        "path": relative,
                        "type": "file",
                        "mode": stat.S_IMODE(stable.st_mode),
                        "size": stable.st_size,
                        "sha256": digest,
                    }
                )
            elif stat.S_ISLNK(metadata.st_mode):
                relative_path = Path(relative)
                excluded_cache = (
                    bool(relative_path.parts)
                    and relative_path.parts[0]
                    in runtime_identity.EXCLUDED_TOP_LEVEL
                )
                try:
                    target = os.readlink(path)
                    resolved = path.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    if not excluded_cache:
                        raise CondaToolchainProvisionError(
                            f"unsafe or unresolved runtime symlink {path}: {exc}"
                        ) from exc
                    target = os.readlink(path)
                    resolved = None
                    unresolved_cache_symlink_count += 1
                if resolved is not None and not _is_relative_to(resolved, root):
                    raise CondaToolchainProvisionError(
                        f"toolchain symlink escapes base prefix: {path} -> {target}"
                    )
                symlink_count += 1
                if excluded_cache:
                    excluded_cache_symlink_count += 1
                else:
                    runtime_symlink_count += 1
                records.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "mode": mode,
                        "target": target,
                        "resolved": (
                            resolved.relative_to(root).as_posix()
                            if resolved is not None
                            else None
                        ),
                        "scope": (
                            "excluded_cache"
                            if excluded_cache
                            else "complete_runtime"
                        ),
                    }
                )
            else:
                raise CondaToolchainProvisionError(
                    f"unsupported special file in toolchain: {path}"
                )
    external_links = [
        {
            "paths": sorted(paths),
            "observed_paths": len(paths),
            "inode_link_count": inode_link_counts[inode],
        }
        for inode, paths in inode_paths.items()
        if inode_link_counts[inode] != len(paths)
    ]
    if external_links:
        raise CondaToolchainProvisionError(
            "toolchain has regular-file inodes shared outside its base prefix"
        )
    records.sort(key=lambda item: (item["path"], item["type"]))
    hardlink_groups = sum(len(paths) > 1 for paths in inode_paths.values())
    return {
        "inventory_sha256": _sha256_bytes(_canonical_bytes(records)),
        "entry_count": len(records),
        "file_count": file_count,
        "directory_count": directory_count,
        "symlink_count": symlink_count,
        "runtime_symlink_count": runtime_symlink_count,
        "excluded_cache_symlink_count": excluded_cache_symlink_count,
        "unresolved_cache_symlink_count": unresolved_cache_symlink_count,
        "total_bytes": total_bytes,
        "hardlink_group_count": hardlink_groups,
        "external_shared_inode_count": 0,
        "all_runtime_symlinks_resolve_inside_prefix": True,
    }


def _runtime_symlink_audit(prefix: Path) -> dict[str, Any]:
    """Audit every runtime link while deliberately excluding Conda caches.

    Extracted package caches can contain package-relative links whose targets exist
    only after packages are linked into an environment.  They remain fully byte- and
    hardlink-inventoried, but are not executable runtime dependencies.  This is the
    same explicit cache boundary used by ``conda_runtime_identity``.
    """

    root = _existing_canonical(
        prefix, description="isolated Conda runtime root", kind="directory"
    )
    for excluded_name in runtime_identity.EXCLUDED_TOP_LEVEL:
        excluded = root / excluded_name
        if not (excluded.exists() or excluded.is_symlink()):
            continue
        metadata = excluded.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise CondaToolchainProvisionError(
                "declared Conda cache boundary is not a real internal directory: "
                f"{excluded}"
            )
    try:
        identity = runtime_identity.conda_runtime_identity(
            root / "bin" / "conda"
        )
    except (
        OSError,
        runtime_identity.CondaRuntimeIdentityError,
    ) as exc:
        raise CondaToolchainProvisionError(
            f"isolated Conda runtime symlink audit failed: {exc}"
        ) from exc
    inventory = identity["runtime_inventory"]

    def resolve_dependency(path: Path) -> Path:
        try:
            pending = list(path.relative_to(root).parts)
        except ValueError as exc:
            raise CondaToolchainProvisionError(
                f"runtime symlink candidate escapes its root: {path}"
            ) from exc
        current = root
        visited: set[Path] = set()
        hop_count = 0
        while pending:
            component = pending.pop(0)
            if component in {"", "."}:
                continue
            if component == "..":
                parent = current.parent
                if not _is_relative_to(parent, root):
                    raise CondaToolchainProvisionError(
                        f"runtime symlink escapes through '..': {path}"
                    )
                current = parent
                continue
            candidate = current / component
            if not _is_relative_to(candidate, root):
                raise CondaToolchainProvisionError(
                    f"runtime symlink traverses an external path: {path}"
                )
            try:
                metadata = candidate.lstat()
            except OSError as exc:
                raise CondaToolchainProvisionError(
                    f"runtime symlink is unresolved: {path}: {exc}"
                ) from exc
            if not stat.S_ISLNK(metadata.st_mode):
                current = candidate
                continue
            hop_count += 1
            if candidate in visited or hop_count > 64:
                raise CondaToolchainProvisionError(
                    f"runtime symlink has a cycle or excessive hop chain: {path}"
                )
            visited.add(candidate)
            try:
                raw_target = os.readlink(candidate)
            except OSError as exc:
                raise CondaToolchainProvisionError(
                    f"cannot read runtime symlink {candidate}: {exc}"
                ) from exc
            target = Path(raw_target)
            if target.is_absolute():
                try:
                    target_parts = list(target.relative_to(root).parts)
                except ValueError as exc:
                    raise CondaToolchainProvisionError(
                        "runtime symlink depends on an external absolute path: "
                        f"{candidate} -> {target}"
                    ) from exc
                current = root
            else:
                target_parts = list(target.parts)
                current = candidate.parent
            pending = [*target_parts, *pending]
        return current

    audited_links = 0

    def walk_error(error: OSError) -> None:
        raise CondaToolchainProvisionError(
            f"cannot traverse isolated Conda runtime: {error}"
        ) from error

    for directory, directory_names, file_names in os.walk(
        root, followlinks=False, onerror=walk_error
    ):
        current = Path(directory)
        if current == root:
            directory_names[:] = [
                name
                for name in sorted(directory_names)
                if name not in runtime_identity.EXCLUDED_TOP_LEVEL
            ]
            file_names = [
                name
                for name in sorted(file_names)
                if name not in runtime_identity.EXCLUDED_TOP_LEVEL
            ]
        else:
            directory_names[:] = sorted(directory_names)
            file_names = sorted(file_names)
        for name in (*directory_names, *file_names):
            path = current / name
            if not stat.S_ISLNK(path.lstat().st_mode):
                continue
            audited_links += 1
            resolved = resolve_dependency(path)
            if not _is_relative_to(resolved, root):
                raise CondaToolchainProvisionError(
                    f"runtime symlink resolves outside its root: {path}"
                )
    if audited_links != inventory["symlink_count"]:
        raise CondaToolchainProvisionError(
            "runtime symlink census differs from complete runtime identity"
        )
    return {
        "audit_scope": "complete_runtime_excluding_declared_conda_caches",
        "excluded_top_level": list(runtime_identity.EXCLUDED_TOP_LEVEL),
        "destination_symlink_count": audited_links,
        "destination_internal_symlink_count": audited_links,
        "destination_external_symlink_count": 0,
        "unresolvable_symlink_count": 0,
        "validated_twice": True,
    }


def _require_recursively_read_only(root: Path, *, description: str) -> None:
    canonical = _existing_canonical(root, description=description, kind="directory")
    for directory, directory_names, file_names in os.walk(
        canonical, followlinks=False
    ):
        directory_names[:] = sorted(directory_names)
        for path in [
            Path(directory),
            *(Path(directory) / name for name in directory_names),
            *(Path(directory) / name for name in sorted(file_names)),
        ]:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                continue
            if stat.S_IMODE(metadata.st_mode) & 0o222:
                raise CondaToolchainProvisionError(
                    f"{description} contains writable entry: {path}"
                )


def _seal_recursively(root: Path, *, seal_root: bool = True) -> None:
    canonical = _existing_canonical(
        root, description="toolchain seal root", kind="directory"
    )
    directories: list[Path] = []
    for directory, directory_names, file_names in os.walk(
        canonical, topdown=True, followlinks=False
    ):
        directory_names[:] = sorted(directory_names)
        current = Path(directory)
        directories.append(current)
        for name in sorted(file_names):
            path = current / name
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode):
                os.chmod(path, stat.S_IMODE(metadata.st_mode) & ~0o222)
            elif not stat.S_ISLNK(metadata.st_mode):
                raise CondaToolchainProvisionError(
                    f"cannot seal special toolchain entry: {path}"
                )
    for directory in reversed(directories):
        if directory == canonical and not seal_root:
            continue
        metadata = directory.lstat()
        os.chmod(directory, stat.S_IMODE(metadata.st_mode) & ~0o222)
    _fsync_directory(canonical.parent)


def _minimal_probe_environment(scratch: Path) -> dict[str, str]:
    paths = {
        "HOME": scratch / "home",
        "XDG_CACHE_HOME": scratch / "xdg-cache",
        "XDG_CONFIG_HOME": scratch / "xdg-config",
        "XDG_DATA_HOME": scratch / "xdg-data",
        "TMPDIR": scratch / "tmp",
        "CONDA_PKGS_DIRS": scratch / "pkgs",
        "CONDA_ENVS_PATH": scratch / "envs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CONDARC": "/dev/null",
        "CONDA_OFFLINE": "true",
        "CONDA_NO_PLUGINS": "true",
        **{key: str(value) for key, value in paths.items()},
    }


def _run_probe(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path,
    description: str,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CondaToolchainProvisionError(f"{description} failed: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise CondaToolchainProvisionError(
            f"{description} failed rc={completed.returncode}: {detail[:2000]}"
        )
    return completed


def _probe_read_only_operation(
    prefix: Path,
    *,
    expected_conda_version: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    conda = prefix / "bin" / "conda"
    summaries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="schema5-conda-readonly-probe.") as raw:
        probe_root = Path(raw)
        if _is_relative_to(probe_root, prefix) or _is_relative_to(prefix, probe_root):
            raise CondaToolchainProvisionError(
                "read-only probe scratch overlaps the sealed toolchain"
            )
        for index in range(2):
            scratch = probe_root / f"pass-{index + 1}"
            scratch.mkdir()
            environment = _minimal_probe_environment(scratch)
            version = _run_probe(
                [str(conda), "--version"],
                environment=environment,
                cwd=scratch,
                description="sealed Conda version probe",
                timeout_seconds=timeout_seconds,
            )
            expected_stdout = f"conda {expected_conda_version}"
            if version.stdout.strip() != expected_stdout or version.stderr.strip():
                raise CondaToolchainProvisionError(
                    "sealed Conda version output is unexpected"
                )
            info = _run_probe(
                [str(conda), "info", "--offline", "--json"],
                environment=environment,
                cwd=scratch,
                description="sealed offline Conda info probe",
                timeout_seconds=timeout_seconds,
            )
            if info.stderr.strip():
                raise CondaToolchainProvisionError(
                    "sealed offline Conda probe emitted stderr"
                )
            try:
                payload = json.loads(info.stdout)
            except (json.JSONDecodeError, ValueError) as exc:
                raise CondaToolchainProvisionError(
                    "sealed offline Conda probe did not return JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise CondaToolchainProvisionError(
                    "sealed offline Conda probe returned malformed JSON"
                )
            allowed_config_roots = (prefix, scratch)
            config_files = payload.get("config_files")
            if not isinstance(config_files, list) or any(
                not isinstance(item, str)
                or (
                    _safe_lexical(item, description="Conda config file")
                    != Path("/dev/null")
                    and not any(
                        _is_relative_to(
                            _safe_lexical(
                                item, description="Conda config file"
                            ),
                            allowed,
                        )
                        for allowed in allowed_config_roots
                    )
                )
                for item in config_files
            ):
                raise CondaToolchainProvisionError(
                    "sealed Conda probe consulted configuration outside its "
                    "prefix or isolated scratch"
                )
            conda_location = _safe_lexical(
                str(payload.get("conda_location", "")),
                description="Conda module location",
            )
            if (
                payload.get("conda_version") != expected_conda_version
                or payload.get("root_prefix") != str(prefix)
                or payload.get("conda_prefix") != str(prefix)
                or payload.get("root_writable") is not False
                or payload.get("offline") is not True
                or not _is_relative_to(conda_location, prefix)
            ):
                raise CondaToolchainProvisionError(
                    "sealed offline Conda provenance or read-only state is invalid"
                )
            summary = {
                "conda_version": payload["conda_version"],
                "python_version": payload.get("python_version"),
                "root_prefix": payload["root_prefix"],
                "conda_prefix": payload["conda_prefix"],
                "conda_location": str(conda_location),
                "root_writable": payload["root_writable"],
                "offline": payload["offline"],
                "external_config_file_count": 0,
                "version_stdout": expected_stdout,
            }
            summaries.append(summary)
    if summaries[0] != summaries[1]:
        raise CondaToolchainProvisionError(
            "sealed offline Conda probe changed across repeated execution"
        )
    return {
        "probe_count": 2,
        "network_disabled": True,
        "isolated_writable_scratch": True,
        "target_prefix_writable": False,
        "environment_allowlist": [
            "CONDA_ENVS_PATH",
            "CONDA_NO_PLUGINS",
            "CONDA_OFFLINE",
            "CONDA_PKGS_DIRS",
            "CONDARC",
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "PYTHONDONTWRITEBYTECODE",
            "TMPDIR",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        ],
        "summary": summaries[0],
    }


def _installer_environment(scratch: Path) -> dict[str, str]:
    environment = _minimal_probe_environment(scratch)
    environment["MINIFORGE_FORCE"] = "1"
    return environment


def _run_installer(
    installer: Path,
    *,
    prefix: Path,
    attempt_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    stdout_path = attempt_root / "installer.stdout"
    stderr_path = attempt_root / "installer.stderr"
    scratch = attempt_root / "scratch"
    scratch.mkdir()
    command = ["/bin/bash", str(installer), "-b", "-p", str(prefix)]
    bash_identity = _stable_file_identity(
        Path("/bin/bash").resolve(strict=True), description="installer Bash runtime"
    )
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = subprocess.run(
                command,
                cwd=attempt_root,
                env=_installer_environment(scratch),
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout_seconds,
            )
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    except subprocess.TimeoutExpired as exc:
        raise CondaToolchainProvisionError(
            f"pinned Miniforge installer exceeded {timeout_seconds} seconds"
        ) from exc
    except OSError as exc:
        raise CondaToolchainProvisionError(
            f"cannot execute pinned Miniforge installer: {exc}"
        ) from exc
    for path in (stdout_path, stderr_path):
        os.chmod(path, 0o400)
    if completed.returncode != 0:
        _, _, stderr = _stable_regular(
            stderr_path,
            description="failed installer stderr",
            require_read_only=True,
        )
        _, _, stdout = _stable_regular(
            stdout_path,
            description="failed installer stdout",
            require_read_only=True,
        )
        detail = (stderr or stdout).decode("utf-8", "replace").strip()
        raise CondaToolchainProvisionError(
            f"pinned Miniforge installer failed rc={completed.returncode}: "
            f"{detail[:2000] or 'no output'}"
        )
    return {
        "argv": [
            "/bin/bash",
            f"inputs/{installer.name}",
            "-b",
            "-p",
            "base",
        ],
        "returncode": completed.returncode,
        "offline_environment": True,
        "environment_inherited": False,
        "bash_runtime": bash_identity,
        "stdout": _stable_file_identity(
            stdout_path,
            description="installer stdout",
            require_read_only=True,
            require_single_link=True,
        ),
        "stderr": _stable_file_identity(
            stderr_path,
            description="installer stderr",
            require_read_only=True,
            require_single_link=True,
        ),
    }


def _next_generation(root: Path) -> str:
    existing: list[int] = []
    if root.exists():
        for path in root.iterdir():
            if path.is_dir() and not path.is_symlink() and path.name.startswith("g"):
                suffix = path.name[1:]
                if suffix.isdigit():
                    existing.append(int(suffix))
    return f"g{max(existing, default=0) + 1:04d}"


def _archive_incomplete_prefix(
    toolchain_root: Path, transaction_root: Path
) -> dict[str, Any] | None:
    prefix = toolchain_root / "base"
    evidence = toolchain_root / "evidence"
    candidates = [path for path in (prefix, evidence) if path.exists() or path.is_symlink()]
    if not candidates:
        return None
    quarantine_root = transaction_root / "quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    generation = _next_generation(quarantine_root)
    destination = quarantine_root / generation
    destination.mkdir()
    moved: list[str] = []
    for path in candidates:
        if path.is_symlink():
            raise CondaToolchainProvisionError(
                f"incomplete toolchain component is symlinked: {path}"
            )
        target = destination / path.name
        os.replace(path, target)
        moved.append(path.name)
    receipt = {
        "schema_version": 1,
        "protocol": f"{PROTOCOL}-incomplete-prefix-quarantine",
        "generation": generation,
        "moved": sorted(moved),
        "source_toolchain_root": str(toolchain_root),
    }
    receipt["receipt_id"] = _self_hash(receipt, "receipt_id")
    _publish_once(
        destination / "QUARANTINE_COMPLETE.json",
        receipt,
        description="incomplete-prefix quarantine receipt",
    )
    _seal_recursively(destination)
    _fsync_directory(quarantine_root)
    return {
        "generation": generation,
        "receipt_id": receipt["receipt_id"],
    }


def _intent_payload(
    *,
    namespace_root: Path,
    toolchain_root: Path,
    installer: Path,
    contract: InstallerContract,
    forbidden_prefixes: Sequence[Path],
) -> dict[str, Any]:
    portable_shebang = _portable_shebang_contract(toolchain_root)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": f"{PROTOCOL}-intent",
        "release_tag": RELEASE_TAG,
        "chain_namespace": CHAIN_NAMESPACE,
        "namespace_root": str(namespace_root),
        "toolchain_root": str(toolchain_root),
        "base_prefix": str(toolchain_root / "base"),
        "portable_shebang": portable_shebang,
        "installer_source": str(installer),
        "installer_contract": contract.as_dict(),
        "forbidden_prefixes": [str(path) for path in forbidden_prefixes],
        "mutation_scope": [
            str(toolchain_root),
            str(namespace_root / TRANSACTION_DIRECTORY_NAME),
        ],
        "live_prefix_queries_permitted": False,
        "shared_base_queries_permitted": False,
    }
    payload["intent_id"] = _self_hash(payload, "intent_id")
    return payload


def _portable_shebang_contract(toolchain_root: Path) -> dict[str, Any]:
    """Fail before installation if Constructor cannot emit an absolute shebang."""

    interpreter = toolchain_root / "base" / "bin" / "python"
    shebang = f"#!{interpreter}\n".encode("utf-8")
    if len(shebang) > MAX_PORTABLE_SHEBANG_BYTES:
        raise CondaToolchainProvisionError(
            "isolated Conda interpreter path exceeds the portable shebang limit: "
            f"{len(shebang)} > {MAX_PORTABLE_SHEBANG_BYTES}: {interpreter}"
        )
    return {
        "interpreter": str(interpreter),
        "shebang_bytes": len(shebang),
        "maximum_shebang_bytes": MAX_PORTABLE_SHEBANG_BYTES,
        "absolute_base_prefix_interpreter_required": True,
    }


def _validate_output_scope(
    namespace_root: Path,
    *,
    forbidden_prefixes: Sequence[Path],
    allow_existing_toolchain: bool,
) -> tuple[Path, tuple[Path, ...]]:
    namespace = _existing_canonical(
        namespace_root, description="release namespace root", kind="directory"
    )
    forbidden: list[Path] = []
    for supplied in forbidden_prefixes:
        candidate = _safe_lexical(supplied, description="forbidden prefix")
        if candidate.exists():
            candidate = _existing_canonical(
                candidate, description="forbidden prefix", kind="directory"
            )
        forbidden.append(candidate)
    for blocked in forbidden:
        if _is_relative_to(namespace, blocked) or _is_relative_to(blocked, namespace):
            raise CondaToolchainProvisionError(
                "release namespace overlaps a live or shared forbidden prefix"
            )
    toolchain = _new_child_path(
        namespace,
        TOOLCHAIN_DIRECTORY_NAME,
        description="isolated Conda toolchain root",
        allow_existing=allow_existing_toolchain,
    )
    return toolchain, tuple(forbidden)


def _validate_marker_shape(
    marker: Mapping[str, Any],
    *,
    toolchain_root: Path,
    contract: InstallerContract,
) -> None:
    expected_keys = {
        "schema_version",
        "protocol",
        "release_tag",
        "chain_namespace",
        "toolchain_root",
        "intent",
        "installer",
        "installation",
        "base_prefix",
        "portable_shebang",
        "conda_executable",
        "runtime_identity",
        "complete_prefix_inventory",
        "symlink_audit",
        "read_only_operation",
        "sealed_read_only",
        "marker_id",
    }
    if set(marker) != expected_keys:
        raise CondaToolchainProvisionError(
            "Conda toolchain completion marker fields drifted"
        )
    installer = marker.get("installer")
    installation = marker.get("installation")
    runtime = marker.get("runtime_identity")
    inventory = marker.get("complete_prefix_inventory")
    if (
        marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("protocol") != PROTOCOL
        or marker.get("release_tag") != RELEASE_TAG
        or marker.get("chain_namespace") != CHAIN_NAMESPACE
        or marker.get("toolchain_root") != str(toolchain_root)
        or marker.get("base_prefix") != str(toolchain_root / "base")
        or marker.get("portable_shebang")
        != _portable_shebang_contract(toolchain_root)
        or marker.get("conda_executable") != str(toolchain_root / "base/bin/conda")
        or marker.get("sealed_read_only") is not True
        or not isinstance(installer, dict)
        or installer.get("contract") != contract.as_dict()
        or installer.get("copied_path")
        != f"inputs/{contract.filename}"
        or installer.get("buffered_copy") is not True
        or installer.get("shared_inode") is not False
        or not isinstance(installation, dict)
        or set(installation)
        != {
            "argv",
            "returncode",
            "offline_environment",
            "environment_inherited",
            "bash_runtime",
            "stdout",
            "stderr",
            "generation",
            "quarantined_incomplete_predecessor",
        }
        or installation.get("argv")
        != [
            "/bin/bash",
            f"inputs/{contract.filename}",
            "-b",
            "-p",
            "base",
        ]
        or installation.get("returncode") != 0
        or installation.get("offline_environment") is not True
        or installation.get("environment_inherited") is not False
        or not isinstance(installation.get("bash_runtime"), dict)
        or not isinstance(installation.get("stdout"), dict)
        or not isinstance(installation.get("stderr"), dict)
        or not isinstance(installation.get("generation"), str)
        or not isinstance(runtime, dict)
        or runtime.get("validated_twice") is not True
        or runtime.get("first") != runtime.get("second")
        or not isinstance(inventory, dict)
        or inventory.get("validated_before_and_after_probes") is not True
        or inventory.get("before") != inventory.get("after")
        or marker.get("marker_id") != _self_hash(marker, "marker_id")
    ):
        raise CondaToolchainProvisionError(
            "Conda toolchain completion marker contract is invalid"
        )


def _recorded_relative_file_identity(
    path: Path, *, root: Path, description: str
) -> dict[str, Any]:
    identity = _stable_file_identity(
        path,
        description=description,
        require_read_only=True,
        require_single_link=True,
    )
    result = {
        key: value
        for key, value in identity.items()
        if key not in {"device", "inode"}
    }
    result["path"] = path.relative_to(root).as_posix()
    return result


def verify_conda_toolchain(
    toolchain_root: str | Path,
    *,
    contract: InstallerContract = PINNED_INSTALLER_CONTRACT,
    exercise: bool = True,
    probe_timeout_seconds: int = 240,
) -> dict[str, Any]:
    """Verify marker, pinned installer copy, full runtime, links, and read-only use."""

    _validate_contract(contract)
    root = _existing_canonical(
        toolchain_root, description="sealed Conda toolchain root", kind="directory"
    )
    marker, _ = _read_canonical_json(
        root / MARKER_NAME, description="Conda toolchain completion marker"
    )
    _validate_marker_shape(marker, toolchain_root=root, contract=contract)
    expected_root_entries = {
        INTENT_NAME,
        MARKER_NAME,
        "base",
        "evidence",
        "inputs",
    }
    if {path.name for path in root.iterdir()} != expected_root_entries:
        raise CondaToolchainProvisionError(
            "sealed Conda toolchain has unexpected or missing root entries"
        )
    inputs = _existing_canonical(
        root / "inputs", description="sealed toolchain inputs", kind="directory"
    )
    evidence = _existing_canonical(
        root / "evidence", description="sealed toolchain evidence", kind="directory"
    )
    if {path.name for path in inputs.iterdir()} != {contract.filename}:
        raise CondaToolchainProvisionError(
            "sealed toolchain installer-input set drifted"
        )
    if {path.name for path in evidence.iterdir()} != {
        "installer.stderr",
        "installer.stdout",
    }:
        raise CondaToolchainProvisionError(
            "sealed toolchain installer-evidence set drifted"
        )
    intent, intent_raw = _read_canonical_json(
        root / INTENT_NAME, description="Conda toolchain provision intent"
    )
    if (
        intent.get("protocol") != f"{PROTOCOL}-intent"
        or intent.get("release_tag") != RELEASE_TAG
        or intent.get("chain_namespace") != CHAIN_NAMESPACE
        or intent.get("toolchain_root") != str(root)
        or intent.get("namespace_root") != str(root.parent)
        or intent.get("base_prefix") != str(root / "base")
        or intent.get("installer_contract") != contract.as_dict()
        or intent.get("live_prefix_queries_permitted") is not False
        or intent.get("shared_base_queries_permitted") is not False
        or not set(DEFAULT_FORBIDDEN_PREFIXES).issubset(
            {Path(path) for path in intent.get("forbidden_prefixes", [])}
            if isinstance(intent.get("forbidden_prefixes"), list)
            else set()
        )
        or intent.get("mutation_scope")
        != [
            str(root),
            str(root.parent / TRANSACTION_DIRECTORY_NAME),
        ]
        or intent.get("intent_id") != _self_hash(intent, "intent_id")
        or marker["intent"]
        != {
            "path": INTENT_NAME,
            "sha256": _sha256_bytes(intent_raw),
            "intent_id": intent.get("intent_id"),
        }
    ):
        raise CondaToolchainProvisionError(
            "Conda toolchain intent binding is invalid"
        )
    if _recorded_relative_file_identity(
        evidence / "installer.stdout",
        root=root,
        description="sealed installer stdout",
    ) != marker["installation"]["stdout"] or _recorded_relative_file_identity(
        evidence / "installer.stderr",
        root=root,
        description="sealed installer stderr",
    ) != marker["installation"]["stderr"]:
        raise CondaToolchainProvisionError(
            "sealed installer output evidence drifted"
        )
    copied_installer = _stable_file_identity(
        root / "inputs" / contract.filename,
        description="sealed copied Miniforge installer",
        require_read_only=True,
        require_single_link=True,
    )
    if (
        copied_installer["sha256"] != contract.sha256
        or copied_installer["size"] != marker["installer"].get("size")
        or marker["installer"].get("sha256") != contract.sha256
    ):
        raise CondaToolchainProvisionError(
            "sealed copied Miniforge installer identity drifted"
        )
    base = root / "base"
    _require_recursively_read_only(base, description="sealed Conda base prefix")
    symlinks = _runtime_symlink_audit(base)
    if symlinks != marker["symlink_audit"]:
        raise CondaToolchainProvisionError("sealed Conda symlink audit drifted")
    try:
        runtime_first = runtime_identity.conda_runtime_identity(
            base / "bin" / "conda"
        )
    except (
        OSError,
        runtime_identity.CondaRuntimeIdentityError,
    ) as exc:
        raise CondaToolchainProvisionError(
            f"sealed Conda runtime identity is invalid: {exc}"
        ) from exc
    complete_before = _complete_prefix_inventory(base)
    if exercise:
        probes = _probe_read_only_operation(
            base,
            expected_conda_version=contract.conda_version,
            timeout_seconds=probe_timeout_seconds,
        )
    else:
        probes = marker["read_only_operation"]
    try:
        runtime_second = runtime_identity.conda_runtime_identity(
            base / "bin" / "conda"
        )
    except (
        OSError,
        runtime_identity.CondaRuntimeIdentityError,
    ) as exc:
        raise CondaToolchainProvisionError(
            f"sealed Conda runtime replay identity is invalid: {exc}"
        ) from exc
    complete_after = _complete_prefix_inventory(base)
    if (
        runtime_first != runtime_second
        or runtime_first != marker["runtime_identity"]["first"]
        or complete_before != complete_after
        or complete_before != marker["complete_prefix_inventory"]["before"]
        or probes != marker["read_only_operation"]
    ):
        raise CondaToolchainProvisionError(
            "sealed Conda runtime or read-only operation drifted"
        )
    _require_recursively_read_only(root, description="sealed Conda toolchain")
    return dict(marker)


def verified_conda_executable(
    toolchain_root: str | Path,
    *,
    contract: InstallerContract = PINNED_INSTALLER_CONTRACT,
    exercise: bool = True,
) -> Path:
    """Return the exact executable only after complete toolchain verification."""

    binding = verified_conda_toolchain_binding(
        toolchain_root,
        contract=contract,
        exercise=exercise,
    )
    return _existing_canonical(
        binding["conda_executable"]["path"],
        description="verified isolated Conda executable",
        kind="file",
    )


def verified_conda_toolchain_binding(
    toolchain_root: str | Path,
    *,
    contract: InstallerContract = PINNED_INSTALLER_CONTRACT,
    exercise: bool = True,
) -> dict[str, Any]:
    """Return the canonical provenance record for one fully verified toolchain.

    This is the sole public binding consumed by production release code.  It is
    deliberately derived from a complete verification rather than trusting a
    caller-supplied executable or a previously serialized subset of the marker.
    The returned record is compact enough to embed verbatim in every downstream
    identity while binding the raw marker, exact executable bytes, complete-prefix
    inventory, runtime identity, and repeatable read-only probes.
    """

    lexical_root = _safe_lexical(
        toolchain_root, description="sealed Conda toolchain root"
    )
    for forbidden in DEFAULT_FORBIDDEN_PREFIXES:
        blocked = _safe_lexical(
            forbidden, description="forbidden live/shared Conda prefix"
        )
        if (
            lexical_root == blocked
            or _is_relative_to(lexical_root, blocked)
            or _is_relative_to(blocked, lexical_root)
        ):
            raise CondaToolchainProvisionError(
                "sealed Conda toolchain overlaps a known live or shared prefix"
            )
    marker = verify_conda_toolchain(
        toolchain_root,
        contract=contract,
        exercise=exercise,
    )
    root = _existing_canonical(
        toolchain_root,
        description="verified sealed Conda toolchain root",
        kind="directory",
    )
    marker_path = root / MARKER_NAME
    reread_marker, marker_raw = _read_canonical_json(
        marker_path,
        description="verified Conda toolchain completion marker",
    )
    if reread_marker != marker:
        raise CondaToolchainProvisionError(
            "Conda toolchain marker changed after complete verification"
        )
    executable_identity = _stable_file_identity(
        marker["conda_executable"],
        description="verified isolated Conda executable",
        require_read_only=True,
    )
    executable = {
        key: executable_identity[key]
        for key in ("path", "sha256", "size", "mode", "link_count")
    }
    runtime = marker.get("runtime_identity", {}).get("first")
    inventory = marker.get("complete_prefix_inventory", {}).get("before")
    probes = marker.get("read_only_operation")
    if (
        not isinstance(runtime, dict)
        or not isinstance(inventory, dict)
        or not isinstance(probes, dict)
        or inventory.get("inventory_sha256") is None
    ):
        raise CondaToolchainProvisionError(
            "verified Conda toolchain lacks canonical identity material"
        )
    binding: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "release_tag": RELEASE_TAG,
        "chain_namespace": CHAIN_NAMESPACE,
        "toolchain_root": str(root),
        "base_prefix": str(root / "base"),
        "portable_shebang": dict(marker["portable_shebang"]),
        "completion_marker": {
            "path": str(marker_path),
            "sha256": _sha256_bytes(marker_raw),
            "size": len(marker_raw),
        },
        "marker_id": marker["marker_id"],
        "installer_contract": contract.as_dict(),
        "intent_id": marker["intent"]["intent_id"],
        "conda_executable": executable,
        "runtime_identity_sha256": _sha256_bytes(_canonical_bytes(runtime)),
        "complete_prefix_inventory_sha256": inventory["inventory_sha256"],
        "read_only_probes": probes,
    }
    binding["binding_id"] = _self_hash(binding, "binding_id")
    return binding


def provision_conda_toolchain(
    *,
    installer: str | Path,
    namespace_root: str | Path,
    forbidden_prefixes: Sequence[str | Path] = (),
    contract: InstallerContract = PINNED_INSTALLER_CONTRACT,
    apply: bool = False,
    installer_timeout_seconds: int = 7200,
    probe_timeout_seconds: int = 240,
) -> dict[str, Any]:
    """Audit or create the isolated toolchain without consulting any Conda base."""

    _validate_contract(contract)
    source = _existing_canonical(
        installer, description="cached pinned Miniforge installer", kind="file"
    )
    source_identity = _stable_file_identity(
        source, description="cached pinned Miniforge installer"
    )
    if (
        source.name != contract.filename
        or source_identity["sha256"] != contract.sha256
    ):
        raise CondaToolchainProvisionError(
            "cached Miniforge installer does not match the immutable contract"
        )
    namespace = _existing_canonical(
        namespace_root, description="release namespace root", kind="directory"
    )
    effective_forbidden: list[Path] = []
    for candidate in (*DEFAULT_FORBIDDEN_PREFIXES, *forbidden_prefixes):
        lexical = _safe_lexical(candidate, description="forbidden prefix")
        if lexical not in effective_forbidden:
            effective_forbidden.append(lexical)
    toolchain, blocked = _validate_output_scope(
        namespace,
        forbidden_prefixes=effective_forbidden,
        allow_existing_toolchain=True,
    )
    if _is_relative_to(source, namespace):
        raise CondaToolchainProvisionError(
            "cached installer must be outside the release namespace"
        )
    expected_intent = _intent_payload(
        namespace_root=namespace,
        toolchain_root=toolchain,
        installer=source,
        contract=contract,
        forbidden_prefixes=blocked,
    )
    marker_path = toolchain / MARKER_NAME
    if marker_path.exists() or marker_path.is_symlink():
        if apply and stat.S_IMODE(toolchain.stat().st_mode) & 0o222:
            marker, _ = _read_canonical_json(
                marker_path,
                description="Conda toolchain completion marker",
            )
            _validate_marker_shape(
                marker, toolchain_root=toolchain, contract=contract
            )
            os.chmod(
                toolchain,
                stat.S_IMODE(toolchain.stat().st_mode) & ~0o222,
            )
            _fsync_directory(namespace)
        marker = verify_conda_toolchain(
            toolchain,
            contract=contract,
            exercise=True,
            probe_timeout_seconds=probe_timeout_seconds,
        )
        intent, _ = _read_canonical_json(
            toolchain / INTENT_NAME,
            description="completed Conda toolchain provision intent",
        )
        if intent != expected_intent:
            raise CondaToolchainProvisionError(
                "completed Conda toolchain was provisioned from different inputs"
            )
        source_after = _stable_file_identity(
            source, description="cached pinned Miniforge installer"
        )
        if source_identity != source_after:
            raise CondaToolchainProvisionError(
                "cached Miniforge installer changed during completed replay"
            )
        return {"action": "already_complete", **marker}
    if not apply:
        if toolchain.exists():
            intent, _ = _read_canonical_json(
                toolchain / INTENT_NAME,
                description="incomplete Conda toolchain provision intent",
            )
            if intent != expected_intent:
                raise CondaToolchainProvisionError(
                    "incomplete Conda toolchain intent conflicts"
                )
            action = "would_resume"
        else:
            action = "would_provision"
        return {
            "action": action,
            "protocol": PROTOCOL,
            "toolchain_root": str(toolchain),
            "base_prefix": str(toolchain / "base"),
            "portable_shebang": _portable_shebang_contract(toolchain),
            "installer": contract.as_dict(),
            "source_installer": source_identity,
            "would_invoke_existing_conda": False,
            "would_query_live_prefixes": False,
            "would_publish_marker_last": True,
        }

    transaction_root = namespace / TRANSACTION_DIRECTORY_NAME
    transaction_root.mkdir(mode=0o700, exist_ok=True)
    lock_path = transaction_root / "provision.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if marker_path.exists() or marker_path.is_symlink():
            return {
                "action": "already_complete",
                **verify_conda_toolchain(
                    toolchain,
                    contract=contract,
                    exercise=True,
                    probe_timeout_seconds=probe_timeout_seconds,
                ),
            }
        if not toolchain.exists():
            toolchain.mkdir(mode=0o755)
            _fsync_directory(namespace)
        else:
            _existing_canonical(
                toolchain,
                description="incomplete Conda toolchain root",
                kind="directory",
            )
            os.chmod(toolchain, 0o755)
        intent_path = toolchain / INTENT_NAME
        if intent_path.exists() or intent_path.is_symlink():
            intent, _ = _read_canonical_json(
                intent_path,
                description="Conda toolchain provision intent",
            )
            if intent != expected_intent:
                raise CondaToolchainProvisionError(
                    "Conda toolchain provision intent conflicts"
                )
        else:
            _publish_once(
                intent_path,
                expected_intent,
                description="Conda toolchain provision intent",
            )

        quarantine = _archive_incomplete_prefix(toolchain, transaction_root)
        attempts_root = transaction_root / "attempts"
        attempts_root.mkdir(mode=0o700, exist_ok=True)
        generation = _next_generation(attempts_root)
        attempt_root = attempts_root / generation
        attempt_root.mkdir(mode=0o700)
        inputs = toolchain / "inputs"
        inputs.mkdir(mode=0o755, exist_ok=True)
        copied_installer = inputs / contract.filename
        copy_evidence = _buffered_copy(
            source,
            copied_installer,
            description="cached pinned Miniforge installer",
            staging_directory=attempt_root,
        )
        if copy_evidence["copy"]["sha256"] != contract.sha256:
            raise CondaToolchainProvisionError(
                "copied Miniforge installer does not match its immutable digest"
            )
        attempt_intent: dict[str, Any] = {
            "schema_version": 1,
            "protocol": f"{PROTOCOL}-installer-attempt",
            "generation": generation,
            "toolchain_root": str(toolchain),
            "base_prefix": str(toolchain / "base"),
            "installer_sha256": contract.sha256,
            "intent_id": expected_intent["intent_id"],
        }
        attempt_intent["attempt_id"] = _self_hash(
            attempt_intent, "attempt_id"
        )
        _publish_once(
            attempt_root / "ATTEMPT_INTENT.json",
            attempt_intent,
            description="Conda installer attempt intent",
        )
        installation = _run_installer(
            copied_installer,
            prefix=toolchain / "base",
            attempt_root=attempt_root,
            timeout_seconds=installer_timeout_seconds,
        )
        source_after_install = _stable_file_identity(
            source, description="cached pinned Miniforge installer"
        )
        if source_after_install != source_identity:
            raise CondaToolchainProvisionError(
                "cached Miniforge installer changed while provisioning"
            )

        base = _existing_canonical(
            toolchain / "base",
            description="freshly installed Conda base",
            kind="directory",
        )
        try:
            _runtime_symlink_audit(base)
            runtime_identity.conda_runtime_identity(base / "bin" / "conda")
            _complete_prefix_inventory(base)
        except (
            OSError,
            runtime_identity.CondaRuntimeIdentityError,
        ) as exc:
            raise CondaToolchainProvisionError(
                f"freshly installed Conda base failed pre-seal validation: {exc}"
            ) from exc
        _seal_recursively(base)
        symlink_audit = _runtime_symlink_audit(base)
        try:
            runtime_first = runtime_identity.conda_runtime_identity(
                base / "bin" / "conda"
            )
        except (
            OSError,
            runtime_identity.CondaRuntimeIdentityError,
        ) as exc:
            raise CondaToolchainProvisionError(
                f"sealed Conda runtime identity failed: {exc}"
            ) from exc
        complete_before = _complete_prefix_inventory(base)
        read_only_operation = _probe_read_only_operation(
            base,
            expected_conda_version=contract.conda_version,
            timeout_seconds=probe_timeout_seconds,
        )
        try:
            runtime_second = runtime_identity.conda_runtime_identity(
                base / "bin" / "conda"
            )
        except (
            OSError,
            runtime_identity.CondaRuntimeIdentityError,
        ) as exc:
            raise CondaToolchainProvisionError(
                f"sealed Conda runtime replay identity failed: {exc}"
            ) from exc
        complete_after = _complete_prefix_inventory(base)
        if runtime_first != runtime_second or complete_before != complete_after:
            raise CondaToolchainProvisionError(
                "sealed Conda runtime changed during read-only operation"
            )

        evidence = toolchain / "evidence"
        evidence.mkdir(mode=0o755)
        for source_log in (
            attempt_root / "installer.stdout",
            attempt_root / "installer.stderr",
        ):
            _buffered_copy(
                source_log,
                evidence / source_log.name,
                description=source_log.name,
                staging_directory=attempt_root,
            )
        installation = {
            **installation,
            "generation": generation,
            "quarantined_incomplete_predecessor": quarantine,
            "stdout": {
                key: value
                for key, value in _stable_file_identity(
                    evidence / "installer.stdout",
                    description="sealed installer stdout",
                    require_read_only=True,
                    require_single_link=True,
                ).items()
                if key not in {"device", "inode"}
            },
            "stderr": {
                key: value
                for key, value in _stable_file_identity(
                    evidence / "installer.stderr",
                    description="sealed installer stderr",
                    require_read_only=True,
                    require_single_link=True,
                ).items()
                if key not in {"device", "inode"}
            },
        }
        installation["stdout"]["path"] = "evidence/installer.stdout"
        installation["stderr"]["path"] = "evidence/installer.stderr"
        _seal_recursively(inputs)
        _seal_recursively(evidence)
        _require_recursively_read_only(base, description="sealed Conda base prefix")

        _, _, intent_raw = _stable_regular(
            intent_path,
            description="Conda toolchain provision intent",
            require_read_only=True,
        )
        marker: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "release_tag": RELEASE_TAG,
            "chain_namespace": CHAIN_NAMESPACE,
            "toolchain_root": str(toolchain),
            "intent": {
                "path": INTENT_NAME,
                "sha256": _sha256_bytes(intent_raw),
                "intent_id": expected_intent["intent_id"],
            },
            "installer": {
                "contract": contract.as_dict(),
                "copied_path": f"inputs/{contract.filename}",
                "sha256": copy_evidence["copy"]["sha256"],
                "size": copy_evidence["copy"]["size"],
                "buffered_copy": True,
                "shared_inode": False,
                "source_reverified_after_install": True,
            },
            "installation": installation,
            "base_prefix": str(base),
            "portable_shebang": _portable_shebang_contract(toolchain),
            "conda_executable": str(base / "bin" / "conda"),
            "runtime_identity": {
                "validated_twice": True,
                "first": runtime_first,
                "second": runtime_second,
            },
            "complete_prefix_inventory": {
                "validated_before_and_after_probes": True,
                "before": complete_before,
                "after": complete_after,
            },
            "symlink_audit": symlink_audit,
            "read_only_operation": read_only_operation,
            "sealed_read_only": True,
        }
        marker["marker_id"] = _self_hash(marker, "marker_id")
        _publish_once(
            marker_path,
            marker,
            description="Conda toolchain completion marker",
        )
        os.chmod(toolchain, 0o555)
        _fsync_directory(namespace)
        verified = verify_conda_toolchain(
            toolchain,
            contract=contract,
            exercise=True,
            probe_timeout_seconds=probe_timeout_seconds,
        )
        return {"action": "provisioned", **verified}
    finally:
        os.close(lock_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    provision = subparsers.add_parser(
        "provision", help="audit or provision the exact pinned installer"
    )
    provision.add_argument("--installer", type=Path, default=DEFAULT_INSTALLER)
    provision.add_argument("--namespace-root", type=Path, required=True)
    provision.add_argument(
        "--forbidden-prefix",
        type=Path,
        action="append",
        default=[],
        help=(
            "additional path that must not overlap the release-local output; "
            "the two live prefixes and shared base are always forbidden"
        ),
    )
    provision.add_argument("--installer-timeout-seconds", type=int, default=7200)
    provision.add_argument("--probe-timeout-seconds", type=int, default=240)
    provision.add_argument(
        "--apply",
        action="store_true",
        help="perform the isolated install; default is a read-only preflight",
    )
    verify = subparsers.add_parser(
        "verify", help="independently verify the sealed release-local toolchain"
    )
    verify.add_argument("--toolchain-root", type=Path, required=True)
    verify.add_argument("--probe-timeout-seconds", type=int, default=240)
    verify.add_argument(
        "--no-exercise",
        action="store_true",
        help="skip command execution while still re-inventorying all sealed bytes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "provision":
            if (
                args.installer_timeout_seconds <= 0
                or args.probe_timeout_seconds <= 0
            ):
                raise CondaToolchainProvisionError("timeouts must be positive")
            result = provision_conda_toolchain(
                installer=args.installer,
                namespace_root=args.namespace_root,
                forbidden_prefixes=(
                    *args.forbidden_prefix,
                ),
                apply=args.apply,
                installer_timeout_seconds=args.installer_timeout_seconds,
                probe_timeout_seconds=args.probe_timeout_seconds,
            )
        else:
            if args.probe_timeout_seconds <= 0:
                raise CondaToolchainProvisionError("timeout must be positive")
            result = verify_conda_toolchain(
                args.toolchain_root,
                exercise=not args.no_exercise,
                probe_timeout_seconds=args.probe_timeout_seconds,
            )
    except (
        CondaToolchainProvisionError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
