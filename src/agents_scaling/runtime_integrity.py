"""Runtime integrity for frozen schema-5 Conda environments.

The release freezer publishes a complete, content-addressed directory inventory for
each production prefix.  This module is the runtime consumer of that inventory.  A
new rollout generation performs one full byte verification and publishes a
generation-scoped attestation under a cross-node advisory lock.  One controller then
renews a short-lived lease from a metadata-only tree scan at most once per refresh
interval.  Other controllers, cells, and servers validate that compact lease in O(1)
time.  The metadata fingerprint includes every path, type, mode, inode, size, mtime,
ctime, and symlink target, so an owner chmod/write/replace operation invalidates the
cache even when the prefix root remains mode 0555.

Workers check the lease before every stochastic coordinate without rescanning the
tree.  A mismatch or expired lease is deliberately not repaired in place: production
fails closed until a trusted controller renews the lease or an operator publishes a
new immutable release or generation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
from contextlib import contextmanager
import math
from typing import Any, Iterator, Mapping, Sequence


ATTESTATION_SCHEMA_VERSION = 1
INVENTORY_ALGORITHM = "schema5-directory-inventory-v1"
METADATA_ALGORITHM = "schema5-lstat-path-type-mode-inode-size-mtime-ctime-link-v1"
LEASE_SCHEMA_VERSION = 1
LEASE_REFRESH_SECONDS = 300.0
LEASE_TTL_SECONDS = 420.0
LEASE_CLOCK_SKEW_SECONDS = 30.0
_CHUNK_SIZE = 8 * 1024 * 1024
_SHA256_LENGTH = 64


class RuntimeIntegrityError(RuntimeError):
    """A frozen runtime tree or its generation attestation cannot be trusted."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeIntegrityError(f"cannot open regular file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeIntegrityError(f"integrity input is not a regular file: {path}")
        while chunk := os.read(descriptor, _CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeIntegrityError(f"file changed while being hashed: {path}")
    return digest.hexdigest()


def _safe_root(root: Path, *, require_read_only: bool = True) -> Path:
    lexical = Path(root).expanduser()
    if lexical.is_symlink() or not lexical.is_dir():
        raise RuntimeIntegrityError(f"runtime prefix is missing or symlinked: {lexical}")
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeIntegrityError(f"cannot resolve runtime prefix {lexical}: {exc}") from exc
    if require_read_only and stat.S_IMODE(resolved.stat().st_mode) & 0o222:
        raise RuntimeIntegrityError(f"runtime prefix root is writable: {resolved}")
    return resolved


def _entry_signature(path: Path, *, relative: str) -> tuple[Any, ...]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeIntegrityError(f"cannot stat runtime entry {path}: {exc}") from exc
    file_type = stat.S_IFMT(info.st_mode)
    if not (
        stat.S_ISDIR(info.st_mode)
        or stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
    ):
        raise RuntimeIntegrityError(f"special file is forbidden in runtime prefix: {path}")
    target: str | None = None
    if stat.S_ISLNK(info.st_mode):
        try:
            target = os.readlink(path)
        except OSError as exc:
            raise RuntimeIntegrityError(f"cannot read runtime symlink {path}: {exc}") from exc
    return (
        relative,
        file_type,
        stat.S_IMODE(info.st_mode),
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        target,
    )


def _tree_signatures(root: Path) -> tuple[tuple[Any, ...], ...]:
    signatures: list[tuple[Any, ...]] = [_entry_signature(root, relative=".")]

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise RuntimeIntegrityError(
                f"cannot enumerate runtime directory {directory}: {exc}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            signature = _entry_signature(path, relative=relative)
            signatures.append(signature)
            if stat.S_ISDIR(signature[1]):
                visit(path)

    visit(root)
    return tuple(signatures)


def _metadata_record(signatures: Sequence[tuple[Any, ...]]) -> dict[str, Any]:
    rows = [
        {
            "path": signature[0],
            "type": signature[1],
            "mode": signature[2],
            "device": signature[3],
            "inode": signature[4],
            "size": signature[5],
            "mtime_ns": signature[6],
            "ctime_ns": signature[7],
            "target": signature[8],
        }
        for signature in signatures
    ]
    return {
        "algorithm": METADATA_ALGORITHM,
        "entry_count_including_root": len(rows),
        "metadata_sha256": sha256_bytes(canonical_bytes(rows)),
    }


def metadata_state(root: Path) -> dict[str, Any]:
    """Return the cheap cache-invalidation state for one entire prefix."""

    resolved = _safe_root(root)
    before = _tree_signatures(resolved)
    after = _tree_signatures(resolved)
    if before != after:
        raise RuntimeIntegrityError(
            f"runtime prefix changed during metadata verification: {resolved}"
        )
    return _metadata_record(after)


def directory_inventory_with_metadata(
    root: Path, *, require_read_only: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the freezer-compatible byte inventory and runtime metadata state."""

    resolved = _safe_root(root, require_read_only=require_read_only)
    before = _tree_signatures(resolved)
    entries: list[dict[str, Any]] = []
    total_file_bytes = 0
    for signature in before[1:]:
        relative, file_type, mode, _dev, _ino, size, _mtime, _ctime, target = signature
        path = resolved / relative
        if stat.S_ISDIR(file_type):
            entries.append({"path": relative, "type": "directory", "mode": mode})
        elif stat.S_ISLNK(file_type):
            entries.append(
                {"path": relative, "type": "symlink", "mode": mode, "target": target}
            )
        elif stat.S_ISREG(file_type):
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "mode": mode,
                    "size": size,
                    "sha256": sha256_file(path),
                }
            )
            total_file_bytes += int(size)
        else:  # pragma: no cover - guarded by _entry_signature
            raise RuntimeIntegrityError(f"special runtime entry changed type: {path}")
    after = _tree_signatures(resolved)
    if before != after:
        raise RuntimeIntegrityError(
            f"runtime prefix changed while being inventoried: {resolved}"
        )
    entries.sort(key=lambda entry: entry["path"])
    inventory = {
        "entries": entries,
        "inventory_sha256": sha256_bytes(canonical_bytes(entries)),
        "entry_count": len(entries),
        "file_count": sum(entry["type"] == "file" for entry in entries),
        "directory_count": sum(entry["type"] == "directory" for entry in entries),
        "symlink_count": sum(entry["type"] == "symlink" for entry in entries),
        "total_file_bytes": total_file_bytes,
    }
    return inventory, _metadata_record(after)


def directory_inventory(root: Path) -> dict[str, Any]:
    """Build the canonical inventory used by the schema-5 release freezer."""

    # Freezing inventories the prefix both before and after the optional permission
    # seal.  Runtime verification calls ``directory_inventory_with_metadata`` directly
    # with the default read-only requirement.
    return directory_inventory_with_metadata(root, require_read_only=False)[0]


def _read_regular_bytes(path: Path, *, description: str) -> bytes:
    """Read one stable, non-symlink regular file through a single descriptor."""

    lexical = Path(path).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise RuntimeIntegrityError(
            f"cannot open regular {description} {lexical}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeIntegrityError(
                f"{description} is not a regular file: {lexical}"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, _CHUNK_SIZE):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (  # noqa: E731 - compact immutable identity helper
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeIntegrityError(f"{description} changed while being read: {lexical}")
    return b"".join(chunks)


def _read_json_object(path: Path, *, description: str) -> tuple[dict[str, Any], bytes]:
    lexical = Path(path).expanduser()
    try:
        raw = _read_regular_bytes(lexical, description=description)
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except RuntimeIntegrityError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeIntegrityError(f"cannot read {description} {lexical}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeIntegrityError(f"{description} must contain one JSON object: {lexical}")
    return value, raw


def verify_environment_manifest_live(
    *,
    role: str,
    prefix: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    release_id: str,
) -> dict[str, Any]:
    """Perform one full byte-for-byte verification against a frozen manifest."""

    if role not in {"harness", "serving"}:
        raise RuntimeIntegrityError(f"invalid runtime role {role!r}")
    _validate_release_id(release_id)
    _validate_sha(expected_manifest_sha256, description=f"{role} manifest digest")
    manifest, raw = _read_json_object(manifest_path, description=f"{role} manifest")
    if sha256_bytes(raw) != expected_manifest_sha256:
        raise RuntimeIntegrityError(f"{role} environment manifest hash drifted")
    resolved = _safe_root(prefix)
    try:
        manifest_prefix = Path(str(manifest.get("prefix", ""))).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeIntegrityError(f"invalid {role} manifest prefix: {exc}") from exc
    expected_inventory = manifest.get("directory_inventory")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("release_id") != release_id
        or manifest.get("role") != role
        or manifest_prefix != resolved
        or manifest.get("sealed_read_only") is not True
        or not isinstance(expected_inventory, dict)
    ):
        raise RuntimeIntegrityError(f"{role} environment manifest identity is invalid")
    inventory, metadata = directory_inventory_with_metadata(resolved)
    if inventory != expected_inventory:
        raise RuntimeIntegrityError(
            f"{role} environment directory inventory drifted: {resolved}"
        )
    locks = manifest.get("locks")
    runtime = manifest.get("runtime")
    if not isinstance(locks, dict) or not isinstance(runtime, dict):
        raise RuntimeIntegrityError(f"{role} environment manifest lacks frozen runtime/locks")
    expected_content = sha256_bytes(
        canonical_bytes(
            {
                "runtime": runtime,
                "locks": locks,
                "release_package": manifest.get("release_package"),
                "inventory_sha256": inventory["inventory_sha256"],
            }
        )
    )
    if manifest.get("environment_content_sha256") != expected_content:
        raise RuntimeIntegrityError(f"{role} environment content identity is invalid")
    return {
        "role": role,
        "prefix": str(resolved),
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "manifest_sha256": expected_manifest_sha256,
        "inventory_sha256": inventory["inventory_sha256"],
        "metadata": metadata,
    }


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    if path.parent.is_symlink():
        raise RuntimeIntegrityError(f"runtime integrity lock directory is symlinked: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        raise RuntimeIntegrityError(f"cannot open runtime integrity lock {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeIntegrityError(f"runtime integrity lock is not regular: {path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            f"pid={os.getpid()} started_ns={time.time_ns()}\n".encode("ascii"),
        )
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _atomic_publish_read_only(path: Path, payload: bytes) -> None:
    if path.parent.is_symlink():
        raise RuntimeIntegrityError(
            f"runtime attestation directory is symlinked: {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".attesting", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path):
            raise RuntimeIntegrityError(f"refusing to replace runtime attestation: {path}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_replace_read_only(path: Path, payload: bytes) -> None:
    """Atomically refresh the small lease file; never mutate it in place."""

    if path.parent.is_symlink():
        raise RuntimeIntegrityError(
            f"runtime lease directory is symlinked: {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".leasing", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path):
            try:
                current = path.lstat()
            except OSError as exc:
                raise RuntimeIntegrityError(
                    f"cannot inspect existing runtime lease {path}: {exc}"
                ) from exc
            if not stat.S_ISREG(current.st_mode):
                raise RuntimeIntegrityError(
                    f"refusing to replace non-regular runtime lease: {path}"
                )
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_sha(value: str, *, description: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise RuntimeIntegrityError(f"{description} is not lowercase SHA-256")


def _validate_generation(generation: int) -> None:
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
    ):
        raise RuntimeIntegrityError("runtime attestation generation must be positive")


def _validate_release_id(release_id: str) -> None:
    if not isinstance(release_id, str) or not release_id.strip():
        raise RuntimeIntegrityError("runtime attestation release ID must be non-empty text")


def _validate_metadata_record(value: Any, *, role: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value)
        != {"algorithm", "entry_count_including_root", "metadata_sha256"}
        or value.get("algorithm") != METADATA_ALGORITHM
        or not isinstance(value.get("entry_count_including_root"), int)
        or isinstance(value.get("entry_count_including_root"), bool)
        or value["entry_count_including_root"] < 1
    ):
        raise RuntimeIntegrityError(
            f"runtime attestation {role} metadata identity is invalid"
        )
    _validate_sha(
        value.get("metadata_sha256"), description=f"{role} metadata digest"
    )


def _attestation_path(state_dir: Path, generation: int) -> Path:
    return state_dir / "runtime_integrity" / f"runtime.g{generation:06d}.json"


def generation_lease_path(state_dir: Path, generation: int) -> Path:
    return state_dir / "runtime_integrity" / f"lease.g{generation:06d}.json"


def _expected_roles(environment_pins: Mapping[str, Mapping[str, Any]]) -> set[str]:
    if not isinstance(environment_pins, Mapping) or set(environment_pins) != {
        "harness",
        "serving",
    }:
        raise RuntimeIntegrityError("runtime attestation requires harness and serving pins")
    for role in ("harness", "serving"):
        pin = environment_pins[role]
        if not isinstance(pin, Mapping) or set(pin) != {
            "prefix",
            "manifest_path",
            "manifest_sha256",
        }:
            raise RuntimeIntegrityError(
                f"runtime attestation {role} pins have the wrong fields"
            )
        _validate_sha(pin["manifest_sha256"], description=f"{role} manifest digest")
        for field in ("prefix", "manifest_path"):
            value = pin[field]
            if not isinstance(value, str) or not value or not Path(value).is_absolute():
                raise RuntimeIntegrityError(
                    f"runtime attestation {role} {field} must be an absolute path"
                )
    return {"harness", "serving"}


def ensure_generation_attestation(
    *,
    state_dir: Path,
    generation: int,
    release_id: str,
    release_bundle_id: str,
    immutable_pins_sha256: str,
    environment_pins: Mapping[str, Mapping[str, Any]],
    force_full: bool = False,
) -> dict[str, Any]:
    """Create once or verify one rollout-scoped environment attestation."""

    _validate_generation(generation)
    _validate_release_id(release_id)
    _validate_sha(release_bundle_id, description="release bundle ID")
    _validate_sha(immutable_pins_sha256, description="immutable pin digest")
    _expected_roles(environment_pins)
    resolved_state = Path(state_dir).expanduser().resolve()
    target = _attestation_path(resolved_state, generation)
    lock = resolved_state / "locks" / "runtime-integrity.lock"
    with _exclusive_lock(lock):
        if target.exists() or target.is_symlink():
            digest = sha256_file(target)
            record = verify_generation_attestation(
                path=target,
                expected_sha256=digest,
                generation=generation,
                release_id=release_id,
                immutable_pins_sha256=immutable_pins_sha256,
                expected_environment_hashes={
                    role: str(environment_pins[role]["manifest_sha256"])
                    for role in sorted(environment_pins)
                },
                expected_prefixes={
                    role: str(environment_pins[role]["prefix"])
                    for role in sorted(environment_pins)
                },
                verify_metadata=True,
            )
            if record.get("release_bundle_id") != release_bundle_id:
                raise RuntimeIntegrityError("runtime attestation release bundle drifted")
            if force_full:
                for role in sorted(environment_pins):
                    pin = environment_pins[role]
                    observed = verify_environment_manifest_live(
                        role=role,
                        prefix=Path(str(pin["prefix"])),
                        manifest_path=Path(str(pin["manifest_path"])),
                        expected_manifest_sha256=str(pin["manifest_sha256"]),
                        release_id=release_id,
                    )
                    if observed["metadata"] != record["environments"][role]["metadata"]:
                        raise RuntimeIntegrityError(
                            f"{role} metadata changed since generation attestation"
                        )
            return {"path": str(target), "sha256": digest, "record": record, "cached": True}

        environments: dict[str, Any] = {}
        for role in sorted(environment_pins):
            pin = environment_pins[role]
            environments[role] = verify_environment_manifest_live(
                role=role,
                prefix=Path(str(pin["prefix"])),
                manifest_path=Path(str(pin["manifest_path"])),
                expected_manifest_sha256=str(pin["manifest_sha256"]),
                release_id=release_id,
            )
        record: dict[str, Any] = {
            "schema_version": ATTESTATION_SCHEMA_VERSION,
            "kind": "schema5-runtime-generation-attestation",
            "generation": generation,
            "release_id": release_id,
            "release_bundle_id": release_bundle_id,
            "immutable_pins_sha256": immutable_pins_sha256,
            "inventory_algorithm": INVENTORY_ALGORITHM,
            "metadata_algorithm": METADATA_ALGORITHM,
            "environments": environments,
        }
        record["attestation_id"] = sha256_bytes(canonical_bytes(record))
        payload = _json_bytes(record)
        _atomic_publish_read_only(target, payload)
        digest = sha256_file(target)
        return {"path": str(target), "sha256": digest, "record": record, "cached": False}


def verify_generation_attestation(
    *,
    path: Path,
    expected_sha256: str,
    generation: int,
    release_id: str,
    immutable_pins_sha256: str,
    expected_environment_hashes: Mapping[str, str],
    expected_prefixes: Mapping[str, str] | None = None,
    verify_metadata: bool = True,
) -> dict[str, Any]:
    """Verify exact attestation bytes and, normally, current tree metadata."""

    _validate_generation(generation)
    _validate_release_id(release_id)
    _validate_sha(expected_sha256, description="runtime attestation digest")
    _validate_sha(immutable_pins_sha256, description="immutable pin digest")
    target = Path(path).expanduser()
    record, raw = _read_json_object(target, description="runtime attestation")
    try:
        target_mode = target.lstat().st_mode
    except OSError as exc:
        raise RuntimeIntegrityError(f"cannot stat runtime attestation {target}: {exc}") from exc
    if stat.S_IMODE(target_mode) & 0o222:
        raise RuntimeIntegrityError(f"runtime attestation remains writable: {target}")
    if sha256_bytes(raw) != expected_sha256:
        raise RuntimeIntegrityError("runtime attestation byte hash drifted")
    required = {
        "schema_version",
        "kind",
        "generation",
        "release_id",
        "release_bundle_id",
        "immutable_pins_sha256",
        "inventory_algorithm",
        "metadata_algorithm",
        "environments",
        "attestation_id",
    }
    if set(record) != required:
        raise RuntimeIntegrityError("runtime attestation has the wrong fields")
    candidate = dict(record)
    attestation_id = candidate.pop("attestation_id", None)
    environments = record.get("environments")
    if (
        record.get("schema_version") != ATTESTATION_SCHEMA_VERSION
        or record.get("kind") != "schema5-runtime-generation-attestation"
        or record.get("generation") != generation
        or record.get("release_id") != release_id
        or record.get("immutable_pins_sha256") != immutable_pins_sha256
        or record.get("inventory_algorithm") != INVENTORY_ALGORITHM
        or record.get("metadata_algorithm") != METADATA_ALGORITHM
        or attestation_id != sha256_bytes(canonical_bytes(candidate))
        or not isinstance(environments, dict)
        or set(environments) != {"harness", "serving"}
        or set(expected_environment_hashes) != {"harness", "serving"}
    ):
        raise RuntimeIntegrityError("runtime attestation identity is invalid")
    _validate_sha(str(record.get("release_bundle_id", "")), description="release bundle ID")
    if expected_prefixes is not None and set(expected_prefixes) != {
        "harness",
        "serving",
    }:
        raise RuntimeIntegrityError(
            "runtime attestation expected prefixes require harness and serving"
        )
    for role in ("harness", "serving"):
        environment = environments.get(role)
        expected_hash = str(expected_environment_hashes[role])
        _validate_sha(expected_hash, description=f"{role} manifest digest")
        if (
            not isinstance(environment, dict)
            or set(environment)
            != {
                "role",
                "prefix",
                "manifest_path",
                "manifest_sha256",
                "inventory_sha256",
                "metadata",
            }
            or environment.get("role") != role
            or environment.get("manifest_sha256") != expected_hash
        ):
            raise RuntimeIntegrityError(f"runtime attestation {role} identity is invalid")
        _validate_sha(
            environment.get("inventory_sha256"),
            description=f"{role} inventory digest",
        )
        _validate_metadata_record(environment.get("metadata"), role=role)
        raw_prefix = environment.get("prefix")
        if not isinstance(raw_prefix, str) or not Path(raw_prefix).is_absolute():
            raise RuntimeIntegrityError(
                f"runtime attestation {role} prefix identity is invalid"
            )
        try:
            prefix = Path(raw_prefix).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeIntegrityError(
                f"runtime attestation {role} prefix cannot be resolved: {exc}"
            ) from exc
        if expected_prefixes is not None:
            expected_value = expected_prefixes[role]
            if not isinstance(expected_value, str) or not Path(expected_value).is_absolute():
                raise RuntimeIntegrityError(
                    f"runtime attestation expected {role} prefix is invalid"
                )
            expected = Path(expected_value).expanduser().resolve()
            if prefix != expected:
                raise RuntimeIntegrityError(f"runtime attestation {role} prefix drifted")
        raw_manifest_path = environment.get("manifest_path")
        if not isinstance(raw_manifest_path, str) or not Path(raw_manifest_path).is_absolute():
            raise RuntimeIntegrityError(
                f"runtime attestation {role} manifest path is invalid"
            )
        manifest_path = Path(raw_manifest_path)
        _manifest, raw = _read_json_object(manifest_path, description=f"{role} manifest")
        if sha256_bytes(raw) != expected_hash:
            raise RuntimeIntegrityError(f"runtime attestation {role} manifest drifted")
        if verify_metadata:
            observed = metadata_state(prefix)
            if observed != environment["metadata"]:
                raise RuntimeIntegrityError(
                    f"{role} runtime metadata drifted after full attestation"
                )
    return record


def verify_generation_lease(
    *,
    lease_path: Path,
    attestation_path: Path,
    attestation_sha256: str,
    generation: int,
    release_id: str,
    immutable_pins_sha256: str,
    expected_environment_hashes: Mapping[str, str],
    expected_prefixes: Mapping[str, str] | None = None,
    now: float | None = None,
    require_fresh: bool = True,
) -> dict[str, Any]:
    """Validate a small, controller-refreshed lease without walking either tree."""

    timestamp = time.time() if now is None else float(now)
    if not math.isfinite(timestamp):
        raise RuntimeIntegrityError("runtime lease verification time is not finite")
    _validate_generation(generation)
    _validate_release_id(release_id)
    _validate_sha(attestation_sha256, description="runtime attestation digest")
    _validate_sha(immutable_pins_sha256, description="immutable pin digest")
    if set(expected_environment_hashes) != {"harness", "serving"}:
        raise RuntimeIntegrityError(
            "runtime lease environment hashes require harness and serving"
        )
    for role, digest in expected_environment_hashes.items():
        _validate_sha(digest, description=f"{role} manifest digest")
    if expected_prefixes is not None and set(expected_prefixes) != {
        "harness",
        "serving",
    }:
        raise RuntimeIntegrityError(
            "runtime lease expected prefixes require harness and serving"
        )
    lexical = Path(lease_path).expanduser()
    lease, _raw = _read_json_object(lexical, description="runtime integrity lease")
    try:
        lease_mode = lexical.lstat().st_mode
    except OSError as exc:
        raise RuntimeIntegrityError(f"cannot stat runtime integrity lease: {exc}") from exc
    if stat.S_IMODE(lease_mode) & 0o222:
        raise RuntimeIntegrityError(f"runtime integrity lease remains writable: {lexical}")
    required = {
        "schema_version",
        "kind",
        "generation",
        "release_id",
        "immutable_pins_sha256",
        "attestation_path",
        "attestation_sha256",
        "attestation_id",
        "metadata_algorithm",
        "environment_metadata_sha256",
        "sequence",
        "verified_timestamp",
        "expires_timestamp",
        "lease_id",
    }
    candidate = dict(lease)
    lease_id = candidate.pop("lease_id", None)
    metadata_hashes = lease.get("environment_metadata_sha256")
    if (
        set(lease) != required
        or lease.get("schema_version") != LEASE_SCHEMA_VERSION
        or lease.get("kind") != "schema5-runtime-integrity-lease"
        or lease.get("generation") != generation
        or lease.get("release_id") != release_id
        or lease.get("immutable_pins_sha256") != immutable_pins_sha256
        or lease.get("attestation_path")
        != str(Path(attestation_path).expanduser().resolve())
        or lease.get("attestation_sha256") != attestation_sha256
        or lease.get("metadata_algorithm") != METADATA_ALGORITHM
        or not isinstance(metadata_hashes, dict)
        or set(metadata_hashes) != {"harness", "serving"}
        or not isinstance(lease.get("sequence"), int)
        or isinstance(lease.get("sequence"), bool)
        or lease["sequence"] < 1
        or not isinstance(lease.get("verified_timestamp"), (int, float))
        or isinstance(lease.get("verified_timestamp"), bool)
        or not isinstance(lease.get("expires_timestamp"), (int, float))
        or isinstance(lease.get("expires_timestamp"), bool)
        or not math.isfinite(float(lease["verified_timestamp"]))
        or not math.isfinite(float(lease["expires_timestamp"]))
        or float(lease["expires_timestamp"]) <= float(lease["verified_timestamp"])
        or lease_id != sha256_bytes(canonical_bytes(candidate))
    ):
        raise RuntimeIntegrityError("runtime integrity lease identity is invalid")
    for role, digest in metadata_hashes.items():
        _validate_sha(digest, description=f"{role} lease metadata digest")
    if timestamp + LEASE_CLOCK_SKEW_SECONDS < float(lease["verified_timestamp"]):
        raise RuntimeIntegrityError(
            "runtime integrity lease was verified in the future"
        )
    if require_fresh and timestamp >= float(lease["expires_timestamp"]):
        raise RuntimeIntegrityError(
            "runtime integrity lease expired before process startup"
        )
    attestation = verify_generation_attestation(
        path=Path(attestation_path),
        expected_sha256=attestation_sha256,
        generation=generation,
        release_id=release_id,
        immutable_pins_sha256=immutable_pins_sha256,
        expected_environment_hashes=expected_environment_hashes,
        expected_prefixes=expected_prefixes,
        verify_metadata=False,
    )
    if lease.get("attestation_id") != attestation.get("attestation_id"):
        raise RuntimeIntegrityError("runtime lease attestation ID drifted")
    expected_metadata_hashes = {
        role: str(attestation["environments"][role]["metadata"]["metadata_sha256"])
        for role in ("harness", "serving")
    }
    if metadata_hashes != expected_metadata_hashes:
        raise RuntimeIntegrityError("runtime lease metadata binding drifted")
    return lease


def refresh_generation_lease(
    *,
    state_dir: Path,
    attestation_path: Path,
    attestation_sha256: str,
    generation: int,
    release_id: str,
    immutable_pins_sha256: str,
    expected_environment_hashes: Mapping[str, str],
    expected_prefixes: Mapping[str, str],
    now: float | None = None,
    force: bool = False,
    refresh_seconds: float = LEASE_REFRESH_SECONDS,
    ttl_seconds: float = LEASE_TTL_SECONDS,
) -> dict[str, Any]:
    """Centrally rescan metadata when due and atomically refresh the startup lease."""

    timestamp = time.time() if now is None else float(now)
    if (
        not math.isfinite(timestamp)
        or isinstance(refresh_seconds, bool)
        or isinstance(ttl_seconds, bool)
        or not isinstance(refresh_seconds, (int, float))
        or not isinstance(ttl_seconds, (int, float))
        or not math.isfinite(float(refresh_seconds))
        or not math.isfinite(float(ttl_seconds))
        or refresh_seconds <= 0
        or ttl_seconds <= refresh_seconds
    ):
        raise RuntimeIntegrityError("runtime lease cadence/TTL is unsafe")
    _validate_generation(generation)
    _validate_release_id(release_id)
    resolved_state = Path(state_dir).expanduser().resolve()
    lease_path = generation_lease_path(resolved_state, generation)
    lock = resolved_state / "locks" / "runtime-integrity.lock"
    with _exclusive_lock(lock):
        previous: dict[str, Any] | None = None
        if lease_path.exists() or lease_path.is_symlink():
            previous = verify_generation_lease(
                lease_path=lease_path,
                attestation_path=attestation_path,
                attestation_sha256=attestation_sha256,
                generation=generation,
                release_id=release_id,
                immutable_pins_sha256=immutable_pins_sha256,
                expected_environment_hashes=expected_environment_hashes,
                expected_prefixes=expected_prefixes,
                now=timestamp,
                require_fresh=False,
            )
            age = timestamp - float(previous["verified_timestamp"])
            if not force and 0 <= age < refresh_seconds:
                if timestamp > float(previous["expires_timestamp"]):
                    raise RuntimeIntegrityError(
                        "runtime lease expired inside its refresh interval"
                    )
                return {"path": str(lease_path), "record": previous, "cached": True}
        attestation = verify_generation_attestation(
            path=Path(attestation_path),
            expected_sha256=attestation_sha256,
            generation=generation,
            release_id=release_id,
            immutable_pins_sha256=immutable_pins_sha256,
            expected_environment_hashes=expected_environment_hashes,
            expected_prefixes=expected_prefixes,
            verify_metadata=True,
        )
        lease: dict[str, Any] = {
            "schema_version": LEASE_SCHEMA_VERSION,
            "kind": "schema5-runtime-integrity-lease",
            "generation": generation,
            "release_id": release_id,
            "immutable_pins_sha256": immutable_pins_sha256,
            "attestation_path": str(Path(attestation_path).expanduser().resolve()),
            "attestation_sha256": attestation_sha256,
            "attestation_id": str(attestation["attestation_id"]),
            "metadata_algorithm": METADATA_ALGORITHM,
            "environment_metadata_sha256": {
                role: str(
                    attestation["environments"][role]["metadata"]["metadata_sha256"]
                )
                for role in ("harness", "serving")
            },
            "sequence": 1 if previous is None else int(previous["sequence"]) + 1,
            "verified_timestamp": timestamp,
            "expires_timestamp": timestamp + ttl_seconds,
        }
        lease["lease_id"] = sha256_bytes(canonical_bytes(lease))
        _atomic_replace_read_only(lease_path, _json_bytes(lease))
        return {"path": str(lease_path), "record": lease, "cached": False}


def _parse_role_pair(values: Sequence[str], *, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise RuntimeIntegrityError(f"{label} must use ROLE=VALUE")
        role, item = value.split("=", 1)
        if role not in {"harness", "serving"} or not item or role in result:
            raise RuntimeIntegrityError(f"invalid {label} {value!r}")
        result[role] = item
    if set(result) != {"harness", "serving"}:
        raise RuntimeIntegrityError(f"{label} requires harness and serving values")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease", required=True)
    parser.add_argument("--attestation", required=True)
    parser.add_argument("--attestation-sha256", required=True)
    parser.add_argument("--generation", required=True, type=int)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--immutable-pins-sha256", required=True)
    parser.add_argument("--environment-hash", action="append", default=[])
    parser.add_argument("--environment-prefix", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        hashes = _parse_role_pair(args.environment_hash, label="environment hash")
        prefixes = _parse_role_pair(args.environment_prefix, label="environment prefix")
        verify_generation_lease(
            lease_path=Path(args.lease),
            attestation_path=Path(args.attestation),
            attestation_sha256=args.attestation_sha256,
            generation=args.generation,
            release_id=args.release_id,
            immutable_pins_sha256=args.immutable_pins_sha256,
            expected_environment_hashes=hashes,
            expected_prefixes=prefixes,
        )
    except RuntimeIntegrityError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
