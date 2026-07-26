"""Immutable authority for schema-5 runtime-generation identities.

Result rows carry a *joint* serving identity.  None of its five marginal fields is
independently sufficient: accepting the cross product of individually observed values
would silently bless a serving generation that never existed.  This module therefore
publishes marker-last, read-only catalog revisions keyed by the exact tuple

``(release fleet, effective fleet, capacity, rollout, endpoint)``.

The mutable ``CURRENT.json`` file is only an atomic discovery pointer.  Trust comes from
the referenced immutable revision, its complete inventory, the endpoint-history marker
for every entry, and sealed copies of the generation evidence.  Finalization can pin a
specific revision and validate it without consulting ``CURRENT.json``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from agents_scaling.serving.registry import (
    EndpointHistoryRecord,
    validate_endpoint_history_marker,
)


CATALOG_DIRECTORY = "trusted-generation-catalog-schema5-v1"
CATALOG_SCHEMA_VERSION = 1
CATALOG_MARKER = "TRUSTED_GENERATION_CATALOG_COMPLETE.json"
CATALOG_PAYLOAD = "CATALOG.json"
CATALOG_INVENTORY = "INVENTORY.sha256"
CATALOG_CURRENT = "CURRENT.json"
CATALOG_LOCK = ".catalog.lock"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ENTRY_FIELDS = (
    "release_fleet_contract_sha256",
    "fleet_contract_sha256",
    "capacity_generation",
    "rollout_generation",
    "endpoint_generation",
)


class GenerationCatalogError(RuntimeError):
    """A trusted-generation catalog cannot be safely published or validated."""


@dataclass(frozen=True)
class GenerationEvidence:
    """One already-validated control-plane artifact copied into a catalog revision."""

    kind: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class TrustedGenerationCatalog:
    """One fully verified immutable catalog revision."""

    marker_path: Path
    marker_sha256: str
    inventory_sha256: str
    catalog_sha256: str
    catalog_id: str
    entries: tuple[Mapping[str, Any], ...]
    generations: tuple[Mapping[str, Any], ...]
    allowed_generation_tuples: frozenset[tuple[str, str, int, int, str]]


def catalog_root(state_dir: str | os.PathLike[str]) -> Path:
    return Path(state_dir).expanduser().resolve() / CATALOG_DIRECTORY


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GenerationCatalogError(f"catalog value is not canonical JSON: {exc}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    """Create one independent regular file and fsync it before publication."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write while creating {path}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_pointer(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


@contextmanager
def _catalog_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / CATALOG_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _relative_endpoint_marker(
    marker: Path, *, server_pool_root: Path
) -> str:
    try:
        relative = marker.resolve().relative_to(server_pool_root.resolve())
    except ValueError as exc:
        raise GenerationCatalogError(
            f"endpoint-history marker escapes the canonical server pool: {marker}"
        ) from exc
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() in {"", "."}
    ):
        raise GenerationCatalogError(f"unsafe endpoint-history path: {relative}")
    return relative.as_posix()


def _tuple_from_mapping(value: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
    release = value.get("release_fleet_contract_sha256")
    fleet = value.get("fleet_contract_sha256")
    capacity = value.get("capacity_generation")
    rollout = value.get("rollout_generation")
    endpoint = value.get("endpoint_generation")
    if (
        not isinstance(release, str)
        or _SHA256_RE.fullmatch(release) is None
        or not isinstance(fleet, str)
        or _SHA256_RE.fullmatch(fleet) is None
        or not isinstance(capacity, int)
        or isinstance(capacity, bool)
        or capacity < 1
        or not isinstance(rollout, int)
        or isinstance(rollout, bool)
        or rollout < 1
        or not isinstance(endpoint, str)
        or not endpoint
        or endpoint == "mixed"
    ):
        raise GenerationCatalogError("trusted-generation tuple is malformed")
    return release, fleet, capacity, rollout, endpoint


def _generation_key(value: Mapping[str, Any]) -> tuple[int, int]:
    capacity = value.get("capacity_generation")
    rollout = value.get("rollout_generation")
    if (
        not isinstance(capacity, int)
        or isinstance(capacity, bool)
        or capacity < 1
        or not isinstance(rollout, int)
        or isinstance(rollout, bool)
        or rollout < 1
    ):
        raise GenerationCatalogError("catalog generation key is malformed")
    return capacity, rollout


def _inventory_rows(directory: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise GenerationCatalogError(f"catalog staging contains a symlink: {path}")
        if path.is_file() and path.name not in {CATALOG_INVENTORY, CATALOG_MARKER}:
            rows.append((path.relative_to(directory).as_posix(), _sha256_file(path)))
    return rows


def _inventory_bytes(rows: Iterable[tuple[str, str]]) -> bytes:
    return "".join(f"{digest}  {name}\n" for name, digest in rows).encode("utf-8")


def _seal_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise GenerationCatalogError(f"cannot seal symlinked catalog member: {path}")
        if path.is_file():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        elif path.is_dir():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)
    _fsync_directory(root)


def _archive_stale_partials(root: Path, *, now: float) -> None:
    staging_root = root / ".staging"
    if not staging_root.exists():
        return
    if staging_root.is_symlink() or not staging_root.is_dir():
        raise GenerationCatalogError("catalog staging root is unsafe")
    partials = sorted(path for path in staging_root.iterdir())
    if not partials:
        return
    archive_root = root / "stale-partials"
    archive_root.mkdir(parents=True, exist_ok=True)
    for partial in partials:
        if partial.is_symlink() or not partial.is_dir():
            raise GenerationCatalogError(f"unsafe catalog partial: {partial}")
        archive_id = f"{int(now):010d}-{partial.name}-{uuid.uuid4().hex[:12]}"
        destination = archive_root / archive_id
        os.rename(partial, destination)
        for name, archived_name in (
            (CATALOG_INVENTORY, "PREIMAGE_INVENTORY.sha256"),
            (CATALOG_MARKER, "PREIMAGE_MARKER.json"),
        ):
            control = destination / name
            if control.exists():
                if control.is_symlink() or not control.is_file():
                    raise GenerationCatalogError(
                        f"unsafe stale catalog control preimage: {control}"
                    )
                os.rename(control, destination / archived_name)
        rows = _inventory_rows(destination)
        inventory = _inventory_bytes(rows)
        _write_new_file(destination / CATALOG_INVENTORY, inventory)
        marker = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "kind": "schema5_trusted_generation_catalog_stale_partial",
            "archive_id": archive_id,
            "file_count": len(rows),
            "inventory_sha256": _sha256_bytes(inventory),
            "archived_timestamp": float(now),
        }
        _write_new_file(destination / CATALOG_MARKER, _canonical_bytes(marker))
        _seal_tree_read_only(destination)
    _fsync_directory(staging_root)
    _fsync_directory(archive_root)


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenerationCatalogError(f"cannot read {description}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GenerationCatalogError(f"{description} is not an object")
    return payload


def validate_trusted_generation_catalog(
    marker_path: str | os.PathLike[str],
    *,
    server_pool_root: str | os.PathLike[str],
) -> TrustedGenerationCatalog:
    """Validate one sealed revision without reading the mutable current pointer."""

    marker_path = Path(marker_path).expanduser().resolve()
    revision = marker_path.parent
    pool = Path(server_pool_root).expanduser().resolve()
    if (
        marker_path.name != CATALOG_MARKER
        or revision.is_symlink()
        or not revision.is_dir()
        or revision.stat().st_mode & 0o222
    ):
        raise GenerationCatalogError("trusted-generation catalog revision is unsafe")
    allowed_top = {CATALOG_PAYLOAD, CATALOG_INVENTORY, CATALOG_MARKER, "evidence"}
    observed_top = {path.name for path in revision.iterdir()}
    if observed_top != allowed_top:
        raise GenerationCatalogError(
            "trusted-generation revision member set drifted: "
            f"{sorted(observed_top ^ allowed_top)}"
        )
    evidence_dir = revision / "evidence"
    if (
        evidence_dir.is_symlink()
        or not evidence_dir.is_dir()
        or evidence_dir.stat().st_mode & 0o222
    ):
        raise GenerationCatalogError("trusted-generation evidence directory is unsafe")

    seen_inodes: set[tuple[int, int]] = set()
    for path in revision.rglob("*"):
        if path.is_symlink():
            raise GenerationCatalogError(f"catalog contains a symlink: {path}")
        if path.is_file():
            observed = path.stat(follow_symlinks=False)
            inode = (observed.st_dev, observed.st_ino)
            if (
                observed.st_nlink != 1
                or observed.st_mode & 0o222
                or inode in seen_inodes
            ):
                raise GenerationCatalogError(
                    f"catalog member is mutable/shared: {path}"
                )
            seen_inodes.add(inode)
        elif path.is_dir() and path.stat().st_mode & 0o222:
            raise GenerationCatalogError(f"catalog directory is writable: {path}")

    inventory_path = revision / CATALOG_INVENTORY
    inventory_bytes = inventory_path.read_bytes()
    expected_inventory = _inventory_bytes(_inventory_rows(revision))
    if inventory_bytes != expected_inventory:
        raise GenerationCatalogError("trusted-generation inventory drifted")
    inventory_sha256 = _sha256_bytes(inventory_bytes)
    marker = _read_json(marker_path, description="trusted-generation marker")
    marker_fields = {
        "schema_version",
        "kind",
        "catalog_id",
        "catalog_sha256",
        "inventory_sha256",
        "file_count",
        "entry_count",
        "generation_count",
        "previous_marker_sha256",
        "published_timestamp",
    }
    catalog_path = revision / CATALOG_PAYLOAD
    catalog_sha256 = _sha256_file(catalog_path)
    if (
        set(marker) != marker_fields
        or marker.get("schema_version") != CATALOG_SCHEMA_VERSION
        or marker.get("kind") != "schema5_trusted_generation_catalog_complete"
        or marker.get("catalog_id") != revision.name
        or marker.get("catalog_sha256") != catalog_sha256
        or marker.get("inventory_sha256") != inventory_sha256
        or marker.get("file_count") != len(_inventory_rows(revision))
        or not isinstance(marker.get("published_timestamp"), (int, float))
        or isinstance(marker.get("published_timestamp"), bool)
    ):
        raise GenerationCatalogError("trusted-generation marker is malformed")
    catalog = _read_json(catalog_path, description="trusted-generation payload")
    payload_fields = {
        "schema_version",
        "kind",
        "server_pool_id",
        "previous_catalog",
        "generations",
        "entries",
    }
    entries = catalog.get("entries")
    generations = catalog.get("generations")
    if (
        set(catalog) != payload_fields
        or catalog.get("schema_version") != CATALOG_SCHEMA_VERSION
        or catalog.get("kind") != "schema5_trusted_generation_catalog"
        or not isinstance(catalog.get("server_pool_id"), str)
        or not catalog["server_pool_id"]
        or not isinstance(entries, list)
        or not isinstance(generations, list)
        or marker.get("entry_count") != len(entries)
        or marker.get("generation_count") != len(generations)
    ):
        raise GenerationCatalogError("trusted-generation catalog payload is malformed")

    previous = catalog.get("previous_catalog")
    if previous is None:
        if marker.get("previous_marker_sha256") is not None:
            raise GenerationCatalogError("catalog genesis has a previous marker hash")
    elif (
        not isinstance(previous, dict)
        or set(previous) != {"catalog_id", "marker_sha256"}
        or _SHA256_RE.fullmatch(str(previous.get("catalog_id", ""))) is None
        or _SHA256_RE.fullmatch(str(previous.get("marker_sha256", ""))) is None
        or marker.get("previous_marker_sha256") != previous["marker_sha256"]
    ):
        raise GenerationCatalogError("catalog previous-revision binding is malformed")

    generation_keys: list[tuple[int, int]] = []
    generation_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    expected_previous: tuple[int, int] | None = None
    for generation in generations:
        if not isinstance(generation, dict) or set(generation) != {
            "release_fleet_contract_sha256",
            "fleet_contract_sha256",
            "capacity_generation",
            "rollout_generation",
            "transition_kind",
            "previous_generation",
            "evidence",
        }:
            raise GenerationCatalogError("catalog generation record is malformed")
        release, fleet, capacity, rollout, _ = _tuple_from_mapping(
            {**generation, "endpoint_generation": "generation-proof"}
        )
        key = (capacity, rollout)
        if key in generation_by_key:
            raise GenerationCatalogError("catalog duplicates a generation")
        previous_generation = generation.get("previous_generation")
        if expected_previous is None:
            if (
                key != (1, 1)
                or previous_generation is not None
                or generation.get("transition_kind") != "initial_readiness"
            ):
                raise GenerationCatalogError(
                    "trusted-generation chain does not start at capacity/rollout 1"
                )
        else:
            if (
                not isinstance(previous_generation, dict)
                or set(previous_generation)
                != {"capacity_generation", "rollout_generation"}
                or (
                    previous_generation["capacity_generation"],
                    previous_generation["rollout_generation"],
                )
                != expected_previous
                or rollout != expected_previous[1] + 1
                or capacity not in {expected_previous[0], expected_previous[0] + 1}
            ):
                raise GenerationCatalogError("trusted-generation chain is discontinuous")
            expected_kind = (
                "capacity_transition"
                if capacity == expected_previous[0] + 1
                else "rollout_transition"
            )
            if generation.get("transition_kind") != expected_kind:
                raise GenerationCatalogError("generation transition kind is incorrect")
        evidence = generation.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise GenerationCatalogError("catalog generation has no sealed evidence")
        for item in evidence:
            if (
                not isinstance(item, dict)
                or set(item) != {"kind", "path", "sha256", "source_path"}
                or not isinstance(item.get("kind"), str)
                or not item["kind"]
                or _SHA256_RE.fullmatch(str(item.get("sha256", ""))) is None
                or not isinstance(item.get("path"), str)
                or Path(item["path"]).is_absolute()
                or ".." in Path(item["path"]).parts
                or not isinstance(item.get("source_path"), str)
            ):
                raise GenerationCatalogError("catalog generation evidence is malformed")
            evidence_path = revision / item["path"]
            if (
                not evidence_path.is_file()
                or _sha256_file(evidence_path) != item["sha256"]
            ):
                raise GenerationCatalogError("catalog generation evidence drifted")
        generation_keys.append(key)
        generation_by_key[key] = generation
        expected_previous = key
        if _SHA256_RE.fullmatch(release) is None or _SHA256_RE.fullmatch(fleet) is None:
            raise GenerationCatalogError("generation hashes are malformed")
    if generation_keys != sorted(generation_keys, key=lambda item: item[1]):
        raise GenerationCatalogError("catalog generations are not rollout ordered")

    allowed: set[tuple[str, str, int, int, str]] = set()
    canonical_entries: list[Mapping[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            *_ENTRY_FIELDS,
            "endpoint_history_path",
            "endpoint_history_marker_sha256",
        }:
            raise GenerationCatalogError("trusted-generation entry is malformed")
        identity = _tuple_from_mapping(entry)
        if identity in allowed:
            raise GenerationCatalogError("trusted-generation tuple is duplicated")
        generation = generation_by_key.get((identity[2], identity[3]))
        if (
            generation is None
            or generation["release_fleet_contract_sha256"] != identity[0]
            or generation["fleet_contract_sha256"] != identity[1]
        ):
            raise GenerationCatalogError(
                "trusted-generation tuple has no exact generation proof"
            )
        relative = Path(str(entry["endpoint_history_path"]))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in {"", "."}
            or _SHA256_RE.fullmatch(
                str(entry.get("endpoint_history_marker_sha256", ""))
            )
            is None
        ):
            raise GenerationCatalogError("endpoint-history catalog path is unsafe")
        try:
            endpoint_record = validate_endpoint_history_marker(pool / relative)
        except (OSError, ValueError) as exc:
            raise GenerationCatalogError(
                f"endpoint-history marker is invalid: {relative}: {exc}"
            ) from exc
        if (
            endpoint_record.allowed_generation_tuple != identity
            or endpoint_record.marker_sha256
            != entry["endpoint_history_marker_sha256"]
        ):
            raise GenerationCatalogError(
                "endpoint-history marker does not prove the catalog tuple"
            )
        allowed.add(identity)
        canonical_entries.append(entry)
    if entries != sorted(
        canonical_entries, key=lambda row: tuple(row[field] for field in _ENTRY_FIELDS)
    ):
        raise GenerationCatalogError("trusted-generation entries are not canonical")
    marker_sha256 = _sha256_file(marker_path)
    return TrustedGenerationCatalog(
        marker_path=marker_path,
        marker_sha256=marker_sha256,
        inventory_sha256=inventory_sha256,
        catalog_sha256=catalog_sha256,
        catalog_id=str(marker["catalog_id"]),
        entries=tuple(entries),
        generations=tuple(generations),
        allowed_generation_tuples=frozenset(allowed),
    )


def load_current_trusted_generation_catalog(
    state_dir: str | os.PathLike[str],
    *,
    server_pool_root: str | os.PathLike[str],
    required: bool = True,
) -> TrustedGenerationCatalog | None:
    """Resolve and verify the immutable revision named by ``CURRENT.json``."""

    root = catalog_root(state_dir)
    pointer_path = root / CATALOG_CURRENT
    if not pointer_path.exists():
        if required:
            raise GenerationCatalogError("trusted-generation CURRENT.json is missing")
        return None
    if pointer_path.is_symlink() or not pointer_path.is_file():
        raise GenerationCatalogError("trusted-generation current pointer is unsafe")
    pointer = _read_json(pointer_path, description="trusted-generation current pointer")
    if set(pointer) != {
        "schema_version",
        "kind",
        "catalog_id",
        "marker_path",
        "marker_sha256",
        "inventory_sha256",
    }:
        raise GenerationCatalogError("trusted-generation current pointer is malformed")
    marker = Path(str(pointer.get("marker_path", ""))).expanduser().resolve()
    expected_parent = (root / "revisions" / str(pointer.get("catalog_id", ""))).resolve()
    if (
        pointer.get("schema_version") != CATALOG_SCHEMA_VERSION
        or pointer.get("kind") != "schema5_trusted_generation_catalog_current"
        or marker.parent != expected_parent
        or marker.name != CATALOG_MARKER
        or _SHA256_RE.fullmatch(str(pointer.get("marker_sha256", ""))) is None
        or _SHA256_RE.fullmatch(str(pointer.get("inventory_sha256", ""))) is None
    ):
        raise GenerationCatalogError("trusted-generation current pointer identity drifted")
    catalog = validate_trusted_generation_catalog(
        marker, server_pool_root=server_pool_root
    )
    if (
        catalog.catalog_id != pointer["catalog_id"]
        or catalog.marker_sha256 != pointer["marker_sha256"]
        or catalog.inventory_sha256 != pointer["inventory_sha256"]
    ):
        raise GenerationCatalogError("trusted-generation current pointer drifted")
    return catalog


def publish_trusted_generation_catalog(
    state_dir: str | os.PathLike[str],
    *,
    server_pool_root: str | os.PathLike[str],
    server_pool_id: str,
    endpoint_records: Sequence[EndpointHistoryRecord],
    release_fleet_contract_sha256: str,
    fleet_contract_sha256: str,
    capacity_generation: int,
    rollout_generation: int,
    generation_evidence: Sequence[GenerationEvidence],
    now: float | None = None,
) -> TrustedGenerationCatalog:
    """Append current endpoint histories to a crash-safe immutable catalog revision."""

    timestamp = time.time() if now is None else float(now)
    root = catalog_root(state_dir)
    pool = Path(server_pool_root).expanduser().resolve()
    with _catalog_lock(root):
        _archive_stale_partials(root, now=timestamp)
        previous = load_current_trusted_generation_catalog(
            state_dir, server_pool_root=pool, required=False
        )
        expected_identity = (
            release_fleet_contract_sha256,
            fleet_contract_sha256,
            int(capacity_generation),
            int(rollout_generation),
        )
        if (
            _SHA256_RE.fullmatch(release_fleet_contract_sha256) is None
            or _SHA256_RE.fullmatch(fleet_contract_sha256) is None
            or capacity_generation < 1
            or rollout_generation < 1
            or not server_pool_id
        ):
            raise GenerationCatalogError("catalog publication identity is malformed")
        if not endpoint_records:
            raise GenerationCatalogError(
                "catalog publication requires at least one endpoint-history record"
            )
        current_records: list[EndpointHistoryRecord] = []
        for record in endpoint_records:
            if record.allowed_generation_tuple[:4] == expected_identity:
                current_records.append(record)
        if not current_records:
            raise GenerationCatalogError(
                "endpoint history contains no record for the target generation"
            )

        previous_entries = [] if previous is None else [dict(row) for row in previous.entries]
        entry_by_tuple = {
            _tuple_from_mapping(row): dict(row) for row in previous_entries
        }
        previous_endpoint_tuples = {
            record.allowed_generation_tuple: record for record in endpoint_records
        }
        # A newly discovered old-generation marker is not retroactively trusted.  It
        # must have been cataloged while that generation was current.
        unknown_historical = sorted(
            identity
            for identity in previous_endpoint_tuples
            if identity[:4] != expected_identity and identity not in entry_by_tuple
        )
        if unknown_historical:
            raise GenerationCatalogError(
                "uncataloged historical endpoint identities appeared: "
                f"{unknown_historical[:3]}"
            )
        for record in current_records:
            identity = record.allowed_generation_tuple
            row = {
                "release_fleet_contract_sha256": identity[0],
                "fleet_contract_sha256": identity[1],
                "capacity_generation": identity[2],
                "rollout_generation": identity[3],
                "endpoint_generation": identity[4],
                "endpoint_history_path": _relative_endpoint_marker(
                    record.marker_path, server_pool_root=pool
                ),
                "endpoint_history_marker_sha256": record.marker_sha256,
            }
            existing = entry_by_tuple.get(identity)
            if existing is not None and existing != row:
                raise GenerationCatalogError(
                    "trusted-generation tuple changed endpoint-history binding"
                )
            entry_by_tuple[identity] = row
        entries = [
            entry_by_tuple[key]
            for key in sorted(entry_by_tuple)
        ]

        previous_generations = (
            [] if previous is None else [dict(row) for row in previous.generations]
        )
        generation_key = (capacity_generation, rollout_generation)
        generation_by_key = {
            _generation_key(row): dict(row) for row in previous_generations
        }
        if generation_key not in generation_by_key:
            if not generation_evidence:
                raise GenerationCatalogError(
                    "a new trusted generation requires sealed readiness evidence"
                )
            if previous_generations:
                prior = _generation_key(previous_generations[-1])
                if (
                    rollout_generation != prior[1] + 1
                    or capacity_generation not in {prior[0], prior[0] + 1}
                ):
                    raise GenerationCatalogError(
                        "new trusted generation is not chain-continuous"
                    )
                transition_kind = (
                    "capacity_transition"
                    if capacity_generation == prior[0] + 1
                    else "rollout_transition"
                )
                previous_generation: dict[str, int] | None = {
                    "capacity_generation": prior[0],
                    "rollout_generation": prior[1],
                }
            else:
                if generation_key != (1, 1):
                    raise GenerationCatalogError(
                        "trusted-generation genesis must be capacity 1, rollout 1"
                    )
                transition_kind = "initial_readiness"
                previous_generation = None
        else:
            generation = generation_by_key[generation_key]
            if (
                generation["release_fleet_contract_sha256"]
                != release_fleet_contract_sha256
                or generation["fleet_contract_sha256"] != fleet_contract_sha256
            ):
                raise GenerationCatalogError(
                    "existing generation has a different fleet identity"
                )
            transition_kind = str(generation["transition_kind"])
            previous_generation = generation["previous_generation"]
        if (
            previous is not None
            and len(entry_by_tuple) == len(previous.entries)
            and generation_key in {
                _generation_key(row) for row in previous.generations
            }
        ):
            # No catalog mutation is necessary.  Revalidation above still detects
            # endpoint-history or revision tampering before this idempotent return.
            return previous

        staging_root = root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = staging_root / uuid.uuid4().hex
        staging.mkdir(mode=0o700)
        (staging / "evidence").mkdir(mode=0o700)
        try:
            # Full revisions are self-contained.  Independently recopy every prior
            # generation-evidence artifact from the previously sealed revision first.
            if previous is not None:
                previous_root = previous.marker_path.parent
                remapped: dict[tuple[int, int], dict[str, Any]] = {}
                evidence_number = 0
                for generation in previous.generations:
                    copied = dict(generation)
                    copied_evidence: list[dict[str, str]] = []
                    for item in generation["evidence"]:
                        source = previous_root / str(item["path"])
                        suffix = source.suffix if source.suffix else ".bin"
                        name = (
                            f"{evidence_number:04d}-"
                            f"{re.sub(r'[^A-Za-z0-9._-]+', '-', str(item['kind']))[:48]}-"
                            f"{str(item['sha256'])[:16]}{suffix}"
                        )
                        evidence_number += 1
                        destination = staging / "evidence" / name
                        with source.open("rb") as source_handle:
                            _write_new_file(destination, source_handle.read())
                        if (
                            destination.stat().st_nlink != 1
                            or destination.stat().st_ino == source.stat().st_ino
                            or _sha256_file(destination) != item["sha256"]
                        ):
                            raise GenerationCatalogError(
                                f"prior generation evidence copy drifted: {source}"
                            )
                        copied_evidence.append(
                            {
                                **dict(item),
                                "path": f"evidence/{name}",
                            }
                        )
                    copied["evidence"] = copied_evidence
                    remapped[_generation_key(copied)] = copied
                generation_by_key = remapped
            evidence_number = len(
                [
                    path
                    for path in (staging / "evidence").iterdir()
                    if path.is_file()
                ]
            )
            if generation_key not in generation_by_key:
                evidence_rows: list[dict[str, str]] = []
                seen_evidence: set[tuple[str, str]] = set()
                for evidence in sorted(
                    generation_evidence, key=lambda item: (item.kind, str(item.path))
                ):
                    source = evidence.path.expanduser().resolve()
                    if (
                        source.is_symlink()
                        or not source.is_file()
                        or _SHA256_RE.fullmatch(evidence.sha256) is None
                        or _sha256_file(source) != evidence.sha256
                        or not evidence.kind
                    ):
                        raise GenerationCatalogError(
                            f"generation evidence is missing or drifted: {source}"
                        )
                    key = (evidence.kind, evidence.sha256)
                    if key in seen_evidence:
                        continue
                    seen_evidence.add(key)
                    suffix = source.suffix if source.suffix else ".bin"
                    name = (
                        f"{evidence_number:04d}-"
                        f"{re.sub(r'[^A-Za-z0-9._-]+', '-', evidence.kind)[:48]}-"
                        f"{evidence.sha256[:16]}{suffix}"
                    )
                    evidence_number += 1
                    destination = staging / "evidence" / name
                    with source.open("rb") as source_handle:
                        _write_new_file(destination, source_handle.read())
                    if (
                        destination.stat().st_nlink != 1
                        or destination.stat().st_ino == source.stat().st_ino
                        or _sha256_file(destination) != evidence.sha256
                    ):
                        raise GenerationCatalogError(
                            f"generation evidence was not independently copied: {source}"
                        )
                    evidence_rows.append(
                        {
                            "kind": evidence.kind,
                            "path": f"evidence/{name}",
                            "sha256": evidence.sha256,
                            "source_path": str(source),
                        }
                    )
                generation_by_key[generation_key] = {
                    "release_fleet_contract_sha256": release_fleet_contract_sha256,
                    "fleet_contract_sha256": fleet_contract_sha256,
                    "capacity_generation": capacity_generation,
                    "rollout_generation": rollout_generation,
                    "transition_kind": transition_kind,
                    "previous_generation": previous_generation,
                    "evidence": evidence_rows,
                }
            generations = sorted(
                generation_by_key.values(),
                key=lambda row: int(row["rollout_generation"]),
            )
            payload = {
                "schema_version": CATALOG_SCHEMA_VERSION,
                "kind": "schema5_trusted_generation_catalog",
                "server_pool_id": server_pool_id,
                "previous_catalog": (
                    None
                    if previous is None
                    else {
                        "catalog_id": previous.catalog_id,
                        "marker_sha256": previous.marker_sha256,
                    }
                ),
                "generations": generations,
                "entries": entries,
            }
            payload_bytes = _canonical_bytes(payload)
            catalog_id = _sha256_bytes(payload_bytes)
            existing_revision = root / "revisions" / catalog_id
            if existing_revision.exists():
                shutil.rmtree(staging)
                if (
                    not existing_revision.is_symlink()
                    and existing_revision.is_dir()
                    and existing_revision.stat().st_mode & 0o222
                ):
                    # Crash recovery after atomic publication but before removal of
                    # the revision directory's write bits.
                    _seal_tree_read_only(existing_revision)
                catalog = validate_trusted_generation_catalog(
                    existing_revision / CATALOG_MARKER,
                    server_pool_root=pool,
                )
            else:
                _write_new_file(staging / CATALOG_PAYLOAD, payload_bytes)
                inventory_rows = _inventory_rows(staging)
                inventory_bytes = _inventory_bytes(inventory_rows)
                _write_new_file(staging / CATALOG_INVENTORY, inventory_bytes)
                marker = {
                    "schema_version": CATALOG_SCHEMA_VERSION,
                    "kind": "schema5_trusted_generation_catalog_complete",
                    "catalog_id": catalog_id,
                    "catalog_sha256": _sha256_bytes(payload_bytes),
                    "inventory_sha256": _sha256_bytes(inventory_bytes),
                    "file_count": len(inventory_rows),
                    "entry_count": len(entries),
                    "generation_count": len(generations),
                    "previous_marker_sha256": (
                        None if previous is None else previous.marker_sha256
                    ),
                    "published_timestamp": timestamp,
                }
                _write_new_file(staging / CATALOG_MARKER, _canonical_bytes(marker))
                # Files and subdirectories are immutable before publication.  Keep
                # only the staging root writable until after rename because some
                # shared filesystems reject renaming a mode-0555 directory.
                for child in sorted(staging.rglob("*"), reverse=True):
                    if child.is_symlink():
                        raise GenerationCatalogError(
                            f"cannot publish symlinked catalog member: {child}"
                        )
                    child.chmod(
                        stat.S_IMODE(child.stat().st_mode) & ~0o222
                    )
                _fsync_directory(staging)
                revisions = root / "revisions"
                revisions.mkdir(parents=True, exist_ok=True)
                try:
                    os.rename(staging, existing_revision)
                except FileExistsError:
                    # Another process could only win through a broken external lock.
                    # Validate identical bytes and adopt rather than overwrite.
                    staging.chmod(0o700)
                    shutil.rmtree(staging)
                _seal_tree_read_only(existing_revision)
                _fsync_directory(revisions)
                catalog = validate_trusted_generation_catalog(
                    existing_revision / CATALOG_MARKER,
                    server_pool_root=pool,
                )
            pointer = {
                "schema_version": CATALOG_SCHEMA_VERSION,
                "kind": "schema5_trusted_generation_catalog_current",
                "catalog_id": catalog.catalog_id,
                "marker_path": str(catalog.marker_path),
                "marker_sha256": catalog.marker_sha256,
                "inventory_sha256": catalog.inventory_sha256,
            }
            _atomic_pointer(root / CATALOG_CURRENT, pointer)
            return catalog
        except BaseException:
            # Preserve the complete pre-marker preimage.  The next invocation archives
            # it under ``stale-partials`` before creating any new revision.
            if staging.exists():
                _fsync_directory(staging)
                _fsync_directory(staging_root)
            raise
