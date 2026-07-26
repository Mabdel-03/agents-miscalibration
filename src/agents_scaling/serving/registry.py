"""Server registry: a file-based directory of which vLLM servers serve which model.

A serving SLURM job, once its vLLM server is up, writes a small JSON file recording its
``node:port`` and the model it serves. Cell jobs (the HTTP clients) discover a server by
reading the registry — no service discovery infrastructure needed, just a shared
filesystem (scratch), which is exactly what we have.

**Multi-endpoint:** a single model size may be served by MORE THAN ONE server (e.g. one
on pi_tpoggio + one on ou_bcs) to spread cell load. Each server writes its OWN file under
``<run_root>/servers/<model_size>/<host>_<port>.json`` (so concurrent registrations never
race), and ``lookup_server`` round-robins across whatever endpoints are present.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import stat
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping


@dataclass
class ServerEntry:
    # Scientific model identity (for example ``32B``).  Runtime layouts live in the
    # separate ``serving_profile`` field so long-context deployment never changes cells.
    model_size: str
    hf_id: str
    host: str
    port: int
    slurm_job_id: str | None = None
    started_at: float = 0.0
    serving_profile: str | None = None
    served_model_name: str | None = None
    max_model_len: int | None = None
    tp_size: int | None = None
    # Immutable schema-5 serving release provenance.  These remain optional solely so
    # sealed legacy registry records can still be read and audited; production endpoint
    # selection requires exact non-null matches via ``entry_matches_frozen_provenance``.
    release_id: str | None = None
    environment_hash: str | None = None
    model_revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    model_contract_sha256: str | None = None
    fleet_contract_sha256: str | None = None
    server_pool_id: str | None = None
    replica_id: str | int | None = None
    replica_index: int | None = None
    # Generation identity is optional only for legacy compatibility.  A schema-5
    # production registration supplies all four values and is staged until its
    # immutable endpoint-history record is sealed by the fleet supervisor.
    release_fleet_contract_sha256: str | None = None
    capacity_generation: int | None = None
    rollout_generation: int | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def registry_key(self) -> str:
        """Endpoint-discovery key; falls back for all legacy registry records."""
        return self.serving_profile or self.model_size


SERVER_POOL_GENERATION_VERSION = 2
STANDBY_REGISTRY_DIRECTORY = ".fleet-transactions-v1/standby-registry"
ENDPOINT_HISTORY_DIRECTORY = ".endpoint-history-schema5-v1"
ENDPOINT_HISTORY_SCHEMA_VERSION = 1
ENDPOINT_HISTORY_MARKER = "ENDPOINT_HISTORY_COMPLETE.json"
ENDPOINT_HISTORY_INVENTORY = "INVENTORY.json"
_ENDPOINT_HISTORY_FILES = frozenset(
    {
        "SERVER_ENTRY.json",
        "BINDING.json",
        "LOCAL_SCRIPT.sbatch",
        "SPOOLED_SCRIPT.sbatch",
        ENDPOINT_HISTORY_INVENTORY,
        ENDPOINT_HISTORY_MARKER,
    }
)
_ENDPOINT_HISTORY_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "server_entry_sha256",
        "endpoint_instance_id",
        "endpoint_generation",
        "release_id",
        "environment_hash",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "model_contract_sha256",
        "release_fleet_contract_sha256",
        "fleet_contract_sha256",
        "capacity_generation",
        "rollout_generation",
        "effective_context_limit",
        "tp_size",
        "serving_profile",
        "replica_id",
        "replica_index",
        "slurm_job_id",
        "intent_token",
        "intent_state",
        "committed_at",
        "ledger_generation",
        "local_script_path",
        "local_script_sha256",
        "spooled_script_sha256",
        "spooled_provenance",
        "scheduler_job_name",
        "scheduler_comment",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_INTENT_RE = re.compile(r"[0-9a-f]{32}")


@dataclass(frozen=True)
class EndpointHistoryRecord:
    """One independently verifiable, immutable serving-process registration."""

    marker_path: Path
    marker_sha256: str
    server_entry: ServerEntry
    endpoint_instance_id: str
    release_fleet_contract_sha256: str
    fleet_contract_sha256: str
    capacity_generation: int
    rollout_generation: int
    endpoint_generation: str
    binding: Mapping[str, Any]

    @property
    def allowed_generation_tuple(self) -> tuple[str, str, int, int, str]:
        return (
            self.release_fleet_contract_sha256,
            self.fleet_contract_sha256,
            self.capacity_generation,
            self.rollout_generation,
            self.endpoint_generation,
        )


@dataclass(frozen=True)
class EndpointHistoryCatalog:
    """Validated endpoint-lineage authority for a canonical schema-5 pool."""

    records: tuple[EndpointHistoryRecord, ...]
    allowed_generation_tuples: frozenset[tuple[str, str, int, int, str]]
    marker_sha256s: frozenset[str]


def server_pool_id(run_root: str | os.PathLike) -> str:
    """Stable identity for one canonical registry root, independent of its basename."""

    canonical = str(Path(run_root).expanduser().resolve())
    canonical_path = Path(canonical)
    if canonical_path.parts[-2:] == ("server_pools", "schema5-v1"):
        return "schema5-v1"
    basename = "".join(
        character if character.isalnum() else "-" for character in Path(canonical).name
    ).strip("-") or "pool"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    return f"{basename[:20]}-{digest}"


def serving_job_name(run_root: str | os.PathLike, profile_name: str) -> str:
    return f"asys-srv-{server_pool_id(run_root)}-{profile_name}"[:128]


def endpoint_instance_id(entry: ServerEntry) -> str:
    """Return the exact, human-auditable identity of one serving process.

    Host and port identify a socket, not a process: a restarted vLLM server can reuse
    both.  The allocation id and exact round-trippable registration timestamp make the
    identity change on such a restart.  This value belongs on generation provenance;
    it is deliberately not a server-pool hash.
    """

    allocation = str(entry.slurm_job_id or "unmanaged")
    started_at = format(float(entry.started_at), ".17g")
    return f"{allocation}@{entry.host}:{int(entry.port)}#{started_at}"


def server_pool_generation(
    entries: Iterable[ServerEntry], *, profile_name: str | None = None
) -> str:
    """Hash the canonical process/layout identity of a serving-profile fleet.

    The hash is invariant to registry enumeration order and duplicate copies of an
    identical record.  It changes when an endpoint process restarts at the same address,
    when membership changes, or when any serving-layout contract changes.  Callers use
    it only to decide whether a dormant runtime failure may be retried; result rows keep
    :func:`endpoint_instance_id` instead.
    """

    canonical_entries: dict[str, dict[str, object]] = {}
    for entry in entries:
        payload: dict[str, object] = {
            "endpoint_instance_id": endpoint_instance_id(entry),
            "slurm_job_id": None if entry.slurm_job_id is None else str(entry.slurm_job_id),
            "host": str(entry.host),
            "port": int(entry.port),
            "started_at": format(float(entry.started_at), ".17g"),
            "model_size": str(entry.model_size),
            "hf_id": str(entry.hf_id),
            "serving_profile": (
                None if entry.serving_profile is None else str(entry.serving_profile)
            ),
            "served_model_name": (
                None if entry.served_model_name is None else str(entry.served_model_name)
            ),
            "max_model_len": (
                None if entry.max_model_len is None else int(entry.max_model_len)
            ),
            "tp_size": None if entry.tp_size is None else int(entry.tp_size),
            "release_id": entry.release_id,
            "environment_hash": entry.environment_hash,
            "model_revision": entry.model_revision,
            "tokenizer_id": entry.tokenizer_id,
            "tokenizer_revision": entry.tokenizer_revision,
            "model_contract_sha256": entry.model_contract_sha256,
            "fleet_contract_sha256": entry.fleet_contract_sha256,
            "server_pool_id": entry.server_pool_id,
            "replica_id": entry.replica_id,
            "replica_index": entry.replica_index,
            "release_fleet_contract_sha256": (
                entry.release_fleet_contract_sha256
            ),
            "capacity_generation": entry.capacity_generation,
            "rollout_generation": entry.rollout_generation,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        canonical_entries[encoded] = payload
    envelope = {
        "version": SERVER_POOL_GENERATION_VERSION,
        "profile_name": profile_name,
        "endpoints": [canonical_entries[key] for key in sorted(canonical_entries)],
    }
    encoded_envelope = json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    digest = hashlib.sha256(encoded_envelope).hexdigest()
    return f"spg-v{SERVER_POOL_GENERATION_VERSION}:{digest}"


def servers_dir(run_root: str | os.PathLike) -> Path:
    d = Path(run_root) / "servers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _size_dir(run_root: str | os.PathLike, model_size: str) -> Path:
    d = servers_dir(run_root) / model_size
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_replica_name(replica_id: str | int) -> str:
    literal = str(replica_id)
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in literal
    )
    digest = hashlib.sha256(literal.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:96]}.{digest}"


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_bytes(raw: bytes, *, artifact: Path) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        material: dict[str, Any] = {}
        for key, value in pairs:
            if key in material:
                raise ValueError(f"duplicate key {key!r}")
            material[key] = value
        return material

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value!r}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid endpoint-history JSON {artifact}: {exc}") from exc


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_below_without_symlinks(root: Path, directory: Path) -> None:
    """Create one directory chain while rejecting every traversed symlink."""

    root = root.resolve()
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"directory escapes endpoint pool root: {directory}") from exc
    current = root
    if not current.exists():
        current.mkdir(parents=True)
    if current.is_symlink() or not current.is_dir():
        raise ValueError(f"endpoint pool root is unsafe: {current}")
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"endpoint directory is symlinked: {current}")
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise ValueError(f"endpoint path is not a directory: {current}")


def _write_private_copy(path: Path, payload: bytes) -> None:
    """Create one independent regular-file preimage and durably write every byte."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written == 0:
                raise OSError(f"zero-byte endpoint-history write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _positive_finite(value: object) -> bool:
    try:
        material = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(material) and material > 0


def _production_lineage_complete(entry: ServerEntry) -> bool:
    """Return whether ``entry`` carries the complete schema-5 generation identity."""

    return bool(
        entry.server_pool_id == "schema5-v1"
        and isinstance(entry.replica_id, str)
        and entry.replica_id.startswith("schema5-v1--")
        and isinstance(entry.replica_index, int)
        and not isinstance(entry.replica_index, bool)
        and entry.replica_index >= 0
        and str(entry.slurm_job_id or "").isdigit()
        and _positive_finite(entry.started_at)
        and isinstance(entry.serving_profile, str)
        and bool(entry.serving_profile)
        and isinstance(entry.served_model_name, str)
        and bool(entry.served_model_name)
        and isinstance(entry.max_model_len, int)
        and not isinstance(entry.max_model_len, bool)
        and entry.max_model_len > 0
        and isinstance(entry.tp_size, int)
        and not isinstance(entry.tp_size, bool)
        and entry.tp_size > 0
        and isinstance(entry.release_id, str)
        and bool(entry.release_id)
        and isinstance(entry.environment_hash, str)
        and _SHA256_RE.fullmatch(entry.environment_hash)
        and isinstance(entry.model_revision, str)
        and bool(entry.model_revision)
        and isinstance(entry.tokenizer_id, str)
        and bool(entry.tokenizer_id)
        and isinstance(entry.tokenizer_revision, str)
        and bool(entry.tokenizer_revision)
        and isinstance(entry.model_contract_sha256, str)
        and _SHA256_RE.fullmatch(entry.model_contract_sha256)
        and isinstance(entry.release_fleet_contract_sha256, str)
        and _SHA256_RE.fullmatch(entry.release_fleet_contract_sha256)
        and isinstance(entry.fleet_contract_sha256, str)
        and _SHA256_RE.fullmatch(entry.fleet_contract_sha256)
        and type(entry.capacity_generation) is int
        and entry.capacity_generation > 0
        and type(entry.rollout_generation) is int
        and entry.rollout_generation > 0
    )


def _claims_schema5_production(entry: ServerEntry) -> bool:
    """Return whether a canonical-pool entry claims any production provenance.

    Old compatibility fixtures and sealed legacy registries can use schema-5-shaped
    replica names without claiming a frozen release.  Once any production identity is
    present, however, omitting the rest must fail closed rather than bypass the immutable
    endpoint-history gate.
    """

    return bool(
        entry.server_pool_id == "schema5-v1"
        and isinstance(entry.replica_id, str)
        and entry.replica_id.startswith("schema5-v1--")
        and any(
            value is not None
            for value in (
                entry.release_id,
                entry.environment_hash,
                entry.model_revision,
                entry.tokenizer_id,
                entry.tokenizer_revision,
                entry.model_contract_sha256,
                entry.fleet_contract_sha256,
                entry.release_fleet_contract_sha256,
                entry.capacity_generation,
                entry.rollout_generation,
            )
        )
    )


def endpoint_history_root(run_root: str | os.PathLike) -> Path:
    return Path(run_root).expanduser().resolve() / ENDPOINT_HISTORY_DIRECTORY


def endpoint_history_directory(
    run_root: str | os.PathLike,
    profile_name: str,
    replica_id: str,
    job_id: str,
) -> Path:
    if (
        not isinstance(profile_name, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", profile_name) is None
        or profile_name == ".staging"
    ):
        raise ValueError("endpoint history requires a safe serving profile")
    if not isinstance(replica_id, str) or not replica_id.startswith("schema5-v1--"):
        raise ValueError("endpoint history requires an exact schema-5 replica")
    if re.fullmatch(r"[0-9]+", str(job_id)) is None:
        raise ValueError("endpoint history requires an exact numeric Slurm job id")
    return (
        endpoint_history_root(run_root)
        / profile_name
        / _safe_replica_name(replica_id)
        / str(job_id)
    )


def _history_inventory_payload(directory: Path) -> dict[str, Any]:
    names = (
        "BINDING.json",
        "LOCAL_SCRIPT.sbatch",
        "SERVER_ENTRY.json",
        "SPOOLED_SCRIPT.sbatch",
    )
    records: list[dict[str, Any]] = []
    inodes: set[tuple[int, int]] = set()
    for name in names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"endpoint-history preimage is unsafe: {path}")
        observed = path.stat(follow_symlinks=False)
        inode = (observed.st_dev, observed.st_ino)
        if observed.st_nlink != 1 or inode in inodes:
            raise ValueError(f"endpoint-history preimage shares an inode: {path}")
        inodes.add(inode)
        raw = path.read_bytes()
        records.append(
            {
                "path": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        )
    return {
        "schema_version": ENDPOINT_HISTORY_SCHEMA_VERSION,
        "algorithm": "sha256",
        "files": records,
    }


def seal_endpoint_history(
    run_root: str | os.PathLike,
    entry: ServerEntry,
    *,
    release_fleet_contract_sha256: str,
    capacity_generation: int,
    rollout_generation: int,
    ledger_generation: int,
    intent_token: str,
    intent_state: str,
    committed_at: float,
    local_script_path: str | os.PathLike,
    local_script_sha256: str,
    spooled_script: str | bytes,
    spooled_provenance: Mapping[str, Any],
    scheduler_job_name: str,
    scheduler_comment: str,
    sealed_at: float | None = None,
) -> EndpointHistoryRecord:
    """Publish one marker-last, immutable, job-specific endpoint registration.

    Publication happens in an unadvertised staging directory.  The complete directory
    is atomically renamed into the catalog only after all independent byte copies,
    their sorted inventory, and the completion marker have been fsynced.  A hard kill
    can therefore leave an ignored staging directory, never a half-authoritative
    endpoint.  Repeating an already completed seal fully revalidates and returns it.
    """

    root = Path(run_root).expanduser().resolve()
    if not _production_lineage_complete(entry):
        raise ValueError("endpoint history requires complete schema-5 lineage")
    if (
        entry.release_fleet_contract_sha256 != release_fleet_contract_sha256
        or entry.capacity_generation != capacity_generation
        or entry.rollout_generation != rollout_generation
        or ledger_generation != rollout_generation
        or type(capacity_generation) is not int
        or capacity_generation < 1
        or type(rollout_generation) is not int
        or rollout_generation < 1
        or type(ledger_generation) is not int
        or ledger_generation < 1
        or _SHA256_RE.fullmatch(release_fleet_contract_sha256) is None
        or _SHA256_RE.fullmatch(str(entry.fleet_contract_sha256)) is None
        or _SHA256_RE.fullmatch(local_script_sha256) is None
        or _INTENT_RE.fullmatch(intent_token) is None
        or intent_state != "committed"
        or not isinstance(committed_at, (int, float))
        or isinstance(committed_at, bool)
        or float(committed_at) <= 0
    ):
        raise ValueError("endpoint-history transaction/generation binding is invalid")
    if (
        not isinstance(scheduler_job_name, str)
        or not scheduler_job_name
        or not isinstance(scheduler_comment, str)
        or not scheduler_comment
    ):
        raise ValueError("endpoint history requires exact scheduler identity")
    expected_endpoint = endpoint_instance_id(entry)
    job_id = str(entry.slurm_job_id)
    supplied_spooled_bytes = (
        spooled_script.encode("utf-8")
        if isinstance(spooled_script, str)
        else bytes(spooled_script)
    )
    supplied_spooled_sha256 = hashlib.sha256(supplied_spooled_bytes).hexdigest()
    supplied_local_path = str(Path(local_script_path).expanduser().resolve())
    destination = endpoint_history_directory(
        root,
        str(entry.serving_profile),
        str(entry.replica_id),
        job_id,
    )
    if destination.exists():
        # Crash recovery for the sole boundary between atomic directory publication
        # and removal of the directory write bits.  All files and the marker were
        # already fsynced before rename; finish the seal, then perform the same complete
        # cryptographic validation as an ordinary idempotent call.
        if (
            not destination.is_symlink()
            and destination.is_dir()
            and stat.S_IMODE(destination.stat().st_mode) & 0o222
            and {path.name for path in destination.iterdir()}
            == _ENDPOINT_HISTORY_FILES
            and all(
                path.is_file()
                and not path.is_symlink()
                and not (stat.S_IMODE(path.stat().st_mode) & 0o222)
                for path in destination.iterdir()
            )
        ):
            destination.chmod(0o555)
            _fsync_dir(destination)
            _fsync_dir(destination.parent)
        record = validate_endpoint_history_marker(destination / ENDPOINT_HISTORY_MARKER)
        if (
            record.server_entry != entry
            or record.endpoint_instance_id != expected_endpoint
            or record.release_fleet_contract_sha256
            != release_fleet_contract_sha256
            or record.fleet_contract_sha256 != entry.fleet_contract_sha256
            or record.capacity_generation != capacity_generation
            or record.rollout_generation != rollout_generation
            or record.binding.get("intent_token") != intent_token
            or record.binding.get("intent_state") != intent_state
            or record.binding.get("committed_at") != float(committed_at)
            or record.binding.get("ledger_generation") != ledger_generation
            or record.binding.get("local_script_path") != supplied_local_path
            or record.binding.get("local_script_sha256") != local_script_sha256
            or record.binding.get("spooled_script_sha256")
            != supplied_spooled_sha256
            or record.binding.get("spooled_provenance")
            != dict(spooled_provenance)
            or record.binding.get("scheduler_job_name") != scheduler_job_name
            or record.binding.get("scheduler_comment") != scheduler_comment
        ):
            raise ValueError(f"endpoint-history collision: {destination}")
        return record

    local_source = Path(local_script_path).expanduser()
    if (
        not local_source.is_absolute()
        or local_source.is_symlink()
        or not local_source.is_file()
        or stat.S_IMODE(local_source.stat().st_mode) & 0o222
    ):
        raise ValueError("endpoint history requires a read-only local Slurm script")
    local_bytes = local_source.read_bytes()
    if hashlib.sha256(local_bytes).hexdigest() != local_script_sha256:
        raise ValueError("endpoint-history local script hash drifted")
    spooled_bytes = supplied_spooled_bytes
    spooled_sha256 = hashlib.sha256(spooled_bytes).hexdigest()
    if spooled_bytes != local_bytes:
        raise ValueError("endpoint-history local and actual spooled scripts differ")

    entry_bytes = _canonical_json_bytes(asdict(entry))
    standby = standby_entry_path(
        root,
        str(entry.serving_profile),
        str(entry.replica_id),
        job_id,
    )
    _mkdir_below_without_symlinks(root, standby.parent)
    _mkdir_below_without_symlinks(
        root,
        root / "servers" / str(entry.serving_profile),
    )
    promoted = promoted_entry_path(
        root,
        str(entry.serving_profile),
        str(entry.replica_id),
    )
    live_preimages = [path for path in (standby, promoted) if path.is_file()]
    if (
        not live_preimages
        or any(path.is_symlink() for path in live_preimages)
        or not any(path.read_bytes() == entry_bytes for path in live_preimages)
    ):
        raise ValueError("endpoint-history ServerEntry bytes lack an exact live preimage")

    expected_provenance = {
        "run_root": str(root),
        "server_pool_id": entry.server_pool_id,
        "replica_id": entry.replica_id,
        "replica_index": entry.replica_index,
        "release_id": entry.release_id,
        "environment_hash": entry.environment_hash,
        "model_revision": entry.model_revision,
        "tokenizer_id": entry.tokenizer_id,
        "tokenizer_revision": entry.tokenizer_revision,
        "model_contract_sha256": entry.model_contract_sha256,
        "release_fleet_contract_sha256": release_fleet_contract_sha256,
        "fleet_contract_sha256": entry.fleet_contract_sha256,
        "capacity_generation": capacity_generation,
        "rollout_generation": rollout_generation,
        "effective_context_limit": entry.max_model_len,
        "tp_size": entry.tp_size,
        "spooled_script_sha256": spooled_sha256,
    }
    if dict(spooled_provenance) != expected_provenance:
        raise ValueError("endpoint-history spooled provenance drifted")

    binding = {
        "schema_version": ENDPOINT_HISTORY_SCHEMA_VERSION,
        "server_entry_sha256": hashlib.sha256(entry_bytes).hexdigest(),
        "endpoint_instance_id": expected_endpoint,
        "endpoint_generation": expected_endpoint,
        "release_id": entry.release_id,
        "environment_hash": entry.environment_hash,
        "model_revision": entry.model_revision,
        "tokenizer_id": entry.tokenizer_id,
        "tokenizer_revision": entry.tokenizer_revision,
        "model_contract_sha256": entry.model_contract_sha256,
        "release_fleet_contract_sha256": release_fleet_contract_sha256,
        "fleet_contract_sha256": entry.fleet_contract_sha256,
        "capacity_generation": capacity_generation,
        "rollout_generation": rollout_generation,
        "effective_context_limit": entry.max_model_len,
        "tp_size": entry.tp_size,
        "serving_profile": entry.serving_profile,
        "replica_id": entry.replica_id,
        "replica_index": entry.replica_index,
        "slurm_job_id": job_id,
        "intent_token": intent_token,
        "intent_state": intent_state,
        "committed_at": float(committed_at),
        "ledger_generation": ledger_generation,
        "local_script_path": str(local_source.resolve()),
        "local_script_sha256": local_script_sha256,
        "spooled_script_sha256": spooled_sha256,
        "spooled_provenance": expected_provenance,
        "scheduler_job_name": scheduler_job_name,
        "scheduler_comment": scheduler_comment,
    }
    binding_bytes = _canonical_json_bytes(binding)

    history = endpoint_history_root(root)
    staging_parent = history / ".staging"
    final_parent = destination.parent
    for directory in (history, staging_parent, final_parent):
        _mkdir_below_without_symlinks(root, directory)
    staging = staging_parent / (
        f"{_safe_replica_name(str(entry.replica_id))}.{job_id}.{uuid.uuid4().hex}.partial"
    )
    staging.mkdir(mode=0o700)
    try:
        _write_private_copy(staging / "SERVER_ENTRY.json", entry_bytes)
        _write_private_copy(staging / "BINDING.json", binding_bytes)
        _write_private_copy(staging / "LOCAL_SCRIPT.sbatch", local_bytes)
        _write_private_copy(staging / "SPOOLED_SCRIPT.sbatch", spooled_bytes)
        inventory = _history_inventory_payload(staging)
        inventory_bytes = _canonical_json_bytes(inventory)
        _write_private_copy(staging / ENDPOINT_HISTORY_INVENTORY, inventory_bytes)
        marker = {
            "schema_version": ENDPOINT_HISTORY_SCHEMA_VERSION,
            "kind": "schema5_endpoint_history_complete",
            "server_pool_id": entry.server_pool_id,
            "serving_profile": entry.serving_profile,
            "replica_id": entry.replica_id,
            "slurm_job_id": job_id,
            "endpoint_instance_id": expected_endpoint,
            "release_fleet_contract_sha256": release_fleet_contract_sha256,
            "fleet_contract_sha256": entry.fleet_contract_sha256,
            "capacity_generation": capacity_generation,
            "rollout_generation": rollout_generation,
            "endpoint_generation": expected_endpoint,
            "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            "binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
            "server_entry_sha256": hashlib.sha256(entry_bytes).hexdigest(),
            "sealed_at": float(time.time() if sealed_at is None else sealed_at),
        }
        # Marker-last: no authoritative name exists until this final file and the
        # complete staging directory are both durable.
        _write_private_copy(
            staging / ENDPOINT_HISTORY_MARKER,
            _canonical_json_bytes(marker),
        )
        for path in staging.iterdir():
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        _fsync_dir(staging)
        _fsync_dir(staging_parent)
        try:
            os.rename(staging, destination)
        except FileExistsError:
            existing = validate_endpoint_history_marker(
                destination / ENDPOINT_HISTORY_MARKER
            )
            if existing.server_entry != entry:
                raise ValueError(f"endpoint-history collision: {destination}")
            return existing
        destination.chmod(0o555)
        _fsync_dir(destination)
        _fsync_dir(final_parent)
    except BaseException:
        if staging.exists():
            try:
                staging.chmod(0o700)
                for child in staging.iterdir():
                    if not child.is_symlink():
                        child.chmod(0o600)
                shutil.rmtree(staging)
                _fsync_dir(staging_parent)
            except OSError:
                # A hard failure may leave an unadvertised staging preimage.  Catalog
                # validation never treats it as lineage and the next seal uses a fresh
                # staging name.
                pass
        raise
    return validate_endpoint_history_marker(destination / ENDPOINT_HISTORY_MARKER)


def validate_endpoint_history_marker(
    marker_path: str | os.PathLike,
) -> EndpointHistoryRecord:
    """Validate a sealed endpoint archive without consulting mutable live pointers."""

    marker = Path(marker_path).expanduser()
    directory = marker.parent
    if marker.name != ENDPOINT_HISTORY_MARKER:
        raise ValueError("endpoint-history marker has the wrong name")
    try:
        replica_directory = directory.parent
        profile_directory = replica_directory.parent
        history_directory = profile_directory.parent
        pool_directory = history_directory.parent
    except IndexError as exc:
        raise ValueError("endpoint-history marker path is too shallow") from exc
    if history_directory.name != ENDPOINT_HISTORY_DIRECTORY:
        raise ValueError(f"endpoint-history marker path is misplaced: {marker}")
    for ancestor in (
        history_directory,
        profile_directory,
        replica_directory,
        directory,
    ):
        if ancestor.is_symlink() or not ancestor.is_dir():
            raise ValueError(f"endpoint-history directory is unsafe: {ancestor}")
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"endpoint-history directory is unsafe: {directory}")
    observed_names = {path.name for path in directory.iterdir()}
    if observed_names != _ENDPOINT_HISTORY_FILES:
        raise ValueError(
            f"endpoint-history files drifted in {directory}: "
            f"{sorted(observed_names ^ _ENDPOINT_HISTORY_FILES)}"
        )
    inodes: set[tuple[int, int]] = set()
    raw_by_name: dict[str, bytes] = {}
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"endpoint-history artifact is unsafe: {path}")
        observed = path.stat(follow_symlinks=False)
        inode = (observed.st_dev, observed.st_ino)
        if (
            observed.st_nlink != 1
            or inode in inodes
            or stat.S_IMODE(observed.st_mode) & 0o222
        ):
            raise ValueError(f"endpoint-history artifact is mutable/shared: {path}")
        inodes.add(inode)
        raw_by_name[path.name] = path.read_bytes()
    if stat.S_IMODE(directory.stat().st_mode) & 0o222:
        raise ValueError(f"endpoint-history directory remains writable: {directory}")

    marker_payload = _strict_json_bytes(
        raw_by_name[ENDPOINT_HISTORY_MARKER], artifact=marker
    )
    expected_marker_fields = {
        "schema_version",
        "kind",
        "server_pool_id",
        "serving_profile",
        "replica_id",
        "slurm_job_id",
        "endpoint_instance_id",
        "release_fleet_contract_sha256",
        "fleet_contract_sha256",
        "capacity_generation",
        "rollout_generation",
        "endpoint_generation",
        "inventory_sha256",
        "binding_sha256",
        "server_entry_sha256",
        "sealed_at",
    }
    if (
        not isinstance(marker_payload, dict)
        or set(marker_payload) != expected_marker_fields
        or marker_payload["schema_version"] != ENDPOINT_HISTORY_SCHEMA_VERSION
        or marker_payload["kind"] != "schema5_endpoint_history_complete"
    ):
        raise ValueError(f"endpoint-history marker fields drifted: {marker}")
    for field in (
        "release_fleet_contract_sha256",
        "fleet_contract_sha256",
        "inventory_sha256",
        "binding_sha256",
        "server_entry_sha256",
    ):
        if _SHA256_RE.fullmatch(str(marker_payload[field])) is None:
            raise ValueError(f"endpoint-history marker has invalid {field}: {marker}")
    if (
        marker_payload.get("server_pool_id") != "schema5-v1"
        or not isinstance(marker_payload.get("serving_profile"), str)
        or not str(marker_payload["serving_profile"])
        or not isinstance(marker_payload.get("replica_id"), str)
        or not str(marker_payload["replica_id"]).startswith("schema5-v1--")
        or not str(marker_payload.get("slurm_job_id", "")).isdigit()
        or type(marker_payload.get("capacity_generation")) is not int
        or marker_payload["capacity_generation"] < 1
        or type(marker_payload.get("rollout_generation")) is not int
        or marker_payload["rollout_generation"] < 1
        or not _positive_finite(marker_payload.get("sealed_at"))
    ):
        raise ValueError(f"endpoint-history marker identity is invalid: {marker}")
    if (
        hashlib.sha256(raw_by_name[ENDPOINT_HISTORY_INVENTORY]).hexdigest()
        != marker_payload["inventory_sha256"]
        or hashlib.sha256(raw_by_name["BINDING.json"]).hexdigest()
        != marker_payload["binding_sha256"]
        or hashlib.sha256(raw_by_name["SERVER_ENTRY.json"]).hexdigest()
        != marker_payload["server_entry_sha256"]
    ):
        raise ValueError(f"endpoint-history marker hash drifted: {marker}")
    inventory = _strict_json_bytes(
        raw_by_name[ENDPOINT_HISTORY_INVENTORY],
        artifact=directory / ENDPOINT_HISTORY_INVENTORY,
    )
    expected_inventory = _history_inventory_payload(directory)
    if inventory != expected_inventory:
        raise ValueError(f"endpoint-history recursive inventory drifted: {directory}")

    entry_payload = _strict_json_bytes(
        raw_by_name["SERVER_ENTRY.json"], artifact=directory / "SERVER_ENTRY.json"
    )
    binding = _strict_json_bytes(
        raw_by_name["BINDING.json"], artifact=directory / "BINDING.json"
    )
    try:
        entry = ServerEntry(**entry_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid endpoint-history ServerEntry: {exc}") from exc
    if (
        not _production_lineage_complete(entry)
        or not isinstance(binding, dict)
        or set(binding) != _ENDPOINT_HISTORY_BINDING_FIELDS
        or binding.get("schema_version") != ENDPOINT_HISTORY_SCHEMA_VERSION
        or binding.get("server_entry_sha256") != marker_payload["server_entry_sha256"]
        or binding.get("endpoint_instance_id") != endpoint_instance_id(entry)
        or binding.get("endpoint_generation") != endpoint_instance_id(entry)
        or marker_payload["endpoint_instance_id"] != endpoint_instance_id(entry)
        or marker_payload["endpoint_generation"] != endpoint_instance_id(entry)
        or marker_payload["server_pool_id"] != entry.server_pool_id
        or marker_payload["serving_profile"] != entry.serving_profile
        or marker_payload["replica_id"] != entry.replica_id
        or marker_payload["slurm_job_id"] != str(entry.slurm_job_id)
        or marker_payload["release_fleet_contract_sha256"]
        != entry.release_fleet_contract_sha256
        or marker_payload["fleet_contract_sha256"] != entry.fleet_contract_sha256
        or marker_payload["capacity_generation"] != entry.capacity_generation
        or marker_payload["rollout_generation"] != entry.rollout_generation
    ):
        raise ValueError(f"endpoint-history entry/binding identity drifted: {directory}")
    expected_binding_values = {
        "release_id": entry.release_id,
        "environment_hash": entry.environment_hash,
        "model_revision": entry.model_revision,
        "tokenizer_id": entry.tokenizer_id,
        "tokenizer_revision": entry.tokenizer_revision,
        "model_contract_sha256": entry.model_contract_sha256,
        "release_fleet_contract_sha256": entry.release_fleet_contract_sha256,
        "fleet_contract_sha256": entry.fleet_contract_sha256,
        "capacity_generation": entry.capacity_generation,
        "rollout_generation": entry.rollout_generation,
        "effective_context_limit": entry.max_model_len,
        "tp_size": entry.tp_size,
        "serving_profile": entry.serving_profile,
        "replica_id": entry.replica_id,
        "replica_index": entry.replica_index,
        "slurm_job_id": str(entry.slurm_job_id),
        "ledger_generation": entry.rollout_generation,
        "intent_state": "committed",
    }
    if any(binding.get(field) != value for field, value in expected_binding_values.items()):
        raise ValueError(f"endpoint-history binding provenance drifted: {directory}")
    if (
        _INTENT_RE.fullmatch(str(binding.get("intent_token"))) is None
        or not _positive_finite(binding.get("committed_at"))
        or not isinstance(binding.get("scheduler_job_name"), str)
        or not binding["scheduler_job_name"]
        or not isinstance(binding.get("scheduler_comment"), str)
        or not binding["scheduler_comment"]
        or not isinstance(binding.get("local_script_path"), str)
        or not Path(binding["local_script_path"]).is_absolute()
        or _SHA256_RE.fullmatch(str(binding.get("local_script_sha256"))) is None
        or binding.get("local_script_sha256")
        != hashlib.sha256(raw_by_name["LOCAL_SCRIPT.sbatch"]).hexdigest()
        or binding.get("spooled_script_sha256")
        != hashlib.sha256(raw_by_name["SPOOLED_SCRIPT.sbatch"]).hexdigest()
        or raw_by_name["LOCAL_SCRIPT.sbatch"] != raw_by_name["SPOOLED_SCRIPT.sbatch"]
    ):
        raise ValueError(f"endpoint-history script provenance drifted: {directory}")
    expected_provenance = {
        "run_root": str(pool_directory.resolve()),
        "server_pool_id": entry.server_pool_id,
        "replica_id": entry.replica_id,
        "replica_index": entry.replica_index,
        "release_id": entry.release_id,
        "environment_hash": entry.environment_hash,
        "model_revision": entry.model_revision,
        "tokenizer_id": entry.tokenizer_id,
        "tokenizer_revision": entry.tokenizer_revision,
        "model_contract_sha256": entry.model_contract_sha256,
        "release_fleet_contract_sha256": entry.release_fleet_contract_sha256,
        "fleet_contract_sha256": entry.fleet_contract_sha256,
        "capacity_generation": entry.capacity_generation,
        "rollout_generation": entry.rollout_generation,
        "effective_context_limit": entry.max_model_len,
        "tp_size": entry.tp_size,
        "spooled_script_sha256": binding.get("spooled_script_sha256"),
    }
    if binding.get("spooled_provenance") != expected_provenance:
        raise ValueError(f"endpoint-history spooled provenance drifted: {directory}")
    expected_comment = (
        f"asys-s5-fleet:pool={entry.server_pool_id};"
        f"profile={entry.serving_profile};replica={entry.replica_id};"
        f"generation={entry.rollout_generation};"
        f"intent={binding['intent_token']};fleet={entry.fleet_contract_sha256}"
    )
    local_script_path = Path(binding["local_script_path"])
    if (
        binding["scheduler_comment"] != expected_comment
        or not binding["scheduler_job_name"].startswith("asys-s5-serve-")
        or local_script_path.name
        != (
            f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', str(entry.replica_id))}."
            f"{binding['intent_token']}.sbatch"
        )
        or local_script_path.parent.name
        != f"g{int(entry.rollout_generation):06d}"
        or local_script_path.parent.parent.name != "sbatch"
        or local_script_path.parent.parent.parent.name != ".fleet-transactions-v1"
    ):
        raise ValueError(f"endpoint-history scheduler/intent binding drifted: {directory}")
    expected_directory = endpoint_history_directory(
        expected_provenance["run_root"],
        str(entry.serving_profile),
        str(entry.replica_id),
        str(entry.slurm_job_id),
    )
    if directory.resolve() != expected_directory.resolve():
        raise ValueError(f"endpoint-history path identity drifted: {directory}")
    return EndpointHistoryRecord(
        marker_path=marker.resolve(),
        marker_sha256=hashlib.sha256(raw_by_name[ENDPOINT_HISTORY_MARKER]).hexdigest(),
        server_entry=entry,
        endpoint_instance_id=endpoint_instance_id(entry),
        release_fleet_contract_sha256=str(entry.release_fleet_contract_sha256),
        fleet_contract_sha256=str(entry.fleet_contract_sha256),
        capacity_generation=int(entry.capacity_generation),
        rollout_generation=int(entry.rollout_generation),
        endpoint_generation=endpoint_instance_id(entry),
        binding=MappingProxyType(dict(binding)),
    )


def collect_endpoint_history_catalog(
    run_root: str | os.PathLike,
) -> EndpointHistoryCatalog:
    """Enumerate every sealed marker and reject unsafe/unlisted catalog structure."""

    root = endpoint_history_root(run_root)
    if not root.exists():
        return EndpointHistoryCatalog((), frozenset(), frozenset())
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"endpoint-history root is unsafe: {root}")
    records: list[EndpointHistoryRecord] = []
    for path in root.rglob("*"):
        try:
            path.relative_to(root / ".staging")
        except ValueError:
            pass
        else:
            if path.is_symlink():
                raise ValueError(f"endpoint-history staging artifact is symlinked: {path}")
            continue
        if path.is_symlink():
            raise ValueError(f"endpoint-history catalog contains a symlink: {path}")
        if path.is_file() and path.name == ENDPOINT_HISTORY_MARKER:
            records.append(validate_endpoint_history_marker(path))
        elif path.is_file():
            # Every authoritative data file must live beside exactly one marker.
            marker = path.parent / ENDPOINT_HISTORY_MARKER
            if not marker.is_file():
                raise ValueError(f"unlisted endpoint-history file: {path}")
    records.sort(
        key=lambda item: (
            str(item.server_entry.serving_profile),
            str(item.server_entry.replica_id),
            int(str(item.server_entry.slurm_job_id)),
        )
    )
    keys = [
        (
            str(record.server_entry.replica_id),
            str(record.server_entry.slurm_job_id),
        )
        for record in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("endpoint-history catalog duplicates a job-specific registration")
    job_ids = [str(record.server_entry.slurm_job_id) for record in records]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError(
            "endpoint-history catalog binds one Slurm job to multiple replicas"
        )
    return EndpointHistoryCatalog(
        records=tuple(records),
        allowed_generation_tuples=frozenset(
            record.allowed_generation_tuple for record in records
        ),
        marker_sha256s=frozenset(record.marker_sha256 for record in records),
    )


def endpoint_history_for_entry(
    run_root: str | os.PathLike,
    entry: ServerEntry,
) -> EndpointHistoryRecord | None:
    """Return the sealed immutable preimage for ``entry``, if it exists."""

    if not _production_lineage_complete(entry):
        return None
    path = endpoint_history_directory(
        run_root,
        str(entry.serving_profile),
        str(entry.replica_id),
        str(entry.slurm_job_id),
    )
    marker = path / ENDPOINT_HISTORY_MARKER
    if not marker.exists():
        return None
    record = validate_endpoint_history_marker(marker)
    if record.server_entry != entry:
        raise ValueError("endpoint-history registration differs from live ServerEntry")
    return record


def promoted_entry_path(
    run_root: str | os.PathLike,
    profile_name: str,
    replica_id: str | int,
) -> Path:
    """Return the one atomic discovery pointer for a schema-5 logical replica."""

    return _size_dir(run_root, profile_name) / (
        f"replica-{_safe_replica_name(replica_id)}.json"
    )


def standby_entry_path(
    run_root: str | os.PathLike,
    profile_name: str,
    replica_id: str | int,
    job_id: str,
) -> Path:
    if not str(job_id).isdigit():
        raise ValueError("standby registry requires an exact numeric Slurm job id")
    root = Path(run_root).expanduser().resolve()
    directory = (
        root
        / STANDBY_REGISTRY_DIRECTORY
        / profile_name
        / _safe_replica_name(replica_id)
    )
    return directory / f"{job_id}.json"


def read_standby_entry(
    run_root: str | os.PathLike,
    profile_name: str,
    replica_id: str | int,
    job_id: str,
) -> ServerEntry | None:
    path = standby_entry_path(run_root, profile_name, replica_id, job_id)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe standby registry entry: {path}")
    try:
        entry = ServerEntry(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid standby registry entry {path}: {exc}") from exc
    if (
        entry.registry_key != profile_name
        or entry.replica_id != replica_id
        or str(entry.slurm_job_id or "") != str(job_id)
    ):
        raise ValueError(f"standby registry identity drifted: {path}")
    return entry


def read_promoted_entry(
    run_root: str | os.PathLike,
    profile_name: str,
    replica_id: str | int,
) -> ServerEntry | None:
    """Read the atomic logical-replica pointer without creating registry state."""

    root = Path(run_root).expanduser().resolve()
    path = (
        root
        / "servers"
        / profile_name
        / f"replica-{_safe_replica_name(replica_id)}.json"
    )
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe promoted registry entry: {path}")
    try:
        entry = ServerEntry(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid promoted registry entry {path}: {exc}") from exc
    if entry.registry_key != profile_name or entry.replica_id != replica_id:
        raise ValueError(f"promoted registry identity drifted: {path}")
    return entry


def write_standby_entry(
    run_root: str | os.PathLike,
    entry: ServerEntry,
) -> Path:
    """Durably stage one exactly identified schema-5 allocation, idempotently."""

    if (
        entry.server_pool_id != "schema5-v1"
        or not isinstance(entry.replica_id, str)
        or not entry.replica_id.startswith("schema5-v1--")
        or not str(entry.slurm_job_id or "").isdigit()
        or not entry.serving_profile
    ):
        raise ValueError("only an exact schema-5 allocation can be staged")
    path = standby_entry_path(
        run_root,
        entry.registry_key,
        entry.replica_id,
        str(entry.slurm_job_id),
    )
    if path.is_symlink():
        raise ValueError(f"unsafe standby registry entry: {path}")
    existing = read_standby_entry(
        run_root,
        entry.registry_key,
        entry.replica_id,
        str(entry.slurm_job_id),
    )
    if existing is not None and existing != entry:
        raise ValueError(f"standby registry collision: {path}")
    if existing is None:
        _mkdir_below_without_symlinks(
            Path(run_root).expanduser().resolve(),
            path.parent,
        )
        from agents_scaling.experiment import io

        io.atomic_write_text(
            path,
            json.dumps(asdict(entry), indent=2, sort_keys=True) + "\n",
        )
    return path


def promote_standby_entry(
    run_root: str | os.PathLike,
    entry: ServerEntry,
) -> Path:
    """Atomically make one verified standby the logical replica's only endpoint."""

    if (
        entry.server_pool_id != "schema5-v1"
        or not isinstance(entry.replica_id, str)
        or not entry.replica_id.startswith("schema5-v1--")
        or not str(entry.slurm_job_id or "").isdigit()
    ):
        raise ValueError("only an exact schema-5 standby can be promoted")
    claims_production = _claims_schema5_production(entry)
    if claims_production and not _production_lineage_complete(entry):
        raise ValueError(
            "incomplete schema-5 production endpoint cannot become routable"
        )
    if claims_production:
        history = endpoint_history_for_entry(run_root, entry)
        if history is None:
            raise ValueError(
                "schema-5 endpoint cannot become routable before immutable "
                "endpoint history is sealed"
            )
    source = standby_entry_path(
        run_root,
        entry.registry_key,
        entry.replica_id,
        str(entry.slurm_job_id),
    )
    canonical_root = Path(run_root).expanduser().resolve()
    _mkdir_below_without_symlinks(canonical_root, source.parent)
    _mkdir_below_without_symlinks(
        canonical_root,
        canonical_root / "servers" / entry.registry_key,
    )
    destination = promoted_entry_path(
        run_root, entry.registry_key, entry.replica_id
    )
    promoted = read_promoted_entry(
        run_root, entry.registry_key, entry.replica_id
    )
    if source.is_symlink():
        raise ValueError(f"standby preimage is unsafe: {source}")
    if source.is_file():
        persisted = read_standby_entry(
            run_root,
            entry.registry_key,
            entry.replica_id,
            str(entry.slurm_job_id),
        )
        if persisted != entry:
            raise ValueError("standby entry changed before promotion")
    elif promoted == entry:
        # Crash recovery after atomic pointer replacement and durable standby
        # unlinking: the canonical pointer is sufficient promotion evidence.
        return destination
    else:
        raise ValueError(f"standby preimage is unavailable: {source}")
    from agents_scaling.experiment import io

    if promoted != entry:
        io.atomic_write_text(
            destination,
            json.dumps(asdict(entry), indent=2, sort_keys=True) + "\n",
        )
    if not _production_lineage_complete(entry):
        # Legacy compatibility: historical tests/tools used the staging file as a
        # replaceable promotion scratchpad.  Production retains both its sealed archive
        # and job-specific staging registration forever.
        try:
            source.unlink()
        except FileNotFoundError:
            pass
        else:
            descriptor = os.open(
                source.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return destination


def register_server(
    run_root: str | os.PathLike,
    model_size: str,
    hf_id: str,
    port: int,
    *,
    serving_profile: str | None = None,
    served_model_name: str | None = None,
    max_model_len: int | None = None,
    tp_size: int | None = None,
    release_id: str | None = None,
    environment_hash: str | None = None,
    model_revision: str | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    model_contract_sha256: str | None = None,
    fleet_contract_sha256: str | None = None,
    expected_server_pool_id: str | None = None,
    replica_id: str | int | None = None,
    replica_index: int | None = None,
    release_fleet_contract_sha256: str | None = None,
    capacity_generation: int | None = None,
    rollout_generation: int | None = None,
    standby: bool = False,
) -> ServerEntry:
    """Called by a serving job after its vLLM server is healthy.

    Writes a per-server file under ``servers/<size>/<host>_<port>.json``. Multiple
    servers for the same size coexist (one file each); re-registration of the same
    host:port just overwrites its own file.
    """
    observed_pool_id = server_pool_id(run_root)
    if (
        expected_server_pool_id is not None
        and expected_server_pool_id != observed_pool_id
    ):
        raise ValueError(
            "serving registry run-root/pool identity mismatch: "
            f"expected {expected_server_pool_id!r}, observed {observed_pool_id!r}"
        )
    if replica_index is not None and (
        not isinstance(replica_index, int)
        or isinstance(replica_index, bool)
        or replica_index < 0
    ):
        raise ValueError("replica_index must be a non-negative integer when provided")
    if replica_id is not None and (
        not isinstance(replica_id, (str, int))
        or isinstance(replica_id, bool)
        or (isinstance(replica_id, str) and not replica_id)
        or (isinstance(replica_id, int) and replica_id < 0)
    ):
        raise ValueError("replica_id must be a non-empty string or non-negative integer")

    host = socket.gethostname()
    entry = ServerEntry(
        model_size=model_size,
        hf_id=hf_id,
        host=host,
        port=port,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        started_at=time.time(),
        serving_profile=serving_profile,
        served_model_name=served_model_name,
        max_model_len=max_model_len,
        tp_size=tp_size,
        release_id=release_id,
        environment_hash=environment_hash,
        model_revision=model_revision,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        model_contract_sha256=model_contract_sha256,
        fleet_contract_sha256=fleet_contract_sha256,
        server_pool_id=observed_pool_id,
        replica_id=replica_id,
        replica_index=replica_index,
        release_fleet_contract_sha256=release_fleet_contract_sha256,
        capacity_generation=capacity_generation,
        rollout_generation=rollout_generation,
    )
    if _claims_schema5_production(entry) and not _production_lineage_complete(entry):
        raise ValueError(
            "schema-5 production registration requires complete fleet/capacity/"
            "rollout lineage"
        )
    # A complete schema-5 registration is never made routable by the serving job.
    # It first lands in a job-specific staging namespace.  Only the fleet supervisor,
    # after binding the committed intent and actual Slurm-spooled script into an
    # immutable endpoint-history archive, may atomically promote it.
    if standby or _production_lineage_complete(entry):
        if entry.server_pool_id != "schema5-v1" or entry.replica_id is None:
            raise ValueError("standby registration is schema-5 only")
        if not str(entry.slurm_job_id or "").isdigit():
            raise ValueError("standby registration requires SLURM_JOB_ID")
        path = standby_entry_path(
            run_root,
            entry.registry_key,
            entry.replica_id,
            str(entry.slurm_job_id),
        )
        _mkdir_below_without_symlinks(
            Path(run_root).expanduser().resolve(),
            path.parent,
        )
        if _production_lineage_complete(entry):
            existing = read_standby_entry(
                run_root,
                entry.registry_key,
                entry.replica_id,
                str(entry.slurm_job_id),
            )
            if existing is not None:
                if existing != entry:
                    raise ValueError(
                        f"schema-5 job-specific registration collision: {path}"
                    )
                return existing
    elif entry.server_pool_id == "schema5-v1" and entry.replica_id is not None:
        path = promoted_entry_path(
            run_root, entry.registry_key, entry.replica_id
        )
    else:
        path = _size_dir(run_root, entry.registry_key) / f"{host}_{port}.json"
    # Registration is shared control-plane state.  Use the same durable, unique-temp
    # replacement primitive as experiment artifacts so a node failure or concurrent
    # re-registration cannot expose a truncated JSON document.
    from agents_scaling.experiment import io

    payload = _canonical_json_bytes(asdict(entry))
    if _production_lineage_complete(entry):
        try:
            _write_private_copy(path, payload)
        except FileExistsError:
            existing = read_standby_entry(
                run_root,
                entry.registry_key,
                str(entry.replica_id),
                str(entry.slurm_job_id),
            )
            if existing != entry:
                raise ValueError(
                    f"schema-5 job-specific registration collision: {path}"
                )
            return existing
        path.chmod(0o444)
        _fsync_dir(path.parent)
    else:
        io.atomic_write_text(
            path,
            payload.decode("utf-8"),
        )
    return entry


def list_servers(run_root: str | os.PathLike, model_size: str) -> list[ServerEntry]:
    """All registered endpoints for a model size (across clusters)."""
    out: list[ServerEntry] = []
    # New layout: servers/<size>/*.json
    sd = servers_dir(run_root) / model_size
    if sd.is_dir():
        for f in sorted(sd.glob("*.json")):
            try:
                out.append(ServerEntry(**json.loads(f.read_text())))
            except (ValueError, TypeError):
                continue
    # Back-compat: legacy single file servers/<size>.json
    legacy = servers_dir(run_root) / f"{model_size}.json"
    if legacy.is_file():
        try:
            out.append(ServerEntry(**json.loads(legacy.read_text())))
        except (ValueError, TypeError):
            pass
    return out


def prune_entry(run_root: str | os.PathLike, entry: ServerEntry) -> None:
    """Delete the registry file(s) for one endpoint (used when it is found dead) so cells
    stop round-robining to it. Handles both the per-server and legacy single-file layouts."""
    candidates = [
        servers_dir(run_root) / entry.registry_key / f"{entry.host}_{entry.port}.json",
        servers_dir(run_root) / f"{entry.registry_key}.json",
    ]
    if entry.server_pool_id == "schema5-v1" and entry.replica_id is not None:
        candidates.append(
            promoted_entry_path(run_root, entry.registry_key, entry.replica_id)
        )
    for p in candidates:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


def entry_matches_profile(entry: ServerEntry, profile_name: str) -> bool:
    """Return whether a registry record truthfully describes ``profile_name``.

    Historical records predate explicit serving-profile fields.  They are compatible only
    with the standard profile whose name is the scientific model size; a legacy record can
    never be treated as the TP=2 40K profile.  New records must provide the complete runtime
    layout, preventing a misplaced or partially written registry file from defeating the
    client's context preflight assumptions.
    """
    from agents_scaling.serving.profiles import get_serving_profile

    try:
        profile = get_serving_profile(profile_name)
    except KeyError:
        return False
    if entry.model_size != profile.model_size or entry.hf_id != profile.hf_id:
        return False

    if entry.serving_profile is None:
        if profile.name != profile.model_size:
            return False
        optional_values = (
            (entry.served_model_name, profile.served_model_name),
            (entry.max_model_len, profile.max_model_len),
            (entry.tp_size, profile.tp_size),
        )
        return all(observed is None or observed == expected for observed, expected in optional_values)

    return bool(
        entry.serving_profile == profile.name
        and entry.served_model_name == profile.served_model_name
        and entry.max_model_len == profile.max_model_len
        and entry.tp_size == profile.tp_size
    )


def entry_has_current_provenance(entry: ServerEntry, profile_name: str) -> bool:
    """Return whether ``entry`` can identify one exact current-protocol process.

    Historical standard-profile records may omit their runtime layout, allocation id, or
    registration timestamp.  They remain readable through :func:`list_servers` and
    :func:`entry_matches_profile` for legacy audit/repair, but they cannot truthfully
    populate a current result row's ``endpoint_generation``.  Current workers and the
    global dispatcher therefore require this stronger predicate before routing or
    admitting work.

    ``started_at`` is the server registration (or verified self-heal) timestamp.  In
    combination with allocation id, host, and port it distinguishes a replacement vLLM
    process even when Slurm requeues a job onto the same address.
    """

    if not entry_matches_profile(entry, profile_name):
        return False
    allocation = entry.slurm_job_id
    if allocation is None or not str(allocation).strip():
        return False
    try:
        registered_at = float(entry.started_at)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(registered_at) or registered_at <= 0:
        return False

    # Unlike the compatibility branch in ``entry_matches_profile``, current provenance
    # must bind the explicit deployed layout rather than infer it from the directory name.
    from agents_scaling.serving.profiles import get_serving_profile

    try:
        profile = get_serving_profile(profile_name)
    except KeyError:
        return False
    return bool(
        entry.serving_profile == profile.name
        and entry.served_model_name == profile.served_model_name
        and entry.max_model_len == profile.max_model_len
        and entry.tp_size == profile.tp_size
    )


def entry_matches_frozen_provenance(
    entry: ServerEntry,
    profile_name: str,
    *,
    release_id: str,
    environment_hash: str,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    model_contract_sha256: str,
    fleet_contract_sha256: str,
    expected_server_pool_id: str | None = None,
) -> bool:
    """Return whether an endpoint exactly matches one immutable schema-5 release.

    Compatibility readers intentionally tolerate historical records.  Production routing
    must not: a partial registration, a different tokenizer commit, or a server from a
    prior environment generation is excluded before any HTTP request is issued.
    """

    if not entry_has_current_provenance(entry, profile_name):
        return False
    expected = {
        "release_id": release_id,
        "environment_hash": environment_hash,
        "model_revision": model_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "model_contract_sha256": model_contract_sha256,
        "fleet_contract_sha256": fleet_contract_sha256,
    }
    if expected_server_pool_id is not None:
        if entry.server_pool_id != expected_server_pool_id:
            return False
        if entry.server_pool_id == "schema5-v1" and (
            not isinstance(entry.replica_id, str)
            or not entry.replica_id.startswith("schema5-v1--")
            or not isinstance(entry.replica_index, int)
            or isinstance(entry.replica_index, bool)
            or entry.replica_index < 0
        ):
            return False
        if entry.server_pool_id != "schema5-v1" and entry.replica_id is None:
            return False
    return all(
        isinstance(value, str)
        and bool(value)
        and getattr(entry, field) == value
        for field, value in expected.items()
    )


def active_slurm_allocations(job_ids: Iterable[str]) -> dict[str, frozenset[str]] | None:
    """Resolve live serving allocations to their current nodes.

    A successful empty query is authoritative: every requested allocation has left the
    queue.  A command/parse failure is different and deliberately falls back to tolerant
    HTTP probes rather than declaring all servers dead.  Matching both job id and node is
    essential because a requeued/preempted job can retain its id while moving nodes,
    leaving a stale registry file from the earlier allocation behind.
    """
    requested = sorted({str(job_id).split("_", 1)[0] for job_id in job_ids if job_id})
    if not requested:
        return {}
    try:
        proc = subprocess.run(
            ["squeue", "-h", "-j", ",".join(requested), "-o", "%i|%T|%N"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None

    active: dict[str, frozenset[str]] = {}
    try:
        for raw in proc.stdout.splitlines():
            if not raw.strip():
                continue
            job_id, state, node_list = (part.strip() for part in raw.split("|", 2))
            if state.upper() in {"RUNNING", "COMPLETING", "CONFIGURING"}:
                # Serving jobs request GPUs on one node.  Keep the representation a set so
                # an unusual comma-separated allocation remains safe; compressed NodeList
                # syntax is treated as unknown and therefore falls back to HTTP below.
                nodes = (
                    frozenset(part.strip() for part in node_list.split(",") if part.strip())
                    if node_list not in {"", "(null)", "N/A"} and "[" not in node_list
                    else frozenset()
                )
                active[job_id.split("_", 1)[0]] = nodes
    except ValueError:
        return None
    return active


def list_live_servers(
    run_root: str | os.PathLike,
    model_size: str,
    *,
    probe_timeout: float = 3.0,
    probe_attempts: int = 3,
    require_current_provenance: bool = False,
) -> list[ServerEntry]:
    """Return validated live endpoints without mutating the shared registry.

    Records tied to a running Slurm job are accepted from scheduler authority, avoiding a
    health-check storm when a busy vLLM engine is merely slow to answer ``/health``.  Legacy
    records with no job id (and all records when Slurm cannot be queried) use repeated HTTP
    probes and accept any success.  A failed observation is never sufficient to delete a
    shared registry record; keepalive/server lifecycle code owns eventual cleanup.  With
    ``require_current_provenance=True``, legacy or incomplete process identities are
    retained on disk for audit but excluded before either scheduler lookup or HTTP probing.
    """
    if probe_attempts < 1:
        raise ValueError("probe_attempts must be >= 1")
    from agents_scaling.serving import healthcheck

    entries = [
        entry
        for entry in list_servers(run_root, model_size)
        if entry_matches_profile(entry, model_size)
        and (
            not require_current_provenance
            or entry_has_current_provenance(entry, model_size)
        )
    ]
    job_entries = [entry for entry in entries if entry.slurm_job_id]
    active_allocations = active_slurm_allocations(
        entry.slurm_job_id or "" for entry in job_entries
    )

    authoritative: list[ServerEntry] = []
    to_probe: list[ServerEntry] = []
    for entry in entries:
        job_id = (entry.slurm_job_id or "").split("_", 1)[0]
        if entry.slurm_job_id and active_allocations is not None:
            allocated_nodes = active_allocations.get(job_id)
            if allocated_nodes and entry.host in allocated_nodes:
                authoritative.append(entry)
            elif allocated_nodes is not None and not allocated_nodes:
                # Allocation is active but its node could not be represented safely.
                to_probe.append(entry)
            # A successful Slurm query that omits this allocation is authoritative too:
            # as is a job id now allocated on a different node.  Retain the file for
            # audit/self-heal, but do not route traffic to it.
            continue
        to_probe.append(entry)

    def tolerantly_alive(entry: ServerEntry) -> bool:
        return any(
            healthcheck.is_alive(entry.host, entry.port, timeout=probe_timeout)
            for _ in range(probe_attempts)
        )

    probed_live: list[ServerEntry] = []
    if to_probe:
        with ThreadPoolExecutor(max_workers=min(32, len(to_probe))) as executor:
            outcomes = executor.map(tolerantly_alive, to_probe)
            probed_live = [entry for entry, alive in zip(to_probe, outcomes) if alive]
    return sorted(
        authoritative + probed_live,
        key=lambda entry: (entry.host, entry.port, entry.slurm_job_id or ""),
    )


def lookup_server(
    run_root: str | os.PathLike,
    model_size: str,
    shard: int = 0,
    *,
    live_only: bool = False,
    probe_timeout: float = 3.0,
    require_current_provenance: bool = False,
) -> ServerEntry | None:
    """Pick one endpoint for ``model_size``, round-robined by ``shard`` (e.g. the SLURM
    array task id) so cells spread across all available servers for that size.

    With ``live_only=True``, invalid or unavailable entries are filtered non-destructively
    via ``list_live_servers`` before round-robin. ``require_current_provenance`` applies the
    stronger current-result identity contract. The keepalive self-heal path keeps the
    default raw-filesystem read so it can intentionally see and reason about stale entries.
    """
    if live_only:
        servers = list_live_servers(
            run_root,
            model_size,
            probe_timeout=probe_timeout,
            require_current_provenance=require_current_provenance,
        )
    else:
        servers = list_servers(run_root, model_size)
    if not servers:
        return None
    return servers[shard % len(servers)]


def wait_for_server(
    run_root: str | os.PathLike,
    model_size: str,
    shard: int = 0,
    timeout_s: float = 300.0,
    poll_s: float = 5.0,
    require_current_provenance: bool = False,
) -> ServerEntry:
    """Block until at least one server has registered for ``model_size`` (cells call this).

    Timeout is 300s (not 3600s): if a size's server is absent (e.g. its GPU jobs are stuck
    pending during a cluster GPU crunch), a cell that waits a full hour burns a whole CPU
    node producing nothing. 300s lets a serverless cell fail fast so the resumable runner
    recycles the chunk quickly instead — yet stays comfortably above a real vLLM server's
    boot+load+health lag (~1-3 min) so we don't false-timeout a server about to register.
    The cell is resumable by cell_id, so a fast failure here costs nothing but is retried
    cheaply once the server returns."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        entry = lookup_server(
            run_root,
            model_size,
            shard=shard,
            live_only=True,
            require_current_provenance=require_current_provenance,
        )
        if entry is not None:
            return entry
        time.sleep(poll_s)
    raise TimeoutError(
        f"no server registered for model_size={model_size!r} within {timeout_s}s "
        f"(looked in {servers_dir(run_root)})"
    )
