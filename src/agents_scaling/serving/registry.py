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
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


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

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def registry_key(self) -> str:
        """Endpoint-discovery key; falls back for all legacy registry records."""
        return self.serving_profile or self.model_size


SERVER_POOL_GENERATION_VERSION = 1


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
    )
    path = _size_dir(run_root, entry.registry_key) / f"{host}_{port}.json"
    # Registration is shared control-plane state.  Use the same durable, unique-temp
    # replacement primitive as experiment artifacts so a node failure or concurrent
    # re-registration cannot expose a truncated JSON document.
    from agents_scaling.experiment import io

    io.atomic_write_text(
        path,
        json.dumps(asdict(entry), indent=2, sort_keys=True) + "\n",
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
