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

import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ServerEntry:
    model_size: str
    hf_id: str
    host: str
    port: int
    slurm_job_id: str | None = None
    started_at: float = 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def servers_dir(run_root: str | os.PathLike) -> Path:
    d = Path(run_root) / "servers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _size_dir(run_root: str | os.PathLike, model_size: str) -> Path:
    d = servers_dir(run_root) / model_size
    d.mkdir(parents=True, exist_ok=True)
    return d


def register_server(run_root: str | os.PathLike, model_size: str, hf_id: str, port: int) -> ServerEntry:
    """Called by a serving job after its vLLM server is healthy.

    Writes a per-server file under ``servers/<size>/<host>_<port>.json``. Multiple
    servers for the same size coexist (one file each); re-registration of the same
    host:port just overwrites its own file.
    """
    host = socket.gethostname()
    entry = ServerEntry(
        model_size=model_size,
        hf_id=hf_id,
        host=host,
        port=port,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        started_at=time.time(),
    )
    path = _size_dir(run_root, model_size) / f"{host}_{port}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(entry), indent=2))
    tmp.replace(path)
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


def lookup_server(
    run_root: str | os.PathLike, model_size: str, shard: int = 0
) -> ServerEntry | None:
    """Pick one endpoint for ``model_size``, round-robined by ``shard`` (e.g. the SLURM
    array task id) so cells spread across all available servers for that size."""
    servers = list_servers(run_root, model_size)
    if not servers:
        return None
    return servers[shard % len(servers)]


def wait_for_server(
    run_root: str | os.PathLike,
    model_size: str,
    shard: int = 0,
    timeout_s: float = 3600.0,
    poll_s: float = 5.0,
) -> ServerEntry:
    """Block until at least one server has registered for ``model_size`` (cells call this)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        entry = lookup_server(run_root, model_size, shard=shard)
        if entry is not None:
            return entry
        time.sleep(poll_s)
    raise TimeoutError(
        f"no server registered for model_size={model_size!r} within {timeout_s}s "
        f"(looked in {servers_dir(run_root)})"
    )
