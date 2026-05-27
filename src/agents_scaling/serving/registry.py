"""Server registry: a file-based directory of which vLLM server serves which model.

A serving SLURM job, once its vLLM server is up, writes a small JSON file recording its
``node:port`` and the model it serves. Cell jobs (the HTTP clients) discover their
server by reading the registry — no service discovery infrastructure needed, just a
shared filesystem (scratch), which is exactly what we have.

Layout: ``<run_root>/servers/<model_size>.json``
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


def _entry_path(run_root: str | os.PathLike, model_size: str) -> Path:
    return servers_dir(run_root) / f"{model_size}.json"


def register_server(run_root: str | os.PathLike, model_size: str, hf_id: str, port: int) -> ServerEntry:
    """Called by the serving job after vLLM is healthy. Records the current node + port."""
    entry = ServerEntry(
        model_size=model_size,
        hf_id=hf_id,
        host=socket.gethostname(),
        port=port,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        started_at=time.time(),
    )
    path = _entry_path(run_root, model_size)
    # Atomic write so a reader never sees a half-written file.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(entry), indent=2))
    tmp.replace(path)
    return entry


def lookup_server(run_root: str | os.PathLike, model_size: str) -> ServerEntry | None:
    path = _entry_path(run_root, model_size)
    if not path.exists():
        return None
    return ServerEntry(**json.loads(path.read_text()))


def wait_for_server(
    run_root: str | os.PathLike, model_size: str, timeout_s: float = 1800.0, poll_s: float = 5.0
) -> ServerEntry:
    """Block until the serving job has registered its endpoint (cell jobs call this)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        entry = lookup_server(run_root, model_size)
        if entry is not None:
            return entry
        time.sleep(poll_s)
    raise TimeoutError(
        f"no server registered for model_size={model_size!r} within {timeout_s}s "
        f"(looked in {servers_dir(run_root)})"
    )
