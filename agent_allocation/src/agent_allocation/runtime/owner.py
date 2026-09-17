"""Process identity for leases and the single-dispatcher claim: ``host|pid|process start time|nonce``.

``owner_is_dead`` is True only when the owner is provably gone: recorded on this host and its pid no longer exists or
now belongs to a process with a different start time (pid reuse). An owner on another host is never presumed dead;
callers fall back to heartbeat staleness (``scheduler.lease_stale_seconds``). Config keys consumed: none.
"""
from __future__ import annotations

import os
import socket
import uuid


def proc_start_time(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            fields = handle.read().rsplit(b")", 1)[1].split()
        return fields[19].decode()
    except (OSError, IndexError):
        return None


def make_owner_id() -> str:
    pid = os.getpid()
    return f"{socket.gethostname()}|{pid}|{proc_start_time(pid)}|{uuid.uuid4().hex[:12]}"


def owner_host(owner: str | None) -> str | None:
    return None if not owner else owner.split("|", 1)[0]


def owner_is_dead(owner: str | None) -> bool:
    if not owner:
        return True
    try:
        host, pid_s, start, _ = owner.split("|", 3)
        pid = int(pid_s)
    except ValueError:
        return False
    if host != socket.gethostname():
        return False
    current = proc_start_time(pid)
    return current is None or current != start
