"""Durable JSON/JSONL I/O and run/cell path helpers.

Completed cell artifacts are publication data, so their writes need stronger semantics
than ``Path.write_text``.  The helpers below write a same-directory temporary file,
``fsync`` it, and atomically replace the destination.  Same-directory replacement is
important on the shared filesystem: it avoids both cross-filesystem rename failures and
readers observing a truncated artifact after preemption.

``append_jsonl`` remains available for the runner's per-question progress journal.  It
serializes appenders with ``flock`` and fsyncs every complete line.  Before a cell is
declared complete, the runner rewrites that journal with :func:`write_jsonl`, giving the
published ``results.jsonl`` the stronger atomic-replacement guarantee too.
"""

from __future__ import annotations

import json
import hashlib
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import fcntl

from agents_scaling.config import DEFAULT_RESULTS_ROOT


def results_root() -> Path:
    return Path(os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))


def run_dir(run_id: str) -> Path:
    d = results_root() / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def cell_dir(run_id: str, cell_id: str) -> Path:
    d = run_dir(run_id) / "cells" / cell_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_jsonl(path: str | os.PathLike, record: dict[str, Any]) -> None:
    """Durably append one complete JSON record.

    A hard kill can still leave a partial final line on some filesystems.  Completion
    repair deliberately treats that tail as invalid and removes it with an atomic
    canonical rewrite.  The advisory file lock prevents cooperating writers from
    interleaving bytes; the cell-level lock normally makes this uncontended.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written == 0:
                raise OSError("zero-byte write while appending JSONL record")
            view = view[written:]
        os.fsync(fd)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def write_json(path: str | os.PathLike, record: dict[str, Any]) -> None:
    """Atomically and durably write one JSON object."""
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(path, payload)


def write_jsonl(
    path: str | os.PathLike, records: Iterable[dict[str, Any]]
) -> None:
    """Atomically and durably replace a JSONL file with ``records``."""
    payload = "".join(
        json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
        for record in records
    )
    atomic_write_text(path, payload)


def atomic_write_text(path: str | os.PathLike, payload: str) -> None:
    """Write ``payload`` via a same-directory temp file and atomic replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, existing_mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except BaseException:
        # ``fdopen`` owns fd after it succeeds.  If it failed before taking ownership,
        # closing an already-closed descriptor is harmlessly ignored here.
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


# Private compatibility for callers written while the durable-I/O work was in flight.
_atomic_write_text = atomic_write_text


def remove_file(path: str | os.PathLike) -> None:
    """Remove a file and durably record the directory-entry change."""
    target = Path(path)
    try:
        target.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(target.parent)


def move_file(source: str | os.PathLike, destination: str | os.PathLike) -> None:
    """Atomically move one file and durably record both directory-entry changes."""
    src = Path(source)
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)
    _fsync_directory(dst.parent)
    if src.parent != dst.parent:
        _fsync_directory(src.parent)


def _fsync_directory(directory: Path) -> None:
    """Best-effort fsync of a directory after replace/unlink.

    Some network filesystems reject directory fsync even though atomic rename works; do
    not turn an otherwise successful artifact write into a failed experiment there.
    """
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def git_commit() -> str | None:
    """Return a code-version identity that changes for dirty source edits too.

    A bare ``git rev-parse HEAD`` is insufficient for failure gating in an editable
    research checkout: a deterministic configuration failure would remain permanently
    blocked while its fix was present but not yet committed.  Hash the executable source
    and prompt/config surfaces, retaining the commit prefix for human traceability.
    """
    try:
        repo = Path(__file__).resolve().parents[3]
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        digest = hashlib.sha256()
        versioned_roots = (repo / "src", repo / "configs" / "prompts")
        paths = [repo / "pyproject.toml"]
        for root in versioned_roots:
            if root.exists():
                paths.extend(
                    path
                    for path in root.rglob("*")
                    if path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix != ".pyc"
                )
        for path in sorted(paths, key=lambda item: item.relative_to(repo).as_posix()):
            relative = path.relative_to(repo).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            payload = path.read_bytes()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return f"{commit}+source.{digest.hexdigest()[:16]}"
    except Exception:
        return None
