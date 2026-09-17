"""Atomic, durable file primitives shared by the raw store, parsed records, status and backups.

``write_once`` is first-writer-wins: temp file in the destination directory -> fsync -> ``os.link`` (atomic
create-or-fail, works on NFSv4) -> directory fsync. ``replace_atomic`` is last-writer-wins for mutable snapshots
(status.json, the SQLite backup).
"""
from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - some filesystems refuse directory opens
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - directory fsync unsupported
        pass
    finally:
        os.close(fd)


def _write_temp(directory: Path, prefix: str, data: bytes) -> Path:
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise
    os.close(fd)
    return Path(name)


def write_once(final: Path, data: bytes) -> bool:
    """Create ``final`` with ``data`` unless it exists. Returns True if this call created it.

    The caller decides what an existing file means (identical bytes = duplicate notification; different bytes =
    conflict). Never overwrites.
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        return False
    tmp = _write_temp(final.parent, f".{final.name}.", data)
    try:
        try:
            os.link(tmp, final)
        except FileExistsError:
            return False
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV):
                raise
            # Filesystem without hard links: exclusive create by rename-if-absent (single writer by contract).
            if final.exists():
                return False
            os.replace(tmp, final)
        fsync_dir(final.parent)
        return True
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def replace_atomic(final: Path, data: bytes) -> None:
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = _write_temp(final.parent, f".{final.name}.", data)
    try:
        os.replace(tmp, final)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    fsync_dir(final.parent)
