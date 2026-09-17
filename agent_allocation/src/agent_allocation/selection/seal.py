"""Seal selections before any grade join (MASTER_SPEC §9, §12; docs/00_contract.md §3).

``seal_selections`` writes ``paths.selections(partition)`` -- one ``dump_record(SelectionResult)`` per line in
canonical JSON, sorted by ``policy_id`` -- and then ``paths.selection_seal(partition)`` =
``{"sha256", "count", "sealed_at"}``, both atomically (tmp + fsync + rename). ``sha256`` is the digest of the exact
JSONL bytes and is returned.

Sealing is append-only: a partition that already has a seal may be re-sealed only with a superset of its policies,
where every previously sealed policy is semantically unchanged (all fields but ``cpu_seconds``); the previously
sealed line is kept byte-for-byte. A changed or removed sealed selection raises ``SealError`` -- a selection that
was sealed (and may already have been joined to grades) is never rewritten. Re-sealing identical content is
idempotent and keeps the original ``sealed_at``.

``load_sealed_selections`` is the only way to obtain a ``SealedSelections``; it verifies the seal digest and count
against the file bytes. ``metrics`` joins grades only through a verified ``SealedSelections``.

Config keys consumed: none.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

from agent_allocation.contracts import SelectionResult, SelectionStatus, canonical_json, dump_record, load_record
from agent_allocation.paths import RunPaths

__all__ = ["SealError", "SealInfo", "SealedSelections", "load_sealed_selections", "read_seal", "seal_selections"]

_VERIFIED = object()


class SealError(RuntimeError):
    """A sealed selection file is missing, tampered with, or would be rewritten."""


@dataclass(frozen=True)
class SealInfo:
    sha256: str
    count: int
    sealed_at: str


@dataclass(frozen=True)
class SealedSelections:
    partition: str
    seal: SealInfo
    by_policy: Mapping[str, SelectionResult]
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _VERIFIED:
            raise SealError("SealedSelections is created only by load_sealed_selections (verified seal)")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _line(result: SelectionResult) -> bytes:
    return canonical_json(dump_record(result)).encode("utf-8") + b"\n"


def _semantic(result: SelectionResult) -> dict[str, object]:
    return {f.name: getattr(result, f.name) for f in fields(result) if f.name != "cpu_seconds"}


def _check_partition(partition: str) -> None:
    if not isinstance(partition, str) or not partition or "/" in partition or partition.startswith("."):
        raise ValueError(f"invalid partition name {partition!r}")


def _validate(partition: str, results: Sequence[SelectionResult]) -> dict[str, SelectionResult]:
    by_policy: dict[str, SelectionResult] = {}
    for result in results:
        if not isinstance(result, SelectionResult):
            raise TypeError(f"seal_selections expects SelectionResult, got {type(result).__name__}")
        if not result.policy_id.startswith(f"{partition}/"):
            raise ValueError(f"policy {result.policy_id!r} does not belong to partition {partition!r}")
        SelectionStatus(result.status)
        if result.policy_id in by_policy:
            raise ValueError(f"duplicate selection for policy {result.policy_id!r}")
        needs_candidate = result.status in (SelectionStatus.SELECTED.value, SelectionStatus.SINGLE_CANDIDATE.value)
        if needs_candidate != (result.selected_candidate_id is not None):
            raise ValueError(f"{result.policy_id}: status {result.status} with selected {result.selected_candidate_id}")
        by_policy[result.policy_id] = result
    return by_policy


def read_seal(paths: RunPaths, partition: str) -> SealInfo | None:
    _check_partition(partition)
    path = paths.selection_seal(partition)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"sha256", "count", "sealed_at"}:
        raise SealError(f"{path}: unexpected seal keys {sorted(data)}")
    return SealInfo(sha256=str(data["sha256"]), count=int(data["count"]), sealed_at=str(data["sealed_at"]))


def _read_verified_lines(paths: RunPaths, partition: str, seal: SealInfo) -> list[bytes]:
    path = paths.selections(partition)
    if not path.exists():
        raise SealError(f"{path}: sealed selections file is missing")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != seal.sha256:
        raise SealError(f"{path}: sha256 does not match its seal")
    lines = raw.splitlines(keepends=True)
    if len(lines) != seal.count:
        raise SealError(f"{path}: {len(lines)} lines but the seal says {seal.count}")
    return lines


def seal_selections(paths: RunPaths, partition: str, results: Sequence[SelectionResult]) -> str:
    _check_partition(partition)
    new = _validate(partition, results)
    if not new:
        raise ValueError(f"refusing to seal an empty selection set for {partition!r}: empty selector output is a "
                         "blocker, not a result")
    lines: dict[str, bytes] = {policy: _line(result) for policy, result in new.items()}
    previous = read_seal(paths, partition)
    if previous is not None:
        for raw in _read_verified_lines(paths, partition, previous):
            old = load_record(SelectionResult, json.loads(raw))
            if old.policy_id not in new:
                raise SealError(f"{partition}: sealed selection {old.policy_id} would be removed")
            if _semantic(old) != _semantic(new[old.policy_id]):
                raise SealError(f"{partition}: sealed selection {old.policy_id} would change")
            lines[old.policy_id] = raw
    body = b"".join(lines[policy] for policy in sorted(lines))
    digest = sha256(body).hexdigest()
    if previous is not None and previous.sha256 == digest and previous.count == len(lines):
        return digest
    _atomic_write(paths.selections(partition), body)
    seal = {"sha256": digest, "count": len(lines),
            "sealed_at": datetime.now(timezone.utc).isoformat(timespec="microseconds")}
    _atomic_write(paths.selection_seal(partition), (json.dumps(seal, sort_keys=True) + "\n").encode("utf-8"))
    return digest


def load_sealed_selections(paths: RunPaths, partition: str) -> SealedSelections:
    seal = read_seal(paths, partition)
    if seal is None:
        raise SealError(f"partition {partition!r} has no selection seal: grades cannot be joined")
    by_policy: dict[str, SelectionResult] = {}
    for raw in _read_verified_lines(paths, partition, seal):
        result = load_record(SelectionResult, json.loads(raw))
        if result.policy_id in by_policy:
            raise SealError(f"{partition}: duplicate sealed policy {result.policy_id}")
        by_policy[result.policy_id] = result
    return SealedSelections(partition=partition, seal=seal, by_policy=by_policy, _token=_VERIFIED)
