"""Seal selections in append-only batches before any grade join (MASTER_SPEC §9, §12; contract v2 D8, R07-1).

Layout (``RunPaths``): ``selections/{partition}/batch_NNNN.jsonl`` holds one ``dump_record(SelectionResult)`` per
line in canonical JSON, sorted by ``policy_id``; ``batch_NNNN.seal.json`` is
``{"sha256", "count", "sealed_at", "previous_seal_sha256"}`` where ``sha256`` is the digest of the batch file's exact
bytes and ``previous_seal_sha256`` is the ``sha256`` recorded by batch ``N-1`` (``null`` for batch 0). Both files
are written atomically (tmp + fsync + rename), the batch first.

``seal_selections(paths, partition, results)`` writes the *next* batch with only the results whose checkpoints are
not sealed yet, and returns that batch's ``sha256``; when every result is already sealed it writes nothing and
returns the latest seal's ``sha256`` (idempotent resume). A sealed batch is never rewritten. For a checkpoint that
is already sealed:

* the same selection (all fields but ``cpu_seconds``) is skipped;
* a sealed ``PAUSED_CHECK_ERROR`` row may be *resumed*: a non-paused selection over the same pool (equal
  ``pool_hash``, selector name and version) goes into the next batch and supersedes the pause -- a pause is the
  absence of a selection, repaired by fixing the public checker;
* a different pool (``pool_hash``) raises ``SealError``: the candidates changed after sealing;
* a different selection over the same pool is not re-sealed (the sealed row stays authoritative) and emits a
  ``SealDriftWarning`` -- e.g. an equivalence timeout that resolved differently on a re-run.

A batch file without its seal (a crash between the two renames) was never sealed and is replaced by the next call.
``load_sealed_selections`` is the only way to obtain a ``SealedSelections``: it verifies every batch digest, count,
chain link and the no-duplicate rule, and returns the effective row per policy (a resumed selection supersedes its
pause). ``metrics`` joins grades only through a verified ``SealedSelections``.

Config keys consumed: none.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

from agent_allocation.contracts import SelectionResult, SelectionStatus, canonical_json, dump_record, load_record
from agent_allocation.paths import RunPaths

__all__ = [
    "SealDriftWarning", "SealError", "SealInfo", "SealedSelections", "load_sealed_selections", "read_seals",
    "seal_selections",
]

_VERIFIED = object()
_SEAL_KEYS = frozenset({"sha256", "count", "sealed_at", "previous_seal_sha256"})
_BATCH_FILE = re.compile(r"^batch_(\d{4})\.(jsonl|seal\.json)$")


class SealError(RuntimeError):
    """A sealed selection batch is missing, tampered with, out of chain, or would be rewritten."""


class SealDriftWarning(UserWarning):
    """A re-run produced a different selection for an already sealed checkpoint; the sealed row stays."""


@dataclass(frozen=True)
class SealInfo:
    batch: int
    sha256: str
    count: int
    sealed_at: str
    previous_seal_sha256: str | None


@dataclass(frozen=True)
class SealedSelections:
    partition: str
    seals: tuple[SealInfo, ...]
    by_policy: Mapping[str, SelectionResult]           # effective sealed row per checkpoint
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _VERIFIED:
            raise SealError("SealedSelections is created only by load_sealed_selections (verified seals)")

    @property
    def seal(self) -> SealInfo:
        return self.seals[-1]


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


def _resumes_pause(old: SelectionResult, new: SelectionResult) -> bool:
    return (old.status is SelectionStatus.PAUSED_CHECK_ERROR and new.status is not SelectionStatus.PAUSED_CHECK_ERROR
            and old.pool_hash == new.pool_hash and old.selector_name == new.selector_name
            and old.selector_version == new.selector_version)


def _check_partition(partition: str) -> None:
    if not isinstance(partition, str) or not partition or "/" in partition or partition.startswith("."):
        raise ValueError(f"invalid partition name {partition!r}")


def _validate(partition: str, results: Sequence[SelectionResult]) -> dict[str, SelectionResult]:
    by_policy: dict[str, SelectionResult] = {}
    for result in results:
        if not isinstance(result, SelectionResult):
            raise TypeError(f"seal_selections expects SelectionResult, got {type(result).__name__}")
        if not isinstance(result.status, SelectionStatus):
            raise TypeError(f"{result.policy_id}: status must be a SelectionStatus, got {result.status!r}")
        if not result.policy_id.startswith(f"{partition}/"):
            raise ValueError(f"policy {result.policy_id!r} does not belong to partition {partition!r}")
        if result.policy_id in by_policy:
            raise ValueError(f"duplicate selection for policy {result.policy_id!r}")
        needs_candidate = result.status in (SelectionStatus.SELECTED, SelectionStatus.SINGLE_CANDIDATE)
        if needs_candidate != (result.selected_candidate_id is not None):
            raise ValueError(f"{result.policy_id}: status {result.status.value} with selected "
                             f"{result.selected_candidate_id!r}")
        if not result.pool_hash or not result.selector_name or not result.selector_version:
            raise ValueError(f"{result.policy_id}: pool hash, selector name and version are required")
        by_policy[result.policy_id] = result
    return by_policy


def _batch_indices(paths: RunPaths, partition: str) -> tuple[list[int], list[int]]:
    """(indices with a batch file, indices with a seal file), sorted."""
    directory = paths.selections_dir(partition)
    batches: set[int] = set()
    seals: set[int] = set()
    if directory.is_dir():
        for entry in directory.iterdir():
            match = _BATCH_FILE.match(entry.name)
            if match:
                (batches if match.group(2) == "jsonl" else seals).add(int(match.group(1)))
    return sorted(batches), sorted(seals)


def _read_seal_file(paths: RunPaths, partition: str, batch: int) -> SealInfo:
    path = paths.selection_seal(partition, batch)
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != _SEAL_KEYS:
        raise SealError(f"{path}: seal keys {sorted(data)} != {sorted(_SEAL_KEYS)}")
    previous = data["previous_seal_sha256"]
    return SealInfo(batch=batch, sha256=str(data["sha256"]), count=int(data["count"]), sealed_at=str(data["sealed_at"]),
                    previous_seal_sha256=None if previous is None else str(previous))


def _verified_chain(paths: RunPaths, partition: str) -> list[tuple[SealInfo, list[SelectionResult]]]:
    _check_partition(partition)
    batches, seals = _batch_indices(paths, partition)
    if seals != list(range(len(seals))):
        raise SealError(f"{partition}: seal batches {seals} are not contiguous from 0")
    stray = [b for b in batches if b not in seals]
    if any(b != len(seals) for b in stray):
        raise SealError(f"{partition}: unsealed batch files {stray} beyond the next batch index {len(seals)}")
    chain: list[tuple[SealInfo, list[SelectionResult]]] = []
    previous: str | None = None
    for batch in seals:
        info = _read_seal_file(paths, partition, batch)
        path = paths.selections(partition, batch)
        if not path.exists():
            raise SealError(f"{path}: sealed batch file is missing")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != info.sha256:
            raise SealError(f"{path}: sha256 does not match its seal")
        lines = raw.splitlines(keepends=True)
        if len(lines) != info.count or info.count == 0:
            raise SealError(f"{path}: {len(lines)} lines but the seal says {info.count}")
        if info.previous_seal_sha256 != previous:
            raise SealError(f"{path}: previous_seal_sha256 breaks the chain")
        chain.append((info, [load_record(SelectionResult, json.loads(line)) for line in lines]))
        previous = info.sha256
    return chain


def _effective(partition: str, chain: Sequence[tuple[SealInfo, list[SelectionResult]]]) -> dict[str, SelectionResult]:
    effective: dict[str, SelectionResult] = {}
    for info, rows in chain:
        seen: set[str] = set()
        for row in rows:
            if not row.policy_id.startswith(f"{partition}/") or row.policy_id in seen:
                raise SealError(f"{partition} batch {info.batch}: foreign or duplicate policy {row.policy_id}")
            seen.add(row.policy_id)
            earlier = effective.get(row.policy_id)
            if earlier is not None and not _resumes_pause(earlier, row):
                raise SealError(f"{partition} batch {info.batch}: policy {row.policy_id} was already sealed")
            effective[row.policy_id] = row
    return effective


def read_seals(paths: RunPaths, partition: str) -> tuple[SealInfo, ...]:
    """The verified seal chain of a partition (empty when nothing is sealed)."""
    return tuple(info for info, _ in _verified_chain(paths, partition))


def seal_selections(paths: RunPaths, partition: str, results: Sequence[SelectionResult]) -> str:
    _check_partition(partition)
    new = _validate(partition, results)
    if not new:
        raise ValueError(f"refusing to seal an empty selection set for {partition!r}: empty selector output is a "
                         "blocker, not a result")
    chain = _verified_chain(paths, partition)
    sealed = _effective(partition, chain)
    pending: dict[str, SelectionResult] = {}
    for policy, result in new.items():
        old = sealed.get(policy)
        if old is None or _resumes_pause(old, result):
            pending[policy] = result
        elif old.pool_hash != result.pool_hash:
            raise SealError(f"{policy}: pool changed after sealing ({old.pool_hash} -> {result.pool_hash})")
        elif _semantic(old) != _semantic(result):
            warnings.warn(f"{policy}: re-run selection differs from the sealed row; the sealed row stays",
                          SealDriftWarning, stacklevel=2)
    if not pending:
        return chain[-1][0].sha256
    batch = len(chain)
    body = b"".join(_line(pending[policy]) for policy in sorted(pending))
    digest = sha256(body).hexdigest()
    _atomic_write(paths.selections(partition, batch), body)
    seal = {"sha256": digest, "count": len(pending),
            "sealed_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "previous_seal_sha256": chain[-1][0].sha256 if chain else None}
    _atomic_write(paths.selection_seal(partition, batch), (json.dumps(seal, sort_keys=True) + "\n").encode("utf-8"))
    return digest


def load_sealed_selections(paths: RunPaths, partition: str) -> SealedSelections:
    chain = _verified_chain(paths, partition)
    if not chain:
        raise SealError(f"partition {partition!r} has no sealed selection batch: grades cannot be joined")
    return SealedSelections(partition=partition, seals=tuple(info for info, _ in chain),
                            by_policy=_effective(partition, chain), _token=_VERIFIED)
