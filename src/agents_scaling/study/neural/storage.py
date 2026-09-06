"""Chunked activation store: ``<run_root>/neural/<stage>/<shard>.cNNNNN.{npz,jsonl}`` (N1).

Spec §8.13: "Store selected sparse layer/token tensors in chunked files keyed by immutable
requests; do not keep every layer/token merely because hooks are available" and the
``ActivationRow`` ledger (StateSnapshot_id, consumer_revision, condition, child_slot,
nonce_hash, block, anchor_kind, structural_span, token_offset, channel,
generated_token_count, missingness, tensor_hash, measurement_cost,
operationally_available_at_checkpoint).  Brief: "Store fp16 vectors in chunked
``.npz``/parquet files keyed by (request_id | report_id, block, anchor) under
``<run_root>/neural/``, with an ActivationRow metadata jsonl".

Layout.  A *stage* (``native`` | ``report``) holds one directory; a *shard* (one Slurm task)
writes chunks ``<shard>.c00000.npz`` + ``<shard>.c00000.jsonl``.  The ``.npz`` carries
``vectors`` (fp16 ``[m, hidden]``) and ``keys`` (``"<snapshot>|<block>|<anchor>"`` per row);
the ``.jsonl`` carries one ActivationRow per (snapshot, block, anchor) — present vectors
point at their row through ``vector_index``, missing anchors (``NOT_REACHED`` …) carry
``vector_index = -1`` and **no vector** (never a zero vector).  Writes are atomic
(same-directory temp file + ``os.replace`` + directory fsync; the ``.npz`` lands before the
``.jsonl``, so a chunk is complete iff its ``.jsonl`` exists and agrees with the ``.npz``).
Resume = ``existing_keys()`` over the complete chunks of a stage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import re
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

NEURAL_DIR = "neural"
STAGES: tuple[str, ...] = ("native", "report")
CHUNK_RE = re.compile(r"^(?P<shard>[A-Za-z0-9_.\-]+)\.c(?P<index>\d{5})\.jsonl$")
DEFAULT_CHUNK_ROWS = 512
STORAGE_SCHEMA_VERSION = 1

#: Spec §8.13 ActivationRow fields, in ledger order.
ACTIVATION_ROW_FIELDS: tuple[str, ...] = (
    "StateSnapshot_id",
    "consumer_revision",
    "condition",
    "child_slot",
    "nonce_hash",
    "block",
    "anchor_kind",
    "structural_span",
    "token_offset",
    "channel",
    "generated_token_count",
    "missingness",
    "tensor_hash",
    "measurement_cost",
    "operationally_available_at_checkpoint",
)


class StorageError(RuntimeError):
    """A chunk on disk is inconsistent (never silently skipped)."""


def tensor_hash(vector: np.ndarray) -> str:
    """sha256 of the fp16 little-endian bytes of a vector."""
    arr = np.ascontiguousarray(np.asarray(vector, dtype=np.float16).astype("<f2"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class ActivationRow:
    """One (snapshot, block, anchor) ledger line (spec §8.13) plus harness join keys."""

    StateSnapshot_id: str
    consumer_revision: str
    condition: str
    child_slot: int | None
    nonce_hash: str
    block: int
    anchor_kind: str
    structural_span: str
    token_offset: int | None
    channel: str
    generated_token_count: int
    missingness: str | None
    tensor_hash: str | None
    measurement_cost: dict[str, Any]
    operationally_available_at_checkpoint: bool
    # harness join keys / provenance (not part of the spec ledger, kept alongside)
    stage: str
    sequence_id: str  # request_id | report_id
    source_id: str | None = None
    method: str | None = None
    role: str | None = None
    phase: str | None = None
    cell_id: str | None = None
    episode_id: str | None = None
    checkpoint: str | None = None
    hidden_size: int | None = None
    sequence_tokens: int | None = None
    hook_convention: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    captured_at: float = 0.0

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.StateSnapshot_id, int(self.block), self.anchor_kind)

    @property
    def key_str(self) -> str:
        return f"{self.StateSnapshot_id}|{int(self.block)}|{self.anchor_kind}"

    @property
    def present(self) -> bool:
        return self.missingness is None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActivationRow":
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in names and k != "vector_index"}
        return cls(**kwargs)


def key_str(snapshot_id: str, block: int, anchor_kind: str) -> str:
    return f"{snapshot_id}|{int(block)}|{anchor_kind}"


def parse_key(key: str) -> tuple[str, int, str]:
    snapshot_id, block, anchor = key.rsplit("|", 2)
    return snapshot_id, int(block), anchor


def stage_dir(run_root: str | os.PathLike, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    return Path(run_root) / NEURAL_DIR / stage


# --------------------------------------------------------------------------- atomic writes


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------- chunks


def chunk_paths(directory: Path, shard: str, index: int) -> tuple[Path, Path]:
    # string concatenation on purpose: Path.with_suffix would eat the ".cNNNNN" part
    return directory / f"{shard}.c{index:05d}.npz", directory / f"{shard}.c{index:05d}.jsonl"


def list_chunks(directory: Path, shard: str | None = None) -> list[tuple[str, int, Path, Path]]:
    """Complete chunks ``(shard, index, npz, jsonl)`` of a stage directory, sorted."""
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.iterdir()):
        m = CHUNK_RE.match(path.name)
        if not m or (shard is not None and m.group("shard") != shard):
            continue
        npz = path.parent / (path.name[: -len(".jsonl")] + ".npz")
        if not npz.is_file():
            raise StorageError(f"{path} has no matching .npz (incomplete chunk)")
        out.append((m.group("shard"), int(m.group("index")), npz, path))
    return out


def next_chunk_index(directory: Path, shard: str) -> int:
    """First unused chunk index (also skips orphan ``.npz`` files of a crashed writer)."""
    used = {index for _, index, _, _ in list_chunks(directory, shard)}
    if directory.is_dir():
        for path in directory.glob(f"{shard}.c*.npz"):
            m = re.match(rf"^{re.escape(shard)}\.c(\d{{5}})\.npz$", path.name)
            if m:
                used.add(int(m.group(1)))
    return (max(used) + 1) if used else 0


def read_chunk(npz_path: Path, jsonl_path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """``(vectors fp16 [m, d], keys [m], rows)``; verifies the two files agree."""
    with np.load(npz_path, allow_pickle=False) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float16)
        keys = np.asarray(data["keys"]).astype(str)
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    present = [r for r in rows if r.get("vector_index", -1) >= 0]
    if len(present) != vectors.shape[0] or len(keys) != vectors.shape[0]:
        raise StorageError(f"{jsonl_path}: {len(present)} present rows vs {vectors.shape[0]} vectors")
    for r in present:
        i = int(r["vector_index"])
        expected = key_str(r["StateSnapshot_id"], r["block"], r["anchor_kind"])
        if i >= len(keys) or keys[i] != expected:
            raise StorageError(f"{jsonl_path}: row {expected} points at vector {i} keyed {keys[i] if i < len(keys) else '?'}")
    return vectors, keys, rows


def iter_rows(run_root: str | os.PathLike, stage: str, shard: str | None = None) -> Iterator[tuple[Path, dict[str, Any]]]:
    directory = stage_dir(run_root, stage)
    for _, _, npz, jsonl in list_chunks(directory, shard):
        _, _, rows = read_chunk(npz, jsonl)
        for row in rows:
            yield jsonl, row


def existing_keys(run_root: str | os.PathLike, stage: str, shard: str | None = None) -> set[tuple[str, int, str]]:
    """Keys ``(snapshot_id, block, anchor_kind)`` already committed (present or missing rows)."""
    keys: set[tuple[str, int, str]] = set()
    for _, row in iter_rows(run_root, stage, shard):
        keys.add((str(row["StateSnapshot_id"]), int(row["block"]), str(row["anchor_kind"])))
    return keys


# --------------------------------------------------------------------------- writer


class ShardWriter:
    """Buffer rows/vectors and flush them as atomic chunks (``chunk_rows`` present vectors
    or ``flush()``).  A row without a vector must carry ``missingness``; a row with a vector
    must not.  ``tensor_hash`` is filled in from the vector when absent."""

    def __init__(self, run_root: str | os.PathLike, stage: str, shard: str, *, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> None:
        if not re.match(r"^[A-Za-z0-9_.\-]+$", shard) or ".c" in shard:
            raise ValueError(f"shard name {shard!r} must be [A-Za-z0-9_.-] and must not contain '.c'")
        self.run_root = Path(run_root)
        self.stage = stage
        self.shard = shard
        self.directory = stage_dir(run_root, stage)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chunk_rows = int(chunk_rows)
        self._index = next_chunk_index(self.directory, shard)
        self._rows: list[dict[str, Any]] = []
        self._vectors: list[np.ndarray] = []
        self._keys: list[str] = []
        self._buffered_keys: set[str] = set()
        self.rows_written = 0
        self.vectors_written = 0
        self.chunks_written: list[Path] = []

    def add(self, row: ActivationRow, vector: np.ndarray | None) -> None:
        if row.key_str in self._buffered_keys:
            raise StorageError(f"duplicate key in one chunk: {row.key_str}")
        if vector is None:
            if row.missingness is None:
                raise StorageError(f"{row.key_str}: a row without a vector must carry missingness")
            record = {**row.to_dict(), "vector_index": -1}
        else:
            if row.missingness is not None:
                raise StorageError(f"{row.key_str}: a missing anchor must not carry a vector")
            arr = np.asarray(vector, dtype=np.float16).reshape(-1)
            if arr.size == 0 or not np.isfinite(arr.astype(np.float32)).all():
                raise StorageError(f"{row.key_str}: vector is empty or non-finite")
            if self._vectors and arr.shape != self._vectors[0].shape:
                raise StorageError(f"{row.key_str}: hidden size {arr.shape} != {self._vectors[0].shape}")
            digest = tensor_hash(arr)
            if row.tensor_hash is not None and row.tensor_hash != digest:
                raise StorageError(f"{row.key_str}: tensor_hash disagrees with the vector")
            record = {**row.to_dict(), "tensor_hash": digest, "vector_index": len(self._vectors)}
            self._vectors.append(arr)
            self._keys.append(row.key_str)
        if not record.get("captured_at"):
            record["captured_at"] = time.time()
        self._rows.append(record)
        self._buffered_keys.add(row.key_str)
        if len(self._vectors) >= self.chunk_rows:
            self.flush()

    def flush(self) -> Path | None:
        if not self._rows:
            return None
        npz_path, jsonl_path = chunk_paths(self.directory, self.shard, self._index)
        if npz_path.exists() or jsonl_path.exists():
            raise StorageError(f"chunk {npz_path} already exists (concurrent writer on the same shard?)")
        vectors = np.stack(self._vectors).astype(np.float16) if self._vectors else np.zeros((0, 0), dtype=np.float16)
        buffer = io.BytesIO()
        np.savez(buffer, vectors=vectors, keys=np.asarray(self._keys, dtype=str), schema_version=np.int64(STORAGE_SCHEMA_VERSION))
        atomic_write_bytes(npz_path, buffer.getvalue())
        payload = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n" for r in self._rows)
        atomic_write_bytes(jsonl_path, payload.encode("utf-8"))
        self.rows_written += len(self._rows)
        self.vectors_written += len(self._vectors)
        self.chunks_written.append(jsonl_path)
        self._index += 1
        self._rows, self._vectors, self._keys, self._buffered_keys = [], [], [], set()
        return jsonl_path

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        if exc[0] is None:
            self.flush()


# --------------------------------------------------------------------------- loader


def load_stage(
    run_root: str | os.PathLike,
    stage: str,
    *,
    blocks: Iterable[int] | None = None,
    anchor_kinds: Iterable[str] | None = None,
    shard: str | None = None,
    include_missing: bool = True,
) -> tuple[np.ndarray, Any]:
    """``(matrix, frame)``: fp16 ``[n_present, hidden]`` plus a pandas frame of every
    selected row (``vector_index`` = row in ``matrix``, ``-1`` for missing anchors).
    Duplicate keys across chunks raise (the store is append-only by key)."""
    directory = stage_dir(run_root, stage)
    want_blocks = None if blocks is None else {int(b) for b in blocks}
    want_anchors = None if anchor_kinds is None else set(anchor_kinds)
    matrices: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    offset = 0
    for _, _, npz, jsonl in list_chunks(directory, shard):
        vectors, _, rows = read_chunk(npz, jsonl)
        keep = np.zeros(vectors.shape[0], dtype=bool)
        local: list[dict[str, Any]] = []
        for row in rows:
            if want_blocks is not None and int(row["block"]) not in want_blocks:
                continue
            if want_anchors is not None and row["anchor_kind"] not in want_anchors:
                continue
            k = key_str(row["StateSnapshot_id"], row["block"], row["anchor_kind"])
            if k in seen:
                raise StorageError(f"duplicate key {k} across chunks of {directory}")
            seen.add(k)
            vi = int(row.get("vector_index", -1))
            if vi < 0 and not include_missing:
                continue
            if vi >= 0:
                keep[vi] = True
            local.append({**row, "chunk": jsonl.name})
        if keep.any():
            remap = -np.ones(vectors.shape[0], dtype=np.int64)
            remap[keep] = np.arange(int(keep.sum())) + offset
            matrices.append(vectors[keep])
            offset += int(keep.sum())
        else:
            remap = -np.ones(vectors.shape[0], dtype=np.int64)
        for r in local:
            vi = int(r.get("vector_index", -1))
            r["vector_index"] = int(remap[vi]) if vi >= 0 else -1
            records.append(r)
    matrix = np.concatenate(matrices, axis=0) if matrices else np.zeros((0, 0), dtype=np.float16)
    try:
        import pandas as pd

        frame = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=list(ACTIVATION_ROW_FIELDS) + ["vector_index"])
        ordered = [c for c in ACTIVATION_ROW_FIELDS if c in frame.columns]
        frame = frame[ordered + [c for c in frame.columns if c not in ordered]]  # ledger fields first
    except ImportError:  # pragma: no cover - pandas is in both envs
        frame = records
    return matrix, frame


__all__ = [
    "ACTIVATION_ROW_FIELDS",
    "DEFAULT_CHUNK_ROWS",
    "NEURAL_DIR",
    "STAGES",
    "STORAGE_SCHEMA_VERSION",
    "ActivationRow",
    "ShardWriter",
    "StorageError",
    "atomic_write_bytes",
    "atomic_write_json",
    "chunk_paths",
    "existing_keys",
    "iter_rows",
    "key_str",
    "list_chunks",
    "load_stage",
    "next_chunk_index",
    "parse_key",
    "read_chunk",
    "stage_dir",
    "tensor_hash",
]
