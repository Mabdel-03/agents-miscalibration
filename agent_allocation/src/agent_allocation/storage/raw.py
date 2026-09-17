"""Immutable per-attempt raw responses ``raw/{shard}/{attempt}.json.gz`` (contract §2 "02", §3).

Write path: canonical JSON -> gzip with ``mtime=0`` (the same outcome always yields the same bytes and sha256) ->
same-directory temp file -> fsync -> ``os.link`` (first writer wins) -> directory fsync. A second write for the same
attempt is accepted only when byte-identical (a duplicate completion notification); different bytes raise
:class:`RawArtifactConflict`. Raw artifacts are never overwritten or deleted.

Payload keys: ``attempt_id, node_id, request_body, response_status, response_headers, response_body_text`` plus the
additive key ``result`` = ``dump_record(outcome.result)`` exactly as the client classified it (with
``raw_response_path``/``raw_response_sha256`` still ``None``). Recovery uses it to re-ingest a response that was
persisted before its database commit without re-sending or re-classifying (request filed in docs/requests/02.md).

Config keys consumed: none.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from agent_allocation.contracts import GenerationOutcome, GenerationResult, dump_record, load_record, to_jsonable
from agent_allocation.paths import RunPaths
from agent_allocation.storage._atomic import write_once

_SECRET_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key"})


class RawArtifactConflict(RuntimeError):
    """An artifact for this attempt already exists with different bytes."""


def encode_raw_payload(attempt_id: str, outcome: GenerationOutcome) -> bytes:
    result = outcome.result
    if result.attempt_id != attempt_id:
        raise ValueError(f"outcome belongs to attempt {result.attempt_id!r}, not {attempt_id!r}")
    body = outcome.response_body
    if isinstance(body, (bytes, bytearray, memoryview)):
        text = bytes(body).decode("utf-8", errors="backslashreplace")
    else:
        text = "" if body is None else str(body)
    headers = {str(k).lower(): ("[redacted]" if str(k).lower() in _SECRET_HEADERS else str(v))
               for k, v in dict(outcome.response_headers or {}).items()}
    payload = {
        "attempt_id": attempt_id,
        "node_id": result.node_id,
        "request_body": to_jsonable(dict(outcome.request_body or {})),
        "response_status": result.http_status,
        "response_headers": headers,
        "response_body_text": text,
        "result": dump_record(result),
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return gzip.compress(data, compresslevel=6, mtime=0)


def write_raw_response(paths: RunPaths, attempt_id: str, outcome: GenerationOutcome) -> tuple[str, str]:
    """Atomically persist one attempt's raw exchange; returns (run-dir-relative path, sha256 of the file bytes)."""
    data = encode_raw_payload(attempt_id, outcome)
    digest = hashlib.sha256(data).hexdigest()
    final = paths.raw_response(attempt_id)
    if not write_once(final, data):
        existing = hashlib.sha256(final.read_bytes()).hexdigest()
        if existing != digest:
            raise RawArtifactConflict(f"{final} already holds different bytes for attempt {attempt_id}")
    return paths.relative(final), digest


def read_raw_response(paths: RunPaths, relative_path: str, *, expected_sha256: str | None = None) -> dict[str, Any]:
    path = paths.run_dir / relative_path
    data = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise RawArtifactConflict(f"{path} does not match its recorded sha256")
    return json.loads(gzip.decompress(data).decode("utf-8"))


def result_from_raw(payload: Mapping[str, Any], *, relative_path: str, sha256: str) -> GenerationResult | None:
    """The typed result stored in a raw artifact, with its raw pointer filled in (``None`` for foreign files)."""
    record = payload.get("result")
    if not isinstance(record, Mapping):
        return None
    result: GenerationResult = load_record(GenerationResult, record)
    return replace(result, raw_response_path=relative_path, raw_response_sha256=sha256)


def raw_root(paths: RunPaths) -> Path:
    """``raw/`` resolved through RunPaths (the shard directory's parent), never joined by hand."""
    return paths.raw_response("00").parent.parent


def iter_raw_files(paths: RunPaths):
    root = raw_root(paths)
    if not root.is_dir():
        return
    for shard_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for file in sorted(shard_dir.glob("*.json.gz")):
            yield file


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
