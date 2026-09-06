"""Content-addressed request store ``<run_root>/requests/<id[:2]>/<id>.json`` (WP2).

Spec: §3.6 ("a second cell finding an existing request_id reads it (alias) and never
regenerates"; alias by exact equality of the identity), §6.2/§6.7 (bank aliasing is
automatic because the id is a pure function of the prompt, decoding, caps, pins and seed),
§10.4 ("duplicate submissions attach to the same committed request").  Architecture:
docs/study_v4/01_architecture.md §1.7, §2.1, §2.3; corrections P1-6 (one store, one write
path, no partial journals) and P1-7 (per-request files, ``O_EXCL`` first-writer-wins).

Write path: same-directory temp file → ``fsync`` → ``os.link(tmp, final)``.  A hard link
either creates the final name or raises ``FileExistsError`` atomically, so exactly one
committed record exists per id; a loser discards its own output and returns the committed
one (``was_new=False``).  Read path: strict JSON (duplicate keys and non-finite constants
rejected), the file name must equal the record's ``request_id`` and ``content_sha256`` must
verify — any violation is a :class:`ProtocolError`; a corrupt record is never silently
regenerated (architecture §2.1).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from agents_scaling.study.types import ProtocolError, RequestRecord, RequestSpec

_ID_RE = re.compile(r"^[0-9a-f]{64}$")
EventFn = Callable[[str, Mapping[str, Any]], None]


class Generator(Protocol):
    """What :meth:`RequestStore.get_or_generate` needs from a client (``VllmChatClient``)."""

    def generate(self, spec: RequestSpec, *, cell_id: str | None = None, **kwargs: Any) -> RequestRecord:
        ...


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name!r}")


def load_record_text(text: str) -> RequestRecord:
    """Strict parse + integrity check of one on-disk record (``ProtocolError`` on any defect)."""
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except ValueError as exc:
        raise ProtocolError(f"request record is not strict JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("request record is not a JSON object")
    try:
        record = RequestRecord.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"request record has an invalid shape: {exc}") from exc
    if record.content_sha256 is None:
        raise ProtocolError(f"request record {record.request_id} has no content_sha256")
    record.verify()
    return record


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RequestStore:
    """First-writer-wins store of committed :class:`RequestRecord` files (see module docstring)."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)

    # ---- layout ---------------------------------------------------------------------
    @staticmethod
    def _check_id(request_id: str) -> str:
        if not isinstance(request_id, str) or not _ID_RE.fullmatch(request_id):
            raise ProtocolError(f"request_id must be 64 lowercase hex characters, got {request_id!r}")
        return request_id

    def path(self, request_id: str) -> Path:
        """``root/<id[:2]>/<id>.json``."""
        request_id = self._check_id(request_id)
        return self.root / request_id[:2] / f"{request_id}.json"

    def exists(self, request_id: str) -> bool:
        return self.path(request_id).is_file()

    # ---- read -------------------------------------------------------------------------
    def get(self, request_id: str) -> RequestRecord | None:
        """The committed record or ``None``; a corrupt/mismatched file raises ``ProtocolError``."""
        path = self.path(request_id)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ProtocolError(f"cannot read request record {path}: {exc}") from exc
        try:
            record = load_record_text(text)
        except ProtocolError as exc:
            raise ProtocolError(f"{path}: {exc}") from exc
        if record.request_id != request_id:
            raise ProtocolError(f"{path} holds request_id {record.request_id!r} (file name disagrees)")
        return record

    # ---- write ------------------------------------------------------------------------
    def publish(self, record: RequestRecord) -> tuple[RequestRecord, bool]:
        """Commit ``record`` unless an id-equal record exists → ``(committed, was_new)``.

        The record must carry a verifying ``content_sha256`` (``with_content_sha256()``).
        On a lost race the committed record must describe the same identity and prompt as
        ours (a hash collision or a corrupt committed file is a ``ProtocolError``).
        """
        request_id = self._check_id(record.request_id)
        if record.content_sha256 is None:
            raise ProtocolError(f"refusing to publish {request_id}: content_sha256 is unset")
        record.verify()
        if record.identity.get("study_id") is None:
            raise ProtocolError(f"refusing to publish {request_id}: identity block is incomplete")
        final = self.path(request_id)
        final.parent.mkdir(parents=True, exist_ok=True)
        payload = record.to_json()
        fd, tmp_name = tempfile.mkstemp(prefix=f".{request_id[:12]}.", suffix=".tmp", dir=final.parent)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(tmp, final)
            except FileExistsError:
                committed = self.get(request_id)
                if committed is None:  # pragma: no cover - the link target vanished under us
                    raise ProtocolError(f"{final} existed during publish but cannot be read back")
                self._check_same_request(record, committed)
                return committed, False
            _fsync_dir(final.parent)
            return record, True
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _check_same_request(ours: RequestRecord, committed: RequestRecord) -> None:
        if committed.identity != ours.identity:
            raise ProtocolError(
                f"committed record {committed.request_id} has a different identity block than the "
                "freshly generated one (request_id collision or corrupt store)"
            )
        if committed.engine_seed != ours.engine_seed or list(committed.messages) != list(ours.messages):
            raise ProtocolError(
                f"committed record {committed.request_id} disagrees on engine_seed/messages with the "
                "freshly generated one"
            )

    # ---- the only entry point cells use --------------------------------------------------
    def get_or_generate(
        self,
        spec: RequestSpec,
        client: Generator,
        cell_id: str | None,
        *,
        on_event: EventFn | None = None,
        **generate_kwargs: Any,
    ) -> tuple[RequestRecord, bool]:
        """``(record, aliased)``: an existing record is returned untouched (``aliased=True``)
        and **nothing is generated**; otherwise the client generates and the result is
        published first-writer-wins.  A lost race (another cell committed the same id while
        we generated) yields the committed record with ``aliased=False`` — the GPU work was
        spent — and fires ``on_event("alias_race", {...})`` so the cell can log it
        (architecture §2.1/§2.2 ``events.jsonl``).  ``generate_kwargs`` (e.g.
        ``guided_json_schema``) are forwarded to ``client.generate``.
        """
        existing = self.get(spec.request_id)
        if existing is not None:
            if on_event is not None:
                on_event("aliased", {"request_id": spec.request_id, "cell_id": cell_id})
            return existing, True
        record = client.generate(spec, cell_id=cell_id, **generate_kwargs)
        if record.request_id != spec.request_id:
            raise ProtocolError(
                f"client returned request_id {record.request_id[:12]} for spec {spec.request_id[:12]}"
            )
        committed, was_new = self.publish(record)
        if not was_new and on_event is not None:
            on_event(
                "alias_race",
                {
                    "request_id": spec.request_id,
                    "cell_id": cell_id,
                    "winner_cell_id": committed.producer.get("cell_id"),
                },
            )
        return committed, False


__all__ = ["EventFn", "Generator", "RequestStore", "load_record_text"]
