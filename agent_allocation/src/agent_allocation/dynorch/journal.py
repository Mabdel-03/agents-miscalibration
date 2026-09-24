"""WP-A, contract §5: the immutable per-call journal, the mutable run store and the dynorch-aware ledger recovery.

``CallJournal`` writes ``<run_root>/raw/<arm>/<rung>/<repeat>/<item_slug>/a<attempt>/call_<nnnn>.json.gz`` through
``storage/_atomic.write_once`` (gzip ``mtime=0``, sorted keys: identical content -> identical bytes; a second write
with different bytes raises ``CallJournalConflict``, never overwrites). Every string in the payload passes through
``apikey.redact`` and every headers mapping through ``apikey.redact_headers`` before encoding.

``RunStore`` keeps ``<run_root>/runs/<arm>/<rung>/<repeat>/<item_slug>.json`` (``replace_atomic``, one file per
planned cell, rewritten at ``open_run``, every call and ``close_run``), ``scan()`` -> ``{run_key: RunRecord}`` and
``is_terminal``. A superseded attempt's record moves to ``runs_superseded/`` (never deleted).

``resolve_inherited(ledger, store, journal)`` is the resume path's ONE owner of recovery charging for dynorch
attempt ids ``f"{run_key}:g{gen}:{call_no}:{phase}:{n}"``: a reserved-only attempt provably never left ->
``release_unsent``; a sent attempt whose exchange was journaled -> ``settle`` at the journaled cost (never
``charge_max`` for a call whose reply was journaled); anything else -> ``charge_max``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Mapping

from agent_allocation.apikey import REDACTED_HEADERS, redact, redact_headers
from agent_allocation.dynorch.records import RunRecord, RunSpec, from_json, item_slug, to_json
from agent_allocation.runtime.ledger import CollectionLedger
from agent_allocation.storage._atomic import replace_atomic, write_once

JOURNAL_LAYOUT = "dynorch/v1"
_ATTEMPT_ID = re.compile(r"^(?P<run_key>[^:]+):g(?P<generation>\d+):(?P<call_no>\d+):(?P<phase>[a-z_]+):(?P<n>\d+)$")


class CallJournalConflict(RuntimeError):
    """A call file already exists with different bytes (immutability)."""


def attempt_id(run_key: str, generation: int, call_no: int, phase: str, attempt_no: int = 1) -> str:
    """Contract §3/§5: ``f"{run_key}:g{gen}:{call_no}:{phase}:1"`` -- the execution generation keeps a resumed
    process from colliding with the ids the interrupted one reserved."""
    if ":" in run_key or not run_key:
        raise ValueError(f"run_key {run_key!r} must be non-empty and contain no ':'")
    if not re.fullmatch(r"[a-z_]+", phase):
        raise ValueError(f"phase {phase!r} must be a lowercase word")
    return f"{run_key}:g{int(generation)}:{int(call_no)}:{phase}:{int(attempt_no)}"


def parse_attempt_id(value: str) -> dict[str, Any]:
    match = _ATTEMPT_ID.match(value)
    if match is None:
        raise ValueError(f"not a dynorch attempt id: {value!r}")
    parts = match.groupdict()
    return {"run_key": parts["run_key"], "generation": int(parts["generation"]), "call_no": int(parts["call_no"]),
            "phase": parts["phase"], "attempt_no": int(parts["n"])}


def redact_payload(value: Any) -> Any:
    """Every string redacted, every secret header value replaced, recursively (bytes decoded first)."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return redact(bytes(value).decode("utf-8", errors="backslashreplace")) or ""
    if isinstance(value, str):
        return redact(value) or ""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.lower() in REDACTED_HEADERS:
                out[name] = "[REDACTED]"
            elif name in ("response_headers", "headers") and isinstance(item, Mapping):
                out[name] = redact_headers({str(k): str(v) for k, v in item.items()})
            else:
                out[name] = redact_payload(item)
        return out
    if isinstance(value, (list, tuple)):
        return [redact_payload(v) for v in value]
    if isinstance(value, Decimal):
        return str(value)
    return value


def encode_payload(payload: Mapping[str, Any]) -> bytes:
    data = json.dumps(redact_payload(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return gzip.compress(data.encode("utf-8"), compresslevel=6, mtime=0)


def decode_payload(data: bytes) -> dict[str, Any]:
    return json.loads(gzip.decompress(data).decode("utf-8"))


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class CallJournal:
    def __init__(self, run_root: Path) -> None:
        self.run_root = Path(run_root)

    def run_dir(self, spec: RunSpec) -> Path:
        return (self.run_root / "raw" / spec.arm / str(spec.rung) / str(spec.repeat) / item_slug(spec.item_id)
                / f"a{spec.attempt}")

    def path(self, spec: RunSpec, call_no: int) -> Path:
        if isinstance(call_no, bool) or not isinstance(call_no, int) or call_no < 1:
            raise ValueError("call_no must be a positive integer")
        return self.run_dir(spec) / f"call_{call_no:04d}.json.gz"

    def relative(self, path: Path) -> str:
        return str(Path(path).resolve().relative_to(self.run_root.resolve()))

    def write(self, spec: RunSpec, call_no: int, payload: Mapping[str, Any]) -> Path:
        """Write once; identical bytes are a no-op, different bytes raise ``CallJournalConflict``."""
        if payload.get("layout") != JOURNAL_LAYOUT:
            raise ValueError(f"payload layout must be {JOURNAL_LAYOUT!r}")
        if payload.get("run_key") != spec.run_key or payload.get("call_no") != call_no:
            raise ValueError("payload run_key/call_no do not match the spec/call")
        final = self.path(spec, call_no)
        data = encode_payload(payload)
        if not write_once(final, data):
            if hashlib.sha256(final.read_bytes()).hexdigest() != hashlib.sha256(data).hexdigest():
                raise CallJournalConflict(f"{final} already holds different bytes for call {call_no}")
        return final

    def read(self, spec: RunSpec, call_no: int) -> dict[str, Any]:
        return decode_payload(self.path(spec, call_no).read_bytes())

    def scan(self, spec: RunSpec, *, include_superseded: bool = False) -> list[dict[str, Any]]:
        files = sorted(self.run_dir(spec).glob("call_*.json.gz"))
        if include_superseded:
            files += sorted((self.run_dir(spec) / "superseded").glob("*/call_*.json.gz"))
        return [decode_payload(f.read_bytes()) for f in files]

    def supersede(self, spec: RunSpec) -> Path | None:
        """Move an interrupted execution's call files to ``<run_dir>/superseded/<utc>/`` (never deleted)."""
        run_dir = self.run_dir(spec)
        files = sorted(run_dir.glob("call_*.json.gz"))
        if not files:
            return None
        target = run_dir / "superseded" / _utc()
        target.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.move(str(file), str(target / file.name))
        return target

    def iter_all(self) -> Iterator[tuple[Path, dict[str, Any]]]:
        root = self.run_root / "raw"
        if not root.is_dir():
            return
        for file in sorted(root.rglob("call_*.json.gz")):
            yield file, decode_payload(file.read_bytes())


class RunStore:
    def __init__(self, run_root: Path) -> None:
        self.run_root = Path(run_root)

    def path(self, spec: RunSpec) -> Path:
        return self.run_root / "runs" / spec.arm / str(spec.rung) / str(spec.repeat) / f"{item_slug(spec.item_id)}.json"

    def superseded_path(self, spec: RunSpec) -> Path:
        return (self.run_root / "runs_superseded" / spec.arm / str(spec.rung) / str(spec.repeat)
                / f"{item_slug(spec.item_id)}.a{spec.attempt}.json")

    def save(self, record: RunRecord) -> Path:
        final = self.path(record.spec)
        data = json.dumps(to_json(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        replace_atomic(final, redact(data.decode("utf-8")).encode("utf-8"))
        return final

    def load(self, spec: RunSpec) -> RunRecord | None:
        path = self.path(spec)
        if not path.exists():
            return None
        return from_json(RunRecord, json.loads(path.read_text(encoding="utf-8")))

    def archive_previous(self, spec: RunSpec) -> Path | None:
        """Before a new attempt of the same cell is opened: move the previous attempt's record aside."""
        path = self.path(spec)
        if not path.exists():
            return None
        previous = from_json(RunRecord, json.loads(path.read_text(encoding="utf-8")))
        if previous.spec.run_key == spec.run_key:
            return None
        target = self.superseded_path(previous.spec)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        return target

    def scan(self, *, include_superseded: bool = False) -> dict[str, RunRecord]:
        found: dict[str, RunRecord] = {}
        roots = [self.run_root / "runs"] + ([self.run_root / "runs_superseded"] if include_superseded else [])
        for root in roots:
            if not root.is_dir():
                continue
            for file in sorted(root.rglob("*.json")):
                record = from_json(RunRecord, json.loads(file.read_text(encoding="utf-8")))
                found[record.spec.run_key] = record
        return found

    @staticmethod
    def is_terminal(record: RunRecord) -> bool:
        return record.is_terminal


def journaled_cost(journal: CallJournal, spec: RunSpec, aid: str) -> Decimal | None:
    """The cost the journal recorded for ledger attempt ``aid`` (``None`` when no journaled phase carries it)."""
    parsed = parse_attempt_id(aid)
    path = journal.path(spec, parsed["call_no"])
    candidates = [path] if path.exists() else []
    candidates += sorted((journal.run_dir(spec) / "superseded").glob(f"*/{path.name}"))
    for file in candidates:
        payload = decode_payload(file.read_bytes())
        for phase in payload.get("phases") or []:
            if phase.get("attempt_id") == aid and phase.get("cost_usd") is not None:
                return Decimal(str(phase["cost_usd"]))
    return None


async def resolve_inherited(ledger: CollectionLedger, store: RunStore, journal: CallJournal | None = None
                            ) -> dict[str, int]:
    """Contract §5 / plan §5: settle inherited SENT attempts from the call journal; charge the maximum only when
    no journaled reply exists. Idempotent (the ledger first applies anything already resolved)."""
    journal = journal or CallJournal(store.run_root)
    report = {"settled_from_journal": 0, "charged_max": 0, "released_unsent": 0, "unparsed": 0}
    pending = sorted(ledger.inherited_attempt_ids)
    if not pending:
        return report
    records = store.scan(include_superseded=True)
    for aid in pending:
        state = ledger.state_of(aid)
        if state == "reserved":
            await ledger.release_unsent(aid)
            report["released_unsent"] += 1
            continue
        if state != "unresolved":
            continue
        try:
            run_key = parse_attempt_id(aid)["run_key"]
        except ValueError:
            report["unparsed"] += 1
            await ledger.charge_max(aid)
            report["charged_max"] += 1
            continue
        record = records.get(run_key)
        cost = journaled_cost(journal, record.spec, aid) if record is not None else None
        if cost is not None:
            await ledger.settle(aid, cost)
            report["settled_from_journal"] += 1
        else:
            await ledger.charge_max(aid)
            report["charged_max"] += 1
    return report
