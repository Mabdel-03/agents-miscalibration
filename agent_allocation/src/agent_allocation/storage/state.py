"""Coordinator state: SQLite in WAL mode on node-local disk (``paths.state_db_local``), one writer.

Tables: ``nodes`` (one row per semantic node: spec, wave, durable state machine
``pending -> leased -> sent -> response_persisted -> terminal``, lease owner/attempt/heartbeat, attempts used,
terminal outcome ``succeeded | exhausted``), ``attempts`` (one row per attempt: ``dump_record(Attempt)``, the
client-classified result once persisted), ``results`` (the one accepted result + candidate per node) and ``meta``.

Ordering (outbox style): raw artifact fsync'd -> ``record_persisted_response`` -> ``parsed/`` file fsync'd ->
``record_result`` commits the terminal row. Durability of the last few commits is traded for speed
(``synchronous=NORMAL``) because every committed fact is re-derivable from the fsync'd journal and artifacts on the
shared filesystem: ``recover`` restores the backup, re-ingests ``parsed/`` and ``raw/``, and settles or charges every
reservation the journal left open. A node is never leased twice: ``try_lease`` is a conditional update from
``pending``; a lease is only taken over (``take_over_lease``) after its attempt is resolved in the ledger.

Config keys consumed: none (lease timing comes from the dispatcher, which reads ``scheduler.lease_*``).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator

from agent_allocation.contracts import (
    Attempt, AttemptState, Candidate, GenerationResult, GenerationSpec, LedgerEvent, LedgerEventKind,
    TransportStatus, dump_record, load_record,
)
from agent_allocation.paths import RunPaths
from agent_allocation.storage._atomic import copy_atomic, write_once
from agent_allocation.storage.journal import LedgerJournal
from agent_allocation.storage.raw import iter_raw_files, read_raw_response, result_from_raw, sha256_file

OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_EXHAUSTED = "exhausted"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    partition_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    benchmark TEXT NOT NULL,
    kind TEXT NOT NULL,
    wave_index INTEGER,
    spec_json TEXT NOT NULL,
    state TEXT NOT NULL,
    outcome TEXT,
    terminal_reason TEXT,
    attempts_used INTEGER NOT NULL DEFAULT 0,
    lease_owner TEXT,
    lease_attempt_id TEXT,
    heartbeat_at REAL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS nodes_by_state ON nodes (state);
CREATE INDEX IF NOT EXISTS nodes_by_partition ON nodes (partition_id, state);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    state TEXT NOT NULL,
    transport_status TEXT,
    sent INTEGER NOT NULL DEFAULT 0,
    record_json TEXT NOT NULL,
    result_json TEXT,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS attempts_by_node ON attempts (node_id);
CREATE TABLE IF NOT EXISTS results (
    node_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    result_json TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    parsed_path TEXT NOT NULL,
    committed_at TEXT NOT NULL
);
"""

_STATE_ORDER = {s.value: i for i, s in enumerate(AttemptState)}


class StateError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json(record: Any) -> str:
    return json.dumps(dump_record(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class NodeRow:
    node_id: str
    partition_id: str
    item_id: str
    benchmark: str
    kind: str
    wave_index: int | None
    state: str
    outcome: str | None
    terminal_reason: str | None
    attempts_used: int
    lease_owner: str | None
    lease_attempt_id: str | None
    heartbeat_at: float | None


def parsed_root(paths: RunPaths) -> Path:
    """``parsed/`` resolved through RunPaths (shard directory's parent)."""
    return paths.parsed("00").parent.parent


def encode_parsed(result: GenerationResult, candidate: Candidate) -> bytes:
    payload = {"result": dump_record(result), "candidate": dump_record(candidate)}
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def read_parsed(path: Path) -> tuple[GenerationResult, Candidate]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_record(GenerationResult, payload["result"]), load_record(Candidate, payload["candidate"])


class StateStore:
    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths
        paths.local_state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = paths.state_db_local
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, isolation_level=None, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        mode = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            raise StateError(f"SQLite refused WAL mode ({mode}); state must live on local disk")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self.owner: str | None = None
        self.duplicate_results = 0
        self.last_backup_at: float | None = None

    # ---- plumbing -----------------------------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _tx(self):
        store = self

        class _Tx:
            def __enter__(self_inner):
                store._lock.acquire()
                store._conn.execute("BEGIN IMMEDIATE")
                return store._conn

            def __exit__(self_inner, exc_type, exc, tb):
                try:
                    if exc_type is None:
                        store._conn.execute("COMMIT")
                    else:
                        store._conn.execute("ROLLBACK")
                finally:
                    store._lock.release()
                return False

        return _Tx()

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def set_meta(self, key: str, value: str) -> None:
        with self._tx() as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                         (key, value))

    def delete_meta(self, key: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM meta WHERE key=?", (key,))

    def get_meta(self, key: str) -> str | None:
        rows = self._query("SELECT value FROM meta WHERE key=?", (key,))
        return rows[0]["value"] if rows else None

    # ---- nodes --------------------------------------------------------------------------------------------
    def register_nodes(self, specs: Iterable[GenerationSpec], *, wave_index: int | None) -> None:
        now = utc_now()
        by_id = {spec.node_id: spec for spec in specs}
        existing: set[str] = set()
        wanted = list(by_id)
        for start in range(0, len(wanted), 500):
            chunk = wanted[start:start + 500]
            marks = ",".join("?" * len(chunk))
            existing.update(r["node_id"] for r in self._query(f"SELECT node_id FROM nodes WHERE node_id IN ({marks})",
                                                               chunk))
        rows = [(node_id, spec.partition_id, spec.item_id, spec.benchmark, spec.kind.value, wave_index, _json(spec),
                 AttemptState.PENDING.value, now) for node_id, spec in by_id.items() if node_id not in existing]
        with self._tx() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO nodes(node_id, partition_id, item_id, benchmark, kind, wave_index, spec_json,"
                " state, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            if wave_index is not None and existing:
                conn.executemany("UPDATE nodes SET wave_index=? WHERE node_id=? AND wave_index IS NULL",
                                 [(wave_index, node_id) for node_id in existing])
            ids = [(r[0],) for r in rows]
            # Facts recorded before registration (recovery re-ingest) win over the fresh 'pending' row.
            conn.executemany(
                "UPDATE nodes SET attempts_used=MAX(attempts_used, COALESCE((SELECT MAX(attempt_no) FROM attempts"
                " WHERE attempts.node_id=nodes.node_id), 0)) WHERE node_id=?", ids)
            conn.executemany(
                "UPDATE nodes SET state='response_persisted' WHERE node_id=? AND state='pending' AND EXISTS("
                "SELECT 1 FROM attempts WHERE attempts.node_id=nodes.node_id AND attempts.state='response_persisted')",
                ids)
            conn.executemany(
                "UPDATE nodes SET state='terminal', outcome='succeeded', lease_owner=NULL, lease_attempt_id=NULL"
                " WHERE node_id=? AND state!='terminal' AND EXISTS(SELECT 1 FROM results WHERE"
                " results.node_id=nodes.node_id)", ids)

    def terminal_node_ids(self) -> set[str]:
        return {r["node_id"] for r in self._query("SELECT node_id FROM nodes WHERE state='terminal'")}

    def node(self, node_id: str) -> NodeRow | None:
        rows = self._query("SELECT * FROM nodes WHERE node_id=?", (node_id,))
        return self._node_row(rows[0]) if rows else None

    @staticmethod
    def _node_row(row: sqlite3.Row) -> NodeRow:
        return NodeRow(row["node_id"], row["partition_id"], row["item_id"], row["benchmark"], row["kind"],
                       row["wave_index"], row["state"], row["outcome"], row["terminal_reason"], row["attempts_used"],
                       row["lease_owner"], row["lease_attempt_id"], row["heartbeat_at"])

    def nodes(self, node_ids: Iterable[str] | None = None) -> dict[str, NodeRow]:
        if node_ids is None:
            return {r["node_id"]: self._node_row(r) for r in self._query("SELECT * FROM nodes")}
        wanted = list(node_ids)
        out: dict[str, NodeRow] = {}
        for start in range(0, len(wanted), 500):
            chunk = wanted[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for r in self._query(f"SELECT * FROM nodes WHERE node_id IN ({marks})", chunk):
                out[r["node_id"]] = self._node_row(r)
        return out

    def spec(self, node_id: str) -> GenerationSpec | None:
        rows = self._query("SELECT spec_json FROM nodes WHERE node_id=?", (node_id,))
        return load_record(GenerationSpec, json.loads(rows[0]["spec_json"])) if rows else None

    def try_lease(self, node_id: str, attempt_id: str, *, owner: str, now: float | None = None) -> bool:
        """Atomically move a pending node to ``leased`` for ``attempt_id``; False if anyone else holds it."""
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE nodes SET state='leased', lease_owner=?, lease_attempt_id=?, heartbeat_at=?, updated_at=?"
                " WHERE node_id=? AND state='pending'",
                (owner, attempt_id, time.time() if now is None else now, utc_now(), node_id))
            return cur.rowcount == 1

    def heartbeat(self, owner: str, *, now: float | None = None) -> int:
        with self._tx() as conn:
            cur = conn.execute("UPDATE nodes SET heartbeat_at=? WHERE lease_owner=? AND state IN ('leased', 'sent')",
                               (time.time() if now is None else now, owner))
            return cur.rowcount

    def leased_nodes(self) -> list[NodeRow]:
        return [self._node_row(r) for r in self._query("SELECT * FROM nodes WHERE state IN ('leased', 'sent')")]

    def take_over_lease(self, node_id: str, lease_attempt_id: str) -> bool:
        """Return a leased/sent node to pending once its attempt is resolved (never while the attempt is live)."""
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE nodes SET state='pending', lease_owner=NULL, lease_attempt_id=NULL, heartbeat_at=NULL,"
                " updated_at=? WHERE node_id=? AND state IN ('leased', 'sent') AND lease_attempt_id=?",
                (utc_now(), node_id, lease_attempt_id))
            return cur.rowcount == 1

    def mark_exhausted(self, node_id: str, reason: str) -> None:
        self.mark_terminal_without_candidate(node_id, OUTCOME_EXHAUSTED, reason)

    def mark_terminal_without_candidate(self, node_id: str, outcome: str, reason: str) -> None:
        """Terminal with no accepted result (attempts exhausted, non-retryable status, context limit)."""
        if outcome == OUTCOME_SUCCEEDED:
            raise StateError("a node without a candidate cannot be 'succeeded'")
        with self._tx() as conn:
            conn.execute(
                "UPDATE nodes SET state='terminal', outcome=?, terminal_reason=?, lease_owner=NULL,"
                " lease_attempt_id=NULL, updated_at=? WHERE node_id=? AND state!='terminal'",
                (outcome, reason, utc_now(), node_id))

    # ---- attempts -----------------------------------------------------------------------------------------
    def record_attempt(self, attempt: Attempt) -> None:
        self._record_attempt(attempt, sent=None)

    def record_unsent_attempt(self, attempt: Attempt, *, detail: str) -> None:
        """Terminal record for an attempt that was marked sent but provably never left the process."""
        self._record_attempt(replace(attempt, state=AttemptState.TERMINAL, sent_at=None, completed_at=utc_now(),
                                     error_detail=detail), sent=False)

    def _record_attempt(self, attempt: Attempt, *, sent: bool | None) -> None:
        """``sent=None`` derives the flag from the record (``sent_at`` set or state SENT/RESPONSE_PERSISTED) and never
        lowers it; an explicit ``sent`` overrides it."""
        now = utc_now()
        state = attempt.state.value
        sent_update = "MAX(attempts.sent, excluded.sent)" if sent is None else "excluded.sent"
        if sent is None:
            sent = attempt.sent_at is not None or attempt.state in (AttemptState.SENT, AttemptState.RESPONSE_PERSISTED)
        with self._tx() as conn:
            existing = conn.execute("SELECT state FROM attempts WHERE attempt_id=?", (attempt.attempt_id,)).fetchone()
            if existing is not None and _STATE_ORDER[existing["state"]] > _STATE_ORDER[state]:
                return  # never regress an attempt
            conn.execute(
                "INSERT INTO attempts(attempt_id, node_id, attempt_no, state, transport_status, sent, record_json,"
                " updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(attempt_id) DO UPDATE SET state=excluded.state,"
                f" transport_status=excluded.transport_status, sent={sent_update},"
                " record_json=excluded.record_json, updated_at=excluded.updated_at",
                (attempt.attempt_id, attempt.node_id, attempt.attempt_no, state,
                 None if attempt.transport_status is None else attempt.transport_status.value, int(sent),
                 _json(attempt), now))
            conn.execute("UPDATE nodes SET attempts_used=MAX(attempts_used, ?) WHERE node_id=?",
                         (attempt.attempt_no, attempt.node_id))
            if attempt.state in (AttemptState.LEASED, AttemptState.SENT):
                conn.execute(
                    "UPDATE nodes SET state=?, lease_attempt_id=?, lease_owner=COALESCE(lease_owner, ?),"
                    " heartbeat_at=?, updated_at=? WHERE node_id=? AND state NOT IN ('terminal', 'response_persisted')",
                    (state, attempt.attempt_id, self.owner, time.time(), now, attempt.node_id))
            elif attempt.state is AttemptState.TERMINAL:
                conn.execute(
                    "UPDATE nodes SET state='pending', lease_owner=NULL, lease_attempt_id=NULL, heartbeat_at=NULL,"
                    " updated_at=? WHERE node_id=? AND state IN ('leased', 'sent') AND lease_attempt_id=?",
                    (now, attempt.node_id, attempt.attempt_id))

    def sends_by_node(self) -> dict[str, int]:
        """Attempts that (may have) left the process, per node: the transport-attempt limit counts these."""
        return {r["node_id"]: r["n"] for r in
                self._query("SELECT node_id, COUNT(*) AS n FROM attempts WHERE sent=1 GROUP BY node_id")}

    def attempts_for(self, node_id: str) -> list[Attempt]:
        rows = self._query("SELECT record_json FROM attempts WHERE node_id=? ORDER BY attempt_no", (node_id,))
        return [load_record(Attempt, json.loads(r["record_json"])) for r in rows]

    def attempt_counts(self) -> dict[str, int]:
        rows = self._query("SELECT COALESCE(transport_status, state) AS k, COUNT(*) AS n FROM attempts GROUP BY k")
        return {r["k"]: r["n"] for r in rows}

    def record_persisted_response(self, attempt: Attempt, result: GenerationResult) -> None:
        """Raw artifact is on disk: remember the classified result so a crash before commit never re-sends."""
        if result.raw_response_path is None or result.raw_response_sha256 is None:
            raise StateError("record_persisted_response requires the raw artifact pointer")
        now = utc_now()
        persisted = replace(attempt, state=AttemptState.RESPONSE_PERSISTED)
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO attempts(attempt_id, node_id, attempt_no, state, transport_status, sent, record_json,"
                " result_json, updated_at) VALUES(?, ?, ?, 'response_persisted', ?, 1, ?, ?, ?) ON CONFLICT(attempt_id)"
                " DO UPDATE SET sent=1, state=CASE WHEN attempts.state='terminal' THEN attempts.state ELSE excluded.state END,"
                " transport_status=excluded.transport_status, record_json=excluded.record_json,"
                " result_json=excluded.result_json, updated_at=excluded.updated_at",
                (attempt.attempt_id, attempt.node_id, attempt.attempt_no, result.transport_status.value,
                 _json(persisted), _json(result), now))
            conn.execute("UPDATE nodes SET state='response_persisted', updated_at=? WHERE node_id=?"
                         " AND state IN ('pending', 'leased', 'sent')", (now, attempt.node_id))

    def persisted_unfinished(self) -> list[tuple[Attempt, GenerationResult]]:
        """Persisted OK responses whose node has no accepted result yet (finish them without re-sending)."""
        rows = self._query(
            "SELECT a.record_json, a.result_json FROM attempts a WHERE a.state='response_persisted'"
            " AND a.result_json IS NOT NULL AND NOT EXISTS (SELECT 1 FROM results r WHERE r.node_id=a.node_id)"
            " ORDER BY a.node_id, a.attempt_no")
        return [(load_record(Attempt, json.loads(r["record_json"])),
                 load_record(GenerationResult, json.loads(r["result_json"]))) for r in rows]

    # ---- results ------------------------------------------------------------------------------------------
    def record_result(self, result: GenerationResult, candidate: Candidate) -> None:
        self.commit_result(result, candidate)

    def commit_result(self, result: GenerationResult, candidate: Candidate) -> bool:
        """``record_result`` returning whether this call accepted the node's result (False = duplicate)."""
        if not (result.node_id == candidate.node_id == candidate.candidate_id):
            raise StateError(f"result/candidate identity mismatch for {result.node_id}")
        if result.raw_response_path is None or result.raw_response_sha256 is None:
            raise StateError(f"result {result.attempt_id} has no persisted raw artifact")
        existing = self._query("SELECT attempt_id FROM results WHERE node_id=?", (result.node_id,))
        if existing:
            self.duplicate_results += 1
            return False
        path = self.paths.parsed(result.node_id)
        if not write_once(path, encode_parsed(result, candidate)):
            stored_result, stored_candidate = read_parsed(path)
            if stored_result.attempt_id != result.attempt_id:
                self.duplicate_results += 1
                self._ingest_result(stored_result, stored_candidate)
                return False
            result, candidate = stored_result, stored_candidate
        return self._ingest_result(result, candidate)

    def _ingest_result(self, result: GenerationResult, candidate: Candidate) -> bool:
        now = utc_now()
        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM results WHERE node_id=?", (result.node_id,)).fetchone():
                return False
            conn.execute(
                "INSERT INTO results(node_id, attempt_id, result_json, candidate_json, parsed_path, committed_at)"
                " VALUES(?, ?, ?, ?, ?, ?)",
                (result.node_id, result.attempt_id, _json(result), _json(candidate),
                 self.paths.relative(self.paths.parsed(result.node_id)), now))
            row = conn.execute("SELECT record_json FROM attempts WHERE attempt_id=?", (result.attempt_id,)).fetchone()
            if row is not None:
                attempt = replace(load_record(Attempt, json.loads(row["record_json"])), state=AttemptState.TERMINAL)
                conn.execute("UPDATE attempts SET state='terminal', record_json=?, result_json=NULL, updated_at=?"
                             " WHERE attempt_id=?", (_json(attempt), now, result.attempt_id))
            else:
                from agent_allocation.runtime.ledger import attempt_number
                attempt = Attempt(result.attempt_id, result.node_id, attempt_number(result.attempt_id),
                                  AttemptState.TERMINAL, Decimal("0"), now, completed_at=now,
                                  transport_status=result.transport_status, http_status=result.http_status,
                                  generation_id=result.generation_id)
                conn.execute(
                    "INSERT INTO attempts(attempt_id, node_id, attempt_no, state, transport_status, sent, record_json,"
                    " updated_at) VALUES(?, ?, ?, 'terminal', ?, 1, ?, ?)",
                    (attempt.attempt_id, attempt.node_id, attempt.attempt_no, result.transport_status.value,
                     _json(attempt), now))
            conn.execute(
                "UPDATE nodes SET state='terminal', outcome='succeeded', terminal_reason=NULL, lease_owner=NULL,"
                " lease_attempt_id=NULL, attempts_used=MAX(attempts_used, (SELECT MAX(attempt_no) FROM attempts"
                " WHERE attempts.node_id=nodes.node_id)), updated_at=? WHERE node_id=?", (now, result.node_id))
        return True

    def candidate(self, node_id: str) -> Candidate | None:
        rows = self._query("SELECT candidate_json FROM results WHERE node_id=?", (node_id,))
        return load_record(Candidate, json.loads(rows[0]["candidate_json"])) if rows else None

    def result(self, node_id: str) -> GenerationResult | None:
        rows = self._query("SELECT result_json FROM results WHERE node_id=?", (node_id,))
        return load_record(GenerationResult, json.loads(rows[0]["result_json"])) if rows else None

    # ---- status -------------------------------------------------------------------------------------------
    def counts(self) -> dict[str, dict[str, int]]:
        """``{partition: {state or terminal outcome: count}}``."""
        out: dict[str, dict[str, int]] = {}
        rows = self._query(
            "SELECT partition_id, CASE WHEN state='terminal' THEN 'terminal_' || COALESCE(outcome, 'unknown')"
            " ELSE state END AS k, COUNT(*) AS n FROM nodes GROUP BY partition_id, k")
        for r in rows:
            out.setdefault(r["partition_id"], {})[r["k"]] = r["n"]
        return out

    # ---- backup / recovery --------------------------------------------------------------------------------
    def backup(self) -> None:
        """Online, consistent copy to ``paths.state_db_backup`` (temp file + fsync + atomic replace)."""
        final = self.paths.state_db_backup
        final.parent.mkdir(parents=True, exist_ok=True)
        local_tmp = self.paths.local_state_dir / f".backup-{os.getpid()}-{threading.get_ident()}.sqlite"
        try:
            source = sqlite3.connect(self.db_path, timeout=30.0)
            dest = sqlite3.connect(local_tmp)
            try:
                source.backup(dest)
            finally:
                dest.close()
                source.close()
            copy_atomic(local_tmp, final)
        finally:
            try:
                local_tmp.unlink()
            except FileNotFoundError:
                pass
        self.last_backup_at = time.time()

    @staticmethod
    def _db_ok(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
            try:
                return conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            finally:
                conn.close()
        except sqlite3.DatabaseError:
            return False

    @classmethod
    def recover(cls, paths: RunPaths) -> "StateStore":
        """Restore backup (if the local DB is missing or corrupt) -> re-ingest ``parsed/`` and ``raw/`` -> resolve
        every reservation the journal left open (settle from a persisted response's confirmed cost, else charge the
        full reservation). Successful nodes are never re-sent afterwards."""
        paths.local_state_dir.mkdir(parents=True, exist_ok=True)
        local = paths.state_db_local
        if not cls._db_ok(local):
            for suffix in ("", "-wal", "-shm"):
                stale = local.with_name(local.name + suffix)
                if stale.exists():
                    stale.rename(stale.with_name(f"{stale.name}.corrupt-{int(time.time())}"))
            if paths.state_db_backup.exists() and cls._db_ok(paths.state_db_backup):
                copy_atomic(paths.state_db_backup, local)
        store = cls(paths)
        from agent_allocation.runtime.owner import owner_host, owner_is_dead
        import socket
        owner = store.get_meta("dispatcher_owner")
        if owner and owner_host(owner) == socket.gethostname() and not owner_is_dead(owner):
            store.close()
            raise StateError(f"dispatcher {owner} is alive on this host: recovery would charge its in-flight "
                             "reservations; stop it first")
        store.recovery_report = store._reingest_and_reconcile()
        return store

    def _reingest_and_reconcile(self) -> dict[str, int]:
        from agent_allocation.runtime.ledger import attempt_number, confirmed_cost, journal_attempts, utc_now as now_iso
        report = {"parsed_ingested": 0, "raw_persisted": 0, "raw_failed_attempts": 0, "journal_settled_from_raw": 0,
                  "journal_charged_max": 0, "journal_attempts_recorded": 0}
        # 1. parsed/ -> accepted results
        root = parsed_root(self.paths)
        if root.is_dir():
            for shard_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                for file in sorted(shard_dir.glob("*.json")):
                    result, candidate = read_parsed(file)
                    if self._ingest_result(result, candidate):
                        report["parsed_ingested"] += 1
        # 2. raw/ -> persisted responses not yet committed, and failed attempts
        journal = LedgerJournal(self.paths.ledger_journal, group_commit_ms=0)
        known = journal_attempts(journal)
        accepted = {r["node_id"] for r in self._query("SELECT node_id FROM results")}
        raw_results: dict[str, GenerationResult] = {}
        for file in iter_raw_files(self.paths):
            relative = self.paths.relative(file)
            payload = read_raw_response(self.paths, relative)
            result = result_from_raw(payload, relative_path=relative, sha256=sha256_file(file))
            if result is None:
                continue
            raw_results[result.attempt_id] = result
            row = self._query("SELECT state FROM attempts WHERE attempt_id=?", (result.attempt_id,))
            if row and row[0]["state"] == AttemptState.TERMINAL.value:
                continue
            reserved = known[result.attempt_id].reserved_usd if result.attempt_id in known else Decimal("0")
            attempt = Attempt(result.attempt_id, result.node_id, attempt_number(result.attempt_id),
                              AttemptState.SENT, reserved, now_iso(), transport_status=result.transport_status,
                              http_status=result.http_status, generation_id=result.generation_id)
            if result.transport_status is TransportStatus.OK and result.node_id not in accepted:
                self.record_persisted_response(attempt, result)
                report["raw_persisted"] += 1
            else:
                self._record_attempt(replace(attempt, state=AttemptState.TERMINAL, completed_at=now_iso()), sent=True)
                report["raw_failed_attempts"] += 1
        # 3. journal -> every attempt known to the DB; every open reservation resolved conservatively
        for aid, info in known.items():
            if not self._query("SELECT 1 FROM attempts WHERE attempt_id=?", (aid,)):
                self._record_attempt(Attempt(aid, info.node_id or aid.rsplit(":", 1)[0], attempt_number(aid),
                                             AttemptState.TERMINAL, info.reserved_usd, now_iso(),
                                             error_detail="recovered_from_journal"),
                                     sent=info.sent or info.resolution is None)
                report["journal_attempts_recorded"] += 1
            if info.resolution is not None:
                continue
            cost = confirmed_cost(raw_results.get(aid))
            if cost is not None:
                kind, amount, detail = LedgerEventKind.SETTLE, cost, "recovery:settled_from_raw"
                report["journal_settled_from_raw"] += 1
            else:
                kind, amount, detail = LedgerEventKind.CHARGE_MAX, info.reserved_usd, "recovery:no_confirmed_cost"
                report["journal_charged_max"] += 1
            journal.append_sync(LedgerEvent(seq=journal.next_seq(), kind=kind, amount_usd=amount,
                                            recorded_at=now_iso(), attempt_id=aid, node_id=info.node_id,
                                            partition_id=info.partition_id, detail=detail))
        return report

    recovery_report: dict[str, int] = {}


def iter_parsed_files(paths: RunPaths) -> Iterator[Path]:
    root = parsed_root(paths)
    if not root.is_dir():
        return
    for shard_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        yield from sorted(shard_dir.glob("*.json"))
