"""Append-only ledger journal ``state/ledger_journal.jsonl``: one ``dump_record(LedgerEvent)`` per line, fsync'd.

Single writer (the pilot's kill -9 resume wrote duplicate seqs through two journal instances; this is the fix):

* Constructing a ``LedgerJournal`` never writes. The *writer role* is taken lazily by the first append
  (``ensure_writer``): an in-process registry keyed by the resolved path plus an exclusive ``flock`` on the sidecar
  ``<journal>.lock`` (a file readers never open, so NFS's per-process POSIX lock emulation is not dropped by a reader
  closing the journal). Another process holding the lock -> :class:`JournalError`. Another instance in this process
  holding it with appends still pending -> :class:`JournalError`; an *idle* in-process holder is revoked (sequential
  use after a simulated crash, or a second Runtime between phases).
* On taking the role the journal repairs a torn tail and reads every line appended since it last looked, assigns
  ``seq`` from that tail, and hands the new events to ``listener`` (the ledger applies them under its lock before it
  decides anything). ``release_writer`` gives the role up (the dispatcher does so when ``run()`` ends).
* ``_accept`` refuses a seq that is not ``tail + 1``, a second ``RESERVE`` for an attempt and a second terminal event
  (``SETTLE``/``CHARGE_MAX``/``RELEASE_UNSENT``) for an attempt: duplicate charges cannot reach the file.
* ``append`` returns only after the line is on disk (``os.fsync``). Appends are group-committed: the first pending
  event starts a batch that lingers ``group_commit_ms`` (one fsync covers every event that arrives in the window),
  then the batch is written and fsync'd off the event loop. An *urgent* submit (the ledger's ``SENT``, written while
  the HTTP slot is held) ends the linger at once.
* A malformed line anywhere but the tail is a :class:`JournalError`. After a failed write the journal is *broken*:
  every later append raises; replay of a partially written batch is conservative (an open reservation is charged).

Config keys consumed: none (``group_commit_ms`` is a constructor argument; 250 ms by contract).
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import threading
import weakref
from pathlib import Path
from typing import Callable, Iterator

from agent_allocation.contracts import LedgerEvent, LedgerEventKind, dump_record, load_record

TERMINAL_KINDS = frozenset({LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX, LedgerEventKind.RELEASE_UNSENT})

_REGISTRY_LOCK = threading.RLock()
_WRITERS: dict[str, "weakref.ref[LedgerJournal]"] = {}


class JournalError(RuntimeError):
    def __init__(self, message: str, *, maybe_written: bool = True) -> None:
        super().__init__(message)
        self.maybe_written = maybe_written


def encode_event(event: LedgerEvent) -> str:
    return json.dumps(dump_record(event), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _decode_line(raw: bytes, *, where: str) -> LedgerEvent:
    try:
        return load_record(LedgerEvent, json.loads(raw.decode("utf-8")))
    except (ValueError, TypeError, KeyError) as exc:
        raise JournalError(f"{where}: malformed ledger event: {exc}") from exc


def replay_events(path: Path) -> Iterator[LedgerEvent]:
    """Read-only replay (never repairs): a torn final line is skipped, a malformed middle line raises."""
    path = Path(path)
    if not path.exists():
        return
    last_seq = 0
    with open(path, "rb") as handle:
        data = handle.read()
    lines = data.split(b"\n")
    complete = lines[:-1]  # the last element is b"" after a final newline, or a torn line without one
    for number, raw in enumerate(complete, start=1):
        if not raw.strip():
            continue
        event = _decode_line(raw, where=f"{path}:{number}")
        if event.seq <= last_seq:
            raise JournalError(f"{path}:{number}: seq {event.seq} does not increase (last {last_seq})")
        last_seq = event.seq
        yield event


class LedgerJournal:
    def __init__(self, path: Path, *, group_commit_ms: int = 250) -> None:
        if group_commit_ms < 0:
            raise ValueError("group_commit_ms must be >= 0")
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.group_commit_ms = int(group_commit_ms)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.listener: Callable[[list[LedgerEvent]], None] | None = None
        self._io_lock = threading.Lock()
        self._pending: list[tuple[str, asyncio.Future]] = []
        self._flusher: asyncio.Task | None = None
        self._urgent: asyncio.Event | None = None
        self._broken: BaseException | None = None
        self._writer_fd: int | None = None
        self._lock_fd: int | None = None
        self._offset = 0                      # bytes of complete lines already seen (read or written by us)
        self._last_seq = 0
        self._reserved: set[str] = set()
        self._terminal: set[str] = set()
        self.torn_tail_bytes = 0
        self.external_events_applied = 0
        self.revocations = 0
        self._scanned = False                 # read-only users (``LedgerJournal(path).replay()``) never pay a scan
        self.batches_written = 0
        self.events_written = 0
        self.max_batch = 0

    # ---- reading ------------------------------------------------------------------------------------------
    def replay(self) -> Iterator[LedgerEvent]:
        """Read-only replay of the whole file (does not move this instance's position)."""
        return replay_events(self.path)

    def reload(self) -> Iterator[LedgerEvent]:
        """Re-read the whole file *as this instance's position*: yields every event and leaves ``_offset`` /
        ``last_seq`` / the attempt indexes exactly at what was yielded, so a later writer take-over hands the ledger
        only events it has not applied. The ledger builds its state from this, never from a separate replay."""
        with _REGISTRY_LOCK:
            self._offset = 0
            self._last_seq = 0
            self._reserved.clear()
            self._terminal.clear()
            self._scanned = True
            events = self._read_new()
        yield from events

    def _ensure_scanned(self) -> None:
        if not self._scanned:
            self._scanned = True
            self._read_new()

    def _read_new(self) -> list[LedgerEvent]:
        """Parse complete lines appended after ``_offset`` and fold them into the seq/attempt indexes."""
        if not self.path.exists():
            return []
        with open(self.path, "rb") as handle:
            handle.seek(self._offset)
            data = handle.read()
        end = data.rfind(b"\n") + 1
        events: list[LedgerEvent] = []
        for raw in data[:end].split(b"\n")[:-1]:
            if not raw.strip():
                continue
            event = _decode_line(raw, where=f"{self.path}@{self._offset}")
            if event.seq <= self._last_seq:
                raise JournalError(f"{self.path}: seq {event.seq} does not increase (last {self._last_seq})",
                                   maybe_written=False)
            self._index(event)
            events.append(event)
        self._offset += end
        return events

    def _index(self, event: LedgerEvent) -> None:
        self._last_seq = event.seq
        if event.attempt_id is None:
            return
        if event.kind is LedgerEventKind.RESERVE:
            self._reserved.add(event.attempt_id)
        elif event.kind in TERMINAL_KINDS:
            self._terminal.add(event.attempt_id)

    @property
    def last_seq(self) -> int:
        self._ensure_scanned()
        return self._last_seq

    def next_seq(self) -> int:
        """The seq the next appended event must carry (callers assign it synchronously before ``append``)."""
        self._ensure_scanned()
        return self._last_seq + 1

    def is_terminal(self, attempt_id: str) -> bool:
        self._ensure_scanned()
        return attempt_id in self._terminal

    @property
    def broken(self) -> BaseException | None:
        return self._broken

    @property
    def is_writer(self) -> bool:
        return self._writer_fd is not None

    @property
    def idle(self) -> bool:
        return not self._pending and (self._flusher is None or self._flusher.done())

    # ---- writer role --------------------------------------------------------------------------------------
    def ensure_writer(self) -> list[LedgerEvent]:
        """Take the single-writer role if not held; returns (and hands ``listener``) events appended meanwhile."""
        if self._writer_fd is not None:
            return []
        if self._broken is not None:
            raise JournalError(f"journal {self.path} is broken after a failed write: {self._broken!r}",
                               maybe_written=False)
        key = str(self.path.resolve())
        with _REGISTRY_LOCK:
            holder_ref = _WRITERS.get(key)
            holder = holder_ref() if holder_ref is not None else None
            if holder is not None and holder is not self and holder.is_writer:
                if not holder.idle:
                    raise JournalError(f"another journal instance in this process is writing {self.path}",
                                       maybe_written=False)
                holder.release_writer()
                holder.revocations += 1
            lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(lock_fd)
                raise JournalError(f"another process holds the writer lock {self.lock_path}",
                                   maybe_written=False) from exc
            try:
                self.torn_tail_bytes += self._repair_torn_tail()
                first_scan = not self._scanned
                self._scanned = True
                new_events = self._read_new()
                if first_scan:
                    new_events = []           # a first full scan is this instance's baseline, not news for a listener
                writer_fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            except BaseException:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                raise
            self._lock_fd, self._writer_fd = lock_fd, writer_fd
            _WRITERS[key] = weakref.ref(self)
        if new_events:
            self.external_events_applied += len(new_events)
            if self.listener is not None:
                self.listener(new_events)
        return new_events

    def release_writer(self) -> None:
        """Give up the writer role (idempotent). Pending appends must be flushed first."""
        if not self.idle:
            raise JournalError(f"release_writer with appends pending on {self.path}", maybe_written=False)
        with _REGISTRY_LOCK:
            with self._io_lock:
                if self._writer_fd is not None:
                    os.close(self._writer_fd)
                    self._writer_fd = None
            if self._lock_fd is not None:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(self._lock_fd)
                    self._lock_fd = None
            key = str(self.path.resolve())
            ref = _WRITERS.get(key)
            if ref is not None and ref() is self:
                del _WRITERS[key]

    close = release_writer

    def __del__(self) -> None:  # pragma: no cover - best effort at interpreter/GC time
        try:
            for fd in (self._writer_fd, self._lock_fd):
                if fd is not None:
                    os.close(fd)
        except Exception:  # noqa: BLE001
            pass

    # ---- writing ------------------------------------------------------------------------------------------
    def _accept(self, event: LedgerEvent) -> str:
        self.ensure_writer()
        if self._broken is not None:
            raise JournalError(f"journal {self.path} is broken after a failed write: {self._broken!r}",
                               maybe_written=False)
        if event.seq != self._last_seq + 1:
            raise JournalError(f"event seq {event.seq} is not the next seq {self._last_seq + 1} on {self.path}",
                               maybe_written=False)
        aid = event.attempt_id
        if aid is not None and event.kind is LedgerEventKind.RESERVE and aid in self._reserved:
            raise JournalError(f"attempt {aid} is already reserved in {self.path}", maybe_written=False)
        if aid is not None and event.kind in TERMINAL_KINDS and aid in self._terminal:
            raise JournalError(f"attempt {aid} already has a terminal event in {self.path}", maybe_written=False)
        line = encode_event(event)
        self._index(event)
        return line

    def submit(self, event: LedgerEvent, *, urgent: bool = False) -> asyncio.Future:
        """Synchronously accept ``event`` (writer role, seq/duplicate checks, queue position) and return the future
        of its fsync. The ledger keeps "check state, assign seq, enqueue" atomic with this; ``append`` is
        ``await submit(event)``. ``urgent`` skips the rest of the linger window. Call from the running event loop."""
        loop = asyncio.get_running_loop()
        line = self._accept(event)
        future: asyncio.Future = loop.create_future()
        self._pending.append((line, future))
        if self._urgent is None:
            self._urgent = asyncio.Event()
        if urgent:
            self._urgent.set()
        if self._flusher is None or self._flusher.done():
            self._flusher = loop.create_task(self._flush_loop())
        return future

    async def append(self, event: LedgerEvent) -> None:
        # Shield: a cancelled caller must not cancel the batch; the line is written either way.
        await asyncio.shield(self.submit(event))

    def append_sync(self, event: LedgerEvent) -> None:
        """Synchronous append + fsync for code that runs without an event loop."""
        if self._pending:
            raise JournalError("append_sync while asynchronous appends are pending", maybe_written=False)
        line = self._accept(event)
        try:
            self._write_and_sync(line.encode("utf-8"))
        except BaseException as exc:
            self._broken = exc
            raise JournalError(f"journal write failed: {exc!r}") from exc

    async def flush(self) -> None:
        """Wait until every pending append is on disk (used at drain)."""
        while self._flusher is not None and not self._flusher.done():
            await asyncio.shield(self._flusher)

    async def _flush_loop(self) -> None:
        while self._pending:
            if self.group_commit_ms and not self._urgent.is_set():
                try:
                    await asyncio.wait_for(self._urgent.wait(), timeout=self.group_commit_ms / 1000.0)
                except asyncio.TimeoutError:
                    pass
            self._urgent.clear()
            batch, self._pending = self._pending, []
            data = "".join(line for line, _ in batch).encode("utf-8")
            try:
                await asyncio.to_thread(self._write_and_sync, data)
            except BaseException as exc:  # noqa: BLE001 - every waiter must learn about the failure
                self._broken = exc
                error = JournalError(f"journal write failed: {exc!r}")
                for _, future in batch + self._pending:
                    if not future.done():
                        future.set_exception(error)
                self._pending = []
                return
            self.batches_written += 1
            self.events_written += len(batch)
            self.max_batch = max(self.max_batch, len(batch))
            for _, future in batch:
                if not future.done():
                    future.set_result(None)

    def _write_and_sync(self, data: bytes) -> None:
        with self._io_lock:
            fd = self._writer_fd
            if fd is None:
                raise JournalError(f"journal {self.path} lost its writer role with appends pending")
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
            self._offset += len(data)

    def _repair_torn_tail(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, "rb+") as handle:
            data = handle.read()
            if not data or data.endswith(b"\n"):
                return 0
            keep = data.rfind(b"\n") + 1
            handle.truncate(keep)
            handle.flush()
            os.fsync(handle.fileno())
            return len(data) - keep
