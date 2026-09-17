"""Append-only ledger journal ``state/ledger_journal.jsonl``: one ``dump_record(LedgerEvent)`` per line, fsync'd.

* ``append`` returns only after the line is on disk (``os.fsync``). Appends are group-committed: the first pending
  event starts a batch that lingers ``group_commit_ms`` (so one fsync covers every event that arrives in the
  window), then the batch is written and fsync'd off the event loop. ``group_commit_ms=0`` flushes as soon as the
  previous batch is done. An *urgent* submit (the ledger's ``SENT``, written while the HTTP slot is held) ends the
  linger at once: it still shares an fsync with whatever is pending, but never waits out the window.
* ``seq`` must strictly increase in append order (a second writer on the same file is detected, not interleaved).
* A torn final line (crash mid-write; its ``append`` never returned, so nothing acted on it) is truncated when the
  journal is opened; a malformed line anywhere else is a :class:`JournalError` - the journal is never silently
  repaired in the middle.
* After a failed write the journal is *broken*: every later append raises. The file may hold part of the failed
  batch; replay of such lines is conservative (a reservation without a resolution is charged at recovery).

Config keys consumed: none (``group_commit_ms`` is a constructor argument; 250 ms by contract).
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Iterator

from agent_allocation.contracts import LedgerEvent, dump_record, load_record


class JournalError(RuntimeError):
    def __init__(self, message: str, *, maybe_written: bool = True) -> None:
        super().__init__(message)
        self.maybe_written = maybe_written


def encode_event(event: LedgerEvent) -> str:
    return json.dumps(dump_record(event), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


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
        try:
            event = load_record(LedgerEvent, json.loads(raw.decode("utf-8")))
        except (ValueError, TypeError, KeyError) as exc:
            raise JournalError(f"{path}:{number}: malformed ledger event: {exc}") from exc
        if event.seq <= last_seq:
            raise JournalError(f"{path}:{number}: seq {event.seq} does not increase (last {last_seq})")
        last_seq = event.seq
        yield event


class LedgerJournal:
    def __init__(self, path: Path, *, group_commit_ms: int = 250) -> None:
        if group_commit_ms < 0:
            raise ValueError("group_commit_ms must be >= 0")
        self.path = Path(path)
        self.group_commit_ms = int(group_commit_ms)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._io_lock = threading.Lock()
        self._pending: list[tuple[str, asyncio.Future]] = []
        self._flusher: asyncio.Task | None = None
        self._urgent: asyncio.Event | None = None
        self._broken: BaseException | None = None
        self.torn_tail_bytes = self._repair_torn_tail()
        self._last_seq = 0
        for event in self.replay():
            self._last_seq = event.seq
        self.batches_written = 0
        self.events_written = 0
        self.max_batch = 0

    # ---- reading ------------------------------------------------------------------------------------------
    def replay(self) -> Iterator[LedgerEvent]:
        return replay_events(self.path)

    @property
    def last_seq(self) -> int:
        return self._last_seq

    def next_seq(self) -> int:
        """The seq the next appended event must carry (callers assign it synchronously before ``append``)."""
        return self._last_seq + 1

    @property
    def broken(self) -> BaseException | None:
        return self._broken

    # ---- writing ------------------------------------------------------------------------------------------
    def _accept(self, event: LedgerEvent) -> str:
        if self._broken is not None:
            raise JournalError(f"journal {self.path} is broken after a failed write: {self._broken!r}",
                               maybe_written=False)
        if event.seq != self._last_seq + 1:
            raise JournalError(f"event seq {event.seq} is not the next seq {self._last_seq + 1} "
                               "(two writers on one journal?)", maybe_written=False)
        line = encode_event(event)
        self._last_seq = event.seq
        return line

    def submit(self, event: LedgerEvent, *, urgent: bool = False) -> asyncio.Future:
        """Synchronously accept ``event`` (seq check, queue position) and return the future of its fsync.

        Callers that must keep "check state, assign seq, enqueue" atomic (the ledger) use this and then await the
        future; ``append`` is ``await submit(event)``. ``urgent`` skips the rest of the linger window. Must be called
        from the running event loop.
        """
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
        """Synchronous append + fsync for recovery code that runs without an event loop."""
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
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                view = memoryview(data)
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)

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
