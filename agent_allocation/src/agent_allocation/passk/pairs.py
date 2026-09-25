"""Append-only, gold-free Math-Verify pair-verdict cache (contract §5.2).

One row per unordered pair of normalized answer texts (``VoteKey.text``) of one item: ``text_a < text_b``
lexicographically, so the cache is order-invariant by construction. ``verdict`` is the gold-blind symmetric
``grading.text.math_equiv.math_equivalent`` (5 s wall clock, True only when both verify directions hold, None on a
parse failure, timeout or evaluator exception). A stored None is final: :meth:`PairCache.fill` never recomputes a
pair that is already present, whatever its verdict.

Only MATH items have pairs; GPQA (OPTION) classes are option identity. Within an item, only texts whose exact vote
keys differ are ever compared by ``selection.plurality.build_classes`` (texts sharing an exact key are one group),
so :func:`enumerate_pairs` emits exactly those pairs.

Parallelism: pairs are evaluated in a spawn-context process pool, one pair per task, so a pathological item cannot
serialize the fill. Each task runs on the worker process's main thread, where SIGALRM is available, so the 5 s
limit is enforced in-process (no isolated child). Results are flushed in chunks: the parquet is rewritten
atomically (temp + ``os.replace``) under an exclusive ``flock`` with the old rows as an unchanged prefix, so an
interrupted fill loses at most one chunk and never corrupts stored rows.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import pandas as pd

__all__ = [
    "PAIR_COLUMNS", "TIMEOUT_S", "MissingPair", "Pair", "PairCache", "canonical_pair", "enumerate_pairs",
    "math_verify_version", "pair_sha",
]

TIMEOUT_S = 5.0
PAIR_COLUMNS = ["item_id", "text_a", "text_b", "pair_sha", "verdict", "math_verify_version", "timeout_s",
                "elapsed_s", "computed_at"]
_FLUSH_EVERY = 500


class MissingPair(KeyError):
    """A pair the class build needs is not in the cache (fill it first; never compute inline)."""


@dataclass(frozen=True, order=True)
class Pair:
    item_id: str
    text_a: str
    text_b: str

    @property
    def sha(self) -> str:
        return pair_sha(self.item_id, self.text_a, self.text_b)


def canonical_pair(item_id: str, a: str, b: str) -> Pair:
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError("pair texts must be str")
    lo, hi = (a, b) if a <= b else (b, a)
    return Pair(item_id, lo, hi)


def pair_sha(item_id: str, a: str, b: str) -> str:
    lo, hi = (a, b) if a <= b else (b, a)
    blob = json.dumps(["passk_cont_v1/pair", item_id, lo, hi], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def math_verify_version() -> str:
    from agent_allocation.grading.text.math_equiv import dependency_versions
    return dependency_versions().get("math-verify", "absent")


def enumerate_pairs(votes: Iterable[tuple[str, str, str]]) -> list[Pair]:
    """Pairs needed by any class build over these votes: ``(item_id, exact_key, text)`` triples of MATH votes.

    For each item, every unordered pair of distinct texts whose exact keys differ. Sorted, deduplicated.
    """
    by_item: dict[str, dict[str, str]] = {}           # item -> text -> exact key
    for item_id, exact, text in votes:
        if exact is None or text is None:
            continue
        seen = by_item.setdefault(item_id, {})
        prior = seen.get(text)
        if prior is not None and prior != exact:
            raise ValueError(f"{item_id}: text {text!r} carries two exact keys ({prior!r}, {exact!r})")
        seen[text] = exact
    out: set[Pair] = set()
    for item_id, texts in by_item.items():
        ordered = sorted(texts)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                if texts[a] != texts[b]:
                    out.add(Pair(item_id, a, b))
    return sorted(out)


def _evaluate(task: tuple[str, str, str, float]) -> tuple[str, str, str, bool | None, float]:
    """Worker body: runs on the spawned worker's main thread, so the SIGALRM limit is in-process."""
    item_id, a, b, timeout_s = task
    from agent_allocation.grading.text.math_equiv import math_equivalent
    started = time.monotonic()
    verdict = math_equivalent(a, b, timeout_s=timeout_s)
    return item_id, a, b, verdict, time.monotonic() - started


def _evaluate_isolated(task: tuple[str, str, str, float]) -> tuple[str, str, str, bool | None, float]:
    """Worker body with a hard kill: the pair runs in the worker's persistent isolated child (``_isolated``), which
    is killed at ``timeout_s + grace`` even inside C code a signal cannot interrupt (a stall = None, unknown)."""
    item_id, a, b, timeout_s = task
    from agent_allocation.grading.text._isolated import shared_evaluator
    started = time.monotonic()
    try:
        value = shared_evaluator().call("math_equivalent", (a, b), {"timeout_s": timeout_s}, timeout_s)
    except BaseException as exc:  # noqa: BLE001 - timeout, outage or start failure: unknown
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        value = None
    if value is not None and not isinstance(value, bool):
        value = None
    return item_id, a, b, value, time.monotonic() - started


def _warm() -> None:
    from agent_allocation.grading.text.math_equiv import _math_verify
    _math_verify()


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _empty() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="object") for c in PAIR_COLUMNS})
    return frame


class PairCache:
    """The pair-verdict cache at ``path`` (``<run_root>/catalog/pair_verdicts.parquet``)."""

    def __init__(self, path: str | Path, *, evaluator: Callable[[tuple], tuple] | None = None) -> None:
        self.path = Path(path)
        self._evaluator = evaluator                    # tests may inject a picklable top-level function
        self._verdicts: dict[tuple[str, str, str], bool | None] = {}
        self._n_rows = 0
        self.reload()

    # ------------------------------------------------------------------ reading
    def reload(self) -> None:
        frame = self.frame()
        verdicts: dict[tuple[str, str, str], bool | None] = {}
        for item_id, a, b, sha, verdict in zip(frame["item_id"], frame["text_a"], frame["text_b"],
                                               frame["pair_sha"], frame["verdict"]):
            if not a < b:
                raise ValueError(f"{self.path}: row {item_id} has text_a >= text_b")
            if sha != pair_sha(item_id, a, b):
                raise ValueError(f"{self.path}: pair_sha mismatch for {item_id}")
            key = (item_id, a, b)
            value = None if verdict is None or (isinstance(verdict, float) and verdict != verdict) else bool(verdict)
            if key in verdicts:
                raise ValueError(f"{self.path}: duplicate pair row for {item_id}")
            verdicts[key] = value
        self._verdicts = verdicts
        self._n_rows = len(frame)

    def frame(self) -> pd.DataFrame:
        if not self.path.exists():
            return _empty()
        frame = pd.read_parquet(self.path)
        missing = [c for c in PAIR_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{self.path}: missing columns {missing}")
        frame["verdict"] = frame["verdict"].astype("object").where(frame["verdict"].notna(), None)
        return frame[PAIR_COLUMNS]

    def __len__(self) -> int:
        return len(self._verdicts)

    def __contains__(self, pair: Pair) -> bool:
        return (pair.item_id, pair.text_a, pair.text_b) in self._verdicts

    def lookup(self, item_id: str, a: str, b: str) -> bool | None:
        if a == b:
            return True
        pair = canonical_pair(item_id, a, b)
        key = (pair.item_id, pair.text_a, pair.text_b)
        if key not in self._verdicts:
            raise MissingPair(f"{item_id}: no cached verdict for ({pair.text_a!r}, {pair.text_b!r})")
        return self._verdicts[key]

    def equivalence(self, item_id: str) -> Callable[[str, str], bool | None]:
        """A ``math_equivalent``-shaped callable served from the cache for one item (raises ``MissingPair``)."""
        def equivalent(a: str, b: str) -> bool | None:
            return self.lookup(item_id, a, b)
        return equivalent

    def missing(self, pairs: Iterable[Pair]) -> list[Pair]:
        return sorted({p for p in pairs if (p.item_id, p.text_a, p.text_b) not in self._verdicts})

    # ------------------------------------------------------------------ writing
    def _append(self, rows: list[dict]) -> None:
        if not rows:
            return
        with _locked(self.path):
            current = self.frame()
            if len(current) < self._n_rows:
                raise RuntimeError(f"{self.path}: the cache shrank on disk ({len(current)} < {self._n_rows})")
            have = set(zip(current["item_id"], current["text_a"], current["text_b"]))
            fresh = [r for r in rows if (r["item_id"], r["text_a"], r["text_b"]) not in have]   # never overwrite
            if fresh:
                add = pd.DataFrame(fresh, columns=PAIR_COLUMNS)
                add["verdict"] = add["verdict"].astype("object")
                out = pd.concat([current, add], ignore_index=True) if len(current) else add
                out["verdict"] = out["verdict"].astype("object")
                tmp = self.path.with_name(self.path.name + f".tmp{os.getpid()}")
                self.path.parent.mkdir(parents=True, exist_ok=True)
                out.to_parquet(tmp, index=False)
                os.replace(tmp, self.path)
        self.reload()

    def fill(self, pairs: Iterable[Pair], workers: int = 1, *, timeout_s: float = TIMEOUT_S,
             flush_every: int = _FLUSH_EVERY, progress: Callable[[int, int], None] | None = None,
             isolate: bool = False) -> dict:
        """Compute and append every pair not already cached (a stored None is final). Returns a summary.

        ``isolate=True`` runs each pair in a hard-killable child of its worker (``_evaluate_isolated``): slower, but
        a stall inside C code cannot hang the fill."""
        todo = self.missing(pairs)
        version = math_verify_version()
        evaluator = self._evaluator or (_evaluate_isolated if isolate else _evaluate)
        started = time.monotonic()
        buffer: list[dict] = []
        counts = {"requested_new": len(todo), "true": 0, "false": 0, "none": 0}

        def record(result: tuple) -> None:
            item_id, a, b, verdict, elapsed = result
            if verdict is not None and not isinstance(verdict, bool):
                raise TypeError(f"verdict must be bool | None, got {type(verdict).__name__}")
            counts["none" if verdict is None else ("true" if verdict else "false")] += 1
            buffer.append({"item_id": item_id, "text_a": a, "text_b": b, "pair_sha": pair_sha(item_id, a, b),
                           "verdict": verdict, "math_verify_version": version, "timeout_s": float(timeout_s),
                           "elapsed_s": float(elapsed),
                           "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            if len(buffer) >= flush_every:
                self._append(buffer)
                buffer.clear()

        tasks = [(p.item_id, p.text_a, p.text_b, float(timeout_s)) for p in todo]
        done = 0
        if tasks and workers <= 1:
            for task in tasks:
                record(evaluator(task))
                done += 1
                if progress:
                    progress(done, len(tasks))
        elif tasks:
            import multiprocessing
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_warm) as pool:
                futures = [pool.submit(evaluator, task) for task in tasks]
                for future in as_completed(futures):
                    record(future.result())
                    done += 1
                    if progress:
                        progress(done, len(tasks))
        self._append(buffer)
        counts["elapsed_s"] = time.monotonic() - started
        counts["rows"] = len(self)
        return counts


def math_votes_from_frame(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """``(item_id, exact, text)`` MATH votes from a draws frame (``vote_exact_key``/``vote_text``, GPQA skipped)."""
    rows = frame[(frame["benchmark"] != "gpqa_diamond") & frame["vote_exact_key"].notna()
                 & frame["vote_text"].notna()]
    return list(zip(rows["item_id"], rows["vote_exact_key"], rows["vote_text"]))


def pairs_for_scopes(scopes: Mapping[str, Sequence[tuple[str, str]]]) -> list[Pair]:
    """Pairs for explicit scopes: ``{item_id: [(exact, text), ...]}``."""
    return enumerate_pairs((item, exact, text) for item, votes in scopes.items() for exact, text in votes)
