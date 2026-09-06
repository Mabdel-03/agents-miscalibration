"""Salted-hash simple random sampling, split assignment and nested panels (WP1).

Spec §3.1 ("simple random sampling by a frozen salted hash within each superdomain"; no
tiny resampling strata), §3.3 (same-stratum reserve order frozen before outcomes; keep a
reserve), §6.2 ("balanced source-item-hashed subset": nested panels are rank prefixes).
Corrections P1-8: plain SRS per superdomain — HLE over all eligible ids with NO
Gold/Revision balancing (strata are covariates), BCB over all 1,140 — dev = first
``items.dev`` per domain in rank order, main = the next ``items.main``, reserve = rest.

``rank_key(salt, source_id) = HMAC-SHA256(salt, b"rank:" + source_id)``; sorting by these
bytes gives one frozen order per domain.  ``PublicTask.rank`` is the 0-based position
inside its split (so every panel of size n is exactly ``rank < n`` on main).
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

from agents_scaling.study.config import StudyConfig
from agents_scaling.study.types import Domain, PublicTask

SPLITS: tuple[str, ...] = ("dev", "main", "reserve")
RANK_PREFIX = b"rank:"


class SplitError(ValueError):
    """Not enough eligible items, duplicate ids or an inconsistent request."""


def rank_key(salt: bytes, source_id: str) -> bytes:
    """``HMAC-SHA256(salt, b"rank:" + source_id)``; compare as bytes."""
    if not isinstance(salt, (bytes, bytearray)) or not salt:
        raise ValueError("salt must be non-empty bytes")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a non-empty str")
    return hmac.new(bytes(salt), RANK_PREFIX + source_id.encode("utf-8"), hashlib.sha256).digest()


def domain_rank_order(tasks: Iterable[PublicTask], salt: bytes) -> dict[Domain, list[PublicTask]]:
    """Group by superdomain and sort each group by :func:`rank_key` (ties impossible: ids unique)."""
    groups: dict[Domain, list[PublicTask]] = {d: [] for d in Domain}
    seen: set[str] = set()
    for task in tasks:
        if task.source_id in seen:
            raise SplitError(f"duplicate source_id {task.source_id!r}")
        seen.add(task.source_id)
        groups[Domain(task.domain)].append(task)
    return {d: sorted(group, key=lambda t: rank_key(salt, t.source_id)) for d, group in groups.items()}


def _counts(cfg: StudyConfig, split: str) -> dict[Domain, int]:
    block = getattr(cfg.items, split)
    return {Domain.HLE: int(block.hle), Domain.BCB: int(block.bcb)}


def assign_splits(tasks: Sequence[PublicTask], cfg: StudyConfig) -> list[PublicTask]:
    """Return new ``PublicTask`` objects with ``split`` and within-split ``rank`` filled.

    Output order: HLE then BCB, each in rank order (dev, then main, then reserve).  Raises
    :class:`SplitError` when a domain has fewer eligible items than ``dev + main``.
    """
    dev = _counts(cfg, "dev")
    main = _counts(cfg, "main")
    ordered = domain_rank_order(tasks, cfg.split_salt)
    out: list[PublicTask] = []
    for domain in Domain:
        group = ordered[domain]
        need = dev[domain] + main[domain]
        if len(group) < need:
            raise SplitError(f"{domain.value}: {len(group)} eligible items < dev+main = {need}")
        cuts = ((0, dev[domain], "dev"), (dev[domain], need, "main"), (need, len(group), "reserve"))
        for start, stop, split in cuts:
            for rank, task in enumerate(group[start:stop]):
                out.append(replace(task, split=split, rank=rank))
    return out


def split_assignment(tasks: Sequence[PublicTask], cfg: StudyConfig) -> dict[str, tuple[str, int]]:
    """``{source_id: (split, rank)}`` view of :func:`assign_splits`."""
    return {t.source_id: (t.split, t.rank) for t in assign_splits(tasks, cfg)}


def panel_items(main_tasks: Iterable[PublicTask], n_per_domain: int) -> list[PublicTask]:
    """The nested panel of size ``n_per_domain`` per domain: main items with ``rank < n``.

    Panels are prefixes of the main rank order, so every smaller panel is a subset of every
    larger one (§6.2).  Raises when ``main_tasks`` contains a non-main item or a domain has
    fewer than ``n_per_domain`` items.
    """
    if not isinstance(n_per_domain, int) or n_per_domain <= 0:
        raise ValueError("n_per_domain must be a positive int")
    per_domain: dict[Domain, list[PublicTask]] = {d: [] for d in Domain}
    for task in main_tasks:
        if task.split != "main":
            raise SplitError(f"{task.source_id} is in split {task.split!r}, not main")
        per_domain[Domain(task.domain)].append(task)
    out: list[PublicTask] = []
    for domain in Domain:
        group = sorted(per_domain[domain], key=lambda t: t.rank)
        if len(group) < n_per_domain:
            raise SplitError(f"{domain.value}: main has {len(group)} items < panel size {n_per_domain}")
        if [t.rank for t in group] != list(range(len(group))):
            raise SplitError(f"{domain.value}: main ranks are not a contiguous 0..n-1 sequence")
        out.extend(group[:n_per_domain])
    return out


def split_summary(tasks: Sequence[PublicTask], panels: Mapping[str, int]) -> dict:
    """JSON-ready description of the assignment (written to ``splits.json``)."""
    domains: dict[str, dict[str, list[str]]] = {}
    for domain in Domain:
        block = {split: [] for split in SPLITS}
        for task in sorted((t for t in tasks if t.domain == domain), key=lambda t: (SPLITS.index(t.split), t.rank)):
            block[task.split].append(task.source_id)
        domains[domain.value] = block
    return {
        "rank_rule": "HMAC-SHA256(split_salt, b'rank:' + source_id); rank = position inside split",
        "sampling": "plain salted-hash SRS per superdomain; Gold/Revision are covariates (P1-8)",
        "counts": {d: {s: len(ids) for s, ids in block.items()} for d, block in domains.items()},
        "panels": dict(panels),
        "domains": domains,
    }


__all__ = [name for name in globals() if not name.startswith("_")]
