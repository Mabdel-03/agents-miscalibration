"""Task-public export reader — the ONLY data entry point for generate/select code (WP1).

Spec §3.3/§10.6 (generator/selector containers mount only the task-public export) and
amendment P2 (public fields = source_id, domain, split, task_text, answer_format,
stratum, rank, task_tokens, entry_point, category).  Corrections P1-4: the loader
asserts at runtime that no row carries a protected key.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

from agents_scaling.study.data.layout import PUBLIC_DIR, TASKS_FILE, public_tasks_path  # noqa: F401
from agents_scaling.study.types import Domain, ProtocolError, PublicTask

#: Keys that must never appear in a public row (§10.6 firewall).
PROTECTED_KEYS: frozenset[str] = frozenset(
    {"answer", "json", "rationale", "test", "canonical_solution", "code_prompt", "correct_answer"}
)
PUBLIC_KEYS: frozenset[str] = frozenset(
    {"source_id", "domain", "split", "task_text", "answer_format", "stratum", "rank", "task_tokens", "entry_point", "category"}
)


def assert_public_row(row: dict, *, where: str = "public row") -> None:
    """Raise :class:`ProtocolError` if ``row`` carries any protected or unknown key."""
    keys = set(row)
    leaked = keys & PROTECTED_KEYS
    if leaked:
        raise ProtocolError(f"{where}: protected keys present in the public export: {sorted(leaked)}")
    unknown = keys - PUBLIC_KEYS
    if unknown:
        raise ProtocolError(f"{where}: unexpected keys in the public export: {sorted(unknown)}")


def load_public_tasks(run_root: str | os.PathLike, split: str | None = None) -> list[PublicTask]:
    """Read ``<run_root>/data/public/tasks.jsonl`` (optionally one split), fail-closed.

    Every row is checked against :data:`PROTECTED_KEYS`; duplicate ids or an unknown split
    raise :class:`ProtocolError`.
    """
    if split is not None and split not in ("dev", "main", "reserve"):
        raise ValueError(f"unknown split {split!r}")
    path = public_tasks_path(run_root)
    if not path.exists():
        raise FileNotFoundError(f"public export missing: {path}")
    tasks: list[PublicTask] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ProtocolError(f"{path}:{lineno}: row is not an object")
            assert_public_row(row, where=f"{path}:{lineno}")
            task = PublicTask.from_dict(row)
            if task.source_id in seen:
                raise ProtocolError(f"{path}:{lineno}: duplicate source_id {task.source_id}")
            seen.add(task.source_id)
            if split is None or task.split == split:
                tasks.append(task)
    return tasks


def public_task_index(tasks: Iterable[PublicTask]) -> dict[str, PublicTask]:
    return {t.source_id: t for t in tasks}


def tasks_by_domain(tasks: Iterable[PublicTask]) -> dict[Domain, list[PublicTask]]:
    out: dict[Domain, list[PublicTask]] = {d: [] for d in Domain}
    for task in tasks:
        out[Domain(task.domain)].append(task)
    return out


__all__ = [name for name in globals() if not name.startswith("_")]
