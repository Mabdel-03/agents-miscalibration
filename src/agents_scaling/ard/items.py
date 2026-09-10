"""Runtime reader for the frozen ARD36 item table (WP3).

Spec §18 (``items.parquet`` field list), §5.1 ("No option key is exposed"), §3.3/§10.6 of
the study architecture (generator and controller lanes mount only the task-public export).
Plan §5.1: "gold (``answer``, ``answer_index``, permuted key, ``solution``, ``cot_content``)
lives only in the 0700/0600 protected namespace read by ``ard/graders``."

This module is **runtime-safe by construction**: its import closure is the standard
library plus ``pyarrow`` plus :mod:`agents_scaling.ard.contract`.  It never imports
``agents_scaling.benchmarks.loaders``, ``datasets`` or any grader, and
:func:`load_ard_tasks` asserts at runtime that no protected key reached the public table —
the firewall is checked on the data, not only on the import graph.

The writer is :mod:`agents_scaling.ard.items_build` (evaluator lane).  The path constants
live here so that the writer, the graders and the runtime agree on one layout without the
runtime importing either of the other two.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

from agents_scaling.ard.contract import (
    FAMILIES,
    MMLU_REQUIRED_OPTIONS,
    SPLITS,
    Family,
)

# --------------------------------------------------------------------------- layout

DATA_DIR = "data"
PUBLIC_DIR = "public"
PROTECTED_DIR = "protected"
ITEMS_FILE = "items.parquet"
ITEMS_SUMMARY_FILE = "items_summary.json"
RESERVE_FILE = "items_reserve.json"
MATH_GOLD_FILE = "math_gold.jsonl"
MMLU_GOLD_FILE = "mmlu_gold.jsonl"
PROTECTED_DIR_MODE = 0o700
PROTECTED_FILE_MODE = 0o600


def data_dir(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / DATA_DIR


def public_dir(run_root: str | os.PathLike) -> Path:
    return data_dir(run_root) / PUBLIC_DIR


def items_path(run_root: str | os.PathLike) -> Path:
    return public_dir(run_root) / ITEMS_FILE


def protected_dir(run_root: str | os.PathLike) -> Path:
    return data_dir(run_root) / PROTECTED_DIR


# --------------------------------------------------------------------------- firewall

#: Keys that must never appear in the public item table (spec §5.1, §18).
PROTECTED_KEYS: frozenset[str] = frozenset({
    "answer", "answer_index", "answer_key", "permuted_key", "correct_answer", "solution",
    "cot_content", "gold", "reference", "is_correct", "correct", "label", "constructed_C0",
    "root_pass_rate",
})


@dataclass(frozen=True)
class ArdTask:
    """One public ARD36 task row.  Every field is safe for a generator worker."""

    item_id: str
    family: str
    dataset_revision: str
    source_id: str
    split: str
    group_id: str
    prompt_hash: str
    option_permutation: str          # JSON list: displayed position -> source option index
    gold_location: str               # pointer into the protected namespace; not the gold
    grader_hash: str
    subject: str
    declared_difficulty_metadata: str
    n_options: int
    answer_format: str               # "exactMatch" | "multipleChoice"
    stratum: str
    rank: int                        # position within (family, split) in frozen rank order
    task_text: str                   # the rendered task statement (already permuted)
    options_json: str                # JSON list of the displayed option texts ("[]" for MATH)

    @property
    def options(self) -> list[str]:
        return list(json.loads(self.options_json or "[]"))

    @property
    def permutation(self) -> list[int]:
        return list(json.loads(self.option_permutation or "[]"))

    @classmethod
    def from_row(cls, row: dict) -> "ArdTask":
        assert_public_row(row)
        return cls(**{field.name: row[field.name] for field in fields(cls)})

    def to_row(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


#: The WP0 interface names this type ``PublicTask``; the concrete name avoids a clash with
#: the frozen ``study.types.PublicTask``, which has a different (HLE/BCB) field set.
PublicTask = ArdTask

#: The exact column set of ``items.parquet``.
PUBLIC_KEYS: frozenset[str] = frozenset(field.name for field in fields(ArdTask))
#: Column order used by the writer, so the parquet schema is deterministic.
PUBLIC_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(ArdTask))


class ItemsError(ValueError):
    """The public item table is missing, malformed or leaks a protected key."""


def assert_public_row(row: dict, *, where: str = "items row") -> None:
    """Raise :class:`ItemsError` if ``row`` carries a protected or unknown key."""
    keys = set(row)
    leaked = keys & PROTECTED_KEYS
    if leaked:
        raise ItemsError(f"{where}: protected keys present in the public item table: {sorted(leaked)}")
    unknown = keys - PUBLIC_KEYS
    if unknown:
        raise ItemsError(f"{where}: unexpected keys in the public item table: {sorted(unknown)}")
    missing = PUBLIC_KEYS - keys
    if missing:
        raise ItemsError(f"{where}: missing public columns: {sorted(missing)}")


def _read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq  # local import: keeps ``import agents_scaling.ard`` light

    table = pq.read_table(path)
    leaked = set(table.column_names) & PROTECTED_KEYS
    if leaked:
        raise ItemsError(f"{path}: protected columns in the public item table: {sorted(leaked)}")
    return table.to_pylist()


def load_ard_tasks(
    run_root: str | os.PathLike,
    split: str | None = None,
    family: Family | str | None = None,
) -> dict[str, ArdTask]:
    """``{item_id: ArdTask}`` from ``<run_root>/data/public/items.parquet``, fail-closed.

    Every row is checked against :data:`PROTECTED_KEYS` before it becomes a task, so a
    rebuilt or hand-edited table that carries gold is rejected here rather than reaching a
    prompt.  Optional ``split``/``family`` filters are applied after the check.
    """
    if split is not None and split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    family_key = family.value if isinstance(family, Family) else family
    if family_key is not None and family_key not in {f.value for f in FAMILIES}:
        raise ValueError(f"unknown family {family!r}")
    path = items_path(run_root)
    if not path.exists():
        raise FileNotFoundError(f"public item table missing: {path}")
    tasks: dict[str, ArdTask] = {}
    for index, row in enumerate(_read_parquet(path)):
        task = ArdTask.from_row(row)
        if task.item_id in tasks:
            raise ItemsError(f"{path}:{index}: duplicate item_id {task.item_id}")
        if task.family == Family.MMLUPRO.value and task.n_options != MMLU_REQUIRED_OPTIONS:
            raise ItemsError(
                f"{path}:{index}: {task.item_id} has {task.n_options} options; ARD36 uses "
                f"only the {MMLU_REQUIRED_OPTIONS}-option MMLU-Pro rows"
            )
        tasks[task.item_id] = task
    if not tasks:
        raise ItemsError(f"{path}: the public item table is empty")
    return {
        item_id: task
        for item_id, task in tasks.items()
        if (split is None or task.split == split) and (family_key is None or task.family == family_key)
    }


def tasks_by_split(tasks: dict[str, ArdTask]) -> dict[str, list[ArdTask]]:
    out: dict[str, list[ArdTask]] = {name: [] for name in SPLITS}
    for task in tasks.values():
        out[task.split].append(task)
    for group in out.values():
        group.sort(key=lambda task: (task.family, task.rank))
    return out


def group_ids(tasks: dict[str, ArdTask]) -> dict[str, str]:
    """``{item_id: group_id}``.  Every variant of a source item shares its group."""
    return {item_id: task.group_id for item_id, task in tasks.items()}


__all__ = [
    "ArdTask",
    "DATA_DIR",
    "ITEMS_FILE",
    "ITEMS_SUMMARY_FILE",
    "ItemsError",
    "MATH_GOLD_FILE",
    "MMLU_GOLD_FILE",
    "PROTECTED_DIR",
    "PROTECTED_DIR_MODE",
    "PROTECTED_FILE_MODE",
    "PROTECTED_KEYS",
    "PUBLIC_COLUMNS",
    "PUBLIC_DIR",
    "PUBLIC_KEYS",
    "PublicTask",
    "RESERVE_FILE",
    "assert_public_row",
    "data_dir",
    "group_ids",
    "items_path",
    "load_ard_tasks",
    "protected_dir",
    "public_dir",
    "tasks_by_split",
]
