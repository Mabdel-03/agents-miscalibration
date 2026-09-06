"""BigCodeBench v0.1.4 adapter (WP1).

Spec §3.2: ``bigcode/bigcodebench`` split ``v0.1.4`` (1,140 tasks), ``instruct_prompt`` is
the task text; official ``test`` and ``canonical_solution`` belong only to the isolated
evaluator (§10.6, amendment P2).  Architecture §1.4.  The local parquet was materialized
once from the pinned revision (``data/raw/bigcodebench_v0.1.4.meta.json`` records it).

Every BCB task is eligible (1,140 rows); the §4.3 envelope is still asserted so the
predicate is the same source-only rule for both superdomains.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents_scaling.study.types import TASK_TOKENS_CAP, Domain, PublicTask, _Record

SOURCE_PREFIX = "bcb:"
ANSWER_FORMAT = "code"
STRATUM = "bcb"
EXPECTED_ROWS = 1140
REQUIRED_COLUMNS: tuple[str, ...] = (
    "task_id",
    "instruct_prompt",
    "canonical_solution",
    "code_prompt",
    "test",
    "entry_point",
    "libs",
)
REASON_ENVELOPE = "envelope"


@dataclass(frozen=True)
class ProtectedBcb(_Record):
    """Evaluator-only BCB record (§3.2).  ``libs`` is kept here for the container driver."""

    source_id: str
    task_id: str
    entry_point: str
    test: str
    canonical_solution: str
    code_prompt: str
    libs: tuple[str, ...]


class BcbRowError(ValueError):
    """A parquet row does not have the frozen shape."""


def source_id_for(task_id: str) -> str:
    return f"{SOURCE_PREFIX}{task_id}"


def load_bcb_rows(parquet_path: str | Path, *, expected_rows: int | None = EXPECTED_ROWS) -> list[dict[str, Any]]:
    """Read the pinned parquet in file order; refuse a row count other than ``expected_rows``."""
    import pyarrow.parquet as pq

    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(path)
    table = pq.read_table(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in table.column_names]
    if missing:
        raise BcbRowError(f"{path} lacks columns {missing}")
    if expected_rows is not None and table.num_rows != expected_rows:
        raise BcbRowError(f"{path} has {table.num_rows} rows, expected {expected_rows}")
    return table.to_pylist()


def parse_libs(value: Any) -> tuple[str, ...]:
    """``libs`` is a Python-literal list string (``"['random', 'itertools']"``)."""
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    if not isinstance(value, str):
        raise BcbRowError(f"libs must be a str or list, got {type(value).__name__}")
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise BcbRowError(f"libs is not a Python list literal: {value!r}") from exc
    if not isinstance(parsed, (list, tuple)) or not all(isinstance(v, str) for v in parsed):
        raise BcbRowError(f"libs must be a list of str: {value!r}")
    return tuple(parsed)


def _require_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BcbRowError(f"row {row.get('task_id')!r}: {key} must be a non-empty str")
    return value


def bcb_public_task(row: Mapping[str, Any], task_tokens: int) -> PublicTask:
    """Task-public view: ``instruct_prompt`` + ``entry_point`` (split/rank assigned later)."""
    task_id = _require_str(row, "task_id")
    return PublicTask(
        source_id=source_id_for(task_id),
        domain=Domain.BCB,
        split="unassigned",
        task_text=_require_str(row, "instruct_prompt"),
        answer_format=ANSWER_FORMAT,
        stratum=STRATUM,
        rank=-1,
        task_tokens=int(task_tokens),
        entry_point=_require_str(row, "entry_point"),
        category=None,
    )


def bcb_protected(row: Mapping[str, Any]) -> ProtectedBcb:
    task_id = _require_str(row, "task_id")
    return ProtectedBcb(
        source_id=source_id_for(task_id),
        task_id=task_id,
        entry_point=_require_str(row, "entry_point"),
        test=_require_str(row, "test"),
        canonical_solution=_require_str(row, "canonical_solution"),
        code_prompt=_require_str(row, "code_prompt"),
        libs=parse_libs(row.get("libs")),
    )


@dataclass(frozen=True)
class BcbBuild:
    tasks: list[PublicTask]
    protected: dict[str, ProtectedBcb]
    exclusions: list[dict[str, Any]]


def build_bcb(rows: Sequence[Mapping[str, Any]], tokenizer, *, cap: int = TASK_TOKENS_CAP) -> BcbBuild:
    """Build both views; the only exclusion rule is the §4.3 envelope (duplicates are an error)."""
    from agents_scaling.study.inference.tokens import count_tokens

    tasks: list[PublicTask] = []
    protected: dict[str, ProtectedBcb] = {}
    exclusions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        task_id = _require_str(row, "task_id")
        if task_id in seen:
            raise BcbRowError(f"duplicate BCB task_id {task_id!r}")
        seen.add(task_id)
        task_tokens = count_tokens(_require_str(row, "instruct_prompt"), tokenizer)
        if task_tokens > cap:
            exclusions.append(
                {
                    "source_id": source_id_for(task_id),
                    "domain": Domain.BCB.value,
                    "reason": REASON_ENVELOPE,
                    "verified_class": None,
                    "task_tokens": task_tokens,
                }
            )
            continue
        task = bcb_public_task(row, task_tokens)
        tasks.append(task)
        protected[task.source_id] = bcb_protected(row)
    return BcbBuild(tasks=tasks, protected=protected, exclusions=exclusions)


__all__ = [name for name in globals() if not name.startswith("_")]
