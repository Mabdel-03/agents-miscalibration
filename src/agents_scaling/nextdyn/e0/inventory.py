"""What the legacy runs actually contain: ``legacy_inventory.parquet`` (spec 7.1; plan §2.6).

The first E0 output, and the one every other leg depends on.  It answers, for each named
artifact in the registry: does it exist, what does it hash to, how many rows does it carry, and
what does it *not* carry.  The last question is the one ARD36 could not answer about itself:
its frozen v1 tables carried ``.empty`` sentinel files and an all-null ``selector_availability``
column, and both read as "the stage ran and found nothing" rather than "the stage never ran".

So every row carries ``completeness_note``: an empty parquet, an ``.empty`` sentinel and an
absent file are three different observations and are recorded as three different notes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import tables
from agents_scaling.nextdyn.e0 import registry as R

EMPTY_SENTINEL: Final[str] = tables.EMPTY_SUFFIX
NOTE_ABSENT: Final[str] = "absent: the registry names it and it is not on disk"
NOTE_EMPTY_SENTINEL: Final[str] = "empty sentinel: the writer ran and produced no rows"
NOTE_EMPTY_TABLE: Final[str] = "zero rows: the file exists and carries no row"
NOTE_DIRECTORY: Final[str] = "directory: counted by entry, never walked for content"
NOTE_OK: Final[str] = "present"


def _parquet_summary(path: Path) -> tuple[int | None, list[str] | None]:
    if path.suffix != ".parquet":
        return None, None
    try:
        import pyarrow.parquet as pq

        handle = pq.ParquetFile(path)
        return int(handle.metadata.num_rows), list(handle.schema_arrow.names)
    except Exception:  # pragma: no cover - a corrupt legacy file is a finding, not a crash
        return None, None


def _json_rows(path: Path) -> int | None:
    if path.suffix not in (".json", ".jsonl"):
        return None
    try:
        if path.suffix == ".jsonl":
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, (list, dict)) else 1
    except (OSError, ValueError):
        return None


def _sibling_sentinels(path: Path) -> list[str]:
    """``.empty`` sentinels beside a named table directory: stages that ran and wrote nothing."""
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.glob(f"*{EMPTY_SENTINEL}"))


def legacy_inventory(sources: R.LegacySources) -> list[dict[str, Any]]:
    """One row per named artifact, with its hash, its size and what it is missing."""
    rows: list[dict[str, Any]] = []
    for resolved in R.resolve_all(sources, hash_files=True):
        study, _, artifact = resolved.name.partition(".")
        n_rows, columns = _parquet_summary(resolved.path)
        if n_rows is None:
            n_rows = _json_rows(resolved.path)
        if not resolved.exists:
            note = NOTE_ABSENT
        elif resolved.is_dir:
            note = NOTE_DIRECTORY
        elif n_rows == 0:
            note = NOTE_EMPTY_TABLE
        else:
            note = NOTE_OK
        sentinels = _sibling_sentinels(resolved.path)
        rows.append({
            "study": study,
            "artifact": artifact,
            "artifact_path": str(resolved.path),
            "declared": resolved.declared,
            "exists": bool(resolved.exists),
            "is_dir": bool(resolved.is_dir),
            "sha256": resolved.sha256,
            "size": resolved.size,
            "n_rows": n_rows,
            "columns_json": json.dumps(columns) if columns else None,
            "role": artifact,
            "completeness_note": (NOTE_EMPTY_SENTINEL if sentinels and note == NOTE_DIRECTORY else note),
            "identifiers_json": json.dumps(sorted((columns or []))[:12]) if columns else None,
            "missing_artifacts_json": json.dumps(sentinels),
            "n_entries": (len(list(resolved.path.iterdir())) if resolved.is_dir else None),
        })
    return rows


def write(out_dir: str | os.PathLike, rows: Sequence[Mapping[str, Any]]) -> Path:
    """``<out_dir>/legacy_inventory.parquet`` in the row order above."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "legacy_inventory.parquet"
    columns = sorted({key for row in rows for key in row})
    tables.write_parquet_rows(path, [dict(r) for r in rows], columns)
    return path


def blocked_sources(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """``<study>.<artifact>`` for every named source that is not on disk."""
    return [f"{r['study']}.{r['artifact']}" for r in rows if not r["exists"]]


__all__ = [
    "EMPTY_SENTINEL", "NOTE_ABSENT", "NOTE_DIRECTORY", "NOTE_EMPTY_SENTINEL", "NOTE_EMPTY_TABLE",
    "NOTE_OK", "blocked_sources", "legacy_inventory", "write",
]
