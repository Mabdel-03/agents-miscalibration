"""Primary-key uniqueness assertions over a compacted ARD36 tables directory.

The audit of this run found that **no table's primary key is asserted anywhere** — not in
``compact.py``, not in the analysis sweeps that join on those keys.  Three of the four tables
the sweeps join by are not in fact unique on that key, so every join in the analysis layer
silently multiplied rows, and a retraction of roughly half the study's own claims followed.
This module is the assertion that was missing.  It is a *checker*, never a repair: it does not
write, and it is expected to report violations against the frozen v1 tables rather than refuse
to run on them — v1 is the input that proves the checker works.

Measured by this module's own CLI against the frozen v1 directory
(``<results>/ard36/tables``, 2026-09-12):

* ``decisions.parquet`` — 67,104 rows, 52,768 distinct ``decision_id``; 2,488 duplicate groups
  carrying 14,336 excess rows (21.4% of the table).  Every duplicated ``decision_id`` is a
  ``round == 0`` / ``phase == "root"`` row; group sizes are 3 (×88), 4 (×320), 6 (×840),
  8 (×1080) and 10 (×160), mean 6.76 over duplicated groups, max 10.  The brief's "round-0
  multiplicity mean 5.46" reproduces only under the row-weighted convention
  (127,872 / 23,424 = 5.459, the multiplicity the average round-0 *row* sits in); over distinct
  round-0 ids the mean is 2.58.  ``aliased`` does **not** identify these rows — it is True on
  21,516 of 67,104 rows overall but on only 15,129 of the 16,824 rows inside a duplicate group,
  so 1,695 duplicated rows would survive a dedupe keyed on it.  No dedupe rule is proposed here
  for that reason; picking the survivor is a repair decision, and this module only reports.
* ``labels.parquet`` — 51,360 rows, 51,360 distinct ``decision_id``, 0 duplicate groups.  The
  one clean table of the four, and the control that shows a passing check is reachable.
* ``episodes.parquet`` — 4,888 rows, 4,240 distinct ``episode_id``; 624 duplicate groups and
  648 excess rows (616 ids on 2 rows, 8 ids on 5).  The brief describes every pair as one
  ``EPISODE.PRIVATE`` row plus one ``ROOTS`` row: that holds for 472 of the 616 pairs, and the
  other 144 are two ``FORK`` rows sharing an ``episode_id``, which the brief does not mention.
  The 8 five-row groups (D-11) are one ``EPISODE`` row plus **four** ``ROOTS`` rows.
* ``round_summary.parquet`` — 13,336 rows, 12,832 distinct ``(episode_id, round)``; 480
  duplicate groups (472 of size 2, 8 of size 5) and 504 excess rows, every one at ``round == 0``
  with ``protocol == "PRIVATE"``.  Their episode ids are exactly the 472 + 8 ids from
  ``episodes`` above; the 144 FORK/FORK pairs do not reach this table.

**``cell_kind`` does not exist in v1** — no parquet file in the frozen directory carries the
column (checked across all 17).  The keys declared here are the v2 keys, because the
EPISODE/ROOTS collisions above are exactly the collisions a kind discriminator resolves.
Against a directory without the column the check falls back to the key with ``cell_kind``
dropped and records the fallback in ``note``; the fallback is reported, never silently applied.
Whether ``cell_kind`` actually makes ``episodes`` unique in v2 is **not** verifiable from v1 and
is not asserted here — the 144 FORK/FORK pairs share a kind, so at minimum those would survive.

Two tables are deliberately left unkeyed.  ``round_selection`` is finer-grained than
``(episode_id, round)`` by construction and is not unique even on
``(episode_id, round, selector, pool)`` (7,200 duplicate groups in 200,040 rows), so declaring a
key for it would be guessing at its grain.  ``minority_frame`` is one row per
(episode, round, agent) and is derived rather than joined against.

    PYTHONPATH=src python -m agents_scaling.ard.keys --tables <run_root>/tables
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Hashable, Mapping, Sequence

#: The v2 discriminator column.  Declared in every episode-grained key below and absent from
#: every v1 table, which is why the fallback path is the one v1 actually exercises.
CELL_KIND: Final[str] = "cell_kind"

#: Key columns that may be dropped when the table does not carry them.  Only ``cell_kind`` is
#: optional: a table missing ``decision_id`` or ``episode_id`` is not the table we think it is,
#: and the check is reported unverifiable rather than quietly weakened.
OPTIONAL_KEY_COLUMNS: Final[frozenset[str]] = frozenset({CELL_KIND})

#: Declared primary key per table, written as the key v2 is meant to have.
PRIMARY_KEYS: Final[Mapping[str, tuple[str, ...]]] = {
    "decisions": ("decision_id",),
    "labels": ("decision_id",),
    "episodes": (CELL_KIND, "episode_id"),
    "round_summary": (CELL_KIND, "episode_id", "round"),
}

#: How many offending key values a violation carries.  Examples exist to make a violation
#: actionable, not to enumerate it; the counts are the assertion.
EXAMPLE_LIMIT: Final[int] = 5

STATUS_OK: Final[str] = "ok"
STATUS_DUPLICATE: Final[str] = "duplicate_keys"
STATUS_UNVERIFIABLE: Final[str] = "unverifiable"


@dataclass(frozen=True)
class DuplicateKey:
    """One offending key value, stringified, and how many rows carry it."""

    values: tuple[str, ...]
    n_rows: int


@dataclass(frozen=True)
class KeyViolation:
    """A table that is not unique on its primary key, or whose key could not be checked.

    ``key`` is the key that was *applied*, which is the declared key minus any optional column
    the table does not carry; ``note`` says so when they differ.
    """

    table: str
    key: tuple[str, ...]
    n_rows: int
    n_distinct: int
    n_duplicate_groups: int
    examples: tuple[DuplicateKey, ...]
    note: str = ""


@dataclass(frozen=True)
class KeyCheck:
    """One table's result, whether or not it violated.  Clean checks carry their note too,
    so a report can show that a key was asserted under a fallback and still held."""

    table: str
    key: tuple[str, ...]
    declared_key: tuple[str, ...]
    n_rows: int
    n_distinct: int
    n_duplicate_groups: int
    n_excess_rows: int
    examples: tuple[DuplicateKey, ...]
    status: str
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def violation(self) -> KeyViolation:
        return KeyViolation(
            table=self.table, key=self.key, n_rows=self.n_rows, n_distinct=self.n_distinct,
            n_duplicate_groups=self.n_duplicate_groups, examples=self.examples, note=self.note)


def _hashable(value: Any) -> Hashable:
    """Key cells are scalars in every table here; anything else is compared by ``repr`` so a
    surprising column type degrades the message rather than raising mid-check."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _columns(path: Path) -> tuple[str, ...]:
    """Column names from the parquet footer, without reading a row of data."""
    import pyarrow.parquet as pq

    return tuple(pq.ParquetFile(path).schema_arrow.names)


def _row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _read_key_columns(path: Path, columns: Sequence[str]) -> list[tuple[Hashable, ...]]:
    """Only the key columns are read: ``decisions.parquet`` is 31 MB and the checker has no
    business paging in ``answer_raw`` to count ids."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(columns))
    values = [table.column(name).to_pylist() for name in columns]
    return [tuple(_hashable(v) for v in row) for row in zip(*values)]


def applied_key(declared: Sequence[str], available: Sequence[str]) -> tuple[tuple[str, ...], str]:
    """``(key to apply, note)``.  An absent optional column is dropped and explained; an absent
    required column yields an empty key, which the caller reports as unverifiable."""
    present = set(available)
    missing = [name for name in declared if name not in present]
    if not missing:
        return tuple(declared), ""
    required_missing = [name for name in missing if name not in OPTIONAL_KEY_COLUMNS]
    if required_missing:
        return (), f"missing required key column(s) {sorted(required_missing)}"
    kept = tuple(name for name in declared if name in present)
    return kept, (f"{sorted(missing)} absent from this table; key asserted on "
                  f"{list(kept)} instead")


def check_table(tables_dir: str | os.PathLike, table: str,
                key: Sequence[str]) -> KeyCheck:
    """Assert ``key`` is unique on ``<tables_dir>/<table>.parquet``."""
    root = Path(tables_dir)
    declared = tuple(key)
    path = root / f"{table}.parquet"
    if not path.exists():
        # ``compact.py`` writes a ``.empty`` sentinel for a table with no rows; that is a
        # vacuously unique table, not a missing one.
        if (root / f"{table}.empty").exists():
            return KeyCheck(table=table, key=declared, declared_key=declared, n_rows=0,
                            n_distinct=0, n_duplicate_groups=0, n_excess_rows=0, examples=(),
                            status=STATUS_OK, note="empty sentinel: the table has no rows")
        return KeyCheck(table=table, key=declared, declared_key=declared, n_rows=0, n_distinct=0,
                        n_duplicate_groups=0, n_excess_rows=0, examples=(),
                        status=STATUS_UNVERIFIABLE, note=f"{path} does not exist")

    used, note = applied_key(declared, _columns(path))
    if not used:
        return KeyCheck(table=table, key=(), declared_key=declared, n_rows=_row_count(path),
                        n_distinct=0, n_duplicate_groups=0, n_excess_rows=0, examples=(),
                        status=STATUS_UNVERIFIABLE, note=note)

    counts = Counter(_read_key_columns(path, used))
    n_rows = sum(counts.values())
    duplicates = {values: n for values, n in counts.items() if n > 1}
    # Most-duplicated first, then lexicographic, so two runs over the same table agree.
    worst = sorted(duplicates.items(), key=lambda kv: (-kv[1], tuple(str(v) for v in kv[0])))
    examples = tuple(DuplicateKey(values=tuple(str(v) for v in values), n_rows=n)
                     for values, n in worst[:EXAMPLE_LIMIT])
    return KeyCheck(
        table=table, key=used, declared_key=declared, n_rows=n_rows, n_distinct=len(counts),
        n_duplicate_groups=len(duplicates), n_excess_rows=n_rows - len(counts), examples=examples,
        status=STATUS_OK if not duplicates else STATUS_DUPLICATE, note=note)


def run_checks(tables_dir: str | os.PathLike, *,
               keys: Mapping[str, Sequence[str]] = PRIMARY_KEYS) -> list[KeyCheck]:
    """Every declared key, checked.  Clean tables are included, so a report can distinguish
    "asserted and held" from "never asserted" — the distinction the audit found missing."""
    return [check_table(tables_dir, table, key) for table, key in keys.items()]


def check_keys(tables_dir: str | os.PathLike, *, strict: bool = True) -> list[KeyViolation]:
    """Primary-key violations in ``tables_dir``, worst first.

    ``strict`` decides what an *unverifiable* check counts as: a table that is absent, or that
    lacks a required key column, is a violation under ``strict=True`` and is passed over under
    ``strict=False``.  A key that was checked and found non-unique is a violation either way —
    ``strict`` never suppresses a measured duplicate.  Note that a missing ``cell_kind`` is not
    an unverifiable check but a recorded fallback, so on the v1 tables both settings return the
    same three violations.
    """
    checks = run_checks(tables_dir)
    violations = [c.violation() for c in checks
                  if c.status == STATUS_DUPLICATE
                  or (strict and c.status == STATUS_UNVERIFIABLE)]
    violations.sort(key=lambda v: (-v.n_duplicate_groups, v.table))
    return violations


def report(tables_dir: str | os.PathLike, *, strict: bool = True) -> dict[str, Any]:
    """The JSON the CLI prints: every check, then the subset that violated."""
    checks = run_checks(tables_dir)
    violations = check_keys(tables_dir, strict=strict)
    return {
        "tables_dir": str(Path(tables_dir)),
        "strict": strict,
        "n_checks": len(checks),
        "n_violations": len(violations),
        "ok": not violations,
        "checks": [dataclasses.asdict(c) for c in checks],
        "violations": [dataclasses.asdict(v) for v in violations],
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agents_scaling.ard.keys",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", required=True, type=Path,
                    help="a compacted tables directory (read only; never written)")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="do not count an absent table or key column as a violation, and exit "
                         "0 even when duplicates were found")
    args = ap.parse_args(argv)

    payload = report(args.tables, strict=args.strict)
    print(json.dumps(payload, indent=2))
    return 1 if (args.strict and payload["violations"]) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
