"""Per-cell JSON + the request store -> the spec-17.3 parquet set, published atomically.

Plan §3.4 ("``materialize`` writes ``tables.<ts>.tmp/``, runs ``assert_keys``, writes
``MANIFEST.json`` last, renames to ``tables_vN/``"), spec 17.3, contract §2.5.

Three rules, each of them an ARD36 repair:

* **A crash leaves no tables directory.**  Everything is built under ``tables.<ts>.tmp/`` and
  the directory is *renamed* into place only after every key assertion has passed and
  ``MANIFEST.json`` has been written.  ``test_partial_materialize_leaves_no_tables_dir`` kills
  the pass mid-write; a reader can then tell "not materialized yet" from "materialized", which
  ARD36 could not (it wrote parquet files into the live directory as it went).
* **Key assertions are fatal here, not advisory.**  ``tables.check_keys(strict=True)`` and
  ``tables.assert_referential(strict=True)`` run before the rename, so a duplicate logical
  decision key or a bank/episode collision aborts the publish (spec 15.3:
  ``duplicate_logical_keys: fatal``).  ARD36 shipped 14,336 duplicate decision keys because
  the check ran after publication and printed a warning.
* **Private rows never enter the public set.**  Protected rows travel in
  ``ItemFile.private_rows`` and are published to ``protected/tables_vN/`` (0700), which the
  status monitor's path allowlist refuses to open at all (R7).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import cli, config as C, deps, plan as PLAN, tables
from agents_scaling.nextdyn.errors import MaterializeError
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.study import identity

TABLES_DIR: Final[str] = "tables"
VERSIONED_PREFIX: Final[str] = "tables_v"
TMP_SUFFIX: Final[str] = ".tmp"
MANIFEST_FILE: Final[str] = "MANIFEST.json"
PROTECTED_DIR: Final[str] = tables.PROTECTED_DIR
SCHEMA_VERSION: Final[int] = 1
#: The only ``failure_policy.duplicate_logical_keys`` this publisher implements (spec 15.3).
DUPLICATE_POLICY_FATAL: Final[str] = "fatal"


# --------------------------------------------------------------------------- collection


@dataclass
class Collected:
    """Rows gathered from every cell, by table, before any projection."""

    public: dict[str, list[Any]]
    private: dict[str, list[Any]]
    cells: int = 0
    items: int = 0
    skipped: int = 0

    @classmethod
    def empty(cls) -> "Collected":
        return cls(public=defaultdict(list), private=defaultdict(list))


def iter_cell_dirs(run_root: str | os.PathLike) -> list[Path]:
    root = Path(run_root) / deps.CELLS_DIR
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def collect(run_root: str | os.PathLike, *, cell_ids: Sequence[str] | None = None) -> Collected:
    """Read every complete cell's item files (public and protected) into row lists.

    A cell without ``meta.json`` is *skipped*, not partially read: an in-flight cell's items
    are real but its census is not final, and publishing half a cell is how a table ends up
    with a cohort nobody can name.
    """
    root = Path(run_root)
    out = Collected.empty()
    wanted = None if cell_ids is None else set(cell_ids)
    for directory in iter_cell_dirs(root):
        cell_id = directory.name
        if wanted is not None and cell_id not in wanted:
            continue
        if not (directory / deps.META_FILE).exists():
            out.skipped += 1
            continue
        out.cells += 1
        for item_path in sorted((directory / tables.ITEMS_DIR).glob("*.json")):
            protected = tables.protected_item_path(root, cell_id, item_path.stem)
            item = tables.ItemFile.read(item_path, protected_path=protected)
            out.items += 1
            for table, rows in item.rows.items():
                out.public[table].extend(rows)
            for table, rows in item.private_rows.items():
                out.private[table].extend(rows)
    return out


def collapse_by_key(rows: Sequence[Any], key: str) -> tuple[list[Any], int]:
    """Collapse rows that describe **one physical object** recorded twice.

    ``requests`` keys on ``request_hash`` and a physical request is one row, but two cells can
    race to generate the same content-addressed request (the store reports the loser as an
    alias, and the winner is whichever committed first).  Collapsing is safe *here* and
    nowhere else: the duplicate is the same object, not two observations of it -- two
    ``decisions`` under one key would be the D-11 defect and stays fatal.  The count is
    recorded in ``MANIFEST.json`` rather than swallowed.
    """
    seen: dict[Any, Any] = {}
    collapsed = 0
    for row in rows:
        value = getattr(row, key)
        if value in seen:
            collapsed += 1
            continue
        seen[value] = row
    return list(seen.values()), collapsed


def fill_first_decision_id(rows: Sequence[Any]) -> list[Any]:
    """``decisions.first_decision_id`` per ``generation_uid`` (plan §1.3).

    Reuse is recorded, never silent: when several decisions share one physical generation the
    *first* one (in blind decision-id order, so it does not depend on dispatch order) is named
    on all of them, which is what lets the analysis count distinct samples rather than rows.
    """
    first: dict[str, str] = {}
    for row in sorted(rows, key=lambda r: r.decision_id):
        uid = getattr(row, "generation_uid", None)
        if uid:
            first.setdefault(uid, row.decision_id)
    out = []
    for row in rows:
        uid = getattr(row, "generation_uid", None)
        out.append(replace(row, first_decision_id=first.get(uid)) if uid else row)
    return out


# --------------------------------------------------------------------------- publication


def _versions(run_root: Path) -> list[int]:
    return sorted(int(p.name[len(VERSIONED_PREFIX):]) for p in run_root.glob(f"{VERSIONED_PREFIX}*")
                  if p.is_dir() and p.name[len(VERSIONED_PREFIX):].isdigit())


def next_version(run_root: str | os.PathLike) -> int:
    existing = _versions(Path(run_root))
    return (existing[-1] + 1) if existing else 1


def _publish_pointer(run_root: Path, versioned: Path, name: str) -> None:
    """Point ``<run_root>/<name>`` at the newly published version.

    A symlink when the filesystem allows one, a copy otherwise: the analysis reads
    ``<run_root>/tables`` (``outcomes.TABLES_DIR``) and must never see a half-written set, so
    the pointer is only moved after the versioned directory is complete.
    """
    pointer = run_root / name
    if pointer.is_symlink() or pointer.is_file():
        pointer.unlink()
    elif pointer.is_dir():
        if (pointer / MANIFEST_FILE).exists() or not any(pointer.iterdir()):
            shutil.rmtree(pointer)
        else:
            raise MaterializeError(
                f"{pointer} exists, is not empty and was not published by materialize; refusing "
                "to replace it (move it aside if it is legacy output)")
    try:
        pointer.symlink_to(versioned.name, target_is_directory=True)
    except OSError:                                     # pragma: no cover - no symlink support
        shutil.copytree(versioned, pointer)


def write_tables(target: Path, rows_by_table: Mapping[str, Sequence[Any]], *,
                 private: bool) -> dict[str, int]:
    """Write one parquet (or ``.empty`` sentinel) per declared table of this visibility."""
    target.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, spec in sorted(tables.TABLES.items()):
        if spec.private != private:
            continue
        rows = list(rows_by_table.get(name, ()))
        tables.write_table(rows, target / f"{name}.parquet", table=name)
        counts[name] = len(rows)
    return counts


def assert_published_keys(public: Path, protected: Path, *, strict: bool = True) -> list[str]:
    """Key checks per **visibility**: public tables in ``tables/``, private in ``protected/``.

    ``tables.check_keys`` demands every declared table on disk, and the two ``__private__``
    tables are deliberately not in the public set (R7) -- so the check runs twice, each time
    over the tables that directory is supposed to hold.  A *duplicate* is fatal under either
    setting; ``strict`` decides whether an absent table is too.
    """
    out: list[str] = []
    for directory, private in ((public, False), (protected, True)):
        wanted = {name: spec for name, spec in tables.TABLES.items() if spec.private is private}
        for check in tables.run_checks(directory, tables=wanted):
            if check.status == tables.STATUS_DUPLICATE or (strict and check.status == tables.STATUS_UNVERIFIABLE):
                detail = (f"{check.table} {check.constraint}{list(check.key)}: "
                          f"{check.n_duplicate_groups} duplicate group(s) {check.note}".strip())
                if strict:
                    raise MaterializeError("primary-key assertion failed: " + detail)
                out.append(detail)
    return out


def manifest_payload(run_root: Path, counts: Mapping[str, int], *, version: int,
                     collected: Collected, started: float) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "run_root": str(run_root),
        "cells": collected.cells,
        "items": collected.items,
        "cells_skipped": collected.skipped,
        "rows": dict(sorted(counts.items())),
        "materialized_at": time.time(),
        "wall_s": time.time() - started,
    }


@dataclass(frozen=True)
class MaterializeReport:
    version: int
    tables_dir: Path
    protected_dir: Path
    rows: Mapping[str, int]
    cells: int
    items: int
    violations: tuple[str, ...]


def assert_duplicate_policy(cfg: Any) -> bool:
    """``study.failure_policy.duplicate_logical_keys`` is what makes a duplicate key fatal.

    Read rather than assumed: the whole atomic-publish design exists because that policy says
    ``fatal``, and a run whose policy said otherwise would need a different publisher, not a
    quieter one.
    """
    policy = cfg.failure_policy.duplicate_logical_keys
    if policy != DUPLICATE_POLICY_FATAL:
        raise MaterializeError(
            f"study.failure_policy.duplicate_logical_keys is {policy!r}; this publisher only "
            f"implements {DUPLICATE_POLICY_FATAL!r} (spec 15.3)")
    return True


def archive_note(cfg: Any) -> dict[str, Any]:
    """What ``MANIFEST.json`` records about the full archive (``study.protocol``).

    ``full_archive_stored_externally`` says the complete per-request archive lives outside the
    published tables; the published set is the spec-17.3 projection of it, and a reader has to
    be able to tell which one they are holding.
    """
    return {"full_archive_stored_externally": bool(cfg.protocol.full_archive_stored_externally)}


def materialize(run_root: str | os.PathLike, *, out_dir: str = TABLES_DIR, strict: bool = True,
                cell_ids: Sequence[str] | None = None, cfg: Any = None) -> MaterializeReport:
    """Publish the spec-17.3 parquet set atomically (plan §3.4)."""
    root = Path(run_root)
    started = time.time()
    collected = collect(root, cell_ids=cell_ids)
    public = dict(collected.public)
    if public.get("decisions"):
        public["decisions"] = fill_first_decision_id(public["decisions"])
    public["requests"], collapsed = collapse_by_key(public.get("requests", ()), "request_hash")
    plan_rows = _plan_rows(root)
    if plan_rows is not None:
        public["plan"] = plan_rows
    for name in ("source_items", "item_variants", "item_splits"):
        rows = _public_store_rows(root, name)
        if rows is not None:
            public.setdefault(name, rows)

    stamp = f"{int(started)}"
    tmp = root / f"{out_dir}.{stamp}{TMP_SUFFIX}"
    tmp_protected = root / PROTECTED_DIR / f"{out_dir}.{stamp}{TMP_SUFFIX}"
    for path in (tmp, tmp_protected):
        if path.exists():
            shutil.rmtree(path)
    try:
        counts = write_tables(tmp, public, private=False)
        counts.update(write_tables(tmp_protected, dict(collected.private), private=True))
        os.chmod(tmp_protected, tables.PROTECTED_DIR_MODE)
        violations = assert_published_keys(tmp, tmp_protected, strict=strict)
        violations += tables.assert_referential(tmp, strict=strict)
        version = next_version(root)
        payload = manifest_payload(root, counts, version=version, collected=collected,
                                   started=started)
        payload["violations"] = violations
        payload["requests_collapsed"] = collapsed
        if cfg is not None:
            assert_duplicate_policy(cfg)
            payload.update(archive_note(cfg))
        payload["manifest_sha256"] = identity.sha256_hex(identity.jcs(
            {k: v for k, v in payload.items() if k not in ("materialized_at", "wall_s")}))
        rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        deps.atomic_write_bytes(tmp / MANIFEST_FILE, rendered)      # MANIFEST.json last
        deps.atomic_write_bytes(tmp_protected / MANIFEST_FILE, rendered)
    except BaseException:
        # Whatever went wrong -- a key violation, a bad row, a kill -- leave no partial table
        # set behind, so a reader can always tell "not materialized" from "materialized"
        # (``test_partial_materialize_leaves_no_tables_dir``).  The publish is the rename
        # below; until it happens nothing outside these two tmp directories has changed.
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(tmp_protected, ignore_errors=True)
        raise

    versioned = root / f"{VERSIONED_PREFIX}{version}"
    versioned_protected = root / PROTECTED_DIR / f"{VERSIONED_PREFIX}{version}"
    os.replace(tmp, versioned)
    versioned_protected.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_protected, versioned_protected)
    _publish_pointer(root, versioned, out_dir)
    _publish_pointer(root / PROTECTED_DIR, versioned_protected, out_dir)
    return MaterializeReport(version=version, tables_dir=versioned,
                             protected_dir=versioned_protected, rows=counts,
                             cells=collected.cells, items=collected.items,
                             violations=tuple(violations))


def _plan_rows(run_root: Path) -> list[Any] | None:
    """The ``plan`` table the planner wrote (``plan/plan.parquet``), re-read as row objects."""
    path = run_root / PLAN.PLAN_DIR / f"{PLAN.PLAN_TABLE}.parquet"
    if not path.exists() and not path.with_suffix(tables.EMPTY_SUFFIX).exists():
        return None
    rows = tables.read_table(path.parent, PLAN.PLAN_TABLE)
    return [tables.row_from_dict(tables.PlanRow, row) for row in rows]


def _public_store_rows(run_root: Path, name: str) -> list[Any] | None:
    """The inventory's public tables, carried into the published set unchanged."""
    try:
        from agents_scaling.nextdyn import items as I
    except ImportError:                                 # pragma: no cover - WP3 pending
        return None
    directory = I.public_dir(run_root)
    if not (directory / f"{name}.parquet").exists() and \
            not (directory / f"{name}{tables.EMPTY_SUFFIX}").exists():
        return None
    row_type = tables.schema(name).row_type
    return [tables.row_from_dict(row_type, row) for row in tables.read_table(directory, name)]


# --------------------------------------------------------------------------- worker + verb


def materialize_worker(runner: Any, source_item_id: str) -> tables.ItemFile:
    """The MATERIALIZE cell worker: one cell, one publish, no rows of its own.

    The cell itself is not in the published set -- it is still running when ``collect`` reads
    the run root -- which is exactly why its own item file carries no rows.
    """
    materialize(runner.run_root, cfg=runner.cfg)
    return tables.ItemFile(
        schema_version=tables.SCHEMA_VERSION, cell_id=runner.cell.cell_id,
        cell_kind=CellKind.MATERIALIZE, source_item_id=source_item_id,
        status=tables.ItemStatus.COMPLETE, bundle_export_hash=None, rows={})


def main(argv: Sequence[str] | None = None) -> cli.Report:
    parser = cli.common_parser("materialize", description="publish the spec-17.3 parquet set")
    parser.add_argument("--out-dir", default=TABLES_DIR)
    parser.add_argument("--no-strict", action="store_true",
                        help="report key violations instead of aborting (diagnosis only)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_root = cli.resolve_run_root(args)
    cfg = C.load_study(args.config)
    try:
        report = materialize(run_root, out_dir=args.out_dir, strict=not args.no_strict, cfg=cfg)
    except MaterializeError as exc:
        return cli.Report(failed=1, detail=str(exc))
    return cli.Report(completed=report.cells, skipped=0,
                      failed=len(report.violations),
                      detail=f"tables_v{report.version}: {report.items} items, "
                             f"{sum(report.rows.values())} rows -> {report.tables_dir}")


__all__ = [
    "Collected", "MANIFEST_FILE", "MaterializeReport", "SCHEMA_VERSION", "TABLES_DIR",
    "VERSIONED_PREFIX", "collect", "fill_first_decision_id", "iter_cell_dirs", "main",
    "DUPLICATE_POLICY_FATAL", "archive_note", "assert_duplicate_policy",
    "assert_published_keys", "collapse_by_key", "materialize", "materialize_worker",
    "next_version",
    "write_tables",
]
