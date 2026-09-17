"""``allocation status --run-dir RUN [--local-state-dir DIR] [--no-write]``.

The live dispatcher writes ``state/status.json`` every ~10 s (spend snapshot, node counts per partition/state,
measured peak/mean in-flight vs configured, 429 count, circuit state). ``cmd_status`` recomputes what the durable
sources can prove - spend from the fsync'd ledger journal, counts from the local SQLite (read-only) when it is on
this node, else from its shared backup - merges the dispatcher's live metrics from the last ``status.json``, prints
the JSON and writes ``status.json`` unless a dispatcher wrote it within the last minute (never clobbers live data).

Config keys consumed: none.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from agent_allocation.contracts import LedgerEventKind
from agent_allocation.paths import RunPaths
from agent_allocation.storage._atomic import replace_atomic

LIVE_STATUS_FRESH_SECONDS = 60.0


def write_status_json(paths: RunPaths, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n").encode("utf-8")
    replace_atomic(paths.status_json, data)


def journal_spend(paths: RunPaths) -> dict[str, Any]:
    """Spend proven by the journal alone (open reservations count as exposure, never as zero)."""
    from agent_allocation.runtime.ledger import journal_attempts

    attempts = journal_attempts(paths.ledger_journal)  # read-only replay: never repairs or appends
    settled = sum((a.resolved_usd for a in attempts.values()
                   if a.resolution in (LedgerEventKind.SETTLE, LedgerEventKind.CHARGE_MAX)), Decimal("0"))
    charged = sum((a.resolved_usd for a in attempts.values() if a.resolution is LedgerEventKind.CHARGE_MAX),
                  Decimal("0"))
    released = sum((a.resolved_usd for a in attempts.values() if a.resolution is LedgerEventKind.RELEASE_UNSENT),
                   Decimal("0"))
    open_ = [a for a in attempts.values() if a.resolution is None]
    return {"settled_usd": str(settled), "open_reservations_usd": str(sum((a.reserved_usd for a in open_), Decimal("0"))),
            "open_attempts": len(open_), "attempts": len(attempts), "charged_max_usd": str(charged),
            "released_unsent_usd": str(released)}


def db_counts(db: Path) -> dict[str, dict[str, int]] | None:
    if not db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return None
    try:
        rows = conn.execute(
            "SELECT partition_id, CASE WHEN state='terminal' THEN 'terminal_' || COALESCE(outcome, 'unknown')"
            " ELSE state END AS k, COUNT(*) FROM nodes GROUP BY partition_id, k").fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    out: dict[str, dict[str, int]] = {}
    for partition, key, count in rows:
        out.setdefault(partition, {})[key] = count
    return out


def build_status(paths: RunPaths) -> tuple[dict[str, Any], bool]:
    live: dict[str, Any] = {}
    live_fresh = False
    if paths.status_json.exists():
        try:
            live = json.loads(paths.status_json.read_text())
            live_fresh = (time.time() - paths.status_json.stat().st_mtime) < LIVE_STATUS_FRESH_SECONDS \
                and live.get("state") == "running"
        except (OSError, ValueError):
            live = {}
    counts = db_counts(paths.state_db_local)
    counts_source = "local_sqlite"
    if counts is None:
        counts = db_counts(paths.state_db_backup)
        counts_source = "sqlite_backup" if counts is not None else "unavailable"
    status = dict(live)
    status.update({
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "journal_spend": journal_spend(paths),
        "counts": counts if counts is not None else live.get("counts"),
        "counts_source": counts_source,
        "control": {"stop": paths.stop_sentinel.exists(), "drain": paths.drain_sentinel.exists()},
        "dispatcher_live": live_fresh,
    })
    return status, live_fresh


def cmd_status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation status")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--local-state-dir", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    paths = RunPaths.for_run(args.run_dir, local_state_dir=args.local_state_dir)
    if not paths.run_dir.is_dir():
        print(f"run dir {paths.run_dir} does not exist", file=sys.stderr)
        return 2
    status, live = build_status(paths)
    if not args.no_write and not live:
        write_status_json(paths, status)
    print(json.dumps(status, sort_keys=True, indent=2, default=str))
    return 0
