"""Planned vs completed vs incomplete reconciliation (WP5; §10.6 "full request/cost
reconciliation", §10.4 intention-to-run: INFRA_INCOMPLETE items are listed, never hidden).

Per cells manifest and per (module, lane): planned cells/items, completed item files,
``incomplete/`` records, ``SUSPENDED.json`` cells, ``meta.json`` presence, aliased vs
generated requests (from ``meta.json``), plus the ambiguous-judge rate (``eval/hle``) and the
singleton-vote rate (sealed VOTE selections).  Output: ``<run_root>/reconcile.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.experiment import io
from agents_scaling.study.cells import load_cells_file
from agents_scaling.study.evaluation.hle_judge import EVAL_HLE_SUBDIR, JUDGE_AMBIGUOUS
from agents_scaling.study.runner import CellPaths, item_file_valid
from agents_scaling.study.selection import seal as seals
from agents_scaling.study.types import SELECTOR_VOTE, ProtocolError


def reconcile_cells(run_root: Path, cells_file: Path) -> dict[str, Any]:
    cells = load_cells_file(cells_file)
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {
        "planned_cells": 0, "cells_with_meta": 0, "suspended_cells": 0, "planned_items": 0, "completed_items": 0,
        "incomplete_items": 0, "aliased_requests": 0, "generated_requests": 0, "flops_total": 0, "incomplete": [], "suspended": [],
    })
    for cell in cells:
        g = groups[(cell.module, cell.lane)]
        paths = CellPaths.of(run_root, cell.cell_id)
        g["planned_cells"] += 1
        g["planned_items"] += len(cell.items)
        for sid in cell.items:
            try:
                if item_file_valid(paths.item(sid), cell, sid):
                    g["completed_items"] += 1
            except ProtocolError as exc:
                g["incomplete"].append({"cell_id": cell.cell_id, "source_id": sid, "error": str(exc)[:300]})
        for sid in paths.incomplete_items():
            g["incomplete_items"] += 1
            g["incomplete"].append({"cell_id": cell.cell_id, "source_id": sid})
        if paths.suspended.exists():
            g["suspended_cells"] += 1
            g["suspended"].append(cell.cell_id)
        if paths.meta.exists():
            meta = json.loads(paths.meta.read_text(encoding="utf-8"))
            g["cells_with_meta"] += 1
            g["aliased_requests"] += int(meta.get("n_aliased_requests", 0))
            g["generated_requests"] += int(meta.get("n_generated_requests", 0))
            g["flops_total"] += int(meta.get("flops_total", 0))
    return {f"{m}|{lane}": g for (m, lane), g in sorted(groups.items())}


def judge_rates(run_root: Path) -> dict[str, Any]:
    folder = run_root.joinpath(*EVAL_HLE_SUBDIR)
    n = amb = 0
    if folder.exists():
        for path in folder.glob("*.json"):
            rec = json.loads(path.read_text(encoding="utf-8"))
            for entry in rec.get("judgements", {}).values():
                n += 1
                amb += entry.get("judge") == JUDGE_AMBIGUOUS
    return {"judged_answers": n, "ambiguous": amb, "ambiguous_rate": (amb / n) if n else None}


def vote_rates(run_root: Path) -> dict[str, Any]:
    n = singleton = none_valid = 0
    seals_root = run_root / seals.SEALS_DIR
    if seals_root.exists():
        for path in seals_root.glob(f"*/{seals.SELECTIONS_FILE}"):
            register = seals.load_selections(run_root, path.parent.name)
            for sel in register["selections"].values():
                if sel["selector_id"] != SELECTOR_VOTE:
                    continue
                n += 1
                singleton += bool(sel["record"]["all_singleton"])
                none_valid += bool(sel["record"]["no_valid_candidate"])
    return {"vote_selections": n, "all_singleton": singleton, "singleton_rate": (singleton / n) if n else None,
            "no_valid_candidate": none_valid, "no_valid_rate": (none_valid / n) if n else None}


def reconcile(run_root: str | os.PathLike, cells_files: Sequence[str | os.PathLike] | None = None) -> dict[str, Any]:
    run_root = Path(run_root)
    files = [Path(f) for f in cells_files] if cells_files else sorted(run_root.glob("cells_*.json"))
    report = {
        "run_root": str(run_root),
        "manifests": {str(f.name): reconcile_cells(run_root, f) for f in files},
        "judge": judge_rates(run_root),
        "vote": vote_rates(run_root),
    }
    io.write_json(run_root / "reconcile.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.study.reconcile", description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    p.add_argument("--cells-file", action="append", default=None, help="manifest(s) to reconcile (default: cells_*.json under the run root)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.results_root) / args.run_id
    files = [Path(f) if Path(f).is_absolute() else run_root / f for f in (args.cells_file or [])] or None
    report = reconcile(run_root, files)
    for name, groups in report["manifests"].items():
        for key, g in groups.items():
            print(f"{name} {key}: cells {g['cells_with_meta']}/{g['planned_cells']} meta, items {g['completed_items']}/{g['planned_items']} done, "
                  f"{g['incomplete_items']} incomplete, {g['suspended_cells']} suspended, aliased {g['aliased_requests']} / generated {g['generated_requests']}")
    print(json.dumps({"judge": report["judge"], "vote": report["vote"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["judge_rates", "reconcile", "reconcile_cells", "vote_rates"]
