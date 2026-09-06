"""Rolling evaluation waves over one sealed generate manifest (ops helper, WP5 follow-up).

A wave is the set of items whose *every* generate cell in the manifest is complete and whose
pools are all in the seal's ``POOLS.json``.  Such an item is judged (JUDGE_BEST), sealed
(selections) and evaluated (JUDGE_HLE / EVAL_BCB) exactly once, so aggregate's seal-order
guard holds for all of its selection records and no eval-kind cell is ever re-run against a
partial pool set.  Items already listed in an earlier wave file of the same seal under
``<run_root>/waves/`` are excluded.  Nothing here reads protected data.

Usage (per wave, in order):
  seal --kind pools --cells-file <manifest>
  waves --run-id R --cells-file <manifest> --seal <sha> --wave 1
  cells --tier 1-select --lane 32B --seal <sha> --items-file waves/<seal8>_w1.json --wave 1 --out …
  (dispatch JUDGE_BEST on the judge fleet) → seal --kind selections --select-cells-file …
  cells --tier 1-eval --lane 32B|eval … --items-file … --wave 1 → dispatch → aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.experiment import io
from agents_scaling.study.cells import cells_file_sha256, load_cells_file
from agents_scaling.study.data.public import load_public_tasks
from agents_scaling.study.selection import seal as seals
from agents_scaling.study.types import CellKind, ProtocolError

WAVES_DIR = "waves"


def wave_file(run_root: str | os.PathLike, seal: str, wave: str) -> Path:
    return Path(run_root) / WAVES_DIR / f"{seal[:8]}_w{wave}.json"


def prior_wave_items(run_root: str | os.PathLike, seal: str) -> dict[str, str]:
    """``{source_id: wave}`` over the existing wave files of ``seal``."""
    out: dict[str, str] = {}
    root = Path(run_root) / WAVES_DIR
    if not root.exists():
        return out
    for path in sorted(root.glob(f"{seal[:8]}_w*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("seal") != seal:
            raise ProtocolError(f"{path} belongs to seal {str(doc.get('seal'))[:12]}, not {seal[:12]}")
        for sid in doc["items"]:
            if sid in out:
                raise ProtocolError(f"{sid} listed in waves {out[sid]} and {doc['wave']}")
            out[sid] = str(doc["wave"])
    return out


def complete_items(run_root: str | os.PathLike, cells_file: str | os.PathLike, seal: str) -> tuple[list[str], dict[str, int]]:
    """Items whose every generate cell of ``cells_file`` has a sealed pool set, plus a per-item done-cell histogram."""
    run_root = Path(run_root)
    cells = [c for c in load_cells_file(cells_file) if c.kind is CellKind.GENERATE]
    if cells_file_sha256(cells_file) != seal:
        raise ProtocolError(f"{cells_file} hashes to {cells_file_sha256(cells_file)[:12]}, not seal {seal[:12]}")
    pools = seals.load_pools(run_root, seal)
    sealed_pairs = {(p["cell_id"], p["source_id"]) for p in pools["pools"].values()}
    required: dict[str, set[str]] = defaultdict(set)
    for cell in cells:
        for sid in cell.items:
            required[sid].add(cell.cell_id)
    done = {sid: sum(1 for cid in cids if (cid, sid) in sealed_pairs) for sid, cids in required.items()}
    complete = sorted(sid for sid, cids in required.items() if done[sid] == len(cids))
    hist: dict[str, int] = defaultdict(int)
    for sid, cids in required.items():
        hist[f"{done[sid]}/{len(cids)}"] += 1
    return complete, dict(sorted(hist.items()))


def plan_wave(run_root: str | os.PathLike, cells_file: str | os.PathLike, seal: str, wave: str, *, min_items: int = 1,
              clock=time.time) -> dict[str, Any]:
    run_root = Path(run_root)
    out = wave_file(run_root, seal, wave)
    if out.exists():
        raise ProtocolError(f"{out} exists; waves are immutable (pick a new wave tag)")
    complete, hist = complete_items(run_root, cells_file, seal)
    prior = prior_wave_items(run_root, seal)
    rank = {t.source_id: (t.domain.value if hasattr(t.domain, "value") else str(t.domain), t.rank) for t in load_public_tasks(run_root)}
    items = sorted((sid for sid in complete if sid not in prior), key=lambda s: (rank[s][1], rank[s][0]))
    doc = {
        "schema_version": 1, "seal": seal, "cells_file": str(cells_file), "wave": str(wave), "items": items,
        "n_items": len(items), "n_complete_total": len(complete), "n_prior": len(prior), "done_histogram": hist,
        "planned_at": clock(), "code_version": io.git_commit(),
    }
    if len(items) < min_items:
        return {**doc, "written": False}
    io.write_json(out, doc)
    return {**doc, "written": True, "path": str(out)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--cells-file", required=True, help="the sealed generate manifest (absolute or under the run root)")
    p.add_argument("--seal", required=True, help="its seal (manifest sha256)")
    p.add_argument("--wave", required=True, help="wave tag (letters/digits)")
    p.add_argument("--min-items", type=int, default=1, help="write nothing when fewer new complete items exist")
    p.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    p.add_argument("--dry-run", action="store_true", help="report the complete/new item counts only")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.results_root) / args.run_id
    cells_file = Path(args.cells_file)
    if not cells_file.is_absolute():
        cells_file = run_root / cells_file
    if args.dry_run:
        complete, hist = complete_items(run_root, cells_file, args.seal)
        prior = prior_wave_items(run_root, args.seal)
        print(json.dumps({"dry_run": True, "n_complete_total": len(complete), "n_prior": len(prior),
                          "n_new": len([s for s in complete if s not in prior]), "done_histogram": hist}, indent=2))
        return 0
    result = plan_wave(run_root, cells_file, args.seal, args.wave, min_items=args.min_items)
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))
    return 0 if result["written"] else 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["WAVES_DIR", "complete_items", "plan_wave", "prior_wave_items", "wave_file"]
