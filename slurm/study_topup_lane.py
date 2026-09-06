"""Dispatch the cells of a manifest that have no ``meta.json`` AND no live array task.

    python slurm/study_topup_lane.py --run-id study_v4 --cells-file cells_1_32B.json --lane 32B [--dry-run]

``study_launch_chunked.py`` gates on whole chunks: while a chunk's array job still has ANY
task queued or running, the chunk is "in flight" and its already-exited stragglers (preempted
tasks, or ``run_one`` exit 3 = incomplete items) are not re-dispatched until the whole array
ends.  Late in a run that leaves the serving fleet under-subscribed for hours.

This tool computes, for one manifest, the cells that lack ``meta.json`` and are not covered by
a live array task of that same manifest (``chunk_jobs_<lane>.json`` → ``squeue -r``), renders
ONE sparse array over exactly those absolute cell indices (the chunk template's array index is
the absolute cell index) and submits it.  Cells covered by a live task are never re-submitted,
so no cell gets two workers; ``run_one`` also holds a per-cell lock and skips cells with
``meta.json``, so a race with a task that finishes meanwhile is a cheap no-op.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_RESULTS_ROOT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results"
#: Slurm states that still own a cell (terminal states are absent, so the cell is free).
LIVE_STATES = "PD,R,CG,CF,S,RS,RQ,RD,RF,RH"


def plan_topup(todo: Iterable[int], live: Iterable[int], *, max_tasks: int | None = None) -> list[int]:
    """Cell indices to (re)dispatch: unfinished minus live, ascending, capped at ``max_tasks``."""
    out = sorted(set(todo) - set(live))
    return out[:max_tasks] if max_tasks is not None else out


def unfinished_indices(run_root: Path, cells: Sequence[dict]) -> list[int]:
    return [i for i, c in enumerate(cells) if not (run_root / "cells" / c["cell_id"] / "meta.json").exists()]


def _squeue_array_indices(job_id: str) -> set[int]:
    out = subprocess.run(["squeue", "-h", "-r", "-j", str(job_id), "-t", LIVE_STATES, "-o", "%K"],
                         capture_output=True, text=True).stdout
    return {int(tok) for tok in out.split() if tok.strip().isdigit()}


def live_indices(run_root: Path, lane: str, cells_file: str) -> set[int]:
    """Absolute cell indices currently owned by a live array task of this manifest."""
    log_path = run_root / f"chunk_jobs_{lane}.json"
    if not log_path.exists():
        return set()
    entries = json.loads(log_path.read_text())
    live: set[int] = set()
    for entry in entries:
        if str(entry.get("cells_file") or "") != cells_file:
            continue
        jid = entry.get("job_id")
        if jid:
            live |= _squeue_array_indices(jid)
    return live


def _user_task_count() -> int:
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%i"], capture_output=True, text=True).stdout
    return len([ln for ln in out.splitlines() if ln.strip()])


def render_topup(template: Path, indices: Sequence[int], throttle: int, out_path: Path) -> Path:
    text = template.read_text()
    array = ",".join(str(i) for i in indices)
    new, n = re.subn(r"^#SBATCH --array=.*$", f"#SBATCH --array={array}%{throttle}", text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"{template}: no '#SBATCH --array=' line to replace")
    out_path.write_text(new)
    out_path.chmod(0o755)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cells-file", required=True)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--throttle", type=int, default=40)
    ap.add_argument("--max-tasks", type=int, default=None, help="cap the sparse array size")
    ap.add_argument("--qos-limit", type=int, default=460, help="refuse when my total queued tasks would exceed this")
    ap.add_argument("--template", default=None, help="chunk sbatch to reuse (default: the manifest's first chunk render)")
    ap.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    run_root = Path(args.results_root) / args.run_id
    cells = json.loads((run_root / args.cells_file).read_text())["cells"]
    todo = unfinished_indices(run_root, cells)
    live = live_indices(run_root, args.lane, args.cells_file)
    plan = plan_topup(todo, live, max_tasks=args.max_tasks)
    stem = args.cells_file[:-5] if args.cells_file.endswith(".json") else args.cells_file
    print(f"[topup:{args.lane}] {args.cells_file}: {len(cells)} cells, {len(todo)} unfinished, {len(live)} live, {len(plan)} to dispatch")
    if not plan:
        print(f"[topup:{args.lane}] nothing to do")
        return 0
    in_queue = _user_task_count()
    if in_queue + len(plan) > args.qos_limit:
        print(f"[topup:{args.lane}] refusing: {in_queue} tasks queued + {len(plan)} > --qos-limit {args.qos_limit}")
        return 4
    template = Path(args.template) if args.template else None
    if template is None:
        cands = sorted(run_root.glob(f"chunk_{args.lane}_{stem}_*.sbatch"))
        cands = [p for p in cands if ".resume." not in p.name and ".topup." not in p.name]
        if not cands:
            print(f"[topup:{args.lane}] no chunk render for {args.cells_file}")
            return 5
        template = cands[0]
    out_path = run_root / f"chunk_{args.lane}_{stem}.topup.sbatch"
    render_topup(template, plan, args.throttle, out_path)
    print(f"[topup:{args.lane}] template {template.name} → {out_path.name}; indices {plan[:12]}{'…' if len(plan) > 12 else ''}")
    if args.dry_run:
        print(f"[topup:{args.lane}] dry-run; not submitted")
        return 0
    res = subprocess.run(["sbatch", str(out_path)], capture_output=True, text=True)
    print(res.stdout.strip() or res.stderr.strip())
    m = re.search(r"Submitted batch job (\d+)", res.stdout)
    if not m:
        return 6
    jid = m.group(1)
    log_path = run_root / f"chunk_jobs_{args.lane}.json"
    entries = json.loads(log_path.read_text()) if log_path.exists() else []
    entries.append({"chunk": -1, "lo": min(plan), "hi": max(plan), "job_id": jid, "lane": args.lane,
                    "cells_file": args.cells_file, "topup_indices": plan})
    log_path.write_text(json.dumps(entries, indent=1) + "\n")
    print(f"[topup:{args.lane}] job {jid} for {len(plan)} cells; chunk log updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
