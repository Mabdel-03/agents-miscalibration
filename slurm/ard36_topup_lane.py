"""Re-dispatch the ARD36 cells of one manifest that have neither ``meta.json`` nor a live task.

Fork of ``slurm/study_topup_lane.py`` (WP4).  ``ard36_launch_chunked.py`` gates on whole
chunks: while a chunk's array still has ANY task queued or running, its already-exited
stragglers (preempted tasks, ``run_one`` exit 3 = incomplete items, exit 5 = a dependency
that has since landed) are not re-dispatched until the whole array ends.  Late in a 36-hour
run that leaves the fleet under-subscribed for hours, so this tool runs every 15 minutes.

It computes, for one manifest, the cells that lack ``meta.json`` and are not covered by a
live array task **of that same manifest**, renders ONE sparse array over exactly those
absolute cell indices, and submits it.  Cells covered by a live task are never re-submitted,
so no cell gets two workers; ``run_one`` also holds a per-cell flock and skips cells with
``meta.json``.  Cells sitting on ``DEPS_UNMET.json`` are included (that is the point: the
controller freeze or the state seal may have landed since) and are reported separately, and
``SUSPENDED.json`` cells are excluded — they are terminal until a human removes the file.

    python slurm/ard36_topup_lane.py --run-id ard36 --cells-file cells_E4_GATE.json --lane 8B
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
DEPS_UNMET_FILE = "DEPS_UNMET.json"


def plan_topup(todo: Iterable[int], live: Iterable[int], *, max_tasks: int | None = None) -> list[int]:
    """Cell indices to (re)dispatch: unfinished minus live, ascending, capped at ``max_tasks``."""
    out = sorted(set(todo) - set(live))
    return out[:max_tasks] if max_tasks is not None else out


def unfinished_indices(run_root: Path, cells: Sequence[dict]) -> list[int]:
    """Cells without ``meta.json``, excluding the terminal ``SUSPENDED.json`` ones."""
    out = []
    for index, cell in enumerate(cells):
        cell_dir = run_root / "cells" / cell["cell_id"]
        if (cell_dir / "meta.json").exists() or (cell_dir / "SUSPENDED.json").exists():
            continue
        out.append(index)
    return out


def waiting_indices(run_root: Path, cells: Sequence[dict]) -> list[int]:
    return [index for index, cell in enumerate(cells)
            if (run_root / "cells" / cell["cell_id"] / DEPS_UNMET_FILE).exists()
            and not (run_root / "cells" / cell["cell_id"] / "meta.json").exists()]


def _squeue_array_indices(job_id: str) -> set[int]:
    out = subprocess.run(["squeue", "-h", "-r", "-j", str(job_id), "-t", LIVE_STATES, "-o", "%K"],
                         capture_output=True, text=True).stdout
    return {int(token) for token in out.split() if token.strip().isdigit()}


def live_indices(run_root: Path, lane: str, cells_file: str) -> set[int]:
    """Absolute cell indices currently owned by a live array task of this manifest."""
    log_path = run_root / f"chunk_jobs_{lane}.json"
    if not log_path.exists():
        return set()
    entries = json.loads(log_path.read_text())
    live: set[int] = set()
    wanted = Path(cells_file).name
    for entry in entries:
        # Compare basenames: the launcher records whatever path it was given, and a topup
        # invoked with an absolute path must still recognise a live task submitted with a
        # relative one.  Getting this wrong reports "0 live" and dispatches a second worker
        # for a cell that already has one.
        if Path(str(entry.get("cells_file") or "")).name != wanted:
            continue
        job_id = entry.get("job_id")
        if job_id:
            live |= _squeue_array_indices(job_id)
    return live


def _user_task_count() -> int:
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-o", "%i"],
                         capture_output=True, text=True).stdout
    return len([line for line in out.splitlines() if line.strip()])


def render_topup(template: Path, indices: Sequence[int], throttle: int, out_path: Path) -> Path:
    text = template.read_text()
    array = ",".join(str(index) for index in indices)
    new, n = re.subn(r"^#SBATCH --array=.*$", f"#SBATCH --array={array}%{throttle}", text,
                     count=1, flags=re.M)
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
    ap.add_argument("--throttle", type=int, default=24)
    ap.add_argument("--max-tasks", type=int, default=None, help="cap the sparse array size")
    ap.add_argument("--qos-limit", type=int, default=420,
                    help="refuse when my total queued tasks would exceed this (230 on ou_bcs_*/pi_*)")
    ap.add_argument("--template", default=None, help="chunk sbatch to reuse")
    ap.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    run_root = Path(args.results_root) / args.run_id
    cells = json.loads((run_root / args.cells_file).read_text())["cells"]
    todo = unfinished_indices(run_root, cells)
    waiting = waiting_indices(run_root, cells)
    live = live_indices(run_root, args.lane, args.cells_file)
    plan = plan_topup(todo, live, max_tasks=args.max_tasks)
    # The chunk renders are named from the manifest's *basename*; deriving the stem from the
    # raw argument put an absolute path inside the glob, which matched nothing and printed
    # "no chunk render" while exiting 0 -- a top-up loop that silently dispatches nothing.
    stem = Path(args.cells_file).name
    stem = stem[:-5] if stem.endswith(".json") else stem
    print(f"[topup:{args.lane}] {args.cells_file}: {len(cells)} cells, {len(todo)} unfinished "
          f"({len(waiting)} waiting on a dependency), {len(live)} live, {len(plan)} to dispatch")
    if not plan:
        print(f"[topup:{args.lane}] nothing to do")
        return 0
    in_queue = _user_task_count()
    if in_queue + len(plan) > args.qos_limit:
        print(f"[topup:{args.lane}] refusing: {in_queue} tasks queued + {len(plan)} > "
              f"--qos-limit {args.qos_limit}")
        return 4
    template = Path(args.template) if args.template else None
    if template is None:
        candidates = sorted(run_root.glob(f"chunk_{args.lane}_{stem}_*.sbatch"))
        candidates = [p for p in candidates if ".resume." not in p.name and ".topup." not in p.name]
        if not candidates:
            print(f"[topup:{args.lane}] no chunk render matching "
                  f"chunk_{args.lane}_{stem}_*.sbatch in {run_root}; dispatch the manifest "
                  f"once with ard36_launch_chunked.py first, or pass --template")
            return 5
        template = candidates[0]
    out_path = run_root / f"chunk_{args.lane}_{stem}.topup.sbatch"
    render_topup(template, plan, args.throttle, out_path)
    print(f"[topup:{args.lane}] template {template.name} -> {out_path.name}; indices "
          f"{plan[:12]}{'…' if len(plan) > 12 else ''}")
    if args.dry_run:
        print(f"[topup:{args.lane}] dry-run; not submitted")
        return 0
    result = subprocess.run(["sbatch", str(out_path)], capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        return 6
    job_id = match.group(1)
    log_path = run_root / f"chunk_jobs_{args.lane}.json"
    entries = json.loads(log_path.read_text()) if log_path.exists() else []
    entries.append({"chunk": -1, "lo": min(plan), "hi": max(plan), "job_id": job_id,
                    "lane": args.lane, "cells_file": args.cells_file, "topup_indices": plan})
    log_path.write_text(json.dumps(entries, indent=1) + "\n")
    print(f"[topup:{args.lane}] job {job_id} for {len(plan)} cells; chunk log updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
