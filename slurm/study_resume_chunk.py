"""Resubmit only the unfinished cells of an already-rendered study chunk as a sparse array.

    python slurm/study_resume_chunk.py --run-id study_v4 --cells-file cells_2_4B.json --lane 4B --chunk 0 [--throttle 12] [--dry-run]

The chunk driver's whole-chunk gate needs QOS room for the full chunk (100 tasks) even when a
handful of cells lack meta.json, and pending array elements count against the per-user
submit limit.  This renders ``<run_root>/chunk_<lane>_<manifest>_<chunk>_<lo>-<hi>.sbatch``
into a ``.resume.sbatch`` whose ``--array`` lists exactly the unfinished indices, submits it
and appends a ``chunk_jobs_<lane>.json`` entry so the driver's in-flight check sees it.
Refuses when the original chunk job is still in the queue.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_RESULTS_ROOT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cells-file", required=True)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--throttle", type=int, default=None, help="array throttle (default: keep the render's)")
    ap.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    run_root = Path(args.results_root) / args.run_id
    cells = json.loads((run_root / args.cells_file).read_text())["cells"]
    lo = args.chunk * args.chunk_size
    hi = min(lo + args.chunk_size, len(cells)) - 1
    stem = Path(args.cells_file).stem
    src = run_root / f"chunk_{args.lane}_{stem}_{args.chunk:03d}_{lo}-{hi}.sbatch"
    if not src.exists():
        raise SystemExit(f"no rendered chunk at {src}; run the driver once (it renders before gating)")
    log_path = run_root / f"chunk_jobs_{args.lane}.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    for e in log:
        if e.get("lo") == lo and e.get("hi") == hi and e.get("cells_file") in (None, args.cells_file) and e.get("job_id"):
            out = subprocess.run(["squeue", "-h", "-j", str(e["job_id"]), "-o", "%i"], capture_output=True, text=True).stdout
            if out.strip():
                raise SystemExit(f"chunk {args.chunk} still in flight as job {e['job_id']}")
    todo = [i for i in range(lo, hi + 1) if not (run_root / "cells" / cells[i]["cell_id"] / "meta.json").exists()]
    if not todo:
        print(f"chunk {args.chunk} ({lo}-{hi}) has no unfinished cells")
        return 0
    text = src.read_text()
    m = re.search(r"^#SBATCH --array=.*?%(\d+)$", text, flags=re.M)
    throttle = args.throttle or (int(m.group(1)) if m else 40)
    new = re.sub(r"^#SBATCH --array=.*$", f"#SBATCH --array={','.join(map(str, todo))}%{throttle}", text, flags=re.M)
    dst = src.with_suffix(".resume.sbatch")
    dst.write_text(new)
    print(f"chunk {args.chunk} ({lo}-{hi}): {len(todo)} unfinished cells -> {dst.name} (throttle {throttle})")
    if args.dry_run:
        return 0
    proc = subprocess.run(["sbatch", str(dst)], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"sbatch failed: {proc.stderr.strip()[:300]}", file=sys.stderr)
        return 1
    job = proc.stdout.strip().split()[-1]
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    log.append({"chunk": args.chunk, "lo": lo, "hi": hi, "job_id": job, "lane": args.lane, "cells_file": args.cells_file, "resume_indices": len(todo)})
    tmp = log_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(log, indent=2))
    os.replace(tmp, log_path)
    print(f"submitted job {job}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
