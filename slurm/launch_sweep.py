"""Orchestrate a full sweep on SLURM.

Steps:
  1. Expand the sweep YAML into canonical cells; write ``cells.json``.
  2. Launch one long-lived vLLM serving job per distinct model size (servers register
     their endpoint to the run's registry once healthy).
  3. Submit the cell array job (each task discovers its server via the registry and waits
     until it is ready, so we don't need an explicit SLURM dependency on the servers).

``--serve-only`` launches just the servers (useful for the smoke test / pilot). Pass
``--models`` to restrict which sizes to serve.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agents_scaling.experiment import io
from agents_scaling.experiment.sweep import load_sweep, models_in_sweep
from agents_scaling.serving.launch_server import submit as submit_server

REPO = Path(__file__).resolve().parent.parent
ARRAY_TMPL = REPO / "slurm" / "run_cell_array.sbatch.tmpl"


def _render_array(run_id: str, cells_file: str, n_cells: int, partition: str, time_limit: str, throttle: int, log_dir: str) -> str:
    text = ARRAY_TMPL.read_text()
    repl = {
        "RUN_ID": run_id,
        "CELLS_FILE": cells_file,
        "LAST_INDEX": str(n_cells - 1),
        "THROTTLE": str(throttle),
        "PARTITION": partition,
        "TIME": time_limit,
        "LOG_DIR": log_dir,
        "REPO": str(REPO),
    }
    for k, v in repl.items():
        text = text.replace("{" + k + "}", v)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Launch a sweep (servers + cell array) on SLURM.")
    ap.add_argument("--config", default=str(REPO / "configs" / "pilot.yaml"))
    ap.add_argument("--run-id", default=f"run_{int(time.time())}")
    ap.add_argument("--serve-only", action="store_true", help="launch only the vLLM servers")
    ap.add_argument("--models", nargs="*", help="restrict served model sizes (default: all in sweep)")
    ap.add_argument("--serve-partition", default="pi_tpoggio")
    ap.add_argument("--serve-gpu-type", default="a100")
    ap.add_argument("--serve-time", default="7-00:00:00")  # servers must outlive the run
    # Cells are CPU-only HTTP clients -> a long-limit non-GPU partition. Preemption is safe
    # because the runner resumes (skips completed cells / answered qids on re-submit).
    ap.add_argument("--cell-partition", default="mit_preemptable")
    ap.add_argument("--cell-time", default="2-00:00:00")
    ap.add_argument("--throttle", type=int, default=16, help="max concurrent array tasks")
    args = ap.parse_args()

    cells = load_sweep(args.config)
    run_root = io.run_dir(args.run_id)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cells_file = run_root / "cells.json"
    cells_file.write_text(json.dumps([c.to_dict() for c in cells], indent=2))
    print(f"[sweep] run_id={args.run_id}  cells={len(cells)}  -> {cells_file}")

    sizes = args.models or models_in_sweep(cells)
    print(f"[sweep] serving model sizes: {sizes}")
    for size in sizes:
        submit_server(size, str(run_root), args.serve_partition, args.serve_gpu_type, args.serve_time)

    if args.serve_only:
        print("[sweep] --serve-only: skipping cell array. Run the pilot/cells once servers are up.")
        return

    array_text = _render_array(
        args.run_id, str(cells_file), len(cells), args.cell_partition, args.cell_time,
        args.throttle, str(log_dir),
    )
    array_path = run_root / "run_cells.sbatch"
    array_path.write_text(array_text)
    import subprocess

    out = subprocess.run(["sbatch", str(array_path)], capture_output=True, text=True, check=True).stdout.strip()
    print(f"[sweep] cell array submitted: {out}  (sbatch: {array_path})")


if __name__ == "__main__":
    main()
