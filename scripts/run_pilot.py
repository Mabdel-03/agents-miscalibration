#!/usr/bin/env python
"""Run the pilot cells locally (servers must already be up via launch_sweep --serve-only).

This runs every pilot cell in-process (sequentially), discovering each cell's vLLM server
through the registry. Use it for the end-to-end smoke test before submitting the array.

Usage:
  # 1) launch servers
  python slurm/launch_sweep.py --serve-only --config configs/pilot.yaml --run-id pilot1
  # 2) once servers register, run the cells
  python scripts/run_pilot.py --config configs/pilot.yaml --run-id pilot1
"""

from __future__ import annotations

import argparse

from agents_scaling.experiment.runner import run_cell
from agents_scaling.experiment.sweep import load_sweep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--judge-prompts", action="store_true")
    args = ap.parse_args()

    cells = load_sweep(args.config)
    print(f"[pilot] {len(cells)} cells")
    for i, cell in enumerate(cells):
        print(f"[pilot] ({i+1}/{len(cells)}) {cell.cell_id}")
        path = run_cell(cell, args.run_id, score_prompt_with_judge=args.judge_prompts)
        print(f"         -> {path}")


if __name__ == "__main__":
    main()
