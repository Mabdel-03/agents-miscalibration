"""SLURM array entrypoint: run the cell at ``--index`` from a cells JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents_scaling.config import ExperimentCell
from agents_scaling.experiment.runner import run_cell


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one sweep cell by array index.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cells-file", required=True, help="JSON list of ExperimentCell dicts")
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--judge-prompts", action="store_true", help="score prompt quality with LLM judge")
    args = ap.parse_args()

    cells = json.loads(Path(args.cells_file).read_text())
    if not 0 <= args.index < len(cells):
        raise IndexError(f"index {args.index} out of range for {len(cells)} cells")
    cell = ExperimentCell.from_dict(cells[args.index])
    path = run_cell(cell, args.run_id, score_prompt_with_judge=args.judge_prompts)
    print(f"[run_one] cell {cell.cell_id} -> {path}")


if __name__ == "__main__":
    main()
