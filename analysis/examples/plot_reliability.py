#!/usr/bin/env python
"""Plot a reliability diagram from a completed cell's raw JSONL results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from agents_scaling.calibration.metrics import compute_calibration
from agents_scaling.experiment import io


def _load_rows(run_id: str, cell_id: str) -> list[dict]:
    path = io.results_root() / run_id / "cells" / cell_id / "results.jsonl"
    if not path.exists():
        raise SystemExit(
            f"missing results file: {path}\n"
            "Run a cell first, or choose a completed cell under "
            f"{io.results_root() / run_id / 'cells'}."
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"results file has no rows: {path}")
    return rows


def _pairs(rows: list[dict], signal: str) -> tuple[list[float], list[bool]]:
    confs: list[float] = []
    corrects: list[bool] = []

    if signal == "per_agent":
        for row in rows:
            key = row["answer_key"]
            for agent in row.get("per_agent", []):
                ans = agent.get("answer")
                probs = agent.get("option_logprobs") or {}
                if ans is None or not probs:
                    continue
                confs.append(float(probs.get(ans, 0.0)))
                corrects.append(ans == key)
        return confs, corrects

    prefix = "system:"
    if not signal.startswith(prefix):
        raise SystemExit("signal must be 'per_agent' or 'system:<confidence_key>'")
    key = signal[len(prefix) :]
    for row in rows:
        system_conf = row.get("system_conf") or {}
        if key not in system_conf:
            continue
        confs.append(float(system_conf[key]))
        corrects.append(bool(row["correct"]))
    return confs, corrects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--signal", default="per_agent")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args()

    rows = _load_rows(args.run_id, args.cell_id)
    confs, corrects = _pairs(rows, args.signal)
    if not confs:
        raise SystemExit(f"no calibration pairs found for signal {args.signal!r}")

    report = compute_calibration(confs, corrects, n_bins=args.bins)
    mids = [(b.lo + b.hi) / 2.0 for b in report.bins]
    acc = [b.accuracy if b.count else float("nan") for b in report.bins]
    counts = [b.count for b in report.bins]

    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    ax1.plot(mids, acc, marker="o", color="#2563eb", label="empirical accuracy")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"{args.cell_id}\n{args.signal} ECE={report.ece:.3f}, n={report.n}")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(mids, counts, width=1.0 / args.bins * 0.85, color="#94a3b8", alpha=0.35)
    ax2.set_ylabel("Count")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
