#!/usr/bin/env python3
"""Render (and optionally submit) the durable global-dispatcher control job.

Rendering is the default and has no scheduler side effects.  Pass ``--submit`` only
after legacy driver successors have been stopped and pending legacy arrays audited.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.config import DEFAULT_DISPATCHER_STATE_DIRNAME, DEFAULT_RESULTS_ROOT
from agents_scaling.experiment import io


TEMPLATE = REPO / "slurm" / "run_global_dispatcher.sbatch.tmpl"


def _build_dispatch_args(args: argparse.Namespace, state_dir: Path) -> list[str]:
    values = [
        "dispatch",
        "--results-root", str(Path(args.results_root).expanduser().resolve()),
        "--state-dir", str(state_dir),
        "--qos-limit", str(args.qos_limit),
        "--reserve", str(args.reserve),
        "--max-batch", str(args.max_batch),
        "--poll-seconds", str(args.poll_seconds),
        "--cell-partition", args.cell_partition,
        "--cell-time", args.cell_time,
        "--cell-mem", args.cell_mem,
        "--fanout-slots-per-server", str(args.fanout_slots_per_server),
        "--validation-budget", str(args.validation_budget),
    ]
    for run in args.run:
        values.extend(["--run", run])
    for pool in args.server_pool:
        values.extend(["--server-pool", pool])
    for weight in args.weight:
        values.extend(["--weight", weight])
    if args.probe_servers:
        values.append("--probe-servers")
    return values


def render_control(args: argparse.Namespace) -> Path:
    results_root = Path(args.results_root).expanduser().resolve()
    state_dir = (
        Path(args.state_dir).expanduser().resolve()
        if args.state_dir
        else results_root / DEFAULT_DISPATCHER_STATE_DIRNAME
    )
    log_dir = state_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    target = state_dir / "global_dispatcher.sbatch"
    dispatch_args = shlex.join(_build_dispatch_args(args, state_dir))
    text = TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "CONTROL_PARTITION": args.control_partition,
        "CONTROL_TIME": args.control_time,
        "LOG_DIR": str(log_dir),
        "REPO": str(REPO),
        "DISPATCH_ARGS": dispatch_args,
        "CONTROL_SBATCH": str(target),
    }
    for name, value in replacements.items():
        text = text.replace("{" + name + "}", value)
    io.atomic_write_text(target, text)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="RUN_ID[=RUN_ROOT]")
    parser.add_argument("--server-pool", action="append", default=[])
    parser.add_argument("--weight", action="append", default=[])
    parser.add_argument(
        "--results-root",
        default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT),
    )
    parser.add_argument("--state-dir")
    parser.add_argument("--qos-limit", type=int, default=448)
    parser.add_argument("--reserve", type=int, default=64)
    parser.add_argument("--max-batch", type=int, default=24)
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--cell-partition", default="mit_normal")
    parser.add_argument("--cell-time", default="12:00:00")
    parser.add_argument("--cell-mem", choices=("2G", "4G"), default="4G")
    parser.add_argument("--fanout-slots-per-server", type=int, default=24)
    parser.add_argument("--validation-budget", type=int, default=64)
    parser.add_argument("--probe-servers", action="store_true")
    parser.add_argument(
        "--control-partition", default="mit_preemptable,mit_normal,mit_normal_gpu"
    )
    parser.add_argument("--control-time", default="12:00:00")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="submit the initial control job; default only renders the auditable sbatch",
    )
    parser.add_argument(
        "--allow-legacy-dispatcher",
        action="store_true",
        help=(
            "explicit emergency override for submitting the retired global-dispatcher; "
            "schema-5 production uses schema5_control.py"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.submit and not args.allow_legacy_dispatcher:
        parser.error(
            "the legacy global-dispatcher launcher is retired: use "
            "slurm/schema5_control.py; pass --allow-legacy-dispatcher only for an "
            "audited emergency rollback"
        )
    path = render_control(args)
    print(f"[dispatcher-launch] rendered {path}")
    if not args.submit:
        print("[dispatcher-launch] render-only; no job submitted (pass --submit when ready)")
        return 0
    proc = subprocess.run(
        [
            "sbatch",
            "--parsable",
            "--export=ALL,ASYS_ALLOW_LEGACY_CONTROL=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"sbatch failed: {proc.stderr.strip()[:500]}")
    print(f"[dispatcher-launch] submitted job {proc.stdout.strip().split(';', 1)[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
