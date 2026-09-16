#!/usr/bin/env python3
"""``validate --smoke`` on a live 1+1 fleet: measure, never assume (plan §3.5 S1, spec 19.1).

Twenty solver calls per model and adapter plus a handful of complete synthetic receiver
episodes, then one file: ``smoke/measured.json`` in the schema ``gates.launch_gate`` reads.
What it is for is the four facts the whole contract rests on, each of which ARD36 assumed and
one of which it got wrong:

1. **the thinking boundary is exact** -- ``boundary_injected == (thinking_tokens == budget)``
   at *this* study's 2048 cap (it was verified at 1024 and 4096, not at 2048);
2. **the response field is where the client reads it** (``choices[0].message.reasoning``);
3. **the JSON object parses** at the rate the launch gate requires, *per phase* -- the ARD36
   defect was a call-type table keyed on a vocabulary the engine never emitted, so one phase's
   validity was silently reported as another's;
4. **the continuation path fires**, and only on ``boundary_injected``.

It is a measuring instrument, not a gate: it writes what it saw and exits 0 even when the
numbers are bad.  ``nextdyn gates launch --measured …`` is what decides.

    PYTHONPATH=src python scripts/nextdyn_smoke.py --run-root $RUN --config $CFG
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from agents_scaling.nextdyn import cells as CE  # noqa: E402
from agents_scaling.nextdyn import config as C  # noqa: E402
from agents_scaling.nextdyn import gates as G  # noqa: E402
from agents_scaling.nextdyn import plan as PLAN  # noqa: E402
from agents_scaling.nextdyn import run_one  # noqa: E402
from agents_scaling.nextdyn.kinds import Lane  # noqa: E402
from agents_scaling.study.types import EXIT_DONE  # noqa: E402

SMOKE_BLOCK = CE.Block.BANKS
SMOKE_MANIFEST = "SMOKE"


def smoke_manifest(run_root: Path, cfg: Any, inventory: PLAN.Inventory) -> Path:
    """One tiny manifest: the bank block at the ``pilot.smoke_solver_calls_per_model`` size.

    A bank cell is the right shape for the smoke: one call per draw, no dependency, every
    model and adapter reachable by changing one cell field -- and the draws it commits are
    real draws the pilot then reuses through the store rather than throwing away.
    """
    calls = int(cfg.pilot.smoke_solver_calls_per_model)
    per_family = max(calls // max(len(inventory.families), 1), 1)
    reduced = inventory.limited(per_family=per_family)
    dag = PLAN.build_dag(cfg, reduced, stage=CE.STAGE_SMOKE, blocks=(SMOKE_BLOCK,))
    path = CE.manifest_path(run_root, SMOKE_BLOCK, SMOKE_MANIFEST)
    CE.write_manifest(path, dag.of_block(SMOKE_BLOCK), block=SMOKE_BLOCK,
                      meta={"stage": CE.STAGE_SMOKE, "solver_calls_per_model": calls})
    return path


def drive(run_root: Path, manifest: Path, cfg_path: Path, *, wait_s: float) -> list[int]:
    cells = CE.load_manifest(manifest)
    return [run_one.main(["--run-root", str(run_root), "--config", str(cfg_path),
                          "--manifest", str(manifest), "--index", str(index),
                          "--wait-s", str(wait_s)])
            for index in range(len(cells))]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=C.DEFAULT_STUDY_CONFIG)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wait-s", type=float, default=600.0,
                        help="how long a cell waits for a live endpoint before exit 2")
    parser.add_argument("--measure-only", action="store_true",
                        help="skip the dispatch and measure what is already committed")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = C.load_study(args.config)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    codes: list[int] = []
    if not args.measure_only:
        try:
            inventory = PLAN.Inventory.from_run_root(run_root, cfg)
        except Exception as exc:                     # noqa: BLE001 - reported, not a traceback
            print(f"[smoke] no inventory under {run_root}: {exc}", file=sys.stderr)
            return 2
        manifest = smoke_manifest(run_root, cfg, inventory)
        codes = drive(run_root, manifest, args.config, wait_s=args.wait_s)
        print(f"[smoke] {len(codes)} cells -> {sorted(set(codes))}")
        if any(code != EXIT_DONE for code in codes):
            print("[smoke] some cells did not finish; the measurement below covers what did",
                  file=sys.stderr)

    measured = G.smoke_measurements(run_root)
    if not measured:
        print(f"[smoke] no decisions under {run_root}: nothing to measure", file=sys.stderr)
        return 1
    path = G.write_measured(run_root, measured)
    for phase, row in sorted(measured.items()):
        print(f"[smoke] {phase:<18} n={row.n_calls:<5} parse_valid={row.parse_valid:.3f} "
              f"boundary={row.boundary_rate:.3f} continuation={row.continuation_rate:.3f} "
              f"gen_tokens={row.generated_tokens:.0f} latency={row.latency_s:.1f}s")
    uncovered = [phase for phase in G.GENERATING_PHASES if phase not in measured]
    if uncovered:
        print(f"[smoke] phases with no measured call: {uncovered} -- the launch gate will "
              "report them as unmeasured, which is a failure, not a pass", file=sys.stderr)
    print(f"[smoke] wrote {path}")
    print(f"[smoke] next: nextdyn gates launch --run-root {run_root} --measured {path} "
          "--required-replicas <floor>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
