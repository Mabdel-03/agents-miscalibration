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
from agents_scaling.study.types import EXIT_DONE  # noqa: E402

SMOKE_MANIFEST = "SMOKE"
AUDIT_MANIFEST = "SMOKE_AUDIT"


def smoke_manifests(run_root: Path, cfg: Any, inventory: PLAN.Inventory,
                    blocks: Sequence[str] | None = None) -> list[Path]:
    """One manifest per smoke block, in dispatch order (``plan.SMOKE_BLOCKS``).

    Every phase the launch gate scores gets a producer here.  The first version of this built
    the bank block alone, so four of the gate's five phases came back ``unmeasured`` -- the
    "both ends built, no middle" shape that rule R3 exists to catch, in the very step whose
    job is to catch it before the campaign starts.
    """
    dag = PLAN.smoke_dag(cfg, inventory)
    wanted = tuple(blocks) if blocks else PLAN.SMOKE_BLOCKS
    out: list[Path] = []
    for block in PLAN.SMOKE_BLOCKS:
        if block not in wanted:
            continue
        cells = dag.of_block(block)
        if not cells:
            continue
        path = CE.manifest_path(run_root, block, SMOKE_MANIFEST)
        CE.write_manifest(path, cells, block=block, meta={"stage": CE.STAGE_SMOKE})
        out.append(path)
    return out


def audit_manifest(run_root: Path, cfg: Any) -> Path | None:
    """The ``audit`` manifest, emitted after the episodes committed and their pools sealed."""
    cells = PLAN.smoke_audit_dag(run_root, cfg)
    if not cells:
        return None
    path = CE.manifest_path(run_root, CE.Block.AUDIT, AUDIT_MANIFEST)
    CE.write_manifest(path, cells, block=CE.Block.AUDIT, meta={"stage": CE.STAGE_SMOKE})
    return path


def endpoints_of(manifests: Sequence[Path]) -> list[tuple[str, str]]:
    """The ``(pool, lane)`` pairs the smoke's cells will ask for an endpoint in.

    The pair, not the pool alone: the E2b block puts its 8B and 32B cells in the *same* pool,
    so a `low` pool with a live 8B and no 32B satisfies a pool-level check and then stalls
    every 32B cell for the full wait.  ``cell.pool`` is a :class:`kinds.Pool`, whose ``str()``
    is ``Pool.LOW`` while its ``value`` is the directory name that names the sub-run-root.
    """
    # ``.value`` on both: ``Pool.LOW``/``Lane.SMALL`` are what ``str()`` gives, while ``low``
    # and ``8B`` are the sub-run-root directory and the serving-profile suffix they mean.
    pairs = {(str(getattr(cell.pool, "value", cell.pool)),
              str(getattr(cell.lane, "value", cell.lane)))
             for path in manifests for cell in CE.load_manifest(path) if cell.pool is not None}
    return sorted(pairs)


def missing_endpoints(run_root: Path, wanted: Sequence[tuple[str, str]]) -> list[str]:
    """Refuse up front when something the smoke needs has no registered endpoint.

    Without this the smoke sits for ``wait_s`` *per cell* against an empty pool and prints
    nothing at all -- the first live run waited half an hour on a `low` pool that had no
    replica because the fleet had been brought up on `deadline`.  A missing endpoint is an
    operator error with an obvious fix, so it is reported before anything is dispatched.
    """
    from agents_scaling.nextdyn import fleet as FL

    out: list[str] = []
    for pool, lane in wanted:
        if not FL.live_endpoints_by_lane(run_root, pool).get(lane):
            out.append(f"{pool}/{lane}")
    return out


def budget_reader(cfg: Any) -> Any:
    """``row -> the thinking budget that row must not have exceeded``, by phase.

    Per phase rather than one ceiling, because the receiver budgets go to 4096 and the main
    phases run at 2048: a single maximum would have let a 2048-budget draw reach 4096 without
    the check noticing, which is exactly the failure being looked for.  Within the receiver's
    own two budgets it stays loose -- the row carries ``budget_hash``, not the value -- so a
    1024 receiver draw is only caught above 4096.  The main phases, where the study's 2048 cap
    lives and where ARD36 has no measurement, are exact.
    """
    from agents_scaling.nextdyn import contract as NK

    receiver = max(int(b) for b in cfg.budgets.receiver_thinking_tokens)
    ceilings = {
        NK.PHASE_BANK: int(cfg.contract.tokens.thinking_cap),
        NK.PHASE_REVISION: int(cfg.contract.tokens.thinking_cap),
        NK.PHASE_RECEIVER_PLAN: receiver,
        NK.PHASE_RECEIVER_DECISION: receiver,
        NK.PHASE_AUDIT: int(cfg.contract.audit_tokens.thinking_cap),
    }
    default = max(ceilings.values())
    return lambda row: ceilings.get(str(getattr(row, "phase", "")), default)


def drive(run_root: Path, manifest: Path, cfg_path: Path, *, wait_s: float) -> list[int]:
    cells = CE.load_manifest(manifest)
    codes: list[int] = []
    for index in range(len(cells)):
        code = run_one.main(["--run-root", str(run_root), "--config", str(cfg_path),
                             "--manifest", str(manifest), "--index", str(index),
                             "--wait-s", str(wait_s)])
        codes.append(code)
        print(f"[smoke] {manifest.stem} cell {index + 1}/{len(cells)} -> exit {code}", flush=True)
    return codes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=C.DEFAULT_STUDY_CONFIG)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wait-s", type=float, default=600.0,
                        help="how long a cell waits for a live endpoint before exit 2")
    parser.add_argument("--measure-only", action="store_true",
                        help="skip the dispatch and measure what is already committed")
    parser.add_argument("--allow-empty-pools", action="store_true",
                        help="dispatch even when a pool has no registered endpoint")
    parser.add_argument("--blocks", default=None,
                        help="comma-separated subset of %s; the manifests are the same cells "
                             "either way, so a later run finishes what this one left"
                             % ",".join(PLAN.SMOKE_BLOCKS))
    parser.add_argument("--skip-audit", action="store_true",
                        help="do not emit the audit manifest (its phase then goes unmeasured)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = C.load_study(args.config)
    compute = C.load_compute(C.sibling_config(args.config, C.DEFAULT_COMPUTE_CONFIG.name)
                             or C.DEFAULT_COMPUTE_CONFIG)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    codes: list[int] = []
    if not args.measure_only:
        try:
            inventory = PLAN.Inventory.from_run_root(run_root, cfg)
        except Exception as exc:                     # noqa: BLE001 - reported, not a traceback
            print(f"[smoke] no inventory under {run_root}: {exc}", file=sys.stderr)
            return 2
        blocks = tuple(b.strip() for b in args.blocks.split(",")) if args.blocks else None
        unknown = sorted(set(blocks or ()) - set(PLAN.SMOKE_BLOCKS))
        if unknown:
            print(f"[smoke] not smoke blocks: {unknown}; choose from {list(PLAN.SMOKE_BLOCKS)}",
                  file=sys.stderr)
            return 2
        manifests = smoke_manifests(run_root, cfg, inventory, blocks)
        wanted = endpoints_of(manifests)
        missing = missing_endpoints(run_root, wanted)
        if missing and not args.allow_empty_pools:
            need = ", ".join(f"{pool}/{lane}" for pool, lane in wanted)
            print(f"[smoke] no registered endpoint for {missing}; the smoke needs {need}.  "
                  "Bring them up (`nextdyn fleet up --pool <p> --spec …`) or pass "
                  "--allow-empty-pools to wait anyway", file=sys.stderr)
            return 2
        for path in manifests:
            n = len(CE.load_manifest(path))
            print(f"[smoke] {path.stem}: {n} cells", flush=True)
            codes += drive(run_root, path, args.config, wait_s=args.wait_s)
        audit = None if args.skip_audit else audit_manifest(run_root, cfg)
        if args.skip_audit:
            print("[smoke] --skip-audit: the audit phase will be unmeasured", file=sys.stderr)
        elif audit is None:
            print("[smoke] no pool sealed, so no audit cell: the launch gate will score the "
                  "audit phase unmeasured, which is a failure", file=sys.stderr)
        else:
            print(f"[smoke] {audit.stem}: {len(CE.load_manifest(audit))} cells", flush=True)
            codes += drive(run_root, audit, args.config, wait_s=args.wait_s)
        print(f"[smoke] {len(codes)} cells -> {sorted(set(codes))}")
        if any(code != EXIT_DONE for code in codes):
            print("[smoke] some cells did not finish; the measurement below covers what did",
                  file=sys.stderr)

    # The budget a row was drawn under, so the measurement can say whether anything ran past
    # it.  Read from the contract rather than assumed: the receiver phases run at B/2.
    measured = G.smoke_measurements(run_root, budget_of=budget_reader(cfg))
    if not measured:
        print(f"[smoke] no decisions under {run_root}: nothing to measure", file=sys.stderr)
        return 1
    path = G.write_measured(run_root, measured)
    for phase, row in sorted(measured.items()):
        print(f"[smoke] {phase:<18} n={row.n_calls:<5} parse_valid={row.parse_valid:.3f} "
              f"boundary={row.boundary_rate:.3f} continuation={row.continuation_rate:.3f} "
              f"gen_tokens={row.generated_tokens:.0f} latency={row.latency_s:.1f}s "
              f"max_think={row.max_thinking_tokens} over_budget={row.over_budget_draws}")
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
