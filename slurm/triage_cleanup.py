"""One-shot triage: probe every registered endpoint, prune dead ones.

The keepalive's trust-the-job policy intentionally never prunes on probe failure, so the
registry accumulates dead entries from every server that ever lived. This script is the
hand-crank version of the read-time GC that Change A adds to the registry — useful as an
emergency reset after fleet churn, and as a smoke test that Change A works as intended.

Usage:
  PYTHONPATH=src python slurm/triage_cleanup.py --run-id full_sweep_v1
  PYTHONPATH=src python slurm/triage_cleanup.py --run-id full_sweep_v1 --dry-run
"""

from __future__ import annotations

import argparse
import os

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.serving import healthcheck, registry


_SIZES = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe and prune dead registry endpoints.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--probe-timeout", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_root = args.run_root or os.path.join(
        os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT), args.run_id
    )

    grand_live = grand_pruned = 0
    for sz in _SIZES:
        eps = registry.list_servers(run_root, sz)
        live = dead = 0
        for e in eps:
            if healthcheck.is_alive(e.host, e.port, timeout=args.probe_timeout):
                live += 1
            else:
                dead += 1
                if not args.dry_run:
                    registry.prune_entry(run_root, e)
        action = "would prune" if args.dry_run else "pruned"
        print(f"  {sz:5s}: live={live:>2d}  {action}={dead:>3d}  (registered={len(eps)})")
        grand_live += live
        grand_pruned += dead

    label = "would prune" if args.dry_run else "pruned"
    print(f"TOTAL: live={grand_live} {label}={grand_pruned}")


if __name__ == "__main__":
    main()
