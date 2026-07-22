#!/usr/bin/env python3
"""Freeze or verify benchmark Question contracts beside an existing cells manifest.

This is the audited attachment path for manifests created before benchmark-contract
sidecars existed.  It never writes ``cells.json`` or ``cells.sha256`` and refuses to
replace an existing sidecar.  Review the pinned source revisions before using this to
grandfather historical results: a newly created sidecar proves the selected contract from
this point forward, but cannot retroactively prove which prompt text an old unprovenanced
row consumed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agents_scaling.benchmarks.contracts import (
    FrozenBenchmarkContracts,
    freeze_benchmark_contracts,
    load_frozen_benchmark_contracts,
    verify_frozen_benchmark_questions,
)
from agents_scaling.benchmarks.loaders import load_benchmark
from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.experiment.manifest import ManifestSnapshot, load_manifest


def verify_current_questions(
    snapshot: ManifestSnapshot,
    frozen: FrozenBenchmarkContracts,
    *,
    benchmark_loader=load_benchmark,
) -> int:
    """Re-read and verify every unique normalized Question contract in a run."""

    return verify_frozen_benchmark_questions(
        snapshot,
        frozen,
        benchmark_loader=benchmark_loader,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="require and verify an existing sidecar without creating one",
    )
    args = parser.parse_args()

    run_root = args.results_root / args.run_id
    snapshot = load_manifest(run_root)
    frozen = (
        load_frozen_benchmark_contracts(run_root, snapshot=snapshot)
        if args.verify_only
        else freeze_benchmark_contracts(run_root, snapshot=snapshot)
    )
    verified_contracts = verify_current_questions(snapshot, frozen)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "manifest_sha256": snapshot.sha256,
                "manifest_cells": len(snapshot.cells),
                "benchmark_contracts": len(frozen.contracts_by_id),
                "verified_current_question_contracts": verified_contracts,
                "benchmark_contract_sidecar": str(frozen.path),
                "benchmark_contract_sidecar_sha256": frozen.sidecar_sha256,
                "manifest_contract_index_sha256": (
                    frozen.manifest_contract_index_sha256
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
