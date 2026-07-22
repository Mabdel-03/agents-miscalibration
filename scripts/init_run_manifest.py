#!/usr/bin/env python3
"""Create and freeze a new sweep manifest without submitting any jobs.

Run manifests are immutable scheduler inputs.  This initializer refuses to touch an
existing ``cells.json`` so extending a sweep always requires a new run id.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

# Permit direct execution from a clean immutable release worktree.
REPO = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agents_scaling.benchmarks.contracts import (  # noqa: E402
    build_run_benchmark_contracts,
    freeze_benchmark_contracts,
)
from agents_scaling.benchmarks.loaders import load_benchmark  # noqa: E402
from agents_scaling.benchmarks.schema import Question  # noqa: E402
from agents_scaling.config import DEFAULT_RESULTS_ROOT  # noqa: E402
from agents_scaling.experiment import io  # noqa: E402
from agents_scaling.experiment.manifest import (  # noqa: E402
    ManifestSnapshot,
    freeze_manifest,
    load_manifest,
)
from agents_scaling.experiment.sweep import load_sweep  # noqa: E402


def initialize(
    *,
    config: Path,
    run_root: Path,
    expected_cells: int | None = None,
    benchmark_loader: Callable[..., list[Question]] = load_benchmark,
) -> tuple[int, str]:
    run_root = Path(run_root)
    run_root.parent.mkdir(parents=True, exist_ok=True)
    # One persistent parent-level lock avoids the unsafe unlink/recreate race of
    # per-run lock files and serializes the rare manifest publication operation.
    lock_path = run_root.parent / ".benchmark-manifest-initializer.lock"
    lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_fd = os.open(lock_path, lock_flags, 0o644)
    stage: Path | None = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # ``Path.exists`` is false for a dangling symlink; replacing one would still be
        # destructive, so reject either kind of pre-existing run path.
        if run_root.exists() or run_root.is_symlink():
            raise FileExistsError(
                f"refusing to regenerate immutable run {run_root}; use a new run id"
            )

        cells = load_sweep(config)
        if expected_cells is not None and len(cells) != expected_cells:
            raise ValueError(
                f"generated {len(cells)} cells from {config}, expected {expected_cells}"
            )
        ids = [cell.cell_id for cell in cells]
        if len(ids) != len(set(ids)):
            raise ValueError(f"generated duplicate cell ids from {config}")

        payload = json.dumps([cell.to_dict() for cell in cells], indent=2) + "\n"
        payload_bytes = payload.encode("utf-8")
        planned_snapshot = ManifestSnapshot(
            path=run_root / "cells.json",
            cells=tuple(cells),
            sha256=hashlib.sha256(payload_bytes).hexdigest(),
        )
        # Resolve and hash every external benchmark contract before creating even a
        # staging artifact. Dataset/authentication/content failures therefore leave no
        # run-shaped directory behind.
        contract_payload = build_run_benchmark_contracts(
            planned_snapshot,
            benchmark_loader=benchmark_loader,
        )

        # Build and fully verify the four-file immutable bundle in a hidden sibling.
        # The sole publication operation is a same-filesystem directory rename, so a
        # hard kill exposes either no run or the complete verified run, never a partial
        # cells/sidecar combination. Stale hidden stages from hard kills are harmless to
        # retry and are deliberately never interpreted as production runs.
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{run_root.name}.initialize.", dir=run_root.parent
            )
        )
        cells_path = stage / "cells.json"
        io.atomic_write_text(cells_path, payload)
        freeze_manifest(stage)
        snapshot = load_manifest(stage)
        frozen_contracts = freeze_benchmark_contracts(
            stage,
            snapshot=snapshot,
            payload=contract_payload,
        )
        if frozen_contracts.manifest_sha256 != snapshot.sha256:
            raise RuntimeError(
                "staged benchmark contract does not bind the cells manifest"
            )
        if cells_path.read_bytes() != payload_bytes:
            raise RuntimeError("staged cells manifest changed before publication")

        if run_root.exists() or run_root.is_symlink():
            raise FileExistsError(
                f"run path appeared during initialization: {run_root}"
            )
        os.rename(stage, run_root)
        stage = None
        _fsync_directory(run_root.parent)
        return len(snapshot.cells), snapshot.sha256
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _fsync_directory(path: Path) -> None:
    """Durably record the atomic publication rename when the filesystem permits it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some shared/network filesystems provide atomic rename but reject directory
        # fsync. This matches the repository's durable I/O contract.
        pass
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)),
    )
    parser.add_argument("--expected-cells", type=int)
    args = parser.parse_args()

    count, digest = initialize(
        config=args.config,
        run_root=args.results_root / args.run_id,
        expected_cells=args.expected_cells,
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "manifest_cells": count,
                "manifest_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
