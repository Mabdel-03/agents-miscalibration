"""Production entrypoints fail closed against split-brain legacy admission."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_chunk_driver_requires_explicit_legacy_override():
    result = _run(
        "slurm/launch_chunked.py",
        "--config",
        "configs/full_sweep.yaml",
        "--run-id",
        "must_not_be_created",
    )
    assert result.returncode == 2
    assert "legacy per-run admission is disabled" in result.stderr


def test_loop_launcher_requires_explicit_legacy_override():
    result = _run(
        "slurm/launch_loops.py",
        "--run-id",
        "must_not_be_created",
        "--spec",
        "0.6B:1:pi_tpoggio:1:00:00",
    )
    assert result.returncode == 2
    assert "legacy per-run drivers are retired" in result.stderr


def test_direct_sweep_array_requires_explicit_legacy_override():
    result = _run(
        "slurm/launch_sweep.py",
        "--config",
        "configs/full_sweep.yaml",
        "--run-id",
        "must_not_be_created",
    )
    assert result.returncode == 2
    assert "direct sweep arrays are retired" in result.stderr


def test_keepalive_requires_explicit_legacy_override():
    result = _run(
        "slurm/keepalive.py",
        "--run-id",
        "must_not_be_created",
        "--spec",
        "0.6B:1:pi_tpoggio:1:00:00",
        "--once",
    )
    assert result.returncode == 2
    assert "legacy keepalive/registry repair is retired" in result.stderr


def test_old_global_dispatcher_submission_requires_explicit_override():
    result = _run(
        "slurm/launch_dispatcher.py",
        "--run",
        "must_not_be_created",
        "--submit",
    )
    assert result.returncode == 2
    assert "legacy global-dispatcher launcher is retired" in result.stderr
