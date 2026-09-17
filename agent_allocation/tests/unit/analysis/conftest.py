from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent_allocation.paths import RunPaths
from tests.unit.analysis.synthrun import Grid, SynthSpec, build_run

REPLICATES = 60  # 8 CPUs shared by 8 agents: keep bootstrap work tiny in unit tests


@pytest.fixture(scope="session")
def built_run(tmp_path_factory) -> RunPaths:
    """One wave (16 items: 4 per benchmark, one per BBH subtask), extension on, n in {2,4}, r<=4."""
    root = tmp_path_factory.mktemp("synth_one_wave")
    return build_run(root, SynthSpec(waves=1, extension_waves=1, grid=Grid(team_sizes=(2, 4))))


@pytest.fixture(scope="session")
def two_wave_run(tmp_path_factory) -> RunPaths:
    """Two waves (two items per BBH subtask) so every stratum has >= 2 clusters; high partition on wave 0."""
    root = tmp_path_factory.mktemp("synth_two_wave")
    return build_run(root, SynthSpec(waves=2, extension_waves=2, high_waves=1, grid=Grid(team_sizes=(4,))))


def copy_run(paths: RunPaths, dest: Path) -> RunPaths:
    """Mutable copy of a session run (run dir + sibling locks)."""
    shutil.copytree(paths.run_dir, dest / "run")
    return RunPaths.for_run(dest / "run", locks_dir=dest / "locks", local_state_dir=dest / "local")


@pytest.fixture
def mutable_run(built_run, tmp_path) -> RunPaths:
    return copy_run(built_run, tmp_path)
