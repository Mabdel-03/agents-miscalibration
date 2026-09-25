from __future__ import annotations

from pathlib import Path

import pytest

LEGACY_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/artifacts")


@pytest.fixture(scope="session")
def legacy_root() -> Path:
    if not (LEGACY_ROOT / "shards" / "mancer" / "manifests" / "jobs" / "main.jsonl").exists():
        pytest.skip("legacy forced_budget_v2 run root not available")
    return LEGACY_ROOT
