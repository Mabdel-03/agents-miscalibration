"""The committed agent-07 fixtures under tests/fixtures/metrics/ are exactly what the code produces."""
from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.metrics.build_fixtures import HERE, build


def test_committed_fixtures_are_current(tmp_path):
    fresh = build(tmp_path)
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert manifest["files"] == dict(sorted(fresh.items()))
    for name in fresh:
        assert (HERE / name).read_bytes() == (tmp_path / name).read_bytes(), name
    assert (HERE / "manifest.json").read_bytes() == (tmp_path / "manifest.json").read_bytes()


def test_fixture_tables_are_populated():
    rows = [json.loads(line) for line in (HERE / "tables" / "checkpoints.jsonl").read_text().splitlines()]
    states = {r["delivered_state"] for r in rows}
    assert {"correct", "wrong", "invalid_output", "unknown", "abstained", "missing_selection"} <= states
    assert all(Path(HERE / "tables" / f"{t}.jsonl").stat().st_size > 0
               for t in ("bank", "slot_transitions", "team_transitions", "root_strata", "failures"))
