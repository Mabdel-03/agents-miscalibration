"""Regenerate agent 06's metric-table fixtures for agent 07: ``python -m tests.fixtures.metrics.build_fixtures``.

Output (this directory): ``tables/*.jsonl`` (one canonical JSON object per row, keys sorted), ``selections.jsonl``
(the sealed ``SelectionResult`` lines), ``schema.json`` (columns, declared types and value vocabularies) and
``manifest.json`` (versions, seed and sha256 of every file). ``tests/unit/metrics/test_fixtures.py`` fails when the
committed files drift from the code.
"""
from __future__ import annotations

import json
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

from agent_allocation.contracts import CheckStatus, canonical_json
from agent_allocation.metrics.io import write_tables
from agent_allocation.metrics.outcomes import CandidateState, DeliveredState
from agent_allocation.metrics.tables import METRICS_VERSION, TABLE_SCHEMAS, build_partition_tables
from agent_allocation.metrics.transitions import TRANSITION_KINDS
from agent_allocation.paths import RunPaths
from agent_allocation.selection.readout import SELECTOR_VERSION
from agent_allocation.selection.seal import load_sealed_selections, seal_selections
from tests.fixtures.metrics.synthetic_panel import (
    SEED, MiniGrid, build_panel, fixture_math_equivalent, select_panel, standard_items,
)

HERE = Path(__file__).resolve().parent
SUMMARY_COLUMNS = ("partition", "benchmark", "name", "view", "family", "n_items", "delivered_lo", "delivered_hi",
                   "delivered", "coverage_lo", "coverage_hi", "archive_coverage_lo", "archive_coverage_hi",
                   "selection_gap_lo", "selection_gap_hi",
                   *(f"n_delivered_{state.value}" for state in DeliveredState), "blocked")


def build(out_dir: Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    panel = build_panel(standard_items(), MiniGrid())
    with tempfile.TemporaryDirectory() as tmp:
        paths = RunPaths.for_run(Path(tmp) / "run", locks_dir=Path(tmp) / "locks", local_state_dir=Path(tmp) / "local")
        seal_selections(paths, panel.partition, select_panel(panel, math_equivalent=fixture_math_equivalent))
        sealed = load_sealed_selections(paths, panel.partition)
        selections = paths.selections(panel.partition).read_bytes()
    tables = build_partition_tables(sealed, panel.records, panel.grades)
    digests = {f"tables/{name}": digest for name, digest in write_tables(tables, out_dir / "tables").items()}
    (out_dir / "selections.jsonl").write_bytes(selections)
    digests["selections.jsonl"] = sha256(selections).hexdigest()
    schema = {
        "tables": {name: [{"name": col, "type": typ} for col, typ in columns] for name, columns in TABLE_SCHEMAS.items()},
        "summary_checkpoints": list(SUMMARY_COLUMNS),
        "vocabularies": {
            "candidate_state": [s.value for s in CandidateState],
            "delivered_state": [s.value for s in DeliveredState],
            "transition_kind": list(TRANSITION_KINDS),
            "root_stratum": ["none_correct", "correct_minority", "correct_half", "correct_majority", "all_correct",
                             "unknown"],
            "public_check_status": [s.value for s in CheckStatus],
        },
    }
    body = (json.dumps(schema, indent=2, sort_keys=True) + "\n").encode()
    (out_dir / "schema.json").write_bytes(body)
    digests["schema.json"] = sha256(body).hexdigest()
    manifest = {"generator": "tests/fixtures/metrics/build_fixtures.py", "metrics_version": METRICS_VERSION,
                "selector_version": SELECTOR_VERSION, "seed": SEED, "grid": MiniGrid().__dict__,
                "items": [i.task.item_id for i in panel.records.items], "files": dict(sorted(digests.items()))}
    (out_dir / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return digests


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    for name, digest in sorted(build(target).items()):
        print(f"{digest}  {name}")
