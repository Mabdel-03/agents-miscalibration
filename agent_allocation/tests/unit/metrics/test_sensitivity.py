"""metrics/sensitivity.py: exact-string versus mutual complete-link MATH classes."""
from __future__ import annotations

from dataclasses import replace

import pytest

from agent_allocation.metrics.sensitivity import math_class_sensitivity
from agent_allocation.paths import RunPaths
from agent_allocation.selection.seal import SealError, load_sealed_selections, seal_selections
from tests.fixtures.metrics.synthetic_panel import (
    SEED, MiniGrid, build_panel, fixture_math_equivalent, select_panel, standard_items,
)

GRID = MiniGrid(bank=6, private_cycles=2, team_sizes=(2,), rounds=1, deep=(), extension=False)


def test_merged_and_exact_readouts_are_both_reported(tmp_path):
    paths = RunPaths.for_run(tmp_path / "run", locks_dir=tmp_path / "locks", local_state_dir=tmp_path / "local")
    math = replace(standard_items()[0], answers={"root/0": "0.5", "root/1": "2", "root/2": "2", "root/3": "1/2",
                                                 "root/4": "\\frac{1}{2}", "root/5": "2.0"},
                   unknown_grades=frozenset())
    panel = build_panel([math], GRID, archive_views=False)
    seal_selections(paths, "main", select_panel(panel, math_equivalent=fixture_math_equivalent))
    sealed = load_sealed_selections(paths, "main")
    rows = {r.name: r for r in math_class_sensitivity(sealed, panel.records, panel.grades, seed=SEED,
                                                       math_equivalent=fixture_math_equivalent)}
    five = rows["sample/5"]
    assert five.reproduced and not five.same_selection
    assert five.merged_selected_candidate_id.endswith("root/0") and five.exact_selected_candidate_id.endswith("root/1")
    assert (five.merged_delivered_state, five.exact_delivered_state) == ("correct", "wrong")
    assert (five.n_classes_exact, five.n_classes_merged) == (4, 2) and five.equivalence_calls > 0
    assert "sample/1" not in rows                                   # single-candidate readouts are not plurality


def test_sensitivity_requires_a_seal(tmp_path):
    panel = build_panel(standard_items()[:1], GRID, archive_views=False)
    with pytest.raises(SealError):
        math_class_sensitivity(object(), panel.records, panel.grades, seed=SEED,  # type: ignore[arg-type]
                               math_equivalent=fixture_math_equivalent)
