"""Axis 4: ReasoningLevel enum mapping (rank, enable_thinking, thinking_budget)."""

import pytest

from agents_scaling.config import ExperimentCell, ReasoningLevel, Topology


def test_rank_is_strictly_ordered():
    order = [
        ReasoningLevel.OFF,
        ReasoningLevel.B512,
        ReasoningLevel.B2048,
        ReasoningLevel.B8192,
        ReasoningLevel.UNLIMITED,
    ]
    ranks = [r.rank for r in order]
    assert ranks == sorted(ranks) == [0, 1, 2, 3, 4]


def test_enable_thinking():
    assert ReasoningLevel.OFF.enable_thinking is False
    for r in [ReasoningLevel.B512, ReasoningLevel.B2048, ReasoningLevel.B8192, ReasoningLevel.UNLIMITED]:
        assert r.enable_thinking is True


def test_thinking_budget_mapping():
    assert ReasoningLevel.OFF.thinking_budget is None        # no thinking
    assert ReasoningLevel.B512.thinking_budget == 512
    assert ReasoningLevel.B2048.thinking_budget == 2048
    assert ReasoningLevel.B8192.thinking_budget == 8192
    assert ReasoningLevel.UNLIMITED.thinking_budget is None  # thinking on, no cap


def test_cell_coerces_string_reasoning_and_id_includes_it():
    cell = ExperimentCell(
        model_size="8B",
        context_share_level="artifact_only",
        prompt_complexity_level=1,
        reasoning_level="b2048",            # passed as a plain string (YAML)
        topology="single_agent",
        benchmark="gpqa",
    )
    assert isinstance(cell.reasoning_level, ReasoningLevel)
    assert cell.reasoning_level is ReasoningLevel.B2048
    assert "_rb2048_" in cell.cell_id
    # round-trips through to_dict/from_dict
    assert ExperimentCell.from_dict(cell.to_dict()).reasoning_level is ReasoningLevel.B2048


def test_distinct_reasoning_levels_give_distinct_cell_ids():
    base = dict(
        model_size="8B", context_share_level="artifact_only", prompt_complexity_level=1,
        topology="decentralized", benchmark="gpqa",
    )
    a = ExperimentCell(reasoning_level="off", **base)
    b = ExperimentCell(reasoning_level="unlimited", **base)
    assert a.cell_id != b.cell_id
