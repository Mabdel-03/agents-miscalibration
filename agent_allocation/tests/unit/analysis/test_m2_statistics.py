"""M2 statistics: item-clustered intervals, BBH strata, hand-computed primary contrasts, partial handling,
shared-subset high/low joins, and no outcome-dependent population changes."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from agent_allocation._vendor.ard_stats import mean_ci
from agent_allocation.analysis import contrasts as K
from agent_allocation.analysis.stats import build_population, column_intervals, value_matrix
from agent_allocation.analysis.summary import stratum_weights
from agent_allocation.contracts import BBH_SUBTASKS, GenerationKind, GradeResult, GradeStatus, dump_record
from agent_allocation.reporting.cli import analyze
from tests.unit.analysis.conftest import REPLICATES
from tests.unit.analysis.synthrun import Grid, SynthSpec, build_run


def _frame(values: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    """values[stratum][checkpoint] = per-item y list (item ids shared across checkpoints)."""
    rows = []
    for stratum, cols in values.items():
        for name, ys in cols.items():
            for i, y in enumerate(ys):
                rows.append({"partition_id": "main", "view": "current", "name": name, "stratum": stratum,
                             "item_id": f"{stratum}/item{i}", "y": y})
    return pd.DataFrame(rows)


def test_intervals_resample_items_and_keep_an_items_checkpoints_together():
    ys = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    frame = _frame({"math500": {"a": ys, "b": ys, "c": [1.0 - y for y in ys]}})
    columns = [("main", "a"), ("main", "b"), ("main", "c")]
    population = build_population(frame, columns, label="t")
    assert population.n_by_stratum == {"math500": 10}          # clusters are items, not rows (30 rows here)
    rs = population.resampler(400, 7)
    assert rs.idx["math500"].shape == (400, 10)
    matrix = value_matrix(frame, population, columns)
    ci = column_intervals(matrix, rs, ["a", "b", "c"])
    ci = ci[ci["level"] == "macro"].set_index("name")
    # identical columns share resample indices: identical intervals; complementary column mirrors them
    assert ci.loc["a", "ci_lo"] == ci.loc["b", "ci_lo"] and ci.loc["a", "ci_hi"] == ci.loc["b", "ci_hi"]
    assert ci.loc["c", "ci_lo"] == pytest.approx(1 - ci.loc["a", "ci_hi"])
    # the vendored engine on the same indices gives the same percentile interval
    ref = mean_ci({"math500": np.array(ys)}, rs, domains=["math500"], weights={"math500": 1.0})
    assert [ci.loc["a", "ci_lo"], ci.loc["a", "ci_hi"]] == pytest.approx(ref["ci95_percentile"])


def test_duplicated_candidate_rows_do_not_narrow_intervals():
    ys = [1.0, 0.0] * 6
    single = _frame({"math500": {"a": ys}})
    many = pd.concat([single] * 5, ignore_index=True)           # the same item rows repeated (e.g. per candidate)
    for frame in (single, many):
        pop = build_population(frame, [("main", "a")], label="dup")
        assert pop.n_items == 12
    a = column_intervals(value_matrix(single, build_population(single, [("main", "a")], label="dup"),
                                      [("main", "a")]), build_population(single, [("main", "a")], label="dup")
                         .resampler(300, 1), ["a"])
    b = column_intervals(value_matrix(many, build_population(many, [("main", "a")], label="dup"),
                                      [("main", "a")]), build_population(many, [("main", "a")], label="dup")
                         .resampler(300, 1), ["a"])
    assert a[["ci_lo", "ci_hi"]].to_numpy().tolist() == b[["ci_lo", "ci_hi"]].to_numpy().tolist()


def test_bbh_subtasks_are_resampling_strata():
    values = {f"bbh_logic4/{BBH_SUBTASKS[0]}": {"a": [1.0] * 5}, f"bbh_logic4/{BBH_SUBTASKS[1]}": {"a": [0.0] * 5},
              f"bbh_logic4/{BBH_SUBTASKS[2]}": {"a": [1.0] * 5}, f"bbh_logic4/{BBH_SUBTASKS[3]}": {"a": [0.0] * 5}}
    frame = _frame(values)
    pop = build_population(frame, [("main", "a")], label="bbh")
    ci = column_intervals(value_matrix(frame, pop, [("main", "a")]), pop.resampler(500, 3), ["a"])
    row = ci[ci["benchmark"] == "bbh_logic4"].iloc[0]
    # items never move across subtasks, so every replicate's equal-subtask mean is exactly 0.5
    assert row["estimate"] == pytest.approx(0.5) and row["ci_lo"] == pytest.approx(0.5) and row["ci_hi"] == pytest.approx(0.5)
    assert stratum_weights(list(values))[f"bbh_logic4/{BBH_SUBTASKS[0]}"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------------------------------------
# Hand-computed primary contrasts
# ---------------------------------------------------------------------------------------------------------

def primary_correct_fn(index: int, benchmark: str, spec) -> bool:
    """j = item position inside its wave block. See test docstring for the implied per-item outcomes."""
    j = index % 4
    kind, r = spec.kind, spec.update_round
    if kind == GenerationKind.PEER:
        return j in (0, 1) if r == 4 else j == 0
    if kind == GenerationKind.CP:
        return j in (0, 1, 2) if r == 3 else j == 0
    if kind == GenerationKind.PRIVATE:
        return j == 0
    return j == 0


@pytest.fixture(scope="module")
def primary_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("primary")
    return build_run(root, SynthSpec(waves=2, extension_waves=2, correct_fn=primary_correct_fn, retry_every=0,
                                     grid=Grid(team_sizes=(4,), deep_multipliers=())))


def _macro(frame: pd.DataFrame, contrast: str) -> pd.Series:
    return frame[(frame["contrast"] == contrast) & (frame["level"] == "macro")].iloc[0]


def test_primary_contrasts_match_hand_computation(primary_run):
    """peer/4/4 delivers 1 for j in {0,1}; private_team/4/4 for j=0 -> (i) = 0.25 in every benchmark and BBH subtask
    average. cp/3 delivers 1 for j in {0,1,2}, cp/0 and private_team/4/1 for j=0 -> (ii) = mean([0,1,1,0]) = 0.5.
    L = 0 and H(1) = H(2) = 0; H(3) equals (ii)."""
    analysis = analyze(primary_run.run_dir, replicates=REPLICATES, figures=False)
    frame = analysis.extras["primary"]
    i = _macro(frame, K.PRIMARY_I)
    ii = _macro(frame, K.PRIMARY_II)
    assert i["status"] == "complete" and i["estimate"] == pytest.approx(0.25) and i["n_items"] == 32
    assert ii["status"] == "complete" and ii["estimate"] == pytest.approx(0.5) and ii["n_items"] == 32
    for bench in ("math500", "mmlu_pro", "bbh_logic4", "mbpp_plus"):
        row = frame[(frame["contrast"] == K.PRIMARY_I) & (frame["benchmark"] == bench)].iloc[0]
        assert row["estimate"] == pytest.approx(0.25)
        assert pd.isna(row["p_value"])
    assert _macro(frame, "L")["estimate"] == pytest.approx(0.0)
    assert _macro(frame, "H(1)")["estimate"] == pytest.approx(0.0)
    assert _macro(frame, "H(2)")["estimate"] == pytest.approx(0.0)
    assert _macro(frame, "H(3)")["estimate"] == pytest.approx(ii["estimate"])
    assert 0 < i["p_value"] <= 1 and i["p_holm"] >= i["p_value"] and ii["p_holm"] >= ii["p_value"]
    assert analysis.extras["primary_estimates"] == {K.PRIMARY_I: pytest.approx(0.25), K.PRIMARY_II: pytest.approx(0.5)}
    assert frame[frame["role"] != "primary"]["p_value"].isna().all()
    exported = pd.read_csv(primary_run.tables_dir / "primary_contrasts.csv")
    assert exported[(exported["contrast"] == K.PRIMARY_I) & (exported["level"] == "macro")]["estimate"].iloc[0] \
        == pytest.approx(0.25)


def test_unknown_primary_outcome_gives_bounds_and_withholds_p_values(primary_run, tmp_path):
    from tests.unit.analysis.conftest import copy_run

    paths = copy_run(primary_run, tmp_path)
    analysis = analyze(paths.run_dir, replicates=REPLICATES, figures=False)
    row = analysis.outcomes[(analysis.outcomes["name"] == "peer/4/4") & (analysis.outcomes["view"] == "current")
                            & (analysis.outcomes["item_id"] == "math500/w0/i1")].iloc[0]
    grade_path = paths.grade(row["selected_candidate_id"])
    grade_path.write_text(json.dumps(dump_record(GradeResult(row["selected_candidate_id"], GradeStatus.EVALUATOR_ERROR,
                                                             None, None, "grader-1", "h"))))
    partial = analyze(paths.run_dir, replicates=REPLICATES, figures=False).extras["primary"]
    i = _macro(partial, K.PRIMARY_I)
    assert i["status"] == "partial" and i["n_items"] == 32 and i["n_complete"] == 31
    assert pd.isna(i["p_value"]) and pd.isna(i["p_holm"])
    # math diffs [0,1,0,0,0,1,0,0]; the unknown peer outcome is filled 0 (lower) or 1 (upper)
    assert i["bound_lower"] == pytest.approx(0.25 - 0.125 / 4)
    assert i["bound_upper"] == pytest.approx(0.25)
    ii = _macro(partial, K.PRIMARY_II)
    assert pd.isna(ii["p_holm"])                              # Holm needs both primaries complete


def test_high_vs_main_uses_shared_item_ids_only(two_wave_run):
    analysis = analyze(two_wave_run.run_dir, replicates=REPLICATES, figures=False)
    frame = analysis.extras["high_low"]
    assert not frame.empty
    macro = frame[frame["level"] == "macro"]
    assert set(macro["n_items"]) == {16}                      # high has wave 0 only; main has 32 items
    high_items = set(analysis.outcomes.loc[analysis.outcomes["partition_id"] == "high", "item_id"])
    assert all("/w0/" in i for i in high_items)
    per_bench = frame[frame["level"] == "benchmark"]
    assert set(per_bench["n_items"]) == {4}
    assert any("shared item ids" in n for n in analysis.extras["notes"])


def test_populations_do_not_depend_on_outcomes(tmp_path):
    def flipped(index, benchmark, spec):
        value = primary_correct_fn(index, benchmark, spec)
        return not value

    spec = dict(waves=1, extension_waves=1, retry_every=0, grid=Grid(team_sizes=(4,), deep_multipliers=()))
    a = analyze(build_run(tmp_path / "a", SynthSpec(correct_fn=primary_correct_fn, **spec)).run_dir,
                replicates=REPLICATES, figures=False)
    b = analyze(build_run(tmp_path / "b", SynthSpec(correct_fn=flipped, **spec)).run_dir,
                replicates=REPLICATES, figures=False)
    keys = ["contrast", "level", "benchmark", "n_items", "n_by_stratum"]
    assert a.extras["primary"][keys].equals(b.extras["primary"][keys])
    acc_keys = ["partition_id", "name", "view", "level", "benchmark", "n_planned"]
    assert a.accuracy[acc_keys].equals(b.accuracy[acc_keys])
    assert a.extras["populations"].equals(b.extras["populations"])
    assert not a.accuracy["acc"].equals(b.accuracy["acc"])
