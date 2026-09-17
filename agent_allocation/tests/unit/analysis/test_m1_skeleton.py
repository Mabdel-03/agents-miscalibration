"""M1: deterministic report skeleton, fixed-population and sealed-before-join invariants, tri-state outcomes."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from agent_allocation.analysis.load import load_run, resolve_config
from agent_allocation.analysis.outcomes import outcome_table
from agent_allocation.analysis.summary import accuracy_summary, stratum_weights
from agent_allocation.contracts import (
    BBH_SUBTASKS, GenerationKind, GradeResult, GradeStatus, SelectionResult, canonical_json, dump_record, load_record,
)
from agent_allocation.reporting.audit import cmd_audit
from agent_allocation.reporting.cli import analyze, cmd_analyze
from tests.unit.analysis.conftest import REPLICATES, copy_run
from tests.unit.analysis.synthrun import Grid, SynthSpec, build_run, reseal, selection_lines, write_selection_batches


def _load(paths):
    cfg, source = resolve_config(paths)
    return load_run(paths, cfg, config_source=source)


def _files(directory: Path) -> dict[str, bytes]:
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_analyze_writes_deterministic_reports_and_tables(built_run, tmp_path):
    first = copy_run(built_run, tmp_path / "a")
    second = copy_run(built_run, tmp_path / "b")
    assert cmd_analyze(["--run-dir", str(first.run_dir), "--replicates", str(REPLICATES)]) == 0
    assert cmd_analyze(["--run-dir", str(second.run_dir), "--replicates", str(REPLICATES)]) == 0
    reports_a, reports_b = _files(first.reports_dir), _files(second.reports_dir)
    assert {"RUN_REPORT.md", "RUN_REPORT.html", "COMPLETENESS.md", "COMPLETENESS.html"} <= set(reports_a)
    assert reports_a == reports_b
    assert _files(first.tables_dir) == _files(second.tables_dir)
    assert _files(first.figures_dir) == _files(second.figures_dir) and _files(first.figures_dir)
    for name in ("policy_outcomes", "accuracy_summary", "completeness", "candidates"):
        assert (first.tables_dir / f"{name}.parquet").is_file() and (first.tables_dir / f"{name}.csv").is_file()
    text = (first.reports_dir / "RUN_REPORT.md").read_text()
    assert "BBH four-subtask zero-shot subset" in text
    assert "public-test selector" in text
    assert "not a FLOP-matched" in text
    assert "INTERIM" not in text and "Final: every planned checkpoint is known and costs are reconciled" in text


def test_hand_computed_fixture_matches_exports(tmp_path):
    """private_single/1 == root[0]. math 3/4, mmlu 1/4 (+1 evaluator error), BBH 2 of 4 subtasks, mbpp 0/4 with
    one invalid output. Macro bounds: lower (0.75+0.25+0.5+0)/4, upper (0.75+0.5+0.5+0)/4."""
    correct_roots = {"math500": {0, 1, 2}, "mmlu_pro": {0}, "bbh_logic4": {0, 1}, "mbpp_plus": set()}

    def correct_fn(index, benchmark, spec):
        j = index % 4
        if benchmark == "mbpp_plus" and j == 3:
            return None
        if spec.kind == GenerationKind.ROOT and spec.root_slot == 0:
            return j in correct_roots[benchmark]
        return j == 0

    paths = build_run(tmp_path, SynthSpec(waves=1, extension_waves=0, correct_fn=correct_fn,
                                          grid=Grid(bank_size=2, private_single_cycles=2, team_sizes=(2,), rounds=1,
                                                    deep_multipliers=(), archive_views=False)))
    run = _load(paths)
    target = next(cp for cp in run.policies["main"] if cp.item_id == "mmlu_pro/w0/i3" and cp.name == "private_single/1")
    grade_path = paths.grade(target.candidate_ids[0])
    grade_path.write_text(json.dumps(dump_record(GradeResult(target.candidate_ids[0], GradeStatus.EVALUATOR_ERROR,
                                                             None, None, "grader-1", "h"))))
    analysis = analyze(paths.run_dir, replicates=REPLICATES, figures=False)
    table = pd.read_csv(paths.tables_dir / "accuracy_summary.csv")
    selected = table[(table["name"] == "private_single/1") & (table["view"] == "current")]
    rows = selected[selected["level"] != "stratum"].set_index("stratum")
    strata = selected[selected["level"] == "stratum"].set_index("stratum")
    assert rows.loc["math500", "acc"] == pytest.approx(0.75)
    assert math.isnan(rows.loc["mmlu_pro", "acc"])
    assert rows.loc["mmlu_pro", "acc_lower"] == pytest.approx(0.25)
    assert rows.loc["mmlu_pro", "acc_upper"] == pytest.approx(0.5)
    assert rows.loc["mmlu_pro", "n_unknown"] == 1
    assert rows.loc["bbh_logic4", "acc"] == pytest.approx(0.5)
    for j, sub in enumerate(BBH_SUBTASKS):
        assert strata.loc[f"bbh_logic4/{sub}", "acc"] == pytest.approx(1.0 if j < 2 else 0.0)
    assert rows.loc["mbpp_plus", "acc"] == pytest.approx(0.0)
    assert rows.loc["mbpp_plus", "n_op_failure"] == 1 and rows.loc["mbpp_plus", "n_wrong"] == 3
    assert math.isnan(rows.loc["macro", "acc"])
    assert rows.loc["macro", "acc_lower"] == pytest.approx((0.75 + 0.25 + 0.5 + 0.0) / 4)
    assert rows.loc["macro", "acc_upper"] == pytest.approx((0.75 + 0.5 + 0.5 + 0.0) / 4)
    assert not analysis.status["complete"]
    assert "INTERIM REPORT: PARTIAL COLLECTION" in (paths.reports_dir / "RUN_REPORT.md").read_text()


def test_bbh_aggregate_weights_subtasks_equally_not_items():
    rows = []
    for k in range(2):
        rows.append({"partition_id": "main", "name": "x", "view": "current", "stratum": f"bbh_logic4/{BBH_SUBTASKS[0]}",
                     "delivered_state": "correct"})
    rows.append({"partition_id": "main", "name": "x", "view": "current", "stratum": f"bbh_logic4/{BBH_SUBTASKS[1]}",
                 "delivered_state": "wrong"})
    rows.append({"partition_id": "main", "name": "x", "view": "current", "stratum": "math500",
                 "delivered_state": "wrong"})
    summary = accuracy_summary(pd.DataFrame(rows))
    summary = summary[summary["level"] != "stratum"].set_index("stratum")
    assert summary.loc["bbh_logic4", "acc"] == pytest.approx(0.5)          # not the pooled 2/3
    assert summary.loc["macro", "acc"] == pytest.approx((0.5 + 0.0) / 2)
    weights = stratum_weights(["math500", *(f"bbh_logic4/{s}" for s in BBH_SUBTASKS), "mmlu_pro", "mbpp_plus"])
    assert weights["math500"] == pytest.approx(0.25)
    assert weights[f"bbh_logic4/{BBH_SUBTASKS[3]}"] == pytest.approx(0.0625)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_audit_passes_on_consistent_run(mutable_run):
    assert cmd_audit(["--run-dir", str(mutable_run.run_dir), "--replicates", str(REPLICATES)]) == 0
    checks = pd.read_csv(mutable_run.tables_dir / "audit_checks.csv").set_index("check")
    assert checks.loc["fixed_population", "status"] == "pass"
    assert checks.loc["sealed_before_join", "status"] == "pass"
    independent = [name for name in checks.index if str(name).startswith("independent")]
    assert independent, "the independent recompute hook must report at least one check row"
    assert all(checks.loc[name, "status"] in {"skip", "pass"} for name in independent)
    assert (mutable_run.reports_dir / "AUDIT.md").is_file()


def test_audit_fails_when_a_checkpoint_drops_an_item(mutable_run):
    path = mutable_run.policies("main")
    lines = path.read_text().splitlines()
    kept = [ln for ln in lines if not (json.loads(ln)["name"] == "peer/4/4" and json.loads(ln)["view"] == "current"
                                       and json.loads(ln)["item_id"] == "math500/w0/i2")]
    assert len(kept) == len(lines) - 1
    path.write_text("\n".join(kept) + "\n")
    assert cmd_audit(["--run-dir", str(mutable_run.run_dir), "--replicates", str(REPLICATES)]) == 1
    checks = pd.read_csv(mutable_run.tables_dir / "audit_checks.csv").set_index("check")
    assert checks.loc["fixed_population", "status"] == "fail"
    assert "peer/4/4" in checks.loc["fixed_population", "violations"]


def test_tampered_selections_are_not_joined_and_fail_audit(mutable_run):
    path = mutable_run.selections("main", 0)
    lines = path.read_text().splitlines()
    first = load_record(SelectionResult, json.loads(lines[0]))
    other = next(json.loads(ln)["selected_candidate_id"] for ln in lines
                 if json.loads(ln)["selected_candidate_id"] not in (None, first.selected_candidate_id))
    lines[0] = canonical_json(dump_record(dataclasses.replace(first, selected_candidate_id=other)))
    path.write_text("\n".join(lines) + "\n")
    run = _load(mutable_run)
    seal = run.seals["main"]
    assert not seal.valid and 0 not in seal.valid_batches
    outcomes = outcome_table(run)
    batch0 = outcomes[outcomes["selection_batch"] == 0]
    assert len(batch0) and batch0["y"].isna().all() and set(batch0["delivered_state"]) == {"unsealed"}
    # batch 1's own seal still verifies, but its chain now points at an intact previous seal: it stays joinable
    assert cmd_audit(["--run-dir", str(mutable_run.run_dir), "--replicates", str(REPLICATES)]) == 1
    checks = pd.read_csv(mutable_run.tables_dir / "audit_checks.csv").set_index("check")
    assert checks.loc["sealed_before_join", "status"] == "fail"


def test_missing_seal_blocks_join_and_resealing_restores_it(mutable_run):
    mutable_run.selection_seal("main", 1).unlink()
    run = _load(mutable_run)
    outcomes = outcome_table(run)
    assert outcomes.loc[outcomes["selection_batch"] == 1, "y"].isna().all()
    assert outcomes.loc[outcomes["selection_batch"] == 0, "y"].notna().all()
    reseal(mutable_run, "main")
    assert outcome_table(_load(mutable_run))["y"].notna().all()


def test_broken_seal_chain_is_not_joined(mutable_run):
    seal_path = mutable_run.selection_seal("main", 1)
    seal = json.loads(seal_path.read_text())
    seal["previous_seal_sha256"] = "0" * 64
    seal_path.write_text(json.dumps(seal))
    run = _load(mutable_run)
    assert run.seals["main"].valid_batches == frozenset({0})
    assert "chain" in "; ".join(run.seals["main"].all_problems)


def test_policy_resealed_in_a_later_batch_is_rejected(mutable_run):
    lines = selection_lines(mutable_run, "main")
    batch1 = mutable_run.selections("main", 1)
    batch1.write_text(batch1.read_text() + canonical_json(lines[0]) + "\n")
    reseal(mutable_run, "main")
    run = _load(mutable_run)
    assert 1 not in run.seals["main"].valid_batches
    assert "already sealed in batch 0" in "; ".join(run.seals["main"].all_problems)


def test_broken_selector_is_missing_not_zero(mutable_run):
    kept = [r for r in selection_lines(mutable_run, "main") if "/peer/4/2/" not in r["policy_id"]]
    write_selection_batches(mutable_run, "main", kept, n_batches=2)
    analysis = analyze(mutable_run.run_dir, replicates=REPLICATES, figures=False)
    rows = analysis.outcomes[analysis.outcomes["name"] == "peer/4/2"]
    assert len(rows) and set(rows["delivered_state"]) == {"missing_selection"}
    assert rows["y"].isna().all()
    acc = analysis.accuracy
    macro = acc[(acc["name"] == "peer/4/2") & (acc["view"] == "current") & (acc["level"] == "macro")].iloc[0]
    assert macro["acc"] is None or math.isnan(macro["acc"])
    assert macro["acc_lower"] == 0.0 and macro["acc_upper"] == 1.0
    assert macro["n_unknown"] == len(rows[rows["view"] == "current"])


def test_unrun_jobs_are_administrative_not_false(mutable_run):
    run = _load(mutable_run)
    item = "math500/w0/i1"
    for node_id, spec in run.jobs["main"].items():
        if spec.item_id == item:
            mutable_run.parsed(node_id).unlink()
            mutable_run.grade(node_id).unlink()
    kept = [r for r in selection_lines(mutable_run, "main") if f"/{item}/" not in r["policy_id"]]
    write_selection_batches(mutable_run, "main", kept, n_batches=2)
    analysis = analyze(mutable_run.run_dir, replicates=REPLICATES, figures=False)
    rows = analysis.outcomes[analysis.outcomes["item_id"] == item]
    assert set(rows["delivered_state"]) == {"not_started"} and rows["y"].isna().all()
    comp = analysis.completeness
    peer = comp[(comp["name"] == "peer/4/4") & (comp["benchmark"] == "math500") & (comp["view"] == "current")].iloc[0]
    assert peer["planned"] == 4 and peer["not_started"] == 1 and peer["graded_known"] == 3
    text = (mutable_run.reports_dir / "COMPLETENESS.md").read_text()
    assert "not_started" in text and "INTERIM" in text
