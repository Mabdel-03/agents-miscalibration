"""M2 costs, replay, reconciliation, audit invariants, figures and reports."""
from __future__ import annotations

import json
import socket
from decimal import Decimal

import pandas as pd
import pytest

from agent_allocation.analysis import checks as C
from agent_allocation.analysis.costs import PriceUnitError, planning_prices, price_unit_checks, standardized_cost
from agent_allocation.contracts import (
    EndpointLock, GenerationKind, LedgerEvent, LedgerEventKind, SelectionResult, canonical_json, dump_record,
    load_record,
)
from agent_allocation.config import load_config
from agent_allocation.paths import PROJECT_ROOT
from agent_allocation.reporting.audit import cmd_audit, hook_checks
from agent_allocation.reporting.cli import analyze
from tests.unit.analysis.conftest import REPLICATES, copy_run
from tests.unit.analysis.synthrun import Grid, SynthSpec, build_run, reseal, selection_lines, write_selection_batches

CFG = load_config(PROJECT_ROOT / "configs" / "balanced150.json")


@pytest.fixture(scope="module")
def analysed(built_run, tmp_path_factory):
    paths = copy_run(built_run, tmp_path_factory.mktemp("analysed"))
    return paths, analyze(paths.run_dir, replicates=REPLICATES)


# ---------------------------------------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------------------------------------

def test_policy_costs_charge_own_ancestors_and_separate_retries(analysed):
    paths, analysis = analysed
    costs = analysis.extras["policy_costs"].set_index("policy_id")
    run = analysis.run
    specs = {nid: s for part in run.jobs.values() for nid, s in part.items()}
    cp = next(c for c in run.policies["main"] if c.name == "peer/4/2" and c.view.value == "current")
    row = costs.loc[cp.policy_id]
    assert row["nominal_generations"] == 12 == len(cp.required_node_ids)
    expected_success = sum(run.parsed[n][0].usage.actual_cost_usd for n in cp.required_node_ids)
    assert Decimal(str(row["success_cost_usd"])) == expected_success
    accepted = {run.parsed[n][0].attempt_id for n in cp.required_node_ids}
    expected_retry = sum((e.amount_usd for e in run.ledger if e.node_id in cp.required_node_ids
                          and e.kind == LedgerEventKind.SETTLE and e.attempt_id not in accepted), Decimal(0))
    assert Decimal(str(row["retry_cost_usd"])) == expected_retry
    assert Decimal(str(row["total_charged_usd"])) == expected_success + expected_retry
    assert row["requested_reasoning_tokens"] == sum(specs[n].reasoning_budget for n in cp.required_node_ids)
    # never the collection total
    collection = sum((e.amount_usd for e in run.ledger if e.kind == LedgerEventKind.SETTLE), Decimal(0))
    assert Decimal(str(row["total_charged_usd"])) < collection
    assert "agent 03 policy_cost_rows" in row["cost_source"] or "local" in row["cost_source"]


def test_price_units_are_verified():
    prices = planning_prices(CFG)
    lite = prices["google/gemini-2.5-flash-lite"]
    assert standardized_cost(1_000_000, 0, lite) == Decimal("0.10")
    assert standardized_cost(0, 1_000_000, lite) == Decimal("0.40")
    assert standardized_cost(None, 5, lite) is None
    bad = json.loads(json.dumps(CFG))
    bad["planning_prices"]["primary"]["input_price"] = "0.0000001"          # a per-token value in a per-M slot
    with pytest.raises(PriceUnitError):
        planning_prices(bad)
    lock = EndpointLock("google/gemini-2.5-flash-lite", "google-ai-studio", "Google AI Studio", "standard",
                        "0.0000001", "0.0000004", "0.0000004", 1048576, 65535, ("reasoning",))
    ok = price_unit_checks(CFG, {lock.model: lock}, pd.DataFrame())
    assert all(r["passed"] for r in ok)
    wrong = EndpointLock("google/gemini-2.5-flash-lite", "google-ai-studio", "Google AI Studio", "standard",
                         "0.10", "0.40", "0.40", 1048576, 65535, ("reasoning",))
    failed = price_unit_checks(CFG, {wrong.model: wrong}, pd.DataFrame())
    assert not all(r["passed"] for r in failed)


def test_cumulative_costs_monotone_while_accuracy_may_decrease(tmp_path):
    def correct_fn(index, benchmark, spec):
        if spec.kind == GenerationKind.PEER:
            return spec.update_round != 2                       # accuracy dips at r=2 and recovers
        return index % 2 == 0

    paths = build_run(tmp_path, SynthSpec(waves=1, extension_waves=0, correct_fn=correct_fn,
                                          grid=Grid(team_sizes=(4,), deep_multipliers=())))
    analysis = analyze(paths.run_dir, replicates=REPLICATES, figures=False)
    acc = analysis.accuracy
    macro = acc[(acc["level"] == "macro") & (acc["view"] == "current")].set_index("name")["acc"]
    assert macro["peer/4/2"] < macro["peer/4/1"] and macro["peer/4/3"] > macro["peer/4/2"]
    results = {r.name: r for r in C.run_checks(analysis.run, analysis.outcomes, analysis=analysis)}
    assert results["cumulative_costs_monotone"].status == "pass"
    assert all(r.status != "fail" for r in results.values())
    costs = analysis.extras["policy_costs"].copy()
    target = costs[(costs["name"] == "peer/4/3") & (costs["view"] == "current")].index[0]
    costs.loc[target, "nominal_generations"] = 3
    assert C.check_cumulative_costs(costs).status == "fail"


# ---------------------------------------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------------------------------------

def test_budget_replay_is_causal_and_never_drops_items(analysed):
    from agent_allocation.analysis.replay import replay

    paths, analysis = analysed
    items, curves, notes = replay(analysis.run, analysis.outcomes, budgets={"main": [1e-9, 10.0]})
    tiny = items[items["budget_usd"] == 1e-9]
    assert set(tiny["status"]) == {"abstain_initialization_does_not_fit"} and (tiny["y"] == 0).all()
    huge = items[items["budget_usd"] == 10.0]
    assert set(huge["status"]) == {"admitted"}
    n_items = analysis.outcomes["item_id"].nunique()
    for label, group in items.groupby(["trajectory", "budget_usd"]):
        assert group["item_id"].nunique() == n_items             # every planned item stays in every curve
    peer = huge[huge["trajectory"] == "peer/4"].set_index("item_id")
    final = analysis.outcomes[(analysis.outcomes["name"] == "peer/4/4")
                              & (analysis.outcomes["view"] == "current")].set_index("item_id")["y"]
    assert (peer["y"] == final.reindex(peer.index)).all()
    assert set(items["reservation_basis"]) == {"rendered_chars"}
    assert set(curves["label"]) == {"budgeted policy replay"}


def test_replay_falls_back_then_labels_descriptive(built_run, tmp_path):
    from agent_allocation.analysis.load import load_run, resolve_config
    from agent_allocation.analysis.outcomes import outcome_table
    from agent_allocation.analysis.replay import replay

    paths = copy_run(built_run, tmp_path)
    for raw in (paths.run_dir / "raw").rglob("*.json.gz"):
        raw.unlink()
    cfg, source = resolve_config(paths)
    run = load_run(paths, cfg, config_source=source)
    items, curves, notes = replay(run, outcome_table(run), budgets={"main": [10.0]})
    assert set(items["reservation_basis"]) == {"prompt_tokens"}
    assert set(curves["label"]) == {"budgeted policy replay"}
    for nid in list(run.parsed):
        result, cand = run.parsed[nid]
        run.parsed[nid] = (result.__class__(**{**result.__dict__, "usage": None}), cand)
    items, curves, notes = replay(run, outcome_table(run), budgets={"main": [10.0]})
    assert set(curves["label"]) == {"descriptive (replay unavailable)"} and notes


# ---------------------------------------------------------------------------------------------------------
# Reconciliation and audit
# ---------------------------------------------------------------------------------------------------------

def test_cost_reconciliation_detects_discrepancy_and_unresolved(built_run, tmp_path):
    paths = copy_run(built_run, tmp_path)
    analysis = analyze(paths.run_dir, replicates=REPLICATES, figures=False)
    assert analysis.extras["reconciliation"]["reconciled"] and analysis.final
    assert "Status: RECONCILED" in (paths.reports_dir / "COST_RECONCILIATION.md").read_text()
    lines = paths.provider_usage.read_text().splitlines()
    last = json.loads(lines[-1])
    last["data"]["usage"] += 1.0                                  # unexplained provider-side spend
    lines[-1] = json.dumps(last)
    events = [json.loads(x) for x in paths.ledger_journal.read_text().splitlines()]
    seq = max(e["seq"] for e in events)
    events += [dump_record(LedgerEvent(seq + 1, LedgerEventKind.RESERVE, Decimal("0.01"), "2026-09-17T08:30:00Z",
                                       "zz:1", "zz", "main")),
               dump_record(LedgerEvent(seq + 2, LedgerEventKind.SENT, Decimal("0"), "2026-09-17T08:30:01Z", "zz:1",
                                       "zz", "main"))]
    paths.ledger_journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    paths.provider_usage.write_text("\n".join(lines) + "\n")
    analysis = analyze(paths.run_dir, replicates=REPLICATES, figures=False)
    rec = analysis.extras["reconciliation"]
    assert rec["provider"]["status"] == "discrepancy" and rec["ledger"]["unresolved_usd"] == Decimal("0.01")
    assert not analysis.final
    assert "NOT RECONCILED" in (paths.reports_dir / "COST_RECONCILIATION.md").read_text()
    assert "COSTS NOT RECONCILED" in (paths.reports_dir / "RUN_REPORT.md").read_text()


def test_audit_algebraic_checks_pass_on_consistent_run(analysed):
    paths, analysis = analysed
    assert cmd_audit(["--run-dir", str(paths.run_dir), "--replicates", str(REPLICATES)]) == 0
    checks = pd.read_csv(paths.tables_dir / "audit_checks.csv").set_index("check")["status"].to_dict()
    for name in ("fixed_population", "sealed_before_join", "selector_legality", "unknown_outcomes_not_coerced",
                 "delivered_within_coverage_candidate_only", "archive_coverage_monotone",
                 "pass_at_1_equals_mean_draw_correctness", "pass_at_bank_size_equals_full_bank_coverage",
                 "transition_identity", "policy_ancestor_counts", "cumulative_costs_monotone", "price_units",
                 "ledger_invariants"):
        assert checks[name] == "pass", name


def test_candidate_only_invariant_scope_exempts_synthesis():
    base = {"partition_id": "main", "view": "current", "delivered_state": "correct", "y": 1.0,
            "selected_in_pool": True}
    pooled = {**base, "policy_id": "main/x/pooled/4/current", "family": "pooled", "coverage": 1.0,
              "input_coverage": 0.0}
    legit = {**base, "policy_id": "main/x/peer/4/1/current", "family": "peer", "coverage": 1.0,
             "input_coverage": float("nan")}
    ok = C.check_delivered_within_coverage(pd.DataFrame([pooled, legit]))
    assert ok.status == "pass" and "1 POOLED rows exceed" in ok.detail
    assert C.check_delivered_within_coverage(pd.DataFrame([pooled])).status == "skip"   # nothing candidate-only
    peer = {**base, "policy_id": "main/x/peer/4/4/current", "family": "peer", "coverage": 0.0,
            "input_coverage": float("nan")}
    assert C.check_delivered_within_coverage(pd.DataFrame([pooled, peer])).status == "fail"
    outside = {**peer, "coverage": 1.0, "selected_in_pool": False}
    assert C.check_delivered_within_coverage(pd.DataFrame([outside])).status == "fail"


def test_selector_legality_catches_wrong_concrete_selector(built_run, tmp_path):
    paths = copy_run(built_run, tmp_path)
    records = selection_lines(paths, "main")
    for rec in records:
        if "/peer/4/3/current" in rec["policy_id"]:
            rec["selector_name"] = "last_answer"
            break
    write_selection_batches(paths, "main", records, n_batches=2)
    assert cmd_audit(["--run-dir", str(paths.run_dir), "--replicates", str(REPLICATES)]) == 1
    checks = pd.read_csv(paths.tables_dir / "audit_checks.csv").set_index("check")
    assert checks.loc["selector_legality", "status"] == "fail"


def test_independent_recompute_hook_contract(tmp_path, analysed):
    paths, analysis = analysed
    own = {"primary_i_peer_vs_private_team_n4_r4": 0.125, "primary_ii_cp_minus_private_team_gain_h3": None}
    agree = hook_checks(lambda run_dir: {"contrasts": dict(own), "checks": [{"name": "x", "passed": True}]},
                        paths.run_dir, own)
    assert [r.status for r in agree] == ["pass", "pass"]
    differ = hook_checks(lambda run_dir: {"contrasts": {**own, "primary_i_peer_vs_private_team_n4_r4": 0.2}},
                         paths.run_dir, own)
    assert differ[0].status == "fail"

    def boom(run_dir):
        raise RuntimeError("independent code failed")
    assert hook_checks(boom, paths.run_dir, own)[0].status == "fail"
    hook = tmp_path / "hook.py"
    hook.write_text("def independent_recompute(run_dir):\n    return {'contrasts': {}, 'checks': []}\n")
    assert cmd_audit(["--run-dir", str(paths.run_dir), "--replicates", str(REPLICATES),
                      "--recompute-hook", str(hook)]) == 0


# ---------------------------------------------------------------------------------------------------------
# Figures and reports
# ---------------------------------------------------------------------------------------------------------

def test_figures_need_no_api_key_or_network(built_run, tmp_path, monkeypatch):
    paths = copy_run(built_run, tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("figure generation attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    records = [r for r in selection_lines(paths, "main") if "/peer/2/3/" not in r["policy_id"]]
    write_selection_batches(paths, "main", records, n_batches=2)       # a broken selector: missing, not zero
    analysis = analyze(paths.run_dir, replicates=REPLICATES, figures=True)
    figures = analysis.extras["figures"]
    assert figures and all(p.is_file() and p.stat().st_size > 1000 for p in figures)
    frontier = analysis.extras["frontier"]
    assert "peer/2/3" not in set(frontier["name"])
    summary = analysis.extras["cost_summary"]
    assert summary.loc[summary["name"] == "peer/2/3", "acc"].isna().all()
    report = (paths.reports_dir / "RUN_REPORT.md").read_text()
    assert "figures/main_accuracy_vs_total_charged_usd.png" in report
    assert "PARTIAL COLLECTION" in report


def test_run_report_states_required_labels(analysed):
    paths, analysis = analysed
    text = (paths.reports_dir / "RUN_REPORT.md").read_text()
    for needle in ("BBH four-subtask zero-shot subset", "public-test selector", "public_select@k", "vote@k",
                   "Oracle pass@k", "requested", "realized", "not a FLOP-matched", "descriptive",
                   "Budgeted policy replay", "Billed retries", "Stop reasons", "share_finish_length",
                   "share_response_cache_hit", "median_latency_seconds", "Independent-item counts",
                   "Small-panel warnings", "Primary contrasts", "CP decomposition", "Coverage and selection gap",
                   "archive coverage"):
        assert needle in text, needle
    assert (paths.reports_dir / "RUN_REPORT.html").read_text().startswith("<!doctype html>")
    for table in ("policy_costs", "cost_summary", "frontier_descriptive", "pass_at_k_vs_readout",
                  "budget_replay_curves", "cost_reconciliation", "operations", "primary_contrasts"):
        assert (paths.tables_dir / f"{table}.parquet").is_file(), table
        assert (paths.tables_dir / f"{table}.csv").is_file(), table


def test_pass_at_k_is_separate_from_readouts(analysed):
    paths, analysis = analysed
    bank = analysis.extras["item_metrics"]["sample_bank"]
    code = bank[bank["benchmark"] == "mbpp_plus"]
    assert set(code["readout_label"]) == {"public_select@k"}
    assert set(bank[bank["benchmark"] != "mbpp_plus"]["readout_label"]) == {"vote@k"}
    known = bank[bank["oracle_pass_at_k"].notna() & (bank["k"] == 1)]
    assert ((known["oracle_pass_at_k"] - known["mean_draw_correct"]).abs() < 1e-12).all()
