"""``allocation analyze --run-dir DIR``: deterministic tables, figures and reports from stored outputs.

Options beyond the frozen ``--run-dir``: ``--config`` (default: the run's config lock, else the committed
preset, labeled in the report), ``--replicates`` (override ``statistics.bootstrap_replicates``; tests only) and
``--no-figures``.

Outputs: ``reports/RUN_REPORT.{md,html}``, ``reports/COST_RECONCILIATION.{md,html}``,
``reports/COMPLETENESS.{md,html}``, Parquet + CSV tables under ``tables/``, PNG figures under ``figures/``.

Config keys consumed: ``statistics.*`` (all six), ``seed``, ``study_id``, ``benchmarks[].report_label``,
``budget.reservation_multiplier``, ``budget.credit_ceiling_usd``, ``planning_prices.*``, ``primary_model``,
``replication_model``, ``reasoning.main_a``, ``reasoning.high_a``. No network, no API key.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from agent_allocation.analysis import contrasts as K
from agent_allocation.analysis.completeness import collection_status, completeness_matrix, node_completeness
from agent_allocation.analysis.costs import (
    COST_MEASURES, attempt_charges, cost_summary, descriptive_frontier, load_endpoint_locks, policy_cost_table,
    price_unit_checks,
)
from agent_allocation.analysis.itemmetrics import load_item_metrics
from agent_allocation.analysis.load import RunData, load_run, resolve_config
from agent_allocation.analysis.names import name_sort_key
from agent_allocation.analysis.operations import generation_operations, retry_table, stop_reasons
from agent_allocation.analysis.outcomes import candidate_table, outcome_table
from agent_allocation.analysis.reconcile import reconciliation, reconciliation_frame
from agent_allocation.analysis.replay import replay
from agent_allocation.analysis.stats import build_population, column_intervals, value_matrix
from agent_allocation.analysis.summary import accuracy_summary, check_statistics_config, weighted_means
from agent_allocation.paths import RunPaths
from agent_allocation.reporting import report_text
from agent_allocation.reporting.render import markdown_to_html, write_table, write_text


@dataclass
class Analysis:
    run: RunData
    replicates: int
    candidates: pd.DataFrame
    outcomes: pd.DataFrame
    accuracy: pd.DataFrame
    completeness: pd.DataFrame
    nodes: pd.DataFrame
    status: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def final(self) -> bool:
        rec = self.extras.get("reconciliation")
        return bool(self.status["complete"] and rec and rec["reconciled"])


def run_paths(run_dir: str | Path) -> RunPaths:
    try:
        return RunPaths.for_run(run_dir)
    except RuntimeError:
        # Analysis never touches node-local SQLite state; any path is acceptable here.
        chosen = Path(run_dir).resolve()
        return RunPaths.for_run(chosen, local_state_dir=chosen / "state")


def build_analysis(run: RunData, *, replicates: int | None = None) -> Analysis:
    stats_cfg = check_statistics_config(run.config)
    reps = int(replicates if replicates is not None else stats_cfg["bootstrap_replicates"])
    candidates = candidate_table(run)
    outcomes = outcome_table(run)
    accuracy = accuracy_summary(outcomes)
    analysis = Analysis(run, reps, candidates, outcomes, accuracy, completeness_matrix(outcomes),
                        node_completeness(candidates), collection_status(outcomes, candidates))
    seed = run.config.get("seed")
    notes: list[str] = []
    warnings: list[str] = []
    problems: list[str] = []

    analysis.accuracy, populations, ci_warnings, ci_mismatch = attach_accuracy_intervals(
        outcomes, accuracy, replicates=reps, study_seed=seed)
    warnings += ci_warnings
    problems += ci_mismatch

    primary, estimates, primary_notes = K.primary_analysis(outcomes, replicates=reps, study_seed=seed,
                                                           item_panels=run.item_panels())
    high_low, hl_notes = K.high_low_shared(outcomes, replicates=reps, study_seed=seed)
    for note in primary_notes + hl_notes:
        (warnings if ("small panel" in note or "stratum" in note or "absent" in note) else notes).append(note)

    costs, cost_problems = policy_cost_table(run, candidates, outcomes)
    problems += cost_problems
    summary = cost_summary(costs, analysis.accuracy)
    frontier = pd.concat([descriptive_frontier(summary, m) for m in COST_MEASURES], ignore_index=True) \
        if not summary.empty else pd.DataFrame()
    price_checks = price_unit_checks(run.config, load_endpoint_locks(run), candidates)

    metrics = load_item_metrics(run, outcomes)
    bank = metrics["sample_bank"]
    passk = weighted_means(bank, ["partition_id", "k"], ["oracle_pass_at_k", "prefix_coverage", "readout",
                                                          "selection_gap_same_pool"]) if not bank.empty else pd.DataFrame()
    trans = metrics["transitions"]
    trans_summary = _transition_summary(trans)

    replay_items, replay_curves, replay_notes = replay(run, outcomes)
    notes += replay_notes
    rec = reconciliation(run)
    charges, attempts_by_node, _ = attempt_charges(run)

    analysis.extras.update({
        "populations": populations, "primary": primary, "primary_estimates": estimates,
        "primary_planned": _primary_planned(outcomes, run), "high_low": high_low,
        "policy_costs": costs, "cost_summary": summary, "frontier": frontier, "price_checks": price_checks,
        "item_metrics": metrics, "passk_summary": passk, "transition_summary": trans_summary,
        "replay_items": replay_items, "replay_curves": replay_curves, "reconciliation": rec,
        "operations": generation_operations(candidates), "retries": retry_table(run, charges, attempts_by_node),
        "stop_reasons": stop_reasons(run),
        "notes": sorted(set(notes)), "warnings": _compact_warnings(warnings), "problems": sorted(set(problems)),
    })
    return analysis


def _primary_planned(outcomes: pd.DataFrame, run: RunData) -> dict[str, int]:
    """Planned population sizes from the manifests alone (never from outcomes)."""
    current = outcomes[(outcomes["view"] == "current") & (outcomes["partition_id"] == K.MAIN_PARTITION)]
    planned: dict[str, int] = {}
    i_items = set(current.loc[current["name"] == "peer/4/4", "item_id"]) & set(
        current.loc[current["name"] == "private_team/4/4", "item_id"])
    if i_items:
        planned[K.PRIMARY_I] = len(i_items)
    ext = K.extension_items(outcomes, K.MAIN_PARTITION, run.item_panels())
    cp3 = set(current.loc[current["name"] == "cp/3", "item_id"])
    if cp3:
        planned[K.PRIMARY_II] = len(cp3 & ext) if ext is not None else len(cp3)
    return planned


def _transition_summary(trans: pd.DataFrame) -> pd.DataFrame:
    if trans.empty:
        return pd.DataFrame()
    known = trans[trans["known"].astype(bool)]
    if known.empty:
        return pd.DataFrame()
    cols = ["W_to_C", "C_to_W", "I_to_C", "C_to_I", "correct_prev", "correct_next"]
    frame = known.assign(**{c: known[c].astype(float) for c in cols},
                         last_correct_lost=known["last_correct_lost"].astype(float),
                         recovered=known["recovered"].astype(float))
    return weighted_means(frame, ["partition_id", "trajectory", "from_step", "to_step"],
                          cols + ["last_correct_lost", "recovered"])


def _compact_warnings(messages: list[str]) -> list[str]:
    """Group identical small-panel messages across populations: ``message (populations: a, b)``."""
    grouped: dict[str, set[str]] = {}
    for msg in messages:
        label, _, text = msg.partition(": ")
        grouped.setdefault(text or msg, set()).add(label if text else "")
    out = []
    for text, labels in sorted(grouped.items()):
        labels.discard("")
        out.append(f"{text} (populations: {len(labels)})" if labels else text)
    return out


def attach_accuracy_intervals(outcomes: pd.DataFrame, accuracy: pd.DataFrame, *, replicates: int, study_seed: Any
                              ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Item-clustered percentile intervals for every current-view checkpoint, one population per partition
    (base families) and one for its extension families; all columns of a population share resample indices."""
    warnings: list[str] = []
    mismatches: list[str] = []
    frames = []
    pop_rows = []
    current = outcomes[outcomes["view"] == "current"]
    for partition, group in current.groupby("partition_id", sort=True):
        names = sorted(set(group["name"]), key=name_sort_key)
        ext = [n for n in names if n.split("/", 1)[0] in ("cp", "pooled")]
        base = [n for n in names if n not in ext]
        for label, subset in ((f"{partition}:accuracy", base), (f"{partition}:extension_accuracy", ext)):
            if not subset:
                continue
            columns = [(str(partition), n) for n in subset]
            population = build_population(outcomes, columns, label=label)
            warnings.extend(population.warnings())
            mismatches.extend(population.mismatches)
            pop_rows.append({"population": label, "partition_id": partition, "checkpoints": len(subset),
                             "independent_items": population.n_items,
                             **{f"items_{b}": n for b, n in population.n_by_benchmark.items()},
                             "strata": ";".join(f"{s}={n}" for s, n in population.n_by_stratum.items())})
            if not population.n_items:
                continue
            matrix = value_matrix(outcomes, population, columns)
            ci = column_intervals(matrix, population.resampler(replicates, study_seed), subset)
            ci["partition_id"] = partition
            ci["view"] = "current"
            frames.append(ci)
    populations = pd.DataFrame(pop_rows)
    if not frames or accuracy.empty:
        return accuracy, populations, warnings, mismatches
    cis = pd.concat(frames, ignore_index=True).rename(columns={"n_items": "ci_n_items", "n_known": "ci_n_known",
                                                              "estimate": "ci_estimate"})
    merged = accuracy.merge(cis, how="left", on=["partition_id", "name", "view", "level", "benchmark"])
    return merged, populations, warnings, mismatches


TABLES = (
    ("populations", "populations"), ("primary", "primary_contrasts"), ("high_low", "high_vs_main_shared_items"),
    ("policy_costs", "policy_costs"), ("cost_summary", "cost_summary"), ("frontier", "frontier_descriptive"),
    ("passk_summary", "pass_at_k_vs_readout"), ("transition_summary", "transitions_summary"),
    ("replay_items", "budget_replay_items"), ("replay_curves", "budget_replay_curves"),
    ("operations", "operations"), ("retries", "billed_retries"),
)


def write_analysis(analysis: Analysis, *, figures: bool = True) -> list[Path]:
    paths = analysis.run.paths
    written: list[Path] = []
    tables = [("candidates", analysis.candidates), ("policy_outcomes", analysis.outcomes),
              ("accuracy_summary", _sorted_accuracy(analysis.accuracy)), ("completeness", analysis.completeness),
              ("node_completeness", analysis.nodes)]
    for key, name in TABLES:
        frame = analysis.extras.get(key)
        if isinstance(frame, pd.DataFrame):
            tables.append((name, frame))
    metrics = analysis.extras.get("item_metrics", {})
    for key, name in (("sample_bank", "sample_bank_items"), ("transitions", "transitions_items")):
        if isinstance(metrics.get(key), pd.DataFrame):
            tables.append((name, metrics[key]))
    rec = analysis.extras.get("reconciliation")
    if rec is not None:
        tables.append(("cost_reconciliation", reconciliation_frame(rec)))
    tables.append(("price_unit_checks", pd.DataFrame(analysis.extras.get("price_checks", []))))
    for name, frame in tables:
        written.extend(write_table(frame, paths.tables_dir, name))
    if figures:
        from agent_allocation.analysis.figures import make_figures
        analysis.extras["figures"] = make_figures(analysis)
        written.extend(analysis.extras["figures"])
    for filename, title, text in (
        ("RUN_REPORT", "Run report", report_text.run_report(analysis)),
        ("COST_RECONCILIATION", "Cost reconciliation", report_text.cost_reconciliation_report(analysis)),
        ("COMPLETENESS", "Completeness", report_text.completeness_report(analysis)),
    ):
        md = paths.reports_dir / f"{filename}.md"
        write_text(md, text)
        write_text(md.with_suffix(".html"), markdown_to_html(text, f"{title}: {analysis.run.config.get('study_id')}"))
        written.extend([md, md.with_suffix(".html")])
    return written


def _sorted_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    order = sorted(range(len(frame)), key=lambda i: (
        str(frame.iloc[i]["partition_id"]), name_sort_key(frame.iloc[i]["name"]), str(frame.iloc[i]["view"])))
    return frame.iloc[order].reset_index(drop=True)


def analyze(run_dir: str | Path, *, config: Mapping[str, Any] | None = None, config_path: str | None = None,
            replicates: int | None = None, figures: bool = True) -> Analysis:
    paths = run_paths(run_dir)
    if config is None:
        cfg, source = resolve_config(paths, config_path)
    else:
        cfg, source = dict(config), "explicit mapping"
    run = load_run(paths, cfg, config_source=source)
    analysis = build_analysis(run, replicates=replicates)
    write_analysis(analysis, figures=figures)
    return analysis


def cmd_analyze(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation analyze")
    parser.add_argument("--run-dir", default=os.environ.get("ALLOCATION_RUN_DIR"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_dir:
        parser.error("--run-dir is required")
    analysis = analyze(args.run_dir, config_path=args.config, replicates=args.replicates,
                       figures=not args.no_figures)
    state = "final" if analysis.final else "INTERIM (partial or unreconciled)"
    print(f"analyze: {state}; reports in {analysis.run.paths.reports_dir}", file=sys.stderr)
    return 0
