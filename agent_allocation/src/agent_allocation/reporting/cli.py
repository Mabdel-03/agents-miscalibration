"""``allocation analyze --run-dir DIR``: deterministic tables, figures and reports from stored outputs.

Options beyond the frozen ``--run-dir``: ``--config`` (default: the run's config lock, else the committed
preset, labeled in the report) and ``--replicates`` (override ``statistics.bootstrap_replicates``; tests only).

Config keys consumed: ``statistics.*`` (all six), ``seed``, ``study_id``, ``benchmarks[].report_label``,
``reasoning.main_a``, ``reasoning.high_a``, ``budget.*``, ``planning_prices.*``, ``primary_model``,
``replication_model``. No network, no API key.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from agent_allocation.analysis.completeness import collection_status, completeness_matrix, node_completeness
from agent_allocation.analysis.load import RunData, load_run, resolve_config
from agent_allocation.analysis.outcomes import candidate_table, outcome_table
from agent_allocation.analysis.summary import accuracy_summary, check_statistics_config
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
    completeness = completeness_matrix(outcomes)
    nodes = node_completeness(candidates)
    status = collection_status(outcomes, candidates)
    return Analysis(run, reps, candidates, outcomes, accuracy, completeness, nodes, status)


def write_analysis(analysis: Analysis) -> list[Path]:
    paths = analysis.run.paths
    written: list[Path] = []
    for name, frame in (("candidates", analysis.candidates), ("policy_outcomes", analysis.outcomes),
                        ("accuracy_summary", analysis.accuracy), ("completeness", analysis.completeness),
                        ("node_completeness", analysis.nodes)):
        written.extend(write_table(frame, paths.tables_dir, name))
    for filename, title, text in (
        ("RUN_REPORT", "Run report", report_text.run_report(analysis)),
        ("COMPLETENESS", "Completeness", report_text.completeness_report(analysis)),
    ):
        md = paths.reports_dir / f"{filename}.md"
        write_text(md, text)
        write_text(md.with_suffix(".html"), markdown_to_html(text, f"{title}: {analysis.run.config.get('study_id')}"))
        written.extend([md, md.with_suffix(".html")])
    return written


def analyze(run_dir: str | Path, *, config: Mapping[str, Any] | None = None, config_path: str | None = None,
            replicates: int | None = None) -> Analysis:
    paths = run_paths(run_dir)
    if config is None:
        cfg, source = resolve_config(paths, config_path)
    else:
        cfg, source = dict(config), "explicit mapping"
    run = load_run(paths, cfg, config_source=source)
    analysis = build_analysis(run, replicates=replicates)
    write_analysis(analysis)
    return analysis


def cmd_analyze(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation analyze")
    parser.add_argument("--run-dir", default=os.environ.get("ALLOCATION_RUN_DIR"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.run_dir:
        parser.error("--run-dir is required")
    analysis = analyze(args.run_dir, config_path=args.config, replicates=args.replicates)
    state = "complete" if analysis.status["complete"] else "PARTIAL"
    print(f"analyze: {state}; reports in {analysis.run.paths.reports_dir}", file=sys.stderr)
    return 0
