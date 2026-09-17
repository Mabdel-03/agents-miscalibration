"""Markdown bodies for RUN_REPORT.md and COMPLETENESS.md. Deterministic: no wall-clock values."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from agent_allocation.reporting.render import bullet_list, md_table

if TYPE_CHECKING:  # pragma: no cover
    from agent_allocation.reporting.cli import Analysis

PARTIAL_BANNER = ("**INTERIM REPORT: PARTIAL COLLECTION.** Planned jobs, selections or grades are still missing; "
                  "every number below covers only what is stored and every unknown is shown as unknown.")


def _labels(analysis: "Analysis") -> dict[str, str]:
    return {b.get("id"): b.get("report_label", b.get("id")) for b in analysis.run.config.get("benchmarks", [])}


def header(analysis: "Analysis", title: str) -> str:
    run = analysis.run
    status = analysis.status
    lines = [f"# {title}: `{run.config.get('study_id')}`", ""]
    lines.append(PARTIAL_BANNER if not status["complete"] else "**Collection status: all planned checkpoints known.**")
    lines += ["", f"- Run directory: `{run.paths.run_dir.name}`",
              f"- Input digest (sha256 over every input file): `{run.input_digest()}`",
              f"- Config source: {run.config_source}",
              f"- Bootstrap replicates: {analysis.replicates}", ""]
    return "\n".join(lines)


def scope_notes(analysis: "Analysis") -> str:
    labels = _labels(analysis)
    notes = [
        f"BBH is the **{labels.get('bbh_logic4', 'BBH four-subtask zero-shot subset')}** (four subtasks, zero-shot), "
        "not the official full BBH score; its aggregate weights the four subtasks equally.",
        f"MBPP+ delivered accuracy uses the **public-test selector** ({labels.get('mbpp_plus', 'mbpp_plus')}): "
        "candidates ranked by prompt-visible examples only; reported as public_select@k, never vote@k.",
        "Requested reasoning allowance A is an API setting, not realized reasoning; realized tokens are reported "
        "separately.",
        "This is a token- and API-cost-accounted study, not a FLOP-matched experiment.",
        "Any frontier shown is descriptive: the best among evaluated configurations on this sample, not a "
        "validated optimal policy.",
        "Macro averages weight the four benchmarks equally; per-benchmark results are always shown.",
    ]
    return "## Scope and labels\n\n" + bullet_list(notes)


def run_report(analysis: "Analysis") -> str:
    parts = [header(analysis, "Run report"), scope_notes(analysis)]
    parts.append(unfinished_section(analysis))
    parts.append(seal_section(analysis))
    parts.append(accuracy_section(analysis))
    for key in ("contrasts_md", "costs_md", "replay_md", "dynamics_md", "operations_md", "figures_md"):
        if analysis.extras.get(key):
            parts.append(analysis.extras[key])
    parts.append(problems_section(analysis))
    return "\n".join(parts)


def unfinished_section(analysis: "Analysis") -> str:
    status = analysis.status
    lines = ["## Collection status", "",
             f"- Generation nodes terminal: {status['nodes_terminal']} / {status['nodes_planned']}",
             f"- Checkpoint rows with a known delivered outcome: {status['checkpoint_rows_known']} / "
             f"{status['checkpoint_rows']}", ""]
    if status["unfinished"]:
        lines += ["Unfinished partitions (rows whose delivered outcome is not yet known):", "",
                  md_table(pd.DataFrame(status["unfinished"]))]
    else:
        lines.append("No unfinished partitions.\n")
    return "\n".join(lines)


def seal_section(analysis: "Analysis") -> str:
    rows = [{"partition": p, "selections": s.selections_present, "sealed": s.seal_present,
             "records": s.record_count, "seal_count": s.seal_count, "hash_ok": s.sha256_ok,
             "basis": s.basis, "sealed_at": s.sealed_at, "valid": s.valid}
            for p, s in sorted(analysis.run.seals.items())]
    return ("## Selection seals\n\nPrivate grades are joined onto selections only for partitions whose seal "
            "verifies.\n\n" + md_table(pd.DataFrame(rows)))


def accuracy_section(analysis: "Analysis") -> str:
    acc = analysis.accuracy
    if acc.empty:
        return "## Delivered accuracy\n\n_(no planned checkpoints)_\n"
    view = acc[(acc["view"] == "current") & (acc["level"] != "stratum")]
    cols = ["partition_id", "name", "benchmark", "n_planned", "n_correct", "n_wrong", "n_op_failure",
            "n_unknown", "n_not_run", "acc", "acc_lower", "acc_upper"]
    extra = [c for c in ("ci_lo", "ci_hi", "ci_basis") if c in view]
    return ("## Delivered accuracy (current view)\n\nFixed denominators: `acc_lower` counts unknowns as wrong, "
            "`acc_upper` as correct; `acc` is shown only when nothing is unknown. Stratum-level rows (BBH "
            "subtasks) are in `tables/accuracy_summary`.\n\n" + md_table(view, cols + extra))


def problems_section(analysis: "Analysis") -> str:
    problems = list(analysis.run.problems) + list(analysis.extras.get("problems", []))
    return "## Load and consistency problems\n\n" + bullet_list(sorted(set(problems)))


def completeness_report(analysis: "Analysis") -> str:
    parts = [header(analysis, "Completeness")]
    parts.append("Every planned partition x benchmark x condition appears below. `not_started` and `in_progress` "
                 "are administrative (never admitted or unfinished) and are not model errors; `op_failure` is a "
                 "known operational failure; the `unknown_*`, `pending_grade`, `missing_selection` and `unsealed` "
                 "columns are unknown outcomes, never zeros.\n")
    parts.append("## Checkpoints\n\n" + md_table(analysis.completeness))
    parts.append("## Generation nodes and candidate grades\n\n" + md_table(analysis.nodes))
    return "\n".join(parts)
