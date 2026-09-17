"""Markdown bodies for RUN_REPORT.md, COST_RECONCILIATION.md and COMPLETENESS.md. Deterministic: every value
comes from stored inputs (no wall-clock time)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd

from agent_allocation.analysis.contrasts import PRIMARY_I, PRIMARY_II
from agent_allocation.analysis.names import name_sort_key
from agent_allocation.contracts import BENCHMARKS
from agent_allocation.reporting.render import bullet_list, fmt, md_table

if TYPE_CHECKING:  # pragma: no cover
    from agent_allocation.reporting.cli import Analysis

LEVELS = ("macro",) + BENCHMARKS


def _labels(analysis: "Analysis") -> dict[str, str]:
    return {b.get("id"): b.get("report_label", b.get("id")) for b in analysis.run.config.get("benchmarks", [])}


def interim_reasons(analysis: "Analysis") -> list[str]:
    reasons = []
    status = analysis.status
    if status["nodes_terminal"] < status["nodes_planned"]:
        reasons.append(f"generation incomplete ({status['nodes_terminal']} of {status['nodes_planned']} planned "
                       f"nodes terminal)")
    if status["checkpoint_rows_known"] < status["checkpoint_rows"]:
        reasons.append(f"{status['checkpoint_rows'] - status['checkpoint_rows_known']} checkpoint rows lack a known "
                       f"delivered outcome (unrun, unselected, unsealed or ungraded)")
    rec = analysis.extras.get("reconciliation")
    if rec is not None and not rec["reconciled"]:
        reasons.append("costs are not reconciled against provider usage (see COST_RECONCILIATION.md)")
    return reasons


def header(analysis: "Analysis", title: str) -> str:
    run = analysis.run
    lines = [f"# {title}: `{run.config.get('study_id')}`", ""]
    reasons = interim_reasons(analysis)
    if reasons:
        kind = "PARTIAL COLLECTION" if not analysis.status["complete"] else "COSTS NOT RECONCILED"
        lines += [f"**INTERIM REPORT: {kind}.** This report is partial and not a final result:", "",
                  bullet_list(reasons)]
    else:
        lines += ["**Final: every planned checkpoint is known and costs are reconciled.**", ""]
    lines += [f"- Run directory: `{run.paths.run_dir.name}`",
              f"- Input digest (sha256 over every input file): `{run.input_digest()}`",
              f"- Config source: {run.config_source}",
              f"- Bootstrap replicates: {analysis.replicates} (item-clustered within benchmark strata)", ""]
    return "\n".join(lines)


def scope_notes(analysis: "Analysis") -> str:
    labels = _labels(analysis)
    cfg = analysis.run.config
    reasoning = cfg.get("reasoning", {})
    notes = [
        f"BBH is the **{labels.get('bbh_logic4', 'BBH four-subtask zero-shot subset')}** (four subtasks, zero-shot), "
        "not the official full BBH score; its aggregate weights the four subtasks equally.",
        f"MBPP+ delivered accuracy uses the **public-test selector** ({labels.get('mbpp_plus', 'mbpp_plus')}): "
        "candidates are ranked by prompt-visible examples only and reported as public_select@k, never vote@k.",
        f"Requested reasoning allowances (A={reasoning.get('main_a')} main, A={reasoning.get('high_a')} high) are "
        "soft API settings, not realized reasoning; realized reasoning and completion tokens are reported beside "
        "them (Operations).",
        "This is a token- and API-cost-accounted study, not a FLOP-matched experiment.",
        "Frontiers are descriptive: the best among evaluated configurations on this sample, never a validated "
        "optimal policy. Everything except the two primary contrasts is exploratory or descriptive.",
        "Oracle pass@k (an unavailable selector) is reported separately from achievable vote@k / public_select@k.",
        "Macro averages weight the four benchmarks equally; per-benchmark results are always shown.",
        "Panel variants (high allowance, second-model replication) are compared with main on shared item ids only "
        "(never the 96-item panel against all main items).",
    ]
    return "## Scope and labels\n\n" + bullet_list(notes)


def run_report(analysis: "Analysis") -> str:
    parts = [header(analysis, "Run report"), scope_notes(analysis), unfinished_section(analysis),
             seal_section(analysis), primary_section(analysis), accuracy_section(analysis),
             coverage_section(analysis), passk_section(analysis), cost_section(analysis), high_low_section(analysis), replay_section(analysis),
             dynamics_section(analysis), operations_section(analysis), figures_section(analysis),
             warnings_section(analysis)]
    return "\n".join(p for p in parts if p)


def unfinished_section(analysis: "Analysis") -> str:
    status = analysis.status
    lines = ["## Collection status", "",
             f"- Generation nodes terminal: {status['nodes_terminal']} / {status['nodes_planned']}",
             f"- Checkpoint rows with a known delivered outcome: {status['checkpoint_rows_known']} / "
             f"{status['checkpoint_rows']}", ""]
    if status["unfinished"]:
        lines += ["Unfinished partitions (rows whose delivered outcome is not yet known; see COMPLETENESS.md):", "",
                  md_table(pd.DataFrame(status["unfinished"]))]
    else:
        lines.append("No unfinished partitions.\n")
    pops = analysis.extras.get("populations")
    if isinstance(pops, pd.DataFrame) and not pops.empty:
        lines += ["Independent-item counts per analysis population (the bootstrap cluster is the source item):", "",
                  md_table(pops)]
    return "\n".join(lines)


def seal_section(analysis: "Analysis") -> str:
    rows = [{"partition": p, "batch": b.batch, "records": b.record_count, "seal_count": b.seal_count,
             "hash_ok": b.sha256_ok, "chain": b.chain_basis, "sealed_at": b.sealed_at, "valid": b.valid,
             "problems": "; ".join(b.problems)}
            for p, s in sorted(analysis.run.seals.items()) for b in s.batches]
    missing = [p for p, s in sorted(analysis.run.seals.items()) if not s.batches]
    text = ("## Selection seals\n\nPrivate grades are joined onto a selection only when its append-only batch seal "
            "verifies (file-byte sha256, count, chained previous seal).\n\n" + md_table(pd.DataFrame(rows)))
    if missing:
        text += f"\nPartitions with planned checkpoints but no sealed selection batch: {', '.join(missing)}.\n"
    return text


def _ci(row: pd.Series, lo: str, hi: str) -> str:
    a, b = row.get(lo), row.get(hi)
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return "n/a"
    return f"[{a:.3f}, {b:.3f}]"


def primary_section(analysis: "Analysis") -> str:
    frame = analysis.extras.get("primary")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return "## Primary contrasts\n\n_(not estimable: checkpoints absent)_\n"
    out = ["## Primary contrasts (frozen, main allowance)", "",
           "Per-item contrast first, then equal-benchmark macro (BBH subtasks equal). Intervals: studentized item-"
           "cluster bootstrap when complete (at small n it can extend past the parameter range; the percentile "
           "interval is shown beside it), otherwise known-only percentile intervals labeled partial with "
           "fixed-population bounds. P-values (Holm over exactly these two) are reported only when both populations "
           "are fully known.", ""]
    rows = []
    for rec in frame[frame["role"] == "primary"].to_dict("records"):
        series = pd.Series(rec)
        macro = rec["benchmark"] == "macro"
        p_value = rec.get("p_value")
        p_text = (fmt(p_value) if macro and not _nan(p_value) else
                  ("withheld (incomplete)" if macro else "not reported (per-benchmark)"))
        holm_text = (fmt(rec.get("p_holm")) if macro and not _nan(rec.get("p_holm")) else
                     ("withheld" if macro else "-"))
        rows.append({"contrast": rec["contrast"], "level": rec["benchmark"], "n_items": rec.get("n_items"),
                     "n_complete": rec.get("n_complete"), "status": rec.get("status"),
                     "estimate": rec.get("estimate"),
                     "ci95 studentized": _ci(series, "ci95_lo", "ci95_hi"),
                     "ci95 percentile": _ci(series, "ci95_percentile_lo", "ci95_percentile_hi"),
                     "bounds": _ci(series, "bound_lower", "bound_upper"), "p": p_text, "p_holm": holm_text})
    out.append(md_table(pd.DataFrame(rows)))
    out.append("### CP decomposition (descriptive)\n")
    out.append("`L = mean(Y_CP,h=0 - Y_PRIVATE_TEAM,r=1)`; `H(h) = mean[(Y_CP,h - Y_CP,0) - (Y_PRIVATE_TEAM,1+h - "
               "Y_PRIVATE_TEAM,1)]`. A descriptive decomposition of a paired controlled-exposure comparison at matched "
               "nominal generations; H alone is not a cost-optimized causal frontier.\n")
    desc = frame[frame["role"].isin(["descriptive", "exploratory"])]
    rows = [{"contrast": r["contrast"], "role": r["role"], "level": r["benchmark"], "n_items": r.get("n_items"),
             "status": r.get("status"), "estimate": r.get("estimate"),
             "ci95": _ci(pd.Series(r), "ci95_percentile_lo", "ci95_percentile_hi"),
             "bounds": _ci(pd.Series(r), "bound_lower", "bound_upper")} for r in desc.to_dict("records")]
    out.append(md_table(pd.DataFrame(rows)))
    notes = [n for n in analysis.extras.get("notes", []) if "Primary p-values" in n or "cp/0" in n]
    if notes:
        out.append(bullet_list(notes))
    return "\n".join(out)


def _wide_accuracy(acc: pd.DataFrame, partition: str) -> pd.DataFrame:
    sub = acc[(acc["partition_id"] == partition) & (acc["view"] == "current") & (acc["level"] != "stratum")]
    rows = []
    for name in sorted(set(sub["name"]), key=name_sort_key):
        row: dict[str, Any] = {"checkpoint": name}
        for level in LEVELS:
            match = sub[(sub["name"] == name) & (sub["benchmark"] == level)]
            if match.empty:
                row[level] = "-"
                continue
            rec = match.iloc[0]
            if rec["acc"] is not None and not pd.isna(rec["acc"]):
                value = f"{rec['acc']:.3f}"
            else:
                value = f"unknown [{rec['acc_lower']:.3f}, {rec['acc_upper']:.3f}]"
            if "ci_lo" in rec and not pd.isna(rec.get("ci_lo")) and rec.get("ci_lo") is not None:
                value += f" ({rec['ci_lo']:.3f}-{rec['ci_hi']:.3f}{'*' if rec.get('ci_basis') != 'planned_population' else ''})"
            row[level] = value
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_section(analysis: "Analysis") -> str:
    acc = analysis.accuracy
    if acc.empty or "coverage" not in acc:
        return ""
    macro = acc[(acc["level"] == "macro")].copy()
    macro = macro.iloc[sorted(range(len(macro)), key=lambda i: (str(macro.iloc[i]["partition_id"]),
                                                                  name_sort_key(macro.iloc[i]["name"]),
                                                                  str(macro.iloc[i]["view"])))]
    return ("## Coverage and selection gap (macro)\n\nOracle coverage of each checkpoint's own declared pool: the "
            "current view gives current coverage, the archive view archive coverage over candidates that policy "
            "really generated. `selection_gap = coverage - delivered` for candidate-only selectors on the same pool; "
            "empty for POOLED synthesis and whenever coverage or delivery is unknown. Per-benchmark rows: "
            "`tables/accuracy_summary`.\n\n"
            + md_table(macro[["partition_id", "name", "view", "acc", "coverage", "selection_gap",
                              "n_coverage_unknown"]], digits=3))


def accuracy_section(analysis: "Analysis") -> str:
    acc = analysis.accuracy
    if acc.empty:
        return "## Delivered accuracy\n\n_(no planned checkpoints)_\n"
    out = ["## Delivered accuracy (current view, every checkpoint)", "",
           "Cells: accuracy (95% item-cluster percentile interval). `unknown [lower, upper]` gives fixed-denominator "
           "bounds when any outcome is unknown or unrun; `*` marks known-only partial intervals. Readouts: last answer "
           "for PRIVATE_SINGLE/DEEP/POOLED, plurality for text sample/team arms, public-test selection for MBPP+. "
           "BBH subtask rows are in `tables/accuracy_summary`.", ""]
    for partition in sorted(set(acc["partition_id"])):
        out += [f"### Partition `{partition}`", "", md_table(_wide_accuracy(acc, partition))]
    return "\n".join(out)


def passk_section(analysis: "Analysis") -> str:
    frame = analysis.extras.get("passk_summary")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return ""
    out = ["## Sampling: oracle pass@k versus achievable readouts", "",
           "Oracle pass@k is the finite-bank estimator on the full independent bank (items with a fully known bank); "
           "it is not an available selector. `prefix coverage` and the readout (vote@k for text, public_select@k for "
           "MBPP+) use the same fixed prefix root[0:k]; their difference is the same-pool selection gap.", ""]
    for partition in sorted(set(frame["partition_id"])):
        sub = frame[frame["partition_id"] == partition]
        rows = []
        for k in sorted(set(sub["k"])):
            row: dict[str, Any] = {"k": int(k)}
            for level in LEVELS:
                m = sub[(sub["k"] == k) & (sub["benchmark"] == level)]
                if m.empty:
                    continue
                r = m.iloc[0]
                label = "public_select" if level == "mbpp_plus" else ("readout" if level == "macro" else "vote")
                row[f"{level} pass@k"] = fmt(r["oracle_pass_at_k"], 3)
                row[f"{level} {label}@k"] = fmt(r["readout"], 3)
            rows.append(row)
        out += [f"### Partition `{partition}`", "", md_table(pd.DataFrame(rows))]
    return "\n".join(out)


def cost_section(analysis: "Analysis") -> str:
    summary = analysis.extras.get("cost_summary")
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return ""
    costs = analysis.extras.get("policy_costs")
    source = costs["cost_source"].iloc[0] if isinstance(costs, pd.DataFrame) and not costs.empty else "n/a"
    out = ["## Accuracy versus resources (policy ledger)", "",
           f"Per-item means of each checkpoint's own from-scratch ancestor set (source: {source}); never total "
           "collection spend. `actual $` = successful + retry charges; retries also shown alone; `standardized $` "
           "prices all prompt tokens uncached at the planning snapshot. Macro rows weight benchmarks equally.", ""]
    macro = summary[summary["level"] == "macro"]
    cols = ["partition_id", "name", "acc", "mean_total_charged_usd", "mean_success_cost_usd",
            "mean_standardized_uncached_cost_usd", "mean_requested_reasoning_tokens", "mean_realized_tokens",
            "mean_reasoning_tokens", "mean_nominal_generations"]
    retry = costs.groupby(["partition_id", "name"])["retry_cost_usd"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").mean()).rename("mean_retry_usd").reset_index() \
        if isinstance(costs, pd.DataFrame) and not costs.empty else pd.DataFrame()
    table = macro[cols].merge(retry, on=["partition_id", "name"], how="left") if not retry.empty else macro[cols]
    precision = {"acc": 3, "mean_total_charged_usd": 6, "mean_success_cost_usd": 6,
                 "mean_standardized_uncached_cost_usd": 6, "mean_requested_reasoning_tokens": 0,
                 "mean_realized_tokens": 0, "mean_reasoning_tokens": 0, "mean_nominal_generations": 1,
                 "mean_retry_usd": 6}
    out.append(md_table(table, headers=["partition", "checkpoint", "macro acc", "actual $", "success $",
                                        "standardized $", "requested reasoning", "realized tokens",
                                        "reasoning tokens", "nominal gens"] + (["retry $"] if not retry.empty else []),
                        digits=precision))
    frontier = analysis.extras.get("frontier")
    if isinstance(frontier, pd.DataFrame) and not frontier.empty:
        out += ["### Descriptive frontier (best among evaluated configurations on this sample; not validated)", "",
                md_table(frontier[frontier["benchmark"] == "macro"][["partition_id", "measure", "name", "mean_cost",
                                                                      "acc"]], digits={"mean_cost": 6, "acc": 3})]
    checks = analysis.extras.get("price_checks", [])
    if checks:
        out += ["### Price units", "", md_table(pd.DataFrame(checks))]
    return "\n".join(out)


def high_low_section(analysis: "Analysis") -> str:
    frame = analysis.extras.get("high_low")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return ""
    macro = frame[frame["level"] == "macro"]
    rows = [{"variant": r.get("variant"), "contrast": r["contrast"], "shared_items": r.get("n_items"),
             "status": r.get("status"), "estimate": r.get("estimate"),
             "ci95": _ci(pd.Series(r), "ci95_percentile_lo", "ci95_percentile_hi")}
            for r in macro.to_dict("records")]
    return ("## Panel variants versus main (shared item ids only, exploratory)\n\nHigher allowance (`high`) and the "
            "optional second-model panel (`replication`) are compared with `main` only on the item ids both "
            "planned; within-Gemini replication does not establish cross-family transfer.\n\n"
            + md_table(pd.DataFrame(rows)) + "\nPer-benchmark rows: `tables/high_vs_main_shared_items`.\n")


def replay_section(analysis: "Analysis") -> str:
    curves = analysis.extras.get("replay_curves")
    if not isinstance(curves, pd.DataFrame) or curves.empty:
        return ""
    macro = curves[curves["level"] == "macro"]
    out = ["## Budgeted policy replay (per-item dollar budgets)", "",
           "Causal admission over stored prefixes: a call or synchronous round is admitted only when its ex ante "
           "reservation (rendered prompt size and declared caps) fits the remaining per-item budget; initialization "
           "that does not fit is an abstention, never a dropped item. Curves without a reservation basis are labeled "
           "descriptive.", "",
           md_table(macro[["partition_id", "trajectory", "budget_usd", "label", "accuracy", "abstention_rate",
                           "binding_cap_rate", "mean_spend_usd", "mean_slack_usd", "n_unknown"]], digits=6)]
    return "\n".join(out)


def dynamics_section(analysis: "Analysis") -> str:
    frame = analysis.extras.get("transition_summary")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return ""
    macro = frame[frame["level"] == "macro"]
    return ("## Dynamics (known transitions, macro means per item)\n\nInvalid outputs (I) stay distinct from valid "
            "wrong answers (W).\n\n"
            + md_table(macro[["partition_id", "trajectory", "from_step", "to_step", "n_items", "correct_prev",
                              "correct_next", "W_to_C", "C_to_W", "I_to_C", "C_to_I", "last_correct_lost",
                              "recovered"]], digits=3))


def operations_section(analysis: "Analysis") -> str:
    ops = analysis.extras.get("operations_condensed")
    out = ["## Operations: realized versus requested budgets, headroom, truncation, cache, latency, retries", "",
           "Per partition x generation kind x round (all benchmarks; per-benchmark rows in `tables/operations`). "
           "Prompt growth shows in `mean_prompt_tokens` across rounds; `mean_completion_over_cap` and "
           "`share_finish_length` show headroom use and truncation; a response-cache HIT on an independent draw "
           "is a blocker.", ""]
    if isinstance(ops, pd.DataFrame) and not ops.empty:
        out.append(md_table(ops[["partition_id", "kind", "team_size", "update_round", "generations",
                                 "mean_prompt_tokens", "mean_requested_reasoning", "mean_reasoning_tokens",
                                 "mean_completion_over_cap", "share_finish_length", "share_parse_invalid",
                                 "share_response_cache_hit", "mean_cached_prompt_share", "median_latency_seconds"]],
                            digits=3))
    retries = analysis.extras.get("retries")
    if isinstance(retries, pd.DataFrame) and not retries.empty:
        charged = pd.to_numeric(retries["retry_charges_usd"], errors="coerce").sum()
        out.append(f"Billed retries: {int(retries['retried_attempts'].sum())} extra attempts on {len(retries)} nodes, "
                   f"{charged:.6f} USD (details: `tables/billed_retries`).\n")
    else:
        out.append("Billed retries: none recorded.\n")
    stops = analysis.extras.get("stop_reasons", [])
    out.append("Stop reasons and gate decisions:\n")
    out.append(md_table(pd.DataFrame(stops)) if stops else "- none recorded in state/status.json or gates/\n")
    return "\n".join(out)


def figures_section(analysis: "Analysis") -> str:
    figures = analysis.extras.get("figures") or []
    if not figures:
        return ""
    paths = analysis.run.paths
    lines = ["## Figures", ""]
    for fig in figures:
        rel = "../" + paths.relative(fig)
        lines.append(f"![{fig.stem}]({rel})")
        lines.append("")
    return "\n".join(lines)


def warnings_section(analysis: "Analysis") -> str:
    out = ["## Small-panel warnings", "", bullet_list(analysis.extras.get("warnings", []))]
    notes = [n for n in analysis.extras.get("notes", []) if "Primary p-values" not in n]
    out += ["## Notes", "", bullet_list(notes)]
    problems = sorted(set(list(analysis.run.problems) + list(analysis.extras.get("problems", []))))
    out += ["## Load and consistency problems", "", bullet_list(problems)]
    return "\n".join(out)


def cost_reconciliation_report(analysis: "Analysis") -> str:
    rec = analysis.extras.get("reconciliation")
    parts = [header(analysis, "Cost reconciliation")]
    if rec is None:
        return "\n".join(parts + ["_(not computed)_\n"])
    ledger, usage, provider = rec["ledger"], rec["usage"], rec["provider"]
    parts.append(f"**Status: {'RECONCILED' if rec['reconciled'] else 'NOT RECONCILED'}.** A final run report "
                 "requires reconciled costs.\n")
    parts.append("## Collection ledger (journal replay)\n")
    rows = [{"quantity": k, "value": str(v)} for k, v in ledger.items() if k not in ("per_partition",
                                                                                      "invariant_violations")]
    rows.append({"quantity": "invariant_violations", "value": str(len(ledger["invariant_violations"]))})
    parts.append(md_table(pd.DataFrame(rows)))
    per = [{"partition": p, **{k: str(v) for k, v in vals.items()}} for p, vals in ledger["per_partition"].items()]
    parts.append("### Per partition\n\n" + md_table(pd.DataFrame(per)))
    if ledger["invariant_violations"]:
        parts.append("### Invariant violations\n\n" + bullet_list(ledger["invariant_violations"][:50]))
    parts.append("## Accepted generation usage versus ledger settlements\n")
    parts.append(md_table(pd.DataFrame([{"quantity": k, "value": str(len(v)) if isinstance(v, list) else str(v)}
                                        for k, v in usage.items()])))
    if usage["attempt_mismatches"]:
        parts.append(bullet_list(usage["attempt_mismatches"][:50]))
    parts.append("## Provider `/key` usage snapshots\n")
    parts.append(md_table(pd.DataFrame([{"quantity": k, "value": str(v)} for k, v in provider.items()])))
    parts.append("A shared account's unrelated spending appears as an unexplained provider-side difference; it is "
                 "never attributed to study policies. Policy costs in RUN_REPORT use each checkpoint's own ancestors, "
                 "not this collection total.\n")
    return "\n".join(parts)


def completeness_report(analysis: "Analysis") -> str:
    parts = [header(analysis, "Completeness")]
    parts.append("Every planned partition x benchmark x condition appears below. `not_started` and `in_progress` "
                 "are administrative (never admitted or unfinished) and are not model errors; `op_failure` is a "
                 "known operational failure; the `unknown_*`, `pending_grade`, `missing_selection` and `unsealed` "
                 "columns are unknown outcomes, never zeros.\n")
    comp = analysis.completeness
    if not comp.empty:
        order = sorted(range(len(comp)), key=lambda i: (str(comp.iloc[i]["partition_id"]),
                                                        (BENCHMARKS.index(comp.iloc[i]["benchmark"])
                                                         if comp.iloc[i]["benchmark"] in BENCHMARKS else 99),
                                                        name_sort_key(comp.iloc[i]["name"]), str(comp.iloc[i]["view"])))
        comp = comp.iloc[order]
    parts.append("## Checkpoints\n\n" + md_table(comp))
    parts.append("## Generation nodes and candidate grades\n\n" + md_table(analysis.nodes))
    return "\n".join(parts)


def _nan(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


__all__ = ["run_report", "cost_reconciliation_report", "completeness_report", "interim_reasons", "PRIMARY_I",
           "PRIMARY_II", "Sequence"]
