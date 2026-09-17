"""Static PNG figures (matplotlib Agg; no network, no API key). Every figure has a table twin under ``tables/``.

Design rules (reference data-viz palette, validated with its script): categorical hues in fixed order; lines use
up to seven adjacent slots, scatter-like panels use only the first three slots with every other family folded to
gray "other checkpoints"; 2 px lines, >= 6 pt markers with a surface ring, solid hairline grid, one y-axis per
panel, small multiples per benchmark, text in ink tokens. Unknown accuracies are never drawn (a missing selector
is absent from the plot, not zero). Interim figures carry "PARTIAL COLLECTION" in their titles.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from agent_allocation.analysis.costs import COST_MEASURES  # noqa: E402
from agent_allocation.analysis.names import CheckpointNameError, parse_name  # noqa: E402
from agent_allocation.contracts import BENCHMARKS  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from agent_allocation.reporting.cli import Analysis

SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OTHER = "#c3c2b7"
LEVELS = ("macro",) + BENCHMARKS
LEVEL_TITLES = {"macro": "Macro (equal benchmarks)", "math500": "MATH-500", "mmlu_pro": "MMLU-Pro",
                "bbh_logic4": "BBH four-subtask subset", "mbpp_plus": "MBPP+ (public-test selector)"}
SCATTER_FAMILIES = ("sample", "private_team", "peer")          # first three slots, all-pairs validated
TRAJECTORY_LINES = (("sample", "sample/k"), ("private_single", "private_single/t"),
                    ("private_team", "private_team/4/r"), ("peer", "peer/4/r"), ("cp", "cp/h"))
FIGURE_MEASURES = ("total_charged_usd", "requested_reasoning_tokens", "realized_tokens", "nominal_generations")


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "legend.frameon": False, "legend.fontsize": 8, "text.color": INK,
        "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
        "svg.hashsalt": "agent-allocation", "path.simplify": False,
    })


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.png")
    fig.savefig(tmp, dpi=110, metadata={"Software": None}, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, path)
    return path


def _title(analysis: "Analysis", text: str) -> str:
    if not analysis.status["complete"]:
        return text + "  |  PARTIAL COLLECTION (interim)"
    if not analysis.final:
        return text + "  |  INTERIM: costs not reconciled"
    return text


def _finish(fig, analysis: "Analysis", axes, title: str, xlabel: str, ylabel: str) -> None:
    for ax in axes:
        ax.set_xlabel("")
    fig.supxlabel(xlabel, fontsize=9, color=INK_2, y=0.02)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 7), bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(_title(analysis, title), fontsize=10, color=INK, y=0.99)


def _num(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def make_figures(analysis: "Analysis") -> list[Path]:
    _style()
    out: list[Path] = []
    summary = analysis.extras.get("cost_summary")
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        for partition in sorted(set(summary["partition_id"])):
            sub = summary[summary["partition_id"] == partition]
            for measure in FIGURE_MEASURES:
                if f"mean_{measure}" in sub:
                    out.append(accuracy_vs_measure(analysis, sub, partition, measure))
            out.append(trajectory_figure(analysis, sub, partition))
    passk = analysis.extras.get("passk_summary")
    if isinstance(passk, pd.DataFrame) and not passk.empty:
        for partition in sorted(set(passk["partition_id"])):
            out.append(passk_figure(analysis, passk[passk["partition_id"] == partition], partition))
    curves = analysis.extras.get("replay_curves")
    if isinstance(curves, pd.DataFrame) and not curves.empty:
        for partition in sorted(set(curves["partition_id"])):
            out.append(replay_figure(analysis, curves[curves["partition_id"] == partition], partition))
    return out


def _levels_present(frame: pd.DataFrame) -> list[str]:
    present = set(frame["benchmark"])
    return [lvl for lvl in LEVELS if lvl in present]


def accuracy_vs_measure(analysis: "Analysis", sub: pd.DataFrame, partition: str, measure: str) -> Path:
    levels = _levels_present(sub)
    fig, axes = plt.subplots(1, len(levels), figsize=(3.3 * len(levels), 3.2), squeeze=False)
    col = f"mean_{measure}"
    frontier = analysis.extras.get("frontier")
    for ax, level in zip(axes[0], levels):
        data = sub[sub["benchmark"] == level]
        known = data[data["acc"].map(_num).notna() & data[col].map(_num).notna()]
        other = known[~known["family"].isin(SCATTER_FAMILIES)]
        ax.scatter(other[col].map(float), other["acc"].map(float), s=22, color=OTHER, edgecolors=SURFACE,
                   linewidths=1.2, label="other checkpoints", zorder=2)
        for slot, family in enumerate(SCATTER_FAMILIES):
            fam = known[known["family"] == family]
            if fam.empty:
                continue
            ax.scatter(fam[col].map(float), fam["acc"].map(float), s=34, color=SLOTS[slot], edgecolors=SURFACE,
                       linewidths=1.2, label=family.replace("_", " "), zorder=3)
        if isinstance(frontier, pd.DataFrame) and not frontier.empty:
            f = frontier[(frontier["partition_id"] == partition) & (frontier["benchmark"] == level)
                         & (frontier["measure"] == measure)].sort_values("mean_cost")
            if not f.empty:
                ax.step(f["mean_cost"].astype(float), f["acc"].astype(float), where="post", color=INK, linewidth=1.2,
                        label="descriptive frontier", zorder=4)
        missing = int(len(data) - len(known))
        if missing:
            ax.text(0.02, 0.02, f"{missing} checkpoint(s) unknown: not plotted", transform=ax.transAxes, fontsize=7,
                    color=MUTED)
        if measure.endswith("usd") or measure.endswith("tokens"):
            if (known[col].map(float) > 0).all() and len(known):
                ax.set_xscale("log")
        ax.set_title(LEVEL_TITLES.get(level, level))
        ax.set_ylim(-0.02, 1.02)
    _finish(fig, analysis, list(axes[0]), f"{partition}: delivered accuracy vs resources (frontier is descriptive, "
            "not a validated policy)", COST_MEASURES[measure], "Delivered accuracy")
    return _save(fig, analysis.run.paths.figures_dir / f"{partition}_accuracy_vs_{measure}.png")


def trajectory_figure(analysis: "Analysis", sub: pd.DataFrame, partition: str) -> Path:
    levels = _levels_present(sub)
    fig, axes = plt.subplots(1, len(levels), figsize=(3.3 * len(levels), 3.2), squeeze=False)
    for ax, level in zip(axes[0], levels):
        data = sub[sub["benchmark"] == level]
        for slot, (family, label) in enumerate(TRAJECTORY_LINES):
            points = []
            for rec in data.to_dict("records"):
                try:
                    name = parse_name(rec["name"])
                except CheckpointNameError:
                    continue
                if name.family != family or (family in ("private_team", "peer") and name.params[0] != 4):
                    continue
                acc = _num(rec["acc"])
                if acc is not None:
                    points.append((name.nominal_generations, acc))
            if not points:
                continue
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points], color=SLOTS[slot], linewidth=2,
                    marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.0, label=label,
                    solid_joinstyle="round", solid_capstyle="round")
        ax.set_xscale("log", base=2)
        ax.set_title(LEVEL_TITLES.get(level, level))
        ax.set_ylim(-0.02, 1.02)
    _finish(fig, analysis, list(axes[0]), f"{partition}: delivered accuracy along nested prefixes (team size 4; "
            "accuracy need not be monotone)", "Nominal generations per item (log scale)", "Delivered accuracy")
    return _save(fig, analysis.run.paths.figures_dir / f"{partition}_trajectories.png")


def passk_figure(analysis: "Analysis", sub: pd.DataFrame, partition: str) -> Path:
    levels = _levels_present(sub)
    fig, axes = plt.subplots(1, len(levels), figsize=(3.3 * len(levels), 3.2), squeeze=False)
    for ax, level in zip(axes[0], levels):
        data = sub[sub["benchmark"] == level].sort_values("k")
        readout = ("public_select@k" if level == "mbpp_plus" else
                   "vote@k (text) / public_select@k (code)" if level == "macro" else "vote@k")
        for slot, (col, label) in enumerate((("oracle_pass_at_k", "oracle pass@k (not a selector)"),
                                             ("prefix_coverage", "prefix coverage root[0:k]"),
                                             ("readout", readout))):
            pts = [(int(k), _num(v)) for k, v in zip(data["k"], data[col]) if _num(v) is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SLOTS[slot], linewidth=2, marker="o",
                        markersize=4, markeredgecolor=SURFACE, markeredgewidth=1.0, label=label)
        ax.set_title(LEVEL_TITLES.get(level, level))
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="lower right", fontsize=7)
    for ax in axes[0]:
        ax.set_xlabel("")
    fig.supxlabel("k (independent samples, fixed prefix root[0:k])", fontsize=9, color=INK_2, y=0.02)
    axes[0][0].set_ylabel("Rate")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.suptitle(_title(analysis, f"{partition}: oracle pass@k versus achievable readouts"), fontsize=10, color=INK,
                 y=0.99)
    return _save(fig, analysis.run.paths.figures_dir / f"{partition}_passk_vs_readout.png")


REPLAY_TRAJECTORIES = ("sample", "private_single", "private_team/4", "peer/4", "cp", "pooled/4", "deep/16")


def replay_figure(analysis: "Analysis", sub: pd.DataFrame, partition: str) -> Path:
    levels = _levels_present(sub)
    fig, axes = plt.subplots(1, len(levels), figsize=(3.3 * len(levels), 3.2), squeeze=False)
    for ax, level in zip(axes[0], levels):
        data = sub[sub["benchmark"] == level]
        for slot, traj in enumerate(REPLAY_TRAJECTORIES):
            t = data[data["trajectory"] == traj].sort_values("budget_usd")
            pts = [(float(b), _num(a)) for b, a in zip(t["budget_usd"], t["accuracy"]) if _num(a) is not None]
            if not pts:
                continue
            descriptive = not (t["label"] == "budgeted policy replay").all()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SLOTS[slot], linewidth=2,
                    alpha=0.45 if descriptive else 1.0, label=traj + (" (descriptive)" if descriptive else ""))
        ax.set_xscale("log")
        ax.set_title(LEVEL_TITLES.get(level, level))
        ax.set_ylim(-0.02, 1.02)
    _finish(fig, analysis, list(axes[0]), f"{partition}: budgeted policy replay (causal admission, prefix-observable "
            "reservations)", "Per-item budget (USD, log scale)", "Delivered accuracy under the budget")
    return _save(fig, analysis.run.paths.figures_dir / f"{partition}_budget_replay.png")
