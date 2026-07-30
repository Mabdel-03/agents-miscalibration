#!/usr/bin/env python
"""Copy the existing analysis figures into paper/figures/atlas/ and write the atlas appendix.

The source PNGs are 1620-2700 px wide at 180 dpi, i.e. 250-415 effective dpi when
placed at \\linewidth.  They are copied verbatim: resampling them to a nominal 300 dpi
measurably increases total size because matplotlib output is already near-optimal for
flat-colour images.

Float placement matters here.  Roughly sixty consecutive [htbp] floats overflow LaTeX's
output routine and abort the run with "Output loop---100 consecutive dead cycles".
placeins is not installed in this TeX tree, so the appendix uses float's [H] specifier
and an explicit \\clearpage every two figures.

Usage:  stage_atlas.py --src ../analysis/figures --out figures/atlas --tex sections/appendix/E_atlas.tex
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Per-figure captions.  Keyed by file stem; the value is (short title, description).
CAPTIONS: dict[str, tuple[str, str]] = {
    # notebook 01: data health and coverage
    "nb01_coverage_heatmaps": ("Design-cell coverage", "Distinct design cells present per model size and reasoning rung."),
    "nb01_seed_replication": ("Seed replication", "Distribution of completed seeds per design cell."),
    "nb01_intended_grid_completion": ("Intended-grid completion", "Share of the intended post-trim grid that is filled, by stratum."),
    "nb01_missingness_forest": ("Missingness model", "Logistic model of cell completion over the intended core grid, showing which factors predict absence."),
    "nb01_dupes_audit": ("Double-append audit", "Duplicate burden across affected cells, and the accuracy shift that deduplication produces."),
    "nb01_ece_estimators": ("ECE estimator agreement", "Per-agent ECE under the primary equal-mass estimator against the alternative estimators, MCQ cells."),
    "nb01_ece_vs_n": ("ECE against sample size", "The 15-bin plug-in estimator inflates at small $n$ while the debiased estimator stays flat."),
    "nb01_vote_atoms": ("Vote-fraction support", "Vote fraction is discrete with support $\\{1/3, 2/3, 1\\}$, which is why it is scored by exact atom rather than by binning."),
    "nb01_ctx_cap_tokens": ("Realized reasoning tokens", "Realized thinking tokens by capacity and reasoning rung, showing the 32B context ceiling."),
    # notebook 02: single-axis scaling
    "nb02_size_error": ("Error against capacity", "Error rate against model size, faceted by topology and reasoning rung, with power-law fits."),
    "nb02_size_ece_off": ("Calibration against capacity, reasoning off", "Calibration error against model size at reasoning=off, MCQ cells."),
    "nb02_size_ece_unlimited": ("Calibration against capacity, unlimited reasoning", "Calibration error against model size at reasoning=unlimited, MCQ cells."),
    "nb02_delta_vs_size": ("$\\Delta$ECE against capacity", "The coordination calibration penalty against model size, pooled over reasoning rungs."),
    "nb02_reas_knob": ("Reasoning as a knob", "Outcome metrics against the nominal reasoning rung."),
    "nb02_reas_dose": ("Reasoning as a dose", "The same metrics against realized reasoning tokens, which decouples the nominal budget from what the model actually spent."),
    "nb02_mas_vs_sas": ("Three agents against one", "Multi-agent accuracy minus the matched single-agent baseline."),
    "nb02_ae_scaling": ("Error amplification", "Error amplification against capacity; values above 1 indicate that coordination amplifies single-agent errors."),
    "nb02_topology_means": ("Topology means", "Per-benchmark topology means with cluster-bootstrap intervals."),
    "nb02_ctx_contrast": ("Context-sharing contrast", "The plus-CoT minus artifact-only contrast within the communicating topologies."),
    "nb02_comm_compute": ("Communication against compute", "Accuracy against realized per-question tokens, separating communication cost from generation cost."),
    "nb02_fe_forest": ("Item-level fixed-effect model", "Axis effects from an item-level logit with question fixed effects absorbed and standard errors clustered by design cell."),
    # notebook 03: interactions, recipes, frontiers
    "nb03_forest_interactions": ("Two-way interactions", "The largest two-way interaction terms over the pooled core grid."),
    "nb03_gbm_importances": ("Gradient-boosting importances", "Permutation importance of each design factor for predicting accuracy, with grouped cross-validation."),
    "nb03_pdp_strongest_pair": ("Strongest interaction surface", "Two-dimensional partial dependence of accuracy on the strongest interacting factor pair."),
    "nb03_vote_decomp_off": ("Voting-lift decomposition", "Decomposition of the voting lift at reasoning=off into independence and correlation components."),
    "nb03_phi_vs_size": ("Error correlation against capacity", "Cross-agent error correlation $\\phi$ against model size."),
    "nb03_frontier": ("Cost-accuracy frontier", "Accuracy against the inference cost proxy, with the Pareto front marked."),
    "nb03_iso_accuracy": ("Iso-accuracy surface", "The single-agent accuracy surface over capacity and reasoning, with multi-agent cells overlaid."),
    "nb03_recipe_transfer": ("Recipe transfer", "Rank correlation of configuration lift across benchmark pairs."),
    # notebook 04: item-level mechanisms
    "nb04_delta_ece_cells": ("Cell-level $\\Delta$ECE", "The headline estimand: system ECE minus per-agent ECE per topology, core MCQ cells."),
    "nb04_reliability_agent": ("Per-agent reliability", "Reliability diagrams for per-agent confidence, faceted by size band and reasoning rung."),
    "nb04_reliability_system": ("System reliability", "Reliability diagrams for the system-level agreement confidence."),
    "nb04_signed_gap": ("Signed overconfidence", "Signed confidence gap against capacity and against realized reasoning."),
    "nb04_confident_wrong": ("Confident-wrong consensus", "Probability that a team is unanimous and wrong, decentralized against independent."),
    "nb04_tournament_ranks": ("Confidence-signal tournament", "ECE rank of each candidate confidence signal within each stratum."),
    "nb04_verbal_vs_logprob": ("Verbalized against logprob confidence", "Agreement between stated and option-logprob confidence."),
    "nb04_round_dynamics": ("Debate round dynamics", "Answer flips between rounds by capacity, and the accuracy change against confidence drift."),
    "nb04_centralized_anatomy": ("Centralized anatomy", "Orchestrator, sub-agent and vote calibration inside centralized cells."),
    "nb04_context_sharing": ("Context sharing at item level", "Item-level effect of exposing peer chain-of-thought."),
    "nb04_math_rescue": ("MATH confidence families", "Which confidence families remain available on a free-form benchmark with no option logprobs."),
    "nb04_difficulty_spectrum": ("Difficulty spectrum", "Distribution of item difficulty and where coordination changes outcomes."),
    "nb04_difficulty_by_band": ("Difficulty by capacity band", "The difficulty spectrum resolved by model-size band."),
    # axis_relationships subdirectory
    "axis_model_size_outcomes": ("Capacity axis", "All reported outcome metrics against model size."),
    "axis_reasoning_outcomes": ("Reasoning axis", "All reported outcome metrics against the reasoning rung."),
    "axis_topology_outcomes": ("Topology axis", "All reported outcome metrics by topology."),
    "axis_agent_count_outcomes": ("Agent-count axis", "All reported outcome metrics for one against three agents."),
    "axis_prompt_outcomes": ("Prompt axis", "All reported outcome metrics by prompt complexity level."),
    "axis_context_outcomes": ("Context-sharing axis", "All reported outcome metrics by context-sharing level."),
    "axis_size_reasoning_accuracy_heatmap": ("Capacity by reasoning", "Accuracy over the capacity and reasoning grid."),
    "axis_cost_frontier": ("Cost frontier", "Accuracy against the cost proxy with the frontier marked."),
    "axis_controlled_context_accuracy": ("Controlled context effect, accuracy", "The plus-CoT contrast recomputed inside every benchmark, size and reasoning stratum."),
    "axis_controlled_context_vote_ece": ("Controlled context effect, vote ECE", "As above for vote-fraction calibration error."),
    "axis_controlled_context_delta_vote": ("Controlled context effect, $\\Delta$ECE", "As above for the coordination calibration penalty."),
    "axis_controlled_prompt_accuracy": ("Controlled prompt effect", "The L3 minus L0 contrast within every stratum."),
    "axis_controlled_decentralized_vs_independent_accuracy": ("Debate against voting", "The decentralized minus independent accuracy contrast within every stratum."),
    "axis_controlled_centralized_vs_decentralized_accuracy": ("Orchestration against debate", "The centralized minus decentralized accuracy contrast within every stratum."),
    "axis_controlled_agent_family_accuracy": ("Three agents against one, controlled", "The multi-agent minus single-agent accuracy contrast within every stratum."),
    # scaling_equivalence subdirectory
    "rung_equivalent_param_multiplier": ("Rungs as parameter multipliers", "Each reasoning rung expressed as the parameter multiplier that would buy the same accuracy."),
    "adjacent_accuracy_plateaus": ("Adjacent-rung plateaus", "Adjacent-level accuracy contrasts on the capacity and reasoning axes, with plateaus marked."),
}

GROUPS = [
    ("Data health and coverage", "nb01_", "figures produced by notebook 01"),
    ("Single-axis scaling", "nb02_", "figures produced by notebook 02"),
    ("Interactions, recipes and frontiers", "nb03_", "figures produced by notebook 03"),
    ("Item-level mechanisms", "nb04_", "figures produced by notebook 04"),
    ("Axis relationships", "axis_", "figures produced by analysis/scripts/axis\\_relationships.py"),
    ("Scaling equivalence", "", "figures produced by analysis/scripts/scaling\\_equivalence.py"),
]

TALL = {"topologies"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="figures/atlas")
    ap.add_argument("--tex", default="sections/appendix/E_atlas.tex")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pngs = sorted(src.glob("*.png")) + sorted(src.glob("*/*.png"))
    if not pngs:
        raise SystemExit(f"no PNGs under {src}")

    staged: list[tuple[str, str]] = []  # (stem, subdir)
    for p in pngs:
        sub = p.parent.name if p.parent != src else ""
        dest_dir = out / sub if sub else out
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest_dir / p.name)
        staged.append((p.stem, sub))

    assigned: set[str] = set()
    lines = [
        "% Generated by paper/tools/stage_atlas.py.  Do not edit by hand.",
        "\\section{Figure atlas}",
        "\\label{app:atlas}",
        "",
        "This appendix reproduces every figure generated by the analysis pipeline for this",
        "run, in the order the pipeline produces them. Each caption states what the panel",
        "shows; the corresponding numerical evidence is in the tables of",
        "Appendices~\\ref{app:marginals} through~\\ref{app:fits} and in the main text.",
        "",
    ]

    n_written = 0
    for title, prefix, blurb in GROUPS:
        members = [
            (stem, sub)
            for stem, sub in staged
            if stem not in assigned and (stem.startswith(prefix) if prefix else True)
        ]
        if not members:
            continue
        for stem, _ in members:
            assigned.add(stem)
        lines += ["\\clearpage", f"\\subsection{{{title}}}", f"\\noindent {blurb}.", ""]
        for i, (stem, sub) in enumerate(members):
            short, desc = CAPTIONS.get(stem, (stem.replace("_", " "), ""))
            rel = f"{sub}/{stem}.png" if sub else f"{stem}.png"
            caption = f"\\textbf{{{short}.}} {desc}" if desc else short
            lines += [
                "\\begin{figure}[H]",
                "\\centering",
                f"\\includegraphics[width=\\linewidth,height=0.42\\textheight,keepaspectratio]{{{rel}}}",
                f"\\caption{{{caption} Source file: \\texttt{{{stem.replace('_', chr(92) + '_')}.png}}.}}",
                f"\\label{{fig:atlas-{stem.replace('_', '-')}}}",
                "\\end{figure}",
                "",
            ]
            n_written += 1
            if i % 2 == 1:
                lines += ["\\clearpage", ""]
        lines += ["\\clearpage", ""]

    tex = Path(args.tex)
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text("\n".join(lines) + "\n")

    missing = [s for s, _ in staged if s not in CAPTIONS]
    if missing:
        print(f"  note: {len(missing)} figures have no hand-written caption: {missing[:6]}")
    print(f"staged {len(staged)} atlas figures, wrote {n_written} float blocks to {tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
