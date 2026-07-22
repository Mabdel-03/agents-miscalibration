"""Expand a sweep YAML into a deduplicated list of ExperimentCells.

Validity collapsing:
  * SINGLE_AGENT ignores context-share and rounds -> collapse to canonical values so the
    cross-product does not emit duplicate single-agent cells.
  * INDEPENDENT agents never see peers -> context-share is a no-op -> collapse to a single
    canonical context level.
Baseline coupling (plan Risk 6):
  * Every (model_size, benchmark, seed, reasoning_level) present in the sweep MUST get a
    SINGLE_AGENT cell (at a canonical prompt level) so efficiency ratios have their
    T_SAS / E_SAS baseline. Efficiency is reasoning-conditioned, so the baseline is too.

Reasoning axis (Axis 4) is per-agent and is NOT collapsed by topology (every topology can
think); only context-share collapses for SAS/Independent.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml

from agents_scaling.config import ContextShareLevel, ExperimentCell, ReasoningLevel, Topology

_CANONICAL_CONTEXT = ContextShareLevel.ARTIFACT_ONLY
_CANONICAL_PROMPT = 1  # the "standard" prompt level used for baseline SAS cells


def _canonicalize(cell: ExperimentCell) -> ExperimentCell:
    """Collapse axis values that a topology ignores, so dedup works."""
    ctx = cell.context_share_level
    n_agents = cell.n_agents
    rounds = cell.rounds
    if cell.topology == Topology.SINGLE_AGENT:
        ctx, n_agents, rounds = _CANONICAL_CONTEXT, 1, 1
    elif cell.topology == Topology.INDEPENDENT:
        ctx = _CANONICAL_CONTEXT  # never shares context
    return ExperimentCell(
        model_size=cell.model_size,
        context_share_level=ctx,
        prompt_complexity_level=cell.prompt_complexity_level,
        reasoning_level=cell.reasoning_level,
        topology=cell.topology,
        benchmark=cell.benchmark,
        n_agents=n_agents,
        rounds=rounds,
        n_samples=cell.n_samples,
        temperature=cell.temperature,
        n_questions=cell.n_questions,
        seed=cell.seed,
    )


def generate_cells(spec: dict) -> list[ExperimentCell]:
    """Expand a sweep spec dict into canonical, deduplicated cells (SAS baselines enforced)."""
    axes = spec["axes"]
    fixed = spec.get("fixed", {})

    # reasoning_level defaults to ["off"] if the axis is absent (back-compat with 3-axis configs).
    reasoning_levels = axes.get("reasoning_level", ["off"])
    seeds = axes.get("seed", [fixed.get("seed", 0)])
    agent_counts = axes.get("n_agents", [fixed.get("n_agents", 3)])
    include_sas_baselines = spec.get("include_sas_baselines", True)

    raw_cells: list[ExperimentCell] = []
    combos = itertools.product(
        axes["model_size"],
        axes["topology"],
        axes["context_share_level"],
        axes["prompt_complexity_level"],
        reasoning_levels,
        axes["benchmark"],
        seeds,
        agent_counts,
    )
    for model_size, topo, ctx, prompt_lvl, reasoning, bench, seed, n_agents in combos:
        topology = Topology(topo)
        raw_cells.append(
            ExperimentCell(
                model_size=model_size,
                topology=topology,
                context_share_level=ContextShareLevel(ctx),
                prompt_complexity_level=int(prompt_lvl),
                reasoning_level=ReasoningLevel(reasoning),
                benchmark=bench,
                n_agents=1 if topology == Topology.SINGLE_AGENT else int(n_agents),
                rounds=fixed.get("rounds", 2),
                n_samples=fixed.get("n_samples", 5),
                temperature=fixed.get("temperature", 0.7),
                n_questions=fixed.get("n_questions"),
                seed=seed,
            )
        )

    # Enforce a SAS baseline per (model_size, benchmark, seed, reasoning_level) — efficiency
    # ratios are reasoning-conditioned, so each reasoning level needs its own baseline.
    if include_sas_baselines:
        for model_size, bench, seed, reasoning in itertools.product(
            axes["model_size"], axes["benchmark"], seeds, reasoning_levels
        ):
            raw_cells.append(
                ExperimentCell(
                    model_size=model_size,
                    topology=Topology.SINGLE_AGENT,
                    context_share_level=_CANONICAL_CONTEXT,
                    prompt_complexity_level=_CANONICAL_PROMPT,
                    reasoning_level=ReasoningLevel(reasoning),
                    benchmark=bench,
                    n_agents=1,
                    rounds=1,
                    n_samples=fixed.get("n_samples", 5),
                    temperature=fixed.get("temperature", 0.7),
                    n_questions=fixed.get("n_questions"),
                    seed=seed,
                )
            )

    # Canonicalize + dedup by cell_id.
    seen: dict[str, ExperimentCell] = {}
    for c in raw_cells:
        cc = _canonicalize(c)
        seen[cc.cell_id] = cc
    return sorted(seen.values(), key=lambda c: c.cell_id)


def load_sweep(path: str | Path) -> list[ExperimentCell]:
    spec = yaml.safe_load(Path(path).read_text())
    return generate_cells(spec)


def models_in_sweep(cells: list[ExperimentCell]) -> list[str]:
    return sorted({c.model_size for c in cells})
