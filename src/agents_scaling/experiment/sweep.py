"""Expand a sweep YAML into a deduplicated list of ExperimentCells.

Validity collapsing:
  * SINGLE_AGENT ignores context-share and rounds -> collapse to canonical values so the
    cross-product does not emit duplicate single-agent cells.
  * INDEPENDENT agents never see peers -> context-share is a no-op -> collapse to a single
    canonical context level.
Baseline coupling (plan Risk 6):
  * Every (model_size, benchmark) present in the sweep MUST get a SINGLE_AGENT cell (at a
    canonical prompt level) so efficiency ratios have their T_SAS / E_SAS baseline.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml

from agents_scaling.config import ContextShareLevel, ExperimentCell, Topology

_CANONICAL_CONTEXT = ContextShareLevel.ARTIFACT_ONLY
_CANONICAL_PROMPT = 1  # the "standard" prompt level used for baseline SAS cells


def _canonicalize(cell: ExperimentCell) -> ExperimentCell:
    """Collapse axis values that a topology ignores, so dedup works."""
    ctx = cell.context_share_level
    rounds = cell.rounds
    if cell.topology == Topology.SINGLE_AGENT:
        ctx, rounds = _CANONICAL_CONTEXT, 1
    elif cell.topology == Topology.INDEPENDENT:
        ctx = _CANONICAL_CONTEXT  # never shares context
    return ExperimentCell(
        model_size=cell.model_size,
        context_share_level=ctx,
        prompt_complexity_level=cell.prompt_complexity_level,
        topology=cell.topology,
        benchmark=cell.benchmark,
        n_agents=cell.n_agents,
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

    raw_cells: list[ExperimentCell] = []
    combos = itertools.product(
        axes["model_size"],
        axes["topology"],
        axes["context_share_level"],
        axes["prompt_complexity_level"],
        axes["benchmark"],
        axes.get("seed", [fixed.get("seed", 0)]),
    )
    for model_size, topo, ctx, prompt_lvl, bench, seed in combos:
        raw_cells.append(
            ExperimentCell(
                model_size=model_size,
                topology=Topology(topo),
                context_share_level=ContextShareLevel(ctx),
                prompt_complexity_level=int(prompt_lvl),
                benchmark=bench,
                n_agents=fixed.get("n_agents", 3),
                rounds=fixed.get("rounds", 2),
                n_samples=fixed.get("n_samples", 5),
                temperature=fixed.get("temperature", 0.7),
                n_questions=fixed.get("n_questions"),
                seed=seed,
            )
        )

    # Enforce a SAS baseline per (model_size, benchmark, seed).
    seeds = axes.get("seed", [fixed.get("seed", 0)])
    for model_size, bench, seed in itertools.product(axes["model_size"], axes["benchmark"], seeds):
        raw_cells.append(
            ExperimentCell(
                model_size=model_size,
                topology=Topology.SINGLE_AGENT,
                context_share_level=_CANONICAL_CONTEXT,
                prompt_complexity_level=_CANONICAL_PROMPT,
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
