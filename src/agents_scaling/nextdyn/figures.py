"""F1-F6 and their manifest (spec 20.1-20.2; plan §4.5).

Every figure reads **only** derived tables, and every figure's series are written to
``figures/figure_manifest.json`` beside the sha256 of each table it read.  That manifest is the
point of the module: spec 20.2 requires source hashes, filters, item and root counts, task
weights, the uncertainty procedure, the missingness convention, the protocol and selector
hashes, the budget definition and the script revision, and ``release.check`` refuses a release
whose figure hashes do not match the current derived tables.  A figure that cannot be tied back
to the exact rows it was drawn from is a picture, not evidence.

Rendering is deliberately separable.  :func:`render` always writes the manifest and the series;
it draws PNGs only when matplotlib is available and ``draw=True``, so the confirmatory artifact
(the numbers and their provenance) never depends on a plotting backend being installed on the
node that produced it.

Which figure carries which claim (spec 20.1; the plan's F1-F6):

* **F1** natural root ``C_0`` distributions per task, and the natural vs standardised
  PEER-PRIVATE effect with its overlap and ESS beside it -- so a standardised number is never
  shown without how much of the frame it retained.
* **F2** C1/C4 trajectories of ``M``, ``Q`` and ``A`` over rounds with the continuous-survival
  and validity panels, the two cohorts in **separate** panels (a pooled mean here would hide a
  sign reversal, which spec 20.2 forbids).
* **F3** the E2a/E2b endorsement effect crossed with the checker and the budget, by depth.
* **F4** the four directed cells: focal retention, non-focal adoption, and the wrong mirror.
* **F5** 32B transfer rows and their prediction errors; the Llama row is drawn as NOT_RUN.
* **F6** pass@k lo/hi with vote@k and audit@k, and delivered accuracy against attributed cost.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from agents_scaling.nextdyn import analysis, outcomes, panels, tables
from agents_scaling.nextdyn.errors import ContractViolation

FIGURES_DIR: Final[str] = "figures"
FIGURE_MANIFEST: Final[str] = "figure_manifest.json"

#: The declared figure manifest that says which derived tables each figure may read.  It ships
#: as ``configs/nextdyn/figure_manifest.json``: the guard in ``tests/nextdyn/test_guards.py``
#: pins ``configs/nextdyn/*.yaml`` to exactly the six compiled YAMLs, so this seventh config is
#: JSON (see the contract §2.7 note).
DEFAULT_MANIFEST_CONFIG: Final[str] = "figure_manifest.json"


@dataclass(frozen=True)
class FigureSpec:
    """One figure: what it claims, which derived tables it may read, and how it is drawn."""

    figure_id: str
    caption: str
    population_kind: str          # natural | constructed | mirrored | restricted_transfer
    sources: tuple[str, ...]
    builder: Callable[["FigureContext"], list[dict[str, Any]]] = field(repr=False)
    panels_note: str = ""


@dataclass(frozen=True)
class FigureContext:
    """The derived tables one render pass reads, plus the provenance every figure must cite."""

    run_root: Path
    ctx: analysis.AnalysisContext
    json_artifacts: Mapping[str, Any]

    def artifact(self, name: str) -> Any:
        return self.json_artifacts.get(name, [])


def _load_json(path: Path) -> Any:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return []


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(derived: Path, sources: Sequence[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in sources:
        for candidate in (derived / f"{name}.parquet", derived / f"{name}.json",
                          derived / f"{name}{tables.EMPTY_SUFFIX}"):
            if candidate.exists():
                out[name] = _sha256(candidate)
                break
        else:
            out[name] = None
    return out


# --------------------------------------------------------------------------- builders


def _f1(fc: FigureContext) -> list[dict[str, Any]]:
    counts: dict[tuple[str, int], int] = {}
    for row in fc.ctx.frame.team_rounds:
        if int(row["round"]) != 0 or str(row.get("cohort")) != panels.COHORT_NATURAL:
            continue
        counts[(str(row.get("family") or ""), int(row["c_count"]))] = \
            counts.get((str(row.get("family") or ""), int(row["c_count"])), 0) + 1
    series = [{"panel": "root_c0_distribution", "family": family, "c0": c0, "n": n}
              for (family, c0), n in sorted(counts.items())]
    for row in fc.artifact("standardization"):
        if row.get("stratum") == ["__summary__"]:
            series.append({"panel": "standardized_effect", "delta_std": row.get("delta_std"),
                           "overlap_retained": row.get("overlap_retained"), "ess": row.get("ess"),
                           "n_strata_supported": row.get("n_strata_supported")})
    return series


def _f2(fc: FigureContext) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in fc.ctx.frame.team_rounds:
        composition = str(row.get("composition") or "")
        if not composition:
            continue
        cells.setdefault((composition, str(row.get("display_label") or ""), int(row["round"])), []).append(row)
    series = []
    for (composition, protocol, round_), rows in sorted(cells.items()):
        series.append({
            "panel": composition, "protocol": protocol, "round": round_, "n": len(rows),
            "m_lo": sum(float(r["m_lo"]) for r in rows) / len(rows),
            "m_hi": sum(float(r["m_hi"]) for r in rows) / len(rows),
            "q_lo": sum(float(r["q_lo"]) for r in rows) / len(rows),
            "q_hi": sum(float(r["q_hi"]) for r in rows) / len(rows),
            "a_lo": (sum(float(r["a_lo"]) for r in rows if r.get("a_lo") is not None)
                     / max(1, sum(1 for r in rows if r.get("a_lo") is not None))
                     if any(r.get("a_lo") is not None for r in rows) else None),
            "continuous_preservation": (
                sum(1 for r in rows if r.get("continuous_preservation_lo")) / len(rows)),
            "validity_rate": sum(float(r["n_valid"]) / float(r["n_agents"]) for r in rows) / len(rows),
        })
    return series


def _f3(fc: FigureContext) -> list[dict[str, Any]]:
    return [dict(row) for row in fc.artifact("e2_factorial")]


def _f4(fc: FigureContext) -> list[dict[str, Any]]:
    return [dict(row) for row in fc.artifact("e3_factorial")]


def _f5(fc: FigureContext) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in fc.ctx.frame.team_rounds:
        cells.setdefault((str(row.get("model_hash") or ""), str(row.get("display_label") or "")), []).append(row)
    series = [{"panel": "transfer", "model_hash": model, "protocol": protocol, "n": len(rows),
               "q_lo": sum(float(r["q_lo"]) for r in rows) / len(rows)}
              for (model, protocol), rows in sorted(cells.items())]
    series.append({"panel": "transfer", "model_hash": "llama-3.1-8B", "protocol": "-",
                   "n": 0, "q_lo": None, "verdict": analysis.VERDICT_NOT_RUN,
                   "note": "weights are gated and absent (amendment A4)"})
    return series


def _f6(fc: FigureContext) -> list[dict[str, Any]]:
    series = [dict(row) for row in fc.artifact("passk_panel")]
    for row in tables.read_table(fc.run_root / outcomes.DERIVED_DIR, "efficiency_frontier") \
            if (fc.run_root / outcomes.DERIVED_DIR / "efficiency_frontier.parquet").exists() else []:
        series.append({"panel": "efficiency", **row})
    return series


FIGURES: Final[Mapping[str, FigureSpec]] = {
    spec.figure_id: spec for spec in (
        FigureSpec("F1", "Natural root correct-count distributions, and the natural vs "
                         "standardised PEER-PRIVATE effect with overlap and ESS",
                   "natural", ("team_rounds", "standardization"), _f1,
                   "one panel per task; the standardised point is never shown without its overlap"),
        FigureSpec("F2", "C1 and C4 trajectories of margin, coverage and delivered accuracy "
                         "over rounds, with continuous survival and validity",
                   "constructed", ("team_rounds", "selections_graded"), _f2,
                   "the two cohorts are separate panels: a pooled mean would hide a sign reversal"),
        FigureSpec("F3", "E2a/E2b endorsement effect crossed with checker access and thinking "
                         "budget, by chain depth",
                   "mirrored", ("agent_rounds", "e2_factorial"), _f3),
        FigureSpec("F4", "The four directed cells: focal retention, non-focal adoption of the "
                         "shared wrong class, and the wrong mirror",
                   "constructed", ("team_rounds", "e3_factorial"), _f4),
        FigureSpec("F5", "32B transfer rows and prediction errors; the Llama row is NOT_RUN",
                   "restricted_transfer", ("team_rounds",), _f5),
        FigureSpec("F6", "pass@k lo/hi with vote@k and audit@k, and delivered accuracy against "
                         "attributed cost",
                   "natural", ("passk_panel", "efficiency_frontier"), _f6),
    )
}


# --------------------------------------------------------------------------- render


def _git_revision(repo_root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git on the node
        return "unknown"


def _draw(path: Path, figure_id: str, series: Sequence[Mapping[str, Any]]) -> str | None:
    """A minimal line/bar rendering; absence of matplotlib is reported, never fatal."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - environment dependent
        return None
    figure, axes = plt.subplots(figsize=(6, 4))
    numeric = [(i, v) for i, row in enumerate(series)
               for key, v in row.items()
               if key in ("q_lo", "pass_at_k_lo", "accuracy_lo", "mean_lo", "n")
               and isinstance(v, (int, float))]
    if numeric:
        axes.plot([i for i, _ in numeric], [v for _, v in numeric], marker="o", linewidth=1)
    axes.set_title(figure_id)
    axes.set_xlabel("series index")
    axes.set_ylabel("value")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return _sha256(path)


def render(run_root: str | os.PathLike, *, config_path: str | os.PathLike | None = None,
           profile: str | None = None, draw: bool = False,
           only: Sequence[str] | None = None) -> dict[str, Any]:
    """Build F1-F6 from the derived tables and write ``figures/figure_manifest.json``."""
    root = Path(run_root)
    ctx = analysis.load_context(root, config_path=config_path, profile=profile)
    derived = root / outcomes.DERIVED_DIR
    artifacts = {name: _load_json(derived / f"{name}.json")
                 for name in ("standardization", "calibration_panel", "passk_panel",
                              "e2_factorial", "e3_factorial", "copying_placebo")}
    fc = FigureContext(run_root=root, ctx=ctx, json_artifacts=artifacts)
    selected = tuple(only) if only else tuple(FIGURES)
    unknown = sorted(set(selected) - set(FIGURES))
    if unknown:
        raise ContractViolation(f"unknown figure(s) {unknown}; declared: {sorted(FIGURES)}")
    folder = root / FIGURES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for figure_id in selected:
        spec = FIGURES[figure_id]
        series = spec.builder(fc)
        series_path = folder / f"{figure_id}.json"
        series_path.write_text(json.dumps(series, indent=1, sort_keys=True, default=str) + "\n",
                               encoding="utf-8")
        png = _draw(folder / f"{figure_id}.png", figure_id, series) if draw else None
        entries[figure_id] = {
            "figure_id": figure_id,
            "caption": spec.caption,
            "population_kind": spec.population_kind,
            "panels_note": spec.panels_note,
            "sources": list(spec.sources),
            "source_sha256": _source_hashes(derived, spec.sources),
            "series_file": series_path.name,
            "series_sha256": _sha256(series_path),
            "png_sha256": png,
            "n_series": len(series),
            "filters": {"endpoint_round": ctx.endpoint_round,
                        "primary_hypotheses": list(ctx.cfg.analysis.primary_hypotheses)},
            "n_items": len({str(r["source_item_id"]) for r in ctx.frame.team_rounds}),
            "n_root_teams": len(ctx.frame.root_teams),
            "task_weights": ctx.family_weights,
            "uncertainty_procedure": ("studentised wild cluster bootstrap over source items, "
                                      f"{ctx.settings.n_resamples} resamples; intervals from "
                                      "ard.stats.paired_contrast percentile-t"),
            "missingness_convention": ("three-state grades; every rate is reported as a lo/hi "
                                       "pair with a complete-case sensitivity beside it"),
            "protocol_hashes": sorted({str(r.get("protocol_hash") or "")
                                       for r in ctx.frame.team_rounds}),
            "selector_hashes": sorted({f"{r['selector']}.{r['selector_version']}"
                                       for r in ctx.selections_graded}),
            "budget_definition": ("thinking + final token caps of the episode's TokenContract; "
                                  "cost is the attributed linear compute proxy"),
            "script_revision": _git_revision(Path(__file__).resolve().parents[3]),
        }
    manifest = {"figures": entries, "derived_dir": str(derived),
                "study_id": str(ctx.cfg.study.id)}
    path = folder / FIGURE_MANIFEST
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")
    return manifest


def manifest_path(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / FIGURES_DIR / FIGURE_MANIFEST


def stale_figures(run_root: str | os.PathLike) -> list[str]:
    """Figures whose recorded source hashes no longer match the derived tables on disk."""
    path = manifest_path(run_root)
    if not path.exists():
        return sorted(FIGURES)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    derived = Path(run_root) / outcomes.DERIVED_DIR
    stale = []
    for figure_id, entry in manifest.get("figures", {}).items():
        current = _source_hashes(derived, entry.get("sources", ()))
        if current != entry.get("source_sha256"):
            stale.append(figure_id)
    stale.extend(sorted(set(FIGURES) - set(manifest.get("figures", {}))))
    return sorted(set(stale))


__all__ = [
    "DEFAULT_MANIFEST_CONFIG", "FIGURES", "FIGURES_DIR", "FIGURE_MANIFEST", "FigureContext",
    "FigureSpec", "manifest_path", "render", "stale_figures",
]
