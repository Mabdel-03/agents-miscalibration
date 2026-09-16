"""Sample-size planning from the pilot (spec 14.3, 19.1; plan §4.4).

The ``power`` verb, run once between the pilot and the freeze.  It answers one question per
primary hypothesis: **how many items per task does this contrast need for 80% power at the
Holm-corrected alpha?**  Three design choices make the answer honest rather than optimistic:

* **The simulation scores with the same test that will decide the claim.**  Planting an effect
  and scoring it with a normal approximation would report the power of a test nobody is going
  to run; :func:`simulate` resamples pilot items and runs ``stats.wild_cluster_test`` exactly as
  :mod:`agents_scaling.nextdyn.analysis` will.
* **The alpha is the corrected one.**  The planning alpha is
  ``study.analysis.familywise_alpha / len(primary_hypotheses)``, computed here and never
  re-typed in ``analysis.yaml`` (contract §0 rule 9).  Planning at 0.05 and testing at 0.00625
  is how a study arrives under-powered with a pre-registration that says otherwise.
* **The screened-item requirement, not the analysed one, is what has to be dispatched.**  An
  item enters the paired constructed panel only if the donor bank makes it paired-eligible, so
  ``n / eligibility_rate`` per family is reported beside every ``n``; the pilot measures that
  rate, and where it did not, the row says so instead of assuming one.

The effect is planted by **flipping outcomes**, not by adding a constant: the endpoints are
bounded (a coverage indicator, a share of four slots), so a shift model would plant effects the
outcome cannot express and report power for an impossible alternative.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np

from agents_scaling.nextdyn import analysis, cli, panels, stats
from agents_scaling.nextdyn.errors import ContractViolation

POWER_REPORT: Final[str] = "power_report.json"
POWER_LOCK: Final[str] = "power_lock.json"

#: The effect sizes the pilot report sweeps, in outcome units (proportions).  They bracket the
#: ARD36 measured PEER-PRIVATE coverage change (-0.046) on both sides.
DEFAULT_DELTAS: Final[tuple[float, ...]] = (0.03, 0.05, 0.08, 0.12)


@dataclass(frozen=True)
class PilotFrame:
    """Per-item arm outcomes from the pilot, per hypothesis, plus eligibility and cost."""

    per_hypothesis: Mapping[str, Mapping[str, Mapping[str, float]]]
    families: Mapping[str, Mapping[str, str]]
    eligibility: Mapping[str, float]
    missingness: Mapping[str, float]
    cost: Mapping[str, Any] = field(default_factory=dict)

    def items(self, h_id: str) -> list[str]:
        return sorted(self.per_hypothesis.get(h_id, {}))


def pilot_covariance(pilot_root: str | os.PathLike, *,
                     config_path: str | os.PathLike | None = None,
                     profile: str | None = None) -> PilotFrame:
    """Read the pilot's derived tables into the per-item arm vectors the simulation resamples.

    "Covariance" is the point: the arms are paired **within** an item, so what the simulation
    needs is the joint per-item vector, not two marginal rates.  A design that resampled the
    two arms independently would overstate the variance of the paired contrast and plan for
    more items than the study needs.
    """
    ctx = analysis.load_context(pilot_root, config_path=config_path, profile=profile)
    per_hypothesis: dict[str, dict[str, dict[str, float]]] = {}
    families: dict[str, dict[str, str]] = {}
    missingness: dict[str, float] = {}
    for h_id, spec in analysis.HYPOTHESIS_SPECS.items():
        if spec.kind.startswith("e2b"):
            matrix = _e2b_matrix(ctx, spec)
        else:
            base = analysis.HYPOTHESIS_SPECS["H1"] if spec.kind == "joint" else spec
            panel = panels.population(base.population, ctx.frame, ctx.cfg)
            matrix = analysis.endpoint_matrix(
                panel, base.outcome, ctx.endpoint_round, arms=tuple(base.arm_weights),
                rows=ctx.frame.team_rounds, composition=base.composition)
        if isinstance(matrix, analysis.ItemMatrix):
            items = matrix.paired_items()
            per_hypothesis[h_id] = {i: dict(matrix.lo[i]) for i in items}
            families[h_id] = {i: matrix.families[i] for i in items}
            missingness[h_id] = (matrix.n_missing / matrix.n_rows) if matrix.n_rows else 0.0
        else:
            per_hypothesis[h_id], families[h_id], missingness[h_id] = matrix
    eligibility = _eligibility_rates(ctx)
    cost_path = Path(pilot_root) / "cost_profile.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else {}
    return PilotFrame(per_hypothesis=per_hypothesis, families=families,
                      eligibility=eligibility, missingness=missingness, cost=cost)


def _e2b_matrix(ctx: analysis.AnalysisContext, spec: analysis.HypothesisSpec):
    """H4/H5 cluster on the problem, so the "item" is a problem and the "arm" is the reduction."""
    result = analysis.hypothesis(spec.h_id, ctx)
    if not result.estimable:
        return {}, {}, 1.0
    problems = result.detail.get("test", {}).get("n", {})
    values: dict[str, dict[str, float]] = {}
    fams: dict[str, str] = {}
    # the reduced per-problem scalar is the contrast itself; a one-column matrix keeps the
    # simulation's resampling identical to the paired case
    per_problem = result.detail.get("per_problem", {})
    for problem, value in per_problem.items():
        values[problem] = {"contrast": float(value)}
        fams[problem] = panels.E2B
    if not values and problems:
        return {}, {}, 0.0
    return values, fams, 0.0


def _eligibility_rates(ctx: analysis.AnalysisContext) -> dict[str, float]:
    """Paired-eligible items per family divided by the items screened, measured on the pilot."""
    screened: dict[str, set[str]] = {}
    eligible: dict[str, set[str]] = {}
    for row in ctx.frame.banks:
        screened.setdefault(_family_of(ctx, str(row["source_item_id"])), set()).add(str(row["source_item_id"]))
    paired = tuple(ctx.cfg.cohorts.paired_compositions)
    seen: dict[str, set[str]] = {}
    for row in ctx.frame.team_rounds:
        composition = str(row.get("composition") or "")
        if composition in paired:
            seen.setdefault(str(row["source_item_id"]), set()).add(composition)
    for item, have in seen.items():
        if set(paired) <= have:
            eligible.setdefault(_family_of(ctx, item), set()).add(item)
    return {family: (len(eligible.get(family, ())) / len(items)) if items else 0.0
            for family, items in screened.items()}


def _family_of(ctx: analysis.AnalysisContext, item: str) -> str:
    for row in ctx.frame.team_rounds:
        if str(row["source_item_id"]) == item:
            return str(row.get("family") or "")
    for row in ctx.frame.agent_rounds:
        if str(row["source_item_id"]) == item:
            return str(row.get("family") or "")
    return ""


# --------------------------------------------------------------------------- simulation


def _plant(values: np.ndarray, delta: float, rng: np.random.Generator) -> np.ndarray:
    """Plant an effect of ``delta`` by flipping bounded outcomes, never by shifting them.

    The treatment column is moved toward 1 (or toward 0 for a negative delta) on a random
    subset of the items where there is room to move.  The realised shift is the planted one in
    expectation, and the outcome stays inside its own support -- which a constant shift would
    not, on an indicator.
    """
    out = values.copy()
    if delta == 0:
        return out
    target = 1.0 if delta > 0 else 0.0
    movable = np.flatnonzero(np.abs(out - target) > 1e-12)
    if movable.size == 0:
        return out
    room = float(np.mean(np.abs(out[movable] - target)))
    share = min(1.0, abs(delta) * out.size / (movable.size * max(room, 1e-9)))
    chosen = movable[rng.random(movable.size) < share]
    out[chosen] = target
    return out


def simulate(frame: PilotFrame, h_id: str, n_items: int, delta: float, *,
             n_sims: int, alpha: float, seed: int, weights: Mapping[str, float],
             n_resamples: int = 200, webb_below_clusters: int = 10) -> dict[str, Any]:
    """Power of the **test of record** at ``n_items`` per task against a planted ``delta``.

    Pilot items are resampled with replacement to ``n_items`` per family, the effect is planted
    by flipping treatment outcomes, and the resulting per-item contrast is scored with
    ``stats.wild_cluster_test`` at ``alpha``.  The returned ``power`` is the rejection rate; the
    ``n_sims`` behind it is reported beside it, because a test-profile run at ``n_sims = 10``
    has a Monte-Carlo error of about 0.15 and must never be read as a power estimate.
    """
    per_item = frame.per_hypothesis.get(h_id, {})
    fams = frame.families.get(h_id, {})
    if not per_item:
        return {"h_id": h_id, "n": int(n_items), "delta": float(delta), "power": None,
                "n_sims": int(n_sims), "reason": "the pilot frame carries no item for this hypothesis"}
    spec = analysis.HYPOTHESIS_SPECS[h_id]
    arms = tuple(spec.arm_weights) or ("contrast",)
    arm_weights = dict(spec.arm_weights) or {"contrast": 1.0}
    by_family: dict[str, list[str]] = {}
    for item, family in fams.items():
        by_family.setdefault(family, []).append(item)
    rng = np.random.default_rng(int(seed))
    rejections = 0
    usable = 0
    for sim in range(int(n_sims)):
        draw: dict[str, np.ndarray] = {}
        for family, items in by_family.items():
            if not items:
                continue
            picked = [items[i] for i in rng.integers(0, len(items), int(n_items))]
            columns = {arm: np.asarray([float(per_item[i].get(arm, float("nan"))) for i in picked])
                       for arm in arms}
            if any(np.isnan(column).any() for column in columns.values()):
                continue
            treat = [a for a, w in arm_weights.items() if w > 0] or [arms[0]]
            columns[treat[0]] = _plant(columns[treat[0]], float(delta), rng)
            draw[family] = np.asarray([
                math.fsum(float(w) * float(columns[a][j]) for a, w in arm_weights.items())
                for j in range(int(n_items))])
        if not draw:
            continue
        usable += 1
        result = stats.wild_cluster_test(
            draw, weights=weights, n_resamples=n_resamples, seed=int(seed) + sim,
            webb_below_clusters=webb_below_clusters, label=f"{h_id}:power")
        if result.get("p") is not None and float(result["p"]) <= float(alpha):
            rejections += 1
    return {"h_id": h_id, "n": int(n_items), "delta": float(delta),
            "power": (rejections / usable) if usable else None, "n_sims": int(n_sims),
            "n_usable_sims": usable, "alpha": float(alpha), "seed": int(seed),
            "n_resamples": n_resamples,
            "monte_carlo_se": (math.sqrt(0.25 / usable) if usable else None)}


def power_report(pilot_root: str | os.PathLike, *, config_path: str | os.PathLike | None = None,
                 profile: str | None = None, deltas: Sequence[float] = DEFAULT_DELTAS,
                 ) -> dict[str, Any]:
    """``power_report.json``: power per hypothesis x n x delta, and the smallest sufficient n.

    ``n`` values, the simulation count and the power target come from ``analysis.yaml``'s
    profile (``power.n_values``, ``power.n_sims``, ``power.target``); the alpha is the
    Holm-corrected ``familywise_alpha / n_primary`` computed here.
    """
    ctx = analysis.load_context(pilot_root, config_path=config_path, profile=profile)
    power_block = ctx.analysis.profile.power
    n_values = tuple(int(n) for n in power_block.n_values)
    n_sims = int(power_block.n_sims)
    target = float(power_block.target)
    declared_target = float(ctx.cfg.analysis.power_target)
    if abs(target - declared_target) > 1e-9:
        raise ContractViolation(
            f"analysis.profiles.*.power.target {target} != study.analysis.power_target "
            f"{declared_target}: the power target is one fact (R1)")
    alpha = ctx.settings.alpha / max(1, ctx.settings.n_primary)
    frame = pilot_covariance(pilot_root, config_path=config_path, profile=profile)
    weights = ctx.family_weights
    rows: list[dict[str, Any]] = []
    smallest: dict[str, dict[str, Any]] = {}
    for h_id in tuple(ctx.cfg.analysis.primary_hypotheses):
        for delta in deltas:
            for n in n_values:
                seed = stats.analysis_seed(ctx.cfg.seeds.study_seed_hex, f"power:{h_id}:{n}:{delta}")
                row = simulate(frame, h_id, n, float(delta), n_sims=n_sims, alpha=alpha,
                               seed=seed, weights=weights,
                               n_resamples=min(ctx.settings.n_resamples, 1000),
                               webb_below_clusters=ctx.settings.webb_below_clusters)
                rows.append(row)
            reached = [r for r in rows
                       if r["h_id"] == h_id and r["delta"] == float(delta)
                       and r["power"] is not None and r["power"] >= target]
            smallest[f"{h_id}@{delta}"] = {
                "h_id": h_id, "delta": float(delta),
                "smallest_n": min((r["n"] for r in reached), default=None),
                "screened_items_required": _screened(frame, min((r["n"] for r in reached), default=None)),
            }
    return {
        "alpha": alpha, "familywise_alpha": ctx.settings.alpha, "n_primary": ctx.settings.n_primary,
        "target": target, "n_values": list(n_values), "n_sims": n_sims, "deltas": list(deltas),
        "rows": rows, "smallest_n": smallest, "eligibility": dict(frame.eligibility),
        "missingness": dict(frame.missingness),
        "variance_uncertainty": (
            "Every power number is a Monte-Carlo estimate from the pilot's own item variance; "
            f"with n_sims={n_sims} its standard error is about {math.sqrt(0.25 / n_sims):.3f}, and "
            "the pilot variance itself is estimated from few items.  Read the smallest sufficient "
            "n as a planning figure with that uncertainty, never as a threshold."),
        "cost_profile_present": bool(frame.cost),
    }


def _screened(frame: PilotFrame, n: int | None) -> dict[str, float | None] | None:
    """``n / eligibility_rate`` per family: what has to be *screened* to analyse ``n``."""
    if n is None:
        return None
    return {family: (n / rate) if rate else None for family, rate in frame.eligibility.items()}


def write_power_lock(pilot_root: str | os.PathLike, report: Mapping[str, Any]) -> dict[str, Any]:
    """The sample-size lock a human copies into ``study.analysis.confirmatory_sample_size_lock``."""
    lock = {
        "target": report["target"], "alpha": report["alpha"],
        "per_hypothesis": {key: value["smallest_n"] for key, value in report["smallest_n"].items()},
        "note": ("Written by `nextdyn power`; the human sets confirmatory_sample_size_lock in "
                 "study.yaml at S3.  A null smallest_n means no simulated n reached the target."),
    }
    path = Path(pilot_root) / POWER_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def run(pilot_root: str | os.PathLike, *, config_path: str | os.PathLike | None = None,
        profile: str | None = None) -> cli.Report:
    report = power_report(pilot_root, config_path=config_path, profile=profile)
    path = Path(pilot_root) / POWER_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    write_power_lock(pilot_root, report)
    return cli.Report(completed=len(report["rows"]), detail=str(path))


def main(argv: Sequence[str] | None = None) -> cli.Report:
    parser = cli.common_parser("power")
    parser.add_argument("--pilot-root", default=None, help="the pilot sub-run-root ($RUN/pilot)")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.pilot_root) if args.pilot_root else cli.resolve_run_root(args) / "pilot"
    if not root.exists():
        return cli.Report(blocked=1, detail=f"{root} does not exist; run the pilot first")
    return run(root, config_path=args.config, profile=args.profile)


__all__ = [
    "DEFAULT_DELTAS", "POWER_LOCK", "POWER_REPORT", "PilotFrame", "main", "pilot_covariance",
    "power_report", "run", "simulate", "write_power_lock",
]
