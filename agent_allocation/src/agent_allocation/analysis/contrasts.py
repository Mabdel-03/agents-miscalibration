"""The two frozen primary contrasts (MASTER_SPEC §12), the descriptive CP decomposition (§5.6), exploratory
CP-versus-matched-team rows, and high-A versus low-A comparisons on shared item ids only.

Primary family (A = main allowance, Holm over exactly these two when p-values are reported):

* ``primary_i_peer_vs_private_team_n4_r4`` — main panel: ``Y[peer/4/4] - Y[private_team/4/4]``.
* ``primary_ii_cp_minus_private_team_gain_h3`` — extension panel:
  ``(Y[cp/3] - Y[cp/0]) - (Y[private_team/4/4] - Y[private_team/4/1])``.

Per-item contrasts are formed first (one scalar per source item), then the equal-benchmark macro with equal BBH
subtask weights. P-values and Holm are reported only when every item of both primary populations is known and
every stratum has at least two items; otherwise estimates are labeled partial with fixed-population bounds and
the p-values are withheld. Everything outside the primary family is descriptive/exploratory.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from agent_allocation._vendor.ard_stats import StatsError, holm, paired_contrast
from agent_allocation.analysis.names import CheckpointNameError, parse_name
from agent_allocation.analysis.stats import Population, build_population, value_matrix
from agent_allocation.analysis.summary import benchmark_of_stratum, stratum_weights
from agent_allocation.contracts import BENCHMARKS
from agent_allocation.protocols.grid import CP_PEER_ROUND, CP_TEAM_SIZE

CONTRAST_COLUMNS = ("contrast", "role", "panel", "description", "terms", "note", "level", "benchmark", "n_items",
                    "n_complete", "n_by_stratum", "bound_lower", "bound_upper", "status", "estimate", "se", "ci95_lo",
                    "ci95_hi", "ci95_percentile_lo", "ci95_percentile_hi", "ci_basis", "p_value", "n_resamples",
                    "reason")
PRIMARY_I = "primary_i_peer_vs_private_team_n4_r4"
PRIMARY_II = "primary_ii_cp_minus_private_team_gain_h3"
N_PRIMARY = 2
MAIN_PARTITION = "main"
HIGH_PARTITION = "high"
CP0_POOL = f"peer/{CP_TEAM_SIZE}/{CP_PEER_ROUND}"      # cp/0 IS this pool (protocols/grid.py)
DEGENERATE_ZERO_NOTE = ("zero-width interval: every per-item difference is exactly 0, so the bootstrap is "
                        "degenerate (se=0); read it as no item ever flipping, never as a precisely estimated null")


@dataclass(frozen=True)
class LinearContrast:
    name: str
    terms: tuple[tuple[str, str, float], ...]      # (partition, checkpoint name, coefficient)
    panel: str                                      # main | extension | shared_high_main
    role: str                                       # primary | descriptive | exploratory
    description: str


def primary_contrasts(partition: str = MAIN_PARTITION) -> list[LinearContrast]:
    p = partition
    return [
        LinearContrast(PRIMARY_I, ((p, "peer/4/4", 1.0), (p, "private_team/4/4", -1.0)), "main", "primary",
                       "PEER n=4 r=4 minus PRIVATE_TEAM n=4 r=4, delivered accuracy"),
        LinearContrast(PRIMARY_II, ((p, "cp/3", 1.0), (p, "cp/0", -1.0), (p, "private_team/4/4", -1.0),
                                    (p, "private_team/4/1", 1.0)), "extension", "primary",
                       "(CP h=3 - CP h=0) - (PRIVATE_TEAM r=4 - PRIVATE_TEAM r=1), n=4"),
    ]


def decomposition_contrasts(partition: str = MAIN_PARTITION) -> list[LinearContrast]:
    p = partition
    rows = [LinearContrast("L", ((p, "cp/0", 1.0), (p, "private_team/4/1", -1.0)), "extension", "descriptive",
                           "L = mean(Y_CP,h=0 - Y_PRIVATE_TEAM,r=1)")]
    for h in (1, 2, 3):
        rows.append(LinearContrast(f"H({h})", ((p, f"cp/{h}", 1.0), (p, "cp/0", -1.0),
                                               (p, f"private_team/4/{1 + h}", -1.0), (p, "private_team/4/1", 1.0)),
                                   "extension", "descriptive",
                                   f"H({h}) = mean[(Y_CP,{h} - Y_CP,0) - (Y_PT,{1 + h} - Y_PT,1)]"))
    for h in (0, 1, 2, 3):
        rows.append(LinearContrast(f"cp/{h} - private_team/4/{1 + h}", ((p, f"cp/{h}", 1.0),
                                                                         (p, f"private_team/4/{1 + h}", -1.0)),
                                   "extension", "exploratory", "matched nominal generations 4(2+h)"))
        rows.append(LinearContrast(f"cp/{h} - peer/4/{1 + h}", ((p, f"cp/{h}", 1.0), (p, f"peer/4/{1 + h}", -1.0)),
                                   "extension", "exploratory", "matched nominal generations 4(2+h)"))
    return rows


def _alias_cp0(outcomes: pd.DataFrame, contrast: LinearContrast) -> tuple[LinearContrast, str | None]:
    """``cp/0`` is the PEER n=4 r=1 pool; use ``peer/4/1`` when the manifest has no ``cp/0`` checkpoint."""
    terms = []
    note = None
    for part, name, coef in contrast.terms:
        if name == "cp/0" and not ((outcomes["partition_id"] == part) & (outcomes["name"] == "cp/0")).any():
            name, note = "peer/4/1", "cp/0 checkpoint absent from the manifest; its pool peer/4/1 used"
        terms.append((part, name, coef))
    return LinearContrast(contrast.name, tuple(terms), contrast.panel, contrast.role, contrast.description), note


def identity_note(contrast: LinearContrast) -> str | None:
    """A contrast whose terms cancel is identically zero by construction, not an empirical null.

    ``cp/0`` and ``peer/4/1`` name the same pool (``protocols/grid.py``: ``cp[0] = peer[CP_TEAM_SIZE,
    CP_PEER_ROUND]``), so they are the same term here even when both checkpoints exist in the manifest.
    """
    residual: dict[tuple[str, str], float] = {}
    for part, name, coef in contrast.terms:
        key = (part, CP0_POOL if name == "cp/0" else name)
        residual[key] = residual.get(key, 0.0) + coef
    if any(abs(c) > 1e-12 for c in residual.values()):
        return None
    return (f"{contrast.name} is identically 0 by construction: its terms name the same checkpoint pool "
            f"(cp/0 is the {CP0_POOL} pool), so this row is a tautology, not a finding")


def extension_items(outcomes: pd.DataFrame, partition: str, item_panels: Mapping[str, Sequence[str]]) -> set[str] | None:
    """Items declared in the extension panel (waves.json); ``None`` when no wave declares it."""
    declared = {i for i, panels in item_panels.items() if "extension" in panels}
    return declared or None


def evaluate_contrast(outcomes: pd.DataFrame, contrast: LinearContrast, *, replicates: int, study_seed: Any,
                      items: set[str] | None = None, n_primary: int = N_PRIMARY) -> tuple[list[dict[str, Any]], Population]:
    contrast, note = _alias_cp0(outcomes, contrast)
    note = "; ".join(n for n in (note, identity_note(contrast)) if n) or None
    columns = [(p, n) for p, n, _ in contrast.terms]
    population = build_population(outcomes, columns, label=contrast.name, items=items)
    coefs = np.array([c for _, _, c in contrast.terms])
    matrix = value_matrix(outcomes, population, columns)
    rows: list[dict[str, Any]] = []
    base = {"contrast": contrast.name, "role": contrast.role, "panel": contrast.panel,
            "description": contrast.description, "terms": " ".join(f"{c:+g}*{p}:{n}" for p, n, c in contrast.terms),
            "note": note}
    strata = population.strata
    if not strata:
        rows.append({**base, "level": "macro", "benchmark": "macro", "n_items": 0, "n_complete": 0,
                     "status": "not_estimable", "estimate": None, "reason": "no items planned for every term"})
        return rows, population
    diff = {s: matrix[s] @ coefs for s in strata}                  # NaN if any term unknown
    lower = {s: _filled(matrix[s], coefs, "lower") @ coefs for s in strata}
    upper = {s: _filled(matrix[s], coefs, "upper") @ coefs for s in strata}
    complete = all(not np.isnan(diff[s]).any() for s in strata)
    rs = population.resampler(replicates, study_seed)
    levels = [("macro", stratum_weights(strata))] + [
        (b, stratum_weights(strata, within=b)) for b in sorted({benchmark_of_stratum(s) for s in strata},
                                                               key=BENCHMARKS.index)]
    for level, weights in levels:
        ds = list(weights)
        n_items = sum(len(diff[s]) for s in ds)
        n_complete = sum(int((~np.isnan(diff[s])).sum()) for s in ds)
        row = {**base, "level": "macro" if level == "macro" else "benchmark", "benchmark": level,
               "n_items": n_items, "n_complete": n_complete,
               "n_by_stratum": ";".join(f"{s}={len(diff[s])}" for s in ds),
               "bound_lower": _weighted_mean(lower, weights), "bound_upper": _weighted_mean(upper, weights)}
        level_complete = all(not np.isnan(diff[s]).any() for s in ds)
        small = [s for s in ds if len(diff[s]) < 2]
        if level_complete and not small:
            res = paired_contrast({s: diff[s] for s in ds}, rs, domains=ds, weights=weights, n_primary=n_primary)
            if res["se"] == 0 and not identity_note(contrast):
                row["note"] = "; ".join(n for n in (note, DEGENERATE_ZERO_NOTE) if n)
            row.update({"status": "complete", "estimate": res["estimate"], "se": res["se"],
                        "ci95_lo": res["ci95_studentized"][0], "ci95_hi": res["ci95_studentized"][1],
                        "ci95_percentile_lo": res["ci95_percentile"][0], "ci95_percentile_hi": res["ci95_percentile"][1],
                        "ci_basis": "studentized_item_cluster_bootstrap",
                        "p_value": res["p"] if (contrast.role == "primary" and level == "macro") else None,
                        "n_resamples": res["n_resamples"]})
        else:
            est = _weighted_mean(diff, weights, nan_ok=True)
            lo, hi, valid = _nan_percentile(diff, weights, rs)
            reason = []
            if not level_complete:
                reason.append(f"{n_items - n_complete} item(s) with an unknown term")
            if small:
                reason.append(f"strata with <2 items: {', '.join(small)}")
            row.update({"status": "partial" if not level_complete else "not_estimable_small_strata",
                        "estimate": est, "ci95_percentile_lo": lo, "ci95_percentile_hi": hi,
                        "ci_basis": "known_only_partial_percentile", "p_value": None, "n_resamples": valid,
                        "reason": "; ".join(reason)})
        rows.append(row)
    del complete
    return rows, population


def _filled(Y: np.ndarray, coefs: np.ndarray, bound: str) -> np.ndarray:
    """Fixed-population bound: unknown cells set to the value that minimises (lower) or maximises (upper)."""
    out = Y.copy()
    for j, c in enumerate(coefs):
        fill = (1.0 if c < 0 else 0.0) if bound == "lower" else (1.0 if c > 0 else 0.0)
        col = out[:, j]
        col[np.isnan(col)] = fill
    return out


def _weighted_mean(values: Mapping[str, np.ndarray], weights: Mapping[str, float], nan_ok: bool = False) -> float | None:
    total = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for s, w in weights.items():
            v = values[s]
            m = float(np.nanmean(v)) if nan_ok else float(np.mean(v))
            if math.isnan(m):
                return None
            total += w * m
    return total


def _nan_percentile(diff: Mapping[str, np.ndarray], weights: Mapping[str, float], rs, alpha: float = 0.05):
    B = rs.n_resamples
    theta = np.zeros(B)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for s, w in weights.items():
            if len(diff[s]) == 0:
                return None, None, 0
            theta += w * np.nanmean(diff[s][rs.idx[s]], axis=1)
    valid = int((~np.isnan(theta)).sum())
    if valid == 0:
        return None, None, 0
    lo, hi = np.nanquantile(theta, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), valid


def primary_analysis(outcomes: pd.DataFrame, *, replicates: int, study_seed: Any,
                     item_panels: Mapping[str, Sequence[str]], partition: str = MAIN_PARTITION
                     ) -> tuple[pd.DataFrame, dict[str, float | None], list[str]]:
    """Primary contrasts (+Holm when both complete), CP decomposition and exploratory CP rows."""
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    ext = extension_items(outcomes, partition, item_panels)
    for contrast in primary_contrasts(partition) + decomposition_contrasts(partition):
        items = ext if contrast.panel == "extension" else None
        if contrast.panel == "extension" and ext is None:
            notes.append(f"{contrast.name}: no wave declares the extension panel; population = items planned for "
                         f"every term")
        found, population = evaluate_contrast(outcomes, contrast, replicates=replicates, study_seed=study_seed,
                                              items=items)
        notes.extend(population.warnings())
        notes.extend(population.mismatches)
        rows.extend(found)
    frame = pd.DataFrame(rows)
    for col in CONTRAST_COLUMNS:
        if not frame.empty and col not in frame:
            frame[col] = None
    if not frame.empty:                       # row notes (tautologies, degenerate zero-width CIs) reach the report
        for text in sorted({n for n in frame["note"].dropna() if n}):
            where = [f"{c}:{b}" for c, b in frame.loc[frame["note"] == text, ["contrast", "benchmark"]]
                     .itertuples(index=False)]
            shown = ", ".join(where[:6]) + (f", +{len(where) - 6} more" if len(where) > 6 else "")
            notes.append(f"{text} [{shown}]")
    estimates: dict[str, float | None] = {}
    if not frame.empty:
        macro = frame[(frame["level"] == "macro") & (frame["role"] == "primary")].set_index("contrast")
        for name in (PRIMARY_I, PRIMARY_II):
            estimates[name] = (float(macro.loc[name, "estimate"])
                               if name in macro.index and macro.loc[name, "estimate"] is not None
                               and not pd.isna(macro.loc[name, "estimate"]) else None)
        pvals = {name: macro.loc[name, "p_value"] for name in (PRIMARY_I, PRIMARY_II) if name in macro.index}
        frame["p_holm"] = None
        frame["holm_family"] = None
        if len(pvals) == N_PRIMARY and all(v is not None and not pd.isna(v) for v in pvals.values()):
            adjusted = holm({k: float(v) for k, v in pvals.items()}, n_primary=N_PRIMARY)
            for name, res in adjusted.items():
                mask = (frame["contrast"] == name) & (frame["level"] == "macro")
                frame.loc[mask, "p_holm"] = res["p_holm"]
                frame.loc[mask, "holm_family"] = f"{PRIMARY_I},{PRIMARY_II}"
        else:
            notes.append("Primary p-values withheld: both primary populations must be fully known (no unknown or "
                         "unrun outcomes, >=2 items per stratum) before p-values and Holm are reported.")
    return frame, estimates, notes


def shared_panel_comparisons(outcomes: pd.DataFrame, *, replicates: int, study_seed: Any
                             ) -> tuple[pd.DataFrame, list[str]]:
    """Core panel variants against the main panel on shared item ids only: ``high`` (A=4,096) and
    ``replication`` (second model, reduced grid). Exploratory; never an unpaired 96-vs-400 aggregate."""
    frames, notes = [], []
    present = set(outcomes["partition_id"]) if not outcomes.empty else set()
    for variant in ("high", "replication"):
        if variant in present and MAIN_PARTITION in present:
            frame, variant_notes = high_low_shared(outcomes, replicates=replicates, study_seed=study_seed,
                                                   high=variant, low=MAIN_PARTITION)
            if not frame.empty:
                frame.insert(0, "variant", variant)
                frames.append(frame)
            notes.extend(variant_notes)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), notes


def high_low_shared(outcomes: pd.DataFrame, *, replicates: int, study_seed: Any,
                    high: str = HIGH_PARTITION, low: str = MAIN_PARTITION) -> tuple[pd.DataFrame, list[str]]:
    """``high`` minus ``low`` for every checkpoint name planned in both partitions, on shared item ids only."""
    names_high = set(outcomes.loc[(outcomes["partition_id"] == high) & (outcomes["view"] == "current"), "name"])
    names_low = set(outcomes.loc[(outcomes["partition_id"] == low) & (outcomes["view"] == "current"), "name"])
    shared_names = sorted(names_high & names_low, key=_name_key)
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    if not shared_names:
        return pd.DataFrame(rows), notes
    high_items = set(outcomes.loc[outcomes["partition_id"] == high, "item_id"])
    low_items = set(outcomes.loc[outcomes["partition_id"] == low, "item_id"])
    shared_items = high_items & low_items
    notes.append(f"high-vs-{low}: {len(shared_items)} shared item ids ({len(high_items)} high, {len(low_items)} "
                 f"{low}); comparisons use shared ids only")
    for name in shared_names:
        contrast = LinearContrast(f"{name}: {high} - {low}", ((high, name, 1.0), (low, name, -1.0)),
                                  f"shared_{high}_{low}", "exploratory", f"{high} panel minus {low} panel, shared ids")
        found, population = evaluate_contrast(outcomes, contrast, replicates=replicates, study_seed=study_seed,
                                              items=shared_items)
        rows.extend(found)
    frame = pd.DataFrame(rows)
    for col in CONTRAST_COLUMNS:
        if not frame.empty and col not in frame:
            frame[col] = None
    return frame, notes


def _name_key(name: str):
    try:
        return (0, parse_name(name).sort_key)
    except CheckpointNameError:
        return (1, (name,))


__all__ = ["PRIMARY_I", "PRIMARY_II", "LinearContrast", "evaluate_contrast", "high_low_shared", "primary_analysis",
           "shared_panel_comparisons",
           "primary_contrasts", "decomposition_contrasts", "StatsError"]
