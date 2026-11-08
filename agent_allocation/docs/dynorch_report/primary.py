"""dynorch_v1, WP-C: the pre-registered primary family (plan section 7; config ``statistics`` block).

``dynorch_primary_analysis`` evaluates the four frozen contrasts ``H1/16 - deep/16``, ``H1/16 - sample/16``,
``H3/16 - deep/16``, ``H3/16 - sample/16`` (read from ``statistics.primary_family``) with
``analysis/contrasts.evaluate_contrast(n_primary=4)`` on ``items = main_96`` (one ``build_population`` per contrast
inside), applies Holm over exactly four via ``_vendor/ard_stats.holm(n_primary=4)`` when all four macro rows are
complete, and reports the pre-registered anchor ``H0/16 - deep/16`` first, as a descriptive row (no p-value).
P-values are withheld, with a note, when any primary population has an unknown outcome or a stratum under two
items, exactly as ``analysis/contrasts.primary_analysis`` does for the ladder's own family.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd

from agent_allocation._vendor.ard_stats import holm
from agent_allocation.analysis.contrasts import CONTRAST_COLUMNS, LinearContrast, evaluate_contrast

PARTITION = "main"
ANCHOR_ROLE = "descriptive"


def parse_contrast(text: str, *, role: str, partition: str = PARTITION) -> LinearContrast:
    """``"H1/16-deep/16"`` -> ``LinearContrast`` with terms ``(+1 H1/16, -1 deep/16)``. Checkpoint names never
    contain ``-``, so the first ``-`` separates the two names."""
    if text.count("-") != 1:
        raise ValueError(f"primary contrast {text!r} must be '<name>-<name>' with one '-'")
    left, right = text.split("-")
    if not left or not right:
        raise ValueError(f"primary contrast {text!r} has an empty side")
    return LinearContrast(text, ((partition, left, 1.0), (partition, right, -1.0)), "main", role,
                          f"{left} minus {right}, delivered accuracy (macro, paired item bootstrap)")


def primary_contrasts(cfg: Mapping[str, Any]) -> list[LinearContrast]:
    """The frozen family from ``statistics.primary_family`` (must have exactly ``statistics.n_primary`` entries)."""
    st = cfg["statistics"]
    family = list(st["primary_family"])
    if len(family) != int(st["n_primary"]):
        raise ValueError(f"statistics.primary_family has {len(family)} entries, n_primary is {st['n_primary']}")
    if len(set(family)) != len(family):
        raise ValueError("statistics.primary_family has duplicates")
    return [parse_contrast(t, role="primary") for t in family]


def anchor_contrast(cfg: Mapping[str, Any]) -> LinearContrast:
    """``statistics.anchor_reported_first`` (``H0/16-deep/16``), descriptive."""
    return parse_contrast(str(cfg["statistics"]["anchor_reported_first"]), role=ANCHOR_ROLE)


def dynorch_primary_analysis(outcomes: pd.DataFrame, cfg: Mapping[str, Any], *, replicates: int, study_seed: Any,
                             items: Iterable[str], n_primary: int = 4
                             ) -> tuple[pd.DataFrame, dict[str, float | None], list[str]]:
    """Anchor + primary family rows (macro and per benchmark) with Holm-adjusted macro p-values.

    Returns ``(frame, estimates, notes)``: ``frame`` has ``CONTRAST_COLUMNS`` plus ``p_holm``, ``holm_reject``,
    ``holm_family``; ``estimates`` the macro estimates by contrast name; ``notes`` population warnings, mismatches
    and the withheld-p-values note.
    """
    contrasts = primary_contrasts(cfg)
    if len(contrasts) != int(n_primary):
        raise ValueError(f"expected {n_primary} primary contrasts, config has {len(contrasts)}")
    items = set(items)
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for contrast in [anchor_contrast(cfg), *contrasts]:
        found, population = evaluate_contrast(outcomes, contrast, replicates=replicates, study_seed=study_seed,
                                              items=items, n_primary=n_primary)
        if population.mismatches:
            raise ValueError(f"{contrast.name}: population mismatches {population.mismatches}")
        notes.extend(population.warnings())
        rows.extend(found)
    frame = pd.DataFrame(rows)
    for col in CONTRAST_COLUMNS:
        if col not in frame:
            frame[col] = None
    frame["p_holm"] = None
    frame["holm_reject"] = None
    frame["holm_family"] = None
    macro = frame[(frame["level"] == "macro") & (frame["role"] == "primary")].set_index("contrast")
    pvals = {c.name: macro.loc[c.name, "p_value"] for c in contrasts if c.name in macro.index}
    estimates = {name: (None if pd.isna(macro.loc[name, "estimate"]) else float(macro.loc[name, "estimate"]))
                 for name in macro.index}
    if len(pvals) == n_primary and all(v is not None and not pd.isna(v) for v in pvals.values()):
        adjusted = holm({k: float(v) for k, v in pvals.items()}, n_primary=n_primary)
        family = ",".join(c.name for c in contrasts)
        for name, res in adjusted.items():
            mask = (frame["contrast"] == name) & (frame["level"] == "macro")
            frame.loc[mask, "p_holm"] = res["p_holm"]
            frame.loc[mask, "holm_reject"] = res["reject"]
            frame.loc[mask, "holm_family"] = family
    else:
        notes.append("Primary p-values withheld: every primary population must be fully known (no pending or "
                     "unknown outcomes, >= 2 items per stratum) before p-values and Holm are reported.")
    return frame, estimates, notes


__all__ = ["anchor_contrast", "dynorch_primary_analysis", "parse_contrast", "primary_contrasts"]
