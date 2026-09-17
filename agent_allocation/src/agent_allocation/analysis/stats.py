"""Item-clustered paired bootstrap over source items within benchmark strata (BBH subtasks are strata).

The cluster is the source item: one resample draws item indices independently inside each stratum
(``_vendor.ard_stats.Resampler``), and *every* checkpoint column of that item moves together, so all methods,
rounds and team sizes for an item stay in the same replicate. Agent messages, candidates or checkpoint rows are
never resampling units.

Weights: equal per benchmark in the macro; equal per BBH subtask inside BBH (``summary.stratum_weights``).
Unknown outcomes (NaN) are handled per resample with a NaN-aware mean; such intervals are labeled
``known_only_partial`` and are never used for confirmatory claims.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from agent_allocation._vendor.ard_stats import Resampler, analysis_seed
from agent_allocation.analysis.summary import benchmark_of_stratum, strata_order, stratum_weights
from agent_allocation.contracts import BENCHMARKS

ALPHA = 0.05
SMALL_BENCHMARK_ITEMS = 20
SMALL_STRATUM_ITEMS = 5
CHUNK = 256


Column = tuple[str, str]   # (partition_id, checkpoint name)


@dataclass
class Population:
    """A fixed item population: the items planned for every requested column, grouped by stratum."""
    label: str
    seed_label: str
    items: dict[str, list[str]]                       # stratum -> sorted item ids
    columns: tuple[Column, ...] = ()
    mismatches: list[str] = field(default_factory=list)

    @property
    def strata(self) -> list[str]:
        return strata_order(s for s, ids in self.items.items() if ids)

    @property
    def n_by_stratum(self) -> dict[str, int]:
        return {s: len(self.items[s]) for s in self.strata}

    @property
    def n_by_benchmark(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s, n in self.n_by_stratum.items():
            out[benchmark_of_stratum(s)] = out.get(benchmark_of_stratum(s), 0) + n
        return out

    @property
    def n_items(self) -> int:
        return sum(self.n_by_stratum.values())

    @property
    def item_ids(self) -> list[str]:
        return [i for s in self.strata for i in self.items[s]]

    def warnings(self) -> list[str]:
        out = []
        for bench, n in self.n_by_benchmark.items():
            if n < SMALL_BENCHMARK_ITEMS:
                out.append(f"{self.label}: small panel for {bench} ({n} independent items); intervals are unstable")
        for stratum, n in self.n_by_stratum.items():
            if "/" in stratum and n < SMALL_STRATUM_ITEMS:
                out.append(f"{self.label}: BBH subtask stratum {stratum.split('/', 1)[1]} has {n} items")
        missing = [b for b in BENCHMARKS if b not in self.n_by_benchmark]
        if missing and self.n_items:
            out.append(f"{self.label}: benchmarks absent from this population: {', '.join(missing)}; the macro "
                       f"covers {len(BENCHMARKS) - len(missing)} of {len(BENCHMARKS)}")
        return out

    def resampler(self, replicates: int, study_seed: Any) -> Resampler:
        return Resampler(self.n_by_stratum, replicates, analysis_seed(str(study_seed), self.seed_label),
                         domains=self.strata)


def _column_frame(outcomes: pd.DataFrame, columns: Sequence[Column], view: str) -> pd.DataFrame:
    keys = {f"{p}\x1f{n}" for p, n in columns}
    frame = outcomes[outcomes["view"] == view]
    tag = frame["partition_id"].astype(str) + "\x1f" + frame["name"].astype(str)
    return frame[tag.isin(keys)].assign(_column=tag[tag.isin(keys)])


def build_population(outcomes: pd.DataFrame, columns: Sequence[Column], *, view: str = "current",
                     label: str, items: Iterable[str] | None = None) -> Population:
    """Items planned for *all* ``columns`` (optionally intersected with ``items``: a pre-declared panel).

    Items planned for only some columns are recorded in ``mismatches`` (a fixed-population violation) and are
    excluded from the joint population rather than kept for a subset of columns.
    """
    frame = _column_frame(outcomes, columns, view)
    per_col = {f"{p}\x1f{n}": set(frame.loc[frame["_column"] == f"{p}\x1f{n}", "item_id"]) for p, n in columns}
    common = set.intersection(*per_col.values()) if per_col else set()
    union = set.union(*per_col.values()) if per_col else set()
    mismatches = []
    if union != common:
        for key, s in per_col.items():
            if s != union and not (items is not None and set(items) <= s):
                mismatches.append(f"{key.replace(chr(31), '/')}/{view}: {len(union - s)} of {len(union)} items not planned")
    if items is not None:
        common &= set(items)
    strata = frame.drop_duplicates("item_id").set_index("item_id")["stratum"].to_dict()
    grouped: dict[str, list[str]] = {}
    for item in sorted(common):
        grouped.setdefault(strata[item], []).append(item)
    seed_label = label + "|" + ",".join(f"{p}:{n}" for p, n in columns)
    return Population(label, seed_label, grouped, tuple(columns), mismatches)


def value_matrix(outcomes: pd.DataFrame, population: Population, columns: Sequence[Column], *,
                 view: str = "current", column: str = "y") -> dict[str, np.ndarray]:
    """``{stratum: array (n_items, len(columns))}`` in the population's item order."""
    frame = _column_frame(outcomes, columns, view)
    pivot = frame.pivot_table(index="item_id", columns="_column", values=column, aggfunc="first", dropna=False)
    keys = [f"{p}\x1f{n}" for p, n in columns]
    out: dict[str, np.ndarray] = {}
    for stratum in population.strata:
        sub = pivot.reindex(index=population.items[stratum], columns=keys)
        out[stratum] = sub.to_numpy(dtype=float)
    return out


def _levels(strata: Sequence[str]) -> dict[str, dict[str, float]]:
    levels = {"macro": stratum_weights(strata)}
    for bench in sorted({benchmark_of_stratum(s) for s in strata}, key=BENCHMARKS.index):
        levels[bench] = stratum_weights(strata, within=bench)
    return levels


def column_intervals(matrix: Mapping[str, np.ndarray], rs: Resampler, names: Sequence[str],
                     *, alpha: float = ALPHA) -> pd.DataFrame:
    """Percentile intervals for the weighted mean of every column at macro and per-benchmark level.

    All columns share the resample indices (items kept together). NaN cells are unknown outcomes: the estimate
    and the per-resample means use the known cells only and the row is labeled ``known_only_partial``.
    """
    strata = [s for s in rs.domains if s in matrix]
    levels = _levels(strata)
    P = len(names)
    B = rs.n_resamples
    acc = {lvl: np.zeros((B, P)) for lvl in levels}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for start in range(0, B, CHUNK):
            sl = slice(start, min(B, start + CHUNK))
            for s in strata:
                Y = matrix[s]
                if Y.shape[0] == 0:
                    continue
                m = np.nanmean(Y[rs.idx[s][sl]], axis=1)          # (b, P)
                for lvl, w in levels.items():
                    if s in w:
                        acc[lvl][sl] += w[s] * m
        rows = []
        for lvl, w in levels.items():
            est = np.zeros(P)
            known = np.zeros(P, dtype=int)
            planned = 0
            for s in w:
                est += w[s] * np.nanmean(matrix[s], axis=0) if matrix[s].shape[0] else 0.0
                known += np.sum(~np.isnan(matrix[s]), axis=0)
                planned += matrix[s].shape[0]
            lo, hi = np.nanquantile(acc[lvl], [alpha / 2, 1 - alpha / 2], axis=0)
            valid = np.sum(~np.isnan(acc[lvl]), axis=0)
            for j, name in enumerate(names):
                complete = bool(known[j] == planned)
                rows.append({"name": name, "level": "macro" if lvl == "macro" else "benchmark", "benchmark": lvl,
                             "n_items": planned, "n_known": int(known[j]),
                             "estimate": float(est[j]) if not math.isnan(est[j]) else None,
                             "ci_lo": float(lo[j]) if not math.isnan(lo[j]) else None,
                             "ci_hi": float(hi[j]) if not math.isnan(hi[j]) else None,
                             "ci_basis": "planned_population" if complete else "known_only_partial",
                             "valid_resamples": int(valid[j]), "n_resamples": B})
    return pd.DataFrame(rows)
