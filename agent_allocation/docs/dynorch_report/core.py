"""dynorch_v1, WP-C: the merged ladder + harness bootstrap frame (contract section 7; plan section 7).

``Frame`` subclasses ``docs/forced_v2_report/core.py::Frame`` (loaded from its file, so the sealed report's loading
and weighting code is reused unchanged) with ``families = LADDER_FAMILIES + ["harness"]`` and ``SEED = 20260923``.
It differs from the sealed frame in one pre-registered way: there is ONE ``analysis/stats.build_population`` call
PER CONTRAST (its columns, ``items = main_96``, ``mismatches == []`` asserted), so every interval is a paired
item-clustered bootstrap over exactly the items planned for the columns it compares. Point estimates are the
sealed ones (the regression test checks deep/16 = 0.7917 and sample/16 = 0.4167); the sealed report's CIs were
drawn from one joint population with a different ``seed_label``, so ladder CIs differ slightly from
``docs/forced_v2/report`` and the report says so.

Nothing here reads anything but the sealed ``forced_budget_v2/artifacts/merged/tables`` (read-only) and the
dynorch run's ``tables/`` (``dynorch/tables.py``); the harness rows are passed in as ``extra_outcomes``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from agent_allocation.analysis import stats as astats
from agent_allocation.metrics.passk import pass_at_k

HERE = Path(__file__).resolve().parent


def _load_forced_core():
    """``docs/forced_v2_report/core.py`` as a module (it is a script-style module, not a package)."""
    name = "forced_v2_report_core"
    if name in sys.modules:
        return sys.modules[name]
    path = HERE.parent / "forced_v2_report" / "core.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


fc = _load_forced_core()

BENCH_ORDER = list(fc.BENCH_ORDER)
BENCH_LABEL = dict(fc.BENCH_LABEL)
BENCH_PLAIN = dict(fc.BENCH_PLAIN)
FAMILY_LABEL = dict(fc.FAMILY_LABEL)
FAMILY_COLOR = dict(fc.FAMILY_COLOR)
ORACLE_COLOR = fc.ORACLE_COLOR
LADDER_FAMILIES = list(fc.LADDER_FAMILIES)
HARNESS_FAMILY = "harness"
FAMILIES = LADDER_FAMILIES + [HARNESS_FAMILY]
A0 = fc.A0
SEED = 20260923
REPLICATES = 10_000
LEVELS = ["macro", *BENCH_ORDER]
CORE_ARMS = ["H1", "H3", "H0"]
ARM_LABEL = {"H1": "H1 OpenClaude", "H3": "H3 Codex", "H0": "H0 harness-free control", "H1c": "H1 capped",
             "H3c": "H3 capped", "H1n": "H1 no spawn", "H3n": "H3 no spawn", "H2": "H2 OpenClaw"}
# Colour-blind-safe (Okabe-Ito) hues not used by the ladder families in the figures, with marker shape as the
# secondary encoding; H0 is drawn hollow so it never competes with the oracle's black dashed line.
ARM_COLOR = {"H1": "#E69F00", "H3": "#56B4E9", "H0": "#000000", "H1c": "#E69F00", "H3c": "#56B4E9",
             "H1n": "#E69F00", "H3n": "#56B4E9", "H2": "#F0E442"}
ARM_MARKER = {"H1": "s", "H3": "^", "H0": "D", "H1c": "s", "H3c": "^", "H1n": "s", "H3n": "^", "H2": "v"}
BANK_SIZE = 25


def main_items(panels_path: str | Path) -> list[str]:
    """``main_item_ids`` of every benchmark in ``manifests/panels.json`` (the fixed population, ``main_96``)."""
    panels = json.loads(Path(panels_path).read_text())
    items: list[str] = []
    for bench in BENCH_ORDER:
        if bench not in panels:
            raise KeyError(f"panels.json has no benchmark {bench!r}")
        items.extend(panels[bench]["main_item_ids"])
    return items


def derive_c(po: pd.DataFrame) -> pd.DataFrame:
    """``B, R, A, C, rung`` from the sealed columns (``core.py``'s ``C = B*R*A``); harness rows satisfy ``C == cap``."""
    po = po.copy()
    po["B"] = po["team_size"].astype(int)
    po["R"] = po["total_cycles"].astype(int)
    po["A"] = po["reasoning_budget"].astype(int)
    po["C"] = po["B"] * po["R"] * po["A"]
    po["rung"] = po["C"] // A0
    return po


def harness_outcomes(frame: pd.DataFrame, *, partition: str = "main") -> pd.DataFrame:
    """``policy_outcomes_dynorch`` rows of one partition, current view, with the derived compute columns."""
    sub = frame[(frame["partition_id"] == partition) & (frame["view"] == "current")]
    if (sub["family"] != HARNESS_FAMILY).any():
        raise ValueError("extra_outcomes must carry family 'harness' only")
    return derive_c(sub)


def arm_of(name: str) -> str:
    return str(name).split("/")[0]


def rung_of(name: str) -> int:
    return int(str(name).split("/")[1])


def repeat_of(name: str) -> int:
    parts = str(name).split("/")
    return int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("r") else 1


class Frame(fc.Frame):
    """Ladder + harness checkpoints on one fixed item population; one ``build_population`` per contrast.

    ``base_merged_dir`` is the sealed ``forced_budget_v2/artifacts/merged``; ``extra_outcomes`` the harness rows
    (``dynorch/tables.merged_outcomes`` output is also accepted: ladder rows are re-read from the sealed dir anyway);
    ``items`` the pre-declared population (``main_items(panels.json)``), default: every main item of the ladder.
    """

    def __init__(self, base_merged_dir: str | Path, extra_outcomes: pd.DataFrame | None = None,
                 families: Sequence[str] | None = None, *, label: str = "dynorch", items: Iterable[str] | None = None,
                 replicates: int = REPLICATES, seed: int = SEED, partition: str = "main"):
        families = list(FAMILIES if families is None else families)
        ladder = fc.outcomes(str(base_merged_dir), families=[f for f in families if f != HARNESS_FAMILY])
        parts = [ladder]
        if HARNESS_FAMILY in families:
            if extra_outcomes is None:
                raise ValueError("families include 'harness' but no extra_outcomes were given")
            extra = extra_outcomes[extra_outcomes["family"] == HARNESS_FAMILY]
            parts.append(harness_outcomes(extra, partition=partition))
        self.oc = pd.concat(parts, ignore_index=True)
        self.families = families
        self.partition = partition
        self.idx = fc.checkpoint_index(self.oc)
        self.names = list(self.idx["name"])
        self.col = {n: j for j, n in enumerate(self.names)}
        self.items = sorted(items) if items is not None else sorted(ladder["item_id"].unique())
        present = set(self.oc["item_id"])
        absent = [i for i in self.items if i not in present]
        if absent:
            raise ValueError(f"{len(absent)} population items have no outcome row (e.g. {absent[0]})")
        strata = sorted({str(s) for s in self.oc[self.oc["item_id"].isin(self.items)]["stratum"]})
        self.levels = fc.levels_of(strata)
        self.n_items = len(self.items)
        self.label = label
        self.replicates = int(replicates)
        self.seed = seed
        self._pops: dict[tuple[tuple[str, ...], frozenset[str] | None], tuple[Any, dict, dict, Any]] = {}

    # ------------------------------------------------------------------ populations

    def has(self, name: str) -> bool:
        return name in self.col

    def population(self, names: Sequence[str], items: Iterable[str] | None = None):
        """ONE ``build_population`` for these columns on ``items`` (default: the frame population); asserts
        ``mismatches == []`` (a planned row exists for every item and column, pending rows included). Returns
        ``(population, matrix, boot, resampler)`` and caches by (names, items)."""
        names = tuple(names)
        for n in names:
            if n not in self.col:
                raise KeyError(f"checkpoint {n!r} is not in the frame")
        subset = None if items is None else frozenset(items)
        key = (names, subset)
        if key not in self._pops:
            cols = [(self.partition, n) for n in names]
            pop = astats.build_population(self.oc, cols, view="current", label=self.label,
                                          items=self.items if subset is None else sorted(subset))
            if pop.mismatches:
                raise ValueError(f"population mismatches for {names}: {pop.mismatches}")
            planned = set(self.items) if subset is None else (subset & set(self.items))
            if set(pop.item_ids) != planned:
                raise ValueError(f"population for {names} covers {pop.n_items} items, expected {len(planned)}")
            mat = astats.value_matrix(self.oc, pop, cols, view="current", column="y")
            rs = pop.resampler(self.replicates, self.seed)
            boot = fc.boot_means(mat, rs)
            self._pops[key] = (pop, mat, boot, rs)
        return self._pops[key]

    @staticmethod
    def _levels(pop) -> dict:
        return fc.levels_of(pop.strata)

    # ------------------------------------------------------------------ estimates

    def acc(self, name: str, level: str = "macro", alpha: float = 0.05, items: Iterable[str] | None = None):
        """Point estimate and percentile CI of one checkpoint (macro = equal benchmarks)."""
        pop, mat, boot, _ = self.population((name,), items)
        w = self._levels(pop)[level]
        b = fc.wboot(boot, w, 0)
        lo, hi = np.nanquantile(b, [alpha / 2, 1 - alpha / 2])
        return fc.wpoint(mat, w, 0), float(lo), float(hi)

    def diff(self, a: str, b: str, level: str = "macro", alpha: float = 0.05, items: Iterable[str] | None = None):
        """Paired ``a - b`` on one population of the two columns: estimate, percentile CI, null-centred p."""
        pop, mat, boot, _ = self.population((a, b), items)
        w = self._levels(pop)[level]
        est = fc.wpoint(mat, w, 0) - fc.wpoint(mat, w, 1)
        d = fc.wboot(boot, w, 0) - fc.wboot(boot, w, 1)
        lo, hi = np.nanquantile(d, [alpha / 2, 1 - alpha / 2])
        p = float(np.mean(np.abs(d - np.nanmean(d)) >= abs(est)))
        return est, float(lo), float(hi), p

    def diff_vs_reference(self, name: str, reference: Mapping[str, float], level: str = "macro",
                          alpha: float = 0.05, items: Iterable[str] | None = None):
        """Paired ``y_i - reference_i`` (e.g. the per-item oracle pass@k) under the checkpoint's own resample
        indices; a NaN reference makes the item unknown for this contrast (NaN-aware means)."""
        pop, mat, _, rs = self.population((name,), items)
        w = self._levels(pop)[level]
        diff = {}
        for s in pop.strata:
            ref = np.array([reference.get(i, np.nan) for i in pop.items[s]], dtype=float)
            diff[s] = mat[s][:, 0] - ref
        est = float(sum(wt * np.nanmean(diff[s]) for s, wt in w.items() if s in diff))
        acc = None
        for s, wt in w.items():
            if s not in diff or diff[s].shape[0] == 0:
                continue
            v = wt * np.nanmean(diff[s][rs.idx[s]], axis=1)
            acc = v if acc is None else acc + v
        lo, hi = np.nanquantile(acc, [alpha / 2, 1 - alpha / 2])
        n_known = int(sum(np.sum(~np.isnan(diff[s])) for s in diff))
        return est, float(lo), float(hi), n_known

    def unknown_share(self, name: str) -> float:
        sub = self.oc[(self.oc["name"] == name) & self.oc["item_id"].isin(self.items)]
        return float(sub["y"].isna().mean())

    def state_counts(self, name: str) -> dict[str, int]:
        sub = self.oc[(self.oc["name"] == name) & self.oc["item_id"].isin(self.items)]
        return {str(k): int(v) for k, v in sub["delivered_state"].value_counts().items()}


def bank_correct_by_item(base_merged_dir: str | Path) -> pd.Series:
    """``bank_correct`` (count of correct draws in the 25-draw sampling bank) per main item from
    ``sample_bank_items``; NaN where the bank has unknown grades."""
    sb = fc.T(str(base_merged_dir), "sample_bank_items")
    sb = sb[sb["partition_id"] == "main"]
    if (sb["bank_size"] != BANK_SIZE).any():
        raise ValueError(f"sample_bank_items bank_size is not {BANK_SIZE} everywhere")
    per = sb.groupby("item_id")["bank_correct"].agg(["min", "max"])
    if ((per["min"] != per["max"]) & per["min"].notna()).any():
        raise ValueError("bank_correct differs across k rows of one item")
    return per["max"]


def oracle_by_item(bank_correct: pd.Series, k: int) -> dict[str, float]:
    """Per-item oracle pass@k paired with the harness cell: ``metrics/passk.pass_at_k(25, c_i, k)``."""
    out: dict[str, float] = {}
    for item, c in bank_correct.items():
        out[str(item)] = float(pass_at_k(BANK_SIZE, int(c), int(k))) if not pd.isna(c) else float("nan")
    return out


def bank_curve(base_merged_dir: str | Path) -> pd.DataFrame:
    return fc.bank_curve(str(base_merged_dir))


def dump(obj: Any, path: str | Path) -> None:
    fc.dump(obj, str(path))


__all__ = ["A0", "ARM_COLOR", "ARM_LABEL", "ARM_MARKER", "BENCH_LABEL", "BENCH_ORDER", "BENCH_PLAIN", "CORE_ARMS",
           "FAMILIES", "FAMILY_COLOR", "FAMILY_LABEL", "Frame", "HARNESS_FAMILY", "LADDER_FAMILIES", "LEVELS",
           "ORACLE_COLOR", "REPLICATES", "SEED", "arm_of", "bank_correct_by_item", "bank_curve", "derive_c", "dump",
           "fc", "harness_outcomes", "main_items", "oracle_by_item", "repeat_of", "rung_of"]
