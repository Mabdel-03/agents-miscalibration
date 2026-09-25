"""Model-side inputs (WP-M): configurations, classes, pools and observed pool outcomes.

Reads ONLY ``analysis_inputs/`` frames handed in by the caller (``draws.parquet``, ``pools.parquet``, optionally
``classes.parquet``) and nothing else: no generation-side directories and no legacy run root (seal static scan).

Classes: WP-A may publish ``analysis_inputs/classes.parquet`` with columns ``scope_id, candidate_id, class_id``
(scope = a pool_id, a ``diag5`` pool_id, or ``e2fit:<item_id>``). When it is absent the fallback builds classes by
exact vote key (``vote_exact_key``; GPQA = option identity) -- gold-free and deterministic, but without the
Math-Verify merges; this is recorded in every model manifest as ``classes_source``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PASSK_SEED = 20260925
A = 1024
BENCHMARKS = ("aime", "hmmt", "omni_math_hard", "gpqa_diamond")
MIXTURE = {"aime": 0.10, "hmmt": 0.10, "omni_math_hard": 0.50, "gpqa_diamond": 0.30}

CONFIGS_16A = ("16x1", "8x2", "4x4", "2x8", "1x16")
CONFIGS_8A = ("8x1", "4x2", "2x4", "1x8")
PREDICT_CONFIGS = CONFIGS_16A + CONFIGS_8A
DEPTHS = (1, 2, 4, 8, 16)
CROSSOVER_K = (2, 4, 8, 16)


def parse_config(config_id: str) -> tuple[int, int]:
    n, a = config_id.split("x")
    return int(n), int(a)


def config_id(n: int, a: int) -> str:
    return f"{int(n)}x{int(a)}"


def vote_class_key(row) -> str | None:
    """Gold-free exact class key of a draw (None = casts no vote)."""
    key = row.get("option") if isinstance(row, dict) else getattr(row, "option", None)
    if key is None or (isinstance(key, float) and key != key):
        key = row.get("vote_exact_key") if isinstance(row, dict) else getattr(row, "vote_exact_key", None)
    if key is None or (isinstance(key, float) and key != key):
        return None
    return str(key)


def label_value(v) -> float:
    """Tri-state label -> 1.0 / 0.0 / nan."""
    if v is None:
        return np.nan
    try:
        if pd.isna(v):
            return np.nan
    except (TypeError, ValueError):
        pass
    return 1.0 if bool(v) else 0.0


# ---------------------------------------------------------------------- fitting draws / pools
def fitting_draws(draws: pd.DataFrame, items: Iterable[str], *, a_slots: int = 16,
                  partitions: tuple[str, ...] = ("legacy_main", "e1_fill", "c48_extra")) -> pd.DataFrame:
    """E2/E5 development draws of ``items``: A slots 0..a_slots-1 plus every deep replicate at 2..16A.

    Never returns a row from a fenced bank (``v48_bank``/``t200_bank``) or ``t200_diag``.
    """
    items = set(items)
    d = draws[draws.item_id.isin(items) & draws.partition.isin(partitions)]
    shallow = (d.allowance_a == 1) & (d.replicate < a_slots)
    deep = d.allowance_a.isin([2, 4, 8, 16])
    return d[shallow | deep].copy()


def prefix_members(draws_item: pd.DataFrame, n: int, a: int) -> list[str] | None:
    """Prefix rule (contract §5.4): replicates r0..r(n-1) at allowance a; None when any is missing."""
    d = draws_item[draws_item.allowance_a == a].sort_values("replicate")
    d = d[d.replicate < n]
    if len(d) < n or d.replicate.nunique() < n:
        return None
    return list(d.candidate_id)


def build_prefix_pools(draws: pd.DataFrame, items: Iterable[str], configs: Iterable[str], panel: str) -> pd.DataFrame:
    """Local pool table (same columns as WP-A ``pools.parquet``) for dev use and tests."""
    rows = []
    by_item = {k: g for k, g in draws.groupby("item_id")}
    for item in items:
        g = by_item.get(item)
        if g is None:
            continue
        for cid in configs:
            n, a = parse_config(cid)
            mem = prefix_members(g, n, a)
            if mem is None:
                continue
            pid = hashlib.sha256(f"{item}|{cid}|current|{'|'.join(mem)}".encode()).hexdigest()
            rows.append(dict(pool_id=pid, item_id=item, panel=panel, config_id=cid, n=n, a=a, R=1, B=n * a,
                             member_ids=mem, view="current"))
    return pd.DataFrame(rows, columns=["pool_id", "item_id", "panel", "config_id", "n", "a", "R", "B",
                                       "member_ids", "view"])


def exact_classes(draws: pd.DataFrame, scope_id: str, member_ids: list[str]) -> dict[str, str | None]:
    """candidate_id -> class id within one scope (exact vote key; None = no vote)."""
    sub = draws.set_index("candidate_id").loc[member_ids]
    out = {}
    for cid, row in zip(member_ids, sub.to_dict("records")):
        k = vote_class_key(row)
        out[cid] = None if k is None else f"{scope_id}:{k}"
    return out


@dataclass
class ClassSource:
    """Scope -> {candidate_id: class_id|None}; WP-A classes when given, else the exact-key fallback."""

    draws: pd.DataFrame
    table: pd.DataFrame | None = None
    n_table: int = 0
    n_fallback: int = 0

    @property
    def source(self) -> str:
        return "analysis_inputs/classes.parquet" if self.table is not None else "exact_vote_key_fallback"

    def stats(self) -> dict:
        return {"source": self.source, "scopes_from_table": self.n_table, "scopes_exact_fallback": self.n_fallback}

    def classes(self, scope_id: str, member_ids: list[str]) -> dict[str, str | None]:
        if self.table is not None:
            if not hasattr(self, "_by_scope"):
                self._by_scope = {k: dict(zip(g.candidate_id, g.class_id)) for k, g in self.table.groupby("scope_id")}
            m = self._by_scope.get(scope_id)
            if m is not None and all(c in m for c in member_ids):
                self.n_table += 1
                return {c: (None if pd.isna(m.get(c)) else m.get(c)) for c in member_ids}
        self.n_fallback += 1
        return exact_classes(self.draws, scope_id, member_ids)


def pool_outcome(member_ids: list[str], classes: dict[str, str | None], lab_map: dict[str, float]) -> dict:
    """Observed coverage and tie-averaged plurality correctness of one pool (tri-state aware).

    Coverage: 1 if any member correct; nan if none correct but some unknown; else 0.
    Plurality: modal class among voters, ties averaged (= expectation of the seeded uniform tie rule); a class is
    correct when the majority of its known member labels are correct; nan when a tied class's label is unknown.
    """
    labs = [lab_map.get(c, np.nan) for c in member_ids]
    if any(v == 1.0 for v in labs):
        cov = 1.0
    elif any(np.isnan(v) for v in labs):
        cov = np.nan
    else:
        cov = 0.0
    support: dict[str, list[str]] = {}
    for c in member_ids:
        k = classes.get(c)
        if k is not None:
            support.setdefault(k, []).append(c)
    if not support:
        return {"coverage": cov, "plurality": 0.0}     # nobody votes: nothing is delivered
    top = max(len(v) for v in support.values())
    tied = [k for k, v in support.items() if len(v) == top]
    vals = []
    for k in tied:
        known = [lab_map.get(c, np.nan) for c in support[k]]
        known = [v for v in known if not np.isnan(v)]
        vals.append(np.nan if not known else float(np.mean(known) > 0.5))
    plur = np.nan if any(np.isnan(v) for v in vals) else float(np.mean(vals))
    return {"coverage": cov, "plurality": plur}


def observed_pool_table(pools: pd.DataFrame, draws: pd.DataFrame, cls: ClassSource) -> pd.DataFrame:
    lab_map = {c: label_value(v) for c, v in zip(draws.candidate_id, draws.get("correct", pd.Series(
        [None] * len(draws))))}
    rows = []
    for p in pools.itertuples(index=False):
        mem = list(p.member_ids)
        out = pool_outcome(mem, cls.classes(p.pool_id, mem), lab_map)
        rows.append(dict(pool_id=p.pool_id, item_id=p.item_id, config_id=p.config_id, n=p.n, a=p.a, **out))
    return pd.DataFrame(rows)


def load_analysis_inputs(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    ai = Path(run_root) / "analysis_inputs"
    draws = pd.read_parquet(ai / "draws.parquet")
    pools = pd.read_parquet(ai / "pools.parquet") if (ai / "pools.parquet").exists() else None
    classes = pd.read_parquet(ai / "classes.parquet") if (ai / "classes.parquet").exists() else None
    return draws, pools, classes


def load_panels(run_root: Path) -> dict[str, list[str]]:
    import json
    path = Path(run_root) / "artifacts" / "manifests" / "splits.json"
    return json.loads(path.read_text())["panels"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
