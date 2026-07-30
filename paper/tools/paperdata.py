"""Read-only access to the full_sweep_v1 analysis caches for the paper build.

Deliberately does not use ``nb_lib.data.load_cells``/``load_items``/``load_agents``:
``nb_lib.data.CACHE`` points at ``analysis/cache/supplementary_legacy``, a directory
that does not exist in this checkout, so those loaders raise.  The ordering and
labelling constants in that module are still authoritative and are re-exported here.

Nothing in this module writes to the repository.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "analysis" / "cache"
TABLES = REPO / "analysis" / "tables"
FIGURES = REPO / "analysis" / "figures"
REPORTS = REPO / "analysis"

sys.path.insert(0, str(REPO))
from analysis.nb_lib import data as _nbd  # noqa: E402  constants only

SIZE_ORDER = list(_nbd.SIZE_ORDER)
REAS_ORDER = list(_nbd.REAS_ORDER)
TOPO_ORDER = list(_nbd.TOPO_ORDER)
CTX_ORDER = list(_nbd.CTX_ORDER)
BENCHMARKS = list(_nbd.BENCHMARKS)
MCQ_BENCHMARKS = list(_nbd.MCQ_BENCHMARKS)
REAS_TOKENS = dict(_nbd.REAS_TOKENS)
REAS_RANK = dict(_nbd.REAS_RANK)
AXES = list(_nbd.AXES)

# Display labels.  Used by both the table and the figure builders so a level is
# spelled the same way everywhere in the document.
BENCH_LABEL = {
    "gpqa": "GPQA",
    "mmlu_pro": "MMLU-Pro",
    "truthfulqa": "TruthfulQA",
    "math": "MATH",
}
TOPO_LABEL = {
    "single_agent": "single agent",
    "independent": "independent",
    "decentralized": "decentralized",
    "centralized": "centralized",
}
CTX_LABEL = {
    "artifact_only": "artifact only",
    "plus_intermediate": "plus intermediate",
    "plus_cot": "plus CoT",
}
AXIS_LABEL = {
    "model_size": "model size",
    "reasoning_level": "reasoning budget",
    "topology": "topology",
    "n_agents": "agent count",
    "context_share_level": "context sharing",
    "prompt_complexity_level": "prompt complexity",
}
METRIC_LABEL = {
    "accuracy": "accuracy",
    "pa_ece_prim": "per-agent ECE",
    "vote_ece_prim": "vote ECE",
    "fp_ece_prim": "final-producer ECE",
    "delta_vote_prim": r"$\Delta$ECE (vote)",
    "delta_fp_prim": r"$\Delta$ECE (final producer)",
    "pa_signed_gap": "per-agent signed gap",
    "vote_signed_gap": "vote signed gap",
    "cost": "cost proxy",
    "mean_total_tokens": "mean total tokens",
    "mean_reasoning_tokens": "mean reasoning tokens",
    "Ec": r"$E_c$",
    "Ae": r"$A_e$",
}

MCQ = ("gpqa", "mmlu_pro", "truthfulqa")


# --------------------------------------------------------------------------- #
# cache loaders
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def view() -> pd.DataFrame:
    """Post-trim core analysis view: 3201 rows x 77 columns, n_agents in {1, 3}."""
    return pd.read_parquet(CACHE / "analysis_view_v1.parquet")


@lru_cache(maxsize=None)
def cells() -> pd.DataFrame:
    """Canonical per-cell table before the core-grid filter: 4301 rows x 66 columns."""
    return pd.read_parquet(CACHE / "cells_dedup_v1.parquet")


def items(columns) -> pd.DataFrame:
    """Per-question rows (858,089).  Always column-projected."""
    return pd.read_parquet(CACHE / "items_v1.parquet", columns=list(columns))


def agents(columns) -> pd.DataFrame:
    """Per-agent-round rows (3,952,802).  Always column-projected: an unprojected
    read of all 19 columns is roughly two orders of magnitude slower."""
    return pd.read_parquet(CACHE / "agents_v1.parquet", columns=list(columns))


@lru_cache(maxsize=None)
def manifest() -> dict:
    return json.loads((CACHE / "ingest_manifest_v1.json").read_text())


@lru_cache(maxsize=None)
def csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / rel)


def parquet_schema(path: Path) -> pd.Series:
    """Column -> dtype string, without materialising the frame."""
    import pyarrow.parquet as pq

    sch = pq.read_schema(path)
    return pd.Series({n: str(t) for n, t in zip(sch.names, sch.types)})


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #
def order_key(col: str):
    """Sort key for a categorical axis column, falling back to lexical order."""
    orders = {
        "model_size": SIZE_ORDER,
        "reasoning_level": REAS_ORDER,
        "topology": TOPO_ORDER,
        "context_share_level": CTX_ORDER,
        "benchmark": BENCHMARKS,
    }
    if col not in orders:
        return None
    rank = {v: i for i, v in enumerate(orders[col])}
    return lambda s: s.map(lambda v: rank.get(str(v), len(rank)))


def sort_axis(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    out = df.copy()
    tmp = []
    for c in cols:
        if c not in out.columns:
            continue
        k = order_key(c)
        name = f"__k_{c}"
        out[name] = k(out[c]) if k is not None else out[c].astype(str)
        tmp.append(name)
    if not tmp:
        return out
    out = out.sort_values(tmp, kind="stable").drop(columns=tmp)
    return out.reset_index(drop=True)
