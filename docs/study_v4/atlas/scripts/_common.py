"""Shared paths, loaders and the provenance-carrying emit helper for the atlas."""
from __future__ import annotations
import json, os, math, hashlib, collections
import numpy as np, pandas as pd

RUN_ROOT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4"
CACHE = "/orcd/data/tpoggio/001/mabdel03/.cache/mech_cache"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
FIGS = os.path.join(os.path.dirname(HERE), "figures")
TABS = os.path.join(os.path.dirname(HERE), "tables")
FLAGSHIP = dict(module="A", checkpoint="32B", N=5, B=4)
POLICIES = ["S_FRESH", "S_HISTORY", "IND_VOTE", "DEC", "CEN_FLAT"]
POLICY_LABEL = {"S_FRESH": "Stateless-fresh", "S_HISTORY": "Stateless-history", "IND_VOTE": "Independent+vote",
                "DEC": "Decentralised", "CEN_FLAT": "Centralised hub"}
DOMAIN_LABEL = {"hle": "Exam (HLE)", "bcb": "Code (BCB)"}


class Emitter:
    """Collects {key: record} with provenance; merged into atlas_data.json."""

    def __init__(self, script: str):
        self.script = script
        self.rows: dict[str, dict] = {}

    def emit(self, key, value, n=None, stat=None, domain=None, note=None, source=None):
        if key in self.rows:
            raise KeyError(f"duplicate atlas key: {key}")
        rec = {"value": _clean(value), "script": self.script}
        if n is not None: rec["n"] = int(n)
        if stat is not None: rec["stat"] = _clean(stat)
        if domain is not None: rec["domain"] = domain
        if note is not None: rec["note"] = note
        if source is not None: rec["source"] = source
        self.rows[key] = rec
        return rec

    def save(self, name):
        os.makedirs(DATA, exist_ok=True)
        p = os.path.join(DATA, f"_frag_{name}.json")
        with open(p, "w") as f:
            json.dump(self.rows, f, indent=1, sort_keys=True)
        print(f"[{self.script}] wrote {len(self.rows)} keys -> {p}")


def _clean(v):
    if isinstance(v, dict): return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_clean(x) for x in v]
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, (np.integer, int)): return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v); return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if v is None or isinstance(v, str): return v
    if isinstance(v, (np.ndarray,)): return _clean(v.tolist())
    return str(v)


# ---------- loaders ----------
def selections():
    d = pd.read_parquet(f"{RUN_ROOT}/tables/selections.parquet")
    d["domain"] = d.domain.astype(str)
    return d


def episodes():
    ep = pd.read_parquet(f"{RUN_ROOT}/tables/episodes.parquet")
    sel = selections()[["source_id", "split"]].drop_duplicates("source_id")
    ep = ep.merge(sel, on="source_id", how="left")
    ep["acc"] = ep.native_final_correct.fillna(False).astype(float)  # missing final scored incorrect
    return ep


def banks():
    return pd.read_parquet(f"{RUN_ROOT}/tables/banks.parquet")


def difficulty(framing="00", checkpoint="32B"):
    """Item difficulty from the neutral-framing 10-draw bank: the study's own difficulty proxy."""
    bk = banks()
    f = bk[(bk.framing.astype(str) == framing) & (bk.checkpoint.astype(str) == checkpoint)]
    f = f.drop_duplicates("source_id")[["source_id", "pass_at_1", "pass_at_10"]]
    f["band"] = pd.cut(f.pass_at_1, [-0.01, 0.0, 0.3, 0.7, 1.0],
                       labels=["never", "hard", "contested", "easy"])
    return f


def geometry():
    g = pd.read_parquet(f"{RUN_ROOT}/neural/tables/geometry.parquet")
    g["domain"] = np.where(g.item.astype(str).str.contains("/"), "bcb", "hle")
    return g


def flagship(ep=None):
    """Module-A flagship panel: 32B, N=5, B=4, main split."""
    ep = episodes() if ep is None else ep
    return ep[(ep.module == "A") & (ep.checkpoint == "32B") & (ep.N == 5) & (ep.B == 4) & (ep.split == "main")]


# ---------- statistics (bootstrap, no significance testing — intervals only) ----------
def boot_mean(x, n_boot=10000, seed=0):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0: return dict(mean=None, lo=None, hi=None, n=0)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return dict(mean=float(x.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=len(x))


def boot_paired(a, b, n_boot=10000, seed=0):
    """Paired difference a - b with a percentile interval. Trends, not tests."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    if len(a) == 0: return dict(diff=None, lo=None, hi=None, n=0)
    d = a - b; rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return dict(diff=float(d.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=len(d))


def pair_on(df, key_col, val_col, a, b, on="source_id"):
    """Pair two levels of key_col on the same items; returns (va, vb, items)."""
    x = df[df[key_col] == a].drop_duplicates(on).set_index(on)[val_col]
    y = df[df[key_col] == b].drop_duplicates(on).set_index(on)[val_col]
    idx = x.index.intersection(y.index)
    return x.loc[idx].values, y.loc[idx].values, list(idx)
