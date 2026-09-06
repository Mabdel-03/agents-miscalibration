"""Confirmatory statistics for study_v4 — spec §9.2 six-family Holm and §9.3 bootstrap.

Reads ``<run_root>/tables/{selections,banks,episodes}.parquet`` (``aggregate.py``) and writes
``<run_root>/tables/stats.json`` + ``stats.md``.

Primary families (§9.2):

* **A — awareness**: two-sided mean effect of VOTE_AWARE on VOTE@5, averaging TEAM_FRAME=0/1
  equally, on the N_main four-cell independent banks (framing cell ``<TEAM_FRAME><VOTE_AWARE>``).
* **O — orchestration**: omnibus equality of native final accuracy across the executed full
  policies at B4 on N_main; max-T over all pairwise contrasts (global p, simultaneous CIs).
* **C**, **G**: read from ``forecast/tables/family_C.json`` / ``neural/tables/family_G.json`` when
  those stages ran (``{"p": float, "estimate": float, "ci95": [lo, hi], "n": {...}, "note": str}``);
  otherwise p = 1 (unexecuted / estimation-only).
* **R**, **M**: unexecuted in this run (amendment register): p = 1.

Resampling (§9.3): deterministic source-cluster resamples (default 20,000), independently
within the two superdomains, 0.5/0.5 superdomain weights and equal item weights within a
superdomain (§9.1); null-centred studentised cluster bootstrap for scalar contrasts (analytic
cluster SE inside each resample); Monte-Carlo p with the +1/+1 adjustment.  The analysis panel of
each family is the smallest common completed rank prefix (§9.1; blocks of 25 per domain, as in
``aggregate.common_prefix``); the all-complete-items panel is reported as a sensitivity.
Panel rule: the primary panel is the smallest common completed rank prefix (§9.1) when it holds
at least one block per domain; otherwise — infrastructure-incomplete items leave holes that would
collapse the prefix to zero and silently turn a family into 'no rejection' — the primary panel is the
MATCHED complete-case panel (items complete in every config of that family), reported with
``panel_used`` and a note.  Completion here is infrastructure-driven (preemption), not outcome-driven:
budget/format/truncation failures DO complete and are scored incorrect.  Both panels are always
reported side by side.

Secondary tables (S1–S5) are estimates with ordinary percentile intervals, never Holm-adjusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RESULTS_ROOT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results"
DOMAINS: tuple[str, ...] = ("hle", "bcb")
WEIGHTS: dict[str, float] = {"hle": 0.5, "bcb": 0.5}
BLOCK_PER_DOMAIN = 25
N_RESAMPLES = 20_000
N_RESAMPLES_SECONDARY = 2_000
ALPHA = 0.05
FAMILIES: tuple[str, ...] = ("A", "O", "R", "C", "G", "M")
O_METHODS: tuple[str, ...] = ("S_FRESH", "S_HISTORY", "IND_VOTE", "DEC", "CEN_FLAT", "CEN_RLM")
FRAMINGS: tuple[str, ...] = ("00", "01", "10", "11")  # <TEAM_FRAME><VOTE_AWARE>


# --------------------------------------------------------------------------- resampling


def analysis_seed(study_seed_hex: str, label: str) -> int:
    """Deterministic 63-bit seed derived from the frozen study seed and an analysis label."""
    digest = hashlib.sha256(f"{label}:{study_seed_hex}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class Resampler:
    """Source-cluster resample indices, drawn independently within each superdomain."""

    def __init__(self, n_by_domain: Mapping[str, int], n_resamples: int, seed: int) -> None:
        self.n_resamples = int(n_resamples)
        self.seed = int(seed)
        self.n_by_domain = {d: int(n_by_domain.get(d, 0)) for d in DOMAINS}
        rng = np.random.default_rng(self.seed)
        self.idx: dict[str, np.ndarray] = {}
        for d in DOMAINS:
            n = self.n_by_domain[d]
            self.idx[d] = (
                rng.integers(0, n, size=(self.n_resamples, n), dtype=np.int32)
                if n > 0
                else np.zeros((self.n_resamples, 0), dtype=np.int32)
            )


def _norm_weights(domains: Sequence[str], weights: Mapping[str, float]) -> dict[str, float]:
    total = math.fsum(weights[d] for d in domains)
    return {d: weights[d] / total for d in domains}


def paired_contrast(
    diff_by_domain: Mapping[str, np.ndarray],
    rs: Resampler,
    *,
    weights: Mapping[str, float] = WEIGHTS,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Null-centred studentised cluster bootstrap for a pooled mean of per-item contrasts."""
    ds = [d for d in DOMAINS if d in diff_by_domain and len(diff_by_domain[d]) >= 2]
    if not ds:
        return {"estimate": None, "p": 1.0, "n": {}, "note": "no domain with >= 2 items"}
    w = _norm_weights(ds, weights)
    vals = {d: np.asarray(diff_by_domain[d], dtype=float) for d in ds}
    n_d = {d: int(vals[d].shape[0]) for d in ds}
    est_d = {d: float(vals[d].mean()) for d in ds}
    var_d = {d: float(vals[d].var(ddof=1)) for d in ds}
    est = math.fsum(w[d] * est_d[d] for d in ds)
    se = math.sqrt(math.fsum(w[d] ** 2 * var_d[d] / n_d[d] for d in ds))
    B = rs.n_resamples
    theta = np.zeros(B)
    var_acc = np.zeros(B)
    for d in ds:
        v = vals[d][rs.idx[d]]  # (B, n_d)
        theta += w[d] * v.mean(axis=1)
        var_acc += w[d] ** 2 * v.var(axis=1, ddof=1) / n_d[d]
    se_b = np.sqrt(var_acc)
    ok = se_b > 0
    t_b = np.where(ok, (theta - est) / np.where(ok, se_b, 1.0), 0.0)
    t_obs = est / se if se > 0 else 0.0
    if se <= 0:
        p = 1.0
    elif alternative == "two-sided":
        p = (1 + int((np.abs(t_b) >= abs(t_obs)).sum())) / (B + 1)
    elif alternative == "greater":
        p = (1 + int((t_b >= t_obs).sum())) / (B + 1)
    elif alternative == "less":
        p = (1 + int((t_b <= t_obs).sum())) / (B + 1)
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    q_lo, q_hi = (float(x) for x in np.quantile(t_b, [ALPHA / 2, 1 - ALPHA / 2]))
    p_lo, p_hi = (float(x) for x in np.quantile(theta, [ALPHA / 2, 1 - ALPHA / 2]))
    return {
        "estimate": est,
        "by_domain": est_d,
        "n": n_d,
        "weights": w,
        "se": se,
        "t": t_obs,
        "p": float(p),
        "alternative": alternative,
        "ci95_studentized": [est - q_hi * se, est - q_lo * se],
        "ci95_percentile": [p_lo, p_hi],
        "n_resamples": B,
    }


def max_t_family(
    y_by_domain: Mapping[str, np.ndarray],
    labels: Sequence[str],
    rs: Resampler,
    *,
    weights: Mapping[str, float] = WEIGHTS,
    chunk: int = 2_000,
) -> dict[str, Any]:
    """All-pair max-T over the centred resampling distribution (§9.3, family O).

    ``y_by_domain[d]`` is an ``(n_d, k)`` matrix of per-item binary outcomes, columns = ``labels``.
    Returns the global p-value, per-pair estimates with max-T-adjusted p-values and simultaneous
    95% intervals, and per-method pooled means with percentile intervals.
    """
    k = len(labels)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    ds = [d for d in DOMAINS if d in y_by_domain and y_by_domain[d] is not None and len(y_by_domain[d]) >= 2]
    if not ds or not pairs:
        return {"p_global": 1.0, "pairs": [], "methods": [], "n": {}, "note": "insufficient panel"}
    w = _norm_weights(ds, weights)
    Y = {d: np.asarray(y_by_domain[d], dtype=float) for d in ds}
    n_d = {d: int(Y[d].shape[0]) for d in ds}
    P = len(pairs)
    est = np.zeros(P)
    var_sum = np.zeros(P)
    mean_m = np.zeros(k)
    for d in ds:
        D = Y[d][:, ii] - Y[d][:, jj]  # (n, P)
        est += w[d] * D.mean(axis=0)
        var_sum += w[d] ** 2 * D.var(axis=0, ddof=1) / n_d[d]
        mean_m += w[d] * Y[d].mean(axis=0)
    se = np.sqrt(var_sum)
    t = np.where(se > 0, est / np.where(se > 0, se, 1.0), 0.0)
    B = rs.n_resamples
    maxabs = np.zeros(B)
    means_b = np.zeros((B, k))
    for start in range(0, B, chunk):
        sl = slice(start, min(B, start + chunk))
        b = sl.stop - sl.start
        th = np.zeros((b, P))
        va = np.zeros((b, P))
        for d in ds:
            Yb = Y[d][rs.idx[d][sl]]  # (b, n_d, k)
            Db = Yb[:, :, ii] - Yb[:, :, jj]  # (b, n_d, P)
            th += w[d] * Db.mean(axis=1)
            va += w[d] ** 2 * Db.var(axis=1, ddof=1) / n_d[d]
            means_b[sl] += w[d] * Yb.mean(axis=1)
        seb = np.sqrt(va)
        tb = np.where(seb > 0, (th - est) / np.where(seb > 0, seb, 1.0), 0.0)
        maxabs[sl] = np.abs(tb).max(axis=1)
    obs_max = float(np.abs(t).max())
    p_global = (1 + int((maxabs >= obs_max).sum())) / (B + 1)
    q95 = float(np.quantile(maxabs, 1 - ALPHA))
    pair_rows = []
    for p_idx, (i, j) in enumerate(pairs):
        pair_rows.append(
            {
                "contrast": f"{labels[i]} - {labels[j]}",
                "estimate": float(est[p_idx]),
                "se": float(se[p_idx]),
                "t": float(t[p_idx]),
                "p_maxT_adjusted": (1 + int((maxabs >= abs(t[p_idx])).sum())) / (B + 1) if se[p_idx] > 0 else 1.0,
                "ci95_simultaneous": [float(est[p_idx] - q95 * se[p_idx]), float(est[p_idx] + q95 * se[p_idx])],
            }
        )
    method_rows = []
    for m in range(k):
        lo, hi = (float(x) for x in np.quantile(means_b[:, m], [ALPHA / 2, 1 - ALPHA / 2]))
        method_rows.append(
            {
                "method": labels[m],
                "mean": float(mean_m[m]),
                "by_domain": {d: float(Y[d][:, m].mean()) for d in ds},
                "ci95_percentile": [lo, hi],
            }
        )
    return {
        "p_global": float(p_global),
        "max_abs_t_observed": obs_max,
        "max_abs_t_q95": q95,
        "n": n_d,
        "weights": w,
        "n_resamples": B,
        "methods": method_rows,
        "pairs": pair_rows,
    }


def mean_ci(values_by_domain: Mapping[str, np.ndarray], rs: Resampler, *, weights: Mapping[str, float] = WEIGHTS) -> dict[str, Any]:
    """Pooled mean with an ordinary percentile interval (secondary tables)."""
    ds = [d for d in DOMAINS if d in values_by_domain and len(values_by_domain[d]) >= 1]
    if not ds:
        return {"mean": None, "n": {}}
    w = _norm_weights(ds, weights)
    vals = {d: np.asarray(values_by_domain[d], dtype=float) for d in ds}
    est_d = {d: float(vals[d].mean()) for d in ds}
    est = math.fsum(w[d] * est_d[d] for d in ds)
    theta = np.zeros(rs.n_resamples)
    for d in ds:
        theta += w[d] * vals[d][rs.idx[d]].mean(axis=1)
    lo, hi = (float(x) for x in np.quantile(theta, [ALPHA / 2, 1 - ALPHA / 2]))
    return {"mean": est, "by_domain": est_d, "n": {d: int(vals[d].shape[0]) for d in ds}, "ci95_percentile": [lo, hi]}


def choose_panel(result: dict, *, min_prefix: int = BLOCK_PER_DOMAIN) -> dict:
    """Pick the family's primary panel and stamp ``panel_used``/``panel_note`` on ``result``.

    ``prefix`` when the smallest common completed rank prefix holds ``min_prefix`` items per domain,
    else the matched complete-case panel (§9.1 holes are infrastructure, not outcomes)."""
    n_prefix = int(result.get("n_prefix") or 0)
    use_prefix = n_prefix >= min_prefix and result.get("prefix", {}).get("estimate" if "estimate" in result.get("prefix", {}) else "p_global") is not None
    key = "prefix" if use_prefix else "complete"
    result["panel_used"] = key
    result["panel_note"] = (
        f"smallest common completed rank prefix, {n_prefix} items per domain (§9.1)"
        if use_prefix else
        f"matched complete-case panel: the completed rank prefix held {n_prefix} < {min_prefix} items per domain "
        f"(infrastructure holes), so the panel is the items complete in every config of this family"
    )
    return result[key]


# --------------------------------------------------------------------------- Holm


def holm(pvalues: Mapping[str, float], alpha: float = ALPHA) -> dict[str, dict[str, Any]]:
    """Holm step-down over the given families; adjusted p-values are monotone and capped at 1."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, Any]] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[name] = {"p": float(p), "p_holm": float(running), "reject": bool(running <= alpha), "rank": rank + 1}
    return out


# --------------------------------------------------------------------------- panels


def common_prefix_n(ranks_by_config_domain: Mapping[str, Mapping[str, set[int]]], *, block: int = BLOCK_PER_DOMAIN) -> int:
    """Largest multiple of ``block`` n such that every config has ranks 0..n-1 in both domains (§9.1)."""
    best: int | None = None
    for ranks_by_domain in ranks_by_config_domain.values():
        for d in DOMAINS:
            ranks = ranks_by_domain.get(d, set())
            n = 0
            while n in ranks:
                n += 1
            best = n if best is None else min(best, n)
    if best is None:
        return 0
    return (best // block) * block


def build_panel(
    rows: Iterable[Mapping[str, Any]],
    *,
    configs: Sequence[str],
    config_of: Callable[[Mapping[str, Any]], str],
    value_of: Callable[[Mapping[str, Any]], float],
    block: int = BLOCK_PER_DOMAIN,
) -> dict[str, Any]:
    """Item × config matrices per domain on (a) the common completed rank prefix and (b) all
    complete items.  Duplicate (item, config) rows must agree (first one wins, count recorded)."""
    table: dict[str, dict[int, dict[str, float]]] = {d: {} for d in DOMAINS}
    ids: dict[str, dict[int, str]] = {d: {} for d in DOMAINS}
    dup = 0
    for r in rows:
        d = str(r["domain"])
        if d not in table:
            continue
        cfg = config_of(r)
        if cfg not in configs:
            continue
        rank = int(r["rank"])
        slot = table[d].setdefault(rank, {})
        if cfg in slot:
            dup += 1
            continue
        slot[cfg] = float(value_of(r))
        ids[d].setdefault(rank, str(r["source_id"]))
    ranks_by_cfg = {c: {d: {rk for rk, slot in table[d].items() if c in slot} for d in DOMAINS} for c in configs}
    n_prefix = common_prefix_n(ranks_by_cfg, block=block)

    def matrix(select: Callable[[int, Mapping[str, float]], bool]) -> dict[str, Any]:
        out: dict[str, Any] = {"ranks": {}, "ids": {}, "Y": {}}
        for d in DOMAINS:
            rks = sorted(rk for rk, slot in table[d].items() if all(c in slot for c in configs) and select(rk, slot))
            out["ranks"][d] = rks
            out["ids"][d] = [ids[d][rk] for rk in rks]
            out["Y"][d] = np.array([[table[d][rk][c] for c in configs] for rk in rks], dtype=float).reshape(len(rks), len(configs))
        return out

    return {
        "configs": list(configs),
        "n_prefix": n_prefix,
        "prefix": matrix(lambda rk, _slot: rk < n_prefix),
        "complete": matrix(lambda _rk, _slot: True),
        "duplicate_rows_ignored": dup,
        "ranks_present": {c: {d: len(ranks_by_cfg[c][d]) for d in DOMAINS} for c in configs},
    }


# --------------------------------------------------------------------------- tables


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def load_tables(run_root: Path) -> dict[str, list[dict[str, Any]]]:
    t = run_root / "tables"
    sel = _read_parquet(t / "selections.parquet")
    banks = _read_parquet(t / "banks.parquet")
    eps = _read_parquet(t / "episodes.parquet")
    split_of = {r["source_id"]: r["split"] for r in sel if r.get("split")}
    for r in eps:
        r.setdefault("split", split_of.get(r["source_id"], "main"))
    for r in banks:
        r.setdefault("split", split_of.get(r["source_id"], "main"))
    return {"selections": sel, "banks": banks, "episodes": eps}


def _external_family(run_root: Path, family: str) -> dict[str, Any]:
    """Family C/G results from the N3 analysis stage (``neural/C_summary.json`` /
    ``neural/G_confirm_summary.json``: ``primary.bootstrap.{estimate,p_one_sided,ci_low,ci_high}``),
    or the generic ``{"p", "estimate", "ci95", "n", "note"}`` file; p = 1 when neither exists."""
    generic = run_root / ("forecast" if family == "C" else "neural") / "tables" / f"family_{family}.json"
    summary = run_root / "neural" / ("C_summary.json" if family == "C" else "G_confirm_summary.json")
    if summary.exists():
        d = json.loads(summary.read_text())
        prim = (d.get("primary") or {}).get("bootstrap") or {}
        if prim.get("p_one_sided") is not None:
            return {"p": float(prim["p_one_sided"]), "estimate": prim.get("estimate"), "ci95": [prim.get("ci_low"), prim.get("ci_high")], "n": d.get("n") or (d.get("primary") or {}).get("n"),
                    "note": f"family {family}: one-sided (improvement > 0) from {summary.name}; " + str(d.get("note") or (d.get("primary") or {}).get("note") or ""), "source": str(summary), "executed": True}
        return {"p": 1.0, "estimate": None, "note": f"family {family}: {summary.name} present but no primary bootstrap (estimation-only): " + str(d.get("note") or d.get("status") or ""), "source": str(summary), "executed": False}
    if generic.exists():
        d = json.loads(generic.read_text())
        return {"p": float(d.get("p", 1.0)), "estimate": d.get("estimate"), "ci95": d.get("ci95"), "n": d.get("n"), "note": d.get("note", f"family {family} from {generic.name}"), "source": str(generic), "executed": True}
    return {"p": 1.0, "estimate": None, "note": f"family {family} not executed in this run (p = 1 by §9.2)", "executed": False}


def family_A(sel: Sequence[Mapping[str, Any]], seed: int, n_resamples: int, *, checkpoint: str = "32B") -> dict[str, Any]:
    rows = [
        r
        for r in sel
        if r.get("module") == "F"
        and r.get("pool_kind") == "bank_prefix"
        and r.get("prefix_k") == 5
        and r.get("selector_id") == "VOTE"
        and r.get("checkpoint") == checkpoint
        and r.get("split") == "main"
        and r.get("framing") in FRAMINGS
    ]
    panel = build_panel(rows, configs=FRAMINGS, config_of=lambda r: str(r["framing"]), value_of=lambda r: 1.0 if r.get("selected_correct") else 0.0)
    out: dict[str, Any] = {"definition": "0.5*[(Y01-Y00)+(Y11-Y10)] on VOTE@5 of the four framing banks; framing = <TEAM_FRAME><VOTE_AWARE>", "n_prefix": panel["n_prefix"], "ranks_present": panel["ranks_present"]}
    for key in ("prefix", "complete"):
        Y = panel[key]["Y"]
        diffs = {d: 0.5 * ((Y[d][:, 1] - Y[d][:, 0]) + (Y[d][:, 3] - Y[d][:, 2])) for d in DOMAINS if Y[d].shape[0] > 0}
        rs = Resampler({d: len(v) for d, v in diffs.items()}, n_resamples, analysis_seed(str(seed), f"A.{key}"))
        res = paired_contrast(diffs, rs, alternative="two-sided")
        team = {d: 0.5 * ((Y[d][:, 2] - Y[d][:, 0]) + (Y[d][:, 3] - Y[d][:, 1])) for d in diffs}
        inter = {d: (Y[d][:, 3] - Y[d][:, 2]) - (Y[d][:, 1] - Y[d][:, 0]) for d in diffs}
        res["secondary_S1"] = {
            "team_frame_effect": paired_contrast(team, rs),
            "interaction_(11-10)-(01-00)": paired_contrast(inter, rs),
            "cell_means": {f: mean_ci({d: Y[d][:, i] for d in diffs}, rs) for i, f in enumerate(FRAMINGS)},
        }
        out[key] = res
    panel = choose_panel(out)
    out["p"] = panel["p"] if panel.get("estimate") is not None else 1.0
    out["estimate"] = panel.get("estimate")
    out["ci95"] = panel.get("ci95_studentized")
    out["n"] = panel.get("n", {})
    out["executed"] = panel.get("estimate") is not None
    return out


def family_O(eps: Sequence[Mapping[str, Any]], seed: int, n_resamples: int, *, checkpoint: str = "32B", B: int = 4) -> dict[str, Any]:
    rows = [r for r in eps if r.get("module") == "A" and r.get("checkpoint") == checkpoint and int(r.get("B", -1)) == B and int(r.get("episode_rep", 0)) == 0 and r.get("split") == "main"]
    present = [m for m in O_METHODS if any(r.get("method") == m for r in rows)]
    out: dict[str, Any] = {"definition": f"omnibus equality of native final accuracy across {present} at B{B} on N_main; max-T over all pairs", "methods": present, "unexecuted_methods": [m for m in O_METHODS if m not in present]}
    if len(present) < 2:
        out.update({"p": 1.0, "executed": False, "note": "fewer than two executed policies"})
        return out
    panel = build_panel(rows, configs=present, config_of=lambda r: str(r["method"]), value_of=lambda r: 1.0 if r.get("native_final_correct") else 0.0)
    out["scoring"] = "native_final_correct None (no native final candidate) scored as incorrect under the frozen common failure rule"
    out["native_final_missing_by_method"] = {m: sum(1 for r in rows if r.get("method") == m and r.get("native_final_correct") is None) for m in present}
    out["n_prefix"] = panel["n_prefix"]
    out["ranks_present"] = panel["ranks_present"]
    for key in ("prefix", "complete"):
        Y = {d: panel[key]["Y"][d] for d in DOMAINS if panel[key]["Y"][d].shape[0] > 0}
        rs = Resampler({d: v.shape[0] for d, v in Y.items()}, n_resamples, analysis_seed(str(seed), f"O.{key}"))
        out[key] = max_t_family(Y, present, rs)
    panel = choose_panel(out)
    out["p"] = panel["p_global"] if panel.get("pairs") else 1.0
    out["executed"] = bool(panel.get("pairs"))
    out["n"] = panel.get("n", {})
    return out


def group_table(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], value_of: Callable[[Mapping[str, Any]], float | None], seed: int, n_resamples: int, *, label: str) -> list[dict[str, Any]]:
    """Per-group pooled means with percentile intervals; items resampled within domain."""
    groups: dict[tuple, dict[str, dict[str, float]]] = {}
    for r in rows:
        v = value_of(r)
        if v is None:
            continue
        g = tuple(r.get(k) for k in keys)
        groups.setdefault(g, {d: {} for d in DOMAINS}).get(str(r["domain"]), {}).setdefault(str(r["source_id"]), float(v))
    out = []
    for g, per_domain in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        vals = {d: np.array(list(per_domain[d].values()), dtype=float) for d in DOMAINS if per_domain[d]}
        rs = Resampler({d: len(v) for d, v in vals.items()}, n_resamples, analysis_seed(str(seed), f"{label}:{g}"))
        row = {k: g[i] for i, k in enumerate(keys)}
        row.update(mean_ci(vals, rs))
        out.append(row)
    return out


def secondary_tables(tables: Mapping[str, list[dict[str, Any]]], seed: int, n_resamples: int) -> dict[str, Any]:
    sel, banks, eps = tables["selections"], tables["banks"], tables["episodes"]
    main_sel = [r for r in sel if r.get("split") == "main"]
    main_banks = [r for r in banks if r.get("split") == "main"]
    main_eps = [r for r in eps if r.get("split") == "main"]
    # Correctness fields: a missing/invalid final or selection is INCORRECT under the frozen
    # common failure rule (§5.2, §9.1 intention-to-run) — never dropped from the denominator.
    yes = lambda r, k: 1.0 if r.get(k) else 0.0
    out: dict[str, Any] = {}
    out["S4_pass_at_k_by_framing"] = {
        f"pass_at_{k}": group_table(main_banks, ["checkpoint", "framing"], lambda r, k=k: (float(r[f"pass_at_{k}"]) if r.get(f"pass_at_{k}") is not None else None), seed, n_resamples, label=f"S4.pass{k}")
        for k in (1, 2, 3, 5, 10)
    }
    bank_sel = [r for r in main_sel if r.get("module") == "F" and r.get("pool_kind") == "bank_prefix"]
    out["S4_bank_prefix_selection_accuracy"] = group_table(bank_sel, ["checkpoint", "framing", "prefix_k", "selector_id"], lambda r: yes(r, "selected_correct"), seed, n_resamples, label="S4.banksel")
    a_sel = [r for r in main_sel if r.get("module") == "A"]
    out["S4_architecture_selection"] = {
        "selected_accuracy": group_table(a_sel, ["checkpoint", "method", "N", "B", "pool_kind", "selector_id"], lambda r: yes(r, "selected_correct"), seed, n_resamples, label="S4.A.sel"),
        "candidate_mean": group_table(a_sel, ["checkpoint", "method", "N", "B", "pool_kind", "selector_id"], lambda r: (float(r["candidate_mean"]) if r.get("candidate_mean") is not None else None), seed, n_resamples, label="S4.A.cm"),
        "oracle_coverage": group_table(a_sel, ["checkpoint", "method", "N", "B", "pool_kind", "selector_id"], lambda r: yes(r, "oracle_coverage"), seed, n_resamples, label="S4.A.oc"),
        "selection_gap": group_table(a_sel, ["checkpoint", "method", "N", "B", "pool_kind", "selector_id"], lambda r: (float(r["selection_gap"]) if r.get("selection_gap") is not None else None), seed, n_resamples, label="S4.A.gap"),
    }
    out["S5_budget_response_native_final"] = group_table([r for r in main_eps if r.get("module") == "A"], ["checkpoint", "method", "N", "B"], lambda r: yes(r, "native_final_correct"), seed, n_resamples, label="S5.budget")
    out["S3_checkpoint_panel_native_final"] = group_table([r for r in main_eps if r.get("module") == "M"], ["checkpoint", "method", "N", "B"], lambda r: yes(r, "native_final_correct"), seed, n_resamples, label="S3.M")
    out["S3_checkpoint_panel_selected"] = group_table([r for r in main_sel if r.get("module") == "M"], ["checkpoint", "method", "N", "B", "pool_kind", "selector_id"], lambda r: yes(r, "selected_correct"), seed, n_resamples, label="S3.Msel")
    out["S2_membership_panel_native_final"] = group_table([r for r in main_eps if r.get("module") == "N"], ["checkpoint", "method", "N", "B"], lambda r: yes(r, "native_final_correct"), seed, n_resamples, label="S2.N")
    out["S2_membership_panel_selected"] = group_table([r for r in main_sel if r.get("module") == "N"], ["checkpoint", "method", "N", "B", "pool_kind", "selector_id"], lambda r: yes(r, "selected_correct"), seed, n_resamples, label="S2.Nsel")
    out["S2_degree_native_final"] = group_table([r for r in main_eps if r.get("module") == "D"], ["checkpoint", "method", "degree", "B"], lambda r: yes(r, "native_final_correct"), seed, n_resamples, label="S2.D")
    out["S2_degree_selected"] = group_table([r for r in main_sel if r.get("module") == "D"], ["checkpoint", "method", "degree", "B", "pool_kind", "selector_id"], lambda r: yes(r, "selected_correct"), seed, n_resamples, label="S2.Dsel")
    e_eps = [r for r in main_eps if r.get("module") == "E"]
    out["E_repeated_episodes_native_final"] = group_table(e_eps, ["checkpoint", "method", "episode_rep"], lambda r: yes(r, "native_final_correct"), seed, n_resamples, label="E.rep")
    # within-item dispersion across episodes (estimation-only)
    per_item: dict[tuple, list[float]] = {}
    for r in e_eps:
        if r.get("native_final_correct") is not None:
            per_item.setdefault((r["method"], r["domain"], r["source_id"]), []).append(1.0 if r["native_final_correct"] else 0.0)
    disp: dict[str, dict[str, list[float]]] = {}
    for (m, d, _sid), ys in per_item.items():
        if len(ys) >= 2:
            disp.setdefault(m, {}).setdefault(d, []).append(float(np.var(ys, ddof=1)))
    out["E_within_item_variance"] = {m: mean_ci({d: np.array(v) for d, v in dd.items()}, Resampler({d: len(v) for d, v in dd.items()}, n_resamples, analysis_seed(str(seed), f"E.var.{m}"))) for m, dd in disp.items()}
    miss: dict[tuple, dict[str, int]] = {}
    for r in main_eps:
        k = (r.get("module"), r.get("method"), r.get("checkpoint"), r.get("B"))
        m = miss.setdefault(k, {"episodes": 0, "native_final_missing": 0, "native_final_invalid": 0})
        m["episodes"] += 1
        if r.get("native_final_correct") is None:
            m["native_final_missing"] += 1
        elif r.get("native_final_valid") is False:
            m["native_final_invalid"] += 1
    out["native_final_missingness"] = [{"module": k[0], "method": k[1], "checkpoint": k[2], "B": k[3], **v} for k, v in sorted(miss.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]
    stop = {}
    for r in main_eps:
        stop.setdefault((r.get("module"), r.get("method"), r.get("checkpoint"), r.get("B")), {}).setdefault(r.get("stop_reason"), 0)
        stop[(r.get("module"), r.get("method"), r.get("checkpoint"), r.get("B"))][r.get("stop_reason")] += 1
    out["stop_reasons"] = [{"module": k[0], "method": k[1], "checkpoint": k[2], "B": k[3], "counts": v} for k, v in sorted(stop.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]
    return out


# --------------------------------------------------------------------------- driver


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parents[3]).stdout.strip()
    except Exception:
        return None


def _study_seed(run_root: Path) -> str:
    frozen = run_root / "FROZEN.yaml"
    if frozen.exists():
        for line in frozen.read_text().splitlines():
            if line.startswith("study_seed_hex:"):
                return line.split(":", 1)[1].strip()
    return "0" * 64


def run_stats(run_root: Path, *, n_resamples: int = N_RESAMPLES, n_resamples_secondary: int = N_RESAMPLES_SECONDARY, with_secondary: bool = True) -> dict[str, Any]:
    tables = load_tables(run_root)
    seed_hex = _study_seed(run_root)
    base_seed = analysis_seed(seed_hex, "study_v4.stats")
    fam: dict[str, Any] = {}
    fam["A"] = family_A(tables["selections"], base_seed, n_resamples)
    fam["O"] = family_O(tables["episodes"], base_seed, n_resamples)
    fam["R"] = {"p": 1.0, "executed": False, "note": "family R (recursive scaling, L32 D2 vs D1) not executed: amendment register (no RLM frontier)"}
    fam["C"] = _external_family(run_root, "C")
    fam["G"] = _external_family(run_root, "G")
    fam["M"] = {"p": 1.0, "executed": False, "note": "family M (content use / activation edits) not executed: amendment register"}
    decisions = holm({f: float(fam[f]["p"]) for f in FAMILIES})
    for f in FAMILIES:
        fam[f]["holm"] = decisions[f]
    manifest = {
        "study_seed_hex": seed_hex,
        "analysis_seed": base_seed,
        "n_resamples_primary": n_resamples,
        "n_resamples_secondary": n_resamples_secondary,
        "alpha": ALPHA,
        "families": list(FAMILIES),
        "weights": WEIGHTS,
        "block_per_domain": BLOCK_PER_DOMAIN,
        "code_version": _git_sha(),
        "tables": {k: len(v) for k, v in tables.items()},
    }
    out = {"manifest": manifest, "families": fam, "holm": decisions}
    if with_secondary:
        out["secondary"] = secondary_tables(tables, base_seed, n_resamples_secondary)
    return out


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
        return f"[{x[0]:.{nd}f}, {x[1]:.{nd}f}]"
    return str(x)


def render_markdown(res: Mapping[str, Any]) -> str:
    m = res["manifest"]
    lines = ["# study_v4 confirmatory statistics", "", f"analysis seed `{m['analysis_seed']}` from study seed; {m['n_resamples_primary']:,} primary resamples; code `{m['code_version']}`; tables: {m['tables']}", ""]
    lines += ["## Six primary families (Holm, α = 0.05)", "", "| family | executed | estimate | 95% CI | n (hle/bcb) | p | Holm p | reject | note |", "|---|---|---|---|---|---|---|---|---|"]
    for f in FAMILIES:
        d = res["families"][f]
        n = d.get("n") or {}
        ci = d.get("ci95") if f != "O" else None
        note = d.get("note", d.get("definition", ""))
        if d.get("panel_used"):
            note = f"panel: {d['panel_used']} ({d.get('panel_note', '')}). {note}"
        lines.append(f"| {f} | {'yes' if d.get('executed') else 'no'} | {_fmt(d.get('estimate'))} | {_fmt(ci)} | {n.get('hle', '—')}/{n.get('bcb', '—')} | {_fmt(d['holm']['p'], 4)} | {_fmt(d['holm']['p_holm'], 4)} | {'**yes**' if d['holm']['reject'] else 'no'} | {note} |")
    A = res["families"]["A"]
    if A.get("executed"):
        lines += ["", f"### A — awareness (primary panel: {A.get('panel_used')} — {A.get('panel_note')}; prefix n = {A['n_prefix']} per domain; complete-case n = {A['complete'].get('n')})", "", "| panel | estimate | by domain | SE | t | p (two-sided) | 95% CI (studentized) |", "|---|---|---|---|---|---|---|"]
        for key in ("prefix", "complete"):
            r = A[key]
            lines.append(f"| {key} | {_fmt(r.get('estimate'))} | {_fmt(r.get('by_domain'))} | {_fmt(r.get('se'))} | {_fmt(r.get('t'))} | {_fmt(r.get('p'), 4)} | {_fmt(r.get('ci95_studentized'))} |")
        s1 = A["prefix"].get("secondary_S1", {})
        if s1:
            lines += ["", "S1 (secondary, prefix panel): TEAM_FRAME effect " + _fmt(s1["team_frame_effect"].get("estimate")) + " " + _fmt(s1["team_frame_effect"].get("ci95_studentized")) + "; interaction " + _fmt(s1["interaction_(11-10)-(01-00)"].get("estimate")) + " " + _fmt(s1["interaction_(11-10)-(01-00)"].get("ci95_studentized")) + "; cell means " + ", ".join(f"{f}={_fmt(v.get('mean'))}" for f, v in s1["cell_means"].items())]
    O = res["families"]["O"]
    if O.get("executed"):
        lines += ["", f"### O — orchestration (primary panel: {O.get('panel_used')} — {O.get('panel_note')}; prefix n = {O['n_prefix']} per domain; prefix max-T p = {_fmt(O['prefix'].get('p_global'), 4)}; complete-case p = {_fmt(O['complete'].get('p_global'), 4)})", "", "| method | mean | by domain | 95% CI |", "|---|---|---|---|"]
        for r in O[O.get("panel_used", "prefix")]["methods"]:
            lines.append(f"| {r['method']} | {_fmt(r['mean'])} | {_fmt(r['by_domain'])} | {_fmt(r['ci95_percentile'])} |")
        lines += ["", "| contrast | estimate | SE | t | max-T adjusted p | simultaneous 95% CI |", "|---|---|---|---|---|---|"]
        for r in O[O.get("panel_used", "prefix")]["pairs"]:
            lines.append(f"| {r['contrast']} | {_fmt(r['estimate'])} | {_fmt(r['se'])} | {_fmt(r['t'])} | {_fmt(r['p_maxT_adjusted'], 4)} | {_fmt(r['ci95_simultaneous'])} |")
    sec = res.get("secondary")
    if sec:
        lines += ["", "## Secondary tables (estimates with percentile intervals; not Holm-adjusted)"]
        for name, tab in sec.items():
            if name == "S4_pass_at_k_by_framing":
                for k, rows in tab.items():
                    lines += ["", f"### {name}.{k}", "", "| checkpoint | framing | mean | by domain | 95% CI | n |", "|---|---|---|---|---|---|"]
                    for r in rows:
                        lines.append(f"| {r['checkpoint']} | {r['framing']} | {_fmt(r['mean'])} | {_fmt(r.get('by_domain'))} | {_fmt(r.get('ci95_percentile'))} | {r.get('n')} |")
            elif name == "S4_architecture_selection":
                for sub, rows in tab.items():
                    lines += ["", f"### {name}.{sub}", "", "| checkpoint | method | N | B | pool | selector | mean | by domain | 95% CI | n |", "|---|---|---|---|---|---|---|---|---|---|"]
                    for r in rows:
                        lines.append(f"| {r['checkpoint']} | {r['method']} | {r['N']} | {r['B']} | {r['pool_kind']} | {r['selector_id']} | {_fmt(r['mean'])} | {_fmt(r.get('by_domain'))} | {_fmt(r.get('ci95_percentile'))} | {r.get('n')} |")
            elif name in ("E_within_item_variance",):
                lines += ["", f"### {name}", ""] + [f"- {mth}: {_fmt(v.get('mean'))} {_fmt(v.get('ci95_percentile'))} n={v.get('n')}" for mth, v in tab.items()]
            elif name == "native_final_missingness":
                lines += ["", f"### {name} (missing/invalid finals are scored incorrect; counts shown for transparency)", ""] + [f"- {r['module']}.{r['method']}.{r['checkpoint']}.B{r['B']}: episodes={r['episodes']} missing_final={r['native_final_missing']} invalid_final={r['native_final_invalid']}" for r in tab]
            elif name == "stop_reasons":
                lines += ["", f"### {name}", ""] + [f"- {r['module']}.{r['method']}.{r['checkpoint']}.B{r['B']}: {r['counts']}" for r in tab]
            elif isinstance(tab, list) and tab:
                keys = [k for k in tab[0] if k not in ("mean", "by_domain", "n", "ci95_percentile")]
                lines += ["", f"### {name}", "", "| " + " | ".join(keys) + " | mean | by domain | 95% CI | n |", "|" + "---|" * (len(keys) + 4)]
                for r in tab:
                    lines.append("| " + " | ".join(str(r.get(k)) for k in keys) + f" | {_fmt(r['mean'])} | {_fmt(r.get('by_domain'))} | {_fmt(r.get('ci95_percentile'))} | {r.get('n')} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    p.add_argument("--resamples", type=int, default=N_RESAMPLES)
    p.add_argument("--resamples-secondary", type=int, default=N_RESAMPLES_SECONDARY)
    p.add_argument("--no-secondary", action="store_true")
    p.add_argument("--out", default=None, help="JSON path (default <run_root>/tables/stats.json)")
    p.add_argument("--md", default=None, help="Markdown path (default <run_root>/tables/stats.md)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.results_root) / args.run_id
    res = run_stats(run_root, n_resamples=args.resamples, n_resamples_secondary=args.resamples_secondary, with_secondary=not args.no_secondary)
    out = Path(args.out) if args.out else run_root / "tables" / "stats.json"
    md = Path(args.md) if args.md else run_root / "tables" / "stats.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    md.write_text(render_markdown(res))
    h = res["holm"]
    print(f"[stats] wrote {out} and {md}; Holm: " + ", ".join(f"{f}: p={h[f]['p']:.4f} adj={h[f]['p_holm']:.4f}{' REJECT' if h[f]['reject'] else ''}" for f in FAMILIES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
