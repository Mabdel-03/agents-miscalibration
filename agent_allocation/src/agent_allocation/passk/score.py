"""Post-release scoring: the PREREG primary family and the allocation menus (``passk-analyze --stage score``).

Runs only against a verifying seal. Reads ``analysis_inputs/draws.parquet`` (after ``--release``), the pair cache,
``sealed/e1_selection.json`` and ``sealed/t200/predictions.parquet``; writes ``scored/``:

* ``menus.csv`` -- every panel × configuration: H, S (passk tie rule), S_tie, entropy, top-wrong mass under macro,
  mixture and micro;
* ``primary.json`` -- (1) locked M* vs ``1x16`` delivered accuracy on T200 (mixture estimand, paired, two-sided,
  items where both are known; best/worst bounds), (2) P2 vs P1 item-level mean Brier over the five B = 16A
  configurations (items with < 3 known configurations dropped and reported), Holm over the two.

WP-M's own E2/E5 scorers (``passk.models.e2.score`` / ``passk.models.e5.score``) run afterwards when present.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from agent_allocation.passk.extract import load_draws
from agent_allocation.passk.metrics import config_summary, pool_metrics
from agent_allocation.passk.pairs import PairCache
from agent_allocation.passk.pools import B8_MENU, B16_MENU, build_pools
from agent_allocation.passk.readout import readout_pools
from agent_allocation.passk.stats import MACRO, MIXTURE, contrast, holm2

__all__ = ["ScoreBlocked", "brier_items", "coverage_tables", "mixed_vs_deep", "score"]


def _seal(run_root: Path) -> dict:
    from agent_allocation.passk.grade_scope import verify_seal
    seal = verify_seal(run_root)
    if seal is None:
        raise PermissionError(f"{run_root}: no verifying SEAL_predictions.json; scoring refused")
    return seal


def mixed_vs_deep(metrics: pd.DataFrame, locked: str, *, panel: str = "T200", n_resamples: int = 10_000) -> dict:
    sub = metrics[(metrics["panel"] == panel) & metrics["config_id"].isin([locked, "1x16"])]
    wide = sub.pivot_table(index=["item_id", "benchmark"], columns="config_id",
                           values=["S_known", "S_lo", "S_hi"], aggfunc="first")
    items = pd.DataFrame({"item_id": [i for i, _ in wide.index], "benchmark": [b for _, b in wide.index],
                          "m": wide[("S_known", locked)].values, "d": wide[("S_known", "1x16")].values,
                          "m_lo": wide[("S_lo", locked)].values, "m_hi": wide[("S_hi", locked)].values,
                          "d_lo": wide[("S_lo", "1x16")].values, "d_hi": wide[("S_hi", "1x16")].values})
    result = contrast(items, "m", "d", weights=MIXTURE, n_resamples=n_resamples)
    result["macro"] = contrast(items, "m", "d", weights=MACRO, n_resamples=n_resamples)["estimate"]
    items["best"] = items["m_hi"] - items["d_lo"]
    items["worst"] = items["m_lo"] - items["d_hi"]
    result["bounds_best_worst"] = [contrast(items.assign(z=0.0), "best", "z", n_resamples=200)["estimate"],
                                   contrast(items.assign(z=0.0), "worst", "z", n_resamples=200)["estimate"]]
    result["locked"] = locked
    return result


def brier_items(predictions: pd.DataFrame, metrics: pd.DataFrame, *, panel: str = "T200",
                min_known: int = 3) -> tuple[pd.DataFrame, list[str]]:
    """Per item: mean Brier of ``p_delivered`` vs delivered correctness over its known 16A configurations."""
    obs = metrics[(metrics["panel"] == panel) & metrics["config_id"].isin(B16_MENU)][
        ["item_id", "benchmark", "config_id", "S_known"]]
    # WP-M's sealed frame also carries ``benchmark`` (and p_coverage): join on the keys only, benchmark from metrics.
    pred = predictions.loc[predictions["config_id"].isin(B16_MENU), ["item_id", "config_id", "predictor",
                                                                     "p_delivered"]]
    joined = pred.merge(obs, on=["item_id", "config_id"], how="inner")
    joined = joined.dropna(subset=["S_known"])
    joined["brier"] = (joined["p_delivered"].astype(float) - joined["S_known"].astype(float)) ** 2
    per = joined.groupby(["item_id", "benchmark", "predictor"]).agg(brier=("brier", "mean"),
                                                                   n_known=("brier", "size")).reset_index()
    wide = per.pivot_table(index=["item_id", "benchmark"], columns="predictor", values=["brier", "n_known"])
    known = wide[("n_known", "P1")].fillna(0).clip(upper=wide[("n_known", "P2")].fillna(0))
    dropped = sorted(i for (i, _), k in known.items() if k < min_known)
    keep = known >= min_known
    items = pd.DataFrame({"item_id": [i for i, _ in wide.index[keep]], "benchmark": [b for _, b in wide.index[keep]],
                          "p2": wide[("brier", "P2")][keep].values, "p1": wide[("brier", "P1")][keep].values})
    return items, dropped


def coverage_tables(draws: pd.DataFrame) -> pd.DataFrame:
    """Finite-bank coverage G per item for every menu configuration whose bank exceeds the pool (never k > m)."""
    from agent_allocation.passk.metrics import bank_counts, item_mixed_coverage
    from agent_allocation.passk.pools import CONFIGS
    frames = []
    for panel, parts in (("L96", ("legacy_main", "e1_fill", "c48_extra")), ("V48", ("v48_bank",)),
                         ("T200", ("t200_bank",))):
        sub = draws[draws["partition"].isin(parts)]
        if panel == "L96":
            sub = sub[sub["partition"] != "legacy_pilot"]
        if sub.empty:
            continue
        banks = bank_counts(sub, parts)
        for config_id in B16_MENU + B8_MENU:
            n, a = CONFIGS[config_id]
            g = item_mixed_coverage(banks, {a: n})
            if len(g):
                frames.append(g.assign(panel=panel, config_id=config_id))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class ScoreBlocked(RuntimeError):
    """Scoring inputs are not complete yet (pending classes, nothing released): exit 2, the conductor retries."""


def _released_sealed(run_root: Path, seal: dict) -> dict[str, bool]:
    """Per split: released by the extract (sticky ``releases``) AND its predictions are in the verifying seal."""
    manifest_path = run_root / "analysis_inputs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    pending = int(manifest.get("classes_pending_pools") or 0)
    if pending:
        raise ScoreBlocked(f"classes_pending_pools={pending}: run pairs then extract before score")
    releases = set(manifest.get("releases") or [])
    return {split: split in releases and f"sealed/{split}/predictions.parquet" in (seal.get("artifacts") or {})
            for split in ("v48", "t200")}


def score(run_root: Path, *, n_resamples: int = 10_000) -> dict:
    run_root = Path(run_root)
    seal = _seal(run_root)
    ready = _released_sealed(run_root, seal)
    if not any(ready.values()):
        raise ScoreBlocked("no split is both released and sealed; nothing to score")
    draws = load_draws(run_root)
    cache = PairCache(run_root / "catalog" / "pair_verdicts.parquet")
    panels = ("L96",) + (("V48",) if ready["v48"] else ()) + (("T200",) if ready["t200"] else ())
    pools = build_pools(draws, panels=panels,
                        configs=B16_MENU + B8_MENU + ("1x1", "2x1", "1x2", "4x1", "2x2", "1x4"))
    metrics = pool_metrics(readout_pools(pools, draws, cache=cache, mode="passk"), draws)
    out = run_root / "scored"
    out.mkdir(parents=True, exist_ok=True)
    metrics.drop(columns=["class_sizes", "class_first_ids"]).to_parquet(out / "pool_metrics.parquet", index=False)
    config_summary(metrics).to_csv(out / "menus.csv", index=False)
    from agent_allocation.passk.stats import menu_contrasts
    pd.concat([menu_contrasts(metrics, B16_MENU, "1x16", n_resamples=n_resamples),
               menu_contrasts(metrics, B8_MENU, "1x8", n_resamples=n_resamples)], ignore_index=True) \
        .to_csv(out / "menu_contrasts.csv", index=False)
    coverage_tables(draws).to_csv(out / "finite_bank_coverage.csv", index=False)
    locked = json.loads((run_root / "sealed" / "e1_selection.json").read_text())["selected"]
    primary: dict = {"seal_code_commit": seal.get("code_commit"), "locked": locked,
                     "scored_splits": sorted(k for k, v in ready.items() if v)}
    primary["mixed_vs_deep"] = primary["mixed_vs_deep_tie_averaged"] = None
    primary["p2_vs_p1"] = primary["holm"] = None
    if ready["t200"]:                                          # both primary tests live on T200
        primary["mixed_vs_deep"] = mixed_vs_deep(metrics, locked, n_resamples=n_resamples)
        tie = metrics.assign(S_known=metrics["S_tie"])        # secondary: the analytic tie-averaged readout
        primary["mixed_vs_deep_tie_averaged"] = {
            k: v for k, v in mixed_vs_deep(tie, locked, n_resamples=n_resamples).items()
            if k in ("estimate", "ci95_studentized", "p", "n_items")}
        pred_path = run_root / "sealed" / "t200" / "predictions.parquet"
        items, dropped = brier_items(pd.read_parquet(pred_path), metrics)
        primary["p2_vs_p1"] = contrast(items, "p2", "p1", weights=MIXTURE, n_resamples=n_resamples)
        primary["p2_vs_p1"]["dropped_items_lt3_known"] = dropped
        primary["holm"] = holm2(primary["mixed_vs_deep"]["p"], primary["p2_vs_p1"]["p"])
    from agent_allocation.passk.records import write_records
    primary["records"] = write_records(run_root, draws, pools,
                                       metrics.drop(columns=["class_sizes", "class_first_ids"]))["sha256"]
    tmp = out / f".primary.json.tmp{os.getpid()}"
    tmp.write_text(json.dumps(primary, indent=2, sort_keys=True, default=float) + "\n")
    os.replace(tmp, out / "primary.json")
    import importlib
    for which, split in (("e2", "v48"), ("e5", "t200")):     # WP-M scorers only on released + sealed splits
        if not ready[split]:
            continue
        module = importlib.import_module(f"agent_allocation.passk.models.{which}")
        module.score(run_root)
    return primary
