"""N3 analysis CLI: geometry tables, the family-G predictor and the family-C recalibrators.

    python -m agents_scaling.study.neural.analysis --run-id study_v4 --stage geometry  [--blocks 15,31,47]
    python -m agents_scaling.study.neural.analysis --run-id study_v4 --stage G-dev     [--seal <sha>] [--blocks 15,31,47]
    python -m agents_scaling.study.neural.analysis --run-id study_v4 --stage G-confirm [--seal <sha>] [--bootstrap 20000]
    python -m agents_scaling.study.neural.analysis --run-id study_v4 --stage C-dev     [--seal <sha>] [--refreeze]
    python -m agents_scaling.study.neural.analysis --run-id study_v4 --stage C-confirm [--seal <sha>] [--bootstrap 20000]
    python -m agents_scaling.study.neural.analysis --run-id study_v4 --stage C         (C-dev only if no C_frozen.json, then C-confirm)

Outputs go to ``<run_root>/neural/tables/*.parquet`` plus one summary JSON per stage under
``<run_root>/neural/`` (``geometry_summary.json``, ``G_dev_summary.json``, ``G_frozen.json``,
``G_confirm_summary.json``, ``C_dev_summary.json``, ``C_frozen.json``, ``C_summary.json``).
Frozen artifacts (``G_frozen.*``, ``C_frozen.json``) are written by the ``*-dev`` stages only
and sha-verified on load; the ``*-confirm`` stages never fit (spec §8.7-8.8).

Label firewall (spec §10.6, brief "correctness joins only in aggregate"): correctness enters
only through ``<run_root>/tables/selections.parquet``, i.e. after ``aggregate`` has run; the
``G-dev``/``G-confirm``/``C`` stages refuse to start without that table, the geometry stage
runs without labels (its outcome association is simply skipped), and nothing in this
package opens the protected export directory.  ``run_root = <results_root>/<run_id>``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:  # the GPU env has no agents_scaling install
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from agents_scaling.config import DEFAULT_RESULTS_ROOT  # noqa: E402
from agents_scaling.study.neural import calibration as C  # noqa: E402
from agents_scaling.study.neural import geometry as G  # noqa: E402
from agents_scaling.study.neural import readout as R  # noqa: E402
from agents_scaling.study.neural.storage import atomic_write_json  # noqa: E402

STAGES: tuple[str, ...] = ("geometry", "G-dev", "G-confirm", "C-dev", "C-confirm", "C")
NEURAL_DIR = "neural"
TABLES_SUBDIR = "tables"
DEFAULT_BLOCKS_32B: tuple[int, ...] = (15, 31, 47)
GEOMETRY_ANCHORS: tuple[str, ...] = ("NATIVE_PREFILL", "GENERATED_32", "GENERATED_128", "GENERATED_512", "FINAL_OBJECT_CLOSE")
DEFAULT_PANEL_PER_DOMAIN = 150


def log(message: str) -> None:
    print(f"[neural.analysis {time.strftime('%H:%M:%S')}] {message}", flush=True)


def neural_dir(run_root: Path) -> Path:
    return run_root / NEURAL_DIR


def tables_out(run_root: Path) -> Path:
    path = neural_dir(run_root) / TABLES_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_parquet(path: Path, frame: Any) -> Path:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(frame).to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def json_safe(value: Any) -> Any:
    """NaN/inf → null (a degenerate bootstrap or an insufficient-s metric is reported as
    missing, never as a fake number), numpy scalars → Python, tuples → lists."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    atomic_write_json(path, json_safe(summary))


def study_seed() -> bytes:
    from agents_scaling.study.config import load_config

    return load_config().study_seed


def primary_budget() -> int:
    from agents_scaling.study.config import load_config

    return int(load_config().budget.primary)


def require_join_tables(run_root: Path) -> Path:
    """The only permitted label source; refuse when aggregate has not produced it."""
    path = run_root / "tables" / "selections.parquet"
    if not path.is_file():
        raise R.ReadoutError(f"{path} is missing: correctness labels are read only through the evaluation join tables (run aggregate first)")
    return path


def parse_blocks(text: str | None) -> tuple[int, ...]:
    if not text:
        return DEFAULT_BLOCKS_32B
    return tuple(sorted({int(b) for b in text.split(",") if b.strip()}))


# --------------------------------------------------------------------------- geometry stage


def sealed_distinct_classes(run_root: Path, seal: str | None) -> dict[str, int]:
    """``selection_id → number of distinct answer classes`` among the valid candidates of the
    sealed VOTE pool (distinct public ``vote_keys``), from the seal register; empty when no
    seal is given or the register is absent.  Public fields only (no correctness)."""
    if not seal:
        return {}
    from agents_scaling.study.selection.seal import load_selections

    try:
        register = load_selections(run_root, seal)
    except Exception as exc:  # noqa: BLE001 — a missing register only disables the duplicate-frequency outcome
        log(f"seal register unavailable ({exc}); duplicate_frequency is not joined")
        return {}
    out: dict[str, int] = {}
    for sid, sel in register.get("selections", {}).items():
        keys = (sel.get("record") or {}).get("vote_keys") or {}
        distinct = {json.dumps(k, sort_keys=True) for k in keys.values() if k is not None}
        out[str(sid)] = len(distinct)
    return out


def outcome_table(run_root: Path, *, seal: str | None = None, B: int | None = None) -> dict[tuple[str, str], dict[str, Any]] | None:
    """Per (item, method) sealed-selection outcomes (VOTE over the method's primary pool at
    N=5 / the primary budget) from the evaluation join table, or None before aggregate.

    ``agreement`` = winning_count / valid_count; ``duplicate_frequency`` = 1 − distinct
    answer classes / valid_count, computed from the sealed pool's grouping (the seal
    register's ``vote_keys``) and only when ``seal`` is given — ``tied_classes`` counts tied
    *winning* classes and is not a duplicate frequency (P1-5); ``all_singleton`` is the
    sealed flag.
    """
    import pandas as pd

    path = run_root / "tables" / "selections.parquet"
    if not path.is_file():
        return None
    table = pd.read_parquet(path)
    budget = primary_budget() if B is None else int(B)
    distinct = sealed_distinct_classes(run_root, seal)
    primary = {"IND_VOTE": "archive", "DEC": "latest_slots", "CEN_FLAT": "native"}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in table.to_dict("records"):
        if r.get("selector_id") != "VOTE" or primary.get(str(r.get("method"))) != r.get("pool_kind"):
            continue
        if int(r.get("N") or 0) != 5 or int(r.get("B") or 0) != budget:
            continue
        key = (str(r["source_id"]), str(r["method"]))
        valid = float(r.get("valid_count") or 0)
        winning = float(r.get("winning_count") or 0)
        n_classes = distinct.get(str(r.get("selection_id")))
        out[key] = {
            "selected_correct": None if r.get("selected_correct") is None else float(bool(r["selected_correct"])),
            "oracle_coverage": None if r.get("oracle_coverage") is None else float(r["oracle_coverage"]),
            "candidate_mean": None if r.get("candidate_mean") is None else float(r["candidate_mean"]),
            "agreement": (winning / valid) if valid > 0 else None,
            "duplicate_frequency": (1.0 - float(n_classes) / valid) if (valid > 0 and n_classes is not None) else None,
            "all_singleton": None if r.get("all_singleton") is None else float(bool(r["all_singleton"])),
            "tied_winning_classes": None if r.get("tied_classes") is None else float(r["tied_classes"]),
            "selection_gap": None if r.get("selection_gap") is None else float(r["selection_gap"]),
        }
    return out


def run_geometry(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd

    seed = study_seed()
    blocks = parse_blocks(args.blocks)
    anchors = tuple(a for a in (args.anchors.split(",") if args.anchors else GEOMETRY_ANCHORS) if a)
    X, meta = G.load_states(run_root, "native", blocks=blocks, anchor_kinds=anchors, include_missing=True)
    if len(meta) == 0:
        raise R.ReadoutError(f"no native-stage rows under {run_root / NEURAL_DIR / 'native'}")
    dev_items = set()
    try:
        from agents_scaling.study.data.public import load_public_tasks

        tasks = {t.source_id: t for t in load_public_tasks(run_root)}
        dev_items = {sid for sid, t in tasks.items() if t.split == "dev"}
    except FileNotFoundError:
        tasks = {}
    groups = G.assemble_comparable_states(meta, blocks=blocks, anchor_kinds=anchors)
    # development standardizer + display transform per (block, anchor), fitted on dev-item states
    standardizers: dict[tuple[int, str], G.Standardizer | None] = {}
    displays: dict[tuple[int, str], str] = {}
    for b in blocks:
        for a in anchors:
            rows = [g for g in groups if g.block == b and g.anchor == a and g.item in dev_items]
            idx = np.asarray([r for g in rows for r in g.rows], dtype=np.int64)
            if idx.size >= 2:
                Xd = X[idx].astype(np.float64)
                standardizers[(b, a)] = G.Standardizer.fit(Xd)
                pca = G.display_transform(Xd, rank=args.pca_rank, label=f"display_{args.checkpoint}_b{b}_{a}")
                path = pca.save(neural_dir(run_root) / "display" / f"display_pca.{args.checkpoint}.b{b}.{a}")
                displays[(b, a)] = pca.components_hash
                log(f"display transform b{b} {a}: {pca.rank}-d from {idx.size} dev states → {path}")
            else:
                standardizers[(b, a)] = None
    outcomes = outcome_table(run_root, seal=args.seal)
    records: list[dict[str, Any]] = []
    for g in groups:
        s_contract = G.contract_s(g.method, g.role, g.phase)
        s = s_contract if s_contract is not None else g.s_available
        if g.s_available == 0:
            records.append({"item": g.item, "method": g.method, "role": g.role, "phase": g.phase, "block": g.block, "anchor": g.anchor,
                            "s": s, "n_available": 0, "n_missing": g.n_missing, "status": "no_states", "rank_ceiling": 0, "d_eff": 0,
                            "contract_registered": s_contract is not None, "is_dev": g.item in dev_items})
            continue
        Xg = X[np.asarray(g.rows, dtype=np.int64)].astype(np.float64)
        fm = G.fixed_s_metrics(Xg, s, study_seed=seed, item=g.item, config=g.config, standardizer=standardizers.get((g.block, g.anchor)), count=args.subsamples)
        rec = {"item": g.item, "method": g.method, "role": g.role, "phase": g.phase, "block": g.block, "anchor": g.anchor, "s": fm.s,
               "n_available": fm.n_available, "n_missing": g.n_missing, "n_subsamples": fm.n_subsamples, "status": fm.status,
               "rank_ceiling": fm.rank_ceiling, "d_eff": fm.d_eff, "contract_registered": s_contract is not None, "is_dev": g.item in dev_items,
               "standardized": standardizers.get((g.block, g.anchor)) is not None, **fm.metrics,
               **{f"{k}_sd": v for k, v in fm.metrics_sd.items()}}
        if fm.full_set is not None:
            rec.update({"full_s": fm.full_set.s, "full_rank_ceiling": fm.full_set.rank_ceiling, "full_participation_ratio": fm.full_set.participation_ratio,
                        "full_entropy_effective_rank": fm.full_set.entropy_effective_rank, "full_n_nonzero_eigenvalues": fm.full_set.n_nonzero_eigenvalues})
        if outcomes is not None:
            rec.update({f"outcome_{k}": v for k, v in (outcomes.get((g.item, g.method)) or {}).items()})
        records.append(rec)
    frame = pd.DataFrame.from_records(records)
    path = write_parquet(tables_out(run_root) / "geometry.parquet", frame)
    summary: dict[str, Any] = {
        "stage": "geometry", "run_root": str(run_root), "blocks": list(blocks), "anchors": list(anchors), "n_groups": int(len(frame)),
        "n_rows_loaded": int(len(meta)), "n_dev_items": len(dev_items), "subsamples": args.subsamples, "outcomes_joined": outcomes is not None,
        "display_transforms": {f"b{b}:{a}": h for (b, a), h in displays.items()}, "table": str(path),
        "status_counts": {str(k): int(v) for k, v in frame["status"].value_counts().items()} if len(frame) else {},
        "by_config": {},
        "association": {},
    }
    if len(frame):
        metric_cols = [c for c in ("participation_ratio", "entropy_effective_rank", "cosine_distance_standardized", "cosine_distance_unit", "raw_pairwise_mean", "raw_norm_mean") if c in frame]
        for key, sub in frame.groupby(["method", "role", "phase", "block", "anchor"]):
            k = "|".join(str(v) for v in key)
            entry = {"n_groups": int(len(sub)), "s": int(sub["s"].iloc[0]), "rank_ceiling": int(sub["rank_ceiling"].max()),
                     "status": {str(a): int(b) for a, b in sub["status"].value_counts().items()}}
            for c in metric_cols:
                vals = sub[c].to_numpy(dtype=np.float64)
                entry[c] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else None
            summary["by_config"][k] = entry
        if outcomes is not None:
            from scipy.stats import spearmanr

            for key, sub in frame.groupby(["method", "role", "block", "anchor"]):
                k = "|".join(str(v) for v in key)
                for oc in ("outcome_selected_correct", "outcome_oracle_coverage", "outcome_agreement", "outcome_duplicate_frequency", "outcome_all_singleton"):
                    if oc not in sub:
                        continue
                    for c in metric_cols:
                        pair = sub[[c, oc]].dropna()
                        if len(pair) >= 5 and pair[c].nunique() > 1 and pair[oc].nunique() > 1:
                            rho, p = spearmanr(pair[c], pair[oc])
                            summary["association"][f"{k}|{c}|{oc}"] = {"spearman": float(rho), "p_descriptive": float(p), "n": int(len(pair))}
    summary["note"] = ("Descriptive geometry (spec §8.5): rank ceilings min(d_eff, s-1) are printed beside every estimate; "
                       "outcome associations are predictive/explanatory diagnostics, not causal effects (§8.5, §9.6).")
    write_summary(neural_dir(run_root) / "geometry_summary.json", summary)
    log(f"geometry: {len(frame)} groups → {path}")
    return summary


# --------------------------------------------------------------------------- G stages


def g_config(args: argparse.Namespace) -> R.GConfig:
    return R.GConfig(blocks=parse_blocks(args.blocks), ranks=tuple(int(r) for r in args.ranks.split(",")) if args.ranks else R.RANK_GRID,
                     penalties=tuple(float(p) for p in args.penalties.split(",")) if args.penalties else R.PENALTY_GRID,
                     text_prefer=args.text, seed=study_seed())


def run_g_dev(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    require_join_tables(run_root)
    cfg = g_config(args)
    frames = R.build_feature_frames(run_root, "dev", blocks=cfg.blocks, seal=args.seal, panel_per_domain=None)
    log(f"G-dev: {frames.stats}")
    if len(frames.rows) == 0:
        raise R.ReadoutError("no labelled development report rows with state-anchor vectors")
    result = R.nested_cv(frames, cfg)
    losses_path = write_parquet(tables_out(run_root) / "G_dev_losses.parquet", result.losses)
    seed = R.bootstrap_seed(cfg.seed, R.BOOTSTRAP_NAMESPACE, "G-dev", args.seal or "")
    contrast = R.confirmation_contrast(result.losses, seed=seed, n_resamples=args.bootstrap)
    frozen = R.freeze(frames, cfg, meta={"seal": args.seal, "stage": "G-dev", "assembly": frames.stats, "report_anchor": R.STATE_ANCHOR,
                                         "report_anchor_note": "STATE_ANCHOR = token closing '=== END REPORT ===' (N2 state_anchor); the last chat-template token is stored separately as LAST_PREFILL"})
    json_path, npz_path = frozen.save(neural_dir(run_root))
    summary = {
        "stage": "G-dev", "run_root": str(run_root), "seal": args.seal, "blocks": list(cfg.blocks), "ranks": list(cfg.ranks), "penalties": list(cfg.penalties),
        "assembly": frames.stats, "outer": result.summary, "dev_contrast_descriptive": contrast, "frozen": {"json": str(json_path), "npz": str(npz_path),
        "selected": {v: s.to_dict() for v, s in frozen.selections.items()}, "text_kind": frozen.features.text.kind, "text_model": frozen.features.text.model,
        "text_revision": frozen.features.text.revision}, "losses_table": str(losses_path),
        "caveat": "60 development items make G estimation-oriented (brief R1 step 3); the dev contrast is descriptive, confirmation is the test.",
    }
    write_summary(neural_dir(run_root) / "G_dev_summary.json", summary)
    log(f"G-dev: outer weighted Brier {result.summary['weighted_brier']} → frozen {json_path}")
    return summary


def run_g_confirm(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    require_join_tables(run_root)
    frozen = R.FrozenG.load(neural_dir(run_root))
    sel = frozen.selections["augmented"]
    frames = R.build_feature_frames(run_root, "confirmation", blocks=(int(sel.block),), seal=args.seal, panel_per_domain=args.panel)
    log(f"G-confirm: {frames.stats} (frozen block {sel.block}, rank {sel.rank}, penalty {sel.penalty})")
    if len(frames.rows) == 0:
        raise R.ReadoutError("no labelled confirmation report rows with state-anchor vectors at the frozen block")
    preds = frozen.predict(frames)
    losses = frames.rows[["source_id", "method", "superdomain", "cluster", "report_id", "selection_id"]].copy()
    losses["y"] = frames.y
    for v, p in preds.items():
        losses[f"p_{v}"] = p
        losses[f"brier_{v}"] = R.brier(p, frames.y)
    losses_path = write_parquet(tables_out(run_root) / "G_confirm_losses.parquet", losses)
    seed = args.bootstrap_seed if args.bootstrap_seed is not None else R.bootstrap_seed(study_seed(), R.BOOTSTRAP_NAMESPACE, "G-confirm", args.seal or "")
    primary = R.confirmation_contrast(losses, seed=seed, n_resamples=args.bootstrap)
    matched = R.confirmation_contrast(losses, seed=seed, n_resamples=args.bootstrap, baseline="text_expanded", augmented="augmented")
    obs_only = R.confirmation_contrast(losses, seed=seed, n_resamples=args.bootstrap, baseline="observables", augmented="baseline")
    summary = {
        "stage": "G-confirm", "run_root": str(run_root), "seal": args.seal, "panel_per_domain": args.panel, "assembly": frames.stats,
        "frozen": {"selected": {v: s.to_dict() for v, s in frozen.selections.items()}, "meta": frozen.meta},
        "primary": primary, "parameter_matched_text_expansion": matched, "text_over_observables": obs_only, "losses_table": str(losses_path),
        "bootstrap_seed": seed, "n_resamples": args.bootstrap,
        "claim": "Family G: positive out-of-sample Brier improvement from the frozen internal features over the identical text/observable baseline (spec §9.2); conditional on the frozen readout, no confirmation refitting (§8.8).",
    }
    write_summary(neural_dir(run_root) / "G_confirm_summary.json", summary)
    log(f"G-confirm: contrast {primary['bootstrap']['estimate']:.5f} p1={primary['bootstrap']['p_one_sided']:.4g} CI=[{primary['bootstrap']['ci_low']:.5f}, {primary['bootstrap']['ci_high']:.5f}]")
    return summary


# --------------------------------------------------------------------------- C stage


def _c_scopes_summary(frozen: C.FrozenC) -> dict[str, Any]:
    return {k: {kk: vv for kk, vv in v.to_dict().items() if kk not in ("common", "conditioned")} for k, v in frozen.scopes.items()}


def run_c_dev(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Fit both recalibrators on development and freeze ``C_frozen.json`` (§8.7 "freeze both
    recalibrators before confirmation").  Refuses to overwrite an existing frozen artifact
    unless ``--refreeze`` is passed (P1-2)."""
    require_join_tables(run_root)
    seed = study_seed()
    frozen_path = neural_dir(run_root) / C.C_FROZEN_JSON
    if frozen_path.is_file() and not args.refreeze:
        raise C.CalibrationError(f"{frozen_path} exists: the C recalibrators are frozen (pass --refreeze to refit on development deliberately)")
    all_frames = R.build_feature_frames(run_root, "dev", blocks=(), seal=args.seal, panel_per_domain=None)
    dev_rows = all_frames.rows
    log(f"C-dev: dev {all_frames.stats}")
    if len(dev_rows) == 0:
        raise R.ReadoutError("no labelled development report rows")
    frozen = C.fit_recalibrators(dev_rows, seed=seed, meta={"seal": args.seal, "stage": "C-dev", "dev_assembly": all_frames.stats})
    frozen_path = frozen.save(neural_dir(run_root))
    dev_scored = C.score_rows(dev_rows, frozen)
    dev_losses = write_parquet(tables_out(run_root) / "C_dev_losses.parquet", dev_scored)
    boot_seed = args.bootstrap_seed if args.bootstrap_seed is not None else R.bootstrap_seed(seed, R.BOOTSTRAP_NAMESPACE, "C", args.seal or "")
    summary: dict[str, Any] = {"stage": "C-dev", "run_root": str(run_root), "seal": args.seal, "frozen": str(frozen_path), "refrozen": bool(args.refreeze),
                               "scopes": _c_scopes_summary(frozen), "dev_assembly": all_frames.stats, "dev_losses_table": str(dev_losses),
                               "dev_descriptive": C.contrast_C(dev_scored, seed=boot_seed, n_resamples=min(args.bootstrap, 2000), frozen=frozen),
                               "caveat": "development-only fit; the dev contrast is descriptive (in-sample recalibration), confirmation is the test."}
    write_summary(neural_dir(run_root) / "C_dev_summary.json", summary)
    log(f"C-dev: recalibrators frozen → {frozen_path}")
    return summary


def run_c_confirm(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Score the confirmation panel with the sha-verified frozen recalibrators; never fits."""
    require_join_tables(run_root)
    seed = study_seed()
    frozen = C.FrozenC.load(neural_dir(run_root))
    frozen_path = neural_dir(run_root) / C.C_FROZEN_JSON
    frozen_sha = json.loads(frozen_path.read_text(encoding="utf-8")).get("sha256")
    conf = R.build_feature_frames(run_root, "confirmation", blocks=(), seal=args.seal, panel_per_domain=args.panel)
    log(f"C-confirm: {conf.stats} (frozen {frozen_path}, sha {str(frozen_sha)[:12]})")
    summary: dict[str, Any] = {"stage": "C-confirm", "run_root": str(run_root), "seal": args.seal, "panel_per_domain": args.panel,
                               "frozen": str(frozen_path), "frozen_sha256": frozen_sha, "frozen_meta": dict(frozen.meta), "scopes": _c_scopes_summary(frozen),
                               "confirmation_assembly": conf.stats}
    boot_seed = args.bootstrap_seed if args.bootstrap_seed is not None else R.bootstrap_seed(seed, R.BOOTSTRAP_NAMESPACE, "C", args.seal or "")
    if len(conf.rows):
        scored = C.score_rows(conf.rows, frozen)
        losses_path = write_parquet(tables_out(run_root) / "C_losses.parquet", scored)
        summary["primary"] = C.contrast_C(scored, seed=boot_seed, n_resamples=args.bootstrap, frozen=frozen)
        summary["losses_table"] = str(losses_path)
        summary["bootstrap_seed"] = boot_seed
        log(f"C-confirm: contrast {summary['primary']['bootstrap']['estimate']:.5f} p1={summary['primary']['bootstrap']['p_one_sided']:.4g}")
    else:
        summary["primary"] = None
        log("C-confirm: no confirmation rows yet (recalibrators stay frozen on development)")
    summary["claim"] = ("Family C: positive source-item-weighted Brier improvement of the explicit TEAM_SELECTED forecast over the "
                        "development-calibrated selected PERSONAL_FINAL baseline (spec §9.2); forecast quality, not by itself calibration (§8.7); "
                        "scored with the frozen recalibrators only (no confirmation refitting, §8.8).")
    write_summary(neural_dir(run_root) / "C_summary.json", summary)
    return summary


def run_c(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """``C`` = C-dev (only when no frozen artifact exists, or under ``--refreeze``) followed by
    C-confirm.  An existing ``C_frozen.json`` is loaded and sha-verified, never refitted, so
    re-running ``C`` after more dev labels arrive cannot silently change the recalibrators
    that score confirmation (P1-2)."""
    frozen_path = neural_dir(run_root) / C.C_FROZEN_JSON
    if frozen_path.is_file() and not args.refreeze:
        log(f"C: {frozen_path} exists — loading the frozen recalibrators (pass --refreeze to refit)")
        dev_summary = None
        source = "loaded"
    else:
        dev_summary = run_c_dev(run_root, args)
        source = "refitted" if args.refreeze else "fitted"
    summary = run_c_confirm(run_root, args)
    summary.update({"stage": "C", "frozen_source": source, "dev": dev_summary})
    write_summary(neural_dir(run_root) / "C_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m agents_scaling.study.neural.analysis", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--seal", default=None, help="restrict reports/forecasts to this sealed selection register")
    ap.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    ap.add_argument("--checkpoint", default="32B")
    ap.add_argument("--blocks", default=None, help="comma-separated residual blocks (default 15,31,47 for Qwen3-32B)")
    ap.add_argument("--anchors", default=None, help="geometry: comma-separated anchor kinds (default: the five native anchors)")
    ap.add_argument("--ranks", default=None, help="G: PCA rank grid (default 16,32,64,128)")
    ap.add_argument("--penalties", default=None, help="G: L2 penalty grid (default 0.1,1,10,100,1000)")
    ap.add_argument("--text", default="auto", choices=("auto", "tfidf", "sentence-transformers"))
    ap.add_argument("--panel", type=int, default=DEFAULT_PANEL_PER_DOMAIN, help="confirmation items per superdomain (rank prefix)")
    ap.add_argument("--bootstrap", type=int, default=R.DEFAULT_BOOTSTRAP)
    ap.add_argument("--bootstrap-seed", type=int, default=None, help="override the manifest-derived Generator seed (recorded)")
    ap.add_argument("--subsamples", type=int, default=G.DEFAULT_SUBSAMPLES)
    ap.add_argument("--pca-rank", type=int, default=G.DEFAULT_PCA_RANK)
    ap.add_argument("--refreeze", action="store_true", help="C-dev / C: refit and overwrite an existing C_frozen.json (deliberate re-freeze only)")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.results_root) / args.run_id
    if not run_root.is_dir():
        log(f"run root {run_root} does not exist")
        return 2
    runners = {"geometry": run_geometry, "G-dev": run_g_dev, "G-confirm": run_g_confirm, "C-dev": run_c_dev, "C-confirm": run_c_confirm, "C": run_c}
    try:
        runners[args.stage](run_root, args)
    except (R.ReadoutError, C.CalibrationError, G.GeometryError) as exc:
        log(f"refused: {exc}")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
