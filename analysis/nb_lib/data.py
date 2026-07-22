"""Loaders and derived columns for the preserved supplementary-legacy notebooks.

These helpers intentionally read only ``analysis/cache/supplementary_legacy``. Their
trim-era and context-cap columns describe the historical v1 rollout and must never be
applied to the homogeneous schema-5 primary cache. Primary schema-5 ingestion is provided
by :mod:`nb_lib.ingest`; downstream primary analyses will be frozen only after the new
three-run sweep has complete, representative coverage.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "analysis"
CACHE = ANALYSIS / "cache" / "supplementary_legacy"
FIGURES = ANALYSIS / "figures"
CACHE_MANIFEST_FILENAME = "ingest_manifest_v1.json"
CACHE_BUILD_MARKER_FILENAME = ".cache_generation_in_progress.json"

RUN_ID = "full_sweep_v1"

SIZE_ORDER = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
REAS_ORDER = ["off", "b512", "b2048", "b8192", "unlimited"]
TOPO_ORDER = ["single_agent", "independent", "decentralized", "centralized"]
AGENT_ORDER = [1, 2, 3, 4, 5, 6, 7]
CTX_ORDER = ["artifact_only", "plus_intermediate", "plus_cot"]
BENCHMARKS = ["gpqa", "mmlu_pro", "truthfulqa", "math"]
MCQ_BENCHMARKS = ["gpqa", "mmlu_pro", "truthfulqa"]  # native option-logprob calibration

# Nominal thinking-token budget per reasoning rung (unlimited has no nominal budget).
REAS_TOKENS = {"off": 0, "b512": 512, "b2048": 2048, "b8192": 8192, "unlimited": np.nan}
REAS_RANK = {r: i for i, r in enumerate(REAS_ORDER)}
CTX_RANK = {c: i for i, c in enumerate(CTX_ORDER)}

# The sweep grid was trimmed on 2026-06-10 (prompt {0..3}->{0,3}; context 3->2 levels).
# Era is defined from meta.json started_at timestamps, NOT from factor-level membership.
# Pinned to the trim commit's epoch (86cc9b3, 2026-06-10 01:45:58 -0400) — a naive
# datetime would be tz-dependent and ~1h46m early, mislabeling 13 boundary cells.
TRIM_CUTOFF = 1781070358.0

# 32B is served at max_model_len=16384 (all other sizes 32768) — see
# src/agents_scaling/models.py. Long-reasoning 32B cells are context-capped; flag them.
CTX_CAPPED_SIZE = "32B"
CTX_CAPPED_REASONING = ["b8192", "unlimited"]

AXES = ["model_size", "topology", "n_agents", "context_share_level", "prompt_complexity_level",
        "reasoning_level"]
DESIGN_KEY = AXES + ["benchmark"]  # design cell = config without seed


def _cache_manifest(cache: Path | None = None) -> dict:
    """Load only a fully published cache generation.

    Ingestion withdraws the old manifest and leaves an in-progress marker before it
    replaces any parquet.  Refusing that marker prevents a notebook from reading a
    cross-generation mixture after a killed or concurrent refresh.
    """

    cache = CACHE if cache is None else cache
    marker = cache / CACHE_BUILD_MARKER_FILENAME
    if marker.exists():
        raise RuntimeError(
            f"analysis cache generation is incomplete: {marker}; rerun refresh"
        )
    path = cache / CACHE_MANIFEST_FILENAME
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"analysis cache manifest is not an object: {path}")
    mode = value.get("analysis_mode")
    if mode not in (None, "supplementary-legacy"):
        raise RuntimeError(
            "legacy notebook helpers refuse a primary schema-5 cache; use primary "
            "analysis code that does not apply trim-era assumptions"
        )
    return value


def verify_cache_integrity(cache: Path | None = None) -> dict:
    """Cryptographically verify every parquet bound by the current cache manifest."""

    cache = CACHE if cache is None else cache
    manifest_record = _cache_manifest(cache)
    contracts = manifest_record.get("cache_artifacts")
    if not isinstance(contracts, dict) or not contracts:
        raise RuntimeError("analysis cache manifest lacks artifact hash contracts")
    for filename, contract in contracts.items():
        path = cache / str(filename)
        if (
            not isinstance(contract, dict)
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != contract.get("size")
        ):
            raise RuntimeError(f"analysis cache artifact contract failed: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != contract.get("sha256"):
            raise RuntimeError(f"analysis cache artifact hash drift: {path}")
    return manifest_record


def _categorize(df: pd.DataFrame) -> pd.DataFrame:
    for col, order in [("model_size", SIZE_ORDER), ("reasoning_level", REAS_ORDER),
                       ("topology", TOPO_ORDER), ("context_share_level", CTX_ORDER)]:
        if col in df.columns:
            df[col] = pd.Categorical(df[col].astype(str), order, ordered=True)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns shared by every notebook. Idempotent."""
    df = _categorize(df)
    if "param_count" in df.columns:
        df["log2_params"] = np.log2(df["param_count"].astype(float))
    if "mean_reasoning_tokens" in df.columns:
        df["log_reas_tok"] = np.log1p(df["mean_reasoning_tokens"].astype(float))
    if "reasoning_level" in df.columns:
        df["reas_rank"] = df["reasoning_level"].astype(str).map(REAS_RANK)
        df["reas_budget"] = df["reasoning_level"].astype(str).map(REAS_TOKENS)
    if "context_share_level" in df.columns:
        df["ctx_rank"] = df["context_share_level"].astype(str).map(CTX_RANK)
    if "topology" in df.columns:
        df["is_mas"] = df["topology"].astype(str) != "single_agent"
        inferred = pd.Series(np.where(df["is_mas"], 3, 1), index=df.index)
        if "n_agents" in df.columns:
            df["n_agents"] = pd.to_numeric(df["n_agents"], errors="coerce").fillna(inferred)
        else:
            df["n_agents"] = inferred
        df["n_agents"] = df["n_agents"].astype(int)
    if set(DESIGN_KEY) <= set(df.columns):
        df["design_cell"] = df[DESIGN_KEY].astype(str).agg("|".join, axis=1)
    if {"prompt_complexity_level", "context_share_level"} <= set(df.columns):
        # Post-trim factor levels (grid membership — distinct from calendar era).
        df["is_core"] = df["prompt_complexity_level"].isin([0, 3]) & df[
            "context_share_level"].astype(str).isin(["artifact_only", "plus_cot"])
    if "started_at" in df.columns:
        df["era_post_trim"] = df["started_at"].astype(float) >= TRIM_CUTOFF
    if {"model_size", "reasoning_level"} <= set(df.columns):
        standard_32b = df["model_size"].astype(str) == CTX_CAPPED_SIZE
        if "serving_profile" in df.columns:
            standard_32b &= df["serving_profile"].astype(str) != "32B-long"
        df["ctx32b_capped"] = standard_32b & df["reasoning_level"].astype(str).isin(
            CTX_CAPPED_REASONING
        )
    if {"param_count", "mean_total_tokens"} <= set(df.columns):
        # Per-question inference-FLOPs proxy (B-params x tokens); MAS cost is automatically
        # ~3-6x SAS at the same size because mean_total_tokens sums over agents/rounds.
        df["cost"] = df["param_count"].astype(float) * df["mean_total_tokens"].astype(float)
    return df


def load_cells(source: str = "dedup") -> pd.DataFrame:
    """Cell-level table. source='dedup' (canonical, from ingest) or 'orig' (live findings)."""
    if source == "dedup":
        _cache_manifest()
        df = pd.read_parquet(CACHE / "cells_dedup_v1.parquet")
    elif source == "orig":
        df = pd.read_parquet(ANALYSIS / f"{RUN_ID}_findings.parquet")
    else:
        raise ValueError(f"unknown source {source!r}")
    return add_derived(df)


def load_items() -> pd.DataFrame:
    """Item level: one row per (cell_id, qid), deduped, all four benchmarks."""
    _cache_manifest()
    return add_derived(pd.read_parquet(CACHE / "items_v1.parquet"))


def load_agents(columns: list[str] | None = None) -> pd.DataFrame:
    """Agent-round level: one row per (cell_id, qid, agent_id, round)."""
    _cache_manifest()
    return add_derived(pd.read_parquet(CACHE / "agents_v1.parquet", columns=columns))


def manifest() -> dict:
    return _cache_manifest()


def freshness_banner() -> str:
    m = manifest()
    if not m:
        return "[nb_lib] no ingest manifest found — run analysis/refresh.sh first."
    ts = datetime.datetime.fromtimestamp(m["timestamp"]).strftime("%Y-%m-%d %H:%M")
    completed = m.get("n_cells_complete", m.get("completion_states", {}).get("complete", 0))
    directories = m.get("n_cell_dirs", sum(m.get("manifest_cells_by_run", {}).values()))
    return (
        f"[nb_lib] SUPPLEMENTARY LEGACY cache built {ts} — "
        f"{completed}/{directories} cells complete; mixed protocol, partial/non-random "
        "coverage; not a primary schema-5 estimand."
    )


def analysis_view(df: pd.DataFrame | None = None, post_trim_only: bool = True,
                  benchmark: str | None = None, min_seeds: int = 1,
                  drop_ctx_capped: bool = False) -> pd.DataFrame:
    """Historical filtered cell view for supplementary v1 analyses.

    post_trim_only keeps the core factor grid (prompt {0,3}, context {artifact_only,
    plus_cot}); pre-trim-only levels are supplementary. min_seeds filters design cells by
    seed replication — use only as a stability check (seed count correlates with runtime,
    so this subset is biased toward fast strata).
    """
    if df is None:
        df = load_cells()
    out = df
    if post_trim_only:
        out = out[out["is_core"]]
    if benchmark is not None:
        out = out[out["benchmark"] == benchmark]
    if drop_ctx_capped:
        out = out[~out["ctx32b_capped"]]
    if min_seeds > 1:
        counts = out.groupby("design_cell")["seed"].transform("nunique")
        out = out[counts >= min_seeds]
    return out.copy()


def coverage_table(df: pd.DataFrame, rows: str = "model_size",
                   cols: str = "reasoning_level") -> pd.DataFrame:
    """Design cells present per (rows x cols), for the non-random-coverage caveat."""
    d = df.drop_duplicates("design_cell")
    return pd.crosstab(d[rows], d[cols], dropna=False)


def matched_sas_join(cells: pd.DataFrame) -> pd.DataFrame:
    """MAS cells joined to their matched single-agent baseline.

    Baseline key = (model_size, benchmark, seed, reasoning_level) — efficiency is
    reasoning-conditioned, mirroring the harness (experiment/analyze.py).
    Returns MAS rows with `sas_*` columns; rows without a baseline are kept with NaN.
    """
    key = ["model_size", "benchmark", "seed", "reasoning_level"]
    sas = cells[~cells["is_mas"]]
    # Baseline is unique per key up to prompt level; prefer the same prompt level, else any.
    sas_cols = key + ["prompt_complexity_level", "cell_id", "accuracy", "mean_total_tokens",
                      "n_questions"]
    extra = [c for c in ["pa_ece_prim", "pa_ece", "acc_first_sample"] if c in sas.columns]
    sas = sas[sas_cols + extra].rename(
        columns={c: f"sas_{c}" for c in sas_cols[len(key):] + extra})
    mas = cells[cells["is_mas"]].copy()
    same_prompt = mas.merge(
        sas[sas["sas_prompt_complexity_level"].notna()],
        left_on=key + ["prompt_complexity_level"],
        right_on=key + ["sas_prompt_complexity_level"], how="left")
    # Fall back to any-prompt baseline where the same-prompt one is missing.
    any_prompt = sas.sort_values("sas_prompt_complexity_level").drop_duplicates(key)
    fallback = mas.merge(any_prompt, on=key, how="left", suffixes=("", "_any"))
    for c in [c for c in same_prompt.columns if c.startswith("sas_")]:
        if c in fallback.columns:
            same_prompt[c] = same_prompt[c].fillna(fallback[c])
    return same_prompt
