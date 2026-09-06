"""Family G — incremental internal readout at FINAL_HANDOFF_REPORT (N3; spec §8.8, §9.2).

Pipeline
1. :func:`build_feature_frames` joins, per (item, method): the compiled report render
   (``forecast/reports``), the shadow forecast (``forecast/``), the STATE_ANCHOR residuals of
   the N1 ``report`` stage and — only through ``<run_root>/tables/selections.parquet``, i.e.
   only after ``aggregate`` — the sealed selected-correctness label.  ``data/protected`` is
   never opened.
2. Feature families, every learned transform fitted INSIDE the training fold:
   * observables (architecture one-hot, domain, answer format, selected personal confidence
     + missing flag, vote winning_count/tied_classes/all_singleton, calls by role, spent /
     slack fractions, stop reason one-hot, task tokens, selected candidate length, report
     prompt/evidence tokens);
   * text (pinned sentence-transformers model from the offline HF cache when importable
     *and* cached, else TF-IDF + TruncatedSVD-64; which one is recorded);
   * neural (``h`` at the state anchor: fold-fitted PCA projections at rank r in
     {16, 32, 64, 128} clipped to the training rank, plus norm statistics).
3. :func:`nested_cv` — 5 source-grouped, superdomain-stratified outer folds x 5 grouped
   inner folds; the inner Brier selects (block, PCA rank, L2 penalty) with the one-SE rule
   preferring stronger regularization, then the shallower block, then the smaller rank.
   Baseline (observables + text) and a parameter-count-matched text expansion get the same
   folds and tuning budget.
4. :func:`freeze` — rerun the selection on all development items, refit, write
   ``<run_root>/neural/G_frozen.json`` (+ ``G_frozen.npz``) with the selected block / rank /
   penalty, coefficients, PCA components hash, feature list and a sha256.
5. :func:`confirmation_contrast` — paired per-item Brier(baseline) - Brier(augmented) with
   the §9.3 20,000-resample source-cluster bootstrap within superdomain (null-centered
   studentized, one-sided, +1/+1 Monte Carlo correction, 95% percentile interval, 0.5/0.5
   superdomain weights, equal weights across methods).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agents_scaling.study import identity
from agents_scaling.study.neural import geometry as G

# --------------------------------------------------------------------------- frozen constants

#: Spec §8.8 penalty grid (L2 strength λ; sklearn ``C = 1/λ``) and PCA rank candidates.
PENALTY_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
RANK_GRID: tuple[int, ...] = (16, 32, 64, 128)
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
DEFAULT_BOOTSTRAP = 20_000
TEXT_SVD_DIMS = 64
#: Pinned text embedding model (loaded strictly from the offline HF cache; the cached
#: snapshot hash is recorded as the revision).  Falls back to TF-IDF + SVD when absent.
PINNED_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NORM_STAT_NAMES: tuple[str, ...] = ("h_norm", "h_log_norm", "h_centered_norm")
REPORT_METHODS: tuple[str, ...] = ("IND_VOTE", "DEC", "CEN_FLAT")
STOP_REASONS: tuple[str, ...] = ("CALL_CAP", "BUDGET", "NATIVE_FINAL", "ROUNDS", "CONTEXT_FAILURE", "COMPLETE", "OTHER")
ANSWER_FORMATS: tuple[str, ...] = ("multipleChoice", "exactMatch", "code")
SUPERDOMAINS: tuple[str, ...] = ("hle", "bcb")
ROLES: tuple[str, ...] = ("root", "revise", "hub", "worker")
FOLD_NAMESPACE = "G_FOLDS"
BOOTSTRAP_NAMESPACE = "G_BOOTSTRAP"
G_FROZEN_JSON = "G_frozen.json"
G_FROZEN_NPZ = "G_frozen.npz"
VARIANTS: tuple[str, ...] = ("baseline", "augmented", "text_expanded", "observables")
MIN_CLUSTERS_FOR_INFERENCE = 30
STATE_ANCHOR = "STATE_ANCHOR"
_EPS = 1e-6
#: A cluster SE below this (relative) tolerance is zero variance → no studentized p-value (§9.3).
_SE_TOL = 1e-10


class ReadoutError(RuntimeError):
    """A refusal (labels before aggregate, missing frozen artifact, corrupt hash)."""


# --------------------------------------------------------------------------- rows


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2


def superdomain_of(source_id: str) -> str:
    return "hle" if str(source_id).startswith("hle:") else "bcb" if str(source_id).startswith("bcb:") else "other"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _calls_by_role(value: Any) -> dict[str, float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {}
    if not isinstance(value, Mapping):
        return {}
    return {str(k): _num(v) for k, v in value.items()}


def row_from_report(render: Mapping[str, Any], *, forecast: Mapping[str, Any] | None = None, task_tokens: int | None = None) -> dict[str, Any]:
    """One feature row (no label) from a report render (N2 ``report_render``) and, when
    available, its shadow forecast (N2 ``forecast_output``)."""
    report = render.get("report") or render
    item = report.get("item") or {}
    vote = report.get("vote_metadata") or {}
    budget = report.get("budget_metadata") or {}
    selected = report.get("selected_candidate")
    roles = _calls_by_role(budget.get("calls_by_role"))
    b_flops = _num(budget.get("B_flops"), 0.0)
    q_personal = report.get("selected_personal_confidence")
    missing = bool(report.get("personal_confidence_missing")) or q_personal is None
    parsed = (forecast or {}).get("parsed") or {}
    status = (forecast or {}).get("parse_status")
    forecast_valid = bool(forecast) and status == "ok" and parsed.get("q_team_now") is not None
    source_id = str(report.get("source_id") or render.get("source_id"))
    return {
        "source_id": source_id,
        "method": str(report.get("method") or render.get("method")),
        "report_id": str(report.get("report_id") or render.get("report_id")),
        "selection_id": report.get("selection_id") or render.get("selection_id"),
        "seal": report.get("seal") or render.get("seal"),
        "superdomain": item.get("domain") or superdomain_of(source_id),
        "cluster": source_id,
        "split": item.get("split"),
        "answer_format": item.get("answer_format"),
        "N": int(_num(item.get("N"), 0)),
        "B": int(_num(item.get("B"), 0)),
        "personal_conf": None if missing else float(q_personal),
        "personal_missing": bool(missing),
        "selected_is_sentinel": bool(report.get("selected_is_sentinel")),
        "winning_count": _num(vote.get("winning_count")),
        "tied_classes": _num(vote.get("tied_classes")),
        "all_singleton": float(bool(vote.get("all_singleton"))),
        "valid_count": _num(vote.get("valid_count")),
        "planned_count": _num(vote.get("planned_count")),
        "pool_size": _num(vote.get("pool_size")),
        "calls_root": roles.get("root", 0.0), "calls_revise": roles.get("revise", 0.0),
        "calls_hub": roles.get("hub", 0.0), "calls_worker": roles.get("worker", 0.0),
        "calls_total": float(sum(roles.values())),
        "calls_admitted": _num(budget.get("calls_admitted")),
        "spent_frac": (_num(budget.get("spent_flops")) / b_flops) if b_flops > 0 else 0.0,
        "slack_frac": (_num(budget.get("slack_flops")) / b_flops) if b_flops > 0 else 0.0,
        "stop_reason": str(budget.get("stop_reason") or "OTHER"),
        "task_tokens": float(task_tokens if task_tokens is not None else len((report.get("task_text") or "").split())),
        "selected_len": float(len(json.dumps(selected, ensure_ascii=False)) if selected else 0.0),
        "final_answer_len": float(len(str((selected or {}).get("final_answer") or ""))),
        "prompt_tokens": _num(render.get("prompt_tokens")),
        "evidence_tokens": _num(render.get("evidence_tokens") or report.get("evidence_tokens")),
        "packets": float(len(report.get("nonselected") or [])),
        "text": str(report.get("text") or ""),
        "q_team_now": float(parsed["q_team_now"]) if forecast_valid else None,
        "q_personal_forecast": float(parsed["q_personal"]) if (forecast_valid and parsed.get("q_personal") is not None) else None,
        "forecast_valid": bool(forecast_valid),
        "forecast_status": status,
    }


@dataclass
class FeatureFrames:
    """Rows (pandas) + the state-anchor matrices per block (``rows['h_row_b<block>']`` indexes
    ``X[block]``; ``-1`` = no vector) + the assembly bookkeeping."""

    rows: Any
    X: dict[int, np.ndarray]
    blocks: tuple[int, ...]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        return self.rows["y"].to_numpy(dtype=np.float64)

    def h(self, block: int, index: np.ndarray | None = None) -> np.ndarray:
        idx = self.rows[f"h_row_b{int(block)}"].to_numpy(dtype=np.int64)
        if index is not None:
            idx = idx[index]
        if np.any(idx < 0):
            raise ReadoutError(f"block {block}: {int(np.sum(idx < 0))} rows have no state-anchor vector")
        return self.X[int(block)][idx]


def assemble_rows(
    renders: Iterable[Mapping[str, Any]],
    *,
    forecasts: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    labels: Mapping[str, Any] | None = None,
    label_meta: Mapping[str, Mapping[str, Any]] | None = None,
    task_tokens: Mapping[str, int] | None = None,
    neural: tuple[np.ndarray, Any] | None = None,
    blocks: Sequence[int] = (),
    require_labels: bool = True,
    require_neural: bool = True,
) -> FeatureFrames:
    """Pure assembly (disk-free; the unit tests feed it directly).

    ``labels`` maps ``selection_id -> selected_correct`` (from the evaluation join table);
    ``neural`` is ``(X, meta)`` from :func:`geometry.load_states` of the ``report`` stage
    restricted to STATE_ANCHOR rows.  Rows without a label or without every requested block
    are dropped and counted when required.
    """
    import pandas as pd

    forecasts = forecasts or {}
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"renders": 0, "unlabeled": 0, "missing_neural": 0, "label_mismatch": 0}
    index: dict[tuple[str, int], int] = {}
    X_by_block: dict[int, np.ndarray] = {}
    if neural is not None:
        X, meta = neural
        for r in meta.to_dict("records"):
            if not bool(r.get("present", int(r.get("vector_index", -1)) >= 0)):
                continue
            if r.get("anchor") not in (None, STATE_ANCHOR):
                continue
            index[(str(r["id"]), int(r["block"]))] = int(r["vector_index"])
        for b in blocks:
            X_by_block[int(b)] = np.asarray(X, dtype=np.float32)
    for render in renders:
        stats["renders"] += 1
        report = render.get("report") or render
        key = (str(report.get("source_id") or render.get("source_id")), str(report.get("method") or render.get("method")))
        row = row_from_report(render, forecast=forecasts.get(key), task_tokens=(task_tokens or {}).get(key[0]))
        sid = row["selection_id"]
        if labels is not None and sid in labels and labels[sid] is not None:
            row["y"] = int(bool(labels[sid]))
            meta_row = (label_meta or {}).get(sid) or {}
            for col in ("split", "rank", "domain", "answer_format"):
                if meta_row.get(col) is not None:
                    row[col if col != "domain" else "superdomain"] = meta_row[col]
            if meta_row.get("source_id") is not None and str(meta_row["source_id"]) != row["source_id"]:
                stats["label_mismatch"] += 1
                continue
        else:
            row["y"] = None
            if require_labels:
                stats["unlabeled"] += 1
                continue
        ok = True
        for b in blocks:
            vi = index.get((row["report_id"], int(b)), -1)
            row[f"h_row_b{int(b)}"] = vi
            ok = ok and vi >= 0
        if blocks and not ok and require_neural:
            stats["missing_neural"] += 1
            continue
        rows.append(row)
    frame = pd.DataFrame.from_records(rows)
    if len(frame):
        frame = frame.sort_values(["source_id", "method"], kind="stable").reset_index(drop=True)
    stats["rows"] = int(len(frame))
    return FeatureFrames(rows=frame, X=X_by_block, blocks=tuple(int(b) for b in blocks), stats=stats)


# --------------------------------------------------------------------------- disk loaders


def tables_dir(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / "tables"


def load_labels(run_root: str | os.PathLike) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """``selection_id -> selected_correct`` from ``tables/selections.parquet`` (the only
    permitted label source); refuses when ``aggregate`` has not written the table."""
    import pandas as pd

    path = tables_dir(run_root) / "selections.parquet"
    if not path.is_file():
        raise ReadoutError(f"{path} is missing: correctness labels are read only through the evaluation join tables (run aggregate first)")
    table = pd.read_parquet(path)
    labels: dict[str, Any] = {}
    meta: dict[str, dict[str, Any]] = {}
    for r in table.to_dict("records"):
        sid = str(r["selection_id"])
        labels[sid] = None if r.get("selected_correct") is None or (isinstance(r.get("selected_correct"), float) and math.isnan(r["selected_correct"])) else bool(r["selected_correct"])
        meta[sid] = {k: r.get(k) for k in ("source_id", "split", "rank", "domain", "answer_format", "method", "cell_id")}
    return labels, meta


def load_renders(run_root: str | os.PathLike, *, seal: str | None = None) -> list[dict[str, Any]]:
    from agents_scaling.study.forecast.shadow import REPORTS_DIR, forecast_dir

    out = []
    directory = forecast_dir(run_root) / REPORTS_DIR
    if not directory.is_dir():
        return out
    for path in sorted(directory.rglob("*.json")):  # BCB ids contain "/" → nested one level
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        if seal is not None and str(data.get("seal")) != str(seal):
            continue
        out.append(data)
    return out


def load_forecasts(run_root: str | os.PathLike, *, seal: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    from agents_scaling.study.forecast.shadow import forecast_dir

    out: dict[tuple[str, str], dict[str, Any]] = {}
    directory = forecast_dir(run_root)
    if not directory.is_dir():
        return out
    for path in sorted(directory.rglob("*.json")):  # BCB ids contain "/" → nested one level
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        if data.get("kind") != "SHADOW_FORECAST":
            continue
        if seal is not None and str(data.get("seal")) != str(seal):
            continue
        out[(str(data["source_id"]), str(data["method"]))] = data
    return out


def load_task_tokens(run_root: str | os.PathLike) -> dict[str, int]:
    from agents_scaling.study.data.public import load_public_tasks

    try:
        return {t.source_id: int(t.task_tokens) for t in load_public_tasks(run_root)}
    except FileNotFoundError:
        return {}


def select_split(frames: FeatureFrames, split: str, *, panel_per_domain: int | None = None) -> FeatureFrames:
    """``dev`` = the development split; ``confirmation`` = ``main`` items with rank below the
    per-domain panel size (the C/G panel is the balanced rank prefix, §6.2)."""
    rows = frames.rows
    if len(rows) == 0 or "split" not in rows:
        stats = {**frames.stats, "split": split, "rows_in_split": 0}
        return FeatureFrames(rows=rows, X=frames.X, blocks=frames.blocks, stats=stats)
    if split == "dev":
        keep = rows["split"] == "dev"
    elif split in ("confirmation", "main"):
        keep = rows["split"] == "main"
        if panel_per_domain is not None and "rank" in rows:
            keep &= rows["rank"].fillna(10**9).astype(int) < int(panel_per_domain)
    else:
        raise ValueError(f"unknown split {split!r}")
    sub = rows[keep].reset_index(drop=True)
    stats = {**frames.stats, "split": split, "rows_in_split": int(len(sub))}
    return FeatureFrames(rows=sub, X=frames.X, blocks=frames.blocks, stats=stats)


def build_feature_frames(
    run_root: str | os.PathLike,
    split: str,
    *,
    blocks: Sequence[int],
    seal: str | None = None,
    panel_per_domain: int | None = 150,
    require_labels: bool = True,
) -> FeatureFrames:
    """Disk entry point: renders + forecasts + labels (tables only) + report-stage states."""
    renders = load_renders(run_root, seal=seal)
    forecasts = load_forecasts(run_root, seal=seal)
    labels, meta = load_labels(run_root) if require_labels else ({}, {})
    neural = G.load_states(run_root, "report", blocks=blocks, anchor_kinds=[STATE_ANCHOR]) if blocks else None
    frames = assemble_rows(renders, forecasts=forecasts, labels=labels if require_labels else None, label_meta=meta,
                           task_tokens=load_task_tokens(run_root), neural=neural, blocks=blocks, require_labels=require_labels)
    return select_split(frames, split, panel_per_domain=panel_per_domain)


# --------------------------------------------------------------------------- folds


def fold_hash(seed: bytes, namespace: str, *parts: Any) -> int:
    return int.from_bytes(identity.blind_order_key(seed, namespace, *parts)[:8], "big")


def grouped_stratified_folds(clusters: Sequence[str], strata: Sequence[str], n_folds: int, seed: bytes, *, namespace: str = FOLD_NAMESPACE) -> np.ndarray:
    """Fold id per row: every cluster (source item + all its derivatives) stays in one fold;
    clusters are ordered by a study-hashed key within each stratum and dealt round-robin, so
    the strata (superdomains) are balanced across folds.  Deterministic in ``seed``."""
    clusters = [str(c) for c in clusters]
    strata = [str(s) for s in strata]
    stratum_of: dict[str, str] = {}
    for c, s in zip(clusters, strata):
        stratum_of.setdefault(c, s)
    fold_of: dict[str, int] = {}
    for stratum in sorted(set(stratum_of.values())):
        members = sorted((c for c, s in stratum_of.items() if s == stratum), key=lambda c: (fold_hash(seed, namespace, stratum, c), c))
        for i, c in enumerate(members):
            fold_of[c] = i % n_folds
    return np.asarray([fold_of[c] for c in clusters], dtype=np.int64)


# --------------------------------------------------------------------------- feature transforms


@dataclass
class ObservableEncoder:
    """Numeric + one-hot observables with fold-fitted vocabularies (label-free)."""

    numeric: tuple[str, ...] = (
        "personal_conf_filled", "personal_missing", "selected_is_sentinel", "winning_count", "tied_classes", "all_singleton",
        "valid_count", "planned_count", "pool_size", "calls_root", "calls_revise", "calls_hub", "calls_worker", "calls_total",
        "calls_admitted", "spent_frac", "slack_frac", "task_tokens", "selected_len", "final_answer_len", "prompt_tokens",
        "evidence_tokens", "packets", "N",
    )
    categorical: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: {
        "method": REPORT_METHODS, "superdomain": SUPERDOMAINS, "answer_format": ANSWER_FORMATS, "stop_reason": STOP_REASONS,
    })
    vocab_: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fill_conf_: float = 0.5

    def fit(self, rows: Any) -> "ObservableEncoder":
        self.vocab_ = {}
        for col, fixed in self.categorical.items():
            seen = sorted({str(v) for v in rows[col].tolist()}) if col in rows else []
            self.vocab_[col] = tuple(dict.fromkeys([*fixed, *seen]))
        conf = rows["personal_conf"].dropna() if "personal_conf" in rows else []
        self.fill_conf_ = float(np.mean(conf)) if len(conf) else 0.5
        return self

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric)
        for col in self.categorical:  # fixed declaration order (a sorted-key JSON round trip must not reorder columns)
            names.extend(f"{col}={v}" for v in self.vocab_.get(col, ()))
        return names

    def transform(self, rows: Any) -> np.ndarray:
        n = len(rows)
        conf = rows["personal_conf"].to_numpy(dtype=object)
        filled = np.asarray([self.fill_conf_ if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in conf], dtype=np.float64)
        cols = []
        for name in self.numeric:
            if name == "personal_conf_filled":
                cols.append(filled)
            elif name in rows:
                cols.append(np.asarray([_num(v) for v in rows[name].tolist()], dtype=np.float64))
            else:
                cols.append(np.zeros(n))
        for col in self.categorical:
            vocab = self.vocab_.get(col, ())
            values = [str(v) for v in rows[col].tolist()] if col in rows else ["OTHER"] * n
            for v in vocab:
                cols.append(np.asarray([1.0 if x == v else 0.0 for x in values]))
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    def to_dict(self) -> dict[str, Any]:
        return {"numeric": list(self.numeric), "vocab": {k: list(v) for k, v in self.vocab_.items()}, "fill_conf": self.fill_conf_}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservableEncoder":
        enc = cls(numeric=tuple(data["numeric"]))
        enc.vocab_ = {k: tuple(v) for k, v in data["vocab"].items()}
        enc.fill_conf_ = float(data["fill_conf"])
        return enc


_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def _cached_snapshot(hf_id: str) -> Path | None:
    """The single cached snapshot directory of ``hf_id`` under ``$HF_HOME/hub`` (offline pin)."""
    home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    root = home / "hub" / ("models--" + hf_id.replace("/", "--")) / "snapshots"
    if not root.is_dir():
        return None
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    return snaps[0] if len(snaps) == 1 else None


@dataclass
class TextFeaturizer:
    """Report-text embedding: pinned sentence-transformers (offline cache) or fold-fitted
    TF-IDF (sublinear tf, l2) + TruncatedSVD; ``kind`` records which one was used."""

    dims: int = TEXT_SVD_DIMS
    prefer: str = "auto"  # "auto" | "tfidf" | "sentence-transformers"
    max_features: int = 20000
    kind: str = ""
    model: str | None = None
    revision: str | None = None
    vocab_: dict[str, int] = field(default_factory=dict)
    idf_: np.ndarray | None = None
    svd_components_: np.ndarray | None = None
    _st_model: Any = None
    _st_cache: dict[str, np.ndarray] = field(default_factory=dict)

    # --- sentence-transformers -------------------------------------------------
    def _try_st(self) -> bool:
        if self.prefer == "tfidf":
            return False
        snapshot = _cached_snapshot(PINNED_TEXT_MODEL)
        if snapshot is None:
            return False
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True)
        except Exception:
            return False
        self.kind, self.model, self.revision = "sentence-transformers", PINNED_TEXT_MODEL, snapshot.name
        return True

    def _st_embed(self, texts: Sequence[str]) -> np.ndarray:
        todo = [t for t in texts if t not in self._st_cache]
        if todo:
            vecs = self._st_model.encode(todo, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(todo, vecs):
                self._st_cache[t] = np.asarray(v, dtype=np.float64)
        return np.stack([self._st_cache[t] for t in texts]) if texts else np.zeros((0, 0))

    # --- tf-idf ----------------------------------------------------------------
    def _counts(self, texts: Sequence[str]) -> Any:
        from sklearn.feature_extraction.text import CountVectorizer

        cv = CountVectorizer(vocabulary=self.vocab_, lowercase=True, token_pattern=_TOKEN_RE.pattern, ngram_range=(1, 2))
        return cv.transform(list(texts))

    def _tfidf(self, texts: Sequence[str]) -> Any:
        from sklearn.preprocessing import normalize

        counts = self._counts(texts).astype(np.float64)
        counts.data = 1.0 + np.log(counts.data)
        tf = counts.multiply(self.idf_[None, :]).tocsr()
        return normalize(tf, norm="l2", copy=False)

    def fit(self, texts: Sequence[str]) -> "TextFeaturizer":
        texts = [str(t) for t in texts]
        if self.prefer in ("auto", "sentence-transformers") and self._try_st():
            return self
        if self.prefer == "sentence-transformers":
            raise ReadoutError(f"{PINNED_TEXT_MODEL} is not in the offline HF cache")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(lowercase=True, token_pattern=_TOKEN_RE.pattern, ngram_range=(1, 2), sublinear_tf=True, norm="l2",
                              smooth_idf=True, max_features=self.max_features, dtype=np.float64)
        try:
            matrix = vec.fit_transform(texts)
        except ValueError:  # empty vocabulary
            self.kind, self.vocab_, self.idf_, self.svd_components_ = "tfidf_svd", {}, np.zeros(0), np.zeros((0, 0))
            return self
        self.vocab_ = {str(k): int(v) for k, v in vec.vocabulary_.items()}
        self.idf_ = np.asarray(vec.idf_, dtype=np.float64)
        k = int(min(self.dims, max(1, min(matrix.shape) - 1)))
        if k >= 1 and matrix.shape[0] >= 2 and matrix.shape[1] >= 2:
            svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=7, random_state=0)
            svd.fit(matrix)
            self.svd_components_ = np.asarray(svd.components_, dtype=np.float64)
        else:
            self.svd_components_ = np.zeros((0, matrix.shape[1]))
        self.kind, self.model, self.revision = "tfidf_svd", None, None
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        if self.kind == "sentence-transformers":
            return self._st_embed(texts)
        if self.svd_components_ is None or self.svd_components_.shape[0] == 0:
            return np.zeros((len(texts), 0))
        return np.asarray(self._tfidf(texts) @ self.svd_components_.T)

    @property
    def n_features(self) -> int:
        if self.kind == "sentence-transformers":
            return int(self._st_model.get_sentence_embedding_dimension())
        return 0 if self.svd_components_ is None else int(self.svd_components_.shape[0])

    def state(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        meta = {"kind": self.kind, "dims": self.dims, "model": self.model, "revision": self.revision, "n_features": self.n_features}
        arrays: dict[str, np.ndarray] = {}
        if self.kind == "tfidf_svd":
            vocab = sorted(self.vocab_.items(), key=lambda kv: kv[1])
            arrays["text_vocab"] = np.asarray([k for k, _ in vocab], dtype=str)
            arrays["text_idf"] = np.asarray(self.idf_, dtype=np.float64)
            arrays["text_svd"] = np.asarray(self.svd_components_, dtype=np.float64)
        return meta, arrays

    @classmethod
    def from_state(cls, meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> "TextFeaturizer":
        tf = cls(dims=int(meta["dims"]), kind=str(meta["kind"]), model=meta.get("model"), revision=meta.get("revision"))
        if tf.kind == "tfidf_svd":
            tf.vocab_ = {str(k): i for i, k in enumerate(arrays["text_vocab"].astype(str).tolist())}
            tf.idf_ = np.asarray(arrays["text_idf"], dtype=np.float64)
            tf.svd_components_ = np.asarray(arrays["text_svd"], dtype=np.float64)
        elif tf.kind == "sentence-transformers":
            if not tf._try_st() or tf.revision != meta.get("revision"):
                raise ReadoutError("the frozen sentence-transformers pin is not the cached snapshot")
        return tf


@dataclass
class NeuralFeaturizer:
    """Fold-fitted PCA projections (rank clipped to the training rank) + norm statistics."""

    rank: int
    pca: G.DevPCA | None = None
    train_mean_: np.ndarray | None = None

    def fit(self, H: np.ndarray) -> "NeuralFeaturizer":
        H = np.asarray(H, dtype=np.float64)
        self.pca = G.DevPCA(rank_requested=int(self.rank), label=f"G_pca_r{self.rank}").fit(H)
        self.train_mean_ = H.mean(axis=0)
        return self

    @property
    def effective_rank(self) -> int:
        return 0 if self.pca is None else self.pca.rank

    @property
    def feature_names(self) -> list[str]:
        return [f"h_pc{i}" for i in range(self.effective_rank)] + list(NORM_STAT_NAMES)

    def transform(self, H: np.ndarray) -> np.ndarray:
        if self.pca is None or self.train_mean_ is None:
            raise ReadoutError("neural featurizer is not fitted")
        H = np.asarray(H, dtype=np.float64)
        proj = self.pca.transform(H)
        norm = np.linalg.norm(H, axis=1)
        centered = np.linalg.norm(H - self.train_mean_, axis=1)
        return np.column_stack([proj, norm, np.log(norm + _EPS), centered])

    @property
    def n_features(self) -> int:
        return self.effective_rank + len(NORM_STAT_NAMES)


@dataclass
class TextExpansion:
    """The parameter-count-matched text expansion: ``extra`` more TF-IDF-SVD dimensions
    fitted in the fold, appended to the baseline text block (same count as the neural block)."""

    extra: int
    featurizer: TextFeaturizer | None = None

    def fit(self, texts: Sequence[str], base: TextFeaturizer) -> "TextExpansion":
        # a disjoint SVD basis: components ``n_base .. n_base+extra`` of a larger TF-IDF-SVD
        total = base.n_features + int(self.extra) if base.kind == "tfidf_svd" else int(self.extra)
        tf = TextFeaturizer(dims=total, prefer="tfidf").fit(texts)
        if base.kind == "tfidf_svd" and tf.svd_components_ is not None and tf.svd_components_.shape[0] > base.n_features:
            tf.svd_components_ = tf.svd_components_[base.n_features:]
        self.featurizer = tf
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.featurizer is None:
            raise ReadoutError("text expansion is not fitted")
        out = self.featurizer.transform(texts)
        return out[:, : int(self.extra)]


# --------------------------------------------------------------------------- model


@dataclass
class LogisticModel:
    """Standard-scaled L2 logistic regression (penalty λ → sklearn ``C = 1/λ``)."""

    penalty: float
    feature_names: list[str] = field(default_factory=list)
    scale_mean_: np.ndarray | None = None
    scale_std_: np.ndarray | None = None
    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    constant_: float | None = None  # single-class training fold → constant probability

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.scale_mean_ = X.mean(axis=0) if X.shape[1] else np.zeros(0)
        std = X.std(axis=0) if X.shape[1] else np.zeros(0)
        self.scale_std_ = np.where(std > _EPS, std, 1.0)
        if len(np.unique(y)) < 2 or X.shape[1] == 0:
            self.constant_ = float(np.clip((y.sum() + 0.5) / (len(y) + 1.0), _EPS, 1 - _EPS))
            self.coef_ = np.zeros(X.shape[1])
            return self
        Z = (X - self.scale_mean_) / self.scale_std_
        clf = LogisticRegression(C=1.0 / float(self.penalty), solver="lbfgs", max_iter=5000, tol=1e-8)  # L2 (sklearn default; `penalty=` is deprecated in 1.8)
        clf.fit(Z, y)
        self.coef_ = np.asarray(clf.coef_[0], dtype=np.float64)
        self.intercept_ = float(clf.intercept_[0])
        self.constant_ = None
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.constant_ is not None:
            return np.full(X.shape[0], self.constant_)
        Z = (X - self.scale_mean_) / self.scale_std_
        logits = Z @ self.coef_ + self.intercept_
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

    def to_dict(self) -> dict[str, Any]:
        return {"penalty": self.penalty, "feature_names": list(self.feature_names), "scale_mean": self.scale_mean_.tolist(),
                "scale_std": self.scale_std_.tolist(), "coef": self.coef_.tolist(), "intercept": self.intercept_, "constant": self.constant_}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LogisticModel":
        m = cls(penalty=float(data["penalty"]), feature_names=list(data["feature_names"]))
        m.scale_mean_ = np.asarray(data["scale_mean"], dtype=np.float64)
        m.scale_std_ = np.asarray(data["scale_std"], dtype=np.float64)
        m.coef_ = np.asarray(data["coef"], dtype=np.float64)
        m.intercept_ = float(data["intercept"])
        m.constant_ = None if data.get("constant") is None else float(data["constant"])
        return m


# --------------------------------------------------------------------------- fold features


@dataclass
class FoldFeatures:
    """Everything fitted on one training set, applied to a held-out set."""

    obs: ObservableEncoder
    text: TextFeaturizer
    neural: dict[tuple[int, int], NeuralFeaturizer]  # (block, rank) → featurizer
    expansion: dict[int, TextExpansion] = field(default_factory=dict)  # extra dims → expansion

    def matrix(self, frames: FeatureFrames, index: np.ndarray, *, variant: str, block: int | None = None, rank: int | None = None,
               extra: int | None = None) -> tuple[np.ndarray, list[str]]:
        rows = frames.rows.iloc[index]
        parts = [self.obs.transform(rows)]
        names = list(self.obs.feature_names)
        if variant == "observables":
            return np.column_stack(parts), names
        T = self.text.transform(rows["text"].tolist())
        parts.append(T)
        names += [f"text_{i}" for i in range(T.shape[1])]
        if variant == "augmented":
            nf = self.neural[(int(block), int(rank))]
            parts.append(nf.transform(frames.h(int(block), index)))
            names += nf.feature_names
        elif variant == "text_expanded":
            E = self.expansion[int(extra)].transform(rows["text"].tolist())
            parts.append(E)
            names += [f"text_x{i}" for i in range(E.shape[1])]
        elif variant != "baseline":
            raise ValueError(variant)
        return np.column_stack(parts), names


def fit_fold_features(frames: FeatureFrames, train: np.ndarray, *, blocks: Sequence[int], ranks: Sequence[int], text_prefer: str = "auto",
                      expansions: Sequence[int] = (), text_dims: int = TEXT_SVD_DIMS) -> FoldFeatures:
    rows = frames.rows.iloc[train]
    obs = ObservableEncoder().fit(rows)
    text = TextFeaturizer(dims=int(text_dims), prefer=text_prefer).fit(rows["text"].tolist())
    neural: dict[tuple[int, int], NeuralFeaturizer] = {}
    for b in blocks:
        H = frames.h(int(b), train)
        for r in ranks:
            neural[(int(b), int(r))] = NeuralFeaturizer(rank=int(r)).fit(H)
    ff = FoldFeatures(obs=obs, text=text, neural=neural)
    for extra in expansions:
        ff.expansion[int(extra)] = TextExpansion(extra=int(extra)).fit(rows["text"].tolist(), text)
    return ff


# --------------------------------------------------------------------------- selection


@dataclass(frozen=True)
class GConfig:
    blocks: tuple[int, ...]
    ranks: tuple[int, ...] = RANK_GRID
    penalties: tuple[float, ...] = PENALTY_GRID
    n_outer: int = N_OUTER_FOLDS
    n_inner: int = N_INNER_FOLDS
    text_prefer: str = "auto"
    seed: bytes = b"\x00" * 32
    text_dims: int = TEXT_SVD_DIMS  # frozen at 64 (brief R1); smaller only in tests


@dataclass(frozen=True)
class Selection:
    block: int | None
    rank: int | None
    penalty: float
    inner_mean: float
    inner_se: float
    n_effective_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def one_se_select(candidates: Sequence[tuple[tuple[int | None, int | None, float], float, float]], *, block_order: Sequence[int]) -> Selection:
    """One-SE rule: among configurations whose inner mean Brier is within one SE of the
    minimum, prefer the strongest penalty, then the shallowest block, then the smallest rank."""
    if not candidates:
        raise ReadoutError("no candidate configurations")
    best_mean, best_se = min((m, se) for _, m, se in candidates)
    threshold = best_mean + best_se
    depth = {int(b): i for i, b in enumerate(sorted(int(x) for x in block_order))}
    eligible = [(cfg, m, se) for cfg, m, se in candidates if m <= threshold + 1e-15]
    eligible.sort(key=lambda c: (-c[0][2], depth.get(c[0][0], -1) if c[0][0] is not None else -1, c[0][1] if c[0][1] is not None else -1, c[1]))
    (block, rank, penalty), m, se = eligible[0]
    return Selection(block=block, rank=rank, penalty=float(penalty), inner_mean=float(m), inner_se=float(se))


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0


def inner_select(frames: FeatureFrames, train: np.ndarray, cfg: GConfig, *, fold_seed_tag: str) -> dict[str, Selection]:
    """Inner grouped CV on ``train``: selects the augmented (block, rank, penalty), the
    baseline penalty, the observables-only penalty and (parameter-matched) the text-expansion
    penalty, all from the same inner folds."""
    rows = frames.rows.iloc[train]
    y = frames.y[train]
    inner = grouped_stratified_folds(rows["cluster"].tolist(), rows["superdomain"].tolist(), cfg.n_inner, cfg.seed, namespace=f"{FOLD_NAMESPACE}:inner:{fold_seed_tag}")
    per: dict[tuple[str, tuple[Any, ...]], list[float]] = {}
    eff_rank: dict[tuple[int, int], int] = {}
    max_extra = max(cfg.ranks) + len(NORM_STAT_NAMES)
    extras = sorted({int(r) + len(NORM_STAT_NAMES) for r in cfg.ranks} | {max_extra})
    for j in range(cfg.n_inner):
        tr, va = train[inner != j], train[inner == j]
        if len(va) == 0 or len(tr) < 3:
            continue
        ff = fit_fold_features(frames, tr, blocks=cfg.blocks, ranks=cfg.ranks, text_prefer=cfg.text_prefer, expansions=extras, text_dims=cfg.text_dims)
        y_tr, y_va = frames.y[tr], frames.y[va]
        for variant in ("observables", "baseline"):
            Xtr, _ = ff.matrix(frames, tr, variant=variant)
            Xva, _ = ff.matrix(frames, va, variant=variant)
            for lam in cfg.penalties:
                p = LogisticModel(lam).fit(Xtr, y_tr).predict_proba(Xva)
                per.setdefault((variant, (None, None, lam)), []).append(float(brier(p, y_va).mean()))
        for b in cfg.blocks:
            for r in cfg.ranks:
                eff_rank[(int(b), int(r))] = min(eff_rank.get((int(b), int(r)), 10**9), ff.neural[(int(b), int(r))].effective_rank)
                Xtr, _ = ff.matrix(frames, tr, variant="augmented", block=b, rank=r)
                Xva, _ = ff.matrix(frames, va, variant="augmented", block=b, rank=r)
                for lam in cfg.penalties:
                    p = LogisticModel(lam).fit(Xtr, y_tr).predict_proba(Xva)
                    per.setdefault(("augmented", (int(b), int(r), lam)), []).append(float(brier(p, y_va).mean()))
        for extra in extras:
            Xtr, _ = ff.matrix(frames, tr, variant="text_expanded", extra=extra)
            Xva, _ = ff.matrix(frames, va, variant="text_expanded", extra=extra)
            for lam in cfg.penalties:
                p = LogisticModel(lam).fit(Xtr, y_tr).predict_proba(Xva)
                per.setdefault(("text_expanded", (None, extra, lam)), []).append(float(brier(p, y_va).mean()))
    out: dict[str, Selection] = {}
    for variant in ("observables", "baseline", "augmented"):
        cands = [(cfg_key, *_mean_se(v)) for (var, cfg_key), v in per.items() if var == variant]
        out[variant] = one_se_select(cands, block_order=cfg.blocks)
    sel = out["augmented"]
    extra = int(eff_rank.get((sel.block, sel.rank), sel.rank)) + len(NORM_STAT_NAMES)
    out["augmented"] = dataclasses.replace(sel, n_effective_rank=int(eff_rank.get((sel.block, sel.rank), sel.rank)))
    cands = [(cfg_key, *_mean_se(v)) for (var, cfg_key), v in per.items() if var == "text_expanded" and cfg_key[1] == extra]
    if not cands:  # the matched count was not on the grid (clipped rank): use the nearest
        nearest = min(extras, key=lambda e: abs(e - extra))
        cands = [(cfg_key, *_mean_se(v)) for (var, cfg_key), v in per.items() if var == "text_expanded" and cfg_key[1] == nearest]
    out["text_expanded"] = one_se_select(cands, block_order=cfg.blocks)
    return out


def fit_variants(frames: FeatureFrames, train: np.ndarray, selections: Mapping[str, Selection], cfg: GConfig) -> tuple[FoldFeatures, dict[str, LogisticModel]]:
    sel = selections["augmented"]
    extra = int(selections["text_expanded"].rank)
    ff = fit_fold_features(frames, train, blocks=(int(sel.block),), ranks=(int(sel.rank),), text_prefer=cfg.text_prefer, expansions=(extra,), text_dims=cfg.text_dims)
    y = frames.y[train]
    models: dict[str, LogisticModel] = {}
    for variant in VARIANTS:
        X, names = _variant_matrix(ff, frames, train, variant, selections)
        m = LogisticModel(float(selections[variant].penalty), feature_names=names).fit(X, y)
        models[variant] = m
    return ff, models


def _variant_matrix(ff: FoldFeatures, frames: FeatureFrames, index: np.ndarray, variant: str, selections: Mapping[str, Selection]) -> tuple[np.ndarray, list[str]]:
    sel = selections["augmented"]
    if variant == "augmented":
        return ff.matrix(frames, index, variant=variant, block=sel.block, rank=sel.rank)
    if variant == "text_expanded":
        return ff.matrix(frames, index, variant=variant, extra=int(selections["text_expanded"].rank))
    return ff.matrix(frames, index, variant=variant)


def predict_variants(ff: FoldFeatures, models: Mapping[str, LogisticModel], frames: FeatureFrames, index: np.ndarray, selections: Mapping[str, Selection]) -> dict[str, np.ndarray]:
    out = {}
    for variant, m in models.items():
        X, _ = _variant_matrix(ff, frames, index, variant, selections)
        out[variant] = m.predict_proba(X)
    return out


# --------------------------------------------------------------------------- nested CV


@dataclass
class DevResult:
    losses: Any  # pandas: one row per (item, method) with p_<variant>, brier_<variant>, outer_fold
    selections: list[dict[str, Any]]
    summary: dict[str, Any]


def nested_cv(frames: FeatureFrames, cfg: GConfig) -> DevResult:
    """§8.8 nested five-outer/five-inner source-grouped, superdomain-stratified CV."""
    import pandas as pd

    rows = frames.rows
    if len(rows) == 0:
        raise ReadoutError("no development rows")
    y = frames.y
    outer = grouped_stratified_folds(rows["cluster"].tolist(), rows["superdomain"].tolist(), cfg.n_outer, cfg.seed, namespace=f"{FOLD_NAMESPACE}:outer")
    preds = {v: np.full(len(rows), np.nan) for v in VARIANTS}
    fold_records = []
    all_index = np.arange(len(rows))
    for k in range(cfg.n_outer):
        train, test = all_index[outer != k], all_index[outer == k]
        if len(test) == 0:
            continue
        selections = inner_select(frames, train, cfg, fold_seed_tag=f"outer{k}")
        ff, models = fit_variants(frames, train, selections, cfg)
        p = predict_variants(ff, models, frames, test, selections)
        for v in VARIANTS:
            preds[v][test] = p[v]
        fold_records.append({"outer_fold": k, "n_train": int(len(train)), "n_test": int(len(test)), "text_kind": ff.text.kind,
                             **{f"{v}_selection": selections[v].to_dict() for v in VARIANTS}})
    losses = rows[["source_id", "method", "superdomain", "cluster", "report_id", "selection_id"]].copy()
    losses["y"] = y
    losses["outer_fold"] = outer
    for v in VARIANTS:
        losses[f"p_{v}"] = preds[v]
        losses[f"brier_{v}"] = brier(preds[v], y)
    summary = summarize_losses(losses)
    summary["folds"] = fold_records
    summary["n_rows"], summary["n_items"] = int(len(losses)), int(losses["source_id"].nunique())
    return DevResult(losses=losses, selections=fold_records, summary=summary)


def item_weights(superdomain: Sequence[str]) -> np.ndarray:
    """Equal item weights within a superdomain, 0.5/0.5 across the two (§9.1)."""
    sd = np.asarray([str(s) for s in superdomain])
    w = np.zeros(len(sd))
    present = [d for d in sorted(set(sd)) if np.sum(sd == d) > 0]
    for d in present:
        w[sd == d] = (1.0 / len(present)) / float(np.sum(sd == d))
    return w


def _log(message: str) -> None:
    print(f"[neural.readout] {message}", file=sys.stderr, flush=True)


def method_coverage(losses: Any) -> dict[str, Any]:
    """Per-item method coverage of a losses table: the method set of the table, the items
    that carry every method exactly once, and the items dropped for unequal weights (P1-6)."""
    if len(losses) == 0:
        return {"methods": [], "n_items_total": 0, "n_items_complete": 0, "n_items_dropped": 0, "dropped_items": [], "complete_items": []}
    methods = sorted(str(m) for m in losses["method"].unique())
    per_item = losses.groupby("source_id")["method"].agg(lambda m: sorted(str(x) for x in m))
    complete = sorted(str(sid) for sid, ms in per_item.items() if list(ms) == methods)
    complete_set = set(complete)
    dropped = sorted(str(sid) for sid in per_item.index if str(sid) not in complete_set)
    return {"methods": methods, "n_items_total": int(len(per_item)), "n_items_complete": len(complete), "n_items_dropped": len(dropped),
            "dropped_items": dropped[:50], "complete_items": complete}


def per_item_losses(losses: Any, columns: Sequence[str], *, require_complete: bool = True) -> Any:
    """Average each loss column over the methods of an item (equal method weights).

    With ``require_complete`` (the default) an item enters only when it carries every method
    of the table exactly once, so no item is weighted by fewer methods than the others; the
    dropped items are listed by :func:`method_coverage`, never silently averaged (P1-6).
    """
    keep = ["source_id", "superdomain", "cluster", *columns]
    sub = losses
    if require_complete:
        cov = method_coverage(losses)
        if cov["n_items_dropped"]:
            sub = losses[losses["source_id"].astype(str).isin(set(cov["complete_items"]))]
    return sub[keep].groupby(["source_id", "superdomain", "cluster"], as_index=False).mean()


def summarize_losses(losses: Any) -> dict[str, Any]:
    items = per_item_losses(losses, [f"brier_{v}" for v in VARIANTS if f"brier_{v}" in losses])
    w = item_weights(items["superdomain"].tolist())
    out: dict[str, Any] = {"weighted_brier": {}, "unweighted_brier": {}, "by_method": {}, "by_superdomain": {},
                           "method_coverage": {k: v for k, v in method_coverage(losses).items() if k != "complete_items"}}
    for v in VARIANTS:
        col = f"brier_{v}"
        if col not in items:
            continue
        out["weighted_brier"][v] = float(np.sum(w * items[col].to_numpy()))
        out["unweighted_brier"][v] = float(items[col].mean())
    for m, sub in losses.groupby("method"):
        out["by_method"][str(m)] = {v: float(sub[f"brier_{v}"].mean()) for v in VARIANTS if f"brier_{v}" in sub}
    for d, sub in items.groupby("superdomain"):
        out["by_superdomain"][str(d)] = {v: float(sub[f"brier_{v}"].mean()) for v in VARIANTS if f"brier_{v}" in sub}
    if "brier_baseline" in items and "brier_augmented" in items:
        out["contrast_baseline_minus_augmented"] = float(np.sum(w * (items["brier_baseline"] - items["brier_augmented"]).to_numpy()))
    if "brier_text_expanded" in items and "brier_augmented" in items:
        out["contrast_text_expanded_minus_augmented"] = float(np.sum(w * (items["brier_text_expanded"] - items["brier_augmented"]).to_numpy()))
    return out


# --------------------------------------------------------------------------- freeze


@dataclass
class FrozenG:
    """The development-frozen predictors (all variants) + transforms; immutable by hash."""

    selections: dict[str, Selection]
    features: FoldFeatures
    models: dict[str, LogisticModel]
    meta: dict[str, Any] = field(default_factory=dict)

    def predict(self, frames: FeatureFrames, index: np.ndarray | None = None) -> dict[str, np.ndarray]:
        idx = np.arange(len(frames.rows)) if index is None else np.asarray(index)
        return predict_variants(self.features, self.models, frames, idx, self.selections)

    # --- serialization -----------------------------------------------------------
    def save(self, directory: str | os.PathLike) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sel = self.selections["augmented"]
        nf = self.features.neural[(int(sel.block), int(sel.rank))]
        text_meta, arrays = self.features.text.state()
        arrays["pca_mean"] = nf.pca.mean_
        arrays["pca_components"] = nf.pca.components_
        arrays["pca_explained_variance"] = nf.pca.explained_variance_
        arrays["neural_train_mean"] = nf.train_mean_
        exp = self.features.expansion.get(int(self.selections["text_expanded"].rank))
        if exp is not None and exp.featurizer is not None:
            emeta, earrays = exp.featurizer.state()
            for k, v in earrays.items():
                arrays["expansion_" + k] = v
        else:
            emeta = {}
        npz_path = directory / G_FROZEN_NPZ
        np.savez(npz_path, **arrays)
        npz_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        payload = {
            "schema_version": 1,
            "kind": "G_FROZEN",
            "selected": {"block": sel.block, "rank": sel.rank, "effective_rank": nf.effective_rank, "penalty": sel.penalty,
                         "inner_brier_mean": sel.inner_mean, "inner_brier_se": sel.inner_se},
            "selections": {v: s.to_dict() for v, s in self.selections.items()},
            "models": {v: m.to_dict() for v, m in self.models.items()},
            "feature_lists": {v: list(m.feature_names) for v, m in self.models.items()},
            "observables": self.features.obs.to_dict(),
            "text": text_meta,
            "text_expansion": {"extra": int(self.selections["text_expanded"].rank), **emeta},
            "pca": {"rank_requested": nf.pca.rank_requested, "rank": nf.pca.rank, "clipped": nf.pca.clipped, "n_fit": nf.pca.n_fit_,
                    "d": nf.pca.d_, "components_hash": nf.pca.components_hash, "label": nf.pca.label},
            "arrays_file": G_FROZEN_NPZ,
            "arrays_sha256": npz_sha,
            "meta": dict(self.meta),
        }
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
        payload["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
        json_path = directory / G_FROZEN_JSON
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        return json_path, npz_path

    @classmethod
    def load(cls, directory: str | os.PathLike) -> "FrozenG":
        directory = Path(directory)
        json_path, npz_path = directory / G_FROZEN_JSON, directory / G_FROZEN_NPZ
        if not json_path.is_file() or not npz_path.is_file():
            raise ReadoutError(f"frozen G artifacts missing under {directory}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        claimed = payload.pop("sha256", None)
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != claimed:
            raise ReadoutError(f"{json_path}: sha256 mismatch (frozen artifact was modified)")
        if hashlib.sha256(npz_path.read_bytes()).hexdigest() != payload["arrays_sha256"]:
            raise ReadoutError(f"{npz_path}: sha256 mismatch (frozen arrays were modified)")
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        selections = {v: Selection(**s) for v, s in payload["selections"].items()}
        sel = selections["augmented"]
        text = TextFeaturizer.from_state(payload["text"], arrays)
        nf = NeuralFeaturizer(rank=int(sel.rank))
        pca_meta = payload["pca"]
        nf.pca = G.DevPCA.from_dict({**pca_meta, "mean": arrays["pca_mean"], "components": arrays["pca_components"],
                                     "explained_variance": arrays["pca_explained_variance"], "extra": {}})
        nf.train_mean_ = np.asarray(arrays["neural_train_mean"], dtype=np.float64)
        obs = ObservableEncoder.from_dict(payload["observables"])
        ff = FoldFeatures(obs=obs, text=text, neural={(int(sel.block), int(sel.rank)): nf})
        extra = int(payload["text_expansion"]["extra"])
        emeta = {k: v for k, v in payload["text_expansion"].items() if k != "extra"}
        if emeta.get("kind"):
            earrays = {k[len("expansion_"):]: v for k, v in arrays.items() if k.startswith("expansion_")}
            ff.expansion[extra] = TextExpansion(extra=extra, featurizer=TextFeaturizer.from_state(emeta, earrays))
        models = {v: LogisticModel.from_dict(m) for v, m in payload["models"].items()}
        return cls(selections=selections, features=ff, models=models, meta=dict(payload.get("meta") or {}))


def freeze(frames: FeatureFrames, cfg: GConfig, *, meta: Mapping[str, Any] | None = None) -> FrozenG:
    """Rerun the frozen selection procedure on all development items and fit the final maps."""
    all_index = np.arange(len(frames.rows))
    selections = inner_select(frames, all_index, cfg, fold_seed_tag="freeze")
    ff, models = fit_variants(frames, all_index, selections, cfg)
    info = {"n_dev_rows": int(len(all_index)), "n_dev_items": int(frames.rows["source_id"].nunique()), "blocks": list(cfg.blocks),
            "ranks": list(cfg.ranks), "penalties": list(cfg.penalties), "text_dims": int(cfg.text_dims), "text_kind": ff.text.kind, "text_model": ff.text.model,
            "text_revision": ff.text.revision, **dict(meta or {})}
    return FrozenG(selections=selections, features=ff, models=models, meta=info)


# --------------------------------------------------------------------------- bootstrap


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    se: float
    t: float
    p_one_sided: float
    p_two_sided: float
    ci_low: float
    ci_high: float
    n_resamples: int
    n_clusters: dict[str, int]
    weights: dict[str, float]
    seed: int
    estimation_only: bool
    degenerate_replicates: int
    status: str
    by_superdomain: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _ratio_and_se(S: np.ndarray, n: np.ndarray, counts: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Item-weighted mean ``sum S_c / sum n_c`` over clusters with resample multiplicities
    ``counts`` (``[B, K]``) and its linearized cluster SE."""
    if counts is None:
        counts = np.ones((1, S.size))
    K = S.size
    N = counts @ n
    theta = (counts @ S) / np.where(N > 0, N, np.nan)
    resid = S[None, :] - theta[:, None] * n[None, :]
    var = (counts * resid**2).sum(axis=1) / np.where(N > 0, N, np.nan) ** 2
    fpc = K / (K - 1) if K > 1 else float("nan")
    return theta, np.sqrt(var * fpc)


def paired_cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[str],
    superdomains: Sequence[str],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP,
    seed: int,
    alternative: str = "greater",
    superdomain_weights: Mapping[str, float] | None = None,
) -> BootstrapResult:
    """§9.3 source-cluster bootstrap of a paired per-item contrast.

    ``values`` are per-item paired differences (already averaged over methods); clusters are
    resampled with replacement independently within each superdomain; the estimate is the
    0.5/0.5 superdomain-weighted item mean; the p-value is null-centered studentized with the
    +1/+1 Monte Carlo correction; the interval is the 95% percentile interval.
    """
    import pandas as pd

    v = np.asarray(values, dtype=np.float64)
    df = pd.DataFrame({"v": v, "c": [str(c) for c in clusters], "d": [str(s) for s in superdomains]})
    domains = sorted(df["d"].unique())
    if superdomain_weights is None:
        superdomain_weights = {d: 1.0 / len(domains) for d in domains}
    wsum = sum(superdomain_weights.get(d, 0.0) for d in domains)
    weights = {d: superdomain_weights.get(d, 0.0) / wsum for d in domains}
    rng = np.random.default_rng(int(seed))
    theta_hat, var_hat = 0.0, 0.0
    theta_star = np.zeros(n_resamples)
    var_star = np.zeros(n_resamples)
    n_clusters: dict[str, int] = {}
    by_domain: dict[str, float] = {}
    for d in domains:
        sub = df[df["d"] == d].groupby("c")["v"].agg(["sum", "count"])
        S, n = sub["sum"].to_numpy(dtype=np.float64), sub["count"].to_numpy(dtype=np.float64)
        K = S.size
        n_clusters[d] = int(K)
        th, se = _ratio_and_se(S, n)
        by_domain[d] = float(th[0])
        theta_hat += weights[d] * float(th[0])
        var_hat += weights[d] ** 2 * (float(se[0]) ** 2 if np.isfinite(se[0]) else 0.0)
        counts = rng.multinomial(K, np.full(K, 1.0 / K), size=n_resamples).astype(np.float64)
        th_s, se_s = _ratio_and_se(S, n, counts)
        theta_star += weights[d] * th_s
        var_star += weights[d] ** 2 * np.where(np.isfinite(se_s), se_s**2, 0.0)
    se_hat = math.sqrt(var_hat)
    se_star = np.sqrt(var_star)
    total_clusters = int(sum(n_clusters.values()))
    estimation_only = total_clusters < MIN_CLUSTERS_FOR_INFERENCE or any(k < 2 for k in n_clusters.values())
    tol = _SE_TOL * max(1.0, abs(theta_hat), float(np.nanmax(np.abs(theta_star))) if np.any(np.isfinite(theta_star)) else 1.0)
    ok = np.isfinite(theta_star) & (se_star > tol)
    degenerate = int(np.sum(~ok))
    if se_hat <= tol or not np.isfinite(se_hat) or ok.sum() == 0:
        status = "degenerate"
        t_hat, p1, p2 = float("nan"), float("nan"), float("nan")
    else:
        status = "ok"
        t_hat = theta_hat / se_hat
        t_star = (theta_star[ok] - theta_hat) / se_star[ok]
        B = int(ok.sum())
        if alternative == "greater":
            p1 = (1.0 + float(np.sum(t_star >= t_hat))) / (B + 1.0)
        elif alternative == "less":
            p1 = (1.0 + float(np.sum(t_star <= t_hat))) / (B + 1.0)
        else:
            raise ValueError(alternative)
        p2 = (1.0 + float(np.sum(np.abs(t_star) >= abs(t_hat)))) / (B + 1.0)
    finite = theta_star[np.isfinite(theta_star)]
    lo, hi = (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))) if finite.size else (float("nan"), float("nan"))
    return BootstrapResult(estimate=float(theta_hat), se=float(se_hat), t=float(t_hat), p_one_sided=float(p1), p_two_sided=float(p2),
                           ci_low=lo, ci_high=hi, n_resamples=int(n_resamples), n_clusters=n_clusters, weights=weights, seed=int(seed),
                           estimation_only=bool(estimation_only), degenerate_replicates=degenerate, status=status, by_superdomain=by_domain)


def bootstrap_seed(study_seed: bytes, namespace: str, *parts: Any) -> int:
    """Manifest seed → numpy Generator seed (recorded in every summary)."""
    return int.from_bytes(identity.blind_order_key(study_seed, namespace, *parts)[:8], "big")


def confirmation_contrast(
    losses: Any,
    *,
    seed: int,
    n_resamples: int = DEFAULT_BOOTSTRAP,
    baseline: str = "baseline",
    augmented: str = "augmented",
) -> dict[str, Any]:
    """Family G contrast: paired per-item ``Brier(baseline) - Brier(augmented)`` (positive =
    the frozen internal features help), equal method weights, source-cluster bootstrap."""
    cols = [f"brier_{baseline}", f"brier_{augmented}"]
    coverage = method_coverage(losses)
    items = per_item_losses(losses, cols)
    if coverage["n_items_dropped"]:
        _log(f"contrast {baseline}-{augmented}: {coverage['n_items_dropped']} of {coverage['n_items_total']} items lack a method and are dropped from the paired contrast")
    diff = (items[cols[0]] - items[cols[1]]).to_numpy(dtype=np.float64)
    boot = paired_cluster_bootstrap(diff, items["cluster"].tolist(), items["superdomain"].tolist(), n_resamples=n_resamples, seed=seed, alternative="greater")
    out = {"contrast": f"{baseline}_minus_{augmented}", "bootstrap": boot.to_dict(), "n_items": int(len(items)), "n_rows": int(len(losses)),
           "n_items_complete": coverage["n_items_complete"], "n_items_dropped": coverage["n_items_dropped"], "dropped_items": coverage["dropped_items"],
           "methods": coverage["methods"], "losses": summarize_losses(losses)}
    per_method = {}
    for m, sub in losses.groupby("method"):
        d = (sub[cols[0]] - sub[cols[1]]).to_numpy(dtype=np.float64)
        per_method[str(m)] = {"mean": float(d.mean()), "n": int(len(d))}
    out["by_method"] = per_method
    return out


__all__ = [
    "ANSWER_FORMATS",
    "BOOTSTRAP_NAMESPACE",
    "DEFAULT_BOOTSTRAP",
    "FOLD_NAMESPACE",
    "G_FROZEN_JSON",
    "G_FROZEN_NPZ",
    "MIN_CLUSTERS_FOR_INFERENCE",
    "NORM_STAT_NAMES",
    "PENALTY_GRID",
    "PINNED_TEXT_MODEL",
    "RANK_GRID",
    "REPORT_METHODS",
    "STATE_ANCHOR",
    "TEXT_SVD_DIMS",
    "VARIANTS",
    "BootstrapResult",
    "DevResult",
    "FeatureFrames",
    "FoldFeatures",
    "FrozenG",
    "GConfig",
    "LogisticModel",
    "NeuralFeaturizer",
    "ObservableEncoder",
    "ReadoutError",
    "Selection",
    "TextExpansion",
    "TextFeaturizer",
    "assemble_rows",
    "bootstrap_seed",
    "brier",
    "build_feature_frames",
    "confirmation_contrast",
    "fit_fold_features",
    "fit_variants",
    "freeze",
    "grouped_stratified_folds",
    "inner_select",
    "item_weights",
    "load_forecasts",
    "load_labels",
    "load_renders",
    "load_task_tokens",
    "nested_cv",
    "one_se_select",
    "paired_cluster_bootstrap",
    "method_coverage",
    "per_item_losses",
    "predict_variants",
    "row_from_report",
    "select_split",
    "summarize_losses",
    "superdomain_of",
]
