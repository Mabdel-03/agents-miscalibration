"""E5 predictors P0, P1, P3, P2 (+ P1-conc, P2-conc) and the withheld-target guard (numpy/scipy only).

* P0: benchmark + compositional configuration (log n, log a/A, R, B/16A).
* P1: P0 + five-draw features + provider.     P1-conc: P1 without CONC.
* P3: ridge on P1 inputs plus pairwise products and squares.
* P2 (minimum executable): a glm mixed model on draw correctness (categorical depth, benchmark, features,
  log2(depth) x feature, Gauss-Hermite item intercept) -> per simulation one latent item effect shared by every call
  and depth -> predicted coverage G; a second glm for eta = P(delivered | covered) on covered multi-call development
  pools using out-of-fold response predictions; eta = 1 for singletons; P(delivered) = G * eta.
  P2-conc: the same without CONC.

Ridge strengths are chosen by problem-grouped 5-fold CV on the development frame only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import special

from agent_allocation.passk.features import CONC, LATENCY, OTHER
from agent_allocation.passk.models import glm as glm_mod
from agent_allocation.passk.models.inputs import BENCHMARKS, DEPTHS, PASSK_SEED

WITHHELD = frozenset({(4, 4), (8, 1), (4, 2), (2, 4)})
FENCED_PARTITIONS = frozenset({"v48_bank", "t200_bank"})
PREDICTORS = ("P0", "P1", "P2", "P3", "P1-conc", "P2-conc")
RIDGES = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
PROVIDERS = ("mancer", "parasail")


class WithheldTargetError(RuntimeError):
    pass


def guard_training_frame(frame: pd.DataFrame, *, draw_level: bool = False) -> None:
    """Raise on any withheld aggregate target, any fenced-bank row, or any t200_diag label column/row."""
    if not draw_level and {"n", "a"} <= set(frame.columns):
        bad = {(int(n), int(a)) for n, a in zip(frame.n, frame.a) if (int(n), int(a)) in WITHHELD}
        if bad:
            raise WithheldTargetError(f"withheld configurations in a training frame: {sorted(bad)}")
    for col in ("partition", "member_partitions"):
        if col in frame.columns:
            vals = frame[col]
            flat = set()
            for v in vals:
                if isinstance(v, (list, tuple, np.ndarray)):
                    flat |= set(map(str, v))
                else:
                    flat.add(str(v))
            if flat & FENCED_PARTITIONS:
                raise WithheldTargetError(f"fenced partitions in a training frame: {sorted(flat & FENCED_PARTITIONS)}")
            if "t200_diag" in flat and any(c in frame.columns for c in ("correct", "grade_state", "y", "coverage",
                                                                          "delivered")):
                raise WithheldTargetError("t200_diag labels in a training frame")
    for c in frame.columns:
        if c.startswith("t200_diag") and any(t in c for t in ("correct", "grade", "label")):
            raise WithheldTargetError(f"t200_diag label column {c!r} in a training frame")


def config_rank(frame: pd.DataFrame) -> int:
    M = np.column_stack([np.ones(len(frame)), np.log(frame.n.astype(float)), np.log(frame.a.astype(float))])
    return int(np.linalg.matrix_rank(M))


# ---------------------------------------------------------------------- designs
def feature_list(predictor: str, use_latency: bool) -> list[str]:
    if predictor == "P0":
        return []
    feats = ([] if predictor.endswith("-conc") else list(CONC)) + list(OTHER)
    if use_latency:
        feats += list(LATENCY)
    return feats


def ridge_design(frame: pd.DataFrame, predictor: str, use_latency: bool) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for b in BENCHMARKS[1:]:
        cols.append((frame.benchmark == b).to_numpy(float))
        names.append(f"bench:{b}")
    n = frame.n.to_numpy(float)
    a = frame.a.to_numpy(float)
    comp = {"log_n": np.log(n), "log_a": np.log(a), "R": np.ones(len(frame)), "B": n * a / 16.0}
    for k, v in comp.items():
        cols.append(v)
        names.append(k)
    base = predictor.replace("-conc", "")
    if base in ("P1", "P3"):
        cols.append((frame.provider.astype(str) == PROVIDERS[1]).to_numpy(float))
        names.append("provider:parasail")
        for f in feature_list(predictor, use_latency):
            cols.append(frame[f].to_numpy(float))
            names.append(f)
    X = np.column_stack(cols)
    if base == "P3":
        cont = [j for j, nm in enumerate(names) if not nm.startswith(("bench:", "provider:")) and nm != "R"]
        extra, en = [], []
        for i_, i in enumerate(cont):
            for j in cont[i_:]:
                extra.append(X[:, i] * X[:, j])
                en.append(f"{names[i]}*{names[j]}")
        X = np.column_stack([X] + extra)
        names = names + en
    return X, names


@dataclass
class RidgePredictor:
    predictor: str
    use_latency: bool
    fits: dict = field(default_factory=dict)      # target -> GLMFit
    cv: dict = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, *, seed: int = PASSK_SEED) -> "RidgePredictor":
        guard_training_frame(frame)
        X, names = ridge_design(frame, self.predictor, self.use_latency)
        for target in ("delivered", "coverage"):
            k = frame[target].notna().to_numpy()
            best, table = glm_mod.cv_ridge(X[k], frame[target].to_numpy(float)[k], frame.item_id.to_numpy()[k],
                                           item_intercept=False, ridges=RIDGES, seed=seed)
            self.fits[target] = glm_mod.fit_glm(X[k], frame[target].to_numpy(float)[k], None, item_intercept=False,
                                                ridge=best, columns=names)
            self.cv[target] = {"best_ridge": best, "table": table}
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X, _ = ridge_design(frame, self.predictor, self.use_latency)
        return pd.DataFrame({"p_delivered": self.fits["delivered"].predict_proba(X),
                             "p_coverage": self.fits["coverage"].predict_proba(X)}, index=frame.index)

    def to_dict(self) -> dict:
        return {"predictor": self.predictor, "use_latency": self.use_latency,
                "fits": {k: v.to_dict() for k, v in self.fits.items()}, "cv": self.cv}


# ---------------------------------------------------------------------- P2
def response_design(frame: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, list[str]]:
    """Draw-level: categorical depth x benchmark cells, features, log2(depth) x feature."""
    cols, names = [], []
    depth = frame.a.to_numpy(int)
    for b in BENCHMARKS:
        isb = (frame.benchmark == b).to_numpy(float)
        for d in DEPTHS:
            cols.append(isb * (depth == d))
            names.append(f"{b}:d{d}")
    ld = np.log2(depth.astype(float))
    for f in feats:
        v = frame[f].to_numpy(float)
        cols += [v, v * ld]
        names += [f, f"{f}*log2a"]
    return np.column_stack(cols), names


def eta_design(frame: pd.DataFrame, feats: list[str]) -> np.ndarray:
    logit_p = special.logit(np.clip(frame.p_call_oof.to_numpy(float), 1e-4, 1 - 1e-4))
    cols = [logit_p, np.log(frame.n.to_numpy(float)), np.log(frame.a.to_numpy(float))]
    cols += [(frame.benchmark == b).to_numpy(float) for b in BENCHMARKS[1:]]
    cols += [frame[f].to_numpy(float) for f in feats if f in CONC]
    return np.column_stack(cols)


@dataclass
class P2Predictor:
    predictor: str = "P2"
    use_latency: bool = False
    n_sim: int = 2000
    param_uncertainty: bool = True       # propagate Laplace draws of the fixed effects into every simulation
    response: glm_mod.GLMFit | None = None
    eta: glm_mod.GLMFit | None = None
    cv: dict = field(default_factory=dict)

    @property
    def feats(self) -> list[str]:
        return feature_list(self.predictor, self.use_latency)

    def fit(self, draw_frame: pd.DataFrame, pool_frame: pd.DataFrame, *, seed: int = PASSK_SEED,
            folds: int = 5) -> "P2Predictor":
        guard_training_frame(draw_frame, draw_level=True)
        guard_training_frame(pool_frame)
        X, names = response_design(draw_frame, self.feats)
        y = draw_frame.y.to_numpy(float)
        g = draw_frame.item_id.to_numpy(object)
        best, table = glm_mod.cv_ridge(X, y, g, item_intercept=True, ridges=(1.0, 3.0, 10.0, 30.0), seed=seed,
                                       gh_nodes=12)
        self.response = glm_mod.fit_glm(X, y, g, item_intercept=True, ridge=best, columns=names,
                                        laplace=self.param_uncertainty)
        self.cv["response"] = {"best_ridge": best, "table": table}
        # out-of-fold per-call predictions for the eta model (problem-grouped folds)
        fid = glm_mod.grouped_folds(g, folds, seed)
        item_fold = dict(zip(g, fid))
        pf = pool_frame.copy()
        pf["p_call_oof"] = np.nan
        for f in range(folds):
            tr = fid != f
            m = glm_mod.fit_glm(X[tr], y[tr], g[tr], item_intercept=True, ridge=best, gh_nodes=12)
            sel = pf.item_id.map(item_fold).to_numpy() == f
            if sel.any():
                Xp, _ = response_design(pf[sel], self.feats)
                pf.loc[sel, "p_call_oof"] = m.predict_proba(Xp)
        cov = pf[(pf.n > 1) & (pf.coverage == 1.0) & pf.delivered.notna() & pf.p_call_oof.notna()]
        Xe = eta_design(cov, self.feats)
        be, te = glm_mod.cv_ridge(Xe, cov.delivered.to_numpy(float), cov.item_id.to_numpy(), item_intercept=False,
                                  ridges=RIDGES, seed=seed)
        self.eta = glm_mod.fit_glm(Xe, cov.delivered.to_numpy(float), None, item_intercept=False, ridge=be)
        self.cv["eta"] = {"best_ridge": be, "table": te, "n_rows": int(len(cov))}
        return self

    def predict(self, frame: pd.DataFrame, *, seed: int = PASSK_SEED) -> pd.DataFrame:
        """frame: one row per (item, config) with n, a, benchmark and features."""
        rng = np.random.default_rng(seed)
        X, _ = response_design(frame, self.feats)
        if self.param_uncertainty and self.response.cov_std is not None:
            b0, coef, _ = self.response.sample_params(self.n_sim, rng)
            eta_sims = b0[:, None] + coef @ X.T                                      # (S, rows)
        else:
            eta_sims = np.broadcast_to(self.response.linear_predictor(X)[None, :], (self.n_sim, len(frame)))
        out_g = np.empty(len(frame))
        # one latent item effect per simulation, shared across every call and depth of an item
        for item, idx in frame.groupby("item_id").indices.items():
            b = self.response.sample_item_effects(self.n_sim, rng)
            p = special.expit(eta_sims[:, idx] + b[:, None])                         # (S, rows)
            n = frame.n.to_numpy(float)[idx]
            out_g[idx] = (1.0 - (1.0 - p) ** n[None, :]).mean(0)
        pcall = self.response.predict_proba(X)
        fe = frame.assign(p_call_oof=pcall)
        eta = np.where(frame.n.to_numpy() > 1, self.eta.predict_proba(eta_design(fe, self.feats)), 1.0)
        return pd.DataFrame({"p_delivered": out_g * eta, "p_coverage": out_g, "eta": eta, "p_call": pcall},
                            index=frame.index)

    def to_dict(self) -> dict:
        return {"predictor": self.predictor, "use_latency": self.use_latency, "n_sim": self.n_sim,
                "param_uncertainty": self.param_uncertainty,
                "response": self.response.to_dict(), "eta": self.eta.to_dict(), "cv": self.cv}
