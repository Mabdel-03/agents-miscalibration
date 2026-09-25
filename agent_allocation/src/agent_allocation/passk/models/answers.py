"""E2 answer-distribution model and pool simulator (numpy only).

Per C48 item, one class scope over its fitting draws only (all depths). Classes are labelled correct when the
majority of their known member labels are correct. A draw at depth a, in posterior simulation s, lands in:

* a correct class, with total mass p_a^s (from a response model); split over the observed correct classes by a
  Dirichlet over their counts (one unseen correct class when none was observed);
* "no vote" (invalid / unparseable output), mass (1 - p_a^s) * v^s with v ~ Beta(1/2 + #novote, 1/2 + #wrong voters);
* a wrong voting class, mass (1 - p_a^s) * (1 - v^s) split by Dirichlet(c_1, ..., c_K, alpha_res) over the observed
  wrong classes plus an explicit **unseen residual**. Two treatments of the residual:
  ``diffuse`` -- every residual draw is its own singleton class; ``concentrated`` -- all residual draws form one wrong
  class.
* ``competition=False`` (ablation "depth without class competition"): every wrong voting draw is a distinct singleton.

Class shares are depth-invariant within an item (only the correct mass depends on depth).

Pools of n calls at depth a are simulated by multinomial draws; plurality correctness is the tie-averaged modal
class outcome, i.e. the expectation of the passk seeded uniform tie rule (a seeded tie is uniform over tied modal
classes). Returns per-posterior-draw conditional probabilities (averaged over ``inner`` pool simulations).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agent_allocation.passk.models.inputs import ClassSource, label_value

ALPHA_RES = 1.0


@dataclass
class ItemClasses:
    item_id: str
    correct_counts: np.ndarray    # observed correct classes (support over fitting draws)
    wrong_counts: np.ndarray      # observed wrong voting classes
    n_novote: int
    n_wrong_voters: int

    @property
    def n_classes(self) -> int:
        return len(self.correct_counts) + len(self.wrong_counts)


def item_classes(fit_draws_item: pd.DataFrame, cls: ClassSource, *, scope_id: str | None = None,
                 depths: tuple[int, ...] | None = None) -> ItemClasses:
    d = fit_draws_item if depths is None else fit_draws_item[fit_draws_item.allowance_a.isin(depths)]
    d = d.sort_values(["allowance_a", "replicate"])
    item = str(d.item_id.iloc[0]) if len(d) else ""
    mem = list(d.candidate_id)
    classes = cls.classes(scope_id or f"e2fit:{item}", mem) if mem else {}
    labels = dict(zip(d.candidate_id, [label_value(v) for v in d["correct"]]))
    members: dict[str, list[float]] = {}
    novote = 0
    for c in mem:
        k = classes.get(c)
        if k is None:
            novote += 1
        else:
            members.setdefault(k, []).append(labels.get(c, np.nan))
    corr, wrong = [], []
    for k, labs in members.items():
        known = [v for v in labs if not np.isnan(v)]
        if not known:
            continue           # unknown class: excluded from the class model (tri-state)
        (corr if np.mean(known) > 0.5 else wrong).append(len(labs))
    return ItemClasses(item, np.array(corr, float), np.array(wrong, float), novote, int(sum(wrong)))


def simulate_pools(P_a: np.ndarray, ic: ItemClasses, n: int, rng: np.random.Generator, *, treatment: str = "diffuse",
                   competition: bool = True, inner: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Per posterior draw s: (P(plurality correct), P(covered)) for a pool of n calls at one depth.

    ``P_a`` is (S,) per-call correctness at that depth.
    """
    S = len(P_a)
    n_c = max(len(ic.correct_counts), 1)
    corr_alpha = ic.correct_counts if len(ic.correct_counts) else np.ones(1)
    wrong_alpha = ic.wrong_counts if (competition and len(ic.wrong_counts)) else np.zeros(0)
    # posterior draws of shares
    cshare = rng.dirichlet(corr_alpha, size=S) if n_c > 1 else np.ones((S, 1))
    v = rng.beta(0.5 + ic.n_novote, 0.5 + ic.n_wrong_voters, size=S)
    walpha = np.concatenate([wrong_alpha, [ALPHA_RES]])
    wshare = rng.dirichlet(walpha, size=S) if len(walpha) > 1 else np.ones((S, 1))
    pa = np.clip(P_a, 0.0, 1.0)
    probs = np.concatenate([
        pa[:, None] * cshare,
        ((1 - pa) * v)[:, None],
        ((1 - pa) * (1 - v))[:, None] * wshare,
    ], axis=1)
    probs = probs / probs.sum(1, keepdims=True)
    k_w = len(wrong_alpha)
    counts = rng.multinomial(n, probs, size=(inner, S))           # (inner, S, C)
    cc = counts[..., :n_c]
    wc = counts[..., n_c + 1:n_c + 1 + k_w]
    r = counts[..., -1]
    diffuse = (treatment == "diffuse") or not competition
    top_res = np.where(r > 0, 1, 0) if diffuse else r
    top = np.maximum(cc.max(-1), top_res)
    if k_w:
        top = np.maximum(top, wc.max(-1))
    tied_c = (cc == top[..., None]).sum(-1) * (top > 0)
    tied_w = np.zeros_like(top)
    if k_w:
        tied_w = tied_w + (wc == top[..., None]).sum(-1)
    if diffuse:
        tied_w = tied_w + np.where(top == 1, r, 0)
    else:
        tied_w = tied_w + ((r == top) & (r > 0))
    tied_w = tied_w * (top > 0)
    denom = tied_c + tied_w
    plur = np.where(denom > 0, tied_c / np.maximum(denom, 1), 0.0)
    cov = 1.0 - (1.0 - pa) ** n
    return plur.mean(0), cov
