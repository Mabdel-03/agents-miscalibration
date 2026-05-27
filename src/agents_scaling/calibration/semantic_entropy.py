"""Semantic-entropy confidence for free-form answers (Kuhn et al. 2023).

Tiered clustering of sampled answers into 'meanings', then confidence = mass of the
largest cluster (and entropy over clusters as a secondary signal):

* Tier 1 (always): exact match after normalization (cheap, deterministic).
* Tier 2 (optional): embedding cosine similarity via sentence-transformers, merging
  clusters whose representatives are near-duplicates. Skipped gracefully if the package
  is unavailable, falling back to Tier-1 clusters.
"""

from __future__ import annotations

import math
from collections import Counter

from agents_scaling.benchmarks.grading import _normalize_numeric


def _normalize_answer(a: str) -> str:
    return _normalize_numeric(a).lower()


def cluster_answers(answers: list[str], use_embeddings: bool = False, sim_threshold: float = 0.85) -> list[list[int]]:
    """Cluster sampled answers by meaning; return lists of indices per cluster."""
    answers = [a for a in answers if a]
    if not answers:
        return []
    # Tier 1: exact normalized match.
    buckets: dict[str, list[int]] = {}
    for i, a in enumerate(answers):
        buckets.setdefault(_normalize_answer(a), []).append(i)
    clusters = list(buckets.values())

    if not use_embeddings or len(clusters) <= 1:
        return clusters

    # Tier 2: merge clusters whose representative strings are semantically close.
    try:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2")
        reps = [answers[c[0]] for c in clusters]
        emb = model.encode(reps, convert_to_tensor=True, normalize_embeddings=True)
        sim = util.cos_sim(emb, emb)
        merged: list[list[int]] = []
        used = set()
        for i in range(len(clusters)):
            if i in used:
                continue
            group = list(clusters[i])
            used.add(i)
            for j in range(i + 1, len(clusters)):
                if j not in used and float(sim[i][j]) >= sim_threshold:
                    group += clusters[j]
                    used.add(j)
            merged.append(group)
        return merged
    except Exception:
        # Embeddings unavailable -> Tier-1 clusters (logged by caller).
        return clusters


def semantic_entropy_conf(answers: list[str], use_embeddings: bool = False) -> tuple[float, float]:
    """Return (largest-cluster mass as confidence, normalized semantic entropy)."""
    clusters = cluster_answers(answers, use_embeddings=use_embeddings)
    n = sum(len(c) for c in clusters)
    if n == 0:
        return 0.0, 0.0
    sizes = [len(c) for c in clusters]
    conf = max(sizes) / n
    probs = [s / n for s in sizes]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(len(sizes)) if len(sizes) > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return conf, norm_entropy
