"""Redundancy R = mean pairwise cosine similarity of agent output embeddings.

Uses sentence-transformers if available; otherwise returns 0.0 and flags unavailability so
the runner can record that R was not computed (rather than silently reporting a fake 0).
"""

from __future__ import annotations

from itertools import combinations


def redundancy(texts: list[str]) -> tuple[float, bool]:
    """Return (mean pairwise cosine similarity, computed?). 0.0 if <2 texts or no embedder."""
    texts = [t for t in texts if t and t.strip()]
    if len(texts) < 2:
        return 0.0, False
    try:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        sims = [float(util.cos_sim(emb[i], emb[j])) for i, j in combinations(range(len(texts)), 2)]
        return (sum(sims) / len(sims) if sims else 0.0), True
    except Exception:
        return 0.0, False
