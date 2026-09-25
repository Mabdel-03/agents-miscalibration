"""Seal-then-join guard, vendored from ``agents_scaling.nextdyn.analysis`` (origin commit 6498531).

Origin: ``src/agents_scaling/nextdyn/analysis.py`` (sha256 69a104e0...), ``LABEL_LIKE``,
``write_held_out_predictions`` (line 110) and ``join_labels`` (line 146). Provenance row in ``PROVENANCE.json``.

Changes from the origin (and nothing else):

* frames are pandas DataFrames written as parquet, not JSON row lists, and the hash registry is the caller's seal
  (``sealed/SEAL_predictions.json`` ``artifacts``) instead of ``predictions/MANIFEST.json``;
* ``LABEL_LIKE`` adds this study's outcome columns (``grade_state``, ``y``, ``delivered_correct``, ``S_known``,
  ``H_known``, ``tie_avg_lo``, ``tie_avg_hi``);
* the refusal exception is local (``LabelJoinRefused``, a ``RuntimeError``).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping

import pandas as pd

__all__ = ["LABEL_LIKE", "LabelJoinRefused", "join_sealed", "write_held_out_predictions"]

LABEL_LIKE = frozenset({
    "semantic_correct", "is_correct", "correct", "selected_correct", "selected_state",
    "gold_ref_hash", "u_reason", "grading_status", "oracle_lo", "oracle_hi", "state",
    "grade_state", "y", "delivered_correct", "S_known", "H_known", "tie_avg_lo", "tie_avg_hi",
})


class LabelJoinRefused(RuntimeError):
    """A label was about to be joined onto predictions that were never sealed, or that moved."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_held_out_predictions(path: str | os.PathLike, frame: pd.DataFrame) -> str:
    """Write a held-out prediction frame with **no** label-like column; returns its sha256."""
    offending = sorted(set(frame.columns) & LABEL_LIKE)
    if offending:
        raise LabelJoinRefused(f"held-out predictions {Path(path).name} carry label column(s) {offending}; "
                               "predictions are sealed before any label join")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return _sha256_file(path)


def join_sealed(path: str | os.PathLike, sealed_sha256: str | None, labels: pd.DataFrame, *,
                on: list[str] | tuple[str, ...], artifacts: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Join ``labels`` onto the sealed prediction file at ``path``, or refuse (unsealed, missing or moved)."""
    path = Path(path)
    if sealed_sha256 is None:
        raise LabelJoinRefused(f"no sealed hash for {path.name}: seal the predictions before joining labels")
    if not path.exists():
        raise LabelJoinRefused(f"sealed prediction file {path.name} is missing")
    digest = _sha256_file(path)
    if digest != sealed_sha256:
        raise LabelJoinRefused(f"predictions {path.name} changed after they were sealed "
                               f"({digest[:12]} != {sealed_sha256[:12]}); the label join is refused")
    return pd.read_parquet(path).merge(labels, on=list(on), how="left", validate="many_to_one")
