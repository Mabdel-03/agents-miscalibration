"""Seal verification, the held-out fence and the guarded label join (contract §5.6).

``verify_seal`` and ``blocked_node_ids`` are re-exports of :mod:`passk.grade_scope` (one implementation; the grader,
``extract`` and ``score`` all use it). ``verify_seal(run_root)`` returns the parsed ``sealed/SEAL_predictions.json`` only when every artifact it lists
still exists with the recorded sha256 (and the seal names a code commit); otherwise ``None``. A seal that is missing,
malformed or whose artifacts changed never verifies, so the fence cannot lift by accident.

``blocked_node_ids(paths, cfg)`` is the grader's skip set: every node id in the held-out partitions'
``manifests/jobs/<partition>.jsonl`` of that shard while no verifying seal exists.

``join_labels`` follows the nextdyn pattern (``agents_scaling/src/agents_scaling/nextdyn/analysis.py``
``write_held_out_predictions``/``join_labels``, vendored as ``_vendor/seal_join.py``; WP-M writes held-out
predictions with ``write_held_out_predictions``, which refuses label-like columns): predictions are written and
hashed first; labels are joined only against a verifying seal whose recorded prediction hash matches the file on
disk.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


import pandas as pd

from agent_allocation._vendor.seal_join import LabelJoinRefused, join_sealed, write_held_out_predictions
# ONE fence implementation: the grader's (grade_scope). These names are re-exports of the same function objects.
from agent_allocation.passk.grade_scope import blocked_node_ids, verify_seal

__all__ = ["HELD_OUT_PARTITIONS", "SEAL_NAME", "LabelJoinRefused", "SealError", "blocked_node_ids", "join_labels",
           "run_root_of", "seal_path", "verify_seal", "write_held_out_predictions"]

HELD_OUT_PARTITIONS = ("v48_bank", "t200_bank", "t200_diag")
SEAL_NAME = "SEAL_predictions.json"


class SealError(LabelJoinRefused):
    """A label join was attempted without a verifying seal, or against changed predictions."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_path(run_root: str | Path) -> Path:
    return Path(run_root) / "sealed" / SEAL_NAME


def run_root_of(shard_run_dir: str | Path) -> Path:
    """``<run_root>/artifacts/shards/<shard>`` -> ``<run_root>``."""
    return Path(shard_run_dir).resolve().parents[2]


def join_labels(run_root: str | Path, predictions_rel: str, labels: pd.DataFrame, *,
                on: tuple[str, ...] = ("item_id",)) -> pd.DataFrame:
    """Join held-out labels onto sealed predictions; refuses without a verifying seal or on changed predictions."""
    seal = verify_seal(run_root)
    if seal is None:
        raise SealError(f"{run_root}: no verifying {SEAL_NAME}; labels stay fenced")
    recorded = seal["artifacts"].get(predictions_rel)
    if recorded is None:
        raise SealError(f"{predictions_rel} is not in the seal")
    return join_sealed(Path(run_root) / predictions_rel, recorded, labels, on=on)
