"""Seal verification, the held-out fence and the guarded label join (contract §5.6).

``verify_seal(run_root)`` returns the parsed ``sealed/SEAL_predictions.json`` only when every artifact it lists
still exists with the recorded sha256 (and the seal names a code commit); otherwise ``None``. A seal that is missing,
malformed or whose artifacts changed never verifies, so the fence cannot lift by accident.

``blocked_node_ids(paths, cfg)`` is the grader's skip set: every node id in the held-out partitions'
``manifests/jobs/<partition>.jsonl`` of that shard while no verifying seal exists.

``join_labels`` follows the nextdyn pattern (``agents_scaling/src/agents_scaling/nextdyn/analysis.py``
``write_held_out_predictions``/``join_labels``, vendored as ``_vendor/seal_join.py``): predictions are written and
hashed first; labels are joined only against a verifying seal whose recorded prediction hash matches the file on
disk.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["HELD_OUT_PARTITIONS", "SEAL_NAME", "SealError", "blocked_node_ids", "join_labels", "run_root_of",
           "seal_path", "verify_seal"]

HELD_OUT_PARTITIONS = ("v48_bank", "t200_bank", "t200_diag")
SEAL_NAME = "SEAL_predictions.json"


class SealError(RuntimeError):
    """A label join was attempted without a verifying seal, or against changed predictions."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seal_path(run_root: str | Path) -> Path:
    return Path(run_root) / "sealed" / SEAL_NAME


def verify_seal(run_root: str | Path) -> dict | None:
    path = seal_path(run_root)
    if not path.exists():
        return None
    try:
        seal = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    artifacts = seal.get("artifacts") if isinstance(seal, dict) else None
    if not isinstance(artifacts, dict) or not artifacts or not seal.get("code_commit"):
        return None
    for rel, digest in artifacts.items():
        target = Path(run_root) / rel
        if not target.is_file() or _sha256(target) != digest:
            return None
    return seal


def run_root_of(shard_run_dir: str | Path) -> Path:
    """``<run_root>/artifacts/shards/<shard>`` -> ``<run_root>``."""
    return Path(shard_run_dir).resolve().parents[2]


def blocked_node_ids(paths: Any, cfg: Any = None) -> frozenset[str]:
    """Held-out node ids of this shard while no verifying seal exists (the grader skips them)."""
    run_dir = Path(getattr(paths, "run_dir", paths))
    if verify_seal(run_root_of(run_dir)) is not None:
        return frozenset()
    blocked: set[str] = set()
    for partition in HELD_OUT_PARTITIONS:
        jobs = run_dir / "manifests" / "jobs" / f"{partition}.jsonl"
        if jobs.exists():
            with jobs.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        blocked.add(json.loads(line)["node_id"])
    return frozenset(blocked)


def join_labels(run_root: str | Path, predictions_rel: str, labels: pd.DataFrame, *,
                on: tuple[str, ...] = ("item_id",)) -> pd.DataFrame:
    """Join held-out labels onto sealed predictions; refuses without a verifying seal or on changed predictions."""
    seal = verify_seal(run_root)
    if seal is None:
        raise SealError(f"{run_root}: no verifying {SEAL_NAME}; labels stay fenced")
    recorded = seal["artifacts"].get(predictions_rel)
    target = Path(run_root) / predictions_rel
    if recorded is None:
        raise SealError(f"{predictions_rel} is not in the seal")
    if _sha256(target) != recorded:
        raise SealError(f"{predictions_rel} changed after the seal")
    predictions = pd.read_parquet(target)
    return predictions.merge(labels, on=list(on), how="left", validate="many_to_one")
