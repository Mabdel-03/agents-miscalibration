"""Held-out fence for grading (WP-G; contract sections 5.6 and "Held-out fence").

``allocation grade`` on a passk shard dir recomputes, on every ``--follow`` pass:

* :func:`blocked_node_ids` -- node ids of the held-out partitions (``cfg['passk']['heldout']``: ``v48`` ->
  ``v48_bank``; ``t200`` -> ``t200_bank``, ``t200_diag``) read from ``manifests/jobs/<partition>.jsonl`` of the shard,
  for every split without a verified seal;
* :func:`gradeable_node_ids` -- every node id listed in the shard's jobs manifests minus the blocked ones. The grader
  grades ONLY these (fail closed: a parsed candidate that no jobs file lists is never graded).

:func:`verify_seal` accepts ``sealed/SEAL_predictions.json`` only when it parses, names a ``code_commit``, lists at least
one artifact and every listed artifact exists with the recorded sha256. A seal may restrict what it releases with ``"splits": [...]``;
without that key a verified seal releases every held-out split. The seal lives at the study root
(``.../passk_cont_v1/sealed``), found from a shard dir via :func:`passk.banks.study_root`.
"""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

SEAL_REL = Path("sealed") / "SEAL_predictions.json"
DEFAULT_HELDOUT = {"v48": ["v48_bank"], "t200": ["t200_bank", "t200_diag"]}


def is_passk_config(cfg: Mapping[str, Any] | None) -> bool:
    return bool(cfg) and isinstance(cfg.get("passk"), Mapping)


def _study_root(run_root: Any) -> Path:
    from agent_allocation.passk.banks import study_root
    root = Path(getattr(run_root, "run_dir", run_root))
    if (root / "sealed").is_dir() or (root / "artifacts").is_dir():
        return root
    return study_root(root)


def _file_sha(path: Path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal(run_root: Any) -> dict[str, Any] | None:
    """The parsed seal when ``sealed/SEAL_predictions.json`` verifies (every artifact sha256 matches), else None."""
    root = _study_root(run_root)
    target = root / SEAL_REL
    if not target.is_file():
        return None
    try:
        seal = json.loads(target.read_text())
    except (OSError, ValueError):
        return None
    artifacts = seal.get("artifacts") if isinstance(seal, Mapping) else None
    if not isinstance(artifacts, Mapping) or not artifacts or not seal.get("code_commit"):
        return None                                  # same acceptance rule as WP-A's labels.verify_seal
    for rel, expected in artifacts.items():
        path = Path(rel)
        path = path if path.is_absolute() else root / path
        try:
            if not path.is_file() or _file_sha(path) != str(expected):
                return None
        except OSError:
            return None
    return dict(seal)


def heldout_splits(cfg: Mapping[str, Any]) -> dict[str, list[str]]:
    return {k: list(v) for k, v in ((cfg.get("passk") or {}).get("heldout") or DEFAULT_HELDOUT).items()}


def released_splits(cfg: Mapping[str, Any], seal: Mapping[str, Any] | None) -> set[str]:
    if seal is None:
        return set()
    splits = heldout_splits(cfg)
    allowed = seal.get("splits")
    return set(splits) if allowed is None else {s for s in splits if s in set(allowed)}


def _jobs_ids(path: Path) -> set[str]:
    """Node ids of one jobs manifest; unreadable/partial lines are ignored (the caller fails closed on them)."""
    out: set[str] = set()
    if not path.is_file():
        return out
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            out.add(str(json.loads(line)["node_id"]))
        except (ValueError, KeyError, TypeError):
            continue
    return out


def _jobs_by_partition(paths) -> dict[str, set[str]]:
    """``{partition: node ids}`` from ONE read of every jobs manifest of the run dir."""
    jobs_dir = paths.manifests_dir / "jobs"
    if not jobs_dir.is_dir():
        return {}
    return {path.stem: _jobs_ids(path) for path in sorted(jobs_dir.glob("*.jsonl"))}


def _scope(paths, cfg: Mapping[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    """``(blocked, gradeable)`` computed from the same single read of the jobs manifests, so a manifest rewritten
    mid-pass can only drop ids from BOTH sets (never grade a fenced id)."""
    by_partition = _jobs_by_partition(paths)
    released = released_splits(cfg, verify_seal(paths))
    fenced_partitions = {p for split, parts in heldout_splits(cfg).items() if split not in released for p in parts}
    blocked: set[str] = set()
    allowed: set[str] = set()
    for partition, ids in by_partition.items():
        (blocked if partition in fenced_partitions else allowed).update(ids)
    return frozenset(blocked), frozenset(allowed - blocked)


def blocked_node_ids(paths, cfg: Mapping[str, Any]) -> frozenset[str]:
    """Held-out node ids of this run dir whose split has no verified seal (empty for a non-passk config)."""
    if not is_passk_config(cfg):
        return frozenset()
    return _scope(paths, cfg)[0]


def all_job_ids(paths) -> set[str]:
    return {i for ids in _jobs_by_partition(paths).values() for i in ids}


def gradeable_node_ids(paths, cfg: Mapping[str, Any]) -> frozenset[str] | None:
    """Node ids the grader may grade on a passk run dir (``None``: no restriction, legacy behaviour)."""
    if not is_passk_config(cfg):
        return None
    return _scope(paths, cfg)[1]


def grade_scope(paths, cfg: Mapping[str, Any] | None) -> tuple[frozenset[str], frozenset[str] | None]:
    """``(skip, only)`` for one grading pass (recomputed every pass, from one read of the manifests)."""
    if not is_passk_config(cfg):
        return frozenset(), None
    return _scope(paths, cfg)


def fenced_counts(paths, cfg: Mapping[str, Any], candidate_ids: Iterable[str]) -> dict[str, int]:
    blocked = blocked_node_ids(paths, cfg)
    ids = set(candidate_ids)
    return {"candidates": len(ids), "fenced": len(ids & blocked)}
