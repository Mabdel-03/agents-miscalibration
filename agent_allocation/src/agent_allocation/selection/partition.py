"""Select every ready checkpoint of a partition from disk and seal the results (public inputs only).

Reads, through ``RunPaths``: ``manifests/policies/{partition}.jsonl``, ``manifests/public/*.jsonl``,
``parsed/**/{node_id}.json`` and, for code, ``public_checks/**/{candidate_id}.json``. It never opens ``private/`` or
``grades/``. A checkpoint is *ready* when every pool candidate is parsed and, for code, every validly parsed pool
candidate has a public check; other checkpoints are reported as pending and selected on a later call (sealing
merges, see ``selection.seal``). ``math_equivalent`` is passed to MATH checkpoints only.

Config keys consumed: none (``seed`` is the study's top-level ``seed``, passed by the caller).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from agent_allocation.contracts import (
    BENCHMARKS, Candidate, OutputKind, ParseStatus, PolicyCheckpoint, PublicCheckResult, PublicTask, SelectionResult, load_record,
)
from agent_allocation.paths import RunPaths
from agent_allocation.selection.readout import select_checkpoint
from agent_allocation.selection.seal import seal_selections

__all__ = ["PartitionSelection", "select_partition"]


@dataclass(frozen=True)
class PartitionSelection:
    partition: str
    results: tuple[SelectionResult, ...]
    pending_policy_ids: tuple[str, ...]
    sealed_sha256: str | None


def _load_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_partition(paths: RunPaths, partition: str, *, seed: int,
                     math_equivalent: Callable[[str, str], bool | None] | None = None,
                     seal: bool = True) -> PartitionSelection:
    policies_path = paths.policies(partition)
    if not policies_path.exists():
        raise FileNotFoundError(f"{policies_path}: no policy manifest for partition {partition!r}")
    with policies_path.open(encoding="utf-8") as handle:
        checkpoints = [load_record(PolicyCheckpoint, json.loads(line)) for line in handle if line.strip()]
    tasks: dict[str, PublicTask] = {}
    for benchmark in BENCHMARKS:
        manifest = paths.public_manifest(benchmark)
        if not manifest.exists():
            continue
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    task = load_record(PublicTask, json.loads(line))
                    tasks[task.item_id] = task
    candidates: dict[str, Candidate] = {}
    checks: dict[str, PublicCheckResult] = {}
    results: list[SelectionResult] = []
    pending: list[str] = []
    for cp in checkpoints:
        task = tasks.get(cp.item_id)
        if task is None:
            raise ValueError(f"{cp.policy_id}: no public task for item {cp.item_id}")
        ready = True
        for cid in cp.candidate_ids:
            if cid not in candidates:
                path = paths.parsed(cid)
                if not path.exists():
                    ready = False
                    break
                candidates[cid] = load_record(Candidate, _load_json(path)["candidate"])
            if task.output_kind is OutputKind.PYTHON and cid not in checks:
                check_path = paths.public_check(cid)
                if check_path.exists():
                    checks[cid] = load_record(PublicCheckResult, _load_json(check_path))
                elif candidates[cid].parse_status is ParseStatus.VALID and len(cp.candidate_ids) > 1:
                    ready = False
                    break
        if not ready:
            pending.append(cp.policy_id)
            continue
        pool = {cid: candidates[cid] for cid in cp.candidate_ids}
        pool_checks = {cid: checks[cid] for cid in cp.candidate_ids if cid in checks}
        results.append(select_checkpoint(cp, task, pool, pool_checks, seed=seed,
                                         math_equivalent=math_equivalent if task.output_kind is OutputKind.MATH
                                         else None))
    digest = seal_selections(paths, partition, results) if seal and results else None
    return PartitionSelection(partition=partition, results=tuple(results), pending_policy_ids=tuple(pending),
                              sealed_sha256=digest)
