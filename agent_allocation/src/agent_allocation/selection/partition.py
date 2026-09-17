"""Select every ready checkpoint of a partition from disk and seal the results (public inputs only).

Reads, through ``RunPaths``: ``manifests/policies/{partition}.jsonl``, ``manifests/public/*.jsonl``,
``parsed/**/{node_id}.json``, for code ``public_checks/**/{candidate_id}.json``, and the coordinator's state backup
(read-only) for nodes that ended without a response. It never opens ``private/`` or ``grades/``. A checkpoint is
*ready* when every pool member is parsed or ended without output (a parsed record whose ``candidate`` is ``null``,
or a state row with a non-succeeded terminal outcome -- see ``selection.terminal``); the latter are passed to the
selector as ``absent_ids`` (07-R12: invalid members, marked absent in the pool hash). For code, every validly parsed
pool candidate must also have a public check; other checkpoints are reported as
pending and selected on a later call (sealing appends only unsealed checkpoints and never seals a paused selection,
see ``selection.seal``).
``math_equivalent`` is memoized once per call (``memoized_equivalence``) and passed to MATH checkpoints only.

Config keys consumed: none (``seed`` is the study's top-level ``seed``, passed by the caller).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from agent_allocation.contracts import (
    BENCHMARKS, Candidate, OutputKind, ParseStatus, PolicyCheckpoint, PublicCheckResult, PublicTask, SelectionResult,
    load_record,
)
from agent_allocation.paths import RunPaths
from agent_allocation.selection.plurality import memoized_equivalence
from agent_allocation.selection.readout import select_checkpoint
from agent_allocation.selection.seal import seal_selections
from agent_allocation.selection.terminal import load_terminal_without_output

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
    equivalent = None if math_equivalent is None else memoized_equivalence(math_equivalent)
    terminal = load_terminal_without_output(paths)
    candidates: dict[str, Candidate] = {}
    absent: set[str] = set()                         # ended without any candidate (07-R12 absent members)
    checks: dict[str, PublicCheckResult] = {}
    results: list[SelectionResult] = []
    pending: list[str] = []
    for cp in checkpoints:
        task = tasks.get(cp.item_id)
        if task is None:
            raise ValueError(f"{cp.policy_id}: no public task for item {cp.item_id}")
        ready = True
        for cid in cp.candidate_ids:
            if cid not in candidates and cid not in absent:
                path = paths.parsed(cid)
                record = _load_json(path).get("candidate") if path.exists() else None
                if record is not None:
                    candidates[cid] = load_record(Candidate, record)
                elif path.exists() or cid in terminal:
                    absent.add(cid)                  # a stored null candidate, or terminal in the coordinator state
                else:
                    ready = False
                    break
            if cid in absent:
                continue
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
        pool = {cid: candidates[cid] for cid in cp.candidate_ids if cid in candidates}
        pool_checks = {cid: checks[cid] for cid in cp.candidate_ids if cid in checks}
        results.append(select_checkpoint(cp, task, pool, pool_checks, seed=seed,
                                         math_equivalent=equivalent if task.output_kind is OutputKind.MATH
                                         else None, absent_ids=frozenset(c for c in cp.candidate_ids if c in absent)))
    digest = seal_selections(paths, partition, results) if seal and results else None
    return PartitionSelection(partition=partition, results=tuple(results), pending_policy_ids=tuple(pending),
                              sealed_sha256=digest)
