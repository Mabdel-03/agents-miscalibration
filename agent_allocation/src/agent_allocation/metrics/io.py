"""Disk side of the metrics join, through ``RunPaths`` only (docs/00_contract.md §3, R3/R4).

``load_partition_tables(paths, partition)`` verifies the partition's selection seal *before* it opens any grade file
(``SealError`` otherwise), then reads the policies manifest, job specs, public tasks, parsed candidates/results and
grades, and builds the per-item tables. ``write_tables`` writes one canonical-JSON-lines file per table plus
``summary_checkpoints.jsonl`` into a directory the caller chooses; ``write_partition_tables`` writes the same files
through ``RunPaths.metrics_table(metrics_table_name(partition, table))`` (contract v2, R06-3).

Config keys consumed: none.
"""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from agent_allocation.contracts import (
    BENCHMARKS, Candidate, GenerationResult, GenerationSpec, GradeResult, PolicyCheckpoint, PublicTask, canonical_json,
    load_record,
)
from agent_allocation.metrics.tables import (
    ItemRecords, PartitionRecords, PartitionTables, build_partition_tables, summarize_checkpoints,
)
from agent_allocation.paths import RunPaths
from agent_allocation.selection.seal import load_sealed_selections

__all__ = [
    "TABLE_FILES", "load_grades", "load_partition_records", "load_partition_tables", "load_public_tasks",
    "metrics_table_name", "write_partition_tables", "write_tables",
]

#: Every table ``write_tables`` emits, in order (``summary_checkpoints`` is ``summarize_checkpoints``).
TABLE_FILES: tuple[str, ...] = ("checkpoints", "bank", "slot_transitions", "team_transitions", "root_strata",
                                "failures", "summary_checkpoints")


def metrics_table_name(partition: str, table: str) -> str:
    """The ``RunPaths.metrics_table`` name of one partition's table, e.g. ``"main.checkpoints"``."""
    if table not in TABLE_FILES:
        raise ValueError(f"unknown metrics table {table!r}")
    return f"{partition}.{table}"


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: not JSON") from exc


def load_public_tasks(paths: RunPaths) -> dict[str, PublicTask]:
    tasks: dict[str, PublicTask] = {}
    for benchmark in BENCHMARKS:
        path = paths.public_manifest(benchmark)
        if not path.exists():
            continue
        for data in _jsonl(path):
            task = load_record(PublicTask, data)
            if task.item_id in tasks:
                raise ValueError(f"duplicate public task {task.item_id}")
            tasks[task.item_id] = task
    return tasks


def load_partition_records(paths: RunPaths, partition: str) -> PartitionRecords:
    policies_path = paths.policies(partition)
    if not policies_path.exists():
        raise FileNotFoundError(f"{policies_path}: no policy manifest for partition {partition!r}")
    checkpoints = [load_record(PolicyCheckpoint, data) for data in _jsonl(policies_path)]
    specs: dict[str, GenerationSpec] = {}
    jobs_path = paths.jobs(partition)
    if jobs_path.exists():
        for data in _jsonl(jobs_path):
            spec = load_record(GenerationSpec, data["record"])
            if spec.node_id != data["node_id"]:
                raise ValueError(f"{jobs_path}: node_id {data['node_id']} does not match its GenerationSpec")
            specs[spec.node_id] = spec
    tasks = load_public_tasks(paths)
    by_item: dict[str, list[PolicyCheckpoint]] = {}
    for cp in checkpoints:
        by_item.setdefault(cp.item_id, []).append(cp)
    items = []
    for item_id in sorted(by_item):
        if item_id not in tasks:
            raise ValueError(f"item {item_id} has checkpoints but no public task")
        ids = sorted({c for cp in by_item[item_id] for c in (*cp.candidate_ids, *cp.required_node_ids)})
        candidates: dict[str, Candidate] = {}
        results: dict[str, GenerationResult] = {}
        for node_id in ids:
            path = paths.parsed(node_id)
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            result = load_record(GenerationResult, data["result"])
            candidate = load_record(Candidate, data["candidate"])
            if result.node_id != node_id or candidate.candidate_id != node_id:
                raise ValueError(f"{path}: parsed record does not belong to node {node_id}")
            candidates[node_id], results[node_id] = candidate, result
        items.append(ItemRecords(task=tasks[item_id], checkpoints=tuple(by_item[item_id]), candidates=candidates,
                                 results=results, specs={n: specs[n] for n in ids if n in specs}))
    return PartitionRecords(partition=partition, items=tuple(items))


def load_grades(paths: RunPaths, candidate_ids: Iterable[str]) -> dict[str, GradeResult]:
    grades: dict[str, GradeResult] = {}
    for cid in sorted(set(candidate_ids)):
        path = paths.grade(cid)
        if path.exists():
            grade = load_record(GradeResult, json.loads(path.read_text(encoding="utf-8")))
            if grade.candidate_id != cid:
                raise ValueError(f"{path}: grade belongs to {grade.candidate_id}")
            grades[cid] = grade
    return grades


def load_partition_tables(paths: RunPaths, partition: str, *, strict_populations: bool = True) -> PartitionTables:
    sealed = load_sealed_selections(paths, partition)          # refuses an unsealed partition before any grade read
    records = load_partition_records(paths, partition)
    ids = {c for item in records.items for cp in item.checkpoints for c in (*cp.candidate_ids, *cp.required_node_ids)}
    return build_partition_tables(sealed, records, load_grades(paths, ids), strict_populations=strict_populations)


def _payload(tables: PartitionTables) -> dict[str, bytes]:
    rows: Mapping[str, list[dict[str, Any]]] = {**tables.as_dicts(),
                                                "summary_checkpoints": summarize_checkpoints(tables.checkpoints)}
    return {name: "".join(canonical_json(row) + "\n" for row in rows[name]).encode("utf-8") for name in TABLE_FILES}


def _write(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(body)
    tmp.replace(path)
    return sha256(body).hexdigest()


def write_tables(tables: PartitionTables, directory: Path) -> dict[str, str]:
    """Write ``{table}.jsonl`` for every table in ``TABLE_FILES`` into ``directory``; returns ``{file name: sha256}``."""
    return {f"{name}.jsonl": _write(Path(directory) / f"{name}.jsonl", body)
            for name, body in _payload(tables).items()}


def write_partition_tables(paths: RunPaths, partition: str, tables: PartitionTables) -> dict[str, str]:
    """Write every table to ``paths.metrics_table(metrics_table_name(partition, table))``; returns name -> sha256."""
    return {metrics_table_name(partition, name): _write(paths.metrics_table(metrics_table_name(partition, name)), body)
            for name, body in _payload(tables).items()}
