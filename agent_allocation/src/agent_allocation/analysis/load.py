"""Read a run directory laid out per ``docs/00_contract.md`` §3 into typed records.

This is the one place analysis touches on-disk formats; every path comes from :class:`RunPaths` and every
record goes through ``contracts.load_record``. Nothing here joins grades onto selections: that happens in
:mod:`agent_allocation.analysis.outcomes`, and only for partitions whose seal verifies (sealed-before-join).

Seal verification accepts the seal's ``sha256`` when it equals either the sha256 of the selections file bytes
(``basis="file_bytes"``) or the canonical hash of the loaded record list (``basis="canonical_records"``); the
basis that matched is recorded. ``count`` must equal the number of records.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping

from agent_allocation.contracts import (
    BBH_SUBTASKS,
    Candidate,
    GateResult,
    GenerationResult,
    GenerationSpec,
    GradeResult,
    LedgerEvent,
    PanelDraw,
    Partition,
    PolicyCheckpoint,
    PublicCheckResult,
    PublicTask,
    SelectionResult,
    Wave,
    canonical_hash,
    dump_record,
    load_record,
)
from agent_allocation.paths import PROJECT_ROOT, RunPaths

PARTITION_ORDER: tuple[str, ...] = tuple(p.value for p in Partition)


class RunLoadError(RuntimeError):
    """The run directory cannot be read as a contract §3 layout."""


@dataclass(frozen=True)
class SealStatus:
    partition: str
    selections_present: bool
    seal_present: bool
    record_count: int
    seal_count: int | None
    sha256_ok: bool
    basis: str | None                 # file_bytes | canonical_records | None
    sealed_at: str | None
    problems: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return (self.selections_present and self.seal_present and self.sha256_ok
                and self.seal_count == self.record_count and not self.problems)


@dataclass
class RunData:
    paths: RunPaths
    config: dict[str, Any]
    config_source: str
    tasks: dict[str, PublicTask] = field(default_factory=dict)
    panels: dict[str, PanelDraw] = field(default_factory=dict)
    waves: list[Wave] = field(default_factory=list)
    jobs: dict[str, dict[str, GenerationSpec]] = field(default_factory=dict)       # partition -> node_id -> spec
    policies: dict[str, list[PolicyCheckpoint]] = field(default_factory=dict)      # partition -> checkpoints
    parsed: dict[str, tuple[GenerationResult, Candidate]] = field(default_factory=dict)
    public_checks: dict[str, PublicCheckResult] = field(default_factory=dict)
    grades: dict[str, GradeResult] = field(default_factory=dict)
    selections: dict[str, dict[str, SelectionResult]] = field(default_factory=dict)  # partition -> policy_id -> sel
    seals: dict[str, SealStatus] = field(default_factory=dict)
    ledger: list[LedgerEvent] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    amendments: list[dict[str, Any]] = field(default_factory=list)
    provider_snapshots: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    input_files: list[Path] = field(default_factory=list)

    # ------------------------------------------------------------------------------------------------
    @property
    def partitions(self) -> list[str]:
        names = set(self.jobs) | set(self.policies) | set(self.selections)
        return sorted(names, key=lambda p: (PARTITION_ORDER.index(p) if p in PARTITION_ORDER else 99, p))

    def item_panels(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, set[str]] = {}
        for wave in self.waves:
            for item in wave.item_ids:
                out.setdefault(item, set()).update(wave.panels)
        return {k: tuple(sorted(v)) for k, v in out.items()}

    def wave_of(self) -> dict[str, int]:
        return {item: wave.index for wave in self.waves for item in wave.item_ids}

    def benchmark_of(self, item_id: str) -> str | None:
        task = self.tasks.get(item_id)
        return task.benchmark if task else None

    def subtask_of(self, item_id: str) -> str | None:
        task = self.tasks.get(item_id)
        return task.subtask if task else None

    def input_digest(self) -> str:
        """Order-independent digest of every input file's bytes (report provenance, deterministic)."""
        h = sha256()
        for path in sorted(set(self.input_files)):
            h.update(self.paths.relative(path).encode())
            h.update(b"\0")
            h.update(sha256(path.read_bytes()).hexdigest().encode())
            h.update(b"\n")
        return h.hexdigest()


# ----------------------------------------------------------------------------------------------------
# Config resolution
# ----------------------------------------------------------------------------------------------------

def resolve_config(paths: RunPaths, explicit: str | Path | None = None) -> tuple[dict[str, Any], str]:
    """``--config`` wins; then the run's frozen config lock; then the committed default preset (labeled)."""
    if explicit:
        return json.loads(Path(explicit).read_text()), str(explicit)
    lock = paths.config_lock
    if lock.is_file():
        data = json.loads(lock.read_text())
        if isinstance(data, dict) and "study_id" in data:
            return data, paths.relative(lock) if _inside(lock, paths.run_dir) else str(lock)
        if isinstance(data, dict) and isinstance(data.get("config"), dict):
            return data["config"], str(lock)
    default = PROJECT_ROOT / "configs" / "balanced150.json"
    return json.loads(default.read_text()), f"default preset {default.name} (no config lock found)"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ----------------------------------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------------------------------

def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if line:
                yield lineno, json.loads(line)


def _partition_names(directory: Path, suffix: str) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(p.name[: -len(suffix)] for p in directory.iterdir() if p.name.endswith(suffix))


def load_run(paths: RunPaths, config: Mapping[str, Any] | None = None, *, config_source: str = "explicit",
             load_grades: bool = True) -> RunData:
    if not paths.run_dir.is_dir():
        raise RunLoadError(f"run directory does not exist: {paths.run_dir}")
    if config is None:
        config, config_source = resolve_config(paths)
    run = RunData(paths=paths, config=dict(config), config_source=config_source)

    # --- public manifests ----------------------------------------------------------------------------
    public_dir = paths.manifests_dir / "public"
    for bench_file in sorted(public_dir.glob("*.jsonl")) if public_dir.is_dir() else []:
        run.input_files.append(bench_file)
        for lineno, raw in _jsonl(bench_file):
            task = load_record(PublicTask, raw)
            if task.item_id in run.tasks:
                run.problems.append(f"duplicate public task {task.item_id} ({bench_file.name}:{lineno})")
            run.tasks[task.item_id] = task
            if task.benchmark == "bbh_logic4" and task.subtask not in BBH_SUBTASKS:
                run.problems.append(f"BBH item {task.item_id} has subtask {task.subtask!r} outside BBH_SUBTASKS")

    if paths.panels.is_file():
        run.input_files.append(paths.panels)
        run.panels = {k: load_record(PanelDraw, v) for k, v in json.loads(paths.panels.read_text()).items()}
    if paths.waves.is_file():
        run.input_files.append(paths.waves)
        run.waves = [load_record(Wave, w) for w in json.loads(paths.waves.read_text())]

    # --- jobs and policies ---------------------------------------------------------------------------
    for partition in _partition_names(paths.manifests_dir / "jobs", ".jsonl"):
        path = paths.jobs(partition)
        run.input_files.append(path)
        specs: dict[str, GenerationSpec] = {}
        for lineno, raw in _jsonl(path):
            spec = load_record(GenerationSpec, raw["record"])
            if spec.node_id != raw.get("node_id"):
                run.problems.append(f"jobs/{partition}.jsonl:{lineno}: node_id does not match its GenerationSpec")
            if spec.partition_id != partition:
                run.problems.append(f"jobs/{partition}.jsonl:{lineno}: spec.partition_id={spec.partition_id!r}")
            specs[spec.node_id] = spec
        run.jobs[partition] = specs
    for partition in _partition_names(paths.manifests_dir / "policies", ".jsonl"):
        path = paths.policies(partition)
        run.input_files.append(path)
        checkpoints = [load_record(PolicyCheckpoint, raw) for _, raw in _jsonl(path)]
        seen: set[str] = set()
        for cp in checkpoints:
            if cp.policy_id in seen:
                run.problems.append(f"policies/{partition}.jsonl: duplicate policy_id {cp.policy_id}")
            seen.add(cp.policy_id)
        run.policies[partition] = checkpoints

    # --- per-node artifacts (manifest-driven) ---------------------------------------------------------
    all_nodes = {nid for specs in run.jobs.values() for nid in specs}
    for node_id in sorted(all_nodes):
        path = paths.parsed(node_id)
        if path.is_file():
            run.input_files.append(path)
            raw = json.loads(path.read_text())
            result = load_record(GenerationResult, raw["result"])
            candidate = load_record(Candidate, raw["candidate"])
            if result.node_id != node_id or candidate.node_id != node_id or candidate.candidate_id != node_id:
                run.problems.append(f"parsed record for {node_id} carries a different node/candidate id")
            run.parsed[node_id] = (result, candidate)
        check_path = paths.public_check(node_id)
        if check_path.is_file():
            run.input_files.append(check_path)
            run.public_checks[node_id] = load_record(PublicCheckResult, json.loads(check_path.read_text()))
        if load_grades:
            grade_path = paths.grade(node_id)
            if grade_path.is_file():
                run.input_files.append(grade_path)
                grade = load_record(GradeResult, json.loads(grade_path.read_text()))
                if grade.candidate_id != node_id:
                    run.problems.append(f"grade file for {node_id} names candidate {grade.candidate_id}")
                run.grades[node_id] = grade

    # --- selections and seals ------------------------------------------------------------------------
    sel_dir = paths.run_dir / "selections"
    for partition in _partition_names(sel_dir, ".jsonl"):
        records, status = read_sealed_selections(paths, partition)
        run.input_files.append(paths.selections(partition))
        if paths.selection_seal(partition).is_file():
            run.input_files.append(paths.selection_seal(partition))
        run.seals[partition] = status
        by_policy: dict[str, SelectionResult] = {}
        for rec in records:
            if rec.policy_id in by_policy:
                run.problems.append(f"selections/{partition}.jsonl: duplicate policy_id {rec.policy_id}")
            by_policy[rec.policy_id] = rec
        run.selections[partition] = by_policy
    for partition in run.policies:
        if partition not in run.seals:
            run.seals[partition] = SealStatus(partition, False, paths.selection_seal(partition).is_file(), 0, None,
                                              False, None, None, ("no selections file",))

    # --- ledger, gates, amendments, provider snapshots -----------------------------------------------
    if paths.ledger_journal.is_file():
        run.input_files.append(paths.ledger_journal)
        run.ledger = [load_record(LedgerEvent, raw) for _, raw in _jsonl(paths.ledger_journal)]
    gates_dir = paths.run_dir / "gates"
    if gates_dir.is_dir():
        for gate_file in sorted(gates_dir.glob("*.json")):
            run.input_files.append(gate_file)
            run.gates.append(load_record(GateResult, json.loads(gate_file.read_text())))
    if paths.amendments.is_file():
        run.input_files.append(paths.amendments)
        run.amendments = [raw for _, raw in _jsonl(paths.amendments)]
    run.provider_snapshots = load_provider_snapshots(paths, run.input_files)
    return run


def read_sealed_selections(paths: RunPaths, partition: str) -> tuple[list[SelectionResult], SealStatus]:
    sel_path = paths.selections(partition)
    seal_path = paths.selection_seal(partition)
    problems: list[str] = []
    if not sel_path.is_file():
        return [], SealStatus(partition, False, seal_path.is_file(), 0, None, False, None, None,
                              ("no selections file",))
    payload = sel_path.read_bytes()
    records = [load_record(SelectionResult, json.loads(line)) for line in payload.decode("utf-8").splitlines()
               if line.strip()]
    if not seal_path.is_file():
        return records, SealStatus(partition, True, False, len(records), None, False, None, None,
                                   ("selections are not sealed",))
    seal = json.loads(seal_path.read_text())
    declared = str(seal.get("sha256", ""))
    basis = None
    if declared and declared == sha256(payload).hexdigest():
        basis = "file_bytes"
    elif declared and declared == canonical_hash([dump_record(r) for r in records]):
        basis = "canonical_records"
    else:
        problems.append("seal sha256 does not match the selections content")
    count = seal.get("count")
    if count != len(records):
        problems.append(f"seal count {count} != {len(records)} selection records")
    if not seal.get("sealed_at"):
        problems.append("seal has no sealed_at")
    return records, SealStatus(partition, True, True, len(records), count if isinstance(count, int) else None,
                               basis is not None, basis, seal.get("sealed_at"), tuple(problems))


def load_provider_snapshots(paths: RunPaths, input_files: list[Path] | None = None) -> list[dict[str, Any]]:
    """Provider-side ``/key`` (and ``/credits``) usage snapshots the lead stores under ``state/``.

    Accepted layouts (see docs/requests/07.md): ``state/provider_usage.jsonl`` with one
    ``{"fetched_at", "stage", "endpoint": "key"|"credits", "data": {...}}`` per line, or individual
    ``state/key_usage_*.json`` / ``state/credits_*.json`` files holding the same object.
    """
    out: list[dict[str, Any]] = []
    state = paths.state_dir
    if not state.is_dir():
        return out
    jsonl = state / "provider_usage.jsonl"
    if jsonl.is_file():
        if input_files is not None:
            input_files.append(jsonl)
        out.extend(raw for _, raw in _jsonl(jsonl))
    for pattern, endpoint in (("key_usage_*.json", "key"), ("credits_*.json", "credits")):
        for snap in sorted(state.glob(pattern)):
            if input_files is not None:
                input_files.append(snap)
            raw = json.loads(snap.read_text())
            raw.setdefault("endpoint", endpoint)
            raw.setdefault("source_file", snap.name)
            out.append(raw)
    out.sort(key=lambda r: (str(r.get("fetched_at", "")), str(r.get("endpoint", ""))))
    return out
