"""Read a run directory laid out per ``docs/00_contract.md`` §3 into typed records.

This is the one place analysis touches on-disk formats; every path comes from :class:`RunPaths` and every
record goes through ``contracts.load_record``. Nothing here joins grades onto selections: that happens in
:mod:`agent_allocation.analysis.outcomes`, and only for partitions whose seal verifies (sealed-before-join).

Selections (contract v2, D8): append-only batches ``selections/{partition}/batch_NNNN.jsonl``, each with
``batch_NNNN.seal.json`` = ``{sha256 of the batch file bytes, count, sealed_at, previous_seal_sha256}``. A batch is
joinable only when its seal verifies: digest and line count match, ``sealed_at`` is present, numbering is contiguous
from 0, and ``previous_seal_sha256`` chains to the previous batch (empty for batch 0; the previous seal's ``sha256``
or the sha256 of its seal-file bytes, the matched basis is recorded). A policy sealed in two batches invalidates the
later batch (a sealed selection is never rewritten).
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
    load_record,
)
from agent_allocation.paths import PROJECT_ROOT, RunPaths

PARTITION_ORDER: tuple[str, ...] = tuple(p.value for p in Partition)


class RunLoadError(RuntimeError):
    """The run directory cannot be read as a contract §3 layout."""


@dataclass(frozen=True)
class BatchSeal:
    batch: int
    seal_present: bool
    record_count: int
    seal_count: int | None
    sha256_ok: bool
    sealed_at: str | None
    chain_basis: str | None                 # genesis | seal_sha256 | seal_file_bytes | None
    problems: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.seal_present and self.sha256_ok and self.seal_count == self.record_count and not self.problems


@dataclass(frozen=True)
class PartitionSeal:
    partition: str
    batches: tuple[BatchSeal, ...] = ()
    problems: tuple[str, ...] = ()

    @property
    def selections_present(self) -> bool:
        return bool(self.batches)

    @property
    def valid_batches(self) -> frozenset[int]:
        return frozenset(b.batch for b in self.batches if b.valid)

    @property
    def valid(self) -> bool:
        """Every batch present verifies (a partition with no batches is not valid: nothing is sealed)."""
        return bool(self.batches) and all(b.valid for b in self.batches) and not self.problems

    @property
    def record_count(self) -> int:
        return sum(b.record_count for b in self.batches)

    @property
    def sealed_at(self) -> str | None:
        stamps = [b.sealed_at for b in self.batches if b.sealed_at]
        return max(stamps) if stamps else None

    @property
    def all_problems(self) -> tuple[str, ...]:
        return self.problems + tuple(f"batch {b.batch}: {p}" for b in self.batches for p in b.problems)


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
    selection_batch: dict[str, dict[str, int]] = field(default_factory=dict)          # partition -> policy_id -> batch
    seals: dict[str, PartitionSeal] = field(default_factory=dict)
    ledger: list[LedgerEvent] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)
    amendments: list[dict[str, Any]] = field(default_factory=list)
    provider_snapshots: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    input_files: list[Path] = field(default_factory=list)

    # ------------------------------------------------------------------------------------------------
    def is_sealed(self, partition: str, policy_id: str) -> bool:
        """True when the policy's selection sits in a verified batch (the only rows grades may join)."""
        seal = self.seals.get(partition)
        batch = self.selection_batch.get(partition, {}).get(policy_id)
        return seal is not None and batch is not None and batch in seal.valid_batches

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
        label = paths.relative(lock) if _inside(lock, paths.run_dir) else str(lock)
        if isinstance(data, dict) and isinstance(data.get("config"), dict):
            return data["config"], f"{label} (frozen config lock)"
        if isinstance(data, dict) and "study_id" in data:
            return data, f"{label} (raw config object)"
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

    # --- selections and seals (v2 batches) ------------------------------------------------------------
    sel_root = paths.run_dir / "selections"
    partitions = sorted(p.name for p in sel_root.iterdir() if p.is_dir()) if sel_root.is_dir() else []
    for partition in partitions:
        by_policy, batch_of, status, files = read_sealed_selections(paths, partition)
        run.input_files.extend(files)
        run.seals[partition] = status
        run.selections[partition] = by_policy
        run.selection_batch[partition] = batch_of
    for partition in run.policies:
        if partition not in run.seals:
            run.seals[partition] = PartitionSeal(partition, (), ("no selection batches",))

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


def selection_batch_numbers(paths: RunPaths, partition: str) -> list[int]:
    directory = paths.selections_dir(partition)
    if not directory.is_dir():
        return []
    numbers = []
    for path in directory.glob("batch_*.jsonl"):
        stem = path.name[len("batch_"):-len(".jsonl")]
        if stem.isdigit():
            numbers.append(int(stem))
    return sorted(numbers)


def read_sealed_selections(paths: RunPaths, partition: str
                           ) -> tuple[dict[str, SelectionResult], dict[str, int], PartitionSeal, list[Path]]:
    """All batches of a partition with per-batch seal verification; records of every batch are returned (the
    caller joins grades only for ``PartitionSeal.valid_batches``)."""
    by_policy: dict[str, SelectionResult] = {}
    batch_of: dict[str, int] = {}
    batches: list[BatchSeal] = []
    files: list[Path] = []
    partition_problems: list[str] = []
    numbers = selection_batch_numbers(paths, partition)
    if numbers and numbers != list(range(len(numbers))):
        partition_problems.append(f"batch numbering is not contiguous from 0: {numbers}")
    previous: tuple[dict[str, Any], bytes] | None = None
    for number in numbers:
        path = paths.selections(partition, number)
        seal_path = paths.selection_seal(partition, number)
        files.append(path)
        payload = path.read_bytes()
        records = [load_record(SelectionResult, json.loads(line)) for line in payload.decode("utf-8").splitlines()
                   if line.strip()]
        problems: list[str] = []
        for rec in records:
            if rec.policy_id in by_policy:
                problems.append(f"policy {rec.policy_id} already sealed in batch {batch_of[rec.policy_id]}")
                continue
            by_policy[rec.policy_id] = rec
            batch_of[rec.policy_id] = number
        if not seal_path.is_file():
            batches.append(BatchSeal(number, False, len(records), None, False, None, None,
                                     tuple(problems + ["batch is not sealed"])))
            previous = None
            continue
        files.append(seal_path)
        seal_bytes = seal_path.read_bytes()
        seal = json.loads(seal_bytes)
        sha_ok = str(seal.get("sha256", "")) == sha256(payload).hexdigest()
        if not sha_ok:
            problems.append("seal sha256 does not match the batch file bytes")
        count = seal.get("count")
        if count != len(records):
            problems.append(f"seal count {count} != {len(records)} records")
        if not seal.get("sealed_at"):
            problems.append("seal has no sealed_at")
        prev_declared = seal.get("previous_seal_sha256")
        chain = None
        if number == 0:
            if prev_declared:
                problems.append("batch 0 declares a previous seal")
            else:
                chain = "genesis"
        elif previous is None:
            problems.append("previous batch is unsealed; the seal chain is broken")
        elif prev_declared == previous[0].get("sha256"):
            chain = "seal_sha256"
        elif prev_declared == sha256(previous[1]).hexdigest():
            chain = "seal_file_bytes"
        else:
            problems.append("previous_seal_sha256 does not chain to the previous batch seal")
        batches.append(BatchSeal(number, True, len(records), count if isinstance(count, int) else None, sha_ok,
                                 seal.get("sealed_at"), chain, tuple(problems)))
        previous = (seal, seal_bytes)
    return by_policy, batch_of, PartitionSeal(partition, tuple(batches), tuple(partition_problems)), files


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
