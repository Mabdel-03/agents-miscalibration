"""The single registry of artifact locations. Modules resolve paths here and never build run-dir paths by hand.

Run dir (``--run-dir`` / ``ALLOCATION_RUN_DIR``, default ``artifacts/openrouter_agent_allocation_v1``) holds one
cumulative study ledger for canary, pilot, main, high, extension and replication. SQLite lives on node-local disk; its
backup, the ledger journal and every artifact live on the shared filesystem.

v2 (resolves D5, D6, D8, D9, R03-1/2, R05-1, R06-3, R07-2): locks are run-scoped (``<run_dir>/locks``, the spec §14
tree; the config's ``*_lock_file`` basenames are validated against these names), node-local state is keyed by a hash
of the resolved run dir, selections are sealed in append-only batches, and a re-grade supersedes (moves) the previous
verdict instead of overwriting it.
"""
from __future__ import annotations

import getpass
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from agent_allocation.contracts import STUDY_ID

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts"
RESULTS_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/openrouter_agent_allocation_v1")
PRIVATE_DIR_MODE = 0o700


def shard(identifier: str) -> str:
    return identifier[:2]


def safe_name(identifier: str) -> str:
    return identifier.replace(":", "_").replace("/", "__")


def run_key(run_dir: Path) -> str:
    resolved = str(Path(run_dir).resolve())
    return f"{Path(resolved).name}-{sha256(resolved.encode()).hexdigest()[:12]}"


def default_local_state_dir(run_dir: Path) -> Path:
    user = getpass.getuser()
    for base in (Path("/scratch"), Path(os.environ.get("TMPDIR", "/tmp"))):
        if base.is_dir() and os.access(base, os.W_OK):
            return base / user / "agent_allocation" / run_key(run_dir)
    raise RuntimeError("No writable node-local directory for SQLite state (/scratch or $TMPDIR)")


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    locks_dir: Path
    local_state_dir: Path

    @classmethod
    def for_run(cls, run_dir: str | Path | None = None, *, locks_dir: str | Path | None = None,
                local_state_dir: str | Path | None = None) -> "RunPaths":
        chosen = Path(run_dir or os.environ.get("ALLOCATION_RUN_DIR") or (ARTIFACTS / STUDY_ID))
        if not chosen.is_absolute():
            chosen = PROJECT_ROOT / chosen
        locks = Path(locks_dir) if locks_dir else chosen / "locks"
        local = Path(local_state_dir or os.environ.get("ALLOCATION_LOCAL_STATE")
                     or default_local_state_dir(chosen))
        return cls(chosen, locks, local)

    # --- write-once locks -----------------------------------------------------------------------------
    @property
    def endpoint_lock(self) -> Path: return self.locks_dir / "openrouter_endpoints.json"
    @property
    def dataset_lock(self) -> Path: return self.locks_dir / "datasets.json"
    @property
    def prompt_lock(self) -> Path: return self.locks_dir / "prompts.json"
    @property
    def environment_lock(self) -> Path: return self.locks_dir / "environment.json"
    @property
    def config_lock(self) -> Path: return self.locks_dir / "config.json"
    @property
    def lock_snapshots_dir(self) -> Path: return self.locks_dir / "snapshots"

    # --- manifests --------------------------------------------------------------------------------------
    @property
    def manifests_dir(self) -> Path: return self.run_dir / "manifests"
    def public_manifest(self, benchmark: str) -> Path: return self.manifests_dir / "public" / f"{benchmark}.jsonl"
    @property
    def panels(self) -> Path: return self.manifests_dir / "panels.json"
    @property
    def waves(self) -> Path: return self.manifests_dir / "waves.json"
    @property
    def exclusions(self) -> Path: return self.manifests_dir / "exclusions.json"
    def jobs(self, partition: str) -> Path: return self.manifests_dir / "jobs" / f"{partition}.jsonl"
    def policies(self, partition: str) -> Path: return self.manifests_dir / "policies" / f"{partition}.jsonl"
    @property
    def alias_report(self) -> Path: return self.manifests_dir / "alias_report.json"

    # --- private (mode 0700; graders only) --------------------------------------------------------------
    @property
    def private_dir(self) -> Path: return self.run_dir / "private"
    def private_manifest(self, benchmark: str) -> Path: return self.private_dir / f"{benchmark}.jsonl"
    @property
    def code_oracles_dir(self) -> Path: return self.private_dir / "evalplus_oracles"
    @property
    def bbh_sources_dir(self) -> Path: return self.private_dir / "bbh_sources"

    # --- coordinator state ------------------------------------------------------------------------------
    @property
    def state_dir(self) -> Path: return self.run_dir / "state"
    @property
    def state_db_local(self) -> Path: return self.local_state_dir / "state.sqlite"
    @property
    def state_db_backup(self) -> Path: return self.state_dir / "state.sqlite"
    @property
    def ledger_journal(self) -> Path: return self.state_dir / "ledger_journal.jsonl"
    @property
    def status_json(self) -> Path: return self.state_dir / "status.json"
    @property
    def authorization(self) -> Path: return self.state_dir / "authorization.json"
    @property
    def render_reports(self) -> Path: return self.state_dir / "render_reports.jsonl"
    @property
    def context_limit_events(self) -> Path: return self.state_dir / "context_limit_events.jsonl"
    @property
    def provider_usage(self) -> Path: return self.state_dir / "provider_usage.jsonl"
    @property
    def generation_metadata(self) -> Path: return self.state_dir / "generation_metadata.jsonl"
    @property
    def run_decisions(self) -> Path: return self.state_dir / "run_decisions.jsonl"
    @property
    def control_dir(self) -> Path: return self.run_dir / "control"
    @property
    def stop_sentinel(self) -> Path: return self.control_dir / "STOP"
    @property
    def drain_sentinel(self) -> Path: return self.control_dir / "DRAIN"

    # --- per-record artifacts ---------------------------------------------------------------------------
    def raw_response(self, attempt_id: str) -> Path:
        return self.run_dir / "raw" / shard(attempt_id) / f"{safe_name(attempt_id)}.json.gz"
    def parsed(self, node_id: str) -> Path:
        return self.run_dir / "parsed" / shard(node_id) / f"{node_id}.json"
    def public_check(self, candidate_id: str) -> Path:
        """Current verdict. A re-check with a different checker version first moves the old file to
        ``superseded_public_check(candidate_id, old_version)`` (never an in-place overwrite)."""
        return self.run_dir / "public_checks" / shard(candidate_id) / f"{candidate_id}.json"
    def superseded_public_check(self, candidate_id: str, checker_version: str) -> Path:
        return (self.run_dir / "public_checks_superseded" / safe_name(checker_version) / shard(candidate_id)
                / f"{candidate_id}.json")
    @property
    def grades_dir(self) -> Path: return self.run_dir / "grades"
    def grade(self, candidate_id: str) -> Path:
        """Current verdict (the record carries ``grader_version``). A repaired grader moves the old verdict to
        ``superseded_grade`` before writing; readers of ``grades/`` only ever see current verdicts."""
        return self.grades_dir / shard(candidate_id) / f"{candidate_id}.json"
    def superseded_grade(self, candidate_id: str, grader_version: str) -> Path:
        return (self.run_dir / "grades_superseded" / safe_name(grader_version) / shard(candidate_id)
                / f"{candidate_id}.json")
    @property
    def grade_cache_dir(self) -> Path: return self.run_dir / "grade_cache"
    def selections_dir(self, partition: str) -> Path: return self.run_dir / "selections" / partition
    def selections(self, partition: str, batch: int) -> Path:
        """Append-only batches: a sealed batch is never rewritten; later batches hold only checkpoints not yet sealed."""
        return self.selections_dir(partition) / f"batch_{batch:04d}.jsonl"
    def selection_seal(self, partition: str, batch: int) -> Path:
        """``{sha256 (of the batch .jsonl file bytes), count, sealed_at, previous_seal_sha256}``."""
        return self.selections_dir(partition) / f"batch_{batch:04d}.seal.json"

    # --- outputs ------------------------------------------------------------------------------------------
    @property
    def tables_dir(self) -> Path: return self.run_dir / "tables"
    @property
    def figures_dir(self) -> Path: return self.run_dir / "figures"
    @property
    def reports_dir(self) -> Path: return self.run_dir / "reports"
    def metrics_table(self, name: str) -> Path: return self.tables_dir / "metrics" / f"{name}.jsonl"
    def gate_result(self, gate: str, stage: str) -> Path: return self.run_dir / "gates" / f"{gate}_{stage}.json"
    @property
    def amendments(self) -> Path: return self.run_dir / "amendments.jsonl"
    @property
    def logs_dir(self) -> Path: return self.run_dir / "logs"

    def relative(self, path: Path) -> str:
        return str(Path(path).resolve().relative_to(self.run_dir.resolve()))

    def ensure(self) -> None:
        for directory in (self.locks_dir, self.lock_snapshots_dir, self.manifests_dir, self.state_dir,
                          self.control_dir, self.tables_dir, self.figures_dir, self.reports_dir, self.logs_dir,
                          self.local_state_dir):
            directory.mkdir(parents=True, exist_ok=True)
        for private in (self.private_dir, self.grades_dir, self.grade_cache_dir):
            private.mkdir(parents=True, exist_ok=True)
            os.chmod(private, PRIVATE_DIR_MODE)
