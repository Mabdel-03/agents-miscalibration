"""Global dispatcher: fair admission, crash safety, and legacy-array reconciliation."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents_scaling.config import ExperimentCell
from agents_scaling.experiment.manifest import ManifestSnapshot, load_manifest

SLURM = Path(__file__).resolve().parents[1] / "slurm"
sys.path.insert(0, str(SLURM))

import dispatch_sweeps as ds  # noqa: E402
import launch_dispatcher as ld  # noqa: E402
from slurm import schema5_control as control  # noqa: E402


_SIDECAR_SHA256 = "f" * 64


class _QuestionCatalogStub:
    sidecar_sha256 = _SIDECAR_SHA256
    frozen = SimpleNamespace()

    def __init__(self):
        self.snapshot = None

    def verify_unchanged(self):
        return None

    def questions_for(self, _cell):
        return (SimpleNamespace(qid="q"),)


def _cell(
    seed: int,
    *,
    model: str = "8B",
    topology: str = "single_agent",
    n_agents: int = 1,
    context: str = "artifact_only",
) -> ExperimentCell:
    return ExperimentCell.from_dict(
        {
            "model_size": model,
            "context_share_level": context,
            "prompt_complexity_level": 0,
            "reasoning_level": "off",
            "topology": topology,
            "benchmark": "truthfulqa",
            "n_agents": n_agents,
            "rounds": 2,
            "n_samples": 1,
            "temperature": 0.0,
            "n_questions": 1,
            "seed": seed,
        }
    )


def _run(tmp_path: Path, run_id: str, cells: list[ExperimentCell]) -> ds.RunSpec:
    root = tmp_path / run_id
    root.mkdir(parents=True)
    (root / "cells.json").write_text(
        json.dumps([cell.to_dict() for cell in cells]), encoding="utf-8"
    )
    snapshot = load_manifest(root)
    catalog = _QuestionCatalogStub()
    catalog.snapshot = snapshot
    return ds.RunSpec(
        run_id,
        root,
        snapshot,
        None,
        root,
        1.0,
        catalog,
    )


def _candidate(
    run_id: str,
    index: int,
    *,
    pool: str = "/pool",
    cost: int = 1,
    model: str = "8B",
) -> ds.Candidate:
    cell = _cell(index, model=model)
    return ds.Candidate(
        run_id=run_id,
        run_root=f"/results/{run_id}",
        source_index=index,
        cell=cell,
        manifest_sha256="abc",
        server_pool_arg=None,
        server_pool_root=pool,
        serving_profile=model,
        fanout_cost=cost,
        benchmark_contracts_sha256=_SIDECAR_SHA256,
    )


def _poll_args(tmp_path: Path, run: ds.RunSpec) -> SimpleNamespace:
    state = tmp_path / "state"
    return SimpleNamespace(
        assume_total_jobs=0,
        assume_cell_jobs=0,
        qos_limit=448,
        reserve=64,
        max_batch=24,
        no_probe_servers=True,
        probe_servers=False,
        probe_timeout=0.01,
        server_capacity=[f"{run.run_id}:8B=1"],
        validation_budget=4,
        fanout_slots_per_server=24,
        state_dir=state,
        ledger_path=state / "ledger.json",
        cell_partition="mit_preemptable",
        cell_time="01:00:00",
        cell_mem="2G",
    )


def test_global_qos_reserve_and_absolute_gate():
    assert ds.available_cell_slots(total_jobs=64, active_cell_jobs=0) == 24
    assert ds.available_cell_slots(total_jobs=430, active_cell_jobs=370) == 14
    assert ds.available_cell_slots(total_jobs=400, active_cell_jobs=384) == 0
    assert ds.available_cell_slots(total_jobs=448, active_cell_jobs=10) == 0


def test_weighted_drr_respects_run_weights_and_shared_budget():
    candidates = [
        *[_candidate("a", index) for index in range(30)],
        *[_candidate("b", index + 100) for index in range(30)],
    ]
    profile = ("/pool", "8B")
    result = ds.plan_admission(
        candidates,
        deficits={},
        cursor=0,
        max_tasks=12,
        live_servers={profile: 1},
        profile_headroom={profile: 12},
        run_weights={"a": 2.0, "b": 1.0},
    )
    counts = {run: sum(c.run_id == run for c in result.selected) for run in ("a", "b")}
    assert counts == {"a": 8, "b": 4}
    assert sum(c.fanout_cost for c in result.selected) <= 12


def test_drr_skips_expensive_head_when_only_small_cell_fits():
    expensive = _candidate("a", 0, cost=6)
    cheap = _candidate("a", 1, cost=2)
    profile = expensive.profile_key
    result = ds.plan_admission(
        [expensive, cheap],
        deficits={ds._key(expensive.group_key): 10.0},
        cursor=0,
        max_tasks=2,
        live_servers={profile: 1},
        profile_headroom={profile: 2},
    )
    assert [candidate.source_index for candidate in result.selected] == [1]


def test_drr_no_server_means_no_admission():
    candidate = _candidate("a", 0)
    result = ds.plan_admission(
        [candidate],
        deficits={},
        cursor=0,
        max_tasks=24,
        live_servers={candidate.profile_key: 0},
        profile_headroom={candidate.profile_key: 100},
    )
    assert result.selected == ()


def test_drr_admits_backlogged_model_groups_without_cross_profile_oversubscription():
    eight = [_candidate("run", index, model="8B") for index in range(20)]
    fourteen = [_candidate("run", 100 + index, model="14B") for index in range(20)]
    capacities = {("/pool", "8B"): 1, ("/pool", "14B"): 1}
    result = ds.plan_admission(
        [*eight, *fourteen],
        deficits={},
        cursor=0,
        max_tasks=8,
        live_servers=capacities,
        profile_headroom={key: 4 for key in capacities},
    )
    assert {candidate.serving_profile for candidate in result.selected} == {"8B", "14B"}
    for profile in capacities:
        assert sum(candidate.profile_key == profile for candidate in result.selected) <= 4


def test_queue_interleaves_scientific_strata_instead_of_manifest_source_order():
    cells = []
    for index, (reasoning, benchmark) in enumerate(
        [
            ("off", "gpqa"),
            ("off", "gpqa"),
            ("off", "gpqa"),
            ("unlimited", "truthfulqa"),
            ("unlimited", "truthfulqa"),
        ]
    ):
        cell = ExperimentCell.from_dict(
            {
                **_cell(index).to_dict(),
                "reasoning_level": reasoning,
                "benchmark": benchmark,
            }
        )
        cells.append(
            ds.Candidate(
                run_id="run",
                run_root="/results/run",
                source_index=index,
                cell=cell,
                manifest_sha256="abc",
                server_pool_arg=None,
                server_pool_root="/pool",
                serving_profile="8B",
                fanout_cost=1,
                benchmark_contracts_sha256=_SIDECAR_SHA256,
            )
        )

    ordered = ds._stratified_candidate_order(cells)

    assert [candidate.source_index for candidate in ordered] == [0, 3, 1, 4, 2]


def test_fairness_state_survives_restart_and_rotates_first_group():
    candidates = [_candidate("a", 0), _candidate("b", 1)]
    profile = candidates[0].profile_key
    first = ds.plan_admission(
        candidates,
        deficits={},
        cursor=0,
        max_tasks=1,
        live_servers={profile: 1},
        profile_headroom={profile: 10},
    )
    persisted = json.loads(json.dumps({"deficits": first.deficits, "cursor": first.cursor}))
    second = ds.plan_admission(
        candidates,
        deficits=persisted["deficits"],
        cursor=persisted["cursor"],
        max_tasks=1,
        live_servers={profile: 1},
        profile_headroom={profile: 10},
    )
    assert first.selected[0].run_id != second.selected[0].run_id


def test_validation_allocation_rotates_remainder_across_restarts():
    first = ds.plan_validation_allocation(
        ["n7", "base", "extension"],
        demand={"base": 10, "extension": 10, "n7": 10},
        budget=4,
        next_run_id=None,
    )
    assert first.allocations == {"base": 2, "extension": 1, "n7": 1}
    assert first.next_run_id == "extension"
    assert first.run_order == ("base", "extension", "n7", "base")

    persisted = json.loads(json.dumps({"next_run_id": first.next_run_id}))
    second = ds.plan_validation_allocation(
        ["extension", "base", "n7"],
        demand={"base": 10, "extension": 10, "n7": 10},
        budget=4,
        next_run_id=persisted["next_run_id"],
    )
    assert second.allocations == {"base": 1, "extension": 2, "n7": 1}
    assert second.next_run_id == "n7"
    assert second.run_order == ("extension", "n7", "base", "extension")


def test_validation_allocation_is_work_conserving_when_runs_have_no_demand():
    allocation = ds.plan_validation_allocation(
        ["base", "extension", "n7"],
        demand={"base": 1, "extension": 5, "n7": 0},
        budget=4,
        next_run_id=None,
    )
    assert allocation.allocations == {"base": 1, "extension": 3, "n7": 0}
    assert sum(allocation.allocations.values()) == 4


def test_starvation_warning_fires_on_second_backlogged_poll_only():
    records = {"a": {}, "b": {}}
    records, warnings = ds.update_starvation_counters(
        records, backlogged_runs={"a"}, admitted_runs=set(), poll_number=1
    )
    assert warnings == ()
    records, warnings = ds.update_starvation_counters(
        records, backlogged_runs={"a"}, admitted_runs=set(), poll_number=2
    )
    assert warnings == ("a",)
    records, warnings = ds.update_starvation_counters(
        records, backlogged_runs={"a"}, admitted_runs=set(), poll_number=3
    )
    assert warnings == ()
    records, _ = ds.update_starvation_counters(
        records, backlogged_runs={"a"}, admitted_runs={"a"}, poll_number=4
    )
    assert records["a"]["backlogged_polls_without_admission"] == 0


def test_squeue_parser_uses_array_job_task_and_command_fields():
    [row] = ds.parse_squeue(
        "17866869|1304|17866869_1304|asys-cells|RUNNING|/r/run/chunk.sbatch\n"
    )
    assert row.array_job_id == "17866869"
    assert row.array_task_id == 1304
    assert row.job_name == "asys-cells"
    assert row.command == "/r/run/chunk.sbatch"


def test_bare_legacy_array_is_resolved_from_command_and_uses_actual_pool(tmp_path):
    run_a = _run(tmp_path, "run_a", [_cell(0)])
    original_b = _run(tmp_path, "run_b", [_cell(1)])
    # New dispatches from run_b would share run_a's servers; its legacy job did not.
    run_b = ds.RunSpec(
        original_b.run_id,
        original_b.run_root,
        original_b.manifest,
        "run_a",
        run_a.run_root,
        1.0,
    )
    command = run_b.run_root / "chunk_000_0-0.sbatch"
    row = ds.QueueRow("10", 0, "10_0", "asys-cells", "RUNNING", str(command))
    active, load, _, unmappable = ds._active_cells(
        [row], ds._empty_ledger(), [run_a, run_b], now=10.0
    )
    assert set(active) == {("run_b", _cell(1).cell_id)}
    assert load[(str(run_b.run_root), "8B")] == 1
    assert (str(run_a.run_root), "8B") not in load
    assert unmappable == []


def test_duplicate_legacy_tasks_count_twice_in_active_load(tmp_path):
    run = _run(tmp_path, "run", [_cell(0)])
    command = str(run.run_root / "chunk.sbatch")
    rows = [
        ds.QueueRow("10", 0, "10_0", "asys-cells", "RUNNING", command),
        ds.QueueRow("11", 0, "11_0", "asys-cells", "RUNNING", command),
    ]
    active, load, _, unmappable = ds._active_cells(
        rows, ds._empty_ledger(), [run], now=10.0
    )
    assert len(active) == 1
    assert load[(str(run.run_root), "8B")] == 2
    assert unmappable == []


def test_unmappable_cell_job_fails_closed_in_poll(tmp_path, monkeypatch):
    run = _run(tmp_path, "run", [_cell(0)])
    args = _poll_args(tmp_path, run)
    args.assume_total_jobs = None
    args.assume_cell_jobs = None
    monkeypatch.setattr(
        ds,
        "_query_squeue",
        lambda: [ds.QueueRow("10", 0, "10_0", "asys-cells", "RUNNING", "/unknown/x")],
    )
    outcome = ds._dispatch_poll(args, [run], ds._empty_ledger(), dry_run=True)
    assert outcome["report"]["qos"]["available_slots"] == 0
    assert outcome["report"]["selected"] == []
    assert outcome["report"]["unmappable_cell_jobs"]


def test_dispatcher_orphan_is_recovered_from_batch_command(tmp_path):
    run = _run(tmp_path, "run", [_cell(0)])
    candidate = ds.Candidate(
        "run", str(run.run_root), 0, run.manifest.cells[0], run.manifest.sha256,
        None, str(run.run_root), "8B", 1, _SIDECAR_SHA256,
    )
    _, _, sbatch_path, _ = ds._write_batch(
        tmp_path / "state",
        [candidate],
        partition="mit_preemptable",
        time_limit="01:00:00",
        memory="2G",
        now=10.0,
    )
    row = ds.QueueRow(
        "99", 0, "99_0", "asys-dispatch-deadbeef", "PENDING", str(sbatch_path)
    )
    ledger = ds._empty_ledger()
    active, load, warnings, unmappable = ds._active_cells(
        [row], ledger, [run], now=20.0
    )
    assert ("run", candidate.cell_id) in active
    assert load[(str(run.run_root), "8B")] == 1
    assert ledger["jobs"]["99"]["state"] == "active"
    assert any("reconciled" in warning for warning in warnings)
    assert unmappable == []


def test_unrecoverable_dispatcher_row_is_unmappable_and_fail_closed(tmp_path):
    run = _run(tmp_path, "run", [_cell(0)])
    row = ds.QueueRow(
        "99", 0, "99_0", "asys-dispatch-deadbeef", "RUNNING", "/missing/batch.sbatch"
    )
    active, load, _, unmappable = ds._active_cells(
        [row], ds._empty_ledger(), [run], now=20.0
    )
    assert active == {}
    assert load == {}
    assert unmappable


def test_capacity_excludes_incomplete_endpoint_generation_records(tmp_path, monkeypatch):
    server_dir = tmp_path / "pool" / "servers" / "8B"
    server_dir.mkdir(parents=True)
    entries = [
        ("authoritative", 8001, "100"),
        ("stale-job", 8002, "999"),
        ("legacy-null", 8003, None),
    ]
    for host, port, job_id in entries:
        (server_dir / f"{host}_{port}.json").write_text(
            json.dumps(
                {
                    "model_size": "8B",
                    "hf_id": "Qwen/Qwen3-8B",
                    "host": host,
                    "port": port,
                    "slurm_job_id": job_id,
                }
            )
        )
    attempts = {"legacy-null": 0, "authoritative": 0, "stale-job": 0}

    def alive(host, _port, timeout):
        attempts[host] += 1
        return host == "legacy-null" and attempts[host] == 3

    monkeypatch.setattr(ds.healthcheck, "is_alive", alive)
    monkeypatch.setattr(
        ds,
        "active_slurm_allocations",
        lambda _ids: {"100": frozenset({"authoritative"})},
    )
    key = (str(tmp_path / "pool"), "8B")
    snapshot = ds.discover_capacity(
        [key], probe=False, probe_timeout=0.01, active_job_ids={"100"}, probe_attempts=3
    )
    assert snapshot.live_servers[key] == 0
    assert attempts == {"legacy-null": 0, "authoritative": 0, "stale-job": 0}


def test_capacity_rejects_stale_endpoint_from_requeued_job_node(tmp_path, monkeypatch):
    server_dir = tmp_path / "pool" / "servers" / "8B"
    server_dir.mkdir(parents=True)
    for host in ("old-node", "new-node"):
        (server_dir / f"{host}_8001.json").write_text(
            json.dumps(
                {
                    "model_size": "8B",
                    "hf_id": "Qwen/Qwen3-8B",
                    "host": host,
                    "port": 8001,
                    "slurm_job_id": "100",
                    "started_at": 10.0,
                    "serving_profile": "8B",
                    "served_model_name": "8B",
                    "max_model_len": 32768,
                    "tp_size": 1,
                }
            )
        )
    monkeypatch.setattr(
        ds,
        "active_slurm_allocations",
        lambda _ids: {"100": frozenset({"new-node"})},
    )
    monkeypatch.setattr(
        ds.healthcheck,
        "is_alive",
        lambda *_args, **_kwargs: pytest.fail("exact Slurm authority should not probe"),
    )

    key = (str(tmp_path / "pool"), "8B")
    snapshot = ds.discover_capacity([key], probe=False, active_job_ids={"100"})
    assert snapshot.live_servers[key] == 1


def test_capacity_generation_changes_on_same_address_process_restart(
    tmp_path, monkeypatch
):
    server_dir = tmp_path / "pool" / "servers" / "8B"
    server_dir.mkdir(parents=True)
    path = server_dir / "nodeA_8001.json"

    def write_entry(started_at):
        path.write_text(
            json.dumps(
                {
                    "model_size": "8B",
                    "hf_id": "Qwen/Qwen3-8B",
                    "host": "nodeA",
                    "port": 8001,
                    "slurm_job_id": "100",
                    "started_at": started_at,
                    "serving_profile": "8B",
                    "served_model_name": "8B",
                    "max_model_len": 32768,
                    "tp_size": 1,
                }
            )
        )

    monkeypatch.setattr(
        ds,
        "active_slurm_allocations",
        lambda _ids: {"100": frozenset({"nodeA"})},
    )
    key = (str(tmp_path / "pool"), "8B")
    write_entry(100.0)
    first = ds.discover_capacity([key], probe=False, active_job_ids={"100"})
    unchanged = ds.discover_capacity([key], probe=False, active_job_ids={"100"})
    write_entry(101.0)  # same host:port and allocation, replacement server process
    restarted = ds.discover_capacity([key], probe=False, active_job_ids={"100"})

    assert first.server_pool_generation[key] == unchanged.server_pool_generation[key]
    assert first.server_pool_generation[key] != restarted.server_pool_generation[key]


def test_capacity_rejects_registry_record_with_wrong_profile_layout(tmp_path, monkeypatch):
    server_dir = tmp_path / "pool" / "servers" / "32B-long"
    server_dir.mkdir(parents=True)
    (server_dir / "nodeA_8001.json").write_text(
        json.dumps(
            {
                "model_size": "32B",
                "hf_id": "Qwen/Qwen3-32B",
                "host": "nodeA",
                "port": 8001,
                "slurm_job_id": "100",
                # A standard/partial 32B record cannot supply the 40K TP=2 contract.
                "serving_profile": "32B",
                "served_model_name": "32B",
                "max_model_len": 16384,
                "tp_size": 1,
            }
        )
    )
    monkeypatch.setattr(
        ds,
        "active_slurm_allocations",
        lambda _ids: {"100": frozenset({"nodeA"})},
    )

    key = (str(tmp_path / "pool"), "32B-long")
    snapshot = ds.discover_capacity([key], probe=False, active_job_ids={"100"})
    assert snapshot.live_servers[key] == 0


def test_schema5_capacity_counts_only_exact_frozen_pool_replica(tmp_path, monkeypatch):
    from agents_scaling.serving.fleet_contract import load_fleet_contract
    from agents_scaling.serving.launch_server import _port_for
    from agents_scaling.serving.model_contracts import load_model_contracts

    contracts = load_model_contracts()
    fleet = load_fleet_contract(None, model_contracts=contracts)
    replica = fleet.for_replica("8B", 0)
    pool = tmp_path / "server_pools" / "schema5-v1"
    server_dir = pool / "servers" / "8B"
    server_dir.mkdir(parents=True)
    path = server_dir / "nodeA.json"
    entry = {
        "model_size": "8B",
        "hf_id": "Qwen/Qwen3-8B",
        "host": "nodeA",
        "port": _port_for("8B", 0),
        "slurm_job_id": "100",
        "started_at": 10.0,
        "serving_profile": "8B",
        "served_model_name": "8B",
        "max_model_len": 32768,
        "tp_size": 1,
        "server_pool_id": "schema5-v1",
        "replica_id": replica.replica_id,
        "replica_index": replica.replica_index,
        "fleet_contract_sha256": "0" * 64,
    }
    path.write_text(json.dumps(entry), encoding="utf-8")
    monkeypatch.setattr(
        ds,
        "active_slurm_allocations",
        lambda _ids: {"100": frozenset({"nodeA"})},
    )
    key = (str(pool), "8B")

    rejected = ds.discover_capacity(
        [key], probe=False, active_job_ids={"100"}, frozen_fleet=fleet
    )
    assert rejected.live_servers[key] == 0

    entry["fleet_contract_sha256"] = fleet.sha256
    path.write_text(json.dumps(entry), encoding="utf-8")
    accepted = ds.discover_capacity(
        [key], probe=False, active_job_ids={"100"}, frozen_fleet=fleet
    )
    assert accepted.live_servers[key] == 1

    entry["replica_id"] = fleet.for_replica("8B", 1).replica_id
    path.write_text(json.dumps(entry), encoding="utf-8")
    wrong_logical_replica = ds.discover_capacity(
        [key], probe=False, active_job_ids={"100"}, frozen_fleet=fleet
    )
    assert wrong_logical_replica.live_servers[key] == 0


def test_pre_submit_intent_reserves_tasks_during_visibility_window(tmp_path):
    run = _run(tmp_path, "run", [_cell(0)])
    candidate = ds.Candidate(
        "run", str(run.run_root), 0, run.manifest.cells[0], run.manifest.sha256,
        None, str(run.run_root), "8B", 1, _SIDECAR_SHA256,
    )
    ledger = ds._empty_ledger(now=10.0)
    ledger["intents"]["batch"] = {
        "state": "prepared", "created_at": 10.0, "tasks": [ds._task_from_candidate(candidate)]
    }
    active, load, _, unmappable = ds._active_cells(
        [], ledger, [run], now=20.0, visibility_grace_s=30.0
    )
    assert ("run", candidate.cell_id) in active
    assert load[candidate.profile_key] == 1
    assert unmappable == []


def test_incremental_validation_cache_never_rescans_unchanged_jsonl(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0), _cell(1), _cell(2)])
    for cell in run.manifest.cells[:2]:
        cdir = run.run_root / "cells" / cell.cell_id
        cdir.mkdir(parents=True)
        (cdir / "results.jsonl").write_text("placeholder\n", encoding="utf-8")
    calls: list[str] = []

    def fake_status(cell, *_args, **kwargs):
        assert kwargs["expected_qids"] == ("q",)
        assert tuple(question.qid for question in kwargs["expected_questions"]) == (
            "q",
        )
        calls.append(cell.cell_id)
        return SimpleNamespace(
            status=ds.CompletionState.PARTIAL,
            eligible_for_retry=True,
            next_eligible_at=None,
        )

    monkeypatch.setattr(ds, "get_completion_status", fake_status)
    monkeypatch.setattr(ds, "is_cell_active", lambda _path: False)
    ledger = ds._empty_ledger()
    profile = (str(run.run_root), "8B")
    capacity = ds.CapacitySnapshot({profile: 1}, {profile: "generation"})

    _, first_counts, _, _ = ds._scan_candidates(
        [run], active={}, active_task_load={}, capacity=capacity, ledger=ledger,
        now=10.0, code_version="abc", validation_budget=1,
    )
    assert len(calls) == 1
    assert first_counts["run"]["unvalidated"] == 1

    ds._scan_candidates(
        [run], active={}, active_task_load={}, capacity=capacity, ledger=ledger,
        now=20.0, code_version="abc", validation_budget=1,
    )
    assert len(calls) == 2
    ds._scan_candidates(
        [run], active={}, active_task_load={}, capacity=capacity, ledger=ledger,
        now=30.0, code_version="abc", validation_budget=1,
    )
    assert len(calls) == 2  # both artifact-bearing cells reused cached semantics

    first_dir = run.run_root / "cells" / run.manifest.cells[0].cell_id
    (first_dir / "results.jsonl").write_text("changed-and-longer\n", encoding="utf-8")
    ds._scan_candidates(
        [run], active={}, active_task_load={}, capacity=capacity, ledger=ledger,
        now=40.0, code_version="abc", validation_budget=1,
    )
    assert len(calls) == 3


def test_semantic_validation_budget_is_fair_and_restart_safe_across_runs(
    tmp_path, monkeypatch
):
    specs = [
        _run(tmp_path, run_id, [_cell(seed) for seed in range(3)])
        for run_id in ("base", "extension", "n7")
    ]
    for spec in specs:
        for cell in spec.manifest.cells:
            cdir = spec.run_root / "cells" / cell.cell_id
            cdir.mkdir(parents=True)
            (cdir / "results.jsonl").write_text("placeholder\n", encoding="utf-8")

    calls: list[str] = []

    def fake_status(_cell, cell_dir, *_args, **_kwargs):
        calls.append(Path(cell_dir).parent.parent.name)
        return SimpleNamespace(
            status=ds.CompletionState.PARTIAL,
            eligible_for_retry=True,
            next_eligible_at=None,
        )

    monkeypatch.setattr(ds, "get_completion_status", fake_status)
    monkeypatch.setattr(ds, "is_cell_active", lambda _path: False)
    ledger = ds._empty_ledger()
    ledger["fairness"] = {"cursor": 7, "deficits": {"admission-sentinel": 3.5}}
    capacities = {
        (str(spec.run_root), "8B"): 1
        for spec in specs
    }
    capacity = ds.CapacitySnapshot(
        capacities,
        {profile: f"generation-{index}" for index, profile in enumerate(capacities)},
    )

    ds._scan_candidates(
        specs,
        active={},
        active_task_load={},
        capacity=capacity,
        ledger=ledger,
        now=10.0,
        code_version="abc",
        validation_budget=4,
    )
    assert {run_id: calls.count(run_id) for run_id in ("base", "extension", "n7")} == {
        "base": 2,
        "extension": 1,
        "n7": 1,
    }
    assert calls == ["base", "extension", "n7", "base"]
    assert ledger["validation_fairness"]["next_run_id"] == "extension"
    assert ledger["fairness"] == {
        "cursor": 7,
        "deficits": {"admission-sentinel": 3.5},
    }

    restarted = json.loads(json.dumps(ledger))
    first_poll_calls = len(calls)
    ds._scan_candidates(
        list(reversed(specs)),
        active={},
        active_task_load={},
        capacity=capacity,
        ledger=restarted,
        now=20.0,
        code_version="abc",
        validation_budget=4,
    )
    second_calls = calls[first_poll_calls:]
    assert {
        run_id: second_calls.count(run_id)
        for run_id in ("base", "extension", "n7")
    } == {"base": 1, "extension": 2, "n7": 1}
    assert second_calls == ["extension", "n7", "base", "extension"]
    assert restarted["validation_fairness"]["next_run_id"] == "n7"
    assert restarted["validation_fairness"]["last_used"] == 4
    assert restarted["fairness"] == ledger["fairness"]


def test_cached_complete_state_is_invalidated_by_code_contract_change(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0)])
    cell = run.manifest.cells[0]
    cdir = run.run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    (cdir / "results.jsonl").write_text("placeholder\n", encoding="utf-8")
    calls: list[str] = []

    def fake_status(observed_cell, *_args, **_kwargs):
        calls.append(observed_cell.cell_id)
        return SimpleNamespace(
            status=ds.CompletionState.COMPLETE,
            eligible_for_retry=False,
            next_eligible_at=None,
        )

    monkeypatch.setattr(ds, "get_completion_status", fake_status)
    monkeypatch.setattr(ds, "is_cell_active", lambda _path: False)
    ledger = ds._empty_ledger()
    profile = (str(run.run_root), "8B")
    capacity = ds.CapacitySnapshot({profile: 1}, {profile: "generation"})

    for now, code_version in ((10.0, "old"), (20.0, "old"), (30.0, "new")):
        ds._scan_candidates(
            [run], active={}, active_task_load={}, capacity=capacity, ledger=ledger,
            now=now, code_version=code_version, validation_budget=1,
        )

    assert calls == [cell.cell_id, cell.cell_id]
    cached = ledger["cells"][ds._key(("run", cell.cell_id))]
    assert cached["validation_context"]["code_version"] == "new"
    assert cached["validation_context"]["manifest_sha256"] == run.manifest.sha256


def test_cached_validation_errors_do_not_starve_later_manifest_cells(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0), _cell(1), _cell(2)])
    for cell in run.manifest.cells:
        cdir = run.run_root / "cells" / cell.cell_id
        cdir.mkdir(parents=True)
        (cdir / "results.jsonl").write_text("bad\n")
    calls: list[str] = []

    def fail_validation(cell, *_args, **_kwargs):
        calls.append(cell.cell_id)
        raise RuntimeError("deterministic validator failure")

    monkeypatch.setattr(ds, "get_completion_status", fail_validation)
    monkeypatch.setattr(ds, "is_cell_active", lambda _path: False)
    ledger = ds._empty_ledger()
    profile = (str(run.run_root), "8B")
    capacity = ds.CapacitySnapshot({profile: 1}, {profile: "generation"})
    for now in (10.0, 20.0, 30.0):
        ds._scan_candidates(
            [run], active={}, active_task_load={}, capacity=capacity, ledger=ledger,
            now=now, code_version="abc", validation_budget=1, validation_retry_s=1000.0,
        )
    assert calls == [cell.cell_id for cell in run.manifest.cells]
    ds._scan_candidates(
        [run], active={}, active_task_load={}, capacity=capacity, ledger=ledger,
        now=40.0, code_version="abc", validation_budget=1, validation_retry_s=1000.0,
    )
    assert len(calls) == 3


def test_benchmark_question_drift_aborts_candidate_scan(tmp_path, monkeypatch):
    run = _run(tmp_path, "run", [_cell(0)])
    cell = run.manifest.cells[0]
    cdir = run.run_root / "cells" / cell.cell_id
    cdir.mkdir(parents=True)
    (cdir / "results.jsonl").write_text("artifact\n", encoding="utf-8")

    class DriftedCatalog(_QuestionCatalogStub):
        def questions_for(self, _cell):
            raise ds.BenchmarkContractError("normalized Question contract drift")

    drifted = ds.RunSpec(
        run.run_id,
        run.run_root,
        run.manifest,
        run.server_pool_arg,
        run.server_pool_root,
        run.weight,
        DriftedCatalog(),
    )
    monkeypatch.setattr(ds, "is_cell_active", lambda _path: False)
    profile = (str(run.run_root), "8B")
    capacity = ds.CapacitySnapshot({profile: 1}, {profile: "generation"})

    with pytest.raises(ds.DispatcherError, match="benchmark Question drift"):
        ds._scan_candidates(
            [drifted],
            active={},
            active_task_load={},
            capacity=capacity,
            ledger=ds._empty_ledger(),
            now=10.0,
            code_version="abc",
            validation_budget=1,
        )


def test_atomic_ledger_replace_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "ledger.json"
    ds._atomic_write_json(path, {"version": 1})
    original_replace = ds.os.replace

    def fail_replace(source, destination):
        if Path(destination) == path:
            raise OSError("simulated crash")
        return original_replace(source, destination)

    monkeypatch.setattr(ds.os, "replace", fail_replace)
    with pytest.raises(OSError):
        ds._atomic_write_json(path, {"version": 2})
    assert json.loads(path.read_text()) == {"version": 1}


def test_pre_fairness_ledger_loads_with_restart_safe_validation_cursor(tmp_path):
    path = tmp_path / "ledger.json"
    legacy = ds._empty_ledger(now=10.0)
    legacy.pop("validation_fairness")
    path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = ds._load_ledger(path)
    assert loaded["validation_fairness"] == {"next_run_id": None}
    assert loaded["fairness"] == legacy["fairness"]


def test_singleton_lock_rejects_second_coordinator(tmp_path):
    with ds.singleton_lock(tmp_path):
        with pytest.raises(ds.SingletonAlreadyRunning):
            with ds.singleton_lock(tmp_path):
                pass


def test_results_root_lock_rejects_alternate_state_directory(tmp_path):
    first_state = tmp_path / "state-a"
    second_state = tmp_path / "state-b"
    with ds.global_dispatcher_lock(tmp_path):
        with ds.singleton_lock(first_state):
            with pytest.raises(ds.SingletonAlreadyRunning):
                with ds.global_dispatcher_lock(tmp_path):
                    with ds.singleton_lock(second_state):
                        pass
    assert not second_state.exists()


def test_live_dispatch_loser_writes_no_state_and_queues_no_successor(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0)])
    monkeypatch.setattr(
        ds,
        "VerifiedQuestionCatalog",
        lambda *_args, **_kwargs: _QuestionCatalogStub(),
    )
    losing_state = tmp_path / "alternate-state"
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        ds, "_queue_successor", lambda *_args: pytest.fail("loser queued a successor")
    )
    with ds.global_dispatcher_lock(tmp_path):
        with pytest.raises(ds.SingletonAlreadyRunning):
            ds.main(
                [
                    "dispatch",
                    "--once",
                    "--run",
                    f"run={run.run_root}",
                    "--results-root",
                    str(tmp_path),
                    "--state-dir",
                    str(losing_state),
                    "--successor-sbatch",
                    str(tmp_path / "control.sbatch"),
                ]
            )
    assert not losing_state.exists()


def test_live_dispatch_lock_order_is_global_then_state(monkeypatch, tmp_path):
    run = _run(tmp_path, "run", [_cell(0)])
    monkeypatch.setattr(
        ds,
        "VerifiedQuestionCatalog",
        lambda *_args, **_kwargs: _QuestionCatalogStub(),
    )
    events = []

    @contextmanager
    def global_lock(_root):
        events.append("enter-global")
        try:
            yield
        finally:
            events.append("exit-global")

    @contextmanager
    def state_lock(_state):
        events.append("enter-state")
        try:
            yield
        finally:
            events.append("exit-state")

    monkeypatch.setattr(ds, "global_dispatcher_lock", global_lock)
    monkeypatch.setattr(ds, "singleton_lock", state_lock)
    monkeypatch.setattr(
        ds,
        "_dispatch_poll",
        lambda _args, _specs, ledger, dry_run: {
            "ledger": ledger,
            "report": {
                "starvation_warnings": [],
                "warnings": [],
                "submission_error": None,
                "validation_errors": [],
            },
        },
    )
    assert (
        ds.main(
            [
                "dispatch",
                "--once",
                "--run",
                f"run={run.run_root}",
                "--results-root",
                str(tmp_path),
                "--state-dir",
                str(tmp_path / "state"),
            ]
        )
        == 0
    )
    assert events == ["enter-global", "enter-state", "exit-state", "exit-global"]


def test_manifest_pin_refuses_same_run_id_with_changed_manifest(tmp_path):
    run = _run(tmp_path, "run", [_cell(0)])
    ledger = ds._empty_ledger()
    ds._register_run_pins(ledger, [run])
    changed = ManifestSnapshot(run.manifest.path, run.manifest.cells, "different")
    changed_spec = ds.RunSpec("run", run.run_root, changed, None, run.run_root, 1.0)
    with pytest.raises(ds.DispatcherError, match="immutable manifest changed"):
        ds._register_run_pins(ledger, [changed_spec])


def test_microbatch_template_renders_explicit_resources_and_max_24(tmp_path):
    path = tmp_path / "batch.json"
    text = ds._render_batch_sbatch(
        path,
        n_tasks=24,
        partition="mit_preemptable",
        time_limit="01:00:00",
        memory="2G",
        log_dir=tmp_path,
        batch_tag="abc",
    )
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=4G" in text
    assert "#SBATCH --time=12:00:00" in text
    assert "#SBATCH --signal=B:USR1@1200" in text
    assert "#SBATCH --no-requeue" in text
    assert "#SBATCH --array=0-23%24" in text
    assert '--batch-manifest-sha256 "' + "0" * 64 + '" \\' in text
    assert 'mamba activate "$ASYS_HARNESS_ENV"' in text
    with pytest.raises(ValueError):
        ds._render_batch_sbatch(
            path, n_tasks=25, partition="p", time_limit="t", memory="2G",
            log_dir=tmp_path, batch_tag="abc",
        )


def test_schema5_microbatch_uses_only_pinned_release_and_harness(
    tmp_path, monkeypatch
):
    release = tmp_path / "immutable-release"
    template = release / "slurm" / "run_dispatch_batch.sbatch.tmpl"
    template.parent.mkdir(parents=True)
    template.write_text(ds.ARRAY_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    harness = tmp_path / "immutable-harness"
    python = harness / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    dispatcher = release / "slurm" / "dispatch_sweeps.py"
    dispatcher.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    execution = {
        "harness_prefix": str(harness.resolve()),
        "python": str(python.resolve()),
        "release_worktree": str(release.resolve()),
        "hf_home": str((tmp_path / "hf-cache").resolve()),
        "dispatcher_script": str(dispatcher.resolve()),
        "batch_template": str(template.resolve()),
    }
    validated = {}
    monkeypatch.setattr(
        control,
        "production_cell_execution_from_state",
        lambda _state: execution,
    )
    monkeypatch.setattr(control, "load_control", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(
        control,
        "validate_production_batch_sbatch",
        lambda state, *, payload, task_count: validated.update(
            state=state, payload=payload, task_count=task_count
        ),
    )

    text = ds._render_batch_sbatch(
        tmp_path / "batch.json",
        n_tasks=2,
        partition="mit_preemptable",
        time_limit="12:00:00",
        memory="4G",
        log_dir=tmp_path,
        batch_tag="schema5",
        control_state_dir=tmp_path / "control",
    )

    assert f"exec {python.resolve()} -u {dispatcher.resolve()} run-task" in text
    assert f"--expected-release-root {release.resolve()}" in text
    assert f"--expected-harness-prefix {harness.resolve()}" in text
    assert "export PYTHONDONTWRITEBYTECODE=1" in text
    assert "export HF_HUB_OFFLINE=1" in text
    assert "export TRANSFORMERS_OFFLINE=1" in text
    assert "export HF_DATASETS_OFFLINE=1" in text
    assert f'export HF_HOME="{execution["hf_home"]}"' in text
    assert f'export ASYS_RELEASE_WORKTREE="{release.resolve()}"' in text
    assert "mamba activate" not in text
    assert "source \"" not in text
    assert "$ASYS_HARNESS_ENV" not in text
    assert validated["payload"] == text
    assert validated["task_count"] == 2


def test_production_dispatcher_parsers_default_to_four_gb_after_pilot():
    dispatch_args = ds._build_parser().parse_args(
        ["dispatch", "--run", "placeholder"]
    )
    launcher_args = ld._parser().parse_args(["--run", "placeholder"])
    assert ds.CELL_CPUS_DEFAULT == 1
    assert ds.CELL_MEM_DEFAULT == "4G"
    assert dispatch_args.cell_mem == "4G"
    assert launcher_args.cell_mem == "4G"


def test_batch_worker_verifies_manifest_and_passes_explicit_server_pool(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0)])
    runtime_environment = {key: "x" for key in ds.PRODUCTION_ENVIRONMENT_KEYS}
    runtime_environment.update(
        {
            "ASYS_ROLLOUT_GENERATION": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    candidate = ds.Candidate(
        "run", str(run.run_root), 0, run.manifest.cells[0], run.manifest.sha256,
        "server_run", str(tmp_path / "server_run"), "8B", 1, _SIDECAR_SHA256,
        tuple(sorted(runtime_environment.items())),
    )
    monkeypatch.setattr(
        ds,
        "load_frozen_benchmark_contracts",
        lambda *_args, **_kwargs: SimpleNamespace(sidecar_sha256=_SIDECAR_SHA256),
    )
    release = Path(__file__).resolve().parents[1]
    command = ds.build_run_one_command(
        ds._task_from_candidate(candidate), release_worktree=release
    )
    assert command[1] == "-I"
    assert command[command.index("--server-pool") + 1] == str(tmp_path / "server_run")
    assert command[command.index("--index") + 1] == "0"
    assert command[command.index("--benchmark-contracts-sha256") + 1] == _SIDECAR_SHA256
    assert command[command.index("--release-worktree") + 1] == str(release)
    assert command[command.index("--model-contract") + 1] == str(
        release / "configs" / "model_contracts.v1.json"
    )
    assert command[command.index("--fleet-contract") + 1] == str(
        release / "configs" / "schema5_fleet.v1.json"
    )
    assert command[command.index("--prompt-root") + 1] == str(
        release / "configs" / "prompts"
    )


def test_batch_worker_rejects_benchmark_sidecar_changed_after_admission(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0)])
    candidate = ds.Candidate(
        "run",
        str(run.run_root),
        0,
        run.manifest.cells[0],
        run.manifest.sha256,
        None,
        str(run.run_root),
        "8B",
        1,
        "e" * 64,
    )
    monkeypatch.setattr(
        ds,
        "load_frozen_benchmark_contracts",
        lambda *_args, **_kwargs: SimpleNamespace(sidecar_sha256=_SIDECAR_SHA256),
    )

    with pytest.raises(ds.DispatcherError, match="sidecar"):
        ds.build_run_one_command(ds._task_from_candidate(candidate))


def test_run_task_pins_results_root_to_verified_manifest_parent(tmp_path, monkeypatch):
    run = _run(tmp_path, "run", [_cell(0)])
    candidate = ds.Candidate(
        "run", str(run.run_root), 0, run.manifest.cells[0], run.manifest.sha256,
        None, str(run.run_root), "8B", 1, _SIDECAR_SHA256,
    )
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(
        json.dumps({"schema_version": 1, "tasks": [ds._task_from_candidate(candidate)]})
    )
    batch_sha256 = ds._seal_dispatch_artifact(batch_path)
    captured = {}

    def fake_exec(_program, _command):
        captured["results_root"] = ds.os.environ["ASYS_RESULTS_ROOT"]

    monkeypatch.setattr(ds.os, "execv", fake_exec)
    monkeypatch.setattr(
        ds,
        "load_frozen_benchmark_contracts",
        lambda *_args, **_kwargs: SimpleNamespace(sidecar_sha256=_SIDECAR_SHA256),
    )
    assert ds._run_task(
        SimpleNamespace(
            batch_manifest=str(batch_path),
            batch_manifest_sha256=batch_sha256,
            index=0,
        )
    ) == 127
    assert captured["results_root"] == str(tmp_path)


def test_run_root_basename_must_match_run_id(tmp_path):
    wrong = _run(tmp_path, "wrong_name", [_cell(0)])
    with pytest.raises(ds.DispatcherError, match="basename must equal run id"):
        ds.load_run_specs(
            [f"expected_id={wrong.run_root}"], results_root=tmp_path
        )


def test_dry_run_is_read_only_and_never_calls_sbatch(tmp_path, monkeypatch, capsys):
    run = _run(tmp_path, "run", [_cell(0)])
    state = tmp_path / "dispatcher-state"
    monkeypatch.setattr(
        ds,
        "VerifiedQuestionCatalog",
        lambda *_args, **_kwargs: _QuestionCatalogStub(),
    )
    monkeypatch.setattr(ds, "_submit_sbatch", lambda _path: pytest.fail("sbatch called"))
    rc = ds.main(
        [
            "dispatch", "--dry-run", "--run", f"run={run.run_root}",
            "--results-root", str(tmp_path), "--state-dir", str(state),
            "--server-capacity", "run:8B=1", "--assume-total-jobs", "0",
            "--assume-cell-jobs", "0",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(report["selected"]) == 1
    assert not state.exists()


def test_load_run_specs_fails_closed_without_benchmark_contract_sidecar(tmp_path):
    run = _run(tmp_path, "run", [_cell(0)])
    with pytest.raises(ds.DispatcherError, match="no valid frozen benchmark Question contract"):
        ds.load_run_specs([f"run={run.run_root}"], results_root=tmp_path)


def test_sbatch_rejection_is_recorded_and_returned_for_next_poll(tmp_path, monkeypatch):
    run = _run(tmp_path, "run", [_cell(0)])
    args = _poll_args(tmp_path, run)
    monkeypatch.setattr(
        ds, "_submit_sbatch", lambda _path: (_ for _ in ()).throw(ds.DispatcherError("qos race"))
    )
    outcome = ds._dispatch_poll(args, [run], ds._empty_ledger(), dry_run=False)
    assert "qos race" in outcome["report"]["submission_error"]
    assert outcome["report"]["submission"] is None
    # Once sbatch has been invoked, even a client-side error is ambiguous: Slurm may
    # have accepted the job before the reply was lost.  Keep the intent fenced until a
    # complete squeue+sacct observation proves non-acceptance after the grace period.
    assert {record["state"] for record in outcome["ledger"]["intents"].values()} == {
        "submitting"
    }


def test_submit_sbatch_binds_cli_intent_and_requires_numeric_job_id(
    tmp_path, monkeypatch
):
    path = tmp_path / "batch-20260721T000000-abc123.sbatch"
    path.write_text("#!/bin/bash\n", encoding="utf-8")
    calls = []

    def accepted(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, "321;cluster\n", "")

    monkeypatch.setattr(ds.subprocess, "run", accepted)
    assert ds._submit_sbatch(path) == "321"
    assert calls[0][0] == [
        "sbatch",
        "--parsable",
        "--comment=asys-schema5-intent:20260721T000000-abc123",
        str(path),
    ]
    assert calls[0][1]["timeout"] == 60.0

    monkeypatch.setattr(
        ds.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "accepted\n", ""),
    )
    with pytest.raises(ds.DispatcherError, match="invalid job id"):
        ds._submit_sbatch(path)


def test_submitting_intent_is_durable_before_sbatch_and_fairness_commits_once(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0)])
    args = _poll_args(tmp_path, run)
    persisted_states = []
    real_atomic_write = ds._atomic_write_json

    def capture_ledger(path, value):
        if Path(path).parent.name == "batches":
            real_atomic_write(path, value)
            return
        if "intents" not in value:
            return
        persisted_states.append(
            [intent["state"] for intent in value.get("intents", {}).values()]
        )

    def submit(_path):
        assert persisted_states[-1] == ["submitting"]
        return "321"

    monkeypatch.setattr(ds, "_atomic_write_json", capture_ledger)
    monkeypatch.setattr(ds, "_submit_sbatch", submit)
    outcome = ds._dispatch_poll(args, [run], ds._empty_ledger(), dry_run=False)
    intent_id, intent = next(iter(outcome["ledger"]["intents"].items()))
    committed = dict(outcome["ledger"]["fairness"])

    assert persisted_states[:2] == [["prepared"], ["submitting"]]
    assert intent["state"] == "submitted"
    assert intent["fairness_committed"] is True
    ds._commit_intent_fairness(outcome["ledger"], intent_id)
    assert outcome["ledger"]["fairness"] == committed


def test_squeue_sacct_intent_reconciliation_rejects_duplicate_jobs_and_commits_once(
    tmp_path
):
    batch_id = "20260721T000000-abc123"
    sbatch = tmp_path / f"batch-{batch_id}.sbatch"
    manifest = sbatch.with_suffix(".json")
    manifest.write_text("{}\n", encoding="utf-8")
    sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    manifest_sha256 = ds._seal_dispatch_artifact(manifest)
    sbatch_sha256 = ds._seal_dispatch_artifact(sbatch)
    task = {"run_id": "run", "cell_id": "cell"}
    ledger = ds._empty_ledger()
    ledger["intents"][batch_id] = {
        "state": "submitting",
        "created_at": 10.0,
        "submit_started_at": 11.0,
        "batch_manifest": str(manifest),
        "batch_manifest_sha256": manifest_sha256,
        "sbatch_path": str(sbatch),
        "sbatch_sha256": sbatch_sha256,
        "tasks": [task],
        "fairness_after": {"cursor": 3, "deficits": {"run": 1.5}},
        "fairness_committed": False,
    }

    def job(job_id):
        return SimpleNamespace(
            job_id=job_id,
            job_name=f"asys-dispatch-{batch_id[-10:]}",
            state="RUNNING",
            comment=f"asys-schema5-intent:{batch_id}",
            command=f"sbatch {sbatch}",
            source="squeue",
            active=True,
        )

    snapshot = SimpleNamespace(jobs=(job("321_0"), job("321_1")))
    warnings, errors = ds._reconcile_schema5_intents(
        ledger, scheduler_snapshot=snapshot, now=20.0
    )
    assert not errors
    assert warnings
    assert ledger["intents"][batch_id]["job_id"] == "321"
    assert ledger["intents"][batch_id]["fairness_committed"] is True
    assert ledger["fairness"] == {"cursor": 3, "deficits": {"run": 1.5}}

    duplicate_ledger = ds._empty_ledger()
    duplicate_ledger["intents"][batch_id] = dict(ledger["intents"][batch_id]) | {
        "state": "submitting",
        "fairness_committed": False,
    }
    _, duplicate_errors = ds._reconcile_schema5_intents(
        duplicate_ledger,
        scheduler_snapshot=SimpleNamespace(jobs=(job("321_0"), job("322_0"))),
        now=20.0,
    )
    assert "ambiguously maps" in duplicate_errors[0]


def test_intent_reconciliation_allows_blank_sacct_comment_only_with_exact_path(
    tmp_path,
):
    batch_id = "20260721T000000-abc123"
    sbatch = (tmp_path / f"batch-{batch_id}.sbatch").resolve()
    manifest = sbatch.with_suffix(".json")
    manifest.write_text("{}\n", encoding="utf-8")
    sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    manifest_sha256 = ds._seal_dispatch_artifact(manifest)
    sbatch_sha256 = ds._seal_dispatch_artifact(sbatch)
    task = {"run_id": "run", "cell_id": "cell"}
    ledger = ds._empty_ledger()
    ledger["intents"][batch_id] = {
        "state": "submitting",
        "created_at": 10.0,
        "submit_started_at": 11.0,
        "batch_manifest": str(manifest),
        "batch_manifest_sha256": manifest_sha256,
        "sbatch_path": str(sbatch),
        "sbatch_sha256": sbatch_sha256,
        "tasks": [task],
        "fairness_after": {"cursor": 1, "deficits": {"run": 0.0}},
        "fairness_committed": False,
    }
    accounting = SimpleNamespace(
        job_id="321",
        job_name=f"asys-dispatch-{batch_id[-10:]}",
        state="COMPLETED",
        comment="",
        command=(
            f"sbatch --comment=asys-schema5-intent:{batch_id} {sbatch}"
        ),
        source="sacct",
        active=False,
    )
    warnings, errors = ds._reconcile_schema5_intents(
        ledger, scheduler_snapshot=SimpleNamespace(jobs=(accounting,)), now=20.0
    )
    assert not errors
    assert warnings
    assert ledger["intents"][batch_id]["job_id"] == "321"

    wrong_path = SimpleNamespace(
        **{
            **accounting.__dict__,
            "job_id": "322",
            "command": "sbatch /tmp/foreign.sbatch",
        }
    )
    second = ds._empty_ledger()
    second["intents"][batch_id] = {
        **ledger["intents"][batch_id],
        "state": "submitting",
        "job_id": None,
        "fairness_committed": False,
    }
    _, errors = ds._reconcile_schema5_intents(
        second, scheduler_snapshot=SimpleNamespace(jobs=(wrong_path,)), now=20.0
    )
    assert errors and "provenance drift" in errors[0]


def test_intent_visibility_grace_starts_at_durable_sbatch_boundary(tmp_path):
    batch_id = "20260721T000000-grace"
    manifest = tmp_path / f"batch-{batch_id}.json"
    sbatch = tmp_path / f"batch-{batch_id}.sbatch"
    manifest.write_text("{}\n", encoding="utf-8")
    sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    ledger = ds._empty_ledger()
    ledger["intents"][batch_id] = {
        "state": "submitting",
        "created_at": 1.0,
        "submit_started_at": 990.0,
        "batch_manifest": str(manifest),
        "batch_manifest_sha256": ds._seal_dispatch_artifact(manifest),
        "sbatch_path": str(sbatch),
        "sbatch_sha256": ds._seal_dispatch_artifact(sbatch),
        "tasks": [],
        "fairness_after": {"cursor": 0, "deficits": {}},
        "fairness_committed": False,
    }

    warnings, errors = ds._reconcile_schema5_intents(
        ledger, scheduler_snapshot=SimpleNamespace(jobs=()), now=1_000.0
    )
    assert not warnings and not errors
    assert ledger["intents"][batch_id]["state"] == "submitting"

    warnings, errors = ds._reconcile_schema5_intents(
        ledger, scheduler_snapshot=SimpleNamespace(jobs=()), now=1_301.0
    )
    assert not errors and warnings
    assert ledger["intents"][batch_id]["state"] == "not_accepted"


def test_known_accepted_intent_cannot_silently_disappear(tmp_path):
    batch_id = "20260721T000000-known"
    manifest = tmp_path / f"batch-{batch_id}.json"
    sbatch = tmp_path / f"batch-{batch_id}.sbatch"
    manifest.write_text("{}\n", encoding="utf-8")
    sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    task = {"run_id": "run", "cell_id": "cell"}
    ledger = ds._empty_ledger()
    ledger["intents"][batch_id] = {
        "state": "submitted",
        "created_at": 1.0,
        "submit_started_at": 990.0,
        "submitted_at": 991.0,
        "job_id": "321",
        "batch_manifest": str(manifest),
        "batch_manifest_sha256": ds._seal_dispatch_artifact(manifest),
        "sbatch_path": str(sbatch),
        "sbatch_sha256": ds._seal_dispatch_artifact(sbatch),
        "tasks": [task],
        "fairness_after": {"cursor": 0, "deficits": {}},
        "fairness_committed": True,
    }
    ledger["jobs"]["321"] = {
        "job_id": "321",
        "batch_id": batch_id,
        "state": "submitted",
        "tasks": [task],
        "batch_manifest_sha256": ledger["intents"][batch_id][
            "batch_manifest_sha256"
        ],
        "sbatch_sha256": ledger["intents"][batch_id]["sbatch_sha256"],
    }

    warnings, errors = ds._reconcile_schema5_intents(
        ledger, scheduler_snapshot=SimpleNamespace(jobs=()), now=1_000.0
    )
    assert not warnings and not errors

    _warnings, errors = ds._reconcile_schema5_intents(
        ledger, scheduler_snapshot=SimpleNamespace(jobs=()), now=1_301.0
    )
    assert errors and "accepted intent" in errors[0]

    ledger["jobs"]["321"]["state"] = "terminal"
    _warnings, errors = ds._reconcile_schema5_intents(
        ledger, scheduler_snapshot=SimpleNamespace(jobs=()), now=1_301.0
    )
    assert not errors


def test_schema5_task_runtime_environment_rejects_missing_or_untrusted_keys():
    environment = {key: "x" for key in ds.PRODUCTION_ENVIRONMENT_KEYS}
    environment.update(
        {
            "ASYS_ROLLOUT_GENERATION": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    assert ds._runtime_environment({"runtime_environment": environment}) == environment
    with pytest.raises(ds.DispatcherError, match="untrusted"):
        ds._runtime_environment(
            {"runtime_environment": environment | {"PATH": "/untrusted"}}
        )
    with pytest.raises(ds.DispatcherError, match="positive"):
        ds._runtime_environment(
            {"runtime_environment": environment | {"ASYS_ROLLOUT_GENERATION": "0"}}
        )


def test_schema5_task_refuses_to_start_without_execution_path_pins(tmp_path):
    environment = {key: "x" for key in ds.PRODUCTION_ENVIRONMENT_KEYS}
    environment.update(
        {
            "ASYS_ROLLOUT_GENERATION": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tasks": [{"runtime_environment": environment}],
            }
        ),
        encoding="utf-8",
    )
    batch_sha256 = ds._seal_dispatch_artifact(batch)
    with pytest.raises(ds.DispatcherError, match="immutable release and harness"):
        ds._run_task(
            SimpleNamespace(
                batch_manifest=str(batch),
                batch_manifest_sha256=batch_sha256,
                index=0,
            )
        )


def test_batch_worker_requires_exact_immutable_manifest_digest(tmp_path):
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "tasks": [{}]}), encoding="utf-8"
    )
    expected_sha256 = ds._seal_dispatch_artifact(manifest)
    assert ds._load_batch_task(
        manifest, 0, expected_sha256=expected_sha256
    ) == {}

    manifest.chmod(0o644)
    with pytest.raises(ds.DispatcherError, match="unsafe or unpinned"):
        ds._load_batch_task(manifest, 0, expected_sha256=expected_sha256)

    manifest.chmod(0o444)
    with pytest.raises(ds.DispatcherError, match="drifted"):
        ds._load_batch_task(manifest, 0, expected_sha256="0" * 64)

    alias = tmp_path / "batch-alias.json"
    alias.symlink_to(manifest)
    with pytest.raises(ds.DispatcherError, match="unsafe or unpinned"):
        ds._load_batch_task(alias, 0, expected_sha256=expected_sha256)


def test_runtime_attestation_rejects_interpreter_prefix_or_release_drift(
    tmp_path, monkeypatch
):
    release = tmp_path / "release"
    script = release / "slurm" / "dispatch_sweeps.py"
    script.parent.mkdir(parents=True)
    script.write_text("# pinned\n", encoding="utf-8")
    harness = tmp_path / "harness"
    python = harness / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("# pinned\n", encoding="utf-8")
    monkeypatch.setattr(ds, "REPO", release)
    monkeypatch.setattr(ds, "__file__", str(script))
    monkeypatch.setattr(ds.sys, "prefix", str(harness))
    monkeypatch.setattr(ds.sys, "executable", str(python))

    ds._verify_pinned_cell_runtime(
        expected_release_root=str(release),
        expected_harness_prefix=str(harness),
    )
    with pytest.raises(ds.DispatcherError, match="harness_prefix"):
        ds._verify_pinned_cell_runtime(
            expected_release_root=str(release),
            expected_harness_prefix=str(tmp_path / "other-harness"),
        )


def test_first_live_poll_persists_throughput_epoch_without_resetting_on_restart(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0)])
    args = _poll_args(tmp_path, run)
    args.assume_total_jobs = 448  # keep this ledger-only test free of submission effects
    ledger = ds._empty_ledger(now=10.0)
    ledger["fairness"] = {"cursor": 7, "deficits": {"sentinel": 2.5}}
    ledger["jobs"] = {"old": {"state": "inactive", "tasks": []}}

    monkeypatch.setattr(ds.time, "time", lambda: 100.0)
    first = ds._dispatch_poll(args, [run], ledger, dry_run=False)["ledger"]
    assert first["created_at"] == 10.0
    assert first["throughput_observation_started_at"] == 100.0
    assert first["fairness"] == ledger["fairness"]
    assert first["jobs"] == ledger["jobs"]

    monkeypatch.setattr(ds.time, "time", lambda: 200.0)
    restarted = ds._dispatch_poll(args, [run], first, dry_run=False)["ledger"]
    assert restarted["throughput_observation_started_at"] == 100.0
    assert restarted["fairness"] == ledger["fairness"]
    assert restarted["jobs"] == ledger["jobs"]

    dry_ledger = ds._empty_ledger(now=20.0)
    dry = ds._dispatch_poll(args, [run], dry_ledger, dry_run=True)["ledger"]
    assert "throughput_observation_started_at" not in dry


def test_durable_launcher_renders_afterany_successor_control(tmp_path):
    run = _run(tmp_path, "run", [_cell(0)])
    state = tmp_path / "state"
    rc = ld.main(
        ["--run", f"run={run.run_root}", "--results-root", str(tmp_path), "--state-dir", str(state)]
    )
    assert rc == 0
    text = (state / "global_dispatcher.sbatch").read_text()
    assert "#SBATCH --job-name=asys-global-dispatcher" in text
    assert "--successor-sbatch" in text
    assert "--cpus-per-task=1" in text


def test_successor_submission_is_afterany(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    seen = {}

    def fake_run(command, **_kwargs):
        seen["command"] = command
        return SimpleNamespace(returncode=0, stdout="456\n", stderr="")

    monkeypatch.setattr(ds.subprocess, "run", fake_run)
    assert ds._queue_successor(tmp_path / "control.sbatch") == "456"
    assert "--dependency=afterany:123" in seen["command"]
