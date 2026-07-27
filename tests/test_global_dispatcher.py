"""Global dispatcher: fair admission, crash safety, and legacy-array reconciliation."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import time
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


def _cell_submission_transport(batch_id: str) -> dict[str, str]:
    return {
        "submission_transport": ds.STDIN_EXACT_SUBMISSION_TRANSPORT,
        "submission_argv_sha256": ds._stdin_submission_argv_sha256(batch_id),
    }


def _cell_submission_command(batch_id: str) -> str:
    return " ".join(ds._stdin_submission_argv(batch_id))


class _QuestionCatalogStub:
    sidecar_sha256 = _SIDECAR_SHA256
    frozen = SimpleNamespace()

    def __init__(self):
        self.snapshot = None

    def verify_unchanged(self):
        return None

    def questions_for(self, _cell):
        return (SimpleNamespace(qid="q"),)


def test_complete_scheduler_rows_rejects_active_sacct_only_crash_window():
    snapshot = control.SchedulerSnapshot(
        jobs=(
            control.SchedulerJob(
                "654",
                "asys-dispatch-crash",
                "PENDING",
                "asys-schema5-intent:crash",
                "sbatch /state/batch-crash.sbatch",
                "sacct",
                "",
                "protected_client",
                "client_qos",
            ),
        ),
        captured_at=100.0,
        squeue_ok=True,
        sacct_ok=True,
    )
    with pytest.raises(ds.DispatcherError, match="active sacct-only"):
        ds._complete_scheduler_rows(snapshot, observation="crash-window")


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


def _qualification_authority(
    tmp_path: Path,
    *,
    run: ds.RunSpec,
    generation: int = 7,
) -> tuple[Path, dict[str, str], dict[str, str]]:
    release = (tmp_path / "sealed-release").resolve()
    harness = (tmp_path / "sealed-harness").resolve()
    python = (harness / "bin" / "python").resolve()
    dispatcher = (release / "slurm" / "dispatch_sweeps.py").resolve()
    qualification_runner = (
        release
        / "scripts"
        / "run_schema5_throughput_qualification.py"
    ).resolve()
    template = (
        release / "slurm" / "run_dispatch_batch.sbatch.tmpl"
    ).resolve()
    python.parent.mkdir(parents=True, exist_ok=True)
    template.parent.mkdir(parents=True, exist_ok=True)
    qualification_runner.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    dispatcher.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    qualification_runner.write_text(
        "#!/usr/bin/env python3\n", encoding="utf-8"
    )
    template.write_text(
        ds.ARRAY_TEMPLATE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    python.chmod(0o555)
    dispatcher.chmod(0o444)
    qualification_runner.chmod(0o444)
    template.chmod(0o444)
    runtime_environment = {
        key: "x" for key in ds.PRODUCTION_ENVIRONMENT_KEYS
    }
    protected_path = (tmp_path / "PROTECTED_CAPACITY_COMPLETE.json").resolve()
    runtime_environment.update(
        {
            "ASYS_RELEASE_GIT_COMMIT": "1" * 40,
            "ASYS_PROTECTED_CAPACITY_MARKER": str(protected_path),
            "ASYS_PROTECTED_CAPACITY_MARKER_SHA256": "2" * 64,
            "ASYS_PROTECTED_CAPACITY_MARKER_ID": "3" * 64,
            "ASYS_MODEL_CONTRACT_SHA256": "4" * 64,
            "ASYS_FLEET_CONTRACT_SHA256": "5" * 64,
            "ASYS_FLEET_CONTRACT_PATH": str(
                (tmp_path / "effective-fleet.json").resolve()
            ),
            "ASYS_RELEASE_FLEET_CONTRACT_SHA256": "6" * 64,
            "ASYS_CAPACITY_GENERATION": "2",
            "ASYS_HARNESS_ENVIRONMENT_SHA256": "7" * 64,
            "ASYS_SERVING_ENVIRONMENT_SHA256": "8" * 64,
            "ASYS_ROLLOUT_GENERATION": str(generation),
            "ASYS_IMMUTABLE_PINS_SHA256": "9" * 64,
            "ASYS_RUNTIME_ATTESTATION": str(
                (tmp_path / "attestation.json").resolve()
            ),
            "ASYS_RUNTIME_ATTESTATION_SHA256": "a" * 64,
            "ASYS_RUNTIME_INTEGRITY_LEASE": str(
                (tmp_path / "lease.json").resolve()
            ),
            "ASYS_ARTIFACT_POLICY_SHA256": "b" * 64,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    execution = {
        "release_worktree": str(release),
        "harness_prefix": str(harness),
        "hf_home": str((tmp_path / "hf").resolve()),
        "python": str(python),
        "python_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
        "dispatcher_script": str(dispatcher),
        "dispatcher_script_sha256": hashlib.sha256(
            dispatcher.read_bytes()
        ).hexdigest(),
        "qualification_runner_script": str(qualification_runner),
        "qualification_runner_script_sha256": hashlib.sha256(
            qualification_runner.read_bytes()
        ).hexdigest(),
        "batch_template": str(template),
        "batch_template_sha256": hashlib.sha256(
            template.read_bytes()
        ).hexdigest(),
    }
    identity = {
        "schema_version": (
            ds.QUALIFICATION_EXECUTION_AUTHORITY_SCHEMA_VERSION
        ),
        "protocol": ds.QUALIFICATION_EXECUTION_AUTHORITY_PROTOCOL,
        "intent_id": "c" * 64,
        "chain_id": "d" * 64,
        "run_id": run.run_id,
        "run_root": str(run.run_root.resolve()),
        "release_git_commit": "1" * 40,
        "release_tag_object": "2" * 40,
        "source_tree_sha256": "4" * 64,
        "qualification_runner_source_sha256": hashlib.sha256(
            qualification_runner.read_bytes()
        ).hexdigest(),
        "protected_capacity": {
            "path": str(protected_path),
            "sha256": "2" * 64,
            "marker_id": "3" * 64,
        },
        "readiness_generation": {
            "catalog_id": "e" * 64,
            "marker_path": str((tmp_path / "catalog.json").resolve()),
            "marker_sha256": "f" * 64,
            "inventory_sha256": "0" * 64,
            "catalog_payload_sha256": "1" * 64,
            "allowed_generation_tuple_count": 24,
            "release_fleet_contract_sha256": "6" * 64,
            "fleet_contract_sha256": "5" * 64,
            "capacity_generation": 2,
            "rollout_generation": generation,
        },
        "execution": execution,
        "runtime_environment": runtime_environment,
    }
    payload = {
        **identity,
        "authority_id": hashlib.sha256(
            ds._qualification_canonical_bytes(identity)
        ).hexdigest(),
    }
    authority_path = (tmp_path / "QUALIFICATION_EXECUTION_AUTHORITY.json").resolve()
    authority_path.write_bytes(ds._qualification_canonical_bytes(payload))
    authority_path.chmod(0o444)
    return authority_path, runtime_environment, execution


def test_global_qos_reserve_and_absolute_gate():
    # Initial production occupancy: 22 logical serving jobs plus both controller
    # chains. The inclusive reserve still permits a full 24-cell stage-18 microbatch.
    assert ds.available_cell_slots(total_jobs=24, active_cell_jobs=0) == 24
    # The held recovery DAG/sentinel jobs also count globally; they do not consume
    # client CPU/memory and still leave the first qualification microbatch intact.
    assert ds.available_cell_slots(total_jobs=65, active_cell_jobs=0) == 24
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


@pytest.mark.parametrize("run_id", sorted(ds.AUTHORITATIVE_SCHEMA5_RUN_IDS))
def test_authoritative_schema5_run_rejects_uncontrolled_admission_before_intent(
    tmp_path, monkeypatch, run_id
):
    run = _run(tmp_path, run_id, [_cell(0)])
    args = _poll_args(tmp_path, run)
    ledger = ds._empty_ledger()
    before = copy.deepcopy(ledger)
    monkeypatch.setattr(
        ds,
        "_submit_sbatch",
        lambda _path, **_kwargs: pytest.fail(
            "sbatch called without production authority"
        ),
    )

    with pytest.raises(
        ds.DispatcherError,
        match="authoritative schema-5 runs require --control-state-dir",
    ):
        ds._dispatch_poll(args, [run], ledger, dry_run=False)

    assert ledger == before
    assert not args.state_dir.exists()


def test_corrupt_or_permanent_state_fails_closed_before_sbatch(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0)])
    args = _poll_args(tmp_path, run)
    candidate = _candidate(
        run.run_id,
        0,
        pool=str(run.server_pool_root),
    )
    candidate = ds.Candidate(
        run_id=candidate.run_id,
        run_root=str(run.run_root),
        source_index=candidate.source_index,
        cell=run.manifest.cells[0],
        manifest_sha256=run.manifest.sha256,
        server_pool_arg=run.server_pool_arg,
        server_pool_root=str(run.server_pool_root),
        serving_profile=candidate.serving_profile,
        fanout_cost=candidate.fanout_cost,
        benchmark_contracts_sha256=_SIDECAR_SHA256,
    )
    monkeypatch.setattr(
        ds,
        "_scan_candidates",
        lambda *_a, **_kw: (
            [candidate],
            {"run": {"corrupt": 1, "permanent": 1}},
            {},
            ["cell failed semantic validation"],
        ),
    )
    monkeypatch.setattr(
        ds,
        "_submit_sbatch",
        lambda _path, **_kwargs: pytest.fail("sbatch called"),
    )

    outcome = ds._dispatch_poll(
        args, [run], ds._empty_ledger(), dry_run=False
    )

    assert outcome["report"]["selected"] == []
    assert outcome["report"]["qos"]["available_slots"] == 0
    assert outcome["report"]["safety_alert_keys"] == [
        "monitor:corrupt",
        "monitor:permanent",
    ]


def test_dispatcher_safety_alerts_are_persisted_critical_and_fail_closed(
    tmp_path, monkeypatch
):
    calls = []
    resolved = []
    findings = ds._dispatcher_safety_findings(
        state_counts={"run": {"validation_error": 1}},
        validation_errors=["bad metadata"],
        unmappable_jobs=["job 123 has no intent"],
    )

    def record(_state_dir, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(control, "record_alert", record)
    monkeypatch.setattr(
        control,
        "resolve_alert",
        lambda _state_dir, **kwargs: resolved.append(kwargs["dedupe_key"]),
    )
    ds._persist_dispatcher_safety_findings(
        tmp_path,
        findings=findings,
        now=100.0,
    )
    assert {call["dedupe_key"] for call in calls} == {
        "monitor:corrupt",
        ds.DISPATCHER_SCHEDULER_AMBIGUITY_ALERT,
    }
    assert all(call["severity"] == "critical" for call in calls)
    assert all(call["send_email"] is True for call in calls)
    assert resolved == []

    def cannot_persist(*_args, **_kwargs):
        raise control.ControlError("control journal unavailable")

    monkeypatch.setattr(control, "record_alert", cannot_persist)
    with pytest.raises(control.ControlError, match="journal unavailable"):
        ds._persist_dispatcher_safety_findings(
            tmp_path,
            findings=findings,
            now=101.0,
        )


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


def test_server_resource_exemption_requires_sealed_endpoint_history(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0)])
    fleet_sha256 = "f" * 64
    entry = SimpleNamespace(
        fleet_contract_sha256=fleet_sha256,
        slurm_job_id="123",
    )
    transaction_directory = tmp_path / "fleet-transactions"
    script_path = (
        transaction_directory
        / "sbatch"
        / "g000001"
        / "fleet.sbatch"
    )
    script_path.parent.mkdir(parents=True)
    script_path.write_text("#!/bin/bash\n", encoding="utf-8")
    generation_ledger_path = (
        transaction_directory / "ledgers" / "g000001.json"
    )
    generation_ledger_path.parent.mkdir(parents=True)
    generation_ledger_path.write_text("{}\n", encoding="utf-8")
    scheduler_comment = (
        "asys-s5-fleet:pool=schema5-v1;profile=8B;"
        "replica=8B-r00;generation=1;"
        f"intent={'a' * 32};fleet={fleet_sha256}"
    )
    scheduler_row = ds.QueueRow(
        "123",
        None,
        "123",
        "asys-s5-serve-8B-r00",
        "RUNNING",
        " ".join(ds.fleet_transactions.submission_argv(scheduler_comment)),
        scheduler_comment,
        "server_partition",
        "server_qos",
    )
    monkeypatch.setattr(
        ds,
        "_registered_endpoints",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        ds,
        "endpoint_history_for_entry",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(
        ds.DispatcherError,
        match="lack exact ledger/registry/history provenance",
    ):
        ds._trusted_server_scheduler_bindings(
            [run],
            expected_fleet_sha256=fleet_sha256,
            frozen_fleet=SimpleNamespace(),
            scheduler_rows=[scheduler_row],
        )

    history = SimpleNamespace(
        binding={
            "scheduler_job_name": "asys-s5-serve-8B-r00",
            "scheduler_comment": scheduler_row.comment,
            "local_script_path": str(script_path),
            "local_script_sha256": hashlib.sha256(
                script_path.read_bytes()
            ).hexdigest(),
            "intent_token": "a" * 32,
            "replica_id": "8B-r00",
            "ledger_generation": 1,
        }
    )
    monkeypatch.setattr(
        ds,
        "endpoint_history_for_entry",
        lambda *_args, **_kwargs: history,
    )
    replica = SimpleNamespace(
        serving_profile="8B",
        partition="server_partition",
        qos="server_qos",
        gpus_per_replica=1,
        gpu_type="a100",
    )
    frozen_fleet = SimpleNamespace(
        for_replica=lambda profile, index: (
            replica
            if profile == "8B" and index == 0
            else pytest.fail("unexpected frozen replica lookup")
        )
    )
    entry.replica_index = 0
    assert ds._trusted_server_scheduler_bindings(
        [run],
        expected_fleet_sha256=fleet_sha256,
        frozen_fleet=frozen_fleet,
        scheduler_rows=[scheduler_row],
    ) == {
        "123": {
            "job_name": history.binding["scheduler_job_name"],
            "comment": history.binding["scheduler_comment"],
            "sbatch_path": history.binding["local_script_path"],
            "sbatch_sha256": history.binding["local_script_sha256"],
            "intent_token": history.binding["intent_token"],
            "replica_id": history.binding["replica_id"],
            "serving_profile": "8B",
            "ledger_generation": history.binding["ledger_generation"],
            "partition": "server_partition",
            "qos": "server_qos",
            "allocated_gpus": 1,
            "gpu_type": "a100",
            "ledger_path": str(generation_ledger_path),
            "ledger_sha256": hashlib.sha256(
                generation_ledger_path.read_bytes()
            ).hexdigest(),
        }
    }
    with pytest.raises(
        ds.DispatcherError,
        match="lack exact ledger/registry/history provenance",
    ):
            ds._trusted_server_scheduler_bindings(
                [run],
                expected_fleet_sha256=fleet_sha256,
                frozen_fleet=frozen_fleet,
                scheduler_rows=[
                ds.QueueRow(
                    scheduler_row.array_job_id,
                    scheduler_row.array_task_id,
                    scheduler_row.job_id,
                    scheduler_row.job_name,
                    scheduler_row.state,
                        str(tmp_path / "different.sbatch"),
                        scheduler_row.comment,
                        scheduler_row.partition,
                        scheduler_row.qos,
                    )
            ],
        )


def test_protected_capacity_rejects_mismatched_trusted_binding_sets():
    provenance = SimpleNamespace(
        payload={
            "trusted_cell_job_ids": ["101_0"],
            "trusted_fleet_job_ids": ["201"],
        }
    )
    with pytest.raises(ds.DispatcherError, match="exact active mappings"):
        ds._require_exact_trusted_scientific_binding_sets(
            provenance,
            client_bindings={"101_0": {}},
            nonclient_bindings={"202": {}},
        )


def test_rich_provenance_projection_runs_real_headroom_validator(
    monkeypatch,
):
    clients = {
        "101_0": {
            "job_name": "asys-dispatch-batch",
            "comment": "asys-schema5-intent:batch",
            "sbatch_path": "/sealed/batch.sbatch",
            "sbatch_sha256": "a" * 64,
            "batch_manifest_path": "/sealed/batch.json",
            "batch_manifest_sha256": "b" * 64,
        }
    }
    fleet = {
        "201": {
            "job_name": "asys-s5-serve-8B-r00",
            "comment": "asys-s5-fleet:sealed",
            "ledger_path": "/sealed/g000001.json",
            "ledger_sha256": "c" * 64,
        }
    }
    monkeypatch.setattr(
        ds.scheduler_safety,
        "validate_user_partition_usage",
        lambda _usage: {
            "jobs": [],
            "job_count": 0,
            "used_cpus": 0,
            "used_memory_mib": 0,
        },
    )
    assert (
        ds.scheduler_safety.client_task_headroom(
            {},
            cell_cpus=1,
            cell_memory_mib=4 * 1024,
            cpu_limit=384,
            memory_limit_mib=384 * 4 * 1024,
            max_submit_jobs=448,
            reserve_jobs=64,
            cell_ceiling=384,
            absolute_job_ceiling=448,
            live_user_job_elements=0,
            trusted_client_jobs=(
                ds._scheduler_headroom_binding_projection(clients)
            ),
            trusted_nonclient_jobs=(
                ds._scheduler_headroom_binding_projection(fleet)
            ),
        )
        == 384
    )


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


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("run_root", "/redirected/results"),
        ("config_hash", "0" * 64),
        ("manifest_sha256", "1" * 64),
        ("benchmark_contracts_sha256", "2" * 64),
        ("model_size", "32B"),
        ("serving_profile", "32B-long"),
        ("fanout_cost", 99),
        ("server_pool_id", "foreign-pool"),
        ("server_run_id", "/foreign/pool"),
        ("server_pool_root", "/foreign/pool"),
        ("runtime_environment", {"ASYS_RELEASE_ID": "foreign-release"}),
    ],
)
def test_persisted_task_requires_exact_run_and_control_identity(
    tmp_path, field, replacement
):
    base = _run(tmp_path, "run", [_cell(0)])
    pool = tmp_path / "canonical-pool"
    run = ds.RunSpec(
        base.run_id,
        base.run_root,
        base.manifest,
        "canonical-pool",
        pool,
        base.weight,
        base.question_catalog,
        (("ASYS_RELEASE_ID", "schema5-v1.2"), ("ASYS_ROLLOUT_GENERATION", "7")),
    )
    candidate = ds.Candidate(
        run_id=run.run_id,
        run_root=str(run.run_root),
        source_index=0,
        cell=run.manifest.cells[0],
        manifest_sha256=run.manifest.sha256,
        server_pool_arg=run.server_pool_arg,
        server_pool_root=str(run.server_pool_root),
        serving_profile="8B",
        fanout_cost=1,
        benchmark_contracts_sha256=_SIDECAR_SHA256,
        runtime_environment=run.runtime_environment,
    )
    canonical = ds._task_from_candidate(candidate)
    assert ds._candidate_from_task(canonical, {run.run_id: run}) == candidate

    corrupted = dict(canonical)
    corrupted[field] = replacement
    assert ds._candidate_from_task(corrupted, {run.run_id: run}) is None


def test_pre_submit_intent_with_corrupt_identity_is_unmappable(tmp_path):
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
        _SIDECAR_SHA256,
    )
    corrupted = ds._task_from_candidate(candidate)
    corrupted["fanout_cost"] = 0
    corrupted["server_pool_root"] = str(tmp_path / "foreign-pool")
    ledger = ds._empty_ledger(now=10.0)
    ledger["intents"]["batch"] = {
        "state": "prepared",
        "created_at": 10.0,
        "tasks": [corrupted],
    }

    active, load, _, unmappable = ds._active_cells(
        [],
        ledger,
        [run],
        now=20.0,
        visibility_grace_s=30.0,
        schema5_strict=True,
    )

    assert active == {}
    assert load == {}
    assert unmappable == [
        "intent batch: task does not match a registered manifest"
    ]


def test_active_schema5_job_with_corrupt_task_identity_is_unmappable(tmp_path):
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
        _SIDECAR_SHA256,
    )
    batch_id, manifest_path, sbatch_path, batch = ds._write_batch(
        tmp_path / "state",
        [candidate],
        partition="mit_preemptable",
        time_limit="12:00:00",
        memory="4G",
        now=10.0,
    )
    ledger = ds._empty_ledger(now=10.0)
    ledger["intents"][batch_id] = {
        "state": "prepared",
        "created_at": 10.0,
        "batch_manifest": str(manifest_path),
        "batch_manifest_sha256": ds._sealed_artifact_sha256(manifest_path),
        "sbatch_path": str(sbatch_path),
        "sbatch_sha256": ds._sealed_artifact_sha256(sbatch_path),
        "tasks": json.loads(json.dumps(batch["tasks"])),
        "fairness_after": {"cursor": 0, "deficits": {}},
        "fairness_committed": False,
        "submission_transport": ds.STDIN_EXACT_SUBMISSION_TRANSPORT,
        "submission_argv_sha256": ds._stdin_submission_argv_sha256(batch_id),
    }
    ds._spooled_script_receipt(
        batch_id=batch_id,
        job_id="321",
        expected_name=f"asys-dispatch-{batch_id[-10:]}",
        expected_comment=f"asys-schema5-intent:{batch_id}",
        sbatch_path=sbatch_path,
        sbatch_sha256=ledger["intents"][batch_id]["sbatch_sha256"],
        spooled_script_reader=lambda _job_id: sbatch_path.read_bytes(),
        now=11.0,
    )
    ds._record_submission(
        ledger,
        job_id="321",
        batch_id=batch_id,
        manifest_path=manifest_path,
        sbatch_path=sbatch_path,
        batch=batch,
        now=11.0,
    )
    ledger["jobs"]["321"]["tasks"][0]["runtime_environment"] = {
        "ASYS_RELEASE_ID": "attacker-controlled"
    }
    row = ds.QueueRow(
        "321",
        0,
        "321_0",
        f"asys-dispatch-{batch_id[-10:]}",
        "RUNNING",
        " ".join(ds._stdin_submission_argv(batch_id)),
        f"asys-schema5-intent:{batch_id}",
    )

    active, load, _, unmappable = ds._active_cells(
        [row],
        ledger,
        [run],
        now=20.0,
        schema5_strict=True,
    )

    assert active == {}
    assert load == {}
    assert unmappable == [
        f"321_0 (asys-dispatch-{batch_id[-10:]}): invalid ledger task mapping"
    ]


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


@pytest.mark.parametrize(
    "payload",
    (
        (
            '{"schema_version":1,"created_at":1,"updated_at":1,'
            '"poll_number":0,"fairness":{},"validation_fairness":{},'
            '"runs":{},"jobs":{},"jobs":{},"intents":{},"cells":{}}\n'
        ),
        (
            '{"schema_version":1,"created_at":1,"updated_at":1e9999,'
            '"poll_number":0,"fairness":{},"validation_fairness":{},'
            '"runs":{},"jobs":{},"intents":{},"cells":{}}\n'
        ),
    ),
)
def test_dispatcher_ledger_load_rejects_duplicate_or_nonfinite_json(
    tmp_path, payload
):
    path = tmp_path / "ledger.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ds.DispatcherError):
        ds._load_ledger(path)


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
        partition="mit_normal",
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


def test_qualification_authority_pins_batch_task_argv_and_exact_generation(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "qualification", [_cell(0)])
    authority_path, runtime_environment, execution = (
        _qualification_authority(tmp_path, run=run, generation=7)
    )
    lease_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        ds.runtime_integrity,
        "verify_generation_lease",
        lambda **kwargs: lease_calls.append(kwargs) or {},
    )

    authority = ds.load_qualification_execution_authority(authority_path)
    assert authority.runtime_environment == runtime_environment
    assert authority.execution == execution
    assert lease_calls[-1]["generation"] == 7
    assert lease_calls[-1]["lease_path"] == Path(
        runtime_environment["ASYS_RUNTIME_INTEGRITY_LEASE"]
    )
    assert lease_calls[-1]["attestation_path"] == Path(
        runtime_environment["ASYS_RUNTIME_ATTESTATION"]
    )

    text = ds._render_batch_sbatch(
        tmp_path / "batch.json",
        n_tasks=1,
        partition="ou_bcs_normal",
        qos="normal",
        time_limit="12:00:00",
        memory="4G",
        log_dir=tmp_path,
        batch_tag="qual",
        qualification_execution=authority.execution,
    )
    assert (
        f"exec {execution['python']} -u "
        f"{execution['dispatcher_script']} run-task"
    ) in text
    assert (
        f"--expected-release-root {execution['release_worktree']}"
        in text
    )
    assert (
        f"--expected-harness-prefix {execution['harness_prefix']}"
        in text
    )
    assert "#SBATCH --partition=ou_bcs_normal" in text
    assert "#SBATCH --qos=normal" in text
    assert "mamba activate" not in text
    assert f'source "{ds.REPO}/slurm/common.sh"' not in text

    candidate = ds.Candidate(
        run.run_id,
        str(run.run_root),
        0,
        run.manifest.cells[0],
        run.manifest.sha256,
        None,
        str(run.run_root),
        "8B",
        1,
        _SIDECAR_SHA256,
        tuple(sorted(runtime_environment.items())),
    )
    task = ds._task_from_candidate(candidate)
    batch = tmp_path / "sealed-task.json"
    batch.write_text(
        json.dumps({"schema_version": 1, "tasks": [task]}),
        encoding="utf-8",
    )
    batch_sha256 = ds._seal_dispatch_artifact(batch)
    loaded_task = ds._load_batch_task(
        batch, 0, expected_sha256=batch_sha256
    )
    monkeypatch.setattr(
        ds,
        "load_frozen_benchmark_contracts",
        lambda *_args, **_kwargs: SimpleNamespace(
            sidecar_sha256=_SIDECAR_SHA256
        ),
    )
    command = ds.build_run_one_command(
        loaded_task,
        release_worktree=execution["release_worktree"],
    )
    assert command[1] == "-I"
    assert command[command.index("--release-worktree") + 1] == execution[
        "release_worktree"
    ]
    assert command[command.index("--model-contract") + 1] == str(
        Path(execution["release_worktree"])
        / "configs"
        / "model_contracts.v1.json"
    )
    assert command[command.index("--fleet-contract") + 1] == (
        runtime_environment["ASYS_FLEET_CONTRACT_PATH"]
    )
    assert command[command.index("--prompt-root") + 1] == str(
        Path(execution["release_worktree"]) / "configs" / "prompts"
    )
    ds._verify_runtime_environment_attestation(runtime_environment)
    assert lease_calls[-1]["generation"] == 7


def test_production_dispatcher_parsers_default_to_four_gb_after_pilot():
    dispatch_args = ds._build_parser().parse_args(
        ["dispatch", "--run", "placeholder"]
    )
    launcher_args = ld._parser().parse_args(["--run", "placeholder"])
    assert ds.CELL_CPUS_DEFAULT == 1
    assert ds.CELL_MEM_DEFAULT == "4G"
    assert dispatch_args.cell_mem == "4G"
    assert launcher_args.cell_mem == "4G"
    assert dispatch_args.cell_partition == "mit_normal"
    assert launcher_args.cell_partition == "mit_normal"
    assert dispatch_args.cell_time == "12:00:00"
    assert launcher_args.cell_time == "12:00:00"


def test_batch_worker_verifies_manifest_and_passes_explicit_server_pool(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0)])
    runtime_environment = {key: "x" for key in ds.PRODUCTION_ENVIRONMENT_KEYS}
    runtime_environment.update(
        {
            "ASYS_ROLLOUT_GENERATION": "1",
            "ASYS_CAPACITY_GENERATION": "1",
            "ASYS_FLEET_CONTRACT_PATH": str(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "schema5_fleet.v1.json"
            ),
            "ASYS_FLEET_CONTRACT_SHA256": "f" * 64,
            "ASYS_RELEASE_FLEET_CONTRACT_SHA256": "f" * 64,
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
    monkeypatch.setattr(
        ds,
        "_submit_sbatch",
        lambda _path, **_kwargs: pytest.fail("sbatch called"),
    )
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
        ds,
        "_submit_sbatch",
        lambda _path, **_kwargs: (_ for _ in ()).throw(
            ds.DispatcherError("qos race")
        ),
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


def test_explicit_sbatch_rejection_is_not_recorded_as_ambiguous(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0)])
    args = _poll_args(tmp_path, run)
    monkeypatch.setattr(
        ds,
        "_submit_sbatch",
        lambda _path, **_kwargs: (_ for _ in ()).throw(
            ds.SubmissionRejectedError("Slurm rejected the request")
        ),
    )
    outcome = ds._dispatch_poll(
        args, [run], ds._empty_ledger(), dry_run=False
    )
    assert "Slurm rejected" in outcome["report"]["submission_error"]
    assert {
        record["state"]
        for record in outcome["ledger"]["intents"].values()
    } == {"submission_rejected"}


def test_submit_sbatch_binds_cli_intent_and_requires_numeric_job_id(
    tmp_path, monkeypatch
):
    path = tmp_path / "batch-20260721T000000-abc123.sbatch"
    manifest_path = path.with_suffix(".json")
    manifest_path.write_text('{"schema_version":1}\n', encoding="utf-8")
    manifest_path.chmod(0o444)
    path.write_text("#!/bin/bash\n", encoding="utf-8")
    path.chmod(0o444)
    sbatch_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    calls = []

    def accepted(argv, **kwargs):
        calls.append((list(argv), kwargs))
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess(
                argv, 0, "#!/bin/bash\n", ""
            )
        return subprocess.CompletedProcess(argv, 0, "321;cluster\n", "")

    monkeypatch.setattr(ds.subprocess, "run", accepted)
    assert (
        ds._submit_sbatch(
            path,
            expected_sbatch_sha256=sbatch_sha256,
            batch_manifest_path=manifest_path,
            expected_batch_manifest_sha256=manifest_sha256,
        )
        == "321"
    )
    assert calls[0][0] == [
        "sbatch",
        "--parsable",
        "--comment=asys-schema5-intent:20260721T000000-abc123",
    ]
    assert calls[0][1]["timeout"] == 60.0
    assert calls[0][1]["input"] == "#!/bin/bash\n"
    assert calls[1][0] == [
        "scontrol",
        "write",
        "batch_script",
        "321",
        "-",
    ]

    monkeypatch.setattr(
        ds.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, "accepted\n", ""
        ),
    )
    with pytest.raises(ds.SubmissionAmbiguousError, match="invalid job id"):
        ds._submit_sbatch(
            path,
            expected_sbatch_sha256=sbatch_sha256,
            batch_manifest_path=manifest_path,
            expected_batch_manifest_sha256=manifest_sha256,
        )


@pytest.mark.parametrize(
    "mutation",
    ("tamper", "replace", "writable", "parent-symlink"),
)
def test_submit_sbatch_preflight_drift_never_invokes_slurm(
    tmp_path, monkeypatch, mutation
):
    artifact_dir = tmp_path / "sealed"
    artifact_dir.mkdir()
    path = artifact_dir / "batch-20260721T000000-abc123.sbatch"
    manifest_path = path.with_suffix(".json")
    manifest_path.write_text('{"schema_version":1}\n', encoding="utf-8")
    path.write_text("#!/bin/bash\n", encoding="utf-8")
    manifest_path.chmod(0o444)
    path.chmod(0o444)
    sbatch_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    submitted = []
    monkeypatch.setattr(
        ds.subprocess,
        "run",
        lambda *_args, **_kwargs: submitted.append(True),
    )
    if mutation == "tamper":
        path.chmod(0o644)
        path.write_text("#!/bin/bash\nexit 99\n", encoding="utf-8")
        path.chmod(0o444)
    elif mutation == "replace":
        replacement = path.with_suffix(".replacement")
        replacement.write_text("#!/bin/bash\nexit 98\n", encoding="utf-8")
        replacement.chmod(0o444)
        replacement.replace(path)
    elif mutation == "writable":
        path.chmod(0o644)
    else:
        real_dir = tmp_path / "real-sealed"
        artifact_dir.rename(real_dir)
        artifact_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(ds.SubmissionPreflightError):
        ds._submit_sbatch(
            path,
            expected_sbatch_sha256=sbatch_sha256,
            batch_manifest_path=manifest_path,
            expected_batch_manifest_sha256=manifest_sha256,
        )
    assert submitted == []


def test_submit_sbatch_uses_validated_bytes_despite_atomic_path_replacement(
    tmp_path, monkeypatch
):
    path = tmp_path / "batch-20260721T000000-atomic.sbatch"
    manifest_path = path.with_suffix(".json")
    original = b"#!/bin/bash\nexit 0\n"
    replacement_bytes = b"#!/bin/bash\nexit 99\n"
    manifest_path.write_text('{"schema_version":1}\n', encoding="utf-8")
    manifest_path.chmod(0o444)
    path.write_bytes(original)
    path.chmod(0o444)
    observed_submission = {}

    def slurm(argv, **kwargs):
        if argv[0] == "sbatch":
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(replacement_bytes)
            replacement.chmod(0o444)
            replacement.replace(path)
            observed_submission["input"] = kwargs["input"].encode("utf-8")
            return subprocess.CompletedProcess(argv, 0, "321\n", "")
        assert argv[:3] == ["scontrol", "write", "batch_script"]
        return subprocess.CompletedProcess(
            argv, 0, observed_submission["input"].decode("utf-8"), ""
        )

    monkeypatch.setattr(ds.subprocess, "run", slurm)
    with pytest.raises(
        ds.SubmissionAmbiguousError, match="changed across submission"
    ):
        ds._submit_sbatch(
            path,
            expected_sbatch_sha256=hashlib.sha256(original).hexdigest(),
            batch_manifest_path=manifest_path,
            expected_batch_manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        )
    assert observed_submission["input"] == original
    assert path.read_bytes() == replacement_bytes


def test_spooled_receipt_replays_after_slurm_purge_and_rejects_tamper(
    tmp_path,
):
    batch_id = "20260721T000000-replay"
    sbatch = tmp_path / f"batch-{batch_id}.sbatch"
    sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    sbatch.chmod(0o444)
    digest = hashlib.sha256(sbatch.read_bytes()).hexdigest()
    first = ds._spooled_script_receipt(
        batch_id=batch_id,
        job_id="321",
        expected_name=f"asys-dispatch-{batch_id[-10:]}",
        expected_comment=f"asys-schema5-intent:{batch_id}",
        sbatch_path=sbatch,
        sbatch_sha256=digest,
        spooled_script_reader=lambda _job_id: sbatch.read_bytes(),
        now=10.0,
    )
    replay = ds._spooled_script_receipt(
        batch_id=batch_id,
        job_id="321",
        expected_name=f"asys-dispatch-{batch_id[-10:]}",
        expected_comment=f"asys-schema5-intent:{batch_id}",
        sbatch_path=sbatch,
        sbatch_sha256=digest,
        spooled_script_reader=lambda _job_id: pytest.fail(
            "sealed receipt replay must not query purged Slurm history"
        ),
        now=20.0,
    )
    assert replay == first

    receipt = Path(first[0])
    replacement = receipt.with_name("attacker-receipt.json")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["job_id"] = "999"
    replacement.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replacement.chmod(0o444)
    replacement.replace(receipt)
    with pytest.raises(ds.DispatcherError, match="identity drifted"):
        ds._spooled_script_receipt(
            batch_id=batch_id,
            job_id="321",
            expected_name=f"asys-dispatch-{batch_id[-10:]}",
            expected_comment=f"asys-schema5-intent:{batch_id}",
            sbatch_path=sbatch,
            sbatch_sha256=digest,
            spooled_script_reader=lambda _job_id: pytest.fail(
                "tampered receipt must not be repaired from mutable Slurm state"
            ),
            now=30.0,
        )


def test_spooled_receipt_recovers_prelink_and_postlink_crash_states(tmp_path):
    batch_id = "20260721T000000-crash"
    sbatch = tmp_path / f"batch-{batch_id}.sbatch"
    sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    sbatch.chmod(0o444)
    digest = hashlib.sha256(sbatch.read_bytes()).hexdigest()
    receipt = sbatch.with_suffix(".spooled.json")
    payload = {
        "schema_version": 1,
        "kind": "schema5_dispatch_spooled_script_receipt",
        "batch_id": batch_id,
        "job_id": "321",
        "job_name": f"asys-dispatch-{batch_id[-10:]}",
        "scheduler_comment": f"asys-schema5-intent:{batch_id}",
        "sbatch_path": str(sbatch),
        "sbatch_sha256": digest,
        "spooled_sbatch_sha256": digest,
        "verified_at": 10.0,
    }
    prelink_transaction = receipt.parent / (
        f".{receipt.name}.publish.123.{'a' * 32}.txn"
    )
    prelink_transaction.mkdir(mode=0o700)
    prelink = prelink_transaction / "PAYLOAD"
    prelink.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prelink.chmod(0o444)
    ds._spooled_script_receipt(
        batch_id=batch_id,
        job_id="321",
        expected_name=payload["job_name"],
        expected_comment=payload["scheduler_comment"],
        sbatch_path=sbatch,
        sbatch_sha256=digest,
        spooled_script_reader=lambda _job_id: pytest.fail(
            "complete pre-link temp must be adopted without Slurm"
        ),
        now=20.0,
    )
    assert receipt.is_file() and not prelink_transaction.exists()
    assert receipt.stat().st_nlink == 1

    postlink_transaction = receipt.parent / (
        f".{receipt.name}.publish.456.{'b' * 32}.txn"
    )
    postlink_transaction.mkdir(mode=0o700)
    postlink = postlink_transaction / "PAYLOAD"
    postlink.hardlink_to(receipt)
    assert receipt.stat().st_nlink == 2
    ds._spooled_script_receipt(
        batch_id=batch_id,
        job_id="321",
        expected_name=payload["job_name"],
        expected_comment=payload["scheduler_comment"],
        sbatch_path=sbatch,
        sbatch_sha256=digest,
        spooled_script_reader=lambda _job_id: pytest.fail(
            "post-link recovery must use the durable receipt"
        ),
        now=30.0,
    )
    assert not postlink_transaction.exists()
    assert receipt.stat().st_nlink == 1
    assert not list(tmp_path.glob(f".{receipt.name}.publish.*"))


@pytest.mark.parametrize("mutation", ("conflicting-temp", "target-symlink"))
def test_spooled_receipt_fails_closed_on_publish_namespace_attack(
    tmp_path, mutation
):
    batch_id = "20260721T000000-attack"
    sbatch = tmp_path / f"batch-{batch_id}.sbatch"
    sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    sbatch.chmod(0o444)
    digest = hashlib.sha256(sbatch.read_bytes()).hexdigest()
    receipt = sbatch.with_suffix(".spooled.json")
    attacker = tmp_path / "attacker.json"
    attacker.write_text("{}\n", encoding="utf-8")
    attacker.chmod(0o444)
    if mutation == "conflicting-temp":
        attacker.rename(
            receipt.parent
            / f".{receipt.name}.publish.999.{'c' * 32}.tmp"
        )
        match = "foreign namespace"
    else:
        receipt.symlink_to(attacker)
        match = "symlink"
    with pytest.raises(ds.DispatcherError, match=match):
        ds._spooled_script_receipt(
            batch_id=batch_id,
            job_id="321",
            expected_name=f"asys-dispatch-{batch_id[-10:]}",
            expected_comment=f"asys-schema5-intent:{batch_id}",
            sbatch_path=sbatch,
            sbatch_sha256=digest,
            spooled_script_reader=lambda _job_id: pytest.fail(
                "unsafe publication state must not cross back into Slurm"
            ),
            now=10.0,
        )


@pytest.mark.parametrize(
    "crash_point",
    (
        "open",
        "partial_write",
        "post_fsync",
        "pre_fchmod",
        "post_fchmod",
        "prelink",
        "postlink",
    ),
)
def test_readonly_publication_recovers_every_private_transaction_boundary(
    tmp_path, crash_point
):
    target = tmp_path / "SEALED.bin"
    payload = b"durable-publication-payload\n"
    crashed = False

    def crash_hook(point):
        nonlocal crashed
        if point == crash_point and not crashed:
            crashed = True
            raise KeyboardInterrupt(point)

    with pytest.raises(KeyboardInterrupt, match=crash_point):
        ds._publish_readonly_bytes_once(
            target,
            payload,
            crash_hook=crash_hook,
        )
    ds._publish_readonly_bytes_once(target, payload)

    assert target.read_bytes() == payload
    assert target.stat().st_mode & 0o222 == 0
    assert target.stat().st_nlink == 1
    assert not list(tmp_path.glob(f".{target.name}.publish.*"))


def _closed_production_ledger(tmp_path):
    batch_id = "20260721T000000-closed"
    manifest = (tmp_path / f"batch-{batch_id}.json").resolve()
    sbatch = manifest.with_suffix(".sbatch")
    task = {
        "run_id": "run",
        "run_root": str((tmp_path / "run").resolve()),
        "source_index": 0,
        "cell_id": "cell",
        "config_hash": "a" * 12,
        "manifest_sha256": "b" * 64,
        "benchmark_contracts_sha256": "c" * 64,
        "model_size": "8B",
        "serving_profile": "8B",
        "fanout_cost": 1,
        "server_pool_id": None,
        "server_run_id": None,
        "server_pool_root": str((tmp_path / "pool").resolve()),
    }
    manifest.write_text("{}\n", encoding="utf-8")
    sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    manifest.chmod(0o444)
    sbatch.chmod(0o444)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    sbatch_sha = hashlib.sha256(sbatch.read_bytes()).hexdigest()
    receipt_path, receipt_sha = ds._spooled_script_receipt(
        batch_id=batch_id,
        job_id="321",
        expected_name=f"asys-dispatch-{batch_id[-10:]}",
        expected_comment=f"asys-schema5-intent:{batch_id}",
        sbatch_path=sbatch,
        sbatch_sha256=sbatch_sha,
        spooled_script_reader=lambda _job_id: sbatch.read_bytes(),
        now=2.0,
    )
    ledger = ds._empty_ledger(now=1.0)
    ledger["runs"]["run"] = {
        "run_root": task["run_root"],
        "manifest_path": str((tmp_path / "run" / "cells.json").resolve()),
        "manifest_sha256": task["manifest_sha256"],
        "manifest_cells": 1,
        "benchmark_contracts_sha256": task[
            "benchmark_contracts_sha256"
        ],
        "server_pool_arg": None,
        "server_pool_root": task["server_pool_root"],
        "weight": 1.0,
        "backlogged_polls_without_admission": 0,
    }
    ledger["intents"][batch_id] = {
        "state": "submitted",
        "created_at": 1.0,
        "submit_started_at": 2.0,
        "submitted_at": 3.0,
        "job_id": "321",
        "batch_manifest": str(manifest),
        "batch_manifest_sha256": manifest_sha,
        "sbatch_path": str(sbatch),
        "sbatch_sha256": sbatch_sha,
        "spooled_sbatch_sha256": sbatch_sha,
        "spooled_receipt_path": receipt_path,
        "spooled_receipt_sha256": receipt_sha,
        "tasks": [copy.deepcopy(task)],
        "fairness_after": {"cursor": 0, "deficits": {"run": 0.0}},
        "fairness_committed": True,
        **_cell_submission_transport(batch_id),
    }
    ledger["jobs"]["321"] = {
        "job_id": "321",
        "batch_id": batch_id,
        "batch_manifest": str(manifest),
        "batch_manifest_sha256": manifest_sha,
        "sbatch_path": str(sbatch),
        "sbatch_sha256": sbatch_sha,
        "spooled_sbatch_sha256": sbatch_sha,
        "spooled_receipt_path": receipt_path,
        "spooled_receipt_sha256": receipt_sha,
        "submitted_at": 3.0,
        "last_seen_at": 4.0,
        "state": "active",
        "task_count": 1,
        "tasks": [copy.deepcopy(task)],
        **_cell_submission_transport(batch_id),
    }
    ledger["cells"][ds._key(("run", "cell"))] = {
        "run_id": "run",
        "cell_id": "cell",
        "source_index": 0,
        "model_size": "8B",
        "serving_profile": "8B",
        "fanout_cost": 1,
        "completion_state": "active",
        "next_eligible_at": None,
        "last_checked_at": 4.0,
        "eligible_for_retry": False,
    }
    ledger["updated_at"] = 4.0
    return ledger, batch_id


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("legacy-schema", "legacy"),
        ("task-identity", "coordinate identity"),
        ("job-timestamp", "closed accepted transaction"),
        ("scheduler-states", "closed accepted transaction"),
        ("fairness-commit", "sealed spool proof"),
        ("run-manifest", "manifest record"),
        ("cell-retry", "retry record"),
    ),
)
def test_production_ledger_validator_rejects_each_admission_field_class(
    tmp_path, mutation, match
):
    ledger, batch_id = _closed_production_ledger(tmp_path)
    if mutation == "legacy-schema":
        ledger["schema_version"] = ds.LEGACY_LEDGER_SCHEMA_VERSION
    elif mutation == "task-identity":
        ledger["intents"][batch_id]["tasks"][0]["config_hash"] = "bad"
        ledger["jobs"]["321"]["tasks"][0]["config_hash"] = "bad"
    elif mutation == "job-timestamp":
        ledger["jobs"]["321"]["last_seen_at"] = "yesterday"
    elif mutation == "scheduler-states":
        ledger["jobs"]["321"]["scheduler_states"] = [None]
    elif mutation == "fairness-commit":
        ledger["intents"][batch_id]["fairness_committed"] = False
    elif mutation == "run-manifest":
        ledger["runs"]["run"]["manifest_sha256"] = "bad"
    else:
        cell = ledger["cells"][ds._key(("run", "cell"))]
        cell["submission_attempts"] = -1
    with pytest.raises(ds.DispatcherError, match=match):
        ds.validate_production_ledger_structure(ledger)


def test_preflight_failure_fences_coordinates_as_integrity_blocked(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0)])
    args = _poll_args(tmp_path, run)
    monkeypatch.setattr(
        ds,
        "_submit_sbatch",
        lambda _path, **_kwargs: (_ for _ in ()).throw(
            ds.SubmissionPreflightError("sealed artifact changed")
        ),
    )

    with pytest.raises(
        ds.SubmissionPreflightError, match="sealed artifact changed"
    ):
        ds._dispatch_poll(
            args,
            [run],
            ds._empty_ledger(),
            dry_run=False,
        )

    persisted = ds._load_ledger(args.ledger_path)
    [intent] = persisted["intents"].values()
    assert intent["state"] == "integrity_blocked"
    assert intent["integrity_alert_key"] == (
        ds.DISPATCHER_SUBMISSION_INTEGRITY_ALERT
    )
    assert intent["tasks"]


def test_integrity_blocked_rejects_any_post_boundary_or_job_evidence(
    tmp_path,
):
    ledger, batch_id = _closed_production_ledger(tmp_path)
    ledger["intents"][batch_id].update(
        {
            "state": "integrity_blocked",
            "fairness_committed": False,
            "error": "invalid accepted-to-blocked mutation",
            "integrity_blocked_at": 10.0,
            "integrity_alert_key": (
                ds.DISPATCHER_SUBMISSION_INTEGRITY_ALERT
            ),
        }
    )
    with pytest.raises(ds.DispatcherError, match="pre-boundary"):
        ds.validate_production_ledger_structure(ledger)


@pytest.mark.parametrize(
    ("state", "valid_fields", "contradictory_field", "contradictory_value"),
    (
        ("prepared", {}, "job_id", "321"),
        ("submitting", {"submit_started_at": 2.0}, "job_id", "321"),
        (
            "submitted",
            {
                "submit_started_at": 2.0,
                "submitted_at": 3.0,
                "job_id": "321",
            },
            "error",
            "stale rejection",
        ),
        (
            "reconciled",
            {
                "submit_started_at": 2.0,
                "reconciled_at": 3.0,
                "job_id": "321",
            },
            "integrity_alert_key",
            ds.DISPATCHER_SUBMISSION_INTEGRITY_ALERT,
        ),
        (
            "not_accepted",
            {"reconciled_at": 3.0, "error": "proven absent"},
            "spooled_sbatch_sha256",
            "d" * 64,
        ),
        (
            "submission_rejected",
            {
                "submit_started_at": 2.0,
                "error": "explicit rejection",
                "last_submit_error_at": 3.0,
            },
            "submitted_at",
            3.0,
        ),
        (
            "integrity_blocked",
            {
                "error": "sealed preflight failed",
                "integrity_blocked_at": 3.0,
                "integrity_alert_key": (
                    ds.DISPATCHER_SUBMISSION_INTEGRITY_ALERT
                ),
            },
            "submit_started_at",
            2.0,
        ),
        (
            "integrity_retired",
            {
                "error": "sealed preflight failed",
                "integrity_blocked_at": 3.0,
                "integrity_alert_key": (
                    ds.DISPATCHER_SUBMISSION_INTEGRITY_ALERT
                ),
                "integrity_retired_at": 4.0,
                "retirement_id": "e" * 64,
                "retirement_receipt_path": "/sealed/retirement.json",
                "retirement_receipt_sha256": "f" * 64,
                "retirement_semantic_report_path": "/sealed/semantic.json",
                "retirement_semantic_report_sha256": "1" * 64,
                "retirement_scheduler_absence_sha256": "2" * 64,
                "retirement_operator_note_sha256": "3" * 64,
                "retirement_identity_changes": ["0:run:cell:ASYS_RELEASE_ID"],
            },
            "unexpected_extension_field",
            True,
        ),
    ),
)
def test_production_intent_states_are_closed_and_reject_impossible_artifacts(
    tmp_path,
    state,
    valid_fields,
    contradictory_field,
    contradictory_value,
):
    ledger, batch_id = _closed_production_ledger(tmp_path)
    original = ledger["intents"][batch_id]
    common = {
        key: copy.deepcopy(original[key])
        for key in (
            "created_at",
            "batch_manifest",
            "batch_manifest_sha256",
            "sbatch_path",
            "sbatch_sha256",
            "submission_transport",
            "submission_argv_sha256",
            "tasks",
            "fairness_after",
        )
    }
    accepted = state in {"submitted", "reconciled"}
    intent = {
        **common,
        "state": state,
        "fairness_committed": accepted,
        **copy.deepcopy(valid_fields),
    }
    if accepted:
        intent.update(
            {
                "spooled_sbatch_sha256": original[
                    "spooled_sbatch_sha256"
                ],
                "spooled_receipt_path": original["spooled_receipt_path"],
                "spooled_receipt_sha256": original[
                    "spooled_receipt_sha256"
                ],
            }
        )
        if state == "reconciled":
            ledger["jobs"]["321"]["reconciled_at"] = 3.0
    else:
        ledger["jobs"] = {}
    ledger["intents"][batch_id] = intent
    ds.validate_production_ledger_structure(ledger)

    intent[contradictory_field] = contradictory_value
    with pytest.raises(ds.DispatcherError):
        ds.validate_production_ledger_structure(ledger)


def test_submission_integrity_alert_is_critical_and_human_latched(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        control,
        "record_alert",
        lambda _state_dir, **kwargs: calls.append(kwargs),
    )
    ds._persist_dispatcher_submission_integrity_hold(
        tmp_path,
        message="immutable admission artifact drift",
        now=100.0,
    )
    assert calls == [
        {
            "kind": "dispatcher-submission-integrity",
            "severity": "critical",
            "message": "immutable admission artifact drift",
            "dedupe_key": ds.DISPATCHER_SUBMISSION_INTEGRITY_ALERT,
            "send_email": True,
            "now": 100.0,
        }
    ]
    assert (
        ds.DISPATCHER_SUBMISSION_INTEGRITY_ALERT
        in control.SCIENTIFIC_INTEGRITY_ALERT_KEYS
    )


def _integrity_retirement_fixture(tmp_path, monkeypatch):
    ledger, batch_id = _closed_production_ledger(tmp_path)
    intent = ledger["intents"][batch_id]
    environment = {
        key: f"old-{key.lower()}" for key in ds.PRODUCTION_ENVIRONMENT_KEYS
    }
    environment.update(
        {
            "ASYS_ROLLOUT_GENERATION": "1",
            "ASYS_CAPACITY_GENERATION": "1",
            "ASYS_FLEET_CONTRACT_PATH": str(
                (tmp_path / "fleet.v1.json").resolve()
            ),
            "ASYS_FLEET_CONTRACT_SHA256": "d" * 64,
            "ASYS_RELEASE_FLEET_CONTRACT_SHA256": "e" * 64,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    intent["tasks"][0]["runtime_environment"] = environment
    intent.update(
        {
            "state": "integrity_blocked",
            "fairness_committed": False,
            "error": "sealed artifact changed before sbatch",
            "integrity_blocked_at": 10.0,
            "integrity_alert_key": (
                ds.DISPATCHER_SUBMISSION_INTEGRITY_ALERT
            ),
        }
    )
    for field in (
        "job_id",
        "submit_started_at",
        "submitted_at",
        "reconciled_at",
        "spooled_sbatch_sha256",
        "spooled_receipt_path",
        "spooled_receipt_sha256",
    ):
        intent.pop(field, None)
    ledger["jobs"] = {}
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    ds._atomic_write_json(state_dir / "ledger.json", ledger)
    semantic_path = (state_dir / "monitoring" / "semantic.json").resolve()
    semantic_path.parent.mkdir()
    semantic_path.write_text(
        json.dumps({"semantic": {"scan_successful": True}}) + "\n",
        encoding="utf-8",
    )
    semantic_path.chmod(0o444)
    semantic_sha256 = hashlib.sha256(
        semantic_path.read_bytes()
    ).hexdigest()
    semantic = {
        "path": str(semantic_path),
        "sha256": semantic_sha256,
        "captured_timestamp": 20.0,
        "committed_timestamp": 21.0,
        "cadence": "semantic",
        "control_immutable_sha256": "3" * 64,
        "rollout_generation": 2,
        "fleet_generation": "capacity-g000001-current",
        "report_sha256": semantic_sha256,
    }
    acknowledgement = {
        "timestamp": 22.0,
        "at": "1970-01-01T00:00:22Z",
        "operator_note": "release fixed preflight",
        "transition_sha256": "2" * 64,
    }
    monkeypatch.setattr(
        ds,
        "_validated_integrity_retirement_semantic_evidence",
        lambda *_args, **_kwargs: copy.deepcopy(semantic),
    )
    monkeypatch.setattr(
        ds,
        "_ensure_integrity_retirement_acknowledgement",
        lambda *_args, **_kwargs: copy.deepcopy(acknowledgement),
    )
    monkeypatch.setattr(
        ds,
        "_matching_integrity_acknowledgement",
        lambda *_args, **_kwargs: copy.deepcopy(acknowledgement),
    )
    monkeypatch.setattr(
        control,
        "admission_boundary_lock",
        lambda _state_dir: ds.nullcontext(),
    )
    monkeypatch.setattr(
        control,
        "load_control",
        lambda *_args, **_kwargs: {
            "created_timestamp": 5.0,
            "immutable_sha256": "3" * 64,
            "rollout_generation": 2,
            "resume_intent": {
                "rollout_generation": 2,
                "created_timestamp": 12.0,
            },
            "capacity": {
                "current_generation": 1,
                "current_contract": None,
            },
            "throughput_epochs": [
                {
                    "epoch": 1,
                    "rollout_generation": 2,
                    "fleet_generation": "capacity-g000001-current",
                    "started_timestamp": 18.0,
                    "closed_at": None,
                }
            ],
            "admission_ramp": {
                "last_observation": {
                    "path": str(semantic_path),
                    "sha256": semantic_sha256,
                    "cadence": "semantic",
                    "captured_timestamp": 20.0,
                    "timestamp": 21.0,
                }
            },
        },
    )
    monkeypatch.setattr(
        ds,
        "_current_integrity_runtime_identities",
        lambda _state, blocked: [
            {
                **copy.deepcopy(identity),
                "runtime_environment": {
                    **copy.deepcopy(identity["runtime_environment"]),
                    "ASYS_RELEASE_ID": "fixed-release",
                    "ASYS_ROLLOUT_GENERATION": "2",
                },
            }
            for identity in blocked
        ],
    )
    scheduler = SimpleNamespace(
        jobs=(),
        captured_at=30.0,
        squeue_ok=True,
        sacct_ok=True,
        errors=(),
    )
    return {
        "state_dir": state_dir,
        "control_state_dir": state_dir,
        "batch_id": batch_id,
        "semantic_path": semantic_path,
        "semantic_sha256": semantic_sha256,
        "note": acknowledgement["operator_note"],
        "scheduler": scheduler,
    }


def _retire_integrity_fixture(fixture, **kwargs):
    return ds.retire_integrity_blocked_intent(
        fixture["state_dir"],
        control_state_dir=fixture["control_state_dir"],
        batch_id=fixture["batch_id"],
        semantic_evidence_path=fixture["semantic_path"],
        semantic_evidence_sha256=fixture["semantic_sha256"],
        operator_note=fixture["note"],
        scheduler_reader=lambda: fixture["scheduler"],
        now=40.0,
        **kwargs,
    )


def test_integrity_retirement_rejects_old_scan_after_control_generation_change(
    tmp_path,
):
    semantic_path = (tmp_path / "monitoring" / "semantic.json").resolve()
    semantic_path.parent.mkdir()
    semantic_path.write_text("{}\n", encoding="utf-8")
    semantic_sha256 = hashlib.sha256(
        semantic_path.read_bytes()
    ).hexdigest()
    semantic = {
        "path": str(semantic_path),
        "sha256": semantic_sha256,
        "captured_timestamp": 20.0,
        "committed_timestamp": 21.0,
        "cadence": "semantic",
        "control_immutable_sha256": "a" * 64,
        "rollout_generation": 1,
        "fleet_generation": "capacity-g000001-old",
        "report_sha256": semantic_sha256,
    }
    current = {
        "created_timestamp": 5.0,
        "immutable_sha256": "b" * 64,
        "rollout_generation": 2,
        "resume_intent": {
            "rollout_generation": 2,
            "created_timestamp": 12.0,
        },
        "capacity": {
            "current_generation": 2,
            "current_contract": {
                "activated_timestamp": 18.0,
            },
        },
        "throughput_epochs": [
            {
                "epoch": 2,
                "rollout_generation": 2,
                "fleet_generation": "capacity-g000002-current",
                "started_timestamp": 19.0,
                "closed_at": None,
            }
        ],
        "admission_ramp": {
            "last_observation": {
                "path": str(semantic_path),
                "sha256": semantic_sha256,
                "cadence": "semantic",
                "captured_timestamp": 20.0,
                "timestamp": 21.0,
            }
        },
    }

    with pytest.raises(ds.DispatcherError, match="current immutable control"):
        ds._bind_integrity_retirement_semantic_to_current_control(
            current, semantic
        )

    semantic.update(
        {
            "control_immutable_sha256": "b" * 64,
            "rollout_generation": 2,
            "fleet_generation": "capacity-g000001-old",
        }
    )
    with pytest.raises(ds.DispatcherError, match="current fleet generation"):
        ds._bind_integrity_retirement_semantic_to_current_control(
            current, semantic
        )


def test_integrity_retirement_seals_preimages_and_releases_only_its_coordinates(
    tmp_path, monkeypatch
):
    fixture = _integrity_retirement_fixture(tmp_path, monkeypatch)
    receipt = _retire_integrity_fixture(fixture)
    persisted = ds.load_production_ledger(
        fixture["state_dir"] / "ledger.json"
    )
    intent = persisted["intents"][fixture["batch_id"]]

    assert intent["state"] == "integrity_retired"
    assert intent["fairness_committed"] is False
    assert len(receipt["archived_preimages"]) == 3
    assert {
        row["logical_name"] for row in receipt["archived_preimages"]
    } == {"blocked_intent", "batch_manifest", "sbatch"}
    assert all(
        Path(row["archive_path"]).stat().st_mode & 0o222 == 0
        for row in receipt["archived_preimages"]
    )
    assert (
        Path(intent["retirement_receipt_path"]).stat().st_mode & 0o222
        == 0
    )
    active, load, _legacy, unmappable = ds._active_cells(
        [],
        persisted,
        [],
        now=50.0,
        schema5_strict=True,
    )
    assert active == {}
    assert load == {}
    assert unmappable == []
    assert _retire_integrity_fixture(fixture) == receipt


@pytest.mark.parametrize(
    "crash_point",
    (
        "after_intent",
        "after_preimages",
        "after_scheduler_absence",
        "after_receipt",
    ),
)
def test_integrity_retirement_recovers_every_marker_boundary(
    tmp_path, monkeypatch, crash_point
):
    fixture = _integrity_retirement_fixture(tmp_path, monkeypatch)
    crashed = False

    def crash_hook(point):
        nonlocal crashed
        if point == crash_point and not crashed:
            crashed = True
            raise KeyboardInterrupt(point)

    with pytest.raises(KeyboardInterrupt, match=crash_point):
        _retire_integrity_fixture(fixture, crash_hook=crash_hook)
    receipt = _retire_integrity_fixture(fixture)
    assert receipt["batch_id"] == fixture["batch_id"]
    persisted = ds.load_production_ledger(
        fixture["state_dir"] / "ledger.json"
    )
    assert (
        persisted["intents"][fixture["batch_id"]]["state"]
        == "integrity_retired"
    )


def test_integrity_retirement_rejects_scheduler_ambiguity_before_ledger_change(
    tmp_path, monkeypatch
):
    fixture = _integrity_retirement_fixture(tmp_path, monkeypatch)
    fixture["scheduler"].jobs = (
        SimpleNamespace(
            job_id="456",
            source="sacct",
            active=False,
            job_name=f"asys-dispatch-{fixture['batch_id'][-10:]}",
            comment=f"asys-schema5-intent:{fixture['batch_id']}",
            command=ds._cell_submission_command(fixture["batch_id"])
            if hasattr(ds, "_cell_submission_command")
            else " ".join(ds._stdin_submission_argv(fixture["batch_id"])),
        ),
    )
    with pytest.raises(ds.DispatcherError, match="cannot prove.*absent"):
        _retire_integrity_fixture(fixture)
    persisted = ds.load_production_ledger(
        fixture["state_dir"] / "ledger.json"
    )
    assert (
        persisted["intents"][fixture["batch_id"]]["state"]
        == "integrity_blocked"
    )


def test_retired_integrity_archive_tamper_relatches_global_hold(
    tmp_path, monkeypatch
):
    fixture = _integrity_retirement_fixture(tmp_path, monkeypatch)
    receipt = _retire_integrity_fixture(fixture)
    archive = Path(receipt["archived_preimages"][0]["archive_path"])
    archive.chmod(0o644)
    archive.write_bytes(b"tampered\n")
    archive.chmod(0o444)
    calls = []
    monkeypatch.setattr(
        ds,
        "_persist_dispatcher_submission_integrity_hold",
        lambda *_args, **kwargs: calls.append(kwargs),
    )
    ledger = ds.load_production_ledger(
        fixture["state_dir"] / "ledger.json"
    )
    with pytest.raises(ds.SubmissionPreflightError, match="preimage"):
        ds._recover_dispatcher_submission_integrity_hold(
            fixture["state_dir"],
            ledger=ledger,
            now=60.0,
        )
    assert calls and "admission remains fenced" in calls[0]["message"]


def _stable_scheduler_snapshot(
    *job_ids,
    partition="ou_bcs_normal",
    qos="normal",
    job_name="unrelated-target-job",
    comment="unrelated",
):
    return SimpleNamespace(
        captured_at=time.time(),
        jobs=tuple(
            SimpleNamespace(
                job_id=job_id,
                job_name=job_name,
                state="RUNNING",
                comment=comment,
                command="",
                source="squeue",
                active=True,
                partition=partition,
                qos=qos,
            )
            for job_id in job_ids
        ),
        squeue_ok=True,
        sacct_ok=True,
        errors=(),
    )


def test_stable_occupancy_retries_churn_then_accepts_logical_elements(
    monkeypatch,
):
    snapshots = iter(
        (
            _stable_scheduler_snapshot(),
            _stable_scheduler_snapshot("500_0"),
            _stable_scheduler_snapshot("500_0"),
            _stable_scheduler_snapshot("500_0"),
        )
    )
    usages = iter(
        (
            {
                "partition": "ou_bcs_normal",
                "jobs": [
                    {
                        "job_id": "500_0",
                        "job_name": "unrelated-target-job",
                        "comment": "unrelated",
                        "qos": "normal",
                    }
                ],
            },
            {
                "partition": "ou_bcs_normal",
                "jobs": [
                    {
                        "job_id": "500_0",
                        "job_name": "unrelated-target-job",
                        "comment": "unrelated",
                        "qos": "normal",
                    }
                ],
            },
        )
    )
    monkeypatch.setattr(
        ds.scheduler_safety,
        "validate_user_partition_usage",
        lambda usage: usage,
    )

    stable = ds._capture_stable_admission_occupancy(
        user="tester",
        partition="ou_bcs_normal",
        scheduler_reader=lambda: next(snapshots),
        usage_reader=lambda: next(usages),
    )

    assert stable.attempts == 2
    assert stable.live_job_ids == ("500_0",)
    assert [row.job_id for row in stable.rows] == ["500_0"]


@pytest.mark.parametrize(
    "drifted_snapshot",
    (
        _stable_scheduler_snapshot(
            "500_0", partition="other_partition"
        ),
        _stable_scheduler_snapshot("500_0", qos="other_qos"),
    ),
    ids=("partition", "qos"),
)
def test_stable_occupancy_rejects_same_id_placement_drift(
    monkeypatch, drifted_snapshot
):
    snapshots = iter(
        (
            _stable_scheduler_snapshot("500_0"),
            drifted_snapshot,
            _stable_scheduler_snapshot("500_0"),
            drifted_snapshot,
        )
    )
    usage = {
        "partition": "ou_bcs_normal",
        "jobs": [
            {
                "job_id": "500_0",
                "job_name": "unrelated-target-job",
                "comment": "unrelated",
                "qos": "normal",
            }
        ],
    }
    monkeypatch.setattr(
        ds.scheduler_safety,
        "validate_user_partition_usage",
        lambda value: value,
    )

    with pytest.raises(ds.DispatcherError, match=r"changed=\['500_0'\]"):
        ds._capture_stable_admission_occupancy(
            user="tester",
            partition="ou_bcs_normal",
            scheduler_reader=lambda: next(snapshots),
            usage_reader=lambda: usage,
        )


def _trusted_client_target_occupancy(
    *,
    scheduler_partition="ou_bcs_normal",
    scheduler_qos="normal",
    usage_job=True,
    usage_qos="normal",
    usage_cpus=1,
    usage_memory_mib=4_096,
):
    name = "asys-dispatch-abc123def0"
    comment = "asys-schema5-intent:20260726T120000-abc123def0"
    raw_jobs = []
    if usage_job:
        raw_jobs.append(
            {
                "job_id": 777,
                "job_state": ["RUNNING"],
                "partition": "ou_bcs_normal",
                "name": name,
                "comment": comment,
                "qos": usage_qos,
                "cpus": {
                    "set": True,
                    "infinite": False,
                    "number": usage_cpus,
                },
                "node_count": {
                    "set": True,
                    "infinite": False,
                    "number": 1,
                },
                "memory_per_cpu": {
                    "set": False,
                    "infinite": False,
                    "number": 0,
                },
                "memory_per_node": {
                    "set": True,
                    "infinite": False,
                    "number": usage_memory_mib,
                },
                "array_task_id": {
                    "set": True,
                    "infinite": False,
                    "number": 0,
                },
            }
        )
    payload = {"jobs": raw_jobs, "errors": [], "warnings": []}

    def runner(argv, *, timeout):
        assert timeout == 30.0
        return subprocess.CompletedProcess(
            argv, 0, json.dumps(payload), ""
        )

    usage = ds.scheduler_safety.capture_user_partition_usage(
        user="tester",
        partition="ou_bcs_normal",
        runner=runner,
        captured_timestamp=100.0,
    )
    row = ds.QueueRow(
        array_job_id="777",
        array_task_id=0,
        job_id="777_0",
        job_name=name,
        state="RUNNING",
        command="/sealed/batch.sbatch",
        comment=comment,
        partition=scheduler_partition,
        qos=scheduler_qos,
    )
    occupancy = ds.StableAdmissionOccupancy(
        scheduler_snapshot=_stable_scheduler_snapshot("777_0"),
        rows=(row,),
        usage=usage,
        usage_summary=(
            ds.scheduler_safety.validate_user_partition_usage(usage)
        ),
        live_job_ids=("777_0",),
        attempts=1,
    )
    binding = {
        "777_0": {
            "job_name": name,
            "comment": comment,
        }
    }
    return occupancy, binding


def test_trusted_client_target_placement_uses_real_usage_validator():
    occupancy, binding = _trusted_client_target_occupancy()
    ds._require_trusted_client_target_placement(
        occupancy,
        trusted_client_bindings=binding,
        expected_partition="ou_bcs_normal",
        expected_qos="normal",
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {"scheduler_partition": "other_partition"},
        {"scheduler_qos": "other_qos"},
        {"usage_job": False},
        {"usage_qos": "other_qos"},
        {"usage_cpus": 2},
        {"usage_memory_mib": 8_192},
    ),
    ids=(
        "scheduler-partition",
        "scheduler-qos",
        "missing-target-usage",
        "usage-qos",
        "usage-cpus",
        "usage-memory",
    ),
)
def test_trusted_client_target_placement_rejects_drift(overrides):
    occupancy, binding = _trusted_client_target_occupancy(**overrides)
    with pytest.raises(ds.DispatcherError, match="drifted"):
        ds._require_trusted_client_target_placement(
            occupancy,
            trusted_client_bindings=binding,
            expected_partition="ou_bcs_normal",
            expected_qos="normal",
        )


@pytest.mark.parametrize(
    ("snapshots", "usage"),
    (
        (
            (
                _stable_scheduler_snapshot(),
                _stable_scheduler_snapshot("600_0"),
                _stable_scheduler_snapshot(),
                _stable_scheduler_snapshot("600_0"),
            ),
            {
                "partition": "ou_bcs_normal",
                "jobs": [
                    {
                        "job_id": "600_0",
                        "job_name": "unrelated-target-job",
                        "comment": "unrelated",
                        "qos": "normal",
                    }
                ],
            },
        ),
        (
            (
                _stable_scheduler_snapshot("700_0"),
                _stable_scheduler_snapshot(),
                _stable_scheduler_snapshot("700_0"),
                _stable_scheduler_snapshot(),
            ),
            {"partition": "ou_bcs_normal", "jobs": []},
        ),
        (
            (
                _stable_scheduler_snapshot("750_0"),
                _stable_scheduler_snapshot("750_0"),
                _stable_scheduler_snapshot("750_0"),
                _stable_scheduler_snapshot("750_0"),
            ),
            {"partition": "ou_bcs_normal", "jobs": []},
        ),
        (
            (
                _stable_scheduler_snapshot(),
                _stable_scheduler_snapshot(),
                _stable_scheduler_snapshot(),
                _stable_scheduler_snapshot(),
            ),
            {
                "partition": "ou_bcs_normal",
                "jobs": [
                    {
                        "job_id": "800_0",
                        "job_name": "unrelated-target-job",
                        "comment": "unrelated",
                        "qos": "normal",
                    }
                ],
            },
        ),
    ),
    ids=(
        "target-appears",
        "target-disappears",
        "target-missing-usage",
        "target-transient",
    ),
)
def test_stable_occupancy_fails_closed_on_persistent_between_read_churn(
    monkeypatch,
    snapshots,
    usage,
):
    observations = iter(snapshots)
    monkeypatch.setattr(
        ds.scheduler_safety,
        "validate_user_partition_usage",
        lambda value: value,
    )

    with pytest.raises(ds.DispatcherError, match="occupancy changed"):
        ds._capture_stable_admission_occupancy(
            user="tester",
            partition="ou_bcs_normal",
            scheduler_reader=lambda: next(observations),
            usage_reader=lambda: usage,
        )


def test_post_intent_occupancy_excludes_current_batch_exactly_once():
    ledger = ds._empty_ledger()
    ledger["jobs"]["100"] = {
        "state": "visibility_grace",
        "task_count": 2,
        "tasks": [{}, {}],
    }
    ledger["intents"] = {
        "other": {
            "state": "prepared",
            "created_at": 990.0,
            "tasks": [{}, {}, {}],
        },
        "current": {
            "state": "prepared",
            "created_at": 995.0,
            "tasks": [{}, {}, {}, {}],
        },
        "old": {
            "state": "prepared",
            "created_at": 1.0,
            "tasks": [{}] * 100,
        },
    }

    assert ds._invisible_reservation_count(ledger, now=1_000.0) == 9
    # At the post-intent boundary, the current four tasks are compared with the
    # returned headroom.  Excluding that intent here charges them once, not twice.
    assert ds._invisible_reservation_count(
        ledger,
        now=1_000.0,
        exclude_intent_ids=frozenset({"current"}),
    ) == 5


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

    def submit(path, **_kwargs):
        assert persisted_states[-1] == ["submitting"]
        batch_id = Path(path).stem.removeprefix("batch-")
        ds._spooled_script_receipt(
            batch_id=batch_id,
            job_id="321",
            expected_name=f"asys-dispatch-{batch_id[-10:]}",
            expected_comment=f"asys-schema5-intent:{batch_id}",
            sbatch_path=Path(path),
            sbatch_sha256=_kwargs["expected_sbatch_sha256"],
            spooled_script_reader=lambda _job_id: Path(path).read_bytes(),
            now=11.0,
        )
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


def test_lost_reply_intent_fences_next_admission_until_late_adoption(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "run", [_cell(0), _cell(1)])
    args = _poll_args(tmp_path, run)
    args.max_batch = 1
    calls = 0

    def lose_reply(_path, **_kwargs):
        nonlocal calls
        calls += 1
        raise ds.SubmissionAmbiguousError("accepted reply lost")

    monkeypatch.setattr(ds, "_submit_sbatch", lose_reply)
    first = ds._dispatch_poll(
        args, [run], ds._empty_ledger(), dry_run=False
    )
    first_ledger = first["ledger"]
    [batch_id] = first_ledger["intents"]
    first_intent = first_ledger["intents"][batch_id]
    fairness_before = copy.deepcopy(first_ledger["fairness"])
    assert first_intent["state"] == "submitting"
    assert calls == 1

    second = ds._dispatch_poll(
        args, [run], first_ledger, dry_run=False
    )
    assert calls == 1
    assert second["report"]["selected"] == []
    assert second["ledger"]["fairness"] == fairness_before
    assert list(second["ledger"]["intents"]) == [batch_id]

    sbatch = Path(first_intent["sbatch_path"])
    scheduler_job = SimpleNamespace(
        job_id="321_0",
        job_name=f"asys-dispatch-{batch_id[-10:]}",
        state="RUNNING",
        comment=f"asys-schema5-intent:{batch_id}",
        command=_cell_submission_command(batch_id),
        source="squeue",
        active=True,
    )
    warnings, errors = ds._reconcile_schema5_intents(
        second["ledger"],
        scheduler_snapshot=SimpleNamespace(jobs=(scheduler_job,)),
        now=time.time(),
        spooled_script_reader=lambda _job_id: sbatch.read_bytes(),
    )
    assert not errors and warnings
    assert (
        second["ledger"]["fairness"]
        == first_intent["fairness_after"]
    )
    assert second["ledger"]["intents"][batch_id]["fairness_committed"] is True


def test_production_recaptures_stable_occupancy_after_durable_intent(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "production", [_cell(0)])
    args = _poll_args(tmp_path, run)
    args.control_state_dir = tmp_path / "control"
    args.control_state_dir.mkdir()
    args.cell_partition = "ou_bcs_normal"
    args.cell_qos = "normal"
    args.cell_mem = "4G"
    args.cell_time = "12:00:00"
    args.assume_total_jobs = None
    args.assume_cell_jobs = None
    fleet_sha256 = "f" * 64
    protected_ref = {
        "path": str(tmp_path / "PROTECTED_CAPACITY_COMPLETE.json"),
        "sha256": "a" * 64,
        "marker_id": "b" * 64,
    }
    admission = {
        "qos_limit": 448,
        "reserve": 64,
        "max_batch": 24,
        "cell_cpus": 1,
        "cell_memory": "4G",
        "effective_cell_time": "12:00:00",
        "current_ceiling": 24,
        "configured_ceiling": 24,
        "client_capacity": {
            "partition": "ou_bcs_normal",
            "qos": "normal",
            "cpu_limit": 384,
            "memory_limit_mib": 384 * 4_096,
            "max_submit_jobs": 448,
            "reserve_jobs": 64,
            "capacity_generation": 1,
            "authorization_sha256": "c" * 64,
        },
        "protected_capacity": protected_ref,
    }
    control_payload = {
        "immutable": {
            "git_commit": "1" * 40,
            "source_tree_sha256": "2" * 64,
            "model_contract_path": str(tmp_path / "models.json"),
            "model_contract_sha256": "3" * 64,
            "server_pool_root": str(run.server_pool_root),
        },
        "rollout_generation": 1,
    }
    runtime_environment = {
        key: "x" for key in ds.PRODUCTION_ENVIRONMENT_KEYS
    }
    runtime_environment.update(
        {
            "ASYS_FLEET_CONTRACT_SHA256": fleet_sha256,
            "ASYS_RELEASE_FLEET_CONTRACT_SHA256": fleet_sha256,
            "ASYS_CAPACITY_GENERATION": "1",
            "ASYS_ROLLOUT_GENERATION": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    execution = {
        "batch_template": str(ds.ARRAY_TEMPLATE),
        "hf_home": str(tmp_path / "hf"),
        "release_worktree": str(Path(__file__).resolve().parents[1]),
        "python": sys.executable,
        "dispatcher_script": str(
            Path(__file__).resolve().parents[1]
            / "slurm"
            / "dispatch_sweeps.py"
        ),
        "harness_prefix": str(tmp_path / "harness"),
    }
    capacity_contract = SimpleNamespace(
        path=Path(protected_ref["path"]),
        sha256=protected_ref["sha256"],
        marker_id=protected_ref["marker_id"],
    )
    fleet = SimpleNamespace(
        sha256=fleet_sha256,
        verify_pool_root=lambda _root: None,
    )
    monkeypatch.setattr(
        control,
        "admission_contract_from_state",
        lambda _state_dir: copy.deepcopy(admission),
    )
    monkeypatch.setattr(
        control,
        "load_control",
        lambda *_args, **_kwargs: control_payload,
    )
    monkeypatch.setattr(
        control,
        "effective_fleet_contract_binding",
        lambda *_args, **_kwargs: {
            "path": str(tmp_path / "fleet.json"),
            "sha256": fleet_sha256,
        },
    )
    monkeypatch.setattr(
        control,
        "load_effective_protected_capacity_contract",
        lambda *_args, **_kwargs: capacity_contract,
    )
    monkeypatch.setattr(
        control,
        "production_environment_from_state",
        lambda *_args, **_kwargs: dict(runtime_environment),
    )
    monkeypatch.setattr(
        control,
        "production_cell_execution_from_state",
        lambda *_args, **_kwargs: execution,
    )
    monkeypatch.setattr(
        control,
        "validate_production_batch_sbatch",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        control,
        "admission_boundary_lock",
        lambda _state_dir: ds.nullcontext(),
    )
    monkeypatch.setattr(
        control,
        "query_scheduler",
        lambda **_kwargs: _stable_scheduler_snapshot(),
    )
    monkeypatch.setattr(ds, "load_model_contracts", lambda *_a, **_k: object())
    monkeypatch.setattr(
        ds,
        "load_fleet_contract",
        lambda *_args, **_kwargs: fleet,
    )
    monkeypatch.setattr(
        ds,
        "discover_capacity",
        lambda *_args, **_kwargs: ds.CapacitySnapshot(
            {(str(run.server_pool_root), "8B"): 1},
            {(str(run.server_pool_root), "8B"): "generation"},
        ),
    )
    monkeypatch.setattr(
        ds,
        "_trusted_server_scheduler_bindings",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        ds,
        "_persist_dispatcher_safety_findings",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ds.protected_capacity,
        "load_contract",
        lambda *_args, **_kwargs: capacity_contract,
    )
    monkeypatch.setattr(
        ds.protected_capacity,
        "authorize_client",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        ds.protected_capacity,
        "verify_live_placements",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        ds.protected_capacity,
        "capture_live_client_capacity",
        lambda *_args, **_kwargs: {"evidence_id": "9" * 64},
    )
    monkeypatch.setattr(
        ds.protected_capacity,
        "validate_live_client_capacity_evidence",
        lambda *_args, **_kwargs: {},
    )
    usage_calls = 0

    def capture_usage(**_kwargs):
        nonlocal usage_calls
        usage_calls += 1
        return {
            "capture": usage_calls,
            "partition": "ou_bcs_normal",
        }

    monkeypatch.setattr(
        ds.scheduler_safety,
        "capture_user_partition_usage",
        capture_usage,
    )
    monkeypatch.setattr(
        ds.scheduler_safety,
        "validate_user_partition_usage",
        lambda _usage: {
            "partition": "ou_bcs_normal",
            "jobs": [],
            "job_count": 0,
            "used_cpus": 0,
            "used_memory_mib": 0,
        },
    )
    monkeypatch.setattr(
        ds.scheduler_safety,
        "client_task_headroom",
        lambda usage, **_kwargs: 24 if usage["capture"] <= 2 else 0,
    )
    monkeypatch.setattr(
        ds,
        "_submit_sbatch",
        lambda _path, **_kwargs: pytest.fail(
            "sbatch crossed stale production occupancy"
        ),
    )
    original_reconcile = (
        control.reconcile_trusted_scientific_job_provenance
    )
    provenance_cuts = []

    def reconcile_with_durable_cut(state_dir, **kwargs):
        raw = (Path(state_dir) / "ledger.json").read_bytes()
        durable = json.loads(raw)
        provenance_cuts.append(
            {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "updated_at": durable["updated_at"],
                "intents": copy.deepcopy(durable["intents"]),
            }
        )
        return original_reconcile(state_dir, **kwargs)

    monkeypatch.setattr(
        control,
        "reconcile_trusted_scientific_job_provenance",
        reconcile_with_durable_cut,
    )

    with pytest.raises(ds.DispatcherError, match="headroom shrank"):
        ds._dispatch_poll(args, [run], ds._empty_ledger(), dry_run=False)

    assert usage_calls == 3
    persisted = ds._load_ledger(args.ledger_path)
    intent = next(iter(persisted["intents"].values()))
    assert intent["state"] == "prepared"
    assert "headroom shrank" in intent["error"]
    submitting_cuts = [
        cut
        for cut in provenance_cuts
        if cut["intents"]
        and next(iter(cut["intents"].values()))["state"] == "submitting"
    ]
    assert len(submitting_cuts) == 1
    submitted_intent = next(iter(submitting_cuts[0]["intents"].values()))
    assert (
        submitting_cuts[0]["updated_at"]
        >= submitted_intent["submit_started_at"]
    )
    assert len({cut["sha256"] for cut in provenance_cuts}) >= 2


def test_protected_qualification_recaptures_unrelated_usage_before_sbatch(
    tmp_path, monkeypatch
):
    run = _run(tmp_path, "qualification", [_cell(0)])
    args = _poll_args(tmp_path, run)
    args.control_state_dir = None
    args.cell_partition = "ou_bcs_normal"
    args.cell_qos = "normal"
    args.cell_mem = "4G"
    args.cell_time = "12:00:00"
    args.assume_total_jobs = None
    args.assume_cell_jobs = None
    authority_path, runtime_environment, _execution = _qualification_authority(
        tmp_path, run=run
    )
    payload = json.loads(authority_path.read_text(encoding="utf-8"))
    protected_ref = payload["protected_capacity"]
    args.protected_capacity_marker = protected_ref["path"]
    args.protected_capacity_marker_sha256 = protected_ref["sha256"]
    args.protected_capacity_marker_id = protected_ref["marker_id"]
    args.protected_capacity_release_git_commit = payload[
        "release_git_commit"
    ]
    args.qualification_execution_authority = authority_path
    contract = SimpleNamespace(
        path=Path(protected_ref["path"]),
        sha256=protected_ref["sha256"],
        marker_id=protected_ref["marker_id"],
        release_git_commit=payload["release_git_commit"],
    )
    placement = SimpleNamespace(
        partition="ou_bcs_normal",
        qos="normal",
        capacity={
            "slots": 384,
            "cpus": 384,
            "memory_mib": 384 * 4_096,
            "reserve_jobs": 64,
            "submit_headroom": 448,
        },
    )
    monkeypatch.setattr(
        ds.runtime_integrity,
        "verify_generation_lease",
        lambda **_kwargs: {},
    )
    qualification_fleet = SimpleNamespace(
        sha256=runtime_environment["ASYS_FLEET_CONTRACT_SHA256"],
        verify_pool_root=lambda _root: None,
    )
    monkeypatch.setattr(ds, "load_model_contracts", lambda *_a, **_k: object())
    monkeypatch.setattr(
        ds,
        "load_fleet_contract",
        lambda *_args, **_kwargs: qualification_fleet,
    )
    monkeypatch.setattr(
        ds.protected_capacity,
        "load_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        ds.protected_capacity,
        "authorize_client",
        lambda *_args, **_kwargs: placement,
    )
    monkeypatch.setattr(
        ds.protected_capacity,
        "verify_live_placements",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        ds.protected_capacity,
        "capture_live_client_capacity",
        lambda *_args, **_kwargs: {"evidence_id": "9" * 64},
    )
    monkeypatch.setattr(
        ds.protected_capacity,
        "validate_live_client_capacity_evidence",
        lambda *_args, **_kwargs: {
            "evidence_id": "9" * 64,
            "partition": "ou_bcs_normal",
            "qos": "normal",
            "cpu_limit": 384,
            "memory_limit_mib": 384 * 4_096,
            "max_submit_jobs": 448,
        },
    )
    usage_calls = 0

    def capture_usage(**_kwargs):
        nonlocal usage_calls
        usage_calls += 1
        return {
            "capture": usage_calls,
            "partition": "ou_bcs_normal",
        }

    monkeypatch.setattr(
        ds.scheduler_safety,
        "capture_user_partition_usage",
        capture_usage,
    )
    monkeypatch.setattr(
        ds.scheduler_safety,
        "validate_user_partition_usage",
        lambda _usage: {
            "partition": "ou_bcs_normal",
            "jobs": [],
            "job_count": 0,
            "used_cpus": 0,
            "used_memory_mib": 0,
        },
    )
    monkeypatch.setattr(
        ds.scheduler_safety,
        "client_task_headroom",
        lambda usage, **_kwargs: 24 if usage["capture"] <= 2 else 0,
    )
    monkeypatch.setattr(
        control,
        "query_scheduler",
        lambda **_kwargs: SimpleNamespace(
            captured_at=time.time(),
            jobs=(),
            squeue_ok=True,
            sacct_ok=True,
            errors=(),
        ),
    )
    monkeypatch.setattr(
        ds,
        "_submit_sbatch",
        lambda _path, **_kwargs: pytest.fail(
            "sbatch crossed a stale protected-partition usage snapshot"
        ),
    )

    with pytest.raises(ds.DispatcherError, match="headroom shrank"):
        ds._dispatch_poll(
            args, [run], ds._empty_ledger(), dry_run=False
        )
    assert usage_calls == 3
    persisted = ds._load_ledger(args.ledger_path)
    intent = next(iter(persisted["intents"].values()))
    assert intent["state"] == "prepared"
    assert "headroom shrank" in intent["error"]
    assert (
        intent["qualification_execution_authority"]
        == persisted["qualification_execution_authority"]
    )
    assert intent["tasks"][0]["runtime_environment"] == runtime_environment


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
        **_cell_submission_transport(batch_id),
    }

    def job(job_id, *, source="squeue"):
        return SimpleNamespace(
            job_id=job_id,
            job_name=f"asys-dispatch-{batch_id[-10:]}",
            state="RUNNING",
            comment=f"asys-schema5-intent:{batch_id}",
            command=_cell_submission_command(batch_id),
            source=source,
            active=True,
        )

    # Real `sacct --array` may expose a consolidated parent in addition to the
    # logical task identities from `squeue -r`.  All three rows must commit one
    # base-array intent, never three jobs or an ambiguity.
    snapshot = SimpleNamespace(
        jobs=(job("321", source="sacct"), job("321_0"), job("321_1"))
    )
    warnings, errors = ds._reconcile_schema5_intents(
        ledger,
        scheduler_snapshot=snapshot,
        now=20.0,
        spooled_script_reader=lambda _job_id: sbatch.read_bytes(),
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
        spooled_script_reader=lambda _job_id: sbatch.read_bytes(),
    )
    assert "ambiguously maps" in duplicate_errors[0]


def test_intent_reconciliation_allows_blank_sacct_comment_with_exact_spool(
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
        **_cell_submission_transport(batch_id),
    }
    accounting = SimpleNamespace(
        job_id="321",
        job_name=f"asys-dispatch-{batch_id[-10:]}",
        state="COMPLETED",
        comment="",
        command=_cell_submission_command(batch_id),
        source="sacct",
        active=False,
    )
    warnings, errors = ds._reconcile_schema5_intents(
        ledger,
        scheduler_snapshot=SimpleNamespace(jobs=(accounting,)),
        now=20.0,
        spooled_script_reader=lambda _job_id: sbatch.read_bytes(),
    )
    assert not errors
    assert warnings
    assert ledger["intents"][batch_id]["job_id"] == "321"

    wrong_identity = SimpleNamespace(
        **{
            **accounting.__dict__,
            "job_id": "322",
            "comment": "asys-schema5-intent:foreign",
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
        second,
        scheduler_snapshot=SimpleNamespace(jobs=(wrong_identity,)),
        now=20.0,
        spooled_script_reader=lambda _job_id: sbatch.read_bytes(),
    )
    assert errors and "provenance drift" in errors[0]


@pytest.mark.parametrize("mutation", ("parent-symlink", "file-symlink"))
def test_intent_reconciliation_rejects_lexical_sbatch_symlink_replacement(
    tmp_path, mutation
):
    batch_id = "20260721T000000-symlink"
    artifact_dir = tmp_path / "sealed"
    artifact_dir.mkdir()
    sbatch = artifact_dir / f"batch-{batch_id}.sbatch"
    manifest = sbatch.with_suffix(".json")
    manifest.write_text("{}\n", encoding="utf-8")
    sbatch.write_text("#!/bin/bash\n", encoding="utf-8")
    manifest_sha256 = ds._seal_dispatch_artifact(manifest)
    sbatch_sha256 = ds._seal_dispatch_artifact(sbatch)
    ledger = ds._empty_ledger()
    ledger["intents"][batch_id] = {
        "state": "submitting",
        "created_at": 10.0,
        "submit_started_at": 11.0,
        "batch_manifest": str(manifest),
        "batch_manifest_sha256": manifest_sha256,
        "sbatch_path": str(sbatch),
        "sbatch_sha256": sbatch_sha256,
        "tasks": [{"run_id": "run", "cell_id": "cell"}],
        "fairness_after": {"cursor": 1, "deficits": {"run": 0.0}},
        "fairness_committed": False,
        **_cell_submission_transport(batch_id),
    }
    if mutation == "parent-symlink":
        real_dir = tmp_path / "real-sealed"
        artifact_dir.rename(real_dir)
        artifact_dir.symlink_to(real_dir, target_is_directory=True)
    else:
        real_sbatch = sbatch.with_suffix(".real")
        sbatch.rename(real_sbatch)
        sbatch.symlink_to(real_sbatch)
    job = SimpleNamespace(
        job_id="321_0",
        job_name=f"asys-dispatch-{batch_id[-10:]}",
        state="RUNNING",
        comment=f"asys-schema5-intent:{batch_id}",
        command=_cell_submission_command(batch_id),
        source="squeue",
        active=True,
    )

    _warnings, errors = ds._reconcile_schema5_intents(
        ledger,
        scheduler_snapshot=SimpleNamespace(jobs=(job,)),
        now=20.0,
    )
    assert errors and "symlink" in errors[0]
    assert ledger["intents"][batch_id]["fairness_committed"] is False
    assert ledger["jobs"] == {}


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
        **_cell_submission_transport(batch_id),
    }

    warnings, errors = ds._reconcile_schema5_intents(
        ledger, scheduler_snapshot=SimpleNamespace(jobs=()), now=1_000.0
    )
    assert not warnings and not errors
    assert ledger["intents"][batch_id]["state"] == "submitting"

    warnings, errors = ds._reconcile_schema5_intents(
        ledger,
        scheduler_snapshot=SimpleNamespace(
            jobs=(), accounting_start_timestamp=990.000001
        ),
        now=1_301.0,
    )
    assert not warnings
    assert errors and "integrity-ambiguous" in errors[0]
    assert ledger["intents"][batch_id]["state"] == "submitting"

    warnings, errors = ds._reconcile_schema5_intents(
        ledger,
        scheduler_snapshot=SimpleNamespace(
            jobs=(), accounting_start_timestamp=990.0
        ),
        now=1_301.0,
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
        **_cell_submission_transport(batch_id),
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
            "ASYS_CAPACITY_GENERATION": "1",
            "ASYS_FLEET_CONTRACT_PATH": "/sealed/schema5_fleet.v1.json",
            "ASYS_FLEET_CONTRACT_SHA256": "f" * 64,
            "ASYS_RELEASE_FLEET_CONTRACT_SHA256": "f" * 64,
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
            "ASYS_CAPACITY_GENERATION": "1",
            "ASYS_FLEET_CONTRACT_PATH": "/sealed/schema5_fleet.v1.json",
            "ASYS_FLEET_CONTRACT_SHA256": "f" * 64,
            "ASYS_RELEASE_FLEET_CONTRACT_SHA256": "f" * 64,
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
    script = tmp_path / "control.sbatch"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    script.chmod(0o444)
    seen = {"commands": [], "released": False}

    def fake_run(command, **kwargs):
        seen["commands"].append(list(command))
        if command[0] == "sbatch":
            seen["input"] = kwargs["input"]
            return SimpleNamespace(returncode=0, stdout="456\n", stderr="")
        if command[:3] == ["scontrol", "write", "batch_script"]:
            return SimpleNamespace(
                returncode=0,
                stdout=script.read_text(encoding="utf-8"),
                stderr="",
            )
        if command[:4] == ["scontrol", "show", "job", "-o"]:
            state = "RUNNING" if seen["released"] else "PENDING"
            reason = "None" if seen["released"] else "JobHeldUser"
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"JobId=456 JobState={state} Reason={reason} "
                    "Dependency=afterany:123\n"
                ),
                stderr="",
            )
        if command[:2] == ["scontrol", "release"]:
            seen["released"] = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(ds.subprocess, "run", fake_run)
    assert ds._queue_successor(script, state_dir=tmp_path / "state") == "456"
    sbatch = seen["commands"][0]
    assert "--dependency=afterany:123" in sbatch
    assert "--hold" in sbatch
    assert sbatch[-1].startswith("--comment=asys-schema5-successor:123:")
    assert seen["input"] == "#!/bin/bash\n"


class _SuccessorSlurm:
    def __init__(self, script: Path, *, lost_reply=False, spool_mismatch=False):
        self.script = script
        self.lost_reply = lost_reply
        self.spool_mismatch = spool_mismatch
        self.sbatch_calls = 0
        self.release_calls = 0
        self.released = False

    def __call__(self, command, **kwargs):
        if command[0] == "sbatch":
            self.sbatch_calls += 1
            assert kwargs["input"] == self.script.read_text(encoding="utf-8")
            if self.lost_reply:
                self.lost_reply = False
                raise KeyboardInterrupt("lost successor reply")
            return SimpleNamespace(returncode=0, stdout="456\n", stderr="")
        if command[:3] == ["scontrol", "write", "batch_script"]:
            text = self.script.read_text(encoding="utf-8")
            if self.spool_mismatch:
                text += "# mismatch\n"
            return SimpleNamespace(returncode=0, stdout=text, stderr="")
        if command[:4] == ["scontrol", "show", "job", "-o"]:
            state = "RUNNING" if self.released else "PENDING"
            reason = "None" if self.released else "JobHeldUser"
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"JobId=456 JobState={state} Reason={reason} "
                    "Dependency=afterany:123\n"
                ),
                stderr="",
            )
        if command[:2] == ["scontrol", "release"]:
            assert command == ["scontrol", "release", "456"]
            self.release_calls += 1
            self.released = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)


def _successor_snapshot(state_dir: Path, *, present: bool):
    intent = json.loads(
        (
            state_dir
            / "successor-transactions"
            / "after-123"
            / "INTENT.json"
        ).read_text(encoding="utf-8")
    )
    jobs = ()
    if present:
        jobs = (
            SimpleNamespace(
                job_id="456",
                comment=intent["scheduler_comment"],
                command=" ".join(intent["submission_argv"]),
                state="RUNNING",
                active=True,
            ),
        )
    return SimpleNamespace(
        jobs=jobs,
        squeue_ok=True,
        sacct_ok=True,
        errors=(),
        accounting_start_timestamp=0.0,
    )


@pytest.mark.parametrize(
    "crash_point",
    (
        "after_submitting",
        "after_acceptance",
        "before_release",
        "after_release_command",
        "after_release",
    ),
)
def test_successor_transaction_recovers_every_external_boundary(
    tmp_path, monkeypatch, crash_point
):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    script = tmp_path / "control.sbatch"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    script.chmod(0o444)
    state_dir = tmp_path / "state"
    slurm = _SuccessorSlurm(script)
    crashed = False

    def crash_hook(point):
        nonlocal crashed
        if point == crash_point and not crashed:
            crashed = True
            raise KeyboardInterrupt(point)

    with pytest.raises(KeyboardInterrupt, match=crash_point):
        ds._queue_successor(
            script,
            state_dir=state_dir,
            runner=slurm,
            scheduler_reader=lambda: _successor_snapshot(
                state_dir, present=False
            ),
            now=100.0,
            crash_hook=crash_hook,
        )
    present = crash_point != "after_submitting"
    assert (
        ds._queue_successor(
            script,
            state_dir=state_dir,
            runner=slurm,
            scheduler_reader=lambda: _successor_snapshot(
                state_dir, present=present
            ),
            now=401.0,
        )
        == "456"
    )
    assert slurm.sbatch_calls == 1
    intent = json.loads(
        (
            state_dir
            / "successor-transactions"
            / "after-123"
            / "INTENT.json"
        ).read_text(encoding="utf-8")
    )
    assert intent["state"] == "committed"
    assert intent["job_id"] == "456"
    assert slurm.release_calls == 1


def test_successor_lost_reply_adopts_without_duplicate_submission(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    script = tmp_path / "control.sbatch"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    script.chmod(0o444)
    state_dir = tmp_path / "state"
    slurm = _SuccessorSlurm(script, lost_reply=True)
    with pytest.raises(KeyboardInterrupt, match="lost successor reply"):
        ds._queue_successor(
            script,
            state_dir=state_dir,
            runner=slurm,
            now=100.0,
        )
    assert (
        ds._queue_successor(
            script,
            state_dir=state_dir,
            runner=slurm,
            scheduler_reader=lambda: _successor_snapshot(
                state_dir, present=True
            ),
            now=101.0,
        )
        == "456"
    )
    assert slurm.sbatch_calls == 1


def test_successor_release_replay_rejects_tampered_result_schema(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    script = tmp_path / "control.sbatch"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    script.chmod(0o444)
    state_dir = tmp_path / "state"
    slurm = _SuccessorSlurm(script)
    with pytest.raises(KeyboardInterrupt, match="after_release"):
        ds._queue_successor(
            script,
            state_dir=state_dir,
            runner=slurm,
            now=100.0,
            crash_hook=lambda point: (
                (_ for _ in ()).throw(KeyboardInterrupt(point))
                if point == "after_release"
                else None
            ),
        )
    result_path = (
        state_dir
        / "successor-transactions"
        / "after-123"
        / "RELEASE_RESULT.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["foreign"] = True
    result_path.chmod(0o644)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result_path.chmod(0o444)

    with pytest.raises(ds.DispatcherError, match="release result drifted"):
        ds._queue_successor(
            script,
            state_dir=state_dir,
            runner=slurm,
            scheduler_reader=lambda: _successor_snapshot(
                state_dir, present=True
            ),
            now=101.0,
        )
    assert slurm.release_calls == 1


def test_successor_spool_mismatch_latches_without_resubmission(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    script = tmp_path / "control.sbatch"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    script.chmod(0o444)
    state_dir = tmp_path / "state"
    slurm = _SuccessorSlurm(script, spool_mismatch=True)
    with pytest.raises(ds.DispatcherError, match="spooled successor"):
        ds._queue_successor(
            script,
            state_dir=state_dir,
            runner=slurm,
            now=100.0,
        )
    with pytest.raises(ds.DispatcherError, match="integrity remains blocked"):
        ds._queue_successor(
            script,
            state_dir=state_dir,
            runner=slurm,
            now=101.0,
        )
    assert slurm.sbatch_calls == 1


def test_successor_rejects_script_replacement_after_durable_boundary(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    script = tmp_path / "control.sbatch"
    script.write_text("#!/bin/bash\n", encoding="utf-8")
    script.chmod(0o444)
    state_dir = tmp_path / "state"
    slurm = _SuccessorSlurm(script)
    with pytest.raises(KeyboardInterrupt, match="after_submitting"):
        ds._queue_successor(
            script,
            state_dir=state_dir,
            runner=slurm,
            now=100.0,
            crash_hook=lambda point: (
                (_ for _ in ()).throw(KeyboardInterrupt(point))
                if point == "after_submitting"
                else None
            ),
        )
    replacement = tmp_path / "replacement.sbatch"
    replacement.write_text("#!/bin/bash\n# changed\n", encoding="utf-8")
    replacement.chmod(0o444)
    replacement.replace(script)
    with pytest.raises(ds.DispatcherError, match="identity drifted"):
        ds._queue_successor(
            script,
            state_dir=state_dir,
            runner=slurm,
            now=401.0,
        )
    assert slurm.sbatch_calls == 0
