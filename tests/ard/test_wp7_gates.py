"""WP7 gates: the hour-2 launch gate and the hour-12/18/24 decision-coverage assertions.

Spec §16.3 (required measured fields, the 20% reserve, "a sum of GPU-hours does not prove the
critical path fits"), §16.4 (tier frozen before any new test outcome); plan §7 (hour-18
train/val >= 99% and test >= 40%; hour-24 test >= 90% **by protocol x checkpoint x cohort**)
and §10 ("Runtime gates").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import gates as G
from agents_scaling.ard.types import (
    ArdCellKind,
    ArdCellSpec,
    DecisionRef,
    EpisodeRecord,
    decision_id_for,
    episode_id_for,
)
from agents_scaling.experiment import io


@dataclass
class FakeEntry:
    host: str
    port: int = 8000
    model_size: str = K.CHECKPOINT_SMALL


def _lister(by_profile):
    def lister(run_root, profile, **_kwargs):
        return by_profile.get(profile, [])
    return lister


def _measured(**overrides):
    base = {
        "small_tps_per_replica": 220.0,
        "large_tps_per_replica": 130.0,
        "root_parse_validity": 0.97,
        "http_error_rate": 0.002,
        "grader_items_per_s": 4.0,
        "required_grader_items_per_s": 2.0,
        "replay_sequences_per_s": 1.5,
        "replay_hours": 8.0,
        "bf16_confirmed": True,
        "projected_critical_path_hours": 12.0,
        "projected_total_hours": 24.0,
        "throughput_by_call_type": {
            call_type: {"tokens_per_s": 200.0, "latency_mean_s": 12.0, "latency_p95_s": 30.0,
                        "prompt_tokens_mean": 900.0, "generated_tokens_mean": 1100.0,
                        "n_requests": 40}
            for call_type in G.CALL_TYPES
        },
    }
    base.update(overrides)
    return base


REQUIRED = {K.PROFILE_SMALL: {"pi_manoli": 3}, K.PROFILE_LARGE: {"pi_manoli": 2}}
PARTITIONS = {"node3904": "pi_manoli", "node4001": "ou_bcs_low"}


def _census_lister(small_hosts=("node3904",) * 3, large_hosts=("node3904",) * 2):
    return _lister({K.PROFILE_SMALL: [FakeEntry(h) for h in small_hosts],
                    K.PROFILE_LARGE: [FakeEntry(h) for h in large_hosts]})


# --------------------------------------------------------------------------- launch gate


def test_launch_gate_passes_on_healthy_measurements(tmp_path):
    report = G.launch_gate(tmp_path, measured=_measured(), required_replicas=REQUIRED,
                           partition_of=PARTITIONS, lister=_census_lister(),
                           run_budget=False)
    assert report["passed"] is True
    assert report["failed"] == []
    assert report["replica_census"][K.PROFILE_SMALL] == {"pi_manoli": 3}
    assert (tmp_path / "gates" / "launch_gate.json").exists()
    on_disk = json.loads((tmp_path / "gates" / "launch_gate.json").read_text())
    assert on_disk["gate"] == "launch" and on_disk["hour"] == 2


def test_replicas_are_asserted_per_profile_and_partition(tmp_path):
    """Three live 8B replicas do not satisfy a requirement of three *on pi_manoli*."""
    lister = _census_lister(small_hosts=("node3904", "node4001", "node4001"))
    report = G.launch_gate(tmp_path, measured=_measured(), required_replicas=REQUIRED,
                           partition_of=PARTITIONS, lister=lister, run_budget=False)
    assert report["passed"] is False
    assert f"replicas:{K.PROFILE_SMALL}:pi_manoli" in report["failed"]
    assert report["replica_census"][K.PROFILE_SMALL] == {"pi_manoli": 1, "ou_bcs_low": 2}


def test_an_unmappable_host_is_visible_not_silently_credited(tmp_path):
    lister = _census_lister(small_hosts=("node9999", "node9999", "node9999"))
    report = G.launch_gate(tmp_path, measured=_measured(), required_replicas=REQUIRED,
                           partition_of=PARTITIONS, lister=lister, run_budget=False)
    assert report["replica_census"][K.PROFILE_SMALL] == {"unknown": 3}
    assert report["passed"] is False


def test_parse_validity_floor(tmp_path):
    report = G.launch_gate(tmp_path, measured=_measured(root_parse_validity=0.90),
                           required_replicas=REQUIRED, partition_of=PARTITIONS,
                           lister=_census_lister(), run_budget=False)
    assert "root_parse_validity" in report["failed"]
    check = next(c for c in report["checks"] if c["name"] == "root_parse_validity")
    assert check["threshold"] == G.PARSE_VALIDITY_MIN == 0.95


def test_http_error_rate_above_one_percent_fails(tmp_path):
    report = G.launch_gate(tmp_path, measured=_measured(http_error_rate=0.02),
                           required_replicas=REQUIRED, partition_of=PARTITIONS,
                           lister=_census_lister(), run_budget=False)
    assert "http_error_rate" in report["failed"]


def test_grader_and_replay_rates_are_gated(tmp_path):
    slow = G.launch_gate(tmp_path, measured=_measured(grader_items_per_s=0.5),
                         required_replicas=REQUIRED, partition_of=PARTITIONS,
                         lister=_census_lister(), run_budget=False)
    assert "grader_items_per_s" in slow["failed"]
    late = G.launch_gate(tmp_path, measured=_measured(replay_hours=44.0),
                         required_replicas=REQUIRED, partition_of=PARTITIONS,
                         lister=_census_lister(), run_budget=False)
    assert "replay_sequences_per_s" in late["failed"]


def test_throughput_must_be_stratified_by_call_type(tmp_path):
    table = _measured()["throughput_by_call_type"]
    del table["HUB_WORKER"]
    table["RECEIVER"] = dict(table["RECEIVER"], latency_p95_s=None)
    report = G.launch_gate(tmp_path, measured=_measured(throughput_by_call_type=table),
                           required_replicas=REQUIRED, partition_of=PARTITIONS,
                           lister=_census_lister(), run_budget=False)
    assert "throughput:HUB_WORKER" in report["failed"]
    assert "throughput:RECEIVER" in report["failed"]
    assert "throughput:ROOT" not in report["failed"]


def test_projection_applies_the_twenty_percent_reserve(tmp_path):
    report = G.launch_gate(tmp_path,
                           measured=_measured(projected_critical_path_hours=14.0),
                           required_replicas=REQUIRED, partition_of=PARTITIONS,
                           lister=_census_lister(), run_budget=False)
    check = next(c for c in report["checks"] if c["name"] == "projected_critical_path_h")
    assert check["value"] == pytest.approx(14.0 * 1.2)
    assert check["passed"] is False           # 16.8 h > the 16 h deadline
    assert G.RESERVE_FRACTION == 0.20


def test_tier_is_reduced_unless_the_reserved_projection_fits_by_hour_thirty(tmp_path):
    full = G.launch_gate(tmp_path, measured=_measured(projected_total_hours=24.0),
                         required_replicas=REQUIRED, partition_of=PARTITIONS,
                         lister=_census_lister(), run_budget=False)
    assert full["tier"] == "full_core"                       # 28.8 h <= 30 h
    reduced = G.launch_gate(tmp_path, measured=_measured(projected_total_hours=27.0),
                            required_replicas=REQUIRED, partition_of=PARTITIONS,
                            lister=_census_lister(), run_budget=False)
    assert reduced["passed"] is True and reduced["tier"] == "reduced_core"  # 32.4 h > 30 h
    assert "HUB5 dropped" in reduced["tier_note"]


def test_a_missing_measurement_is_an_error_not_a_default(tmp_path):
    measured = _measured()
    del measured["grader_items_per_s"]
    with pytest.raises(G.GateError) as excinfo:
        G.launch_gate(tmp_path, measured=measured, required_replicas=REQUIRED,
                      lister=_census_lister(), run_budget=False)
    assert "grader_items_per_s" in str(excinfo.value)


# --------------------------------------------------------------------------- budget_check


def test_budget_check_runs_the_bundle_script_with_measured_rates():
    out = G.run_budget_check(small_tps_per_replica=220.0, large_tps_per_replica=100.0,
                             replay_hours=8.0)
    assert out["small_lane"]["measured_effective_tps"] == 220.0
    assert out["small_lane"]["output_lane_gate_passed"] is True
    assert out["large_lane"]["output_lane_gate_passed"] is False   # 100 < the required ~105
    assert out["ledger"]["totals"]["small_logical_decisions"] > 0
    assert out["replay_hours"] == 8.0


def test_budget_check_refuses_a_nonpositive_measured_rate():
    with pytest.raises(G.GateError):
        G.run_budget_check(small_tps_per_replica=0.0, large_tps_per_replica=100.0)


def test_launch_gate_feeds_budget_check_and_scores_both_lanes(tmp_path):
    report = G.launch_gate(tmp_path, measured=_measured(large_tps_per_replica=100.0),
                           required_replicas=REQUIRED, partition_of=PARTITIONS,
                           lister=_census_lister(), run_budget=True)
    names = {c["name"] for c in report["checks"]}
    assert {"budget_check:small_lane", "budget_check:large_lane"} <= names
    assert "budget_check:large_lane" in report["failed"]
    assert report["budget_check"]["ledger"]["totals"]["large_logical_decisions"] > 0


# --------------------------------------------------------------------------- coverage gate


def _cell(protocol, *, items, kind=ArdCellKind.EPISODE, checkpoint=K.CHECKPOINT_SMALL,
          cohort=K.Cohort.NATURAL, split="test", shard=0, options=None):
    return ArdCellSpec(
        cell_id=f"{kind.value}.{cohort.value}.{getattr(protocol, 'value', protocol)}."
                f"{checkpoint}.{split}.s{shard:03d}",
        kind=kind, protocol=protocol, round=None, cohort=cohort, checkpoint=checkpoint,
        reasoning_allowance=K.THINKING_TOKENS_LOW, final_allowance=K.FINAL_TOKENS,
        items=tuple(items), team_id=None, parent_ref=None, leader_seed=None, depends_on=(),
        max_inflight=K.N_AGENTS, parallel_items=1, lane=checkpoint, split=split,
        options=dict(options or {}),
    )


def _write_episode(run_root: Path, cell: ArdCellSpec, item_id: str, *, n_rounds: int):
    episode_id = episode_id_for(cell, item_id)
    decisions = tuple(
        DecisionRef(
            decision_id=decision_id_for(episode_id, round_=r, agent_id=a, phase="revise"),
            episode_id=episode_id, request_id=f"req-{r}-{a}", agent_id=a, round=r,
            phase="revise", protocol=str(cell.protocol.value), checkpoint=cell.checkpoint,
            answer_class="A", answer_raw="A", content_sha256="0" * 64, confidence=0.7,
        )
        for r in range(1, n_rounds + 1) for a in range(K.N_AGENTS)
    )
    episode = EpisodeRecord(
        episode_id=episode_id, item_id=item_id, cohort=str(cell.cohort.value),
        protocol=str(cell.protocol.value), checkpoint=cell.checkpoint,
        checkpoint_revision=K.REVISIONS[cell.checkpoint], decisions=decisions,
    )
    payload = {
        "schema_version": 1, "cell": cell.to_dict(), "source_id": item_id,
        "family": K.Family.MATH500.value, "split": cell.split, "status": "complete",
        "episodes": [episode.to_dict()], "timing": {"finished_at": 0.0},
        "worker": {}, "grades": [], "code_version": "test",
    }
    io.write_json(run_root / "cells" / cell.cell_id / "items" / f"{item_id}.json", payload)


def _coverage_run_root(tmp_path, *, gate_rounds=K.UPDATE_ROUNDS):
    items = [f"math500-{i:04d}" for i in range(4)]
    cells = [_cell(K.Protocol.PEER, items=items), _cell(K.Protocol.GATE, items=items)]
    for cell in cells:
        rounds = K.UPDATE_ROUNDS if cell.protocol is K.Protocol.PEER else gate_rounds
        for item in items:
            _write_episode(tmp_path, cell, item, n_rounds=rounds)
    return cells


def test_decision_coverage_is_grouped_by_protocol_checkpoint_cohort(tmp_path):
    cells = _coverage_run_root(tmp_path)
    rows = G.decision_coverage(tmp_path, cells)
    assert {(r["protocol"], r["checkpoint"], r["cohort"], r["split"]) for r in rows} == {
        ("PEER", "8B", "natural", "test"), ("GATE", "8B", "natural", "test")}
    for row in rows:
        assert row["planned_decisions"] == 4 * K.N_AGENTS * K.UPDATE_ROUNDS
        assert row["coverage"] == pytest.approx(1.0)


def test_coverage_gate_fails_the_arm_that_is_short_not_the_aggregate(tmp_path):
    """An aggregate 'complete enough' can be 100% PEER and 33% GATE - and GATE carries two
    of the three primaries."""
    cells = _coverage_run_root(tmp_path, gate_rounds=1)
    report = G.coverage_gate(tmp_path, cells, hour=24)
    assert report["passed"] is False
    assert report["failed"] == ["coverage:GATE:8B:natural:test"]
    gate_row = next(r for r in report["rows"] if r["protocol"] == "GATE")
    assert gate_row["coverage"] == pytest.approx(1 / 3)
    assert (tmp_path / "gates" / "coverage_gate_h24.json").exists()


def test_hour_thresholds_are_registered_and_rise(tmp_path):
    cells = _coverage_run_root(tmp_path, gate_rounds=2)
    assert G.coverage_gate(tmp_path, cells, hour=12)["passed"] is True    # test floor 0.10
    assert G.coverage_gate(tmp_path, cells, hour=18)["passed"] is True    # 0.667 >= 0.40
    assert G.coverage_gate(tmp_path, cells, hour=24)["passed"] is False   # 0.667 < 0.90
    assert sorted(G.COVERAGE_THRESHOLDS) == [12, 18, 24]


def test_an_unregistered_hour_is_refused(tmp_path):
    cells = _coverage_run_root(tmp_path)
    with pytest.raises(G.GateError):
        G.coverage_gate(tmp_path, cells, hour=6)


def test_cell_reconciliation_reuses_the_study_item_validity_rule(tmp_path):
    cells = _coverage_run_root(tmp_path)
    report = G.ard_cell_reconciliation(tmp_path, cells)
    assert set(report) == {"PEER|8B|natural|test", "GATE|8B|natural|test"}
    for group in report.values():
        assert group["planned_items"] == 4 and group["completed_items"] == 4
    # An item file that belongs to another cell is not credited.
    stray = cells[0]
    path = tmp_path / "cells" / stray.cell_id / "items" / "math500-0000.json"
    payload = json.loads(path.read_text())
    payload["source_id"] = "math500-9999"
    path.write_text(json.dumps(payload))
    again = G.ard_cell_reconciliation(tmp_path, cells)
    assert again["PEER|8B|natural|test"]["completed_items"] == 3


def test_a_cell_may_declare_its_own_decisions_per_item(tmp_path):
    items = ["math500-0000"]
    cell = _cell(K.Protocol.PRIVATE, items=items, kind=ArdCellKind.FORK,
                 options={"expected_decisions_per_item": K.N_AGENTS * 2})
    _write_episode(tmp_path, cell, items[0], n_rounds=2)
    rows = G.decision_coverage(tmp_path, [cell])
    assert rows[0]["planned_decisions"] == K.N_AGENTS * 2
    assert rows[0]["coverage"] == pytest.approx(1.0)


def test_non_decision_kinds_are_not_scored(tmp_path):
    cell = _cell(None, items=("ALL",), kind=ArdCellKind.COMPACT)
    assert G.decision_coverage(tmp_path, [cell]) == []
