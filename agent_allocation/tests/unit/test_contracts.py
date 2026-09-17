"""Lead-owned contract tests: codec round-trips, stable lock hashes, config validation, grid arithmetic."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from agent_allocation.config import grids_from_config, load_config, validate_config
from agent_allocation.contracts import (
    BENCHMARKS, STUDY_ID, Attempt, AttemptState, Candidate, CompletionStatus, EndpointLock, GateCriterion,
    GateResult, GenerationKind, GenerationResult, GenerationSpec, GradeResult, GradeStatus, LedgerEvent,
    LedgerEventKind, OutputKind, PanelDraw, ParseStatus, PolicyCheckpoint, PrivateEvalSpec, PublicCheckResult,
    PublicTask, CheckStatus, SelectionResult, SelectionStatus, TransportStatus, Usage, Wave, EvalKind, dump_record, load_record,
    public_task_hash,
)

ROOT = Path(__file__).resolve().parents[2]


def _lock(**kw):
    base = dict(model="google/gemini-2.5-flash-lite", provider_tag="google-ai-studio", provider_name="Google AI Studio",
                service_tier="standard", input_price_per_token="0.0000001", output_price_per_token="0.0000004",
                internal_reasoning_price_per_token="0.0000004", context_length=1048576, max_completion_tokens=65535,
                supported_parameters=("reasoning", "max_tokens", "temperature", "top_p"))
    base.update(kw)
    return EndpointLock(**base)


def test_lock_hash_ignores_volatile_fields_and_parameter_order():
    a = _lock(snapshot_path="a", snapshot_sha256="1", fetched_at="t1")
    b = _lock(snapshot_path="b", snapshot_sha256="2", fetched_at="t2",
              supported_parameters=("top_p", "temperature", "max_tokens", "reasoning"))
    assert a.lock_hash == b.lock_hash
    assert a.lock_hash != _lock(provider_tag="google-ai-studio/flex").lock_hash
    assert a.lock_hash != _lock(output_price_per_token="0.0000005").lock_hash


def test_every_contract_record_round_trips():
    task = PublicTask("math500/1", "math500", "rev", "What is 1+1?", OutputKind.MATH)
    spec = GenerationSpec(run_id=STUDY_ID, partition_id="main", item_id=task.item_id, benchmark="math500",
                          kind=GenerationKind.PEER, model="m", endpoint_lock_hash=_lock().lock_hash,
                          input_manifest_hash=public_task_hash(task), decoding_config_hash="d", prompt_version="p",
                          reasoning_budget=1024, completion_headroom=1024, root_slot=1, team_size=3,
                          update_round=2, draw_id=0, parent_ids=("a", "b", "c"))
    usage = Usage(10, 20, None, None, Decimal("0.000123"), {"cost": 0.000123})
    records = [
        task, spec, usage, _lock(fetched_at="x"),
        PrivateEvalSpec("math500/1", "math500", "rev", EvalKind.MATH_VERIFY, "2", None, "h"),
        PanelDraw("math500", ("a",), ("b", "c"), ("d",), 7, "stratified_v1"),
        Wave(0, tuple(f"i{j}" for j in range(16)), ("main", "extension")),
        Attempt(spec.node_id + ":1", spec.node_id, 1, AttemptState.SENT, Decimal("0.002"), "t0", sent_at="t1"),
        GenerationResult(spec.node_id, spec.node_id + ":1", TransportStatus.OK, CompletionStatus.COMPLETED, "gen",
                         "m", "Google AI Studio", "stop", "STOP", "<final>2</final>", False, usage, "MISS",
                         "raw/ab/x.json.gz", "sha", 1.25, 200),
        LedgerEvent(1, LedgerEventKind.RESERVE, Decimal("0.0021"), "t0", attempt_id="x", node_id="n", partition_id="main"),
        Candidate(spec.node_id, spec.node_id, task.item_id, "<final>2</final>", "2", None, ParseStatus.VALID, "stop",
                  OutputKind.MATH),
        PolicyCheckpoint("main/math500/1/peer/3/2/current", task.item_id, "peer", 3, 3, 2, ("a", "b", "c"),
                         ("a", "b", "c", "r0", "r1", "r2"), "plurality", "v1", "main", "peer/3/2"),
        PublicCheckResult("c", "mbpp/1", "sha", CheckStatus.OK, 1, 1, ("pass",), "v1", "in", "env", 0.01),
        SelectionResult("p", "c", "plurality", "v1", "pool", None, 0.001, SelectionStatus.SELECTED),
        GradeResult("c", GradeStatus.GRADED, False, True, "v1", "h"),
        GradeResult("c", GradeStatus.EVALUATOR_ERROR, None, None, "v1", "h"),
        GateResult("gate1", "canary", False, "stop_notify", (GateCriterion("cache_hits", "1", "0", False, "g"),), "t"),
    ]
    for record in records:
        assert load_record(type(record), dump_record(record)) == record


def test_load_record_refuses_unknown_fields_and_wrong_tags():
    task = PublicTask("x", "math500", "rev", "p", OutputKind.MATH)
    data = dump_record(task)
    with pytest.raises(ValueError):
        load_record(PublicTask, {**data, "answer": "leak"})
    with pytest.raises(ValueError):
        load_record(Candidate, data)


def test_packaged_configs_validate():
    for name in ("balanced150", "lean100"):
        assert validate_config(load_config(ROOT / "configs" / f"{name}.json")) == []


def _reference_dag():
    path = ROOT / "tools" / "reference_dag.py"
    spec = importlib.util.spec_from_file_location("reference_dag_for_contracts", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grids_match_the_pack_reference_counts():
    grids = grids_from_config(load_config(ROOT / "configs" / "balanced150.json"))
    ref = _reference_dag()

    def count(grid):
        bank = grid.bank_size
        private_single = grid.private_single_cycles - 1
        team_slots = max(grid.team_sizes) - 1
        private_team = team_slots * grid.private_team_rounds
        peer = sum(grid.team_sizes) * grid.peer_rounds
        extension = (4 * 3 + 1) if grid.extension else 0
        return bank + private_single + private_team + peer + len(grid.deep_multipliers) + extension

    assert count(grids["main"]) == len(ref.build_plan().nodes) == 125
    assert count(grids["high"]) == len(ref.build_plan(a=4096, high=True).nodes) == 123
    assert count(grids["replication"]) == len(ref.build_plan(reduced=True).nodes) == 57
    assert count(grids["main_extension"]) == len(ref.build_plan(extension=True).nodes) == 138
    assert count(grids["pilot_r1"]) == 12 and count(grids["pilot_r2"]) == 15


def test_benchmark_order_is_the_config_order():
    cfg = load_config(ROOT / "configs" / "balanced150.json")
    assert tuple(b["id"] for b in cfg["benchmarks"]) == BENCHMARKS
