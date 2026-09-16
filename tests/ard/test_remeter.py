"""Tests for ``ard/remeter.py`` — recovering the FORK rows' lost metering from the request store.

Two halves, deliberately.  The synthetic half builds records whose token counts are known by
construction, so the replay's arithmetic is pinned without depending on any run.  The real-run
half re-measures the claims the module's docstring makes about the ARD36 run and skips when that
run root is not mounted, because a docstring that asserts measured numbers is only worth what a
test that re-measures them is worth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard.remeter import (
    VERIFIED_FIELDS,
    decisions_table,
    flop_oracle_consistent,
    main,
    remeter_decisions,
    remeter_one,
    store_for,
    verify_against,
)
from agents_scaling.study.resources.oracle import FlopOracle
from agents_scaling.study.types import ProtocolError, RequestRecord, SeedKey

ARD36_RUN_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36")
requires_run = pytest.mark.skipif(
    not decisions_table(ARD36_RUN_ROOT).exists(),
    reason="the frozen ARD36 v1 tables are not mounted",
)

ORACLE = FlopOracle.from_table("8B")
#: ``"\n\n<final>"`` under the pinned Qwen3 vocabulary, measured off the run's own two-phase
#: records (remeter.py docstring).  The synthetic records reuse it so their prefix arithmetic is
#: the run's, not an invention of this test.
PREFILL_IDS = (271, 27, 11822, 29)


# --------------------------------------------------------------------------- synthetic records


def _rid(tag: str) -> str:
    import hashlib

    return hashlib.sha256(tag.encode()).hexdigest()


def _record(
    tag: str,
    *,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    finish_reason: str,
    latency_s: float,
    cached_prefill_tokens: int = 0,
) -> RequestRecord:
    """A committed record with a real FLOP block (the placeholder is refused upstream)."""
    return RequestRecord(
        schema_version=1,
        request_id=_rid(tag),
        identity={"study_id": K.STUDY_ID},
        seed_key=SeedKey("item", "train", "Qwen3-8B@x", 0, 0, "revise", 0, "ARD_FORK"),
        engine_seed=1,
        model={"size": "8B", "served_model_name": "8B"},
        messages=({"role": "user", "content": "task"},),
        chat_template_kwargs={"enable_thinking": True},
        chat_template_hash="h",
        sampling={"max_tokens": K.max_tokens_for(1024)},
        prompt_token_ids=tuple(int(i) for i in prompt_ids),
        prompt_tokens=len(prompt_ids),
        response={
            "token_ids": [int(i) for i in completion_ids],
            "finish_reason": finish_reason,
            "cached_prefill_tokens": int(cached_prefill_tokens),
        },
        flops=ORACLE.flops_record(len(prompt_ids), len(completion_ids)),
        timing={"latency_s": float(latency_s)},
        endpoint={"host": "node1", "port": 8000},
        attempts=1,
        producer={},
    ).with_content_sha256()


def _completion(thinking: int, final: int, *, closed: bool = True) -> list[int]:
    """``<think> · thinking · </think> · final`` (+ ``<|im_end|>`` when the turn closed)."""
    ids = [K.THINK_START_ID, *([9] * thinking), K.THINK_END_ID, *([7] * final)]
    return [*ids, K.IM_END_ID] if closed else ids


def _publish(run_root: Path, record: RequestRecord) -> str:
    store_for(run_root).publish(record)
    return record.request_id


def _table(run_root: Path, rows: Sequence[dict[str, Any]]) -> None:
    from agents_scaling.study.aggregate import _write_parquet

    path = decisions_table(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(path, rows)


def _row(decision_id: str, physical: Sequence[str], overrides: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "decision_id": decision_id,
        "cohort": "natural",
        "checkpoint": "8B",
        "budget_tokens": 1024,
        "physical_request_ids": list(physical),
        "final_overrun": False,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def single_phase(tmp_path: Path) -> tuple[Path, str, dict[str, Any]]:
    """A one-request decision that stopped naturally: 40 thinking ids, 12 report ids."""
    run_root = tmp_path / "run"
    prompt = list(range(100))
    record = _record(
        "single",
        prompt_ids=prompt,
        completion_ids=_completion(40, 12),
        finish_reason="stop",
        latency_s=1.5,
        cached_prefill_tokens=64,
    )
    rid = _publish(run_root, record)
    frozen = {
        "thinking_tokens": 40,
        "final_tokens": 12,
        "prompt_tokens": 100,
        "cached_prefill_tokens": 64,
        "latency": 1.5,
        "cap_hit": False,
        "boundary_injected": False,
        "final_overrun": False,
        "flops_total": record.flops["total"],
    }
    _table(run_root, [_row("d-single", [rid], frozen)])
    return run_root, rid, frozen


# --------------------------------------------------------------------------- the replay


def test_single_phase_replay_recovers_every_metering_field(single_phase) -> None:
    run_root, rid, frozen = single_phase
    got = remeter_one(store_for(run_root), [rid], budget_tokens=1024)
    for field, expected in frozen.items():
        assert got[field] == expected, field
    assert got["endpoint_id"] == "node1:8000"
    assert got["finish_reason"] == "stop"
    assert got["think_status"] is None
    assert got["physical_request_ids"] == [rid]


def test_boundary_injected_is_exact_equality_with_the_budget(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    rid = _publish(
        run_root,
        _record("bnd", prompt_ids=list(range(10)), completion_ids=_completion(64, 3),
                finish_reason="stop", latency_s=0.5),
    )
    store = store_for(run_root)
    assert remeter_one(store, [rid], budget_tokens=64)["boundary_injected"] is True
    assert remeter_one(store, [rid], budget_tokens=65)["boundary_injected"] is False


def test_cap_hit_tracks_the_whole_response_cap_not_the_thinking_budget(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    rid = _publish(
        run_root,
        _record("cap", prompt_ids=list(range(10)), completion_ids=_completion(64, 5, closed=False),
                finish_reason="length", latency_s=0.5),
    )
    got = remeter_one(store_for(run_root), [rid], budget_tokens=64)
    assert got["cap_hit"] is True and got["boundary_injected"] is True


def test_broken_delimiters_are_a_status_not_an_exception(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    rid = _publish(
        run_root,
        _record("nothink", prompt_ids=list(range(10)), completion_ids=[7, 7, 7],
                finish_reason="stop", latency_s=0.2),
    )
    got = remeter_one(store_for(run_root), [rid], budget_tokens=64)
    assert got["think_status"] == K.ParseStatus.NO_THINK_OPEN.value
    assert got["thinking_tokens"] == 0 and got["final_tokens"] == 0


# --------------------------------------------------------------------------- two-phase


@pytest.fixture()
def two_phase(tmp_path: Path) -> tuple[Path, list[str], dict[str, Any]]:
    """A forced-boundary decision: phase 1 thinks to the budget, phase 2 delivers the report."""
    run_root = tmp_path / "run"
    prompt = list(range(100))
    first = _record(
        "tp1", prompt_ids=prompt, completion_ids=_completion(64, 9, closed=False),
        finish_reason="length", latency_s=2.0, cached_prefill_tokens=32,
    )
    # Phase 2's prompt is phase 1's prompt, phase 1's ids through ``</think>``, then the
    # prefilled report opening -- exactly how the run built it.
    prefix = [*prompt, K.THINK_START_ID, *([9] * 64), K.THINK_END_ID, *PREFILL_IDS]
    second = _record(
        "tp2", prompt_ids=prefix, completion_ids=[7] * 20 + [K.IM_END_ID],
        finish_reason="stop", latency_s=0.75, cached_prefill_tokens=len(prefix) - 1,
    )
    ids = [_publish(run_root, first), _publish(run_root, second)]
    frozen = {
        "thinking_tokens": 64,                       # phase 1's
        "final_tokens": 20,                          # what phase 2 generated
        "prompt_tokens": 100,                        # phase 1's prompt
        "cached_prefill_tokens": 32,                 # phase 1's, never phase 2's
        "latency": 2.75,                             # the decision is both calls
        "cap_hit": False,                            # phase 2's finish reason
        "boundary_injected": True,
        "final_overrun": False,
        "flops_total": first.flops["total"] + second.flops["total"],
    }
    _table(run_root, [_row("d-two", ids, frozen)])
    return run_root, ids, frozen


def test_two_phase_replay_merges_the_way_the_run_did(two_phase) -> None:
    run_root, ids, frozen = two_phase
    got = remeter_one(store_for(run_root), ids, budget_tokens=64)
    for field, expected in frozen.items():
        assert got[field] == expected, field
    assert got["physical_request_ids"] == ids


def test_two_phase_final_overrun_has_the_two_documented_rules(tmp_path: Path) -> None:
    """253 generated report tokens is under the 256 allowance, but 4 prefilled ids push the
    post-``</think>`` span over it — which is the whole reason frozen v1 spells the column two
    ways (remeter.py docstring)."""
    run_root = tmp_path / "run"
    prompt = list(range(10))
    first = _record("o1", prompt_ids=prompt, completion_ids=_completion(64, 2, closed=False),
                    finish_reason="length", latency_s=1.0)
    prefix = [*prompt, K.THINK_START_ID, *([9] * 64), K.THINK_END_ID, *PREFILL_IDS]
    second = _record("o2", prompt_ids=prefix, completion_ids=[7] * 253 + [K.IM_END_ID],
                     finish_reason="stop", latency_s=1.0)
    ids = [_publish(run_root, first), _publish(run_root, second)]
    got = remeter_one(store_for(run_root), ids, budget_tokens=64)
    assert got["final_tokens"] == 253
    assert got["final_overrun"] is False           # nothing the model generated was dropped
    assert got["final_overrun_span"] is True       # 4 + 253 > FINAL_TOKENS
    assert 4 + got["final_tokens"] > K.FINAL_TOKENS


def test_more_than_two_physical_requests_is_refused(two_phase) -> None:
    run_root, ids, _ = two_phase
    with pytest.raises(ProtocolError, match="physical requests"):
        remeter_one(store_for(run_root), [*ids, ids[0]], budget_tokens=64)


def test_an_empty_id_list_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="at least one physical request id"):
        remeter_one(store_for(tmp_path / "run"), [], budget_tokens=64)


def test_a_continuation_on_an_unclosed_thinking_channel_is_refused(tmp_path: Path) -> None:
    """With no ``</think>`` there is nowhere to split the re-sent prefix, so the prefilled
    opening cannot be located and every span-rule field would be quietly wrong."""
    run_root = tmp_path / "run"
    prompt = list(range(10))
    first = _record("u1", prompt_ids=prompt, completion_ids=[K.THINK_START_ID] + [9] * 64,
                    finish_reason="length", latency_s=1.0)
    second = _record("u2", prompt_ids=[*prompt, K.THINK_START_ID, *([9] * 64), *PREFILL_IDS],
                     completion_ids=[7] * 5 + [K.IM_END_ID], finish_reason="stop", latency_s=1.0)
    ids = [_publish(run_root, first), _publish(run_root, second)]
    with pytest.raises(ProtocolError, match="premature_reasoning_eos|cannot be split"):
        remeter_one(store_for(run_root), ids, budget_tokens=64)


# --------------------------------------------------------------------------- the table API


def test_remeter_decisions_keys_by_decision_id(single_phase) -> None:
    run_root, _, frozen = single_phase
    out = remeter_decisions(run_root, ["d-single"])
    assert set(out) == {"d-single"}
    assert out["d-single"]["thinking_tokens"] == frozen["thinking_tokens"]


def test_remeter_decisions_omits_rather_than_invents(single_phase) -> None:
    """An unknown id and an id whose record is not in the store both drop out; neither becomes
    a row of zeros, which is the defect being repaired."""
    run_root, rid, frozen = single_phase
    _table(run_root, [
        _row("d-single", [rid], frozen),
        _row("d-absent", ["0" * 64], frozen),
    ])
    out = remeter_decisions(run_root, ["d-single", "d-absent", "d-never-existed"])
    assert set(out) == {"d-single"}


def test_remeter_decisions_prefers_the_longer_provenance(two_phase) -> None:
    """When one decision id carries both a one-id and a two-id spelling, the two-id row wins:
    a decision executed as two physical requests is two (67 real rows are the short spelling)."""
    run_root, ids, frozen = two_phase
    _table(run_root, [
        _row("d-two", ids[:1], frozen),
        _row("d-two", ids, frozen),
    ])
    out = remeter_decisions(run_root, ["d-two"])
    assert out["d-two"]["physical_request_ids"] == ids
    assert out["d-two"]["latency"] == frozen["latency"]


# --------------------------------------------------------------------------- verification


def test_verify_against_passes_on_a_single_phase_row_that_agrees(single_phase) -> None:
    run_root, _, _ = single_phase
    report = verify_against(run_root, ["d-single"])
    assert report.ok
    assert report.rows_compared == 1
    assert not any(report.mismatches.values())
    assert report.missing_record == 0


def test_verify_against_passes_on_a_two_phase_row_that_agrees(two_phase) -> None:
    """The continuation path has to be covered too, or the gate only proves the easy half."""
    run_root, _, _ = two_phase
    report = verify_against(run_root, ["d-two"])
    assert report.ok and report.rows_compared == 1


def test_verify_against_catches_a_single_corrupted_field(single_phase) -> None:
    run_root, rid, frozen = single_phase
    _table(run_root, [_row("d-single", [rid], {**frozen, "latency": frozen["latency"] + 1.0})])
    report = verify_against(run_root, ["d-single"])
    assert not report.ok
    assert report.mismatches["latency"] == 1
    assert [m.field for m in report.examples] == ["latency"]
    assert report.examples[0].remetered == frozen["latency"]


def test_verify_against_has_no_float_tolerance(single_phase) -> None:
    """Byte-equality means byte-equality; a one-ulp latency drift is a mismatch."""
    run_root, rid, frozen = single_phase
    import math

    _table(run_root, [_row("d-single", [rid], {**frozen, "latency": math.nextafter(1.5, 2.0)})])
    assert not verify_against(run_root, ["d-single"]).ok


def test_verify_against_skips_fork_rows(single_phase) -> None:
    """FORK rows carry the zeros the module exists to replace; verifying against them would
    measure the defect instead of the replay."""
    run_root, rid, frozen = single_phase
    zeroed = {f: (False if isinstance(frozen[f], bool) else 0) for f in VERIFIED_FIELDS}
    _table(run_root, [_row("d-fork", [rid], {**zeroed, "cohort": "fork"})])
    report = verify_against(run_root, ["d-fork"])
    assert report.rows_skipped_fork == 1
    assert report.rows_compared == 0
    assert not report.ok            # nothing was compared, so nothing was established


def test_verify_against_counts_a_missing_record(single_phase) -> None:
    run_root, _, frozen = single_phase
    _table(run_root, [_row("d-single", ["0" * 64], frozen)])
    report = verify_against(run_root, ["d-single"])
    assert report.missing_record == 1 and report.unreplayable == 0 and not report.ok


def test_verify_against_separates_unreplayable_from_missing(single_phase) -> None:
    """A record that is present but cannot be replayed is a different fact from an absent one,
    and the report must not blur them."""
    run_root, rid, frozen = single_phase
    _table(run_root, [_row("d-single", [rid, rid, rid], frozen)])
    report = verify_against(run_root, ["d-single"])
    assert report.unreplayable == 1 and report.missing_record == 0 and not report.ok


def test_verify_against_does_not_double_count_a_repeated_id(single_phase) -> None:
    run_root, _, _ = single_phase
    report = verify_against(run_root, ["d-single", "d-single", "d-single"])
    assert report.requested == 1 and report.rows_compared == 1 and report.ok


def test_verify_against_counts_an_unknown_id(single_phase) -> None:
    run_root, _, _ = single_phase
    report = verify_against(run_root, ["d-single", "d-nowhere"])
    assert report.unknown_ids == 1 and report.rows_compared == 1


def test_verify_against_accepts_the_span_rule_and_counts_it(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    prompt = list(range(10))
    first = _record("s1", prompt_ids=prompt, completion_ids=_completion(64, 2, closed=False),
                    finish_reason="length", latency_s=1.0)
    prefix = [*prompt, K.THINK_START_ID, *([9] * 64), K.THINK_END_ID, *PREFILL_IDS]
    second = _record("s2", prompt_ids=prefix, completion_ids=[7] * 254 + [K.IM_END_ID],
                     finish_reason="stop", latency_s=1.0)
    ids = [_publish(run_root, first), _publish(run_root, second)]
    _table(run_root, [_row("d-span", ids, {
        "thinking_tokens": 64, "final_tokens": 254, "prompt_tokens": 10,
        "cached_prefill_tokens": 0, "latency": 2.0, "cap_hit": False,
        "boundary_injected": True,
        "final_overrun": True,          # the WP2/episode lane's spelling
        "flops_total": first.flops["total"] + second.flops["total"],
    })])
    report = verify_against(run_root, ["d-span"])
    assert report.ok
    assert report.overrun_span_rule == 1


# --------------------------------------------------------------------------- the FLOP oracle


def test_flop_oracle_consistent_on_a_sound_single_request_row() -> None:
    prompt, thinking, final = 100, 40, 12
    completion = 1 + thinking + 1 + final + 1          # the turn closed, so <|im_end|> counts
    row = {
        "checkpoint": "8B", "prompt_tokens": prompt, "thinking_tokens": thinking,
        "final_tokens": final, "cap_hit": False, "physical_request_ids": ["a"],
        "flops_total": ORACLE.call(prompt, completion),
    }
    assert flop_oracle_consistent(row)
    assert not flop_oracle_consistent({**row, "flops_total": row["flops_total"] + 1})


def test_flop_oracle_consistent_drops_the_eos_token_on_a_cap_hit() -> None:
    prompt, thinking, final = 100, 40, 12
    row = {
        "checkpoint": "8B", "prompt_tokens": prompt, "thinking_tokens": thinking,
        "final_tokens": final, "cap_hit": True, "physical_request_ids": ["a"],
        "flops_total": ORACLE.call(prompt, 1 + thinking + 1 + final),
    }
    assert flop_oracle_consistent(row)
    assert not flop_oracle_consistent({**row, "cap_hit": False})


def test_flop_oracle_consistent_states_the_fork_defect() -> None:
    """Non-zero FLOPs beside zero token counts is exactly what the FORK rows look like."""
    row = {
        "checkpoint": "8B", "prompt_tokens": 0, "thinking_tokens": 0, "final_tokens": 0,
        "cap_hit": False, "physical_request_ids": ["a"], "flops_total": 17_246_284_891_704,
    }
    assert not flop_oracle_consistent(row)


def test_flop_oracle_consistent_is_false_when_flops_are_null() -> None:
    assert not flop_oracle_consistent({
        "checkpoint": "8B", "prompt_tokens": 100, "thinking_tokens": 40, "final_tokens": 12,
        "cap_hit": False, "physical_request_ids": ["a"], "flops_total": None,
    })


def test_flop_oracle_consistent_is_false_out_of_domain() -> None:
    """A two-phase row's FLOPs are a sum over two prompts, one of which is not in the row."""
    prompt, thinking, final = 100, 40, 12
    row = {
        "checkpoint": "8B", "prompt_tokens": prompt, "thinking_tokens": thinking,
        "final_tokens": final, "cap_hit": False, "physical_request_ids": ["a", "b"],
        "flops_total": ORACLE.call(prompt, 1 + thinking + 1 + final + 1),
    }
    assert not flop_oracle_consistent(row)


# --------------------------------------------------------------------------- CLI


def test_main_exits_nonzero_when_verification_fails(single_phase, capsys) -> None:
    run_root, rid, frozen = single_phase
    _table(run_root, [_row("d-single", [rid], {**frozen, "thinking_tokens": 41})])
    assert main(["--run-root", str(run_root), "--verify", "--sample", "0"]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["verify"]["mismatches"]["thinking_tokens"] == 1


def test_main_writes_the_summary_it_prints(single_phase, tmp_path: Path, capsys) -> None:
    run_root, _, _ = single_phase
    out = tmp_path / "summary.json"
    assert main(["--run-root", str(run_root), "--verify", "--sample", "0", "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == json.loads(capsys.readouterr().out)


def test_main_refuses_a_run_root_without_a_compacted_table(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="decisions.parquet"):
        main(["--run-root", str(tmp_path / "empty"), "--verify"])


# --------------------------------------------------------------------------- the real run


@requires_run
def test_frozen_fork_rows_are_zeroed_and_non_fork_rows_are_not() -> None:
    """The brief's premise, re-measured rather than taken on trust."""
    import pyarrow.parquet as pq

    table = pq.read_table(
        decisions_table(ARD36_RUN_ROOT),
        columns=["cohort", "thinking_tokens", "final_tokens", "prompt_tokens",
                 "cached_prefill_tokens", "latency", "flops_total"],
    ).to_pylist()
    forks = [r for r in table if r["cohort"] == "fork"]
    others = [r for r in table if r["cohort"] != "fork"]
    assert (len(forks), len(others)) == (1440, 65664)
    for row in forks:
        assert row["thinking_tokens"] == row["final_tokens"] == 0
        assert row["prompt_tokens"] == row["cached_prefill_tokens"] == 0
        assert row["latency"] == 0.0
        assert row["flops_total"] > 0
    assert not any(r["latency"] == 0.0 for r in others)


@requires_run
def test_every_fork_decision_is_recoverable_and_its_flops_prove_it() -> None:
    """All 1,440 replay, and the replayed FLOPs are byte-identical to the frozen column — which
    is what makes the recovered token counts the run's own and not a plausible reconstruction."""
    import pyarrow.parquet as pq

    rows = pq.read_table(
        decisions_table(ARD36_RUN_ROOT),
        columns=["decision_id", "cohort", "flops_total", "final_overrun"],
    ).to_pylist()
    forks = [r for r in rows if r["cohort"] == "fork"]
    recovered = remeter_decisions(ARD36_RUN_ROOT, [r["decision_id"] for r in forks])
    assert len(recovered) == len(forks) == 1440

    got = [(r, recovered[r["decision_id"]]) for r in forks]
    assert sum(1 for r, o in got if o["flops_total"] == r["flops_total"]) == 1440
    # The FORK lane spells ``final_overrun`` the WP2/episode way; the generated-token rule
    # misses 6 of the 1,440.  Both numbers are docstring claims.
    assert sum(1 for r, o in got if o["final_overrun_span"] == r["final_overrun"]) == 1440
    assert sum(1 for r, o in got if o["final_overrun"] == r["final_overrun"]) == 1434
    assert sum(1 for _, o in got if o["boundary_injected"]) == 196
    assert all(o["latency"] > 0 and o["prompt_tokens"] > 0 for _, o in got)


@requires_run
def test_replay_reproduces_frozen_metering_on_non_fork_decisions() -> None:
    """The gate: a re-meterer that cannot reproduce known-good rows must not be used on the
    broken ones.  Drawn from the ids whose own rows agree, so the 67 defective-provenance rows
    are not silently smuggled into the pass."""
    import pyarrow.parquet as pq
    import random
    from collections import defaultdict

    rows = pq.read_table(
        decisions_table(ARD36_RUN_ROOT),
        columns=["decision_id", "cohort", *VERIFIED_FIELDS],
    ).to_pylist()
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[row["decision_id"]].append(row)
    disagreeing = {
        k for k, v in by_id.items()
        if len(v) > 1 and any(len({r[f] for r in v}) > 1 for f in VERIFIED_FIELDS)
    }
    assert len(disagreeing) == 67

    clean = sorted({
        r["decision_id"] for r in rows
        if r["cohort"] != "fork" and r["decision_id"] not in disagreeing
    })
    sample = random.Random(7).sample(clean, 40)
    report = verify_against(ARD36_RUN_ROOT, sample)
    assert report.ok, report.to_dict()
    assert report.rows_compared >= len(sample)
    assert report.missing_record == 0


@requires_run
def test_flop_oracle_flags_exactly_the_defective_rows_from_the_row_alone() -> None:
    """1,244 single-request FORK rows plus the 67 whose provenance lost a phase-2 id, and
    nothing else, out of 44,904 single-request rows."""
    import pyarrow.parquet as pq

    rows = pq.read_table(
        decisions_table(ARD36_RUN_ROOT),
        columns=["cohort", "checkpoint", "prompt_tokens", "thinking_tokens", "final_tokens",
                 "cap_hit", "flops_total", "physical_request_ids"],
    ).to_pylist()
    single = [r for r in rows if len(r["physical_request_ids"]) == 1]
    assert len(single) == 44904
    bad = [r for r in single if not flop_oracle_consistent(r)]
    assert len(bad) == 1311
    assert sum(1 for r in bad if r["cohort"] == "fork") == 1244
    assert sum(1 for r in bad if r["cohort"] != "fork") == 67
