"""WP-B: ``candidate_from_text`` == ``extract_candidate`` on text kinds; gold self-grade on all four benchmarks."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from agent_allocation.contracts import (
    CompletionStatus, GenerationKind, GenerationResult, GenerationSpec, OutputKind, ParseStatus, PublicTask,
    TransportStatus, Usage,
)
from agent_allocation.dynorch import grade as grademod
from agent_allocation.dynorch.grade import candidate_from_text, delivered_state, finalize_run, outcome_y
from agent_allocation.dynorch.plan import BaseInputs
from agent_allocation.dynorch.records import RunSpec, new_run_record
from agent_allocation.extraction.final import extract_candidate

PROJECT = Path(__file__).resolve().parents[3]
CFG = json.loads((PROJECT / "configs" / "dynorch_v1.json").read_text("utf-8"))

TEXTS = [
    "<rationale>r</rationale>\n<final>42</final>",
    "<final>  42 </final>",
    "no tags at all",
    "<final>42</final><final>43</final>",
    "<final></final>",
    "<final>   </final>",
    "</final>42<final>",
    "<final>42",
    "   \n\t ",
    "",
    None,
    "<rationale>a</rationale><rationale>b</rationale><final>x</final>",
    "prefix <final>\\boxed{7}</final> suffix",
    "<Final>42</Final>",
]


def _task(kind: OutputKind = OutputKind.MATH, benchmark: str = "aime") -> PublicTask:
    return PublicTask(item_id=f"{benchmark}/2025/1", benchmark=benchmark, source_revision="rev", prompt="p",
                      output_kind=kind, allowed_choices=("A", "B", "C", "D") if kind is OutputKind.OPTION else ())


def _spec(task: PublicTask) -> GenerationSpec:
    return GenerationSpec(run_id="r", partition_id="main", item_id=task.item_id, benchmark=task.benchmark,
                          kind=GenerationKind.DEEP, model="m", endpoint_lock_hash="e", input_manifest_hash="i",
                          decoding_config_hash="d", prompt_version="p", reasoning_budget=1024, completion_headroom=1,
                          root_slot=0, team_size=1, update_round=0, draw_id=0)


def _result(spec: GenerationSpec, text: str | None) -> GenerationResult:
    return GenerationResult(node_id=spec.node_id, attempt_id=f"{spec.node_id}:1", transport_status=TransportStatus.OK,
                            completion_status=CompletionStatus.COMPLETED, generation_id="g", observed_model="m",
                            observed_provider="p", finish_reason="stop", native_finish_reason="stop",
                            visible_content=text, reasoning_returned=False,
                            usage=Usage(prompt_tokens=1, completion_tokens=1, reasoning_tokens=None,
                                        cached_prompt_tokens=None, actual_cost_usd=None),
                            cache_status="MISS", raw_response_path=None, raw_response_sha256=None, latency_seconds=0.1,
                            http_status=200)


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("kind", [OutputKind.MATH, OutputKind.OPTION])
def test_candidate_from_text_matches_extract_candidate(text, kind):
    task = _task(kind, "gpqa_diamond" if kind is OutputKind.OPTION else "aime")
    spec = _spec(task)
    oracle = extract_candidate(_result(spec, text), spec, task)
    ours = candidate_from_text(text, task, node_id=spec.node_id, finish_reason="stop")
    assert ours == oracle


def test_python_kind_is_refused():
    with pytest.raises(ValueError):
        candidate_from_text("<final>x</final>", _task(OutputKind.PYTHON, "mbpp"), node_id="n")


def test_candidate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        candidate_from_text("x", _task(), node_id="")
    with pytest.raises(TypeError):
        candidate_from_text(b"x", _task(), node_id="n")


@pytest.fixture(scope="module")
def base() -> BaseInputs:
    root = Path(CFG["base_run_root"])
    if not (root / "private" / "aime.jsonl").is_file():
        pytest.skip("base run root not readable here")
    return BaseInputs.load(root, model=CFG["primary_model"])


def _record(spec_item: str, benchmark: str, final_message: str | None, status: str = "done", rerun_reason=None):
    spec = RunSpec(run_key="k" * 24, arm="H1", harness="openclaude", item_id=spec_item, benchmark=benchmark,
                   stratum=benchmark, shard_id="parasail", endpoint_tag="parasail/fp8", rung=4, cap_tokens=4096,
                   repeat=1, attempt=1, generation=1, partition="main", allowed_tools=("Agent",), spawn_tools=("Agent",),
                   root_call_ceiling=None, sampling_hash="s", template_sha256="t", prompt_version="dynorch-v1",
                   proxy_version="p", harness_version="h")
    record = new_run_record(spec, status="running", started_at="2026-09-23T00:00:00+00:00")
    record.status = status
    record.final_message = final_message
    record.final_source = "proxy_root" if final_message is not None else "none"
    record.rerun_reason = rerun_reason
    return record


def test_gold_self_grade_on_all_four_benchmarks(base: BaseInputs):
    """Every main item's own gold, wrapped in <final>, grades correct; a wrong answer grades incorrect."""
    n_correct = 0
    for item_id in base.main_items():
        task, private = base.tasks[item_id], base.privates[item_id]
        record = _record(item_id, task.benchmark, f"<rationale>gold</rationale>\n<final>{private.gold_answer}</final>")
        finalize_run(record, task, private)
        assert record.parse_status == "valid", item_id
        assert record.grade_status == "graded" and record.correct is True, (item_id, record.grade_detail)
        assert delivered_state(record) == "delivered" and outcome_y(record) == 1.0
        n_correct += 1
    assert n_correct == 96
    for benchmark in ("aime", "hmmt", "omni_math_hard", "gpqa_diamond"):
        item_id = base.panels[benchmark].main_item_ids[0]
        task, private = base.tasks[item_id], base.privates[item_id]
        wrong = "E" if benchmark == "gpqa_diamond" else "-99999"
        record = _record(item_id, benchmark, f"<final>{wrong}</final>")
        finalize_run(record, task, private)
        if benchmark == "gpqa_diamond":
            assert record.grade_status == "invalid_output" and record.correct is False   # E is not a legal option
            assert delivered_state(record) == "op_no_output" and outcome_y(record) == 0.0
        else:
            assert record.grade_status == "graded" and record.correct is False
            assert delivered_state(record) == "delivered" and outcome_y(record) == 0.0


def test_finalize_not_valid_is_op_no_output_y_zero(base: BaseInputs):
    item_id = base.panels["aime"].main_item_ids[1]
    task, private = base.tasks[item_id], base.privates[item_id]
    for text in ("no final here", None, "<final></final>"):
        record = _record(item_id, "aime", text)
        finalize_run(record, task, private)
        assert record.parse_status in ("missing_final", "empty_content", "malformed_final")
        assert record.correct is False and record.grade_status == "invalid_output"
        assert delivered_state(record) == "op_no_output" and outcome_y(record) == 0.0


def test_finalize_infra_failure_is_nan_never_zero(base: BaseInputs):
    item_id = base.panels["hmmt"].main_item_ids[0]
    task, private = base.tasks[item_id], base.privates[item_id]
    for status, reason in (("transport_failed", "last_reply_transport_or_ledger_error"),
                           ("refused_before_start", None), ("failed", "zero_proxy_calls_served"),
                           ("timeout", "zero_proxy_calls_served"), ("failed", "killed_by_drain")):
        record = _record(item_id, "hmmt", None, status=status, rerun_reason=reason)
        finalize_run(record, task, private)
        assert record.parse_status is None and record.correct is None and record.grade_status is None
        assert delivered_state(record) == "unknown" and math.isnan(outcome_y(record))
    # a timeout WITH calls served is graded like any other run (never rerun, never NaN)
    record = _record(item_id, "hmmt", None, status="timeout")
    finalize_run(record, task, private)
    assert outcome_y(record) == 0.0 and delivered_state(record) == "op_no_output"


def test_finalize_refuses_mismatched_task_or_open_run(base: BaseInputs):
    item_id = base.panels["aime"].main_item_ids[0]
    other = base.panels["aime"].main_item_ids[1]
    task, private = base.tasks[item_id], base.privates[item_id]
    with pytest.raises(ValueError):
        finalize_run(_record(other, "aime", "<final>1</final>"), task, private)
    with pytest.raises(ValueError):
        finalize_run(_record(item_id, "aime", "<final>1</final>", status="running"), task, private)
    assert delivered_state(_record(item_id, "aime", None, status="running")) == "pending"
    assert math.isnan(outcome_y(_record(item_id, "aime", None, status="running")))
    assert grademod.outcome_row(_record(item_id, "aime", None, status="running"))["delivered_state"] == "pending"
