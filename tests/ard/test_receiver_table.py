"""Stage 0 step 10: grading the 1,280 receiver probes and recovering their margins.

Two layers, following ``tests/ard/test_panels.py``.  The synthetic layer builds the smallest
tables that exhibit each thing this module exists to get right -- a grade the GRADE lane
cannot produce, a tri-state that keeps "unresolvable" out of "wrong", a control differenced
within program *and* budget *and* completion seed, a report span realigned to stored
alternatives without a tokenizer, and the two margin traps (a decimal prefix pair, a span that
will not align).  The frozen layer runs the real thing against ``ard36/tables_v2`` and pins
every number quoted in the module docstring; it skips when that run root is not mounted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import receiver as R
from agents_scaling.ard import receiver_table as RT
from agents_scaling.study.aggregate import _write_parquet
from agents_scaling.study.types import ProtocolError

RUN_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36")
V2_TABLES = RUN_ROOT / "tables_v2"

FROZEN = pytest.mark.skipif(not (V2_TABLES / "decisions.parquet").exists(),
                            reason="ard36 tables_v2 not mounted")


# --------------------------------------------------------------------------- keys


def test_the_arm_comes_from_the_packet_policy_and_only_the_frozen_five():
    assert RT.arm_of("receiver:MINORITY_CHECKED") == "MINORITY_CHECKED"
    assert RT.arm_of("receiver:PRIVATE_TRACE") == RT.CONTROL_ARM
    assert set(RT.ARMS) == set(K.RECEIVER_PRESENTATIONS)
    with pytest.raises(ProtocolError):
        RT.arm_of("receiver:MAJORITY_SHOUTED")
    with pytest.raises(ProtocolError):
        RT.arm_of("task+own_root+own_latest+packet")


def test_episode_rep_zero_has_no_suffix_and_that_is_not_a_malformed_id():
    """``types.episode_id_for`` omits ``.e0``; reading that as an error loses half the run."""
    assert RT.episode_rep_of("CELL|receiver:p07", "receiver:p07") == 0
    assert RT.episode_rep_of("CELL|receiver:p07.e1", "receiver:p07") == 1
    with pytest.raises(ProtocolError):
        RT.episode_rep_of("CELL|receiver:p07.rep1", "receiver:p07")
    with pytest.raises(ProtocolError):
        RT.episode_rep_of("CELL|receiver:p08", "receiver:p07")


# --------------------------------------------------------------------------- the reference


def test_the_reference_is_the_independent_evaluator_and_agrees_with_the_generator():
    """``graders/trace.py`` parses the rendered text; ``receiver.interpret`` walks the records."""
    trace = pytest.importorskip("agents_scaling.ard.graders.trace")
    references = RT.reference_values()
    assert len(references) == K.RECEIVER_PROGRAMS
    for program in R.programs():
        assert references[program.program_id] == trace.evaluate_program(program.source).final_value
        assert references[program.program_id] == R.interpret(program)[-1].value


def test_the_grade_lane_still_cannot_produce_these_grades():
    """The trap the table exists to route around, pinned so a future fix is noticed."""
    graders = pytest.importorskip("agents_scaling.ard.graders")
    # grader_module knows the trace family; grade() does not dispatch to it.
    assert graders.grader_module(graders.TRACE_FAMILY) is not None
    with pytest.raises(ValueError, match="unknown grader family"):
        graders.grade(graders.TRACE_FAMILY, "41", "receiver:p00", store=_EmptyGold())
    # And a receiver row does not even carry that family: it is stamped math500, so the
    # lane would ask the protected store for a math gold that was never written.
    assert R.RECEIVER_TASK_FAMILY == K.Family.MATH500.value
    assert R.RECEIVER_FAMILY == graders.TRACE_FAMILY
    store = graders.GoldStore(run_root="", math={}, mmlu={})
    with pytest.raises(graders.GoldUnavailable):
        store.reference(K.Family.MATH500.value, "receiver:p00")
    with pytest.raises(graders.GoldUnavailable):
        store.reference(graders.TRACE_FAMILY, "receiver:p00")


class _EmptyGold:
    """A gold store with no trace table -- which is the real one's shape for E3."""

    def reference(self, family: str, item_id: str) -> dict[str, Any]:
        return {}


# --------------------------------------------------------------------------- the margin


def _response(tokens: list[tuple[str, float, list[tuple[str, float]]]], content: str,
              *, trailing: tuple[str, float] | None = None) -> dict[str, Any]:
    """A store ``response`` block in the shape ``client._top_logprobs`` writes."""
    rows = [[{"token": t, "logprob": lp} for t, lp in alts] for _, _, alts in tokens]
    logprobs = [lp for _, lp, _ in tokens]
    if trailing is not None:
        rows.append([{"token": trailing[0], "logprob": trailing[1]}])
        logprobs.append(trailing[1])
    return {"content": content, "token_logprobs": logprobs, "top_logprobs": rows}


def test_the_report_span_is_realigned_without_a_tokenizer():
    """The store keeps sampled tokens as ids; the text is recovered from content + logprobs."""
    response = _response(
        [("9", -0.1, [("9", -0.1), ("2", -8.0)]),
         ("1", -0.2, [("1", -0.2), ("5", -3.0)]),
         ("</final>", -0.3, [("</final>", -0.3)])],
        "91</final>")
    entries = RT.report_span_entries(response)
    assert [e["token"] for e in entries] == ["9", "1", "</final>"]
    assert entries[0]["logprob"] == -0.1
    assert entries[0]["top_logprobs"][1] == {"token": "2", "logprob": -8.0}


def test_a_trailing_turn_token_is_not_part_of_the_content_span():
    response = _response(
        [("4", -0.1, [("4", -0.1), ("7", -5.0)])], "4", trailing=("<|im_end|>", -0.01))
    entries = RT.report_span_entries(response)
    assert [e["token"] for e in entries] == ["4"]


def test_the_sampled_token_is_disambiguated_by_the_content_not_by_rank():
    """Temperature 0.6 means the sampled token is often not rank 0 (ard36: 75/85 on one span)."""
    response = _response(
        [("2", -6.5, [("9", -0.1), ("2", -6.5)])], "2")
    entries = RT.report_span_entries(response)
    assert [e["token"] for e in entries] == ["2"]


def test_an_unalignable_span_is_refused_whole_rather_than_returned_partial():
    """A partial span would put the margin at the wrong token instead of declaring itself."""
    response = _response([("9", -0.1, [("9", -0.1)])], "42")
    assert RT.report_span_entries(response) is None
    assert RT.report_span_entries({"content": "42"}) is None
    assert RT.report_span_entries(_response([], "")) is None


def test_the_margin_is_the_difference_at_the_first_distinguishing_token():
    program = R.programs()[0]
    packet = R.build_packet(program, RT.CONTROL_ARM)
    chosen, alternative = (str(v) for v in packet.candidate_answers)
    assert chosen[0] != alternative[0], "this fixture needs answers differing at digit 0"
    response = _response(
        [(chosen[0], -0.5, [(chosen[0], -0.5), (alternative[0], -4.5)]),
         (chosen[1], -0.2, [(chosen[1], -0.2)])],
        chosen)
    margin = RT.recover_margin(_Store({"rid": response}), ["rid"], packet, produced=chosen)
    assert margin.status == R.MARGIN_OK
    assert margin.value == pytest.approx(4.0)
    assert margin.alternative_token == alternative[0]


def test_a_decimal_prefix_pair_never_yields_a_number():
    """``first_differing_token`` cannot compare "1" with "15"; a wrong margin is worse than none."""
    packet = _packet_with_answers(15, 1)
    margin = RT.recover_margin(_Store({}), ["rid"], packet, produced="15")
    assert margin.status == RT.MARGIN_PREFIX_PAIR and margin.value is None
    # The dangerous direction -- the shorter answer chosen -- is guarded too.
    margin = RT.recover_margin(_Store({}), ["rid"], packet, produced="1")
    assert margin.status == RT.MARGIN_PREFIX_PAIR and margin.value is None
    # A pair that is not a prefix pair is not caught by the guard.
    assert not RT._is_prefix_pair(_packet_with_answers(91, 21), "91")


def test_an_unalignable_span_is_missing_under_its_own_status_not_the_runtime_one():
    """``top_logprobs_not_retained`` would claim the run discarded data it in fact kept."""
    packet = _packet_with_answers(91, 21)
    store = _Store({"rid": _response([("9", -0.1, [("9", -0.1)])], "42")})
    margin = RT.recover_margin(store, ["rid"], packet, produced="91")
    assert margin.status == RT.MARGIN_SPAN_UNALIGNED and margin.value is None
    assert RT.MARGIN_SPAN_UNALIGNED not in R.MARGIN_STATUSES


def test_the_span_is_read_from_the_last_physical_request():
    """A two-phase decision's report was generated by the continuation, not by phase 1."""
    packet = _packet_with_answers(91, 21)
    phase1 = _response([("2", -0.1, [("2", -0.1), ("9", -9.0)])], "2")
    phase2 = _response([("9", -0.1, [("9", -0.1), ("2", -5.0)]),
                        ("1", -0.2, [("1", -0.2)])], "91")
    margin = RT.recover_margin(_Store({"a": phase1, "b": phase2}), ["a", "b"], packet,
                               produced="91")
    assert margin.value == pytest.approx(4.9)


def _packet_with_answers(first: int, second: int) -> R.ReceiverPacket:
    program = R.programs()[0]
    packet = R.build_packet(program, RT.CONTROL_ARM)
    from dataclasses import replace

    return replace(packet, candidate_answers=(first, second), gold=first)


class _Store:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self._responses = responses

    def get(self, request_id: str) -> Any:
        response = self._responses.get(request_id)
        if response is None:
            return None
        return type("Record", (), {"response": response})()


# --------------------------------------------------------------------------- the table


def _tables(tmp_path: Path, decisions: list[dict[str, Any]]) -> Path:
    tables = tmp_path / "tables"
    tables.mkdir(exist_ok=True)
    episodes = [{"episode_id": d["episode_id"], "cohort": "receiver",
                 "packet_policy": d.pop("packet_policy")} for d in decisions]
    _write_parquet(tables / RT.DECISIONS_TABLE, decisions)
    _write_parquet(tables / RT.EPISODES_TABLE, episodes)
    return tables


def _decision(item_id: str, arm: str, *, answer: str | None, budget: int = 1024,
              rep: int = 0, parse_status: str = "ok") -> dict[str, Any]:
    # One cell runs one presentation (``cells.options["presentation"]``), so the arm has to
    # separate the cells or two arms of one probe collide on a single episode id.
    rep_tag = f".e{rep}" if rep else ""
    cell = f"RECEIVER.receiver.none.8B.T{budget}{rep_tag}.test.s{RT.ARMS.index(arm):03d}"
    episode = f"{cell}|{item_id}{rep_tag}"
    return {
        "decision_id": f"{episode}|r0|a0|receiver", "episode_id": episode, "cell_id": cell,
        "cell_kind": "RECEIVER", "item_id": item_id, "phase": K.PURPOSE_RECEIVER,
        "checkpoint": "8B", "answer_raw": answer, "answer_class": None, "answer_logprob": -0.1,
        "confidence": 0.9, "parse_status": parse_status, "budget_tokens": budget,
        "thinking_tokens": 7, "final_tokens": 11, "latency": 1.5, "cap_hit": False,
        "boundary_injected": False, "final_overrun": False, "n_packets": 0,
        "packet_tokens": 0, "seen_message_ids": [], "physical_request_ids": [],
        "packet_policy": f"receiver:{arm}",
    }


def test_an_unresolvable_answer_is_null_not_false(tmp_path: Path):
    """``compact._is_correct``'s rule: ambiguous is missing, never 'the model was wrong'."""
    gold = RT.reference_values()["receiver:p00"]
    rows, summary = RT.build_rows(_tables(tmp_path, [
        _decision("receiver:p00", RT.CONTROL_ARM, answer=str(gold)),
        _decision("receiver:p01", RT.CONTROL_ARM, answer="the answer is not a residue"),
    ]))
    by_item = {r["item_id"]: r for r in rows}
    assert by_item["receiver:p00"]["is_correct"] is True
    assert by_item["receiver:p00"]["evaluator_status"] == "correct"
    assert by_item["receiver:p01"]["is_correct"] is None
    assert by_item["receiver:p01"]["evaluator_status"] == "ambiguous"
    assert summary["graded"] == 1 and summary["ungraded"] == 1


def test_the_control_is_matched_on_program_budget_and_completion_seed(tmp_path: Path):
    """A control from another budget would turn a budget effect into a presentation effect."""
    gold = RT.reference_values()["receiver:p00"]
    wrong = (gold + 1) % K.RECEIVER_MODULUS
    rows, _ = RT.build_rows(_tables(tmp_path, [
        _decision("receiver:p00", RT.CONTROL_ARM, answer=str(wrong), budget=1024),
        _decision("receiver:p00", "MINORITY_CHECKED", answer=str(gold), budget=1024),
        _decision("receiver:p00", "MINORITY_CHECKED", answer=str(gold), budget=4096),
    ]))
    by_key = {(r["arm"], r["budget_tokens"]): r for r in rows}
    assert by_key[("MINORITY_CHECKED", 1024)]["deviation_from_control"] == 1
    assert by_key[(RT.CONTROL_ARM, 1024)]["deviation_from_control"] == 0
    # No 4096 control exists, so the 4096 draw differences against nothing.
    assert by_key[("MINORITY_CHECKED", 4096)]["deviation_from_control"] is None
    assert by_key[("MINORITY_CHECKED", 4096)]["control_decision_id"] is None


def test_a_deviation_is_null_when_either_side_is_unresolvable(tmp_path: Path):
    gold = RT.reference_values()["receiver:p00"]
    rows, _ = RT.build_rows(_tables(tmp_path, [
        _decision("receiver:p00", RT.CONTROL_ARM, answer="no residue here"),
        _decision("receiver:p00", "MAJORITY_CHECKED", answer=str(gold)),
    ]))
    assert all(r["deviation_from_control"] is None for r in rows)


def test_the_arm_label_is_carried_with_its_provenance(tmp_path: Path):
    """Nothing in the compacted tables can corroborate it, so the row says where it came from."""
    gold = RT.reference_values()["receiver:p00"]
    rows, summary = RT.build_rows(_tables(tmp_path, [
        _decision("receiver:p00", "MINORITY_UNCHECKED", answer=str(gold)),
        _decision("receiver:p00", "MAJORITY_CHECKED", answer=str(gold)),
        _decision("receiver:p00", RT.CONTROL_ARM, answer=str(gold)),
    ]))
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["MINORITY_UNCHECKED"]["arm_count"] == "MINORITY"
    assert by_arm["MINORITY_UNCHECKED"]["arm_checked"] is False
    assert by_arm["MAJORITY_CHECKED"]["arm_checked"] is True
    assert by_arm[RT.CONTROL_ARM]["arm_count"] is None
    assert by_arm[RT.CONTROL_ARM]["arm_is_control"] is True
    assert all(r["arm_source"] == "packet_policy" for r in rows)
    assert summary["packet_label_checkable"]["rows_with_seen_message_ids"] == 0


def test_a_degenerate_probe_is_flagged_on_every_one_of_its_draws(tmp_path: Path):
    """``panels.panel_p7``'s variance filter, carried per row so no caller has to redo it."""
    gold = RT.reference_values()["receiver:p00"]
    other = RT.reference_values()["receiver:p01"]
    decisions = [_decision("receiver:p00", arm, answer=str(gold))
                 for arm in ("PRIVATE_TRACE", "MINORITY_CHECKED")]
    decisions += [_decision("receiver:p01", "PRIVATE_TRACE", answer=str(other)),
                  _decision("receiver:p01", "MINORITY_CHECKED", answer="0")]
    # The compacted answer_class is what P7 counts, so the fixture must carry it.
    for row in decisions:
        row["answer_class"] = f"mod{K.RECEIVER_MODULUS}:{row['answer_raw']}"
    rows, summary = RT.build_rows(_tables(tmp_path, decisions))
    flagged = {r["item_id"]: r["degenerate_probe"] for r in rows}
    assert flagged == {"receiver:p00": True, "receiver:p01": False}
    assert summary["probes"]["degenerate"] == 1
    assert summary["probes"]["degenerate_all_correct"] == 1


# --------------------------------------------------------------------------- frozen ard36


@pytest.fixture(scope="module")
def ard36_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return RT.build_rows(V2_TABLES, requests_root=RUN_ROOT / "requests",
                         cells_root=RUN_ROOT / "cells")


@FROZEN
def test_ard36_grades_all_1280_and_pins_the_headline(ard36_rows):
    rows, summary = ard36_rows
    assert summary["rows"] == 1280 and summary["graded"] == 1280
    assert summary["correct"] == 947
    assert summary["accuracy"] == pytest.approx(947 / 1280, abs=1e-6)
    assert summary["evaluator_status"] == {"correct": 947, "incorrect": 333}
    assert summary["outcome"] == {"picked-correct": 947, "picked-wrong": 323,
                                  "produced-other-wrong": 10}


@FROZEN
def test_ard36_every_recomputed_grade_agrees_with_the_persisted_event(ard36_rows):
    """The independent evaluator and the generator's interpreter class all 1,280 identically."""
    _rows, summary = ard36_rows
    check = summary["persisted_event_check"]
    assert check["events_found"] == 1280
    assert check["rows_compared"] == 1280 and check["disagree"] == 0


@FROZEN
def test_ard36_arm_means_and_the_within_program_deviation(ard36_rows):
    _rows, summary = ard36_rows
    by_arm = summary["by_arm"]
    assert {a: by_arm[a]["correct"] for a in RT.ARMS} == {
        "PRIVATE_TRACE": 171, "MINORITY_UNCHECKED": 189, "MAJORITY_UNCHECKED": 200,
        "MINORITY_CHECKED": 191, "MAJORITY_CHECKED": 196}
    assert all(by_arm[a]["n"] == 256 for a in RT.ARMS)
    assert by_arm[RT.CONTROL_ARM]["mean_deviation_from_control"] == 0.0
    # Every social arm beats its own matched control; the budget gap is far larger than any
    # of them, which is why the deviation is taken within budget.
    assert all(by_arm[a]["mean_deviation_from_control"] > 0 for a in RT.ARMS
               if a != RT.CONTROL_ARM)
    assert by_arm[RT.CONTROL_ARM]["by_budget"]["1024"]["accuracy"] == pytest.approx(0.5)
    assert by_arm[RT.CONTROL_ARM]["by_budget"]["4096"]["correct"] == 107


@FROZEN
def test_ard36_caveat_31_of_64_probes_are_degenerate_and_all_of_them_correct(ard36_rows):
    _rows, summary = ard36_rows
    probes = summary["probes"]
    assert probes["n"] == 64 and probes["draws_per_probe"] == {20: 64}
    assert probes["degenerate"] == 31 and probes["non_degenerate"] == 33
    assert probes["degenerate_all_correct"] == 31 and probes["degenerate_all_wrong"] == 0


@FROZEN
def test_ard36_caveat_the_arm_labels_cannot_be_checked_against_packet_content(ard36_rows):
    _rows, summary = ard36_rows
    checkable = summary["packet_label_checkable"]
    assert checkable["rows_with_packet_tokens"] == 0
    assert checkable["rows_with_seen_message_ids"] == 0


@FROZEN
def test_ard36_margin_recovered_on_all_but_the_one_prefix_pair_probe(ard36_rows):
    rows, summary = ard36_rows
    margin = summary["answer_logit_margin"]
    assert margin["recovered"] == 1260 and margin["missing"] == 20
    assert margin["status"] == {R.MARGIN_OK: 1260, RT.MARGIN_PREFIX_PAIR: 20}
    assert margin["top_logprobs_realized"] == {0: 20, 20: 1260}
    unresolved = {r["item_id"] for r in rows if r["answer_logit_margin"] is None}
    assert unresolved == {"receiver:p29"}
    assert R.build_packet(R.program_for("receiver:p29"),
                          RT.CONTROL_ARM).candidate_answers == (15, 1)


@FROZEN
def test_ard36_the_runtime_never_recorded_a_margin_that_the_store_still_holds(ard36_rows):
    """The gap this module closes: k=20 was requested, kept, and then never read."""
    rows, _summary = ard36_rows
    persisted = RT._persisted_outcomes(RUN_ROOT / "cells")
    assert persisted, "the receiver cells are not mounted"
    assert {str(e["answer_logit_margin_status"]) for e in persisted.values()} == {
        R.MARGIN_UNAVAILABLE}
    assert sum(1 for r in rows if r["answer_logit_margin"] is not None) == 1260


@FROZEN
def test_ard36_receiver_decisions_carry_no_label_row_at_all(ard36_rows):
    """"evaluator_status is null" is really "there is no row": the join finds nothing."""
    import pyarrow.parquet as pq

    rows, _summary = ard36_rows
    labels = pq.read_table(V2_TABLES / "labels.parquet", columns=["decision_id"])
    labelled = set(labels.column("decision_id").to_pylist())
    assert not labelled.intersection({r["decision_id"] for r in rows})


@FROZEN
def test_ard36_the_written_table_is_separate_and_moves_no_main_panel_denominator():
    assert RT.OUTPUT_TABLE == "receiver_outcomes.parquet"
    # ``analysis.secondary_analysis`` keys the receiver family on the table named "receiver";
    # writing that name would flip it from "unexecuted" to "descriptive" as a side effect.
    assert RT.OUTPUT_TABLE != "receiver.parquet"
    assert not (V2_TABLES / "receiver.parquet").exists()


@FROZEN
def test_ard36_the_written_table_round_trips(tmp_path: Path, ard36_rows):
    import pyarrow.parquet as pq

    rows, _summary = ard36_rows
    _write_parquet(tmp_path / RT.OUTPUT_TABLE, rows)
    back = pq.read_table(tmp_path / RT.OUTPUT_TABLE).to_pylist()
    assert len(back) == 1280
    assert {r["decision_id"] for r in back} == {r["decision_id"] for r in rows}
    required = {"decision_id", "item_id", "arm", "budget_tokens", "evaluator_status",
                "is_correct", "answer_class", "deviation_from_control", "confidence",
                "thinking_tokens", "final_tokens", "latency_s", "answer_logit_margin"}
    assert required.issubset(set(back[0]))


@FROZEN
def test_ard36_cli_writes_the_table(tmp_path: Path, capsys):
    code = RT.main(["--tables", str(V2_TABLES), "--out", str(tmp_path),
                    "--no-margin", "--no-event-check"])
    assert code == 0
    assert (tmp_path / RT.OUTPUT_TABLE).exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary["rows"] == 1280 and summary["correct"] == 947
    # Without the store every margin is missing -- and missing under the *runtime* status,
    # because that branch really is "the record carried nothing".
    assert summary["answer_logit_margin"]["recovered"] == 0
