"""WP1 — prompts/render.py: byte-exact framing cells, state array, truthfulness, anchors.

Golden files live in ``tests/study/fixtures/wp1_render/`` and were generated from the frozen
templates; any template or rule change must be a deliberate re-freeze (§5.5).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents_scaling.study import prompts
from agents_scaling.study.prompts import render as R
from agents_scaling.study.types import (
    SENTINEL,
    SENTINEL_JSON,
    Assignment,
    Candidate,
    CandidateRecord,
    DelegateAction,
    Domain,
    Evidence,
    FinalAction,
    Framing,
    Packet,
    PublicTask,
    SubtaskResult,
)

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "wp1_render"

HLE = PublicTask("hle:golden", Domain.HLE, "dev", "Which option is prime?\n\nAnswer Choices:\nA. 4\nB. 7", "multipleChoice", "Gold", 0, 14)
BCB = PublicTask(
    "bcb:golden", Domain.BCB, "dev",
    "Return the sum of a list.\nYou should write self-contained code starting with:\n```\ndef task_func(xs):\n```",
    "code", "bcb", 0, 20, entry_point="task_func",
)
CAND = Candidate("count divisors", (Evidence("7 has two divisors", "definition", "low"),), ("4",), ("checked 7 mod 2..6",), "B", 0.9)
PK = Packet("pk-1", 3, "0" * 64, {"final_answer": "B"}, (), {}, False, 12,
            '{"sender_slot":3,"candidate_sha256":"0000","final_answer":"B","partial":false}', False)
PK_UNAVAILABLE = Packet("pk-2", 1, "1" * 64, {}, (), {}, False, 4, '{"sender_slot":1,"unavailable":true}', True)
ARM_NAMES = ("S_FRESH", "S_HISTORY", "IND_VOTE", "DEC", "CEN_FLAT", "R_CORRECT", "F00", "F11", "JUDGE_BEST")


def golden(name: str) -> str:
    return (GOLDEN / name).read_bytes().decode("utf-8")


def span_text(rendered: R.Rendered, span) -> str:
    start, end = span
    return rendered.content.encode("utf-8")[start:end].decode("utf-8")


# ----------------------------------------------------------------------------- roots


@pytest.mark.parametrize("cell", ["00", "01", "10", "11"])
def test_root_framing_cells_golden(cell):
    rendered = R.render_root(HLE, Framing(cell))
    assert rendered.messages == [{"role": "user", "content": golden(f"root_hle_{cell}.txt")}]
    assert len(rendered.messages) == 1 and rendered.messages[0]["role"] == "user"


def test_root_bcb_11_golden_uses_code_selector():
    rendered = R.render_root(BCB, Framing.F11)
    assert rendered.content == golden("root_bcb_11.txt")
    clauses = prompts.load_framing_clauses()
    assert clauses["CODE_SELECTOR"] in rendered.content
    assert clauses["HLE_CHOICE_SELECTOR"] not in rendered.content
    assert "AST" in rendered.content


def test_root_clause_placement_and_empty_line_rule():
    clauses = prompts.load_framing_clauses()
    team, vote, sel = clauses["TEAM_1"], clauses["VOTE_1"], clauses["HLE_CHOICE_SELECTOR"]
    head = prompts.load_template("independent_root").split("\n\n")[0]
    tail = "\n\nYour confidence is the probability"
    c00 = R.render_root(HLE, Framing.F00).content
    c01 = R.render_root(HLE, Framing.F01).content
    c10 = R.render_root(HLE, Framing.F10).content
    c11 = R.render_root(HLE, Framing.F11).content
    # frozen rule: each empty placeholder line is removed with its own newline, so the blank line
    # before the block and the blank line after it both survive (head + "\n\n\n" + ...)
    assert c00.startswith(head + "\n\n\n" + tail.lstrip("\n"))
    assert not c00.startswith(head + "\n\n\n\n")
    assert c10.startswith(head + "\n\n" + team + tail)
    assert c01.startswith(head + "\n\n" + vote + "\n" + sel + tail)
    assert c11.startswith(head + "\n\n" + team + "\n" + vote + "\n" + sel + tail)
    # everything after the clause block is identical in all four cells (§5.5)
    suffixes = {c.split("Your confidence is", 1)[1] for c in (c00, c01, c10, c11)}
    assert len(suffixes) == 1
    assert team not in c00 and vote not in c00 and team not in c01 and vote not in c10
    for content in (c00, c01, c10, c11):
        assert "Task: " + HLE.task_text + "\nOutput contract: " + R.CANDIDATE_SCHEMA_LINE + "\n" in content
        assert content.endswith("\n")
        assert not any(arm in content for arm in ARM_NAMES)


def test_root_anchors_are_byte_offsets():
    for cell in ("00", "11"):
        rendered = R.render_root(HLE, Framing(cell))
        assert span_text(rendered, rendered.anchors["spans"]["task"]) == HLE.task_text
        assert span_text(rendered, rendered.anchors["spans"]["output_contract"]) == R.CANDIDATE_SCHEMA_LINE
        assert rendered.anchors["task_only_anchor"] == rendered.anchors["state_anchor"] == rendered.anchors["spans"]["task"][1]
        assert rendered.anchors["framing"] == cell
    r11 = R.render_root(HLE, Framing.F11)
    assert span_text(r11, r11.anchors["spans"]["team_clause"]) == prompts.load_framing_clauses()["TEAM_1"]
    r00 = R.render_root(HLE, Framing.F00)
    assert r00.anchors["spans"]["team_clause"][0] == r00.anchors["spans"]["team_clause"][1]


def test_root_unicode_task_offsets():
    task = PublicTask("hle:u", Domain.HLE, "main", "数学: what is π ≈ ? 🚀", "exactMatch", "Revision", 3, 9)
    rendered = R.render_root(task, Framing.F01)
    assert span_text(rendered, rendered.anchors["spans"]["task"]) == task.task_text


def test_root_native_requires_clause_and_matches_dec_root():
    with pytest.raises(R.RenderError):
        R.render_root(HLE, Framing.NATIVE)
    with pytest.raises(R.RenderError):
        R.render_root(HLE, Framing.F00, native_clause="extra")
    native = R.render_root(HLE, Framing.NATIVE, R.DEC_ROOT_TRUTHFUL_CLAUSE)
    dec = R.render_dec_root(HLE, 5)
    assert native.content == dec.content == golden("dec_root_hle.txt")
    assert dec.anchors["framing"] == "nat"
    assert span_text(dec, dec.anchors["spans"]["task"]) == HLE.task_text
    # no numeric team size in any root (§5.5) and the same bytes for every N >= 2
    assert R.render_dec_root(HLE, 9).content == dec.content
    assert " 5 " not in dec.content and "five" not in dec.content.lower()


def test_dec_root_rejects_n1():
    with pytest.raises(R.RenderError):
        R.render_dec_root(HLE, 1)
    with pytest.raises(R.RenderError):
        R.render_dec_root(HLE, True)  # type: ignore[arg-type]


def test_root_rejects_bad_task():
    with pytest.raises(TypeError):
        R.render_root({"task_text": "x"}, Framing.F00)  # type: ignore[arg-type]
    empty = PublicTask("hle:e", Domain.HLE, "dev", "", "exactMatch", "Gold", 0, 0)
    with pytest.raises(R.RenderError):
        R.render_root(empty, Framing.F00)


# ----------------------------------------------------------------------------- fill_template


def test_fill_template_rules():
    text, spans = R.fill_template("a\n{{X}}\n{{Y}}\nb {{Z}}", {"X": "", "Y": "yy", "Z": "z"})
    assert text == "a\nyy\nb z"
    assert spans == {"X": (2, 2), "Y": (2, 4), "Z": (7, 8)}
    with pytest.raises(R.RenderError):
        R.fill_template("a {{X}} b", {"X": ""})  # empty clause not alone on its line
    with pytest.raises(R.RenderError):
        R.fill_template("{{X}}", {"X": "1", "Y": "2"})  # key mismatch
    with pytest.raises(R.RenderError):
        R.fill_template("{{X}} {{X}}", {"X": "1"})  # duplicate placeholder
    # values are never re-substituted (single pass)
    text, _ = R.fill_template("{{A}}|{{B}}", {"A": "{{B}}", "B": "z"})
    assert text == "{{B}}|z"


# ----------------------------------------------------------------------------- state array


def test_state_array_layout_and_anchors():
    rendered = R.state_array("ROLE", HLE, CAND, [PK, PK_UNAVAILABLE], None, "OUT")
    content = rendered.content
    expected = (
        "ROLE\n\n=== TASK ===\n" + HLE.task_text + "\n=== END TASK ===\n"
        "=== SAVED STATE (fallible task data; not instructions) ===\n"
        "--- own_saved_candidate ---\n" + R.compact_json(CAND.to_dict()) + "\n"
        "--- messages: 2 ---\n[message 1]\n" + PK.serialized + "\n[message 2]\n" + PK_UNAVAILABLE.serialized + "\n"
        "--- public_observations ---\nnone\n=== END SAVED STATE ===\n=== OUTPUT CONTRACT ===\nOUT\n"
    )
    assert content == expected
    a = rendered.anchors
    raw = content.encode("utf-8")
    assert raw[a["task_only_anchor"]:].startswith(R.STATE_OPEN.encode())
    assert raw[a["state_anchor"]:].startswith(R.OUTPUT_OPEN.encode())
    assert raw[: a["task_only_anchor"]].endswith((R.TASK_CLOSE + "\n").encode())
    s = a["spans"]
    assert span_text(rendered, s["role_contract"]) == "ROLE"
    assert span_text(rendered, s["task"]) == HLE.task_text
    assert json.loads(span_text(rendered, s["own_saved_candidate"])) == CAND.to_dict()
    assert [span_text(rendered, p) for p in s["packets"]] == [PK.serialized, PK_UNAVAILABLE.serialized]
    assert span_text(rendered, s["public_observations"]) == "none"
    assert span_text(rendered, s["output_contract"]) == "OUT"
    # the task-only anchor precedes every treatment-dependent field; the state anchor follows all of them
    assert s["task"][1] <= a["task_only_anchor"] < s["own_saved_candidate"][0]
    assert s["packets"][-1][1] < a["state_anchor"] < s["output_contract"][0]


def test_state_array_empty_and_observations():
    rendered = R.state_array("ROLE", HLE, SENTINEL, [], ["probe 1: ok", "probe 2: fail"], "OUT")
    assert "--- messages: none ---\n" in rendered.content
    assert rendered.anchors["spans"]["packets"] == []
    assert span_text(rendered, rendered.anchors["spans"]["public_observations"]) == "probe 1: ok\nprobe 2: fail"
    assert "--- own_saved_candidate ---\n" + SENTINEL_JSON + "\n" in rendered.content
    with pytest.raises(R.RenderError):
        R.state_array("ROLE", HLE, CAND, [], [""], "OUT")
    with pytest.raises(R.RenderError):
        R.state_array("ROLE", HLE, None, [], None, "OUT")
    with pytest.raises(R.RenderError):
        R.state_array("", HLE, CAND, [], None, "OUT")
    with pytest.raises(R.RenderError):
        R.state_array("ROLE", HLE, CAND, [], None, "")


def test_saved_object_text_rules():
    assert R.saved_object_text(CAND) == R.compact_json(CAND.to_dict())
    assert R.saved_object_text(SENTINEL) == SENTINEL_JSON
    assert R.saved_object_text(SENTINEL_JSON) == SENTINEL_JSON
    assert R.saved_object_text(None) == "none"
    valid = CandidateRecord("c", "r", 0, "root", True, None, CAND, "s", "raw")
    invalid = CandidateRecord("c", "r", 0, "root", False, "INVALID_JSON", None, "s", "raw")
    broken = CandidateRecord("c", "r", 0, "root", True, None, None, "s", "raw")
    assert R.saved_object_text(valid) == R.compact_json(CAND.to_dict())
    assert R.saved_object_text(invalid) == SENTINEL_JSON  # exact sentinel bytes (§3.5)
    with pytest.raises(R.RenderError):
        R.saved_object_text(broken)
    with pytest.raises(R.RenderError):
        R.saved_object_text('{"final_answer": "raw model text"}')
    with pytest.raises(R.RenderError):
        R.saved_object_text({"status": "unavailable"})
    with pytest.raises(R.RenderError):
        R.saved_object_text(3.5)
    assert R.saved_object_text(FinalAction(CAND)) == R.compact_json({"action": "final", "candidate": CAND.to_dict()})


def test_packet_text_rules():
    assert R.packet_text(PK) == PK.serialized
    result = SubtaskResult("s1", "contract", "partial", "res", ("a",), (), 0.4)
    assert json.loads(R.packet_text(result)) == result.to_dict()
    with pytest.raises(R.RenderError):
        R.packet_text("loose text")
    with pytest.raises(R.RenderError):
        R.packet_text(Packet("p", 1, "h", {}, (), {}, False, 0, "", True))


# ----------------------------------------------------------------------------- revisions


def test_dec_revision_golden_and_truthful_n1():
    n1 = R.render_dec_revision(HLE, SENTINEL, [], 1, 1)
    assert n1.content == golden("dec_revision_n1_hle.txt")
    role = span_text(n1, n1.anchors["spans"]["role_contract"]).lower()
    for word in ("peer", "other solver", "other member", "team", "{{n}}"):
        assert word not in role
    assert "only solver" in role and "will not receive any messages" in role
    assert "--- messages: none ---" in n1.content
    n3 = R.render_dec_revision(HLE, CAND, [PK, PK_UNAVAILABLE], 3, 2)
    assert n3.content == golden("dec_revision_n3_hle.txt")
    assert "You are one of 3 solvers. You may receive bounded messages from the other 2 solvers" in n3.content
    assert n3.anchors["N"] == 3 and n3.anchors["round"] == 2
    assert prompts.load_framing_clauses()["HLE_CHOICE_SELECTOR"] in n3.content
    assert R.revision_instruction() in n3.content


def test_dec_revision_validation():
    with pytest.raises(R.RenderError):
        R.render_dec_revision(HLE, CAND, [PK], 3, 1)  # needs N-1 = 2 packets
    with pytest.raises(R.RenderError):
        R.render_dec_revision(HLE, CAND, [], 1, 0)  # round >= 1
    with pytest.raises(R.RenderError):
        R.render_dec_revision(HLE, None, [], 1, 1)  # own required
    with pytest.raises(R.RenderError):
        R.render_dec_revision(HLE, CAND, ["not a packet"], 2, 1)


def test_dec_revision_previous_round_only_bytes():
    """Prompt bytes of round r contain exactly the packets passed (previous round), nothing else."""
    r = R.render_dec_revision(BCB, CAND, [PK_UNAVAILABLE, PK], 3, 3)
    assert [span_text(r, p) for p in r.anchors["spans"]["packets"]] == [PK_UNAVAILABLE.serialized, PK.serialized]
    assert r.content.count("[message ") == 2


def test_s_history_focal_consumer():
    hist = R.render_s_history(HLE, SENTINEL)
    assert hist.content.startswith(R.revision_instruction() + "\n\n=== TASK ===\n")
    assert "--- messages: none ---" in hist.content and SENTINEL_JSON in hist.content
    focal = R.render_focal_revision(HLE, CAND, [PK] * 4)
    assert focal.content.count("[message ") == 4
    zero = R.render_focal_revision(HLE, CAND, [])
    assert "--- messages: none ---" in zero.content
    consumer = R.render_common_consumer(HLE, CAND)
    assert consumer.content.startswith(prompts.load_template("common_consumer").rstrip("\n") + "\n\n=== TASK ===\n")
    assert "--- messages: none ---" in consumer.content
    for rendered in (hist, focal, consumer):
        assert rendered.content.endswith(R.OUTPUT_PREFIX + R.CANDIDATE_SCHEMA_LINE + "\n")
        assert len(rendered.messages) == 1


# ----------------------------------------------------------------------------- hub / worker


def test_hub_cycle0_golden_and_capacity():
    hub = R.render_hub(HLE, 5, 4, None, [], 0)
    assert hub.content == golden("hub_cycle0_n5.txt")
    assert "You have 5 total role slots,\nincluding yourself, and 4 available worker slots." in hub.content
    assert "--- prior_plan ---\nnone\n--- returned_results: none ---" in hub.content
    assert hub.content.endswith(R.OUTPUT_PREFIX + R.COORDINATOR_ACTION_LINE + "\n")
    assert prompts.load_template("cen_final_instruction") not in hub.content
    n1 = R.render_hub(HLE, 1, 0, None, [], 0)
    assert "1 total role slots" in n1.content and "0 available worker slots" in n1.content


def test_hub_later_cycle_and_final():
    plan = DelegateAction((Assignment(1, "s1", "q1", (), "text", "c1"), Assignment(2, "s2", "q2", ("task",), "text", "c2")))
    results = [SubtaskResult("s1", "c1", "complete", "r1", (), (), 0.8), SubtaskResult("s2", "c2", "failed", "", (), (), None)]
    hub = R.render_hub(HLE, 5, 4, plan, results, 1)
    assert json.loads(span_text(hub, hub.anchors["spans"]["own_saved_candidate"])) == plan.to_dict()
    assert [json.loads(span_text(hub, p)) for p in hub.anchors["spans"]["packets"]] == [r.to_dict() for r in results]
    assert "[returned_result 1]" in hub.content and "[returned_result 2]" in hub.content
    final = R.render_hub(HLE, 5, 4, plan, results, 8, final=True)
    instruction = prompts.load_template("cen_final_instruction")
    assert instruction in span_text(final, final.anchors["spans"]["role_contract"])
    assert final.anchors["final"] is True and final.anchors["cycle"] == 8
    assert final.content != hub.content


def test_hub_validation():
    plan = DelegateAction((Assignment(1, "s1", "q1", (), "text", "c1"),))
    with pytest.raises(R.RenderError):
        R.render_hub(HLE, 5, 5, None, [], 0)  # workers > N_total - 1
    with pytest.raises(R.RenderError):
        R.render_hub(HLE, 5, 4, plan, [], 0)  # cycle 0 with a plan
    with pytest.raises(R.RenderError):
        R.render_hub(HLE, 5, 4, None, [], 1)  # cycle 1 without a plan
    with pytest.raises(R.RenderError):
        R.render_hub(HLE, 5, 4, plan, [], 1)  # result count mismatch
    with pytest.raises(R.RenderError):
        R.render_hub(HLE, 0, 0, None, [], 0)


def test_worker_render():
    assignment = Assignment(2, "s2", "Check the parity of 7", ("task",), "text", "state whether 7 is odd")
    forwarded = [SubtaskResult("s1", "c1", "complete", "7 > 4", (), (), 0.9)]
    worker = R.render_worker(HLE, assignment, forwarded)
    assert worker.content.startswith(prompts.load_template("central_worker").split("\n\n{{")[0] + "\n\n=== TASK ===\n")
    assert "--- assignment ---\n" + R.compact_json(assignment.to_dict()) in worker.content
    assert "--- forwarded_results: 1 ---\n[forwarded_result 1]\n" in worker.content
    assert worker.content.endswith(R.OUTPUT_PREFIX + R.SUBTASK_SCHEMA_LINE + "\n")
    assert worker.anchors["worker_slot"] == 2
    assert "majority" not in worker.content.lower() and "vote" not in worker.content.lower()
    with pytest.raises(R.RenderError):
        R.render_worker(HLE, {"worker_slot": 2}, [])  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- judges / forecast


def test_judge_best_render():
    judge = R.render_judge_best(BCB, CAND)
    assert judge.content.startswith(prompts.load_template("judge_best").split("\n\n{{")[0] + "\n\n=== TASK ===\n")
    assert "--- candidate ---\n" + R.compact_json(CAND.to_dict()) in judge.content
    assert "--- allowed_public_results: none ---" in judge.content
    assert judge.content.endswith(R.OUTPUT_PREFIX + R.JUDGE_BEST_SCHEMA_LINE + "\n")
    with pytest.raises(R.RenderError):
        R.render_judge_best(BCB, CAND.to_dict())  # type: ignore[arg-type]


def test_hle_judge_matches_official_format():
    rendered = R.render_hle_judge("Q?", "The answer is B.", "B")
    template = prompts.load_template("hle_judge")
    assert rendered.content == template.format(question="Q?", response="The answer is B.", correct_answer="B")
    assert rendered.messages[0]["role"] == "user" and len(rendered.messages) == 1
    s = rendered.anchors["spans"]
    assert span_text(rendered, s["question"]) == "Q?"
    assert span_text(rendered, s["response"]) == "The answer is B."
    assert span_text(rendered, s["correct_answer"]) == "B"
    # braces inside the values are inert (single-pass)
    tricky = R.render_hle_judge("{response}", "{correct_answer} {x}", "{question}")
    assert tricky.content == template.format(question="{response}", response="{correct_answer} {x}", correct_answer="{question}")
    with pytest.raises(R.RenderError):
        R.render_hle_judge("", "r", "a")


def test_forecast_render():
    manifest = {
        "scope": ["PERSONAL_FINAL", "TEAM_SELECTED"], "mask": {"q_child_contract": None}, "observer_role": "report_reader",
        "information_set": "PEER_EXPOSED", "selected_pool_id": "pool-1", "checkpoint_id": "Qwen3-32B@9216db57",
        "operation_id": "STOP", "remaining_allowance": 0.0,
    }
    rendered = R.render_forecast("REPORT BYTES\nline 2", manifest)
    template_head = prompts.load_template("forecast").split("\n\n{{")[0]
    assert rendered.content.startswith(template_head + "\n\n" + R.MANIFEST_PREFIX)
    assert R.OUTPUT_PREFIX + R.FORECAST_SCHEMA_LINE + "\n" + R.REPORT_OPEN + "\nREPORT BYTES\nline 2\n" + R.REPORT_CLOSE + "\n" in rendered.content
    assert span_text(rendered, rendered.anchors["spans"]["report"]) == "REPORT BYTES\nline 2"
    assert json.loads(span_text(rendered, rendered.anchors["spans"]["manifest"])[len(R.MANIFEST_PREFIX):].split("\n")[0]) == manifest
    with pytest.raises(R.RenderError):
        R.render_forecast("r", {"scope": "PERSONAL_FINAL"})
    with pytest.raises(R.RenderError):
        R.render_forecast("", manifest)


def test_selector_description_by_domain():
    clauses = prompts.load_framing_clauses()
    assert R.selector_description(Domain.HLE) == clauses["HLE_CHOICE_SELECTOR"]
    assert R.selector_description("bcb") == clauses["CODE_SELECTOR"]
    assert R.dec_revision_contract(BCB, 1).endswith(clauses["CODE_SELECTOR"])
    with pytest.raises(R.RenderError):
        R.dec_revision_contract(BCB, 0)


def test_no_arm_names_or_protected_words_in_any_prompt():
    renders = [
        R.render_root(HLE, Framing.F11).content,
        R.render_dec_root(BCB, 5).content,
        R.render_dec_revision(HLE, CAND, [PK], 2, 1).content,
        R.render_hub(HLE, 5, 4, None, [], 0).content,
        R.render_common_consumer(BCB, CAND).content,
        R.render_judge_best(HLE, CAND).content,
    ]
    for content in renders:
        for arm in ARM_NAMES:
            assert arm not in content
        assert "correct_answer" not in content and "canonical_solution" not in content


def test_candidate_schema_line_is_spec_literal():
    assert R.CANDIDATE_SCHEMA_LINE == (
        '{"approach":"brief method","evidence":[{"claim":"checkable statement","support":"basis",'
        '"uncertainty":"low"}],"alternatives_considered":[],"failure_checks":[],'
        '"final_answer":"answer or complete code","confidence":0.5}'
    )
    assert list(json.loads(R.CANDIDATE_SCHEMA_LINE)) == ["approach", "evidence", "alternatives_considered", "failure_checks", "final_answer", "confidence"]
    assert list(json.loads(R.FORECAST_SCHEMA_LINE)) == ["q_personal", "q_child_contract", "q_team_now", "q_recover", "q_preserve"]
    assert list(json.loads(R.SUBTASK_SCHEMA_LINE)) == ["subtask_id", "contract", "status", "result", "assumptions", "evidence_handles", "confidence"]
