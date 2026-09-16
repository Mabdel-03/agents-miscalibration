"""WP5: the E2b local-step checker (spec 9.3; spec 18.4 names).

Every test here is a *negative capability*: the tool must not be able to do something.  H4 is
the difference between being told and being able to find out, so a checker that leaked the
answer, recomputed the chain, or handed out a fifth check would not test that difference -- it
would test a weaker, unnamed treatment.
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.ard.graders import trace as EVAL
from agents_scaling.nextdyn import checker as CH
from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import mod97 as M
from agents_scaling.nextdyn.errors import ContractViolation
from tests.nextdyn.test_mod97 import CONFIG_DIR, SEED

DECISION = "d:testdecision"
EPISODE = "receiver|panel|cond|model|budget|rep0"
CORRECT_PACKET, WRONG_PACKET = "pk:aaaa", "pk:bbbb"


@pytest.fixture(scope="module")
def e2b_block():
    return C.load_study(CONFIG_DIR / "study.yaml").experiments.E2b


@pytest.fixture(scope="module")
def problem(e2b_block) -> M.Problem:
    return M.generate(SEED, 12, 0, spec=M.spec_from_config(e2b_block))


@pytest.fixture
def traces(problem):
    return {CORRECT_PACKET: M.correct_trace(problem), WRONG_PACKET: M.wrong_trace(problem, SEED)}


@pytest.fixture
def tool(traces, problem, e2b_block):
    return CH.CheckStepTool(traces, problem, enabled=True, max_checks=e2b_block.max_checks)


def serve(tool, pairs):
    return tool.serve(tuple(CH.CheckRequest(p, s) for p, s in pairs),
                      decision_id=DECISION, episode_id=EPISODE)


# --------------------------------------------------------------------------- spec 18.4


def test_checker_only_reads_named_public_packet_and_step(tool, problem, traces):
    """Spec 18.4.  One request names one packet and one row, and the answer is that row's own
    local equation -- so a *locally consistent downstream row of the wrong trace* is ``valid``.

    That is the honest answer and it is also the point: if the tool answered "invalid" for
    every row after the injected error, four checks would localize the error by bisection and
    the checker would be a correctness oracle instead of a local one.
    """
    wrong = traces[WRONG_PACKET]
    downstream = [row.j for row in wrong.rows
                  if row.j in problem.chain and row.j > wrong.violated_step]
    responses, _ = serve(tool, [(WRONG_PACKET, wrong.violated_step), (WRONG_PACKET, downstream[0])])
    assert [r.status for r in responses] == [CH.CheckStatus.INVALID, CH.CheckStatus.VALID]

    # and the correct packet's same rows are all valid
    ok, _ = serve(tool, [(CORRECT_PACKET, wrong.violated_step)])
    assert ok[0].status is CH.CheckStatus.VALID

    # a packet the receiver was never shown, and a step that is not a displayed operation
    bad, _ = serve(tool, [("pk:unknown", 1), (CORRECT_PACKET, 0)])
    assert [r.status for r in bad] == [CH.CheckStatus.BAD_REQUEST, CH.CheckStatus.BAD_REQUEST]


def test_checker_does_not_reveal_gold_or_recompute_final_answer(tool, problem, traces, monkeypatch):
    """Spec 18.4.  The tool object holds no final answer, never calls the evaluator, and its
    responses carry a status and nothing else."""
    def forbidden(*args, **kwargs):                     # pragma: no cover - must never run
        raise AssertionError("the checker recomputed the trace")

    monkeypatch.setattr(EVAL, "evaluate_program", forbidden)
    monkeypatch.setattr(EVAL, "check_trace", forbidden)
    responses, events = serve(tool, [(p, s) for p in (CORRECT_PACKET, WRONG_PACKET) for s in (1, 2)])

    finals = {M.correct_trace(problem).final, traces[WRONG_PACKET].final}
    blob = json.dumps([r.as_dict() for r in responses] + [e.response_json for e in events])
    for value in finals:
        assert f": {value}" not in blob
    assert set(json.loads(responses[0].as_line())) == {"packet_id", "step_id", "status"}
    assert not any(attr for attr in vars(tool) if "final" in attr or "answer" in attr)
    # the tool never names the injected error either
    assert str(traces[WRONG_PACKET].violated_step) not in json.dumps(
        [r.status.value for r in responses])


def test_disabled_checker_never_returns_validity(traces, problem, e2b_block):
    """Spec 18.4.  Every response of a disabled tool is ``disabled`` in the *same* schema, and
    a disabled tool cannot even be probed for whether a packet or a step exists -- the status
    is identical for a real row, an unknown packet and a nonsense step."""
    off = CH.CheckStepTool(traces, problem, enabled=False, max_checks=e2b_block.max_checks)
    responses, events = serve(off, [(CORRECT_PACKET, 1), ("pk:unknown", 1), (WRONG_PACKET, 999)])
    assert [r.status for r in responses] == [CH.CheckStatus.DISABLED] * 3
    assert all(set(r.as_dict()) == {"packet_id", "step_id", "status"} for r in responses)
    assert {e.access_condition for e in events} == {CH.ACCESS_DISABLED}
    assert CH.CheckStatus.VALID.value not in {e.status for e in events}


def test_at_most_four_check_requests_and_fixed_duplicate_policy(tool, e2b_block):
    """Spec 18.4.  The first four requests consume slots **including duplicates and bad
    requests**; the fifth is not served at all and is logged ``over_budget``.

    "No adaptive exception grants extra checks": a duplicate is answered again with the same
    typed response and still costs a slot, so asking twice is never a way to buy a look.
    """
    budget = e2b_block.max_checks
    assert budget == 4
    responses, events = serve(tool, [(CORRECT_PACKET, 1), (CORRECT_PACKET, 1),
                                     ("pk:unknown", 3), (CORRECT_PACKET, 0),
                                     (WRONG_PACKET, 2), (WRONG_PACKET, 3)])
    assert len(responses) == budget
    assert len(events) == 6
    assert [e.status for e in events[budget:]] == [CH.OVER_BUDGET, CH.OVER_BUDGET]
    assert [e.duplicate for e in events] == [False, True, False, False, False, False]
    assert responses[0] == responses[1]                # a duplicate gets the same typed answer
    assert [e.request_index for e in events] == list(range(6))
    assert len({e.tool_event_id for e in events}) == 6


# --------------------------------------------------------------------------- plan parsing


def test_invalid_plan_output_yields_an_empty_check_set_and_continues():
    """Spec 9.4: "Invalid plan output yields an empty check set and continues to decision
    without a repair call" -- so the parser never raises and never asks again."""
    for text in ("", None, "not json", "[]", '{"other": 1}', '{"check_requests": "x"}',
                 '{"check_requests": [1, null, {"packet_id": 2, "step_id": 3}]}'):
        assert CH.parse_plan_output(text) == ()


def test_a_well_formed_plan_keeps_requests_the_tool_will_refuse():
    """A request naming an unknown packet is *kept*: deciding that is the tool's job, and it
    costs the receiver a slot (spec 9.3).  Dropping it here would refund the slot silently."""
    plan = json.dumps({"check_requests": [{"packet_id": "pk:aaaa", "step_id": 3},
                                          {"packet_id": "pk:ghost", "step_id": 99},
                                          {"packet_id": "pk:aaaa", "step_id": True}]})
    requests = CH.parse_plan_output(plan)
    assert [(r.packet_id, r.step_id) for r in requests] == [("pk:aaaa", 3), ("pk:ghost", 99)]


# --------------------------------------------------------------------------- construction


def test_the_tool_refuses_a_packet_from_another_problem(problem, traces, e2b_block):
    other = M.generate(b"another-seed", 4, 0, spec=M.spec_from_config(e2b_block))
    with pytest.raises(ContractViolation):
        CH.CheckStepTool({CORRECT_PACKET: M.correct_trace(other)}, problem, enabled=True,
                         max_checks=e2b_block.max_checks)


def test_tool_events_carry_the_request_and_the_response_verbatim(tool):
    _, events = serve(tool, [(CORRECT_PACKET, 5)])
    assert json.loads(events[0].request_json) == {"packet_id": CORRECT_PACKET, "step_id": 5}
    assert json.loads(events[0].response_json)["status"] in {"valid", "invalid"}
    assert events[0].decision_id == DECISION and events[0].episode_id == EPISODE
    assert events[0].elapsed_s >= 0.0


def test_the_budget_comes_from_study_yaml(traces, problem, e2b_block):
    """R2: ``experiments.E2b.max_checks`` is the only declaration of the request budget."""
    tight = CH.CheckStepTool(traces, problem, enabled=True, max_checks=1)
    responses, events = serve(tight, [(CORRECT_PACKET, 1), (CORRECT_PACKET, 2)])
    assert len(responses) == 1 and events[1].status == CH.OVER_BUDGET
    assert e2b_block.max_checks == tool_budget_of(traces, problem, e2b_block)


def tool_budget_of(traces, problem, e2b_block) -> int:
    return CH.CheckStepTool(traces, problem, enabled=True,
                            max_checks=e2b_block.max_checks).max_checks
