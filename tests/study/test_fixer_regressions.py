"""Regression fixtures for the fixer pass over the two study-v4 reviews.

spec-must review: P0-A (CEN_FLAT at N=9 must honour the 32,768-token envelope; Table E
never clips), P1-A (a >4,300-digit integer literal is ``NOT_JSON``, never a crash), P1-B
(the hub is re-prompted with the typed error object of its last invalid action, audit
A-6), P1-C (DEC reports ``CALL_CAP`` when the 64-call cap fixed ``r_max``); ops review:
P1-1 (``SUSPENDED.json`` is terminal for the runner and the chunk driver), P1-2 (EVAL_BCB
cells hold 10 items); plus the trivial P2 ledger guard (``keep_final=False`` only after
``release_final``).  Each test fails on the pre-fix code path.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from agents_scaling.study import cells as C
from agents_scaling.study import types as T
from agents_scaling.study.parse import candidate as pc
from agents_scaling.study.parse import coordinator as co
from agents_scaling.study.policies import cen_flat as CF
from agents_scaling.study.policies.dec import dec_round_cap
from agents_scaling.study.prompts import render as R
from agents_scaling.study.resources import envelopes as E
from agents_scaling.study.resources.broker import LedgerError
from agents_scaling.study.runner import CellPaths
from tests.study import test_wp4_broker as TB
from tests.study import wp4_stubs as S
from tests.study import wp5_support as W
from tests.study.fake_vllm import FakeTokenizer
from tests.study.test_wp1_render import HLE
from tests.study.test_wp4_policies import factory, hub_delegates_until_final, prompt_of, run
from tests.study.test_wp5_cells import tasks  # noqa: F401  (fixture)
from tests.study.test_wp5_runner import world  # noqa: F401  (fixture)
from tests.study.test_wp5_slurm import _load

# ----------------------------------------------------------------------------- P0-A: envelopes


def test_table_e_bounds_n9_hub_and_refuses_infeasible_caps_P0A(study_config):
    wrappers = E.measure_wrappers(FakeTokenizer())
    caps = study_config.caps
    table = E.TableE(caps, wrappers)
    for n in E.ENVELOPE_NS:
        assert table.hub_prompt_max(n, cycle0=False) <= T.PROMPT_TOKENS_CAP
        assert table.dec_revision_prompt_max(n) <= T.PROMPT_TOKENS_CAP
    assert table.hub_returned_max(9) == caps.hub_returned_results_tokens == 16384
    assert table.hub_returned_max(5) == 4 * caps.subtask_result_tokens == 16384
    assert table.hub_returned_max(2) == caps.subtask_result_tokens
    d = table.to_dict()
    assert d["hub_returned_max"]["9"] == 16384 and d["hub_action_error_tokens"] == caps.hub_action_error_tokens == 512
    assert all(v <= T.PROMPT_TOKENS_CAP for v in d["hub_prompt_max"]["9"].values())
    # the pre-fix N=9 block (8 x 4,096) does not fit: the table must raise, never clip to the cap
    fat = dataclasses.replace(caps, hub_returned_results_tokens=8 * caps.subtask_result_tokens)
    with pytest.raises(E.EnvelopeError, match="hub_final@N9"):
        E.TableE(fat, wrappers).hub_prompt_max(9, cycle0=False)
    with pytest.raises(E.EnvelopeError):
        E.TableE(fat, wrappers).to_dict()
    assert E.TableE(fat, wrappers).hub_prompt_max(5, cycle0=False) <= T.PROMPT_TOKENS_CAP  # N=5 still fits
    with pytest.raises(E.EnvelopeError, match="root"):
        E.TableE(caps, wrappers, task_tokens=T.PROMPT_TOKENS_CAP).root_prompt_max()
    with pytest.raises(ValueError):
        table.hub_returned_max(0)


def test_cen_n9_returned_block_bounded_and_no_context_failure_P0A(tmp_run_root, study_config):
    task = S.make_task()
    caps = study_config.caps
    big = " ".join(f"w{i}" for i in range(5000))  # 5,000 whitespace tokens: above every per-result cap
    subtask_re = re.compile(r'"subtask_id"\s*:\s*"([^"]+)"')

    def responder(info):
        if info["mode"] != "worker":
            return None
        match = subtask_re.search(info["messages"][0]["content"])
        return json.dumps(
            {
                "subtask_id": match.group(1),
                "contract": "c",
                "status": "complete",
                "result": big,
                "assumptions": [],
                "evidence_handles": ["task"],
                "confidence": 0.5,
            }
        )

    cell = S.make_cell(T.Method.CEN_FLAT, N=9, framing=T.Framing.NATIVE)
    with factory(tmp_run_root, study_config, hub_mode=hub_delegates_until_final, delegate_count=8, responder=responder) as f:
        ctx, res = run(f, task, cell, 2000 * f.root_reservation(task), call_cap=30)  # 3 full cycles, then the final
        ep = res.episode
        delegate_cycles = [c for c in ep["cycles"] if c["action"] == "delegate"]
        assert len(delegate_cycles) == 3 and all(c["workers"] == list(range(1, 9)) for c in delegate_cycles)
        per_cap = CF.CenFlatPolicy.returned_result_cap(caps, 8)
        assert per_cap == caps.hub_returned_results_tokens // 8 == 2048 < caps.subtask_result_tokens
        assert all(c["returned_result_cap"] == per_cap and c["returned_tokens"] <= caps.hub_returned_results_tokens for c in delegate_cycles)
        returned = [p for p in ep["packets"] if p["kind"] == "subtask_result"]
        assert len(returned) == 24 and all(p["recipient_tokens"] <= per_cap and p["truncated"].get("result") for p in returned)
        hub_prompts = [c for c in ctx.calls if c.role == "hub"]
        assert len(hub_prompts) == 4 and all(c.prompt_tokens <= T.PROMPT_TOKENS_CAP and c.context_failure is None for c in hub_prompts)
        assert ep["final_context_failure"] is None and ep["native_final"]["valid"] is True
        assert ep["counters"]["context_failures_by_role"] in ({}, {"hub": 0, "worker": 0}) or not any(ep["counters"]["context_failures_by_role"].values())
        assert ep["envelopes"]["hub_prompt_max"]["9"]["later"] <= T.PROMPT_TOKENS_CAP
        assert ep["stop_reason"] == "CALL_CAP" and ep["ledger"]["calls_admitted"] == 28
    # a four-assignment cycle keeps the full per-result cap
    assert CF.CenFlatPolicy.returned_result_cap(caps, 4) == caps.subtask_result_tokens
    with pytest.raises(ValueError):
        CF.CenFlatPolicy.returned_result_cap(caps, 0)


# ----------------------------------------------------------------------------- P1-A: parsers


@pytest.mark.parametrize("digits", [4301, 5001])
def test_huge_integer_literal_is_not_json_never_a_crash_P1A(digits):
    huge = "1" + "0" * digits
    content = (
        '{"approach":"a","evidence":[],"alternatives_considered":[],"failure_checks":[],'
        '"final_answer":"x","confidence":' + huge + "}"
    )
    parsed = pc.parse_candidate(content, "stop")
    assert parsed.valid is False and parsed.failure_code == "NOT_JSON"
    with pytest.raises(pc.StrictJSONError) as exc:
        pc.load_strict_json(huge)
    assert exc.value.code == "NOT_JSON"
    # the reasoning-channel probe walks the same decoder
    assert pc.parse_candidate("no json here", "stop", reasoning="{" + huge + "}").failure_code == "NOT_JSON"
    act = co.parse_coordinator_action('{"action":"delegate","assignments":[{"worker_slot":' + huge + "}]}", 5, ["task"], "stop")
    assert isinstance(act, co.ActionError) and act.code == "INVALID" and act.reason == "NOT_JSON"
    sub = co.parse_subtask_result('{"subtask_id":"s","confidence":' + huge + "}", "stop")
    assert isinstance(sub, co.SubtaskError) and sub.code == "NOT_JSON"
    # a large-but-legal integer is still a schema failure, not a crash
    assert pc.parse_candidate(content.replace(huge, "1" + "0" * 4000), "stop").failure_code == "SCHEMA"


# ----------------------------------------------------------------------------- P1-B: hub re-prompt


def test_action_error_field_renders_and_is_bounded_P1B():
    count = FakeTokenizer()
    counter = lambda text: len(count.encode(text))  # noqa: E731
    detail = " ".join(f"d{i}" for i in range(1000))
    obj = CF.bounded_action_error(3, "DUP_SLOT", detail, counter, 512)
    assert set(obj) == {"cycle", "code", "detail"} and obj["cycle"] == 3 and obj["code"] == "DUP_SLOT"
    assert detail.startswith(obj["detail"]) and 0 < len(obj["detail"]) < len(detail)
    assert counter(R.action_error_text(obj)) <= 512 - CF.ACTION_ERROR_FIELD_OVERHEAD
    short = CF.bounded_action_error(0, "INVALID", "bad", counter, 512)
    assert short["detail"] == "bad"
    with pytest.raises(T.ProtocolError):
        CF.bounded_action_error(0, "INVALID", "x", counter, CF.ACTION_ERROR_FIELD_OVERHEAD)
    plan = T.DelegateAction((T.Assignment(1, "s1", "q1", ("task",), "text", "c1"),))
    results = [T.SubtaskResult("s1", "c1", "complete", "r1", (), (), 0.8)]
    plain = R.render_hub(HLE, 5, 4, plan, results, 2)
    with_error = R.render_hub(HLE, 5, 4, plan, results, 2, action_error=obj)
    assert R.ACTION_ERROR_LABEL not in plain.content and plain.anchors["action_error"] is False
    expected = "--- last_action_error ---\n" + R.action_error_text(obj) + "\n--- public_observations ---\n"
    assert expected in with_error.content and with_error.anchors["action_error"] is True
    assert with_error.content.index("[returned_result 1]") < with_error.content.index("--- last_action_error ---")
    span = with_error.anchors["spans"]["last_action_error"]
    assert json.loads(with_error.content.encode("utf-8")[span[0]:span[1]].decode("utf-8")) == obj
    # the error may accompany a cycle-0 state (re-prompt after a cycle-0 error) and the final instruction
    assert R.ACTION_ERROR_LABEL in R.render_hub(HLE, 5, 4, None, [], 0, action_error=obj).content
    assert R.ACTION_ERROR_LABEL in R.render_hub(HLE, 5, 4, plan, results, 8, final=True, action_error=obj).content
    for bad in ({"cycle": 1, "code": "X"}, {"cycle": -1, "code": "X", "detail": ""}, {"cycle": 1, "code": "", "detail": ""}, {"cycle": True, "code": "X", "detail": ""}):
        with pytest.raises(R.RenderError):
            R.action_error_text(bad)


def test_hub_is_reprompted_with_its_last_action_error_P1B(tmp_run_root, study_config):
    task = S.make_task()
    calls: list[str] = []

    def assignment(slot):
        return {"worker_slot": slot, "subtask_id": f"s{slot}", "question": "q", "source_handles": ["task"], "required_output_type": "text", "return_contract": "c"}

    def responder(info):
        if info["mode"] != "hub":
            return None
        n = len(calls)
        calls.append(info["key"])
        if n == 0:
            return json.dumps({"action": "delegate", "assignments": [assignment(1)]})  # valid
        if n == 1:
            return json.dumps({"action": "delegate", "assignments": [assignment(1), assignment(1)]})  # DUP_SLOT
        if n == 2:
            return json.dumps({"action": "plan", "steps": ["x"]})  # INVALID
        if n == 3:
            return json.dumps({"action": "delegate", "assignments": [assignment(2)]})  # valid again
        return None  # server default: final

    cell = S.make_cell(T.Method.CEN_FLAT, N=5, framing=T.Framing.NATIVE)
    with factory(tmp_run_root, study_config, hub_mode="final", responder=responder) as f:
        ctx, res = run(f, task, cell, 2000 * f.root_reservation(task))
        hubs = [prompt_of(ctx, c) for c in ctx.calls if c.role == "hub"]
        assert len(hubs) == 5
        assert R.ACTION_ERROR_LABEL not in hubs[0] and R.ACTION_ERROR_LABEL not in hubs[1]
        assert '--- last_action_error ---\n{"cycle":1,"code":"DUP_SLOT","detail":' in hubs[2]
        assert '--- last_action_error ---\n{"cycle":2,"code":"INVALID","detail":' in hubs[3] and "DUP_SLOT" not in hubs[3]
        assert R.ACTION_ERROR_LABEL not in hubs[4]  # dropped after the next valid action
        # the last valid state (plan + its returned result) stays alongside the error
        assert all("--- returned_results: 1 ---" in h for h in hubs[1:4])
        ep = res.episode
        assert [e["code"] for e in ep["hub_action_errors"]] == ["DUP_SLOT", "INVALID"]
        assert ep["counters"]["hub_error_reprompts"] == 2 and ep["counters"]["hub_calls"] == 5 and ep["counters"]["hub_action_errors"] == 2
        assert [c["seed_key"][6] for c in ep["calls"] if c["role"] == "hub"] == [0, 1, 2, 3, 4]  # cycles still consumed (T7)
        assert ep["stop_reason"] == "VOLUNTARY_FINISH" and ep["native_final"]["valid"] is True
        assert ep["counters"]["realized_participation"] == 2


# ----------------------------------------------------------------------------- P1-C: DEC stop reason


def test_dec_n9_reports_call_cap_when_it_binds_P1C(tmp_run_root, study_config):
    assert dec_round_cap(9, 64, 8) == (6, "call_cap") and dec_round_cap(5, 64, 8) == (8, "dec_max_rounds")
    task = S.make_task()
    cell = S.make_cell(T.Method.DEC, N=9, framing=T.Framing.NATIVE)
    with factory(tmp_run_root, study_config) as f:
        ctx, res = run(f, task, cell, 2000 * f.root_reservation(task))
        ep = res.episode
        assert ep["counters"]["rounds_completed"] == 6 and ep["counters"]["r_max_binding"] == "call_cap"
        assert ep["stop_reason"] == "CALL_CAP" and ep["ledger"]["calls_admitted"] == 63


# ----------------------------------------------------------------------------- ops P1-1: SUSPENDED is terminal


def test_suspended_cell_is_terminal_for_the_runner_P1_1(world):
    rr, srv, h, tasks_ = world["run_root"], world["server"], world["harness"], world["tasks"]
    cell = W.make_cell(T.Method.BANK, [t.source_id for t in tasks_], module="F", N=1, B=0, parallel_items=2, max_inflight=10)
    paths = CellPaths.of(rr, cell.cell_id)
    paths.cell_dir.mkdir(parents=True, exist_ok=True)
    paths.suspended.write_text(json.dumps({"cell_id": cell.cell_id, "error": "ProtocolError: earlier defect"}))
    assert h.run(cell) == T.EXIT_SUSPENDED
    assert srv.chat_count == 0 and not paths.meta.exists() and not paths.items.exists()
    events = [json.loads(line)["event"] for line in paths.events.read_text().splitlines()]
    assert events == ["already_suspended"]
    assert h.run(cell) == T.EXIT_SUSPENDED and srv.chat_count == 0  # every re-queue is a no-op
    paths.suspended.unlink()  # the operator's explicit reset after the fix
    assert h.run(cell) == T.EXIT_DONE and paths.meta.exists() and srv.chat_count == 40


def test_suspended_cell_is_terminal_for_the_driver_P1_1(tmp_path: Path, capsys):
    lc = _load("study_launch_chunked")
    cells = [{"cell_id": "F.BANK.32B.N1.B0.F00.e0.s000"}, {"cell_id": "A.DEC.32B.N5.B4.Fnat.e0.s000"}, {"cell_id": "A.CEN_FLAT.32B.N5.B4.Fnat.e0.s000"}]
    done, suspended, pending = (tmp_path / "cells" / c["cell_id"] for c in cells)
    for d in (done, suspended, pending):
        d.mkdir(parents=True)
    (done / "meta.json").write_text("{}")
    (suspended / "SUSPENDED.json").write_text("{}")
    assert lc._chunk_complete(tmp_path, 0, 2, cells) is False  # one cell still pending
    assert lc._chunk_complete(tmp_path, 0, 1, cells) is True  # finished + suspended = terminal
    err = capsys.readouterr().err
    assert "SUSPENDED" in err and cells[1]["cell_id"] in err and cells[0]["cell_id"] not in err


# ----------------------------------------------------------------------------- ops P1-2: EVAL_BCB sizing


def test_eval_bcb_cells_hold_ten_items_P1_2(study_config, tasks):  # noqa: F811
    assert C.EVAL_BCB_ITEMS_PER_CELL == 10
    seal = "ab" * 32
    main = [t.source_id for t in tasks if t.split == "main"]
    sealed = main[:30] + main[200:230]
    ev = C.build_cells(study_config, tasks, "1-eval", "eval", seal=seal, sealed_items=sealed)
    assert [len(c.items) for c in ev] == [10, 10, 10] and all(c.kind is T.CellKind.EVAL_BCB and c.max_inflight == 1 for c in ev)
    assert len(C.build_cells(study_config, tasks, "1-eval", "eval", seal=seal, sealed_items=sealed, eval_bcb_items_per_cell=200)) == 1
    assert C.build_parser().parse_args(["--run-id", "study_v4", "--tier", "1-eval", "--lane", "eval", "--out", "x.json"]).eval_bcb_items_per_cell == 10


# ----------------------------------------------------------------------------- P2: ledger keep_final guard


def test_keep_final_false_requires_release_first_P2():
    led = TB.ledger(10 * TB.R, solver_call_cap=4)
    led.set_final_reserve(TB.R, 1)
    with pytest.raises(LedgerError, match="release_final"):
        led.try_reserve([TB.R], keep_final=False, owner="final")
    with pytest.raises(LedgerError, match="release_final"):
        led.fits([TB.R], keep_final=False)
    res = led.try_reserve([TB.R])
    led.debit(res, [TB.R])
    assert led.release_final() == TB.R
    fin = led.try_reserve([TB.R], keep_final=False, owner="final")
    assert fin is not None
    led.debit(fin, [TB.R])
