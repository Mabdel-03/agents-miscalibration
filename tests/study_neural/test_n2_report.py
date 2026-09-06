"""N2 report compiler: cap enforcement, blind order, sentinel, spans/anchors, manifest, render."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents_scaling.study import identity
from agents_scaling.study import types as T
from agents_scaling.study.config import load_config
from agents_scaling.study.forecast import manifest as M
from agents_scaling.study.forecast import report as RP
from agents_scaling.study.forecast.report import EVIDENCE_TOKENS_CAP, MAX_NONSELECTED_PACKETS, compile_report
from agents_scaling.study.inference.tokens import render_chat_text, render_chat_token_ids
from agents_scaling.study.packets import make_counter
from agents_scaling.study.prompts import render as R
from agents_scaling.study.selection import seal as seals
from tests.study_neural import n2_support as N


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def stub():
    return N.StubTokenizer()


@pytest.fixture
def world(tmp_path: Path, cfg):
    """2 HLE (MC + exact) + 2 BCB main items; IND_VOTE / DEC / CEN_FLAT cells; sealed."""
    tasks = N.make_tasks(2, "main")
    w = N.World(tmp_path / "run", cfg, tasks)
    hle_mc, hle_exact, bcb0, bcb1 = [t.source_id for t in tasks]
    # IND_VOTE: 8 attempts, one invalid, plurality winner "B"
    ind = {
        hle_mc: (N.ind_pool("mc", ["B", "A", "B", "C", "B", "A", "B", "B"], invalid=[3]), None, {"optional_draws": 3}),
        hle_exact: (N.ind_pool("ex", ["4", "5", "4", "4", "6"]), None, None),
        bcb0: (N.ind_pool("b0", ["x = 1\n", "x = 2\n", "x = 1\n", "x = 3\n", "x = 1\n"]), None, None),
        bcb1: (N.ind_pool("b1", [""] * 5, invalid=[0, 1, 2, 3, 4]), None, None),  # no valid candidate at all
    }
    w.add(T.Method.IND_VOTE, ind)
    # DEC: 5 roots + 5 round-1 revisions; the latest slots are the pool
    dec = {}
    for sid, tag in ((hle_mc, "dm"), (hle_exact, "de"), (bcb0, "db"), (bcb1, "dz")):
        roots = [N.make_record(f"{tag}:r{s}", s, "root", N.make_candidate("A" if s % 2 else "B", tag=f"{tag}r{s}")) for s in range(5)]
        answers = ["B", "B", "A", "B", "C"] if sid != bcb1 else [None] * 5
        latest = [N.make_record(f"{tag}:l{s}", s, "round1", None if answers[s] is None else N.make_candidate(answers[s], tag=f"{tag}l{s}")) for s in range(5)]
        dec[sid] = (roots + latest, [r.candidate_id for r in latest], {"rounds_completed": 1, "r_max": 8, "r_max_binding": "dec_max_rounds"})
    w.add(T.Method.DEC, dec)
    # CEN_FLAT: the single native final (invalid on bcb1)
    cen = {
        sid: ([N.make_record(f"cen:{sid}", 0, "final", None if sid == bcb1 else N.make_candidate("B", 0.9, tag="hub"))], None,
              {"delegation_cycles": 2, "hub_calls": 3, "realized_participation": 4, "workers_assigned": 4, "hub_action_errors": 0})
        for sid in (hle_mc, hle_exact, bcb0, bcb1)
    }
    w.add(T.Method.CEN_FLAT, cen)
    w.seal(skip=[(w.cells[T.Method.DEC].cell_id, bcb0)])
    return w


def _compile(w: N.World, method: T.Method, sid: str, tokenizer, **kw):
    return compile_report(w.run_root, sid, w.cells[method].cell_id, tokenizer, seal=N.SEAL, cfg=w.cfg, **kw)


def _bytes(text: str, span) -> str:
    return text.encode("utf-8")[span[0]:span[1]].decode("utf-8")


# --------------------------------------------------------------------------- layout and spans


def test_ind_vote_report_layout_and_spans(world, stub):
    sid = world.tasks[0].source_id
    rep = _compile(world, T.Method.IND_VOTE, sid, stub)
    text = rep.text
    assert text.startswith(R.TASK_OPEN + "\n" + world.tasks[0].task_text + "\n" + R.TASK_CLOSE + "\n--- item ---\n")
    assert _bytes(text, rep.spans["task"]) == world.tasks[0].task_text
    assert rep.task_only_anchor == rep.spans["task"][1]
    assert text.encode("utf-8")[rep.task_only_anchor:].startswith(("\n" + R.TASK_CLOSE).encode())
    assert _bytes(text, rep.spans["evidence"]).startswith("--- decision_rule ---\n" + RP.DECISION_RULES[T.Method.IND_VOTE])
    assert rep.spans["evidence"][1] == len(text.encode("utf-8"))
    for key in ("item", "selected_candidate", "selected_personal_confidence", "decision_rule", "vote_metadata", "budget_metadata"):
        assert key in rep.spans
    assert json.loads(_bytes(text, rep.spans["item"]))["method"] == "IND_VOTE"
    # selected: the plurality winner "B" (5 of 7 valid) with its own confidence
    assert rep.selected_candidate is not None and rep.selected_candidate.final_answer == "B"
    assert rep.selected_personal_confidence == rep.selected_candidate.confidence and not rep.personal_confidence_missing
    assert json.loads(_bytes(text, rep.spans["selected_candidate"])) == rep.selected_candidate.to_dict()
    assert json.loads(_bytes(text, rep.spans["selected_personal_confidence"])) == {"scope": "PERSONAL_FINAL", "value": rep.selected_personal_confidence, "missing": False}
    vm = json.loads(_bytes(text, rep.spans["vote_metadata"]))
    assert vm["selector_id"] == "VOTE" and vm["pool_kind"] == "archive" and vm["pool_size"] == 8 and vm["valid_count"] == 7
    assert vm["winning_count"] == 5 and vm["tied_classes"] == 1 and vm["grouping_mode"] == "mc_letter" and vm["optional_draws"] == 3
    bm = json.loads(_bytes(text, rep.spans["budget_metadata"]))
    assert bm["B"] == 4 and bm["slack_flops"] == 900_000 and bm["calls_by_role"] == {"root": 8} and bm["stop_reason"] == "CALL_CAP"
    # packets: 4 of the 7 non-selected, exact bytes at the recorded spans
    assert rep.nonselected_total == 7 and len(rep.packets) == MAX_NONSELECTED_PACKETS and not rep.packets_clipped_last and rep.packets_dropped == 0
    assert f"--- nonselected_candidates: 4 of 7 ---\n[packet 1]\n" in text
    for span, packet in zip(rep.spans["packets"], rep.packets):
        assert _bytes(text, span) == packet.serialized
    assert rep.evidence_tokens == make_counter(stub)(_bytes(text, rep.spans["evidence"])) <= EVIDENCE_TOKENS_CAP
    assert not text.endswith("\n")
    # report id is a pure function of the sealed inputs
    again = _compile(world, T.Method.IND_VOTE, sid, N.StubTokenizer())
    assert again.report_id == rep.report_id and again.text == text and again.to_dict() == rep.to_dict()


def test_blind_order_is_hmac_and_validity_independent(world, stub):
    sid = world.tasks[0].source_id
    rep = _compile(world, T.Method.IND_VOTE, sid, stub)
    item = seals.load_item_file(world.run_root, world.cells[T.Method.IND_VOTE].cell_id, sid)
    records = {r.candidate_id: r for r in seals.candidate_records_of(item)}
    others = [r for cid, r in records.items() if cid != rep.selected_candidate_id]
    expected = sorted(others, key=lambda r: identity.blind_order_key(world.cfg.study_seed, "REPORT_PACKETS", sid, r.candidate_id))
    assert [m["candidate_id"] for m in rep.nonselected] == [r.candidate_id for r in expected[:4]]
    # the invalid attempt keeps its hashed position and renders as the typed unavailable packet
    invalid = next(r for r in others if not r.valid)
    pos = [r.candidate_id for r in expected].index(invalid.candidate_id)
    if pos < 4:
        assert rep.packets[pos].serialized == T.PACKET_UNAVAILABLE_JSON and rep.nonselected[pos]["unavailable"]
    # a different study seed permutes the order
    other = RP.blind_packet_order(b"other-seed", sid, others)
    assert {r.candidate_id for r in other} == {r.candidate_id for r in expected}
    assert RP.blind_packet_order(world.cfg.study_seed, sid, list(reversed(others))) == expected


def test_cap_enforcement_with_huge_pool(tmp_path: Path, cfg, stub):
    tasks = N.make_tasks(1, "main")[:1]
    w = N.World(tmp_path / "run", cfg, tasks)
    sid = tasks[0].source_id
    # 12 attempts, each with a ~3,000-token approach: every packet clips at 2,048 and four
    # whole packets (8,192) plus the metadata cannot fit the 8,192-token evidence span.
    w.add(T.Method.IND_VOTE, {sid: (N.ind_pool("big", ["B", "A", "C", "D", "B", "A", "C", "D", "B", "A", "C", "D"], words=3000), None, None)})
    w.seal()
    rep = _compile(w, T.Method.IND_VOTE, sid, stub)
    count = make_counter(stub)
    assert rep.nonselected_total == 11 and 1 <= len(rep.packets) <= MAX_NONSELECTED_PACKETS
    assert rep.evidence_tokens == count(_bytes(rep.text, rep.spans["evidence"])) <= EVIDENCE_TOKENS_CAP
    assert all(p.recipient_tokens <= T.PACKET_TOKENS_CAP for p in rep.packets)
    assert rep.packets_clipped_last and rep.packets_dropped == 0
    last = rep.nonselected[-1]
    assert last["clipped_to_fit"] and last["packet_cap_used"] < T.PACKET_TOKENS_CAP and last["recipient_tokens"] <= last["packet_cap_used"]
    assert all(not m["clipped_to_fit"] and m["packet_cap_used"] == T.PACKET_TOKENS_CAP for m in rep.nonselected[:-1])
    # the clipped packet is as large as the rule allows (whitespace tokens: the clipper lands within a few tokens)
    assert rep.evidence_tokens >= EVIDENCE_TOKENS_CAP - 16
    # deterministic
    again = _compile(w, T.Method.IND_VOTE, sid, N.StubTokenizer())
    assert again.text == rep.text and again.nonselected == rep.nonselected
    # a much tighter cap admits at most one (clipped) packet, never an over-cap render
    small = _compile(w, T.Method.IND_VOTE, sid, stub, evidence_cap=600)
    assert small.evidence_tokens <= 600 and len(small.packets) <= 1
    assert (small.packets_clipped_last and small.packets_dropped == 0) if small.packets else small.packets_dropped == 1
    assert "--- nonselected_candidates: %d of 11 ---" % len(small.packets) in small.text


def test_metadata_alone_over_cap_is_protocol_error(world, stub):
    with pytest.raises(T.ProtocolError, match="metadata alone"):
        _compile(world, T.Method.IND_VOTE, world.tasks[0].source_id, stub, evidence_cap=20)


# --------------------------------------------------------------------------- sentinel / methods


def test_no_valid_candidate_uses_sentinel_and_missing_flag(world, stub):
    bcb1 = world.tasks[3].source_id
    for method in (T.Method.IND_VOTE, T.Method.DEC, T.Method.CEN_FLAT):
        rep = _compile(world, method, bcb1, stub)
        assert rep.selected_candidate is None and rep.selected_is_sentinel and rep.selected_candidate_id is None
        assert rep.selected_personal_confidence is None and rep.personal_confidence_missing
        assert _bytes(rep.text, rep.spans["selected_candidate"]) == T.SENTINEL_JSON
        assert json.loads(_bytes(rep.text, rep.spans["selected_personal_confidence"])) == {"scope": "PERSONAL_FINAL", "value": None, "missing": True}
        assert rep.vote_metadata["no_valid_candidate"] is True and rep.vote_metadata["valid_count"] == 0
        if method is T.Method.CEN_FLAT:
            # the invalid native final is the one non-selected pool member: a typed unavailable packet
            assert rep.nonselected_total == 1 and len(rep.packets) == 1 and rep.packets[0].unavailable
            assert "--- nonselected_candidates: 1 of 1 ---\n[packet 1]\n" + T.PACKET_UNAVAILABLE_JSON in rep.text
        else:
            assert rep.nonselected_total == 5 and all(p.unavailable for p in rep.packets) and len(rep.packets) == 4


def test_dec_uses_latest_slots_pool(world, stub):
    sid = world.tasks[0].source_id
    rep = _compile(world, T.Method.DEC, sid, stub)
    assert rep.vote_metadata["pool_kind"] == "latest_slots" and rep.vote_metadata["pool_size"] == 5
    assert rep.selected_candidate.final_answer == "B" and rep.vote_metadata["winning_count"] == 3
    assert rep.vote_metadata["rounds_completed"] == 1 and rep.decision_rule == RP.DECISION_RULES[T.Method.DEC]
    assert rep.nonselected_total == 4 and len(rep.packets) == 4
    stages = {m["stage"] for m in rep.nonselected}
    assert stages == {"round1"}  # roots are not in the latest_slots pool
    assert rep.budget_metadata["calls_by_role"] == {"root": 5, "revise": 5}


def test_cen_flat_native_pool(world, stub):
    sid = world.tasks[0].source_id
    rep = _compile(world, T.Method.CEN_FLAT, sid, stub)
    assert rep.vote_metadata["pool_kind"] == "native" and rep.vote_metadata["pool_size"] == 1
    assert rep.selected_candidate.confidence == 0.9 and rep.selected_personal_confidence == 0.9
    assert rep.vote_metadata["delegation_cycles"] == 2 and rep.vote_metadata["hub_calls"] == 3 and rep.vote_metadata["realized_participation"] == 4
    assert rep.packets == () and rep.nonselected_total == 0 and rep.decision_rule.startswith("NATIVE")


def test_refuses_unsealed_or_unsupported(world, stub):
    bcb0 = world.tasks[2].source_id
    with pytest.raises(T.ProtocolError, match="no sealed VOTE selection"):
        _compile(world, T.Method.DEC, bcb0, stub)
    with pytest.raises(T.ProtocolError, match="is missing"):
        compile_report(world.run_root, "hle:nope", world.cells[T.Method.DEC].cell_id, stub, seal=N.SEAL, cfg=world.cfg)
    with pytest.raises(T.ProtocolError):
        RP.find_sealed_selection({"selections": {}}, bcb0, "A.S_FRESH.32B.N5.B4.F00.e0.s000", T.Method.S_FRESH)
    with pytest.raises(T.ProtocolError, match="selections are not sealed"):
        compile_report(world.run_root, bcb0, world.cells[T.Method.IND_VOTE].cell_id, stub, seal="cd" * 32, cfg=world.cfg)


# --------------------------------------------------------------------------- manifest + render


def test_manifest_per_method(world, stub):
    sid = world.tasks[1].source_id
    expected_set = {T.Method.IND_VOTE: "SELF_ONLY", T.Method.DEC: "PEER_EXPOSED", T.Method.CEN_FLAT: "HUB_STATE"}
    for method, info in expected_set.items():
        rep = _compile(world, method, sid, stub)
        man = M.forecast_manifest(rep)
        assert set(R.FORECAST_MANIFEST_KEYS) <= set(man)
        assert man["scope"] == ["PERSONAL_FINAL", "TEAM_SELECTED"]
        assert man["mask"] == {"q_personal": "required", "q_team_now": "required", "q_child_contract": None, "q_recover": None, "q_preserve": None}
        assert man["observer_role"] == "COMMON_REPORT_READER" and man["information_set"] == info
        assert man["selected_pool_id"] == rep.pool_id and len(rep.pool_id) == 64
        assert man["checkpoint_id"] == "FINAL_HANDOFF_REPORT" and man["operation_id"] == "NONE"
        assert man["remaining_allowance"] == rep.budget_metadata["slack_flops"] == 900_000
        assert M.required_fields() == ("q_personal", "q_team_now") and set(M.null_fields()) == {"q_child_contract", "q_recover", "q_preserve"}


def test_rendered_request_anchors_match_bytes(world, stub):
    sid = world.tasks[0].source_id
    rep = _compile(world, T.Method.IND_VOTE, sid, stub)
    rendered = M.render_forecast_request(rep)
    assert len(rendered.messages) == 1 and rendered.messages[0]["role"] == "user"
    content = rendered.content.encode("utf-8")
    a = rendered.anchors
    assert content[a["spans"]["report"][0]:a["spans"]["report"][1]].decode() == rep.text
    assert content[a["spans"]["task"][0]:a["spans"]["task"][1]].decode() == rep.task_text
    assert a["task_only_anchor"] == a["spans"]["task"][1]
    assert content[a["task_only_anchor"] - len(rep.task_text.encode()):a["task_only_anchor"]].decode() == rep.task_text
    assert content[:a["state_anchor"]].endswith(R.REPORT_CLOSE.encode()) and content[a["state_anchor"]:] == b"\n"
    assert content[a["spans"]["evidence"][0]:a["spans"]["evidence"][1]].decode() == _bytes(rep.text, rep.spans["evidence"])
    for (s, e), packet in zip(a["spans"]["packets"], rep.packets):
        assert content[s:e].decode() == packet.serialized
    assert rendered.content.startswith("Estimate probabilities for the explicitly named targets")
    assert "Trusted manifest: " in rendered.content and "q_personal" in rendered.content
    # forecast spec: FORECAST_DECODING, the frozen seed key, role forecast
    spec = M.forecast_spec(rep, rendered, world.cfg, world.cfg.flagship_checkpoint)
    assert spec.decoding == T.FORECAST_DECODING and spec.decoding.max_tokens == 256 and not spec.decoding.enable_thinking and not spec.decoding.guided_json
    assert spec.seed_key.as_array() == [sid, "main", world.cfg.flagship_checkpoint.model_cell, 0, 0, "forecast", 0, "forecast"]
    assert spec.role == "forecast" and len(spec.request_id) == 64


def test_anchor_tokens_with_whitespace_tokenizer(world, stub):
    sid = world.tasks[0].source_id
    rep = _compile(world, T.Method.DEC, sid, stub)
    rendered = M.render_forecast_request(rep)
    tokens = M.anchor_tokens(stub, rendered.messages, rendered.anchors, enable_thinking=False)
    ids = list(render_chat_token_ids(stub, rendered.messages, False))
    assert tokens["prompt_token_ids"] == ids and tokens["prompt_tokens"] == len(ids)
    text = render_chat_text(stub, rendered.messages, False)
    for name in ("task_only_anchor", "state_anchor"):
        info = tokens["anchors"][name]
        piece = text[info["char_start"]:info["char_end"]]
        assert stub.encode(piece) == [ids[info["index"]]]
    task_tok = tokens["anchors"]["task_only_anchor"]
    assert text[task_tok["char_start"]:task_tok["char_end"]] == rep.task_text.split()[-1] and task_tok["exact"]
    state_tok = tokens["anchors"]["state_anchor"]
    assert text[state_tok["char_start"]:state_tok["char_end"]] == "===" and state_tok["index"] < tokens["last_prompt_token"]
    assert state_tok["index"] > task_tok["index"]


def test_report_render_payload(world, stub):
    sid = world.tasks[2].source_id
    rep = _compile(world, T.Method.IND_VOTE, sid, stub)
    payload = M.report_render(rep, stub, world.cfg, world.cfg.flagship_checkpoint)
    json.dumps(payload)  # JSON-safe
    assert payload["kind"] == "FINAL_HANDOFF_REPORT" and payload["report_id"] == rep.report_id and payload["method"] == "IND_VOTE"
    rendered = M.render_forecast_request(rep, payload["manifest"])
    assert payload["messages"] == rendered.messages
    assert payload["request_id"] == M.forecast_spec(rep, rendered, world.cfg, world.cfg.flagship_checkpoint).request_id
    assert payload["prompt_token_ids"] == list(render_chat_token_ids(stub, rendered.messages, False))
    assert payload["decoding"]["enable_thinking"] is False and payload["decoding"]["max_tokens"] == 256
    assert payload["byte_anchors"] == {"task_only_anchor": rendered.anchors["task_only_anchor"], "state_anchor": rendered.anchors["state_anchor"]}
    assert set(payload["anchor_tokens"]) == {"task_only_anchor", "state_anchor"}
    assert payload["report"]["text"] == rep.text and payload["evidence_tokens"] == rep.evidence_tokens
    assert payload["checkpoint"]["model_revision"] == world.cfg.flagship_checkpoint.model_revision


# --------------------------------------------------------------------------- real tokenizer


def test_real_tokenizer_cap_and_anchors(tmp_path: Path, cfg):
    try:
        from agents_scaling.study.inference.tokens import load_tokenizer

        tok = load_tokenizer(cfg.flagship_checkpoint)
    except Exception as exc:  # cache absent
        pytest.skip(f"Qwen3-32B tokenizer unavailable: {exc!r}")
    tasks = N.make_tasks(1, "main")[:1]
    w = N.World(tmp_path / "run", cfg, tasks)
    sid = tasks[0].source_id
    w.add(T.Method.IND_VOTE, {sid: (N.ind_pool("real", ["B", "A", "C", "B", "D", "A", "C"], words=2500), None, None)})
    w.seal()
    rep = _compile(w, T.Method.IND_VOTE, sid, tok)
    count = make_counter(tok)
    assert rep.evidence_tokens == count(_bytes(rep.text, rep.spans["evidence"])) <= EVIDENCE_TOKENS_CAP
    assert len(rep.packets) <= 4 and all(p.recipient_tokens <= T.PACKET_TOKENS_CAP for p in rep.packets)
    assert rep.packets_clipped_last  # 4 x 2,048 + metadata cannot fit 8,192
    payload = M.report_render(rep, tok, cfg, cfg.flagship_checkpoint)
    ids = payload["prompt_token_ids"]
    assert ids == list(render_chat_token_ids(tok, payload["messages"], False)) and len(ids) <= T.PROMPT_TOKENS_CAP
    text = render_chat_text(tok, payload["messages"], False)
    task_tok = payload["anchor_tokens"]["task_only_anchor"]
    piece = text[task_tok["char_start"]:task_tok["char_end"]]
    assert tok.decode([ids[task_tok["index"]]]) == piece and rep.task_text[-1] in piece
    assert piece.endswith(rep.task_text[-1]) or task_tok["straddles"]
    assert text[:task_tok["char"]].endswith(rep.task_text)
    state_tok = payload["anchor_tokens"]["state_anchor"]
    assert text[:state_tok["char"]].endswith(R.REPORT_CLOSE)
    assert tok.decode(ids[state_tok["index"] + 1:]).startswith("\n<|im_end|>")
    assert tok.decode(ids[: task_tok["index"] + 1]).endswith(piece)
    assert payload["chat_template_hash"] is not None
