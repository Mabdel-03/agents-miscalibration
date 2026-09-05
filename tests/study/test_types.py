"""types.py: round trips, the exact sentinel, frozen constants (§3.5, §3.6, §4.3)."""

from __future__ import annotations

import json

import pytest

from agents_scaling.study import types as T


def _roundtrip(obj):
    data = json.loads(json.dumps(obj.to_dict(), allow_nan=False))
    back = type(obj).from_dict(data)
    assert back == obj
    assert back.to_dict() == obj.to_dict()
    return back


def test_sentinel_bytes_exact():
    assert json.dumps(T.SENTINEL, separators=(",", ":")) == '{"status":"unavailable","failure_code":"NO_VALID_PARENT"}'
    assert T.SENTINEL_JSON == '{"status":"unavailable","failure_code":"NO_VALID_PARENT"}'
    assert list(T.SENTINEL) == ["status", "failure_code"]


def test_frozen_constants():
    assert T.PURPOSES == ("root", "revise", "hub", "worker", "consumer", "hle_judge", "judge_best", "forecast")
    assert T.NAMESPACES == ("stateless_bank", "S_HISTORY", "DEC", "CEN_FLAT", "DEGREE", "judge", "forecast")
    assert (T.EXIT_DONE, T.EXIT_NO_SERVER, T.EXIT_INCOMPLETE, T.EXIT_SUSPENDED) == (0, 2, 3, 4)
    assert T.SEED_KEY_ORDER == ("source_id", "split", "model_cell", "episode_rep", "actor_slot", "purpose",
                                "step_slot", "namespace")
    assert T.SOLVER_DECODING == T.Decoding(0.6, 0.95, 20, 0.0, 0.0, 1.0, 8192, True)
    assert T.JUDGE_DECODING == T.Decoding(0.0, 1.0, 1, 0.0, 0.0, 1.0, 1024, False)
    assert T.FORECAST_DECODING.max_tokens == 256 and T.FORECAST_DECODING.enable_thinking is False
    assert T.PACKET_FIELD_PRIORITY[0] == "final_answer" and T.PACKET_FIELD_PRIORITY[-1] == "confidence"
    assert [m.value for m in T.Method] == ["BANK", "S_FRESH", "S_HISTORY", "IND_VOTE", "DEC", "CEN_FLAT",
                                           "IND_PRIVATE_REVISION", "DEC_ONE_ROUND", "DEGREE"]
    assert [k.value for k in T.CellKind] == ["GENERATE", "JUDGE_BEST", "JUDGE_HLE", "EVAL_BCB", "FORECAST"]
    assert (T.Framing.F01.team_frame, T.Framing.F01.vote_aware, T.Framing.F10.team_frame) == (0, 1, 1)
    with pytest.raises(ValueError):
        T.Framing.NATIVE.team_frame
    for exc in (T.InfraFailure, T.ProtocolError, T.ContextFailure):
        assert issubclass(exc, RuntimeError)


def test_simple_dataclass_roundtrips():
    task = T.PublicTask("hle:1", T.Domain.HLE, "main", "Q?", "multipleChoice", "Gold", 7, 12, category="Math")
    assert _roundtrip(task).domain is T.Domain.HLE
    assert task.to_dict()["domain"] == "hle"
    ckpt = T.Checkpoint("32B", "Qwen/Qwen3-32B", "9216db57" + "0" * 32, "9216db57" + "0" * 32, "32B-long", 2, "32B")
    assert _roundtrip(ckpt).model_cell == "Qwen3-32B@9216db57"
    _roundtrip(T.SOLVER_DECODING)
    key = T.SeedKey("hle:1", "main", ckpt.model_cell, 0, 2, T.PURPOSE_REVISE, 1, T.NS_DEC)
    assert _roundtrip(key).as_array() == ["hle:1", "main", ckpt.model_cell, 0, 2, "revise", 1, "DEC"]
    spec = T.RequestSpec(({"role": "user", "content": "x"},), T.SOLVER_DECODING, ckpt, key, "revise", "sid", "ab" * 32)
    back = _roundtrip(spec)
    assert back.request_id == spec.request_id and isinstance(back.messages, tuple)


def test_candidate_and_selection_roundtrips():
    cand = T.Candidate("m", (T.Evidence("c", "s", "low"),), ("alt",), (), "42", 0.5)
    assert list(cand.to_dict()) == ["approach", "evidence", "alternatives_considered", "failure_checks",
                                    "final_answer", "confidence"]
    _roundtrip(cand)
    rec = T.CandidateRecord("cid", "rid", 0, "root", True, None, cand, "h" * 64, "r" * 64, "42", "exact_norm")
    _roundtrip(rec)
    invalid = T.CandidateRecord("cid2", "rid2", 1, "root", False, "INVALID_JSON", None, "h" * 64, "r" * 64)
    assert _roundtrip(invalid).candidate is None
    packet = T.Packet("pid", 1, "h" * 64, {"final_answer": "42", "evidence": ["c: s"]}, ((0, 2), (2, 9)),
                      {"evidence": True}, False, 40, "{...}", False)
    assert _roundtrip(packet).spans == ((0, 2), (2, 9))
    sub = T.SubtaskResult("s1", "return the count", "partial", "3", ("assume a",), ("h1",), None)
    _roundtrip(sub)


def test_coordinator_action_variants():
    cand = T.Candidate("m", (), (), (), "42", 1.0)
    final = T.FinalAction(cand)
    assert final.to_dict() == {"action": "final", "candidate": cand.to_dict()}
    assert T.coordinator_action_from_dict(json.loads(final.to_json())) == final
    delegate = T.DelegateAction((T.Assignment(1, "s1", "q", ("task",), "integer", "return an int"),))
    data = json.loads(delegate.to_json())
    assert data["action"] == "delegate" and data["assignments"][0]["worker_slot"] == 1
    assert T.coordinator_action_from_dict(data) == delegate
    with pytest.raises(ValueError):
        T.coordinator_action_from_dict({"action": "plan"})


def test_cellspec_roundtrip():
    cell = T.CellSpec("A.DEC.32B.N5.B4.nat.e0.s000", T.CellKind.GENERATE, "A", T.Method.DEC, "32B", 5, 4,
                      T.Framing.NATIVE, 0, "main", ("hle:1", "bcb:2"), ("F.BANK.32B.N1.B0.F00.e0.s000",), 5, 2, "32B")
    back = _roundtrip(cell)
    assert back.items == ("hle:1", "bcb:2") and back.degree is None and back.framing is T.Framing.NATIVE
    assert cell.to_dict()["framing"] == "nat" and cell.to_dict()["method"] == "DEC"


def test_request_record_fixture_roundtrip_and_hash(request_record_fixture):
    rec = T.RequestRecord.from_dict(request_record_fixture)
    rec.verify()
    assert rec.to_dict() == request_record_fixture
    assert rec.identity["hook_hash"] == "none" and rec.identity["local_caps"] == {"max_input": 32768, "max_tokens": 8192}
    assert isinstance(rec.seed_key, T.SeedKey) and rec.engine_seed < 2**63
    tampered = T.RequestRecord.from_dict({**request_record_fixture, "prompt_tokens": 999})
    with pytest.raises(T.ProtocolError):
        tampered.verify()


def test_episode_result_fixture_roundtrip_and_dict_view(episode_result_fixture):
    result = T.EpisodeResult.from_dict(episode_result_fixture)
    assert result.to_dict() == episode_result_fixture
    assert json.loads(result.to_json()) == episode_result_fixture
    assert result["source_id"] == episode_result_fixture["source_id"]
    assert "bank" in result and "nope" not in result
    assert result.get("episode") is None and result.get("missing", 1) == 1
    assert set(result.keys()) == set(episode_result_fixture)
    assert isinstance(result.cell, T.CellSpec) and result.cell.kind is T.CellKind.GENERATE
    assert result.bank[0].candidate.final_answer == "4"
    assert result.domain is T.Domain.HLE


def test_from_dict_reports_missing_required_field():
    with pytest.raises(KeyError, match="missing field"):
        T.Evidence.from_dict({"claim": "c", "support": "s"})
