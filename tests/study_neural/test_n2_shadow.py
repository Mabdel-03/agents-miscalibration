"""N2 strict forecast parsing (§8.7 / handoff forecast.schema.json) and the forecast file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents_scaling.study import types as T
from agents_scaling.study.forecast import shadow as S
from agents_scaling.study.forecast.shadow import parse_forecast

OK = '{"q_personal":0.62,"q_child_contract":null,"q_team_now":0.71,"q_recover":null,"q_preserve":null}'


@pytest.mark.parametrize(
    "content,finish,status,detail",
    [
        (OK, "stop", "ok", ""),
        ("\n\n" + OK + "\n", "stop", "ok", ""),  # 05_vllm_response_shape: content may start with blank lines
        ("```json\n" + OK + "\n```", "stop", "ok", ""),  # exactly one enclosing fence is stripped
        ("```json\n```json\n" + OK + "\n```\n```", "stop", "NOT_JSON", ""),  # a second fence is not salvaged
        (OK, "length", "TRUNCATED", "256-token"),
        ("", "stop", "EMPTY", ""),
        (None, "stop", "EMPTY", "no content"),
        ("not json at all", "stop", "NOT_JSON", ""),
        (OK + " trailing", "stop", "TRAILING_TEXT", ""),
        ('{"q_personal":0.5,"q_personal":0.6,"q_child_contract":null,"q_team_now":0.7,"q_recover":null,"q_preserve":null}', "stop", "DUPLICATE_KEY", "q_personal"),
        ('{"q_personal":NaN,"q_child_contract":null,"q_team_now":0.7,"q_recover":null,"q_preserve":null}', "stop", "NONFINITE", ""),
        ('{"q_personal":1e999,"q_child_contract":null,"q_team_now":0.7,"q_recover":null,"q_preserve":null}', "stop", "NONFINITE", ""),
        ('{"q_personal":0.5,"q_team_now":0.7}', "stop", "SCHEMA", "missing"),  # missing keys
        (OK[:-1] + ',"extra":1}', "stop", "SCHEMA", "extra"),  # extra key
        ('{"q_personal":1.5,"q_child_contract":null,"q_team_now":0.7,"q_recover":null,"q_preserve":null}', "stop", "SCHEMA", "outside"),
        ('{"q_personal":"0.5","q_child_contract":null,"q_team_now":0.7,"q_recover":null,"q_preserve":null}', "stop", "SCHEMA", "number or null"),
        ('{"q_personal":true,"q_child_contract":null,"q_team_now":0.7,"q_recover":null,"q_preserve":null}', "stop", "SCHEMA", "number or null"),
        ('{"q_personal":null,"q_child_contract":null,"q_team_now":0.7,"q_recover":null,"q_preserve":null}', "stop", "REQUIRED_NULL", "q_personal"),
        ('[0.5, 0.7]', "stop", "SCHEMA", "object"),
        ('{"q_personal":0,"q_child_contract":0.2,"q_team_now":1,"q_recover":0.3,"q_preserve":0.4}', "stop", "ok", ""),  # ints at the bounds; nulls not enforced for inapplicable fields
    ],
)
def test_parse_forecast_cases(content, finish, status, detail):
    parsed = parse_forecast(content, finish_reason=finish)
    assert parsed.status == status, parsed
    assert parsed.valid == (status == "ok")
    assert detail in parsed.detail
    if status == "ok":
        assert parsed.forecast is not None
        values = parsed.forecast.to_dict()
        assert list(values) == list(S.FORECAST_FIELDS)
        assert all(v is None or (isinstance(v, float) and 0.0 <= v <= 1.0) for v in values.values())
        assert values["q_personal"] is not None and values["q_team_now"] is not None
    else:
        assert parsed.forecast is None and status in S.PARSE_FAILURES


def test_parse_ok_values_and_mask_variants():
    parsed = parse_forecast(OK)
    assert parsed.forecast.to_dict() == {"q_personal": 0.62, "q_child_contract": None, "q_team_now": 0.71, "q_recover": None, "q_preserve": None}
    # a mask that requires nothing accepts an all-null object
    all_null = '{"q_personal":null,"q_child_contract":null,"q_team_now":null,"q_recover":null,"q_preserve":null}'
    assert parse_forecast(all_null, mask={k: None for k in S.FORECAST_FIELDS}).valid
    assert parse_forecast(all_null).status == "REQUIRED_NULL"
    assert parsed.to_dict()["status"] == "ok" and parse_forecast("x").to_dict()["values"] is None


def test_paths_and_forecast_output(tmp_path: Path):
    assert S.forecast_path(tmp_path, "hle:a b", T.Method.DEC) == tmp_path / "forecast" / "hle:a b.DEC.json"
    assert S.report_path(tmp_path, "bcb:1", "CEN_FLAT") == tmp_path / "forecast" / "reports" / "bcb:1.CEN_FLAT.json"
    assert S.error_path(tmp_path, "bcb:1", "IND_VOTE").parent.name == "errors"

    class Rep:  # the fields forecast_output reads
        report_id = "r" * 64
        source_id = "hle:x"
        method = T.Method.IND_VOTE
        cell_id = "A.IND_VOTE.32B.N5.B4.F11.e0.s000"
        seal = "s" * 64
        selection_id = "sel" * 21 + "x"
        pool_id = "p" * 64
        selected_personal_confidence = 0.8
        personal_confidence_missing = False
        selected_is_sentinel = False

    record = T.RequestRecord(
        schema_version=1, request_id="q" * 64, identity={"input_hash": "i" * 64}, seed_key=T.SeedKey("hle:x", "main", "Qwen3-32B@9216db57", 0, 0, "forecast", 0, "forecast"),
        engine_seed=1, model={}, messages=({"role": "user", "content": "x"},), chat_template_kwargs={"enable_thinking": False}, chat_template_hash="h",
        sampling={}, prompt_token_ids=(1, 2, 3), prompt_tokens=3, response={"content": OK, "finish_reason": "stop", "completion_tokens": 30, "reasoning_tokens": 0, "reasoning": None},
        flops={"prefill": 1, "decode": 2, "total": 3}, timing={}, endpoint={}, attempts=1, producer={}, content_sha256=None,
    )
    manifest = {"scope": ["PERSONAL_FINAL", "TEAM_SELECTED"], "observer_role": "COMMON_REPORT_READER", "information_set": "SELF_ONLY",
                "selected_pool_id": "p" * 64, "operation_id": "NONE", "remaining_allowance": 7, "mask": {}}
    payload = S.forecast_output(Rep(), manifest, record, parse_forecast(OK), aliased=False, produced_at=5.0, cell_id="forecast.IND_VOTE")
    json.dumps(payload)
    assert payload["parse_status"] == "ok" and payload["parsed"]["q_team_now"] == 0.71 and payload["raw_content"] == OK
    assert payload["request_id"] == "q" * 64 and payload["forecast_prompt_hash"] == "i" * 64 and payload["cost"]["completion_tokens"] == 30
    row = payload["confidence_row"]
    assert row["StateSnapshot_id"] == "r" * 64 and row["scope"] == ["PERSONAL_FINAL", "TEAM_SELECTED"] and row["parsing_status"] == "ok"
    assert row["future_label_ids_sealed"] == Rep.selection_id and row["remaining_allowance"] == 7 and row["probability_fields"]["q_personal"] == 0.62
    path = S.write_forecast(tmp_path, payload)
    assert path == S.forecast_path(tmp_path, "hle:x", "IND_VOTE") and json.loads(path.read_text())["report_id"] == "r" * 64
