"""WP4 resources: FLOP wiring guard, GPU sampling, sacct capture, batched attribution.

Spec §16, §18 (``resources.parquet``; "do not claim exact per-request FLOPs from token count
alone", "for batched GPU time attribution use a documented approximation and retain
server-level totals").  Plan §5.8 "Compute cost".
"""

from __future__ import annotations

import math

import pytest

from agents_scaling.ard import resources as RES
from agents_scaling.study.resources.oracle import FlopOracle
from agents_scaling.study.types import ProtocolError


def test_the_oracle_factory_is_the_frozen_study_one():
    # a classmethod is re-bound on every attribute access, so compare the function
    assert RES.oracle_factory.__func__ is FlopOracle.from_checkpoint.__func__


@pytest.mark.parametrize("flops", [None, {}, {"prefill": 1, "decode": None, "total": 1}])
def test_a_placeholder_flop_block_is_refused(flops):
    with pytest.raises(ProtocolError):
        RES.assert_flops_populated(flops, where="decision-1")


def test_a_populated_flop_block_passes():
    RES.assert_flops_populated({"prefill": 1, "decode": 2, "total": 3, "oracle": "abc"})


# --------------------------------------------------------------------------- nvidia-smi


def test_the_sampler_query_is_the_field_list_the_spec_names():
    argv = RES.nvidia_smi_argv()
    assert argv[1] == ("--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,power.draw")
    assert "--format=csv,noheader,nounits" in argv
    line = RES.sampler_line("/tmp/gpu.csv")
    assert "nvidia-smi" in line and "systime()" in line and line.rstrip().endswith("&")


def test_sampler_rows_parse_into_resources_columns():
    text = "1700000000,GPU-abc,NVIDIA A100-SXM4-80GB,81920,40960,73,213.5\nbroken row\n"
    rows = RES.parse_nvidia_smi_csv(text, host="node3904", job_id="123")
    assert len(rows) == 1
    row = rows[0]
    assert row["gpu_uuid"] == "GPU-abc" and row["gpu_type"].startswith("NVIDIA A100")
    assert row["memory_total"] == 81920 and row["utilization"] == 73
    assert row["power_if_available"] == 213.5 and row["node"] == "node3904"


def test_sacct_rows_become_allocated_gpu_seconds():
    text = "4242|asys-srv|pi_manoli|node3904|billing=16,cpu=16,gres/gpu=2,mem=120G|3600|COMPLETED\n"
    rows = RES.parse_sacct_gpu_seconds(text)
    assert rows[0]["gpus"] == 2 and rows[0]["elapsed_s"] == 3600
    assert rows[0]["allocated_gpu_seconds"] == 7200


# --------------------------------------------------------------------------- attribution


def test_the_batch_proxy_is_the_endpoint_and_a_one_second_arrival_bucket():
    assert RES.server_batch_id("h:8000", 10.2) == RES.server_batch_id("h:8000", 10.9)
    assert RES.server_batch_id("h:8000", 10.2) != RES.server_batch_id("h:8000", 11.2)
    assert RES.server_batch_id("h:8000", 10.2) != RES.server_batch_id("h:8001", 10.2)
    assert "documented approximation" not in RES.ATTRIBUTION_RULE  # it *is* the documentation
    assert "divided by" in RES.ATTRIBUTION_RULE


def test_a_solo_request_keeps_its_whole_wall_time():
    rows, totals = RES.attribute_gpu_seconds([RES.RequestResource("r", "e", 0.0, 2.0)])
    assert rows[0]["attributed_gpu_seconds"] == pytest.approx(2.0)
    assert rows[0]["mean_concurrency"] == pytest.approx(1.0)
    assert totals["e"]["server_busy_seconds"] == pytest.approx(2.0)


def test_two_fully_overlapping_requests_split_the_replica_second():
    requests = [RES.RequestResource("a", "e", 0.0, 2.0), RES.RequestResource("b", "e", 0.0, 2.0)]
    rows, totals = RES.attribute_gpu_seconds(requests)
    assert [row["attributed_gpu_seconds"] for row in rows] == pytest.approx([1.0, 1.0])
    assert totals["e"]["server_busy_seconds"] == pytest.approx(2.0)
    assert totals["e"]["summed_request_seconds"] == pytest.approx(4.0)
    assert totals["e"]["attributed_seconds"] <= totals["e"]["server_busy_seconds"] + 1e-9


def test_server_level_totals_are_retained_per_replica():
    requests = [RES.RequestResource("a", "e1", 0.0, 1.0), RES.RequestResource("b", "e2", 5.0, 3.0)]
    _rows, totals = RES.attribute_gpu_seconds(requests)
    assert set(totals) == {"e1", "e2"}
    assert totals["e2"]["first_arrival"] == 5.0 and totals["e2"]["last_completion"] == 8.0


def test_physical_prefill_is_the_prompt_minus_the_cache():
    request = RES.RequestResource("a", "e", 0.0, 1.0, prompt_tokens=1000, cached_prefill_tokens=768)
    assert request.prefill_tokens == 232
    rows, _totals = RES.attribute_gpu_seconds([request])
    assert rows[0]["prefill_tokens"] == 232 and rows[0]["prompt_tokens"] == 1000


def test_decision_rows_become_attribution_inputs():
    decisions = [{"decision_id": "d1", "request_id": "r1", "endpoint_id": "e", "arrival_ts": 1.0,
                  "latency_s": 2.0, "prompt_tokens": 100, "cached_prefill_tokens": 40,
                  "thinking_tokens": 30, "final_tokens": 12}]
    requests = RES.request_resources_from_decisions(decisions)
    assert requests[0].decode_tokens == 42 and requests[0].prefill_tokens == 60
    rows, _totals = RES.attribute_gpu_seconds(requests)
    assert rows[0]["decision_id"] == "d1"
    assert math.isclose(rows[0]["attributed_gpu_seconds"], 2.0)
