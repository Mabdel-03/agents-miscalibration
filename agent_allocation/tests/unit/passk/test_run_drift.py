"""identity_reuse review: the passk drift check refuses only protocol-relevant endpoint changes."""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from agent_allocation.api.fake import fake_forced_endpoint_lock
from agent_allocation.passk import run as prun


def _lock():
    lock = fake_forced_endpoint_lock("qwen/qwen3.8-27b", tag="parasail/fp8")
    params = set(lock.supported_parameters) | set(prun.PROTOCOL_PARAMETERS) | {"tools"}
    return replace(lock, supported_parameters=tuple(sorted(params)), input_price_per_token="0.00000024",
                   output_price_per_token="0.0000024")


def test_cosmetic_changes_are_soft():
    lock = _lock()
    for live in (lock,
                 replace(lock, supported_parameters=tuple(sorted({*lock.supported_parameters, "min_p"}))),
                 replace(lock, supported_parameters=tuple(p for p in lock.supported_parameters if p != "tools")),
                 replace(lock, max_completion_tokens=lock.max_completion_tokens * 2),
                 replace(lock, input_price_per_token="0.000000240"),           # Decimal-equal
                 replace(lock, output_price_per_token="0.0000020"),            # a decrease
                 replace(lock, output_price_per_token="0.00000256")):          # +6.7%
        hard, _soft = prun.classify_endpoint_drift(lock, live)
        assert hard == [], (live, hard)
    assert prun.classify_endpoint_drift(lock, lock) == ([], [])


@pytest.mark.parametrize("change", [
    {"provider_tag": "parasail/bf16"},
    {"provider_name": "Other"},
    {"model": "qwen/qwen3.8-27b-other"},
    {"output_price_per_token": "0.0000030"},                                 # +25%
    {"max_completion_tokens": 8192},
])
def test_protocol_changes_are_hard(change):
    lock = _lock()
    hard, _ = prun.classify_endpoint_drift(lock, replace(lock, **change))
    assert hard, change


def test_dropped_forced_parameter_is_hard():
    lock = _lock()
    live = replace(lock, supported_parameters=tuple(p for p in lock.supported_parameters if p != "logit_bias"))
    hard, _ = prun.classify_endpoint_drift(lock, live)
    assert any("logit_bias" in h for h in hard)


def test_probe_retries_transient_listing_failure(monkeypatch):
    lock = _lock()
    calls = []

    class Client:
        async def list_endpoints(self, model):
            calls.append(model)
            raise RuntimeError("502")

    with pytest.raises(prun.PasskRunRefused, match="probe unavailable"):
        asyncio.run(prun.passk_endpoint_drift(Client(), lock, {"primary_model": lock.model}, attempts=3, delay_s=0))
    assert len(calls) == 3
