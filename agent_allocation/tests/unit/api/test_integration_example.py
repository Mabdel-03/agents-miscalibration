"""Mock-backed integration example for the lead's ``live.py`` (canary wiring), end to end on the fake transport.

Live wiring differs in exactly two places, marked LIVE below: ``OpenRouterClient(api_key=load_api_key())`` without a
``transport``, and a real run directory. Everything else -- discovery, the write-once lock, per-attempt retry
decisions, raw persistence, metadata reconciliation, the canary report -- is the same code path.
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from agent_allocation.api.canary import canary_report
from agent_allocation.api.client import OpenRouterClient
from agent_allocation.api.endpoints import select_standard_endpoint
from agent_allocation.api.fake import FakeReply, FakeScript, Fault, fake_generation_outcome_sync, \
    fake_openrouter_transport
from agent_allocation.api.reconcile import CONSISTENT, PENDING, reconcile_generation
from agent_allocation.api.request import spec_lock_problems
from agent_allocation.config import decoding_config_hash, load_config
from agent_allocation.contracts import (
    CIRCUIT_BREAKING_TRANSPORT,
    RETRYABLE_TRANSPORT,
    EndpointLock,
    GenerationKind,
    GenerationResult,
    GenerationSpec,
    PublicTask,
    OutputKind,
    TransportStatus,
    dump_record,
    load_record,
    public_task_hash,
)

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "balanced150.json"


class CircuitOpen(RuntimeError):
    pass


async def dispatch_node(client, spec, messages, lock, cfg, raw_dir: Path):
    """What the scheduler does per node, minus reservations: <= max_transport_attempts HTTP attempts, retry only
    RETRYABLE_TRANSPORT, open the circuit on CIRCUIT_BREAKING_TRANSPORT, never redraw a completed model outcome."""
    attempts = int(cfg["scheduler"]["max_transport_attempts"])
    for attempt_no in range(1, attempts + 1):
        outcome = await client.generate(spec, messages, attempt_id=f"{spec.node_id}:{attempt_no}", lock=lock,
                                        sampling=cfg["sampling"])
        result = outcome.result
        # 02's write_raw_response persists exactly these fields before any terminal record is committed.
        raw = {"attempt_id": result.attempt_id, "node_id": result.node_id, "request_body": outcome.request_body,
               "response_status": result.http_status, "response_headers": dict(outcome.response_headers),
               "response_body_text": outcome.response_body.decode("utf-8", errors="replace")}
        (raw_dir / f"{result.attempt_id.replace(':', '_')}.json.gz").write_bytes(
            gzip.compress(json.dumps(raw).encode()))
        status = result.transport_status
        if status in CIRCUIT_BREAKING_TRANSPORT:
            raise CircuitOpen(f"{status.value}: {result.error_detail}")
        if status in RETRYABLE_TRANSPORT and attempt_no < attempts:
            await asyncio.sleep(min(result.retry_after_seconds or 0.0, 0.01))  # live: honor it + jitter
            continue
        return result
    return result


async def test_canary_wiring_example(tmp_path):
    cfg = load_config(CONFIG)
    model = cfg["primary_model"]
    script = FakeScript(metadata_lag_polls=1)
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))  # LIVE: api_key, no transport

    # 1. Discovery: save the raw listing bytes beside the lock, hash them, select the exact standard endpoint.
    raw_listing, listing = await client.list_endpoints(model)
    snapshot = tmp_path / "locks" / "snapshots" / "endpoints_flash_lite.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(raw_listing)
    lock = select_standard_endpoint(listing, model, snapshot_path="snapshots/endpoints_flash_lite.json",
                                    snapshot_sha256=hashlib.sha256(raw_listing).hexdigest(),
                                    fetched_at="2026-09-17T07:00:00Z")
    lock_file = tmp_path / "locks" / "openrouter_endpoints.json"
    lock_file.write_text(json.dumps(dump_record(lock)))
    lock = load_record(EndpointLock, json.loads(lock_file.read_text()))  # resume reads the lock, never re-fetches
    assert lock.provider_tag == "google-ai-studio" and lock.snapshot_sha256 == hashlib.sha256(raw_listing).hexdigest()

    # 2. Canary nodes: A = 512 / 1,024 / 4,096 plus one long setting, on a development item.
    task = PublicTask(item_id="math500/dev/1", benchmark="math500", source_revision="rev", prompt="What is 6*7?",
                      output_kind=OutputKind.MATH)
    specs = [GenerationSpec(run_id="canary", partition_id="canary", item_id=task.item_id, benchmark=task.benchmark,
                            kind=GenerationKind.ROOT if a < 16384 else GenerationKind.DEEP, model=model,
                            endpoint_lock_hash=lock.lock_hash, input_manifest_hash=public_task_hash(task),
                            decoding_config_hash=decoding_config_hash(cfg), prompt_version="prompts-v1-test",
                            reasoning_budget=a, completion_headroom=cfg["reasoning"]["text_completion_headroom"],
                            root_slot=0, team_size=0, update_round=0, draw_id=draw)
             for a in (512, 1024, 4096, 16384) for draw in (0, 1)]
    assert all(spec_lock_problems(spec, lock) == [] for spec in specs)
    messages = [{"role": "system", "content": "Solve."}, {"role": "user", "content": task.prompt}]

    # Transient faults the loop must absorb: a 429 then success, an ambiguous timeout then success.
    script.by_node[specs[0].node_id] = [FakeReply(fault=Fault.RATE_LIMITED, retry_after="1"), FakeReply()]
    script.by_node[specs[3].node_id] = [FakeReply(fault=Fault.READ_TIMEOUT), FakeReply()]

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = []
    for spec in specs:
        result = await dispatch_node(client, spec, messages, lock, cfg, raw_dir)
        assert result.transport_status is TransportStatus.OK
        metadata = await client.generation_metadata(result.generation_id)
        assert reconcile_generation(result, metadata, lock).state == PENDING   # metadata lags: keep reserving
        metadata = await client.generation_metadata(result.generation_id)
        reconciliation = reconcile_generation(result, metadata, lock)
        assert reconciliation.state == CONSISTENT, reconciliation.problems
        parsed = load_record(GenerationResult, json.loads(json.dumps(dump_record(result))))
        rows.append((spec, parsed, metadata))

    # 3. Gate 1 facts.
    report = canary_report(rows)
    assert [b["requested_reasoning_budget"] for b in report["budgets"]] == [512, 1024, 4096, 16384]
    assert report["overall"]["transport_statuses"] == {"ok": 8}
    assert report["overall"]["metadata"]["service_tiers"] == {"standard": 8}
    assert not [flag for flag in report["flags"] if "cache" in flag or "mismatch" in flag]

    # 4. Money: what the fake billed = settled usage + the ambiguous attempt that was billed without a response.
    settled = sum((r.usage.actual_cost_usd for _, r, _ in rows), Decimal("0"))
    key = await client.key_info()
    await client.aclose()
    assert key["usage"] == script.billed_usd > settled                     # the timed-out attempt was charged too
    assert len(list(raw_dir.iterdir())) == 10                              # 8 nodes + 2 retried attempts
    assert len(script.completion_calls()) == 10


async def test_a_circuit_breaking_status_stops_dispatch(tmp_path, make_spec, lock):
    cfg = load_config(CONFIG)
    script = FakeScript(default=FakeReply(fault=Fault.PAYMENT_REQUIRED))
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(script))
    spec = make_spec()
    try:
        await dispatch_node(client, spec, [{"role": "user", "content": "x"}], lock, cfg, tmp_path)
    except CircuitOpen as exc:
        assert "payment_required" in str(exc)
    else:
        raise AssertionError("402 must open the circuit")
    finally:
        await client.aclose()
    assert len(script.completion_calls()) == 1   # no retry after a circuit-breaking status


def test_sync_outcome_helper_for_non_async_tests(make_spec):
    outcome = fake_generation_outcome_sync(make_spec(benchmark="bbh_logic4", item_id="bbh_logic4/boolean_expressions/1"))
    assert outcome.result.visible_content.endswith(("<final>True</final>", "<final>False</final>"))
    length = fake_generation_outcome_sync(make_spec(), reply=FakeReply(fault=Fault.EMPTY_LENGTH))
    assert length.result.visible_content == "" and length.result.usage.reasoning_tokens is not None


async def test_sync_outcome_helper_inside_a_running_loop(make_spec):
    outcome = fake_generation_outcome_sync(make_spec())
    assert outcome.result.transport_status is TransportStatus.OK
