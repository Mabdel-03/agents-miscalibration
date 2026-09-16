"""WP1: the single ``decisions`` row site, and the compute ledger.

ARD36's defect D-02 was not a wrong number: it was a *second place* that built a row carrying
physical ids.  The two drifted, and the phase-2 request of every continued draw fell out of the
cost ledger -- its prefill, its decode, and the phase-1 span it replaced.  The structural fix is
here: ``telemetry.decision_row`` is the only ``DecisionRow`` constructor (R6), ``request_rows``
emits one row per *physical* request, and the GPU-second attribution reads ``requests`` rather
than ``decisions``, so nothing can fall out without a row going missing.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids as I
from agents_scaling.nextdyn import inference as INF
from agents_scaling.nextdyn import tables as T
from agents_scaling.nextdyn import telemetry as TM
from tests.nextdyn.conftest import nd_request

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "src" / "agents_scaling" / "nextdyn"


# --------------------------------------------------------------------------- a PublicOutput stub

@dataclass(frozen=True)
class StubOutput:
    """WP2's ``schema.PublicOutput``, reduced to the fields the contract lets telemetry read.

    Written out here rather than imported so this group pins the *contract* surface: if WP2
    renamed one of these, that is a contract change, not a quiet coupling.
    """

    reasoning: str | None = "a public rationale"
    shareable_claims: tuple[str, ...] = ("one claim",)
    final_answer: str | None = "4"
    confidence: float | None = 0.75
    field_order: tuple[str, ...] = ("reasoning", "shareable_claims", "final_answer", "confidence")
    order_violation: bool = False
    parse_status: str = "ok"
    raw_text: str = ""
    content_hash: str = ""
    over_limits: dict = None
    final_answer_span: tuple[int, int] | None = (10, 11)
    confidence_span: tuple[int, int] | None = (20, 24)
    confidence_emitted_before_answer: bool = False


def _cell(cell_id="donor_bank/MATH/0", episode_id="donor_bank|item|variant|m|b|k0"):
    return TM.CellContext(cell_id=cell_id, episode_id=episode_id, display_label=None,
                          bundle_export_hash=None)


def _producer(cell_id="donor_bank/MATH/0"):
    return TM.ProducerStamp(cell_id=cell_id, run_id="nd1-test", slurm_job_id="12345",
                            host="node0000", endpoint_id="node0000:8884", gpu_type="l40s",
                            partition="ou_bcs_low")


# --------------------------------------------------------------------------- R6


def test_telemetry_is_the_only_decision_row_constructor():
    """R6, AST-checked over the package: the D-02 recurrence is a *second* construction site."""
    offenders = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if path.name in ("telemetry.py", "tables.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else None)
                if name == "DecisionRow":
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"DecisionRow built outside telemetry.decision_row (R6): {offenders}"
    assert "DecisionRow" in ast.dump(ast.parse((PACKAGE_DIR / "telemetry.py").read_text()))


# --------------------------------------------------------------------------- the decisions row


def test_a_decision_row_keeps_parse_validity_delivery_and_infrastructure_apart(
        nd_serve, nd_ctx, nd_contract, nd_tokens):
    """Spec 4.6: missingness and validity are not semantic error, and each has its own field."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=4):
        record = INF.generate_decision(req, ctx=ctx)

    row = TM.decision_row(record, req.key, StubOutput(), cell=_cell(), parents=(),
                          seen_transmissions=(), aliased=False, reuse_of=None)
    assert isinstance(row, T.DecisionRow)
    assert row.decision_id == req.key.decision_id()
    assert row.generation_uid == record.generation_uid
    assert (row.parse_status, row.validity, row.delivery_success, row.infrastructure_status) == (
        "ok", True, True, TM.INFRA_OK)
    assert row.termination_reason in TM.TERMINATION_REASONS
    assert row.canonical_class is None          # grading is writer-side (R7)
    assert row.first_decision_id is None        # materialize fills it per generation_uid
    assert json.loads(row.request_hashes_json) == list(record.physical_request_ids)
    assert row.content_hash == I.content_hash(record.final_text.encode("utf-8"))
    assert row.receiver_model_hash == I.model_hash(nd_contract.model("primary"))

    unextractable = TM.decision_row(record, req.key, StubOutput(final_answer=None,
                                                                parse_status="missing_final"),
                                    cell=_cell(), parents=(), seen_transmissions=(),
                                    aliased=False, reuse_of=None)
    assert unextractable.validity is False and unextractable.delivery_success is False
    assert unextractable.infrastructure_status == TM.INFRA_OK      # not an infra failure


def test_the_logprob_sum_and_mean_describe_the_same_span(nd_serve, nd_ctx, nd_contract,
                                                          nd_tokens):
    """R1: the record defines ``answer_logprob`` as the mean over exactly ``final_tokens``, so
    the two columns agree by construction instead of being measured twice."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=4):
        record = INF.generate_decision(req, ctx=ctx)
    row = TM.decision_row(record, req.key, StubOutput(), cell=_cell(), parents=(),
                          seen_transmissions=(), aliased=False, reuse_of=None)
    assert row.answer_token_count == record.final_tokens > 0
    assert row.answer_logprob_mean is not None
    assert row.answer_logprob_sum == pytest.approx(row.answer_logprob_mean * row.answer_token_count)
    assert row.logprob_norm_scheme == TM.LOGPROB_NORM_MEAN_FINAL_SPAN


def test_a_row_whose_generation_uid_does_not_hash_its_own_requests_is_refused(
        nd_serve, nd_ctx, nd_contract, nd_tokens):
    import dataclasses

    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=4):
        record = INF.generate_decision(req, ctx=ctx)
    tampered = dataclasses.replace(record, generation_uid="g:" + "0" * 64)
    with pytest.raises(Exception, match="one sample, one uid"):
        TM.decision_row(tampered, req.key, StubOutput(), cell=_cell(), parents=(),
                        seen_transmissions=(), aliased=False, reuse_of=None)


# --------------------------------------------------------------------------- the requests rows


def test_a_continued_decision_is_two_requests_rows_in_phase_order(nd_serve, nd_ctx, nd_contract,
                                                                   nd_tokens):
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64):
        record = INF.generate_decision(req, ctx=ctx)
    specs = INF.request_specs_for(req, ctx, record)
    rows = TM.request_rows(record, req.key, cell_id="donor_bank/MATH/0", request_specs=specs,
                           producer=_producer())

    assert len(rows) == 2
    assert [r.phase_index for r in rows] == [0, 1]
    assert tuple(r.request_hash for r in rows) == record.physical_request_ids
    assert all(r.generation_uid == record.generation_uid for r in rows)
    assert all(r.first_decision_id is None for r in rows)
    assert rows[0].decoding_hash != rows[1].decoding_hash      # phase 2 is its own decoding
    assert rows[0].seed_hex != rows[1].seed_hex                # ... and its own seed
    assert rows[1].prompt_tokens > rows[0].prompt_tokens       # the prefix is prompt + thinking
    # The hardware columns are the *replica stamp* (where the request ran, as the pool resolved
    # it), while ``producer_json`` is what the client was told; the fake endpoint has no Slurm
    # job, so the stamp is null and the producer's is not.  They are separate facts on purpose.
    assert all(r.gpu_type == record.gpu_type and r.partition == record.partition for r in rows)
    producer = json.loads(rows[0].producer_json)
    assert producer["cell_id"] == "donor_bank/MATH/0"
    assert (producer["gpu_type"], producer["partition"]) == ("l40s", "ou_bcs_low")
    assert all(r.server_batch_id.startswith(r.endpoint_id + "@") for r in rows)
    # Each row carries its own phase's attempt ledger and its own alias status: the merged
    # record concatenates the first and conjoins the second, and a per-request row that repeated
    # the merged values would double-count every retry in the ledger.
    assert sum(len(json.loads(r.attempts_json)) for r in rows) == len(record.attempts)
    assert [r.aliased for r in rows] == [False, False]
    assert not any(r.uncertain_execution for r in rows)
    for row in rows:
        T.project("requests", T.row_to_dict(row))             # the declared column whitelist


def test_request_rows_refuse_specs_that_are_not_this_records(nd_serve, nd_ctx, nd_contract,
                                                              nd_tokens):
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=4):
        record = INF.generate_decision(req, ctx=ctx)
    other = INF.request_spec(nd_request(nd_contract, nd_tokens, draw=7), ctx)
    with pytest.raises(Exception, match="not the specs"):
        TM.request_rows(record, req.key, cell_id="c", request_specs=(other,), producer=_producer())
    with pytest.raises(Exception, match="one row per physical request"):
        TM.request_rows(record, req.key, cell_id="c", request_specs=(), producer=_producer())


# --------------------------------------------------------------------------- metering


def test_metering_prices_the_discarded_span_and_reports_unused_budget(nd_serve, nd_ctx,
                                                                      nd_contract, nd_tokens):
    """Those tokens were decoded on a GPU; pricing them at zero is the ARD36 understatement."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64):
        record = INF.generate_decision(req, ctx=ctx)
    meter = TM.metering(record, nd_tokens)

    rows = TM.request_rows(record, req.key, cell_id="c",
                           request_specs=INF.request_specs_for(req, ctx, record),
                           producer=_producer())
    assert meter.calls == 2 == len(rows)
    assert meter.discarded_span_tokens > 0
    # The per-decision meter and the per-request ledger are the same number by construction (R1);
    # both are larger than the two spans by the two sampled reasoning delimiters (amendment A1).
    assert meter.generated_tokens == sum(r.completion_tokens for r in rows)
    assert meter.generated_tokens == (record.thinking_tokens + record.final_tokens
                                      + record.discarded_span_tokens + 2)
    assert meter.uncached_prompt_tokens == sum(
        r.prompt_tokens - r.cached_prefill_tokens for r in rows)
    assert meter.unused_thinking_tokens == 0          # the draw hit the cap
    assert meter.unused_final_tokens == nd_tokens.final_cap - record.final_tokens
    assert meter.flops_total == record.flops["total"] > 0
    assert set(meter.to_dict()) == set(TM.Metering.__dataclass_fields__)


def test_linear_compute_proxy_prices_only_what_the_engine_had_to_compute():
    """Spec 18: do not claim exact per-request FLOPs from a token count; this is the proxy."""
    assert TM.linear_compute_proxy(8_000_000_000, 100, 50) == 2 * 8_000_000_000 * 150
    assert TM.linear_compute_proxy(1, 0, 0) == 0
    with pytest.raises(Exception):
        TM.linear_compute_proxy(-1, 0, 0)


# --------------------------------------------------------------------------- attribution


def _request_row(request_hash, endpoint, arrival, latency, *, prompt=100, cached=0, completion=50):
    return T.RequestRow(
        request_hash=request_hash, generation_uid="g:x", phase_index=0, first_decision_id=None,
        prompt_token_hash="p", completion_token_hash="c", decoding_hash="d", seed_hex="s",
        model_hash="m", tokenizer_hash="t", engine_digest="e", adapter=K.ADAPTER_NATIVE_THINKING,
        endpoint_id=endpoint, replica_id=endpoint, host=endpoint.split(":")[0], gpu_type="l40s",
        partition="ou_bcs_low", server_batch_id=INF.server_batch_id(endpoint, arrival),
        arrival_ts=arrival, finish_ts=arrival + latency, attempts_json="[]",
        uncertain_execution=False, aliased=False, prompt_tokens=prompt, completion_tokens=completion,
        cached_prefill_tokens=cached, latency_s=latency, cell_id="cell", producer_json="{}")


def test_attributed_gpu_time_conserves_device_totals():
    """Spec 18.5.  Attribution is an approximation, so the report states the inequalities that
    make it auditable: attributed <= union busy <= allocated, per replica, and busy is the
    *union* of the request intervals -- a replica serving eight streams at once is busy once."""
    rows = [_request_row(f"r{i}", "node0:8884", 100.0 + i, 10.0) for i in range(4)]
    rows += [_request_row("r9", "node1:8884", 200.0, 5.0)]
    attributed, totals = TM.attribute(rows)

    assert {r["request_id"] for r in attributed} == {"r0", "r1", "r2", "r3", "r9"}
    assert all(r["attribution_rule"] == TM.ATTRIBUTION_RULE for r in attributed)
    node0 = totals["node0:8884"]
    assert node0["summed_request_seconds"] == pytest.approx(40.0)
    assert node0["server_busy_seconds"] == pytest.approx(13.0)      # 100.0 .. 113.0, union
    assert node0["attributed_seconds"] <= node0["server_busy_seconds"] + 1e-6

    report = TM.conservation_report(
        totals,
        replicas=[TM.ReplicaRef("node0:8884", "node0", "1"), TM.ReplicaRef("node1:8884", "node1", "2")],
        gpu_samples=[{"job_id": "1", "utilization": 88.0}, {"job_id": "1", "utilization": 92.0}],
        sacct_rows=[{"job_id": "1", "allocated_gpu_seconds": 3600.0},
                    {"job_id": "2", "allocated_gpu_seconds": 3600.0}])

    assert report["conserves"] is True and report["violations"] == []
    assert all(r["attributed_within_busy"] and r["busy_within_allocated"]
               for r in report["replicas"])
    assert report["replicas"][0]["coverage"] == "sampled"
    assert report["replicas"][0]["mean_utilization"] == pytest.approx(90.0)
    assert report["replicas"][1]["coverage"] == "allocated_only"    # never a silent NULL
    assert report["n_unsampled_replicas"] == 0
    assert report["attribution_rule"] == TM.ATTRIBUTION_RULE


def test_a_replica_that_was_never_sampled_is_recorded_as_unsampled_not_as_idle():
    rows = [_request_row("r0", "node0:8884", 100.0, 10.0)]
    _attributed, totals = TM.attribute(rows)
    report = TM.conservation_report(totals)
    assert report["replicas"][0]["coverage"] == "unsampled"
    assert report["replicas"][0]["allocated_gpu_seconds"] is None
    assert report["replicas"][0]["busy_within_allocated"] is None
    assert report["n_unsampled_replicas"] == 1


def test_a_violated_inequality_is_reported_not_raised():
    rows = [_request_row("r0", "node0:8884", 100.0, 3600.0)]
    _attributed, totals = TM.attribute(rows)
    report = TM.conservation_report(totals, replicas=[TM.ReplicaRef("node0:8884", "node0", "1")],
                                    sacct_rows=[{"job_id": "1", "allocated_gpu_seconds": 60.0}])
    assert report["conserves"] is False and report["violations"] == ["node0:8884"]


def test_two_phase_prefill_is_attributed_not_dropped(nd_serve, nd_ctx, nd_contract, nd_tokens):
    """Defect D-02, structurally.  The attribution reads ``requests``, so a continued decision
    contributes both physical requests: phase 2's prefill is priced, and the phase-1 span the
    continuation discarded is inside phase 1's completion tokens."""
    ctx = nd_ctx()
    req = nd_request(nd_contract, nd_tokens)
    with nd_serve(thinking_words=nd_tokens.thinking_cap + 64):
        record = INF.generate_decision(req, ctx=ctx)
    rows = TM.request_rows(record, req.key, cell_id="c",
                           request_specs=INF.request_specs_for(req, ctx, record),
                           producer=_producer())
    resources = TM.request_resources(rows, {r.request_hash: record.generation_uid for r in rows})

    assert len(resources) == len(record.physical_request_ids) == 2
    phase1, phase2 = resources
    assert phase2.prefill_tokens > 0                       # the re-sent prefix is real work
    assert phase2.prefill_tokens > phase1.prefill_tokens   # prompt + the whole thinking span
    # The discarded post-boundary span is inside phase 1's decode tokens, not lost: counting
    # only the merged completion would price phase 1 at the span the continuation kept.
    assert record.discarded_span_tokens > 0
    assert phase1.decode_tokens >= record.thinking_tokens + record.discarded_span_tokens
    merged_only = len(record.completion_token_ids) - len(record.continuation["prefill_token_ids"]) \
        - record.continuation["generated_tokens"]
    assert phase1.decode_tokens == merged_only + record.discarded_span_tokens

    attributed, totals = TM.attribute(rows)
    assert len(attributed) == 2
    assert sum(r["prefill_tokens"] for r in attributed) == phase1.prefill_tokens + phase2.prefill_tokens
    assert totals[record.endpoint_id]["n_requests"] == 2.0


def test_campaign_and_per_system_cost_are_separate():
    """Spec 18.5 / 16.1-16.2.  "What the study spent" and "what a protocol costs per delivered
    item" are different questions, so they are different functions over different groupings and
    neither can be quoted as the other."""
    def meter(flops, generated, calls=1, unused_thinking=0):
        return TM.Metering(calls=calls, thinking_tokens=10, final_tokens=5,
                           discarded_span_tokens=2, generated_tokens=generated, prompt_tokens=50,
                           cached_prefill_tokens=0, uncached_prompt_tokens=50, latency_s=1.0,
                           tool_elapsed_s=0.0, flops_prefill=flops // 2, flops_decode=flops // 2,
                           flops_total=flops, unused_thinking_tokens=unused_thinking,
                           unused_final_tokens=0)

    peer = [meter(100, 17), meter(100, 17)]
    private = [meter(60, 17)]
    unrun = [meter(40, 17)]
    attributed = [{"attributed_gpu_seconds": 2.0}, {"attributed_gpu_seconds": 3.0}]

    campaign = TM.campaign_cost(peer + private + unrun, attributed)
    systems = TM.per_system_cost({"PEER": peer, "PRIVATE": private, "UNDELIVERED": unrun},
                                 {"PEER": 2, "PRIVATE": 2, "UNDELIVERED": 0})

    assert campaign["scope"] == "campaign"
    assert campaign["n_decisions"] == 4 and campaign["flops_total"] == 300
    assert campaign["attributed_gpu_seconds"] == pytest.approx(5.0)
    assert campaign["discarded_span_tokens"] == 8          # counted, not dropped

    by_name = {row["system"]: row for row in systems}
    assert [row["scope"] for row in systems] == ["system"] * 3
    assert by_name["PEER"]["flops_per_delivered_item"] == 100.0
    assert by_name["PRIVATE"]["flops_per_delivered_item"] == 30.0
    # A system that delivered nothing has a null unit cost -- never a zero, which reads as free.
    assert by_name["UNDELIVERED"]["flops_per_delivered_item"] is None
    # The campaign total is not any system's number, and no system's is the campaign's.
    assert campaign["flops_total"] != by_name["PEER"]["flops_total"]
    assert "delivered_items" not in campaign and "attributed_gpu_seconds" not in by_name["PEER"]


# --------------------------------------------------------------------------- vendoring / sampler


def test_the_server_batch_id_bucket_matches_the_ard_original():
    """``server_batch_id`` is vendored from ``ard/resources.py:177-186`` (not on the
    ``ARD_IMPORTS`` allowlist), so it carries a parity test like every other vendored piece."""
    from agents_scaling.ard import resources as AR

    assert INF.ARRIVAL_BUCKET_S == AR.ARRIVAL_BUCKET_S
    for endpoint, ts in (("node0:8884", 0.0), ("node0:8884", 1.999), ("n:1", 1700000000.5),
                         ("n:1", -0.5)):
        assert INF.server_batch_id(endpoint, ts) == AR.server_batch_id(endpoint, ts)


def test_the_sampler_query_is_the_declared_field_list():
    """The sampler writes what ``parse_nvidia_smi_csv`` reads; the two cannot drift apart."""
    from agents_scaling.ard import resources as AR

    script = (REPO_ROOT / "slurm" / "nextdyn_gpu_sampler.sh").read_text()
    assert f"--query-gpu={','.join(AR.NVIDIA_SMI_FIELDS)}" in script
    assert "nextdyn_start_gpu_sampler" in script and "nextdyn_stop_gpu_sampler" in script
    assert "ard36_start_gpu_sampler" not in script       # a fork, not a second definition

    rows = TM.parse_nvidia_smi_csv(
        "1700000000,GPU-abc,NVIDIA L40S,46068,20480,88,210.5\n", host="node0", job_id="1")
    assert rows[0]["gpu_type"] == "NVIDIA L40S" and rows[0]["utilization"] == 88.0
    assert rows[0]["job_id"] == "1" and rows[0]["node"] == "node0"
