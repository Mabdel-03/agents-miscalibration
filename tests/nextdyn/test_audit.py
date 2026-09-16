"""WP5: the SELECTION_AUDIT cell (spec 6.5, 15.1; plan §1.9).

ARD36 ran a verifier and never joined its output, so "how much better than plurality is a
verifier?" was unanswerable from the tables.  The tests here pin the join: the cell dispatches
every generative selector in the registry, writes one ``selections`` row per (selector, pool)
with the audit cell id on it, and turns an unparseable score into a typed fallback rather than
into evidence about the candidate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from agents_scaling.nextdyn import audit as AU
from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids as I
from agents_scaling.nextdyn import inference as INF
from agents_scaling.nextdyn import schema
from agents_scaling.nextdyn import selectors as S
from agents_scaling.nextdyn.errors import ContractViolation, ManifestError
from agents_scaling.nextdyn.ids import CellKind
from tests.nextdyn.test_mod97 import CONFIG_DIR
from tests.nextdyn.test_selectors import SEED, candidate, ctx, pool

AUDIT_PROFILE = "nd1-32B"
SPLIT_ROLE = "pilot"


@dataclass(frozen=True)
class FakeItem:
    source_item_id: str = "nditem0000000001"
    item_variant_id: str = "nditem0000000001.v0"
    rendered_question: str = "A question."


def bodies(sealed) -> dict[str, str]:
    return {c.packet_hash: json.dumps({"from": f"agent-{c.slot}", "final_answer": c.canonical_class})
            for c in sealed.valid_candidates}


class Runner:
    """The WP4 runner surface the SELECTION_AUDIT worker reads, and nothing more."""

    def __init__(self, contract, generate, sealed, *, selection_ctx=None, registry=None):
        self.cell_id = "selection_audit/test"
        self.contract = contract
        self.generate = generate
        self.split_role = SPLIT_ROLE
        self.top_logprobs = 1
        self._pools = tuple(sealed)
        self.selection_ctx = selection_ctx or ctx()
        self.selector_registry = registry

    def audit_pools(self, source_item_id):
        return self._pools

    def audit_bodies(self, source_item_id):
        out = {}
        for sealed in self._pools:
            out.update(bodies(sealed))
        return out

    def audit_item(self, source_item_id):
        return FakeItem()


def audit_responder(value="0.8"):
    def respond(payload):
        if "prompt_token_ids" in payload:
            return json.dumps({"p_correct": float(value)}) if value is not None else "no score"
        return json.dumps({"p_correct": float(value)}) if value is not None else "no score"

    return respond


# --------------------------------------------------------------------------- parsing


def test_the_pointwise_audit_parses_a_strict_p_correct_object():
    """One parser, not two.  ``parse_score`` is the single declaration of the ``p_correct``
    rule (R1): a second entry point for the same fact drifts the moment one of them is fixed.
    """
    assert not hasattr(AU, "parse_audit")
    assert AU.parse_score('{"p_correct": 0.75}').value == pytest.approx(0.75)
    assert AU.parse_score('{"p_correct": 0}').value == 0.0
    for text in ("", "0.75", "the candidate looks right", '{"p": 0.5}', '{"p_correct": "0.5"}',
                 '{"p_correct": true}', '{"p_correct": 1.5}', '{"p_correct": -0.1}'):
        assert isinstance(AU.parse_score(text), S.Failure)


def test_the_audit_row_says_why_the_verifier_failed_and_keeps_confidence_out_of_it():
    """Two facts the audit row used to blur.

    (1) Three typed reasons got one ``invalid_json`` status, so a row could not say whether the
    verifier wrote prose or wrote a clean object with no score -- and the typed reason the
    parser carefully built was thrown away.  (2) The score went into ``confidence``, which is
    spec 13.1's *self-reported confidence of the model that produced the answer*; a verifier's
    judgement about someone else's answer is a different quantity and merging them would make
    any calibration panel that reads that column report a mixture.
    """
    statuses = {reason: AU.parse_status_of(reason) for reason in
                (AU.FAILURE_NO_OBJECT, AU.FAILURE_MISSING_FIELD, AU.FAILURE_NOT_A_NUMBER)}
    assert len(set(statuses.values())) == 3
    assert statuses[AU.FAILURE_NO_OBJECT] is schema.ParseStatus.NO_JSON

    class Record:
        final_text = '{"p_correct": 0.9}'

    scored = AU._audit_output(Record(), AU.parse_score(Record.final_text, packet_hash="pkt"))
    assert scored.parse_status is schema.ParseStatus.OK and scored.valid is True
    assert float(scored.final_answer) == pytest.approx(0.9)
    assert scored.confidence is None

    for text, reason in (("prose", AU.FAILURE_NO_OBJECT), ('{"p": 1}', AU.FAILURE_MISSING_FIELD),
                         ('{"p_correct": 2}', AU.FAILURE_NOT_A_NUMBER)):
        record = type("R", (), {"final_text": text})()
        out = AU._audit_output(record, AU.parse_score(text, packet_hash="pkt"))
        assert out.parse_status is statuses[reason]
        assert out.final_answer is None and out.confidence is None


def test_a_failed_score_is_a_typed_execution_status_never_a_semantic_label():
    """Spec 6.5 / D9.  "The verifier did not answer" and "the candidate is wrong" are
    different facts; conflating them would make a flaky verifier look like a hard pool."""
    failures = [AU.parse_score(text, packet_hash="pkt") for text in
                ("prose", '{"p": 1}', '{"p_correct": "x"}')]
    assert [f.reason for f in failures] == [AU.FAILURE_NO_OBJECT, AU.FAILURE_MISSING_FIELD,
                                            AU.FAILURE_NOT_A_NUMBER]
    assert all(isinstance(f, S.Failure) for f in failures)
    assert all(not hasattr(f, "correct") for f in failures)

    ok = AU.parse_score('{"p_correct": 0.6}', packet_hash="pkt")
    assert isinstance(ok, S.Score) and ok.value == pytest.approx(0.6)


# --------------------------------------------------------------------------- requests


def test_one_request_per_unique_packet_hash(nd_contract):
    """Spec 6.5.  Two agents that produced byte-identical packets are one object to the
    verifier: the audit costs distinct evidence, not team size."""
    shared = S.SealedPool(
        source_item_id=FakeItem().source_item_id, episode_id="episode|t|c|m|b|rep0",
        cell_kind=CellKind.EPISODE, round_=4, scope="terminal_only", protocol_hash="cond-a",
        candidates=(candidate(0, "A", packet="pkt-same"), candidate(1, "A", packet="pkt-same"),
                    candidate(2, "B", packet="pkt-other"), candidate(3, None)))
    plans = AU.audit_plans([shared])
    assert [p.packet_hash for p in plans] == ["pkt-same", "pkt-other"]
    requests = AU.audit_requests([shared], item=FakeItem(), bodies=bodies(shared),
                                 contract=nd_contract, split_role=SPLIT_ROLE, top_logprobs=1)
    assert set(requests) == {"pkt-same", "pkt-other"}
    assert len(requests) < len(shared.valid_candidates)


def test_the_verifier_sees_one_candidate_and_no_correctness_cue(nd_contract):
    """Spec 6.5.  Candidate confidence, endorsement counts, the source slot and every
    correctness signal are absent from the audit prompt, and the verifier is told not to
    synthesize a new answer."""
    sealed = S.SealedPool(
        source_item_id=FakeItem().source_item_id, episode_id="episode|t|c|m|b|rep0",
        cell_kind=CellKind.EPISODE, round_=4, scope="terminal_only", protocol_hash="cond-a",
        candidates=(candidate(0, "A", confidence=0.95), candidate(1, "B", confidence=0.05)))
    requests = AU.audit_requests([sealed], item=FakeItem(), bodies=bodies(sealed),
                                 contract=nd_contract, split_role=SPLIT_ROLE, top_logprobs=1)
    for packet_hash, request in requests.items():
        text = "\n".join(m["content"] for m in request.messages)
        assert AU.P_CORRECT in text
        assert "0.95" not in text and "0.05" not in text
        assert "confidence" not in text.lower()
        assert "agent_slot" not in text and "endors" not in text.lower()
        assert sum(text.count(json.dumps(body)[:20]) for body in ()) == 0
        assert text.count("CANDIDATE") == 1


def test_the_audit_key_is_the_pool_and_never_the_audited_episode(nd_contract):
    """Contract §2.4.  The audit row's ``root_team_id`` is the sealed pool and its phase is
    ``audit``, so an audit row can never be mistaken for the decision it scores (D-11's shape)."""
    sealed = pool("A", "B")
    requests = AU.audit_requests([sealed], item=FakeItem(), bodies=bodies(sealed),
                                 contract=nd_contract, split_role=SPLIT_ROLE, top_logprobs=1)
    model = nd_contract.model(nd_contract.selection.audit_model)
    for request in requests.values():
        key = request.key
        assert key.cell_kind is CellKind.SELECTION_AUDIT
        assert key.root_team_id == sealed.pool_id
        assert key.protocol_hash == sealed.protocol_hash
        assert key.receiver_model_hash == I.model_hash(model)
        assert key.budget_hash == nd_contract.audit_tokens.budget_hash()
        assert key.phase == K.PHASE_AUDIT and key.round == sealed.round_
        episode = I.audit_episode_id(sealed.pool_id, I.model_hash(model), key.budget_hash)
        assert episode.startswith(f"{CellKind.SELECTION_AUDIT.value}|")
        assert episode != sealed.episode_id


def test_a_packet_body_that_was_not_supplied_is_refused(nd_contract):
    sealed = pool("A", "B")
    with pytest.raises(ContractViolation):
        AU.audit_requests([sealed], item=FakeItem(), bodies={}, contract=nd_contract,
                          split_role=SPLIT_ROLE, top_logprobs=1)


# --------------------------------------------------------------------------- the cell


def audit_run(nd_contract, nd_ctx, nd_serve, nd_pool, sealed, *, value="0.8", selection_ctx=None,
              registry=None):
    ctx_obj = nd_ctx(cell_id="selection_audit/test", pool=nd_pool(AUDIT_PROFILE))
    with nd_serve(profile=AUDIT_PROFILE, responder=audit_responder(value)):
        runner = Runner(nd_contract, lambda req: INF.generate_decision(req, ctx=ctx_obj), sealed,
                        selection_ctx=selection_ctx, registry=registry)
        return AU.audit_worker(runner, FakeItem().source_item_id)


def test_the_cell_writes_one_decision_per_packet_and_one_selection_per_selector(
        nd_contract, nd_ctx, nd_serve, nd_pool):
    sealed = pool("A", "A", "B")
    item = audit_run(nd_contract, nd_ctx, nd_serve, nd_pool, [sealed])

    assert item.cell_kind is CellKind.SELECTION_AUDIT
    decisions = item.rows["decisions"]
    assert len(decisions) == len(sealed.unique_packet_hashes) == 3
    assert {d.phase for d in decisions} == {K.PHASE_AUDIT}
    assert {d.cell_kind for d in decisions} == {CellKind.SELECTION_AUDIT.value}
    assert all(d.canonical_class is None for d in decisions)     # R7: grades are writer-side

    selections = item.rows["selections"]
    assert [s.selector for s in selections] == [S.POINTWISE_AUDIT]
    row = selections[0]
    assert row.pool_id == sealed.pool_id and row.pool_scope == sealed.scope
    assert row.audit_cell_id == AU.audit_cell_id(S.POINTWISE_AUDIT, S.SELECTOR_VERSION,
                                                 sealed.pool_id)
    assert f"{S.POINTWISE_AUDIT}.{S.SELECTOR_VERSION}" in row.selection_id
    assert row.fallback_status == S.FALLBACK_NONE
    assert row.selected_decision_id in {c.candidate_id for c in sealed.valid_candidates}
    assert row.cost_tokens > 0


def test_an_unparseable_score_falls_back_to_the_preregistered_vote(
        nd_contract, nd_ctx, nd_serve, nd_pool):
    """D9.  The fallback is the study's declared ``audit_failure_fallback`` -- the current-round
    plurality -- recorded as a status, and the audit rows still exist so the failure is visible.
    """
    sealed = pool("A", "A", "B")
    item = audit_run(nd_contract, nd_ctx, nd_serve, nd_pool, [sealed], value=None)
    row = item.rows["selections"][0]
    assert row.fallback_status == S.FALLBACK_AUDIT
    assert nd_contract.selection.audit_failure_fallback == S.CURRENT_ROUND_PLURALITY
    assert row.selected_class == "A"
    assert len(item.rows["decisions"]) == 3


def test_a_pool_sealed_at_another_scope_is_refused(nd_contract, nd_ctx, nd_serve, nd_pool):
    """R1: the audited population is ``selection.pool_scope`` in ``study.yaml``, once."""
    sealed = pool("A", "B", scope="full_archive")
    with pytest.raises(ManifestError):
        audit_run(nd_contract, nd_ctx, nd_serve, nd_pool, [sealed])


def test_an_item_with_no_sealed_pool_produces_an_empty_but_present_item_file(nd_contract):
    runner = Runner(nd_contract, lambda req: pytest.fail("no pool, no generation"), [])
    item = AU.audit_worker(runner, FakeItem().source_item_id)
    assert item.rows["decisions"] == () and item.rows["selections"] == ()
    assert item.status.value == "complete"


def test_a_second_generative_selector_gets_its_own_cell_id(nd_contract):
    """Plan §1.10: a new generative selector is one registry entry; the cell id and the
    selection id both carry ``<selector>.<version>`` so the two never collide."""
    sealed = pool("A", "B")
    first = AU.audit_cell_id(S.POINTWISE_AUDIT, "v1", sealed.pool_id)
    second = AU.audit_cell_id("pairwise_audit", "v1", sealed.pool_id)
    third = AU.audit_cell_id(S.POINTWISE_AUDIT, "v2", sealed.pool_id)
    assert len({first, second, third}) == 3
    assert all(cell.startswith(f"{CellKind.SELECTION_AUDIT.value}/") for cell in (first, second))


def test_the_audit_model_and_budget_come_from_study_yaml():
    contract = C.contract_of(C.load_study(CONFIG_DIR / "study.yaml"))
    assert contract.selection.audit_model == "scale"
    assert contract.audit_tokens.thinking_cap == 1024
    assert contract.audit_tokens.final_cap == 256
    assert contract.selection.pool_scope == "terminal_only"


# --------------------------------------------------------------------------- routing


class SecondVerifier(S._BaseSelector):
    """A pairwise verifier with its **own** prompt and its own output format.

    It exists to prove the cell routes rather than hard-codes: its requests carry a sentinel
    the pointwise builder would never emit, its parser reads a format ``audit.parse_score``
    would reject, and it records what it was handed so the test can check it was asked at all.
    """

    name = "pairwise_audit"
    needs_cells = True
    SENTINEL = "PAIRWISE PROMPT SENTINEL"

    def __init__(self, contract, item, bodies_by_hash):
        self.contract = contract
        self.item = item
        self.bodies = dict(bodies_by_hash)
        self.built_for: list[str] = []
        self.parsed: list[str] = []
        self.scored_with: list[dict] = []

    def audit_requests(self, sealed, context):
        from dataclasses import replace as _replace

        self.built_for.append(sealed.pool_id)
        base = AU.audit_requests([sealed], item=self.item, bodies=self.bodies,
                                 contract=self.contract, split_role=SPLIT_ROLE, top_logprobs=1)
        out = {}
        for packet_hash, request in base.items():
            # its own key (replicate 1) so its rows never collide with the pointwise ones
            key = _replace(request.key, replicate_id=1)
            messages = ({"role": "user", "content": f"{self.SENTINEL}\n{self.item.rendered_question}"},)
            out[packet_hash] = _replace(request, key=key, messages=messages,
                                        seed_key=I.seed_key_for(key, split_role=SPLIT_ROLE))
        return out

    def parse_score(self, record):
        """Its own format: this verifier reports a *risk*, so the score is one minus it."""
        text = str(getattr(record, "final_text", record) or "")
        self.parsed.append(text)
        value = json.loads(text)["p_correct"] if AU.P_CORRECT in text else 0.0
        return S.Score(packet_hash="", value=round(1.0 - float(value), 6))

    def select(self, sealed, context):
        self.scored_with.append(dict(context.scores))
        best = max(sealed.valid_candidates,
                   key=lambda c: context.scores.get(c.packet_hash, -1.0))
        return S.Selection(selector=self.name, version=self.version, pool_id=sealed.pool_id,
                           selected_id=best.candidate_id, selected_class=best.canonical_class,
                           tie_class_ids=(), fallback_status=S.FALLBACK_NONE, cost={},
                           failed=False)


def test_a_second_generative_selector_is_asked_for_its_own_requests_and_parser(
        nd_contract, nd_ctx, nd_serve, nd_pool):
    """Plan §1.9: "every generative selector owns its own request builder and parser".

    The defect this closes: the cell called the module-level pointwise builder and parser for
    *every* ``needs_cells`` selector, so a pairwise verifier would have been handed pointwise
    ``templates/audit.txt`` prompts and pointwise scores while ``bound_selector`` sat unused --
    and the extension test only exercised ``select()``, so the whole routing claim could be
    deleted without a failure.
    """
    sealed = pool("A", "A", "B")
    second = SecondVerifier(nd_contract, FakeItem(), bodies(sealed))
    registry = {**S.SELECTORS, second.name: second}
    item = audit_run(nd_contract, nd_ctx, nd_serve, nd_pool, [sealed], registry=registry)

    # it was asked for its own requests, and they were its own bytes
    assert second.built_for == [sealed.pool_id]
    assert second.parsed and len(second.parsed) == len(sealed.unique_packet_hashes)
    requests_seen = {d.request_hashes_json for d in item.rows["decisions"]}
    assert len(requests_seen) == 2 * len(sealed.unique_packet_hashes), \
        "two verifiers sent the same physical request"
    assert AU.P_CORRECT in "".join(second.parsed)          # it read the served bytes itself
    assert len(item.rows["decisions"]) == 2 * len(sealed.unique_packet_hashes)
    assert len({d.decision_id for d in item.rows["decisions"]}) == len(item.rows["decisions"])

    # ... and its own parser's numbers, not the pointwise ones
    assert second.scored_with == [{h: pytest.approx(0.2) for h in sealed.unique_packet_hashes}]
    selections = {row.selector: row for row in item.rows["selections"]}
    assert set(selections) == {S.POINTWISE_AUDIT, second.name}
    assert selections[second.name].audit_cell_id == AU.audit_cell_id(
        second.name, S.SELECTOR_VERSION, sealed.pool_id)
    assert selections[S.POINTWISE_AUDIT].audit_cell_id != selections[second.name].audit_cell_id


def test_two_selectors_that_issue_the_same_request_key_are_refused(
        nd_contract, nd_ctx, nd_serve, nd_pool):
    """A second verifier that reused the pointwise key would write two prompts into one
    ``decisions`` row -- ARD36's D-11 shape.  The cell says so instead of overwriting."""
    sealed = pool("A", "B")

    class Colliding(SecondVerifier):
        def audit_requests(self, pool_, context):          # the pointwise key, unchanged
            return AU.audit_requests([pool_], item=self.item, bodies=self.bodies,
                                     contract=self.contract, split_role=SPLIT_ROLE,
                                     top_logprobs=1)

    clash = Colliding(nd_contract, FakeItem(), bodies(sealed))
    with pytest.raises(ContractViolation):
        audit_run(nd_contract, nd_ctx, nd_serve, nd_pool, [sealed],
                  registry={**S.SELECTORS, clash.name: clash})


def test_the_cells_scoring_calls_go_out_together_rather_than_one_at_a_time(
        nd_contract, nd_ctx, nd_serve, nd_pool):
    """R2's behavioural companion: ``max_inflight`` must govern the thing it names.

    The static consumer check was satisfied -- ``worker_concurrency`` read the knob -- while
    the worker looped over its packets calling ``generate`` one at a time and never used it.
    That is D-03, and a read-check alone cannot see it, which is why the plan asks for a
    "peak in-flight" companion.  The audit block is the largest single load on the 32B lane,
    which the plan already calls the critical path.
    """
    import threading

    sealed = pool("A", "B", "C", "D")
    n_packets = len(bodies(sealed))
    assert n_packets >= 2, "this test needs a pool with several distinct packets"
    ctx_obj = nd_ctx(cell_id="selection_audit/test", pool=nd_pool(AUDIT_PROFILE))
    # A barrier rather than a peak counter: every call must be in flight *at the same time* or
    # the wait times out, so the test states the property instead of racing to observe it.
    barrier = threading.Barrier(n_packets, timeout=10)

    def counting(request):
        barrier.wait()
        return INF.generate_decision(request, ctx=ctx_obj)

    with nd_serve(profile=AUDIT_PROFILE, responder=audit_responder("0.8")):
        runner = Runner(nd_contract, counting, [sealed])
        runner.concurrency = type("C", (), {"max_inflight": n_packets})()
        AU.audit_worker(runner, FakeItem().source_item_id)
    assert not barrier.broken, f"the audit's {n_packets} scoring calls were issued serially"
