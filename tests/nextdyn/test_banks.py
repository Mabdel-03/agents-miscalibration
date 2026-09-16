"""Typed banks, the extension rule and the difficulty estimator (spec 5.3, 8.3; plan §2.3, D14).

ARD36's single bank doubled as difficulty, donors and PRIVATE round 0 (defect D-11).  The
tests here pin the three properties that make that impossible: a bank draw carries a bank id
and a bank kind, every draw is an independent sample with its own ``generation_uid``, and a
difficulty draw is never a donor.
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.nextdyn import banks as BK
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn import ids as ID
from agents_scaling.nextdyn import items as I
from agents_scaling.nextdyn import schema, tables
from agents_scaling.nextdyn.errors import ContractViolation, IdentityError, ManifestError
from agents_scaling.nextdyn.ids import CellKind

ITEM = "itembank00000001"
VARIANT = "itembank00000001.v0"
CELL = "donor_bank/test/0000"


# --------------------------------------------------------------------------- fixtures


def tokens() -> K.TokenContract:
    return K.TokenContract(thinking_cap=16, final_cap=32, served_window=4096, reserve=8,
                           packet_cap=64, rationale_cap=32, claims_max=4, claim_cap=16)


def public_output(answer: str = "4") -> schema.PublicOutput:
    text = json.dumps({"reasoning": "because", "shareable_claims": ["one"],
                       "final_answer": answer, "confidence": 0.5})
    return schema.parse_public_output(text, "stop", (1, 2, 3), lambda s: len(s.split()),
                                      limits=tokens())


def record(*request_ids: str, final_text: str | None = None):
    """A :class:`DecisionRecord` with every field filled, for the one-site row builder."""
    from agents_scaling.nextdyn.inference import DecisionRecord

    ids = tuple(request_ids)
    text = final_text if final_text is not None else json.dumps(
        {"reasoning": "because", "shareable_claims": ["one"], "final_answer": "4",
         "confidence": 0.5})
    return DecisionRecord(
        request_ids=ids, physical_request_ids=ids, generation_uid=ID.generation_uid(ids),
        prompt_token_ids=(1, 2), completion_token_ids=(3, 4, 5), native_reasoning_text="think",
        final_text=text, thinking_tokens=4, final_tokens=3, boundary_injected=False,
        natural_stop=True, cap_hit=False, final_overrun=False, answer_logprob=-0.5,
        final_top_logprobs=None, prompt_tokens=2, cached_prefill_tokens=0, finish_reason="stop",
        latency_s=0.1, endpoint_id="ep0", replica_id="r0", server_batch_id="b0", arrival_ts=1.0,
        gpu_type="a100", partition="mit_normal", host="node0",
        flops={"prefill": 1, "decode": 2, "total": 3}, continuation=None,
        discarded_span_tokens=0, aliased=False, uncertain_execution=False, attempts=())


def bank_row(role: BK.BankRole, draw: int, *, uid: str | None = None, validity: bool = True,
             content: str | None = None) -> tables.BankRow:
    return tables.row_from_dict(tables.BankRow, {
        "bank_draw_id": f"{BK.ROLE_KINDS[role].value}|{ITEM}|{VARIANT}|m" + f"|k{draw}",
        "decision_id": f"d:{draw:040d}", "generation_uid": uid or f"g:{draw:064d}",
        "cell_kind": BK.ROLE_KINDS[role].value, "bank_role": role.value,
        "source_item_id": ITEM, "item_variant_id": VARIANT, "model_hash": "m" * 16,
        "budget_hash": "b" * 64, "draw_index": draw, "extension_stage": 0,
        "request_hash": f"r{draw}", "content_hash": content or f"c:{draw:064d}",
        "parse_status": "ok", "validity": validity, "cell_id": CELL})


# --------------------------------------------------------------------------- roles and draws


def test_the_three_roles_map_one_to_one_onto_the_bank_kinds():
    assert set(BK.ROLE_KINDS) == set(BK.BankRole)
    assert set(BK.ROLE_KINDS.values()) == ID.BANK_KINDS
    for role, kind in BK.ROLE_KINDS.items():
        assert BK.role_of(kind) is role
    with pytest.raises(ManifestError):
        BK.role_of(CellKind.EPISODE)


def test_draw_ranges_follow_the_bank_spec_and_the_extension_continues_the_index_space(study_cfg):
    banks = study_cfg.contract.banks
    assert list(BK.draw_range(BK.BankRole.DIFFICULTY, banks)) == list(range(banks.difficulty_draws))
    assert list(BK.draw_range(BK.BankRole.NATURAL_ROOT, banks)) == list(range(banks.natural_root_draws))
    initial = list(BK.draw_range(BK.BankRole.DONOR, banks))
    extended = list(BK.draw_range(BK.BankRole.DONOR, banks, extension_stage=BK.STAGE_EXTENDED))
    assert initial == list(range(banks.donor_initial))
    assert extended == list(range(banks.donor_initial, banks.donor_max))
    assert not set(initial) & set(extended)
    with pytest.raises(ContractViolation):
        BK.draw_range(BK.BankRole.DONOR, banks, extension_stage=2)


def test_bank_keys_carry_the_per_kind_layout_of_the_contract(study_cfg):
    contract = study_cfg.contract
    model = contract.model("primary")
    key = BK.bank_key(BK.BankRole.DONOR, ITEM, VARIANT, model, tokens(), 7,
                      study_id=contract.study_id)
    assert key.root_team_id == "" and key.protocol_hash == ""
    assert key.phase == K.PHASE_BANK and key.round == 0 and key.replicate_id == 0
    assert key.agent_slot == 7
    assert key.receiver_model_hash == ID.model_hash(model)
    assert key.budget_hash == tokens().budget_hash()
    assert BK.bank_draw_id(BK.BankRole.DONOR, key).startswith(CellKind.DONOR_BANK.value + "|")


def test_bank_role_cannot_alias_private_episode(study_cfg):
    """Spec 18.1 / defect D-11: a bank draw and an episode row can never share an id."""
    contract = study_cfg.contract
    model = contract.model("primary")
    key = BK.bank_key(BK.BankRole.DIFFICULTY, ITEM, VARIANT, model, tokens(), 0,
                      study_id=contract.study_id)
    draw_id = BK.bank_draw_id(BK.BankRole.DIFFICULTY, key)
    assert draw_id.startswith(CellKind.DIFFICULTY_BANK.value + "|")
    with pytest.raises(IdentityError):
        ID.episode_id(CellKind.DIFFICULTY_BANK, "team:x" * 3, "cond", "m" * 16, "b" * 8, 0)
    with pytest.raises(IdentityError):
        ID.bank_draw_id(CellKind.EPISODE, ITEM, VARIANT, "m" * 16, "b" * 8, 0)
    # a bank key may not pretend to belong to a team, and an episode key may not be a bank draw
    with pytest.raises(IdentityError):
        ID.CompositeKey(study_id=contract.study_id, cell_kind=CellKind.DONOR_BANK,
                        source_item_id=ITEM, item_variant_id=VARIANT, root_team_id="team:abc",
                        protocol_hash="", receiver_model_hash="m" * 16, budget_hash="b" * 64,
                        replicate_id=0, round=0, agent_slot=0, phase=K.PHASE_BANK)
    # and the view refuses a row whose kind is not the role's
    decision = tables.row_from_dict(tables.DecisionRow, dict(
        tables.row_to_dict(_decision_row(study_cfg, BK.BankRole.DIFFICULTY, 0))))
    with pytest.raises(ManifestError):
        BK.bank_row(BK.BankRole.DONOR, key, decision, extension_stage=0, cell_id=CELL)


def _decision_row(study_cfg, role: BK.BankRole, draw: int):
    from agents_scaling.nextdyn import telemetry

    contract = study_cfg.contract
    model = contract.model("primary")
    key = BK.bank_key(role, ITEM, VARIANT, model, tokens(), draw, study_id=contract.study_id)
    return telemetry.decision_row(
        record(f"req-{role.value}-{draw}"), key, public_output(),
        cell=telemetry.CellContext(cell_id=CELL, episode_id=BK.bank_draw_id(role, key),
                                   display_label=None, bundle_export_hash=None),
        parents=(), seen_transmissions=(), aliased=False, reuse_of=None)


def test_independent_draws_have_distinct_generation_uids(study_cfg):
    """Spec 18.1: k draws are k samples, whatever they say."""
    rows = [_decision_row(study_cfg, BK.BankRole.DONOR, draw) for draw in range(8)]
    assert len({row.generation_uid for row in rows}) == 8
    assert len({row.decision_id for row in rows}) == 8
    assert {row.phase for row in rows} == {K.PHASE_BANK}
    assert {row.root_team_id for row in rows} == {""}
    same_text = [_decision_row(study_cfg, BK.BankRole.DONOR, draw) for draw in (0, 1)]
    assert same_text[0].content_hash == same_text[1].content_hash
    assert same_text[0].generation_uid != same_text[1].generation_uid


# --------------------------------------------------------------------------- difficulty


def grades(states: str) -> dict[str, str]:
    return {f"g:{index:064d}": state for index, state in enumerate(states)}


def test_difficulty_separates_resolved_operational_and_bounded_rates():
    rows = [bank_row(BK.BankRole.DIFFICULTY, draw) for draw in range(16)]
    estimate = BK.difficulty(rows, grades("CCCWWWWWWWWWWUUU"), cutpoints=(0.2, 0.5, 0.8),
                             ci_alpha=0.05)
    assert (estimate.n_draws, estimate.c, estimate.w, estimate.u) == (16, 3, 10, 3)
    assert estimate.g_resolved == 13
    assert estimate.raw == pytest.approx(3 / 13)
    assert estimate.operational == pytest.approx(3 / 16)
    assert estimate.lo == pytest.approx(3 / 16) and estimate.hi == pytest.approx(6 / 16)
    assert estimate.smoothed == pytest.approx(3.5 / 14)
    assert estimate.bin == BK.difficulty_bin(estimate.smoothed, (0.2, 0.5, 0.8))
    assert estimate.beta_ci_lo <= estimate.raw <= estimate.beta_ci_hi
    assert estimate.bank_role == BK.BankRole.DIFFICULTY.value


def test_an_unlabelled_draw_is_unresolved_exactly_like_an_explicit_u():
    rows = [bank_row(BK.BankRole.DIFFICULTY, draw) for draw in range(4)]
    explicit = BK.difficulty(rows, grades("CCUU"), cutpoints=(0.5,), ci_alpha=0.05)
    implicit = BK.difficulty(rows, grades("CC"), cutpoints=(0.5,), ci_alpha=0.05)
    assert (explicit.c, explicit.u, explicit.raw) == (implicit.c, implicit.u, implicit.raw)


def test_difficulty_bins_and_intervals_are_frozen_and_bounded():
    assert BK.difficulty_bin(0.0, (0.2, 0.5, 0.8)) == 0
    assert BK.difficulty_bin(0.2, (0.2, 0.5, 0.8)) == 1
    assert BK.difficulty_bin(1.0, (0.2, 0.5, 0.8)) == 3
    with pytest.raises(ContractViolation):
        BK.difficulty_bin(0.5, (0.8, 0.2))
    assert BK.beta_ci(0, 0, alpha=0.05) == (0.0, 1.0)      # no resolved draw is no information
    assert BK.beta_ci(16, 16, alpha=0.05)[1] == 1.0
    with pytest.raises(ContractViolation):
        BK.beta_ci(1, 2, alpha=0.0)


def test_difficulty_refuses_a_mixture_of_items_or_roles():
    mixed = [bank_row(BK.BankRole.DIFFICULTY, 0), bank_row(BK.BankRole.DONOR, 1)]
    with pytest.raises(ContractViolation):
        BK.difficulty(mixed, grades("CC"), cutpoints=(0.5,), ci_alpha=0.05)
    with pytest.raises(ContractViolation):
        BK.difficulty([], {}, cutpoints=(0.5,), ci_alpha=0.05)
    table = BK.difficulty_table(mixed, grades("CC"), cutpoints=(0.5,), ci_alpha=0.05)
    assert {row.bank_role for row in table} == {"difficulty", "donor"}


def test_the_difficulty_cutpoints_come_from_the_analysis_profile(analysis_cfg):
    assert BK.cutpoints_of(analysis_cfg.profile) == tuple(analysis_cfg.profile.difficulty_cutpoints)


# --------------------------------------------------------------------------- extension rule


def test_the_extension_rule_is_frozen_and_stage_bound():
    rule = BK.ExtensionRule(name="extend_once_when_near_eligible_v1", min_correct_resolved=1,
                            min_wrong_resolved=1)
    assert BK.extension_authorized(rule, stage=0, paired_eligible=False, n_correct=1, n_wrong=1)
    assert not BK.extension_authorized(rule, stage=0, paired_eligible=True, n_correct=8, n_wrong=8)
    assert not BK.extension_authorized(rule, stage=1, paired_eligible=False, n_correct=1, n_wrong=1)
    assert not BK.extension_authorized(rule, stage=0, paired_eligible=False, n_correct=0, n_wrong=4)
    with pytest.raises(ContractViolation):
        BK.extension_authorized(BK.ExtensionRule(name="", min_correct_resolved=1,
                                                 min_wrong_resolved=1),
                                stage=0, paired_eligible=False, n_correct=1, n_wrong=1)


# --------------------------------------------------------------------------- the worker


class StubRunner:
    """The runner surface ``bank_worker`` documents (WP4 supplies the real one)."""

    def __init__(self, study_cfg, item, kind, *, extension_stage=0):
        self.contract = study_cfg.contract
        self.model = self.contract.model("primary")
        self.tokens = tokens()
        self.top_logprobs = 1
        self.run_root = "/nonexistent"
        self.cell = type("Cell", (), {"cell_id": CELL, "kind": kind,
                                      "options": {"extension_stage": extension_stage}})()
        self._item = item
        self.requests = []

    def item(self, source_item_id):
        assert source_item_id == self._item.source_item_id
        return self._item

    def tokenizer_for(self, model):
        return lambda text: len(text.split())

    def generate(self, request):
        self.requests.append(request)
        return record(f"req-{request.key.agent_slot}")


def stub_item() -> I.SourceItem:
    variant = I.ItemVariant(item_variant_id=VARIANT, source_item_id=ITEM,
                            variant_kind=I.VARIANT_CANONICAL, permutation=(), options=(),
                            task_text="Compute one plus one.", rendered_task_hash="h",
                            prompt_hash="p")
    return I.SourceItem(source_item_id=ITEM, family=F.MATH500, task_family="MATH",
                        answer_format=F.ANSWER_FORMAT_EXACT, n_options=None,
                        answer_instruction=F.family(F.MATH500).answer_instruction,
                        subject=None, level=None, split_role="pilot",
                        prior_use_status="unknown", normalized_text_hash="n", grader_hash="g",
                        gold_location="protected/gold/MATH500.jsonl#item", variants=(variant,))


def test_the_bank_worker_writes_one_decision_and_one_view_row_per_draw(study_cfg):
    runner = StubRunner(study_cfg, stub_item(), CellKind.DONOR_BANK)
    item_file = BK.bank_worker(runner, ITEM)
    banks = study_cfg.contract.banks
    decisions = item_file.rows["decisions"]
    views = item_file.rows["banks"]
    assert len(decisions) == len(views) == banks.donor_initial
    assert [row.draw_index for row in views] == list(range(banks.donor_initial))
    assert len({row.generation_uid for row in views}) == banks.donor_initial
    assert {row.bank_role for row in views} == {BK.BankRole.DONOR.value}
    assert not item_file.has_private_rows
    assert item_file.status is tables.ItemStatus.COMPLETE
    # every draw asked for a condition-free seed and the same prompt bytes
    assert len({request.seed_key.actor_slot for request in runner.requests}) == banks.donor_initial
    assert len({tuple(m["content"] for m in request.messages) for request in runner.requests}) == 1


def test_the_extension_stage_of_the_donor_bank_draws_the_upper_range(study_cfg):
    runner = StubRunner(study_cfg, stub_item(), CellKind.DONOR_BANK, extension_stage=1)
    views = BK.bank_worker(runner, ITEM).rows["banks"]
    banks = study_cfg.contract.banks
    assert [row.draw_index for row in views] == list(range(banks.donor_initial, banks.donor_max))
    assert {row.extension_stage for row in views} == {1}


def test_the_bank_view_row_names_the_draw_and_its_decision(study_cfg):
    runner = StubRunner(study_cfg, stub_item(), CellKind.NATURAL_ROOT)
    item_file = BK.bank_worker(runner, ITEM)
    decisions = {row.decision_id: row for row in item_file.rows["decisions"]}
    for view in item_file.rows["banks"]:
        decision = decisions[view.decision_id]
        assert view.generation_uid == decision.generation_uid
        assert view.cell_kind == CellKind.NATURAL_ROOT.value == decision.cell_kind
        assert view.request_hash == json.loads(decision.request_hashes_json)[0]


def test_a_cells_draws_go_out_together_rather_than_one_at_a_time(study_cfg):
    """``max_inflight`` has to govern the thing it names (D-03).

    Measured live before this: ~30 s per draw with one request in flight and the replica
    otherwise idle, so a 16-draw item took eight minutes and a cell's wall clock was the
    **sum** of its draws rather than the max over them.  Banks are the largest block in the
    campaign, so this is where a serial loop costs most.  The draws are exchangeable by
    construction -- independent samples, distinct seed keys, none observing another -- which
    is what makes dispatching them together result-identical and not merely faster.
    """
    import threading

    runner = StubRunner(study_cfg, stub_item(), CellKind.DONOR_BANK)
    runner.concurrency = type("C", (), {"max_inflight": 16})()
    peak, live, lock = 0, 0, threading.Lock()
    barrier = threading.Barrier(16, timeout=10)
    inner = runner.generate

    def counting(request):
        nonlocal peak, live
        with lock:
            live += 1
            peak = max(peak, live)
        barrier.wait()                      # every draw must be in flight at once, or time out
        with lock:
            live -= 1
        return inner(request)

    runner.generate = counting
    item_file = BK.bank_worker(runner, ITEM)
    assert peak == 16, f"only {peak} draws were ever in flight at once"
    # and batching changed nothing about the result: order, count and identities all hold
    views = item_file.rows["banks"]
    assert [row.draw_index for row in views] == list(range(study_cfg.contract.banks.donor_initial))
    assert len({row.generation_uid for row in views}) == len(views)


def test_a_runner_that_declares_no_width_keeps_the_old_serial_behaviour(study_cfg):
    """The fallback is 1, not a guessed width: a number nobody chose is how D-03 starts."""
    from agents_scaling.nextdyn.kinds import batch_width

    assert batch_width(object(), 16) == 1
    assert batch_width(type("R", (), {"concurrency": None})(), 16) == 1
    # and it never exceeds the work there is to do
    runner = type("R", (), {"concurrency": type("C", (), {"max_inflight": 16})()})()
    assert batch_width(runner, 3) == 3
