"""WP5 selectors: one sealed pool, five comparable rows, and an id-only LLM audit.

Spec §12.4 ("evaluate root/current/final candidate selection via plurality,
confidence-weighted plurality, CURRENT, and a dynamics-only static selector"; "Return a
candidate ID only; do not allow a newly synthesized answer"; "Candidate order is
randomized"), §7 (invalid answers cannot form a winning content class); plan §5.8 (selected
correctness is long form over selector x pool).
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import controller as CT
from agents_scaling.ard import runner as ard_runner
from agents_scaling.ard import selectors as SEL
from agents_scaling.ard import vote as VOTE
from agents_scaling.ard.types import EXIT_DONE
from tests.ard import wp5_support as S

SEED = bytes.fromhex("7d" * 32)


# --------------------------------------------------------------------------- the pool


def test_the_registered_selector_set_is_the_frozen_one():
    assert SEL.SELECTOR_ORDER == K.SELECTORS
    assert SEL.AUDIT_POOLS == K.SELECTOR_AUDIT_POOLS


def test_every_selector_row_of_one_pool_shares_one_pool_id_and_one_validity_convention():
    """Spec §12.4 only compares selectors that ranked the same candidates under one rule."""
    pool = S.pool_of()
    rows = SEL.score_all(pool, study_seed=SEED, audit=None)
    assert [row.selector for row in rows] == list(K.SELECTORS)
    assert {row.pool_id for row in rows} == {pool.pool_id}
    assert {row.validity_convention for row in rows} == {SEL.VALIDITY_CONVENTION}
    assert {row.pool_candidate_ids for row in rows} == {tuple(c.candidate_id for c in pool.candidates)}
    assert {row.planned_count for row in rows} == {len(pool.candidates)}
    assert {row.valid_count for row in rows} == {len(pool.eligible)}


def test_the_pool_id_is_stable_content_addressed_and_carries_the_convention():
    pool = S.pool_of()
    assert pool.pool_id.startswith(SEL.POOL_ID_PREFIX)
    assert pool.pool_id == S.pool_of().pool_id
    assert pool.pool_id != S.pool_of(("7", "7", "9", None, "13")).pool_id
    assert pool.pool_id != S.pool_of(pool_kind="current").pool_id
    assert pool.identity_object()["validity_convention"] == SEL.VALIDITY_CONVENTION


def test_the_pool_id_ignores_confidence_so_the_same_pool_is_one_pool():
    """Confidence is a *score*, not pool content: it must not fork the join key."""
    a = S.pool_of(confidences=[0.1, 0.2, 0.3, None, 0.4])
    b = S.pool_of(confidences=[0.9, 0.8, 0.7, None, 0.6])
    assert a.pool_id == b.pool_id


def test_a_pool_refuses_a_duplicated_candidate():
    with pytest.raises(SEL.SelectorError, match="twice"):
        SEL.sealed_pool("i", "root", [S.candidate("d0", "7"), S.candidate("d0", "9")])


def test_an_unknown_pool_kind_is_refused():
    with pytest.raises(SEL.SelectorError, match="unknown pool kind"):
        SEL.sealed_pool("i", "elsewhere", [S.candidate("d0", "7")])


def test_the_validity_convention_is_exactly_the_vote_keyed_rule():
    pool = S.pool_of(("7", None, "9", "9", None))
    assert [c.keyed for c in pool.candidates] == [c.keyed for c in pool.vote_pool()]
    assert len(pool.eligible) == 3


def test_an_all_invalid_pool_fails_selection_for_every_selector():
    """Spec §7: "if all outputs are invalid, selection fails"."""
    pool = S.pool_of((None, None, None, None, None))
    rows = SEL.score_all(pool, study_seed=SEED, audit=None)
    assert all(row.selection_failed for row in rows)
    assert all(row.selected_candidate_id is None for row in rows)
    assert pool.selection_failed


# --------------------------------------------------------------------------- the two votes


def test_plurality_and_confidence_weighted_delegate_to_the_frozen_vote_module():
    pool = S.pool_of(("7", "7", "9", None, "9"), confidences=[0.2, 0.2, 0.9, None, 0.9])
    plurality = SEL.score_selector(SEL.SELECTOR_PLURALITY, pool, study_seed=SEED)
    weighted = SEL.score_selector(SEL.SELECTOR_CONFIDENCE, pool, study_seed=SEED)
    reference = VOTE.plurality(pool.vote_pool(), source_id=pool.source_id, study_seed=SEED,
                               pool_kind=pool.pool_kind)
    weighted_reference = VOTE.confidence_weighted_plurality(
        pool.vote_pool(), source_id=pool.source_id, study_seed=SEED, pool_kind=pool.pool_kind)
    assert plurality.selected_candidate_id == reference.selected_candidate_id
    assert weighted.selected_candidate_id == weighted_reference.selected_candidate_id
    assert weighted.selected_class == "9"          # the confident minority wins on weight
    assert plurality.selected_class in {"7", "9"}


def test_an_invalid_candidate_is_never_selected_by_any_selector():
    features = [{**S.feature_row(), "own_confidence": v} for v in (0.1, 0.2, 0.99, 0.3)]
    pool = SEL.sealed_pool("i", "root", [
        S.candidate("d0", "7", features=features[0]),
        S.candidate("d1", "7", features=features[1]),
        S.candidate("d2", None, valid=False, features=features[2]),   # the highest score
        S.candidate("d3", "9", features=features[3]),
    ])
    rows = SEL.score_all(pool, study_seed=SEED, controller=S.stub_controller(),
                         audit=SEL.AuditOutcome(SEL.AUDIT_OK, "CANDIDATE-9", "d2", "CANDIDATE-9",
                                                pool.pool_id))
    assert all(row.selected_candidate_id != "d2" for row in rows)


# --------------------------------------------------------------------------- model heads


def test_the_model_selectors_pick_the_argmax_of_their_frozen_head():
    scores = (0.1, 0.9, 0.4)
    features = [{**S.feature_row(), "own_confidence": v} for v in scores]
    pool = SEL.sealed_pool("i", "current", [
        S.candidate(f"d{index}", answer, features=feature)
        for index, (answer, feature) in enumerate(zip(("7", "9", "11"), features))
    ], round_=1)
    for selector in (SEL.SELECTOR_CURRENT, SEL.SELECTOR_DYNAMICS):
        row = SEL.score_selector(selector, pool, study_seed=SEED, controller=S.stub_controller())
        assert row.status == SEL.STATUS_OK
        assert row.selected_candidate_id == "d1"
        assert row.scores["d1"] == pytest.approx(0.9)


def test_a_model_selector_without_features_reports_the_gap_instead_of_guessing():
    pool = S.pool_of()
    row = SEL.score_selector(SEL.SELECTOR_CURRENT, pool, study_seed=SEED,
                             controller=S.stub_controller())
    assert row.status == SEL.STATUS_FEATURES_UNAVAILABLE
    assert row.selected_candidate_id is None and row.selection_failed


def test_a_model_selector_without_a_controller_reports_that():
    pool = S.pool_of(features=[S.feature_row()] * 5)
    row = SEL.score_selector(SEL.SELECTOR_DYNAMICS, pool, study_seed=SEED, controller=None)
    assert row.status == SEL.STATUS_NO_CONTROLLER


def test_a_controller_without_the_head_is_reported_not_swallowed():
    pool = S.pool_of(features=[S.feature_row()] * 5)
    row = SEL.score_selector(SEL.SELECTOR_CURRENT, pool, study_seed=SEED,
                             controller=S.stub_controller(heads=(CT.HEAD_DYNAMICS,)))
    assert row.status == SEL.STATUS_NO_CONTROLLER
    assert CT.HEAD_CURRENT in row.detail


# --------------------------------------------------------------------------- the audit


def test_the_presented_order_is_a_deterministic_blind_permutation_of_the_eligible_set():
    pool = S.pool_of(("7", "9", "11", None, "13"))
    presentation = SEL.audit_presentation(pool, study_seed=SEED)
    assert sorted(presentation.order) == list(range(len(pool.eligible)))
    assert presentation.labels == tuple(f"{SEL.AUDIT_LABEL_PREFIX}{i + 1}"
                                        for i in range(len(pool.eligible)))
    assert SEL.audit_presentation(pool, study_seed=SEED).order == presentation.order
    assert set(presentation.candidate_ids) == {c.candidate_id for c in pool.eligible}


def test_the_order_does_not_depend_on_confidence_or_on_a_different_seed_agreeing():
    """Correctness-blind: the HMAC reads the pool id and the position, nothing else."""
    quiet = S.pool_of(confidences=[0.1, 0.1, 0.1, None, 0.1])
    loud = S.pool_of(confidences=[0.99, 0.02, 0.5, None, 0.7])
    assert SEL.audit_order(quiet, study_seed=SEED) == SEL.audit_order(loud, study_seed=SEED)
    other = SEL.audit_order(quiet, study_seed=bytes.fromhex("11" * 32))
    assert sorted(other) == sorted(SEL.audit_order(quiet, study_seed=SEED))


def test_the_prompt_shows_the_opaque_labels_and_never_the_true_candidate_ids():
    pool = S.pool_of()
    presentation = SEL.audit_presentation(pool, study_seed=SEED)
    text = SEL.render_audit_prompt("What is 6 times 7?", presentation)[0]["content"]
    for label in presentation.labels:
        assert label in text
    for candidate in pool.candidates:
        assert candidate.candidate_id not in text
    assert K.FINAL_OPEN in text and "Do not write a new answer" in text


@pytest.mark.parametrize("raw,status", [
    ("CANDIDATE-2", SEL.AUDIT_OK),
    ("candidate-2", SEL.AUDIT_OK),
    ("  CANDIDATE-2  ", SEL.AUDIT_OK),
    ("CANDIDATE-9", SEL.AUDIT_UNKNOWN_LABEL),
    ("I choose CANDIDATE-2 because it is right", SEL.AUDIT_SYNTHESIZED),
    ("42", SEL.AUDIT_SYNTHESIZED),
    ("the answer is 7", SEL.AUDIT_SYNTHESIZED),
])
def test_the_audit_accepts_only_a_presented_id_and_never_a_synthesized_answer(raw, status):
    pool = S.pool_of()
    presentation = SEL.audit_presentation(pool, study_seed=SEED)
    outcome = SEL.parse_audit_choice(S.report(raw), presentation)
    assert outcome.status == status
    if status == SEL.AUDIT_OK:
        assert outcome.candidate_id == presentation.candidate_for("CANDIDATE-2")
    else:
        assert outcome.candidate_id is None


def test_an_unparseable_audit_report_is_a_refusal_not_a_selection():
    pool = S.pool_of()
    presentation = SEL.audit_presentation(pool, study_seed=SEED)
    outcome = SEL.parse_audit_choice(S.report(None), presentation)
    assert outcome.status == SEL.AUDIT_REFUSED and outcome.candidate_id is None
    row = SEL.score_selector(SEL.SELECTOR_LLM_AUDIT, pool, study_seed=SEED, audit=outcome)
    assert row.selection_failed and row.status == SEL.AUDIT_REFUSED


def test_a_synthesized_answer_never_becomes_the_selected_answer():
    pool = S.pool_of()
    presentation = SEL.audit_presentation(pool, study_seed=SEED)
    outcome = SEL.parse_audit_choice(S.report("13"), presentation)
    row = SEL.score_selector(SEL.SELECTOR_LLM_AUDIT, pool, study_seed=SEED, audit=outcome)
    assert row.selected_answer is None and row.selected_class is None
    assert row.status == SEL.AUDIT_SYNTHESIZED


def test_an_audit_outcome_from_another_pool_is_refused():
    pool = S.pool_of()
    stranger = SEL.AuditOutcome(SEL.AUDIT_OK, "CANDIDATE-1", "d0", "CANDIDATE-1", "pool:ffff")
    with pytest.raises(SEL.SelectorError, match="share one pool id"):
        SEL.score_selector(SEL.SELECTOR_LLM_AUDIT, pool, study_seed=SEED, audit=stranger)


def test_an_unrun_audit_is_a_status_not_a_silent_null():
    pool = S.pool_of()
    row = SEL.score_selector(SEL.SELECTOR_LLM_AUDIT, pool, study_seed=SEED, audit=None)
    assert row.status == SEL.STATUS_AUDIT_NOT_RUN and row.selection_failed


# --------------------------------------------------------------------------- pool reader


@pytest.fixture
def populated_root(tmp_path):
    root = S.make_run_root(tmp_path)
    item = "math500-0000"
    S.write_episode_item(root, "cellROOT", item, protocol="PRIVATE", round_=0,
                         answers=("7", "7", "9", None, "11"))
    S.write_episode_item(root, "cellPEER", item, protocol=K.Protocol.PEER.value,
                         round_=K.UPDATE_ROUNDS, answers=("7", "7", "7", "9", "9"))
    S.write_episode_item(root, "cellGATE", item, protocol=K.Protocol.GATE.value,
                         round_=K.UPDATE_ROUNDS, answers=("11", "11", "7", "9", "9"))
    return root, item


@pytest.mark.parametrize("pool_kind,expected", [
    ("root", 0), ("peer_final", K.UPDATE_ROUNDS), ("gate_final", K.UPDATE_ROUNDS),
])
def test_each_audit_pool_reads_its_own_declared_slice(populated_root, pool_kind, expected):
    root, item = populated_root
    pool = SEL.collect_pool(root, item, pool_kind)
    assert len(pool) == K.N_AGENTS
    assert pool.round_ == expected
    assert pool.pool_kind == pool_kind
    assert all(c.round_ == expected for c in pool.candidates)


def test_the_pools_are_disjoint_and_ordered_deterministically(populated_root):
    root, item = populated_root
    peer = SEL.collect_pool(root, item, "peer_final")
    gate = SEL.collect_pool(root, item, "gate_final")
    assert not set(c.candidate_id for c in peer.candidates) & \
        set(c.candidate_id for c in gate.candidates)
    assert [c.candidate_id for c in peer.candidates] == sorted(c.candidate_id for c in peer.candidates)
    assert SEL.collect_pool(root, item, "peer_final").pool_id == peer.pool_id


def test_the_root_pool_deduplicates_a_shared_content_addressed_decision(populated_root):
    """One root decision is shared by every protocol that drew it (contract §3)."""
    root, item = populated_root
    S.write_episode_item(root, "cellPEERROOT", item, protocol=K.Protocol.PEER.value, round_=0,
                         answers=("7", "7", "9", None, "11"), episode_id="cellROOT|" + item)
    pool = SEL.collect_pool(root, item, "root")
    assert len(pool) == K.N_AGENTS


def test_an_item_with_no_written_cell_yields_an_empty_pool(tmp_path):
    pool = SEL.collect_pool(S.make_run_root(tmp_path), "nothing", "root")
    assert len(pool) == 0 and pool.selection_failed


def test_collect_pool_refuses_an_unknown_kind(populated_root):
    root, item = populated_root
    with pytest.raises(SEL.SelectorError, match="unknown audit pool"):
        SEL.collect_pool(root, item, "current")


# --------------------------------------------------------------------------- the cell


def test_the_audit_cell_runs_one_call_and_writes_five_rows_on_one_pool(tmp_path):
    run_root = S.make_run_root(tmp_path)
    tasks = S.synthetic_tasks(1, splits=("test",))
    S.write_items(run_root, tasks)
    item = tasks[0].item_id
    S.write_episode_item(run_root, "cellPEER", item, protocol=K.Protocol.PEER.value,
                         round_=K.UPDATE_ROUNDS, answers=("7", "7", "9", None, "11"))
    state_sha = ard_runner.freeze_register(run_root, "state",
                                           {"schema": "ard36-state-v1", "states": []})
    cell = S.audit_cell([tasks[0]], pool="peer_final", state_sha=state_sha)

    tokenizer = S.shared_tokenizer()
    with S.fake_server(run_root, tokenizer=tokenizer, answer_pool=("CANDIDATE-1",)) as server:
        code = S.cell_runner(cell, run_root, tokenizer=tokenizer)
        assert code == EXIT_DONE
        assert server.chat_count == 1          # one logical 8B call per (item, pool)

    payload = json.loads((run_root / "cells" / cell.cell_id / "items" / f"{item}.json").read_text())
    episode = payload["episodes"][0]
    assert len(episode["decisions"]) == 1
    assert episode["decisions"][0]["phase"] == K.PURPOSE_SELECTOR
    assert episode["decisions"][0]["answer_class"] is None      # an id is not a content class
    events = episode["events"]
    rows = [e for e in events if e["event"] == "selector_row"]
    choice = next(e for e in events if e["event"] == "selector_audit_choice")
    presentation = next(e for e in events if e["event"] == "selector_audit_presentation")
    assert [r["selector"] for r in rows] == list(K.SELECTORS)
    assert len({r["pool_id"] for r in rows}) == 1
    assert episode["condition_manifest_hash"] == rows[0]["pool_id"]
    assert choice["status"] == SEL.AUDIT_OK
    assert choice["candidate_id"] in presentation["candidate_ids"]
    audit_row = next(r for r in rows if r["selector"] == SEL.SELECTOR_LLM_AUDIT)
    assert audit_row["selected_decision_id"] == choice["candidate_id"]


def test_the_audit_cell_makes_no_call_on_an_empty_pool(tmp_path):
    run_root = S.make_run_root(tmp_path)
    tasks = S.synthetic_tasks(1, splits=("test",))
    S.write_items(run_root, tasks)
    state_sha = ard_runner.freeze_register(run_root, "state",
                                           {"schema": "ard36-state-v1", "states": []})
    cell = S.audit_cell([tasks[0]], pool="gate_final", state_sha=state_sha)
    with S.fake_server(run_root) as server:
        assert S.cell_runner(cell, run_root) == EXIT_DONE
        assert server.chat_count == 0
    payload = json.loads((run_root / "cells" / cell.cell_id / "items" /
                          f"{tasks[0].item_id}.json").read_text())
    rows = [e for e in payload["episodes"][0]["events"] if e["event"] == "selector_row"]
    assert all(row["selection_failed"] for row in rows)


def test_the_audit_cell_is_the_worker_the_kind_table_names():
    from agents_scaling.ard.kinds import KIND_TABLE
    from agents_scaling.ard.types import ArdCellKind

    spec = KIND_TABLE[ArdCellKind.SELECTOR_AUDIT]
    assert spec.worker == "agents_scaling.ard.selectors:run_selector_audit_cell"
    assert ard_runner.resolve_worker(ArdCellKind.SELECTOR_AUDIT) is SEL.run_selector_audit_cell


def test_the_module_level_import_closure_never_reaches_the_gold_reader():
    import ast
    from pathlib import Path

    forbidden = {"agents_scaling.ard.graders", "agents_scaling.ard.items_build", "datasets"}
    tree = ast.parse(Path(SEL.__file__).read_text())
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets = [node.module]
        assert not set(targets) & forbidden, targets
