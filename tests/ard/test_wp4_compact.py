"""WP4 compaction: schemas for every spec and descriptive table, keys, and the identities.

Spec §10 "Required round outputs", §9 (bank and selection panel), §18 (the seven tables).
Plan §5.8: ``round_summary`` keyed by **episode_id × round** (an item × protocol × checkpoint
key would collide for the two challenge teams and the two budget arms), selected correctness
long form over selector × pool, ``transitions`` at agent-round grain with CC/CW/WC/WW plus
the invalidity classes and **both** answer-change definitions, and the
``C_{t+1} - C_t`` accounting identity asserted at compaction.
"""

from __future__ import annotations

import json

import pytest

from agents_scaling.ard import compact as C
from agents_scaling.ard import contract as K
from agents_scaling.ard.types import (
    ArdCellKind,
    ArdCellSpec,
    DecisionRef,
    EpisodeRecord,
    RoundState,
    decision_id_for,
)
from tests.ard.wp4_support import make_run_root, synthetic_tasks, write_items

ITEM = "math500-0000"


def _cell(**overrides) -> ArdCellSpec:
    base = dict(
        cell_id="EPISODE.natural.PEER.8B.T1024.s000", kind=ArdCellKind.EPISODE,
        protocol=K.Protocol.PEER, round=None, cohort=K.Cohort.NATURAL,
        checkpoint=K.CHECKPOINT_SMALL, reasoning_allowance=K.THINKING_TOKENS_LOW,
        final_allowance=K.FINAL_TOKENS, items=(ITEM,), team_id=None, parent_ref=None,
        leader_seed=None, depends_on=(), max_inflight=5, parallel_items=1, lane="8B",
    )
    base.update(overrides)
    return ArdCellSpec(**base)


def _decision(episode_id, *, round_, agent, cls, valid=True, confidence=0.7, content=None):
    return DecisionRef(
        decision_id=decision_id_for(episode_id, round_=round_, agent_id=agent,
                                    phase="root" if round_ == 0 else "revise"),
        episode_id=episode_id, request_id=f"req-{episode_id}-{round_}-{agent}", agent_id=agent,
        round=round_, phase="root" if round_ == 0 else "revise", protocol="PEER",
        checkpoint=K.CHECKPOINT_SMALL, answer_raw=cls, answer_class=cls if valid else None,
        content_sha256=content or f"{cls}-{round_}-{agent}",
        parse_status=K.ParseStatus.OK.value if valid else K.ParseStatus.MISSING_FINAL.value,
        confidence=confidence, thinking_tokens=100, final_tokens=20, prompt_tokens=300,
        cached_prefill_tokens=100, latency_s=1.0, arrival_ts=float(round_), endpoint_id="e:8000",
        flops={"prefill": 10, "decode": 20, "total": 30, "oracle": "abc"},
        n_packets=0 if round_ == 0 else 4,
    )


def _episode(cell, *, classes_by_round, episode_id=None, team_id=None, allowance=None, cohort=None):
    episode_id = episode_id or f"{cell.cell_id}|{ITEM}"
    decisions, states = [], []
    for round_, classes in sorted(classes_by_round.items()):
        refs = [_decision(episode_id, round_=round_, agent=agent, cls=cls, valid=cls is not None)
                for agent, cls in enumerate(classes)]
        decisions += refs
        states.append(RoundState(
            state_id=f"state-{round_}", episode_id=episode_id, round=round_,
            decision_ids=tuple(ref.decision_id for ref in refs),
            answer_classes=tuple(ref.answer_class for ref in refs),
            valid=tuple(ref.valid for ref in refs)))
    return EpisodeRecord(
        episode_id=episode_id, item_id=ITEM, cohort=(cohort or cell.cohort).value
        if hasattr(cohort or cell.cohort, "value") else str(cohort or cell.cohort.value),
        protocol=cell.protocol.value, checkpoint=cell.checkpoint,
        checkpoint_revision=K.REVISIONS[cell.checkpoint],
        reasoning_allowance=allowance or cell.reasoning_allowance,
        packet_policy="peer", team_id=team_id, decisions=tuple(decisions), states=tuple(states))


def _labels(triples, correct_classes=("A",)):
    labels = {}
    for _cell, item_id, episode in triples:
        for decision in episode.decisions:
            labels[decision.decision_id] = {
                "decision_id": decision.decision_id, "item_id": item_id,
                "is_correct": decision.answer_class in correct_classes,
                "evaluator_status": "correct" if decision.answer_class in correct_classes else "incorrect",
                "grader_version": "g1", "grading_duration": 0.01,
                "answer_class": decision.answer_class,
            }
    return labels


# --------------------------------------------------------------------------- schemas


def test_every_contract_table_has_a_declared_schema():
    assert set(K.ALL_TABLES) <= set(C.SCHEMAS)
    assert C.SELECTION_TABLE in C.SCHEMAS
    for name, columns in C.SCHEMAS.items():
        assert columns and len(set(columns)) == len(columns), name


def test_the_seven_spec_tables_carry_the_spec_field_lists():
    assert set(K.SPEC_TABLES) <= set(C.SCHEMAS)
    for column in ("item_id", "family", "group_id", "gold_location", "grader_hash"):
        assert column in C.SCHEMAS["items"]
    for column in ("episode_id", "root_candidate_ids", "reasoning_allowance", "packet_policy",
                   "leader_id", "condition_manifest_hash"):
        assert column in C.SCHEMAS["episodes"]
    for column in ("decision_id", "parent_state_id", "seen_message_ids", "boundary_injected",
                   "logical_decision_id", "physical_request_ids", "endpoint_id", "cap_hit"):
        assert column in C.SCHEMAS["decisions"]
    for column in ("decision_id", "is_correct", "evaluator_status", "grader_version",
                   "grading_duration"):
        assert column in C.SCHEMAS["labels"]
    for column in ("fork_id", "shared_prefix_hash", "focal_agent", "branch_seed",
                   "common_suffix_policy", "suffix_seed"):
        assert column in C.SCHEMAS["forks"]
    for column in ("server_batch_id", "physical_request_ids", "prefill_tokens", "decode_tokens",
                   "allocated_gpu_seconds", "gpu_uuid", "power_if_available"):
        assert column in C.SCHEMAS["resources"]


def test_round_summary_carries_the_plan_panel_including_the_mirrored_wrong_minority():
    columns = set(C.SCHEMAS["round_summary"])
    for column in ("episode_id", "round", "cohort", "constructed_C0", "team_id",
                   "reasoning_allowance", "parent_ref", "C_t", "m_t", "delta_m",
                   "n_wrong_to_correct", "n_correct_to_wrong", "accounting_residual",
                   "a_correction", "d_destruction", "coverage_current", "coverage_archive",
                   "class_entropy", "correct_plurality_status", "entered_unanimity",
                   "left_unanimity", "unanimous_wrong_escape", "first_correct_loss_round",
                   "recovery_round", "right_censored", "persistence_same_agent",
                   "dissemination_new_agents", "persistence_without_original_holder",
                   "wrong_minority_persists", "wrong_class_disseminated",
                   "wrong_unanimity_reached", "novel_correct_on_c0_zero", "selection_failed",
                   "parse_status_histogram", "cap_hit_rate", "boundary_injected_rate",
                   "final_overrun_rate", "conf_missing_rate"):
        assert column in columns, column


def test_the_selector_pool_long_form_covers_every_registered_selector_and_pool():
    assert {"selector", "pool", "expected_correct_under_ties", "repeated_descendants"} <= \
        set(C.SCHEMAS[C.SELECTION_TABLE])


# --------------------------------------------------------------------------- keys


def test_round_summary_is_keyed_by_episode_and_round_so_challenge_teams_never_collide():
    cell_a = _cell(cell_id="CHALLENGE.challenge.PRIVATE.8B.T1024.gteamA.s000",
                   kind=ArdCellKind.CHALLENGE, protocol=K.Protocol.PRIVATE,
                   cohort=K.Cohort.CHALLENGE, team_id="teamA")
    cell_b = _cell(cell_id="CHALLENGE.challenge.PRIVATE.8B.T1024.gteamB.s000",
                   kind=ArdCellKind.CHALLENGE, protocol=K.Protocol.PRIVATE,
                   cohort=K.Cohort.CHALLENGE, team_id="teamB")
    triples = [
        (cell_a, ITEM, _episode(cell_a, classes_by_round={0: ["A"] * 5, 1: ["A"] * 5},
                                episode_id="ep-teamA", team_id="teamA")),
        (cell_b, ITEM, _episode(cell_b, classes_by_round={0: ["B"] * 5, 1: ["B"] * 5},
                                episode_id="ep-teamB", team_id="teamB")),
    ]
    rows = C.round_summary_rows(triples, _labels(triples), families={ITEM: "math500"},
                                challenge_c0={"teamA": 4, "teamB": 1})
    keys = {(row["episode_id"], row["round"]) for row in rows}
    assert len(keys) == len(rows) == 4
    assert {row["constructed_C0"] for row in rows} == {4, 1}
    assert {row["team_id"] for row in rows} == {"teamA", "teamB"}


def test_the_two_budget_arms_of_one_item_are_separate_rows():
    low = _cell(cell_id="EPISODE.natural.PEER.8B.T1024.s000")
    high = _cell(cell_id="EPISODE.high_budget.PEER.8B.T4096.s000",
                 cohort=K.Cohort.HIGH_BUDGET, reasoning_allowance=K.THINKING_TOKENS_HIGH)
    triples = [
        (low, ITEM, _episode(low, classes_by_round={0: ["A"] * 5}, episode_id="ep-low")),
        (high, ITEM, _episode(high, classes_by_round={0: ["A"] * 5}, episode_id="ep-high",
                              allowance=K.THINKING_TOKENS_HIGH)),
    ]
    rows = C.round_summary_rows(triples, _labels(triples), families={ITEM: "math500"})
    assert {row["reasoning_allowance"] for row in rows} == {K.THINKING_TOKENS_LOW,
                                                            K.THINKING_TOKENS_HIGH}
    assert len({row["episode_id"] for row in rows}) == 2


# --------------------------------------------------------------------------- identities


def test_the_correct_count_accounting_identity_holds_and_a_breach_raises():
    cell = _cell()
    triples = [(cell, ITEM, _episode(cell, classes_by_round={
        0: ["A", "A", "B", "B", "B"], 1: ["A", "B", "A", "B", "B"]}))]
    rows = C.round_summary_rows(triples, _labels(triples), families={ITEM: "math500"})
    round1 = [row for row in rows if row["round"] == 1][0]
    assert round1["C_t"] == 2 and round1["delta_C"] == 0
    assert round1["n_wrong_to_correct"] == 1 and round1["n_correct_to_wrong"] == 1
    assert round1["accounting_residual"] == 0
    assert round1["a_correction"] == pytest.approx(1 / 3)
    assert round1["d_destruction"] == pytest.approx(1 / 2)

    # A lost decision (four agents recorded in round 1 instead of five) breaks the identity,
    # and compaction must refuse rather than publish a panel that does not add up.
    episode = triples[0][2]
    lost = EpisodeRecord.from_dict({
        **episode.to_dict(),
        "decisions": [d for d in episode.to_dict()["decisions"]
                      if not (d["round"] == 1 and d["agent_id"] == 0)],
    })
    with pytest.raises(C.CompactionError, match="residual"):
        C.round_summary_rows([(cell, ITEM, lost)], _labels(triples),
                             families={ITEM: "math500"})


def test_a_partially_graded_run_leaves_the_residual_null_instead_of_crashing():
    cell = _cell()
    triples = [(cell, ITEM, _episode(cell, classes_by_round={
        0: ["A", "A", "B", "B", "B"], 1: ["A", "B", "A", "B", "B"]}))]
    labels = _labels(triples)
    labels.pop(next(iter(labels)))
    rows = C.round_summary_rows(triples, labels, families={ITEM: "math500"})
    assert [row for row in rows if row["round"] == 1][0]["accounting_residual"] is None


def test_transitions_carry_the_four_classes_the_invalidity_classes_and_both_change_rules():
    cell = _cell()
    triples = [(cell, ITEM, _episode(cell, classes_by_round={
        0: ["A", "A", "B", "B", None], 1: ["A", "B", "A", "B", "A"]}))]
    rows = C.transition_rows(triples, _labels(triples))
    assert len(rows) == K.N_AGENTS
    by_agent = {row["agent_id"]: row for row in rows}
    assert by_agent[0]["transition"] == "CC" and by_agent[1]["transition"] == "CW"
    assert by_agent[2]["transition"] == "WC" and by_agent[3]["transition"] == "WW"
    assert by_agent[4]["invalidity_class"] == "invalid_to_valid"
    assert by_agent[0]["changed_class"] is False and by_agent[0]["changed_content"] is True
    assert by_agent[1]["changed_class"] is True


def test_coverage_series_are_separate_and_the_archive_never_falls():
    cell = _cell()
    triples = [(cell, ITEM, _episode(cell, classes_by_round={
        0: ["A", "B", "B", "B", "B"], 1: ["B"] * 5, 2: ["B"] * 5, 3: ["B"] * 5}))]
    rows = sorted(C.round_summary_rows(triples, _labels(triples), families={ITEM: "math500"}),
                  key=lambda row: row["round"])
    assert [row["coverage_current"] for row in rows] == [1.0, 0.0, 0.0, 0.0]
    assert [row["coverage_archive"] for row in rows] == [1.0, 1.0, 1.0, 1.0]
    assert rows[1]["first_correct_loss_round"] == 1
    assert rows[3]["right_censored"] is True
    assert rows[1]["wrong_unanimity_reached"] is True and rows[1]["unanimity_wrong"] is True
    assert rows[1]["entered_unanimity"] is True


def test_selection_is_long_form_over_selector_and_pool_with_the_analytic_tie_expectation():
    cell = _cell()
    triples = [(cell, ITEM, _episode(cell, classes_by_round={0: ["A", "A", "B", "B", "C"]}))]
    rows = C.selection_rows(triples, _labels(triples))
    assert {row["selector"] for row in rows} == set(K.SELECTORS)
    assert {row["pool"] for row in rows} == set(K.POOL_KINDS)
    plurality = [row for row in rows if row["selector"] == "plurality" and row["pool"] == "current"][0]
    assert plurality["tied_winning_classes"] == 2
    assert plurality["expected_correct_under_ties"] == pytest.approx(0.5)


def test_c_dist_and_drift_are_descriptive_summaries_over_their_declared_keys():
    cell = _cell()
    triples = [(cell, ITEM, _episode(cell, classes_by_round={
        0: ["A", "A", "B", "B", "B"], 1: ["A", "A", "A", "B", "B"]}))]
    summary = C.round_summary_rows(triples, _labels(triples), families={ITEM: "math500"})
    c_dist = C.c_dist_rows(summary)
    assert {row["c"] for row in c_dist} == set(range(K.N_AGENTS + 1))
    assert all(set(row) == set(C.SCHEMAS["c_dist"]) for row in c_dist)
    assert sum(row["p"] for row in c_dist if row["round"] == 1) == pytest.approx(1.0)
    drift = C.drift_rows(summary)
    assert drift and set(drift[0]) == set(C.SCHEMAS["drift"])
    assert drift[0]["a_of_m"] == pytest.approx(1 / 3) and drift[0]["d_of_m"] == pytest.approx(0.0)
    assert drift[0]["realized_delta_m"] == pytest.approx(0.2)


def test_bank_summary_uses_the_finite_bank_estimator_over_the_contract_pass_ks():
    cell = _cell(cell_id="ROOTS.natural.none.8B.T1024.s000", kind=ArdCellKind.ROOTS,
                 protocol=None)
    episode = _episode(_cell(), classes_by_round={0: ["A", "A", "B", "B", "C"]},
                       episode_id="ep-bank")
    rows = C.bank_summary_rows([(cell, ITEM, episode)], _labels([(cell, ITEM, episode)]),
                               families={ITEM: "math500"})
    assert len(rows) == 1
    row = rows[0]
    assert row["n_draws"] == 5 and row["c_correct"] == 2
    assert row["pass_at_1"] == pytest.approx(0.4)
    assert all(f"pass_at_{k}" in row for k in K.PASS_KS)
    assert row["selection_gap"] == pytest.approx(row["oracle_coverage"] - row["plurality_correct"])
    assert json.loads(row["class_histogram"]) == {"A": 2, "B": 2, "C": 1}


def test_cost_tables_report_decision_matched_not_flop_matched():
    cell = _cell()
    triples = [(cell, ITEM, _episode(cell, classes_by_round={0: ["A"] * 5, 1: ["A"] * 5,
                                                             2: ["A"] * 5, 3: ["A"] * 5}))]
    costs = C.cost_summary_rows(triples)
    assert set(costs[0]) == set(C.SCHEMAS["cost_summary"])
    assert costs[0]["logical_decisions"] == K.N_AGENTS
    assert costs[-1]["cumulative_logical_flops"] == 4 * K.N_AGENTS * 30
    assert costs[0]["prefill_tokens"] == K.N_AGENTS * 200
    summary = C.round_summary_rows(triples, _labels(triples), families={ITEM: "math500"})
    accuracy = C.cost_accuracy_rows(costs, summary)
    assert set(accuracy[0]) == set(C.SCHEMAS["cost_accuracy"])
    assert all(row["decision_matched"] and not row["flop_matched"] for row in accuracy)
    final = [row for row in accuracy if row["round"] == K.UPDATE_ROUNDS][0]
    assert final["round3_selected_accuracy"] == pytest.approx(1.0)


def test_an_undeclared_column_is_refused_at_projection_time():
    with pytest.raises(C.CompactionError, match="undeclared"):
        C._project("c_dist", {"protocol": "PEER", "gold": "42"})


# --------------------------------------------------------------------------- writing


def test_compaction_writes_every_table_with_empty_sentinels_for_the_deferred_ones(tmp_path):
    run_root = make_run_root(tmp_path)
    tasks = synthetic_tasks(1, splits=("test",))
    write_items(run_root, tasks)
    counts = C.compact(run_root, cell_ids=[])
    tables = run_root / C.TABLES_DIR
    for name in list(K.ALL_TABLES) + [C.SELECTION_TABLE]:
        assert name in counts
        assert (tables / f"{name}.parquet").exists() or (tables / f"{name}.empty").exists()
    for name in C.DEFERRED_TABLES:
        assert counts[name] == 0
    assert counts["items"] == len(tasks)
    assert json.loads((tables / "compaction.json").read_text())["study_id"] == K.STUDY_ID


def test_written_parquet_columns_are_exactly_the_declared_schema(tmp_path):
    import pyarrow.parquet as pq

    run_root = make_run_root(tmp_path)
    write_items(run_root, synthetic_tasks(1, splits=("test",)))
    C.compact(run_root, cell_ids=[])
    table = pq.read_table(run_root / C.TABLES_DIR / "items.parquet")
    assert set(table.column_names) == set(C.SCHEMAS["items"])
