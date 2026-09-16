"""WP7 selector audit: why three selectors are null, and what the offline repair may claim.

Two layers, the pattern of ``test_panels.py``.  The synthetic layer builds the smallest
tables that exhibit each cause — a run with no ``SELECTOR_AUDIT`` cell and no frozen
``CURRENT`` head, a one-agent probe whose features cannot be built, a pool with no keyed
candidate — and asserts the emitted ``unavailable_reason`` names that cause rather than
leaving a null beside a null.  The frozen layer runs the probe against the real ``ard36``
run root and pins the numbers the report quotes; it skips when that root is not mounted.

The load-bearing claim under test is the negative one: a structurally empty cell must never
be readable as a measured zero, so ``selected_decision_id is None`` and
``unavailable_reason == ""`` may not both hold on one row.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from agents_scaling.ard import contract as K
from agents_scaling.ard import features as FE
from agents_scaling.ard import selector_audit as SA
from agents_scaling.ard import selectors as SEL

RUN_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36")

FROZEN = pytest.mark.skipif(not (RUN_ROOT / "cells").is_dir(),
                            reason="the ard36 run root is not mounted")


# --------------------------------------------------------------------------- fixtures


def _decision(cell: str, episode: str, round_: int, agent: int, *, answer: str = "4",
              confidence: float | None = 0.6, parse_status: str = "ok") -> dict[str, Any]:
    decision_id = f"d:{cell}:{episode}:{round_}:{agent}"
    return {
        "decision_id": decision_id, "episode_id": episode, "cell_id": cell,
        "round": round_, "agent_id": agent, "answer_raw": answer,
        "answer_class": f"expr:{answer}" if answer else None,
        "evidence_text": "because", "confidence": confidence, "parse_status": parse_status,
        "content_sha256": hashlib.sha256(f"{decision_id}|{answer}".encode()).hexdigest(),
        "thinking_tokens": 32, "cap_hit": False, "answer_logprob": -0.5,
    }


def _episode(cell: str, episode: str, *, item: str = "i1", split: str = "train",
             family: str = "math500", cell_kind: str = "EPISODE") -> dict[str, Any]:
    return {
        "episode_id": episode, "cell_id": cell, "cell_kind": cell_kind, "item_id": item,
        "family": family, "split": split, "protocol": "PEER", "checkpoint": K.CHECKPOINT_SMALL,
        "cohort": K.Cohort.NATURAL.value,
    }


def _team(cell: str, episode: str, *, rounds: tuple[int, ...] = (0, 1),
          answers: tuple[str, ...] = ("4", "4", "4", "7", "9")) -> list[dict[str, Any]]:
    return [_decision(cell, episode, r, a, answer=answers[a])
            for r in rounds for a in range(K.N_AGENTS)]


def _tables(decisions, episodes, labels=()) -> dict[str, list[dict[str, Any]]]:
    return {"decisions": list(decisions), "episodes": list(episodes),
            "labels": list(labels), "items": []}


class _StubController:
    """A controller with both static heads: score = the candidate's own confidence."""

    def predict(self, head: str, rows):
        assert head in ("CURRENT", "dynamics_only")
        return [float(row["own_confidence"]) for row in rows]


# --------------------------------------------------------------------------- vocabulary


def test_the_reason_vocabulary_is_the_selectors_status_vocabulary():
    """One vocabulary in the study: a reason is a selector status, or the wiring gap."""
    assert SA.REASON_NONE == ""
    statuses = {SEL.STATUS_EMPTY_POOL, SEL.STATUS_NO_KEYED_CANDIDATE, SEL.STATUS_NO_CONTROLLER,
                SEL.STATUS_FEATURES_UNAVAILABLE, SEL.STATUS_AUDIT_NOT_RUN}
    assert statuses < set(SA.REASON_ORDER)
    assert SA.REASON_NOT_WIRED not in statuses
    assert SA.unavailable_reason(SEL.STATUS_OK) == SA.REASON_NONE
    assert SA.unavailable_reason(SEL.STATUS_NO_CONTROLLER) == SA.REASON_NO_CONTROLLER_HEAD


def test_static_selectors_are_the_two_that_need_no_model_call():
    assert SA.STATIC_SELECTORS == (SEL.SELECTOR_CURRENT, SEL.SELECTOR_DYNAMICS)
    assert SEL.SELECTOR_LLM_AUDIT not in SA.STATIC_SELECTORS


# --------------------------------------------------------------------------- features


def test_features_are_the_frozen_v1_builder_on_a_reconstructed_state(tmp_path):
    decisions = _team("EPISODE.c1", "e1")
    episodes = {("EPISODE.c1", "e1"): _episode("EPISODE.c1", "e1")}
    features, coverage = SA.features_by_decision(decisions, episodes)
    assert coverage["featurised_rounds"] == 2
    assert coverage["featurised_decisions"] == 2 * K.N_AGENTS
    row = features[("EPISODE.c1", "e1", "d:EPISODE.c1:e1:1:0")]
    assert set(row) == set(FE.FEATURES_V1)
    # Agent 0, 1 and 2 all answered "4": the own class count is three, and two peers compete.
    assert row["own_class_count"] == 3.0
    assert row["largest_competing_count"] == 1.0
    assert row["history_unavailable"] == 0.0     # round 1 has a round 0 behind it
    assert features[("EPISODE.c1", "e1", "d:EPISODE.c1:e1:0:0")]["history_unavailable"] == 1.0


def test_a_team_that_is_not_five_agents_gets_no_features_and_is_counted():
    """RECEIVER probes and ROOTS banks are not five-agent teams; a guessed row is worse."""
    decisions = [_decision("RECEIVER.c2", "e2", 0, 0)]
    features, coverage = SA.features_by_decision(
        decisions, {("RECEIVER.c2", "e2"): _episode("RECEIVER.c2", "e2", cell_kind="RECEIVER")})
    assert features == {}
    assert coverage["skipped_not_five_agent_team"] == 1


def test_one_decision_aliased_into_two_cells_gets_two_feature_rows_and_one_frame_row():
    """Contract §3: a root decision is shared, but it is one candidate with one grade."""
    shared = _decision("EPISODE.a", "e1", 0, 0)
    rows = []
    for cell in ("EPISODE.a", "EPISODE.b"):
        for agent in range(K.N_AGENTS):
            row = _decision(cell, "e1", 0, agent)
            row["decision_id"] = shared["decision_id"] if agent == 0 else row["decision_id"]
            rows.append(row)
    episodes = {("EPISODE.a", "e1"): _episode("EPISODE.a", "e1"),
                ("EPISODE.b", "e1"): _episode("EPISODE.b", "e1")}
    features, _ = SA.features_by_decision(rows, episodes)
    assert ("EPISODE.a", "e1", shared["decision_id"]) in features
    assert ("EPISODE.b", "e1", shared["decision_id"]) in features
    frame, counts = SA.static_current_rows(features, rows, episodes, {})
    assert counts["aliased_duplicates_dropped"] == 1
    assert len({r.row_id for r in frame}) == len(frame)


# --------------------------------------------------------------------------- reasons


def test_without_a_controller_every_static_row_says_no_controller(tmp_path):
    tables = _tables(_team("EPISODE.c1", "e1"), [_episode("EPISODE.c1", "e1")])
    repair = SA.build_repair(tables, run_root=tmp_path, refit=False)
    static = [r for r in repair.rows if r["selector"] in SA.STATIC_SELECTORS]
    assert static and all(r["selected_decision_id"] is None for r in static)
    assert {r["unavailable_reason"] for r in static} == {SA.REASON_NO_CONTROLLER_HEAD}


def test_the_audit_row_says_the_audit_never_ran_and_needs_generation(tmp_path):
    tables = _tables(_team("EPISODE.c1", "e1"), [_episode("EPISODE.c1", "e1")])
    repair = SA.build_repair(tables, run_root=tmp_path, refit=False)
    audit = [r for r in repair.rows if r["selector"] == SEL.SELECTOR_LLM_AUDIT]
    assert {r["unavailable_reason"] for r in audit} == {SA.REASON_AUDIT_NOT_RUN}
    row = next(a for a in repair.availability if a["selector"] == SEL.SELECTOR_LLM_AUDIT)
    assert row["needs_generation"] is True and row["repairable_offline"] is False
    for selector in SA.STATIC_SELECTORS:
        row = next(a for a in repair.availability if a["selector"] == selector)
        assert row["needs_generation"] is False and row["repairable_offline"] is True


def test_a_supplied_controller_populates_the_static_selectors(tmp_path):
    tables = _tables(_team("EPISODE.c1", "e1"), [_episode("EPISODE.c1", "e1")])
    repair = SA.build_repair(tables, run_root=tmp_path, refit=False,
                             controller=_StubController())
    static = [r for r in repair.rows if r["selector"] == SEL.SELECTOR_CURRENT]
    assert all(r["selected_decision_id"] is not None for r in static)
    assert {r["unavailable_reason"] for r in static} == {SA.REASON_NONE}
    assert {r["selector_source"] for r in static} == {SA.SOURCE_REFIT}


def test_a_pool_with_no_keyed_candidate_fails_for_every_selector(tmp_path):
    """Spec §7: if all outputs are invalid, selection fails — for the model selectors too."""
    decisions = [_decision("EPISODE.c1", "e1", 0, a, answer="", parse_status="empty_final")
                 for a in range(K.N_AGENTS)]
    tables = _tables(decisions, [_episode("EPISODE.c1", "e1")])
    repair = SA.build_repair(tables, run_root=tmp_path, refit=False,
                             controller=_StubController())
    assert {r["unavailable_reason"] for r in repair.rows} == {SA.REASON_NO_KEYED_CANDIDATE}


def test_a_featureless_round_is_features_unavailable_not_silently_null(tmp_path):
    decisions = [_decision("RECEIVER.c2", "e2", 0, 0)]
    tables = _tables(decisions, [_episode("RECEIVER.c2", "e2", cell_kind="RECEIVER")])
    repair = SA.build_repair(tables, run_root=tmp_path, refit=False,
                             controller=_StubController())
    static = [r for r in repair.rows if r["selector"] in SA.STATIC_SELECTORS]
    assert {r["unavailable_reason"] for r in static} == {SA.REASON_FEATURES_UNAVAILABLE}


def test_no_row_is_null_without_a_reason(tmp_path):
    """The whole point: a structurally empty cell can never read as a measured zero."""
    decisions = _team("EPISODE.c1", "e1") + [_decision("RECEIVER.c2", "e2", 0, 0)]
    episodes = [_episode("EPISODE.c1", "e1"),
                _episode("RECEIVER.c2", "e2", cell_kind="RECEIVER")]
    repair = SA.build_repair(_tables(decisions, episodes), run_root=tmp_path, refit=False,
                             controller=_StubController())
    for row in repair.rows:
        null = row["selected_decision_id"] is None
        assert null == bool(row["unavailable_reason"]), row
        assert row["unavailable_reason"] in SA.REASON_ORDER


def test_roots_cells_are_skipped_exactly_as_the_compaction_skips_them(tmp_path):
    decisions = _team("ROOTS.c3", "e3", rounds=(0,))
    tables = _tables(decisions, [_episode("ROOTS.c3", "e3", cell_kind="ROOTS")])
    repair = SA.build_repair(tables, run_root=tmp_path, refit=False)
    assert repair.rows == []
    assert repair.summary["by_selector_reason"]["roots_rounds_skipped"] == 1


# --------------------------------------------------------------------------- the firewall


def test_the_refit_refuses_a_frame_with_no_training_eligible_row(tmp_path):
    """Spec §12.4: trained only on training/validation items — a test-only frame is refused."""
    from agents_scaling.ard import controller as CT

    decisions = _team("EPISODE.c1", "e1")
    episodes = {("EPISODE.c1", "e1"): _episode("EPISODE.c1", "e1", split="test")}
    features, _ = SA.features_by_decision(decisions, episodes)
    labels = {d["decision_id"]: {"evaluator_status": "correct"} for d in decisions}
    rows, _ = SA.static_current_rows(features, decisions, episodes, labels)
    assert rows and all(r.split == "test" for r in rows)
    with pytest.raises(CT.ControllerError):
        SA.fit_static_heads(rows)


def test_an_ungraded_row_never_enters_the_frame(tmp_path):
    """``ambiguous``/``timeout`` is "the grader could not resolve", not "the agent was wrong"."""
    from agents_scaling.ard import controller as CT

    decisions = _team("EPISODE.c1", "e1")
    episodes = {("EPISODE.c1", "e1"): _episode("EPISODE.c1", "e1")}
    features, _ = SA.features_by_decision(decisions, episodes)
    labels = {d["decision_id"]: {"evaluator_status": "ambiguous"} for d in decisions}
    rows, _ = SA.static_current_rows(features, decisions, episodes, labels)
    assert all(r.y_correct is None for r in rows)
    with pytest.raises(CT.ControllerError):
        SA.fit_static_heads(rows)


# --------------------------------------------------------------------------- frozen run


@FROZEN
def test_the_frozen_run_has_no_audit_cell_and_no_static_head():
    """The measured diagnosis this module's report quotes."""
    probe = SA.probe_run(RUN_ROOT)
    assert probe["n_selector_audit_cells"] == 0
    assert probe["has_current_head"] is False
    assert probe["has_dynamics_head"] is False
    assert probe["controller_artifacts"], "the run froze a controller"
    assert all(sorted(heads) == ["p_peer", "p_private"]
               for heads in probe["controller_artifacts"].values())


@FROZEN
def test_the_compaction_still_does_not_join_wp5_selector_output():
    """If someone wires the join, this fails — and the module's docstring needs an edit."""
    assert SA.probe_run(RUN_ROOT)["compaction_joins_wp5_rows"] is False


@FROZEN
def test_the_written_repair_agrees_with_the_compactions_own_pool_counts():
    """The reconstructed pools are the compaction's pools, checked against plurality's rows."""
    pq = pytest.importorskip("pyarrow.parquet")
    tables = RUN_ROOT / "tables_v2"
    repair = tables / f"{SA.REPAIR_TABLE}.parquet"
    if not repair.exists():
        pytest.skip("the repair has not been written into tables_v2")
    key = ("episode_id", "round", "pool")
    static = {tuple(r[k] for k in key): r for r in pq.read_table(repair).to_pylist()
              if r["selector"] == SEL.SELECTOR_CURRENT}
    plurality = [r for r in pq.read_table(tables / "round_selection.parquet").to_pylist()
                 if r["selector"] == "plurality"]
    assert len(plurality) == len(static)
    mismatched = [r for r in plurality
                  if static[tuple(r[k] for k in key)]["pool_size"] != r["pool_size"]
                  or static[tuple(r[k] for k in key)]["valid_count"] != r["valid_count"]]
    assert mismatched == []
    # Every pool the compaction could not select over is a pool this module names a reason for.
    for row in plurality:
        if row["selected_decision_id"] is None:
            assert static[tuple(row[k] for k in key)]["unavailable_reason"] in (
                SA.REASON_EMPTY_POOL, SA.REASON_NO_KEYED_CANDIDATE)
