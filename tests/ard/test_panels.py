"""WP7 panels: the eight canonical row filters and the bank's dynamic range.

Two layers.  The synthetic layer builds the smallest tables that exhibit each defect the panels
exist to prevent -- a bank sharing an ``episode_id`` with a team, a round-0 decision aliased
into two arms, an ungradeable member censoring ``n_correct``, a protocol arm that silently went
missing, a receiver probe with no variance -- and asserts the panel notices.  The frozen layer
runs the real census against ``ard36/tables/`` and pins every number in the module docstring;
it skips when that run root is not mounted, so the file is still useful off the cluster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agents_scaling.ard import panels as PA
from agents_scaling.study.aggregate import _write_parquet

V1_TABLES = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/ard36/tables")

FROZEN = pytest.mark.skipif(not (V1_TABLES / "decisions.parquet").exists(),
                            reason="frozen ard36 v1 tables not mounted")


def _decision(episode: str, round_: int, agent: int, *, decision_id: str | None = None,
              protocol: str = "PEER", phase: str = "update", checkpoint: str = "8B",
              cohort: str = "natural", answer_class: str = "expr:1",
              budget_tokens: int = 1024, aliased: bool = False,
              parent_state_id: str | None = None) -> dict[str, Any]:
    return {
        "episode_id": episode, "round": round_, "agent_id": agent,
        "decision_id": decision_id or f"{episode}|r{round_}|a{agent}",
        "protocol": protocol, "checkpoint": checkpoint, "cohort": cohort, "phase": phase,
        "aliased": aliased, "budget_tokens": budget_tokens, "answer_class": answer_class,
        "parent_state_id": parent_state_id,
    }


def _team(episode: str, round_: int, **kw: Any) -> list[dict[str, Any]]:
    return [_decision(episode, round_, a, **kw) for a in range(PA.N_AGENTS)]


def _write(tmp_path: Path, *, decisions: list[dict[str, Any]],
           episodes: list[dict[str, Any]] | None = None,
           labels: list[dict[str, Any]] | None = None,
           forks: list[dict[str, Any]] | None = None,
           banks: list[dict[str, Any]] | None = None) -> Path:
    """A tables/ directory holding exactly the rows a test needs."""
    tables = tmp_path / "tables"
    tables.mkdir(exist_ok=True)
    if episodes is None:
        episodes = [{"episode_id": e, "item_id": e.split("|")[0], "split": "test",
                     "family": "math500"}
                    for e in sorted({d["episode_id"] for d in decisions})]
    if labels is None:
        labels = [{"decision_id": d["decision_id"], "evaluator_status": "correct",
                   "is_correct": True} for d in decisions]
    _write_parquet(tables / PA.DECISIONS_TABLE, decisions)
    _write_parquet(tables / PA.EPISODES_TABLE, episodes)
    _write_parquet(tables / PA.LABELS_TABLE, labels)
    _write_parquet(tables / PA.FORKS_TABLE, forks or [])
    _write_parquet(tables / PA.BANKS_TABLE, banks or [])
    return tables


# --------------------------------------------------------------------------- P0

def test_p0_keeps_a_team_whose_episode_id_is_shared_with_a_root_bank(tmp_path):
    """The defect P0 exists for: a size-5 filter applied before the lane split loses the team."""
    episode = "math500-aaaa|8B|natural|e0|PRIVATE"
    bank = [_decision(episode, 0, draw, decision_id=f"{episode}|bank{draw}",
                      protocol=PA.BANK_PROTOCOL, phase="root") for draw in range(16)]
    team = _team(episode, 0, protocol="PRIVATE", phase="root")
    tables = _write(tmp_path, decisions=bank + team)
    rows = PA.load_rows(tables)

    p0 = PA.panel_p0(rows)
    assert p0.size == PA.N_AGENTS
    assert p0.census["teams"] == 1
    # The raw group is 21 rows wide, so this is exactly the team minority.py drops.
    assert p0.census["teams_rescued_from_size_filter"] == 1
    assert p0.census["rows_excluded_bank"] == 16
    assert p0.coverage_rate == pytest.approx(5 / 21)


def test_p0_excludes_receiver_probes(tmp_path):
    probe = _decision("RECEIVER.s000|receiver:p00", 0, 0, protocol=PA.BANK_PROTOCOL,
                      phase=PA.RECEIVER_PHASE)
    tables = _write(tmp_path, decisions=_team("e|8B|natural|e0|PEER", 1) + [probe])
    p0 = PA.panel_p0(PA.load_rows(tables))
    assert p0.census["rows_excluded_receiver"] == 1
    assert p0.size == PA.N_AGENTS
    assert p0.census["teams"] == 1


def test_p0_raises_when_the_team_lane_is_not_five_distinct_agents(tmp_path):
    """A malformed group is a broken filter, not a row to drop quietly."""
    rows = _team("e|8B|natural|e0|PEER", 1)[:4]
    tables = _write(tmp_path, decisions=rows)
    with pytest.raises(ValueError, match="not 5 distinct agents"):
        PA.panel_p0(PA.load_rows(tables))


def test_p0_raises_when_two_members_share_an_agent_id(tmp_path):
    rows = _team("e|8B|natural|e0|PEER", 1)
    rows[4]["agent_id"] = 3
    tables = _write(tmp_path, decisions=rows)
    with pytest.raises(ValueError, match="not 5 distinct agents"):
        PA.panel_p0(PA.load_rows(tables))


def test_load_rows_refuses_an_ambiguous_episode_join(tmp_path):
    """Two episodes rows for one id that disagree on item_id would silently mis-join."""
    decisions = _team("e|8B|natural|e0|PEER", 1)
    episodes = [{"episode_id": "e|8B|natural|e0|PEER", "item_id": "a", "split": "test",
                 "family": "math500"},
                {"episode_id": "e|8B|natural|e0|PEER", "item_id": "b", "split": "test",
                 "family": "math500"}]
    tables = _write(tmp_path, decisions=decisions, episodes=episodes)
    with pytest.raises(ValueError, match="disagree on"):
        PA.load_rows(tables)


# --------------------------------------------------------------------------- P0d

def test_p0d_deduplicates_a_round0_decision_aliased_into_two_arms(tmp_path):
    """One generation, two arms: P0 counts it twice, P0d once, and says so."""
    shared = [f"root|a{a}" for a in range(PA.N_AGENTS)]
    peer = [_decision("item|8B|natural|e0|PEER", 0, a, decision_id=shared[a],
                      protocol="PEER", phase="root", aliased=True)
            for a in range(PA.N_AGENTS)]
    private = [_decision("item|8B|natural|e0|PRIVATE", 0, a, decision_id=shared[a],
                         protocol="PRIVATE", phase="root", aliased=True)
               for a in range(PA.N_AGENTS)]
    tables = _write(tmp_path, decisions=peer + private)
    rows = PA.load_rows(tables)
    p0 = PA.panel_p0(rows)
    p0d = PA.panel_p0d(rows, p0)

    assert p0.size == 10 and p0d.size == 5
    assert p0d.census["round0_replication"] == pytest.approx(2.0)
    assert p0d.census["multiplicity_histogram"] == {"2": 5}
    assert p0d.census["duplicate_rows"] == 5
    assert len(p0d.select(rows)) == p0d.size


def test_p0d_leaves_update_rounds_alone(tmp_path):
    tables = _write(tmp_path, decisions=_team("e|8B|natural|e0|PEER", 1))
    rows = PA.load_rows(tables)
    p0d = PA.panel_p0d(rows, PA.panel_p0(rows))
    assert p0d.census["rounds_1_3_replication"] == pytest.approx(1.0)
    assert p0d.census["round0_rows"] == 0


# --------------------------------------------------------------------------- P1 / P2

def test_p1_treats_ambiguous_as_missing_not_wrong(tmp_path):
    decisions = _team("e|8B|natural|e0|PEER", 1)
    labels = [{"decision_id": d["decision_id"],
               "evaluator_status": "ambiguous" if d["agent_id"] == 0 else "incorrect",
               "is_correct": False} for d in decisions]
    tables = _write(tmp_path, decisions=decisions, labels=labels)
    rows = PA.load_rows(tables)
    p1 = PA.panel_p1(rows, PA.panel_p0(rows))

    assert p1.size == 4
    assert p1.census["status_histogram"] == {"ambiguous": 1, "incorrect": 4}
    assert [r["correct"] for r in rows if r["agent_id"] == 0] == [None]


def test_p2_drops_a_team_with_one_ungradeable_member(tmp_path):
    complete = _team("a|8B|natural|e0|PEER", 1)
    censored = _team("b|8B|natural|e0|PEER", 1)
    labels = [{"decision_id": d["decision_id"], "evaluator_status": "correct",
               "is_correct": True} for d in complete]
    labels += [{"decision_id": d["decision_id"],
                "evaluator_status": "timeout" if d["agent_id"] == 2 else "correct",
                "is_correct": d["agent_id"] != 2} for d in censored]
    tables = _write(tmp_path, decisions=complete + censored, labels=labels)
    rows = PA.load_rows(tables)
    p2 = PA.panel_p2(rows, PA.panel_p0(rows))

    assert p2.size == 1 and p2.universe == 2
    assert p2.census["rows"] == PA.N_AGENTS
    assert p2.coverage_rate == pytest.approx(0.5)
    assert {r["episode_id"] for r in p2.select(rows)} == {"a|8B|natural|e0|PEER"}


def test_p2_reports_classed_teams_separately_from_graded_teams(tmp_path):
    """Grading and canonical-class extraction fail on different rows; P2 means graded."""
    decisions = _team("a|8B|natural|e0|PEER", 1)
    decisions[0]["answer_class"] = ""          # graded, but no class
    tables = _write(tmp_path, decisions=decisions)
    rows = PA.load_rows(tables)
    p2 = PA.panel_p2(rows, PA.panel_p0(rows))
    assert p2.size == 1
    assert p2.census["teams_all_classed"] == 0


# --------------------------------------------------------------------------- P3 / P5 / P8

def _arm_rows(protocol: str, items: list[str], *, checkpoint: str = "8B",
              cohort: str = "natural", rounds: tuple[int, ...] = (0, 1, 2, 3),
              budget_tokens: int = 1024) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        episode = f"{item}|{checkpoint}|{cohort}|e0|{protocol}"
        for round_ in rounds:
            out += _team(episode, round_, protocol=protocol, checkpoint=checkpoint,
                         cohort=cohort, budget_tokens=budget_tokens)
    return out


def test_p3_is_the_intersection_and_names_the_arms_it_found(tmp_path):
    decisions: list[dict[str, Any]] = []
    for arm in PA.ARMS_8B:
        items = ["i0", "i1"] if arm != "GATE" else ["i0"]     # GATE never ran i1
        decisions += _arm_rows(arm, items)
    tables = _write(tmp_path, decisions=decisions)
    rows = PA.load_rows(tables)
    p3 = PA.panel_p3(rows, PA.panel_p0(rows))

    assert p3.keys == frozenset({"i0"})
    assert p3.universe == 2 and p3.coverage_rate == pytest.approx(0.5)
    assert p3.census["arms_match_declared"] is True
    assert p3.census["cells_observed"] == p3.census["cells_expected"] == 7 * 1 * 4
    assert {r["item_id"] for r in p3.select(rows)} == {"i0"}


def test_p3_flags_a_missing_arm_rather_than_reporting_a_full_intersection(tmp_path):
    decisions: list[dict[str, Any]] = []
    for arm in PA.ARMS_8B[:-1]:                               # RANDOM_PROTECT never ran
        decisions += _arm_rows(arm, ["i0"])
    tables = _write(tmp_path, decisions=decisions)
    rows = PA.load_rows(tables)
    p3 = PA.panel_p3(rows, PA.panel_p0(rows))

    assert p3.size == 1                                       # the intersection still looks full
    assert p3.census["arms_match_declared"] is False          # but the census does not
    assert "RANDOM_PROTECT" not in p3.census["arms_found"]


def test_p5_excludes_round0_and_measures_why(tmp_path):
    decisions: list[dict[str, Any]] = []
    for checkpoint in ("8B", "32B"):
        for protocol in PA.ARMS_BUDGET:
            decisions += _arm_rows(protocol, ["i0"], checkpoint=checkpoint)
            # The high-budget arm reuses the T1024 root draws: same decision_id, same allowance.
            episode = f"i0|{checkpoint}|high_budget|e0|{protocol}"
            root = f"i0|{checkpoint}|natural|e0|{protocol}|r0"
            decisions += [_decision(episode, 0, a, decision_id=f"{root}|a{a}",
                                    protocol=protocol, phase="root", checkpoint=checkpoint,
                                    cohort="high_budget", aliased=True)
                          for a in range(PA.N_AGENTS)]
            for round_ in PA.BUDGET_ROUNDS:
                decisions += _team(episode, round_, protocol=protocol, checkpoint=checkpoint,
                                   cohort="high_budget", budget_tokens=4096)
    tables = _write(tmp_path, decisions=decisions)
    rows = PA.load_rows(tables)
    p5 = PA.panel_p5(rows, PA.panel_p0(rows))

    assert p5.keys == frozenset({"i0"})
    assert {r["round"] for r in p5.select(rows)} == set(PA.BUDGET_ROUNDS)
    assert p5.census["round0_high_budget_allowances"] == [1024]
    assert (p5.census["round0_shared_with_t1024"]
            == p5.census["round0_high_budget_decisions"] == 4 * PA.N_AGENTS)
    assert p5.census["cells_observed"] == p5.census["cells_expected"]


def test_p8_takes_the_protocols_both_scales_ran_and_reports_the_p3_overlap(tmp_path):
    """32B ran three protocols on a subset; P8 is that subset, not P3."""
    decisions: list[dict[str, Any]] = []
    for arm in PA.ARMS_8B:
        items = ["i0", "i1"] if arm in PA.ARMS_32B else ["i0"]   # 8B-only arms are test-only
        decisions += _arm_rows(arm, items)
    for arm in PA.ARMS_32B:
        decisions += _arm_rows(arm, ["i0", "i1"], checkpoint="32B")
    tables = _write(tmp_path, decisions=decisions)
    rows = PA.load_rows(tables)
    p0 = PA.panel_p0(rows)
    p3, p4 = PA.panel_p3(rows, p0), PA.panel_p4(rows, p0)
    p8 = PA.panel_p8(rows, p0, p3, p4)

    assert p8.keys == frozenset({"i0", "i1"})
    assert p3.keys == frozenset({"i0"})
    assert p8.census["items_also_in_p3"] == 1                   # strict P3 n P4 is smaller
    assert p8.census["items_in_p3_and_p4"] == 1
    assert p8.census["p4_items_contained"] is True
    assert p8.census["rows_by_checkpoint"]["8B"] == p8.census["rows_by_checkpoint"]["32B"]


# --------------------------------------------------------------------------- P6 / P7

def test_p6_closes_a_pair_through_parent_state_id(tmp_path):
    """forks.parent_decision_ids is null in v1, so the pair is closed on the content address."""
    parent = "state:abc"
    peer = _decision("i0|8B|natural|e0|PEER", 2, 0, decision_id="peer-child",
                     protocol="PEER", parent_state_id=parent)
    others = [_decision("i0|8B|natural|e0|PEER", 2, a, protocol="PEER",
                        parent_state_id=parent) for a in range(1, PA.N_AGENTS)]
    fork_child = _decision("i0|8B|fork|e0|PRIVATE", 2, 0, decision_id="fork-child",
                           protocol="PRIVATE", cohort=PA.FORK_COHORT, parent_state_id=parent)
    fork_mates = [_decision("i0|8B|fork|e0|PRIVATE", 2, a, protocol="PRIVATE",
                            cohort=PA.FORK_COHORT) for a in range(1, PA.N_AGENTS)]
    forks = [{"fork_id": "f0", "episode_id": "i0|8B|fork|e0|PRIVATE", "item_id": "i0",
              "round": 2, "focal_agent": 0, "child_decision_ids": ["fork-child"],
              "shared_prefix_hash": parent}]
    tables = _write(tmp_path, decisions=[peer, *others, fork_child, *fork_mates], forks=forks)
    rows = PA.load_rows(tables)
    p6 = PA.panel_p6(tables, rows)

    assert p6.keys == frozenset({"f0"})
    assert p6.census["pairs_missing_peer"] == p6.census["pairs_ambiguous"] == 0
    assert {r["decision_id"] for r in p6.select(rows)} == {"fork-child", "peer-child"}
    assert p6.census["pairs_graded_both_sides"] == 1


def test_p6_counts_an_unclosable_pair_instead_of_guessing(tmp_path):
    fork_child = _decision("i0|8B|fork|e0|PRIVATE", 2, 0, decision_id="fork-child",
                           protocol="PRIVATE", cohort=PA.FORK_COHORT,
                           parent_state_id="state:abc")
    mates = [_decision("i0|8B|fork|e0|PRIVATE", 2, a, protocol="PRIVATE",
                       cohort=PA.FORK_COHORT) for a in range(1, PA.N_AGENTS)]
    forks = [{"fork_id": "f0", "episode_id": "i0|8B|fork|e0|PRIVATE", "item_id": "i0",
              "round": 2, "focal_agent": 0, "child_decision_ids": ["fork-child"],
              "shared_prefix_hash": "state:abc"}]
    tables = _write(tmp_path, decisions=[fork_child, *mates], forks=forks)
    p6 = PA.panel_p6(tables, PA.load_rows(tables))

    assert p6.size == 0 and p6.census["pairs_missing_peer"] == 1
    assert p6.coverage_rate == pytest.approx(0.0)


def test_p7_calls_a_single_class_probe_degenerate(tmp_path):
    draws: list[dict[str, Any]] = []
    for probe, classes in (("receiver:p00", ["A"] * 4), ("receiver:p01", ["A", "A", "B", "A"])):
        for draw, answer in enumerate(classes):
            draws.append(_decision(f"RECEIVER.s{draw:03d}|{probe}", 0, 0,
                                   decision_id=f"{probe}|{draw}", protocol=PA.BANK_PROTOCOL,
                                   phase=PA.RECEIVER_PHASE, answer_class=answer))
    episodes = [{"episode_id": d["episode_id"], "item_id": d["episode_id"].split("|")[1],
                 "split": "test", "family": ""} for d in draws]
    tables = _write(tmp_path, decisions=draws, episodes=episodes)
    p7 = PA.panel_p7(PA.load_rows(tables))

    assert p7.keys == frozenset({"receiver:p01"})
    assert p7.census["degenerate_probes"] == 1
    assert p7.census["draws_per_probe"] == {"4": 2}
    assert p7.coverage_rate == pytest.approx(0.5)


# --------------------------------------------------------------------------- dynamic range

def test_dynamic_range_splits_the_bank_three_ways(tmp_path):
    banks = [{"item_id": "i0", "checkpoint": "8B", "split": "test", "n_draws": 16,
              "c_correct": 16},
             {"item_id": "i1", "checkpoint": "8B", "split": "test", "n_draws": 16,
              "c_correct": 0},
             {"item_id": "i2", "checkpoint": "8B", "split": "test", "n_draws": 16,
              "c_correct": 7},
             {"item_id": "i3", "checkpoint": "8B", "split": "train", "n_draws": 32,
              "c_correct": 16},
             {"item_id": "i4", "checkpoint": "32B", "split": "test", "n_draws": 16,
              "c_correct": 3}]
    tables = _write(tmp_path, decisions=_team("e|8B|natural|e0|PEER", 1), banks=banks)
    out = PA.dynamic_range(tables)

    assert (out["items"], out["all_correct"], out["all_wrong"], out["non_degenerate"]) \
        == (3, 1, 1, 1)
    assert out["draws_per_item"] == {"16": 3}
    # The doubled-pool banks are named even though none of them is in this panel.
    assert out["banks_with_unexpected_draw_count"] == ["i3"]
    assert out["matches_expected"] is False


# --------------------------------------------------------------------------- frozen v1

@FROZEN
def test_v1_census_matches_the_frozen_panel_sizes():
    report = PA.census(V1_TABLES)
    sizes = {name: panel["size"] for name, panel in report["panels"].items()}
    assert sizes == dict(PA.V1_PANEL_SIZES)
    assert report["mismatches"] == []


@FROZEN
def test_v1_p0_rescues_every_private_round0_team():
    rows = PA.load_rows(V1_TABLES)
    p0 = PA.panel_p0(rows)
    assert p0.census["teams"] == 11552
    assert p0.census["teams_rescued_from_size_filter"] == 480
    # Every rescued team is 5 rows, so minority_frame is short by exactly 2,400 rows.
    assert p0.size - 480 * PA.N_AGENTS == 55360
    assert p0.census["rows_without_any_label"] == 0


@FROZEN
def test_v1_round0_replication_is_not_a_constant():
    rows = PA.load_rows(V1_TABLES)
    p0d = PA.panel_p0d(rows, PA.panel_p0(rows))
    assert p0d.census["round0_replication"] == pytest.approx(5.866667, abs=1e-6)
    assert p0d.census["rounds_1_3_replication"] == pytest.approx(1.0)
    assert {k for k, v in p0d.census["multiplicity_histogram"].items() if k != "1"} \
        == {"3", "5", "7", "9"}


@FROZEN
def test_v1_dynamic_range_bounds_every_disagreement_statistic():
    out = PA.dynamic_range(V1_TABLES)
    assert (out["items"], out["all_correct"], out["all_wrong"], out["non_degenerate"]) \
        == (240, 123, 34, 83)
    assert out["draws_per_item"] == {"16": 240}
    assert len(out["banks_with_unexpected_draw_count"]) == 8   # D-11, all 8B train


@FROZEN
def test_v1_scale_category_is_96_items_on_three_protocols():
    """The plan's "declare it or delete the scale category" question, answered from the data."""
    panels = PA.build_panels(V1_TABLES)
    p8 = panels["P8"]
    assert p8.size == 96
    assert p8.census["items_also_in_p3"] == 60
    assert p8.census["arms_match_declared"] is True
    assert p8.census["rows_by_checkpoint"] == {"32B": 5760, "8B": 5760}
    assert p8.coverage_rate == pytest.approx(0.25)


@FROZEN
def test_v1_budget_contrast_does_not_exist_at_round0():
    rows = PA.load_rows(V1_TABLES)
    p5 = PA.panel_p5(rows, PA.panel_p0(rows))
    assert p5.census["round0_high_budget_allowances"] == [1024]
    assert p5.census["round0_shared_with_t1024"] == 320
    assert p5.census["round0_high_budget_aliased"] == 640
    assert set(p5.census["allowance_cells"]) == {
        f"{c}|{p}|T{t}" for c in ("8B", "32B") for p in PA.ARMS_BUDGET
        for t in (1024, 4096)}


@FROZEN
def test_v1_every_panel_row_filter_agrees_with_its_census():
    rows = PA.load_rows(V1_TABLES)
    panels = PA.build_panels(V1_TABLES, rows)
    for name, expected in (("P0", "size"), ("P0d", "size"), ("P1", "size"),
                           ("P2", "rows"), ("P3", "decision_rows"), ("P4", "decision_rows"),
                           ("P5", "decision_rows"), ("P6", "decision_rows"),
                           ("P7", "decision_rows"), ("P8", "decision_rows")):
        panel = panels[name]
        assert len(panel.select(rows)) == panel.census[expected], name
