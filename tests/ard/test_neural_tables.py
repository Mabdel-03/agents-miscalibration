"""WP6 shards → tables: ``ard.neural.tables`` on a synthetic capture store.

No real capture has run, so the fixture here writes shards through the **real** writer
path — ``capture._write_decision`` → ``storage.ShardWriter`` — driven by a fake engine
result.  Everything downstream of the ledger row (chunk layout, key strings, the
present/missing convention, the ``extra`` join block) is therefore the production format;
only the numbers inside the vectors are synthetic.  What is *not* covered: whether a real
GPU capture emits the anchors and ``extra`` keys this assumes, which only a real replay
can show.

The design the fixture reproduces is the one property of the real cohort that breaks a
naive adapter: a round-0 root decision is captured once and referenced by all three
protocol episodes of its item.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from agents_scaling.ard import compact as CP
from agents_scaling.ard import contract as K
from agents_scaling.ard.neural import anchors as AA
from agents_scaling.ard.neural import capture as C
from agents_scaling.ard.neural import geometry as G
from agents_scaling.ard.neural import tables as NT
from agents_scaling.study.neural import storage as S

HIDDEN = 16
BLOCKS = (8, 17)
ANCHORS = ("PROMPT_END", "PRE_FINAL")
PROTOCOLS = ("PRIVATE", "PEER", "BALANCED")
ITEMS = ("math500-aaa", "math500-bbb")
VOCAB = 32


# --------------------------------------------------------------------------- fixture


class FakeResult:
    """What ``engine.forward`` returns for one request, in the shape capture consumes."""

    def __init__(self, decision_id: str, offsets: Sequence[int], hidden: int = HIDDEN) -> None:
        self.seq_id = decision_id
        self.n_tokens = 64
        self.measurement_cost = {"sequence_tokens": 64, "share_seconds": 0.01}
        self._hidden = hidden
        self._offsets = tuple(offsets)
        self._decision_id = decision_id

    def residual_at(self, block: int, token_offset: int) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((self._decision_id, int(block), int(token_offset)))) % (2**32))
        return rng.normal(size=self._hidden).astype(np.float32)

    def logits_at(self, token_offset: int) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((self._decision_id, "logits", int(token_offset)))) % (2**32))
        return rng.normal(size=VOCAB).astype(np.float32)


class FakeEngine:
    hidden_size = HIDDEN
    blocks = BLOCKS


def _anchor_set(decision_id: str, *, pre_final_missing: bool = False) -> AA.AnchorSet:
    """All four contract anchors (``assert_four_anchors``); PRE_FINAL optionally unreached."""
    anchors = [
        AA.TagAnchor("PROMPT_END", 10, None, AA.CHANNEL_PROMPT, "prompt", 0, byte_offset=0),
        AA.TagAnchor("PRE_FINAL", None, AA.MISSING_ANCHOR_UNRESOLVED, AA.CHANNEL_NONE, "final_span", 0)
        if pre_final_missing else
        AA.TagAnchor("PRE_FINAL", 30, None, AA.CHANNEL_CONTENT, "final_span", 20, byte_offset=40),
        AA.TagAnchor("ANSWER_END", 34, None, AA.CHANNEL_CONTENT, "final_span", 24, byte_offset=60),
        AA.TagAnchor("REPORT_END", 40, None, AA.CHANNEL_CONTENT, "report_span", 30, byte_offset=80),
    ]
    return AA.AnchorSet(decision_id=decision_id, prompt_tokens=11, completion_tokens=30,
                        anchors=tuple(anchors), parse_status="ok", markers={"final": True},
                        report_end_source="tag", truncated=False, report_span=(11, 41))


def _decision_id(item: str, round_: int, agent: int, protocol: str) -> str:
    """Round 0 is the common neutral root: one decision id shared by the three protocols."""
    stem = f"{item}.r0.a{agent}" if round_ == 0 else f"{item}.{protocol}.r{round_}.a{agent}"
    return f"{abs(hash(stem)) % (16**16):016x}"


def design_rows(*, missing_pre_final: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """``decisions``-shaped rows for the synthetic design (tables_v2 field names)."""
    rows: list[dict[str, Any]] = []
    for item in ITEMS:
        for protocol in PROTOCOLS:
            episode_id = f"{item}|8B|natural|e0|{protocol}"
            cell_id = f"EPISODE.natural.{protocol}.8B.T1024.test.s000"
            for round_ in K.ROUNDS:
                for agent in range(K.N_AGENTS):
                    rows.append({
                        "decision_id": _decision_id(item, round_, agent, protocol),
                        "episode_id": episode_id, "cell_id": cell_id, "item_id": item,
                        "agent_id": agent, "round": round_, "protocol": protocol,
                        "cohort": "natural", "checkpoint": "8B", "split": "test",
                        "family": "math", "boundary_injected": False,
                        "action": "root" if round_ == 0 else "update",
                    })
    return rows


def write_store(run_root: Path, rows: Sequence[Mapping[str, Any]], *,
                missing_pre_final: tuple[str, ...] = (), shard: str = "ard36.s000") -> dict[str, Any]:
    """Write the fixture's shards through the production writer path.

    ``capture._write_decision`` is called directly (it is the function that turns a work
    item into ledger rows); using it rather than hand-built ``ActivationRow``s is what makes
    this fixture worth trusting.
    """
    config = C.CaptureConfig(revision="rev-synthetic", checkpoint="8B", shard_name=shard,
                             chunk_rows=8, anchors=ANCHORS,
                             entropy_anchors=frozenset({"PROMPT_END"}))
    engine = FakeEngine()
    stats: dict[str, Any] = {"rows": 0, "rows_skipped_committed": 0, "anchors_missing": {},
                             "entropy_rows": 0}
    seen: set[str] = set()
    items = C.work_list(rows)
    with S.ShardWriter(run_root, C.CAPTURE_STAGE, shard, chunk_rows=8) as writer:
        for w in items:
            if w.decision_id in seen:
                continue
            seen.add(w.decision_id)
            anchor_set = _anchor_set(w.decision_id,
                                     pre_final_missing=w.decision_id in missing_pre_final)
            record = {"identity": {"input_hash": f"nonce-{w.decision_id[:8]}",
                                   "model_revision": "rev-synthetic"}}
            result = FakeResult(w.decision_id, anchor_set.present_offsets)
            C._write_decision(writer, w, record, anchor_set, result, blocks=list(BLOCKS),
                              done=set(), config=config, engine=engine, stats=stats,
                              model_rev="rev-synthetic")
    return stats


@pytest.fixture()
def store(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    rows = design_rows()
    write_store(tmp_path, rows)
    return tmp_path, rows


# --------------------------------------------------------------------------- the store


def test_the_fixture_writes_the_production_chunk_layout(store):
    run_root, rows = store
    chunks = S.list_chunks(S.stage_dir(run_root, C.CAPTURE_STAGE))
    assert chunks, "no committed chunk"
    n_decisions = len({r["decision_id"] for r in rows})
    ledger = list(NT.iter_capture_rows(run_root, anchors=ANCHORS, blocks=BLOCKS))
    assert len(ledger) == n_decisions * len(BLOCKS) * len(ANCHORS)
    assert all(vector is not None and vector.shape == (HIDDEN,) for _row, vector in ledger)
    assert {row["anchor_kind"] for row, _ in ledger} == set(ANCHORS)
    # the two anchors the brief excludes were never written: config.anchors is the filter
    assert not list(NT.iter_capture_rows(run_root, anchors=("ANSWER_END", "REPORT_END")))


def test_a_duplicate_key_across_chunks_is_refused(tmp_path: Path):
    rows = design_rows()[:5]
    write_store(tmp_path, rows, shard="ard36.s000")
    write_store(tmp_path, rows, shard="ard36.s001")   # the same decisions written twice
    with pytest.raises(NT.TableError, match="duplicate key"):
        list(NT.iter_capture_rows(tmp_path, anchors=ANCHORS))


# --------------------------------------------------------------------------- fan-out


def test_a_root_decision_is_captured_once_and_reaches_all_three_protocol_episodes(store):
    run_root, rows = store
    root_ids = {r["decision_id"] for r in rows if r["round"] == 0}
    assert len(root_ids) == len(ITEMS) * K.N_AGENTS          # shared across the protocols
    sites = NT.decision_sites(rows)
    assert {len(sites[d]) for d in root_ids} == {3}
    assert {len(sites[r["decision_id"]]) for r in rows if r["round"] > 0} == {1}

    with_join = NT.neural_tables_from_shards(run_root, decisions=rows, anchors=ANCHORS)
    without = NT.neural_tables_from_shards(run_root, anchors=ANCHORS)
    r0 = [g for g in with_join["neural_geometry"] if g["round"] == 0 and not g["normalised"]]
    r0_without = [g for g in without["neural_geometry"] if g["round"] == 0 and not g["normalised"]]
    # 2 items x 3 protocols x 2 blocks x 2 anchors with the join; a third of that without it
    assert len(r0) == len(ITEMS) * len(PROTOCOLS) * len(BLOCKS) * len(ANCHORS)
    assert len(r0_without) == len(r0) // len(PROTOCOLS)
    assert {g["protocol"] for g in r0} == set(PROTOCOLS)


def test_round_one_has_a_previous_state_in_every_protocol_only_because_of_the_fan_out(store):
    run_root, rows = store
    with_join = NT.neural_tables_from_shards(run_root, decisions=rows, anchors=ANCHORS)
    r1 = [a for a in with_join["neural_agent"] if a["round"] == 1]
    assert r1 and all(a["has_previous_state"] for a in r1)
    assert {a["protocol"] for a in r1} == set(PROTOCOLS)

    without = NT.neural_tables_from_shards(run_root, anchors=ANCHORS)
    r1_without = [a for a in without["neural_agent"] if a["round"] == 1]
    assert sum(a["has_previous_state"] for a in r1_without) < len(r1_without)


# --------------------------------------------------------------------------- geometry


def test_team_rows_are_emitted_raw_and_normalised_on_one_key(store):
    run_root, rows = store
    tables = NT.neural_tables_from_shards(run_root, decisions=rows, anchors=ANCHORS)
    geometry = tables["neural_geometry"]
    keyed: dict[tuple, list[dict]] = {}
    for row in geometry:
        keyed.setdefault((row["cell_id"], row["episode_id"], row["round"], row["layer"],
                          row["anchor"]), []).append(row)
    assert keyed and all(len(v) == 2 for v in keyed.values())
    assert all(sorted(r["normalised"] for r in v) == [False, True] for v in keyed.values())
    n_groups = len(ITEMS) * len(PROTOCOLS) * len(K.ROUNDS) * len(BLOCKS) * len(ANCHORS)
    assert len(keyed) == n_groups
    for row in geometry:
        assert row["n_states"] == K.N_AGENTS and row["team_complete"] is True
        assert row["agent_ids"] == "0,1,2,3,4"
        assert row["rank_ceiling"] == min(HIDDEN, K.N_AGENTS - 1) == 4
        assert 1.0 <= row["participation_ratio"] <= row["rank_ceiling"]
        assert 0.0 < row["entropy_rank"] <= row["rank_ceiling"]
        assert 0.0 < row["leading_eigenvalue_share"] <= 1.0
        assert row["hidden_size"] == HIDDEN
        assert row["consumer_revision"] == "rev-synthetic"


def test_the_estimates_are_the_estimator_run_on_the_stored_vectors(store):
    """No re-implementation: the table must equal ``geometry.team_metrics`` on the shard."""
    run_root, rows = store
    tables = NT.neural_tables_from_shards(run_root, decisions=rows, anchors=ANCHORS)
    row = next(r for r in tables["neural_geometry"]
               if r["round"] == 2 and r["layer"] == 17 and r["anchor"] == "PROMPT_END"
               and not r["normalised"])
    ledger = {(r["StateSnapshot_id"], int(r["block"]), r["anchor_kind"]): v
              for r, v in NT.iter_capture_rows(run_root, anchors=ANCHORS)}
    states = []
    for agent in range(K.N_AGENTS):
        decision_id = next(d["decision_id"] for d in rows
                           if d["episode_id"] == row["episode_id"] and d["round"] == 2
                           and d["agent_id"] == agent)
        states.append(np.asarray(ledger[(decision_id, 17, "PROMPT_END")], dtype=np.float64))
    expected = G.team_metrics(np.stack(states), normalised=False)
    for key in ("participation_ratio", "entropy_rank", "leading_eigenvalue_share",
                "total_variance", "mean_pairwise", "min_pairwise", "max_pairwise",
                "cosine_distance"):
        assert row[key] == pytest.approx(expected[key])


def test_displacement_is_measured_against_the_same_agents_one_round_earlier(store):
    run_root, rows = store
    tables = NT.neural_tables_from_shards(run_root, decisions=rows, anchors=ANCHORS)
    ledger = {(r["StateSnapshot_id"], int(r["block"]), r["anchor_kind"]): v
              for r, v in NT.iter_capture_rows(run_root, anchors=ANCHORS)}
    agent_row = next(a for a in tables["neural_agent"]
                     if a["round"] == 1 and a["layer"] == 8 and a["anchor"] == "PRE_FINAL"
                     and a["agent_id"] == 3)
    previous_id = agent_row["previous_decision_id"]
    assert previous_id is not None
    now = np.asarray(ledger[(agent_row["decision_id"], 8, "PRE_FINAL")], dtype=np.float64)
    before = np.asarray(ledger[(previous_id, 8, "PRE_FINAL")], dtype=np.float64)
    assert agent_row["own_displacement"] == pytest.approx(float(np.linalg.norm(now - before)))
    assert -1.0 <= agent_row["displacement_alignment"] <= 1.0
    assert agent_row["near_zero_denominator"] is False

    round0 = [a for a in tables["neural_agent"] if a["round"] == 0]
    assert all(a["has_previous_state"] is False for a in round0)
    assert all(math.isnan(a["own_displacement"]) for a in round0)
    assert all(a["previous_decision_id"] is None for a in round0)


def test_the_team_row_carries_the_raw_displacement_mean_and_not_a_normalised_one(store):
    run_root, rows = store
    tables = NT.neural_tables_from_shards(run_root, decisions=rows, anchors=ANCHORS)
    pairs = {}
    for row in tables["neural_geometry"]:
        pairs.setdefault((row["episode_id"], row["round"], row["layer"], row["anchor"]),
                         {})[row["normalised"]] = row
    raw, unit = pairs[(f"{ITEMS[0]}|8B|natural|e0|PEER", 2, 17, "PROMPT_END")][False], \
        pairs[(f"{ITEMS[0]}|8B|natural|e0|PEER", 2, 17, "PROMPT_END")][True]
    assert raw["n_displacement_states"] == K.N_AGENTS
    assert raw["mean_own_displacement"] > 0
    assert unit["n_displacement_states"] == 0
    assert math.isnan(unit["mean_own_displacement"])
    assert math.isnan(unit["mean_displacement_alignment"])


# --------------------------------------------------------------------------- missingness


def test_an_unreached_anchor_is_a_row_with_no_geometry_never_a_zero_vector(tmp_path: Path):
    rows = design_rows()
    victim = next(r for r in rows if r["round"] == 2 and r["agent_id"] == 1
                  and r["protocol"] == "PEER")["decision_id"]
    write_store(tmp_path, rows, missing_pre_final=(victim,))
    tables = NT.neural_tables_from_shards(tmp_path, decisions=rows, anchors=ANCHORS)

    hurt = [a for a in tables["neural_agent"]
            if a["decision_id"] == victim and a["anchor"] == "PRE_FINAL"]
    assert len(hurt) == len(BLOCKS)
    for row in hurt:
        assert row["present"] is False
        assert row["missingness"] == AA.MISSING_ANCHOR_UNRESOLVED
        assert math.isnan(row["state_norm"]) and math.isnan(row["distance_to_peer_centroid"])
        assert row["n_team_states"] == K.N_AGENTS - 1
    # its PROMPT_END state is untouched, and the team it left is four states, not five
    assert all(a["present"] for a in tables["neural_agent"]
               if a["decision_id"] == victim and a["anchor"] == "PROMPT_END")
    team = [g for g in tables["neural_geometry"]
            if g["round"] == 2 and g["anchor"] == "PRE_FINAL" and g["protocol"] == "PEER"
            and g["item_id"] == next(r["item_id"] for r in rows if r["decision_id"] == victim)]
    assert team and all(g["n_states"] == K.N_AGENTS - 1 for g in team)
    assert all(g["team_complete"] is False for g in team)
    assert all(g["n_missing_states"] == 1 for g in team)
    assert all(g["missing_codes"] == AA.MISSING_ANCHOR_UNRESOLVED for g in team)
    assert all("4" not in g["agent_ids"].split(",")[1:2] for g in team)  # ids stay real


# --------------------------------------------------------------------------- joins


def test_agent_rows_join_to_decisions_and_carry_the_clustering_keys(store):
    """The agent grain is (decision, **episode**, layer, anchor), matching ``minority_frame``.

    A root decision is one captured state referenced by three episodes, and
    ``minority_frame`` already fans it out the same way (measured on this run's
    ``tables_v2``: 23,040 eligible rows over 19,200 distinct ``decision_id``s).  Joining on
    ``decision_id`` alone therefore fans out 1→3 at round 0; join on the pair.
    """
    run_root, rows = store
    tables = NT.neural_tables_from_shards(run_root, decisions=rows, anchors=ANCHORS)
    agent = tables["neural_agent"]
    by_reference = {(r["decision_id"], r["episode_id"]): r for r in rows}
    assert len(agent) == len(rows) * len(BLOCKS) * len(ANCHORS)
    for row in agent:
        source = by_reference[(row["decision_id"], row["episode_id"])]
        assert (row["item_id"], row["cell_id"]) == (source["item_id"], source["cell_id"])
        assert (row["round"], row["agent_id"]) == (source["round"], source["agent_id"])
    assert {r["decision_id"] for r in agent} == {r["decision_id"] for r in rows}

    # the three references of one root carry the same state, under three episode ids
    root = next(r["decision_id"] for r in rows if r["round"] == 0)
    copies = [a for a in agent if a["decision_id"] == root and a["layer"] == 8
              and a["anchor"] == "PROMPT_END"]
    assert len(copies) == len(PROTOCOLS)
    assert len({c["episode_id"] for c in copies}) == len(PROTOCOLS)
    assert len({round(c["state_norm"], 12) for c in copies}) == 1


def test_team_rows_join_to_round_summary_on_cell_episode_round(store):
    run_root, rows = store
    tables = NT.neural_tables_from_shards(run_root, decisions=rows, anchors=ANCHORS)
    round_summary_keys = {(r["cell_id"], r["episode_id"], r["round"]) for r in rows}
    geometry_keys = {(g["cell_id"], g["episode_id"], g["round"]) for g in tables["neural_geometry"]}
    assert geometry_keys == round_summary_keys
    minority_keys = {(r["cell_id"], r["episode_id"], r["round"], r["agent_id"]) for r in rows}
    agent_keys = {(a["cell_id"], a["episode_id"], a["round"], a["agent_id"])
                  for a in tables["neural_agent"]}
    assert agent_keys == minority_keys


def test_the_anchor_horizon_travels_with_every_row(store):
    run_root, rows = store
    tables = NT.neural_tables_from_shards(run_root, decisions=rows, anchors=ANCHORS)
    for table in ("neural_geometry", "neural_agent"):
        for row in tables[table]:
            assert row["feature_time"] == AA.feature_time_of(row["anchor"])
            assert row["last_allowed_round"] == AA.last_allowed_round(row["anchor"], row["round"])
    # both captured anchors are prospective; PROMPT_END is the entropy anchor of the pair
    assert {r["feature_time"] for r in tables["neural_geometry"]} == {AA.FEATURE_TIME_PRE}
    entropy = {r["anchor"] for r in tables["neural_agent"] if r["entropy_anchor"]}
    assert entropy == {"PROMPT_END"}
    assert all(r["entropy_bits"] > 0 for r in tables["neural_agent"]
               if r["anchor"] == "PROMPT_END" and r["present"])
    assert all(math.isnan(r["entropy_bits"]) for r in tables["neural_agent"]
               if r["anchor"] == "PRE_FINAL")


# --------------------------------------------------------------------------- the schema


def test_every_emitted_column_is_declared_in_compact_schemas(store):
    run_root, rows = store
    tables = NT.neural_tables_from_shards(run_root, decisions=rows, anchors=ANCHORS)
    for name in ("neural_geometry", "neural_agent"):
        emitted = set().union(*(set(r) for r in tables[name]))
        assert emitted <= set(CP.SCHEMAS[name]), sorted(emitted - set(CP.SCHEMAS[name]))
        projected = [CP._project(name, row) for row in tables[name]]   # raises on an extra
        assert all(set(p) == set(CP.SCHEMAS[name]) for p in projected)
        # nothing declared is dead: every column is actually produced
        assert set(CP.SCHEMAS[name]) == emitted


def test_the_three_declarations_of_each_table_agree():
    assert G.GEOMETRY_COLUMNS == CP.SCHEMAS["neural_geometry"]
    assert G.AGENT_COLUMNS == CP.SCHEMAS["neural_agent"]
    assert NT.geometry_table_columns() == CP.SCHEMAS["neural_geometry"]
    assert NT.agent_table_columns() == CP.SCHEMAS["neural_agent"]
    assert G.GEOMETRY_COLUMNS[: len(G.GEOMETRY_SPEC_COLUMNS)] == G.GEOMETRY_SPEC_COLUMNS
    assert G.AGENT_COLUMNS[: len(G.AGENT_SPEC_COLUMNS)] == G.AGENT_SPEC_COLUMNS
    for columns in (G.GEOMETRY_COLUMNS, G.AGENT_COLUMNS):
        assert len(columns) == len(set(columns))


def test_the_state_set_path_still_projects(tmp_path: Path):
    """``geometry.neural_tables`` (the constructed-states path) must stay projectable."""
    rng = np.random.default_rng(0)
    group = G.StateSet(item_id="i", protocol="PEER", round_=1, layer=17, anchor="PRE_FINAL",
                       states=rng.normal(size=(K.N_AGENTS, 8)),
                       agent_ids=tuple(range(K.N_AGENTS)), episode_id="e")
    tables = G.neural_tables([group])
    for name in ("neural_geometry", "neural_agent"):
        for row in tables[name]:
            declared = {k: v for k, v in row.items() if k in CP.SCHEMAS[name]}
            assert CP._project(name, declared)


# --------------------------------------------------------------------------- compact


def test_the_compact_row_builders_return_the_two_tables(store):
    run_root, rows = store
    shard_tables = CP.neural_shard_tables(run_root, rows, anchors=ANCHORS)
    geometry = CP.neural_geometry_rows(run_root, rows, shard_tables=shard_tables)
    agent = CP.neural_agent_rows(run_root, rows, shard_tables=shard_tables)
    assert geometry and agent
    assert geometry == shard_tables["neural_geometry"]
    assert agent == shard_tables["neural_agent"]


def test_a_run_without_a_capture_store_still_compacts(tmp_path: Path):
    assert CP.neural_geometry_rows(tmp_path, []) == []
    assert CP.neural_agent_rows(tmp_path, []) == []
    assert NT.neural_tables_from_shards(tmp_path)["coverage"]["rows"] == 0


# --------------------------------------------------------------------------- coverage / CLI


def test_coverage_counts_every_ledger_row_and_the_unreferenced_ones(store):
    run_root, rows = store
    eligible = [r for r in rows if r["protocol"] != "BALANCED"]
    tables = NT.neural_tables_from_shards(run_root, decisions=eligible, anchors=ANCHORS)
    coverage = tables["coverage"]
    n_decisions = len({r["decision_id"] for r in rows})
    assert coverage["rows"] == n_decisions * len(BLOCKS) * len(ANCHORS)
    assert coverage["present"] == coverage["rows"] and coverage["missing"] == 0
    # BALANCED-only decisions are now referenced by nothing and are counted, not dropped
    assert coverage["rows_unreferenced"] > 0
    assert {g["protocol"] for g in tables["neural_geometry"]} == {"PRIVATE", "PEER"}


def test_the_cli_writes_both_parquet_files(store, tmp_path: Path):
    run_root, rows = store
    out = tmp_path / "out"
    assert NT.main(["--run-root", str(run_root), "--out-dir", str(out),
                    "--anchor", "PROMPT_END", "--layer", "17"]) == 0
    assert (out / "neural_geometry.parquet").is_file()
    assert (out / "neural_agent.parquet").is_file()
    import pandas as pd

    frame = pd.read_parquet(out / "neural_geometry.parquet")
    assert set(frame["layer"]) == {17} and set(frame["anchor"]) == {"PROMPT_END"}


def test_the_load_estimate_is_the_documented_one():
    # the full eligible cohort: 19,200 distinct decisions x 3 layers x 2 anchors x 4096 fp16
    assert NT.load_estimate_bytes(19_200, n_layers=3, n_anchors=2) == 943_718_400
    assert NT.load_estimate_bytes(19_200, n_layers=1, n_anchors=2) / 3 == pytest.approx(104_857_600)


def test_split_and_family_come_from_episodes_because_decisions_has_neither(store):
    """Measured on ``tables_v2``: ``decisions.parquet`` has 50 columns, and ``split`` and
    ``family`` are not among them — they live in ``episodes.parquet``."""
    run_root, rows = store
    stripped = [{k: v for k, v in r.items() if k not in ("split", "family")} for r in rows]
    episodes = [{"episode_id": r["episode_id"], "split": "test", "family": "math"}
                for r in rows]

    without = NT.neural_tables_from_shards(run_root, decisions=stripped, anchors=ANCHORS)
    assert {g["split"] for g in without["neural_geometry"]} == {""}
    assert {g["family"] for g in without["neural_geometry"]} == {""}

    with_episodes = NT.neural_tables_from_shards(run_root, decisions=stripped,
                                                 episodes=episodes, anchors=ANCHORS)
    assert {g["split"] for g in with_episodes["neural_geometry"]} == {"test"}
    assert {a["family"] for a in with_episodes["neural_agent"]} == {"math"}
    # and nothing else moved
    assert len(with_episodes["neural_geometry"]) == len(without["neural_geometry"])
