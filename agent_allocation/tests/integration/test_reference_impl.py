"""The reference implementations are only useful if they are right: check them against closed forms, the pack's
structural reference, the lead's golden node ids, and hand-made adversarial inputs."""
from __future__ import annotations

import gzip
from decimal import Decimal
from fractions import Fraction
from itertools import combinations

import pytest

from agent_allocation.contracts import STUDY_ID, GenerationKind, LedgerEvent, LedgerEventKind as K, public_task_hash

from tests.integration import reference_impl as ref
from tests.unit.test_node_identity_golden import COMMON, PINNED, TASK


def test_pass_at_k_matches_brute_force_subsets():
    for n in (1, 2, 5, 7):
        for c in range(n + 1):
            bank = [1] * c + [0] * (n - c)
            for k in range(1, n + 1):
                subsets = list(combinations(range(n), k))
                brute = Fraction(sum(any(bank[i] for i in s) for s in subsets), len(subsets))
                assert ref.pass_at_k(n, c, k) == brute


def test_pass_at_k_edges_and_identities():
    assert ref.pass_at_k(25, 0, 25) == 0 and ref.pass_at_k(25, 1, 25) == 1
    for c in range(26):
        assert ref.pass_at_k(25, c, 1) == Fraction(c, 25)            # pass@1 = mean draw correctness
    for bad in ((0, 0, 1), (5, 6, 1), (5, 1, 0), (5, 1, 6), (5, -1, 1)):
        with pytest.raises(ValueError):
            ref.pass_at_k(*bad)


@pytest.mark.parametrize("name,kwargs,grid_name,count", [
    ("main", {}, "main", 125), ("high", {"a": 4096, "high": True}, "high", 123),
    ("replication", {"reduced": True}, "replication", 57), ("extension", {"extension": True}, "main_extension", 138),
])
def test_reference_graph_matches_pack_reference_dag(reference_dag, grids, name, kwargs, grid_name, count):
    budget = kwargs.get("a", 1024)
    graph = ref.reference_graph(grids[grid_name], run_id=STUDY_ID, partition_id="main", task=TASK,
                                model=COMMON["model"], base_budget=budget, headroom=1024,
                                endpoint_lock_hash="x", decoding_config_hash="y", prompt_version="z")
    pack = reference_dag.build_plan(**kwargs)
    assert len(graph.nodes) == len(pack.nodes) == ref.expected_node_count(grids[grid_name]) == count
    assert set(graph.named) == set(pack.nodes)
    for cp_name, members in pack.checkpoints.items():
        assert cp_name in graph.checkpoints, cp_name
        assert [graph.named[m] for m in members] == list(graph.checkpoints[cp_name].candidate_ids), cp_name
        assert len(graph.ancestors_of(cp_name)) == len(pack.ancestors(cp_name)), cp_name
    for readable, node in pack.nodes.items():
        spec = graph.nodes[graph.named[readable]]
        assert spec.reasoning_budget == node.reasoning_budget
        assert [graph.named[p] for p in node.parents] == list(spec.parent_ids)


def test_reference_graph_reproduces_lead_golden_ids(grids):
    graph = ref.reference_graph(grids["main"], run_id=STUDY_ID, partition_id="main", task=TASK, model=COMMON["model"],
                                base_budget=1024, headroom=1024, endpoint_lock_hash=COMMON["endpoint_lock_hash"],
                                decoding_config_hash=COMMON["decoding_config_hash"],
                                prompt_version=COMMON["prompt_version"])
    names = {"root0": "root/0", "root4": "root/4", "private_0_1": "private/0/1", "deep_16": "deep/16"}
    for golden, readable in names.items():
        assert graph.named[readable] == PINNED[golden]
    # PINNED peer_3_1_2 is receiver 2 of the n=3 round-1 team
    assert graph.named["peer/3/1/2"] == PINNED["peer_3_1_2"]
    assert graph.nodes[PINNED["root0"]].input_manifest_hash == public_task_hash(TASK)


def test_transition_counts_identity_and_unknowns():
    prev = ["wrong", "correct", "correct", "wrong", "invalid", "unknown"]
    nxt = ["correct", "wrong", "correct", "wrong", "wrong", "correct"]
    t = ref.transition_counts(prev, nxt)
    assert (t.w_to_c, t.c_to_w, t.c_to_c, t.w_to_w, t.invalid_involved, t.unknown_involved) == (1, 1, 1, 1, 1, 1)
    assert not t.fully_known
    known_prev, known_next = prev[:4], nxt[:4]
    k = ref.transition_counts(known_prev, known_next)
    assert known_next.count("correct") - known_prev.count("correct") == k.w_to_c - k.c_to_w
    assert ref.correct_count_bounds(["correct", "unknown", "wrong", "unknown"]) == (1, 3)


def test_complete_link_refuses_one_direction_merges():
    one_way = lambda a, b: (a, b) == ("0.5", "1/2")                    # verify(0.5 -> 1/2) only
    both = lambda a, b: {a, b} == {"0.5", "1/2"}
    answers = ["1/2", "0.5", "3", "1/2"]
    assert ref.complete_link_classes(answers, one_way) == [[0, 3], [1], [2]]
    assert ref.complete_link_classes(answers, both) == [[0, 1, 3], [2]]
    assert ref.complete_link_classes(answers, lambda a, b: None) == [[0, 3], [1], [2]]


def test_delivered_vs_coverage_scope():
    rows = [
        {"policy_id": "peer", "family": "peer", "delivered_correct": True, "pool_states": ["wrong", "wrong"]},
        {"policy_id": "pooled", "family": "pooled", "delivered_correct": True, "pool_states": ["wrong"] * 4},
        {"policy_id": "unk", "family": "sample", "delivered_correct": True, "pool_states": ["unknown", "wrong"]},
        {"policy_id": "ok", "family": "cp", "delivered_correct": True, "pool_states": ["correct", "wrong"]},
    ]
    assert ref.delivered_vs_coverage_violations(rows) == ["peer: delivered correct with no correct pool member"]


def _ev(seq, kind, amount, attempt="a:1"):
    return LedgerEvent(seq, kind, Decimal(amount), "t", attempt_id=attempt, node_id=attempt.split(":")[0],
                       partition_id="main")


def test_ledger_checker_accepts_a_clean_stream():
    events = [_ev(1, K.RESERVE, "0.40", "a:1"), _ev(2, K.SENT, "0", "a:1"), _ev(3, K.RESERVE, "0.40", "b:1"),
              _ev(4, K.SETTLE, "0.10", "a:1"), _ev(5, K.RELEASE_UNSENT, "0", "b:1"), _ev(6, K.RESERVE, "0.50", "c:1"),
              _ev(7, K.SENT, "0", "c:1"), _ev(8, K.CHARGE_MAX, "0.50", "c:1")]
    state = ref.LedgerChecker(Decimal("1.00")).replay(events)
    assert state.violations == []
    assert state.settled == Decimal("0.60") and state.open_sent == state.open_unsent == 0


@pytest.mark.parametrize("events,needle", [
    ([_ev(1, K.RESERVE, "0.6"), _ev(2, K.RESERVE, "0.6", "b:1")], "exceeds ceiling"),
    ([_ev(1, K.RESERVE, "0.1"), _ev(2, K.SENT, "0"), _ev(3, K.RELEASE_UNSENT, "0")], "releases unknown money"),
    ([_ev(1, K.RESERVE, "0.1"), _ev(2, K.SETTLE, "0.1"), _ev(3, K.SETTLE, "0.1")], "after terminal"),
    ([_ev(1, K.RESERVE, "0.3"), _ev(2, K.SENT, "0"), _ev(3, K.CHARGE_MAX, "0.1")], "below the reservation"),
    ([_ev(2, K.RESERVE, "0.1"), _ev(1, K.RESERVE, "0.1", "b:1")], "seq not strictly"),
    ([_ev(1, K.SETTLE, "0.1")], "before RESERVE"),
    ([_ev(1, K.RESERVE, "0.1"), _ev(2, K.RESERVE, "0.1")], "second RESERVE"),
])
def test_ledger_checker_flags_adversarial_streams(events, needle):
    violations = ref.LedgerChecker(Decimal("1.00")).replay(events).violations
    assert any(needle in v for v in violations), violations


def test_ledger_checker_keeps_unknown_money_counted_after_illegal_release():
    events = [_ev(1, K.RESERVE, "0.9"), _ev(2, K.SENT, "0"), _ev(3, K.RELEASE_UNSENT, "0"), _ev(4, K.RESERVE, "0.9", "b:1")]
    state = ref.LedgerChecker(Decimal("1.00")).replay(events)
    assert any("exceeds ceiling" in v for v in state.violations)


def test_canary_scanner_sees_gzip_and_sqlite(tmp_path):
    import sqlite3
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "x.json.gz").write_bytes(gzip.compress(b'{"k": "CANARY-ONE"}'))
    con = sqlite3.connect(tmp_path / "state.sqlite")
    con.execute("create table t (v text)")
    con.execute("insert into t values ('CANARY-TWO')")
    con.commit()
    con.close()
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "gold.jsonl").write_text("CANARY-ONE")
    hits = ref.scan_tree_for_canaries(tmp_path, {"one": "CANARY-ONE", "two": "CANARY-TWO"},
                                      exclude_dirs=[tmp_path / "private"])
    assert sorted(hits) == [("raw/x.json.gz", "one"), ("state.sqlite", "two")]


def test_kind_counts_of_main_graph(grids):
    graph = ref.reference_graph(grids["main"], run_id=STUDY_ID, partition_id="main", task=TASK, model="m",
                                base_budget=1024, headroom=1024, endpoint_lock_hash="x", decoding_config_hash="y",
                                prompt_version="z")
    assert graph.kind_counts() == {GenerationKind.ROOT.value: 25, GenerationKind.PRIVATE.value: 40,
                                   GenerationKind.PEER.value: 56, GenerationKind.DEEP.value: 4}
