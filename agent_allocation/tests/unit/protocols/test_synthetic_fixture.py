"""The two-item synthetic run shipped to agents 02 and 06: no drift, loads through the one codec, leaks nothing."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from agent_allocation.contracts import (
    Candidate, CheckpointView, GenerationKind, GenerationResult, ItemGraph, PublicTask, load_record,
)
from agent_allocation.prompts.render import render_messages_with_report
from agent_allocation.protocols.policy_cost import policy_ancestors, policy_cost_rows
from agent_allocation.protocols.synthetic import build_synthetic_run, read_payload, synthetic_run_payloads

from .helpers import ROOT

FIXTURE = ROOT / "tests" / "fixtures" / "protocols" / "two_item_run"


@lru_cache(maxsize=1)
def _run():
    return build_synthetic_run()


def _lines(name: str) -> list[dict]:
    return [json.loads(line) for line in read_payload(FIXTURE / name).decode("utf-8").splitlines()]


def test_committed_fixture_equals_regeneration():
    fresh = synthetic_run_payloads(_run())
    assert sorted(p.name for p in FIXTURE.iterdir()) == sorted(fresh)
    for name, data in fresh.items():
        assert read_payload(FIXTURE / name) == data, (
            f"{name} drifted: regenerate with python -m agent_allocation.protocols.synthetic {FIXTURE}")


def test_fixture_records_load_and_are_complete():
    graphs = [load_record(ItemGraph, row) for row in _lines("graphs.jsonl.gz")]
    tasks = {t.item_id: t for t in (load_record(PublicTask, row) for row in _lines("tasks.jsonl.gz"))}
    candidates = {c.node_id: c for c in (load_record(Candidate, row) for row in _lines("candidates.jsonl.gz"))}
    results = {r.node_id: r for r in (load_record(GenerationResult, row) for row in _lines("results.jsonl.gz"))}
    prompts = _lines("prompts.jsonl.gz")
    assert graphs == _run()["graphs"]
    assert [len(g.nodes) for g in graphs] == [138, 125] and set(tasks) == {g.item_id for g in graphs}
    node_ids = [n.node_id for g in graphs for n in g.nodes]
    assert set(candidates) == set(results) == set(node_ids) and [p["node_id"] for p in prompts] == node_ids
    assert all(c.candidate_id == c.node_id for c in candidates.values())
    for g in graphs:
        index = {n.node_id: n for n in g.nodes}
        for cp in g.checkpoints:
            assert policy_ancestors(g, cp) == cp.required_node_ids
        assert len(policy_cost_rows(g, results)) == len(g.checkpoints)
        # one prompt re-rendered from the loaded records equals the shipped prompt
        pooled_or_peer = next(n for n in g.nodes if n.kind in (GenerationKind.POOLED, GenerationKind.PEER))
        shipped = next(p for p in prompts if p["node_id"] == pooled_or_peer.node_id)
        messages, report = render_messages_with_report(
            pooled_or_peer, tasks[g.item_id], [(index[p], candidates[p]) for p in pooled_or_peer.parent_ids],
            seed=_run()["seed"])
        assert messages == shipped["messages"] and report == shipped["report"]
    availability = {p["availability"] for row in prompts for p in row["report"]["packets"]}
    assert availability == {"valid", "raw", "absent"}
    assert any(row["report"]["trimmed_packets"] for row in prompts)
    assert {cp.view for g in graphs for cp in g.checkpoints} == set(CheckpointView)


_FORBIDDEN = re.compile(
    r"\b(root|private|peer|deep|pooled|sample|communicate_then_private|team|round|horizon|agent|expert|majority|"
    r"minority|verified|checkpoint|partition|slot|archive|draw|iteration|cycle|gold|score)s?\b", re.IGNORECASE)


def test_rendered_prompts_carry_no_protocol_labels_ids_or_hidden_fields():
    run = _run()
    ids = {n.node_id for g in run["graphs"] for n in g.nodes}
    system = (ROOT / "prompts" / "system.txt").read_text()     # frozen pack text, identical for every condition
    for row in run["prompts"]:
        assert [m["role"] for m in row["messages"]] == ["system", "user"]
        assert row["messages"][0]["content"] == system
        user = row["messages"][1]["content"]
        assert not _FORBIDDEN.search(user), (row["node_id"], _FORBIDDEN.search(user).group(0))
        text = system + user
        assert not re.search(r"[0-9a-f]{64}", text)
        assert "math500" not in text and "mbpp_plus" not in text and row["item_id"] not in text
    assert ids.isdisjoint(json.dumps([r["messages"] for r in run["prompts"]]).split('"'))
