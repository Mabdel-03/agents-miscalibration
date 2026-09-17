"""MASTER_SPEC section 15: the twenty acceptance checks that block dispatch or claims.

Naming: ``test_s15_NN_<what>_reference`` runs now against contracts, config, the pack's reference DAG and agent
08's independent reference implementations. ``test_s15_NN_<what>_production`` exercises the real module through
its frozen signature and skips (``optional_module``) until the lead merges it. A production module that exists
but fails to import is an error, not a skip.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import json
from decimal import Decimal
from fractions import Fraction
from itertools import permutations

import pytest

from agent_allocation.config import grids_from_config
from agent_allocation.contracts import (
    CIRCUIT_BREAKING_TRANSPORT,
    RETRYABLE_TRANSPORT,
    STUDY_ID,
    CheckStatus,
    Candidate,
    CheckpointView,
    CompletionStatus,
    GenerationKind,
    GenerationResult,
    GenerationSpec,
    GradeResult,
    GradeStatus,
    LedgerEvent,
    ParseStatus,
    PolicyCheckpoint,
    PrivateEvalSpec,
    PublicCheckResult,
    PublicTask,
    SelectionResult,
    SelectionStatus,
    TransportStatus,
    Usage,
    canonical_hash,
    dump_record,
    load_record,
)

from tests.integration import adversarial
from tests.integration import helpers as h
from tests.integration import reference_impl as ref
from tests.integration.conftest import contract_lock, optional_module
from tests.integration.openrouter_stub import StubOpenRouter, cache_hit, ok

PRIVATE_FIELD_NAMES = {"answer", "answer_index", "gold", "gold_answer", "canonical_solution", "plus_input",
                       "base_input", "assertion", "private_tests", "private_record_ref", "private_hash",
                       "semantic_correct", "cot_content", "target", "solution"}


def _task(adv_items, benchmark: str, n: int = 0) -> PublicTask:
    return [t for t, _, _ in adv_items if t.benchmark == benchmark][n]


# =========================================================================================================
# 15.1  Same-prompt different-draw requests are distinct nodes / distinct network opportunities;
#       explicit shared roots are not re-generated.
# =========================================================================================================

def test_s15_01_identical_prompts_distinct_nodes_reference(reference_dag, grids, adv_items):
    fx = adversarial.load_json("prompt_identical_draws.json")
    specs = [load_record(GenerationSpec, r["record"]) for r in fx["roots"] + fx["canary_repeats"]]
    assert len({s.node_id for s in specs}) == len(specs) == 28
    # A prompt-hash-keyed cache would collapse all 25 roots into one opportunity: that is the hazard.
    assert len({(s.input_manifest_hash, s.kind, s.prompt_version) for s in specs[:25]}) == 1
    pack = reference_dag.build_plan()
    assert len(set(pack.checkpoints["sample/25"])) == 25
    assert pack.checkpoints["sample/5"] == pack.checkpoints["peer/5/0"] == pack.checkpoints["private_team/5/0"]
    graph = h.ref_graph(grids["main"], _task(adv_items, "math500"))
    assert graph.kind_counts()["root"] == 25                      # shared roots exist once


def test_s15_01_identical_prompts_distinct_nodes_production(grids, adv_items, cfg):
    grid_mod = optional_module("agent_allocation.protocols.grid")
    render = optional_module("agent_allocation.prompts.render")
    request = optional_module("agent_allocation.api.request")
    task = _task(adv_items, "math500")
    graph = h.production_graph(grid_mod, grids["main"], task, prompt_version=render.prompt_version())
    roots = [n for n in graph.nodes if n.kind is GenerationKind.ROOT]
    assert len(roots) == 25 and len({n.node_id for n in roots}) == 25
    messages = [render.render_messages(n, task, [], seed=cfg["seed"]) for n in roots]
    assert all(m == messages[0] for m in messages), "root prompts should be identical; identity must not need them to differ"
    bodies = [request.build_request_body(n, m, contract_lock(), cfg["sampling"]) for n, m in zip(roots, messages)]
    assert all(b == bodies[0] for b in bodies), "no per-draw nonce may be injected to defeat caching (spec 4.2)"
    assert "seed" not in bodies[0]
    for name in ("peer/5/0", "private_team/5/0", "sample/5"):
        assert h.checkpoint(graph, name).candidate_ids == tuple(r.node_id for r in roots[:5]), name


async def test_s15_01_identical_bodies_are_distinct_network_opportunities_production(grids, adv_items, cfg):
    client_mod = optional_module("agent_allocation.api.client")
    render = optional_module("agent_allocation.prompts.render")
    grid_mod = optional_module("agent_allocation.protocols.grid")
    task = _task(adv_items, "math500")
    roots = [n for n in h.production_graph(grid_mod, grids["main"], task).nodes if n.kind is GenerationKind.ROOT]
    stub = StubOpenRouter(default=lambda req: ok())
    client = client_mod.OpenRouterClient(api_key=None, transport=stub.transport())
    try:
        outcomes = [await client.generate(n, render.render_messages(n, task, [], seed=cfg["seed"]),
                                          attempt_id=f"{n.node_id}:1", lock=contract_lock(), sampling=cfg["sampling"])
                    for n in roots]
    finally:
        await client.aclose()
    assert len(stub.completion_requests) == 25
    assert {o.result.node_id for o in outcomes} == {n.node_id for n in roots}
    assert all(r.headers.get("x-openrouter-cache") == "false" for r in stub.completion_requests)


# =========================================================================================================
# 15.2  A roots record cannot overwrite a private or peer decision even if an episode label matches.
# =========================================================================================================

def test_s15_02_episode_label_collision_reference(run_paths):
    fx = adversarial.load_json("episode_label_collision.json")
    legacy_store = {}
    for row in fx["legacy_rows"]:
        legacy_store[row["episode_id"]] = row                     # the ARD36 defect: last writer wins
    assert len(legacy_store) == 1
    specs = [load_record(GenerationSpec, s["record"]) for s in fx["specs"]]
    node_store = {s.node_id: s for s in specs}
    assert len(node_store) == len(specs)
    paths = {run_paths.parsed(s.node_id) for s in specs} | {run_paths.grade(s.node_id) for s in specs}
    assert len(paths) == 2 * len(specs), "per-node artifact paths must not collide"
    kinds_by_label: dict[str, set[str]] = {}
    for s in specs:
        kinds_by_label.setdefault(adversarial.legacy_episode_label(s), set()).add(s.kind.value)
    assert any(len(k) >= 3 for k in kinds_by_label.values())


def test_s15_02_graph_ids_unique_and_extension_reuses_core_ids_production(grids, adv_items):
    grid_mod = optional_module("agent_allocation.protocols.grid")
    for task, _, _ in adv_items:
        main = h.production_graph(grid_mod, grids["main"], task)
        ext = h.production_graph(grid_mod, grids["main_extension"], task)
        high = h.production_graph(grid_mod, grids["high"], task, partition="high", budget=4096)
        main_ids = [n.node_id for n in main.nodes]
        ext_ids = {n.node_id for n in ext.nodes}
        assert len(set(main_ids)) == len(main_ids) == 125
        assert set(main_ids) <= ext_ids, "extension items must reuse (not regenerate) the 125 core nodes"
        assert not ({n.node_id for n in high.nodes} & set(main_ids))


def test_s15_02_state_store_keeps_colliding_label_records_production(run_paths):
    state_mod = optional_module("agent_allocation.storage.state")
    raw_mod = optional_module("agent_allocation.storage.raw")
    fx = adversarial.load_json("episode_label_collision.json")
    task = load_record(PublicTask, fx["item"])
    specs = [load_record(GenerationSpec, s["record"]) for s in fx["specs"]]
    main_specs = [s for s in specs if s.partition_id == "main"]
    store = state_mod.StateStore(run_paths)
    store.register_nodes(main_specs, wave_index=0)
    for s in main_specs:
        h.persist_success(store, raw_mod, run_paths, s, task, h.marked_content(s))
    assert {s.node_id for s in main_specs} <= store.terminal_node_ids()
    for s in main_specs:
        parsed = run_paths.parsed(s.node_id)
        if parsed.exists():
            data = json.loads(parsed.read_text())
            cand = load_record(Candidate, data["candidate"])
            assert cand.node_id == s.node_id and h.marker(s.node_id) in cand.visible_content


# =========================================================================================================
# 15.3  Full main DAG 125/item; high 123; reduced replication 57; CP adds 12 and POOLED adds 1.
# =========================================================================================================

@pytest.mark.parametrize("grid_name,count", [("main", 125), ("high", 123), ("replication", 57),
                                             ("main_extension", 138), ("pilot_r1", 12), ("pilot_r2", 15)])
def test_s15_03_node_counts_reference(grids, lean_cfg, grid_name, count):
    assert ref.expected_node_count(grids[grid_name]) == count
    task = PublicTask("x/1", "math500", "r", "p", adversarial.load_items()[0][0].output_kind)
    assert len(h.ref_graph(grids[grid_name], task).nodes) == count
    assert grids_from_config(lean_cfg)[grid_name] == grids[grid_name], "presets differ only in panels/budget"


def test_s15_03_extension_adds_exactly_cp12_pooled1_reference(grids, adv_items):
    task = _task(adv_items, "mmlu_pro")
    main, ext = h.ref_graph(grids["main"], task), h.ref_graph(grids["main_extension"], task)
    extra = [ext.nodes[i] for i in set(ext.nodes) - set(main.nodes)]
    assert sorted(s.kind.value for s in extra) == ["communicate_then_private"] * 12 + ["pooled"]


@pytest.mark.parametrize("grid_name,partition,budget", [("main", "main", 1024), ("high", "high", 4096),
                                                         ("replication", "replication", 1024),
                                                         ("main_extension", "main", 1024)])
def test_s15_03_production_graph_equals_independent_reference(grids, adv_items, grid_name, partition, budget):
    grid_mod = optional_module("agent_allocation.protocols.grid")
    for task, _, _ in adv_items:
        kw = dict(partition=partition, budget=budget)
        prod = h.production_graph(grid_mod, grids[grid_name], task, **kw)
        mine = h.ref_graph(grids[grid_name], task, **kw)
        prod_ids = [n.node_id for n in prod.nodes]
        assert len(prod_ids) == len(set(prod_ids)) == len(mine.nodes)
        missing = set(mine.nodes) - set(prod_ids)
        assert not missing, f"{task.item_id}: production lacks {len(missing)} reference nodes, e.g. " \
                            f"{[k for k, v in mine.named.items() if v in missing][:5]}"
        position = {nid: i for i, nid in enumerate(prod_ids)}
        for n in prod.nodes:
            assert all(position[p] < position[n.node_id] for p in n.parent_ids), "nodes not topologically ordered"
        headroom = h.headroom_for(task)
        assert all(n.completion_headroom == headroom for n in prod.nodes)


# =========================================================================================================
# 15.4  PRIVATE_SINGLE t=1 and all team r=0 reference the expected immutable roots; a 25-draw bank does not
#       replace a five-root team baseline.
# =========================================================================================================

def test_s15_04_root_references_reference(reference_dag, grids, adv_items):
    pack = reference_dag.build_plan()
    assert pack.checkpoints["private_single/1"] == ("root/0",)
    for n in (2, 3, 4, 5):
        assert pack.checkpoints[f"private_team/{n}/0"] == pack.checkpoints[f"peer/{n}/0"] == tuple(
            f"root/{j}" for j in range(n))
        assert len(pack.ancestors(f"peer/{n}/4")) == 5 * n
    graph = h.ref_graph(grids["main"], _task(adv_items, "bbh_logic4"))
    assert not ({graph.named[f"root/{j}"] for j in range(5, 25)} & graph.ancestors_of("peer/5/4"))


def test_s15_04_root_references_production(grids, adv_items):
    grid_mod = optional_module("agent_allocation.protocols.grid")
    task = _task(adv_items, "bbh_logic4")
    graph = h.production_graph(grid_mod, grids["main"], task)
    mine = h.ref_graph(grids["main"], task)
    roots = [mine.named[f"root/{j}"] for j in range(25)]
    assert h.checkpoint(graph, "private_single/1").candidate_ids == (roots[0],)
    for n in (2, 3, 4, 5):
        for fam in ("private_team", "peer"):
            cp = h.checkpoint(graph, f"{fam}/{n}/0")
            assert cp.candidate_ids == tuple(roots[:n]), cp.policy_id
            assert set(cp.required_node_ids) == set(roots[:n]), cp.policy_id
        for fam, rounds in (("peer", 4), ("private_team", 4)):
            for view in (CheckpointView.CURRENT, CheckpointView.ARCHIVE):
                cp = h.checkpoint(graph, f"{fam}/{n}/{rounds}", view)
                assert not (set(cp.candidate_ids) | set(cp.required_node_ids)) & set(roots[n:]), \
                    f"{cp.policy_id} uses bank roots beyond its team (a SAMPLE-25 bank is not a free archive)"


# =========================================================================================================
# 15.5  PEER n=5,R=4 has 25 required ancestors; PRIVATE_TEAM matches; no future or same-round peer messages.
# =========================================================================================================

def test_s15_05_ancestor_counts_and_synchrony_reference(reference_dag, grids, adv_items):
    graph = h.ref_graph(grids["main_extension"], _task(adv_items, "math500"))
    for n in (2, 3, 4, 5):
        for r in range(5):
            assert len(graph.ancestors_of(f"peer/{n}/{r}")) == n * (1 + r)
            assert len(graph.ancestors_of(f"private_team/{n}/{r}")) == n * (1 + r)
    assert len(graph.ancestors_of("cp/3")) == 20 and len(graph.ancestors_of("pooled/4")) == 5
    for n in (2, 3, 4, 5):
        for r in range(1, 5):
            prev = graph.checkpoints[f"peer/{n}/{r - 1}"].candidate_ids
            for nid in graph.checkpoints[f"peer/{n}/{r}"].candidate_ids:
                assert graph.nodes[nid].parent_ids == prev


def test_s15_05_policy_ancestors_production(grids, adv_items):
    grid_mod = optional_module("agent_allocation.protocols.grid")
    cost_mod = optional_module("agent_allocation.protocols.policy_cost")
    task = _task(adv_items, "math500")
    graph = h.production_graph(grid_mod, grids["main_extension"], task)
    mine = h.ref_graph(grids["main_extension"], task)
    for cp in graph.checkpoints:
        if cp.view is not CheckpointView.CURRENT:
            continue
        expected = mine.ancestors_of(cp.name)
        got = cost_mod.policy_ancestors(graph, cp)
        assert len(got) == len(set(got)), f"{cp.policy_id}: duplicated ancestors"
        assert set(got) == expected == set(cp.required_node_ids), cp.policy_id


def test_s15_05_rendered_prompts_see_only_parents_production(grids, adv_items, cfg):
    grid_mod = optional_module("agent_allocation.protocols.grid")
    render = optional_module("agent_allocation.prompts.render")
    task = _task(adv_items, "mmlu_pro")
    graph = h.production_graph(grid_mod, grids["main_extension"], task)
    by_id = {n.node_id: n for n in graph.nodes}
    cands = {n.node_id: h.valid_candidate(n, task) for n in graph.nodes}
    for node in graph.nodes:
        if not node.parent_ids:
            continue
        parents = [(by_id[p], cands[p]) for p in node.parent_ids]
        text = h.all_text(render.render_messages(node, task, parents, seed=cfg["seed"]))
        seen = set(h.MARK_RE.findall(text))
        assert seen == {h.marker(p) for p in node.parent_ids}, \
            f"{node.kind.value} slot {node.root_slot} round {node.update_round}: visible markers differ from parents"
        assert task.prompt in text, "the original question must be preserved every time"


# =========================================================================================================
# 15.6  PRIVATE_TEAM trajectories reuse across n; PEER trajectories at different n do not.
# =========================================================================================================

def _reuse_assertions(cands: dict[str, tuple[str, ...]]) -> None:
    for r in range(5):
        for n, m in ((2, 3), (3, 4), (4, 5)):
            assert cands[f"private_team/{m}/{r}"][:n] == cands[f"private_team/{n}/{r}"]
    for r in range(1, 5):
        for n in (2, 3, 4, 5):
            for m in (2, 3, 4, 5):
                if n != m:
                    assert not set(cands[f"peer/{n}/{r}"]) & set(cands[f"peer/{m}/{r}"])


def test_s15_06_reuse_rules_reference(grids, adv_items):
    graph = h.ref_graph(grids["main"], _task(adv_items, "mbpp_plus"))
    _reuse_assertions({k: v.candidate_ids for k, v in graph.checkpoints.items()})


def test_s15_06_reuse_rules_production(grids, adv_items):
    grid_mod = optional_module("agent_allocation.protocols.grid")
    graph = h.production_graph(grid_mod, grids["main"], _task(adv_items, "mbpp_plus"))
    _reuse_assertions({c.name: c.candidate_ids for c in graph.checkpoints if c.view is CheckpointView.CURRENT})


# =========================================================================================================
# 15.7  One usage object charged once in collection, once per logical occurrence in a policy ancestor set;
#       no reasoning-token double counting.
# =========================================================================================================

def _distinct_usage(graph) -> dict[str, Usage]:
    out = {}
    for i, nid in enumerate(graph.nodes):
        out[nid] = h.usage(prompt=1000 + 7 * i, completion=3000 + 11 * i, reasoning=500 + 13 * i,
                           cost=str(Decimal("0.000100") + Decimal(i) * Decimal("0.000003")))
    return out


def test_s15_07_collection_vs_policy_accounting_reference(grids, adv_items):
    graph = h.ref_graph(grids["main_extension"], _task(adv_items, "math500"))
    use = _distinct_usage(graph)
    per_policy = ref.policy_cost_reference(graph, use)
    collection = ref.collection_cost_reference(use)
    assert per_policy["peer/5/4"]["nominal_generations"] == 25
    # root/0 feeds sample, private_single, private_team, peer, cp and pooled: charged once in each policy ...
    root0 = graph.named["root/0"]
    containing = [k for k in per_policy if root0 in graph.ancestors_of(k)]
    assert len(containing) > 50
    # ... and exactly once in collection.
    assert collection == sum((u.actual_cost_usd for u in use.values()), Decimal(0))
    charged_somewhere = set().union(*(graph.ancestors_of(k) for k in per_policy))
    assert charged_somewhere == set(graph.nodes), "every collected node is charged to at least one policy"
    shared = graph.ancestors_of("sample/25") & graph.ancestors_of("private_single/25")
    assert shared == {root0}
    assert per_policy["sample/25"]["cost_usd"] + per_policy["private_single/25"]["cost_usd"] - use[root0].actual_cost_usd \
        == ref.collection_cost_reference({n: use[n] for n in graph.ancestors_of("sample/25") | graph.ancestors_of("private_single/25")})
    for row in per_policy.values():
        assert row["reasoning_tokens"] < row["completion_tokens"]       # subset, never added to the total
    missing = dict(use)
    missing[root0] = dataclasses.replace(use[root0], reasoning_tokens=None)
    assert ref.policy_cost_reference(graph, missing)["peer/2/1"]["reasoning_tokens"] is None


def _values(row) -> list:
    out = []
    for v in row.values():
        if isinstance(v, (int, float, Decimal, str)) and not isinstance(v, bool):
            with contextlib.suppress(Exception):
                out.append(Decimal(str(v)))
    return out


def test_s15_07_policy_cost_rows_production(grids, adv_items):
    grid_mod = optional_module("agent_allocation.protocols.grid")
    cost_mod = optional_module("agent_allocation.protocols.policy_cost")
    task = _task(adv_items, "math500")
    graph = h.production_graph(grid_mod, grids["main_extension"], task)
    mine = h.ref_graph(grids["main_extension"], task)
    use = _distinct_usage(mine)
    results = {nid: h.ok_result(mine.nodes[nid], "<final>1</final>", use=use[nid]) for nid in mine.nodes}
    rows = cost_mod.policy_cost_rows(graph, results)
    expected = ref.policy_cost_reference(mine, use)
    by_policy = {}
    for row in rows:
        by_policy.setdefault(row.get("policy_id"), []).append(row)
    assert by_policy, "policy_cost_rows returned nothing"
    for cp in graph.checkpoints:
        if cp.view is not CheckpointView.CURRENT or cp.policy_id not in by_policy:
            continue
        row = by_policy[cp.policy_id][0]
        want = expected[cp.name]
        values = _values(row)
        assert want["cost_usd"] in values, f"{cp.policy_id}: charged cost {want['cost_usd']} not in row {row}"
        assert Decimal(want["nominal_generations"]) in values, cp.policy_id
        double = want["prompt_tokens"] + want["completion_tokens"] + want["reasoning_tokens"]
        assert Decimal(double) not in values, f"{cp.policy_id}: reasoning tokens double counted"
    assert len(by_policy) >= len([c for c in graph.checkpoints if c.view is CheckpointView.CURRENT])


# =========================================================================================================
# 15.8  Concurrent reservations cannot exceed the ceiling; timeout, restart, delayed billing and 429 retain
#       correct reservations.
# =========================================================================================================

def test_s15_08_ledger_invariants_reference():
    K = ref.LedgerEventKind
    ev = lambda s, k, a, att: LedgerEvent(s, k, Decimal(a), "t", attempt_id=att)
    timeline = [ev(1, K.RESERVE, "0.02", "timeout:1"), ev(2, K.SENT, "0", "timeout:1"),
                ev(3, K.RESERVE, "0.02", "late_bill:1"), ev(4, K.SENT, "0", "late_bill:1"),
                ev(5, K.RESERVE, "0.02", "r429:1"), ev(6, K.SENT, "0", "r429:1"), ev(7, K.SETTLE, "0", "r429:1"),
                ev(8, K.RESERVE, "0.02", "r429:2")]
    checker = ref.LedgerChecker(Decimal("0.08"))
    state = checker.replay(timeline)
    assert state.violations == [] and state.open_sent == Decimal("0.04")      # timeout + late bill still held
    after_restart = checker.replay(timeline + [ev(9, K.RELEASE_UNSENT, "0", "timeout:1")])
    assert any("releases unknown money" in v for v in after_restart.violations)
    assert checker.replay(timeline + [ev(9, K.RESERVE, "0.02", "x:1")]).violations == [], "0.08 exactly fits"
    assert checker.replay(timeline + [ev(9, K.RESERVE, "0.03", "x:1")]).violations


async def _ledger(paths, ceiling: str):
    journal_mod = optional_module("agent_allocation.storage.journal")
    ledger_mod = optional_module("agent_allocation.runtime.ledger")
    journal = journal_mod.LedgerJournal(paths.ledger_journal, group_commit_ms=1)
    return journal, ledger_mod.CollectionLedger(journal, ceiling_usd=Decimal(ceiling), max_unresolved_usd=Decimal("2"))


async def test_s15_08_concurrent_reservations_tiny_ceiling_production(run_paths):
    journal, ledger = await _ledger(run_paths, "0.05")
    amount = Decimal("0.004")
    admitted = await asyncio.gather(*[ledger.reserve(attempt_id=f"{i:064x}:1", node_id=f"{i:064x}",
                                                     partition_id="main", amount_usd=amount) for i in range(64)])
    assert 1 <= sum(admitted) <= 12
    snap = ledger.snapshot()
    assert snap.settled + snap.in_flight + snap.unresolved <= snap.ceiling == Decimal("0.05")
    replay = ref.LedgerChecker(Decimal("0.05")).replay(list(journal.replay()))
    assert replay.violations == []
    assert replay.exposure == snap.settled + snap.in_flight + snap.unresolved


async def test_s15_08_sent_money_is_never_released_production(run_paths):
    journal, ledger = await _ledger(run_paths, "1.00")
    a = f"{1:064x}:1"
    assert await ledger.reserve(attempt_id=a, node_id=a[:64], partition_id="main", amount_usd=Decimal("0.30"))
    await ledger.mark_sent(a)
    with contextlib.suppress(Exception):
        await ledger.release_unsent(a)                                 # must refuse (raise) or be a no-op
    snap = ledger.snapshot()
    assert snap.in_flight + snap.unresolved + snap.settled >= Decimal("0.30")
    assert ref.LedgerChecker(Decimal("1.00")).replay(list(journal.replay())).violations == []


async def test_s15_08_restart_and_delayed_billing_retain_exposure_production(run_paths):
    journal, ledger = await _ledger(run_paths, "1.00")
    sent, unsent = f"{2:064x}:1", f"{3:064x}:1"
    for att in (sent, unsent):
        assert await ledger.reserve(attempt_id=att, node_id=att[:64], partition_id="main", amount_usd=Decimal("0.25"))
    await ledger.mark_sent(sent)
    # process dies here; a new coordinator reopens the same journal
    journal2, ledger2 = await _ledger(run_paths, "1.00")
    snap = ledger2.snapshot()
    assert snap.in_flight + snap.unresolved >= Decimal("0.25"), "restart forgot a sent reservation (unknown money)"
    await ledger2.settle(sent, Decimal("0.07"))                         # the bill arrives late
    snap = ledger2.snapshot()
    assert snap.settled == Decimal("0.07")
    assert ref.LedgerChecker(Decimal("1.00")).replay(list(journal2.replay())).violations == []


# =========================================================================================================
# 15.9  400-question results keep one population across checkpoints; unknown grades and unrun jobs not dropped.
# =========================================================================================================

def test_s15_09_fixed_population_reference():
    states = {f"item{i}": ["correct" if (i * j) % 3 == 0 else ("unknown" if j == 7 else "wrong") for j in range(25)]
              for i in range(8)}
    pops = {}
    for k in range(1, 26):
        rows = []
        for item, bank in states.items():
            lo, hi = ref.correct_count_bounds(bank)
            rows.append((item, ref.pass_at_k(25, lo, k), ref.pass_at_k(25, hi, k)))
        pops[k] = {r[0] for r in rows}
        assert all(lo <= hi for _, lo, hi in rows)
    assert all(p == pops[1] for p in pops.values()), "denominator must not shrink with k"


def test_s15_09_pass_at_k_exact_production():
    metrics = optional_module("agent_allocation.metrics")
    fn = getattr(metrics, "pass_at_k", None)
    if fn is None:
        passk = optional_module("agent_allocation.metrics.passk")
        fn = passk.pass_at_k
    for c in range(26):
        for k in range(1, 26):
            got = fn(25, c, k)
            want = ref.pass_at_k(25, c, k)
            if isinstance(got, (Fraction, int)):
                assert Fraction(got) == want, (c, k)
            else:
                assert abs(Fraction(str(got)) - want) < Fraction(1, 10**12), (c, k, got)
    assert fn(25, 3, 1) == pytest.approx(3 / 25) and fn(25, 1, 25) == 1


# =========================================================================================================
# 15.10 A malformed response is not retried as a fresh solution; infrastructure retries bounded and metered.
# =========================================================================================================

def test_s15_10_retry_classification_reference(cfg):
    assert cfg["scheduler"]["max_transport_attempts"] == 3
    assert cfg["scheduler"]["api_sdk_automatic_retries"] is False
    assert TransportStatus.OK not in RETRYABLE_TRANSPORT
    assert not RETRYABLE_TRANSPORT & CIRCUIT_BREAKING_TRANSPORT
    # every CompletionStatus is terminal model output: there is no path from a completed answer to a retry
    assert all(isinstance(s, CompletionStatus) for s in CompletionStatus)
    for case in adversarial.load_json("malformed_finals.json"):
        assert "valid" not in case["acceptable_parse_statuses"], case["case"]


@pytest.mark.xfail(strict=True, reason="D1: contracts.py:121-133 TransportStatus.ERROR_ENVELOPE is in neither "
                                       "RETRYABLE_TRANSPORT nor CIRCUIT_BREAKING_TRANSPORT (see docs/requests/08.md)")
def test_s15_10_every_non_ok_transport_status_has_a_declared_policy():
    undeclared = {s for s in TransportStatus if s is not TransportStatus.OK} - RETRYABLE_TRANSPORT - CIRCUIT_BREAKING_TRANSPORT
    assert not undeclared, undeclared


async def test_s15_10_malformed_final_is_a_completed_transport_ok_production(adv_items, cfg):
    client_mod = optional_module("agent_allocation.api.client")
    task = _task(adv_items, "mmlu_pro")
    node = next(iter(h.ref_graph(grids_from_config(cfg)["main"], task).nodes.values()))
    stub = StubOpenRouter([ok(content="<rationale>two answers</rationale><final>A</final><final>B</final>")])
    client = client_mod.OpenRouterClient(api_key=None, transport=stub.transport())
    try:
        out = await client.generate(node, [{"role": "user", "content": "q"}], attempt_id=f"{node.node_id}:1",
                                    lock=contract_lock(), sampling=cfg["sampling"])
    finally:
        await client.aclose()
    assert len(stub.completion_requests) == 1, "the client must never retry internally"
    assert out.result.transport_status is TransportStatus.OK
    assert out.result.completion_status is CompletionStatus.COMPLETED


def test_s15_10_extraction_never_repairs_malformed_finals_production(adv_items, cfg):
    extraction = optional_module("agent_allocation.extraction.final")
    tasks = {t.output_kind.value: t for t, _, _ in adv_items}
    for case in adversarial.load_json("malformed_finals.json"):
        task = tasks[case["output_kind"]]
        node = next(iter(h.ref_graph(grids_from_config(cfg)["pilot_r1"], task).nodes.values()))
        result = dataclasses.replace(h.ok_result(node, case["visible_content"]), finish_reason=case["finish_reason"],
                                     completion_status=CompletionStatus.LENGTH if case["finish_reason"] == "length"
                                     else CompletionStatus.COMPLETED)
        cand = extraction.extract_candidate(result, node, task)
        assert cand.parse_status is not ParseStatus.VALID, case["case"]
        if case["expected_parse_status"]:
            assert cand.parse_status.value == case["expected_parse_status"], case["case"]
        assert cand.candidate_id == cand.node_id == node.node_id
        assert cand.visible_content == case["visible_content"], "raw output must be preserved regardless of compliance"


# =========================================================================================================
# 15.11 Selector API cannot access private evaluation fields; canaries never appear in request logs.
# =========================================================================================================

def test_s15_11_public_records_carry_no_private_fields_reference(adv_items, canaries):
    for cls in (PublicTask, Candidate, PolicyCheckpoint, PublicCheckResult, SelectionResult, GenerationSpec):
        names = {f.name for f in dataclasses.fields(cls)}
        assert not names & (PRIVATE_FIELD_NAMES - {"private_hash"}), (cls.__name__, names & PRIVATE_FIELD_NAMES)
    for task, private, record in adv_items:
        assert ref.text_contains_canary(json.dumps(dump_record(task)), canaries) == []
        with pytest.raises(ValueError):
            load_record(PublicTask, {**dump_record(task), "answer": private.gold_answer})


def test_s15_11_selector_signature_has_no_private_inputs_production():
    readout = optional_module("agent_allocation.selection.readout")
    sig = inspect.signature(readout.select_checkpoint)
    assert list(sig.parameters)[:4] == ["checkpoint", "task", "candidates", "public_checks"]
    for name, param in sig.parameters.items():
        annotation = str(param.annotation)
        assert "PrivateEvalSpec" not in annotation and "GradeResult" not in annotation, name
        assert name not in PRIVATE_FIELD_NAMES and "gold" not in name and "grade" not in name, name


def test_s15_11_rendered_prompts_and_bodies_have_no_canaries_production(grids, adv_items, cfg, canaries):
    grid_mod = optional_module("agent_allocation.protocols.grid")
    render = optional_module("agent_allocation.prompts.render")
    request = optional_module("agent_allocation.api.request")
    for task, _, _ in adv_items:
        graph = h.production_graph(grid_mod, grids["main_extension"], task)
        by_id = {n.node_id: n for n in graph.nodes}
        cands = {n.node_id: h.valid_candidate(n, task) for n in graph.nodes}
        for node in graph.nodes:
            msgs = render.render_messages(node, task, [(by_id[p], cands[p]) for p in node.parent_ids], seed=cfg["seed"])
            body = request.build_request_body(node, msgs, contract_lock(), cfg["sampling"])
            blob = json.dumps(body, ensure_ascii=False)
            assert ref.text_contains_canary(blob, canaries) == [], (task.item_id, node.kind.value)
    assert all(k.lower() != "authorization" for k in request.REQUEST_HEADERS)


async def test_s15_11_key_and_canaries_absent_from_outcome_and_raw_production(run_paths, adv_items, cfg, canaries):
    client_mod = optional_module("agent_allocation.api.client")
    raw_mod = optional_module("agent_allocation.storage.raw")
    task = _task(adv_items, "mbpp_plus")
    node = next(iter(h.ref_graph(grids_from_config(cfg)["pilot_r1"], task).nodes.values()))
    stub = StubOpenRouter([ok(content="<final>def add_two(a, b):\n    return a + b</final>")])
    client = client_mod.OpenRouterClient(api_key=adversarial.fake_api_key(), transport=stub.transport())
    try:
        out = await client.generate(node, [{"role": "user", "content": task.prompt}], attempt_id=f"{node.node_id}:1",
                                    lock=contract_lock(), sampling=cfg["sampling"])
    finally:
        await client.aclose()
    sent = stub.completion_requests[0]
    assert adversarial.fake_api_key() in sent.headers.get("authorization", ""), "key must travel only in the header"
    assert ref.text_contains_canary(sent.body_text, canaries) == []
    assert ref.text_contains_canary(json.dumps(out.request_body, default=str), canaries) == []
    assert ref.text_contains_canary(json.dumps(dict(out.response_headers)), canaries) == []
    raw_mod.write_raw_response(run_paths, f"{node.node_id}:1", out)
    hits = ref.scan_tree_for_canaries(run_paths.run_dir, canaries)
    hits += ref.scan_tree_for_canaries(run_paths.local_state_dir, canaries)
    assert hits == []


# =========================================================================================================
# 15.12 MBPP public selection never uses private outcomes; hidden reference code inaccessible; attacks contained.
# =========================================================================================================

def test_s15_12_public_private_split_reference(adv_items):
    for task, private, record in adv_items:
        if task.benchmark != "mbpp_plus":
            continue
        assert len(task.public_examples) == 1 and task.public_examples[0] in task.prompt
        assert record["assertion"] not in task.prompt and record["canonical_solution"] not in task.prompt
        assert private.gold_answer is None and private.private_record_ref
    names = {f.name for f in dataclasses.fields(PublicCheckResult)}
    assert not names & {"semantic_correct", "private_eval_hash", "private_hash", "grade"}


# (code, contained_means_public_case_passes): a contained sandbox makes the benign branch run.
ATTACKS = {
    "read_key_env": ("import os\ndef add_two(a, b):\n    return -1 if os.environ.get('OPENROUTER_API_KEY') else a + b", True),
    "read_host_fixture": ("def add_two(a, b):\n    try:\n        open(%r).read()\n        return -1\n"
                          "    except OSError:\n        return a + b" % str(adversarial.FIXTURE_DIR / "items.jsonl"), True),
    "network": ("import socket\ndef add_two(a, b):\n    try:\n        socket.create_connection(('1.1.1.1', 443), timeout=2)\n"
                "        return -1\n    except OSError:\n        return a + b", True),
    "fork_bomb": ("import os\ndef add_two(a, b):\n    while True:\n        os.fork()", False),
    "infinite_loop": ("def add_two(a, b):\n    while True:\n        pass", False),
    "stdout_flood": ("def add_two(a, b):\n    print('x' * 10_000_000)\n    return a + b", True),
}


@pytest.mark.sandbox
@pytest.mark.parametrize("attack", sorted(ATTACKS))
def test_s15_12_public_checks_contain_attacks_production(adv_items, attack, monkeypatch, canaries, cfg):
    public = optional_module("agent_allocation.grading.code.public")
    monkeypatch.setenv("OPENROUTER_API_KEY", adversarial.fake_api_key())
    task = _task(adv_items, "mbpp_plus")
    node = next(iter(h.ref_graph(grids_from_config(cfg)["pilot_r1"], task).nodes.values()))
    code, contained_passes = ATTACKS[attack]
    cand = h.valid_candidate(node, task, content=f"<final>{code}</final>", final=code)
    res = public.run_public_checks(cand, task)
    assert res.status is CheckStatus.OK, f"{attack}: a hostile candidate is a candidate outcome, not {res.status}"
    assert res.cases_total == len(task.public_examples) == 1
    if contained_passes:
        assert res.cases_passed == 1, f"{attack} escaped containment or broke the checker: {res.case_outcomes}"
    else:
        assert res.cases_passed == 0 and res.case_outcomes[0] in ("timeout", "error", "fail"), res.case_outcomes
    assert ref.text_contains_canary(json.dumps(dump_record(res)), canaries) == []


# =========================================================================================================
# 15.13 Text answer classes never use gold; MATH asymmetry cannot create a one-direction merge.
# =========================================================================================================

def test_s15_13_complete_link_reference():
    one_way = lambda a, b: (a, b) == ("0.5", "1/2")
    answers = ["1/2", "1/2", "0.5", "0.5", "3", "3", "3"]
    classes = ref.complete_link_classes(answers, one_way)
    assert max(classes, key=len) == [4, 5, 6], "without a two-way merge the '3' class wins 3 vs 2 vs 2"
    symmetric = lambda a, b: {a, b} == {"0.5", "1/2"}
    assert max(ref.complete_link_classes(answers, symmetric), key=len) == [0, 1, 2, 3]


def _math_pool(task, answers, cfg):
    grid = grids_from_config(cfg)["main"]
    graph = h.ref_graph(grid, task)
    ids = graph.checkpoints[f"sample/{len(answers)}"].candidate_ids
    cands = {nid: h.valid_candidate(graph.nodes[nid], task, content=f"<final>{a}</final>", final=a)
             for nid, a in zip(ids, answers)}
    cp = PolicyCheckpoint(policy_id=f"main/{task.item_id}/sample/{len(answers)}/current", item_id=task.item_id,
                          family="sample", team_size=len(answers), total_cycles=len(answers), peer_rounds=0,
                          candidate_ids=tuple(ids), required_node_ids=tuple(ids), selector_name="plurality",
                          selector_version="v1", partition_id="main", name=f"sample/{len(answers)}")
    return cp, cands


def test_s15_13_selector_refuses_one_direction_merge_production(adv_items, cfg):
    readout = optional_module("agent_allocation.selection.readout")
    task = _task(adv_items, "math500")
    answers = ["1/2", "1/2", "0.5", "0.5", "3", "3", "3"]
    cp, cands = _math_pool(task, answers, cfg)
    one_way = lambda a, b: True if (a, b) == ("0.5", "1/2") else (None if {a, b} == {"0.5", "1/2"} else False)
    res = readout.select_checkpoint(cp, task, cands, {}, seed=cfg["seed"], math_equivalent=one_way)
    assert res.selected_candidate_id is not None
    assert cands[res.selected_candidate_id].final_answer == "3", "a one-direction equivalence merged classes"
    symmetric = lambda a, b: {a, b} == {"0.5", "1/2"} or a == b
    res2 = readout.select_checkpoint(cp, task, cands, {}, seed=cfg["seed"], math_equivalent=symmetric)
    assert cands[res2.selected_candidate_id].final_answer in {"1/2", "0.5"}


def test_s15_13_selection_independent_of_candidate_mapping_order_production(adv_items, cfg):
    readout = optional_module("agent_allocation.selection.readout")
    task = _task(adv_items, "mmlu_pro")
    cp, cands = _math_pool(task, ["A", "B", "A", "B", "C"], cfg)
    chosen = set()
    for order in list(permutations(cands))[:24]:
        res = readout.select_checkpoint(cp, task, {k: cands[k] for k in order}, {}, seed=cfg["seed"])
        chosen.add(res.selected_candidate_id)
    assert len(chosen) == 1, "tie-break depends on dict/completion order"


# =========================================================================================================
# 15.14 Unknown grader status stays unknown; parser failure and valid wrong answer distinguishable.
# =========================================================================================================

def test_s15_14_tri_state_reference():
    rows = {r["case"]: (load_record(GradeResult, r["record"]), r["expected_state"])
            for r in adversarial.load_jsonl("grades_unknown.jsonl")}
    for case, (grade, state) in rows.items():
        assert ref.tri_state(grade) == state, case
        assert load_record(GradeResult, dump_record(grade)).semantic_correct is grade.semantic_correct
    t = ref.transition_counts(["invalid", "wrong", "unknown"], ["correct", "correct", "correct"])
    assert (t.w_to_c, t.invalid_involved, t.unknown_involved) == (1, 1, 1)


@pytest.mark.xfail(strict=True, reason="D3: contracts.py:478-488 GradeResult accepts status/correctness combinations "
                                       "that contradict the tri-state rule (see docs/requests/08.md)")
def test_s15_14_grade_result_rejects_inconsistent_tri_state():
    with pytest.raises((ValueError, TypeError)):
        GradeResult("c", GradeStatus.EVALUATOR_ERROR, False, None, "v1", "h")


def test_s15_14_text_grader_keeps_unknown_and_invalid_distinct_production(adv_items, cfg):
    grade_mod = optional_module("agent_allocation.grading.text.grade")
    grid = grids_from_config(cfg)["pilot_r1"]
    for task, private, _ in adv_items:
        if task.benchmark == "mbpp_plus":
            continue
        node = next(iter(h.ref_graph(grid, task).nodes.values()))
        missing = dataclasses.replace(h.valid_candidate(node, task, content="no final here"),
                                      final_answer=None, parse_status=ParseStatus.MISSING)
        g = grade_mod.grade_text(missing, private)
        assert g.status is GradeStatus.INVALID_OUTPUT and g.operational_success is False, task.item_id
        wrong_answer = "Z-not-the-gold" if task.output_kind.value == "math" else (
            "True" if private.gold_answer == "False" else ("A" if private.gold_answer != "A" else "B"))
        wrong = h.valid_candidate(node, task, content=f"<final>{wrong_answer}</final>", final=wrong_answer)
        gw = grade_mod.grade_text(wrong, private)
        if private.gold_answer and private.gold_answer.startswith("CANARY"):
            assert gw.semantic_correct is not True
            continue
        assert gw.status is GradeStatus.GRADED and gw.semantic_correct is False, task.item_id
        assert (g.status, g.semantic_correct) != (gw.status, gw.semantic_correct)


def test_s15_14_math_grader_exception_is_unknown_not_wrong_production(adv_items, monkeypatch):
    """Evaluator outage injection: every Math-Verify entry point raises. The verdict must be unknown, never False.
    (Patched in-process and in module-level aliases; a grader that *spawns* fresh worker processes would not see
    the patch -- if this fails for that reason, expose a fault hook rather than deleting the test.)"""
    import sys
    import math_verify
    grade_mod = optional_module("agent_allocation.grading.text.grade")

    def outage(*args, **kwargs):
        raise RuntimeError("injected evaluator outage")

    originals = {name: getattr(math_verify, name) for name in ("parse", "verify") if hasattr(math_verify, name)}
    for name in originals:
        monkeypatch.setattr(math_verify, name, outage)
    for modname, mod in list(sys.modules.items()):
        if modname.startswith("agent_allocation.") and mod is not None:
            for attr, value in list(vars(mod).items()):
                if any(value is orig for orig in originals.values()):
                    monkeypatch.setattr(mod, attr, outage)
    task, private, _ = next((t, p, r) for t, p, r in adv_items if t.benchmark == "math500" and p.gold_answer == "4")
    node = next(iter(h.ref_graph(grids_from_config(adversarial_config())["pilot_r1"], task).nodes.values()))
    cand = h.valid_candidate(node, task, content="<final>4</final>", final="4")
    grade = grade_mod.grade_text(cand, private)
    assert grade.status is GradeStatus.EVALUATOR_ERROR, grade
    assert grade.semantic_correct is None and grade.operational_success is not False, grade


def adversarial_config():
    from agent_allocation.config import load_config
    from tests.integration.conftest import PROJECT
    return load_config(PROJECT / "configs" / "balanced150.json")


# =========================================================================================================
# 15.15 Missing reasoning usage is null, not zero; a response-cache HIT blocks the partition.
# =========================================================================================================

def test_s15_15_null_usage_and_cache_hit_policy_reference():
    u = Usage(10, 20, None, None, Decimal("0.1"))
    assert load_record(Usage, dump_record(u)).reasoning_tokens is None
    assert TransportStatus.CACHE_HIT in CIRCUIT_BREAKING_TRANSPORT
    assert TransportStatus.CACHE_HIT not in RETRYABLE_TRANSPORT
    assert ref.binding_cap(1024, None, "stop") is None


async def test_s15_15_client_null_reasoning_and_cache_hit_production(adv_items, cfg):
    client_mod = optional_module("agent_allocation.api.client")
    task = _task(adv_items, "math500")
    nodes = list(h.ref_graph(grids_from_config(cfg)["pilot_r1"], task).nodes.values())
    stub = StubOpenRouter([ok(reasoning_tokens=None, cached_tokens=None), cache_hit()])
    client = client_mod.OpenRouterClient(api_key=None, transport=stub.transport())
    try:
        first = await client.generate(nodes[0], [{"role": "user", "content": "q"}], attempt_id=f"{nodes[0].node_id}:1",
                                      lock=contract_lock(), sampling=cfg["sampling"])
        second = await client.generate(nodes[1], [{"role": "user", "content": "q"}], attempt_id=f"{nodes[1].node_id}:1",
                                       lock=contract_lock(), sampling=cfg["sampling"])
    finally:
        await client.aclose()
    assert first.result.transport_status is TransportStatus.OK
    assert first.result.usage is not None and first.result.usage.reasoning_tokens is None
    assert first.result.usage.cached_prompt_tokens is None
    assert second.result.transport_status is TransportStatus.CACHE_HIT


# =========================================================================================================
# 15.16 A larger nominal budget is not automatically reported as more realized reasoning.
# =========================================================================================================

def test_s15_16_realized_vs_requested_reference(grids, adv_items):
    graph = h.ref_graph(grids["main"], _task(adv_items, "bbh_logic4"))
    realized = {"deep/2": (1600, "stop"), "deep/4": (900, "stop"), "deep/8": (8192, "length"), "deep/16": (700, "stop")}
    rows = []
    for name, (tokens, finish) in realized.items():
        spec = graph.nodes[graph.checkpoints[name].candidate_ids[0]]
        rows.append((spec.reasoning_budget, tokens, ref.binding_cap(spec.reasoning_budget, tokens, finish)))
    assert [r[0] for r in rows] == [2048, 4096, 8192, 16384]
    assert [r[1] for r in rows] != sorted(r[1] for r in rows), "fixture: realized reasoning is not monotone"
    assert [r[2] for r in rows] == [False, False, True, False]


def test_s15_16_analysis_reports_realized_reasoning_production():
    optional_module("agent_allocation.analysis")
    optional_module("agent_allocation.reporting.cli")
    pytest.skip("waiting on agent 07 table formats; exercised by test_e2e_mocked (M2)")


# =========================================================================================================
# 15.17 Terminal selectors have real populated rows; no all-null selector plotted as zero accuracy.
# =========================================================================================================

def test_s15_17_all_null_selector_blocks_reference():
    null_rows = [load_record(SelectionResult, r) for r in adversarial.load_jsonl("selector_all_null.jsonl")]
    rows = [{"policy_id": r.policy_id, "status": r.status, "selected_candidate_id": r.selected_candidate_id,
             "correct": None} for r in null_rows]
    with pytest.raises(ValueError):
        ref.delivered_accuracy_reference(rows)
    with pytest.raises(ValueError):
        ref.delivered_accuracy_reference([])
    abstained = [load_record(SelectionResult, r) for r in adversarial.load_jsonl("selector_abstained.jsonl")]
    acc = ref.delivered_accuracy_reference([{"policy_id": r.policy_id, "status": r.status,
                                             "selected_candidate_id": None, "correct": None} for r in abstained])
    assert acc["lower"] == acc["upper"] == 0 and acc["abstained"] == len(abstained)


def test_s15_17_seal_refuses_null_selected_rows_production(run_paths):
    seal = optional_module("agent_allocation.selection.seal")
    null_rows = [load_record(SelectionResult, r) for r in adversarial.load_jsonl("selector_all_null.jsonl")]
    with pytest.raises(Exception):
        seal.seal_selections(run_paths, "main", null_rows)
    with pytest.raises(Exception):
        seal.seal_selections(run_paths, "main", [])


def test_s15_17_selector_never_returns_selected_without_candidate_production(adv_items, cfg):
    readout = optional_module("agent_allocation.selection.readout")
    task = _task(adv_items, "bbh_logic4")
    cp, cands = _math_pool(task, ["A", "B", "C"], cfg)
    invalid = {k: dataclasses.replace(v, final_answer=None, parse_status=ParseStatus.MISSING) for k, v in cands.items()}
    res = readout.select_checkpoint(cp, task, invalid, {}, seed=cfg["seed"])
    assert res.selected_candidate_id is None and res.status == SelectionStatus.ABSTAINED_ALL_INVALID.value
    ok_res = readout.select_checkpoint(cp, task, cands, {}, seed=cfg["seed"])
    assert ok_res.selected_candidate_id in cands and ok_res.status == SelectionStatus.SELECTED.value
    assert ok_res.pool_hash


# =========================================================================================================
# 15.18 Synthesis may exceed input-bank coverage; candidate-only selection may not.
# =========================================================================================================

def test_s15_18_invariant_scope_reference():
    rows = [
        {"policy_id": "main/x/pooled/4/current", "family": "pooled", "delivered_correct": True,
         "pool_states": ["wrong"] * 4},
        {"policy_id": "main/x/peer/4/4/current", "family": "peer", "delivered_correct": True,
         "pool_states": ["wrong"] * 4},
        {"policy_id": "main/x/cp/3/current", "family": "cp", "delivered_correct": True,
         "pool_states": ["unknown", "wrong", "wrong", "wrong"]},
    ]
    assert ref.delivered_vs_coverage_violations(rows) == ["main/x/peer/4/4/current: delivered correct with no correct pool member"]


def test_s15_18_metrics_scope_production():
    optional_module("agent_allocation.metrics")
    pytest.skip("metric table schema not frozen; exercised on the mocked run in test_e2e_mocked (M2)")


# =========================================================================================================
# 15.19 Peer order and root identities do not depend on completion time.
# =========================================================================================================

def test_s15_19_identity_has_no_time_reference(grids, adv_items):
    assert not {f.name for f in dataclasses.fields(GenerationSpec)} & {"created_at", "sent_at", "completed_at",
                                                                         "latency_seconds", "attempt_no"}
    task = _task(adv_items, "mmlu_pro")
    assert set(h.ref_graph(grids["main"], task).nodes) == set(h.ref_graph(grids["main"], task).nodes)


def test_s15_19_graph_and_peer_order_independent_of_completion_order_production(grids, adv_items, cfg):
    grid_mod = optional_module("agent_allocation.protocols.grid")
    render = optional_module("agent_allocation.prompts.render")
    task = _task(adv_items, "math500")
    g1 = h.production_graph(grid_mod, grids["main"], task)
    g2 = h.production_graph(grid_mod, grids["main"], task)
    assert [n.node_id for n in g1.nodes] == [n.node_id for n in g2.nodes]
    by_id = {n.node_id: n for n in g1.nodes}
    cands = {n.node_id: h.valid_candidate(n, task) for n in g1.nodes}
    non_identity = 0
    for node in g1.nodes:
        if node.kind is not GenerationKind.PEER or node.team_size != 5:
            continue
        parents = [(by_id[p], cands[p]) for p in node.parent_ids]
        base = render.render_messages(node, task, parents, seed=cfg["seed"])
        assert render.render_messages(node, task, list(reversed(parents)), seed=cfg["seed"]) == base, \
            "peer packet order follows the order parents were supplied (completion order)"
        text = h.all_text(base)
        order = [m for m in h.MARK_RE.findall(text)]
        peer_order = [m for m in dict.fromkeys(order) if m != h.marker(node.parent_ids[node.root_slot])]
        slot_order = [h.marker(p) for j, p in enumerate(node.parent_ids) if j != node.root_slot]
        non_identity += peer_order != slot_order
    assert non_identity > 0, "no peer packet was ever permuted: blind order is not applied"


# =========================================================================================================
# 15.20 Resume is idempotent for stored successful nodes; no regeneration, no lost raw, no erased failures.
# =========================================================================================================

def test_s15_20_resume_planner_reference(grids, adv_items):
    graph = h.ref_graph(grids["main"], _task(adv_items, "mmlu_pro"))
    order = list(graph.nodes)
    done = set(order[:40])
    ready = [n for n in order if n not in done and set(graph.nodes[n].parent_ids) <= done]
    assert ready and not set(ready) & done
    assert all(set(graph.nodes[n].parent_ids) <= done for n in ready)


def test_s15_20_state_store_recover_keeps_terminal_nodes_and_raw_production(run_paths, adv_items, cfg):
    state_mod = optional_module("agent_allocation.storage.state")
    raw_mod = optional_module("agent_allocation.storage.raw")
    task = _task(adv_items, "math500")
    graph = h.ref_graph(grids_from_config(cfg)["pilot_r1"], task)
    specs = list(graph.nodes.values())
    store = state_mod.StateStore(run_paths)
    store.register_nodes(specs, wave_index=0)
    done = [h.persist_success(store, raw_mod, run_paths, s, task, h.marked_content(s)) for s in specs[:3]]
    store.backup()
    raw_files = sorted(p for p in (run_paths.run_dir / "raw").rglob("*") if p.is_file())
    assert len(raw_files) == 3
    recovered = state_mod.StateStore.recover(run_paths)
    assert {r.node_id for r in done} <= recovered.terminal_node_ids()
    assert not {s.node_id for s in specs[3:]} & recovered.terminal_node_ids()
    assert sorted(p for p in (run_paths.run_dir / "raw").rglob("*") if p.is_file()) == raw_files
