"""Per-item protocol DAGs: generation nodes plus CURRENT/ARCHIVE policy checkpoints (agent 03).

One call of :func:`build_item_graph` turns a :class:`GridSpec` into every generation node an item needs and every
policy checkpoint that reads them. A generation node is one fresh API sampling opportunity; a checkpoint is a
declared candidate pool plus the ancestors a deployed policy would have had to generate (MASTER_SPEC §5, §7.2).

Node coordinates are exactly the table in ``GridSpec``'s docstring (pinned by
``tests/unit/test_node_identity_golden.py``)::

    ROOT    root_slot=j  team_size=0 update_round=0 parents=()
    PRIVATE root_slot=j  team_size=0 update_round=r parents=(previous candidate of slot j,)
    PEER    root_slot=j  team_size=n update_round=r parents=all n round r-1 nodes in slot order
    DEEP    root_slot=0  team_size=0 update_round=0 reasoning_budget=A*m parents=()
    CP      root_slot=j  team_size=4 update_round=h parents=(peer n=4 r=1 slot j, or previous CP of slot j,)
    POOLED  root_slot=0  team_size=4 update_round=0 parents=(root 0..3 in slot order)

Reuse is explicit and structural, never by prompt equality: PRIVATE_SINGLE's chain *is* slot 0 of every
PRIVATE_TEAM; ``sample/k``, ``private_team/n/0`` and ``peer/n/0`` all point at the same roots; PEER nodes carry
``team_size=n`` and their own parents, so two team sizes never alias. ``draw_id`` is always 0 here (deliberate
repeats are a canary-only concern of the lead).

Checkpoints (names as in ``tools/reference_dag.py``, plus ``deep/1`` = root[0], the A-level point of the deep curve,
emitted whenever the grid has a deep curve), each in two views:

* CURRENT: ``candidate_ids`` is the declared pool the primary selector reads.
* ARCHIVE: ``candidate_ids`` is every candidate the policy itself generated (its ancestors, topological order).

``required_node_ids`` is the deduplicated ancestor set in both views (topological order). Checkpoint metadata:

    family          team_size  total_cycles  peer_rounds  reasoning_budget   len(required_node_ids)
    sample/k        k          1             0            A                  k
    private_single/t 1         t             0            A                  t
    private_team/n/r n         1+r           0            A                  n(1+r)
    peer/n/r        n          1+r           r            A                  n(1+r)
    deep/m          1          1             0            A*m                1
    cp/h            4          2+h           1            A                  4(2+h)
    pooled/4        4          2             0            A                  5

``team_size`` is the pool width, ``total_cycles`` the generations on the deepest path (1 + rounds), and
``peer_rounds`` the synchronous communication rounds on that path.

Config keys consumed (R9): the grid itself comes from ``config.grids_from_config``. :func:`check_config_constants`
reads ``protocols.cp_team_size``, ``protocols.cp_peer_rounds``, ``protocols.cp_private_rounds``,
``protocols.pooled_synthesis_team_size``, ``protocols.communication_mode`` and ``protocols.context_policy``;
:func:`check_budgets` takes ``reasoning.requested_reasoning_max`` and the endpoint lock's
``max_completion_tokens`` from the caller. Selector names mirror ``benchmarks[].selector``.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from agent_allocation.config import CODE_BENCHMARKS
from agent_allocation.contracts import (
    CheckpointView,
    GenerationKind,
    GenerationSpec,
    GridSpec,
    ItemGraph,
    PolicyCheckpoint,
    PublicTask,
    public_task_hash,
)

# Extension constants. GridSpec carries only ``extension: bool``; check_config_constants() pins these to the config.
CP_TEAM_SIZE = 4
CP_PEER_ROUND = 1
CP_PRIVATE_STEPS = 3
POOLED_TEAM_SIZE = 4

# Declared selectors (config ``benchmarks[].selector``); agent 06 owns their implementation and version strings.
TEXT_SELECTOR = "plurality"
CODE_SELECTOR = "public_tests_only"
SELECTOR_VERSION = "v1"

FAMILIES: tuple[str, ...] = ("sample", "private_single", "private_team", "peer", "deep", "cp", "pooled")


class GridError(ValueError):
    """A grid, budget or graph that cannot be built as declared. Never silently repaired."""


class BudgetUnsupported(GridError):
    """A requested reasoning budget or completion cap the endpoint/config does not allow (never clipped)."""


def validate_grid(grid: GridSpec) -> None:
    """Raise :class:`GridError` unless the grid is internally consistent."""
    problems: list[str] = []
    if grid.bank_size < 1:
        problems.append("bank_size must be >= 1")
    if grid.private_single_cycles < 1:
        problems.append("private_single_cycles must be >= 1 (cycle 1 is root[0])")
    sizes = tuple(grid.team_sizes)
    if not sizes:
        problems.append("team_sizes must be non-empty")
    if any(n < 2 for n in sizes):
        problems.append("team sizes must be >= 2 (a team of one is PRIVATE_SINGLE)")
    if len(set(sizes)) != len(sizes):
        problems.append("team sizes must be distinct")
    if sizes and max(sizes) > grid.bank_size:
        problems.append("the largest team needs more roots than the bank holds")
    if grid.private_team_rounds < 0 or grid.peer_rounds < 0:
        problems.append("round counts must be >= 0")
    if grid.private_single_cycles - 1 < grid.private_team_rounds:
        problems.append("PRIVATE_TEAM slot 0 reuses PRIVATE_SINGLE: private_single_cycles-1 >= private_team_rounds")
    multipliers = tuple(grid.deep_multipliers)
    if any(m < 2 for m in multipliers):
        problems.append("deep multipliers must be >= 2 (m=1 is root[0], checkpoint deep/1)")
    if len(set(multipliers)) != len(multipliers):
        problems.append("deep multipliers must be distinct")
    if grid.extension:
        if CP_TEAM_SIZE not in sizes:
            problems.append(f"extension needs team size {CP_TEAM_SIZE} for CP")
        if grid.peer_rounds < CP_PEER_ROUND:
            problems.append(f"extension needs PEER round {CP_PEER_ROUND}")
        if grid.bank_size < POOLED_TEAM_SIZE:
            problems.append(f"extension needs {POOLED_TEAM_SIZE} roots for POOLED")
    if problems:
        raise GridError(f"grid {grid.name!r}: " + "; ".join(problems))


def expected_node_count(grid: GridSpec) -> int:
    """Closed-form node count (the lead's arithmetic in tests/unit/test_contracts.py)."""
    extension = CP_TEAM_SIZE * CP_PRIVATE_STEPS + 1 if grid.extension else 0
    return (grid.bank_size + (grid.private_single_cycles - 1) + (max(grid.team_sizes) - 1) * grid.private_team_rounds
            + sum(grid.team_sizes) * grid.peer_rounds + len(grid.deep_multipliers) + extension)


def selector_for(task: PublicTask) -> str:
    return CODE_SELECTOR if task.benchmark in CODE_BENCHMARKS else TEXT_SELECTOR


def build_item_graph(grid: GridSpec, *, run_id: str, partition_id: str, task: PublicTask, model: str,
                     base_budget: int, headroom: int, endpoint_lock_hash: str, decoding_config_hash: str,
                     prompt_version: str) -> ItemGraph:
    """Every generation node (topological order) and every CURRENT/ARCHIVE checkpoint of one item."""
    validate_grid(grid)
    if base_budget <= 0 or headroom <= 0:
        raise GridError("base_budget and headroom must be positive")
    for label, value in (("run_id", run_id), ("partition_id", partition_id), ("model", model),
                         ("endpoint_lock_hash", endpoint_lock_hash), ("decoding_config_hash", decoding_config_hash),
                         ("prompt_version", prompt_version), ("item_id", task.item_id)):
        if not value:
            raise GridError(f"{label} must be non-empty")

    common: dict[str, Any] = dict(
        run_id=run_id, partition_id=partition_id, item_id=task.item_id, benchmark=task.benchmark, model=model,
        endpoint_lock_hash=endpoint_lock_hash, input_manifest_hash=public_task_hash(task),
        decoding_config_hash=decoding_config_hash, prompt_version=prompt_version, completion_headroom=headroom,
        draw_id=0,
    )
    nodes: list[GenerationSpec] = []
    by_id: dict[str, GenerationSpec] = {}

    def add(kind: GenerationKind, *, root_slot: int = 0, team_size: int = 0, update_round: int = 0,
            parents: Sequence[GenerationSpec] = (), budget: int = base_budget) -> GenerationSpec:
        node = GenerationSpec(kind=kind, reasoning_budget=budget, root_slot=root_slot, team_size=team_size,
                              update_round=update_round, parent_ids=tuple(p.node_id for p in parents), **common)
        if node.node_id in by_id:
            raise GridError(f"duplicate generation node {kind.value} slot={root_slot} n={team_size} r={update_round}")
        missing = [p for p in node.parent_ids if p not in by_id]
        if missing:
            raise GridError(f"parent emitted after child: {missing}")
        by_id[node.node_id] = node
        nodes.append(node)
        return node

    # SAMPLE bank.
    roots = [add(GenerationKind.ROOT, root_slot=j) for j in range(grid.bank_size)]

    # PRIVATE chains: slot 0 through private_single_cycles; slots 1..max(n)-1 through private_team_rounds.
    private: dict[tuple[int, int], GenerationSpec] = {(j, 0): roots[j] for j in range(max(grid.team_sizes))}
    for r in range(1, grid.private_single_cycles):
        private[0, r] = add(GenerationKind.PRIVATE, root_slot=0, update_round=r, parents=(private[0, r - 1],))
    for j in range(1, max(grid.team_sizes)):
        for r in range(1, grid.private_team_rounds + 1):
            private[j, r] = add(GenerationKind.PRIVATE, root_slot=j, update_round=r, parents=(private[j, r - 1],))

    # PEER: synchronous rounds, every receiver reads the whole previous round in slot order.
    peer: dict[tuple[int, int], tuple[GenerationSpec, ...]] = {}
    for n in grid.team_sizes:
        peer[n, 0] = tuple(roots[:n])
        for r in range(1, grid.peer_rounds + 1):
            previous = peer[n, r - 1]
            peer[n, r] = tuple(add(GenerationKind.PEER, root_slot=j, team_size=n, update_round=r, parents=previous)
                               for j in range(n))

    # DEEP_SINGLE: independent longer calls; the A-level point is root[0].
    deep = {m: add(GenerationKind.DEEP, budget=base_budget * m) for m in grid.deep_multipliers}

    # Extension: CP continues PEER n=4 r=1 privately; POOLED synthesizes roots 0..3.
    cp: dict[int, tuple[GenerationSpec, ...]] = {}
    pooled: GenerationSpec | None = None
    if grid.extension:
        cp[0] = peer[CP_TEAM_SIZE, CP_PEER_ROUND]
        for h in range(1, CP_PRIVATE_STEPS + 1):
            cp[h] = tuple(add(GenerationKind.CP, root_slot=j, team_size=CP_TEAM_SIZE, update_round=h,
                              parents=(cp[h - 1][j],)) for j in range(CP_TEAM_SIZE))
        pooled = add(GenerationKind.POOLED, team_size=POOLED_TEAM_SIZE, parents=tuple(roots[:POOLED_TEAM_SIZE]))

    if len(nodes) != expected_node_count(grid):
        raise GridError(f"grid {grid.name!r}: built {len(nodes)} nodes, expected {expected_node_count(grid)}")

    position = {node.node_id: i for i, node in enumerate(nodes)}
    selector = selector_for(task)
    checkpoints: list[PolicyCheckpoint] = []

    def ancestors_of(pool: Sequence[GenerationSpec]) -> tuple[str, ...]:
        found: set[str] = set()
        todo = [p.node_id for p in pool]
        while todo:
            key = todo.pop()
            if key not in found:
                found.add(key)
                todo.extend(by_id[key].parent_ids)
        return tuple(sorted(found, key=position.__getitem__))

    def checkpoint(name: str, family: str, pool: Sequence[GenerationSpec], *, team_size: int, total_cycles: int,
                   peer_rounds: int, budget: int = base_budget) -> None:
        required = ancestors_of(pool)
        current = tuple(p.node_id for p in pool)
        if len(set(current)) != len(current):
            raise GridError(f"checkpoint {name}: duplicate candidate in pool")
        for view, candidates in ((CheckpointView.CURRENT, current), (CheckpointView.ARCHIVE, required)):
            checkpoints.append(PolicyCheckpoint(
                policy_id=f"{partition_id}/{task.item_id}/{name}/{view.value}", item_id=task.item_id,
                family=family, team_size=team_size, total_cycles=total_cycles, peer_rounds=peer_rounds,
                candidate_ids=candidates, required_node_ids=required, selector_name=selector,
                selector_version=SELECTOR_VERSION, partition_id=partition_id, name=name, view=view,
                reasoning_budget=budget))

    for k in range(1, grid.bank_size + 1):
        checkpoint(f"sample/{k}", "sample", roots[:k], team_size=k, total_cycles=1, peer_rounds=0)
    for t in range(1, grid.private_single_cycles + 1):
        checkpoint(f"private_single/{t}", "private_single", (private[0, t - 1],), team_size=1, total_cycles=t,
                   peer_rounds=0)
    for n in grid.team_sizes:
        for r in range(grid.private_team_rounds + 1):
            checkpoint(f"private_team/{n}/{r}", "private_team", tuple(private[j, r] for j in range(n)),
                       team_size=n, total_cycles=1 + r, peer_rounds=0)
    for n in grid.team_sizes:
        for r in range(grid.peer_rounds + 1):
            checkpoint(f"peer/{n}/{r}", "peer", peer[n, r], team_size=n, total_cycles=1 + r, peer_rounds=r)
    if grid.deep_multipliers:
        checkpoint("deep/1", "deep", (roots[0],), team_size=1, total_cycles=1, peer_rounds=0)
        for m in grid.deep_multipliers:
            checkpoint(f"deep/{m}", "deep", (deep[m],), team_size=1, total_cycles=1, peer_rounds=0,
                       budget=base_budget * m)
    if grid.extension:
        assert pooled is not None
        for h in range(CP_PRIVATE_STEPS + 1):
            checkpoint(f"cp/{h}", "cp", cp[h], team_size=CP_TEAM_SIZE, total_cycles=1 + CP_PEER_ROUND + h,
                       peer_rounds=CP_PEER_ROUND)
        checkpoint(f"pooled/{POOLED_TEAM_SIZE}", "pooled", (pooled,), team_size=POOLED_TEAM_SIZE, total_cycles=2,
                   peer_rounds=0)

    return ItemGraph(item_id=task.item_id, partition_id=partition_id, grid=grid, nodes=tuple(nodes),
                     checkpoints=tuple(checkpoints))


def check_budgets(graph: ItemGraph, *, max_completion_tokens: int, requested_reasoning_max: int) -> None:
    """Refuse (never clip) any node whose requested allowance or completion cap the endpoint/config cannot serve.

    ``max_completion_tokens`` is ``EndpointLock.max_completion_tokens``; ``requested_reasoning_max`` is
    ``reasoning.requested_reasoning_max`` from the config.
    """
    bad = [(n.kind.value, n.reasoning_budget, n.max_completion_tokens) for n in graph.nodes
           if n.reasoning_budget > requested_reasoning_max or n.max_completion_tokens > max_completion_tokens]
    if bad:
        raise BudgetUnsupported(f"{graph.item_id}: unsupported budgets (kind, reasoning, max_tokens): {sorted(set(bad))}")


def check_config_constants(cfg: Mapping[str, Any]) -> None:
    """The extension shape and prompt context policy this module hard-codes must equal the config's."""
    p = cfg["protocols"]
    expected = {
        "cp_team_size": CP_TEAM_SIZE,
        "cp_peer_rounds": CP_PEER_ROUND,
        "cp_private_rounds": CP_PRIVATE_STEPS,
        "pooled_synthesis_team_size": POOLED_TEAM_SIZE,
        "communication_mode": "synchronous",
        "context_policy": "latest_visible_solution",
    }
    drift = {key: (p.get(key), value) for key, value in expected.items() if p.get(key) != value}
    if drift:
        raise GridError(f"config protocols differ from the protocol builder (config, builder): {drift}")


def checkpoint_by_name(graph: ItemGraph, name: str, view: CheckpointView = CheckpointView.CURRENT) -> PolicyCheckpoint:
    for cp in graph.checkpoints:
        if cp.name == name and cp.view is view:
            return cp
    raise KeyError(f"{graph.item_id}: no checkpoint {name!r} ({view.value})")


def node_index(graph: ItemGraph) -> dict[str, GenerationSpec]:
    return {node.node_id: node for node in graph.nodes}
