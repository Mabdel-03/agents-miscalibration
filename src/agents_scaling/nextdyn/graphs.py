"""Communication graphs as data: builders, a registry, and a per-round schedule (plan §1.5).

ARD36 spelled every visibility pattern as a branch on a closed ``Protocol`` enum, so the
directed E3a channels could not be expressed without editing the episode loop, and N was
frozen at five in the branch bodies.  Here a condition names a *builder* and its parameters
(``configs/nextdyn/protocols.yaml``); this module turns that name into an ``n x n`` boolean
matrix and nothing else knows how an edge is chosen.  Adding a graph is one function plus one
registry entry (plan §1.10).

Orientation, once, here: ``G[j][i] is True`` means **agent i receives agent j's previous
packet**.  The first index is the sender, the second the recipient; the diagonal is always
``False`` (an agent's own previous draft is the ``own_draft`` knob of the condition, never an
edge).  ``in_degree(g, i)`` therefore counts the packets *i* sees.

Rewiring is output-independent (spec 6.4): a ``per_round`` schedule draws its permutation from
``ids.blind`` over ``(pairing_id, round)``, the same correctness-blind HMAC that orders packets
and pseudonyms, so a ring cannot re-form around who happens to be right.

Spec: 6.4 (synchronous updates, blind order), 10.3 (directed focal channels), 10.4 (random
protection), 11.1 (rings, N in {3,5,9}).  Plan: §1.5, §1.10, D7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Final, Literal, Mapping, Sequence

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids
from agents_scaling.nextdyn.errors import ContractViolation

#: ``G[j][i]``: agent ``i`` receives agent ``j``'s previous packet.  No self edges.
Graph = tuple[tuple[bool, ...], ...]

#: Placeholders a ``protocols.yaml`` builder parameter may carry (plan §1.5).  They are
#: resolved by :func:`resolve_graph` against the episode, never by the builder itself, so a
#: condition stays team-independent and its hash is stable.
FOCAL: Final[str] = "$focal"
PROTECTED: Final[str] = "$protected"
PERM_ONCE: Final[str] = "$perm_once"
PERM_ROUND: Final[str] = "$perm_round"
PLACEHOLDERS: Final[tuple[str, ...]] = (FOCAL, PROTECTED, PERM_ONCE, PERM_ROUND)

#: The ring permutation token of a ``fixed`` schedule: one draw for the whole episode.
ONCE: Final[str] = "once"

SCHEDULE_FIXED: Final[str] = "fixed"
SCHEDULE_PER_ROUND: Final[str] = "per_round"
SCHEDULE_KINDS: Final[tuple[str, ...]] = (SCHEDULE_FIXED, SCHEDULE_PER_ROUND)


def _n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ContractViolation(f"a communication graph needs n >= 2 agents, got {n!r}")
    return n


def _slot(name: str, slot: int, n: int) -> int:
    if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < n:
        raise ContractViolation(f"{name} {slot!r} is outside 0..{n - 1}")
    return slot


def _freeze(rows: Sequence[Sequence[bool]]) -> Graph:
    graph = tuple(tuple(bool(v) for v in row) for row in rows)
    n = len(graph)
    if any(len(row) != n for row in graph):
        raise ContractViolation("a graph must be square")
    if any(graph[i][i] for i in range(n)):
        raise ContractViolation("a graph never carries a self edge (own_draft is a condition "
                                "knob, not an edge)")
    return graph


# --------------------------------------------------------------------------- builders


def complete(n: int) -> Graph:
    """Everyone receives everyone else's previous packet (PEER; spec 6.3)."""
    n = _n(n)
    return _freeze([[i != j for i in range(n)] for j in range(n)])


def empty(n: int) -> Graph:
    """Nobody receives anybody (PRIVATE and FRESH; spec 6.3)."""
    n = _n(n)
    return _freeze([[False] * n for _ in range(n)])


def focal_channels(n: int, focal: int, *, incoming: bool, outgoing: bool) -> Graph:
    """Spec 10.3: cut only the focal agent's channels; every non-focal edge stays.

    ``incoming`` keeps the edges *into* the focal slot, ``outgoing`` the edges *out of* it.
    The four E3a arms are the four combinations, and isolating the focal agent must not
    isolate anybody else -- the non-focal block is always complete
    (``test_isolated_focal_does_not_isolate_other_agents``).
    """
    n = _n(n)
    focal = _slot("focal", focal, n)
    rows = []
    for sender in range(n):
        row = []
        for recipient in range(n):
            if sender == recipient:
                edge = False
            elif recipient == focal:
                edge = bool(incoming)
            elif sender == focal:
                edge = bool(outgoing)
            else:
                edge = True
            row.append(edge)
        rows.append(row)
    return _freeze(rows)


def ring_fixed(n: int, perm: Sequence[int]) -> Graph:
    """A bidirectional ring in the slot order ``perm`` (spec 11.1, E3b).

    ``perm`` is a permutation of ``0..n-1``: consecutive entries (wrapping) exchange packets
    in both directions, so every in- and out-degree is two for ``n >= 3``.  At ``n = 3`` the
    ring *is* the complete graph -- an equality, not a coincidence, and the reason the planner
    treats an N=3 ring as explicit reuse rather than a separate treatment.
    """
    n = _n(n)
    order = tuple(int(p) for p in perm)
    if sorted(order) != list(range(n)):
        raise ContractViolation(f"ring perm {order} is not a permutation of 0..{n - 1}")
    rows = [[False] * n for _ in range(n)]
    for position, sender in enumerate(order):
        nxt = order[(position + 1) % n]
        if nxt == sender:
            continue
        rows[sender][nxt] = True
        rows[nxt][sender] = True
    return _freeze(rows)


def protect_slot(n: int, slot: int) -> Graph:
    """Spec 10.4 natural protection: the protected slot speaks but never listens."""
    return focal_channels(_n(n), slot, incoming=False, outgoing=True)


#: name -> builder.  A new graph is one function above plus one entry here (plan §1.10).
GRAPH_BUILDERS: Final[Mapping[str, Callable[..., Graph]]] = {
    "complete": complete,
    "empty": empty,
    "focal_channels": focal_channels,
    "ring_fixed": ring_fixed,
    "protect_slot": protect_slot,
}


# --------------------------------------------------------------------------- degrees


def edge_count(graph: Graph) -> int:
    """Directed edges: ``sum(G)``.  At N=5, IN1_OUT1/IN0_OUT1/IN1_OUT0/IN0_OUT0 = 20/16/16/12."""
    return sum(1 for row in graph for value in row if value)


def in_degree(graph: Graph, slot: int) -> int:
    """How many packets ``slot`` receives."""
    _slot("slot", slot, len(graph))
    return sum(1 for row in graph if row[slot])


def out_degree(graph: Graph, slot: int) -> int:
    """How many recipients ``slot``'s packet reaches."""
    _slot("slot", slot, len(graph))
    return sum(1 for value in graph[slot] if value)


def senders_for(graph: Graph, recipient: int) -> tuple[int, ...]:
    """The slots whose previous packet ``recipient`` receives, in slot order."""
    _slot("recipient", recipient, len(graph))
    return tuple(j for j in range(len(graph)) if graph[j][recipient])


# --------------------------------------------------------------------------- schedules


@dataclass(frozen=True)
class GraphRef:
    """A builder name plus its literal or placeholder parameters (a ``protocols.yaml`` value)."""

    name: str
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.name not in GRAPH_BUILDERS:
            raise ContractViolation(
                f"unknown graph builder {self.name!r}; declared: {sorted(GRAPH_BUILDERS)}")
        params = {str(key): value for key, value in dict(self.params).items()}
        for key, value in params.items():
            if isinstance(value, str) and value not in PLACEHOLDERS:
                raise ContractViolation(
                    f"graph {self.name!r} parameter {key}={value!r} is a string that is not one "
                    f"of the placeholders {PLACEHOLDERS}")
        object.__setattr__(self, "params", params)

    def identity_object(self) -> dict[str, Any]:
        """The JCS-safe symbolic form that enters a condition hash (labels never do)."""
        return {"name": self.name, "params": dict(self.params)}


@dataclass(frozen=True)
class GraphSchedule:
    """``fixed`` = one graph for the whole episode; ``per_round`` = redrawn each round."""

    kind: Literal["fixed", "per_round"]
    ref: GraphRef

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULE_KINDS:
            raise ContractViolation(f"graph schedule kind {self.kind!r} not in {SCHEDULE_KINDS}")
        uses_round = PERM_ROUND in self.ref.params.values()
        if uses_round and self.kind != SCHEDULE_PER_ROUND:
            raise ContractViolation(
                f"graph {self.ref.name!r} uses {PERM_ROUND} but is scheduled {self.kind!r}: a "
                "per-round permutation needs a per_round schedule")
        if not uses_round and self.kind == SCHEDULE_PER_ROUND:
            raise ContractViolation(
                f"graph {self.ref.name!r} is scheduled per_round but no parameter is "
                f"{PERM_ROUND}, so every round would resolve to the same graph")

    def identity_object(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref.identity_object()}


def canonical_ref(ref: GraphRef) -> GraphRef:
    """The canonical spelling of a builder call whose graph does not depend on its parameters.

    ``focal_channels(focal=anything, incoming=True, outgoing=True)`` is the complete graph for
    every focal slot and every ``n``, so IN1_OUT1 canonicalizes to ``complete`` and therefore
    hashes equal to PEER (spec 10.1 explicit reuse; ``protocols.yaml`` declares the
    equivalence and ``load_registry`` checks it).  Canonicalization is symbolic -- it never
    looks at a team -- so the hash stays team-independent.
    """
    if ref.name == "focal_channels" and ref.params.get("incoming") is True \
            and ref.params.get("outgoing") is True:
        return GraphRef(name="complete", params={})
    return ref


def ring_permutation(study_seed: bytes, pairing_id: str, round_token: Any, n: int) -> tuple[int, ...]:
    """The blind ring order: slots sorted by ``blind(ORDER_RING, pairing_id, round_token, slot)``.

    ``round_token`` is :data:`ONCE` for a fixed ring and the round number for a rewired one, so
    the permutation depends on the episode and the round -- never on any output (spec 6.4).
    """
    n = _n(n)
    return tuple(sorted(
        range(n),
        key=lambda slot: ids.blind(study_seed, K.ORDER_RING, str(pairing_id), str(round_token), int(slot)),
    ))


def protected_slot(study_seed: bytes, pairing_id: str, n: int) -> int:
    """Spec 10.4: the protected slot, drawn once per episode by a correctness-blind key."""
    n = _n(n)
    return min(range(n),
               key=lambda slot: ids.blind(study_seed, K.ORDER_PROTECT, str(pairing_id), int(slot)))


def resolve_graph(schedule: GraphSchedule, *, n: int, round_: int, focal_slot: int | None,
                  study_seed: bytes, pairing_id: str) -> Graph:
    """Build the graph that governs delivery **into** ``round_``.

    Every placeholder is resolved here: ``$focal`` from the root bundle's focal slot,
    ``$protected`` and the ring permutations from :func:`ids.blind`.  A ``fixed`` schedule
    ignores ``round_`` by construction (its permutation token is :data:`ONCE`), which is what
    ``test_focal_slot_is_fixed_after_root`` pins.
    """
    n = _n(n)
    ref = schedule.ref
    resolved: dict[str, Any] = {}
    for key, value in ref.params.items():
        if value == FOCAL:
            if focal_slot is None:
                raise ContractViolation(
                    f"graph {ref.name!r} needs {FOCAL} but the root bundle declares no focal slot")
            resolved[key] = _slot("focal_slot", int(focal_slot), n)
        elif value == PROTECTED:
            resolved[key] = protected_slot(study_seed, pairing_id, n)
        elif value == PERM_ONCE:
            resolved[key] = ring_permutation(study_seed, pairing_id, ONCE, n)
        elif value == PERM_ROUND:
            resolved[key] = ring_permutation(study_seed, pairing_id, int(round_), n)
        else:
            resolved[key] = value
    builder = GRAPH_BUILDERS[ref.name]
    if ref.name == "focal_channels":
        return builder(n, resolved["focal"], incoming=bool(resolved["incoming"]),
                       outgoing=bool(resolved["outgoing"]))
    if ref.name == "protect_slot":
        return builder(n, resolved["slot"])
    if ref.name == "ring_fixed":
        return builder(n, resolved["perm"])
    return builder(n)


__all__ = [
    "FOCAL", "GRAPH_BUILDERS", "Graph", "GraphRef", "GraphSchedule", "ONCE", "PERM_ONCE",
    "PERM_ROUND", "PLACEHOLDERS", "PROTECTED", "SCHEDULE_FIXED", "SCHEDULE_KINDS",
    "SCHEDULE_PER_ROUND", "canonical_ref", "complete", "edge_count", "empty", "focal_channels",
    "in_degree", "out_degree", "protect_slot", "protected_slot", "resolve_graph", "ring_fixed",
    "ring_permutation", "senders_for",
]
