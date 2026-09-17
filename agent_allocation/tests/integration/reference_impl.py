"""Independent reference implementations for agent 08's acceptance tests.

Everything here is written from ``MASTER_SPEC.md`` and the frozen contract (``docs/00_contract.md`` and the
``GridSpec`` coordinate table in ``contracts.py``), never from another agent's module. The point is independent
recomputation: when a production module and this file disagree, one of them is wrong and the run pauses.

Contents
  * ``reference_graph``      -- from-scratch GenerationSpec DAG + checkpoints for any ``GridSpec``
  * ``ancestors``            -- transitive ancestor sets from ``parent_ids``
  * ``pass_at_k``            -- exact finite-bank estimator with ``fractions.Fraction``
  * ``transition_counts``    -- tri-state W->C / C->W counter that refuses to invent unknowns
  * ``LedgerChecker``        -- replays ``LedgerEvent`` streams and reports every invariant violation
  * ``complete_link_classes``-- gold-blind MATH classes with mutual (two-way) equivalence
  * ``delivered_vs_coverage_violations`` -- the section 12/15.18 invariant, scoped to candidate-only families
  * ``scan_tree_for_canaries`` -- byte scan of a run directory (gz-aware) for planted secrets
"""
from __future__ import annotations

import gzip
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent_allocation.contracts import (
    CheckpointView,
    GenerationKind,
    GenerationSpec,
    GradeResult,
    GradeStatus,
    GridSpec,
    LedgerEvent,
    LedgerEventKind,
    PublicTask,
    public_task_hash,
)

CANDIDATE_ONLY_FAMILIES = frozenset({"sample", "private_single", "private_team", "peer", "deep", "cp"})
SYNTHESIS_FAMILIES = frozenset({"pooled"})


# ---------------------------------------------------------------------------------------------------------
# From-scratch protocol DAG (GridSpec docstring coordinates)
# ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RefCheckpoint:
    name: str                       # sample/k, private_single/t, private_team/n/r, peer/n/r, deep/m, cp/h, pooled/4
    family: str
    candidate_ids: tuple[str, ...]
    team_size: int
    peer_rounds: int
    total_cycles: int


@dataclass
class RefGraph:
    nodes: dict[str, GenerationSpec]                      # node_id -> spec, insertion = topological order
    named: dict[str, str]                                 # readable name ("root/3", "peer/4/1/2") -> node_id
    checkpoints: dict[str, RefCheckpoint]

    def ancestors_of(self, checkpoint: str) -> set[str]:
        return ancestors(self.nodes, self.checkpoints[checkpoint].candidate_ids)

    def kind_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for spec in self.nodes.values():
            out[spec.kind.value] = out.get(spec.kind.value, 0) + 1
        return out


def ancestors(nodes: Mapping[str, GenerationSpec], start: Iterable[str]) -> set[str]:
    found: set[str] = set()
    todo = list(start)
    while todo:
        key = todo.pop()
        if key in found:
            continue
        if key not in nodes:
            raise KeyError(f"unknown node {key}")
        found.add(key)
        todo.extend(nodes[key].parent_ids)
    return found


def reference_graph(grid: GridSpec, *, run_id: str, partition_id: str, task: PublicTask, model: str,
                    base_budget: int, headroom: int, endpoint_lock_hash: str, decoding_config_hash: str,
                    prompt_version: str) -> RefGraph:
    """Build every generation node and checkpoint the spec requires for one item, from the coordinate table."""
    common = dict(run_id=run_id, partition_id=partition_id, item_id=task.item_id, benchmark=task.benchmark,
                  model=model, endpoint_lock_hash=endpoint_lock_hash, input_manifest_hash=public_task_hash(task),
                  decoding_config_hash=decoding_config_hash, prompt_version=prompt_version,
                  completion_headroom=headroom, draw_id=0)
    nodes: dict[str, GenerationSpec] = {}
    named: dict[str, str] = {}
    cps: dict[str, RefCheckpoint] = {}

    def add(name: str, kind: GenerationKind, *, slot: int = 0, team: int = 0, rnd: int = 0,
            budget: int = base_budget, parents: Sequence[str] = ()) -> str:
        spec = GenerationSpec(kind=kind, reasoning_budget=budget, root_slot=slot, team_size=team,
                              update_round=rnd, parent_ids=tuple(parents), **common)
        if spec.node_id in nodes:
            raise ValueError(f"duplicate generation node {name}")
        if name in named:
            raise ValueError(f"duplicate readable name {name}")
        nodes[spec.node_id] = spec
        named[name] = spec.node_id
        return spec.node_id

    def cp(name: str, family: str, ids: Sequence[str], *, team: int, rounds: int, cycles: int) -> None:
        cps[name] = RefCheckpoint(name, family, tuple(ids), team, rounds, cycles)

    roots = [add(f"root/{j}", GenerationKind.ROOT, slot=j) for j in range(grid.bank_size)]
    for k in range(1, grid.bank_size + 1):
        cp(f"sample/{k}", "sample", roots[:k], team=k, rounds=0, cycles=k)

    latest: dict[tuple[int, int], str] = {}
    max_team = max(grid.team_sizes) if grid.team_sizes else 1
    for j in range(max_team):
        latest[j, 0] = roots[j]
    cp("private_single/1", "private_single", (roots[0],), team=1, rounds=0, cycles=1)
    for r in range(1, grid.private_single_cycles):
        latest[0, r] = add(f"private/0/{r}", GenerationKind.PRIVATE, slot=0, rnd=r, parents=(latest[0, r - 1],))
        cp(f"private_single/{r + 1}", "private_single", (latest[0, r],), team=1, rounds=0, cycles=r + 1)
    for j in range(1, max_team):
        for r in range(1, grid.private_team_rounds + 1):
            latest[j, r] = add(f"private/{j}/{r}", GenerationKind.PRIVATE, slot=j, rnd=r,
                               parents=(latest[j, r - 1],))
    for n in grid.team_sizes:
        for r in range(grid.private_team_rounds + 1):
            cp(f"private_team/{n}/{r}", "private_team", [latest[j, r] for j in range(n)], team=n, rounds=0,
               cycles=n * (1 + r))
    for n in grid.team_sizes:
        prev = roots[:n]
        cp(f"peer/{n}/0", "peer", prev, team=n, rounds=0, cycles=n)
        for r in range(1, grid.peer_rounds + 1):
            cur = [add(f"peer/{n}/{r}/{j}", GenerationKind.PEER, slot=j, team=n, rnd=r, parents=prev)
                   for j in range(n)]
            cp(f"peer/{n}/{r}", "peer", cur, team=n, rounds=r, cycles=n * (1 + r))
            prev = cur
    if grid.deep_multipliers:
        cp("deep/1", "deep", (roots[0],), team=1, rounds=0, cycles=1)     # spec 5.5: the A-level is root[0]
    for m in grid.deep_multipliers:
        node = add(f"deep/{m}", GenerationKind.DEEP, budget=base_budget * m)
        cp(f"deep/{m}", "deep", (node,), team=1, rounds=0, cycles=1)
    if grid.extension:
        prev = list(cps["peer/4/1"].candidate_ids)
        cp("cp/0", "cp", prev, team=4, rounds=1, cycles=8)
        for h in range(1, 4):
            cur = [add(f"cp/{h}/{j}", GenerationKind.CP, slot=j, team=4, rnd=h, parents=(prev[j],)) for j in range(4)]
            cp(f"cp/{h}", "cp", cur, team=4, rounds=1, cycles=4 * (2 + h))
            prev = cur
        pooled = add("pooled/4", GenerationKind.POOLED, team=4, parents=roots[:4])
        cp("pooled/4", "pooled", (pooled,), team=4, rounds=0, cycles=5)
    return RefGraph(nodes, named, cps)


def expected_node_count(grid: GridSpec) -> int:
    """Closed-form count, derived independently of both builders."""
    team_slots = max(grid.team_sizes) - 1 if grid.team_sizes else 0
    return (grid.bank_size + (grid.private_single_cycles - 1) + team_slots * grid.private_team_rounds
            + sum(grid.team_sizes) * grid.peer_rounds + len(grid.deep_multipliers) + (13 if grid.extension else 0))


def policy_id(partition_id: str, item_id: str, name: str, view: CheckpointView | str = CheckpointView.CURRENT) -> str:
    value = view.value if isinstance(view, CheckpointView) else view
    return f"{partition_id}/{item_id}/{name}/{value}"


# ---------------------------------------------------------------------------------------------------------
# pass@k, coverage, transitions
# ---------------------------------------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> Fraction:
    """Exact ``1 - C(n-c, k) / C(n, k)``. Raises for impossible arguments instead of guessing."""
    if not (isinstance(n, int) and isinstance(c, int) and isinstance(k, int)):
        raise TypeError("n, c, k must be integers")
    if n <= 0 or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"invalid pass@k arguments n={n} c={c} k={k}")
    return 1 - Fraction(comb(n - c, k), comb(n, k))


def tri_state(grade: GradeResult | None) -> str:
    """Candidate state for transitions: correct | wrong | invalid | unknown. Never coerces unknown to wrong."""
    if grade is None:
        return "unknown"
    if grade.status is GradeStatus.INVALID_OUTPUT:
        return "invalid"
    if grade.status is not GradeStatus.GRADED or grade.semantic_correct is None:
        return "unknown"
    return "correct" if grade.semantic_correct else "wrong"


@dataclass
class Transitions:
    w_to_c: int = 0
    c_to_w: int = 0
    c_to_c: int = 0
    w_to_w: int = 0
    invalid_involved: int = 0
    unknown_involved: int = 0

    @property
    def fully_known(self) -> bool:
        return self.unknown_involved == 0


def transition_counts(prev_states: Sequence[str], next_states: Sequence[str]) -> Transitions:
    """Per-agent transitions between aligned rounds. Invalid is an operational state, not 'wrong'."""
    if len(prev_states) != len(next_states):
        raise ValueError("rounds are not aligned")
    t = Transitions()
    for a, b in zip(prev_states, next_states):
        if "unknown" in (a, b):
            t.unknown_involved += 1
        elif "invalid" in (a, b):
            t.invalid_involved += 1
        elif (a, b) == ("wrong", "correct"):
            t.w_to_c += 1
        elif (a, b) == ("correct", "wrong"):
            t.c_to_w += 1
        elif (a, b) == ("correct", "correct"):
            t.c_to_c += 1
        else:
            t.w_to_w += 1
    return t


def correct_count_bounds(states: Sequence[str]) -> tuple[int, int]:
    """Fixed-denominator lower/upper bounds on correct candidates: unknown counts 0 below and 1 above."""
    lower = sum(s == "correct" for s in states)
    return lower, lower + sum(s == "unknown" for s in states)


def delivered_vs_coverage_violations(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """``rows``: {policy_id, family, delivered_correct: bool|None, pool_states: [state,...]}.

    Candidate-only readouts cannot be correct unless some pool member is correct. POOLED synthesis is exempt:
    it may produce a new correct answer. Unknown values never count as a violation or as a pass.
    """
    problems: list[str] = []
    for row in rows:
        family = row["family"]
        if family in SYNTHESIS_FAMILIES:
            continue
        if family not in CANDIDATE_ONLY_FAMILIES:
            problems.append(f"{row['policy_id']}: unknown family {family!r}")
            continue
        if row["delivered_correct"] is True:
            states = row["pool_states"]
            if "correct" not in states and "unknown" not in states:
                problems.append(f"{row['policy_id']}: delivered correct with no correct pool member")
    return problems


# ---------------------------------------------------------------------------------------------------------
# Gold-blind MATH classes: mutual equivalence, complete-link admission
# ---------------------------------------------------------------------------------------------------------

def complete_link_classes(answers: Sequence[str], equivalent: Callable[[str, str], bool | None] | None) -> list[list[int]]:
    """Deterministic classes in candidate order. A candidate joins the first class whose every member is
    exactly equal to it or mutually equivalent in *both* directions; ``None``/timeout means no merge."""
    classes: list[list[int]] = []
    for i, ans in enumerate(answers):
        for members in classes:
            def same(j: int) -> bool:
                other = answers[j]
                if ans == other:
                    return True
                if equivalent is None:
                    return False
                return equivalent(ans, other) is True and equivalent(other, ans) is True
            if all(same(j) for j in members):
                members.append(i)
                break
        else:
            classes.append([i])
    return classes


# ---------------------------------------------------------------------------------------------------------
# Collection-ledger invariant checker
# ---------------------------------------------------------------------------------------------------------

@dataclass
class _AttemptLedger:
    reserved: Decimal
    sent: bool = False
    terminal: str | None = None
    settled: Decimal = Decimal(0)


@dataclass
class LedgerReplay:
    settled: Decimal = Decimal(0)
    open_unsent: Decimal = Decimal(0)      # reserved, no SENT: provably still local
    open_sent: Decimal = Decimal(0)        # reserved and SENT, not terminal: unknown money
    attempts: dict[str, _AttemptLedger] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    peak_exposure: Decimal = Decimal(0)

    @property
    def exposure(self) -> Decimal:
        return self.settled + self.open_unsent + self.open_sent


class LedgerChecker:
    """Replays ``LedgerEvent``s and reports violations of spec section 7.1 and 10.5.

    * every event has a strictly increasing ``seq`` and a nonnegative Decimal amount
    * RESERVE is the first event of an attempt, exactly once, with ``attempt_id`` set
    * settled + open reservations never exceed the ceiling at the moment of a RESERVE
    * RELEASE_UNSENT is forbidden after SENT (unknown money is never released)
    * exactly one terminal event per attempt (SETTLE | RELEASE_UNSENT | CHARGE_MAX); nothing after it
    * CHARGE_MAX settles the full reservation (not less)
    """

    def __init__(self, ceiling: Decimal, *, connect_failed: Iterable[str] | None = None) -> None:
        self.ceiling = Decimal(ceiling)
        # Attempts with independent evidence (wire log / attempt records) of a provable connect failure: the only
        # case in which a SENT reservation may be released (contract: CONNECT_FAILED "provably never sent").
        self.connect_failed = set(connect_failed or ())

    def replay(self, events: Iterable[LedgerEvent]) -> LedgerReplay:
        state = LedgerReplay()
        last_seq: int | None = None
        for ev in events:
            where = f"seq={ev.seq} {ev.kind.value} {ev.attempt_id}"
            if last_seq is not None and ev.seq <= last_seq:
                state.violations.append(f"{where}: seq not strictly increasing")
            last_seq = ev.seq
            if not isinstance(ev.amount_usd, Decimal) or ev.amount_usd < 0:
                state.violations.append(f"{where}: amount must be a nonnegative Decimal")
            if ev.kind is LedgerEventKind.RECONCILE:
                continue
            if ev.attempt_id is None:
                state.violations.append(f"{where}: missing attempt_id")
                continue
            att = state.attempts.get(ev.attempt_id)
            if ev.kind is LedgerEventKind.RESERVE:
                if att is not None:
                    state.violations.append(f"{where}: second RESERVE for one attempt")
                    continue
                if state.exposure + ev.amount_usd > self.ceiling:
                    state.violations.append(f"{where}: reservation exceeds ceiling "
                                            f"({state.exposure} + {ev.amount_usd} > {self.ceiling})")
                state.attempts[ev.attempt_id] = _AttemptLedger(ev.amount_usd)
                state.open_unsent += ev.amount_usd
                state.peak_exposure = max(state.peak_exposure, state.exposure)
                continue
            if att is None:
                state.violations.append(f"{where}: event before RESERVE")
                continue
            if att.terminal is not None:
                state.violations.append(f"{where}: event after terminal {att.terminal}")
                continue
            if ev.kind is LedgerEventKind.SENT:
                if att.sent:
                    state.violations.append(f"{where}: duplicate SENT")
                    continue
                att.sent = True
                state.open_unsent -= att.reserved
                state.open_sent += att.reserved
                continue
            bucket = "open_sent" if att.sent else "open_unsent"
            setattr(state, bucket, getattr(state, bucket) - att.reserved)
            if ev.kind is LedgerEventKind.RELEASE_UNSENT:
                if att.sent and ev.attempt_id not in self.connect_failed:
                    state.violations.append(f"{where}: RELEASE_UNSENT after SENT releases unknown money")
                    setattr(state, bucket, getattr(state, bucket) + att.reserved)   # keep it counted
                    continue
                att.terminal = "release_unsent"
            elif ev.kind is LedgerEventKind.SETTLE:
                att.terminal = "settle"
                att.settled = ev.amount_usd
                state.settled += ev.amount_usd
            elif ev.kind is LedgerEventKind.CHARGE_MAX:
                if ev.amount_usd < att.reserved:
                    state.violations.append(f"{where}: CHARGE_MAX below the reservation {att.reserved}")
                att.terminal = "charge_max"
                att.settled = max(ev.amount_usd, att.reserved)
                state.settled += att.settled
        return state


# ---------------------------------------------------------------------------------------------------------
# Canary scanning
# ---------------------------------------------------------------------------------------------------------

def _file_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix == ".gz":
        try:
            return gzip.decompress(data)
        except OSError:
            return data
    return data


def _sqlite_dump(path: Path) -> bytes:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return "\n".join(con.iterdump()).encode("utf-8", errors="replace")
        finally:
            con.close()
    except sqlite3.Error:
        return path.read_bytes()


def scan_tree_for_canaries(root: Path, canaries: Mapping[str, str], *,
                           exclude_dirs: Iterable[Path] = ()) -> list[tuple[str, str]]:
    """Return ``(relative path, canary name)`` for every file under ``root`` containing a canary value.

    Gzip members are decompressed; SQLite databases are dumped so WAL-committed rows are seen too.
    """
    excluded = [Path(p).resolve() for p in exclude_dirs]
    hits: list[tuple[str, str]] = []
    if not root.exists():
        return hits
    needles = {name: value.encode("utf-8") for name, value in canaries.items()}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        resolved = path.resolve()
        if any(resolved.is_relative_to(ex) for ex in excluded):
            continue
        data = _sqlite_dump(path) if path.suffix in {".sqlite", ".db"} else _file_bytes(path)
        for name, needle in needles.items():
            if needle in data:
                hits.append((str(path.relative_to(root)), name))
    return hits


def text_contains_canary(text: str, canaries: Mapping[str, str]) -> list[str]:
    return [name for name, value in canaries.items() if value in text]


# ---------------------------------------------------------------------------------------------------------
# Collection vs policy accounting (spec 7.1 / 7.2, acceptance 15.7)
# ---------------------------------------------------------------------------------------------------------

def _sum_or_none(values: Sequence[Any]) -> Any:
    if any(v is None for v in values):
        return None
    return sum(values[1:], values[0]) if values else 0


def policy_cost_reference(graph: RefGraph, usage_by_node: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per checkpoint: each ancestor charged exactly once; reasoning is a subset of completion (never added);
    any missing field makes the total unknown (None) instead of silently smaller."""
    out: dict[str, dict[str, Any]] = {}
    for name in graph.checkpoints:
        anc = sorted(graph.ancestors_of(name))
        us = [usage_by_node[n] for n in anc]
        out[name] = {
            "nominal_generations": len(anc),
            "requested_reasoning": sum(graph.nodes[n].reasoning_budget for n in anc),
            "cost_usd": _sum_or_none([u.actual_cost_usd for u in us]),
            "prompt_tokens": _sum_or_none([u.prompt_tokens for u in us]),
            "completion_tokens": _sum_or_none([u.completion_tokens for u in us]),
            "reasoning_tokens": _sum_or_none([u.reasoning_tokens for u in us]),
        }
    return out


def collection_cost_reference(usage_by_node: Mapping[str, Any]) -> Decimal:
    return sum((u.actual_cost_usd for u in usage_by_node.values()), Decimal(0))


def binding_cap(reasoning_budget: int, reasoning_tokens: int | None, finish_reason: str | None) -> bool | None:
    """Was the requested allowance binding? Unknown reasoning usage without a length stop is unknown."""
    if finish_reason == "length":
        return True
    if reasoning_tokens is None:
        return None
    return reasoning_tokens >= reasoning_budget


def delivered_accuracy_reference(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """rows: {policy_id, status, selected_candidate_id, correct: bool|None}. All-null 'selected' rows block."""
    rows = list(rows)
    if not rows:
        raise ValueError("no selector rows: blocker, not zero accuracy")
    broken = [r["policy_id"] for r in rows if r["status"] == "selected" and r["selected_candidate_id"] is None]
    if broken:
        raise ValueError(f"selector rows claim a selection but carry none: {broken[:3]}")
    known = [r for r in rows if r["status"] != "paused_public_check_error"]
    if not known:
        raise ValueError("every selector row is paused: blocker")
    correct = sum(1 for r in known if r.get("correct") is True)
    unknown = sum(1 for r in known if r["status"] in {"selected", "single_candidate"} and r.get("correct") is None)
    n = len(known)
    return {"n": n, "lower": Fraction(correct, n), "upper": Fraction(correct + unknown, n),
            "abstained": sum(1 for r in known if r["status"] == "abstained_all_invalid")}
