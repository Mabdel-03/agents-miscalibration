"""Deterministic tiered cell manifests (WP5).

Spec §6.2 (frozen nested matrix; every panel a balanced hash-prefix subset), §6.3 (neutral
``00`` root wording at every N of the membership panel; CEN_FLAT hub/worker roles), §5.5
(four framing banks, R_bank=10), §5.6 (E: fully reset episodes, ``episode_rep``), §4.6
(D degree module, one-round controls), §3.6 (engine-seed collision check: uniqueness
within every episode), §10.4 (resumable idempotent shards).  Architecture §1.13, §2.3
(F before A so aliases are hits), §3.  Corrections P0-1 (alias table → framing per cell:
main DEC ``NATIVE``; N-panel DEC/IND/one-round controls ``00``; IND-11 in module A),
P0-3 (three waves: generate → ``<tier>-select`` (JUDGE_BEST on sealed pools) →
``<tier>-eval`` (JUDGE_HLE / EVAL_BCB on sealed selections)), P0-4 (items per cell and
``parallel_items``), P0-7 (per-lane manifests and job names), P1-2 (N=9 only at B4;
M-cross B1 only for N=3), §5 cut order (C forecasts = tier ``2b`` is cut).
Design brief "Proposed scope" as amended.

Blocks: main items are ranked per domain (``PublicTask.rank``); a block is 50 items =
25 HLE + 25 BCB interleaved by rank (h0, b0, h1, b1, …) so every shard of a block holds
both domains and any completed prefix of blocks is balanced.  Within a block the dispatch
order is F00, F11, F01, F10 → S_HISTORY → DEC → CEN_FLAT → IND_VOTE → S_FRESH (ops order:
the long poles first; F before A because A aliases F).

``cell_id`` = ``{module}.{method}.{ckpt}.N{N}.B{B}.F{framing}.e{rep}[.d{degree}].s{shard:03d}``
for generate cells and ``eval.{kind}.{ckpt}.N0.B0.Fnat.e0.x{seal[:8]}.s{shard:03d}`` for
evaluation-kind cells (the seal prefix keeps waves of different seals distinct).

Evaluation-kind cells (``JUDGE_BEST`` / ``JUDGE_HLE`` / ``EVAL_BCB``) span every method's
sealed pools of an item, so ``CellSpec.method`` is not applicable to them: it carries the
placeholder ``Method.BANK`` (the other fixed-opportunity, non-budgeted kind) and ``kind`` is
the discriminator.  The seal they consume is recorded as the ``depends_on`` entry
``seal:<manifest_sha>`` (:func:`cell_seal`), keeping the WP0 ``CellSpec`` contract unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.experiment import io
from agents_scaling.study import identity
from agents_scaling.study.config import StudyConfig, load_config
from agents_scaling.study.data.public import load_public_tasks
from agents_scaling.study.types import (
    NS_CEN_FLAT,
    NS_DEC,
    NS_DEGREE,
    NS_S_HISTORY,
    NS_STATELESS_BANK,
    PURPOSE_CONSUMER,
    PURPOSE_HUB,
    PURPOSE_REVISE,
    PURPOSE_ROOT,
    PURPOSE_WORKER,
    CellKind,
    CellSpec,
    Domain,
    Framing,
    Method,
    ProtocolError,
    PublicTask,
    SeedKey,
)

LANES: tuple[str, ...] = ("32B", "14B", "8B", "4B", "eval")
GENERATE_TIERS: tuple[str, ...] = ("pilot", "1", "2", "2b", "3")
TIERS: tuple[str, ...] = GENERATE_TIERS + tuple(f"{t}-select" for t in ("pilot", "1", "2", "3")) + tuple(
    f"{t}-eval" for t in ("pilot", "1", "2", "3")
)
BLOCK_ITEMS = 50
BLOCK_PER_DOMAIN = BLOCK_ITEMS // 2
PILOT_ITEMS = 20
PILOT_S_HISTORY_ITEMS = 4
R_BANK = 10
BANK_FRAMINGS: tuple[Framing, ...] = (Framing.F00, Framing.F11, Framing.F01, Framing.F10)
#: Dispatch order of the module-A methods inside a block (04_critic_corrections P0-4; ops §3).
A_METHOD_ORDER: tuple[Method, ...] = (Method.S_HISTORY, Method.DEC, Method.CEN_FLAT, Method.IND_VOTE, Method.S_FRESH)
N_PANEL: tuple[int, ...] = (1, 2, 3, 5, 9)
DEGREES: tuple[int, ...] = (0, 1, 2, 4, 8)
E_REPS: tuple[int, ...] = (1, 2, 3, 4, 5)
E_METHODS: tuple[Method, ...] = (Method.S_FRESH, Method.IND_VOTE, Method.DEC, Method.CEN_FLAT)
M_DENSE_SIZES: tuple[str, ...] = ("4B", "8B", "14B")
M_METHODS: tuple[Method, ...] = (Method.IND_VOTE, Method.DEC, Method.CEN_FLAT)
M_CROSS_SIZES: tuple[str, ...] = ("8B", "32B")
M_CROSS_NS: tuple[int, ...] = (3, 9)
A_BUDGET_MULTIPLIERS: tuple[int, ...] = (1, 2)
JUDGE_ITEMS_PER_CELL = 25
JUDGE_MAX_INFLIGHT = 8
#: Review P1-2: one sequential 1-CPU cell at ~5 s x ~265 candidates x 10 items ≈ 3.7 h, well
#: under the 24 h eval walltime (200 items ≈ 73 h needed three walltime kills per cell).
EVAL_BCB_ITEMS_PER_CELL = 10


@dataclass(frozen=True)
class Sizing:
    """Items per cell, outer item parallelism and per-item request concurrency (P0-4)."""

    items: int
    parallel_items: int
    max_inflight: int | None  # None → N (symmetric groups run in parallel)


SIZING: Mapping[Method, Sizing] = {
    Method.BANK: Sizing(20, 2, R_BANK),
    Method.S_FRESH: Sizing(10, 1, 8),
    Method.IND_VOTE: Sizing(10, 1, 8),
    Method.S_HISTORY: Sizing(4, 4, 1),
    Method.DEC: Sizing(10, 2, None),
    Method.DEC_ONE_ROUND: Sizing(10, 2, None),
    Method.IND_PRIVATE_REVISION: Sizing(10, 2, None),
    Method.CEN_FLAT: Sizing(5, 2, None),
    Method.DEGREE: Sizing(10, 2, 9),
}


class ManifestError(ValueError):
    """A manifest request that the frozen matrix does not admit."""


SEAL_DEP_PREFIX = "seal:"
EVAL_METHOD_PLACEHOLDER = Method.BANK


def seal_dependency(seal: str) -> str:
    if not isinstance(seal, str) or len(seal) != 64:
        raise ManifestError("seal must be the 64-hex sha256 of the sealed cells manifest")
    return SEAL_DEP_PREFIX + seal


def cell_seal(cell: CellSpec) -> str | None:
    """The seal an evaluation-kind cell consumes (``None`` for generate cells)."""
    seals = [d[len(SEAL_DEP_PREFIX):] for d in cell.depends_on if d.startswith(SEAL_DEP_PREFIX)]
    if len(seals) > 1:
        raise ProtocolError(f"cell {cell.cell_id} names several seals")
    return seals[0] if seals else None


# --------------------------------------------------------------------------- items / blocks


def interleave_domains(tasks: Sequence[PublicTask]) -> list[PublicTask]:
    """h0, b0, h1, b1, … by rank; a shorter domain's tail is appended (still rank order)."""
    by_domain: dict[Domain, list[PublicTask]] = {d: [] for d in Domain}
    for task in tasks:
        by_domain[Domain(task.domain)].append(task)
    hle = sorted(by_domain[Domain.HLE], key=lambda t: t.rank)
    bcb = sorted(by_domain[Domain.BCB], key=lambda t: t.rank)
    out: list[PublicTask] = []
    for i in range(max(len(hle), len(bcb))):
        if i < len(hle):
            out.append(hle[i])
        if i < len(bcb):
            out.append(bcb[i])
    return out


def split_items(tasks: Sequence[PublicTask], split: str, n_total: int | None = None) -> list[PublicTask]:
    """The first ``n_total`` items of ``split`` balanced per domain (rank prefix; §6.2).

    ``n_total`` must be even (``n_total/2`` per domain) and both domains must hold that many
    items; ``None`` takes every item of the split.
    """
    pool = [t for t in tasks if t.split == split]
    if n_total is None:
        return interleave_domains(pool)
    if n_total <= 0 or n_total % 2:
        raise ManifestError(f"panel size must be a positive even total, got {n_total}")
    per = n_total // 2
    out: list[PublicTask] = []
    for domain in Domain:
        group = sorted((t for t in pool if Domain(t.domain) is domain), key=lambda t: t.rank)
        if len(group) < per:
            raise ManifestError(f"{split}/{domain.value}: {len(group)} items < panel {per} per domain")
        if [t.rank for t in group[:per]] != list(range(per)):
            raise ManifestError(f"{split}/{domain.value}: ranks are not a contiguous 0..n-1 prefix")
        out.extend(group[:per])
    return interleave_domains(out)


def blocks_of(items: Sequence[PublicTask], block_items: int = BLOCK_ITEMS) -> list[list[PublicTask]]:
    """Consecutive blocks of the interleaved item order (the last block may be shorter)."""
    if block_items <= 0:
        raise ValueError("block_items must be positive")
    return [list(items[i : i + block_items]) for i in range(0, len(items), block_items)]


def shards_of(items: Sequence[PublicTask], per_cell: int) -> list[list[str]]:
    """Item-id shards of ``per_cell`` consecutive interleaved items (both domains per shard)."""
    if per_cell <= 0:
        raise ValueError("per_cell must be positive")
    return [[t.source_id for t in items[i : i + per_cell]] for i in range(0, len(items), per_cell)]


# --------------------------------------------------------------------------- cell construction


def cell_id_for(
    module: str,
    method: Method,
    checkpoint: str,
    N: int,
    B: int,
    framing: Framing,
    episode_rep: int,
    shard: int,
    *,
    degree: int | None = None,
    kind: CellKind = CellKind.GENERATE,
    seal: str | None = None,
) -> str:
    if kind is CellKind.GENERATE:
        deg = f".d{degree}" if degree is not None else ""
        return f"{module}.{method.value}.{checkpoint}.N{N}.B{B}.F{framing.value}.e{episode_rep}{deg}.s{shard:03d}"
    if not seal:
        raise ManifestError(f"{kind.value} cells need the seal they consume")
    return f"{module}.{kind.value}.{checkpoint}.N0.B0.F{Framing.NATIVE.value}.e0.x{seal[:8]}.s{shard:03d}"


class _Builder:
    """Accumulates cells with per-configuration shard counters (deterministic order)."""

    def __init__(self, cfg: StudyConfig, lane: str) -> None:
        self.cfg = cfg
        self.lane = lane
        self.cells: list[CellSpec] = []
        self._shards: dict[tuple[Any, ...], int] = {}
        self._ids: set[str] = set()

    def _next_shard(self, key: tuple[Any, ...]) -> int:
        n = self._shards.get(key, 0)
        self._shards[key] = n + 1
        return n

    def add(self, cell: CellSpec) -> None:
        if cell.cell_id in self._ids:
            raise ManifestError(f"duplicate cell_id {cell.cell_id}")
        if cell.lane != self.lane:
            return
        self._ids.add(cell.cell_id)
        self.cells.append(cell)

    def generate(
        self,
        *,
        module: str,
        method: Method,
        checkpoint: str,
        N: int,
        B: int,
        framing: Framing,
        items: Sequence[PublicTask],
        split: str,
        episode_rep: int = 0,
        degree: int | None = None,
        depends_on: Sequence[str] = (),
        sizing: Sizing | None = None,
    ) -> list[CellSpec]:
        if checkpoint not in self.cfg.checkpoints:
            raise ManifestError(f"unknown checkpoint {checkpoint!r}")
        sz = sizing or SIZING[method]
        made: list[CellSpec] = []
        key = (module, method.value, checkpoint, N, B, framing.value, episode_rep, degree)
        for shard_items in shards_of(items, sz.items):
            shard = self._next_shard(key)
            cell = CellSpec(
                cell_id=cell_id_for(module, method, checkpoint, N, B, framing, episode_rep, shard, degree=degree),
                kind=CellKind.GENERATE,
                module=module,
                method=method,
                checkpoint=checkpoint,
                N=N,
                B=B,
                framing=framing,
                episode_rep=episode_rep,
                split=split,
                items=tuple(shard_items),
                depends_on=tuple(depends_on),
                max_inflight=sz.max_inflight if sz.max_inflight is not None else max(1, N),
                parallel_items=sz.parallel_items,
                lane=checkpoint,
                degree=degree,
            )
            self.add(cell)
            made.append(cell)
        return made

    def evaluation(
        self,
        *,
        kind: CellKind,
        checkpoint: str,
        items: Sequence[str],
        split: str,
        seal: str,
        per_cell: int,
        max_inflight: int,
        lane: str,
        depends_on: Sequence[str] = (),
    ) -> list[CellSpec]:
        made: list[CellSpec] = []
        key = ("eval", kind.value, checkpoint, seal)
        for i in range(0, len(items), per_cell):
            shard_items = list(items[i : i + per_cell])
            shard = self._next_shard(key)
            cell = CellSpec(
                cell_id=cell_id_for("eval", EVAL_METHOD_PLACEHOLDER, checkpoint, 0, 0, Framing.NATIVE, 0, shard, kind=kind, seal=seal),
                kind=kind,
                module="eval",
                method=EVAL_METHOD_PLACEHOLDER,
                checkpoint=checkpoint,
                N=0,
                B=0,
                framing=Framing.NATIVE,
                episode_rep=0,
                split=split,
                items=tuple(shard_items),
                depends_on=(seal_dependency(seal), *depends_on),
                max_inflight=max_inflight,
                parallel_items=1,
                lane=lane,
                degree=None,
            )
            self.add(cell)
            made.append(cell)
        return made


def _framing_for(method: Method, module: str) -> Framing:
    """Alias table (P0-1): main-tier DEC is truthful-native; IND in module A is ``11``;
    every membership-panel / control / degree / E / M root is neutral ``00`` unless it
    replicates the main tier (E, M dense at N=5, A-budget)."""
    if method is Method.CEN_FLAT:
        return Framing.NATIVE
    if module in ("A", "E", "M", "B"):
        if method is Method.DEC:
            return Framing.NATIVE
        if method is Method.IND_VOTE:
            return Framing.F11
        return Framing.F00
    return Framing.F00


def _a_block(b: _Builder, block: Sequence[PublicTask], *, module: str, checkpoint: str, B: int, split: str,
             episode_rep: int = 0, methods: Sequence[Method] = A_METHOD_ORDER, N: int = 5,
             s_history_items: Sequence[PublicTask] | None = None, deps: Sequence[str] = ()) -> None:
    for method in methods:
        items = block if (method is not Method.S_HISTORY or s_history_items is None) else s_history_items
        if not items:
            continue
        b.generate(module=module, method=method, checkpoint=checkpoint, N=N, B=B, framing=_framing_for(method, module),
                   items=items, split=split, episode_rep=episode_rep, depends_on=deps)


def _banks(b: _Builder, block: Sequence[PublicTask], *, checkpoint: str, split: str) -> list[str]:
    ids: list[str] = []
    for framing in BANK_FRAMINGS:
        for cell in b.generate(module="F", method=Method.BANK, checkpoint=checkpoint, N=1, B=0, framing=framing, items=block, split=split):
            ids.append(cell.cell_id)
    return ids


# --------------------------------------------------------------------------- tiers


def build_cells(
    cfg: StudyConfig,
    tasks: Sequence[PublicTask],
    tier: str,
    lane: str,
    *,
    seal: str | None = None,
    sealed_items: Sequence[str] | None = None,
    eval_bcb_items_per_cell: int = EVAL_BCB_ITEMS_PER_CELL,
) -> list[CellSpec]:
    """The deterministic cell list of ``tier`` restricted to ``lane`` (see module docstring).

    ``seal``/``sealed_items`` are required for the ``*-select`` and ``*-eval`` tiers: the
    seal directory name (manifest sha256 of the sealed generate manifest) and the item ids
    it covers (``seal.sealed_item_ids``).  Raises :class:`ManifestError` for a tier the
    plan does not run (``2b``: C forecasts, cut per 04_critic_corrections §5).
    """
    if tier not in TIERS:
        raise ManifestError(f"unknown tier {tier!r}; expected one of {TIERS}")
    if lane not in LANES:
        raise ManifestError(f"unknown lane {lane!r}; expected one of {LANES}")
    flagship = cfg.flagship
    B4 = int(cfg.budget.primary)
    b = _Builder(cfg, lane)

    if tier.endswith("-select") or tier.endswith("-eval"):
        if not seal or sealed_items is None:
            raise ManifestError(f"tier {tier} needs --seal (the sealed manifest sha) and its sealed item ids")
        by_id = {t.source_id: t for t in tasks}
        missing = [sid for sid in sealed_items if sid not in by_id]
        if missing:
            raise ManifestError(f"sealed items absent from the public export: {missing[:5]}")
        ordered = interleave_domains([by_id[sid] for sid in sealed_items])
        split = "dev" if tier.startswith("pilot") else "main"
        if any(t.split != split for t in ordered):
            raise ManifestError(f"tier {tier} expects {split} items only")
        judge_ckpt = cfg.judge_checkpoint
        if tier.endswith("-select"):
            b.evaluation(kind=CellKind.JUDGE_BEST, checkpoint=judge_ckpt, items=[t.source_id for t in ordered], split=split,
                         seal=seal, per_cell=JUDGE_ITEMS_PER_CELL, max_inflight=JUDGE_MAX_INFLIGHT, lane=judge_ckpt)
        else:
            hle = [t.source_id for t in ordered if Domain(t.domain) is Domain.HLE]
            bcb = [t.source_id for t in ordered if Domain(t.domain) is Domain.BCB]
            b.evaluation(kind=CellKind.JUDGE_HLE, checkpoint=judge_ckpt, items=hle, split=split, seal=seal,
                         per_cell=JUDGE_ITEMS_PER_CELL, max_inflight=JUDGE_MAX_INFLIGHT, lane=judge_ckpt)
            if eval_bcb_items_per_cell <= 0:
                raise ManifestError("eval_bcb_items_per_cell must be positive")
            b.evaluation(kind=CellKind.EVAL_BCB, checkpoint=judge_ckpt, items=bcb, split=split, seal=seal,
                         per_cell=eval_bcb_items_per_cell, max_inflight=1, lane="eval")
        return b.cells

    if tier == "2b":
        raise ManifestError("tier 2b (C shadow forecasts) is cut: 04_critic_corrections §4 item 10 / §5 item 2")

    if tier == "pilot":
        items = split_items(tasks, "dev", PILOT_ITEMS)
        history = interleave_domains(items)[:PILOT_S_HISTORY_ITEMS]
        deps = _banks(b, items, checkpoint=flagship, split="dev")
        _a_block(b, items, module="A", checkpoint=flagship, B=B4, split="dev", s_history_items=history, deps=deps)
        return b.cells

    main = split_items(tasks, "main")
    if tier == "1":
        for block in blocks_of(main):
            deps = _banks(b, block, checkpoint=flagship, split="main")
            _a_block(b, block, module="A", checkpoint=flagship, B=B4, split="main", deps=deps)
        return b.cells

    panels = cfg.items.panels
    if tier == "2":
        # M dense: 100 items committed, 150 as a nested extension (blocks 0-1 then 2).
        m_items = split_items(tasks, "main", int(panels["M_extension"]))
        for size in M_DENSE_SIZES:
            for block in blocks_of(m_items):
                for method in M_METHODS:
                    b.generate(module="M", method=method, checkpoint=size, N=5, B=B4, framing=_framing_for(method, "M"),
                               items=block, split="main")
        # N membership panel (32B): neutral 00 roots at every N; N=9 only at B4 (P1-2).
        n_items = split_items(tasks, "main", int(panels["N"]))
        for block in blocks_of(n_items):
            for N in N_PANEL:
                for method in (Method.IND_VOTE, Method.DEC, Method.CEN_FLAT):
                    b.generate(module="N", method=method, checkpoint=flagship, N=N, B=B4, framing=_framing_for(method, "N"),
                               items=block, split="main")
            for method in (Method.IND_PRIVATE_REVISION, Method.DEC_ONE_ROUND):
                b.generate(module="N", method=method, checkpoint=flagship, N=5, B=B4, framing=Framing.F00, items=block, split="main")
        # D degree module.
        d_items = split_items(tasks, "main", int(panels["D"]))
        for block in blocks_of(d_items):
            for d in DEGREES:
                b.generate(module="D", method=Method.DEGREE, checkpoint=flagship, N=1, B=B4, framing=Framing.F00,
                           items=block, split="main", degree=d)
        # E repeated episodes: reps 1..5 (rep 0 is the module-A record, §5.6).
        e_items = split_items(tasks, "main", int(panels["E"]))
        for rep in E_REPS:
            for method in E_METHODS:
                b.generate(module="E", method=method, checkpoint=flagship, N=5, B=B4, framing=_framing_for(method, "E"),
                           items=e_items, split="main", episode_rep=rep)
        return b.cells

    if tier == "3":
        ab_items = split_items(tasks, "main", int(panels["B"]))
        for mult in A_BUDGET_MULTIPLIERS:
            if mult not in cfg.budget.budget_multipliers:
                raise ManifestError(f"budget multiplier {mult} is not in the frozen grid")
            for block in blocks_of(ab_items):
                _a_block(b, block, module="B", checkpoint=flagship, B=mult, split="main")
        for size in M_CROSS_SIZES:
            for N in M_CROSS_NS:
                multipliers = (B4,) if N == 9 else (B4, 1)
                for mult in multipliers:
                    for block in blocks_of(ab_items):
                        for method in M_METHODS:
                            b.generate(module="MX", method=method, checkpoint=size, N=N, B=mult,
                                       framing=_framing_for(method, "N"), items=block, split="main")
        return b.cells
    raise ManifestError(f"tier {tier!r} is declared but not built")  # pragma: no cover


# --------------------------------------------------------------------------- invariants


def episode_seed_keys(cell: CellSpec, task: PublicTask, cfg: StudyConfig) -> list[SeedKey]:
    """Every semantic-seed key the policy of ``cell`` can issue for ``task`` (upper bound).

    Used for the §3.6 collision check (engine-seed uniqueness within an episode).  The key
    set is the frozen seed table of architecture §2.3 / WP4: stateless draws
    ``(0, root, k, stateless_bank)`` for ``k < solver_calls``; S_HISTORY revisions
    ``(0, revise, j, S_HISTORY)``; DEC revisions ``(s, revise, r, DEC)``; CEN hub
    ``(0, hub, c, CEN_FLAT)`` and workers ``(slot, worker, c, CEN_FLAT)``; DEGREE focal
    ``(0, revise, d, DEGREE)`` and consumers ``(j, consumer, d, DEGREE)``.
    """
    if cell.kind is not CellKind.GENERATE:
        return []
    ckpt = cfg.checkpoint(cell.checkpoint)
    cap = int(cfg.caps.solver_calls)

    def key(actor: int, purpose: str, step: int, ns: str) -> SeedKey:
        return SeedKey(task.source_id, task.split, ckpt.model_cell, int(cell.episode_rep), actor, purpose, step, ns)

    keys: list[SeedKey] = []
    method = Method(cell.method)
    if method is Method.BANK:
        keys += [key(0, PURPOSE_ROOT, k, NS_STATELESS_BANK) for k in range(R_BANK)]
    elif method in (Method.S_FRESH, Method.IND_VOTE):
        keys += [key(0, PURPOSE_ROOT, k, NS_STATELESS_BANK) for k in range(cap)]
    elif method is Method.S_HISTORY:
        keys += [key(0, PURPOSE_ROOT, 0, NS_STATELESS_BANK)] + [key(0, PURPOSE_REVISE, j, NS_S_HISTORY) for j in range(1, cap)]
    elif method in (Method.DEC, Method.DEC_ONE_ROUND, Method.IND_PRIVATE_REVISION):
        rounds = int(cfg.caps.dec_max_rounds) if method is Method.DEC else 1
        keys += [key(0, PURPOSE_ROOT, s, NS_STATELESS_BANK) for s in range(cell.N)]
        keys += [key(s, PURPOSE_REVISE, r, NS_DEC) for r in range(1, rounds + 1) for s in range(cell.N)]
    elif method is Method.CEN_FLAT:
        cycles = int(cfg.caps.cen_max_cycles)
        keys += [key(0, PURPOSE_HUB, c, NS_CEN_FLAT) for c in range(cycles + 1)]
        keys += [key(w, PURPOSE_WORKER, c, NS_CEN_FLAT) for c in range(cycles) for w in range(1, cell.N)]
    elif method is Method.DEGREE:
        d = int(cell.degree if cell.degree is not None else -1)
        keys += [key(0, PURPOSE_ROOT, i, NS_STATELESS_BANK) for i in range(9)]
        keys += [key(0, PURPOSE_REVISE, d, NS_DEGREE)] + [key(j, PURPOSE_CONSUMER, d, NS_DEGREE) for j in range(1, 5)]
    else:
        raise ManifestError(f"no seed table for method {method.value}")
    return keys


def assert_engine_seed_uniqueness(cells: Iterable[CellSpec], tasks: Mapping[str, PublicTask], cfg: StudyConfig) -> int:
    """§3.6 collision check: distinct engine seeds within every (cell, item) episode.

    Returns the number of (episode, key) pairs checked; raises ``ProtocolError`` on a
    collision (a harness defect: two opportunities of one episode would share an engine seed).
    """
    checked = 0
    for cell in cells:
        for sid in cell.items:
            task = tasks.get(sid)
            if task is None:
                raise ProtocolError(f"cell {cell.cell_id} names unknown item {sid}")
            seen: dict[int, SeedKey] = {}
            for k in episode_seed_keys(cell, task, cfg):
                seed = identity.engine_seed(k.semantic_seed(cfg.study_seed))
                other = seen.get(seed)
                if other is not None and other != k:
                    raise ProtocolError(f"engine seed collision in {cell.cell_id}/{sid}: {other.as_array()} vs {k.as_array()}")
                seen[seed] = k
                checked += 1
    return checked


def assert_tier_nesting(cells: Sequence[CellSpec], tasks: Mapping[str, PublicTask]) -> None:
    """Every module's item set is a balanced rank prefix per domain (§6.2 nested panels)."""
    # study-v4: select/eval tiers derive their item set from a SEAL (the complete pools at
    # sealing time), which is not necessarily a rank prefix while generation is still
    # running; the nesting rule (§6.2 balanced hash prefixes) applies to generate tiers.
    if any(getattr(c.kind, "value", c.kind) in ("JUDGE_BEST", "JUDGE_HLE", "EVAL_BCB", "FORECAST") for c in cells):
        return
    by_module: dict[str, set[str]] = {}
    for cell in cells:
        by_module.setdefault(cell.module, set()).update(cell.items)
    for module, ids in by_module.items():
        for domain in Domain:
            ranks = sorted(tasks[sid].rank for sid in ids if Domain(tasks[sid].domain) is domain)
            if ranks and ranks != list(range(len(ranks))):
                raise ProtocolError(f"module {module}/{domain.value}: items are not a rank prefix (ranks {ranks[:6]}…)")
        counts = {d: sum(1 for sid in ids if Domain(tasks[sid].domain) is d) for d in Domain}
        if counts[Domain.HLE] and counts[Domain.BCB] and counts[Domain.HLE] != counts[Domain.BCB]:
            raise ProtocolError(f"module {module}: unbalanced domains {counts}")


def assert_f_before_a(cells: Sequence[CellSpec]) -> None:
    """Within each item block the F banks precede every A cell that aliases them (§2.3)."""
    first_a: dict[str, int] = {}
    last_f: dict[str, int] = {}
    for index, cell in enumerate(cells):
        for sid in cell.items:
            if cell.module == "F":
                last_f[sid] = max(last_f.get(sid, -1), index)
            elif cell.module == "A":
                first_a.setdefault(sid, index)
    for sid, a_index in first_a.items():
        if sid in last_f and last_f[sid] > a_index:
            raise ProtocolError(f"item {sid}: an A cell (index {a_index}) precedes its F bank (index {last_f[sid]})")


# --------------------------------------------------------------------------- files


def cells_file_sha256(path: str | os.PathLike) -> str:
    return identity.sha256_hex(Path(path).read_bytes())


def load_cells_file(path: str | os.PathLike) -> list[CellSpec]:
    """Strict load of a cells manifest (verifies the ``.sha256`` sidecar when present)."""
    path = Path(path)
    raw = path.read_bytes()
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").split()[0]
        actual = identity.sha256_hex(raw)
        if expected != actual:
            raise ProtocolError(f"{path}: sha256 {actual} != frozen {expected} (edited manifest)")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        raise ProtocolError(f"{path}: expected an object with a 'cells' list")
    cells = [CellSpec.from_dict(c) for c in data["cells"]]
    ids = [c.cell_id for c in cells]
    if len(set(ids)) != len(ids):
        raise ProtocolError(f"{path}: duplicate cell ids")
    return cells


def write_cells_file(path: str | os.PathLike, cells: Sequence[CellSpec], meta: Mapping[str, Any]) -> str:
    """Write ``<path>`` + ``<path>.sha256``; refuse to overwrite a *differing* frozen manifest.

    Returns the manifest sha256 (the seal directory name of this manifest).
    """
    path = Path(path)
    payload = json.dumps({"meta": dict(meta), "cells": [c.to_dict() for c in cells]}, indent=1, sort_keys=True) + "\n"
    digest = identity.sha256_hex(payload)
    sidecar = path.with_name(path.name + ".sha256")
    if path.exists():
        existing = path.read_bytes()
        if identity.sha256_hex(existing) != digest:
            raise ProtocolError(f"{path} exists with a different content; refusing to overwrite a frozen manifest")
        return digest
    io.atomic_write_text(path, payload)
    io.atomic_write_text(sidecar, f"{digest}  {path.name}\n")
    return digest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.study.cells", description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--tier", required=True, choices=TIERS)
    p.add_argument("--lane", required=True, choices=LANES)
    p.add_argument("--out", required=True, help="file name under the run root (e.g. cells_1_32B.json)")
    p.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    p.add_argument("--seal", default=None, help="select/eval tiers: the sealed generate manifest sha (seal dir name)")
    p.add_argument("--eval-bcb-items-per-cell", type=int, default=EVAL_BCB_ITEMS_PER_CELL,
                   help="EVAL_BCB items per (sequential, 1-CPU) cell; 10 ≈ 5 s x ~265 candidates x 10 items ≈ 3.7 h per cell (review P1-2)")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    run_root = Path(args.results_root) / args.run_id
    tasks = load_public_tasks(run_root)
    sealed: list[str] | None = None
    if args.seal:
        from agents_scaling.study.selection.seal import load_pools, sealed_item_ids

        sealed = sorted(sealed_item_ids(load_pools(run_root, args.seal)))
    cells = build_cells(cfg, tasks, args.tier, args.lane, seal=args.seal, sealed_items=sealed, eval_bcb_items_per_cell=args.eval_bcb_items_per_cell)
    index = {t.source_id: t for t in tasks}
    checked = assert_engine_seed_uniqueness(cells, index, cfg)
    assert_tier_nesting(cells, index)
    assert_f_before_a(cells)
    meta = {
        "run_id": args.run_id,
        "tier": args.tier,
        "lane": args.lane,
        "config_sha256": cfg.config_sha256,
        "study_id": cfg.study_id,
        "seal": args.seal,
        "n_cells": len(cells),
        "n_items": len({sid for c in cells for sid in c.items}),
        "seed_keys_checked": checked,
        "code_version": io.git_commit(),
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **meta}, indent=2))
        return 0
    out = run_root / args.out
    digest = write_cells_file(out, cells, meta)
    print(json.dumps({"cells_file": str(out), "sha256": digest, **meta}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [name for name in globals() if not name.startswith("_")]
