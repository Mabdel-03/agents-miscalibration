"""Frozen ARD36 cell manifests, one builder per experiment module (WP4).

Spec §5.1-§5.4 (cohorts and splits), §9 (E1 root banks), §10 (E2a natural 8B panel, E2b
32B replication, E2c challenge starts, E2d matched-root budget replication), §11 (E3
receiver), §12 (E4 forks, GATE / RANDOM_PROTECT, selector audit), §13 (E5 common suffix),
§14.2 (E6 capture cohort), §18 (manifest hashes).  Plan §5.3 (modules → cells), §6 (cell
sizes and lanes), §7 (dispatch priority: train+validation first in every 8B manifest so the
controller can freeze while test trajectories generate).

Reuse (never re-implemented): ``study/cells.py`` ``_Builder`` (shard counters + duplicate-id
refusal), ``shards_of`` (consecutive interleaved shards), ``write_cells_file`` (frozen
manifest + ``.sha256`` sidecar, refusing to overwrite a differing manifest).
``interleave_domains`` is rebound to the ARD family axis by :func:`interleave_families` —
``study.types.Domain`` is frozen to ``('hle', 'bcb')`` and ARD36's axis is
``('math500', 'mmlu_pro')`` (plan §8: the family axis is rebound, never edited in place).

Label discipline: a manifest carries item ids, opaque team tags and root candidate ids
only.  ``constructed_C0``, eligibility counts and correctness never appear in a cell id, a
cell field or a cell option block — they live in the analysis-side ``challenge_teams``
table (spec §5.3, §12.2).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from agents_scaling.ard import contract as K
from agents_scaling.ard import kinds as KINDS
from agents_scaling.ard.items import ArdTask, large_panel, load_ard_tasks
from agents_scaling.ard.types import (
    ANALYSIS_ITEM,
    ArdCellKind,
    ArdCellSpec,
    ArdManifestError,
    cell_dependency,
    controller_dependency,
    state_dependency,
)
from agents_scaling.study import identity
from agents_scaling.study.cells import (  # reused verbatim
    _Builder,
    shards_of,
    write_cells_file,
)
from agents_scaling.study.types import ProtocolError

MODULES: tuple[str, ...] = ("E1", "E2a", "E2b", "E2c", "E2d", "E3", "E4", "E5", "E6", "OPS")

#: Split dispatch order (plan §7 / spec §10 E2a): train and validation first, test last.
SPLIT_ORDER: tuple[str, ...] = ("train", "validation", "test")

_TEAM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
#: Team tags must be opaque.  ``c1`` / ``C0=4`` / ``correct`` would put the label-conditioned
#: composition into the manifest, which spec §5.3 forbids.
_LEAKY_TEAM_RE = re.compile(r"(?i)(c0|c_?[0-5]\b|correct|wrong|gold)")


class _IdView(NamedTuple):
    """Adapter so the frozen ``study.cells.shards_of`` chunks ARD tasks by ``item_id``."""

    source_id: str


def assert_opaque_team_id(team_id: str) -> str:
    """Refuse a team tag that encodes its own composition (spec §5.3)."""
    if not _TEAM_RE.match(str(team_id)):
        raise ArdManifestError(f"team tag {team_id!r} must match {_TEAM_RE.pattern}")
    if _LEAKY_TEAM_RE.search(str(team_id)):
        raise ArdManifestError(
            f"team tag {team_id!r} encodes a label-conditioned composition; use an opaque "
            "tag and keep constructed_C0 in challenge_teams.parquet (spec §5.3)")
    return str(team_id)


def interleave_families(tasks: Sequence[ArdTask]) -> list[ArdTask]:
    """m0, p0, m1, p1, … by rank, so any completed prefix is family-balanced.

    The ARD analogue of ``study.cells.interleave_domains`` (same algorithm, family axis).
    """
    by_family: dict[str, list[ArdTask]] = {family.value: [] for family in K.FAMILIES}
    for task in tasks:
        if task.family not in by_family:
            raise ArdManifestError(f"task {task.item_id} has unknown family {task.family!r}")
        by_family[task.family].append(task)
    ordered = [sorted(by_family[family.value], key=lambda t: t.rank) for family in K.FAMILIES]
    out: list[ArdTask] = []
    for index in range(max((len(group) for group in ordered), default=0)):
        for group in ordered:
            if index < len(group):
                out.append(group[index])
    return out


def dispatch_order(tasks: Iterable[ArdTask]) -> list[ArdTask]:
    """Train and validation items first, each family-interleaved (plan §7 / spec §10 E2a).

    The order changes dispatch only: every seed is a pure function of the seed key, so an
    item's requests are byte-identical whatever position it takes here.
    """
    pool = list(tasks)
    out: list[ArdTask] = []
    for split in SPLIT_ORDER:
        out.extend(interleave_families([t for t in pool if t.split == split]))
    unknown = [t for t in pool if t.split not in SPLIT_ORDER]
    if unknown:
        raise ArdManifestError(f"{len(unknown)} tasks carry a split outside {SPLIT_ORDER}")
    return out


def item_shards(tasks: Sequence[ArdTask], per_cell: int) -> list[list[str]]:
    """Item-id shards of ``per_cell`` consecutive dispatch-ordered items."""
    return shards_of([_IdView(task.item_id) for task in tasks], per_cell)


# --------------------------------------------------------------------------- cell ids


def parent_tag(parent_ref: str | None) -> str:
    """The 8-hex parent hash carried by every dependent stage's cell id.

    Two waves of one dependent stage (a re-run against a *new* controller or a *new* sealed
    state) therefore never share a cell directory, so the second wave can never resume onto
    the first wave's item files.
    """
    if parent_ref is None:
        return ""
    value = str(parent_ref)
    for prefix in ("controller:", "state:", "cell:"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    digest = value if re.fullmatch(r"[0-9a-f]{8,64}", value) else identity.sha256_hex(value)
    return digest[:8]


def cell_id_for(
    kind: ArdCellKind,
    *,
    cohort: K.Cohort,
    checkpoint: str,
    shard: int,
    protocol: K.Protocol | None = None,
    reasoning_allowance: int = K.THINKING_TOKENS_LOW,
    round_: int | None = None,
    team_id: str | None = None,
    parent_ref: str | None = None,
    instance: str | None = None,
    split: str | None = None,
    episode_rep: int = 0,
) -> str:
    """``KIND.cohort.PROTOCOL.ckpt.T<budget>[.g<team>][.r<round>][.p<parent8>][.w<inst>].SPLIT.s###``.

    The split is part of the id because shard numbers restart at ``s000`` in every builder,
    and a cell id is a *directory*.  Without it, ``--split test`` and ``--split train``
    manifests of the same module collide: two different item lists share one cell directory,
    one overwrites the other's ``meta.json``, and coverage is counted against the wrong cohort.
    """
    parts = [kind.value, cohort.value, protocol.value if protocol else "none",
             checkpoint, f"T{int(reasoning_allowance)}"]
    if team_id is not None:
        parts.append(f"g{assert_opaque_team_id(team_id)}")
    if round_ is not None:
        parts.append(f"r{int(round_)}")
    tag = parent_tag(parent_ref)
    if tag:
        parts.append(f"p{tag}")
    if instance is not None:
        parts.append(f"w{instance}")
    if int(episode_rep):
        # ``episode_rep`` is a *seed*, and it is part of the shard counter's key -- so two
        # reps each restart at s000 and, without it here, produce the same id.  The receiver
        # grid (2 seeds per condition) is the first thing to ask for a second rep.  Only a
        # non-zero rep is stamped, so every single-seed id is unchanged.
        parts.append(f"e{int(episode_rep)}")
    if split is not None:
        parts.append(str(split))
    parts.append(f"s{int(shard):03d}")
    return ".".join(parts)


# --------------------------------------------------------------------------- builder


class ArdBuilder(_Builder):
    """``study.cells._Builder`` (shard counters, duplicate-id refusal) over ARD cells.

    ``_Builder.add`` only reads ``cell_id`` and ``lane``, so an :class:`ArdCellSpec` passes
    through it unchanged; ``cfg`` is unused because ARD cells never go through
    ``_Builder.generate`` (the v4 factorial).
    """

    def __init__(self, lane: str | None = None) -> None:
        super().__init__(cfg=None, lane=lane)  # type: ignore[arg-type]
        self.lane = lane

    def add(self, cell: ArdCellSpec) -> None:  # type: ignore[override]
        if cell.cell_id in self._ids:
            raise ArdManifestError(f"duplicate cell_id {cell.cell_id}")
        if self.lane is not None and cell.lane != self.lane:
            return
        self._ids.add(cell.cell_id)
        self.cells.append(cell)

    def emit(
        self,
        kind: ArdCellKind,
        tasks: Sequence[ArdTask],
        *,
        cohort: K.Cohort,
        checkpoint: str,
        protocol: K.Protocol | None = None,
        reasoning_allowance: int = K.THINKING_TOKENS_LOW,
        round_: int | None = None,
        team_id: str | None = None,
        parent_ref: str | None = None,
        instance: str | None = None,
        depends_on: Sequence[str] = (),
        leader_seed: str | None = None,
        episode_rep: int = 0,
        options: Mapping[str, Any] | None = None,
        per_cell: int | None = None,
    ) -> list[ArdCellSpec]:
        spec = KINDS.spec_for(kind)
        size = per_cell or KINDS.items_per_cell(
            kind, checkpoint=checkpoint, reasoning_allowance=reasoning_allowance,
            protocol=protocol)
        base_key = (kind.value, cohort.value, protocol.value if protocol else "", checkpoint,
                    reasoning_allowance, round_, team_id, parent_tag(parent_ref), instance,
                    episode_rep)
        made: list[ArdCellSpec] = []
        # One shard never straddles two splits: train and validation are dispatched first
        # (plan §7) and a cell's ``split`` must name exactly one frozen split.
        for split, group in split_groups(tasks):
            # Shard numbers restart per split, and the split is in the cell id, so a cell
            # built by a --split test run can never land on a --split train run's directory.
            key = (*base_key, split)
            for shard_items in item_shards(group, size):
                shard = self._next_shard(key)
                cell = self._make(
                    kind, shard_items, cohort=cohort, checkpoint=checkpoint, protocol=protocol,
                    reasoning_allowance=reasoning_allowance, round_=round_, team_id=team_id,
                    parent_ref=parent_ref, instance=instance, depends_on=depends_on,
                    leader_seed=leader_seed, episode_rep=episode_rep, options=options,
                    split=split, shard=shard, spec=spec)
                self.add(cell)
                made.append(cell)
        return made

    def _make(
        self,
        kind: ArdCellKind,
        shard_items: Sequence[str],
        *,
        cohort: K.Cohort,
        checkpoint: str,
        protocol: K.Protocol | None,
        reasoning_allowance: int,
        round_: int | None,
        team_id: str | None,
        parent_ref: str | None,
        instance: str | None,
        depends_on: Sequence[str],
        leader_seed: str | None,
        episode_rep: int,
        options: Mapping[str, Any] | None,
        split: str,
        shard: int,
        spec: KINDS.KindSpec,
    ) -> ArdCellSpec:
        cell = ArdCellSpec(
            cell_id=cell_id_for(
                kind, cohort=cohort, checkpoint=checkpoint, shard=shard, protocol=protocol,
                reasoning_allowance=reasoning_allowance, round_=round_, team_id=team_id,
                parent_ref=parent_ref, instance=instance, split=split,
                episode_rep=episode_rep),
            kind=kind,
            protocol=protocol,
            round=round_,
            cohort=cohort,
            checkpoint=checkpoint,
            reasoning_allowance=int(reasoning_allowance),
            final_allowance=K.FINAL_TOKENS,
            items=tuple(shard_items),
            team_id=team_id,
            parent_ref=parent_ref,
            leader_seed=leader_seed,
            depends_on=tuple(depends_on),
            max_inflight=spec.max_inflight,
            parallel_items=spec.parallel_items,
            lane=spec.lane_for(checkpoint),
            split=split,
            instance=instance,
            shard=shard,
            episode_rep=int(episode_rep),
            options=dict(options or {}),
        )
        return cell


def split_groups(tasks: Sequence[ArdTask]) -> list[tuple[str, list[ArdTask]]]:
    """``[(split, family-interleaved items)]`` in dispatch order, empty splits dropped."""
    out: list[tuple[str, list[ArdTask]]] = []
    for split in SPLIT_ORDER:
        group = interleave_families([task for task in tasks if task.split == split])
        if group:
            out.append((split, group))
    unknown = [task for task in tasks if task.split not in SPLIT_ORDER]
    if unknown:
        raise ArdManifestError(f"{len(unknown)} tasks carry a split outside {SPLIT_ORDER}")
    return out


# --------------------------------------------------------------------------- modules


def build_e1_roots(
    builder: ArdBuilder, tasks: Sequence[ArdTask], *, checkpoint: str, instance: str | None = None
) -> list[ArdCellSpec]:
    """E1 (spec §9, §5.2): the ``ROOT_DRAWS`` stateless bank of every panel item.

    Draws 0-4 are the common team every protocol shares; 5-15 are the auxiliary bank used
    for E1 statistics and difficulty only.  One frozen root decoding, so no arm ever
    regenerates a root (spec §10 E2d).
    """
    return builder.emit(
        ArdCellKind.ROOTS, tasks, cohort=K.Cohort.NATURAL, checkpoint=checkpoint,
        reasoning_allowance=K.ROOT_THINKING_TOKENS, instance=instance,
        options={"draws": K.ROOT_DRAWS, "team_draws": list(K.TEAM_DRAWS)},
    )


def build_e2_natural(
    builder: ArdBuilder,
    tasks: Sequence[ArdTask],
    *,
    checkpoint: str,
    protocols: Sequence[K.Protocol] | None = None,
    controller_sha: str | None = None,
    instance: str | None = None,
    leader_seed: str | None = None,
) -> list[ArdCellSpec]:
    """E2a / E2b (spec §10): every protocol of the tier on the natural panel.

    8B runs :data:`contract.NATURAL_PROTOCOLS_SMALL`, 32B the three of
    :data:`contract.NATURAL_PROTOCOLS_LARGE` on the frozen 96-item subset.  GATE /
    RANDOM_PROTECT are dispatched here too when the tier admits them; GATE carries
    ``controller:<sha>`` and its cell id carries that hash, so a controller re-freeze lands
    in a new directory.
    """
    chosen = tuple(protocols) if protocols is not None else (
        K.NATURAL_PROTOCOLS_SMALL if checkpoint == K.CHECKPOINT_SMALL
        else K.NATURAL_PROTOCOLS_LARGE)
    made: list[ArdCellSpec] = []
    for protocol in chosen:
        deps = KINDS.required_dependencies(protocol, controller_sha=controller_sha)
        parent = deps[0] if deps else None
        made += builder.emit(
            ArdCellKind.EPISODE, tasks, cohort=K.Cohort.NATURAL, checkpoint=checkpoint,
            protocol=protocol, depends_on=deps, parent_ref=parent, instance=instance,
            leader_seed=leader_seed if protocol is K.Protocol.HUB5 else None,
        )
    return made


def build_e2c_challenge(
    builder: ArdBuilder,
    tasks: Sequence[ArdTask],
    *,
    checkpoint: str,
    team_ids: Sequence[str],
    teams_manifest_sha: str,
    protocols: Sequence[K.Protocol] = K.CHALLENGE_PROTOCOLS,
    controller_sha: str | None = None,
    instance: str | None = None,
) -> list[ArdCellSpec]:
    """E2c (spec §10, §5.3): constructed starts, test items only.

    ``team_ids`` are opaque tags; the cell carries the tag and the frozen teams-manifest
    hash, never the composition.  The three fixed arms run before the controller freeze;
    the two controller arms carry ``controller:<sha>``.
    """
    non_test = [task.item_id for task in tasks if task.split != "test"]
    if non_test:
        raise ArdManifestError(
            f"challenge teams are drawn from natural TEST items only (spec §5.3); got {non_test[:3]}")
    made: list[ArdCellSpec] = []
    for team_id in team_ids:
        for protocol in protocols:
            deps = list(KINDS.required_dependencies(protocol, controller_sha=controller_sha))
            made += builder.emit(
                ArdCellKind.CHALLENGE, tasks, cohort=K.Cohort.CHALLENGE, checkpoint=checkpoint,
                protocol=protocol, team_id=team_id, depends_on=deps,
                parent_ref=deps[0] if deps else f"teams:{teams_manifest_sha}",
                instance=instance,
                options={"teams_manifest_sha256": teams_manifest_sha},
            )
    return made


def build_e2d_high_budget(
    builder: ArdBuilder, tasks: Sequence[ArdTask], *, checkpoint: str, instance: str | None = None
) -> list[ArdCellSpec]:
    """E2d (spec §10): PRIVATE and PEER again at 4,096 thinking tokens on the frozen subset.

    Roots are taken **by reference** from the frozen 1,024-token bank — no root request is
    ever issued at the high budget — so these cells differ from their E2a/E2b twins only in
    ``reasoning_allowance`` (which is part of the request identity through ``Decoding``).
    """
    made: list[ArdCellSpec] = []
    for protocol in K.HIGH_BUDGET_PROTOCOLS:
        made += builder.emit(
            ArdCellKind.EPISODE, tasks, cohort=K.Cohort.HIGH_BUDGET, checkpoint=checkpoint,
            protocol=protocol, reasoning_allowance=K.THINKING_TOKENS_HIGH, instance=instance,
            options={"root_budget": K.ROOT_THINKING_TOKENS, "roots_by_reference": True},
        )
    return made


def build_e3_receiver(
    builder: ArdBuilder,
    programs: Sequence[ArdTask],
    *,
    checkpoint: str,
    presentations: Sequence[str] = K.RECEIVER_PRESENTATIONS,
    budgets: Sequence[int] = (K.THINKING_TOKENS_LOW, K.THINKING_TOKENS_HIGH),
    seeds: int = K.RECEIVER_SEEDS_PER_CONDITION,
    instance: str | None = None,
) -> list[ArdCellSpec]:
    """E3 (spec §11): 5 presentations × 2 budgets × 2 seeds × 2 checkpoints.

    The programs arrive as synthetic ``ArdTask`` rows built by WP5; the manifest only fixes
    which presentation, budget and seed each cell runs.
    """
    made: list[ArdCellSpec] = []
    for budget in budgets:
        for presentation in presentations:
            if presentation not in K.RECEIVER_PRESENTATIONS:
                raise ArdManifestError(f"unknown receiver presentation {presentation!r}")
            for rep in range(int(seeds)):
                made += builder.emit(
                    ArdCellKind.RECEIVER, programs, cohort=K.Cohort.RECEIVER,
                    checkpoint=checkpoint, reasoning_allowance=budget, episode_rep=rep,
                    instance=instance,
                    options={"presentation": presentation,
                             "checks_per_packet": K.RECEIVER_CHECKS_PER_PACKET,
                             "top_logprobs": K.RECEIVER_TOP_LOGPROBS},
                )
    return made


def build_e4_forks(
    builder: ArdBuilder,
    tasks: Sequence[ArdTask],
    *,
    checkpoint: str,
    state_sha: str,
    parent_rounds: Sequence[int] = K.FORK_PARENT_ROUNDS,
    parent_cells: Sequence[str] = (),
    instance: str | None = None,
) -> list[ArdCellSpec]:
    """E4 training data (spec §12.1): one private counterfactual branch per agent per state.

    ``state:<sha>`` is a hard dependency: the sealed PEER round-t states must exist and hash
    to ``sha`` before a single fork request is generated.
    """
    dep = state_dependency(state_sha)
    made: list[ArdCellSpec] = []
    for round_ in parent_rounds:
        made += builder.emit(
            ArdCellKind.FORK, tasks, cohort=K.Cohort.FORK, checkpoint=checkpoint,
            protocol=K.Protocol.PRIVATE, round_=int(round_), depends_on=(dep,),
            parent_ref=dep, instance=instance,
            options={"focal_agents": list(range(K.N_AGENTS)),
                     "parent_protocol": K.Protocol.PEER.value,
                     # The PEER cells holding the parent states.  A fork reads its parent back
                     # from the run rather than regenerating it: a fork whose parent is
                     # re-rolled is not a counterfactual of anything (defect D-10).
                     "parent_cells": list(parent_cells)},
        )
    return made


def build_e4_gate(
    builder: ArdBuilder,
    tasks: Sequence[ArdTask],
    *,
    checkpoint: str,
    controller_sha: str,
    protocols: Sequence[K.Protocol] = (K.Protocol.GATE, K.Protocol.RANDOM_PROTECT),
    instance: str | None = None,
) -> list[ArdCellSpec]:
    """E4 closed loop (spec §12.3): GATE and its matched RANDOM_PROTECT control.

    RANDOM_PROTECT needs no controller, but it is dispatched with the same instance tag so
    the paired arms are always one wave.
    """
    made: list[ArdCellSpec] = []
    for protocol in protocols:
        deps = KINDS.required_dependencies(protocol, controller_sha=controller_sha)
        made += builder.emit(
            ArdCellKind.EPISODE, tasks, cohort=K.Cohort.NATURAL, checkpoint=checkpoint,
            protocol=protocol, depends_on=deps,
            parent_ref=deps[0] if deps else controller_dependency(controller_sha),
            instance=instance,
        )
    return made


def build_e4_selector_audit(
    builder: ArdBuilder,
    tasks: Sequence[ArdTask],
    *,
    checkpoint: str,
    state_sha: str,
    pools: Sequence[str] = K.SELECTOR_AUDIT_POOLS,
    instance: str | None = None,
) -> list[ArdCellSpec]:
    """E4 static competitors (spec §12.4): the set-wise LLM audit selector over sealed pools."""
    unknown = [pool for pool in pools if pool not in K.SELECTOR_AUDIT_POOLS]
    if unknown:
        raise ArdManifestError(f"unknown selector-audit pools {unknown}")
    dep = state_dependency(state_sha)
    made: list[ArdCellSpec] = []
    for pool in pools:
        made += builder.emit(
            ArdCellKind.SELECTOR_AUDIT, tasks, cohort=K.Cohort.NATURAL, checkpoint=checkpoint,
            depends_on=(dep,), parent_ref=dep, instance=instance, options={"pool": pool},
        )
    return made


def build_e5_suffix(
    builder: ArdBuilder,
    tasks: Sequence[ArdTask],
    *,
    checkpoint: str,
    state_sha: str,
    parent_protocols: Sequence[K.Protocol] = K.SUFFIX_PARENT_PROTOCOLS,
    seeds: int = K.SUFFIX_SEEDS,
    instance: str | None = None,
) -> list[ArdCellSpec]:
    """E5 (spec §13): two independent private continuations off each sealed round-1 state."""
    dep = state_dependency(state_sha)
    made: list[ArdCellSpec] = []
    for parent in parent_protocols:
        for rep in range(int(seeds)):
            made += builder.emit(
                ArdCellKind.SUFFIX, tasks, cohort=K.Cohort.SUFFIX, checkpoint=checkpoint,
                protocol=K.Protocol.PRIVATE, round_=1, depends_on=(dep,), parent_ref=dep,
                episode_rep=rep, instance=instance,
                options={"parent_protocol": parent.value, "suffix_seed": rep},
            )
    return made


def build_e6_capture(
    builder: ArdBuilder,
    tasks: Sequence[ArdTask],
    *,
    decision_manifest_sha: str,
    protocols: Sequence[K.Protocol] = K.CAPTURE_PROTOCOLS,
    layers: Sequence[float] = K.CAPTURE_RELATIVE_LAYERS,
    instance: str | None = None,
) -> list[ArdCellSpec]:
    """E6 (spec §14.2): teacher-forced replay of the frozen decision work list, 8B only."""
    made: list[ArdCellSpec] = []
    for protocol in protocols:
        made += builder.emit(
            ArdCellKind.CAPTURE, tasks, cohort=K.Cohort.CAPTURE, checkpoint=K.CHECKPOINT_SMALL,
            protocol=protocol, parent_ref=f"decisions:{decision_manifest_sha}", instance=instance,
            options={"relative_layers": list(layers), "anchors": list(K.CAPTURE_ANCHORS),
                     "decision_manifest_sha256": decision_manifest_sha},
        )
    return made


def build_grade_cells(
    builder: ArdBuilder,
    tasks: Sequence[ArdTask],
    *,
    generation_cells: Sequence[str] = (),
    instance: str | None = None,
) -> list[ArdCellSpec]:
    """Evaluator lane: grade every committed answer of the named generation cells."""
    return builder.emit(
        ArdCellKind.GRADE, tasks, cohort=K.Cohort.NATURAL, checkpoint=K.CHECKPOINT_SMALL,
        depends_on=tuple(cell_dependency(cell_id) for cell_id in generation_cells),
        instance=instance, options={"cells": list(generation_cells)},
    )


def build_analysis_cells(
    builder: ArdBuilder,
    *,
    kind: ArdCellKind = ArdCellKind.COMPACT,
    depends_on: Sequence[str] = (),
    instance: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> list[ArdCellSpec]:
    """A single-item analysis cell (COMPACT or E0) on the ``analysis`` lane."""
    spec = KINDS.spec_for(kind)
    cell = ArdCellSpec(
        cell_id=cell_id_for(kind, cohort=K.Cohort.NATURAL, checkpoint=K.CHECKPOINT_SMALL,
                            shard=0, instance=instance, split="test"),
        kind=kind, protocol=None, round=None, cohort=K.Cohort.NATURAL,
        checkpoint=K.CHECKPOINT_SMALL, reasoning_allowance=K.THINKING_TOKENS_LOW,
        final_allowance=K.FINAL_TOKENS, items=(ANALYSIS_ITEM,), team_id=None, parent_ref=None,
        leader_seed=None, depends_on=tuple(depends_on), max_inflight=spec.max_inflight,
        parallel_items=spec.parallel_items, lane=spec.lane, split="test", instance=instance,
        options=dict(options or {}),
    )
    builder.add(cell)
    return [cell]


# --------------------------------------------------------------------------- files


def manifest_meta(
    *, module: str, lane: str | None, cells: Sequence[ArdCellSpec], **extra: Any
) -> dict[str, Any]:
    return {
        "study_id": K.STUDY_ID,
        "contract_version": K.CONTRACT_VERSION,
        "schema": "ard36-cells-v1",
        "module": module,
        "lane": lane,
        "n_cells": len(cells),
        "n_items": sum(len(cell.items) for cell in cells),
        **extra,
    }


def write_ard_cells_file(
    path: str | os.PathLike, cells: Sequence[ArdCellSpec], meta: Mapping[str, Any]
) -> str:
    """``study.cells.write_cells_file`` verbatim: frozen manifest + ``.sha256`` sidecar."""
    return write_cells_file(path, cells, meta)


def load_ard_cells_file(path: str | os.PathLike) -> list[ArdCellSpec]:
    """Strict load: verify the ``.sha256`` sidecar, then rebuild every :class:`ArdCellSpec`."""
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
    cells = [ArdCellSpec.from_dict(entry) for entry in data["cells"]]
    ids = [cell.cell_id for cell in cells]
    if len(set(ids)) != len(ids):
        raise ProtocolError(f"{path}: duplicate cell ids")
    return cells


# --------------------------------------------------------------------------- CLI


def build_module(
    module: str,
    tasks: Sequence[ArdTask],
    *,
    lane: str | None = None,
    checkpoint: str = K.CHECKPOINT_SMALL,
    controller_sha: str | None = None,
    state_sha: str | None = None,
    teams_sha: str | None = None,
    team_ids: Sequence[str] = (),
    instance: str | None = None,
) -> list[ArdCellSpec]:
    """One module's cells, in dispatch order (``MODULES``)."""
    builder = ArdBuilder(lane)
    if module == "E1":
        build_e1_roots(builder, tasks, checkpoint=checkpoint, instance=instance)
    elif module in ("E2a", "E2b"):
        build_e2_natural(builder, tasks, checkpoint=checkpoint, controller_sha=controller_sha,
                         instance=instance)
    elif module == "E2c":
        if not teams_sha or not team_ids:
            raise ArdManifestError("E2c needs --teams-sha and --team-id (opaque tags)")
        build_e2c_challenge(builder, tasks, checkpoint=checkpoint, team_ids=team_ids,
                            teams_manifest_sha=teams_sha, controller_sha=controller_sha,
                            instance=instance)
    elif module == "E2d":
        build_e2d_high_budget(builder, tasks, checkpoint=checkpoint, instance=instance)
    elif module == "E3":
        build_e3_receiver(builder, tasks, checkpoint=checkpoint, instance=instance)
    elif module == "E4":
        if not state_sha:
            raise ArdManifestError("E4 forks need --state-sha (the sealed PEER state register)")
        build_e4_forks(builder, tasks, checkpoint=checkpoint, state_sha=state_sha, instance=instance)
        if controller_sha:
            build_e4_gate(builder, tasks, checkpoint=checkpoint, controller_sha=controller_sha,
                          instance=instance)
            build_e4_selector_audit(builder, tasks, checkpoint=checkpoint, state_sha=state_sha,
                                    instance=instance)
    elif module == "E5":
        if not state_sha:
            raise ArdManifestError("E5 suffixes need --state-sha (the sealed round-1 register)")
        build_e5_suffix(builder, tasks, checkpoint=checkpoint, state_sha=state_sha, instance=instance)
    elif module == "E6":
        build_e6_capture(builder, tasks, decision_manifest_sha=state_sha or "0" * 64,
                         instance=instance)
    elif module == "OPS":
        build_grade_cells(builder, tasks, instance=instance)
        build_analysis_cells(builder, kind=ArdCellKind.COMPACT, instance=instance)
    else:
        raise ArdManifestError(f"unknown module {module!r}; expected one of {MODULES}")
    return builder.cells


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.ard.cells",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--run-root", required=True, help="ARD36 run root (holds data/public/items.parquet)")
    p.add_argument("--module", required=True, choices=MODULES)
    p.add_argument("--lane", default=None, choices=list(K.LANES))
    p.add_argument("--checkpoint", default=K.CHECKPOINT_SMALL,
                   choices=[K.CHECKPOINT_SMALL, K.CHECKPOINT_LARGE])
    p.add_argument("--out", required=True, help="file name under the run root, e.g. cells_E2a_8B.json")
    p.add_argument("--split", action="append", default=[], choices=list(K.SPLITS))
    p.add_argument("--controller-sha", default=None)
    p.add_argument("--state-sha", default=None)
    p.add_argument("--teams-sha", default=None)
    p.add_argument("--team-id", action="append", default=[])
    p.add_argument("--instance", default=None, help="wave tag; keeps two waves in separate directories")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.run_root)
    all_tasks = load_ard_tasks(run_root)
    if args.checkpoint == K.CHECKPOINT_LARGE:
        # The 32B arms run the frozen 96-item panel, never the whole natural cohort: the
        # large lane is ~20x the cost per decision and its cohort is fixed in the spec.
        all_tasks = large_panel(all_tasks)
    tasks = list(all_tasks.values())
    if args.split:
        tasks = [task for task in tasks if task.split in set(args.split)]
    cells = build_module(
        args.module, tasks, lane=args.lane, checkpoint=args.checkpoint,
        controller_sha=args.controller_sha, state_sha=args.state_sha, teams_sha=args.teams_sha,
        team_ids=args.team_id, instance=args.instance,
    )
    meta = manifest_meta(module=args.module, lane=args.lane, cells=cells,
                         checkpoint=args.checkpoint, instance=args.instance)
    out = Path(args.out) if Path(args.out).is_absolute() else run_root / args.out
    if args.dry_run:
        print(json.dumps({"out": str(out), "meta": meta,
                          "cells": [cell.cell_id for cell in cells]}, indent=2))
        return 0
    digest = write_ard_cells_file(out, cells, meta)
    print(json.dumps({"cells_file": str(out), "sha256": digest, **meta}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "ArdBuilder", "MODULES", "SPLIT_ORDER", "assert_opaque_team_id", "build_analysis_cells",
    "build_e1_roots", "build_e2_natural", "build_e2c_challenge", "build_e2d_high_budget",
    "build_e3_receiver", "build_e4_forks", "build_e4_gate", "build_e4_selector_audit",
    "build_e5_suffix", "build_e6_capture", "build_grade_cells", "build_module", "cell_id_for",
    "dispatch_order", "interleave_families", "item_shards", "load_ard_cells_file", "split_groups",
    "manifest_meta", "parent_tag", "write_ard_cells_file",
]
