"""ARD36 shards → tables adapter: the E6 capture store becomes ``neural_geometry`` /
``neural_agent`` (WP6).

Why this module exists
----------------------
:mod:`agents_scaling.ard.neural.geometry` computes every §14.3 statistic from an
in-memory ``s x d`` matrix, and :mod:`agents_scaling.ard.neural.capture` writes fp16
vectors plus an ActivationRow ledger into ``<run_root>/neural/native/*.{npz,jsonl}``.
Nothing joined the two: ``ard.compact`` hardcoded both neural tables to ``[]``, so a
completed capture would have produced shards that no table could ever reach.  This module
is that missing adapter, and nothing else — it reads the store, regroups the ledger into
comparable teams, and calls the existing estimators.

The question it serves
----------------------
Descriptive geometry over the capture-eligible cohort (8B, ``natural``, PRIVATE/PEER/
BALANCED, rounds 0-3): participation ratio, effective rank, and displacement/alignment by
round, protocol and team state.  It is **not** a probe and makes no prediction claim, so
every metric here is a description of stored states.  Nothing is computed that the shards
do not carry: the ledger's own ``missingness`` codes decide which decisions have a state,
and a decision without a vector gets a row with NaN geometry rather than a zero vector.

Three things the on-disk format forces
--------------------------------------
1. **A shard row is keyed by decision, not by episode.**  ``capture.work_list`` collapses
   duplicate ``decision_id``s, and round-0 root decisions are *shared* by the three
   protocol episodes of one item (measured on ``tables_v2/decisions.parquet``: 23,040
   eligible decision rows, 19,200 distinct ``decision_id``s; the 3,840 difference is
   exactly 384 items × 5 agents × 2 extra protocol references).  A shard row therefore
   names only **one** of the three episodes.  Passing ``decisions`` (the ``decisions``
   rows, which carry ``cell_id`` and ``item_id`` in ``tables_v2``) fans each captured
   vector back out over every episode that referenced it — without that fan-out two of
   the three protocols would have no round-0 team at all, and with it the round-1
   displacement baseline exists for all three.
2. **Grouping is by episode, not by item × protocol.**  ``ard.compact``'s own keying rule
   (module docstring) is that item × protocol × checkpoint collides for the two challenge
   teams of one item; ``round_summary`` and ``minority_frame`` are keyed by
   ``episode_id × round``.  Team rows here use ``(checkpoint, cell_id, episode_id, round,
   layer, anchor)`` so the join to those tables is exact, and keep ``item_id``/``protocol``
   as columns.
3. **The previous-round state is looked up inside the episode.**  Displacement and
   alignment need the same five agents at ``t-1``; the key is the group key with
   ``round - 1``, and the agent ids must match, otherwise the row says
   ``has_previous_state = False`` and the two columns are NaN.

Anchors.  The default is :data:`agents_scaling.ard.contract.PROSPECTIVE_ANCHORS`
(``PROMPT_END``, ``PRE_FINAL``) — the two the analysis brief names.  ``ANSWER_END`` and
``REPORT_END`` can be widened in with ``--anchor``; they sit after the answer content and
must never be framed as prediction.

Cost, measured.  One pass holds every selected fp16 vector:
``n_decisions × n_layers × n_anchors × hidden × 2`` bytes (:func:`load_estimate_bytes`).
On a synthetic store at the **real** shape — 19,200 distinct eligible decisions, hidden
4096, layer 17 × ``PROMPT_END`` only — the store was 192 MiB on disk and this module took
6 s at 0.86 GiB peak RSS (which includes the ``decisions`` frame).  The full 3 layers ×
2 anchors is ~6× that: ~1.2 GiB of shards, ~40 s, ~3 GiB resident.  ``--layer`` /
``--anchor`` are the knob if that is too much for the analysis node.

Owned by WP6 (``docs/ard36/00_contract.md`` §1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Iterator, Mapping, Sequence

import numpy as np

from agents_scaling.ard import contract as K
from agents_scaling.ard.neural import anchors as AA
from agents_scaling.ard.neural import geometry as G
from agents_scaling.ard.neural.capture import CAPTURE_STAGE
from agents_scaling.study.neural import storage as S

__all__ = [
    "AGENT_EXTRA_COLUMNS",
    "AgentState",
    "CaptureSite",
    "GEOMETRY_EXTRA_COLUMNS",
    "MIN_TEAM_STATES",
    "TableError",
    "agent_table_columns",
    "decision_sites",
    "geometry_table_columns",
    "group_states",
    "iter_capture_rows",
    "load_estimate_bytes",
    "main",
    "neural_tables_from_shards",
    "shard_coverage",
]

#: A spectrum needs at least two states to have a centered covariance at all
#: (``study.neural.geometry.MIN_S_FOR_SPECTRUM = 3`` is the stricter bar for the
#: participation ratio / effective rank, which come back NaN with ``spectrum_testable``
#: False below it).  A team row is emitted from two present states upward; the row says
#: how many it had.
MIN_TEAM_STATES: Final[int] = 2

#: The adapter's half of the two schemas lives in :mod:`agents_scaling.ard.neural.geometry`
#: beside the WP4-frozen prefix, so there is exactly one declaration of each table and
#: ``compact.SCHEMAS`` can be asserted equal to it.
GEOMETRY_EXTRA_COLUMNS: Final[tuple[str, ...]] = G.GEOMETRY_JOIN_COLUMNS
AGENT_EXTRA_COLUMNS: Final[tuple[str, ...]] = G.AGENT_JOIN_COLUMNS


class TableError(RuntimeError):
    """The store cannot be turned into a table (never silently dropped rows)."""


def geometry_table_columns() -> tuple[str, ...]:
    """The full declared ``neural_geometry`` schema: frozen prefix, then the join tail."""
    return tuple(G.GEOMETRY_COLUMNS)


def agent_table_columns() -> tuple[str, ...]:
    """The full declared ``neural_agent`` schema: frozen prefix, then the join tail."""
    return tuple(G.AGENT_COLUMNS)


def load_estimate_bytes(
    n_decisions: int, *, n_layers: int, n_anchors: int, hidden_size: int = 4096
) -> int:
    """Resident fp16 bytes of one :func:`neural_tables_from_shards` pass (see the docstring)."""
    return int(n_decisions) * int(n_layers) * int(n_anchors) * int(hidden_size) * 2


# --------------------------------------------------------------------------- the store


def iter_capture_rows(
    run_root: str | os.PathLike,
    *,
    stage: str = CAPTURE_STAGE,
    shard: str | None = None,
    blocks: Sequence[int] | None = None,
    anchors: Sequence[str] | None = None,
) -> Iterator[tuple[dict[str, Any], np.ndarray | None]]:
    """``(ledger row, vector | None)`` for every committed row of a stage.

    Straight over ``storage.list_chunks`` / ``read_chunk`` — the same completeness rule the
    writer commits under (a chunk counts only once its ``.jsonl`` exists and agrees with
    its ``.npz``).  Missing anchors come back with ``None``, never a zero vector.  A
    duplicate ``(snapshot, block, anchor)`` across chunks raises: the store is append-only
    by key, and a silently-deduplicated team would change a spectrum.
    """
    directory = S.stage_dir(run_root, stage)
    want_blocks = None if blocks is None else {int(b) for b in blocks}
    want_anchors = None if anchors is None else set(anchors)
    seen: set[str] = set()
    for _shard, _index, npz, jsonl in S.list_chunks(directory, shard):
        vectors, _keys, rows = S.read_chunk(npz, jsonl)
        for row in rows:
            if want_blocks is not None and int(row["block"]) not in want_blocks:
                continue
            if want_anchors is not None and str(row["anchor_kind"]) not in want_anchors:
                continue
            key = S.key_str(row["StateSnapshot_id"], row["block"], row["anchor_kind"])
            if key in seen:
                raise TableError(f"duplicate key {key} across chunks of {directory}")
            seen.add(key)
            index = int(row.get("vector_index", -1))
            yield row, (np.asarray(vectors[index]) if index >= 0 else None)


# --------------------------------------------------------------------------- fan-out


@dataclass(frozen=True)
class CaptureSite:
    """One (decision, episode) reference: where a captured vector belongs in the panel.

    A root decision is referenced by the three protocol episodes of its item, so one
    ``decision_id`` can have three sites (see the module docstring).
    """

    decision_id: str
    cell_id: str
    episode_id: str
    item_id: str
    protocol: str
    round_: int
    agent_id: int
    checkpoint: str = K.CHECKPOINT_SMALL
    cohort: str = K.Cohort.NATURAL.value
    split: str = ""
    family: str = ""
    boundary_injected: bool = False


def decision_sites(
    rows: Iterable[Mapping[str, Any]],
    *,
    episodes: Iterable[Mapping[str, Any]] | None = None,
    checkpoints: Sequence[str] | None = (K.CHECKPOINT_SMALL,),
    cohorts: Sequence[str] | None = (K.Cohort.NATURAL.value,),
    protocols: Sequence[str] | None = tuple(p.value for p in K.CAPTURE_PROTOCOLS),
    rounds: Sequence[int] | None = K.ROUNDS,
) -> dict[str, tuple[CaptureSite, ...]]:
    """``{decision_id: every episode that references it}`` from ``decisions`` rows.

    The default filter is the capture-eligible cohort of the analysis brief (8B, natural,
    the three captured protocols, rounds 0-3).  Pass ``None`` for any axis to keep it whole.
    A row without ``cell_id``/``item_id`` still yields a site with those fields empty —
    ``tables_v2/decisions.parquet`` carries both, the frozen v1 table did not.  ``split``
    and ``family`` are **not** decision columns at all (measured: ``decisions.parquet`` has
    50 columns and neither is among them), so pass ``episodes`` rows to fill them by
    ``episode_id``; without it both come back empty rather than wrong.
    """
    by_episode: dict[str, Mapping[str, Any]] = {}
    if episodes is not None:
        by_episode = {str(e.get("episode_id") or ""): e for e in episodes}
    want_ck = None if checkpoints is None else {str(c) for c in checkpoints}
    want_co = None if cohorts is None else {str(c) for c in cohorts}
    want_pr = None if protocols is None else {str(p) for p in protocols}
    want_rd = None if rounds is None else {int(r) for r in rounds}
    out: dict[str, list[CaptureSite]] = defaultdict(list)
    for row in rows:
        checkpoint = str(row.get("checkpoint") or K.CHECKPOINT_SMALL)
        cohort = str(row.get("cohort") or "")
        protocol = str(row.get("protocol") or "")
        round_ = int(row.get("round") or 0)
        if want_ck is not None and checkpoint not in want_ck:
            continue
        if want_co is not None and cohort not in want_co:
            continue
        if want_pr is not None and protocol not in want_pr:
            continue
        if want_rd is not None and round_ not in want_rd:
            continue
        decision_id = str(row.get("decision_id") or "")
        if not decision_id:
            continue
        episode = by_episode.get(str(row.get("episode_id") or ""), {})
        site = CaptureSite(
            decision_id=decision_id,
            cell_id=str(row.get("cell_id") or ""),
            episode_id=str(row.get("episode_id") or ""),
            item_id=str(row.get("item_id") or ""),
            protocol=protocol,
            round_=round_,
            agent_id=int(row.get("agent_id") or 0),
            checkpoint=checkpoint,
            cohort=cohort,
            split=str(row.get("split") or episode.get("split") or ""),
            family=str(row.get("family") or episode.get("family") or ""),
            boundary_injected=bool(row.get("boundary_injected") or False),
        )
        if site not in out[decision_id]:
            out[decision_id].append(site)
    return {k: tuple(v) for k, v in out.items()}


def _site_from_row(row: Mapping[str, Any]) -> CaptureSite:
    """The single site a shard row names by itself (no ``decisions`` join available)."""
    extra = dict(row.get("extra") or {})
    return CaptureSite(
        decision_id=str(row["StateSnapshot_id"]),
        cell_id=str(row.get("cell_id") or extra.get("cell_id") or ""),
        episode_id=str(row.get("episode_id") or extra.get("episode_id") or ""),
        item_id=str(row.get("source_id") or extra.get("item_id") or ""),
        protocol=str(row.get("method") or extra.get("protocol") or ""),
        round_=int(extra.get("round", 0)),
        agent_id=int(extra.get("agent_id", 0)),
        checkpoint=str(row.get("checkpoint") or K.CHECKPOINT_SMALL),
        cohort=str(extra.get("cohort") or K.Cohort.CAPTURE.value),
        split=str(extra.get("split") or ""),
        family=str(extra.get("family") or ""),
        boundary_injected=bool(extra.get("boundary_injected") or False),
    )


# --------------------------------------------------------------------------- grouping


@dataclass(frozen=True)
class AgentState:
    """One captured (decision, layer, anchor) state at one site, present or not."""

    site: CaptureSite
    layer: int
    anchor: str
    vector: np.ndarray | None
    missingness: str | None
    ledger: Mapping[str, Any] = field(default_factory=dict)

    @property
    def present(self) -> bool:
        return self.vector is not None

    @property
    def extra(self) -> Mapping[str, Any]:
        return dict(self.ledger.get("extra") or {})


def _group_key(state: AgentState) -> tuple[str, str, str, int, int, str]:
    site = state.site
    return (site.checkpoint, site.cell_id, site.episode_id, int(site.round_),
            int(state.layer), state.anchor)


def _previous_key(key: tuple[str, str, str, int, int, str]) -> tuple[str, str, str, int, int, str]:
    checkpoint, cell_id, episode_id, round_, layer, anchor = key
    return (checkpoint, cell_id, episode_id, int(round_) - 1, layer, anchor)


def group_states(
    run_root: str | os.PathLike,
    *,
    decisions: Iterable[Mapping[str, Any]] | None = None,
    episodes: Iterable[Mapping[str, Any]] | None = None,
    stage: str = CAPTURE_STAGE,
    shard: str | None = None,
    blocks: Sequence[int] | None = None,
    anchors: Sequence[str] | None = None,
) -> tuple[dict[tuple[str, str, str, int, int, str], list[AgentState]], dict[str, Any]]:
    """``({group key: states sorted by agent_id}, coverage stats)``.

    With ``decisions`` given, every ledger row is fanned out over each episode that
    referenced its decision and rows whose decision is outside the eligible cohort are
    dropped (counted in the stats as ``rows_unreferenced``).  Without it, each row lands at
    the single site it names itself.
    """
    sites = None if decisions is None else decision_sites(decisions, episodes=episodes)
    groups: dict[tuple[str, str, str, int, int, str], list[AgentState]] = defaultdict(list)
    stats: dict[str, Any] = {
        "rows": 0, "present": 0, "missing": 0, "rows_unreferenced": 0, "states": 0,
        "missing_codes": {}, "anchors": {}, "layers": {},
    }
    for row, vector in iter_capture_rows(run_root, stage=stage, shard=shard, blocks=blocks,
                                         anchors=anchors):
        stats["rows"] += 1
        code = row.get("missingness")
        if vector is None:
            stats["missing"] += 1
            stats["missing_codes"][str(code)] = stats["missing_codes"].get(str(code), 0) + 1
        else:
            stats["present"] += 1
        anchor = str(row["anchor_kind"])
        layer = int(row["block"])
        stats["anchors"][anchor] = stats["anchors"].get(anchor, 0) + 1
        stats["layers"][str(layer)] = stats["layers"].get(str(layer), 0) + 1
        if sites is None:
            row_sites: tuple[CaptureSite, ...] = (_site_from_row(row),)
        else:
            row_sites = sites.get(str(row["StateSnapshot_id"]), ())
            if not row_sites:
                stats["rows_unreferenced"] += 1
                continue
        for site in row_sites:
            state = AgentState(site=site, layer=layer, anchor=anchor, vector=vector,
                               missingness=None if vector is not None else (
                                   str(code) if code is not None else "UNRECORDED"),
                               ledger=row)
            groups[_group_key(state)].append(state)
            stats["states"] += 1
    for key, states in groups.items():
        states.sort(key=lambda s: int(s.site.agent_id))
        ids = [int(s.site.agent_id) for s in states]
        if len(set(ids)) != len(ids):
            raise TableError(f"group {key} has duplicate agent ids {ids}")
    stats["groups"] = len(groups)
    return dict(groups), stats


# --------------------------------------------------------------------------- rows


def _identity(key: tuple[str, str, str, int, int, str], states: Sequence[AgentState]) -> dict[str, Any]:
    checkpoint, cell_id, episode_id, round_, layer, anchor = key
    first = states[0].site
    return {
        "item_id": first.item_id,
        "protocol": first.protocol,
        "round": int(round_),
        "layer": int(layer),
        "anchor": anchor,
        "cell_id": cell_id,
        "episode_id": episode_id,
        "checkpoint": checkpoint,
        "cohort": first.cohort,
        "split": first.split,
        "family": first.family,
        "feature_time": AA.feature_time_of(anchor),
        "last_allowed_round": AA.last_allowed_round(anchor, int(round_)),
    }


def _entropy_of(state: AgentState) -> dict[str, Any]:
    """The scalar entropy block the capture wrote at ``contract.ENTROPY_ANCHORS`` only."""
    extra = state.extra
    return {
        "entropy_anchor": bool(extra.get("entropy_anchor", AA.is_entropy_anchor(state.anchor))),
        "entropy_nats": _float_or_nan(extra.get("entropy_nats")),
        "entropy_bits": _float_or_nan(extra.get("entropy_bits")),
        "top1_prob": _float_or_nan(extra.get("top1_prob")),
        "logit_margin": _float_or_nan(extra.get("logit_margin")),
    }


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean_or_nan(values: Sequence[float]) -> float:
    finite = [v for v in values if v == v]
    return float(np.mean(finite)) if finite else float("nan")


def _matrix(states: Sequence[AgentState]) -> np.ndarray:
    """fp16 shard vectors → one float64 ``s x d`` matrix (the estimators are float64)."""
    return np.stack([np.asarray(s.vector, dtype=np.float64).reshape(-1) for s in states])


def neural_tables_from_shards(
    run_root: str | os.PathLike,
    *,
    decisions: Iterable[Mapping[str, Any]] | None = None,
    episodes: Iterable[Mapping[str, Any]] | None = None,
    stage: str = CAPTURE_STAGE,
    shard: str | None = None,
    blocks: Sequence[int] | None = None,
    anchors: Sequence[str] | None = tuple(sorted(K.PROSPECTIVE_ANCHORS)),
    min_team_states: int = MIN_TEAM_STATES,
) -> dict[str, list[dict[str, Any]]]:
    """``{"neural_geometry": [...], "neural_agent": [...], "coverage": {...}}``.

    Team rows are emitted twice (raw and unit-normalised, spec §14.3 "raw and normalized
    metrics must be shown together").  Per-agent geometry is computed on the **raw** states
    only — ``neural_agent`` has no ``normalised`` axis — so the team-level displacement
    means ride on the raw row and are NaN on the normalised one, rather than being silently
    reused across representations.

    Every site gets an agent row, including sites whose anchor was never reached: those
    carry ``present = False``, the ledger's ``missingness`` code and NaN geometry, so the
    denominator of any coverage statement is in the table itself.
    """
    groups, coverage = group_states(run_root, decisions=decisions, episodes=episodes,
                                    stage=stage, shard=shard, blocks=blocks, anchors=anchors)
    geometry_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    coverage.update({"team_rows": 0, "agent_rows": 0, "groups_below_min_states": 0,
                     "groups_with_previous": 0})

    for key in sorted(groups):
        states = groups[key]
        identity = _identity(key, states)
        present = [s for s in states if s.present]
        missing = [s for s in states if not s.present]
        ledger = states[0].ledger
        provenance = {
            "hidden_size": int(ledger.get("hidden_size") or 0) or None,
            "consumer_revision": ledger.get("consumer_revision"),
            "hook_convention": ledger.get("hook_convention"),
        }

        # ---- the previous round of the same episode, same agents -----------------------
        previous = groups.get(_previous_key(key))
        previous_present = [s for s in (previous or []) if s.present]
        agent_ids = tuple(int(s.site.agent_id) for s in present)
        prev_ids = tuple(int(s.site.agent_id) for s in previous_present)
        has_previous = bool(previous_present) and prev_ids == agent_ids and len(agent_ids) > 0
        previous_by_agent = {int(s.site.agent_id): s for s in previous_present} if has_previous else {}
        if has_previous:
            coverage["groups_with_previous"] += 1

        # ---- per-agent geometry (raw states only) --------------------------------------
        metrics_by_agent: dict[int, dict[str, Any]] = {}
        if len(present) >= MIN_TEAM_STATES:
            X = _matrix(present)
            previous_X = _matrix([previous_by_agent[a] for a in agent_ids]) if has_previous else None
            for row in G.agent_metrics(X, previous=previous_X, agent_ids=list(agent_ids)):
                metrics_by_agent[int(row["agent_id"])] = row

        for state in states:
            agent_id = int(state.site.agent_id)
            metrics = metrics_by_agent.get(agent_id, {})
            previous_state = previous_by_agent.get(agent_id)
            agent_rows.append({
                **identity,
                "agent_id": agent_id,
                "decision_id": state.site.decision_id,
                "boundary_injected": bool(state.site.boundary_injected),
                "present": bool(state.present),
                "missingness": state.missingness,
                "state_norm": _float_or_nan(metrics.get("state_norm")),
                "distance_to_peer_centroid": _float_or_nan(metrics.get("distance_to_peer_centroid")),
                "distance_to_nearest_peer": _float_or_nan(metrics.get("distance_to_nearest_peer")),
                "own_displacement": _float_or_nan(metrics.get("own_displacement")),
                "displacement_alignment": _float_or_nan(metrics.get("displacement_alignment")),
                "near_zero_denominator": bool(metrics.get("near_zero_denominator", False)),
                "nearest_peer_agent": metrics.get("nearest_peer_agent"),
                "has_previous_state": bool(metrics.get("has_previous_state", False)),
                "previous_decision_id": None if previous_state is None else previous_state.site.decision_id,
                "displacement_denominator": _float_or_nan(metrics.get("displacement_denominator")),
                "direction_denominator": _float_or_nan(metrics.get("direction_denominator")),
                "n_team_states": len(present),
                "token_offset": state.ledger.get("token_offset"),
                "channel": state.ledger.get("channel"),
                "generated_token_count": state.ledger.get("generated_token_count"),
                "prompt_tokens": state.extra.get("prompt_tokens"),
                "completion_tokens": state.extra.get("completion_tokens"),
                **_entropy_of(state),
                **provenance,
            })
        coverage["agent_rows"] += len(states)

        # ---- team geometry -------------------------------------------------------------
        if len(present) < int(min_team_states):
            coverage["groups_below_min_states"] += 1
            continue
        X = _matrix(present)
        entropies = [_float_or_nan(s.extra.get("entropy_bits")) for s in present]
        displacements = [_float_or_nan(metrics_by_agent[a].get("own_displacement")) for a in agent_ids]
        alignments = [_float_or_nan(metrics_by_agent[a].get("displacement_alignment")) for a in agent_ids]
        team_tail = {
            "agent_ids": ",".join(str(a) for a in agent_ids),
            "n_states": len(present),
            "n_sites": len(states),
            "team_complete": len(present) == K.N_AGENTS,
            "n_missing_states": len(missing),
            "missing_codes": ",".join(sorted({str(s.missingness) for s in missing})) or None,
            "entropy_anchor": AA.is_entropy_anchor(identity["anchor"]),
            "entropy_bits_mean": _mean_or_nan(entropies),
            "n_entropy_states": sum(1 for v in entropies if v == v),
            "has_previous_state": has_previous,
            **provenance,
        }
        for normalised in (False, True):
            metrics = G.team_metrics(X, normalised=normalised)
            geometry_rows.append({
                **identity,
                "participation_ratio": metrics["participation_ratio"],
                "entropy_rank": metrics["entropy_rank"],
                "leading_eigenvalue_share": metrics["leading_eigenvalue_share"],
                "total_variance": metrics["total_variance"],
                "mean_pairwise": metrics["mean_pairwise"],
                "min_pairwise": metrics["min_pairwise"],
                "max_pairwise": metrics["max_pairwise"],
                "cosine_distance": metrics["cosine_distance"],
                "normalised": bool(normalised),
                "rank_ceiling": metrics["rank_ceiling"],
                "n_nonzero_eigenvalues": metrics["n_nonzero_eigenvalues"],
                "spectrum_testable": metrics["spectrum_testable"],
                "representation": metrics["representation"],
                "state_norm_mean": metrics["state_norm_mean"],
                "state_norm_std": metrics["state_norm_std"],
                **team_tail,
                # raw-only: agent geometry has no normalised axis (see the docstring)
                "mean_own_displacement": float("nan") if normalised else _mean_or_nan(displacements),
                "mean_displacement_alignment": float("nan") if normalised else _mean_or_nan(alignments),
                "n_displacement_states": 0 if normalised else sum(1 for v in displacements if v == v),
            })
            coverage["team_rows"] += 1

    return {"neural_geometry": geometry_rows, "neural_agent": agent_rows, "coverage": coverage}


def shard_coverage(
    run_root: str | os.PathLike, *, stage: str = CAPTURE_STAGE, anchors: Sequence[str] | None = None
) -> dict[str, Any]:
    """Ledger-only coverage of a capture store (no vectors are grouped)."""
    _groups, stats = group_states(run_root, stage=stage, anchors=anchors)
    return stats


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.ard.neural.tables",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--run-root", required=True)
    p.add_argument("--decisions", default=None,
                   help="decisions.parquet (tables_v2) — enables the root fan-out across episodes")
    p.add_argument("--episodes", default=None,
                   help="episodes.parquet — the only source of split/family (decisions has neither)")
    p.add_argument("--stage", default=CAPTURE_STAGE)
    p.add_argument("--shard", default=None)
    p.add_argument("--layer", type=int, action="append", default=None,
                   help="restrict to these blocks (repeatable); default: every captured block")
    p.add_argument("--anchor", action="append", default=None,
                   help=f"default: {sorted(K.PROSPECTIVE_ANCHORS)}")
    p.add_argument("--out-dir", default=None,
                   help="write neural_geometry.parquet / neural_agent.parquet here "
                        "(omit to only print the counts; never defaults to a frozen tables dir)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decisions = episodes = None
    if args.decisions or args.episodes:
        import pandas as pd

        if args.decisions:
            decisions = pd.read_parquet(args.decisions).to_dict(orient="records")
        if args.episodes:
            episodes = pd.read_parquet(args.episodes).to_dict(orient="records")
    tables = neural_tables_from_shards(
        args.run_root, decisions=decisions, episodes=episodes, stage=args.stage, shard=args.shard,
        blocks=args.layer,
        anchors=tuple(args.anchor) if args.anchor else tuple(sorted(K.PROSPECTIVE_ANCHORS)),
    )
    summary = {
        "neural_geometry": len(tables["neural_geometry"]),
        "neural_agent": len(tables["neural_agent"]),
        "coverage": tables["coverage"],
    }
    if args.out_dir:
        from agents_scaling.study.aggregate import _write_parquet

        # Resolve against the run root unless absolute, matching every other ard CLI:
        # a bare "tables_v2" otherwise lands in the caller's cwd, not beside the tables
        # it is meant to join.
        out = Path(args.out_dir)
        if not out.is_absolute():
            out = Path(args.run_root) / out
        out.mkdir(parents=True, exist_ok=True)
        for name in ("neural_geometry", "neural_agent"):
            _write_parquet(out / f"{name}.parquet", tables[name])
        summary["out_dir"] = str(out)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
