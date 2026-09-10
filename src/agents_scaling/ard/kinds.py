"""What each ARD36 cell kind needs: worker, endpoint, Slurm shape, lane, dependencies (WP4).

Spec §10 (E2 modules), §12.3 (GATE runs only behind a frozen controller), §13 (E5 reads
sealed round-1 states), §16.2 (10-GPU allocation), §16.5 (schedule).  Plan §5.3 (cell
kinds), §6 (lane sizing: generation cells are 1 CPU / 4 G / 1-day HTTP clients on
``mit_preemptable``; ``grade`` and ``analysis`` are their own loops; ``capture`` is GPU),
§7 (dispatch priority key).

The table is the single place a dispatcher asks "does this kind need a live endpoint, which
lane does it belong to, what does it depend on, and how many items fit in one cell".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from agents_scaling.ard import contract as K
from agents_scaling.ard.types import (
    CONTROLLER_DEP_PREFIX,
    STATE_DEP_PREFIX,
    ArdCellKind,
    ArdManifestError,
)

#: Generation lanes are named by the checkpoint they serve (contract ``LANES``).
LANE_SMALL: Final[str] = K.CHECKPOINT_SMALL
LANE_LARGE: Final[str] = K.CHECKPOINT_LARGE
LANE_GRADE: Final[str] = "grade"
LANE_ANALYSIS: Final[str] = "analysis"
LANE_CAPTURE: Final[str] = "capture"


@dataclass(frozen=True)
class KindSpec:
    """Static properties of one cell kind.

    ``worker`` is ``"<module>:<callable>"``; :func:`agents_scaling.ard.runner.resolve_worker`
    imports it at dispatch time, so a kind owned by a work package that has not landed yet
    fails loudly on its own cell instead of breaking every import of the runner.
    """

    kind: ArdCellKind
    worker: str
    needs_endpoint: bool
    lane: str
    cpus: int
    mem: str
    time: str
    items_per_cell: int
    parallel_items: int
    max_inflight: int
    dep_prefixes: tuple[str, ...] = ()
    owner: str = "WP4"
    doc: str = ""

    def lane_for(self, checkpoint: str) -> str:
        """Generation kinds follow their checkpoint; CPU/GPU-replay kinds are fixed."""
        if self.lane in (LANE_SMALL, LANE_LARGE):
            if checkpoint not in (K.CHECKPOINT_SMALL, K.CHECKPOINT_LARGE):
                raise ArdManifestError(f"unknown checkpoint {checkpoint!r}")
            return checkpoint
        return self.lane


#: One entry per :class:`ArdCellKind`.  Sizes are the plan §6 per-kind cell sizes; the
#: checkpoint- and budget-dependent adjustment is :func:`items_per_cell`.
KIND_TABLE: Final[Mapping[ArdCellKind, KindSpec]] = {
    ArdCellKind.ROOTS: KindSpec(
        kind=ArdCellKind.ROOTS, worker="agents_scaling.ard.runner:episode_worker",
        needs_endpoint=True, lane=LANE_SMALL, cpus=1, mem="4G", time="1-00:00:00",
        items_per_cell=8, parallel_items=2, max_inflight=8,
        doc="16 stateless bank draws per item (spec §5.2); draws 0-4 are the common team.",
    ),
    ArdCellKind.EPISODE: KindSpec(
        kind=ArdCellKind.EPISODE, worker="agents_scaling.ard.runner:episode_worker",
        needs_endpoint=True, lane=LANE_SMALL, cpus=1, mem="4G", time="1-00:00:00",
        items_per_cell=8, parallel_items=1, max_inflight=K.N_AGENTS,
        dep_prefixes=(CONTROLLER_DEP_PREFIX,),
        doc="3 update rounds of one protocol (spec §7); GATE/CONF carry controller:<sha>.",
    ),
    ArdCellKind.CHALLENGE: KindSpec(
        kind=ArdCellKind.CHALLENGE, worker="agents_scaling.ard.runner:episode_worker",
        needs_endpoint=True, lane=LANE_SMALL, cpus=1, mem="4G", time="1-00:00:00",
        items_per_cell=4, parallel_items=1, max_inflight=K.N_AGENTS,
        dep_prefixes=(CONTROLLER_DEP_PREFIX,),
        doc="Constructed-team continuations; the manifest lists root candidate ids only (§5.3).",
    ),
    ArdCellKind.FORK: KindSpec(
        kind=ArdCellKind.FORK, worker="agents_scaling.ard.runner:episode_worker",
        needs_endpoint=True, lane=LANE_SMALL, cpus=1, mem="4G", time="1-00:00:00",
        items_per_cell=8, parallel_items=1, max_inflight=K.N_AGENTS,
        dep_prefixes=(STATE_DEP_PREFIX,),
        doc="One private counterfactual branch per agent off a sealed PEER state (spec §12.1).",
    ),
    ArdCellKind.SUFFIX: KindSpec(
        kind=ArdCellKind.SUFFIX, worker="agents_scaling.ard.runner:episode_worker",
        needs_endpoint=True, lane=LANE_SMALL, cpus=1, mem="4G", time="1-00:00:00",
        items_per_cell=4, parallel_items=1, max_inflight=K.N_AGENTS,
        dep_prefixes=(STATE_DEP_PREFIX,),
        doc="Two common private continuations off a sealed round-1 state (spec §13).",
    ),
    ArdCellKind.RECEIVER: KindSpec(
        kind=ArdCellKind.RECEIVER, worker="agents_scaling.ard.receiver:run_receiver_cell",
        needs_endpoint=True, lane=LANE_SMALL, cpus=1, mem="4G", time="1-00:00:00",
        items_per_cell=8, parallel_items=2, max_inflight=8, owner="WP5",
        doc="Frozen-content receiver assay over synthetic programs (spec §11).",
    ),
    ArdCellKind.SELECTOR_AUDIT: KindSpec(
        kind=ArdCellKind.SELECTOR_AUDIT, worker="agents_scaling.ard.selectors:run_selector_audit_cell",
        needs_endpoint=True, lane=LANE_SMALL, cpus=1, mem="4G", time="1-00:00:00",
        items_per_cell=8, parallel_items=2, max_inflight=4, owner="WP5",
        dep_prefixes=(STATE_DEP_PREFIX,),
        doc="Set-wise LLM audit selector over sealed pools (spec §12.4).",
    ),
    ArdCellKind.GRADE: KindSpec(
        kind=ArdCellKind.GRADE, worker="agents_scaling.ard.runner:grade_worker",
        needs_endpoint=False, lane=LANE_GRADE, cpus=2, mem="4G", time="12:00:00",
        items_per_cell=32, parallel_items=2, max_inflight=1,
        doc="Evaluator lane: MATH/MMLU grading of committed answers (spec §18 labels).",
    ),
    ArdCellKind.E0: KindSpec(
        kind=ArdCellKind.E0, worker="agents_scaling.ard.legacy_e0:run_e0_cell",
        needs_endpoint=False, lane=LANE_ANALYSIS, cpus=4, mem="32G", time="2-00:00:00",
        items_per_cell=1, parallel_items=1, max_inflight=1, owner="WP7",
        doc="CPU reanalysis of the study_v4 run root (spec §8).",
    ),
    ArdCellKind.COMPACT: KindSpec(
        kind=ArdCellKind.COMPACT, worker="agents_scaling.ard.compact:run_compact_cell",
        needs_endpoint=False, lane=LANE_ANALYSIS, cpus=4, mem="32G", time="12:00:00",
        items_per_cell=1, parallel_items=1, max_inflight=1,
        doc="Cells JSON -> the seven §18 tables plus the plan §5.8 descriptive tables.",
    ),
    ArdCellKind.CAPTURE: KindSpec(
        kind=ArdCellKind.CAPTURE, worker="agents_scaling.ard.neural.capture:run_capture_cell",
        needs_endpoint=False, lane=LANE_CAPTURE, cpus=8, mem="60G", time="12:00:00",
        items_per_cell=8, parallel_items=1, max_inflight=1, owner="WP6",
        doc="Teacher-forced replay capture on a replay GPU (spec §14.5).",
    ),
}

if set(KIND_TABLE) != set(ArdCellKind):  # pragma: no cover - import-time contract
    raise RuntimeError("KIND_TABLE must cover every ArdCellKind")

#: Kinds that must never be dispatched to an ou_bcs_low endpoint (plan §6: deadline-bearing).
DEADLINE_BEARING: Final[frozenset[ArdCellKind]] = frozenset({
    ArdCellKind.FORK, ArdCellKind.EPISODE, ArdCellKind.CAPTURE,
})

#: Dispatch priority key (plan §7): lower sorts first inside one lane loop.
PRIORITY: Final[Mapping[ArdCellKind, int]] = {
    ArdCellKind.ROOTS: 0,
    ArdCellKind.EPISODE: 1,
    ArdCellKind.FORK: 2,
    ArdCellKind.GRADE: 2,
    ArdCellKind.CHALLENGE: 3,
    ArdCellKind.SUFFIX: 3,
    ArdCellKind.RECEIVER: 3,
    ArdCellKind.SELECTOR_AUDIT: 4,
    ArdCellKind.CAPTURE: 4,
    ArdCellKind.E0: 5,
    ArdCellKind.COMPACT: 6,
}


def spec_for(kind: ArdCellKind | str) -> KindSpec:
    try:
        return KIND_TABLE[ArdCellKind(kind)]
    except KeyError as exc:  # pragma: no cover - ArdCellKind() already raises
        raise ArdManifestError(f"no KIND_TABLE entry for {kind!r}") from exc


def items_per_cell(
    kind: ArdCellKind | str,
    *,
    checkpoint: str = K.CHECKPOINT_SMALL,
    reasoning_allowance: int = K.THINKING_TOKENS_LOW,
    protocol: K.Protocol | None = None,
) -> int:
    """Plan §6 cell sizing: 8 items for 8B 1,024-token ROOTS/EPISODE, 4 for HUB5 and every
    32B kind, 2-3 for any 4,096-budget kind (so no cell exceeds about an hour and a
    re-dispatch after a preemption is cheap)."""
    spec = spec_for(kind)
    if not spec.needs_endpoint:
        return spec.items_per_cell
    if int(reasoning_allowance) == K.THINKING_TOKENS_HIGH:
        return 2 if checkpoint == K.CHECKPOINT_LARGE else 3
    if checkpoint == K.CHECKPOINT_LARGE:
        return min(spec.items_per_cell, 4)
    if protocol is K.Protocol.HUB5:
        return min(spec.items_per_cell, 4)
    return spec.items_per_cell


def required_dependencies(protocol: K.Protocol | None, *, controller_sha: str | None) -> tuple[str, ...]:
    """GATE and CONF_PROTECT refuse to run without the frozen controller (spec §12.3)."""
    from agents_scaling.ard.types import controller_dependency

    if protocol in (K.Protocol.GATE, K.Protocol.CONF_PROTECT):
        if not controller_sha:
            raise ArdManifestError(
                f"{protocol.value} cells need the frozen controller sha "
                "(controller:<sha>); the earliest safe freeze is plan §7 P2")
        return (controller_dependency(controller_sha),)
    return ()


__all__ = [
    "DEADLINE_BEARING", "KIND_TABLE", "KindSpec", "LANE_ANALYSIS", "LANE_CAPTURE",
    "LANE_GRADE", "LANE_LARGE", "LANE_SMALL", "PRIORITY", "items_per_cell",
    "required_dependencies", "spec_for",
]
