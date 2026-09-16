"""The DAG: every block manifest once, the workload, the pilot subset, the plan table.

Plan §3.4 ("**Frozen manifests, no waves.** ``freeze`` writes every block manifest once ...
Bucket cells reference ``bundle:<family>/<bucket>``"), §3.1 (the workload table), §1.3 (the
mechanical reuse census), spec 19.1 (the pilot subset), 16.4 (budget admission), 17.5 (the
``plan`` verb).

What this module refuses to do, and why:

* **No waves, no INDEX ledger.**  ARD36 emitted one manifest per planner wave and kept an
  ``INDEX.json`` of them, which made "which manifest owns this cell" a runtime question and
  let two waves name one directory.  Here the whole DAG is emitted at freeze: a constructed
  bucket cell exists from t0 and simply *waits* for ``bundle:<family>/<bucket>``, which the
  CONSTRUCT cell writes hours later at a path both sides compute from the same bucket index.
* **No status from a directory listing.**  ``plan`` table rows get their status from the
  per-cell terminal census (:func:`cell_status`): ``meta.json`` **with directory ownership**,
  ``SUSPENDED.json``, ``DEPS_UNMET.json``, a ``reuse_of`` declaration.  "The directory exists"
  is not a status -- that is how ARD36 counted half-written cells as complete.
* **No guessed reuse.**  :func:`reuse_census` renders the packet plan of one synthetic team
  per condition and hashes the prompts (no model call, no GPU): the fraction of an arm's
  requests that alias an already-planned request is *computed*.  ``plan --dry-run`` prints it,
  ``workload`` charges against it, and ``status`` compares the live alias rate to it.
"""

from __future__ import annotations

import json
import math
import fcntl
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import cells as CE
from agents_scaling.nextdyn import cli, config as C, contract as K
from agents_scaling.nextdyn import deps, ids, kinds, tables, workload as WL
from agents_scaling.nextdyn.cells import Block, Bucket, CellSpec, PlanState
from agents_scaling.nextdyn.errors import ConfigError, ContractViolation, ManifestError
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.study import identity

PLAN_DIR: Final[str] = "plan"
PLAN_TABLE: Final[str] = "plan"
DRY_RUN_PREFIX: Final[str] = "dry_run"
REGISTER_PAYLOAD_DIR: Final[str] = "register_payloads"
PLANNER_LOCK: Final[str] = ".planner.lock"

STATUS_PLANNED: Final[str] = "planned"
STATUS_RUNNING: Final[str] = "running"
STATUS_COMPLETE: Final[str] = "complete"
STATUS_FAILED: Final[str] = "failed"
STATUS_BLOCKED: Final[str] = "blocked"
STATUS_SKIPPED: Final[str] = "skipped_quota_unfilled"
STATUS_REUSED: Final[str] = "reused"
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {STATUS_COMPLETE, STATUS_FAILED, STATUS_SKIPPED, STATUS_REUSED})

DEPS_UNMET_FILE: Final[str] = "DEPS_UNMET.json"
#: ``item_splits`` column prefix for sub-panel membership (``sub_e3a``, ...).
SUB_PANEL_PREFIX: Final[str] = "sub_"
#: The grade-pass tag of the planner's episode grading (cells.PlanState.grade_tag).
GRADE_TAG_EPISODES: Final[str] = "g1"


# --------------------------------------------------------------------------- inventory


def _screened_prefix(quota: int, rates: Mapping[str, float], family: str, frame: int) -> int:
    """How many items of ``family`` the frozen scan screens to fill ``quota`` paired items.

    A rate at or below zero means "no assumption": screen the whole frame.  The result is never
    larger than the frame, so an exhausted frame stops the scan exactly as spec 5.2 says, and
    the recruitment flow reports the realized eligibility beside the assumed one.
    """
    rate = float(rates.get(family, 0.0) or 0.0)
    if rate <= 0.0:
        return frame
    if not 0.0 < rate <= 1.0:
        raise ConfigError(f"study.cohorts.assumed_paired_eligibility[{family}] must be in (0, 1], got {rate}")
    return min(frame, int(math.ceil(quota / rate)))


@dataclass(frozen=True)
class Inventory:
    """The pre-outcome facts the planner reads: which items exist and which panel they are in.

    Split roles and sub-panel membership are decided by ``inventory`` **before any outcome**
    (plan §2.2), so building the DAG needs no grade and reads nothing protected (R7).
    """

    families: tuple[str, ...]
    items: Mapping[str, tuple[str, ...]]
    natural_items: Mapping[str, tuple[str, ...]]
    panels: Mapping[str, Mapping[str, tuple[str, ...]]]
    buckets: Mapping[str, tuple[Bucket, ...]]
    problems: tuple[str, ...] = ()

    @classmethod
    def from_run_root(cls, run_root: str | os.PathLike, cfg: Any) -> "Inventory":
        """Read the public item store and ``item_splits`` (never ``protected/``)."""
        from agents_scaling.nextdyn import items as I

        store = I.load_items(run_root)
        splits = {row["source_item_id"]: row for row in I.load_splits(run_root)}
        families = tuple(cfg.cohorts.families)
        by_family: dict[str, list[str]] = {f: [] for f in families}
        natural: dict[str, list[str]] = {f: [] for f in families}
        panels: dict[str, dict[str, list[str]]] = {p: {f: [] for f in families} for p in CE.PANELS}
        for item_id, item in sorted(store.items()):
            # ``cohorts.families`` names the *cohort* (MATH), while an item's ``family`` is the
            # dataset panel it came from (MATH500, MATH_EXPANDED): ``families.TASK_FAMILY`` is
            # the one map between them (R1), and the planner plans cohorts.
            cohort = item.task_family
            if cohort not in by_family:
                continue
            by_family[cohort].append(item_id)
            row = splits.get(item_id, {})
            if row.get("natural_rank") is not None:
                natural[cohort].append(item_id)
            for panel in CE.PANELS:
                # item_splits declares sub-panel membership as ``sub_<panel>`` (tables.py);
                # the planner reads it and never re-derives a panel from an outcome.
                if _truthy(row.get(f"{SUB_PANEL_PREFIX}{panel}")):
                    panels[panel][cohort].append(item_id)
        quota = int(cfg.cohorts.constructed_target_per_family)
        # Spec 5.2: scan a frozen ordering of the eligible frame and stop at the preregistered
        # quota or an exhausted frame -- NOT at the end of the frame regardless.  Eligibility is
        # only knowable by screening, so the screened prefix is the quota divided by the assumed
        # paired-eligibility rate for that cohort; the pilot measures the real rate and `power`
        # rewrites it before freeze.  Screening the whole frame instead would have cost 740k
        # bank calls (755 GPU-hours) to fill a 900-item quota -- three quarters of it discarded.
        screen = {f: _screened_prefix(quota, cfg.cohorts.assumed_paired_eligibility, f,
                                      len(by_family[f])) for f in families}
        by_family = {f: v[:screen[f]] for f, v in by_family.items()}
        panel_of = {item_id: item.family for item_id, item in store.items()}
        buckets = {f: tuple(Bucket(family=f, index=i, source_item_id=item, pair=_pair_name(cfg),
                                   panel_family=panel_of.get(item, f))
                            for i, item in enumerate(by_family[f][:quota]))
                   for f in families}
        return cls(families=families, problems=mod97_problem_ids(cfg),
                   items={f: tuple(v) for f, v in by_family.items()},
                   natural_items={f: tuple(v or by_family[f]) for f, v in natural.items()},
                   panels={p: {f: tuple(v) for f, v in fam.items()} for p, fam in panels.items()},
                   buckets=buckets)

    def panel(self, name: str, family: str) -> tuple[str, ...]:
        """One sub-panel's items for one cohort (the same reader ``PlanState`` exposes)."""
        return tuple(self.panels.get(name, {}).get(family, ()))

    def limited(self, *, per_family: int | None = None,
                panels: Mapping[str, int] | None = None,
                problems: int | None = None) -> "Inventory":
        """A prefix of every list (the pilot subset is exactly this, spec 19.1).

        ``problems`` is separate from ``per_family`` because the E2b panel is *generated* from
        the study seed rather than read from the item store, so it has no family to be cut
        within: a ``limited(per_family=1)`` inventory still carried all 33,600 E2b cells.
        """
        def cut(values: Sequence[str], n: int | None) -> tuple[str, ...]:
            return tuple(values) if n is None else tuple(values[:n])

        panel_sizes = dict(panels or {})
        return replace(
            self,
            items={f: cut(v, per_family) for f, v in self.items.items()},
            natural_items={f: cut(v, per_family) for f, v in self.natural_items.items()},
            panels={p: {f: cut(v, panel_sizes.get(p, per_family)) for f, v in fam.items()}
                    for p, fam in self.panels.items()},
            buckets={f: tuple(b for b in v if per_family is None or b.index < per_family)
                     for f, v in self.buckets.items()},
            problems=cut(self.problems, problems),
        )


def mod97_problem_ids(cfg: Any) -> tuple[str, ...]:
    """The E2b panel's problem ids: a pure function of the study seed (spec 9.2).

    Generated rather than read from disk because that is what makes an E2b cell dispatchable
    at t0 with no dependency at all (plan §3.4): the receiver worker regenerates exactly the
    same panel from the same seed, so the manifest and the worker cannot disagree.
    """
    from agents_scaling.nextdyn import mod97

    e2b = cfg.experiments.E2b
    spec = mod97.spec_from_config(e2b)
    per_depth = max(int(e2b.source_problems) // max(len(e2b.chain_depths), 1), 1)
    return tuple(problem.problem_id for problem in
                 mod97.generate_panel(cfg.contract.study_seed, spec=spec, per_depth=per_depth))


def _truthy(value: Any) -> bool:
    return bool(value) and value not in ("false", "False")


def _pair_name(cfg: Any) -> str:
    """The *name* of the declared paired panel (``compositions.yaml: pairs``).

    The name, never the member list: a bucket cell's ``pair`` option travels in a manifest and
    ``cells.assert_option_is_label_free`` refuses a value that spells out ``C1_COHERENT`` --
    which is the point, because the pair name is opaque and the membership is a writer-side
    fact the CONSTRUCT worker resolves from ``compositions.yaml`` itself.
    """
    from agents_scaling.nextdyn import construct as CO

    declared = tuple(cfg.cohorts.paired_compositions)
    comps = CO.load_compositions(C.sibling_config(cfg.path, "compositions.yaml"))
    for name, members in comps.pairs.items():
        if tuple(members) == declared:
            return str(name)
    raise ContractViolation(
        f"compositions.yaml declares no pair with members {list(declared)} "
        f"(study.cohorts.paired_compositions); pairs: {sorted(comps.pairs)}")


# --------------------------------------------------------------------------- the plan


@dataclass(frozen=True)
class Plan:
    """The whole DAG of one stage: the cells of every block, in dispatch order."""

    stage: str
    state: PlanState
    manifests: Mapping[str, tuple[CellSpec, ...]]
    edges: tuple[tuple[str, str], ...]
    reuse: Mapping[str, float]

    @property
    def blocks(self) -> tuple[str, ...]:
        return tuple(self.manifests)

    @property
    def cells(self) -> tuple[CellSpec, ...]:
        return tuple(cell for block in self.manifests for cell in self.manifests[block])

    def cell(self, cell_id: str) -> CellSpec:
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        raise ManifestError(f"no planned cell {cell_id!r}")

    def of_block(self, block: str) -> tuple[CellSpec, ...]:
        return tuple(self.manifests.get(block, ()))

    def write_manifests(self, run_root: str | os.PathLike, *, meta: Mapping[str, Any] | None = None
                        ) -> dict[str, str]:
        out: dict[str, str] = {}
        for block, group in self.manifests.items():
            out[block] = CE.write_manifest(CE.manifest_path(run_root, block), group, block=block,
                                           meta=dict(meta or {}))
        return out


def plan_state(cfg: Any, inventory: Inventory, *, stage: str, compute: Any | None = None,
               registry: Any | None = None, contract: K.StudyContract | None = None,
               **extra: Any) -> PlanState:
    """Compile the inputs every block builder reads."""
    return PlanState(
        stage=stage, contract=contract or cfg.contract, cfg=cfg,
        compute=compute if compute is not None else C.load_compute(C.DEFAULT_COMPUTE_CONFIG),
        registry=registry if registry is not None else _load_registry(cfg),
        families=inventory.families, items=inventory.items,
        natural_items=inventory.natural_items, buckets=inventory.buckets,
        panels=inventory.panels, problems=inventory.problems, **extra)


def _load_registry(cfg: Any) -> Any:
    from agents_scaling.nextdyn import protocols as P

    path = C.sibling_config(cfg.path, C.DEFAULT_PROTOCOLS_CONFIG.name) or C.DEFAULT_PROTOCOLS_CONFIG
    return P.load_registry(path)


def enabled_blocks(cfg: Any) -> tuple[str, ...]:
    """The blocks this configuration dispatches (plan D11: E3b/E3c are expressible, not run)."""
    experiments = cfg.experiments
    gate = {
        Block.E1_NATURAL: experiments.E1.enabled,
        Block.E1_CONSTRUCTED: experiments.E1.enabled,
        Block.E1_FRAGMENTED: experiments.E1.enabled,
        Block.E1_BRIDGE: experiments.E1.enabled,
        Block.E3A: experiments.E3a.enabled,
        Block.E3A_PROTECTION: experiments.E3a.enabled,
        Block.E3A_MATCHED_SLOT: experiments.E3a.enabled,
        Block.E2A: experiments.E2a.enabled,
        Block.E2B: experiments.E2b.enabled,
        Block.E0: experiments.E0.enabled,
        Block.CAPTURE: cfg.instrumentation.enabled,
        Block.E3B_RING: experiments.E3b.enabled,
        Block.E3B_SIZE: experiments.E3b.enabled,
        Block.E3C_REPLAY: experiments.E3c.enabled,
        Block.E3C_FAMILY: experiments.E3C_FAMILY.enabled,
    }
    return tuple(b for b in CE.BLOCKS if gate.get(b, True))


def build_dag(cfg: Any, inventory: Inventory, *, stage: str, compute: Any | None = None,
              registry: Any | None = None, blocks: Sequence[str] | None = None) -> Plan:
    """Every block manifest, once, with path-addressed dependencies (plan §3.4).

    The order matters only for the dependency wiring: NATURAL_ROOTS before the natural
    episodes (which depend on those cells), the generation blocks before GRADE (which grades
    them), GRADE before CONSTRUCT, and everything before MATERIALIZE.  Nothing here waits for
    a *runtime* fact: the bundle path of bucket k is known at freeze.
    """
    state = plan_state(cfg, inventory, stage=stage, compute=compute, registry=registry)
    wanted = tuple(blocks) if blocks is not None else enabled_blocks(cfg)
    builder = CE.ManifestBuilder(study_seed=state.contract.study_seed)
    manifests: dict[str, tuple[CellSpec, ...]] = {}

    # 1. the t0 blocks; natural roots first so the natural episodes can name their cells
    for block in (Block.BANKS, Block.NATURAL_ROOTS):
        if block in wanted:
            builder.extend(CE.build_block(block, state), block=block)
    root_cells = {f: tuple(c.cell_id for c in builder.cells(Block.NATURAL_ROOTS) if c.family == f)
                  for f in state.families}
    bank_cells = {f: tuple(c.cell_id for c in builder.cells(Block.BANKS) if c.family == f)
                  for f in state.families}
    state = replace(state, root_cells=root_cells, grade_targets=bank_cells)

    # 2. the bank -> grade -> construct -> bundle chain.  GRADE is emitted first with the bank
    #    cells as its targets, then CONSTRUCT with the GRADE cells as its dependency: a team is
    #    built from *graded* donors, so a CONSTRUCT cell that ran before its grades would find
    #    every donor unresolved and quietly build nothing.
    if Block.GRADE in wanted:
        builder.extend(CE.build_block(Block.GRADE, state), block=Block.GRADE)
    grade_cells = {f: tuple(c.cell_id for c in builder.cells(Block.GRADE) if c.family == f)
                   for f in state.families}
    state = replace(state, grade_targets=grade_cells)
    if Block.CONSTRUCT in wanted:
        builder.extend(CE.build_block(Block.CONSTRUCT, state), block=Block.CONSTRUCT)

    # 3. the episode and receiver blocks
    episode_blocks = (Block.E1_NATURAL, Block.E1_CONSTRUCTED, Block.E1_FRAGMENTED, Block.E1_BRIDGE,
                      Block.E3A, Block.E3A_PROTECTION, Block.E3A_MATCHED_SLOT, Block.E2A, Block.E2B,
                      Block.E3B_RING, Block.E3B_SIZE, Block.E3C_REPLAY, Block.E3C_FAMILY)
    for block in episode_blocks:
        if block in wanted:
            builder.extend(CE.build_block(block, state), block=block)

    # 4. capture and the close-out cells
    episode_cells = {f: tuple(c.cell_id for block in episode_blocks for c in builder.cells(block)
                              if c.family == f) for f in state.families}
    state = replace(state, episode_cells=episode_cells)
    for block in (Block.AUDIT, Block.CAPTURE, Block.E0, Block.MATERIALIZE):
        if block in wanted:
            builder.extend(CE.build_block(block, state), block=block)

    for block in builder.blocks:
        manifests[block] = tuple(builder.cells(block))
    manifests = _mark_reuse(manifests, cfg)
    # MATERIALIZE deliberately declares NO cell dependency.  "All complete -> materialize"
    # (plan §3.4) is the happy path, but a bucket the frozen scan never filled terminates as
    # ``skipped_quota_unfilled`` (spec 5.2) and a blanket ``cell:`` dependency on every planned
    # cell would let one unfillable bucket block the close-out of the whole run -- forever, and
    # silently, because an unmet dependency is not a failure.  What actually has to hold is
    # that every planned cell is *terminal*, which is the plan table's census and is asserted
    # by ``release-check``; ``materialize`` publishes what is committed and records how many
    # cells it skipped in ``MANIFEST.json``.
    edges = tuple((dep, cell.cell_id) for block in manifests for cell in manifests[block]
                  for dep in cell.depends_on)
    reuse = reuse_census(cfg, state.registry, contract=state.contract).fractions
    return Plan(stage=stage, state=state, manifests=manifests, edges=edges, reuse=reuse)


def _mark_reuse(manifests: Mapping[str, tuple[CellSpec, ...]], cfg: Any) -> dict[str, tuple[CellSpec, ...]]:
    """A condition declaring ``equivalent_to`` is dispatched once, as a ``reuse_of`` cell.

    ``IN1_OUT1`` resolves to the complete graph, so its episodes are byte-identical to PEER's
    (``test_n3_ring_equals_all_to_all_is_not_duplicate_treatment`` makes the same point for an
    N=3 ring).  Emitting a second dispatch would double the cost and, worse, would report one
    treatment twice; emitting a ``reuse_of`` cell records the equivalence instead.
    """
    registry = _load_registry(cfg)
    equivalent = {name: spec.equivalent_to for name, spec in registry.episodes.items()
                  if getattr(spec, "equivalent_to", None)}
    if not equivalent:
        return dict(manifests)
    hash_of = {name: spec.condition_hash for name, spec in registry.episodes.items()}
    # keyed on (family, condition hash, items): the *requests* alias whatever cell dispatches
    # them, so the twin need not be in the same block or carry the same composition.
    target_index: dict[tuple[str, str, str], str] = {}
    for block, group in sorted(manifests.items()):
        for cell in sorted(group, key=lambda c: c.cell_id):
            if cell.condition is None:
                continue
            target_index.setdefault(
                (cell.family, cell.protocol_hash, ",".join(cell.items)), cell.cell_id)
    out: dict[str, tuple[CellSpec, ...]] = {}
    for block, group in manifests.items():
        rebuilt = []
        for cell in group:
            label = cell.display_label
            twin = equivalent.get(label or "")
            if twin and twin in hash_of:
                target = target_index.get((cell.family, hash_of[twin], ",".join(cell.items)))
                if target and target != cell.cell_id:
                    cell = replace(cell, reuse_of=target)
            rebuilt.append(cell)
        out[block] = tuple(rebuilt)
    return out


# --------------------------------------------------------------------------- reuse census


@dataclass(frozen=True)
class ReuseCensus:
    """The mechanically computed request-level reuse of the planned conditions (plan §1.3)."""

    prompts_by_condition: Mapping[str, int]
    unique_by_condition: Mapping[str, int]
    fractions: Mapping[str, float]
    total_prompts: int
    total_unique: int

    @property
    def overall_fraction(self) -> float:
        return 0.0 if not self.total_prompts else 1.0 - self.total_unique / self.total_prompts

    def to_dict(self) -> dict[str, Any]:
        return {"prompts_by_condition": dict(self.prompts_by_condition),
                "unique_by_condition": dict(self.unique_by_condition),
                "fractions": dict(self.fractions), "total_prompts": self.total_prompts,
                "total_unique": self.total_unique, "overall_fraction": self.overall_fraction}


def reuse_census(cfg: Any, registry: Any, *, contract: K.StudyContract | None = None) -> ReuseCensus:
    """Render one synthetic team's prompts under every condition and hash them (plan §1.3).

    No model is called and no GPU is touched.  Two requests alias in the store when their
    **prompt bytes and their seed key** agree, and the seed key is condition-free by D4, so
    the whole question is: does this condition's slot-``i`` round-``r`` prompt equal one
    another condition already planned for the same slot and round?  A slot's *packet* at
    round r is itself determined by that slot's request at round r (common random numbers),
    so the census carries a packet id forward rather than a fixed string: the isolated focal
    slot of IN0_OUT1 diverges from PEER's slot at round 2 exactly when its round-1 prompt
    did, which is what makes the printed number the one ``status`` can compare a live alias
    rate against.

    Conditions are visited with the core protocols first, so a later arm is credited with
    reusing an earlier arm's request and never the other way round.
    """
    from agents_scaling.nextdyn import graphs as G
    from agents_scaling.nextdyn import prompts as PR
    from agents_scaling.nextdyn import protocols as P
    from agents_scaling.nextdyn import states as ST

    contract = contract or cfg.contract
    n = int(contract.team.n_agents)
    revisions = int(contract.team.revision_rounds)
    seed = contract.study_seed
    item = _SyntheticItem()
    refs = tuple(ST.DecisionRef(decision_id=f"d:{slot}", round_=0, agent_slot=slot,
                                generation_uid=f"g:{slot}", output=None) for slot in range(n))
    snapshot = ST.initial_state(pairing_id="census", roots=refs, classify=lambda text: text,
                                study_seed=seed)
    seen: set[str] = set()
    per_condition: dict[str, int] = {}
    unique: dict[str, int] = {}
    for name in _census_order(cfg, registry):
        spec = registry.episodes[name]
        total = 0
        fresh = 0
        state = snapshot
        # packet[slot] identifies the bytes slot's packet carries INTO the next round; at
        # round 0 that is its root draw, which every condition shares (spec 8.6).
        packet = {slot: f"root:{slot}" for slot in range(n)}
        for round_ in range(revisions):
            # the census fixes the focal slot (the placeholder an episode resolves from its
            # root bundle): the aliasing structure is the same for any one choice of slot.
            graph = G.resolve_graph(spec.graph, n=n, round_=round_ + 1,
                                    focal_slot=CENSUS_FOCAL_SLOT, study_seed=seed,
                                    pairing_id="census")
            plan = P.packet_plan(replace(state, graph=graph), spec, round_=round_ + 1,
                                 study_seed=seed)
            state = ST.advance(state, {slot: ST.DecisionRef(
                decision_id=f"d:{round_}:{slot}", round_=round_ + 1, agent_slot=slot,
                generation_uid=f"g:{round_}:{slot}", output=None) for slot in range(n)},
                round_=round_ + 1)
            produced: dict[int, str] = {}
            for slot in range(n):
                own = packet[slot] if spec.own_draft else None
                # a filler occupies the slot without carrying the sender's bytes, and a
                # packet-content mode (answer-only) changes the bytes a real packet shows:
                # both are prompt differences and must not be counted as reuse.
                peers = [f"filler:{intent.display_position}" if intent.filler
                         else f"{packet[intent.sender_slot]}|{intent.content.value}"
                         for intent in plan.get(slot, ())]
                if spec.template is P.Template.ROOT:
                    messages = PR.render_root(item, item)
                else:
                    messages = PR.render_revision(item, own, peers)
                digest = identity.sha256_hex(identity.jcs(
                    [round_ + 1, slot, [dict(m) for m in messages]]))
                produced[slot] = digest
                total += 1
                if digest not in seen:
                    seen.add(digest)
                    fresh += 1
            packet = produced
        per_condition[spec.display_label] = total
        unique[spec.display_label] = fresh
    fractions = {label: (0.0 if not total else 1.0 - unique[label] / total)
                 for label, total in per_condition.items()}
    return ReuseCensus(prompts_by_condition=per_condition, unique_by_condition=unique,
                       fractions=fractions, total_prompts=sum(per_condition.values()),
                       total_unique=sum(unique.values()))


def _census_order(cfg: Any, registry: Any) -> tuple[str, ...]:
    """Core protocols first, then every other enabled episode condition, in name order."""
    core = tuple(p for p in cfg.protocol.protocols if p in registry.episodes)
    rest = tuple(sorted(name for name, spec in registry.episodes.items()
                        if name not in core and getattr(spec, "enabled", True)))
    return core + rest


#: The focal slot the census resolves ``$focal`` to (any one slot; see :func:`reuse_census`).
CENSUS_FOCAL_SLOT: Final[int] = 0


class _SyntheticItem:
    """One item that is not an item: the census needs a question, never a task (R7)."""

    @property
    def rendered_question(self) -> str:
        return "census"

    @property
    def answer_instruction(self) -> str:
        return "census"


# --------------------------------------------------------------------------- the plan table


def cell_status(run_root: str | os.PathLike, cell: CellSpec, *, final: bool = False) -> str:
    """The cell's terminal census, never "the directory exists" (plan §3.4).

    ``final`` is the close-out pass: a bucket cell whose bundle never arrived is
    ``skipped_quota_unfilled`` (the scan stopped at the quota, spec 5.2), which is a planned
    outcome rather than a failure.
    """
    if cell.reuse_of:
        return STATUS_REUSED
    directory = deps.cell_dir(run_root, cell.cell_id)
    if (directory / deps.SUSPENDED_FILE).exists():
        return STATUS_FAILED
    status = deps.dependency_status(run_root, deps.cell_dependency(cell.cell_id))
    if status.met:
        return STATUS_COMPLETE
    if status.status == deps.STATUS_TAMPERED:
        return STATUS_FAILED
    started = (directory / "items").exists() and any((directory / "items").glob("*.json"))
    if final:
        # At close-out an unmet dependency is not "waiting" any more: the bundle its bucket
        # needed was never written because the frozen scan found the item was not
        # paired-eligible, and the cell terminates as ``skipped_quota_unfilled`` (spec 5.2).
        # The ordering matters -- ``DEPS_UNMET.json`` is on disk for exactly these cells, and
        # reading it first would report a whole quota's tail as blocked forever.
        if deps.unmet(run_root, cell.depends_on):
            return STATUS_SKIPPED
        return STATUS_RUNNING if started else STATUS_FAILED
    if (directory / DEPS_UNMET_FILE).exists():
        return STATUS_BLOCKED
    if started:
        return STATUS_RUNNING
    return STATUS_PLANNED


def plan_rows(run_root: str | os.PathLike, plan: Plan, *, manifest_hashes: Mapping[str, str] | None = None,
              final: bool = False) -> list[tables.PlanRow]:
    """One :class:`tables.PlanRow` per planned cell, status from the terminal census."""
    hashes = dict(manifest_hashes or {})
    rows: list[tables.PlanRow] = []
    for block, group in plan.manifests.items():
        for cell in group:
            status = cell_status(run_root, cell, final=final)
            census = {"status": status, "n_items": cell.n_items, "block": block}
            rows.append(tables.PlanRow(
                cell_id=cell.cell_id, cell_kind=cell.kind.value, lane=cell.lane.value,
                pool=None if cell.pool is None else cell.pool.value, experiment=block,
                manifest_hash=hashes.get(block, ""), priority_rank=kinds.priority_rank(cell.kind),
                n_items=cell.n_items,
                depends_on_json=json.dumps(list(cell.depends_on), sort_keys=True),
                status=status,
                terminal_census_json=json.dumps(census, sort_keys=True),
                reuse_of=cell.reuse_of))
    return rows


def write_plan_table(run_root: str | os.PathLike, rows: Sequence[tables.PlanRow]) -> Path:
    """``<run_root>/plan/plan.parquet`` -- the producer side of the ``plan`` table (R1/R3)."""
    path = Path(run_root) / PLAN_DIR / f"{PLAN_TABLE}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tables.write_table(list(rows), path, table=PLAN_TABLE)
    return path


# --------------------------------------------------------------------------- pilot subsets


@dataclass(frozen=True)
class PilotSubset:
    """Spec 19.1 verbatim: the pilot dispatch subset, every size read from ``study.yaml``."""

    natural_items_per_family: int
    constructed_paired_items_per_family: int
    directional_items_per_family: int
    natural_receiver_items: int
    natural_receiver_tasks: tuple[str, ...]
    synthetic_receiver_problems: int
    adapter_check_problems: int
    capture_items_per_family: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "natural_items_per_family": self.natural_items_per_family,
            "constructed_paired_items_per_family": self.constructed_paired_items_per_family,
            "directional_items_per_family": self.directional_items_per_family,
            "natural_receiver_items": self.natural_receiver_items,
            "natural_receiver_tasks": list(self.natural_receiver_tasks),
            "synthetic_receiver_problems": self.synthetic_receiver_problems,
            "adapter_check_problems": self.adapter_check_problems,
            "capture_items_per_family": self.capture_items_per_family,
        }


def pilot_subsets(cfg: Any) -> PilotSubset:
    """The spec-19.1 row sizes, read (never re-typed) from ``study.yaml``'s ``pilot`` block."""
    pilot = cfg.pilot
    return PilotSubset(
        natural_items_per_family=int(pilot.natural_items_per_family),
        constructed_paired_items_per_family=int(pilot.constructed_paired_items_per_family),
        directional_items_per_family=int(pilot.directional_items_per_family),
        natural_receiver_items=int(pilot.natural_receiver_items),
        natural_receiver_tasks=tuple(pilot.natural_receiver_tasks),
        synthetic_receiver_problems=int(pilot.synthetic_receiver_problems),
        adapter_check_problems=int(pilot.adapter_check_problems),
        capture_items_per_family=int(pilot.capture_items_per_family))


def pilot_inventory(cfg: Any, inventory: Inventory) -> Inventory:
    """The pilot's inventory: the spec-19.1 prefixes of every panel (plan D18, §3.1)."""
    subset = pilot_subsets(cfg)
    panels = {
        CE.PANEL_E3A: subset.directional_items_per_family,
        CE.PANEL_E3A_FILL: subset.directional_items_per_family,
        CE.PANEL_CAPTURE: subset.capture_items_per_family,
        CE.PANEL_CAPTURE_NATURAL: subset.capture_items_per_family,
        CE.PANEL_E2A: subset.natural_receiver_items,
        CE.PANEL_BRIDGE: subset.constructed_paired_items_per_family,
        CE.PANEL_FRAGMENTED: subset.constructed_paired_items_per_family,
        CE.PANEL_SECOND_ROOT: subset.natural_items_per_family,
        CE.PANEL_NATPROT: subset.natural_items_per_family,
    }
    reduced = inventory.limited(per_family=subset.constructed_paired_items_per_family, panels=panels)
    natural = {f: tuple(v[:subset.natural_items_per_family])
               for f, v in reduced.natural_items.items()}
    problems = tuple(inventory.problems[:subset.synthetic_receiver_problems])
    return replace(reduced, natural_items=natural, problems=problems)


# --------------------------------------------------------------------------- registers


def register_payload_path(run_root: str | os.PathLike, kind: str, sha: str) -> Path:
    """Where the planner keeps the payload it sealed, so ``--repair-registers`` can re-seal."""
    return Path(run_root) / PLAN_DIR / REGISTER_PAYLOAD_DIR / kind / f"{sha}.json"


def seal_register(run_root: str | os.PathLike, kind: str, payload: Mapping[str, Any]) -> str:
    """Seal a register and keep its payload beside the plan (idempotent, atomic)."""
    sha = deps.write_register(run_root, kind, payload)
    path = register_payload_path(run_root, kind, sha)
    if not path.exists():
        deps.atomic_write_bytes(path, (json.dumps(dict(payload), sort_keys=True, indent=2) + "\n")
                                .encode("utf-8"))
    return sha


def repair_registers(run_root: str | os.PathLike) -> cli.Report:
    """Regenerate the registers whose ``REGISTER.json`` is missing; write only what is missing.

    A tampered register is *reported*, never silently rewritten: the register is
    content-addressed, so different bytes under the same sha mean something edited the sealed
    set, and a repair pass that overwrote it would erase the evidence.
    """
    root = Path(run_root)
    base = root / PLAN_DIR / REGISTER_PAYLOAD_DIR
    repaired = skipped = tampered = 0
    details: list[str] = []
    for kind in deps.REGISTER_KINDS:
        for payload_path in sorted((base / kind).glob("*.json")) if (base / kind).exists() else []:
            sha = payload_path.stem
            status = deps.verify_register(root, kind, sha)
            if status.status == deps.STATUS_MET:
                skipped += 1
                continue
            if status.status == deps.STATUS_TAMPERED:
                tampered += 1
                details.append(f"register:{kind}/{sha[:12]} is tampered: {status.detail}")
                continue
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            written = deps.write_register(root, kind, payload)
            if written != sha:
                tampered += 1
                details.append(f"register:{kind}/{sha[:12]} re-seals as {written[:12]}")
                continue
            repaired += 1
    for family_dir in sorted((root / "plan" / "root_teams").glob("*")) if (root / "plan" / "root_teams").exists() else []:
        for bundle in sorted(family_dir.glob("*.json")):
            status = deps.verify_bundle(root, family_dir.name, bundle.stem)
            if status.status == deps.STATUS_TAMPERED:
                tampered += 1
                details.append(f"bundle:{family_dir.name}/{bundle.stem} is tampered: {status.detail}")
    return cli.Report(completed=repaired, skipped=skipped, failed=tampered,
                      detail="; ".join(details) if details else f"registers repaired={repaired}")


# --------------------------------------------------------------------------- the planner pass


def complete_cells(run_root: str | os.PathLike, kind: CellKind) -> list[tuple[str, Any]]:
    """``(cell_id, ItemFile)`` for every item of every **complete** cell of one kind.

    The protected sibling is read too, because the planner's writer-side passes (the E2a packet
    pair is chosen from *graded* donors) need ``private_rows``; a reader that opened only the
    public file would find no grades and quietly seal an empty panel.
    """
    root = Path(run_root)
    out: list[tuple[str, Any]] = []
    base = root / deps.CELLS_DIR
    for directory in sorted(base.glob("*")) if base.exists() else []:
        if not directory.is_dir() or not (directory / deps.META_FILE).exists():
            continue
        if not directory.name.startswith(kind.value + CE.ID_SEP):
            continue
        for path in sorted((directory / tables.ITEMS_DIR).glob("*.json")):
            protected = tables.protected_item_path(root, directory.name, path.stem)
            out.append((directory.name, tables.ItemFile.read(path, protected_path=protected)))
    return out


def terminal_pools(run_root: str | os.PathLike, cfg: Any, *, counter: Any = None,
                   counter_hash: str = "", classify: Any = None) -> list[tuple[Any, dict[str, str]]]:
    """Seal one pool per completed episode at its terminal round (spec 6.5 ``terminal_only``).

    The planner may not read a grade (the selection service cannot, test 18.1), so a
    candidate's voting class is the **gold-independent** canonical form of its own answer and
    its displayed bytes are rendered here through ``packets.render_packet`` -- the same
    renderer the episode used, at the same cap and with the same denylist.
    """
    from agents_scaling.nextdyn import packets as PK
    from agents_scaling.nextdyn import protocols as P
    from agents_scaling.nextdyn import selectors as SEL

    contract = cfg.contract
    count = counter if counter is not None else (lambda text: len(str(text).split()))
    canonical = classify if classify is not None else (lambda text: text)
    # which tokenizer counted the packet: resolved from the row's own generating model, so a
    # 32B replay of an 8B team stamps the 32B tokenizer (spec 6.1, cross-model replay).
    tokenizer_hashes = {ids.model_hash(model): PK.tokenizer_hash(model)
                        for model in contract.models.values() if model.tokenizer_revision}
    out: list[tuple[Any, dict[str, str]]] = []
    for _cell_id, item_file in complete_cells(run_root, CellKind.EPISODE):
        by_episode: dict[str, list[Any]] = {}
        for row in item_file.rows.get("decisions", ()):
            by_episode.setdefault(row.episode_id, []).append(row)
        for episode_id, rows in sorted(by_episode.items()):
            terminal = max(int(r.round) for r in rows)
            final = sorted((r for r in rows if int(r.round) == terminal),
                           key=lambda r: int(r.agent_slot))
            if not final:
                continue
            names = PK.pseudonym_map(contract.study_seed, episode_id, len(final))
            candidates = []
            bodies: dict[str, str] = {}
            for index, row in enumerate(final):
                output = _output_of(row)
                if output is None:
                    continue
                stamp = counter_hash or tokenizer_hashes.get(row.receiver_model_hash, "")
                packet = PK.render_packet(output, pseudonym=names[index], counter=count,
                                          counter_tokenizer_hash=stamp,
                                          cap=int(contract.tokens.packet_cap),
                                          content=P.PacketContent.FULL)
                bodies[packet.packet_hash] = packet.serialized
                valid = bool(output.valid)
                candidates.append(SEL.Candidate(
                    candidate_id=row.decision_id, generation_uid=row.generation_uid,
                    content_hash=row.content_hash,
                    canonical_class=(canonical(output.final_answer) if valid else None),
                    valid=valid, confidence=row.confidence, packet_hash=packet.packet_hash,
                    slot=int(row.agent_slot)))
            if not candidates:
                continue
            out.append((SEL.SealedPool(
                source_item_id=item_file.source_item_id, episode_id=episode_id,
                cell_kind=CellKind.EPISODE, round_=terminal,
                scope=contract.selection.pool_scope, candidates=tuple(candidates),
                protocol_hash=final[0].protocol_hash), bodies))
    return out


def _output_of(row: Any) -> Any:
    from agents_scaling.nextdyn import states as ST

    return ST._output_of(row)


def seal_pools(run_root: str | os.PathLike, cfg: Any, **kwargs: Any) -> Mapping[str, str]:
    """``register:pool/<sha>`` per item: every sealed pool of that item, with its bytes."""
    from agents_scaling.nextdyn import runner as RUN

    by_item: dict[str, list[dict[str, Any]]] = {}
    for pool, bodies in terminal_pools(run_root, cfg, **kwargs):
        by_item.setdefault(pool.source_item_id, []).append(
            RUN.SealedPoolCodec.to_entry(pool, bodies=bodies))
    out: dict[str, str] = {}
    for item_id, entries in sorted(by_item.items()):
        payload = {"scope": cfg.contract.selection.pool_scope, "source_item_id": item_id,
                   deps.REGISTER_ENTRIES_FIELD: entries}
        out[item_id] = seal_register(run_root, deps.REGISTER_POOL, payload)
    return out


def seal_capture_panel(run_root: str | os.PathLike, cfg: Any) -> Mapping[str, str]:
    """``register:panel/<sha>`` per family: which items the read-only replay captures."""
    out: dict[str, str] = {}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for cell_id, item_file in complete_cells(run_root, CellKind.EPISODE):
        rows = item_file.rows.get("decisions", ())
        if not rows:
            continue
        by_family.setdefault(_family_of(cell_id), []).append(
            {"source_item_id": item_file.source_item_id, "cell_id": cell_id,
             "decision_ids": sorted({r.decision_id for r in rows})})
    for family, entries in sorted(by_family.items()):
        payload = {"family": family, "anchors": list(cfg.instrumentation.anchors),
                   deps.REGISTER_ENTRIES_FIELD: entries}
        out[family] = seal_register(run_root, deps.REGISTER_PANEL, payload)
    return out


def _family_of(cell_id: str) -> str:
    """The family component of a cell id (``<kind>.<family>.…``, cells.cell_id_for)."""
    parts = cell_id.split(CE.ID_SEP)
    return parts[1] if len(parts) > 1 else ""


def seal_e2a_packets(run_root: str | os.PathLike, cfg: Any, inventory: Inventory, *,
                     counter: Any = None, counter_hash: str = ""
                     ) -> Mapping[str, Mapping[str, str]]:
    """``register:packets/<sha>`` per task: the fixed natural packet pair of each E2a item.

    The writer-side half of spec 9.1: for each item of the E2a sub-panel, take the first
    gradeable *correct* and first gradeable *wrong* donor draw in blind order and freeze the
    two packets **through** ``packets.render_packet`` -- same cap, same counter, same denylist
    as an episode packet, so the receiver replays bytes an agent could have sent.  An item
    whose donor bank supplies no contrast is simply not in the panel (``fix_natural_pair``
    returns ``None``), which the register's entry count records.
    """
    from agents_scaling.nextdyn import banks as BK
    from agents_scaling.nextdyn import items as I
    from agents_scaling.nextdyn import natural_packets as NP
    from agents_scaling.nextdyn import packets as PK
    from agents_scaling.nextdyn import states as ST

    root = Path(run_root)
    e2a = cfg.experiments.E2a
    contract = cfg.contract
    model = contract.model(e2a.models[0])
    count = counter if counter is not None else (lambda text: len(str(text).split()))
    stamp = counter_hash or (PK.tokenizer_hash(model) if model.tokenizer_revision else "")
    try:
        store = I.load_items(root)
    except Exception:                                   # noqa: BLE001 - no inventory, no panel
        return {}
    draws: dict[str, list[Any]] = {}
    grades: dict[str, Any] = {}
    for _cell_id, item_file in complete_cells(root, CellKind.DONOR_BANK):
        for row in item_file.rows.get("banks", ()):
            draws.setdefault(row.source_item_id, []).append(row)
        for row in item_file.rows.get("decisions", ()):
            if row.generation_uid:
                grades.setdefault(row.generation_uid, ST._output_of(row))
    grade_rows: dict[str, Any] = {}
    for _cell_id, item_file in complete_cells(root, CellKind.GRADE):
        for row in item_file.private_rows.get("grades_private", ()):
            grade_rows.setdefault(row.generation_uid, row)

    out: dict[str, dict[str, str]] = {}
    for family in e2a.tasks:
        sealed: dict[str, str] = {}
        for item_id in inventory.panel(CE.PANEL_E2A, family)[:int(e2a.items_per_task)]:
            item = store.get(item_id)
            if item is None:
                continue
            donors = [NP.DonorDraw(generation_uid=row.generation_uid, decision_id=row.decision_id,
                                   draw_index=int(row.draw_index),
                                   output=grades.get(row.generation_uid),
                                   grade_row=grade_rows.get(row.generation_uid))
                      for row in sorted(draws.get(item_id, ()), key=lambda r: r.draw_index)
                      if row.bank_role == BK.BankRole.DONOR.value
                      and grade_rows.get(row.generation_uid) is not None
                      and grades.get(row.generation_uid) is not None]
            if not donors:
                continue
            pair = NP.fix_natural_pair(item, donors, study_seed=contract.study_seed,
                                       counter=count, counter_tokenizer_hash=stamp,
                                       tokens=contract.tokens)
            if pair is None:
                continue
            # one register per item: the receiver's episode id is built from the panel id
            # (contract §2.0), so a panel shared by many items would give them one episode
            sealed[item_id] = NP.seal_panel(root, [pair], study_seed=contract.study_seed,
                                            family=family, model_hash=ids.model_hash(model),
                                            budget_hash=contract.tokens.budget_hash())
        if sealed:
            out[family] = sealed
    return out


def planner_lock_path(run_root: str | os.PathLike) -> Path:
    """``<run_root>/plan/.planner.lock`` -- the one file a planner pass holds (plan §3.4)."""
    return Path(run_root) / PLAN_DIR / PLANNER_LOCK


def planner_pass(run_root: str | os.PathLike, cfg: Any, *, inventory: Inventory | None = None,
                 counter: Any = None, counter_hash: str = "", classify: Any = None
                 ) -> cli.Report:
    """One planner pass, under the §3.4 ``flock``: a second instance does nothing and exits 0.

    The lock is the whole point of the wrapper.  Sealing a pool, writing its content-addressed
    payload and emitting the manifests it unblocks is idempotent *against itself*, but two
    passes interleaved -- the Slurm planner job mid-pass and a hand-run
    ``nextdyn plan --planner-pass`` on the login node -- are two sealers racing over the same
    registers, payloads and manifests.  ``slurm/nextdyn_planner.sbatch.tmpl`` states that this
    lock is held; declaring it and never taking it is ARD36's D-03 shape, which this WP exists
    to design out.  Non-blocking on purpose: the second instance is a *duplicate*, so waiting
    for the first would only queue a pass whose work is already being done.
    """
    lock_path = planner_lock_path(run_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return cli.Report(skipped=1, detail="another planner pass holds the lock")
        try:
            return _planner_pass(run_root, cfg, inventory=inventory, counter=counter,
                                 counter_hash=counter_hash, classify=classify)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _planner_pass(run_root: str | os.PathLike, cfg: Any, *, inventory: Inventory | None = None,
                  counter: Any = None, counter_hash: str = "", classify: Any = None
                  ) -> cli.Report:
    """The pass itself: seal what completed, emit what that unblocks, repair, re-census.

    Idempotent and a pure function of disk state, so a pass killed anywhere is repaired by the
    next one (every artifact is tmp+fsync+rename and every register is content-addressed).
    """
    root = Path(run_root)
    pools = seal_pools(root, cfg, counter=counter, counter_hash=counter_hash, classify=classify)
    panels = seal_capture_panel(root, cfg)
    inventory = inventory if inventory is not None else Inventory.from_run_root(root, cfg)
    packets = seal_e2a_packets(root, cfg, inventory, counter=counter, counter_hash=counter_hash)
    repaired = repair_registers(root)
    state_extra = {
        "pool_registers": tuple(sorted(pools.items())),
        "panel_registers": dict(panels),
        "packet_registers": dict(packets),
        "episode_cells": _episode_cells_by_family(root),
    }
    state = plan_state(cfg, inventory, stage=CE.STAGE_CONFIRMATION, **state_extra)
    builder = CE.ManifestBuilder(study_seed=state.contract.study_seed)
    for block in (Block.AUDIT, Block.CAPTURE, Block.E2A):
        builder.extend(CE.build_block(block, state), block=block)
    # the second grading pass: the episodes the banks' GRADE cells never saw.  Without it the
    # outcome tables have no grades for any revision and every primary loses its bounds.
    graded = replace(state, grade_targets=state_extra["episode_cells"], grade_tag=GRADE_TAG_EPISODES)
    builder.extend(CE.build_block(Block.GRADE, graded), block=Block.GRADE)
    written = builder.write(root, meta={"stage": "planner"}, content_addressed=True)
    return cli.Report(completed=len(written), skipped=repaired.skipped, failed=repaired.failed,
                      detail=f"pools={len(pools)} panels={len(panels)} "
                             f"packets={sum(len(v) for v in packets.values())} "
                             f"manifests={sorted(written)}")


def _episode_cells_by_family(run_root: Path) -> Mapping[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    base = run_root / deps.CELLS_DIR
    for directory in sorted(base.glob("*")) if base.exists() else []:
        if directory.is_dir() and directory.name.startswith(CellKind.EPISODE.value + CE.ID_SEP) \
                and (directory / deps.META_FILE).exists():
            out.setdefault(_family_of(directory.name), []).append(directory.name)
    return {family: tuple(cells) for family, cells in out.items()}


# --------------------------------------------------------------------------- the smoke

#: The blocks ``validate --smoke`` dispatches, in the order it dispatches them.  Chosen to be
#: the smallest set that reaches every phase ``gates.launch_gate`` scores: BANKS and
#: NATURAL_ROOTS produce ``bank``, E1_NATURAL produces ``revision``, E2B produces
#: ``receiver_plan`` and ``receiver_decision``.  The fifth phase, ``audit``, has no block here
#: because its cells do not exist until the planner seals a pool from completed episodes --
#: :func:`smoke_audit_dag` emits it after the episode blocks have run.
SMOKE_BLOCKS: Final[tuple[str, ...]] = (
    Block.BANKS, Block.NATURAL_ROOTS, Block.E1_NATURAL, Block.E2B,
)


def smoke_inventory(cfg: Any, inventory: Inventory) -> Inventory:
    """The smallest inventory that still reaches every scored phase.

    One item per family rather than ``smoke_solver_calls_per_model`` items per family, which is
    what the first version of this did: a bank *item* is 16 or 32 draws, so reading a 20-*call*
    budget as 20 items per family dispatched 864 calls -- a knob whose declared meaning was not
    what it did, which is the D-03 shape this package is built to design out.  One item per
    family is the floor, and it clears the gate's ``>= 20`` comfortably (144 bank calls).
    """
    return inventory.limited(per_family=int(cfg.pilot.smoke_episode_items),
                             problems=int(cfg.pilot.smoke_receiver_episodes))


def smoke_dag(cfg: Any, inventory: Inventory, *, compute: Any | None = None) -> Plan:
    """The dispatchable part of the smoke: every block whose cells exist before any run.

    E2b is cut *per model* rather than by a flat cell prefix.  The builder orders its cells by
    arm and then by model, so a flat prefix would have been entirely 8B and the 32B adapter --
    the thing the smoke exists to check on both models -- would have gone unmeasured while the
    gate reported the receiver phases green.
    """
    plan = build_dag(cfg, smoke_inventory(cfg, inventory), stage=CE.STAGE_SMOKE,
                     compute=compute, blocks=SMOKE_BLOCKS)
    per_model = int(cfg.pilot.smoke_receiver_episodes)
    kept: dict[str, int] = {}
    e2b: list[CellSpec] = []
    for cell in plan.of_block(Block.E2B):
        if kept.get(cell.model, 0) >= per_model:
            continue
        kept[cell.model] = kept.get(cell.model, 0) + 1
        e2b.append(cell)
    manifests = dict(plan.manifests)
    manifests[Block.E2B] = tuple(e2b)
    return replace(plan, manifests=manifests)


def smoke_audit_dag(run_root: str | os.PathLike, cfg: Any, *, compute: Any | None = None
                    ) -> tuple[CellSpec, ...]:
    """The ``audit`` phase: SELECTION_AUDIT cells over the pools the smoke's episodes sealed.

    Read off disk rather than planned, because an audit cell's identity *is* the sha of a pool
    that did not exist until the episodes committed.  Returns empty when no pool sealed, which
    the caller reports -- the launch gate then scores ``audit`` as unmeasured, which is a
    failure, and that is the correct outcome for a smoke whose episodes never landed.
    """
    root = Path(run_root)
    pools = seal_pools(root, cfg)
    if not pools:
        return ()
    inventory = Inventory.from_run_root(root, cfg)
    state = plan_state(cfg, smoke_inventory(cfg, inventory), stage=CE.STAGE_SMOKE,
                       compute=compute, pool_registers=tuple(sorted(pools.items())),
                       panel_registers={}, packet_registers={}, episode_cells={})
    cells = CE.build_block(Block.AUDIT, state)
    return tuple(cells[:int(cfg.pilot.smoke_audit_cells)])


# --------------------------------------------------------------------------- the verb


def dry_run(cfg: Any, *, run_root: Path, inventory: Inventory, stage: str,
            blocks: Sequence[str] | None = None, compute: Any | None = None) -> tuple[Plan, dict[str, Any]]:
    """Build the DAG, cost it, admit it, and write ``plan/dry_run_<ts>.json``."""
    plan = build_dag(cfg, inventory, stage=stage, compute=compute, blocks=blocks)
    compute = compute if compute is not None else C.load_compute(
        C.sibling_config(getattr(cfg, "path", C.DEFAULT_STUDY_CONFIG),
                         C.DEFAULT_COMPUTE_CONFIG.name) or C.DEFAULT_COMPUTE_CONFIG)
    profile = WL.load_cost_profile(run_root / "pilot")
    load = WL.calls(plan.cells, contract=plan.state.contract, profile=profile,
                    reuse_fraction=plan.reuse)
    admission = WL.admit(load, budgets=cfg.budgets)
    census = reuse_census(cfg, plan.state.registry, contract=plan.state.contract)
    floor = WL.fleet_floor(load, hours=_target_hours(compute))
    projection = WL.projection(load, replicas=floor,
                               reserve_fraction=float(cfg.budgets.compute_reserve_fraction))
    report = {
        "stage": stage,
        "blocks": {block: len(group) for block, group in plan.manifests.items()},
        "workload": load.to_dict(),
        "admission": admission.to_dict(),
        "reuse_census": census.to_dict(),
        "fleet_floor": floor,
        "projection": projection.to_dict(),
        "manifest_sha": {block: identity.sha256_hex(CE.manifest_bytes(group, block=block))
                         for block, group in plan.manifests.items()},
    }
    out = run_root / PLAN_DIR / f"{DRY_RUN_PREFIX}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(WL.render_table(load, admission))
    print("\n".join(budget_lines(cfg)))
    print("\n".join(kind_lines()))
    print(f"mechanical reuse census: {census.total_unique}/{census.total_prompts} prompts unique "
          f"({census.overall_fraction:.3f} reused)")
    print(f"fleet floor for {_target_hours(compute):.0f} h: {floor}")
    print(f"at that fleet: {projection.critical_path_hours:.1f} h on the critical lane, "
          f"{projection.gpu_hours_with_reserve:.0f} GPU-h with the reserve")
    print(f"dag: {len(plan.cells)} cells, {len(plan.edges)} dependency edges -> {out}")
    return plan, report


def budget_lines(cfg: Any) -> list[str]:
    """What a human authorized, printed by ``plan --dry-run`` before anything is dispatched.

    Every ceiling is named, including the ones that are null: a null ``campaign_gpu_hours`` is
    the spec-16.4 launch blocker, and a null ``campaign_api_currency_limit`` is the (correct)
    statement that this study buys no API calls.  Printing them is how the human who has to
    write them sees what is still missing.
    """
    budgets = cfg.budgets
    selection = cfg.contract.selection
    return [
        f"campaign_gpu_hours   {budgets.campaign_gpu_hours}",
        f"campaign_cpu_hours   {budgets.campaign_cpu_hours}",
        f"api currency limit   {budgets.campaign_api_currency_limit} (no paid API in this study)",
        f"storage_limit_bytes  {budgets.storage_limit_bytes}",
        f"reserves             compute x{1 + budgets.compute_reserve_fraction:.2f}, "
        f"storage x{1 + budgets.storage_reserve_fraction:.2f}",
        f"archive audit        {'on' if selection.archive_audit_enabled else 'off'}",
    ]


def kind_lines() -> list[str]:
    """One line per cell kind: what it is, from ``KindSpec.doc`` (the kind table's own words)."""
    return [f"{kind.value:<18} {kinds.spec_for(kind).doc}" for kind in kinds.PRIORITY]


def planner_interval_s(compute: Any) -> float:
    """How often the planner pass runs (``compute.dispatch.planner_pass_s``)."""
    return float(compute.dispatch.planner_pass_s)


def _target_hours(compute: Any) -> float:
    """The wall clock the fleet floor is sized against (``compute.dispatch.target_wall_hours``).

    A dispatch quantity, so it lives in ``compute.yaml`` and nowhere else (plan §3.3): it is
    what ``plan --dry-run`` prints, what sizes the floor ``gates launch --required-replicas
    floor`` checks, and therefore what decides whether the freeze tier is reduced.
    """
    return float(compute.dispatch.target_wall_hours)


def main(argv: Sequence[str] | None = None) -> cli.Report:
    parser = cli.common_parser("plan", description="build the DAG, cost it, admit it")
    parser.add_argument("--dry-run", action="store_true", help="print the workload; dispatch nothing")
    parser.add_argument("--block", default=None, help="one block (the expressibility gate)")
    parser.add_argument("--repair-registers", action="store_true",
                        help="regenerate missing registers from their sealed payloads")
    parser.add_argument("--planner-pass", action="store_true",
                        help="seal what completed and emit the manifests it unblocks")
    parser.add_argument("--interval", type=float, default=None,
                        help="with --planner-pass: repeat every N seconds (the planner job)")
    parser.add_argument("--stage", default=CE.STAGE_CONFIRMATION, choices=list(CE.STAGES))
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_root = cli.resolve_run_root(args)
    if args.repair_registers and not args.planner_pass:
        return repair_registers(run_root)
    cfg = C.load_study(args.config)
    if args.planner_pass:
        report = cli.Report()
        while True:
            report = planner_pass(run_root, cfg)
            print(f"[planner] {report.line()} {report.detail}")
            interval = args.interval if args.interval is not None else None
            if interval is None:
                return report
            time.sleep(float(interval) if float(interval) > 0
                       else planner_interval_s(C.load_compute(
                           C.sibling_config(args.config, C.DEFAULT_COMPUTE_CONFIG.name)
                           or C.DEFAULT_COMPUTE_CONFIG)))
    try:
        inventory = Inventory.from_run_root(run_root, cfg)
    except Exception as exc:                        # noqa: BLE001 - reported, never a traceback
        return cli.Report(blocked=1, detail=f"no inventory under {run_root}: {exc}")
    blocks = (args.block,) if args.block else None
    if args.block and args.block not in CE.BLOCKS:
        return cli.Report(blocked=1, detail=f"unknown block {args.block!r}")
    plan, report = dry_run(cfg, run_root=run_root, inventory=inventory, stage=args.stage,
                           blocks=blocks)
    admitted = report["admission"]["admitted"] and report["admission"]["storage_admitted"]
    if args.dry_run:
        return cli.Report(completed=len(plan.cells), blocked=0 if admitted else 1,
                          detail=report["admission"]["detail"])
    rows = plan_rows(run_root, plan)
    write_plan_table(run_root, rows)
    return cli.Report(completed=len(rows), blocked=0 if admitted else 1,
                      detail=f"plan table written for {len(rows)} cells")


def pilot_main(argv: Sequence[str] | None = None) -> cli.Report:
    """``pilot``: the same DAG at the spec-19.1 sizes, in the sub-run-root ``$RUN/pilot`` (D18)."""
    parser = cli.common_parser("pilot", description="build and write the pilot manifests")
    parser.add_argument("--extend", type=int, default=None,
                        help="the frozen eligibility extension (spec 14.3): 120 or 180")
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = C.load_study(args.config)
    parent = cli.resolve_run_root(args)
    run_root = parent / CE.STAGE_PILOT
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        inventory = Inventory.from_run_root(parent, cfg)
    except Exception as exc:                        # noqa: BLE001
        return cli.Report(blocked=1, detail=f"no inventory under {parent}: {exc}")
    if args.extend is not None:
        allowed = tuple(cfg.cohorts.pilot_eligibility_extensions)
        if args.extend not in allowed:
            return cli.Report(blocked=1, detail=f"--extend must be one of {list(allowed)}")
        inventory = inventory.limited(per_family=args.extend)
    subset = pilot_subsets(cfg)
    plan = build_dag(cfg, pilot_inventory(cfg, inventory), stage=CE.STAGE_PILOT)
    hashes = plan.write_manifests(run_root, meta={"stage": CE.STAGE_PILOT,
                                                  "subset": subset.to_dict()})
    write_plan_table(run_root, plan_rows(run_root, plan, manifest_hashes=hashes))
    return cli.Report(completed=len(plan.cells),
                      detail=f"pilot manifests under {run_root} ({len(hashes)} blocks)")


__all__ = [
    "DEPS_UNMET_FILE", "Inventory", "PLANNER_LOCK", "PLAN_DIR", "PLAN_TABLE", "PilotSubset",
    "Plan", "ReuseCensus", "STATUS_BLOCKED", "STATUS_COMPLETE", "STATUS_FAILED", "STATUS_PLANNED",
    "STATUS_REUSED", "STATUS_RUNNING", "STATUS_SKIPPED",
    "TERMINAL_STATUSES", "build_dag", "cell_status", "dry_run", "enabled_blocks", "main",
    "pilot_inventory", "pilot_main", "pilot_subsets", "plan_rows", "plan_state",
    "budget_lines", "complete_cells", "kind_lines", "mod97_problem_ids", "planner_interval_s",
    "planner_lock_path", "planner_pass", "register_payload_path", "repair_registers",
    "reuse_census", "seal_capture_panel", "seal_e2a_packets", "seal_pools", "seal_register",
    "terminal_pools",
    "write_plan_table",
]
