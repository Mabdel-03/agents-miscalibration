"""Prior use, near duplicates and split roles -- decided before any outcome exists (spec 5.2,
7.1; plan §2.2, D19; verb ``inventory``).

**Writer side (R7).**

Three refusals shape this module:

* **"untouched" is never claimed.**  An item nobody can prove was used is ``unknown``, not
  untouched (spec 5.2).  Evidence is graded ``used_verified > used_inferred > unknown >
  reserved_unused`` and *every* root that said anything is listed in
  ``source_items.prior_use_sources_json``.  Run0's twelve benchmark contracts are **replayed**
  -- ``random.Random(f"{seed}-{i}")`` through the pinned loader, then
  ``benchmarks.contracts.canonical_question_sha256`` compared against the contract's
  ``question_sha256s`` -- and a row whose hash does not reproduce is ``used_inferred``, never
  silently ``used_verified``.  The twelve prior roots with no contract sidecar can only say
  ``unknown``.
* **Splits are hash order, not outcomes.**  Every eligible item of a cohort family is ranked
  by ``study.data.splits.rank_key(split_salt, source_item_id)`` before a single draw exists;
  the pilot block is the first ``pilot_items_per_family`` ranks plus the two §14.3 extension
  blocks (recorded in ``pilot_stage``, whether or not they are activated, so a confirmation
  item can never have been screened in the pilot), the natural frame is the D19 rule -- the
  post-pilot items whose prior use is in ``cohorts.natural_frame_prior_use`` -- and the
  constructed scan frame is the rest of the post-pilot order.
* **Every sub-panel is a rank prefix, written now.**  ``sub_*`` membership is a prefix of
  ``rank_key(split_salt, "sub:<name>:" + source_item_id)`` over the frame it draws from, so a
  panel can never be re-drawn after seeing a result; a panel larger than its frame is a
  :class:`~agents_scaling.nextdyn.errors.ContractViolation`, never a quietly smaller panel.

All variants of an item inherit its ``split_role`` and its cluster (``cluster_id ==
source_item_id``): variants exist to control option position, never to enter a second frame.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import datasets as D
from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn import items as I
from agents_scaling.nextdyn import tables
from agents_scaling.nextdyn.cli import Report, common_parser, resolve_run_root
from agents_scaling.nextdyn.config import StudyConfig, load_study
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.study import identity
from agents_scaling.study.data.splits import rank_key

if TYPE_CHECKING:  # pragma: no cover - the writer's draft type, imported lazily at runtime
    from agents_scaling.nextdyn.items_build import ItemDraft

# --------------------------------------------------------------------------- vocabulary

STATUS_USED_VERIFIED: Final[str] = "used_verified"
STATUS_USED_INFERRED: Final[str] = "used_inferred"
STATUS_UNKNOWN: Final[str] = "unknown"
STATUS_RESERVED_UNUSED: Final[str] = "reserved_unused"
#: Weakest first; the status of an item is the strongest any root justifies.
STATUS_ORDER: Final[tuple[str, ...]] = (
    STATUS_RESERVED_UNUSED, STATUS_UNKNOWN, STATUS_USED_INFERRED, STATUS_USED_VERIFIED,
)

MATCH_SOURCE_ID: Final[str] = "source_id"
MATCH_TEXT_HASH: Final[str] = "text_hash"
MATCH_REPLAY_VERIFIED: Final[str] = "replay_verified"
MATCH_REPLAY_INFERRED: Final[str] = "replay_inferred"
MATCH_NONE: Final[str] = "none"

SPLIT_PILOT: Final[str] = "pilot"
SPLIT_NATURAL: Final[str] = "natural"
SPLIT_CONSTRUCTED: Final[str] = "constructed"
SPLIT_BOTH: Final[str] = "natural_and_constructed"
SPLIT_RESERVE: Final[str] = "reserve"
SPLIT_EXCLUDED_DUPLICATE: Final[str] = "excluded_duplicate"
SPLIT_EXCLUDED_INELIGIBLE: Final[str] = "excluded_ineligible"
SPLIT_ROLES: Final[tuple[str, ...]] = (
    SPLIT_PILOT, SPLIT_NATURAL, SPLIT_CONSTRUCTED, SPLIT_BOTH, SPLIT_RESERVE,
    SPLIT_EXCLUDED_DUPLICATE, SPLIT_EXCLUDED_INELIGIBLE,
)

PILOT_STAGE_NONE: Final[str] = "none"
PILOT_STAGE_BASE: Final[str] = "pilot"
PILOT_STAGE_PREFIX: Final[str] = "pilot_ext_"

#: The near-duplicate audit (plan §2.2): 5-gram word shingles, Jaccard at or above 0.8 inside
#: one family.  Flagged and excluded from every frame, never merged -- two items that look
#: alike are still two items (spec 5.2).
SHINGLE_N: Final[int] = 5
SHINGLE_JACCARD: Final[float] = 0.8
#: Blocking for the audit: compare only items that share one of their rarest shingles, and
#: skip a block bigger than this (a block that large is boilerplate, not a duplicate cluster).
SHINGLE_BLOCK_KEYS: Final[int] = 3
SHINGLE_BLOCK_MAX: Final[int] = 256

#: ARD36's family names -> nextdyn panels (its ``items.parquet`` / ``items_reserve.json``).
ARD36_FAMILY_MAP: Final[Mapping[str, str]] = {"math500": F.MATH500, "mmlu_pro": F.MMLU_PRO}
#: Run0 / study_v4 benchmark names -> nextdyn panels (``benchmarks.loaders`` keys).
BENCHMARK_FAMILY_MAP: Final[Mapping[str, str]] = {
    "math": F.MATH500, "mmlu_pro": F.MMLU_PRO, "truthfulqa": F.TRUTHFULQA,
}
#: ``qid`` prefixes the Run0 contracts use, per benchmark.
BENCHMARK_QID_PREFIX: Final[Mapping[str, str]] = {
    "math": "math-", "mmlu_pro": "mmlupro-", "truthfulqa": "tqa-",
}

ROOT_ARD36: Final[str] = "ard36"
ROOT_STUDY_V4: Final[str] = "study_v4"
ROOT_FULL_SWEEP: Final[str] = "full_sweep_v1"
ARD36_ITEMS: Final[str] = "data/public/items.parquet"
ARD36_RESERVE: Final[str] = "data/public/items_reserve.json"
STUDY_V4_TASKS: Final[str] = "data/public/tasks.jsonl"
RUN0_CONTRACTS: Final[str] = "benchmark_contracts.v1.json"


# --------------------------------------------------------------------------- rows


@dataclass(frozen=True)
class PriorUse:
    """One statement by one legacy root about one item (the ``overlap_inventory`` row)."""

    source_item_id: str
    family: str
    root: str
    status: str
    match_method: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in STATUS_ORDER:
            raise ContractViolation(f"unknown prior-use status {self.status!r}")

    def as_dict(self) -> dict[str, str]:
        return {"source_item_id": self.source_item_id, "family": self.family, "root": self.root,
                "status": self.status, "match_method": self.match_method, "detail": self.detail}


@dataclass(frozen=True)
class NearDuplicate:
    """Two items of one family whose 5-gram shingles overlap at or above the threshold."""

    source_item_id: str
    near_duplicate_of: str
    family: str
    jaccard: float


@dataclass(frozen=True)
class FrameAssignment:
    """Everything ``inventory`` decides about one item, in one object."""

    source_item_id: str
    prior_use_status: str
    prior_use_sources: tuple[str, ...]
    near_duplicate_of: str | None
    sampling_weight: float
    frame_size: int | None
    frame_rule: str
    split_role: str
    split_row: tables.ItemSplitRow


# --------------------------------------------------------------------------- prior use


def _by_original_id(drafts: Sequence["ItemDraft"]) -> dict[tuple[str, str], str]:
    return {(draft.family, draft.row.original_id): draft.source_item_id for draft in drafts}


def _by_text_hash(drafts: Sequence["ItemDraft"]) -> dict[str, list["ItemDraft"]]:
    out: dict[str, list["ItemDraft"]] = {}
    for draft in drafts:
        out.setdefault(draft.row.normalized_text_hash, []).append(draft)
    return out


def _ard36_prior_use(root: Path, by_id: Mapping[tuple[str, str], str],
                     families: Mapping[str, str]) -> list[PriorUse]:
    out: list[PriorUse] = []
    items = root / ARD36_ITEMS
    if items.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(items, columns=["source_id", "family"])
        for source_id, family in zip(table.column("source_id").to_pylist(),
                                     table.column("family").to_pylist()):
            name = ARD36_FAMILY_MAP.get(str(family))
            item_id = by_id.get((name, str(source_id))) if name else None
            if item_id is not None:
                out.append(PriorUse(item_id, name, ROOT_ARD36, STATUS_USED_VERIFIED,
                                    MATCH_SOURCE_ID, f"items.parquet source_id {source_id}"))
    reserve = root / ARD36_RESERVE
    if reserve.exists():
        payload = json.loads(reserve.read_text(encoding="utf-8")).get("reserve") or {}
        for family, source_ids in payload.items():
            name = ARD36_FAMILY_MAP.get(str(family))
            if name is None:
                continue
            for source_id in source_ids:
                item_id = by_id.get((name, str(source_id)))
                if item_id is not None:
                    out.append(PriorUse(item_id, name, ROOT_ARD36, STATUS_RESERVED_UNUSED,
                                        MATCH_SOURCE_ID, "items_reserve.json"))
    del families
    return out


def _study_v4_prior_use(root: Path, by_text: Mapping[str, list["ItemDraft"]]) -> list[PriorUse]:
    tasks = root / STUDY_V4_TASKS
    if not tasks.exists():
        return []
    out: list[PriorUse] = []
    for line in tasks.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        digest = D.normalized_text_hash(str(task.get("task_text") or ""))
        for draft in by_text.get(digest, ()):
            out.append(PriorUse(draft.source_item_id, draft.family, ROOT_STUDY_V4,
                                STATUS_USED_VERIFIED, MATCH_TEXT_HASH,
                                f"tasks.jsonl {task.get('source_id')}"))
    return out


def _run0_index(benchmark: str, hf_home: str | os.PathLike) -> dict[int, str]:
    """``raw row index -> original_id`` for one Run0 benchmark, in the loader's own order."""
    family = BENCHMARK_FAMILY_MAP[benchmark]
    spec = F.family(family).loader
    raw, _path = D._hf_rows(spec)
    if benchmark == "math":
        return {index: str(row.get("unique_id") or "") for index, row in enumerate(raw)}
    if benchmark == "mmlu_pro":
        return {index: f"mmlu_pro-{int(row['question_id'])}" for index, row in enumerate(raw)}
    return {index: f"tqa-{index}" for index, _row in enumerate(raw)}


def _run0_prior_use(contracts_path: Path, by_id: Mapping[tuple[str, str], str],
                    hf_home: str | os.PathLike) -> list[PriorUse]:
    """Replay every contract and compare the canonical question hashes (spec 7.1)."""
    if not contracts_path.exists():
        return []
    from agents_scaling.benchmarks.contracts import canonical_question_sha256
    from agents_scaling.benchmarks.loaders import load_benchmark

    payload = json.loads(contracts_path.read_text(encoding="utf-8"))
    out: list[PriorUse] = []
    indices: dict[str, dict[int, str]] = {}
    for contract in payload.get("contracts") or ():
        key = contract.get("key") or {}
        benchmark = str(key.get("benchmark") or "")
        family = BENCHMARK_FAMILY_MAP.get(benchmark)
        if family is None:
            continue
        if benchmark not in indices:
            indices[benchmark] = _run0_index(benchmark, hf_home)
        questions = load_benchmark(benchmark, n=key.get("n_questions"), seed=int(key.get("seed", 0)))
        expected = list(contract.get("question_sha256s") or ())
        prefix = BENCHMARK_QID_PREFIX[benchmark]
        for position, question in enumerate(questions):
            if not str(question.qid).startswith(prefix):
                continue
            row_index = int(str(question.qid)[len(prefix):])
            original_id = indices[benchmark].get(row_index)
            item_id = by_id.get((family, original_id or ""))
            if item_id is None:
                continue
            verified = (position < len(expected)
                        and canonical_question_sha256(question) == expected[position])
            out.append(PriorUse(
                item_id, family, ROOT_FULL_SWEEP,
                STATUS_USED_VERIFIED if verified else STATUS_USED_INFERRED,
                MATCH_REPLAY_VERIFIED if verified else MATCH_REPLAY_INFERRED,
                f"{benchmark} seed={key.get('seed')} index={row_index}"))
    return out


def _prior_root_use(registry: D.LegacyRegistry, drafts: Sequence["ItemDraft"]) -> list[PriorUse]:
    """The roots with no contract sidecar: every item of a family they touched is ``unknown``."""
    out: list[PriorUse] = []
    for root in registry.prior_roots:
        path = Path(root)
        if not path.exists():
            continue
        touched = _root_families(path)
        for draft in drafts:
            if touched is None or draft.family in touched:
                out.append(PriorUse(draft.source_item_id, draft.family, path.name, STATUS_UNKNOWN,
                                    MATCH_NONE, "root present, no contract sidecar"))
    return out


def _root_families(root: Path) -> frozenset[str] | None:
    """The nextdyn panels a sidecar-less root may have touched (``None`` = cannot tell)."""
    cells = root / "cells.json"
    if not cells.exists():
        return None
    try:
        payload = json.loads(cells.read_text(encoding="utf-8"))
    except ValueError:
        return None
    names: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, Mapping):
            value = node.get("benchmark")
            if isinstance(value, str) and value in BENCHMARK_FAMILY_MAP:
                names.add(BENCHMARK_FAMILY_MAP[value])
            for child in node.values():
                _walk(child)
        elif isinstance(node, list):
            for child in node:
                _walk(child)

    _walk(payload)
    return frozenset(names) if names else None


def prior_use_inventory(drafts: Sequence["ItemDraft"], hf_home: str | os.PathLike, *,
                        registry: D.LegacyRegistry | None = None) -> list[PriorUse]:
    """Every statement every legacy root makes about these items (spec 5.2, 7.1)."""
    sources = registry if registry is not None else D.load_legacy_sources()
    by_id = _by_original_id(drafts)
    by_text = _by_text_hash(drafts)
    out: list[PriorUse] = []
    ard36 = sources.roots.get(ROOT_ARD36)
    if ard36 and Path(ard36).exists():
        out.extend(_ard36_prior_use(Path(ard36), by_id, BENCHMARK_FAMILY_MAP))
    study_v4 = sources.roots.get(ROOT_STUDY_V4)
    if study_v4 and Path(study_v4).exists():
        out.extend(_study_v4_prior_use(Path(study_v4), by_text))
    run0 = sources.roots.get(ROOT_FULL_SWEEP)
    if run0:
        out.extend(_run0_prior_use(Path(run0) / RUN0_CONTRACTS, by_id, hf_home))
    out.extend(_prior_root_use(sources, drafts))
    return out


def prior_use_status(records: Sequence[PriorUse]) -> str:
    """The strongest claim any root makes; ``unknown`` when no root says anything."""
    if not records:
        return STATUS_UNKNOWN
    return max((r.status for r in records), key=STATUS_ORDER.index)


# --------------------------------------------------------------------------- near duplicates


def _shingles(text: str) -> frozenset[str]:
    words = D._normalize(text).split()
    if len(words) < SHINGLE_N:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[i:i + SHINGLE_N]) for i in range(len(words) - SHINGLE_N + 1))


def near_duplicate_audit(drafts: Sequence["ItemDraft"]) -> list[NearDuplicate]:
    """Flag (never merge) near-duplicate pairs inside one family.

    Blocking keeps the audit near-linear: an item's block keys are its three rarest shingles
    that *some other item also has*, so a pair is compared exactly when it shares rare content,
    and a shingle so common that more than ``SHINGLE_BLOCK_MAX`` items carry it is boilerplate
    rather than a duplicate cluster and is skipped.
    """
    out: list[NearDuplicate] = []
    by_family: dict[str, list["ItemDraft"]] = {}
    for draft in drafts:
        by_family.setdefault(draft.family, []).append(draft)
    for family, group in sorted(by_family.items()):
        shingles = {draft.source_item_id: _shingles(draft.row.question) for draft in group}
        frequency: Counter[str] = Counter()
        for values in shingles.values():
            frequency.update(values)
        blocks: dict[str, list[str]] = {}
        for item_id, values in shingles.items():
            shared = [value for value in values if frequency[value] > 1]
            for key in sorted(shared, key=lambda s: (frequency[s], s))[:SHINGLE_BLOCK_KEYS]:
                blocks.setdefault(key, []).append(item_id)
        seen: set[tuple[str, str]] = set()
        for members in blocks.values():
            if len(members) > SHINGLE_BLOCK_MAX:
                continue
            for i, left in enumerate(sorted(members)):
                for right in sorted(members)[i + 1:]:
                    if (left, right) in seen:
                        continue
                    seen.add((left, right))
                    a, b = shingles[left], shingles[right]
                    union = len(a | b)
                    if not union:
                        continue
                    score = len(a & b) / union
                    if score >= SHINGLE_JACCARD:
                        out.append(NearDuplicate(source_item_id=right, near_duplicate_of=left,
                                                 family=family, jaccard=float(score)))
    return sorted(out, key=lambda d: (d.family, d.source_item_id))


# --------------------------------------------------------------------------- split roles


@dataclass(frozen=True)
class SubPanel:
    """One hash-selected sub-panel: a rank prefix over a named frame (plan §1.4)."""

    name: str
    size: int
    frame: str


def sub_panels(cfg: StudyConfig) -> tuple[SubPanel, ...]:
    """The nine ``item_splits.sub_*`` panels and where each one draws from.

    Sizes come from their single declaration (R1): ``cohorts.sub_panels`` for the panels with
    no other home, ``experiments.E3a.target_items_per_task`` for the directed panel,
    ``instrumentation.target_items_per_task`` for capture, ``experiments.E2a.items_per_task``
    for the endorsement assay and ``cohorts.natural_target_per_family`` for the natural
    random-protection panel.
    """
    panels = cfg.cohorts.sub_panels
    return (
        SubPanel("second_root", panels.second_root, SPLIT_CONSTRUCTED),
        SubPanel("bridge", panels.bridge, SPLIT_CONSTRUCTED),
        SubPanel("fragmented", panels.fragmented, SPLIT_CONSTRUCTED),
        SubPanel("e3a", cfg.experiments.E3a.target_items_per_task, SPLIT_CONSTRUCTED),
        SubPanel("e3a_fill", panels.e3a_fill, SPLIT_CONSTRUCTED),
        SubPanel("capture", cfg.instrumentation.target_items_per_task, SPLIT_CONSTRUCTED),
        SubPanel("capture_natural", panels.capture_natural, SPLIT_NATURAL),
        SubPanel("e2a", cfg.experiments.E2a.items_per_task, SPLIT_CONSTRUCTED),
        SubPanel("natprot", cfg.cohorts.natural_target_per_family, SPLIT_NATURAL),
    )


def _pilot_stage(rank: int, cfg: StudyConfig) -> str | None:
    """``pilot`` / ``pilot_ext_120`` / ``pilot_ext_180`` for a rank inside the pilot block."""
    base = cfg.cohorts.pilot_items_per_family
    if rank < base:
        return PILOT_STAGE_BASE
    for bound in cfg.cohorts.pilot_eligibility_extensions:
        if rank < bound:
            return f"{PILOT_STAGE_PREFIX}{bound}"
    return None


def _pilot_depth(cfg: StudyConfig) -> int:
    extensions = cfg.cohorts.pilot_eligibility_extensions
    return max([cfg.cohorts.pilot_items_per_family, *extensions])


def assign_split_roles(drafts: Sequence["ItemDraft"], prior_use: Sequence[PriorUse],
                       duplicates: Sequence[NearDuplicate], cfg: StudyConfig,
                       split_salt: bytes) -> dict[str, FrameAssignment]:
    """One :class:`FrameAssignment` per item, decided by hash order alone (plan §2.2)."""
    if cfg.failure_policy.null_item_ids != "fatal":
        raise ContractViolation("study.failure_policy.null_item_ids must be 'fatal' before any "
                                "item table is built")
    manifest_hash = identity.sha256_hex(identity.jcs(
        [cfg.raw_sha256, sorted(cfg.cohorts.families), cfg.cohorts.natural_target_per_family]))
    statuses: dict[str, list[PriorUse]] = {}
    for record in prior_use:
        statuses.setdefault(record.source_item_id, []).append(record)
    duplicate_of = {d.source_item_id: d.near_duplicate_of for d in duplicates}
    cohorts = set(cfg.cohorts.families)
    natural_rule = set(cfg.cohorts.natural_frame_prior_use)
    allow_overlap = cfg.cohorts.allow_natural_constructed_item_overlap
    if cfg.cohorts.allow_pilot_confirmation_overlap:
        raise ContractViolation("study.cohorts.allow_pilot_confirmation_overlap is frozen at "
                                "false: a pilot item was screened before the frame was fixed")

    by_cohort: dict[str, list["ItemDraft"]] = {}
    out: dict[str, FrameAssignment] = {}
    for draft in drafts:
        by_cohort.setdefault(draft.task_family, []).append(draft)

    for cohort, group in sorted(by_cohort.items()):
        ranked = sorted(group, key=lambda d: rank_key(split_salt, d.source_item_id))
        eligible = [d for d in ranked
                    if cohort in cohorts and d.source_item_id not in duplicate_of]
        eligible_rank = {d.source_item_id: rank for rank, d in enumerate(eligible)}
        pilot_depth = _pilot_depth(cfg)
        post_pilot = eligible[pilot_depth:]
        natural_frame = [d for d in post_pilot
                         if prior_use_status(statuses.get(d.source_item_id, [])) in natural_rule]
        natural = natural_frame[:cfg.cohorts.natural_target_per_family]
        natural_ids = {d.source_item_id for d in natural}
        constructed = [d for d in post_pilot
                       if allow_overlap or d.source_item_id not in natural_ids]
        natural_rank = {d.source_item_id: rank for rank, d in enumerate(natural)}
        scan_rank = {d.source_item_id: rank for rank, d in enumerate(constructed)}
        frames = {SPLIT_NATURAL: [d.source_item_id for d in natural],
                  SPLIT_CONSTRUCTED: [d.source_item_id for d in constructed]}
        members = _sub_panel_members(cfg, frames, split_salt)

        for draft in ranked:
            item_id = draft.source_item_id
            records = statuses.get(item_id, [])
            status = prior_use_status(records)
            near = duplicate_of.get(item_id)
            if item_id not in eligible_rank:
                role = SPLIT_EXCLUDED_DUPLICATE if near else SPLIT_EXCLUDED_INELIGIBLE
                rule = ("excluded:near_duplicate" if near
                        else f"excluded:cohort_not_in_study({cohort})")
                frame_size, weight = None, 0.0
                stage, rank_in_frame, scan = PILOT_STAGE_NONE, None, None
            else:
                stage = _pilot_stage(eligible_rank[item_id], cfg) or PILOT_STAGE_NONE
                in_natural = item_id in natural_ids
                in_constructed = item_id in scan_rank
                if stage != PILOT_STAGE_NONE:
                    role = SPLIT_PILOT
                    rule = f"pilot:rank<{pilot_depth}"
                    frame_size = pilot_depth
                elif in_natural and in_constructed:
                    role, rule = SPLIT_BOTH, "natural+constructed:post_pilot"
                    frame_size = len(natural)
                elif in_natural:
                    role = SPLIT_NATURAL
                    rule = f"natural:post_pilot,prior_use in {sorted(natural_rule)}"
                    frame_size = len(natural)
                elif in_constructed:
                    role, rule = SPLIT_CONSTRUCTED, "constructed:post_pilot_scan_order"
                    frame_size = len(constructed)
                else:
                    role, rule = SPLIT_RESERVE, "reserve:post_pilot_outside_every_frame"
                    frame_size = len(post_pilot)
                weight = 1.0
                rank_in_frame = natural_rank.get(item_id)
                scan = scan_rank.get(item_id)
            out[item_id] = FrameAssignment(
                source_item_id=item_id, prior_use_status=status,
                prior_use_sources=tuple(sorted({r.root for r in records})),
                near_duplicate_of=near, sampling_weight=weight, frame_size=frame_size,
                frame_rule=rule, split_role=role,
                split_row=tables.ItemSplitRow(
                    source_item_id=item_id, family=draft.family,
                    rank_hex=rank_key(split_salt, item_id).hex(), split_role=role,
                    pilot_stage=stage, natural_rank=rank_in_frame, constructed_scan_rank=scan,
                    sub_second_root=item_id in members["second_root"],
                    sub_bridge=item_id in members["bridge"],
                    sub_fragmented=item_id in members["fragmented"],
                    sub_e3a=item_id in members["e3a"],
                    sub_e3a_fill=item_id in members["e3a_fill"],
                    sub_capture=item_id in members["capture"],
                    sub_capture_natural=item_id in members["capture_natural"],
                    sub_e2a=item_id in members["e2a"],
                    sub_natprot=item_id in members["natprot"],
                    assigned_at_manifest_hash=manifest_hash))
    return out


def _sub_panel_members(cfg: StudyConfig, frames: Mapping[str, Sequence[str]],
                       split_salt: bytes) -> dict[str, frozenset[str]]:
    """``{panel: members}``: a prefix of ``rank_key(salt, "sub:<name>:" + id)`` over its frame."""
    out: dict[str, frozenset[str]] = {}
    for panel in sub_panels(cfg):
        frame = list(frames.get(panel.frame, ()))
        if panel.size > len(frame):
            raise ContractViolation(
                f"sub-panel {panel.name!r} asks for {panel.size} items but its {panel.frame} frame "
                f"holds {len(frame)}; a panel is never quietly smaller than declared (spec 14.3)")
        ordered = sorted(frame, key=lambda item_id: rank_key(split_salt, f"sub:{panel.name}:{item_id}"))
        out[panel.name] = frozenset(ordered[:panel.size])
    return out


# --------------------------------------------------------------------------- hashes, summary


def source_inventory_hash(rows: Sequence[tables.SourceItemRow]) -> str:
    """``study.locks.source_inventory_hash``: the identity of the built item table."""
    return identity.sha256_hex(identity.jcs(
        [[row.source_item_id, row.normalized_text_hash, row.split_role, row.grader_hash]
         for row in sorted(rows, key=lambda r: r.source_item_id)]))


def prior_use_inventory_hash(records: Sequence[PriorUse]) -> str:
    """``study.locks.prior_use_inventory_hash``: the identity of the prior-use evidence."""
    return identity.sha256_hex(identity.jcs(
        sorted([[r.source_item_id, r.root, r.status, r.match_method] for r in records])))


def split_counts(rows: Sequence[tables.ItemSplitRow]) -> dict[str, int]:
    counts = Counter(row.split_role for row in rows)
    return {role: int(counts.get(role, 0)) for role in SPLIT_ROLES}


# --------------------------------------------------------------------------- the verb


def main(argv: Sequence[str]) -> Report:
    """``inventory``: build the item store, the splits and the prior-use evidence, then record
    the two inventory hashes in ``datasets.lock.json`` (spec 17.4, 17.5)."""
    from agents_scaling.nextdyn import items_build
    from agents_scaling.nextdyn.config import lock_paths
    from agents_scaling.nextdyn.locks import hf_home, load_datasets_lock, write_datasets_lock

    parser = common_parser("inventory", description="build the public item store and the splits")
    parser.add_argument("--hf-home", type=Path, default=None, help="HF cache (default $HF_HOME)")
    args = parser.parse_args(list(argv))
    cfg = load_study(args.config)
    contract = cfg.contract
    run_root = resolve_run_root(args)
    split_salt = bytes.fromhex(contract.split_salt_hex)
    items_build.build_items(run_root, cfg, split_salt=split_salt, study_seed=contract.study_seed,
                            hf_home=args.hf_home or hf_home())
    summary = items_build.load_summary(run_root)
    report = Report(completed=1, detail=(f"{summary['n_items']} items over "
                                         f"{len(contract.families)} cohort families -> "
                                         f"{I.public_dir(run_root)}"))
    paths = lock_paths(args.config)
    if paths.datasets.exists():
        lock = D.record_inventory_hashes(
            load_datasets_lock(paths.datasets),
            source_inventory_hash=str(summary["source_inventory_hash"]),
            prior_use_inventory_hash=str(summary["prior_use_inventory_hash"]))
        write_datasets_lock(paths.datasets, lock)
        report = report + Report(detail=f"inventory hashes -> {paths.datasets}")
    else:
        report = report + Report(blocked=1, detail=f"{paths.datasets} is absent; run lock-inputs")
    return report


__all__ = [
    "ARD36_FAMILY_MAP", "BENCHMARK_FAMILY_MAP", "FrameAssignment", "MATCH_NONE",
    "MATCH_REPLAY_INFERRED", "MATCH_REPLAY_VERIFIED", "MATCH_SOURCE_ID", "MATCH_TEXT_HASH",
    "NearDuplicate", "PILOT_STAGE_BASE", "PILOT_STAGE_NONE", "PriorUse", "SHINGLE_JACCARD",
    "SHINGLE_N", "SPLIT_BOTH", "SPLIT_CONSTRUCTED", "SPLIT_EXCLUDED_DUPLICATE",
    "SPLIT_EXCLUDED_INELIGIBLE", "SPLIT_NATURAL", "SPLIT_PILOT", "SPLIT_RESERVE", "SPLIT_ROLES",
    "STATUS_ORDER", "STATUS_RESERVED_UNUSED", "STATUS_UNKNOWN", "STATUS_USED_INFERRED",
    "STATUS_USED_VERIFIED", "SubPanel", "assign_split_roles", "main", "near_duplicate_audit",
    "prior_use_inventory", "prior_use_inventory_hash", "prior_use_status", "source_inventory_hash",
    "split_counts", "sub_panels",
]
