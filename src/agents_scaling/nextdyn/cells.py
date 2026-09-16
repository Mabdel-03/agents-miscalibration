"""Frozen block manifests: what one dispatchable cell is, and one builder per block.

Plan §3.3 (kinds, lanes, cells), §3.4 ("**Frozen manifests, no waves.** ``freeze`` writes
every block manifest once"), §1.4 (the ``plan`` table), spec 8.3, 10.1, 16.3, 17.2
(randomized dispatch), 19.1 (the pilot subset).

Three ARD36 defects are structural here:

* **D-11 / D-04 (one directory, two cohorts).**  ``cell_kind`` is the first component of every
  cell id (:func:`cell_id_for`), and :class:`ManifestBuilder` refuses a second cell with an id
  it has already seen, across every block of one build.  A bank cell id can therefore never
  equal an episode cell id, and two manifests of one module can never share a directory
  (``test_two_manifests_of_the_same_module_cannot_share_a_cell_id``).
* **Waves.**  ARD36 emitted a manifest per wave and kept an ``INDEX.json`` ledger of them, so
  "which manifest is authoritative" was a runtime question.  Here every block manifest is
  emitted **once at freeze** with path-addressed ``bundle:<family>/<bucket>`` dependencies:
  the bucket cell exists from t0 and waits for its bundle, and ``plan --dry-run`` reproduces
  the same manifest bytes (``release-check: replan_reproduces_manifests``).
* **A label in the manifest.**  ``CellSpec.options`` is validated against a per-kind allowlist
  **and** refuses any key that names a composition, an eligibility, a C0 or a correctness
  token (``ard/cells.py:72``'s regex idea), so a manifest can never carry the construction.

Order (spec 17.2): cells are ordered by ``ids.blind(study_seed, ORDER_DISPATCH, cell_id)``
*within each condition* and then round-robined across conditions, so any chunk of a block
manifest carries every condition and a partial run is never one arm's worth of data.  The
order changes dispatch only: every seed is a pure function of the seed key.

No number lives here (``test_no_numeric_dispatch_constant_lives_in_code``): panel sizes,
replicate counts and quotas all arrive through :class:`PlanState`, which ``plan.py`` builds
from ``study.yaml``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import deps, ids, kinds
from agents_scaling.nextdyn.errors import ManifestError
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.nextdyn.kinds import Lane, Pool
from agents_scaling.study import identity

MANIFEST_SCHEMA_VERSION: Final[int] = 1
MANIFESTS_DIR: Final[tuple[str, ...]] = ("plan", "manifests")
SHA_SUFFIX: Final[str] = ".sha256"

#: Cell-id components are dot separated; ``deps._CELL_ID_RE`` admits exactly this alphabet.
ID_SEP: Final[str] = "."
NONE_TOKEN: Final[str] = "x"

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
#: An option key (or string value) that would put the label-conditioned construction into a
#: manifest.  ``ard/cells.py:72`` refuses the same shapes on a team tag.
#: KEYS: a manifest key that *names* the construction or a correctness fact.  Broad on purpose:
#: an option called ``correct_slots`` is the defect whatever it holds.
_LEAKY_KEY_RE = re.compile(r"(?i)(c0|composition|eligib|correct|wrong|gold|grade|w_?star|"
                           r"semantic|label)")
#: VALUES: a *value* that spells out the construction.  Narrower than the key rule, because a
#: published condition name legitimately says what the receiver's assigned draft was
#: (``E2B_1_4_CHECK_CORRECT`` is spec 9.1's arm, a declared treatment, not a hidden label);
#: what may never appear is a composition name, a C0, a w*, a gold reference or a grade.
_LEAKY_VALUE_RE = re.compile(r"(?i)(c[0-9]+_(coherent|fragmented)|\bc0\b|composition|eligib|"
                             r"\bgold\b|grades?_|w_?star|semantic_correct)")


# --------------------------------------------------------------------------- blocks

class Block:
    """The block names ``freeze`` emits a manifest for (plan §3.4).

    Strings rather than an enum member per block because a block name is also a file name and
    a ``plan --block <X> --dry-run`` argument; the tuple below is the closed set.
    """

    BANKS: Final[str] = "BANKS"
    NATURAL_ROOTS: Final[str] = "NATURAL_ROOTS"
    E1_NATURAL: Final[str] = "E1_NATURAL"
    E1_CONSTRUCTED: Final[str] = "E1_CONSTRUCTED"
    E1_FRAGMENTED: Final[str] = "E1_FRAGMENTED"
    E1_BRIDGE: Final[str] = "E1_BRIDGE"
    E3A: Final[str] = "E3A"
    E3A_PROTECTION: Final[str] = "E3A_PROTECTION"
    E3A_MATCHED_SLOT: Final[str] = "E3A_MATCHED_SLOT"
    E2A: Final[str] = "E2A"
    E2B: Final[str] = "E2B"
    AUDIT: Final[str] = "AUDIT"
    CAPTURE: Final[str] = "CAPTURE"
    GRADE: Final[str] = "GRADE"
    CONSTRUCT: Final[str] = "CONSTRUCT"
    E0: Final[str] = "E0"
    MATERIALIZE: Final[str] = "MATERIALIZE"
    E3B_RING: Final[str] = "E3B_RING"
    E3B_SIZE: Final[str] = "E3B_SIZE"
    E3C_REPLAY: Final[str] = "E3C_REPLAY"
    E3C_FAMILY: Final[str] = "E3C_FAMILY"


#: The core blocks, in DAG order (plan §3.4); every one is emitted at freeze.
CORE_BLOCKS: Final[tuple[str, ...]] = (
    Block.BANKS, Block.NATURAL_ROOTS, Block.GRADE, Block.CONSTRUCT, Block.E1_NATURAL,
    Block.E1_CONSTRUCTED, Block.E1_FRAGMENTED, Block.E1_BRIDGE, Block.E3A, Block.E3A_PROTECTION,
    Block.E3A_MATCHED_SLOT, Block.E2A, Block.E2B, Block.AUDIT, Block.CAPTURE, Block.E0,
    Block.MATERIALIZE,
)
#: Expressible, never dispatched by default (plan D11); ``plan --block <X> --dry-run`` must
#: build them with zero engine changes, which is the release gate.
EXTENSION_BLOCKS: Final[tuple[str, ...]] = (
    Block.E3B_RING, Block.E3B_SIZE, Block.E3C_REPLAY, Block.E3C_FAMILY,
)
BLOCKS: Final[tuple[str, ...]] = CORE_BLOCKS + EXTENSION_BLOCKS

STAGE_SMOKE: Final[str] = "smoke"
STAGE_PILOT: Final[str] = "pilot"
STAGE_CONFIRMATION: Final[str] = "confirmation"
STAGES: Final[tuple[str, ...]] = (STAGE_SMOKE, STAGE_PILOT, STAGE_CONFIRMATION)


# --------------------------------------------------------------------------- options

#: kind -> the option keys a cell of that kind may carry.  An option outside its kind's set is
#: a planning error, not a hint: the worker would ignore it and the manifest would lie.
OPTION_ALLOWLIST: Final[Mapping[CellKind, frozenset[str]]] = {
    CellKind.DIFFICULTY_BANK: frozenset({"extension_stage", "role"}),
    CellKind.DONOR_BANK: frozenset({"extension_stage", "role"}),
    CellKind.NATURAL_ROOT: frozenset({"extension_stage", "role", "draws"}),
    CellKind.EPISODE: frozenset({"bucket", "arm", "origin", "protocol", "roots", "focal_slot",
                                 "revisions", "n_agents", "budget"}),
    CellKind.RECEIVER: frozenset({"assay", "arm", "budget", "panel", "problems", "packets",
                                  "replicates"}),
    CellKind.SELECTION_AUDIT: frozenset({"pool", "scope", "episodes"}),
    CellKind.GRADE: frozenset({"cells", "stage"}),
    CellKind.CONSTRUCT: frozenset({"bucket", "pair", "cells", "stage"}),
    CellKind.CAPTURE: frozenset({"cells", "panel", "model", "blocks", "anchors", "staging",
                                 "device_map", "max_batch_tokens", "fidelity_tokens",
                                 "snapshot_path", "model_hash", "shard_name", "chunk_rows",
                                 "group_size"}),
    CellKind.E0: frozenset({"task", "shard", "n_shards"}),
    CellKind.MATERIALIZE: frozenset({"scope"}),
}
if set(OPTION_ALLOWLIST) != set(CellKind):                      # pragma: no cover - import-time
    raise RuntimeError("OPTION_ALLOWLIST must cover every CellKind")


def assert_option_is_label_free(key: str, value: Any) -> None:
    """Refuse an option key or string value that encodes the construction (spec 5.3, R7)."""
    if _LEAKY_KEY_RE.search(str(key)):
        raise ManifestError(
            f"cell option {key!r} names a label-conditioned fact (composition / eligibility / "
            "C0 / correctness); the construction lives in root_teams_private, never in a "
            "manifest the generation lane reads")
    if isinstance(value, str) and _LEAKY_VALUE_RE.search(value):
        raise ManifestError(f"cell option {key}={value!r} encodes a label-conditioned fact")
    if isinstance(value, (list, tuple)):
        for item in value:
            assert_option_is_label_free(key, item)


@dataclass(frozen=True)
class ConditionRef:
    """The condition a cell dispatches, as the *manifest* records it (plan §1.5).

    ``protocol_hash`` is the ``ConditionSpec.condition_hash`` and the only part that reaches a
    key; the label is for humans and for ``decisions.display_label``.  ``composition`` names
    which paired member a constructed bucket cell runs, which is a *planning* fact -- it never
    travels to a prompt, a seed or a bundle.
    """

    protocol_hash: str
    display_label: str
    composition: str | None = None
    budget_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.protocol_hash:
            raise ManifestError("a ConditionRef names its condition hash")
        if not self.display_label:
            raise ManifestError("a ConditionRef carries the human label of its arm")

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_hash": self.protocol_hash, "display_label": self.display_label,
                "composition": self.composition, "budget_hash": self.budget_hash}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConditionRef":
        unknown = sorted(set(data) - {"protocol_hash", "display_label", "composition", "budget_hash"})
        if unknown:
            raise ManifestError(f"condition carries unknown key(s) {unknown}")
        return cls(protocol_hash=str(data["protocol_hash"]), display_label=str(data["display_label"]),
                   composition=data.get("composition"), budget_hash=data.get("budget_hash"))


# --------------------------------------------------------------------------- the cell


@dataclass(frozen=True)
class CellSpec:
    """One dispatchable unit of work (contract §2.5).

    Everything a lane driver, a runner and the ``plan`` table need, and nothing a prompt could
    read.  ``depends_on`` is the ``deps.py`` vocabulary and is checked against the kind's
    declared ``dep_prefixes``; ``reuse_of`` names the cell whose requests this one would alias
    byte-for-byte (a declared ``equivalent_to`` condition, plan §1.3), which the planner emits
    instead of a second dispatch.
    """

    cell_id: str
    kind: CellKind
    lane: Lane
    pool: Pool | None
    model: str | None
    family: str
    tokens_hash: str
    condition: ConditionRef | None
    items: tuple[str, ...]
    replicate: int
    split_role: str
    depends_on: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    reuse_of: str | None = None
    block: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ids.coerce_kind(self.kind))
        object.__setattr__(self, "lane", Lane(self.lane))
        object.__setattr__(self, "pool", None if self.pool is None else Pool(self.pool))
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "options", dict(self.options))
        if not deps._CELL_ID_RE.match(self.cell_id):
            raise ManifestError(f"cell id {self.cell_id!r} is not a valid directory token")
        if not self.cell_id.startswith(self.kind.value + ID_SEP):
            raise ManifestError(f"cell id {self.cell_id!r} must start with its kind "
                                f"{self.kind.value!r} (R4: cell_kind is the first component)")
        if not self.items:
            raise ManifestError(f"{self.cell_id}: a cell with no items is never dispatched")
        if len(set(self.items)) != len(self.items):
            raise ManifestError(f"{self.cell_id}: an item appears twice")
        if self.block and self.block not in BLOCKS:
            raise ManifestError(f"{self.cell_id}: unknown block {self.block!r}")
        spec = kinds.spec_for(self.kind)
        if spec.needs_endpoint and self.model is None:
            raise ManifestError(f"{self.cell_id}: {self.kind.value} generates and needs a model cell")
        allowed = OPTION_ALLOWLIST[self.kind]
        for key, value in self.options.items():
            assert_option_is_label_free(key, value)
            if key not in allowed:
                raise ManifestError(
                    f"{self.cell_id}: option {key!r} is not one of the {self.kind.value} options "
                    f"{sorted(allowed)}; an option the worker never reads is a manifest that lies")
        for dep in self.depends_on:
            parsed = deps.parse_dependency(dep)
            prefix = {"cell": deps.CELL_PREFIX, "bundle": deps.BUNDLE_PREFIX,
                      "register": deps.REGISTER_PREFIX}[parsed.kind]
            if prefix not in spec.dep_prefixes:
                raise ManifestError(
                    f"{self.cell_id}: {self.kind.value} cells may depend on {list(spec.dep_prefixes)}, "
                    f"not on {dep!r}")

    # -- views ---------------------------------------------------------------------
    @property
    def protocol_hash(self) -> str:
        return "" if self.condition is None else self.condition.protocol_hash

    @property
    def display_label(self) -> str | None:
        return None if self.condition is None else self.condition.display_label

    @property
    def n_items(self) -> int:
        return len(self.items)

    def option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "kind": self.kind.value,
            "lane": self.lane.value,
            "pool": None if self.pool is None else self.pool.value,
            "model": self.model,
            "family": self.family,
            "tokens_hash": self.tokens_hash,
            "condition": None if self.condition is None else self.condition.to_dict(),
            "items": list(self.items),
            "replicate": int(self.replicate),
            "split_role": self.split_role,
            "depends_on": list(self.depends_on),
            "options": dict(self.options),
            "reuse_of": self.reuse_of,
            "block": self.block,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CellSpec":
        known = {f for f in cls.__dataclass_fields__} | {"kind"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ManifestError(f"cell {data.get('cell_id')!r} carries unknown key(s) {unknown}")
        condition = data.get("condition")
        return cls(cell_id=str(data["cell_id"]), kind=data["kind"], lane=data["lane"],
                   pool=data.get("pool"), model=data.get("model"), family=str(data["family"]),
                   tokens_hash=str(data["tokens_hash"]),
                   condition=None if condition is None else ConditionRef.from_dict(condition),
                   items=tuple(data["items"]), replicate=int(data["replicate"]),
                   split_role=str(data["split_role"]), depends_on=tuple(data.get("depends_on", ())),
                   options=dict(data.get("options", {})), reuse_of=data.get("reuse_of"),
                   block=str(data.get("block", "")))


def _token(value: Any) -> str:
    """One cell-id component; ``None`` becomes the reserved ``x``."""
    if value is None or value == "":
        return NONE_TOKEN
    text = str(value)
    if not _TOKEN_RE.match(text):
        raise ManifestError(f"cell-id component {value!r} is not a token ({_TOKEN_RE.pattern})")
    return text


def cell_id_for(kind: CellKind | str, *, family: str | None, model: str | None,
                tokens_hash8: str | None, condition_hash12: str | None, split_role: str | None,
                replicate: int, shard: int, extra: Sequence[str] = ()) -> str:
    """``<kind>.<family>.<model>.<budget8>.<cond12>.<split>.r<k>[.<extra>].s<shard>``.

    Kind first (R4), shard last, and every other component a bounded token, so two cells of
    different kinds -- or of the same kind and different conditions, buckets or replicates --
    can never collide, and a bank cell id can never equal an episode cell id.
    """
    cell_kind = ids.coerce_kind(kind)
    parts = [
        cell_kind.value,
        _token(family),
        _token(model),
        _token(None if not tokens_hash8 else str(tokens_hash8)[:8]),
        _token(None if not condition_hash12 else str(condition_hash12)[:12]),
        _token(split_role),
        f"r{int(replicate)}",
        *(_token(e) for e in extra),
        f"s{int(shard)}",
    ]
    return ID_SEP.join(parts)


# --------------------------------------------------------------------------- manifests


def manifest_path(run_root: str | os.PathLike, block: str, name: str | None = None) -> Path:
    """``<run_root>/plan/manifests/<block>[.<name>].json``."""
    if block not in BLOCKS:
        raise ManifestError(f"unknown block {block!r}; blocks: {list(BLOCKS)}")
    stem = block if name is None else f"{block}.{_token(name)}"
    return Path(run_root).joinpath(*MANIFESTS_DIR) / f"{stem}.json"


def manifest_bytes(cells: Sequence[CellSpec], *, block: str = "", meta: Mapping[str, Any] | None = None) -> bytes:
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "block": block,
        "meta": dict(meta or {}),
        "cells": [c.to_dict() for c in cells],
    }
    return (json.dumps(payload, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False)
            + "\n").encode("utf-8")


def write_manifest(path: str | os.PathLike, cells: Sequence[CellSpec], *, block: str = "",
                   meta: Mapping[str, Any] | None = None) -> str:
    """Write a block manifest + ``.sha256`` sidecar; return the sha.

    Refuses a differing overwrite: a manifest is a frozen artifact, so a re-plan that produces
    different bytes is a *finding* (``replan_reproduces_manifests``), never an edit.  Writing
    identical bytes again is idempotent, which is what makes ``freeze`` re-runnable.
    """
    path = Path(path)
    raw = manifest_bytes(cells, block=block, meta=meta)
    sha = identity.sha256_hex(raw)
    sidecar = path.with_name(path.name + SHA_SUFFIX)
    if path.exists():
        if identity.sha256_hex(path.read_bytes()) == sha:
            if not sidecar.exists():
                deps.atomic_write_bytes(sidecar, f"{sha}  {path.name}\n".encode("utf-8"))
            return sha
        raise ManifestError(
            f"{path} exists with different content ({identity.sha256_hex(path.read_bytes())[:12]} "
            f"!= {sha[:12]}); a block manifest is written once (plan §3.4: no waves)")
    deps.atomic_write_bytes(path, raw)
    deps.atomic_write_bytes(sidecar, f"{sha}  {path.name}\n".encode("utf-8"))
    return sha


def manifest_sha(path: str | os.PathLike) -> str:
    """The sha of a manifest on disk, verified against its sidecar."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"{path} does not exist")
    actual = identity.sha256_hex(path.read_bytes())
    sidecar = path.with_name(path.name + SHA_SUFFIX)
    if not sidecar.exists():
        raise ManifestError(f"{sidecar} is missing; a manifest without its sidecar is not frozen")
    declared = sidecar.read_text(encoding="utf-8").split()
    if not declared or declared[0].lower() != actual:
        raise ManifestError(f"{path}: sha256 {actual[:12]} != sidecar "
                            f"{(declared[0][:12] if declared else '')}")
    return actual


def load_manifest(path: str | os.PathLike) -> list[CellSpec]:
    """Load a block manifest after verifying its sidecar (ARD36's `load_cells` rule, hardened:
    the sidecar is mandatory, never optional)."""
    path = Path(path)
    manifest_sha(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or "cells" not in data:
        raise ManifestError(f"{path}: expected a nextdyn block manifest with a 'cells' list")
    if int(data.get("schema_version", 0)) != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"{path}: manifest schema_version {data.get('schema_version')!r} != "
                            f"{MANIFEST_SCHEMA_VERSION}")
    return [CellSpec.from_dict(cell) for cell in data["cells"]]


def load_manifest_meta(path: str | os.PathLike) -> Mapping[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(data.get("meta", {}))


# --------------------------------------------------------------------------- ordering


def _condition_key(cell: CellSpec) -> str:
    """What "every condition" means for interleaving: the arm, the composition and the budget.

    A block whose cells carry no condition (banks, grades) has one group, so the round robin
    degenerates to the blind order, which is what spec 17.2 asks for there.
    """
    if cell.condition is None:
        return f"{cell.kind.value}|{cell.model or ''}"
    return "|".join([cell.condition.display_label, cell.condition.composition or "",
                     cell.condition.budget_hash or "", cell.model or ""])


def order_cells(cells: Sequence[CellSpec], *, study_seed: bytes) -> list[CellSpec]:
    """Blind order within each condition, round-robin across conditions (spec 17.2).

    Any prefix -- and therefore any chunk a lane driver admits -- carries every condition of
    the block, so a run that stops early stops with balanced arms rather than with one arm
    finished and another untouched.
    """
    groups: dict[str, list[CellSpec]] = {}
    for cell in cells:
        groups.setdefault(_condition_key(cell), []).append(cell)
    ordered_groups = [
        sorted(group, key=lambda c: ids.blind(study_seed, K.ORDER_DISPATCH, c.cell_id))
        for _key, group in sorted(groups.items())
    ]
    out: list[CellSpec] = []
    index = 0
    while any(index < len(group) for group in ordered_groups):
        for group in ordered_groups:
            if index < len(group):
                out.append(group[index])
        index += 1
    return out


class ManifestBuilder:
    """Collects cells for one build and refuses a duplicate id (defect D-11 / D-04).

    The builder is per *build*, not per block: the ids of every block of one ``freeze`` pass
    are checked against one another, so a bank cell and an episode cell can never land in the
    same directory even when two different builders make the same mistake.
    """

    def __init__(self, *, study_seed: bytes) -> None:
        self.study_seed = bytes(study_seed)
        self._by_block: dict[str, list[CellSpec]] = {}
        self._ids: dict[str, str] = {}

    def add(self, cell: CellSpec, *, block: str | None = None) -> CellSpec:
        name = block or cell.block
        if name not in BLOCKS:
            raise ManifestError(f"{cell.cell_id}: unknown block {name!r}")
        if name != cell.block:
            cell = replace(cell, block=name)
        seen = self._ids.get(cell.cell_id)
        if seen is not None:
            raise ManifestError(
                f"duplicate cell id {cell.cell_id!r} (already emitted by block {seen}); two "
                "manifests would share one directory and one would overwrite the other's "
                "meta.json (defect D-04)")
        self._ids[cell.cell_id] = name
        self._by_block.setdefault(name, []).append(cell)
        return cell

    def extend(self, cells: Iterable[CellSpec], *, block: str | None = None) -> None:
        for cell in cells:
            self.add(cell, block=block)

    @property
    def blocks(self) -> tuple[str, ...]:
        return tuple(name for name in BLOCKS if name in self._by_block)

    def cells(self, block: str) -> list[CellSpec]:
        """The block's cells in dispatch order (blind, condition-interleaved)."""
        return order_cells(self._by_block.get(block, ()), study_seed=self.study_seed)

    def all_cells(self) -> list[CellSpec]:
        return [cell for block in self.blocks for cell in self.cells(block)]

    def study_meta(self, contract: K.StudyContract) -> dict[str, Any]:
        """What every manifest this builder writes says about the run it belongs to."""
        return {"study_id": contract.study_id, "n_cells": len(self._ids)}

    def write(self, run_root: str | os.PathLike, *, meta: Mapping[str, Any] | None = None,
              content_addressed: bool = False) -> dict[str, str]:
        """Write every block manifest once; return ``{block: sha}``.

        ``content_addressed`` names the file after its own bytes.  The planner needs it: the
        blocks whose dependency is a *register* (AUDIT, CAPTURE) can only be emitted once the
        register exists, and a later pass that sees more completed episodes emits a genuinely
        different set.  Naming each by its content keeps "written once" literally true -- a
        pass that sees the same completions rewrites the same bytes at the same path -- without
        an INDEX ledger deciding which file is authoritative (plan §3.4).
        """
        out: dict[str, str] = {}
        for block in self.blocks:
            cells = self.cells(block)
            name = None
            if content_addressed:
                name = identity.sha256_hex(manifest_bytes(cells, block=block, meta=meta))[:12]
            out[block] = write_manifest(manifest_path(run_root, block, name), cells, block=block,
                                        meta=meta)
        return out


# --------------------------------------------------------------------------- plan state


@dataclass(frozen=True)
class Bucket:
    """One constructed-team bucket: the k-th paired-eligible item of its family.

    The bucket index is a position in the frozen scan order, so ``bundle:<family>/<bucket>``
    is knowable at freeze even though the bundle itself is written hours later (plan §3.4).
    """

    family: str
    index: int
    source_item_id: str
    pair: str
    #: The *panel* the item came from (MATH500 / MATH_EXPANDED), which is the directory
    #: ``construct.seal_root_teams`` writes the bundle under.  ``family`` is the **cohort**
    #: the cell is planned for (MATH), and the two differ for every multi-panel cohort
    #: (``families.TASK_FAMILY``); wiring the dependency to the cohort would name a path
    #: nothing ever writes.
    panel_family: str = ""

    @property
    def token(self) -> str:
        return deps.bucket_name(self.index)

    @property
    def bundle_family(self) -> str:
        return self.panel_family or self.family


@dataclass(frozen=True)
class PlanState:
    """Every input a block builder reads, resolved before any cell exists.

    ``plan.py`` builds it from ``study.yaml`` plus the inventory; the pilot subset is the same
    object with smaller panels (spec 19.1), which is why no builder knows what stage it is in.
    """

    stage: str
    contract: K.StudyContract
    cfg: Any                                   # config.StudyConfig
    compute: Any                               # config.ComputeConfig
    registry: Any                              # protocols.Registry | None
    families: tuple[str, ...]
    items: Mapping[str, tuple[str, ...]]       # family -> bank/scan order
    natural_items: Mapping[str, tuple[str, ...]]
    buckets: Mapping[str, tuple[Bucket, ...]]
    panels: Mapping[str, Mapping[str, tuple[str, ...]]]   # panel name -> family -> items
    problems: tuple[str, ...] = ()             # E2b mod97 problem ids
    #: family -> {item: register sha}.  One register per **item**, because the receiver
    #: episode id is built from the panel id (contract §2.0) and a panel shared by many items
    #: would give two items one episode id.  E2b already works this way: its panel id is the
    #: mod97 problem id.  An item is present only if its donor bank supplied a correct *and* a
    #: wrong draw (spec 9.1), so the sealed set is a subset of the sub-panel.
    packet_registers: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    pool_registers: tuple[tuple[str, str], ...] = ()      # (item, register sha)
    panel_registers: Mapping[str, str] = field(default_factory=dict)    # family -> register sha
    episode_cells: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # family -> cell ids
    root_cells: Mapping[str, tuple[str, ...]] = field(default_factory=dict)    # family -> NATURAL_ROOT cell ids
    #: Which grading pass these GRADE cells are: the banks are graded at freeze and the
    #: episodes as they complete, and the tag keeps the two sets' cell ids disjoint.
    grade_tag: str = ""
    grade_targets: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # family -> cell ids

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ManifestError(f"unknown stage {self.stage!r}; stages: {list(STAGES)}")

    def panel(self, name: str, family: str) -> tuple[str, ...]:
        return tuple(self.panels.get(name, {}).get(family, ()))


#: Panel names ``item_splits`` carries (plan §1.4) and the builders read.
PANEL_SECOND_ROOT: Final[str] = "second_root"
PANEL_BRIDGE: Final[str] = "bridge"
PANEL_FRAGMENTED: Final[str] = "fragmented"
PANEL_E3A: Final[str] = "e3a"
PANEL_E3A_FILL: Final[str] = "e3a_fill"
PANEL_CAPTURE: Final[str] = "capture"
PANEL_CAPTURE_NATURAL: Final[str] = "capture_natural"
PANEL_E2A: Final[str] = "e2a"
PANEL_NATPROT: Final[str] = "natprot"
PANELS: Final[tuple[str, ...]] = (
    PANEL_SECOND_ROOT, PANEL_BRIDGE, PANEL_FRAGMENTED, PANEL_E3A, PANEL_E3A_FILL,
    PANEL_CAPTURE, PANEL_CAPTURE_NATURAL, PANEL_E2A, PANEL_NATPROT,
)


# --------------------------------------------------------------------------- helpers


def _chunks(items: Sequence[str], size: int) -> list[tuple[str, ...]]:
    if size < 1:
        raise ManifestError("a cell holds at least one item")
    return [tuple(items[i:i + size]) for i in range(0, len(items), size)]


def _shape(state: PlanState, kind: CellKind, *, model: K.ModelCell | None = None,
           grade_resources: Mapping[str, Any] | None = None) -> kinds.CellShape:
    return kinds.cell_shape(kind, compute=state.compute, model=model,
                            thinking_cap=state.contract.tokens.thinking_cap,
                            team=state.contract.team, grade_resources=grade_resources)


def _pool(state: PlanState, kind: CellKind, model: K.ModelCell | None) -> Pool | None:
    return kinds.pool_for(kind, model, state.compute)


def _episode_condition(state: PlanState, name: str) -> Any:
    """The :class:`VisibilitySpec` of one protocol name, or a refusal naming the registry."""
    registry = state.registry
    if registry is None:
        raise ManifestError("this block needs protocols.yaml; pass the loaded registry in PlanState")
    try:
        return registry.episodes[name]
    except KeyError as exc:
        raise ManifestError(f"protocols.yaml declares no episode condition {name!r}") from exc


def _receiver_condition(state: PlanState, name: str) -> Any:
    registry = state.registry
    if registry is None:
        raise ManifestError("this block needs protocols.yaml; pass the loaded registry in PlanState")
    try:
        return registry.receivers[name]
    except KeyError as exc:
        raise ManifestError(f"protocols.yaml declares no receiver condition {name!r}") from exc


def _condition_ref(spec: Any, *, composition: str | None = None,
                   budget_hash: str | None = None) -> ConditionRef:
    return ConditionRef(protocol_hash=spec.condition_hash, display_label=spec.display_label,
                        composition=composition, budget_hash=budget_hash)


# --------------------------------------------------------------------------- block builders


def build_banks(state: PlanState) -> list[CellSpec]:
    """DIFFICULTY_BANK + DONOR_BANK(stage 32) over the frozen scan order (plan §3.4 t0).

    The stage-64 extension is not emitted here: which items need it is a *graded* fact, so the
    planner emits those cells when the stage-32 GRADE cell completes.
    """
    contract = state.contract
    model = contract.model(_bank_model(state))
    out: list[CellSpec] = []
    for kind in (CellKind.DIFFICULTY_BANK, CellKind.DONOR_BANK):
        shape = _shape(state, kind, model=model)
        for family in state.families:
            for shard, items in enumerate(_chunks(state.items.get(family, ()), shape.items_per_cell)):
                out.append(CellSpec(
                    cell_id=cell_id_for(kind, family=family, model=model.name,
                                        tokens_hash8=contract.tokens.budget_hash(),
                                        condition_hash12=None, split_role=_split_of(state, family),
                                        replicate=0, shard=shard),
                    kind=kind, lane=kinds.lane_for(kind, model), pool=_pool(state, kind, model),
                    model=model.name, family=family, tokens_hash=contract.tokens.budget_hash(),
                    condition=None, items=items, replicate=0, split_role=_split_of(state, family),
                    options={"extension_stage": 0}, block=Block.BANKS))
    return out


def build_natural_roots(state: PlanState) -> list[CellSpec]:
    """NATURAL_ROOT draws for the natural cohort (spec 8.1); no dependency, runs at t0."""
    contract = state.contract
    model = contract.model(_bank_model(state))
    kind = CellKind.NATURAL_ROOT
    shape = _shape(state, kind, model=model)
    out: list[CellSpec] = []
    for family in state.families:
        for shard, items in enumerate(_chunks(state.natural_items.get(family, ()), shape.items_per_cell)):
            out.append(CellSpec(
                cell_id=cell_id_for(kind, family=family, model=model.name,
                                    tokens_hash8=contract.tokens.budget_hash(),
                                    condition_hash12=None, split_role=_split_of(state, family),
                                    replicate=0, shard=shard),
                kind=kind, lane=kinds.lane_for(kind, model), pool=_pool(state, kind, model),
                model=model.name, family=family, tokens_hash=contract.tokens.budget_hash(),
                condition=None, items=items, replicate=0, split_role=_split_of(state, family),
                options={"draws": int(contract.banks.natural_root_draws)},
                block=Block.NATURAL_ROOTS))
    return out


def _episode_cells(state: PlanState, *, block: str, protocols: Sequence[str],
                   items_of: Callable[[str], Sequence[str]], origin: str,
                   bucket_of: Callable[[str, str], Bucket | None] | None = None,
                   compositions: Sequence[str] = (), replicates: int | None = None,
                   arm_option: str | None = None,
                   model_name: str | None = None) -> list[CellSpec]:
    """The one EPISODE cell builder every E1/E3a block composes.

    A constructed block passes ``bucket_of`` so each cell depends on exactly its
    ``bundle:<family>/<bucket>``; a natural block passes none and depends on the natural-root
    cells instead.  ``reuse_of`` is filled by ``plan.py`` (it needs the whole cell set).
    """
    contract = state.contract
    model = contract.model(model_name or _episode_model(state))
    kind = CellKind.EPISODE
    shape = _shape(state, kind, model=model)
    out: list[CellSpec] = []
    for family in state.families:
        items = tuple(items_of(family))
        if not items:
            continue
        for protocol in protocols:
            spec = _episode_condition(state, protocol)
            for member, composition in enumerate(compositions or (None,)):
                # The id carries the member *index*, never the composition name: a cell id is a
                # directory name and it travels into every option that names this cell (the
                # capture panel's ``cells`` list), where a composition name would be the
                # label-conditioned construction in a manifest (spec 5.3).  The name itself
                # lives on ``ConditionRef.composition``, which is a planning field.
                member_token = f"m{member}" if composition else ""
                for replicate in range(_replicates(state, replicates)):
                    condition = _condition_ref(spec, composition=composition,
                                               budget_hash=contract.tokens.budget_hash())
                    if bucket_of is None:
                        for shard, chunk in enumerate(_chunks(items, shape.items_per_cell)):
                            options: dict[str, Any] = {"origin": origin}
                            if arm_option:
                                options["arm"] = arm_option
                            out.append(CellSpec(
                                cell_id=cell_id_for(
                                    kind, family=family, model=model.name,
                                    tokens_hash8=contract.tokens.budget_hash(),
                                    condition_hash12=spec.condition_hash, split_role=_split_of(state, family),
                                    replicate=replicate, shard=shard,
                                    extra=(origin,) + ((member_token,) if composition else ())),
                                kind=kind, lane=kinds.lane_for(kind, model),
                                pool=_pool(state, kind, model), model=model.name, family=family,
                                tokens_hash=contract.tokens.budget_hash(), condition=condition,
                                items=chunk, replicate=replicate, split_role=_split_of(state, family),
                                depends_on=_natural_root_deps(state, family, chunk),
                                options=options, block=block))
                    else:
                        for shard, item in enumerate(items):
                            bucket = bucket_of(family, item)
                            if bucket is None:
                                continue
                            options = {"origin": origin, "bucket": bucket.token}
                            if arm_option:
                                options["arm"] = arm_option
                            out.append(CellSpec(
                                cell_id=cell_id_for(
                                    kind, family=family, model=model.name,
                                    tokens_hash8=contract.tokens.budget_hash(),
                                    condition_hash12=spec.condition_hash, split_role=_split_of(state, family),
                                    replicate=replicate, shard=shard,
                                    extra=(origin, bucket.token)
                                    + ((member_token,) if composition else ())),
                                kind=kind, lane=kinds.lane_for(kind, model),
                                pool=_pool(state, kind, model), model=model.name, family=family,
                                tokens_hash=contract.tokens.budget_hash(), condition=condition,
                                items=(item,), replicate=replicate, split_role=_split_of(state, family),
                                depends_on=(deps.bundle_dependency(bucket.bundle_family,
                                                                  bucket.token),),
                                options=options, block=block))
    return out


def build_e1_natural(state: PlanState) -> list[CellSpec]:
    """E1 on the natural cohort: the core protocol set on the natural-root teams (spec 8.1)."""
    return _episode_cells(state, block=Block.E1_NATURAL, protocols=state.cfg.protocol.protocols,
                          items_of=lambda f: state.natural_items.get(f, ()), origin="natural")


def build_e1_constructed(state: PlanState) -> list[CellSpec]:
    """E1 on the constructed cohort: one cell per (bucket, composition, protocol) (spec 8.2)."""
    return _episode_cells(
        state, block=Block.E1_CONSTRUCTED, protocols=state.cfg.protocol.protocols,
        items_of=lambda f: tuple(b.source_item_id for b in state.buckets.get(f, ())),
        origin="constructed", bucket_of=_bucket_of(state),
        compositions=state.cfg.cohorts.paired_compositions)


def build_e1_fragmented(state: PlanState) -> list[CellSpec]:
    """The fragmented-minority panel: one composition, the two primary arms (spec 8.4)."""
    protocols = _primary_arms(state)
    return _episode_cells(
        state, block=Block.E1_FRAGMENTED, protocols=protocols,
        items_of=lambda f: _panel_buckets(state, f, PANEL_FRAGMENTED), origin="fragmented",
        bucket_of=_bucket_of(state), compositions=(state.cfg.cohorts.fragmented_composition,))


def build_e1_bridge(state: PlanState) -> list[CellSpec]:
    """The answer-only bridge panel (spec 8.5): PEER vs PEER_ANSWER_ONLY on one sub-panel."""
    bridge = state.cfg.experiments.E1.bridge_protocol
    protocols = (_peer_arm(state), bridge)
    return _episode_cells(
        state, block=Block.E1_BRIDGE, protocols=protocols,
        items_of=lambda f: _panel_buckets(state, f, PANEL_BRIDGE), origin="bridge",
        bucket_of=_bucket_of(state))


def build_e3a(state: PlanState) -> list[CellSpec]:
    """E3a directed channels: the four declared arms plus both filled arms (spec 10.1-10.3)."""
    e3a = state.cfg.experiments.E3a
    out = _episode_cells(
        state, block=Block.E3A, protocols=tuple(e3a.arms),
        items_of=lambda f: _panel_buckets(state, f, PANEL_E3A)[:int(e3a.target_items_per_task)],
        origin="directed", bucket_of=_bucket_of(state))
    out += _episode_cells(
        state, block=Block.E3A, protocols=tuple(e3a.filled_arms),
        items_of=lambda f: _panel_buckets(state, f, PANEL_E3A_FILL), origin="directed_filled",
        bucket_of=_bucket_of(state))
    return out


def build_e3a_protection(state: PlanState) -> list[CellSpec]:
    """The natural random-protection arm (spec 10.4): one protected slot, chosen blind."""
    e3a = state.cfg.experiments.E3a
    if not e3a.natural_random_protection:
        return []
    return _episode_cells(state, block=Block.E3A_PROTECTION, protocols=(e3a.protection_protocol,),
                          items_of=lambda f: state.natural_items.get(f, ()), origin="protection")


def build_e3a_matched_slot(state: PlanState) -> list[CellSpec]:
    """The matched-slot diagnostic: both filled arms on the constructed matched sub-panel."""
    e3a = state.cfg.experiments.E3a
    return _episode_cells(
        state, block=Block.E3A_MATCHED_SLOT, protocols=tuple(e3a.filled_arms),
        items_of=lambda f: _panel_buckets(state, f, PANEL_E3A_FILL), origin="matched_slot",
        bucket_of=_bucket_of(state), arm_option="matched_slot")


def assert_grid_is_the_declared_one(state: PlanState, arms: Sequence[tuple[str, Any]]) -> None:
    """The arms built are the arms ``study.yaml`` declares (spec 17.4's E2b grid).

    ``experiments.E2b.{endorsement, checker_access, initial_draft}`` is the declared product
    and ``protocols.yaml`` instantiates it, *plus* the draft-only controls spec 9.5 asks for --
    so the registry legitimately offers more arms than the grid has cells.  What may never
    happen is the other direction: a grid the study declared that nobody instantiates would be
    a treatment the run reports and never dispatches ("both ends built, no middle").
    """
    e2b = state.cfg.experiments.E2b
    declared = len(e2b.endorsement) * len(e2b.checker_access) * len(e2b.initial_draft)
    if len(arms) < declared:
        raise ManifestError(
            f"experiments.E2b declares a grid of {declared} ({list(e2b.endorsement)} x "
            f"{list(e2b.checker_access)} x {list(e2b.initial_draft)}) but protocols.yaml "
            f"instantiates only {len(arms)} two-phase arms")


def build_e2a(state: PlanState) -> list[CellSpec]:
    """E2a: one-call receiver episodes over the natural fixed-packet pairs (spec 9.1).

    The packets are a *register* (``register:packets/<sha>``), because the pair is fixed by a
    writer-side pass over graded donors and the receiver must replay exactly those bytes.  One
    cell is one item: the register is the item's, and the receiver's episode id is built from
    it (contract §2.0), so two items can never share an episode.
    """
    cfg = state.cfg
    e2a = cfg.experiments.E2a
    contract = state.contract
    kind = CellKind.RECEIVER
    out: list[CellSpec] = []
    for arm, spec in _receiver_arms(state, phases=1):
        for model_name in e2a.models:
            model = contract.model(model_name)
            for family in e2a.tasks:
                # the E2a panel is ``experiments.E2a.items_per_task`` items of this task, and
                # an item is in it only if the writer could freeze a correct/wrong pair for it
                sealed = dict(state.packet_registers.get(family, {}))
                items = [item for item in _panel_items(state, family, PANEL_E2A) if item in sealed]
                budget = _first_budget(state)
                budget_hash = _receiver_budget_hash(state, budget)
                for replicate in range(int(e2a.replicates)):
                    for shard, item in enumerate(items[:int(e2a.items_per_task)]):
                        register = sealed[item]
                        out.append(CellSpec(
                            cell_id=cell_id_for(kind, family=family, model=model.name,
                                                tokens_hash8=budget_hash,
                                                condition_hash12=spec.condition_hash,
                                                split_role=_split_of(state, family),
                                                replicate=replicate, shard=shard, extra=("e2a",)),
                            kind=kind, lane=kinds.lane_for(kind, model),
                            pool=_pool(state, kind, model), model=model.name, family=family,
                            tokens_hash=budget_hash,
                            condition=_condition_ref(spec, budget_hash=budget_hash),
                            items=(item,), replicate=replicate,
                            split_role=_split_of(state, family),
                            depends_on=(deps.register_dependency(deps.REGISTER_PACKETS,
                                                                 register),),
                            options={"assay": ASSAY_E2A, "arm": arm, "budget": budget,
                                     "packets": register},
                            block=Block.E2A))
    return out


def build_e2b(state: PlanState) -> list[CellSpec]:
    """E2b: the synthetic mod-97 receiver grid, at both budgets, on both models (spec 9.2-9.5).

    E2b depends on nothing: its problems are generated from the study seed, so the whole block
    starts at t0 (plan §3.4).
    """
    cfg = state.cfg
    e2b = cfg.experiments.E2b
    contract = state.contract
    kind = CellKind.RECEIVER
    out: list[CellSpec] = []
    arms = _receiver_arms(state, phases=e2b.generation_phases)
    assert_grid_is_the_declared_one(state, arms)
    # the declared panel size, so a manifest can never quietly dispatch a different one
    problems = tuple(state.problems)[:int(e2b.source_problems)]
    for arm, spec in arms:
        for model_name in e2b.models:
            model = contract.model(model_name)
            shape = _shape(state, kind, model=model)
            for budget in cfg.budgets.receiver_thinking_tokens:
                budget_hash = _receiver_budget_hash(state, budget)
                for replicate in range(int(e2b.replicates)):
                    for shard, chunk in enumerate(_chunks(problems, shape.items_per_cell)):
                        out.append(CellSpec(
                            cell_id=cell_id_for(kind, family=_E2B_FAMILY, model=model.name,
                                                tokens_hash8=budget_hash,
                                                condition_hash12=spec.condition_hash,
                                                split_role=_E2B_SPLIT, replicate=replicate,
                                                shard=shard, extra=("e2b", f"b{int(budget)}")),
                            kind=kind, lane=kinds.lane_for(kind, model), pool=_pool(state, kind, model),
                            model=model.name, family=_E2B_FAMILY, tokens_hash=budget_hash,
                            condition=_condition_ref(spec, budget_hash=budget_hash), items=chunk,
                            replicate=replicate, split_role=_E2B_SPLIT,
                            options={"assay": ASSAY_E2B, "arm": arm, "budget": int(budget)},
                            block=Block.E2B))
    return out


def build_audit(state: PlanState) -> list[CellSpec]:
    """SELECTION_AUDIT over sealed pools (spec 6.5); one cell per pool-register batch.

    The pools are sealed by the planner once the episodes complete, so the register sha is a
    runtime fact and this builder runs on the planner's pass, not at freeze.
    """
    contract = state.contract
    model = contract.model(contract.selection.audit_model)
    kind = CellKind.SELECTION_AUDIT
    shape = _shape(state, kind, model=model)
    budget_hash = contract.audit_tokens.budget_hash()
    out: list[CellSpec] = []
    by_register: dict[str, list[str]] = {}
    for item, register in state.pool_registers:
        by_register.setdefault(register, []).append(item)
    for shard, (register, items) in enumerate(sorted(by_register.items())):
        for chunk_index, chunk in enumerate(_chunks(tuple(dict.fromkeys(items)), shape.items_per_cell)):
            split_role = _split_of(state, "")
            out.append(CellSpec(
                cell_id=cell_id_for(kind, family=None, model=model.name, tokens_hash8=budget_hash,
                                    condition_hash12=register, split_role=split_role, replicate=0,
                                    shard=chunk_index, extra=(f"p{shard}",)),
                kind=kind, lane=kinds.lane_for(kind, model), pool=_pool(state, kind, model),
                model=model.name, family="", tokens_hash=budget_hash, condition=None, items=chunk,
                replicate=0, split_role=split_role,
                depends_on=(deps.register_dependency(deps.REGISTER_POOL, register),),
                options={"pool": register, "scope": contract.selection.pool_scope},
                block=Block.AUDIT))
    return out


def build_capture(state: PlanState) -> list[CellSpec]:
    """Read-only replay capture of the sealed panel (spec 13; plan §3.3 capture lane)."""
    cfg = state.cfg
    if not cfg.instrumentation.enabled:
        return []
    contract = state.contract
    model = contract.model(_episode_model(state))
    kind = CellKind.CAPTURE
    shape = _shape(state, kind, model=model)
    out: list[CellSpec] = []
    for family in state.families:
        items = _panel_items(state, family, PANEL_CAPTURE)[:int(cfg.instrumentation.target_items_per_task)]
        register = state.panel_registers.get(family)
        if not items or register is None:
            continue
        cells = tuple(state.episode_cells.get(family, ()))
        for shard, chunk in enumerate(_chunks(items, shape.items_per_cell)):
            out.append(CellSpec(
                cell_id=cell_id_for(kind, family=family, model=model.name,
                                    tokens_hash8=contract.tokens.budget_hash(),
                                    condition_hash12=register, split_role=_split_of(state, family),
                                    replicate=0, shard=shard),
                kind=kind, lane=kinds.lane_for(kind, None), pool=None, model=model.name,
                family=family, tokens_hash=contract.tokens.budget_hash(), condition=None,
                items=chunk, replicate=0, split_role=_split_of(state, family),
                depends_on=(deps.register_dependency(deps.REGISTER_PANEL, register),)
                + tuple(deps.cell_dependency(c) for c in cells),
                options={"panel": register, "cells": list(cells), "model": model.name},
                block=Block.CAPTURE))
    return out


def build_grade(state: PlanState) -> list[CellSpec]:
    """GRADE cells over the completed generation cells of one family (cpu lane, WRITER_SIDE)."""
    kind = CellKind.GRADE
    out: list[CellSpec] = []
    for family in state.families:
        cells = tuple(state.grade_targets.get(family, ()))
        items = tuple(dict.fromkeys(state.items.get(family, ())))
        if not cells or not items:
            continue
        shape = _shape(state, kind, grade_resources=_grade_resources(state, family))
        tag = (state.grade_tag,) if state.grade_tag else ()
        for shard, chunk in enumerate(_chunks(items, shape.items_per_cell)):
            out.append(CellSpec(
                cell_id=cell_id_for(kind, family=family, model=None, tokens_hash8=None,
                                    condition_hash12=None, split_role=_split_of(state, family),
                                    replicate=0, shard=shard, extra=tag),
                kind=kind, lane=kinds.lane_for(kind, None), pool=None, model=None, family=family,
                tokens_hash="", condition=None, items=chunk, replicate=0,
                split_role=_split_of(state, family),
                depends_on=tuple(deps.cell_dependency(c) for c in cells),
                options={"cells": list(cells), "stage": state.grade_tag or "banks"},
                block=Block.GRADE))
    return out


def build_construct(state: PlanState) -> list[CellSpec]:
    """One CONSTRUCT cell per bucket: it writes the bucket's ``bundle:<family>/<bucket>``."""
    kind = CellKind.CONSTRUCT
    out: list[CellSpec] = []
    for family in state.families:
        cells = tuple(state.grade_targets.get(family, ()))
        for bucket in state.buckets.get(family, ()):
            out.append(CellSpec(
                cell_id=cell_id_for(kind, family=family, model=None, tokens_hash8=None,
                                    condition_hash12=None, split_role=_split_of(state, family),
                                    replicate=0, shard=bucket.index, extra=(bucket.token,)),
                kind=kind, lane=kinds.lane_for(kind, None), pool=None, model=None, family=family,
                tokens_hash="", condition=None, items=(bucket.source_item_id,), replicate=0,
                split_role=_split_of(state, family),
                depends_on=tuple(deps.cell_dependency(c) for c in cells),
                options={"bucket": bucket.token, "pair": bucket.pair}, block=Block.CONSTRUCT))
    return out


def build_e0(state: PlanState) -> list[CellSpec]:
    """E0 legacy reanalysis: the registry pass plus the BCB behavioral-grouping array."""
    cfg = state.cfg
    if not cfg.experiments.E0.enabled:
        return []
    kind = CellKind.E0
    out = [CellSpec(
        cell_id=cell_id_for(kind, family=None, model=None, tokens_hash8=None,
                            condition_hash12=None, split_role=None, replicate=0, shard=0,
                            extra=("legacy",)),
        kind=kind, lane=kinds.lane_for(kind, None), pool=None, model=None, family="",
        tokens_hash="", condition=None, items=(E0_ITEM,), replicate=0, split_role="",
        options={"task": "legacy"}, block=Block.E0)]
    n_shards = int(cfg.experiments.E0.bcb_array_cells)
    for shard in range(n_shards):
        out.append(CellSpec(
            cell_id=cell_id_for(kind, family=None, model=None, tokens_hash8=None,
                                condition_hash12=None, split_role=None, replicate=0, shard=shard,
                                extra=("bcb",)),
            kind=kind, lane=kinds.lane_for(kind, None), pool=None, model=None, family="",
            tokens_hash="", condition=None, items=(E0_ITEM,), replicate=0, split_role="",
            options={"task": "bcb_repair", "shard": shard, "n_shards": n_shards}, block=Block.E0))
    return out


def build_materialize(state: PlanState) -> list[CellSpec]:
    """The single close-out cell: it depends on every other cell (plan §3.4)."""
    kind = CellKind.MATERIALIZE
    return [CellSpec(
        cell_id=cell_id_for(kind, family=None, model=None, tokens_hash8=None,
                            condition_hash12=None, split_role=None, replicate=0, shard=0),
        kind=kind, lane=kinds.lane_for(kind, None), pool=None, model=None, family="",
        tokens_hash="", condition=None, items=(MATERIALIZE_ITEM,), replicate=0, split_role="",
        options={"scope": "all"}, block=Block.MATERIALIZE)]


# --------------------------------------------------------------------------- extensions


def build_e3b_ring(state: PlanState) -> list[CellSpec]:
    """E3b rings: the ring protocols on the constructed buckets, at the declared horizon.

    Expressible with zero engine changes (plan D11): a ring is a ``protocols.yaml`` entry and
    the horizon is a cell option, so this builder composes the same EPISODE kind.
    """
    e3b = state.cfg.experiments.E3b
    cells = _episode_cells(
        state, block=Block.E3B_RING, protocols=tuple(e3b.ring_protocols),
        items_of=lambda f: tuple(b.source_item_id for b in state.buckets.get(f, ())),
        origin="ring", bucket_of=_bucket_of(state))
    # the horizon and the rounds it is scored at travel together: a ring run to R=8 whose
    # score rounds still stopped at 4 would be a horizon nobody measured (spec 11.3)
    scored = max(int(r) for r in e3b.horizon_score_rounds)
    return [replace(c, options={**c.options,
                                "revisions": max(int(e3b.horizon_rounds), scored)})
            for c in cells]


def build_e3b_size(state: PlanState) -> list[CellSpec]:
    """E3b team sizes N in {3, 9}: the same protocols at another ``n_agents`` (config only)."""
    e3b = state.cfg.experiments.E3b
    out: list[CellSpec] = []
    for n in e3b.team_sizes:
        cells = _episode_cells(
            state, block=Block.E3B_SIZE, protocols=state.cfg.protocol.protocols,
            items_of=lambda f: tuple(b.source_item_id for b in state.buckets.get(f, ())),
            origin=f"n{int(n)}", bucket_of=_bucket_of(state))
        out += [replace(c, options={**c.options, "n_agents": int(n)}) for c in cells]
    return out


def build_e3c_replay(state: PlanState) -> list[CellSpec]:
    """E3c: the same teams replayed by the larger model (one ``model`` change, plan D11).

    ``native_init`` is the arm that would draw the larger model's *own* roots instead of
    replaying the 8B team's; it is disabled in v1 and the cells carry the flag so the two
    cannot be confused after the fact.
    """
    e3c = state.cfg.experiments.E3c
    cells = _episode_cells(
        state, block=Block.E3C_REPLAY, protocols=state.cfg.protocol.protocols,
        items_of=lambda f: tuple(b.source_item_id for b in state.buckets.get(f, ())),
        origin="replay", bucket_of=_bucket_of(state), model_name=e3c.model)
    return [replace(c, options={**c.options, "roots": "native" if e3c.native_init else "replayed"})
            for c in cells]


def build_e3c_family(state: PlanState) -> list[CellSpec]:
    """E3C_FAMILY: the envelope-adapter companions (Qwen3 thinking-off; Llama blocked)."""
    out: list[CellSpec] = []
    for model_name in state.cfg.experiments.E3C_FAMILY.models:
        out += _episode_cells(
            state, block=Block.E3C_FAMILY, protocols=_primary_arms(state),
            items_of=lambda f: tuple(b.source_item_id for b in state.buckets.get(f, ())),
            origin="family", bucket_of=_bucket_of(state), model_name=model_name)
    return out


#: block -> builder.  ``plan.build_dag`` walks this table, so a new block is one entry here
#: plus a DAG row -- never a change to the runner, the drivers or the engine (plan §1.10).
BUILDERS: Final[Mapping[str, Callable[[PlanState], list[CellSpec]]]] = {
    Block.BANKS: build_banks,
    Block.NATURAL_ROOTS: build_natural_roots,
    Block.E1_NATURAL: build_e1_natural,
    Block.E1_CONSTRUCTED: build_e1_constructed,
    Block.E1_FRAGMENTED: build_e1_fragmented,
    Block.E1_BRIDGE: build_e1_bridge,
    Block.E3A: build_e3a,
    Block.E3A_PROTECTION: build_e3a_protection,
    Block.E3A_MATCHED_SLOT: build_e3a_matched_slot,
    Block.E2A: build_e2a,
    Block.E2B: build_e2b,
    Block.AUDIT: build_audit,
    Block.CAPTURE: build_capture,
    Block.GRADE: build_grade,
    Block.CONSTRUCT: build_construct,
    Block.E0: build_e0,
    Block.MATERIALIZE: build_materialize,
    Block.E3B_RING: build_e3b_ring,
    Block.E3B_SIZE: build_e3b_size,
    Block.E3C_REPLAY: build_e3c_replay,
    Block.E3C_FAMILY: build_e3c_family,
}
if set(BUILDERS) != set(BLOCKS):                                # pragma: no cover - import-time
    raise RuntimeError("every block needs exactly one builder")


def build_block(block: str, state: PlanState) -> list[CellSpec]:
    """The cells of one block, unordered (``ManifestBuilder`` orders them)."""
    try:
        builder = BUILDERS[block]
    except KeyError as exc:
        raise ManifestError(f"unknown block {block!r}; blocks: {list(BLOCKS)}") from exc
    return builder(state)


# --------------------------------------------------------------------------- small readers
# (kept at the bottom: they are one-line reads of PlanState / the config, and putting them
# above the builders would hide the builders behind plumbing.)

#: The synthetic item id an E0 / MATERIALIZE cell carries: those cells run once over the whole
#: run root and are deliberately not items (``ard/types.ANALYSIS_ITEM`` plays the same role).
E0_ITEM: Final[str] = "e0"
MATERIALIZE_ITEM: Final[str] = "all"
#: The assay names are WP5's (``receiver.ASSAYS``): one vocabulary, one declaration (R1).
ASSAY_E2A: Final[str] = "E2a"
ASSAY_E2B: Final[str] = "E2b"
_E2B_FAMILY: Final[str] = "MOD97"
_E2B_SPLIT: Final[str] = "synthetic"


def _bank_model(state: PlanState) -> str:
    """Banks run on the primary (8B) model only: the 32B banks E3c would need are disabled."""
    return _first_enabled_model(state, lane=Lane.SMALL.value)


def _episode_model(state: PlanState) -> str:
    return _first_enabled_model(state, lane=Lane.SMALL.value)


def _first_enabled_model(state: PlanState, *, lane: str) -> str:
    for name, block in state.cfg.models.items():
        if block.enabled and block.lane == lane:
            return name
    raise ManifestError(f"study.yaml enables no model on lane {lane!r}")


def _split_of(state: PlanState, family: str) -> str:
    return STAGE_PILOT if state.stage == STAGE_PILOT else "confirmation"


def _replicates(state: PlanState, replicates: int | None) -> int:
    """Episodes have ONE replicate unless a block asks for more (plan §3.1's call table).

    The spec's E1 count is ``items x compositions x protocols x n x R``: the repetition is the
    five root agents and the R rounds, not a repeated episode.  The receiver assays *do* carry
    a replicate (``experiments.E2a.replicates`` / ``E2b.replicates``) and pass it explicitly.
    """
    del state
    return int(replicates) if replicates is not None else 1


def _primary_arms(state: PlanState) -> tuple[str, ...]:
    """The two arms every differential gate is computed between (H1's contrast)."""
    protocols = tuple(state.cfg.protocol.protocols)
    return protocols[:2]


def _peer_arm(state: PlanState) -> str:
    return _primary_arms(state)[1]


def _bucket_of(state: PlanState) -> Callable[[str, str], Bucket | None]:
    index = {(b.family, b.source_item_id): b for f in state.buckets for b in state.buckets[f]}
    return lambda family, item: index.get((family, item))


def _panel_buckets(state: PlanState, family: str, panel: str) -> tuple[str, ...]:
    """The panel's items, restricted to the ones that have a constructed bucket."""
    have = {b.source_item_id for b in state.buckets.get(family, ())}
    return tuple(item for item in _panel_items(state, family, panel) if item in have)


def _panel_items(state: PlanState, family: str, panel: str) -> tuple[str, ...]:
    return state.panel(panel, family)


def _natural_root_deps(state: PlanState, family: str, items: Sequence[str]) -> tuple[str, ...]:
    """A natural episode cell waits for the NATURAL_ROOT cells holding its items."""
    del items
    return tuple(deps.cell_dependency(c) for c in state.root_cells.get(family, ()))


def _receiver_arms(state: PlanState, *, phases: int) -> list[tuple[str, Any]]:
    registry = state.registry
    if registry is None:
        raise ManifestError("the receiver blocks need protocols.yaml; pass the loaded registry")
    return [(name, spec) for name, spec in sorted(registry.receivers.items())
            if int(spec.phases) == int(phases) and getattr(spec, "enabled", True)]


def _first_budget(state: PlanState) -> int:
    return int(state.cfg.budgets.receiver_thinking_tokens[0])


def _receiver_budget_hash(state: PlanState, budget: int) -> str:
    return state.contract.receiver_tokens[f"{int(budget)}/one_call"].budget_hash()


def _grade_resources(state: PlanState, family: str) -> Mapping[str, Any] | None:
    try:
        from agents_scaling.nextdyn import families as F
    except ImportError:                                         # pragma: no cover - WP3 pending
        return None
    # ``family`` is the cohort; its grade resources are the panel's (families.TASK_FAMILY).
    panels = F.panels_of(family) if family in set(F.TASK_FAMILY.values()) else ()
    spec = F.FAMILIES.get(panels[0]) if panels else F.FAMILIES.get(family)
    if spec is None:
        return None
    res = spec.grade_resources
    return {"cpus": res.cpus, "mem": res.mem, "time": res.time,
            "items_per_cell": res.items_per_cell}


__all__ = [
    "BLOCKS", "BUILDERS", "Block", "Bucket", "CORE_BLOCKS", "CellSpec", "ConditionRef",
    "ASSAY_E2A", "ASSAY_E2B", "EXTENSION_BLOCKS", "E0_ITEM", "MANIFESTS_DIR", "MANIFEST_SCHEMA_VERSION",
    "MATERIALIZE_ITEM", "ManifestBuilder", "OPTION_ALLOWLIST", "PANELS", "PANEL_BRIDGE",
    "PANEL_CAPTURE", "PANEL_CAPTURE_NATURAL", "PANEL_E2A", "PANEL_E3A", "PANEL_E3A_FILL",
    "PANEL_FRAGMENTED", "PANEL_NATPROT", "PANEL_SECOND_ROOT", "PlanState", "STAGES",
    "STAGE_CONFIRMATION", "STAGE_PILOT", "STAGE_SMOKE", "assert_option_is_label_free",
    "build_block", "cell_id_for", "load_manifest", "load_manifest_meta", "manifest_bytes",
    "manifest_path", "manifest_sha", "order_cells", "write_manifest",
]
