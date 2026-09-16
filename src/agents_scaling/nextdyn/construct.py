"""The CONSTRUCT service: composition-driven root teams and their label-free export
(spec 8.2, 8.3, 8.6, 10.1, 14.3; plan §2.4; **writer side**, R7).

ARD36 built its constructed teams inside the episode loop, from the same bank it measured
difficulty on, and exported the composition alongside the team, so a solver could in
principle have read which slot was the minority one.  Here construction is a *cell*: it runs
on the cpu lane, its hard input is the donor bank's grades and its hard output is a
``root_teams`` row plus a label-free bundle that the generation lane depends on by
``bundle:<family>/<bucket>``.

The rules, all frozen before any data (plan §2.4):

* **eligibility** -- a composition is buildable when the largest correct class holds at least
  ``c0`` resolved draws and one wrong class holds at least ``n - c0`` (a fragmented pattern
  needs its per-class counts); a *pair* is eligible only when **both** members are, so C1 and
  C4 are built on the same item and the same ``w*`` or not at all;
* **``w*``** -- ``argmin sha256(JCS([construct_seed, item, class]))`` over the qualifying wrong
  classes: a hash, not a size, so the choice cannot follow the outcome;
* **donor order** -- every class is ordered by ``blind_order_key(seed, "ND_DONOR_ORDER", item,
  class, uid)``, so ``c0`` and ``w0`` are shared across the pair by construction and the shared
  uids are recorded in ``root_teams.shared_donor_uids``;
* **slots** -- a separate blind permutation per team, seeded by the team's *pre-permutation*
  content id, so slot numbering carries no correctness signal; ``focal_slot`` is the unique
  minority slot when ``min(c0, n - c0) == 1`` and ``None`` otherwise (E3a refuses the rest);
* **validation** -- ``validate_team`` asserts ``n`` **distinct generation uids** (four copies of
  one draw are not four draws) and exactly ``c0`` correct members;
* **sealing** -- ``construction_hash`` is content-addressed over every input that determined
  the team, and a sealed team is never mutated: a different team set is a new bucket.

The exported bundle carries ordered uids, decision ids and public outputs, the opaque
``team_tag`` and the focal slot -- no composition, no ``C0``, no ``w*``, no grade
(:func:`agents_scaling.nextdyn.bundles.assert_label_free` re-checks that on the way in).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import yaml

from agents_scaling.nextdyn import banks as BK
from agents_scaling.nextdyn import bundles as BU
from agents_scaling.nextdyn import deps
from agents_scaling.nextdyn import ids as ID
from agents_scaling.nextdyn import tables
from agents_scaling.nextdyn.config import CONFIG_DIR
from agents_scaling.nextdyn.contract import ORDER_DONOR_ORDER, ORDER_PROTECT, ORDER_SLOT, ORDER_WSTAR
from agents_scaling.nextdyn.errors import ContractViolation, ManifestError
from agents_scaling.nextdyn.ids import CellKind
from agents_scaling.nextdyn.items_build import permutation_from_digest
from agents_scaling.study import identity

COMPOSITIONS_FILE: Final[str] = "compositions.yaml"
PATTERN_COHERENT: Final[str] = "coherent"
PATTERN_FRAGMENTED: Final[str] = "fragmented"

COHORT_CONSTRUCTED: Final[str] = "constructed"
COHORT_NATURAL: Final[str] = "natural"

STATE_CORRECT: Final[str] = BK.STATE_CORRECT
STATE_WRONG: Final[str] = BK.STATE_WRONG
STATE_UNKNOWN: Final[str] = BK.STATE_UNKNOWN

#: The natural-team composition label (protected): a natural team has no composition, and
#: saying so explicitly keeps ``root_teams_private.composition`` non-null for every team.
NATURAL_COMPOSITION: Final[str] = "NATURAL"


# --------------------------------------------------------------------------- compositions.yaml


@dataclass(frozen=True)
class WrongPattern:
    """How the ``n - c0`` wrong members are spread over wrong classes."""

    kind: str
    fragmented: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fragmented", tuple(int(c) for c in self.fragmented))
        if self.kind not in (PATTERN_COHERENT, PATTERN_FRAGMENTED):
            raise ContractViolation(f"unknown wrong_pattern {self.kind!r}")
        if (self.kind == PATTERN_FRAGMENTED) != bool(self.fragmented):
            raise ContractViolation("a fragmented wrong_pattern carries its per-class counts")
        if any(count < 1 for count in self.fragmented):
            raise ContractViolation("every fragmented class holds at least one member")

    def counts(self, n_wrong: int) -> tuple[int, ...]:
        """Members per wrong class, the first being ``w*``."""
        if self.kind == PATTERN_COHERENT:
            return (int(n_wrong),)
        if sum(self.fragmented) != int(n_wrong):
            raise ContractViolation(f"fragmented pattern {self.fragmented} sums to "
                                    f"{sum(self.fragmented)}, not {n_wrong}")
        return self.fragmented

    @classmethod
    def parse(cls, value: Any) -> "WrongPattern":
        if isinstance(value, str):
            return cls(kind=value)
        if isinstance(value, Mapping) and set(value) == {PATTERN_FRAGMENTED}:
            return cls(kind=PATTERN_FRAGMENTED, fragmented=tuple(value[PATTERN_FRAGMENTED]))
        raise ContractViolation(f"wrong_pattern must be 'coherent' or {{fragmented: [...]}}, "
                                f"got {value!r}")


@dataclass(frozen=True)
class Composition:
    """One root-team composition (``compositions.yaml``)."""

    name: str
    n: int
    c0: int
    wrong_pattern: WrongPattern
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.n < 1 or not 0 <= self.c0 <= self.n:
            raise ContractViolation(f"composition {self.name}: n={self.n}, c0={self.c0} is not a "
                                    "team shape")
        self.wrong_pattern.counts(self.n_wrong)

    @property
    def n_wrong(self) -> int:
        return int(self.n) - int(self.c0)

    @property
    def has_focal_slot(self) -> bool:
        """A directed arm needs one unique minority slot (spec 10.1)."""
        return min(int(self.c0), self.n_wrong) == 1

    @property
    def minority_state(self) -> str | None:
        if not self.has_focal_slot:
            return None
        return STATE_CORRECT if int(self.c0) == 1 else STATE_WRONG


@dataclass(frozen=True)
class CompositionSet:
    """``compositions.yaml`` compiled: the compositions, the declared pairs, the extension rule."""

    compositions: Mapping[str, Composition]
    pairs: Mapping[str, tuple[str, ...]]
    extension_rule: BK.ExtensionRule
    construction_rule_version: str

    def composition(self, name: str) -> Composition:
        try:
            return self.compositions[name]
        except KeyError as exc:
            raise ContractViolation(f"unknown composition {name!r}; declared: "
                                    f"{sorted(self.compositions)}") from exc

    def pair(self, name: str) -> tuple[Composition, ...]:
        try:
            members = self.pairs[name]
        except KeyError as exc:
            raise ContractViolation(f"unknown composition pair {name!r}; declared: "
                                    f"{sorted(self.pairs)}") from exc
        return tuple(self.composition(member) for member in members)


def study_compositions(cfg: Any, comps: CompositionSet) -> tuple[tuple[Composition, ...], Composition]:
    """The paired panel and the fragmented composition the *study* declares, cross-checked.

    ``compositions.yaml`` declares the shapes; ``study.yaml`` declares which of them this run
    recruits for (``cohorts.paired_compositions``, ``cohorts.fragmented_composition``).  Two
    files naming the same fact is exactly the ARD36 D-05 shape, so nothing reads one without
    the other: this resolves the study's names against the composition set, refuses a name the
    set does not declare or has disabled, and refuses a pair whose members disagree on ``n`` or
    share a ``c0`` (a paired panel is the mirrored challenge of spec 8.2, not two copies).
    """
    cohorts = cfg.cohorts
    names = tuple(str(n) for n in cohorts.paired_compositions)
    if len(names) != 2:
        raise ContractViolation(
            f"study.cohorts.paired_compositions must name exactly two compositions, got {list(names)}")
    pair = tuple(comps.composition(n) for n in names)
    disabled = [c.name for c in pair if not c.enabled]
    if disabled:
        raise ContractViolation(f"study.cohorts.paired_compositions names disabled composition(s) "
                                f"{disabled} (enable them in compositions.yaml or drop them here)")
    if pair[0].n != pair[1].n:
        raise ContractViolation(f"paired compositions {names} disagree on n: "
                                f"{pair[0].n} vs {pair[1].n}")
    if pair[0].c0 == pair[1].c0:
        raise ContractViolation(f"paired compositions {names} share c0={pair[0].c0}; a paired panel "
                                "is the mirrored challenge, not the same composition twice")
    fragmented = comps.composition(str(cohorts.fragmented_composition))
    if not fragmented.enabled:
        raise ContractViolation(
            f"study.cohorts.fragmented_composition names disabled composition {fragmented.name!r}")
    if fragmented.n != pair[0].n:
        raise ContractViolation(f"fragmented composition {fragmented.name} has n={fragmented.n}, "
                                f"but the paired panel is n={pair[0].n}")
    return pair, fragmented


def load_compositions(path: str | os.PathLike | None = None) -> CompositionSet:
    """Compile ``configs/nextdyn/compositions.yaml`` (unknown keys refused)."""
    source = Path(path) if path is not None else CONFIG_DIR / COMPOSITIONS_FILE
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(data) - {"compositions", "pairs", "extension_rule",
                                  "construction_rule_version"})
    if unknown:
        raise ContractViolation(f"{source}: unknown key(s) {unknown}")
    compositions: dict[str, Composition] = {}
    for name, block in (data.get("compositions") or {}).items():
        extra = sorted(set(block) - {"n", "c0", "wrong_pattern", "enabled"})
        if extra:
            raise ContractViolation(f"{source}: composition {name} has unknown key(s) {extra}")
        compositions[str(name)] = Composition(
            name=str(name), n=int(block["n"]), c0=int(block["c0"]),
            wrong_pattern=WrongPattern.parse(block["wrong_pattern"]),
            enabled=bool(block["enabled"]))
    pairs = {str(name): tuple(str(m) for m in members)
             for name, members in (data.get("pairs") or {}).items()}
    for name, members in pairs.items():
        missing = [m for m in members if m not in compositions]
        if missing:
            raise ContractViolation(f"{source}: pair {name} names undeclared composition(s) {missing}")
    rule_block = data.get("extension_rule") or {}
    rule = BK.ExtensionRule(name=str(rule_block["name"]),
                            min_correct_resolved=int(rule_block["min_correct_resolved"]),
                            min_wrong_resolved=int(rule_block["min_wrong_resolved"]))
    return CompositionSet(compositions=compositions, pairs=pairs, extension_rule=rule,
                          construction_rule_version=str(data["construction_rule_version"]))


# --------------------------------------------------------------------------- donors


@dataclass(frozen=True)
class Donor:
    """One graded, resolved bank draw available as a team member."""

    generation_uid: str
    decision_id: str
    source_item_id: str
    item_variant_id: str
    draw_index: int
    state: str
    canonical_class: str
    content_hash: str | None

    def __post_init__(self) -> None:
        if self.state not in (STATE_CORRECT, STATE_WRONG):
            raise ContractViolation(f"a donor is resolved C or W, not {self.state!r}")
        if not self.canonical_class:
            raise ContractViolation(f"{self.generation_uid} has no canonical class and cannot "
                                    "join a class")


@dataclass(frozen=True)
class DonorClasses:
    """The resolved donor draws of one item, grouped by canonical class and blind-ordered."""

    source_item_id: str
    item_variant_id: str
    correct: Mapping[str, tuple[Donor, ...]]
    wrong: Mapping[str, tuple[Donor, ...]]
    n_draws: int
    n_resolved: int
    n_correct: int
    n_wrong: int
    n_unknown: int

    @property
    def largest_correct(self) -> str | None:
        return _largest(self.correct)

    def wrong_classes_with(self, size: int) -> tuple[str, ...]:
        return tuple(sorted(name for name, members in self.wrong.items() if len(members) >= size))


def _largest(classes: Mapping[str, tuple[Donor, ...]]) -> str | None:
    """The biggest class; ties broken by the class name so the choice never depends on order."""
    if not classes:
        return None
    return sorted(classes, key=lambda name: (-len(classes[name]), name))[0]


def classify_donors(bank_rows: Sequence[tables.BankRow], grades: Mapping[str, str],
                    classes: Mapping[str, str | None], *, study_seed: bytes) -> DonorClasses:
    """Group the resolved draws of one item into correct and wrong canonical classes.

    Only ``C``/``W`` draws with a canonical class are donors: an unresolved or unclassed draw
    is counted and dropped, never coerced into a class (spec 4.6).
    """
    if not bank_rows:
        raise ContractViolation("classify_donors needs at least one bank draw")
    items = {(row.source_item_id, row.item_variant_id) for row in bank_rows}
    if len(items) != 1:
        raise ContractViolation(f"classify_donors is per item variant; got {sorted(items)}")
    roles = {row.bank_role for row in bank_rows}
    if roles != {BK.BankRole.DONOR.value}:
        raise ContractViolation(f"constructed teams draw from the donor bank only; got {sorted(roles)} "
                                "(difficulty draws are never donors, spec 5.3)")
    source_item_id, item_variant_id = items.pop()
    correct: dict[str, list[Donor]] = {}
    wrong: dict[str, list[Donor]] = {}
    counts = {STATE_CORRECT: 0, STATE_WRONG: 0, STATE_UNKNOWN: 0}
    for row in bank_rows:
        state = grades.get(row.generation_uid, STATE_UNKNOWN)
        canonical = classes.get(row.generation_uid)
        if state not in (STATE_CORRECT, STATE_WRONG) or not canonical:
            counts[STATE_UNKNOWN] += 1
            continue
        counts[state] += 1
        donor = Donor(generation_uid=row.generation_uid, decision_id=row.decision_id,
                      source_item_id=row.source_item_id, item_variant_id=row.item_variant_id,
                      draw_index=row.draw_index, state=state, canonical_class=canonical,
                      content_hash=row.content_hash)
        (correct if state == STATE_CORRECT else wrong).setdefault(canonical, []).append(donor)

    def _ordered(table: Mapping[str, list[Donor]]) -> dict[str, tuple[Donor, ...]]:
        return {name: tuple(sorted(members, key=lambda d: ID.blind(
            study_seed, ORDER_DONOR_ORDER, source_item_id, name, d.generation_uid)))
            for name, members in table.items()}

    return DonorClasses(
        source_item_id=source_item_id, item_variant_id=item_variant_id,
        correct=_ordered(correct), wrong=_ordered(wrong), n_draws=len(bank_rows),
        n_resolved=counts[STATE_CORRECT] + counts[STATE_WRONG], n_correct=counts[STATE_CORRECT],
        n_wrong=counts[STATE_WRONG], n_unknown=counts[STATE_UNKNOWN])


# --------------------------------------------------------------------------- eligibility


@dataclass(frozen=True)
class Eligibility:
    """Which compositions this item can support, and which declared pairs (plan §2.4)."""

    source_item_id: str
    per_composition: Mapping[str, bool]
    per_pair: Mapping[str, bool]
    detail: Mapping[str, str]

    def buildable(self, name: str) -> bool:
        return bool(self.per_composition.get(name, False))

    @property
    def any_pair(self) -> bool:
        return any(self.per_pair.values())


def composition_eligible(dc: DonorClasses, comp: Composition) -> tuple[bool, str]:
    """``(eligible, reason)`` for one composition on one item."""
    largest = dc.largest_correct
    have_correct = len(dc.correct.get(largest or "", ())) if largest else 0
    if have_correct < comp.c0:
        return False, f"largest correct class holds {have_correct} < c0={comp.c0}"
    counts = comp.wrong_pattern.counts(comp.n_wrong)
    sizes = sorted((len(members) for members in dc.wrong.values()), reverse=True)
    if len(sizes) < len(counts):
        return False, f"{len(sizes)} wrong classes < {len(counts)} the pattern needs"
    for index, need in enumerate(sorted(counts, reverse=True)):
        if sizes[index] < need:
            return False, f"wrong class {index} holds {sizes[index]} < {need}"
    return True, "eligible"


def eligibility(dc: DonorClasses, compositions: Sequence[Composition],
                pairs: Mapping[str, Sequence[str]] | None = None) -> Eligibility:
    """Per-composition and per-pair eligibility; a pair needs **both** members."""
    per_composition: dict[str, bool] = {}
    detail: dict[str, str] = {}
    for comp in compositions:
        ok, reason = composition_eligible(dc, comp)
        per_composition[comp.name] = ok
        detail[comp.name] = reason
    per_pair = {name: all(per_composition.get(member, False) for member in members)
                for name, members in (pairs or {}).items()}
    return Eligibility(source_item_id=dc.source_item_id, per_composition=per_composition,
                       per_pair=per_pair, detail=detail)


def choose_w_star(dc: DonorClasses, study_seed: bytes, source_item_id: str, *,
                  min_size: int) -> str | None:
    """``argmin sha256(JCS([construct_seed, item, class]))`` over the qualifying wrong classes."""
    qualifying = dc.wrong_classes_with(min_size)
    if not qualifying:
        return None
    return min(qualifying, key=lambda name: ID.blind(study_seed, ORDER_WSTAR, source_item_id, name))


@dataclass(frozen=True)
class SharedDonors:
    """The donors a declared pair shares (``root_teams.shared_donor_uids``)."""

    correct: tuple[str, ...] = ()
    wrong: tuple[str, ...] = ()

    def uids(self) -> frozenset[str]:
        return frozenset(self.correct) | frozenset(self.wrong)


def shared_donors(dc: DonorClasses, comps: Sequence[Composition], w_star: str) -> SharedDonors:
    """The prefix both members of a pair draw from: ``c0`` correct and ``w0`` wrong donors."""
    largest = dc.largest_correct or ""
    n_correct = min(int(comp.c0) for comp in comps) if comps else 0
    n_wrong = min(comp.wrong_pattern.counts(comp.n_wrong)[0] for comp in comps) if comps else 0
    return SharedDonors(
        correct=tuple(d.generation_uid for d in dc.correct.get(largest, ())[:n_correct]),
        wrong=tuple(d.generation_uid for d in dc.wrong.get(w_star, ())[:n_wrong]))


# --------------------------------------------------------------------------- teams


@dataclass(frozen=True)
class RootTeam:
    """One built team: public facts, protected facts and the content-addressed hash."""

    root_team_id: str
    team_tag: str
    family: str
    source_item_id: str
    item_variant_id: str
    cohort: str
    composition: str
    c0: int
    w_star_class: str | None
    correct_class: str | None
    members: tuple[Donor, ...]
    slot_states: tuple[str, ...]
    focal_slot: int | None
    construction_rule_version: str
    construction_hash: str
    shared_donor_uids: tuple[str, ...]
    bucket: str
    eligibility: Mapping[str, Any]

    @property
    def n(self) -> int:
        return len(self.members)

    @property
    def generation_uids(self) -> tuple[str, ...]:
        return tuple(member.generation_uid for member in self.members)

    @property
    def decision_ids(self) -> tuple[str, ...]:
        return tuple(member.decision_id for member in self.members)


def team_tag(study_seed: bytes, root_team_id: str) -> str:
    """The opaque name the runtime sees (spec 8.2: no composition in any id)."""
    return identity.sha256_hex(identity.jcs([study_seed.hex(), root_team_id]))[:16]


def _wrong_members(dc: DonorClasses, comp: Composition, w_star: str,
                   study_seed: bytes) -> tuple[Donor, ...]:
    counts = comp.wrong_pattern.counts(comp.n_wrong)
    others = sorted((name for name in dc.wrong if name != w_star),
                    key=lambda name: ID.blind(study_seed, ORDER_WSTAR, dc.source_item_id, name))
    chosen = [w_star, *others][:len(counts)]
    if len(chosen) < len(counts):
        raise ContractViolation(f"{dc.source_item_id}: {len(chosen)} wrong classes for a pattern "
                                f"needing {len(counts)}")
    members: list[Donor] = []
    for name, need in zip(chosen, counts):
        available = dc.wrong.get(name, ())
        if len(available) < need:
            raise ContractViolation(f"{dc.source_item_id}: wrong class {name} holds "
                                    f"{len(available)} donors, needs {need}")
        members.extend(available[:need])
    return tuple(members)


def _focal_slot(comp: Composition, states: Sequence[str]) -> int | None:
    if not comp.has_focal_slot:
        return None
    minority = comp.minority_state
    positions = [index for index, state in enumerate(states) if state == minority]
    if len(positions) != 1:
        raise ContractViolation(f"{comp.name}: expected exactly one {minority} slot, "
                                f"got {len(positions)}")
    return positions[0]


def build_team(item: Any, donors: DonorClasses, comp: Composition, wstar: str, study_seed: bytes,
               *, shared: SharedDonors, rule_version: str, bucket: str = "",
               cohort: str = COHORT_CONSTRUCTED) -> RootTeam:
    """Build one constructed team: blind donor order, blind slot permutation, exact occupancy."""
    correct_class = donors.largest_correct
    if correct_class is None and comp.c0:
        raise ContractViolation(f"{donors.source_item_id}: no correct class for c0={comp.c0}")
    correct = donors.correct.get(correct_class or "", ())[:comp.c0]
    if len(correct) != comp.c0:
        raise ContractViolation(f"{donors.source_item_id}: {len(correct)} correct donors < c0={comp.c0}")
    base = tuple(correct) + _wrong_members(donors, comp, wstar, study_seed)
    if len(base) != comp.n:
        raise ContractViolation(f"{comp.name}: built {len(base)} members for n={comp.n}")
    provisional = ID.root_team_id(CellKind.DONOR_BANK, donors.source_item_id,
                                  donors.item_variant_id,
                                  [d.generation_uid for d in base], None, f"{rule_version}:base")
    order = permutation_from_digest(ID.blind(study_seed, ORDER_SLOT, provisional), comp.n)
    members = tuple(base[index] for index in order)
    states = tuple(member.state for member in members)
    focal = _focal_slot(comp, states)
    root_id = ID.root_team_id(CellKind.DONOR_BANK, donors.source_item_id, donors.item_variant_id,
                              [d.generation_uid for d in members], focal, rule_version)
    construction_hash = identity.sha256_hex(identity.jcs([
        rule_version, comp.name, comp.n, comp.c0, comp.wrong_pattern.kind,
        list(comp.wrong_pattern.fragmented), donors.source_item_id, donors.item_variant_id,
        correct_class, wstar, [d.generation_uid for d in members], focal]))
    team = RootTeam(
        root_team_id=root_id, team_tag=team_tag(study_seed, root_id),
        family=getattr(item, "family", ""), source_item_id=donors.source_item_id,
        item_variant_id=donors.item_variant_id, cohort=cohort, composition=comp.name,
        c0=int(comp.c0), w_star_class=wstar, correct_class=correct_class, members=members,
        slot_states=states, focal_slot=focal, construction_rule_version=rule_version,
        construction_hash=construction_hash,
        shared_donor_uids=tuple(sorted(shared.uids() & set(d.generation_uid for d in members))),
        bucket=str(bucket), eligibility={"n_resolved": donors.n_resolved,
                                         "n_correct": donors.n_correct,
                                         "n_wrong": donors.n_wrong,
                                         "n_unknown": donors.n_unknown})
    validate_team(team, comp)
    return team


def natural_team(item: Any, draws: Sequence[tables.BankRow], study_seed: bytes, *,
                 rule_version: str, bucket: str = "") -> RootTeam:
    """A natural team: the natural-root draws in draw order, no composition, no selection.

    The protection slot is drawn blind **before** the roots are known to be anything (spec
    10.4), so ``focal_slot`` here is a position, not a correctness statement.
    """
    rows = sorted(draws, key=lambda row: row.draw_index)
    if not rows:
        raise ContractViolation("a natural team needs its natural_root draws")
    roles = {row.bank_role for row in rows}
    if roles != {BK.BankRole.NATURAL_ROOT.value}:
        raise ContractViolation(f"natural teams draw from the natural_root bank only; got {sorted(roles)}")
    members = tuple(Donor(generation_uid=row.generation_uid, decision_id=row.decision_id,
                          source_item_id=row.source_item_id, item_variant_id=row.item_variant_id,
                          draw_index=row.draw_index, state=STATE_WRONG,
                          canonical_class=NATURAL_COMPOSITION, content_hash=row.content_hash)
                    for row in rows)
    n = len(members)
    source_item_id, item_variant_id = rows[0].source_item_id, rows[0].item_variant_id
    focal = int.from_bytes(ID.blind(study_seed, ORDER_PROTECT, source_item_id,
                                    item_variant_id)[:8], "big") % n
    root_id = ID.root_team_id(CellKind.NATURAL_ROOT, source_item_id, item_variant_id,
                              [m.generation_uid for m in members], focal, rule_version)
    construction_hash = identity.sha256_hex(identity.jcs(
        [rule_version, NATURAL_COMPOSITION, n, source_item_id, item_variant_id,
         [m.generation_uid for m in members], focal]))
    return RootTeam(
        root_team_id=root_id, team_tag=team_tag(study_seed, root_id),
        family=getattr(item, "family", ""), source_item_id=source_item_id,
        item_variant_id=item_variant_id, cohort=COHORT_NATURAL, composition=NATURAL_COMPOSITION,
        c0=-1, w_star_class=None, correct_class=None, members=members,
        slot_states=tuple(STATE_UNKNOWN for _ in members), focal_slot=focal,
        construction_rule_version=rule_version, construction_hash=construction_hash,
        shared_donor_uids=(), bucket=str(bucket),
        eligibility={"n_draws": n, "rule": "natural_roots_in_draw_order"})


def validate_team(team: RootTeam, comp: Composition | None = None) -> None:
    """``n`` distinct generation uids and exactly ``c0`` correct members (spec 8.2)."""
    uids = team.generation_uids
    if len(set(uids)) != len(uids):
        raise ContractViolation(f"{team.root_team_id}: one generation record appears twice; four "
                                "copies of one draw are not four draws")
    if comp is None:
        return
    if len(uids) != comp.n:
        raise ContractViolation(f"{team.root_team_id}: {len(uids)} members for n={comp.n}")
    n_correct = sum(1 for state in team.slot_states if state == STATE_CORRECT)
    if n_correct != comp.c0:
        raise ContractViolation(f"{team.root_team_id}: {n_correct} correct members, c0={comp.c0}")
    wrong_classes = [m.canonical_class for m in team.members if m.state == STATE_WRONG]
    counts = sorted((wrong_classes.count(name) for name in set(wrong_classes)), reverse=True)
    if counts != sorted(comp.wrong_pattern.counts(comp.n_wrong), reverse=True):
        raise ContractViolation(f"{team.root_team_id}: wrong occupancy {counts} does not match "
                                f"{comp.wrong_pattern.counts(comp.n_wrong)}")


# --------------------------------------------------------------------------- rows and export


def team_rows(team: RootTeam, export_hash: str) -> tuple[tables.RootTeamRow, tables.RootTeamPrivateRow]:
    """The public row and the protected row of one sealed team (never the same file)."""
    public = tables.RootTeamRow(
        root_team_id=team.root_team_id, team_tag=team.team_tag, family=team.family,
        source_item_id=team.source_item_id, item_variant_id=team.item_variant_id,
        cohort=team.cohort,
        ordered_generation_uids_json=json.dumps(list(team.generation_uids)),
        ordered_decision_ids_json=json.dumps(list(team.decision_ids)),
        n_independent_generations=len(set(team.generation_uids)), focal_slot=team.focal_slot,
        construction_rule_version=team.construction_rule_version,
        construction_hash=team.construction_hash, export_hash=export_hash,
        shared_donor_uids_json=json.dumps(list(team.shared_donor_uids)), bucket=team.bucket)
    private = tables.RootTeamPrivateRow(
        root_team_id=team.root_team_id, team_tag=team.team_tag, composition=team.composition,
        c0=int(team.c0), w_star_class=team.w_star_class, correct_class=team.correct_class,
        slot_states_json=json.dumps(list(team.slot_states)),
        eligibility_json=json.dumps(dict(team.eligibility), sort_keys=True),
        extension_stage_used=int(team.eligibility.get("extension_stage", 0)))
    return public, private


def export_root_bundle(team: RootTeam, drafts: Mapping[str, Any]) -> Mapping[str, Any]:
    """The label-free payload of one team: ordered slots, uids, decision ids, public outputs."""
    slots = []
    for slot, member in enumerate(team.members):
        output = drafts.get(member.generation_uid)
        if output is None:
            raise ContractViolation(f"{team.root_team_id}: no public output for slot {slot} "
                                    f"({member.generation_uid})")
        slots.append({"slot": slot, "generation_uid": member.generation_uid,
                      "decision_id": member.decision_id, "content_hash": member.content_hash,
                      "output": BU.output_to_json(output)})
    payload = {"root_team_id": team.root_team_id, "team_tag": team.team_tag,
               "source_item_id": team.source_item_id, "item_variant_id": team.item_variant_id,
               "n": team.n, "focal_slot": team.focal_slot, "slots": slots}
    BU.assert_label_free(payload)
    return payload


def bundle_payload(family: str, bucket: str, teams: Sequence[RootTeam],
                   drafts: Mapping[str, Any], rule_version: str) -> dict[str, Any]:
    """The bucket file: one or both members of a pair, label-free."""
    payload = {"schema_version": BU.BUNDLE_SCHEMA_VERSION, "family": family, "bucket": str(bucket),
               "construction_rule_version": rule_version,
               "teams": [export_root_bundle(team, drafts) for team in teams]}
    BU.assert_label_free(payload)
    return payload


def seal_root_teams(run_root: str | os.PathLike, family: str, bucket: str,
                    teams: Sequence[RootTeam], drafts: Mapping[str, Any], *,
                    rule_version: str) -> tuple[str, list[tables.RootTeamRow],
                                                list[tables.RootTeamPrivateRow]]:
    """Write the label-free bucket and return ``(export_hash, public rows, protected rows)``.

    ``deps.write_bundle`` refuses a differing overwrite, so a sealed team never mutates: a
    different team set is a different bucket (plan §2.4).
    """
    payload = bundle_payload(family, bucket, teams, drafts, rule_version)
    export_hash = deps.write_bundle(run_root, family, bucket, payload)
    public, private = [], []
    for team in teams:
        rows = team_rows(team, export_hash)
        public.append(rows[0])
        private.append(rows[1])
    return export_hash, public, private


# --------------------------------------------------------------------------- recruitment flow


@dataclass(frozen=True)
class RecruitmentRecord:
    """What happened to one screened item (spec 14.3; ``recruitment_flow`` aggregates these)."""

    source_item_id: str
    family: str
    screened: bool
    resolved: int
    eligible_initial: bool
    extended: bool
    eligible_extended: bool
    paired: bool
    fragmented: bool
    single: bool


def recruitment_flow(records: Sequence[RecruitmentRecord], *, stop_reason: str) -> dict[str, Any]:
    """The per-family screened -> eligible -> paired census and why recruitment stopped."""
    out: dict[str, Any] = {"stop_reason": stop_reason, "families": {}}
    for record in records:
        family = out["families"].setdefault(record.family, {
            "screened": 0, "resolved": 0, "eligible_32": 0, "extended": 0, "eligible_64": 0,
            "paired": 0, "fragmented": 0, "single": 0})
        family["screened"] += int(record.screened)
        family["resolved"] += int(record.resolved)
        family["eligible_32"] += int(record.eligible_initial)
        family["extended"] += int(record.extended)
        family["eligible_64"] += int(record.eligible_extended)
        family["paired"] += int(record.paired)
        family["fragmented"] += int(record.fragmented)
        family["single"] += int(record.single)
    return out


# --------------------------------------------------------------------------- the worker


@dataclass(frozen=True)
class DonorInput:
    """What a CONSTRUCT cell needs about one item; the runner assembles it (WP4).

    ``grades`` and ``classes`` come from ``grading.tri_state_index`` /
    ``grading.canonical_class_index`` -- the sanctioned label readers -- and ``drafts`` maps
    each donor's ``generation_uid`` to its :class:`~agents_scaling.nextdyn.schema.PublicOutput`.
    """

    item: Any
    bank_rows: tuple[tables.BankRow, ...]
    grades: Mapping[str, str]
    classes: Mapping[str, str | None]
    drafts: Mapping[str, Any]
    bucket: str
    pair: str


def construct_worker(runner: Any, source_item_id: str) -> tables.ItemFile:
    """The CONSTRUCT cell worker (cpu lane, WRITER_SIDE).

    Runner contract: ``runner.cell``, ``runner.run_root``, ``runner.contract``,
    ``runner.compositions`` (a :class:`CompositionSet`) and
    ``runner.donor_input(source_item_id) -> DonorInput``.  The protected team rows go back in
    ``ItemFile.private_rows`` so the runner writes them under ``protected/cells/`` (R7).
    """
    cell = runner.cell
    contract = runner.contract
    comps: CompositionSet = runner.compositions
    data: DonorInput = runner.donor_input(source_item_id)
    study_seed = contract.study_seed
    dc = classify_donors(data.bank_rows, data.grades, data.classes, study_seed=study_seed)
    members = comps.pair(data.pair)
    eligible = eligibility(dc, members, {data.pair: [c.name for c in members]})
    if not eligible.per_pair.get(data.pair, False):
        return tables.ItemFile(
            schema_version=tables.SCHEMA_VERSION, cell_id=cell.cell_id, cell_kind=cell.kind,
            source_item_id=source_item_id, status=tables.ItemStatus.COMPLETE,
            bundle_export_hash=None, rows={"root_teams": ()},
            private_rows={"root_teams_private": ()})
    need = max(comp.wrong_pattern.counts(comp.n_wrong)[0] for comp in members)
    w_star = choose_w_star(dc, study_seed, source_item_id, min_size=need)
    if w_star is None:
        raise ManifestError(f"{source_item_id}: pair {data.pair} is eligible but no wrong class "
                            f"holds {need} donors")
    shared = shared_donors(dc, members, w_star)
    teams = [build_team(data.item, dc, comp, w_star, study_seed, shared=shared,
                        rule_version=comps.construction_rule_version, bucket=data.bucket)
             for comp in members]
    export_hash, public, private = seal_root_teams(
        runner.run_root, data.item.family, data.bucket, teams, data.drafts,
        rule_version=comps.construction_rule_version)
    return tables.ItemFile(
        schema_version=tables.SCHEMA_VERSION, cell_id=cell.cell_id, cell_kind=cell.kind,
        source_item_id=source_item_id, status=tables.ItemStatus.COMPLETE,
        bundle_export_hash=export_hash, rows={"root_teams": tuple(public)},
        private_rows={"root_teams_private": tuple(private)})


__all__ = [
    "study_compositions",
    "COHORT_CONSTRUCTED", "COHORT_NATURAL", "COMPOSITIONS_FILE", "Composition", "CompositionSet",
    "Donor", "DonorClasses", "DonorInput", "Eligibility", "NATURAL_COMPOSITION",
    "PATTERN_COHERENT", "PATTERN_FRAGMENTED", "RecruitmentRecord", "RootTeam", "SharedDonors",
    "WrongPattern", "build_team", "bundle_payload", "choose_w_star", "classify_donors",
    "composition_eligible", "construct_worker", "eligibility", "export_root_bundle",
    "load_compositions", "natural_team", "recruitment_flow", "seal_root_teams", "shared_donors",
    "team_rows", "team_tag", "validate_team",
]
