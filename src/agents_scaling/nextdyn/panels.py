"""The declared analysis populations (plan §4.2; spec 14.2, 15.1).

Half of ARD36's sweep claims were retracted because a statistic was computed on one row set and
read as though it had been computed on another.  ``ard/panels.py`` is the repair pattern and
this module is that pattern applied to nextdyn's thirteen populations: each one is **named**,
carries a **row filter** that is the only way rows enter it, and carries a **measured census**
that is compared against what the plan says the population should be -- reporting the mismatch
rather than hiding it.

Three rules the populations enforce structurally:

* **A population is declared before it is measured.**  ``expected`` comes from ``study.yaml``
  (the cohort sizes, the sub-panel sizes, the experiment sizes), never from the data.  A panel
  whose realised size differs is a finding printed by :meth:`Panel.census`, not a silently
  smaller ``n``.
* **A pass@k panel is a *fixed* item set with the right bank role.**  The 25 draws of a PASSK
  item are its five ``natural_root`` bank draws plus twenty FRESH episode decisions, and they
  must share prompt, decoding and model hashes; constructed roots and donor/difficulty draws
  are refused by role (spec 18.3 ``test_passk_uses_fixed_items_and_correct_bank_role``).  The
  panel is fixed across every ``k`` so ``pass@1`` and ``pass@25`` are two readings of one set.
* **Membership is a set of keys, not a filtered frame.**  ``Panel.keys`` is the membership in
  the panel's own unit (items, teams, problems), so a caller joins on it instead of re-deriving
  the filter and getting a different answer.

Column names are never listed here: the rows are the derived tables declared in ``tables.py``
(R1), and a panel only ever reads them by name through the filters below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Mapping, Sequence

from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.nextdyn.ids import CellKind

#: The thirteen declared populations (plan §4.2).
NAT: Final[str] = "NAT"
NAT2: Final[str] = "NAT2"
CON_PAIRED: Final[str] = "CON_PAIRED"
CON_SUPP: Final[str] = "CON_SUPP"
FRAG: Final[str] = "FRAG"
BRIDGE: Final[str] = "BRIDGE"
E3A: Final[str] = "E3A"
E3A_FILL: Final[str] = "E3A_FILL"
E3A_NATPROT: Final[str] = "E3A_NATPROT"
E2A: Final[str] = "E2A"
E2B: Final[str] = "E2B"
CAPTURE: Final[str] = "CAPTURE"
PASSK: Final[str] = "PASSK"

POPULATIONS: Final[tuple[str, ...]] = (
    NAT, NAT2, CON_PAIRED, CON_SUPP, FRAG, BRIDGE, E3A, E3A_FILL, E3A_NATPROT,
    E2A, E2B, CAPTURE, PASSK,
)

UNIT_ITEM: Final[str] = "source_item"
UNIT_PROBLEM: Final[str] = "problem"

COHORT_NATURAL: Final[str] = "natural"
COHORT_NATURAL_SECOND: Final[str] = "natural_second"
COHORT_CONSTRUCTED: Final[str] = "constructed"

#: Bank roles.  A pass@k panel admits ``natural_root`` draws only: a difficulty draw is a
#: screening observation and a donor draw is a construction input, and counting either as an
#: independent attempt at the item is the ARD36 bank/episode conflation in a new costume.
BANK_ROLE_NATURAL_ROOT: Final[str] = "natural_root"
PASSK_BANK_ROLES: Final[frozenset[str]] = frozenset({BANK_ROLE_NATURAL_ROOT})
PASSK_FRESH_LABEL: Final[str] = "FRESH"
PASSK_DRAWS: Final[int] = 25


@dataclass(frozen=True)
class Panel:
    """One declared population: what it selects, how big it is, and out of what."""

    name: str
    unit: str
    question: str
    keys: frozenset
    universe: int
    expected: int | None
    row_filter: Callable[[Mapping[str, Any]], bool] = field(repr=False)
    detail: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def size(self) -> int:
        return len(self.keys)

    @property
    def coverage_rate(self) -> float:
        return (self.size / self.universe) if self.universe else 0.0

    @property
    def matches_expected(self) -> bool | None:
        return None if self.expected is None else self.size == int(self.expected)

    def select(self, rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [row for row in rows if self.row_filter(row)]

    def census(self) -> dict[str, Any]:
        """The measured membership beside what the plan declared (never one without the other)."""
        return {
            "panel": self.name, "unit": self.unit, "question": self.question,
            "size": self.size, "universe": self.universe, "coverage_rate": self.coverage_rate,
            "expected": self.expected, "matches_expected": self.matches_expected,
            **dict(self.detail),
        }


# --------------------------------------------------------------------------- declared sizes


@dataclass(frozen=True)
class PanelSizes:
    """The per-family sizes ``study.yaml`` declares, resolved once (R1: one home each)."""

    families: tuple[str, ...]
    natural: int
    second_root: int
    constructed: int
    fragmented: int
    bridge: int
    e3a: int
    e3a_fill: int
    natprot: int
    e2a: int
    e2b_problems: int
    capture: int

    def expected(self, name: str) -> int | None:
        """Declared size of one population over all families (``None`` where the plan is silent)."""
        per_family = {
            NAT: self.natural, NAT2: self.second_root, CON_PAIRED: self.constructed,
            FRAG: self.fragmented, BRIDGE: self.bridge, E3A: self.e3a,
            E3A_FILL: self.e3a_fill, E3A_NATPROT: self.natprot, E2A: self.e2a,
            CAPTURE: self.capture, PASSK: self.natural,
        }
        if name == E2B:
            return int(self.e2b_problems)          # problems, not per-family items
        if name in per_family:
            return int(per_family[name]) * len(self.families)
        return None


def panel_sizes(cfg: Any) -> PanelSizes:
    """Compile a ``config.StudyConfig`` into the declared population sizes."""
    cohorts = cfg.cohorts
    sub = cohorts.sub_panels
    return PanelSizes(
        families=tuple(cohorts.families),
        natural=int(cohorts.natural_target_per_family),
        second_root=int(sub.second_root),
        constructed=int(cohorts.constructed_target_per_family),
        fragmented=int(sub.fragmented),
        bridge=int(sub.bridge),
        e3a=int(cfg.experiments.E3a.target_items_per_task),
        e3a_fill=int(sub.e3a_fill),
        natprot=int(cohorts.natural_target_per_family),
        e2a=int(cfg.experiments.E2a.items_per_task),
        e2b_problems=int(cfg.experiments.E2b.source_problems),
        capture=int(cfg.instrumentation.target_items_per_task),
    )


def passk_ks(profile: Any) -> tuple[int, ...]:
    """``analysis.profiles.*.passk_ks``: the k values the pass@k panel reports (spec 15.1).

    They are read once, here, and the panel is fixed across all of them -- which is what makes
    ``pass@1`` and ``pass@25`` two readings of one item set rather than two panels.
    """
    ks = tuple(int(k) for k in profile.passk_ks)
    if not ks or min(ks) < 1:
        raise ContractViolation(f"analysis passk_ks must be positive ints, got {ks}")
    if max(ks) > PASSK_DRAWS:
        raise ContractViolation(
            f"passk_ks {ks} asks for k > {PASSK_DRAWS}, but a PASSK item has exactly "
            f"{PASSK_DRAWS} draws (5 natural roots + 20 FRESH)")
    return ks


# --------------------------------------------------------------------------- frames


@dataclass(frozen=True)
class PanelFrame:
    """The rows every population is measured on, read once from the derived tables."""

    team_rounds: tuple[Mapping[str, Any], ...] = ()
    agent_rounds: tuple[Mapping[str, Any], ...] = ()
    item_splits: tuple[Mapping[str, Any], ...] = ()
    banks: tuple[Mapping[str, Any], ...] = ()
    root_teams: tuple[Mapping[str, Any], ...] = ()

    def split_flags(self) -> dict[str, Mapping[str, Any]]:
        return {str(r["source_item_id"]): r for r in self.item_splits}


def _labels(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("display_label") or ""), str(row.get("cohort") or ""),
            str(row.get("composition") or ""))


def _split_flag(splits: Mapping[str, Mapping[str, Any]], item: str, flag: str) -> bool:
    row = splits.get(item)
    return bool(row and row.get(flag))


# --------------------------------------------------------------------------- populations


def _items_where(frame: PanelFrame, predicate: Callable[[Mapping[str, Any]], bool]) -> frozenset[str]:
    return frozenset(str(r["source_item_id"]) for r in frame.team_rounds if predicate(r))


def _paired_items(frame: PanelFrame, compositions: Sequence[str]) -> frozenset[str]:
    """Items carrying **every** named composition: a pair is only a pair if both members ran."""
    seen: dict[str, set[str]] = {}
    for row in frame.team_rounds:
        composition = str(row.get("composition") or "")
        if composition in compositions:
            seen.setdefault(str(row["source_item_id"]), set()).add(composition)
    wanted = set(compositions)
    return frozenset(item for item, have in seen.items() if wanted <= have)


def population(name: str, frame: PanelFrame, cfg: Any) -> Panel:
    """The declared population ``name`` measured on ``frame`` (plan §4.2)."""
    if name not in POPULATIONS:
        raise ContractViolation(f"unknown population {name!r}; declared: {POPULATIONS}")
    sizes = panel_sizes(cfg)
    splits = frame.split_flags()
    universe = len({str(r["source_item_id"]) for r in frame.team_rounds})
    paired = tuple(cfg.cohorts.paired_compositions)
    detail: dict[str, Any] = {}

    if name == NAT:
        keys = _items_where(frame, lambda r: str(r.get("cohort")) == COHORT_NATURAL)
        question = "natural five-agent teams: does exchange help or harm as sampled?"
        row_filter = lambda r: str(r.get("cohort")) == COHORT_NATURAL  # noqa: E731
    elif name == NAT2:
        keys = _items_where(frame, lambda r: str(r.get("cohort")) == COHORT_NATURAL_SECOND)
        question = "the second independent natural root: is the natural result root-specific?"
        row_filter = lambda r: str(r.get("cohort")) == COHORT_NATURAL_SECOND  # noqa: E731
    elif name == CON_PAIRED:
        keys = _paired_items(frame, paired)
        question = "items on which both paired compositions were built (H1, H3)"
        row_filter = lambda r: (str(r["source_item_id"]) in keys  # noqa: E731
                                and str(r.get("composition") or "") in paired)
        detail["compositions"] = list(paired)
    elif name == CON_SUPP:
        built = _items_where(frame, lambda r: str(r.get("cohort")) == COHORT_CONSTRUCTED)
        keys = built - _paired_items(frame, paired)
        question = "constructed items where only one composition was eligible (supplementary)"
        row_filter = lambda r: str(r["source_item_id"]) in keys  # noqa: E731
    elif name == FRAG:
        fragmented = str(cfg.cohorts.fragmented_composition)
        keys = _items_where(frame, lambda r: str(r.get("composition") or "") == fragmented)
        question = "fragmented wrong majority vs the coherent one"
        row_filter = lambda r: str(r.get("composition") or "") == fragmented  # noqa: E731
        detail["composition"] = fragmented
    elif name == BRIDGE:
        bridge = str(cfg.experiments.E1.bridge_protocol)
        keys = _items_where(frame, lambda r: str(r.get("display_label") or "") == bridge)
        question = "answer-only packets: is the effect carried by the rationale?"
        row_filter = lambda r: str(r.get("display_label") or "") == bridge  # noqa: E731
        detail["bridge_protocol"] = bridge
    elif name in (E3A, E3A_FILL):
        arms = tuple(cfg.experiments.E3a.filled_arms if name == E3A_FILL else cfg.experiments.E3a.arms)
        flag = "sub_e3a_fill" if name == E3A_FILL else "sub_e3a"
        keys = _items_where(frame, lambda r: (str(r.get("display_label") or "") in arms
                                              and _split_flag(splits, str(r["source_item_id"]), flag)))
        question = "directed focal in/out channels" + (" with inert filler" if name == E3A_FILL else "")
        row_filter = lambda r: (str(r.get("display_label") or "") in arms  # noqa: E731
                                and str(r["source_item_id"]) in keys)
        detail["arms"] = list(arms)
    elif name == E3A_NATPROT:
        protection = str(cfg.experiments.E3a.protection_protocol)
        keys = _items_where(frame, lambda r: (str(r.get("display_label") or "") == protection
                                              and str(r.get("cohort")) == COHORT_NATURAL))
        question = "a randomly protected slot in a natural team (spec 10.4)"
        row_filter = lambda r: str(r.get("display_label") or "") == protection  # noqa: E731
        detail["protection_protocol"] = protection
    elif name == E2A:
        keys = frozenset(str(r["source_item_id"]) for r in frame.agent_rounds
                         if str(r.get("cell_kind")) == CellKind.RECEIVER.value
                         and _split_flag(splits, str(r["source_item_id"]), "sub_e2a"))
        question = "natural-packet endorsement assay (one-call receiver)"
        row_filter = lambda r: str(r["source_item_id"]) in keys  # noqa: E731
        universe = len({str(r["source_item_id"]) for r in frame.agent_rounds})
    elif name == E2B:
        keys = frozenset(str(r["source_item_id"]) for r in frame.agent_rounds
                         if str(r.get("cell_kind")) == CellKind.RECEIVER.value
                         and not _split_flag(splits, str(r["source_item_id"]), "sub_e2a"))
        question = "synthetic mod-97 receiver with a real local-step checker"
        row_filter = lambda r: str(r["source_item_id"]) in keys  # noqa: E731
        universe = len({str(r["source_item_id"]) for r in frame.agent_rounds})
        detail["unit"] = UNIT_PROBLEM
    elif name == CAPTURE:
        keys = frozenset(item for item in {str(r["source_item_id"]) for r in frame.team_rounds}
                         if _split_flag(splits, item, "sub_capture"))
        question = "read-only activation capture panel"
        row_filter = lambda r: str(r["source_item_id"]) in keys  # noqa: E731
    else:  # PASSK
        return passk_population(frame, cfg)

    return Panel(name=name, unit=UNIT_ITEM if name != E2B else UNIT_PROBLEM, question=question,
                 keys=keys, universe=universe, expected=sizes.expected(name),
                 row_filter=row_filter, detail=detail)


def passk_population(frame: PanelFrame, cfg: Any) -> Panel:
    """PASSK: natural items whose 5 natural-root draws and 20 FRESH decisions are comparable.

    The membership rule is **asserted, not assumed** (spec 15.1): an item joins only when it has
    exactly five ``natural_root`` bank draws and twenty FRESH agent rounds, and when every one
    of those 25 generations shares one model hash and one budget hash.  Difficulty and donor
    draws are refused by ``bank_role``; a constructed root team is refused by cohort.
    """
    sizes = panel_sizes(cfg)
    roots: dict[str, list[Mapping[str, Any]]] = {}
    for row in frame.banks:
        if str(row.get("bank_role")) in PASSK_BANK_ROLES:
            roots.setdefault(str(row["source_item_id"]), []).append(row)
    fresh: dict[str, list[Mapping[str, Any]]] = {}
    for row in frame.agent_rounds:
        if str(row.get("display_label") or "") == PASSK_FRESH_LABEL:
            fresh.setdefault(str(row["source_item_id"]), []).append(row)
    n_roots = int(cfg.cohorts.natural_root_draws)
    n_fresh = PASSK_DRAWS - n_roots
    keys: set[str] = set()
    rejected: dict[str, str] = {}
    for item in sorted(set(roots) | set(fresh)):
        bank_rows, fresh_rows = roots.get(item, []), fresh.get(item, [])
        if len(bank_rows) != n_roots:
            rejected[item] = f"{len(bank_rows)} natural_root draws (need {n_roots})"
            continue
        if len(fresh_rows) != n_fresh:
            rejected[item] = f"{len(fresh_rows)} FRESH decisions (need {n_fresh})"
            continue
        models = {str(r.get("model_hash") or "") for r in bank_rows} | \
                 {str(r.get("model_hash") or "") for r in fresh_rows}
        budgets = {str(r.get("budget_hash") or "") for r in bank_rows} | \
                  {str(r.get("budget_hash") or "") for r in fresh_rows}
        if len(models) != 1 or len(budgets) != 1:
            rejected[item] = f"draws span {len(models)} models / {len(budgets)} budgets"
            continue
        keys.add(item)
    universe = len(set(roots) | set(fresh))
    return Panel(
        name=PASSK, unit=UNIT_ITEM,
        question="pass@k over one fixed bank of 25 comparable generations per item",
        keys=frozenset(keys), universe=universe, expected=sizes.expected(PASSK),
        row_filter=lambda r: str(r["source_item_id"]) in keys,
        detail={"n_roots_required": n_roots, "n_fresh_required": n_fresh,
                "bank_roles": sorted(PASSK_BANK_ROLES), "n_rejected": len(rejected),
                "rejected_examples": dict(sorted(rejected.items())[:5])},
    )


def all_panels(frame: PanelFrame, cfg: Any) -> dict[str, Panel]:
    return {name: population(name, frame, cfg) for name in POPULATIONS}


def census_report(frame: PanelFrame, cfg: Any) -> list[dict[str, Any]]:
    """One census row per declared population, in declaration order."""
    return [panel.census() for panel in all_panels(frame, cfg).values()]


def write_census(path, frame: PanelFrame, cfg: Any) -> list[dict[str, Any]]:
    rows = census_report(frame, cfg)
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return rows


__all__ = [
    "BANK_ROLE_NATURAL_ROOT", "BRIDGE", "CAPTURE", "COHORT_CONSTRUCTED", "COHORT_NATURAL",
    "COHORT_NATURAL_SECOND", "CON_PAIRED", "CON_SUPP", "E2A", "E2B", "E3A", "E3A_FILL",
    "E3A_NATPROT", "FRAG", "NAT", "NAT2", "PASSK", "PASSK_BANK_ROLES", "PASSK_DRAWS",
    "PASSK_FRESH_LABEL", "POPULATIONS", "Panel", "PanelFrame", "PanelSizes", "UNIT_ITEM",
    "UNIT_PROBLEM", "all_panels", "census_report", "panel_sizes", "passk_ks",
    "passk_population", "population", "write_census",
]
