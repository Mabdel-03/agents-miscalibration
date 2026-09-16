"""The three typed banks and the difficulty estimator (spec 5.3, 8.3, 15.1; plan §2.3, D14).

ARD36 ran **one** 16-draw bank per item and used it for difficulty, for constructed donors
*and* as the round-0 record of a PRIVATE episode.  That is defect D-11: the draws carried the
episode's id, ``decisions`` grew 14,336 duplicate keys, and "difficulty" was estimated on the
same samples the teams were built from.  Here the bank is **typed**:

===============  =========================  ===================================================
role             cell kind                  used for
===============  =========================  ===================================================
``difficulty``   ``DIFFICULTY_BANK``        difficulty only -- never a donor, never a root
``donor``        ``DONOR_BANK``             constructed teams (the extension stage adds draws)
``natural_root`` ``NATURAL_ROOT``           natural teams, in draw order
===============  =========================  ===================================================

A draw is a ``decisions`` row (D14): ``phase="bank"``, ``round=0``, ``agent_slot`` = the draw
index, ``episode_id = ids.bank_draw_id(...)``, ``root_team_id=""``.  ``banks`` is the keyed
view ``materialize`` derives from those rows, so one sample is written once and counted by a
distinct ``generation_uid`` (``test_independent_draws_have_distinct_generation_uids``;
``test_bank_role_cannot_alias_private_episode``).

This module is on the **generation** side of R7: it never imports a grader, a gold store or
``grading``.  :func:`difficulty` takes the tri-state labels as a plain
``{generation_uid: "C"|"W"|"U"}`` mapping, which ``grading.tri_state_index`` -- the one
sanctioned label reader -- produces for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import ids as ID
from agents_scaling.nextdyn import items as I
from agents_scaling.nextdyn import prompts, schema, tables
from agents_scaling.nextdyn.errors import ContractViolation, ManifestError
from agents_scaling.nextdyn.ids import CellKind

STATE_CORRECT: Final[str] = "C"
STATE_WRONG: Final[str] = "W"
STATE_UNKNOWN: Final[str] = "U"

#: ``cells.options`` key that carries the extension stage of a DONOR_BANK cell.
EXTENSION_STAGE_OPTION: Final[str] = "extension_stage"
STAGE_INITIAL: Final[int] = 0
STAGE_EXTENDED: Final[int] = 1


class BankRole(str, Enum):
    """The three disjoint banks of one ``(item, model, budget)`` (spec 5.3)."""

    DIFFICULTY = "difficulty"
    DONOR = "donor"
    NATURAL_ROOT = "natural_root"


ROLE_KINDS: Final[Mapping[BankRole, CellKind]] = {
    BankRole.DIFFICULTY: CellKind.DIFFICULTY_BANK,
    BankRole.DONOR: CellKind.DONOR_BANK,
    BankRole.NATURAL_ROOT: CellKind.NATURAL_ROOT,
}
KIND_ROLES: Final[Mapping[CellKind, BankRole]] = {kind: role for role, kind in ROLE_KINDS.items()}


def role_of(kind: CellKind | str) -> BankRole:
    try:
        return KIND_ROLES[ID.coerce_kind(kind)]
    except KeyError as exc:
        raise ManifestError(f"{kind} is not a bank kind; bank kinds are "
                            f"{[k.value for k in ROLE_KINDS.values()]}") from exc


def draw_range(role: BankRole, banks: K.BankSpec, *, extension_stage: int = STAGE_INITIAL) -> range:
    """The draw indices one cell of this role and stage produces.

    The extension stage of the donor bank continues the *same* index space
    (``donor_initial .. donor_max``), so an extended item has 64 distinct draws and no draw
    index is ever produced twice (plan §2.4 "extension draws 32..63").
    """
    if role is BankRole.DIFFICULTY:
        return range(0, banks.difficulty_draws)
    if role is BankRole.NATURAL_ROOT:
        return range(0, banks.natural_root_draws)
    if extension_stage == STAGE_INITIAL:
        return range(0, banks.donor_initial)
    if extension_stage == STAGE_EXTENDED:
        return range(banks.donor_initial, banks.donor_max)
    raise ContractViolation(f"unknown extension stage {extension_stage!r}; "
                            f"expected {STAGE_INITIAL} or {STAGE_EXTENDED}")


def bank_key(role: BankRole, source_item_id: str, item_variant_id: str, model: K.ModelCell,
             tokens: K.TokenContract, draw: int, *, study_id: str) -> ID.CompositeKey:
    """The composite key of one bank draw (contract §2.0, the per-kind table)."""
    return ID.CompositeKey(
        study_id=study_id, cell_kind=ROLE_KINDS[role], source_item_id=source_item_id,
        item_variant_id=item_variant_id, root_team_id="", protocol_hash="",
        receiver_model_hash=ID.model_hash(model), budget_hash=tokens.budget_hash(),
        replicate_id=0, round=0, agent_slot=int(draw), phase=K.PHASE_BANK)


def bank_draw_id(role: BankRole, key: ID.CompositeKey) -> str:
    return ID.bank_draw_id(ROLE_KINDS[role], key.source_item_id, key.item_variant_id,
                           key.receiver_model_hash, key.budget_hash, key.agent_slot)


def bank_row(role: BankRole, key: ID.CompositeKey, decision: tables.DecisionRow, *,
             extension_stage: int, cell_id: str) -> tables.BankRow:
    """The ``banks`` view row of one draw (``materialize`` derives these from ``decisions``)."""
    if key.cell_kind is not ROLE_KINDS[role]:
        raise ManifestError(f"a {role.value} draw cannot carry cell kind {key.cell_kind.value}")
    request_hashes = json.loads(decision.request_hashes_json)
    return tables.BankRow(
        bank_draw_id=bank_draw_id(role, key), decision_id=decision.decision_id,
        generation_uid=decision.generation_uid, cell_kind=key.cell_kind.value,
        bank_role=role.value, source_item_id=key.source_item_id,
        item_variant_id=key.item_variant_id, model_hash=key.receiver_model_hash,
        budget_hash=key.budget_hash, draw_index=key.agent_slot,
        extension_stage=int(extension_stage), request_hash=str(request_hashes[0]),
        content_hash=decision.content_hash, parse_status=decision.parse_status,
        validity=decision.validity, cell_id=cell_id)


# --------------------------------------------------------------------------- the extension rule


@dataclass(frozen=True)
class ExtensionRule:
    """The frozen rule that authorizes a second donor stage (``compositions.yaml``).

    Frozen before any data (plan §2.4): extend to ``donor_max`` **iff** the item is at stage 0,
    is not yet paired-eligible, and has at least one resolved correct and one resolved wrong
    draw.  Nothing about how close it came, and no per-item judgement.
    """

    name: str
    min_correct_resolved: int
    min_wrong_resolved: int

    def __post_init__(self) -> None:
        if self.min_correct_resolved < 0 or self.min_wrong_resolved < 0:
            raise ContractViolation("extension thresholds are non-negative counts")


def extension_authorized(rule: ExtensionRule, *, stage: int, paired_eligible: bool,
                         n_correct: int, n_wrong: int) -> bool:
    """Whether a DONOR_BANK extension cell may be emitted for this item."""
    if not rule.name:
        raise ContractViolation("compositions.extension_rule.name is empty: the rule that "
                                "authorizes a second donor stage is named and frozen")
    return (stage == STAGE_INITIAL and not paired_eligible
            and n_correct >= rule.min_correct_resolved and n_wrong >= rule.min_wrong_resolved)


# --------------------------------------------------------------------------- difficulty


@dataclass(frozen=True)
class Difficulty:
    """Per ``(item, model, budget)`` difficulty, with its unresolved mass carried explicitly.

    ``raw`` conditions on the resolved draws, ``operational`` counts every drawn sample, and
    ``lo``/``hi`` are the Manski bounds: ``lo`` credits nothing to the unresolved draws and
    ``hi`` credits them all, so the pair brackets every feasible labelling.  ``smoothed`` is
    ``(c + 0.5) / (g + 1)`` -- the Jeffreys point estimate -- which is what ``bin`` is cut on,
    so an item with 0/3 resolved draws does not share a bin with one at 0/16.
    """

    source_item_id: str
    model_hash: str
    budget_hash: str
    bank_role: str
    n_draws: int
    g_resolved: int
    c: int
    w: int
    u: int
    raw: float | None
    operational: float
    lo: float
    hi: float
    smoothed: float
    bin: int
    beta_ci_lo: float
    beta_ci_hi: float


def cutpoints_of(profile: Any) -> tuple[float, ...]:
    """The frozen difficulty cutpoints of an analysis profile (``analysis.yaml``).

    The bands are a configured value, not a rule of this module: ``difficulty`` takes them as
    an argument and this is where the profile field is read (R2).
    """
    values = tuple(float(c) for c in profile.difficulty_cutpoints)
    if not values:
        raise ContractViolation("analysis.profiles.*.difficulty_cutpoints is empty")
    return values


def difficulty_bin(smoothed: float, cutpoints: Sequence[float]) -> int:
    """The index of the band ``smoothed`` falls in; cutpoints come from ``analysis.yaml``."""
    values = [float(c) for c in cutpoints]
    if list(values) != sorted(values) or not values:
        raise ContractViolation(f"difficulty cutpoints {cutpoints!r} must be a non-empty "
                                "increasing sequence (analysis.profiles.*.difficulty_cutpoints)")
    for index, cut in enumerate(values):
        if smoothed < cut:
            return index
    return len(values)


def beta_ci(c: int, g: int, *, alpha: float) -> tuple[float, float]:
    """Clopper-Pearson interval for ``c`` successes in ``g`` resolved draws.

    ``g == 0`` gives the whole unit interval: no resolved draw is no information, not a
    point estimate at zero.
    """
    if not 0.0 < alpha < 1.0:
        raise ContractViolation(f"beta_ci alpha {alpha!r} must be in (0, 1)")
    if g <= 0:
        return (0.0, 1.0)
    from scipy.stats import beta as _beta

    lo = 0.0 if c == 0 else float(_beta.ppf(alpha / 2.0, c, g - c + 1))
    hi = 1.0 if c == g else float(_beta.ppf(1.0 - alpha / 2.0, c + 1, g - c))
    return (lo, hi)


def difficulty(rows: Sequence[tables.BankRow], grades: Mapping[str, str], *,
               cutpoints: Sequence[float], ci_alpha: float) -> Difficulty:
    """The difficulty of one ``(item, model, budget)`` from its bank rows and tri-state labels.

    ``grades`` is ``{generation_uid: "C"|"W"|"U"}`` (``grading.tri_state_index``); a draw with
    no grade row is unresolved, exactly like an explicit ``U``.  ``cutpoints`` and ``ci_alpha``
    are passed in because both are configured values -- ``analysis.profiles.*.difficulty_cutpoints``
    and ``study.analysis.familywise_alpha`` -- and no number lives in this module.
    """
    if not rows:
        raise ContractViolation("difficulty needs at least one bank draw")
    keys = {(row.source_item_id, row.model_hash, row.budget_hash, row.bank_role) for row in rows}
    if len(keys) != 1:
        raise ContractViolation(f"difficulty is per (item, model, budget, role); got {sorted(keys)}")
    item, model_hash, budget_hash, role = keys.pop()
    uids = {row.generation_uid for row in rows}
    n_draws = len(uids)
    states = [grades.get(uid, STATE_UNKNOWN) for uid in sorted(uids)]
    c = sum(1 for s in states if s == STATE_CORRECT)
    w = sum(1 for s in states if s == STATE_WRONG)
    u = n_draws - c - w
    g = c + w
    smoothed = (c + 0.5) / (g + 1)
    lo, hi = beta_ci(c, g, alpha=ci_alpha)
    return Difficulty(
        source_item_id=item, model_hash=model_hash, budget_hash=budget_hash, bank_role=role,
        n_draws=n_draws, g_resolved=g, c=c, w=w, u=u,
        raw=(c / g) if g else None, operational=c / n_draws, lo=c / n_draws, hi=(c + u) / n_draws,
        smoothed=smoothed, bin=difficulty_bin(smoothed, cutpoints), beta_ci_lo=lo, beta_ci_hi=hi)


def difficulty_table(rows: Sequence[tables.BankRow], grades: Mapping[str, str], *,
                     cutpoints: Sequence[float], ci_alpha: float) -> list[Difficulty]:
    """One :class:`Difficulty` per ``(item, model, budget, role)`` group."""
    groups: dict[tuple[str, str, str, str], list[tables.BankRow]] = {}
    for row in rows:
        groups.setdefault((row.source_item_id, row.model_hash, row.budget_hash, row.bank_role),
                          []).append(row)
    return [difficulty(group, grades, cutpoints=cutpoints, ci_alpha=ci_alpha)
            for _key, group in sorted(groups.items())]


def bank_spec_draws(banks: K.BankSpec) -> dict[str, int]:
    """The declared size of each bank (``contract.BankSpec``), for the planner's census."""
    return {BankRole.DIFFICULTY.value: banks.difficulty_draws,
            BankRole.DONOR.value: banks.donor_initial,
            f"{BankRole.DONOR.value}_max": banks.donor_max,
            BankRole.NATURAL_ROOT.value: banks.natural_root_draws}


# --------------------------------------------------------------------------- the worker


def bank_worker(runner: Any, source_item_id: str) -> tables.ItemFile:
    """The DIFFICULTY_BANK / DONOR_BANK / NATURAL_ROOT cell worker.

    The runner contract this worker reads (WP4 supplies it):

    ``runner.cell``              the :class:`CellSpec` (``cell_id``, ``kind``, ``options``)
    ``runner.run_root``          the run root holding the public item store
    ``runner.contract``          the compiled :class:`~agents_scaling.nextdyn.contract.StudyContract`
    ``runner.model``             the :class:`ModelCell` this cell generates with
    ``runner.tokens``            the :class:`TokenContract` of this cell
    ``runner.top_logprobs``      how many final-token logprobs to request
    ``runner.generate(request)`` ``Generate`` bound to the store, pool and cost function
    ``runner.tokenizer_for(model)``  the token counter for the public-output parse
    ``runner.item(source_item_id)`` (optional) a cached item, else the public store is read

    Every draw is one independent sample: distinct ``agent_slot``, therefore distinct
    ``decision_id`` and distinct ``generation_uid``, and the same content twice is still two
    samples (``test_same_content_does_not_imply_same_sample``).
    """
    from agents_scaling.nextdyn import inference, telemetry

    cell = runner.cell
    role = role_of(cell.kind)
    contract = runner.contract
    model, tokens = runner.model, runner.tokens
    stage = int(dict(getattr(cell, "options", {}) or {}).get(EXTENSION_STAGE_OPTION, STAGE_INITIAL))
    item = (runner.item(source_item_id) if hasattr(runner, "item")
            else I.load_items(runner.run_root)[source_item_id])
    variant = item.primary
    messages = prompts.render_root(variant, item)
    counter = runner.tokenizer_for(model)
    rows: list[tables.DecisionRow] = []
    bank_rows: list[tables.BankRow] = []
    for draw in draw_range(role, contract.banks, extension_stage=stage):
        key = bank_key(role, source_item_id, variant.item_variant_id, model, tokens, draw,
                       study_id=contract.study_id)
        request = inference.DecisionRequest(
            messages=messages, model=model, adapter=model.adapter, tokens=tokens,
            seed_key=ID.seed_key_for(key, split_role=item.split_role), purpose=key.phase,
            top_logprobs=runner.top_logprobs, key=key)
        record = runner.generate(request)
        out = schema.parse_public_output(record.final_text, record.finish_reason,
                                         record.completion_token_ids, counter, limits=tokens)
        decision = telemetry.decision_row(
            record, key, out,
            cell=telemetry.CellContext(cell_id=cell.cell_id, episode_id=bank_draw_id(role, key),
                                       display_label=None, bundle_export_hash=None),
            parents=(), seen_transmissions=(), aliased=record.aliased, reuse_of=None)
        rows.append(decision)
        bank_rows.append(bank_row(role, key, decision, extension_stage=stage,
                                  cell_id=cell.cell_id))
    return tables.ItemFile(
        schema_version=tables.SCHEMA_VERSION, cell_id=cell.cell_id, cell_kind=cell.kind,
        source_item_id=source_item_id, status=tables.ItemStatus.COMPLETE,
        bundle_export_hash=None,
        rows={"decisions": tuple(rows), "banks": tuple(bank_rows)})


__all__ = [
    "BankRole", "Difficulty", "EXTENSION_STAGE_OPTION", "ExtensionRule", "KIND_ROLES",
    "ROLE_KINDS", "STAGE_EXTENDED", "STAGE_INITIAL", "bank_draw_id", "bank_key", "bank_row",
    "bank_spec_draws", "bank_worker", "beta_ci", "cutpoints_of", "difficulty", "difficulty_bin",
    "difficulty_table", "draw_range", "extension_authorized", "role_of",
]
