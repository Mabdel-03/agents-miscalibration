"""Build the public item store and the protected gold (spec 5.1, 5.2, 17.3; plan §2.1, §2.2).

**Writer side (R7).**  This module and :mod:`agents_scaling.nextdyn.datasets` are the only
ones that see a reference answer while the item table is built; everything downstream reads
:mod:`agents_scaling.nextdyn.items`.

Three things happen here, once, before any generation:

1. **Identity.**  ``source_item_id = sha256(JCS([study_id, family, revision, original_id]))[:16]``
   -- the study id and the dataset revision are inside, so no nextdyn item id can collide
   with an ARD36 one and a re-pinned revision is a different item (spec 17.2).
2. **Permutation.**  :func:`option_permutation` is ``ard/items_build.py:334-361`` generalized
   from ten options to ``n`` (rejection-sampled Fisher-Yates over an HMAC stream, so it is
   uniform, reproducible and correctness-blind).  The permuted **key letter** is written only
   to ``protected/gold/<family>.jsonl``; the public variant row carries the permutation and
   the displayed options, never the key (spec 5.1).
3. **Rendering.**  ``rendered_task_hash`` is the hash of the displayed task text and
   ``prompt_hash`` the hash of ``(family, answer format, task text, displayed options,
   answer instruction)`` -- the bytes an agent will actually read, so a renderer change
   invalidates the request cache instead of silently re-using it (R5).

The frame columns of ``source_items`` (``prior_use_status``, ``split_role``, ``frame_rule``,
``sampling_weight``) are not defaults filled in later: :func:`build_items` calls
:mod:`agents_scaling.nextdyn.inventory` and writes each row once, complete.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import datasets as D
from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn import inventory as INV
from agents_scaling.nextdyn import items as I
from agents_scaling.nextdyn import tables
from agents_scaling.nextdyn.config import StudyConfig
from agents_scaling.nextdyn.contract import ORDER_OPTION_PERM
from agents_scaling.nextdyn.errors import ContractViolation
from agents_scaling.nextdyn.graders import families as GF
from agents_scaling.nextdyn.graders import mc as MC
from agents_scaling.study import identity

#: Length of a ``source_item_id`` (hex characters).
ITEM_ID_LEN: Final[int] = 16
#: Permutation variants: 0 is the one every agent sees, 1 is the spec-14.4 control.
PERMUTATION_VARIANTS: Final[tuple[int, ...]] = (0, 1)
SUMMARY_FILE: Final[str] = "items_summary.json"


# --------------------------------------------------------------------------- identity


def source_item_id(study_id: str, family: str, revision: str | None, original_id: str) -> str:
    """``sha256(JCS([study_id, family, revision, original_id]))[:16]`` (spec 5.2)."""
    if not original_id:
        raise ContractViolation("an item with no original id cannot be given a source_item_id "
                                "(failure_policy.null_item_ids = fatal)")
    return identity.sha256_hex(identity.jcs(
        [str(study_id), str(family), revision, str(original_id)]))[:ITEM_ID_LEN]


def item_variant_id(item_id: str, variant_kind: str, permutation: Sequence[int]) -> str:
    """Content-addressed over the permutation, so two variants never share an id."""
    if variant_kind not in I.VARIANT_KINDS:
        raise ContractViolation(f"unknown variant kind {variant_kind!r}; expected {I.VARIANT_KINDS}")
    digest = identity.sha256_hex(identity.jcs([item_id, variant_kind, [int(p) for p in permutation]]))
    return f"{item_id}.{digest[:8]}"


# --------------------------------------------------------------------------- permutation


def permutation_from_digest(digest: bytes, n_options: int) -> tuple[int, ...]:
    """Rejection-sampled draw without replacement from an HMAC stream.

    Vendored verbatim in behaviour from ``ard/items_build.py:344-361`` with the fixed ten
    replaced by ``n_options``; ``test_graders.py`` / ``test_items_build.py`` assert parity at
    ``n = 10`` against the ARD36 function.
    """
    remaining = list(range(int(n_options)))
    permutation: list[int] = []
    counter = 0
    while remaining:
        stream = hashlib.sha256(digest + counter.to_bytes(4, "big")).digest()
        counter += 1
        for byte in stream:
            if not remaining:
                break
            limit = len(remaining)
            bound = 256 - (256 % limit)
            if byte >= bound:              # rejection sampling keeps the draw uniform
                continue
            permutation.append(remaining.pop(byte % limit))
    return tuple(permutation)


def option_permutation(item_id: str, study_seed: bytes, n_options: int, variant: int = 0) -> tuple[int, ...]:
    """``displayed position -> canonical option index`` for one item and variant.

    Seeded by ``blind_order_key(study_seed, "ND_OPTION_PERM", item_id, variant)``: an HMAC,
    so the permutation is reproducible, independent of iteration order and blind to which
    option is correct.
    """
    if variant not in PERMUTATION_VARIANTS:
        raise ContractViolation(f"unknown permutation variant {variant!r}; expected {PERMUTATION_VARIANTS}")
    MC.mc_letters(n_options)   # the alphabet bound is the one declaration of "how many options"
    digest = identity.blind_order_key(study_seed, ORDER_OPTION_PERM, item_id, variant)
    return permutation_from_digest(digest, n_options)


def permuted_key_letter(permutation: Sequence[int], canonical_index: int) -> str:
    """The displayed letter of the reference option: **protected**, never public (spec 5.1)."""
    return MC.displayed_letter(canonical_index, permutation)


# --------------------------------------------------------------------------- hashes


def rendered_task_hash(task_text: str, options: Sequence[str]) -> str:
    return identity.sha256_hex(identity.jcs({"task_text": task_text, "options": list(options)}))


def prompt_hash(family: str, answer_format: str, task_text: str, options: Sequence[str],
                answer_instruction: str) -> str:
    """The identity of the bytes an agent reads for this variant (R5)."""
    return identity.sha256_hex(identity.jcs({
        "answer_format": answer_format, "answer_instruction": answer_instruction,
        "family": family, "options": list(options), "task_text": task_text,
    }))


# --------------------------------------------------------------------------- drafts


@dataclass(frozen=True)
class VariantDraft:
    """One built variant, before the public and protected halves are split apart."""

    item_variant_id: str
    variant_kind: str
    permutation: tuple[int, ...]
    options: tuple[str, ...]
    task_text: str

    def row(self, item_id: str, family: str, answer_format: str, instruction: str) -> tables.ItemVariantRow:
        return tables.ItemVariantRow(
            item_variant_id=self.item_variant_id, source_item_id=item_id,
            variant_kind=self.variant_kind,
            option_permutation_json=json.dumps(list(self.permutation)),
            rendered_task_hash=rendered_task_hash(self.task_text, self.options),
            prompt_hash=prompt_hash(family, answer_format, self.task_text, self.options, instruction))


@dataclass(frozen=True)
class ItemDraft:
    """One item with everything both halves need; :func:`build_items` splits it."""

    source_item_id: str
    family: str
    row: D.SourceRow
    variants: tuple[VariantDraft, ...]

    @property
    def task_family(self) -> str:
        return F.TASK_FAMILY[self.family]


def build_variants(item_id: str, row: D.SourceRow, spec: F.FamilySpec,
                   study_seed: bytes) -> tuple[VariantDraft, ...]:
    """The canonical variant, plus the two permuted ones for a multiple-choice family."""
    render = spec.renderer
    canonical = VariantDraft(
        item_variant_id=item_variant_id(item_id, I.VARIANT_CANONICAL, range(len(row.options))),
        variant_kind=I.VARIANT_CANONICAL, permutation=tuple(range(len(row.options))),
        options=row.options, task_text=render(row.question, row.options))
    if spec.answer_format != F.ANSWER_FORMAT_MC:
        return (canonical,)
    out = [canonical]
    for variant, kind in zip(PERMUTATION_VARIANTS, (I.VARIANT_PERM_V0, I.VARIANT_PERM_V1)):
        permutation = option_permutation(item_id, study_seed, len(row.options), variant)
        options = tuple(row.options[index] for index in permutation)
        out.append(VariantDraft(item_variant_id=item_variant_id(item_id, kind, permutation),
                                variant_kind=kind, permutation=permutation, options=options,
                                task_text=render(row.question, options)))
    return tuple(out)


def build_drafts(cfg: StudyConfig, hf_home: str | os.PathLike) -> dict[str, list[ItemDraft]]:
    """``{family: [ItemDraft]}`` for every panel of every cohort family in ``study.yaml``."""
    study_seed = cfg.contract.study_seed
    panels: dict[str, list[ItemDraft]] = {}
    math500 = D.load_panel(F.MATH500, hf_home)
    loaded = {F.MATH500: math500}
    exclude = frozenset(row.normalized_text_hash for row in math500.rows)
    for cohort in cfg.cohorts.families:
        for name in F.panels_of(cohort):
            spec = F.family(name)
            panel = loaded.get(name) or D.load_panel(name, hf_home, exclude_text_hashes=exclude)
            loaded[name] = panel
            drafts: list[ItemDraft] = []
            for row in panel.rows:
                item_id = source_item_id(cfg.study.id, name, spec.loader.revision, row.original_id)
                drafts.append(ItemDraft(source_item_id=item_id, family=name, row=row,
                                        variants=build_variants(item_id, row, spec, study_seed)))
            panels[name] = drafts
    return panels


# --------------------------------------------------------------------------- gold


def gold_records(drafts: Sequence[ItemDraft]) -> list[dict[str, Any]]:
    """The protected records of one family: the reference plus the per-variant key letters."""
    out: list[dict[str, Any]] = []
    for draft in drafts:
        record: dict[str, Any] = {GF.GOLD_ITEM_KEY: draft.source_item_id,
                                  "family": draft.family,
                                  "original_id": draft.row.original_id}
        record.update({key: draft.row.gold[key] for key in sorted(draft.row.gold)})
        index = draft.row.gold.get("canonical_index")
        if isinstance(index, int) and not isinstance(index, bool):
            record["permuted_keys"] = {v.item_variant_id: permuted_key_letter(v.permutation, index)
                                       for v in draft.variants}
        out.append(record)
    return out


# --------------------------------------------------------------------------- writing


def _option_map_json(row: D.SourceRow) -> str | None:
    """Canonical option identity: the ordered content hashes of the *canonical* options, so
    analysis can check that two variants present the same option set without the text."""
    if not row.options:
        return None
    return json.dumps([identity.sha256_hex(MC.normalise_option_text(o).encode("utf-8"))[:16]
                       for o in row.options])


def source_item_row(draft: ItemDraft, *, frame: INV.FrameAssignment, grader: str) -> tables.SourceItemRow:
    spec = F.family(draft.family)
    row = draft.row
    return tables.SourceItemRow(
        source_item_id=draft.source_item_id, family=draft.family,
        dataset_id=spec.loader.repo_id or draft.family,
        dataset_revision=spec.loader.revision or "",
        source_split=spec.loader.split or "", original_id=row.original_id,
        subject=row.subject, level=row.level, answer_format=spec.answer_format,
        n_options=row.n_options, option_map_json=_option_map_json(row),
        normalized_text_hash=row.normalized_text_hash, variant_parent_id=None,
        prior_use_status=frame.prior_use_status,
        prior_use_sources_json=json.dumps(list(frame.prior_use_sources)),
        near_duplicate_of=frame.near_duplicate_of, sampling_weight=frame.sampling_weight,
        frame_size=frame.frame_size, frame_rule=frame.frame_rule, split_role=frame.split_role,
        grader_hash=grader,
        gold_location=f"{GF.PROTECTED_DIR}/{GF.GOLD_DIR}/{draft.family}{GF.GOLD_SUFFIX}"
                      f"#{draft.source_item_id}")


def build_items(run_root: str | os.PathLike, cfg: StudyConfig, *, split_salt: bytes,
                study_seed: bytes, hf_home: str | os.PathLike | None = None,
                registry: D.LegacyRegistry | None = None) -> None:
    """Build the whole public item store and the protected gold, once (spec 5.2).

    ``split_salt`` and ``study_seed`` are passed explicitly rather than read off ``cfg`` so a
    test can build a small store with its own salt without editing a config; the verb passes
    ``cfg.contract.split_salt`` / ``.study_seed``.  ``registry`` names the legacy roots the
    prior-use pass may read (default: ``configs/nextdyn/legacy_sources.yaml``); an empty one
    builds a store with no prior-use evidence, which is what a synthetic fixture wants.
    """
    from agents_scaling.nextdyn.locks import hf_home as default_hf_home

    home = hf_home if hf_home is not None else default_hf_home()
    panels = build_drafts(cfg, home)
    all_drafts = [draft for drafts in panels.values() for draft in drafts]
    prior_use = INV.prior_use_inventory(all_drafts, home, registry=registry)
    duplicates = INV.near_duplicate_audit(all_drafts)
    frames = INV.assign_split_roles(all_drafts, prior_use, duplicates, cfg, split_salt)

    graders = {family: GF.grader_hash(family) for family in panels}
    item_rows = [source_item_row(draft, frame=frames[draft.source_item_id],
                                 grader=graders[draft.family]) for draft in all_drafts]
    variant_rows: list[tables.ItemVariantRow] = []
    texts: list[I.ItemText] = []
    for draft in all_drafts:
        spec = F.family(draft.family)
        for variant in draft.variants:
            variant_rows.append(variant.row(draft.source_item_id, draft.family,
                                            spec.answer_format, spec.answer_instruction))
            texts.append(I.ItemText(item_variant_id=variant.item_variant_id,
                                    source_item_id=draft.source_item_id,
                                    task_text=variant.task_text, options=variant.options))
    split_rows = [frames[draft.source_item_id].split_row for draft in all_drafts]

    public = I.public_dir(run_root)
    public.mkdir(parents=True, exist_ok=True)
    tables.write_table(item_rows, public / f"{I.SOURCE_ITEMS_TABLE}.parquet")
    tables.write_table(variant_rows, public / f"{I.ITEM_VARIANTS_TABLE}.parquet")
    tables.write_table(split_rows, public / f"{I.ITEM_SPLITS_TABLE}.parquet")
    payload = "".join(json.dumps(text.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
                      for text in texts)
    for text in texts:
        I.assert_no_protected_key(text.to_dict(), where="item_texts.jsonl")
    I.texts_path(run_root).write_text(payload, encoding="utf-8")
    for family, drafts in panels.items():
        GF.write_gold(run_root, family, gold_records(drafts))

    summary = {
        "schema_version": tables.SCHEMA_VERSION,
        "study_id": cfg.study.id,
        "families": {family: len(drafts) for family, drafts in sorted(panels.items())},
        "n_items": len(item_rows),
        "n_variants": len(variant_rows),
        "n_near_duplicates": len(duplicates),
        "source_inventory_hash": INV.source_inventory_hash(item_rows),
        "prior_use_inventory_hash": INV.prior_use_inventory_hash(prior_use),
        "split_counts": INV.split_counts(split_rows),
    }
    (public / SUMMARY_FILE).write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_summary(run_root: str | os.PathLike) -> Mapping[str, Any]:
    path = I.public_dir(run_root) / SUMMARY_FILE
    if not path.exists():
        raise I.ItemsError(f"{path} does not exist; run the inventory verb first")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "ITEM_ID_LEN", "ItemDraft", "PERMUTATION_VARIANTS", "SUMMARY_FILE", "VariantDraft",
    "build_drafts", "build_items", "build_variants", "gold_records", "item_variant_id",
    "load_summary", "option_permutation", "permutation_from_digest", "permuted_key_letter", "prompt_hash",
    "rendered_task_hash", "source_item_id", "source_item_row",
]
