"""The runtime item reader and the gold firewall (spec 5.1, 5.2; plan §2.2; rule R7).

ARD36's contract rule 2 -- "gold/reference fields live in the evaluator namespace and must
not be loaded by serving/controller workers" -- was enforced by an import-closure test alone.
Here it is enforced **on the data as well**: :func:`load_items` refuses a public table that
carries any :data:`PROTECTED_KEYS` column or key, so a hand-edited or rebuilt store that
leaks an answer is rejected here rather than reaching a prompt.

This module's import closure is the standard library plus ``pyarrow`` plus
:mod:`agents_scaling.nextdyn.tables` and :mod:`agents_scaling.nextdyn.families`.  It never
imports a loader, a grader, the HF ``datasets`` library or
:mod:`agents_scaling.benchmarks`; :data:`WRITER_SIDE` is the list of nextdyn modules that
may, and ``tests/nextdyn/test_leakage.py`` walks the generation-side import closure against
it.

Layout written by :mod:`agents_scaling.nextdyn.items_build` (writer side)::

    <run_root>/data/public/source_items.parquet    the source_items table (tables.py columns)
    <run_root>/data/public/item_variants.parquet   the item_variants table
    <run_root>/data/public/item_splits.parquet     the item_splits table (written by inventory)
    <run_root>/data/public/item_texts.jsonl        the rendered task text + displayed options
    <run_root>/protected/gold/<family>.jsonl       0600 in 0700: the reference answers

The task *text* is deliberately not a table column: tables are the materialized analysis
surface, and an item's rendered statement is an input the generation lane reads once.  Its
one declaration is :class:`ItemText` here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn import tables
from agents_scaling.nextdyn.errors import LeakageError, ManifestError

DATA_DIR: Final[str] = "data"
PUBLIC_DIR: Final[str] = "public"
SOURCE_ITEMS_TABLE: Final[str] = "source_items"
ITEM_VARIANTS_TABLE: Final[str] = "item_variants"
ITEM_SPLITS_TABLE: Final[str] = "item_splits"
ITEM_TEXTS_FILE: Final[str] = "item_texts.jsonl"

#: Variant kinds (plan §1.4 ``item_variants``): the unpermuted reference, the frozen
#: permutation every agent sees, and the spec-14.4 answer-identity control.
VARIANT_CANONICAL: Final[str] = "canonical"
VARIANT_PERM_V0: Final[str] = "option_perm_v0"
VARIANT_PERM_V1: Final[str] = "option_perm_v1"
VARIANT_KINDS: Final[tuple[str, ...]] = (VARIANT_CANONICAL, VARIANT_PERM_V0, VARIANT_PERM_V1)

#: Keys that must never appear in the public item store (spec 5.1 "no option key is exposed").
PROTECTED_KEYS: Final[frozenset[str]] = frozenset({
    "answer", "answer_index", "answer_key", "permuted_key", "permuted_keys", "correct_answer",
    "correct_option", "solution", "cot_content", "gold", "reference", "residue", "is_correct",
    "correct", "labels", "canonical_index", "source_answer",
})

#: The nextdyn modules allowed to touch gold, a grade or a composition label (R7).  Every
#: other module's import closure is held clear of them, of ``agents_scaling.benchmarks``, of
#: ``agents_scaling.ard.graders`` and of the HF ``datasets`` library.
WRITER_SIDE: Final[tuple[str, ...]] = (
    "agents_scaling.nextdyn.datasets",
    "agents_scaling.nextdyn.items_build",
    "agents_scaling.nextdyn.inventory",
    "agents_scaling.nextdyn.grading",
    "agents_scaling.nextdyn.construct",
    "agents_scaling.nextdyn.graders.families",
    "agents_scaling.nextdyn.graders.math",
    "agents_scaling.nextdyn.graders.trace",
    "agents_scaling.nextdyn.e0",
    "agents_scaling.nextdyn.analysis",
)

#: Third-party / sibling packages the generation lane never reaches (R7).
FOREIGN_WRITER_SIDE: Final[tuple[str, ...]] = (
    "agents_scaling.benchmarks",
    "agents_scaling.ard.graders",
    "agents_scaling.study.data.protected",
    "datasets",
)


class ItemsError(ManifestError):
    """The public item store is missing, malformed or inconsistent."""


# --------------------------------------------------------------------------- layout


def data_dir(run_root: str | os.PathLike) -> Path:
    return Path(run_root) / DATA_DIR


def public_dir(run_root: str | os.PathLike) -> Path:
    return data_dir(run_root) / PUBLIC_DIR


def texts_path(run_root: str | os.PathLike) -> Path:
    return public_dir(run_root) / ITEM_TEXTS_FILE


# --------------------------------------------------------------------------- the public view


@dataclass(frozen=True)
class ItemText:
    """The rendered statement of one item variant: the only declaration of this file's keys."""

    item_variant_id: str
    source_item_id: str
    task_text: str
    options: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, where: str) -> "ItemText":
        assert_no_protected_key(data, where=where)
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ItemsError(f"{where}: unexpected keys {unknown}")
        missing = sorted(names - set(data))
        if missing:
            raise ItemsError(f"{where}: missing keys {missing}")
        return cls(item_variant_id=str(data["item_variant_id"]),
                   source_item_id=str(data["source_item_id"]),
                   task_text=str(data["task_text"]),
                   options=tuple(str(o) for o in data["options"]))

    def to_dict(self) -> dict[str, Any]:
        return {"item_variant_id": self.item_variant_id, "source_item_id": self.source_item_id,
                "task_text": self.task_text, "options": list(self.options)}


@dataclass(frozen=True)
class ItemVariant:
    """One presentation of an item: its permutation, its displayed options, its hashes."""

    item_variant_id: str
    source_item_id: str
    variant_kind: str
    permutation: tuple[int, ...]
    options: tuple[str, ...]
    task_text: str
    rendered_task_hash: str
    prompt_hash: str

    @property
    def rendered_question(self) -> str:
        """What a prompt template asks for (``prompts.TaskItem``): the displayed statement."""
        return self.task_text


@dataclass(frozen=True)
class SourceItem:
    """One public item: everything a generation worker may know about it, and nothing else."""

    source_item_id: str
    family: str
    task_family: str
    answer_format: str
    n_options: int | None
    answer_instruction: str
    subject: str | None
    level: str | None
    split_role: str
    prior_use_status: str
    normalized_text_hash: str
    grader_hash: str
    gold_location: str
    variants: tuple[ItemVariant, ...]

    @property
    def rendered_question(self) -> str:
        """The primary variant's statement (``prompts.TaskItem``)."""
        return self.primary.task_text

    @property
    def item_variant_id(self) -> str:
        """The variant every key of this item carries (``ids.CompositeKey.item_variant_id``)."""
        return self.primary.item_variant_id

    @property
    def canonical(self) -> ItemVariant:
        return self.variant_of(VARIANT_CANONICAL)

    @property
    def primary(self) -> ItemVariant:
        """The variant every agent sees: the frozen permutation for a multiple-choice family,
        the canonical rendering for an exact-answer one."""
        if self.answer_format == F.ANSWER_FORMAT_MC:
            return self.variant_of(VARIANT_PERM_V0)
        return self.canonical

    def variant_of(self, kind: str) -> ItemVariant:
        for variant in self.variants:
            if variant.variant_kind == kind:
                return variant
        raise ItemsError(f"{self.source_item_id} has no {kind!r} variant "
                         f"(has {[v.variant_kind for v in self.variants]})")

    def variant(self, item_variant_id: str) -> ItemVariant:
        for variant in self.variants:
            if variant.item_variant_id == item_variant_id:
                return variant
        raise ItemsError(f"{self.source_item_id} has no variant {item_variant_id!r}")


# --------------------------------------------------------------------------- firewall


def assert_no_protected_key(payload: Mapping[str, Any] | Sequence[str], *, where: str) -> None:
    """Raise :class:`LeakageError` if a gold-bearing key reached a public object."""
    keys = set(payload) if isinstance(payload, Mapping) else set(payload)
    leaked = sorted(keys & PROTECTED_KEYS)
    if leaked:
        raise LeakageError(f"{where}: protected key(s) {leaked} in the public item store; gold "
                           f"lives only under protected/gold (spec 5.1, R7)")


# --------------------------------------------------------------------------- reading


def _read_public_table(run_root: str | os.PathLike, name: str) -> list[dict[str, Any]]:
    directory = public_dir(run_root)
    path = directory / f"{name}.parquet"
    if not path.exists() and not (directory / f"{name}{tables.EMPTY_SUFFIX}").exists():
        raise ItemsError(f"{path} does not exist; run the inventory verb first")
    rows = tables.read_table(directory, name)
    for row in rows:
        assert_no_protected_key(row, where=f"{path}")
    return rows


def load_items(run_root: str | os.PathLike, *, family: str | None = None,
               split_role: str | None = None) -> Mapping[str, SourceItem]:
    """``{source_item_id: SourceItem}`` from the public store, fail-closed.

    Optional ``family`` / ``split_role`` filters are applied **after** the firewall check, so
    a leak in a row nobody asked for is still fatal.
    """
    items_rows = _read_public_table(run_root, SOURCE_ITEMS_TABLE)
    variant_rows = _read_public_table(run_root, ITEM_VARIANTS_TABLE)
    path = texts_path(run_root)
    if not path.exists():
        raise ItemsError(f"{path} does not exist; run the inventory verb first")
    texts: dict[str, ItemText] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        text = ItemText.from_dict(json.loads(line), where=f"{path}:{lineno}")
        texts[text.item_variant_id] = text

    variants: dict[str, list[ItemVariant]] = {}
    for row in variant_rows:
        text = texts.get(row["item_variant_id"])
        if text is None:
            raise ItemsError(f"{path} has no text for variant {row['item_variant_id']!r}")
        permutation = tuple(json.loads(row["option_permutation_json"] or "[]"))
        variants.setdefault(row["source_item_id"], []).append(ItemVariant(
            item_variant_id=row["item_variant_id"], source_item_id=row["source_item_id"],
            variant_kind=row["variant_kind"], permutation=permutation, options=text.options,
            task_text=text.task_text, rendered_task_hash=row["rendered_task_hash"],
            prompt_hash=row["prompt_hash"]))

    out: dict[str, SourceItem] = {}
    for row in items_rows:
        item_id = row["source_item_id"]
        if not item_id:
            raise ItemsError("a source_items row has a null source_item_id (spec 17.2: fatal)")
        if item_id in out:
            raise ItemsError(f"duplicate source_item_id {item_id!r} in the public store")
        spec = F.family(row["family"])
        mine = tuple(sorted(variants.get(item_id, ()), key=lambda v: VARIANT_KINDS.index(v.variant_kind)))
        if not mine:
            raise ItemsError(f"{item_id} has no item_variants row")
        out[item_id] = SourceItem(
            source_item_id=item_id, family=spec.name, task_family=F.TASK_FAMILY[spec.name],
            answer_format=row["answer_format"], n_options=row["n_options"],
            answer_instruction=spec.answer_instruction, subject=row["subject"], level=row["level"],
            split_role=row["split_role"], prior_use_status=row["prior_use_status"],
            normalized_text_hash=row["normalized_text_hash"], grader_hash=row["grader_hash"],
            gold_location=row["gold_location"], variants=mine)
    if not out:
        raise ItemsError(f"{public_dir(run_root)}: the public item store is empty")
    return {item_id: item for item_id, item in out.items()
            if (family is None or item.family == family)
            and (split_role is None or item.split_role == split_role)}


def load_splits(run_root: str | os.PathLike) -> list[dict[str, Any]]:
    """The ``item_splits`` rows as dicts (the planner's view; no outcome is involved)."""
    return _read_public_table(run_root, ITEM_SPLITS_TABLE)


__all__ = [
    "DATA_DIR", "FOREIGN_WRITER_SIDE", "ITEM_SPLITS_TABLE", "ITEM_TEXTS_FILE",
    "ITEM_VARIANTS_TABLE", "ItemText", "ItemVariant", "ItemsError", "PROTECTED_KEYS",
    "PUBLIC_DIR", "SOURCE_ITEMS_TABLE", "SourceItem", "VARIANT_CANONICAL", "VARIANT_KINDS",
    "VARIANT_PERM_V0", "VARIANT_PERM_V1", "WRITER_SIDE", "assert_no_protected_key", "data_dir",
    "load_items", "load_splits", "public_dir", "texts_path",
]
