"""The item writer: identity, permutation, rendering and the gold split (spec 5.1, 5.2; plan §2.2).

The synthetic panel here is deliberately large enough (420 items per cohort family) for the
real split policy to apply -- a 180-rank pilot block, a 200-item natural frame and every
sub-panel -- without touching the HF cache, so the split rules are tested rather than a
special case of them.
"""

from __future__ import annotations

import json
import stat

import pytest

from agents_scaling.nextdyn import datasets as D
from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn import items as I
from agents_scaling.nextdyn import items_build as IB
from agents_scaling.nextdyn import tables
from agents_scaling.nextdyn.errors import ContractViolation, LeakageError
from agents_scaling.nextdyn.graders import families as GF
from agents_scaling.nextdyn.graders import mc as MC
from agents_scaling.study import identity

SEED = bytes.fromhex("11" * 32)
SALT = bytes.fromhex("22" * 32)
#: Enough items for the 180-rank pilot block plus a 200-item natural frame and every sub-panel.
PER_FAMILY = 420
EMPTY_REGISTRY = D.LegacyRegistry(path=None, roots={}, prior_roots=())


def math_row(index: int) -> D.SourceRow:
    return D.SourceRow(family=F.MATH500, original_id=f"test/algebra/{index}.json",
                       question=f"Compute the value of {index} plus seven.", options=(),
                       gold={"answer": str(index + 7), "solution": f"\\boxed{{{index + 7}}}"},
                       subject="Algebra", level="Level 3")


def mc_row(index: int, n_options: int = F.MMLU_REQUIRED_OPTIONS) -> D.SourceRow:
    return D.SourceRow(
        family=F.MMLU_PRO, original_id=f"mmlu_pro-{index}",
        question=f"Which option is the {index}th answer?",
        options=tuple(f"answer {index}.{position}" for position in range(n_options)),
        gold={"canonical_index": index % n_options, "source_answer": "x"},
        subject="engineering", level=None)


def draft_of(row: D.SourceRow, *, study_id: str = "task_conditioned_dynamics_v1",
             study_seed: bytes = SEED) -> IB.ItemDraft:
    spec = F.family(row.family)
    item_id = IB.source_item_id(study_id, row.family, spec.loader.revision, row.original_id)
    return IB.ItemDraft(source_item_id=item_id, family=row.family, row=row,
                        variants=IB.build_variants(item_id, row, spec, study_seed))


def synthetic_drafts(per_family: int = PER_FAMILY) -> dict[str, list[IB.ItemDraft]]:
    return {F.MATH500: [draft_of(math_row(index)) for index in range(per_family)],
            F.MMLU_PRO: [draft_of(mc_row(index)) for index in range(per_family)]}


@pytest.fixture(scope="module")
def built(tmp_path_factory, request):
    """A complete public item store built from the synthetic panel (no HF cache needed)."""
    from agents_scaling.nextdyn import config as C
    from tests.nextdyn.conftest import CONFIG_DIR

    cfg = C.load_study(CONFIG_DIR / "study.yaml", mode="test")
    run_root = tmp_path_factory.mktemp("nd_items")
    panels = synthetic_drafts()
    monkey = pytest.MonkeyPatch()
    monkey.setattr(IB, "build_drafts", lambda cfg_, home: panels)
    try:
        IB.build_items(run_root, cfg, split_salt=SALT, study_seed=SEED, hf_home=run_root,
                       registry=EMPTY_REGISTRY)
    finally:
        monkey.undo()
    return run_root, panels, cfg


# --------------------------------------------------------------------------- identity


def test_source_item_id_never_null():
    """Spec 18.1: an item id is never empty, never derived from an empty original id, and
    carries the study and the dataset revision so it cannot alias an ARD36 item."""
    ids = {IB.source_item_id("s", F.MATH500, "rev", f"test/algebra/{i}.json") for i in range(200)}
    assert len(ids) == 200
    assert all(len(value) == IB.ITEM_ID_LEN and value.strip() for value in ids)
    assert (IB.source_item_id("s", F.MATH500, "rev", "x")
            != IB.source_item_id("s", F.MATH500, "rev2", "x"))
    assert (IB.source_item_id("s", F.MATH500, "rev", "x")
            != IB.source_item_id("other", F.MATH500, "rev", "x"))
    for empty in ("", None):
        with pytest.raises(ContractViolation):
            IB.source_item_id("s", F.MATH500, "rev", empty)


def test_item_variant_ids_are_content_addressed_over_the_permutation():
    row = mc_row(1)
    draft = draft_of(row)
    kinds = {variant.variant_kind for variant in draft.variants}
    assert kinds == {I.VARIANT_CANONICAL, I.VARIANT_PERM_V0, I.VARIANT_PERM_V1}
    assert len({variant.item_variant_id for variant in draft.variants}) == 3
    assert all(variant.item_variant_id.startswith(draft.source_item_id)
               for variant in draft.variants)
    assert len(draft_of(math_row(1)).variants) == 1     # an exactMatch item has one rendering


# --------------------------------------------------------------------------- permutation


def test_option_permutation_matches_the_ard36_function_at_ten_options():
    """Vendoring parity (``ard/items_build.py:334-361`` generalized to n options)."""
    from agents_scaling.ard.items_build import NS_OPTION_PERMUTATION
    from agents_scaling.ard.items_build import option_permutation as ard_permutation

    for index in range(25):
        item = f"MMLU_PRO-{index:06d}"
        for variant in (0, 1):
            digest = identity.blind_order_key(SEED, NS_OPTION_PERMUTATION, item, variant)
            assert (IB.permutation_from_digest(digest, 10)
                    == ard_permutation(item, SEED, variant=variant))


@pytest.mark.parametrize("n_options", [4, 5, 10, 13, 26])
def test_option_permutation_is_a_permutation_and_moves_with_the_variant(n_options):
    first = IB.option_permutation("item0001", SEED, n_options, 0)
    second = IB.option_permutation("item0001", SEED, n_options, 1)
    assert sorted(first) == list(range(n_options))
    assert sorted(second) == list(range(n_options))
    assert IB.option_permutation("item0001", SEED, n_options, 0) == first     # deterministic
    assert IB.option_permutation("item0002", SEED, n_options, 0) != first
    if n_options > 3:
        assert first != second
    with pytest.raises(ContractViolation):
        IB.option_permutation("item0001", SEED, n_options, 2)


def test_the_permuted_key_letter_is_the_inverse_of_the_permutation():
    permutation = IB.option_permutation("item0001", SEED, 10, 0)
    for canonical_index in range(10):
        letter = IB.permuted_key_letter(permutation, canonical_index)
        assert MC.canonical_option_id(letter, permutation) == canonical_index


# --------------------------------------------------------------------------- rendering


def test_render_and_prompt_hashes_separate_the_variants_and_stay_stable():
    draft = draft_of(mc_row(3))
    spec = F.family(F.MMLU_PRO)
    rows = [variant.row(draft.source_item_id, draft.family, spec.answer_format,
                        spec.answer_instruction) for variant in draft.variants]
    assert len({row.prompt_hash for row in rows}) == 3
    assert len({row.rendered_task_hash for row in rows}) == 3
    again = draft_of(mc_row(3))
    assert [v.task_text for v in again.variants] == [v.task_text for v in draft.variants]
    assert (IB.prompt_hash(F.MMLU_PRO, "multipleChoice", "q", ["a"], "instruction")
            != IB.prompt_hash(F.MMLU_PRO, "multipleChoice", "q", ["a"], "other instruction"))


# --------------------------------------------------------------------------- the built store


def test_build_items_writes_the_public_store_and_the_protected_gold(built):
    run_root, panels, _cfg = built
    items = I.load_items(run_root)
    assert len(items) == 2 * PER_FAMILY
    summary = IB.load_summary(run_root)
    assert summary["n_items"] == len(items)
    assert summary["families"] == {name: len(drafts) for name, drafts in panels.items()}
    assert len(summary["source_inventory_hash"]) == 64
    for family in panels:
        path = GF.gold_path(run_root, family)
        assert stat.S_IMODE(path.stat().st_mode) == GF.PROTECTED_FILE_MODE
    assert stat.S_IMODE(GF.gold_dir(run_root).stat().st_mode) == GF.PROTECTED_DIR_MODE


def test_the_public_store_never_carries_gold_or_a_key_letter(built):
    run_root, _panels, _cfg = built
    columns = set(tables.schema("source_items").column_names) | set(
        tables.schema("item_variants").column_names)
    assert not columns & I.PROTECTED_KEYS
    text = I.texts_path(run_root).read_text(encoding="utf-8")
    store = GF.GoldStore.load(run_root)
    record = next(iter(store.records[F.MMLU_PRO].values()))
    letter = next(iter(record["permuted_keys"].values()))
    assert record["canonical_index"] is not None and letter in MC.mc_letters(10)
    for key in ("permuted_keys", "canonical_index", "solution"):
        assert f'"{key}"' not in text
    with pytest.raises(LeakageError):
        I.assert_no_protected_key({"source_item_id": "x", "answer": "4"}, where="test")


def test_the_primary_variant_is_the_frozen_permutation_for_multiple_choice(built):
    run_root, _panels, _cfg = built
    items = I.load_items(run_root, family=F.MMLU_PRO)
    item = next(iter(items.values()))
    assert item.primary.variant_kind == I.VARIANT_PERM_V0
    assert item.item_variant_id == item.primary.item_variant_id
    assert item.rendered_question == item.primary.task_text
    assert sorted(item.primary.permutation) == list(range(10))
    assert item.canonical.permutation == tuple(range(10))
    math_item = next(iter(I.load_items(run_root, family=F.MATH500).values()))
    assert math_item.primary.variant_kind == I.VARIANT_CANONICAL
    assert math_item.n_options is None


def test_the_displayed_options_are_the_permuted_canonical_options(built):
    run_root, panels, _cfg = built
    items = I.load_items(run_root, family=F.MMLU_PRO)
    by_original = {draft.source_item_id: draft for draft in panels[F.MMLU_PRO]}
    for item_id, item in list(items.items())[:20]:
        canonical = by_original[item_id].row.options
        for variant in item.variants:
            assert list(variant.options) == [canonical[index] for index in variant.permutation]
        option_map = json.loads(
            next(row for row in tables.read_table(I.public_dir(run_root), "source_items")
                 if row["source_item_id"] == item_id)["option_map_json"])
        assert len(option_map) == len(canonical)


def test_a_gold_record_exists_for_every_item_and_names_its_variants(built):
    run_root, panels, _cfg = built
    store = GF.GoldStore.load(run_root)
    for family, drafts in panels.items():
        table = store.records[family]
        assert set(table) == {draft.source_item_id for draft in drafts}
        record = table[drafts[0].source_item_id]
        if family == F.MMLU_PRO:
            assert set(record["permuted_keys"]) == {v.item_variant_id for v in drafts[0].variants}
        assert len(store.ref_hash(family, drafts[0].source_item_id)) == 64
