"""WP3 data tests: stratified splits, group inheritance, the firewall, permutations.

Spec §5.1 / §18; plan §5.1 and §10 ("firewall ... transitive import-closure scan",
"MMLU permutation round-trip").  Everything here runs on synthetic pinned rows so the
group is hermetic: no HF cache, no ``datasets`` import, no network.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from agents_scaling.ard import graders, items, items_build
from agents_scaling.ard.contract import (
    MMLU_REQUIRED_OPTIONS,
    SPLIT_COUNTS_PER_FAMILY,
    SPLITS,
    Family,
)

COT_MARKER = "COT-SECRET-MUST-NEVER-BE-EXPORTED"
SOLUTION_MARKER = "SOLUTION-GOLD-MARKER"

MATH_SUBJECTS = ("Algebra", "Geometry", "Number Theory", "Prealgebra", "Precalculus",
                 "Intermediate Algebra", "Counting & Probability")
MMLU_CATEGORIES = ("biology", "business", "chemistry", "computer science", "economics",
                   "engineering", "health", "history", "law", "math", "other",
                   "philosophy", "physics", "psychology")


def _math_rows(n: int = 500) -> list[dict]:
    rows = []
    for index in range(n):
        subject = MATH_SUBJECTS[index % len(MATH_SUBJECTS)]
        level = 1 + (index // len(MATH_SUBJECTS)) % 5
        rows.append({
            "unique_id": f"test/{subject.lower().replace(' ', '_')}/{index}.json",
            "problem": f"Problem number {index}: compute {index} + {level}.",
            "solution": f"{SOLUTION_MARKER} we get $\\boxed{{{index + level}}}$.",
            "answer": str(index + level),
            "subject": subject,
            "level": level,
        })
    return rows


def _mmlu_rows(n: int = 700) -> list[dict]:
    rows = []
    for index in range(n):
        category = MMLU_CATEGORIES[index % len(MMLU_CATEGORIES)]
        options = [f"option {index}-{letter}" for letter in range(MMLU_REQUIRED_OPTIONS)]
        rows.append({
            "question_id": index,
            "question": f"Question number {index} about {category}?  ",
            "options": options,
            "answer": "C",
            "answer_index": index % MMLU_REQUIRED_OPTIONS,
            "cot_content": f"{COT_MARKER} step {index}",
            "category": category,
            "src": "synthetic",
        })
    # one four-option row (ineligible) and one duplicate question string (de-duplicated)
    rows.append({"question_id": n, "question": "short row?", "options": ["a", "b", "c", "d"],
                 "answer": "A", "answer_index": 0, "cot_content": COT_MARKER,
                 "category": "law", "src": "synthetic"})
    duplicate = dict(rows[0])
    duplicate["question_id"] = n + 1
    duplicate["answer_index"] = 1
    rows.append(duplicate)
    return rows


def _sources() -> items_build.RawSources:
    return items_build.RawSources(
        math_rows=tuple(_math_rows()),
        mmlu_rows=tuple(_mmlu_rows()),
        math_revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        mmlu_revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
    )


@pytest.fixture(scope="module")
def sources() -> items_build.RawSources:
    return _sources()


@pytest.fixture(scope="module")
def built(tmp_path_factory, sources) -> Path:
    root = tmp_path_factory.mktemp("ard36_items")
    items_build.build_items(root, sources=sources)
    return root


# --------------------------------------------------------------------------- splits


def test_split_counts_are_the_frozen_per_family_totals(built):
    tasks = items.load_ard_tasks(built)
    assert len(tasks) == 2 * sum(SPLIT_COUNTS_PER_FAMILY.values())
    for family in Family:
        for split in SPLITS:
            block = [task for task in tasks.values()
                     if task.family == family.value and task.split == split]
            assert len(block) == SPLIT_COUNTS_PER_FAMILY[split], (family, split)


def test_splits_are_disjoint_and_ranks_are_contiguous(built):
    tasks = items.load_ard_tasks(built)
    by_source: dict[str, set[str]] = {}
    for task in tasks.values():
        by_source.setdefault(task.source_id, set()).add(task.split)
    assert all(len(splits) == 1 for splits in by_source.values())
    assert len(by_source) == len(tasks)
    for family in Family:
        for split in SPLITS:
            ranks = sorted(task.rank for task in tasks.values()
                           if task.family == family.value and task.split == split)
            assert ranks == list(range(len(ranks)))


def test_split_assignment_is_deterministic(tmp_path, sources):
    first, second = tmp_path / "a", tmp_path / "b"
    items_build.build_items(first, sources=sources)
    items_build.build_items(second, sources=sources)
    assert items.items_path(first).read_bytes() == items.items_path(second).read_bytes()
    assert {i: t.split for i, t in items.load_ard_tasks(first).items()} == \
           {i: t.split for i, t in items.load_ard_tasks(second).items()}


def test_a_different_salt_moves_items(tmp_path, sources):
    other = tmp_path / "other"
    items_build.build_items(other, sources=sources, split_salt=b"ard36/v1/some-other-salt")
    base = items.load_ard_tasks(_built_once(tmp_path, sources))
    moved = items.load_ard_tasks(other)
    assert {t.source_id for t in base.values()} != {t.source_id for t in moved.values()}


def _built_once(tmp_path: Path, sources) -> Path:
    root = tmp_path / "base"
    if not items.items_path(root).exists():
        items_build.build_items(root, sources=sources)
    return root


def test_every_stratum_contributes_at_least_one_item(built):
    summary = json.loads((items.public_dir(built) / items.ITEMS_SUMMARY_FILE).read_text())
    for family, allocation in summary["stratum_allocation"].items():
        assert min(allocation.values()) >= 1, family
        assert sum(allocation.values()) == sum(SPLIT_COUNTS_PER_FAMILY.values())
    tasks = items.load_ard_tasks(built)
    drawn = {(task.family, task.stratum) for task in tasks.values()}
    for family, allocation in summary["stratum_allocation"].items():
        for stratum in allocation:
            assert (family, stratum) in drawn


def test_strata_are_spread_across_the_three_splits(built):
    """Streaming largest-remainder keeps each stratum proportional, not clumped."""
    tasks = items.load_ard_tasks(built)
    for family in Family:
        for split in SPLITS:
            strata = {task.stratum for task in tasks.values()
                      if task.family == family.value and task.split == split}
            assert len(strata) >= 5, (family, split, strata)


def test_reserve_is_disjoint_from_the_panel(built):
    reserve = json.loads((items.public_dir(built) / items.RESERVE_FILE).read_text())["reserve"]
    tasks = items.load_ard_tasks(built)
    drawn = {task.source_id for task in tasks.values()}
    for family, ids in reserve.items():
        assert not (set(ids) & drawn), family
        assert len(set(ids)) == len(ids)


def test_only_ten_option_rows_are_eligible(built, sources):
    eligibility = items_build.mmlu_candidates(sources.mmlu_rows)
    assert eligibility.dropped["wrong_option_count"] == 1
    assert eligibility.dropped["duplicate_question"] == 1
    assert all(len(candidate.options) == MMLU_REQUIRED_OPTIONS
               for candidate in eligibility.candidates)
    tasks = items.load_ard_tasks(built, family=Family.MMLUPRO)
    assert all(task.n_options == MMLU_REQUIRED_OPTIONS for task in tasks.values())


def test_duplicate_question_strings_are_removed_before_sampling(sources):
    eligibility = items_build.mmlu_candidates(sources.mmlu_rows)
    questions = [candidate.task_text for candidate in eligibility.candidates]
    assert len(questions) == len(set(questions))


# --------------------------------------------------------------------------- groups


def test_group_id_is_the_source_item_and_is_inherited(built):
    tasks = items.load_ard_tasks(built)
    for task in tasks.values():
        assert task.group_id == task.item_id
    assert items.group_ids(tasks) == {i: t.group_id for i, t in tasks.items()}
    # every variant of an item (permutation, challenge team, fork, suffix) inherits the
    # group of its source item, so a group never spans two source items.
    groups: dict[str, set[str]] = {}
    for task in tasks.values():
        groups.setdefault(task.group_id, set()).add(task.source_id)
    assert all(len(sources) == 1 for sources in groups.values())


# --------------------------------------------------------------------------- firewall


def test_public_table_carries_no_protected_column(built):
    import pyarrow.parquet as pq

    table = pq.read_table(items.items_path(built))
    assert set(table.column_names) == set(items.PUBLIC_COLUMNS)
    assert not set(table.column_names) & items.PROTECTED_KEYS


def test_cot_content_and_solutions_never_reach_the_public_table(built):
    blob = items.items_path(built).read_bytes()
    assert COT_MARKER.encode() not in blob
    assert SOLUTION_MARKER.encode() not in blob
    for name in (items.MATH_GOLD_FILE, items.MMLU_GOLD_FILE):
        gold = (items.protected_dir(built) / name).read_bytes()
        assert COT_MARKER.encode() not in gold


def test_gold_lives_only_in_the_protected_namespace(built):
    tasks = items.load_ard_tasks(built)
    for task in tasks.values():
        assert task.gold_location.startswith(f"{items.PROTECTED_DIR}/")
        assert task.item_id in task.gold_location
        row = task.to_row()
        assert not set(row) & items.PROTECTED_KEYS


def test_protected_files_are_0600_inside_a_0700_directory(built):
    directory = items.protected_dir(built)
    assert stat.S_IMODE(directory.stat().st_mode) == items.PROTECTED_DIR_MODE
    for name in (items.MATH_GOLD_FILE, items.MMLU_GOLD_FILE):
        assert stat.S_IMODE((directory / name).stat().st_mode) == items.PROTECTED_FILE_MODE


def test_gold_store_refuses_a_widened_protected_file(tmp_path, sources):
    root = tmp_path / "widened"
    items_build.build_items(root, sources=sources)
    path = items.protected_dir(root) / items.MATH_GOLD_FILE
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError):
        graders.GoldStore.load(root)


def test_loader_rejects_a_row_carrying_a_protected_key(built):
    row = next(iter(items.load_ard_tasks(built).values())).to_row()
    row["answer"] = "42"
    with pytest.raises(items.ItemsError):
        items.assert_public_row(row)


def test_runtime_reader_import_closure_excludes_the_dataset_stack():
    """``ard.items`` must not reach ``benchmarks.loaders`` or ``datasets`` (contract §0.2)."""
    import ast
    import sys

    seen: set[str] = set()
    forbidden = {"datasets", "agents_scaling.benchmarks.loaders", "agents_scaling.ard.items_build"}
    frontier = ["agents_scaling.ard.items"]
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        module = sys.modules.get(name) or __import__(name, fromlist=["__file__"])
        path = getattr(module, "__file__", None)
        if not path or not path.endswith(".py"):
            continue
        tree = ast.parse(Path(path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                targets = [node.module]
            else:
                continue
            for target in targets:
                assert target not in forbidden, f"{name} imports {target}"
                if target.startswith("agents_scaling"):
                    frontier.append(target)


# --------------------------------------------------------------------------- permutations


def test_option_permutation_round_trip(built):
    store = graders.GoldStore.load(built)
    tasks = items.load_ard_tasks(built, family=Family.MMLUPRO)
    sources = {row["question_id"]: row for row in _mmlu_rows()}
    assert tasks
    for task in tasks.values():
        permutation = task.permutation
        assert sorted(permutation) == list(range(MMLU_REQUIRED_OPTIONS))
        gold = store.reference(Family.MMLUPRO, task.item_id)
        assert gold["option_permutation"] == permutation
        source_options = sources[int(task.source_id.split("-")[1])]["options"]
        assert task.options == [source_options[index] for index in permutation]
        position = "ABCDEFGHIJ".index(gold["permuted_key"])
        assert permutation[position] == gold["answer_index"]
        assert task.options[position] == source_options[gold["answer_index"]]
        assert gold["permuted_key"] not in task.task_text.split("\n")[0]


def test_option_permutation_is_frozen_per_item_and_variant(built):
    tasks = items.load_ard_tasks(built, family=Family.MMLUPRO)
    seed = items_build.DEFAULT_STUDY_SEED
    differing = 0
    for task in tasks.values():
        assert items_build.option_permutation(task.item_id, seed) == tuple(task.permutation)
        second = items_build.option_permutation(task.item_id, seed, variant=1)
        differing += second != tuple(task.permutation)
    assert differing >= len(tasks) - 1     # spec §14.4 control permutation is a new draw
    with pytest.raises(ValueError):
        items_build.option_permutation("x", seed, variant=7)


def test_permutations_are_not_all_identical(built):
    tasks = items.load_ard_tasks(built, family=Family.MMLUPRO)
    assert len({task.option_permutation for task in tasks.values()}) > 1


# --------------------------------------------------------------------------- reader


def test_load_ard_tasks_filters(built):
    everything = items.load_ard_tasks(built)
    train = items.load_ard_tasks(built, split="train")
    assert len(train) == 2 * SPLIT_COUNTS_PER_FAMILY["train"]
    assert set(train) <= set(everything)
    math_only = items.load_ard_tasks(built, family=Family.MATH500)
    assert {task.family for task in math_only.values()} == {Family.MATH500.value}
    with pytest.raises(ValueError):
        items.load_ard_tasks(built, split="main")
    with pytest.raises(ValueError):
        items.load_ard_tasks(built, family="hle")


def test_build_refuses_to_overwrite_without_force(built, sources):
    with pytest.raises(items_build.BuildError):
        items_build.build_items(built, sources=sources)


def test_items_carry_the_grader_hash_and_the_pinned_revision(built):
    tasks = items.load_ard_tasks(built)
    hashes = graders.grader_hashes()
    for task in tasks.values():
        assert task.grader_hash == hashes[task.family]
        assert len(task.dataset_revision) == 40
        assert task.prompt_hash == items_build.prompt_hash(
            Family(task.family), task.task_text, task.options)


def test_allocation_is_proportional_and_min_one():
    sizes = {"a": 100, "b": 50, "c": 1, "d": 1}
    allocation = items_build.proportional_allocation(sizes, 20, b"salt")
    assert sum(allocation.values()) == 20
    assert min(allocation.values()) >= 1
    assert allocation["a"] > allocation["b"] > allocation["c"]
    assert items_build.proportional_allocation(sizes, 20, b"salt") == allocation
    with pytest.raises(items_build.BuildError):
        items_build.proportional_allocation(sizes, 3, b"salt")
    with pytest.raises(items_build.BuildError):
        items_build.proportional_allocation(sizes, 1000, b"salt")
