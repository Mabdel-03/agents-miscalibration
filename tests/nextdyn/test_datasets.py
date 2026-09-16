"""The pinned loaders and ``datasets.lock.json`` against the real offline cache (plan §2.1, D19).

Every test here reads the shared HF cache and is marked ``needs_hf_cache``; the whole module
skips when the pinned snapshots are absent, so the suite still runs on a machine that has
never pulled a dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agents_scaling.config import DEFAULT_HF_HOME
from agents_scaling.nextdyn import config as C
from agents_scaling.nextdyn import datasets as D
from agents_scaling.nextdyn import families as F

HF_HOME = Path(os.environ.get("HF_HOME") or DEFAULT_HF_HOME)
CACHE_READY = ((HF_HOME / "hub" / "datasets--EleutherAI--hendrycks_math").is_dir()
               and (HF_HOME / "datasets").is_dir())

pytestmark = [
    pytest.mark.needs_hf_cache,
    pytest.mark.skipif(not CACHE_READY, reason=f"the pinned datasets are not cached under {HF_HOME}"),
]


@pytest.fixture(scope="module")
def panels() -> dict[str, D.LoadedPanel]:
    math500 = D.load_panel(F.MATH500, HF_HOME)
    exclude = frozenset(row.normalized_text_hash for row in math500.rows)
    return {F.MATH500: math500,
            F.MATH_EXPANDED: D.load_panel(F.MATH_EXPANDED, HF_HOME, exclude_text_hashes=exclude),
            F.MMLU_PRO: D.load_panel(F.MMLU_PRO, HF_HOME),
            F.TRUTHFULQA: D.load_panel(F.TRUTHFULQA, HF_HOME)}


# --------------------------------------------------------------------------- the overlap key


def test_normalized_text_hash_ignores_case_whitespace_and_option_order():
    base = D.normalized_text_hash("What is 2 + 2?", ["four", "five"])
    assert base == D.normalized_text_hash("  what is 2   +  2? ", ["FIVE", "four"])
    assert base != D.normalized_text_hash("What is 2 + 3?", ["four", "five"])
    assert base != D.normalized_text_hash("What is 2 + 2?", ["four"])


# --------------------------------------------------------------------------- the panels


def test_math500_loads_the_pinned_five_hundred_rows(panels):
    panel = panels[F.MATH500]
    assert len(panel.rows) == 500 and panel.n_rows_raw == 500
    assert all(row.gold["answer"] for row in panel.rows)
    assert panel.file_sha256 and Path(panel.file_path).exists()


def test_math_expanded_reads_seven_subjects_and_removes_every_math500_row(panels):
    """Plan D19: the broader MATH panel is disjoint from MATH-500 by normalized text hash."""
    expanded, math500 = panels[F.MATH_EXPANDED], panels[F.MATH500]
    assert expanded.n_rows_raw == 5000
    assert expanded.n_overlap_removed == 500
    assert len(expanded.rows) == 4500
    assert set(expanded.filters["subjects"]) == set(F.HENDRYCKS_SUBJECTS)
    seen = {row.normalized_text_hash for row in math500.rows}
    assert not seen & {row.normalized_text_hash for row in expanded.rows}


def test_math_expanded_gold_is_the_boxed_value_of_the_solution(panels):
    from agents_scaling.ard.graders._boxed import extract_boxed

    for row in panels[F.MATH_EXPANDED].rows[:50]:
        assert row.gold["answer"] == extract_boxed(row.gold["solution"])


def test_mmlu_pro_eligibility_reproduces_the_ard36_count(panels):
    """Parity with the vendored ARD36 rule: ten options, duplicate questions dropped."""
    panel = panels[F.MMLU_PRO]
    assert len(panel.rows) == 9742
    assert panel.option_count_histogram == {"10": 9742}
    assert panel.filters["duplicate_question"] == 239
    assert all(len(row.options) == F.MMLU_REQUIRED_OPTIONS for row in panel.rows)


def test_truthfulqa_keeps_the_legacy_mc1_window_and_one_correct_label(panels):
    panel = panels[F.TRUTHFULQA]
    counts = {int(k): v for k, v in panel.option_count_histogram.items()}
    assert min(counts) >= F.TRUTHFULQA_MIN_OPTIONS and max(counts) <= F.TRUTHFULQA_MAX_OPTIONS
    assert sum(counts.values()) == len(panel.rows)
    for row in panel.rows:
        assert row.gold["labels"].count(1) == 1
        assert row.gold["canonical_index"] == row.gold["labels"].index(1)


def test_every_panel_row_has_a_non_empty_original_id_and_question(panels):
    for panel in panels.values():
        for row in panel.rows:
            assert row.original_id and row.question.strip()


# --------------------------------------------------------------------------- the lock


@pytest.fixture(scope="module")
def lock():
    return D.lock_datasets(C.load_study(), HF_HOME)


def test_the_lock_pins_every_family_with_its_source_and_filters(lock):
    assert set(lock.datasets) == set(F.FAMILIES)
    for name, entry in lock.datasets.items():
        spec = F.family(name).loader
        assert (entry.repo_id, entry.revision, entry.split) == (spec.repo_id, spec.revision, spec.split)
        assert entry.row_content_sha256 and entry.filters is not None
    assert lock.datasets[F.MATH_EXPANDED].n_overlap_removed == 500
    assert lock.datasets[F.MMLU_PRO].option_count_histogram == {"10": 9742}
    assert lock.datasets[F.MOD97].revision is None      # generated, not loaded
    assert lock.locked_at and lock.hf_home == str(HF_HOME)


def test_the_lock_records_the_legacy_roots_e0_reads(lock):
    assert set(lock.legacy_roots) == {"ard36", "study_v4", "full_sweep_v1"}
    assert any(entry.manifest_sha256 for entry in lock.legacy_roots.values())


def test_verify_lock_passes_on_the_cache_it_was_built_from_and_fails_on_drift(lock):
    import dataclasses

    assert D.verify_lock(lock, HF_HOME) == []
    drifted = dataclasses.replace(
        lock, datasets=dict(lock.datasets,
                            MATH500=dataclasses.replace(lock.datasets[F.MATH500],
                                                        file_sha256="0" * 64)))
    assert any("sha256 drifted" in message for message in D.verify_lock(drifted, HF_HOME))
    skeleton = dataclasses.replace(lock, locked_at=None)
    assert D.verify_lock(skeleton, HF_HOME) == [
        "datasets.lock.json is a skeleton (locked_at null); run lock-inputs"]


def test_the_committed_lock_file_matches_the_families_and_the_cache():
    """The committed ``configs/nextdyn/datasets.lock.json`` is the pin, not a skeleton."""
    from agents_scaling.nextdyn.locks import load_datasets_lock

    committed = load_datasets_lock(C.CONFIG_DIR / "datasets.lock.json")
    assert set(committed.datasets) == set(F.FAMILIES)
    assert committed.locked_at is not None
    assert D.verify_lock(committed, HF_HOME) == []


def test_the_legacy_registry_names_the_prior_roots_without_sidecars():
    registry = D.load_legacy_sources()
    assert len(registry.prior_roots) == 12
    assert registry.roots["ard36"].startswith("/orcd/data/")
    assert registry.resolve("study_v4:data/raw/x.parquet").name == "x.parquet"
    assert registry.resolve(None) is None
