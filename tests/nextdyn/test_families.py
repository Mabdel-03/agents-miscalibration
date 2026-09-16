"""The family registry is one row per panel, and runtime-safe (contract §2.3; plan §2.1, §1.10)."""

from __future__ import annotations

import pytest

from agents_scaling.nextdyn import families as F
from agents_scaling.nextdyn.errors import ConfigError
from agents_scaling.nextdyn.graders import mc


def test_every_declared_family_carries_a_complete_row():
    for name, spec in F.FAMILIES.items():
        assert spec.name == name
        assert spec.answer_format in F.ANSWER_FORMATS
        assert spec.answer_instruction.strip()
        assert spec.grade_resources.container in F.CONTAINERS
        assert spec.loader.reader in F.READERS
        for dotted in (spec.canonicalizer, spec.grader):
            module, _, attr = dotted.partition(":")
            assert module.startswith("agents_scaling.nextdyn.graders") and attr


def test_the_six_families_of_the_plan_are_declared_and_mapped_to_cohorts():
    assert set(F.FAMILIES) == {F.MATH500, F.MATH_EXPANDED, F.MMLU_PRO, F.TRUTHFULQA, F.MOD97,
                               F.BIGCODEBENCH}
    assert F.panels_of("MATH") == (F.MATH500, F.MATH_EXPANDED)
    assert F.E0_ONLY == {F.BIGCODEBENCH}
    assert F.MOD97 not in F.DATASET_FAMILIES


def test_every_cohort_family_of_the_committed_study_has_at_least_one_panel(study_cfg):
    for cohort in study_cfg.cohorts.families:
        assert F.panels_of(cohort)


def test_the_two_math_panels_are_pinned_to_different_sources():
    math500, expanded = F.family(F.MATH500).loader, F.family(F.MATH_EXPANDED).loader
    assert math500.revision != expanded.revision
    assert expanded.parts == F.HENDRYCKS_SUBJECTS and len(expanded.parts) == 7
    assert expanded.reader == F.READER_HUB_PARQUET   # the script loader is dead


def test_a_renderer_prints_the_parametric_alphabet_and_refuses_the_wrong_shape():
    render = F.family(F.TRUTHFULQA).renderer
    text = render("Which is true?", ["a", "b", "c", "d"])
    assert text.splitlines()[2:] == ["A. a", "B. b", "C. c", "D. d"]
    thirteen = render("q", [f"opt{i}" for i in range(13)])
    assert thirteen.splitlines()[-1].startswith(f"{mc.mc_letters(13)[-1]}.")
    with pytest.raises(ConfigError):
        render("q", [f"opt{i}" for i in range(14)])
    with pytest.raises(ConfigError):
        F.family(F.MATH500).renderer("q", ["a"])


def test_grade_resources_name_a_container_and_only_bigcodebench_needs_the_image():
    for name, spec in F.FAMILIES.items():
        resources = spec.grade_resources
        wants_image = resources.container == F.CONTAINER_APPTAINER
        assert wants_image == (name == F.BIGCODEBENCH)
        assert bool(resources.sif) == wants_image
        assert set(resources.as_mapping()) == {"cpus", "mem", "time", "items_per_cell",
                                               "container", "sif"}


def test_the_family_registry_never_imports_a_loader_or_a_gold_store():
    """R7: ``families`` is read by the generation lane, so its closure holds no gold."""
    import ast
    import sys

    tree = ast.parse(open(sys.modules[F.__name__].__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    forbidden = {"datasets", "agents_scaling.benchmarks", "agents_scaling.nextdyn.datasets",
                 "agents_scaling.nextdyn.graders.families", "agents_scaling.ard.graders"}
    assert not (imported & forbidden), sorted(imported & forbidden)
    assert not any(name.startswith("agents_scaling.benchmarks") for name in imported)


def test_resolve_imports_a_grader_lazily():
    assert F.resolve(F.family(F.MMLU_PRO).grader) is mc.grade
    with pytest.raises(ConfigError):
        F.resolve("agents_scaling.nextdyn.graders.mc:not_a_function")
