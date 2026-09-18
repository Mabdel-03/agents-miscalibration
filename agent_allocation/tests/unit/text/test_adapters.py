"""Adapters: allowlisted public records, private gold, pinned sources, frozen exclusions, the seeded GPQA shuffle."""
from __future__ import annotations

import collections
import json
import random

import pytest

from agent_allocation.benchmarks import _matharena, aime, gpqa_diamond, hmmt, omni_math_hard
from agent_allocation.benchmarks.text_manifest import (
    TEXT_BENCHMARKS,
    SourceIntegrityError,
    exclusion_records,
    hf_snapshot_file,
)
from agent_allocation.contracts import BENCHMARKS, EvalKind, OutputKind, PublicTask, dump_record, load_record

from .conftest import CANARY, needs_aime, needs_gpqa, needs_hmmt, needs_omni

PUBLIC_FIELDS = {"__record__", "item_id", "benchmark", "source_revision", "prompt", "output_kind",
                 "public_examples", "allowed_choices", "subtask"}
MODULES = (aime, hmmt, omni_math_hard, gpqa_diamond)


def _public_manifest_text(items) -> str:
    return "\n".join(json.dumps(dump_record(item.public), sort_keys=True) for item in items)


def test_text_benchmarks_are_the_contract_benchmarks():
    assert TEXT_BENCHMARKS == BENCHMARKS == tuple(module.BENCHMARK for module in MODULES)


# ---------------------------------------------------------------------------------------------------------
# Synthetic rows: leakage canaries and record shape
# ---------------------------------------------------------------------------------------------------------

def _hmmt_row(**over):
    row = {"year": 2025, "problem_idx": 4, "problem": "  Compute $1+1$.\n", "answer": CANARY,
           "problem_type": ["Algebra", "Geometry"]}
    row.update(over)
    return row


def _omni_row(**over):
    row = {"row_index": 12, "problem": "Find $x$.\n", "solution": f"Work. {CANARY}-SOLUTION", "answer": "\\frac{1}{2}",
           "difficulty": 7.5, "domain": ["Mathematics -> Number Theory -> Other"], "source": f"src-{CANARY}"}
    row.update(over)
    return row


def _gpqa_row(**over):
    row = {"Record ID": "recABC123", "Question": " Which is right?\n", "Correct Answer": f"right {CANARY}",
           "Incorrect Answer 1": "wrong one", "Incorrect Answer 2": "wrong two\n", "Incorrect Answer 3": "wrong three",
           "Subdomain": f"sub-{CANARY}", "High-level domain": "Physics", "Explanation": f"{CANARY}-EXPLANATION"}
    row.update(over)
    return row


def test_matharena_public_record_carries_problem_only():
    item = hmmt.build_item(_hmmt_row())
    assert item.public.item_id == "hmmt/2025/4"
    assert item.public.prompt == "Compute $1+1$."
    assert item.public.output_kind is OutputKind.MATH and item.public.allowed_choices == ()
    assert item.public.source_revision == hmmt.SOURCES[0].revision
    assert item.private.gold_answer == CANARY and item.private.eval_kind is EvalKind.MATH_VERIFY
    assert item.strata == (("year", "2025"), ("problem_type", "Algebra"))
    assert CANARY not in _public_manifest_text([item])
    assert set(dump_record(item.public)) == PUBLIC_FIELDS


def test_aime_integer_gold_is_a_string_and_strata_is_year_only():
    item = aime.build_item({"year": 2026, "problem_idx": 16, "problem": "P?", "answer": 7})
    assert item.private.gold_answer == "7" and item.strata == (("year", "2026"),)
    assert item.public.source_revision == aime.SOURCES[1].revision
    assert not aime.build_item({"year": 2026, "problem_idx": 1, "problem": "P?", "answer": ""}).eligible
    with pytest.raises(SourceIntegrityError):
        aime.build_item({"year": 2024, "problem_idx": 1, "problem": "P?", "answer": 1})
    with pytest.raises(SourceIntegrityError):
        aime.build_item({"year": 2025, "problem_idx": "1", "problem": "P?", "answer": 1})


def test_omni_public_record_carries_problem_only_and_domain_stratum():
    item = omni_math_hard.build_item(_omni_row())
    assert item.public.item_id == "omni_math_hard/12" and item.public.prompt == "Find $x$."
    assert item.eligible and item.strata == (("domain", "Number Theory"),)
    assert item.private.gold_answer == "\\frac{1}{2}"
    assert CANARY not in _public_manifest_text([item])
    assert set(dump_record(item.public)) == PUBLIC_FIELDS


@pytest.mark.parametrize("over, reason", [
    ({"answer": " "}, "empty_gold_answer"),
    ({"problem": "\n"}, "empty_problem"),
    ({"problem": "The figure below shows a grid. Count it."}, "figure_not_included"),
    ({"answer": "\\text{Yes}"}, "gold_prose"),
    ({"answer": "n \\text{ is odd}"}, "gold_prose"),
    ({"answer": "There is no such function."}, "gold_prose"),
    ({"answer": "x = 1 or x = 2"}, "gold_prose"),
])
def test_omni_rule_exclusions_have_reasons(over, reason):
    item = omni_math_hard.build_item(_omni_row(**over))
    assert not item.eligible and item.exclusion_reason == reason


@pytest.mark.parametrize("gold", ["2n - 1", "\\max(a, b)", "\\sin^2 80^\\circ", "\\lceil \\log_2 n \\rceil",
                                  "f(x) = x^2", "\\binom{n}{2} + 1", "(2, 3, 3)", "\\mathbb{Z}", "\\text{ }3"])
def test_omni_math_golds_are_not_prose(gold):
    assert not omni_math_hard.is_prose(gold)


def test_omni_asy_figure_is_included():
    assert omni_math_hard.rule_reason(_omni_row(problem="As shown below.\n[asy] draw((0,0)--(1,1)); [/asy]")) is None


def test_omni_audit_catches_partial_and_unparseable_golds():
    assert omni_math_hard.audit_gold("\\frac{1}{2}") is None
    assert omni_math_hard.audit_gold("f(x) = x + 1") is None
    assert omni_math_hard.audit_gold("\\dfrac {n^2} 4 + \\dfrac n 2") == "gold_partial_parse"
    assert omni_math_hard.audit_gold("3\\mid n") == "gold_unparseable"


def test_gpqa_shuffle_is_the_reference_recipe_and_hides_the_role():
    row = _gpqa_row()
    item = gpqa_diamond.build_item(row)
    order = list(range(4))
    random.Random(f"20260917-{row['Record ID']}").shuffle(order)
    assert gpqa_diamond.option_order(row["Record ID"]) == order
    options = [row[name] for name in gpqa_diamond.OPTION_FIELDS]
    expected = "Which is right?\n\nOptions:\n" + "\n".join(
        f"{'ABCD'[k]}. {options[position].strip()}" for k, position in enumerate(order))
    assert item.public.prompt == expected
    assert item.private.gold_answer == "ABCD"[order.index(0)]
    assert item.public.allowed_choices == item.private.allowed_choices == ("A", "B", "C", "D")
    assert item.private.eval_kind is EvalKind.OPTION_EXACT and item.strata == (("domain", "Physics"),)
    text = _public_manifest_text([item])
    assert "EXPLANATION" not in text and "sub-" not in text and "Correct" not in text
    assert load_record(PublicTask, dump_record(item.public)) == item.public


def test_gpqa_duplicate_options_are_excluded():
    item = gpqa_diamond.build_item(_gpqa_row(**{"Incorrect Answer 3": " wrong one"}))
    assert not item.eligible and item.exclusion_reason.startswith("duplicate_option_text:")
    with pytest.raises(SourceIntegrityError):
        gpqa_diamond.build_item(_gpqa_row(**{"Record ID": "../x"}))


# ---------------------------------------------------------------------------------------------------------
# Real pinned data
# ---------------------------------------------------------------------------------------------------------

@needs_aime
def test_aime_inventory(aime_inventory):
    assert len(aime_inventory) == aime.SOURCE_ROWS == 60 and all(item.eligible for item in aime_inventory)
    rows = {f"aime/{row['year']}/{row['problem_idx']}": row for row in aime.read_rows()}
    for item in aime_inventory:
        row = rows[item.public.item_id]
        assert item.public.prompt == row["problem"].strip(" \t\n\r\f\v")
        assert item.private.gold_answer == str(row["answer"]) and 0 <= int(item.private.gold_answer) <= 999
    assert collections.Counter(item.strata for item in aime_inventory) == {
        (("year", "2025"),): 30, (("year", "2026"),): 30}


@needs_hmmt
def test_hmmt_inventory(hmmt_inventory):
    assert len(hmmt_inventory) == hmmt.SOURCE_ROWS == 63 and all(item.eligible for item in hmmt_inventory)
    rows = {f"hmmt/{row['year']}/{row['problem_idx']}": row for row in hmmt.read_rows()}
    for item in hmmt_inventory:
        assert item.private.gold_answer == rows[item.public.item_id]["answer"]
    non_integer = [item for item in hmmt_inventory if not item.private.gold_answer.lstrip("-").isdigit()]
    assert len(non_integer) == 33
    assert {item.public.source_revision for item in hmmt_inventory} == {s.revision for s in hmmt.SOURCES}


@needs_omni
def test_omni_inventory_and_exclusion_counts(omni_inventory):
    assert len(omni_inventory) == omni_math_hard.POPULATION_ROWS == 916
    reasons = collections.Counter(record["reason"] for record in exclusion_records(omni_inventory))
    assert reasons == {"gold_prose": 367, "gold_partial_parse": 23, "gold_unparseable": 14,
                       "empty_gold_answer": 9, "figure_not_included": 1, "gold_self_verify_failed": 1}
    assert sum(item.eligible for item in omni_inventory) == 501
    rows = {f"omni_math_hard/{row['row_index']}": row for row in omni_math_hard.read_rows()}
    for item in omni_inventory:
        row = rows[item.public.item_id]
        assert row["difficulty"] >= 7.0 and item.private.gold_answer == row["answer"]
    text = _public_manifest_text(omni_inventory)
    assert all(row["solution"].strip() not in text for row in rows.values() if len(row["solution"]) > 40)


@needs_omni
def test_frozen_omni_exclusions_match_the_recomputed_audit(omni_inventory):
    recomputed = {}
    for row in omni_math_hard.population(omni_math_hard.read_rows()):
        if omni_math_hard.rule_reason(row) is None:
            reason = omni_math_hard.audit_gold(row["answer"])
            if reason is not None:
                recomputed[f"omni_math_hard/{row['row_index']}"] = reason
    assert recomputed == dict(omni_math_hard.ENGINEERING_EXCLUSIONS)


@needs_gpqa
def test_gpqa_inventory(gpqa_inventory):
    assert len(gpqa_inventory) == 198
    records = exclusion_records(gpqa_inventory)
    assert {record["item_id"] for record in records} == {"gpqa_diamond/recZSGUkn56v9kEp1",
                                                         "gpqa_diamond/recnGEpF1srQpaqWq"}
    rows = {row["Record ID"]: row for row in gpqa_diamond.read_rows()}
    for item in gpqa_inventory:
        row = rows[item.public.item_id.split("/", 1)[1]]
        gold_text = row["Correct Answer"].strip(" \t\n\r\f\v")
        assert f"\n{item.private.gold_answer}. {gold_text}" in item.public.prompt
        assert item.public.prompt == item.public.prompt.strip(" \t\n\r\f\v")
        assert set(dump_record(item.public)) == PUBLIC_FIELDS
    letters = collections.Counter(item.private.gold_answer for item in gpqa_inventory)
    assert set(letters) == set("ABCD") and max(letters.values()) < 80


def test_config_source_mismatch_is_refused(cfg, run_paths):
    for module in MODULES:
        bad = json.loads(json.dumps(cfg))
        for entry in bad["benchmarks"]:
            if entry["id"] == module.BENCHMARK:
                entry["source"] = "someone/else"
        with pytest.raises(ValueError):
            module.load_inventory(bad, run_paths)


def test_omni_min_difficulty_must_match(cfg, run_paths):
    bad = json.loads(json.dumps(cfg))
    for entry in bad["benchmarks"]:
        if entry["id"] == "omni_math_hard":
            entry["min_difficulty"] = 6.5
    with pytest.raises(ValueError):
        omni_math_hard.load_inventory(bad, run_paths)


@needs_omni
def test_private_hash_is_stable_and_binds_the_gold(omni_inventory):
    rows = omni_math_hard.population(omni_math_hard.read_rows())
    again = [omni_math_hard.build_item(row) for row in rows]
    assert [item.private.private_hash for item in again] == [item.private.private_hash for item in omni_inventory]
    changed = omni_math_hard.build_item({**rows[0], "answer": rows[0]["answer"] + "1"})
    assert changed.private.private_hash != again[0].private.private_hash
    assert changed.public == again[0].public


@pytest.mark.parametrize("module", [omni_math_hard, gpqa_diamond])
def test_pinned_sha256_is_enforced(module, monkeypatch):
    path = hf_snapshot_file(module.SOURCE_REPO, module.SOURCE_REVISION, module.SOURCE_FILE)
    if not path.is_file():
        pytest.skip("pinned snapshot not in HF cache")
    monkeypatch.setattr(module, "SOURCE_SHA256", "0" * 64)
    with pytest.raises(SourceIntegrityError):
        module.read_rows()


@pytest.mark.parametrize("module", [aime, hmmt])
def test_pinned_matharena_sha256_is_enforced(module):
    source = module.SOURCES[0]
    if not hf_snapshot_file(source.repo, source.revision, source.file).is_file():
        pytest.skip("pinned snapshot not in HF cache")
    tampered = _matharena.YearSource(source.year, source.repo, source.revision, "0" * 64, source.rows)
    spec = _matharena.MathArenaSpec(**{**module.SPEC.__dict__, "sources": (tampered,)})
    with pytest.raises(SourceIntegrityError):
        _matharena.read_rows(spec)


def test_changed_snapshot_is_refused(cfg, run_paths, tmp_path, monkeypatch):
    source = aime.SOURCES[0]
    if not hf_snapshot_file(source.repo, source.revision, source.file).is_file():
        pytest.skip("pinned snapshot not in HF cache")
    fake_home = tmp_path / "hf"
    for s in aime.SOURCES:
        target = fake_home / "hub" / f"datasets--{s.repo.replace('/', '--')}" / "snapshots" / s.revision / s.file
        target.parent.mkdir(parents=True)
        data = hf_snapshot_file(s.repo, s.revision, s.file).read_bytes()
        target.write_bytes(data[:-1] + bytes([data[-1] ^ 1]) if s is source else data)
    monkeypatch.setenv("HF_HOME", str(fake_home))
    with pytest.raises(SourceIntegrityError):
        aime.load_inventory(cfg, run_paths)


def test_source_records_name_revision_hash_and_licence():
    for module in MODULES:
        record = module.source_record()
        assert record["license"] and record["preprocessing_version"] == module.PREPROCESSING_VERSION
        assert json.loads(json.dumps(module.SOURCE_PROVENANCE)) == record
    assert [f["revision"] for f in aime.source_record()["files"]] == [
        "c94da77eb22bbd6439e62a323bec18493a421302", "d2de22f3c656b4f56cf8981212186377d1e23bc3"]
    assert [f["revision"] for f in hmmt.source_record()["files"]] == [
        "6fdc4277120810ff75aa22d2d5489b91f7a262a1", "02fba4f74d8e68e73e66a02d540fd979c05c274c"]
    assert omni_math_hard.source_record()["revision"] == "40ba231d8f16e29ecd40e6407e2c8640145a8f62"
    assert gpqa_diamond.source_record()["sha256"] == (
        "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305")


@pytest.mark.parametrize("benchmark", TEXT_BENCHMARKS)
def test_fixture_inventories_are_typed_drawable_and_leak_free(benchmark):
    from agent_allocation.benchmarks.text_manifest import FIXTURE_REVISION, fixture_inventory, select_panels
    items = fixture_inventory(benchmark)
    assert len(items) == 24 and all(item.eligible for item in items)
    assert {item.public.source_revision for item in items} == {FIXTURE_REVISION}
    for item in items:
        assert item.private.gold_answer is not None
        assert item.private.gold_answer not in item.public.prompt.split("\nOptions:\n")[0]
    draw = select_panels(items, benchmark=benchmark, seed=7, n_dev=4, n_main=8, n_reserve=4)
    assert len(draw.main_item_ids) == 8


def test_every_ineligible_item_carries_a_reason(aime_inventory, hmmt_inventory, omni_inventory, gpqa_inventory):
    for inventory in (aime_inventory, hmmt_inventory, omni_inventory, gpqa_inventory):
        records = exclusion_records(inventory)             # raises if any ineligible item lacks a reason
        assert len(records) == sum(not item.eligible for item in inventory)
