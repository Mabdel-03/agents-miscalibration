"""Adapters: allowlisted public records, private gold, pinned sources, four BBH formats, legal MMLU choices."""
from __future__ import annotations

import collections
import json
import os
import shutil
import string

import pytest

from agent_allocation.benchmarks import bbh_logic4, math500, mmlu_pro
from agent_allocation.benchmarks.text_manifest import SourceIntegrityError, exclusion_records
from agent_allocation.contracts import (
    BBH_SUBTASKS,
    EvalKind,
    OutputKind,
    PublicTask,
    dump_record,
    load_record,
)

from .conftest import CANARY, needs_bbh, needs_math500, needs_mmlu_pro

PUBLIC_FIELDS = {"__record__", "item_id", "benchmark", "source_revision", "prompt", "output_kind",
                 "public_examples", "allowed_choices", "subtask"}


def _public_manifest_text(items) -> str:
    return "\n".join(json.dumps(dump_record(item.public), sort_keys=True) for item in items)


# ---------------------------------------------------------------------------------------------------------
# Synthetic rows: leakage canaries and record shape
# ---------------------------------------------------------------------------------------------------------

def _math_row(**over):
    row = {"problem": "Compute $1+1$.", "solution": f"Add. {CANARY}-SOLUTION", "answer": f"{CANARY}",
           "subject": "Algebra", "level": 1, "unique_id": "test/algebra/1.json"}
    row.update(over)
    return row


def _mmlu_row(**over):
    row = {"question_id": 7, "question": "  Which is prime?\n", "options": [" 4", "9 ", "\n7\n", "8"],
           "answer": "C", "answer_index": 2, "cot_content": f"{CANARY}-RATIONALE", "category": "math",
           "src": f"ori_mmlu-{CANARY}"}
    row.update(over)
    return row


def test_math_public_record_carries_problem_only():
    item = math500.build_item(_math_row())
    assert item.public.item_id == "math500/test/algebra/1.json"
    assert item.public.prompt == "Compute $1+1$."
    assert item.public.output_kind is OutputKind.MATH and item.public.allowed_choices == ()
    assert item.private.gold_answer == CANARY and item.private.eval_kind is EvalKind.MATH_VERIFY
    assert item.strata == (("level", "1"), ("subject", "Algebra"))
    text = _public_manifest_text([item])
    assert CANARY not in text
    assert set(dump_record(item.public)) == PUBLIC_FIELDS


def test_mmlu_prompt_trims_outer_ascii_whitespace_and_keeps_order():
    item = mmlu_pro.build_item(_mmlu_row())
    assert item.public.prompt == "Which is prime?\n\nOptions:\nA. 4\nB. 9\nC. 7\nD. 8"
    assert item.public.allowed_choices == ("A", "B", "C", "D")
    assert item.private.gold_answer == "C" and item.private.eval_kind is EvalKind.OPTION_EXACT
    assert item.eligible and item.strata == (("category", "math"),)
    assert CANARY not in _public_manifest_text([item])
    # Unicode NEL (\x85) is content in the pinned source, not outer whitespace.
    nel = mmlu_pro.build_item(_mmlu_row(question="Role theory suggests that\x85"))
    assert nel.public.prompt.startswith("Role theory suggests that\x85\n\nOptions:\n")


@pytest.mark.parametrize("over, reason", [
    ({"options": ["x", "y", "x ", "z"]}, "duplicate_option_text:A,C"),
    ({"answer": "B"}, "answer_letter_disagrees_with_answer_index"),
    ({"options": ["only"], "answer": "A", "answer_index": 0}, "option_count_1"),
    ({"options": ["a", " \n", "c"]}, "empty_option"),
])
def test_mmlu_engineering_exclusions_have_reasons(over, reason):
    item = mmlu_pro.build_item(_mmlu_row(**over))
    assert not item.eligible and item.exclusion_reason == reason


def test_mmlu_duplicate_stems_keep_the_lowest_eligible_question_id():
    rows = [_mmlu_row(question_id=30, question="Same stem?"),
            _mmlu_row(question_id=12, question=" Same stem?\n", options=["x", "x", "y", "z"]),   # per-row exclusion
            _mmlu_row(question_id=20, question="Same stem?", options=["p", "q", "r", "s"]),
            _mmlu_row(question_id=5, question="Other stem?")]
    items = mmlu_pro.apply_source_item_rule([mmlu_pro.build_item(row) for row in rows],
                                            [row["question"] for row in rows])
    by_id = {item.public.item_id: item for item in items}
    assert by_id["mmlu_pro/20"].eligible and by_id["mmlu_pro/5"].eligible
    assert by_id["mmlu_pro/30"].exclusion_reason == "duplicate_question_stem:mmlu_pro/20"
    assert by_id["mmlu_pro/12"].exclusion_reason == "duplicate_option_text:A,B"


def test_public_record_round_trips_through_the_codec():
    item = mmlu_pro.build_item(_mmlu_row())
    assert load_record(PublicTask, dump_record(item.public)) == item.public


# ---------------------------------------------------------------------------------------------------------
# Real pinned data
# ---------------------------------------------------------------------------------------------------------

@needs_math500
def test_math500_inventory(math_inventory):
    assert len(math_inventory) == 500 and all(item.eligible for item in math_inventory)
    assert {item.public.source_revision for item in math_inventory} == {math500.SOURCE_REVISION}
    rows = {f"math500/{row['unique_id']}": row for row in math500.read_rows()}
    for item in math_inventory:
        row = rows[item.public.item_id]
        assert item.public.prompt == row["problem"] and item.private.gold_answer == row["answer"]
    text = _public_manifest_text(math_inventory)
    assert all(row["solution"] not in text for row in rows.values())


@needs_mmlu_pro
def test_mmlu_inventory_legal_choices_and_no_label_fields(mmlu_inventory):
    assert len(mmlu_inventory) == mmlu_pro.SOURCE_ROWS
    records = exclusion_records(mmlu_inventory)
    reasons = collections.Counter(record["reason"].split(":")[0] for record in records)
    assert reasons == {"duplicate_question_stem": 391, "duplicate_option_text": 1}
    assert {"benchmark": "mmlu_pro", "item_id": "mmlu_pro/8584", "source_revision": mmlu_pro.SOURCE_REVISION,
            "reason": "duplicate_option_text:A,D"} in records
    rows = {f"mmlu_pro/{row['question_id']}": row for row in mmlu_pro.read_rows()}
    by_id = {item.public.item_id: item for item in mmlu_inventory}
    stems = collections.Counter(mmlu_pro.trim_outer(rows[item.public.item_id]["question"])
                                for item in mmlu_inventory if item.eligible)
    assert max(stems.values()) == 1                                       # eligible items are source-item unique
    for record in records:
        if record["reason"].startswith("duplicate_question_stem:"):
            kept = record["reason"].split(":", 1)[1]
            assert by_id[kept].eligible
            assert int(kept.split("/")[1]) < int(record["item_id"].split("/")[1])
            assert (mmlu_pro.trim_outer(rows[kept]["question"])
                    == mmlu_pro.trim_outer(rows[record["item_id"]]["question"]))
    assert {item.public.source_revision for item in mmlu_inventory} == {mmlu_pro.SOURCE_REVISION}
    for item in mmlu_inventory:
        public = item.public
        row = rows[public.item_id]
        assert public.allowed_choices == tuple(string.ascii_uppercase[:len(row["options"])])
        assert public.prompt == mmlu_pro.render_prompt(row["question"], row["options"])
        body = public.prompt.split("\n\nOptions:\n", 1)[1]
        cursor = -1
        for letter, option in zip(public.allowed_choices, row["options"]):   # original order, every option once
            prefix = ("" if letter == "A" else "\n") + f"{letter}. " + mmlu_pro.trim_outer(option)
            found = body.find(prefix, cursor + 1)
            assert found > cursor
            cursor = found
        assert item.private.gold_answer == "ABCDEFGHIJ"[row["answer_index"]]
        assert item.private.gold_answer in public.allowed_choices
        assert public.prompt == public.prompt.strip(" \t\n\r\f\v")
    for key in ("answer_index", "cot_content", '"answer"', '"src"', '"category"'):
        assert key not in _public_manifest_text(mmlu_inventory[:200])


@needs_bbh
def test_bbh_four_formats_have_correct_labels(bbh_inventory):
    by_id = {item.public.item_id: item for item in bbh_inventory}
    assert len(bbh_inventory) == 1000 and all(item.eligible for item in bbh_inventory)
    expected = {
        "logical_deduction_seven_objects": (OutputKind.OPTION, tuple("ABCDEFG"), "D"),
        "tracking_shuffled_objects_seven_objects": (OutputKind.OPTION, tuple("ABCDEFG"), "A"),
        "boolean_expressions": (OutputKind.BOOLEAN, ("True", "False"), "False"),
        "temporal_sequences": (OutputKind.OPTION, tuple("ABCD"), "A"),
    }
    for subtask, (kind, allowed, gold) in expected.items():
        item = by_id[f"bbh_logic4/{subtask}/0"]
        raw = json.loads((bbh_logic4.shared_cache_dir() / f"{subtask}.json").read_text())["examples"][0]
        assert item.public.prompt == raw["input"]
        assert item.public.output_kind is kind and item.public.subtask == subtask
        assert item.public.allowed_choices == allowed and item.private.gold_answer == gold
        assert item.private.eval_kind is EvalKind.TASK_EXACT and item.private.subtask == subtask
        assert raw["target"] in (gold, f"({gold})")
    for item in bbh_inventory:
        assert item.private.gold_answer in item.public.allowed_choices
        assert item.public.source_revision == item.private.source_revision == bbh_logic4.SOURCE_COMMIT
    assert "target" not in _public_manifest_text(bbh_inventory)


@needs_bbh
def test_bbh_sources_are_copied_write_once_into_the_private_dir(cfg, run_paths):
    bbh_logic4.load_inventory(cfg, run_paths)
    copied = sorted(os.listdir(run_paths.bbh_sources_dir))
    assert copied == sorted([f"{s}.json" for s in BBH_SUBTASKS] + ["MANIFEST.json"])
    assert oct(os.stat(run_paths.private_dir).st_mode & 0o777) == "0o700"
    target = run_paths.bbh_sources_dir / "temporal_sequences.json"
    target.write_bytes(target.read_bytes().replace(b"(A)", b"(B)", 1))
    with pytest.raises(SourceIntegrityError):
        bbh_logic4.load_inventory(cfg, run_paths)


@needs_bbh
def test_changed_bbh_cache_file_is_refused(cfg, run_paths, tmp_path, monkeypatch):
    cache = tmp_path / "bbh_cache"
    shutil.copytree(bbh_logic4.shared_cache_dir(), cache)
    path = cache / "boolean_expressions.json"
    path.write_bytes(path.read_bytes().replace(b'"False"', b'"True"', 1))
    monkeypatch.setenv("ALLOCATION_BBH_CACHE", str(cache))
    with pytest.raises(SourceIntegrityError):
        bbh_logic4.load_inventory(cfg, run_paths)


@needs_math500
def test_changed_math500_snapshot_is_refused(cfg, run_paths, tmp_path, monkeypatch):
    fake_home = tmp_path / "hf"
    target = fake_home / "hub" / "datasets--HuggingFaceH4--MATH-500" / "snapshots" / math500.SOURCE_REVISION
    target.mkdir(parents=True)
    original = math500.hf_snapshot_file(math500.SOURCE_REPO, math500.SOURCE_REVISION, math500.SOURCE_FILE)
    (target / "test.jsonl").write_bytes(original.read_bytes().replace(b'"answer": "', b'"answer": "9', 1))
    monkeypatch.setenv("HF_HOME", str(fake_home))
    with pytest.raises(SourceIntegrityError):
        math500.load_inventory(cfg, run_paths)


def test_config_source_mismatch_is_refused(cfg, run_paths):
    bad = json.loads(json.dumps(cfg))
    for entry in bad["benchmarks"]:
        if entry["id"] == "math500":
            entry["source"] = "someone/else"
    with pytest.raises(ValueError):
        math500.load_inventory(bad, run_paths)


@needs_math500
def test_private_hash_is_stable_and_binds_the_gold(math_inventory):
    rows = math500.read_rows()
    again = [math500.build_item(row) for row in rows]
    assert [item.private.private_hash for item in again] == [item.private.private_hash for item in math_inventory]
    changed = math500.build_item({**rows[0], "answer": rows[0]["answer"] + "1"})
    assert changed.private.private_hash != again[0].private.private_hash
    assert changed.public == again[0].public


@pytest.mark.parametrize("module", [math500, mmlu_pro])
def test_pinned_sha256_is_enforced(module, monkeypatch):
    monkeypatch.setattr(module, "SOURCE_SHA256", "0" * 64)
    path = module.hf_snapshot_file(module.SOURCE_REPO, module.SOURCE_REVISION, module.SOURCE_FILE)
    if not path.is_file():
        pytest.skip("pinned snapshot not in HF cache")
    with pytest.raises(SourceIntegrityError):
        module.read_rows()


def test_math500_source_ids_are_validated():
    with pytest.raises(SourceIntegrityError):
        math500.build_item(_math_row(unique_id="../../etc/passwd"))


def test_source_records_name_revision_hash_and_licence():
    for module in (math500, mmlu_pro, bbh_logic4):
        record = module.source_record()
        assert "MIT" in record["license"] and record["preprocessing_version"].endswith("-prompt-v1")
    assert math500.source_record()["revision"] == "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    assert mmlu_pro.source_record()["revision"] == "b189ec765aa7ed75c8acfea42df31fdae71f97be"
    assert bbh_logic4.source_record()["commit"] == "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
    assert set(bbh_logic4.source_record()["files"]) == {f"bbh/{s}.json" for s in BBH_SUBTASKS}


@pytest.mark.parametrize("benchmark", ["math500", "mmlu_pro", "bbh_logic4"])
def test_fixture_inventories_are_typed_drawable_and_leak_free(benchmark):
    from agent_allocation.benchmarks.text_manifest import FIXTURE_REVISION, fixture_inventory, select_panels
    items = fixture_inventory(benchmark)
    assert len(items) == 24 and all(item.eligible for item in items)
    assert {item.public.source_revision for item in items} == {FIXTURE_REVISION}
    for item in items:
        assert item.private.gold_answer is not None
        if item.public.output_kind is not OutputKind.BOOLEAN:
            assert item.private.gold_answer not in item.public.prompt.split("\nOptions:\n")[0]
    draw = select_panels(items, benchmark=benchmark, seed=7, n_dev=4, n_main=8, n_reserve=4)
    assert len(draw.main_item_ids) == 8


def test_bbh_structural_exclusions_have_reasons():
    choice = {"input": "Q?\nOptions:\n(A) same\n(B) same\n(C) other", "target": "(A)"}
    assert bbh_logic4.build_item("temporal_sequences", 0, choice).exclusion_reason == "duplicate_option_text"
    gap = {"input": "Q?\nOptions:\n(A) one\n(C) two", "target": "(A)"}
    assert bbh_logic4.build_item("temporal_sequences", 1, gap).exclusion_reason == "option_letters_not_consecutive_from_A"
    missing = {"input": "Q?\nOptions:\n(A) one\n(B) two", "target": "(D)"}
    assert bbh_logic4.build_item("temporal_sequences", 2, missing).exclusion_reason == "target_letter_not_among_options"
    boolean = {"input": "True and False is", "target": "false"}
    assert bbh_logic4.build_item("boolean_expressions", 3, boolean).exclusion_reason == "boolean_target_not_true_false"
    with pytest.raises(SourceIntegrityError):
        bbh_logic4.build_item("temporal_sequences", 4, {"input": "x", "target": "(A)", "rationale": "leak"})


def test_every_ineligible_item_carries_a_reason(math_inventory, mmlu_inventory, bbh_inventory):
    for inventory in (math_inventory, mmlu_inventory, bbh_inventory):
        records = exclusion_records(inventory)             # raises if any ineligible item lacks a reason
        assert len(records) == sum(not item.eligible for item in inventory)


def test_source_provenance_is_json_for_the_dataset_lock():
    for module in (math500, mmlu_pro, bbh_logic4):
        assert json.loads(json.dumps(module.SOURCE_PROVENANCE)) == module.source_record()
        assert isinstance(module.LICENSE, str) and isinstance(module.PREPROCESSING_VERSION, str)
