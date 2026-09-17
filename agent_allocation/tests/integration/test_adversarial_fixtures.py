"""The adversarial fixtures are themselves checked: no drift from the builder, contract-valid records, canaries only
on the private side, and each fixture really exhibits the hazard it is meant to provoke."""
from __future__ import annotations

import json
from collections import Counter

from agent_allocation.contracts import (
    GenerationResult,
    GenerationSpec,
    GradeResult,
    PublicTask,
    SelectionResult,
    SelectionStatus,
    dump_record,
    load_record,
)

from tests.integration import adversarial
from tests.integration.reference_impl import text_contains_canary, tri_state


def test_committed_fixtures_match_the_builder_byte_for_byte(tmp_path):
    adversarial.write_all(tmp_path)
    for name in adversarial.render_all():
        committed = (adversarial.FIXTURE_DIR / name).read_bytes()
        assert committed == (tmp_path / name).read_bytes(), f"{name} drifted: rerun python -m tests.integration.adversarial"


def test_no_literal_key_shaped_string_is_committed():
    key = adversarial.fake_api_key()
    assert key.startswith("sk-or-v1-") and len(key) > 40
    for path in adversarial.FIXTURE_DIR.iterdir():
        assert key not in path.read_text(encoding="utf-8"), path.name


def test_items_cover_four_benchmarks_and_keep_canaries_private(adv_items, canaries):
    assert Counter(t.benchmark for t, _, _ in adv_items) == {"math500": 2, "mmlu_pro": 2, "bbh_logic4": 2, "mbpp_plus": 2}
    planted = set()
    for task, private, record in adv_items:
        assert task.item_id == private.item_id
        public_json = json.dumps(dump_record(task))
        assert text_contains_canary(public_json, canaries) == [], task.item_id
        planted.update(text_contains_canary(json.dumps(record) + json.dumps(dump_record(private)), canaries))
    assert planted == set(adversarial.CANARIES), "every canary token must be planted in some private record"


def test_episode_label_collision_fixture_really_collides():
    fx = adversarial.load_json("episode_label_collision.json")
    legacy = {row["episode_id"]: row for row in fx["legacy_rows"]}
    assert len(legacy) == 1 < len(fx["legacy_rows"]), "the legacy label-keyed store must lose a row"
    by_label = Counter(s["episode_label"] for s in fx["specs"])
    assert max(by_label.values()) >= 3
    ids = [load_record(GenerationSpec, s["record"]).node_id for s in fx["specs"]]
    assert ids == [s["node_id"] for s in fx["specs"]]
    assert len(set(ids)) == len(ids)


def test_prompt_identical_draws_fixture_has_distinct_ids():
    fx = adversarial.load_json("prompt_identical_draws.json")
    specs = [load_record(GenerationSpec, r["record"]) for r in fx["roots"] + fx["canary_repeats"]]
    assert len({s.node_id for s in specs}) == 28
    assert len({s.input_manifest_hash for s in specs}) == 1


def test_duplicate_completion_fixture_contains_duplicates():
    rows = adversarial.load_jsonl("duplicate_completions.jsonl")
    results = [load_record(GenerationResult, r["result"]) for r in rows]
    assert len({r.attempt_id for r in results}) < len(results)
    assert len({r.node_id for r in results}) == 2


def test_selector_fixtures_load_and_are_null():
    null_rows = [load_record(SelectionResult, r) for r in adversarial.load_jsonl("selector_all_null.jsonl")]
    assert null_rows and all(r.selected_candidate_id is None for r in null_rows)
    assert all(r.status == SelectionStatus.SELECTED.value for r in null_rows)
    abstained = [load_record(SelectionResult, r) for r in adversarial.load_jsonl("selector_abstained.jsonl")]
    assert all(r.status == SelectionStatus.ABSTAINED_ALL_INVALID.value for r in abstained)


def test_grade_fixture_states():
    for row in adversarial.load_jsonl("grades_unknown.jsonl"):
        grade = load_record(GradeResult, row["record"])
        assert tri_state(grade) == row["expected_state"], row["case"]


def test_malformed_and_packet_fixtures_expand():
    cases = adversarial.load_json("malformed_finals.json")
    assert len(cases) >= 10 and all("valid" not in c["acceptable_parse_statuses"] for c in cases)
    growth = adversarial.expand_peer_packet_growth(adversarial.load_json("peer_packet_growth.json"))
    assert growth["rationale_chars"] >= 150_000
    assert all(c["final_marker"] in c["visible_content"] for c in growth["candidates"])
    assert len(growth["oversized_final"]["visible_content"]) > 1_000_000
    load_record(PublicTask, adversarial.load_json("prompt_identical_draws.json")["item"])
