"""Deterministic builder + loader for ``tests/fixtures/adversarial/``.

Run ``uv run --frozen --no-sync python -m tests.integration.adversarial`` to (re)write the fixture files.
``test_adversarial_fixtures.py`` regenerates into a temp dir and requires byte equality, so the committed
fixtures can never drift from this builder silently. Every contract record goes through ``dump_record``.

Fixtures (all synthetic; no benchmark data, no model output):
  canaries.json                 secret tokens planted in private records + a split fake OpenRouter key
  items.jsonl                   two items per benchmark: PublicTask, PrivateEvalSpec, raw private record
  episode_label_collision.json  the ARD36 root-bank / private-baseline collision, recreated with equal labels
  prompt_identical_draws.json   25 roots + repeated canary draws whose rendered prompts are identical
  duplicate_completions.jsonl   completion notifications delivered twice / out of order / late retry
  selector_all_null.jsonl       every SelectionResult null while claiming "selected" (must block)
  selector_abstained.jsonl      legitimate all-invalid abstentions (operational zero, labelled)
  grades_unknown.jsonl          tri-state grades incl. evaluator outage and inconsistent GRADED/None
  malformed_finals.json         outputs whose <final> must not parse as a valid answer
  peer_packet_growth.json       huge rationales around finals that must survive packet trimming
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from agent_allocation.contracts import (
    STUDY_ID,
    BBH_SUBTASKS,
    CompletionStatus,
    EvalKind,
    GenerationKind,
    GenerationResult,
    GenerationSpec,
    GradeResult,
    GradeStatus,
    OutputKind,
    ParseStatus,
    PrivateEvalSpec,
    PublicTask,
    SelectionResult,
    SelectionStatus,
    TransportStatus,
    Usage,
    canonical_hash,
    dump_record,
    load_record,
    public_task_hash,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "adversarial"
MODEL = "google/gemini-2.5-flash-lite"
LOCK_HASH = "adversarial-endpoint-lock"
DECODING_HASH = "adversarial-decoding"
PROMPT_VERSION = "prompts-v1-adversarial"
REVISION = "adversarial-rev-0"

# Canary tokens: [A-Z0-9-] only, so JSON escaping, redaction or HTML rendering cannot disguise them.
CANARIES: dict[str, str] = {
    "gold_math": "CANARY-GOLD-MATH-9C41E2",
    "solution_math": "CANARY-SOLUTION-MATH-2D6A90",
    "rationale_mmlu": "CANARY-RATIONALE-MMLU-5B7D10",
    "target_note_bbh": "CANARY-TARGET-BBH-3E8F66",
    "private_test_mbpp": "CANARY-PRIVATE-TEST-MBPP-A1D2F3",
    "reference_code_mbpp": "CANARY-REFERENCE-CODE-MBPP-77B0C4",
    "plus_input_mbpp": "CANARY-PLUS-INPUT-MBPP-0F9E1D",
}
# Stored split so no literal key-shaped string is committed; ``fake_api_key()`` joins it at runtime.
FAKE_KEY_PARTS = ("sk-or-" + "v1-", "c4n4ry0000fake0000key0000do0not0use0000" + "0123456789abcdef0123456789")


def fake_api_key() -> str:
    return "".join(FAKE_KEY_PARTS)


def all_canaries(include_key: bool = True) -> dict[str, str]:
    out = dict(CANARIES)
    if include_key:
        out["fake_api_key"] = fake_api_key()
    return out


# ---------------------------------------------------------------------------------------------------------
# Items: public/private split with canaries only on the private side
# ---------------------------------------------------------------------------------------------------------

MMLU_CHOICES = tuple("ABCDEFGHIJ")
BBH_CHOICES = ("A", "B", "C", "D", "E", "F", "G")


def _item(public: PublicTask, eval_kind: EvalKind, gold: str | None, private_record: dict[str, Any],
          ref: str | None = None) -> dict[str, Any]:
    private = PrivateEvalSpec(item_id=public.item_id, benchmark=public.benchmark, source_revision=REVISION,
                              eval_kind=eval_kind, gold_answer=gold, private_record_ref=ref,
                              private_hash=canonical_hash(private_record), subtask=public.subtask)
    return {"public": dump_record(public), "private": dump_record(private), "private_record": private_record}


def build_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    # MATH-500: one ordinary gold, one canary gold (a correct answer is then impossible: grading must say wrong).
    items.append(_item(PublicTask("math500/adv/1", "math500", REVISION, "Compute $2+2$.", OutputKind.MATH),
                       EvalKind.MATH_VERIFY, "4",
                       {"problem": "Compute $2+2$.", "answer": "4", "solution": CANARIES["solution_math"],
                        "level": 1, "type": "Prealgebra"}))
    items.append(_item(PublicTask("math500/adv/2", "math500", REVISION, "What is $\\frac{1}{2}+\\frac{1}{2}$?",
                                  OutputKind.MATH),
                       EvalKind.MATH_VERIFY, CANARIES["gold_math"],
                       {"problem": "What is $\\frac{1}{2}+\\frac{1}{2}$?", "answer": CANARIES["gold_math"],
                        "solution": CANARIES["solution_math"], "level": 2, "type": "Algebra"}))
    # MMLU-Pro: letters are gold; the canary rides in the reference rationale.
    options = "\n".join(f"({c}) option {c.lower()}" for c in MMLU_CHOICES)
    for n, gold in ((1, "C"), (2, "J")):
        prompt = f"Question {n}: which option is the adversarial fixture's answer?\n{options}"
        items.append(_item(PublicTask(f"mmlu_pro/adv/{n}", "mmlu_pro", REVISION, prompt, OutputKind.OPTION,
                                      allowed_choices=MMLU_CHOICES),
                           EvalKind.OPTION_EXACT, gold,
                           {"question": prompt, "answer": gold, "answer_index": MMLU_CHOICES.index(gold),
                            "cot_content": CANARIES["rationale_mmlu"], "src": "adversarial"}))
    # BBH: one choice subtask, one boolean subtask.
    items.append(_item(PublicTask("bbh_logic4/adv/1", "bbh_logic4", REVISION,
                                  "Three objects are arranged... Options:\n(A) first\n(B) second\n(C) third",
                                  OutputKind.OPTION, allowed_choices=("A", "B", "C"), subtask=BBH_SUBTASKS[0]),
                       EvalKind.TASK_EXACT, "B",
                       {"input": "Three objects...", "target": "(B)", "target_note": CANARIES["target_note_bbh"]}))
    items.append(_item(PublicTask("bbh_logic4/adv/2", "bbh_logic4", REVISION, "not ( True ) and ( True ) is",
                                  OutputKind.BOOLEAN, allowed_choices=("True", "False"), subtask=BBH_SUBTASKS[2]),
                       EvalKind.TASK_EXACT, "False",
                       {"input": "not ( True ) and ( True ) is", "target": "False",
                        "target_note": CANARIES["target_note_bbh"]}))
    # MBPP+: the prompt carries exactly one public assert; private tests/reference/plus inputs carry canaries.
    for n, (fn, public_assert) in enumerate((("add_two", "assert add_two(1, 2) == 3"),
                                             ("rev_str", "assert rev_str('ab') == 'ba'")), start=1):
        prompt = f"Write a function {fn} as described.\n{public_assert}"
        items.append(_item(PublicTask(f"mbpp_plus/adv/{n}", "mbpp_plus", REVISION, prompt, OutputKind.PYTHON,
                                      public_examples=(public_assert,)),
                           EvalKind.EVALPLUS_FULL, None,
                           {"task_id": f"Mbpp/adv{n}", "prompt": prompt,
                            "canonical_solution": f"# {CANARIES['reference_code_mbpp']}\n",
                            "assertion": f"assert {fn}('{CANARIES['private_test_mbpp']}') is not None",
                            "base_input": [[CANARIES["private_test_mbpp"]]],
                            "plus_input": [[CANARIES["plus_input_mbpp"]]]},
                           ref=f"private/evalplus_oracles/adv{n}.json"))
    return items


# ---------------------------------------------------------------------------------------------------------
# Generation-identity fixtures
# ---------------------------------------------------------------------------------------------------------

def _spec(task: PublicTask, kind: GenerationKind, *, slot: int = 0, team: int = 0, rnd: int = 0, budget: int = 1024,
          headroom: int = 1024, draw: int = 0, parents: tuple[str, ...] = (), partition: str = "main") -> GenerationSpec:
    return GenerationSpec(run_id=STUDY_ID, partition_id=partition, item_id=task.item_id, benchmark=task.benchmark,
                          kind=kind, model=MODEL, endpoint_lock_hash=LOCK_HASH,
                          input_manifest_hash=public_task_hash(task), decoding_config_hash=DECODING_HASH,
                          prompt_version=PROMPT_VERSION, reasoning_budget=budget, completion_headroom=headroom,
                          root_slot=slot, team_size=team, update_round=rnd, draw_id=draw, parent_ids=parents)


def legacy_episode_label(spec: GenerationSpec) -> str:
    """The pre-contract key that collided in ARD36: item + slot + round, no record kind / team / budget."""
    return f"{spec.item_id}/ep{spec.root_slot}/r{spec.update_round}"


def build_episode_label_collision(task: PublicTask) -> dict[str, Any]:
    roots = [_spec(task, GenerationKind.ROOT, slot=j) for j in range(5)]
    private_0_1 = _spec(task, GenerationKind.PRIVATE, slot=0, rnd=1, parents=(roots[0].node_id,))
    private_1_1 = _spec(task, GenerationKind.PRIVATE, slot=1, rnd=1, parents=(roots[1].node_id,))
    peer_2_1_1 = _spec(task, GenerationKind.PEER, slot=1, team=2, rnd=1, parents=(roots[0].node_id, roots[1].node_id))
    peer_3_1_1 = _spec(task, GenerationKind.PEER, slot=1, team=3, rnd=1,
                       parents=tuple(r.node_id for r in roots[:3]))
    deep_2 = _spec(task, GenerationKind.DEEP, budget=2048)
    pooled = _spec(task, GenerationKind.POOLED, team=4, parents=tuple(r.node_id for r in roots[:4]))
    high_root_0 = _spec(task, GenerationKind.ROOT, slot=0, budget=4096, partition="high")
    groups = {
        # every spec in a group shares one legacy episode label
        "slot0_round0": [roots[0], deep_2, pooled, high_root_0],
        "slot1_round1": [private_1_1, peer_2_1_1, peer_3_1_1],
        "slot0_round1": [private_0_1],
    }
    specs = []
    for group, members in groups.items():
        labels = {legacy_episode_label(s) for s in members}
        assert len(labels) == 1, group
        for s in members:
            specs.append({"group": group, "episode_label": legacy_episode_label(s), "node_id": s.node_id,
                          "record": dump_record(s),
                          "decision": {"visible_content": f"<rationale>{s.kind.value} slot {s.root_slot}"
                                                          f"</rationale><final>{s.kind.value}-{s.root_slot}</final>"}})
    # What the prior study wrote: a root-bank row and a private-baseline row under one episode label.
    legacy_rows = [
        {"episode_id": legacy_episode_label(roots[0]), "record_kind": "root_bank", "answer": "7"},
        {"episode_id": legacy_episode_label(roots[0]), "record_kind": "private_baseline", "answer": "9"},
    ]
    return {"item": dump_record(task), "legacy_rows": legacy_rows, "specs": specs}


def build_prompt_identical_draws(task: PublicTask) -> dict[str, Any]:
    roots = [_spec(task, GenerationKind.ROOT, slot=j) for j in range(25)]
    canary_repeats = [_spec(task, GenerationKind.ROOT, slot=0, draw=d, partition="canary") for d in range(3)]
    return {
        "item": dump_record(task),
        "roots": [{"node_id": s.node_id, "record": dump_record(s)} for s in roots],
        "canary_repeats": [{"node_id": s.node_id, "record": dump_record(s)} for s in canary_repeats],
        "identical_response_content": "<rationale>2 plus 2 is 4.</rationale>\n<final>4</final>",
    }


def _result(spec: GenerationSpec, attempt_no: int, *, status: TransportStatus = TransportStatus.OK,
            content: str | None = "<rationale>ok</rationale><final>4</final>", cost: str = "0.000321",
            http: int = 200, reasoning: int | None = 300) -> GenerationResult:
    usage = Usage(prompt_tokens=120, completion_tokens=410, reasoning_tokens=reasoning, cached_prompt_tokens=0,
                  actual_cost_usd=Decimal(cost), raw_usage={"cost": cost})
    ok = status is TransportStatus.OK
    return GenerationResult(node_id=spec.node_id, attempt_id=f"{spec.node_id}:{attempt_no}", transport_status=status,
                            completion_status=CompletionStatus.COMPLETED if ok else None,
                            generation_id=f"gen-adv-{attempt_no}" if ok else None, observed_model=MODEL if ok else None,
                            observed_provider="Google AI Studio" if ok else None, finish_reason="stop" if ok else None,
                            native_finish_reason="STOP" if ok else None, visible_content=content if ok else None,
                            reasoning_returned=False, usage=usage if ok else None, cache_status="MISS" if ok else None,
                            raw_response_path=None, raw_response_sha256=None, latency_seconds=1.5, http_status=http,
                            retry_after_seconds=None if ok else 2.0)


def build_duplicate_completions(task: PublicTask) -> list[dict[str, Any]]:
    a = _spec(task, GenerationKind.ROOT, slot=0)
    b = _spec(task, GenerationKind.ROOT, slot=1)
    rows = [
        ("a first attempt succeeds", _result(a, 1)),
        ("a same notification delivered twice", _result(a, 1)),
        ("b first attempt rate-limited", _result(b, 1, status=TransportStatus.RATE_LIMITED, http=429)),
        ("b retry succeeds", _result(b, 2, content="<final>5</final>", cost="0.000400")),
        ("b stale notification for attempt 1 arrives after success",
         _result(b, 1, status=TransportStatus.RATE_LIMITED, http=429)),
        ("b retry notification duplicated", _result(b, 2, content="<final>5</final>", cost="0.000400")),
    ]
    return [{"seq": i, "case": case, "node_spec": dump_record(a if r.node_id == a.node_id else b),
             "result": dump_record(r)} for i, (case, r) in enumerate(rows)]


# ---------------------------------------------------------------------------------------------------------
# Selector, grade, extraction, packet fixtures
# ---------------------------------------------------------------------------------------------------------

def build_selector_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    names = ["sample/5", "private_team/4/4", "peer/4/4", "peer/5/4"]
    null_rows = [dump_record(SelectionResult(policy_id=f"main/math500/adv/1/{n}/current", selected_candidate_id=None,
                                             selector_name="plurality", selector_version="v1", pool_hash="",
                                             public_evidence_hash=None, cpu_seconds=0.0,
                                             status=SelectionStatus.SELECTED.value)) for n in names]
    abstained = [dump_record(SelectionResult(policy_id=f"main/math500/adv/1/{n}/current", selected_candidate_id=None,
                                             selector_name="plurality", selector_version="v1",
                                             pool_hash=canonical_hash([n]), public_evidence_hash=None,
                                             cpu_seconds=0.001, status=SelectionStatus.ABSTAINED_ALL_INVALID.value))
                 for n in names]
    return null_rows, abstained


def build_grades() -> list[dict[str, Any]]:
    cases = [
        ("graded correct", GradeResult("c-correct", GradeStatus.GRADED, True, True, "v1", "h"), "correct"),
        ("graded valid wrong answer", GradeResult("c-wrong", GradeStatus.GRADED, False, True, "v1", "h"), "wrong"),
        ("parser failure is invalid, not wrong",
         GradeResult("c-invalid", GradeStatus.INVALID_OUTPUT, None, False, "v1", "h", detail_code="missing_final"),
         "invalid"),
        ("evaluator outage is unknown",
         GradeResult("c-outage", GradeStatus.EVALUATOR_ERROR, None, None, "v1", "h", detail_code="sandbox_down"),
         "unknown"),
        ("math-verify timeout is unknown",
         GradeResult("c-timeout", GradeStatus.EVALUATOR_ERROR, None, None, "v1", "h", detail_code="timeout"),
         "unknown"),
        ("not evaluated yet is unknown", GradeResult("c-pending", GradeStatus.NOT_EVALUATED, None, None, "v1", "h"),
         "unknown"),
        ("inconsistent: GRADED with null correctness must stay unknown",
         GradeResult("c-inconsistent", GradeStatus.GRADED, None, True, "v1", "h"), "unknown"),
    ]
    return [{"case": case, "expected_state": state, "record": dump_record(g)} for case, g, state in cases]


def build_malformed_finals() -> list[dict[str, Any]]:
    """Cases whose parse status must not be VALID. ``expected`` pins unambiguous statuses; otherwise any
    non-VALID status is acceptable. No case may yield a repaired final answer."""
    non_valid = [s.value for s in ParseStatus if s is not ParseStatus.VALID]
    rows = [
        ("no final element", "math", "stop", "<rationale>The answer is 4.</rationale>\nThe answer is 4.",
         ParseStatus.MISSING.value),
        ("empty visible content", "option", "stop", "", ParseStatus.EMPTY.value),
        ("length-truncated inside final", "math", "length", "<rationale>x</rationale><final>\\frac{1}{", None),
        ("two competing finals", "option", "stop", "<final>A</final>\n<final>B</final>", None),
        ("unclosed final", "option", "stop", "<rationale>r</rationale><final>C", None),
        ("empty final element", "math", "stop", "<rationale>r</rationale><final>   </final>", None),
        ("illegal option letter", "option", "stop", "<final>Z</final>", None),
        ("option answer only in rationale", "option", "stop", "<rationale>So (B) is right.</rationale><final></final>",
         None),
        ("boolean hedge", "boolean", "stop", "<final>True or False</final>", None),
        ("nested finals", "math", "stop", "<final><final>4</final></final>", None),
        ("python prose instead of code", "python", "stop", "<final>I would write a loop that adds numbers.</final>",
         None),
        ("python with two fences", "python", "stop",
         "<final>```python\ndef add_two(a, b):\n    return a + b\n```\n```python\nprint(1)\n```</final>", None),
    ]
    out = []
    for case, kind, finish, content, expected in rows:
        out.append({"case": case, "output_kind": kind, "finish_reason": finish, "visible_content": content,
                    "expected_parse_status": expected, "acceptable_parse_statuses": non_valid,
                    "allowed_choices": list(MMLU_CHOICES[:4]) if kind == "option" else
                    (["True", "False"] if kind == "boolean" else [])})
    return out


def build_peer_packet_growth() -> dict[str, Any]:
    """A compact recipe (the expanded text is ~3 MB); ``expand_peer_packet_growth`` materializes it."""
    return {
        "rationale_unit": "step ",
        "rationale_repeats": 40000,                  # ~200 KB of rationale per candidate
        "final_markers": [f"PACKET-FINAL-{j}-" + ("x" * 64) for j in range(5)],
        "oversized_final_marker": "PACKET-OVERSIZED-FINAL",
        "oversized_final_filler": "y",
        "oversized_final_repeats": 2_000_000,        # a final that alone cannot fit any sane envelope
    }


def expand_peer_packet_growth(recipe: dict[str, Any]) -> dict[str, Any]:
    rationale = recipe["rationale_unit"] * recipe["rationale_repeats"]
    candidates = [{"slot": j, "final_marker": marker,
                   "visible_content": f"<rationale>{rationale}</rationale>\n<final>{marker}</final>"}
                  for j, marker in enumerate(recipe["final_markers"])]
    big = recipe["oversized_final_marker"] + " " + recipe["oversized_final_filler"] * recipe["oversized_final_repeats"]
    return {"rationale_chars": len(rationale), "candidates": candidates,
            "oversized_final": {"final_marker": recipe["oversized_final_marker"],
                                "visible_content": f"<rationale>short</rationale><final>{big}</final>"}}


# ---------------------------------------------------------------------------------------------------------
# Write / load
# ---------------------------------------------------------------------------------------------------------

def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)


def render_all() -> dict[str, str]:
    items = build_items()
    math_task = load_record(PublicTask, items[0]["public"])
    null_rows, abstained = build_selector_rows()
    return {
        "canaries.json": _dumps({"tokens": CANARIES, "fake_api_key_parts": list(FAKE_KEY_PARTS)}),
        "items.jsonl": _jsonl(items),
        "episode_label_collision.json": _dumps(build_episode_label_collision(math_task)),
        "prompt_identical_draws.json": _dumps(build_prompt_identical_draws(math_task)),
        "duplicate_completions.jsonl": _jsonl(build_duplicate_completions(math_task)),
        "selector_all_null.jsonl": _jsonl(null_rows),
        "selector_abstained.jsonl": _jsonl(abstained),
        "grades_unknown.jsonl": _jsonl(build_grades()),
        "malformed_finals.json": _dumps(build_malformed_finals()),
        "peer_packet_growth.json": _dumps(build_peer_packet_growth()),
    }


def write_all(directory: Path = FIXTURE_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in render_all().items():
        path = directory / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def load_json(name: str, directory: Path = FIXTURE_DIR) -> Any:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def load_jsonl(name: str, directory: Path = FIXTURE_DIR) -> list[Any]:
    return [json.loads(line) for line in (directory / name).read_text(encoding="utf-8").splitlines() if line]


def load_items(directory: Path = FIXTURE_DIR) -> list[tuple[PublicTask, PrivateEvalSpec, dict[str, Any]]]:
    return [(load_record(PublicTask, row["public"]), load_record(PrivateEvalSpec, row["private"]),
             row["private_record"]) for row in load_jsonl("items.jsonl", directory)]


if __name__ == "__main__":
    for p in write_all():
        print(p)
