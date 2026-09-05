"""prompts/: frozen templates, hashes and the amended framing clauses (§5.5, P0-2, S1b)."""

from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path

import pytest

from agents_scaling.study import prompts
from tests.study.conftest import HANDOFF_DIR

HANDOFF_TEMPLATES = (
    "independent_root", "focal_revision", "central_hub", "central_worker", "common_consumer",
    "judge_best", "forecast", "rlm_root", "rlm_child", "transport_mediator",
)
AUTHORED_TEMPLATES = (
    "dec_root_truthful", "hle_judge", "cen_final_instruction", "dec_revision_contract", "dec_revision_contract_n1",
)
JUDGE_SCRIPT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/study_v4/data/raw/hle_run_judge_results.py")


def test_every_template_loads():
    assert set(prompts.TEMPLATE_NAMES) == set(HANDOFF_TEMPLATES) | set(AUTHORED_TEMPLATES)
    for name in prompts.TEMPLATE_NAMES:
        text = prompts.load_template(name)
        assert text.strip(), name
    with pytest.raises(KeyError):
        prompts.load_template("does_not_exist")


def test_handoff_templates_are_byte_exact_copies():
    for name in HANDOFF_TEMPLATES:
        assert prompts.template_path(name).read_bytes() == (HANDOFF_DIR / "prompts" / f"{name}.txt").read_bytes(), name


def test_prompt_hashes_match_recomputed():
    pinned = json.loads(prompts.HASHES_FILE.read_text(encoding="utf-8"))
    assert pinned == prompts.compute_prompt_hashes()
    assert prompts.prompt_hashes() == pinned
    expected_keys = {f"templates/{n}.txt" for n in prompts.TEMPLATE_NAMES} | {"framing_clauses.v4_orcd.json"}
    assert set(pinned) == expected_keys
    assert "PROMPT_HASHES.json" not in pinned
    assert all(len(v) == 64 for v in pinned.values())


def test_framing_clauses_team_vote_identical_and_code_selector_amended():
    ours = prompts.load_framing_clauses()
    theirs = json.loads((HANDOFF_DIR / "prompts" / "framing_clauses.json").read_text(encoding="utf-8"))
    for key in ("TEAM_0", "TEAM_1", "VOTE_0", "VOTE_1", "HLE_CHOICE_SELECTOR"):
        assert ours[key].encode("utf-8") == theirs[key].encode("utf-8"), key
    assert ours["CODE_SELECTOR"] != theirs["CODE_SELECTOR"]
    assert "AST" in ours["CODE_SELECTOR"]
    assert ours["CODE_SELECTOR"] == (
        "Complete programs are grouped by syntactic identity after Python AST canonicalization, with exact-source "
        "identity if the program does not parse. The largest group wins; ties use a fixed rule independent of "
        "correctness. Hidden tests are unavailable to this selection procedure."
    )
    assert "probe suite" not in ours["CODE_SELECTOR"]


def test_dec_root_truthful_structure():
    root = prompts.load_template("independent_root")
    dec = prompts.load_template("dec_root_truthful")
    a3 = ("You are one of several solvers working independently on the same task. In later rounds you may receive "
          "bounded messages from the other solvers and revise. Your final complete answer, and theirs, will be "
          "combined by the fixed answer-selection procedure described below.")
    assert a3 + "\n{{FROZEN_SELECTOR_DESCRIPTION}}\n" in dec
    # same solving instruction, confidence sentence, Task and Output-contract lines as the neutral root
    assert dec.startswith(root.split("\n\n")[0])
    for block in root.split("\n\n")[-2:]:
        assert block in dec
    assert "{{TEAM_CLAUSE_OR_EMPTY}}" not in dec and "{{VOTE_AWARE_CLAUSE" not in dec


def test_dec_revision_contracts():
    n = prompts.load_template("dec_revision_contract")
    assert "{{N}}" in n and "{{N_MINUS_1}}" in n and "{{FROZEN_SELECTOR_DESCRIPTION}}" in n
    n1 = prompts.load_template("dec_revision_contract_n1")
    assert "{{N}}" not in n1 and "other" not in n1.lower() and "peer" not in n1.lower()
    assert "{{FROZEN_SELECTOR_DESCRIPTION}}" in n1


def test_cen_final_instruction_exact():
    assert prompts.load_template("cen_final_instruction") == (
        "No further worker calls are available. Use the original task and the returned worker results as fallible "
        "evidence. Return exactly one coordinator_action object with action \"final\" containing the complete "
        "candidate JSON. You may not delegate again or invent a result of an unperformed check."
    )


def test_hle_judge_prompt_placeholders_and_literal():
    text = prompts.load_template("hle_judge")
    for placeholder in ("{question}", "{response}", "{correct_answer}"):
        assert text.count(placeholder) == 1
    assert text.startswith("Judge whether the following [response] to [question] is correct or not")
    assert text.endswith("Put 100 if there is no confidence score available.")
    if not JUDGE_SCRIPT.exists():
        pytest.skip("official judge script not available on this host")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the upstream script has a non-raw "\\%" escape
        tree = ast.parse(JUDGE_SCRIPT.read_text(encoding="utf-8"))
    literal = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "JUDGE_PROMPT" for t in node.targets)
    )
    assert text == literal
