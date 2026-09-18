"""WP-3: forced-mode realized tokens (THINK + ANSWER completions), the forced budget check, and effective depth."""
from __future__ import annotations

import copy
import gzip
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from agent_allocation.analysis.costs import forced_call_tokens, policy_cost_table, realized_tokens_of
from agent_allocation.analysis.effective_depth import (
    effective_depth_tables, effective_tokens, fallback_reasoning_text, loop_onset,
)
from agent_allocation.analysis.load import load_run
from agent_allocation.analysis.outcomes import candidate_table, outcome_table
from agent_allocation.config import load_config
from agent_allocation.contracts import Usage, dump_record
from agent_allocation.reporting.cli import _forced_budget_check

PROJECT = Path(__file__).resolve().parents[3]
WORDS = ("the sum of residues modulo prime yields a bound hence we consider each case where n exceeds twice the "
         "square of k and the parity argument fails unless the gcd divides both terms so we check small values "
         "then generalize by induction on the exponent while tracking the carry").split()


def prose(chars: int, seed: int = 1) -> str:
    rng = random.Random(seed)
    out, total = [], 0
    while total < chars:
        word = rng.choice(WORDS) + (str(rng.randint(0, 999)) if rng.random() < 0.2 else "")
        out.append(word)
        total += len(word) + 1
    return " ".join(out)[:chars]


def looped(prefix_chars: int, loop_chars: int) -> str:
    unit = "Wait, let me recheck the computation of the residue once more. "
    return prose(prefix_chars) + (unit * (loop_chars // len(unit) + 1))[:loop_chars]


def forced_config() -> dict:
    cfg = copy.deepcopy(load_config(PROJECT / "configs" / "balanced150.json"))
    cfg["reasoning"]["budget_semantics"] = "forced_exact"
    cfg["forced"] = {"enabled": True}
    return cfg


def forced_usage(prompt: int, budget: int, answer: int) -> Usage:
    """WP-1's summed usage for one forced node: THINK (prompt P, completion = budget) + ANSWER (prompt P + budget)."""
    raw = {"think": {"prompt_tokens": prompt, "completion_tokens": budget},
           "answer": {"prompt_tokens": prompt + budget + 3, "completion_tokens": answer}}
    return Usage(prompt_tokens=2 * prompt + budget + 3, completion_tokens=budget + answer, reasoning_tokens=budget,
                 cached_prompt_tokens=0, actual_cost_usd=None, raw_usage=raw)


# ---------------------------------------------------------------------------------------------------------------
# realized tokens
# ---------------------------------------------------------------------------------------------------------------

def test_forced_realized_tokens_exclude_the_refed_reasoning():
    usage = forced_usage(prompt=500, budget=1024, answer=200)
    assert forced_call_tokens(usage) == (1024, 200)
    assert realized_tokens_of(usage, forced=True) == 1224                  # not 2P + 2*budget + answer
    assert realized_tokens_of(usage, forced=False) == usage.prompt_tokens + usage.completion_tokens
    nested = replace(usage, raw_usage={"think": {"usage": {"completion_tokens": 7}},
                                       "answer": {"usage": {"completion_tokens": 3}}})
    assert realized_tokens_of(nested, forced=True) == 10
    bare = replace(usage, raw_usage={})
    assert forced_call_tokens(bare) is None and realized_tokens_of(bare, forced=True) == 1224
    assert realized_tokens_of(None, forced=True) is None


def _forced_run(mutable_run, *, short_node: bool = False):
    """Rewrite every parsed result of a synthetic run as a forced two-call generation."""
    paths = mutable_run
    run = load_run(paths, forced_config())
    answers = {}
    for index, (node_id, (result, candidate)) in enumerate(sorted(run.parsed.items())):
        spec = next(s for part in run.jobs.values() for n, s in part.items() if n == node_id)
        answer = 100 + index % 50
        budget = spec.reasoning_budget - (1 if short_node and index == 0 else 0)
        usage = forced_usage(300, budget, answer)
        answers[node_id] = (budget, answer)
        target = paths.parsed(node_id)
        payload = json.loads(target.read_text())
        payload["result"] = dump_record(replace(result, usage=usage))
        target.write_text(json.dumps(payload))
    return load_run(paths, forced_config()), answers


def test_policy_costs_forced_realized(mutable_run):
    run, answers = _forced_run(mutable_run)
    candidates = candidate_table(run)
    outcomes = outcome_table(run)
    costs, _problems = policy_cost_table(run, candidates, outcomes)
    specs = {n: s for part in run.jobs.values() for n, s in part.items()}
    cps = {cp.policy_id: cp for part in run.policies.values() for cp in part}
    for _, row in costs.iterrows():
        required = cps[row["policy_id"]].required_node_ids
        assert row["realized_tokens"] == sum(sum(answers[n]) for n in required)
        assert row["answer_tokens"] == sum(answers[n][1] for n in required)
        assert row["requested_reasoning_tokens"] == sum(specs[n].reasoning_budget for n in required)
    table, problems = _forced_budget_check(candidates, run.config)
    assert problems == [] and int(table["short"].sum()) == 0 and int(table["exact"].sum()) == len(answers)


def test_forced_budget_check_flags_a_short_think(mutable_run):
    run, _ = _forced_run(mutable_run, short_node=True)
    table, problems = _forced_budget_check(candidate_table(run), run.config)
    assert int(table["short"].sum()) == 1 and len(problems) == 1 and "short 1" in problems[0]
    soft, none = _forced_budget_check(candidate_table(run), load_config(PROJECT / "configs" / "balanced150.json"))
    assert soft is None and none == []


# ---------------------------------------------------------------------------------------------------------------
# effective depth
# ---------------------------------------------------------------------------------------------------------------

def test_loop_onset_prose_versus_loop():
    assert loop_onset(prose(40_000)) is None
    text = looped(20_000, 20_000)
    onset = loop_onset(text)
    assert onset is not None and 0.4 <= onset <= 0.5                   # the loop starts at the midpoint
    assert loop_onset(looped(0, 10_000)) == 0.0
    assert loop_onset("") is None
    assert effective_tokens(0.5, 4096, 4096) == 2048 and effective_tokens(None, 4096, 4000) == 4000


def _think_body(reasoning: str, *, field: bool) -> str:
    choice = {"text": reasoning, "finish_reason": "length"} if not field else \
        {"text": "", "reasoning": reasoning, "finish_reason": "length"}
    return json.dumps({"id": "gen-think", "choices": [choice], "usage": {"completion_tokens": 1024}})


def test_fallback_reasoning_text_parses_forced_and_chat_layouts():
    forced = {"response_body_text": json.dumps({"think": _think_body("abc", field=False),
                                                "answer": json.dumps({"choices": [{"text": "<final>1</final>"}]})})}
    assert fallback_reasoning_text(forced) == "abc"
    as_field = {"response_body_text": json.dumps({"think": json.loads(_think_body("xyz", field=True))})}
    assert fallback_reasoning_text(as_field) == "xyz"
    inline = {"response_body_text": _think_body("thinking</think>answer", field=False)}
    assert fallback_reasoning_text(inline) == "thinking"
    chat = {"response_body_text": json.dumps({"choices": [{"message": {"content": "x", "reasoning": "chat r"}}]})}
    assert fallback_reasoning_text(chat) == "chat r"
    assert fallback_reasoning_text({"response_body_text": "not json"}) is None


def test_effective_depth_tables(mutable_run):
    run, _ = _forced_run(mutable_run)
    specs = {n: s for part in run.jobs.values() for n, s in part.items()}
    deep = {n for n, s in specs.items() if s.kind.value == "deep"}
    for node_id, (result, _) in run.parsed.items():
        text = looped(6_000, 6_000) if node_id in deep else prose(12_000, seed=len(node_id))
        payload = {"attempt_id": result.attempt_id, "node_id": node_id,
                   "response_body_text": json.dumps({"think": _think_body(text, field=False), "answer": "{}"})}
        with gzip.open(run.paths.run_dir / result.raw_response_path, "wt") as handle:
            json.dump(payload, handle)
    tables = effective_depth_tables(run, reader=fallback_reasoning_text)
    nodes = tables["effective_depth_nodes"].set_index("node_id")
    assert nodes["text_available"].all()
    assert set(nodes.index[nodes["looped"]]) == deep
    for n in deep:
        onset = nodes.loc[n, "loop_onset_fraction"]
        assert nodes.loc[n, "effective_tokens"] == pytest.approx(onset * specs[n].reasoning_budget)
    prose_nodes = nodes[~nodes["looped"]]
    assert (prose_nodes["effective_tokens"] == prose_nodes["reasoning_tokens"]).all()
    per_item = tables["effective_depth"]
    summary = tables["effective_depth_summary"].set_index(["partition_id", "name"])
    assert summary.loc[("main", "sample/3"), "mean_effective_fraction"] == pytest.approx(1.0)
    assert summary.loc[("main", "deep/2"), "mean_effective_fraction"] < 0.6
    assert (per_item["requested_reasoning_tokens"] > 0).all()
