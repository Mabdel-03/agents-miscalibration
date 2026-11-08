"""Contract §4 rendering: byte parity with ``api/forced.py::render_chat_prompt`` on the golden cases and every main
prompt, equality with the transformers oracle goldens, no HTML-escaped ``<``, tool / tool-call / tool-response /
empty-reasoning rendering, the frozen effort rule, and a drifted template refusing to render."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_allocation.api import forced
from agent_allocation.api.forced import GENERATION_PROMPT, TemplateError, XHIGH_INSTRUCTION, render_chat_prompt
from agent_allocation.dynorch import template
from agent_allocation.dynorch.records import Message, ToolCall, ToolSpec
from agent_allocation.dynorch.template import (
    RenderError, THINK_DISABLED_SUFFIX, render_chat, requested_effort, template_effort,
)

from .conftest import AGENT_TOOL, PROJECT, ROOT_SYSTEM, SUBAGENT_SYSTEM

FIXTURES = PROJECT / "tests" / "fixtures"
FORCED_GOLDEN = FIXTURES / "forced" / "qwen38_template_golden.json"
ORACLE_CASES = FIXTURES / "dynorch" / "oracle_cases.json"
ORACLE_GOLDENS = FIXTURES / "dynorch" / "oracle_goldens.json"
BASE_RUN_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/artifacts")


def _messages(dicts: list[dict]) -> list[Message]:
    out = []
    for d in dicts:
        calls = tuple(ToolCall(c["id"], c["function"]["name"], c["function"]["arguments"])
                      for c in d.get("tool_calls", []))
        out.append(Message(role=d["role"], content=d["content"], reasoning_content=d.get("reasoning_content"),
                           tool_calls=calls, tool_call_id=d.get("tool_call_id")))
    return out


def _tools(dicts: list[dict] | None) -> tuple[ToolSpec, ...]:
    return tuple(ToolSpec(t["function"]["name"], t["function"]["description"], t["function"]["parameters"])
                 for t in (dicts or []))


# ---- parity with the forced_v2 renderer ---------------------------------------------------------------------------

@pytest.mark.parametrize("index", range(len(json.loads(FORCED_GOLDEN.read_text())["cases"])))
def test_parity_with_render_chat_prompt_on_golden_cases(index):
    case = json.loads(FORCED_GOLDEN.read_text())["cases"][index]
    ours = render_chat([Message(m["role"], m["content"]) for m in case["messages"]])
    assert ours == case["rendered"] == render_chat_prompt(case["messages"])


@pytest.mark.skipif(not (BASE_RUN_ROOT / "manifests" / "public").is_dir(), reason="sealed base run root not mounted")
def test_parity_on_all_main_prompts():
    """Gate 0: byte-identical to ``render_chat_prompt`` on every one of the 96 main prompts (plus the dev items)."""
    from agent_allocation.prompts.render import output_requirement
    from agent_allocation.contracts import PublicTask, load_record

    root_txt = (PROJECT / "prompts" / "root.txt").read_text()
    count = 0
    for manifest in sorted((BASE_RUN_ROOT / "manifests" / "public").glob("*.jsonl")):
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            task = load_record(PublicTask, json.loads(line))
            user = root_txt.replace("{task_prompt}", task.prompt).replace("{output_requirement}",
                                                                          output_requirement(task))
            for system in (ROOT_SYSTEM, (PROJECT / "prompts" / "system.txt").read_text()):
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                assert render_chat([Message(m["role"], m["content"]) for m in messages]) == \
                    render_chat_prompt(messages)
            count += 1
    assert count >= 96


# ---- the transformers oracle --------------------------------------------------------------------------------------

def _oracle_cases():
    cases = json.loads(ORACLE_CASES.read_text(encoding="utf-8"))["cases"]
    goldens = {c["name"]: c["rendered"] for c in json.loads(ORACLE_GOLDENS.read_text(encoding="utf-8"))["cases"]}
    return [(c, goldens[c["name"]]) for c in cases]


def test_oracle_goldens_pin_the_vendored_template():
    goldens = json.loads(ORACLE_GOLDENS.read_text(encoding="utf-8"))
    assert goldens["template_sha256"] == forced.template_sha256()
    assert goldens["renderer"].startswith("transformers 5.9.0")
    assert "jinja2 3.1.6" in goldens["renderer"]


@pytest.mark.parametrize("case,golden", _oracle_cases(), ids=[c["name"] for c, _ in _oracle_cases()])
def test_render_equals_transformers_oracle_byte_for_byte(case, golden):
    ours = render_chat(_messages(case["messages"]), tools=_tools(case["tools"]),
                       reasoning_effort=case["reasoning_effort"], enable_thinking=case["enable_thinking"])
    assert ours == golden
    assert "\\u003c" not in ours and "\\u003c" not in golden
    assert "&lt;" not in ours


def test_tools_block_is_verbatim_json_without_html_escaping():
    tool = ToolSpec("Agent", "a <b> & 'c' \"d\"", {"type": "object", "properties": {"p": {"type": "string",
                                                                                      "enum": ["x<y"]}}})
    out = render_chat([Message("system", "S"), Message("user", "U")], tools=(tool,))
    assert '{"type": "function", "function": {"name": "Agent", "description": "a <b> & \'c\' \\"d\\"", "parameters": ' \
           '{"type": "object", "properties": {"p": {"type": "string", "enum": ["x<y"]}}}}}' in out
    assert "# Tools\n\nYou have access to the following functions:\n\n<tools>" in out
    assert out.startswith("<|im_start|>system\n" + XHIGH_INSTRUCTION + "\n\n# Tools")
    assert out.endswith("S<|im_end|>\n<|im_start|>user\nU<|im_end|>\n" + GENERATION_PROMPT)


def test_tool_calls_tool_responses_and_empty_reasoning_render_the_xml_blocks():
    messages = [
        Message("system", "S"), Message("user", "U"),
        Message("assistant", "Plan.", reasoning_content="R1",
                tool_calls=(ToolCall("call_1_0", "Agent", {"prompt": "a\nb", "subagent_type": "solver", "n": 2}),
                            ToolCall("call_1_1", "Agent", {"prompt": "c", "subagent_type": "solver"}))),
        Message("tool", "<final>1</final>", tool_call_id="call_1_0"),
        Message("tool", "<final>2</final>", tool_call_id="call_1_1"),
        Message("assistant", "", reasoning_content="", tool_calls=(ToolCall("call_2_0", "Agent", {"prompt": "d"}),)),
        Message("tool", "r", tool_call_id="call_2_0"),
    ]
    out = render_chat(messages, tools=(AGENT_TOOL,))
    assert ("<|im_start|>assistant\n<think>\nR1\n</think>\n\nPlan.\n\n<tool_call>\n<function=Agent>\n"
            "<parameter=prompt>\na\nb\n</parameter>\n<parameter=subagent_type>\nsolver\n</parameter>\n"
            "<parameter=n>\n2\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=Agent>\n"
            "<parameter=prompt>\nc\n</parameter>\n<parameter=subagent_type>\nsolver\n</parameter>\n</function>\n"
            "</tool_call><|im_end|>\n") in out
    assert ("<|im_start|>user\n<tool_response>\n<final>1</final>\n</tool_response>\n<tool_response>\n"
            "<final>2</final>\n</tool_response><|im_end|>\n") in out
    # empty reasoning -> an empty think block; empty content -> the call follows the think block directly
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=Agent>\n<parameter=prompt>\nd\n" in out
    assert out.endswith("<tool_response>\nr\n</tool_response><|im_end|>\n" + GENERATION_PROMPT)


def test_missing_reasoning_renders_an_empty_think_block():
    out = render_chat([Message("user", "U"), Message("assistant", "A"), Message("user", "U2")])
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\nA<|im_end|>\n" in out


def test_raw_string_arguments_are_reemitted_verbatim_inside_the_content():
    raw = "<tool_call>\n<function=Agent>\n<parameter=prompt>\nunterminated"
    out = render_chat([Message("user", "U"), Message("assistant", "Text.", reasoning_content="R",
                                                     tool_calls=(ToolCall("call_1_0", "Agent", raw),)),
                       Message("user", "U2")], tools=(AGENT_TOOL,))
    assert "<think>\nR\n</think>\n\nText.\n\n" + raw + "<|im_end|>" in out
    assert out.count("<function=Agent>") == 1          # no synthetic block was added


def test_thinking_disabled_terminal_render():
    out = render_chat([Message("system", ROOT_SYSTEM), Message("user", "U"),
                       Message("user", "Your completion-token budget for this task is exhausted.")],
                      enable_thinking=False)
    assert out.endswith(THINK_DISABLED_SUFFIX)
    assert "Reasoning effort is set to xhigh" not in out      # the template drops the instruction without thinking


def test_effort_is_the_studys_and_the_harness_value_is_only_recorded():
    for requested in (None, "low", "medium", "high", "xhigh", "anything"):
        assert template_effort(requested) == "xhigh"
        assert requested_effort(requested) == requested
    with pytest.raises(RenderError):
        template_effort("high", configured="high")
    with pytest.raises(RenderError):
        requested_effort(3)
    with pytest.raises(RenderError):
        render_chat([Message("user", "U")], reasoning_effort="high")
    low = render_chat([Message("user", "U")], reasoning_effort="low")
    assert "Reasoning effort is set to low" in low


@pytest.mark.parametrize("messages", [
    [Message("system", "s")],
    [Message("user", "q"), Message("system", "late")],
    [Message("user", "q"), Message("weird", "x")],
    [Message("user", "q"), Message("user", "x", tool_calls=(ToolCall("c", "Agent", {}),))],
])
def test_render_refuses_outside_the_template(messages):
    with pytest.raises(RenderError):
        render_chat(messages)


def test_subagent_prompt_renders_like_the_root_prompt():
    out = render_chat([Message("system", SUBAGENT_SYSTEM), Message("user", "Delegated task.")])
    assert out == render_chat_prompt([{"role": "system", "content": SUBAGENT_SYSTEM},
                                      {"role": "user", "content": "Delegated task."}])


def test_a_drifted_template_is_refused(tmp_path, monkeypatch):
    changed = tmp_path / "qwen3.8-27b.chat_template.jinja"
    changed.write_bytes(forced.template_path().read_bytes() + b"\n")
    monkeypatch.setattr(forced, "TEMPLATES_DIR", tmp_path)
    forced._verified_template.cache_clear()
    template.clear_cache()
    try:
        with pytest.raises(TemplateError, match="sha256"):
            render_chat([Message("user", "q")])
    finally:
        forced._verified_template.cache_clear()
        template.clear_cache()
