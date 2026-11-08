"""Contract §4 grammar: single / parallel / multi-line / typed calls, text after a call and unclosed blocks fail,
unknown functions are flagged, raw-string arguments are re-emitted, and the grammar round-trips the template."""
from __future__ import annotations

import pytest

from agent_allocation.api.fake import fake_tool_call_text
from agent_allocation.dynorch.records import Message, ToolCall, ToolSpec
from agent_allocation.dynorch.template import render_chat
from agent_allocation.dynorch.toolcalls import parse_tool_calls, render_tool_call_xml

TOOLS = (
    ToolSpec("Agent", "d", {"type": "object", "properties": {"prompt": {"type": "string"}, "n": {"type": "integer"},
                                                           "flags": {"type": "array"}, "opts": {"type": "object"},
                                                           "maybe": {"type": ["string", "null"]},
                                                           "untyped": {}}}),
    ToolSpec("update_plan", "p", {"type": "object", "properties": {"plan": {"type": "array"}}}),
)


def test_no_block_is_status_none_with_trimmed_content():
    parse = parse_tool_calls("\n\n<rationale>r</rationale>\n<final>3</final>\n", TOOLS, call_no=1)
    assert parse.status == "none" and parse.calls == () and parse.content == "<rationale>r</rationale>\n<final>3</final>"


def test_single_call_with_prefix_text():
    text = fake_tool_call_text("Agent", {"prompt": "Solve it"}, prefix="Delegating now.")
    parse = parse_tool_calls(text, TOOLS, call_no=7)
    assert parse.status == "ok" and parse.reason is None
    assert parse.content == "Delegating now."
    assert parse.calls == (ToolCall("call_7_0", "Agent", {"prompt": "Solve it"}),)


def test_parallel_calls_get_indexed_ids():
    text = fake_tool_call_text("Agent", {"prompt": "a"}) + "\n" + fake_tool_call_text("update_plan", {"plan": ["x"]})
    parse = parse_tool_calls(text, TOOLS, call_no=2)
    assert [c.id for c in parse.calls] == ["call_2_0", "call_2_1"]
    assert parse.calls[1].arguments == {"plan": ["x"]}
    assert parse.content == ""


def test_multi_line_values_and_typing_by_schema():
    text = fake_tool_call_text("Agent", {"prompt": "line 1\nline 2\n\nline 4", "n": 3, "flags": [1, "two"],
                                         "opts": {"k": {"deep": True}}, "maybe": "12", "untyped": "7"})
    parse = parse_tool_calls(text, TOOLS, call_no=1)
    assert parse.status == "ok"
    assert parse.calls[0].arguments == {"prompt": "line 1\nline 2\n\nline 4", "n": 3, "flags": [1, "two"],
                                        "opts": {"k": {"deep": True}}, "maybe": "12", "untyped": "7"}


def test_empty_parameter_value():
    text = "<tool_call>\n<function=Agent>\n<parameter=prompt>\n\n</parameter>\n</function>\n</tool_call>"
    parse = parse_tool_calls(text, TOOLS, call_no=1)
    assert parse.status == "ok" and parse.calls[0].arguments == {"prompt": ""}


def test_uncoercible_typed_value_falls_back_to_the_raw_string_with_a_note():
    parse = parse_tool_calls(fake_tool_call_text("Agent", {"n": "three"}), TOOLS, call_no=1)
    assert parse.status == "ok" and parse.calls[0].arguments == {"n": "three"}
    assert parse.reason == "coercion_failed:n"


def test_text_after_the_last_call_fails_whole_text_returned():
    text = fake_tool_call_text("Agent", {"prompt": "a"}) + "\nThat is all."
    parse = parse_tool_calls(text, TOOLS, call_no=1)
    assert parse.status == "failed" and parse.reason == "text_after_tool_call"
    assert parse.calls == () and parse.content == text


@pytest.mark.parametrize("text,reason", [
    ("<tool_call>\n<function=Agent>\n<parameter=prompt>\na\n</parameter>\n</function>", "malformed_tool_call"),
    ("<tool_call>\n<function=Agent>\n<parameter=prompt>\na\n</function>\n</tool_call>", "malformed_parameter"),
    ("<tool_call>\n<function=Agent>\n<parameter=prompt>\na\n</parameter>\n<parameter=prompt>\nb\n</parameter>\n"
     "</function>\n</tool_call>", "duplicate_parameter:prompt"),
    ("<tool_call>\nnot a function\n</tool_call>", "malformed_tool_call"),
    ("Text <tool_call>", "malformed_tool_call"),
    ("<tool_call>\n<function=Agent>\n<parameter=prompt>a</parameter>\n</function>\n</tool_call>",
     "malformed_parameter"),
])
def test_grammar_violations_fail(text, reason):
    parse = parse_tool_calls(text, TOOLS, call_no=1)
    assert parse.status == "failed" and parse.reason == reason and parse.content == text and parse.calls == ()


def test_unknown_function_is_returned_and_flagged():
    parse = parse_tool_calls(fake_tool_call_text("Bash", {"cmd": "ls"}), TOOLS, call_no=1)
    assert parse.status == "ok" and parse.reason == "unknown_function:Bash"
    assert parse.calls == (ToolCall("call_1_0", "Bash", {"cmd": "ls"}),)


def test_call_no_and_text_are_validated():
    with pytest.raises(ValueError):
        parse_tool_calls("x", TOOLS, call_no=0)
    with pytest.raises(TypeError):
        parse_tool_calls(None, TOOLS, call_no=1)


def test_round_trip_through_the_template():
    """The template renders exactly the grammar the parser reads (parallel, multi-line, typed)."""
    calls = (ToolCall("call_3_0", "Agent", {"prompt": "x\ny", "n": 5, "opts": {"a": [1, 2]}}),
             ToolCall("call_3_1", "update_plan", {"plan": ["p1", "p2"]}))
    xml = "\n".join(render_tool_call_xml(c) for c in calls)
    parsed = parse_tool_calls("Before.\n\n" + xml, TOOLS, call_no=3)
    assert parsed.calls == calls and parsed.content == "Before."
    rendered = render_chat([Message("user", "U"), Message("assistant", "Before.", reasoning_content="R",
                                                          tool_calls=calls), Message("tool", "ok")], tools=TOOLS)
    assert "Before.\n\n" + xml + "<|im_end|>" in rendered


def test_raw_string_arguments_are_reemitted_by_the_renderer_not_reparsed():
    raw = "<tool_call>\n<function=Agent>\n<parameter=prompt>\nbroken"
    rendered = render_chat([Message("user", "U"), Message("assistant", "", tool_calls=(ToolCall("c", "Agent", raw),)),
                            Message("user", "U2")], tools=TOOLS)
    assert raw + "<|im_end|>" in rendered
    with pytest.raises(TypeError):
        render_tool_call_xml(ToolCall("c", "Agent", raw))
