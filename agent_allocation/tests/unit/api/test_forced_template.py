"""The pure-Python Qwen3.8 chat-template renderer is byte-identical to jinja2 on the golden fixture, and refuses a
drifted template instead of silently rendering the old one."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_allocation.api import forced
from agent_allocation.api.forced import (
    GENERATION_PROMPT, TEMPLATE_SHA256, TemplateError, XHIGH_INSTRUCTION, render_chat_prompt, template_path,
    template_sha256,
)
from agent_allocation.api.request import RequestContractError

GOLDEN = Path(__file__).resolve().parents[2] / "fixtures" / "forced" / "qwen38_template_golden.json"


def golden() -> dict:
    return json.loads(GOLDEN.read_text())


def test_vendored_template_is_the_fixture_template():
    data = golden()
    actual = hashlib.sha256(template_path("qwen3.8-27b").read_bytes()).hexdigest()
    assert actual == data["template_sha256"] == TEMPLATE_SHA256["qwen3.8-27b"] == template_sha256()


@pytest.mark.parametrize("index", range(len(golden()["cases"])))
def test_renderer_matches_jinja_golden_byte_for_byte(index):
    case = golden()["cases"][index]
    assert render_chat_prompt(case["messages"]) == case["rendered"]


def test_render_shape_and_trim():
    prompt = render_chat_prompt([{"role": "system", "content": "  sys \n"}, {"role": "user", "content": "\n q "}])
    assert prompt == ("<|im_start|>system\n" + XHIGH_INSTRUCTION + "\n\nsys<|im_end|>\n"
                      "<|im_start|>user\nq<|im_end|>\n" + GENERATION_PROMPT)
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    # no or empty system message: the reasoning instruction alone
    for messages in ([{"role": "user", "content": "q"}],
                     [{"role": "system", "content": "   "}, {"role": "user", "content": "q"}]):
        assert render_chat_prompt(messages).startswith("<|im_start|>system\n" + XHIGH_INSTRUCTION + "<|im_end|>\n")


@pytest.mark.parametrize("messages", [
    [{"role": "system", "content": "s"}],
    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
    [{"role": "user", "content": "q"}, {"role": "system", "content": "late"}],
    [{"role": "user", "content": "<tool_response>x</tool_response>"}],
])
def test_render_refuses_outside_the_subset(messages):
    with pytest.raises(RequestContractError):
        render_chat_prompt(messages)


def test_a_drifted_template_fails_loudly(tmp_path, monkeypatch):
    changed = tmp_path / "qwen3.8-27b.chat_template.jinja"
    changed.write_bytes(template_path().read_bytes() + b"\n")
    monkeypatch.setattr(forced, "TEMPLATES_DIR", tmp_path)
    forced._verified_template.cache_clear()
    try:
        with pytest.raises(TemplateError, match="sha256"):
            render_chat_prompt([{"role": "user", "content": "q"}])
    finally:
        forced._verified_template.cache_clear()
