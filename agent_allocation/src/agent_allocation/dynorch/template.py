"""WP-A, contract §4: the full Jinja2 renderer of the pinned Qwen3.8 chat template with HF-compatible filters.

``render_chat(messages, *, tools, reasoning_effort, enable_thinking=True, template="qwen3.8-27b")`` renders exactly
what ``transformers.utils.chat_template_utils.render_jinja_template`` would render (jinja2 3.1.6
``ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)``, ``tojson`` = plain ``json.dumps`` with
``ensure_ascii=False`` and NO HTML escaping, ``raise_exception``, ``strftime_now``) over the vendored template
(``api/forced.py::template_path``, sha verified by ``template_sha256`` so a drifted template refuses to render).
On ``[system, user]`` it is byte-identical to ``api/forced.py::render_chat_prompt`` (parity test); on tool /
tool-call / tool-response / empty-reasoning fixtures it equals the transformers oracle goldens
(``tests/fixtures/dynorch/oracle_goldens.json``, generated once by ``tests/unit/dynorch/gen_oracle_goldens.py``).

Message mapping (``records.Message`` -> template dict): ``system``/``user``/``tool`` carry ``content``; ``assistant``
carries ``content``, ``reasoning_content`` (``None`` -> not set, so the template renders an empty think block for a
re-injection miss) and ``tool_calls`` as ``{"type":"function","function":{"name","arguments":<dict>}}``. A
``ToolCall`` whose ``arguments`` is a raw string (not JSON-parseable) is re-emitted verbatim inside the assistant
content instead of a ``<tool_call>`` block (§4). Tools are passed as ``ToolSpec.as_openai()`` objects; the template
``tojson``s them verbatim.

Reasoning effort: the frozen table says the template always gets the study effort (``xhigh``) and the harness value
is recorded and ignored -- ``template_effort`` returns the configured value for any requested value and
``requested_effort`` normalises what the harness sent for the record.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from functools import lru_cache
from typing import Any, Mapping, Sequence

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

from agent_allocation.api.forced import GENERATION_PROMPT, TemplateError, template_path, template_sha256
from agent_allocation.dynorch.records import Message, ToolCall, ToolSpec

TEMPLATE_EFFORTS: tuple[str, ...] = ("xhigh", "medium", "low")
THINK_DISABLED_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
_ROLES = frozenset({"system", "user", "assistant", "tool"})


class RenderError(TemplateError):
    """The messages fall outside what the pinned template renders (a loud 400 for the wire layer)."""


def _raise_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _tojson(x: Any, ensure_ascii: bool = False, indent: int | None = None, separators: Any = None,
            sort_keys: bool = False) -> str:
    # transformers' override of jinja's tojson: plain json.dumps, no HTML escaping of < > & '
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _strftime_now(format: str) -> str:
    return datetime.now().strftime(format)


@lru_cache(maxsize=None)
def _compiled(template: str) -> tuple[str, jinja2.Template]:
    """The compiled template keyed by name; the sha is re-verified on first use (a drift raises ``TemplateError``)."""
    digest = template_sha256(template)                 # raises for a drifted vendored file
    source = template_path(template).read_text(encoding="utf-8")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = _strftime_now
    return digest, env.from_string(source)


def clear_cache() -> None:
    """Forget compiled templates (tests that swap the vendored file)."""
    _compiled.cache_clear()


def template_effort(requested: str | None, *, configured: str = "xhigh") -> str:
    """Frozen table: the template effort is the study's (``xhigh``); the harness value is recorded, never applied."""
    if configured not in TEMPLATE_EFFORTS:
        raise RenderError(f"configured reasoning effort {configured!r} is not one of {TEMPLATE_EFFORTS}")
    return configured


def requested_effort(value: Any) -> str | None:
    """The harness's effort as recorded on the ``CallRecord``: a string (any value) or ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise RenderError(f"reasoning effort must be a string, got {type(value).__name__}")


def _tool_call_dict(call: ToolCall) -> dict[str, Any] | None:
    if isinstance(call.arguments, Mapping):
        return {"id": call.id, "type": "function",
                "function": {"name": call.name, "arguments": dict(call.arguments)}}
    return None


def raw_call_text(call: ToolCall) -> str:
    """How a ``ToolCall`` with raw (unparseable) string arguments is re-emitted: verbatim, inside the content."""
    if not isinstance(call.arguments, str):
        raise RenderError("raw_call_text is only for string arguments")
    return call.arguments


def message_dicts(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Template-facing dicts (also what the oracle golden generator feeds transformers)."""
    out: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Message):
            raise RenderError(f"message {index} is not a records.Message")
        if message.role not in _ROLES:
            raise RenderError(f"message {index} has role {message.role!r}")
        if not isinstance(message.content, str):
            raise RenderError(f"message {index} content must be a string")
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "assistant":
            if message.reasoning_content is not None:
                if not isinstance(message.reasoning_content, str):
                    raise RenderError(f"message {index} reasoning_content must be a string or None")
                item["reasoning_content"] = message.reasoning_content
            calls: list[dict[str, Any]] = []
            raw_parts: list[str] = []
            for call in message.tool_calls:
                rendered = _tool_call_dict(call)
                if rendered is None:
                    raw_parts.append(raw_call_text(call))
                else:
                    calls.append(rendered)
            if raw_parts:
                content = item["content"]
                item["content"] = "\n\n".join(([content.rstrip()] if content.strip() else []) + raw_parts)
            if calls:
                item["tool_calls"] = calls
        elif message.tool_calls:
            raise RenderError(f"message {index} ({message.role}) cannot carry tool_calls")
        if message.role == "tool" and message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        out.append(item)
    return out


def tool_dicts(tools: Sequence[ToolSpec]) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [t.as_openai() if isinstance(t, ToolSpec) else _check_tool_dict(t) for t in tools]


def _check_tool_dict(tool: Any) -> dict[str, Any]:
    if not (isinstance(tool, Mapping) and tool.get("type") == "function" and isinstance(tool.get("function"), Mapping)
            and isinstance(tool["function"].get("name"), str)):
        raise RenderError("tools must be ToolSpec or {type: function, function: {name, ...}} objects")
    return dict(tool)


def render_chat(messages: Sequence[Message], *, tools: Sequence[ToolSpec] = (), reasoning_effort: str = "xhigh",
                enable_thinking: bool = True, template: str = "qwen3.8-27b") -> str:
    """§4: the prompt for one upstream call, ``add_generation_prompt=True``. With thinking on it ends in
    ``api/forced.py::GENERATION_PROMPT`` (``<|im_start|>assistant\\n<think>\\n``); with ``enable_thinking=False``
    (the B4 terminal call) in an empty, closed think block. Template errors (no user turn, a late system message,
    an unexpected role) raise ``RenderError``."""
    if reasoning_effort not in TEMPLATE_EFFORTS:
        raise RenderError(f"reasoning_effort {reasoning_effort!r} is not one of {TEMPLATE_EFFORTS}")
    _, compiled = _compiled(template)
    kwargs: dict[str, Any] = {"messages": message_dicts(messages), "tools": tool_dicts(tools),
                              "add_generation_prompt": True, "reasoning_effort": reasoning_effort}
    if not enable_thinking:
        kwargs["enable_thinking"] = False
    try:
        rendered = compiled.render(**kwargs)
    except jinja2.exceptions.TemplateError as exc:
        raise RenderError(f"template refused the messages: {exc}") from exc
    expected = THINK_DISABLED_SUFFIX if not enable_thinking else GENERATION_PROMPT
    if not rendered.endswith(expected):
        raise RenderError("rendered prompt does not end in the generation prompt")
    return rendered


def rendered_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def system_sha256(text: str) -> str:
    """The conversation classifier's key (plan B1): sha256 of the system text as received."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
