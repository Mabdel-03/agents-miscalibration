"""WP-A, contract §4: the pinned template's XML tool-call grammar -> ``ToolCall`` records.

Grammar over the visible text (after ``normalize.answer_visible_text`` / ``split_reasoning``)::

    Reply     := Content? ToolCall (WS ToolCall)* WS?
    ToolCall  := "<tool_call>" WS "<function=" NAME ">" WS Param* "</function>" WS "</tool_call>"
    Param     := "<parameter=" NAME ">" "\\n" VALUE "\\n</parameter>" WS
    NAME      := [A-Za-z0-9_.\\-]+ ; VALUE := any text not containing "\\n</parameter>"

Matched spans must be contiguous up to trailing whitespace: text after the last call, an unclosed block, a malformed
parameter, a duplicate parameter or a ``<tool_call>`` that does not open a well-formed block make the WHOLE text
the reply content with ``status="failed"`` (``tool_parse_failed`` is a pre-registered Gate 2 metric). Typing comes
from the tool's JSON schema (``parameters.properties[P].type``): ``string`` or absent -> the raw value; anything
else -> ``json.loads`` with a fallback to the raw string plus a ``coercion_failed:<P>`` note. An unknown function
name is still returned (the harness decides) and flagged ``unknown_function:<NAME>``. Ids are
``f"call_{call_no}_{i}"`` (0-based ``i``), the proxy-minted ids the reasoning cache keys on.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agent_allocation.dynorch.records import ToolCall, ToolSpec

TOOL_CALL_OPEN = "<tool_call>"
_NAME = r"[A-Za-z0-9_.\-]+"
_CALL = re.compile(r"<tool_call>\s*<function=(" + _NAME + r")>\s*(.*?)</function>\s*</tool_call>", re.DOTALL)
_PARAM = re.compile(r"<parameter=(" + _NAME + r")>\n(.*?)\n</parameter>\s*", re.DOTALL)
_EMPTY_PARAM = re.compile(r"<parameter=(" + _NAME + r")>\n</parameter>\s*", re.DOTALL)


@dataclass(frozen=True)
class ToolParse:
    content: str
    calls: tuple[ToolCall, ...]
    status: str                      # none | ok | failed
    reason: str | None = None


def tool_call_id(call_no: int, index: int) -> str:
    return f"call_{call_no}_{index}"


def _schema_type(tool: ToolSpec | None, parameter: str) -> Any:
    if tool is None or not isinstance(tool.parameters, Mapping):
        return None
    properties = tool.parameters.get("properties")
    if not isinstance(properties, Mapping):
        return None
    schema = properties.get(parameter)
    if not isinstance(schema, Mapping):
        return None
    return schema.get("type")


def _is_string_type(declared: Any) -> bool:
    if declared is None or declared == "string":
        return True
    if isinstance(declared, list):
        return "string" in declared or not declared
    return False


def _typed(tool: ToolSpec | None, parameter: str, raw: str, notes: list[str]) -> Any:
    declared = _schema_type(tool, parameter)
    if _is_string_type(declared):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        notes.append(f"coercion_failed:{parameter}")
        return raw


def _parse_parameters(body: str, tool: ToolSpec | None, notes: list[str]) -> dict[str, Any] | str:
    """The ``<parameter>`` blocks of one function body, or a failure reason (str)."""
    arguments: dict[str, Any] = {}
    position = 0
    text = body.lstrip()
    while position < len(text):
        match = _PARAM.match(text, position)
        if match is None:
            match = _EMPTY_PARAM.match(text, position)
            if match is None:
                return "malformed_parameter"
            name, raw = match.group(1), ""
        else:
            name, raw = match.group(1), match.group(2)
        if name in arguments:
            return f"duplicate_parameter:{name}"
        arguments[name] = _typed(tool, name, raw, notes)
        position = match.end()
    return arguments


def parse_tool_calls(text: str, tools: Sequence[ToolSpec], *, call_no: int) -> ToolParse:
    """§4: parse every ``<tool_call>`` block of ``text``. ``status="none"`` when the text has no block,
    ``"ok"`` with ``calls`` (and ``content`` = the text before the first block, edge whitespace stripped),
    ``"failed"`` with the whole text as content and the ``reason``. Edge whitespace is trimmed exactly as the
    template trims content on re-render, so the model never sees a difference."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(call_no, bool) or not isinstance(call_no, int) or call_no < 1:
        raise ValueError("call_no must be a positive integer")
    first = text.find(TOOL_CALL_OPEN)
    if first < 0:
        return ToolParse(content=text.strip(), calls=(), status="none", reason=None)
    by_name = {t.name: t for t in tools}
    calls: list[ToolCall] = []
    notes: list[str] = []
    position = first
    while True:
        match = _CALL.match(text, position)
        if match is None:
            return ToolParse(content=text.strip(), calls=(), status="failed", reason="malformed_tool_call")
        name, body = match.group(1), match.group(2)
        tool = by_name.get(name)
        if tool is None:
            notes.append(f"unknown_function:{name}")
        parsed = _parse_parameters(body, tool, notes)
        if isinstance(parsed, str):
            return ToolParse(content=text.strip(), calls=(), status="failed", reason=parsed)
        calls.append(ToolCall(id=tool_call_id(call_no, len(calls)), name=name, arguments=parsed))
        position = match.end()
        rest = text[position:]
        if not rest.strip():
            break
        stripped = rest.lstrip()
        if not stripped.startswith(TOOL_CALL_OPEN):
            return ToolParse(content=text.strip(), calls=(), status="failed", reason="text_after_tool_call")
        position += len(rest) - len(stripped)
    content = text[:first].strip()
    return ToolParse(content=content, calls=tuple(calls), status="ok", reason=";".join(notes) or None)


def render_tool_call_xml(call: ToolCall) -> str:
    """The inverse of the grammar for one call (tests round-trip through it; the template renders the real thing)."""
    if not isinstance(call.arguments, Mapping):
        raise TypeError("only dict arguments render as XML")
    parts = [f"<tool_call>\n<function={call.name}>\n"]
    for key, value in call.arguments.items():
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        parts.append(f"<parameter={key}>\n{rendered}\n</parameter>\n")
    parts.append("</function>\n</tool_call>")
    return "".join(parts)
