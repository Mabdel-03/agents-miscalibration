"""Structural anchors: byte-exact token mapping and the anchor resolution rules (N1).

Spec §8.4: "Capture native within-call states at the last native prefill token, processed
generated tokens 32, 128, and 512, and the parsed final-object closing token when present.
Count all model-emitted native channels ..., including thinking tokens ... Capture a vector
when its token has been processed, not the preceding token's prediction vector. ... Missing
counts are ``NOT_REACHED``; unsupported channels are ``CHANNEL_UNAVAILABLE``; invalid final
JSON is ``JSON_CLOSE_MISSING``; a native call whose declared contract does not require a
structured final object has ``FINAL_OBJECT_NOT_APPLICABLE``. ... Resolve ... spans through
the serializer parser and tokenizer offsets, never by searching for a repeated sentinel
string."  Brief R1/R2 (docs/study_v4/09_neural_readouts_brief.md).

Positions are indices into the teacher-forced sequence ``prompt_token_ids + token_ids``:

* ``NATIVE_PREFILL`` = ``prompt_len - 1`` (the last prompt token, after it was processed);
* ``GENERATED_k`` = ``prompt_len + k - 1`` when ``k <= completion_len`` else ``NOT_REACHED``
  (the k-th generated token counted over *all* native channels: the ``<think>`` span is
  part of the stored completion ids and counts);
* ``FINAL_OBJECT_CLOSE`` = the completion token whose bytes contain the closing brace of the
  parsed final object.  The content channel is located structurally (bytes after the
  ``</think>`` token, exact identity with the stored ``content`` asserted), the object span
  is derived with the same one-fence-strip + strict-parse rule as
  ``study/parse/candidate.py`` (the frozen §3.5 salvage), and the brace's UTF-8 byte offset
  is mapped to a token through the tokenizer's per-token byte spans.
* Report readouts: ``STATE_ANCHOR`` (last token of the compiled report span, i.e. the last
  prefill token before generation when the render ends there) and ``TASK_ONLY_ANCHOR`` (last
  token of the task span), both supplied by the report compiler as exclusive-end UTF-8 byte
  offsets into the single user message and resolved to prompt-token indices; ``LAST_PREFILL``
  (``len(prompt_token_ids) - 1``) is recorded separately (§8.4 "The final anchor is not
  necessarily the last chat-template token; record both locations separately").

Token → bytes: Qwen's tokenizer is a GPT-2-style byte-level BPE, so every regular token
string maps back to raw bytes through the fixed ``bytes_to_unicode`` alphabet and added
(special) tokens contribute their literal text.  The concatenation is asserted to equal
``tokenizer.decode(ids)`` (exact identity); tokenizers without that alphabet fall back to
incremental prefix decoding (correct but O(n²); tests only).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agents_scaling.study.parse.candidate import (
    FINISH_LENGTH,
    load_strict_json,
    parse_candidate,
    strip_one_fence,
    validate_candidate_object,
)

try:  # the frozen E3' fence regexes (private names; see json_object_span for the fallback)
    from agents_scaling.study.parse.candidate import _CLOSE_FENCE, _OPEN_FENCE
except ImportError:  # pragma: no cover
    _OPEN_FENCE = _CLOSE_FENCE = None  # type: ignore[assignment]

# --------------------------------------------------------------------------- constants

ANCHOR_NATIVE_PREFILL = "NATIVE_PREFILL"
ANCHOR_FINAL_OBJECT_CLOSE = "FINAL_OBJECT_CLOSE"
ANCHOR_STATE = "STATE_ANCHOR"
ANCHOR_TASK_ONLY = "TASK_ONLY_ANCHOR"
ANCHOR_LAST_PREFILL = "LAST_PREFILL"
GENERATED_KS: tuple[int, ...] = (32, 128, 512)

MISSING_NOT_REACHED = "NOT_REACHED"
MISSING_CHANNEL_UNAVAILABLE = "CHANNEL_UNAVAILABLE"
MISSING_JSON_CLOSE = "JSON_CLOSE_MISSING"
MISSING_NOT_APPLICABLE = "FINAL_OBJECT_NOT_APPLICABLE"
MISSING_ANCHOR_UNRESOLVED = "ANCHOR_UNRESOLVED"
MISSINGNESS_CODES: tuple[str, ...] = (
    MISSING_NOT_REACHED,
    MISSING_CHANNEL_UNAVAILABLE,
    MISSING_JSON_CLOSE,
    MISSING_NOT_APPLICABLE,
    MISSING_ANCHOR_UNRESOLVED,
)

CHANNEL_PROMPT = "prompt"
CHANNEL_THINKING = "thinking"
CHANNEL_CONTENT = "content"
CHANNEL_NONE = "none"

#: Which structured final object a call's declared contract requires (keyed by the
#: ``episode.calls[].role`` value).  ``candidate`` = the §3.5 six-key object (roots and
#: revisions); ``coordinator`` = the hub's ``{"action": ...}`` object; ``subtask`` = a worker's
#: subtask result, which is not a full-task final object → FINAL_OBJECT_NOT_APPLICABLE.
FINAL_OBJECT_CANDIDATE = "candidate"
FINAL_OBJECT_COORDINATOR = "coordinator"
FINAL_OBJECT_SUBTASK = "subtask"
FINAL_OBJECT_BY_ROLE: Mapping[str, str] = {
    "root": FINAL_OBJECT_CANDIDATE,
    "revise": FINAL_OBJECT_CANDIDATE,
    "hub": FINAL_OBJECT_COORDINATOR,
    "worker": FINAL_OBJECT_SUBTASK,
}

THINK_END_TOKEN = "</think>"
THINK_START_TOKEN = "<think>"


class AnchorError(ValueError):
    """The stored record/report cannot be mapped onto the tokenizer (never silently guessed)."""


def generated_anchor_kind(k: int) -> str:
    return f"GENERATED_{int(k)}"


# --------------------------------------------------------------------------- anchors


@dataclass(frozen=True)
class Anchor:
    """One capture site of a sequence.  ``token_offset`` is the absolute index into the
    teacher-forced sequence (``None`` when missing; then ``missingness`` names why).
    ``generated_token_count`` = generated tokens processed at the anchor (0 for prefill)."""

    kind: str
    token_offset: int | None
    missingness: str | None
    channel: str
    structural_span: str
    generated_token_count: int
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.token_offset is None) != (self.missingness is not None):
            raise ValueError("Anchor: token_offset is None iff missingness is set")
        if self.missingness is not None and self.missingness not in MISSINGNESS_CODES:
            raise ValueError(f"unknown missingness {self.missingness!r}")

    @property
    def present(self) -> bool:
        return self.token_offset is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "token_offset": self.token_offset,
            "missingness": self.missingness,
            "channel": self.channel,
            "structural_span": self.structural_span,
            "generated_token_count": self.generated_token_count,
            "detail": dict(self.detail),
        }


# --------------------------------------------------------------------------- token bytes


def _bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte-level alphabet (identical in Qwen2/Qwen3 tokenizers)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


_B2U = _bytes_to_unicode()
_U2B: dict[str, int] = {v: k for k, v in _B2U.items()}


def _added_tokens(tokenizer: Any) -> Mapping[int, Any]:
    added = getattr(tokenizer, "added_tokens_decoder", None)
    return added if isinstance(added, Mapping) else {}


def _byte_level_token_bytes(tokenizer: Any, token_ids: Sequence[int]) -> list[bytes] | None:
    added = _added_tokens(tokenizer)
    try:
        tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
    except Exception:  # noqa: BLE001 - not a byte-level tokenizer
        return None
    out: list[bytes] = []
    for tid, tok in zip(token_ids, tokens):
        if tid in added:
            out.append(str(added[tid]).encode("utf-8"))
            continue
        if not isinstance(tok, str):
            return None
        try:
            out.append(bytes(_U2B[c] for c in tok))
        except KeyError:
            return None
    return out


def _incremental_token_bytes(tokenizer: Any, token_ids: Sequence[int]) -> list[bytes]:
    """Fallback for non-byte-level tokenizers: bytes of token i = decode(ids[:i+1]) minus
    decode(ids[:i]) (a prefix relation the stub tokenizers satisfy)."""
    out: list[bytes] = []
    previous = b""
    ids = list(token_ids)
    for i in range(len(ids)):
        text = tokenizer.decode(ids[: i + 1]).encode("utf-8")
        if not text.startswith(previous):
            raise AnchorError("tokenizer decode is not prefix-monotone; cannot derive token bytes")
        out.append(text[len(previous):])
        previous = text
    return out


def token_bytes(tokenizer: Any, token_ids: Sequence[int]) -> list[bytes]:
    """Exact UTF-8 bytes contributed by every token; ``b"".join(...)`` == ``decode(ids)``."""
    ids = [int(t) for t in token_ids]
    if not ids:
        return []
    decoded = tokenizer.decode(ids)
    if not isinstance(decoded, str):
        raise AnchorError("tokenizer.decode did not return str")
    pieces = _byte_level_token_bytes(tokenizer, ids)
    if pieces is not None and b"".join(pieces) == decoded.encode("utf-8"):
        return pieces
    pieces = _incremental_token_bytes(tokenizer, ids)
    if b"".join(pieces) != decoded.encode("utf-8"):
        raise AnchorError("per-token bytes do not reproduce tokenizer.decode(ids) (exact identity)")
    return pieces


def byte_spans(pieces: Sequence[bytes]) -> list[tuple[int, int]]:
    """Cumulative ``[start, end)`` byte spans of ``token_bytes`` output."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for piece in pieces:
        spans.append((offset, offset + len(piece)))
        offset += len(piece)
    return spans


def token_at_byte(spans: Sequence[tuple[int, int]], byte_offset: int) -> int:
    """Index of the token whose span contains ``byte_offset`` (zero-width tokens are skipped)."""
    if not spans:
        raise AnchorError("no tokens")
    total = spans[-1][1]
    if byte_offset < 0 or byte_offset >= total:
        raise AnchorError(f"byte offset {byte_offset} outside [0, {total})")
    lo, hi = 0, len(spans) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if spans[mid][1] <= byte_offset:
            lo = mid + 1
        else:
            hi = mid
    start, end = spans[lo]
    if not (start <= byte_offset < end):  # pragma: no cover - defensive
        raise AnchorError(f"byte offset {byte_offset} not inside token {lo} span {spans[lo]}")
    return lo


def last_token_of_span(spans: Sequence[tuple[int, int]], byte_end: int) -> int:
    """The token containing the last byte of a span that ends (exclusively) at ``byte_end``."""
    if byte_end <= 0:
        raise AnchorError("a span ending at byte 0 has no last token")
    return token_at_byte(spans, byte_end - 1)


# --------------------------------------------------------------------------- channels


def think_end_id(tokenizer: Any) -> int | None:
    try:
        tid = tokenizer.convert_tokens_to_ids(THINK_END_TOKEN)
    except Exception:  # noqa: BLE001
        return None
    if tid is None or not isinstance(tid, int):
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and tid == unk:
        return None
    return tid


@dataclass(frozen=True)
class ContentChannel:
    """Where the content channel lives inside the completion bytes (``None`` when absent)."""

    think_end_index: int | None
    start_byte: int | None
    end_byte: int | None
    status: str  # "exact" | "suffix" | "substring" | "absent" | "mismatch" | "no_content"
    detail: str = ""

    def channel_of(self, completion_index: int, enable_thinking: bool) -> str:
        if self.think_end_index is not None:
            return CHANNEL_THINKING if completion_index <= self.think_end_index else CHANNEL_CONTENT
        return CHANNEL_THINKING if enable_thinking else CHANNEL_CONTENT


def locate_content_channel(
    tokenizer: Any,
    completion_ids: Sequence[int],
    stored_content: str | None,
    enable_thinking: bool,
    *,
    spans: Sequence[tuple[int, int]] | None = None,
    pieces: Sequence[bytes] | None = None,
) -> ContentChannel:
    """Structural location of the content channel: bytes after the ``</think>`` token (or the
    whole completion when thinking was off), minus trailing special tokens; exact identity
    with the stored ``content`` is checked (a leading-whitespace difference is tolerated as a
    ``suffix`` alignment and recorded)."""
    ids = [int(t) for t in completion_ids]
    if pieces is None:
        pieces = token_bytes(tokenizer, ids)
    if spans is None:
        spans = byte_spans(pieces)
    added = _added_tokens(tokenizer)
    te = think_end_id(tokenizer)
    think_index = ids.index(te) if (te is not None and te in ids) else None
    if think_index is not None:
        start = spans[think_index][1]
    elif enable_thinking:
        return ContentChannel(None, None, None, "absent", "thinking enabled and </think> never emitted")
    else:
        start = 0
    end = spans[-1][1] if spans else 0
    last = len(ids) - 1
    while last >= 0 and ids[last] in added and (think_index is None or last > think_index):
        end = spans[last][0]
        last -= 1
    if stored_content is None:
        return ContentChannel(think_index, None, None, "no_content", "stored content is null")
    decoded = b"".join(pieces)[start:end]
    stored = stored_content.encode("utf-8")
    if decoded == stored:
        return ContentChannel(think_index, start, end, "exact")
    if stored and decoded.endswith(stored):
        return ContentChannel(think_index, end - len(stored), end, "suffix", f"{len(decoded) - len(stored)} leading bytes dropped by the reasoning parser")
    if stored and decoded.count(stored) == 1:
        at = decoded.find(stored)
        return ContentChannel(think_index, start + at, start + at + len(stored), "substring", "content is a unique substring of the channel bytes")
    return ContentChannel(think_index, None, None, "mismatch", f"decoded channel ({len(decoded)} B) != stored content ({len(stored)} B)")


# --------------------------------------------------------------------------- final object


def json_object_span(content: str) -> tuple[int, int]:
    """Code-point span ``[start, end)`` of ``strip_one_fence(content)`` inside ``content``.

    Mirrors the frozen rule step by step (outer strip, one fence, inner strip) and asserts
    the result equals ``strip_one_fence(content)``; when the private regexes are not
    importable the span is located by unique substring search (asserted unique).
    """
    if not isinstance(content, str):
        raise TypeError("json_object_span expects str")
    stripped_text = strip_one_fence(content)
    s_start = len(content) - len(content.lstrip())
    s_end = len(content.rstrip())
    span: tuple[int, int] | None = None
    if _OPEN_FENCE is not None and _CLOSE_FENCE is not None:
        stripped = content[s_start:s_end]
        opening = _OPEN_FENCE.match(stripped)
        if opening is None:
            span = (s_start, s_end)
        else:
            rest = stripped[opening.end():]
            closing = _CLOSE_FENCE.search(rest)
            if closing is None:
                span = (s_start, s_end)
            else:
                rest_start = s_start + opening.end()
                inner = rest[: closing.start()]
                i_start = rest_start + (len(inner) - len(inner.lstrip()))
                i_end = rest_start + len(inner.rstrip())
                span = (i_start, i_end)
    if span is None or content[span[0]:span[1]] != stripped_text:
        if content.count(stripped_text) != 1:
            raise AnchorError("cannot locate the fence-stripped object span uniquely")
        at = content.find(stripped_text)
        span = (at, at + len(stripped_text))
    if content[span[0]:span[1]] != stripped_text:  # pragma: no cover - defensive
        raise AnchorError("object span disagrees with strip_one_fence")
    return span


def final_object_valid(content: str | None, finish_reason: str, reasoning: str | None, final_object: str) -> tuple[bool, str]:
    """``(valid, detail)`` for the declared final object of a role (outcome-blind: only the
    §3.5 / coordinator-action structural rules, never correctness)."""
    if final_object == FINAL_OBJECT_CANDIDATE:
        parsed = parse_candidate(content, finish_reason, reasoning)
        return parsed.valid, (parsed.failure_code or "valid")
    if final_object == FINAL_OBJECT_COORDINATOR:
        if finish_reason == FINISH_LENGTH:
            return False, "TRUNCATED"
        if content is None:
            return False, "EMPTY"
        try:
            value = load_strict_json(strip_one_fence(content))
        except ValueError as exc:
            code = getattr(exc, "code", "NOT_JSON")
            return False, str(code)
        if not isinstance(value, dict) or value.get("action") not in ("final", "delegate"):
            return False, "SCHEMA"
        if value["action"] == "final":
            if set(value) != {"action", "candidate"}:
                return False, "SCHEMA"
            try:
                validate_candidate_object(value["candidate"])
            except ValueError as exc:
                return False, str(getattr(exc, "code", "SCHEMA"))
        elif set(value) != {"action", "assignments"} or not isinstance(value["assignments"], list) or not value["assignments"]:
            return False, "SCHEMA"
        return True, "valid:" + str(value["action"])
    raise AnchorError(f"unknown final object kind {final_object!r}")


def final_object_close(
    tokenizer: Any,
    completion_ids: Sequence[int],
    response: Mapping[str, Any],
    enable_thinking: bool,
    final_object: str,
    *,
    prompt_len: int,
    spans: Sequence[tuple[int, int]] | None = None,
    pieces: Sequence[bytes] | None = None,
) -> Anchor:
    """The ``FINAL_OBJECT_CLOSE`` anchor of one native call (see the module docstring)."""
    kind = ANCHOR_FINAL_OBJECT_CLOSE
    if final_object == FINAL_OBJECT_SUBTASK:
        return Anchor(kind, None, MISSING_NOT_APPLICABLE, CHANNEL_NONE, "final_object", 0, {"final_object": final_object})
    if final_object not in (FINAL_OBJECT_CANDIDATE, FINAL_OBJECT_COORDINATOR):
        raise AnchorError(f"unknown final object kind {final_object!r}")
    content = response.get("content")
    finish_reason = response.get("finish_reason", "stop")
    reasoning = response.get("reasoning")
    valid, why = final_object_valid(content, finish_reason, reasoning, final_object)
    if not valid:
        return Anchor(kind, None, MISSING_JSON_CLOSE, CHANNEL_NONE, "final_object", 0, {"final_object": final_object, "parse": why})
    ids = [int(t) for t in completion_ids]
    if pieces is None:
        pieces = token_bytes(tokenizer, ids)
    if spans is None:
        spans = byte_spans(pieces)
    channel = locate_content_channel(tokenizer, ids, content, enable_thinking, spans=spans, pieces=pieces)
    if channel.start_byte is None:
        return Anchor(kind, None, MISSING_CHANNEL_UNAVAILABLE, CHANNEL_NONE, "final_object", 0,
                      {"final_object": final_object, "content_channel": channel.status, "detail": channel.detail})
    assert content is not None
    start_char, end_char = json_object_span(content)
    brace_char = end_char - 1
    if brace_char < 0 or content[brace_char] != "}":
        return Anchor(kind, None, MISSING_JSON_CLOSE, CHANNEL_NONE, "final_object", 0,
                      {"final_object": final_object, "parse": "object does not end with a closing brace"})
    brace_byte = channel.start_byte + len(content[:brace_char].encode("utf-8"))
    obj_start_byte = channel.start_byte + len(content[:start_char].encode("utf-8"))
    if not (channel.start_byte <= brace_byte < channel.end_byte):
        raise AnchorError("closing brace byte lies outside the content channel")
    j = token_at_byte(spans, brace_byte)
    return Anchor(
        kind,
        prompt_len + j,
        None,
        CHANNEL_CONTENT,
        f"final_object[completion_bytes {obj_start_byte}:{brace_byte + 1})",
        j + 1,
        {
            "final_object": final_object,
            "parse": why,
            "completion_index": j,
            "brace_byte": brace_byte,
            "content_channel": channel.status,
            "content_start_byte": channel.start_byte,
            "content_end_byte": channel.end_byte,
            "token_bytes": pieces[j].decode("utf-8", "replace"),
        },
    )


# --------------------------------------------------------------------------- native calls


def resolve_native_anchors(
    record: Any,
    tokenizer: Any,
    *,
    final_object: str = FINAL_OBJECT_CANDIDATE,
    ks: Sequence[int] = GENERATED_KS,
) -> list[Anchor]:
    """The five R2 anchors of one ``RequestRecord`` (or a mapping with the same fields)."""
    if isinstance(record, Mapping):
        prompt_ids = list(record["prompt_token_ids"])
        response = record["response"]
        ctk = record.get("chat_template_kwargs") or {}
    else:
        prompt_ids = list(record.prompt_token_ids)
        response = record.response
        ctk = record.chat_template_kwargs or {}
    enable_thinking = bool(ctk.get("enable_thinking", False))
    completion = [int(t) for t in response.get("token_ids") or []]
    prompt_len = len(prompt_ids)
    n = len(completion)
    if prompt_len == 0:
        raise AnchorError("record has no prompt tokens")
    pieces = token_bytes(tokenizer, completion) if completion else []
    spans = byte_spans(pieces)
    channel = locate_content_channel(tokenizer, completion, response.get("content"), enable_thinking, spans=spans, pieces=pieces) if completion else None
    anchors: list[Anchor] = [
        Anchor(ANCHOR_NATIVE_PREFILL, prompt_len - 1, None, CHANNEL_PROMPT, f"prompt[0:{prompt_len})", 0,
               {"prompt_len": prompt_len, "completion_len": n})
    ]
    for k in ks:
        k = int(k)
        kind = generated_anchor_kind(k)
        if k < 1:
            raise AnchorError("generated anchor k must be >= 1")
        if k <= n:
            j = k - 1
            ch = channel.channel_of(j, enable_thinking) if channel is not None else CHANNEL_CONTENT
            anchors.append(Anchor(kind, prompt_len + j, None, ch, f"completion[{j}]", k,
                                  {"completion_index": j, "token_bytes": pieces[j].decode("utf-8", "replace")}))
        else:
            anchors.append(Anchor(kind, None, MISSING_NOT_REACHED, CHANNEL_NONE, f"completion[{k - 1}]", 0,
                                  {"completion_len": n, "finish_reason": response.get("finish_reason")}))
    anchors.append(
        final_object_close(tokenizer, completion, response, enable_thinking, final_object, prompt_len=prompt_len, spans=spans, pieces=pieces)
        if completion
        else Anchor(ANCHOR_FINAL_OBJECT_CLOSE, None, MISSING_NOT_APPLICABLE if final_object == FINAL_OBJECT_SUBTASK else MISSING_JSON_CLOSE,
                    CHANNEL_NONE, "final_object", 0, {"final_object": final_object, "parse": "empty completion"})
    )
    return anchors


def k_max_needed(anchors: Sequence[Anchor]) -> int:
    """Completion tokens the teacher-forced replay must include (0 = prompt only)."""
    return max((a.generated_token_count for a in anchors if a.present), default=0)


# --------------------------------------------------------------------------- report prompts


def render_chat_text(tokenizer: Any, messages: Sequence[Mapping[str, Any]], enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)


def content_prefix_bytes(tokenizer: Any, messages: Sequence[Mapping[str, Any]], enable_thinking: bool) -> int:
    """UTF-8 byte length of the chat-template text that precedes the user message content.

    Structural (no sentinel search): the template is linear in the content, so rendering
    two probe contents that differ in their first character puts the first differing byte
    exactly at the prefix length.
    """
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise AnchorError("anchor mapping expects exactly one user message")
    probes = []
    for probe in ("A", "B"):
        text = render_chat_text(tokenizer, [{"role": "user", "content": probe}], enable_thinking)
        probes.append(text.encode("utf-8"))
    a, b = probes
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    raise AnchorError("probe renders do not differ; template is not linear in the content")


def content_byte_to_prompt_token(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    prompt_token_ids: Sequence[int],
    content_byte_end: int,
    enable_thinking: bool,
    *,
    spans: Sequence[tuple[int, int]] | None = None,
) -> int:
    """Map an exclusive-end byte offset inside the user content to the prompt token that
    contains the span's last byte.  Asserts the exact identity ``decode(prompt_token_ids)
    == apply_chat_template(messages)``."""
    ids = [int(t) for t in prompt_token_ids]
    pieces = token_bytes(tokenizer, ids)
    if spans is None:
        spans = byte_spans(pieces)
    rendered = render_chat_text(tokenizer, messages, enable_thinking).encode("utf-8")
    joined = b"".join(pieces)
    if joined != rendered:
        raise AnchorError("prompt_token_ids do not decode to the chat-template render (exact identity failed)")
    content = messages[0]["content"].encode("utf-8")
    if not (0 < content_byte_end <= len(content)):
        raise AnchorError(f"content byte offset {content_byte_end} outside (0, {len(content)}]")
    prefix = content_prefix_bytes(tokenizer, messages, enable_thinking)
    if joined[prefix:prefix + len(content)] != content:
        raise AnchorError("user content is not at the structural prefix offset of the render")
    return last_token_of_span(spans, prefix + content_byte_end)


REPORT_REQUIRED_FIELDS: tuple[str, ...] = ("report_id", "messages", "prompt_token_ids")
REPORT_ANCHOR_NAMES: tuple[str, str] = ("task_only_anchor", "state_anchor")


def report_anchor_fields(report: Mapping[str, Any], name: str) -> tuple[int | None, int | None]:
    """``(token_index, byte_offset)`` of one report anchor, accepting both interface shapes.

    N2 (``forecast.manifest.report_render``) writes ``byte_anchors[name]`` and
    ``anchor_tokens[name]["index"]`` (and, since the P0-1 fix, the flat ``<name>_byte`` /
    ``<name>_token`` mirrors); older hand-written reports carry only the flat keys.  A flat
    key wins when both are present.  Returns ``None`` for an absent form.
    """
    if name not in REPORT_ANCHOR_NAMES:
        raise AnchorError(f"unknown report anchor {name!r}")
    tok = report.get(f"{name}_token")
    if tok is None:
        nested = report.get("anchor_tokens") or {}
        entry = nested.get(name) if isinstance(nested, Mapping) else None
        if isinstance(entry, Mapping):
            tok = entry.get("index")
        elif entry is not None:
            tok = entry
    byte = report.get(f"{name}_byte")
    if byte is None:
        nested = report.get("byte_anchors") or {}
        byte = nested.get(name) if isinstance(nested, Mapping) else None
    return (None if tok is None else int(tok)), (None if byte is None else int(byte))


def resolve_report_anchors(report: Mapping[str, Any], tokenizer: Any) -> list[Anchor]:
    """``TASK_ONLY_ANCHOR``, ``STATE_ANCHOR`` and ``LAST_PREFILL`` of one compiled report.

    ``report`` is the N2 interface object (``<run_root>/forecast/reports/<item>.<method>.json``,
    ``forecast.manifest.report_render``): ``report_id``, ``messages`` (one user message),
    ``prompt_token_ids``, optional ``chat_template_kwargs`` (default thinking off, as
    FORECAST_DECODING), and the anchors in either (or both) of two equivalent shapes:

    * nested, as N2 writes them: ``byte_anchors: {task_only_anchor, state_anchor}`` (exclusive-end
      UTF-8 byte offsets into the user content) and ``anchor_tokens: {task_only_anchor: {index,
      ...}, state_anchor: {index, ...}}`` (absolute prompt-token indices);
    * flat: ``task_only_anchor_byte`` / ``state_anchor_byte`` and/or ``task_only_anchor_token`` /
      ``state_anchor_token``.

    The flat key wins when both are present for the same anchor; when a byte and a token form
    are both present they must agree (byte → "token containing the span's last byte").  A
    missing state anchor defaults to the last prefill token (``source=last_prefill_default``);
    a missing task-only anchor is ``ANCHOR_UNRESOLVED``.  The capture stage refuses both
    defaults (``capture.check_report_anchors``): the interface is required, not optional.
    """
    missing = [k for k in REPORT_REQUIRED_FIELDS if k not in report]
    if missing:
        raise AnchorError(f"report lacks {missing}")
    ids = [int(t) for t in report["prompt_token_ids"]]
    if not ids:
        raise AnchorError("report has no prompt tokens")
    messages = list(report["messages"])
    ctk = report.get("chat_template_kwargs") or {"enable_thinking": False}
    enable_thinking = bool(ctk.get("enable_thinking", False))
    n = len(ids)
    spans: list[tuple[int, int]] | None = None

    def resolve(name: str) -> tuple[int | None, str]:
        nonlocal spans
        tok, byte = report_anchor_fields(report, name)
        from_byte: int | None = None
        if byte is not None:
            if spans is None:
                spans = byte_spans(token_bytes(tokenizer, ids))
            from_byte = content_byte_to_prompt_token(tokenizer, messages, ids, int(byte), enable_thinking, spans=spans)
        if tok is not None:
            tok = int(tok)
            if not (0 <= tok < n):
                raise AnchorError(f"{name}_token {tok} outside [0, {n})")
            if from_byte is not None and from_byte != tok:
                raise AnchorError(f"{name}: byte-derived token {from_byte} != supplied token {tok}")
            return tok, "token" if from_byte is None else "token+byte"
        if from_byte is not None:
            return from_byte, "byte"
        return None, "absent"

    task_tok, task_src = resolve("task_only_anchor")
    state_tok, state_src = resolve("state_anchor")
    anchors: list[Anchor] = []
    if task_tok is None:
        anchors.append(Anchor(ANCHOR_TASK_ONLY, None, MISSING_ANCHOR_UNRESOLVED, CHANNEL_NONE, "task", 0, {"source": task_src}))
    else:
        anchors.append(Anchor(ANCHOR_TASK_ONLY, task_tok, None, CHANNEL_PROMPT, f"task[..{task_tok}]", 0,
                              {"source": task_src, "task_only_anchor_byte": report_anchor_fields(report, "task_only_anchor")[1]}))
    if state_tok is None:
        state_tok, state_src = n - 1, "last_prefill_default"
    if task_tok is not None and task_tok > state_tok:
        raise AnchorError("task-only anchor lies after the state anchor")
    anchors.append(Anchor(ANCHOR_STATE, state_tok, None, CHANNEL_PROMPT, f"report[..{state_tok}]", 0,
                          {"source": state_src, "state_anchor_byte": report_anchor_fields(report, "state_anchor")[1]}))
    anchors.append(Anchor(ANCHOR_LAST_PREFILL, n - 1, None, CHANNEL_PROMPT, f"prompt[0:{n})", 0,
                          {"equals_state_anchor": state_tok == n - 1}))
    return anchors


def anchors_json(anchors: Sequence[Anchor]) -> str:
    return json.dumps([a.to_dict() for a in anchors], ensure_ascii=False, sort_keys=True)


_ROLE_RE = re.compile(r"^[a-z_]+$")


def final_object_for_role(role: str) -> str:
    """``FINAL_OBJECT_BY_ROLE`` lookup (fail closed on an unknown role)."""
    if not isinstance(role, str) or not _ROLE_RE.match(role) or role not in FINAL_OBJECT_BY_ROLE:
        raise AnchorError(f"no final-object contract known for role {role!r}")
    return FINAL_OBJECT_BY_ROLE[role]


__all__ = [
    "ANCHOR_FINAL_OBJECT_CLOSE",
    "ANCHOR_LAST_PREFILL",
    "ANCHOR_NATIVE_PREFILL",
    "ANCHOR_STATE",
    "ANCHOR_TASK_ONLY",
    "CHANNEL_CONTENT",
    "CHANNEL_NONE",
    "CHANNEL_PROMPT",
    "CHANNEL_THINKING",
    "FINAL_OBJECT_BY_ROLE",
    "FINAL_OBJECT_CANDIDATE",
    "FINAL_OBJECT_COORDINATOR",
    "FINAL_OBJECT_SUBTASK",
    "GENERATED_KS",
    "MISSINGNESS_CODES",
    "MISSING_ANCHOR_UNRESOLVED",
    "MISSING_CHANNEL_UNAVAILABLE",
    "MISSING_JSON_CLOSE",
    "MISSING_NOT_APPLICABLE",
    "MISSING_NOT_REACHED",
    "REPORT_ANCHOR_NAMES",
    "REPORT_REQUIRED_FIELDS",
    "Anchor",
    "AnchorError",
    "ContentChannel",
    "anchors_json",
    "byte_spans",
    "content_byte_to_prompt_token",
    "content_prefix_bytes",
    "final_object_close",
    "final_object_for_role",
    "final_object_valid",
    "generated_anchor_kind",
    "json_object_span",
    "k_max_needed",
    "last_token_of_span",
    "locate_content_channel",
    "render_chat_text",
    "report_anchor_fields",
    "resolve_native_anchors",
    "resolve_report_anchors",
    "think_end_id",
    "token_at_byte",
    "token_bytes",
]
