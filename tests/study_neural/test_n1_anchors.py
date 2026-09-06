"""N1 anchors: byte-exact token mapping, the five native anchors, JSON close with fences
and CJK, channel handling, and the report anchors (real pinned Qwen3 tokenizer; CPU)."""

from __future__ import annotations

import pytest

from agents_scaling.study.neural import anchors as A
from tests.study_neural import support

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def tok():
    try:
        return support.qwen_tokenizer("8B")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen3-8B tokenizer unavailable offline: {exc!r}")


def _prompt(tok, content: str, enable_thinking: bool = True):
    messages = [{"role": "user", "content": content}]
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking, return_dict=False)
    return messages, list(ids)


# --------------------------------------------------------------------------- token bytes


def test_token_bytes_reproduce_decode_including_cjk_and_specials(tok):
    messages, ids = _prompt(tok, "你好世界 — naïve {\"a\":1}\n```json\n{\"x\": \"é\"}\n```")
    pieces = A.token_bytes(tok, ids)
    assert b"".join(pieces).decode("utf-8") == tok.decode(ids)
    spans = A.byte_spans(pieces)
    assert spans[0] == (0, len("<|im_start|>".encode()))
    assert spans[-1][1] == len(tok.decode(ids).encode("utf-8"))
    # every byte of a multi-byte CJK character maps to exactly one token
    text = b"".join(pieces)
    at = text.find("世界".encode("utf-8"))
    assert A.token_at_byte(spans, at) == A.token_at_byte(spans, at + 2)


def test_token_at_byte_bounds_and_last_token_of_span():
    spans = [(0, 3), (3, 3), (3, 7), (7, 8)]
    assert A.token_at_byte(spans, 0) == 0
    assert A.token_at_byte(spans, 2) == 0
    assert A.token_at_byte(spans, 3) == 2  # the zero-width token is skipped
    assert A.token_at_byte(spans, 7) == 3
    assert A.last_token_of_span(spans, 7) == 2
    with pytest.raises(A.AnchorError):
        A.token_at_byte(spans, 8)
    with pytest.raises(A.AnchorError):
        A.last_token_of_span(spans, 0)


def test_incremental_fallback_for_non_byte_level_tokenizer():
    class Stub:
        vocab = {1: "ab", 2: "cd", 3: "é"}

        def decode(self, ids):
            return "".join(self.vocab[i] for i in ids)

        def convert_ids_to_tokens(self, ids):
            raise AttributeError

    pieces = A.token_bytes(Stub(), [1, 2, 3])
    assert pieces == [b"ab", b"cd", "é".encode("utf-8")]


# --------------------------------------------------------------------------- native anchors


def _native(tok, content, think="Let me add the two numbers. 二加二。", *, enable_thinking=True, finish_reason="stop", im_end=True, stored_content="same"):
    messages, prompt = _prompt(tok, "What is 2+2?", enable_thinking)
    completion = support.completion_from_text(tok, think if enable_thinking else None, content, im_end=im_end)
    rec = support.synthetic_record(prompt, completion, messages=messages, content=content if stored_content == "same" else stored_content,
                                   reasoning=think if enable_thinking else None, finish_reason=finish_reason, enable_thinking=enable_thinking)
    return rec, prompt, completion


def _by_kind(anchors):
    return {a.kind: a for a in anchors}


def test_five_anchors_plain_json(tok):
    content = "\n\n" + support.CANDIDATE_JSON  # vLLM keeps the "\n\n" after </think>
    rec, prompt, completion = _native(tok, content)
    anchors = A.resolve_native_anchors(rec, tok, ks=(4, 32, 512))
    assert [a.kind for a in anchors] == ["NATIVE_PREFILL", "GENERATED_4", "GENERATED_32", "GENERATED_512", "FINAL_OBJECT_CLOSE"]
    k = _by_kind(anchors)
    plen = len(prompt)
    assert k["NATIVE_PREFILL"].token_offset == plen - 1 and k["NATIVE_PREFILL"].channel == "prompt"
    assert k["GENERATED_4"].token_offset == plen + 3 and k["GENERATED_4"].channel == "thinking"
    assert k["GENERATED_32"].token_offset == plen + 31 and k["GENERATED_32"].channel == "content"
    assert k["GENERATED_512"].token_offset is None and k["GENERATED_512"].missingness == "NOT_REACHED"
    close = k["FINAL_OBJECT_CLOSE"]
    assert close.present and close.channel == "content" and close.detail["content_channel"] == "exact"
    j = close.detail["completion_index"]
    assert "}" in tok.decode([completion[j]])
    # the closing brace is the last non-special token (im_end follows) → generated count = j+1
    assert completion[j + 1] == tok.convert_tokens_to_ids("<|im_end|>")
    assert close.generated_token_count == j + 1 and close.token_offset == plen + j
    assert A.k_max_needed(anchors) == j + 1


@pytest.mark.parametrize("wrap", ["```json\n{}\n```", "```\n{}\n```", "  \n```json  \n{}\n  ```  \n"])
def test_json_close_inside_one_fence(tok, wrap):
    content = "\n\n" + wrap.replace("{}", support.CANDIDATE_JSON)
    rec, prompt, completion = _native(tok, content)
    close = _by_kind(A.resolve_native_anchors(rec, tok))["FINAL_OBJECT_CLOSE"]
    assert close.present, close
    j = close.detail["completion_index"]
    piece = tok.decode([completion[j]])
    assert "}" in piece and "`" not in piece  # the brace token, not the closing fence
    start, end = A.json_object_span(content)
    assert content[start:end] == support.CANDIDATE_JSON


def test_json_close_with_cjk_payload_maps_to_the_brace_token(tok):
    content = "\n\n" + support.CANDIDATE_JSON_CJK
    rec, prompt, completion = _native(tok, content, think="两个整数相加")
    close = _by_kind(A.resolve_native_anchors(rec, tok))["FINAL_OBJECT_CLOSE"]
    assert close.present
    pieces = A.token_bytes(tok, completion)
    spans = A.byte_spans(pieces)
    j = close.detail["completion_index"]
    brace_byte = close.detail["brace_byte"]
    assert spans[j][0] <= brace_byte < spans[j][1]
    assert b"".join(pieces)[brace_byte:brace_byte + 1] == b"}"
    # bytes of the whole object span decode to exactly the stored JSON (CJK preserved)
    obj = b"".join(pieces)[close.detail["content_start_byte"]:].decode("utf-8")
    assert obj.startswith("\n\n" + support.CANDIDATE_JSON_CJK)


def test_invalid_candidate_is_json_close_missing(tok):
    rec, _, _ = _native(tok, "\n\n{\"approach\": \"x\"}")  # schema failure
    close = _by_kind(A.resolve_native_anchors(rec, tok))["FINAL_OBJECT_CLOSE"]
    assert close.missingness == "JSON_CLOSE_MISSING" and close.detail["parse"] == "SCHEMA"
    rec, _, _ = _native(tok, "\n\n" + support.CANDIDATE_JSON, finish_reason="length")
    assert _by_kind(A.resolve_native_anchors(rec, tok))["FINAL_OBJECT_CLOSE"].detail["parse"] == "TRUNCATED"
    rec, _, _ = _native(tok, "\n\n" + support.CANDIDATE_JSON + "\ntrailing")
    assert _by_kind(A.resolve_native_anchors(rec, tok))["FINAL_OBJECT_CLOSE"].detail["parse"] == "TRAILING_TEXT"


def test_subtask_role_is_not_applicable_and_hub_uses_coordinator_object(tok):
    rec, _, _ = _native(tok, "\n\n" + support.CANDIDATE_JSON)
    close = _by_kind(A.resolve_native_anchors(rec, tok, final_object=A.FINAL_OBJECT_SUBTASK))["FINAL_OBJECT_CLOSE"]
    assert close.missingness == "FINAL_OBJECT_NOT_APPLICABLE" and not close.present
    action = '{"action":"final","candidate":' + support.CANDIDATE_JSON + "}"
    rec, prompt, completion = _native(tok, "\n\n" + action)
    close = _by_kind(A.resolve_native_anchors(rec, tok, final_object=A.FINAL_OBJECT_COORDINATOR))["FINAL_OBJECT_CLOSE"]
    assert close.present and close.detail["parse"] == "valid:final"
    assert close.detail["completion_index"] == len(completion) - 2  # last "}" before <|im_end|>
    bad = _by_kind(A.resolve_native_anchors(rec, tok, final_object=A.FINAL_OBJECT_CANDIDATE))["FINAL_OBJECT_CLOSE"]
    assert bad.missingness == "JSON_CLOSE_MISSING"
    assert A.final_object_for_role("worker") == A.FINAL_OBJECT_SUBTASK
    with pytest.raises(A.AnchorError):
        A.final_object_for_role("judge")


def test_thinking_never_closed_means_no_content_channel(tok):
    messages, prompt = _prompt(tok, "Q?")
    completion = tok.encode("<think>\nstill thinking " * 20, add_special_tokens=False)
    rec = support.synthetic_record(prompt, completion, messages=messages, content=None, reasoning="still thinking", finish_reason="length")
    k = _by_kind(A.resolve_native_anchors(rec, tok, ks=(32,)))
    assert k["GENERATED_32"].channel == "thinking"
    assert k["FINAL_OBJECT_CLOSE"].missingness == "JSON_CLOSE_MISSING"
    ch = A.locate_content_channel(tok, completion, None, True)
    assert ch.status == "absent" and ch.start_byte is None


def test_thinking_off_whole_completion_is_content(tok):
    content = "\n\n" + support.CANDIDATE_JSON
    rec, prompt, completion = _native(tok, content, enable_thinking=False)
    k = _by_kind(A.resolve_native_anchors(rec, tok, ks=(2,)))
    assert k["GENERATED_2"].channel == "content"
    assert k["FINAL_OBJECT_CLOSE"].present and k["FINAL_OBJECT_CLOSE"].detail["content_channel"] == "exact"


def test_stored_content_without_leading_newlines_is_a_suffix_alignment(tok):
    rec, _, _ = _native(tok, "\n\n" + support.CANDIDATE_JSON, stored_content=support.CANDIDATE_JSON)
    close = _by_kind(A.resolve_native_anchors(rec, tok))["FINAL_OBJECT_CLOSE"]
    assert close.present and close.detail["content_channel"] == "suffix"


def test_misaligned_content_is_channel_unavailable(tok):
    rec, _, _ = _native(tok, "\n\n" + support.CANDIDATE_JSON, stored_content=support.CANDIDATE_JSON_CJK)
    close = _by_kind(A.resolve_native_anchors(rec, tok))["FINAL_OBJECT_CLOSE"]
    assert close.missingness == "CHANNEL_UNAVAILABLE"


def test_empty_completion(tok):
    messages, prompt = _prompt(tok, "Q?")
    rec = support.synthetic_record(prompt, [], messages=messages, content=None, reasoning=None)
    anchors = A.resolve_native_anchors(rec, tok)
    assert anchors[0].present and all(not a.present for a in anchors[1:])
    assert A.k_max_needed(anchors) == 0


def test_anchor_invariants():
    with pytest.raises(ValueError):
        A.Anchor("X", None, None, "none", "s", 0)
    with pytest.raises(ValueError):
        A.Anchor("X", 3, "NOT_REACHED", "none", "s", 0)
    with pytest.raises(ValueError):
        A.Anchor("X", None, "BOGUS", "none", "s", 0)


# --------------------------------------------------------------------------- report anchors


def _report(tok, content, **fields):
    messages, ids = _prompt(tok, content, enable_thinking=False)
    return {"report_id": "rep", "messages": messages, "prompt_token_ids": ids, "chat_template_kwargs": {"enable_thinking": False}, **fields}


def test_report_anchors_from_byte_offsets(tok):
    task = "=== TASK ===\n二加二等于几？\n=== END TASK ===\n"
    state = "=== REPORT (observable state) ===\nselected: 四\n=== END REPORT ===\n"
    content = task + state + "Output contract: {}"
    task_end = len(task.encode("utf-8"))
    state_end = len((task + state).encode("utf-8"))
    rep = _report(tok, content, task_only_anchor_byte=task_end, state_anchor_byte=state_end)
    k = _by_kind(A.resolve_report_anchors(rep, tok))
    ids = rep["prompt_token_ids"]
    pieces = A.token_bytes(tok, ids)
    spans = A.byte_spans(pieces)
    prefix = A.content_prefix_bytes(tok, rep["messages"], False)
    assert b"".join(pieces)[prefix:prefix + len(content.encode())] == content.encode()
    t = k["TASK_ONLY_ANCHOR"].token_offset
    assert spans[t][0] <= prefix + task_end - 1 < spans[t][1]
    assert b"".join(pieces)[prefix + task_end - 1:prefix + task_end] == b"\n"
    s = k["STATE_ANCHOR"].token_offset
    assert spans[s][0] <= prefix + state_end - 1 < spans[s][1]
    assert t < s < k["LAST_PREFILL"].token_offset == len(ids) - 1
    assert k["LAST_PREFILL"].detail["equals_state_anchor"] is False
    # the token form must agree with the byte form when both are given
    rep2 = dict(rep, task_only_anchor_token=t, state_anchor_token=s)
    assert _by_kind(A.resolve_report_anchors(rep2, tok))["STATE_ANCHOR"].detail["source"] == "token+byte"
    with pytest.raises(A.AnchorError):
        A.resolve_report_anchors(dict(rep, state_anchor_token=s - 1), tok)


def test_report_anchors_from_the_n2_nested_shape(tok):
    """P0-1 regression: N2 (``forecast.manifest.report_render``) writes ``byte_anchors`` and
    ``anchor_tokens{name: {index}}``; the resolver must read them (and agree with the flat form)."""
    task = "=== TASK ===\nWhat is 2+2?\n=== END TASK ===\n"
    state = "=== REPORT (observable state) ===\nselected: 4\n=== END REPORT ===\n"
    content = task + state + "Output contract: {}"
    task_end, state_end = len(task.encode()), len((task + state).encode())
    flat = _by_kind(A.resolve_report_anchors(_report(tok, content, task_only_anchor_byte=task_end, state_anchor_byte=state_end), tok))
    t, s = flat["TASK_ONLY_ANCHOR"].token_offset, flat["STATE_ANCHOR"].token_offset
    nested = _report(tok, content, byte_anchors={"task_only_anchor": task_end, "state_anchor": state_end},
                     anchor_tokens={"task_only_anchor": {"index": t, "straddles": True}, "state_anchor": {"index": s, "straddles": False}})
    assert A.report_anchor_fields(nested, "task_only_anchor") == (t, task_end) and A.report_anchor_fields(nested, "state_anchor") == (s, state_end)
    k = _by_kind(A.resolve_report_anchors(nested, tok))
    assert k["TASK_ONLY_ANCHOR"].present and k["TASK_ONLY_ANCHOR"].token_offset == t and k["TASK_ONLY_ANCHOR"].detail["source"] == "token+byte"
    assert k["STATE_ANCHOR"].token_offset == s and k["STATE_ANCHOR"].detail["source"] == "token+byte"
    assert k["STATE_ANCHOR"].detail["state_anchor_byte"] == state_end and k["LAST_PREFILL"].detail["equals_state_anchor"] is False
    # bytes only / tokens only in the nested shape
    k = _by_kind(A.resolve_report_anchors(_report(tok, content, byte_anchors={"task_only_anchor": task_end, "state_anchor": state_end}), tok))
    assert (k["TASK_ONLY_ANCHOR"].token_offset, k["STATE_ANCHOR"].token_offset) == (t, s) and k["STATE_ANCHOR"].detail["source"] == "byte"
    k = _by_kind(A.resolve_report_anchors(_report(tok, content, anchor_tokens={"task_only_anchor": {"index": t}, "state_anchor": {"index": s}}), tok))
    assert (k["TASK_ONLY_ANCHOR"].token_offset, k["STATE_ANCHOR"].token_offset) == (t, s) and k["STATE_ANCHOR"].detail["source"] == "token"
    # a nested token that disagrees with the byte form is refused
    with pytest.raises(A.AnchorError, match="!="):
        A.resolve_report_anchors(_report(tok, content, byte_anchors={"task_only_anchor": task_end, "state_anchor": state_end},
                                         anchor_tokens={"task_only_anchor": {"index": t}, "state_anchor": {"index": s - 1}}), tok)
    with pytest.raises(A.AnchorError, match="unknown report anchor"):
        A.report_anchor_fields(nested, "last_prefill")


def test_report_state_anchor_defaults_to_last_prefill_and_task_unresolved(tok):
    rep = _report(tok, "just a task")
    k = _by_kind(A.resolve_report_anchors(rep, tok))
    assert k["STATE_ANCHOR"].token_offset == len(rep["prompt_token_ids"]) - 1
    assert k["STATE_ANCHOR"].detail["source"] == "last_prefill_default"
    assert k["TASK_ONLY_ANCHOR"].missingness == "ANCHOR_UNRESOLVED"
    assert k["LAST_PREFILL"].detail["equals_state_anchor"] is True


def test_report_identity_failure_raises(tok):
    rep = _report(tok, "task", task_only_anchor_byte=2)
    rep["prompt_token_ids"] = rep["prompt_token_ids"][:-1]  # not the render any more
    with pytest.raises(A.AnchorError):
        A.resolve_report_anchors(rep, tok)
    with pytest.raises(A.AnchorError):
        A.resolve_report_anchors({"report_id": "x"}, tok)


def test_content_prefix_is_structural(tok):
    messages = [{"role": "user", "content": "user"}]  # a content equal to the role word
    prefix = A.content_prefix_bytes(tok, messages, False)
    assert A.render_chat_text(tok, messages, False).encode()[:prefix] == b"<|im_start|>user\n"
