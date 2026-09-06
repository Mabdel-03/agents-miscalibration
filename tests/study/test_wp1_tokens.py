"""WP1 — inference/tokens.py: pinned tokenizers, exact chat-template ids, envelopes (§4.3, §6.4)."""

from __future__ import annotations

import pytest

from agents_scaling.serving.context import rendered_chat_token_ids
from agents_scaling.study.inference import tokens
from agents_scaling.study.types import PROMPT_TOKENS_CAP, TASK_TOKENS_CAP, ContextFailure

MSGS = [{"role": "user", "content": "Solve: what is 2 + 2 ? Answer briefly."}]


# ----------------------------------------------------------------------------- stub tokenizer


def test_count_tokens_stub(stub_tokenizer):
    assert tokens.count_tokens("a b c", stub_tokenizer) == 3
    assert tokens.count_tokens("", stub_tokenizer) == 0
    with pytest.raises(TypeError):
        tokens.count_tokens(b"bytes", stub_tokenizer)  # type: ignore[arg-type]


def test_render_chat_token_ids_matches_context_helper(stub_tokenizer):
    ids = tokens.render_chat_token_ids(stub_tokenizer, MSGS, True)
    assert isinstance(ids, tuple) and all(isinstance(i, int) for i in ids)
    assert ids == rendered_chat_token_ids(stub_tokenizer, MSGS, enable_thinking=True)
    # thinking off appends the empty think block → strictly more tokens
    assert len(tokens.render_chat_token_ids(stub_tokenizer, MSGS, False)) > len(ids)


def test_render_chat_token_ids_validates_inputs(stub_tokenizer):
    with pytest.raises(TypeError):
        tokens.render_chat_token_ids(stub_tokenizer, MSGS, "yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tokens.render_chat_token_ids(stub_tokenizer, [{"role": "user"}], True)
    with pytest.raises(ValueError):
        tokens.render_chat_token_ids(stub_tokenizer, [{"role": "user", "content": "x", "name": "n"}], True)


def test_render_chat_text_stub(stub_tokenizer):
    text = tokens.render_chat_text(stub_tokenizer, MSGS, False)
    assert MSGS[0]["content"] in text and text.endswith("<think>\n\n</think>\n\n")


def test_envelope_check_passes_and_raises(stub_tokenizer):
    n = tokens.envelope_check(MSGS, stub_tokenizer)
    assert 0 < n <= PROMPT_TOKENS_CAP
    with pytest.raises(ContextFailure):
        tokens.envelope_check(MSGS, stub_tokenizer, cap=n - 1)
    assert tokens.envelope_check(MSGS, stub_tokenizer, cap=n) == n
    with pytest.raises(ValueError):
        tokens.envelope_check(MSGS, stub_tokenizer, cap=0)


def test_task_envelope_check(stub_tokenizer):
    text = " ".join(["w"] * 4096)
    assert tokens.task_envelope_check(text, stub_tokenizer) == TASK_TOKENS_CAP
    with pytest.raises(ContextFailure):
        tokens.task_envelope_check(text + " x", stub_tokenizer)
    with pytest.raises(ContextFailure):
        tokens.task_envelope_check("a b c", stub_tokenizer, cap=2)


def test_chat_template_hash_requires_template(stub_tokenizer):
    with pytest.raises(ValueError):
        tokens.chat_template_hash(stub_tokenizer)  # stub has no chat_template attribute

    class WithTemplate:
        chat_template = "{{ messages }}"

    h = tokens.chat_template_hash(WithTemplate())
    assert len(h) == 64 and h == tokens.chat_template_hash(WithTemplate())


def test_load_tokenizer_rejects_unknown_checkpoint():
    with pytest.raises(KeyError):
        tokens.load_tokenizer("70B")
    with pytest.raises(TypeError):
        tokens.load_tokenizer(32)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- real tokenizer


def test_real_tokenizer_cached_and_pinned(real_tokenizer_32b, study_config):
    tok1 = tokens.load_tokenizer("32B")
    tok2 = tokens.load_tokenizer(study_config.checkpoints["32B"])
    assert tok1 is tok2  # cached once per process
    assert tokens.flagship_tokenizer() is tok1
    assert len(tokens.chat_template_hash(tok1)) == 64
    assert tokens.chat_template_hash(tok1) == tokens.chat_template_hash(real_tokenizer_32b)


def test_real_tokenizer_exact_ids(real_tokenizer_32b):
    ids_on = tokens.render_chat_token_ids(real_tokenizer_32b, MSGS, True)
    ids_off = tokens.render_chat_token_ids(real_tokenizer_32b, MSGS, False)
    assert ids_on == rendered_chat_token_ids(real_tokenizer_32b, MSGS, enable_thinking=True)
    assert ids_on[0] == 151644  # <|im_start|>
    assert len(ids_off) > len(ids_on)
    assert real_tokenizer_32b.decode(ids_off).endswith("<think>\n\n</think>\n\n")
    assert tokens.render_chat_text(real_tokenizer_32b, MSGS, True).startswith("<|im_start|>user\n")
    assert tokens.count_tokens("Hello there 世界 🚀", real_tokenizer_32b) == 7
    assert tokens.envelope_check(MSGS, real_tokenizer_32b) == len(ids_on)


def test_all_four_checkpoints_share_vocab_and_template(real_tokenizer_32b, study_config):
    sizes = list(study_config.checkpoints)
    toks = {size: tokens.load_tokenizer(size) for size in sizes}
    sample = "def task_func(xs):\n    return sum(xs)  # 合計 🚀"
    counts = {size: tokens.count_tokens(sample, t) for size, t in toks.items()}
    assert len(set(counts.values())) == 1, counts
    assert len({tokens.chat_template_hash(t) for t in toks.values()}) == 1
