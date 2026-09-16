"""WP1: the one place a ``Decoding`` is built, and the frozen continuation prefill.

Plan D5 (``max_tokens = 1 + T + 1 + F``, amendment A1), D6 (no guided field is ever sent),
spec 5.5/5.6.  The point of these tests is that the ``max_tokens`` identity is asserted where
the object is *constructed*, so no caller can re-derive it slightly differently, and that the
prefill bytes have exactly one home which ``prompts.lock.json`` copies.
"""

from __future__ import annotations

import hashlib

import pytest

from agents_scaling.nextdyn import adapters as A
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn import decoding as D
from agents_scaling.nextdyn import locks as L
from agents_scaling.nextdyn.errors import ContractViolation


def test_a_native_decoding_pays_for_both_delimiters_and_keeps_its_final_allowance(nd_tokens):
    """Amendment A1: the two delimiters are sampled tokens and sit inside ``max_tokens``."""
    decoding = D.decoding_for(A.ADAPTERS[K.ADAPTER_NATIVE_THINKING], _sampling(), nd_tokens,
                              purpose=K.PHASE_BANK)
    assert decoding.max_tokens == 1 + nd_tokens.thinking_cap + 1 + nd_tokens.final_cap
    assert decoding.max_tokens == nd_tokens.max_new_tokens
    assert (decoding.thinking_tokens, decoding.final_tokens) == (nd_tokens.thinking_cap,
                                                                 nd_tokens.final_cap)
    assert decoding.enable_thinking is True


def test_the_envelope_adapter_asks_for_its_total_output_cap_and_no_thinking_budget(nd_contract,
                                                                                  nd_tokens):
    """Spec 5.6's companion has one output envelope; the native caps do not apply to it."""
    cell = nd_contract.model("qwen_nonthinking")
    adapter = A.adapter_for(cell)
    decoding = D.decoding_for(adapter, _sampling(), nd_tokens, purpose=K.PHASE_BANK)
    assert decoding.max_tokens == cell.total_output_tokens != nd_tokens.max_new_tokens
    assert decoding.thinking_tokens == 0 and decoding.enable_thinking is False
    assert decoding.chat_template_kwargs == {"enable_thinking": False}


def test_a_decoding_whose_max_tokens_drifted_from_the_envelope_is_refused(nd_tokens):
    """The assertion lives with the construction, so a wrong adapter cannot slip one past."""

    class Drifted(A.NativeThinkingAdapter):
        def decoding(self, sampling, tokens, *, purpose):
            return D.build_decoding(sampling, max_tokens=tokens.max_new_tokens - 1,
                                    enable_thinking=True, thinking_tokens=tokens.thinking_cap,
                                    final_tokens=tokens.final_cap)

    with pytest.raises(ContractViolation, match="max_tokens"):
        D.decoding_for(Drifted(), _sampling(), nd_tokens, purpose=K.PHASE_BANK)


def test_no_decoding_this_module_builds_carries_a_guided_field(nd_tokens, nd_contract):
    """Plan D6: guided_json is accepted but not enforced on vLLM 0.21, so it is never sent."""
    for name in (K.ADAPTER_NATIVE_THINKING, K.ADAPTER_TOTAL_OUTPUT_ENVELOPE):
        adapter = A.adapter_for(nd_contract.model(
            "primary" if name == K.ADAPTER_NATIVE_THINKING else "qwen_nonthinking"))
        assert D.decoding_for(adapter, _sampling(), nd_tokens, purpose=K.PHASE_BANK).guided_json is False
    assert D.continuation_decoding(nd_tokens, _sampling()).guided_json is False


def test_a_sampler_field_the_wire_cannot_carry_is_refused_not_dropped(nd_tokens):
    """R5: a non-zero frequency_penalty could only reach vLLM outside the request identity."""
    sampling = K.Sampling.from_floats(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                                      repetition_penalty=1.0, presence_penalty=0.0,
                                      frequency_penalty=0.1)
    with pytest.raises(ContractViolation, match="frequency_penalty"):
        D.build_decoding(sampling, max_tokens=8, enable_thinking=True, thinking_tokens=4,
                         final_tokens=2)


def test_a_continuation_asks_for_the_final_allowance_and_nothing_else(nd_tokens):
    """Plan D5: the thinking channel is already closed, inside the prompt."""
    decoding = D.continuation_decoding(nd_tokens, _sampling())
    assert decoding.max_tokens == decoding.final_tokens == nd_tokens.final_cap
    assert decoding.thinking_tokens == 0
    assert decoding.enable_thinking is False
    assert D.is_continuation(decoding)
    assert not D.is_continuation(D.decoding_for(A.ADAPTERS[K.ADAPTER_NATIVE_THINKING], _sampling(),
                                                nd_tokens, purpose=K.PHASE_BANK))


def test_the_continuation_prefill_is_declared_once_and_the_lock_only_copies_it(tmp_path):
    """R1: ``locks.lock_prompts`` hashes ``decoding.CONTINUATION_PREFILL_TEXT`` itself."""
    assert D.CONTINUATION_PREFILL_TEXT == '\n\n{\n  "reasoning": "'
    assert L.CONTINUATION_PREFILL_ATTR == "CONTINUATION_PREFILL_TEXT"
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "root.txt").write_text("SYSTEM\nSolve.")
    lock = L.lock_prompts(templates, D.prefill_text())
    assert lock.continuation_prefill_sha256 == hashlib.sha256(
        D.CONTINUATION_PREFILL_TEXT.encode("utf-8")).hexdigest()
    assert D.prefill_text(lock) == D.CONTINUATION_PREFILL_TEXT


def test_a_drifted_prompts_lock_fails_closed_rather_than_choosing_a_side(tmp_path):
    """The prefill is prompt bytes: a lock that disagrees would silently change request ids."""
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "root.txt").write_text("SYSTEM\nSolve.")
    drifted = L.lock_prompts(templates, '\n\n{\n  "rationale": "')
    with pytest.raises(ContractViolation, match="drifted"):
        D.prefill_text(drifted)
    empty = L.PromptsLock(schema_version=1, locked_at=None, templates={},
                          continuation_prefill_text=None, continuation_prefill_sha256=None)
    with pytest.raises(ContractViolation, match="lock-inputs"):
        D.prefill_text(empty)


def test_the_prefill_round_trips_through_the_tokenizer(fake_tokenizer):
    """A prefill that did not survive encode/decode would put bytes the model never sees into
    the prompt, so the check is part of building the ids, not a separate audit."""
    ids = D.continuation_prefill_ids(fake_tokenizer)
    assert ids and "".join(fake_tokenizer.decode(list(ids)).split()) == '{"reasoning":"'
    with pytest.raises(ContractViolation):
        D.continuation_prefill_ids(_LosslessTokenizer(), "{ dropped }")


def test_an_unknown_purpose_is_refused(nd_tokens):
    with pytest.raises(ContractViolation, match="purpose"):
        D.decoding_for(A.ADAPTERS[K.ADAPTER_NATIVE_THINKING], _sampling(), nd_tokens,
                       purpose="not_a_phase")


class _LosslessTokenizer:
    """A tokenizer that silently drops a token: the case the round-trip check exists for."""

    def encode(self, text, add_special_tokens=False):
        return [1, 2]

    def decode(self, ids):
        return "{"


def _sampling() -> K.Sampling:
    return K.Sampling.from_floats(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                                  repetition_penalty=1.0, presence_penalty=0.0,
                                  frequency_penalty=0.0)
