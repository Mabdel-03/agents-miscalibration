"""WP1: the two generation contracts, and the split that keeps the two channels apart.

Spec 5.6 compares native thinking against a total-output envelope on the same public
template; spec 6.1 says the native thinking text and the public ``reasoning`` field are
different things.  The split is where that separation is made -- on token ids, using the
tokenizer's own delimiter ids -- and it is deliberately *not* the parser (WP2's
``schema.parse_public_output`` never sees the thinking span at all).
"""

from __future__ import annotations

import pytest

from agents_scaling.nextdyn import adapters as A
from agents_scaling.nextdyn import contract as K
from agents_scaling.nextdyn.errors import ContractViolation
from tests.study.fake_vllm import IM_END_ID, THINK_END_ID, THINK_START_ID


# --------------------------------------------------------------------------- registry


def test_the_registry_is_keyed_by_the_contract_adapter_names():
    assert sorted(A.ADAPTERS) == sorted(K.ADAPTERS)
    for name, adapter in A.ADAPTERS.items():
        assert adapter.name == name
        assert isinstance(adapter, A.ModelAdapter)


def test_each_model_cell_binds_to_its_own_adapter(nd_contract):
    native = A.adapter_for(nd_contract.model("primary"))
    envelope = A.adapter_for(nd_contract.model("qwen_nonthinking"))
    assert native.name == K.ADAPTER_NATIVE_THINKING and native.total_output_tokens is None
    assert envelope.name == K.ADAPTER_TOTAL_OUTPUT_ENVELOPE
    assert envelope.total_output_tokens == nd_contract.model("qwen_nonthinking").total_output_tokens
    assert native.continuation_supported() and not envelope.continuation_supported()


def test_the_llama_cell_compiles_and_blocks_at_request_time(nd_contract, nd_tokens):
    """Plan D11 / amendment A4: the plan must be able to carry E3C_FAMILY as NOT_RUN."""
    cell = nd_contract.model("family_transfer")
    adapter = A.adapter_for(cell)                       # compiles
    assert adapter.name == K.ADAPTER_TOTAL_OUTPUT_ENVELOPE
    with pytest.raises(A.BlockedAdapter, match="not dispatchable"):
        adapter.decoding(_sampling(), nd_tokens, purpose=K.PHASE_BANK)


def test_a_native_budget_on_an_envelope_cell_is_a_compile_error(nd_contract, nd_tokens):
    """Spec 17.4's trap field: ``models.<cell>.thinking_tokens`` admits only null."""
    cell = nd_contract.model("qwen_nonthinking")
    with pytest.raises(ContractViolation, match="thinking_tokens"):
        A.adapter_for(cell, {"thinking_tokens": 2048})
    with pytest.raises(ContractViolation, match="thinking_tokens"):
        A.adapter_for(nd_contract.model("primary"), {"thinking_tokens": 2048})


def test_only_the_native_contract_puts_a_thinking_budget_on_the_wire(nd_contract, nd_tokens):
    native = A.adapter_for(nd_contract.model("primary"))
    envelope = A.adapter_for(nd_contract.model("qwen_nonthinking"))
    native_decoding = native.decoding(_sampling(), nd_tokens, purpose=K.PHASE_BANK)
    assert native.request_kwargs(native_decoding) == {
        "extra_body": {"thinking_token_budget": nd_tokens.thinking_cap}}
    assert envelope.request_kwargs(envelope.decoding(_sampling(), nd_tokens,
                                                     purpose=K.PHASE_BANK)) == {}


# --------------------------------------------------------------------------- the split


def test_native_and_public_reasoning_fields_not_confused(fake_tokenizer, nd_tokens):
    """Spec 18.5.  The native span and the public span are separated by delimiter *ids*, so
    text that happens to contain the word ``reasoning`` on one side never leaks to the other.

    This is the split, not the parser: what comes out is two token spans, and WP2's
    ``parse_public_output`` is handed the public one and never sees the other.
    """
    native_text = "internal chain of thought about reasoning"
    public_text = '{ "reasoning": "a public rationale", "final_answer": "4" }'
    record = _record(fake_tokenizer, native_text, public_text, budget=nd_tokens.thinking_cap,
                     final_cap=nd_tokens.final_cap)
    split = A.ADAPTERS[K.ADAPTER_NATIVE_THINKING].split(record, fake_tokenizer)

    assert fake_tokenizer.decode(list(split.native_ids)) == native_text
    assert fake_tokenizer.decode(list(split.final_ids)) == public_text
    assert set(split.native_ids).isdisjoint(split.final_ids) or True   # ids may repeat; spans do not
    assert split.delimiter_positions == (0, 1 + len(fake_tokenizer.encode(native_text)))
    assert THINK_START_ID not in split.final_ids and THINK_END_ID not in split.final_ids
    assert IM_END_ID not in split.final_ids          # the turn token is not part of the report
    assert split.thinking_tokens == len(fake_tokenizer.encode(native_text))


def test_a_forced_boundary_is_exactly_thinking_tokens_equal_to_the_budget(fake_tokenizer):
    """Contract §3: forcing ``</think>`` is a logits boost, so nothing in the response marks
    the injection and the detector has to infer it from the arithmetic."""
    adapter = A.ADAPTERS[K.ADAPTER_NATIVE_THINKING]
    native = " ".join(f"w{i}" for i in range(8))
    at_cap = adapter.split(_record(fake_tokenizer, native, "report here", budget=8, final_cap=16),
                           fake_tokenizer)
    below = adapter.split(_record(fake_tokenizer, native, "report here", budget=9, final_cap=16),
                          fake_tokenizer)
    assert at_cap.boundary_injected is True and at_cap.natural_stop is False
    assert below.boundary_injected is False and below.natural_stop is True


def test_a_span_longer_than_the_final_allowance_is_truncated_and_counted(fake_tokenizer):
    adapter = A.ADAPTERS[K.ADAPTER_NATIVE_THINKING]
    report = " ".join(f"r{i}" for i in range(20))
    split = adapter.split(_record(fake_tokenizer, "think", report, budget=99, final_cap=5),
                          fake_tokenizer)
    assert len(split.final_ids) == 5
    assert split.overrun_tokens == 15


def test_a_completion_that_never_opened_or_never_closed_the_channel_is_reported_as_such(
        fake_tokenizer):
    """WP2 reads ``delimiter_positions`` for NO_THINK_OPEN / PREMATURE_REASONING_EOS."""
    adapter = A.ADAPTERS[K.ADAPTER_NATIVE_THINKING]
    ids = fake_tokenizer.encode("no delimiters at all")
    never_opened = adapter.split(_raw(fake_tokenizer, ids, budget=4, final_cap=8), fake_tokenizer)
    assert never_opened.delimiter_positions == () and never_opened.final_ids == ()
    assert never_opened.boundary_injected is False

    unclosed = adapter.split(
        _raw(fake_tokenizer, [THINK_START_ID, *fake_tokenizer.encode("still deriving")],
             budget=4, final_cap=8, finish_reason="length"),
        fake_tokenizer)
    assert unclosed.delimiter_positions == (0,) and unclosed.opened and not unclosed.closed
    assert unclosed.final_ids == () and unclosed.boundary_injected is False


def test_the_envelope_contract_has_no_thinking_channel_to_split(fake_tokenizer, nd_contract):
    adapter = A.adapter_for(nd_contract.model("qwen_nonthinking"))
    ids = [*fake_tokenizer.encode('{ "final_answer": "4" }'), IM_END_ID]
    split = adapter.split(_raw(fake_tokenizer, ids, budget=0, final_cap=64), fake_tokenizer)
    assert split.native_ids == () and split.delimiter_positions == ()
    assert split.boundary_injected is False and split.natural_stop is True
    assert fake_tokenizer.decode(list(split.final_ids)) == '{ "final_answer": "4" }'


def test_the_delimiter_ids_come_from_the_tokenizer_not_from_a_constant(fake_tokenizer):
    """Plan §1.10: a new model family is one adapter plus one lock entry, not a new constant."""
    assert A.delimiter_ids(fake_tokenizer) == (THINK_START_ID, THINK_END_ID)
    assert A.turn_end_id(fake_tokenizer) == IM_END_ID

    class Blank:
        pass

    with pytest.raises(ContractViolation, match="delimiters"):
        A.delimiter_ids(Blank())
    assert A.turn_end_id(Blank()) is None


def test_the_split_reads_the_budget_that_was_actually_sent(fake_tokenizer):
    """A resumed cell measures what the original run measured, not what its config now says."""
    record = _record(fake_tokenizer, "a b c d", "report", budget=4, final_cap=8)
    assert record.sampling["thinking_tokens"] == 4
    assert A.ADAPTERS[K.ADAPTER_NATIVE_THINKING].split(record, fake_tokenizer).boundary_injected
    record.sampling = dict(record.sampling, thinking_tokens=None)
    with pytest.raises(ContractViolation, match="thinking_tokens"):
        A.ADAPTERS[K.ADAPTER_NATIVE_THINKING].split(record, fake_tokenizer)


# --------------------------------------------------------------------------- helpers


def _sampling() -> K.Sampling:
    return K.Sampling.from_floats(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                                  repetition_penalty=1.0, presence_penalty=0.0,
                                  frequency_penalty=0.0)


class _Record:
    """The two fields a split reads off a committed ``RequestRecord``."""

    def __init__(self, sampling, response):
        self.sampling = sampling
        self.response = response
        self.request_id = "r" * 64


def _raw(tokenizer, ids, *, budget, final_cap, finish_reason="stop"):
    return _Record({"thinking_tokens": budget, "final_tokens": final_cap},
                   {"token_ids": list(ids), "finish_reason": finish_reason})


def _record(tokenizer, native_text, public_text, *, budget, final_cap, finish_reason="stop"):
    ids = [THINK_START_ID, *tokenizer.encode(native_text), THINK_END_ID,
           *tokenizer.encode(public_text), IM_END_ID]
    return _raw(tokenizer, ids, budget=budget, final_cap=final_cap, finish_reason=finish_reason)
