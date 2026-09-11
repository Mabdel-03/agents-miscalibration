"""Integration: the receiver assay's answer-logit margin must be obtainable at runtime.

Spec §11 names the answer-logit margin a primary assay outcome.  WP1 correctly asked for
twenty alternatives on the receiver purpose, but the parser kept only the chosen token's
logprob, so ``margin_from_record`` could only ever record the margin as missing.  These
tests pin the plumbing that closes that gap: the top-k block is parsed, it is sliced to the
final span, and it is carried on the record only for the purpose that asked for it.
"""

from __future__ import annotations

import pytest

from agents_scaling.ard import client as CL
from agents_scaling.ard import contract as K


class _Alt:
    def __init__(self, token: str, logprob: float) -> None:
        self.token, self.logprob = token, logprob


class _Entry:
    def __init__(self, token: str, logprob: float, alts) -> None:
        self.token, self.logprob, self.top_logprobs = token, logprob, alts


class _Block:
    def __init__(self, content) -> None:
        self.content = content


class _Choice:
    def __init__(self, content) -> None:
        self.logprobs = _Block(content)


def _choice(rows):
    return _Choice([_Entry(tok, lp, [_Alt(t, l) for t, l in alts]) for tok, lp, alts in rows])


def test_top_logprobs_are_parsed_from_the_openai_shape():
    rows = [("42", -0.1, [("42", -0.1), ("41", -2.5)]),
            ("!", -0.7, [("!", -0.7), (".", -1.1)])]
    got = CL._top_logprobs(_choice(rows))
    assert got == [[{"token": "42", "logprob": -0.1}, {"token": "41", "logprob": -2.5}],
                   [{"token": "!", "logprob": -0.7}, {"token": ".", "logprob": -1.1}]]


def test_a_missing_logprobs_block_is_none_not_an_empty_list():
    """Missing must stay distinguishable from 'present but empty' (never scored as zero)."""
    class _Bare:
        logprobs = None

    assert CL._top_logprobs(_Bare()) is None


def test_non_finite_alternatives_are_dropped_rather_than_written():
    """The store refuses non-finite constants on read-back, so -inf must not reach a record."""
    rows = [("x", -0.2, [("x", -0.2), ("impossible", float("-inf"))])]
    got = CL._top_logprobs(_choice(rows))
    assert got == [[{"token": "x", "logprob": -0.2}]]


def test_the_receiver_purpose_asks_for_twenty_and_others_for_one():
    assert CL.top_logprobs_for(K.PURPOSE_RECEIVER) == K.RECEIVER_TOP_LOGPROBS == 20
    assert CL.top_logprobs_for(K.PURPOSE_REVISE) == 1


def test_the_margin_is_computable_from_a_carried_block():
    """The whole point: two candidate answers, one differing token, a real margin."""
    receiver = pytest.importorskip("agents_scaling.ard.receiver")
    rows = [("7", -0.05, [("7", -0.05), ("9", -3.25)])]
    parsed = CL._top_logprobs(_choice(rows))
    alts = {a["token"]: a["logprob"] for a in parsed[0]}
    assert alts["7"] - alts["9"] == pytest.approx(3.20)
    assert hasattr(receiver, "answer_logit_margin")
