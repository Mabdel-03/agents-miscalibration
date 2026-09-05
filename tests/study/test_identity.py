"""identity.py: JCS bytes, semantic seeds, engine seeds, request identity (§3.5, §3.6)."""

from __future__ import annotations

import os

import pytest

from agents_scaling.study import identity
from agents_scaling.study.types import (
    NS_DEC,
    NS_STATELESS_BANK,
    PURPOSE_ROOT,
    SOLVER_DECODING,
    Checkpoint,
    RequestSpec,
    SeedKey,
)

KEY = SeedKey("hle:abc123", "main", "Qwen3-32B@9216db57", 0, 0, PURPOSE_ROOT, 3, NS_STATELESS_BANK)
CKPT = Checkpoint("32B", "Qwen/Qwen3-32B", "9" * 40, "9" * 40, "32B-long", 2, "32B")
C1_CONTROL = chr(0x80)  # U+0080 built at runtime so no raw control byte lives in this file
DALET_DAGESH = chr(0xFB33)  # precomposed U+FB33, as in the RFC example (not U+05D3 U+05BC)


# ----------------------------------------------------------------------------- jcs


def test_jcs_sorted_compact_ascii():
    assert identity.jcs({"b": 1, "a": [True, None, "x"]}) == b'{"a":[true,null,"x"],"b":1}'


def test_jcs_preserves_unicode_bytes_and_escapes_only_required_chars():
    assert identity.jcs({"é": "ü/\n\"\\"}) == '{"é":"ü/\\n\\"\\\\"}'.encode("utf-8")


def test_jcs_rfc8785_key_order_uses_utf16_units():
    # RFC 8785 section 3.2.3 worked example: the emoji (surrogate pair D83D DE00) sorts before U+FB33.
    obj = {
        "€": "Euro Sign",
        "\r": "Carriage Return",
        DALET_DAGESH: "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "\U0001f600": "Emoji: Grinning Face",
        C1_CONTROL: "Control",
        "ö": "Latin Small Letter O With Diaeresis",
    }
    expected = (
        '{"\\r":"Carriage Return","1":"One","' + C1_CONTROL + '":"Control",'
        '"ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign",'
        '"\U0001f600":"Emoji: Grinning Face","' + DALET_DAGESH + '":"Hebrew Letter Dalet With Dagesh"}'
    ).encode("utf-8")
    assert identity.jcs(obj) == expected


def test_jcs_nested_and_tuples_become_arrays():
    assert identity.jcs({"z": {"y": (1, 2), "x": [{"k": "v"}]}}) == b'{"z":{"x":[{"k":"v"}],"y":[1,2]}}'
    assert identity.jcs(KEY.as_array()) == b'["hle:abc123","main","Qwen3-32B@9216db57",0,0,"root",3,"stateless_bank"]'


def test_jcs_rejects_floats_and_non_string_keys():
    with pytest.raises(TypeError):
        identity.jcs({"t": 0.6})
    with pytest.raises(TypeError):
        identity.jcs([1.0])
    with pytest.raises(TypeError):
        identity.jcs({1: "x"})


def test_float_str_fixed_forms():
    assert identity.float_str(0.6) == "0.6"
    assert identity.float_str(1) == "1.0"
    assert identity.float_str(0.95) == "0.95"
    with pytest.raises(ValueError):
        identity.float_str(float("nan"))


# ----------------------------------------------------------------------------- seeds


def test_semantic_seed_golden_value():
    seed = identity.semantic_seed(b"\x00" * 32, KEY)
    assert len(seed) == 16
    assert seed.hex() == "32b018fa843b814f90ae192292c86117"
    assert identity.engine_seed(seed) == 3652446762036855119


def test_semantic_seed_accepts_plain_sequence_and_matches_dataclass():
    assert identity.semantic_seed(b"k" * 32, KEY.as_array()) == identity.semantic_seed(b"k" * 32, KEY)
    assert KEY.semantic_seed(b"k" * 32) == identity.semantic_seed(b"k" * 32, KEY)


def test_engine_seed_range():
    for _ in range(200):
        value = identity.engine_seed(os.urandom(16))
        assert 0 <= value < 2**63
    assert identity.engine_seed(b"\xff" * 16) == 2**63 - 1


def test_seed_keys_differing_only_in_namespace_differ():
    other = SeedKey(*KEY.as_array()[:-1], NS_DEC)
    assert identity.semantic_seed(b"s" * 32, KEY) != identity.semantic_seed(b"s" * 32, other)


def test_blind_order_key_is_deterministic_and_namespaced():
    a = identity.blind_order_key(b"s" * 32, "peer_order", "hle:1", 0, 2)
    assert a == identity.blind_order_key(b"s" * 32, "peer_order", "hle:1", 0, 2)
    assert a != identity.blind_order_key(b"s" * 32, "root_perm", "hle:1", 0, 2)
    assert len(a) == 32


# ----------------------------------------------------------------------------- request identity


def _spec(content: str, key: SeedKey = KEY) -> RequestSpec:
    return RequestSpec(
        messages=({"role": "user", "content": content},),
        decoding=SOLVER_DECODING,
        checkpoint=CKPT,
        seed_key=key,
        role="root",
        study_id="agent_design_v4_orcd",
        study_seed_hex="ab" * 32,
    )


def test_request_id_stable_and_sensitive_to_one_prompt_byte():
    base = _spec("Task: 2+2?")
    assert base.request_id == _spec("Task: 2+2?").request_id
    assert len(base.request_id) == 64
    assert base.request_id != _spec("Task: 2+2!").request_id
    assert base.input_hash != _spec("Task: 2+2!").input_hash


def test_request_id_changes_with_namespace_but_not_role():
    other_ns = _spec("Task: 2+2?", SeedKey(*KEY.as_array()[:-1], NS_DEC))
    assert other_ns.request_id != _spec("Task: 2+2?").request_id
    same_prompt_other_role = RequestSpec(**{**_spec("Task: 2+2?").__dict__, "role": "revise"})
    assert same_prompt_other_role.request_id == _spec("Task: 2+2?").request_id


def test_request_id_matches_manual_formula():
    spec = _spec("Task: 2+2?")
    manual = identity.sha256_hex(
        identity.jcs(
            [
                spec.study_id,
                CKPT.model_revision,
                CKPT.tokenizer_revision,
                "vllm==0.21.0;dtype=bf16;tp=2;reasoning_parser=qwen3;profile=32B-long",
                identity.input_hash(spec.messages, {"enable_thinking": True}),
                identity.decoding_hash(SOLVER_DECODING),
                spec.semantic_seed_hex,
                {"max_input": 32768, "max_tokens": 8192},
                "none",
            ]
        )
    )
    assert spec.request_id == manual
    assert CKPT.engine_digest == "vllm==0.21.0;dtype=bf16;tp=2;reasoning_parser=qwen3;profile=32B-long"


def test_decoding_hash_serializes_floats_as_fixed_strings():
    strings = SOLVER_DECODING.as_strings()
    assert strings["temperature"] == "0.6" and strings["top_p"] == "0.95" and strings["top_k"] == 20
    assert identity.decoding_hash(SOLVER_DECODING) == identity.decoding_hash(strings)
    assert identity.decoding_hash(SOLVER_DECODING) != identity.decoding_hash({**strings, "guided_json": True})


def test_content_sha256_excludes_own_field_and_allows_floats():
    record = {"a": 1.5, "b": [1, 2], "content_sha256": "stale"}
    digest = identity.content_sha256(record)
    assert digest == identity.content_sha256({"b": [1, 2], "a": 1.5})
    assert len(digest) == 64
