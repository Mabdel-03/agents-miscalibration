"""resources/oracle.py: closed forms vs hand formulas on all four archs, the constants table,
config.json verification, the published convention (§6.5, §10.2; architecture §4)."""

from __future__ import annotations

import json

import pytest

from agents_scaling.study import types as T
from agents_scaling.study.resources import oracle as O

HAND_CASES = [(1, 1), (100, 10), (6144, 8192)]


def hand_call(arch: O.DenseArch, L: int, T_: int) -> int:
    """Independent count: token at position p costs c_lin + c_attn*p; prefill L then decode T."""
    n, d, H, Hkv, hd, dff, V = arch.n_layers, arch.d_model, arch.n_heads, arch.n_kv_heads, arch.head_dim, arch.d_ff, arch.vocab
    per_layer = 2 * d * (H * hd + 2 * Hkv * hd) + 2 * (H * hd) * d + 6 * d * dff
    c_lin = n * per_layer + 2 * d * V
    c_attn = n * 4 * H * hd
    total = 0
    for p in range(1, L + T_ + 1):
        total += c_lin + c_attn * p
    return total


@pytest.mark.parametrize("size", sorted(O.QWEN3_ARCH))
@pytest.mark.parametrize("L,T_", HAND_CASES)
def test_call_matches_hand_formula(size, L, T_):
    oracle = O.FlopOracle.from_table(size)
    assert oracle.call(L, T_) == hand_call(O.QWEN3_ARCH[size], L, T_)
    # prefill + Σ decode(L+t) decomposition
    assert oracle.call(L, T_) == oracle.prefill(L) + sum(oracle.decode(L + t) for t in range(1, T_ + 1))
    assert isinstance(oracle.call(L, T_), int)


def test_constants_table_exact():
    for size, (c_lin, c_attn) in O.EXPECTED_CONSTANTS.items():
        oracle = O.FlopOracle.from_table(size)
        assert (oracle.c_lin, oracle.c_attn) == (c_lin, c_attn)
    # architecture §4 table in GFLOP/token (2 decimals)
    assert round(O.FlopOracle.from_table("32B").c_lin / 1e9, 2) == 63.97
    assert round(O.FlopOracle.from_table("14B").c_lin / 1e9, 2) == 27.98
    assert round(O.FlopOracle.from_table("8B").c_lin / 1e9, 2) == 15.14
    assert round(O.FlopOracle.from_table("4B").c_lin / 1e9, 2) == 8.04
    # architecture §4 hand check: one 32B root call at (6144, 8192) ≈ 1.13 PFLOP
    assert round(O.FlopOracle.from_table("32B").call(6144, 8192) / 1e15, 2) == 1.13


def test_reservation_debit_and_flops_record():
    oracle = O.FlopOracle.from_table("8B")
    assert oracle.reservation(100, 8192) == oracle.call(100, 8192)
    assert oracle.debit(100, 10) == oracle.call(100, 10)
    assert oracle.debit(100, 10) <= oracle.reservation(100, 8192)
    rec = oracle.flops_record(100, 10)
    assert rec["prefill"] == oracle.prefill(100) and rec["total"] == oracle.call(100, 10)
    assert rec["decode"] == rec["total"] - rec["prefill"] and rec["oracle"] == oracle.oracle_hash
    with pytest.raises(ValueError):
        oracle.reservation(100, 0)
    with pytest.raises(ValueError):
        oracle.call(-1, 1)
    with pytest.raises(ValueError):
        oracle.decode(0)
    with pytest.raises(ValueError):
        oracle.call(True, 1)  # bools are not token counts


def test_convention_is_published_and_hashable():
    oracle = O.FlopOracle.from_table("32B")
    conv = oracle.convention()
    assert conv["flops_per_multiply_add"] == 2 and conv["c_lin"] == oracle.c_lin and conv["arch"]["n_layers"] == 64
    assert "vocabulary projection" in conv["counted"] and "embedding lookup (0)" in conv["omitted"]
    assert len(oracle.oracle_hash) == 64 and oracle.oracle_hash == O.FlopOracle.from_table("32B").oracle_hash
    assert oracle.oracle_hash != O.FlopOracle.from_table("14B").oracle_hash


def test_dense_arch_validation():
    with pytest.raises(ValueError):
        O.DenseArch(0, 1, 1, 1, 1, 1, 1, False)
    with pytest.raises(ValueError):
        O.DenseArch(1, 1, 3, 2, 1, 1, 1, False)  # heads not a multiple of kv heads
    with pytest.raises(ValueError):
        O.DenseArch(1, 1, 1, 1, 1, 1, 1, "no")


def test_arch_from_config_and_load_arch(tmp_path, study_config):
    ckpt = study_config.checkpoints["8B"]
    home = tmp_path / "hf"
    path = O.arch_config_path(ckpt, home)
    assert path == home / "hub" / "models--Qwen--Qwen3-8B" / "snapshots" / ckpt.model_revision / "config.json"
    with pytest.raises(O.OracleConfigError, match="cannot read"):
        O.load_arch(ckpt, home)
    path.parent.mkdir(parents=True)
    good = {
        "model_type": "qwen3",
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    }
    path.write_text(json.dumps(good))
    arch, digest = O.load_arch(ckpt, home)
    assert arch == O.QWEN3_ARCH["8B"] and len(digest) == 64
    oracle = O.FlopOracle.from_checkpoint(ckpt, home)
    assert oracle.config_sha256 == digest and oracle.size == "8B"
    # a divergent snapshot is refused, never silently overridden
    bad = dict(good, intermediate_size=12289)
    path.write_text(json.dumps(bad))
    with pytest.raises(O.OracleConfigError, match="differ from the frozen table"):
        O.load_arch(ckpt, home)
    path.write_text(json.dumps(dict(good, model_type="qwen3_moe")))
    with pytest.raises(O.OracleConfigError, match="not 'qwen3'"):
        O.load_arch(ckpt, home)
    del good["head_dim"]
    path.write_text(json.dumps(good))
    with pytest.raises(O.OracleConfigError, match="missing"):
        O.load_arch(ckpt, home)


def test_real_hf_cache_matches_table(study_config):
    """The pinned snapshots on this machine agree with the frozen table (skip if uncached)."""
    missing = [s for s, c in study_config.checkpoints.items() if not O.arch_config_path(c).exists()]
    if missing:
        pytest.skip(f"HF cache lacks config.json for {missing}")
    oracles = O.oracles_from_checkpoints(study_config.checkpoints)
    for size, oracle in oracles.items():
        assert (oracle.c_lin, oracle.c_attn) == O.EXPECTED_CONSTANTS[size]
        assert oracle.config_sha256 is not None


def test_make_cost_fn(study_config):
    oracles = O.oracles_from_table(["32B"])
    cost_fn = O.make_cost_fn(oracles)
    rec = cost_fn(study_config.checkpoints["32B"], 10, 5)
    assert rec["total"] == oracles["32B"].call(10, 5)
    with pytest.raises(T.ProtocolError):
        cost_fn(study_config.checkpoints["4B"], 10, 5)
