"""Analytic dense-Qwen3 FLOP oracle (WP4).

Spec §6.5: ``F_m(L,T) = F_prefill,m(L) + Σ_{t=1..T} F_decode,m(·)``, one multiply-add = two
FLOPs, projections + attention + feed-forward + vocabulary projection under a *published
counting convention*.  Spec §10.2: the oracle must reproduce an independently checked
analytic count for short and long inputs in every architecture and publish its convention,
configuration hashes, reference dimensions and prefill/decode tables.  Architecture
docs/study_v4/01_architecture.md §4 (the closed forms and the constants table this module
verifies at load time).

Counting convention (frozen here; ``FlopOracle.convention()`` publishes it)
* ``C_lin`` — FLOPs per token independent of position:
  ``n·[2·d·(H·hd + 2·Hkv·hd) + 2·(H·hd)·d + 6·d·d_ff] + 2·d·V``
  (q/k/v projections, output projection, SwiGLU gate/up/down, vocabulary projection).
  The embedding lookup is counted as 0; RMSNorm, Q/K norms, rotary embedding and the
  softmax are O(d) per token and omitted.  Tied embeddings do not change the count (the
  vocabulary projection is still executed).
* ``C_attn = n·4·H·hd`` — FLOPs per (query token, attended key) for ``QKᵀ`` and ``PV``.
* The token at 1-indexed position ``p`` attends to ``p`` keys (itself included) and costs
  ``C_lin + C_attn·p``.  Hence ``prefill(L) = C_lin·L + C_attn·L(L+1)/2``,
  ``decode(ctx) = C_lin + C_attn·ctx`` (the cost of the token at position ``ctx``) and
  ``call(L, T) = prefill(L) + Σ_{t=1..T} decode(L+t) = C_lin·S + C_attn·S(S+1)/2`` with
  ``S = L + T``.
* All arithmetic is exact Python ``int``; nothing here is a float.

Reference dimensions come from the pinned snapshot's ``config.json`` under ``HF_HOME``; the
frozen table :data:`QWEN3_ARCH` must agree with it (a mismatch is a configuration failure,
never a silent override).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents_scaling.study.identity import jcs, sha256_hex
from agents_scaling.study.types import Checkpoint, ProtocolError

#: HF cache on the data filesystem (never ``~/.cache``); ``HF_HOME`` overrides.
DEFAULT_HF_HOME = "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface"
ORACLE_VERSION = "dense-qwen3-analytic-v1"
FLOPS_PER_MULTIPLY_ADD = 2


class OracleConfigError(ProtocolError):
    """The pinned ``config.json`` is missing, malformed or disagrees with the frozen table."""


@dataclass(frozen=True)
class DenseArch:
    """Reference dimensions of one dense Qwen3 checkpoint (``config.json`` field names in
    the docstrings of :func:`arch_from_config`)."""

    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    tied_embeddings: bool

    def __post_init__(self) -> None:
        for name in ("n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "d_ff", "vocab"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"DenseArch.{name} must be a positive int, got {value!r}")
        if not isinstance(self.tied_embeddings, bool):
            raise ValueError("DenseArch.tied_embeddings must be a bool")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads (grouped-query attention)")

    # ---- closed-form constants (architecture §4) --------------------------------------
    @property
    def c_lin(self) -> int:
        """FLOPs per token independent of position (see the module docstring)."""
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        qkv = 2 * self.d_model * (q_dim + 2 * kv_dim)
        o_proj = 2 * q_dim * self.d_model
        ffn = 6 * self.d_model * self.d_ff
        vocab = 2 * self.d_model * self.vocab
        return self.n_layers * (qkv + o_proj + ffn) + vocab

    @property
    def c_attn(self) -> int:
        """FLOPs per (query token, attended key): ``n·4·H·hd``."""
        return self.n_layers * 4 * self.n_heads * self.head_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "d_ff": self.d_ff,
            "vocab": self.vocab,
            "tied_embeddings": self.tied_embeddings,
        }


#: Frozen reference dimensions (architecture §1.10/§4); verified against ``config.json``.
QWEN3_ARCH: Mapping[str, DenseArch] = {
    "4B": DenseArch(36, 2560, 32, 8, 128, 9728, 151936, True),
    "8B": DenseArch(36, 4096, 32, 8, 128, 12288, 151936, False),
    "14B": DenseArch(40, 5120, 40, 8, 128, 17408, 151936, False),
    "32B": DenseArch(64, 5120, 64, 8, 128, 25600, 151936, False),
}

#: Hand-checked constants (architecture §4 table, critic §3): ``(C_lin, C_attn)`` exactly.
EXPECTED_CONSTANTS: Mapping[str, tuple[int, int]] = {
    "4B": (8_044_544_000, 589_824),
    "8B": (15_136_194_560, 589_824),
    "14B": (27_979_939_840, 819_200),
    "32B": (63_967_068_160, 2_097_152),
}

_CONFIG_FIELDS: Mapping[str, str] = {
    "n_layers": "num_hidden_layers",
    "d_model": "hidden_size",
    "n_heads": "num_attention_heads",
    "n_kv_heads": "num_key_value_heads",
    "head_dim": "head_dim",
    "d_ff": "intermediate_size",
    "vocab": "vocab_size",
    "tied_embeddings": "tie_word_embeddings",
}


def _check_table() -> None:
    for size, (c_lin, c_attn) in EXPECTED_CONSTANTS.items():
        arch = QWEN3_ARCH[size]
        if (arch.c_lin, arch.c_attn) != (c_lin, c_attn):
            raise AssertionError(f"oracle constants table broken for {size}: {(arch.c_lin, arch.c_attn)}")


_check_table()


# --------------------------------------------------------------------------- config.json


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME") or DEFAULT_HF_HOME)


def arch_config_path(checkpoint: Checkpoint, home: str | os.PathLike | None = None) -> Path:
    """``<HF_HOME>/hub/models--<org>--<name>/snapshots/<model_revision>/config.json``."""
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError("checkpoint must be a Checkpoint")
    repo = "models--" + checkpoint.hf_id.replace("/", "--")
    root = Path(home) if home is not None else hf_home()
    return root / "hub" / repo / "snapshots" / checkpoint.model_revision / "config.json"


def arch_from_config(config: Mapping[str, Any], *, where: str = "config.json") -> DenseArch:
    """Build a :class:`DenseArch` from a parsed HF ``config.json`` (fail-closed on any gap)."""
    if config.get("model_type") != "qwen3":
        raise OracleConfigError(f"{where}: model_type {config.get('model_type')!r} is not 'qwen3' (dense)")
    values: dict[str, Any] = {}
    for field, key in _CONFIG_FIELDS.items():
        if key not in config:
            raise OracleConfigError(f"{where}: missing {key!r}")
        values[field] = config[key]
    try:
        return DenseArch(**values)
    except ValueError as exc:
        raise OracleConfigError(f"{where}: {exc}") from exc


def load_arch(checkpoint: Checkpoint, home: str | os.PathLike | None = None) -> tuple[DenseArch, str]:
    """Read the pinned ``config.json`` → ``(DenseArch, sha256 of the file bytes)``.

    Refuses when the file is absent or when its dimensions differ from :data:`QWEN3_ARCH`
    (the frozen table is the published reference; a divergence means the wrong snapshot).
    """
    path = arch_config_path(checkpoint, home)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OracleConfigError(f"cannot read {path}: {exc}") from exc
    try:
        config = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleConfigError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise OracleConfigError(f"{path}: top level is not an object")
    arch = arch_from_config(config, where=str(path))
    expected = QWEN3_ARCH.get(checkpoint.size)
    if expected is None:
        raise OracleConfigError(f"no frozen reference dimensions for checkpoint size {checkpoint.size!r}")
    if arch != expected:
        raise OracleConfigError(
            f"{path}: dimensions {arch.to_dict()} differ from the frozen table {expected.to_dict()}"
        )
    return arch, hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------- oracle


def _nonneg_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return value


class FlopOracle:
    """``F(L, T)`` for one checkpoint (integers only; see the module docstring).

    ``size`` and ``config_sha256`` are provenance carried into :meth:`convention`; the
    arithmetic depends on ``arch`` alone.
    """

    def __init__(self, arch: DenseArch, *, size: str | None = None, config_sha256: str | None = None) -> None:
        if not isinstance(arch, DenseArch):
            raise TypeError("arch must be a DenseArch")
        self.arch = arch
        self.size = size
        self.config_sha256 = config_sha256
        self.c_lin: int = arch.c_lin
        self.c_attn: int = arch.c_attn

    # ---- constructors --------------------------------------------------------------
    @classmethod
    def from_table(cls, size: str) -> "FlopOracle":
        """The frozen reference dimensions without touching the HF cache (tests, dry runs)."""
        if size not in QWEN3_ARCH:
            raise KeyError(f"unknown checkpoint size {size!r}; frozen sizes: {sorted(QWEN3_ARCH)}")
        return cls(QWEN3_ARCH[size], size=size, config_sha256=None)

    @classmethod
    def from_checkpoint(cls, checkpoint: Checkpoint, home: str | os.PathLike | None = None) -> "FlopOracle":
        """Read and verify the pinned ``config.json`` (the runtime and profiler constructor)."""
        arch, digest = load_arch(checkpoint, home)
        return cls(arch, size=checkpoint.size, config_sha256=digest)

    # ---- closed forms ----------------------------------------------------------------
    def prefill(self, L: int) -> int:
        """``C_lin·L + C_attn·L(L+1)/2`` — the prompt of ``L`` tokens."""
        L = _nonneg_int(L, "L")
        return self.c_lin * L + self.c_attn * (L * (L + 1) // 2)

    def decode(self, ctx: int) -> int:
        """``C_lin + C_attn·ctx`` — the token at 1-indexed position ``ctx`` (attends to ``ctx`` keys)."""
        ctx = _nonneg_int(ctx, "ctx")
        if ctx == 0:
            raise ValueError("decode(ctx) needs ctx >= 1 (a decoded token attends to itself)")
        return self.c_lin + self.c_attn * ctx

    def call(self, L: int, T: int) -> int:
        """``prefill(L) + Σ_{t=1..T} decode(L+t) = C_lin·S + C_attn·S(S+1)/2``, ``S = L+T``."""
        S = _nonneg_int(L, "L") + _nonneg_int(T, "T")
        return self.c_lin * S + self.c_attn * (S * (S + 1) // 2)

    def reservation(self, prompt_tokens: int, max_tokens: int) -> int:
        """§6.5 step 1: known-prompt prefill plus the *full* decode cap."""
        if _nonneg_int(max_tokens, "max_tokens") == 0:
            raise ValueError("a reservation needs max_tokens >= 1")
        return self.call(prompt_tokens, max_tokens)

    def debit(self, prompt_tokens: int, completion_tokens: int) -> int:
        """§6.5 step 3: the actual incurred work of a completed call."""
        return self.call(prompt_tokens, completion_tokens)

    def flops_record(self, prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
        """The ``RequestRecord.flops`` block (WP2 ``CostFn`` shape: prefill/decode/total/oracle)."""
        prefill = self.prefill(prompt_tokens)
        total = self.debit(prompt_tokens, completion_tokens)
        return {
            "prefill": prefill,
            "decode": total - prefill,
            "total": total,
            "oracle": self.oracle_hash,
        }

    # ---- publication -------------------------------------------------------------------
    def convention(self) -> dict[str, Any]:
        """The published counting convention with the reference dimensions and config hash (§10.2)."""
        return {
            "oracle_version": ORACLE_VERSION,
            "size": self.size,
            "config_sha256": self.config_sha256,
            "arch": self.arch.to_dict(),
            "c_lin": self.c_lin,
            "c_attn": self.c_attn,
            "flops_per_multiply_add": FLOPS_PER_MULTIPLY_ADD,
            "formula": {
                "c_lin": "n*[2*d*(H*hd + 2*Hkv*hd) + 2*(H*hd)*d + 6*d*d_ff] + 2*d*V",
                "c_attn": "n*4*H*hd",
                "prefill(L)": "c_lin*L + c_attn*L*(L+1)/2",
                "decode(ctx)": "c_lin + c_attn*ctx  (token at 1-indexed position ctx)",
                "call(L,T)": "c_lin*S + c_attn*S*(S+1)/2 with S = L + T",
                "reservation": "call(prompt_tokens, max_tokens)",
                "debit": "call(prompt_tokens, completion_tokens)",
            },
            "counted": ["q/k/v projections", "output projection", "SwiGLU gate/up/down", "vocabulary projection", "QK^T", "PV"],
            "omitted": ["embedding lookup (0)", "RMSNorm", "Q/K norm", "rotary embedding", "softmax", "sampling"],
            "units": "FLOP (exact integers)",
        }

    @property
    def oracle_hash(self) -> str:
        """SHA-256 of the JCS convention (the ``model_forward_oracle_hash`` of the manifest)."""
        return sha256_hex(jcs(self.convention()))

    def __repr__(self) -> str:
        return f"FlopOracle(size={self.size!r}, c_lin={self.c_lin}, c_attn={self.c_attn})"


# --------------------------------------------------------------------------- helpers for callers


def oracles_from_table(sizes: Mapping[str, Any] | list[str] | tuple[str, ...] | None = None) -> dict[str, FlopOracle]:
    """Oracles for every (or the given) frozen size from the table."""
    keys = list(sizes) if sizes is not None else list(QWEN3_ARCH)
    return {size: FlopOracle.from_table(size) for size in keys}


def oracles_from_checkpoints(checkpoints: Mapping[str, Checkpoint], home: str | os.PathLike | None = None) -> dict[str, FlopOracle]:
    """Verified oracles for every checkpoint of a study config (reads the HF cache)."""
    return {size: FlopOracle.from_checkpoint(ckpt, home) for size, ckpt in checkpoints.items()}


def make_cost_fn(oracles: Mapping[str, FlopOracle]):
    """Adapter for :data:`agents_scaling.study.inference.client.CostFn`.

    ``cost_fn(checkpoint, prompt_tokens, completion_tokens) -> flops mapping``; an unknown
    checkpoint size raises (never a placeholder).
    """

    def cost_fn(checkpoint: Checkpoint, prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
        try:
            oracle = oracles[checkpoint.size]
        except KeyError as exc:
            raise ProtocolError(f"no FLOP oracle for checkpoint size {checkpoint.size!r}") from exc
        return oracle.flops_record(int(prompt_tokens), int(completion_tokens))

    return cost_fn


__all__ = [
    "DEFAULT_HF_HOME",
    "DenseArch",
    "EXPECTED_CONSTANTS",
    "FLOPS_PER_MULTIPLY_ADD",
    "FlopOracle",
    "ORACLE_VERSION",
    "OracleConfigError",
    "QWEN3_ARCH",
    "arch_config_path",
    "arch_from_config",
    "hf_home",
    "load_arch",
    "make_cost_fn",
    "oracles_from_checkpoints",
    "oracles_from_table",
]
