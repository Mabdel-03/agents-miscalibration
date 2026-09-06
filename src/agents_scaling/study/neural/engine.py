"""HF-transformers capture engine: hooked prefill over stored token ids (N1).

Hook convention (frozen here; spec §8.4 "store ... hook placement", brief "Engine")
=====================================================================================
A forward hook is registered on ``model.model.layers[b]`` (``Qwen3DecoderLayer``) and
captures that block's **output** hidden state, i.e. the residual stream *after* block b —
identically the tensor that enters block b+1.  Rationale:

* The spec defines sites as "residual-stream outputs after the unique blocks
  ``floor(f*(L-1))``"; a decoder block's return value *is* the residual stream after both
  its attention and MLP sublayers have been added (``h + attn(norm(h)) + mlp(norm(.))``).
* We do **not** hook the block's *input* (which would be the output of block b-1 and would
  make ``b=0`` the embedding) and we do **not** read the pre-norm ("normalized") activation
  that feeds the attention sublayer: the pre-norm tensor is a per-block RMS-normalised
  projection that is not on the residual stream and is not what later blocks read.
* The final ``model.model.norm`` is *not* applied to captured residuals (it is applied
  only to compute logits).  ``b = L-1`` therefore yields the pre-final-norm residual.
* In transformers 5.x the block returns a bare tensor; 4.x returned a tuple whose first
  element was the hidden state.  Both shapes are handled.

Positions are absolute indices into the *unpadded* sequence.  Sequences are **right
padded** (``attention_mask`` zero on the pad tail) so real tokens keep positions
``0..n-1`` and the causal mask makes them independent of the padding; ``use_cache=False``.
Batches are formed by descending length so that ``rows * max_len <= max_batch_tokens``.
Residuals are moved to CPU as fp16 inside the hook (nothing else is retained), logits are
computed only at the requested positions by applying ``lm_head`` to the post-norm hidden
state (never materialising ``[B, T, vocab]``), and GPU memory is released between batches.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_FRACS: tuple[float, ...] = (0.25, 0.5, 0.75)
DEFAULT_MAX_BATCH_TOKENS = 16384
#: Qwen3 ``max_position_embeddings``; a longer sequence is a harness error, not a truncation.
DEFAULT_MAX_SEQ_TOKENS = 40960
HOOK_CONVENTION = (
    "forward hook on model.model.layers[b]: block OUTPUT = residual stream after block b "
    "(input to block b+1); pre-final-norm; right padding; use_cache=False"
)


class EngineError(RuntimeError):
    """A capture request the engine refuses (bad block, position or length)."""


def blocks_for(num_layers: int, fracs: Sequence[float] = DEFAULT_FRACS) -> tuple[int, ...]:
    """Spec §8.4 sites ``floor(f*(L-1))`` for ``f`` in ``fracs`` (unique, ascending).

    Qwen3-32B (L=64) → (15, 31, 47); Qwen3-8B (L=36) → (8, 17, 26); Qwen3-14B (L=40) → (9, 19, 29).
    """
    if not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError("num_layers must be a positive int")
    out = sorted({int(math.floor(float(f) * (num_layers - 1))) for f in fracs})
    if any(b < 0 or b >= num_layers for b in out):
        raise ValueError(f"fracs {fracs} leave the block range for L={num_layers}")
    return tuple(out)


def plan_batches(lengths: Sequence[int], max_batch_tokens: int) -> list[list[int]]:
    """Greedy length batching: indices sorted by descending length, packed so that
    ``len(batch) * max(len)`` never exceeds ``max_batch_tokens`` (a single sequence longer
    than the budget still forms its own batch — the caller bounds sequence length)."""
    if max_batch_tokens < 1:
        raise ValueError("max_batch_tokens must be >= 1")
    order = sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), i))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for idx in order:
        n = int(lengths[idx])
        if not current:
            current, current_max = [idx], n
            continue
        if (len(current) + 1) * current_max <= max_batch_tokens:
            current.append(idx)
        else:
            batches.append(current)
            current, current_max = [idx], n
    if current:
        batches.append(current)
    return batches


@dataclass(frozen=True)
class CaptureRequest:
    """One sequence to prefill: capture residuals at ``positions`` and (optionally) the
    next-token logits at ``logit_positions`` (absolute indices into ``token_ids``)."""

    seq_id: str
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    logit_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.token_ids)
        if n == 0:
            raise EngineError(f"{self.seq_id}: empty token sequence")
        for name, pos in (("positions", self.positions), ("logit_positions", self.logit_positions)):
            if any((not isinstance(p, int)) or p < 0 or p >= n for p in pos):
                raise EngineError(f"{self.seq_id}: {name} {pos} outside [0, {n})")
            if len(set(pos)) != len(pos):
                raise EngineError(f"{self.seq_id}: duplicate {name}")


@dataclass
class CaptureResult:
    """Residuals ``{block: fp16 [len(positions), hidden]}`` and fp32 logits
    ``[len(logit_positions), vocab]`` (``None`` when none were requested) plus the batch
    accounting that feeds ``ActivationRow.measurement_cost``."""

    seq_id: str
    n_tokens: int
    positions: tuple[int, ...]
    residuals: dict[int, np.ndarray]
    logit_positions: tuple[int, ...]
    logits: np.ndarray | None
    batch_index: int
    batch_rows: int
    batch_tokens: int
    batch_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    def residual_at(self, block: int, position: int) -> np.ndarray:
        return self.residuals[block][self.positions.index(position)]

    def logits_at(self, position: int) -> np.ndarray:
        if self.logits is None:
            raise KeyError("no logits were requested")
        return self.logits[self.logit_positions.index(position)]

    @property
    def measurement_cost(self) -> dict[str, Any]:
        share = 0.0 if self.batch_tokens == 0 else self.batch_seconds * self.n_tokens / self.batch_tokens
        return {
            "sequence_tokens": int(self.n_tokens),
            "batch_index": int(self.batch_index),
            "batch_rows": int(self.batch_rows),
            "batch_tokens": int(self.batch_tokens),
            "batch_seconds": float(self.batch_seconds),
            "share_seconds": float(share),
        }


class CaptureEngine:
    """Load ``AutoModelForCausalLM`` once and prefill stored token sequences with hooks.

    ``snapshot_path`` is a local HF snapshot directory (``local_files_only``); alternatively
    pass a preloaded ``model`` (tests use a two-layer random Qwen3 on CPU).  ``device_map``
    is ``"auto"`` (accelerate sharding across the visible GPUs — the 32B path on 2×A100),
    an explicit map, or a single device string (``"cpu"``, ``"cuda:0"``).  ``dtype`` defaults
    to bf16; captured residuals are stored as fp16 (brief: "bf16→fp16").
    """

    def __init__(
        self,
        snapshot_path: str | os.PathLike | None,
        blocks: Sequence[int],
        device_map: str | Mapping[str, Any] = "auto",
        dtype: Any = None,
        max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
        *,
        model: Any | None = None,
        attn_implementation: str | None = None,
        max_seq_tokens: int = DEFAULT_MAX_SEQ_TOKENS,
        pad_token_id: int = 0,
    ) -> None:
        import torch

        self.torch = torch
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.max_batch_tokens = int(max_batch_tokens)
        self.max_seq_tokens = int(max_seq_tokens)
        self.pad_token_id = int(pad_token_id)
        self.snapshot_path = None if snapshot_path is None else str(snapshot_path)
        self.device_map = device_map
        self.load_seconds = 0.0
        if model is None:
            if snapshot_path is None:
                raise EngineError("CaptureEngine needs a snapshot_path or a preloaded model")
            model = self._load(Path(snapshot_path), device_map, attn_implementation)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        base = getattr(self.model, "model", None)
        layers = getattr(base, "layers", None)
        if base is None or layers is None:
            raise EngineError("model has no .model.layers (expected a Qwen3-style causal LM)")
        self.base = base
        self.layers = layers
        self.num_layers = len(layers)
        self.hidden_size = int(self.model.config.hidden_size)
        blocks = tuple(int(b) for b in blocks)
        if not blocks or any(b < 0 or b >= self.num_layers for b in blocks) or len(set(blocks)) != len(blocks):
            raise EngineError(f"blocks {blocks} must be unique and inside [0, {self.num_layers})")
        self.blocks = tuple(sorted(blocks))
        self._plan: list[tuple[int, ...]] | None = None
        self._captured: dict[int, list[Any]] = {}
        self._handles = [layers[b].register_forward_hook(self._make_hook(b)) for b in self.blocks]
        self.input_device = self.base.embed_tokens.weight.device
        self.batches_run = 0
        self.tokens_run = 0
        self.forward_seconds = 0.0

    # ---- loading ----------------------------------------------------------------------
    def _load(self, path: Path, device_map: str | Mapping[str, Any], attn_implementation: str | None):
        from transformers import AutoModelForCausalLM

        if not (path / "config.json").is_file():
            raise EngineError(f"{path} is not an HF snapshot directory (no config.json)")
        kwargs: dict[str, Any] = {"dtype": self.dtype, "local_files_only": True}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        started = time.time()
        if isinstance(device_map, Mapping) or device_map == "auto":
            kwargs["device_map"] = device_map
            model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
            model = model.to(self.torch.device(device_map))
        self.load_seconds = time.time() - started
        return model

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ---- hooks --------------------------------------------------------------------------
    def _make_hook(self, block: int):
        torch = self.torch

        def hook(_module: Any, _args: Any, output: Any) -> None:
            if self._plan is None:
                return
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            rows: list[Any] = []
            for row, positions in enumerate(self._plan):
                if not positions:
                    rows.append(None)
                    continue
                index = torch.as_tensor(positions, dtype=torch.long, device=hidden.device)
                rows.append(hidden[row].index_select(0, index).to(torch.float16).cpu())
            self._captured[block] = rows

        return hook

    # ---- forward --------------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "snapshot_path": self.snapshot_path,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "blocks": list(self.blocks),
            "dtype": str(self.dtype),
            "device_map": self.device_map if isinstance(self.device_map, str) else dict(self.device_map),
            "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
            "max_batch_tokens": self.max_batch_tokens,
            "hook_convention": HOOK_CONVENTION,
            "padding": "right",
            "use_cache": False,
            "stored_dtype": "float16",
            "load_seconds": self.load_seconds,
        }

    def forward(self, requests: Sequence[CaptureRequest]) -> list[CaptureResult]:
        """Prefill every request (batched by length) and return results in input order."""
        torch = self.torch
        for req in requests:
            if len(req.token_ids) > self.max_seq_tokens:
                raise EngineError(f"{req.seq_id}: {len(req.token_ids)} tokens > max_seq_tokens {self.max_seq_tokens}")
        results: list[CaptureResult | None] = [None] * len(requests)
        lengths = [len(r.token_ids) for r in requests]
        for batch in plan_batches(lengths, self.max_batch_tokens):
            self._run_batch([requests[i] for i in batch], batch, results)
        return [r for r in results if r is not None]

    def _run_batch(self, reqs: list[CaptureRequest], indices: list[int], results: list[CaptureResult | None]) -> None:
        torch = self.torch
        rows = len(reqs)
        max_len = max(len(r.token_ids) for r in reqs)
        input_ids = torch.full((rows, max_len), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((rows, max_len), dtype=torch.long)
        for row, req in enumerate(reqs):
            n = len(req.token_ids)
            input_ids[row, :n] = torch.as_tensor(req.token_ids, dtype=torch.long)
            attention[row, :n] = 1
        input_ids = input_ids.to(self.input_device)
        attention = attention.to(self.input_device)
        self._plan = [tuple(r.positions) for r in reqs]
        self._captured = {}
        cuda = torch.cuda.is_available() and self.input_device.type == "cuda"
        if cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            out = self.base(input_ids=input_ids, attention_mask=attention, use_cache=False)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            logits_rows: list[np.ndarray | None] = []
            head = self.model.lm_head
            for row, req in enumerate(reqs):
                if not req.logit_positions:
                    logits_rows.append(None)
                    continue
                index = torch.as_tensor(req.logit_positions, dtype=torch.long, device=hidden.device)
                picked = hidden[row].index_select(0, index)
                picked = picked.to(head.weight.device).to(head.weight.dtype)
                logits_rows.append(head(picked).float().cpu().numpy())
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        batch_tokens = sum(len(r.token_ids) for r in reqs)
        missing = [b for b in self.blocks if b not in self._captured]
        if missing:
            raise EngineError(f"hooks on blocks {missing} did not fire (device map or model layout changed?)")
        for row, req in enumerate(reqs):
            residuals: dict[int, np.ndarray] = {}
            for b in self.blocks:
                captured = self._captured[b][row]
                residuals[b] = (
                    np.zeros((0, self.hidden_size), dtype=np.float16) if captured is None else captured.numpy()
                )
            results[indices[row]] = CaptureResult(
                seq_id=req.seq_id,
                n_tokens=len(req.token_ids),
                positions=tuple(req.positions),
                residuals=residuals,
                logit_positions=tuple(req.logit_positions),
                logits=logits_rows[row],
                batch_index=self.batches_run,
                batch_rows=rows,
                batch_tokens=batch_tokens,
                batch_seconds=seconds,
            )
        self.batches_run += 1
        self.tokens_run += batch_tokens
        self.forward_seconds += seconds
        self._plan = None
        self._captured = {}
        del out, hidden, input_ids, attention
        if cuda:
            torch.cuda.empty_cache()

    @property
    def tokens_per_second(self) -> float:
        return 0.0 if self.forward_seconds == 0 else self.tokens_run / self.forward_seconds


def snapshot_dir(hf_home: str | os.PathLike, hf_id: str, revision: str) -> Path:
    """``$HF_HOME/hub/models--<org>--<name>/snapshots/<revision>`` (offline cache layout)."""
    return Path(hf_home) / "hub" / ("models--" + hf_id.replace("/", "--")) / "snapshots" / revision


__all__ = [
    "DEFAULT_FRACS",
    "DEFAULT_MAX_BATCH_TOKENS",
    "DEFAULT_MAX_SEQ_TOKENS",
    "HOOK_CONVENTION",
    "CaptureEngine",
    "CaptureRequest",
    "CaptureResult",
    "EngineError",
    "blocks_for",
    "plan_batches",
    "snapshot_dir",
]
