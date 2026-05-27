"""Model registry for the capacity axis (Axis 1).

One family (Qwen3 unified / hybrid-thinking) so that parameter count is the *only* thing
varying — and, crucially, so the SAME weights support the reasoning axis (Axis 4) via the
``enable_thinking`` toggle. Qwen2.5 has no thinking mode, so it was superseded by Qwen3.

Tensor-parallel sizes fit bf16 weights + KV cache on A100-80GB GPUs (``pi_tpoggio`` has
8). bf16 ~= 2 bytes/param, so even 32B (~66 GB) fits on a single 80 GB card. The
``param_count`` field (billions) is the capacity regressor; ``hf_id`` is what vLLM loads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    size: str               # registry key, e.g. "8B"
    hf_id: str              # HuggingFace repo id passed to `vllm serve`
    param_count: float      # billions of parameters (the capacity regressor)
    tp_size: int            # tensor-parallel GPUs needed (1 GPU = 1 A100-80GB)
    max_model_len: int      # context cap; keep KV cache in memory
    supports_reasoning: bool = True  # exposes enable_thinking toggle (all Qwen3 do)


# Qwen3 unified dense ladder. All Apache-2.0, ungated, hybrid-thinking (enable_thinking
# toggle + thinking_token_budget). 32K native context. bf16 fits on 1x A100-80GB through
# 32B (use tp=2 only for long-context / high concurrency). Served with --reasoning-parser
# qwen3 (needs vLLM >= 0.9.0).
QWEN3_LADDER: dict[str, ModelSpec] = {
    "0.6B": ModelSpec("0.6B", "Qwen/Qwen3-0.6B", 0.6, 1, 32768),
    "1.7B": ModelSpec("1.7B", "Qwen/Qwen3-1.7B", 1.7, 1, 32768),
    "4B":   ModelSpec("4B",   "Qwen/Qwen3-4B",   4.0, 1, 32768),
    "8B":   ModelSpec("8B",   "Qwen/Qwen3-8B",   8.2, 1, 32768),
    "14B":  ModelSpec("14B",  "Qwen/Qwen3-14B", 14.8, 1, 32768),
    "32B":  ModelSpec("32B",  "Qwen/Qwen3-32B", 32.8, 1, 32768),
}

# Optional cheap-inference MoE comparison point (not needed to fill a dense gap).
QWEN3_MOE: dict[str, ModelSpec] = {
    "30B-A3B": ModelSpec("30B-A3B", "Qwen/Qwen3-30B-A3B", 30.5, 1, 32768),
}

REGISTRY: dict[str, ModelSpec] = {**QWEN3_LADDER, **QWEN3_MOE}


def get_model(size: str) -> ModelSpec:
    if size not in REGISTRY:
        raise KeyError(
            f"unknown model size {size!r}; known: {sorted(REGISTRY)}"
        )
    return REGISTRY[size]
