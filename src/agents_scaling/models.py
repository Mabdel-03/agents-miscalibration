"""Model registry for the capacity axis (Axis 1).

One family (Qwen2.5-Instruct) so that parameter count is the *only* thing varying.
Tensor-parallel sizes are chosen to fit bf16 weights + KV cache on A100-80GB GPUs
(``pi_tpoggio`` has 8). The ``param_count`` field (in billions) is the regressor used
in the scaling-law fit; ``hf_id`` is what vLLM loads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    size: str          # registry key, e.g. "7B"
    hf_id: str         # HuggingFace repo id passed to `vllm serve`
    param_count: float # billions of parameters (the capacity regressor)
    tp_size: int       # tensor-parallel GPUs needed (1 GPU = 1 A100-80GB)
    max_model_len: int # context cap; smaller for big models to keep KV cache in memory


# Qwen2.5-Instruct ladder. tp sizing: bf16 ~= 2 bytes/param; a 72B model is ~145 GB of
# weights, so 4x A100-80GB (320 GB) leaves room for KV cache. 32B (~64 GB) needs 2x.
QWEN25_LADDER: dict[str, ModelSpec] = {
    "0.5B": ModelSpec("0.5B", "Qwen/Qwen2.5-0.5B-Instruct", 0.5, 1, 32768),
    "1.5B": ModelSpec("1.5B", "Qwen/Qwen2.5-1.5B-Instruct", 1.5, 1, 32768),
    "3B":   ModelSpec("3B",   "Qwen/Qwen2.5-3B-Instruct",   3.0, 1, 32768),
    "7B":   ModelSpec("7B",   "Qwen/Qwen2.5-7B-Instruct",   7.0, 1, 32768),
    "14B":  ModelSpec("14B",  "Qwen/Qwen2.5-14B-Instruct", 14.0, 1, 32768),
    "32B":  ModelSpec("32B",  "Qwen/Qwen2.5-32B-Instruct", 32.0, 2, 16384),
    "72B":  ModelSpec("72B",  "Qwen/Qwen2.5-72B-Instruct", 72.0, 4, 16384),
}

# Optional cross-family robustness check (replicate a couple of points on Llama-3.1).
LLAMA31_CHECK: dict[str, ModelSpec] = {
    "L8B":  ModelSpec("L8B",  "meta-llama/Llama-3.1-8B-Instruct",   8.0, 1, 32768),
    "L70B": ModelSpec("L70B", "meta-llama/Llama-3.1-70B-Instruct", 70.0, 4, 16384),
}

REGISTRY: dict[str, ModelSpec] = {**QWEN25_LADDER, **LLAMA31_CHECK}


def get_model(size: str) -> ModelSpec:
    if size not in REGISTRY:
        raise KeyError(
            f"unknown model size {size!r}; known: {sorted(REGISTRY)}"
        )
    return REGISTRY[size]
