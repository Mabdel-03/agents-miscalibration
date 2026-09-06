"""Neural readouts for study_v4 (N1: the HF-transformers capture engine).

Records the spec §8.4/§8.7 residual-stream readouts that vLLM cannot expose, without
touching the frozen generation pipeline: every committed request stores its exact
``prompt_token_ids`` and completion ``token_ids`` (docs/study_v4/05_vllm_response_shape.md),
and spec §8.4 permits "a measurement-only exact-prefix teacher-forced replay".  A separate
``transformers`` engine re-runs prefill over the stored ids and reads the residual stream
at structural anchors (docs/study_v4/09_neural_readouts_brief.md is the authority;
docs/study_v4/09_neural_capture_engine.md documents the conventions chosen here).

Modules
* ``engine``  — :class:`CaptureEngine`: one ``AutoModelForCausalLM`` load, forward hooks on
  ``model.model.layers[b]`` (block OUTPUT = residual stream entering block b+1), length
  batching under ``max_batch_tokens``, fp16 residuals + logits at requested positions.
* ``anchors`` — byte-exact token/offset mapping and the anchor resolution rules
  (NATIVE_PREFILL, GENERATED_k, FINAL_OBJECT_CLOSE; STATE_ANCHOR, TASK_ONLY_ANCHOR).
* ``replay``  — teacher-forced sequences, the exact-identity checks and the §10.5 fidelity
  report (argmax agreement, stored-token log-prob, two-pass logit discrepancy).
* ``storage`` — chunked ``.npz`` + ``.jsonl`` ActivationRow store (spec §8.13), atomic and
  resumable by ``(StateSnapshot_id, block, anchor_kind)``; ``load_stage`` for analysis.
* ``capture`` — the GPU CLI (``python -m agents_scaling.study.neural.capture``).

Only ``engine``/``capture`` need torch; ``anchors``/``storage``/``replay`` import lazily so
the harness env can run the CPU tests and the analysis loader without a GPU build.
"""

from agents_scaling.study.neural.anchors import (  # noqa: F401
    ANCHOR_FINAL_OBJECT_CLOSE,
    ANCHOR_LAST_PREFILL,
    ANCHOR_NATIVE_PREFILL,
    ANCHOR_STATE,
    ANCHOR_TASK_ONLY,
    GENERATED_KS,
    Anchor,
    generated_anchor_kind,
    resolve_native_anchors,
    resolve_report_anchors,
)
from agents_scaling.study.neural.storage import ActivationRow, ShardWriter, load_stage  # noqa: F401

__all__ = [
    "ANCHOR_FINAL_OBJECT_CLOSE",
    "ANCHOR_LAST_PREFILL",
    "ANCHOR_NATIVE_PREFILL",
    "ANCHOR_STATE",
    "ANCHOR_TASK_ONLY",
    "GENERATED_KS",
    "ActivationRow",
    "Anchor",
    "ShardWriter",
    "generated_anchor_kind",
    "load_stage",
    "resolve_native_anchors",
    "resolve_report_anchors",
]
