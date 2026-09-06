"""Teacher-forced replay and the §10.5 fidelity report (N1).

Spec §8.4: "If a final token has not yet been consumed, a measurement-only exact-prefix
teacher-forced replay is permitted and charged."  §10.5 ("hook no-op parity"): the capture
engine must reproduce the serving engine's forward pass on the stored ids.  The brief asks
for, over 20 stored completions: next-token argmax agreement with the stored sampled
tokens, the mean log-prob of the stored tokens, and the max/mean logit discrepancy between
two forward passes of the same input; and for the exact token identity to be asserted
(tokenizer decode of the stored ids == the rendered chat-template text; re-tokenising the
stored messages == the stored ``prompt_token_ids``).

A sampled completion (temperature 0.6, top-p/top-k) does not have to agree with the
argmax, so the agreement rate is a *descriptive* tolerance number (it is expected to be
well below 1 on thinking traces); log-prob of the stored tokens and the two-pass
discrepancy are the parity checks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from agents_scaling.study.neural.anchors import AnchorError, render_chat_text, token_bytes
from agents_scaling.study.neural.engine import CaptureEngine, CaptureRequest

DEFAULT_FIDELITY_TOKENS = 20


class ReplayError(RuntimeError):
    """The stored ids and the tokenizer disagree (exact identity failed)."""


def _fields(record: Any) -> tuple[list[int], list[int], list[dict[str, Any]], dict[str, Any], str]:
    if isinstance(record, Mapping):
        prompt = [int(t) for t in record["prompt_token_ids"]]
        completion = [int(t) for t in (record["response"].get("token_ids") or [])]
        messages = [dict(m) for m in record["messages"]]
        ctk = dict(record.get("chat_template_kwargs") or {})
        rid = str(record.get("request_id", ""))
    else:
        prompt = [int(t) for t in record.prompt_token_ids]
        completion = [int(t) for t in (record.response.get("token_ids") or [])]
        messages = [dict(m) for m in record.messages]
        ctk = dict(record.chat_template_kwargs or {})
        rid = str(record.request_id)
    return prompt, completion, messages, ctk, rid


def teacher_forced_sequence(record: Any, k_max: int | None = None) -> tuple[int, ...]:
    """``prompt_token_ids + token_ids[:k_max]`` (all completion tokens when ``k_max`` is None)."""
    prompt, completion, _, _, _ = _fields(record)
    if not prompt:
        raise ReplayError("record has no prompt tokens")
    if k_max is None:
        k_max = len(completion)
    if k_max < 0 or k_max > len(completion):
        raise ReplayError(f"k_max {k_max} outside [0, {len(completion)}]")
    return tuple(prompt + completion[:k_max])


def check_prompt_identity(record: Any, tokenizer: Any, *, strict: bool = True) -> dict[str, Any]:
    """Exact-identity checks of the stored prompt against the tokenizer.

    * ``decode_matches_render``: ``decode(prompt_token_ids)`` (via per-token bytes) equals
      ``apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
      enable_thinking=...)``;
    * ``retokenize_matches``: ``apply_chat_template(..., tokenize=True)`` equals the stored ids.
    With ``strict`` a failure raises :class:`ReplayError` (never silently regenerated).
    """
    prompt, _, messages, ctk, rid = _fields(record)
    enable_thinking = bool(ctk.get("enable_thinking", False))
    rendered = render_chat_text(tokenizer, messages, enable_thinking)
    decoded = b"".join(token_bytes(tokenizer, prompt)).decode("utf-8")
    decode_ok = decoded == rendered
    ids = tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking, return_dict=False)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    retok_ok = [int(t) for t in ids] == prompt
    out = {
        "request_id": rid,
        "prompt_tokens": len(prompt),
        "decode_matches_render": decode_ok,
        "retokenize_matches": retok_ok,
        "enable_thinking": enable_thinking,
    }
    if strict and not (decode_ok and retok_ok):
        raise ReplayError(f"{rid[:12]}: prompt identity failed {out}")
    return out


def fidelity(record: Any, engine: CaptureEngine, n_tokens: int = DEFAULT_FIDELITY_TOKENS, *, tokenizer: Any | None = None) -> dict[str, Any]:
    """Teacher-force ``prompt + completion[:n]`` twice and report the §10.5 parity numbers.

    Position ``prompt_len - 1 + i`` predicts stored completion token ``i`` (``i < n``).
    Returns argmax agreement rate, mean/min log-prob of the stored tokens, the per-token
    ranks of the stored tokens, and the max/mean absolute logit difference between the two
    passes (over the scored positions).  ``tokenizer`` (optional) adds the identity check.
    """
    prompt, completion, _, _, rid = _fields(record)
    n = min(int(n_tokens), len(completion))
    if n < 1:
        raise ReplayError(f"{rid[:12]}: no completion tokens to score")
    seq = tuple(prompt + completion[:n])
    plen = len(prompt)
    logit_positions = tuple(range(plen - 1, plen - 1 + n))
    request = CaptureRequest(seq_id=rid or "fidelity", token_ids=seq, positions=(plen - 1,), logit_positions=logit_positions)
    first = engine.forward([request])[0]
    second = engine.forward([request])[0]
    assert first.logits is not None and second.logits is not None
    a = first.logits.astype(np.float64)
    b = second.logits.astype(np.float64)
    targets = np.asarray(completion[:n], dtype=np.int64)
    argmax = a.argmax(axis=1)
    agree = (argmax == targets).astype(np.float64)
    shifted = a - a.max(axis=1, keepdims=True)
    logz = np.log(np.exp(shifted).sum(axis=1))
    logp = shifted[np.arange(n), targets] - logz
    ranks = (a > a[np.arange(n), targets][:, None]).sum(axis=1)
    diff = np.abs(a - b)
    out: dict[str, Any] = {
        "request_id": rid,
        "prompt_tokens": plen,
        "scored_tokens": int(n),
        "argmax_agreement": float(agree.mean()),
        "argmax_agree_count": int(agree.sum()),
        "mean_logprob": float(logp.mean()),
        "min_logprob": float(logp.min()),
        "median_rank": float(np.median(ranks)),
        "max_rank": int(ranks.max()),
        "two_pass_max_abs_logit_diff": float(diff.max()),
        "two_pass_mean_abs_logit_diff": float(diff.mean()),
        "two_pass_argmax_agreement": float((argmax == b.argmax(axis=1)).mean()),
        "per_token": [
            {"i": int(i), "stored": int(targets[i]), "argmax": int(argmax[i]), "logprob": float(logp[i]), "rank": int(ranks[i])}
            for i in range(n)
        ],
        "batch_seconds": first.batch_seconds + second.batch_seconds,
    }
    if tokenizer is not None:
        try:
            out["identity"] = check_prompt_identity(record, tokenizer, strict=False)
        except AnchorError as exc:
            out["identity"] = {"error": str(exc)}
    return out


def summarize_fidelity(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate over records: token-weighted agreement, mean log-prob, worst discrepancies."""
    if not reports:
        return {"records": 0}
    scored = sum(int(r["scored_tokens"]) for r in reports)
    agree = sum(int(r["argmax_agree_count"]) for r in reports)
    logp = sum(float(r["mean_logprob"]) * int(r["scored_tokens"]) for r in reports)
    return {
        "records": len(reports),
        "scored_tokens": scored,
        "argmax_agreement": agree / scored if scored else math.nan,
        "mean_logprob": logp / scored if scored else math.nan,
        "min_logprob": min(float(r["min_logprob"]) for r in reports),
        "two_pass_max_abs_logit_diff": max(float(r["two_pass_max_abs_logit_diff"]) for r in reports),
        "two_pass_mean_abs_logit_diff": float(np.mean([float(r["two_pass_mean_abs_logit_diff"]) for r in reports])),
        "identity_failures": sum(
            1 for r in reports
            if isinstance(r.get("identity"), Mapping)
            and not (r["identity"].get("decode_matches_render") and r["identity"].get("retokenize_matches"))
        ),
    }


__all__ = [
    "DEFAULT_FIDELITY_TOKENS",
    "ReplayError",
    "check_prompt_identity",
    "fidelity",
    "summarize_fidelity",
    "teacher_forced_sequence",
]
