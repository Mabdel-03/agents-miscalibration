"""LogprobClient: the OpenAI-compatible wrapper that ALWAYS captures logprobs.

This is the calibration linchpin (plan Risk 1). Two access patterns:

* ``chat(...)``  — normal chat generation; returns text, per-token top-logprobs, and
  token counts. Used for free-form answers and chain-of-thought.
* ``score_options(...)`` — reads the probability mass over a fixed set of answer letters
  (A/B/C/D...) using the *completions* endpoint with a prompt ending in ``"Answer: "``
  and ``max_tokens=1``. This gives a clean per-option distribution for ECE that does not
  depend on the model emitting a parseable letter in chat. We always log the raw
  top-logprobs alongside so nothing is thrown away.

The client points at a vLLM server (``base_url=http://node:port/v1``, ``api_key="EMPTY"``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import backoff
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError


@dataclass
class ChatResult:
    text: str
    # per generated token: list of {token, logprob} for the top-k alternatives
    top_logprobs: list[list[dict[str, Any]]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    # Axis 4: reasoning ("thinking") content emitted before the answer, split out by the
    # vLLM reasoning parser. Empty when thinking is off / model has no thinking mode.
    reasoning_text: str = ""
    reasoning_tokens: int = 0


@dataclass
class OptionScores:
    """Probability mass over answer options, derived from first-token logprobs."""

    # option letter -> normalized probability in [0,1] (sums to 1 over the given options)
    probs: dict[str, float]
    # option letter -> raw logprob seen at the answer position (-inf if not in top-k)
    raw_logprobs: dict[str, float]
    # the full raw top-logprob list at the answer position (for auditing / fallback)
    raw_top: list[dict[str, Any]] = field(default_factory=list)

    @property
    def argmax(self) -> str:
        return max(self.probs, key=self.probs.get)

    @property
    def confidence(self) -> float:
        """p(chosen) = max normalized option probability."""
        return max(self.probs.values()) if self.probs else 0.0


_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)


class LogprobClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        request_timeout: float = 120.0,
        top_logprobs: int = 20,
    ):
        self.model = model
        self.top_logprobs = top_logprobs
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)

    # ------------------------------------------------------------------ chat
    @backoff.on_exception(backoff.expo, _RETRYABLE, max_tries=5, jitter=backoff.full_jitter)
    def chat(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int = 1024,
        seed: int | None = None,
        capture_logprobs: bool = True,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
    ) -> ChatResult:
        """One chat turn.

        Axis 4 (reasoning): ``enable_thinking`` toggles Qwen3's thinking mode via
        ``chat_template_kwargs``; ``thinking_budget`` caps thinking-phase tokens
        (per-request ``thinking_token_budget``; None = no cap). When thinking is on we
        adopt Qwen3's recommended sampling (T=0.6/top_p=0.95/top_k=20) unless an explicit
        temperature is given; thinking generation must NOT be greedy. ``reasoning_content``
        is captured separately from the answer ``content``.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        # Qwen3-recommended sampling differs for thinking vs non-thinking.
        if temperature is None:
            temperature = 0.6 if enable_thinking else 0.7
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "top_k": 20,
            "top_p": 0.95 if enable_thinking else 0.8,
        }
        if enable_thinking and thinking_budget is not None:
            extra_body["thinking_token_budget"] = thinking_budget

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            logprobs=capture_logprobs,
            top_logprobs=self.top_logprobs if capture_logprobs else None,
            extra_body=extra_body,
        )
        choice = resp.choices[0]
        top: list[list[dict[str, Any]]] = []
        if capture_logprobs and choice.logprobs and choice.logprobs.content:
            for tok in choice.logprobs.content:
                top.append(
                    [{"token": t.token, "logprob": t.logprob} for t in (tok.top_logprobs or [])]
                )
        usage = resp.usage
        # reasoning_content is a vLLM/Qwen3 extension; not typed on the OpenAI client.
        reasoning_text = getattr(choice.message, "reasoning_content", None) or ""
        reasoning_tokens = len(reasoning_text.split()) if reasoning_text else 0
        return ChatResult(
            text=choice.message.content or "",
            top_logprobs=top,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            finish_reason=choice.finish_reason,
            reasoning_text=reasoning_text,
            reasoning_tokens=reasoning_tokens,
        )

    # --------------------------------------------------------- option scoring
    @backoff.on_exception(backoff.expo, _RETRYABLE, max_tries=5, jitter=backoff.full_jitter)
    def score_options(self, prompt: str, option_letters: list[str]) -> OptionScores:
        """Read the first-token distribution after ``prompt`` (which should end in
        something like ``"Answer: "``) and renormalize over ``option_letters``.

        Falls back gracefully: a letter not present in the top-k gets logprob -inf
        (prob 0 before renormalization). If *no* option appears in the top-k, all options
        get equal mass and we flag it via an all-equal distribution (caught in QA).
        """
        resp = self._client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=self.top_logprobs,
        )
        lp = resp.choices[0].logprobs
        raw_top: list[dict[str, Any]] = []
        token_logprob: dict[str, float] = {}
        if lp and lp.top_logprobs:
            first = lp.top_logprobs[0] or {}
            for tok, val in first.items():
                raw_top.append({"token": tok, "logprob": val})
                # Match on the stripped token so " A" and "A" both map to "A".
                key = tok.strip()
                # Keep the max logprob seen for a given normalized token.
                if key and (key not in token_logprob or val > token_logprob[key]):
                    token_logprob[key] = val

        raw_logprobs = {opt: token_logprob.get(opt, float("-inf")) for opt in option_letters}
        # Convert to probabilities and renormalize over the option set.
        unnorm = {opt: (math.exp(v) if v != float("-inf") else 0.0) for opt, v in raw_logprobs.items()}
        total = sum(unnorm.values())
        if total <= 0.0:
            # No option in top-k: uniform (flagged downstream by equal mass).
            n = len(option_letters)
            probs = {opt: 1.0 / n for opt in option_letters}
        else:
            probs = {opt: v / total for opt, v in unnorm.items()}
        return OptionScores(probs=probs, raw_logprobs=raw_logprobs, raw_top=raw_top)
