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

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any

import backoff
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError

from agents_scaling.serving.context import (
    CONTEXT_RESERVE_TOKENS,
    ChatTemplateTokenizer,
    ContextCapacityError,
    ContextPreflight,
    build_chat_messages,
    preflight_profile_completion,
    preflight_profile_chat,
    rendered_chat_token_ids,
    tokenizer_for_profile,
)
from agents_scaling.serving.profiles import ServingProfile, get_serving_profile


# vLLM 0.21 implements thinking budgets natively in the sampler: it tracks the last
# Qwen reasoning delimiters and forces the end token when the configured budget is
# exhausted.  The treatment is therefore one chat generation, not a decode/re-encode
# continuation.  Freeze every scientific/runtime assumption into a protocol hash.
THINKING_BUDGET_PROTOCOL_VERSION = 4
QWEN_THINK_START_TOKEN_ID = 151667
QWEN_THINK_END_TOKEN_ID = 151668
QWEN_IM_END_TOKEN_ID = 151645
QWEN_END_OF_TEXT_TOKEN_ID = 151643
ANSWER_GENERATION_TOKEN_ALLOWANCE = 4096
UNLIMITED_THINKING_TOKEN_ALLOWANCE = 8192
LEGACY_ANSWER_GENERATION_TOKEN_ALLOWANCE = 1024
_THINKING_BUDGET_PROTOCOL_SPEC = {
    "version": THINKING_BUDGET_PROTOCOL_VERSION,
    "backend": "vllm-native-thinking-budget",
    "required_vllm_version": "0.21.0",
    "source": "https://docs.vllm.ai/en/v0.21.0/features/reasoning_outputs/#thinking-budget-control",
    "request_endpoint": "/v1/chat/completions",
    "request_budget_field": "thinking_token_budget",
    "reasoning_parser": "qwen3 with server ReasoningConfig enabled",
    "return_token_ids": True,
    "prompt_token_invariant": "server prompt_token_ids exactly equal local tokenized chat template",
    "usage_invariant": "usage prompt/completion counts exactly equal returned token-id lengths",
    "qwen_think_start_token_id": QWEN_THINK_START_TOKEN_ID,
    "qwen_think_end_token_id": QWEN_THINK_END_TOKEN_ID,
    "qwen_im_end_token_id": QWEN_IM_END_TOKEN_ID,
    "reasoning_parse": (
        "completion contains exactly one think-start at index zero and at least one "
        "later think-end; the first think-end closes the reasoning span; later "
        "think-end IDs are parser-visible answer content; exact first-span and "
        "server-parser text agreement"
    ),
    "qwen3_template_contract": (
        "a thinking prompt must not end in unmatched think-start state; closed marker "
        "pairs and stray think-end tokens are safe; delimiter positions are scanned "
        "separately in prompt and completion"
    ),
    "reasoning_delimiter_count_provenance": (
        "start count is generated completion starts; end count includes the first "
        "reasoning close plus any literal think-end IDs in answer content"
    ),
    "finite_sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "request_seed": "manifest/topology-derived deterministic seed",
    "generation_phases": 1,
    "finish_reason_contract": {
        "completed": "stop with exact terminal/parser validation",
        "censored": "length only after exact full output-envelope exhaustion",
    },
    "answer_generation_token_allowance": ANSWER_GENERATION_TOKEN_ALLOWANCE,
    "unlimited_thinking_token_allowance": UNLIMITED_THINKING_TOKEN_ALLOWANCE,
    "output_capacity_floor_policy": (
        "answer allowance + finite budget, or + unlimited allowance; off uses "
        "answer allowance"
    ),
    "max_tokens_policy": (
        "one request using served_context - exact rendered prompt tokens - "
        "context reserve"
    ),
    "truncation_policy": (
        "a length finish at the full safe context envelope is retained as one "
        "auditable censored generation; never retry it or change its seed"
    ),
    "censored_generation_provenance": (
        "one sampling attempt; exact seed, endpoint generation, profile/context, "
        "token counts/hashes/full IDs, decoded completion, server content/reasoning, "
        "prompt/completion delimiter positions, topology or self-consistency role, "
        "question/agent/round/sample coordinates, and Unix creation time"
    ),
    "endpoint_generation_provenance": (
        "required symmetrically on stopped AgentOutput and censored generations"
    ),
    "token_id_hash": "sha256(canonical compact JSON integer array)",
    "context_reserve_tokens": CONTEXT_RESERVE_TOKENS,
    "context_policy": (
        "remaining output capacity must meet the treatment-specific floor before "
        "HTTP; the admitted request exactly fills prompt+max_tokens+128 to the "
        "served context limit"
    ),
}
THINKING_BUDGET_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(
        _THINKING_BUDGET_PROTOCOL_SPEC,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


# This protocol governs *retention* of an observed response that cannot satisfy the
# frozen generation-v4 answer contract.  It is deliberately separate from the thinking
# budget protocol: request construction, sampling parameters, and seeds are unchanged.
# A response-shape censor is data, not a retry signal.
GENERATION_CENSOR_PROTOCOL_VERSION = 1
GENERATION_CENSOR_VIOLATION_CODES = frozenset(
    {
        "alternate_eos_under_v4",
        "premature_reasoning_eos",
        "reasoning_delimiter_contract",
        "reasoning_budget_exceeded",
        "unexpected_finish_reason",
        "unexpected_reasoning_delimiter",
        "unknown_terminal_under_v4",
    }
)
_GENERATION_CENSOR_PROTOCOL_SPEC = {
    "version": GENERATION_CENSOR_PROTOCOL_VERSION,
    "sampling_protocol": {
        "version": THINKING_BUDGET_PROTOCOL_VERSION,
        "hash": THINKING_BUDGET_PROTOCOL_HASH,
    },
    "retention_rule": (
        "after exact prompt/completion token IDs and usage counts validate, retain one "
        "response-shape violation as protocol_censored; replacement sampling forbidden"
    ),
    "violation_codes": sorted(GENERATION_CENSOR_VIOLATION_CODES),
    "frozen_v4_terminal_token_id": QWEN_IM_END_TOKEN_ID,
    "qwen_alternate_eos_token_id": QWEN_END_OF_TEXT_TOKEN_ID,
    "violation_codes_recomputed_from_exact_token_ids": True,
    "full_response_envelope_required": True,
    "parser_field_retention": (
        "preserve exact nullable content/reasoning fields; null and empty string are "
        "distinct and are never coerced"
    ),
    "parser_validation": (
        "before retention, parser fields have registered nullable text types and "
        "exactly agree with the Qwen3 parse recomputed from returned token IDs"
    ),
}
GENERATION_CENSOR_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(
        _GENERATION_CENSOR_PROTOCOL_SPEC,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def thinking_budget_protocol_metadata() -> dict[str, int | str]:
    return {
        "thinking_budget_protocol_version": THINKING_BUDGET_PROTOCOL_VERSION,
        "thinking_budget_protocol_hash": THINKING_BUDGET_PROTOCOL_HASH,
    }


def token_ids_sha256(token_ids: list[int] | tuple[int, ...]) -> str:
    """Compact, deterministic provenance for an exact prompt or generation ID list."""
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def generation_protocol_violation_codes(
    *,
    finish_reason: str | None,
    completion_token_ids: list[int] | tuple[int, ...],
    enable_thinking: bool,
    thinking_budget: int | None,
) -> tuple[str, ...]:
    """Classify exact post-response shape violations under generation protocol v4.

    The result is pure and order-independent so artifact validation can recompute it
    from retained token IDs instead of trusting an exception message.
    """

    completion_ids = tuple(completion_token_ids)
    codes: set[str] = set()
    starts = [
        index
        for index, token_id in enumerate(completion_ids)
        if token_id == QWEN_THINK_START_TOKEN_ID
    ]
    ends = [
        index
        for index, token_id in enumerate(completion_ids)
        if token_id == QWEN_THINK_END_TOKEN_ID
    ]

    if finish_reason != "stop":
        codes.add("unexpected_finish_reason")
    else:
        terminal = completion_ids[-1] if completion_ids else None
        if terminal == QWEN_END_OF_TEXT_TOKEN_ID:
            codes.add("alternate_eos_under_v4")
        elif terminal != QWEN_IM_END_TOKEN_ID:
            codes.add("unknown_terminal_under_v4")

    if enable_thinking:
        if starts == [0] and not ends:
            codes.add("premature_reasoning_eos")
        elif starts != [0] or not ends or ends[0] <= 0:
            codes.add("reasoning_delimiter_contract")
        elif thinking_budget is not None and ends[0] - 1 > thinking_budget:
            codes.add("reasoning_budget_exceeded")
    elif starts or ends:
        codes.add("unexpected_reasoning_delimiter")

    return tuple(sorted(codes))


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
    reasoning_token_source: str = "none"
    # Historical artifacts called a whitespace-word estimate "reasoning_tokens".  Keep
    # that quantity under an honest name for comparisons without using it as the v2
    # token metric.
    reasoning_word_count_legacy: int = 0
    thinking_budget_protocol_version: int = THINKING_BUDGET_PROTOCOL_VERSION
    thinking_budget_protocol_hash: str = THINKING_BUDGET_PROTOCOL_HASH
    thinking_budget_requested: int | None = None
    thinking_budget_saturated: bool = False
    thinking_budget_consumed_tokens: int = 0
    generation_phase_count: int = 1
    generation_phase_finish_reasons: list[str] = field(default_factory=list)
    generation_phase_seeds: list[int] = field(default_factory=list)
    generation_phase_prompt_tokens: list[int] = field(default_factory=list)
    generation_phase_completion_tokens: list[int] = field(default_factory=list)
    output_capacity_floor_tokens: int = ANSWER_GENERATION_TOKEN_ALLOWANCE
    generation_phase_requested_max_tokens: list[int] = field(default_factory=list)
    generation_phase_prompt_token_id_hashes: list[str] = field(default_factory=list)
    generation_phase_completion_token_id_hashes: list[str] = field(default_factory=list)
    endpoint_generation: str | None = None
    reasoning_start_token_index: int | None = None
    reasoning_end_token_index: int | None = None
    reasoning_start_token_count: int = 0
    reasoning_end_token_count: int = 0
    injected_transition_tokens: int = 0


class GenerationTruncationError(RuntimeError):
    """A one-shot generation exhausted its full safe context envelope.

    The exact sampled response is retained as a censored generation.  Retrying it or
    changing its deterministic seed would condition the dataset on successful stopping,
    so this exception deliberately carries everything needed to audit the one attempt.
    Question/topology coordinates are attached by :meth:`enrich` at the Agent boundary.
    """

    def __init__(
        self,
        *,
        finish_reason: str,
        requested_output_tokens: int,
        completion_tokens: int,
        output_capacity_floor_tokens: int | None = None,
        prompt_tokens: int | None = None,
        prompt_token_ids: list[int] | tuple[int, ...] = (),
        completion_token_ids: list[int] | tuple[int, ...] = (),
        decoded_completion: str = "",
        server_content: str | None = None,
        server_reasoning: str | None = None,
        seed: int | None = None,
        serving_profile: str | None = None,
        effective_context_limit: int | None = None,
        context_reserve_tokens: int = CONTEXT_RESERVE_TOKENS,
        prompt_think_start_positions: list[int] | tuple[int, ...] = (),
        prompt_think_end_positions: list[int] | tuple[int, ...] = (),
        completion_think_start_positions: list[int] | tuple[int, ...] = (),
        completion_think_end_positions: list[int] | tuple[int, ...] = (),
        endpoint_generation: str | None = None,
        created_at: float | None = None,
    ) -> None:
        self.finish_reason = finish_reason
        self.requested_output_tokens = requested_output_tokens
        self.completion_tokens = completion_tokens
        self.output_capacity_floor_tokens = (
            requested_output_tokens
            if output_capacity_floor_tokens is None
            else output_capacity_floor_tokens
        )
        self.prompt_token_ids = tuple(prompt_token_ids)
        self.completion_token_ids = tuple(completion_token_ids)
        self.prompt_tokens = (
            len(self.prompt_token_ids) if prompt_tokens is None else prompt_tokens
        )
        self.prompt_token_id_sha256 = token_ids_sha256(self.prompt_token_ids)
        self.completion_token_id_sha256 = token_ids_sha256(self.completion_token_ids)
        self.decoded_completion = decoded_completion
        self.server_content = server_content
        self.server_reasoning = server_reasoning
        self.seed = seed
        self.serving_profile = serving_profile
        self.effective_context_limit = effective_context_limit
        self.context_reserve_tokens = context_reserve_tokens
        self.prompt_think_start_positions = tuple(prompt_think_start_positions)
        self.prompt_think_end_positions = tuple(prompt_think_end_positions)
        self.completion_think_start_positions = tuple(
            completion_think_start_positions
        )
        self.completion_think_end_positions = tuple(completion_think_end_positions)
        self.endpoint_generation = endpoint_generation
        self.created_at = time.time() if created_at is None else created_at
        self.qid: str | None = None
        self.agent_id: str | None = None
        self.round: int | None = None
        self.generation_role: str | None = None
        self.sample_index: int | None = None
        super().__init__(self._message())

    def _message(self) -> str:
        coordinates = ""
        if self.qid is not None or self.agent_id is not None or self.round is not None:
            coordinates = (
                f", qid={self.qid!r}, agent_id={self.agent_id!r}, "
                f"round={self.round!r}, generation_role={self.generation_role!r}, "
                f"sample_index={self.sample_index!r}"
            )
        return (
            "model generation was truncated before a complete answer could be "
            f"published: finish_reason={self.finish_reason!r}, "
            f"requested_output_tokens={self.requested_output_tokens}, "
            f"output_capacity_floor_tokens={self.output_capacity_floor_tokens}, "
            f"completion_tokens={self.completion_tokens}, "
            f"serving_profile={self.serving_profile!r}, "
            f"effective_context_limit={self.effective_context_limit!r}, "
            f"prompt_token_id_sha256={self.prompt_token_id_sha256}, "
            f"completion_token_id_sha256={self.completion_token_id_sha256}"
            f"{coordinates}"
        )

    def enrich(
        self,
        *,
        qid: str,
        agent_id: str,
        round_idx: int,
        generation_role: str = "topology",
        sample_index: int | None = None,
    ) -> GenerationTruncationError:
        """Attach the semantic request coordinates without changing the sample."""
        self.qid = qid
        self.agent_id = agent_id
        self.round = round_idx
        self.generation_role = generation_role
        self.sample_index = sample_index
        self.args = (self._message(),)
        return self

    def to_censored_generation(self) -> dict[str, Any]:
        """Return the stable, JSON-serializable protocol-v4 censorship record."""
        return {
            "reason": "length",
            "finish_reason": self.finish_reason,
            "sampling_attempt_count": 1,
            "thinking_budget_protocol_version": THINKING_BUDGET_PROTOCOL_VERSION,
            "thinking_budget_protocol_hash": THINKING_BUDGET_PROTOCOL_HASH,
            "qid": self.qid,
            "agent_id": self.agent_id,
            "round": self.round,
            "generation_role": self.generation_role,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "serving_profile": self.serving_profile,
            "effective_context_limit": self.effective_context_limit,
            "context_reserve_tokens": self.context_reserve_tokens,
            "prompt_tokens": self.prompt_tokens,
            "requested_output_tokens": self.requested_output_tokens,
            "output_capacity_floor_tokens": self.output_capacity_floor_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_token_id_sha256": self.prompt_token_id_sha256,
            "completion_token_id_sha256": self.completion_token_id_sha256,
            "prompt_token_ids": list(self.prompt_token_ids),
            "completion_token_ids": list(self.completion_token_ids),
            "decoded_completion": self.decoded_completion,
            # Preserve exact nullable parser values.  In particular, null is distinct
            # from an emitted empty string and must never be normalized during replay.
            "server_content": self.server_content,
            "server_reasoning": self.server_reasoning,
            "prompt_think_start_positions": list(
                self.prompt_think_start_positions
            ),
            "prompt_think_end_positions": list(self.prompt_think_end_positions),
            "completion_think_start_positions": list(
                self.completion_think_start_positions
            ),
            "completion_think_end_positions": list(
                self.completion_think_end_positions
            ),
            "endpoint_generation": self.endpoint_generation,
            "created_at": self.created_at,
        }


class GenerationProtocolCensorError(GenerationTruncationError):
    """One exact sampled response violates the frozen answer-shape contract.

    Unlike :class:`ServerResponseProtocolError`, this exception is raised only after the
    response's prompt IDs, completion IDs, and usage counts have been verified.  It is
    therefore an observed scientific outcome and must be durably retained exactly once.
    """

    def __init__(
        self,
        *,
        protocol_violation_codes: list[str] | tuple[str, ...],
        **kwargs: Any,
    ) -> None:
        codes = tuple(sorted(set(protocol_violation_codes)))
        if not codes or any(code not in GENERATION_CENSOR_VIOLATION_CODES for code in codes):
            raise ValueError("protocol_violation_codes must be a non-empty registered set")
        self.protocol_violation_codes = codes
        super().__init__(**kwargs)

    def _message(self) -> str:
        coordinates = ""
        if self.qid is not None or self.agent_id is not None or self.round is not None:
            coordinates = (
                f", qid={self.qid!r}, agent_id={self.agent_id!r}, "
                f"round={self.round!r}, generation_role={self.generation_role!r}, "
                f"sample_index={self.sample_index!r}"
            )
        terminal = self.completion_token_ids[-1] if self.completion_token_ids else None
        return (
            "model generation was retained as a protocol censor: "
            f"violation_codes={list(self.protocol_violation_codes)!r}, "
            f"finish_reason={self.finish_reason!r}, "
            f"completion_tokens={self.completion_tokens}, "
            f"terminal_token_id={terminal!r}, "
            f"prompt_token_id_sha256={self.prompt_token_id_sha256}, "
            f"completion_token_id_sha256={self.completion_token_id_sha256}"
            f"{coordinates}"
        )

    def to_censored_generation(self) -> dict[str, Any]:
        payload = super().to_censored_generation()
        payload["reason"] = "protocol"
        payload["protocol_violation_codes"] = list(self.protocol_violation_codes)
        payload["actual_terminal_token_id"] = (
            self.completion_token_ids[-1] if self.completion_token_ids else None
        )
        payload["generation_censor_protocol_version"] = (
            GENERATION_CENSOR_PROTOCOL_VERSION
        )
        payload["generation_censor_protocol_hash"] = GENERATION_CENSOR_PROTOCOL_HASH
        return payload


class ThinkingBudgetProtocolError(RuntimeError):
    """A local/configuration violation of the native token-level protocol.

    These failures occur before inference or reflect a frozen local contract and remain
    blocked until the code/configuration changes.  An untrustworthy response envelope
    uses :class:`ServerResponseProtocolError` instead; it is also blocked because
    resampling after seeing an unverifiable response would condition admission on
    validation success.
    """


class ServerResponseProtocolError(ThinkingBudgetProtocolError):
    """A completed server response cannot be admitted as a scientific result.

    This class is reserved for envelopes whose exact observation cannot be trusted
    (for example, token-ID, usage, prompt-identity, or parser disagreement).  The
    response is rejected fail-closed and the cell is configuration-blocked pending
    operator repair; automatically drawing a replacement is forbidden.  Exact,
    trustworthy response-shape violations use :class:`GenerationProtocolCensorError`
    and are retained as data instead.
    """


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
_MISSING = object()


class LogprobClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        request_timeout: float = 600.0,
        top_logprobs: int = 20,
        serving_profile: ServingProfile | str | None = None,
        context_tokenizer: ChatTemplateTokenizer | None = None,
        context_reserve_tokens: int = CONTEXT_RESERVE_TOKENS,
        endpoint_generation: str | None = None,
    ):
        # 600s: unlimited-thinking generations plus server-side queueing under load can take
        # several minutes; 120s caused ReadTimeout storms when many cells shared a server.
        self.model = model
        self.top_logprobs = top_logprobs
        if isinstance(serving_profile, str):
            serving_profile = get_serving_profile(serving_profile)
        if serving_profile is not None and serving_profile.served_model_name != model:
            raise ValueError(
                "serving profile/model mismatch: "
                f"profile {serving_profile.name!r} serves "
                f"{serving_profile.served_model_name!r}, client requested {model!r}"
            )
        if context_tokenizer is not None and serving_profile is None:
            raise ValueError("context_tokenizer requires serving_profile")
        if endpoint_generation is not None and (
            not isinstance(endpoint_generation, str) or not endpoint_generation
        ):
            raise ValueError("endpoint_generation must be a non-empty string or None")
        self.serving_profile = serving_profile
        self._context_tokenizer = context_tokenizer
        self._context_reserve_tokens = context_reserve_tokens
        self.endpoint_generation = endpoint_generation
        self.last_context_preflight: ContextPreflight | None = None
        self.last_context_preflights: list[ContextPreflight] = []
        self.last_context_capacity_envelope: ContextPreflight | None = None
        # Keep each client call to one visible transport attempt. Connection recovery
        # happens above this client under the same checkpoint coordinate and seed;
        # successfully returned outcomes are journaled before control leaves the proxy.
        # SDK retries would be invisible to that contract.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            max_retries=0,
        )

    def _protocol_tokenizer(self) -> Any:
        tokenizer = self._context_tokenizer
        if tokenizer is None and self.serving_profile is not None:
            tokenizer = tokenizer_for_profile(self.serving_profile.name)
        if tokenizer is None:
            raise ThinkingBudgetProtocolError(
                "native reasoning generation requires a registered serving profile and exact tokenizer"
            )
        return tokenizer

    @staticmethod
    def _extension_field_with_presence(
        value: Any,
        field: str,
    ) -> tuple[bool, Any]:
        try:
            observed = getattr(value, field, _MISSING)
        except Exception as exc:
            raise ServerResponseProtocolError(
                f"server response field {field!r} could not be read"
            ) from exc
        if observed is not _MISSING and observed is not None:
            return True, observed
        try:
            model_dump = getattr(value, "model_dump", None)
        except Exception as exc:
            raise ServerResponseProtocolError(
                f"server response model_dump for field {field!r} could not be read"
            ) from exc
        if model_dump is None:
            return observed is not _MISSING, (
                None if observed is _MISSING else observed
            )
        if not callable(model_dump):
            raise ServerResponseProtocolError(
                "server response model_dump attribute is not callable"
            )
        try:
            dumped = model_dump()
        except Exception as exc:
            raise ServerResponseProtocolError(
                f"server response model_dump failed while reading field {field!r}"
            ) from exc
        if not isinstance(dumped, dict):
            raise ServerResponseProtocolError(
                "server response model_dump did not return an object: "
                f"observed_type={type(dumped).__name__}"
            )
        if field in dumped:
            return True, dumped[field]
        return observed is not _MISSING, (
            None if observed is _MISSING else observed
        )

    @staticmethod
    def _extension_field(value: Any, field: str) -> Any:
        _, observed = LogprobClient._extension_field_with_presence(value, field)
        return observed

    @staticmethod
    def _require_attribute(value: Any, field: str, *, label: str) -> Any:
        if value is None:
            raise ServerResponseProtocolError(
                f"server response is missing {label}"
            )
        try:
            observed = getattr(value, field)
        except (AttributeError, KeyError) as exc:
            raise ServerResponseProtocolError(
                f"server response is missing {label}"
            ) from exc
        except Exception as exc:
            raise ServerResponseProtocolError(
                f"server response {label} could not be read"
            ) from exc
        if observed is None:
            raise ServerResponseProtocolError(
                f"server response is missing {label}"
            )
        return observed

    @staticmethod
    def _require_single_chat_choice(response: Any) -> Any:
        return LogprobClient._require_single_choice(response, kind="chat")

    @staticmethod
    def _require_single_choice(response: Any, *, kind: str) -> Any:
        choices = LogprobClient._require_attribute(
            response,
            "choices",
            label="choices",
        )
        if not isinstance(choices, list) or len(choices) != 1:
            raise ServerResponseProtocolError(
                f"server response must contain exactly one {kind} choice: "
                f"observed_type={type(choices).__name__}, "
                f"observed_count={len(choices) if isinstance(choices, list) else None}"
            )
        if choices[0] is None:
            raise ServerResponseProtocolError(
                f"server response {kind} choice must be an object"
            )
        return choices[0]

    @staticmethod
    def _require_finish_reason(choice: Any) -> str:
        finish_reason = LogprobClient._require_attribute(
            choice,
            "finish_reason",
            label="choice.finish_reason",
        )
        if not isinstance(finish_reason, str) or not finish_reason:
            raise ServerResponseProtocolError(
                "server choice.finish_reason must be a non-empty string: "
                f"observed_type={type(finish_reason).__name__}"
            )
        return finish_reason

    @staticmethod
    def _require_parser_fields(
        message: Any,
    ) -> tuple[str | None, str | None]:
        if message is None:
            raise ServerResponseProtocolError(
                "server response is missing choice.message"
            )
        content_present, content = LogprobClient._extension_field_with_presence(
            message,
            "content",
        )
        reasoning_present, reasoning = (
            LogprobClient._extension_field_with_presence(message, "reasoning")
        )
        reasoning_content_present, reasoning_content = (
            LogprobClient._extension_field_with_presence(
                message,
                "reasoning_content",
            )
        )
        if not content_present:
            raise ServerResponseProtocolError(
                "server parser content field is absent"
            )
        if content is not None and not isinstance(content, str):
            raise ServerResponseProtocolError(
                "server parser content must be a string or null: "
                f"observed_type={type(content).__name__}"
            )
        if not reasoning_present and not reasoning_content_present:
            raise ServerResponseProtocolError(
                "server parser reasoning field is absent"
            )
        for field, observed in (
            ("reasoning", reasoning),
            ("reasoning_content", reasoning_content),
        ):
            if observed is not None and not isinstance(observed, str):
                raise ServerResponseProtocolError(
                    f"server parser {field} must be a string or null: "
                    f"observed_type={type(observed).__name__}"
                )
        if (
            reasoning is not None
            and reasoning_content is not None
            and reasoning != reasoning_content
        ):
            raise ServerResponseProtocolError(
                "server parser reasoning aliases disagree"
            )
        server_reasoning = (
            reasoning if reasoning is not None else reasoning_content
        )
        return content, server_reasoning

    @staticmethod
    def _require_token_ids(value: Any, field: str, *, label: str) -> tuple[int, ...]:
        observed = LogprobClient._extension_field(value, field)
        if not isinstance(observed, list) or not all(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and token_id >= 0
            for token_id in observed
        ):
            raise ServerResponseProtocolError(
                f"vLLM 0.21 did not return valid {label} token IDs: "
                f"observed_type={type(observed).__name__}"
            )
        return tuple(observed)

    @staticmethod
    def _require_usage_count(usage: Any, field: str, *, expected: int) -> int:
        observed = LogprobClient._require_attribute(
            usage,
            field,
            label=f"usage.{field}",
        )
        if (
            not isinstance(observed, int)
            or isinstance(observed, bool)
            or observed != expected
        ):
            raise ServerResponseProtocolError(
                f"server {field} disagrees with returned exact token IDs: "
                f"expected={expected}, observed={observed!r}"
            )
        return observed

    @staticmethod
    def _decode_token_ids(tokenizer: Any, token_ids: tuple[int, ...]) -> str:
        try:
            decode = getattr(tokenizer, "decode")
        except (AttributeError, KeyError) as exc:
            raise ServerResponseProtocolError(
                "exact reasoning tokenizer does not expose decode()"
            ) from exc
        except Exception as exc:
            raise ServerResponseProtocolError(
                "exact reasoning tokenizer decode() could not be read"
            ) from exc
        if not callable(decode):
            raise ServerResponseProtocolError(
                "exact reasoning tokenizer decode attribute is not callable"
            )
        try:
            decoded = decode(
                list(token_ids),
                skip_special_tokens=False,
            )
        except Exception as exc:
            raise ServerResponseProtocolError(
                "exact reasoning tokenizer failed to decode returned token IDs"
            ) from exc
        if not isinstance(decoded, str):
            raise ServerResponseProtocolError(
                "reasoning tokenizer decode() returned non-text"
            )
        return decoded

    @staticmethod
    def _validate_parser_agreement(
        *,
        tokenizer: Any,
        completion_ids: tuple[int, ...],
        finish_reason: str,
        enable_thinking: bool,
        server_content: str | None,
        server_reasoning: str | None,
    ) -> tuple[str | None, str | None]:
        """Validate vLLM's parser fields against the exact returned token IDs.

        The Qwen3 parser treats the first generated think-end as its split.  If the
        close is absent, output is reasoning-only when thinking is enabled and
        content-only when it is disabled.  These rules are useful even when the shape
        violates generation protocol v4: only a response whose parser representation
        is independently reproducible is safe to retain as a censor.
        """

        # A stopped non-streaming completion includes its terminal token ID even though
        # the parser fields exclude that token.  Other finish reasons have no trusted
        # terminal-token exclusion.
        payload_ids = (
            completion_ids[:-1]
            if finish_reason == "stop" and completion_ids
            else completion_ids
        )
        starts = [
            index
            for index, token_id in enumerate(payload_ids)
            if token_id == QWEN_THINK_START_TOKEN_ID
        ]
        # Qwen3ReasoningParser first strips everything through the first generated
        # think-start, then splits on the first think-end.  Express that parser in
        # token coordinates so special-token text cannot be normalized away.
        parser_ids = payload_ids[starts[0] + 1 :] if starts else payload_ids
        parser_ends = [
            index
            for index, token_id in enumerate(parser_ids)
            if token_id == QWEN_THINK_END_TOKEN_ID
        ]
        if parser_ends:
            first_end = parser_ends[0]
            expected_reasoning = LogprobClient._decode_token_ids(
                tokenizer,
                parser_ids[:first_end],
            )
            decoded_content = LogprobClient._decode_token_ids(
                tokenizer,
                parser_ids[first_end + 1 :],
            )
            expected_content = decoded_content or None
        elif enable_thinking:
            expected_reasoning = LogprobClient._decode_token_ids(
                tokenizer,
                parser_ids,
            )
            expected_content = None
        else:
            expected_reasoning = None
            expected_content = LogprobClient._decode_token_ids(
                tokenizer,
                parser_ids,
            )

        if (
            expected_reasoning != server_reasoning
            or expected_content != server_content
        ):
            raise ServerResponseProtocolError(
                "local token-ID split disagrees with vLLM's parser fields: "
                "local_reasoning_chars="
                f"{len(expected_reasoning) if expected_reasoning is not None else None}, "
                "server_reasoning_chars="
                f"{len(server_reasoning) if server_reasoning is not None else None}, "
                "local_content_chars="
                f"{len(expected_content) if expected_content is not None else None}, "
                "server_content_chars="
                f"{len(server_content) if server_content is not None else None}, "
                f"completion_sha256={token_ids_sha256(completion_ids)}"
            )
        return expected_content, expected_reasoning

    @staticmethod
    def _extract_top_logprobs(
        choice: Any,
        *,
        capture_logprobs: bool,
    ) -> list[list[dict[str, Any]]]:
        if not capture_logprobs:
            return []
        logprobs = LogprobClient._extension_field(choice, "logprobs")
        if logprobs is None:
            return []
        content = LogprobClient._extension_field(logprobs, "content")
        if content is None:
            return []
        if not isinstance(content, list):
            raise ServerResponseProtocolError(
                "server choice.logprobs.content must be a list or null: "
                f"observed_type={type(content).__name__}"
            )
        top: list[list[dict[str, Any]]] = []
        for token_index, token_logprobs in enumerate(content):
            if token_logprobs is None:
                raise ServerResponseProtocolError(
                    "server choice.logprobs.content contains a null token entry: "
                    f"token_index={token_index}"
                )
            alternatives = LogprobClient._extension_field(
                token_logprobs,
                "top_logprobs",
            )
            if alternatives is None:
                alternatives = []
            if not isinstance(alternatives, list):
                raise ServerResponseProtocolError(
                    "server token top_logprobs must be a list or null: "
                    f"token_index={token_index}, "
                    f"observed_type={type(alternatives).__name__}"
                )
            observed_alternatives: list[dict[str, Any]] = []
            for alternative in alternatives:
                token = LogprobClient._require_attribute(
                    alternative,
                    "token",
                    label="top-logprob token",
                )
                logprob = LogprobClient._require_attribute(
                    alternative,
                    "logprob",
                    label="top-logprob value",
                )
                if not isinstance(token, str):
                    raise ServerResponseProtocolError(
                        "server top-logprob token must be a string"
                    )
                if (
                    not isinstance(logprob, (int, float))
                    or isinstance(logprob, bool)
                    or not math.isfinite(float(logprob))
                ):
                    raise ServerResponseProtocolError(
                        "server top-logprob value must be finite numeric data"
                    )
                observed_alternatives.append(
                    {"token": token, "logprob": logprob}
                )
            top.append(observed_alternatives)
        return top

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int = ANSWER_GENERATION_TOKEN_ALLOWANCE,
        seed: int | None = None,
        capture_logprobs: bool = True,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
    ) -> ChatResult:
        """Run one exact, token-audited vLLM 0.21 chat generation."""
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ThinkingBudgetProtocolError(
                "generation requires an explicit non-negative deterministic seed"
            )
        if thinking_budget is not None and (
            not enable_thinking
            or not isinstance(thinking_budget, int)
            or isinstance(thinking_budget, bool)
            or thinking_budget <= 0
        ):
            raise ThinkingBudgetProtocolError(
                "thinking_budget must be a positive integer with thinking enabled"
            )
        output_capacity_floor_tokens = ANSWER_GENERATION_TOKEN_ALLOWANCE
        if enable_thinking:
            output_capacity_floor_tokens += (
                thinking_budget
                if thinking_budget is not None
                else UNLIMITED_THINKING_TOKEN_ALLOWANCE
            )
        if max_tokens != output_capacity_floor_tokens:
            raise ThinkingBudgetProtocolError(
                "generation output-capacity floor violates the frozen allowance "
                f"policy: expected={output_capacity_floor_tokens}, observed={max_tokens}"
            )
        if self.serving_profile is None:
            raise ThinkingBudgetProtocolError(
                "exact generation requires an explicit serving profile"
            )

        messages = build_chat_messages(system, user)
        if temperature is None:
            temperature = 0.6 if enable_thinking else 0.7
        self.last_context_capacity_envelope = None
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "top_k": 20,
            "top_p": 0.95 if enable_thinking else 0.8,
            "return_token_ids": True,
        }
        if thinking_budget is not None:
            extra_body["thinking_token_budget"] = thinking_budget

        tokenizer = self._protocol_tokenizer()
        local_prompt_ids = rendered_chat_token_ids(
            tokenizer,
            messages,
            enable_thinking=enable_thinking,
        )
        local_prompt_start_positions = [
            index
            for index, token_id in enumerate(local_prompt_ids)
            if token_id == QWEN_THINK_START_TOKEN_ID
        ]
        local_prompt_end_positions = [
            index
            for index, token_id in enumerate(local_prompt_ids)
            if token_id == QWEN_THINK_END_TOKEN_ID
        ]
        if enable_thinking and local_prompt_start_positions and (
            not local_prompt_end_positions
            or local_prompt_start_positions[-1] > local_prompt_end_positions[-1]
        ):
            raise ThinkingBudgetProtocolError(
                "local thinking prompt ends in an unmatched Qwen think-start state: "
                f"prompt_tokens={len(local_prompt_ids)}, "
                f"start_positions={local_prompt_start_positions[-8:]}, "
                f"end_positions={local_prompt_end_positions[-8:]}, "
                f"prompt_sha256={token_ids_sha256(local_prompt_ids)}"
            )
        try:
            floor_preflight = preflight_profile_chat(
                self.serving_profile,
                system=system,
                user=user,
                requested_output_tokens=output_capacity_floor_tokens,
                enable_thinking=enable_thinking,
                tokenizer=tokenizer,
                reserve_tokens=self._context_reserve_tokens,
            )
        except ContextCapacityError as exc:
            # Retain the rejected exact accounting for operator diagnostics.  No HTTP
            # request exists when the treatment-specific minimum cannot fit.
            self.last_context_preflight = exc.preflight
            self.last_context_preflights = [exc.preflight]
            raise
        if floor_preflight.prompt_tokens != len(local_prompt_ids):
            raise ThinkingBudgetProtocolError(
                "local prompt-ID render disagrees with context preflight"
            )
        requested_max_tokens = (
            self.serving_profile.max_model_len
            - len(local_prompt_ids)
            - self._context_reserve_tokens
        )
        # The floor preflight above proves this is positive and scientifically adequate.
        # Request the full remaining safe envelope once, preserving the manifest seed.
        capacity_envelope = ContextPreflight(
            profile_name=self.serving_profile.name,
            prompt_tokens=len(local_prompt_ids),
            requested_output_tokens=requested_max_tokens,
            reserve_tokens=self._context_reserve_tokens,
            served_context=self.serving_profile.max_model_len,
        )
        if not capacity_envelope.fits or (
            capacity_envelope.required_tokens != self.serving_profile.max_model_len
        ):
            raise ThinkingBudgetProtocolError(
                "dynamic context-envelope calculation did not reach the exact served "
                "context boundary"
            )
        self.last_context_capacity_envelope = capacity_envelope
        self.last_context_preflight = capacity_envelope
        self.last_context_preflights = [capacity_envelope]

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=requested_max_tokens,
            seed=seed,
            logprobs=capture_logprobs,
            top_logprobs=self.top_logprobs if capture_logprobs else None,
            extra_body=extra_body,
        )
        choice = self._require_single_chat_choice(resp)
        finish_reason = self._require_finish_reason(choice)
        message = self._require_attribute(
            choice,
            "message",
            label="choice.message",
        )
        prompt_ids = self._require_token_ids(
            resp,
            "prompt_token_ids",
            label="prompt",
        )
        completion_ids = self._require_token_ids(
            choice,
            "token_ids",
            label="completion",
        )
        if not completion_ids:
            raise ServerResponseProtocolError(
                "server returned an empty completion token-ID sequence"
            )
        if prompt_ids != local_prompt_ids:
            raise ServerResponseProtocolError(
                "server prompt token IDs disagree with the exact local chat template: "
                f"local_tokens={len(local_prompt_ids)}, "
                f"server_tokens={len(prompt_ids)}, "
                f"local_sha256={token_ids_sha256(local_prompt_ids)}, "
                f"server_sha256={token_ids_sha256(prompt_ids)}"
            )
        usage = self._require_attribute(resp, "usage", label="usage")
        prompt_tokens = self._require_usage_count(
            usage,
            "prompt_tokens",
            expected=len(prompt_ids),
        )
        completion_tokens = self._require_usage_count(
            usage,
            "completion_tokens",
            expected=len(completion_ids),
        )
        if completion_tokens > requested_max_tokens:
            raise ServerResponseProtocolError(
                "server completion token IDs exceed requested max_tokens: "
                f"requested={requested_max_tokens}, observed={completion_tokens}, "
                f"completion_sha256={token_ids_sha256(completion_ids)}"
            )
        completion_start_positions = [
            index
            for index, token_id in enumerate(completion_ids)
            if token_id == QWEN_THINK_START_TOKEN_ID
        ]
        completion_end_positions = [
            index
            for index, token_id in enumerate(completion_ids)
            if token_id == QWEN_THINK_END_TOKEN_ID
        ]
        server_content, server_reasoning = self._require_parser_fields(
            message,
        )
        decoded_completion = self._decode_token_ids(tokenizer, completion_ids)
        violation_codes = generation_protocol_violation_codes(
            finish_reason=finish_reason,
            completion_token_ids=completion_ids,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
        parsed_content, parsed_reasoning = self._validate_parser_agreement(
            tokenizer=tokenizer,
            completion_ids=completion_ids,
            finish_reason=finish_reason,
            enable_thinking=enable_thinking,
            server_content=server_content,
            server_reasoning=server_reasoning,
        )
        # Validate every optional response component before retaining any sampled
        # anomaly.  A malformed logprob envelope must not be hidden by a censor raised
        # earlier in this method.
        top = self._extract_top_logprobs(
            choice,
            capture_logprobs=capture_logprobs,
        )
        if finish_reason != "stop":
            if (
                finish_reason == "length"
                and completion_tokens == requested_max_tokens
            ):
                raise GenerationTruncationError(
                    finish_reason=finish_reason,
                    requested_output_tokens=requested_max_tokens,
                    completion_tokens=completion_tokens,
                    output_capacity_floor_tokens=output_capacity_floor_tokens,
                    prompt_tokens=prompt_tokens,
                    prompt_token_ids=prompt_ids,
                    completion_token_ids=completion_ids,
                    decoded_completion=decoded_completion,
                    server_content=server_content,
                    server_reasoning=server_reasoning,
                    seed=seed,
                    serving_profile=self.serving_profile.name,
                    effective_context_limit=self.serving_profile.max_model_len,
                    context_reserve_tokens=self._context_reserve_tokens,
                    prompt_think_start_positions=local_prompt_start_positions,
                    prompt_think_end_positions=local_prompt_end_positions,
                    completion_think_start_positions=completion_start_positions,
                    completion_think_end_positions=completion_end_positions,
                    endpoint_generation=self.endpoint_generation,
                )

        if violation_codes:
            raise GenerationProtocolCensorError(
                protocol_violation_codes=violation_codes,
                finish_reason=finish_reason,
                requested_output_tokens=requested_max_tokens,
                completion_tokens=completion_tokens,
                output_capacity_floor_tokens=output_capacity_floor_tokens,
                prompt_tokens=prompt_tokens,
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
                decoded_completion=decoded_completion,
                server_content=server_content,
                server_reasoning=server_reasoning,
                seed=seed,
                serving_profile=self.serving_profile.name,
                effective_context_limit=self.serving_profile.max_model_len,
                context_reserve_tokens=self._context_reserve_tokens,
                prompt_think_start_positions=local_prompt_start_positions,
                prompt_think_end_positions=local_prompt_end_positions,
                completion_think_start_positions=completion_start_positions,
                completion_think_end_positions=completion_end_positions,
                endpoint_generation=self.endpoint_generation,
            )

        if not isinstance(parsed_content, str) or (
            enable_thinking and not isinstance(parsed_reasoning, str)
        ):
            raise ServerResponseProtocolError(
                "protocol-valid stopped response lacks complete parser text"
            )

        start_index: int | None = None
        end_index: int | None = None
        saturated = False
        if enable_thinking:
            # Protocol-shape violations have already been retained above, so these
            # coordinates are guaranteed by ``generation_protocol_violation_codes``.
            # vLLM's qwen3 parser closes reasoning at the first generated think-end.
            # A later identical token is literal, parser-visible answer content (seen in
            # valid live generations), not a second reasoning boundary.
            first_end_position = completion_end_positions[0]
            start_index = len(prompt_ids)
            end_index = len(prompt_ids) + first_end_position
            reasoning_tokens = first_end_position - 1
            budget_consumed_tokens = reasoning_tokens
            if thinking_budget is not None:
                saturated = reasoning_tokens == thinking_budget
            reasoning_text = parsed_reasoning
            text = parsed_content
            reasoning_token_source = "vllm_native_token_ids"
        else:
            reasoning_text = ""
            reasoning_tokens = 0
            budget_consumed_tokens = 0
            reasoning_token_source = "none"
            text = parsed_content
        reasoning_word_count_legacy = len(reasoning_text.split())

        return ChatResult(
            text=text,
            top_logprobs=top,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            reasoning_text=reasoning_text,
            reasoning_tokens=reasoning_tokens,
            reasoning_token_source=reasoning_token_source,
            reasoning_word_count_legacy=reasoning_word_count_legacy,
            thinking_budget_requested=thinking_budget,
            thinking_budget_saturated=saturated,
            thinking_budget_consumed_tokens=budget_consumed_tokens,
            generation_phase_count=1,
            generation_phase_finish_reasons=["stop"],
            generation_phase_seeds=[seed],
            generation_phase_prompt_tokens=[prompt_tokens],
            generation_phase_completion_tokens=[completion_tokens],
            output_capacity_floor_tokens=output_capacity_floor_tokens,
            generation_phase_requested_max_tokens=[requested_max_tokens],
            generation_phase_prompt_token_id_hashes=[token_ids_sha256(prompt_ids)],
            generation_phase_completion_token_id_hashes=[
                token_ids_sha256(completion_ids)
            ],
            endpoint_generation=self.endpoint_generation,
            reasoning_start_token_index=start_index,
            reasoning_end_token_index=end_index,
            # These are completion-only generated-delimiter counts. End count includes
            # the first semantic reasoning close and any later literal end IDs retained
            # in answer content. Prompt control tokens never leak into this provenance.
            reasoning_start_token_count=(
                len(completion_start_positions) if enable_thinking else 0
            ),
            reasoning_end_token_count=(
                len(completion_end_positions) if enable_thinking else 0
            ),
            injected_transition_tokens=0,
        )

    # --------------------------------------------------------- option scoring
    @backoff.on_exception(backoff.expo, _RETRYABLE, max_tries=8, max_time=1800, jitter=backoff.full_jitter)
    def score_options(self, prompt: str, option_letters: list[str]) -> OptionScores:
        """Read the first-token distribution after ``prompt`` (which should end in
        something like ``"Answer: "``) and renormalize over ``option_letters``.

        Falls back gracefully: a letter not present in the top-k gets logprob -inf
        (prob 0 before renormalization). If *no* option appears in the top-k, all options
        get equal mass and we flag it via an all-equal distribution (caught in QA).
        """
        # The completions endpoint consumes the literal prompt rather than a rendered
        # chat template.  Count that exact tokenization before touching HTTP, using the
        # endpoint's one-token output request and the same safety reserve as chat.
        if self.serving_profile is not None:
            self.last_context_preflight = preflight_profile_completion(
                self.serving_profile,
                prompt=prompt,
                requested_output_tokens=1,
                tokenizer=self._context_tokenizer,
                reserve_tokens=self._context_reserve_tokens,
            )

        resp = self._client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=self.top_logprobs,
        )
        choice = self._require_single_choice(resp, kind="completion")
        lp = self._extension_field(choice, "logprobs")
        raw_top: list[dict[str, Any]] = []
        token_logprob: dict[str, float] = {}
        top_logprobs = (
            self._extension_field(lp, "top_logprobs")
            if lp is not None
            else None
        )
        if top_logprobs is not None and not isinstance(top_logprobs, list):
            raise ServerResponseProtocolError(
                "completion logprobs.top_logprobs must be a list or null: "
                f"observed_type={type(top_logprobs).__name__}"
            )
        if top_logprobs:
            first = top_logprobs[0]
            if first is None:
                first = {}
            if not isinstance(first, dict):
                raise ServerResponseProtocolError(
                    "first completion top-logprob entry must be an object or null: "
                    f"observed_type={type(first).__name__}"
                )
            for tok, val in first.items():
                if not isinstance(tok, str):
                    raise ServerResponseProtocolError(
                        "completion top-logprob token must be a string"
                    )
                if (
                    not isinstance(val, (int, float))
                    or isinstance(val, bool)
                    or not math.isfinite(float(val))
                    or val > 0.0
                ):
                    raise ServerResponseProtocolError(
                        "completion top-logprob value must be a finite non-positive number"
                    )
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
