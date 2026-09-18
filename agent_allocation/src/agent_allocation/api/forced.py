"""Forced-budget generation (``forced.enabled``): every node is two raw ``POST /completions`` calls.

1. **THINK** -- the model's own chat template rendered from the node's ``[system, user]`` messages, ending
   ``<|im_start|>assistant\\n<think>\\n``; ``logit_bias`` = ``forced.ban_bias`` on ``forced.ban_token_ids``
   (``</think>``, ``<|im_end|>``, ``<|endoftext|>``) so the model can neither close its reasoning nor stop, and
   ``max_tokens`` = ``spec.reasoning_budget``. Valid only when ``usage.completion_tokens == reasoning_budget``,
   ``finish_reason == "length"`` and no ``</think>`` came back; anything else (a provider-side cut: finish ``None`` or
   ``stop`` with fewer tokens, a leaked close, missing usage) is a *retryable* ``ERROR_ENVELOPE`` whose cost is
   settled from usage -- never a completed node.
2. **ANSWER** -- THINK prompt + realized reasoning (``choices[0].reasoning`` + ``choices[0].text``: a server-side
   reasoning parser moves thinking to ``reasoning``) + ``forced.think_close_text``; ``max_tokens`` =
   ``spec.completion_headroom``; no bias. Retried inside ``generate`` up to ``forced.answer_attempts`` times while
   the failure's cost is provably known, so a retry never re-pays THINK.

``ForcedClient.generate`` returns one ``GenerationOutcome`` per node (``contracts.Generator``): the ANSWER's visible
content / finish reasons / generation id; ``usage.reasoning_tokens`` = THINK completion tokens; prompt, completion
tokens and cost summed over every call; ``usage.raw_usage = {"think": ..., "answer": ...}``. ``request_body`` is
``{"think": body, "answer": body | None}`` and ``response_body`` the JSON bytes of ``RAW_LAYOUT`` (below); both go to
``raw/`` unchanged through ``storage.raw``. A failed ANSWER still returns THINK's (known) cost so the ledger is
charged. Only ``RequestContractError`` is raised, always before the first byte is sent.

Raw ``response_body`` layout (``RAW_LAYOUT``)::

    {"layout": "forced/v1",
     "think": <THINK response JSON | text if not JSON | null if no response>,
     "think_headers": {...}, "think_status": "<TransportStatus>", "think_latency_seconds": float,
     "answer": <final ANSWER response JSON | text | null>, "answer_headers": {...} | null,
     "answer_status": "<TransportStatus>" | null,
     "answer_failures": [{"status", "http_status", "body", "latency_seconds", "cost_usd"}, ...]}

``reasoning_text_from_raw`` / ``forced_usage_from_raw`` / ``forced_generation_ids`` read that layout from a stored raw
artifact payload (``storage.raw.read_raw_response``), a decoded ``response_body`` dict, or its bytes.

Config keys consumed: ``forced.{enabled, endpoint, template, template_file, ban_token_ids, ban_bias,
bias_probe_token_id, think_close_text, answer_attempts}``, ``provider.endpoint_tag`` (``endpoint_selection``),
``sampling.*`` (via ``api.request``; ``top_k`` allowed), ``reasoning.text_completion_headroom`` (Gate 0 answer cap).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import anyio
import httpx

from agent_allocation.apikey import redact_headers
from agent_allocation.api.client import DEFAULT_BASE_URL, TRANSPORT_EXCEPTIONS, OpenRouterClient
from agent_allocation.api.normalize import (
    GATEWAY_TIMEOUT_STATUSES, THINK_CLOSE, ParsedTextCompletion, answer_visible_text, classify_exception,
    split_reasoning,
    classify_http_status, completion_status, error_summary, parse_retry_after, parse_text_completion,
)
from agent_allocation.api.request import (
    REQUEST_EXTENSION_KEY, RequestContractError, _check_messages, build_forced_answer_body, build_forced_probe_body,
    build_forced_think_body, build_natural_body, request_coordinates,
)
from agent_allocation.contracts import (
    RETRYABLE_TRANSPORT, EndpointLock, GateCriterion, GateResult, GenerationOutcome,
    GenerationResult, GenerationSpec, TransportStatus, Usage,
)

RAW_LAYOUT = "forced/v1"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_FILES: Mapping[str, str] = {"qwen3.8-27b": "qwen3.8-27b.chat_template.jinja"}
# sha256 of the vendored official template; tests/fixtures/forced/qwen38_template_golden.json pins the same value.
TEMPLATE_SHA256: Mapping[str, str] = {
    "qwen3.8-27b": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"}
XHIGH_INSTRUCTION = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                     "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
                     "clarity in the final answer.")
GENERATION_PROMPT = "<|im_start|>assistant\n<think>\n"
ANSWER_BACKOFF_BASE_SECONDS = 2.0
ANSWER_BACKOFF_CAP_SECONDS = 30.0
ANSWER_RETRY_AFTER_CAP_SECONDS = 60.0


class TemplateError(RequestContractError):
    """The vendored chat template changed, is unknown, or the messages fall outside the rendered subset."""


# ---------------------------------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------------------------------

def forced_enabled(cfg: Mapping[str, Any]) -> bool:
    forced = cfg.get("forced") if isinstance(cfg, Mapping) else None
    return isinstance(forced, Mapping) and forced.get("enabled") is True


@dataclass(frozen=True)
class ForcedSettings:
    template: str
    ban_token_ids: tuple[int, ...]
    ban_bias: int
    bias_probe_token_id: int
    think_close_text: str
    answer_attempts: int
    endpoint_tag: str | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ForcedSettings":
        if not forced_enabled(cfg):
            raise ValueError("forced.enabled is not true: this config selects chat mode")
        f = cfg["forced"]
        if f.get("endpoint") != "completions":
            raise ValueError(f"forced.endpoint must be 'completions', got {f.get('endpoint')!r}")
        template = f.get("template")
        if template not in TEMPLATE_FILES:
            raise ValueError(f"forced.template {template!r} is not vendored ({sorted(TEMPLATE_FILES)})")
        if f.get("template_file") is not None and Path(str(f["template_file"])).name != TEMPLATE_FILES[template]:
            raise ValueError(f"forced.template_file {f['template_file']!r} is not {TEMPLATE_FILES[template]!r}")
        ids = tuple(f.get("ban_token_ids") or ())
        if not ids or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in ids):
            raise ValueError(f"forced.ban_token_ids must be non-empty non-negative integers, got {ids!r}")
        bias = f.get("ban_bias")
        if isinstance(bias, bool) or not isinstance(bias, int) or not -100 <= bias < 0:
            raise ValueError(f"forced.ban_bias must be an integer in [-100, 0), got {bias!r}")
        probe = f.get("bias_probe_token_id")
        if isinstance(probe, bool) or not isinstance(probe, int) or probe < 0:
            raise ValueError(f"forced.bias_probe_token_id must be a token id, got {probe!r}")
        close = f.get("think_close_text")
        if not isinstance(close, str) or THINK_CLOSE not in close:
            raise ValueError(f"forced.think_close_text must contain {THINK_CLOSE!r}, got {close!r}")
        attempts = f.get("answer_attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError(f"forced.answer_attempts must be a positive integer, got {attempts!r}")
        tag = (cfg.get("provider") or {}).get("endpoint_tag")
        return cls(template=template, ban_token_ids=ids, ban_bias=bias, bias_probe_token_id=probe,
                   think_close_text=close, answer_attempts=attempts,
                   endpoint_tag=tag if isinstance(tag, str) and tag else None)


def endpoint_selection(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keyword arguments for ``endpoints.select_standard_endpoint`` from config: forced mode requires
    ``provider.endpoint_tag`` (the shard's endpoint, e.g. ``parasail/fp8``); chat mode keeps the default tag unless
    ``provider.endpoint_tag`` names one."""
    if cfg is None:
        return {}
    tag = (cfg.get("provider") or {}).get("endpoint_tag")
    if forced_enabled(cfg):
        if not isinstance(tag, str) or not tag:
            raise ValueError("forced mode needs provider.endpoint_tag (the shard's exact OpenRouter endpoint tag)")
        return {"preferred_tag": tag, "forced": True}
    return {"preferred_tag": tag} if isinstance(tag, str) and tag else {}


def forced_watchdog_seconds(connect_timeout: float, read_timeout: float, answer_attempts: int, margin: float) -> float:
    """Upper bound on one forced ``generate``: THINK + every ANSWER attempt, each bounded by the client's own
    connect+read timeouts, plus the capped backoff sleeps between ANSWER attempts."""
    calls = 1 + max(1, int(answer_attempts))
    sleeps = max(0, int(answer_attempts) - 1) * max(ANSWER_BACKOFF_CAP_SECONDS, ANSWER_RETRY_AFTER_CAP_SECONDS)
    return calls * (connect_timeout + read_timeout) + sleeps + margin


# ---------------------------------------------------------------------------------------------------------------------
# chat template (pure-Python subset renderer, byte-identical to the vendored jinja for [system?, user...])
# ---------------------------------------------------------------------------------------------------------------------

def template_path(template: str = "qwen3.8-27b") -> Path:
    if template not in TEMPLATE_FILES:
        raise TemplateError(f"no vendored chat template {template!r}")
    return TEMPLATES_DIR / TEMPLATE_FILES[template]


@lru_cache(maxsize=None)
def _verified_template(template: str) -> str:
    digest = hashlib.sha256(template_path(template).read_bytes()).hexdigest()
    if digest != TEMPLATE_SHA256[template]:
        raise TemplateError(f"vendored template {template!r} sha256 {digest} != pinned {TEMPLATE_SHA256[template]}: "
                            "the renderer only reproduces the pinned template")
    return digest


def template_sha256(template: str = "qwen3.8-27b") -> str:
    """The pinned (and verified) sha256 of the vendored template; raises ``TemplateError`` when the file drifted."""
    return _verified_template(template)


def render_chat_prompt(messages: Sequence[Mapping[str, str]], *, template: str = "qwen3.8-27b") -> str:
    """The Qwen3.8 chat template with ``add_generation_prompt`` and the default reasoning effort (xhigh) for an
    optional leading system message followed by user messages; contents are ``trim``-med as the template does.
    Anything else (assistant/tool turns, a late system message, no user turn) raises ``TemplateError``."""
    _verified_template(template)
    checked = _check_messages(messages)
    parts: list[str] = []
    body = checked
    if checked[0]["role"] == "system":
        content = checked[0]["content"].strip()
        parts.append("<|im_start|>system\n" + XHIGH_INSTRUCTION + ("\n\n" + content if content else "")
                     + "<|im_end|>\n")
        body = checked[1:]
    else:
        parts.append("<|im_start|>system\n" + XHIGH_INSTRUCTION + "<|im_end|>\n")
    if not body:
        raise TemplateError("no user query found in messages")
    for message in body:
        if message["role"] != "user":
            raise TemplateError(f"forced rendering supports [system, user...] only, got role {message['role']!r}")
        content = message["content"].strip()
        if content.startswith("<tool_response>") and content.endswith("</tool_response>"):
            raise TemplateError("tool responses are outside the rendered subset")
        parts.append("<|im_start|>user\n" + content + "<|im_end|>\n")
    return "".join(parts) + GENERATION_PROMPT


def answer_prompt(think_prompt: str, reasoning: str, think_close_text: str) -> str:
    return think_prompt + reasoning + think_close_text


# ---------------------------------------------------------------------------------------------------------------------
# raw payload helpers
# ---------------------------------------------------------------------------------------------------------------------

def _decode_json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace") if raw else None


def _forced_body(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, (bytes, bytearray)):
        payload = _decode_json(bytes(payload))
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return None
    if not isinstance(payload, Mapping):
        return None
    if "response_body_text" in payload:                      # a storage.raw artifact payload
        return _forced_body(payload.get("response_body_text") or "")
    return payload if payload.get("layout") == RAW_LAYOUT or "think" in payload else None


def _choice(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            return choices[0]
    return {}


def _choice_text(choice: Mapping[str, Any], key: str) -> str:
    value = choice.get(key)
    return value if isinstance(value, str) else ""


def is_forced_raw(payload: Any) -> bool:
    return _forced_body(payload) is not None


def reasoning_text_from_raw(payload: Any) -> str:
    """The realized THINK text (``reasoning`` + ``text`` of the THINK response) -- exactly what the ANSWER call
    re-sent; ``""`` when the payload is not forced-shaped or THINK has no body."""
    body = _forced_body(payload)
    if body is None:
        return ""
    choice = _choice(body.get("think"))
    return _choice_text(choice, "reasoning") + _choice_text(choice, "text")


def forced_generation_ids(payload: Any) -> tuple[str | None, str | None]:
    """``(think generation id, answer generation id)`` from a forced raw payload."""
    body = _forced_body(payload)
    if body is None:
        return None, None
    ids = []
    for key in ("think", "answer"):
        response = body.get(key)
        gid = response.get("id") if isinstance(response, Mapping) else None
        ids.append(gid if isinstance(gid, str) else None)
    return ids[0], ids[1]


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def forced_usage_from_raw(payload: Any) -> dict[str, Any]:
    """Per-call accounting of one forced node from its raw payload: the THINK and ANSWER usage objects as returned,
    ``reasoning_tokens`` (THINK completion tokens), the THINK finish reason, ANSWER prompt/completion tokens, the
    re-tokenization delta (ANSWER prompt - THINK prompt - reasoning tokens; live runs show 2-3), generation ids and
    the number of internally retried ANSWER failures. Keys are present with ``None`` when unknown."""
    body = _forced_body(payload) or {}
    think, answer = body.get("think"), body.get("answer")
    think_usage = think.get("usage") if isinstance(think, Mapping) and isinstance(think.get("usage"), Mapping) else None
    answer_usage = answer.get("usage") if isinstance(answer, Mapping) and isinstance(answer.get("usage"), Mapping) \
        else None
    reasoning = _int((think_usage or {}).get("completion_tokens"))
    think_prompt = _int((think_usage or {}).get("prompt_tokens"))
    answer_prompt_tokens = _int((answer_usage or {}).get("prompt_tokens"))
    delta = (answer_prompt_tokens - think_prompt - reasoning
             if None not in (answer_prompt_tokens, think_prompt, reasoning) else None)
    think_id, answer_id = forced_generation_ids(body)
    choice = _choice(think)
    finish = choice.get("finish_reason")
    return {
        "think": None if think_usage is None else dict(think_usage),
        "answer": None if answer_usage is None else dict(answer_usage),
        "reasoning_tokens": reasoning,
        "think_finish_reason": finish if isinstance(finish, str) else None,
        "think_prompt_tokens": think_prompt,
        "answer_prompt_tokens": answer_prompt_tokens,
        "answer_completion_tokens": _int((answer_usage or {}).get("completion_tokens")),
        "retokenization_delta": delta,
        "think_generation_id": think_id,
        "answer_generation_id": answer_id,
        "answer_failures": len(body.get("answer_failures") or []),
    }


def is_forced_usage(usage: Usage | None) -> bool:
    """A result produced by ``ForcedClient`` (its ``raw_usage`` is ``{"think": ..., "answer": ...}``)."""
    return usage is not None and isinstance(usage.raw_usage, Mapping) and "think" in usage.raw_usage \
        and "answer" in usage.raw_usage


# ---------------------------------------------------------------------------------------------------------------------
# one HTTP call
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ForcedCall:
    """One raw ``/completions`` exchange as the client classified it."""
    transport_status: TransportStatus
    http_status: int | None
    parsed: ParsedTextCompletion | None                   # 2xx bodies only
    body_json: Any                                        # decoded response body (dict, text, or None)
    headers: Mapping[str, str]
    latency_seconds: float
    retry_after_seconds: float | None
    error_detail: str | None
    cost_usd: Decimal | None                              # confirmed cost (ledger rules); None = unknown

    @property
    def generation_id(self) -> str | None:
        return self.parsed.generation_id if self.parsed else None

    @property
    def usage(self) -> Usage | None:
        return self.parsed.usage if self.parsed else None

    @property
    def cache_status(self) -> str | None:
        return self.headers.get("x-openrouter-cache-status")

    @property
    def text(self) -> str:
        return self.parsed.text if self.parsed else ""

    @property
    def reasoning(self) -> str:
        return self.parsed.reasoning if self.parsed else ""

    @property
    def finish_reason(self) -> str | None:
        return self.parsed.finish_reason if self.parsed else None


def _confirmed_cost(status: TransportStatus, usage: Usage | None, generation_id: str | None,
                    error_detail: str | None) -> Decimal | None:
    # The ledger's rule (runtime.ledger.confirmed_cost), applied per call; imported lazily (runtime imports api).
    from agent_allocation.runtime.ledger import confirmed_cost
    probe = GenerationResult(
        node_id="", attempt_id="", transport_status=status, completion_status=None, generation_id=generation_id,
        observed_model=None, observed_provider=None, finish_reason=None, native_finish_reason=None,
        visible_content=None, reasoning_returned=False, usage=usage, cache_status=None, raw_response_path=None,
        raw_response_sha256=None, latency_seconds=None, http_status=None, error_detail=error_detail)
    return confirmed_cost(probe)


def _sum_optional(values: Sequence[Any]) -> Any:
    if any(v is None for v in values):
        return None
    total = values[0]
    for v in values[1:]:
        total = total + v
    return total


# ---------------------------------------------------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------------------------------------------------

class ForcedClient(OpenRouterClient):
    """``contracts.Generator`` for forced mode. Account endpoints (``key_info``, ``credits``, ``list_endpoints``)
    are inherited; ``generation_metadata`` merges THINK + ANSWER metadata for ANSWER ids this client produced."""

    def __init__(self, *, settings: ForcedSettings, api_key: str | None, base_url: str = DEFAULT_BASE_URL,
                 transport: httpx.AsyncBaseTransport | None = None, connect_timeout: float = 15.0,
                 read_timeout: float = 600.0, answer_backoff_base_seconds: float = ANSWER_BACKOFF_BASE_SECONDS,
                 answer_backoff_cap_seconds: float = ANSWER_BACKOFF_CAP_SECONDS,
                 sleep: Callable[[float], Any] | None = None) -> None:
        super().__init__(api_key=api_key, base_url=base_url, transport=transport, connect_timeout=connect_timeout,
                         read_timeout=read_timeout)
        template_sha256(settings.template)                 # a drifted template refuses to construct the client
        self.settings = settings
        self._backoff_base = max(0.0, float(answer_backoff_base_seconds))
        self._backoff_cap = max(0.0, float(answer_backoff_cap_seconds))
        self._sleep = sleep or anyio.sleep
        self._think_of_answer: dict[str, str] = {}         # ANSWER generation id -> THINK generation id

    def __repr__(self) -> str:
        return f"ForcedClient(base_url={self._base_url!r}, template={self.settings.template!r})"

    # ---- request validation (the scheduler calls this before reserving) -------------------------------------------
    def render(self, messages: Sequence[Mapping[str, str]]) -> str:
        return render_chat_prompt(messages, template=self.settings.template)

    def validate_request(self, spec: GenerationSpec, messages: Sequence[Mapping[str, str]], lock: EndpointLock,
                         sampling: Mapping[str, Any]) -> dict[str, Any]:
        """Build (and discard the ANSWER half of) both bodies; raises ``RequestContractError`` for anything the
        forced contract refuses. Returns the THINK body."""
        prompt = self.render(messages)
        body = build_forced_think_body(spec, prompt, lock, sampling, ban_token_ids=self.settings.ban_token_ids,
                                       ban_bias=self.settings.ban_bias)
        build_forced_answer_body(spec, prompt + self.settings.think_close_text, lock, sampling)
        return body

    # ---- one raw completion ---------------------------------------------------------------------------------------
    async def raw_completion(self, body: Mapping[str, Any], *, lock: EndpointLock,
                             coordinates: Mapping[str, Any] | None = None) -> ForcedCall:
        """One ``POST /completions`` exchange, classified like ``generate`` classifies chat responses. Never raises
        for transport/model failures (``asyncio.CancelledError`` propagates)."""
        content = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = self._client.build_request(
            "POST", f"{self._base_url}/completions", content=content, headers=self._headers(json_body=True),
            extensions={REQUEST_EXTENSION_KEY: dict(coordinates or {})})
        started = time.monotonic()
        try:
            response = await self._client.send(request)
        except TRANSPORT_EXCEPTIONS as exc:
            status = classify_exception(exc)
            detail = self._scrub_text(f"{status.value}: {type(exc).__name__}: {exc}")
            return ForcedCall(status, None, None, None, {}, time.monotonic() - started, None, detail,
                              _confirmed_cost(status, None, None, detail))
        latency = time.monotonic() - started
        headers = redact_headers(dict(response.headers.items()))
        raw = self._scrub_body(response.content)
        http_status = response.status_code
        non_success = classify_http_status(http_status)
        if non_success is not None:
            payload = _decode_json(raw)
            summary_payload = payload if isinstance(payload, Mapping) else (
                {"error": {"message": str(payload)[:500]}} if payload else None)
            detail = self._scrub_text(error_summary(summary_payload, prefix=f"http {http_status}"))
            if http_status in GATEWAY_TIMEOUT_STATUSES:
                detail = self._scrub_text(f"gateway_timeout_may_be_billed {detail}")
            return ForcedCall(non_success, http_status, None, payload, headers, latency,
                              parse_retry_after(headers.get("retry-after")), detail,
                              _confirmed_cost(non_success, None, None, detail))
        parsed = parse_text_completion(raw, headers.get("x-openrouter-cache-status"), lock)
        detail = None if parsed.error_detail is None else self._scrub_text(parsed.error_detail)
        return ForcedCall(parsed.transport_status, http_status, parsed, _decode_json(raw), headers, latency, None,
                          detail, _confirmed_cost(parsed.transport_status, parsed.usage, parsed.generation_id, detail))

    # ---- generation -----------------------------------------------------------------------------------------------
    async def generate(self, spec: GenerationSpec, messages: Sequence[Mapping[str, str]], *, attempt_id: str,
                       lock: EndpointLock, sampling: Mapping[str, Any], forced: bool = True) -> GenerationOutcome:
        """``forced=False`` is the NATURAL reference (calibration only): one unbanned call, see ``_generate_natural``."""
        node_id = spec.node_id
        if not isinstance(attempt_id, str) or not attempt_id.startswith(f"{node_id}:") or \
                attempt_id == f"{node_id}:":
            raise RequestContractError(
                f"attempt_id {attempt_id!r} must be f'{{node_id}}:{{attempt_no}}' for node {node_id}")
        prompt = self.render(messages)
        if not forced:
            return await self._generate_natural(spec, attempt_id, prompt, lock, sampling)
        think_body = build_forced_think_body(spec, prompt, lock, sampling, ban_token_ids=self.settings.ban_token_ids,
                                             ban_bias=self.settings.ban_bias)
        build_forced_answer_body(spec, prompt + self.settings.think_close_text, lock, sampling)   # fail before send
        coords = request_coordinates(spec, attempt_id)
        think = await self.raw_completion(think_body, lock=lock, coordinates={**coords, "phase": "think"})
        if think.transport_status is not TransportStatus.OK:
            return self._outcome(spec, attempt_id, think_body, None, think, None, [],
                                 think.transport_status, think.error_detail)
        problem = self.think_problem(spec, think)
        if problem is not None:
            return self._outcome(spec, attempt_id, think_body, None, think, None, [],
                                 TransportStatus.ERROR_ENVELOPE, problem)
        reasoning = think.reasoning + think.text
        answer_body = build_forced_answer_body(
            spec, answer_prompt(prompt, reasoning, self.settings.think_close_text), lock, sampling)
        failures: list[ForcedCall] = []
        answer: ForcedCall | None = None
        for number in range(1, self.settings.answer_attempts + 1):
            call = await self.raw_completion(answer_body, lock=lock,
                                             coordinates={**coords, "phase": "answer", "answer_try": number})
            if call.transport_status is TransportStatus.OK:
                answer = call
                break
            failures.append(call)
            retryable = call.transport_status in RETRYABLE_TRANSPORT and call.cost_usd is not None
            if not retryable or number == self.settings.answer_attempts:
                break
            await self._sleep(self._answer_delay(number, call.retry_after_seconds))
        if answer is None:
            last = failures[-1]
            return self._outcome(spec, attempt_id, think_body, answer_body, think, None, failures,
                                 last.transport_status, f"forced_answer_failed: {last.error_detail or ''}".strip(),
                                 last_call=last)
        return self._outcome(spec, attempt_id, think_body, answer_body, think, answer, failures, TransportStatus.OK,
                             None)

    async def _generate_natural(self, spec: GenerationSpec, attempt_id: str, prompt: str, lock: EndpointLock,
                                sampling: Mapping[str, Any]) -> GenerationOutcome:
        """One unbanned call from the same ``<think>\n`` prompt; the reasoning/answer split is the pilot's
        ``split_natural``. ``usage.reasoning_tokens`` is the provider's ``completion_tokens_details.reasoning_tokens``
        when reported, else ``completion_tokens`` apportioned by reasoning characters (flagged in ``raw_usage``). A
        call that never closes its reasoning has empty visible content (an invalid answer, not a transport error)."""
        body = build_natural_body(spec, prompt, lock, sampling)
        call = await self.raw_completion(body, lock=lock,
                                         coordinates={**request_coordinates(spec, attempt_id), "phase": "natural"})
        outcome = self._outcome(spec, attempt_id, body, None, call, None, [], call.transport_status,
                                call.error_detail)
        if call.transport_status is not TransportStatus.OK:
            return outcome
        reasoning, visible, closed = split_reasoning(call.reasoning, call.text)
        visible = visible if closed else ""
        usage = outcome.result.usage
        if usage is not None:
            raw = dict((usage.raw_usage or {}).get("think") or {})
            details = raw.get("completion_tokens_details") or {}
            reported = details.get("reasoning_tokens") if isinstance(details, Mapping) else None
            estimated = False
            if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
                reasoning_tokens = reported
            elif usage.completion_tokens is not None:
                total_chars = len(reasoning) + len(visible)
                reasoning_tokens = round(usage.completion_tokens * len(reasoning) / total_chars) if total_chars else 0
                estimated = True
            else:
                reasoning_tokens = None
            usage = replace(usage, reasoning_tokens=reasoning_tokens,
                            raw_usage={**dict(usage.raw_usage or {}), "natural": True, "closed_think": closed,
                                       "reasoning_tokens_estimated": estimated})
        result = replace(outcome.result, visible_content=visible, usage=usage,
                         completion_status=completion_status(visible, call.finish_reason,
                                                             call.parsed.native_finish_reason if call.parsed else None))
        return replace(outcome, result=result)

    def think_problem(self, spec: GenerationSpec, think: ForcedCall) -> str | None:
        """Why a transport-OK THINK is not a valid forced THINK (``None`` = valid)."""
        usage = think.usage
        realized = usage.completion_tokens if usage is not None else None
        finish = (think.finish_reason or "").strip().lower() or None
        reasoning = think.reasoning + think.text
        if realized is None:
            return "forced_think_unverified: THINK usage.completion_tokens missing"
        if THINK_CLOSE in reasoning:
            return f"forced_think_closed: {THINK_CLOSE} in THINK output ({realized}/{spec.reasoning_budget} tokens)"
        if realized != spec.reasoning_budget or finish != "length":
            return (f"forced_think_short: {realized}/{spec.reasoning_budget} tokens, finish_reason={finish} "
                    "(provider-side cut or ignored ban; retry)")
        return None

    def _answer_delay(self, number: int, retry_after: float | None) -> float:
        delay = min(self._backoff_cap, self._backoff_base * (2 ** (number - 1)))
        if retry_after is not None and retry_after > 0:
            delay = max(delay, min(retry_after, ANSWER_RETRY_AFTER_CAP_SECONDS))
        return delay

    def _outcome(self, spec: GenerationSpec, attempt_id: str, think_body: Mapping[str, Any],
                 answer_body: Mapping[str, Any] | None, think: ForcedCall, answer: ForcedCall | None,
                 failures: Sequence[ForcedCall], status: TransportStatus, error_detail: str | None, *,
                 last_call: ForcedCall | None = None) -> GenerationOutcome:
        calls = [think, *failures] + ([answer] if answer is not None else [])
        costs = [c.cost_usd for c in calls]
        cost = _sum_optional(costs)
        final = answer or last_call or think
        if answer is not None:
            generation_id = answer.generation_id
            if generation_id and think.generation_id:
                self._think_of_answer[generation_id] = think.generation_id
        elif len(calls) == 1:
            generation_id = think.generation_id           # THINK alone: its id reconciles the whole attempt
        else:
            # Several exchanges and no ANSWER: no single id covers them; an unknown cost is charged conservatively.
            generation_id = None
        usage = self._usage(think, answer, failures, cost)
        cache = next((c.cache_status for c in calls if (c.cache_status or "").strip().upper() == "HIT"),
                     final.cache_status)
        visible = None
        completion = None
        if answer is not None and status is TransportStatus.OK:
            visible = answer_visible_text(answer.reasoning, answer.text)
            completion = completion_status(visible, answer.finish_reason,
                                           answer.parsed.native_finish_reason if answer.parsed else None)
        reasoning_text = think.reasoning + think.text
        result = GenerationResult(
            node_id=spec.node_id, attempt_id=attempt_id, transport_status=status, completion_status=completion,
            generation_id=generation_id,
            observed_model=final.parsed.observed_model if final.parsed else None,
            observed_provider=final.parsed.observed_provider if final.parsed else None,
            finish_reason=final.finish_reason if final.parsed else None,
            native_finish_reason=final.parsed.native_finish_reason if final.parsed else None,
            visible_content=visible, reasoning_returned=bool(reasoning_text.strip()), usage=usage,
            cache_status=cache, raw_response_path=None, raw_response_sha256=None,
            latency_seconds=sum(c.latency_seconds for c in calls), http_status=final.http_status,
            retry_after_seconds=final.retry_after_seconds if status is not TransportStatus.OK else None,
            error_detail=None if error_detail is None else self._scrub_text(error_detail))
        raw = {
            "layout": RAW_LAYOUT,
            "think": think.body_json,
            "think_headers": dict(think.headers),
            "think_status": think.transport_status.value,
            "think_latency_seconds": think.latency_seconds,
            "answer": answer.body_json if answer is not None else None,
            "answer_headers": dict(answer.headers) if answer is not None else None,
            "answer_status": answer.transport_status.value if answer is not None else None,
            "answer_failures": [{"status": f.transport_status.value, "http_status": f.http_status,
                                 "body": f.body_json, "latency_seconds": f.latency_seconds,
                                 "cost_usd": None if f.cost_usd is None else str(f.cost_usd)} for f in failures],
        }
        response_body = self._scrub_body(json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        return GenerationOutcome(result=result,
                                 request_body={"think": dict(think_body),
                                               "answer": dict(answer_body) if answer_body is not None else None},
                                 response_body=response_body, response_headers=dict(final.headers))

    @staticmethod
    def _usage(think: ForcedCall, answer: ForcedCall | None, failures: Sequence[ForcedCall],
               cost: Decimal | None) -> Usage | None:
        tu = think.usage
        au = answer.usage if answer is not None else None
        if tu is None and au is None and cost is None:
            return None
        parts = [u for u in (tu, au) if u is not None]
        needed = [tu] + ([au] if answer is not None else [])
        known = all(u is not None for u in needed)

        def total(field: str) -> int | None:
            if not known:
                return None
            return _sum_optional([getattr(u, field) for u in needed])

        upstream = _sum_optional([u.upstream_inference_cost_usd for u in parts]) if parts and known else None
        return Usage(
            prompt_tokens=total("prompt_tokens"),
            completion_tokens=total("completion_tokens"),
            reasoning_tokens=tu.completion_tokens if tu is not None else None,
            cached_prompt_tokens=total("cached_prompt_tokens"),
            actual_cost_usd=cost,
            raw_usage={"think": dict(tu.raw_usage) if tu is not None else None,
                       "answer": dict(au.raw_usage) if au is not None else None},
            upstream_inference_cost_usd=upstream,
        )

    # ---- reconciliation -------------------------------------------------------------------------------------------
    async def generation_metadata(self, generation_id: str) -> Mapping[str, Any] | None:
        """For an ANSWER id this client produced: THINK + ANSWER metadata merged (``None`` until both exist), so a
        cost reconciled by the node's single generation id covers both calls. Any other id: the plain metadata."""
        think_id = self._think_of_answer.get(generation_id)
        answer_meta = await super().generation_metadata(generation_id)
        if think_id is None or answer_meta is None:
            return answer_meta
        think_meta = await super().generation_metadata(think_id)
        if think_meta is None:
            return None
        from agent_allocation.api.reconcile import merge_forced_metadata
        return merge_forced_metadata(think_meta, answer_meta)


def make_generation_client(cfg: Mapping[str, Any], *, api_key: str | None,
                           transport: httpx.AsyncBaseTransport | None = None, connect_timeout: float = 15.0,
                           read_timeout: float = 600.0, **forced_kwargs: Any) -> OpenRouterClient:
    """``ForcedClient`` when ``forced.enabled`` else the chat ``OpenRouterClient`` (unchanged)."""
    if forced_enabled(cfg):
        return ForcedClient(settings=ForcedSettings.from_config(cfg), api_key=api_key, transport=transport,
                            connect_timeout=connect_timeout, read_timeout=read_timeout, **forced_kwargs)
    return OpenRouterClient(api_key=api_key, transport=transport, connect_timeout=connect_timeout,
                            read_timeout=read_timeout)


# ---------------------------------------------------------------------------------------------------------------------
# Gate 0: the forced-endpoint preflight
# ---------------------------------------------------------------------------------------------------------------------

GATE0_THINK_TOKENS = 2048
GATE0_BANANA_TOKENS = 12
GATE0_BANANA_MIN = 5
GATE0_CLOSE_TOKENS = 16
GATE0_BANANA_PROMPT = "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
GATE0_MESSAGES: tuple[Mapping[str, str], ...] = (
    {"role": "system", "content": "Solve the supplied task. Return exactly one final answer inside <final></final>."},
    {"role": "user", "content": "TASK\nHow many positive divisors does 7200 have?\n\nOUTPUT REQUIREMENT\n"
                                "Put exactly one integer inside <final>."},
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def think_closed_immediately(reasoning: str, text: str) -> bool:
    """``</think>``-id test verdict: nothing but whitespace before the first ``</think>`` (inline style), or the
    parser consumed the forced closes and returned nothing at all."""
    if THINK_CLOSE in text:
        return not (reasoning + text.partition(THINK_CLOSE)[0]).strip()
    return not reasoning.strip() and not text.strip()


async def forced_endpoint_gate(client: ForcedClient, lock: EndpointLock, cfg: Mapping[str, Any], *,
                               ledger: Any = None, evidence_path: Path | None = None,
                               evidence: str = "reports/GATE0_FORCED_ENDPOINT.json") -> GateResult:
    """Gate 0 (preflight, before any paid dispatch in forced mode). Four checks against the locked endpoint:

    (a) banana test -- ``+100`` on ``forced.bias_probe_token_id`` for 12 tokens must yield >= 5 "banana";
    (b) ``</think>``-id test -- ``+100`` on ``forced.ban_token_ids[0]`` after ``<think>\\n`` must close immediately;
    (c) a forced THINK of 2,048 tokens must realize exactly 2,048 with ``finish_reason == "length"`` and no close;
    (d) the ANSWER after it must contain a ``<final>...</final>``.

    Every call must be transport-OK (identity, cache MISS). With ``ledger`` each call is reserved, marked sent and
    settled like a dispatched attempt (partition ``gate0``). A failed gate means: do not dispatch."""
    settings = ForcedSettings.from_config(cfg)
    sampling = dict(cfg["sampling"])
    headroom = int(cfg["reasoning"]["text_completion_headroom"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    exchanges: list[dict[str, Any]] = []
    criteria: list[GateCriterion] = []
    spent = Decimal("0")

    async def call(name: str, body: dict[str, Any]) -> ForcedCall | None:
        nonlocal spent
        attempt_id = f"gate0:{stamp}:{name}"
        if ledger is not None:
            from agent_allocation.runtime.estimate import input_token_bound
            price_in = Decimal(lock.input_price_per_token)
            price_out = max(Decimal(lock.output_price_per_token),
                            Decimal(lock.internal_reasoning_price_per_token or "0"))
            amount = (Decimal(input_token_bound(len(body["prompt"]))) * price_in
                      + Decimal(int(body["max_tokens"])) * price_out) * Decimal("1.1")
            if not await ledger.reserve(attempt_id=attempt_id, node_id="gate0", partition_id="gate0",
                                        amount_usd=amount):
                exchanges.append({"name": name, "refused_by_ledger": True})
                return None
            await ledger.mark_sent(attempt_id)
        result = await client.raw_completion(body, lock=lock, coordinates={"phase": f"gate0_{name}",
                                                                           "node_id": f"gate0:{name}",
                                                                           "attempt_id": attempt_id,
                                                                           "item_id": "gate0"})
        if ledger is not None:
            if result.cost_usd is not None:
                await ledger.settle(attempt_id, result.cost_usd)
            else:
                await ledger.charge_max(attempt_id)
        spent += result.cost_usd or Decimal("0")
        exchanges.append({"name": name, "request": {k: v for k, v in body.items() if k != "prompt"},
                          "prompt_chars": len(body["prompt"]), "status": result.transport_status.value,
                          "http_status": result.http_status, "error_detail": result.error_detail,
                          "cost_usd": None if result.cost_usd is None else str(result.cost_usd),
                          "response": result.body_json})
        return result

    def ok(result: ForcedCall | None) -> bool:
        return result is not None and result.transport_status is TransportStatus.OK

    def status_of(result: ForcedCall | None) -> str:
        return "refused_by_ledger" if result is None else result.transport_status.value

    # (a) banana
    banana = await call("banana", build_forced_probe_body(lock, sampling, GATE0_BANANA_PROMPT,
                                                          max_tokens=GATE0_BANANA_TOKENS,
                                                          logit_bias={str(settings.bias_probe_token_id): 100}))
    count = (banana.reasoning + banana.text).lower().count("banana") if ok(banana) else 0
    criteria.append(GateCriterion(
        "logit_bias_honoured_banana", json.dumps({"status": status_of(banana), "banana_count": count,
                                                  "text": (banana.text[:80] if banana else None)}),
        f">= {GATE0_BANANA_MIN} 'banana' in {GATE0_BANANA_TOKENS} tokens with +100 on "
        f"{settings.bias_probe_token_id}", ok(banana) and count >= GATE0_BANANA_MIN, evidence))

    # (b) </think>-id
    think_prompt = render_chat_prompt(GATE0_MESSAGES, template=settings.template)
    close = await call("think_close_id", build_forced_probe_body(
        lock, sampling, think_prompt, max_tokens=GATE0_CLOSE_TOKENS,
        logit_bias={str(settings.ban_token_ids[0]): 100}))
    closed = ok(close) and think_closed_immediately(close.reasoning, close.text)
    criteria.append(GateCriterion(
        "think_close_token_id", json.dumps({"status": status_of(close), "reasoning": close.reasoning[:80] if close
                                            else None, "text": close.text[:80] if close else None}),
        f"+100 on {settings.ban_token_ids[0]} closes the reasoning immediately", closed, evidence))

    # (c) forced THINK of exactly 2,048 tokens
    think_body = build_forced_probe_body(lock, sampling, think_prompt, max_tokens=GATE0_THINK_TOKENS,
                                         logit_bias={str(t): settings.ban_bias for t in settings.ban_token_ids})
    think = await call("forced_think", think_body)
    realized = think.usage.completion_tokens if ok(think) and think.usage is not None else None
    reasoning = (think.reasoning + think.text) if ok(think) else ""
    exact = ok(think) and realized == GATE0_THINK_TOKENS and (think.finish_reason or "").lower() == "length" \
        and THINK_CLOSE not in reasoning
    criteria.append(GateCriterion(
        "forced_think_exact_length", json.dumps({"status": status_of(think), "completion_tokens": realized,
                                                 "finish_reason": think.finish_reason if think else None,
                                                 "leaked_close": THINK_CLOSE in reasoning}),
        f"completion_tokens == {GATE0_THINK_TOKENS}, finish_reason length, no {THINK_CLOSE}", exact, evidence))

    # (d) the ANSWER returns a <final>
    has_final = False
    answer_status = "not_run"
    if ok(think):
        answer = await call("answer", build_forced_probe_body(
            lock, sampling, answer_prompt(think_prompt, reasoning, settings.think_close_text), max_tokens=headroom,
            logit_bias=None))
        answer_status = status_of(answer)
        if ok(answer):
            visible = answer_visible_text(answer.reasoning, answer.text)
            has_final = "<final>" in visible and "</final>" in visible
    criteria.append(GateCriterion("answer_returns_final", json.dumps({"status": answer_status,
                                                                      "has_final": has_final}),
                                  "ANSWER after the forced THINK contains <final>...</final>", has_final, evidence))
    passed = all(c.passed for c in criteria)
    if evidence_path is not None:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps({"endpoint": lock.provider_tag, "provider_name": lock.provider_name,
                                             "model": lock.model, "passed": passed, "spent_usd": str(spent),
                                             "exchanges": exchanges, "recorded_at": _utc()},
                                            indent=1, sort_keys=True, default=str))
    return GateResult(gate="gate0", stage="preflight", passed=passed,
                      decision="advance" if passed else "stop_notify", criteria=tuple(criteria), evaluated_at=_utc())
