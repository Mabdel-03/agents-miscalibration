"""In-process OpenRouter fake shared by every agent's tests. No network, no key, no money.

``fake_openrouter_transport(script)`` returns an ``httpx.MockTransport`` for ``OpenRouterClient(transport=...)`` that
serves OpenRouter-shaped JSON on:

* ``POST .../chat/completions``           -- completion or injected fault (see ``Fault``)
* ``GET  .../key``                        -- ``{"data": {label, limit, limit_remaining, usage, ...}}``
* ``GET  .../credits``                    -- ``{"data": {total_credits, total_usage}}``
* ``GET  .../generation?id=``             -- generation metadata (404 until ``metadata_lag_polls`` polls have passed)
* ``GET  .../models/{author}/{slug}/endpoints`` -- endpoints listing shaped like the saved 2026-09-17 snapshot

How a request is matched to a script entry: ``OpenRouterClient.generate`` attaches the node coordinates
(``item_id``, ``kind``, ``root_slot``, ``update_round``, ``node_id``, ``attempt_id``, ``benchmark``, ...) as the httpx
request extension ``api.request.REQUEST_EXTENSION_KEY``. Extensions reach the transport object only; they are never
serialized onto the wire. Precedence per completion call: ``script.hook(call)`` (if it returns a reply) ->
``script.queue`` (FIFO, any node) -> ``script.by_node[node_id]`` -> the most specific ``script.replies`` key
``(item_id, kind, root_slot, update_round)`` where ``None`` is a wildcard (ties: first inserted) -> ``script.default``.
A sequence value is indexed by the node's served-request number ``FakeCall.served_no`` (1st served request -> element
0, ...; the last element repeats), so ``[FakeReply(fault=Fault.RATE_LIMITED), FakeReply()]`` fails attempt 1 and
succeeds on the retry. Requests refused before the script is consulted -- contract-violation 400s, emulated cache HITs,
account-limit 403/402s and capacity 429s -- never consume a scripted reply.

Default reply: HTTP 200, ``x-openrouter-cache-status: MISS``, provider ``script.provider_name``, the request's model,
``finish_reason "stop"``/``native_finish_reason "STOP"``, a reasoning summary, full usage with ``cost``,
``completion_tokens_details.reasoning_tokens`` (inside ``completion_tokens``, never added on top) and
``prompt_tokens_details.cached_tokens``, and content ``"<rationale>...</rationale>\\n<final>...</final>"`` whose final
answer is derived from the item id (``vary_final_by_node=True`` derives it from the node id instead):
math500 -> an integer, mmlu_pro / bbh multiple choice -> a letter A-C, bbh boolean_expressions -> True/False,
mbpp_plus -> a stub function named after the first ``assert name(`` in the prompt.

Field values in ``FakeReply``: ``AUTO`` = the realistic default, ``OMIT`` = leave the JSON key out, anything else is
used verbatim (``None`` becomes JSON ``null``). The fake also acts as a request-contract checker: a body outside the
contract (streaming, seed, tools, fallbacks, unknown keys, ...) gets HTTP 400 and is appended to
``script.violations``. Omitting ``X-OpenRouter-Cache: false`` turns on response-cache emulation (identical bodies
return the first response verbatim with ``x-openrouter-cache-status: HIT``).

Recorded state for assertions: ``script.calls`` (every request, credentials redacted; ``completion_calls(
reached_server=True)`` excludes scripted connect failures), ``script.billed_usd`` (what the fake "charged", including
requests whose response the client never saw: read timeouts, remote disconnects and client cancellations of a
generation in progress), ``script.generations`` (metadata by id; a ``Fault.CACHE_HIT`` carries a
``response_cache_source_id``, an emulated cache HIT replays the original body and id), ``script.peak_in_flight``
(server-side concurrency actually reached). ``max_concurrent_requests`` emulates upstream capacity (HTTP 429 with
``capacity_retry_after``); with ``enforce_account_limits`` (default) a request arriving when usage has reached
``key_limit_usd`` gets 403 "Key limit exceeded", and when it has reached ``total_credits_usd`` 402.

Shortcuts: ``fake_generation_outcome(spec, reply=FakeReply(...))`` (async) and ``fake_generation_outcome_sync`` return
one realistic ``GenerationOutcome`` through the real client, for tests that only need result records.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from collections import Counter
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import httpx

from agent_allocation.apikey import redact_headers
from agent_allocation.api.request import REQUEST_EXTENSION_KEY
from agent_allocation.contracts import EndpointLock, GenerationOutcome, GenerationSpec


class _Sentinel:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __reduce__(self) -> str:
        return self.name


AUTO: Any = _Sentinel("AUTO")
OMIT: Any = _Sentinel("OMIT")

FAKE_PROVIDER_NAME = "Google AI Studio"
FAKE_CREATED_EPOCH = 1758092400  # 2026-09-17T07:00:00Z; deterministic "created" timestamps


class Fault(str, Enum):
    RATE_LIMITED = "rate_limited"            # HTTP 429 + Retry-After (FakeReply.retry_after)
    PAYMENT_REQUIRED = "payment_required"    # HTTP 402
    AUTH_FAILED = "auth_failed"              # HTTP 401
    FORBIDDEN = "forbidden"                  # HTTP 403
    BAD_REQUEST = "bad_request"              # HTTP 400
    NOT_FOUND = "not_found"                  # HTTP 404
    UNPROCESSABLE = "unprocessable"          # HTTP 422
    REQUEST_TIMEOUT = "request_timeout"      # HTTP 408
    UPSTREAM_ERROR = "upstream_error"        # HTTP 5xx (FakeReply.status_code, default 502)
    ERROR_ENVELOPE = "error_envelope"        # HTTP 200 whose body is {"error": {...}}; not billed
    CACHE_HIT = "cache_hit"                  # normal completion with x-openrouter-cache-status: HIT
    PROVIDER_MISMATCH = "provider_mismatch"  # served by FakeReply.mismatch_provider
    MODEL_MISMATCH = "model_mismatch"        # served model FakeReply.mismatch_model
    EMPTY_LENGTH = "empty_length"            # finish length, empty content, reasoning consumed the cap; billed
    EMPTY_STOP = "empty_stop"                # finish stop, empty content
    BLOCKED = "blocked"                      # finish content_filter, native SAFETY, empty content
    MALFORMED_JSON = "malformed_json"        # HTTP 200, truncated non-JSON body; billed
    MALFORMED_USAGE = "malformed_usage"      # completion whose usage fields have junk types/values
    READ_TIMEOUT = "read_timeout"            # generated and billed, client gets httpx.ReadTimeout
    REMOTE_DISCONNECT = "remote_disconnect"  # generated and billed, client gets httpx.RemoteProtocolError
    CONNECT_ERROR = "connect_error"          # never reached the server: httpx.ConnectError, nothing billed
    CONNECT_TIMEOUT = "connect_timeout"      # never reached the server: httpx.ConnectTimeout, nothing billed


_HTTP_ERROR_STATUS = {
    Fault.RATE_LIMITED: 429,
    Fault.PAYMENT_REQUIRED: 402,
    Fault.AUTH_FAILED: 401,
    Fault.FORBIDDEN: 403,
    Fault.BAD_REQUEST: 400,
    Fault.NOT_FOUND: 404,
    Fault.UNPROCESSABLE: 422,
    Fault.REQUEST_TIMEOUT: 408,
    Fault.UPSTREAM_ERROR: 502,
}
_HTTP_ERROR_MESSAGE = {
    429: "Rate limit exceeded: google/gemini-2.5-flash-lite is temporarily rate-limited upstream.",
    402: "Insufficient credits. Add more using https://openrouter.ai/settings/credits",
    401: "No auth credentials found",
    403: "Key limit exceeded",
    400: "Provider returned error: unsupported parameter",
    404: "No endpoints found matching your data policy and provider preferences.",
    422: "Unprocessable request",
    408: "Request timed out",
}


@dataclass(frozen=True)
class FakeReply:
    fault: Fault | None = None
    content: Any = AUTO                    # message.content
    finish_reason: Any = AUTO
    native_finish_reason: Any = AUTO
    reasoning: Any = AUTO                  # message.reasoning (summary text)
    reasoning_details: Any = AUTO          # message.reasoning_details
    reasoning_tokens: Any = AUTO           # usage.completion_tokens_details.reasoning_tokens
    usage: Any = AUTO                      # whole usage object; AUTO computes it
    usage_overrides: Mapping[str, Any] = field(default_factory=dict)   # top-level usage keys replaced after AUTO
    provider: Any = AUTO                   # top-level "provider"
    model: Any = AUTO                      # top-level "model" (AUTO echoes the request)
    headers: Mapping[str, str] = field(default_factory=dict)            # extra/overriding response headers
    status_code: int | None = None         # HTTP status for UPSTREAM_ERROR (default 502) or a raw ``body``
    retry_after: str | None = "1"          # Retry-After value sent with RATE_LIMITED (seconds or HTTP-date)
    error_message: str | None = None       # message inside error bodies/envelopes
    delay_seconds: float = 0.0             # server think time; beyond the client's read timeout -> ReadTimeout
    body: bytes | None = None              # raw response body override (status_code or 200)
    mismatch_provider: str = "Google"
    mismatch_model: str = "google/gemini-2.5-flash-lite-preview-09-2025"


ReplySpec = FakeReply | Sequence[FakeReply]
ReplyKey = tuple[str | None, str | None, int | None, int | None]


@dataclass(frozen=True)
class FakeCall:
    method: str
    path: str
    body: Any                              # parsed JSON body (None when absent/unparseable)
    headers: Mapping[str, str]             # lowercase, credentials redacted
    coordinates: Mapping[str, Any]         # the request extension ({} when absent)
    sequence: int                          # 1-based order of arrival across all routes
    node_call_no: int = 0                  # 1-based arrival number of this node's completion requests (0: other routes)
    served_no: int = 0                     # 1-based number of this node's requests that reached the script (0: refused
                                           # before scripting -- contract 400, emulated cache HIT, account limit,
                                           # capacity 429); sequences in replies/by_node are indexed by it
    status_code: int | None = None         # None when the fake raised a transport exception
    outcome: str = ""                      # response | connect_error | read_timeout | remote_disconnect | cancelled |
                                           # contract_violation | account_limit | capacity_rate_limited
    reached_server: bool = True            # False for scripted connect failures (nothing was sent)
    fault: str | None = None
    generation_id: str | None = None
    billed_usd: Decimal = Decimal("0")


def _fake_prices(model: str) -> tuple[str, str]:
    if model == "google/gemini-2.5-flash":
        return "0.0000003", "0.0000025"
    return "0.0000001", "0.0000004"


_FAKE_SUPPORTED_PARAMETERS = ("include_reasoning", "max_tokens", "reasoning", "response_format", "seed", "stop",
                              "structured_outputs", "temperature", "tool_choice", "tools", "top_p")


def fake_endpoint_lock(model: str = "google/gemini-2.5-flash-lite") -> EndpointLock:
    """The standard ``google-ai-studio`` lock, identical (semantic fields) to ``select_standard_endpoint`` applied to
    the saved 2026-09-17 listing for the two study models; other model names get Flash-Lite's values."""
    input_price, output_price = _fake_prices(model)
    return EndpointLock(
        model=model,
        provider_tag="google-ai-studio",
        provider_name=FAKE_PROVIDER_NAME,
        service_tier="standard",
        input_price_per_token=input_price,
        output_price_per_token=output_price,
        internal_reasoning_price_per_token=output_price,
        context_length=1048576,
        max_completion_tokens=65535,
        supported_parameters=_FAKE_SUPPORTED_PARAMETERS,
    )


def fake_endpoints_listing(model: str) -> dict[str, Any]:
    """An endpoints listing shaped like ``GET /models/{model}/endpoints`` with standard, flex, priority and Vertex
    variants, so selection logic can be exercised offline."""
    input_price, output_price = (Decimal(p) for p in _fake_prices(model))
    variants = (
        ("Google", "google-vertex/eu", Decimal(1)),
        ("Google", "google-vertex", Decimal(1)),
        (FAKE_PROVIDER_NAME, "google-ai-studio/flex", Decimal("0.5")),
        (FAKE_PROVIDER_NAME, "google-ai-studio", Decimal(1)),
        (FAKE_PROVIDER_NAME, "google-ai-studio/priority", Decimal("1.8")),
    )
    endpoints = []
    for provider_name, tag, factor in variants:
        endpoints.append({
            "name": f"{provider_name} | {model}",
            "model_id": model,
            "model_name": model,
            "context_length": 1048576,
            "pricing": {
                "prompt": _decimal_text(input_price * factor),
                "completion": _decimal_text(output_price * factor),
                "internal_reasoning": _decimal_text(output_price * factor),
                "input_cache_read": _decimal_text(input_price * factor / 10),
                "discount": 0,
            },
            "provider_name": provider_name,
            "tag": tag,
            "quantization": "unknown",
            "max_completion_tokens": 65535,
            "max_prompt_tokens": None,
            "supported_parameters": list(_FAKE_SUPPORTED_PARAMETERS),
            "status": 0,
            "uptime_last_30m": 99.9,
            "supports_implicit_caching": False,
        })
    return {"data": {"id": model, "name": model, "endpoints": endpoints}}


def _decimal_text(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text if text else "0"


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _money_json(value: Decimal) -> float:
    return float(value)


_ASSERT_CALL = re.compile(r"\bassert\s+(?:not\s+)?(?:\(\s*)?(?:set\(|math\.isclose\(|abs\()?\s*([A-Za-z_]\w*)\s*\(")


def default_final_answer(coordinates: Mapping[str, Any], body: Mapping[str, Any] | None, *,
                         vary_by_node: bool = False) -> str:
    item_id = str(coordinates.get("item_id") or "")
    identity = str(coordinates.get("node_id") or "") if vary_by_node else item_id
    if not identity:
        identity = _sha(_json_bytes((body or {}).get("messages")))
    number = int(_sha(identity)[:8], 16)
    benchmark = str(coordinates.get("benchmark") or "")
    if benchmark == "mbpp_plus":
        prompt = ""
        if isinstance(body, Mapping) and isinstance(body.get("messages"), list):
            prompt = "\n".join(str(m.get("content", "")) for m in body["messages"] if isinstance(m, Mapping))
        match = _ASSERT_CALL.search(prompt)
        name = match.group(1) if match else "solution"
        return f"def {name}(*args, **kwargs):\n    return None"
    if benchmark == "mmlu_pro":
        return "ABC"[number % 3]
    if benchmark == "bbh_logic4":
        return ("True", "False")[number % 2] if "boolean" in item_id else "ABC"[number % 3]
    return str(number % 97)


def default_content(coordinates: Mapping[str, Any], body: Mapping[str, Any] | None, *,
                    vary_by_node: bool = False) -> str:
    item_id = coordinates.get("item_id") or "unknown-item"
    final = default_final_answer(coordinates, body, vary_by_node=vary_by_node)
    return (f"<rationale>Fake deterministic rationale for {item_id}.</rationale>\n"
            f"<final>{final}</final>")


_ALLOWED_BODY_KEYS = frozenset({"model", "messages", "temperature", "top_p", "max_tokens", "reasoning", "provider",
                                "stream"})


def request_contract_violations(body: Any) -> list[str]:
    """What the fake refuses (HTTP 400): anything outside the section 4.2 request contract."""
    if not isinstance(body, dict):
        return ["body is not a JSON object"]
    problems = [f"key {key!r} is outside the request contract" for key in sorted(set(body) - _ALLOWED_BODY_KEYS)]
    if body.get("stream") is not False:
        problems.append("stream must be false")
    if not isinstance(body.get("model"), str) or not body.get("model"):
        problems.append("model missing")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or not all(
            isinstance(m, dict) and set(m) == {"role", "content"} and isinstance(m["content"], str)
            for m in messages):
        problems.append("messages must be a non-empty list of {role, content}")
    for key in ("temperature", "top_p"):
        value = body.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{key} must be a number")
    max_tokens = body.get("max_tokens")
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict) or set(reasoning) != {"max_tokens", "exclude"}:
        problems.append("reasoning must be exactly {max_tokens, exclude}")
    else:
        budget = reasoning.get("max_tokens")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            problems.append("reasoning.max_tokens must be a positive integer")
        elif isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens <= budget:
            problems.append("max_tokens must exceed reasoning.max_tokens (completion headroom)")
        if reasoning.get("exclude") is not False:
            problems.append("reasoning.exclude must be false")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        problems.append("max_tokens must be a positive integer")
    provider = body.get("provider")
    if not isinstance(provider, dict) or set(provider) != {"only", "allow_fallbacks", "require_parameters"}:
        problems.append("provider must be exactly {only, allow_fallbacks, require_parameters}")
    else:
        only = provider.get("only")
        if not (isinstance(only, list) and len(only) == 1 and isinstance(only[0], str) and only[0]):
            problems.append("provider.only must list exactly one endpoint tag")
        elif any(marker in only[0] for marker in ("/flex", "/priority", "/batch", ":free")):
            problems.append(f"provider.only tag {only[0]!r} is a non-standard tier variant")
        if provider.get("allow_fallbacks") is not False:
            problems.append("provider.allow_fallbacks must be false")
        if provider.get("require_parameters") is not True:
            problems.append("provider.require_parameters must be true")
    return problems


def _normalize_key(key: Sequence[Any]) -> ReplyKey:
    if len(key) != 4:
        raise ValueError(f"reply keys are (item_id, kind, root_slot, update_round); got {key!r}")
    item_id, kind, root_slot, update_round = key
    return (item_id, getattr(kind, "value", kind), root_slot, update_round)


@dataclass
class FakeScript:
    replies: dict[ReplyKey, ReplySpec] = field(default_factory=dict)
    by_node: dict[str, ReplySpec] = field(default_factory=dict)
    queue: list[FakeReply] = field(default_factory=list)
    hook: Callable[[FakeCall], FakeReply | None] | None = None
    default: FakeReply = field(default_factory=FakeReply)
    provider_name: str = FAKE_PROVIDER_NAME
    service_tier: str | None = "standard"
    vary_final_by_node: bool = False
    input_price_per_token: str | None = None      # None -> the fake lock's prices for the requested model
    output_price_per_token: str | None = None
    metadata_lag_polls: int = 0                   # /generation returns 404 this many times per id before data
    expected_api_key: str | None = None           # when set, requests without "Bearer <key>" get HTTP 401
    key_limit_usd: Decimal | None = Decimal("135")
    total_credits_usd: Decimal = Decimal("150")
    prior_usage_usd: Decimal = Decimal("0")       # account/key usage that predates this script (unrelated spend)
    key_label: str = "sk-or-v1-fak...e00"
    endpoints: dict[str, Any] = field(default_factory=dict)   # model -> listing dict or raw bytes override
    max_concurrent_requests: int | None = None    # emulated upstream capacity: requests beyond it get HTTP 429
    capacity_retry_after: str | None = "1"        # Retry-After sent with those capacity 429s
    enforce_account_limits: bool = True           # usage >= key_limit_usd -> 403; usage >= total_credits_usd -> 402
    # ---- recorded state -----------------------------------------------------------------------------------------
    calls: list[FakeCall] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    billed_usd: Decimal = Decimal("0")
    generations: dict[str, dict[str, Any]] = field(default_factory=dict)
    in_flight: int = 0                            # completion requests currently being "processed"
    peak_in_flight: int = 0                       # measured server-side concurrency (for effective-concurrency tests)
    _node_calls: Counter = field(default_factory=Counter, repr=False)
    _node_served: Counter = field(default_factory=Counter, repr=False)
    _metadata_polls: Counter = field(default_factory=Counter, repr=False)
    _response_cache: dict[str, tuple[int, dict[str, str], bytes]] = field(default_factory=dict, repr=False)
    _sequence: int = field(default=0, repr=False)
    _mutex: Any = field(default_factory=threading.Lock, repr=False, compare=False)

    # ---- convenience ---------------------------------------------------------------------------------------------
    def set_reply(self, item_id: str | None, kind: Any = None, root_slot: int | None = None,
                  update_round: int | None = None, reply: ReplySpec | None = None) -> None:
        self.replies[_normalize_key((item_id, kind, root_slot, update_round))] = reply or FakeReply()

    def completion_calls(self, *, reached_server: bool | None = None) -> list[FakeCall]:
        """Every completion request the client attempted, in arrival order; ``reached_server=True`` drops scripted
        connect failures (the requests a provider could have billed), ``False`` keeps only those."""
        return [c for c in self.calls if c.path.endswith("/chat/completions")
                and (reached_server is None or c.reached_server is reached_server)]

    # ---- routing -------------------------------------------------------------------------------------------------
    def _next_sequence(self) -> int:
        with self._mutex:
            self._sequence += 1
            return self._sequence

    def _record(self, call: FakeCall) -> None:
        with self._mutex:
            self.calls.append(call)
            if call.billed_usd:
                self.billed_usd += call.billed_usd

    async def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if self.expected_api_key is not None and \
                request.headers.get("authorization") != f"Bearer {self.expected_api_key}":
            call = self._base_call(request)
            response = _json_response(401, {"error": {"code": 401, "message": _HTTP_ERROR_MESSAGE[401]}})
            self._record(replace(call, status_code=401, outcome="response"))
            return response
        if request.method == "POST" and path.endswith("/chat/completions"):
            return await self._completion(request)
        if request.method == "GET" and path.endswith("/key"):
            return self._simple(request, 200, {"data": self._key_data()})
        if request.method == "GET" and path.endswith("/credits"):
            return self._simple(request, 200, {"data": {
                "total_credits": _money_json(self.total_credits_usd),
                "total_usage": _money_json(self.prior_usage_usd + self.billed_usd)}})
        if request.method == "GET" and path.endswith("/generation"):
            return self._generation(request)
        match = re.search(r"/models/(.+)/endpoints$", path)
        if request.method == "GET" and match:
            return self._endpoints(request, match.group(1))
        return self._simple(request, 404, {"error": {"code": 404, "message": f"fake: no route {request.method} "
                                                                            f"{path}"}})

    def _base_call(self, request: httpx.Request, **kwargs: Any) -> FakeCall:
        try:
            body = json.loads(request.content) if request.content else None
        except ValueError:
            body = None
        coordinates = request.extensions.get(REQUEST_EXTENSION_KEY) or {}
        return FakeCall(method=request.method, path=request.url.path, body=body,
                        headers=redact_headers(dict(request.headers.items())), coordinates=dict(coordinates),
                        sequence=self._next_sequence(), **kwargs)

    def _simple(self, request: httpx.Request, status: int, payload: Any) -> httpx.Response:
        self._record(replace(self._base_call(request), status_code=status, outcome="response"))
        return _json_response(status, payload)

    def _key_data(self) -> dict[str, Any]:
        usage = self.prior_usage_usd + self.billed_usd
        limit = self.key_limit_usd
        return {
            "label": self.key_label,
            "limit": None if limit is None else _money_json(limit),
            "limit_remaining": None if limit is None else _money_json(limit - usage),
            "limit_reset": None,
            "include_byok_in_limit": False,
            "usage": _money_json(usage),
            "usage_daily": _money_json(usage),
            "usage_weekly": _money_json(usage),
            "usage_monthly": _money_json(usage),
            "byok_usage": 0,
            "is_free_tier": False,
            "is_provisioning_key": False,
            "rate_limit": {"requests": -1, "interval": "10s"},
        }

    def _generation(self, request: httpx.Request) -> httpx.Response:
        generation_id = request.url.params.get("id", "")
        with self._mutex:
            known = generation_id in self.generations
            self._metadata_polls[generation_id] += 1
            polls = self._metadata_polls[generation_id]
        if not known or polls <= self.metadata_lag_polls:
            return self._simple(request, 404, {"error": {"code": 404, "message": "Generation not found"}})
        return self._simple(request, 200, {"data": self.generations[generation_id]})

    def _endpoints(self, request: httpx.Request, model: str) -> httpx.Response:
        override = self.endpoints.get(model)
        if isinstance(override, (bytes, bytearray)):
            self._record(replace(self._base_call(request), status_code=200, outcome="response"))
            return httpx.Response(200, headers={"content-type": "application/json"}, content=bytes(override))
        if override is not None:
            return self._simple(request, 200, override)
        if model not in ("google/gemini-2.5-flash-lite", "google/gemini-2.5-flash"):
            return self._simple(request, 404, {"error": {"code": 404, "message": "Model not found"}})
        return self._simple(request, 200, fake_endpoints_listing(model))

    # ---- completions ---------------------------------------------------------------------------------------------
    def _pick_reply(self, call: FakeCall) -> FakeReply:
        if self.hook is not None:
            chosen = self.hook(call)
            if chosen is not None:
                return chosen
        with self._mutex:
            if self.queue:
                return self.queue.pop(0)
        coordinates = call.coordinates
        node_id = coordinates.get("node_id")
        if node_id is not None and node_id in self.by_node:
            return _indexed(self.by_node[node_id], call.served_no)
        wanted = _normalize_key((coordinates.get("item_id"), coordinates.get("kind"), coordinates.get("root_slot"),
                                 coordinates.get("update_round")))
        best: tuple[int, ReplySpec] | None = None
        for raw_key, spec in self.replies.items():
            key = _normalize_key(raw_key)
            if all(k is None or k == w for k, w in zip(key, wanted)):
                specificity = sum(k is not None for k in key)
                if best is None or specificity > best[0]:
                    best = (specificity, spec)
        if best is not None:
            return _indexed(best[1], call.served_no)
        return self.default

    async def _completion(self, request: httpx.Request) -> httpx.Response:
        call = self._base_call(request)
        node_key = str(call.coordinates.get("node_id") or ("body:" + _sha(request.content)))
        with self._mutex:
            self._node_calls[node_key] += 1
            call = replace(call, node_call_no=self._node_calls[node_key])
        problems = request_contract_violations(call.body)
        if problems:
            with self._mutex:
                self.violations.extend(problems)
            self._record(replace(call, status_code=400, outcome="contract_violation"))
            return _json_response(400, {"error": {"code": 400, "message": "fake request contract violation: "
                                                                          + "; ".join(problems)}})
        cache_enabled = request.headers.get("x-openrouter-cache", "").strip().lower() != "false"
        cache_key = _sha(request.content)
        if cache_enabled:
            with self._mutex:
                cached = self._response_cache.get(cache_key)
            if cached is not None:
                status, headers, content = cached
                self._record(replace(call, status_code=status, outcome="response", fault="emulated_cache_hit"))
                return httpx.Response(status, headers={**headers, "x-openrouter-cache-status": "HIT"},
                                      content=content)
        refusal = self._account_refusal()
        if refusal is not None:
            status, message = refusal
            self._record(replace(call, status_code=status, outcome="account_limit"))
            return _json_response(status, {"error": {"code": status, "message": message}})

        # Capacity is checked before the script is consulted, so a capacity 429 never consumes a scripted reply.
        with self._mutex:
            over_capacity = self.max_concurrent_requests is not None and \
                self.in_flight >= self.max_concurrent_requests
            if not over_capacity:
                self.in_flight += 1
        if over_capacity:
            headers = {"content-type": "application/json"}
            if self.capacity_retry_after is not None:
                headers["retry-after"] = self.capacity_retry_after
            self._record(replace(call, status_code=429, outcome="capacity_rate_limited"))
            return httpx.Response(429, headers=headers, content=_json_bytes(
                {"error": {"code": 429, "message": _HTTP_ERROR_MESSAGE[429],
                           "metadata": {"provider_name": self.provider_name, "raw": "capacity"}}}))
        try:
            with self._mutex:
                self._node_served[node_key] += 1
                call = replace(call, served_no=self._node_served[node_key])
            reply = self._pick_reply(call)
            fault = reply.fault
            call = replace(call, fault=None if fault is None else fault.value)
            if fault in (Fault.CONNECT_ERROR, Fault.CONNECT_TIMEOUT):
                self._record(replace(call, outcome=fault.value, reached_server=False))
                exc = httpx.ConnectError if fault is Fault.CONNECT_ERROR else httpx.ConnectTimeout
                raise exc(f"fake {fault.value}", request=request)
            with self._mutex:
                self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            return await self._reply(request, call, reply, cache_enabled, cache_key)
        finally:
            with self._mutex:
                self.in_flight -= 1

    def _account_refusal(self) -> tuple[int, str] | None:
        if not self.enforce_account_limits:
            return None
        with self._mutex:
            usage = self.prior_usage_usd + self.billed_usd
        if self.key_limit_usd is not None and usage >= self.key_limit_usd:
            return 403, "Key limit exceeded"
        if usage >= self.total_credits_usd:
            return 402, _HTTP_ERROR_MESSAGE[402]
        return None

    @staticmethod
    def _billable(reply: FakeReply) -> bool:
        """Whether the server generated (and charged) something for this reply, even if the client never sees it."""
        if reply.fault in _HTTP_ERROR_STATUS or reply.fault is Fault.ERROR_ENVELOPE:
            return False
        if reply.fault in (Fault.READ_TIMEOUT, Fault.REMOTE_DISCONNECT):
            return True
        return 200 <= (reply.status_code or 200) < 300

    def _record_unseen(self, call: FakeCall, reply: FakeReply, request: httpx.Request, outcome: str) -> None:
        """The client stopped waiting (read timeout or cancellation) after the request reached the server."""
        if self._billable(reply):
            generation_id, _, cost = self._generate(call, reply, request)
            self._record(replace(call, outcome=outcome, generation_id=generation_id, billed_usd=cost))
        else:
            self._record(replace(call, outcome=outcome))

    async def _reply(self, request: httpx.Request, call: FakeCall, reply: FakeReply, cache_enabled: bool,
                     cache_key: str) -> httpx.Response:
        fault = reply.fault
        if reply.delay_seconds > 0:
            read_timeout = (request.extensions.get("timeout") or {}).get("read")
            try:
                if read_timeout is not None and reply.delay_seconds > read_timeout:
                    await asyncio.sleep(read_timeout)
                    self._record_unseen(call, reply, request, "read_timeout")
                    raise httpx.ReadTimeout("fake read timeout (delay beyond the client's read timeout)",
                                            request=request)
                await asyncio.sleep(reply.delay_seconds)
            except asyncio.CancelledError:
                # A cancelled client does not cancel the provider: the generation still happens and is billed.
                self._record_unseen(call, reply, request, "cancelled")
                raise

        if fault in _HTTP_ERROR_STATUS:
            status = reply.status_code or _HTTP_ERROR_STATUS[fault]
            message = reply.error_message or _HTTP_ERROR_MESSAGE.get(status, "Provider returned error")
            error: dict[str, Any] = {"code": status, "message": message}
            headers = {"content-type": "application/json"}
            if fault is Fault.RATE_LIMITED:
                error["metadata"] = {"provider_name": self.provider_name, "raw": "429 RESOURCE_EXHAUSTED"}
                if reply.retry_after is not None:
                    headers["retry-after"] = reply.retry_after
            if fault is Fault.UPSTREAM_ERROR:
                error["metadata"] = {"provider_name": self.provider_name, "raw": "upstream failure"}
            headers.update({k.lower(): v for k, v in reply.headers.items()})
            self._record(replace(call, status_code=status, outcome="response"))
            return httpx.Response(status, headers=headers, content=_json_bytes({"error": error}))

        if fault is Fault.ERROR_ENVELOPE:
            envelope = {"error": {"code": 502, "message": reply.error_message or "Provider returned error",
                                  "metadata": {"provider_name": self.provider_name,
                                               "raw": "{\"error\":{\"code\":500,\"status\":\"INTERNAL\"}}"}},
                        "user_id": "user_fake"}
            headers = {"content-type": "application/json", "x-openrouter-cache-status": "MISS",
                       **{k.lower(): v for k, v in reply.headers.items()}}
            self._record(replace(call, status_code=200, outcome="response"))
            return httpx.Response(200, headers=headers, content=_json_bytes(envelope))

        billable = self._billable(reply)
        generation_id, payload, cost = self._generate(call, reply, request, register=billable)
        if billable:
            call = replace(call, generation_id=generation_id, billed_usd=cost)
        if fault is Fault.READ_TIMEOUT:
            self._record(replace(call, outcome="read_timeout"))
            raise httpx.ReadTimeout("fake read timeout after the request was written", request=request)
        if fault is Fault.REMOTE_DISCONNECT:
            self._record(replace(call, outcome="remote_disconnect"))
            raise httpx.RemoteProtocolError("fake: server disconnected without sending a response",
                                            request=request)

        status = reply.status_code or 200
        headers = {"content-type": "application/json",
                   "x-openrouter-cache-status": "HIT" if fault is Fault.CACHE_HIT else "MISS"}
        headers.update({k.lower(): v for k, v in reply.headers.items()})
        if reply.body is not None:
            content = reply.body
        elif fault is Fault.MALFORMED_JSON:
            content = _json_bytes(payload)[:40]
        else:
            content = _json_bytes(payload)
        if cache_enabled and status == 200:
            with self._mutex:
                self._response_cache.setdefault(cache_key, (status, dict(headers), content))
        self._record(replace(call, status_code=status, outcome="response"))
        return httpx.Response(status, headers=headers, content=content)

    def _generate(self, call: FakeCall, reply: FakeReply, request: httpx.Request, *,
                  register: bool = True) -> tuple[str, dict[str, Any], Decimal]:
        """Build the completion payload and return (id, payload, cost); ``register`` stores its generation metadata
        (only for replies the server actually generated -- never for a scripted non-2xx status)."""
        body = call.body
        coordinates = call.coordinates
        fault = reply.fault
        generation_id = "gen-fake-{:06d}-{}".format(
            call.sequence, _sha(f"{coordinates.get('attempt_id') or ''}|{call.sequence}")[:16])
        max_tokens = int(body["max_tokens"])
        budget = int(body["reasoning"]["max_tokens"])
        identity = str(coordinates.get("node_id") or _sha(request.content))
        number = int(_sha(identity)[:8], 16)

        content = reply.content
        if content is AUTO:
            content = "" if fault in (Fault.EMPTY_LENGTH, Fault.EMPTY_STOP, Fault.BLOCKED) else default_content(
                coordinates, body, vary_by_node=self.vary_final_by_node)
        finish = reply.finish_reason
        if finish is AUTO:
            finish = {Fault.EMPTY_LENGTH: "length", Fault.BLOCKED: "content_filter"}.get(fault, "stop")
        native = reply.native_finish_reason
        if native is AUTO:
            native = {"length": "MAX_TOKENS", "content_filter": "SAFETY"}.get(finish, "STOP")
        reasoning_tokens = reply.reasoning_tokens
        if reasoning_tokens is AUTO:
            reasoning_tokens = max_tokens if fault is Fault.EMPTY_LENGTH else \
                budget // 4 + number % (budget // 2 + 1)
        reasoning = reply.reasoning
        if reasoning is AUTO:
            reasoning = "**Planning the solution**\nFake reasoning summary; never relayed to peers."
        details = reply.reasoning_details
        if details is AUTO:
            if isinstance(reasoning, str) and reasoning:
                details = [{"type": "reasoning.text", "text": reasoning, "format": "google-gemini-v1", "index": 0}]
            else:
                details = reasoning
        provider = reply.provider
        if provider is AUTO:
            provider = reply.mismatch_provider if fault is Fault.PROVIDER_MISMATCH else self.provider_name
        model = reply.model
        if model is AUTO:
            model = reply.mismatch_model if fault is Fault.MODEL_MISMATCH else body["model"]

        prompt_chars = sum(len(m["content"]) for m in body["messages"])
        prompt_tokens = max(1, -(-prompt_chars // 4))
        visible_tokens = -(-len(content) // 4) if isinstance(content, str) else 0
        counted_reasoning = reasoning_tokens if isinstance(reasoning_tokens, int) else 0
        completion_tokens = min(max_tokens, counted_reasoning + visible_tokens)
        default_in, default_out = _fake_prices(str(body["model"]))
        in_price = Decimal(self.input_price_per_token or default_in)
        out_price = Decimal(self.output_price_per_token or default_out)
        prompt_cost = prompt_tokens * in_price
        completion_cost = completion_tokens * out_price
        cost = prompt_cost + completion_cost

        usage = reply.usage
        if usage is AUTO:
            if fault is Fault.MALFORMED_USAGE:
                usage = {"prompt_tokens": "12", "completion_tokens": -3, "total_tokens": None, "cost": "n/a",
                         "prompt_tokens_details": [], "completion_tokens_details": {"reasoning_tokens": True},
                         "cost_details": "none"}
            else:
                completion_details: dict[str, Any] = {"image_tokens": 0}
                if reasoning_tokens is not OMIT:
                    completion_details["reasoning_tokens"] = reasoning_tokens
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "cost": _money_json(cost),
                    "is_byok": False,
                    "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                    "cost_details": {"upstream_inference_cost": None,
                                     "upstream_inference_prompt_cost": _money_json(prompt_cost),
                                     "upstream_inference_completions_cost": _money_json(completion_cost)},
                    "completion_tokens_details": completion_details,
                }
                usage.update(reply.usage_overrides)

        message: dict[str, Any] = {"role": "assistant"}
        _put(message, "content", content)
        message["refusal"] = None
        _put(message, "reasoning", reasoning)
        _put(message, "reasoning_details", details)
        choice: dict[str, Any] = {"logprobs": None}
        _put(choice, "finish_reason", finish)
        _put(choice, "native_finish_reason", native)
        choice["index"] = 0
        choice["message"] = message
        payload: dict[str, Any] = {"id": generation_id}
        _put(payload, "provider", provider)
        _put(payload, "model", model)
        payload.update({"object": "chat.completion", "created": FAKE_CREATED_EPOCH + call.sequence,
                        "choices": [choice]})
        _put(payload, "usage", usage)

        metadata = {
            "id": generation_id,
            "upstream_id": f"fake-upstream-{call.sequence}",
            "total_cost": _money_json(cost),
            "cache_discount": None,
            "upstream_inference_cost": None,
            "created_at": "2026-09-17T07:00:00.000Z",
            "model": model if isinstance(model, str) else body["model"],
            "app_id": None,
            "streamed": False,
            "cancelled": False,
            "provider_name": provider if isinstance(provider, str) else self.provider_name,
            "latency": 850,
            "moderation_latency": None,
            "generation_time": 800,
            "finish_reason": finish if isinstance(finish, str) else None,
            "native_finish_reason": native if isinstance(native, str) else None,
            "tokens_prompt": prompt_tokens,
            "tokens_completion": completion_tokens,
            "native_tokens_prompt": prompt_tokens,
            "native_tokens_completion": completion_tokens,
            "native_tokens_completion_images": 0,
            "native_tokens_reasoning": reasoning_tokens if isinstance(reasoning_tokens, int) else None,
            "native_tokens_cached": 0,
            "num_media_prompt": None,
            "num_media_completion": None,
            "num_search_results": None,
            "origin": "",
            "usage": _money_json(cost),
            "is_byok": False,
            "service_tier": self.service_tier,
            "response_cache_source_id": f"gen-fake-cache-source-{call.sequence}" if fault is Fault.CACHE_HIT else None,
        }
        if register:
            with self._mutex:
                self.generations[generation_id] = metadata
        return generation_id, payload, cost


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not OMIT:
        target[key] = value


def _indexed(spec: ReplySpec, call_no: int) -> FakeReply:
    if isinstance(spec, FakeReply):
        return spec
    sequence = list(spec)
    if not sequence:
        return FakeReply()
    return sequence[min(max(call_no, 1), len(sequence)) - 1]


def _json_response(status: int, payload: Any) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "application/json"}, content=_json_bytes(payload))


def fake_openrouter_transport(script: FakeScript | None = None) -> httpx.MockTransport:
    """A mock transport serving ``script`` (a fresh default ``FakeScript`` when None). The script is also exposed as
    ``transport.script`` so a test that did not keep a reference can still inspect ``calls``."""
    active = script if script is not None else FakeScript()

    async def handler(request: httpx.Request) -> httpx.Response:
        return await active.handle(request)

    transport = httpx.MockTransport(handler)
    transport.script = active  # type: ignore[attr-defined]
    return transport


DEFAULT_FAKE_SAMPLING: Mapping[str, Any] = {"temperature": 0.7, "top_p": 0.95}


async def fake_generation_outcome(spec: GenerationSpec, messages: Sequence[Mapping[str, str]] | None = None, *,
                                  reply: FakeReply | None = None, script: FakeScript | None = None,
                                  lock: EndpointLock | None = None, attempt_no: int = 1,
                                  sampling: Mapping[str, Any] | None = None) -> GenerationOutcome:
    """One realistic ``GenerationOutcome`` for ``spec`` through the real client and the fake transport.

    For tests that need a ``GenerationResult``/``GenerationOutcome`` without wiring a client. ``lock`` defaults to
    ``fake_endpoint_lock(spec.model)``; ``messages`` to a one-line user prompt naming the item; pass ``reply`` (a
    one-off default) or a full ``script``, not both.
    """
    from agent_allocation.api.client import OpenRouterClient  # local import: client imports nothing from fake

    if reply is not None and script is not None:
        raise ValueError("pass reply or script, not both")
    active = script if script is not None else FakeScript(default=reply or FakeReply())
    client = OpenRouterClient(api_key=None, transport=fake_openrouter_transport(active))
    try:
        return await client.generate(
            spec, messages or [{"role": "user", "content": f"Fake task for {spec.item_id}."}],
            attempt_id=f"{spec.node_id}:{attempt_no}", lock=lock or fake_endpoint_lock(spec.model),
            sampling=sampling or DEFAULT_FAKE_SAMPLING)
    finally:
        await client.aclose()


def fake_generation_outcome_sync(spec: GenerationSpec, messages: Sequence[Mapping[str, str]] | None = None,
                                 **kwargs: Any) -> GenerationOutcome:
    """Synchronous ``fake_generation_outcome`` for plain (non-async) tests; safe inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(fake_generation_outcome(spec, messages, **kwargs))
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(fake_generation_outcome(spec, messages, **kwargs))).result()
