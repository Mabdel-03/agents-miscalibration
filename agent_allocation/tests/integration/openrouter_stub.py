"""Agent 08's own OpenRouter-shaped ``httpx`` stub, independent of ``agent_allocation.api.fake``.

Why a second fake: agent 01's client is classified against agent 01's fake; if both share a misreading of the
wire format, the unit tests pass and production fails. This stub is written from the documented response
shape (MASTER_SPEC section 4, contract section 4 facts) and records every request so tests can count network
opportunities, inspect bodies/headers for canaries, and inject faults in a fixed order.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

MODEL = "google/gemini-2.5-flash-lite"
PROVIDER_NAME = "Google AI Studio"


def completion_payload(content: str = "<rationale>ok</rationale>\n<final>4</final>", *, finish: str = "stop",
                       native_finish: str = "STOP", reasoning_tokens: int | None = 300, prompt_tokens: int = 120,
                       completion_tokens: int = 410, cost: float = 0.000176, model: str = MODEL,
                       provider: str = PROVIDER_NAME, generation_id: str = "gen-stub-1",
                       reasoning_text: str | None = None, cached_tokens: int | None = 0) -> dict[str, Any]:
    usage: dict[str, Any] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                             "total_tokens": prompt_tokens + completion_tokens, "cost": cost, "is_byok": False,
                             "cost_details": {"upstream_inference_cost": None}}
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_text is not None:
        message["reasoning"] = reasoning_text
    return {"id": generation_id, "provider": provider, "model": model, "object": "chat.completion",
            "created": 1789000000, "choices": [{"index": 0, "finish_reason": finish,
                                                 "native_finish_reason": native_finish, "message": message}],
            "usage": usage}


@dataclass
class StubReply:
    status: int = 200
    payload: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=lambda: {"x-openrouter-cache-status": "MISS"})
    raise_exc: type[Exception] | None = None       # e.g. httpx.ReadTimeout (sent, outcome unknown)


def ok(**kwargs) -> StubReply:
    return StubReply(payload=completion_payload(**kwargs))


def cache_hit(**kwargs) -> StubReply:
    return StubReply(payload=completion_payload(**kwargs), headers={"x-openrouter-cache-status": "HIT"})


def rate_limited(retry_after: str = "7") -> StubReply:
    return StubReply(status=429, payload={"error": {"code": 429, "message": "Rate limit exceeded"}},
                     headers={"retry-after": retry_after})


def error_envelope(code: int = 502) -> StubReply:
    return StubReply(status=200, payload={"error": {"code": code, "message": "Provider returned error",
                                                    "metadata": {"provider_name": PROVIDER_NAME}}})


def read_timeout() -> StubReply:
    return StubReply(raise_exc=httpx.ReadTimeout)


def connect_failure() -> StubReply:
    return StubReply(raise_exc=httpx.ConnectError)


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body_text: str

    @property
    def body(self) -> Any:
        return json.loads(self.body_text) if self.body_text else None


class StubOpenRouter:
    """Replies are consumed in order; when exhausted, ``default`` is used for chat completions."""

    def __init__(self, replies: list[StubReply] | None = None, *,
                 default: Callable[[RecordedRequest], StubReply] | None = None) -> None:
        self.replies = list(replies or [])
        self.default = default or (lambda req: ok())
        self.requests: list[RecordedRequest] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        rec = RecordedRequest(request.method, request.url.path, {k.lower(): v for k, v in request.headers.items()},
                              request.content.decode("utf-8", errors="replace"))
        self.requests.append(rec)
        reply = self.replies.pop(0) if self.replies else self.default(rec)
        if reply.raise_exc is not None:
            raise reply.raise_exc("injected by StubOpenRouter", request=request)
        return httpx.Response(reply.status, json=reply.payload, headers=reply.headers)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    @property
    def completion_requests(self) -> list[RecordedRequest]:
        return [r for r in self.requests if r.path.endswith("/chat/completions")]
