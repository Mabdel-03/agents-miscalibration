"""Async OpenRouter client: one persistent ``httpx.AsyncClient``, zero automatic retries, no disk, no ledger.

``generate`` sends exactly one HTTP attempt and returns a typed ``GenerationOutcome``; it never raises for transport
or model failures. It raises only ``api.request.RequestContractError`` (a ``ValueError``) for a spec/lock/sampling/
message/attempt-id violation -- always before any byte is sent, so the reservation can be released and the circuit
opened -- and lets ``asyncio.CancelledError`` propagate (a cancelled attempt may have been sent: treat it as
ambiguous). Retrying, backoff, reservation and persistence belong to the scheduler (agent 02). httpx itself never
retries requests; the default transport is built with ``retries=0``.

Account endpoints (``key_info``, ``credits``, ``generation_metadata``, ``list_endpoints``) raise
``OpenRouterAccountError`` on transport failures or unexpected statuses. In ``key_info``, ``credits`` and
``generation_metadata`` every JSON float becomes ``Decimal(str(value))`` -- the same conversion the completion usage
gets, so costs from both sources compare exactly -- and integers stay ``int``. ``list_endpoints`` returns the raw bytes
(save them beside the lock) and the plain parsed JSON.
"""
from __future__ import annotations

import json
import re
import time
from decimal import Decimal
from typing import Any, Mapping, Sequence

import anyio
import httpx

from agent_allocation.apikey import redact, redact_headers
from agent_allocation.api.normalize import (
    GATEWAY_TIMEOUT_STATUSES,
    MAX_ERROR_DETAIL_CHARS,
    classify_exception,
    classify_http_status,
    error_summary,
    parse_completion,
    parse_retry_after,
)
from agent_allocation.api.request import REQUEST_EXTENSION_KEY, REQUEST_HEADERS, RequestContractError, \
    build_request_body, request_coordinates
from agent_allocation.contracts import EndpointLock, GenerationOutcome, GenerationResult, GenerationSpec, \
    TransportStatus

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# What a send can raise besides httpx's own errors: httpcore maps only some low-level errors, so a TLS error after the
# handshake (``ssl.SSLError``, an ``OSError``) or an unmapped anyio stream error can escape. All of them may happen
# after bytes were written, so they classify as AMBIGUOUS_TIMEOUT rather than crashing the dispatcher.
TRANSPORT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.RequestError, OSError, anyio.BrokenResourceError, anyio.ClosedResourceError, anyio.EndOfStream,
)
WRITE_TIMEOUT_SECONDS = 60.0
_MODEL_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*/[A-Za-z0-9][A-Za-z0-9._\-:]*$")


def _decimalize(value: Any) -> Any:
    if isinstance(value, float):
        return Decimal(str(value)) if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {k: _decimalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimalize(v) for v in value]
    return value


class OpenRouterAccountError(RuntimeError):
    """A non-generation endpoint failed. ``transport_status`` uses the same classification as ``generate``."""

    def __init__(self, message: str, *, transport_status: TransportStatus, http_status: int | None = None,
                 retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.transport_status = transport_status
        self.http_status = http_status
        self.retry_after_seconds = retry_after_seconds


class OpenRouterClient:
    def __init__(self, *, api_key: str | None, base_url: str = DEFAULT_BASE_URL,
                 transport: httpx.AsyncBaseTransport | None = None, connect_timeout: float = 15.0,
                 read_timeout: float = 600.0) -> None:
        self._api_key = api_key or None
        self._base_url = base_url.rstrip("/")
        timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=WRITE_TIMEOUT_SECONDS,
                                pool=connect_timeout)
        # The scheduler bounds concurrency (<= 128 in flight); the pool must never be the hidden limiter.
        limits = httpx.Limits(max_connections=512, max_keepalive_connections=256, keepalive_expiry=30.0)
        if transport is None:
            # httpx builds AsyncHTTPTransport(retries=0) itself; letting it do so keeps proxy environment variables
            # (cluster egress) working, which an explicitly passed transport would silently disable.
            self._client = httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=False)
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout, follow_redirects=False)
        self._closed = False

    def __repr__(self) -> str:
        return f"OpenRouterClient(base_url={self._base_url!r}, api_key={'set' if self._api_key else 'none'})"

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    # ---- helpers -------------------------------------------------------------------------------------------------
    def _headers(self, *, json_body: bool) -> dict[str, str]:
        headers = dict(REQUEST_HEADERS) if json_body else {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _scrub_body(self, content: bytes) -> bytes:
        if self._api_key and self._api_key.encode() in content:
            return content.replace(self._api_key.encode(), b"[REDACTED_KEY]")
        return content

    def _scrub_text(self, text: str) -> str:
        if self._api_key:
            text = text.replace(self._api_key, "[REDACTED_KEY]")
        return (redact(text) or "")[:MAX_ERROR_DETAIL_CHARS]

    # ---- generation ----------------------------------------------------------------------------------------------
    async def generate(self, spec: GenerationSpec, messages: Sequence[Mapping[str, str]], *, attempt_id: str,
                       lock: EndpointLock, sampling: Mapping[str, Any]) -> GenerationOutcome:
        node_id = spec.node_id
        if not isinstance(attempt_id, str) or not attempt_id.startswith(f"{node_id}:") or \
                attempt_id == f"{node_id}:":
            raise RequestContractError(
                f"attempt_id {attempt_id!r} must be f'{{node_id}}:{{attempt_no}}' for node {node_id}")
        body = build_request_body(spec, messages, lock, sampling)
        content = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = self._client.build_request(
            "POST", f"{self._base_url}/chat/completions", content=content, headers=self._headers(json_body=True),
            extensions={REQUEST_EXTENSION_KEY: request_coordinates(spec, attempt_id)})
        started = time.monotonic()
        try:
            response = await self._client.send(request)
        except TRANSPORT_EXCEPTIONS as exc:
            latency = time.monotonic() - started
            status = classify_exception(exc)
            detail = self._scrub_text(f"{status.value}: {type(exc).__name__}: {exc}")
            result = GenerationResult(
                node_id=node_id, attempt_id=attempt_id, transport_status=status, completion_status=None,
                generation_id=None, observed_model=None, observed_provider=None, finish_reason=None,
                native_finish_reason=None, visible_content=None, reasoning_returned=False, usage=None,
                cache_status=None, raw_response_path=None, raw_response_sha256=None, latency_seconds=latency,
                http_status=None, retry_after_seconds=None, error_detail=detail)
            return GenerationOutcome(result=result, request_body=body, response_body=b"", response_headers={})
        latency = time.monotonic() - started
        return self._outcome(spec, attempt_id, body, lock, response, latency)

    def _outcome(self, spec: GenerationSpec, attempt_id: str, body: dict[str, Any], lock: EndpointLock,
                 response: httpx.Response, latency: float) -> GenerationOutcome:
        headers = redact_headers(dict(response.headers.items()))
        raw = self._scrub_body(response.content)
        cache_status = headers.get("x-openrouter-cache-status")
        http_status = response.status_code
        non_success = classify_http_status(http_status)
        if non_success is not None:
            try:
                payload: Any = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                payload = {"error": {"message": raw[:500].decode("utf-8", errors="replace")}} if raw else None
            detail = self._scrub_text(error_summary(payload, prefix=f"http {http_status}"))
            if http_status in GATEWAY_TIMEOUT_STATUSES:
                detail = self._scrub_text(f"gateway_timeout_may_be_billed {detail}")
            result = GenerationResult(
                node_id=spec.node_id, attempt_id=attempt_id, transport_status=non_success, completion_status=None,
                generation_id=None, observed_model=None, observed_provider=None, finish_reason=None,
                native_finish_reason=None, visible_content=None, reasoning_returned=False, usage=None,
                cache_status=cache_status, raw_response_path=None, raw_response_sha256=None,
                latency_seconds=latency, http_status=http_status,
                retry_after_seconds=parse_retry_after(headers.get("retry-after")), error_detail=detail)
        else:
            parsed = parse_completion(raw, cache_status, lock)
            result = GenerationResult(
                node_id=spec.node_id, attempt_id=attempt_id, transport_status=parsed.transport_status,
                completion_status=parsed.completion_status, generation_id=parsed.generation_id,
                observed_model=parsed.observed_model, observed_provider=parsed.observed_provider,
                finish_reason=parsed.finish_reason, native_finish_reason=parsed.native_finish_reason,
                visible_content=parsed.visible_content, reasoning_returned=parsed.reasoning_returned,
                usage=parsed.usage, cache_status=cache_status, raw_response_path=None, raw_response_sha256=None,
                latency_seconds=latency, http_status=http_status, retry_after_seconds=None,
                error_detail=None if parsed.error_detail is None else self._scrub_text(parsed.error_detail))
        return GenerationOutcome(result=result, request_body=body, response_body=raw, response_headers=headers)

    # ---- account and discovery endpoints -------------------------------------------------------------------------
    async def _get(self, path: str, *, params: Mapping[str, str] | None = None) -> httpx.Response:
        try:
            return await self._client.get(f"{self._base_url}/{path}", params=params,
                                          headers=self._headers(json_body=False))
        except TRANSPORT_EXCEPTIONS as exc:
            status = classify_exception(exc)
            raise OpenRouterAccountError(self._scrub_text(f"GET /{path}: {type(exc).__name__}: {exc}"),
                                         transport_status=status) from None

    def _raise_for(self, path: str, response: httpx.Response) -> None:
        status = classify_http_status(response.status_code)
        if status is None:
            return
        try:
            payload: Any = json.loads(response.content)
        except (ValueError, UnicodeDecodeError):
            payload = None
        raise OpenRouterAccountError(
            self._scrub_text(error_summary(payload, prefix=f"GET /{path}: http {response.status_code}")),
            transport_status=status, http_status=response.status_code,
            retry_after_seconds=parse_retry_after(response.headers.get("retry-after")))

    def _data(self, path: str, response: httpx.Response) -> Mapping[str, Any]:
        try:
            payload = _decimalize(json.loads(response.content))
        except (ValueError, UnicodeDecodeError):
            raise OpenRouterAccountError(f"GET /{path}: response is not JSON",
                                         transport_status=TransportStatus.ERROR_ENVELOPE,
                                         http_status=response.status_code) from None
        if not isinstance(payload, Mapping) or payload.get("error") is not None or \
                not isinstance(payload.get("data"), Mapping):
            raise OpenRouterAccountError(self._scrub_text(error_summary(payload, prefix=f"GET /{path}: no data")),
                                         transport_status=TransportStatus.ERROR_ENVELOPE,
                                         http_status=response.status_code)
        return payload["data"]

    async def key_info(self) -> Mapping[str, Any]:
        response = await self._get("key")
        self._raise_for("key", response)
        data = self._data("key", response)
        return {k: (self._scrub_text(v) if isinstance(v, str) else v) for k, v in data.items() if k != "label"}

    async def credits(self) -> Mapping[str, Any]:
        response = await self._get("credits")
        self._raise_for("credits", response)
        return dict(self._data("credits", response))

    async def generation_metadata(self, generation_id: str) -> Mapping[str, Any] | None:
        if not isinstance(generation_id, str) or not generation_id:
            raise ValueError("generation_id must be a non-empty string")
        response = await self._get("generation", params={"id": generation_id})
        if response.status_code == 404:
            return None
        self._raise_for("generation", response)
        return dict(self._data("generation", response))

    async def list_endpoints(self, model: str) -> tuple[bytes, Mapping[str, Any]]:
        if not isinstance(model, str) or not _MODEL_SLUG.match(model):
            raise ValueError(f"not an OpenRouter model slug: {model!r}")
        path = f"models/{model}/endpoints"
        response = await self._get(path)
        self._raise_for(path, response)
        raw = response.content
        try:
            parsed = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise OpenRouterAccountError(f"GET /{path}: response is not JSON",
                                         transport_status=TransportStatus.ERROR_ENVELOPE,
                                         http_status=response.status_code) from None
        if not isinstance(parsed, Mapping):
            raise OpenRouterAccountError(f"GET /{path}: response is not a JSON object",
                                         transport_status=TransportStatus.ERROR_ENVELOPE,
                                         http_status=response.status_code)
        return raw, parsed

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._client.aclose()
