"""WP-B: the metering proxy's HTTP/1.1 front (contract §6; plan §5) on ``asyncio.start_server`` + locked ``h11``.

Routes (all under ``/r/<run_key>/v1/`` with ``Authorization: Bearer <run_key>``, else 401 ``bad_bearer``):

* ``POST chat/completions`` (OpenClaude, H0, fake harness) -> ``openai_format``;
* ``POST responses`` (Codex) -> ``responses_format``;
* ``GET models`` (one model), ``GET budget`` (``ProxyCore.budget_view``); ``GET /healthz`` needs no bearer.

Request bodies may be ``Content-Length`` or chunked (h11 handles both) and may carry ``Content-Encoding: gzip``
(Codex's optional request compression). Connections are kept alive across cycles. A streaming reply sends the
response headers at once and an SSE comment ``: dynorch keepalive\\n\\n`` every ``keepalive_seconds`` while the core
call is pending (plan §2 B9: OpenClaude's ``API_TIMEOUT_MS`` and Codex's ``stream_idle_timeout_ms`` count from the
last byte), then the buffered result as chunks. The core call is *shielded*: a client that disconnects mid-call
does not cancel the upstream call (the core finishes, settles and caches it; contract §3 duplicate protection).

Every ``ProxyError`` raised by the core is converted here and nowhere else: before the headers went out as an HTTP
status + JSON body (``openai_format.error_json``), after them as an in-stream error frame (chat) or a
``response.failed`` event (Responses). Any other exception is a proxy bug: it is logged with its traceback and
answered 500 ``internal_error`` so the harness fails loudly instead of hanging.

``ReplyLog`` keeps, per run, the replies served in order (``conversation_class``, content, finish reason): the
runner reads the delivered answer from it (plan §2 B7: the last root-conversation assistant message with non-empty
content *as served by the proxy*), so the wire layer, not the harness, is the source of truth.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping
from urllib.parse import unquote

import h11

from agent_allocation.dynorch import openai_format, responses_format
from agent_allocation.dynorch.records import ChatReply, ChatRequest, ProxyError

__all__ = ["DEFAULT_MODEL_ID", "ProxyServer", "ReplyLog", "ServedError", "ServedReply", "route", "KEEPALIVE_COMMENT",
           "SUBAGENT_HEADER"]

log = logging.getLogger("agent_allocation.dynorch.httpd")

DEFAULT_MODEL_ID = "qwen/qwen3.8-27b"
KEEPALIVE_COMMENT = b": dynorch keepalive\n\n"
SUBAGENT_HEADER = "x-openai-subagent"
_ROUTE = re.compile(r"^/r/(?P<run_key>[A-Za-z0-9_\-]{1,64})/v1/(?P<verb>chat/completions|responses|models|budget)$")
_MAX_BODY = 64 * 1024 * 1024


@dataclass(frozen=True)
class ServedReply:
    call_no: int
    conversation_class: str
    content: str
    finish_reason: str
    tool_calls: int
    wire: str
    kind: str = "reply"


@dataclass(frozen=True)
class ServedError:
    """A ``ProxyError`` the proxy answered with (the runner's rerun rule reads the last one)."""
    status: int
    code: str
    wire: str
    kind: str = "error"


@dataclass
class ReplyLog:
    """Per-run log of served replies and errors, in order; ``delivered`` applies plan §2 B7."""
    served: dict[str, list[ServedReply]] = field(default_factory=dict)
    events: dict[str, list[ServedReply | ServedError]] = field(default_factory=dict)

    def record(self, run_key: str, reply: ChatReply) -> None:
        served = ServedReply(call_no=reply.call.call_no, conversation_class=reply.call.conversation_class,
                             content=reply.content, finish_reason=reply.finish_reason, tool_calls=len(reply.tool_calls),
                             wire=reply.call.wire)
        self.served.setdefault(run_key, []).append(served)
        self.events.setdefault(run_key, []).append(served)

    def record_error(self, run_key: str, err: ProxyError, *, wire: str) -> None:
        self.events.setdefault(run_key, []).append(ServedError(status=err.status, code=err.code, wire=wire))

    def last_event(self, run_key: str) -> ServedReply | ServedError | None:
        events = self.events.get(run_key)
        return events[-1] if events else None

    def delivered(self, run_key: str) -> tuple[str | None, str]:
        """``(message, source)``: the last root reply with non-empty visible content -> ``proxy_root``; else
        ``(None, "none")``. Tool-call-only turns (empty content) are skipped; sub-agent text is never used."""
        for served in reversed(self.served.get(run_key, [])):
            if served.conversation_class == "root" and served.content.strip():
                return served.content, "proxy_root"
        return None, "none"

    def forget(self, run_key: str) -> None:
        self.served.pop(run_key, None)
        self.events.pop(run_key, None)


def route(target: str) -> tuple[str, str | None, str | None]:
    """``(kind, run_key, verb)``: ``("healthz", None, None)``, ``("run", key, verb)`` or ``("unknown", None, None)``."""
    path = target.split("?", 1)[0]
    if path == "/healthz":
        return "healthz", None, None
    match = _ROUTE.match(path)
    if match is None:
        return "unknown", None, None
    return "run", unquote(match.group("run_key")), match.group("verb")


class _Request:
    def __init__(self, event: h11.Request) -> None:
        self.method = event.method.decode("ascii")
        self.target = event.target.decode("utf-8", "replace")
        self.headers: dict[str, str] = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in event.headers}
        self.body = bytearray()

    def bearer(self) -> str | None:
        value = self.headers.get("authorization", "")
        if value.lower().startswith("bearer "):
            return value[7:].strip()
        return None

    def json_body(self) -> Any:
        data = bytes(self.body)
        if self.headers.get("content-encoding", "").lower() == "gzip":
            try:
                data = gzip.decompress(data)
            except (OSError, EOFError) as exc:
                raise ProxyError(400, "bad_request", f"invalid gzip body: {exc}") from exc
        elif self.headers.get("content-encoding", "") not in ("", "identity"):
            raise ProxyError(400, "bad_request", f"unsupported Content-Encoding {self.headers['content-encoding']!r}")
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProxyError(400, "bad_request", f"body is not JSON: {exc}") from exc


class ProxyServer:
    """The loopback HTTP server. ``core`` is any object with ProxyCore's §3 ``chat``/``budget_view``."""

    def __init__(self, core: Any, *, host: str = "127.0.0.1", port: int = 0, keepalive_seconds: float = 15.0,
                 model_id: str = DEFAULT_MODEL_ID, reply_log: ReplyLog | None = None) -> None:
        if keepalive_seconds <= 0:
            raise ValueError("keepalive_seconds must be positive")
        self.core = core
        self.host = host
        self.requested_port = port
        self.keepalive_seconds = keepalive_seconds
        self.model_id = model_id
        self.reply_log = reply_log if reply_log is not None else ReplyLog()
        self._server: asyncio.base_events.Server | None = None
        self._background: set[asyncio.Task[Any]] = set()
        self.requests_served = 0

    # -- lifecycle ---------------------------------------------------------------------------------------
    async def start(self) -> "ProxyServer":
        if self._server is not None:
            raise RuntimeError("server already started")
        self._server = await asyncio.start_server(self._serve, host=self.host, port=self.requested_port)
        return self

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("server not started")
        return self._server.sockets[0].getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def run_base_url(self, run_key: str) -> str:
        return f"{self.base_url}/r/{run_key}/v1"

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        for task in list(self._background):
            task.cancel()
        self._background.clear()

    async def __aenter__(self) -> "ProxyServer":
        return await self.start()

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    # -- connection loop ---------------------------------------------------------------------------------
    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = h11.Connection(our_role=h11.SERVER, max_incomplete_event_size=_MAX_BODY)
        try:
            while True:
                request = await self._read_request(conn, reader)
                if request is None:
                    break
                self.requests_served += 1
                await self._dispatch(conn, writer, request)
                if conn.our_state is h11.MUST_CLOSE or conn.their_state is h11.MUST_CLOSE:
                    break
                if conn.our_state is h11.DONE and conn.their_state is h11.DONE:
                    conn.start_next_cycle()
                    continue
                break
        except h11.RemoteProtocolError as exc:
            await self._protocol_error(conn, writer, exc)
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, OSError):
                pass

    async def _read_request(self, conn: h11.Connection, reader: asyncio.StreamReader) -> _Request | None:
        request: _Request | None = None
        while True:
            event = conn.next_event()
            if event is h11.NEED_DATA:
                data = await reader.read(65536)
                conn.receive_data(data)
                continue
            if isinstance(event, h11.ConnectionClosed):
                return None
            if isinstance(event, h11.Request):
                request = _Request(event)
            elif isinstance(event, h11.Data):
                if request is None:
                    raise h11.RemoteProtocolError("body before request line")
                request.body.extend(event.data)
                if len(request.body) > _MAX_BODY:
                    raise h11.RemoteProtocolError("request body too large")
            elif isinstance(event, h11.EndOfMessage):
                if request is None:
                    raise h11.RemoteProtocolError("end of message before request line")
                return request
            else:
                raise h11.RemoteProtocolError(f"unexpected h11 event {type(event).__name__}")

    async def _protocol_error(self, conn: h11.Connection, writer: asyncio.StreamWriter,
                              exc: h11.RemoteProtocolError) -> None:
        if conn.our_state in (h11.IDLE, h11.SEND_RESPONSE):
            try:
                body = json.dumps({"error": {"code": "bad_request", "message": f"protocol error: {exc}"}}).encode()
                writer.write(conn.send(h11.Response(status_code=400, headers=self._headers(
                    "application/json", extra=[(b"content-length", str(len(body)).encode())]))))
                writer.write(conn.send(h11.Data(data=body)))
                writer.write(conn.send(h11.EndOfMessage()))
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError, h11.LocalProtocolError):
                pass

    # -- dispatch ----------------------------------------------------------------------------------------
    @staticmethod
    def _headers(content_type: str, *, extra: Iterable[tuple[bytes, bytes]] = ()) -> list[tuple[bytes, bytes]]:
        base = [(b"content-type", content_type.encode("ascii")), (b"cache-control", b"no-cache"),
                (b"server", b"dynorch")]
        base.extend(extra)
        return base

    async def _send_json(self, conn: h11.Connection, writer: asyncio.StreamWriter, status: int, payload: Any,
                         *, extra: Mapping[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = self._headers("application/json", extra=[(b"content-length", str(len(body)).encode())])
        for key, value in (extra or {}).items():
            headers.append((key.encode("ascii"), str(value).encode("utf-8")))
        writer.write(conn.send(h11.Response(status_code=status, headers=headers)))
        writer.write(conn.send(h11.Data(data=body)))
        writer.write(conn.send(h11.EndOfMessage()))
        await writer.drain()

    async def _dispatch(self, conn: h11.Connection, writer: asyncio.StreamWriter, request: _Request) -> None:
        kind, run_key, verb = route(request.target)
        try:
            if kind == "healthz":
                if request.method != "GET":
                    raise ProxyError(405, "bad_request", "GET only")
                await self._send_json(conn, writer, 200, self._health())
                return
            if kind != "run":
                raise ProxyError(404, "not_found", f"no route for {request.target}")
            assert run_key is not None and verb is not None
            if request.bearer() != run_key:
                raise ProxyError(401, "bad_bearer", "Authorization bearer must equal the path run_key")
            if verb == "models":
                if request.method != "GET":
                    raise ProxyError(405, "bad_request", "GET only")
                await self._send_json(conn, writer, 200, {"object": "list", "data": [
                    {"id": self.model_id, "object": "model", "owned_by": "dynorch"}]})
                return
            if verb == "budget":
                if request.method != "GET":
                    raise ProxyError(405, "bad_request", "GET only")
                await self._send_json(conn, writer, 200, dict(self.core.budget_view(run_key)))
                return
            if request.method != "POST":
                raise ProxyError(405, "bad_request", "POST only")
            body = request.json_body()
            subagent = SUBAGENT_HEADER in request.headers          # contract §10.1: Codex children carry it
            if verb == "chat/completions":
                chat_request = openai_format.parse_chat_body(body, subagent=subagent)
                await self._complete(conn, writer, run_key, chat_request, wire="chat")
            else:
                chat_request = responses_format.parse_responses_body(body, subagent=subagent)
                await self._complete(conn, writer, run_key, chat_request, wire="responses")
        except ProxyError as err:
            if run_key is not None and verb in ("chat/completions", "responses"):
                self.reply_log.record_error(run_key, err, wire="responses" if verb == "responses" else "chat")
            status, payload = openai_format.error_json(err)
            await self._send_json(conn, writer, status, payload)
        except (ConnectionResetError, BrokenPipeError):
            raise
        except Exception as exc:  # a proxy bug: loud 500, never a hang
            log.error("dynorch httpd internal error on %s %s: %s\n%s", request.method, request.target, exc,
                      traceback.format_exc())
            internal = ProxyError(500, "internal_error", f"{type(exc).__name__}: {exc}")
            if run_key is not None and verb in ("chat/completions", "responses"):
                self.reply_log.record_error(run_key, internal, wire="responses" if verb == "responses" else "chat")
            if conn.our_state is h11.SEND_RESPONSE:
                status, payload = openai_format.error_json(internal)
                await self._send_json(conn, writer, status, payload)
            else:
                raise

    def _health(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": True, "requests_served": self.requests_served}
        runs = getattr(self.core, "runs", None)
        if isinstance(runs, Mapping):
            payload["runs_open"] = len(runs)
        return payload

    # -- completions -------------------------------------------------------------------------------------
    async def _complete(self, conn: h11.Connection, writer: asyncio.StreamWriter, run_key: str,
                        chat_request: ChatRequest, *, wire: str) -> None:
        namespaces = {tool.name: tool.namespace for tool in chat_request.tools}
        response_id = f"resp_{run_key[:12]}_{self.requests_served}"
        if not chat_request.stream:
            reply = await self._call_core(run_key, chat_request)
            self.reply_log.record(run_key, reply)
            if wire == "chat":
                payload = openai_format.chat_reply_json(reply, model=self.model_id)
            else:
                payload = responses_format.responses_json(reply, response_id=response_id, model=self.model_id,
                                                          namespaces=namespaces)
            await self._send_json(conn, writer, 200, payload, extra=reply.headers)
            return
        # Streaming: headers first, keepalive comments while the core works, then the buffered result.
        writer.write(conn.send(h11.Response(status_code=200, headers=self._headers("text/event-stream"))))
        await writer.drain()
        task = asyncio.ensure_future(self._call_core(run_key, chat_request))
        try:
            reply = await self._await_with_keepalives(task, conn, writer)
        except ProxyError as err:
            self.reply_log.record_error(run_key, err, wire=wire)
            if wire == "chat":
                frames: Iterable[bytes] = openai_format.error_sse(err)
            else:
                frames = responses_format.responses_failed_sse(err, response_id=response_id, model=self.model_id)
            await self._write_frames(conn, writer, frames)
            return
        self.reply_log.record(run_key, reply)
        if wire == "chat":
            frames = list(openai_format.chat_reply_sse(reply, model=self.model_id))
        else:
            frames = list(responses_format.responses_sse(reply, response_id=response_id, model=self.model_id,
                                                         namespaces=namespaces))
        await self._write_frames(conn, writer, frames)

    async def _call_core(self, run_key: str, chat_request: ChatRequest) -> ChatReply:
        reply = await self.core.chat(run_key, chat_request)
        if not isinstance(reply, ChatReply):
            raise TypeError(f"core.chat returned {type(reply).__name__}, expected ChatReply")
        return reply

    async def _await_with_keepalives(self, task: "asyncio.Future[ChatReply]", conn: h11.Connection,
                                     writer: asyncio.StreamWriter) -> ChatReply:
        """Await the shielded core call, writing a keepalive comment whenever ``keepalive_seconds`` pass."""
        shielded = asyncio.shield(task)
        try:
            while True:
                try:
                    return await asyncio.wait_for(asyncio.shield(shielded), timeout=self.keepalive_seconds)
                except asyncio.TimeoutError:
                    writer.write(conn.send(h11.Data(data=KEEPALIVE_COMMENT)))
                    await writer.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            self._keep_running(task)
            raise

    def _keep_running(self, task: "asyncio.Future[ChatReply]") -> None:
        """A disconnected client must not cancel the upstream call: park the task until it settles."""
        async def drain() -> None:
            try:
                await task
            except BaseException:  # noqa: BLE001 - the outcome is journaled by the core; nobody is listening
                pass
        keeper = asyncio.ensure_future(drain())
        self._background.add(keeper)
        keeper.add_done_callback(self._background.discard)

    async def _write_frames(self, conn: h11.Connection, writer: asyncio.StreamWriter, frames: Iterable[bytes]) -> None:
        for frame in frames:
            writer.write(conn.send(h11.Data(data=frame)))
        writer.write(conn.send(h11.EndOfMessage()))
        await writer.drain()


async def serve_forever(core: Any, *, host: str = "127.0.0.1", port: int = 0, keepalive_seconds: float = 15.0,
                        on_ready: Callable[[ProxyServer], Awaitable[None] | None] | None = None) -> None:
    """Start the server and block until cancelled (for the WP-D CLI's proxy-only verb)."""
    server = ProxyServer(core, host=host, port=port, keepalive_seconds=keepalive_seconds)
    await server.start()
    try:
        if on_ready is not None:
            result = on_ready(server)
            if asyncio.iscoroutine(result):
                await result
        await asyncio.Event().wait()
    finally:
        await server.close()
