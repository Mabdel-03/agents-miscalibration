"""WP-B: the metering proxy's HTTP/1.1 front (contract §6; plan §5) on ``asyncio.start_server`` + locked ``h11``.

Routes (all under ``/r/<run_key>/v1/`` with ``Authorization: Bearer <run_key>``, else 401 ``bad_bearer``):

* ``POST chat/completions`` (OpenClaude, H0, fake harness) -> ``openai_format``;
* ``POST responses`` (Codex) -> ``responses_format``;
* ``GET models`` (one model), ``GET budget`` (``ProxyCore.budget_view``); ``GET /healthz`` needs no bearer.

Request bodies may be ``Content-Length`` or chunked (h11 handles both) and may carry ``Content-Encoding: gzip``
(Codex's optional request compression). Connections are kept alive across cycles. A streaming reply waits at most
``refusal_grace_seconds`` (default ``min(2, keepalive_seconds)``) for the core to refuse the call outright, then
sends the response headers and an SSE comment ``: dynorch keepalive\\n\\n`` every ``keepalive_seconds`` while the
core call is pending (plan §2 B9: OpenClaude's ``API_TIMEOUT_MS`` and Codex's ``stream_idle_timeout_ms`` count from
the last byte), then the buffered result as chunks. The core call is *shielded*: a client that disconnects mid-call
does not cancel the upstream call (the core finishes, settles and caches it; contract §3 duplicate protection).

Every ``ProxyError`` raised by the core is converted here and nowhere else: before the headers went out as an HTTP
status + JSON body (``openai_format.error_json``), after them as an in-stream error frame (chat) or a
``response.failed`` event (Responses). The refusal grace exists for the first form: a budget refusal (402), a
tool-surface violation (400) or a bad bearer (401) is decided in microseconds without an upstream call, and the
real HTTP status is what the harnesses were verified NOT to retry (captures/openclaude/c: HTTP 402 -> no retry, no
further request, exit 1), whereas the same 402 as an error frame inside a ``200`` stream is retried once per
conversation by OpenClaude (Gate 0 smoke, WP-E 2026-09-23; docs/dynorch/requests/wpE.md item 14, proposed contract
§10.19). Any other exception is a proxy bug:
it is logged with its traceback and answered 500 ``internal_error`` so the harness fails loudly instead of hanging.

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

__all__ = ["DEFAULT_MODEL_ID", "ProxyServer", "REFUSAL_GRACE_SECONDS", "ReplyLog", "ServedError", "ServedReply", "route",
           "KEEPALIVE_COMMENT", "SUBAGENT_HEADER"]

log = logging.getLogger("agent_allocation.dynorch.httpd")

DEFAULT_MODEL_ID = "qwen/qwen3.8-27b"
KEEPALIVE_COMMENT = b": dynorch keepalive\n\n"
#: How long a streaming request waits for an outright refusal before the headers go out (module docstring).
REFUSAL_GRACE_SECONDS = 2.0
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
    delivered: bool = True      # False when the client went away before the reply could be written


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

    def record(self, run_key: str, reply: ChatReply, *, delivered: bool = True) -> None:
        served = ServedReply(call_no=reply.call.call_no, conversation_class=reply.call.conversation_class,
                             content=reply.content, finish_reason=reply.finish_reason, tool_calls=len(reply.tool_calls),
                             wire=reply.call.wire, delivered=delivered)
        self.served.setdefault(run_key, []).append(served)
        self.events.setdefault(run_key, []).append(served)

    def record_error(self, run_key: str, err: ProxyError, *, wire: str) -> None:
        self.events.setdefault(run_key, []).append(ServedError(status=err.status, code=err.code, wire=wire))

    def last_event(self, run_key: str) -> ServedReply | ServedError | None:
        """The run's last served reply or error, ignoring ``run_closed`` refusals: those answer a call made after
        the harness exited or the record was closed (contract §10.33) -- nobody is listening, and the frozen rerun
        rule decides on what the harness actually got."""
        for event in reversed(self.events.get(run_key, [])):
            if isinstance(event, ServedError) and event.code == "run_closed":
                continue
            return event
        return None

    def delivered(self, run_key: str) -> tuple[str | None, str]:
        """``(message, source)``: the last root reply with non-empty visible content that was actually handed to
        the client -> ``proxy_root``; else ``(None, "none")``. Tool-call-only turns (empty content) are skipped;
        sub-agent text is never used; a reply the client never received (its harness was gone) is never the
        delivered answer (plan B7 read with the review's rule: never credit an answer the harness did not get)."""
        for served in reversed(self.served.get(run_key, [])):
            if served.delivered and served.conversation_class == "root" and served.content.strip():
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
                 model_id: str = DEFAULT_MODEL_ID, reply_log: ReplyLog | None = None,
                 refusal_grace_seconds: float | None = None) -> None:
        if keepalive_seconds <= 0:
            raise ValueError("keepalive_seconds must be positive")
        if refusal_grace_seconds is not None and refusal_grace_seconds < 0:
            raise ValueError("refusal_grace_seconds must not be negative")
        self.core = core
        self.host = host
        self.requested_port = port
        self.keepalive_seconds = keepalive_seconds
        self.refusal_grace_seconds = (min(REFUSAL_GRACE_SECONDS, keepalive_seconds) if refusal_grace_seconds is None
                                      else float(refusal_grace_seconds))
        self.model_id = model_id
        self.reply_log = reply_log if reply_log is not None else ReplyLog()
        self._server: asyncio.base_events.Server | None = None
        self._background: set[asyncio.Task[Any]] = set()
        self._parked: dict[str, set[asyncio.Task[Any]]] = {}
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

    async def wait_for_run(self, run_key: str, timeout: float = 60.0) -> int:
        """Await the calls of ``run_key`` the proxy is still finishing for a client that went away (they are
        shielded: the upstream call is paid for and must be settled and journaled). Returns how many were awaited."""
        tasks = list(self._parked.get(run_key, ()))
        if not tasks:
            return 0
        await asyncio.wait(tasks, timeout=timeout)
        return len(tasks)

    async def wait_idle(self, timeout: float = 60.0) -> int:
        """Await every parked call before the server goes away (contract §10.24): cancelling one would abandon a
        SENT ledger reservation with nothing journaled, which ``open_reservations_zero`` then reports as a stop."""
        tasks = list(self._background)
        if not tasks:
            return 0
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            log.error("dynorch httpd: %d parked call(s) did not settle within %.0f s", len(pending), timeout)
        return len(done)

    async def close(self, *, drain_seconds: float = 60.0) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        await self.wait_idle(drain_seconds)             # never cancel a call that already left for the upstream
        for task in list(self._background):
            task.cancel()
        self._background.clear()
        self._parked.clear()

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
                await self._dispatch(conn, writer, request, reader=reader)
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

    async def _dispatch(self, conn: h11.Connection, writer: asyncio.StreamWriter, request: _Request, *,
                        reader: asyncio.StreamReader | None = None) -> None:
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
                wire = "chat"
            else:
                chat_request = responses_format.parse_responses_body(body, subagent=subagent)
                wire = "responses"
            gone = asyncio.Event()
            watcher = asyncio.ensure_future(self._watch_eof(reader, conn, gone)) if reader is not None else None
            try:
                await self._complete(conn, writer, run_key, chat_request, wire=wire, gone=gone)
            finally:
                if watcher is not None:
                    watcher.cancel()
                    await asyncio.gather(watcher, return_exceptions=True)
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
        open_runs = getattr(self.core, "open_runs", None)          # WP-A's ProxyCore.open_runs (int property)
        runs = getattr(self.core, "runs", None)                    # the stub's mapping
        if isinstance(open_runs, int):
            payload["runs_open"] = open_runs
        elif isinstance(runs, Mapping):
            payload["runs_open"] = len(runs)
        return payload

    # -- completions -------------------------------------------------------------------------------------
    @staticmethod
    async def _watch_eof(reader: asyncio.StreamReader, conn: h11.Connection, gone: asyncio.Event) -> None:
        """While a completion is being produced, read the connection so a client that goes away is noticed AT
        ONCE (review finding 2), not at the next keepalive write 15 s later or at the final write (which succeeds
        on a half-closed socket, finding 10). EOF or a reset sets ``gone``; bytes (a pipelined next request, which
        these clients never send) are handed to h11, which buffers them for the next cycle."""
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    gone.set()
                    return
                conn.receive_data(data)
        except (ConnectionResetError, BrokenPipeError, OSError):
            gone.set()

    async def _complete(self, conn: h11.Connection, writer: asyncio.StreamWriter, run_key: str,
                        chat_request: ChatRequest, *, wire: str, gone: asyncio.Event | None = None) -> None:
        gone = gone if gone is not None else asyncio.Event()
        namespaces = {tool.name: tool.namespace for tool in chat_request.tools}
        response_id = f"resp_{run_key[:12]}_{self.requests_served}"
        if not chat_request.stream:
            task = asyncio.ensure_future(self._call_core(run_key, chat_request, gone))
            try:
                reply = await self._await_or_gone(task, gone)
            except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
                self._keep_running(task, run_key)
                raise
            if wire == "chat":
                payload = openai_format.chat_reply_json(reply, model=self.model_id)
            else:
                payload = responses_format.responses_json(reply, response_id=response_id, model=self.model_id,
                                                          namespaces=namespaces)
            await self._deliver(run_key, reply, self._send_json(conn, writer, 200, payload, extra=reply.headers),
                                gone)
            return
        # Streaming: give the core ``refusal_grace_seconds`` to refuse outright (a ProxyError raised here propagates
        # to _dispatch and is answered with its real HTTP status: no byte has gone out yet), then headers, keepalive
        # comments while the core works, and the buffered result.
        task = asyncio.ensure_future(self._call_core(run_key, chat_request, gone))
        reply: ChatReply | None = None
        try:
            reply = await self._await_or_gone(task, gone, timeout=self.refusal_grace_seconds)
        except asyncio.TimeoutError:
            reply = None
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            self._keep_running(task, run_key)
            raise
        writer.write(conn.send(h11.Response(status_code=200, headers=self._headers("text/event-stream"))))
        await writer.drain()
        try:
            if reply is None:
                reply = await self._await_with_keepalives(task, conn, writer, run_key, gone)
        except ProxyError as err:
            self.reply_log.record_error(run_key, err, wire=wire)
            if wire == "chat":
                frames: Iterable[bytes] = openai_format.error_sse(err)
            else:
                frames = responses_format.responses_failed_sse(err, response_id=response_id, model=self.model_id)
            await self._write_frames(conn, writer, frames)
            return
        if wire == "chat":
            frames = list(openai_format.chat_reply_sse(reply, model=self.model_id))
        else:
            frames = list(responses_format.responses_sse(reply, response_id=response_id, model=self.model_id,
                                                         namespaces=namespaces))
        await self._deliver(run_key, reply, self._write_frames(conn, writer, frames), gone)

    async def _deliver(self, run_key: str, reply: ChatReply, write: Any, gone: asyncio.Event | None = None) -> None:
        """Write a finished reply to the client and record whether it actually got there: a stream cut mid-write
        (Codex's dropped stream, a dead harness) leaves the reply UNDELIVERED -- cached for the harness's identical
        retry (B9) and never the delivered answer (B7). The first write after the peer's FIN still succeeds, so the
        write alone is not proof (review finding 10): the connection's EOF watcher is consulted before the write
        and once more after the drain (one loop turn later, so an EOF already queued is seen)."""
        if gone is not None and gone.is_set():
            write.close()                                     # never started: the client is already gone
            self.reply_log.record(run_key, reply, delivered=False)
            self._undelivered(run_key, reply)
            raise ConnectionResetError("client went away before the reply could be written")
        try:
            await write
            if gone is not None:
                await asyncio.sleep(0)
                if gone.is_set():
                    raise ConnectionResetError("client went away while the reply was being written")
        except BaseException:
            self.reply_log.record(run_key, reply, delivered=False)
            self._undelivered(run_key, reply)
            raise
        self.reply_log.record(run_key, reply)
        self._delivered(run_key, reply)

    async def _call_core(self, run_key: str, chat_request: ChatRequest, gone: asyncio.Event | None = None
                         ) -> ChatReply:
        if gone is not None and getattr(self.core, "accepts_client_gone", False) is True:
            reply = await self.core.chat(run_key, chat_request, client_gone=gone)
        else:
            reply = await self.core.chat(run_key, chat_request)
        if not isinstance(reply, ChatReply):
            raise TypeError(f"core.chat returned {type(reply).__name__}, expected ChatReply")
        return reply

    @staticmethod
    async def _await_or_gone(task: "asyncio.Future[ChatReply]", gone: asyncio.Event,
                             timeout: float | None = None) -> ChatReply:
        """Await the (shielded) core call; raise ``ConnectionResetError`` as soon as the client is gone (the caller
        parks the task: the call is paid for and is finished, settled and journaled regardless), and
        ``asyncio.TimeoutError`` after ``timeout``."""
        waiter = asyncio.ensure_future(gone.wait())
        try:
            done, _ = await asyncio.wait({asyncio.shield(task), waiter}, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        if task.done():
            return task.result()
        if gone.is_set():
            raise ConnectionResetError("client went away while its call was being served")
        raise asyncio.TimeoutError()

    async def _await_with_keepalives(self, task: "asyncio.Future[ChatReply]", conn: h11.Connection,
                                     writer: asyncio.StreamWriter, run_key: str,
                                     gone: asyncio.Event | None = None) -> ChatReply:
        """Await the shielded core call, writing a keepalive comment whenever ``keepalive_seconds`` pass; a client
        that goes away parks the call at once (``_keep_running``)."""
        gone = gone if gone is not None else asyncio.Event()
        try:
            while True:
                try:
                    return await self._await_or_gone(task, gone, timeout=self.keepalive_seconds)
                except asyncio.TimeoutError:
                    writer.write(conn.send(h11.Data(data=KEEPALIVE_COMMENT)))
                    await writer.drain()
        except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
            self._keep_running(task, run_key)
            raise

    def _keep_running(self, task: "asyncio.Future[ChatReply]", run_key: str) -> None:
        """A disconnected client must not cancel the upstream call: park the SHIELDED task until it settles, then
        tell the core its reply was never delivered (B9: it is cached for a genuine retry of that request; B7: it
        is not the delivered answer). Shielding matters because a later ``close()`` cancels the keeper, and an
        unshielded await would propagate that cancellation into ``Upstream.complete`` after ``mark_sent``."""
        shielded = asyncio.shield(task)

        async def drain() -> None:
            try:
                reply = await shielded
            except BaseException:  # noqa: BLE001 - the outcome is journaled by the core; nobody is listening
                return
            if isinstance(reply, ChatReply):
                self.reply_log.record(run_key, reply, delivered=False)
                self._undelivered(run_key, reply)

        keeper = asyncio.ensure_future(drain())
        self._background.add(keeper)
        self._parked.setdefault(run_key, set()).add(keeper)
        keeper.add_done_callback(self._background.discard)
        keeper.add_done_callback(lambda t, k=run_key: self._parked.get(k, set()).discard(t))

    def _undelivered(self, run_key: str, reply: ChatReply) -> None:
        self._notify(run_key, reply, "mark_undelivered")

    def _delivered(self, run_key: str, reply: ChatReply) -> None:
        self._notify(run_key, reply, "mark_delivered")

    def _notify(self, run_key: str, reply: ChatReply, method: str) -> None:
        hook = getattr(self.core, method, None)
        if hook is None:
            return
        try:
            hook(run_key, reply.call.call_no)
        except Exception as exc:  # noqa: BLE001 - bookkeeping must never break the connection loop
            log.warning("dynorch httpd: %s(%s, %d) failed: %s", method, run_key, reply.call.call_no, exc)

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
