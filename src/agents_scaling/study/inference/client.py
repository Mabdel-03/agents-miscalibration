"""Endpoint pool and vLLM chat client for the study (WP2).

Spec: §3.6 (request identity; one committed request per id), §4.3 (the rendered prompt
token ids must equal what the server tokenized — a mismatch is a harness defect),
§6.4/§10.1 (frozen sampling recipe sent verbatim), §10.4 (exogenous faults: a fixed
maximum of two retries per canonical request, then ``INFRA_INCOMPLETE``; model EOS and
``length`` truncation are completed outcomes, never retried).  Architecture:
docs/study_v4/01_architecture.md §1.6, §3 steps 3 and 9.  Corrections:
04_critic_corrections.md P0-5 (SDK auto-retry disabled, 3,600 s timeout, the harness owns the
retry accounting) and §0 (``registry.list_live_servers``/``wait_for_server`` signatures).

Wire contract (verified live on vLLM 0.21.0, ``docs/study_v4/05_vllm_response_shape.md``):
``chat.completions.create(model, messages, temperature, max_tokens, seed, extra_body={
chat_template_kwargs, top_k, top_p, min_p, presence_penalty, repetition_penalty,
return_token_ids: True[, guided_json]})``; the response carries top-level
``prompt_token_ids``, ``choices[0].token_ids``, ``choices[0].message.reasoning`` (the
qwen3 reasoning parser's channel; ``reasoning_content`` is read as a fallback and the field
actually used is recorded) and ``usage``.

Failure classes (types.py): :class:`ProtocolError` for anything that means the harness or
the server contract is broken (prompt-id mismatch, missing token ids, count disagreement,
guided_json without a schema); :class:`InfraFailure` after the retry cap or on a
non-exogenous API error (4xx).  Nothing here touches the request store.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai

from agents_scaling.serving import registry
from agents_scaling.serving.context import rendered_chat_token_ids
from agents_scaling.serving.registry import ServerEntry
from agents_scaling.study import identity
from agents_scaling.study.types import (
    Checkpoint,
    InfraFailure,
    ProtocolError,
    RequestRecord,
    RequestSpec,
)

RECORD_SCHEMA_VERSION = 1
DEFAULT_REQUEST_TIMEOUT_S = 3600.0
DEFAULT_MAX_EXOGENOUS_RETRIES = 2
COMPLETED_FINISH_REASONS: frozenset[str] = frozenset({"stop", "length"})
#: The only SDK exceptions treated as exogenous (§10.4).  ``APITimeoutError`` is a subclass
#: of ``APIConnectionError``; ``InternalServerError`` covers every 5xx status.
EXOGENOUS_ERRORS: tuple[type[Exception], ...] = (
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.RateLimitError,
)
THINK_END_TOKEN = "</think>"

#: Cost hook signature: ``cost_fn(checkpoint, prompt_tokens, completion_tokens) -> mapping``
#: with at least ``prefill``/``decode``/``total`` (the analytic oracle is WP4's,
#: architecture §4).  Without a hook the record carries explicit ``None`` placeholders.
CostFn = Callable[[Checkpoint, int, int], Mapping[str, Any]]
FLOPS_PLACEHOLDER: Mapping[str, Any] = {"prefill": None, "decode": None, "total": None, "oracle": None}

_MISSING = object()


class _AbortedCompletion(RuntimeError):
    """The server answered with a non-terminal ``finish_reason`` (e.g. ``abort``): exogenous."""


# --------------------------------------------------------------------------- endpoint pool


class EndpointPool:
    """Live endpoints of one serving profile, shard round-robined with failure rotation.

    ``endpoints()`` is ``registry.list_live_servers(server_run_root, profile)`` cached for
    ``refresh_s``; job-backed entries are validated by Slurm, job-less (fake) entries by
    ``GET /health``.  ``pick()`` returns ``endpoints[(shard + rotation) % n]``;
    :meth:`report_failure` rotates after ``failure_threshold`` (2) consecutive failures on
    one endpoint and forces a refresh at most once per ``refresh_min_interval_s`` (60 s)
    (architecture §1.6, §3 step 9).  ``wait()`` is ``registry.wait_for_server`` and lets its
    ``TimeoutError`` propagate: ``run_one`` maps it to ``EXIT_NO_SERVER`` (§3 step 3).
    """

    def __init__(
        self,
        server_run_root: str | os.PathLike,
        profile: str,
        shard: int = 0,
        refresh_s: float = 300.0,
        *,
        failure_threshold: int = 2,
        refresh_min_interval_s: float = 60.0,
        probe_timeout: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
        list_live: Callable[..., list[ServerEntry]] | None = None,
    ) -> None:
        if shard < 0:
            raise ValueError("shard must be >= 0")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.server_run_root = Path(server_run_root)
        self.profile = str(profile)
        self.shard = int(shard)
        self.refresh_s = float(refresh_s)
        self.failure_threshold = int(failure_threshold)
        self.refresh_min_interval_s = float(refresh_min_interval_s)
        self.probe_timeout = float(probe_timeout)
        self._clock = clock
        self._list_live = list_live or registry.list_live_servers
        self._lock = threading.Lock()
        self._entries: list[ServerEntry] | None = None
        self._refreshed_at: float | None = None
        self._rotation = 0
        self._consecutive: dict[tuple[str, int], int] = {}
        self.refresh_count = 0

    # ---- discovery ------------------------------------------------------------
    @property
    def rotation(self) -> int:
        return self._rotation

    def _refresh_locked(self) -> None:
        entries = list(self._list_live(self.server_run_root, self.profile, probe_timeout=self.probe_timeout))
        self._entries = entries
        self._refreshed_at = self._clock()
        self.refresh_count += 1
        live = {(e.host, e.port) for e in entries}
        self._consecutive = {k: v for k, v in self._consecutive.items() if k in live}

    def endpoints(self, *, force: bool = False) -> list[ServerEntry]:
        """Cached live endpoints (sorted by host, port); refreshed when stale or ``force``."""
        with self._lock:
            stale = self._entries is None or self._refreshed_at is None or (
                self._clock() - self._refreshed_at >= self.refresh_s
            )
            if force or stale:
                self._refresh_locked()
            assert self._entries is not None
            return list(self._entries)

    def pick(self) -> ServerEntry:
        """The shard's current endpoint; ``InfraFailure`` when no endpoint is live."""
        entries = self.endpoints()
        with self._lock:
            if not entries:
                raise InfraFailure(
                    f"no live {self.profile!r} endpoint registered under {self.server_run_root / 'servers'}"
                )
            return entries[(self.shard + self._rotation) % len(entries)]

    # ---- health feedback ----------------------------------------------------------
    def report_success(self, entry: ServerEntry) -> None:
        with self._lock:
            self._consecutive.pop((entry.host, entry.port), None)

    def report_failure(self, entry: ServerEntry) -> bool:
        """Count one exogenous failure on ``entry``; returns True when the pool rotated."""
        with self._lock:
            key = (entry.host, entry.port)
            count = self._consecutive.get(key, 0) + 1
            self._consecutive[key] = count
            if count < self.failure_threshold:
                return False
            self._consecutive.pop(key, None)
            self._rotation += 1
            now = self._clock()
            if self._refreshed_at is None or now - self._refreshed_at >= self.refresh_min_interval_s:
                self._refresh_locked()
            return True

    def wait(self, timeout_s: float = 300.0, poll_s: float = 5.0) -> ServerEntry:
        """Block until one live endpoint exists (``registry.wait_for_server``); ``TimeoutError`` otherwise."""
        entry = registry.wait_for_server(
            self.server_run_root, self.profile, shard=self.shard, timeout_s=timeout_s, poll_s=poll_s
        )
        self.endpoints(force=True)
        return entry


# --------------------------------------------------------------------------- response access


def _extension(obj: Any, name: str) -> tuple[bool, Any]:
    """(present, value) of an SDK model field, including vLLM extension fields."""
    value = getattr(obj, name, _MISSING)
    if value is not _MISSING and value is not None:
        return True, value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, Mapping) and name in extra:
        return True, extra[name]
    if value is not _MISSING:
        return True, value
    return False, None


def _int_list(value: Any, *, what: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProtocolError(f"server {what} is not a token-id array")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ProtocolError(f"server {what} contains a non-integer token id {item!r}")
        out.append(int(item))
    return out


@dataclass
class AttemptInfo:
    """One HTTP attempt of a request (recorded under ``timing.attempts``)."""

    n: int
    endpoint: dict[str, Any]
    started_at: float
    ended_at: float | None = None
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "endpoint": dict(self.endpoint),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": None if self.error is None else dict(self.error),
        }


@dataclass
class _Producer:
    cell_id: str | None
    run_id: str | None
    slurm_job_id: str | None = field(default_factory=lambda: os.environ.get("SLURM_JOB_ID"))
    host: str = field(default_factory=socket.gethostname)

    def to_dict(self) -> dict[str, Any]:
        return {"cell_id": self.cell_id, "run_id": self.run_id, "slurm_job_id": self.slurm_job_id, "host": self.host}


def endpoint_dict(entry: ServerEntry) -> dict[str, Any]:
    return {
        "host": entry.host,
        "port": int(entry.port),
        "slurm_job_id": entry.slurm_job_id,
        "serving_profile": entry.serving_profile,
        "base_url": entry.base_url,
    }


# --------------------------------------------------------------------------- client


class VllmChatClient:
    """Turn a :class:`RequestSpec` into a committed-shape :class:`RequestRecord` (§1.6).

    ``generate`` performs one HTTP call plus at most ``max_exogenous_retries`` retries on
    exogenous faults (connection/timeout/5xx/429, or an aborted completion), re-picking the
    endpoint every attempt and reporting failures to the pool.  Any other API error is an
    :class:`InfraFailure` without retry; contract violations are :class:`ProtocolError`.

    ``tokenizer`` must be the pinned checkpoint tokenizer (or the fake server's) so that
    ``prompt_token_ids`` equality (§4.3) is meaningful.  ``cost_fn`` is WP4's FLOP oracle
    hook (:data:`CostFn`); without it ``flops`` carries ``None`` placeholders.
    """

    def __init__(
        self,
        pool: EndpointPool,
        tokenizer: Any,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        max_exogenous_retries: int = DEFAULT_MAX_EXOGENOUS_RETRIES,
        *,
        cost_fn: CostFn | None = None,
        run_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_exogenous_retries < 0:
            raise ValueError("max_exogenous_retries must be >= 0")
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        self.pool = pool
        self.tokenizer = tokenizer
        self.request_timeout_s = float(request_timeout_s)
        self.max_exogenous_retries = int(max_exogenous_retries)
        self.cost_fn = cost_fn
        self.run_id = run_id
        self._clock = clock
        self._clients: dict[str, openai.OpenAI] = {}
        self._clients_lock = threading.Lock()
        self._think_end_id = self._resolve_think_end_id(tokenizer)

    # ---- helpers ------------------------------------------------------------------
    @staticmethod
    def _resolve_think_end_id(tokenizer: Any) -> int | None:
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if convert is None:
            return None
        try:
            value = convert(THINK_END_TOKEN)
        except Exception:
            return None
        unk = getattr(tokenizer, "unk_token_id", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value == unk:
            return None
        return int(value)

    def _sdk_client(self, entry: ServerEntry) -> openai.OpenAI:
        with self._clients_lock:
            client = self._clients.get(entry.base_url)
            if client is None:
                # P0-5: the harness owns retries and timeouts; the SDK must never retry.
                client = openai.OpenAI(
                    base_url=entry.base_url,
                    api_key="EMPTY",
                    max_retries=0,
                    timeout=self.request_timeout_s,
                )
                self._clients[entry.base_url] = client
            return client

    def render_prompt_ids(self, spec: RequestSpec) -> tuple[int, ...]:
        """Exact local render (``serving.context.rendered_chat_token_ids``; critic §0)."""
        return tuple(
            rendered_chat_token_ids(
                self.tokenizer, list(spec.messages), enable_thinking=bool(spec.decoding.enable_thinking)
            )
        )

    def chat_template_hash(self, spec: RequestSpec) -> str:
        """SHA-256 of the rendered chat-template text (audit §2.0 item 11 "rendered bytes hash")."""
        text = self.tokenizer.apply_chat_template(
            list(spec.messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(spec.decoding.enable_thinking),
        )
        if not isinstance(text, str):
            raise ProtocolError("tokenizer.apply_chat_template(tokenize=False) did not return text")
        return identity.sha256_hex(text)

    @staticmethod
    def request_kwargs(spec: RequestSpec, guided_json_schema: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Exactly what is sent to ``chat.completions.create`` (§10.1 recipe; 05_vllm_response_shape)."""
        decoding = spec.decoding
        if bool(decoding.guided_json) != (guided_json_schema is not None):
            raise ProtocolError(
                "guided_json flag and schema disagree: the Decoding identity says "
                f"guided_json={decoding.guided_json!r} but schema given={guided_json_schema is not None}"
            )
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": dict(spec.chat_template_kwargs),
            "top_k": int(decoding.top_k),
            "top_p": float(decoding.top_p),
            "min_p": float(decoding.min_p),
            "presence_penalty": float(decoding.presence_penalty),
            "repetition_penalty": float(decoding.repetition_penalty),
            "return_token_ids": True,
        }
        if guided_json_schema is not None:
            extra_body["guided_json"] = dict(guided_json_schema)
        return {
            "model": spec.checkpoint.served_model_name,
            "messages": [dict(m) for m in spec.messages],
            "temperature": float(decoding.temperature),
            "max_tokens": int(decoding.max_tokens),
            "seed": int(spec.engine_seed),
            "extra_body": extra_body,
        }

    # ---- main entry -------------------------------------------------------------------
    def generate(
        self,
        spec: RequestSpec,
        *,
        cell_id: str | None = None,
        guided_json_schema: Mapping[str, Any] | None = None,
    ) -> RequestRecord:
        """One canonical request → a :class:`RequestRecord` with ``content_sha256`` set.

        Raises :class:`InfraFailure` after ``1 + max_exogenous_retries`` failed attempts
        (the exception carries ``.attempts``: the per-attempt endpoint/error list) or on a
        non-exogenous API error; :class:`ProtocolError` on a contract violation (never
        retried: the same request would fail identically).
        """
        kwargs = self.request_kwargs(spec, guided_json_schema)
        local_ids = self.render_prompt_ids(spec)
        template_hash = self.chat_template_hash(spec)
        attempts: list[AttemptInfo] = []
        submitted_at = self._clock()
        max_attempts = 1 + self.max_exogenous_retries
        for n in range(1, max_attempts + 1):
            entry = self.pool.pick()
            attempt = AttemptInfo(n=n, endpoint=endpoint_dict(entry), started_at=self._clock())
            attempts.append(attempt)
            try:
                response = self._sdk_client(entry).chat.completions.create(**kwargs)
                parsed = self._parse_response(response, local_ids)
            except EXOGENOUS_ERRORS + (_AbortedCompletion,) as exc:
                attempt.ended_at = self._clock()
                attempt.error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
                self.pool.report_failure(entry)
                continue
            except ProtocolError:
                attempt.ended_at = self._clock()
                raise
            except openai.APIError as exc:
                attempt.ended_at = self._clock()
                attempt.error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
                failure = InfraFailure(
                    f"non-exogenous API error on attempt {n} at {entry.host}:{entry.port}: "
                    f"{type(exc).__name__}: {exc}"
                )
                failure.attempts = [a.to_dict() for a in attempts]  # type: ignore[attr-defined]
                raise failure from exc
            attempt.ended_at = self._clock()
            self.pool.report_success(entry)
            return self._build_record(
                spec,
                parsed,
                local_ids=local_ids,
                template_hash=template_hash,
                entry=entry,
                attempts=attempts,
                submitted_at=submitted_at,
                cell_id=cell_id,
            )
        summary = "; ".join(f"#{a.n} {a.endpoint['host']}:{a.endpoint['port']} {a.error['type']}" for a in attempts if a.error)
        failure = InfraFailure(
            f"request {spec.request_id[:12]} failed after {max_attempts} attempts "
            f"(max {self.max_exogenous_retries} exogenous retries, §10.4): {summary}"
        )
        failure.attempts = [a.to_dict() for a in attempts]  # type: ignore[attr-defined]
        raise failure

    # ---- response parsing --------------------------------------------------------------
    def _parse_response(self, response: Any, local_ids: tuple[int, ...]) -> dict[str, Any]:
        choices = getattr(response, "choices", None)
        if not choices or len(choices) != 1:
            raise ProtocolError(f"server returned {0 if not choices else len(choices)} choices, expected exactly 1")
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise ProtocolError("server choice has no message")
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason not in COMPLETED_FINISH_REASONS:
            # e.g. "abort" on an engine restart — an exogenous, retriable condition.
            raise _AbortedCompletion(f"finish_reason={finish_reason!r}")

        present, prompt_ids_raw = _extension(response, "prompt_token_ids")
        if not present or prompt_ids_raw is None:
            raise ProtocolError("server response lacks prompt_token_ids (return_token_ids unsupported?)")
        prompt_ids = _int_list(prompt_ids_raw, what="prompt_token_ids")
        if tuple(prompt_ids) != tuple(local_ids):
            raise ProtocolError(
                "server prompt_token_ids disagree with the local chat-template render (§4.3): "
                f"local={len(local_ids)} tokens sha256={identity.sha256_hex(identity.jcs(list(local_ids)))[:16]}, "
                f"server={len(prompt_ids)} tokens sha256={identity.sha256_hex(identity.jcs(prompt_ids))[:16]}"
            )
        present, token_ids_raw = _extension(choice, "token_ids")
        if not present or token_ids_raw is None:
            raise ProtocolError("server choice lacks token_ids (return_token_ids unsupported?)")
        token_ids = _int_list(token_ids_raw, what="token_ids")

        content = getattr(message, "content", None)
        if content is not None and not isinstance(content, str):
            raise ProtocolError("server message.content is neither text nor null")
        reasoning_field: str | None = None
        reasoning: str | None = None
        for name in ("reasoning", "reasoning_content"):
            present, value = _extension(message, name)
            if present and value is not None:
                if not isinstance(value, str):
                    raise ProtocolError(f"server message.{name} is not text")
                reasoning_field, reasoning = name, value
                break

        usage = getattr(response, "usage", None)
        usage_dict: dict[str, Any] | None = None
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            for key, ids in (("prompt_tokens", prompt_ids), ("completion_tokens", token_ids)):
                count = usage_dict[key]
                if count is not None and int(count) != len(ids):
                    raise ProtocolError(f"usage.{key}={count} but the server returned {len(ids)} token ids")

        reasoning_tokens, reasoning_source = self._reasoning_tokens(reasoning, token_ids)
        return {
            "content": content,
            "reasoning": reasoning,
            "reasoning_field_name": reasoning_field,
            "token_ids": token_ids,
            "completion_tokens": len(token_ids),
            "reasoning_tokens": reasoning_tokens,
            "reasoning_tokens_source": reasoning_source,
            "finish_reason": finish_reason,
            "stop_reason": _extension(choice, "stop_reason")[1],
            "usage": usage_dict,
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "prompt_token_ids": prompt_ids,
        }

    def _reasoning_tokens(self, reasoning: str | None, token_ids: list[int]) -> tuple[int, str]:
        """Tokens of the thinking channel: up to and including the first ``</think>`` id when the
        tokenizer knows it (the qwen3 parser's split), else the encoded reasoning text."""
        if reasoning is None:
            return 0, "none"
        if self._think_end_id is not None and self._think_end_id in token_ids:
            return token_ids.index(self._think_end_id) + 1, "think_end_id"
        encode = getattr(self.tokenizer, "encode", None)
        if encode is None:
            raise ProtocolError("tokenizer cannot count reasoning tokens (no </think> id and no encode())")
        return len(encode(reasoning, add_special_tokens=False)), "encoded_text"

    # ---- record assembly ------------------------------------------------------------------
    def _build_record(
        self,
        spec: RequestSpec,
        parsed: Mapping[str, Any],
        *,
        local_ids: tuple[int, ...],
        template_hash: str,
        entry: ServerEntry,
        attempts: list[AttemptInfo],
        submitted_at: float,
        cell_id: str | None,
    ) -> RequestRecord:
        completed_at = self._clock()
        last = attempts[-1]
        prompt_tokens = len(local_ids)
        completion_tokens = int(parsed["completion_tokens"])
        if self.cost_fn is not None:
            flops = dict(self.cost_fn(spec.checkpoint, prompt_tokens, completion_tokens))
            for key in ("prefill", "decode", "total"):
                if key not in flops:
                    raise ProtocolError(f"cost_fn result lacks {key!r}")
        else:
            flops = dict(FLOPS_PLACEHOLDER)
        ckpt = spec.checkpoint
        response = {k: v for k, v in parsed.items() if k != "prompt_token_ids"}
        record = RequestRecord(
            schema_version=RECORD_SCHEMA_VERSION,
            request_id=spec.request_id,
            identity=spec.identity_fields(),
            seed_key=spec.seed_key,
            engine_seed=int(spec.engine_seed),
            model={
                "hf_id": ckpt.hf_id,
                "served_model_name": ckpt.served_model_name,
                "profile": ckpt.profile,
                "tp_size": int(ckpt.tp_size),
                "size": ckpt.size,
                "model_revision": ckpt.model_revision,
            },
            messages=tuple(dict(m) for m in spec.messages),
            chat_template_kwargs=dict(spec.chat_template_kwargs),
            chat_template_hash=template_hash,
            sampling=spec.decoding.as_strings(),
            prompt_token_ids=tuple(local_ids),
            prompt_tokens=prompt_tokens,
            response=response,
            flops=flops,
            timing={
                "submitted_at": submitted_at,
                "completed_at": completed_at,
                "latency_s": (last.ended_at or completed_at) - last.started_at,
                "wall_s": completed_at - submitted_at,
                "attempts": [a.to_dict() for a in attempts],
            },
            endpoint=endpoint_dict(entry),
            attempts=len(attempts),
            producer=_Producer(cell_id=cell_id, run_id=self.run_id).to_dict(),
            content_sha256=None,
        )
        return record.with_content_sha256()


__all__ = [
    "AttemptInfo",
    "COMPLETED_FINISH_REASONS",
    "CostFn",
    "DEFAULT_MAX_EXOGENOUS_RETRIES",
    "DEFAULT_REQUEST_TIMEOUT_S",
    "EXOGENOUS_ERRORS",
    "EndpointPool",
    "FLOPS_PLACEHOLDER",
    "RECORD_SCHEMA_VERSION",
    "VllmChatClient",
    "endpoint_dict",
]
