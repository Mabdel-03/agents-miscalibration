"""Fake vLLM 0.21 OpenAI-compatible server for the study tests (WP2).

Every study package tests its request path against this server instead of a GPU.  It
reproduces the *exact* response shape observed on the real fleet
(``docs/study_v4/05_vllm_response_shape.md``, re-verified by WP2's live smoke request on
2026-09-05): ``choices[0].message.content`` plus ``choices[0].message.reasoning`` (the
parsed ``<think>`` channel; ``reasoning_content`` never appears), ``choices[0].token_ids``
and top-level ``prompt_token_ids`` (``return_token_ids``), ``usage`` counts equal to the
token-id array lengths, ``finish_reason`` ``stop``/``length``.

Determinism (spec §3.6, §6.7 "adaptive policies replay deterministically"): the content of
a chat completion is a pure function of ``sha256(JCS(messages) + seed)`` and the server
configuration, so a re-run of the same request produces byte-identical output and the
content-addressed request store aliases it.

Output modes are detected from a frozen marker inside the prompt (the markers are literal
words of the frozen templates under ``study/prompts/templates``):

* ``coordinator_action`` → CEN_FLAT hub (§4.2): ``{"action":"final",...}`` or
  ``{"action":"delegate",...}`` per ``hub_mode``;
* ``subtask_result`` → CEN_FLAT worker: a ``subtask_result`` object per ``worker_mode``;
* ``quality_score`` → JUDGE_BEST (§4.4) score object;
* ``extracted_final_answer`` → the cais/hle judge (§3.3) as a JSON object;
* ``q_personal`` / ``PERSONAL_FINAL`` → §8.7 shadow forecast object;
* otherwise → a valid §3.5 candidate whose ``final_answer`` is drawn from a small
  per-item pool (so §4.4 VOTE is non-degenerate), or an invalid output at ``invalid_rate``.

Fault injection: :meth:`FakeVllmServer.fail_next` (HTTP 5xx/429 for the next *n* chat
requests), :meth:`FakeVllmServer.hang` (sleep before answering, to trigger client
timeouts) and :meth:`FakeVllmServer.kill` (close the listening socket → connection
refused).  ``guided_json`` in the request body is recorded and — like the real stack —
NOT enforced unless ``guided_json_mode="enforce"``.

The server never imports anything from ``agents_scaling.study`` except ``identity.jcs``
so the shape it emits cannot drift with the client under test.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from agents_scaling.study.identity import jcs

# Qwen3 control tokens (real ids), so the client's ``</think>`` split works on fake output.
THINK_START_ID = 151667
THINK_END_ID = 151668
IM_END_ID = 151645
IM_START_ID = 151644

HUB_MARKER = "coordinator_action"
WORKER_MARKER = "subtask_result"
JUDGE_BEST_MARKER = "quality_score"
HLE_JUDGE_MARKER = "extracted_final_answer"
FORECAST_MARKERS = ("q_personal", "PERSONAL_FINAL")

DEFAULT_ANSWER_POOL: tuple[str, ...] = ("A", "B", "C")
#: Cumulative draw weights over the pool, rotated per item so the plurality answer differs
#: between items but agreement inside one item stays common (votes are non-degenerate).
_POOL_WEIGHTS: tuple[float, ...] = (0.55, 0.30, 0.15)


# --------------------------------------------------------------------------- tokenizer


class FakeTokenizer:
    """Deterministic whitespace tokenizer shared by the fake server and the client under test.

    Unlike ``tests/study/conftest.py::StubTokenizer`` (whose ids grow with first use and so
    differ between instances), ids here are a pure function of the token text, so *any* two
    instances agree and the client's ``prompt_token_ids`` equality check (§4.3) passes
    against a server that never saw the client's instance.  ``<think>``/``</think>``/
    ``<|im_end|>``/``<|im_start|>`` map to the real Qwen3 control ids.

    Mirrors the transformers-5.9 surface used by ``serving.context.rendered_chat_token_ids``:
    ``apply_chat_template(tokenize=True)`` returns ``{"input_ids", "attention_mask"}``
    unless ``return_dict=False``; ``enable_thinking=False`` appends the empty think block.
    """

    _SPECIAL = {
        "<think>": THINK_START_ID,
        "</think>": THINK_END_ID,
        "<|im_end|>": IM_END_ID,
        "<|im_start|>": IM_START_ID,
    }

    def __init__(self) -> None:
        self._reverse: dict[int, str] = {v: k for k, v in self._SPECIAL.items()}
        self._lock = threading.Lock()

    @classmethod
    def token_id(cls, token: str) -> int:
        if token in cls._SPECIAL:
            return cls._SPECIAL[token]
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return 1000 + int.from_bytes(digest[:4], "big") % 150_000

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_id(token)

    def encode(self, text: str, add_special_tokens: bool = False, **_: Any) -> list[int]:
        ids: list[int] = []
        with self._lock:
            for token in text.split():
                token_id = self.token_id(token)
                self._reverse.setdefault(token_id, token)
                ids.append(token_id)
        return ids

    def decode(self, ids: Sequence[int], **_: Any) -> str:
        with self._lock:
            return " ".join(self._reverse.get(int(i), "<unk>") for i in ids)

    def render_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
    ) -> str:
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
        if add_generation_prompt:
            tail = "<|im_start|>assistant\n"
            if enable_thinking is False:
                tail += "<think>\n\n</think>\n\n"
            parts.append(tail)
        return "\n".join(parts)

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
        return_dict: bool = True,
        **_: Any,
    ) -> Any:
        text = self.render_chat(messages, add_generation_prompt, enable_thinking)
        if not tokenize:
            return text
        ids = self.encode(text)
        if return_dict:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return ids


# --------------------------------------------------------------------------- helpers


def request_key(messages: Sequence[Mapping[str, Any]], seed: int | None) -> str:
    """``sha256(JCS(messages) + str(seed))`` — the determinism key of one fake completion."""
    return hashlib.sha256(jcs(list(messages)) + str(seed).encode("utf-8")).hexdigest()


def _unit(key: str, salt: str) -> float:
    """Deterministic float in [0, 1) derived from ``key`` and a purpose ``salt``."""
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _prompt_text(messages: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def detect_mode(messages: Sequence[Mapping[str, Any]]) -> str:
    """Classify a prompt by its frozen template marker (see module docstring)."""
    text = _prompt_text(messages)
    if HUB_MARKER in text:
        return "hub"
    if WORKER_MARKER in text:
        return "worker"
    if JUDGE_BEST_MARKER in text:
        return "judge_best"
    if HLE_JUDGE_MARKER in text:
        return "hle_judge"
    if any(marker in text for marker in FORECAST_MARKERS):
        return "forecast"
    return "solver"


_TASK_RE = re.compile(r"Task:\s*(.{1,4096}?)(?:\n\s*Output contract:|\Z)", re.DOTALL)


def item_key(messages: Sequence[Mapping[str, Any]]) -> str:
    """Stable per-item key: the ``Task: ...`` span of the prompt when present, else the prompt."""
    text = _prompt_text(messages)
    match = _TASK_RE.search(text)
    span = match.group(1).strip() if match else text
    return hashlib.sha256(span.encode("utf-8")).hexdigest()


def _fence(content: str) -> str:
    return f"```json\n{content}\n```"


def _dumps(obj: Any) -> str:
    """JSON with spaces after separators so the whitespace tokenizer sees several tokens."""
    return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "), allow_nan=False)


def _example_from_schema(schema: Mapping[str, Any], key: str) -> Any:
    """Smallest instance of a JSON schema (``guided_json_mode="enforce"`` only)."""
    kind = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    if isinstance(kind, list):
        kind = [k for k in kind if k != "null"][0] if any(k != "null" for k in kind) else "null"
    if kind == "object":
        return {name: _example_from_schema(sub, key) for name, sub in schema.get("properties", {}).items()}
    if kind == "array":
        return []
    if kind == "string":
        return f"fake-{key[:8]}"
    if kind in ("number", "integer"):
        lo = schema.get("minimum", 0)
        return lo if kind == "integer" else float(lo) + 0.5 * (schema.get("maximum", lo + 1) - lo)
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    if "oneOf" in schema:
        return _example_from_schema(schema["oneOf"][0], key)
    raise ValueError(f"unsupported guided_json schema fragment: {schema!r}")


# --------------------------------------------------------------------------- server


class FakeVllmServer:
    """In-process fake of one vLLM 0.21 endpoint (see module docstring).

    Parameters
    ----------
    served_model_name
        The ``model`` id returned by ``/v1/models`` (``"32B"`` for ``32B-long``).
    tokenizer
        Tokenizer used for ``prompt_token_ids``/``/tokenize``.  The client under test must
        render with the same tokenizer (any :class:`FakeTokenizer` instance, or the very
        same ``StubTokenizer`` instance).
    invalid_rate
        Deterministic fraction of solver outputs that are not valid §3.5 JSON.
    fence_rate
        Fraction of outputs wrapped in one ```` ```json ```` fence (observed real behaviour).
    answer_pool
        The per-item ``final_answer`` pool (votes are non-degenerate: three draws of one
        item usually contain a plurality).
    hub_mode
        ``"final"`` | ``"delegate"`` | ``"alternate"`` (delegate, then final, per item) |
        ``"invalid"`` | a callable ``(prompt_text) -> mode``.
    worker_mode
        ``"complete"`` | ``"partial"`` | ``"failed"`` | ``"invalid"``.
    guided_json_mode
        ``"ignore"`` (real stack: accepted, not enforced) or ``"enforce"`` (emit the
        smallest instance of the schema, unfenced).
    """

    def __init__(
        self,
        *,
        served_model_name: str = "32B",
        tokenizer: Any | None = None,
        invalid_rate: float = 0.0,
        fence_rate: float = 0.0,
        answer_pool: Sequence[str] = DEFAULT_ANSWER_POOL,
        hub_mode: str | Callable[[str], str] = "final",
        delegate_count: int = 2,
        worker_mode: str = "complete",
        hle_judge_mode: str = "deterministic",
        guided_json_mode: str = "ignore",
        max_model_len: int = 40960,
        reasoning_words: int = 12,
        responder: Callable[[dict[str, Any]], str | None] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if not 0.0 <= invalid_rate <= 1.0 or not 0.0 <= fence_rate <= 1.0:
            raise ValueError("rates must lie in [0, 1]")
        if not answer_pool:
            raise ValueError("answer_pool must not be empty")
        if guided_json_mode not in ("ignore", "enforce"):
            raise ValueError("guided_json_mode must be 'ignore' or 'enforce'")
        if worker_mode not in ("complete", "partial", "failed", "invalid"):
            raise ValueError("worker_mode must be complete|partial|failed|invalid")
        self.served_model_name = served_model_name
        self.tokenizer = tokenizer if tokenizer is not None else FakeTokenizer()
        self.invalid_rate = float(invalid_rate)
        self.fence_rate = float(fence_rate)
        self.answer_pool = tuple(str(a) for a in answer_pool)
        self.hub_mode = hub_mode
        self.delegate_count = int(delegate_count)
        self.worker_mode = worker_mode
        self.hle_judge_mode = hle_judge_mode
        self.guided_json_mode = guided_json_mode
        self.max_model_len = int(max_model_len)
        self.reasoning_words = int(reasoning_words)
        self.responder = responder
        self.finish_reason_override: str | None = None

        self._lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.request_count = 0
        self.chat_count = 0
        self._fail_queue: list[tuple[int, str]] = []
        self._hang_s = 0.0
        self._hub_calls_per_item: dict[str, int] = {}
        self._killed = False

        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "fake-vllm/0.21.0"

            def log_message(self, *_: Any) -> None:  # silence
                return

            def do_GET(self) -> None:  # noqa: N802
                outer._handle(self, "GET")

            def do_POST(self) -> None:  # noqa: N802
                outer._handle(self, "POST")

        class _Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

            def handle_error(self, request: Any, client_address: Any) -> None:
                # A client that timed out (``hang``) closes its socket; ignore the pipe error.
                return

        self._httpd = _Server((host, int(port)), Handler)
        self.host, self.port = self._httpd.server_address[0], int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, name=f"fake-vllm:{self.port}", daemon=True)

    # ---- lifecycle ---------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> "FakeVllmServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        if not self._killed:
            self._killed = True
            self._httpd.shutdown()
            self._httpd.server_close()

    def kill(self) -> None:
        """Stop serving immediately; further connections are refused (APIConnectionError)."""
        self.stop()

    def __enter__(self) -> "FakeVllmServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ---- fault injection ---------------------------------------------------
    def fail_next(self, n: int, status: int = 500, message: str = "injected failure") -> None:
        """Answer the next ``n`` chat requests with HTTP ``status`` (500 → InternalServerError,
        429 → RateLimitError, 400 → BadRequestError: a non-exogenous fault)."""
        with self._lock:
            self._fail_queue.extend([(int(status), message)] * int(n))

    def hang(self, seconds: float) -> None:
        """Sleep ``seconds`` before answering every following chat request (0 disables)."""
        with self._lock:
            self._hang_s = float(seconds)

    def reset_faults(self) -> None:
        with self._lock:
            self._fail_queue.clear()
            self._hang_s = 0.0

    # ---- registry ------------------------------------------------------------
    def register(self, run_root: str | Path, profile: str) -> Path:
        return register_fake(run_root, profile, self.host, self.port)

    # ---- HTTP dispatch ---------------------------------------------------------
    def _handle(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        path = handler.path.split("?", 1)[0]
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length) if length else b""
        with self._lock:
            self.request_count += 1
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            self._send(handler, 400, {"error": {"message": "invalid JSON body", "type": "BadRequestError"}})
            return
        if method == "GET" and path == "/health":
            handler.send_response(200)
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        if method == "GET" and path == "/v1/models":
            self._send(handler, 200, self._models_payload())
            return
        if method == "POST" and path == "/tokenize":
            self._send(handler, *self._tokenize(body))
            return
        if method == "POST" and path == "/v1/chat/completions":
            self._send(handler, *self._chat(body))
            return
        self._send(handler, 404, {"detail": "Not Found"})

    def _send(self, handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _models_payload(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "vllm",
                    "root": self.served_model_name,
                    "parent": None,
                    "max_model_len": self.max_model_len,
                    "permission": [],
                }
            ],
        }

    def _tokenize(self, body: Any) -> tuple[int, Any]:
        if not isinstance(body, Mapping):
            return 400, {"error": {"message": "body must be an object", "type": "BadRequestError"}}
        if "messages" in body:
            kwargs = body.get("chat_template_kwargs") or {}
            ids = self.tokenizer.apply_chat_template(
                body["messages"],
                tokenize=True,
                add_generation_prompt=bool(body.get("add_generation_prompt", True)),
                enable_thinking=kwargs.get("enable_thinking"),
                return_dict=False,
            )
        elif "prompt" in body:
            ids = self.tokenizer.encode(str(body["prompt"]))
        else:
            return 400, {"error": {"message": "prompt or messages required", "type": "BadRequestError"}}
        ids = list(ids)
        return 200, {"count": len(ids), "max_model_len": self.max_model_len, "tokens": ids, "token_strs": None}

    # ---- chat completion -------------------------------------------------------
    def _chat(self, body: Any) -> tuple[int, Any]:
        if not isinstance(body, Mapping) or not isinstance(body.get("messages"), list):
            return 400, {"error": {"message": "messages required", "type": "BadRequestError", "code": 400}}
        with self._lock:
            self.chat_count += 1
            fault = self._fail_queue.pop(0) if self._fail_queue else None
            hang_s = self._hang_s
            self.requests.append(
                {
                    "path": "/v1/chat/completions",
                    "body": json.loads(json.dumps(body)),
                    "guided_json": body.get("guided_json"),
                    "received_at": time.time(),
                    "fault": fault,
                }
            )
        if hang_s > 0:
            time.sleep(hang_s)
        if fault is not None:
            status, message = fault
            return status, {"error": {"message": message, "type": "ServerError", "code": status}}
        if body.get("model") != self.served_model_name:
            return 404, {"error": {"message": f"The model `{body.get('model')}` does not exist.", "type": "NotFoundError", "code": 404}}
        return 200, self.completion_payload(body)

    def completion_payload(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """The full response object for ``body`` (pure: no counters, no faults)."""
        messages = body["messages"]
        seed = body.get("seed")
        kwargs = body.get("chat_template_kwargs") or {}
        enable_thinking = bool(kwargs.get("enable_thinking", True))
        max_tokens = int(body.get("max_tokens") or 16)
        key = request_key(messages, seed)
        mode = detect_mode(messages)

        content = None
        if self.responder is not None:
            content = self.responder({"messages": messages, "seed": seed, "mode": mode, "key": key, "body": dict(body)})
        if content is None:
            if self.guided_json_mode == "enforce" and isinstance(body.get("guided_json"), Mapping):
                content = _dumps(_example_from_schema(body["guided_json"], key))
            else:
                content = self._content_for(mode, messages, key)
                if _unit(key, "fence") < self.fence_rate:
                    content = _fence(content)

        prompt_ids = list(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking, return_dict=False
            )
        )

        reasoning: str | None = None
        completion_ids: list[int] = []
        if enable_thinking:
            words = [f"think-{key[:6]}-{i}" for i in range(self.reasoning_words)]
            reasoning = "\nOkay, let me reason about this carefully. " + " ".join(words) + "\n"
            completion_ids += [THINK_START_ID] + self.tokenizer.encode(reasoning) + [THINK_END_ID]
            content = "\n\n" + content
        answer_ids = self.tokenizer.encode(content)
        completion_ids += answer_ids
        finish_reason = "stop"
        if len(completion_ids) + 1 > max_tokens:
            completion_ids = completion_ids[:max_tokens]
            finish_reason = "length"
            reasoning, content = self._split_truncated(completion_ids, enable_thinking)
        else:
            completion_ids.append(IM_END_ID)
        if self.finish_reason_override is not None:
            finish_reason = self.finish_reason_override

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "refusal": None,
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": [],
            "reasoning": reasoning,
        }
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                    "stop_reason": None,
                    "token_ids": completion_ids,
                    "routed_experts": None,
                }
            ],
            "service_tier": None,
            "system_fingerprint": None,
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "total_tokens": len(prompt_ids) + len(completion_ids),
                "completion_tokens": len(completion_ids),
                "prompt_tokens_details": None,
            },
            "prompt_logprobs": None,
            "prompt_token_ids": prompt_ids,
            "prompt_text": None,
            "prompt_routed_experts": None,
            "kv_transfer_params": None,
        }

    def _split_truncated(self, ids: list[int], enable_thinking: bool) -> tuple[str | None, str | None]:
        """Reasoning/content of a ``length``-truncated completion, as the qwen3 parser would split it."""
        if not enable_thinking:
            return None, self.tokenizer.decode(ids)
        if THINK_END_ID in ids:
            cut = ids.index(THINK_END_ID)
            reasoning = self.tokenizer.decode([i for i in ids[:cut] if i != THINK_START_ID])
            rest = ids[cut + 1 :]
            return reasoning, ("\n\n" + self.tokenizer.decode(rest)) if rest else None
        return self.tokenizer.decode([i for i in ids if i != THINK_START_ID]), None

    # ---- content generators ----------------------------------------------------
    def _content_for(self, mode: str, messages: Sequence[Mapping[str, Any]], key: str) -> str:
        if mode == "hub":
            return self._hub_content(messages, key)
        if mode == "worker":
            return self._worker_content(messages, key)
        if mode == "judge_best":
            return _dumps(
                {
                    "quality_score": round(_unit(key, "quality"), 3),
                    "requirement_coverage": f"coverage {key[:6]}",
                    "reasoning_support": f"support {key[6:12]}",
                    "unresolved_risks": f"risks {key[12:18]}",
                }
            )
        if mode == "hle_judge":
            if self.hle_judge_mode == "deterministic":
                correct = "yes" if _unit(key, "judge") < 0.5 else "no"
            else:
                correct = self.hle_judge_mode
            return _dumps(
                {
                    "extracted_final_answer": self.answer_pool[int(_unit(key, "extract") * len(self.answer_pool))],
                    "reasoning": f"fake judge reasoning {key[:8]}",
                    "correct": correct,
                    "confidence": int(50 + 50 * _unit(key, "jconf")),
                }
            )
        if mode == "forecast":
            return _dumps(
                {
                    "q_personal": round(_unit(key, "qp"), 3),
                    "q_child_contract": None,
                    "q_team_now": round(_unit(key, "qt"), 3),
                    "q_recover": round(_unit(key, "qr"), 3),
                    "q_preserve": round(_unit(key, "qv"), 3),
                }
            )
        if _unit(key, "invalid") < self.invalid_rate:
            return f'{{"approach": "broken output {key[:8]}", "final_answer": '  # unterminated
        return _dumps(self.candidate(messages, key))

    def candidate(self, messages: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
        """A valid §3.5 candidate; ``final_answer`` drawn from the per-item pool."""
        item = item_key(messages)
        rotation = int(item[:8], 16) % len(self.answer_pool)
        pool = self.answer_pool[rotation:] + self.answer_pool[:rotation]
        draw = _unit(key, "answer")
        cumulative = 0.0
        chosen = pool[-1]
        weights = list(_POOL_WEIGHTS) + [0.0] * max(0, len(pool) - len(_POOL_WEIGHTS))
        total = sum(weights[: len(pool)]) or 1.0
        for answer, weight in zip(pool, weights):
            cumulative += weight / total
            if draw < cumulative:
                chosen = answer
                break
        return {
            "approach": f"Fake approach {key[:8]} for item {item[:8]}.",
            "evidence": [
                {"claim": f"claim {key[8:14]}", "support": f"support {key[14:20]}", "uncertainty": "low"}
            ],
            "alternatives_considered": [f"alternative {key[20:26]}"],
            "failure_checks": [f"checked {key[26:32]}"],
            "final_answer": chosen,
            "confidence": round(0.5 + 0.45 * _unit(key, "confidence"), 3),
        }

    _WORKERS_RE = re.compile(r"(\d+)\s+available worker slots")
    _SUBTASK_RE = re.compile(r'"subtask_id"\s*:\s*"([^"]+)"')

    def _hub_content(self, messages: Sequence[Mapping[str, Any]], key: str) -> str:
        text = _prompt_text(messages)
        mode = self.hub_mode(text) if callable(self.hub_mode) else self.hub_mode
        if mode == "alternate":
            item = item_key(messages)
            with self._lock:
                n = self._hub_calls_per_item.get(item, 0)
                self._hub_calls_per_item[item] = n + 1
            mode = "delegate" if n % 2 == 0 else "final"
        match = self._WORKERS_RE.search(text)
        n_workers = int(match.group(1)) if match else self.delegate_count
        if mode == "invalid":
            return _dumps({"action": "plan", "steps": [f"plan {key[:8]}"]})
        if mode == "delegate" and n_workers > 0:
            k = max(1, min(self.delegate_count, n_workers))
            return _dumps(
                {
                    "action": "delegate",
                    "assignments": [
                        {
                            "worker_slot": slot,
                            "subtask_id": f"sub-{key[:6]}-{slot}",
                            "question": f"Check part {slot} of the task ({key[:8]}).",
                            "source_handles": ["task"],
                            "required_output_type": "text",
                            "return_contract": "Return the checked value with assumptions.",
                        }
                        for slot in range(1, k + 1)
                    ],
                }
            )
        if mode not in ("final", "delegate"):
            raise ValueError(f"unknown hub_mode {mode!r}")
        return _dumps({"action": "final", "candidate": self.candidate(messages, key)})

    def _worker_content(self, messages: Sequence[Mapping[str, Any]], key: str) -> str:
        if self.worker_mode == "invalid":
            return _dumps({"subtask_id": "x", "status": "done"})
        match = self._SUBTASK_RE.search(_prompt_text(messages))
        return _dumps(
            {
                "subtask_id": match.group(1) if match else f"sub-{key[:6]}",
                "contract": f"contract {key[:8]}",
                "status": self.worker_mode,
                "result": f"Worker result {key[8:16]}: the checked value is {self.answer_pool[0]}.",
                "assumptions": [f"assumption {key[16:22]}"],
                "evidence_handles": ["task"],
                "confidence": None if self.worker_mode == "failed" else round(_unit(key, "wconf"), 3),
            }
        )


# --------------------------------------------------------------------------- registry


def register_fake(run_root: str | Path, profile: str, host: str, port: int, **overrides: Any) -> Path:
    """Write ``<run_root>/servers/<profile>/<host>_<port>.json`` for a fake endpoint.

    The entry carries the real serving profile's ``model_size``/``hf_id``/``served_model_name``/
    ``max_model_len``/``tp_size`` so ``registry.entry_matches_profile`` accepts it, and
    ``slurm_job_id=None`` so ``registry.list_live_servers`` probes ``GET /health`` instead of
    asking Slurm (architecture §1 "registry layout").
    """
    from agents_scaling.serving.profiles import get_serving_profile

    spec = get_serving_profile(profile)
    entry: dict[str, Any] = {
        "model_size": spec.model_size,
        "hf_id": spec.hf_id,
        "host": host,
        "port": int(port),
        "slurm_job_id": None,
        "started_at": time.time(),
        "serving_profile": spec.name,
        "served_model_name": spec.served_model_name,
        "max_model_len": spec.max_model_len,
        "tp_size": spec.tp_size,
    }
    entry.update(overrides)
    path = Path(run_root) / "servers" / profile / f"{host}_{port}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def unregister_fake(run_root: str | Path, profile: str, host: str, port: int) -> None:
    path = Path(run_root) / "servers" / profile / f"{host}_{port}.json"
    if path.exists():
        path.unlink()


__all__ = [
    "FakeTokenizer",
    "FakeVllmServer",
    "DEFAULT_ANSWER_POOL",
    "THINK_START_ID",
    "THINK_END_ID",
    "IM_END_ID",
    "detect_mode",
    "item_key",
    "register_fake",
    "request_key",
    "unregister_fake",
]
