"""WP-A, contract §2 and §5: the internal message model and the run/call records, with ``to_json``/``from_json``.

Every dataclass here crosses a module boundary (WP-B converts wire bodies to ``ChatRequest``/``ChatReply``; WP-C reads
``RunRecord``/``CallRecord`` from the run store). Records are encoded with the study's one codec
(``contracts.dump_record`` / ``contracts.load_record``: strict, every field present, enums by value, ``Decimal`` as
text) so writer/reader key drift raises instead of defaulting.

``ProxyError`` is the only exception the wire layer converts (§2): ``status`` is the HTTP status it maps to and
``code`` the machine-readable reason (``unknown_run``, ``bad_bearer``, ``run_closed``, ``budget_exhausted``,
``call_limit``, ``input_limit``, ``tool_surface_violation``, ``bad_request``, ``ledger_refused``,
``upstream_failed``).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping

from agent_allocation.contracts import TransportStatus, Usage, dump_record, load_record

CONVERSATION_CLASSES: tuple[str, ...] = ("root", "solver", "other")
FINISH_REASONS_RETURNED: frozenset[str] = frozenset({"stop", "tool_calls"})
FORCED_CLOSE_KINDS: tuple[str, ...] = ("none", "mid_call", "visible_continuation", "terminal", "truncated_subagent",
                                       "ceiling_close")
REINJECTION_STATES: tuple[str, ...] = ("hit", "miss", "explicit", "n/a")
RUN_STATUSES: tuple[str, ...] = ("planned", "running", "done", "failed", "timeout", "transport_failed",
                                 "refused_before_start")
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"done", "failed", "timeout", "transport_failed",
                                                   "refused_before_start"})

# code -> (HTTP status, retryable for the harness)
PROXY_ERROR_CODES: Mapping[str, tuple[int, bool]] = {
    "unknown_run": (404, False),
    "bad_bearer": (401, False),
    "run_closed": (409, False),
    "budget_exhausted": (402, False),
    "call_limit": (402, False),
    "input_limit": (402, False),
    "tool_surface_violation": (400, False),
    "bad_request": (400, False),
    "ledger_refused": (503, True),
    "upstream_failed": (502, False),
    # wire-layer codes (WP-B httpd): an unknown route, and a proxy bug answered loudly instead of hanging
    "not_found": (404, False),
    "internal_error": (500, False),
}


class ProxyError(Exception):
    """§2: the one exception the wire layer converts into an HTTP error envelope.

    Contract signature ``ProxyError(status, code, message, retryable)``: ``code`` must be one of
    ``PROXY_ERROR_CODES`` (an unknown code is a programming error and raises ``ValueError``); ``status`` is the HTTP
    status the wire layer answers with (normally the code's frozen status; a caller may pass another, e.g. 405 for a
    wrong method on a ``bad_request``); ``retryable`` defaults to the code's frozen value when ``None``. ``detail``
    is additive (the ledger's refusal reason / the upstream detail, journaled but never sent to the harness).
    """

    def __init__(self, status: int, code: str, message: str, retryable: bool | None = None, *,
                 detail: str | None = None) -> None:
        if code not in PROXY_ERROR_CODES:
            raise ValueError(f"unknown ProxyError code {code!r}")
        if isinstance(status, bool) or not isinstance(status, int) or not 400 <= status <= 599:
            raise ValueError(f"ProxyError status must be an HTTP error status (400-599), got {status!r}")
        _default_status, default_retryable = PROXY_ERROR_CODES[code]
        self.code = code
        self.message = message
        self.status = status
        self.retryable = default_retryable if retryable is None else bool(retryable)
        self.detail = detail
        super().__init__(f"{self.status} {code}: {message}")

    @classmethod
    def of(cls, code: str, message: str, *, retryable: bool | None = None, detail: str | None = None) -> "ProxyError":
        """The error for ``code`` at its frozen status (``PROXY_ERROR_CODES``)."""
        if code not in PROXY_ERROR_CODES:
            raise ValueError(f"unknown ProxyError code {code!r}")
        return cls(PROXY_ERROR_CODES[code][0], code, message, retryable, detail=detail)


# ---------------------------------------------------------------------------------------------------------------------
# §2 internal message model
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """One OpenAI function tool, already flattened (Responses namespaces removed, kept on ``namespace``)."""
    name: str
    description: str
    parameters: dict[str, Any]
    namespace: str | None = None

    def as_openai(self) -> dict[str, Any]:
        """The ``{"type":"function","function":{...}}`` object the template ``tojson``s verbatim (§4)."""
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | str          # dict when JSON-parseable, else the raw string


@dataclass(frozen=True)
class Message:
    """roles: system | user | assistant | tool."""
    role: str
    content: str
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    requested_max_tokens: int | None
    requested_reasoning_effort: str | None
    stream: bool
    ignored_keys: tuple[str, ...]
    wire: str                                # "chat" | "responses"
    # Additive (defaulted, so the frozen §2 constructor keeps working): set by the wire layer when the harness marked
    # the request as a sub-agent call (Codex sends ``x-openai-subagent``); a Codex V1 fork shares the root's system
    # text and first user text, so nothing else distinguishes it (docs/dynorch/requests/wpA.md).
    subagent: bool = False


@dataclass(frozen=True)
class ReplyUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cached_prompt_tokens: int | None


# ---------------------------------------------------------------------------------------------------------------------
# §5 run and call records
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    run_key: str
    arm: str
    harness: str
    item_id: str
    benchmark: str
    stratum: str
    shard_id: str
    endpoint_tag: str
    rung: int
    cap_tokens: int
    repeat: int
    attempt: int
    generation: int
    partition: str                           # canary|pilot|main|ext_e0...
    allowed_tools: tuple[str, ...]
    spawn_tools: tuple[str, ...]
    root_call_ceiling: int | None
    sampling_hash: str
    template_sha256: str
    prompt_version: str
    proxy_version: str
    harness_version: str

    @property
    def item_slug(self) -> str:
        return item_slug(self.item_id)


def item_slug(item_id: str) -> str:
    """``item_id`` as one path component (``/`` -> ``__``), the journal and run-store layout of §5."""
    return item_id.replace("/", "__")


@dataclass
class CallRecord:
    call_no: int
    conversation_key: str
    conversation_class: str                  # root|solver|other
    requested_max_tokens: int | None
    granted_max_tokens: int
    queue_wait_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cached_prompt_tokens: int | None
    reasoning_chars: int
    visible_chars: int
    forced_close_kind: str
    grace_tokens: int
    finish_reason_upstream: str | None
    finish_reason_returned: str
    tool_calls: int
    spawn_calls: int
    tool_parse: str
    reinjection: str                         # hit|miss|explicit|n/a
    reasoning_effort_requested: str | None
    cost_usd: str
    generation_ids: tuple[str, ...]
    latency_seconds: float
    transport_status: str
    returned_status: int
    stream: bool
    wire: str
    cache_status: str | None
    duplicate_served: bool
    ignored_body_keys: tuple[str, ...]


@dataclass(frozen=True)
class ChatReply:
    content: str
    reasoning_content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str                       # "stop" | "tool_calls" ONLY
    usage: ReplyUsage
    call: CallRecord
    headers: dict[str, str]                  # X-Dynorch-* headers

    def __post_init__(self) -> None:
        if self.finish_reason not in FINISH_REASONS_RETURNED:
            raise ValueError(f"finish_reason {self.finish_reason!r} is outside {sorted(FINISH_REASONS_RETURNED)}")


@dataclass
class RunRecord:
    spec: RunSpec
    status: str                              # planned|running|done|failed|timeout|transport_failed|refused_before_start
    calls: int
    conversations: int
    spawns: int
    calls_by_class: dict[str, int]
    completion_total: int
    reasoning_total: int | None
    prompt_total: int
    grace_tokens: int
    overdraft_tokens: int
    forced_close_kind: str
    exhausted: bool
    refused_calls: int
    guard_triggered: bool
    tool_parse_failures: int
    reinjection_hits: int
    reinjection_misses: int
    duplicates_served: int
    final_message: str | None
    final_source: str                        # proxy_root|none
    cli_result: str | None
    cli_result_mismatch: bool | None
    parse_status: str | None
    final_answer: str | None
    grade_status: str | None
    correct: bool | None
    grade_detail: str | None
    wall_seconds: float
    harness_exit_code: int | None
    harness_reported: dict[str, Any]
    cost_usd: str
    started_at: str
    finished_at: str | None
    error: str | None
    rerun_reason: str | None
    calls_detail: list[CallRecord] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_RUN_STATUSES


def new_run_record(spec: RunSpec, *, status: str, started_at: str) -> RunRecord:
    """A fresh record with every counter at zero (``planned`` before launch, ``running`` at ``open_run``)."""
    if status not in RUN_STATUSES:
        raise ValueError(f"unknown run status {status!r}")
    return RunRecord(
        spec=spec, status=status, calls=0, conversations=0, spawns=0,
        calls_by_class={name: 0 for name in CONVERSATION_CLASSES}, completion_total=0, reasoning_total=0,
        prompt_total=0, grace_tokens=0, overdraft_tokens=0, forced_close_kind="none", exhausted=False,
        refused_calls=0, guard_triggered=False, tool_parse_failures=0, reinjection_hits=0, reinjection_misses=0,
        duplicates_served=0, final_message=None, final_source="none", cli_result=None, cli_result_mismatch=None,
        parse_status=None, final_answer=None, grade_status=None, correct=None, grade_detail=None, wall_seconds=0.0,
        harness_exit_code=None, harness_reported={}, cost_usd="0", started_at=started_at, finished_at=None,
        error=None, rerun_reason=None, calls_detail=[])


# ---------------------------------------------------------------------------------------------------------------------
# §5 upstream result
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class UpstreamResult:
    """One settled upstream exchange (``Upstream.complete``): the raw choice fields plus the classification."""
    reasoning: str
    text: str
    closed: bool
    finish_reason: str | None
    usage: Usage | None
    generation_id: str | None
    cache_status: str | None
    latency_seconds: float
    transport_status: TransportStatus
    http_status: int | None
    raw: dict
    # additive: what the ledger settled for this exchange (sum over transport attempts), and the journal phases
    cost_usd: str = "0"
    attempts: tuple[str, ...] = ()
    phases: tuple[dict[str, Any], ...] = ()
    request_body: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------------------------------------------------

def to_json(obj: Any) -> dict[str, Any]:
    """``contracts.dump_record`` for any record above (tagged with ``__record__``)."""
    if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)):
        raise TypeError(f"to_json expects a dynorch record, got {type(obj).__name__}")
    return dump_record(obj)


def from_json(cls: type, data: Mapping[str, Any]) -> Any:
    """Strict inverse of ``to_json``: every field must be present, unknown fields raise."""
    return load_record(cls, data)
